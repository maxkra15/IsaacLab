# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare Torch and experimental Warp FlashSAC on the same manager Cartpole MDP.

Run ``uv run --no-sync python scripts/benchmarks/compare_flashsac.py run --output logs/flashsac-cartpole``.
The separate ``smoke`` phase exercises Warp learning, checkpoint loading and evaluation.
Compiler caches belong outside the artifact directory. Each recipe trains and evaluates in
fresh processes; common CPU initial weights are generated before measured training runs.
FlashSAC: https://arxiv.org/abs/2604.04539; authors' source: https://github.com/Holiday-Robot/FlashSAC.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
import weakref
from importlib.metadata import version
from pathlib import Path

PROCESS_STARTED = time.perf_counter()
RECIPES = ("torch-fp32-eager", "torch-fp32-compiled", "torch-fp16-compiled", "warp-fp32-captured")
ROLES = ("actor", "critic", "target_critic", "temperature")


def write_json(path: Path, value: dict) -> None:
    """Atomically preserve a readable, finite JSON result."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def protocol(smoke: bool) -> dict:
    """Return the declared budget; smoke results never enter the full comparison."""
    return {
        "num_envs": 64 if smoke else 1024,
        "horizon": 24,
        "iterations": 8 if smoke else 200,
        "eval_envs": 128,
        "eval_seed": 10000,
        "checkpoints": [0, 8] if smoke else [0, 50, 100, 150, 200],
        "buffer_max_length": 131072,
        "buffer_min_length": 2048 if smoke else 100000,
        "sample_batch_size": 256 if smoke else 2048,
        "learning_rate_decay_step": 9600,
        "updates_per_step": 2,
    }


def runner_config(recipe: str, seed: int, specification: dict, device: str) -> dict:
    """Use the maintained runner and shared authors' configuration."""
    from isaaclab_rl.robolearn import RoboLearnRunnerCfg

    return RoboLearnRunnerCfg(
        algorithm="warp_flashsac" if recipe.startswith("warp") else "flashsac",
        seed=seed,
        device=device,
        num_steps_per_env=specification["horizon"],
        max_iterations=specification["iterations"],
        save_interval=specification["checkpoints"][-1] if specification["iterations"] == 8 else 50,
        experiment_name="flashsac-cartpole",
        run_name=f"{recipe}-seed{seed}",
        updates_per_step=specification["updates_per_step"],
        flash_updates_during_rollout=True,
        init_at_random_ep_len=False,
        clip_actions=1.0,
        capture_updates=recipe.startswith("warp"),
        algorithm_cfg={
            **{
                key: specification[key]
                for key in ("buffer_max_length", "buffer_min_length", "sample_batch_size", "learning_rate_decay_step")
            },
            "actor_hidden_dim": 128,
            "critic_hidden_dim": 256,
            "actor_num_blocks": 2,
            "critic_num_blocks": 2,
            "critic_num_bins": 101,
            "actor_update_period": 2,
            "n_step": 3,
            "normalize_reward": True,
            "use_compile": recipe in ("torch-fp32-compiled", "torch-fp16-compiled"),
            "compile_mode": "auto",
            "use_amp": recipe == "torch-fp16-compiled",
        },
    ).to_dict()


def fixture(args: argparse.Namespace) -> None:
    """Export authors' CPU initialization once per seed, including BN and target state."""
    import numpy as np
    import torch
    from robolearn.flashsac.agent import _init_flashsac_networks
    from robolearn.flashsac.config import FlashSACConfig

    started = time.perf_counter()
    torch.manual_seed(args.worker_seed)
    networks = _init_flashsac_networks(4, 4, 1, FlashSACConfig(device="cpu"), torch.device("cpu"))
    arrays, parameter_names = {}, {}
    for role, network in zip(ROLES, networks, strict=True):
        arrays.update(
            {f"{role}.{name}": value.detach().numpy() for name, value in network._raw_network.state_dict().items()}
        )
        parameter_names[role] = [name for name, _ in network._raw_network.named_parameters()]
    np.savez(args.fixture, **arrays)
    write_json(
        args.fixture.with_suffix(".json"),
        {
            "seed": args.worker_seed,
            "device": "cpu",
            "parameter_names": parameter_names,
            "sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
            "fixture_seconds": time.perf_counter() - started,
            "process_seconds_before_exit": time.perf_counter() - PROCESS_STARTED,
        },
    )


def load_fixture(runner, path: Path) -> None:
    """Load exactly the same actor, critics, BN buffers and log temperature."""
    import numpy as np
    import torch
    import warp as wp

    with np.load(path) as arrays:
        states = {
            role: {key.removeprefix(role + "."): arrays[key] for key in arrays.files if key.startswith(role + ".")}
            for role in ROLES
        }
        if runner._is_warp_flash:
            with wp.ScopedStream(runner._warp_stream, sync_enter=False):
                for role in ROLES[:3]:
                    getattr(runner.agent, role).load_author_state_dict(states[role])
                runner.agent.log_temperature.assign(states["temperature"]["log_temp"])
        else:
            for role in ROLES:
                getattr(runner.agent, "_" + role)._raw_network.load_state_dict(
                    {key: torch.from_numpy(value).to(runner.device) for key, value in states[role].items()}
                )
    torch.cuda.synchronize(runner.device)


def checkpoint_evidence(
    directory: Path, warp: bool, attempted: dict, parameter_names: dict, *, allow_amp: bool = False
) -> tuple[dict, dict]:
    """Audit saved finite state and completed Adam steps, including AMP skipped steps."""
    import numpy as np
    import torch

    finite_weights, finite_optimizer, changed, optimizer = True, True, {}, {}
    if warp:
        with np.load(directory / "initial/agent.npz") as initial, np.load(directory / "final/agent.npz") as final:
            finite_weights = all(
                np.isfinite(final[key]).all() for key in final.files if key.startswith((*ROLES[:3], "log_temperature"))
            )
            finite_optimizer = all(np.isfinite(final[key]).all() for key in final.files if key.startswith("optimizer."))
            for role in ("actor", "critic"):
                changed[role] = any(
                    not np.array_equal(initial[f"{role}.{key}"], final[f"{role}.{key}"])
                    for key in parameter_names[role]
                )
            steps = {role: int(final[f"optimizer.{role}.timestep"][0]) for role in ("actor", "critic", "temperature")}
            if list(final["update_counters"]) != [steps["critic"], steps["actor"], steps["temperature"]]:
                raise RuntimeError("Warp device update counters disagree with Adam steps.")
            for role, count in steps.items():
                optimizer[role] = {"completed": count, "min": count, "max": count, "coverage": 1.0}
    else:
        for role in ROLES:
            initial = torch.load(directory / "initial" / f"{role}.pt", map_location="cpu", weights_only=True)
            final = torch.load(directory / "final" / f"{role}.pt", map_location="cpu", weights_only=True)
            finite_weights &= all(torch.isfinite(value).all().item() for value in final["network_state_dict"].values())
            if role in ("actor", "critic"):
                changed[role] = any(
                    not torch.equal(initial["network_state_dict"][key], final["network_state_dict"][key])
                    for key in parameter_names[role]
                )
            if role == "target_critic":
                continue
            state = final["optimizer_state_dict"]
            parameters = [parameter for group in state["param_groups"] for parameter in group["params"]]
            counts = [float(state["state"].get(parameter, {}).get("step", 0)) for parameter in parameters]
            finite_optimizer &= all(
                torch.isfinite(value).all().item()
                for values in state["state"].values()
                for value in values.values()
                if isinstance(value, torch.Tensor)
            )
            if any(not math.isfinite(count) or count != int(count) for count in counts) or min(counts) != max(counts):
                raise RuntimeError(f"Invalid or inconsistent completed {role} optimizer steps: {counts}.")
            optimizer[role] = {
                "completed": int(min(counts)),
                "min": int(min(counts)),
                "max": int(max(counts)),
                "coverage": sum(parameter in state["state"] for parameter in parameters) / len(parameters),
            }
    for role, evidence in optimizer.items():
        evidence.update(attempted=attempted[role], amp_skipped=attempted[role] - evidence["completed"])
        if evidence["amp_skipped"] < 0 or (evidence["amp_skipped"] != 0 and (not allow_amp or role == "temperature")):
            raise RuntimeError("Completed optimizer counts do not match the declared update schedule.")
    health = {
        "finite_weights": bool(finite_weights),
        "finite_optimizer": bool(finite_optimizer),
        "actor_parameters_changed": changed["actor"],
        "critic_parameters_changed": changed["critic"],
        "nonfinite_actions": 0,
    }
    if not all(health[key] for key in health if key != "nonfinite_actions"):
        raise RuntimeError(f"Invalid learned checkpoint: {health}.")
    return health, optimizer


def cleanup(runner, env) -> None:
    """Release benchmark references while the simulator CUDA context is alive."""
    import torch
    import warp as wp

    device = str(runner.device) if runner is not None else str(env.device)
    wp.get_device(device).make_current()
    torch.cuda.synchronize(device)
    wp.synchronize_device(device)
    # Callers clear their own runner and policy references before closing the environment.
    gc.collect()


def train_worker(args: argparse.Namespace) -> None:
    """Instrument the native runner without changing its algorithm or update order."""
    import gymnasium as gym
    import torch
    import warp as wp

    from isaaclab.app import launch_simulation
    from isaaclab.utils.io import dump_yaml

    from isaaclab_rl.robolearn import RoboLearnRunner

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    class TimedRunner(RoboLearnRunner):
        def __init__(self, *values, **kwargs):
            super().__init__(*values, **kwargs)
            self.phase_times = {
                "prepare_seconds": 0.0,
                "checkpoint_seconds": 0.0,
                "first_policy_call_seconds": None,
                "first_update_group_seconds": None,
            }
            self.bad_transitions = torch.zeros((), dtype=torch.bool, device=self.device)
            runner_ref, agent_ref, act = weakref.ref(self), weakref.ref(self.agent), type(self.agent).act

            def timed_act(*values, **kwargs):
                owner = runner_ref()
                agent = agent_ref()
                first = (
                    owner.phase_times["first_policy_call_seconds"] is None
                    and agent.ready
                    and kwargs.get("training", True)
                )
                if first:
                    owner._synchronize()
                    started = time.perf_counter()
                result = act(agent, *values, **kwargs)
                if first:
                    owner._synchronize()
                    owner.phase_times["first_policy_call_seconds"] = time.perf_counter() - started
                return result

            self.agent.act = timed_act

        def _prepare_learning(self):
            started = time.perf_counter()
            super()._prepare_learning()
            self.phase_times["prepare_seconds"] += time.perf_counter() - started

        def save(self, path):
            self._synchronize()
            started = time.perf_counter()
            super().save(path)
            self._synchronize()
            self.phase_times["checkpoint_seconds"] += time.perf_counter() - started

        def _flash_update_group(self, step):
            first = self.phase_times["first_update_group_seconds"] is None
            if first:
                self._synchronize()
                started = time.perf_counter()
            super()._flash_update_group(step)
            if first:
                self._synchronize()
                self.phase_times["first_update_group_seconds"] = time.perf_counter() - started

        def _collect_step(self, observations, step):
            observations, transition = super()._collect_step(observations, step)
            for key in ("action", "reward", "next_observation"):
                self.bad_transitions.logical_or_(~torch.isfinite(transition[key]).all())
            return observations, transition

        def _update(self):
            losses, updates, actors = super()._update()
            if self.bad_transitions.item() or any(not math.isfinite(value) for value in losses.values()):
                raise RuntimeError("Nonfinite actions, transitions or learning metrics.")
            return losses, updates, actors

    specification = protocol(args.smoke_worker)
    configuration = runner_config(args.worker_recipe, args.worker_seed, specification, args.device)
    env_cfg = parse_env_cfg(
        "Isaac-Cartpole", device=args.device, num_envs=specification["num_envs"], overrides=["physics=newton_mjwarp"]
    )
    env_cfg.compute_final_obs = True
    env_cfg.seed = args.worker_seed
    dump_yaml(str(args.output / "env.yaml"), env_cfg)
    runner, failure = None, None
    with launch_simulation(env_cfg):
        env = gym.make("Isaac-Cartpole", cfg=env_cfg).unwrapped
        try:
            environment_startup = time.perf_counter() - PROCESS_STARTED
            started = time.perf_counter()
            runner = TimedRunner(env, configuration, log_dir=str(args.output), device=args.device)
            torch.cuda.synchronize(args.device)
            constructor = time.perf_counter() - started
            started = time.perf_counter()
            load_fixture(runner, args.fixture)
            fixture_load = time.perf_counter() - started
            started = time.perf_counter()
            runner.learn(specification["iterations"], init_at_random_ep_len=False)
            learn_seconds = time.perf_counter() - started
            runner.save(str(args.output / "final.json"))
            rows = [json.loads(line) for line in (args.output / "metrics.jsonl").read_text().splitlines()]
            if len(rows) != specification["iterations"] or rows[-1]["total_steps"] != math.prod(
                specification[key] for key in ("num_envs", "horizon", "iterations")
            ):
                raise RuntimeError("Actual transitions or iterations do not match the declared budget.")
            active_rows = [row for row in rows if row["updates_this_iteration"]]
            first_steady = active_rows[min(5, len(active_rows) - 1)]["iteration"]
            steady = [row for row in rows if row["iteration"] >= first_steady]
            attempted = {
                "actor": runner.actor_gradient_updates,
                "critic": runner.critic_gradient_updates,
                "temperature": runner.actor_gradient_updates,
            }
            expected = 2 * max(
                0,
                specification["horizon"] * specification["iterations"]
                - math.ceil(specification["buffer_min_length"] / specification["num_envs"])
                - (3 - 1)
                + 1,
            )
            if attempted["critic"] != expected or attempted["actor"] != expected // 2:
                raise RuntimeError(f"Update budget mismatch: {attempted}, expected {expected} critic updates.")
            fixture_metadata = json.loads(args.fixture.with_suffix(".json").read_text())
            parameter_names = fixture_metadata["parameter_names"]
            if runner._is_warp_flash:
                parameter_names = {
                    role: [name for name, _ in getattr(runner.agent, role).named_parameters()]
                    for role in ("actor", "critic")
                }
            health, optimizer = checkpoint_evidence(
                args.output,
                runner._is_warp_flash,
                attempted,
                parameter_names,
                allow_amp=configuration["algorithm_cfg"]["use_amp"],
            )
            timing = {
                "environment_startup_seconds": environment_startup,
                "constructor_seconds": constructor,
                "fixture_load_seconds": fixture_load,
                "learn_seconds": learn_seconds,
                **runner.phase_times,
                "steady_from_iteration": first_steady,
            }
            for prefix, selected in (("", rows), ("steady_", steady)):
                for name, field in (("loop", "iteration"), ("collection", "rollout"), ("update", "update")):
                    timing[prefix + name + "_seconds"] = sum(row[field + "_seconds"] for row in selected)
            timing["steady_transitions"] = len(steady) * specification["num_envs"] * specification["horizon"]
            timing["steady_fps"] = timing["steady_transitions"] / timing["steady_loop_seconds"]
            timing["startup_seconds"] = environment_startup + constructor + fixture_load + timing["prepare_seconds"]
            timing["learn_overhead_seconds"] = learn_seconds - timing["loop_seconds"] - timing["prepare_seconds"]
            write_json(
                args.output / "run.json",
                {
                    "recipe": args.worker_recipe,
                    "seed": args.worker_seed,
                    "algorithm": configuration["algorithm"],
                    "agent_config": configuration,
                    "protocol": specification,
                    "log_dir": str(args.output),
                    "fixture_sha256": fixture_metadata["sha256"],
                    "transitions": runner.total_steps,
                    "iterations": runner.current_learning_iteration,
                    "timing": timing,
                    "health": health,
                    "optimizer": optimizer,
                    "training_curve": rows,
                    "evaluations": [],
                    "gpu_isolation": os.environ.get("FLASHSAC_GPU_ISOLATION", "unverified"),
                    "hardware": {
                        "gpu": torch.cuda.get_device_name(args.device),
                        "total_memory_bytes": torch.cuda.get_device_properties(args.device).total_memory,
                        "compute_capability": list(torch.cuda.get_device_capability(args.device)),
                    },
                },
            )
        except Exception as error:
            failure = traceback.format_exc()
            cleanup(runner, env)
            traceback.clear_frames(error.__traceback__)
            error.__traceback__ = None
        finally:
            cleanup(runner, env)
            runner = None
            gc.collect()
            wp.get_device(args.device).make_current()
            env.close()
    if failure is not None:
        raise RuntimeError("Training failed before simulator cleanup:\n" + failure) from None


def evaluate_worker(args: argparse.Namespace) -> None:
    """Measure first episodes only, preserving terminal observations before automatic resets."""
    import gymnasium as gym
    import numpy as np
    import torch
    import warp as wp

    from isaaclab.app import launch_simulation

    from isaaclab_rl.robolearn import RoboLearnRunner

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    run = json.loads((args.output / "run.json").read_text())
    specification = run["protocol"]
    env_cfg = parse_env_cfg(
        "Isaac-Cartpole", device=args.device, num_envs=specification["eval_envs"], overrides=["physics=newton_mjwarp"]
    )
    env_cfg.seed = specification["eval_seed"]
    env_cfg.compute_final_obs = True

    def evaluate_episodes(env, runner):
        # The caller retains learner streams until every scratch tensor leaves this frame.
        evaluations = []
        robot = env.scene["robot"]
        primer = {"seed": specification["eval_seed"], "unscored_native_resets": 1}
        for stage in ("before", "after"):
            if stage == "after":
                # Native initialization populates soft velocity limits during the first reset.
                env.reset(seed=specification["eval_seed"])
            limits = robot.data.soft_joint_vel_limits.torch.detach().cpu().numpy()
            primer[f"soft_joint_velocity_limits_{stage}"] = {
                "sha256": hashlib.sha256(limits.tobytes()).hexdigest(),
                "shape": list(limits.shape),
                "per_joint_min": limits.min(axis=0).tolist(),
                "per_joint_max": limits.max(axis=0).tolist(),
            }
        cart = robot.find_joints("slider_to_cart")[0][0]
        pole = robot.find_joints("cart_to_pole")[0][0]
        for iteration in specification["checkpoints"]:
            checkpoint = args.output / ("initial.json" if iteration == 0 else f"model_{iteration}.json")
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy()
            observations, _ = env.reset(seed=specification["eval_seed"])
            hashes = {}
            for name, value in (
                ("joint_pos", robot.data.joint_pos.torch),
                ("joint_vel", robot.data.joint_vel.torch),
                ("root_pose", robot.data.root_link_pose_w.torch),
                ("root_velocity", robot.data.root_link_vel_w.torch),
                ("observations", observations["policy"]),
            ):
                array = value.detach().cpu().numpy()
                hashes[name] = {
                    "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                }
            active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
            totals = torch.zeros((env.num_envs, 10), device=env.device)
            finished = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.bool)
            action_max = torch.zeros((), device=env.device)
            bad_actions = torch.zeros((), device=env.device, dtype=torch.bool)
            with torch.inference_mode():
                for _ in range(env.max_episode_length):
                    actions = policy(observations)
                    bad_actions |= ~torch.isfinite(actions).all()
                    observations, rewards, terminated, truncated, extras = env.step(actions)
                    done = terminated | truncated
                    terminal = extras.get("final_obs")
                    if terminal is None:
                        if done.any().item():
                            raise RuntimeError("A completed episode has no terminal observation.")
                        state = observations["policy"]
                    else:
                        state = torch.where(done[:, None], terminal["policy"], observations["policy"])
                    raw_angle, position = state[:, pole], state[:, cart]
                    angle = torch.atan2(torch.sin(raw_angle), torch.cos(raw_angle))
                    values = torch.stack(
                        (
                            rewards,
                            torch.ones_like(rewards),
                            (angle.abs() < 0.25).float(),
                            angle.abs(),
                            angle.square(),
                            position.abs(),
                            position.square(),
                            raw_angle.square(),
                            actions.square().mean(-1),
                            (actions.abs() > 0.95).float().mean(-1),
                        ),
                        dim=-1,
                    )
                    totals += values * active[:, None]
                    action_max = torch.maximum(action_max, torch.where(active[:, None], actions.abs(), 0).max())
                    finished[:, 0] |= active & truncated & ~terminated
                    finished[:, 1] |= active & terminated
                    active &= ~done
            if bad_actions.item() or active.any().item() or not torch.isfinite(totals).all().item():
                raise RuntimeError("Evaluation has nonfinite actions/states or unfinished first episodes.")
            data = totals.cpu().numpy()
            per_episode = data[:, 2:] / data[:, 1:2]
            mean = per_episode.mean(0)
            evaluation = {
                "iteration": iteration,
                "transitions": iteration * specification["num_envs"] * specification["horizon"],
                "training_seconds": sum(row["iteration_seconds"] for row in run["training_curve"][:iteration]),
                "return_mean": float(data[:, 0].mean()),
                "return_std": float(data[:, 0].std()),
                "normalized_score": float(100 * data[:, 0].mean() / (env.max_episode_length * env.step_dt)),
                "episode_length_mean": float(data[:, 1].mean()),
                "survival_rate": float(finished[:, 0].float().mean()),
                "time_limit_fraction": float(finished[:, 0].float().mean()),
                "cart_bound_fraction": float(finished[:, 1].float().mean()),
                "upright_fraction": float(mean[0]),
                "pole_angle_abs_mean": float(mean[1]),
                "pole_angle_rms": float(np.sqrt(mean[2])),
                "cart_position_abs_mean": float(mean[3]),
                "cart_position_rms": float(np.sqrt(mean[4])),
                "unwrapped_pole_angle_squared_mean": float(mean[5]),
                "action_rms": float(np.sqrt(mean[6])),
                "action_saturation_fraction": float(mean[7]),
                "action_abs_max": float(action_max),
                "initial_state_hashes": hashes,
                "episode_count": env.num_envs,
                "max_episode_steps": env.max_episode_length,
                "step_dt": env.step_dt,
            }
            evaluations.append(evaluation)
            write_json(args.output / "evaluations.json", {"initialization": primer, "evaluations": evaluations})
        cleanup(runner, env)
        return evaluations

    runner, failure = None, None
    with launch_simulation(env_cfg):
        env = gym.make("Isaac-Cartpole", cfg=env_cfg).unwrapped
        try:
            runner = RoboLearnRunner(env, run["agent_config"], device=args.device)
            evaluate_episodes(env, runner)
        except Exception as error:
            failure = traceback.format_exc()
            cleanup(runner, env)
            # Exception tracebacks otherwise retain completed frames and their CUDA tensors.
            traceback.clear_frames(error.__traceback__)
            error.__traceback__ = None
        finally:
            cleanup(runner, env)
            runner = None
            gc.collect()
            wp.get_device(args.device).make_current()
            env.close()
    if failure is not None:
        raise RuntimeError("Evaluation failed before simulator cleanup:\n" + failure) from None


def run(args: argparse.Namespace) -> None:
    """Launch isolated workers serially and retain results after every completed phase."""
    args.output.mkdir(parents=True, exist_ok=False)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    specification = protocol(args.phase == "smoke")
    results = {
        "metadata": {
            "task": "Isaac-Cartpole",
            "physics": "newton_mjwarp",
            "mdp": "registered manager Torch MDP",
            "smoke": args.phase == "smoke",
            "protocol": specification,
            "paper": "https://arxiv.org/abs/2604.04539",
            "authors_source": "https://github.com/Holiday-Robot/FlashSAC",
            "versions": {name: version(name) for name in ("torch", "warp-lang", "warp-nn", "robolearn-rl", "newton")},
            "source": json.loads(args.source_manifest.read_text()) if args.source_manifest else {},
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "timing_definitions": {
                "training_seconds": "Cumulative synchronized collection+update loop; excludes startup and I/O.",
                "process_seconds": "Subprocess wall including imports, setup, training, final audit and shutdown.",
                "learn_seconds": "Native learn wall: initial checkpoint, preparation, loop and I/O.",
                "loop_seconds": "Native synchronized collection+update iteration wall, including first compilation.",
                "learn_overhead_seconds": "learn_seconds - loop_seconds - prepare_seconds; includes in-learn I/O.",
                "checkpoint_seconds": "All native save calls, including explicit final save outside learn_seconds.",
                "first_policy_call_seconds": "First ready training actor call, already included in loop_seconds.",
                "first_update_group_seconds": "First pair of replay updates, already included in loop_seconds.",
            },
            "notes": [
                "Same MDP, force scale 100 N, reset distributions and budgets; initial episode counters start at zero.",
                "All recipes load the same authors' CPU actor, critics, BN buffers and temperature per seed.",
                "Seeds are paired but Torch and Warp RNG streams differ; learning trajectories need not match.",
                "Warp captures learner updates; actor inference and the Torch MDP remain eager.",
                "MJWarp captures physics separately; physics and learning are not in one combined graph.",
                "Steady timings omit five replay-update iterations; cold costs and process wall remain available.",
                "Loop timings use synchronized wall and CUDA events; checkpoint I/O is outside the loop.",
                "First policy/update timings overlap loop timings and are not additive components.",
                "Cart bounds terminate this MDP. Survival alone does not prove balancing; upright threshold 0.25 rad.",
                "Normalized score = 100*raw return/(300*step_dt), upper bound 100%; raw returns are also retained.",
                "Completed Adam steps are audited separately from attempted calls. Experimental Warp is FP32 only.",
                "One unscored native reset initializes velocity-limit buffers before common scored evaluation resets.",
            ],
        },
        "fixtures": [],
        "runs": [],
    }
    manifest = args.output / "comparison.json"
    script = str(Path(__file__).resolve())

    def worker(phase: str, output: Path, seed: int, recipe: str, fixture_path: Path, log: Path) -> float:
        command = [
            sys.executable,
            script,
            phase,
            "--output",
            str(output),
            "--worker-seed",
            str(seed),
            "--worker-recipe",
            recipe,
            "--fixture",
            str(fixture_path),
            "--device",
            args.device,
            "--cache-root",
            str(args.cache_root / args.output.name / f"{recipe}-seed{seed}" / phase),
        ]
        if args.phase == "smoke":
            command.append("--smoke-worker")
        started = time.perf_counter()
        with log.open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        return time.perf_counter() - started

    try:
        fixtures = args.output / "fixtures"
        fixtures.mkdir()
        seeds = [0] if args.phase == "smoke" else args.seed
        selected = [RECIPES[-1]] if args.phase == "smoke" else args.recipe
        for seed in seeds:
            fixture_path = fixtures / f"seed{seed}.npz"
            cost = worker("fixture", fixtures, seed, "torch-fp32-eager", fixture_path, fixtures / f"seed{seed}.log")
            results["fixtures"].append(
                {**json.loads(fixture_path.with_suffix(".json").read_text()), "process_seconds": cost}
            )
            order = RECIPES[seed % len(RECIPES) :] + RECIPES[: seed % len(RECIPES)]
            for recipe in (value for value in order if value in selected):
                output = args.output / "runs" / f"{recipe}-seed{seed}"
                output.mkdir(parents=True)
                print(f"Training {recipe}, seed{seed}: {specification['iterations']} iterations", flush=True)
                elapsed = worker("train-worker", output, seed, recipe, fixture_path, output / "worker.log")
                result = json.loads((output / "run.json").read_text())
                result["process_seconds"] = elapsed
                evaluation_cost = worker("evaluate-worker", output, seed, recipe, fixture_path, output / "eval.log")
                evaluation = json.loads((output / "evaluations.json").read_text())
                result["evaluations"] = evaluation["evaluations"]
                result["evaluation_initialization"] = evaluation["initialization"]
                result["evaluation_process_seconds"] = evaluation_cost
                results["runs"].append(result)
                hashes = [
                    evaluation["initial_state_hashes"] for run in results["runs"] for evaluation in run["evaluations"]
                ]
                if any(value != hashes[0] for value in hashes):
                    raise RuntimeError("Common evaluation initial joint/root/observation states differ.")
                write_json(manifest, results)
    except Exception as error:
        results["error"] = str(error)
        write_json(manifest, results)
        raise
    print(f"Saved {manifest}", flush=True)


def main() -> None:
    """Set compiler caches before importing any learning or simulation backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "smoke", "fixture", "train-worker", "evaluate-worker"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--seed", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--recipe", nargs="+", choices=RECIPES, default=list(RECIPES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-recipe", choices=RECIPES, default=RECIPES[0], help=argparse.SUPPRESS)
    parser.add_argument("--fixture", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.cache_root = (args.cache_root or args.output.parent / (args.output.name + "-caches")).resolve()
    args.cache_root.mkdir(parents=True, exist_ok=True)
    for variable, name in (
        ("WARP_CACHE_PATH", "warp"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TRITON_CACHE_DIR", "triton"),
        ("CUDA_CACHE_PATH", "cuda"),
    ):
        os.environ[variable] = str(args.cache_root / name)
    if args.phase in ("run", "smoke"):
        run(args)
    else:
        {"fixture": fixture, "train-worker": train_worker, "evaluate-worker": evaluate_worker}[args.phase](args)


if __name__ == "__main__":
    main()
