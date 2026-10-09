# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train healthy G1 learners and score them using common native episodes.

The RSL-RL path retains the stock ``G1FlatPPORunnerCfg``, including unclipped
actions, ELU networks, scalar standard deviations and adaptive learning rates::

    uv run --no-sync python scripts/benchmarks/run_g1_learning_baselines.py train \
        --output logs/g1-stock-rsl-seed0 --algorithm rsl_rl_ppo
    uv run --no-sync python scripts/benchmarks/run_g1_learning_baselines.py evaluate \
        --output logs/g1-stock-rsl-seed0

Run one learner and seed per directory. Evaluation does not introduce an action
clip. FlashSAC's bounded policy support is an algorithm configuration choice,
while the registered environment's action, reward and reset definitions remain
unchanged. The private ``profile-rsl`` phase instruments the native entrypoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import uuid
from importlib.metadata import distribution
from pathlib import Path
from urllib.parse import unquote, urlparse

from compare_g1 import HORIZON, SCENARIOS, TASK, json_value, metadata, write_results


def profile_rsl(argv: list[str]) -> None:
    """Add action telemetry and exact-budget checkpoints to native RSL training."""
    import torch
    from profile_rsl_ppo import TimedOnPolicyRunner

    from isaaclab_rl.entrypoints.backends import train_rsl_rl

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--completed_checkpoints", default="")
    settings, native_argv = parser.parse_known_args(argv)
    checkpoints = {int(value) for value in settings.completed_checkpoints.split(",") if value}

    class BaselineOnPolicyRunner(TimedOnPolicyRunner):
        """Keep native PPO updates and log diagnostics after synchronized timings."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.logger.log_dir is not None:
                initial = self.alg.save()
                initial.update(iter=-1, infos=None)
                torch.save(initial, Path(self.logger.log_dir) / "initial.pt")

        def _write_metrics(self, information: dict) -> None:
            super()._write_metrics(information)
            if self.logger.log_dir is None:
                return
            iteration = information["it"] + 1
            actions = self.alg.storage.actions.detach()
            std = information["action_std"].detach()
            telemetry = {
                "iteration": iteration,
                "action_std_mean": float(std.mean()),
                "action_std_min": float(std.min()),
                "action_std_max": float(std.max()),
                "learning_rate": float(information["learning_rate"]),
                "action_rms": float(actions.square().mean().sqrt()),
                "action_abs_max": float(actions.abs().max()),
                "action_outside_unit_fraction": float((actions.abs() > 1.0).float().mean()),
                "wrapper_clip_actions": self.cfg["clip_actions"],
            }
            with (Path(self.logger.log_dir) / "health_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(telemetry, allow_nan=False) + "\n")
            if iteration in checkpoints:
                self.save(str(Path(self.logger.log_dir) / f"model_{iteration - 1}.pt"))

    original = train_rsl_rl.OnPolicyRunner
    train_rsl_rl.OnPolicyRunner = BaselineOnPolicyRunner
    try:
        train_rsl_rl.run(native_argv)
    finally:
        train_rsl_rl.OnPolicyRunner = original


def learning_evidence(checkpoint: Path, *, algorithm: str) -> dict:
    """Audit saved policy changes and optimizer state without changing weights."""
    if algorithm == "warp_ppo":
        import numpy as np

        saved = json.loads(checkpoint.read_text())
        network_parameters = 2 * (len(saved["config"]["algorithm_cfg"]["hidden_dims"]) + 1)
        changes = {}
        with np.load(checkpoint.with_suffix("") / "policy.npz", allow_pickle=False) as current:
            with np.load(checkpoint.parent / "initial/policy.npz", allow_pickle=False) as initial:
                if not all(np.isfinite(current[key]).all() for key in current.files):
                    raise ValueError("The Warp policy or optimizer has nonfinite saved values.")
                for name, start in (("actor", 0), ("critic", network_parameters)):
                    changes[f"{name}_maximum_weight_change"] = max(
                        float(np.abs(current[f"parameter_{index}"] - initial[f"parameter_{index}"]).max())
                        for index in range(start, start + network_parameters)
                    )
                changes["optimizer_steps"] = float(current["adam_timestep"][0])
                changes["parameter_count"] = sum(
                    current[key].size for key in current.files if key.startswith("parameter_")
                )
        changes["weights_and_optimizer_finite"] = True
        changes["weights_updated"] = all(changes[f"{name}_maximum_weight_change"] > 0 for name in ("actor", "critic"))
        return changes
    import torch

    if algorithm == "flashsac":
        changes = {"parameter_count": 0}
        # These names are buffers declared by the upstream normalization and
        # categorical value layers. Their movement is not a learned-weight proof.
        buffers = {"running_mean", "running_var", "bin_values"}
        for name in ("actor", "critic", "target_critic", "temperature"):
            current = torch.load(checkpoint.with_suffix("") / f"{name}.pt", map_location="cpu", weights_only=True)
            initial = torch.load(checkpoint.parent / "initial" / f"{name}.pt", map_location="cpu", weights_only=True)
            state = current["network_state_dict"]
            if not all(torch.isfinite(value).all() for value in state.values()):
                raise ValueError(f"The saved FlashSAC {name} has nonfinite weights or buffers.")
            parameter_keys = [key for key in state if key.rsplit(".", 1)[-1] not in buffers]
            changes[f"{name}_maximum_weight_change"] = max(
                float((state[key] - initial["network_state_dict"][key]).abs().max()) for key in parameter_keys
            )
            if name in ("actor", "critic"):
                changes["parameter_count"] += sum(state[key].numel() for key in parameter_keys)
            optimizer = current["optimizer_state_dict"]
            if optimizer is not None:
                states = optimizer["state"]
                if not states:
                    raise ValueError(f"The saved FlashSAC {name} optimizer has no update state.")
                if not all(torch.isfinite(value).all() for item in states.values() for value in item.values()):
                    raise ValueError(f"The saved FlashSAC {name} optimizer has nonfinite state.")
                counts = [float(item["step"]) for item in states.values()]
                changes[f"{name}_optimizer_steps_min"] = min(counts)
                changes[f"{name}_optimizer_steps_max"] = max(counts)
        agent_state = torch.load(checkpoint.with_suffix("") / "agent_state.pt", map_location="cpu", weights_only=True)
        changes["replay_update_calls"] = agent_state["update_step"]
        changes["amp_scaler_state"] = agent_state["grad_scaler_state_dict"]
        changes["weights_and_optimizer_finite"] = True
        changes["weights_updated"] = all(changes[f"{name}_maximum_weight_change"] > 0 for name in ("actor", "critic"))
        return changes

    current = torch.load(checkpoint, map_location="cpu", weights_only=True)
    initial = torch.load(checkpoint.parent / "initial.pt", map_location="cpu", weights_only=True)
    changes = {}
    for name in ("actor", "critic"):
        saved = current[f"{name}_state_dict"]
        if not all(torch.isfinite(value).all() for value in saved.values()):
            raise ValueError(f"The saved RSL {name} has nonfinite weights.")
        changes[f"{name}_maximum_weight_change"] = max(
            float((value - initial[f"{name}_state_dict"][key]).abs().max()) for key, value in saved.items()
        )
    optimizer = current["optimizer_state_dict"]["state"]
    if not optimizer:
        raise ValueError("The saved RSL optimizer has no update state.")
    if not all(torch.isfinite(value).all() for state in optimizer.values() for value in state.values()):
        raise ValueError("The saved RSL optimizer has nonfinite state.")
    changes["optimizer_steps"] = max(float(state["step"]) for state in optimizer.values())
    changes["weights_updated"] = all(changes[f"{name}_maximum_weight_change"] > 0 for name in ("actor", "critic"))
    changes["weights_and_optimizer_finite"] = True
    return changes


def train(args: argparse.Namespace) -> None:
    """Launch a native learner and retain source/configuration/budget evidence."""
    from isaaclab.utils.io import load_yaml

    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "comparison.json"
    if manifest.exists():
        raise FileExistsError(f"Use a fresh run directory; a manifest already exists at {manifest}.")
    information = metadata(args)
    direct_url = distribution("robolearn-rl").read_text("direct_url.json")
    if direct_url:
        installed = json.loads(direct_url)
        parsed = urlparse(installed["url"])
        if parsed.scheme == "file" and installed.get("dir_info", {}).get("editable"):
            robolearn_root = Path(unquote(parsed.path))
            information["robolearn_revision"] = subprocess.check_output(
                ["git", "-C", str(robolearn_root), "rev-parse", "HEAD"], text=True
            ).strip()
            information["robolearn_diff_sha256"] = hashlib.sha256(
                subprocess.check_output(["git", "-C", str(robolearn_root), "diff", "HEAD"])
            ).hexdigest()
            sources = ["src/robolearn/warp/ppo.py", "src/robolearn/flashsac/agent.py", "src/robolearn/isaaclab.py"]
            information["robolearn_source_files"] = {
                relative: hashlib.sha256((robolearn_root / relative).read_bytes()).hexdigest() for relative in sources
            }
    harness_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    information.update(
        protocol="g1_native_learning_baselines_v1",
        harness_source_sha256=harness_sha,
        gpu_isolation=os.environ.get("G1_GPU_ISOLATION", "unknown"),
        osmo_workflow_id=os.environ.get("WORKFLOW_ID"),
        allocation_cpus=os.environ.get("ALLOCATION_CPUS"),
        walking_gate={
            "scenario": "forward_0_5",
            "walking_success_rate_min": 0.8,
            "linear_velocity_error_mean_max": 0.2,
            "survival_rate_min": 0.9,
        },
        mdp_execution="torch_eager",
        capture_scope="separate_physics_and_learner",
        notes=[
            "The unmodified registered G1 flat MDP and native Newton MJWarp physics are used.",
            "The saved agent_config and algorithm_config identify this run's actual learner recipe and policy support.",
            "The collection budget is matched; algorithms may perform different numbers and sizes of replay updates.",
            "CUDA-synchronized phase timings exclude logging and checkpoint I/O; process wall time includes them.",
            "Common deterministic full native episodes measure learning quality independently of training log windows.",
            "No reward, observation, action scaling, reset or command distribution is changed for training.",
        ],
    )
    is_rsl = args.algorithm == "rsl_rl_ppo"
    library = "rsl_rl" if is_rsl else "robolearn"
    tag = f"{args.output.name}_{args.algorithm}_s{args.seed}_{uuid.uuid4().hex[:8]}"
    common = [
        "--task",
        TASK,
        "--num_envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
        "--max_iterations",
        str(args.iterations),
        "--device",
        args.device,
        "--frontend",
        "torch",
        "--run_name",
        tag,
        "physics=newton_mjwarp",
        "env.compute_final_obs=true",
        f"agent.device={args.device}",
    ]
    if is_rsl:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "profile-rsl",
            "--completed_checkpoints",
            ",".join(map(str, args.checkpoint_iterations)),
            *common,
        ]
    else:
        command = [
            str(Path(sys.executable).with_name("isaaclab")),
            "train",
            "--rl_library",
            "robolearn",
            "--algorithm",
            args.algorithm,
            *common,
        ]
        command.append(f"agent.save_interval={math.gcd(50, args.iterations, *args.checkpoint_iterations)}")
    command += args.train_override
    names = {"rsl_rl_ppo": "RSL-RL PPO · stock G1", "warp_ppo": "Warp PPO · stock G1 recipe", "flashsac": "FlashSAC"}
    run = {
        "name": names[args.algorithm],
        "algorithm": args.algorithm,
        "variant": args.algorithm,
        "seed": args.seed,
        "device": args.device,
        "command": command,
        "status": "running",
        "evaluations": [],
    }
    results = {"metadata": information, "runs": [run]}
    write_results(manifest, results)
    log_root = root / "logs" / library / "g1_flat"
    started, started_unix = time.perf_counter(), time.time()
    run["process_started_unix"] = started_unix
    first_metrics_seconds = None
    print(
        f"Training {run['name']}: {args.iterations} iterations, "
        f"{args.num_envs * HORIZON * args.iterations:,} transitions",
        flush=True,
    )
    with (args.output / f"{args.algorithm}.log").open("w") as log:
        with subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT) as process:
            while process.poll() is None:
                if first_metrics_seconds is None:
                    candidates = list(log_root.glob(f"*_{tag}/metrics.jsonl"))
                    if candidates and candidates[0].stat().st_size > 0:
                        first_metrics_seconds = time.perf_counter() - started
                time.sleep(0.1)
            run.update(process_wall_seconds=time.perf_counter() - started, return_code=process.returncode)
            if process.returncode != 0:
                run["status"] = "failed"
                write_results(manifest, results)
                raise subprocess.CalledProcessError(process.returncode, command)
    created = [path for path in log_root.glob(f"*_{tag}") if path.is_dir()]
    if len(created) != 1:
        raise RuntimeError(f"Expected one tagged native run directory; found {created}.")
    log_dir = created[0]
    rows = [json.loads(line) for line in (log_dir / "metrics.jsonl").read_text().splitlines()]
    expected_steps = args.iterations * args.num_envs * HORIZON
    if len(rows) != args.iterations or rows[-1]["total_steps"] != expected_steps:
        raise RuntimeError("The actual collection budget does not match the baseline protocol.")
    cfg = json_value(load_yaml(str(log_dir / "params/agent.yaml")))
    if is_rsl and not args.train_override:
        stock = {
            "clip_actions": cfg["clip_actions"],
            "activation": cfg["actor"]["activation"],
            "std_type": cfg["actor"]["distribution_cfg"]["std_type"],
            "epochs": cfg["algorithm"]["num_learning_epochs"],
            "minibatches": cfg["algorithm"]["num_mini_batches"],
            "learning_rate": cfg["algorithm"]["learning_rate"],
            "schedule": cfg["algorithm"]["schedule"],
            "desired_kl": cfg["algorithm"]["desired_kl"],
            "value_loss_coef": cfg["algorithm"]["value_loss_coef"],
        }
        expected = {
            "clip_actions": None,
            "activation": "elu",
            "std_type": "scalar",
            "epochs": 5,
            "minibatches": 4,
            "learning_rate": 0.001,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "value_loss_coef": 1.0,
        }
        if stock != expected:
            raise ValueError(f"The native G1 PPO recipe unexpectedly changed: {stock}.")
        run["stock_recipe_verified"] = stock
    elapsed, curve = 0.0, []
    for row in rows:
        elapsed += row["iteration_seconds"]
        if row["mean_episode_return"] is not None:
            curve.append(
                {
                    "iteration": row["iteration"],
                    "transitions": row["total_steps"],
                    "time_seconds": elapsed,
                    "wall_time_seconds": row["timestamp_unix"] - started_unix,
                    "return_mean": row["mean_episode_return"],
                }
            )
    checkpoints = list(log_dir.glob("model_*.pt" if is_rsl else "model_*.json"))
    final_checkpoint = max(checkpoints, key=lambda path: int(path.stem.split("_")[-1]))
    profile = json.loads((log_dir / "profile.json").read_text()) if is_rsl else {}
    steady, warmup = rows[5:] or rows, sum(row["warmup_seconds"] for row in rows)
    run.update(
        status="completed",
        log_dir=str(log_dir),
        iterations=args.iterations,
        num_envs=args.num_envs,
        horizon=HORIZON,
        transitions=expected_steps,
        training_seconds=elapsed,
        rollout_seconds=sum(row["rollout_seconds"] for row in rows),
        update_seconds=sum(row["update_seconds"] for row in rows),
        warmup_seconds=warmup,
        non_loop_seconds=run["process_wall_seconds"] - elapsed - warmup,
        cold_startup_seconds=max(0.0, first_metrics_seconds - rows[0]["iteration_seconds"] - rows[0]["warmup_seconds"])
        if first_metrics_seconds is not None
        else None,
        steady_fps=len(steady) * args.num_envs * HORIZON / sum(row["iteration_seconds"] for row in steady),
        actor_gradient_updates=rows[-1]["actor_gradient_updates"],
        critic_gradient_updates=rows[-1]["critic_gradient_updates"],
        parameter_count=profile["actor_parameters"] + profile["critic_parameters"] if profile else None,
        agent_config=cfg,
        agent_config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        environment_config_file_sha256=hashlib.sha256((log_dir / "params/env.yaml").read_bytes()).hexdigest(),
        training_curve=curve,
        final_checkpoint=str(final_checkpoint),
        learning_evidence=learning_evidence(final_checkpoint, algorithm=args.algorithm),
    )
    if not run["learning_evidence"]["weights_updated"]:
        run["status"] = "invalid_learning"
        write_results(manifest, results)
        raise RuntimeError("Actor or critic weights did not change; refusing to report this as successful training.")
    if is_rsl:
        run["algorithm_config"] = cfg["algorithm"]
        run["policy_support"] = {"distribution": "Gaussian", "wrapper_clip_actions": cfg["clip_actions"]}
        if run["learning_evidence"]["optimizer_steps"] != run["actor_gradient_updates"]:
            raise ValueError("RSL saved optimizer work does not match the metric update count.")
    else:
        run["algorithm_config"] = cfg["algorithm_cfg"]
        run["parameter_count"] = run["learning_evidence"]["parameter_count"]
        run["policy_support"] = {
            "distribution": "Gaussian" if args.algorithm == "warp_ppo" else "Tanh Gaussian",
            "wrapper_clip_actions": cfg["clip_actions"],
        }
        saved = json.loads(final_checkpoint.read_text())
        if saved["actor_gradient_updates"] != run["actor_gradient_updates"]:
            raise ValueError("The saved actor update counter does not match the training metrics.")
        if saved["critic_gradient_updates"] != run["critic_gradient_updates"]:
            raise ValueError("The saved critic update counter does not match the training metrics.")
        if args.algorithm == "warp_ppo":
            if run["learning_evidence"]["optimizer_steps"] != run["actor_gradient_updates"]:
                raise ValueError("Warp saved optimizer work does not match the metric update count.")
        elif run["learning_evidence"]["replay_update_calls"] != run["critic_gradient_updates"]:
            raise ValueError("FlashSAC replay update calls do not match the saved critic update counter.")
        if args.algorithm == "flashsac":
            information["notes"].append(
                "AMP optimizer step counts are audited separately from attempted replay update calls; "
                "overflow can skip an optimizer step. Interleaved phase timing uses CUDA events."
            )
    health_path = log_dir / "health_metrics.jsonl"
    if health_path.exists():
        run["health_curve"] = [json.loads(line) for line in health_path.read_text().splitlines()]
    elif args.algorithm == "warp_ppo":
        run["health_curve"] = [{"iteration": row["iteration"], **row["health"]} for row in rows]
    else:
        run["health_curve"] = [
            {
                "iteration": row["iteration"],
                "actor_gradient_updates": row["actor_gradient_updates"],
                "critic_gradient_updates": row["critic_gradient_updates"],
                **row["losses"],
            }
            for row in rows
        ]
    write_results(manifest, results)
    print(f"Finished {run['name']}: loop={elapsed:.2f}s, process={run['process_wall_seconds']:.2f}s", flush=True)


def evaluate_policy(env, policy, *, tensor_dict: bool, seed: int, scenario: str, foot_contacts: bool = False) -> dict:
    """Score deterministic native actions for one full episode per world."""
    import torch
    from tensordict import TensorDict

    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    observations, _ = env.reset(seed=seed)
    active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    keys = (
        "return",
        "length",
        "upright",
        "linear_error",
        "linear_squared",
        "yaw_error",
        "yaw_squared",
        "forward_velocity",
        "forward_command",
        "action_squared",
        "action_outside_unit",
    )
    sums = {key: torch.zeros(env.num_envs, device=env.device) for key in keys}
    survived = torch.zeros_like(active)
    action_abs_max = torch.zeros((), device=env.device)
    robot = env.scene["robot"]
    if foot_contacts:
        sensor = env.scene["contact_forces"]
        foot_ids, foot_names = sensor.find_sensors(".*_ankle_roll_link", preserve_order=True)
        if len(foot_ids) != 2:
            raise ValueError(f"Expected two G1 feet, found {foot_names}.")
        contact_sums = torch.zeros((env.num_envs, 2), device=env.device)
        touchdowns = torch.zeros_like(contact_sums)
        previous_contact = torch.zeros_like(contact_sums, dtype=torch.bool)
        last_touchdown_side = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.int64)
        alternations = torch.zeros(env.num_envs, device=env.device)
        comparisons = torch.zeros_like(alternations)
        support_sums = torch.zeros((env.num_envs, 3), device=env.device)
        contact_trace = []
    for step in range(env.max_episode_length):
        velocity = quat_apply_inverse(yaw_quat(robot.data.root_quat_w.torch), robot.data.root_lin_vel_w.torch)
        command = env.command_manager.get_command("base_velocity")
        linear_error = torch.linalg.norm(command[:, :2] - velocity[:, :2], dim=-1)
        yaw_error = (command[:, 2] - robot.data.root_ang_vel_w.torch[:, 2]).abs()
        sums["linear_error"] += linear_error * active
        sums["linear_squared"] += linear_error.square() * active
        sums["yaw_error"] += yaw_error * active
        sums["yaw_squared"] += yaw_error.square() * active
        sums["forward_velocity"] += velocity[:, 0] * active
        sums["forward_command"] += command[:, 0] * active
        sums["upright"] += (robot.data.projected_gravity_b.torch[:, 2] < -0.9) * active
        if foot_contacts:
            contact = torch.linalg.norm(sensor.data.net_forces_w.torch[:, foot_ids], dim=-1) > 1.0
            touchdown = contact & ~previous_contact & active[:, None]
            touchdowns += touchdown
            contact_sums += contact * active[:, None]
            support_count = contact.sum(dim=-1)
            for count in range(3):
                support_sums[:, count] += (support_count == count) * active
            side = torch.where(
                touchdown[:, 0] & ~touchdown[:, 1], 0, torch.where(touchdown[:, 1] & ~touchdown[:, 0], 1, -1)
            )
            comparable = (side >= 0) & (last_touchdown_side >= 0)
            comparisons += comparable
            alternations += comparable & (side != last_touchdown_side)
            last_touchdown_side = torch.where(side >= 0, side, last_touchdown_side)
            previous_contact.copy_(contact)
            if step % 5 == 0:
                contact_trace.append(
                    {
                        "seconds": step * env.step_dt,
                        "contact": contact[:4].cpu().tolist(),
                        "active": active[:4].cpu().tolist(),
                    }
                )
        inputs = TensorDict(observations, batch_size=[env.num_envs]) if tensor_dict else observations
        actions = policy(inputs)
        if not torch.isfinite(actions).all():
            raise ValueError("The deterministic policy produced nonfinite actions.")
        sums["action_squared"] += actions.square().mean(dim=-1) * active
        sums["action_outside_unit"] += (actions.abs() > 1.0).float().mean(dim=-1) * active
        action_abs_max = torch.maximum(action_abs_max, (actions.abs() * active[:, None]).max())
        observations, reward, terminated, truncated, _ = env.step(actions)
        sums["return"] += reward * active
        sums["length"] += active
        survived |= active & truncated & ~terminated
        active &= ~(terminated | truncated)
        if not active.any():
            break
    lengths = sums["length"].clamp_min(1)
    linear, yaw, forward = (
        sums["linear_error"] / lengths,
        sums["yaw_error"] / lengths,
        sums["forward_velocity"] / lengths,
    )
    result = {
        "scenario": scenario,
        "evaluation_seed": seed,
        "evaluation_envs": env.num_envs,
        "return_mean": float(sums["return"].mean()),
        "return_std": float(sums["return"].std(unbiased=False)),
        "episode_length_mean": float(lengths.mean()),
        "episode_seconds_mean": float(lengths.mean()) * env.step_dt,
        "survival_rate": float(survived.float().mean()),
        "upright_fraction": float((sums["upright"] / lengths).mean()),
        "linear_velocity_error_mean": float(linear.mean()),
        "linear_velocity_rmse": float((sums["linear_squared"] / lengths).sqrt().mean()),
        "yaw_velocity_error_mean": float(yaw.mean()),
        "yaw_velocity_rmse": float((sums["yaw_squared"] / lengths).sqrt().mean()),
        "forward_velocity_mean": float(forward.mean()),
        "forward_command_mean": float((sums["forward_command"] / lengths).mean()),
        "forward_displacement_mean": float(sums["forward_velocity"].mean()) * env.step_dt,
        "tracking_success_rate": float((survived & (linear < 0.5) & (yaw < 0.8)).float().mean()),
        "action_rms": float((sums["action_squared"] / lengths).mean().sqrt()),
        "action_abs_max": float(action_abs_max),
        "action_outside_unit_fraction": float((sums["action_outside_unit"] / lengths).mean()),
        "evaluation_extra_action_clip": None,
    }
    if foot_contacts:
        result["foot_contacts"] = {
            "body_names": foot_names,
            "force_threshold_newtons": 1.0,
            "contact_fraction": (contact_sums / lengths[:, None]).mean(dim=0).cpu().tolist(),
            "touchdowns_per_episode_mean": touchdowns.mean(dim=0).cpu().tolist(),
            "flight_fraction": float((support_sums[:, 0] / lengths).mean()),
            "single_support_fraction": float((support_sums[:, 1] / lengths).mean()),
            "double_support_fraction": float((support_sums[:, 2] / lengths).mean()),
            "alternating_touchdown_fraction": float(alternations.sum() / comparisons.sum().clamp_min(1)),
            "touchdown_comparisons": int(comparisons.sum()),
            "trace": contact_trace,
            "trace_note": "Pre-step contacts for the first four worlds, sampled every five control steps; "
            "inactive worlds have finished their scored episode.",
        }
    if scenario == "forward_0_5":
        result["walking_success_rate"] = float(
            (survived & (linear < 0.25) & (yaw < 0.4) & (forward > 0.25)).float().mean()
        )
        result["walking_gate_passed"] = (
            result["walking_success_rate"] >= 0.8
            and result["linear_velocity_error_mean"] <= 0.2
            and result["survival_rate"] >= 0.9
        )
    return result


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate exact completed budgets with noise/pushes disabled in both scenarios."""
    import gymnasium as gym
    import torch

    from isaaclab.app import launch_simulation
    from isaaclab.utils.io import load_yaml

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    manifest = args.output / "comparison.json"
    results = json.loads(manifest.read_text())
    results["metadata"]["evaluation_notes"] = [
        "Deterministic native actor actions; no added wrapper clipping.",
        "64 environments by default, one full native 20-second episode, reset seed=10000+training seed.",
        "Observation noise and random pushes are disabled; reset and startup mass randomization remain native.",
        "Native commands retain training command ranges; "
        "forward commands fix x=.5m/s,y=0,yaw=0 without heading/standing.",
        "Planar velocity uses the pre-step yaw frame and yaw angular velocity uses world z, "
        "matching native reward definitions.",
        "Walking success requires survival, mean planar error<.25m/s, yaw error<.4rad/s and forward speed>.25m/s.",
        "Checkpoint budgets are exact completed iterations, including native RSL's zero-based filenames.",
    ]
    for run in results["runs"]:
        if run["status"] != "completed":
            raise RuntimeError("Evaluation requires completed training.")
        log_dir, is_rsl = Path(run["log_dir"]), run["algorithm"] == "rsl_rl_ppo"
        available = list(log_dir.glob("model_*.pt" if is_rsl else "model_*.json"))
        completed = {int(path.stem.split("_")[-1]) + int(is_rsl): path for path in available}
        missing = set(args.checkpoint_iterations) - completed.keys()
        if missing:
            raise ValueError(f"Requested exact completed budgets are missing: {sorted(missing)}.")
        rows = [json.loads(line) for line in (log_dir / "metrics.jsonl").read_text().splitlines()]
        run["evaluations"] = []
        for scenario in args.scenarios:
            seed = run["seed"] + 10_000
            env_cfg = parse_env_cfg(
                TASK, device=args.device, num_envs=args.eval_envs, overrides=["physics=newton_mjwarp"]
            )
            env_cfg.seed, env_cfg.compute_final_obs = seed, True
            env_cfg.observations.policy.enable_corruption, env_cfg.events.push_robot = False, None
            if scenario == "forward_0_5":
                command = env_cfg.commands.base_velocity
                command.heading_command, command.rel_standing_envs = False, 0.0
                command.ranges.lin_vel_x, command.ranges.lin_vel_y, command.ranges.ang_vel_z = (
                    (0.5, 0.5),
                    (0.0, 0.0),
                    (0.0, 0.0),
                )
            with launch_simulation(env_cfg):
                env = gym.make(TASK, cfg=env_cfg).unwrapped
                try:
                    if is_rsl:
                        from rsl_rl.runners import OnPolicyRunner

                        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

                        cfg = load_yaml(str(log_dir / "params/agent.yaml"))
                        runner = OnPolicyRunner(
                            RslRlVecEnvWrapper(env, clip_actions=cfg["clip_actions"]),
                            cfg,
                            log_dir=None,
                            device=env.device,
                        )
                        run["parameter_count"] = sum(
                            p.numel() for model in (runner.alg.actor, runner.alg.critic) for p in model.parameters()
                        )
                    else:
                        from isaaclab_rl.robolearn import RoboLearnRunner

                        cfg = json.loads(completed[max(completed)].read_text())["config"]
                        cfg["device"] = env.device
                        runner = RoboLearnRunner(env, cfg, device=env.device)
                    for iteration in sorted(args.checkpoint_iterations):
                        checkpoint = completed[iteration]
                        if is_rsl:
                            runner.load(str(checkpoint), map_location=env.device)
                        else:
                            runner.load(str(checkpoint))
                        with torch.inference_mode():
                            measurement = evaluate_policy(
                                env,
                                runner.get_inference_policy(device=env.device),
                                tensor_dict=is_rsl,
                                seed=seed,
                                scenario=scenario,
                                foot_contacts=args.foot_contacts,
                            )
                        measurement.update(
                            checkpoint=str(checkpoint),
                            iteration=iteration,
                            transitions=iteration * run["num_envs"] * run["horizon"],
                            time_seconds=sum(row["iteration_seconds"] for row in rows[:iteration]),
                            wall_time_seconds=rows[iteration - 1]["timestamp_unix"] - run["process_started_unix"],
                        )
                        run["evaluations"].append(measurement)
                        write_results(manifest, results)
                        print(
                            f"{run['name']} {scenario} iteration={iteration} return={measurement['return_mean']:.3f} "
                            f"survival={measurement['survival_rate']:.1%} "
                            f"linear_error={measurement['linear_velocity_error_mean']:.3f}m/s "
                            f"walking={measurement.get('walking_success_rate')}",
                            flush=True,
                        )
                    del runner
                finally:
                    env.close()
            torch.cuda.empty_cache()
    print(f"Results: {manifest.resolve()}", flush=True)


def main() -> None:
    """Parse the baseline protocol without overriding the native task configuration."""
    if len(sys.argv) > 1 and sys.argv[1] == "profile-rsl":
        profile_rsl(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["train", "evaluate"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithm", choices=["rsl_rl_ppo", "warp_ppo", "flashsac"], default="rsl_rl_ppo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--eval_envs", type=int, default=64)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[250, 500, 1000, 1500])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument(
        "--train_override", action="append", default=[], help="Explicit Hydra override recorded in the run command."
    )
    parser.add_argument(
        "--foot_contacts", action="store_true", help="Record native foot-contact diagnostics during evaluation."
    )
    args = parser.parse_args()
    if min(args.num_envs, args.iterations, args.eval_envs, *args.checkpoint_iterations) < 1:
        parser.error("Environment counts, iterations and checkpoint budgets must be positive.")
    args.output = args.output.resolve()
    args.capture_rollout, args.optimized_linear_backward = False, False
    {"train": train, "evaluate": evaluate}[args.phase](args)


if __name__ == "__main__":
    main()
