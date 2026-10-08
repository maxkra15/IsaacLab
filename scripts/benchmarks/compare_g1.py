# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train and evaluate matched G1 walking PPO implementations.

Launch one run per output directory and GPU from the Isaac Lab checkout::

    uv run --no-sync python scripts/benchmarks/compare_g1.py train \
        --output logs/g1-warp-seed0 --algorithm warp_ppo --device cuda:0
    uv run --no-sync python scripts/benchmarks/compare_g1.py evaluate \
        --output logs/g1-warp-seed0 --device cuda:0 --checkpoint_iterations 500 1000 1500

Both implementations use the same registered MDP and five full-batch epochs.
The comparison changes the stock G1 agent's ELU activation, adaptive learning
rate and minibatch count to match RoboLearn's current Warp PPO capabilities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import MISSING, asdict
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

TASK = "Isaac-Velocity-Flat-G1"
HORIZON = 24
SCENARIOS = ("native_commands", "forward_0_5")


def json_value(value):
    """Represent configuration values deterministically without nonfinite JSON numbers."""
    if value is MISSING:
        return "MISSING"
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported configuration value: {type(value).__module__}.{type(value).__qualname__}")


def write_results(path: Path, results: dict) -> None:
    """Atomically save completed stages and failure diagnostics."""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def warp_learning_evidence(checkpoint: Path) -> dict:
    """Confirm finite weights moved from initialization in a saved G1 training run."""
    import numpy as np

    saved = json.loads(checkpoint.read_text())
    cfg = saved["config"]
    algorithm = cfg["algorithm_cfg"]
    rng = np.random.default_rng(cfg["seed"])
    index, changes = 0, {}
    with np.load(checkpoint.with_suffix("") / "policy.npz", allow_pickle=False) as state:
        if not all(np.isfinite(state[key]).all() for key in state.files):
            raise ValueError("Warp policy or optimizer state contains nonfinite values.")
        for name, output in (("actor", saved["action_dim"]), ("critic", 1)):
            first_index = index
            widths = (saved["observation_dim"], *algorithm["hidden_dims"], output)
            maximum = 0.0
            for input_dim, output_dim in zip(widths[:-1], widths[1:], strict=True):
                bound = 1.0 / math.sqrt(input_dim)
                for shape in ((output_dim, input_dim), (output_dim, 1)):
                    initial = rng.uniform(-bound, bound, shape).astype(np.float32)
                    maximum = max(maximum, float(np.abs(state[f"parameter_{index}"] - initial).max()))
                    index += 1
            changes[f"{name}_maximum_weight_change"] = maximum
            changes[f"{name}_adam_first_moment_norm"] = float(
                np.linalg.norm(state[f"adam_m1_{first_index}"].astype(np.float64))
            )
        changes["optimizer_steps"] = float(state["adam_timestep"][0])
    changes["weights_updated"] = all(changes[f"{name}_maximum_weight_change"] > 0 for name in ("actor", "critic"))
    return changes


def metadata(args: argparse.Namespace) -> dict:
    """Record the hardware and normalized pre-launch MDP configuration."""
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs, overrides=["physics=newton_mjwarp"])
    cfg.seed = args.seed
    cfg.compute_final_obs = True
    environment = json_value(cfg.to_dict())

    def normalize(value):
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items() if key not in {"seed", "device", "log_dir"}}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    normalized = normalize(environment)
    config_hash = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()
    paths = [
        "source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/g1/flat_env_cfg.py",
        "source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/g1/rough_env_cfg.py",
        "source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py",
        "source/isaaclab_tasks/isaaclab_tasks/core/velocity/mdp/rewards.py",
        "source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py",
        "source/isaaclab_assets/isaaclab_assets/robots/unitree.py",
    ]
    source_hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths}
    benchmark_paths = [
        "scripts/benchmarks/compare_g1.py",
        "scripts/benchmarks/profile_rsl_ppo.py",
        "source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/g1/agents/robolearn_cfg.py",
        "source/isaaclab_rl/isaaclab_rl/robolearn/runner.py",
        "source/isaaclab_rl/isaaclab_rl/entrypoints/backends/train_robolearn.py",
        "source/isaaclab_rl/isaaclab_rl/entrypoints/backends/train_rsl_rl.py",
    ]
    benchmark_hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in benchmark_paths}
    inventory_text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,utilization.gpu,memory.used,clocks.current.sm",
            "--format=csv,nounits",
        ],
        text=True,
    )
    inventory = [
        {key.strip(): value.strip() for key, value in row.items()}
        for row in csv.DictReader(io.StringIO(inventory_text))
    ]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    device_index = int(args.device.split(":")[-1])
    physical_device = visible.split(",")[device_index].strip() if visible else str(device_index)
    selected = next(
        (gpu for gpu in inventory if gpu["index"] == physical_device or gpu["uuid"].startswith(physical_device)),
        None,
    )
    versions = {}
    for name in ("torch", "rsl-rl-lib", "warp-lang", "warp-nn", "newton", "mujoco-warp", "robolearn-rl"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    direct_url = distribution("robolearn-rl").read_text("direct_url.json")
    robolearn_source = json.loads(direct_url) if direct_url else {}
    cpu_name = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    return {
        "task": TASK,
        "physics": "newton_mjwarp",
        "frontend": "torch",
        "seed": args.seed,
        "device": args.device,
        "hardware": selected["name"] if selected else inventory_text.strip(),
        "gpu_inventory": inventory,
        "selected_gpu": selected,
        "cuda_visible_devices": visible,
        "cpu": {"model": cpu_name, "logical_cores": os.cpu_count(), "available_cores": len(os.sched_getaffinity(0))},
        "platform": platform.platform(),
        "isaaclab_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "isaaclab_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
        "robolearn_revision": robolearn_source.get("vcs_info", {}).get("commit_id"),
        "versions": versions,
        "num_envs": args.num_envs,
        "horizon": HORIZON,
        "iterations": args.iterations,
        "step_dt": cfg.sim.dt * cfg.decimation,
        "episode_seconds": cfg.episode_length_s,
        "mdp_sha256": config_hash,
        "environment_config_sha256": config_hash,
        "mdp_source_sha256": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
        "mdp_source_files": source_hashes,
        "benchmark_source_sha256": hashlib.sha256(json.dumps(benchmark_hashes, sort_keys=True).encode()).hexdigest(),
        "benchmark_source_files": benchmark_hashes,
        "environment_config": normalized,
        "notes": [
            "Matched manager MDP, observation group, Newton MJWarp physics, noise, reset and command distributions.",
            "Executed actions are clipped to [-1,1], unlike the stock unclipped G1 PPO configuration.",
            "PPO: separate 256/128/128 Tanh MLPs, log std, fixed lr=.0003, "
            "five full-batch epochs, gamma=.99, lambda=.95.",
            "Warp captures the learning update; Isaac Lab rollout assembly remains eager.",
            "Implementations differ in initialization, timeout bootstrap, gradient clipping and numeric precision.",
            "GPU-synchronized phase timings exclude checkpoint I/O and logging; process wall time includes them.",
            "Steady throughput excludes the first five iterations; capture preparation is reported separately.",
            "Cold startup estimate subtracts first-loop and capture times from first metrics appearance; "
            "polling resolution=.1 s.",
            "Training returns use different episode windows; "
            "common deterministic evaluations support quality comparisons.",
            "Native runner synchronization, episode statistics, full-batch shuffling and loss reads also differ; "
            "timings do not isolate CUDA graph capture alone.",
            "Configuration hashes exclude seed, device and log paths; "
            "source fingerprints identify the MDP definitions.",
        ],
    }


def train(args: argparse.Namespace) -> None:
    """Run one learner on its selected GPU and retain its actual measured budget."""
    from isaaclab.utils.io import load_yaml

    args.output.mkdir(parents=True, exist_ok=False)
    root = Path.cwd()
    manifest = args.output / "comparison.json"
    results = {"metadata": metadata(args), "runs": []}
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
    ]
    if is_rsl:
        command = [sys.executable, str(root / "scripts/benchmarks/profile_rsl_ppo.py"), *common]
        command += [
            f"agent.device={args.device}",
            "agent.num_steps_per_env=24",
            "agent.actor.hidden_dims=[256,128,128]",
            "agent.critic.hidden_dims=[256,128,128]",
            "agent.actor.activation=tanh",
            "agent.critic.activation=tanh",
            "agent.actor.obs_normalization=false",
            "agent.critic.obs_normalization=false",
            "agent.actor.distribution_cfg.std_type=log",
            "agent.actor.distribution_cfg.init_std=1.0",
            "agent.algorithm.num_learning_epochs=5",
            "agent.algorithm.num_mini_batches=1",
            "agent.algorithm.schedule=fixed",
            "agent.algorithm.learning_rate=0.0003",
            "agent.algorithm.gamma=0.99",
            "agent.algorithm.lam=0.95",
            "agent.algorithm.clip_param=0.2",
            "agent.algorithm.entropy_coef=0.008",
            "agent.algorithm.value_loss_coef=1.0",
            "agent.algorithm.use_clipped_value_loss=true",
            "agent.algorithm.max_grad_norm=1.0",
            "agent.clip_actions=1.0",
            "agent.init_at_random_ep_len=false",
        ]
    else:
        command = [str(Path(sys.executable).with_name("isaaclab")), "train", "--rl_library", library]
        command += ["--algorithm", "warp_ppo", *common]
    run = {
        "name": "RSL-RL PPO" if is_rsl else "Warp PPO",
        "algorithm": args.algorithm,
        "seed": args.seed,
        "device": args.device,
        "command": command,
        "status": "running",
        "evaluations": [],
    }
    results["runs"].append(run)
    write_results(manifest, results)
    log_root = root / "logs" / library / "g1_flat"
    first_metrics_seconds = None
    print(
        f"Training {run['name']}: {args.iterations} iterations, "
        f"{args.num_envs * HORIZON * args.iterations:,} transitions",
        flush=True,
    )
    started = time.perf_counter()
    with (args.output / f"{args.algorithm}.log").open("w") as log:
        with subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT) as process:
            while process.poll() is None:
                if first_metrics_seconds is None:
                    candidates = list(log_root.glob(f"*_{tag}/metrics.jsonl"))
                    if candidates and candidates[0].stat().st_size > 0:
                        first_metrics_seconds = time.perf_counter() - started
                time.sleep(0.1)
            process_seconds = time.perf_counter() - started
            run["process_wall_seconds"] = process_seconds
            run["return_code"] = process.returncode
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
        raise RuntimeError("The actual collection budget does not match the comparison protocol.")
    cfg = json_value(load_yaml(str(log_dir / "params/agent.yaml")))
    elapsed, curve = 0.0, []
    for row in rows:
        elapsed += row["iteration_seconds"]
        if row["mean_episode_return"] is not None:
            curve.append(
                {
                    "iteration": row["iteration"],
                    "transitions": row["total_steps"],
                    "time_seconds": elapsed,
                    "return_mean": row["mean_episode_return"],
                }
            )
    profile = json.loads((log_dir / "profile.json").read_text()) if is_rsl else {}
    steady = rows[5:] or rows
    warmup = sum(row["warmup_seconds"] for row in rows)
    checkpoints = list(log_dir.glob("model_*.pt" if is_rsl else "model_*.json"))
    final_checkpoint = max(checkpoints, key=lambda path: int(path.stem.split("_")[-1]))
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
        cold_startup_seconds=max(0.0, first_metrics_seconds - rows[0]["iteration_seconds"] - rows[0]["warmup_seconds"])
        if first_metrics_seconds is not None
        else None,
        non_loop_seconds=process_seconds - elapsed - warmup,
        steady_fps=len(steady) * args.num_envs * HORIZON / sum(row["iteration_seconds"] for row in steady),
        parameter_count=(profile["actor_parameters"] + profile["critic_parameters"]) if profile else None,
        actor_gradient_updates=rows[-1]["actor_gradient_updates"],
        critic_gradient_updates=rows[-1]["critic_gradient_updates"],
        peak_torch_memory_mib=max((row.get("torch_peak_allocated_bytes", 0) for row in rows), default=0) / 2**20
        or None,
        agent_config=cfg,
        agent_config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        environment_config_file_sha256=hashlib.sha256((log_dir / "params/env.yaml").read_bytes()).hexdigest(),
        training_curve=curve,
        final_checkpoint=str(final_checkpoint),
    )
    if not is_rsl:
        checkpoint = json.loads(final_checkpoint.read_text())
        run["observation_dim"], run["action_dim"] = checkpoint["observation_dim"], checkpoint["action_dim"]
        run["algorithm_config"] = checkpoint["config"]["algorithm_cfg"]
        run["learning_evidence"] = warp_learning_evidence(final_checkpoint)
        if not run["learning_evidence"]["weights_updated"]:
            run["status"] = "invalid_learning"
            write_results(manifest, results)
            raise RuntimeError("Warp actor or critic weights did not change; refusing to report this as training.")
    write_results(manifest, results)
    print(
        f"Finished {run['name']}: loop={elapsed:.2f}s, process={process_seconds:.2f}s, "
        f"steady={run['steady_fps']:.0f} transitions/s",
        flush=True,
    )


def evaluate_policy(env, policy, *, tensor_dict: bool, seed: int, scenario: str) -> dict:
    """Measure one deterministic whole episode per environment, before automatic resets."""
    import torch
    from tensordict import TensorDict

    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    observations, _ = env.reset(seed=seed)
    active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    sums = {
        key: torch.zeros(env.num_envs, device=env.device)
        for key in (
            "return",
            "length",
            "upright",
            "linear_error",
            "linear_squared",
            "yaw_error",
            "yaw_squared",
            "forward_velocity",
            "forward_command",
        )
    }
    survived = torch.zeros_like(active)
    robot = env.scene["robot"]
    for _ in range(env.max_episode_length):
        # Post-step root state may already be reset; score the current state and command.
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
        inputs = TensorDict(observations, batch_size=[env.num_envs]) if tensor_dict else observations
        observations, reward, terminated, truncated, _ = env.step(policy(inputs).clamp(-1, 1))
        sums["return"] += reward * active
        sums["length"] += active
        survived |= active & truncated & ~terminated
        active &= ~(terminated | truncated)
        if not active.any():
            break
    lengths = sums["length"].clamp_min(1)
    linear = sums["linear_error"] / lengths
    yaw = sums["yaw_error"] / lengths
    forward = sums["forward_velocity"] / lengths
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
    }
    if scenario == "forward_0_5":
        result["walking_success_rate"] = float(
            (survived & (linear < 0.25) & (yaw < 0.4) & (forward > 0.25)).float().mean()
        )
    return result


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate selected saved policies on common native and forward command scenarios."""
    import gymnasium as gym
    import torch

    from isaaclab.app import launch_simulation
    from isaaclab.utils.io import load_yaml

    from isaaclab_rl.robolearn import RoboLearnRunner

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    manifest = args.output / "comparison.json"
    results = json.loads(manifest.read_text())
    results["metadata"]["evaluation_notes"] = [
        "Deterministic actors, 64 environments by default, one complete episode each; "
        "common reset seed=10000+training seed.",
        "Both scenarios disable observation corruption and random pushes; "
        "remaining reset and mass randomization are retained.",
        "Native scenario retains training command distribution; "
        "forward scenario fixes x=.5 m/s, y=0, yaw=0 without heading control or standing samples.",
        "Tracking errors use pre-step yaw-frame linear velocity and world-z angular velocity, "
        "matching G1 reward definitions.",
        "Forward displacement is the integral of yaw-frame forward velocity, not straight-line world displacement.",
        "Walking success requires 20-second survival, mean planar error<.25 m/s, "
        "yaw error<.4 rad/s and forward speed>.25 m/s.",
        "RSL checkpoint filenames are zero-based; plots use actual completed iterations, "
        "including closest selected checkpoints.",
    ]
    for run in results["runs"]:
        if run["status"] != "completed":
            raise RuntimeError("Evaluation requires a completed training run.")
        log_dir = Path(run["log_dir"])
        is_rsl = run["algorithm"] == "rsl_rl_ppo"
        available = sorted(
            log_dir.glob("model_*.pt" if is_rsl else "model_*.json"), key=lambda path: int(path.stem.split("_")[-1])
        )
        completed = {path: int(path.stem.split("_")[-1]) + int(is_rsl) for path in available}
        selected = sorted(
            {
                min(available, key=lambda path: abs(completed[path] - requested))
                for requested in args.checkpoint_iterations
            },
            key=completed.get,
        )
        rows = [json.loads(line) for line in (log_dir / "metrics.jsonl").read_text().splitlines()]
        run["evaluations"] = []
        for scenario in args.scenarios:
            seed = run["seed"] + 10_000
            env_cfg = parse_env_cfg(
                TASK, device=args.device, num_envs=args.eval_envs, overrides=["physics=newton_mjwarp"]
            )
            env_cfg.seed, env_cfg.compute_final_obs = seed, True
            env_cfg.observations.policy.enable_corruption = False
            env_cfg.events.push_robot = None
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
                            RslRlVecEnvWrapper(env, clip_actions=1.0), cfg, log_dir=None, device=env.device
                        )
                        run["parameter_count"] = sum(
                            p.numel() for model in (runner.alg.actor, runner.alg.critic) for p in model.parameters()
                        )
                    else:
                        cfg = json.loads(available[-1].read_text())["config"]
                        cfg["device"] = env.device
                        runner = RoboLearnRunner(env, cfg, device=env.device)
                        run["algorithm_config"] = asdict(runner.agent.config)
                        run["parameter_count"] = sum(
                            math.prod(parameter.shape) for parameter in runner.agent.parameters
                        )
                    for checkpoint in selected:
                        if is_rsl:
                            runner.load(str(checkpoint), map_location=env.device)
                        else:
                            runner.load(str(checkpoint))
                        with torch.inference_mode():
                            evaluation = evaluate_policy(
                                env,
                                runner.get_inference_policy(device=env.device),
                                tensor_dict=is_rsl,
                                seed=seed,
                                scenario=scenario,
                            )
                        iteration = completed[checkpoint]
                        evaluation.update(
                            checkpoint=str(checkpoint),
                            iteration=iteration,
                            transitions=iteration * run["num_envs"] * run["horizon"],
                            time_seconds=sum(row["iteration_seconds"] for row in rows[:iteration]),
                        )
                        run["evaluations"].append(evaluation)
                        write_results(manifest, results)
                        print(
                            f"{run['name']} {scenario} iteration={iteration} "
                            f"return={evaluation['return_mean']:.3f} survival={evaluation['survival_rate']:.1%} "
                            f"linear_error={evaluation['linear_velocity_error_mean']:.3f} m/s",
                            flush=True,
                        )
                    del runner
                finally:
                    env.close()
            torch.cuda.empty_cache()
    print(f"Results: {manifest.resolve()}", flush=True)


def merge(args: argparse.Namespace) -> None:
    """Combine paired seeds while rejecting incompatible protocols or incomplete budgets."""
    if not args.inputs:
        raise ValueError("The merge phase requires --inputs with run directories or comparison JSON files.")
    sources = []
    for path in args.inputs:
        manifest = path / "comparison.json" if path.is_dir() else path
        sources.append((manifest, json.loads(manifest.read_text())))
    first = sources[0][1]["metadata"]
    common_fields = (
        "task",
        "physics",
        "frontend",
        "num_envs",
        "horizon",
        "iterations",
        "step_dt",
        "episode_seconds",
        "mdp_sha256",
        "environment_config_sha256",
        "mdp_source_sha256",
        "versions",
        "isaaclab_revision",
        "isaaclab_diff_sha256",
        "robolearn_revision",
    )
    runs, identities = [], set()
    for manifest, source in sources:
        information = source["metadata"]
        for key in common_fields:
            if key not in information or information[key] != first[key]:
                raise ValueError(f"Incompatible {key} in {manifest}; expected {first[key]!r}.")
        # Earlier smoke manifests predate explicit fingerprints for untracked benchmark sources.
        if information.get("benchmark_source_sha256") != first.get("benchmark_source_sha256"):
            raise ValueError(f"Incompatible benchmark source fingerprint in {manifest}.")
        for original in source["runs"]:
            run = dict(original)
            identity = (run["algorithm"], run["seed"])
            if identity in identities:
                raise ValueError(f"Duplicate algorithm and seed: {identity}.")
            if run["status"] != "completed":
                raise ValueError(f"Run {identity} in {manifest} has not completed.")
            for key in ("num_envs", "horizon", "iterations"):
                if run[key] != first[key]:
                    raise ValueError(f"Collection budget {key} differs for {identity}.")
            expected = run["num_envs"] * run["horizon"] * run["iterations"]
            if run["transitions"] != expected:
                raise ValueError(f"Actual transitions {run['transitions']} differ from {expected} for {identity}.")
            identities.add(identity)
            gpu = information.get("selected_gpu") or {}
            run.update(
                metadata=information,
                source_manifest=str(manifest.resolve()),
                selected_gpu=gpu,
                gpu_index=gpu.get("index", run["device"]),
                gpu_uuid=gpu.get("uuid"),
                hardware=information["hardware"],
                versions=information["versions"],
            )
            runs.append(run)
    seeds = sorted({seed for _, seed in identities})
    for seed in seeds:
        algorithms = {algorithm for algorithm, run_seed in identities if run_seed == seed}
        if algorithms != {"warp_ppo", "rsl_rl_ppo"}:
            raise ValueError(f"Seed {seed} requires both Warp PPO and RSL-RL PPO; found {sorted(algorithms)}.")
    if not seeds:
        raise ValueError("No completed paired runs were supplied.")
    evaluation_counts = {point["evaluation_envs"] for run in runs for point in run["evaluations"]}
    if len(evaluation_counts) > 1:
        raise ValueError(f"Evaluation environment counts differ: {sorted(evaluation_counts)}.")
    specific = {"seed", "device", "selected_gpu", "gpu_inventory", "cuda_visible_devices", "cpu", "platform"}
    common = {key: value for key, value in first.items() if key not in specific}
    common.update(
        artifact_label="G1 walking · RSL-RL PPO and Warp PPO",
        seeds=seeds,
        hardware=", ".join(dict.fromkeys(source["metadata"]["hardware"] for _, source in sources)),
        hardware_details=[
            {
                "algorithm": run["algorithm"],
                "seed": run["seed"],
                "device": run["device"],
                **{key: run["metadata"][key] for key in specific if key not in {"seed", "device"}},
            }
            for run in runs
        ],
        notes=list(dict.fromkeys(note for _, source in sources for note in source["metadata"].get("notes", []))),
        evaluation_notes=list(
            dict.fromkeys(note for _, source in sources for note in source["metadata"].get("evaluation_notes", []))
        ),
    )
    common["notes"].append(
        f"Paired comparison across {len(seeds)} training seed(s); "
        "seed spread describes these measured runs and is not a population confidence interval."
    )
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = args.output / "comparison.json"
    write_results(manifest, {"metadata": common, "runs": sorted(runs, key=lambda run: (run["seed"], run["algorithm"]))})
    print(f"Merged {len(runs)} runs across {len(seeds)} paired seeds: {manifest.resolve()}", flush=True)


def main() -> None:
    """Parse a single training run or its common evaluation phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["train", "evaluate", "merge"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", nargs="+", type=Path, help="Merge phase: run directories or comparison JSON files.")
    parser.add_argument("--algorithm", choices=["warp_ppo", "rsl_rl_ppo"], default="warp_ppo")
    parser.add_argument("--device", choices=["cuda:0", "cuda:1"], default="cuda:0")
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--eval_envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[500, 1000, 1500])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    args = parser.parse_args()
    if min(args.iterations, args.num_envs, args.eval_envs, *args.checkpoint_iterations) < 1:
        parser.error("Iteration and environment counts must be positive.")
    if args.seed < 0:
        parser.error("Use a nonnegative explicit seed for reproducible comparisons.")
    {"train": train, "evaluate": evaluate, "merge": merge}[args.phase](args)


if __name__ == "__main__":
    main()
