# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train three Cartpole learners serially and evaluate their saved policies.

Run from an Isaac Lab checkout with the ``robolearn`` extra installed::

    uv run --no-sync python scripts/benchmarks/compare_robolearn.py train --output logs/cartpole-comparison
    uv run --no-sync python scripts/benchmarks/compare_robolearn.py evaluate --output logs/cartpole-comparison

This is a single-seed implementation comparison, using the same registered MDP
and collection budget. Both PPO configurations use five full-batch epochs.
The RSL-RL helper instruments its native learning loop without replacing PPO.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path


def write_results(path: Path, results: dict) -> None:
    """Write a readable manifest after every completed stage."""
    path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


def train(args: argparse.Namespace) -> None:
    """Run the three learners sequentially in fresh native run directories."""
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path.cwd()
    manifest = args.output / "comparison.json"
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True
    ).strip()
    results = {
        "metadata": {
            "task": "Isaac-Cartpole",
            "physics": "newton_mjwarp",
            "seed": args.seed,
            "hardware": hardware,
            "isaaclab_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "robolearn_revision": "cae1cf9fdd40eb5f22adf3cba667263f6c9ea5b3",
            "versions": {name: version(name) for name in ("torch", "rsl-rl-lib", "warp-lang", "warp-nn", "newton")},
            "num_envs": args.num_envs,
            "horizon": 16,
            "iterations": args.iterations,
            "step_dt": 1 / 60,
            "episode_seconds": 5.0,
            "normalization": "100 * episode return / 5 (theoretical upper bound; scores may be negative)",
            "notes": [
                "One training seed; exploratory implementation comparison, without confidence intervals.",
                "Same manager MDP, Newton MJWarp physics and reset distributions; executed actions clipped to [-1,1].",
                "PPO: separate 32x32 Tanh MLPs, log std, fixed lr=.001, five full-batch epochs, gamma=.99, lambda=.95.",
                "PPO implementations differ in initialization, joint/separate gradient clipping and timeout bootstrap.",
                "Warp captures the learning update; Isaac Lab rollout assembly remains eager.",
                "FlashSAC uses eager float32 updates in this quick run; compilation and AMP are disabled.",
                "GPU-synchronized phase timings exclude logging, checkpoint I/O and Warp capture preparation.",
                "Steady throughput omits the first five iterations; process wall time includes launch and shutdown.",
                "Other workloads share this GPU; observed times are not isolated hardware benchmarks.",
                "Training return windows differ; deterministic evaluations use the same reset seed and episode count.",
            ],
            "paper": "https://arxiv.org/html/2604.04539v1#S12",
        },
        "runs": [],
    }
    common = ["--task", "Isaac-Cartpole", "--num_envs", str(args.num_envs), "--seed", str(args.seed)]
    common += ["--max_iterations", str(args.iterations), "physics=newton_mjwarp"]
    tag = args.output.name
    for algorithm, name in (("flashsac", "FlashSAC"), ("warp_ppo", "Warp PPO"), ("rsl_rl_ppo", "RSL-RL PPO")):
        if algorithm not in args.algorithms:
            continue
        library = "rsl_rl" if algorithm == "rsl_rl_ppo" else "robolearn"
        log_root = root / "logs" / library / "cartpole"
        before = set(log_root.glob("*"))
        if library == "robolearn":
            command = [str(Path(sys.executable).with_name("isaaclab")), "train", "--rl_library", library]
            command += ["--algorithm", algorithm, "--run_name", tag, *common]
        else:
            command = [sys.executable, str(root / "scripts/benchmarks/profile_rsl_ppo.py"), *common]
            command += [
                "--run_name",
                tag,
                "agent.actor.activation=tanh",
                "agent.critic.activation=tanh",
                "agent.actor.distribution_cfg.std_type=log",
                "agent.algorithm.num_mini_batches=1",
                "agent.algorithm.schedule=fixed",
                "agent.clip_actions=1.0",
                "agent.init_at_random_ep_len=false",
            ]
        print(
            f"Training {name}: {args.iterations} iterations, {args.num_envs * 16 * args.iterations:,} transitions",
            flush=True,
        )
        started = time.perf_counter()
        with (args.output / f"{algorithm}.log").open("w") as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        process_seconds = time.perf_counter() - started
        created = [path for path in set(log_root.glob("*")) - before if path.is_dir()]
        if len(created) != 1:
            raise RuntimeError(f"Expected one new {library} run directory; found {created}.")
        log_dir = created[0]
        rows = [json.loads(line) for line in (log_dir / "metrics.jsonl").read_text().splitlines()]
        if len(rows) != args.iterations or rows[-1]["total_steps"] != args.iterations * args.num_envs * 16:
            raise RuntimeError("The actual collection budget does not match the comparison protocol.")
        elapsed = 0.0
        curve = []
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
        steady = rows[5:] or rows
        profile = json.loads((log_dir / "profile.json").read_text()) if (log_dir / "profile.json").exists() else {}
        results["runs"].append(
            {
                "name": name,
                "algorithm": algorithm,
                "command": command,
                "log_dir": str(log_dir),
                "iterations": args.iterations,
                "num_envs": args.num_envs,
                "horizon": 16,
                "transitions": rows[-1]["total_steps"],
                "process_wall_seconds": process_seconds,
                "training_seconds": elapsed,
                "rollout_seconds": sum(row["rollout_seconds"] for row in rows),
                "update_seconds": sum(row["update_seconds"] for row in rows),
                "warmup_seconds": sum(row["warmup_seconds"] for row in rows),
                "steady_fps": len(steady) * args.num_envs * 16 / sum(row["iteration_seconds"] for row in steady),
                "parameter_count": (profile["actor_parameters"] + profile["critic_parameters"]) if profile else None,
                "actor_gradient_updates": rows[-1]["actor_gradient_updates"],
                "critic_gradient_updates": rows[-1]["critic_gradient_updates"],
                "peak_torch_memory_mib": max((row.get("torch_peak_allocated_bytes", 0) for row in rows), default=0)
                / 2**20
                or None,
                "training_curve": curve,
                "evaluations": [],
            }
        )
        write_results(manifest, results)
        print(
            f"Finished {name}: loop={elapsed:.2f}s, process={process_seconds:.2f}s, "
            f"steady={results['runs'][-1]['steady_fps']:.0f} transitions/s",
            flush=True,
        )


def evaluate_policy(env, policy, *, tensor_dict: bool, seed: int) -> dict:
    """Evaluate one full deterministic episode per environment from the same resets."""
    import torch
    from tensordict import TensorDict

    observations, _ = env.reset(seed=seed)
    active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    returns = torch.zeros(env.num_envs, device=env.device)
    lengths = torch.zeros_like(returns)
    upright_steps = torch.zeros_like(returns)
    survived = torch.zeros_like(active)
    pole_index = env.scene["robot"].find_joints("cart_to_pole")[0][0]
    for _ in range(env.max_episode_length):
        inputs = TensorDict(observations, batch_size=[env.num_envs]) if tensor_dict else observations
        observations, rewards, terminated, truncated, extras = env.step(policy(inputs).clamp(-1, 1))
        terminal = extras.get("final_obs", observations)["policy"]
        state = torch.where((terminated | truncated)[:, None], terminal, observations["policy"])
        returns += rewards * active
        lengths += active
        upright_steps += (state[:, pole_index].abs() < 0.2) * active
        survived |= active & truncated & ~terminated
        active &= ~(terminated | truncated)
        if not active.any():
            break
    mean = float(returns.mean())
    return {
        "return_mean": mean,
        "return_std": float(returns.std(unbiased=False)),
        "normalized_score": 100 * mean / (env.max_episode_length * env.step_dt),
        "episode_length_mean": float(lengths.mean()),
        "survival_rate": float(survived.float().mean()),
        "upright_fraction": float((upright_steps / lengths).mean()),
    }


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate stored checkpoints using a common unmodified Cartpole MDP."""
    import gymnasium as gym
    import torch

    from isaaclab.app import launch_simulation
    from isaaclab.utils.io import load_yaml

    from isaaclab_rl.robolearn import RoboLearnRunner

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    manifest = args.output / "comparison.json"
    results = json.loads(manifest.read_text())
    flash_note = "FlashSAC uses eager float32 updates in this quick run; compilation and AMP are disabled."
    if flash_note not in results["metadata"]["notes"]:
        results["metadata"]["notes"].append(flash_note)
    env_cfg = parse_env_cfg(
        "Isaac-Cartpole", device="cuda:0", num_envs=args.eval_envs, overrides=["physics=newton_mjwarp"]
    )
    env_cfg.compute_final_obs = True
    with launch_simulation(env_cfg):
        env = gym.make("Isaac-Cartpole", cfg=env_cfg).unwrapped
        try:
            for run in results["runs"]:
                log_dir = Path(run["log_dir"])
                is_rsl = run["algorithm"] == "rsl_rl_ppo"
                if is_rsl:
                    from rsl_rl.runners import OnPolicyRunner

                    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

                    cfg = load_yaml(log_dir / "params/agent.yaml")
                    runner = OnPolicyRunner(
                        RslRlVecEnvWrapper(env, clip_actions=1.0), cfg, log_dir=None, device=env.device
                    )
                    checkpoints = sorted(log_dir.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[-1]))
                else:
                    checkpoints = sorted(log_dir.glob("model_*.json"), key=lambda path: int(path.stem.split("_")[-1]))
                    cfg = json.loads(checkpoints[-1].read_text())["config"]
                    runner = RoboLearnRunner(env, cfg, device=env.device)
                    if run["algorithm"] == "flashsac":
                        run["algorithm_config"] = asdict(runner.agent._cfg)
                        run["parameter_count"] = sum(
                            p.numel()
                            for net in (runner.agent._actor, runner.agent._critic)
                            for p in net.network.parameters()
                        )
                    else:
                        run["algorithm_config"] = asdict(runner.agent.config)
                        run["parameter_count"] = sum(math.prod(p.shape) for p in runner.agent.parameters)
                rows = [json.loads(line) for line in (log_dir / "metrics.jsonl").read_text().splitlines()]
                run["evaluations"] = []
                run["agent_config"] = cfg
                for checkpoint in checkpoints:
                    runner.load(str(checkpoint))
                    # RSL checkpoint indices are zero-based completed iteration indices.
                    iteration = int(checkpoint.stem.split("_")[-1]) + int(is_rsl)
                    with torch.inference_mode():
                        evaluation = evaluate_policy(
                            env,
                            runner.get_inference_policy(device=env.device),
                            tensor_dict=is_rsl,
                            seed=results["metadata"]["seed"] + 10_000,
                        )
                    evaluation.update(
                        iteration=iteration,
                        transitions=iteration * run["num_envs"] * run["horizon"],
                        time_seconds=sum(row["iteration_seconds"] for row in rows[:iteration]),
                    )
                    run["evaluations"].append(evaluation)
                    print(
                        f"{run['name']} iteration={iteration} return={evaluation['return_mean']:.4f} "
                        f"score={evaluation['normalized_score']:.1f}%",
                        flush=True,
                    )
                run["final_checkpoint"] = str(checkpoints[-1])
                write_results(manifest, results)
                del runner
                torch.cuda.empty_cache()
        finally:
            env.close()
    print(f"Results: {manifest.resolve()}", flush=True)


def main() -> None:
    """Parse a reproducible train or evaluation phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["train", "evaluate"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--eval_envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["flashsac", "warp_ppo", "rsl_rl_ppo"],
        default=["flashsac", "warp_ppo", "rsl_rl_ppo"],
        help="Training algorithms to run serially; evaluation uses the recorded manifest.",
    )
    args = parser.parse_args()
    if min(args.iterations, args.num_envs, args.eval_envs) < 1:
        parser.error("Iteration and environment counts must be positive.")
    (train if args.phase == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
