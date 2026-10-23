# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train ordinary RSL-RL PPO on short learned rollouts, then evaluate in real MPM.

This uses PPO rather than SuperMimic's analytical policy gradients. Only the
coupled dynamics model is differentiable; preserving the existing policy and
checkpoint interface makes model-versus-MPM comparisons possible.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
from rsl_rl.algorithms import PPO

from .dataset import load_episodes
from .learned_env import LearnedParticleEnv, bootstrap_truncations
from .models import load_checkpoint
from .recording import atomic_json


class ActionBoundedPolicy(torch.nn.Module):
    """Export the executed actuator limit around an ordinary RSL JIT policy.

    The native RSL checkpoint retains the raw actor distribution and optimizer
    interface. This wrapper affects deterministic TorchScript deployment only.
    """

    def __init__(self, policy: torch.nn.Module, action_limit: float = 1.0):
        super().__init__()
        if not 0.0 < action_limit <= 1.0:
            raise ValueError("Action limit must lie in (0, 1]")
        self.policy = policy
        self.action_limit = float(action_limit)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.policy(observation).clamp(-self.action_limit, self.action_limit)


def default_runner_config():
    """Give new tasks a small policy while keeping Franka's exported config exact."""
    return {
        "obs_groups": {"actor": ["policy"], "critic": ["privileged"]},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 0.3,
                "std_type": "log",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": None,
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 2,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-4,
            "schedule": "fixed",
            "gamma": 0.99,
            "lam": 0.95,
            "entropy_coef": 0.001,
            "max_grad_norm": 1.0,
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
        },
        "torch_compile_mode": None,
    }


def train(args):
    """Fit a policy without reference futures and save native RSL and JIT actors."""
    if args.task == "franka_pour" and args.action_limit != 1.0:
        raise ValueError("Franka native RSL checkpoint consumers require the original action limit of 1")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model_final.pt").exists():
        raise FileExistsError(f"Refusing to overwrite policy run: {output}")
    models, model_checkpoint = load_checkpoint(args.model, device=args.device)
    episodes = []
    declared_manifests = {entry["manifest_sha256"] for entry in model_checkpoint.get("run", {}).get("data", [])}
    for directory in args.data:
        digest = hashlib.sha256((Path(directory) / "manifest.json").read_bytes()).hexdigest()
        if digest not in declared_manifests:
            raise ValueError("PPO replay manifests must match the dynamics model's recorded training datasets")
        episodes.extend(load_episodes(directory))
    declared_groups = model_checkpoint.get("run", {}).get("split_groups", {}).get(args.task)
    if declared_groups is None:
        raise ValueError("PPO requires a model checkpoint with auditable training reset groups")
    actual_groups = {e["reset_group"] for e in episodes if e["task_id"] == args.task and e["split"] == "train"}
    if not actual_groups.issubset(set(declared_groups["train"])):
        raise ValueError("PPO seed episodes must belong to the dynamics model's training reset groups")
    env = LearnedParticleEnv(
        models,
        episodes,
        task_id=args.task,
        num_envs=args.num_envs,
        horizon=args.horizon,
        seed_bank_size=args.seed_bank_size,
        uncertainty_limit=args.uncertainty_limit,
        uncertainty_penalty=args.uncertainty_penalty,
        mass_tolerance=args.mass_tolerance,
        action_limit=args.action_limit,
        device=args.device,
        seed=args.seed,
    )
    cfg = json.loads(Path(args.runner_config).read_text()) if args.runner_config else default_runner_config()
    if args.task == "franka_pour" and not args.runner_config:
        raise ValueError("Franka requires its exported native runner_cfg.json")
    cfg = copy.deepcopy(cfg)
    cfg["multi_gpu"], cfg["num_steps_per_env"] = None, args.rollout_steps
    cfg["algorithm"].update(learning_rate=args.learning_rate, num_learning_epochs=args.epochs)
    cfg["algorithm"]["schedule"] = "fixed"
    if args.task != "franka_pour":
        # Preserve the approximate time constants when the task controls at 100Hz.
        dt = float(next(episode for episode in episodes if episode["task_id"] == args.task).get("dt", 0.01))
        cfg["algorithm"]["gamma"], cfg["algorithm"]["lam"] = 0.99 ** (dt / 0.1), 0.95 ** (dt / 0.1)
    obs = env.get_observations()
    alg = PPO.construct_algorithm(obs, env, cfg, args.device)
    initial_iteration = 0
    if args.checkpoint:
        source = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        alg.load(source, {"actor": True, "critic": True}, strict=True)
        initial_iteration = int(source.get("iter", 0))
    alg.train_mode()
    provenance = {
        "arguments": vars(args),
        "runner_config": cfg,
        "model_sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
        "policy_source_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        if args.checkpoint
        else None,
        "reward_bounds": [env.reward_min, env.reward_max],
        "reset_split": "train",
        "seed_bank_size": args.seed_bank_size,
        "mass_admissibility_check_enabled": bool(env.mass_indices),
        "physical_decoder": getattr(models[0].config, "physical_decoder", "none"),
        "action_limit": env.action_limit,
        "action_semantics": (
            "Imagined environment and policy.pt clip executed actions to this limit; "
            "the RSL checkpoint retains its raw actor distribution"
        ),
        "native_checkpoint_action_limit_requirement": (
            "Native RSL checkpoint consumers must separately enforce this action_limit; "
            "only policy.pt embeds the clipping wrapper. Franka accepts action_limit=1 only."
        ),
        "algorithm": "RSL-RL PPO; no analytical policy gradient",
        "interpretation": "Model rewards are diagnostic; accept only real-MPM task results.",
    }
    atomic_json(output / "provenance.json", provenance)
    atomic_json(output / "runner_cfg.json", cfg)
    logs, total_transitions = [], 0
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    for iteration in range(args.iterations):
        step_started = time.perf_counter()
        reward_sum = truncations = terminations = nonfinite = uncertainty_sum = inadmissible = 0.0
        with torch.no_grad():
            for _ in range(args.rollout_steps):
                action = alg.act(obs)
                obs, reward, done, info = env.step(action)
                terminal_value = alg.critic(info["terminal_observations"]).squeeze(-1)
                corrected_reward = bootstrap_truncations(reward, info["truncated"], terminal_value, alg.gamma)
                # RSL's time_outs convention uses V(current). We already bootstrap
                # V(predicted endpoint), so do not pass time_outs a second time.
                alg.process_env_step(obs, corrected_reward, done, {})
                reward_sum += float(reward.sum())
                truncations += int(info["truncated"].sum())
                terminations += int(info["terminated"].sum())
                nonfinite += int(info["nonfinite"].sum())
                inadmissible += int(info["inadmissible"].sum())
                uncertainty_sum += float(info["disagreement"].nan_to_num().sum())
            alg.compute_returns(obs)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        rollout_seconds = time.perf_counter() - step_started
        losses = alg.update()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - step_started
        count = args.num_envs * args.rollout_steps
        total_transitions += count
        entry = {
            "iteration": iteration + 1,
            "transitions": count,
            "seconds": elapsed,
            "rollout_seconds": rollout_seconds,
            "ppo_transitions_per_second": count / elapsed,
            "model_transitions_per_second": count / rollout_seconds,
            "mean_predicted_reward": reward_sum / count,
            "truncations": truncations,
            "model_terminations": terminations,
            "nonfinite": nonfinite,
            "mass_admissibility_cutoffs": inadmissible,
            "mean_disagreement": uncertainty_sum / count,
            "loss": losses,
        }
        logs.append(entry)
        print(json.dumps(entry), flush=True)
        if nonfinite:
            raise RuntimeError("Nonfinite imagined transitions invalidate this policy run")
    total_seconds = time.perf_counter() - started
    infos = {"world_model": provenance, "runner_cfg": cfg}
    torch.save({**alg.save(), "iter": initial_iteration + args.iterations, "infos": infos}, output / "model_final.pt")
    policy = ActionBoundedPolicy(alg.get_policy().as_jit().cpu().eval(), args.action_limit)
    torch.jit.script(policy).save(str(output / "policy.pt"))
    result = {
        "complete": True,
        "iterations": logs,
        "total_transitions": total_transitions,
        "total_training_seconds": total_seconds,
        "ppo_transitions_per_second": total_transitions / total_seconds,
        "real_mpm_evaluation_required": True,
    }
    atomic_json(output / "training.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default="franka_pour")
    parser.add_argument("--runner-config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed-bank-size", type=int, default=4096)
    parser.add_argument("--uncertainty-limit", type=float, default=2.0)
    parser.add_argument("--uncertainty-penalty", type=float, default=0.0)
    parser.add_argument("--mass-tolerance", type=float, default=0.05)
    parser.add_argument("--action-limit", type=float, default=1.0, help="Executed and exported action bound in (0, 1]")
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.iterations, args.rollout_steps, args.epochs, args.threads) < 1 or args.learning_rate <= 0:
        parser.error("Training budgets and learning rate must be positive")
    if not 0.0 < args.action_limit <= 1.0:
        parser.error("Action limit must lie in (0, 1]")
    if args.task == "franka_pour" and args.action_limit != 1.0:
        parser.error("Franka native RSL checkpoint consumers require the original action limit of 1")
    train(args)


if __name__ == "__main__":
    main()
