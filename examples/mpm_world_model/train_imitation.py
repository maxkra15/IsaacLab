# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clone recorded teacher actions into an ordinary RSL-RL policy.

This is supervised initialization, not reinforcement learning or evidence of a
successful native task. Only valid training transitions fit actor weights and
observation normalization. Validation action error selects the actor; the test
split is never evaluated. Native closed-loop MPM replay remains essential.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from rsl_rl.algorithms import PPO

from .dataset import load_episodes, task_specs
from .learned_env import observation_groups
from .recording import atomic_json
from .teapot_full_task import FULL_TASK_ID
from .train_policy import ActionBoundedPolicy, default_runner_config


def _transitions(episodes, split, action_limit):
    observations, critics, actions, previous_actions, persistence_targets = [], [], [], [], []
    counts = {"recorded": 0, "valid": 0, "episodes": 0}
    for episode in episodes:
        if episode["split"] != split:
            continue
        counts["episodes"] += 1
        action = episode["action"]
        valid = episode["valid_mask"][:-1] & episode["valid_mask"][1:]
        valid &= torch.isfinite(action).all(-1) & torch.isfinite(episode["reward"])
        counts["recorded"] += len(action)
        counts["valid"] += int(valid.sum())
        if (action[valid].abs() > action_limit + 1.0e-6).any():
            raise ValueError("Valid demonstration actions exceed the declared executed action limit")
        observations.append(episode["observation"][:-1][valid])
        critics.append(episode["critic_observation"][:-1][valid])
        actions.append(action[valid])
        # This baseline uses the previous recorded action in the same episode;
        # it never crosses a reset or a numerically/capacity-invalid transition.
        paired = valid[1:] & valid[:-1]
        previous_actions.append(action[:-1][paired])
        persistence_targets.append(action[1:][paired])
    if not actions or counts["valid"] == 0:
        raise ValueError(f"Imitation requires finite, capacity-valid {split} transitions")
    return {
        "observation": torch.cat(observations),
        "critic_observation": torch.cat(critics),
        "action": torch.cat(actions),
        "previous_action": torch.cat(previous_actions),
        "persistence_target": torch.cat(persistence_targets),
        "counts": counts,
    }


@torch.no_grad()
def _evaluate(actor, data, task, device, batch_size, action_limit, train_action_mean, seed):
    actor.eval()
    predictions = []
    for start in range(0, len(data["action"]), batch_size):
        stop = start + batch_size
        obs = observation_groups(
            data["observation"][start:stop].to(device),
            data["critic_observation"][start:stop].to(device),
            task,
        )
        raw = actor(obs)
        if not torch.isfinite(raw).all():
            raise RuntimeError("Nonfinite validation predictions invalidate the imitation run")
        predictions.append(raw.clamp(-action_limit, action_limit).cpu())
    prediction, target = torch.cat(predictions), data["action"]
    error = prediction - target
    permutation = torch.randperm(len(target), generator=torch.Generator().manual_seed(seed))
    persistence_count = len(data["persistence_target"])
    return {
        "action_rmse": float(error.square().mean().sqrt()),
        "action_rmse_per_dimension": error.square().mean(0).sqrt().tolist(),
        "action_permutation_rmse": float((prediction - target[permutation]).square().mean().sqrt()),
        "train_mean_action_rmse": float((target - train_action_mean).square().mean().sqrt()),
        "previous_action_rmse": (
            float((data["previous_action"] - data["persistence_target"]).square().mean().sqrt())
            if persistence_count
            else None
        ),
        "previous_action_transition_count": persistence_count,
        "transition_count": len(target),
    }


def train(args):
    """Fit an actor and export native RSL weights plus a bounded JIT policy."""
    if min(args.updates, args.batch_size, args.validate_every, args.threads) < 1:
        raise ValueError("Training and validation budgets must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if not 0.0 < args.action_limit <= 1.0:
        raise ValueError("Action limit must lie in (0, 1]")
    if args.observation_mode not in ("full", "teapot_reduced", "teapot_phase"):
        raise ValueError("Unknown imitation observation mode")
    if args.observation_mode != "full" and args.task != FULL_TASK_ID:
        raise ValueError("Masked observations belong to the full teapot task only")
    if args.task == "franka_pour" and (not args.runner_config or args.action_limit != 1.0):
        raise ValueError("Franka requires its native runner configuration and action limit of 1")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("provenance.json", "model_imitation.pt", "policy.pt")):
        raise FileExistsError(f"Refusing to overwrite imitation run: {output}")
    episodes, datasets, group_splits = [], [], {}
    for directory in args.data:
        root = Path(directory)
        manifest = json.loads((root / "manifest.json").read_text())
        if not manifest.get("complete", False):
            raise ValueError("Imitation datasets must have a completed collection manifest")
        loaded = load_episodes(root)
        episodes.extend(episode for episode in loaded if episode["task_id"] == args.task)
        datasets.append(
            {
                "directory": str(root.resolve()),
                "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
                "metadata": manifest.get("metadata", {}),
            }
        )
    for episode in episodes:
        group, split = episode["reset_group"], episode["split"]
        if group in group_splits and group_splits[group] != split:
            raise ValueError("Reset groups cannot cross imitation dataset splits")
        group_splits[group] = split
    # Reuse the task owner's native shape, controller, action and clock schema
    # before constructing a deployable actor from inferred array dimensions.
    task_specs(episodes)
    data = {split: _transitions(episodes, split, args.action_limit) for split in ("train", "validation")}
    interface = [
        (value["observation"].shape[1], value["critic_observation"].shape[1], value["action"].shape[1])
        for value in data.values()
    ]
    if interface[0] != interface[1]:
        raise ValueError("Training and validation interfaces must agree")
    train_data, validation_data = data["train"], data["validation"]
    cfg = json.loads(Path(args.runner_config).read_text()) if args.runner_config else default_runner_config()
    cfg = copy.deepcopy(cfg)
    cfg.update(multi_gpu=None, num_steps_per_env=1)
    cfg["torch_compile_mode"] = None
    cfg["algorithm"].update(learning_rate=args.learning_rate, schedule="fixed")
    seed_obs = observation_groups(
        train_data["observation"][:1].to(args.device), train_data["critic_observation"][:1].to(args.device), args.task
    )
    env = SimpleNamespace(num_envs=1, num_actions=interface[0][2], device=args.device)
    alg = PPO.construct_algorithm(seed_obs, env, cfg, args.device)
    excluded = []
    first_layer = None
    if args.observation_mode in ("teapot_reduced", "teapot_phase"):
        if cfg["obs_groups"]["actor"] != ["policy"] or interface[0][0] != 182:
            raise ValueError("Masked teapot modes require the native 182-value actor policy group")
        excluded = (
            [index for index in range(182) if index != 179]
            if args.observation_mode == "teapot_phase"
            else [*range(29, 58), *range(106, 166), *range(173, 179)]
        )
        first_layer = next(module for module in alg.actor.modules() if isinstance(module, torch.nn.Linear))
        if first_layer.in_features != 182:
            raise ValueError("Masked teapot modes require a native first linear layer with 182 inputs")
        with torch.no_grad():
            first_layer.weight[:, excluded] = 0
    alg.train_mode()
    with torch.no_grad():
        for start in range(0, len(train_data["action"]), args.batch_size):
            stop = start + args.batch_size
            obs = observation_groups(
                train_data["observation"][start:stop].to(args.device),
                train_data["critic_observation"][start:stop].to(args.device),
                args.task,
            )
            alg.actor.update_normalization(obs)
            alg.critic.update_normalization(obs)
    train_mean = train_data["action"].mean(0)
    initial = _evaluate(
        alg.actor, validation_data, args.task, args.device, args.batch_size, args.action_limit, train_mean, args.seed
    )
    best, best_update = initial, 0
    best_actor = copy.deepcopy(alg.actor.state_dict())
    optimizer = torch.optim.Adam(alg.actor.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    logs = []
    started = time.perf_counter()
    for update in range(1, args.updates + 1):
        alg.actor.train()
        rows = torch.randint(len(train_data["action"]), (args.batch_size,), generator=generator)
        obs = observation_groups(
            train_data["observation"][rows].to(args.device),
            train_data["critic_observation"][rows].to(args.device),
            args.task,
        )
        # Supervise the raw deterministic mean. Clamping during fitting could
        # remove the gradient needed to recover a saturated incorrect output.
        loss = (alg.actor(obs) - train_data["action"][rows].to(args.device)).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite imitation loss invalidates the run")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if first_layer is not None:
            first_layer.weight.grad[:, excluded] = 0
        torch.nn.utils.clip_grad_norm_(alg.actor.parameters(), cfg["algorithm"]["max_grad_norm"])
        optimizer.step()
        if first_layer is not None:
            # A checkpoint contains ordinary RSL weights. Exact zero columns
            # make the deployed actor independent of these fields without an
            # extra runtime wrapper or a changed observation interface.
            with torch.no_grad():
                first_layer.weight[:, excluded] = 0
        if update % args.validate_every == 0 or update == args.updates:
            metrics = _evaluate(
                alg.actor,
                validation_data,
                args.task,
                args.device,
                args.batch_size,
                args.action_limit,
                train_mean,
                args.seed,
            )
            entry = {"update": update, "train_batch_mse": float(loss.detach()), "validation": metrics}
            logs.append(entry)
            print(json.dumps(entry), flush=True)
            if metrics["action_rmse"] < best["action_rmse"]:
                best, best_update = metrics, update
                best_actor = copy.deepcopy(alg.actor.state_dict())
    alg.actor.load_state_dict(best_actor, strict=True)
    alg.eval_mode()
    seconds = time.perf_counter() - started
    provenance = {
        "arguments": vars(args),
        "datasets": datasets,
        "split_groups": {
            split: sorted(group for group, value in group_splits.items() if value == split)
            for split in ("train", "validation", "test")
        },
        "transition_counts": {split: value["counts"] for split, value in data.items()},
        "normalization_split": "train",
        "actor_fit_split": "train",
        "selection_split": "validation",
        "test_split_evaluated": False,
        "interface": {"actor": interface[0][0], "critic": interface[0][1], "actions": interface[0][2]},
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("train_imitation.py", "train_policy.py", "dataset.py", "learned_env.py")
        },
        "rsl_rl_version": importlib.metadata.version("rsl-rl-lib"),
        "action_limit": args.action_limit,
        "observation_mode": args.observation_mode,
        "excluded_observation_indices": excluded,
        "excluded_observation_fields": (
            [episodes[0]["observation_names"][index] for index in excluded] if excluded else []
        ),
        "observation_mode_semantics": (
            "Clock-only fixed-scene neural trajectory bootstrap: only elapsed-time input 179 is retained. "
            "All other first-layer columns are zeroed at initialization, in gradients and after each Adam step. "
            "This is not a reactive grasp policy or learned world-model physics. The 182->30 native/JIT interface "
            "is unchanged. Later ordinary PPO can unfreeze these columns unless separately constrained."
            if args.observation_mode == "teapot_phase"
            else (
                "Reduced diagnostic zeros joint velocity, controller/action history/gain and palm-twist "
                "first-layer columns at initialization and after each supervised step. Observation/action "
                "interfaces are unchanged. Later ordinary PPO can unfreeze these columns unless the same "
                "constraint is separately enforced."
                if excluded
                else "The actor receives every declared observation field."
            )
        ),
        "critic": "Weights are untrained; observation normalization uses training data only. PPO must learn values.",
        "optimizer": "Supervised Adam state is omitted; native PPO optimizer is fresh.",
        "stochastic_distribution": (
            "Only deterministic actor means are fitted; exploration scale retains runner initialization."
        ),
        "native_checkpoint_action_limit_requirement": (
            "Only policy.pt embeds clipping; raw checkpoint consumers must enforce action_limit."
        ),
        "interpretation": (
            "Offline action error is not a native closed-loop success result. Real MPM evaluation is required."
        ),
    }
    atomic_json(output / "provenance.json", provenance)
    atomic_json(output / "runner_cfg.json", cfg)
    torch.save(
        {**alg.save(), "iter": 0, "infos": {"imitation": provenance, "runner_cfg": cfg}}, output / "model_imitation.pt"
    )
    policy = ActionBoundedPolicy(alg.get_policy().as_jit().cpu().eval(), args.action_limit)
    torch.jit.script(policy).save(str(output / "policy.pt"))
    result = {
        "complete": True,
        "selected_update": best_update,
        "updates": args.updates,
        "seconds": seconds,
        "initial_validation": initial,
        "selected_validation": best,
        "validation_history": logs,
        "test_split_evaluated": False,
        "real_mpm_evaluation_required": True,
    }
    atomic_json(output / "training.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default="rizon_sharpa_teapot_full")
    parser.add_argument("--runner-config", help="Native RSL runner JSON, including actor and critic hidden dimensions")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--action-limit", type=float, default=1.0)
    parser.add_argument("--observation-mode", choices=("full", "teapot_reduced", "teapot_phase"), default="full")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
