# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate held-out action-conditioned rollouts without selecting on test results.

Prediction errors and admissibility checks concern recorded observations, not
unobserved full particle trajectories. Passing these checks does not establish
accurate constitutive forces, successful pouring, or zero-shot task transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .dataset import EpisodeWindows, load_episodes, task_specs
from .models import load_checkpoint, quaternion_geodesic_error


def _quaternion_groups(task_id):
    if task_id == "franka_pour":
        return (19, 26, 33, 40), "xyzw"
    if task_id == "rizon_sharpa_teapot_held_pour":
        return (61, 74), "wxyz"
    return (), "wxyz"


def _ensemble_mean_observation(observations, *, task_id):
    """Align equivalent quaternion signs before averaging, without projection."""
    stacked = torch.stack(observations)
    mean = stacked.mean(0)
    starts, _ = _quaternion_groups(task_id)
    for start in starts:
        quaternions = stacked[..., start : start + 4]
        sign = torch.where((quaternions * quaternions[:1]).sum(-1, keepdim=True) < 0.0, -1.0, 1.0)
        mean[..., start : start + 4] = (quaternions * sign).mean(0)
    return mean


def physical_observation_errors(observation, *, task_id, action=None):
    """Measure task-specific physical violations without projecting predictions.

    Fractions are dimensionless, kinetic energy is in joules, speed in m/s,
    covariance in m², and quaternion error is dimensionless. Indices are the
    unchanged Franka interface and the explicit held-teapot prototype interface.
    """
    finite = torch.isfinite(observation).all(-1)
    safe = torch.where(finite[..., None], observation, 0.0)
    result = {"nonfinite_observation": (~finite).float()}
    if task_id == "franka_pour":
        fractions = safe[..., [68, 69, 70, 83]]
        quaternions = torch.stack([safe[..., index : index + 4] for index in (19, 26, 33, 40)], -2)
        if action is not None:
            result["last_action_absolute_error"] = (safe[..., 54:62] - 0.2 * action).abs().mean(-1)
    elif task_id == "rizon_sharpa_teapot_held_pour":
        fractions = safe[..., 91:96]
        quaternions = torch.stack((safe[..., 61:65], safe[..., 74:78]), -2)
        result["negative_kinetic_energy"] = (safe[..., 90] < -1e-6).float()
        result["negative_speed"] = ((safe[..., 96:98] < -1e-6).any(-1)).float()
        result["rms_exceeds_max_speed"] = (safe[..., 96] > safe[..., 97] + 1e-4).float()
        result["mean_exceeds_rms_speed"] = (safe[..., 87:90].norm(dim=-1) > safe[..., 96] + 1e-4).float()
        result["reversed_z_extrema"] = (safe[..., 98] > safe[..., 99] + 1e-4).float()
        xx, yy, zz, xy, xz, yz = safe[..., 100:106].unbind(-1)
        covariance = torch.stack((xx, xy, xz, xy, yy, yz, xz, yz, zz), -1).reshape(*safe.shape[:-1], 3, 3)
        result["non_psd_position_covariance"] = (torch.linalg.eigvalsh(covariance).amin(-1) < -1e-7).float()
        if action is not None:
            result["last_action_absolute_error"] = (safe[..., 112:118] - action).abs().mean(-1)
    else:
        return result
    result["mass_fraction_out_of_bounds"] = ((fractions < -1e-5) | (fractions > 1.0 + 1e-5)).any(-1).float()
    violation = torch.maximum((-fractions).clamp_min(0.0), (fractions - 1.0).clamp_min(0.0))
    result["mass_fraction_bound_violation_max"] = violation.amax(-1)
    result["mass_fraction_bound_violation_mean_square"] = violation.square().mean(-1)
    result["mass_fraction_sum_absolute_error"] = (fractions.sum(-1) - 1.0).abs()
    quaternion_error = (quaternions.norm(dim=-1) - 1.0).abs()
    result["quaternion_norm_absolute_error"] = quaternion_error.mean(-1)
    result["nonunit_quaternion"] = (quaternion_error > 0.01).any(-1).float()
    return result


def _severity_summary(values):
    values = torch.cat(values) if values else torch.empty(0)
    if not len(values):
        return {"count": 0, "minimum": None, "maximum": None, "rms": None, "p50": None, "p95": None, "p99": None}
    quantiles = values.quantile(torch.tensor([0.5, 0.95, 0.99])).tolist()
    return dict(
        count=len(values),
        minimum=float(values.min()),
        maximum=float(values.max()),
        rms=float(values.square().mean().sqrt()),
        **dict(zip(("p50", "p95", "p99"), quantiles)),
    )


def _statistics():
    return {
        "squared_error": 0.0,
        "normalized_squared_error": 0.0,
        "elements": 0.0,
        "nonfinite_predictions": 0,
        "component_squared_error": None,
        "component_count": 0.0,
        "quaternion_angular_squared_error": 0.0,
        "quaternion_angular_count": 0,
        "undefined_quaternion_angle_count": 0,
    }


def _accumulate(statistics, prediction, target, scale, mask, *, task_id=None):
    finite = torch.isfinite(prediction).all(-1)
    statistics["nonfinite_predictions"] += int((~finite & mask.bool()).sum())
    valid = mask * finite
    error = torch.where(finite[..., None], prediction - target, 0.0)
    squared = error.square() * valid[..., None]
    statistics["squared_error"] += float(squared.sum())
    statistics["normalized_squared_error"] += float((squared / scale.square()).sum())
    statistics["elements"] += float(valid.sum()) * prediction.shape[-1]
    component = squared.sum(0).cpu()
    if statistics["component_squared_error"] is None:
        statistics["component_squared_error"] = component
    else:
        statistics["component_squared_error"] += component
    statistics["component_count"] += float(valid.sum())
    starts, order = _quaternion_groups(task_id)
    if starts:
        predicted_quat = torch.stack([prediction[..., i : i + 4] for i in starts], -2)
        target_quat = torch.stack([target[..., i : i + 4] for i in starts], -2)
        angular_valid = valid.bool()[..., None] & (predicted_quat.norm(dim=-1) > 1e-12)
        angular_valid &= target_quat.norm(dim=-1) > 1e-12
        statistics["undefined_quaternion_angle_count"] += int((valid.bool()[..., None] & ~angular_valid).sum())
        angle = quaternion_geodesic_error(predicted_quat[angular_valid], target_quat[angular_valid], order=order)
        statistics["quaternion_angular_squared_error"] += float(angle.square().sum())
        statistics["quaternion_angular_count"] += int(angular_valid.sum())


def _finish(statistics):
    count = max(1.0, statistics["elements"])
    component_count = max(1.0, statistics["component_count"])
    return {
        "observation_rmse": (statistics["squared_error"] / count) ** 0.5,
        "train_std_normalized_observation_rmse": (statistics["normalized_squared_error"] / count) ** 0.5,
        "observation_rmse_by_component": (statistics["component_squared_error"] / component_count).sqrt().tolist(),
        "finite_prediction_frames": statistics["component_count"],
        "nonfinite_prediction_frames": statistics["nonfinite_predictions"],
        "error_scope": (
            "Errors are over finite predictions; nonfinite frames are counted separately and invalidate a run"
        ),
        "quaternion_angular_rmse_rad": (
            statistics["quaternion_angular_squared_error"] / statistics["quaternion_angular_count"]
        )
        ** 0.5
        if statistics["quaternion_angular_count"]
        else None,
        "quaternion_angular_count": statistics["quaternion_angular_count"],
        "undefined_quaternion_angle_count": statistics["undefined_quaternion_angle_count"],
        "quaternion_angle_scope": "Shortest sign-invariant angle; vector norm errors are reported separately",
    }


def _physical_indices(task_id, dimension):
    if task_id == "franka_pour":
        return {
            "dynamics": [i for i in range(84) if i not in (14, 15) and not 54 <= i < 62],
            "material": list(range(68, 84)),
        }
    if task_id == "rizon_sharpa_teapot_held_pour":
        return {"dynamics": list(range(106)), "material": list(range(84, 106))}
    return {"dynamics": list(range(dimension)), "material": []}


def _subset_metrics(metrics, subsets, scale):
    components = torch.as_tensor(metrics["observation_rmse_by_component"], dtype=torch.float64)
    scale = scale.detach().cpu().double()
    for name, indices in subsets.items():
        if not indices or metrics["finite_prediction_frames"] == 0:
            metrics[f"{name}_observation_rmse"] = None
            metrics[f"train_std_normalized_{name}_rmse"] = None
            continue
        selected = components[indices]
        metrics[f"{name}_observation_rmse"] = float(selected.square().mean().sqrt())
        metrics[f"train_std_normalized_{name}_rmse"] = float((selected / scale[indices]).square().mean().sqrt())
    return metrics


@torch.no_grad()
def evaluate_task(models, episodes, *, task_id, split, horizon, batches, batch_size, device, seed):
    """Compare the original action sequence, shuffled actions and static persistence."""
    windows = EpisodeWindows(episodes, task_id=task_id, split=split, horizon=horizon, seed=seed)
    scale = models[0].observation_scale(task_id)
    statistics = {key: _statistics() for key in ("model", "shuffled_action", "persistence")}
    physics_sum, reference_physics_sum = {}, {}
    bound_severity, reference_bound_severity = [], []
    member_statistics = [_statistics() for _ in models]
    member_physics_sum = [{} for _ in models]
    member_physical_count = [0.0 for _ in models]
    member_bound_severity = [[] for _ in models]
    physical_count = reference_physical_count = 0.0
    valid_steps = [0.0] * horizon
    endpoint_statistics = {key: _statistics() for key in statistics}
    reward_squared_error = reward_count = termination_brier_sum = disagreement_sum = 0.0
    for _ in range(batches):
        batch = windows.sample(batch_size, device=device)
        permutation = torch.randperm(batch_size, generator=windows.generator).to(device)
        states = {}
        for branch in ("model", "shuffled_action"):
            states[branch] = [
                (
                    batch["observation"][:, 0],
                    batch["critic_observation"][:, 0],
                    model.encode_batch(batch, task_id=task_id),
                )
                for model in models
            ]
        for step in range(horizon):
            target, mask = batch["observation"][:, step + 1], batch["transition_mask"][:, step]
            predictions = {}
            selected_reward = selected_probability = selected_results = None
            for branch, branch_actions in (
                ("model", batch["action"]),
                ("shuffled_action", batch["action"][permutation]),
            ):
                results = []
                for member, model in enumerate(models):
                    observation, critic, hidden = states[branch][member]
                    result = model.step(
                        observation, critic, branch_actions[:, step], batch["context"], hidden, task_id=task_id
                    )
                    states[branch][member] = (result["observation"], result["critic_observation"], result["hidden"])
                    results.append(result)
                predictions[branch] = _ensemble_mean_observation(
                    [result["observation"] for result in results], task_id=task_id
                )
                if branch == "model":
                    selected_results = results
                    selected_reward = torch.stack([result["reward"] for result in results]).mean(0)
                    selected_probability = torch.stack(
                        [result["termination_logit"].sigmoid() for result in results]
                    ).mean(0)
                    disagreement = (
                        (torch.stack([result["observation"] for result in results]).std(0, unbiased=False) / scale)
                        .square()
                        .mean(-1)
                        .sqrt()
                    )
                    disagreement_sum += float((torch.nan_to_num(disagreement) * mask).sum())
            predictions["persistence"] = batch["observation"][:, 0]
            for branch, prediction in predictions.items():
                _accumulate(statistics[branch], prediction, target, scale, mask, task_id=task_id)
                if step == horizon - 1:
                    _accumulate(endpoint_statistics[branch], prediction, target, scale, mask, task_id=task_id)
            valid_steps[step] += float(mask.sum())
            reward_finite = torch.isfinite(selected_reward) & torch.isfinite(selected_probability)
            reward_mask = mask * reward_finite
            reward_error = torch.where(reward_finite, selected_reward - batch["reward"][:, step], 0.0)
            termination_error = torch.where(
                reward_finite, selected_probability - batch["terminated"][:, step].float(), 0.0
            )
            reward_squared_error += float((reward_error.square() * reward_mask).sum())
            termination_brier_sum += float((termination_error.square() * reward_mask).sum())
            reward_count += float(reward_mask.sum())
            prediction = predictions["model"]
            finite_prediction_mask = mask * torch.isfinite(prediction).all(-1)
            physical = physical_observation_errors(prediction, task_id=task_id, action=batch["action"][:, step])
            reference_physical = physical_observation_errors(target, task_id=task_id, action=batch["action"][:, step])
            if "mass_fraction_bound_violation_max" in physical:
                bound_severity.append(
                    physical["mass_fraction_bound_violation_max"][finite_prediction_mask.bool()].cpu()
                )
                reference_bound_severity.append(
                    reference_physical["mass_fraction_bound_violation_max"][mask.bool()].cpu()
                )
            for name, values in physical.items():
                # Keep nonfinite incidence unconditional; other checks have a
                # meaningful interpretation only on finite state vectors.
                weight = mask if name == "nonfinite_observation" else finite_prediction_mask
                physics_sum[name] = physics_sum.get(name, 0.0) + float((values * weight).sum())
            for name, values in reference_physical.items():
                reference_physics_sum[name] = reference_physics_sum.get(name, 0.0) + float((values * mask).sum())
            physical_count += float(finite_prediction_mask.sum())
            reference_physical_count += float(mask.sum())
            for member, result in enumerate(selected_results):
                observation = result["observation"]
                _accumulate(member_statistics[member], observation, target, scale, mask, task_id=task_id)
                member_finite = mask * torch.isfinite(observation).all(-1)
                member_physics = physical_observation_errors(
                    observation, task_id=task_id, action=batch["action"][:, step]
                )
                for name, values in member_physics.items():
                    weight = mask if name == "nonfinite_observation" else member_finite
                    member_physics_sum[member][name] = member_physics_sum[member].get(name, 0.0) + float(
                        (values * weight).sum()
                    )
                if "mass_fraction_bound_violation_max" in member_physics:
                    member_bound_severity[member].append(
                        member_physics["mass_fraction_bound_violation_max"][member_finite.bool()].cpu()
                    )
                member_physical_count[member] += float(member_finite.sum())
    subsets = _physical_indices(task_id, len(scale))
    finished = {name: _subset_metrics(_finish(value), subsets, scale) for name, value in statistics.items()}
    endpoint = {name: _subset_metrics(_finish(value), subsets, scale) for name, value in endpoint_statistics.items()}
    original_error = finished["model"]["train_std_normalized_observation_rmse"]
    shuffled_error = finished["shuffled_action"]["train_std_normalized_observation_rmse"]
    return {
        "horizon": horizon,
        "windows": batches * batch_size,
        "valid_transition_count_by_step": valid_steps,
        "rollout_average": finished,
        "endpoint": endpoint,
        "shuffled_to_original_error_ratio": shuffled_error / max(original_error, 1e-12),
        "physical_action_control_error_ratios": {
            name: (
                finished["shuffled_action"][f"train_std_normalized_{name}_rmse"]
                / finished["model"][f"train_std_normalized_{name}_rmse"]
            )
            if finished["model"][f"train_std_normalized_{name}_rmse"] is not None
            and finished["model"][f"train_std_normalized_{name}_rmse"] > 1e-12
            else None
            for name in subsets
        },
        "observation_subsets": subsets,
        "physical_action_control_note": (
            "Physical subsets exclude previous actions, clocks and commands: copying known action memory alone "
            "cannot establish a learned material response. Ratios are undefined when original error is zero."
        ),
        "action_control_note": (
            "A higher shuffled-action error supports action use, not accuracy outside recorded action coverage"
        ),
        "reward_rmse": (reward_squared_error / max(1, reward_count)) ** 0.5,
        "termination_brier_score": termination_brier_sum / max(1, reward_count),
        "mean_train_normalized_ensemble_disagreement": disagreement_sum / max(1, sum(valid_steps)),
        "ensemble_mean_physical_observables": {
            key: value / max(1, reference_physical_count if key == "nonfinite_observation" else physical_count)
            for key, value in physics_sum.items()
        },
        "per_member_diagnostics": [
            {
                "member": member,
                "rollout_average": _subset_metrics(_finish(member_statistics[member]), subsets, scale),
                "physical_observables": {
                    key: value
                    / max(
                        1, reference_physical_count if key == "nonfinite_observation" else member_physical_count[member]
                    )
                    for key, value in member_physics_sum[member].items()
                },
                "mass_fraction_bound_violation_severity": _severity_summary(member_bound_severity[member]),
            }
            for member in range(len(models))
        ],
        "reference_physical_observables": {
            key: value / max(1, reference_physical_count) for key, value in reference_physics_sum.items()
        },
        "ensemble_mean_mass_fraction_bound_violation_severity": _severity_summary(bound_severity),
        "reference_mass_fraction_bound_violation_severity": _severity_summary(reference_bound_severity),
        "mass_fraction_bound_severity_scope": (
            "Dimensionless maximum violation across bins per finite frame; quantiles include zero-violation frames. "
            "The reported mean square averages bins and frames; take its square root for RMS violation."
        ),
        "ensemble_aggregation": (
            "Quaternion signs align to member0 before arithmetic averaging; no norm projection. "
            "Ensemble-mean quaternion norms can depart from one even when each member is valid. "
            "Grade typed constraints using per_member_diagnostics; learned_env uses a selected member per environment."
        ),
        "physical_scope": "Fractions, moments and poses in observations; no force or full-particle validation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", nargs="+")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--horizons", type=int, nargs="+", default=(1, 4, 8, 16))
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    if min(*args.horizons, args.batches, args.batch_size, args.threads) < 1:
        parser.error("Evaluation horizons and budgets must be positive")
    torch.set_num_threads(args.threads)
    models, checkpoint = load_checkpoint(args.checkpoint, device=args.device)
    episodes = [episode for root in args.data for episode in load_episodes(root)]
    tasks = args.task or [spec.task_id for spec in models[0].config.tasks]
    observed_specs = {spec.task_id: spec for spec in task_specs(episodes)}
    for spec in models[0].config.tasks:
        if spec.task_id not in tasks:
            continue
        actual = observed_specs.get(spec.task_id)
        dimensions = ("observation_dim", "critic_observation_dim", "action_dim", "context_dim", "controller_dim")
        if actual is None or any(getattr(actual, name) != getattr(spec, name) for name in dimensions):
            raise ValueError(f"Recorded interface differs from checkpoint task {spec.task_id}")
    run = checkpoint.get("run", {})
    for task_id in tasks:
        trained_groups = set(run.get("split_groups", {}).get(task_id, {}).get("train", []))
        for episode in episodes:
            if episode["task_id"] != task_id or episode["split"] != args.split:
                continue
            if episode["reset_group"] in trained_groups:
                raise ValueError("Requested held-out evaluation includes a checkpoint training reset group")
            for field in ("context_names", "observation_names", "action_names"):
                expected_names = run.get("task_metadata", {}).get(task_id, {}).get(field, [])
                recorded_names = episode.get(field, episode.get("metadata", {}).get(field, []))
                if expected_names != recorded_names:
                    raise ValueError(f"{field} ordering differs from checkpoint for {task_id}")
    result = {
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "checkpoint_step": checkpoint.get("step"),
        "arguments": vars(args),
        "tasks": {},
        "checkpoint_selection": "This evaluator never changes weights or selects checkpoints",
    }
    for task_id in tasks:
        result["tasks"][task_id] = {
            str(horizon): evaluate_task(
                models,
                episodes,
                task_id=task_id,
                split=args.split,
                horizon=horizon,
                batches=args.batches,
                batch_size=args.batch_size,
                device=args.device,
                seed=args.seed,
            )
            for horizon in args.horizons
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "tasks": tasks, "horizons": args.horizons}), flush=True)


if __name__ == "__main__":
    main()
