#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a deterministic RJ45 policy uniformly across reset phases."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

from isaaclab_tasks.contrib.franka_rj45_insertion.config.franka.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    FrankaRJ45InsertionPPORunnerCfg,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.reset_dataset_io import (  # noqa: E402
    reset_dataset_digest,
    reset_dataset_validate_runtime,
    reset_validation_report_validate_runtime,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.rj45_env import (  # noqa: E402
    TERMINAL_OUTCOME_NAMES,
    FrankaRJ45InsertionEnv,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.rj45_env_cfg import (  # noqa: E402
    FrankaRJ45InsertionEnvCfg,
    reset_dataset_task_contract,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-content-sha256", required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--episodes-per-phase", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=142)
    add_launcher_args(parser)
    args = parser.parse_args()
    for name in ("num_envs", "episodes_per_phase", "max_steps"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if not args.dataset.is_file():
        parser.error(f"dataset does not exist: {args.dataset}")
    if not args.validation_report.is_file():
        parser.error(f"validation report does not exist: {args.validation_report}")
    if args.device is None:
        parser.error("--device is required")
    return args


@torch.inference_mode()
def _evaluate(args: argparse.Namespace) -> dict:
    dataset_payload = torch.load(args.dataset, map_location="cpu", weights_only=True)
    cfg = FrankaRJ45InsertionEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.reset_dataset_path = str(args.dataset.resolve())
    cfg.reset_validation_report_path = str(args.validation_report.resolve())
    cfg.reset_dataset_content_sha256 = args.dataset_content_sha256
    cfg.reset_dataset_sampling_mode = "uniform"
    cfg.curriculum_freeze = True
    cfg.validate_config()
    metadata, dataset_states, _ = reset_dataset_validate_runtime(
        dataset_payload,
        expected_content_sha256=args.dataset_content_sha256,
        expected_task_contract=reset_dataset_task_contract(cfg),
    )
    validation_report = json.loads(args.validation_report.read_text(encoding="utf-8"))
    validation_checks = reset_validation_report_validate_runtime(
        validation_report,
        expected_content_sha256=args.dataset_content_sha256,
        expected_row_count=len(dataset_states["phase"]),
        expected_task_contract=reset_dataset_task_contract(cfg),
    )
    phase_names = list(metadata.get("phase_names", []))
    phase_count = int(dataset_states["phase"].max().item()) + 1
    if len(phase_names) != phase_count:
        phase_names = [f"phase_{phase}" for phase in range(phase_count)]

    agent_cfg = FrankaRJ45InsertionPPORunnerCfg()
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))

    counts = [defaultdict(float) for _ in range(phase_count)]
    steps_taken = 0
    drive_disabled_check_count = 0
    from rsl_rl.runners import OnPolicyRunner

    with launch_simulation(cfg, args):
        env = FrankaRJ45InsertionEnv(cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        try:

            def assert_task_drive_disabled(stage: str) -> None:
                nonlocal drive_disabled_check_count
                drive_enabled = wp.to_torch(env._ensure_rj45_runtime().drive_enabled).to(dtype=torch.bool)
                drive_disabled_check_count += 1
                if bool(drive_enabled.any()):
                    enabled_ids = torch.where(drive_enabled)[0].detach().cpu().tolist()
                    raise RuntimeError(f"Task insertion drive became enabled at {stage} in envs {enabled_ids}.")

            assert_task_drive_disabled("initial reset")
            runner = OnPolicyRunner(
                wrapped,
                agent_cfg.to_dict(),
                log_dir=None,
                device=agent_cfg.device,
            )
            runner.load(str(args.checkpoint.resolve()), map_location=agent_cfg.device)
            policy = runner.get_inference_policy(device=env.device)
            observations = wrapped.get_observations()
            episode_return = torch.zeros(env.num_envs, device=env.device)
            episode_length = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
            required_terms = TERMINAL_OUTCOME_NAMES
            missing_terms = sorted(set(required_terms) - set(env.termination_manager.active_terms))
            if missing_terms:
                raise RuntimeError(f"Evaluation requires missing termination terms: {missing_terms}")

            while steps_taken < args.max_steps and any(
                int(phase_counts["episodes"]) < args.episodes_per_phase for phase_counts in counts
            ):
                actions = policy(observations)
                observations, rewards, dones, _ = wrapped.step(actions)
                policy.reset(dones)
                episode_return += rewards.reshape(-1)
                episode_length += 1
                steps_taken += 1

                done_ids = torch.where(dones.bool())[0]
                if done_ids.numel() == 0:
                    continue
                assert_task_drive_disabled(f"post-reset step {steps_taken}")
                terminal_rows = env.last_terminal_row_id[done_ids]
                if bool((terminal_rows < 0).any()):
                    raise RuntimeError("Autoreset did not preserve the completed reset-row identifiers.")
                term_values = {
                    name: env.last_terminal_outcomes[name][done_ids].detach().cpu() for name in required_terms
                }
                done_phases = env._reset_dataset_states["phase"][terminal_rows].detach().cpu()
                done_returns = episode_return[done_ids].detach().cpu()
                done_lengths = episode_length[done_ids].detach().cpu()
                for local_index, phase_tensor in enumerate(done_phases):
                    phase = int(phase_tensor.item())
                    if int(counts[phase]["episodes"]) >= args.episodes_per_phase:
                        continue
                    counts[phase]["episodes"] += 1
                    counts[phase]["return_sum"] += float(done_returns[local_index].item())
                    counts[phase]["length_sum"] += int(done_lengths[local_index].item())
                    active_outcomes = 0
                    for name in required_terms:
                        outcome = int(term_values[name][local_index].item())
                        counts[phase][name] += outcome
                        active_outcomes += outcome
                    counts[phase]["multiple_terminal_causes"] += int(active_outcomes > 1)
                episode_return[done_ids] = 0.0
                episode_length[done_ids] = 0
            assert_task_drive_disabled("evaluation completion")
        finally:
            wrapped.close()

    complete = all(int(phase_counts["episodes"]) >= args.episodes_per_phase for phase_counts in counts)
    per_phase = []
    totals = defaultdict(float)
    for phase, phase_counts in enumerate(counts):
        episodes = int(phase_counts["episodes"])
        row: dict[str, float | int | str] = {
            "phase": phase,
            "name": phase_names[phase],
            "episodes": episodes,
            "success_count": int(phase_counts["success"]),
            "lost_grasp_count": int(phase_counts["lost_grasp"]),
            "nonfinite_count": int(phase_counts["nonfinite"]),
            "task_out_of_bounds_count": int(phase_counts["task_out_of_bounds"]),
            "time_out_count": int(phase_counts["time_out"]),
            "multiple_terminal_causes_count": int(phase_counts["multiple_terminal_causes"]),
        }
        denominator = max(episodes, 1)
        for outcome in ("success", "lost_grasp", "nonfinite", "task_out_of_bounds", "time_out"):
            row[f"{outcome}_rate"] = float(phase_counts[outcome]) / denominator
            totals[outcome] += phase_counts[outcome]
        row["mean_return"] = float(phase_counts["return_sum"]) / denominator
        row["mean_episode_length"] = float(phase_counts["length_sum"]) / denominator
        totals["episodes"] += episodes
        totals["return_sum"] += phase_counts["return_sum"]
        totals["length_sum"] += phase_counts["length_sum"]
        totals["multiple_terminal_causes"] += phase_counts["multiple_terminal_causes"]
        per_phase.append(row)

    total_episodes = int(totals["episodes"])
    overall_denominator = max(total_episodes, 1)
    overall = {
        "episodes": total_episodes,
        "success_count": int(totals["success"]),
        "success_rate": float(totals["success"]) / overall_denominator,
        "lost_grasp_count": int(totals["lost_grasp"]),
        "lost_grasp_rate": float(totals["lost_grasp"]) / overall_denominator,
        "nonfinite_count": int(totals["nonfinite"]),
        "nonfinite_rate": float(totals["nonfinite"]) / overall_denominator,
        "task_out_of_bounds_count": int(totals["task_out_of_bounds"]),
        "task_out_of_bounds_rate": float(totals["task_out_of_bounds"]) / overall_denominator,
        "time_out_count": int(totals["time_out"]),
        "time_out_rate": float(totals["time_out"]) / overall_denominator,
        "multiple_terminal_causes_count": int(totals["multiple_terminal_causes"]),
        "mean_return": float(totals["return_sum"]) / overall_denominator,
        "mean_episode_length": float(totals["length_sum"]) / overall_denominator,
    }
    return {
        "format": "isaaclab-franka-rj45-policy-evaluation",
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "complete": complete,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "dataset": str(args.dataset.resolve()),
        "dataset_content_sha256": args.dataset_content_sha256,
        "task_contract_sha256": reset_dataset_digest(metadata["task_contract"]),
        "reset_validation": {
            "path": str(args.validation_report.resolve()),
            "sha256": _sha256(args.validation_report),
            "checks": validation_checks,
            "selected_row_count": validation_report["selected_row_count"],
            "goal_replay": validation_report["goal_replay"],
        },
        "evaluation_assumptions": {
            "fixed_goal_from_validated_artifact": True,
            "full_reset_dataset_was_physically_replayed": True,
            "reset_sampling_mode": "uniform",
            "adaptive_curriculum_frozen": True,
            "task_insertion_drive_disabled": True,
            "task_drive_disabled_check_count": drive_disabled_check_count,
            "policy_is_deterministic": True,
            "terminal_causes_captured_before_same_step_autoreset": True,
            "terminal_cause_counts_are_nonexclusive": True,
        },
        "seed": args.seed,
        "num_envs": args.num_envs,
        "episodes_per_phase": args.episodes_per_phase,
        "steps_taken": steps_taken,
        "max_steps": args.max_steps,
        "per_phase": per_phase,
        "overall": overall,
    }


def main() -> None:
    args = _parse_args()
    report = _evaluate(args)
    _write_json_atomic(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["complete"]:
        raise SystemExit("Evaluation did not collect the requested number of episodes for every phase.")


if __name__ == "__main__":
    main()
