#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a deterministic RJ45 pick-and-insert policy across all six reset phases."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

from isaaclab_tasks.contrib.franka_rj45_insertion.asset_provenance import (  # noqa: E402
    configured_franka_rj45_asset_closure,
    franka_rj45_asset_contract,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.config.franka.agents.pick_insert_rsl_rl_ppo_cfg import (  # noqa: E402
    FrankaRJ45PickInsertPPORunnerCfg,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_env import (  # noqa: E402
    FrankaRJ45PickInsertEnv,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_env_cfg import (  # noqa: E402
    PICK_INSERT_PHASE_NAMES,
    FrankaRJ45PickInsertEnvCfg,
    pick_insert_reset_dataset_task_contract,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_reset_dataset_io import (  # noqa: E402
    FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY,
    PICK_INSERT_RESET_PHASE_IDS,
    franka_rj45_validation_source_sha256,
    reset_dataset_digest,
    reset_dataset_validate_full_pick_diversity,
    reset_dataset_validate_phase_row_counts,
    reset_dataset_validate_runtime,
    reset_validation_report_validate_runtime,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.rj45_env import TERMINAL_OUTCOME_NAMES  # noqa: E402

_MINIMUM_EPISODES_PER_PHASE = 20
_STAGE_NAMES = ("approach", "grasp_acquired", "transport", "preinsertion", "inserted")


class _EvaluationFrankaRJ45PickInsertEnv(FrankaRJ45PickInsertEnv):
    """Preserve stateful pick diagnostics at the same-step autoreset boundary."""

    def _setup_after_physics(self) -> None:
        super()._setup_after_physics()
        self.evaluation_drive_ever_enabled = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._evaluation_initial_stage = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.last_terminal_pick_diagnostics_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.last_terminal_initial_stage = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.last_terminal_maximum_stage = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.last_terminal_ever_grasped = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.last_terminal_learning_progress = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _reset_idx(self, env_ids) -> None:
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if ids.numel() and hasattr(self, "evaluation_drive_ever_enabled"):
            drive_enabled = wp.to_torch(self._ensure_rj45_runtime().drive_enabled).to(
                device=self.device, dtype=torch.bool
            )
            self.evaluation_drive_ever_enabled[ids] |= drive_enabled[ids]
            self.last_terminal_pick_diagnostics_valid[ids] = False
            completed = (self.episode_length_buf[ids] > 0) & (self.reset_dataset_row_id[ids] >= 0)
            completed_ids = ids[completed]
            if completed_ids.numel():
                tracker = self.pick_insert_stage_tracker()
                progress = self.termination_manager.get_term_cfg("learning_progress_context").func
                self.last_terminal_initial_stage[completed_ids] = self._evaluation_initial_stage[completed_ids]
                self.last_terminal_maximum_stage[completed_ids] = tracker.maximum_stage[completed_ids]
                self.last_terminal_ever_grasped[completed_ids] = tracker.ever_grasped[completed_ids]
                self.last_terminal_learning_progress[completed_ids] = progress.ever_success[completed_ids]
                self.last_terminal_pick_diagnostics_valid[completed_ids] = True

        super()._reset_idx(env_ids)

        if ids.numel() and hasattr(self, "termination_manager"):
            self._evaluation_initial_stage[ids] = self.pick_insert_stage_tracker().maximum_stage[ids]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
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
    parser.add_argument("--episodes-per-phase", type=int, default=_MINIMUM_EPISODES_PER_PHASE)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=142)
    add_launcher_args(parser)
    args = parser.parse_args()
    for name in ("num_envs", "max_steps"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.episodes_per_phase < _MINIMUM_EPISODES_PER_PHASE:
        parser.error(f"--episodes-per-phase must be at least {_MINIMUM_EPISODES_PER_PHASE}")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if not args.dataset.is_file():
        parser.error(f"dataset does not exist: {args.dataset}")
    if not args.validation_report.is_file():
        parser.error(f"validation report does not exist: {args.validation_report}")
    if args.device is None:
        parser.error("--device is required")
    return args


def _new_phase_counts() -> list[defaultdict[str, float]]:
    return [defaultdict(float) for _ in PICK_INSERT_RESET_PHASE_IDS]


def _record_completed_episodes(
    counts: Sequence[defaultdict[str, float]],
    *,
    phase_ids: torch.Tensor,
    episode_returns: torch.Tensor,
    episode_lengths: torch.Tensor,
    terminal_outcomes: Mapping[str, torch.Tensor],
    starts_grasped: torch.Tensor,
    initial_stages: torch.Tensor,
    maximum_stages: torch.Tensor,
    ever_grasped: torch.Tensor,
    learning_progress: torch.Tensor,
    episodes_per_phase: int,
) -> int:
    """Record a terminal batch while capping each phase at the requested count."""
    batch_size = int(phase_ids.numel())
    one_dimensional = {
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "starts_grasped": starts_grasped,
        "initial_stages": initial_stages,
        "maximum_stages": maximum_stages,
        "ever_grasped": ever_grasped,
        "learning_progress": learning_progress,
        **terminal_outcomes,
    }
    malformed = {
        name: tuple(value.shape)
        for name, value in one_dimensional.items()
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (batch_size,)
    }
    if malformed:
        raise RuntimeError(f"Terminal evaluation tensors do not share shape ({batch_size},): {malformed}")
    if set(terminal_outcomes) != set(TERMINAL_OUTCOME_NAMES):
        raise RuntimeError(
            "Terminal evaluation outcomes do not match the task contract: "
            f"expected={list(TERMINAL_OUTCOME_NAMES)}, actual={sorted(terminal_outcomes)}"
        )

    recorded = 0
    for local_index, phase_tensor in enumerate(phase_ids):
        phase = int(phase_tensor.item())
        if phase not in PICK_INSERT_RESET_PHASE_IDS:
            raise RuntimeError(f"Completed episode referenced invalid reset phase {phase}.")
        phase_counts = counts[phase]
        if int(phase_counts["episodes"]) >= episodes_per_phase:
            continue

        initial_stage = int(initial_stages[local_index].item())
        maximum_stage = int(maximum_stages[local_index].item())
        if not 0 <= initial_stage < len(_STAGE_NAMES):
            raise RuntimeError(f"Completed episode had invalid initial stage {initial_stage}.")
        if not initial_stage <= maximum_stage < len(_STAGE_NAMES):
            raise RuntimeError(f"Completed episode had invalid stage progression {initial_stage}->{maximum_stage}.")

        active_outcomes = sum(int(bool(terminal_outcomes[name][local_index].item())) for name in TERMINAL_OUTCOME_NAMES)
        if active_outcomes == 0:
            raise RuntimeError("Completed episode had no recognized terminal cause.")

        began_grasped = bool(starts_grasped[local_index].item())
        grasped_at_least_once = bool(ever_grasped[local_index].item())
        phase_counts["episodes"] += 1
        phase_counts["return_sum"] += float(episode_returns[local_index].item())
        phase_counts["length_sum"] += int(episode_lengths[local_index].item())
        for name in TERMINAL_OUTCOME_NAMES:
            phase_counts[name] += int(bool(terminal_outcomes[name][local_index].item()))
        phase_counts["multiple_terminal_causes"] += int(active_outcomes > 1)
        phase_counts["starts_grasped"] += int(began_grasped)
        phase_counts["ever_grasped"] += int(grasped_at_least_once)
        if not began_grasped:
            phase_counts["grasp_acquisition_eligible"] += 1
            phase_counts["grasp_acquisition"] += int(grasped_at_least_once)
        phase_counts["initial_stage_sum"] += initial_stage
        phase_counts["maximum_stage_sum"] += maximum_stage
        phase_counts["stage_gain_sum"] += maximum_stage - initial_stage
        phase_counts["stage_advance"] += int(maximum_stage > initial_stage)
        phase_counts["learning_progress"] += int(bool(learning_progress[local_index].item()))
        phase_counts[f"maximum_stage_{maximum_stage}"] += 1
        recorded += 1
    return recorded


def _optional_rate(numerator: float, denominator: int) -> float | None:
    return float(numerator) / denominator if denominator else None


def _summarize_one(counts: Mapping[str, float]) -> dict[str, Any]:
    episodes = int(counts.get("episodes", 0.0))
    acquisition_eligible = int(counts.get("grasp_acquisition_eligible", 0.0))
    summary: dict[str, Any] = {
        "episodes": episodes,
        "success_count": int(counts.get("success", 0.0)),
        "lost_grasp_count": int(counts.get("lost_grasp", 0.0)),
        "nonfinite_count": int(counts.get("nonfinite", 0.0)),
        "task_out_of_bounds_count": int(counts.get("task_out_of_bounds", 0.0)),
        "time_out_count": int(counts.get("time_out", 0.0)),
        "multiple_terminal_causes_count": int(counts.get("multiple_terminal_causes", 0.0)),
        "starts_grasped_count": int(counts.get("starts_grasped", 0.0)),
        "ever_grasped_count": int(counts.get("ever_grasped", 0.0)),
        "grasp_acquisition_eligible_episodes": acquisition_eligible,
        "grasp_acquisition_count": int(counts.get("grasp_acquisition", 0.0)),
        "stage_advance_count": int(counts.get("stage_advance", 0.0)),
        "learning_progress_count": int(counts.get("learning_progress", 0.0)),
        "maximum_stage_histogram": {
            str(stage): int(counts.get(f"maximum_stage_{stage}", 0.0)) for stage in range(len(_STAGE_NAMES))
        },
    }
    for outcome in TERMINAL_OUTCOME_NAMES:
        summary[f"{outcome}_rate"] = _optional_rate(counts.get(outcome, 0.0), episodes)
    summary["multiple_terminal_causes_rate"] = _optional_rate(counts.get("multiple_terminal_causes", 0.0), episodes)
    summary["starts_grasped_rate"] = _optional_rate(counts.get("starts_grasped", 0.0), episodes)
    summary["ever_grasped_rate"] = _optional_rate(counts.get("ever_grasped", 0.0), episodes)
    summary["grasp_acquisition_rate"] = _optional_rate(counts.get("grasp_acquisition", 0.0), acquisition_eligible)
    summary["stage_advance_rate"] = _optional_rate(counts.get("stage_advance", 0.0), episodes)
    summary["learning_progress_rate"] = _optional_rate(counts.get("learning_progress", 0.0), episodes)
    summary["mean_return"] = _optional_rate(counts.get("return_sum", 0.0), episodes)
    summary["mean_episode_length"] = _optional_rate(counts.get("length_sum", 0.0), episodes)
    summary["mean_initial_stage"] = _optional_rate(counts.get("initial_stage_sum", 0.0), episodes)
    summary["mean_maximum_stage"] = _optional_rate(counts.get("maximum_stage_sum", 0.0), episodes)
    summary["mean_stage_gain"] = _optional_rate(counts.get("stage_gain_sum", 0.0), episodes)
    return summary


def _summarize_counts(
    counts: Sequence[Mapping[str, float]],
    phase_names: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return phase-balanced summaries and their aggregate."""
    if len(counts) != len(PICK_INSERT_RESET_PHASE_IDS) or len(phase_names) != len(counts):
        raise ValueError("Evaluation summaries require exactly six phase counters and names.")
    totals: defaultdict[str, float] = defaultdict(float)
    per_phase: list[dict[str, Any]] = []
    for phase, phase_counts in enumerate(counts):
        row = {"phase": phase, "name": phase_names[phase], **_summarize_one(phase_counts)}
        per_phase.append(row)
        for name, value in phase_counts.items():
            totals[name] += value
    return per_phase, _summarize_one(totals)


@torch.inference_mode()
def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    configured_franka_rj45_asset_closure(required=True)
    dataset_payload = torch.load(args.dataset, map_location="cpu", weights_only=True)
    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.reset_dataset_path = str(args.dataset.resolve())
    cfg.reset_validation_report_path = str(args.validation_report.resolve())
    cfg.reset_dataset_content_sha256 = args.dataset_content_sha256
    cfg.reset_dataset_sampling_mode = "uniform"
    cfg.curriculum_freeze = True
    cfg.validate_config()
    task_contract = pick_insert_reset_dataset_task_contract(cfg)
    metadata, dataset_states, _ = reset_dataset_validate_runtime(
        dataset_payload,
        expected_content_sha256=args.dataset_content_sha256,
        expected_task_contract=task_contract,
    )
    artifact_phases = dataset_states["phase"]
    reset_dataset_validate_phase_row_counts(
        artifact_phases,
        expected_rows_per_phase=cfg.reset_dataset_rows_per_phase,
    )
    diversity_evidence = reset_dataset_validate_full_pick_diversity(
        dataset_states,
        task_contract=task_contract,
    )
    validation_report = json.loads(args.validation_report.read_text(encoding="utf-8"))
    validation_checks = reset_validation_report_validate_runtime(
        validation_report,
        expected_content_sha256=args.dataset_content_sha256,
        expected_row_count=len(artifact_phases),
        expected_phases=artifact_phases,
        expected_task_contract=task_contract,
        expected_validation_policy=FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY,
        expected_source_sha256=franka_rj45_validation_source_sha256(),
        expected_asset_closure=franka_rj45_asset_contract(),
        expected_full_pick_diversity=diversity_evidence,
    )
    phase_names = list(PICK_INSERT_PHASE_NAMES)
    metadata_phase_names = metadata.get("phase_names")
    if metadata_phase_names is not None and list(metadata_phase_names) != phase_names:
        raise ValueError("Reset dataset phase names do not match the six-phase task contract.")

    agent_cfg = FrankaRJ45PickInsertPPORunnerCfg()
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))

    counts = _new_phase_counts()
    steps_taken = 0
    accepted_episodes = 0
    drive_disabled_check_count = 0
    torch.manual_seed(args.seed)
    from rsl_rl.runners import OnPolicyRunner

    with launch_simulation(cfg, args):
        env = _EvaluationFrankaRJ45PickInsertEnv(cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        try:

            def assert_task_drive_disabled(stage: str) -> None:
                nonlocal drive_disabled_check_count
                drive_enabled = wp.to_torch(env._ensure_rj45_runtime().drive_enabled).to(
                    device=env.device, dtype=torch.bool
                )
                env.evaluation_drive_ever_enabled |= drive_enabled
                drive_disabled_check_count += 1
                enabled = drive_enabled | env.evaluation_drive_ever_enabled
                if bool(enabled.any()):
                    enabled_ids = torch.where(enabled)[0].detach().cpu().tolist()
                    raise RuntimeError(
                        f"Task construction drive became enabled at or before {stage} in envs {enabled_ids}."
                    )

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
            missing_terms = sorted(set(TERMINAL_OUTCOME_NAMES) - set(env.termination_manager.active_terms))
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
                assert_task_drive_disabled(f"policy step {steps_taken}")

                done_ids = torch.where(dones.bool())[0]
                if done_ids.numel() == 0:
                    continue
                terminal_rows = env.last_terminal_row_id[done_ids]
                if bool((terminal_rows < 0).any()):
                    raise RuntimeError("Autoreset did not preserve the completed reset-row identifiers.")
                if not bool(env.last_terminal_pick_diagnostics_valid[done_ids].all()):
                    raise RuntimeError("Autoreset did not preserve the completed pick-stage diagnostics.")
                terminal_outcomes = {
                    name: env.last_terminal_outcomes[name][done_ids].detach().cpu() for name in TERMINAL_OUTCOME_NAMES
                }
                done_phases = env._reset_dataset_states["phase"][terminal_rows].detach().cpu()
                starts_grasped = env._reset_dataset_states["starts_grasped"][terminal_rows].detach().cpu()
                accepted_episodes += _record_completed_episodes(
                    counts,
                    phase_ids=done_phases,
                    episode_returns=episode_return[done_ids].detach().cpu(),
                    episode_lengths=episode_length[done_ids].detach().cpu(),
                    terminal_outcomes=terminal_outcomes,
                    starts_grasped=starts_grasped,
                    initial_stages=env.last_terminal_initial_stage[done_ids].detach().cpu(),
                    maximum_stages=env.last_terminal_maximum_stage[done_ids].detach().cpu(),
                    ever_grasped=env.last_terminal_ever_grasped[done_ids].detach().cpu(),
                    learning_progress=env.last_terminal_learning_progress[done_ids].detach().cpu(),
                    episodes_per_phase=args.episodes_per_phase,
                )
                episode_return[done_ids] = 0.0
                episode_length[done_ids] = 0
            assert_task_drive_disabled("evaluation completion")
        finally:
            wrapped.close()

    complete = all(int(phase_counts["episodes"]) == args.episodes_per_phase for phase_counts in counts)
    per_phase, overall = _summarize_counts(counts, phase_names)
    artifact_phase_counts = torch.bincount(artifact_phases, minlength=len(PICK_INSERT_RESET_PHASE_IDS)).tolist()
    return {
        "format": "isaaclab-franka-rj45-pick-insert-policy-evaluation",
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "complete": complete,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "dataset": str(args.dataset.resolve()),
        "dataset_file_sha256": _sha256(args.dataset),
        "dataset_content_sha256": args.dataset_content_sha256,
        "task_contract_sha256": reset_dataset_digest(task_contract),
        "artifact_phase_counts": {
            str(phase): int(artifact_phase_counts[phase]) for phase in PICK_INSERT_RESET_PHASE_IDS
        },
        "reset_validation": {
            "path": str(args.validation_report.resolve()),
            "sha256": _sha256(args.validation_report),
            "checks": validation_checks,
            "selected_row_count": len(validation_report["selected_row_ids"]),
            "goal_replay": validation_report["goal_replay"],
        },
        "evaluation_assumptions": {
            "per_row_socket_conditioned_goal_from_validated_artifact": True,
            "full_reset_dataset_was_physically_replayed": True,
            "reset_sampling_mode": "uniform",
            "adaptive_curriculum_frozen": True,
            "task_construction_drive_disabled": True,
            "task_drive_disabled_check_count": drive_disabled_check_count,
            "task_drive_checked_before_same_step_autoreset": True,
            "policy_is_deterministic": True,
            "evaluation_seed_is_fixed": True,
            "terminal_causes_captured_before_same_step_autoreset": True,
            "pick_stage_diagnostics_captured_before_same_step_autoreset": True,
            "terminal_cause_counts_are_nonexclusive": True,
            "phase_results_are_capped_for_equal_weight": True,
        },
        "phase_names": phase_names,
        "stage_names": list(_STAGE_NAMES),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "episodes_per_phase": args.episodes_per_phase,
        "accepted_episodes": accepted_episodes,
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
