# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independently replay Franka RJ45 pick-and-insert reset artifacts.

The canonical seated goal is held drive-free for at least sixty simulated
seconds.  Every selected row is restored through the public tool boundary,
cold-settled, checked against its six-stage semantics, and recovered with
only Franka commands.  Open-start phases must first establish bilateral
physical contact.  A stable training gate is published only after a passing,
non-quick replay of every artifact row.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import warp as wp
from _franka_rj45_reset_tools import (
    FrankaResetIK,
    RJ45PickInsertResetToolEnv,
    _active_waypoint_count,
    _PerLaneTargetHold,
    _runtime_bilateral_grasp_proxy_contact_mask,
    advance_exact_success_dwell,
    advance_reset_absolute_target_hold,
    collision_metrics,
    grasp_metrics,
    interpolate_arm_motion,
    joint_limit_mask,
    package_versions,
    pick_insert_tool_physical_contract,
    plug_relative_latch_angle,
    runtime_persistent_arm_target,
    scalar_goal_error,
    scripted_recovery,
    task_state_is_finite_and_normalized,
)
from isaaclab_newton.physics import NewtonManager

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import math as math_utils

from isaaclab_contrib.coupling import NewtonCouplerManager

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.franka_rj45_insertion.franka_robot_cfg import (
    PICK_INSERT_ARM_TARGET_TRACKING_LIMITS,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_env_cfg import (
    PICK_INSERT_CLOSED_FINGER_POSITION,
    PICK_INSERT_OPEN_FINGER_POSITION,
    PICK_INSERT_PHASE_NAMES,
    FrankaRJ45PickInsertEnvCfg,
    pick_insert_reset_dataset_task_contract,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_reset_dataset_io import (
    FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_FORMAT,
    FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY,
    FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_SCHEMA_VERSION,
    PICK_INSERT_GOAL_MAX_ARM_JOINT_SPEED_RAD_S,
    PICK_INSERT_GOAL_MAX_AUTHORED_PLUG_ANGLE_RAD,
    PICK_INSERT_GOAL_MAX_AUTHORED_SEAT_ERROR_M,
    PICK_INSERT_GOAL_MAX_CABLE_SPEED_M_S,
    PICK_INSERT_GOAL_MAX_FINGER_JOINT_SPEED_M_S,
    PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD,
    PICK_INSERT_GOAL_MAX_SOCKET_DRIFT_M,
    PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M,
    PICK_INSERT_RESET_MAX_ARM_JOINT_SPEED_RAD_S,
    PICK_INSERT_RESET_MAX_ARM_TARGET_CLAMP_DELTA_RAD,
    PICK_INSERT_RESET_MAX_BODY_EXCURSION_M,
    PICK_INSERT_RESET_MAX_CABLE_SPEED_M_S,
    PICK_INSERT_RESET_MAX_FINGER_JOINT_SPEED_M_S,
    PICK_INSERT_RESET_MAX_PLUG_EXCURSION_M,
    PICK_INSERT_RESET_MAX_SOCKET_EXCURSION_M,
    PICK_INSERT_RESET_PHASE_IDS,
    PICK_INSERT_RESET_REPLAY_DURATION_S,
    PICK_INSERT_RESET_REPLAY_POST_STEP_SAMPLES,
    franka_rj45_validation_source_sha256,
    reset_dataset_digest,
    reset_dataset_validate_full_pick_diversity,
    reset_dataset_validate_phase_row_counts,
    reset_dataset_validate_runtime,
    reset_validation_report_content_digest,
    reset_validation_report_validate_runtime,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.rj45_env_cfg import RIGID_ENTRY, RJ45_ENTRY

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = _REPO_ROOT / "datasets/franka_rj45_pick_insert/reset_dataset.pt"
DEFAULT_VALIDATION_DIR = _REPO_ROOT / "logs/rsl_rl/franka_rj45_pick_insert/validation"
DEFAULT_STABLE_REPORT_PATH = DEFAULT_VALIDATION_DIR / "reset_validation.json"
_VALIDATION_CHECKPOINT_FORMAT = "isaaclab-franka-rj45-pick-insert-reset-validation-checkpoint"
_VALIDATION_CHECKPOINT_SCHEMA_VERSION = 1
_VALIDATION_CHECKPOINT_STATUSES = ("goal-pending", "rows", "report-ready", "stable-published")
_PHASE_STARTS_GRASPED = (True, True, True, True, False, False)
_FULL_PICK_PHASE = 5
_FULL_PICK_TCP_ROUND_DECIMALS = 4
_FULL_PICK_MINIMUM_UNIQUE_TCP_POSITIONS = 90
_FULL_PICK_MINIMUM_TCP_SPAN_M = (0.05, 0.05, 0.05)
_FULL_PICK_MINIMUM_TCP_TO_GRASP_DISTANCE_M = 0.10


@dataclass(frozen=True)
class ValidationCfg:
    """Strict deterministic replay thresholds."""

    seed: int = 2027
    quick: bool = False
    sample_count: int | None = None
    ik_sampler: str = "none"
    ik_seed_count: int = 1
    ik_iterations: int = 160
    ik_noise_std: float = 0.0
    goal_replay_s: float = 60.0
    row_settle_s: float = PICK_INSERT_RESET_REPLAY_DURATION_S
    grasp_approach_s: float = 2.5
    grasp_close_s: float = 0.8
    grasp_hold_s: float = 1.5
    grasp_post_contact_settle_s: float = 1.0
    grasp_open_clearance_m: float = 0.045
    grasp_route_world_height_m: float = 0.22
    grasp_route_maximum_translation_step_m: float = 0.05
    grasp_coarse_descent_step_m: float = 0.005
    grasp_near_descent_step_m: float = 0.001
    grasp_descent_waypoint_motion_s: float = 0.15
    grasp_descent_waypoint_settle_s: float = 1.0 / 30.0
    grasp_descent_tracking_recovery_s: float = 0.05
    grasp_near_correction_step_m: float = 0.001
    grasp_near_correction_max_iterations: int = 3
    grasp_near_maximum_raw_ik_joint_step_rad: float = 0.02
    grasp_clearance_calibration_step_m: float = 0.001
    grasp_clearance_calibration_max_iterations: int = 24
    maximum_open_approach_plug_drift_m: float = 5.0e-4
    finger_open_position: float = PICK_INSERT_OPEN_FINGER_POSITION
    finger_closed_target: float = PICK_INSERT_CLOSED_FINGER_POSITION
    recovery_motion_s: float = 4.0
    recovery_settle_s: float = 0.75
    recovery_compensation_iterations: int = 6
    recovery_compensation_tolerance_m: float = 0.0015
    maximum_ik_joint_step_rad: float = 0.6
    maximum_goal_socket_drift_m: float = PICK_INSERT_GOAL_MAX_SOCKET_DRIFT_M
    maximum_goal_body_drift_m: float = PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M
    maximum_goal_cable_speed_m_s: float = PICK_INSERT_GOAL_MAX_CABLE_SPEED_M_S
    maximum_goal_arm_joint_speed_rad_s: float = PICK_INSERT_GOAL_MAX_ARM_JOINT_SPEED_RAD_S
    maximum_goal_finger_joint_speed_m_s: float = PICK_INSERT_GOAL_MAX_FINGER_JOINT_SPEED_M_S
    maximum_goal_authored_seat_error_m: float = PICK_INSERT_GOAL_MAX_AUTHORED_SEAT_ERROR_M
    maximum_goal_authored_plug_angle_rad: float = PICK_INSERT_GOAL_MAX_AUTHORED_PLUG_ANGLE_RAD
    maximum_goal_plug_relative_latch_angle_rad: float = PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD
    maximum_row_socket_drift_m: float = PICK_INSERT_RESET_MAX_SOCKET_EXCURSION_M
    maximum_row_plug_drift_m: float = PICK_INSERT_RESET_MAX_PLUG_EXCURSION_M
    maximum_row_body_drift_m: float = PICK_INSERT_RESET_MAX_BODY_EXCURSION_M
    maximum_row_cable_speed_m_s: float = PICK_INSERT_RESET_MAX_CABLE_SPEED_M_S
    maximum_row_arm_joint_speed_rad_s: float = PICK_INSERT_RESET_MAX_ARM_JOINT_SPEED_RAD_S
    maximum_row_finger_joint_speed_m_s: float = PICK_INSERT_RESET_MAX_FINGER_JOINT_SPEED_M_S

    def __post_init__(self) -> None:
        if self.sample_count is not None and (isinstance(self.sample_count, bool) or self.sample_count < 1):
            raise ValueError("sample_count must be a positive integer when provided.")
        if self.goal_replay_s < 60.0:
            raise ValueError("Canonical goal replay must last at least sixty simulated seconds.")
        expected_ik = FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY["ik"]
        observed_ik = {
            "sampler": self.ik_sampler,
            "seed_count": self.ik_seed_count,
            "iterations": self.ik_iterations,
            "noise_std": self.ik_noise_std,
        }
        required_ik = {name: expected_ik[name] for name in observed_ik}
        if observed_ik != required_ik:
            raise ValueError(f"Independent validation IK policy is immutable: {observed_ik} != {required_ik}.")
        for name in (
            "row_settle_s",
            "grasp_approach_s",
            "grasp_close_s",
            "grasp_hold_s",
            "recovery_motion_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 0.04 <= self.grasp_open_clearance_m <= 0.06:
            raise ValueError("grasp_open_clearance_m must preserve the validated 45 mm pregrasp clearance.")
        if self.finger_open_position != PICK_INSERT_OPEN_FINGER_POSITION:
            raise ValueError("Pick-insert validation requires the exact task 0.04 m open-finger posture.")
        if self.finger_closed_target != PICK_INSERT_CLOSED_FINGER_POSITION:
            raise ValueError("Pick-insert validation requires the immutable production closed-finger target.")
        if self.grasp_close_s < 0.8 or self.grasp_hold_s < 1.5:
            raise ValueError("The 0.04 m open gripper requires a 0.8 s close ramp and 1.5 s contact hold.")
        if self.grasp_post_contact_settle_s < 1.0:
            raise ValueError("Independent physical acquisition requires one second of bilateral settling.")
        if not 0.20 <= self.grasp_route_world_height_m <= 0.25:
            raise ValueError("The open-gripper route height must remain in the measured 0.20-0.25 m safe band.")
        if not 0.0 < self.grasp_route_maximum_translation_step_m <= 0.05:
            raise ValueError("The overhead Cartesian route step must be at most 50 mm.")
        if not 0.0 < self.grasp_coarse_descent_step_m <= 0.005:
            raise ValueError("The coarse open descent must use at most 5 mm Cartesian steps.")
        if not 0.0 < self.grasp_near_descent_step_m <= 0.001:
            raise ValueError("The near-plug open descent must use at most 1 mm Cartesian steps.")
        if self.grasp_descent_waypoint_motion_s < 0.15:
            raise ValueError("Each open-descent waypoint must last at least 0.15 simulated seconds.")
        if not 0.0 < self.grasp_descent_waypoint_settle_s <= 0.05:
            raise ValueError("Open-descent waypoint settle must lie in (0, 0.05] seconds.")
        if not 0.0 < self.grasp_descent_tracking_recovery_s <= 0.05:
            raise ValueError("Open-descent tracking recovery must lie in (0, 0.05] seconds.")
        if not 0.0 < self.grasp_near_correction_step_m <= 0.001:
            raise ValueError("Near-plug Cartesian corrections must be at most 1 mm.")
        if type(self.grasp_near_correction_max_iterations) is not int or not (
            1 <= self.grasp_near_correction_max_iterations <= 3
        ):
            raise ValueError("Near-plug Cartesian feedback must use one to three bounded corrections.")
        if not 0.0 < self.grasp_near_maximum_raw_ik_joint_step_rad <= 0.02:
            raise ValueError("Near-plug IK continuation must reject raw joint steps above 0.02 rad.")
        if not 0.0 < self.grasp_clearance_calibration_step_m <= 0.001:
            raise ValueError("Clearance calibration must use corrections no larger than 1 mm.")
        if self.grasp_clearance_calibration_max_iterations < 1:
            raise ValueError("Clearance calibration requires at least one bounded iteration.")
        if not 0.0 < self.maximum_open_approach_plug_drift_m <= 5.0e-4:
            raise ValueError("Open approach must reject plug drift beyond 0.5 mm.")
        goal_contract = {
            "maximum_goal_socket_drift_m": (
                self.maximum_goal_socket_drift_m,
                PICK_INSERT_GOAL_MAX_SOCKET_DRIFT_M,
            ),
            "maximum_goal_body_drift_m": (
                self.maximum_goal_body_drift_m,
                PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M,
            ),
            "maximum_goal_cable_speed_m_s": (
                self.maximum_goal_cable_speed_m_s,
                PICK_INSERT_GOAL_MAX_CABLE_SPEED_M_S,
            ),
            "maximum_goal_arm_joint_speed_rad_s": (
                self.maximum_goal_arm_joint_speed_rad_s,
                PICK_INSERT_GOAL_MAX_ARM_JOINT_SPEED_RAD_S,
            ),
            "maximum_goal_finger_joint_speed_m_s": (
                self.maximum_goal_finger_joint_speed_m_s,
                PICK_INSERT_GOAL_MAX_FINGER_JOINT_SPEED_M_S,
            ),
            "maximum_goal_authored_seat_error_m": (
                self.maximum_goal_authored_seat_error_m,
                PICK_INSERT_GOAL_MAX_AUTHORED_SEAT_ERROR_M,
            ),
            "maximum_goal_authored_plug_angle_rad": (
                self.maximum_goal_authored_plug_angle_rad,
                PICK_INSERT_GOAL_MAX_AUTHORED_PLUG_ANGLE_RAD,
            ),
            "maximum_goal_plug_relative_latch_angle_rad": (
                self.maximum_goal_plug_relative_latch_angle_rad,
                PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD,
            ),
        }
        mismatched_goal = {
            name: observed for name, (observed, required) in goal_contract.items() if observed != required
        }
        if mismatched_goal:
            raise ValueError(f"Canonical goal replay thresholds are immutable: {mismatched_goal}.")
        row_contract = {
            "row_settle_s": (self.row_settle_s, PICK_INSERT_RESET_REPLAY_DURATION_S),
            "maximum_row_socket_drift_m": (
                self.maximum_row_socket_drift_m,
                PICK_INSERT_RESET_MAX_SOCKET_EXCURSION_M,
            ),
            "maximum_row_plug_drift_m": (
                self.maximum_row_plug_drift_m,
                PICK_INSERT_RESET_MAX_PLUG_EXCURSION_M,
            ),
            "maximum_row_body_drift_m": (
                self.maximum_row_body_drift_m,
                PICK_INSERT_RESET_MAX_BODY_EXCURSION_M,
            ),
            "maximum_row_cable_speed_m_s": (
                self.maximum_row_cable_speed_m_s,
                PICK_INSERT_RESET_MAX_CABLE_SPEED_M_S,
            ),
            "maximum_row_arm_joint_speed_rad_s": (
                self.maximum_row_arm_joint_speed_rad_s,
                PICK_INSERT_RESET_MAX_ARM_JOINT_SPEED_RAD_S,
            ),
            "maximum_row_finger_joint_speed_m_s": (
                self.maximum_row_finger_joint_speed_m_s,
                PICK_INSERT_RESET_MAX_FINGER_JOINT_SPEED_M_S,
            ),
        }
        mismatched = {name: observed for name, (observed, required) in row_contract.items() if observed != required}
        if mismatched:
            raise ValueError(f"Reset row replay thresholds are immutable: {mismatched}.")


def _selected_rows(states: dict[str, torch.Tensor], cfg: ValidationCfg) -> torch.Tensor:
    """Select all rows by default, or one deterministic row per phase in quick mode."""
    row_count = len(states["phase"])
    if cfg.quick:
        selected = []
        for phase in PICK_INSERT_RESET_PHASE_IDS:
            phase_rows = torch.where(states["phase"] == phase)[0]
            if not phase_rows.numel():
                raise RuntimeError(f"Quick validation found no reset row for phase {phase}.")
            selected.append(phase_rows[0])
        return torch.stack(selected).long()
    if cfg.sample_count is None or cfg.sample_count >= row_count:
        return torch.arange(row_count, dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    return torch.randperm(row_count, generator=generator)[: cfg.sample_count]


def _validate_invocation_phase_counts(
    phases: torch.Tensor,
    cfg: ValidationCfg,
    *,
    expected_rows_per_phase: int,
) -> tuple[int, ...] | None:
    """Require the canonical bank shape before any non-quick physical replay."""
    if cfg.quick:
        return None
    return reset_dataset_validate_phase_row_counts(
        phases,
        expected_rows_per_phase=expected_rows_per_phase,
    )


def _validation_batch_plan(
    phases: torch.Tensor,
    selected: torch.Tensor,
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Return the exact phase-major batch identity used by physical replay."""
    plan: list[dict[str, Any]] = []
    for phase in PICK_INSERT_RESET_PHASE_IDS:
        phase_rows = selected[phases[selected] == phase]
        for begin in range(0, len(phase_rows), batch_size):
            selected_batch = phase_rows[begin : begin + batch_size]
            plan.append(
                {
                    "ordinal": len(plan),
                    "phase": phase,
                    "phase_batch_index": begin // batch_size + 1,
                    "row_ids": [int(row_id) for row_id in selected_batch.tolist()],
                }
            )
    return plan


def _validate_completed_batch_prefix(
    completed_batches: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    plan: list[dict[str, Any]],
) -> None:
    """Reject a resume state unless it ends on one exact batch boundary."""
    if len(completed_batches) > len(plan):
        raise ValueError("Validator checkpoint contains more completed batches than the replay plan.")
    for observed, expected in zip(completed_batches, plan, strict=False):
        if reset_dataset_digest(observed) != reset_dataset_digest(expected):
            raise ValueError("Validator checkpoint completed-batch prefix does not match this invocation.")
    expected_row_ids = [row_id for batch in completed_batches for row_id in batch["row_ids"]]
    observed_row_ids = [row.get("row_id") for row in rows]
    if observed_row_ids != expected_row_ids:
        raise ValueError("Validator checkpoint rows are not the exact completed-batch prefix.")


def _validation_source_digests() -> dict[str, str]:
    """Hash the complete repository source closure that can affect validation."""
    return franka_rj45_validation_source_sha256(_REPO_ROOT)


def _validated_asset_closure_snapshot() -> dict[str, Any]:
    """Verify the configured local closure and return only its path-free contract."""
    from isaaclab_tasks.contrib.franka_rj45_insertion.asset_provenance import (
        configured_franka_rj45_asset_closure,
        franka_rj45_asset_contract,
    )

    configured_franka_rj45_asset_closure(required=True)
    return franka_rj45_asset_contract()


def _resolve_output_path(path: Path) -> Path:
    """Resolve a CLI output path relative to the repository root."""
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (_REPO_ROOT / expanded).resolve()


def _resolve_validation_output_paths(
    *,
    input_path: Path,
    output_dir: Path,
    stable_output: Path,
) -> tuple[Path, Path]:
    """Freeze output targets and reject any path that could replace the dataset."""
    resolved_input = input_path.expanduser().resolve()
    resolved_output_dir = _resolve_output_path(output_dir)
    resolved_stable = _resolve_output_path(stable_output)
    if resolved_stable == resolved_input:
        raise ValueError("Stable validation output cannot alias the reset artifact.")
    if resolved_output_dir == resolved_input or (resolved_output_dir.exists() and not resolved_output_dir.is_dir()):
        raise ValueError("Validation output directory must be a directory distinct from the reset artifact.")
    if resolved_stable.exists() and not resolved_stable.is_file():
        raise ValueError("Stable validation output must be a regular file target.")
    return resolved_output_dir, resolved_stable


def _reject_protected_output_alias(output: Path, protected_paths: tuple[Path, ...], *, label: str) -> Path:
    """Return a resolved output after rejecting all protected file aliases."""
    resolved = _resolve_output_path(output)
    protected = {_resolve_output_path(path) for path in protected_paths}
    if resolved in protected:
        raise ValueError(f"{label} cannot alias a protected validator input or output path.")
    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"{label} must be a regular file target.")
    return resolved


def _timestamped_validation_report_path(
    output_dir: Path,
    *,
    artifact_content_sha256: str,
    quick: bool,
    protected_paths: tuple[Path, ...],
    created_at: datetime | None = None,
) -> Path:
    """Freeze and validate the timestamped report target before simulation."""
    stamp = (datetime.now(UTC) if created_at is None else created_at).strftime("%Y%m%dT%H%M%SZ")
    suffix = "quick" if quick else "full"
    output = output_dir / f"reset_validation_{stamp}_{artifact_content_sha256[:12]}_{suffix}.json"
    return _reject_protected_output_alias(output, protected_paths, label="Timestamped validation report")


def _validate_checkpoint_invocation(
    *,
    checkpoint: Path | None,
    resume: Path | None,
    keep_checkpoint: bool,
    input_path: Path,
    stable_output: Path,
    cfg: ValidationCfg,
) -> tuple[Path | None, bool]:
    """Resolve the opt-in full-validation checkpoint policy before simulation."""
    resolved_input = _resolve_output_path(input_path)
    resolved_stable = _resolve_output_path(stable_output)
    if resolved_input == resolved_stable:
        raise ValueError("Stable validation output cannot alias the reset artifact.")
    if checkpoint is not None and resume is not None:
        raise ValueError("--checkpoint and --resume are mutually exclusive.")
    requested = checkpoint if checkpoint is not None else resume
    if requested is None:
        if keep_checkpoint:
            raise ValueError("--keep-checkpoint requires --checkpoint or --resume.")
        return None, False
    if cfg.quick or cfg.sample_count is not None:
        raise ValueError("Validator checkpoint/resume is supported only for a full-dataset replay.")
    resolved = _resolve_output_path(requested)
    if resolved.suffix != ".json":
        raise ValueError("Validator checkpoint paths must use the .json suffix.")
    aliases = {resolved_input, resolved_stable}
    if resolved in aliases:
        raise ValueError("Validator checkpoint cannot alias the reset artifact or stable validation report.")
    if resume is not None:
        if not resolved.is_file():
            raise FileNotFoundError(f"Validator resume checkpoint does not exist: {resolved}")
    elif resolved.exists():
        raise FileExistsError(f"Refusing to overwrite an existing validator checkpoint: {resolved}")
    return resolved, resume is not None


def _torch_rng_state_json() -> list[int]:
    """Capture the CPU Torch RNG without relying on pickle serialization."""
    return [int(value) for value in torch.get_rng_state().tolist()]


def _validation_progress_counters(
    rows: list[dict[str, Any]],
    completed_batches: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize checkpointed row rejections and oracle evidence by phase."""
    per_phase: list[dict[str, int]] = []
    for phase in PICK_INSERT_RESET_PHASE_IDS:
        phase_rows = [row for row in rows if row.get("phase") == phase]
        per_phase.append(
            {
                "phase": phase,
                "completed_batches": sum(batch.get("phase") == phase for batch in completed_batches),
                "completed_rows": len(phase_rows),
                "passed_rows": sum(row.get("passed") is True for row in phase_rows),
                "failed_rows": sum(row.get("passed") is not True for row in phase_rows),
                "oracle_passed_rows": sum(
                    isinstance(row.get("oracle"), Mapping) and all(value is True for value in row["oracle"].values())
                    for row in phase_rows
                ),
            }
        )
    return {
        "completed_batch_count": len(completed_batches),
        "completed_row_count": len(rows),
        "passed_row_count": sum(row.get("passed") is True for row in rows),
        "failed_row_count": sum(row.get("passed") is not True for row in rows),
        "failed_row_ids": [row.get("row_id") for row in rows if row.get("passed") is not True],
        "per_phase": per_phase,
    }


def _restore_torch_rng_state(values: Any) -> None:
    """Restore one strictly validated CPU Torch RNG state."""
    torch.set_rng_state(_validated_torch_rng_state(values))


def _validated_torch_rng_state(values: Any) -> torch.Tensor:
    """Return a checkpoint RNG tensor after strict byte and runtime-size checks."""
    if (
        not isinstance(values, list)
        or not values
        or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255 for value in values)
    ):
        raise ValueError("Checkpoint Torch RNG state must be a non-empty list of bytes.")
    state = torch.tensor(values, dtype=torch.uint8, device="cpu")
    if state.numel() != torch.get_rng_state().numel():
        raise ValueError("Checkpoint Torch RNG state has the wrong size for this runtime.")
    return state


def _checkpoint_content_digest(checkpoint: Mapping[str, Any]) -> str:
    unsigned = dict(checkpoint)
    unsigned.pop("content_sha256", None)
    return reset_dataset_digest(json.loads(json.dumps(unsigned, allow_nan=False)))


def _checkpoint_document(metadata: Mapping[str, Any], progress: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = {
        "format": _VALIDATION_CHECKPOINT_FORMAT,
        "schema_version": _VALIDATION_CHECKPOINT_SCHEMA_VERSION,
        "metadata": json.loads(json.dumps(dict(metadata), allow_nan=False)),
        "progress": json.loads(json.dumps(dict(progress), allow_nan=False)),
    }
    checkpoint["content_sha256"] = _checkpoint_content_digest(checkpoint)
    return checkpoint


def _load_validation_checkpoint(path: Path) -> dict[str, Any]:
    """Load an atomic JSON checkpoint and reject any structural or content drift."""
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read validator checkpoint {path}.") from exc
    if not isinstance(checkpoint, dict):
        raise TypeError("Validator checkpoint must be a JSON object.")
    if set(checkpoint) != {"format", "schema_version", "metadata", "progress", "content_sha256"}:
        raise ValueError("Validator checkpoint fields do not match schema 1.")
    if checkpoint.get("format") != _VALIDATION_CHECKPOINT_FORMAT:
        raise ValueError("Validator checkpoint has the wrong format.")
    if checkpoint.get("schema_version") != _VALIDATION_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Validator checkpoint has the wrong schema version.")
    if not isinstance(checkpoint.get("metadata"), dict) or not isinstance(checkpoint.get("progress"), dict):
        raise TypeError("Validator checkpoint metadata and progress must be JSON objects.")
    content_sha256 = checkpoint.get("content_sha256")
    if (
        not isinstance(content_sha256, str)
        or len(content_sha256) != 64
        or any(character not in "0123456789abcdef" for character in content_sha256)
        or content_sha256 != _checkpoint_content_digest(checkpoint)
    ):
        raise ValueError("Validator checkpoint content digest does not match its payload.")
    progress = checkpoint["progress"]
    expected_progress_names = {
        "status",
        "created_utc",
        "goal_replay",
        "completed_batches",
        "rows",
        "counters",
        "ik_solve_call_count",
        "torch_rng_state",
        "report",
    }
    if set(progress) != expected_progress_names:
        raise ValueError("Validator checkpoint progress fields do not match schema 1.")
    if progress.get("status") not in _VALIDATION_CHECKPOINT_STATUSES:
        raise ValueError("Validator checkpoint has an invalid lifecycle status.")
    if not isinstance(progress.get("created_utc"), str) or not progress["created_utc"]:
        raise ValueError("Validator checkpoint created_utc must be a non-empty string.")
    if not isinstance(progress.get("completed_batches"), list) or not isinstance(progress.get("rows"), list):
        raise TypeError("Validator checkpoint completed_batches and rows must be lists.")
    counters = progress.get("counters")
    if not isinstance(counters, dict) or reset_dataset_digest(counters) != reset_dataset_digest(
        _validation_progress_counters(progress["rows"], progress["completed_batches"])
    ):
        raise ValueError("Validator checkpoint counters do not exactly match its row and batch evidence.")
    solve_calls = progress.get("ik_solve_call_count")
    if isinstance(solve_calls, bool) or not isinstance(solve_calls, int) or solve_calls < 0:
        raise ValueError("Validator checkpoint IK solve-call evidence must be a non-negative integer.")
    status = progress["status"]
    if status == "goal-pending" and (
        progress.get("goal_replay") is not None
        or progress["completed_batches"]
        or progress["rows"]
        or progress.get("report") is not None
    ):
        raise ValueError("A goal-pending validator checkpoint cannot contain completed evidence.")
    if status in {"rows", "report-ready", "stable-published"} and not isinstance(progress.get("goal_replay"), dict):
        raise TypeError("A post-goal validator checkpoint requires goal replay evidence.")
    if status == "rows" and progress.get("report") is not None:
        raise ValueError("A rows validator checkpoint cannot already contain a final report.")
    if status in {"report-ready", "stable-published"}:
        report = progress.get("report")
        if not isinstance(report, dict):
            raise TypeError("A report-ready validator checkpoint requires a final report.")
        if report.get("content_sha256") != reset_validation_report_content_digest(report):
            raise ValueError("Checkpointed validation report content digest does not match its payload.")
        for name, expected in {
            "created_utc": progress["created_utc"],
            "goal_replay": progress["goal_replay"],
            "rows": progress["rows"],
            "ik_solve_call_count": progress["ik_solve_call_count"],
        }.items():
            if reset_dataset_digest(report.get(name)) != reset_dataset_digest(expected):
                raise ValueError(f"Checkpointed validation report {name} does not match checkpoint progress.")
    _validated_torch_rng_state(progress.get("torch_rng_state"))
    return checkpoint


def _validation_checkpoint_metadata(
    *,
    payload: Mapping[str, Any],
    states: Mapping[str, torch.Tensor],
    cfg: ValidationCfg,
    batch_size: int,
    task_contract: Mapping[str, Any],
    physical_contract: Mapping[str, Any],
    physics_versions: Mapping[str, Any],
    source_sha256: Mapping[str, str],
    asset_closure: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete path-independent resume identity."""
    return {
        "artifact_content_sha256": payload["content_sha256"],
        "artifact_metadata": payload["metadata"],
        "dataset_row_count": len(states["phase"]),
        "dataset_phases": [int(value) for value in states["phase"].tolist()],
        "selected_row_ids": list(range(len(states["phase"]))),
        "batch_size": batch_size,
        "validation_cfg": asdict(cfg),
        "validation_policy": FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY,
        "task_contract": dict(task_contract),
        "physical_contract": dict(physical_contract),
        "physics_versions": dict(physics_versions),
        "source_sha256": dict(source_sha256),
        "asset_closure": dict(asset_closure),
    }


def _validate_checkpoint_metadata(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    normalized_actual = json.loads(json.dumps(actual, allow_nan=False))
    normalized_expected = json.loads(json.dumps(expected, allow_nan=False))
    if reset_dataset_digest(normalized_actual) != reset_dataset_digest(normalized_expected):
        raise ValueError("Validator checkpoint metadata does not exactly match this artifact/runtime invocation.")


class _CountingValidatorIK:
    """Count calls as evidence while keeping solver state out of checkpoints."""

    def __init__(self, solver: FrankaResetIK, *, prior_solve_calls: int = 0) -> None:
        self._solver = solver
        self.solve_calls = prior_solve_calls

    def solve(self, *args, **kwargs):
        self.solve_calls += 1
        return self._solver.solve(*args, **kwargs)


def _new_validator_ik(
    env: RJ45PickInsertResetToolEnv,
    cfg: ValidationCfg,
    *,
    prior_solve_calls: int = 0,
) -> _CountingValidatorIK:
    """Construct the one exact sampler-free validator IK owner."""
    return _CountingValidatorIK(
        FrankaResetIK(
            env,
            seed=cfg.seed,
            seeds=cfg.ik_seed_count,
            iterations=cfg.ik_iterations,
            noise_std=cfg.ik_noise_std,
            sampler=cfg.ik_sampler,
        ),
        prior_solve_calls=prior_solve_calls,
    )


def _full_pick_live_tcp_diversity(
    rows: list[dict[str, Any]],
    *,
    quick: bool,
    required_row_count: int,
    offline_artifact_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    """Aggregate independent phase-5 TCP coverage from physically replayed rows."""
    phase_rows = [row for row in rows if row.get("phase") == _FULL_PICK_PHASE]
    base: dict[str, Any] = {
        "required_phase": _FULL_PICK_PHASE,
        "required_rows": required_row_count,
        "round_decimals": _FULL_PICK_TCP_ROUND_DECIMALS,
        "minimum_unique_tcp_positions": _FULL_PICK_MINIMUM_UNIQUE_TCP_POSITIONS,
        "minimum_tcp_xyz_span_m": list(_FULL_PICK_MINIMUM_TCP_SPAN_M),
        "minimum_tcp_to_grasp_distance_m": _FULL_PICK_MINIMUM_TCP_TO_GRASP_DISTANCE_M,
        "observed_rows": len(phase_rows),
        "offline_artifact": offline_artifact_evidence,
    }
    if quick:
        return {
            **base,
            "skipped_due_to_quick": True,
            "passed": None,
            "failures": [],
        }

    tcp_xyz = [row.get("metrics", {}).get("initial_tcp_xyz_replayed_m") for row in phase_rows]
    tcp_distance = [row.get("metrics", {}).get("initial_tcp_distance_replayed_m") for row in phase_rows]
    valid_xyz = all(
        isinstance(position, list)
        and len(position) == 3
        and all(
            isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
            for value in position
        )
        for position in tcp_xyz
    )
    valid_distance = all(
        isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
        for value in tcp_distance
    )
    if valid_xyz and tcp_xyz:
        xyz_min = [min(float(position[axis]) for position in tcp_xyz) for axis in range(3)]
        xyz_max = [max(float(position[axis]) for position in tcp_xyz) for axis in range(3)]
        xyz_span = [upper - lower for lower, upper in zip(xyz_min, xyz_max, strict=True)]
        scale = 10**_FULL_PICK_TCP_ROUND_DECIMALS
        unique_positions = len({tuple(round(float(value) * scale) for value in position) for position in tcp_xyz})
    else:
        xyz_min = [0.0, 0.0, 0.0]
        xyz_max = [0.0, 0.0, 0.0]
        xyz_span = [0.0, 0.0, 0.0]
        unique_positions = 0
    minimum_distance = min((float(value) for value in tcp_distance), default=0.0) if valid_distance else 0.0
    failures: list[str] = []
    if len(phase_rows) != required_row_count:
        failures.append("phase_5_row_count")
    if not valid_xyz:
        failures.append("initial_tcp_xyz_finite")
    if any(observed < required for observed, required in zip(xyz_span, _FULL_PICK_MINIMUM_TCP_SPAN_M, strict=True)):
        failures.append("initial_tcp_xyz_span")
    if unique_positions < _FULL_PICK_MINIMUM_UNIQUE_TCP_POSITIONS:
        failures.append("unique_initial_tcp_positions")
    if not valid_distance or minimum_distance < _FULL_PICK_MINIMUM_TCP_TO_GRASP_DISTANCE_M:
        failures.append("minimum_tcp_to_grasp_distance")
    offline_passed = isinstance(offline_artifact_evidence, dict) and offline_artifact_evidence.get("passed") is True
    if not offline_passed:
        failures.append("offline_artifact_diversity")
    return {
        **base,
        "skipped_due_to_quick": False,
        "tcp_xyz_min_m": xyz_min,
        "tcp_xyz_max_m": xyz_max,
        "tcp_xyz_span_m": xyz_span,
        "unique_tcp_positions": unique_positions,
        "observed_minimum_tcp_to_grasp_distance_m": minimum_distance,
        "passed": not failures,
        "failures": failures,
    }


def _contact_count() -> int:
    """Count outer and proxy-local contacts after a history flush."""
    proxy_contacts, _, _ = NewtonCouplerManager.get_proxy_contact_data(RIGID_ENTRY, RJ45_ENTRY)
    count = 0
    for contacts in (NewtonManager.get_contacts(), proxy_contacts):
        if contacts is not None and contacts.rigid_contact_count is not None:
            count += int(wp.to_torch(contacts.rigid_contact_count)[0])
    return count


def _drive_disabled(env: RJ45PickInsertResetToolEnv) -> bool:
    runtime = env.rj45_runtime
    return not bool(wp.to_torch(runtime.drive_enabled).any()) and not bool(
        wp.to_torch(runtime.orientation_hold_enabled).any()
    )


def _repeat_goal_state(goal: dict[str, torch.Tensor], env: RJ45PickInsertResetToolEnv) -> dict[str, torch.Tensor]:
    return {
        name: value.to(env.device).unsqueeze(0).repeat(env.num_envs, *([1] * value.ndim))
        for name, value in goal.items()
    }


def _write_state(
    env: RJ45PickInsertResetToolEnv,
    state: dict[str, torch.Tensor],
) -> dict[str, object]:
    env.write_task_state(state["task_body_pose"], state["task_body_velocity"])
    env.write_robot_state(
        state["arm_joint_position"],
        state["finger_joint_position"],
        arm_target=state["arm_joint_target"],
        arm_qd=state["arm_joint_velocity"],
        finger_qd=state["finger_joint_velocity"],
        finger_target=state["finger_joint_target"],
    )
    env.set_drive(False)
    env.flush_reset_history()
    evidence = env.restore_task_pose_history_e(
        state["task_body_previous_pose"],
        state["task_body_coupling_previous_pose"],
    )
    queued = (
        bool(torch.as_tensor(evidence["restore_queued"]).all())
        and bool(torch.as_tensor(evidence["pending_at_queue"]).all())
        and bool(torch.as_tensor(evidence["previous_pose_queued"]).all())
        and bool(torch.as_tensor(evidence["coupling_previous_pose_queued"]).all())
        and evidence["body_order_exact"] is True
        and evidence["world_order_exact"] is True
        and tuple(evidence["body_order"]) == tuple(env.rj45_runtime.layout.body_names)
    )
    if not queued:
        raise RuntimeError("Validator failed to queue both VBD pose histories in exact task/world order.")
    return evidence


def _vbd_pose_history_applied_mask(
    env: RJ45PickInsertResetToolEnv,
    evidence: dict[str, object],
) -> torch.Tensor:
    """Return worlds whose deferred histories were consumed exactly once."""
    return (
        torch.as_tensor(evidence["restore_queued"], device=env.device)
        & torch.as_tensor(evidence["pending_at_queue"], device=env.device)
        & torch.as_tensor(evidence["previous_pose_queued"], device=env.device)
        & torch.as_tensor(evidence["coupling_previous_pose_queued"], device=env.device)
        & torch.as_tensor(evidence["applied_exactly_once"], device=env.device)
        & ~torch.as_tensor(evidence["failed"], device=env.device)
        & ~torch.as_tensor(evidence["superseded"], device=env.device)
        & ~torch.as_tensor(evidence["pending_after_first_solve"], device=env.device)
        & (torch.as_tensor(evidence["application_count_delta"], device=env.device) == 1)
        & (
            torch.as_tensor(evidence["body_application_count_delta"], device=env.device)
            == torch.as_tensor(evidence["expected_body_count"], device=env.device)
        )
    )


def _desired_tcp_pose(
    env: RJ45PickInsertResetToolEnv,
    plug_pose: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    offset = torch.as_tensor(env.cfg.plug_grasp_offset, device=env.device, dtype=torch.float32)
    offset = offset.expand(env.num_envs, -1)
    local_orientation = torch.as_tensor(
        env.cfg.plug_grasp_orientation_xyzw, device=env.device, dtype=torch.float32
    ).expand(env.num_envs, -1)
    position = plug_pose[:, :3] + math_utils.quat_apply(plug_pose[:, 3:7], offset)
    orientation = math_utils.quat_mul(plug_pose[:, 3:7], local_orientation)
    return position, orientation


def _move_tcp_for_acquisition(
    env: RJ45PickInsertResetToolEnv,
    ik: FrankaResetIK,
    position: torch.Tensor,
    orientation: torch.Tensor,
    arm_seed: torch.Tensor,
    finger_target: torch.Tensor,
    cfg: ValidationCfg,
    *,
    motion_s: float | None = None,
    lane_hold: _PerLaneTargetHold,
    failure_reason: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move at safe clearance with bounded residual correction and command continuity."""
    initial_active = lane_hold.active_mask
    current_target = torch.where(initial_active[:, None], arm_seed, lane_hold.last_sent_arm_target)
    valid = initial_active.clone()
    commanded = position.clone()
    for iteration in range(5):
        active_before = lane_hold.active_mask
        if not bool(active_before.any()):
            break
        actual_tcp = env.tcp_pose_e()
        fallback_orientation = torch.zeros_like(orientation)
        fallback_orientation[:, 3] = 1.0
        fallback_orientation = torch.where(
            torch.isfinite(actual_tcp[:, 3:7]).all(dim=-1, keepdim=True),
            actual_tcp[:, 3:7],
            fallback_orientation,
        )
        solution = ik.solve(
            torch.where(active_before[:, None], commanded, torch.nan_to_num(actual_tcp[:, :3])),
            torch.where(active_before[:, None], orientation, fallback_orientation),
            torch.where(active_before[:, None], finger_target, lane_hold.last_sent_finger_target),
            arm_seed=current_target,
        )
        solution_valid = solution.valid & joint_limit_mask(env, solution.arm_q, margin=0.02)
        valid &= ~active_before | solution_valid
        command_mask = active_before & solution_valid
        lane_hold.deactivate(active_before & ~solution_valid, reason=failure_reason)
        held_target = lane_hold.last_sent_arm_target
        safe_arm_target = torch.where(command_mask[:, None], solution.arm_q, held_target)
        interpolate_arm_motion(
            env,
            held_target,
            safe_arm_target,
            finger_target,
            (cfg.grasp_approach_s if motion_s is None else motion_s) if iteration == 0 else 0.4,
        )
        env.set_robot_targets(safe_arm_target, finger_target)
        env.advance(0.3)
        current_target = lane_hold.last_sent_arm_target
        error = position - env.tcp_pose_e()[:, :3]
        error_norm = torch.linalg.vector_norm(error, dim=-1)
        correction_needed = lane_hold.active_mask & (error_norm > 0.002)
        if not bool(correction_needed.any()):
            break
        correction = error * torch.clamp(0.02 / error_norm[:, None].clamp_min(1.0e-9), max=1.0)
        commanded += torch.where(correction_needed[:, None], correction, torch.zeros_like(correction))
    final_error = torch.linalg.vector_norm(position - env.tcp_pose_e()[:, :3], dim=-1)
    return lane_hold.last_sent_arm_target, valid & (final_error <= 0.002) & lane_hold.active_mask


def _move_tcp_with_fixed_bias(
    env: RJ45PickInsertResetToolEnv,
    ik: FrankaResetIK,
    position: torch.Tensor,
    orientation: torch.Tensor,
    finger_target: torch.Tensor,
    *,
    raw_arm_seed: torch.Tensor,
    arm_target_bias: torch.Tensor,
    cfg: ValidationCfg,
    maximum_raw_joint_step_rad: float,
    lane_hold: _PerLaneTargetHold,
    failure_reason: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance one guarded Cartesian waypoint without measured-start target resets."""
    active_before = lane_hold.active_mask
    if not bool(active_before.any()):
        return raw_arm_seed, lane_hold.last_sent_arm_target, active_before
    actual_tcp = env.tcp_pose_e()
    fallback_orientation = torch.zeros_like(orientation)
    fallback_orientation[:, 3] = 1.0
    fallback_orientation = torch.where(
        torch.isfinite(actual_tcp[:, 3:7]).all(dim=-1, keepdim=True),
        actual_tcp[:, 3:7],
        fallback_orientation,
    )
    solution = ik.solve(
        torch.where(active_before[:, None], position, torch.nan_to_num(actual_tcp[:, :3])),
        torch.where(active_before[:, None], orientation, fallback_orientation),
        torch.where(active_before[:, None], finger_target, lane_hold.last_sent_finger_target),
        arm_seed=raw_arm_seed,
    )
    raw_joint_step = torch.abs(solution.arm_q - raw_arm_seed).amax(dim=-1)
    proposed_target = solution.arm_q + arm_target_bias
    solution_valid = (
        solution.valid
        & (raw_joint_step <= maximum_raw_joint_step_rad)
        & joint_limit_mask(env, proposed_target, margin=0.02)
    )
    valid = active_before & solution_valid
    lane_hold.deactivate(active_before & ~solution_valid, reason=failure_reason)
    held_target = lane_hold.last_sent_arm_target
    safe_raw = torch.where(valid[:, None], solution.arm_q, raw_arm_seed)
    safe_target = torch.where(valid[:, None], proposed_target, held_target)
    interpolate_arm_motion(
        env,
        held_target,
        safe_target,
        finger_target,
        cfg.grasp_descent_waypoint_motion_s,
    )
    env.set_robot_targets(safe_target, finger_target)
    env.advance(cfg.grasp_descent_waypoint_settle_s)
    return safe_raw, lane_hold.last_sent_arm_target, valid & lane_hold.active_mask


def _close_gripper(
    env: RJ45PickInsertResetToolEnv,
    arm_target: torch.Tensor,
    open_target: torch.Tensor,
    closed_target: torch.Tensor,
    cfg: ValidationCfg,
) -> None:
    def update(_step: int, _steps: int, progress: float) -> None:
        blend = progress * progress * (3.0 - 2.0 * progress)
        env.set_robot_targets(arm_target, torch.lerp(open_target, closed_target, blend))

    env.advance(cfg.grasp_close_s, update)
    env.set_robot_targets(arm_target, closed_target)
    env.advance(cfg.grasp_hold_s)


def _settle_physical_grasp(
    env: RJ45PickInsertResetToolEnv,
    arm_target: torch.Tensor,
    closed_target: torch.Tensor,
    cfg: ValidationCfg,
    *,
    lane_hold: _PerLaneTargetHold | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Independently prove the new grasp remains bilateral while it equilibrates."""
    layout = env.rj45_runtime.layout
    plug_index = int(layout.plug_body_index)
    cable_slice = layout.cable_body_slice
    all_finite = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    all_collision_free = all_finite.clone()
    all_bilateral = all_finite.clone()
    all_grasp_valid = all_finite.clone()
    all_drives_disabled = all_finite.clone()
    maximum_cable_speed = torch.zeros(env.num_envs, device=env.device)
    maximum_plug_linear_speed = torch.zeros_like(maximum_cable_speed)
    maximum_plug_angular_speed = torch.zeros_like(maximum_cable_speed)
    any_overflow = False
    invalid_pairs: list[str] = []

    def sample(_step: int, _steps: int, _progress: float) -> None:
        nonlocal any_overflow
        task_q, task_qd = env.read_task_state()
        collision = collision_metrics(env, require_bilateral_grasp=False)
        if collision.contact_overflow:
            raise RuntimeError("Global contact-buffer overflow during validator grasp settling.")
        cable_speed = torch.linalg.vector_norm(task_qd[:, cable_slice, :3], dim=-1).amax(dim=-1)
        plug_linear_speed = torch.linalg.vector_norm(task_qd[:, plug_index, :3], dim=-1)
        plug_angular_speed = torch.linalg.vector_norm(task_qd[:, plug_index, 3:6], dim=-1)
        bilateral = _runtime_bilateral_grasp_proxy_contact_mask(
            env,
            collision.left_grasp_contact_count,
            collision.right_grasp_contact_count,
        )
        grasp = grasp_metrics(env, closed_target, retaining_grasp=False)
        finite = task_state_is_finite_and_normalized(task_q, task_qd)
        drives_disabled = torch.full_like(all_finite, _drive_disabled(env))
        if not bool(drives_disabled.all()):
            raise RuntimeError("A construction drive became enabled during validator grasp settling.")
        all_finite.logical_and_(finite)
        all_collision_free.logical_and_(collision.valid)
        all_bilateral.logical_and_(bilateral)
        all_grasp_valid.logical_and_(grasp.valid)
        all_drives_disabled.logical_and_(drives_disabled)
        maximum_cable_speed.copy_(torch.maximum(maximum_cable_speed, cable_speed))
        maximum_plug_linear_speed.copy_(torch.maximum(maximum_plug_linear_speed, plug_linear_speed))
        maximum_plug_angular_speed.copy_(torch.maximum(maximum_plug_angular_speed, plug_angular_speed))
        any_overflow |= collision.contact_overflow
        if lane_hold is not None:
            lane_hold.deactivate(~finite, reason="validator-post-contact-non-finite")
            lane_hold.deactivate(~collision.valid, reason="validator-post-contact-collision")
            lane_hold.deactivate(~bilateral, reason="validator-post-contact-lost-bilateral-contact")
            lane_hold.deactivate(~grasp.valid, reason="validator-post-contact-invalid-grasp")
        for pair in collision.invalid_contact_pairs:
            if pair not in invalid_pairs and len(invalid_pairs) < 64:
                invalid_pairs.append(pair)

    env.set_robot_targets(arm_target, closed_target)
    env.advance(
        cfg.grasp_post_contact_settle_s,
        lambda _step, _steps, _progress: env.set_robot_targets(arm_target, closed_target),
        post_step=sample,
    )
    _, final_task_qd = env.read_task_state()
    _, final_arm_qd, _, final_finger_qd = env.read_robot_state()
    final_cable_speed = torch.linalg.vector_norm(final_task_qd[:, cable_slice, :3], dim=-1).amax(dim=-1)
    final_plug_linear_speed = torch.linalg.vector_norm(final_task_qd[:, plug_index, :3], dim=-1)
    final_plug_angular_speed = torch.linalg.vector_norm(final_task_qd[:, plug_index, 3:6], dim=-1)
    final_arm_speed = torch.abs(final_arm_qd).amax(dim=-1)
    final_finger_speed = torch.abs(final_finger_qd).amax(dim=-1)
    valid = (
        all_finite
        & all_collision_free
        & all_bilateral
        & all_grasp_valid
        & all_drives_disabled
        & (not any_overflow)
        & (final_cable_speed <= cfg.maximum_row_cable_speed_m_s)
        & (final_plug_linear_speed <= 0.04)
        & (final_plug_angular_speed <= 0.05)
        & (final_arm_speed <= cfg.maximum_row_arm_joint_speed_rad_s)
        & (final_finger_speed <= cfg.maximum_row_finger_joint_speed_m_s)
    )
    if lane_hold is not None:
        lane_hold.deactivate(~valid, reason="validator-post-contact-final-state")
    return valid, {
        "all_samples_finite": all_finite,
        "all_samples_collision_free": all_collision_free,
        "all_samples_bilateral_proxy_contact": all_bilateral,
        "all_samples_drives_disabled": all_drives_disabled,
        "any_contact_overflow": any_overflow,
        "invalid_contact_pairs": invalid_pairs,
        "maximum_cable_speed_m_s": maximum_cable_speed,
        "maximum_plug_linear_speed_m_s": maximum_plug_linear_speed,
        "maximum_plug_angular_speed_rad_s": maximum_plug_angular_speed,
        "final_cable_speed_m_s": final_cable_speed,
        "final_plug_linear_speed_m_s": final_plug_linear_speed,
        "final_plug_angular_speed_rad_s": final_plug_angular_speed,
        "final_arm_joint_speed_rad_s": final_arm_speed,
        "final_finger_joint_speed_m_s": final_finger_speed,
    }


def _acquire_physical_grasp(
    env: RJ45PickInsertResetToolEnv,
    ik: FrankaResetIK,
    arm_seed: torch.Tensor,
    cfg: ValidationCfg,
    *,
    active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Acquire active lanes independently and preserve every lane's last command."""
    open_target = torch.full((env.num_envs, 2), cfg.finger_open_position, device=env.device)
    initial_active = (
        torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        if active_mask is None
        else torch.as_tensor(active_mask, device=env.device, dtype=torch.bool)
    )
    with _PerLaneTargetHold(env, initial_active, arm_seed, open_target) as lane_hold:
        arm_target, valid, evidence = _acquire_physical_grasp_per_lane(
            env,
            ik,
            arm_seed,
            cfg,
            lane_hold=lane_hold,
        )
        lane_hold.deactivate(~valid, reason="validator-acquisition-final-validation")
        valid &= lane_hold.active_mask
        evidence["lane_failure_masks"] = {
            reason: mask.detach().cpu().tolist() for reason, mask in lane_hold.reason_masks.items()
        }
        evidence["last_arm_target"] = lane_hold.last_sent_arm_target.detach().cpu().tolist()
        evidence["last_finger_target"] = lane_hold.last_sent_finger_target.detach().cpu().tolist()
        return lane_hold.last_sent_arm_target, valid, evidence


def _acquire_physical_grasp_per_lane(
    env: RJ45PickInsertResetToolEnv,
    ik: FrankaResetIK,
    arm_seed: torch.Tensor,
    cfg: ValidationCfg,
    *,
    lane_hold: _PerLaneTargetHold,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Independently acquire the live plug through the proven guarded route."""
    layout = env.rj45_runtime.layout
    plug_index = int(layout.plug_body_index)
    open_target = torch.full((env.num_envs, 2), cfg.finger_open_position, device=env.device)
    closed_target = torch.full((env.num_envs, 2), cfg.finger_closed_target, device=env.device)
    reference_q, _ = env.read_task_state()
    reference_plug = reference_q[:, plug_index].clone()
    exact_position, orientation = _desired_tcp_pose(env, reference_plug)
    clearance_position = exact_position.clone()
    clearance_position[:, 2] += cfg.grasp_open_clearance_m
    true_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    false_mask = torch.zeros_like(true_mask)
    all_collision_free = true_mask.clone()
    all_zero_proxy = true_mask.clone()
    all_finite = true_mask.clone()
    all_drives_disabled = true_mask.clone()
    maximum_plug_drift = torch.zeros(env.num_envs, device=env.device)
    maximum_left_contacts = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    maximum_right_contacts = torch.zeros_like(maximum_left_contacts)
    invalid_pairs: list[str] = []
    any_overflow = False
    samples = 0
    abort_reason: str | None = None
    original_advance = env.advance

    def sample_open_approach() -> None:
        nonlocal abort_reason, any_overflow, samples
        task_q, task_qd = env.read_task_state()
        sample_collision = collision_metrics(env, require_bilateral_grasp=False)
        if sample_collision.contact_overflow:
            raise RuntimeError("Global contact-buffer overflow during validator open acquisition.")
        plug_drift = torch.linalg.vector_norm(task_q[:, plug_index, :3] - reference_plug[:, :3], dim=-1)
        zero_proxy = (sample_collision.left_grasp_contact_count == 0) & (
            sample_collision.right_grasp_contact_count == 0
        )
        finite = task_state_is_finite_and_normalized(task_q, task_qd)
        drives_disabled = torch.full_like(true_mask, _drive_disabled(env))
        if not bool(drives_disabled.all()):
            raise RuntimeError("A construction drive became enabled during validator open acquisition.")
        all_collision_free.logical_and_(sample_collision.valid)
        all_zero_proxy.logical_and_(zero_proxy)
        all_finite.logical_and_(finite)
        all_drives_disabled.logical_and_(drives_disabled)
        maximum_plug_drift.copy_(torch.maximum(maximum_plug_drift, plug_drift))
        maximum_left_contacts.copy_(torch.maximum(maximum_left_contacts, sample_collision.left_grasp_contact_count))
        maximum_right_contacts.copy_(torch.maximum(maximum_right_contacts, sample_collision.right_grasp_contact_count))
        any_overflow |= sample_collision.contact_overflow
        samples += 1
        for pair in sample_collision.invalid_contact_pairs:
            if pair not in invalid_pairs and len(invalid_pairs) < 64:
                invalid_pairs.append(pair)
        violated = (
            ~sample_collision.valid
            | ~zero_proxy
            | ~finite
            | ~drives_disabled
            | (plug_drift > cfg.maximum_open_approach_plug_drift_m)
        )
        if bool(violated.any()):
            reasons: list[str] = []
            if bool((~sample_collision.valid).any()):
                reasons.append("collision")
            if bool((~zero_proxy).any()):
                reasons.append("proxy-contact")
            if bool((~finite).any()):
                reasons.append("non-finite")
            if bool((~drives_disabled).any()):
                reasons.append("construction-drive")
            if bool((plug_drift > cfg.maximum_open_approach_plug_drift_m).any()):
                reasons.append("plug-drift")
            abort_reason = "+".join(reasons)
            lane_hold.deactivate(~sample_collision.valid, reason="validator-open-collision")
            lane_hold.deactivate(~zero_proxy, reason="validator-open-proxy-contact")
            lane_hold.deactivate(~finite, reason="validator-open-non-finite")
            lane_hold.deactivate(
                plug_drift > cfg.maximum_open_approach_plug_drift_m,
                reason="validator-open-plug-drift",
            )

    def guarded_advance(duration_s: float, update=None, *, post_step=None):
        def combined(step: int, steps: int, progress: float) -> None:
            if post_step is not None:
                post_step(step, steps, progress)
            sample_open_approach()

        return original_advance(duration_s, update, post_step=combined)

    route_valid = true_mask.clone()
    clearance_valid = false_mask.clone()
    descent_valid = false_mask.clone()
    clearance_error = torch.linalg.vector_norm(env.tcp_pose_e()[:, :3] - clearance_position, dim=-1)
    maximum_descent_error = torch.zeros(env.num_envs, device=env.device)
    arm_target = arm_seed.clone()
    env.advance = guarded_advance
    try:
        current_tcp = env.tcp_pose_e()[:, :3].clone()
        route_height = torch.maximum(
            current_tcp[:, 2],
            torch.full_like(current_tcp[:, 2], cfg.grasp_route_world_height_m),
        )
        lift_position = current_tcp.clone()
        lift_position[:, 2] = route_height
        arm_target, lift_valid = _move_tcp_for_acquisition(
            env,
            ik,
            lift_position,
            orientation,
            arm_target,
            open_target,
            cfg,
            lane_hold=lane_hold,
            failure_reason="validator-route-lift-ik",
        )
        route_valid &= lift_valid
        lane_hold.deactivate(~lift_valid, reason="validator-route-lift-ik")
        arm_target = lane_hold.last_sent_arm_target
        overhead_position = exact_position.clone()
        overhead_position[:, 2] = route_height
        overhead_distance = torch.linalg.vector_norm(overhead_position - lift_position, dim=-1)
        overhead_steps, invalid_overhead_distance = _active_waypoint_count(
            overhead_distance,
            lane_hold.active_mask,
            cfg.grasp_route_maximum_translation_step_m,
        )
        route_valid &= ~invalid_overhead_distance
        lane_hold.deactivate(invalid_overhead_distance, reason="validator-route-distance-invalid")
        for waypoint in range(1, overhead_steps + 1):
            waypoint_position = torch.lerp(lift_position, overhead_position, waypoint / overhead_steps)
            arm_target, waypoint_valid = _move_tcp_for_acquisition(
                env,
                ik,
                waypoint_position,
                orientation,
                arm_target,
                open_target,
                cfg,
                motion_s=cfg.grasp_descent_waypoint_motion_s,
                lane_hold=lane_hold,
                failure_reason="validator-route-overhead-ik",
            )
            route_valid &= waypoint_valid
            lane_hold.deactivate(~waypoint_valid, reason="validator-route-overhead-ik")
            arm_target = lane_hold.last_sent_arm_target

        raw_route_active = lane_hold.active_mask
        if bool(raw_route_active.any()):
            current_tcp_pose = env.tcp_pose_e()
            fallback_orientation = torch.zeros_like(orientation)
            fallback_orientation[:, 3] = 1.0
            fallback_orientation = torch.where(
                torch.isfinite(current_tcp_pose[:, 3:7]).all(dim=-1, keepdim=True),
                current_tcp_pose[:, 3:7],
                fallback_orientation,
            )
            raw_route = ik.solve(
                torch.where(
                    raw_route_active[:, None],
                    overhead_position,
                    torch.nan_to_num(current_tcp_pose[:, :3]),
                ),
                torch.where(raw_route_active[:, None], orientation, fallback_orientation),
                torch.where(raw_route_active[:, None], open_target, lane_hold.last_sent_finger_target),
                arm_seed=arm_target,
            )
            raw_route_valid = raw_route_active & raw_route.valid & joint_limit_mask(env, arm_target, margin=0.02)
            raw_arm = torch.where(raw_route_valid[:, None], raw_route.arm_q, arm_target)
        else:
            raw_route_valid = torch.zeros_like(raw_route_active)
            raw_arm = arm_target
        route_valid &= ~raw_route_active | raw_route_valid
        lane_hold.deactivate(raw_route_active & ~raw_route_valid, reason="validator-route-bias-ik")
        arm_target = lane_hold.last_sent_arm_target
        route_bias = arm_target - raw_arm
        coarse_distance = torch.clamp(route_height - clearance_position[:, 2], min=0.0)
        coarse_steps, invalid_coarse_distance = _active_waypoint_count(
            coarse_distance,
            lane_hold.active_mask,
            cfg.grasp_coarse_descent_step_m,
        )
        route_valid &= ~invalid_coarse_distance
        lane_hold.deactivate(invalid_coarse_distance, reason="validator-coarse-distance-invalid")
        for waypoint in range(1, coarse_steps + 1):
            waypoint_position = torch.lerp(overhead_position, clearance_position, waypoint / coarse_steps)
            raw_arm, arm_target, waypoint_valid = _move_tcp_with_fixed_bias(
                env,
                ik,
                waypoint_position,
                orientation,
                open_target,
                raw_arm_seed=raw_arm,
                arm_target_bias=route_bias,
                cfg=cfg,
                maximum_raw_joint_step_rad=cfg.maximum_ik_joint_step_rad,
                lane_hold=lane_hold,
                failure_reason="validator-coarse-ik-continuity",
            )
            route_valid &= waypoint_valid
            lane_hold.deactivate(~waypoint_valid, reason="validator-coarse-ik-continuity")
            arm_target = lane_hold.last_sent_arm_target

        clearance_command = clearance_position.clone()
        for _iteration in range(cfg.grasp_clearance_calibration_max_iterations):
            clearance_error = clearance_position - env.tcp_pose_e()[:, :3]
            clearance_error_norm = torch.linalg.vector_norm(clearance_error, dim=-1)
            correction_needed = lane_hold.active_mask & (clearance_error_norm > 0.002)
            if not bool(correction_needed.any()):
                break
            correction = clearance_error * torch.clamp(
                cfg.grasp_clearance_calibration_step_m / clearance_error_norm[:, None].clamp_min(1.0e-9),
                max=1.0,
            )
            correction = torch.where(correction_needed[:, None], correction, torch.zeros_like(correction))
            clearance_command += correction
            raw_arm, arm_target, calibration_valid = _move_tcp_with_fixed_bias(
                env,
                ik,
                clearance_command,
                orientation,
                open_target,
                raw_arm_seed=raw_arm,
                arm_target_bias=route_bias,
                cfg=cfg,
                maximum_raw_joint_step_rad=cfg.grasp_near_maximum_raw_ik_joint_step_rad,
                lane_hold=lane_hold,
                failure_reason="validator-clearance-ik-continuity",
            )
            route_valid &= calibration_valid
            lane_hold.deactivate(~calibration_valid, reason="validator-clearance-ik-continuity")
            arm_target = lane_hold.last_sent_arm_target
        clearance_error = torch.linalg.vector_norm(env.tcp_pose_e()[:, :3] - clearance_position, dim=-1)
        clearance_valid = route_valid & (clearance_error <= 0.002)
        lane_hold.deactivate(~clearance_valid, reason="validator-clearance-final-state")
        arm_target = lane_hold.last_sent_arm_target

        near_steps = max(
            1, int(torch.ceil(torch.tensor(cfg.grasp_open_clearance_m / cfg.grasp_near_descent_step_m)).item())
        )
        near_command = clearance_command.clone()
        previous_waypoint = clearance_position.clone()
        descent_valid = lane_hold.active_mask
        for waypoint in range(1, near_steps + 1):
            waypoint_position = torch.lerp(clearance_position, exact_position, waypoint / near_steps)
            near_command += waypoint_position - previous_waypoint
            raw_arm, arm_target, waypoint_valid = _move_tcp_with_fixed_bias(
                env,
                ik,
                near_command,
                orientation,
                open_target,
                raw_arm_seed=raw_arm,
                arm_target_bias=route_bias,
                cfg=cfg,
                maximum_raw_joint_step_rad=cfg.grasp_near_maximum_raw_ik_joint_step_rad,
                lane_hold=lane_hold,
                failure_reason="validator-near-ik-continuity",
            )
            waypoint_error = torch.linalg.vector_norm(env.tcp_pose_e()[:, :3] - waypoint_position, dim=-1)
            lane_hold.deactivate(~waypoint_valid, reason="validator-near-ik-continuity")
            arm_target = lane_hold.last_sent_arm_target
            recovery_needed = lane_hold.active_mask & (waypoint_error > 0.002)
            if bool(recovery_needed.any()):
                env.set_robot_targets(arm_target, open_target)
                env.advance(cfg.grasp_descent_tracking_recovery_s)
                waypoint_error = torch.linalg.vector_norm(env.tcp_pose_e()[:, :3] - waypoint_position, dim=-1)
            correction_count = 0
            while (
                bool((lane_hold.active_mask & (waypoint_error > 0.002)).any())
                and correction_count < cfg.grasp_near_correction_max_iterations
            ):
                correction_needed = lane_hold.active_mask & (waypoint_error > 0.002)
                error_vector = waypoint_position - env.tcp_pose_e()[:, :3]
                error_norm = torch.linalg.vector_norm(error_vector, dim=-1)
                correction = error_vector * torch.clamp(
                    cfg.grasp_near_correction_step_m / error_norm[:, None].clamp_min(1.0e-9),
                    max=1.0,
                )
                near_command += torch.where(
                    correction_needed[:, None],
                    correction,
                    torch.zeros_like(correction),
                )
                correction_count += 1
                raw_arm, arm_target, correction_valid = _move_tcp_with_fixed_bias(
                    env,
                    ik,
                    near_command,
                    orientation,
                    open_target,
                    raw_arm_seed=raw_arm,
                    arm_target_bias=route_bias,
                    cfg=cfg,
                    maximum_raw_joint_step_rad=cfg.grasp_near_maximum_raw_ik_joint_step_rad,
                    lane_hold=lane_hold,
                    failure_reason="validator-near-correction-ik-continuity",
                )
                waypoint_valid &= correction_valid
                lane_hold.deactivate(~correction_valid, reason="validator-near-correction-ik-continuity")
                arm_target = lane_hold.last_sent_arm_target
                waypoint_error = torch.linalg.vector_norm(env.tcp_pose_e()[:, :3] - waypoint_position, dim=-1)
            waypoint_valid &= waypoint_error <= 0.002
            maximum_descent_error = torch.maximum(maximum_descent_error, waypoint_error)
            descent_valid &= waypoint_valid
            lane_hold.deactivate(waypoint_error > 0.002, reason="validator-near-tracking")
            arm_target = lane_hold.last_sent_arm_target
            previous_waypoint = waypoint_position
    finally:
        env.advance = original_advance

    contact_error = torch.linalg.vector_norm(env.tcp_pose_e()[:, :3] - exact_position, dim=-1)
    contact_preclose = collision_metrics(env, require_bilateral_grasp=False)
    if contact_preclose.contact_overflow:
        raise RuntimeError("Global contact-buffer overflow at the validator acquisition pre-close boundary.")
    approach_valid = (
        lane_hold.active_mask
        & route_valid
        & clearance_valid
        & descent_valid
        & all_collision_free
        & all_zero_proxy
        & all_finite
        & all_drives_disabled
        & (maximum_plug_drift <= cfg.maximum_open_approach_plug_drift_m)
        & (not any_overflow)
        & contact_preclose.valid
        & (contact_error <= 0.002)
    )
    lane_hold.deactivate(~approach_valid, reason="validator-open-final-validation")
    arm_target = lane_hold.last_sent_arm_target
    if bool(lane_hold.active_mask.any()):
        _close_gripper(env, arm_target, open_target, closed_target, cfg)
        settle_valid, settle_evidence = _settle_physical_grasp(
            env,
            arm_target,
            closed_target,
            cfg,
            lane_hold=lane_hold,
        )
    else:
        settle_valid = false_mask
        settle_evidence = {
            "all_samples_finite": false_mask,
            "all_samples_collision_free": false_mask,
            "all_samples_bilateral_proxy_contact": false_mask,
            "all_samples_drives_disabled": false_mask,
            "any_contact_overflow": False,
            "invalid_contact_pairs": [],
            "maximum_cable_speed_m_s": torch.zeros(env.num_envs, device=env.device),
            "maximum_plug_linear_speed_m_s": torch.zeros(env.num_envs, device=env.device),
            "maximum_plug_angular_speed_rad_s": torch.zeros(env.num_envs, device=env.device),
            "final_cable_speed_m_s": torch.zeros(env.num_envs, device=env.device),
            "final_plug_linear_speed_m_s": torch.zeros(env.num_envs, device=env.device),
            "final_plug_angular_speed_rad_s": torch.zeros(env.num_envs, device=env.device),
            "final_arm_joint_speed_rad_s": torch.zeros(env.num_envs, device=env.device),
            "final_finger_joint_speed_m_s": torch.zeros(env.num_envs, device=env.device),
        }
    grasp = grasp_metrics(env, closed_target)
    collision = collision_metrics(env)
    if collision.contact_overflow:
        raise RuntimeError("Global contact-buffer overflow at the validator acquisition result boundary.")
    valid = approach_valid & lane_hold.active_mask & settle_valid & grasp.valid & collision.valid
    lane_hold.deactivate(~grasp.valid, reason="validator-acquisition-final-grasp")
    lane_hold.deactivate(~collision.valid, reason="validator-acquisition-final-collision")
    lane_hold.deactivate(~valid, reason="validator-acquisition-result-validation")
    valid &= lane_hold.active_mask
    arm_target = lane_hold.last_sent_arm_target
    abort_reason = "+".join(lane_hold.reason_masks) or None
    evidence = {
        "open_clearance_above_cfg_target_m": cfg.grasp_open_clearance_m,
        "route_world_height_m": cfg.grasp_route_world_height_m,
        "approach_abort_reason": abort_reason,
        "approach_samples": samples,
        "clearance_approach_valid": clearance_valid.detach().cpu().tolist(),
        "clearance_tcp_error_m": clearance_error.detach().cpu().tolist(),
        "open_descent_valid": descent_valid.detach().cpu().tolist(),
        "maximum_open_descent_tcp_error_m": maximum_descent_error.detach().cpu().tolist(),
        "open_approach_all_samples_collision_free": all_collision_free.detach().cpu().tolist(),
        "open_approach_all_samples_zero_proxy_contacts": all_zero_proxy.detach().cpu().tolist(),
        "open_approach_all_samples_finite": all_finite.detach().cpu().tolist(),
        "open_approach_all_samples_drives_disabled": all_drives_disabled.detach().cpu().tolist(),
        "open_approach_maximum_plug_drift_m": maximum_plug_drift.detach().cpu().tolist(),
        "open_approach_maximum_left_proxy_contacts": maximum_left_contacts.detach().cpu().tolist(),
        "open_approach_maximum_right_proxy_contacts": maximum_right_contacts.detach().cpu().tolist(),
        "open_approach_any_contact_overflow": any_overflow,
        "open_approach_invalid_pairs": invalid_pairs,
        "contact_preclose_invalid_contacts": contact_preclose.invalid_contact_count.detach().cpu().tolist(),
        "contact_preclose_tcp_error_m": contact_error.detach().cpu().tolist(),
        "maximum_tcp_distance_m": float(grasp.tcp_distance.max()),
        "minimum_bilateral_deflection_m": float(grasp.bilateral_deflection.min()),
        "left_proxy_contacts": collision.left_grasp_contact_count.detach().cpu().tolist(),
        "right_proxy_contacts": collision.right_grasp_contact_count.detach().cpu().tolist(),
        "invalid_contacts": collision.invalid_contact_count.detach().cpu().tolist(),
        "post_contact_settle": {
            key: value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
            for key, value in settle_evidence.items()
        },
    }
    return arm_target, valid, evidence


def _goal_replay(
    env: RJ45PickInsertResetToolEnv,
    goal: dict[str, torch.Tensor],
    cfg: ValidationCfg,
) -> dict[str, Any]:
    repeated = _repeat_goal_state(goal, env)
    history_evidence = _write_state(env, repeated)
    reset_contact_count = _contact_count()
    start_q, start_qd = env.read_task_state()
    exact_success, exact_metrics = advance_exact_success_dwell(
        env,
        repeated["task_body_pose"],
        repeated["arm_joint_target"],
        repeated["finger_joint_target"],
        duration_s=cfg.goal_replay_s,
        require_all_samples=True,
        sample_physical_validity=True,
        arm_target_is_absolute=True,
    )
    final_q, final_qd = env.read_task_state()
    layout = env.rj45_runtime.layout
    socket_index = int(layout.socket_body_index)
    plug_index = int(layout.plug_body_index)
    latch_index = int(layout.latch_body_index)
    cable_slice = layout.cable_body_slice
    body_drift = torch.linalg.vector_norm(final_q[..., :3] - start_q[..., :3], dim=-1)
    socket_drift = body_drift[:, socket_index]
    cable_start_speed = torch.linalg.vector_norm(start_qd[:, cable_slice, :3], dim=-1).amax(dim=-1)
    cable_final_speed = torch.linalg.vector_norm(final_qd[:, cable_slice, :3], dim=-1).amax(dim=-1)
    finite = task_state_is_finite_and_normalized(final_q, final_qd)
    authored_target_e = wp.to_torch(env.rj45_runtime.default_goal_target_w) - env.env_origins
    authored_orientation = wp.to_torch(env.rj45_runtime.default_orientation_target_w)
    stored_authored_seat_error = torch.linalg.vector_norm(
        start_q[:, plug_index, :3] - authored_target_e,
        dim=-1,
    )
    final_authored_seat_error = torch.linalg.vector_norm(
        final_q[:, plug_index, :3] - authored_target_e,
        dim=-1,
    )
    stored_authored_plug_angle = math_utils.quat_error_magnitude(
        start_q[:, plug_index, 3:7],
        authored_orientation,
    )
    final_authored_plug_angle = math_utils.quat_error_magnitude(
        final_q[:, plug_index, 3:7],
        authored_orientation,
    )
    stored_plug_relative_latch_angle = plug_relative_latch_angle(
        start_q,
        plug_body_index=plug_index,
        latch_body_index=latch_index,
    )
    final_plug_relative_latch_angle = plug_relative_latch_angle(
        final_q,
        plug_body_index=plug_index,
        latch_body_index=latch_index,
    )
    authored_geometry_valid = (
        (stored_authored_seat_error <= cfg.maximum_goal_authored_seat_error_m)
        & (final_authored_seat_error <= cfg.maximum_goal_authored_seat_error_m)
        & (stored_authored_plug_angle <= cfg.maximum_goal_authored_plug_angle_rad)
        & (final_authored_plug_angle <= cfg.maximum_goal_authored_plug_angle_rad)
        & (stored_plug_relative_latch_angle <= cfg.maximum_goal_plug_relative_latch_angle_rad)
        & (final_plug_relative_latch_angle <= cfg.maximum_goal_plug_relative_latch_angle_rad)
    )
    grasp = grasp_metrics(env, repeated["finger_joint_target"], retaining_grasp=True)
    collision = collision_metrics(env)
    if collision.contact_overflow:
        raise RuntimeError("Global contact-buffer overflow at the canonical goal replay boundary.")
    socket_stable = socket_drift <= cfg.maximum_goal_socket_drift_m
    body_stable = (body_drift.amax(dim=-1) <= cfg.maximum_goal_body_drift_m) & (
        exact_metrics["maximum_body_excursion"] <= cfg.maximum_goal_body_drift_m
    )
    cable_stable = (torch.maximum(cable_start_speed, cable_final_speed) <= cfg.maximum_goal_cable_speed_m_s) & (
        exact_metrics["maximum_cable_linear_speed"] <= cfg.maximum_goal_cable_speed_m_s
    )
    sampled_physical_validity = (
        exact_metrics["all_samples_collision_free"]
        & exact_metrics["all_samples_bilateral_grasp"]
        & exact_metrics["all_samples_finite"]
    )
    robot_equilibrium = (exact_metrics["maximum_arm_joint_speed"] <= cfg.maximum_goal_arm_joint_speed_rad_s) & (
        exact_metrics["maximum_finger_joint_speed"] <= cfg.maximum_goal_finger_joint_speed_m_s
    )
    zero_action_unclamped = ~exact_metrics["any_arm_target_clamped"]
    absolute_target_stable = exact_metrics["maximum_arm_target_drift"] <= 1.0e-7
    target_tracking_bounded = exact_metrics["all_samples_arm_target_tracking_bounded"]
    no_contact_overflow = not exact_metrics["any_contact_overflow"]
    drive_disabled = _drive_disabled(env)
    if not drive_disabled:
        raise RuntimeError("A construction drive became enabled during canonical goal replay.")
    history_applied = _vbd_pose_history_applied_mask(env, history_evidence)
    passed = (
        finite
        & grasp.valid
        & collision.valid
        & socket_stable
        & body_stable
        & cable_stable
        & exact_success
        & sampled_physical_validity
        & robot_equilibrium
        & zero_action_unclamped
        & absolute_target_stable
        & target_tracking_bounded
        & authored_geometry_valid
        & no_contact_overflow
        & history_applied
    )
    steps = int(exact_metrics["sample_steps"])
    return {
        "passed": bool(passed.all()) and drive_disabled and reset_contact_count == 0,
        "drive_disabled": drive_disabled,
        "vbd_pose_history_restore_queued": bool(torch.as_tensor(history_evidence["restore_queued"]).all()),
        "vbd_pose_history_pending_at_queue": bool(torch.as_tensor(history_evidence["pending_at_queue"]).all()),
        "vbd_previous_pose_queued": bool(torch.as_tensor(history_evidence["previous_pose_queued"]).all()),
        "vbd_coupling_previous_pose_queued": bool(
            torch.as_tensor(history_evidence["coupling_previous_pose_queued"]).all()
        ),
        "vbd_pose_history_applied_exactly_once": bool(history_applied.all()),
        "vbd_pose_history_failed": bool(torch.as_tensor(history_evidence["failed"]).any()),
        "vbd_pose_history_superseded": bool(torch.as_tensor(history_evidence["superseded"]).any()),
        "vbd_pose_history_pending_after_first_solve": bool(
            torch.as_tensor(history_evidence["pending_after_first_solve"]).any()
        ),
        "vbd_pose_history_minimum_application_count_delta": int(
            torch.as_tensor(history_evidence["application_count_delta"]).min()
        ),
        "vbd_pose_history_maximum_application_count_delta": int(
            torch.as_tensor(history_evidence["application_count_delta"]).max()
        ),
        "vbd_pose_history_expected_body_count": int(torch.as_tensor(history_evidence["expected_body_count"]).min()),
        "vbd_pose_history_minimum_body_application_count_delta": int(
            torch.as_tensor(history_evidence["body_application_count_delta"]).min()
        ),
        "vbd_pose_history_maximum_body_application_count_delta": int(
            torch.as_tensor(history_evidence["body_application_count_delta"]).max()
        ),
        "vbd_pose_history_generation": int(history_evidence["generation"]),
        "vbd_pose_history_body_order_exact": history_evidence["body_order_exact"] is True,
        "vbd_pose_history_world_order_exact": history_evidence["world_order_exact"] is True,
        "vbd_pose_history_entry_name": history_evidence["entry_name"],
        "vbd_pose_history_body_count": history_evidence["body_count"],
        "socket_stable": bool(socket_stable.all()),
        "whole_cable_stable": bool(cable_stable.all()),
        "exact_runtime_success_dwell": bool(exact_success.all()),
        "simulation_steps": steps,
        "simulation_time_s": steps * env.advance_dt,
        "contact_count_after_history_reset": reset_contact_count,
        "maximum_socket_drift_m": float(socket_drift.max()),
        "maximum_task_body_drift_m": float(body_drift.max()),
        "maximum_start_cable_speed_m_s": float(cable_start_speed.max()),
        "maximum_final_cable_speed_m_s": float(cable_final_speed.max()),
        "sampled_maximum_task_body_excursion_m": float(exact_metrics["maximum_body_excursion"].max()),
        "sampled_maximum_cable_speed_m_s": float(exact_metrics["maximum_cable_linear_speed"].max()),
        "sampled_maximum_arm_joint_speed_rad_s": float(exact_metrics["maximum_arm_joint_speed"].max()),
        "sampled_maximum_finger_joint_speed_m_s": float(exact_metrics["maximum_finger_joint_speed"].max()),
        "maximum_allowed_socket_drift_m": cfg.maximum_goal_socket_drift_m,
        "maximum_allowed_task_body_drift_m": cfg.maximum_goal_body_drift_m,
        "maximum_allowed_cable_speed_m_s": cfg.maximum_goal_cable_speed_m_s,
        "maximum_allowed_arm_joint_speed_rad_s": cfg.maximum_goal_arm_joint_speed_rad_s,
        "maximum_allowed_finger_joint_speed_m_s": cfg.maximum_goal_finger_joint_speed_m_s,
        "robot_equilibrium": bool(robot_equilibrium.all()),
        "zero_action_unclamped": bool(zero_action_unclamped.all()),
        "maximum_arm_target_clamp_delta_rad": float(exact_metrics["maximum_arm_target_clamp_delta"].max()),
        "controller_semantics": exact_metrics["arm_target_semantics"],
        "absolute_target_stable": bool(absolute_target_stable.all()),
        "maximum_arm_target_drift_rad": float(exact_metrics["maximum_arm_target_drift"].max()),
        "all_samples_arm_target_tracking_bounded": bool(target_tracking_bounded.all()),
        "maximum_arm_target_tracking_error_by_joint_rad": exact_metrics["maximum_arm_target_tracking_error_by_joint"]
        .amax(dim=0)
        .detach()
        .cpu()
        .tolist(),
        "authored_goal_geometry_valid": bool(authored_geometry_valid.all()),
        "maximum_allowed_authored_seat_error_m": cfg.maximum_goal_authored_seat_error_m,
        "maximum_allowed_authored_plug_angle_rad": cfg.maximum_goal_authored_plug_angle_rad,
        "maximum_allowed_plug_relative_latch_angle_rad": cfg.maximum_goal_plug_relative_latch_angle_rad,
        "maximum_stored_authored_seat_error_m": float(stored_authored_seat_error.max()),
        "maximum_final_authored_seat_error_m": float(final_authored_seat_error.max()),
        "maximum_stored_authored_plug_angle_rad": float(stored_authored_plug_angle.max()),
        "maximum_final_authored_plug_angle_rad": float(final_authored_plug_angle.max()),
        "maximum_stored_plug_relative_latch_angle_rad": float(stored_plug_relative_latch_angle.max()),
        "maximum_final_plug_relative_latch_angle_rad": float(final_plug_relative_latch_angle.max()),
        "all_samples_collision_free": bool(exact_metrics["all_samples_collision_free"].all()),
        "all_samples_bilateral_grasp": bool(exact_metrics["all_samples_bilateral_grasp"].all()),
        "all_samples_proxy_bilateral_contact": bool(exact_metrics["all_samples_proxy_bilateral_contact"].all()),
        "all_samples_finite": bool(exact_metrics["all_samples_finite"].all()),
        "minimum_left_proxy_contact_count": int(exact_metrics["minimum_left_grasp_contact_count"].min()),
        "minimum_right_proxy_contact_count": int(exact_metrics["minimum_right_grasp_contact_count"].min()),
        "maximum_invalid_contact_count": int(exact_metrics["maximum_invalid_contact_count"].max()),
        "any_contact_overflow": exact_metrics["any_contact_overflow"],
        "no_contact_overflow": no_contact_overflow,
        "sampled_invalid_contact_pairs": exact_metrics["sampled_invalid_contact_pairs"],
        "stored_capture_exact_success": bool(exact_metrics["stored_capture_success"].all()),
        "all_post_step_exact_success": bool(exact_metrics["all_post_step_success"].all()),
        "required_dwell_steps": int(exact_metrics["required_dwell_steps"]),
        "final_consecutive_steps": exact_metrics["final_consecutive_steps"].detach().cpu().tolist(),
        "collision_valid": bool(collision.valid.all()),
        "closed_bilateral_grasp": bool(grasp.valid.all()),
    }


def _phase_semantics(
    env: RJ45PickInsertResetToolEnv,
    phase: int,
    task_q: torch.Tensor,
    goal_q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layout = env.rj45_runtime.layout
    plug_index = int(layout.plug_body_index)
    latch_index = int(layout.latch_body_index)
    tcp_distance = torch.linalg.vector_norm(env.tcp_pose_e()[:, :3] - env.plug_grasp_position_e(), dim=-1)
    goal_error = scalar_goal_error(
        task_q,
        goal_q,
        plug_body_index=plug_index,
        latch_body_index=latch_index,
    )
    if phase == 0:
        semantic = goal_error <= 0.020
    elif phase == 1:
        semantic = (goal_error >= 0.015) & (goal_error <= 0.090)
    elif phase == 2:
        semantic = (task_q[:, plug_index, 2] >= 0.06) & (goal_error >= 0.05)
    elif phase == 3:
        semantic = (task_q[:, plug_index, 2] >= 0.02) & (goal_error >= 0.08)
    elif phase == 4:
        semantic = (tcp_distance >= 0.02) & (tcp_distance <= 0.10)
    else:
        semantic = tcp_distance >= 0.10
    return semantic, goal_error, tcp_distance


def _reset_replay_row_evidence(
    evidence: dict[str, Any],
    *,
    history_evidence: dict[str, object],
    local: int,
    starts_grasped: bool,
    maximum_arm_target_clamp_delta: torch.Tensor,
    zero_action_unclamped: torch.Tensor,
    cfg: ValidationCfg,
) -> dict[str, Any]:
    """Serialize one world's raw, independently measured row replay evidence."""

    def flag(name: str) -> bool:
        return bool(evidence[name][local])

    def scalar(name: str) -> float:
        return float(evidence[name][local])

    def count(name: str) -> int:
        return int(evidence[name][local])

    return {
        "simulation_time_s": cfg.row_settle_s,
        "simulation_steps": int(evidence["post_step_samples"]),
        "post_step_samples": int(evidence["post_step_samples"]),
        "required_simulation_time_s": PICK_INSERT_RESET_REPLAY_DURATION_S,
        "required_post_step_samples": PICK_INSERT_RESET_REPLAY_POST_STEP_SAMPLES,
        "starts_grasped": starts_grasped,
        "contact_expectation": "bilateral-proxy" if starts_grasped else "zero-proxy",
        "vbd_pose_history_restore_queued": bool(torch.as_tensor(history_evidence["restore_queued"])[local]),
        "vbd_pose_history_pending_at_queue": bool(torch.as_tensor(history_evidence["pending_at_queue"])[local]),
        "vbd_previous_pose_queued": bool(torch.as_tensor(history_evidence["previous_pose_queued"])[local]),
        "vbd_coupling_previous_pose_queued": bool(
            torch.as_tensor(history_evidence["coupling_previous_pose_queued"])[local]
        ),
        "vbd_pose_history_applied_exactly_once": bool(torch.as_tensor(history_evidence["applied_exactly_once"])[local]),
        "vbd_pose_history_failed": bool(torch.as_tensor(history_evidence["failed"])[local]),
        "vbd_pose_history_superseded": bool(torch.as_tensor(history_evidence["superseded"])[local]),
        "vbd_pose_history_pending_after_first_solve": bool(
            torch.as_tensor(history_evidence["pending_after_first_solve"])[local]
        ),
        "vbd_pose_history_application_count_delta": int(
            torch.as_tensor(history_evidence["application_count_delta"])[local]
        ),
        "vbd_pose_history_expected_body_count": int(torch.as_tensor(history_evidence["expected_body_count"])[local]),
        "vbd_pose_history_body_application_count_delta": int(
            torch.as_tensor(history_evidence["body_application_count_delta"])[local]
        ),
        "vbd_pose_history_generation": int(history_evidence["generation"]),
        "vbd_pose_history_body_order_exact": history_evidence["body_order_exact"] is True,
        "vbd_pose_history_world_order_exact": history_evidence["world_order_exact"] is True,
        "vbd_pose_history_entry_name": history_evidence["entry_name"],
        "vbd_pose_history_body_count": history_evidence["body_count"],
        "stored_state_finite": flag("stored_state_finite"),
        "stored_task_state_finite_and_normalized": flag("stored_task_state_finite_and_normalized"),
        "stored_drive_disabled": flag("stored_drive_disabled"),
        "stored_maximum_cable_speed_m_s": scalar("stored_maximum_cable_speed_m_s"),
        "stored_maximum_arm_joint_speed_rad_s": scalar("stored_maximum_arm_joint_speed_rad_s"),
        "stored_maximum_finger_joint_speed_m_s": scalar("stored_maximum_finger_joint_speed_m_s"),
        "stored_maximum_arm_target_tracking_error_rad": scalar("stored_maximum_arm_target_tracking_error_rad"),
        "stored_arm_target_tracking_error_by_joint_rad": evidence["stored_arm_target_tracking_error_by_joint_rad"][
            local
        ]
        .detach()
        .cpu()
        .tolist(),
        "stored_arm_target_tracking_bounded": flag("stored_arm_target_tracking_bounded"),
        "all_post_step_state_finite": flag("all_post_step_state_finite"),
        "all_post_step_task_state_finite_and_normalized": flag("all_post_step_task_state_finite_and_normalized"),
        "all_post_step_collision_free": flag("all_post_step_collision_free"),
        "all_post_step_drive_disabled": flag("all_post_step_drive_disabled"),
        "all_post_step_expected_contact_state": flag("all_post_step_expected_contact_state"),
        "all_post_step_bilateral_grasp": flag("all_post_step_bilateral_grasp"),
        "all_post_step_proxy_bilateral_contact": flag("all_post_step_proxy_bilateral_contact"),
        "all_post_step_zero_proxy_contacts": flag("all_post_step_zero_proxy_contacts"),
        "all_post_step_arm_target_tracking_bounded": flag("all_post_step_arm_target_tracking_bounded"),
        "maximum_body_excursion_m": scalar("maximum_body_excursion_m"),
        "maximum_plug_excursion_m": scalar("maximum_plug_excursion_m"),
        "maximum_socket_excursion_m": scalar("maximum_socket_excursion_m"),
        "maximum_post_step_cable_speed_m_s": scalar("maximum_post_step_cable_speed_m_s"),
        "maximum_post_step_arm_joint_speed_rad_s": scalar("maximum_post_step_arm_joint_speed_rad_s"),
        "maximum_post_step_finger_joint_speed_m_s": scalar("maximum_post_step_finger_joint_speed_m_s"),
        "maximum_post_step_arm_target_tracking_error_rad": scalar("maximum_post_step_arm_target_tracking_error_rad"),
        "maximum_arm_target_tracking_error_by_joint_rad": evidence[
            "maximum_post_step_arm_target_tracking_error_by_joint_rad"
        ][local]
        .detach()
        .cpu()
        .tolist(),
        "arm_target_semantics": evidence["arm_target_semantics"],
        "arm_target_tracking_limits_rad": list(PICK_INSERT_ARM_TARGET_TRACKING_LIMITS),
        "maximum_arm_target_drift_rad": scalar("maximum_arm_target_drift_rad"),
        "absolute_target_stable": scalar("maximum_arm_target_drift_rad") <= 1.0e-7,
        "final_cable_speed_m_s": scalar("final_cable_speed_m_s"),
        "final_arm_joint_speed_rad_s": scalar("final_arm_joint_speed_rad_s"),
        "final_finger_joint_speed_m_s": scalar("final_finger_joint_speed_m_s"),
        "minimum_left_proxy_contact_count": count("minimum_left_proxy_contact_count"),
        "minimum_right_proxy_contact_count": count("minimum_right_proxy_contact_count"),
        "maximum_left_proxy_contact_count": count("maximum_left_proxy_contact_count"),
        "maximum_right_proxy_contact_count": count("maximum_right_proxy_contact_count"),
        "maximum_invalid_contact_count": count("maximum_invalid_contact_count"),
        "no_contact_overflow": not evidence["any_contact_overflow"],
        "invalid_contact_pairs": list(evidence["invalid_contact_pairs"]),
        "maximum_arm_target_clamp_delta_rad": float(maximum_arm_target_clamp_delta[local]),
        "zero_action_unclamped": bool(zero_action_unclamped[local]),
        "maximum_allowed_arm_joint_speed_rad_s": cfg.maximum_row_arm_joint_speed_rad_s,
        "maximum_allowed_finger_joint_speed_m_s": cfg.maximum_row_finger_joint_speed_m_s,
        "maximum_allowed_cable_speed_m_s": cfg.maximum_row_cable_speed_m_s,
        "maximum_allowed_body_excursion_m": cfg.maximum_row_body_drift_m,
        "maximum_allowed_plug_excursion_m": cfg.maximum_row_plug_drift_m,
        "maximum_allowed_socket_excursion_m": cfg.maximum_row_socket_drift_m,
        "maximum_allowed_arm_target_clamp_delta_rad": PICK_INSERT_RESET_MAX_ARM_TARGET_CLAMP_DELTA_RAD,
    }


def _row_report(
    *,
    row_id: int,
    phase: int,
    checks: dict[str, bool],
    oracle: dict[str, bool],
    reset_replay: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "phase": phase,
        "passed": all(checks.values()) and all(oracle.values()),
        "checks": checks,
        "oracle": oracle,
        "reset_replay": reset_replay,
        "metrics": metrics,
    }


@torch.inference_mode()
def validate_payload(  # noqa: C901
    env: RJ45PickInsertResetToolEnv,
    payload: dict[str, Any],
    cfg: ValidationCfg,
    *,
    resume_progress: Mapping[str, Any] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    source_sha256: Mapping[str, str] | None = None,
    asset_closure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay an already-loaded artifact using independent physical checks."""
    gripper_cfg = env.cfg.actions.gripper_action
    if (
        cfg.finger_open_position != PICK_INSERT_OPEN_FINGER_POSITION
        or float(gripper_cfg.neutral_position) != cfg.finger_open_position
        or float(gripper_cfg.default_position) != cfg.finger_open_position
    ):
        raise ValueError("Validator and live pick-insert gripper must share the exact 0.04 m open posture.")
    physical_contract = pick_insert_tool_physical_contract(
        env,
        finger_closed_target=cfg.finger_closed_target,
    )
    contract = pick_insert_reset_dataset_task_contract(env.cfg)
    metadata, states_raw, goal_raw = reset_dataset_validate_runtime(
        payload,
        expected_task_contract=contract,
    )
    states = {name: tensor.detach().cpu() for name, tensor in states_raw.items()}
    goal = {name: tensor.detach() for name, tensor in goal_raw.items()}
    selected = _selected_rows(states, cfg)
    validation_source = dict(_validation_source_digests() if source_sha256 is None else source_sha256)
    validation_assets = dict(_validated_asset_closure_snapshot() if asset_closure is None else asset_closure)
    if reset_dataset_digest(validation_assets) != reset_dataset_digest(contract.get("external_assets")):
        raise ValueError("Validator asset closure does not exactly match the task contract.")

    created_utc = datetime.now(UTC).isoformat()
    goal_result: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    completed_batches: list[dict[str, Any]] = []
    prior_ik_solve_calls = 0
    stored_report: dict[str, Any] | None = None
    if resume_progress is not None:
        if not isinstance(resume_progress, Mapping):
            raise TypeError("Validator resume progress must be a mapping.")
        created_value = resume_progress.get("created_utc")
        if not isinstance(created_value, str) or not created_value:
            raise ValueError("Validator resume progress is missing created_utc.")
        created_utc = created_value
        resumed_goal = resume_progress.get("goal_replay")
        if resumed_goal is not None and not isinstance(resumed_goal, Mapping):
            raise TypeError("Validator checkpoint goal_replay must be a mapping or None.")
        goal_result = None if resumed_goal is None else dict(resumed_goal)
        resumed_rows = resume_progress.get("rows")
        resumed_batches = resume_progress.get("completed_batches")
        if not isinstance(resumed_rows, list) or not isinstance(resumed_batches, list):
            raise TypeError("Validator checkpoint rows and completed_batches must be lists.")
        if any(not isinstance(row, dict) for row in resumed_rows) or any(
            not isinstance(batch, dict) for batch in resumed_batches
        ):
            raise TypeError("Validator checkpoint row and batch entries must be mappings.")
        rows = [dict(row) for row in resumed_rows]
        completed_batches = [dict(batch) for batch in resumed_batches]
        prior_ik_solve_calls = resume_progress.get("ik_solve_call_count", 0)
        if (
            isinstance(prior_ik_solve_calls, bool)
            or not isinstance(prior_ik_solve_calls, int)
            or prior_ik_solve_calls < 0
        ):
            raise ValueError("Validator checkpoint IK solve-call evidence must be a non-negative integer.")
        resumed_report = resume_progress.get("report")
        if resumed_report is not None and not isinstance(resumed_report, Mapping):
            raise TypeError("Validator checkpoint report must be a mapping or None.")
        stored_report = None if resumed_report is None else dict(resumed_report)

    def emit_progress(status: str, *, report: dict[str, Any] | None = None, solve_calls: int) -> None:
        if checkpoint_callback is None:
            return
        checkpoint_callback(
            {
                "status": status,
                "created_utc": created_utc,
                "goal_replay": goal_result,
                "completed_batches": completed_batches,
                "rows": rows,
                "counters": _validation_progress_counters(rows, completed_batches),
                "ik_solve_call_count": solve_calls,
                "torch_rng_state": _torch_rng_state_json(),
                "report": report,
            }
        )

    if stored_report is not None:
        if stored_report.get("content_sha256") != reset_validation_report_content_digest(stored_report):
            raise ValueError("Checkpointed validation report content digest is invalid.")
        expected_report_identity = {
            "artifact_content_sha256": payload["content_sha256"],
            "task_contract": dict(metadata["task_contract"]),
            "validation_cfg": asdict(cfg),
            "validation_policy": FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY,
            "physical_contract": physical_contract,
            "physics_versions": package_versions(),
            "source_sha256": validation_source,
            "asset_closure": validation_assets,
        }
        for name, expected in expected_report_identity.items():
            observed = json.loads(json.dumps(stored_report.get(name), allow_nan=False))
            normalized_expected = json.loads(json.dumps(expected, allow_nan=False))
            if reset_dataset_digest(observed) != reset_dataset_digest(normalized_expected):
                raise ValueError(f"Checkpointed validation report {name} does not match the live invocation.")
        return stored_report

    if goal_result is None:
        emit_progress("goal-pending", solve_calls=prior_ik_solve_calls)
        print(
            f"[PICK-INSERT VALIDATION] goal replay start duration_s={cfg.goal_replay_s:.1f} worlds={env.num_envs}",
            flush=True,
        )
        goal_result = _goal_replay(env, goal, cfg)
        print(
            f"[PICK-INSERT VALIDATION] goal replay complete passed={goal_result['passed']} "
            f"simulation_steps={goal_result['simulation_steps']}",
            flush=True,
        )
        emit_progress("rows", solve_calls=prior_ik_solve_calls)

    # One sampler-free owner is sufficient because every solve receives the
    # physical continuation state explicitly; solve-call count is evidence only.
    ik = _new_validator_ik(env, cfg, prior_solve_calls=prior_ik_solve_calls)
    layout = env.rj45_runtime.layout
    plug_index = int(layout.plug_body_index)
    latch_index = int(layout.latch_body_index)
    batch_plan = _validation_batch_plan(states["phase"], selected, batch_size=env.num_envs)
    _validate_completed_batch_prefix(completed_batches, rows, batch_plan)
    global_batch_index = 0

    for phase in PICK_INSERT_RESET_PHASE_IDS:
        phase_rows = selected[states["phase"][selected] == phase]
        phase_batch_count = math.ceil(len(phase_rows) / env.num_envs)
        print(
            f"[PICK-INSERT VALIDATION] phase={phase}:{PICK_INSERT_PHASE_NAMES[phase]} start "
            f"selected_rows={len(phase_rows)} batches={phase_batch_count}",
            flush=True,
        )
        for begin in range(0, len(phase_rows), env.num_envs):
            selected_batch = phase_rows[begin : begin + env.num_envs]
            real_count = len(selected_batch)
            if not real_count:
                continue
            batch_index = begin // env.num_envs + 1
            batch_record = {
                "ordinal": global_batch_index,
                "phase": phase,
                "phase_batch_index": batch_index,
                "row_ids": [int(row_id) for row_id in selected_batch.tolist()],
            }
            if global_batch_index < len(completed_batches):
                if reset_dataset_digest(completed_batches[global_batch_index]) != reset_dataset_digest(batch_record):
                    raise ValueError("Validator checkpoint completed-batch prefix does not match this invocation.")
                print(
                    f"[PICK-INSERT VALIDATION] phase={phase}:{PICK_INSERT_PHASE_NAMES[phase]} "
                    f"batch={batch_index}/{phase_batch_count} resume-skip rows={real_count}",
                    flush=True,
                )
                global_batch_index += 1
                continue
            print(
                f"[PICK-INSERT VALIDATION] phase={phase}:{PICK_INSERT_PHASE_NAMES[phase]} "
                f"batch={batch_index}/{phase_batch_count} start rows={real_count} "
                f"completed={len(rows)}/{len(selected)}",
                flush=True,
            )
            repetitions = math.ceil(env.num_envs / real_count)
            simulation_rows = selected_batch.repeat(repetitions)[: env.num_envs]
            state = {
                name: states[name][simulation_rows].to(env.device)
                for name in (
                    "arm_joint_position",
                    "arm_joint_target",
                    "arm_joint_velocity",
                    "finger_joint_position",
                    "finger_joint_velocity",
                    "finger_joint_target",
                    "task_body_pose",
                    "task_body_previous_pose",
                    "task_body_coupling_previous_pose",
                    "task_body_velocity",
                )
            }
            goal_q = states["goal_task_body_pose"][simulation_rows].to(env.device)
            goal_arm_target = states["goal_arm_joint_target"][simulation_rows].to(env.device)
            history_evidence = _write_state(env, state)
            reset_contact_count = _contact_count()
            replay_start_q, replay_start_qd = env.read_task_state()
            initial_tcp_xyz = env.tcp_pose_e()[:, :3].clone()
            immediate_finite = task_state_is_finite_and_normalized(replay_start_q, replay_start_qd)
            immediate_joints = joint_limit_mask(env, state["arm_joint_position"])
            immediate_targets = joint_limit_mask(env, state["arm_joint_target"])
            semantic, replay_error, tcp_distance = _phase_semantics(
                env,
                phase,
                replay_start_q,
                goal_q,
            )

            reset_clamp_evidence: dict[str, torch.Tensor] = {}
            reset_replay_evidence: dict[str, Any] = {}
            reset_steps = advance_reset_absolute_target_hold(
                env,
                cfg.row_settle_s,
                state["arm_joint_target"],
                state["finger_joint_target"],
                clamp_evidence=reset_clamp_evidence,
                replay_evidence=reset_replay_evidence,
                starts_grasped=_PHASE_STARTS_GRASPED[phase],
            )
            settled_q, settled_qd = env.read_task_state()
            settled_arm_q, _, _, _ = env.read_robot_state()
            settled_target, settled_target_clamp_delta = runtime_persistent_arm_target(
                env,
                state["arm_joint_target"],
            )
            maximum_reset_arm_target_clamp_delta = torch.maximum(
                reset_clamp_evidence["maximum_arm_target_clamp_delta"],
                settled_target_clamp_delta,
            )
            reset_zero_action_unclamped = ~(
                reset_clamp_evidence["any_arm_target_clamped"]
                | (settled_target_clamp_delta > PICK_INSERT_RESET_MAX_ARM_TARGET_CLAMP_DELTA_RAD)
            )
            reset_absolute_target_stable = (
                reset_clamp_evidence["maximum_arm_target_drift"] <= PICK_INSERT_RESET_MAX_ARM_TARGET_CLAMP_DELTA_RAD
            )
            reset_target_tracking_bounded = (
                reset_replay_evidence["stored_arm_target_tracking_bounded"]
                & reset_replay_evidence["all_post_step_arm_target_tracking_bounded"]
            )
            if _PHASE_STARTS_GRASPED[phase]:
                reset_contact_state_valid = (
                    reset_replay_evidence["all_post_step_bilateral_grasp"]
                    & reset_replay_evidence["all_post_step_proxy_bilateral_contact"]
                    & (reset_replay_evidence["minimum_left_proxy_contact_count"] >= 1)
                    & (reset_replay_evidence["minimum_right_proxy_contact_count"] >= 1)
                )
            else:
                reset_contact_state_valid = (
                    reset_replay_evidence["all_post_step_zero_proxy_contacts"]
                    & (reset_replay_evidence["maximum_left_proxy_contact_count"] == 0)
                    & (reset_replay_evidence["maximum_right_proxy_contact_count"] == 0)
                )
            reset_sample_count_exact = (
                reset_steps == PICK_INSERT_RESET_REPLAY_POST_STEP_SAMPLES
                and reset_replay_evidence["post_step_samples"] == PICK_INSERT_RESET_REPLAY_POST_STEP_SAMPLES
            )
            reset_post_step_finite = (
                reset_replay_evidence["all_post_step_state_finite"]
                & reset_replay_evidence["all_post_step_task_state_finite_and_normalized"]
            )
            reset_collision_valid = (
                reset_replay_evidence["all_post_step_collision_free"]
                & (reset_replay_evidence["maximum_invalid_contact_count"] == 0)
                & (not reset_replay_evidence["any_contact_overflow"])
            )
            if reset_replay_evidence["any_contact_overflow"]:
                raise RuntimeError("Global contact-buffer overflow during validator reset replay.")
            if not bool(reset_replay_evidence["all_post_step_drive_disabled"].all()):
                raise RuntimeError("A construction drive became enabled during validator reset replay.")
            reset_robot_speed_bounded = (
                (reset_replay_evidence["stored_maximum_arm_joint_speed_rad_s"] <= cfg.maximum_row_arm_joint_speed_rad_s)
                & (
                    reset_replay_evidence["maximum_post_step_arm_joint_speed_rad_s"]
                    <= cfg.maximum_row_arm_joint_speed_rad_s
                )
                & (reset_replay_evidence["final_arm_joint_speed_rad_s"] <= cfg.maximum_row_arm_joint_speed_rad_s)
                & (
                    reset_replay_evidence["stored_maximum_finger_joint_speed_m_s"]
                    <= cfg.maximum_row_finger_joint_speed_m_s
                )
                & (
                    reset_replay_evidence["maximum_post_step_finger_joint_speed_m_s"]
                    <= cfg.maximum_row_finger_joint_speed_m_s
                )
                & (reset_replay_evidence["final_finger_joint_speed_m_s"] <= cfg.maximum_row_finger_joint_speed_m_s)
            )
            reset_cable_speed_bounded = (
                (reset_replay_evidence["stored_maximum_cable_speed_m_s"] <= cfg.maximum_row_cable_speed_m_s)
                & (reset_replay_evidence["maximum_post_step_cable_speed_m_s"] <= cfg.maximum_row_cable_speed_m_s)
                & (reset_replay_evidence["final_cable_speed_m_s"] <= cfg.maximum_row_cable_speed_m_s)
            )
            reset_body_excursion_bounded = (
                (reset_replay_evidence["maximum_body_excursion_m"] <= cfg.maximum_row_body_drift_m)
                & (reset_replay_evidence["maximum_plug_excursion_m"] <= cfg.maximum_row_plug_drift_m)
                & (reset_replay_evidence["maximum_socket_excursion_m"] <= cfg.maximum_row_socket_drift_m)
            )
            reset_vbd_pose_history_queued = (
                torch.as_tensor(history_evidence["restore_queued"], device=env.device)
                & torch.as_tensor(history_evidence["pending_at_queue"], device=env.device)
                & torch.as_tensor(history_evidence["previous_pose_queued"], device=env.device)
                & torch.as_tensor(history_evidence["coupling_previous_pose_queued"], device=env.device)
            )
            reset_vbd_pose_history_applied = _vbd_pose_history_applied_mask(env, history_evidence)
            reset_vbd_pose_history_body_order_exact = history_evidence["body_order_exact"] is True
            reset_stable = (
                immediate_finite
                & task_state_is_finite_and_normalized(settled_q, settled_qd)
                & reset_replay_evidence["stored_state_finite"]
                & reset_replay_evidence["stored_task_state_finite_and_normalized"]
                & reset_replay_evidence["stored_drive_disabled"]
                & reset_post_step_finite
                & reset_collision_valid
                & reset_replay_evidence["all_post_step_drive_disabled"]
                & reset_replay_evidence["all_post_step_expected_contact_state"]
                & reset_contact_state_valid
                & immediate_joints
                & immediate_targets
                & joint_limit_mask(env, settled_arm_q)
                & joint_limit_mask(env, settled_target)
                & reset_zero_action_unclamped
                & reset_absolute_target_stable
                & reset_target_tracking_bounded
                & reset_robot_speed_bounded
                & reset_cable_speed_bounded
                & reset_body_excursion_bounded
                & reset_sample_count_exact
                & reset_vbd_pose_history_applied
                & reset_vbd_pose_history_body_order_exact
            )
            socket_stable = reset_replay_evidence["maximum_socket_excursion_m"] <= cfg.maximum_row_socket_drift_m
            drive_disabled_before_oracle = _drive_disabled(env)
            real_lane_mask = torch.arange(env.num_envs, device=env.device) < real_count
            oracle_prerequisites = real_lane_mask & reset_stable

            if _PHASE_STARTS_GRASPED[phase]:
                grasp = grasp_metrics(env, state["finger_joint_target"], retaining_grasp=True)
                acquired = (
                    oracle_prerequisites
                    & grasp.valid
                    & reset_collision_valid
                    & reset_replay_evidence["all_post_step_expected_contact_state"]
                    & reset_contact_state_valid
                )
                oracle_arm_target = settled_target
                acquisition_last_finger_target = state["finger_joint_target"]
                acquisition_evidence: dict[str, Any] = {
                    "started_with_physical_bilateral_grasp": acquired.detach().cpu().tolist(),
                    "last_arm_target": oracle_arm_target.detach().cpu().tolist(),
                    "last_finger_target": acquisition_last_finger_target.detach().cpu().tolist(),
                }
            else:
                oracle_arm_target, acquired, acquisition_evidence = _acquire_physical_grasp(
                    env,
                    ik,
                    settled_target,
                    cfg,
                    active_mask=oracle_prerequisites,
                )
                acquisition_last_finger_target = acquisition_evidence["last_finger_target"]
            closed_target = torch.full((env.num_envs, 2), cfg.finger_closed_target, device=env.device)
            with _PerLaneTargetHold(
                env,
                acquired,
                oracle_arm_target,
                acquisition_last_finger_target,
            ) as recovery_hold:
                recovery, recovery_metrics = scripted_recovery(
                    env,
                    ik,
                    goal_q,
                    None,
                    closed_target,
                    arm_target_start=oracle_arm_target,
                    goal_arm_target=goal_arm_target,
                    motion_s=cfg.recovery_motion_s,
                    settle_s=cfg.recovery_settle_s,
                    compensation_max_iterations=cfg.recovery_compensation_iterations,
                    compensation_tolerance_m=cfg.recovery_compensation_tolerance_m,
                    maximum_ik_joint_step_rad=cfg.maximum_ik_joint_step_rad,
                    plug_body_index=plug_index,
                    latch_body_index=latch_index,
                    arm_target_is_absolute=True,
                    lane_hold=recovery_hold,
                    motion_policy=FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY["scripted_recovery"]["motion_policy"],
                    pick_insert_phase=phase,
                )
                recovery_hold.deactivate(~recovery, reason="validator-recovery-final-validation")
                recovery &= recovery_hold.active_mask
                recovery_metrics["lane_failure_masks"] = recovery_hold.reason_masks
                recovery_metrics["last_arm_target"] = recovery_hold.last_sent_arm_target
                recovery_metrics["last_finger_target"] = recovery_hold.last_sent_finger_target
            recovery_collision = collision_metrics(env)
            if recovery_collision.contact_overflow:
                raise RuntimeError("Global contact-buffer overflow at the validator recovery boundary.")
            exact_dwell = recovery_metrics["exact_success_dwell_satisfied"]
            exact_collision = recovery_metrics["exact_success_all_samples_collision_free"]
            exact_grasp = recovery_metrics["exact_success_all_samples_bilateral_grasp"]
            exact_finite = recovery_metrics["exact_success_all_samples_finite"]
            drive_disabled_after_oracle = _drive_disabled(env)
            if not drive_disabled_before_oracle or not drive_disabled_after_oracle:
                raise RuntimeError("A construction drive became enabled during validator oracle recovery.")

            for local in range(real_count):
                row_id = int(selected_batch[local])
                artifact_error = float(states["initial_goal_error"][row_id])
                replayed_error = float(replay_error[local])
                artifact_tcp_distance = float(states["initial_tcp_grasp_distance"][row_id])
                replayed_tcp_distance = float(tcp_distance[local])
                phase_label_valid = (
                    int(states["phase"][row_id]) == phase
                    and bool(states["starts_grasped"][row_id]) == _PHASE_STARTS_GRASPED[phase]
                )
                row_reset_replay = _reset_replay_row_evidence(
                    reset_replay_evidence,
                    history_evidence=history_evidence,
                    local=local,
                    starts_grasped=_PHASE_STARTS_GRASPED[phase],
                    maximum_arm_target_clamp_delta=maximum_reset_arm_target_clamp_delta,
                    zero_action_unclamped=reset_zero_action_unclamped,
                    cfg=cfg,
                )
                checks = {
                    "reset_stable": bool(reset_stable[local]) and reset_contact_count == 0,
                    "reset_zero_action_unclamped": bool(reset_zero_action_unclamped[local]),
                    "reset_absolute_target_stable": bool(reset_absolute_target_stable[local]),
                    "reset_target_tracking_bounded": bool(reset_target_tracking_bounded[local]),
                    "reset_replay_sample_count_exact": reset_sample_count_exact,
                    "reset_all_post_step_finite": bool(reset_post_step_finite[local]),
                    "reset_all_post_step_collision_valid": bool(reset_collision_valid[local]),
                    "reset_all_post_step_contact_state_valid": bool(
                        reset_replay_evidence["all_post_step_expected_contact_state"][local]
                        & reset_contact_state_valid[local]
                    ),
                    "reset_all_post_step_drives_disabled": bool(
                        reset_replay_evidence["all_post_step_drive_disabled"][local]
                    ),
                    "reset_vbd_pose_history_queued": bool(reset_vbd_pose_history_queued[local]),
                    "reset_vbd_pose_history_applied_exactly_once": bool(reset_vbd_pose_history_applied[local]),
                    "reset_vbd_pose_history_body_order_exact": reset_vbd_pose_history_body_order_exact,
                    "reset_robot_speed_bounded": bool(reset_robot_speed_bounded[local]),
                    "reset_cable_speed_bounded": bool(reset_cable_speed_bounded[local]),
                    "reset_body_excursion_bounded": bool(reset_body_excursion_bounded[local]),
                    "phase_semantics": bool(semantic[local]) and phase_label_valid,
                    "socket_stable": bool(socket_stable[local]),
                    "drive_disabled": drive_disabled_before_oracle and drive_disabled_after_oracle,
                    "collision_valid": bool(reset_collision_valid[local] & recovery_collision.valid[local]),
                    "expected_initial_grasp_state": bool(
                        reset_replay_evidence["all_post_step_expected_contact_state"][local]
                        & reset_contact_state_valid[local]
                    ),
                    "oracle_grasp_acquired": bool(acquired[local]),
                    "oracle_full_recovery": bool(recovery[local]),
                    "exact_insertion_success_dwell": bool(exact_dwell[local]),
                    "oracle_all_samples_collision_free": bool(exact_collision[local]),
                    "oracle_all_samples_bilateral_grasp": bool(exact_grasp[local]),
                    "oracle_all_samples_finite": bool(exact_finite[local]),
                }
                oracle = {
                    "drive_disabled": drive_disabled_after_oracle,
                    "physical_grasp_acquired": bool(acquired[local]),
                    "exact_success_dwell": bool(exact_dwell[local]),
                    "no_invalid_contacts": bool(recovery_collision.valid[local]),
                    "all_samples_collision_free": bool(exact_collision[local]),
                    "all_samples_bilateral_grasp": bool(exact_grasp[local]),
                    "all_samples_finite": bool(exact_finite[local]),
                }
                metrics = {
                    "initial_goal_error_artifact": artifact_error,
                    "initial_goal_error_replayed": replayed_error,
                    "initial_goal_error_matches": abs(artifact_error - replayed_error) <= 5.0e-4,
                    "initial_tcp_distance_artifact_m": artifact_tcp_distance,
                    "initial_tcp_distance_replayed_m": replayed_tcp_distance,
                    "initial_tcp_distance_matches": abs(artifact_tcp_distance - replayed_tcp_distance) <= 2.0e-3,
                    "initial_tcp_xyz_replayed_m": initial_tcp_xyz[local].detach().cpu().tolist(),
                    "settle_socket_drift_m": float(reset_replay_evidence["maximum_socket_excursion_m"][local]),
                    "settle_plug_drift_m": float(reset_replay_evidence["maximum_plug_excursion_m"][local]),
                    "settle_max_body_drift_m": float(reset_replay_evidence["maximum_body_excursion_m"][local]),
                    "capture_max_cable_speed_m_s": float(
                        reset_replay_evidence["stored_maximum_cable_speed_m_s"][local]
                    ),
                    "settled_max_cable_speed_m_s": float(reset_replay_evidence["final_cable_speed_m_s"][local]),
                    "maximum_reset_arm_target_clamp_delta_rad": float(maximum_reset_arm_target_clamp_delta[local]),
                    "maximum_reset_arm_target_drift_rad": float(
                        reset_clamp_evidence["maximum_arm_target_drift"][local]
                    ),
                    "reset_maximum_invalid_contacts": int(
                        reset_replay_evidence["maximum_invalid_contact_count"][local]
                    ),
                    "recovery_invalid_contacts": int(recovery_collision.invalid_contact_count[local]),
                    "recovery_goal_error": float(recovery_metrics["goal_error"][local]),
                    "recovery_plug_speed": float(recovery_metrics["plug_speed"][local]),
                    "recovery_maximum_body_excursion_m": float(
                        recovery_metrics["exact_success_maximum_body_excursion"][local]
                    ),
                    "recovery_maximum_cable_linear_speed_m_s": float(
                        recovery_metrics["exact_success_maximum_cable_linear_speed"][local]
                    ),
                    "recovery_maximum_arm_joint_speed_rad_s": float(
                        recovery_metrics["exact_success_maximum_arm_joint_speed"][local]
                    ),
                    "recovery_maximum_finger_joint_speed_m_s": float(
                        recovery_metrics["exact_success_maximum_finger_joint_speed"][local]
                    ),
                    "acquisition": acquisition_evidence,
                }
                checks["reset_stable"] &= bool(metrics["initial_goal_error_matches"])
                checks["phase_semantics"] &= bool(metrics["initial_tcp_distance_matches"])
                rows.append(
                    _row_report(
                        row_id=row_id,
                        phase=phase,
                        checks=checks,
                        oracle=oracle,
                        reset_replay=row_reset_replay,
                        metrics=metrics,
                    )
                )
            completed_batch = rows[-real_count:]
            completed_batches.append(batch_record)
            global_batch_index += 1
            emit_progress("rows", solve_calls=ik.solve_calls)
            print(
                f"[PICK-INSERT VALIDATION] phase={phase}:{PICK_INSERT_PHASE_NAMES[phase]} "
                f"batch={batch_index}/{phase_batch_count} complete "
                f"passed={sum(bool(row['passed']) for row in completed_batch)}/{real_count} "
                f"completed={len(rows)}/{len(selected)}",
                flush=True,
            )
        phase_report_rows = [row for row in rows if row["phase"] == phase]
        print(
            f"[PICK-INSERT VALIDATION] phase={phase}:{PICK_INSERT_PHASE_NAMES[phase]} complete "
            f"passed={sum(bool(row['passed']) for row in phase_report_rows)}/{len(phase_report_rows)}",
            flush=True,
        )

    if global_batch_index != len(completed_batches):
        raise ValueError("Validator checkpoint contains completed batches beyond the deterministic replay plan.")
    rows.sort(key=lambda row: row["row_id"])
    failed_rows = [int(row["row_id"]) for row in rows if not row["passed"]]
    all_rows_selected = len(selected) == len(states["phase"])
    selected_phase_counts = [int((states["phase"][selected] == phase).sum()) for phase in PICK_INSERT_RESET_PHASE_IDS]
    task_contract = dict(metadata["task_contract"])
    required_rows_per_phase = int(task_contract["pick_insert"]["reset_dataset_rows_per_phase"])
    if cfg.quick:
        offline_diversity: dict[str, Any] = {"skipped_due_to_quick": True, "passed": None}
    else:
        try:
            offline_diversity = reset_dataset_validate_full_pick_diversity(
                states,
                task_contract=task_contract,
            )
        except (TypeError, ValueError) as exc:
            offline_diversity = {
                "skipped_due_to_quick": False,
                "passed": False,
                "error": str(exc),
            }
    full_pick_diversity = _full_pick_live_tcp_diversity(
        rows,
        quick=cfg.quick,
        required_row_count=required_rows_per_phase,
        offline_artifact_evidence=offline_diversity,
    )
    diversity_required_pass = cfg.quick or full_pick_diversity["passed"] is True
    if reset_dataset_digest(validation_source) != reset_dataset_digest(_validation_source_digests()):
        raise RuntimeError("Validator source changed during physical replay; refusing to certify mixed evidence.")
    report = {
        "format": FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_FORMAT,
        "schema_version": FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_SCHEMA_VERSION,
        "created_utc": created_utc,
        "artifact_content_sha256": payload["content_sha256"],
        "task_contract": task_contract,
        "validation_cfg": asdict(cfg),
        "validation_policy": FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY,
        "physical_contract": physical_contract,
        "physics_versions": package_versions(),
        "source_sha256": validation_source,
        "asset_closure": validation_assets,
        "ik_solve_call_count": ik.solve_calls,
        "quick": cfg.quick,
        "full_dataset_replay": all_rows_selected,
        "evidence_complete": all_rows_selected and not cfg.quick and diversity_required_pass,
        "selected_row_count": len(selected),
        "dataset_row_count": len(states["phase"]),
        "selected_row_ids": sorted(selected.tolist()),
        "phase_counts": selected_phase_counts,
        "goal_replay": goal_result,
        "full_pick_diversity": full_pick_diversity,
        "rows": rows,
        "failed_row_ids": failed_rows,
        "passed": bool(goal_result["passed"]) and not failed_rows and diversity_required_pass,
    }
    report["content_sha256"] = reset_validation_report_content_digest(report)
    emit_progress("report-ready", report=report, solve_calls=ik.solve_calls)
    return report


def _write_json_atomic(payload: dict[str, Any], output: Path) -> Path:
    output = _resolve_output_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        directory_descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _write_validation_checkpoint(
    metadata: Mapping[str, Any],
    progress: Mapping[str, Any],
    output: Path,
) -> Path:
    """Atomically replace one complete goal/batch/report lifecycle checkpoint."""
    return _write_json_atomic(_checkpoint_document(metadata, progress), output)


def write_validation_report(
    report: dict[str, Any],
    output_dir: Path = DEFAULT_VALIDATION_DIR,
    *,
    output_path: Path | None = None,
    protected_paths: tuple[Path, ...] = (),
) -> Path:
    """Write one timestamped report, including honest quick or failed evidence."""
    output = (
        _timestamped_validation_report_path(
            output_dir,
            artifact_content_sha256=str(report.get("artifact_content_sha256", "unknown")),
            quick=report.get("quick") is True,
            protected_paths=protected_paths,
        )
        if output_path is None
        else _reject_protected_output_alias(
            output_path,
            protected_paths,
            label="Timestamped validation report",
        )
    )
    return _write_json_atomic(report, output)


def write_stable_validation_report(
    report: dict[str, Any],
    payload: dict[str, Any],
    env_cfg: FrankaRJ45PickInsertEnvCfg,
    output: Path,
    *,
    expected_validation_policy: Mapping[str, Any],
    expected_source_sha256: Mapping[str, Any],
    expected_asset_closure: Mapping[str, Any],
    protected_paths: tuple[Path, ...] = (),
) -> Path:
    """Publish the training gate only after strict full-report validation."""
    reset_dataset_validate_phase_row_counts(
        payload["states"]["phase"],
        expected_rows_per_phase=env_cfg.reset_dataset_rows_per_phase,
    )
    offline_diversity = reset_dataset_validate_full_pick_diversity(
        payload["states"],
        task_contract=payload["metadata"]["task_contract"],
    )
    reset_validation_report_validate_runtime(
        report,
        expected_content_sha256=payload["content_sha256"],
        expected_row_count=len(payload["states"]["phase"]),
        expected_phases=payload["states"]["phase"],
        expected_task_contract=pick_insert_reset_dataset_task_contract(env_cfg),
        expected_validation_policy=expected_validation_policy,
        expected_source_sha256=expected_source_sha256,
        expected_asset_closure=expected_asset_closure,
        expected_full_pick_diversity=offline_diversity,
    )
    resolved_output = _reject_protected_output_alias(
        output,
        protected_paths,
        label="Stable validation report",
    )
    return _write_json_atomic(report, resolved_output)


def _republish_stable_published_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    keep_checkpoint: bool,
    payload: dict[str, Any],
    env_cfg: FrankaRJ45PickInsertEnvCfg,
    stable_output: Path,
    expected_source_sha256: Mapping[str, Any],
    expected_asset_closure: Mapping[str, Any],
    protected_paths: tuple[Path, ...],
) -> Path:
    """Strictly republish one completed report and finish checkpoint cleanup."""
    progress = checkpoint.get("progress")
    metadata = checkpoint.get("metadata")
    if not isinstance(progress, Mapping) or progress.get("status") != "stable-published":
        raise ValueError("Stable checkpoint republication requires stable-published progress.")
    if not isinstance(metadata, Mapping):
        raise TypeError("Stable checkpoint republication requires checkpoint metadata.")
    completed_task_contract = pick_insert_reset_dataset_task_contract(env_cfg)
    completed_task_normalized = json.loads(json.dumps(completed_task_contract, allow_nan=False))
    checkpoint_task_normalized = json.loads(json.dumps(metadata.get("task_contract"), allow_nan=False))
    if reset_dataset_digest(completed_task_normalized) != reset_dataset_digest(checkpoint_task_normalized):
        raise ValueError("Stable-published validator checkpoint task contract does not match this invocation.")
    report = progress.get("report")
    if not isinstance(report, dict):
        raise TypeError("Stable-published validator checkpoint requires a final report.")
    stable_path = write_stable_validation_report(
        report,
        payload,
        env_cfg,
        stable_output,
        expected_validation_policy=FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY,
        expected_source_sha256=expected_source_sha256,
        expected_asset_closure=expected_asset_closure,
        protected_paths=protected_paths,
    )
    if not keep_checkpoint:
        checkpoint_path.unlink()
    return stable_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--stable-output", type=Path, default=DEFAULT_STABLE_REPORT_PATH)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--quick", action="store_true", help="Replay one deterministic row from every phase.")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--checkpoint", type=Path, help="Atomically checkpoint each completed full batch.")
    checkpoint_group.add_argument("--resume", type=Path, help="Resume a full replay from an exact checkpoint.")
    parser.add_argument(
        "--keep-checkpoint",
        action="store_true",
        help="Retain a stable-published checkpoint instead of deleting it after success.",
    )
    add_launcher_args(parser)
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Pick-insert reset artifact does not exist: {input_path}")
    output_dir, stable_output = _resolve_validation_output_paths(
        input_path=input_path,
        output_dir=args.output_dir,
        stable_output=args.stable_output,
    )
    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    _, states_raw, _ = reset_dataset_validate_runtime(payload)
    states = dict(states_raw)
    validation_cfg = ValidationCfg(seed=args.seed, quick=args.quick, sample_count=args.sample_count)
    checkpoint_path, resuming = _validate_checkpoint_invocation(
        checkpoint=args.checkpoint,
        resume=args.resume,
        keep_checkpoint=args.keep_checkpoint,
        input_path=input_path,
        stable_output=stable_output,
        cfg=validation_cfg,
    )
    checkpoint_protected = () if checkpoint_path is None else (checkpoint_path,)
    report_protected = (input_path, stable_output, *checkpoint_protected)
    timestamped_report_output = _timestamped_validation_report_path(
        output_dir,
        artifact_content_sha256=payload["content_sha256"],
        quick=validation_cfg.quick,
        protected_paths=report_protected,
    )
    env_cfg = FrankaRJ45PickInsertEnvCfg()
    _validate_invocation_phase_counts(
        states["phase"],
        validation_cfg,
        expected_rows_per_phase=env_cfg.reset_dataset_rows_per_phase,
    )
    selected_count = len(_selected_rows(states, validation_cfg))
    batch_size = min(max(1, args.batch_size), selected_count)
    source_sha256 = _validation_source_digests()
    asset_closure = _validated_asset_closure_snapshot()
    loaded_checkpoint = (
        _load_validation_checkpoint(checkpoint_path) if resuming and checkpoint_path is not None else None
    )
    if loaded_checkpoint is not None:
        checkpoint_metadata = loaded_checkpoint["metadata"]
        preflight_expected = {
            "artifact_content_sha256": payload["content_sha256"],
            "artifact_metadata": payload["metadata"],
            "dataset_row_count": len(states["phase"]),
            "dataset_phases": [int(value) for value in states["phase"].tolist()],
            "selected_row_ids": list(range(len(states["phase"]))),
            "batch_size": batch_size,
            "validation_cfg": asdict(validation_cfg),
            "validation_policy": FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY,
            "source_sha256": source_sha256,
            "asset_closure": asset_closure,
        }
        for name, expected in preflight_expected.items():
            observed_normalized = json.loads(json.dumps(checkpoint_metadata.get(name), allow_nan=False))
            expected_normalized = json.loads(json.dumps(expected, allow_nan=False))
            if reset_dataset_digest(observed_normalized) != reset_dataset_digest(expected_normalized):
                raise ValueError(f"Validator checkpoint {name} does not match this invocation.")
    env_cfg.scene.num_envs = batch_size
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    env_cfg.validate_config()
    protected_outputs = (input_path, *checkpoint_protected)
    if loaded_checkpoint is not None and loaded_checkpoint["progress"]["status"] == "stable-published":
        assert checkpoint_path is not None
        selected = _selected_rows(states, validation_cfg)
        batch_plan = _validation_batch_plan(states["phase"], selected, batch_size=batch_size)
        checkpoint_progress = loaded_checkpoint["progress"]
        _validate_completed_batch_prefix(
            checkpoint_progress["completed_batches"],
            checkpoint_progress["rows"],
            batch_plan,
        )
        if len(checkpoint_progress["completed_batches"]) != len(batch_plan):
            raise ValueError("Stable-published validator checkpoint does not contain every completed batch.")
        stable_path = _republish_stable_published_checkpoint(
            loaded_checkpoint,
            checkpoint_path=checkpoint_path,
            keep_checkpoint=args.keep_checkpoint,
            payload=payload,
            env_cfg=env_cfg,
            stable_output=stable_output,
            expected_source_sha256=source_sha256,
            expected_asset_closure=asset_closure,
            protected_paths=protected_outputs,
        )
        print(f"[INFO] Canonical validation gate already complete: {stable_path}")
        if not args.keep_checkpoint:
            print(f"[INFO] Removed completed validator checkpoint: {checkpoint_path}")
        return
    report: dict[str, Any] | None = None
    latest_checkpoint_progress: dict[str, Any] | None = None
    with launch_simulation(env_cfg, args):
        try:
            env = RJ45PickInsertResetToolEnv(env_cfg)
        except Exception as exc:
            raise RuntimeError(
                "Could not construct the real coupled pick-insert validator; no validation evidence was emitted."
            ) from exc
        try:
            task_contract = pick_insert_reset_dataset_task_contract(env.cfg)
            physical_contract = pick_insert_tool_physical_contract(
                env,
                finger_closed_target=validation_cfg.finger_closed_target,
            )
            runtime_versions = package_versions()
            checkpoint_metadata = _validation_checkpoint_metadata(
                payload=payload,
                states=states,
                cfg=validation_cfg,
                batch_size=batch_size,
                task_contract=task_contract,
                physical_contract=physical_contract,
                physics_versions=runtime_versions,
                source_sha256=source_sha256,
                asset_closure=asset_closure,
            )
            if loaded_checkpoint is not None:
                _validate_checkpoint_metadata(loaded_checkpoint["metadata"], checkpoint_metadata)
                _restore_torch_rng_state(loaded_checkpoint["progress"]["torch_rng_state"])

            def checkpoint_callback(progress: dict[str, Any]) -> None:
                nonlocal latest_checkpoint_progress
                latest_checkpoint_progress = progress
                if checkpoint_path is not None:
                    _write_validation_checkpoint(checkpoint_metadata, progress, checkpoint_path)

            report = validate_payload(
                env,
                payload,
                validation_cfg,
                resume_progress=None if loaded_checkpoint is None else loaded_checkpoint["progress"],
                checkpoint_callback=checkpoint_callback if checkpoint_path is not None else None,
                source_sha256=source_sha256,
                asset_closure=asset_closure,
            )
        finally:
            env.close()

    assert report is not None
    report_path = write_validation_report(
        report,
        output_dir,
        output_path=timestamped_report_output,
        protected_paths=report_protected,
    )
    print(f"[INFO] Validation report: {report_path}")
    print(
        f"[INFO] Passed={report['passed']} goal={report['goal_replay']['passed']} "
        f"rows={report['selected_row_count'] - len(report['failed_row_ids'])}/{report['selected_row_count']}"
    )
    if report["passed"] and report["evidence_complete"] and report["full_dataset_replay"] and not report["quick"]:
        stable_path = write_stable_validation_report(
            report,
            payload,
            env_cfg,
            stable_output,
            expected_validation_policy=FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY,
            expected_source_sha256=source_sha256,
            expected_asset_closure=asset_closure,
            protected_paths=protected_outputs,
        )
        print(f"[INFO] Canonical validation gate: {stable_path}")
        if checkpoint_path is not None:
            if latest_checkpoint_progress is None:
                latest_checkpoint_progress = (
                    dict(loaded_checkpoint["progress"]) if loaded_checkpoint is not None else None
                )
            if latest_checkpoint_progress is None:
                raise RuntimeError("Stable publication completed without report-ready checkpoint progress.")
            stable_progress = dict(latest_checkpoint_progress)
            stable_progress["status"] = "stable-published"
            stable_progress["report"] = report
            stable_progress["torch_rng_state"] = _torch_rng_state_json()
            _write_validation_checkpoint(checkpoint_metadata, stable_progress, checkpoint_path)
            if not args.keep_checkpoint:
                checkpoint_path.unlink()
                print(f"[INFO] Removed completed validator checkpoint: {checkpoint_path}")
    if not report["passed"]:
        raise RuntimeError(
            f"Pick-insert reset validation failed for rows {report['failed_row_ids']}; see {report_path}."
        )


if __name__ == "__main__":
    main()
