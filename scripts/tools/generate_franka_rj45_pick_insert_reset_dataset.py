# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate six-stage Franka RJ45 pick-and-insert reset states.

The script first derives one central, fully seated connector state with the
Newton reference +35 mm construction drive.  Each training row receives a
rigidly transformed copy of that solved goal for its randomized socket pose.
The default physical-oracle mode produces cable shapes through coupled motion
and proves every accepted row with cold replay and robot recovery.  The
explicit fast-IK mode instead constructs coherent rows directly and applies
only static IK, finite-state, joint-limit, workspace, collision, and phase
semantic gates.

Open phase-4 pregrasp rows retain the 45 mm clearance while sampling a
plug-relative closing-axis twist uniformly over +/-60 degrees and a top-down
tilt uniformly by solid angle over a 0-25 degree cone.  Starts-grasped phases
remain at the canonical grasp orientation, and phase 5 keeps its away pose.

Production runs first certify the canonical goal with one or four worlds.  The
legacy physical row stream remains fixed at 96 rows per phase and batch 24.
The reference fast bank uses 3,334 rows per phase (20,004 total), batch 256,
and a phase-0 reverse curriculum sampled just outside geometric success.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import math
import os
import stat
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
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
    batched_quat_slerp,
    collide_only_metrics,
    collision_metrics,
    configured_arm_home,
    exact_success_from_state,
    grasp_metrics,
    interpolate_arm_motion,
    joint_limit_mask,
    package_versions,
    pick_insert_tool_physical_contract,
    plug_relative_latch_angle,
    runtime_persistent_arm_target,
    save_torch_atomic,
    scalar_goal_error,
    scripted_recovery,
    task_state_is_finite_and_normalized,
)
from isaaclab_newton.physics import NewtonManager
from newton import BodyFlags

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import math as math_utils

from isaaclab_contrib.coupling import NewtonCouplerManager

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.franka_rj45_insertion.asset_provenance import franka_rj45_asset_contract
from isaaclab_tasks.contrib.franka_rj45_insertion.physics.rj45_assembly import (
    CABLE_KINEMATIC_COUNT,
    CABLE_RADIUS,
    GRASP_FRICTION,
    GRASP_PROXY_CENTER,
    GRASP_PROXY_HALF_EXTENTS,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_env_cfg import (
    PICK_INSERT_CLOSED_FINGER_POSITION,
    PICK_INSERT_EFFECTIVE_GRASP_FRICTION,
    PICK_INSERT_GRASP_PROXY_FRICTION,
    PICK_INSERT_OPEN_FINGER_POSITION,
    PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_AXIAL_RANGES_M,
    PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_BAND_NAMES,
    PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_WEIGHTS,
    PICK_INSERT_PHASE_4_PREGRASP_HEIGHT_M,
    PICK_INSERT_PHASE_4_PREGRASP_MAXIMUM_CLOSING_AXIS_TWIST_ERROR_RAD,
    PICK_INSERT_PHASE_4_PREGRASP_MAXIMUM_TOP_DOWN_TILT_ERROR_RAD,
    PICK_INSERT_PHASE_NAMES,
    FrankaRJ45PickInsertEnvCfg,
    pick_insert_phase_0_reverse_curriculum_sampling_contract,
    pick_insert_phase_4_pregrasp_orientation_sampling_contract,
    pick_insert_reset_dataset_task_contract,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_reset_dataset_io import (
    FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY,
    FRANKA_RJ45_PICK_INSERT_RESET_DATASET_FORMAT,
    FRANKA_RJ45_PICK_INSERT_RESET_DATASET_SCHEMA_VERSION,
    PICK_INSERT_FAST_RESET_PHASE_0_BAND_ACCEPTANCE_CONTRACT,
    PICK_INSERT_FAST_RESET_ROW_BINDING_CONTRACT,
    PICK_INSERT_GOAL_MAX_ARM_JOINT_SPEED_RAD_S,
    PICK_INSERT_GOAL_MAX_AUTHORED_PLUG_ANGLE_RAD,
    PICK_INSERT_GOAL_MAX_AUTHORED_SEAT_ERROR_M,
    PICK_INSERT_GOAL_MAX_FINGER_JOINT_SPEED_M_S,
    PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD,
    PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M,
    PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY,
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
    RESET_DATASET_GOAL_STATE_NAMES,
    RESET_DATASET_STATE_NAMES,
    pick_insert_fast_reset_phase_0_band_fraction_tolerance,
    pick_insert_reset_dataset_row_digest,
    reset_dataset_content_digest,
    reset_dataset_validate_runtime,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.reset_dataset_io import reset_dataset_digest
from isaaclab_tasks.contrib.franka_rj45_insertion.rj45_env_cfg import RIGID_ENTRY, RJ45_ENTRY

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = _REPO_ROOT / "datasets/franka_rj45_pick_insert/reset_dataset.pt"
_DEFAULT_STABLE_VALIDATION_REPORT_PATH = (
    _REPO_ROOT / "logs/rsl_rl/franka_rj45_pick_insert/validation/reset_validation.json"
)
_CANONICAL_ROWS_PER_PHASE = 96
_CANONICAL_BATCH_SIZE = 24
_REFERENCE_FAST_ROWS_PER_PHASE = 3_334
_REFERENCE_FAST_BATCH_SIZE = 256
_REFERENCE_FAST_TOTAL_ROWS = _REFERENCE_FAST_ROWS_PER_PHASE * len(PICK_INSERT_RESET_PHASE_IDS)
_GENERATION_MODE_PHYSICAL_ORACLE = "physical-oracle"
_GENERATION_MODE_FAST_IK = "fast-ik"
_GENERATION_MODES = (_GENERATION_MODE_PHYSICAL_ORACLE, _GENERATION_MODE_FAST_IK)
_PHASE_STARTS_GRASPED = (True, True, True, True, False, False)
_PHASE_DIFFICULTY = (0.10, 0.25, 0.45, 0.62, 0.80, 1.00)
_DIAGNOSTIC_RESEAT_ROLLING_MAX_DURATION_S = 4.0
_DIAGNOSTIC_RESEAT_ROLLING_DWELL_S = 2.0
_CANONICAL_GOAL_CERTIFICATE_FORMAT = "isaaclab-franka-rj45-pick-insert-canonical-goal-certificate"
_CANONICAL_GOAL_CERTIFICATE_SCHEMA_VERSION = 1
_GENERATION_CHECKPOINT_FORMAT = "isaaclab-franka-rj45-pick-insert-generation-checkpoint"
_GENERATION_CHECKPOINT_SCHEMA_VERSION = 1
_CANONICAL_GOAL_CERTIFIER_ENV_COUNTS = (1, 4)
_ROW_IK_SAMPLER = "none"
_ROW_IK_SEED_COUNT = 1
_ROW_IK_NOISE_STD = 0.0
_ROW_IK_ITERATIONS = 160
_PICKUP_CONSTRUCTION_SEQUENCE_VERSION = 2
_ORACLE_ENTRY_REPLAY_CONTRACT_VERSION = 1
_ORACLE_ACQUISITION_MOVE_ATTEMPT_COUNT = 5
_ORACLE_ACQUISITION_MOVE_SETTLE_S = 0.30
_GRASPED_TRANSPORT_SCHEDULE_VERSION = 5
_GRASPED_TRANSPORT_C2_RAMP_FRACTION = 0.10
_GRASPED_TRANSPORT_TERMINAL_PROGRESS_EPSILON_M = 1.0e-6
_GRASPED_TRANSPORT_STRICT_ENDPOINT_POLICY = "strict"
_GRASPED_TRANSPORT_RESET_ROW_ENDPOINT_POLICY = "reset-row"
_RECOVERY_CARTESIAN_SPEED_COMPONENTS = (
    "plug_linear_speed",
    "plug_angular_speed",
    "arm_joint_speed",
    "finger_joint_speed",
)
_RECOVERY_DIAGNOSTIC_EVIDENCE_NAMES = (
    "motion_policy",
    "pick_insert_phase",
    "overtravel_distance",
    "used_canonical_goal_arm_target",
    "compensation_iterations",
    "goal_error",
    "goal_error_history",
    "plug_translation_error_history",
    "correction_norm_history",
    "start_preload_bias_by_joint_rad",
    "goal_preload_bias_by_joint_rad",
    "preload_bias_difference_by_joint_rad",
    "maximum_preload_bias_difference_rad",
    "maximum_observed_raw_ik_joint_step_rad",
    "maximum_commanded_joint_step_before_densification_rad",
    "maximum_commanded_joint_step_after_densification_rad",
    "command_densification_required_subknot_count",
    "command_densification_executed_subknot_count",
    "start_target_anchor_error_rad",
    "canonical_endpoint_anchor_error_rad",
    "maximum_segment_boundary_command_jump_rad",
    "cartesian_route_waypoint_count_before_densification",
    "cartesian_route_waypoint_count_after_densification",
    "cartesian_route_waypoint_count",
    "cartesian_motion_sample_count",
    "cartesian_motion_maximum_cable_linear_speed",
    "cartesian_motion_maximum_plug_linear_speed",
    "cartesian_motion_maximum_plug_angular_speed",
    "cartesian_motion_maximum_arm_joint_speed",
    "cartesian_motion_maximum_finger_joint_speed",
    *(
        f"cartesian_motion_first_{component}_failure_{field}"
        for component in _RECOVERY_CARTESIAN_SPEED_COMPONENTS
        for field in ("mask", "step", "segment", "knot", "time_s")
    ),
    "lane_failure_masks",
)


def _contact_count() -> int:
    """Count outer and proxy-local contacts after a history flush."""
    proxy_contacts, _, _ = NewtonCouplerManager.get_proxy_contact_data(RIGID_ENTRY, RJ45_ENTRY)
    count = 0
    for contacts in (NewtonManager.get_contacts(), proxy_contacts):
        if contacts is not None and contacts.rigid_contact_count is not None:
            count += int(wp.to_torch(contacts.rigid_contact_count)[0])
    return count


_LOCAL_PICKUP_DESCENT_STEP_M = 0.002
_LOCAL_PICKUP_CLEARANCE_TRANSLATION_TOLERANCE_M = 0.002
_LOCAL_PICKUP_CLEARANCE_ROTATION_TOLERANCE_RAD = math.radians(2.0)
_CANONICAL_GOAL_CERTIFICATE_NAMES = {
    "format",
    "schema_version",
    "metadata",
    "goal_state",
    "row_rng_state",
    "content_sha256",
}
_CANONICAL_GOAL_CERTIFICATE_METADATA_NAMES = {
    "generator",
    "certifier_env_count",
    "task_contract",
    "physical_contract",
    "generation_contract",
    "package_versions",
    "source_sha256",
    "production_evidence",
    "rng_contract",
}
_GENERATION_CHECKPOINT_NAMES = {"format", "schema_version", "metadata", "progress", "content_sha256"}
_GENERATION_CHECKPOINT_METADATA_NAMES = {
    "generator",
    "artifact_contract",
    "generator_cfg",
    "generation_contract",
    "task_contract",
    "physical_contract",
    "package_versions",
    "source_sha256",
    "asset_closure",
    "canonical_goal_certificate",
    "initial_row_rng_contract",
}
_GENERATION_CHECKPOINT_PROGRESS_NAMES = {
    "status",
    "created_utc",
    "canonical_goal",
    "canonical_goal_evidence",
    "completed_batches",
    "accepted_chunks",
    "attempt_counts",
    "rejection_counts",
    "accepted_oracle_metrics",
    "logical_ik_solve_call_count",
    "row_rng_state",
    "final_artifact",
}
_GENERATION_CHECKPOINT_BATCH_NAMES = {"ordinal", "phase", "phase_batch_index", "row_ids"}
_GENERATION_CHECKPOINT_CHUNK_NAMES = {"ordinal", "phase", "row_ids", "states"}
_GENERATION_CHECKPOINT_FINAL_ARTIFACT_NAMES = {"content_sha256", "row_count", "permutation"}
_GENERATION_CHECKPOINT_STATUSES = {
    "goal-ready",
    "generating",
    "rows-complete",
    "artifact-ready",
    "stable-published",
}
_CANONICAL_GOAL_SOURCE_ROOTS = (
    "source/isaaclab/isaaclab",
    "source/isaaclab_newton/isaaclab_newton",
    "source/isaaclab_contrib/isaaclab_contrib",
    "source/isaaclab_assets/isaaclab_assets",
    "source/isaaclab_tasks/isaaclab_tasks",
)
_CANONICAL_GOAL_SOURCE_FILES = (
    "scripts/tools/generate_franka_rj45_pick_insert_reset_dataset.py",
    "scripts/tools/_franka_rj45_reset_tools.py",
    "uv.lock",
)


class _RecoveryDiagnosticBatchComplete(Exception):
    """Stop an exact production phase loop after its first completed batch."""

    def __init__(self, phase: int, batch_index: int, accepted_row_count: int) -> None:
        super().__init__(phase, batch_index, accepted_row_count)
        self.phase = phase
        self.batch_index = batch_index
        self.accepted_row_count = accepted_row_count


def _canonical_goal_source_digests() -> dict[str, str]:
    """Return stable repository-relative digests for every goal-defining source."""
    relative_names = set(_CANONICAL_GOAL_SOURCE_FILES)
    for relative_root in _CANONICAL_GOAL_SOURCE_ROOTS:
        source_root = _REPO_ROOT / relative_root
        if not source_root.is_dir():
            raise FileNotFoundError(f"Canonical-goal source root is missing: {relative_root}.")
        relative_names.update(
            source.relative_to(_REPO_ROOT).as_posix()
            for source in source_root.rglob("*")
            if source.is_file() and source.suffix in {".py", ".pyi"}
        )
    result: dict[str, str] = {}
    for relative_name in sorted(relative_names):
        source = _REPO_ROOT / relative_name
        if not source.is_file():
            raise FileNotFoundError(f"Canonical-goal source is missing: {relative_name}.")
        result[relative_name] = hashlib.sha256(source.read_bytes()).hexdigest()
    return result


def _require_unchanged_canonical_goal_validation_snapshot(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    operation: str,
) -> None:
    """Reject an artifact if any live validation input changed during its proof."""
    if reset_dataset_digest(before) != reset_dataset_digest(after):
        raise RuntimeError(
            f"Canonical-goal validation/source snapshot changed during {operation}; no artifact was emitted."
        )


def _canonical_goal_package_versions() -> dict[str, str]:
    """Return exact runtime package identifiers covered by the certificate."""
    return {**package_versions(), "torch": str(torch.__version__)}


def _pickup_construction_sequence_contract() -> dict[str, Any]:
    """Return the versioned construction-only pickup sequence bound into artifacts."""
    return {
        "construction_sequence_version": _PICKUP_CONSTRUCTION_SEQUENCE_VERSION,
        "construction_robot_staging": "kinematic-open-clearance-before-loose-task-placement",
        "coherent_task_placement": "task-only-rigid-write-preserves-staged-robot-state-and-targets",
        "drive_free_local_alignment": "fresh-live-clearance-with-2mm-translation-and-2deg-rotation-gates",
        "physical_close": "prepositioned-2mm-step-descent-bilateral-close-and-post-contact-settle",
        "local_descent_maximum_translation_step_m": _LOCAL_PICKUP_DESCENT_STEP_M,
        "local_descent_waypoint_count": 23,
        "live_clearance_maximum_translation_error_m": _LOCAL_PICKUP_CLEARANCE_TRANSLATION_TOLERANCE_M,
        "live_clearance_maximum_rotation_error_rad": _LOCAL_PICKUP_CLEARANCE_ROTATION_TOLERANCE_RAD,
        "full_route_reserved_for_ungrasped_oracle": True,
    }


def _grasped_transport_schedule_contract(cfg: GeneratorCfg) -> dict[str, Any]:
    """Return the versioned loaded-carry schedule bound into artifacts."""
    return {
        "schedule_version": _GRASPED_TRANSPORT_SCHEDULE_VERSION,
        "planning": "precomputed-sequential-ik-knots",
        "execution": "c2-endpoint-time-law-piecewise-linear-joint-cruise",
        "time_law": "c2-endpoint-ramp-with-constant-speed-cruise",
        "time_law_endpoint_continuity": "C2",
        "joint_path_interpolation": "piecewise-linear-through-precomputed-ik-knots",
        "joint_path_internal_knot_continuity": "C0-with-bounded-target-velocity-jumps",
        "internal_knot_settles": 0,
        "segment_end_settle": True,
        "scope": "all-shared-scripted-grasped-carry-including-phase-realization-and-canonical-reseat",
        "transient_cable_speed_policy": "sample-every-step-observation-only-during-scripted-carry",
        "transient_cable_speed_is_rejection_gate": False,
        "stored_final_reset_replay_cable_speed_limit_m_s": cfg.maximum_row_cable_speed_m_s,
        "final_cable_speed_is_rejection_gate": True,
        "cold_reset_replay_cable_speed_is_rejection_gate": True,
        "canonical_reseat_followup_goal_cable_speed_limit_m_s": cfg.maximum_goal_cable_speed_m_s,
        "transient_motion_speed_gates": {
            "plug_linear_m_s": cfg.maximum_grasped_transport_plug_linear_speed_m_s,
            "plug_angular_rad_s": cfg.maximum_grasped_transport_plug_angular_speed_rad_s,
            "arm_joint_rad_s": cfg.maximum_grasped_transport_arm_joint_speed_rad_s,
            "finger_joint_m_s": cfg.maximum_row_finger_joint_speed_m_s,
        },
        "c2_ramp_fraction": _GRASPED_TRANSPORT_C2_RAMP_FRACTION,
        "maximum_normalized_progress_rate": 1.0 / (1.0 - _GRASPED_TRANSPORT_C2_RAMP_FRACTION),
        "segment_duration_per_knot_s": cfg.grasped_transport_waypoint_motion_s,
        "maximum_translation_step_m": cfg.grasped_transport_maximum_translation_step_m,
        "maximum_rotation_step_rad": cfg.grasped_transport_maximum_rotation_step_rad,
        "maximum_raw_ik_joint_step_rad": cfg.grasped_transport_maximum_raw_ik_joint_step_rad,
        "derived_maximum_internal_target_velocity_jump_rad_s": (
            2.0
            * cfg.grasped_transport_maximum_raw_ik_joint_step_rad
            / ((1.0 - _GRASPED_TRANSPORT_C2_RAMP_FRACTION) * cfg.grasped_transport_waypoint_motion_s)
        ),
        "maximum_waypoints": cfg.grasped_transport_maximum_waypoints,
        "endpoint_policies": {
            "canonical-and-default": {
                "policy": _GRASPED_TRANSPORT_STRICT_ENDPOINT_POLICY,
                "position_tolerance_m": cfg.tcp_compensation_tolerance_m,
                "terminal_correction_enabled": True,
            },
            "reset-row-phases-0-through-3": {
                "policy": _GRASPED_TRANSPORT_RESET_ROW_ENDPOINT_POLICY,
                "position_tolerance_m": cfg.grasped_transport_row_endpoint_position_tolerance_m,
                "terminal_correction_enabled": False,
                "acceptance": "settled-endpoint",
            },
        },
        "terminal_correction": {
            "scope": "strict-endpoint-policy-only",
            "trigger": "plug-position-or-tcp-position-outside-final-tolerance",
            "priority": "plug-position-before-tcp-position",
            "translation_vector": "selected-goal-position-minus-selected-live-position",
            "maximum_translation_step_m": cfg.grasped_transport_final_correction_step_m,
            "rotation_correction": "none",
            "maximum_iterations": cfg.grasped_transport_final_correction_max_iterations,
            "position_tolerance_m": cfg.tcp_compensation_tolerance_m,
            "progress_gate": (
                "selected-metric-reaches-tolerance-or-improves-beyond-epsilon-and-"
                "already-in-tolerance-secondary-remains-within-tolerance-plus-epsilon"
            ),
            "progress_epsilon_m": _GRASPED_TRANSPORT_TERMINAL_PROGRESS_EPSILON_M,
            "ik_raw-step-and-joint-limit-gates_unchanged": True,
            "final_strict_tcp-and-plug-position-and-orientation-gates_unchanged": True,
            "final_reset_row_orientation-and-physical-gates_unchanged": True,
        },
    }


def _phase_4_pregrasp_orientation_sampling_contract(cfg: GeneratorCfg) -> dict[str, Any]:
    """Describe the bounded open-pregrasp orientation curriculum."""
    contract = pick_insert_phase_4_pregrasp_orientation_sampling_contract()
    configured = (
        cfg.phase_4_pregrasp_height_m,
        cfg.phase_4_pregrasp_maximum_top_down_tilt_error_rad,
        cfg.phase_4_pregrasp_maximum_closing_axis_twist_error_rad,
    )
    expected = (
        contract["clearance_height_m"],
        contract["top_down_tilt_range_rad"][1],
        contract["closing_axis_twist_range_rad"][1],
    )
    if configured != expected:
        raise ValueError("Phase-4 pregrasp sampling must exactly match the runtime task contract.")
    return contract


def _fast_reset_bank_profile_contract(cfg: GeneratorCfg) -> dict[str, Any]:
    """Describe the balanced fast reset-bank shape and reference profile."""
    counts = phase_counts(cfg)
    reference_profile = (
        not cfg.quick
        and cfg.rows_per_phase == _REFERENCE_FAST_ROWS_PER_PHASE
        and cfg.batch_size == _REFERENCE_FAST_BATCH_SIZE
    )
    return {
        "contract_version": 1,
        "profile": "balanced-20004-v1" if reference_profile else "custom-balanced-fast-ik",
        "reference_profile": reference_profile,
        "rows_per_phase": cfg.rows_per_phase,
        "phase_counts": counts,
        "total_rows": _REFERENCE_FAST_TOTAL_ROWS if reference_profile else sum(counts),
        "batch_size": cfg.batch_size,
        "maximum_batches_per_phase": cfg.max_batches_per_phase,
        "simulation_steps_per_row": 0,
    }


def _phase_0_reverse_curriculum_evidence(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize accepted phase-0 sampler evidence without storing trajectories."""
    band_counts = {name: 0 for name in PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_BAND_NAMES}
    shortfalls: list[float] = []
    runtime_success_count = 0
    for record in records:
        band_name = record.get("phase_0_reverse_curriculum_band")
        if band_name not in band_counts:
            raise RuntimeError("Accepted phase-0 row has no valid reverse-curriculum band evidence.")
        band_counts[str(band_name)] += 1
        shortfalls.append(float(record["phase_0_axial_shortfall_m"]))
        runtime_success_count += int(bool(record["initial_runtime_geometric_success"]))
    if not shortfalls:
        raise RuntimeError("Fast reset generation produced no phase-0 reverse-curriculum evidence.")
    row_count = len(records)
    band_fractions = {name: count / row_count for name, count in band_counts.items()}
    maximum_fraction_error = max(
        abs(band_fractions[name] - weight)
        for name, weight in zip(
            PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_BAND_NAMES,
            PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_WEIGHTS,
            strict=True,
        )
    )
    allowed_fraction_error = pick_insert_fast_reset_phase_0_band_fraction_tolerance(row_count)
    if maximum_fraction_error > allowed_fraction_error:
        raise RuntimeError(
            "Accepted phase-0 reverse-curriculum band fractions exceed the deterministic tolerance: "
            f"fractions={band_fractions}, maximum_error={maximum_fraction_error}, "
            f"allowed={allowed_fraction_error}."
        )
    return {
        "accepted_row_count": row_count,
        "accepted_band_counts": band_counts,
        "accepted_band_fractions": band_fractions,
        "maximum_absolute_band_fraction_error": maximum_fraction_error,
        "allowed_absolute_band_fraction_error": allowed_fraction_error,
        "band_proportions_within_tolerance": True,
        "minimum_axial_shortfall_m": min(shortfalls),
        "maximum_axial_shortfall_m": max(shortfalls),
        "initial_runtime_geometric_success_count": runtime_success_count,
        "all_rows_preseat_and_outside_geometric_success": runtime_success_count == 0,
        "simulation_steps": 0,
    }


def _bind_fast_accepted_metrics_to_final_rows(
    accepted_metrics: Mapping[int, Sequence[Mapping[str, Any]]],
    final_states: Mapping[str, torch.Tensor],
    permutation: torch.Tensor,
) -> dict[str, list[dict[str, Any]]]:
    """Bind every fast admission record to its final permuted artifact row."""
    source_records: list[tuple[int, Mapping[str, Any]]] = []
    for phase in PICK_INSERT_RESET_PHASE_IDS:
        source_records.extend((phase, record) for record in accepted_metrics[phase])
    source_indices = permutation.detach().cpu().tolist()
    if len(source_records) != len(source_indices) or sorted(source_indices) != list(range(len(source_records))):
        raise RuntimeError("Fast accepted-row evidence does not match the final artifact permutation.")

    bound = {str(phase): [] for phase in PICK_INSERT_RESET_PHASE_IDS}
    for final_row_id, source_row_id in enumerate(source_indices):
        phase, source_record = source_records[source_row_id]
        if int(final_states["phase"][final_row_id]) != phase:
            raise RuntimeError("Fast accepted-row evidence phase does not match its final artifact row.")
        if "final_row_id" in source_record or "state_sha256" in source_record:
            raise RuntimeError("Unbound fast accepted-row evidence contains reserved row-binding fields.")
        record = dict(source_record)
        record["final_row_id"] = final_row_id
        record["state_sha256"] = pick_insert_reset_dataset_row_digest(final_states, final_row_id)
        bound[str(phase)].append(record)
    return bound


def _canonical_goal_generation_contract(cfg: GeneratorCfg) -> dict[str, Any]:
    """Return the canonical row-generation contract independent of certifier width."""
    generator_cfg = asdict(cfg)
    # Goal certification is independent of how already-certified reset rows
    # are screened.  Keeping this out of the goal contract also lets fast-IK
    # consume an existing, still task/physics-compatible goal certificate.
    generator_cfg.pop("generation_mode", None)
    generator_cfg.update(
        {
            "rows_per_phase": _CANONICAL_ROWS_PER_PHASE,
            "batch_size": _CANONICAL_BATCH_SIZE,
            "quick": False,
        }
    )
    return {
        "generator_cfg": generator_cfg,
        "canonical_rows_per_phase": _CANONICAL_ROWS_PER_PHASE,
        "canonical_batch_size": _CANONICAL_BATCH_SIZE,
        "phase_ids": tuple(PICK_INSERT_RESET_PHASE_IDS),
        "phase_names": tuple(PICK_INSERT_PHASE_NAMES),
        "phase_starts_grasped": _PHASE_STARTS_GRASPED,
        "dataset_state_names": tuple(RESET_DATASET_STATE_NAMES),
        "goal_state_names": tuple(RESET_DATASET_GOAL_STATE_NAMES),
        "row_sampling_rng_owner": "PickInsertResetDatasetGenerator.random",
        "row_ik_stream": {
            "owner": "PickInsertResetDatasetGenerator.ik",
            "owner_count": 1,
            "seed": cfg.seed,
            "sampler": _ROW_IK_SAMPLER,
            "stochastic_sampler": False,
            "seed_count": _ROW_IK_SEED_COUNT,
            "noise_std": _ROW_IK_NOISE_STD,
            "iterations": _ROW_IK_ITERATIONS,
            "serialized_cursor": False,
            "fresh_row_stream": True,
            "required_initial_state": "fresh-sampler-free-single-owner-before-any-solve",
            "goal_derivation_process_separate": True,
        },
        "pickup_construction_sequence": _pickup_construction_sequence_contract(),
        "oracle_entry_replay": {
            "contract_version": _ORACLE_ENTRY_REPLAY_CONTRACT_VERSION,
            "restore_source": "stored-candidate-with-vbd-pose-history",
            "controller_semantics": "persistent-absolute",
            "duration_s": PICK_INSERT_RESET_REPLAY_DURATION_S,
            "post_step_samples": PICK_INSERT_RESET_REPLAY_POST_STEP_SAMPLES,
            "post_replay_arm_target": "runtime-persistent-arm-target",
            "ungrasped_acquisition_move_attempt_count": _ORACLE_ACQUISITION_MOVE_ATTEMPT_COUNT,
            "ungrasped_acquisition_move_settle_s": _ORACLE_ACQUISITION_MOVE_SETTLE_S,
            "phase_modes": {
                str(phase): (
                    "verify-existing-physical-grasp-after-replay"
                    if starts_grasped
                    else "guarded-full-physical-acquisition-after-replay"
                )
                for phase, starts_grasped in enumerate(_PHASE_STARTS_GRASPED)
            },
            "gates": (
                "finite",
                "contact-buffers-empty-after-restore",
                "collision",
                "phase-contact-state",
                "construction-drives-disabled",
                "joint-and-target-limits",
                "absolute-target-unclamped-and-stable",
                "target-tracking",
                "robot-and-cable-speed",
                "task-body-drift",
                "vbd-pose-history-applied-exactly-once",
            ),
        },
        "grasped_transport_schedule": _grasped_transport_schedule_contract(cfg),
        "scripted_recovery": PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY,
    }


def _canonical_goal_task_contract_projection(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Remove reset-bank cardinality fields that cannot affect the certified goal."""
    projected = deepcopy(dict(contract))
    pick_insert = projected.get("pick_insert")
    if not isinstance(pick_insert, dict):
        raise ValueError("Canonical-goal task contract has no pick_insert mapping.")
    pick_insert.pop("reset_dataset_rows_per_phase", None)
    diversity = pick_insert.get("full_pick_diversity")
    if not isinstance(diversity, dict):
        raise ValueError("Canonical-goal task contract has no full-pick diversity mapping.")
    for name in ("minimum_unique_socket_rows", "minimum_unique_plug_rows", "minimum_unique_arm_rows"):
        diversity.pop(name, None)
    return projected


def _validity_and_recovery_failure_checks(
    validity_checks: Mapping[str, torch.Tensor],
    lane_failure_masks: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep acceptance unchanged while exposing every recovery failure as a pass check."""
    valid = torch.stack(tuple(validity_checks.values())).all(dim=0)
    reported_checks = dict(validity_checks)
    reported_checks.update(
        {f"oracle_recovery_lane_{reason}": ~failure_mask for reason, failure_mask in sorted(lane_failure_masks.items())}
    )
    return valid, reported_checks


def _scripted_recovery_diagnostic_evidence(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Select the complete shared Cartesian-v3 evidence contract for one batch."""
    return {name: metrics[name] for name in _RECOVERY_DIAGNOSTIC_EVIDENCE_NAMES}


def _plain_certificate_value(value: Any) -> Any:
    """Convert tensors in evidence to path-free, digestible plain containers."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(name): _plain_certificate_value(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain_certificate_value(item) for item in value)
    if isinstance(value, list):
        return [_plain_certificate_value(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"Unsupported canonical-goal certificate value: {type(value).__name__}.")


def _validate_certificate_sha256(value: Any, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be one lowercase hexadecimal SHA-256 digest.")
    return value


def _validate_canonical_goal_state(
    goal_state: Any,
    *,
    task_body_count: int,
) -> dict[str, torch.Tensor]:
    """Require the exact ten CPU tensors used by runtime canonical replay."""
    if not isinstance(goal_state, Mapping):
        raise TypeError("Canonical-goal certificate goal_state must be a mapping.")
    expected_names = set(RESET_DATASET_GOAL_STATE_NAMES)
    if set(goal_state) != expected_names:
        raise ValueError(
            "Canonical-goal certificate goal_state must contain exactly the ten runtime fields: "
            f"expected={sorted(expected_names)}, actual={sorted(goal_state)}."
        )
    specs = {
        "arm_joint_position": (7,),
        "arm_joint_target": (7,),
        "arm_joint_velocity": (7,),
        "finger_joint_position": (2,),
        "finger_joint_velocity": (2,),
        "finger_joint_target": (2,),
        "task_body_pose": (task_body_count, 7),
        "task_body_previous_pose": (task_body_count, 7),
        "task_body_coupling_previous_pose": (task_body_count, 7),
        "task_body_velocity": (task_body_count, 6),
    }
    validated: dict[str, torch.Tensor] = {}
    for name in RESET_DATASET_GOAL_STATE_NAMES:
        tensor = goal_state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Canonical-goal certificate goal_state.{name} must be a torch.Tensor.")
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32 or tuple(tensor.shape) != specs[name]:
            raise ValueError(
                f"Canonical-goal certificate goal_state.{name} must be CPU torch.float32 with shape "
                f"{specs[name]}, got {tensor.device}/{tensor.dtype}/{tuple(tensor.shape)}."
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Canonical-goal certificate goal_state.{name} must be finite.")
        validated[name] = tensor.detach().clone().contiguous()
    for name in (
        "task_body_pose",
        "task_body_previous_pose",
        "task_body_coupling_previous_pose",
    ):
        quaternion_norm = torch.linalg.vector_norm(validated[name][..., 3:7], dim=-1)
        if not bool(torch.allclose(quaternion_norm, torch.ones_like(quaternion_norm), atol=1.0e-4, rtol=0.0)):
            raise ValueError(f"Canonical-goal certificate goal_state.{name} quaternions must be normalized.")
    return validated


def _validate_production_goal_evidence(
    evidence: Any,
    *,
    physical_contract: Mapping[str, Any],
    certifier_env_count: int,
) -> dict[str, Any]:
    """Require production relaxation, reseat, and two cold proofs with no promotion."""
    if not isinstance(evidence, Mapping):
        raise TypeError("Canonical-goal certificate production_evidence must be a mapping.")
    if evidence.get("passed") is not True:
        raise ValueError("Canonical-goal certificate production evidence did not pass.")
    if evidence.get("classification") != "production-canonical-relax-reseat-cold-proof":
        raise ValueError("Canonical-goal certificate has the wrong production evidence classification.")
    if evidence.get("diagnostic_cli") is not False:
        raise ValueError("Canonical-goal certificates cannot contain diagnostic evidence.")
    evidence_physical = evidence.get("physical_contract")
    expected_evidence_physical = {
        "finger_closed_target_m": physical_contract["finger_closed_target_m"],
        "finger_raw_friction": float(GRASP_FRICTION),
        "grasp_proxy_raw_friction": physical_contract["live_grasp_proxy_raw_friction"],
        "effective_finger_proxy_friction": physical_contract["effective_finger_proxy_friction"],
    }
    if reset_dataset_digest(evidence_physical) != reset_dataset_digest(expected_evidence_physical):
        raise ValueError("Canonical-goal certificate production evidence has a different physical contract.")
    continuous = evidence.get("continuous_relaxation")
    reseat = evidence.get("authored_reseat")
    rolling = evidence.get("reseat_trailing_equilibrium")
    cold = evidence.get("cold_proofs")
    if not isinstance(continuous, Mapping) or continuous.get("passed") is not True:
        raise ValueError("Canonical-goal certificate is missing a passing continuous relaxation.")
    if not isinstance(reseat, Mapping) or reseat.get("count") != 1:
        raise ValueError("Canonical-goal certificate must contain exactly one authored reseat.")
    if not isinstance(rolling, Mapping) or rolling.get("passed") is not True:
        raise ValueError("Canonical-goal certificate is missing a passing post-reseat rolling window.")
    if not isinstance(cold, Mapping):
        raise ValueError("Canonical-goal certificate is missing cold-proof evidence.")
    if cold.get("same_original_capture_restored_both_times") is not True:
        raise ValueError("Canonical-goal certificate cold proofs must restore the same original capture.")
    if cold.get("endpoint_promotion_count") != 0:
        raise ValueError("Canonical-goal certificate cold proofs cannot promote an endpoint.")
    for name, stage in (("cold_30s", "canonical-cold-30s"), ("cold_60s", "canonical-cold-60s")):
        proof = cold.get(name)
        if not isinstance(proof, Mapping) or proof.get("passed") is not True or proof.get("stage") != stage:
            raise ValueError(f"Canonical-goal certificate is missing the passing {stage} proof.")
    construction_mask = evidence.get("construction_surviving_mask")
    final_mask = evidence.get("final_surviving_mask")
    final_lane_ids = evidence.get("final_surviving_lane_ids")
    if (
        not isinstance(construction_mask, list)
        or len(construction_mask) != certifier_env_count
        or not all(isinstance(value, bool) for value in construction_mask)
        or not isinstance(final_mask, list)
        or len(final_mask) != certifier_env_count
        or not all(isinstance(value, bool) for value in final_mask)
        or not isinstance(final_lane_ids, list)
        or not final_lane_ids
        or any(type(index) is not int or not 0 <= index < certifier_env_count for index in final_lane_ids)
    ):
        raise ValueError("Canonical-goal certificate has malformed survivor evidence.")
    expected_final_lane_ids = [index for index, survives in enumerate(final_mask) if survives]
    if final_lane_ids != expected_final_lane_ids:
        raise ValueError("Canonical-goal certificate final survivor mask and lane identifiers disagree.")
    if evidence.get("selected_original_lane") != min(final_lane_ids):
        raise ValueError("Canonical-goal certificate did not select the lowest original surviving lane.")
    return _plain_certificate_value(evidence)


def _build_canonical_goal_certificate(
    *,
    goal_state: Mapping[str, torch.Tensor],
    production_evidence: Mapping[str, Any],
    row_rng_state: torch.Tensor,
    certifier_env_count: int,
    task_body_count: int,
    task_contract: Mapping[str, Any],
    physical_contract: Mapping[str, Any],
    generation_contract: Mapping[str, Any],
    versions: Mapping[str, str],
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Build one self-contained canonical-goal certificate."""
    if certifier_env_count not in _CANONICAL_GOAL_CERTIFIER_ENV_COUNTS:
        raise ValueError("Canonical-goal certifier mode requires exactly one or four environments.")
    validated_goal = _validate_canonical_goal_state(goal_state, task_body_count=task_body_count)
    if (
        not isinstance(row_rng_state, torch.Tensor)
        or row_rng_state.device.type != "cpu"
        or row_rng_state.dtype != torch.uint8
        or row_rng_state.ndim != 1
        or row_rng_state.numel() == 0
    ):
        raise ValueError("Canonical-goal certificate row_rng_state must be a non-empty CPU uint8 vector.")
    row_rng_state = row_rng_state.detach().clone().contiguous()
    evidence = _validate_production_goal_evidence(
        production_evidence,
        physical_contract=physical_contract,
        certifier_env_count=certifier_env_count,
    )
    rng_digest = reset_dataset_digest(row_rng_state)
    payload: dict[str, Any] = {
        "format": _CANONICAL_GOAL_CERTIFICATE_FORMAT,
        "schema_version": _CANONICAL_GOAL_CERTIFICATE_SCHEMA_VERSION,
        "metadata": {
            "generator": Path(__file__).name,
            "certifier_env_count": certifier_env_count,
            "task_contract": _plain_certificate_value(task_contract),
            "physical_contract": _plain_certificate_value(physical_contract),
            "generation_contract": _plain_certificate_value(generation_contract),
            "package_versions": _plain_certificate_value(versions),
            "source_sha256": _plain_certificate_value(source_sha256),
            "production_evidence": evidence,
            "rng_contract": {
                "owner": "PickInsertResetDatasetGenerator.random",
                "seed": generation_contract["generator_cfg"]["seed"],
                "goal_derivation_consumed_row_rng": False,
                "state_sha256": rng_digest,
                "row_ik_stream": generation_contract["row_ik_stream"],
            },
        },
        "goal_state": validated_goal,
        "row_rng_state": row_rng_state,
    }
    payload["content_sha256"] = reset_dataset_content_digest(payload)
    return payload


def _validate_canonical_goal_certificate(
    certificate: Any,
    *,
    task_body_count: int,
    expected_task_contract: Mapping[str, Any],
    expected_physical_contract: Mapping[str, Any],
    expected_generation_contract: Mapping[str, Any],
    expected_versions: Mapping[str, str],
    expected_source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Validate an untrusted safely loaded certificate against the live process."""
    if not isinstance(certificate, Mapping):
        raise TypeError("Canonical-goal certificate must be a mapping.")
    if set(certificate) != _CANONICAL_GOAL_CERTIFICATE_NAMES:
        raise ValueError("Canonical-goal certificate has unexpected or missing top-level fields.")
    if certificate.get("format") != _CANONICAL_GOAL_CERTIFICATE_FORMAT:
        raise ValueError(f"Expected canonical-goal certificate format {_CANONICAL_GOAL_CERTIFICATE_FORMAT!r}.")
    if certificate.get("schema_version") != _CANONICAL_GOAL_CERTIFICATE_SCHEMA_VERSION:
        raise ValueError(
            f"Expected canonical-goal certificate schema version {_CANONICAL_GOAL_CERTIFICATE_SCHEMA_VERSION}."
        )
    content_sha256 = _validate_certificate_sha256(
        certificate.get("content_sha256"),
        field_name="canonical-goal certificate content_sha256",
    )
    if content_sha256 != reset_dataset_content_digest(certificate):
        raise ValueError("Canonical-goal certificate content digest does not match its payload.")
    metadata = certificate.get("metadata")
    if not isinstance(metadata, Mapping) or set(metadata) != _CANONICAL_GOAL_CERTIFICATE_METADATA_NAMES:
        raise ValueError("Canonical-goal certificate has unexpected or missing metadata fields.")
    if metadata.get("generator") != Path(__file__).name:
        raise ValueError("Canonical-goal certificate generator identity does not match this source.")
    certifier_env_count = metadata.get("certifier_env_count")
    if type(certifier_env_count) is not int or certifier_env_count not in _CANONICAL_GOAL_CERTIFIER_ENV_COUNTS:
        raise ValueError("Canonical-goal certificate certifier_env_count must be exactly one or four.")
    exact_contracts = (
        ("task", metadata.get("task_contract"), expected_task_contract),
        ("physical", metadata.get("physical_contract"), expected_physical_contract),
        ("generation", metadata.get("generation_contract"), expected_generation_contract),
        ("package-version", metadata.get("package_versions"), expected_versions),
        ("source-digest", metadata.get("source_sha256"), expected_source_sha256),
    )
    for label, actual, expected in exact_contracts:
        if reset_dataset_digest(actual) != reset_dataset_digest(expected):
            raise ValueError(f"Canonical-goal certificate {label} contract does not exactly match the live process.")
    goal_state = _validate_canonical_goal_state(certificate.get("goal_state"), task_body_count=task_body_count)
    row_rng_state = certificate.get("row_rng_state")
    if (
        not isinstance(row_rng_state, torch.Tensor)
        or row_rng_state.device.type != "cpu"
        or row_rng_state.dtype != torch.uint8
        or row_rng_state.ndim != 1
        or row_rng_state.numel() == 0
    ):
        raise ValueError("Canonical-goal certificate row_rng_state must be a non-empty CPU uint8 vector.")
    row_rng_state = row_rng_state.detach().clone().contiguous()
    rng_contract = metadata.get("rng_contract")
    expected_rng_contract = {
        "owner": "PickInsertResetDatasetGenerator.random",
        "seed": expected_generation_contract["generator_cfg"]["seed"],
        "goal_derivation_consumed_row_rng": False,
        "state_sha256": reset_dataset_digest(row_rng_state),
        "row_ik_stream": expected_generation_contract["row_ik_stream"],
    }
    if reset_dataset_digest(rng_contract) != reset_dataset_digest(expected_rng_contract):
        raise ValueError("Canonical-goal certificate row RNG contract or state does not match generation.")
    production_evidence = _validate_production_goal_evidence(
        metadata.get("production_evidence"),
        physical_contract=expected_physical_contract,
        certifier_env_count=certifier_env_count,
    )
    return {
        "format": certificate["format"],
        "schema_version": certificate["schema_version"],
        "metadata": {**dict(metadata), "production_evidence": production_evidence},
        "goal_state": goal_state,
        "row_rng_state": row_rng_state,
        "content_sha256": content_sha256,
    }


def _load_canonical_goal_certificate(path: Path, **validation_kwargs: Any) -> dict[str, Any]:
    """Safely load one certificate on CPU before validating every contract."""
    certificate = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=True)
    return _validate_canonical_goal_certificate(certificate, **validation_kwargs)


def _canonical_goal_certificate_embedding(certificate: Mapping[str, Any]) -> dict[str, Any]:
    """Return a path-free plain-data certificate copy for reset-dataset metadata."""
    return {
        "format": certificate["format"],
        "schema_version": certificate["schema_version"],
        "content_sha256": certificate["content_sha256"],
        "metadata": _plain_certificate_value(certificate["metadata"]),
        "goal_state": {
            name: certificate["goal_state"][name].detach().cpu().tolist() for name in RESET_DATASET_GOAL_STATE_NAMES
        },
    }


def _checkpoint_plain_value(value: Any) -> Any:
    """Return one strict JSON value with tuples normalized to lists."""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for name, item in value.items():
            if not isinstance(name, str):
                raise TypeError("Generation checkpoint mapping keys must be strings.")
            result[name] = _checkpoint_plain_value(item)
        return result
    if isinstance(value, tuple | list):
        return [_checkpoint_plain_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Generation checkpoints cannot contain non-finite floats.")
        return value
    if value is None or isinstance(value, bool | int | str):
        return value
    raise TypeError(f"Unsupported generation checkpoint value: {type(value).__name__}.")


_CHECKPOINT_TENSOR_DTYPES = {
    "bool": torch.bool,
    "float32": torch.float32,
    "int64": torch.int64,
    "uint8": torch.uint8,
}
_CHECKPOINT_TENSOR_DTYPE_NAMES = {value: name for name, value in _CHECKPOINT_TENSOR_DTYPES.items()}


def _encode_checkpoint_tensor(value: torch.Tensor) -> dict[str, Any]:
    """Encode one finite CPU tensor without pickle or device-local state."""
    if not isinstance(value, torch.Tensor):
        raise TypeError("Generation checkpoint tensor values must be torch.Tensor instances.")
    tensor = value.detach().cpu().contiguous()
    dtype_name = _CHECKPOINT_TENSOR_DTYPE_NAMES.get(tensor.dtype)
    if dtype_name is None:
        raise ValueError(f"Generation checkpoints do not support tensor dtype {tensor.dtype}.")
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise ValueError("Generation checkpoints cannot contain non-finite tensors.")
    return {
        "dtype": dtype_name,
        "shape": list(tensor.shape),
        "data": tensor.reshape(-1).tolist(),
    }


def _decode_checkpoint_tensor(
    value: Any,
    *,
    path: str,
    expected_dtype: torch.dtype | None = None,
    expected_shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Decode and strictly validate one JSON tensor record on CPU."""
    if not isinstance(value, Mapping) or set(value) != {"dtype", "shape", "data"}:
        raise ValueError(f"{path} must be an exact encoded tensor record.")
    dtype_name = value.get("dtype")
    if not isinstance(dtype_name, str) or dtype_name not in _CHECKPOINT_TENSOR_DTYPES:
        raise ValueError(f"{path}.dtype is unsupported.")
    dtype = _CHECKPOINT_TENSOR_DTYPES[dtype_name]
    if expected_dtype is not None and dtype != expected_dtype:
        raise ValueError(f"{path} has dtype {dtype}; expected {expected_dtype}.")
    shape = value.get("shape")
    if (
        not isinstance(shape, list)
        or any(type(dimension) is not int or dimension < 0 for dimension in shape)
        or (expected_shape is not None and tuple(shape) != expected_shape)
    ):
        raise ValueError(f"{path}.shape is invalid.")
    data = value.get("data")
    if not isinstance(data, list) or math.prod(shape) != len(data):
        raise ValueError(f"{path}.data length does not match its shape.")
    if dtype == torch.bool:
        valid_values = all(type(item) is bool for item in data)
    elif dtype in (torch.int64, torch.uint8):
        valid_values = all(type(item) is int for item in data)
        if dtype == torch.uint8:
            valid_values = valid_values and all(0 <= item <= 255 for item in data)
    else:
        valid_values = all(not isinstance(item, bool) and isinstance(item, int | float) for item in data)
        valid_values = valid_values and all(math.isfinite(float(item)) for item in data)
    if not valid_values:
        raise ValueError(f"{path}.data contains an invalid value for {dtype}.")
    tensor = torch.tensor(data, dtype=dtype).reshape(shape).contiguous()
    return tensor


def _generation_checkpoint_row_specs(task_body_count: int) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    """Return the exact non-leading shapes of serialized reset rows."""
    return {
        "arm_joint_position": (torch.float32, (7,)),
        "arm_joint_target": (torch.float32, (7,)),
        "arm_joint_velocity": (torch.float32, (7,)),
        "finger_joint_position": (torch.float32, (2,)),
        "finger_joint_velocity": (torch.float32, (2,)),
        "finger_joint_target": (torch.float32, (2,)),
        "task_body_pose": (torch.float32, (task_body_count, 7)),
        "task_body_previous_pose": (torch.float32, (task_body_count, 7)),
        "task_body_coupling_previous_pose": (torch.float32, (task_body_count, 7)),
        "task_body_velocity": (torch.float32, (task_body_count, 6)),
        "goal_task_body_pose": (torch.float32, (task_body_count, 7)),
        "goal_arm_joint_target": (torch.float32, (7,)),
        "phase": (torch.int64, ()),
        "starts_grasped": (torch.bool, ()),
        "difficulty": (torch.float32, ()),
        "initial_goal_error": (torch.float32, ()),
        "initial_tcp_grasp_distance": (torch.float32, ()),
        "progress_threshold": (torch.float32, ()),
    }


def _generation_checkpoint_content_digest(payload: Mapping[str, Any]) -> str:
    """Return the canonical schema-1 checkpoint digest excluding itself."""
    unsigned = dict(payload)
    unsigned.pop("content_sha256", None)
    return reset_dataset_digest(_checkpoint_plain_value(unsigned))


@dataclass(frozen=True)
class _ValidatedGenerationCheckpoint:
    """Validated JSON checkpoint plus its decoded CPU tensors."""

    document: dict[str, Any]
    canonical_goal: dict[str, torch.Tensor]
    accepted_chunks: list[dict[str, Any]]
    row_rng_state: torch.Tensor


def _validate_generation_checkpoint(  # noqa: C901
    payload: Any,
    *,
    expected_metadata: Mapping[str, Any],
) -> _ValidatedGenerationCheckpoint:
    """Validate an untrusted schema-1 generation checkpoint exactly."""
    if not isinstance(payload, Mapping) or set(payload) != _GENERATION_CHECKPOINT_NAMES:
        raise ValueError("Generation checkpoint has unexpected or missing top-level fields.")
    if payload.get("format") != _GENERATION_CHECKPOINT_FORMAT:
        raise ValueError(f"Expected generation checkpoint format {_GENERATION_CHECKPOINT_FORMAT!r}.")
    if payload.get("schema_version") != _GENERATION_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Expected generation checkpoint schema version {_GENERATION_CHECKPOINT_SCHEMA_VERSION}.")
    content_sha256 = _validate_certificate_sha256(
        payload.get("content_sha256"), field_name="generation checkpoint content_sha256"
    )
    if content_sha256 != _generation_checkpoint_content_digest(payload):
        raise ValueError("Generation checkpoint content digest does not match its payload.")
    metadata = payload.get("metadata")
    normalized_expected_metadata = _checkpoint_plain_value(expected_metadata)
    if not isinstance(metadata, Mapping) or set(metadata) != _GENERATION_CHECKPOINT_METADATA_NAMES:
        raise ValueError("Generation checkpoint has unexpected or missing metadata fields.")
    if reset_dataset_digest(metadata) != reset_dataset_digest(normalized_expected_metadata):
        raise ValueError("Generation checkpoint metadata does not exactly match the live production contract.")
    artifact_contract = metadata.get("artifact_contract")
    if not isinstance(artifact_contract, Mapping):
        raise ValueError("Generation checkpoint artifact_contract must be a mapping.")
    rows_per_phase = artifact_contract.get("rows_per_phase")
    batch_size = artifact_contract.get("batch_size")
    if type(rows_per_phase) is not int or rows_per_phase < 1:
        raise ValueError("Generation checkpoint rows_per_phase must be positive.")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Generation checkpoint batch_size must be positive.")
    expected_artifact_core = {
        "format": FRANKA_RJ45_PICK_INSERT_RESET_DATASET_FORMAT,
        "schema_version": FRANKA_RJ45_PICK_INSERT_RESET_DATASET_SCHEMA_VERSION,
        "rows_per_phase": rows_per_phase,
        "batch_size": batch_size,
        "phase_ids": list(PICK_INSERT_RESET_PHASE_IDS),
        "phase_names": list(PICK_INSERT_PHASE_NAMES),
    }
    if any(artifact_contract.get(name) != expected for name, expected in expected_artifact_core.items()):
        raise ValueError("Generation checkpoint is not for the exact production artifact shape.")
    max_batches_per_phase = artifact_contract.get("max_batches_per_phase")
    if type(max_batches_per_phase) is not int or max_batches_per_phase < 1:
        raise ValueError("Generation checkpoint max_batches_per_phase must be positive.")
    task_contract = metadata.get("task_contract")
    task_body_count = task_contract.get("task_body_count") if isinstance(task_contract, Mapping) else None
    if type(task_body_count) is not int or task_body_count < 1:
        raise ValueError("Generation checkpoint task contract has no valid task body count.")
    progress = payload.get("progress")
    if not isinstance(progress, Mapping) or set(progress) != _GENERATION_CHECKPOINT_PROGRESS_NAMES:
        raise ValueError("Generation checkpoint has unexpected or missing progress fields.")
    status = progress.get("status")
    if status not in _GENERATION_CHECKPOINT_STATUSES:
        raise ValueError("Generation checkpoint has an invalid lifecycle status.")
    created_utc = progress.get("created_utc")
    if not isinstance(created_utc, str):
        raise ValueError("Generation checkpoint created_utc must be a string.")
    try:
        parsed_created = datetime.fromisoformat(created_utc)
    except ValueError as exc:
        raise ValueError("Generation checkpoint created_utc is not ISO-8601.") from exc
    if parsed_created.tzinfo is None:
        raise ValueError("Generation checkpoint created_utc must include a timezone.")

    encoded_goal = progress.get("canonical_goal")
    if not isinstance(encoded_goal, Mapping) or set(encoded_goal) != set(RESET_DATASET_GOAL_STATE_NAMES):
        raise ValueError("Generation checkpoint canonical_goal has unexpected or missing fields.")
    goal_specs = {
        name: spec for name, spec in _generation_checkpoint_row_specs(task_body_count).items() if name in encoded_goal
    }
    canonical_goal = {
        name: _decode_checkpoint_tensor(
            encoded_goal[name],
            path=f"progress.canonical_goal.{name}",
            expected_dtype=goal_specs[name][0],
            expected_shape=goal_specs[name][1],
        )
        for name in RESET_DATASET_GOAL_STATE_NAMES
    }
    validated_goal = _validate_canonical_goal_state(canonical_goal, task_body_count=task_body_count)
    certificate_binding = metadata.get("canonical_goal_certificate")
    if not isinstance(certificate_binding, Mapping):
        raise ValueError("Generation checkpoint canonical-goal certificate binding must be a mapping.")
    if certificate_binding.get("goal_state_sha256") != reset_dataset_digest(validated_goal):
        raise ValueError("Generation checkpoint canonical goal does not match its certificate binding.")
    _checkpoint_plain_value(progress.get("canonical_goal_evidence"))

    completed_batches = progress.get("completed_batches")
    accepted_chunk_records = progress.get("accepted_chunks")
    if not isinstance(completed_batches, list) or not isinstance(accepted_chunk_records, list):
        raise ValueError("Generation checkpoint batches and accepted chunks must be lists.")
    attempt_counts = progress.get("attempt_counts")
    if (
        not isinstance(attempt_counts, list)
        or len(attempt_counts) != len(PICK_INSERT_RESET_PHASE_IDS)
        or any(type(value) is not int or value < 0 for value in attempt_counts)
    ):
        raise ValueError("Generation checkpoint attempt_counts are invalid.")
    rejection_counts = progress.get("rejection_counts")
    metrics = progress.get("accepted_oracle_metrics")
    expected_phase_keys = {str(phase) for phase in PICK_INSERT_RESET_PHASE_IDS}
    if not isinstance(rejection_counts, Mapping) or set(rejection_counts) != expected_phase_keys:
        raise ValueError("Generation checkpoint rejection_counts must cover the exact phase set.")
    if not isinstance(metrics, Mapping) or set(metrics) != expected_phase_keys:
        raise ValueError("Generation checkpoint accepted_oracle_metrics must cover the exact phase set.")
    for phase in PICK_INSERT_RESET_PHASE_IDS:
        counts = rejection_counts[str(phase)]
        phase_metrics = metrics[str(phase)]
        if (
            not isinstance(counts, Mapping)
            or any(not isinstance(name, str) or type(value) is not int or value < 0 for name, value in counts.items())
            or not isinstance(phase_metrics, list)
        ):
            raise ValueError(f"Generation checkpoint phase {phase} counters or metrics are invalid.")
        _checkpoint_plain_value(phase_metrics)
    logical_ik_count = progress.get("logical_ik_solve_call_count")
    if type(logical_ik_count) is not int or logical_ik_count < 0:
        raise ValueError("Generation checkpoint logical IK solve count must be non-negative.")
    row_rng_state = _decode_checkpoint_tensor(
        progress.get("row_rng_state"),
        path="progress.row_rng_state",
        expected_dtype=torch.uint8,
    )
    if row_rng_state.ndim != 1 or row_rng_state.numel() == 0:
        raise ValueError("Generation checkpoint row RNG state must be a non-empty CPU vector.")

    row_specs = _generation_checkpoint_row_specs(task_body_count)
    decoded_chunks: list[dict[str, Any]] = []
    chunk_by_ordinal: dict[int, dict[str, Any]] = {}
    for chunk_index, chunk in enumerate(accepted_chunk_records):
        if not isinstance(chunk, Mapping) or set(chunk) != _GENERATION_CHECKPOINT_CHUNK_NAMES:
            raise ValueError(f"Generation checkpoint accepted chunk {chunk_index} is malformed.")
        ordinal = chunk.get("ordinal")
        phase = chunk.get("phase")
        row_ids = chunk.get("row_ids")
        states = chunk.get("states")
        if (
            type(ordinal) is not int
            or ordinal < 0
            or ordinal in chunk_by_ordinal
            or type(phase) is not int
            or phase not in PICK_INSERT_RESET_PHASE_IDS
            or not isinstance(row_ids, list)
            or not row_ids
            or any(type(row_id) is not int for row_id in row_ids)
            or not isinstance(states, Mapping)
            or set(states) != set(RESET_DATASET_STATE_NAMES)
        ):
            raise ValueError(f"Generation checkpoint accepted chunk {chunk_index} is malformed.")
        decoded_states = {
            name: _decode_checkpoint_tensor(
                states[name],
                path=f"progress.accepted_chunks[{chunk_index}].states.{name}",
                expected_dtype=row_specs[name][0],
                expected_shape=(len(row_ids), *row_specs[name][1]),
            )
            for name in RESET_DATASET_STATE_NAMES
        }
        if not bool((decoded_states["phase"] == phase).all()):
            raise ValueError("Generation checkpoint accepted chunk phase tensor disagrees with its record.")
        decoded = {"ordinal": ordinal, "phase": phase, "row_ids": list(row_ids), "states": decoded_states}
        decoded_chunks.append(decoded)
        chunk_by_ordinal[ordinal] = decoded

    phase_batch_counts = [0 for _ in PICK_INSERT_RESET_PHASE_IDS]
    accepted_counts = [0 for _ in PICK_INSERT_RESET_PHASE_IDS]
    for ordinal, record in enumerate(completed_batches):
        if not isinstance(record, Mapping) or set(record) != _GENERATION_CHECKPOINT_BATCH_NAMES:
            raise ValueError(f"Generation checkpoint completed batch {ordinal} is malformed.")
        phase = record.get("phase")
        phase_batch_index = record.get("phase_batch_index")
        row_ids = record.get("row_ids")
        if (
            type(phase) is not int
            or phase not in PICK_INSERT_RESET_PHASE_IDS
            or type(phase_batch_index) is not int
            or not isinstance(row_ids, list)
            or len(row_ids) > batch_size
            or any(type(row_id) is not int for row_id in row_ids)
        ):
            raise ValueError(f"Generation checkpoint completed batch {ordinal} is malformed.")
        next_phase = next(
            (candidate for candidate in PICK_INSERT_RESET_PHASE_IDS if accepted_counts[candidate] < rows_per_phase),
            None,
        )
        expected_row_ids = (
            []
            if next_phase is None
            else list(
                range(
                    next_phase * rows_per_phase + accepted_counts[next_phase],
                    next_phase * rows_per_phase + min(rows_per_phase, accepted_counts[next_phase] + len(row_ids)),
                )
            )
        )
        if (
            record.get("ordinal") != ordinal
            or phase != next_phase
            or type(phase_batch_index) is not int
            or phase_batch_index != phase_batch_counts[phase]
            or phase_batch_index >= max_batches_per_phase
            or not isinstance(row_ids, list)
            or row_ids != expected_row_ids
        ):
            raise ValueError("Generation checkpoint completed batches are not an exact phase-major prefix.")
        phase_batch_counts[phase] += 1
        accepted_counts[phase] += len(row_ids)
        chunk = chunk_by_ordinal.pop(ordinal, None)
        if bool(row_ids) != (chunk is not None):
            raise ValueError("Generation checkpoint accepted chunks do not match completed batch rows.")
        if chunk is not None and (chunk["phase"] != phase or chunk["row_ids"] != row_ids):
            raise ValueError("Generation checkpoint accepted chunk identity disagrees with its batch record.")
    if chunk_by_ordinal or len(decoded_chunks) != sum(bool(record["row_ids"]) for record in completed_batches):
        raise ValueError("Generation checkpoint contains an uncommitted accepted chunk.")
    expected_chunk_ordinals = [record["ordinal"] for record in completed_batches if record["row_ids"]]
    if [chunk["ordinal"] for chunk in decoded_chunks] != expected_chunk_ordinals:
        raise ValueError("Generation checkpoint accepted chunks are not in committed batch order.")
    expected_attempt_counts = [count * batch_size for count in phase_batch_counts]
    if attempt_counts != expected_attempt_counts:
        raise ValueError("Generation checkpoint attempt counts do not match completed whole batches.")
    if any(len(metrics[str(phase)]) != accepted_counts[phase] for phase in PICK_INSERT_RESET_PHASE_IDS):
        raise ValueError("Generation checkpoint oracle metrics do not match accepted rows.")
    total_rows = rows_per_phase * len(PICK_INSERT_RESET_PHASE_IDS)
    all_rows_complete = accepted_counts == [rows_per_phase] * len(PICK_INSERT_RESET_PHASE_IDS)
    final_artifact = progress.get("final_artifact")
    if status == "goal-ready":
        if completed_batches or logical_ik_count or any(attempt_counts):
            raise ValueError("Goal-ready generation checkpoint cannot contain batch progress.")
        initial_rng = metadata.get("initial_row_rng_contract")
        if not isinstance(initial_rng, Mapping) or initial_rng.get("state_sha256") != reset_dataset_digest(
            row_rng_state
        ):
            raise ValueError("Goal-ready generation checkpoint does not contain the certified initial RNG state.")
    if status in {"rows-complete", "artifact-ready", "stable-published"} and not all_rows_complete:
        raise ValueError(f"Generation checkpoint status {status!r} requires every production row.")
    if status in {"goal-ready", "generating", "rows-complete"}:
        if final_artifact is not None:
            raise ValueError(f"Generation checkpoint status {status!r} cannot contain a final artifact record.")
    else:
        if (
            not isinstance(final_artifact, Mapping)
            or set(final_artifact) != _GENERATION_CHECKPOINT_FINAL_ARTIFACT_NAMES
        ):
            raise ValueError("Artifact-ready generation checkpoint has a malformed final artifact record.")
        _validate_certificate_sha256(
            final_artifact.get("content_sha256"), field_name="generation checkpoint final artifact content_sha256"
        )
        permutation = final_artifact.get("permutation")
        if (
            final_artifact.get("row_count") != total_rows
            or not isinstance(permutation, list)
            or len(permutation) != total_rows
            or any(type(index) is not int for index in permutation)
            or sorted(permutation) != list(range(total_rows))
        ):
            raise ValueError("Generation checkpoint final permutation is invalid.")
    return _ValidatedGenerationCheckpoint(
        document=_checkpoint_plain_value(payload),
        canonical_goal=validated_goal,
        accepted_chunks=decoded_chunks,
        row_rng_state=row_rng_state,
    )


def _load_generation_checkpoint(
    path: Path,
    *,
    expected_metadata: Mapping[str, Any],
) -> _ValidatedGenerationCheckpoint:
    """Load one untrusted JSON checkpoint and validate it without pickle."""
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read generation checkpoint {path}: {exc}") from exc
    return _validate_generation_checkpoint(payload, expected_metadata=expected_metadata)


def _write_generation_checkpoint_atomic(
    payload: Mapping[str, Any],
    path: Path,
    *,
    expected_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Durably replace one checkpoint only after a strict temp-file reload."""
    normalized = _checkpoint_plain_value(payload)
    normalized["content_sha256"] = _generation_checkpoint_content_digest(normalized)
    _validate_generation_checkpoint(normalized, expected_metadata=expected_metadata)
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(normalized, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        reloaded = _load_generation_checkpoint(temporary, expected_metadata=expected_metadata)
        if reloaded.document != normalized:
            raise RuntimeError("Generation checkpoint temp-file reload changed its canonical JSON payload.")
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    return normalized


def _unlink_generation_checkpoint_durable(path: Path) -> None:
    """Remove a successful checkpoint and fsync its containing directory."""
    destination = path.expanduser().resolve()
    destination.unlink(missing_ok=True)
    directory_descriptor = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


class _GenerationCheckpointLock:
    """Non-following advisory lock held across generation, publish, and validation."""

    def __init__(self, checkpoint: Path, *, protected_paths: tuple[Path, ...] = ()) -> None:
        self.path = checkpoint.with_name(f"{checkpoint.name}.lock")
        self.protected_paths = tuple(path.expanduser().resolve() for path in protected_paths)
        self._descriptor: int | None = None

    @staticmethod
    def _identity(path_stat: os.stat_result) -> tuple[int, int]:
        return path_stat.st_dev, path_stat.st_ino

    def _require_safe_open_identity(self, descriptor: int) -> None:
        """Require one singly linked regular inode at the exact lock pathname."""
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise RuntimeError(f"Generation checkpoint lock must be a regular file: {self.path}")
        if opened_stat.st_nlink != 1:
            raise RuntimeError(f"Generation checkpoint lock must not be hard-linked: {self.path}")
        try:
            named_stat = self.path.lstat()
        except FileNotFoundError as exc:
            raise RuntimeError(f"Generation checkpoint lock pathname disappeared: {self.path}") from exc
        if stat.S_ISLNK(named_stat.st_mode) or not stat.S_ISREG(named_stat.st_mode):
            raise RuntimeError(f"Generation checkpoint lock must be a non-symlink regular file: {self.path}")
        if self._identity(named_stat) != self._identity(opened_stat):
            raise RuntimeError(f"Generation checkpoint lock identity changed while opening: {self.path}")
        opened_identity = self._identity(opened_stat)
        for protected_path in self.protected_paths:
            try:
                protected_stat = protected_path.stat()
            except FileNotFoundError:
                continue
            if self._identity(protected_stat) == opened_identity:
                raise RuntimeError(
                    f"Generation checkpoint lock cannot alias protected artifact {protected_path}: {self.path}"
                )

    def __enter__(self) -> _GenerationCheckpointLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing_stat = self.path.lstat()
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(existing_stat.st_mode) or not stat.S_ISREG(existing_stat.st_mode):
                raise RuntimeError(f"Generation checkpoint lock must be a non-symlink regular file: {self.path}")
            if existing_stat.st_nlink != 1:
                raise RuntimeError(f"Generation checkpoint lock must not be hard-linked: {self.path}")
        flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.ENXIO):
                raise RuntimeError(
                    f"Generation checkpoint lock must be a non-symlink regular file: {self.path}"
                ) from exc
            raise
        try:
            self._require_safe_open_identity(descriptor)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._require_safe_open_identity(descriptor)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise RuntimeError(f"Generation checkpoint is already locked: {self.path}") from exc
        except Exception:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        if self._descriptor is not None:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            os.close(self._descriptor)
            self._descriptor = None


def _validate_generation_checkpoint_invocation(args: argparse.Namespace) -> tuple[Path | None, bool]:
    """Resolve checkpoint CLI policy before simulator startup."""
    checkpoint = getattr(args, "checkpoint", None)
    resume_from = getattr(args, "resume_from", None)
    keep_checkpoint = bool(getattr(args, "keep_checkpoint", False))
    if checkpoint is not None and resume_from is not None:
        raise ValueError("--checkpoint and --resume-from are mutually exclusive.")
    requested = resume_from if resume_from is not None else checkpoint
    if requested is None:
        if keep_checkpoint:
            raise ValueError("--keep-checkpoint requires --checkpoint or --resume-from.")
        return None, False
    certificate_input = getattr(args, "canonical_goal_certificate_input", None)
    if certificate_input is None:
        raise ValueError("Generation checkpointing requires --canonical-goal-certificate-input.")
    certificate_input_path = Path(certificate_input).expanduser().resolve()
    if not certificate_input_path.is_file():
        raise ValueError(
            "Generation checkpointing requires an existing --canonical-goal-certificate-input file: "
            f"{certificate_input_path}"
        )
    resolved = Path(requested).expanduser().resolve()
    if resolved.suffix != ".json":
        raise ValueError("Generation checkpoint paths must use the .json suffix.")
    resuming = resume_from is not None
    if resuming and not resolved.is_file():
        raise ValueError(f"--resume-from must name an existing checkpoint file: {resolved}")
    if not resuming and resolved.exists():
        raise ValueError(f"--checkpoint refuses to overwrite an existing path: {resolved}")
    artifact_paths = {
        "reset-dataset output": Path(args.output).expanduser().resolve(),
        "canonical-goal certificate input": certificate_input_path,
    }
    if bool(getattr(args, "validate", False)):
        artifact_paths["stable validation report"] = _DEFAULT_STABLE_VALIDATION_REPORT_PATH.resolve()
    certificate_output = getattr(args, "canonical_goal_certificate_output", None)
    if certificate_output is not None:
        artifact_paths["canonical-goal certificate output"] = Path(certificate_output).expanduser().resolve()
    for label, artifact_path in artifact_paths.items():
        if resolved == artifact_path:
            raise ValueError(f"Generation checkpoint path cannot alias the {label}.")
        if resolved.with_name(f"{resolved.name}.lock") == artifact_path:
            raise ValueError(f"Generation checkpoint lock path cannot alias the {label}.")
    return resolved, resuming


def _validate_parsed_artifact_policy(args: argparse.Namespace) -> bool:  # noqa: C901
    """Validate artifact modes and return whether reset-dataset saving is disabled."""
    generation_mode = getattr(args, "generation_mode", _GENERATION_MODE_PHYSICAL_ORACLE)
    if generation_mode not in _GENERATION_MODES:
        raise ValueError(f"--generation-mode must be one of {_GENERATION_MODES}.")
    diagnostic_pickup_only = bool(getattr(args, "diagnostic_pickup_only", False))
    diagnostic_phase0_transport_only = bool(getattr(args, "diagnostic_phase0_transport_only", False))
    diagnostic_recovery_phase = getattr(args, "diagnostic_recovery_phase", None)
    if diagnostic_recovery_phase is not None and (
        isinstance(diagnostic_recovery_phase, bool)
        or not isinstance(diagnostic_recovery_phase, int)
        or diagnostic_recovery_phase not in (1, 2, 4, 5)
    ):
        raise ValueError("--diagnostic-recovery-phase must be 1, 2, 4, or 5.")
    diagnostic_mode_count = sum(
        (
            diagnostic_pickup_only,
            diagnostic_phase0_transport_only,
            diagnostic_recovery_phase is not None,
        )
    )
    if diagnostic_mode_count > 1:
        raise ValueError("Pickup, phase-0 transport, and recovery diagnostic modes are mutually exclusive.")
    batch_diagnostic = diagnostic_mode_count == 1
    certificate_batch_diagnostic = diagnostic_phase0_transport_only or diagnostic_recovery_phase is not None
    if diagnostic_pickup_only:
        batch_diagnostic_name = "Pickup-only"
    elif diagnostic_phase0_transport_only:
        batch_diagnostic_name = "Phase-0 transport-only"
    else:
        batch_diagnostic_name = f"Phase-{diagnostic_recovery_phase} recovery"
    checkpoint_control_requested = (
        getattr(args, "checkpoint", None) is not None
        or getattr(args, "resume_from", None) is not None
        or bool(getattr(args, "keep_checkpoint", False))
    )
    if batch_diagnostic and checkpoint_control_requested:
        raise ValueError(
            f"{batch_diagnostic_name} diagnostic mode cannot use generation checkpoint controls because it writes "
            "no artifact."
        )
    _validate_generation_checkpoint_invocation(args)
    certificate_output = getattr(args, "canonical_goal_certificate_output", None)
    certificate_input = getattr(args, "canonical_goal_certificate_input", None)
    if generation_mode == _GENERATION_MODE_FAST_IK and certificate_input is None:
        raise ValueError("Fast-IK generation requires --canonical-goal-certificate-input.")
    if generation_mode == _GENERATION_MODE_FAST_IK and (certificate_output is not None or diagnostic_mode_count):
        raise ValueError("Fast-IK generation cannot be combined with certifier or diagnostic modes.")
    if certificate_output is not None and certificate_input is not None:
        raise ValueError("Canonical-goal certificate input and output modes are mutually exclusive.")
    if (
        certificate_input is not None
        and Path(certificate_input).expanduser().resolve() == Path(args.output).expanduser().resolve()
    ):
        raise ValueError("Canonical-goal certificate input cannot also be the reset-dataset output path.")
    goal_diagnostic = bool(
        args.diagnostic_goal_only
        or args.diagnostic_reset_abcd
        or args.diagnostic_reset_e
        or args.diagnostic_p_relax_reseat
    )
    if diagnostic_phase0_transport_only and certificate_input is None:
        raise ValueError("Phase-0 transport-only diagnostic mode requires --canonical-goal-certificate-input.")
    if diagnostic_recovery_phase is not None and certificate_input is None:
        raise ValueError("Recovery diagnostic mode requires --canonical-goal-certificate-input.")
    if batch_diagnostic and (
        goal_diagnostic
        or certificate_output is not None
        or (certificate_input is not None and not certificate_batch_diagnostic)
        or bool(getattr(args, "diagnostic_zero_finger_close_target", False))
        or bool(getattr(args, "diagnostic_forward_grasp_offset", False))
        or bool(getattr(args, "diagnostic_effective_grasp_friction_three", False))
    ):
        raise ValueError(f"{batch_diagnostic_name} diagnostic mode cannot be combined with goal or certificate modes.")
    if batch_diagnostic and (args.quick or getattr(args, "validate", False)):
        raise ValueError(
            f"{batch_diagnostic_name} diagnostic mode cannot use --quick or --validate because it writes no artifact."
        )
    diagnostic_only = goal_diagnostic or batch_diagnostic
    if certificate_output is not None:
        if Path(certificate_output).expanduser().resolve() == DEFAULT_DATASET_PATH.expanduser().resolve():
            raise ValueError("Canonical-goal certificate output cannot overwrite the canonical reset-dataset path.")
        if diagnostic_only:
            raise ValueError("Canonical-goal certifier mode cannot be combined with a diagnostic mode.")
        if args.quick or args.rows_per_phase != _CANONICAL_ROWS_PER_PHASE:
            raise ValueError("Canonical-goal certifier mode requires the non-quick canonical generation contract.")
        if args.batch_size not in _CANONICAL_GOAL_CERTIFIER_ENV_COUNTS:
            raise ValueError("Canonical-goal certifier mode requires --batch-size 1 or --batch-size 4.")
        if getattr(args, "validate", False):
            raise ValueError("Canonical-goal certifier mode does not create a reset dataset to validate.")
        return True
    if certificate_input is not None:
        if certificate_batch_diagnostic:
            if args.batch_size != _CANONICAL_BATCH_SIZE:
                raise ValueError(
                    f"Certificate-backed {batch_diagnostic_name.lower()} diagnostic mode requires production batch "
                    "size "
                    f"{_CANONICAL_BATCH_SIZE}."
                )
            if diagnostic_recovery_phase is not None and args.rows_per_phase != _CANONICAL_ROWS_PER_PHASE:
                raise ValueError(
                    "Certificate-backed recovery diagnostic mode requires exact production shape with "
                    f"{_CANONICAL_ROWS_PER_PHASE} rows per phase and batch size {_CANONICAL_BATCH_SIZE}."
                )
            return True
        if diagnostic_only:
            raise ValueError("Canonical-goal certificate input cannot be combined with a diagnostic mode.")
        if generation_mode == _GENERATION_MODE_FAST_IK:
            output = Path(args.output).expanduser().resolve()
            canonical_output = DEFAULT_DATASET_PATH.expanduser().resolve()
            reference_fast_shape = (
                not args.quick
                and args.rows_per_phase == _REFERENCE_FAST_ROWS_PER_PHASE
                and args.batch_size == _REFERENCE_FAST_BATCH_SIZE
            )
            if output == canonical_output and not reference_fast_shape:
                raise ValueError(
                    "Canonical fast-IK reset-bank output requires exactly "
                    f"{_REFERENCE_FAST_ROWS_PER_PHASE} rows per phase and batch size "
                    f"{_REFERENCE_FAST_BATCH_SIZE}; custom fast banks require an explicit noncanonical --output."
                )
            return False
        if args.quick or args.rows_per_phase != _CANONICAL_ROWS_PER_PHASE or args.batch_size != _CANONICAL_BATCH_SIZE:
            raise ValueError(
                "Canonical-goal certificate input requires exact non-quick production generation with "
                f"{_CANONICAL_ROWS_PER_PHASE} rows per phase and batch size {_CANONICAL_BATCH_SIZE}."
            )
        if Path(args.output).expanduser().resolve() == DEFAULT_DATASET_PATH.expanduser().resolve():
            raise ValueError(
                "Legacy 96-row physical-oracle generation cannot overwrite the canonical 20,004-row reset bank; "
                "use an explicit noncanonical --output path."
            )
        return False
    if diagnostic_only:
        return True

    output = Path(args.output).expanduser().resolve()
    canonical_output = DEFAULT_DATASET_PATH.expanduser().resolve()
    canonical_shape = (
        not args.quick and args.rows_per_phase == _CANONICAL_ROWS_PER_PHASE and args.batch_size == _CANONICAL_BATCH_SIZE
    )
    if output == canonical_output and not canonical_shape:
        raise ValueError(
            "The canonical reset dataset output is reserved for exact non-quick production generation "
            f"with {_CANONICAL_ROWS_PER_PHASE} rows per phase and batch size {_CANONICAL_BATCH_SIZE}. "
            "Quick or custom-size generation requires an explicit noncanonical --output path."
        )
    if output == canonical_output:
        raise ValueError(
            "Canonical reset-dataset generation requires --canonical-goal-certificate-input; first create the "
            "production certificate with --canonical-goal-certificate-output."
        )
    return False


@dataclass
class _DiagnosticRollingDwell:
    """Track one uninterrupted diagnostic window without weakening any gate."""

    required_duration_s: float
    streak_start_time_s: float | None = None
    streak_start_sample_index: int | None = None
    maximum_streak_duration_s: float = 0.0
    miss_sample_count: int = 0
    miss_episode_count: int = 0
    first_miss_time_s: float | None = None
    _miss_active: bool = False

    def observe(self, elapsed_s: float, sample_index: int, passes: bool) -> bool:
        """Record one ordered sample and return whether the dwell is complete."""
        if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
            raise ValueError("Rolling-dwell sample times must be finite and non-negative.")
        if sample_index < 0:
            raise ValueError("Rolling-dwell sample indices must be non-negative.")
        if passes:
            if self.streak_start_time_s is None:
                self.streak_start_time_s = elapsed_s
                self.streak_start_sample_index = sample_index
            self._miss_active = False
            duration_s = elapsed_s - self.streak_start_time_s
            self.maximum_streak_duration_s = max(self.maximum_streak_duration_s, duration_s)
            return duration_s + 1.0e-12 >= self.required_duration_s

        self.streak_start_time_s = None
        self.streak_start_sample_index = None
        self.miss_sample_count += 1
        if self.first_miss_time_s is None:
            self.first_miss_time_s = elapsed_s
        if not self._miss_active:
            self.miss_episode_count += 1
        self._miss_active = True
        return False


@dataclass
class _PerLaneRollingDwell:
    """Track independent uninterrupted dwell windows for active simulation lanes."""

    required_duration_s: float
    lane_count: int
    device: torch.device | str
    streak_start_time_s: torch.Tensor = field(init=False)
    streak_start_sample_index: torch.Tensor = field(init=False)
    maximum_streak_duration_s: torch.Tensor = field(init=False)
    miss_sample_count: torch.Tensor = field(init=False)
    miss_episode_count: torch.Tensor = field(init=False)
    first_miss_time_s: torch.Tensor = field(init=False)
    _miss_active: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.required_duration_s) or self.required_duration_s <= 0.0:
            raise ValueError("Per-lane rolling-dwell duration must be finite and positive.")
        if isinstance(self.lane_count, bool) or self.lane_count < 1:
            raise ValueError("Per-lane rolling dwell requires at least one lane.")
        self.device = torch.device(self.device)
        self.streak_start_time_s = torch.full((self.lane_count,), torch.nan, device=self.device)
        self.streak_start_sample_index = torch.full((self.lane_count,), -1, device=self.device, dtype=torch.long)
        self.maximum_streak_duration_s = torch.zeros(self.lane_count, device=self.device)
        self.miss_sample_count = torch.zeros(self.lane_count, device=self.device, dtype=torch.long)
        self.miss_episode_count = torch.zeros_like(self.miss_sample_count)
        self.first_miss_time_s = torch.full((self.lane_count,), torch.nan, device=self.device)
        self._miss_active = torch.zeros(self.lane_count, device=self.device, dtype=torch.bool)

    def observe(
        self,
        elapsed_s: float,
        sample_index: int,
        passes: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Record one batch sample and return active lanes with a complete fresh dwell."""
        if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
            raise ValueError("Rolling-dwell sample times must be finite and non-negative.")
        if sample_index < 0:
            raise ValueError("Rolling-dwell sample indices must be non-negative.")
        passes = torch.as_tensor(passes, device=self.device, dtype=torch.bool)
        active_mask = torch.as_tensor(active_mask, device=self.device, dtype=torch.bool)
        expected_shape = (self.lane_count,)
        if tuple(passes.shape) != expected_shape or tuple(active_mask.shape) != expected_shape:
            raise ValueError(f"passes and active_mask must both have shape {expected_shape}.")

        passing = active_mask & passes
        missing = active_mask & ~passes
        starting = passing & torch.isnan(self.streak_start_time_s)
        self.streak_start_time_s.copy_(
            torch.where(starting, torch.full_like(self.streak_start_time_s, elapsed_s), self.streak_start_time_s)
        )
        self.streak_start_sample_index.copy_(
            torch.where(
                starting,
                torch.full_like(self.streak_start_sample_index, sample_index),
                self.streak_start_sample_index,
            )
        )
        duration_s = torch.where(
            passing,
            torch.full_like(self.streak_start_time_s, elapsed_s) - self.streak_start_time_s,
            torch.zeros_like(self.streak_start_time_s),
        )
        self.maximum_streak_duration_s.copy_(
            torch.maximum(self.maximum_streak_duration_s, torch.nan_to_num(duration_s))
        )
        new_miss_episode = missing & ~self._miss_active
        self.miss_sample_count.add_(missing.to(dtype=torch.long))
        self.miss_episode_count.add_(new_miss_episode.to(dtype=torch.long))
        self.first_miss_time_s.copy_(
            torch.where(
                missing & torch.isnan(self.first_miss_time_s),
                torch.full_like(self.first_miss_time_s, elapsed_s),
                self.first_miss_time_s,
            )
        )
        self.streak_start_time_s.copy_(
            torch.where(missing, torch.full_like(self.streak_start_time_s, torch.nan), self.streak_start_time_s)
        )
        self.streak_start_sample_index.copy_(
            torch.where(missing, torch.full_like(self.streak_start_sample_index, -1), self.streak_start_sample_index)
        )
        self._miss_active.copy_(torch.where(active_mask, ~passes, self._miss_active))
        return passing & (duration_s + 1.0e-12 >= self.required_duration_s)

    def evidence(self) -> dict[str, list[float] | list[int | None]]:
        """Return lane-indexed serializable counters and timing evidence."""
        first_miss = [None if math.isnan(value) else value for value in self.first_miss_time_s.cpu().tolist()]
        streak_start = [None if math.isnan(value) else value for value in self.streak_start_time_s.cpu().tolist()]
        return {
            "streak_start_time_s_by_lane": streak_start,
            "streak_start_sample_index_by_lane": self.streak_start_sample_index.cpu().tolist(),
            "maximum_streak_duration_s_by_lane": self.maximum_streak_duration_s.cpu().tolist(),
            "miss_sample_count_by_lane": self.miss_sample_count.cpu().tolist(),
            "miss_episode_count_by_lane": self.miss_episode_count.cpu().tolist(),
            "first_miss_time_s_by_lane": first_miss,
        }


def _quarantine_inactive_state(
    state: dict[str, torch.Tensor],
    active_mask: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], int]:
    """Copy every inactive state row from the deterministic lowest surviving lane."""
    if not state:
        raise ValueError("Canonical-goal quarantine requires a non-empty state mapping.")
    active_mask = torch.as_tensor(active_mask, dtype=torch.bool)
    if active_mask.ndim != 1:
        raise ValueError("Canonical-goal active_mask must be one-dimensional.")
    surviving = torch.where(active_mask)[0]
    if surviving.numel() == 0:
        raise RuntimeError("Canonical-goal quarantine has no surviving lane to use as a donor.")
    lane_count = int(active_mask.numel())
    donor = int(surviving[0])
    quarantined: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Canonical-goal state field {name!r} must be a tensor.")
        if value.ndim < 1 or value.shape[0] != lane_count:
            raise ValueError(f"Canonical-goal state field {name!r} must have leading batch dimension {lane_count}.")
        result = value.clone()
        inactive = ~active_mask.to(device=value.device)
        result[inactive] = value[donor]
        quarantined[name] = result
    return quarantined, donor


def _select_lowest_surviving_lane(
    state: dict[str, torch.Tensor],
    surviving_mask: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], int]:
    """Return the lowest original surviving lane as a contiguous CPU float32 state."""
    quarantined, selected_lane = _quarantine_inactive_state(state, surviving_mask)
    del quarantined
    return {
        name: value[selected_lane].detach().cpu().to(torch.float32).contiguous() for name, value in state.items()
    }, selected_lane


def _diagnostic_failure_statistics(samples: list[dict[str, Any]], pass_key: str) -> dict[str, Any]:
    """Summarize sampled failures and distinct recurrence episodes."""
    miss_indices = [index for index, sample in enumerate(samples) if not bool(sample[pass_key])]
    miss_episodes = sum(index == 0 or bool(samples[index - 1][pass_key]) for index in miss_indices)
    return {
        "miss_sample_count": len(miss_indices),
        "miss_episode_count": miss_episodes,
        "recurrent_miss_episode_count": max(0, miss_episodes - 1),
        "first_miss_time_s": None if not miss_indices else float(samples[miss_indices[0]]["time_s"]),
    }


def _diagnostic_exact_geometry_gate_masks(exact: Any, success_cfg: Any) -> dict[str, torch.Tensor]:
    """Return the speed-independent exact insertion geometry gates."""
    return {
        "exact_axial_error": exact.axial_error <= success_cfg.success_axial_tolerance,
        "exact_axial_overtravel": exact.signed_axial_error <= success_cfg.success_axial_overtravel_tolerance,
        "exact_radial_error": exact.radial_error <= success_cfg.success_radial_tolerance,
        "exact_plug_angle_error": exact.plug_angle_error <= success_cfg.success_plug_angle_tolerance,
        "exact_latch_angle_error": exact.latch_angle_error <= success_cfg.success_latch_angle_tolerance,
    }


def _summarize_diagnostic_rolling_window(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Return serializable full or qualifying-window diagnostic evidence."""
    if not samples:
        raise ValueError("A diagnostic rolling-window summary requires at least one sample.")
    start_body_positions = samples[0]["body_positions"]
    start_plug_position = samples[0]["plug_position"]
    start_plug_orientation = samples[0]["plug_orientation"]
    body_excursions: list[float] = []
    plug_translation_excursions: list[float] = []
    plug_orientation_excursions: list[float] = []
    for sample in samples:
        body_excursions.append(
            float(torch.linalg.vector_norm(sample["body_positions"] - start_body_positions, dim=-1).max())
        )
        plug_translation_excursions.append(
            float(torch.linalg.vector_norm(sample["plug_position"] - start_plug_position))
        )
        plug_orientation_excursions.append(
            float(
                math_utils.quat_error_magnitude(
                    sample["plug_orientation"].unsqueeze(0),
                    start_plug_orientation.unsqueeze(0),
                )[0]
            )
        )

    derived_metrics = {
        "body_excursion_m": body_excursions,
        "plug_translation_excursion_m": plug_translation_excursions,
        "plug_orientation_excursion_rad": plug_orientation_excursions,
    }
    sampled_metric_names = (
        "plug_linear_speed_m_s",
        "plug_angular_speed_rad_s",
        "plug_spatial_speed",
        "cable_speed_m_s",
        "arm_joint_speed_rad_s",
        "finger_joint_speed_m_s",
        "authored_seat_error_m",
        "authored_plug_tilt_rad",
        "plug_relative_latch_angle_rad",
        "arm_target_tracking_error_rad",
        "arm_target_drift_rad",
        "arm_target_clamp_delta_rad",
    )
    result: dict[str, Any] = {
        "start_time_s": float(samples[0]["time_s"]),
        "end_time_s": float(samples[-1]["time_s"]),
        "duration_s": float(samples[-1]["time_s"] - samples[0]["time_s"]),
        "sample_count": len(samples),
        "all_samples_hard_valid": all(bool(sample["hard_valid"]) for sample in samples),
        "all_samples_exact_success": all(bool(sample["exact_success"]) for sample in samples),
        "all_samples_speed_limits_satisfied": all(bool(sample["speed_limits_satisfied"]) for sample in samples),
        "all_samples_qualifying": all(bool(sample["qualifies"]) for sample in samples),
    }
    for name in sampled_metric_names:
        values = [float(sample[name]) for sample in samples]
        peak_index = max(range(len(values)), key=values.__getitem__)
        result[f"maximum_{name}"] = values[peak_index]
        result[f"maximum_{name}_time_s"] = float(samples[peak_index]["time_s"])
        result[f"final_{name}"] = values[-1]
    for name, values in derived_metrics.items():
        peak_index = max(range(len(values)), key=values.__getitem__)
        result[f"maximum_{name}"] = values[peak_index]
        result[f"maximum_{name}_time_s"] = float(samples[peak_index]["time_s"])
        result[f"final_{name}"] = values[-1]
    result["exact_success_failures"] = _diagnostic_failure_statistics(samples, "exact_success")
    result["speed_limit_failures"] = _diagnostic_failure_statistics(samples, "speed_limits_satisfied")
    result["qualification_failures"] = _diagnostic_failure_statistics(samples, "qualifies")
    result["speed_failure_breakdown"] = {
        name: _diagnostic_failure_statistics(samples, pass_key)
        for name, pass_key in (
            ("plug_spatial", "plug_spatial_speed_satisfied"),
            ("cable_linear", "cable_speed_satisfied"),
            ("arm_joint", "arm_speed_satisfied"),
            ("finger_joint", "finger_speed_satisfied"),
        )
    }
    return result


def _diagnostic_collision_summary(collision: Any) -> dict[str, Any]:
    """Return the existing collision reduction as plain diagnostic data."""
    return {
        "valid": collision.valid.detach().cpu().tolist(),
        "invalid_contact_count": collision.invalid_contact_count.detach().cpu().tolist(),
        "grasp_contact_count": collision.grasp_contact_count.detach().cpu().tolist(),
        "left_grasp_contact_count": collision.left_grasp_contact_count.detach().cpu().tolist(),
        "right_grasp_contact_count": collision.right_grasp_contact_count.detach().cpu().tolist(),
        "contact_overflow": collision.contact_overflow,
        "invalid_contact_pairs": list(collision.invalid_contact_pairs),
    }


def _diagnostic_pickup_lane_results(
    *,
    active_mask: torch.Tensor,
    survival_mask: torch.Tensor,
    placement_gate_masks: Mapping[str, torch.Tensor],
    acquisition_failure_masks: Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    """Explain every failed lane without changing the production gate reduction."""
    active = active_mask.detach().cpu().tolist()
    survived = survival_mask.detach().cpu().tolist()
    placement = {name: mask.detach().cpu().tolist() for name, mask in placement_gate_masks.items()}
    acquisition = {name: mask.detach().cpu().tolist() for name, mask in acquisition_failure_masks.items()}
    result: list[dict[str, Any]] = []
    for lane in range(len(active)):
        reasons = []
        if not active[lane]:
            reasons.append("initial/inactive")
        reasons.extend(f"placement/{name}" for name, passed in placement.items() if active[lane] and not passed[lane])
        reasons.extend(f"acquisition/{name}" for name, failed in acquisition.items() if failed[lane])
        if not survived[lane] and not reasons:
            reasons.append("pickup/final-validation")
        result.append(
            {
                "lane": lane,
                "active": active[lane],
                "survived": survived[lane],
                "failure_reasons": reasons,
            }
        )
    return result


def _diagnostic_pickup_acquisition_summary(
    acquisition_valid: torch.Tensor,
    evidence: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Serialize the acquisition evidence already produced by the production path."""
    if evidence is None:
        return None
    approach = evidence["open_approach"]
    post_contact = evidence["post_contact_settle"]
    return {
        "route": evidence.get("acquisition_route", "full-physical-approach"),
        "valid": acquisition_valid.detach().cpu().tolist(),
        "counts": {
            "local_acquisition_call_count": evidence.get("local_acquisition_call_count"),
            "advance_call_count": evidence.get("advance_call_count"),
            "control_step_count": evidence.get("control_step_count"),
            "ik_solve_call_count": evidence.get("ik_solve_call_count"),
            "local_descent_waypoint_count": evidence.get("local_descent_waypoint_count"),
        },
        "lane_failure_masks": {
            name: mask.detach().cpu().tolist() for name, mask in evidence["lane_failure_masks"].items()
        },
        "ik": {
            "raw_bias_preflight_valid": _plain_certificate_value(evidence.get("raw_bias_preflight_valid")),
            "raw_bias_preflight_maximum_joint_delta_rad": _plain_certificate_value(
                evidence.get("raw_bias_preflight_maximum_joint_delta_rad")
            ),
            "raw_bias_preflight_clearance_position_error_m": _plain_certificate_value(
                evidence.get("raw_bias_preflight_clearance_position_error_m")
            ),
            "raw_bias_preflight_clearance_orientation_error_rad": _plain_certificate_value(
                evidence.get("raw_bias_preflight_clearance_orientation_error_rad")
            ),
            "clearance_valid": evidence["clearance_ik_valid"].detach().cpu().tolist(),
            "open_descent_valid": evidence["open_descent_ik_valid"].detach().cpu().tolist(),
            "clearance_tcp_error_m": evidence["preclose_error"].detach().cpu().tolist(),
            "maximum_open_descent_tcp_error_m": evidence["maximum_open_descent_tcp_error"].detach().cpu().tolist(),
            "contact_tcp_error_m": evidence["contact_position_error"].detach().cpu().tolist(),
            "final_tcp_distance_m": evidence["final_distance"].detach().cpu().tolist(),
            "failure_diagnostics": evidence["ik_diagnostics"],
        },
        "drift": {
            "maximum_open_approach_plug_position_m": approach["maximum_plug_position_drift_m"].detach().cpu().tolist(),
        },
        "collision": {
            "preclose": _diagnostic_collision_summary(evidence["preclose_collision"]),
            "open_descent_valid": evidence["open_descent_collision_valid"].detach().cpu().tolist(),
            "contact_preclose": _diagnostic_collision_summary(evidence["contact_preclose_collision"]),
            "final": _diagnostic_collision_summary(evidence["collision"]),
            "open_approach_all_samples_valid": approach["all_samples_collision_free"].detach().cpu().tolist(),
            "post_contact_all_samples_valid": post_contact["all_samples_collision_free"].detach().cpu().tolist(),
        },
        "grasp": {
            "valid": evidence["grasp"].valid.detach().cpu().tolist(),
            "tcp_distance_m": evidence["grasp"].tcp_distance.detach().cpu().tolist(),
            "minimum_bilateral_finger_deflection_m": evidence["grasp"].bilateral_deflection.detach().cpu().tolist(),
        },
        "proxy": {
            "open_approach_all_samples_zero": approach["all_samples_zero_proxy_contacts"].detach().cpu().tolist(),
            "open_approach_maximum_left_contact_count": approach["maximum_left_proxy_contact_count"]
            .detach()
            .cpu()
            .tolist(),
            "open_approach_maximum_right_contact_count": approach["maximum_right_proxy_contact_count"]
            .detach()
            .cpu()
            .tolist(),
            "post_contact_all_samples_bilateral": post_contact["all_samples_bilateral_proxy_contact"]
            .detach()
            .cpu()
            .tolist(),
        },
        "sample_counts": {
            "open_approach": approach["samples"],
        },
        "open_approach_abort_reason": approach["abort_reason"],
        "post_contact_settle": _plain_certificate_value(post_contact),
    }


class _AdvanceCounter:
    """Count nested reset-tool advance calls while preserving the exact bound attribute."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.advance_call_count = 0
        self.control_step_count = 0
        self._original_advance = env.advance
        env_dict = getattr(env, "__dict__", {})
        self._had_instance_advance = "advance" in env_dict
        self._instance_advance = env_dict.get("advance")

    def _counted_advance(self, duration_s: float, update=None, *, post_step=None):
        steps = self._original_advance(duration_s, update, post_step=post_step)
        self.advance_call_count += 1
        self.control_step_count += int(steps)
        return steps

    def __enter__(self) -> _AdvanceCounter:
        self.env.advance = self._counted_advance
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._had_instance_advance:
            self.env.advance = self._instance_advance
        else:
            del self.env.advance


@dataclass(frozen=True)
class GeneratorCfg:
    """Bounded physical generation settings."""

    generation_mode: str = _GENERATION_MODE_PHYSICAL_ORACLE
    rows_per_phase: int = _CANONICAL_ROWS_PER_PHASE
    batch_size: int = _CANONICAL_BATCH_SIZE
    seed: int = 2026
    max_batches_per_phase: int = 96
    quick: bool = False
    diagnostic_reset_abcd: bool = False
    diagnostic_reset_e_only: bool = False
    diagnostic_p_relax_reseat: bool = False
    diagnostic_zero_finger_close_target: bool = False
    diagnostic_forward_grasp_offset: bool = False
    diagnostic_effective_grasp_friction_three: bool = False
    goal_drive_distance_m: float = 0.035
    goal_drive_ramp_s: float = 7.0
    goal_drive_hold_s: float = 1.0
    goal_drive_cable_settle_s: float = 20.0
    maximum_authored_plug_angle_rad: float = PICK_INSERT_GOAL_MAX_AUTHORED_PLUG_ANGLE_RAD
    goal_passive_settle_s: float = 15.0
    goal_stability_window_s: float = 2.0
    goal_passive_pre_stability_max_cycles: int = 6
    goal_cold_equilibrium_relax_s: float = 30.0
    goal_cold_fixed_point_max_cycles: int = 4
    goal_cold_replay_s: float = 10.0
    goal_cold_final_replay_s: float = 60.0
    canonical_peak_latch_angle_min_rad: float = 0.09
    robot_park_s: float = 0.25
    tcp_motion_s: float = 1.8
    tcp_settle_s: float = 0.35
    tcp_compensation_iterations: int = 5
    tcp_compensation_tolerance_m: float = 0.002
    tcp_compensation_max_step_m: float = 0.02
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
    grasp_close_s: float = 0.8
    grasp_hold_s: float = 1.5
    grasp_post_contact_settle_s: float = 1.0
    grasped_transport_maximum_translation_step_m: float = 0.002
    grasped_transport_maximum_rotation_step_rad: float = 0.03490658503988659
    grasped_transport_maximum_raw_ik_joint_step_rad: float = 0.02
    grasped_transport_waypoint_motion_s: float = 0.20
    grasped_transport_waypoint_settle_s: float = 1.0 / 30.0
    grasped_transport_final_settle_s: float = 0.75
    grasped_transport_final_correction_step_m: float = 0.001
    grasped_transport_final_correction_max_iterations: int = 6
    grasped_transport_row_endpoint_position_tolerance_m: float = 0.006
    grasped_transport_maximum_waypoints: int = 430
    maximum_grasped_transport_plug_linear_speed_m_s: float = 0.04
    maximum_grasped_transport_plug_angular_speed_rad_s: float = 0.35
    maximum_grasped_transport_arm_joint_speed_rad_s: float = 0.5
    pickup_transport_s: float = 3.0
    pickup_drive_hold_s: float = 1.0
    pickup_drive_free_settle_s: float = 2.0
    pickup_settle_s: float = 0.75
    row_settle_s: float = PICK_INSERT_RESET_REPLAY_DURATION_S
    finger_open_position: float = PICK_INSERT_OPEN_FINGER_POSITION
    finger_closed_target: float = PICK_INSERT_CLOSED_FINGER_POSITION
    phase_0_axial_offset_m: float = 0.006
    phase_1_axial_offset_m: float = 0.030
    phase_2_lift_m: float = 0.10
    phase_3_lift_m: float = 0.025
    phase_4_pregrasp_height_m: float = PICK_INSERT_PHASE_4_PREGRASP_HEIGHT_M
    phase_4_pregrasp_maximum_top_down_tilt_error_rad: float = (
        PICK_INSERT_PHASE_4_PREGRASP_MAXIMUM_TOP_DOWN_TILT_ERROR_RAD
    )
    phase_4_pregrasp_maximum_closing_axis_twist_error_rad: float = (
        PICK_INSERT_PHASE_4_PREGRASP_MAXIMUM_CLOSING_AXIS_TWIST_ERROR_RAD
    )
    maximum_socket_drift_m: float = PICK_INSERT_RESET_MAX_SOCKET_EXCURSION_M
    maximum_row_plug_drift_m: float = PICK_INSERT_RESET_MAX_PLUG_EXCURSION_M
    maximum_row_body_drift_m: float = PICK_INSERT_RESET_MAX_BODY_EXCURSION_M
    maximum_row_cable_speed_m_s: float = PICK_INSERT_RESET_MAX_CABLE_SPEED_M_S
    maximum_row_arm_joint_speed_rad_s: float = PICK_INSERT_RESET_MAX_ARM_JOINT_SPEED_RAD_S
    maximum_row_finger_joint_speed_m_s: float = PICK_INSERT_RESET_MAX_FINGER_JOINT_SPEED_M_S
    maximum_pickup_position_error_m: float = 0.025
    maximum_pickup_orientation_error_rad: float = 0.08726646259971647
    maximum_pickup_plug_linear_speed_m_s: float = 0.04
    maximum_pickup_plug_angular_speed_rad_s: float = 0.05
    minimum_pickup_cable_socket_center_distance_m: float = 0.02
    maximum_goal_body_drift_m: float = PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M
    maximum_goal_cable_speed_m_s: float = 0.01
    maximum_goal_arm_bias_rad: float = 0.1
    maximum_goal_arm_drift_rad: float = 0.01
    maximum_goal_arm_joint_speed_rad_s: float = PICK_INSERT_GOAL_MAX_ARM_JOINT_SPEED_RAD_S
    maximum_goal_finger_joint_speed_m_s: float = PICK_INSERT_GOAL_MAX_FINGER_JOINT_SPEED_M_S
    maximum_cable_support_penetration_m: float = 5.0e-4
    maximum_canonical_seat_error_m: float = PICK_INSERT_GOAL_MAX_AUTHORED_SEAT_ERROR_M
    recovery_motion_s: float = 4.0
    recovery_settle_s: float = 0.75
    recovery_compensation_iterations: int = 6
    recovery_compensation_tolerance_m: float = 0.0015
    maximum_ik_joint_step_rad: float = 0.6

    def __post_init__(self) -> None:
        if self.generation_mode not in _GENERATION_MODES:
            raise ValueError(f"generation_mode must be one of {_GENERATION_MODES}, got {self.generation_mode!r}.")
        for name in ("rows_per_phase", "batch_size", "max_batches_per_phase"):
            if isinstance(getattr(self, name), bool) or int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.tcp_compensation_tolerance_m != 0.002:
            raise ValueError("Canonical/default Cartesian endpoints require the exact 2 mm position tolerance.")
        if self.grasped_transport_row_endpoint_position_tolerance_m != 0.006:
            raise ValueError("Reset-row grasped-carry endpoints require the exact 6 mm position tolerance.")
        self._validate_remaining_controls()
        if self.goal_drive_distance_m != 0.035:
            raise ValueError("The canonical connector drive must remain exactly +35 mm.")

    def _validate_remaining_controls(self) -> None:
        """Validate diagnostic and physical controls after the scalar-count checks."""
        self._validate_diagnostic_controls()
        if self.goal_drive_ramp_s < 7.0:
            raise ValueError("The canonical +35 mm ramp must last at least seven simulated seconds.")
        if self.goal_drive_cable_settle_s < 10.0:
            raise ValueError("The extended cable must settle under the construction drive for at least ten seconds.")
        if self.goal_passive_settle_s < 10.0:
            raise ValueError("The canonical goal must remain drive-free for at least ten simulated seconds.")
        if not 0.0 < self.goal_stability_window_s < self.goal_passive_settle_s:
            raise ValueError("goal_stability_window_s must lie inside the passive goal replay.")
        if (
            isinstance(self.goal_passive_pre_stability_max_cycles, bool)
            or self.goal_passive_pre_stability_max_cycles < 1
        ):
            raise ValueError("goal_passive_pre_stability_max_cycles must be a positive integer.")
        if self.goal_cold_equilibrium_relax_s < 30.0:
            raise ValueError(
                "Cold-history cable equilibration must last at least thirty seconds before endpoint promotion."
            )

    def _validate_diagnostic_controls(self) -> None:
        """Keep every one-off physics change on an explicit no-artifact path."""
        if self.diagnostic_zero_finger_close_target and not self.diagnostic_p_relax_reseat:
            raise ValueError("diagnostic_zero_finger_close_target is only legal with diagnostic_p_relax_reseat.")
        if self.diagnostic_forward_grasp_offset and not (
            self.diagnostic_p_relax_reseat and self.diagnostic_zero_finger_close_target
        ):
            raise ValueError(
                "diagnostic_forward_grasp_offset is only legal with the zero-close P-relaxation diagnostic."
            )
        if self.diagnostic_effective_grasp_friction_three and not (
            self.diagnostic_p_relax_reseat and self.diagnostic_zero_finger_close_target
        ):
            raise ValueError(
                "diagnostic_effective_grasp_friction_three is only legal with the zero-close P-relaxation diagnostic."
            )
        if self.diagnostic_forward_grasp_offset and self.diagnostic_effective_grasp_friction_three:
            raise ValueError("Diagnostic grasp-offset and grasp-friction changes must remain isolated A/B tests.")
        if self.finger_closed_target != PICK_INSERT_CLOSED_FINGER_POSITION:
            raise ValueError("finger_closed_target must equal the immutable production pick-insert close target.")
        if self.maximum_goal_body_drift_m != PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M:
            raise ValueError("maximum_goal_body_drift_m must equal the immutable canonical-goal replay limit.")
        if self.goal_cold_fixed_point_max_cycles != 4:
            raise ValueError("The canonical cold-rewrite fixed point requires exactly four bounded attempts.")
        if self.goal_cold_replay_s < 10.0:
            raise ValueError("Canonical cold replay requires at least ten seconds.")
        if self.goal_cold_final_replay_s < 60.0:
            raise ValueError("The decisive persistent-target cold proof must last at least sixty seconds.")
        if not 0.0 < self.canonical_peak_latch_angle_min_rad < 0.2:
            raise ValueError("canonical_peak_latch_angle_min_rad must be a plausible positive hinge angle.")
        self._validate_open_approach()
        if self.pickup_drive_hold_s < 0.5:
            raise ValueError("Coherent loose-cable pickup placement requires a driven settling hold.")
        if self.pickup_drive_free_settle_s < 1.0:
            raise ValueError("The randomized loose cable must settle drive-free for at least one second.")
        if not 0.0 < self.maximum_pickup_position_error_m <= 0.025:
            raise ValueError("maximum_pickup_position_error_m must lie in (0, 0.025].")
        if not 0.0 < self.maximum_pickup_orientation_error_rad <= 0.08726646259971647:
            raise ValueError("maximum_pickup_orientation_error_rad must be at most five degrees.")
        if self.phase_4_pregrasp_height_m != PICK_INSERT_PHASE_4_PREGRASP_HEIGHT_M:
            raise ValueError("Phase-4 pregrasp height must remain at the exact 45 mm task-contract clearance.")
        if (
            self.phase_4_pregrasp_maximum_top_down_tilt_error_rad
            != PICK_INSERT_PHASE_4_PREGRASP_MAXIMUM_TOP_DOWN_TILT_ERROR_RAD
        ):
            raise ValueError("Phase-4 top-down tilt sampling must use the exact 0-25 degree task-contract cone.")
        if (
            self.phase_4_pregrasp_maximum_closing_axis_twist_error_rad
            != PICK_INSERT_PHASE_4_PREGRASP_MAXIMUM_CLOSING_AXIS_TWIST_ERROR_RAD
        ):
            raise ValueError("Phase-4 closing-axis twist sampling must use the exact +/-60 degree task-contract range.")
        if not 0.0 < self.maximum_pickup_plug_linear_speed_m_s <= self.maximum_row_cable_speed_m_s:
            raise ValueError("Pickup plug linear speed must be no larger than the row cable-speed limit.")
        if not 0.0 < self.maximum_pickup_plug_angular_speed_rad_s <= 0.05:
            raise ValueError("maximum_pickup_plug_angular_speed_rad_s must lie in (0, 0.05].")
        if self.minimum_pickup_cable_socket_center_distance_m < 2.0 * CABLE_RADIUS:
            raise ValueError("minimum_pickup_cable_socket_center_distance_m must exceed one cable diameter.")
        row_contract = {
            "row_settle_s": (self.row_settle_s, PICK_INSERT_RESET_REPLAY_DURATION_S),
            "maximum_socket_drift_m": (
                self.maximum_socket_drift_m,
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
        if not 0.0 < self.maximum_canonical_seat_error_m <= 0.002:
            raise ValueError("maximum_canonical_seat_error_m must be in (0, 0.002].")
        if not 0.0 < self.maximum_goal_arm_bias_rad <= 0.2:
            raise ValueError("maximum_goal_arm_bias_rad must be in (0, 0.2].")
        if not 0.0 < self.maximum_goal_arm_drift_rad <= 0.05:
            raise ValueError("maximum_goal_arm_drift_rad must be in (0, 0.05].")
        if not 0.0 < self.maximum_goal_arm_joint_speed_rad_s <= 0.5:
            raise ValueError("maximum_goal_arm_joint_speed_rad_s must be in (0, 0.5].")
        if not 0.0 < self.maximum_goal_finger_joint_speed_m_s <= 0.2:
            raise ValueError("maximum_goal_finger_joint_speed_m_s must be in (0, 0.2].")
        if not 0.0 <= self.maximum_cable_support_penetration_m <= 0.002:
            raise ValueError("maximum_cable_support_penetration_m must be in [0, 0.002].")
        if len(PICK_INSERT_PHASE_NAMES) != len(PICK_INSERT_RESET_PHASE_IDS):
            raise ValueError("The task config and reset schema must describe the same six phases.")

    def _validate_open_approach(self) -> None:
        if not 0.04 <= self.grasp_open_clearance_m <= 0.06:
            raise ValueError("grasp_open_clearance_m must preserve the measured 45 mm global pregrasp clearance.")
        if not 0.20 <= self.grasp_route_world_height_m <= 0.25:
            raise ValueError("The open-gripper route height must remain in the measured 0.20-0.25 m safe band.")
        if not 0.0 < self.grasp_route_maximum_translation_step_m <= 0.05:
            raise ValueError("The overhead Cartesian route step must be at most 50 mm.")
        if not 0.0 < self.grasp_coarse_descent_step_m <= 0.005:
            raise ValueError("The coarse open descent must use at most 5 mm Cartesian steps.")
        if not 0.0 < self.grasp_near_descent_step_m <= 0.002:
            raise ValueError("The near-plug open descent must use at most 2 mm Cartesian steps.")
        if self.grasp_descent_waypoint_motion_s < 0.1:
            raise ValueError("Each open-descent waypoint must allow at least 0.1 simulated seconds of tracking.")
        if not 0.0 <= self.grasp_descent_waypoint_settle_s <= 0.05:
            raise ValueError("Open-descent waypoint settle must remain in [0, 0.05] seconds.")
        if not 0.0 < self.grasp_descent_tracking_recovery_s <= 0.05:
            raise ValueError("Open-descent tracking recovery must remain in (0, 0.05] seconds.")
        if not 0.0 < self.grasp_near_correction_step_m <= 0.001:
            raise ValueError("Near-plug Cartesian feedback corrections must be at most 1 mm.")
        if (
            type(self.grasp_near_correction_max_iterations) is not int
            or not 1 <= self.grasp_near_correction_max_iterations <= 3
        ):
            raise ValueError("Near-plug Cartesian feedback must use one to three bounded corrections.")
        if not 0.0 < self.grasp_near_maximum_raw_ik_joint_step_rad <= 0.02:
            raise ValueError("Near-plug IK continuation must reject raw joint steps above 0.02 rad.")
        if not 0.0 < self.grasp_clearance_calibration_step_m <= 0.001:
            raise ValueError("The guarded 45 mm calibration must use at most 1 mm Cartesian corrections.")
        if self.grasp_clearance_calibration_max_iterations < 1:
            raise ValueError("The guarded 45 mm calibration requires at least one bounded correction.")
        if not 0.0 < self.maximum_open_approach_plug_drift_m <= 5.0e-4:
            raise ValueError("Open approach must reject plug drift beyond 0.5 mm.")
        if self.grasp_close_s < 0.8 or self.grasp_hold_s < 1.5:
            raise ValueError("The 0.04 m open gripper requires a 0.8 s close ramp and 1.5 s contact hold.")
        if self.grasp_post_contact_settle_s < 1.0:
            raise ValueError("Drive-free pickup must settle under bilateral grasp for at least one second.")
        if not 0.0 < self.grasped_transport_maximum_translation_step_m <= 0.002:
            raise ValueError("Grasped carry must use Cartesian translation steps no larger than 2 mm.")
        if not 0.0 < self.grasped_transport_maximum_rotation_step_rad <= 0.03490658503988659:
            raise ValueError("Grasped carry must use rotation steps no larger than two degrees.")
        if not 0.0 < self.grasped_transport_maximum_raw_ik_joint_step_rad <= 0.02:
            raise ValueError("Grasped carry must reject raw IK joint steps above 0.02 rad.")
        if self.grasped_transport_waypoint_motion_s < 0.20:
            raise ValueError("Each loaded grasped-carry waypoint must last at least 0.20 seconds.")
        if not 0.0 < self.grasped_transport_waypoint_settle_s <= 0.05:
            raise ValueError("Grasped-carry waypoint settle must lie in (0, 0.05] seconds.")
        if self.grasped_transport_final_settle_s < 0.5:
            raise ValueError("Grasped carry requires a bounded final equilibrium settle.")
        if not 0.0 < self.grasped_transport_final_correction_step_m <= 0.001:
            raise ValueError("Each terminal grasped-carry correction must be at most 1 mm.")
        if self.grasped_transport_final_correction_max_iterations != 6:
            raise ValueError("Terminal grasped-carry feedforward must retain its six-iteration hard cap.")
        if self.grasped_transport_maximum_waypoints != 430:
            raise ValueError("The staged grasped-carry route must retain its 430-waypoint hard cap.")
        if not 0.0 < self.maximum_grasped_transport_plug_linear_speed_m_s <= 0.04:
            raise ValueError("Grasped-carry plug linear speed must remain at most 0.04 m/s.")
        if not 0.0 < self.maximum_grasped_transport_plug_angular_speed_rad_s <= 0.35:
            raise ValueError("Grasped-carry plug angular speed must remain at most 0.35 rad/s.")
        if not 0.0 < self.maximum_grasped_transport_arm_joint_speed_rad_s <= 0.5:
            raise ValueError("Grasped-carry arm speed must remain at most 0.5 rad/s.")
        if self.finger_open_position != PICK_INSERT_OPEN_FINGER_POSITION:
            raise ValueError("Pick-insert generation requires the exact task 0.04 m open-finger posture.")


def phase_counts(cfg: GeneratorCfg) -> tuple[int, ...]:
    """Return the exact six-phase row counts."""
    return tuple(cfg.rows_per_phase for _ in PICK_INSERT_RESET_PHASE_IDS)


def sample_phase_0_reverse_curriculum_axial_shortfalls(
    sample_count: int,
    *,
    device: torch.device | str,
    rng: torch.Generator,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample phase-0 pre-seat shortfalls and their curriculum-band indices.

    One uniform variate per row selects both the weighted band and the uniform
    position inside that band.  This keeps the row stream deterministic across
    batch partitions while consuming exactly one random tensor per batch.

    Args:
        sample_count: Number of axial shortfalls to sample.
        device: Torch device that owns both returned tensors and ``rng``.
        rng: Dedicated deterministic reset-row random-number generator.
        dtype: Floating-point dtype of the returned shortfalls [m].

    Returns:
        Axial shortfalls [m], shape ``(sample_count,)``, and integer band
        indices, shape ``(sample_count,)``.
    """
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 0:
        raise ValueError("sample_count must be a non-negative integer.")
    if not dtype.is_floating_point:
        raise ValueError("Phase-0 axial shortfalls require a floating-point dtype.")
    ranges = torch.as_tensor(
        PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_AXIAL_RANGES_M,
        device=device,
        dtype=dtype,
    )
    weights = torch.as_tensor(
        PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_WEIGHTS,
        device=device,
        dtype=dtype,
    )
    if ranges.shape != (len(PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_BAND_NAMES), 2):
        raise RuntimeError("Phase-0 reverse-curriculum ranges do not match their band names.")
    if weights.shape != (len(PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_BAND_NAMES),):
        raise RuntimeError("Phase-0 reverse-curriculum weights do not match their band names.")

    draws = torch.rand((sample_count,), device=device, generator=rng, dtype=dtype)
    cumulative = torch.cumsum(weights, dim=0)
    band_indices = torch.bucketize(draws, cumulative[:-1], right=True)
    probability_lower = torch.cat((torch.zeros_like(cumulative[:1]), cumulative[:-1]))
    within_band = (draws - probability_lower[band_indices]) / weights[band_indices]
    selected_ranges = ranges[band_indices]
    shortfalls = selected_ranges[:, 0] + within_band * (selected_ranges[:, 1] - selected_ranges[:, 0])
    return shortfalls, band_indices


def phase_0_reverse_curriculum_band_indices(axial_shortfall_m: torch.Tensor) -> torch.Tensor:
    """Classify pre-seat axial shortfalls [m], returning ``-1`` outside all bands."""
    band_indices = torch.full_like(axial_shortfall_m, -1, dtype=torch.int64)
    last_band = len(PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_AXIAL_RANGES_M) - 1
    for band_index, (lower, upper) in enumerate(PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_AXIAL_RANGES_M):
        upper_valid = axial_shortfall_m <= upper if band_index == last_band else axial_shortfall_m < upper
        in_band = (axial_shortfall_m >= lower) & upper_valid
        band_indices = torch.where(in_band, band_index, band_indices)
    return band_indices


def sample_phase_4_pregrasp_orientation_errors(
    sample_count: int,
    *,
    device: torch.device | str,
    rng: torch.Generator,
    dtype: torch.dtype = torch.float32,
    maximum_top_down_tilt_error_rad: float = PICK_INSERT_PHASE_4_PREGRASP_MAXIMUM_TOP_DOWN_TILT_ERROR_RAD,
    maximum_closing_axis_twist_error_rad: float = (PICK_INSERT_PHASE_4_PREGRASP_MAXIMUM_CLOSING_AXIS_TWIST_ERROR_RAD),
) -> torch.Tensor:
    """Sample deterministic canonical-grasp-local phase-4 orientation errors.

    Tilt directions are uniform by solid angle in a cone about the canonical
    tool Z axis.  Closing-axis twist is uniform about that tool Z axis, and
    tilt is composed after twist so the sampled cone angle remains exact.

    Args:
        sample_count: Number of orientation errors to sample.
        device: Torch device that owns both the returned tensor and ``rng``.
        rng: Dedicated deterministic reset-row random-number generator.
        dtype: Floating-point dtype of the returned quaternions.
        maximum_top_down_tilt_error_rad: Maximum top-down cone angle [rad].
        maximum_closing_axis_twist_error_rad: Symmetric closing-axis twist limit [rad].

    Returns:
        Unit quaternions in XYZW convention, shape ``(sample_count, 4)``.
    """
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("sample_count must be a positive integer.")
    if not (0.0 < maximum_top_down_tilt_error_rad <= PICK_INSERT_PHASE_4_PREGRASP_MAXIMUM_TOP_DOWN_TILT_ERROR_RAD):
        raise ValueError("maximum_top_down_tilt_error_rad must lie in (0, 25 degrees].")
    if not (
        0.0 < maximum_closing_axis_twist_error_rad <= PICK_INSERT_PHASE_4_PREGRASP_MAXIMUM_CLOSING_AXIS_TWIST_ERROR_RAD
    ):
        raise ValueError("maximum_closing_axis_twist_error_rad must lie in (0, 60 degrees].")

    uniform = torch.rand((sample_count, 3), device=device, dtype=dtype, generator=rng)
    minimum_cosine = math.cos(maximum_top_down_tilt_error_rad)
    tilt_cosine = 1.0 - uniform[:, 0] * (1.0 - minimum_cosine)
    tilt_angle = torch.acos(tilt_cosine.clamp(-1.0, 1.0))
    tilt_azimuth = (2.0 * uniform[:, 1] - 1.0) * math.pi
    tilt_axis = torch.stack(
        (torch.cos(tilt_azimuth), torch.sin(tilt_azimuth), torch.zeros_like(tilt_azimuth)),
        dim=-1,
    )
    tilt = math_utils.quat_from_angle_axis(tilt_angle, tilt_axis)

    twist_angle = (2.0 * uniform[:, 2] - 1.0) * maximum_closing_axis_twist_error_rad
    twist_axis = torch.zeros((sample_count, 3), device=device, dtype=dtype)
    twist_axis[:, 2] = 1.0
    twist = math_utils.quat_from_angle_axis(twist_angle, twist_axis)
    error = math_utils.quat_mul(tilt, twist)
    return error / torch.linalg.vector_norm(error, dim=-1, keepdim=True).clamp_min(1.0e-9)


def _smoothstep(progress: float) -> float:
    return progress * progress * progress * (10.0 + progress * (-15.0 + 6.0 * progress))


def _grasped_transport_c2_progress(progress: float) -> float:
    """Map unit time to scalar path progress with C2 endpoint ramps and a constant-speed cruise."""
    if not math.isfinite(progress):
        raise ValueError("Grasped-transport schedule progress must be finite.")
    progress = min(max(progress, 0.0), 1.0)
    ramp = _GRASPED_TRANSPORT_C2_RAMP_FRACTION
    peak_rate = 1.0 / (1.0 - ramp)

    def ramp_integral(value: float) -> float:
        normalized = value / ramp
        return peak_rate * ramp * (normalized**3 - 0.5 * normalized**4)

    if progress < ramp:
        return ramp_integral(progress)
    if progress > 1.0 - ramp:
        return 1.0 - ramp_integral(1.0 - progress)
    return peak_rate * (0.5 * ramp + progress - ramp)


def _interpolate_grasped_transport_knots(knot_targets: torch.Tensor, path_progress: float) -> torch.Tensor:
    """Interpolate a monotone unit path through precomputed joint-target knots."""
    if knot_targets.ndim != 3 or knot_targets.shape[0] < 2:
        raise ValueError("Grasped-transport knots must have shape (K + 1, N, J) with K >= 1.")
    if not math.isfinite(path_progress):
        raise ValueError("Grasped-transport path progress must be finite.")
    path_progress = min(max(path_progress, 0.0), 1.0)
    knot_count = knot_targets.shape[0] - 1
    coordinate = path_progress * knot_count
    lower = min(int(math.floor(coordinate)), knot_count - 1)
    blend = coordinate - lower
    return torch.lerp(knot_targets[lower], knot_targets[lower + 1], blend)


def _grasped_transport_internal_target_velocity_jump(
    knot_targets: torch.Tensor,
    *,
    duration_per_knot_s: float,
) -> torch.Tensor:
    """Return each lane's maximum joint-target velocity jump at an internal knot [rad/s]."""
    if knot_targets.ndim != 3 or knot_targets.shape[0] < 2:
        raise ValueError("Grasped-transport knots must have shape (K + 1, N, J) with K >= 1.")
    if not math.isfinite(duration_per_knot_s) or duration_per_knot_s <= 0.0:
        raise ValueError("Grasped-transport duration per knot must be finite and positive.")
    if knot_targets.shape[0] == 2:
        return torch.zeros(knot_targets.shape[1], device=knot_targets.device, dtype=knot_targets.dtype)
    knot_delta = knot_targets[1:] - knot_targets[:-1]
    peak_rate = 1.0 / (1.0 - _GRASPED_TRANSPORT_C2_RAMP_FRACTION)
    return torch.abs(knot_delta[1:] - knot_delta[:-1]).amax(dim=(0, 2)) * peak_rate / duration_per_knot_s


def _grasped_transport_route_control_budget(
    segment_waypoint_counts: Sequence[int],
    *,
    duration_per_knot_s: float,
    segment_end_settle_s: float,
    advance_dt: float,
) -> dict[str, Any]:
    """Return exact full-route old/new control budgets under the tool environment's ceil stepping."""
    counts = tuple(int(count) for count in segment_waypoint_counts)
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("Grasped-transport segment waypoint counts must be positive and non-empty.")
    if any(
        not math.isfinite(value) or value <= 0.0 for value in (duration_per_knot_s, segment_end_settle_s, advance_dt)
    ):
        raise ValueError("Grasped-transport control-budget durations must be finite and positive.")
    waypoint_count = sum(counts)
    legacy_motion_steps_per_knot = int(math.ceil(duration_per_knot_s / advance_dt))
    legacy_settle_steps_per_knot = int(math.ceil(segment_end_settle_s / advance_dt))
    legacy_motion_steps = waypoint_count * legacy_motion_steps_per_knot
    legacy_internal_settle_steps = waypoint_count * legacy_settle_steps_per_knot
    scheduled_segment_motion_steps = tuple(int(math.ceil(count * duration_per_knot_s / advance_dt)) for count in counts)
    scheduled_motion_steps = sum(scheduled_segment_motion_steps)
    scheduled_segment_settle_steps = len(counts) * legacy_settle_steps_per_knot
    legacy_total = legacy_motion_steps + legacy_internal_settle_steps
    scheduled_total = scheduled_motion_steps + scheduled_segment_settle_steps
    return {
        "segment_waypoint_counts": counts,
        "waypoint_count": waypoint_count,
        "legacy_motion_steps_per_knot": legacy_motion_steps_per_knot,
        "legacy_settle_steps_per_knot": legacy_settle_steps_per_knot,
        "legacy_route_motion_control_step_count": legacy_motion_steps,
        "legacy_internal_knot_settle_control_step_count": legacy_internal_settle_steps,
        "legacy_route_control_step_count": legacy_total,
        "scheduled_segment_motion_control_step_counts": scheduled_segment_motion_steps,
        "scheduled_route_motion_control_step_count": scheduled_motion_steps,
        "scheduled_segment_end_settle_control_step_count": scheduled_segment_settle_steps,
        "scheduled_route_control_step_count": scheduled_total,
        "scheduled_route_control_step_reduction": legacy_total - scheduled_total,
        "scheduled_route_control_step_reduction_fraction": (legacy_total - scheduled_total) / legacy_total,
    }


def _retain_active_grasped_transport_target(
    proposed_target: torch.Tensor,
    held_target: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Select a proposed target only for active lanes and retain every failed lane."""
    if proposed_target.shape != held_target.shape or proposed_target.ndim != 2:
        raise ValueError("Proposed and held grasped-transport targets must have one identical (N, J) shape.")
    if active_mask.shape != proposed_target.shape[:1] or active_mask.dtype != torch.bool:
        raise ValueError("Grasped-transport active_mask must be Boolean with shape (N,).")
    return torch.where(active_mask[:, None], proposed_target, held_target)


def _resolve_grasped_transport_endpoint_policy(cfg: GeneratorCfg, endpoint_policy: str) -> tuple[float, bool]:
    """Return the exact endpoint position tolerance [m] and correction policy."""
    if endpoint_policy == _GRASPED_TRANSPORT_STRICT_ENDPOINT_POLICY:
        return cfg.tcp_compensation_tolerance_m, True
    if endpoint_policy == _GRASPED_TRANSPORT_RESET_ROW_ENDPOINT_POLICY:
        return cfg.grasped_transport_row_endpoint_position_tolerance_m, False
    raise ValueError(f"Unknown grasped-transport endpoint policy: {endpoint_policy!r}.")


def _grasped_transport_endpoint_position_mask(
    tcp_error_m: torch.Tensor,
    plug_error_m: torch.Tensor,
    *,
    position_tolerance_m: float,
) -> torch.Tensor:
    """Return lanes whose TCP and plug position errors satisfy one inclusive endpoint gate."""
    if tcp_error_m.ndim != 1 or plug_error_m.shape != tcp_error_m.shape:
        raise ValueError("Grasped-transport endpoint errors must share one-dimensional lane shape.")
    if not math.isfinite(position_tolerance_m) or position_tolerance_m <= 0.0:
        raise ValueError("Grasped-transport endpoint position tolerance must be finite and positive.")
    return (
        torch.isfinite(tcp_error_m)
        & torch.isfinite(plug_error_m)
        & (tcp_error_m <= position_tolerance_m)
        & (plug_error_m <= position_tolerance_m)
    )


def _grasped_transport_terminal_translation_step(
    current_tcp_position: torch.Tensor,
    target_tcp_position: torch.Tensor,
    current_plug_position: torch.Tensor,
    target_plug_position: torch.Tensor,
    *,
    position_tolerance_m: float,
    maximum_step_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a bounded plug-priority terminal translation [m] and its lane masks."""
    positions = (current_tcp_position, target_tcp_position, current_plug_position, target_plug_position)
    if any(position.ndim != 2 or position.shape[-1] != 3 for position in positions):
        raise ValueError("Terminal-correction positions must have shape (N, 3).")
    if any(position.shape != current_tcp_position.shape for position in positions):
        raise ValueError("Terminal-correction positions must share one shape.")
    if not math.isfinite(position_tolerance_m) or position_tolerance_m <= 0.0:
        raise ValueError("Terminal-correction position tolerance must be finite and positive.")
    if not math.isfinite(maximum_step_m) or maximum_step_m <= 0.0:
        raise ValueError("Terminal-correction maximum step must be finite and positive.")

    tcp_error = target_tcp_position - current_tcp_position
    plug_error = target_plug_position - current_plug_position
    tcp_error_norm = torch.linalg.vector_norm(tcp_error, dim=-1)
    plug_error_norm = torch.linalg.vector_norm(plug_error, dim=-1)
    plug_priority = plug_error_norm > position_tolerance_m
    correction_needed = plug_priority | (tcp_error_norm > position_tolerance_m)
    selected_error = torch.where(plug_priority[:, None], plug_error, tcp_error)
    selected_error_norm = torch.linalg.vector_norm(selected_error, dim=-1)
    bounded_step = selected_error * torch.clamp(
        maximum_step_m / selected_error_norm[:, None].clamp_min(1.0e-9),
        max=1.0,
    )
    return (
        torch.where(correction_needed[:, None], bounded_step, torch.zeros_like(bounded_step)),
        correction_needed,
        plug_priority,
    )


def _grasped_transport_terminal_progress_mask(
    tcp_error_before_m: torch.Tensor,
    tcp_error_after_m: torch.Tensor,
    plug_error_before_m: torch.Tensor,
    plug_error_after_m: torch.Tensor,
    *,
    correction_mask: torch.Tensor,
    plug_priority: torch.Tensor,
    position_tolerance_m: float,
    progress_epsilon_m: float,
) -> torch.Tensor:
    """Gate selected-error progress while retaining an already-valid secondary position [m]."""
    errors = (tcp_error_before_m, tcp_error_after_m, plug_error_before_m, plug_error_after_m)
    if any(error.ndim != 1 or error.shape != tcp_error_before_m.shape for error in errors):
        raise ValueError("Terminal-correction error vectors must share one-dimensional lane shape.")
    if any(
        mask.shape != tcp_error_before_m.shape or mask.dtype != torch.bool for mask in (correction_mask, plug_priority)
    ):
        raise ValueError("Terminal-correction masks must be Boolean and match the error-vector shape.")
    if not math.isfinite(position_tolerance_m) or position_tolerance_m <= 0.0:
        raise ValueError("Terminal-correction position tolerance must be finite and positive.")
    if not math.isfinite(progress_epsilon_m) or progress_epsilon_m < 0.0:
        raise ValueError("Terminal-correction progress epsilon must be finite and non-negative.")

    selected_before = torch.where(plug_priority, plug_error_before_m, tcp_error_before_m)
    selected_after = torch.where(plug_priority, plug_error_after_m, tcp_error_after_m)
    secondary_before = torch.where(plug_priority, tcp_error_before_m, plug_error_before_m)
    secondary_after = torch.where(plug_priority, tcp_error_after_m, plug_error_after_m)
    selected_progress = (selected_after <= position_tolerance_m) | (
        selected_after + progress_epsilon_m < selected_before
    )
    secondary_preserved = (secondary_before > position_tolerance_m) | (
        secondary_after <= position_tolerance_m + progress_epsilon_m
    )
    finite = (
        torch.isfinite(selected_before)
        & torch.isfinite(selected_after)
        & torch.isfinite(secondary_before)
        & torch.isfinite(secondary_after)
    )
    return ~correction_mask | (finite & selected_progress & secondary_preserved)


def _grasped_transport_transient_speed_masks(
    cable_speed: torch.Tensor,
    plug_linear_speed: torch.Tensor,
    plug_angular_speed: torch.Tensor,
    arm_speed: torch.Tensor,
    finger_speed: torch.Tensor,
    *,
    maximum_reset_cable_speed_m_s: float,
    maximum_transport_plug_linear_speed_m_s: float,
    maximum_transport_plug_angular_speed_rad_s: float,
    maximum_transport_arm_joint_speed_rad_s: float,
    maximum_transport_finger_joint_speed_m_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Separate observed transient cable speed from commanded-motion speed gates."""
    speeds = (cable_speed, plug_linear_speed, plug_angular_speed, arm_speed, finger_speed)
    if any(speed.ndim != 1 or speed.shape != cable_speed.shape for speed in speeds):
        raise ValueError("Grasped-transport speed samples must share one-dimensional lane shape.")
    transient_cable_within_reset_limit = cable_speed <= maximum_reset_cable_speed_m_s
    transport_motion_speeds_bounded = (
        (plug_linear_speed <= maximum_transport_plug_linear_speed_m_s)
        & (plug_angular_speed <= maximum_transport_plug_angular_speed_rad_s)
        & (arm_speed <= maximum_transport_arm_joint_speed_rad_s)
        & (finger_speed <= maximum_transport_finger_joint_speed_m_s)
    )
    return transient_cable_within_reset_limit, transport_motion_speeds_bounded


def _rigid_transform_task_state(
    task_q: torch.Tensor,
    task_qd: torch.Tensor,
    source_frame: torch.Tensor,
    destination_frame: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rigidly transform every task body and spatial velocity in XYZW convention."""
    if task_q.ndim != 3 or task_q.shape[-1] != 7 or task_qd.shape != (*task_q.shape[:-1], 6):
        raise ValueError("Task state must have shapes (N, B, 7)/(N, B, 6).")
    if source_frame.shape != (task_q.shape[0], 7) or destination_frame.shape != source_frame.shape:
        raise ValueError("Source and destination frames must have shape (N, 7).")
    delta_q = math_utils.quat_mul(destination_frame[:, 3:7], math_utils.quat_conjugate(source_frame[:, 3:7]))
    body_count = task_q.shape[1]
    rotation = delta_q[:, None, :].expand(-1, body_count, -1)
    relative_position = task_q[..., :3] - source_frame[:, None, :3]
    position = destination_frame[:, None, :3] + math_utils.quat_apply(rotation, relative_position)
    orientation = math_utils.quat_mul(rotation, task_q[..., 3:7])
    orientation = orientation / torch.linalg.vector_norm(orientation, dim=-1, keepdim=True).clamp_min(1.0e-9)
    linear = math_utils.quat_apply(rotation, task_qd[..., :3])
    angular = math_utils.quat_apply(rotation, task_qd[..., 3:6])
    return torch.cat((position, orientation), dim=-1), torch.cat((linear, angular), dim=-1)


class PickInsertResetDatasetGenerator:
    """Construct and reject-test physically realized reset rows."""

    def __init__(self, env: RJ45PickInsertResetToolEnv, cfg: GeneratorCfg) -> None:
        if env.num_envs != cfg.batch_size:
            raise ValueError(f"Expected {cfg.batch_size} tool worlds, got {env.num_envs}.")
        gripper_cfg = env.cfg.actions.gripper_action
        if (
            cfg.finger_open_position != PICK_INSERT_OPEN_FINGER_POSITION
            or float(gripper_cfg.neutral_position) != cfg.finger_open_position
            or float(gripper_cfg.default_position) != cfg.finger_open_position
        ):
            raise ValueError("Generator and live pick-insert gripper must share the exact 0.04 m open posture.")
        pick_insert_tool_physical_contract(env, finger_closed_target=cfg.finger_closed_target)
        self.env = env
        self.cfg = cfg
        self.device = torch.device(env.device)
        self.random = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.layout = env.rj45_runtime.layout
        if self.layout.socket_body_index is None:
            raise RuntimeError("Pick-insert generation requires a resettable socket body.")
        self.socket_index = int(self.layout.socket_body_index)
        self.plug_index = int(self.layout.plug_body_index)
        self.latch_index = int(self.layout.latch_body_index)
        self.cable_slice = self.layout.cable_body_slice
        if self.cable_slice.start is None or self.cable_slice.stop is None:
            raise RuntimeError("Pick-insert generation requires a bounded cable body slice.")
        self.cable_body_start = int(self.cable_slice.start)
        # Row generation owns one sampler-free Newton IK solver.  Certificate
        # input still requires a fresh owner so goal derivation cannot alter
        # its call history before the deterministic row stream begins.  The
        # complete stream contract invalidates certificates after any solver
        # policy change.
        self._ik_solve_call_count = 0
        self.ik = FrankaResetIK(
            env,
            seed=cfg.seed,
            seeds=_ROW_IK_SEED_COUNT,
            iterations=_ROW_IK_ITERATIONS,
            noise_std=_ROW_IK_NOISE_STD,
            sampler=_ROW_IK_SAMPLER,
        )
        # Helpers that accept an IK object must still pass through the sole
        # generator counter; this adapter owns no solver state of its own.
        self._counted_ik = SimpleNamespace(solve=self._solve_ik)
        self.home_arm_q = configured_arm_home(env).repeat(env.num_envs, 1)
        self.open_finger_q = torch.full(
            (env.num_envs, 2), cfg.finger_open_position, device=self.device, dtype=torch.float32
        )
        self.closed_finger_target = torch.full(
            (env.num_envs, 2), cfg.finger_closed_target, device=self.device, dtype=torch.float32
        )
        self.local_grasp_orientation = torch.as_tensor(
            env.cfg.plug_grasp_orientation_xyzw, device=self.device, dtype=torch.float32
        ).repeat(env.num_envs, 1)
        env.restore_default_task()
        default_q, _ = env.read_task_state()
        task_position = torch.as_tensor(env.cfg.task_translation, device=self.device, dtype=torch.float32).repeat(
            env.num_envs, 1
        )
        task_orientation = torch.as_tensor(env.cfg.task_rotation_xyzw, device=self.device, dtype=torch.float32).repeat(
            env.num_envs, 1
        )
        default_socket = default_q[:, self.socket_index]
        self.socket_local_position = math_utils.quat_apply_inverse(
            task_orientation, default_socket[:, :3] - task_position
        )
        self.socket_local_orientation = math_utils.quat_mul(
            math_utils.quat_conjugate(task_orientation), default_socket[:, 3:7]
        )
        self.reference_socket_body_pose = default_socket[0].clone()
        self.attempt_counts = [0 for _ in PICK_INSERT_RESET_PHASE_IDS]
        self.rejection_counts: dict[int, dict[str, int]] = {
            phase: defaultdict(int) for phase in PICK_INSERT_RESET_PHASE_IDS
        }
        self.accepted_oracle_metrics: dict[int, list[dict[str, Any]]] = {
            phase: [] for phase in PICK_INSERT_RESET_PHASE_IDS
        }

    def _canonical_goal_certificate_validation_kwargs(self) -> dict[str, Any]:
        """Return live contracts used for both certificate creation and loading."""
        return {
            "task_body_count": self.layout.body_count,
            "expected_task_contract": pick_insert_reset_dataset_task_contract(self.env.cfg),
            "expected_physical_contract": pick_insert_tool_physical_contract(
                self.env,
                finger_closed_target=self.cfg.finger_closed_target,
            ),
            "expected_generation_contract": _canonical_goal_generation_contract(self.cfg),
            "expected_versions": _canonical_goal_package_versions(),
            "expected_source_sha256": _canonical_goal_source_digests(),
        }

    @torch.inference_mode()
    def derive_goal_certificate(self) -> dict[str, Any]:
        """Derive and certify one production goal without advancing the row RNG stream."""
        if self.env.num_envs not in _CANONICAL_GOAL_CERTIFIER_ENV_COUNTS:
            raise RuntimeError("Canonical-goal certifier mode requires exactly one or four environments.")
        if self._ik_solve_call_count != 0:
            raise RuntimeError("Canonical-goal certification requires a freshly constructed IK stream.")
        validation = self._canonical_goal_certificate_validation_kwargs()
        row_rng_state = self.random.get_state().detach().cpu().clone().contiguous()
        canonical_goal, production_evidence = self.derive_goal()
        live_validation = self._canonical_goal_certificate_validation_kwargs()
        _require_unchanged_canonical_goal_validation_snapshot(
            validation,
            live_validation,
            operation="canonical-goal certification",
        )
        if not torch.equal(self.random.get_state().detach().cpu(), row_rng_state):
            raise RuntimeError("Canonical-goal derivation consumed the dedicated reset-row RNG stream.")
        return _build_canonical_goal_certificate(
            goal_state=canonical_goal,
            production_evidence=production_evidence,
            row_rng_state=row_rng_state,
            certifier_env_count=self.env.num_envs,
            task_body_count=validation["task_body_count"],
            task_contract=validation["expected_task_contract"],
            physical_contract=validation["expected_physical_contract"],
            generation_contract=validation["expected_generation_contract"],
            versions=validation["expected_versions"],
            source_sha256=validation["expected_source_sha256"],
        )

    def load_goal_certificate(self, path: Path) -> dict[str, Any]:
        """Load a CPU-only certificate and require every live contract to match."""
        certificate = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=True)
        return self.validate_goal_certificate(certificate)

    def validate_goal_certificate(self, certificate: Any) -> dict[str, Any]:
        """Validate a goal certificate for the selected row-screening mode."""
        validation = self._canonical_goal_certificate_validation_kwargs()
        if self.cfg.generation_mode == _GENERATION_MODE_FAST_IK:
            metadata = certificate.get("metadata") if isinstance(certificate, Mapping) else None
            source_sha256 = metadata.get("source_sha256") if isinstance(metadata, Mapping) else None
            generation_contract = metadata.get("generation_contract") if isinstance(metadata, Mapping) else None
            certificate_task_contract = metadata.get("task_contract") if isinstance(metadata, Mapping) else None
            if (
                not isinstance(source_sha256, Mapping)
                or not isinstance(generation_contract, Mapping)
                or not isinstance(certificate_task_contract, Mapping)
            ):
                raise ValueError("Fast-IK generation requires a source-, task-, and generation-bound goal certificate.")
            live_goal_contract = _canonical_goal_task_contract_projection(validation["expected_task_contract"])
            certificate_goal_contract = _canonical_goal_task_contract_projection(certificate_task_contract)
            if reset_dataset_digest(live_goal_contract) != reset_dataset_digest(certificate_goal_contract):
                raise ValueError("Fast-IK goal certificate task contract differs beyond reset-bank cardinality fields.")
            # Fast-IK does not reinterpret or regenerate the canonical goal.
            # Its own report binds the current row generator source, while the
            # embedded certificate retains the source that produced the goal.
            # Row count and diversity minima are artifact properties, so they
            # do not invalidate a physically identical certified goal.
            validation["expected_source_sha256"] = source_sha256
            validation["expected_generation_contract"] = generation_contract
            validation["expected_task_contract"] = certificate_task_contract
        return _validate_canonical_goal_certificate(certificate, **validation)

    def _solve_ik(self, *args: Any, **kwargs: Any) -> Any:
        """Solve once while tracking the certificate-input IK cursor contract."""
        self._ik_solve_call_count = getattr(self, "_ik_solve_call_count", 0) + 1
        return self.ik.solve(*args, **kwargs)

    def _drive_enabled(self) -> torch.Tensor:
        return wp.to_torch(self.env.rj45_runtime.drive_enabled).to(dtype=torch.bool)

    def _orientation_hold_enabled(self) -> torch.Tensor:
        return wp.to_torch(self.env.rj45_runtime.orientation_hold_enabled).to(dtype=torch.bool)

    def _assert_drive_disabled(self, context: str) -> None:
        translation = self._drive_enabled()
        orientation = self._orientation_hold_enabled()
        if bool(translation.any()) or bool(orientation.any()):
            raise RuntimeError(
                f"Construction drive remained enabled during {context}: "
                f"translation={torch.where(translation)[0]}, orientation={torch.where(orientation)[0]}."
            )

    def _capture_state(self, arm_target: torch.Tensor, finger_target: torch.Tensor) -> dict[str, torch.Tensor]:
        self._assert_drive_disabled("state capture")
        task_q, task_qd = self.env.read_task_state()
        task_previous_q, task_coupling_previous_q = self.env.snapshot_task_pose_history_e()
        arm_q, arm_qd, finger_q, finger_qd = self.env.read_robot_state()
        return {
            "arm_joint_position": arm_q.clone(),
            "arm_joint_target": arm_target.clone(),
            "arm_joint_velocity": arm_qd.clone(),
            "finger_joint_position": finger_q.clone(),
            "finger_joint_velocity": finger_qd.clone(),
            "finger_joint_target": finger_target.clone(),
            "task_body_pose": task_q.clone(),
            "task_body_previous_pose": task_previous_q.clone(),
            "task_body_coupling_previous_pose": task_coupling_previous_q.clone(),
            "task_body_velocity": task_qd.clone(),
        }

    def _restore_state(
        self,
        state: dict[str, torch.Tensor],
        *,
        restore_pose_history: bool = True,
    ) -> dict[str, object] | None:
        self.env.write_task_state(state["task_body_pose"], state["task_body_velocity"])
        self.env.write_robot_state(
            state["arm_joint_position"],
            state["finger_joint_position"],
            arm_target=state["arm_joint_target"],
            arm_qd=state["arm_joint_velocity"],
            finger_qd=state["finger_joint_velocity"],
            finger_target=state["finger_joint_target"],
        )
        self.env.set_drive(False)
        self.env.flush_reset_history()
        history_evidence = None
        if restore_pose_history:
            history_evidence = self.env.restore_task_pose_history_e(
                state["task_body_previous_pose"],
                state["task_body_coupling_previous_pose"],
            )
            queued = (
                bool(torch.as_tensor(history_evidence["restore_queued"]).all())
                and bool(torch.as_tensor(history_evidence["pending_at_queue"]).all())
                and bool(torch.as_tensor(history_evidence["previous_pose_queued"]).all())
                and bool(torch.as_tensor(history_evidence["coupling_previous_pose_queued"]).all())
                and history_evidence["body_order_exact"] is True
                and history_evidence["world_order_exact"] is True
                and tuple(history_evidence["body_order"]) == tuple(self.layout.body_names)
            )
            if not queued:
                raise RuntimeError("Generator failed to queue both VBD pose histories in exact task/world order.")
        self._assert_drive_disabled("cold restore")
        return history_evidence

    def _vbd_pose_history_applied_mask(self, evidence: dict[str, object]) -> torch.Tensor:
        """Return worlds whose queued histories were consumed exactly once."""
        return (
            torch.as_tensor(evidence["restore_queued"], device=self.device)
            & torch.as_tensor(evidence["pending_at_queue"], device=self.device)
            & torch.as_tensor(evidence["previous_pose_queued"], device=self.device)
            & torch.as_tensor(evidence["coupling_previous_pose_queued"], device=self.device)
            & torch.as_tensor(evidence["applied_exactly_once"], device=self.device)
            & ~torch.as_tensor(evidence["failed"], device=self.device)
            & ~torch.as_tensor(evidence["superseded"], device=self.device)
            & ~torch.as_tensor(evidence["pending_after_first_solve"], device=self.device)
            & (torch.as_tensor(evidence["application_count_delta"], device=self.device) == 1)
            & (
                torch.as_tensor(evidence["body_application_count_delta"], device=self.device)
                == torch.as_tensor(evidence["expected_body_count"], device=self.device)
            )
        )

    def _vbd_pose_history_report_evidence(self, evidence: dict[str, object]) -> dict[str, object]:
        """Return serializable deferred-restore evidence without the private ticket."""
        return {
            "vbd_pose_history_restore_queued": evidence["restore_queued"],
            "vbd_pose_history_pending_at_queue": evidence["pending_at_queue"],
            "vbd_previous_pose_queued": evidence["previous_pose_queued"],
            "vbd_coupling_previous_pose_queued": evidence["coupling_previous_pose_queued"],
            "vbd_pose_history_applied_exactly_once": evidence["applied_exactly_once"],
            "vbd_pose_history_failed": evidence["failed"],
            "vbd_pose_history_superseded": evidence["superseded"],
            "vbd_pose_history_pending_after_first_solve": evidence["pending_after_first_solve"],
            "vbd_pose_history_application_count_delta": evidence["application_count_delta"],
            "vbd_pose_history_expected_body_count": evidence["expected_body_count"],
            "vbd_pose_history_body_application_count_delta": evidence["body_application_count_delta"],
            "vbd_pose_history_generation": evidence["generation"],
            "vbd_pose_history_body_order_exact": evidence["body_order_exact"],
            "vbd_pose_history_world_order_exact": evidence["world_order_exact"],
            "vbd_pose_history_entry_name": evidence["entry_name"],
            "vbd_pose_history_body_count": evidence["body_count"],
        }

    def _task_vbd_pose_history(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return current and both VBD history poses in environment coordinates."""
        current_q, _ = self.env.read_task_state()
        previous_q, coupling_previous_q = self.env.snapshot_task_pose_history_e()
        return current_q, previous_q, coupling_previous_q

    def _cable_table_contact_metrics(
        self,
        task_q: torch.Tensor,
        cable_body_local_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Inspect exact outer contacts for one selected cable body per world.

        This is intentionally diagnostic-only host-side inspection.  The
        signed separation is reconstructed with the same Newton contact
        points, normals, and margins used by the collision validity oracle.
        """
        rows = torch.arange(self.env.num_envs, device=self.device)
        cable_position = task_q[:, self.cable_slice, :3]
        selected_position = cable_position[rows, cable_body_local_index]
        result = {
            "position": selected_position,
            "support_clearance": selected_position[:, 2] - CABLE_RADIUS,
            "contact_buffer_available": torch.zeros(
                self.env.num_envs,
                device=self.device,
                dtype=torch.bool,
            ),
            "contact_active": torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool),
            "contact_count": torch.zeros(self.env.num_envs, device=self.device, dtype=torch.long),
            "nonpositive_separation_count": torch.zeros(
                self.env.num_envs,
                device=self.device,
                dtype=torch.long,
            ),
            "minimum_separation": torch.full(
                (self.env.num_envs,),
                torch.nan,
                device=self.device,
            ),
            "maximum_separation": torch.full(
                (self.env.num_envs,),
                torch.nan,
                device=self.device,
            ),
        }
        contacts = NewtonManager.get_contacts()
        if contacts is None or contacts.rigid_contact_count is None:
            return result
        result["contact_buffer_available"][:] = True
        reported = int(wp.to_torch(contacts.rigid_contact_count)[0].item())
        count = min(reported, int(contacts.rigid_contact_max))
        if count <= 0:
            return result

        model = NewtonManager.get_model()
        state_q = wp.to_torch(NewtonManager.get_state_0().body_q)
        shape0 = wp.to_torch(contacts.rigid_contact_shape0)[:count].long()
        shape1 = wp.to_torch(contacts.rigid_contact_shape1)[:count].long()
        shape_body = wp.to_torch(model.shape_body).long()
        body0 = shape_body[shape0]
        body1 = shape_body[shape1]
        point0 = wp.to_torch(contacts.rigid_contact_point0)[:count]
        point1 = wp.to_torch(contacts.rigid_contact_point1)[:count]
        world_point0 = point0.clone()
        world_point1 = point1.clone()
        dynamic0 = body0 >= 0
        dynamic1 = body1 >= 0
        if bool(dynamic0.any()):
            world_point0[dynamic0] = state_q[body0[dynamic0], :3] + math_utils.quat_apply(
                state_q[body0[dynamic0], 3:7],
                point0[dynamic0],
            )
        if bool(dynamic1.any()):
            world_point1[dynamic1] = state_q[body1[dynamic1], :3] + math_utils.quat_apply(
                state_q[body1[dynamic1], 3:7],
                point1[dynamic1],
            )
        normal = wp.to_torch(contacts.rigid_contact_normal)[:count]
        margin0 = wp.to_torch(contacts.rigid_contact_margin0)[:count]
        margin1 = wp.to_torch(contacts.rigid_contact_margin1)[:count]
        separation = (normal * (world_point1 - world_point0)).sum(-1) - margin0 - margin1

        shape_labels = [str(label) for label in model.shape_label]
        body_labels = [str(label) for label in model.body_label]
        table_shape = torch.as_tensor(
            [label.endswith("/TableContactSurface") for label in shape_labels],
            device=self.device,
            dtype=torch.bool,
        )
        table_body = torch.as_tensor(
            [label.endswith("/TableContactSurface") for label in body_labels],
            device=self.device,
            dtype=torch.bool,
        )
        if not bool(table_shape.any()) and not bool(table_body.any()):
            raise RuntimeError("Diagnostic contact buffer does not contain TableContactSurface labels.")
        table0 = table_shape[shape0] | (dynamic0 & table_body[body0.clamp_min(0)])
        table1 = table_shape[shape1] | (dynamic1 & table_body[body1.clamp_min(0)])
        task_body_ids = self.env._task_body_ids.to(dtype=torch.long).reshape(
            self.env.num_envs,
            self.layout.body_count,
        )
        selected_task_index = cable_body_local_index + self.cable_body_start
        selected_global_body = task_body_ids[rows, selected_task_index]
        for row in range(self.env.num_envs):
            matches = ((body0 == selected_global_body[row]) & table1) | ((body1 == selected_global_body[row]) & table0)
            selected_separation = separation[matches]
            if selected_separation.numel() == 0:
                continue
            result["contact_active"][row] = True
            result["contact_count"][row] = selected_separation.numel()
            result["nonpositive_separation_count"][row] = (selected_separation <= 0.0).sum()
            result["minimum_separation"][row] = selected_separation.min()
            result["maximum_separation"][row] = selected_separation.max()
        return result

    def _vbd_pose_history_residual(self) -> dict[str, Any]:
        """Measure the persisted task pose against VBD's pre-alignment pose history."""
        current_q, previous_q, coupling_previous_q = self._task_vbd_pose_history()
        position_residual = torch.linalg.vector_norm(current_q[..., :3] - previous_q[..., :3], dim=-1)
        orientation_residual = math_utils.quat_error_magnitude(
            current_q[..., 3:7].reshape(-1, 4),
            previous_q[..., 3:7].reshape(-1, 4),
        ).reshape(self.env.num_envs, self.layout.body_count)
        history_position_delta = torch.linalg.vector_norm(
            previous_q[..., :3] - coupling_previous_q[..., :3],
            dim=-1,
        )
        history_orientation_delta = math_utils.quat_error_magnitude(
            previous_q[..., 3:7].reshape(-1, 4),
            coupling_previous_q[..., 3:7].reshape(-1, 4),
        ).reshape(self.env.num_envs, self.layout.body_count)
        history_unequal = ~(previous_q == coupling_previous_q).all(dim=-1)
        worst_position, worst_position_body = position_residual.max(dim=-1)
        worst_orientation, worst_orientation_body = orientation_residual.max(dim=-1)
        cable_orientation = orientation_residual[:, self.cable_slice]
        cable_worst_orientation, cable_worst_body = cable_orientation.max(dim=-1)
        cable_worst_body = cable_worst_body + self.cable_body_start
        solver_dt = float(NewtonManager.get_solver_dt())
        return {
            "maximum_position_residual_m": float(worst_position.max()),
            "maximum_orientation_residual_rad": float(worst_orientation.max()),
            "maximum_cable_orientation_residual_rad": float(cable_worst_orientation.max()),
            "maximum_cable_effective_angular_velocity_delta_rad_s": float((cable_worst_orientation / solver_dt).max()),
            "maximum_body_previous_vs_coupling_position_delta_m": float(history_position_delta.max()),
            "maximum_body_previous_vs_coupling_angle_delta_rad": float(history_orientation_delta.max()),
            "body_previous_vs_coupling_unequal_count_by_world": history_unequal.sum(dim=-1).detach().cpu().tolist(),
            "worst_position_body_index_by_world": worst_position_body.detach().cpu().tolist(),
            "worst_position_body_name_by_world": [self.layout.body_names[int(index)] for index in worst_position_body],
            "worst_orientation_body_index_by_world": worst_orientation_body.detach().cpu().tolist(),
            "worst_orientation_body_name_by_world": [
                self.layout.body_names[int(index)] for index in worst_orientation_body
            ],
            "worst_cable_orientation_body_index_by_world": cable_worst_body.detach().cpu().tolist(),
            "worst_cable_orientation_body_name_by_world": [
                self.layout.body_names[int(index)] for index in cable_worst_body
            ],
            "solver_dt_s": solver_dt,
        }

    def _signed_latch_state(
        self,
        task_q: torch.Tensor,
        task_qd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return angle/rate about the authored plug-local ``-X`` hinge axis."""
        plug_quat = task_q[:, self.plug_index, 3:7]
        latch_quat = task_q[:, self.latch_index, 3:7]
        relative_quat = math_utils.quat_unique(math_utils.quat_mul(math_utils.quat_conjugate(plug_quat), latch_quat))
        relative_axis_angle = math_utils.axis_angle_from_quat(relative_quat)
        relative_angular_velocity_w = task_qd[:, self.latch_index, 3:6] - task_qd[:, self.plug_index, 3:6]
        relative_angular_velocity_plug = math_utils.quat_apply_inverse(
            plug_quat,
            relative_angular_velocity_w,
        )
        return -relative_axis_angle[:, 0], -relative_angular_velocity_plug[:, 0]

    def _plug_tcp_relative_pose(self, task_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the live TCP transform in the plug frame."""
        plug = task_q[:, self.plug_index]
        tcp = self.env.tcp_pose_e()
        relative_position = math_utils.quat_apply_inverse(plug[:, 3:7], tcp[:, :3] - plug[:, :3])
        relative_orientation = math_utils.quat_unique(
            math_utils.quat_mul(math_utils.quat_conjugate(plug[:, 3:7]), tcp[:, 3:7])
        )
        return relative_position, relative_orientation

    def _coupling_effective_task_velocity(
        self,
        task_q: torch.Tensor,
        task_qd: torch.Tensor,
        vbd_previous_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add the pose delta that coupled VBD observes before reset rebaselining."""
        solver_dt = float(NewtonManager.get_solver_dt())
        position_delta_velocity = (task_q[..., :3] - vbd_previous_q[..., :3]) / solver_dt
        orientation_delta = math_utils.quat_unique(
            math_utils.quat_mul(
                task_q[..., 3:7].reshape(-1, 4),
                math_utils.quat_conjugate(vbd_previous_q[..., 3:7].reshape(-1, 4)),
            )
        )
        angular_delta_velocity = (
            math_utils.axis_angle_from_quat(orientation_delta).reshape(
                self.env.num_envs,
                self.layout.body_count,
                3,
            )
            / solver_dt
        )
        coupling_delta = torch.cat((position_delta_velocity, angular_delta_velocity), dim=-1)
        task_body_ids = self.env._task_body_ids.to(dtype=torch.long)
        body_flags = wp.to_torch(NewtonManager.get_model().body_flags)[task_body_ids].reshape(
            self.env.num_envs,
            self.layout.body_count,
        )
        dynamic = (body_flags & int(BodyFlags.KINEMATIC)) == 0
        coupling_delta = torch.where(dynamic[..., None], coupling_delta, torch.zeros_like(coupling_delta))
        return task_qd + coupling_delta, coupling_delta

    def _run_reset_abcd_discriminator(
        self,
        candidate: dict[str, torch.Tensor],
        authored_seat_target_w: torch.Tensor,
        authored_plug_orientation: torch.Tensor,
    ) -> dict[str, Any]:
        """Compare continuous and four cold materializations of one exact candidate."""
        live_q, live_previous_q, _ = self._task_vbd_pose_history()
        candidate_q = candidate["task_body_pose"]
        candidate_pose_error = torch.linalg.vector_norm(live_q[..., :3] - candidate_q[..., :3], dim=-1).amax(dim=-1)
        candidate_angle_error = (
            math_utils.quat_error_magnitude(
                live_q[..., 3:7].reshape(-1, 4),
                candidate_q[..., 3:7].reshape(-1, 4),
            )
            .reshape(self.env.num_envs, self.layout.body_count)
            .amax(dim=-1)
        )
        if bool((candidate_pose_error > 1.0e-7).any()) or bool((candidate_angle_error > 1.0e-6).any()):
            raise RuntimeError(
                "A/B/C/D/E candidate capture does not match the continuous live pose: "
                f"position={candidate_pose_error.tolist()}, orientation={candidate_angle_error.tolist()}."
            )

        effective_qd, coupling_delta = self._coupling_effective_task_velocity(
            candidate_q,
            candidate["task_body_velocity"],
            live_previous_q,
        )
        task_body_ids = self.env._task_body_ids.to(dtype=torch.long)
        body_flags = wp.to_torch(NewtonManager.get_model().body_flags)[task_body_ids].reshape(
            self.env.num_envs,
            self.layout.body_count,
        )
        dynamic = (body_flags & int(BodyFlags.KINEMATIC)) == 0
        continuous_expected_input_q = torch.where(dynamic[..., None], live_previous_q, candidate_q)
        restored_effective_qd, _ = self._coupling_effective_task_velocity(
            candidate_q,
            candidate["task_body_velocity"],
            candidate["task_body_previous_pose"],
        )
        restored_expected_input_q = torch.where(
            dynamic[..., None],
            candidate["task_body_previous_pose"],
            candidate_q,
        )
        zero_qd = torch.zeros_like(candidate["task_body_velocity"])
        branch_states = {
            "A_continuous_history": candidate,
            "B_cold_raw_qd": candidate,
            "C_cold_zero_task_qd": {
                **candidate,
                "task_body_velocity": zero_qd,
            },
            "D_cold_coupling_effective_qd": {
                **candidate,
                "task_body_velocity": effective_qd,
            },
            "E_cold_restored_pose_history": candidate,
        }
        if self.cfg.diagnostic_reset_e_only:
            branch_states = {
                name: branch_states[name]
                for name in (
                    "A_continuous_history",
                    "E_cold_restored_pose_history",
                )
            }
        reference_q = candidate_q.clone()
        reference_q[:, self.plug_index, :3] = authored_seat_target_w - self.env.env_origins
        reference_q[:, self.plug_index, 3:7] = authored_plug_orientation
        cable03_index = self.cable_body_start + 3
        duration_s = self.cfg.goal_cold_equilibrium_relax_s
        branch_results: dict[str, Any] = {}
        first_post_step_states: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        def optional_times(values: torch.Tensor) -> list[float | None]:
            return [None if value < 0.0 else float(value) for value in values.detach().cpu().tolist()]

        for branch_index, (name, branch_state) in enumerate(branch_states.items()):
            branch_history_evidence = None
            if branch_index > 0:
                branch_history_evidence = self._restore_state(
                    branch_state,
                    restore_pose_history=name == "E_cold_restored_pose_history",
                )
            self._assert_drive_disabled(f"A/B/C/D/E branch {name}")
            self.env.set_robot_targets(
                candidate["arm_joint_target"],
                candidate["finger_joint_target"],
            )
            start_q, start_qd = self.env.read_task_state()
            start_cable_position = start_q[:, self.cable_slice, :3].clone()
            start_cable_qd = start_qd[:, self.cable_slice].clone()
            previous_cable_position = start_cable_position.clone()
            start_relative_position, start_relative_orientation = self._plug_tcp_relative_pose(start_q)
            first_exact_failure = torch.full((self.env.num_envs,), -1.0, device=self.device)
            first_plug_speed_failure = torch.full_like(first_exact_failure, -1.0)
            first_cable_speed_failure = torch.full_like(first_exact_failure, -1.0)
            first_latch_failure = torch.full_like(first_exact_failure, -1.0)
            all_finite = torch.ones(self.env.num_envs, device=self.device, dtype=torch.bool)
            all_collision_free = all_finite.clone()
            all_bilateral = all_finite.clone()
            all_drives_disabled = all_finite.clone()
            maximum_cable_speed = torch.full(
                (self.env.num_envs,),
                -torch.inf,
                device=self.device,
            )
            maximum_cable_speed_time = torch.zeros(self.env.num_envs, device=self.device)
            maximum_cable_body_local_index = torch.zeros(
                self.env.num_envs,
                device=self.device,
                dtype=torch.long,
            )
            maximum_cable_body_qd = torch.zeros(
                self.env.num_envs,
                6,
                device=self.device,
            )
            maximum_cable_body_step_displacement = torch.zeros(
                self.env.num_envs,
                3,
                device=self.device,
            )
            maximum_cable_body_displacement_from_start = torch.zeros_like(maximum_cable_body_step_displacement)
            maximum_cable_body_position = torch.zeros_like(maximum_cable_body_step_displacement)
            maximum_cable_body_support_clearance = torch.full_like(maximum_cable_speed, torch.nan)
            maximum_cable_body_table_contact_buffer_available = torch.zeros(
                self.env.num_envs,
                device=self.device,
                dtype=torch.bool,
            )
            maximum_cable_body_table_contact_active = torch.zeros(
                self.env.num_envs,
                device=self.device,
                dtype=torch.bool,
            )
            maximum_cable_body_table_contact_count = torch.zeros(
                self.env.num_envs,
                device=self.device,
                dtype=torch.long,
            )
            maximum_cable_body_table_nonpositive_separation_count = torch.zeros_like(
                maximum_cable_body_table_contact_count
            )
            maximum_cable_body_table_minimum_separation = torch.full_like(maximum_cable_speed, torch.nan)
            maximum_cable_body_table_maximum_separation = torch.full_like(maximum_cable_speed, torch.nan)
            maximum_cable_body_previous_span_axis = torch.full_like(
                maximum_cable_body_step_displacement,
                torch.nan,
            )
            maximum_cable_body_previous_span_relative_linear_velocity = torch.full_like(
                maximum_cable_body_step_displacement,
                torch.nan,
            )
            maximum_cable_body_previous_span_axial_velocity = torch.full_like(maximum_cable_speed, torch.nan)
            maximum_cable_body_previous_span_shear_velocity = torch.full_like(maximum_cable_speed, torch.nan)
            maximum_cable_body_next_span_axis = torch.full_like(
                maximum_cable_body_step_displacement,
                torch.nan,
            )
            maximum_cable_body_next_span_relative_linear_velocity = torch.full_like(
                maximum_cable_body_step_displacement,
                torch.nan,
            )
            maximum_cable_body_next_span_axial_velocity = torch.full_like(maximum_cable_speed, torch.nan)
            maximum_cable_body_next_span_shear_velocity = torch.full_like(maximum_cable_speed, torch.nan)
            maximum_plug_speed = torch.zeros_like(maximum_cable_speed)
            maximum_latch_angle = torch.full_like(maximum_cable_speed, -torch.inf)
            minimum_latch_angle = torch.full_like(maximum_cable_speed, torch.inf)
            maximum_absolute_latch_rate = torch.zeros_like(maximum_cable_speed)
            maximum_plug_tcp_position_drift = torch.zeros_like(maximum_cable_speed)
            maximum_plug_tcp_angle_drift = torch.zeros_like(maximum_cable_speed)
            maximum_cable03_history_residual = torch.zeros_like(maximum_cable_speed)
            any_overflow = False
            invalid_pairs: list[str] = []
            sample_count = 0
            initial_metrics: dict[str, Any] = {}
            final_metrics: dict[str, Any] = {}
            first_post_step_q: torch.Tensor | None = None
            first_post_step_qd: torch.Tensor | None = None

            def sample(time_s: float, *, sample_contacts: bool) -> None:
                nonlocal any_overflow, sample_count, initial_metrics, final_metrics
                nonlocal first_post_step_q, first_post_step_qd
                task_q, task_qd = self.env.read_task_state()
                exact = exact_success_from_state(
                    self.env,
                    task_q,
                    task_qd,
                    reference_q,
                    plug_body_index=self.plug_index,
                    latch_body_index=self.latch_index,
                )
                cable_qd = task_qd[:, self.cable_slice]
                cable_speed_by_body = torch.linalg.vector_norm(cable_qd[..., :3], dim=-1)
                cable_speed, cable_body_local_index = cable_speed_by_body.max(dim=-1)
                cable_position = task_q[:, self.cable_slice, :3]
                rows = torch.arange(self.env.num_envs, device=self.device)
                peak_qd = cable_qd[rows, cable_body_local_index]
                peak_step_displacement = (cable_position - previous_cable_position)[rows, cable_body_local_index]
                peak_displacement_from_start = (cable_position - start_cable_position)[rows, cable_body_local_index]
                previous_body_local_index = (cable_body_local_index - 1).clamp_min(0)
                next_body_local_index = (cable_body_local_index + 1).clamp_max(cable_qd.shape[1] - 1)
                previous_span = (
                    cable_position[rows, cable_body_local_index]
                    - cable_position[
                        rows,
                        previous_body_local_index,
                    ]
                )
                next_span = (
                    cable_position[rows, next_body_local_index]
                    - cable_position[
                        rows,
                        cable_body_local_index,
                    ]
                )
                previous_span_axis = previous_span / torch.linalg.vector_norm(
                    previous_span,
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1.0e-12)
                next_span_axis = next_span / torch.linalg.vector_norm(
                    next_span,
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1.0e-12)
                previous_relative_velocity = (
                    cable_qd[rows, cable_body_local_index, :3]
                    - cable_qd[
                        rows,
                        previous_body_local_index,
                        :3,
                    ]
                )
                next_relative_velocity = (
                    cable_qd[rows, next_body_local_index, :3]
                    - cable_qd[
                        rows,
                        cable_body_local_index,
                        :3,
                    ]
                )
                previous_axial_velocity = (previous_relative_velocity * previous_span_axis).sum(dim=-1)
                next_axial_velocity = (next_relative_velocity * next_span_axis).sum(dim=-1)
                previous_shear_velocity = torch.linalg.vector_norm(
                    previous_relative_velocity - previous_axial_velocity[:, None] * previous_span_axis,
                    dim=-1,
                )
                next_shear_velocity = torch.linalg.vector_norm(
                    next_relative_velocity - next_axial_velocity[:, None] * next_span_axis,
                    dim=-1,
                )
                has_previous_span = cable_body_local_index > 0
                has_next_span = cable_body_local_index + 1 < cable_qd.shape[1]
                nan_vector = torch.full_like(previous_span_axis, torch.nan)
                nan_scalar = torch.full_like(previous_axial_velocity, torch.nan)
                previous_span_axis = torch.where(has_previous_span[:, None], previous_span_axis, nan_vector)
                previous_relative_velocity = torch.where(
                    has_previous_span[:, None],
                    previous_relative_velocity,
                    nan_vector,
                )
                previous_axial_velocity = torch.where(
                    has_previous_span,
                    previous_axial_velocity,
                    nan_scalar,
                )
                previous_shear_velocity = torch.where(
                    has_previous_span,
                    previous_shear_velocity,
                    nan_scalar,
                )
                next_span_axis = torch.where(has_next_span[:, None], next_span_axis, nan_vector)
                next_relative_velocity = torch.where(
                    has_next_span[:, None],
                    next_relative_velocity,
                    nan_vector,
                )
                next_axial_velocity = torch.where(has_next_span, next_axial_velocity, nan_scalar)
                next_shear_velocity = torch.where(has_next_span, next_shear_velocity, nan_scalar)
                new_peak = cable_speed > maximum_cable_speed
                peak_contact = (
                    self._cable_table_contact_metrics(task_q, cable_body_local_index) if bool(new_peak.any()) else None
                )
                maximum_cable_speed_time.copy_(
                    torch.where(
                        new_peak,
                        torch.full_like(maximum_cable_speed_time, time_s),
                        maximum_cable_speed_time,
                    )
                )
                maximum_cable_body_local_index.copy_(
                    torch.where(new_peak, cable_body_local_index, maximum_cable_body_local_index)
                )
                maximum_cable_body_qd.copy_(torch.where(new_peak[:, None], peak_qd, maximum_cable_body_qd))
                maximum_cable_body_step_displacement.copy_(
                    torch.where(
                        new_peak[:, None],
                        peak_step_displacement,
                        maximum_cable_body_step_displacement,
                    )
                )
                maximum_cable_body_displacement_from_start.copy_(
                    torch.where(
                        new_peak[:, None],
                        peak_displacement_from_start,
                        maximum_cable_body_displacement_from_start,
                    )
                )
                maximum_cable_body_position.copy_(
                    torch.where(
                        new_peak[:, None],
                        cable_position[rows, cable_body_local_index],
                        maximum_cable_body_position,
                    )
                )
                if peak_contact is not None:
                    maximum_cable_body_support_clearance.copy_(
                        torch.where(
                            new_peak,
                            peak_contact["support_clearance"],
                            maximum_cable_body_support_clearance,
                        )
                    )
                    maximum_cable_body_table_contact_buffer_available.copy_(
                        torch.where(
                            new_peak,
                            peak_contact["contact_buffer_available"],
                            maximum_cable_body_table_contact_buffer_available,
                        )
                    )
                    maximum_cable_body_table_contact_active.copy_(
                        torch.where(
                            new_peak,
                            peak_contact["contact_active"],
                            maximum_cable_body_table_contact_active,
                        )
                    )
                    maximum_cable_body_table_contact_count.copy_(
                        torch.where(
                            new_peak,
                            peak_contact["contact_count"],
                            maximum_cable_body_table_contact_count,
                        )
                    )
                    maximum_cable_body_table_nonpositive_separation_count.copy_(
                        torch.where(
                            new_peak,
                            peak_contact["nonpositive_separation_count"],
                            maximum_cable_body_table_nonpositive_separation_count,
                        )
                    )
                    maximum_cable_body_table_minimum_separation.copy_(
                        torch.where(
                            new_peak,
                            peak_contact["minimum_separation"],
                            maximum_cable_body_table_minimum_separation,
                        )
                    )
                    maximum_cable_body_table_maximum_separation.copy_(
                        torch.where(
                            new_peak,
                            peak_contact["maximum_separation"],
                            maximum_cable_body_table_maximum_separation,
                        )
                    )
                maximum_cable_body_previous_span_axis.copy_(
                    torch.where(
                        new_peak[:, None],
                        previous_span_axis,
                        maximum_cable_body_previous_span_axis,
                    )
                )
                maximum_cable_body_previous_span_relative_linear_velocity.copy_(
                    torch.where(
                        new_peak[:, None],
                        previous_relative_velocity,
                        maximum_cable_body_previous_span_relative_linear_velocity,
                    )
                )
                maximum_cable_body_previous_span_axial_velocity.copy_(
                    torch.where(
                        new_peak,
                        previous_axial_velocity,
                        maximum_cable_body_previous_span_axial_velocity,
                    )
                )
                maximum_cable_body_previous_span_shear_velocity.copy_(
                    torch.where(
                        new_peak,
                        previous_shear_velocity,
                        maximum_cable_body_previous_span_shear_velocity,
                    )
                )
                maximum_cable_body_next_span_axis.copy_(
                    torch.where(
                        new_peak[:, None],
                        next_span_axis,
                        maximum_cable_body_next_span_axis,
                    )
                )
                maximum_cable_body_next_span_relative_linear_velocity.copy_(
                    torch.where(
                        new_peak[:, None],
                        next_relative_velocity,
                        maximum_cable_body_next_span_relative_linear_velocity,
                    )
                )
                maximum_cable_body_next_span_axial_velocity.copy_(
                    torch.where(
                        new_peak,
                        next_axial_velocity,
                        maximum_cable_body_next_span_axial_velocity,
                    )
                )
                maximum_cable_body_next_span_shear_velocity.copy_(
                    torch.where(
                        new_peak,
                        next_shear_velocity,
                        maximum_cable_body_next_span_shear_velocity,
                    )
                )
                latch_angle, latch_rate = self._signed_latch_state(task_q, task_qd)
                relative_position, relative_orientation = self._plug_tcp_relative_pose(task_q)
                plug_tcp_position_drift = torch.linalg.vector_norm(relative_position - start_relative_position, dim=-1)
                plug_tcp_angle_drift = math_utils.quat_error_magnitude(
                    relative_orientation,
                    start_relative_orientation,
                )
                history_q, history_previous_q, history_coupling_previous_q = self._task_vbd_pose_history()
                cable03_history_residual = math_utils.quat_error_magnitude(
                    history_q[:, cable03_index, 3:7],
                    history_previous_q[:, cable03_index, 3:7],
                )
                cable03_history_pair_delta = math_utils.quat_error_magnitude(
                    history_previous_q[:, cable03_index, 3:7],
                    history_coupling_previous_q[:, cable03_index, 3:7],
                )
                current_metrics = {
                    "time_s": time_s,
                    "vbd_pose_history_semantics": (
                        "continuous-active"
                        if name == "A_continuous_history"
                        else "cold-restore-queued-pending"
                        if name == "E_cold_restored_pose_history" and not sample_contacts
                        else "cold-restore-applied"
                        if name == "E_cold_restored_pose_history"
                        else "cold-rebaseline-pending"
                        if not sample_contacts
                        else "cold-post-rebaseline-active"
                    ),
                    "signed_latch_angle_rad": latch_angle.detach().cpu().tolist(),
                    "signed_latch_rate_rad_s": latch_rate.detach().cpu().tolist(),
                    "plug_spatial_speed_m_s": exact.plug_spatial_speed.detach().cpu().tolist(),
                    "cable_speed_m_s": cable_speed.detach().cpu().tolist(),
                    "plug_tcp_position_drift_m": plug_tcp_position_drift.detach().cpu().tolist(),
                    "plug_tcp_angle_drift_rad": plug_tcp_angle_drift.detach().cpu().tolist(),
                    "cable03_q_vs_body_q_prev_angle_rad": cable03_history_residual.detach().cpu().tolist(),
                    "cable03_body_q_prev_vs_coupling_snapshot_angle_rad": (
                        cable03_history_pair_delta.detach().cpu().tolist()
                    ),
                    "exact_success": exact.mask.detach().cpu().tolist(),
                }
                if sample_count == 0:
                    initial_metrics = current_metrics
                if sample_contacts and first_post_step_q is None:
                    first_post_step_q = task_q.clone()
                    first_post_step_qd = task_qd.clone()
                final_metrics = current_metrics
                sample_count += 1
                time_value = torch.full_like(first_exact_failure, time_s)
                first_exact_failure.copy_(
                    torch.where((first_exact_failure < 0.0) & ~exact.mask, time_value, first_exact_failure)
                )
                first_plug_speed_failure.copy_(
                    torch.where(
                        (first_plug_speed_failure < 0.0)
                        & (exact.plug_spatial_speed > self.env.cfg.success_max_plug_speed),
                        time_value,
                        first_plug_speed_failure,
                    )
                )
                first_cable_speed_failure.copy_(
                    torch.where(
                        (first_cable_speed_failure < 0.0) & (cable_speed > self.cfg.maximum_goal_cable_speed_m_s),
                        time_value,
                        first_cable_speed_failure,
                    )
                )
                first_latch_failure.copy_(
                    torch.where(
                        (first_latch_failure < 0.0)
                        & (torch.abs(latch_angle) > PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD),
                        time_value,
                        first_latch_failure,
                    )
                )
                all_finite.logical_and_(task_state_is_finite_and_normalized(task_q, task_qd))
                maximum_cable_speed.copy_(torch.maximum(maximum_cable_speed, cable_speed))
                maximum_plug_speed.copy_(torch.maximum(maximum_plug_speed, exact.plug_spatial_speed))
                maximum_latch_angle.copy_(torch.maximum(maximum_latch_angle, latch_angle))
                minimum_latch_angle.copy_(torch.minimum(minimum_latch_angle, latch_angle))
                maximum_absolute_latch_rate.copy_(torch.maximum(maximum_absolute_latch_rate, torch.abs(latch_rate)))
                maximum_plug_tcp_position_drift.copy_(
                    torch.maximum(maximum_plug_tcp_position_drift, plug_tcp_position_drift)
                )
                maximum_plug_tcp_angle_drift.copy_(torch.maximum(maximum_plug_tcp_angle_drift, plug_tcp_angle_drift))
                if (
                    name
                    in (
                        "A_continuous_history",
                        "E_cold_restored_pose_history",
                    )
                    or sample_contacts
                ):
                    maximum_cable03_history_residual.copy_(
                        torch.maximum(maximum_cable03_history_residual, cable03_history_residual)
                    )
                if sample_contacts:
                    collision = collision_metrics(self.env, require_bilateral_grasp=False)
                    if collision.contact_overflow:
                        raise RuntimeError("Global contact-buffer overflow during reset-history diagnostic replay.")
                    grasp = grasp_metrics(self.env, candidate["finger_joint_target"], retaining_grasp=True)
                    bilateral = _runtime_bilateral_grasp_proxy_contact_mask(
                        self.env,
                        collision.left_grasp_contact_count,
                        collision.right_grasp_contact_count,
                    )
                    all_collision_free.logical_and_(collision.valid)
                    all_bilateral.logical_and_(grasp.valid & bilateral)
                    all_drives_disabled.logical_and_(~self._drive_enabled() & ~self._orientation_hold_enabled())
                    any_overflow |= collision.contact_overflow
                    for pair in collision.invalid_contact_pairs:
                        if pair not in invalid_pairs and len(invalid_pairs) < 64:
                            invalid_pairs.append(pair)
                previous_cable_position.copy_(cable_position)

            sample(0.0, sample_contacts=False)

            def update(_step: int, _steps: int, _progress: float) -> None:
                self.env.set_robot_targets(
                    candidate["arm_joint_target"],
                    candidate["finger_joint_target"],
                )

            def post_step(step: int, _steps: int, _progress: float) -> None:
                sample((step + 1) * self.env.advance_dt, sample_contacts=True)

            steps = self.env.advance(duration_s, update, post_step=post_step)
            final_q, final_qd = self.env.read_task_state()
            if first_post_step_q is None or first_post_step_qd is None:
                raise RuntimeError(f"A/B/C/D/E branch {name} did not execute a real physics step.")
            first_post_step_states[name] = (first_post_step_q, first_post_step_qd)
            final_arm_q, _, _, _ = self.env.read_robot_state()
            target_drift = torch.abs(candidate["arm_joint_target"] - branch_state["arm_joint_target"]).amax(dim=-1)
            history_applied = (
                bool(self._vbd_pose_history_applied_mask(branch_history_evidence).all())
                if branch_history_evidence is not None
                else False
            )
            rows = torch.arange(self.env.num_envs, device=self.device)
            peak_body_task_index = maximum_cable_body_local_index + self.cable_body_start
            peak_body_start_qd = start_cable_qd[rows, maximum_cable_body_local_index]
            peak_body_final_qd = final_qd[:, self.cable_slice][rows, maximum_cable_body_local_index]
            peak_previous_body_task_index = peak_body_task_index - 1
            peak_next_body_task_index = peak_body_task_index + 1
            peak_has_previous_body = maximum_cable_body_local_index > 0
            peak_has_next_body = maximum_cable_body_local_index + 1 < self.cable_slice.stop - self.cable_slice.start
            history_ticket = None
            if branch_history_evidence is not None:
                history_ticket = {
                    "restore_queued": bool(torch.as_tensor(branch_history_evidence["restore_queued"]).all()),
                    "pending_at_queue": bool(torch.as_tensor(branch_history_evidence["pending_at_queue"]).all()),
                    "previous_pose_queued": bool(
                        torch.as_tensor(branch_history_evidence["previous_pose_queued"]).all()
                    ),
                    "coupling_previous_pose_queued": bool(
                        torch.as_tensor(branch_history_evidence["coupling_previous_pose_queued"]).all()
                    ),
                    "applied_exactly_once": history_applied,
                    "failed": bool(torch.as_tensor(branch_history_evidence["failed"]).any()),
                    "superseded": bool(torch.as_tensor(branch_history_evidence["superseded"]).any()),
                    "pending_after_first_solve": bool(
                        torch.as_tensor(branch_history_evidence["pending_after_first_solve"]).any()
                    ),
                    "application_count_delta": torch.as_tensor(branch_history_evidence["application_count_delta"])
                    .detach()
                    .cpu()
                    .tolist(),
                    "expected_body_count": torch.as_tensor(branch_history_evidence["expected_body_count"])
                    .detach()
                    .cpu()
                    .tolist(),
                    "body_application_count_delta": torch.as_tensor(
                        branch_history_evidence["body_application_count_delta"]
                    )
                    .detach()
                    .cpu()
                    .tolist(),
                    "generation": int(branch_history_evidence["generation"]),
                }
            branch_results[name] = {
                "history_mode": (
                    "continuous"
                    if name == "A_continuous_history"
                    else "cold-reset-with-both-captured-vbd-pose-histories-restored"
                    if name == "E_cold_restored_pose_history"
                    else "cold-reset-and-cleared"
                ),
                "task_velocity_mode": (
                    "captured-raw"
                    if name
                    in (
                        "A_continuous_history",
                        "B_cold_raw_qd",
                        "E_cold_restored_pose_history",
                    )
                    else "zero"
                    if name == "C_cold_zero_task_qd"
                    else "captured-raw-plus-q-vs-body-q-prev-over-dt"
                ),
                "vbd_pose_history_applied_exactly_once_after_cold_reset": history_applied,
                "duration_s": duration_s,
                "steps": steps,
                "initial": initial_metrics,
                "final": final_metrics,
                "first_exact_failure_time_s": optional_times(first_exact_failure),
                "first_plug_speed_failure_time_s": optional_times(first_plug_speed_failure),
                "first_cable_speed_failure_time_s": optional_times(first_cable_speed_failure),
                "first_latch_failure_time_s": optional_times(first_latch_failure),
                "maximum_cable_speed_m_s": maximum_cable_speed.detach().cpu().tolist(),
                "maximum_cable_speed_time_s": maximum_cable_speed_time.detach().cpu().tolist(),
                "maximum_cable_speed_body_index": peak_body_task_index.detach().cpu().tolist(),
                "maximum_cable_speed_body_name": [self.layout.body_names[int(index)] for index in peak_body_task_index],
                "maximum_cable_speed_body_qd": maximum_cable_body_qd.detach().cpu().tolist(),
                "maximum_cable_speed_body_start_qd": peak_body_start_qd.detach().cpu().tolist(),
                "maximum_cable_speed_body_final_qd": peak_body_final_qd.detach().cpu().tolist(),
                "maximum_cable_speed_body_position_e_m": maximum_cable_body_position.detach().cpu().tolist(),
                "maximum_cable_speed_body_support_clearance_m": (
                    maximum_cable_body_support_clearance.detach().cpu().tolist()
                ),
                "maximum_cable_speed_body_table_contact_buffer_available": (
                    maximum_cable_body_table_contact_buffer_available.detach().cpu().tolist()
                ),
                "maximum_cable_speed_body_table_contact_active": (
                    maximum_cable_body_table_contact_active.detach().cpu().tolist()
                ),
                "maximum_cable_speed_body_table_contact_count": (
                    maximum_cable_body_table_contact_count.detach().cpu().tolist()
                ),
                "maximum_cable_speed_body_table_nonpositive_separation_count": (
                    maximum_cable_body_table_nonpositive_separation_count.detach().cpu().tolist()
                ),
                "maximum_cable_speed_body_table_minimum_separation_m": (
                    maximum_cable_body_table_minimum_separation.detach().cpu().tolist()
                ),
                "maximum_cable_speed_body_table_maximum_separation_m": (
                    maximum_cable_body_table_maximum_separation.detach().cpu().tolist()
                ),
                "maximum_cable_speed_previous_body_index": [
                    int(index) if bool(valid) else None
                    for index, valid in zip(peak_previous_body_task_index, peak_has_previous_body, strict=True)
                ],
                "maximum_cable_speed_previous_body_name": [
                    self.layout.body_names[int(index)] if bool(valid) else None
                    for index, valid in zip(peak_previous_body_task_index, peak_has_previous_body, strict=True)
                ],
                "maximum_cable_speed_previous_span_axis_e": (
                    maximum_cable_body_previous_span_axis.detach().cpu().tolist()
                ),
                "maximum_cable_speed_previous_span_relative_linear_velocity_e_m_s": (
                    maximum_cable_body_previous_span_relative_linear_velocity.detach().cpu().tolist()
                ),
                "maximum_cable_speed_previous_span_axial_velocity_m_s": (
                    maximum_cable_body_previous_span_axial_velocity.detach().cpu().tolist()
                ),
                "maximum_cable_speed_previous_span_shear_velocity_m_s": (
                    maximum_cable_body_previous_span_shear_velocity.detach().cpu().tolist()
                ),
                "maximum_cable_speed_next_body_index": [
                    int(index) if bool(valid) else None
                    for index, valid in zip(peak_next_body_task_index, peak_has_next_body, strict=True)
                ],
                "maximum_cable_speed_next_body_name": [
                    self.layout.body_names[int(index)] if bool(valid) else None
                    for index, valid in zip(peak_next_body_task_index, peak_has_next_body, strict=True)
                ],
                "maximum_cable_speed_next_span_axis_e": maximum_cable_body_next_span_axis.detach().cpu().tolist(),
                "maximum_cable_speed_next_span_relative_linear_velocity_e_m_s": (
                    maximum_cable_body_next_span_relative_linear_velocity.detach().cpu().tolist()
                ),
                "maximum_cable_speed_next_span_axial_velocity_m_s": (
                    maximum_cable_body_next_span_axial_velocity.detach().cpu().tolist()
                ),
                "maximum_cable_speed_next_span_shear_velocity_m_s": (
                    maximum_cable_body_next_span_shear_velocity.detach().cpu().tolist()
                ),
                "maximum_cable_speed_body_step_displacement_m": (
                    maximum_cable_body_step_displacement.detach().cpu().tolist()
                ),
                "maximum_cable_speed_body_displacement_from_start_m": (
                    maximum_cable_body_displacement_from_start.detach().cpu().tolist()
                ),
                "maximum_plug_spatial_speed_m_s": maximum_plug_speed.detach().cpu().tolist(),
                "minimum_signed_latch_angle_rad": minimum_latch_angle.detach().cpu().tolist(),
                "maximum_signed_latch_angle_rad": maximum_latch_angle.detach().cpu().tolist(),
                "maximum_absolute_latch_rate_rad_s": maximum_absolute_latch_rate.detach().cpu().tolist(),
                "maximum_plug_tcp_position_drift_m": maximum_plug_tcp_position_drift.detach().cpu().tolist(),
                "maximum_plug_tcp_angle_drift_rad": maximum_plug_tcp_angle_drift.detach().cpu().tolist(),
                "maximum_cable03_q_vs_body_q_prev_angle_rad": maximum_cable03_history_residual.detach().cpu().tolist(),
                "all_post_step_finite": all_finite.detach().cpu().tolist(),
                "all_post_step_collision_free": all_collision_free.detach().cpu().tolist(),
                "all_post_step_bilateral_grasp": all_bilateral.detach().cpu().tolist(),
                "all_post_step_drives_disabled": all_drives_disabled.detach().cpu().tolist(),
                "any_contact_overflow": any_overflow,
                "invalid_contact_pairs": invalid_pairs,
                "absolute_arm_target_bitwise_unchanged": torch.equal(
                    candidate["arm_joint_target"],
                    branch_state["arm_joint_target"],
                ),
                "maximum_absolute_arm_target_difference_rad": target_drift.detach().cpu().tolist(),
                "final_arm_tracking_error_rad": torch.abs(final_arm_q - candidate["arm_joint_target"])
                .detach()
                .cpu()
                .tolist(),
                "final_task_state_finite": task_state_is_finite_and_normalized(final_q, final_qd)
                .detach()
                .cpu()
                .tolist(),
                "vbd_pose_history_ticket": history_ticket,
            }
            print(f"[PICK-INSERT RESET A/B/C/D/E {name}] {branch_results[name]}", flush=True)

        first_solve_equivalence: dict[str, Any] = {}
        if (
            "A_continuous_history" in first_post_step_states
            and "E_cold_restored_pose_history" in first_post_step_states
        ):
            continuous_first_q, continuous_first_qd = first_post_step_states["A_continuous_history"]
            restored_first_q, restored_first_qd = first_post_step_states["E_cold_restored_pose_history"]
            first_solve_equivalence = {
                "expected_input_q_bitwise_equal": torch.equal(
                    continuous_expected_input_q,
                    restored_expected_input_q,
                ),
                "expected_input_qd_bitwise_equal": torch.equal(effective_qd, restored_effective_qd),
                "expected_input_maximum_q_difference": float(
                    torch.abs(continuous_expected_input_q - restored_expected_input_q).max()
                ),
                "expected_input_maximum_qd_difference": float(torch.abs(effective_qd - restored_effective_qd).max()),
                "first_post_solve_maximum_position_difference_m": float(
                    torch.abs(continuous_first_q[..., :3] - restored_first_q[..., :3]).max()
                ),
                "first_post_solve_maximum_orientation_difference_rad": float(
                    math_utils.quat_error_magnitude(
                        continuous_first_q[..., 3:7].reshape(-1, 4),
                        restored_first_q[..., 3:7].reshape(-1, 4),
                    ).max()
                ),
                "first_post_solve_maximum_qd_difference": float(
                    torch.abs(continuous_first_qd - restored_first_qd).max()
                ),
            }

        self._restore_state(candidate, restore_pose_history=False)
        result = {
            "passed_as_diagnostic": True,
            "candidate_live_position_error_m": candidate_pose_error.detach().cpu().tolist(),
            "candidate_live_orientation_error_rad": candidate_angle_error.detach().cpu().tolist(),
            "solver_dt_s": float(NewtonManager.get_solver_dt()),
            "cable03_body_index": cable03_index,
            "cable03_body_name": self.layout.body_names[cable03_index],
            "maximum_coupling_velocity_delta": torch.abs(coupling_delta).amax(dim=(1, 2)).detach().cpu().tolist(),
            "cable03_captured_raw_qd": candidate["task_body_velocity"][:, cable03_index].detach().cpu().tolist(),
            "cable03_coupling_velocity_delta": coupling_delta[:, cable03_index].detach().cpu().tolist(),
            "cable03_effective_qd": effective_qd[:, cable03_index].detach().cpu().tolist(),
            "first_solve_equivalence": first_solve_equivalence,
            "branches": branch_results,
        }
        print(f"[PICK-INSERT RESET A/B/C/D/E COMPLETE] {result}", flush=True)
        return result

    def _sample_phase_tcp_orientation_error(self, phase: int) -> torch.Tensor | None:
        """Sample the one phase-specific TCP orientation error, if any."""
        if phase != 4:
            return None
        return sample_phase_4_pregrasp_orientation_errors(
            self.env.num_envs,
            device=self.device,
            rng=self.random,
            maximum_top_down_tilt_error_rad=(self.cfg.phase_4_pregrasp_maximum_top_down_tilt_error_rad),
            maximum_closing_axis_twist_error_rad=(self.cfg.phase_4_pregrasp_maximum_closing_axis_twist_error_rad),
        )

    def _desired_tcp_pose(
        self,
        plug_pose: torch.Tensor,
        *,
        orientation_error_xyzw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grasp_offset = torch.as_tensor(self.env.cfg.plug_grasp_offset, device=self.device, dtype=torch.float32)
        grasp_offset = grasp_offset.expand(len(plug_pose), -1)
        position = plug_pose[:, :3] + math_utils.quat_apply(plug_pose[:, 3:7], grasp_offset)
        orientation = math_utils.quat_mul(plug_pose[:, 3:7], self.local_grasp_orientation[: len(plug_pose)])
        if orientation_error_xyzw is not None:
            orientation_error_xyzw = torch.as_tensor(
                orientation_error_xyzw,
                device=self.device,
                dtype=orientation.dtype,
            )
            if orientation_error_xyzw.shape != orientation.shape:
                raise ValueError("orientation_error_xyzw must match the batched plug quaternion shape.")
            orientation_error_xyzw = orientation_error_xyzw / torch.linalg.vector_norm(
                orientation_error_xyzw,
                dim=-1,
                keepdim=True,
            ).clamp_min(1.0e-9)
            orientation = math_utils.quat_mul(orientation, orientation_error_xyzw)
        return position, orientation

    def _move_tcp(
        self,
        position: torch.Tensor,
        orientation: torch.Tensor,
        finger_target: torch.Tensor,
        *,
        arm_seed: torch.Tensor,
        duration_s: float | None = None,
        diagnostic_label: str | None = None,
        diagnostics: list[dict[str, Any]] | None = None,
        attempt_count: int | None = None,
        settle_s: float | None = None,
        lane_hold: _PerLaneTargetHold,
        failure_reason: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move to a Cartesian target with bounded measured residual compensation."""
        commanded = position.clone()
        initial_active = lane_hold.active_mask
        current_target = torch.where(initial_active[:, None], arm_seed, lane_hold.last_sent_arm_target)
        valid = initial_active.clone()
        motion_s = self.cfg.tcp_motion_s if duration_s is None else duration_s
        move_attempt_count = self.cfg.tcp_compensation_iterations + 1 if attempt_count is None else attempt_count
        move_settle_s = self.cfg.tcp_settle_s if settle_s is None else settle_s
        if isinstance(move_attempt_count, bool) or not isinstance(move_attempt_count, int) or move_attempt_count < 1:
            raise ValueError("Cartesian move attempt_count must be a positive integer.")
        if not math.isfinite(move_settle_s) or move_settle_s <= 0.0:
            raise ValueError("Cartesian move settle_s must be finite and positive.")
        for iteration in range(move_attempt_count):
            active_before = lane_hold.active_mask
            if not bool(active_before.any()):
                break
            actual_before = self.env.tcp_pose_e()
            fallback_orientation = torch.zeros_like(orientation)
            fallback_orientation[:, 3] = 1.0
            actual_orientation_finite = torch.isfinite(actual_before[:, 3:7]).all(dim=-1)
            fallback_orientation = torch.where(
                actual_orientation_finite[:, None],
                actual_before[:, 3:7],
                fallback_orientation,
            )
            solve_position = torch.where(
                active_before[:, None],
                commanded,
                torch.nan_to_num(actual_before[:, :3]),
            )
            solve_orientation = torch.where(active_before[:, None], orientation, fallback_orientation)
            solve_finger_target = torch.where(
                active_before[:, None],
                finger_target,
                lane_hold.last_sent_finger_target,
            )
            solution = self._solve_ik(
                solve_position,
                solve_orientation,
                solve_finger_target,
                arm_seed=current_target,
            )
            solver_invalid = active_before & ~solution.valid
            if bool(solver_invalid.any()):
                invalid = torch.where(solver_invalid)[0][:4]
                print(
                    "[PICK-INSERT IK INVALID] "
                    f"iteration={iteration}, worlds={invalid.detach().cpu().tolist()}, "
                    f"target={position[invalid].detach().cpu().tolist()}, "
                    f"commanded={commanded[invalid].detach().cpu().tolist()}, "
                    f"predicted_tcp={solution.tcp_position[invalid].detach().cpu().tolist()}, "
                    f"position_residual={solution.position_residual[invalid].detach().cpu().tolist()}, "
                    f"rotation_residual={solution.rotation_residual[invalid].detach().cpu().tolist()}, "
                    f"arm_seed={current_target[invalid].detach().cpu().tolist()}, "
                    f"selected_arm={solution.arm_q[invalid].detach().cpu().tolist()}, "
                    f"actual_tcp_before={actual_before[invalid, :3].detach().cpu().tolist()}.",
                    flush=True,
                )
            solution_valid = solution.valid & joint_limit_mask(self.env, solution.arm_q, margin=0.02)
            valid &= ~active_before | solution_valid
            command_mask = active_before & solution_valid
            lane_hold.deactivate(active_before & ~solution_valid, reason=failure_reason)
            held_target = lane_hold.last_sent_arm_target
            safe_arm_target = torch.where(command_mask[:, None], solution.arm_q, held_target)
            interpolate_arm_motion(
                self.env,
                held_target,
                safe_arm_target,
                finger_target,
                motion_s if iteration == 0 else 0.4,
            )
            self.env.set_robot_targets(safe_arm_target, finger_target)
            self.env.advance(move_settle_s)
            current_target = lane_hold.last_sent_arm_target
            error = position - self.env.tcp_pose_e()[:, :3]
            norm = torch.linalg.vector_norm(error, dim=-1)
            diagnostic_failure = active_before & ~solution_valid
            if diagnostics is not None and bool(diagnostic_failure.any()):
                limits = self.env._robot.data.soft_joint_pos_limits.torch[0, self.env._arm_joint_ids]
                selected_margin = torch.minimum(
                    solution.arm_q - limits[:, 0],
                    limits[:, 1] - solution.arm_q,
                ).amin(dim=-1)
                diagnostics.append(
                    {
                        "label": diagnostic_label or "tcp_move",
                        "failure": "ik-or-joint-limit",
                        "failed_worlds": torch.where(diagnostic_failure)[0].detach().cpu().tolist(),
                        "compensation_iteration": iteration,
                        "solver_valid": solution.valid.detach().cpu().tolist(),
                        "selected_joint_limit_margin_rad": selected_margin.detach().cpu().tolist(),
                        "selected_arm_joint_position_rad": solution.arm_q.detach().cpu().tolist(),
                        "predicted_position_residual_m": solution.position_residual.detach().cpu().tolist(),
                        "predicted_rotation_residual_rad": solution.rotation_residual.detach().cpu().tolist(),
                        "measured_tcp_error_m": norm.detach().cpu().tolist(),
                    }
                )
            correction_needed = lane_hold.active_mask & (norm > self.cfg.tcp_compensation_tolerance_m)
            if not bool(correction_needed.any()):
                break
            correction = error * torch.clamp(
                self.cfg.tcp_compensation_max_step_m / norm[:, None].clamp_min(1.0e-9), max=1.0
            )
            commanded += torch.where(correction_needed[:, None], correction, torch.zeros_like(correction))
        final_error = torch.linalg.vector_norm(position - self.env.tcp_pose_e()[:, :3], dim=-1)
        valid &= final_error <= self.cfg.tcp_compensation_tolerance_m
        tracking_failure = initial_active & (final_error > self.cfg.tcp_compensation_tolerance_m)
        if diagnostics is not None and bool(tracking_failure.any()):
            diagnostics.append(
                {
                    "label": diagnostic_label or "tcp_move",
                    "failure": "tracking-tolerance",
                    "failed_worlds": torch.where(tracking_failure)[0].detach().cpu().tolist(),
                    "measured_tcp_error_m": final_error.detach().cpu().tolist(),
                    "maximum_allowed_tcp_error_m": self.cfg.tcp_compensation_tolerance_m,
                }
            )
        return lane_hold.last_sent_arm_target, valid & lane_hold.active_mask

    def _close_gripper(self, arm_target: torch.Tensor) -> None:
        def update(_step: int, _steps: int, progress: float) -> None:
            blend = progress * progress * (3.0 - 2.0 * progress)
            self.env.set_robot_targets(
                arm_target,
                torch.lerp(self.open_finger_q, self.closed_finger_target, blend),
            )

        self.env.advance(self.cfg.grasp_close_s, update)
        self.env.set_robot_targets(arm_target, self.closed_finger_target)
        self.env.advance(self.cfg.grasp_hold_s)

    def _move_tcp_with_fixed_arm_bias(
        self,
        position: torch.Tensor,
        orientation: torch.Tensor,
        finger_target: torch.Tensor,
        *,
        raw_arm_seed: torch.Tensor,
        arm_target_bias: torch.Tensor,
        duration_s: float,
        diagnostic_label: str,
        diagnostics: list[dict[str, Any]],
        compensation_iterations: int = 2,
        require_tracking_tolerance: bool = True,
        settle_s: float | None = None,
        maximum_raw_ik_joint_step_rad: float | None = None,
        lane_hold: _PerLaneTargetHold,
        failure_reason: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move along one Cartesian segment while preserving calibrated actuator preload."""
        commanded = position.clone()
        current_raw = raw_arm_seed.clone()
        initial_active = lane_hold.active_mask
        valid = initial_active.clone()
        limits = self.env._robot.data.soft_joint_pos_limits.torch[0, self.env._arm_joint_ids]
        settle_duration = self.cfg.tcp_settle_s if settle_s is None else settle_s
        raw_joint_step_limit = (
            self.cfg.maximum_ik_joint_step_rad
            if maximum_raw_ik_joint_step_rad is None
            else maximum_raw_ik_joint_step_rad
        )
        for iteration in range(compensation_iterations + 1):
            active_before = lane_hold.active_mask
            if not bool(active_before.any()):
                break
            actual_tcp = self.env.tcp_pose_e()
            fallback_orientation = torch.zeros_like(orientation)
            fallback_orientation[:, 3] = 1.0
            actual_orientation_finite = torch.isfinite(actual_tcp[:, 3:7]).all(dim=-1)
            fallback_orientation = torch.where(
                actual_orientation_finite[:, None],
                actual_tcp[:, 3:7],
                fallback_orientation,
            )
            solve_position = torch.where(
                active_before[:, None],
                commanded,
                torch.nan_to_num(actual_tcp[:, :3]),
            )
            solve_orientation = torch.where(active_before[:, None], orientation, fallback_orientation)
            solve_finger_target = torch.where(
                active_before[:, None],
                finger_target,
                lane_hold.last_sent_finger_target,
            )
            solution = self._solve_ik(
                solve_position,
                solve_orientation,
                solve_finger_target,
                arm_seed=current_raw,
            )
            raw_ik_joint_delta = torch.abs(solution.arm_q - current_raw).amax(dim=-1)
            proposed_target = solution.arm_q + arm_target_bias
            selected_margin = torch.minimum(
                proposed_target - limits[:, 0],
                limits[:, 1] - proposed_target,
            ).amin(dim=-1)
            solution_valid = (
                solution.valid
                & joint_limit_mask(self.env, proposed_target, margin=0.02)
                & (raw_ik_joint_delta <= raw_joint_step_limit)
            )
            valid &= ~active_before | solution_valid
            diagnostic_failure = active_before & ~solution_valid
            if bool(diagnostic_failure.any()):
                diagnostics.append(
                    {
                        "label": diagnostic_label,
                        "failure": "ik-joint-limit-or-continuity",
                        "failed_worlds": torch.where(diagnostic_failure)[0].detach().cpu().tolist(),
                        "compensation_iteration": iteration,
                        "solver_valid": solution.valid.detach().cpu().tolist(),
                        "selected_joint_limit_margin_rad": selected_margin.detach().cpu().tolist(),
                        "maximum_raw_ik_joint_delta_rad": raw_ik_joint_delta.detach().cpu().tolist(),
                        "maximum_allowed_raw_ik_joint_delta_rad": raw_joint_step_limit,
                        "selected_arm_joint_position_rad": solution.arm_q.detach().cpu().tolist(),
                        "arm_target_bias_rad": arm_target_bias.detach().cpu().tolist(),
                        "predicted_position_residual_m": solution.position_residual.detach().cpu().tolist(),
                        "predicted_rotation_residual_rad": solution.rotation_residual.detach().cpu().tolist(),
                        "measured_tcp_error_m": torch.linalg.vector_norm(
                            position - self.env.tcp_pose_e()[:, :3], dim=-1
                        )
                        .detach()
                        .cpu()
                        .tolist(),
                        "motion_skipped_before_physics": not bool((active_before & solution_valid).any()),
                    }
                )
            command_mask = active_before & solution_valid
            lane_hold.deactivate(active_before & ~solution_valid, reason=failure_reason)
            held_target = lane_hold.last_sent_arm_target
            safe_raw = torch.where(command_mask[:, None], solution.arm_q, current_raw)
            safe_target = torch.where(command_mask[:, None], proposed_target, held_target)
            interpolate_arm_motion(
                self.env,
                held_target,
                safe_target,
                finger_target,
                duration_s if iteration == 0 else 0.25,
            )
            self.env.set_robot_targets(safe_target, finger_target)
            if settle_duration > 0.0:
                self.env.advance(settle_duration)
            current_raw = safe_raw
            error = position - self.env.tcp_pose_e()[:, :3]
            norm = torch.linalg.vector_norm(error, dim=-1)
            if bool(diagnostic_failure.any()):
                diagnostics[-1]["measured_tcp_error_after_motion_m"] = norm.detach().cpu().tolist()
            correction_needed = lane_hold.active_mask & (norm > self.cfg.tcp_compensation_tolerance_m)
            if not bool(correction_needed.any()):
                break
            correction = error * torch.clamp(
                self.cfg.tcp_compensation_max_step_m / norm[:, None].clamp_min(1.0e-9), max=1.0
            )
            commanded += torch.where(correction_needed[:, None], correction, torch.zeros_like(correction))
        final_error = torch.linalg.vector_norm(position - self.env.tcp_pose_e()[:, :3], dim=-1)
        if require_tracking_tolerance:
            valid &= final_error <= self.cfg.tcp_compensation_tolerance_m
            tracking_failure = initial_active & (final_error > self.cfg.tcp_compensation_tolerance_m)
            if bool(tracking_failure.any()):
                diagnostics.append(
                    {
                        "label": diagnostic_label,
                        "failure": "tracking-tolerance",
                        "failed_worlds": torch.where(tracking_failure)[0].detach().cpu().tolist(),
                        "measured_tcp_error_m": final_error.detach().cpu().tolist(),
                        "maximum_allowed_tcp_error_m": self.cfg.tcp_compensation_tolerance_m,
                    }
                )
        return current_raw, lane_hold.last_sent_arm_target, valid & lane_hold.active_mask

    def _open_gripper(self, arm_target: torch.Tensor) -> None:
        _, _, finger_start, _ = self.env.read_robot_state()

        def update(_step: int, _steps: int, progress: float) -> None:
            blend = progress * progress * (3.0 - 2.0 * progress)
            self.env.set_robot_targets(arm_target, torch.lerp(finger_start, self.open_finger_q, blend))

        self.env.advance(self.cfg.grasp_close_s, update)
        self.env.set_robot_targets(arm_target, self.open_finger_q)
        self.env.advance(self.cfg.grasp_hold_s)

    def _settle_acquired_plug(
        self,
        arm_target: torch.Tensor,
        *,
        require_construction_drives_enabled: bool = False,
        lane_hold: _PerLaneTargetHold | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Prove a newly closed grasp under the explicitly required construction-drive state."""
        all_finite = torch.ones(self.env.num_envs, device=self.device, dtype=torch.bool)
        all_collision_free = all_finite.clone()
        all_bilateral = all_finite.clone()
        all_grasp_valid = all_finite.clone()
        all_drives_disabled = all_finite.clone()
        all_construction_drives_enabled = all_finite.clone()
        all_drive_states_expected = all_finite.clone()
        maximum_cable_speed = torch.zeros(self.env.num_envs, device=self.device)
        maximum_plug_linear_speed = torch.zeros_like(maximum_cable_speed)
        maximum_plug_angular_speed = torch.zeros_like(maximum_cable_speed)
        any_overflow = False
        invalid_pairs: list[str] = []

        def sample(_step: int, _steps: int, _progress: float) -> None:
            nonlocal any_overflow
            sample_q, sample_qd = self.env.read_task_state()
            sample_collision = collision_metrics(self.env, require_bilateral_grasp=False)
            if sample_collision.contact_overflow:
                raise RuntimeError("Global contact-buffer overflow during acquired-plug settling.")
            cable_speed = torch.linalg.vector_norm(sample_qd[:, self.cable_slice, :3], dim=-1).amax(dim=-1)
            plug_linear_speed = torch.linalg.vector_norm(sample_qd[:, self.plug_index, :3], dim=-1)
            plug_angular_speed = torch.linalg.vector_norm(sample_qd[:, self.plug_index, 3:6], dim=-1)
            bilateral = _runtime_bilateral_grasp_proxy_contact_mask(
                self.env,
                sample_collision.left_grasp_contact_count,
                sample_collision.right_grasp_contact_count,
            )
            grasp = grasp_metrics(self.env, self.closed_finger_target, retaining_grasp=False)
            translation_drive_enabled = self._drive_enabled()
            orientation_hold_enabled = self._orientation_hold_enabled()
            drives_disabled = ~translation_drive_enabled & ~orientation_hold_enabled
            construction_drives_enabled = translation_drive_enabled & orientation_hold_enabled
            drive_state_expected = (
                construction_drives_enabled if require_construction_drives_enabled else drives_disabled
            )
            if not bool(drive_state_expected.all()):
                raise RuntimeError("Construction-drive state changed during acquired-plug settling.")
            finite = task_state_is_finite_and_normalized(sample_q, sample_qd)
            all_finite.logical_and_(finite)
            all_collision_free.logical_and_(sample_collision.valid)
            all_bilateral.logical_and_(bilateral)
            all_grasp_valid.logical_and_(grasp.valid)
            all_drives_disabled.logical_and_(drives_disabled)
            all_construction_drives_enabled.logical_and_(construction_drives_enabled)
            all_drive_states_expected.logical_and_(drive_state_expected)
            maximum_cable_speed.copy_(torch.maximum(maximum_cable_speed, cable_speed))
            maximum_plug_linear_speed.copy_(torch.maximum(maximum_plug_linear_speed, plug_linear_speed))
            maximum_plug_angular_speed.copy_(torch.maximum(maximum_plug_angular_speed, plug_angular_speed))
            any_overflow |= sample_collision.contact_overflow
            if lane_hold is not None:
                lane_hold.deactivate(~finite, reason="post-contact-non-finite")
                lane_hold.deactivate(~sample_collision.valid, reason="post-contact-collision")
                lane_hold.deactivate(~bilateral, reason="post-contact-lost-bilateral-contact")
                lane_hold.deactivate(~grasp.valid, reason="post-contact-invalid-grasp")
            for pair in sample_collision.invalid_contact_pairs:
                if pair not in invalid_pairs and len(invalid_pairs) < 64:
                    invalid_pairs.append(pair)

        self.env.set_robot_targets(arm_target, self.closed_finger_target)
        self.env.advance(
            self.cfg.grasp_post_contact_settle_s,
            lambda _step, _steps, _progress: self.env.set_robot_targets(
                arm_target,
                self.closed_finger_target,
            ),
            post_step=sample,
        )
        _, final_task_qd = self.env.read_task_state()
        _, final_arm_qd, _, final_finger_qd = self.env.read_robot_state()
        final_cable_speed = torch.linalg.vector_norm(
            final_task_qd[:, self.cable_slice, :3],
            dim=-1,
        ).amax(dim=-1)
        final_plug_linear_speed = torch.linalg.vector_norm(final_task_qd[:, self.plug_index, :3], dim=-1)
        final_plug_angular_speed = torch.linalg.vector_norm(final_task_qd[:, self.plug_index, 3:6], dim=-1)
        final_arm_speed = torch.abs(final_arm_qd).amax(dim=-1)
        final_finger_speed = torch.abs(final_finger_qd).amax(dim=-1)
        valid = (
            all_finite
            & all_collision_free
            & all_bilateral
            & all_grasp_valid
            & all_drive_states_expected
            & (not any_overflow)
            & (final_cable_speed <= self.cfg.maximum_row_cable_speed_m_s)
            & (final_plug_linear_speed <= self.cfg.maximum_pickup_plug_linear_speed_m_s)
            & (final_plug_angular_speed <= self.cfg.maximum_pickup_plug_angular_speed_rad_s)
            & (final_arm_speed <= self.cfg.maximum_row_arm_joint_speed_rad_s)
            & (final_finger_speed <= self.cfg.maximum_row_finger_joint_speed_m_s)
        )
        if lane_hold is not None:
            lane_hold.deactivate(~valid, reason="post-contact-final-state")
        return valid, {
            "all_samples_finite": all_finite,
            "all_samples_collision_free": all_collision_free,
            "all_samples_bilateral_proxy_contact": all_bilateral,
            "all_samples_drives_disabled": all_drives_disabled,
            "all_samples_construction_drives_enabled": all_construction_drives_enabled,
            "all_samples_drive_state_expected": all_drive_states_expected,
            "required_construction_drives_enabled": require_construction_drives_enabled,
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

    def _acquire_current_plug(
        self,
        arm_seed: torch.Tensor,
        *,
        duration_s: float,
        require_construction_drives_enabled: bool = False,
        active_mask: torch.Tensor | None = None,
        move_attempt_count: int | None = None,
        move_settle_s: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Acquire the live plug while permanently isolating failed batch lanes."""
        initial_active = (
            torch.ones(self.env.num_envs, device=self.device, dtype=torch.bool)
            if active_mask is None
            else torch.as_tensor(active_mask, device=self.device, dtype=torch.bool)
        )
        with _PerLaneTargetHold(
            self.env,
            initial_active,
            arm_seed,
            self.open_finger_q,
        ) as lane_hold:
            arm_target, valid, evidence = self._acquire_current_plug_per_lane(
                arm_seed,
                duration_s=duration_s,
                require_construction_drives_enabled=require_construction_drives_enabled,
                move_attempt_count=move_attempt_count,
                move_settle_s=move_settle_s,
                lane_hold=lane_hold,
            )
            lane_hold.deactivate(~valid, reason="acquisition-final-validation")
            valid &= lane_hold.active_mask
            evidence["lane_failure_masks"] = lane_hold.reason_masks
            evidence["last_arm_target"] = lane_hold.last_sent_arm_target
            evidence["last_finger_target"] = lane_hold.last_sent_finger_target
            return lane_hold.last_sent_arm_target, valid, evidence

    def _acquire_current_plug_per_lane(
        self,
        arm_seed: torch.Tensor,
        *,
        duration_s: float,
        require_construction_drives_enabled: bool,
        move_attempt_count: int | None,
        move_settle_s: float | None,
        lane_hold: _PerLaneTargetHold,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Acquire a settled plug through a guarded route under an explicit drive-state contract."""
        task_q, _ = self.env.read_task_state()
        reference_plug = task_q[:, self.plug_index].clone()
        exact_position, orientation = self._desired_tcp_pose(reference_plug)
        clearance_position = exact_position.clone()
        clearance_position[:, 2] += self.cfg.grasp_open_clearance_m
        ik_diagnostics: list[dict[str, Any]] = []
        true_mask = torch.ones(self.env.num_envs, device=self.device, dtype=torch.bool)
        false_mask = torch.zeros_like(true_mask)
        approach_evidence: dict[str, Any] = {
            "all_samples_collision_free": true_mask.clone(),
            "all_samples_zero_proxy_contacts": true_mask.clone(),
            "all_samples_drives_disabled": true_mask.clone(),
            "all_samples_construction_drives_enabled": true_mask.clone(),
            "all_samples_drive_state_expected": true_mask.clone(),
            "required_construction_drives_enabled": require_construction_drives_enabled,
            "all_samples_task_state_valid": true_mask.clone(),
            "maximum_plug_position_drift_m": torch.zeros(self.env.num_envs, device=self.device),
            "maximum_left_proxy_contact_count": torch.zeros(self.env.num_envs, device=self.device, dtype=torch.long),
            "maximum_right_proxy_contact_count": torch.zeros(self.env.num_envs, device=self.device, dtype=torch.long),
            "any_contact_overflow": False,
            "invalid_contact_pairs": [],
            "samples": 0,
            "abort_reason": None,
        }
        original_advance = self.env.advance

        def sample_open_approach() -> None:
            sample_q, sample_qd = self.env.read_task_state()
            sample_collision = collision_metrics(self.env, require_bilateral_grasp=False)
            if sample_collision.contact_overflow:
                raise RuntimeError("Global contact-buffer overflow during open-gripper acquisition.")
            plug_drift = torch.linalg.vector_norm(sample_q[:, self.plug_index, :3] - reference_plug[:, :3], dim=-1)
            zero_proxy = (sample_collision.left_grasp_contact_count == 0) & (
                sample_collision.right_grasp_contact_count == 0
            )
            translation_drive_enabled = self._drive_enabled()
            orientation_hold_enabled = self._orientation_hold_enabled()
            drives_disabled = ~translation_drive_enabled & ~orientation_hold_enabled
            construction_drives_enabled = translation_drive_enabled & orientation_hold_enabled
            drive_state_expected = (
                construction_drives_enabled if require_construction_drives_enabled else drives_disabled
            )
            if not bool(drive_state_expected.all()):
                raise RuntimeError("Construction-drive state changed during open-gripper acquisition.")
            state_valid = task_state_is_finite_and_normalized(sample_q, sample_qd)
            approach_evidence["samples"] += 1
            approach_evidence["all_samples_collision_free"] &= sample_collision.valid
            approach_evidence["all_samples_zero_proxy_contacts"] &= zero_proxy
            approach_evidence["all_samples_drives_disabled"] &= drives_disabled
            approach_evidence["all_samples_construction_drives_enabled"] &= construction_drives_enabled
            approach_evidence["all_samples_drive_state_expected"] &= drive_state_expected
            approach_evidence["all_samples_task_state_valid"] &= state_valid
            approach_evidence["maximum_plug_position_drift_m"] = torch.maximum(
                approach_evidence["maximum_plug_position_drift_m"], plug_drift
            )
            approach_evidence["maximum_left_proxy_contact_count"] = torch.maximum(
                approach_evidence["maximum_left_proxy_contact_count"],
                sample_collision.left_grasp_contact_count,
            )
            approach_evidence["maximum_right_proxy_contact_count"] = torch.maximum(
                approach_evidence["maximum_right_proxy_contact_count"],
                sample_collision.right_grasp_contact_count,
            )
            approach_evidence["any_contact_overflow"] |= sample_collision.contact_overflow
            for pair in sample_collision.invalid_contact_pairs:
                if pair not in approach_evidence["invalid_contact_pairs"]:
                    approach_evidence["invalid_contact_pairs"].append(pair)
            violated = (
                ~sample_collision.valid
                | ~zero_proxy
                | ~drive_state_expected
                | ~state_valid
                | (plug_drift > self.cfg.maximum_open_approach_plug_drift_m)
            )
            if bool(violated.any()):
                reasons: list[str] = []
                if bool((~sample_collision.valid).any()):
                    reasons.append("collision")
                if bool((~zero_proxy).any()):
                    reasons.append("proxy-contact")
                if bool((~drive_state_expected).any()):
                    reasons.append("unexpected-construction-drive-state")
                if bool((~state_valid).any()):
                    reasons.append("invalid-task-state")
                if bool((plug_drift > self.cfg.maximum_open_approach_plug_drift_m).any()):
                    reasons.append("plug-drift")
                approach_evidence["abort_reason"] = "+".join(reasons)
                lane_hold.deactivate(~sample_collision.valid, reason="open-approach-collision")
                lane_hold.deactivate(~zero_proxy, reason="open-approach-proxy-contact")
                lane_hold.deactivate(~state_valid, reason="open-approach-invalid-task-state")
                lane_hold.deactivate(
                    plug_drift > self.cfg.maximum_open_approach_plug_drift_m,
                    reason="open-approach-plug-drift",
                )

        def guarded_advance(duration_s: float, update=None, *, post_step=None):
            def combined_post_step(step: int, steps: int, progress: float) -> None:
                if post_step is not None:
                    post_step(step, steps, progress)
                sample_open_approach()

            return original_advance(duration_s, update, post_step=combined_post_step)

        route_valid = true_mask.clone()
        clearance_ik_valid = false_mask.clone()
        open_descent_ik_valid = false_mask.clone()
        open_descent_collision_valid = false_mask.clone()
        maximum_descent_error = torch.zeros(self.env.num_envs, device=self.device)
        contact_arm_target = arm_seed.clone()
        clearance_arm_target_bias = torch.zeros_like(arm_seed)
        clearance_tcp_position = self.env.tcp_pose_e()[:, :3].clone()
        preclose_collision = collision_metrics(self.env, require_bilateral_grasp=False)
        preclose_error = torch.linalg.vector_norm(
            self.env.tcp_pose_e()[:, :3] - clearance_position,
            dim=-1,
        )
        self.env.advance = guarded_advance
        try:
            current_tcp = self.env.tcp_pose_e()[:, :3].clone()
            route_height = torch.maximum(
                current_tcp[:, 2],
                torch.full_like(current_tcp[:, 2], self.cfg.grasp_route_world_height_m),
            )
            lift_position = current_tcp.clone()
            lift_position[:, 2] = route_height
            route_arm_target, lift_valid = self._move_tcp(
                lift_position,
                orientation,
                self.open_finger_q,
                arm_seed=arm_seed,
                duration_s=duration_s,
                diagnostic_label="open_route_lift_rotate",
                diagnostics=ik_diagnostics,
                attempt_count=move_attempt_count,
                settle_s=move_settle_s,
                lane_hold=lane_hold,
                failure_reason="open-route-lift-ik",
            )
            route_valid &= lift_valid
            lane_hold.deactivate(~lift_valid, reason="open-route-lift-ik")
            route_arm_target = lane_hold.last_sent_arm_target

            overhead_position = exact_position.clone()
            overhead_position[:, 2] = route_height
            route_distance = torch.linalg.vector_norm(overhead_position - lift_position, dim=-1)
            route_steps, invalid_route_distance = _active_waypoint_count(
                route_distance,
                lane_hold.active_mask,
                self.cfg.grasp_route_maximum_translation_step_m,
            )
            route_valid &= ~invalid_route_distance
            lane_hold.deactivate(invalid_route_distance, reason="open-route-distance-invalid")
            for waypoint in range(1, route_steps + 1):
                progress = waypoint / route_steps
                waypoint_position = torch.lerp(lift_position, overhead_position, progress)
                route_arm_target, waypoint_valid = self._move_tcp(
                    waypoint_position,
                    orientation,
                    self.open_finger_q,
                    arm_seed=route_arm_target,
                    duration_s=self.cfg.grasp_descent_waypoint_motion_s,
                    diagnostic_label=f"open_route_overhead_{waypoint:02d}_of_{route_steps:02d}",
                    diagnostics=ik_diagnostics,
                    attempt_count=move_attempt_count,
                    settle_s=move_settle_s,
                    lane_hold=lane_hold,
                    failure_reason="open-route-overhead-ik",
                )
                route_valid &= waypoint_valid
                lane_hold.deactivate(~waypoint_valid, reason="open-route-overhead-ik")
                route_arm_target = lane_hold.last_sent_arm_target

            raw_route_active = lane_hold.active_mask
            if bool(raw_route_active.any()):
                current_tcp_pose = self.env.tcp_pose_e()
                fallback_orientation = torch.zeros_like(orientation)
                fallback_orientation[:, 3] = 1.0
                fallback_orientation = torch.where(
                    torch.isfinite(current_tcp_pose[:, 3:7]).all(dim=-1, keepdim=True),
                    current_tcp_pose[:, 3:7],
                    fallback_orientation,
                )
                raw_route = self._solve_ik(
                    torch.where(
                        raw_route_active[:, None],
                        overhead_position,
                        torch.nan_to_num(current_tcp_pose[:, :3]),
                    ),
                    torch.where(raw_route_active[:, None], orientation, fallback_orientation),
                    torch.where(
                        raw_route_active[:, None],
                        self.open_finger_q,
                        lane_hold.last_sent_finger_target,
                    ),
                    arm_seed=route_arm_target,
                )
                raw_route_valid = (
                    raw_route_active & raw_route.valid & joint_limit_mask(self.env, route_arm_target, margin=0.02)
                )
                raw_contact_arm = torch.where(raw_route_valid[:, None], raw_route.arm_q, route_arm_target)
            else:
                raw_route_valid = torch.zeros_like(raw_route_active)
                raw_contact_arm = route_arm_target
            route_valid &= ~raw_route_active | raw_route_valid
            lane_hold.deactivate(raw_route_active & ~raw_route_valid, reason="open-route-bias-ik")
            contact_arm_target = lane_hold.last_sent_arm_target
            route_bias = contact_arm_target - raw_contact_arm
            coarse_distance = torch.clamp(route_height - clearance_position[:, 2], min=0.0)
            coarse_steps, invalid_coarse_distance = _active_waypoint_count(
                coarse_distance,
                lane_hold.active_mask,
                self.cfg.grasp_coarse_descent_step_m,
            )
            route_valid &= ~invalid_coarse_distance
            lane_hold.deactivate(invalid_coarse_distance, reason="open-coarse-distance-invalid")
            for waypoint in range(1, coarse_steps + 1):
                progress = waypoint / coarse_steps
                waypoint_position = torch.lerp(overhead_position, clearance_position, progress)
                raw_contact_arm, contact_arm_target, waypoint_valid = self._move_tcp_with_fixed_arm_bias(
                    waypoint_position,
                    orientation,
                    self.open_finger_q,
                    raw_arm_seed=raw_contact_arm,
                    arm_target_bias=route_bias,
                    duration_s=self.cfg.grasp_descent_waypoint_motion_s,
                    diagnostic_label=f"open_coarse_descent_{waypoint:02d}_of_{coarse_steps:02d}",
                    diagnostics=ik_diagnostics,
                    compensation_iterations=0,
                    require_tracking_tolerance=False,
                    settle_s=self.cfg.grasp_descent_waypoint_settle_s,
                    lane_hold=lane_hold,
                    failure_reason="open-coarse-ik-continuity",
                )
                route_valid &= waypoint_valid
                lane_hold.deactivate(~waypoint_valid, reason="open-coarse-ik-continuity")
                contact_arm_target = lane_hold.last_sent_arm_target

            clearance_ik_valid = route_valid.clone()
            clearance_command = clearance_position.clone()
            for calibration_iteration in range(self.cfg.grasp_clearance_calibration_max_iterations):
                clearance_error = clearance_position - self.env.tcp_pose_e()[:, :3]
                clearance_error_norm = torch.linalg.vector_norm(clearance_error, dim=-1)
                correction_needed = lane_hold.active_mask & (
                    clearance_error_norm > self.cfg.tcp_compensation_tolerance_m
                )
                if not bool(correction_needed.any()):
                    break
                correction = clearance_error * torch.clamp(
                    self.cfg.grasp_clearance_calibration_step_m / clearance_error_norm[:, None].clamp_min(1.0e-9),
                    max=1.0,
                )
                correction = torch.where(correction_needed[:, None], correction, torch.zeros_like(correction))
                clearance_command += correction
                raw_contact_arm, contact_arm_target, calibration_valid = self._move_tcp_with_fixed_arm_bias(
                    clearance_command,
                    orientation,
                    self.open_finger_q,
                    raw_arm_seed=raw_contact_arm,
                    arm_target_bias=route_bias,
                    duration_s=self.cfg.grasp_descent_waypoint_motion_s,
                    diagnostic_label=(
                        f"open_clearance_calibration_{calibration_iteration + 1:02d}_of_"
                        f"{self.cfg.grasp_clearance_calibration_max_iterations:02d}"
                    ),
                    diagnostics=ik_diagnostics,
                    compensation_iterations=0,
                    require_tracking_tolerance=False,
                    settle_s=self.cfg.grasp_descent_waypoint_settle_s,
                    maximum_raw_ik_joint_step_rad=self.cfg.grasp_near_maximum_raw_ik_joint_step_rad,
                    lane_hold=lane_hold,
                    failure_reason="open-clearance-ik-continuity",
                )
                clearance_ik_valid &= calibration_valid
                lane_hold.deactivate(~calibration_valid, reason="open-clearance-ik-continuity")
                contact_arm_target = lane_hold.last_sent_arm_target
            preclose_error = torch.linalg.vector_norm(
                self.env.tcp_pose_e()[:, :3] - clearance_position,
                dim=-1,
            )
            preclose_collision = collision_metrics(self.env, require_bilateral_grasp=False)
            clearance_tcp_position = self.env.tcp_pose_e()[:, :3].clone()
            # The clearance correction is Cartesian, not an actuator preload.
            # Carry its commanded pose down the vertical continuation while
            # preserving the one actuator bias calibrated safely overhead.
            clearance_arm_target_bias = route_bias.clone()
            clearance_ik_valid &= (preclose_error <= self.cfg.tcp_compensation_tolerance_m) & joint_limit_mask(
                self.env,
                contact_arm_target,
                margin=0.02,
            )
            lane_hold.deactivate(~clearance_ik_valid, reason="open-clearance-final-state")
            contact_arm_target = lane_hold.last_sent_arm_target
            ik_diagnostics.append(
                {
                    "label": "open_clearance_45mm_calibrated_bias",
                    "solver_valid": clearance_ik_valid.detach().cpu().tolist(),
                    "raw_arm_joint_position_rad": raw_contact_arm.detach().cpu().tolist(),
                    "arm_target_bias_rad": clearance_arm_target_bias.detach().cpu().tolist(),
                    "calibrated_cartesian_command_position_m": clearance_command.detach().cpu().tolist(),
                    "cartesian_command_correction_m": (clearance_command - clearance_position).detach().cpu().tolist(),
                    "measured_tcp_error_m": preclose_error.detach().cpu().tolist(),
                }
            )

            near_steps = max(
                1,
                int(
                    torch.ceil(
                        torch.tensor(self.cfg.grasp_open_clearance_m / self.cfg.grasp_near_descent_step_m)
                    ).item()
                ),
            )
            open_descent_ik_valid = lane_hold.active_mask
            open_descent_collision_valid = lane_hold.active_mask
            near_command = clearance_command.clone()
            previous_waypoint_position = clearance_position.clone()
            for waypoint in range(1, near_steps + 1):
                progress = waypoint / near_steps
                waypoint_position = torch.lerp(clearance_position, exact_position, progress)
                near_command += waypoint_position - previous_waypoint_position
                diagnostic_label = f"open_near_descent_{waypoint:02d}_of_{near_steps:02d}"
                waypoint_active_before = lane_hold.active_mask
                waypoint_diagnostic_start = len(ik_diagnostics)
                raw_contact_arm, contact_arm_target, waypoint_valid = self._move_tcp_with_fixed_arm_bias(
                    near_command,
                    orientation,
                    self.open_finger_q,
                    raw_arm_seed=raw_contact_arm,
                    arm_target_bias=route_bias,
                    duration_s=self.cfg.grasp_descent_waypoint_motion_s,
                    diagnostic_label=diagnostic_label,
                    diagnostics=ik_diagnostics,
                    compensation_iterations=0,
                    require_tracking_tolerance=False,
                    settle_s=self.cfg.grasp_descent_waypoint_settle_s,
                    maximum_raw_ik_joint_step_rad=self.cfg.grasp_near_maximum_raw_ik_joint_step_rad,
                    lane_hold=lane_hold,
                    failure_reason="open-near-ik-continuity",
                )
                waypoint_collision = collision_metrics(self.env, require_bilateral_grasp=False)
                waypoint_error = torch.linalg.vector_norm(
                    self.env.tcp_pose_e()[:, :3] - waypoint_position,
                    dim=-1,
                )
                initial_waypoint_error = waypoint_error.clone()
                lane_hold.deactivate(~waypoint_valid, reason="open-near-ik-continuity")
                contact_arm_target = lane_hold.last_sent_arm_target
                recovery_needed = lane_hold.active_mask & (waypoint_error > self.cfg.tcp_compensation_tolerance_m)
                if bool(recovery_needed.any()):
                    self.env.set_robot_targets(contact_arm_target, self.open_finger_q)
                    self.env.advance(self.cfg.grasp_descent_tracking_recovery_s)
                    waypoint_collision = collision_metrics(self.env, require_bilateral_grasp=False)
                    if waypoint_collision.contact_overflow:
                        raise RuntimeError("Global contact-buffer overflow during near-grasp recovery.")
                    waypoint_error = torch.linalg.vector_norm(
                        self.env.tcp_pose_e()[:, :3] - waypoint_position,
                        dim=-1,
                    )
                post_hold_error = waypoint_error.clone()
                correction_count = 0
                while (
                    bool((lane_hold.active_mask & (waypoint_error > self.cfg.tcp_compensation_tolerance_m)).any())
                    and correction_count < self.cfg.grasp_near_correction_max_iterations
                ):
                    correction_needed = lane_hold.active_mask & (waypoint_error > self.cfg.tcp_compensation_tolerance_m)
                    error_vector = waypoint_position - self.env.tcp_pose_e()[:, :3]
                    correction_norm = torch.linalg.vector_norm(error_vector, dim=-1)
                    correction = error_vector * torch.clamp(
                        self.cfg.grasp_near_correction_step_m / correction_norm[:, None].clamp_min(1.0e-9),
                        max=1.0,
                    )
                    correction = torch.where(correction_needed[:, None], correction, torch.zeros_like(correction))
                    near_command += correction
                    correction_count += 1
                    raw_contact_arm, contact_arm_target, correction_valid = self._move_tcp_with_fixed_arm_bias(
                        near_command,
                        orientation,
                        self.open_finger_q,
                        raw_arm_seed=raw_contact_arm,
                        arm_target_bias=route_bias,
                        duration_s=self.cfg.grasp_descent_waypoint_motion_s,
                        diagnostic_label=f"{diagnostic_label}_correction_{correction_count:02d}",
                        diagnostics=ik_diagnostics,
                        compensation_iterations=0,
                        require_tracking_tolerance=False,
                        settle_s=self.cfg.grasp_descent_waypoint_settle_s,
                        maximum_raw_ik_joint_step_rad=self.cfg.grasp_near_maximum_raw_ik_joint_step_rad,
                        lane_hold=lane_hold,
                        failure_reason="open-near-correction-ik-continuity",
                    )
                    waypoint_valid &= correction_valid
                    lane_hold.deactivate(~correction_valid, reason="open-near-correction-ik-continuity")
                    contact_arm_target = lane_hold.last_sent_arm_target
                    waypoint_collision = collision_metrics(self.env, require_bilateral_grasp=False)
                    if waypoint_collision.contact_overflow:
                        raise RuntimeError("Global contact-buffer overflow during near-grasp correction.")
                    waypoint_error = torch.linalg.vector_norm(
                        self.env.tcp_pose_e()[:, :3] - waypoint_position,
                        dim=-1,
                    )
                waypoint_ik_valid = waypoint_valid.clone()
                waypoint_ik_failure = waypoint_active_before & ~waypoint_ik_valid
                waypoint_collision_failure = waypoint_active_before & ~waypoint_collision.valid
                waypoint_tracking_failure = waypoint_active_before & (
                    waypoint_error > self.cfg.tcp_compensation_tolerance_m
                )
                waypoint_failure = waypoint_ik_failure | waypoint_collision_failure | waypoint_tracking_failure
                waypoint_valid &= ~waypoint_tracking_failure
                if bool(waypoint_failure.any()):
                    failure_diagnostics = ik_diagnostics[waypoint_diagnostic_start:] or [
                        {
                            "label": diagnostic_label,
                            "failure": "post-motion-waypoint-validation",
                            "failed_worlds": torch.where(waypoint_failure)[0].detach().cpu().tolist(),
                        }
                    ]
                    ik_diagnostics[waypoint_diagnostic_start:] = failure_diagnostics
                    for diagnostic in failure_diagnostics:
                        diagnostic["waypoint_ik_valid"] = waypoint_ik_valid.detach().cpu().tolist()
                        diagnostic["waypoint_collision_valid"] = waypoint_collision.valid.detach().cpu().tolist()
                        diagnostic["desired_waypoint_initial_error_m"] = initial_waypoint_error.detach().cpu().tolist()
                        diagnostic["desired_waypoint_post_hold_error_m"] = post_hold_error.detach().cpu().tolist()
                        diagnostic["desired_waypoint_error_m"] = waypoint_error.detach().cpu().tolist()
                        diagnostic["bounded_cartesian_correction_count"] = correction_count
                open_descent_ik_valid &= waypoint_valid
                open_descent_collision_valid &= waypoint_collision.valid
                maximum_descent_error = torch.maximum(maximum_descent_error, waypoint_error)
                lane_hold.deactivate(~waypoint_collision.valid, reason="open-near-collision")
                lane_hold.deactivate(
                    waypoint_error > self.cfg.tcp_compensation_tolerance_m,
                    reason="open-near-tracking",
                )
                contact_arm_target = lane_hold.last_sent_arm_target
                previous_waypoint_position = waypoint_position
        finally:
            self.env.advance = original_advance

        contact_preclose_collision = collision_metrics(self.env, require_bilateral_grasp=False)
        contact_position_error = torch.linalg.vector_norm(
            self.env.tcp_pose_e()[:, :3] - exact_position,
            dim=-1,
        )
        contact_tcp_position = self.env.tcp_pose_e()[:, :3].clone()
        if preclose_collision.contact_overflow or contact_preclose_collision.contact_overflow:
            raise RuntimeError("Global contact-buffer overflow at the acquisition pre-close boundary.")
        approach_valid = (
            lane_hold.active_mask
            & route_valid
            & clearance_ik_valid
            & preclose_collision.valid
            & open_descent_ik_valid
            & open_descent_collision_valid
            & contact_preclose_collision.valid
            & (contact_position_error <= self.cfg.tcp_compensation_tolerance_m)
            & approach_evidence["all_samples_collision_free"]
            & approach_evidence["all_samples_zero_proxy_contacts"]
            & approach_evidence["all_samples_drive_state_expected"]
            & approach_evidence["all_samples_task_state_valid"]
            & (approach_evidence["maximum_plug_position_drift_m"] <= self.cfg.maximum_open_approach_plug_drift_m)
            & (not approach_evidence["any_contact_overflow"])
        )
        lane_hold.deactivate(~approach_valid, reason="open-approach-final-validation")
        contact_arm_target = lane_hold.last_sent_arm_target
        if bool(lane_hold.active_mask.any()):
            self._close_gripper(contact_arm_target)
            post_contact_valid, post_contact_evidence = self._settle_acquired_plug(
                contact_arm_target,
                require_construction_drives_enabled=require_construction_drives_enabled,
                lane_hold=lane_hold,
            )
        else:
            post_contact_valid = false_mask
            post_contact_evidence = {
                "all_samples_finite": false_mask,
                "all_samples_collision_free": false_mask,
                "all_samples_bilateral_proxy_contact": false_mask,
                "all_samples_drives_disabled": false_mask,
                "all_samples_construction_drives_enabled": false_mask,
                "all_samples_drive_state_expected": false_mask,
                "required_construction_drives_enabled": require_construction_drives_enabled,
                "any_contact_overflow": False,
                "invalid_contact_pairs": [],
                "maximum_cable_speed_m_s": torch.zeros(self.env.num_envs, device=self.device),
                "maximum_plug_linear_speed_m_s": torch.zeros(self.env.num_envs, device=self.device),
                "maximum_plug_angular_speed_rad_s": torch.zeros(self.env.num_envs, device=self.device),
                "final_cable_speed_m_s": torch.zeros(self.env.num_envs, device=self.device),
                "final_plug_linear_speed_m_s": torch.zeros(self.env.num_envs, device=self.device),
                "final_plug_angular_speed_rad_s": torch.zeros(self.env.num_envs, device=self.device),
                "final_arm_joint_speed_rad_s": torch.zeros(self.env.num_envs, device=self.device),
                "final_finger_joint_speed_m_s": torch.zeros(self.env.num_envs, device=self.device),
            }
        grasp = grasp_metrics(self.env, self.closed_finger_target)
        collision = collision_metrics(self.env)
        if collision.contact_overflow:
            raise RuntimeError("Global contact-buffer overflow at the acquisition result boundary.")
        final_distance = torch.linalg.vector_norm(
            self.env.tcp_pose_e()[:, :3] - self.env.plug_grasp_position_e(),
            dim=-1,
        )
        final_tcp_position = self.env.tcp_pose_e()[:, :3].clone()
        valid = (
            approach_valid
            & lane_hold.active_mask
            & (final_distance <= self.cfg.tcp_compensation_tolerance_m)
            & grasp.valid
            & collision.valid
            & post_contact_valid
        )
        lane_hold.deactivate(~valid, reason="acquisition-result-validation")
        contact_arm_target = lane_hold.last_sent_arm_target
        approach_evidence["abort_reason"] = "+".join(lane_hold.reason_masks) or None
        return (
            contact_arm_target,
            valid,
            {
                "clearance_ik_valid": clearance_ik_valid,
                "preclose_collision": preclose_collision,
                "preclose_error": preclose_error,
                "clearance_target_position": clearance_position,
                "clearance_tcp_position": clearance_tcp_position,
                "clearance_arm_target_bias": clearance_arm_target_bias,
                "open_descent_ik_valid": open_descent_ik_valid,
                "open_descent_collision_valid": open_descent_collision_valid,
                "maximum_open_descent_tcp_error": maximum_descent_error,
                "ik_diagnostics": ik_diagnostics,
                "contact_preclose_collision": contact_preclose_collision,
                "contact_position_error": contact_position_error,
                "contact_target_position": exact_position,
                "contact_tcp_position": contact_tcp_position,
                "final_distance": final_distance,
                "final_tcp_position": final_tcp_position,
                "grasp": grasp,
                "collision": collision,
                "open_approach": approach_evidence,
                "post_contact_settle": post_contact_evidence,
            },
        )

    @torch.inference_mode()
    def _acquire_prepositioned_current_plug(
        self,
        arm_seed: torch.Tensor,
        reference_plug: torch.Tensor,
        *,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Acquire from a proven live +45 mm clearance without running the broad approach route."""
        initial_active = torch.as_tensor(active_mask, device=self.device, dtype=torch.bool)
        ik_count_before = self._ik_solve_call_count
        with _AdvanceCounter(self.env) as counter:
            with _PerLaneTargetHold(
                self.env,
                initial_active,
                arm_seed,
                self.open_finger_q,
            ) as lane_hold:
                arm_target, valid, evidence = self._acquire_prepositioned_current_plug_per_lane(
                    arm_seed,
                    reference_plug,
                    lane_hold=lane_hold,
                )
                lane_hold.deactivate(~valid, reason="local-acquisition-final-validation")
                valid &= lane_hold.active_mask
                evidence["lane_failure_masks"] = lane_hold.reason_masks
                evidence["last_arm_target"] = lane_hold.last_sent_arm_target
                evidence["last_finger_target"] = lane_hold.last_sent_finger_target
                arm_target = lane_hold.last_sent_arm_target
        evidence["local_acquisition_call_count"] = 1
        evidence["advance_call_count"] = counter.advance_call_count
        evidence["control_step_count"] = counter.control_step_count
        evidence["ik_solve_call_count"] = self._ik_solve_call_count - ik_count_before
        return arm_target, valid, evidence

    def _acquire_prepositioned_current_plug_per_lane(
        self,
        arm_seed: torch.Tensor,
        reference_plug: torch.Tensor,
        *,
        lane_hold: _PerLaneTargetHold,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Run the local preflight, 2 mm descent, physical close, and post-contact proof."""
        reference_plug = torch.as_tensor(reference_plug, device=self.device, dtype=torch.float32).clone()
        exact_position, orientation = self._desired_tcp_pose(reference_plug)
        clearance_position = exact_position.clone()
        clearance_position[:, 2] += self.cfg.grasp_open_clearance_m
        active_before_preflight = lane_hold.active_mask
        actual_tcp = self.env.tcp_pose_e()
        preflight_clearance_position_error = torch.linalg.vector_norm(
            actual_tcp[:, :3] - clearance_position,
            dim=-1,
        )
        preflight_clearance_orientation_error = math_utils.quat_error_magnitude(
            actual_tcp[:, 3:7],
            orientation,
        )
        fallback_orientation = torch.zeros_like(orientation)
        fallback_orientation[:, 3] = 1.0
        fallback_orientation = torch.where(
            torch.isfinite(actual_tcp[:, 3:7]).all(dim=-1, keepdim=True),
            actual_tcp[:, 3:7],
            fallback_orientation,
        )
        preflight = self._solve_ik(
            torch.where(
                active_before_preflight[:, None],
                clearance_position,
                torch.nan_to_num(actual_tcp[:, :3]),
            ),
            torch.where(active_before_preflight[:, None], orientation, fallback_orientation),
            torch.where(
                active_before_preflight[:, None],
                self.open_finger_q,
                lane_hold.last_sent_finger_target,
            ),
            arm_seed=arm_seed,
        )
        preflight_raw_delta = torch.abs(preflight.arm_q - arm_seed).amax(dim=-1)
        preflight_valid = (
            active_before_preflight
            & preflight.valid
            & torch.isfinite(preflight.arm_q).all(dim=-1)
            & joint_limit_mask(self.env, preflight.arm_q, margin=0.02)
            & joint_limit_mask(self.env, arm_seed, margin=0.02)
            & (preflight_raw_delta <= self.cfg.grasp_near_maximum_raw_ik_joint_step_rad)
            & (preflight_clearance_position_error <= _LOCAL_PICKUP_CLEARANCE_TRANSLATION_TOLERANCE_M)
            & (preflight_clearance_orientation_error <= _LOCAL_PICKUP_CLEARANCE_ROTATION_TOLERANCE_RAD)
        )
        lane_hold.deactivate(~preflight_valid, reason="local-acquisition-raw-bias-preflight")
        raw_arm = torch.where(preflight_valid[:, None], preflight.arm_q, arm_seed)
        arm_target_bias = arm_seed - raw_arm
        contact_arm_target = lane_hold.last_sent_arm_target
        ik_diagnostics: list[dict[str, Any]] = []
        true_mask = torch.ones(self.env.num_envs, device=self.device, dtype=torch.bool)
        false_mask = torch.zeros_like(true_mask)
        approach_evidence: dict[str, Any] = {
            "all_samples_collision_free": true_mask.clone(),
            "all_samples_zero_proxy_contacts": true_mask.clone(),
            "all_samples_drives_disabled": true_mask.clone(),
            "all_samples_construction_drives_enabled": false_mask.clone(),
            "all_samples_drive_state_expected": true_mask.clone(),
            "required_construction_drives_enabled": False,
            "all_samples_task_state_valid": true_mask.clone(),
            "maximum_plug_position_drift_m": torch.zeros(self.env.num_envs, device=self.device),
            "maximum_left_proxy_contact_count": torch.zeros(
                self.env.num_envs,
                device=self.device,
                dtype=torch.long,
            ),
            "maximum_right_proxy_contact_count": torch.zeros(
                self.env.num_envs,
                device=self.device,
                dtype=torch.long,
            ),
            "any_contact_overflow": False,
            "invalid_contact_pairs": [],
            "samples": 0,
            "abort_reason": None,
        }
        original_advance = self.env.advance

        def sample_open_approach() -> None:
            sample_q, sample_qd = self.env.read_task_state()
            sample_collision = collision_metrics(self.env, require_bilateral_grasp=False)
            if sample_collision.contact_overflow:
                raise RuntimeError("Global contact-buffer overflow during local open-gripper acquisition.")
            plug_drift = torch.linalg.vector_norm(
                sample_q[:, self.plug_index, :3] - reference_plug[:, :3],
                dim=-1,
            )
            zero_proxy = (sample_collision.left_grasp_contact_count == 0) & (
                sample_collision.right_grasp_contact_count == 0
            )
            drives_disabled = ~self._drive_enabled() & ~self._orientation_hold_enabled()
            state_valid = task_state_is_finite_and_normalized(sample_q, sample_qd)
            if not bool(drives_disabled.all()):
                raise RuntimeError("Construction-drive state changed during local open-gripper acquisition.")
            approach_evidence["samples"] += 1
            approach_evidence["all_samples_collision_free"] &= sample_collision.valid
            approach_evidence["all_samples_zero_proxy_contacts"] &= zero_proxy
            approach_evidence["all_samples_drives_disabled"] &= drives_disabled
            approach_evidence["all_samples_drive_state_expected"] &= drives_disabled
            approach_evidence["all_samples_task_state_valid"] &= state_valid
            approach_evidence["maximum_plug_position_drift_m"] = torch.maximum(
                approach_evidence["maximum_plug_position_drift_m"],
                plug_drift,
            )
            approach_evidence["maximum_left_proxy_contact_count"] = torch.maximum(
                approach_evidence["maximum_left_proxy_contact_count"],
                sample_collision.left_grasp_contact_count,
            )
            approach_evidence["maximum_right_proxy_contact_count"] = torch.maximum(
                approach_evidence["maximum_right_proxy_contact_count"],
                sample_collision.right_grasp_contact_count,
            )
            approach_evidence["any_contact_overflow"] |= sample_collision.contact_overflow
            for pair in sample_collision.invalid_contact_pairs:
                if pair not in approach_evidence["invalid_contact_pairs"]:
                    approach_evidence["invalid_contact_pairs"].append(pair)
            lane_hold.deactivate(~sample_collision.valid, reason="local-open-approach-collision")
            lane_hold.deactivate(~zero_proxy, reason="local-open-approach-proxy-contact")
            lane_hold.deactivate(~state_valid, reason="local-open-approach-invalid-task-state")
            lane_hold.deactivate(
                plug_drift > self.cfg.maximum_open_approach_plug_drift_m,
                reason="local-open-approach-plug-drift",
            )

        def guarded_advance(duration_s: float, update=None, *, post_step=None):
            def combined_post_step(step: int, steps: int, progress: float) -> None:
                if post_step is not None:
                    post_step(step, steps, progress)
                sample_open_approach()

            return original_advance(duration_s, update, post_step=combined_post_step)

        preclose_collision = collision_metrics(self.env, require_bilateral_grasp=False)
        preclose_error = torch.linalg.vector_norm(self.env.tcp_pose_e()[:, :3] - clearance_position, dim=-1)
        open_descent_ik_valid = preflight_valid.clone()
        open_descent_collision_valid = lane_hold.active_mask
        maximum_descent_error = torch.zeros(self.env.num_envs, device=self.device)
        near_command = clearance_position.clone()
        previous_waypoint_position = clearance_position.clone()
        local_steps = max(1, math.ceil(self.cfg.grasp_open_clearance_m / _LOCAL_PICKUP_DESCENT_STEP_M))
        self.env.advance = guarded_advance
        try:
            for waypoint in range(1, local_steps + 1):
                waypoint_position = torch.lerp(clearance_position, exact_position, waypoint / local_steps)
                near_command += waypoint_position - previous_waypoint_position
                diagnostic_label = f"local_open_descent_{waypoint:02d}_of_{local_steps:02d}"
                raw_arm, contact_arm_target, waypoint_valid = self._move_tcp_with_fixed_arm_bias(
                    near_command,
                    orientation,
                    self.open_finger_q,
                    raw_arm_seed=raw_arm,
                    arm_target_bias=arm_target_bias,
                    duration_s=self.cfg.grasp_descent_waypoint_motion_s,
                    diagnostic_label=diagnostic_label,
                    diagnostics=ik_diagnostics,
                    compensation_iterations=0,
                    require_tracking_tolerance=False,
                    settle_s=self.cfg.grasp_descent_waypoint_settle_s,
                    maximum_raw_ik_joint_step_rad=self.cfg.grasp_near_maximum_raw_ik_joint_step_rad,
                    lane_hold=lane_hold,
                    failure_reason="local-open-descent-ik-continuity",
                )
                waypoint_collision = collision_metrics(self.env, require_bilateral_grasp=False)
                if waypoint_collision.contact_overflow:
                    raise RuntimeError("Global contact-buffer overflow during local open descent.")
                waypoint_error = torch.linalg.vector_norm(
                    self.env.tcp_pose_e()[:, :3] - waypoint_position,
                    dim=-1,
                )
                lane_hold.deactivate(~waypoint_valid, reason="local-open-descent-ik-continuity")
                contact_arm_target = lane_hold.last_sent_arm_target
                recovery_needed = lane_hold.active_mask & (waypoint_error > self.cfg.tcp_compensation_tolerance_m)
                if bool(recovery_needed.any()):
                    self.env.set_robot_targets(contact_arm_target, self.open_finger_q)
                    self.env.advance(self.cfg.grasp_descent_tracking_recovery_s)
                    waypoint_collision = collision_metrics(self.env, require_bilateral_grasp=False)
                    if waypoint_collision.contact_overflow:
                        raise RuntimeError("Global contact-buffer overflow during local descent recovery.")
                    waypoint_error = torch.linalg.vector_norm(
                        self.env.tcp_pose_e()[:, :3] - waypoint_position,
                        dim=-1,
                    )
                correction_count = 0
                while (
                    bool((lane_hold.active_mask & (waypoint_error > self.cfg.tcp_compensation_tolerance_m)).any())
                    and correction_count < self.cfg.grasp_near_correction_max_iterations
                ):
                    correction_needed = lane_hold.active_mask & (waypoint_error > self.cfg.tcp_compensation_tolerance_m)
                    error_vector = waypoint_position - self.env.tcp_pose_e()[:, :3]
                    correction_norm = torch.linalg.vector_norm(error_vector, dim=-1)
                    correction = error_vector * torch.clamp(
                        self.cfg.grasp_near_correction_step_m / correction_norm[:, None].clamp_min(1.0e-9),
                        max=1.0,
                    )
                    near_command += torch.where(
                        correction_needed[:, None],
                        correction,
                        torch.zeros_like(correction),
                    )
                    correction_count += 1
                    raw_arm, contact_arm_target, correction_valid = self._move_tcp_with_fixed_arm_bias(
                        near_command,
                        orientation,
                        self.open_finger_q,
                        raw_arm_seed=raw_arm,
                        arm_target_bias=arm_target_bias,
                        duration_s=self.cfg.grasp_descent_waypoint_motion_s,
                        diagnostic_label=f"{diagnostic_label}_correction_{correction_count:02d}",
                        diagnostics=ik_diagnostics,
                        compensation_iterations=0,
                        require_tracking_tolerance=False,
                        settle_s=self.cfg.grasp_descent_waypoint_settle_s,
                        maximum_raw_ik_joint_step_rad=self.cfg.grasp_near_maximum_raw_ik_joint_step_rad,
                        lane_hold=lane_hold,
                        failure_reason="local-open-descent-correction-ik-continuity",
                    )
                    waypoint_valid &= correction_valid
                    lane_hold.deactivate(
                        ~correction_valid,
                        reason="local-open-descent-correction-ik-continuity",
                    )
                    contact_arm_target = lane_hold.last_sent_arm_target
                    waypoint_collision = collision_metrics(self.env, require_bilateral_grasp=False)
                    if waypoint_collision.contact_overflow:
                        raise RuntimeError("Global contact-buffer overflow during local descent correction.")
                    waypoint_error = torch.linalg.vector_norm(
                        self.env.tcp_pose_e()[:, :3] - waypoint_position,
                        dim=-1,
                    )
                waypoint_tracking_valid = waypoint_error <= self.cfg.tcp_compensation_tolerance_m
                waypoint_valid &= waypoint_tracking_valid
                open_descent_ik_valid &= waypoint_valid
                open_descent_collision_valid &= waypoint_collision.valid
                maximum_descent_error = torch.maximum(maximum_descent_error, waypoint_error)
                lane_hold.deactivate(~waypoint_collision.valid, reason="local-open-descent-collision")
                lane_hold.deactivate(~waypoint_tracking_valid, reason="local-open-descent-tracking")
                contact_arm_target = lane_hold.last_sent_arm_target
                previous_waypoint_position = waypoint_position
        finally:
            self.env.advance = original_advance

        contact_preclose_collision = collision_metrics(self.env, require_bilateral_grasp=False)
        contact_position_error = torch.linalg.vector_norm(
            self.env.tcp_pose_e()[:, :3] - exact_position,
            dim=-1,
        )
        if preclose_collision.contact_overflow or contact_preclose_collision.contact_overflow:
            raise RuntimeError("Global contact-buffer overflow at the local acquisition pre-close boundary.")
        approach_valid = (
            lane_hold.active_mask
            & preflight_valid
            & preclose_collision.valid
            & open_descent_ik_valid
            & open_descent_collision_valid
            & contact_preclose_collision.valid
            & (contact_position_error <= self.cfg.tcp_compensation_tolerance_m)
            & approach_evidence["all_samples_collision_free"]
            & approach_evidence["all_samples_zero_proxy_contacts"]
            & approach_evidence["all_samples_drive_state_expected"]
            & approach_evidence["all_samples_task_state_valid"]
            & (approach_evidence["maximum_plug_position_drift_m"] <= self.cfg.maximum_open_approach_plug_drift_m)
            & (not approach_evidence["any_contact_overflow"])
        )
        lane_hold.deactivate(~approach_valid, reason="local-open-approach-final-validation")
        contact_arm_target = lane_hold.last_sent_arm_target
        if bool(lane_hold.active_mask.any()):
            self._close_gripper(contact_arm_target)
            post_contact_valid, post_contact_evidence = self._settle_acquired_plug(
                contact_arm_target,
                lane_hold=lane_hold,
            )
        else:
            post_contact_valid = false_mask
            post_contact_evidence = {
                "all_samples_finite": false_mask,
                "all_samples_collision_free": false_mask,
                "all_samples_bilateral_proxy_contact": false_mask,
                "all_samples_drives_disabled": false_mask,
                "all_samples_construction_drives_enabled": false_mask,
                "all_samples_drive_state_expected": false_mask,
                "required_construction_drives_enabled": False,
                "any_contact_overflow": False,
                "invalid_contact_pairs": [],
                "maximum_cable_speed_m_s": torch.zeros(self.env.num_envs, device=self.device),
                "maximum_plug_linear_speed_m_s": torch.zeros(self.env.num_envs, device=self.device),
                "maximum_plug_angular_speed_rad_s": torch.zeros(self.env.num_envs, device=self.device),
                "final_cable_speed_m_s": torch.zeros(self.env.num_envs, device=self.device),
                "final_plug_linear_speed_m_s": torch.zeros(self.env.num_envs, device=self.device),
                "final_plug_angular_speed_rad_s": torch.zeros(self.env.num_envs, device=self.device),
                "final_arm_joint_speed_rad_s": torch.zeros(self.env.num_envs, device=self.device),
                "final_finger_joint_speed_m_s": torch.zeros(self.env.num_envs, device=self.device),
            }
        grasp = grasp_metrics(self.env, self.closed_finger_target)
        collision = collision_metrics(self.env)
        if collision.contact_overflow:
            raise RuntimeError("Global contact-buffer overflow at the local acquisition result boundary.")
        final_distance = torch.linalg.vector_norm(
            self.env.tcp_pose_e()[:, :3] - self.env.plug_grasp_position_e(),
            dim=-1,
        )
        valid = (
            approach_valid
            & lane_hold.active_mask
            & (final_distance <= self.cfg.tcp_compensation_tolerance_m)
            & grasp.valid
            & collision.valid
            & post_contact_valid
        )
        lane_hold.deactivate(~valid, reason="local-acquisition-result-validation")
        approach_evidence["abort_reason"] = "+".join(lane_hold.reason_masks) or None
        return (
            lane_hold.last_sent_arm_target,
            valid,
            {
                "acquisition_route": "prepositioned-local-2mm-descent",
                "raw_bias_preflight_valid": preflight_valid,
                "raw_bias_preflight_maximum_joint_delta_rad": preflight_raw_delta,
                "raw_bias_preflight_clearance_position_error_m": preflight_clearance_position_error,
                "raw_bias_preflight_clearance_orientation_error_rad": preflight_clearance_orientation_error,
                "clearance_ik_valid": preflight_valid,
                "preclose_collision": preclose_collision,
                "preclose_error": preclose_error,
                "clearance_target_position": clearance_position,
                "clearance_tcp_position": actual_tcp[:, :3].clone(),
                "clearance_arm_target_bias": arm_target_bias,
                "local_descent_waypoint_count": local_steps,
                "local_descent_maximum_translation_step_m": _LOCAL_PICKUP_DESCENT_STEP_M,
                "open_descent_ik_valid": open_descent_ik_valid,
                "open_descent_collision_valid": open_descent_collision_valid,
                "maximum_open_descent_tcp_error": maximum_descent_error,
                "ik_diagnostics": ik_diagnostics,
                "contact_preclose_collision": contact_preclose_collision,
                "contact_position_error": contact_position_error,
                "contact_target_position": exact_position,
                "contact_tcp_position": self.env.tcp_pose_e()[:, :3].clone(),
                "final_distance": final_distance,
                "final_tcp_position": self.env.tcp_pose_e()[:, :3].clone(),
                "grasp": grasp,
                "collision": collision,
                "open_approach": approach_evidence,
                "post_contact_settle": post_contact_evidence,
            },
        )

    def _sample_scene(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.env.cfg
        socket_lower = torch.tensor(cfg.socket_position_lower, device=self.device)
        socket_upper = torch.tensor(cfg.socket_position_upper, device=self.device)
        pickup_lower = torch.tensor(cfg.pickup_position_lower, device=self.device)
        pickup_upper = torch.tensor(cfg.pickup_position_upper, device=self.device)
        assembly_position = socket_lower + torch.rand(
            (self.env.num_envs, 3), device=self.device, generator=self.random
        ) * (socket_upper - socket_lower)
        socket_yaw = cfg.socket_yaw_range[0] + torch.rand(
            self.env.num_envs, device=self.device, generator=self.random
        ) * (cfg.socket_yaw_range[1] - cfg.socket_yaw_range[0])
        socket_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(socket_yaw), torch.zeros_like(socket_yaw), socket_yaw
        )
        socket_position = assembly_position + math_utils.quat_apply(socket_quat, self.socket_local_position)
        socket_orientation = math_utils.quat_mul(socket_quat, self.socket_local_orientation)
        socket_pose = torch.cat((socket_position, socket_orientation), dim=-1)

        pickup_position = pickup_lower + torch.rand(
            (self.env.num_envs, 3), device=self.device, generator=self.random
        ) * (pickup_upper - pickup_lower)
        for _ in range(16):
            too_close = (
                torch.linalg.vector_norm(pickup_position[:, :2] - socket_position[:, :2], dim=-1)
                < cfg.minimum_pickup_socket_distance
            )
            if not bool(too_close.any()):
                break
            resampled = pickup_lower + torch.rand((self.env.num_envs, 3), device=self.device, generator=self.random) * (
                pickup_upper - pickup_lower
            )
            pickup_position = torch.where(too_close[:, None], resampled, pickup_position)
        pickup_yaw = cfg.pickup_yaw_range[0] + torch.rand(
            self.env.num_envs, device=self.device, generator=self.random
        ) * (cfg.pickup_yaw_range[1] - cfg.pickup_yaw_range[0])
        pickup_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(pickup_yaw), torch.zeros_like(pickup_yaw), pickup_yaw
        )
        return socket_pose, torch.cat((pickup_position, pickup_quat), dim=-1)

    @torch.inference_mode()
    def run_diagnostic_pickup_once(self) -> dict[str, Any]:
        """Time exactly one randomized production pickup attempt without producing rows."""
        active_mask = torch.ones(self.env.num_envs, device=self.device, dtype=torch.bool)
        ik_solve_count_before = self._ik_solve_call_count
        advance_call_count = 0
        control_step_count = 0
        original_advance = self.env.advance
        env_dict = getattr(self.env, "__dict__", {})
        had_instance_advance = "advance" in env_dict
        instance_advance = env_dict.get("advance")

        def counted_advance(duration_s: float, update=None, *, post_step=None):
            nonlocal advance_call_count, control_step_count
            steps = original_advance(duration_s, update, post_step=post_step)
            advance_call_count += 1
            control_step_count += int(steps)
            return steps

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        self.env.advance = counted_advance
        pickup_evidence: dict[str, Any] = {}
        try:
            socket_pose, pickup_pose = self._sample_scene()
            self._construct_pickup(
                socket_pose,
                pickup_pose,
                acquire=True,
                active_mask=active_mask,
                diagnostic_evidence=pickup_evidence,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed_wall_time_s = time.perf_counter() - started
        finally:
            if had_instance_advance:
                self.env.advance = instance_advance
            else:
                del self.env.advance

        ik_solve_count_after = self._ik_solve_call_count
        survival_mask = pickup_evidence["survival_mask"]
        survivor_count = sum(survival_mask)
        advance_dt = getattr(self.env, "advance_dt", None)
        staging_counts = pickup_evidence.get("construction_robot_staging", {})
        local_counts = pickup_evidence.get("drive_free_local_alignment", {}).get("local_acquisition_counts") or {}
        return {
            "mode": "diagnostic-pickup-only",
            "attempted_batch_count": 1,
            "sampled_scene_count": 1,
            "construct_pickup_call_count": 1,
            "batch_size": self.env.num_envs,
            "seed": self.cfg.seed,
            "elapsed_wall_time_s": elapsed_wall_time_s,
            "ik_solve_call_count_before": ik_solve_count_before,
            "ik_solve_call_count_after": ik_solve_count_after,
            "ik_solve_call_delta": ik_solve_count_after - ik_solve_count_before,
            "advance_call_count": advance_call_count,
            "control_step_count": control_step_count,
            "construction_staging_advance_call_count": staging_counts.get("advance_call_count"),
            "construction_staging_control_step_count": staging_counts.get("control_step_count"),
            "construction_staging_ik_solve_call_count": staging_counts.get("ik_solve_call_count"),
            "local_acquisition_call_count": local_counts.get("call_count"),
            "local_acquisition_advance_call_count": local_counts.get("advance_call_count"),
            "local_acquisition_control_step_count": local_counts.get("control_step_count"),
            "local_acquisition_ik_solve_call_count": local_counts.get("ik_solve_call_count"),
            "local_descent_waypoint_count": local_counts.get("descent_waypoint_count"),
            "simulated_time_s": None if advance_dt is None else control_step_count * float(advance_dt),
            "survivor_count": survivor_count,
            "yield_fraction": survivor_count / self.env.num_envs,
            "sampled_socket_pose": socket_pose.detach().cpu().tolist(),
            "sampled_pickup_pose": pickup_pose.detach().cpu().tolist(),
            **pickup_evidence,
        }

    @torch.inference_mode()
    def run_diagnostic_phase0_transport_once(
        self,
        canonical_goal: Mapping[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Run one certificate-backed production pickup and phase-0 transport without producing rows."""
        ik_solve_count_before = self._ik_solve_call_count
        advance_call_count = 0
        control_step_count = 0
        original_advance = self.env.advance
        env_dict = getattr(self.env, "__dict__", {})
        had_instance_advance = "advance" in env_dict
        instance_advance = env_dict.get("advance")

        def counted_advance(duration_s: float, update=None, *, post_step=None):
            nonlocal advance_call_count, control_step_count
            steps = original_advance(duration_s, update, post_step=post_step)
            advance_call_count += 1
            control_step_count += int(steps)
            return steps

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        self.env.advance = counted_advance
        pickup_evidence: dict[str, Any] = {}
        try:
            self.env.restore_default_task()
            socket_pose, pickup_pose = self._sample_scene()
            diagnostic_goal_q, _, _, goal_ik_valid = self._row_goal(
                dict(canonical_goal),
                socket_pose,
            )
            goal_ik_solve_call_count = self._ik_solve_call_count - ik_solve_count_before
            pickup_arm_target, pickup_finger_target, pickup_valid = self._construct_pickup(
                socket_pose,
                pickup_pose,
                acquire=True,
                active_mask=goal_ik_valid,
                diagnostic_evidence=pickup_evidence,
            )
            pickup_advance_call_count = advance_call_count
            pickup_control_step_count = control_step_count
            pickup_ik_solve_call_count = self._ik_solve_call_count - ik_solve_count_before - goal_ik_solve_call_count
            _, _, phase0_valid = self._realize_phase(
                0,
                pickup_pose,
                diagnostic_goal_q,
                pickup_arm_target,
                pickup_finger_target=pickup_finger_target,
                active_mask=pickup_valid,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed_wall_time_s = time.perf_counter() - started
        finally:
            if had_instance_advance:
                self.env.advance = instance_advance
            else:
                del self.env.advance

        ik_solve_count_after = self._ik_solve_call_count
        pickup_survivor_count = int(pickup_valid.sum().item())
        phase0_survivor_count = int(phase0_valid.sum().item())
        advance_dt = getattr(self.env, "advance_dt", None)
        return {
            "mode": "diagnostic-phase0-transport-only",
            "attempted_batch_count": 1,
            "sampled_scene_count": 1,
            "construct_pickup_call_count": 1,
            "realize_phase0_call_count": 1,
            "batch_size": self.env.num_envs,
            "seed": self.cfg.seed,
            "elapsed_wall_time_s": elapsed_wall_time_s,
            "ik_solve_call_count_before": ik_solve_count_before,
            "ik_solve_call_count_after": ik_solve_count_after,
            "ik_solve_call_delta": ik_solve_count_after - ik_solve_count_before,
            "goal_ik_solve_call_count": goal_ik_solve_call_count,
            "pickup_ik_solve_call_count": pickup_ik_solve_call_count,
            "phase0_transport_ik_solve_call_count": (
                ik_solve_count_after - ik_solve_count_before - goal_ik_solve_call_count - pickup_ik_solve_call_count
            ),
            "advance_call_count": advance_call_count,
            "control_step_count": control_step_count,
            "pickup_advance_call_count": pickup_advance_call_count,
            "pickup_control_step_count": pickup_control_step_count,
            "phase0_transport_advance_call_count": advance_call_count - pickup_advance_call_count,
            "phase0_transport_control_step_count": control_step_count - pickup_control_step_count,
            "simulated_time_s": None if advance_dt is None else control_step_count * float(advance_dt),
            "pickup_valid_mask": pickup_valid.detach().cpu().tolist(),
            "pickup_survivor_count": pickup_survivor_count,
            "pickup_yield_fraction": pickup_survivor_count / self.env.num_envs,
            "phase0_realization_valid_mask": phase0_valid.detach().cpu().tolist(),
            "phase0_survivor_count": phase0_survivor_count,
            "phase0_yield_fraction": phase0_survivor_count / self.env.num_envs,
            "phase0_yield_given_pickup": (
                phase0_survivor_count / pickup_survivor_count if pickup_survivor_count else 0.0
            ),
            "goal_source": "validated-canonical-goal-certificate-rigidly-transformed-to-sampled-socket",
            "goal_is_canonical_certificate": True,
            "goal_ik_valid_mask": goal_ik_valid.detach().cpu().tolist(),
            "transport_schedule_contract": _grasped_transport_schedule_contract(self.cfg),
            "sampled_socket_pose": socket_pose.detach().cpu().tolist(),
            "sampled_pickup_pose": pickup_pose.detach().cpu().tolist(),
            "pickup_evidence": _plain_certificate_value(pickup_evidence),
            "grasped_transport_evidence": _plain_certificate_value(getattr(self, "last_grasped_motion_evidence", {})),
        }

    @torch.inference_mode()
    def run_diagnostic_recovery_phase_once(
        self,
        phase: int,
        canonical_goal: Mapping[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Run exactly one phase-1/2/4/5 batch through the complete production row path."""
        if isinstance(phase, bool) or not isinstance(phase, int) or phase not in (1, 2, 4, 5):
            raise ValueError("Recovery diagnostics support only phase 1, 2, 4, or 5.")
        if (
            self.cfg.quick
            or self.cfg.rows_per_phase != _CANONICAL_ROWS_PER_PHASE
            or self.env.num_envs != _CANONICAL_BATCH_SIZE
        ):
            raise RuntimeError("Recovery diagnostics require the exact non-quick production batch-24 shape.")
        if self._ik_solve_call_count != 0:
            raise RuntimeError("Certificate-backed recovery diagnostic requires a fresh, unadvanced row IK stream.")
        attempt_count_before = self.attempt_counts[phase]
        ik_solve_count_before = self._ik_solve_call_count
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        scripted_recovery_evidence: dict[str, Any] | None = None

        def capture_scripted_recovery_evidence(
            completed_phase: int,
            batch_index: int,
            batch_evidence: Mapping[str, Any],
        ) -> None:
            nonlocal scripted_recovery_evidence
            if completed_phase != phase or batch_index != 0 or scripted_recovery_evidence is not None:
                raise RuntimeError("Recovery diagnostic received evidence outside its requested first batch.")
            scripted_recovery_evidence = _plain_certificate_value(batch_evidence)

        def stop_after_completed_batch(
            completed_phase: int,
            batch_index: int,
            accepted_chunk: Mapping[str, torch.Tensor] | None,
        ) -> None:
            accepted_row_count = 0 if accepted_chunk is None else int(accepted_chunk["phase"].shape[0])
            raise _RecoveryDiagnosticBatchComplete(completed_phase, batch_index, accepted_row_count)

        try:
            self._generate_phase(
                phase,
                dict(canonical_goal),
                diagnostic_batch_evidence_callback=capture_scripted_recovery_evidence,
                completed_batch_callback=stop_after_completed_batch,
            )
        except _RecoveryDiagnosticBatchComplete as completed:
            if completed.phase != phase or completed.batch_index != 0:
                raise RuntimeError("Recovery diagnostic did not stop at the requested phase's first batch.") from None
            accepted_row_count = completed.accepted_row_count
        else:
            raise RuntimeError("Recovery diagnostic production path completed without a batch boundary.")

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_wall_time_s = time.perf_counter() - started
        attempt_count_delta = self.attempt_counts[phase] - attempt_count_before
        if attempt_count_delta != self.env.num_envs:
            raise RuntimeError("Recovery diagnostic did not execute exactly one full production batch.")
        if scripted_recovery_evidence is None:
            raise RuntimeError("Recovery diagnostic completed without scripted-recovery batch evidence.")
        return {
            "mode": "diagnostic-recovery-phase",
            "phase": phase,
            "phase_name": PICK_INSERT_PHASE_NAMES[phase],
            "attempted_batch_count": 1,
            "production_phase_batch_index": 0,
            "sampled_scene_count": 1,
            "row_goal_call_count": 1,
            "construct_pickup_call_count": 1,
            "realize_phase_call_count": 1,
            "cold_replay_call_count": 1,
            "phase_semantics_call_count": 1,
            "oracle_call_count": 1,
            "batch_size": self.env.num_envs,
            "seed": self.cfg.seed,
            "goal_source": "validated-canonical-goal-certificate",
            "row_rng_source": "validated-canonical-goal-certificate",
            "row_ik_initial_state": "fresh-sampler-free-single-owner-before-any-solve",
            "elapsed_wall_time_s": elapsed_wall_time_s,
            "attempt_count_delta": attempt_count_delta,
            "accepted_row_count": accepted_row_count,
            "yield_fraction": accepted_row_count / self.env.num_envs,
            "ik_solve_call_count_before": ik_solve_count_before,
            "ik_solve_call_count_after": self._ik_solve_call_count,
            "ik_solve_call_delta": self._ik_solve_call_count - ik_solve_count_before,
            "rejection_counts": dict(sorted(self.rejection_counts[phase].items())),
            "production_path": (
                "sample-scene",
                "row-goal",
                "pickup-construction",
                "phase-realization",
                "cold-replay",
                "phase-semantics",
                "scripted-recovery-oracle",
            ),
            "scripted_recovery": PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY,
            "scripted_recovery_evidence": scripted_recovery_evidence,
        }

    def _park_and_place_socket(
        self,
        socket_pose: torch.Tensor,
        *,
        hold_plug_orientation: bool = False,
    ) -> torch.Tensor:
        self.env.restore_default_task()
        default_q, default_qd = self.env.read_task_state()
        transformed_q, transformed_qd = _rigid_transform_task_state(
            default_q,
            default_qd,
            default_q[:, self.socket_index],
            socket_pose,
        )
        self.env.write_task_state(transformed_q, transformed_qd)
        self.env.write_robot_state(self.home_arm_q, self.open_finger_q, finger_target=self.open_finger_q)
        plug_target_w = transformed_q[:, self.plug_index, :3] + self.env.env_origins
        self.env.set_drive(True, plug_target_w)
        if hold_plug_orientation:
            self.env.set_orientation_hold(True, transformed_q[:, self.plug_index, 3:7])
        self.env.flush_reset_history()
        self.env.advance(
            self.cfg.robot_park_s,
            lambda _step, _steps, _progress: self.env.set_drive(True, plug_target_w),
        )
        return transformed_q

    def _canonical_goal_fixed_point(
        self,
        initial_state: dict[str, torch.Tensor],
        *,
        restore_after: bool = True,
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
        """Cold-prove one captured persistent absolute target without endpoint promotion."""
        candidate = {name: value.clone() for name, value in initial_state.items()}
        candidate_history_evidence = self._restore_state(candidate)
        assert candidate_history_evidence is not None
        start_q, start_qd = self.env.read_task_state()
        duration_s = max(
            self.cfg.goal_cold_final_replay_s,
            self.cfg.goal_cold_replay_s,
            self.cfg.goal_passive_settle_s,
            60.0,
        )
        exact, metrics = advance_exact_success_dwell(
            self.env,
            candidate["task_body_pose"],
            candidate["arm_joint_target"],
            candidate["finger_joint_target"],
            duration_s=duration_s,
            require_all_samples=True,
            sample_physical_validity=True,
            arm_target_is_absolute=True,
            plug_body_index=self.plug_index,
            latch_body_index=self.latch_index,
        )
        final_q, final_qd = self.env.read_task_state()
        final_arm_q, final_arm_qd, final_finger_q, final_finger_qd = self.env.read_robot_state()
        stored_arm_q = candidate["arm_joint_position"]
        stored_arm_target = candidate["arm_joint_target"]
        target_error_limits = torch.as_tensor(
            self.env.cfg.actions.arm_action.tracking_error_limits,
            device=self.device,
            dtype=stored_arm_q.dtype,
        )
        stored_target_error = torch.abs(stored_arm_target - stored_arm_q)
        final_target_error = torch.abs(stored_arm_target - final_arm_q)
        arm_drift = torch.abs(final_arm_q - stored_arm_q).amax(dim=-1)
        body_drift_all = torch.linalg.vector_norm(final_q[..., :3] - start_q[..., :3], dim=-1)
        maximum_body_drift, worst_body = body_drift_all.max(dim=-1)
        socket_drift = body_drift_all[:, self.socket_index]
        start_cable_speed, start_fastest = torch.linalg.vector_norm(start_qd[:, self.cable_slice, :3], dim=-1).max(
            dim=-1
        )
        final_cable_speed, final_fastest = torch.linalg.vector_norm(final_qd[:, self.cable_slice, :3], dim=-1).max(
            dim=-1
        )
        authored_target_e = wp.to_torch(self.env.rj45_runtime.default_goal_target_w) - self.env.env_origins
        authored_orientation = wp.to_torch(self.env.rj45_runtime.default_orientation_target_w)
        start_seat_error = torch.linalg.vector_norm(
            start_q[:, self.plug_index, :3] - authored_target_e,
            dim=-1,
        )
        final_seat_error = torch.linalg.vector_norm(
            final_q[:, self.plug_index, :3] - authored_target_e,
            dim=-1,
        )
        start_plug_tilt = math_utils.quat_error_magnitude(
            start_q[:, self.plug_index, 3:7],
            authored_orientation,
        )
        final_plug_tilt = math_utils.quat_error_magnitude(
            final_q[:, self.plug_index, 3:7],
            authored_orientation,
        )
        start_latch_angle = plug_relative_latch_angle(
            start_q,
            plug_body_index=self.plug_index,
            latch_body_index=self.latch_index,
        )
        final_latch_angle = plug_relative_latch_angle(
            final_q,
            plug_body_index=self.plug_index,
            latch_body_index=self.latch_index,
        )
        grasp = grasp_metrics(self.env, candidate["finger_joint_target"], retaining_grasp=True)
        collision = collision_metrics(self.env)
        stored_arm_speed = torch.abs(candidate["arm_joint_velocity"]).amax(dim=-1)
        final_arm_speed = torch.abs(final_arm_qd).amax(dim=-1)
        stored_finger_speed = torch.abs(candidate["finger_joint_velocity"]).amax(dim=-1)
        final_finger_speed = torch.abs(final_finger_qd).amax(dim=-1)
        history_applied = self._vbd_pose_history_applied_mask(candidate_history_evidence)
        valid = (
            task_state_is_finite_and_normalized(start_q, start_qd)
            & task_state_is_finite_and_normalized(final_q, final_qd)
            & grasp.valid
            & collision.valid
            & exact
            & metrics["all_samples_collision_free"]
            & metrics["all_samples_bilateral_grasp"]
            & metrics["all_samples_proxy_bilateral_contact"]
            & metrics["all_samples_finite"]
            & metrics["all_samples_arm_target_tracking_bounded"]
            & ~torch.as_tensor(metrics["any_contact_overflow"], device=self.device)
            & ~metrics["any_arm_target_clamped"]
            & (metrics["maximum_arm_target_clamp_delta"] <= 1.0e-7)
            & (metrics["maximum_arm_target_drift"] <= 1.0e-7)
            & joint_limit_mask(self.env, stored_arm_q)
            & joint_limit_mask(self.env, stored_arm_target)
            & joint_limit_mask(self.env, final_arm_q)
            & (stored_target_error <= target_error_limits).all(dim=-1)
            & (final_target_error <= target_error_limits).all(dim=-1)
            & (arm_drift <= self.cfg.maximum_goal_arm_drift_rad)
            & (maximum_body_drift <= self.cfg.maximum_goal_body_drift_m)
            & (metrics["maximum_body_excursion"] <= self.cfg.maximum_goal_body_drift_m)
            & (socket_drift <= self.cfg.maximum_socket_drift_m)
            & (start_cable_speed <= self.cfg.maximum_goal_cable_speed_m_s)
            & (final_cable_speed <= self.cfg.maximum_goal_cable_speed_m_s)
            & (metrics["maximum_cable_linear_speed"] <= self.cfg.maximum_goal_cable_speed_m_s)
            & (stored_arm_speed <= self.cfg.maximum_goal_arm_joint_speed_rad_s)
            & (final_arm_speed <= self.cfg.maximum_goal_arm_joint_speed_rad_s)
            & (metrics["maximum_arm_joint_speed"] <= self.cfg.maximum_goal_arm_joint_speed_rad_s)
            & (stored_finger_speed <= self.cfg.maximum_goal_finger_joint_speed_m_s)
            & (final_finger_speed <= self.cfg.maximum_goal_finger_joint_speed_m_s)
            & (metrics["maximum_finger_joint_speed"] <= self.cfg.maximum_goal_finger_joint_speed_m_s)
            & (start_seat_error <= self.cfg.maximum_canonical_seat_error_m)
            & (final_seat_error <= self.cfg.maximum_canonical_seat_error_m)
            & (start_plug_tilt <= self.cfg.maximum_authored_plug_angle_rad)
            & (final_plug_tilt <= self.cfg.maximum_authored_plug_angle_rad)
            & (start_latch_angle <= PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD)
            & (final_latch_angle <= PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD)
            & (metrics["maximum_plug_relative_latch_angle"] <= PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD)
            & ~self._drive_enabled()
            & ~self._orientation_hold_enabled()
            & history_applied
        )
        evidence = {
            "cycle": 1,
            "controller_semantics": "persistent-absolute",
            "native_mjwarp_gravity_compensation": True,
            "duration_s": duration_s,
            "passed": bool(valid.all()),
            "vbd_pose_history_restore_queued": bool(
                torch.as_tensor(candidate_history_evidence["restore_queued"]).all()
            ),
            "vbd_pose_history_pending_at_queue": bool(
                torch.as_tensor(candidate_history_evidence["pending_at_queue"]).all()
            ),
            "vbd_previous_pose_queued": bool(torch.as_tensor(candidate_history_evidence["previous_pose_queued"]).all()),
            "vbd_coupling_previous_pose_queued": bool(
                torch.as_tensor(candidate_history_evidence["coupling_previous_pose_queued"]).all()
            ),
            "vbd_pose_history_applied_exactly_once": bool(history_applied.all()),
            "vbd_pose_history_failed": bool(torch.as_tensor(candidate_history_evidence["failed"]).any()),
            "vbd_pose_history_superseded": bool(torch.as_tensor(candidate_history_evidence["superseded"]).any()),
            "vbd_pose_history_pending_after_first_solve": bool(
                torch.as_tensor(candidate_history_evidence["pending_after_first_solve"]).any()
            ),
            "vbd_pose_history_minimum_application_count_delta": int(
                torch.as_tensor(candidate_history_evidence["application_count_delta"]).min()
            ),
            "vbd_pose_history_maximum_application_count_delta": int(
                torch.as_tensor(candidate_history_evidence["application_count_delta"]).max()
            ),
            "vbd_pose_history_expected_body_count": int(
                torch.as_tensor(candidate_history_evidence["expected_body_count"]).min()
            ),
            "vbd_pose_history_minimum_body_application_count_delta": int(
                torch.as_tensor(candidate_history_evidence["body_application_count_delta"]).min()
            ),
            "vbd_pose_history_maximum_body_application_count_delta": int(
                torch.as_tensor(candidate_history_evidence["body_application_count_delta"]).max()
            ),
            "vbd_pose_history_generation": int(candidate_history_evidence["generation"]),
            "vbd_pose_history_body_order_exact": candidate_history_evidence["body_order_exact"] is True,
            "vbd_pose_history_world_order_exact": candidate_history_evidence["world_order_exact"] is True,
            "vbd_pose_history_entry_name": candidate_history_evidence["entry_name"],
            "vbd_pose_history_body_count": candidate_history_evidence["body_count"],
            "maximum_absolute_target_drift_rad": float(metrics["maximum_arm_target_drift"].max()),
            "maximum_target_clamp_delta_rad": float(metrics["maximum_arm_target_clamp_delta"].max()),
            "all_samples_target_tracking_bounded": bool(metrics["all_samples_arm_target_tracking_bounded"].all()),
            "maximum_target_tracking_error_by_joint_rad": metrics["maximum_arm_target_tracking_error_by_joint"]
            .amax(dim=0)
            .tolist(),
            "maximum_arm_drift_rad": float(arm_drift.max()),
            "maximum_body_drift_m": float(maximum_body_drift.max()),
            "maximum_socket_drift_m": float(socket_drift.max()),
            "maximum_sampled_body_excursion_m": float(metrics["maximum_body_excursion"].max()),
            "maximum_cable_speed_m_s": float(metrics["maximum_cable_linear_speed"].max()),
            "maximum_arm_joint_speed_rad_s": float(metrics["maximum_arm_joint_speed"].max()),
            "maximum_finger_joint_speed_m_s": float(metrics["maximum_finger_joint_speed"].max()),
            "maximum_authored_seat_error_m": float(torch.maximum(start_seat_error, final_seat_error).max()),
            "maximum_authored_plug_angle_rad": float(torch.maximum(start_plug_tilt, final_plug_tilt).max()),
            "maximum_latch_angle_rad": float(metrics["maximum_plug_relative_latch_angle"].max()),
            "all_samples_collision_free": bool(metrics["all_samples_collision_free"].all()),
            "all_samples_bilateral_grasp": bool(metrics["all_samples_bilateral_grasp"].all()),
            "all_samples_proxy_bilateral_contact": bool(metrics["all_samples_proxy_bilateral_contact"].all()),
            "all_samples_finite": bool(metrics["all_samples_finite"].all()),
            "contact_overflow": bool(metrics["any_contact_overflow"]),
            "invalid_contact_pairs": metrics["sampled_invalid_contact_pairs"],
            "worst_body_index_by_world": worst_body.tolist(),
            "worst_body_name_by_world": [self.layout.body_names[int(index)] for index in worst_body],
            "start_fastest_cable_body_index_by_world": (start_fastest + self.cable_body_start).tolist(),
            "final_fastest_cable_body_index_by_world": (final_fastest + self.cable_body_start).tolist(),
        }
        print(f"[PICK-INSERT GOAL PERSISTENT COLD] {evidence}", flush=True)
        if not bool(valid.all()):
            raise RuntimeError(f"Canonical seated goal failed the 60-second persistent-target cold proof: {evidence}")
        if restore_after:
            self._restore_state(candidate)
        return candidate, [evidence]

    def _cold_goal_equilibrium_cycle(
        self,
        initial_candidate: dict[str, torch.Tensor],
        authored_seat_target_w: torch.Tensor,
        authored_plug_orientation: torch.Tensor,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Cold-rewrite one immutable-target candidate and sample its equilibrium replay."""
        candidate = {name: value.clone() for name, value in initial_candidate.items()}
        pose_history_residual = self._vbd_pose_history_residual()
        history_evidence = self._restore_state(candidate)
        assert history_evidence is not None
        reference_q = candidate["task_body_pose"].clone()
        authored_target_e = authored_seat_target_w - self.env.env_origins
        reference_q[:, self.plug_index, :3] = authored_target_e
        reference_q[:, self.plug_index, 3:7] = authored_plug_orientation
        exact, metrics = advance_exact_success_dwell(
            self.env,
            reference_q,
            candidate["arm_joint_target"],
            candidate["finger_joint_target"],
            duration_s=self.cfg.goal_cold_equilibrium_relax_s,
            require_all_samples=True,
            sample_physical_validity=True,
            arm_target_is_absolute=True,
            plug_body_index=self.plug_index,
            latch_body_index=self.latch_index,
        )
        final_q, final_qd = self.env.read_task_state()
        final_arm_q, final_arm_qd, final_finger_q, final_finger_qd = self.env.read_robot_state()
        stored_q = candidate["task_body_pose"]
        stored_qd = candidate["task_body_velocity"]
        stored_arm_q = candidate["arm_joint_position"]
        stored_arm_target = candidate["arm_joint_target"]
        target_limits = torch.as_tensor(
            self.env.cfg.actions.arm_action.tracking_error_limits,
            device=self.device,
            dtype=stored_arm_q.dtype,
        )
        stored_seat_error = torch.linalg.vector_norm(
            stored_q[:, self.plug_index, :3] - authored_target_e,
            dim=-1,
        )
        final_seat_error = torch.linalg.vector_norm(
            final_q[:, self.plug_index, :3] - authored_target_e,
            dim=-1,
        )
        stored_plug_tilt = math_utils.quat_error_magnitude(
            stored_q[:, self.plug_index, 3:7],
            authored_plug_orientation,
        )
        final_plug_tilt = math_utils.quat_error_magnitude(
            final_q[:, self.plug_index, 3:7],
            authored_plug_orientation,
        )
        stored_latch_angle = plug_relative_latch_angle(
            stored_q,
            plug_body_index=self.plug_index,
            latch_body_index=self.latch_index,
        )
        final_latch_angle = plug_relative_latch_angle(
            final_q,
            plug_body_index=self.plug_index,
            latch_body_index=self.latch_index,
        )
        stored_cable_speed = torch.linalg.vector_norm(
            stored_qd[:, self.cable_slice, :3],
            dim=-1,
        ).amax(dim=-1)
        final_cable_speed = torch.linalg.vector_norm(
            final_qd[:, self.cable_slice, :3],
            dim=-1,
        ).amax(dim=-1)
        stored_arm_speed = torch.abs(candidate["arm_joint_velocity"]).amax(dim=-1)
        stored_finger_speed = torch.abs(candidate["finger_joint_velocity"]).amax(dim=-1)
        final_arm_speed = torch.abs(final_arm_qd).amax(dim=-1)
        final_finger_speed = torch.abs(final_finger_qd).amax(dim=-1)
        stored_target_error = torch.abs(stored_arm_target - stored_arm_q)
        final_target_error = torch.abs(stored_arm_target - final_arm_q)
        final_grasp = grasp_metrics(self.env, candidate["finger_joint_target"], retaining_grasp=True)
        final_collision = collision_metrics(self.env)
        history_applied = self._vbd_pose_history_applied_mask(history_evidence)
        hard_valid = (
            history_applied
            & task_state_is_finite_and_normalized(stored_q, stored_qd)
            & task_state_is_finite_and_normalized(final_q, final_qd)
            & metrics["all_samples_collision_free"]
            & metrics["all_samples_bilateral_grasp"]
            & metrics["all_samples_proxy_bilateral_contact"]
            & metrics["all_samples_finite"]
            & metrics["all_samples_arm_target_tracking_bounded"]
            & ~torch.as_tensor(metrics["any_contact_overflow"], device=self.device)
            & ~metrics["any_arm_target_clamped"]
            & (metrics["maximum_arm_target_clamp_delta"] <= 1.0e-7)
            & (metrics["maximum_arm_target_drift"] <= 1.0e-7)
            & joint_limit_mask(self.env, stored_arm_q, margin=0.02)
            & joint_limit_mask(self.env, stored_arm_target, margin=0.02)
            & joint_limit_mask(self.env, final_arm_q, margin=0.02)
            & (stored_target_error <= target_limits).all(dim=-1)
            & (final_target_error <= target_limits).all(dim=-1)
            & ~self._drive_enabled()
            & ~self._orientation_hold_enabled()
        )
        endpoint_promotable = (
            hard_valid
            & final_grasp.valid
            & final_collision.valid
            & joint_limit_mask(self.env, final_arm_q, margin=0.02)
            & joint_limit_mask(self.env, stored_arm_target, margin=0.02)
            & (final_target_error <= target_limits).all(dim=-1)
            & (final_cable_speed <= self.cfg.maximum_goal_cable_speed_m_s)
            & (metrics["final_plug_spatial_speed"] <= self.env.cfg.success_max_plug_speed)
            & (final_arm_speed <= self.cfg.maximum_goal_arm_joint_speed_rad_s)
            & (final_finger_speed <= self.cfg.maximum_goal_finger_joint_speed_m_s)
            & (final_seat_error <= self.cfg.maximum_canonical_seat_error_m)
            & (final_plug_tilt <= self.cfg.maximum_authored_plug_angle_rad)
            & (final_latch_angle <= PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD)
        )
        converged = (
            hard_valid
            & exact
            & metrics["all_samples_success"]
            & (metrics["maximum_body_excursion"] <= self.cfg.maximum_goal_body_drift_m)
            & (metrics["maximum_cable_linear_speed"] <= self.cfg.maximum_goal_cable_speed_m_s)
            & (metrics["maximum_arm_joint_speed"] <= self.cfg.maximum_goal_arm_joint_speed_rad_s)
            & (metrics["maximum_finger_joint_speed"] <= self.cfg.maximum_goal_finger_joint_speed_m_s)
            & (metrics["maximum_plug_relative_latch_angle"] <= PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD)
            & (stored_cable_speed <= self.cfg.maximum_goal_cable_speed_m_s)
            & (final_cable_speed <= self.cfg.maximum_goal_cable_speed_m_s)
            & (stored_arm_speed <= self.cfg.maximum_goal_arm_joint_speed_rad_s)
            & (final_arm_speed <= self.cfg.maximum_goal_arm_joint_speed_rad_s)
            & (stored_finger_speed <= self.cfg.maximum_goal_finger_joint_speed_m_s)
            & (final_finger_speed <= self.cfg.maximum_goal_finger_joint_speed_m_s)
            & (stored_seat_error <= self.cfg.maximum_canonical_seat_error_m)
            & (final_seat_error <= self.cfg.maximum_canonical_seat_error_m)
            & (stored_plug_tilt <= self.cfg.maximum_authored_plug_angle_rad)
            & (final_plug_tilt <= self.cfg.maximum_authored_plug_angle_rad)
            & (stored_latch_angle <= PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD)
            & (final_latch_angle <= PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD)
        )
        endpoint_candidate = self._capture_state(
            stored_arm_target,
            candidate["finger_joint_target"],
        )
        target_bitwise_unchanged = torch.equal(
            endpoint_candidate["arm_joint_target"],
            candidate["arm_joint_target"],
        )
        if not target_bitwise_unchanged:
            raise RuntimeError("Cold endpoint promotion changed the immutable persistent absolute arm target.")
        evidence = {
            "duration_s": self.cfg.goal_cold_equilibrium_relax_s,
            "passed_hard_gates": bool(hard_valid.all()),
            "converged": bool(converged.all()),
            "endpoint_promotable": bool(endpoint_promotable.all()),
            "endpoint_target_bitwise_unchanged": target_bitwise_unchanged,
            "pre_restore_vbd_pose_history_residual": pose_history_residual,
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
            "vbd_pose_history_application_count_delta": torch.as_tensor(history_evidence["application_count_delta"])
            .detach()
            .cpu()
            .tolist(),
            "vbd_pose_history_expected_body_count": torch.as_tensor(history_evidence["expected_body_count"])
            .detach()
            .cpu()
            .tolist(),
            "vbd_pose_history_body_application_count_delta": torch.as_tensor(
                history_evidence["body_application_count_delta"]
            )
            .detach()
            .cpu()
            .tolist(),
            "vbd_pose_history_generation": int(history_evidence["generation"]),
            "maximum_absolute_target_drift_rad": float(metrics["maximum_arm_target_drift"].max()),
            "maximum_target_clamp_delta_rad": float(metrics["maximum_arm_target_clamp_delta"].max()),
            "maximum_body_excursion_m": float(metrics["maximum_body_excursion"].max()),
            "maximum_cable_speed_m_s": float(metrics["maximum_cable_linear_speed"].max()),
            "final_cable_speed_m_s": float(final_cable_speed.max()),
            "final_plug_spatial_speed": float(metrics["final_plug_spatial_speed"].max()),
            "final_arm_joint_speed_rad_s": float(final_arm_speed.max()),
            "final_finger_joint_speed_m_s": float(final_finger_speed.max()),
            "maximum_arm_joint_speed_rad_s": float(metrics["maximum_arm_joint_speed"].max()),
            "maximum_finger_joint_speed_m_s": float(metrics["maximum_finger_joint_speed"].max()),
            "maximum_signed_axial_error_m": float(metrics["maximum_signed_axial_error"].max()),
            "minimum_signed_axial_error_m": float(metrics["minimum_signed_axial_error"].min()),
            "maximum_radial_error_m": float(metrics["maximum_radial_error"].max()),
            "maximum_plug_angle_error_rad": float(metrics["maximum_plug_angle_error"].max()),
            "maximum_latch_angle_error_rad": float(metrics["maximum_latch_angle_error"].max()),
            "maximum_plug_spatial_speed": float(metrics["maximum_plug_spatial_speed"].max()),
            "maximum_authored_seat_error_m": float(torch.maximum(stored_seat_error, final_seat_error).max()),
            "maximum_authored_plug_angle_rad": float(torch.maximum(stored_plug_tilt, final_plug_tilt).max()),
            "maximum_plug_relative_latch_angle_rad": float(metrics["maximum_plug_relative_latch_angle"].max()),
            "stored_plug_relative_latch_angle_rad": float(stored_latch_angle.max()),
            "final_plug_relative_latch_angle_rad": float(final_latch_angle.max()),
            "all_samples_exact_success": bool(metrics["all_samples_success"].all()),
            "all_samples_collision_free": bool(metrics["all_samples_collision_free"].all()),
            "all_samples_bilateral_grasp": bool(metrics["all_samples_bilateral_grasp"].all()),
            "all_samples_proxy_bilateral_contact": bool(metrics["all_samples_proxy_bilateral_contact"].all()),
            "all_samples_finite": bool(metrics["all_samples_finite"].all()),
            "contact_overflow": bool(metrics["any_contact_overflow"]),
            "invalid_contact_pairs": metrics["sampled_invalid_contact_pairs"],
            "endpoint_collision_free": bool(final_collision.valid.all()),
            "endpoint_bilateral_grasp": bool(final_grasp.valid.all()),
        }
        return candidate, endpoint_candidate, hard_valid, converged, endpoint_promotable, evidence

    def _sample_canonical_goal_stage(
        self,
        *,
        goal_task_q: torch.Tensor,
        immutable_arm_target: torch.Tensor,
        baseline_q: torch.Tensor,
        authored_seat_target_e: torch.Tensor,
        authored_plug_orientation: torch.Tensor,
        tracking_limits: torch.Tensor,
        context: str,
    ) -> dict[str, Any]:
        """Sample lane-local canonical-goal gates while retaining global simulator faults."""
        self._assert_drive_disabled(context)
        task_q, task_qd = self.env.read_task_state()
        arm_q, arm_qd, finger_q, finger_qd = self.env.read_robot_state()
        collision = collision_metrics(self.env, require_bilateral_grasp=False)
        if collision.contact_overflow:
            raise RuntimeError(f"Global contact-buffer overflow during {context}.")
        grasp = grasp_metrics(self.env, self.closed_finger_target, retaining_grasp=True)
        exact = exact_success_from_state(
            self.env,
            task_q,
            task_qd,
            goal_task_q,
            plug_body_index=self.plug_index,
            latch_body_index=self.latch_index,
        )
        resolved_target, clamp_delta = runtime_persistent_arm_target(self.env, immutable_arm_target)
        tracking_error = torch.abs(resolved_target - arm_q)
        target_drift = torch.abs(resolved_target - immutable_arm_target).amax(dim=-1)
        body_excursion = torch.linalg.vector_norm(task_q[..., :3] - baseline_q[..., :3], dim=-1).amax(dim=-1)
        cable_speed = torch.linalg.vector_norm(task_qd[:, self.cable_slice, :3], dim=-1).amax(dim=-1)
        arm_speed = torch.abs(arm_qd).amax(dim=-1)
        finger_speed = torch.abs(finger_qd).amax(dim=-1)
        seat_error = torch.linalg.vector_norm(
            task_q[:, self.plug_index, :3] - authored_seat_target_e,
            dim=-1,
        )
        plug_tilt = math_utils.quat_error_magnitude(
            task_q[:, self.plug_index, 3:7],
            authored_plug_orientation,
        )
        relative_latch = plug_relative_latch_angle(
            task_q,
            plug_body_index=self.plug_index,
            latch_body_index=self.latch_index,
        )
        proxy_bilateral = _runtime_bilateral_grasp_proxy_contact_mask(
            self.env,
            collision.left_grasp_contact_count,
            collision.right_grasp_contact_count,
        )
        finite = (
            task_state_is_finite_and_normalized(task_q, task_qd)
            & torch.isfinite(arm_q).all(dim=-1)
            & torch.isfinite(arm_qd).all(dim=-1)
            & torch.isfinite(finger_q).all(dim=-1)
            & torch.isfinite(finger_qd).all(dim=-1)
            & torch.isfinite(resolved_target).all(dim=-1)
        )
        return {
            "task_q": task_q,
            "task_qd": task_qd,
            "exact": exact,
            "hard_gate_masks": {
                "finite": finite,
                "collision-free": collision.valid,
                "bilateral-grasp": grasp.valid,
                "bilateral-proxy-contact": proxy_bilateral,
                "arm-joint-limits": joint_limit_mask(self.env, arm_q, margin=0.02),
                "target-joint-limits": joint_limit_mask(self.env, resolved_target, margin=0.02),
                "target-tracking": (tracking_error <= tracking_limits).all(dim=-1),
                "target-unclamped": clamp_delta <= 1.0e-7,
                "target-immutable": target_drift <= 1.0e-7,
            },
            "exact_geometry_gate_masks": _diagnostic_exact_geometry_gate_masks(exact, self.env.cfg),
            "speed_gate_masks": {
                "plug-spatial-speed": exact.plug_spatial_speed <= self.env.cfg.success_max_plug_speed,
                "cable-speed": cable_speed <= self.cfg.maximum_goal_cable_speed_m_s,
                "arm-joint-speed": arm_speed <= self.cfg.maximum_goal_arm_joint_speed_rad_s,
                "finger-joint-speed": finger_speed <= self.cfg.maximum_goal_finger_joint_speed_m_s,
            },
            "metric_values": {
                "body_excursion_m": body_excursion,
                "plug_linear_speed_m_s": torch.linalg.vector_norm(task_qd[:, self.plug_index, :3], dim=-1),
                "plug_angular_speed_rad_s": torch.linalg.vector_norm(task_qd[:, self.plug_index, 3:6], dim=-1),
                "plug_spatial_speed": exact.plug_spatial_speed,
                "cable_speed_m_s": cable_speed,
                "arm_joint_speed_rad_s": arm_speed,
                "finger_joint_speed_m_s": finger_speed,
                "authored_seat_error_m": seat_error,
                "authored_plug_tilt_rad": plug_tilt,
                "plug_relative_latch_angle_rad": relative_latch,
                "arm_target_tracking_error_rad": tracking_error.amax(dim=-1),
                "arm_target_drift_rad": target_drift,
                "arm_target_clamp_delta_rad": clamp_delta,
            },
            "invalid_contact_pairs": collision.invalid_contact_pairs,
        }

    def _run_diagnostic_reseat_rolling_equilibrium(
        self,
        goal_task_q: torch.Tensor,
        arm_target: torch.Tensor,
        authored_seat_target_e: torch.Tensor,
        authored_plug_orientation: torch.Tensor,
        tracking_limits: torch.Tensor,
        *,
        lane_hold: _PerLaneTargetHold,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Retain lanes obtaining a fresh strict two-second dwell within four seconds."""
        maximum_duration_s = _DIAGNOSTIC_RESEAT_ROLLING_MAX_DURATION_S
        required_duration_s = _DIAGNOSTIC_RESEAT_ROLLING_DWELL_S
        maximum_steps = math.floor(maximum_duration_s / self.env.advance_dt + 1.0e-12)
        if maximum_steps < 1:
            raise RuntimeError("The rolling-equilibrium cap must contain at least one physics step.")
        baseline_q, _ = self.env.read_task_state()
        starting_mask = lane_hold.active_mask
        dwell = _PerLaneRollingDwell(required_duration_s, self.env.num_envs, self.device)
        maxima: dict[str, torch.Tensor] = {}
        invalid_contact_pairs: list[str] = []
        trace: list[dict[str, Any]] = []
        completed = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool)
        completed_step = 0

        def sample(elapsed_s: float, sample_index: int) -> torch.Tensor:
            stage = self._sample_canonical_goal_stage(
                goal_task_q=goal_task_q,
                immutable_arm_target=arm_target,
                baseline_q=baseline_q,
                authored_seat_target_e=authored_seat_target_e,
                authored_plug_orientation=authored_plug_orientation,
                tracking_limits=tracking_limits,
                context="canonical authored-reseat rolling equilibrium",
            )
            hard_gate_masks = {
                **stage["hard_gate_masks"],
                "authored-seat": (
                    stage["metric_values"]["authored_seat_error_m"] <= self.cfg.maximum_canonical_seat_error_m
                ),
                "authored-plug-tilt": (
                    stage["metric_values"]["authored_plug_tilt_rad"] <= self.cfg.maximum_authored_plug_angle_rad
                ),
                "plug-relative-latch": (
                    stage["metric_values"]["plug_relative_latch_angle_rad"]
                    <= PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD
                ),
                "body-excursion": (stage["metric_values"]["body_excursion_m"] <= self.cfg.maximum_goal_body_drift_m),
                **stage["exact_geometry_gate_masks"],
            }
            for name, valid in hard_gate_masks.items():
                lane_hold.deactivate(~valid, reason=f"canonical-rolling-{name}")
            active = lane_hold.active_mask
            if not bool(active.any()):
                failure_masks = {name: mask.tolist() for name, mask in lane_hold.reason_masks.items()}
                raise RuntimeError(
                    f"Authored reseat rolling equilibrium has zero surviving lanes: reasons={failure_masks}."
                )
            speed_valid = torch.ones_like(active)
            for valid in stage["speed_gate_masks"].values():
                speed_valid &= valid
            for name, value in stage["metric_values"].items():
                maxima[name] = value.clone() if name not in maxima else torch.maximum(maxima[name], value)
            for pair in stage["invalid_contact_pairs"]:
                if pair not in invalid_contact_pairs and len(invalid_contact_pairs) < 64:
                    invalid_contact_pairs.append(pair)
            if (
                sample_index == 0
                or sample_index == maximum_steps
                or sample_index % max(1, round(0.1 / self.env.advance_dt)) == 0
            ):
                trace.append(
                    {
                        "time_s": elapsed_s,
                        "active_lane_ids": torch.where(active)[0].detach().cpu().tolist(),
                        "speed_valid_by_lane": speed_valid.detach().cpu().tolist(),
                    }
                )
            return dwell.observe(elapsed_s, sample_index, speed_valid, active)

        completed = sample(0.0, 0)
        for step in range(1, maximum_steps + 1):
            self.env.set_robot_targets(arm_target, self.closed_finger_target)
            self.env.advance(self.env.advance_dt)
            completed = sample(step * self.env.advance_dt, step)
            completed_step = step
            if bool(completed.any()):
                lane_hold.deactivate(lane_hold.active_mask & ~completed, reason="canonical-rolling-no-fresh-dwell")
                break
        survivors = lane_hold.active_mask & completed
        if not bool(survivors.any()):
            lane_hold.deactivate(lane_hold.active_mask, reason="canonical-rolling-timeout")
            raise RuntimeError(
                "Authored reseat obtained no fresh two-second rolling equilibrium within four seconds: "
                f"dwell={dwell.evidence()}."
            )
        evidence = {
            "passed": True,
            "starting_lane_ids": torch.where(starting_mask)[0].detach().cpu().tolist(),
            "surviving_lane_ids": torch.where(survivors)[0].detach().cpu().tolist(),
            "surviving_mask": survivors.detach().cpu().tolist(),
            "maximum_duration_s": maximum_duration_s,
            "required_uninterrupted_exact_duration_s": required_duration_s,
            "actual_duration_s": completed_step * self.env.advance_dt,
            "dwell": dwell.evidence(),
            "maximum_metrics_by_lane": {name: value.detach().cpu().tolist() for name, value in maxima.items()},
            "lane_failure_masks": {name: mask.detach().cpu().tolist() for name, mask in lane_hold.reason_masks.items()},
            "invalid_contact_pairs": tuple(invalid_contact_pairs),
            "time_series": trace,
        }
        return survivors, evidence

    def _run_continuous_goal_relaxation(
        self,
        goal_task_q: torch.Tensor,
        arm_target: torch.Tensor,
        authored_seat_target_e: torch.Tensor,
        authored_plug_orientation: torch.Tensor,
        tracking_limits: torch.Tensor,
        *,
        lane_hold: _PerLaneTargetHold,
    ) -> dict[str, Any]:
        """Run 30--60 seconds of relaxation and retain lanes with a fresh low-speed dwell."""
        minimum_duration_s = self.cfg.goal_cold_equilibrium_relax_s
        maximum_duration_s = max(60.0, minimum_duration_s)
        required_duration_s = self.cfg.goal_stability_window_s
        minimum_steps = math.ceil(minimum_duration_s / self.env.advance_dt)
        maximum_steps = math.ceil(maximum_duration_s / self.env.advance_dt)
        baseline_q, _ = self.env.read_task_state()
        starting_mask = lane_hold.active_mask
        dwell = _PerLaneRollingDwell(required_duration_s, self.env.num_envs, self.device)
        maxima: dict[str, torch.Tensor] = {}
        invalid_contact_pairs: list[str] = []
        trace: list[dict[str, Any]] = []
        completed = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool)
        completed_step = 0

        def sample(elapsed_s: float, sample_index: int) -> torch.Tensor:
            stage = self._sample_canonical_goal_stage(
                goal_task_q=goal_task_q,
                immutable_arm_target=arm_target,
                baseline_q=baseline_q,
                authored_seat_target_e=authored_seat_target_e,
                authored_plug_orientation=authored_plug_orientation,
                tracking_limits=tracking_limits,
                context="canonical continuous grasp relaxation",
            )
            for name, valid in stage["hard_gate_masks"].items():
                lane_hold.deactivate(~valid, reason=f"canonical-relaxation-{name}")
            active = lane_hold.active_mask
            if not bool(active.any()):
                failure_masks = {name: mask.tolist() for name, mask in lane_hold.reason_masks.items()}
                raise RuntimeError(
                    f"Continuous canonical relaxation has zero surviving lanes: reasons={failure_masks}."
                )
            speed_valid = torch.ones_like(active)
            for valid in stage["speed_gate_masks"].values():
                speed_valid &= valid
            for name, value in stage["metric_values"].items():
                maxima[name] = value.clone() if name not in maxima else torch.maximum(maxima[name], value)
            for pair in stage["invalid_contact_pairs"]:
                if pair not in invalid_contact_pairs and len(invalid_contact_pairs) < 64:
                    invalid_contact_pairs.append(pair)
            if (
                sample_index == 0
                or sample_index == maximum_steps
                or sample_index % max(1, round(0.1 / self.env.advance_dt)) == 0
            ):
                trace.append(
                    {
                        "time_s": elapsed_s,
                        "active_lane_ids": torch.where(active)[0].detach().cpu().tolist(),
                        "speed_valid_by_lane": speed_valid.detach().cpu().tolist(),
                        "exact_success_by_lane": stage["exact"].mask.detach().cpu().tolist(),
                    }
                )
            return dwell.observe(elapsed_s, sample_index, speed_valid, active)

        completed = sample(0.0, 0)
        for step in range(1, maximum_steps + 1):
            self.env.set_robot_targets(arm_target, self.closed_finger_target)
            self.env.advance(self.env.advance_dt)
            completed = sample(step * self.env.advance_dt, step)
            completed_step = step
            if step >= minimum_steps and bool(completed.any()):
                lane_hold.deactivate(
                    lane_hold.active_mask & ~completed,
                    reason="canonical-relaxation-no-fresh-dwell",
                )
                break
        survivors = lane_hold.active_mask & completed
        if not bool(survivors.any()):
            lane_hold.deactivate(lane_hold.active_mask, reason="canonical-relaxation-timeout")
            raise RuntimeError(
                "Continuous canonical relaxation obtained no fresh two-second low-speed endpoint by 60 seconds: "
                f"dwell={dwell.evidence()}."
            )
        return {
            "passed": True,
            "starting_lane_ids": torch.where(starting_mask)[0].detach().cpu().tolist(),
            "surviving_lane_ids": torch.where(survivors)[0].detach().cpu().tolist(),
            "surviving_mask": survivors.detach().cpu().tolist(),
            "minimum_duration_s": minimum_duration_s,
            "maximum_duration_s": maximum_duration_s,
            "actual_duration_s": completed_step * self.env.advance_dt,
            "required_trailing_low_speed_s": required_duration_s,
            "dwell": dwell.evidence(),
            "maximum_metrics_by_lane": {name: value.detach().cpu().tolist() for name, value in maxima.items()},
            "lane_failure_masks": {name: mask.detach().cpu().tolist() for name, mask in lane_hold.reason_masks.items()},
            "invalid_contact_pairs": tuple(invalid_contact_pairs),
            "time_series": trace,
        }

    def _canonical_cold_goal_violation_masks(
        self,
        snapshot: dict[str, Any],
        *,
        captured_state: dict[str, torch.Tensor],
        authored_seat_target_e: torch.Tensor,
        authored_plug_orientation: torch.Tensor,
        history_evidence: dict[str, object],
    ) -> dict[str, torch.Tensor]:
        """Return per-lane hard violations for one post-step cold-proof sample."""
        task_q = snapshot["task_q"]
        arm_q = snapshot["arm_q"]
        body_drift = torch.linalg.vector_norm(
            task_q[..., :3] - captured_state["task_body_pose"][..., :3],
            dim=-1,
        )
        seat_error = torch.linalg.vector_norm(
            task_q[:, self.plug_index, :3] - authored_seat_target_e,
            dim=-1,
        )
        plug_tilt = math_utils.quat_error_magnitude(
            task_q[:, self.plug_index, 3:7],
            authored_plug_orientation,
        )
        return {
            "history-not-applied-exactly-once": ~self._vbd_pose_history_applied_mask(history_evidence),
            "arm-target-clamped": snapshot["arm_target_clamp_delta"] > 1.0e-7,
            "arm-target-drift": snapshot["arm_target_drift"] > 1.0e-7,
            "arm-joint-limits": ~joint_limit_mask(self.env, arm_q, margin=0.02),
            "target-joint-limits": ~joint_limit_mask(self.env, snapshot["arm_target"], margin=0.02),
            "arm-drift": (
                torch.abs(arm_q - captured_state["arm_joint_position"]).amax(dim=-1)
                > self.cfg.maximum_goal_arm_drift_rad
            ),
            "body-drift": body_drift.amax(dim=-1) > self.cfg.maximum_goal_body_drift_m,
            "socket-drift": body_drift[:, self.socket_index] > self.cfg.maximum_socket_drift_m,
            "cable-speed": snapshot["cable_linear_speed"] > self.cfg.maximum_goal_cable_speed_m_s,
            "arm-joint-speed": snapshot["arm_joint_speed"] > self.cfg.maximum_goal_arm_joint_speed_rad_s,
            "finger-joint-speed": snapshot["finger_joint_speed"] > self.cfg.maximum_goal_finger_joint_speed_m_s,
            "authored-seat": seat_error > self.cfg.maximum_canonical_seat_error_m,
            "authored-plug-tilt": plug_tilt > self.cfg.maximum_authored_plug_angle_rad,
            "plug-relative-latch": (
                snapshot["plug_relative_latch_angle"] > PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD
            ),
        }

    def _canonical_cold_stored_violation_masks(
        self,
        state: dict[str, torch.Tensor],
        authored_seat_target_e: torch.Tensor,
        authored_plug_orientation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return hard violations already present in a restored canonical capture."""
        task_q = state["task_body_pose"]
        task_qd = state["task_body_velocity"]
        arm_q = state["arm_joint_position"]
        arm_qd = state["arm_joint_velocity"]
        finger_q = state["finger_joint_position"]
        finger_qd = state["finger_joint_velocity"]
        tracking_limits = torch.as_tensor(
            self.env.cfg.actions.arm_action.tracking_error_limits,
            device=self.device,
            dtype=arm_q.dtype,
        )
        cable_speed = torch.linalg.vector_norm(task_qd[:, self.cable_slice, :3], dim=-1).amax(dim=-1)
        seat_error = torch.linalg.vector_norm(
            task_q[:, self.plug_index, :3] - authored_seat_target_e,
            dim=-1,
        )
        plug_tilt = math_utils.quat_error_magnitude(
            task_q[:, self.plug_index, 3:7],
            authored_plug_orientation,
        )
        return {
            "stored-non-finite": ~(
                task_state_is_finite_and_normalized(task_q, task_qd)
                & torch.isfinite(arm_q).all(dim=-1)
                & torch.isfinite(arm_qd).all(dim=-1)
                & torch.isfinite(finger_q).all(dim=-1)
                & torch.isfinite(finger_qd).all(dim=-1)
            ),
            "stored-arm-joint-limits": ~joint_limit_mask(self.env, arm_q, margin=0.02),
            "stored-target-joint-limits": ~joint_limit_mask(
                self.env,
                state["arm_joint_target"],
                margin=0.02,
            ),
            "stored-target-tracking": (torch.abs(state["arm_joint_target"] - arm_q) > tracking_limits).any(dim=-1),
            "stored-cable-speed": cable_speed > self.cfg.maximum_goal_cable_speed_m_s,
            "stored-arm-joint-speed": torch.abs(arm_qd).amax(dim=-1) > self.cfg.maximum_goal_arm_joint_speed_rad_s,
            "stored-finger-joint-speed": (
                torch.abs(finger_qd).amax(dim=-1) > self.cfg.maximum_goal_finger_joint_speed_m_s
            ),
            "stored-authored-seat": seat_error > self.cfg.maximum_canonical_seat_error_m,
            "stored-authored-plug-tilt": plug_tilt > self.cfg.maximum_authored_plug_angle_rad,
            "stored-plug-relative-latch": (
                plug_relative_latch_angle(
                    task_q,
                    plug_body_index=self.plug_index,
                    latch_body_index=self.latch_index,
                )
                > PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD
            ),
        }

    def _run_canonical_goal_cold_proof(
        self,
        original_capture: dict[str, torch.Tensor],
        active_mask: torch.Tensor,
        authored_seat_target_w: torch.Tensor,
        authored_plug_orientation: torch.Tensor,
        *,
        duration_s: float,
        use_authored_exact_reference: bool,
        stage_name: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Restore one quarantined copy and cold-prove surviving lanes without promotion."""
        quarantined, donor_lane = _quarantine_inactive_state(original_capture, active_mask)
        history_evidence = self._restore_state(quarantined)
        assert history_evidence is not None
        authored_target_e = authored_seat_target_w - self.env.env_origins
        goal_task_q = quarantined["task_body_pose"].clone()
        if use_authored_exact_reference:
            goal_task_q[:, self.plug_index, :3] = authored_target_e
            goal_task_q[:, self.plug_index, 3:7] = authored_plug_orientation
        starting_mask = torch.as_tensor(active_mask, device=self.device, dtype=torch.bool).clone()
        first_cable_speed_failure: dict[str, Any] | None = None

        def goal_gate(snapshot: dict[str, Any]) -> dict[str, torch.Tensor]:
            nonlocal first_cable_speed_failure
            if not bool(snapshot["active_mask"].any()):
                failures = {
                    reason: torch.where(mask)[0].detach().cpu().tolist()
                    for reason, mask in lane_hold.reason_masks.items()
                    if bool(mask.any())
                }
                raise RuntimeError(
                    f"{stage_name} cold proof has zero surviving lanes: reasons={failures}, "
                    f"first_cable_speed_failure={first_cable_speed_failure}."
                )
            violations = self._canonical_cold_goal_violation_masks(
                snapshot,
                captured_state=quarantined,
                authored_seat_target_e=authored_target_e,
                authored_plug_orientation=authored_plug_orientation,
                history_evidence=history_evidence,
            )
            cable_speed_failure = snapshot["active_mask"] & violations["cable-speed"]
            if first_cable_speed_failure is None and bool(cable_speed_failure.any()):
                cable_velocity = snapshot["task_qd"][:, self.cable_slice, :3]
                body_speed = torch.linalg.vector_norm(cable_velocity, dim=-1)
                peak_speed, peak_cable_index = body_speed.max(dim=-1)
                lane_ids = torch.where(cable_speed_failure)[0]
                peak_task_index = peak_cable_index[lane_ids] + self.cable_body_start
                first_cable_speed_failure = {
                    "step": int(snapshot["step"]),
                    "time_s": float(snapshot["step"]) * float(self.env.advance_dt),
                    "limit_m_s": self.cfg.maximum_goal_cable_speed_m_s,
                    "lane_ids": lane_ids.detach().cpu().tolist(),
                    "peak_speed_m_s": peak_speed[lane_ids].detach().cpu().tolist(),
                    "peak_task_body_index": peak_task_index.detach().cpu().tolist(),
                    "peak_task_body_name": [
                        self.layout.body_names[int(index)] for index in peak_task_index.detach().cpu()
                    ],
                    "peak_linear_velocity_m_s": cable_velocity[
                        lane_ids,
                        peak_cable_index[lane_ids],
                    ]
                    .detach()
                    .cpu()
                    .tolist(),
                    "peak_position_m": snapshot["task_q"][lane_ids, peak_task_index, :3].detach().cpu().tolist(),
                }
            return violations

        with _PerLaneTargetHold(
            self.env,
            starting_mask,
            quarantined["arm_joint_target"],
            quarantined["finger_joint_target"],
        ) as lane_hold:
            for reason, violation in self._canonical_cold_stored_violation_masks(
                quarantined,
                authored_target_e,
                authored_plug_orientation,
            ).items():
                lane_hold.deactivate(violation, reason=f"{stage_name}-{reason}")
            if not bool(lane_hold.active_mask.any()):
                raise RuntimeError(
                    f"{stage_name} cold proof has zero lanes after stored-state gates: "
                    f"reasons={lane_hold.reason_masks}."
                )
            passed, metrics = advance_exact_success_dwell(
                self.env,
                goal_task_q,
                quarantined["arm_joint_target"],
                quarantined["finger_joint_target"],
                duration_s=duration_s,
                require_all_samples=True,
                sample_physical_validity=True,
                arm_target_is_absolute=True,
                plug_body_index=self.plug_index,
                latch_body_index=self.latch_index,
                lane_hold=lane_hold,
                per_step_lane_goal_gate=goal_gate,
            )
            history_applied = self._vbd_pose_history_applied_mask(history_evidence)
            lane_hold.deactivate(~history_applied, reason=f"{stage_name}-history-not-applied-exactly-once")
            lane_hold.deactivate(~passed, reason=f"{stage_name}-final-proof")
            survivors = lane_hold.active_mask & passed & history_applied
            reason_masks = lane_hold.reason_masks
        if not bool(survivors.any()):
            raise RuntimeError(
                f"{stage_name} cold proof has zero surviving lanes after {duration_s:.1f} seconds: "
                f"reasons={reason_masks}, first_cable_speed_failure={first_cable_speed_failure}."
            )

        def serializable(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, dict):
                return {name: serializable(item) for name, item in value.items()}
            if isinstance(value, tuple | list):
                return [serializable(item) for item in value]
            return value

        evidence = {
            "passed": True,
            "stage": stage_name,
            "duration_s": duration_s,
            "source_capture": "original-rolling-capture",
            "endpoint_promotion": False,
            "authored_exact_reference": use_authored_exact_reference,
            "starting_lane_ids": torch.where(starting_mask)[0].detach().cpu().tolist(),
            "surviving_lane_ids": torch.where(survivors)[0].detach().cpu().tolist(),
            "surviving_mask": survivors.detach().cpu().tolist(),
            "quarantine_donor_lane": donor_lane,
            "quarantined_lane_ids": torch.where(~starting_mask)[0].detach().cpu().tolist(),
            "quarantined_fields": tuple(quarantined),
            "vbd_pose_history": serializable(self._vbd_pose_history_report_evidence(history_evidence)),
            "vbd_pose_history_applied_exactly_once_by_lane": history_applied.detach().cpu().tolist(),
            "exact_success_by_lane": passed.detach().cpu().tolist(),
            "lane_goal_gate_passed": metrics["lane_goal_gate_passed"].detach().cpu().tolist(),
            "lane_goal_gate_violation_masks": serializable(metrics["lane_goal_gate_violation_masks"]),
            "lane_goal_gate_first_failure_steps": serializable(metrics["lane_goal_gate_first_failure_steps"]),
            "lane_failure_masks": serializable(reason_masks),
            "maximum_body_excursion_m_by_lane": metrics["maximum_body_excursion"].detach().cpu().tolist(),
            "maximum_cable_speed_m_s_by_lane": metrics["maximum_cable_linear_speed"].detach().cpu().tolist(),
            "maximum_arm_joint_speed_rad_s_by_lane": metrics["maximum_arm_joint_speed"].detach().cpu().tolist(),
            "maximum_finger_joint_speed_m_s_by_lane": metrics["maximum_finger_joint_speed"].detach().cpu().tolist(),
            "maximum_plug_relative_latch_angle_rad_by_lane": (
                metrics["maximum_plug_relative_latch_angle"].detach().cpu().tolist()
            ),
            "any_contact_overflow": bool(metrics["any_contact_overflow"]),
            "invalid_contact_pairs": metrics["sampled_invalid_contact_pairs"],
        }
        return survivors, evidence

    def _run_canonical_goal_cold_sequence(
        self,
        original_capture: dict[str, torch.Tensor],
        active_mask: torch.Tensor,
        authored_seat_target_w: torch.Tensor,
        authored_plug_orientation: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Cold-prove the same original capture for 30 and then 60 seconds."""
        cold_30_survivors, cold_30_evidence = self._run_canonical_goal_cold_proof(
            original_capture,
            active_mask,
            authored_seat_target_w,
            authored_plug_orientation,
            duration_s=self.cfg.goal_cold_equilibrium_relax_s,
            use_authored_exact_reference=True,
            stage_name="canonical-cold-30s",
        )
        cold_60_survivors, cold_60_evidence = self._run_canonical_goal_cold_proof(
            original_capture,
            cold_30_survivors,
            authored_seat_target_w,
            authored_plug_orientation,
            duration_s=max(self.cfg.goal_cold_final_replay_s, 60.0),
            use_authored_exact_reference=False,
            stage_name="canonical-cold-60s",
        )
        return cold_60_survivors, {
            "same_original_capture_restored_both_times": True,
            "endpoint_promotion_count": 0,
            "cold_30s": cold_30_evidence,
            "cold_60s": cold_60_evidence,
        }

    def _run_production_canonical_goal_sequence(
        self,
        arm_target: torch.Tensor,
        finger_target: torch.Tensor,
        active_mask: torch.Tensor,
        authored_seat_target_w: torch.Tensor,
        authored_plug_orientation: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Run the batch-safe production relaxation, reseat, capture, and cold proofs."""
        self._assert_drive_disabled("production canonical-goal sequence")
        active_mask = torch.as_tensor(active_mask, device=self.device, dtype=torch.bool)
        if not bool(active_mask.any()):
            raise RuntimeError("Production canonical-goal sequence received zero surviving construction lanes.")
        authored_target_e = authored_seat_target_w - self.env.env_origins
        relaxation_start_q, _ = self.env.read_task_state()
        relaxation_goal_q = relaxation_start_q.clone()
        relaxation_goal_q[:, self.plug_index, :3] = authored_target_e
        relaxation_goal_q[:, self.plug_index, 3:7] = authored_plug_orientation
        tracking_limits = torch.as_tensor(
            self.env.cfg.actions.arm_action.tracking_error_limits,
            device=self.device,
            dtype=arm_target.dtype,
        )
        construction_survivors = active_mask.clone()

        with _PerLaneTargetHold(
            self.env,
            active_mask,
            arm_target,
            finger_target,
        ) as lane_hold:
            continuous_evidence = self._run_continuous_goal_relaxation(
                relaxation_goal_q,
                arm_target,
                authored_target_e,
                authored_plug_orientation,
                tracking_limits,
                lane_hold=lane_hold,
            )
            authored_plug_pose_e = torch.cat((authored_target_e, authored_plug_orientation), dim=-1)
            relaxed_q, _ = self.env.read_task_state()
            relaxed_plug_pose_e = relaxed_q[:, self.plug_index]
            reseat_intermediate_targets = tuple(
                torch.cat(
                    (
                        torch.lerp(relaxed_plug_pose_e[:, :3], authored_plug_pose_e[:, :3], fraction),
                        batched_quat_slerp(
                            relaxed_plug_pose_e[:, 3:7],
                            authored_plug_pose_e[:, 3:7],
                            fraction,
                        ),
                    ),
                    dim=-1,
                )
                for fraction in (0.25, 0.5, 0.75)
            )
            reseat_target, reseat_valid = self._move_grasped_plug(
                authored_plug_pose_e,
                lane_hold.last_sent_arm_target,
                duration_s=self.cfg.tcp_motion_s,
                intermediate_targets=reseat_intermediate_targets,
                endpoint_policy=_GRASPED_TRANSPORT_STRICT_ENDPOINT_POLICY,
                lane_hold=lane_hold,
            )
            lane_hold.deactivate(~reseat_valid, reason="canonical-authored-reseat")
            if not bool(lane_hold.active_mask.any()):
                raise RuntimeError(
                    f"The single authored canonical reseat has zero surviving lanes: reasons={lane_hold.reason_masks}."
                )
            trailing_start_q, _ = self.env.read_task_state()
            trailing_goal_q = trailing_start_q.clone()
            trailing_goal_q[:, self.plug_index, :3] = authored_target_e
            trailing_goal_q[:, self.plug_index, 3:7] = authored_plug_orientation
            rolling_survivors, rolling_evidence = self._run_diagnostic_reseat_rolling_equilibrium(
                trailing_goal_q,
                reseat_target,
                authored_target_e,
                authored_plug_orientation,
                tracking_limits,
                lane_hold=lane_hold,
            )
            original_capture = self._capture_state(
                lane_hold.last_sent_arm_target,
                lane_hold.last_sent_finger_target,
            )
            pre_cold_failure_masks = {
                name: mask.detach().cpu().tolist() for name, mask in lane_hold.reason_masks.items()
            }
            reseat_motion_evidence = self.last_grasped_motion_evidence

        cold_survivors, cold_evidence = self._run_canonical_goal_cold_sequence(
            original_capture,
            rolling_survivors,
            authored_seat_target_w,
            authored_plug_orientation,
        )
        canonical, selected_lane = _select_lowest_surviving_lane(original_capture, cold_survivors)

        def serializable(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, dict):
                return {name: serializable(item) for name, item in value.items()}
            if isinstance(value, tuple | list):
                return [serializable(item) for item in value]
            return value

        evidence = {
            "passed": True,
            "classification": "production-canonical-relax-reseat-cold-proof",
            "diagnostic_cli": self.cfg.diagnostic_p_relax_reseat,
            "physical_contract": {
                "finger_closed_target_m": self.cfg.finger_closed_target,
                "finger_raw_friction": GRASP_FRICTION,
                "grasp_proxy_raw_friction": self.env.grasp_proxy_friction,
                "effective_finger_proxy_friction": PICK_INSERT_EFFECTIVE_GRASP_FRICTION,
            },
            "original_lane_ids": list(range(self.env.num_envs)),
            "construction_surviving_mask": construction_survivors.detach().cpu().tolist(),
            "continuous_relaxation": continuous_evidence,
            "authored_reseat": {
                "count": 1,
                "intermediate_fractions": (0.25, 0.5, 0.75),
                "surviving_mask": reseat_valid.detach().cpu().tolist(),
                "motion": serializable(reseat_motion_evidence),
            },
            "reseat_trailing_equilibrium": rolling_evidence,
            "rolling_capture_surviving_mask": rolling_survivors.detach().cpu().tolist(),
            "pre_cold_lane_failure_masks": pre_cold_failure_masks,
            "cold_proofs": cold_evidence,
            "final_surviving_mask": cold_survivors.detach().cpu().tolist(),
            "final_surviving_lane_ids": torch.where(cold_survivors)[0].detach().cpu().tolist(),
            "selected_original_lane": selected_lane,
            "selection_rule": "deterministic-lowest-original-surviving-lane",
            "captured_state_fields": tuple(original_capture),
        }
        print(f"[PICK-INSERT PRODUCTION CANONICAL GOAL COMPLETE] {evidence}", flush=True)
        return canonical, evidence

    def _run_p_relax_reseat_discriminator(
        self,
        arm_target: torch.Tensor,
        authored_seat_target_w: torch.Tensor,
        authored_plug_orientation: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Distinguish premature construction capture from a persistent P equilibrium failure."""
        if self.env.num_envs != 1:
            raise RuntimeError("The P relax/reseat discriminator requires exactly one environment.")
        self._assert_drive_disabled("P relax/reseat discriminator")
        immutable_relaxation_target = arm_target.clone()
        authored_target_e = authored_seat_target_w - self.env.env_origins
        start_q, _ = self.env.read_task_state()
        start_tcp = self.env.tcp_pose_e()
        start_plug_quat = start_q[:, self.plug_index, 3:7]
        start_tcp_position_in_plug = math_utils.quat_apply(
            math_utils.quat_conjugate(start_plug_quat),
            start_tcp[:, :3] - start_q[:, self.plug_index, :3],
        )
        start_tcp_orientation_in_plug = math_utils.quat_mul(
            math_utils.quat_conjugate(start_plug_quat),
            start_tcp[:, 3:7],
        )
        relaxation_goal_q = start_q.clone()
        relaxation_goal_q[:, self.plug_index, :3] = authored_target_e
        relaxation_goal_q[:, self.plug_index, 3:7] = authored_plug_orientation
        tracking_limits = torch.as_tensor(
            self.env.cfg.actions.arm_action.tracking_error_limits,
            device=self.device,
            dtype=immutable_relaxation_target.dtype,
        )
        finger_stiffness_n_m = float(self.env.cfg.scene.robot.actuators["panda_hand"].stiffness)
        if not math.isfinite(finger_stiffness_n_m) or finger_stiffness_n_m <= 0.0:
            raise RuntimeError("The P relax/reseat discriminator requires a finite positive finger stiffness.")
        trace_interval_s = 0.1
        trace_stride = max(1, round(trace_interval_s / self.env.advance_dt))
        trace: list[dict[str, Any]] = []
        first_failures_s: dict[str, float | None] = {
            "hard_physical_or_controller": None,
            "exact_success": None,
            "body_excursion": None,
            "cable_speed": None,
            "plug_speed": None,
            "authored_seat": None,
            "authored_plug_tilt": None,
            "plug_relative_latch": None,
        }
        invalid_contact_pairs: list[str] = []
        maxima = {
            "body_excursion_m": 0.0,
            "cable_speed_m_s": 0.0,
            "plug_spatial_speed": 0.0,
            "authored_seat_error_m": 0.0,
            "authored_plug_tilt_rad": 0.0,
            "plug_relative_latch_angle_rad": 0.0,
            "tcp_in_plug_position_drift_m": 0.0,
            "tcp_in_plug_angle_drift_rad": 0.0,
            "arm_joint_speed_rad_s": 0.0,
            "finger_joint_speed_m_s": 0.0,
            "arm_target_tracking_error_rad": 0.0,
            "arm_target_clamp_delta_rad": 0.0,
        }

        def record_first(name: str, failed: torch.Tensor | bool, elapsed_s: float) -> None:
            if first_failures_s[name] is not None:
                return
            failed_bool = bool(failed) if isinstance(failed, bool) else bool(failed.any())
            if failed_bool:
                first_failures_s[name] = elapsed_s

        def sample_relaxation(elapsed_s: float, sample_index: int, *, force_trace: bool = False) -> torch.Tensor:
            task_q, task_qd = self.env.read_task_state()
            arm_q, arm_qd, finger_q, finger_qd = self.env.read_robot_state()
            collision = collision_metrics(self.env, require_bilateral_grasp=False)
            if collision.contact_overflow:
                raise RuntimeError("Global contact-buffer overflow during canonical grasp relaxation.")
            grasp = grasp_metrics(self.env, self.closed_finger_target, retaining_grasp=True)
            exact = exact_success_from_state(
                self.env,
                task_q,
                task_qd,
                relaxation_goal_q,
                plug_body_index=self.plug_index,
                latch_body_index=self.latch_index,
            )
            cable_speed = torch.linalg.vector_norm(
                task_qd[:, self.cable_slice, :3],
                dim=-1,
            ).amax(dim=-1)
            body_excursion = torch.linalg.vector_norm(
                task_q[..., :3] - start_q[..., :3],
                dim=-1,
            ).amax(dim=-1)
            seat_error = torch.linalg.vector_norm(
                task_q[:, self.plug_index, :3] - authored_target_e,
                dim=-1,
            )
            plug_tilt = math_utils.quat_error_magnitude(
                task_q[:, self.plug_index, 3:7],
                authored_plug_orientation,
            )
            relative_latch = plug_relative_latch_angle(
                task_q,
                plug_body_index=self.plug_index,
                latch_body_index=self.latch_index,
            )
            signed_latch_angle, signed_latch_rate = self._signed_latch_state(task_q, task_qd)
            tcp = self.env.tcp_pose_e()
            plug_quat = task_q[:, self.plug_index, 3:7]
            tcp_position_in_plug = math_utils.quat_apply(
                math_utils.quat_conjugate(plug_quat),
                tcp[:, :3] - task_q[:, self.plug_index, :3],
            )
            tcp_orientation_in_plug = math_utils.quat_mul(
                math_utils.quat_conjugate(plug_quat),
                tcp[:, 3:7],
            )
            tcp_position_drift = torch.linalg.vector_norm(
                tcp_position_in_plug - start_tcp_position_in_plug,
                dim=-1,
            )
            tcp_position_drift_in_plug = tcp_position_in_plug - start_tcp_position_in_plug
            tcp_orientation_drift_in_plug = math_utils.quat_unique(
                math_utils.quat_mul(
                    tcp_orientation_in_plug,
                    math_utils.quat_conjugate(start_tcp_orientation_in_plug),
                )
            )
            tcp_axis_angle_drift_in_plug = math_utils.axis_angle_from_quat(tcp_orientation_drift_in_plug)
            tcp_angle_drift = math_utils.quat_error_magnitude(
                tcp_orientation_in_plug,
                start_tcp_orientation_in_plug,
            )
            arm_speed = torch.abs(arm_qd).amax(dim=-1)
            finger_speed = torch.abs(finger_qd).amax(dim=-1)
            estimated_finger_spring_preload = finger_stiffness_n_m * (finger_q - self.closed_finger_target)
            resolved_target, clamp_delta = runtime_persistent_arm_target(
                self.env,
                immutable_relaxation_target,
            )
            target_drift = torch.abs(resolved_target - immutable_relaxation_target).amax(dim=-1)
            tracking_error = torch.abs(resolved_target - arm_q).amax(dim=-1)
            proxy_bilateral = _runtime_bilateral_grasp_proxy_contact_mask(
                self.env,
                collision.left_grasp_contact_count,
                collision.right_grasp_contact_count,
            )
            finite = (
                task_state_is_finite_and_normalized(task_q, task_qd)
                & torch.isfinite(arm_q).all(dim=-1)
                & torch.isfinite(arm_qd).all(dim=-1)
                & torch.isfinite(finger_q).all(dim=-1)
                & torch.isfinite(finger_qd).all(dim=-1)
            )
            hard_valid = (
                finite
                & collision.valid
                & grasp.valid
                & proxy_bilateral
                & ~self._drive_enabled()
                & ~self._orientation_hold_enabled()
                & joint_limit_mask(self.env, arm_q, margin=0.02)
                & joint_limit_mask(self.env, resolved_target, margin=0.02)
                & (torch.abs(resolved_target - arm_q) <= tracking_limits).all(dim=-1)
                & (clamp_delta <= 1.0e-7)
                & (target_drift <= 1.0e-7)
            )
            if collision.contact_overflow:
                hard_valid &= torch.zeros_like(hard_valid)
            for pair in collision.invalid_contact_pairs:
                if pair not in invalid_contact_pairs and len(invalid_contact_pairs) < 64:
                    invalid_contact_pairs.append(pair)

            values = {
                "body_excursion_m": body_excursion,
                "cable_speed_m_s": cable_speed,
                "plug_spatial_speed": exact.plug_spatial_speed,
                "authored_seat_error_m": seat_error,
                "authored_plug_tilt_rad": plug_tilt,
                "plug_relative_latch_angle_rad": relative_latch,
                "tcp_in_plug_position_drift_m": tcp_position_drift,
                "tcp_in_plug_angle_drift_rad": tcp_angle_drift,
                "arm_joint_speed_rad_s": arm_speed,
                "finger_joint_speed_m_s": finger_speed,
                "arm_target_tracking_error_rad": tracking_error,
                "arm_target_clamp_delta_rad": clamp_delta,
            }
            for name, value in values.items():
                maxima[name] = max(maxima[name], float(value.max()))

            record_first("hard_physical_or_controller", ~hard_valid, elapsed_s)
            record_first("exact_success", ~exact.mask, elapsed_s)
            record_first(
                "body_excursion",
                body_excursion > self.cfg.maximum_goal_body_drift_m,
                elapsed_s,
            )
            record_first(
                "cable_speed",
                cable_speed > self.cfg.maximum_goal_cable_speed_m_s,
                elapsed_s,
            )
            record_first(
                "plug_speed",
                exact.plug_spatial_speed > self.env.cfg.success_max_plug_speed,
                elapsed_s,
            )
            record_first(
                "authored_seat",
                seat_error > self.cfg.maximum_canonical_seat_error_m,
                elapsed_s,
            )
            record_first(
                "authored_plug_tilt",
                plug_tilt > self.cfg.maximum_authored_plug_angle_rad,
                elapsed_s,
            )
            record_first(
                "plug_relative_latch",
                relative_latch > PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD,
                elapsed_s,
            )

            strict_low_speed = (
                hard_valid
                & (cable_speed <= self.cfg.maximum_goal_cable_speed_m_s)
                & (exact.plug_spatial_speed <= self.env.cfg.success_max_plug_speed)
                & (arm_speed <= self.cfg.maximum_goal_arm_joint_speed_rad_s)
                & (finger_speed <= self.cfg.maximum_goal_finger_joint_speed_m_s)
            )
            if force_trace or sample_index % trace_stride == 0:
                trace.append(
                    {
                        "time_s": elapsed_s,
                        "hard_valid": bool(hard_valid.all()),
                        "strict_low_speed": bool(strict_low_speed.all()),
                        "exact_success": bool(exact.mask.all()),
                        "body_excursion_m": float(body_excursion.max()),
                        "cable_speed_m_s": float(cable_speed.max()),
                        "plug_spatial_speed": float(exact.plug_spatial_speed.max()),
                        "authored_seat_error_m": float(seat_error.max()),
                        "authored_plug_tilt_rad": float(plug_tilt.max()),
                        "absolute_latch_angle_error_rad": float(exact.latch_angle_error.max()),
                        "plug_relative_latch_angle_rad": float(relative_latch.max()),
                        "signed_latch_angle_rad": float(signed_latch_angle.abs().max()),
                        "signed_latch_rate_rad_s": float(signed_latch_rate.abs().max()),
                        "tcp_in_plug_position_drift_m": float(tcp_position_drift.max()),
                        "tcp_in_plug_angle_drift_rad": float(tcp_angle_drift.max()),
                        "tcp_in_plug_translation_m": tcp_position_in_plug.detach().cpu().tolist(),
                        "tcp_in_plug_translation_drift_m_xyz": (tcp_position_drift_in_plug.detach().cpu().tolist()),
                        "tcp_in_plug_axis_angle_drift_rad_xyz": (tcp_axis_angle_drift_in_plug.detach().cpu().tolist()),
                        "arm_joint_speed_rad_s": float(arm_speed.max()),
                        "finger_joint_speed_m_s": float(finger_speed.max()),
                        "finger_joint_position_m": finger_q.detach().cpu().tolist(),
                        "finger_joint_velocity_m_s": finger_qd.detach().cpu().tolist(),
                        "estimated_finger_spring_preload_n": (estimated_finger_spring_preload.detach().cpu().tolist()),
                        "arm_target_tracking_error_rad": float(tracking_error.max()),
                        "arm_target_clamp_delta_rad": float(clamp_delta.max()),
                        "left_proxy_contacts": int(collision.left_grasp_contact_count.min()),
                        "right_proxy_contacts": int(collision.right_grasp_contact_count.min()),
                        "invalid_contact_count": int(collision.invalid_contact_count.max()),
                        "contact_overflow": collision.contact_overflow,
                    }
                )
            if not bool(hard_valid.all()):
                raise RuntimeError(
                    "Continuous P relaxation encountered a hard physical/controller failure: "
                    f"time_s={elapsed_s}, sample={trace[-1] if trace else None}, "
                    f"invalid_pairs={invalid_contact_pairs}."
                )
            return strict_low_speed

        minimum_relaxation_s = self.cfg.goal_cold_equilibrium_relax_s
        maximum_relaxation_s = max(60.0, minimum_relaxation_s)
        trailing_s = self.cfg.goal_stability_window_s
        minimum_steps = math.ceil(minimum_relaxation_s / self.env.advance_dt)
        maximum_steps = math.ceil(maximum_relaxation_s / self.env.advance_dt)
        required_trailing_steps = math.ceil(trailing_s / self.env.advance_dt)
        trailing_steps = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.long)
        sample_relaxation(0.0, 0, force_trace=True)
        completed_steps = 0
        for step in range(1, maximum_steps + 1):
            self.env.set_robot_targets(immutable_relaxation_target, self.closed_finger_target)
            self.env.advance(self.env.advance_dt)
            strict_low_speed = sample_relaxation(step * self.env.advance_dt, step)
            trailing_steps = torch.where(
                strict_low_speed,
                trailing_steps + 1,
                torch.zeros_like(trailing_steps),
            )
            completed_steps = step
            if step >= minimum_steps and bool((trailing_steps >= required_trailing_steps).all()):
                break
        sample_relaxation(completed_steps * self.env.advance_dt, completed_steps, force_trace=True)
        continuous_relaxation_passed = bool((trailing_steps >= required_trailing_steps).all())
        continuous_evidence = {
            "passed": continuous_relaxation_passed,
            "minimum_duration_s": minimum_relaxation_s,
            "maximum_duration_s": maximum_relaxation_s,
            "actual_duration_s": completed_steps * self.env.advance_dt,
            "required_trailing_low_speed_s": trailing_s,
            "trailing_low_speed_steps": int(trailing_steps.min()),
            "immutable_absolute_target": True,
            "finger_closed_target_m": self.cfg.finger_closed_target,
            "plug_grasp_offset_m": tuple(self.env.cfg.plug_grasp_offset),
            "finger_raw_friction": GRASP_FRICTION,
            "grasp_proxy_raw_friction": self.env.grasp_proxy_friction,
            "effective_finger_proxy_friction": PICK_INSERT_EFFECTIVE_GRASP_FRICTION,
            "finger_stiffness_n_m": finger_stiffness_n_m,
            "maximum_metrics": maxima,
            "first_failure_times_s": first_failures_s,
            "invalid_contact_pairs": tuple(invalid_contact_pairs),
            "trace_interval_s": trace_interval_s,
            "time_series": trace,
        }
        print(f"[PICK-INSERT P CONTINUOUS RELAXATION] {continuous_evidence}", flush=True)
        if not continuous_relaxation_passed:
            raise RuntimeError(
                "Continuous P relaxation did not obtain a strict two-second low-speed endpoint by 60 seconds: "
                f"{continuous_evidence}"
            )

        authored_plug_pose_e = torch.cat((authored_target_e, authored_plug_orientation), dim=-1)
        relaxed_q, _ = self.env.read_task_state()
        relaxed_plug_pose_e = relaxed_q[:, self.plug_index]
        reseat_intermediate_targets = tuple(
            torch.cat(
                (
                    torch.lerp(relaxed_plug_pose_e[:, :3], authored_plug_pose_e[:, :3], fraction),
                    batched_quat_slerp(
                        relaxed_plug_pose_e[:, 3:7],
                        authored_plug_pose_e[:, 3:7],
                        fraction,
                    ),
                ),
                dim=-1,
            )
            for fraction in (0.25, 0.5, 0.75)
        )
        reseat_target, reseat_valid = self._move_grasped_plug(
            authored_plug_pose_e,
            immutable_relaxation_target,
            duration_s=self.cfg.tcp_motion_s,
            intermediate_targets=reseat_intermediate_targets,
            endpoint_policy=_GRASPED_TRANSPORT_STRICT_ENDPOINT_POLICY,
        )

        def serializable(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, dict):
                return {name: serializable(item) for name, item in value.items()}
            if isinstance(value, tuple | list):
                return [serializable(item) for item in value]
            return value

        reseat_evidence = serializable(self.last_grasped_motion_evidence)
        print(
            "[PICK-INSERT P AUTHORED RESEAT] "
            + str(
                {
                    "passed": bool(reseat_valid.all()),
                    "target_change_max_rad": float(torch.abs(reseat_target - immutable_relaxation_target).max()),
                    "motion": reseat_evidence,
                }
            ),
            flush=True,
        )
        if not bool(reseat_valid.all()):
            raise RuntimeError(f"Strict drive-free authored-pose reseat failed: {reseat_evidence}")

        trailing_start_q, _ = self.env.read_task_state()
        trailing_goal_q = trailing_start_q.clone()
        trailing_goal_q[:, self.plug_index, :3] = authored_target_e
        trailing_goal_q[:, self.plug_index, 3:7] = authored_plug_orientation
        trailing_valid, trailing_evidence = self._run_diagnostic_reseat_rolling_equilibrium(
            trailing_goal_q,
            reseat_target,
            authored_target_e,
            authored_plug_orientation,
            tracking_limits,
        )
        print(f"[PICK-INSERT P RESEAT ROLLING EQUILIBRIUM] {trailing_evidence}", flush=True)
        if not trailing_valid:
            raise RuntimeError(f"Authored reseat failed its strict trailing equilibrium: {trailing_evidence}")

        candidate = self._capture_state(reseat_target, self.closed_finger_target)
        (
            cold_candidate,
            _,
            cold_hard_valid,
            cold_converged,
            _,
            cold_evidence,
        ) = self._cold_goal_equilibrium_cycle(
            candidate,
            authored_seat_target_w,
            authored_plug_orientation,
        )
        cold_evidence = {"cycle": 1, "no_endpoint_promotion": True, **cold_evidence}
        print(f"[PICK-INSERT P RESEAT 30S COLD] {cold_evidence}", flush=True)
        if not torch.equal(cold_candidate["arm_joint_target"], candidate["arm_joint_target"]):
            raise RuntimeError("The reseated cold proof changed its immutable absolute target.")
        if not bool(cold_hard_valid.all()) or not bool(cold_converged.all()):
            raise RuntimeError(
                "Reseated candidate did not retain exact P equilibrium under its own 30-second cold replay: "
                f"{cold_evidence}"
            )

        final_candidate, final_history = self._canonical_goal_fixed_point(
            candidate,
            restore_after=False,
        )
        result = {
            "passed": True,
            "classification": "premature-construction-capture",
            "continuous_relaxation": continuous_evidence,
            "authored_reseat": reseat_evidence,
            "reseat_trailing_equilibrium": trailing_evidence,
            "cold_30s": cold_evidence,
            "cold_60s": final_history,
        }
        print(f"[PICK-INSERT P RELAX/RESEAT COMPLETE] {result}", flush=True)
        return final_candidate, result

    @torch.inference_mode()
    def derive_goal(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Derive the central seated state and prove a cold, drive-free goal."""
        self.env.restore_default_task()
        default_q, _ = self.env.read_task_state()
        central_socket = default_q[:, self.socket_index].clone()
        self._park_and_place_socket(central_socket)
        start_q, _ = self.env.read_task_state()
        start_plug_w = start_q[:, self.plug_index, :3] + self.env.env_origins
        drive_target_w = start_plug_w.clone()
        drive_target_w[:, 1] += self.cfg.goal_drive_distance_m
        authored_seat_target_w = wp.to_torch(self.env.rj45_runtime.default_goal_target_w).clone()
        authored_plug_orientation = wp.to_torch(self.env.rj45_runtime.default_orientation_target_w).clone()
        peak_latch_angle = plug_relative_latch_angle(
            start_q,
            plug_body_index=self.plug_index,
            latch_body_index=self.latch_index,
        )

        self.env.set_drive(True, start_plug_w)
        self.env.set_orientation_hold(True, authored_plug_orientation)
        if not bool(self._drive_enabled().all()) or not bool(self._orientation_hold_enabled().all()):
            raise RuntimeError("Both canonical construction drives must be enabled before the +35 mm ramp.")

        def sample_latch_angle() -> None:
            task_q, _ = self.env.read_task_state()
            angle = plug_relative_latch_angle(
                task_q,
                plug_body_index=self.plug_index,
                latch_body_index=self.latch_index,
            )
            peak_latch_angle.copy_(torch.maximum(peak_latch_angle, angle))

        def ramp(_step: int, _steps: int, progress: float) -> None:
            blend = _smoothstep(progress)
            self.env.set_drive(True, torch.lerp(start_plug_w, drive_target_w, blend))
            self.env.set_robot_targets(self.home_arm_q, self.open_finger_q)
            sample_latch_angle()

        ramp_steps = self.env.advance(self.cfg.goal_drive_ramp_s, ramp)
        self.env.set_drive(True, drive_target_w)
        self.env.set_robot_targets(self.home_arm_q, self.open_finger_q)

        def hold(_step: int, _steps: int, _progress: float) -> None:
            self.env.set_drive(True, drive_target_w)
            self.env.set_robot_targets(self.home_arm_q, self.open_finger_q)
            sample_latch_angle()

        hold_steps = self.env.advance(self.cfg.goal_drive_hold_s, hold)
        cable_settle_start_q, cable_settle_start_qd = self.env.read_task_state()
        cable_settle_start_speed, cable_settle_start_fastest = torch.linalg.vector_norm(
            cable_settle_start_qd[:, self.cable_slice, :3],
            dim=-1,
        ).max(dim=-1)
        cable_settle_steps = self.env.advance(self.cfg.goal_drive_cable_settle_s, hold)
        driven_open_q, driven_open_qd = self.env.read_task_state()
        cable_settle_body_drift = torch.linalg.vector_norm(
            driven_open_q[:, self.cable_slice, :3] - cable_settle_start_q[:, self.cable_slice, :3],
            dim=-1,
        ).amax(dim=-1)
        cable_settle_final_speed, cable_settle_final_fastest = torch.linalg.vector_norm(
            driven_open_qd[:, self.cable_slice, :3],
            dim=-1,
        ).max(dim=-1)
        cable_settle_start_fastest_body = cable_settle_start_fastest + self.cable_body_start
        cable_settle_final_fastest_body = cable_settle_final_fastest + self.cable_body_start
        world_ids = torch.arange(self.env.num_envs, device=self.device)
        cable_settle_start_fastest_qd = cable_settle_start_qd[world_ids, cable_settle_start_fastest_body]
        cable_settle_final_fastest_qd = driven_open_qd[world_ids, cable_settle_final_fastest_body]
        cable_settle_final_fastest_position = driven_open_q[world_ids, cable_settle_final_fastest_body, :3]
        cable_settle_minimum_support_clearance = driven_open_q[:, self.cable_slice, 2].amin(dim=-1) - CABLE_RADIUS
        cable_settle_minimum_center_height, cable_settle_lowest_segment = driven_open_q[:, self.cable_slice, 2].min(
            dim=-1
        )
        driven_open_latch_angle = plug_relative_latch_angle(
            driven_open_q,
            plug_body_index=self.plug_index,
            latch_body_index=self.latch_index,
        )
        peak_latch_angle.copy_(torch.maximum(peak_latch_angle, driven_open_latch_angle))
        driven_open_seat_error = torch.linalg.vector_norm(
            driven_open_q[:, self.plug_index, :3] + self.env.env_origins - authored_seat_target_w,
            dim=-1,
        )
        driven_open_plug_tilt = math_utils.quat_error_magnitude(
            driven_open_q[:, self.plug_index, 3:7],
            authored_plug_orientation,
        )
        driven_open_plug_speed = torch.linalg.vector_norm(
            driven_open_qd[:, self.plug_index, :3],
            dim=-1,
        )
        driven_open_collision = collision_metrics(self.env, require_bilateral_grasp=False)
        if driven_open_collision.contact_overflow:
            raise RuntimeError("Global contact-buffer overflow at the canonical pre-grasp boundary.")
        support_evidence = {
            "final_speed_m_s": float(cable_settle_final_speed.max()),
            "fastest_body_index": int(cable_settle_final_fastest_body[0]),
            "fastest_body_name": self.layout.body_names[int(cable_settle_final_fastest_body[0])],
            "fastest_spatial_velocity": cable_settle_final_fastest_qd[0].detach().cpu().tolist(),
            "fastest_position_m": cable_settle_final_fastest_position[0].detach().cpu().tolist(),
            "minimum_center_height_m": float(cable_settle_minimum_center_height.min()),
            "lowest_segment_index": int(cable_settle_lowest_segment[0]),
            "lowest_body_index": int(cable_settle_lowest_segment[0]) + self.cable_body_start,
            "lowest_body_name": self.layout.body_names[int(cable_settle_lowest_segment[0]) + self.cable_body_start],
            "minimum_radial_support_clearance_m": float(cable_settle_minimum_support_clearance.min()),
        }
        print(f"[PICK-INSERT TABLE SUPPORT] {support_evidence}", flush=True)
        construction_pregrasp_valid = (
            task_state_is_finite_and_normalized(driven_open_q, driven_open_qd)
            & (peak_latch_angle >= self.cfg.canonical_peak_latch_angle_min_rad)
            & (driven_open_seat_error <= self.cfg.maximum_canonical_seat_error_m)
            & (driven_open_plug_tilt <= self.cfg.maximum_authored_plug_angle_rad)
            & (cable_settle_final_speed <= self.cfg.maximum_goal_cable_speed_m_s)
            & (cable_settle_minimum_support_clearance >= -self.cfg.maximum_cable_support_penetration_m)
            & driven_open_collision.valid
        )
        if not bool(construction_pregrasp_valid.any()):
            raise RuntimeError(
                "Orientation-held +35 mm construction left zero lanes at the authored seated pose: "
                f"valid={construction_pregrasp_valid.tolist()}, "
                f"peak_latch={peak_latch_angle.tolist()}, "
                f"drive_active_latch={driven_open_latch_angle.tolist()}, "
                f"authored_seat_error={driven_open_seat_error.tolist()}, "
                f"plug_tilt={driven_open_plug_tilt.tolist()}, "
                f"plug_speed={driven_open_plug_speed.tolist()}, "
                f"cable_settle_start_speed={cable_settle_start_speed.tolist()}, "
                f"cable_settle_start_fastest_body={cable_settle_start_fastest_body.tolist()}, "
                f"cable_settle_start_fastest_name="
                f"{[self.layout.body_names[int(index)] for index in cable_settle_start_fastest_body]}, "
                f"cable_settle_start_fastest_qd={cable_settle_start_fastest_qd.tolist()}, "
                f"cable_settle_final_speed={cable_settle_final_speed.tolist()}, "
                f"cable_settle_final_fastest_body={cable_settle_final_fastest_body.tolist()}, "
                f"cable_settle_final_fastest_name="
                f"{[self.layout.body_names[int(index)] for index in cable_settle_final_fastest_body]}, "
                f"cable_settle_final_fastest_qd={cable_settle_final_fastest_qd.tolist()}, "
                f"cable_settle_final_fastest_position={cable_settle_final_fastest_position.tolist()}, "
                f"cable_settle_minimum_support_clearance={cable_settle_minimum_support_clearance.tolist()}, "
                f"cable_settle_drift={cable_settle_body_drift.tolist()}, "
                f"collision={driven_open_collision.valid.tolist()}, "
                f"invalid_contacts={driven_open_collision.invalid_contact_count.tolist()}, "
                f"invalid_pairs={driven_open_collision.invalid_contact_pairs}."
            )

        goal_arm_target, seated_acquisition_valid, seated_acquisition = self._acquire_current_plug(
            arm_seed=self.home_arm_q,
            duration_s=2.0,
            require_construction_drives_enabled=True,
            active_mask=construction_pregrasp_valid,
        )
        self.nominal_grasp_arm_seed = goal_arm_target.clone()
        if not bool(self._drive_enabled().all()) or not bool(self._orientation_hold_enabled().all()):
            raise RuntimeError("Both construction drives must remain enabled throughout seated acquisition.")

        driven_q, driven_qd = self.env.read_task_state()
        driven_latch_angle = plug_relative_latch_angle(
            driven_q,
            plug_body_index=self.plug_index,
            latch_body_index=self.latch_index,
        )
        peak_latch_angle.copy_(torch.maximum(peak_latch_angle, driven_latch_angle))
        driven_seat_error = torch.linalg.vector_norm(
            driven_q[:, self.plug_index, :3] + self.env.env_origins - authored_seat_target_w,
            dim=-1,
        )
        driven_plug_tilt = math_utils.quat_error_magnitude(
            driven_q[:, self.plug_index, 3:7],
            authored_plug_orientation,
        )
        proxy_center = torch.as_tensor(GRASP_PROXY_CENTER, device=self.device)
        proxy_half_extents = torch.as_tensor(GRASP_PROXY_HALF_EXTENTS, device=self.device)
        proxy_corner_signs = torch.tensor(
            [
                (-1.0, -1.0, -1.0),
                (-1.0, -1.0, 1.0),
                (-1.0, 1.0, -1.0),
                (-1.0, 1.0, 1.0),
                (1.0, -1.0, -1.0),
                (1.0, -1.0, 1.0),
                (1.0, 1.0, -1.0),
                (1.0, 1.0, 1.0),
            ],
            device=self.device,
        )
        proxy_corners_local = proxy_center + proxy_corner_signs * proxy_half_extents
        proxy_rotation = driven_q[:, self.plug_index, None, 3:7].expand(-1, 8, -1)
        proxy_top_z = (
            driven_q[:, self.plug_index, None, 2]
            + math_utils.quat_apply(proxy_rotation, proxy_corners_local.expand(self.env.num_envs, -1, -1))[:, :, 2]
        ).amax(dim=-1)
        release_grasp = grasp_metrics(self.env, self.closed_finger_target)
        release_collision = collision_metrics(self.env)
        if release_collision.contact_overflow:
            raise RuntimeError("Global contact-buffer overflow at the canonical seated-acquisition boundary.")
        preclose_collision = seated_acquisition["contact_preclose_collision"]
        preclose_tcp_distance = torch.linalg.vector_norm(
            self.env.tcp_pose_e()[:, :3] - self.env.plug_grasp_position_e(),
            dim=-1,
        )
        driven_arm_q, _, _, _ = self.env.read_robot_state()
        driven_arm_target_error = torch.abs(driven_arm_q - goal_arm_target).amax(dim=-1)
        construction_valid = (
            construction_pregrasp_valid
            & seated_acquisition_valid
            & task_state_is_finite_and_normalized(driven_q, driven_qd)
            & (peak_latch_angle >= self.cfg.canonical_peak_latch_angle_min_rad)
            & (driven_seat_error <= self.cfg.maximum_canonical_seat_error_m)
            & (driven_plug_tilt <= self.cfg.maximum_authored_plug_angle_rad)
            & release_grasp.valid
            & release_collision.valid
        )
        if not bool(construction_valid.any()):
            raise RuntimeError(
                "Table-clearance acquisition of the orientation-held seated plug left zero lanes: "
                f"valid={construction_valid.tolist()}, "
                f"acquisition_valid={seated_acquisition_valid.tolist()}, "
                f"peak_latch={peak_latch_angle.tolist()}, "
                f"drive_active_latch={driven_latch_angle.tolist()}, "
                f"authored_seat_error={driven_seat_error.tolist()}, "
                f"plug_tilt={driven_plug_tilt.tolist()}, "
                f"clearance_error={seated_acquisition['preclose_error'].tolist()}, "
                f"clearance_collision={seated_acquisition['preclose_collision'].valid.tolist()}, "
                f"clearance_pairs={seated_acquisition['preclose_collision'].invalid_contact_pairs}, "
                f"contact_error={seated_acquisition['contact_position_error'].tolist()}, "
                f"maximum_descent_error={seated_acquisition['maximum_open_descent_tcp_error'].tolist()}, "
                f"descent_collision={seated_acquisition['open_descent_collision_valid'].tolist()}, "
                f"ik_diagnostics={seated_acquisition['ik_diagnostics']}, "
                f"grasp_offset={self.env.cfg.plug_grasp_offset}, "
                f"proxy_top_z={proxy_top_z.tolist()}, "
                f"clearance_target={seated_acquisition['clearance_target_position'].tolist()}, "
                f"clearance_tcp={seated_acquisition['clearance_tcp_position'].tolist()}, "
                f"contact_target={seated_acquisition['contact_target_position'].tolist()}, "
                f"contact_tcp={seated_acquisition['contact_tcp_position'].tolist()}, "
                f"final_tcp={seated_acquisition['final_tcp_position'].tolist()}, "
                f"contact_collision={preclose_collision.valid.tolist()}, "
                f"contact_pairs={preclose_collision.invalid_contact_pairs}, "
                f"final_distance={preclose_tcp_distance.tolist()}, "
                f"grasp={release_grasp.valid.tolist()}, "
                f"collision={release_collision.valid.tolist()}, "
                f"invalid_contacts={release_collision.invalid_contact_count.tolist()}, "
                f"left_proxy_contacts={release_collision.left_grasp_contact_count.tolist()}, "
                f"right_proxy_contacts={release_collision.right_grasp_contact_count.tolist()}, "
                f"invalid_pairs={release_collision.invalid_contact_pairs}, "
                "post_contact_settle={"
                + ", ".join(
                    f"{key}={value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value}"
                    for key, value in seated_acquisition["post_contact_settle"].items()
                )
                + "}."
            )

        release_arm_q, _, _, _ = self.env.read_robot_state()
        release_arm_bias = goal_arm_target - release_arm_q
        self.env.set_orientation_hold(False)
        self.env.set_drive(False)
        self._assert_drive_disabled("canonical passive settling")
        if not (self.cfg.diagnostic_reset_abcd or self.cfg.diagnostic_reset_e_only):
            canonical, evidence = self._run_production_canonical_goal_sequence(
                goal_arm_target,
                seated_acquisition["last_finger_target"],
                construction_valid,
                authored_seat_target_w,
                authored_plug_orientation,
            )
            evidence["construction_lane_evidence"] = {
                "original_lane_ids": list(range(self.env.num_envs)),
                "pregrasp_valid_mask": construction_pregrasp_valid.detach().cpu().tolist(),
                "acquisition_valid_mask": seated_acquisition_valid.detach().cpu().tolist(),
                "construction_valid_mask": construction_valid.detach().cpu().tolist(),
                "acquisition_lane_failure_masks": {
                    name: mask.detach().cpu().tolist()
                    for name, mask in seated_acquisition["lane_failure_masks"].items()
                },
                "peak_latch_angle_rad_by_lane": peak_latch_angle.detach().cpu().tolist(),
                "authored_seat_error_m_by_lane": driven_seat_error.detach().cpu().tolist(),
                "authored_plug_tilt_rad_by_lane": driven_plug_tilt.detach().cpu().tolist(),
                "release_grasp_valid_mask": release_grasp.valid.detach().cpu().tolist(),
                "release_collision_valid_mask": release_collision.valid.detach().cpu().tolist(),
            }
            return canonical, evidence

        cold_rewrite_equilibrium_history: list[dict[str, Any]] = []
        cold_rewrite_endpoint_promotion_history: list[dict[str, Any]] = []
        converged_candidate: dict[str, torch.Tensor] | None = None
        fixed_point_candidate = self._capture_state(goal_arm_target, self.closed_finger_target)
        immutable_goal_arm_target = fixed_point_candidate["arm_joint_target"].clone()
        if self.cfg.diagnostic_reset_abcd:
            abcd_evidence = self._run_reset_abcd_discriminator(
                fixed_point_candidate,
                authored_seat_target_w,
                authored_plug_orientation,
            )
            return fixed_point_candidate, {"reset_abcd_discriminator": abcd_evidence}
        for cold_cycle in range(1, self.cfg.goal_cold_fixed_point_max_cycles + 1):
            (
                candidate,
                endpoint_candidate,
                cold_hard_valid,
                cold_converged,
                endpoint_promotable,
                cold_evidence,
            ) = self._cold_goal_equilibrium_cycle(
                fixed_point_candidate,
                authored_seat_target_w,
                authored_plug_orientation,
            )
            cold_evidence = {"cycle": cold_cycle, **cold_evidence}
            cold_rewrite_equilibrium_history.append(cold_evidence)
            print(f"[PICK-INSERT GOAL COLD-REWRITE CYCLE] {cold_evidence}", flush=True)
            if not bool(cold_hard_valid.all()):
                raise RuntimeError(
                    "Canonical cold-rewrite equilibrium encountered a hard physical/controller failure: "
                    f"{cold_rewrite_equilibrium_history}"
                )
            if bool(cold_converged.all()) and not self.cfg.diagnostic_reset_e_only:
                converged_candidate = candidate
                break
            if not bool(endpoint_promotable.all()):
                raise RuntimeError(
                    f"Canonical cold-rewrite endpoint failed the fixed-target promotion gates: {cold_evidence}"
                )
            if not torch.equal(endpoint_candidate["arm_joint_target"], immutable_goal_arm_target):
                raise RuntimeError("Cold fixed-point endpoint did not preserve the absolute arm target bitwise.")
            promotion_evidence = {
                "cycle": cold_cycle,
                "passed": True,
                "source": "hard-valid-final-endpoint",
                "absolute_arm_target_bitwise_unchanged": True,
                "final_cable_speed_m_s": cold_evidence["final_cable_speed_m_s"],
                "final_plug_spatial_speed": cold_evidence["final_plug_spatial_speed"],
                "final_plug_relative_latch_angle_rad": cold_evidence["final_plug_relative_latch_angle_rad"],
            }
            cold_rewrite_endpoint_promotion_history.append(promotion_evidence)
            print(f"[PICK-INSERT GOAL COLD-REWRITE PROMOTION] {promotion_evidence}", flush=True)
            fixed_point_candidate = endpoint_candidate
        if self.cfg.diagnostic_reset_e_only:
            reset_e_evidence = self._run_reset_abcd_discriminator(
                fixed_point_candidate,
                authored_seat_target_w,
                authored_plug_orientation,
            )
            return fixed_point_candidate, {
                "cold_rewrite_equilibrium": cold_rewrite_equilibrium_history,
                "cold_rewrite_endpoint_promotions": cold_rewrite_endpoint_promotion_history,
                "reset_e_discriminator": reset_e_evidence,
            }
        if converged_candidate is None:
            raise RuntimeError(
                "Canonical seated goal did not reach its strict cold-rewrite equilibrium within four cycles: "
                f"equilibrium={cold_rewrite_equilibrium_history}, "
                f"promotions={cold_rewrite_endpoint_promotion_history}"
            )

        # First converge the coupled arm/cable system against the same fixed
        # absolute arm command used for acquisition.  The acquisition-time
        # tracking error is not a valid reset preload: replaying it relative
        # to every newly measured position can walk the arm into the table.
        def hold_fixed_goal_arm_target(_step: int, _steps: int, _progress: float) -> None:
            self.env.set_robot_targets(goal_arm_target, self.closed_finger_target)

        early_steps = self.env.advance(
            self.cfg.goal_passive_settle_s - self.cfg.goal_stability_window_s,
            hold_fixed_goal_arm_target,
        )
        passive_pre_stability_history: list[dict[str, Any]] = []
        for relaxation_cycle in range(1, self.cfg.goal_passive_pre_stability_max_cycles + 1):
            relaxation_q, relaxation_qd = self.env.read_task_state()
            relaxation_cable_speed, relaxation_fastest = torch.linalg.vector_norm(
                relaxation_qd[:, self.cable_slice, :3],
                dim=-1,
            ).max(dim=-1)
            relaxation_fastest_body = relaxation_fastest + self.cable_body_start
            relaxation_evidence = {
                "cycle": relaxation_cycle,
                "maximum_cable_speed_m_s": float(relaxation_cable_speed.max()),
                "fastest_cable_body_index_by_world": relaxation_fastest_body.detach().cpu().tolist(),
                "fastest_cable_body_name_by_world": [
                    self.layout.body_names[int(index)] for index in relaxation_fastest_body
                ],
                "passed": bool((relaxation_cable_speed <= self.cfg.maximum_goal_cable_speed_m_s).all()),
            }
            passive_pre_stability_history.append(relaxation_evidence)
            print(f"[PICK-INSERT PASSIVE PRE-STABILITY] {relaxation_evidence}", flush=True)
            if relaxation_evidence["passed"]:
                break
            if relaxation_cycle == self.cfg.goal_passive_pre_stability_max_cycles:
                raise RuntimeError(
                    "Canonical +35 mm goal cable did not settle before its strict passive proof window: "
                    f"{passive_pre_stability_history}"
                )
            early_steps += self.env.advance(
                self.cfg.goal_stability_window_s,
                hold_fixed_goal_arm_target,
            )
        settled_arm_q, settled_arm_qd, _, settled_finger_qd = self.env.read_robot_state()
        settled_arm_bias = goal_arm_target - settled_arm_q
        settled_arm_bias_magnitude = torch.abs(settled_arm_bias).amax(dim=-1)
        release_to_settled_arm_drift = torch.abs(settled_arm_q - release_arm_q).amax(dim=-1)
        stability_start_q, stability_start_qd = self.env.read_task_state()
        stability_start_arm_q = settled_arm_q.clone()
        stability_start_grasp = grasp_metrics(self.env, self.closed_finger_target, retaining_grasp=True)
        stability_start_collision = collision_metrics(self.env)
        canonical_target_e = authored_seat_target_w - self.env.env_origins
        trailing_goal_q = stability_start_q.clone()
        trailing_goal_q[:, self.plug_index, :3] = canonical_target_e
        trailing_goal_q[:, self.plug_index, 3:7] = authored_plug_orientation
        trailing_exact, trailing_metrics = advance_exact_success_dwell(
            self.env,
            trailing_goal_q,
            goal_arm_target,
            self.closed_finger_target,
            duration_s=self.cfg.goal_stability_window_s,
            require_all_samples=True,
            sample_physical_validity=True,
            arm_target_is_absolute=True,
            plug_body_index=self.plug_index,
            latch_body_index=self.latch_index,
        )
        stability_steps = int(trailing_metrics["sample_steps"])
        passive_q, passive_qd = self.env.read_task_state()
        passive_body_drift_all = torch.linalg.vector_norm(
            passive_q[..., :3] - stability_start_q[..., :3],
            dim=-1,
        )
        passive_body_drift, passive_worst_body = passive_body_drift_all.max(dim=-1)
        passive_start_cable_speed, passive_start_fastest = torch.linalg.vector_norm(
            stability_start_qd[:, self.cable_slice, :3],
            dim=-1,
        ).max(dim=-1)
        passive_cable_speed, passive_fastest = torch.linalg.vector_norm(
            passive_qd[:, self.cable_slice, :3],
            dim=-1,
        ).max(dim=-1)
        passive_start_fastest_body = passive_start_fastest + self.cable_body_start
        passive_fastest_body = passive_fastest + self.cable_body_start
        world_ids = torch.arange(self.env.num_envs, device=self.device)
        passive_start_fastest_qd = stability_start_qd[world_ids, passive_start_fastest_body]
        passive_fastest_qd = passive_qd[world_ids, passive_fastest_body]
        passive_start_fastest_position = stability_start_q[world_ids, passive_start_fastest_body, :3]
        passive_fastest_position = passive_q[world_ids, passive_fastest_body, :3]
        passive_start_fastest_support_clearance = passive_start_fastest_position[:, 2] - CABLE_RADIUS
        passive_fastest_support_clearance = passive_fastest_position[:, 2] - CABLE_RADIUS
        passive_latch_angle = plug_relative_latch_angle(
            passive_q,
            plug_body_index=self.plug_index,
            latch_body_index=self.latch_index,
        )
        passive_grasp = grasp_metrics(self.env, self.closed_finger_target, retaining_grasp=True)
        passive_collision = collision_metrics(self.env)
        passive_arm_q, passive_arm_qd, _, passive_finger_qd = self.env.read_robot_state()
        passive_arm_drift = torch.abs(passive_arm_q - stability_start_arm_q).amax(dim=-1)
        passive_arm_target, passive_arm_target_clamp_delta = runtime_persistent_arm_target(
            self.env,
            goal_arm_target,
        )
        stability_start_arm_speed = torch.abs(settled_arm_qd).amax(dim=-1)
        stability_start_finger_speed = torch.abs(settled_finger_qd).amax(dim=-1)
        passive_arm_speed = torch.abs(passive_arm_qd).amax(dim=-1)
        passive_finger_speed = torch.abs(passive_finger_qd).amax(dim=-1)
        passive_seat_error = torch.linalg.vector_norm(
            passive_q[:, self.plug_index, :3] - canonical_target_e,
            dim=-1,
        )
        passive_plug_tilt = math_utils.quat_error_magnitude(
            passive_q[:, self.plug_index, 3:7],
            authored_plug_orientation,
        )
        passive_goal_valid = (
            construction_valid
            & preclose_collision.valid
            & release_grasp.valid
            & release_collision.valid
            & stability_start_grasp.valid
            & stability_start_collision.valid
            & passive_grasp.valid
            & passive_collision.valid
            & task_state_is_finite_and_normalized(stability_start_q, stability_start_qd)
            & task_state_is_finite_and_normalized(passive_q, passive_qd)
            & joint_limit_mask(self.env, stability_start_arm_q, margin=0.02)
            & joint_limit_mask(self.env, goal_arm_target, margin=0.02)
            & joint_limit_mask(self.env, passive_arm_q, margin=0.02)
            & joint_limit_mask(self.env, passive_arm_target, margin=0.02)
            & torch.isfinite(settled_arm_bias).all(dim=-1)
            & (settled_arm_bias_magnitude <= self.cfg.maximum_goal_arm_bias_rad)
            & (passive_arm_drift <= self.cfg.maximum_goal_arm_drift_rad)
            & (stability_start_arm_speed <= self.cfg.maximum_goal_arm_joint_speed_rad_s)
            & (passive_arm_speed <= self.cfg.maximum_goal_arm_joint_speed_rad_s)
            & (stability_start_finger_speed <= self.cfg.maximum_goal_finger_joint_speed_m_s)
            & (passive_finger_speed <= self.cfg.maximum_goal_finger_joint_speed_m_s)
            & (passive_arm_target_clamp_delta <= 1.0e-7)
            & (passive_body_drift <= self.cfg.maximum_goal_body_drift_m)
            & (passive_start_cable_speed <= self.cfg.maximum_goal_cable_speed_m_s)
            & (passive_cable_speed <= self.cfg.maximum_goal_cable_speed_m_s)
            & trailing_exact
            & trailing_metrics["all_samples_collision_free"]
            & trailing_metrics["all_samples_bilateral_grasp"]
            & trailing_metrics["all_samples_proxy_bilateral_contact"]
            & trailing_metrics["all_samples_finite"]
            & trailing_metrics["all_samples_arm_target_tracking_bounded"]
            & ~trailing_metrics["any_arm_target_clamped"]
            & (trailing_metrics["maximum_arm_target_clamp_delta"] <= 1.0e-7)
            & (trailing_metrics["maximum_arm_target_drift"] <= 1.0e-7)
            & (trailing_metrics["maximum_body_excursion"] <= self.cfg.maximum_goal_body_drift_m)
            & (trailing_metrics["maximum_cable_linear_speed"] <= self.cfg.maximum_goal_cable_speed_m_s)
            & (trailing_metrics["maximum_arm_joint_speed"] <= self.cfg.maximum_goal_arm_joint_speed_rad_s)
            & (trailing_metrics["maximum_finger_joint_speed"] <= self.cfg.maximum_goal_finger_joint_speed_m_s)
            & (
                trailing_metrics["maximum_plug_relative_latch_angle"]
                <= PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD
            )
            & ~torch.as_tensor(trailing_metrics["any_contact_overflow"], device=self.device)
            & (passive_latch_angle <= PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD)
            & (passive_seat_error <= self.cfg.maximum_canonical_seat_error_m)
            & (passive_plug_tilt <= self.cfg.maximum_authored_plug_angle_rad)
        )
        if not bool(passive_goal_valid.all()):
            raise RuntimeError(
                "Canonical +35 mm goal failed its drive-free grasped settle: "
                f"valid={passive_goal_valid.tolist()}, peak_latch={peak_latch_angle.tolist()}, "
                f"final_latch={passive_latch_angle.tolist()}, drift={passive_body_drift.tolist()}, "
                f"seat_error={passive_seat_error.tolist()}, plug_tilt={passive_plug_tilt.tolist()}, "
                f"worst_body={passive_worst_body.tolist()}, "
                f"worst_body_name={[self.layout.body_names[int(index)] for index in passive_worst_body]}, "
                f"cable_start_speed={passive_start_cable_speed.tolist()}, "
                f"cable_start_fastest_body={passive_start_fastest_body.tolist()}, "
                f"cable_start_fastest_name="
                f"{[self.layout.body_names[int(index)] for index in passive_start_fastest_body]}, "
                f"cable_start_fastest_qd={passive_start_fastest_qd.tolist()}, "
                f"cable_start_fastest_position={passive_start_fastest_position.tolist()}, "
                f"cable_start_fastest_support_clearance={passive_start_fastest_support_clearance.tolist()}, "
                f"cable_final_speed={passive_cable_speed.tolist()}, "
                f"cable_final_fastest_body={passive_fastest_body.tolist()}, "
                f"cable_final_fastest_name="
                f"{[self.layout.body_names[int(index)] for index in passive_fastest_body]}, "
                f"cable_final_fastest_qd={passive_fastest_qd.tolist()}, "
                f"cable_final_fastest_position={passive_fastest_position.tolist()}, "
                f"cable_final_fastest_support_clearance={passive_fastest_support_clearance.tolist()}, "
                f"release_arm_bias={torch.abs(release_arm_bias).amax(dim=-1).tolist()}, "
                f"settled_arm_bias={settled_arm_bias_magnitude.tolist()}, "
                f"release_to_settled_arm_drift={release_to_settled_arm_drift.tolist()}, "
                f"verification_arm_drift={passive_arm_drift.tolist()}, "
                f"settled_arm_speed={stability_start_arm_speed.tolist()}, "
                f"passive_arm_speed={passive_arm_speed.tolist()}, "
                f"settled_finger_speed={stability_start_finger_speed.tolist()}, "
                f"passive_finger_speed={passive_finger_speed.tolist()}, "
                f"passive_arm_target_clamp_delta={passive_arm_target_clamp_delta.tolist()}, "
                f"preclose_distance={preclose_tcp_distance.tolist()}, "
                f"release_grasp={release_grasp.valid.tolist()}, "
                f"release_collision={release_collision.valid.tolist()}, "
                f"release_invalid_pairs={release_collision.invalid_contact_pairs}, "
                f"settled_grasp={stability_start_grasp.valid.tolist()}, "
                f"settled_collision={stability_start_collision.valid.tolist()}, "
                f"settled_invalid_pairs={stability_start_collision.invalid_contact_pairs}, "
                f"passive_grasp={passive_grasp.valid.tolist()}, "
                f"passive_collision={passive_collision.valid.tolist()}, "
                f"invalid_contacts={passive_collision.invalid_contact_count.tolist()}, "
                f"invalid_pairs={passive_collision.invalid_contact_pairs}."
            )

        # The final proof must restore the candidate whose own cold rewrite
        # already passed, never a replay endpoint or a subsequently relaxed
        # live state.
        warm_goal = converged_candidate
        canonical_goal, cold_goal_history = self._canonical_goal_fixed_point(warm_goal)
        exact_success, exact_metrics = advance_exact_success_dwell(
            self.env,
            canonical_goal["task_body_pose"],
            canonical_goal["arm_joint_target"],
            self.closed_finger_target,
            duration_s=self.cfg.goal_passive_settle_s,
            require_all_samples=True,
            sample_physical_validity=True,
            arm_target_is_absolute=True,
            plug_body_index=self.plug_index,
            latch_body_index=self.latch_index,
        )
        final_q, final_qd = self.env.read_task_state()
        goal_body_drift_all = torch.linalg.vector_norm(
            final_q[..., :3] - canonical_goal["task_body_pose"][..., :3],
            dim=-1,
        )
        goal_drift, goal_worst_body = goal_body_drift_all.max(dim=-1)
        socket_drift = goal_body_drift_all[:, self.socket_index]
        final_cable_speed, final_fastest_cable = torch.linalg.vector_norm(
            final_qd[:, self.cable_slice, :3],
            dim=-1,
        ).max(dim=-1)
        goal_grasp = grasp_metrics(self.env, self.closed_finger_target, retaining_grasp=True)
        final_collision = collision_metrics(self.env)
        final_seat_error = torch.linalg.vector_norm(
            final_q[:, self.plug_index, :3] - canonical_target_e,
            dim=-1,
        )
        final_plug_tilt = math_utils.quat_error_magnitude(
            final_q[:, self.plug_index, 3:7],
            authored_plug_orientation,
        )
        goal_valid = (
            passive_goal_valid
            & goal_grasp.valid
            & final_collision.valid
            & exact_success
            & exact_metrics["all_samples_collision_free"]
            & exact_metrics["all_samples_bilateral_grasp"]
            & exact_metrics["all_samples_proxy_bilateral_contact"]
            & exact_metrics["all_samples_finite"]
            & exact_metrics["all_samples_arm_target_tracking_bounded"]
            & ~torch.as_tensor(exact_metrics["any_contact_overflow"], device=self.device)
            & (goal_drift <= self.cfg.maximum_goal_body_drift_m)
            & (exact_metrics["maximum_body_excursion"] <= self.cfg.maximum_goal_body_drift_m)
            & (socket_drift <= self.cfg.maximum_socket_drift_m)
            & (final_cable_speed <= self.cfg.maximum_goal_cable_speed_m_s)
            & (exact_metrics["maximum_cable_linear_speed"] <= self.cfg.maximum_goal_cable_speed_m_s)
            & (exact_metrics["maximum_arm_joint_speed"] <= self.cfg.maximum_goal_arm_joint_speed_rad_s)
            & (exact_metrics["maximum_finger_joint_speed"] <= self.cfg.maximum_goal_finger_joint_speed_m_s)
            & (exact_metrics["maximum_plug_relative_latch_angle"] <= PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD)
            & ~exact_metrics["any_arm_target_clamped"]
            & (exact_metrics["maximum_arm_target_drift"] <= 1.0e-7)
            & (final_seat_error <= self.cfg.maximum_canonical_seat_error_m)
            & (final_plug_tilt <= self.cfg.maximum_authored_plug_angle_rad)
        )
        if not bool(goal_valid.all()):
            raise RuntimeError(
                "Canonical seated goal failed exact cold replay: "
                f"valid={goal_valid.tolist()}, exact={exact_success.tolist()}, "
                f"body_drift={goal_drift.tolist()}, worst_body={goal_worst_body.tolist()}, "
                f"socket_drift={socket_drift.tolist()}, seat_error={final_seat_error.tolist()}, "
                f"plug_tilt={final_plug_tilt.tolist()}, cable_speed={final_cable_speed.tolist()}, "
                f"exact_initial_axial={exact_metrics['initial_signed_axial_error'].tolist()}, "
                f"exact_final_axial={exact_metrics['final_signed_axial_error'].tolist()}, "
                f"exact_max_axial={exact_metrics['maximum_axial_error'].tolist()}, "
                f"exact_max_radial={exact_metrics['maximum_radial_error'].tolist()}, "
                f"exact_max_plug_angle={exact_metrics['maximum_plug_angle_error'].tolist()}, "
                f"exact_max_latch_angle={exact_metrics['maximum_latch_angle_error'].tolist()}, "
                f"exact_max_plug_speed={exact_metrics['maximum_plug_spatial_speed'].tolist()}, "
                f"exact_max_consecutive={exact_metrics['maximum_consecutive_steps'].tolist()}, "
                f"all_samples_collision={exact_metrics['all_samples_collision_free'].tolist()}, "
                f"all_samples_grasp={exact_metrics['all_samples_bilateral_grasp'].tolist()}, "
                f"all_samples_proxy={exact_metrics['all_samples_proxy_bilateral_contact'].tolist()}, "
                f"min_proxy_contacts=(L={exact_metrics['minimum_left_grasp_contact_count'].tolist()}, "
                f"R={exact_metrics['minimum_right_grasp_contact_count'].tolist()}), "
                f"contact_overflow={exact_metrics['any_contact_overflow']}, "
                f"sampled_max_body_excursion={exact_metrics['maximum_body_excursion'].tolist()}, "
                f"sampled_max_arm_speed={exact_metrics['maximum_arm_joint_speed'].tolist()}, "
                f"sampled_max_finger_speed={exact_metrics['maximum_finger_joint_speed'].tolist()}, "
                f"sampled_invalid_pairs={exact_metrics['sampled_invalid_contact_pairs']}, "
                f"invalid_contacts={final_collision.invalid_contact_count.tolist()}."
            )

        canonical = {
            name: value[0].detach().cpu().to(torch.float32).contiguous() for name, value in canonical_goal.items()
        }
        evidence: dict[str, Any] = {
            "passed": True,
            "drive_distance_m": self.cfg.goal_drive_distance_m,
            "drive_direction": "+Y",
            "drive_ramp_s": self.cfg.goal_drive_ramp_s,
            "drive_hold_s": self.cfg.goal_drive_hold_s,
            "drive_overtravel_beyond_authored_seat_m": float(
                torch.linalg.vector_norm(drive_target_w - authored_seat_target_w, dim=-1).max()
            ),
            "orientation_hold_target": "per-world authored plug quaternion",
            "orientation_hold_stiffness": self.env.rj45_runtime.drive_cfg.orientation_stiffness,
            "orientation_hold_damping": self.env.rj45_runtime.drive_cfg.orientation_damping,
            "orientation_hold_enabled_during_ramp_hold_and_acquisition": True,
            "orientation_hold_disabled_before_passive_settle": True,
            "drive_free_stability_s": self.cfg.goal_passive_settle_s,
            "drive_free_stability_steps": int(exact_metrics["sample_steps"]),
            "construction_ramp_steps": ramp_steps,
            "construction_hold_steps": hold_steps,
            "construction_cable_settle_s": self.cfg.goal_drive_cable_settle_s,
            "construction_cable_settle_steps": cable_settle_steps,
            "construction_cable_settle_start_speed_m_s": float(cable_settle_start_speed.max()),
            "construction_cable_settle_start_fastest_segment_index": int(cable_settle_start_fastest[0]),
            "construction_cable_settle_start_fastest_body_index": int(cable_settle_start_fastest_body[0]),
            "construction_cable_settle_start_fastest_body_name": self.layout.body_names[
                int(cable_settle_start_fastest_body[0])
            ],
            "construction_cable_settle_final_speed_m_s": float(cable_settle_final_speed.max()),
            "construction_cable_settle_final_fastest_segment_index": int(cable_settle_final_fastest[0]),
            "construction_cable_settle_final_fastest_body_index": int(cable_settle_final_fastest_body[0]),
            "construction_cable_settle_final_fastest_body_name": self.layout.body_names[
                int(cable_settle_final_fastest_body[0])
            ],
            "construction_cable_settle_body_drift_m": float(cable_settle_body_drift.max()),
            "authored_start_grasp_before_drive": False,
            "ungrasped_orientation_held_newton_drive": True,
            "construction_open_drive_active_latch_angle_rad": float(driven_open_latch_angle.max()),
            "construction_open_authored_seat_error_m": float(driven_open_seat_error.max()),
            "construction_open_authored_plug_tilt_rad": float(driven_open_plug_tilt.max()),
            "construction_open_plug_speed_m_s": float(driven_open_plug_speed.max()),
            "maximum_canonical_seat_error_m": self.cfg.maximum_canonical_seat_error_m,
            "seated_acquisition_while_both_drives_enabled": True,
            "seated_open_clearance_above_cfg_target_m": self.cfg.grasp_open_clearance_m,
            "seated_clearance_tcp_error_m": float(seated_acquisition["preclose_error"].max()),
            "seated_clearance_invalid_contacts": int(
                seated_acquisition["preclose_collision"].invalid_contact_count.sum()
            ),
            "seated_open_descent_ik_valid": bool(seated_acquisition["open_descent_ik_valid"].all()),
            "seated_open_descent_collision_free": bool(seated_acquisition["open_descent_collision_valid"].all()),
            "seated_maximum_open_descent_tcp_error_m": float(
                seated_acquisition["maximum_open_descent_tcp_error"].max()
            ),
            "seated_ik_diagnostics": seated_acquisition["ik_diagnostics"],
            "seated_contact_preclose_invalid_contacts": int(preclose_collision.invalid_contact_count.sum()),
            "seated_contact_tcp_error_m": float(seated_acquisition["contact_position_error"].max()),
            "seated_final_tcp_distance_m": float(preclose_tcp_distance.max()),
            "seated_bilateral_grasp": bool(release_grasp.valid.all()),
            "seated_left_proxy_contacts": int(release_collision.left_grasp_contact_count.sum()),
            "seated_right_proxy_contacts": int(release_collision.right_grasp_contact_count.sum()),
            "seated_release_invalid_contact_pairs": release_collision.invalid_contact_pairs,
            "construction_final_authored_seat_error_m": float(driven_seat_error.max()),
            "construction_final_authored_plug_tilt_rad": float(driven_plug_tilt.max()),
            "construction_final_arm_target_error_rad": float(driven_arm_target_error.max()),
            "construction_release_arm_bias_max_rad": float(torch.abs(release_arm_bias).max()),
            "construction_settled_arm_bias_max_rad": float(settled_arm_bias_magnitude.max()),
            "construction_release_to_settled_arm_drift_max_rad": float(release_to_settled_arm_drift.max()),
            "construction_relative_bias_verification_arm_drift_max_rad": float(passive_arm_drift.max()),
            "maximum_goal_arm_bias_rad": self.cfg.maximum_goal_arm_bias_rad,
            "maximum_goal_arm_drift_rad": self.cfg.maximum_goal_arm_drift_rad,
            "maximum_goal_arm_joint_speed_rad_s": self.cfg.maximum_goal_arm_joint_speed_rad_s,
            "maximum_goal_finger_joint_speed_m_s": self.cfg.maximum_goal_finger_joint_speed_m_s,
            "single_deterministic_sampler_free_ik_solver_owner": True,
            "ik_solver_seed_count": _ROW_IK_SEED_COUNT,
            "ik_solver_iterations": _ROW_IK_ITERATIONS,
            "construction_passive_early_steps": early_steps,
            "construction_passive_stability_steps": stability_steps,
            "construction_passive_pre_stability": passive_pre_stability_history,
            "cold_rewrite_equilibrium_cycles": cold_rewrite_equilibrium_history,
            "cold_rewrite_endpoint_promotions": cold_rewrite_endpoint_promotion_history,
            "cold_rewrite_converged_cycle": len(cold_rewrite_equilibrium_history),
            "peak_latch_crossing_angle_rad": float(peak_latch_angle.max()),
            "peak_latch_crossing_threshold_rad": self.cfg.canonical_peak_latch_angle_min_rad,
            "driven_latch_angle_rad": float(driven_latch_angle.max()),
            "final_latch_angle_rad": float(passive_latch_angle.max()),
            "maximum_passive_seat_error_m": float(passive_seat_error.max()),
            "maximum_passive_plug_tilt_rad": float(passive_plug_tilt.max()),
            "maximum_passive_body_drift_m": float(passive_body_drift.max()),
            "passive_worst_body_index": int(passive_worst_body[0]),
            "passive_worst_body_name": self.layout.body_names[int(passive_worst_body[0])],
            "maximum_passive_start_cable_speed_m_s": float(passive_start_cable_speed.max()),
            "passive_start_fastest_cable_segment_index": int(passive_start_fastest[0]),
            "passive_start_fastest_cable_body_index": int(passive_start_fastest_body[0]),
            "passive_start_fastest_cable_body_name": self.layout.body_names[int(passive_start_fastest_body[0])],
            "passive_start_fastest_cable_spatial_velocity": passive_start_fastest_qd[0].detach().cpu().tolist(),
            "passive_start_fastest_cable_position_m": passive_start_fastest_position[0].detach().cpu().tolist(),
            "passive_start_fastest_cable_support_clearance_m": float(passive_start_fastest_support_clearance[0]),
            "maximum_passive_final_cable_speed_m_s": float(passive_cable_speed.max()),
            "passive_final_fastest_cable_segment_index": int(passive_fastest[0]),
            "passive_final_fastest_cable_body_index": int(passive_fastest_body[0]),
            "passive_final_fastest_cable_body_name": self.layout.body_names[int(passive_fastest_body[0])],
            "passive_final_fastest_cable_spatial_velocity": passive_fastest_qd[0].detach().cpu().tolist(),
            "passive_final_fastest_cable_position_m": passive_fastest_position[0].detach().cpu().tolist(),
            "passive_final_fastest_cable_support_clearance_m": float(passive_fastest_support_clearance[0]),
            "cold_goal_fixed_point": cold_goal_history,
            "maximum_goal_replay_body_drift_m": float(goal_drift.max()),
            "goal_replay_worst_body_index": int(goal_worst_body[0]),
            "goal_replay_worst_body_name": self.layout.body_names[int(goal_worst_body[0])],
            "maximum_goal_replay_socket_drift_m": float(socket_drift.max()),
            "maximum_goal_replay_seat_error_m": float(final_seat_error.max()),
            "maximum_goal_replay_plug_tilt_rad": float(final_plug_tilt.max()),
            "maximum_goal_replay_cable_speed_m_s": float(final_cable_speed.max()),
            "goal_replay_fastest_cable_segment_index": int(final_fastest_cable[0]),
            "goal_replay_fastest_cable_body_index": int(final_fastest_cable[0]) + self.cable_body_start,
            "goal_replay_fastest_cable_body_name": self.layout.body_names[
                int(final_fastest_cable[0]) + self.cable_body_start
            ],
            "exact_runtime_success_dwell": bool(exact_success.all()),
            "exact_runtime_success_required_steps": int(exact_metrics["required_dwell_steps"]),
            "exact_all_samples_collision_free": bool(exact_metrics["all_samples_collision_free"].all()),
            "exact_all_samples_bilateral_grasp": bool(exact_metrics["all_samples_bilateral_grasp"].all()),
            "exact_all_samples_proxy_bilateral_contact": bool(
                exact_metrics["all_samples_proxy_bilateral_contact"].all()
            ),
            "exact_minimum_left_proxy_contact_count": int(exact_metrics["minimum_left_grasp_contact_count"].min()),
            "exact_minimum_right_proxy_contact_count": int(exact_metrics["minimum_right_grasp_contact_count"].min()),
            "exact_any_contact_overflow": exact_metrics["any_contact_overflow"],
            "exact_maximum_plug_relative_latch_angle_rad": float(
                exact_metrics["maximum_plug_relative_latch_angle"].max()
            ),
            "construction_translation_drive_disabled_in_snapshot": True,
            "construction_orientation_hold_disabled_in_snapshot": True,
            "closed_bilateral_grasp": bool(goal_grasp.valid.all()),
            "collision_valid": bool(final_collision.valid.all()),
            "passive_invalid_contact_pairs": passive_collision.invalid_contact_pairs,
            "goal_replay_invalid_contact_pairs": final_collision.invalid_contact_pairs,
        }
        return canonical, evidence

    def _row_goal(
        self,
        canonical_goal: dict[str, torch.Tensor],
        socket_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        canonical_q = canonical_goal["task_body_pose"].to(self.device).unsqueeze(0).repeat(self.env.num_envs, 1, 1)
        canonical_qd = canonical_goal["task_body_velocity"].to(self.device).unsqueeze(0).repeat(self.env.num_envs, 1, 1)
        source_socket = canonical_q[:, self.socket_index]
        goal_q, goal_qd = _rigid_transform_task_state(canonical_q, canonical_qd, source_socket, socket_pose)
        tcp_position, tcp_orientation = self._desired_tcp_pose(goal_q[:, self.plug_index])
        seed = canonical_goal["arm_joint_target"].to(self.device).repeat(self.env.num_envs, 1)
        goal_ik = self._solve_ik(
            tcp_position,
            tcp_orientation,
            self.closed_finger_target,
            arm_seed=seed,
        )
        self._last_goal_ik_result = goal_ik
        canonical_bias = (canonical_goal["arm_joint_target"] - canonical_goal["arm_joint_position"]).to(self.device)
        goal_arm_target = goal_ik.arm_q + canonical_bias
        valid = goal_ik.valid & joint_limit_mask(self.env, goal_arm_target)
        return goal_q, goal_qd, torch.where(valid[:, None], goal_arm_target, goal_ik.arm_q), valid

    def _robot_position_targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Read the live arm/finger position targets used by the reset tool."""
        targets = self.env._robot.data.joint_pos_target.torch
        return (
            targets[:, self.env._arm_joint_ids].clone(),
            targets[:, self.env._finger_joint_ids].clone(),
        )

    def _write_task_state_preserving_staged_robot(
        self,
        task_q: torch.Tensor,
        task_qd: torch.Tensor,
        *,
        staged_arm_target: torch.Tensor,
        staged_finger_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Write only task bodies and prove every staged robot field stayed bitwise unchanged."""
        before_arm_q, before_arm_qd, before_finger_q, before_finger_qd = self.env.read_robot_state()
        before_arm_target, before_finger_target = self._robot_position_targets()
        self.env.write_task_state(task_q, task_qd)
        after_arm_q, after_arm_qd, after_finger_q, after_finger_qd = self.env.read_robot_state()
        after_arm_target, after_finger_target = self._robot_position_targets()
        unchanged = (
            torch.eq(after_arm_q, before_arm_q).all(dim=-1)
            & torch.eq(after_arm_qd, before_arm_qd).all(dim=-1)
            & torch.eq(after_finger_q, before_finger_q).all(dim=-1)
            & torch.eq(after_finger_qd, before_finger_qd).all(dim=-1)
            & torch.eq(after_arm_target, before_arm_target).all(dim=-1)
            & torch.eq(after_finger_target, before_finger_target).all(dim=-1)
            & torch.eq(after_arm_target, staged_arm_target).all(dim=-1)
            & torch.eq(after_finger_target, staged_finger_target).all(dim=-1)
        )
        evidence = {
            "unchanged": unchanged,
            "maximum_arm_position_delta_rad": torch.abs(after_arm_q - before_arm_q).amax(dim=-1),
            "maximum_arm_velocity_delta_rad_s": torch.abs(after_arm_qd - before_arm_qd).amax(dim=-1),
            "maximum_finger_position_delta_m": torch.abs(after_finger_q - before_finger_q).amax(dim=-1),
            "maximum_finger_velocity_delta_m_s": torch.abs(after_finger_qd - before_finger_qd).amax(dim=-1),
            "maximum_arm_target_delta_rad": torch.abs(after_arm_target - before_arm_target).amax(dim=-1),
            "maximum_finger_target_delta_m": torch.abs(after_finger_target - before_finger_target).amax(dim=-1),
        }
        if not bool(unchanged.all()):
            raise RuntimeError(
                "Task-only loose-cable placement mutated staged robot state or targets: "
                f"{_plain_certificate_value(evidence)}."
            )
        return evidence

    def _stage_robot_for_pickup(
        self,
        pickup_pose: torch.Tensor,
        *,
        active_mask: torch.Tensor,
        orientation_error_xyzw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Kinematically stage the open gripper above the future loose-plug pose."""
        active_mask = torch.as_tensor(active_mask, device=self.device, dtype=torch.bool)
        entry_arm_q, _, _, _ = self.env.read_robot_state()
        entry_arm_target, _ = self._robot_position_targets()
        entry_equilibrium_bias = entry_arm_target - entry_arm_q
        tracking_error_limits = torch.as_tensor(
            self.env.cfg.actions.arm_action.tracking_error_limits,
            device=self.device,
            dtype=entry_arm_q.dtype,
        )
        entry_bias_finite = torch.isfinite(entry_equilibrium_bias).all(dim=-1)
        entry_bias_within_tracking_limits = (torch.abs(entry_equilibrium_bias) <= tracking_error_limits).all(dim=-1)
        entry_bias_valid = entry_bias_finite & entry_bias_within_tracking_limits
        clearance_position, clearance_orientation = self._desired_tcp_pose(
            pickup_pose,
            orientation_error_xyzw=orientation_error_xyzw,
        )
        clearance_position = clearance_position.clone()
        clearance_position[:, 2] += self.cfg.grasp_open_clearance_m
        ik_count_before = self._ik_solve_call_count
        solution = self._solve_ik(
            clearance_position,
            clearance_orientation,
            self.open_finger_q,
            arm_seed=self.home_arm_q,
        )
        solution_finite = torch.isfinite(solution.arm_q).all(dim=-1)
        solution_joint_limit_valid = joint_limit_mask(self.env, solution.arm_q, margin=0.02)
        biased_staged_arm_target = solution.arm_q + entry_equilibrium_bias
        biased_target_finite = torch.isfinite(biased_staged_arm_target).all(dim=-1)
        biased_target_joint_limit_valid = joint_limit_mask(
            self.env,
            biased_staged_arm_target,
            margin=0.02,
        )
        solve_valid = (
            active_mask
            & solution.valid
            & solution_finite
            & solution_joint_limit_valid
            & entry_bias_valid
            & biased_target_finite
            & biased_target_joint_limit_valid
        )
        staged_raw_arm_q = torch.where(solve_valid[:, None], solution.arm_q, self.home_arm_q)
        staged_arm_target = torch.where(
            solve_valid[:, None],
            biased_staged_arm_target,
            self.home_arm_q,
        )
        safe_raw_q_valid = torch.isfinite(staged_raw_arm_q).all(dim=-1) & joint_limit_mask(
            self.env,
            staged_raw_arm_q,
            margin=0.02,
        )
        safe_target_valid = torch.isfinite(staged_arm_target).all(dim=-1) & joint_limit_mask(
            self.env,
            staged_arm_target,
            margin=0.02,
        )
        if not bool((safe_raw_q_valid & safe_target_valid).all()):
            raise RuntimeError("Pickup staging did not produce finite joint-limit-valid safe robot states.")
        self.env.write_robot_state(
            staged_raw_arm_q,
            self.open_finger_q,
            arm_target=staged_arm_target,
            arm_qd=torch.zeros_like(staged_raw_arm_q),
            finger_qd=torch.zeros_like(self.open_finger_q),
            finger_target=self.open_finger_q,
        )
        self.env.flush_reset_history()

        sample_count = 0
        all_finite = torch.ones_like(active_mask)
        all_collision_free = torch.ones_like(active_mask)
        all_zero_proxy_contacts = torch.ones_like(active_mask)
        all_expected_drive_state = torch.ones_like(active_mask)
        all_tracking_bounded = torch.ones_like(active_mask)
        all_arm_speed_bounded = torch.ones_like(active_mask)
        maximum_tcp_position_error = torch.zeros(self.env.num_envs, device=self.device)
        maximum_tcp_orientation_error = torch.zeros_like(maximum_tcp_position_error)
        maximum_arm_speed = torch.zeros_like(maximum_tcp_position_error)
        any_contact_overflow = False
        invalid_contact_pairs: list[str] = []

        def sample_staging_settle(_step: int, _steps: int, _progress: float) -> None:
            nonlocal sample_count, any_contact_overflow
            sample_count += 1
            task_q, task_qd = self.env.read_task_state()
            arm_q, arm_qd, finger_q, finger_qd = self.env.read_robot_state()
            tcp = self.env.tcp_pose_e()
            collision = collision_metrics(self.env, require_bilateral_grasp=False)
            if collision.contact_overflow:
                raise RuntimeError("Global contact-buffer overflow during kinematic robot staging.")
            tcp_position_error = torch.linalg.vector_norm(tcp[:, :3] - clearance_position, dim=-1)
            tcp_orientation_error = math_utils.quat_error_magnitude(tcp[:, 3:7], clearance_orientation)
            arm_speed = torch.abs(arm_qd).amax(dim=-1)
            finite = (
                task_state_is_finite_and_normalized(task_q, task_qd)
                & torch.isfinite(arm_q).all(dim=-1)
                & torch.isfinite(arm_qd).all(dim=-1)
                & torch.isfinite(finger_q).all(dim=-1)
                & torch.isfinite(finger_qd).all(dim=-1)
                & torch.isfinite(tcp).all(dim=-1)
            )
            zero_proxy = (collision.left_grasp_contact_count == 0) & (collision.right_grasp_contact_count == 0)
            expected_drive_state = self._drive_enabled() & self._orientation_hold_enabled()
            all_finite.logical_and_(finite)
            all_collision_free.logical_and_(collision.valid)
            all_zero_proxy_contacts.logical_and_(zero_proxy)
            all_expected_drive_state.logical_and_(expected_drive_state)
            all_tracking_bounded.logical_and_(
                (tcp_position_error <= _LOCAL_PICKUP_CLEARANCE_TRANSLATION_TOLERANCE_M)
                & (tcp_orientation_error <= _LOCAL_PICKUP_CLEARANCE_ROTATION_TOLERANCE_RAD)
            )
            all_arm_speed_bounded.logical_and_(arm_speed <= self.cfg.maximum_row_arm_joint_speed_rad_s)
            maximum_tcp_position_error.copy_(torch.maximum(maximum_tcp_position_error, tcp_position_error))
            maximum_tcp_orientation_error.copy_(torch.maximum(maximum_tcp_orientation_error, tcp_orientation_error))
            maximum_arm_speed.copy_(torch.maximum(maximum_arm_speed, arm_speed))
            any_contact_overflow |= collision.contact_overflow
            for pair in collision.invalid_contact_pairs:
                if pair not in invalid_contact_pairs and len(invalid_contact_pairs) < 64:
                    invalid_contact_pairs.append(pair)

        control_steps = self.env.advance(
            self.cfg.robot_park_s,
            lambda _step, _steps, _progress: self.env.set_robot_targets(
                staged_arm_target,
                self.open_finger_q,
            ),
            post_step=sample_staging_settle,
        )
        stage_valid = (
            solve_valid
            & all_finite
            & all_collision_free
            & all_zero_proxy_contacts
            & all_expected_drive_state
            & all_tracking_bounded
            & all_arm_speed_bounded
            & (not any_contact_overflow)
        )
        return (
            staged_arm_target,
            stage_valid,
            {
                "construction_sequence_version": _PICKUP_CONSTRUCTION_SEQUENCE_VERSION,
                "staging_mode": "kinematic_joint_state_write",
                "persistent_target_mode": "raw-stage-q-plus-parked-home-equilibrium-bias",
                "active_mask": active_mask,
                "solver_valid": solution.valid,
                "solver_solution_finite": solution_finite,
                "solver_joint_limit_valid": solution_joint_limit_valid,
                "stage_entry_measured_arm_q": entry_arm_q,
                "stage_entry_arm_target": entry_arm_target,
                "stage_entry_equilibrium_bias": entry_equilibrium_bias,
                "stage_entry_bias_finite": entry_bias_finite,
                "stage_entry_bias_within_tracking_limits": entry_bias_within_tracking_limits,
                "stage_entry_tracking_error_limits": tracking_error_limits,
                "biased_target_finite": biased_target_finite,
                "biased_target_joint_limit_valid": biased_target_joint_limit_valid,
                "valid": stage_valid,
                "clearance_target_position": clearance_position,
                "clearance_target_orientation": clearance_orientation,
                "raw_staged_arm_q": staged_raw_arm_q,
                "biased_staged_arm_target": staged_arm_target,
                "staged_arm_target": staged_arm_target,
                "staged_finger_target": self.open_finger_q,
                "sample_count": sample_count,
                "advance_call_count": 1,
                "control_step_count": control_steps,
                "ik_solve_call_count": self._ik_solve_call_count - ik_count_before,
                "all_samples_finite": all_finite,
                "all_samples_collision_free": all_collision_free,
                "all_samples_zero_proxy_contacts": all_zero_proxy_contacts,
                "all_samples_expected_construction_drive_state": all_expected_drive_state,
                "all_samples_tracking_bounded": all_tracking_bounded,
                "all_samples_arm_speed_bounded": all_arm_speed_bounded,
                "maximum_tcp_position_error_m": maximum_tcp_position_error,
                "maximum_tcp_orientation_error_rad": maximum_tcp_orientation_error,
                "maximum_arm_joint_speed_rad_s": maximum_arm_speed,
                "any_contact_overflow": any_contact_overflow,
                "invalid_contact_pairs": invalid_contact_pairs,
            },
        )

    def _construct_pickup(
        self,
        socket_pose: torch.Tensor,
        pickup_pose: torch.Tensor,
        *,
        acquire: bool,
        active_mask: torch.Tensor | None = None,
        diagnostic_evidence: dict[str, Any] | None = None,
        pregrasp_orientation_error_xyzw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Physically place the loose cable, then optionally acquire it drive-free."""
        if acquire and pregrasp_orientation_error_xyzw is not None:
            raise ValueError("Starts-grasped pickup construction must retain the canonical grasp orientation.")
        if active_mask is None:
            active_mask = torch.ones(self.env.num_envs, device=self.device, dtype=torch.bool)
        else:
            active_mask = torch.as_tensor(active_mask, device=self.device, dtype=torch.bool)
        self._park_and_place_socket(socket_pose, hold_plug_orientation=True)
        staged_arm_target, staging_valid, staging_evidence = self._stage_robot_for_pickup(
            pickup_pose,
            active_mask=active_mask,
            orientation_error_xyzw=pregrasp_orientation_error_xyzw,
        )
        staged_finger_target = self.open_finger_q
        parked_q, parked_qd = self.env.read_task_state()

        # Randomize the entire loose connector/cable assembly coherently.  The
        # socket remains at its independently sampled pose, while one rigid
        # transform preserves every plug/latch/cable rest-space relationship.
        # Moving only the plug would preload the long cable against its pinned
        # tail and makes the subsequent drive-free pickup physically invalid.
        transformed_q, _transformed_qd = _rigid_transform_task_state(
            parked_q,
            parked_qd,
            parked_q[:, self.plug_index],
            pickup_pose,
        )
        loose_body_mask = torch.ones(self.layout.body_count, device=self.device, dtype=torch.bool)
        loose_body_mask[self.socket_index] = False
        placed_q = parked_q.clone()
        placed_qd = torch.zeros_like(parked_qd)
        placed_q[:, loose_body_mask] = transformed_q[:, loose_body_mask]
        # A coherent rigid placement has no construction impulse.  The real
        # VBD solve below establishes cable/slab equilibrium before acquisition.
        source_loose_q = parked_q[:, loose_body_mask]
        placed_loose_q = placed_q[:, loose_body_mask]
        source_pairwise_delta = source_loose_q[:, :, None, :3] - source_loose_q[:, None, :, :3]
        placed_pairwise_delta = placed_loose_q[:, :, None, :3] - placed_loose_q[:, None, :, :3]
        rigid_pairwise_distance_error = torch.abs(
            torch.linalg.vector_norm(source_pairwise_delta, dim=-1)
            - torch.linalg.vector_norm(placed_pairwise_delta, dim=-1)
        ).amax(dim=(-2, -1))
        source_plug_inverse = math_utils.quat_conjugate(parked_q[:, self.plug_index, 3:7])
        placed_plug_inverse = math_utils.quat_conjugate(placed_q[:, self.plug_index, 3:7])
        source_relative_orientation = math_utils.quat_mul(
            source_plug_inverse[:, None, :].expand_as(source_loose_q[..., 3:7]),
            source_loose_q[..., 3:7],
        )
        placed_relative_orientation = math_utils.quat_mul(
            placed_plug_inverse[:, None, :].expand_as(placed_loose_q[..., 3:7]),
            placed_loose_q[..., 3:7],
        )
        rigid_relative_orientation_error = math_utils.quat_error_magnitude(
            source_relative_orientation,
            placed_relative_orientation,
        ).amax(dim=-1)
        task_only_robot_evidence = self._write_task_state_preserving_staged_robot(
            placed_q,
            placed_qd,
            staged_arm_target=staged_arm_target,
            staged_finger_target=staged_finger_target,
        )
        written_q, written_qd = self.env.read_task_state()
        write_position_error = torch.linalg.vector_norm(written_q[..., :3] - placed_q[..., :3], dim=-1).amax(dim=-1)
        write_orientation_error = math_utils.quat_error_magnitude(
            written_q[..., 3:7],
            placed_q[..., 3:7],
        ).amax(dim=-1)
        written_speed = torch.linalg.vector_norm(written_qd, dim=-1).amax(dim=-1)
        source_cable_q = parked_q[:, self.cable_slice]
        written_cable_q = written_q[:, self.cable_slice]
        source_cable_lengths = torch.linalg.vector_norm(
            source_cable_q[:, 1:, :3] - source_cable_q[:, :-1, :3],
            dim=-1,
        )
        written_cable_lengths = torch.linalg.vector_norm(
            written_cable_q[:, 1:, :3] - written_cable_q[:, :-1, :3],
            dim=-1,
        )
        cable_rest_length_error = torch.abs(written_cable_lengths - source_cable_lengths).amax(dim=-1)

        def plug_relative_position(task_q: torch.Tensor, body_q: torch.Tensor) -> torch.Tensor:
            plug_q = task_q[:, self.plug_index]
            inverse = math_utils.quat_conjugate(plug_q[:, 3:7])
            return math_utils.quat_apply(
                inverse[:, None, :].expand_as(body_q[..., 3:7]),
                body_q[..., :3] - plug_q[:, None, :3],
            )

        source_anchor_q = source_cable_q[:, :CABLE_KINEMATIC_COUNT]
        written_anchor_q = written_cable_q[:, :CABLE_KINEMATIC_COUNT]
        anchor_position_error = torch.linalg.vector_norm(
            plug_relative_position(written_q, written_anchor_q) - plug_relative_position(parked_q, source_anchor_q),
            dim=-1,
        ).amax(dim=-1)
        source_anchor_relative_orientation = math_utils.quat_mul(
            math_utils.quat_conjugate(parked_q[:, self.plug_index, 3:7])[:, None, :].expand_as(
                source_anchor_q[..., 3:7]
            ),
            source_anchor_q[..., 3:7],
        )
        written_anchor_relative_orientation = math_utils.quat_mul(
            math_utils.quat_conjugate(written_q[:, self.plug_index, 3:7])[:, None, :].expand_as(
                written_anchor_q[..., 3:7]
            ),
            written_anchor_q[..., 3:7],
        )
        anchor_orientation_error = math_utils.quat_error_magnitude(
            written_anchor_relative_orientation,
            source_anchor_relative_orientation,
        ).amax(dim=-1)
        tail_start = CABLE_KINEMATIC_COUNT
        placed_cable_q = placed_q[:, self.cable_slice]
        tail_position_error = torch.linalg.vector_norm(
            written_cable_q[:, tail_start:, :3] - placed_cable_q[:, tail_start:, :3],
            dim=-1,
        ).amax(dim=-1)
        tail_orientation_error = math_utils.quat_error_magnitude(
            written_cable_q[:, tail_start:, 3:7],
            placed_cable_q[:, tail_start:, 3:7],
        ).amax(dim=-1)
        socket_write_error = torch.linalg.vector_norm(
            written_q[:, self.socket_index, :3] - parked_q[:, self.socket_index, :3],
            dim=-1,
        )
        placement_invariants_valid = (
            (rigid_pairwise_distance_error <= 1.0e-6)
            & (rigid_relative_orientation_error <= 1.0e-6)
            & (write_position_error <= 1.0e-6)
            & (write_orientation_error <= 1.0e-6)
            & (written_speed <= 1.0e-7)
            & (cable_rest_length_error <= 1.0e-6)
            & (anchor_position_error <= 1.0e-5)
            & (anchor_orientation_error <= 1.0e-5)
            & (tail_position_error <= 1.0e-6)
            & (tail_orientation_error <= 1.0e-6)
            & (socket_write_error <= 1.0e-7)
        )

        plug_end_w = pickup_pose[:, :3] + self.env.env_origins
        self.env.set_drive(True, plug_end_w)
        self.env.set_orientation_hold(True, pickup_pose[:, 3:7])
        self.env.set_robot_targets(staged_arm_target, staged_finger_target)
        self.env.flush_reset_history()
        driven_first_sample_count = 0
        driven_first_collision_valid = torch.zeros_like(active_mask)
        driven_first_zero_proxy_contacts = torch.zeros_like(active_mask)
        driven_first_collision: Any | None = None

        def sample_first_driven_step(_step: int, _steps: int, _progress: float) -> None:
            nonlocal driven_first_collision, driven_first_sample_count
            if driven_first_sample_count:
                return
            driven_first_sample_count = 1
            driven_first_collision = collision_metrics(self.env, require_bilateral_grasp=False)
            if driven_first_collision.contact_overflow:
                raise RuntimeError("Global contact-buffer overflow after coherent task-placement flush.")
            driven_first_collision_valid.copy_(driven_first_collision.valid)
            driven_first_zero_proxy_contacts.copy_(
                (driven_first_collision.left_grasp_contact_count == 0)
                & (driven_first_collision.right_grasp_contact_count == 0)
            )

        driven_control_step_count = self.env.advance(
            self.cfg.pickup_drive_hold_s,
            lambda _step, _steps, _progress: self.env.set_robot_targets(
                staged_arm_target,
                staged_finger_target,
            ),
            post_step=sample_first_driven_step,
        )
        if driven_first_sample_count != 1 or driven_first_collision is None:
            raise RuntimeError("Coherent task placement produced no real post-flush collision sample.")
        driven_q, driven_qd = self.env.read_task_state()
        driven_position_error = torch.linalg.vector_norm(
            driven_q[:, self.plug_index, :3] - pickup_pose[:, :3],
            dim=-1,
        )
        driven_orientation_error = math_utils.quat_error_magnitude(
            driven_q[:, self.plug_index, 3:7],
            pickup_pose[:, 3:7],
        )

        # The construction constraints are only placement aids.  The cable,
        # plug, and robot must establish a genuine slab-supported state before
        # any grasp, snapshot, or oracle motion.
        self.env.set_drive(False)
        self._assert_drive_disabled("randomized loose-pickup release")
        settle_all_finite = torch.ones(self.env.num_envs, device=self.device, dtype=torch.bool)
        settle_all_collision_free = torch.ones_like(settle_all_finite)
        settle_all_drives_disabled = torch.ones_like(settle_all_finite)
        settle_maximum_cable_speed = torch.zeros(self.env.num_envs, device=self.device)
        settle_maximum_plug_linear_speed = torch.zeros_like(settle_maximum_cable_speed)
        settle_maximum_plug_angular_speed = torch.zeros_like(settle_maximum_cable_speed)
        settle_minimum_cable_clearance = torch.full_like(settle_maximum_cable_speed, torch.inf)
        settle_any_contact_overflow = False
        settle_invalid_pairs: list[str] = []
        settle_sample_count = 0

        def sample_drive_free_settle(_step: int, _steps: int, _progress: float) -> None:
            nonlocal settle_any_contact_overflow, settle_sample_count
            settle_sample_count += 1
            task_q, task_qd = self.env.read_task_state()
            collision = collision_metrics(self.env, require_bilateral_grasp=False)
            if collision.contact_overflow:
                raise RuntimeError("Global contact-buffer overflow during randomized pickup settling.")
            cable_speed = torch.linalg.vector_norm(task_qd[:, self.cable_slice, :3], dim=-1).amax(dim=-1)
            plug_linear = torch.linalg.vector_norm(task_qd[:, self.plug_index, :3], dim=-1)
            plug_angular = torch.linalg.vector_norm(task_qd[:, self.plug_index, 3:6], dim=-1)
            cable_clearance = task_q[:, self.cable_slice, 2].amin(dim=-1) - CABLE_RADIUS
            settle_all_finite.logical_and_(task_state_is_finite_and_normalized(task_q, task_qd))
            settle_all_collision_free.logical_and_(collision.valid)
            settle_all_drives_disabled.logical_and_(~self._drive_enabled() & ~self._orientation_hold_enabled())
            settle_maximum_cable_speed.copy_(torch.maximum(settle_maximum_cable_speed, cable_speed))
            settle_maximum_plug_linear_speed.copy_(torch.maximum(settle_maximum_plug_linear_speed, plug_linear))
            settle_maximum_plug_angular_speed.copy_(torch.maximum(settle_maximum_plug_angular_speed, plug_angular))
            settle_minimum_cable_clearance.copy_(torch.minimum(settle_minimum_cable_clearance, cable_clearance))
            settle_any_contact_overflow |= collision.contact_overflow
            for pair in collision.invalid_contact_pairs:
                if pair not in settle_invalid_pairs and len(settle_invalid_pairs) < 64:
                    settle_invalid_pairs.append(pair)

        free_settle_control_step_count = self.env.advance(
            self.cfg.pickup_drive_free_settle_s,
            lambda _step, _steps, _progress: self.env.set_robot_targets(
                staged_arm_target,
                staged_finger_target,
            ),
            post_step=sample_drive_free_settle,
        )
        self._assert_drive_disabled("randomized loose-pickup settling")
        settled_q, settled_qd = self.env.read_task_state()
        settled_collision = collision_metrics(self.env, require_bilateral_grasp=False)
        if settled_collision.contact_overflow:
            raise RuntimeError("Global contact-buffer overflow at the randomized pickup result boundary.")
        settled_position_error = torch.linalg.vector_norm(
            settled_q[:, self.plug_index, :3] - pickup_pose[:, :3],
            dim=-1,
        )
        settled_orientation_error = math_utils.quat_error_magnitude(
            settled_q[:, self.plug_index, 3:7],
            pickup_pose[:, 3:7],
        )
        live_reference_plug = settled_q[:, self.plug_index].clone()
        live_clearance_position, live_clearance_orientation = self._desired_tcp_pose(
            live_reference_plug,
            orientation_error_xyzw=pregrasp_orientation_error_xyzw,
        )
        live_clearance_position = live_clearance_position.clone()
        live_clearance_position[:, 2] += self.cfg.grasp_open_clearance_m
        staged_tcp = self.env.tcp_pose_e()
        live_clearance_position_error = torch.linalg.vector_norm(
            staged_tcp[:, :3] - live_clearance_position,
            dim=-1,
        )
        live_clearance_orientation_error = math_utils.quat_error_magnitude(
            staged_tcp[:, 3:7],
            live_clearance_orientation,
        )
        live_clearance_alignment_valid = (
            live_clearance_position_error <= _LOCAL_PICKUP_CLEARANCE_TRANSLATION_TOLERANCE_M
        ) & (live_clearance_orientation_error <= _LOCAL_PICKUP_CLEARANCE_ROTATION_TOLERANCE_RAD)
        socket_drift = torch.linalg.vector_norm(
            settled_q[:, self.socket_index, :3] - socket_pose[:, :3],
            dim=-1,
        )
        cable_linear_speed = torch.linalg.vector_norm(
            settled_qd[:, self.cable_slice, :3],
            dim=-1,
        ).amax(dim=-1)
        plug_linear_speed = torch.linalg.vector_norm(settled_qd[:, self.plug_index, :3], dim=-1)
        plug_angular_speed = torch.linalg.vector_norm(settled_qd[:, self.plug_index, 3:6], dim=-1)
        minimum_cable_support_clearance = settled_q[:, self.cable_slice, 2].amin(dim=-1) - CABLE_RADIUS
        workspace_lower = torch.as_tensor(
            self.env.cfg.task_body_workspace_lower,
            device=self.device,
            dtype=torch.float32,
        )
        workspace_upper = torch.as_tensor(
            self.env.cfg.task_body_workspace_upper,
            device=self.device,
            dtype=torch.float32,
        )
        loose_body_positions = settled_q[:, loose_body_mask, :3]
        loose_bodies_in_workspace = (
            ((loose_body_positions >= workspace_lower) & (loose_body_positions <= workspace_upper))
            .all(dim=-1)
            .all(dim=-1)
        )
        contact_surface = self.env.cfg.scene.table_contact_surface
        slab_center = torch.as_tensor(contact_surface.init_state.pos, device=self.device, dtype=torch.float32)
        slab_size = torch.as_tensor(contact_surface.spawn.size, device=self.device, dtype=torch.float32)
        slab_lower_xy = slab_center[:2] - 0.5 * slab_size[:2] + CABLE_RADIUS
        slab_upper_xy = slab_center[:2] + 0.5 * slab_size[:2] - CABLE_RADIUS
        cable_xy = settled_q[:, self.cable_slice, :2]
        cable_centers_in_slab_footprint = (
            ((cable_xy >= slab_lower_xy) & (cable_xy <= slab_upper_xy)).all(dim=-1).all(dim=-1)
        )
        cable_socket_center_distance = torch.linalg.vector_norm(
            settled_q[:, self.cable_slice, :3] - settled_q[:, self.socket_index, None, :3],
            dim=-1,
        ).amin(dim=-1)
        cable_socket_separated = cable_socket_center_distance >= self.cfg.minimum_pickup_cable_socket_center_distance_m
        print(
            "[PICK-INSERT COHERENT PICKUP PLACEMENT] "
            + str(
                {
                    "maximum_pairwise_distance_error_m": rigid_pairwise_distance_error.detach().cpu().tolist(),
                    "maximum_relative_orientation_error_rad": rigid_relative_orientation_error.detach().cpu().tolist(),
                    "driven_position_error_m": driven_position_error.detach().cpu().tolist(),
                    "driven_orientation_error_rad": driven_orientation_error.detach().cpu().tolist(),
                    "drive_free_position_error_m": settled_position_error.detach().cpu().tolist(),
                    "drive_free_orientation_error_rad": settled_orientation_error.detach().cpu().tolist(),
                    "live_clearance_position_error_m": live_clearance_position_error.detach().cpu().tolist(),
                    "live_clearance_orientation_error_rad": live_clearance_orientation_error.detach().cpu().tolist(),
                    "construction_robot_staging_valid": staging_valid.detach().cpu().tolist(),
                    "task_only_robot_state_unchanged": task_only_robot_evidence["unchanged"].detach().cpu().tolist(),
                    "first_driven_sample_collision_valid": driven_first_collision_valid.detach().cpu().tolist(),
                    "first_driven_sample_zero_proxy_contacts": driven_first_zero_proxy_contacts.detach().cpu().tolist(),
                    "drive_free_cable_linear_speed_m_s": cable_linear_speed.detach().cpu().tolist(),
                    "drive_free_plug_linear_speed_m_s": plug_linear_speed.detach().cpu().tolist(),
                    "drive_free_plug_angular_speed_rad_s": plug_angular_speed.detach().cpu().tolist(),
                    "sampled_maximum_cable_speed_m_s": settle_maximum_cable_speed.detach().cpu().tolist(),
                    "sampled_maximum_plug_linear_speed_m_s": settle_maximum_plug_linear_speed.detach().cpu().tolist(),
                    "sampled_maximum_plug_angular_speed_rad_s": settle_maximum_plug_angular_speed.detach()
                    .cpu()
                    .tolist(),
                    "sampled_minimum_cable_support_clearance_m": settle_minimum_cable_clearance.detach().cpu().tolist(),
                    "all_settle_samples_finite": settle_all_finite.detach().cpu().tolist(),
                    "all_settle_samples_collision_free": settle_all_collision_free.detach().cpu().tolist(),
                    "all_settle_samples_drives_disabled": settle_all_drives_disabled.detach().cpu().tolist(),
                    "settle_any_contact_overflow": settle_any_contact_overflow,
                    "settle_invalid_pairs": settle_invalid_pairs,
                    "minimum_cable_support_clearance_m": minimum_cable_support_clearance.detach().cpu().tolist(),
                    "minimum_loose_body_center_height_m": loose_body_positions[..., 2]
                    .amin(dim=-1)
                    .detach()
                    .cpu()
                    .tolist(),
                    "loose_bodies_in_workspace": loose_bodies_in_workspace.detach().cpu().tolist(),
                    "placement_invariants_valid": placement_invariants_valid.detach().cpu().tolist(),
                    "cable_rest_length_error_m": cable_rest_length_error.detach().cpu().tolist(),
                    "anchor_position_error_m": anchor_position_error.detach().cpu().tolist(),
                    "anchor_orientation_error_rad": anchor_orientation_error.detach().cpu().tolist(),
                    "tail_write_position_error_m": tail_position_error.detach().cpu().tolist(),
                    "tail_write_orientation_error_rad": tail_orientation_error.detach().cpu().tolist(),
                    "socket_write_error_m": socket_write_error.detach().cpu().tolist(),
                    "written_maximum_spatial_speed": written_speed.detach().cpu().tolist(),
                    "cable_centers_in_slab_footprint": cable_centers_in_slab_footprint.detach().cpu().tolist(),
                    "minimum_cable_socket_center_distance_m": cable_socket_center_distance.detach().cpu().tolist(),
                    "cable_socket_separated": cable_socket_separated.detach().cpu().tolist(),
                    "collision_valid": settled_collision.valid.detach().cpu().tolist(),
                    "drive_disabled": (~self._drive_enabled()).detach().cpu().tolist(),
                    "orientation_hold_disabled": (~self._orientation_hold_enabled()).detach().cpu().tolist(),
                }
            ),
            flush=True,
        )
        settled_valid = (
            staging_valid
            & task_only_robot_evidence["unchanged"]
            & driven_first_collision_valid
            & driven_first_zero_proxy_contacts
            & task_state_is_finite_and_normalized(driven_q, driven_qd)
            & task_state_is_finite_and_normalized(settled_q, settled_qd)
            & settle_all_finite
            & settle_all_collision_free
            & settle_all_drives_disabled
            & settled_collision.valid
            & ~self._drive_enabled()
            & ~self._orientation_hold_enabled()
            & (driven_position_error <= self.cfg.maximum_pickup_position_error_m)
            & (driven_orientation_error <= self.cfg.maximum_pickup_orientation_error_rad)
            & (settled_position_error <= self.cfg.maximum_pickup_position_error_m)
            & (settled_orientation_error <= self.cfg.maximum_pickup_orientation_error_rad)
            & (socket_drift <= self.cfg.maximum_socket_drift_m)
            & (cable_linear_speed <= self.cfg.maximum_row_cable_speed_m_s)
            & (plug_linear_speed <= self.cfg.maximum_pickup_plug_linear_speed_m_s)
            & (plug_angular_speed <= self.cfg.maximum_pickup_plug_angular_speed_rad_s)
            & (minimum_cable_support_clearance >= -self.cfg.maximum_cable_support_penetration_m)
            & (settle_maximum_cable_speed <= self.cfg.maximum_row_cable_speed_m_s)
            & (settle_maximum_plug_linear_speed <= self.cfg.maximum_pickup_plug_linear_speed_m_s)
            & (settle_maximum_plug_angular_speed <= self.cfg.maximum_pickup_plug_angular_speed_rad_s)
            & (settle_minimum_cable_clearance >= -self.cfg.maximum_cable_support_penetration_m)
            & (not settle_any_contact_overflow)
            & loose_bodies_in_workspace
            & cable_centers_in_slab_footprint
            & cable_socket_separated
            & placement_invariants_valid
            & live_clearance_alignment_valid
        )

        acquisition_valid = torch.ones_like(settled_valid)
        acquisition_evidence: dict[str, Any] | None = None
        arm_target = staged_arm_target
        finger_target = staged_finger_target
        if acquire:
            arm_target, acquisition_valid, acquisition_evidence = self._acquire_prepositioned_current_plug(
                staged_arm_target,
                live_reference_plug,
                active_mask=active_mask & settled_valid & live_clearance_alignment_valid,
            )
            finger_target = acquisition_evidence["last_finger_target"]
            self._assert_drive_disabled("drive-free randomized pickup acquisition")
        valid = active_mask & settled_valid & acquisition_valid
        if diagnostic_evidence is not None:
            placement_gate_masks = {
                "construction-robot-staging-valid": staging_valid,
                "task-only-robot-state-and-targets-unchanged": task_only_robot_evidence["unchanged"],
                "first-driven-sample-collision-valid": driven_first_collision_valid,
                "first-driven-sample-zero-proxy-contacts": driven_first_zero_proxy_contacts,
                "driven-task-state-finite": task_state_is_finite_and_normalized(driven_q, driven_qd),
                "settled-task-state-finite": task_state_is_finite_and_normalized(settled_q, settled_qd),
                "all-settle-samples-finite": settle_all_finite,
                "all-settle-samples-collision-free": settle_all_collision_free,
                "all-settle-samples-drives-disabled": settle_all_drives_disabled,
                "settled-collision-valid": settled_collision.valid,
                "translation-drive-disabled": ~self._drive_enabled(),
                "orientation-hold-disabled": ~self._orientation_hold_enabled(),
                "driven-position-error-bounded": driven_position_error <= self.cfg.maximum_pickup_position_error_m,
                "driven-orientation-error-bounded": (
                    driven_orientation_error <= self.cfg.maximum_pickup_orientation_error_rad
                ),
                "settled-position-error-bounded": settled_position_error <= self.cfg.maximum_pickup_position_error_m,
                "settled-orientation-error-bounded": (
                    settled_orientation_error <= self.cfg.maximum_pickup_orientation_error_rad
                ),
                "socket-drift-bounded": socket_drift <= self.cfg.maximum_socket_drift_m,
                "settled-cable-speed-bounded": cable_linear_speed <= self.cfg.maximum_row_cable_speed_m_s,
                "settled-plug-linear-speed-bounded": (
                    plug_linear_speed <= self.cfg.maximum_pickup_plug_linear_speed_m_s
                ),
                "settled-plug-angular-speed-bounded": (
                    plug_angular_speed <= self.cfg.maximum_pickup_plug_angular_speed_rad_s
                ),
                "settled-cable-clearance-valid": (
                    minimum_cable_support_clearance >= -self.cfg.maximum_cable_support_penetration_m
                ),
                "sampled-cable-speed-bounded": (settle_maximum_cable_speed <= self.cfg.maximum_row_cable_speed_m_s),
                "sampled-plug-linear-speed-bounded": (
                    settle_maximum_plug_linear_speed <= self.cfg.maximum_pickup_plug_linear_speed_m_s
                ),
                "sampled-plug-angular-speed-bounded": (
                    settle_maximum_plug_angular_speed <= self.cfg.maximum_pickup_plug_angular_speed_rad_s
                ),
                "sampled-cable-clearance-valid": (
                    settle_minimum_cable_clearance >= -self.cfg.maximum_cable_support_penetration_m
                ),
                "settle-contact-buffer-no-overflow": torch.full_like(
                    settled_valid,
                    not settle_any_contact_overflow,
                ),
                "loose-bodies-in-workspace": loose_bodies_in_workspace,
                "cable-centers-in-slab-footprint": cable_centers_in_slab_footprint,
                "cable-socket-separated": cable_socket_separated,
                "placement-invariants-valid": placement_invariants_valid,
                "live-clearance-position-aligned": (
                    live_clearance_position_error <= _LOCAL_PICKUP_CLEARANCE_TRANSLATION_TOLERANCE_M
                ),
                "live-clearance-orientation-aligned": (
                    live_clearance_orientation_error <= _LOCAL_PICKUP_CLEARANCE_ROTATION_TOLERANCE_RAD
                ),
            }
            acquisition_failure_masks = (
                {} if acquisition_evidence is None else acquisition_evidence["lane_failure_masks"]
            )
            diagnostic_evidence.update(
                {
                    "construction_sequence_version": _PICKUP_CONSTRUCTION_SEQUENCE_VERSION,
                    "initial_active_mask": active_mask.detach().cpu().tolist(),
                    "survival_mask": valid.detach().cpu().tolist(),
                    "lane_results": _diagnostic_pickup_lane_results(
                        active_mask=active_mask,
                        survival_mask=valid,
                        placement_gate_masks=placement_gate_masks,
                        acquisition_failure_masks=acquisition_failure_masks,
                    ),
                    "construction_robot_staging": _plain_certificate_value(staging_evidence),
                    "coherent_task_placement": {
                        "task_only_robot_state": _plain_certificate_value(task_only_robot_evidence),
                        "driven_advance_call_count": 1,
                        "driven_control_step_count": driven_control_step_count,
                        "first_post_flush_sample_count": driven_first_sample_count,
                        "first_post_flush_collision": _diagnostic_collision_summary(driven_first_collision),
                        "first_post_flush_zero_proxy_contacts": (
                            driven_first_zero_proxy_contacts.detach().cpu().tolist()
                        ),
                    },
                    "drive_free_local_alignment": {
                        "free_settle_advance_call_count": 1,
                        "free_settle_control_step_count": free_settle_control_step_count,
                        "free_settle_sample_count": settle_sample_count,
                        "live_clearance_position_error_m": (live_clearance_position_error.detach().cpu().tolist()),
                        "live_clearance_orientation_error_rad": (
                            live_clearance_orientation_error.detach().cpu().tolist()
                        ),
                        "maximum_position_error_m": _LOCAL_PICKUP_CLEARANCE_TRANSLATION_TOLERANCE_M,
                        "maximum_orientation_error_rad": _LOCAL_PICKUP_CLEARANCE_ROTATION_TOLERANCE_RAD,
                        "valid": live_clearance_alignment_valid.detach().cpu().tolist(),
                        "local_acquisition_counts": None
                        if acquisition_evidence is None
                        else {
                            "call_count": acquisition_evidence["local_acquisition_call_count"],
                            "advance_call_count": acquisition_evidence["advance_call_count"],
                            "control_step_count": acquisition_evidence["control_step_count"],
                            "ik_solve_call_count": acquisition_evidence["ik_solve_call_count"],
                            "descent_waypoint_count": acquisition_evidence["local_descent_waypoint_count"],
                        },
                    },
                    "physical_close": {
                        "performed": acquire,
                        "acquisition": _diagnostic_pickup_acquisition_summary(
                            acquisition_valid,
                            acquisition_evidence,
                        ),
                    },
                    "placement": {
                        "valid": settled_valid.detach().cpu().tolist(),
                        "gate_masks": {
                            name: mask.detach().cpu().tolist() for name, mask in placement_gate_masks.items()
                        },
                        "drive_free_settle_sample_count": settle_sample_count,
                        "invariants": {
                            "valid": placement_invariants_valid.detach().cpu().tolist(),
                            "maximum_pairwise_distance_error_m": rigid_pairwise_distance_error.detach().cpu().tolist(),
                            "maximum_relative_orientation_error_rad": rigid_relative_orientation_error.detach()
                            .cpu()
                            .tolist(),
                            "cable_rest_length_error_m": cable_rest_length_error.detach().cpu().tolist(),
                            "anchor_position_error_m": anchor_position_error.detach().cpu().tolist(),
                            "anchor_orientation_error_rad": anchor_orientation_error.detach().cpu().tolist(),
                            "tail_write_position_error_m": tail_position_error.detach().cpu().tolist(),
                            "tail_write_orientation_error_rad": tail_orientation_error.detach().cpu().tolist(),
                            "socket_write_error_m": socket_write_error.detach().cpu().tolist(),
                            "written_maximum_spatial_speed": written_speed.detach().cpu().tolist(),
                        },
                        "pose_error": {
                            "driven_position_m": driven_position_error.detach().cpu().tolist(),
                            "driven_orientation_rad": driven_orientation_error.detach().cpu().tolist(),
                            "settled_position_m": settled_position_error.detach().cpu().tolist(),
                            "settled_orientation_rad": settled_orientation_error.detach().cpu().tolist(),
                        },
                        "drift": {
                            "settled_socket_position_m": socket_drift.detach().cpu().tolist(),
                        },
                        "speed": {
                            "settled_cable_linear_m_s": cable_linear_speed.detach().cpu().tolist(),
                            "settled_plug_linear_m_s": plug_linear_speed.detach().cpu().tolist(),
                            "settled_plug_angular_rad_s": plug_angular_speed.detach().cpu().tolist(),
                            "sampled_maximum_cable_linear_m_s": settle_maximum_cable_speed.detach().cpu().tolist(),
                            "sampled_maximum_plug_linear_m_s": settle_maximum_plug_linear_speed.detach().cpu().tolist(),
                            "sampled_maximum_plug_angular_rad_s": settle_maximum_plug_angular_speed.detach()
                            .cpu()
                            .tolist(),
                        },
                        "minimum_cable_support_clearance_m": minimum_cable_support_clearance.detach().cpu().tolist(),
                        "sampled_minimum_cable_support_clearance_m": settle_minimum_cable_clearance.detach()
                        .cpu()
                        .tolist(),
                        "loose_bodies_in_workspace": loose_bodies_in_workspace.detach().cpu().tolist(),
                        "cable_centers_in_slab_footprint": cable_centers_in_slab_footprint.detach().cpu().tolist(),
                        "minimum_cable_socket_center_distance_m": cable_socket_center_distance.detach().cpu().tolist(),
                        "cable_socket_separated": cable_socket_separated.detach().cpu().tolist(),
                        "collision": {
                            "all_settle_samples_valid": settle_all_collision_free.detach().cpu().tolist(),
                            "any_settle_contact_overflow": settle_any_contact_overflow,
                            "settle_invalid_pairs": settle_invalid_pairs,
                            "settled": _diagnostic_collision_summary(settled_collision),
                        },
                    },
                    "acquisition": _diagnostic_pickup_acquisition_summary(
                        acquisition_valid,
                        acquisition_evidence,
                    ),
                }
            )
        if not bool(valid.all()):
            acquisition_diagnostics: dict[str, Any] = {}
            if acquisition_evidence is not None:
                acquisition_diagnostics = {
                    "acquisition_valid": acquisition_valid.detach().cpu().tolist(),
                    "clearance_ik_valid": acquisition_evidence["clearance_ik_valid"].detach().cpu().tolist(),
                    "clearance_tcp_error_m": acquisition_evidence["preclose_error"].detach().cpu().tolist(),
                    "clearance_invalid_pairs": acquisition_evidence["preclose_collision"].invalid_contact_pairs,
                    "descent_ik_valid": acquisition_evidence["open_descent_ik_valid"].detach().cpu().tolist(),
                    "descent_collision_valid": acquisition_evidence["open_descent_collision_valid"]
                    .detach()
                    .cpu()
                    .tolist(),
                    "maximum_descent_tcp_error_m": acquisition_evidence["maximum_open_descent_tcp_error"]
                    .detach()
                    .cpu()
                    .tolist(),
                    "contact_tcp_error_m": acquisition_evidence["contact_position_error"].detach().cpu().tolist(),
                    "contact_invalid_pairs": acquisition_evidence["contact_preclose_collision"].invalid_contact_pairs,
                    "acquisition_final_tcp_distance_m": acquisition_evidence["final_distance"].detach().cpu().tolist(),
                    "acquisition_grasp_valid": acquisition_evidence["grasp"].valid.detach().cpu().tolist(),
                    "acquisition_collision_valid": acquisition_evidence["collision"].valid.detach().cpu().tolist(),
                    "acquisition_invalid_pairs": acquisition_evidence["collision"].invalid_contact_pairs,
                    "post_contact_settle": {
                        key: value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
                        for key, value in acquisition_evidence["post_contact_settle"].items()
                    },
                    "open_approach": {
                        key: value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
                        for key, value in acquisition_evidence["open_approach"].items()
                    },
                    "acquisition_ik_diagnostics": acquisition_evidence["ik_diagnostics"],
                }
            print(
                "[PICK-INSERT PICKUP REJECT] "
                + str(
                    {
                        "socket_pose": socket_pose.detach().cpu().tolist(),
                        "pickup_pose": pickup_pose.detach().cpu().tolist(),
                        "acquire": acquire,
                        "construction_sequence_version": _PICKUP_CONSTRUCTION_SEQUENCE_VERSION,
                        "construction_robot_staging_valid": staging_valid.detach().cpu().tolist(),
                        "task_only_robot_state_unchanged": task_only_robot_evidence["unchanged"]
                        .detach()
                        .cpu()
                        .tolist(),
                        "first_driven_sample_collision_valid": driven_first_collision_valid.detach().cpu().tolist(),
                        "first_driven_sample_zero_proxy_contacts": driven_first_zero_proxy_contacts.detach()
                        .cpu()
                        .tolist(),
                        "driven_position_error_m": driven_position_error.detach().cpu().tolist(),
                        "driven_orientation_error_rad": driven_orientation_error.detach().cpu().tolist(),
                        "settled_position_error_m": settled_position_error.detach().cpu().tolist(),
                        "settled_orientation_error_rad": settled_orientation_error.detach().cpu().tolist(),
                        "live_clearance_position_error_m": live_clearance_position_error.detach().cpu().tolist(),
                        "live_clearance_orientation_error_rad": live_clearance_orientation_error.detach()
                        .cpu()
                        .tolist(),
                        "live_clearance_alignment_valid": live_clearance_alignment_valid.detach().cpu().tolist(),
                        "settled_socket_drift_m": socket_drift.detach().cpu().tolist(),
                        "settled_cable_linear_speed_m_s": cable_linear_speed.detach().cpu().tolist(),
                        "settled_plug_linear_speed_m_s": plug_linear_speed.detach().cpu().tolist(),
                        "settled_plug_angular_speed_rad_s": plug_angular_speed.detach().cpu().tolist(),
                        "minimum_cable_support_clearance_m": minimum_cable_support_clearance.detach().cpu().tolist(),
                        "loose_bodies_in_workspace": loose_bodies_in_workspace.detach().cpu().tolist(),
                        "settled_collision_valid": settled_collision.valid.detach().cpu().tolist(),
                        "settled_invalid_pairs": settled_collision.invalid_contact_pairs,
                        **acquisition_diagnostics,
                    }
                ),
                flush=True,
            )
        return arm_target, finger_target, valid

    def _move_grasped_plug(
        self,
        plug_target: torch.Tensor,
        arm_seed: torch.Tensor,
        *,
        duration_s: float,
        intermediate_targets: tuple[torch.Tensor, ...] = (),
        endpoint_policy: str = _GRASPED_TRANSPORT_STRICT_ENDPOINT_POLICY,
        active_mask: torch.Tensor | None = None,
        finger_target: torch.Tensor | None = None,
        lane_hold: _PerLaneTargetHold | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Carry active grasped lanes while retaining prior targets for failed lanes."""
        if lane_hold is None:
            if active_mask is None:
                active_mask = torch.ones(self.env.num_envs, device=self.device, dtype=torch.bool)
            if finger_target is None:
                finger_target = self.closed_finger_target
            with _PerLaneTargetHold(self.env, active_mask, arm_seed, finger_target) as owned_hold:
                return self._move_grasped_plug(
                    plug_target,
                    arm_seed,
                    duration_s=duration_s,
                    intermediate_targets=intermediate_targets,
                    endpoint_policy=endpoint_policy,
                    lane_hold=owned_hold,
                )
        arm_target, valid = self._move_grasped_plug_per_lane(
            plug_target,
            arm_seed,
            duration_s=duration_s,
            intermediate_targets=intermediate_targets,
            endpoint_policy=endpoint_policy,
            lane_hold=lane_hold,
        )
        lane_hold.deactivate(~valid, reason="grasped-motion-final-validation")
        valid &= lane_hold.active_mask
        self.last_grasped_motion_evidence["lane_failure_masks"] = lane_hold.reason_masks
        self.last_grasped_motion_evidence["last_arm_target"] = lane_hold.last_sent_arm_target
        self.last_grasped_motion_evidence["last_finger_target"] = lane_hold.last_sent_finger_target
        return lane_hold.last_sent_arm_target, valid

    def _move_grasped_plug_per_lane(  # noqa: C901
        self,
        plug_target: torch.Tensor,
        arm_seed: torch.Tensor,
        *,
        duration_s: float,
        intermediate_targets: tuple[torch.Tensor, ...],
        endpoint_policy: str,
        lane_hold: _PerLaneTargetHold,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Carry a grasped plug along one drive-free, preload-preserving Cartesian route."""
        endpoint_position_tolerance_m, terminal_correction_enabled = _resolve_grasped_transport_endpoint_policy(
            self.cfg,
            endpoint_policy,
        )
        all_finite = torch.ones(self.env.num_envs, device=self.device, dtype=torch.bool)
        all_collision_free = all_finite.clone()
        all_bilateral = all_finite.clone()
        all_grasp_valid = all_finite.clone()
        all_drives_disabled = all_finite.clone()
        all_transient_cable_speeds_within_reset_limit = all_finite.clone()
        all_transport_motion_speeds_bounded = all_finite.clone()
        maximum_cable_speed = torch.zeros(self.env.num_envs, device=self.device)
        maximum_plug_linear_speed = torch.zeros_like(maximum_cable_speed)
        maximum_plug_angular_speed = torch.zeros_like(maximum_cable_speed)
        maximum_arm_speed = torch.zeros_like(maximum_cable_speed)
        maximum_finger_speed = torch.zeros_like(maximum_cable_speed)
        maximum_raw_ik_joint_step = torch.zeros_like(maximum_cable_speed)
        maximum_internal_target_velocity_jump = torch.zeros_like(maximum_cable_speed)
        minimum_left_contacts = torch.full(
            (self.env.num_envs,),
            torch.iinfo(torch.int64).max,
            device=self.device,
            dtype=torch.long,
        )
        minimum_right_contacts = minimum_left_contacts.clone()
        any_overflow = False
        invalid_pairs: list[str] = []
        samples = 0
        abort_reason: str | None = None
        failed_waypoint: dict[str, Any] | None = None
        original_advance = self.env.advance
        start_task_q, _ = self.env.read_task_state()
        start_plug = start_task_q[:, self.plug_index].clone()
        start_tcp = self.env.tcp_pose_e().clone()
        current_target = arm_seed.clone()

        def sample() -> None:
            nonlocal abort_reason, any_overflow, samples
            task_q, task_qd = self.env.read_task_state()
            arm_q, arm_qd, finger_q, finger_qd = self.env.read_robot_state()
            collision = collision_metrics(self.env, require_bilateral_grasp=False)
            if collision.contact_overflow:
                raise RuntimeError("Global contact-buffer overflow during grasped transport.")
            grasp = grasp_metrics(self.env, self.closed_finger_target, retaining_grasp=True)
            cable_speed = torch.linalg.vector_norm(task_qd[:, self.cable_slice, :3], dim=-1).amax(dim=-1)
            plug_linear_speed = torch.linalg.vector_norm(task_qd[:, self.plug_index, :3], dim=-1)
            plug_angular_speed = torch.linalg.vector_norm(task_qd[:, self.plug_index, 3:6], dim=-1)
            arm_speed = torch.abs(arm_qd).amax(dim=-1)
            finger_speed = torch.abs(finger_qd).amax(dim=-1)
            bilateral = _runtime_bilateral_grasp_proxy_contact_mask(
                self.env,
                collision.left_grasp_contact_count,
                collision.right_grasp_contact_count,
            )
            finite = (
                task_state_is_finite_and_normalized(task_q, task_qd)
                & torch.isfinite(arm_q).all(dim=-1)
                & torch.isfinite(arm_qd).all(dim=-1)
                & torch.isfinite(finger_q).all(dim=-1)
                & torch.isfinite(finger_qd).all(dim=-1)
            )
            drives_disabled = ~self._drive_enabled() & ~self._orientation_hold_enabled()
            if not bool(drives_disabled.all()):
                raise RuntimeError("A construction drive became enabled during grasped transport.")
            transient_cable_within_reset_limit, transport_motion_speeds_bounded = (
                _grasped_transport_transient_speed_masks(
                    cable_speed,
                    plug_linear_speed,
                    plug_angular_speed,
                    arm_speed,
                    finger_speed,
                    maximum_reset_cable_speed_m_s=self.cfg.maximum_row_cable_speed_m_s,
                    maximum_transport_plug_linear_speed_m_s=(self.cfg.maximum_grasped_transport_plug_linear_speed_m_s),
                    maximum_transport_plug_angular_speed_rad_s=(
                        self.cfg.maximum_grasped_transport_plug_angular_speed_rad_s
                    ),
                    maximum_transport_arm_joint_speed_rad_s=(self.cfg.maximum_grasped_transport_arm_joint_speed_rad_s),
                    maximum_transport_finger_joint_speed_m_s=self.cfg.maximum_row_finger_joint_speed_m_s,
                )
            )
            all_finite.logical_and_(finite)
            all_collision_free.logical_and_(collision.valid)
            all_bilateral.logical_and_(bilateral)
            all_grasp_valid.logical_and_(grasp.valid)
            all_drives_disabled.logical_and_(drives_disabled)
            all_transient_cable_speeds_within_reset_limit.logical_and_(transient_cable_within_reset_limit)
            all_transport_motion_speeds_bounded.logical_and_(transport_motion_speeds_bounded)
            maximum_cable_speed.copy_(torch.maximum(maximum_cable_speed, cable_speed))
            maximum_plug_linear_speed.copy_(torch.maximum(maximum_plug_linear_speed, plug_linear_speed))
            maximum_plug_angular_speed.copy_(torch.maximum(maximum_plug_angular_speed, plug_angular_speed))
            maximum_arm_speed.copy_(torch.maximum(maximum_arm_speed, arm_speed))
            maximum_finger_speed.copy_(torch.maximum(maximum_finger_speed, finger_speed))
            minimum_left_contacts.copy_(torch.minimum(minimum_left_contacts, collision.left_grasp_contact_count))
            minimum_right_contacts.copy_(torch.minimum(minimum_right_contacts, collision.right_grasp_contact_count))
            any_overflow |= collision.contact_overflow
            samples += 1
            for pair in collision.invalid_contact_pairs:
                if pair not in invalid_pairs and len(invalid_pairs) < 64:
                    invalid_pairs.append(pair)
            violated = (
                ~finite
                | ~collision.valid
                | ~bilateral
                | ~grasp.valid
                | ~drives_disabled
                | ~transport_motion_speeds_bounded
            )
            if bool(violated.any()):
                reasons = [
                    name
                    for name, failed in (
                        ("non-finite", ~finite),
                        ("collision", ~collision.valid),
                        ("lost-bilateral-contact", ~bilateral),
                        ("grasp-geometry", ~grasp.valid),
                        ("construction-drive", ~drives_disabled),
                        ("transport-motion-speed", ~transport_motion_speeds_bounded),
                    )
                    if bool(failed.any())
                ]
                abort_reason = "+".join(reasons)
                lane_hold.deactivate(~finite, reason="grasped-motion-non-finite")
                lane_hold.deactivate(~collision.valid, reason="grasped-motion-collision")
                lane_hold.deactivate(~bilateral, reason="grasped-motion-lost-bilateral-contact")
                lane_hold.deactivate(~grasp.valid, reason="grasped-motion-grasp-geometry")
                lane_hold.deactivate(
                    ~transport_motion_speeds_bounded,
                    reason="grasped-motion-transport-speed",
                )

        def guarded_advance(duration: float, update=None, *, post_step=None):
            def combined_post_step(step: int, steps: int, progress: float) -> None:
                if post_step is not None:
                    post_step(step, steps, progress)
                sample()

            return original_advance(duration, update, post_step=combined_post_step)

        path_targets = (*intermediate_targets, plug_target)
        segment_starts = (start_plug, *path_targets[:-1])
        segment_lane_waypoint_counts: list[torch.Tensor] = []
        lane_waypoint_count = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.long)
        for segment_start, segment_target in zip(segment_starts, path_targets, strict=True):
            plug_translation = torch.linalg.vector_norm(
                segment_target[:, :3] - segment_start[:, :3],
                dim=-1,
            )
            plug_rotation = math_utils.quat_error_magnitude(
                segment_target[:, 3:7],
                segment_start[:, 3:7],
            )
            translation_steps = torch.ceil(plug_translation / self.cfg.grasped_transport_maximum_translation_step_m).to(
                torch.long
            )
            rotation_steps = torch.ceil(plug_rotation / self.cfg.grasped_transport_maximum_rotation_step_rad).to(
                torch.long
            )
            lane_steps = torch.maximum(translation_steps, rotation_steps).clamp_min(1)
            segment_lane_waypoint_counts.append(lane_steps)
            lane_waypoint_count += lane_steps
        lane_route_within_cap = lane_waypoint_count <= self.cfg.grasped_transport_maximum_waypoints
        lane_hold.deactivate(~lane_route_within_cap, reason="grasped-motion-waypoint-cap")

        segment_waypoint_counts: list[int] = []
        for lane_steps in segment_lane_waypoint_counts:
            active_steps = torch.where(lane_hold.active_mask, lane_steps, torch.zeros_like(lane_steps))
            segment_waypoint_counts.append(max(1, int(active_steps.amax().item())))
        waypoint_count = sum(segment_waypoint_counts)
        route_within_cap = bool(lane_route_within_cap.all())
        waypoint_motion_s = max(self.cfg.grasped_transport_waypoint_motion_s, duration_s / waypoint_count)
        start_solution = self._solve_ik(
            start_tcp[:, :3],
            start_tcp[:, 3:7],
            self.closed_finger_target,
            arm_seed=arm_seed,
        )
        current_raw = torch.where(start_solution.valid[:, None], start_solution.arm_q, arm_seed)
        arm_target_bias = arm_seed - current_raw
        start_preload_joint_limits_valid = joint_limit_mask(self.env, arm_seed, margin=0.02)
        start_valid = start_solution.valid & start_preload_joint_limits_valid
        valid = start_valid & lane_route_within_cap
        lane_hold.deactivate(~start_valid, reason="grasped-motion-route-preflight")
        current_target = lane_hold.last_sent_arm_target
        final_tcp_target = start_tcp.clone()
        terminal_correction_requested = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool)
        terminal_correction_ik_valid = torch.ones_like(terminal_correction_requested)
        terminal_correction_progress_valid = torch.ones_like(terminal_correction_requested)
        terminal_correction_plug_priority = torch.zeros_like(terminal_correction_requested)
        terminal_correction_executed = torch.zeros_like(terminal_correction_requested)
        terminal_correction_requested_count = 0
        terminal_correction_executed_count = 0
        terminal_correction_command_offset = torch.zeros((self.env.num_envs, 3), device=self.device)
        terminal_correction_step_history: list[dict[str, Any]] = []
        pre_correction_tcp_position_error = torch.zeros(self.env.num_envs, device=self.device)
        pre_correction_plug_position_error = torch.zeros(self.env.num_envs, device=self.device)
        precomputed_segment_knots: list[torch.Tensor] = []
        precomputed_segment_waypoint_counts: list[int] = []
        precomputed_ik_knot_count = 0
        for segment_index, (segment_start, segment_target, segment_waypoints) in enumerate(
            zip(segment_starts, path_targets, segment_waypoint_counts, strict=True)
        ):
            segment_knots = [current_target.clone()]
            for waypoint in range(1, segment_waypoints + 1):
                if not bool(lane_hold.active_mask.any()):
                    break
                progress = waypoint / segment_waypoints
                waypoint_plug_position = torch.lerp(segment_start[:, :3], segment_target[:, :3], progress)
                waypoint_plug_orientation = batched_quat_slerp(
                    segment_start[:, 3:7],
                    segment_target[:, 3:7],
                    progress,
                )
                delta_orientation = math_utils.quat_mul(
                    waypoint_plug_orientation,
                    math_utils.quat_conjugate(start_plug[:, 3:7]),
                )
                waypoint_tcp_position = waypoint_plug_position + math_utils.quat_apply(
                    delta_orientation,
                    start_tcp[:, :3] - start_plug[:, :3],
                )
                waypoint_tcp_orientation = math_utils.quat_mul(delta_orientation, start_tcp[:, 3:7])
                solution = self._solve_ik(
                    waypoint_tcp_position,
                    waypoint_tcp_orientation,
                    self.closed_finger_target,
                    arm_seed=current_raw,
                )
                raw_joint_step = torch.abs(solution.arm_q - current_raw).amax(dim=-1)
                maximum_raw_ik_joint_step.copy_(torch.maximum(maximum_raw_ik_joint_step, raw_joint_step))
                proposed_target = solution.arm_q + arm_target_bias
                proposed_target_joint_limits_valid = joint_limit_mask(self.env, proposed_target, margin=0.02)
                waypoint_valid = (
                    solution.valid
                    & (raw_joint_step <= self.cfg.grasped_transport_maximum_raw_ik_joint_step_rad)
                    & proposed_target_joint_limits_valid
                )
                active_before_waypoint = lane_hold.active_mask
                valid &= ~active_before_waypoint | waypoint_valid
                newly_invalid = active_before_waypoint & ~waypoint_valid
                if bool(newly_invalid.any()):
                    abort_reason = f"segment-{segment_index}-waypoint-ik-continuity"
                    if failed_waypoint is None:
                        failed_waypoint = {
                            "segment_index": segment_index,
                            "waypoint_index": waypoint,
                            "segment_waypoint_count": segment_waypoints,
                            "failed_lanes": newly_invalid.detach().cpu().tolist(),
                            "solver_valid": solution.valid.detach().cpu().tolist(),
                            "raw_joint_step_rad": raw_joint_step.detach().cpu().tolist(),
                            "maximum_allowed_raw_joint_step_rad": (
                                self.cfg.grasped_transport_maximum_raw_ik_joint_step_rad
                            ),
                            "proposed_target_joint_limits_valid": (
                                proposed_target_joint_limits_valid.detach().cpu().tolist()
                            ),
                        }
                lane_hold.deactivate(newly_invalid, reason="grasped-motion-waypoint-ik-continuity")
                command_mask = active_before_waypoint & waypoint_valid
                current_raw = torch.where(command_mask[:, None], solution.arm_q, current_raw)
                current_target = _retain_active_grasped_transport_target(
                    proposed_target,
                    lane_hold.last_sent_arm_target,
                    command_mask,
                )
                segment_knots.append(current_target.clone())
                precomputed_ik_knot_count += 1
                waypoint_tcp_target = torch.cat((waypoint_tcp_position, waypoint_tcp_orientation), dim=-1)
                final_tcp_target = torch.where(
                    command_mask[:, None],
                    waypoint_tcp_target,
                    final_tcp_target,
                )
            if len(segment_knots) > 1:
                stacked_knots = torch.stack(segment_knots)
                segment_velocity_jump = _grasped_transport_internal_target_velocity_jump(
                    stacked_knots,
                    duration_per_knot_s=waypoint_motion_s,
                )
                maximum_internal_target_velocity_jump.copy_(
                    torch.where(
                        lane_hold.active_mask,
                        torch.maximum(maximum_internal_target_velocity_jump, segment_velocity_jump),
                        maximum_internal_target_velocity_jump,
                    )
                )
                precomputed_segment_knots.append(stacked_knots)
                precomputed_segment_waypoint_counts.append(len(segment_knots) - 1)
            if not bool(lane_hold.active_mask.any()):
                break

        maximum_allowed_internal_target_velocity_jump = (
            2.0
            * self.cfg.grasped_transport_maximum_raw_ik_joint_step_rad
            / ((1.0 - _GRASPED_TRANSPORT_C2_RAMP_FRACTION) * waypoint_motion_s)
        )
        internal_target_velocity_jump_valid = (
            maximum_internal_target_velocity_jump <= maximum_allowed_internal_target_velocity_jump + 1.0e-6
        )
        valid &= ~lane_hold.active_mask | internal_target_velocity_jump_valid
        lane_hold.deactivate(
            ~internal_target_velocity_jump_valid,
            reason="grasped-motion-internal-target-velocity-jump",
        )

        route_motion_advance_count = 0
        route_motion_control_step_count = 0
        segment_end_settle_advance_count = 0
        segment_end_settle_control_step_count = 0
        segment_motion_durations_s: list[float] = []
        self.env.advance = guarded_advance
        try:
            for segment_knots, segment_waypoints in zip(
                precomputed_segment_knots,
                precomputed_segment_waypoint_counts,
                strict=True,
            ):
                if not bool(lane_hold.active_mask.any()):
                    break
                segment_motion_s = segment_waypoints * waypoint_motion_s
                segment_motion_durations_s.append(segment_motion_s)

                def update_segment(_step: int, _steps: int, progress: float) -> None:
                    path_progress = _grasped_transport_c2_progress(progress)
                    scheduled_target = _interpolate_grasped_transport_knots(segment_knots, path_progress)
                    self.env.set_robot_targets(scheduled_target, self.closed_finger_target)

                route_motion_control_step_count += self.env.advance(segment_motion_s, update_segment)
                route_motion_advance_count += 1
                self.env.set_robot_targets(segment_knots[-1], self.closed_finger_target)
                segment_end_settle_control_step_count += self.env.advance(self.cfg.grasped_transport_waypoint_settle_s)
                segment_end_settle_advance_count += 1
                current_target = lane_hold.last_sent_arm_target
            if bool(lane_hold.active_mask.any()):
                self.env.set_robot_targets(current_target, self.closed_finger_target)
                self.env.advance(self.cfg.grasped_transport_final_settle_s)
                current_tcp = self.env.tcp_pose_e()
                pre_correction_tcp_position_error = torch.linalg.vector_norm(
                    current_tcp[:, :3] - final_tcp_target[:, :3],
                    dim=-1,
                )
                current_task_q, _ = self.env.read_task_state()
                current_plug_position = current_task_q[:, self.plug_index, :3]
                pre_correction_plug_position_error = torch.linalg.vector_norm(
                    current_plug_position - plug_target[:, :3],
                    dim=-1,
                )
                current_tcp_position_error = pre_correction_tcp_position_error.clone()
                current_plug_position_error = pre_correction_plug_position_error.clone()
                terminal_command_position = final_tcp_target[:, :3].clone()
                while (
                    terminal_correction_enabled
                    and terminal_correction_requested_count < self.cfg.grasped_transport_final_correction_max_iterations
                ):
                    correction_step, correction_needed, plug_priority = _grasped_transport_terminal_translation_step(
                        current_tcp[:, :3],
                        final_tcp_target[:, :3],
                        current_plug_position,
                        plug_target[:, :3],
                        position_tolerance_m=endpoint_position_tolerance_m,
                        maximum_step_m=self.cfg.grasped_transport_final_correction_step_m,
                    )
                    correction_needed &= lane_hold.active_mask
                    correction_step = torch.where(
                        correction_needed[:, None],
                        correction_step,
                        torch.zeros_like(correction_step),
                    )
                    terminal_correction_requested |= correction_needed
                    terminal_correction_plug_priority |= correction_needed & plug_priority
                    if not bool(correction_needed.any()):
                        break
                    terminal_correction_requested_count += 1
                    proposed_terminal_command_position = terminal_command_position + correction_step
                    correction_solution = self._solve_ik(
                        proposed_terminal_command_position,
                        final_tcp_target[:, 3:7],
                        self.closed_finger_target,
                        arm_seed=current_raw,
                    )
                    correction_raw_joint_step = torch.abs(correction_solution.arm_q - current_raw).amax(dim=-1)
                    correction_target = correction_solution.arm_q + arm_target_bias
                    correction_valid = (
                        correction_solution.valid
                        & (correction_raw_joint_step <= self.cfg.grasped_transport_maximum_raw_ik_joint_step_rad)
                        & joint_limit_mask(self.env, correction_target, margin=0.02)
                    )
                    terminal_correction_ik_valid &= ~correction_needed | correction_valid
                    valid &= ~correction_needed | correction_valid
                    correction_command_mask = correction_needed & correction_valid
                    correction_failed = correction_needed & ~correction_valid
                    if bool(correction_failed.any()):
                        abort_reason = "terminal-correction-ik-continuity"
                    lane_hold.deactivate(
                        correction_failed,
                        reason="grasped-motion-terminal-correction-ik-continuity",
                    )
                    executed_correction_step = torch.where(
                        correction_command_mask[:, None],
                        correction_step,
                        torch.zeros_like(correction_step),
                    )
                    terminal_command_position = _retain_active_grasped_transport_target(
                        proposed_terminal_command_position,
                        terminal_command_position,
                        correction_command_mask,
                    )
                    safe_correction_target = torch.where(
                        correction_command_mask[:, None],
                        correction_target,
                        current_target,
                    )
                    interpolate_arm_motion(
                        self.env,
                        current_target,
                        safe_correction_target,
                        self.closed_finger_target,
                        self.cfg.grasped_transport_waypoint_motion_s,
                    )
                    current_raw = torch.where(
                        correction_command_mask[:, None],
                        correction_solution.arm_q,
                        current_raw,
                    )
                    current_target = safe_correction_target
                    self.env.set_robot_targets(current_target, self.closed_finger_target)
                    self.env.advance(self.cfg.grasped_transport_final_settle_s)
                    current_target = lane_hold.last_sent_arm_target
                    terminal_correction_executed |= correction_command_mask
                    if bool(correction_command_mask.any()):
                        terminal_correction_executed_count += 1
                    current_tcp = self.env.tcp_pose_e()
                    next_tcp_position_error = torch.linalg.vector_norm(
                        current_tcp[:, :3] - final_tcp_target[:, :3],
                        dim=-1,
                    )
                    current_task_q, _ = self.env.read_task_state()
                    next_plug_position = current_task_q[:, self.plug_index, :3]
                    next_plug_position_error = torch.linalg.vector_norm(
                        next_plug_position - plug_target[:, :3],
                        dim=-1,
                    )
                    correction_progress_valid = _grasped_transport_terminal_progress_mask(
                        current_tcp_position_error,
                        next_tcp_position_error,
                        current_plug_position_error,
                        next_plug_position_error,
                        correction_mask=correction_command_mask,
                        plug_priority=plug_priority,
                        position_tolerance_m=endpoint_position_tolerance_m,
                        progress_epsilon_m=_GRASPED_TRANSPORT_TERMINAL_PROGRESS_EPSILON_M,
                    )
                    terminal_correction_progress_valid &= correction_progress_valid
                    terminal_correction_command_offset += executed_correction_step
                    terminal_correction_step_history.append(
                        {
                            "iteration": terminal_correction_requested_count,
                            "selected_metric": [
                                ("plug-position" if selected else "tcp-position") if executed else None
                                for selected, executed in zip(
                                    plug_priority.detach().cpu().tolist(),
                                    correction_command_mask.detach().cpu().tolist(),
                                    strict=True,
                                )
                            ],
                            "step_m": executed_correction_step.detach().cpu().tolist(),
                            "step_norm_m": torch.linalg.vector_norm(executed_correction_step, dim=-1)
                            .detach()
                            .cpu()
                            .tolist(),
                            "raw_joint_step_rad": correction_raw_joint_step.detach().cpu().tolist(),
                            "ik_valid": correction_valid.detach().cpu().tolist(),
                            "progress_valid": correction_progress_valid.detach().cpu().tolist(),
                            "tcp_error_before_m": current_tcp_position_error.detach().cpu().tolist(),
                            "tcp_error_after_m": next_tcp_position_error.detach().cpu().tolist(),
                            "plug_error_before_m": current_plug_position_error.detach().cpu().tolist(),
                            "plug_error_after_m": next_plug_position_error.detach().cpu().tolist(),
                        }
                    )
                    valid &= correction_progress_valid
                    no_progress = correction_command_mask & ~correction_progress_valid
                    if bool(no_progress.any()):
                        abort_reason = "terminal-correction-selected-metric-progress"
                    lane_hold.deactivate(
                        no_progress,
                        reason="grasped-motion-terminal-correction-selected-metric-progress",
                    )
                    current_tcp_position_error = next_tcp_position_error
                    current_plug_position_error = next_plug_position_error
                    current_plug_position = next_plug_position
        finally:
            self.env.advance = original_advance
        grasp = grasp_metrics(self.env, self.closed_finger_target, retaining_grasp=True)
        collision = collision_metrics(self.env)
        if collision.contact_overflow:
            raise RuntimeError("Global contact-buffer overflow at the grasped-transport result boundary.")
        final_task_q, final_task_qd = self.env.read_task_state()
        _, final_arm_qd, _, final_finger_qd = self.env.read_robot_state()
        final_cable_speed = torch.linalg.vector_norm(
            final_task_qd[:, self.cable_slice, :3],
            dim=-1,
        ).amax(dim=-1)
        final_plug_linear_speed = torch.linalg.vector_norm(final_task_qd[:, self.plug_index, :3], dim=-1)
        final_plug_angular_speed = torch.linalg.vector_norm(final_task_qd[:, self.plug_index, 3:6], dim=-1)
        final_arm_speed = torch.abs(final_arm_qd).amax(dim=-1)
        final_finger_speed = torch.abs(final_finger_qd).amax(dim=-1)
        final_plug_position_error = torch.linalg.vector_norm(
            final_task_q[:, self.plug_index, :3] - plug_target[:, :3],
            dim=-1,
        )
        final_plug_orientation_error = math_utils.quat_error_magnitude(
            final_task_q[:, self.plug_index, 3:7],
            plug_target[:, 3:7],
        )
        final_tcp = self.env.tcp_pose_e()
        final_tcp_position_error = torch.linalg.vector_norm(final_tcp[:, :3] - final_tcp_target[:, :3], dim=-1)
        final_tcp_orientation_error = math_utils.quat_error_magnitude(
            final_tcp[:, 3:7],
            final_tcp_target[:, 3:7],
        )
        final_endpoint_position_valid = _grasped_transport_endpoint_position_mask(
            final_tcp_position_error,
            final_plug_position_error,
            position_tolerance_m=endpoint_position_tolerance_m,
        )
        if samples == 0:
            minimum_left_contacts.zero_()
            minimum_right_contacts.zero_()
        route_control_budget = _grasped_transport_route_control_budget(
            segment_waypoint_counts,
            duration_per_knot_s=waypoint_motion_s,
            segment_end_settle_s=self.cfg.grasped_transport_waypoint_settle_s,
            advance_dt=float(self.env.advance_dt),
        )
        self.last_grasped_motion_evidence = {
            "transport_schedule_version": _GRASPED_TRANSPORT_SCHEDULE_VERSION,
            "transport_schedule": "c2-endpoint-time-law-piecewise-linear-joint-cruise",
            "transport_schedule_c2_ramp_fraction": _GRASPED_TRANSPORT_C2_RAMP_FRACTION,
            "transport_time_law_endpoint_continuity": "C2",
            "transport_joint_path_internal_knot_continuity": "C0-with-bounded-target-velocity-jumps",
            "transport_schedule_scope": (
                "all-shared-scripted-grasped-carry-including-phase-realization-and-canonical-reseat"
            ),
            "transient_cable_speed_policy": "sample-every-step-observation-only-during-scripted-carry",
            "transient_cable_speed_is_rejection_gate": False,
            "stored_final_reset_replay_cable_speed_limit_m_s": self.cfg.maximum_row_cable_speed_m_s,
            "final_cable_speed_is_rejection_gate": True,
            "maximum_internal_target_velocity_jump_rad_s": maximum_internal_target_velocity_jump,
            "maximum_allowed_internal_target_velocity_jump_rad_s": (maximum_allowed_internal_target_velocity_jump),
            "internal_target_velocity_jump_valid": internal_target_velocity_jump_valid,
            "precomputed_ik_knot_count": precomputed_ik_knot_count,
            "precomputed_segment_waypoint_counts": precomputed_segment_waypoint_counts,
            "segment_motion_durations_s": segment_motion_durations_s,
            "route_motion_advance_count": route_motion_advance_count,
            "route_motion_control_step_count": route_motion_control_step_count,
            "internal_knot_settle_advance_count": 0,
            "internal_knot_settle_control_step_count": 0,
            "segment_end_settle_advance_count": segment_end_settle_advance_count,
            "segment_end_settle_control_step_count": segment_end_settle_control_step_count,
            "legacy_route_motion_advance_count": waypoint_count,
            "legacy_route_motion_control_step_count": route_control_budget["legacy_route_motion_control_step_count"],
            "legacy_internal_knot_settle_advance_count": waypoint_count,
            "legacy_internal_knot_settle_control_step_count": route_control_budget[
                "legacy_internal_knot_settle_control_step_count"
            ],
            "legacy_route_control_step_count": route_control_budget["legacy_route_control_step_count"],
            "scheduled_full_route_motion_control_step_count": route_control_budget[
                "scheduled_route_motion_control_step_count"
            ],
            "scheduled_full_route_segment_end_settle_control_step_count": route_control_budget[
                "scheduled_segment_end_settle_control_step_count"
            ],
            "scheduled_full_route_control_step_count": route_control_budget["scheduled_route_control_step_count"],
            "scheduled_executed_route_control_step_count": (
                route_motion_control_step_count + segment_end_settle_control_step_count
            ),
            "scheduled_full_route_control_step_reduction": route_control_budget[
                "scheduled_route_control_step_reduction"
            ],
            "scheduled_full_route_control_step_reduction_fraction": route_control_budget[
                "scheduled_route_control_step_reduction_fraction"
            ],
            "samples": samples,
            "segment_count": len(path_targets),
            "segment_waypoint_counts": segment_waypoint_counts,
            "lane_waypoint_counts": lane_waypoint_count,
            "waypoint_count": waypoint_count,
            "waypoint_cap": self.cfg.grasped_transport_maximum_waypoints,
            "route_within_waypoint_cap": route_within_cap,
            "lane_route_within_waypoint_cap": lane_route_within_cap,
            "start_ik_valid": start_solution.valid,
            "start_preload_joint_limits_valid": start_preload_joint_limits_valid,
            "start_valid": start_valid,
            "abort_reason": abort_reason,
            "failed_waypoint": failed_waypoint,
            "all_samples_finite": all_finite,
            "all_samples_collision_free": all_collision_free,
            "all_samples_bilateral_proxy_contact": all_bilateral,
            "all_samples_grasp_valid": all_grasp_valid,
            "all_samples_drives_disabled": all_drives_disabled,
            "all_samples_transient_cable_speeds_within_reset_limit": (all_transient_cable_speeds_within_reset_limit),
            "all_samples_transport_motion_speeds_bounded": all_transport_motion_speeds_bounded,
            "any_contact_overflow": any_overflow,
            "invalid_contact_pairs": invalid_pairs,
            "minimum_left_proxy_contact_count": minimum_left_contacts,
            "minimum_right_proxy_contact_count": minimum_right_contacts,
            "maximum_cable_speed_m_s": maximum_cable_speed,
            "maximum_plug_linear_speed_m_s": maximum_plug_linear_speed,
            "maximum_plug_angular_speed_rad_s": maximum_plug_angular_speed,
            "maximum_arm_joint_speed_rad_s": maximum_arm_speed,
            "maximum_finger_joint_speed_m_s": maximum_finger_speed,
            "maximum_raw_ik_joint_step_rad": maximum_raw_ik_joint_step,
            "final_cable_speed_m_s": final_cable_speed,
            "final_cable_speed_within_reset_limit": final_cable_speed <= self.cfg.maximum_row_cable_speed_m_s,
            "final_plug_linear_speed_m_s": final_plug_linear_speed,
            "final_plug_angular_speed_rad_s": final_plug_angular_speed,
            "final_arm_joint_speed_rad_s": final_arm_speed,
            "final_finger_joint_speed_m_s": final_finger_speed,
            "final_plug_position_error_m": final_plug_position_error,
            "final_plug_orientation_error_rad": final_plug_orientation_error,
            "final_tcp_position_error_m": final_tcp_position_error,
            "final_tcp_orientation_error_rad": final_tcp_orientation_error,
            "pre_correction_tcp_position_error_m": pre_correction_tcp_position_error,
            "pre_correction_plug_position_error_m": pre_correction_plug_position_error,
            "endpoint_policy": endpoint_policy,
            "endpoint_position_tolerance_m": endpoint_position_tolerance_m,
            "final_endpoint_position_valid": final_endpoint_position_valid,
            "terminal_correction_enabled": terminal_correction_enabled,
            "terminal_correction_controller": (
                "plug-priority-translation-only" if terminal_correction_enabled else "disabled"
            ),
            "terminal_correction_progress_epsilon_m": _GRASPED_TRANSPORT_TERMINAL_PROGRESS_EPSILON_M,
            "terminal_correction_requested": terminal_correction_requested,
            "terminal_correction_ik_valid": terminal_correction_ik_valid,
            "terminal_correction_progress_valid": terminal_correction_progress_valid,
            "terminal_correction_plug_priority": terminal_correction_plug_priority,
            "terminal_correction_executed": terminal_correction_executed,
            "terminal_correction_requested_count": terminal_correction_requested_count,
            "terminal_correction_executed_count": terminal_correction_executed_count,
            "terminal_correction_max_iterations": self.cfg.grasped_transport_final_correction_max_iterations,
            "terminal_correction_command_offset_m": terminal_correction_command_offset,
            "terminal_correction_command_offset_norm_m": torch.linalg.vector_norm(
                terminal_correction_command_offset,
                dim=-1,
            ),
            "terminal_correction_step_history": terminal_correction_step_history,
        }
        motion_valid = (
            all_finite
            & all_collision_free
            & all_bilateral
            & all_grasp_valid
            & all_drives_disabled
            & all_transport_motion_speeds_bounded
            & (not any_overflow)
            & (final_cable_speed <= self.cfg.maximum_row_cable_speed_m_s)
            & (final_plug_linear_speed <= self.cfg.maximum_pickup_plug_linear_speed_m_s)
            & (final_plug_angular_speed <= self.cfg.maximum_pickup_plug_angular_speed_rad_s)
            & (final_arm_speed <= self.cfg.maximum_row_arm_joint_speed_rad_s)
            & (final_finger_speed <= self.cfg.maximum_row_finger_joint_speed_m_s)
            & final_endpoint_position_valid
            & (final_plug_orientation_error <= PICK_INSERT_GOAL_MAX_AUTHORED_PLUG_ANGLE_RAD)
            & (final_tcp_orientation_error <= PICK_INSERT_GOAL_MAX_AUTHORED_PLUG_ANGLE_RAD)
        )
        valid &= lane_hold.active_mask & grasp.valid & collision.valid & motion_valid
        lane_hold.deactivate(~grasp.valid, reason="grasped-motion-final-grasp")
        lane_hold.deactivate(~collision.valid, reason="grasped-motion-final-collision")
        lane_hold.deactivate(~motion_valid, reason="grasped-motion-final-state")
        valid &= lane_hold.active_mask
        current_target = lane_hold.last_sent_arm_target
        self.last_grasped_motion_evidence["abort_reason"] = "+".join(lane_hold.reason_masks) or None
        if not bool(valid.all()):
            print(
                "[PICK-INSERT GRASPED MOTION REJECT] "
                + str(
                    {
                        "plug_target": plug_target.detach().cpu().tolist(),
                        "motion_valid": motion_valid.detach().cpu().tolist(),
                        "grasp_valid": grasp.valid.detach().cpu().tolist(),
                        "collision_valid": collision.valid.detach().cpu().tolist(),
                        "evidence": {
                            key: value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
                            for key, value in self.last_grasped_motion_evidence.items()
                        },
                    }
                ),
                flush=True,
            )
        return current_target, valid

    def _realize_phase(
        self,
        phase: int,
        pickup_pose: torch.Tensor,
        goal_q: torch.Tensor,
        pickup_arm_target: torch.Tensor,
        *,
        pickup_finger_target: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Realize one phase while retaining failed pickup-lane targets."""
        if pickup_finger_target is None:
            pickup_finger_target = self.closed_finger_target if _PHASE_STARTS_GRASPED[phase] else self.open_finger_q
        if active_mask is None:
            active_mask = torch.ones(self.env.num_envs, device=self.device, dtype=torch.bool)
        with _PerLaneTargetHold(
            self.env,
            active_mask,
            pickup_arm_target,
            pickup_finger_target,
        ) as lane_hold:
            _arm_target, _finger_target, valid = self._realize_phase_per_lane(
                phase,
                pickup_pose,
                goal_q,
                pickup_arm_target,
                lane_hold=lane_hold,
            )
            lane_hold.deactivate(~valid, reason="phase-realization-final-validation")
            return lane_hold.last_sent_arm_target, lane_hold.last_sent_finger_target, valid & lane_hold.active_mask

    def _realize_phase_per_lane(
        self,
        phase: int,
        pickup_pose: torch.Tensor,
        goal_q: torch.Tensor,
        pickup_arm_target: torch.Tensor,
        *,
        lane_hold: _PerLaneTargetHold,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        arm_target = pickup_arm_target
        valid = lane_hold.active_mask
        goal_plug = goal_q[:, self.plug_index]
        if phase in (0, 1):
            axial = self.cfg.phase_0_axial_offset_m if phase == 0 else self.cfg.phase_1_axial_offset_m
            local_offset = torch.zeros((self.env.num_envs, 3), device=self.device)
            local_offset[:, 1] = -axial
            plug_target = goal_plug.clone()
            plug_target[:, :3] += math_utils.quat_apply(goal_plug[:, 3:7], local_offset)
            live_q, _ = self.env.read_task_state()
            live_plug = live_q[:, self.plug_index].clone()
            preinsert_offset = torch.zeros_like(local_offset)
            preinsert_offset[:, 1] = -0.08
            preinsert_target = goal_plug.clone()
            preinsert_target[:, :3] += math_utils.quat_apply(goal_plug[:, 3:7], preinsert_offset)
            transport_height = torch.maximum(live_plug[:, 2], preinsert_target[:, 2]) + self.cfg.phase_2_lift_m
            lift_target = live_plug.clone()
            lift_target[:, 2] = transport_height
            midpoint_target = preinsert_target.clone()
            midpoint_target[:, :2] = 0.5 * (live_plug[:, :2] + preinsert_target[:, :2])
            midpoint_target[:, 2] = transport_height
            overhead_target = preinsert_target.clone()
            overhead_target[:, 2] = transport_height
            arm_target, valid = self._move_grasped_plug(
                plug_target,
                arm_target,
                duration_s=self.cfg.pickup_transport_s,
                intermediate_targets=(lift_target, midpoint_target, overhead_target, preinsert_target),
                endpoint_policy=_GRASPED_TRANSPORT_RESET_ROW_ENDPOINT_POLICY,
                lane_hold=lane_hold,
            )
        elif phase == 2:
            preinsert = goal_plug.clone()
            offset = torch.zeros((self.env.num_envs, 3), device=self.device)
            offset[:, 1] = -0.08
            preinsert[:, :3] += math_utils.quat_apply(goal_plug[:, 3:7], offset)
            live_q, _ = self.env.read_task_state()
            live_plug = live_q[:, self.plug_index].clone()
            transport_height = torch.maximum(live_plug[:, 2], preinsert[:, 2]) + self.cfg.phase_2_lift_m
            lift_target = live_plug.clone()
            lift_target[:, 2] = transport_height
            plug_target = preinsert.clone()
            plug_target[:, :2] = 0.5 * (live_plug[:, :2] + preinsert[:, :2])
            plug_target[:, 2] = transport_height
            arm_target, valid = self._move_grasped_plug(
                plug_target,
                arm_target,
                duration_s=self.cfg.pickup_transport_s,
                intermediate_targets=(lift_target,),
                endpoint_policy=_GRASPED_TRANSPORT_RESET_ROW_ENDPOINT_POLICY,
                lane_hold=lane_hold,
            )
        elif phase == 3:
            plug_target = pickup_pose.clone()
            plug_target[:, 2] += self.cfg.phase_3_lift_m
            arm_target, valid = self._move_grasped_plug(
                plug_target,
                arm_target,
                duration_s=1.2,
                endpoint_policy=_GRASPED_TRANSPORT_RESET_ROW_ENDPOINT_POLICY,
                lane_hold=lane_hold,
            )
        elif phase == 4:
            # Construction already returned this ungrasped row at the proven
            # live-plug clearance.  Preserve that exact open target instead of
            # paying for a second broad pregrasp motion.
            arm_target = lane_hold.last_sent_arm_target
            self.env.set_robot_targets(arm_target, self.open_finger_q)
        elif phase == 5:
            self._open_gripper(arm_target)
            noise = (2.0 * torch.rand(self.home_arm_q.shape, device=self.device, generator=self.random) - 1.0) * float(
                self.env.cfg.arm_reset_joint_noise
            )
            away_target = self.home_arm_q + noise
            away_valid = joint_limit_mask(self.env, away_target)
            valid &= away_valid
            lane_hold.deactivate(~away_valid, reason="phase-realization-away-joint-limit")
            current_arm_q, _, _, _ = self.env.read_robot_state()
            safe_away_target = torch.where(lane_hold.active_mask[:, None], away_target, lane_hold.last_sent_arm_target)
            interpolate_arm_motion(self.env, current_arm_q, safe_away_target, self.open_finger_q, 2.0)
            self.env.set_robot_targets(safe_away_target, self.open_finger_q)
            self.env.advance(self.cfg.tcp_settle_s)
            arm_target = lane_hold.last_sent_arm_target
        else:
            raise ValueError(f"Unknown pick-insert reset phase {phase}.")
        self.env.advance(self.cfg.pickup_settle_s)
        self._assert_drive_disabled("phase realization")
        return lane_hold.last_sent_arm_target, lane_hold.last_sent_finger_target, valid & lane_hold.active_mask

    def _cold_replay(
        self,
        warm_state: dict[str, torch.Tensor],
        *,
        starts_grasped: bool,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], torch.Tensor]:
        """Settle once, then require the stored post-state to survive a fresh replay."""
        if active_mask is None:
            active_mask = torch.ones(self.env.num_envs, device=self.device, dtype=torch.bool)
        else:
            active_mask = torch.as_tensor(active_mask, device=self.device, dtype=torch.bool)
        self._restore_state(warm_state)
        warm_clamp_evidence: dict[str, torch.Tensor] = {}
        warm_replay_evidence: dict[str, Any] = {}
        advance_reset_absolute_target_hold(
            self.env,
            self.cfg.row_settle_s,
            warm_state["arm_joint_target"],
            warm_state["finger_joint_target"],
            clamp_evidence=warm_clamp_evidence,
            replay_evidence=warm_replay_evidence,
            starts_grasped=starts_grasped,
        )
        candidate_arm_target, candidate_capture_clamp_delta = runtime_persistent_arm_target(
            self.env,
            warm_state["arm_joint_target"],
        )
        candidate = self._capture_state(
            candidate_arm_target,
            warm_state["finger_joint_target"],
        )
        candidate_history_evidence = self._restore_state(candidate)
        assert candidate_history_evidence is not None
        candidate_clamp_evidence: dict[str, torch.Tensor] = {}
        candidate_replay_evidence: dict[str, Any] = {}
        candidate_steps = advance_reset_absolute_target_hold(
            self.env,
            self.cfg.row_settle_s,
            candidate["arm_joint_target"],
            candidate["finger_joint_target"],
            clamp_evidence=candidate_clamp_evidence,
            replay_evidence=candidate_replay_evidence,
            starts_grasped=starts_grasped,
        )
        maximum_arm_target_clamp_delta = torch.maximum(
            torch.maximum(
                warm_clamp_evidence["maximum_arm_target_clamp_delta"],
                candidate_capture_clamp_delta,
            ),
            candidate_clamp_evidence["maximum_arm_target_clamp_delta"],
        )
        any_arm_target_clamped = (
            warm_clamp_evidence["any_arm_target_clamped"]
            | (candidate_capture_clamp_delta > 1.0e-7)
            | candidate_clamp_evidence["any_arm_target_clamped"]
        )
        reset_zero_action_unclamped = ~any_arm_target_clamped
        maximum_arm_target_drift = torch.maximum(
            warm_clamp_evidence["maximum_arm_target_drift"],
            candidate_clamp_evidence["maximum_arm_target_drift"],
        )
        reset_absolute_target_stable = maximum_arm_target_drift <= 1.0e-7
        warm_hard_valid = (
            warm_replay_evidence["stored_state_finite"]
            & warm_replay_evidence["stored_task_state_finite_and_normalized"]
            & warm_replay_evidence["stored_drive_disabled"]
            & warm_replay_evidence["all_post_step_state_finite"]
            & warm_replay_evidence["all_post_step_task_state_finite_and_normalized"]
            & warm_replay_evidence["all_post_step_collision_free"]
            & warm_replay_evidence["all_post_step_drive_disabled"]
            & warm_replay_evidence["all_post_step_expected_contact_state"]
            & warm_replay_evidence["stored_arm_target_tracking_bounded"]
            & warm_replay_evidence["all_post_step_arm_target_tracking_bounded"]
            & (not warm_replay_evidence["any_contact_overflow"])
        )
        if warm_replay_evidence["any_contact_overflow"]:
            raise RuntimeError("Global contact-buffer overflow during warm reset replay.")
        if not bool(warm_replay_evidence["all_post_step_drive_disabled"].all()):
            raise RuntimeError("A construction drive became enabled during warm reset replay.")
        if starts_grasped:
            contact_state_valid = (
                candidate_replay_evidence["all_post_step_bilateral_grasp"]
                & candidate_replay_evidence["all_post_step_proxy_bilateral_contact"]
                & (candidate_replay_evidence["minimum_left_proxy_contact_count"] >= 1)
                & (candidate_replay_evidence["minimum_right_proxy_contact_count"] >= 1)
            )
        else:
            contact_state_valid = (
                candidate_replay_evidence["all_post_step_zero_proxy_contacts"]
                & (candidate_replay_evidence["maximum_left_proxy_contact_count"] == 0)
                & (candidate_replay_evidence["maximum_right_proxy_contact_count"] == 0)
            )
        valid = (
            active_mask
            & warm_hard_valid
            & candidate_replay_evidence["stored_state_finite"]
            & candidate_replay_evidence["stored_task_state_finite_and_normalized"]
            & candidate_replay_evidence["stored_drive_disabled"]
            & candidate_replay_evidence["all_post_step_state_finite"]
            & candidate_replay_evidence["all_post_step_task_state_finite_and_normalized"]
            & candidate_replay_evidence["all_post_step_collision_free"]
            & candidate_replay_evidence["all_post_step_drive_disabled"]
            & candidate_replay_evidence["all_post_step_expected_contact_state"]
            & candidate_replay_evidence["stored_arm_target_tracking_bounded"]
            & candidate_replay_evidence["all_post_step_arm_target_tracking_bounded"]
            & contact_state_valid
            & joint_limit_mask(self.env, candidate["arm_joint_position"])
            & joint_limit_mask(self.env, candidate["arm_joint_target"])
            & reset_zero_action_unclamped
            & reset_absolute_target_stable
            & (candidate_steps == PICK_INSERT_RESET_REPLAY_POST_STEP_SAMPLES)
            & (candidate_replay_evidence["post_step_samples"] == PICK_INSERT_RESET_REPLAY_POST_STEP_SAMPLES)
            & (candidate_replay_evidence["stored_maximum_cable_speed_m_s"] <= self.cfg.maximum_row_cable_speed_m_s)
            & (candidate_replay_evidence["maximum_post_step_cable_speed_m_s"] <= self.cfg.maximum_row_cable_speed_m_s)
            & (candidate_replay_evidence["final_cable_speed_m_s"] <= self.cfg.maximum_row_cable_speed_m_s)
            & (
                candidate_replay_evidence["stored_maximum_arm_joint_speed_rad_s"]
                <= self.cfg.maximum_row_arm_joint_speed_rad_s
            )
            & (
                candidate_replay_evidence["maximum_post_step_arm_joint_speed_rad_s"]
                <= self.cfg.maximum_row_arm_joint_speed_rad_s
            )
            & (candidate_replay_evidence["final_arm_joint_speed_rad_s"] <= self.cfg.maximum_row_arm_joint_speed_rad_s)
            & (
                candidate_replay_evidence["stored_maximum_finger_joint_speed_m_s"]
                <= self.cfg.maximum_row_finger_joint_speed_m_s
            )
            & (
                candidate_replay_evidence["maximum_post_step_finger_joint_speed_m_s"]
                <= self.cfg.maximum_row_finger_joint_speed_m_s
            )
            & (candidate_replay_evidence["final_finger_joint_speed_m_s"] <= self.cfg.maximum_row_finger_joint_speed_m_s)
            & (candidate_replay_evidence["maximum_socket_excursion_m"] <= self.cfg.maximum_socket_drift_m)
            & (candidate_replay_evidence["maximum_plug_excursion_m"] <= self.cfg.maximum_row_plug_drift_m)
            & (candidate_replay_evidence["maximum_body_excursion_m"] <= self.cfg.maximum_row_body_drift_m)
            & (candidate_replay_evidence["maximum_invalid_contact_count"] == 0)
            & (not candidate_replay_evidence["any_contact_overflow"])
            & self._vbd_pose_history_applied_mask(candidate_history_evidence)
        )
        if candidate_replay_evidence["any_contact_overflow"]:
            raise RuntimeError("Global contact-buffer overflow during cold reset replay.")
        if not bool(candidate_replay_evidence["all_post_step_drive_disabled"].all()):
            raise RuntimeError("A construction drive became enabled during cold reset replay.")
        public_replay_evidence = {
            name: value for name, value in candidate_replay_evidence.items() if not name.startswith("_")
        }
        public_replay_evidence.update(self._vbd_pose_history_report_evidence(candidate_history_evidence))
        public_replay_evidence.update(
            {
                "simulation_time_s": self.cfg.row_settle_s,
                "simulation_steps": candidate_steps,
                "maximum_arm_target_clamp_delta_rad": maximum_arm_target_clamp_delta,
                "zero_action_unclamped": reset_zero_action_unclamped,
                "maximum_arm_target_drift_rad": maximum_arm_target_drift,
                "absolute_target_stable": reset_absolute_target_stable,
            }
        )
        evidence = {
            "maximum_socket_drift_m": candidate_replay_evidence["maximum_socket_excursion_m"],
            "maximum_body_drift_m": candidate_replay_evidence["maximum_body_excursion_m"],
            "maximum_cable_speed_m_s": torch.maximum(
                candidate_replay_evidence["stored_maximum_cable_speed_m_s"],
                candidate_replay_evidence["maximum_post_step_cable_speed_m_s"],
            ),
            "reset_zero_action_unclamped": reset_zero_action_unclamped,
            "maximum_arm_target_clamp_delta_rad": maximum_arm_target_clamp_delta,
            "maximum_arm_target_drift_rad": maximum_arm_target_drift,
            "reset_absolute_target_stable": reset_absolute_target_stable,
            "invalid_contact_count": candidate_replay_evidence["maximum_invalid_contact_count"],
            "reset_replay": public_replay_evidence,
        }
        return candidate, evidence, valid

    def _phase_semantics(
        self,
        phase: int,
        task_q: torch.Tensor,
        goal_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tcp_distance = torch.linalg.vector_norm(self.env.tcp_pose_e()[:, :3] - self.env.plug_grasp_position_e(), dim=-1)
        goal_error = scalar_goal_error(
            task_q,
            goal_q,
            plug_body_index=self.plug_index,
            latch_body_index=self.latch_index,
        )
        if phase == 0:
            semantic = goal_error <= 0.020
        elif phase == 1:
            semantic = (goal_error >= 0.015) & (goal_error <= 0.090)
        elif phase == 2:
            semantic = (task_q[:, self.plug_index, 2] >= 0.06) & (goal_error >= 0.05)
        elif phase == 3:
            semantic = (task_q[:, self.plug_index, 2] >= 0.02) & (goal_error >= 0.08)
        elif phase == 4:
            semantic = (tcp_distance >= 0.02) & (tcp_distance <= 0.10)
        else:
            semantic = tcp_distance >= 0.10
        return semantic, goal_error, tcp_distance

    def _acquire_grasp(
        self,
        arm_seed: torch.Tensor,
        *,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        arm_target, acquired, evidence = self._acquire_current_plug(
            arm_seed,
            duration_s=2.5,
            active_mask=active_mask,
            move_attempt_count=_ORACLE_ACQUISITION_MOVE_ATTEMPT_COUNT,
            move_settle_s=_ORACLE_ACQUISITION_MOVE_SETTLE_S,
        )
        return arm_target, acquired, evidence["last_finger_target"]

    def _replay_oracle_entry(
        self,
        state: dict[str, torch.Tensor],
        *,
        starts_grasped: bool,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Restore and cold-replay one candidate before any oracle action."""
        history_evidence = self._restore_state(state)
        if history_evidence is None:
            raise RuntimeError("Oracle entry replay requires both stored VBD pose histories.")
        reset_contact_count = _contact_count()
        clamp_evidence: dict[str, torch.Tensor] = {}
        replay_evidence: dict[str, Any] = {}
        replay_steps = advance_reset_absolute_target_hold(
            self.env,
            self.cfg.row_settle_s,
            state["arm_joint_target"],
            state["finger_joint_target"],
            clamp_evidence=clamp_evidence,
            replay_evidence=replay_evidence,
            starts_grasped=starts_grasped,
        )
        settled_q, settled_qd = self.env.read_task_state()
        settled_arm_q, _, _, _ = self.env.read_robot_state()
        settled_target, settled_target_clamp_delta = runtime_persistent_arm_target(
            self.env,
            state["arm_joint_target"],
        )
        maximum_target_clamp_delta = torch.maximum(
            clamp_evidence["maximum_arm_target_clamp_delta"],
            settled_target_clamp_delta,
        )
        zero_action_unclamped = ~(
            clamp_evidence["any_arm_target_clamped"]
            | (settled_target_clamp_delta > PICK_INSERT_RESET_MAX_ARM_TARGET_CLAMP_DELTA_RAD)
        )
        absolute_target_stable = (
            clamp_evidence["maximum_arm_target_drift"] <= PICK_INSERT_RESET_MAX_ARM_TARGET_CLAMP_DELTA_RAD
        )
        target_tracking_bounded = (
            replay_evidence["stored_arm_target_tracking_bounded"]
            & replay_evidence["all_post_step_arm_target_tracking_bounded"]
        )
        if starts_grasped:
            contact_state_valid = (
                replay_evidence["all_post_step_bilateral_grasp"]
                & replay_evidence["all_post_step_proxy_bilateral_contact"]
                & (replay_evidence["minimum_left_proxy_contact_count"] >= 1)
                & (replay_evidence["minimum_right_proxy_contact_count"] >= 1)
            )
        else:
            contact_state_valid = (
                replay_evidence["all_post_step_zero_proxy_contacts"]
                & (replay_evidence["maximum_left_proxy_contact_count"] == 0)
                & (replay_evidence["maximum_right_proxy_contact_count"] == 0)
            )
        post_step_finite = (
            replay_evidence["all_post_step_state_finite"]
            & replay_evidence["all_post_step_task_state_finite_and_normalized"]
        )
        collision_valid = (
            replay_evidence["all_post_step_collision_free"]
            & (replay_evidence["maximum_invalid_contact_count"] == 0)
            & (not replay_evidence["any_contact_overflow"])
        )
        if replay_evidence["any_contact_overflow"]:
            raise RuntimeError("Global contact-buffer overflow during generator oracle entry replay.")
        if not bool(replay_evidence["all_post_step_drive_disabled"].all()):
            raise RuntimeError("A construction drive became enabled during generator oracle entry replay.")
        robot_speed_bounded = (
            (replay_evidence["stored_maximum_arm_joint_speed_rad_s"] <= self.cfg.maximum_row_arm_joint_speed_rad_s)
            & (replay_evidence["maximum_post_step_arm_joint_speed_rad_s"] <= self.cfg.maximum_row_arm_joint_speed_rad_s)
            & (replay_evidence["final_arm_joint_speed_rad_s"] <= self.cfg.maximum_row_arm_joint_speed_rad_s)
            & (replay_evidence["stored_maximum_finger_joint_speed_m_s"] <= self.cfg.maximum_row_finger_joint_speed_m_s)
            & (
                replay_evidence["maximum_post_step_finger_joint_speed_m_s"]
                <= self.cfg.maximum_row_finger_joint_speed_m_s
            )
            & (replay_evidence["final_finger_joint_speed_m_s"] <= self.cfg.maximum_row_finger_joint_speed_m_s)
        )
        cable_speed_bounded = (
            (replay_evidence["stored_maximum_cable_speed_m_s"] <= self.cfg.maximum_row_cable_speed_m_s)
            & (replay_evidence["maximum_post_step_cable_speed_m_s"] <= self.cfg.maximum_row_cable_speed_m_s)
            & (replay_evidence["final_cable_speed_m_s"] <= self.cfg.maximum_row_cable_speed_m_s)
        )
        body_excursion_bounded = (
            (replay_evidence["maximum_body_excursion_m"] <= self.cfg.maximum_row_body_drift_m)
            & (replay_evidence["maximum_plug_excursion_m"] <= self.cfg.maximum_row_plug_drift_m)
            & (replay_evidence["maximum_socket_excursion_m"] <= self.cfg.maximum_socket_drift_m)
        )
        valid = (
            active_mask
            & torch.full_like(active_mask, reset_contact_count == 0)
            & replay_evidence["stored_state_finite"]
            & replay_evidence["stored_task_state_finite_and_normalized"]
            & replay_evidence["stored_drive_disabled"]
            & task_state_is_finite_and_normalized(settled_q, settled_qd)
            & post_step_finite
            & collision_valid
            & replay_evidence["all_post_step_drive_disabled"]
            & replay_evidence["all_post_step_expected_contact_state"]
            & contact_state_valid
            & joint_limit_mask(self.env, state["arm_joint_position"])
            & joint_limit_mask(self.env, state["arm_joint_target"])
            & joint_limit_mask(self.env, settled_arm_q)
            & joint_limit_mask(self.env, settled_target)
            & zero_action_unclamped
            & absolute_target_stable
            & target_tracking_bounded
            & robot_speed_bounded
            & cable_speed_bounded
            & body_excursion_bounded
            & (maximum_target_clamp_delta <= PICK_INSERT_RESET_MAX_ARM_TARGET_CLAMP_DELTA_RAD)
            & (replay_steps == PICK_INSERT_RESET_REPLAY_POST_STEP_SAMPLES)
            & (replay_evidence["post_step_samples"] == PICK_INSERT_RESET_REPLAY_POST_STEP_SAMPLES)
            & self._vbd_pose_history_applied_mask(history_evidence)
            & torch.full_like(active_mask, history_evidence["body_order_exact"] is True)
        )
        self._assert_drive_disabled("generator oracle entry replay")
        return settled_target, valid

    def _oracle(
        self,
        state: dict[str, torch.Tensor],
        goal_q: torch.Tensor,
        goal_arm_target: torch.Tensor,
        *,
        phase: int,
        starts_grasped: bool,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if active_mask is None:
            active_mask = torch.ones(self.env.num_envs, device=self.device, dtype=torch.bool)
        else:
            active_mask = torch.as_tensor(active_mask, device=self.device, dtype=torch.bool)
        arm_target, entry_valid = self._replay_oracle_entry(
            state,
            starts_grasped=starts_grasped,
            active_mask=active_mask,
        )
        if starts_grasped:
            acquired = (
                entry_valid
                & grasp_metrics(
                    self.env,
                    state["finger_joint_target"],
                    retaining_grasp=True,
                ).valid
            )
            recovery_start_finger_target = state["finger_joint_target"]
        else:
            arm_target, acquired, recovery_start_finger_target = self._acquire_grasp(
                arm_target,
                active_mask=entry_valid,
            )
        with _PerLaneTargetHold(
            self.env,
            acquired,
            arm_target,
            recovery_start_finger_target,
        ) as recovery_hold:
            recovery, metrics = scripted_recovery(
                self.env,
                self._counted_ik,
                goal_q,
                self.local_grasp_orientation,
                self.closed_finger_target,
                arm_target_start=arm_target,
                goal_arm_target=goal_arm_target,
                motion_s=self.cfg.recovery_motion_s,
                settle_s=self.cfg.recovery_settle_s,
                compensation_max_iterations=self.cfg.recovery_compensation_iterations,
                compensation_tolerance_m=self.cfg.recovery_compensation_tolerance_m,
                maximum_ik_joint_step_rad=self.cfg.maximum_ik_joint_step_rad,
                plug_body_index=self.plug_index,
                latch_body_index=self.latch_index,
                arm_target_is_absolute=True,
                lane_hold=recovery_hold,
                motion_policy=PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["motion_policy"],
                pick_insert_phase=phase,
            )
            recovery_hold.deactivate(~recovery, reason="generator-oracle-recovery-final-validation")
            recovery &= recovery_hold.active_mask
            metrics["lane_failure_masks"] = recovery_hold.reason_masks
            metrics["last_arm_target"] = recovery_hold.last_sent_arm_target
            metrics["last_finger_target"] = recovery_hold.last_sent_finger_target
        final_collision = collision_metrics(self.env)
        if final_collision.contact_overflow:
            raise RuntimeError("Global contact-buffer overflow at the generator oracle result boundary.")
        drive_disabled = ~self._drive_enabled() & ~self._orientation_hold_enabled()
        if not bool(drive_disabled.all()):
            raise RuntimeError("A construction drive became enabled during generator oracle recovery.")
        exact_dwell = metrics["exact_success_dwell_satisfied"]
        exact_collision = metrics["exact_success_all_samples_collision_free"]
        exact_grasp = metrics["exact_success_all_samples_bilateral_grasp"]
        exact_finite = metrics["exact_success_all_samples_finite"]
        valid = (
            acquired
            & recovery
            & final_collision.valid
            & drive_disabled
            & exact_dwell
            & exact_collision
            & exact_grasp
            & exact_finite
        )
        return valid, {
            "physical_grasp_acquired": acquired,
            "full_recovery": recovery,
            "exact_success_dwell": exact_dwell,
            "exact_all_samples_collision_free": exact_collision,
            "exact_all_samples_bilateral_grasp": exact_grasp,
            "exact_all_samples_finite": exact_finite,
            "exact_maximum_body_excursion": metrics["exact_success_maximum_body_excursion"],
            "exact_maximum_cable_linear_speed": metrics["exact_success_maximum_cable_linear_speed"],
            "drive_disabled": drive_disabled,
            "no_invalid_contacts": final_collision.valid,
            "goal_error": metrics["goal_error"],
            "recovery_lane_failure_masks": metrics["lane_failure_masks"],
            "recovery_diagnostic_evidence": _scripted_recovery_diagnostic_evidence(metrics),
        }

    def _record_rejections(self, phase: int, checks: dict[str, torch.Tensor], valid: torch.Tensor) -> None:
        for name, check in checks.items():
            self.rejection_counts[phase][name] += int((~check).sum())
        self.rejection_counts[phase]["accepted"] += int(valid.sum())

    def _fast_task_state(
        self,
        phase: int,
        pickup_pose: torch.Tensor,
        goal_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct one coherent zero-velocity phase batch without stepping physics."""
        goal_plug = goal_q[:, self.plug_index]
        plug_target = goal_plug.clone()
        if phase in (0, 1, 2):
            local_offset = torch.zeros((self.env.num_envs, 3), device=self.device)
            if phase == 0:
                axial_shortfall, _band_indices = sample_phase_0_reverse_curriculum_axial_shortfalls(
                    self.env.num_envs,
                    device=self.device,
                    rng=self.random,
                    dtype=goal_q.dtype,
                )
                local_offset[:, 1] = -axial_shortfall
            else:
                local_offset[:, 1] = -self.cfg.phase_1_axial_offset_m
            if phase == 2:
                local_offset[:, 1] = -0.08
            plug_target[:, :3] += math_utils.quat_apply(goal_plug[:, 3:7], local_offset)
            if phase == 2:
                plug_target[:, 2] += self.cfg.phase_2_lift_m
        elif phase == 3:
            plug_target = pickup_pose.clone()
            plug_target[:, 2] += self.cfg.phase_3_lift_m
        elif phase in (4, 5):
            plug_target = pickup_pose.clone()
        else:
            raise ValueError(f"Unknown pick-insert reset phase {phase}.")

        zero_velocity = torch.zeros(
            (self.env.num_envs, self.layout.body_count, 6),
            device=self.device,
            dtype=goal_q.dtype,
        )
        transformed_q, _ = _rigid_transform_task_state(
            goal_q,
            zero_velocity,
            goal_plug,
            plug_target,
        )
        # The socket owns the sampled goal frame.  Every other task body is
        # moved by one rigid transform so cable rest lengths and plug-relative
        # anchors are preserved exactly.
        transformed_q[:, self.socket_index] = goal_q[:, self.socket_index]
        return transformed_q, zero_velocity

    def _fast_row_ik(
        self,
        phase: int,
        task_q: torch.Tensor,
        goal_arm_target: torch.Tensor,
    ) -> tuple[Any, torch.Tensor]:
        """Solve the one static TCP target associated with a fast reset row."""
        plug_pose = task_q[:, self.plug_index]
        target_position, target_orientation = self._desired_tcp_pose(
            plug_pose,
            orientation_error_xyzw=self._sample_phase_tcp_orientation_error(phase),
        )
        if phase == 4:
            target_position = target_position.clone()
            target_position[:, 2] += self.cfg.phase_4_pregrasp_height_m
        elif phase == 5:
            away_lower = torch.tensor((0.34, -0.02, 0.26), device=self.device)
            away_upper = torch.tensor((0.62, 0.22, 0.42), device=self.device)
            target_position = away_lower + torch.rand(
                (self.env.num_envs, 3), device=self.device, generator=self.random
            ) * (away_upper - away_lower)
        finger_target = self.closed_finger_target if _PHASE_STARTS_GRASPED[phase] else self.open_finger_q
        arm_seed = goal_arm_target if phase in (0, 1, 2) else self.home_arm_q
        solution = self._solve_ik(
            target_position,
            target_orientation,
            finger_target,
            arm_seed=arm_seed,
        )
        return solution, finger_target

    def _fast_static_checks(
        self,
        phase: int,
        task_q: torch.Tensor,
        task_qd: torch.Tensor,
        goal_q: torch.Tensor,
        arm_q: torch.Tensor,
        finger_q: torch.Tensor,
        tcp_position: torch.Tensor,
        ik_valid: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Apply the bounded static checks declared by the fast-reset policy."""
        workspace_lower = torch.as_tensor(
            self.env.cfg.task_body_workspace_lower,
            device=self.device,
            dtype=task_q.dtype,
        )
        workspace_upper = torch.as_tensor(
            self.env.cfg.task_body_workspace_upper,
            device=self.device,
            dtype=task_q.dtype,
        )
        lower_margin = task_q[..., :3] - workspace_lower
        upper_margin = workspace_upper - task_q[..., :3]
        minimum_workspace_margin = torch.minimum(lower_margin, upper_margin).amin(dim=(1, 2))
        workspace = minimum_workspace_margin >= 0.0

        contact_surface = self.env.cfg.scene.table_contact_surface
        table_center_z = float(contact_surface.init_state.pos[2])
        table_half_height = 0.5 * float(contact_surface.spawn.size[2])
        table_top_z = table_center_z + table_half_height
        cable_position = task_q[:, self.cable_slice, :3]
        minimum_cable_support_clearance = cable_position[..., 2].amin(dim=-1) - CABLE_RADIUS - table_top_z

        cable_distance = torch.cdist(cable_position, cable_position)
        cable_count = cable_position.shape[1]
        nonadjacent = torch.triu(
            torch.ones((cable_count, cable_count), device=self.device, dtype=torch.bool),
            diagonal=2,
        )
        minimum_nonadjacent_cable_separation = cable_distance[:, nonadjacent].amin(dim=-1)
        socket_position = task_q[:, self.socket_index, None, :3]
        minimum_cable_socket_center_distance = torch.linalg.vector_norm(
            cable_position - socket_position,
            dim=-1,
        ).amin(dim=-1)
        collision_policy = FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY["collision_filter"]
        collision_filtered = (
            (minimum_cable_support_clearance >= -float(collision_policy["maximum_table_penetration_m"]))
            & (
                minimum_nonadjacent_cable_separation
                >= 2.0 * CABLE_RADIUS + float(collision_policy["minimum_nonadjacent_cable_surface_gap_m"])
            )
            & (
                minimum_cable_socket_center_distance
                >= float(collision_policy["minimum_cable_socket_center_distance_m"])
            )
        )

        grasp_position = task_q[:, self.plug_index, :3] + math_utils.quat_apply(
            task_q[:, self.plug_index, 3:7],
            torch.as_tensor(self.env.cfg.plug_grasp_offset, device=self.device, dtype=task_q.dtype).expand(
                self.env.num_envs, -1
            ),
        )
        tcp_distance = torch.linalg.vector_norm(tcp_position - grasp_position, dim=-1)
        goal_error = scalar_goal_error(
            task_q,
            goal_q,
            plug_body_index=self.plug_index,
            latch_body_index=self.latch_index,
        )
        goal_plug = goal_q[:, self.plug_index]
        plug_translation_error_local = math_utils.quat_apply_inverse(
            goal_plug[:, 3:7],
            task_q[:, self.plug_index, :3] - goal_plug[:, :3],
        )
        signed_axial_error = plug_translation_error_local[:, 1]
        phase_0_axial_shortfall = -signed_axial_error
        phase_0_band_index = phase_0_reverse_curriculum_band_indices(phase_0_axial_shortfall)
        initial_runtime_success = exact_success_from_state(
            self.env,
            task_q,
            task_qd,
            goal_q,
            plug_body_index=self.plug_index,
            latch_body_index=self.latch_index,
        ).mask
        if phase == 0:
            phase_semantics = (
                (signed_axial_error < 0.0)
                & (phase_0_band_index >= 0)
                & (goal_error <= 0.020)
                & ~initial_runtime_success
            )
        elif phase == 1:
            phase_semantics = (goal_error >= 0.015) & (goal_error <= 0.090)
        elif phase == 2:
            phase_semantics = (task_q[:, self.plug_index, 2] >= 0.06) & (goal_error >= 0.05)
        elif phase == 3:
            phase_semantics = (task_q[:, self.plug_index, 2] >= 0.02) & (goal_error >= 0.08)
        elif phase == 4:
            phase_semantics = (tcp_distance >= 0.02) & (tcp_distance <= 0.10)
        else:
            phase_semantics = tcp_distance >= 0.10

        finite = (
            task_state_is_finite_and_normalized(task_q, task_qd)
            & torch.isfinite(arm_q).all(dim=-1)
            & torch.isfinite(finger_q).all(dim=-1)
            & torch.isfinite(tcp_position).all(dim=-1)
        )
        checks = {
            "finite": finite,
            "ik_solved": ik_valid,
            "joint_limits": joint_limit_mask(
                self.env,
                arm_q,
                margin=float(FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY["joint_limit_margin_rad"]),
            ),
            "workspace": workspace,
            "collision_filtered": collision_filtered,
            "phase_semantics": phase_semantics,
        }
        return checks, {
            "initial_goal_error": goal_error,
            "initial_tcp_grasp_distance": tcp_distance,
            "minimum_workspace_margin_m": minimum_workspace_margin,
            "minimum_cable_support_clearance_m": minimum_cable_support_clearance,
            "minimum_nonadjacent_cable_separation_m": minimum_nonadjacent_cable_separation,
            "minimum_cable_socket_center_distance_m": minimum_cable_socket_center_distance,
            "phase_0_signed_axial_error_m": signed_axial_error,
            "phase_0_axial_shortfall_m": phase_0_axial_shortfall,
            "phase_0_reverse_curriculum_band_index": phase_0_band_index,
            "initial_runtime_geometric_success": initial_runtime_success,
        }

    def _fast_collide_only_checks(
        self,
        *,
        task_q: torch.Tensor,
        task_qd: torch.Tensor,
        arm_target: torch.Tensor,
        finger_position: torch.Tensor,
        finger_target: torch.Tensor,
        starts_grasped: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Author one candidate batch and run exact zero-step Newton collision checks."""
        self.env.write_task_state(task_q, task_qd)
        self.env.write_robot_state(
            arm_target,
            finger_position,
            arm_target=arm_target,
            finger_target=finger_target,
        )
        collision = collide_only_metrics(
            self.env,
            penetration_tolerance=float(
                FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY["collision_filter"]["newton_query"]["penetration_tolerance_m"]
            ),
            require_bilateral_grasp=starts_grasped,
        )
        if collision.contact_overflow:
            raise RuntimeError("Global contact-buffer overflow during fast collide-only reset admission.")
        if starts_grasped:
            valid = (
                collision.valid
                & grasp_metrics(
                    self.env,
                    finger_target,
                    retaining_grasp=True,
                ).valid
            )
        else:
            valid = collision.valid & (collision.grasp_contact_count == 0)
        return valid, {
            "collide_only_invalid_contact_count": collision.invalid_contact_count,
            "collide_only_grasp_contact_count": collision.grasp_contact_count,
            "collide_only_left_grasp_contact_count": collision.left_grasp_contact_count,
            "collide_only_right_grasp_contact_count": collision.right_grasp_contact_count,
            "collide_only_contact_overflow": torch.full(
                (self.env.num_envs,),
                collision.contact_overflow,
                device=self.device,
                dtype=torch.bool,
            ),
        }

    def _fast_collision_authoring_state(
        self,
        *,
        task_q: torch.Tensor,
        task_qd: torch.Tensor,
        goal_q: torch.Tensor,
        arm_target: torch.Tensor,
        ik_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replace rejected lanes with finite states before invoking collision kernels."""
        authorable = (
            ik_valid & task_state_is_finite_and_normalized(task_q, task_qd) & torch.isfinite(arm_target).all(dim=-1)
        )
        collision_task_q = torch.where(authorable[:, None, None], task_q, goal_q)
        collision_task_qd = torch.where(authorable[:, None, None], task_qd, torch.zeros_like(task_qd))
        collision_arm_target = torch.where(authorable[:, None], arm_target, self.home_arm_q)
        return collision_task_q, collision_task_qd, collision_arm_target

    def _append_fast_accepted_metrics(
        self,
        *,
        phase: int,
        chosen: torch.Tensor,
        checks: Mapping[str, torch.Tensor],
        metrics: Mapping[str, torch.Tensor],
        row_ik: Any,
    ) -> None:
        """Batch-copy accepted evidence to CPU before creating plain metadata records."""
        check_rows = {name: value[chosen].detach().cpu() for name, value in checks.items()}
        metric_rows = {name: value[chosen].detach().cpu() for name, value in metrics.items()}
        goal_position_residual = self._last_goal_ik_result.position_residual[chosen].detach().cpu()
        goal_rotation_residual = self._last_goal_ik_result.rotation_residual[chosen].detach().cpu()
        row_position_residual = row_ik.position_residual[chosen].detach().cpu()
        row_rotation_residual = row_ik.rotation_residual[chosen].detach().cpu()
        for row_index in range(chosen.numel()):
            record: dict[str, Any] = {
                "checks": {name: bool(value[row_index]) for name, value in check_rows.items()},
                "goal_ik_position_residual_m": float(goal_position_residual[row_index]),
                "goal_ik_rotation_residual_rad": float(goal_rotation_residual[row_index]),
                "row_ik_position_residual_m": float(row_position_residual[row_index]),
                "row_ik_rotation_residual_rad": float(row_rotation_residual[row_index]),
                **{name: float(value[row_index]) for name, value in metric_rows.items()},
            }
            record.update(
                {
                    "collide_only_invalid_contact_count": int(
                        metric_rows["collide_only_invalid_contact_count"][row_index]
                    ),
                    "collide_only_grasp_contact_count": int(metric_rows["collide_only_grasp_contact_count"][row_index]),
                    "collide_only_left_grasp_contact_count": int(
                        metric_rows["collide_only_left_grasp_contact_count"][row_index]
                    ),
                    "collide_only_right_grasp_contact_count": int(
                        metric_rows["collide_only_right_grasp_contact_count"][row_index]
                    ),
                    "collide_only_contact_overflow": bool(metric_rows["collide_only_contact_overflow"][row_index]),
                }
            )
            if phase == 0:
                band_index = int(metric_rows["phase_0_reverse_curriculum_band_index"][row_index])
                record["phase_0_reverse_curriculum_band"] = PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_BAND_NAMES[
                    band_index
                ]
                record["initial_runtime_geometric_success"] = bool(
                    metric_rows["initial_runtime_geometric_success"][row_index]
                )
            self.accepted_oracle_metrics[phase].append(record)

    @torch.inference_mode()
    def _generate_phase_fast(
        self,
        phase: int,
        canonical_goal: dict[str, torch.Tensor],
        *,
        initial_accepted: list[dict[str, torch.Tensor]] | None = None,
        start_batch_index: int = 0,
        completed_batch_callback: Any = None,
    ) -> dict[str, torch.Tensor]:
        """Generate one complete phase using static transforms and batched IK only."""
        accepted = [] if initial_accepted is None else list(initial_accepted)
        accepted_count = sum(int(part["phase"].shape[0]) for part in accepted)
        if not 0 <= start_batch_index <= self.cfg.max_batches_per_phase:
            raise ValueError("Generation phase resume batch index is outside the configured limit.")
        if not 0 <= accepted_count <= self.cfg.rows_per_phase:
            raise ValueError("Generation phase resume rows exceed the configured phase count.")
        starts_grasped = _PHASE_STARTS_GRASPED[phase]
        for batch_index in range(start_batch_index, self.cfg.max_batches_per_phase):
            if accepted_count >= self.cfg.rows_per_phase:
                break
            socket_pose, pickup_pose = self._sample_scene()
            goal_q, _goal_qd, goal_arm_target, goal_ik_valid = self._row_goal(canonical_goal, socket_pose)
            task_q, task_qd = self._fast_task_state(phase, pickup_pose, goal_q)
            row_ik, finger_target = self._fast_row_ik(phase, task_q, goal_arm_target)
            arm_target = row_ik.arm_q
            finger_position = finger_target
            if starts_grasped:
                finger_position = canonical_goal["finger_joint_position"].to(self.device).repeat(self.env.num_envs, 1)
            ik_valid = goal_ik_valid & row_ik.valid
            collision_task_q, collision_task_qd, collision_arm_target = self._fast_collision_authoring_state(
                task_q=task_q,
                task_qd=task_qd,
                goal_q=goal_q,
                arm_target=arm_target,
                ik_valid=ik_valid,
            )
            collide_only_valid, collide_only_evidence = self._fast_collide_only_checks(
                task_q=collision_task_q,
                task_qd=collision_task_qd,
                arm_target=collision_arm_target,
                finger_position=finger_position,
                finger_target=finger_target,
                starts_grasped=starts_grasped,
            )
            checks, metrics = self._fast_static_checks(
                phase,
                task_q,
                task_qd,
                goal_q,
                arm_target,
                finger_position,
                row_ik.tcp_position,
                ik_valid,
            )
            checks["collision_filtered"] &= collide_only_valid
            metrics.update(collide_only_evidence)
            valid = torch.stack(tuple(checks.values())).all(dim=0)
            self._record_rejections(phase, checks, valid)
            self.attempt_counts[phase] += self.env.num_envs
            snapshot = {
                "arm_joint_position": arm_target.clone(),
                "arm_joint_target": arm_target.clone(),
                "arm_joint_velocity": torch.zeros_like(arm_target),
                "finger_joint_position": finger_position.clone(),
                "finger_joint_velocity": torch.zeros_like(finger_target),
                "finger_joint_target": finger_target.clone(),
                "task_body_pose": task_q.clone(),
                "task_body_previous_pose": task_q.clone(),
                "task_body_coupling_previous_pose": task_q.clone(),
                "task_body_velocity": task_qd.clone(),
                "goal_task_body_pose": goal_q.clone(),
                "goal_arm_joint_target": goal_arm_target.clone(),
                "phase": torch.full((self.env.num_envs,), phase, device=self.device, dtype=torch.int64),
                "starts_grasped": torch.full(
                    (self.env.num_envs,), starts_grasped, device=self.device, dtype=torch.bool
                ),
                "difficulty": torch.full(
                    (self.env.num_envs,), _PHASE_DIFFICULTY[phase], device=self.device, dtype=torch.float32
                ),
                "initial_goal_error": metrics["initial_goal_error"].clone(),
                "initial_tcp_grasp_distance": metrics["initial_tcp_grasp_distance"].clone(),
                "progress_threshold": (
                    0.15 * (metrics["initial_goal_error"] if starts_grasped else metrics["initial_tcp_grasp_distance"])
                ).clamp(2.0e-4, 0.02),
            }
            remaining = self.cfg.rows_per_phase - accepted_count
            chosen = torch.where(valid)[0][:remaining]
            accepted_chunk: dict[str, torch.Tensor] | None = None
            if chosen.numel():
                accepted_chunk = {name: value[chosen].detach().clone() for name, value in snapshot.items()}
                accepted.append(accepted_chunk)
                self._append_fast_accepted_metrics(
                    phase=phase,
                    chosen=chosen,
                    checks=checks,
                    metrics=metrics,
                    row_ik=row_ik,
                )
                accepted_count += int(chosen.numel())
            check_counts = ", ".join(f"{name}={int(value.sum())}" for name, value in checks.items())
            print(
                f"[PICK-INSERT FAST RESET] phase={phase}:{PICK_INSERT_PHASE_NAMES[phase]} "
                f"batch={batch_index + 1} accepted={accepted_count}/{self.cfg.rows_per_phase} "
                f"checks({check_counts})",
                flush=True,
            )
            if completed_batch_callback is not None:
                completed_batch_callback(phase, batch_index, accepted_chunk)
        if accepted_count != self.cfg.rows_per_phase:
            raise RuntimeError(
                f"Fast phase {phase}:{PICK_INSERT_PHASE_NAMES[phase]} exhausted "
                f"{self.attempt_counts[phase]} attempts with {accepted_count}/{self.cfg.rows_per_phase} accepted; "
                f"rejections={dict(self.rejection_counts[phase])}."
            )
        return {name: torch.cat([part[name] for part in accepted], dim=0) for name in RESET_DATASET_STATE_NAMES}

    @torch.inference_mode()
    def _generate_phase(
        self,
        phase: int,
        canonical_goal: dict[str, torch.Tensor],
        *,
        initial_accepted: list[dict[str, torch.Tensor]] | None = None,
        start_batch_index: int = 0,
        diagnostic_batch_evidence_callback: Any = None,
        completed_batch_callback: Any = None,
    ) -> dict[str, torch.Tensor]:
        accepted = [] if initial_accepted is None else list(initial_accepted)
        accepted_count = sum(int(part["phase"].shape[0]) for part in accepted)
        if not 0 <= start_batch_index <= self.cfg.max_batches_per_phase:
            raise ValueError("Generation phase resume batch index is outside the configured limit.")
        if not 0 <= accepted_count <= self.cfg.rows_per_phase:
            raise ValueError("Generation phase resume rows exceed the configured phase count.")
        starts_grasped = _PHASE_STARTS_GRASPED[phase]
        for batch_index in range(start_batch_index, self.cfg.max_batches_per_phase):
            if accepted_count >= self.cfg.rows_per_phase:
                break
            socket_pose, pickup_pose = self._sample_scene()
            goal_q, _goal_qd, goal_arm_target, goal_ik_valid = self._row_goal(canonical_goal, socket_pose)
            pregrasp_orientation_error = self._sample_phase_tcp_orientation_error(phase)
            pickup_arm_target, pickup_finger_target, pickup_valid = self._construct_pickup(
                socket_pose,
                pickup_pose,
                acquire=starts_grasped,
                active_mask=goal_ik_valid,
                pregrasp_orientation_error_xyzw=pregrasp_orientation_error,
            )
            arm_target, finger_target, realization_valid = self._realize_phase(
                phase,
                pickup_pose,
                goal_q,
                pickup_arm_target,
                pickup_finger_target=pickup_finger_target,
                active_mask=pickup_valid,
            )
            warm = self._capture_state(arm_target, finger_target)
            candidate, cold_evidence, cold_valid = self._cold_replay(
                warm,
                starts_grasped=starts_grasped,
                active_mask=realization_valid,
            )
            # Stored semantic geometry is inspected before contact buffers are
            # rebuilt.  The oracle performs the next history-aware restore;
            # queueing here would create two tickets before a real solve.
            self._restore_state(candidate, restore_pose_history=False)
            task_q, task_qd = self.env.read_task_state()
            semantic, initial_error, tcp_distance = self._phase_semantics(phase, task_q, goal_q)
            finite = task_state_is_finite_and_normalized(task_q, task_qd)
            reset_replay = cold_evidence["reset_replay"]
            reset_sample_count_exact = torch.full(
                (self.env.num_envs,),
                reset_replay["post_step_samples"] == PICK_INSERT_RESET_REPLAY_POST_STEP_SAMPLES,
                device=self.device,
                dtype=torch.bool,
            )
            reset_post_step_finite = (
                reset_replay["all_post_step_state_finite"]
                & reset_replay["all_post_step_task_state_finite_and_normalized"]
            )
            reset_robot_speed_bounded = (
                (reset_replay["stored_maximum_arm_joint_speed_rad_s"] <= self.cfg.maximum_row_arm_joint_speed_rad_s)
                & (
                    reset_replay["maximum_post_step_arm_joint_speed_rad_s"]
                    <= self.cfg.maximum_row_arm_joint_speed_rad_s
                )
                & (reset_replay["final_arm_joint_speed_rad_s"] <= self.cfg.maximum_row_arm_joint_speed_rad_s)
                & (reset_replay["stored_maximum_finger_joint_speed_m_s"] <= self.cfg.maximum_row_finger_joint_speed_m_s)
                & (
                    reset_replay["maximum_post_step_finger_joint_speed_m_s"]
                    <= self.cfg.maximum_row_finger_joint_speed_m_s
                )
                & (reset_replay["final_finger_joint_speed_m_s"] <= self.cfg.maximum_row_finger_joint_speed_m_s)
            )
            reset_cable_speed_bounded = (
                (reset_replay["stored_maximum_cable_speed_m_s"] <= self.cfg.maximum_row_cable_speed_m_s)
                & (reset_replay["maximum_post_step_cable_speed_m_s"] <= self.cfg.maximum_row_cable_speed_m_s)
                & (reset_replay["final_cable_speed_m_s"] <= self.cfg.maximum_row_cable_speed_m_s)
            )
            reset_body_excursion_bounded = (
                (reset_replay["maximum_body_excursion_m"] <= self.cfg.maximum_row_body_drift_m)
                & (reset_replay["maximum_plug_excursion_m"] <= self.cfg.maximum_row_plug_drift_m)
                & (reset_replay["maximum_socket_excursion_m"] <= self.cfg.maximum_socket_drift_m)
            )
            snapshot = {
                **{name: value.clone() for name, value in candidate.items()},
                "goal_task_body_pose": goal_q.clone(),
                "goal_arm_joint_target": goal_arm_target.clone(),
                "phase": torch.full((self.env.num_envs,), phase, device=self.device, dtype=torch.int64),
                "starts_grasped": torch.full(
                    (self.env.num_envs,), starts_grasped, device=self.device, dtype=torch.bool
                ),
                "difficulty": torch.full(
                    (self.env.num_envs,), _PHASE_DIFFICULTY[phase], device=self.device, dtype=torch.float32
                ),
                "initial_goal_error": initial_error.clone(),
                "initial_tcp_grasp_distance": tcp_distance.clone(),
                "progress_threshold": (0.15 * (initial_error if starts_grasped else tcp_distance)).clamp(2.0e-4, 0.02),
            }
            pre_oracle_valid = goal_ik_valid & pickup_valid & realization_valid & cold_valid & semantic & finite
            oracle_valid, oracle = self._oracle(
                candidate,
                goal_q,
                goal_arm_target,
                phase=phase,
                starts_grasped=starts_grasped,
                active_mask=pre_oracle_valid,
            )
            checks = {
                "goal_ik": goal_ik_valid,
                "pickup_construction": pickup_valid,
                "phase_realization": realization_valid,
                "cold_replay": cold_valid,
                "reset_zero_action_unclamped": cold_evidence["reset_zero_action_unclamped"],
                "reset_replay_sample_count_exact": reset_sample_count_exact,
                "reset_all_post_step_finite": reset_post_step_finite,
                "reset_all_post_step_collision_valid": reset_replay["all_post_step_collision_free"]
                & (reset_replay["maximum_invalid_contact_count"] == 0)
                & (not reset_replay["any_contact_overflow"]),
                "reset_all_post_step_contact_state_valid": reset_replay["all_post_step_expected_contact_state"],
                "reset_all_post_step_drives_disabled": reset_replay["all_post_step_drive_disabled"],
                "reset_vbd_pose_history_queued": reset_replay["vbd_pose_history_restore_queued"]
                & reset_replay["vbd_pose_history_pending_at_queue"]
                & reset_replay["vbd_previous_pose_queued"]
                & reset_replay["vbd_coupling_previous_pose_queued"],
                "reset_vbd_pose_history_applied_exactly_once": reset_replay["vbd_pose_history_applied_exactly_once"]
                & ~reset_replay["vbd_pose_history_failed"]
                & ~reset_replay["vbd_pose_history_superseded"]
                & ~reset_replay["vbd_pose_history_pending_after_first_solve"]
                & (reset_replay["vbd_pose_history_application_count_delta"] == 1)
                & (
                    reset_replay["vbd_pose_history_body_application_count_delta"]
                    == reset_replay["vbd_pose_history_expected_body_count"]
                ),
                "reset_vbd_pose_history_body_order_exact": torch.full(
                    (self.env.num_envs,),
                    bool(reset_replay["vbd_pose_history_body_order_exact"]),
                    device=self.device,
                    dtype=torch.bool,
                ),
                "reset_robot_speed_bounded": reset_robot_speed_bounded,
                "reset_cable_speed_bounded": reset_cable_speed_bounded,
                "reset_body_excursion_bounded": reset_body_excursion_bounded,
                "phase_semantics": semantic,
                "finite": finite,
                "drive_disabled": ~self._drive_enabled(),
                "oracle_physical_grasp": oracle["physical_grasp_acquired"],
                "oracle_full_recovery": oracle_valid,
                "oracle_exact_success_dwell": oracle["exact_success_dwell"],
                "oracle_all_samples_collision_free": oracle["exact_all_samples_collision_free"],
                "oracle_all_samples_bilateral_grasp": oracle["exact_all_samples_bilateral_grasp"],
                "oracle_all_samples_finite": oracle["exact_all_samples_finite"],
            }
            valid, reported_checks = _validity_and_recovery_failure_checks(
                checks,
                oracle["recovery_lane_failure_masks"],
            )
            self._record_rejections(phase, reported_checks, valid)
            self.attempt_counts[phase] += self.env.num_envs
            remaining = self.cfg.rows_per_phase - accepted_count
            chosen = torch.where(valid)[0][:remaining]
            accepted_chunk: dict[str, torch.Tensor] | None = None
            if chosen.numel():
                accepted_chunk = {name: value[chosen].detach().clone() for name, value in snapshot.items()}
                accepted.append(accepted_chunk)
                for index in chosen.tolist():
                    self.accepted_oracle_metrics[phase].append(
                        {
                            "goal_error": float(oracle["goal_error"][index]),
                            "socket_drift_m": float(cold_evidence["maximum_socket_drift_m"][index]),
                            "body_drift_m": float(cold_evidence["maximum_body_drift_m"][index]),
                            "cable_speed_m_s": float(cold_evidence["maximum_cable_speed_m_s"][index]),
                            "reset_zero_action_unclamped": bool(cold_evidence["reset_zero_action_unclamped"][index]),
                            "maximum_reset_arm_target_clamp_delta_rad": float(
                                cold_evidence["maximum_arm_target_clamp_delta_rad"][index]
                            ),
                            "reset_replay_post_step_samples": int(reset_replay["post_step_samples"]),
                            "reset_replay_all_post_step_collision_free": bool(
                                reset_replay["all_post_step_collision_free"][index]
                            ),
                            "reset_replay_all_post_step_expected_contact_state": bool(
                                reset_replay["all_post_step_expected_contact_state"][index]
                            ),
                            "reset_replay_maximum_arm_joint_speed_rad_s": float(
                                reset_replay["maximum_post_step_arm_joint_speed_rad_s"][index]
                            ),
                            "reset_replay_maximum_finger_joint_speed_m_s": float(
                                reset_replay["maximum_post_step_finger_joint_speed_m_s"][index]
                            ),
                            "physical_grasp_acquired": bool(oracle["physical_grasp_acquired"][index]),
                            "exact_success_dwell": bool(oracle["exact_success_dwell"][index]),
                            "exact_all_samples_collision_free": bool(oracle["exact_all_samples_collision_free"][index]),
                            "exact_all_samples_bilateral_grasp": bool(
                                oracle["exact_all_samples_bilateral_grasp"][index]
                            ),
                            "exact_all_samples_finite": bool(oracle["exact_all_samples_finite"][index]),
                            "exact_maximum_body_excursion_m": float(oracle["exact_maximum_body_excursion"][index]),
                            "exact_maximum_cable_linear_speed_m_s": float(
                                oracle["exact_maximum_cable_linear_speed"][index]
                            ),
                        }
                    )
                accepted_count += int(chosen.numel())
            check_counts = ", ".join(f"{name}={int(value.sum())}" for name, value in reported_checks.items())
            print(
                f"[PICK-INSERT RESET] phase={phase}:{PICK_INSERT_PHASE_NAMES[phase]} "
                f"batch={batch_index + 1} accepted={accepted_count}/{self.cfg.rows_per_phase} "
                f"checks({check_counts})",
                flush=True,
            )
            if diagnostic_batch_evidence_callback is not None:
                diagnostic_batch_evidence_callback(
                    phase,
                    batch_index,
                    oracle["recovery_diagnostic_evidence"],
                )
            if completed_batch_callback is not None:
                completed_batch_callback(phase, batch_index, accepted_chunk)
        if accepted_count != self.cfg.rows_per_phase:
            raise RuntimeError(
                f"Phase {phase}:{PICK_INSERT_PHASE_NAMES[phase]} exhausted "
                f"{self.attempt_counts[phase]} attempts with {accepted_count}/{self.cfg.rows_per_phase} accepted; "
                f"rejections={dict(self.rejection_counts[phase])}."
            )
        return {name: torch.cat([part[name] for part in accepted], dim=0) for name in RESET_DATASET_STATE_NAMES}

    @torch.inference_mode()
    def generate(
        self,
        canonical_goal_certificate: Mapping[str, Any] | None = None,
        *,
        generation_checkpoint: _GenerationCheckpoint | None = None,
    ) -> dict[str, Any]:
        """Generate reset rows from an inline or separately certified canonical goal."""
        certificate_embedding: dict[str, Any] | None = None
        if canonical_goal_certificate is None:
            if generation_checkpoint is not None:
                raise RuntimeError("Generation checkpointing requires a canonical-goal certificate input.")
            row_rng_state = self.random.get_state().detach().cpu().clone().contiguous()
            canonical_goal, goal_evidence = self.derive_goal()
            if not torch.equal(self.random.get_state().detach().cpu(), row_rng_state):
                raise RuntimeError("Canonical-goal derivation consumed the dedicated reset-row RNG stream.")
        else:
            legacy_shape_required = self.cfg.generation_mode == _GENERATION_MODE_PHYSICAL_ORACLE and (
                self.cfg.quick
                or self.cfg.rows_per_phase != _CANONICAL_ROWS_PER_PHASE
                or self.env.num_envs != _CANONICAL_BATCH_SIZE
            )
            if legacy_shape_required:
                raise RuntimeError("Canonical-goal certificate input is restricted to canonical batch-24 generation.")
            if self._ik_solve_call_count != 0:
                raise RuntimeError(
                    "Canonical-goal certificate input requires a freshly constructed, unadvanced row IK stream."
                )
            certificate = self.validate_goal_certificate(canonical_goal_certificate)
            canonical_goal = {
                name: certificate["goal_state"][name].detach().cpu().clone().contiguous()
                for name in RESET_DATASET_GOAL_STATE_NAMES
            }
            goal_evidence = certificate["metadata"]["production_evidence"]
            row_rng_state = certificate["row_rng_state"].detach().cpu().clone().contiguous()
            certificate_embedding = _canonical_goal_certificate_embedding(certificate)
        generate_phase = (
            self._generate_phase_fast if self.cfg.generation_mode == _GENERATION_MODE_FAST_IK else self._generate_phase
        )
        if generation_checkpoint is None:
            try:
                self.random.set_state(row_rng_state)
            except RuntimeError as exc:
                raise ValueError("Canonical-goal certificate row RNG state is incompatible with this device.") from exc
            phase_parts = [generate_phase(phase, canonical_goal) for phase in PICK_INSERT_RESET_PHASE_IDS]
        else:
            assert canonical_goal_certificate is not None
            generation_checkpoint.restore_generator(self, canonical_goal_certificate)
            phase_parts = []
            for phase in PICK_INSERT_RESET_PHASE_IDS:
                phase_parts.append(
                    generate_phase(
                        phase,
                        canonical_goal,
                        initial_accepted=generation_checkpoint.phase_chunks(phase, device=self.device),
                        start_batch_index=generation_checkpoint.next_batch_index(phase),
                        completed_batch_callback=lambda completed_phase, batch_index, chunk: (
                            generation_checkpoint.commit_batch(self, completed_phase, batch_index, chunk)
                        ),
                    )
                )
            generation_checkpoint.mark_rows_complete(self)
        states = {name: torch.cat([part[name] for part in phase_parts], dim=0) for name in RESET_DATASET_STATE_NAMES}
        if generation_checkpoint is None:
            permutation = torch.randperm(len(states["phase"]), device=self.device, generator=self.random)
        else:
            permutation = generation_checkpoint.final_permutation(self, len(states["phase"]))
        cpu_states = {name: states[name][permutation].detach().cpu().contiguous() for name in RESET_DATASET_STATE_NAMES}
        contract = pick_insert_reset_dataset_task_contract(self.env.cfg)
        if self.cfg.generation_mode == _GENERATION_MODE_FAST_IK:
            accepted_metrics_name = "accepted_fast_reset_metrics"
            accepted_metrics = _bind_fast_accepted_metrics_to_final_rows(
                self.accepted_oracle_metrics,
                cpu_states,
                permutation,
            )
            initial_state_policy = {
                "generation_mode": _GENERATION_MODE_FAST_IK,
                "construction": "direct-rigid-task-transform-plus-batched-ik",
                "whole_cable_generated_by_coherent_rigid_transform": True,
                "independent_cable_body_noise": False,
                "construction_drive_disabled_in_every_snapshot": True,
                "fast_reset_policy": FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY,
                "reset_bank_profile": _fast_reset_bank_profile_contract(self.cfg),
                "accepted_row_binding": PICK_INSERT_FAST_RESET_ROW_BINDING_CONTRACT,
                "phase_0_accepted_band_proportions": PICK_INSERT_FAST_RESET_PHASE_0_BAND_ACCEPTANCE_CONTRACT,
                "phase_0_reverse_curriculum_sampling": (pick_insert_phase_0_reverse_curriculum_sampling_contract()),
                "phase_0_reverse_curriculum_evidence": _phase_0_reverse_curriculum_evidence(
                    self.accepted_oracle_metrics[0]
                ),
            }
        else:
            accepted_metrics_name = "accepted_oracle_metrics"
            accepted_metrics = {
                str(phase): self.accepted_oracle_metrics[phase] for phase in PICK_INSERT_RESET_PHASE_IDS
            }
            initial_state_policy = {
                "generation_mode": _GENERATION_MODE_PHYSICAL_ORACLE,
                "whole_cable_generated_by_coupled_motion": True,
                "independent_cable_body_noise": False,
                "construction_drive_disabled_in_every_snapshot": True,
                "ungrasped_oracle_requires_physical_bilateral_acquisition": True,
                "every_row_cold_replayed": True,
                "every_row_robot_only_recovered": True,
                "grasped_transport_schedule": _grasped_transport_schedule_contract(self.cfg),
                "scripted_recovery": PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY,
                **_pickup_construction_sequence_contract(),
            }
        initial_state_policy["phase_4_pregrasp_orientation_sampling"] = _phase_4_pregrasp_orientation_sampling_contract(
            self.cfg
        )
        payload: dict[str, Any] = {
            "format": FRANKA_RJ45_PICK_INSERT_RESET_DATASET_FORMAT,
            "schema_version": FRANKA_RJ45_PICK_INSERT_RESET_DATASET_SCHEMA_VERSION,
            "metadata": {
                "generator": Path(__file__).name,
                "generator_cfg": asdict(self.cfg),
                "seed": self.cfg.seed,
                "quick": self.cfg.quick,
                "task_contract": contract,
                "phase_names": tuple(PICK_INSERT_PHASE_NAMES),
                "phase_counts": list(phase_counts(self.cfg)),
                "phase_starts_grasped": list(_PHASE_STARTS_GRASPED),
                "attempt_counts": list(self.attempt_counts),
                "rejection_counts": {
                    str(phase): dict(sorted(self.rejection_counts[phase].items()))
                    for phase in PICK_INSERT_RESET_PHASE_IDS
                },
                accepted_metrics_name: accepted_metrics,
                "canonical_goal_evidence": goal_evidence,
                "canonical_goal_certificate": certificate_embedding,
                "goal_policy": {
                    "one_central_canonical_goal": True,
                    "per_row_goal_is_rigid_socket_transform": True,
                    "socket_pose_randomized_within_task_cfg": True,
                    "socket_cfg_position_is_assembly_origin": True,
                    "reference_socket_body_pose": self.reference_socket_body_pose.detach().cpu().tolist(),
                    "randomized_goal_socket_body_z_range_m": [
                        float(cpu_states["goal_task_body_pose"][:, self.socket_index, 2].min()),
                        float(cpu_states["goal_task_body_pose"][:, self.socket_index, 2].max()),
                    ],
                },
                "initial_state_policy": initial_state_policy,
                "physics_versions": package_versions(),
            },
            "states": cpu_states,
            "goal_state": canonical_goal,
        }
        payload["content_sha256"] = reset_dataset_content_digest(payload)
        reset_dataset_validate_runtime(payload, expected_task_contract=contract)
        if generation_checkpoint is not None:
            generation_checkpoint.mark_artifact_ready(self, payload, permutation)
        return payload


def _generation_checkpoint_metadata(
    generator: PickInsertResetDatasetGenerator,
    certificate: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the path-free live contract required by a generation checkpoint."""
    validation = generator._canonical_goal_certificate_validation_kwargs()
    row_rng_state = certificate["row_rng_state"].detach().cpu().contiguous()
    return _checkpoint_plain_value(
        {
            "generator": Path(__file__).name,
            "artifact_contract": {
                "format": FRANKA_RJ45_PICK_INSERT_RESET_DATASET_FORMAT,
                "schema_version": FRANKA_RJ45_PICK_INSERT_RESET_DATASET_SCHEMA_VERSION,
                "rows_per_phase": generator.cfg.rows_per_phase,
                "batch_size": generator.cfg.batch_size,
                "max_batches_per_phase": generator.cfg.max_batches_per_phase,
                "phase_ids": tuple(PICK_INSERT_RESET_PHASE_IDS),
                "phase_names": tuple(PICK_INSERT_PHASE_NAMES),
                "phase_starts_grasped": _PHASE_STARTS_GRASPED,
                "phase_4_pregrasp_orientation_sampling": (
                    _phase_4_pregrasp_orientation_sampling_contract(generator.cfg)
                ),
                "state_names": tuple(RESET_DATASET_STATE_NAMES),
                "goal_state_names": tuple(RESET_DATASET_GOAL_STATE_NAMES),
                "final_row_order": "one-torch-randperm-after-rows-complete",
            },
            "generator_cfg": asdict(generator.cfg),
            "generation_contract": validation["expected_generation_contract"],
            "task_contract": validation["expected_task_contract"],
            "physical_contract": validation["expected_physical_contract"],
            "package_versions": validation["expected_versions"],
            "source_sha256": validation["expected_source_sha256"],
            "asset_closure": franka_rj45_asset_contract(),
            "canonical_goal_certificate": {
                "format": certificate["format"],
                "schema_version": certificate["schema_version"],
                "content_sha256": certificate["content_sha256"],
                "certifier_env_count": certificate["metadata"]["certifier_env_count"],
                "goal_state_sha256": reset_dataset_digest(certificate["goal_state"]),
                "row_rng_state_sha256": reset_dataset_digest(row_rng_state),
            },
            "initial_row_rng_contract": {
                "owner": "PickInsertResetDatasetGenerator.random",
                "seed": generator.cfg.seed,
                "state_sha256": reset_dataset_digest(row_rng_state),
                "state_numel": int(row_rng_state.numel()),
                "row_ik_stream": validation["expected_generation_contract"]["row_ik_stream"],
            },
        }
    )


def _initial_generation_checkpoint_document(
    *,
    metadata: Mapping[str, Any],
    certificate: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the first durable boundary after certified-goal loading."""
    document: dict[str, Any] = {
        "format": _GENERATION_CHECKPOINT_FORMAT,
        "schema_version": _GENERATION_CHECKPOINT_SCHEMA_VERSION,
        "metadata": _checkpoint_plain_value(metadata),
        "progress": {
            "status": "goal-ready",
            "created_utc": datetime.now(UTC).isoformat(),
            "canonical_goal": {
                name: _encode_checkpoint_tensor(certificate["goal_state"][name])
                for name in RESET_DATASET_GOAL_STATE_NAMES
            },
            "canonical_goal_evidence": _checkpoint_plain_value(certificate["metadata"]["production_evidence"]),
            "completed_batches": [],
            "accepted_chunks": [],
            "attempt_counts": [0 for _ in PICK_INSERT_RESET_PHASE_IDS],
            "rejection_counts": {str(phase): {} for phase in PICK_INSERT_RESET_PHASE_IDS},
            "accepted_oracle_metrics": {str(phase): [] for phase in PICK_INSERT_RESET_PHASE_IDS},
            "logical_ik_solve_call_count": 0,
            "row_rng_state": _encode_checkpoint_tensor(certificate["row_rng_state"]),
            "final_artifact": None,
        },
    }
    document["content_sha256"] = _generation_checkpoint_content_digest(document)
    return document


class _GenerationCheckpoint:
    """Own one validated sequence of durable generation batch commits."""

    def __init__(
        self,
        *,
        path: Path,
        expected_metadata: Mapping[str, Any],
        validated: _ValidatedGenerationCheckpoint,
    ) -> None:
        self.path = path
        self.expected_metadata = _checkpoint_plain_value(expected_metadata)
        self._install(validated)
        self._logical_ik_base = int(self.document["progress"]["logical_ik_solve_call_count"])

    @classmethod
    def open(
        cls,
        generator: PickInsertResetDatasetGenerator,
        certificate: Mapping[str, Any],
        *,
        path: Path,
        resuming: bool,
    ) -> _GenerationCheckpoint:
        """Create or resume one checkpoint under the caller-held lock."""
        expected_metadata = _generation_checkpoint_metadata(generator, certificate)
        if resuming:
            validated = _load_generation_checkpoint(path, expected_metadata=expected_metadata)
        else:
            if path.exists():
                raise ValueError(f"Fresh generation checkpoint already exists: {path}")
            document = _initial_generation_checkpoint_document(metadata=expected_metadata, certificate=certificate)
            stored = _write_generation_checkpoint_atomic(document, path, expected_metadata=expected_metadata)
            validated = _validate_generation_checkpoint(stored, expected_metadata=expected_metadata)
        checkpoint = cls(path=path, expected_metadata=expected_metadata, validated=validated)
        expected_evidence = _checkpoint_plain_value(certificate["metadata"]["production_evidence"])
        if reset_dataset_digest(checkpoint.document["progress"]["canonical_goal_evidence"]) != reset_dataset_digest(
            expected_evidence
        ):
            raise ValueError("Generation checkpoint canonical-goal evidence does not match its certificate.")
        if reset_dataset_digest(checkpoint.canonical_goal) != reset_dataset_digest(certificate["goal_state"]):
            raise ValueError("Generation checkpoint canonical goal does not match its certificate.")
        return checkpoint

    @property
    def status(self) -> str:
        return str(self.document["progress"]["status"])

    def _install(self, validated: _ValidatedGenerationCheckpoint) -> None:
        self.document = validated.document
        self.canonical_goal = validated.canonical_goal
        self.accepted_chunks = validated.accepted_chunks
        self.row_rng_state = validated.row_rng_state

    def _persist(self) -> None:
        stored = _write_generation_checkpoint_atomic(
            self.document,
            self.path,
            expected_metadata=self.expected_metadata,
        )
        self._install(_validate_generation_checkpoint(stored, expected_metadata=self.expected_metadata))

    def _require_live_contract(
        self,
        generator: PickInsertResetDatasetGenerator,
        certificate: Mapping[str, Any],
    ) -> None:
        live = _generation_checkpoint_metadata(generator, certificate)
        if reset_dataset_digest(live) != reset_dataset_digest(self.expected_metadata):
            raise RuntimeError("Generation checkpoint contract changed during generation; no artifact was published.")

    def restore_generator(
        self,
        generator: PickInsertResetDatasetGenerator,
        certificate: Mapping[str, Any],
    ) -> None:
        """Restore only CPU progress and RNG; the sampler-free IK owner stays fresh."""
        self._require_live_contract(generator, certificate)
        if generator._ik_solve_call_count != 0:
            raise RuntimeError("Generation checkpoint resume requires a fresh, unadvanced row IK owner.")
        try:
            generator.random.set_state(self.row_rng_state)
        except RuntimeError as exc:
            raise ValueError("Generation checkpoint row RNG state is incompatible with this device.") from exc
        progress = self.document["progress"]
        generator.attempt_counts = list(progress["attempt_counts"])
        generator.rejection_counts = {
            phase: defaultdict(int, progress["rejection_counts"][str(phase)]) for phase in PICK_INSERT_RESET_PHASE_IDS
        }
        generator.accepted_oracle_metrics = {
            phase: list(progress["accepted_oracle_metrics"][str(phase)]) for phase in PICK_INSERT_RESET_PHASE_IDS
        }

    def phase_chunks(self, phase: int, *, device: torch.device) -> list[dict[str, torch.Tensor]]:
        """Return accepted prefix chunks for one phase on the live device."""
        return [
            {name: value.to(device=device) for name, value in chunk["states"].items()}
            for chunk in self.accepted_chunks
            if chunk["phase"] == phase
        ]

    def next_batch_index(self, phase: int) -> int:
        """Return the first uncommitted phase-local batch index."""
        return sum(record["phase"] == phase for record in self.document["progress"]["completed_batches"])

    def _logical_ik_count(self, generator: PickInsertResetDatasetGenerator) -> int:
        return self._logical_ik_base + int(generator._ik_solve_call_count)

    def _copy_generator_evidence(self, generator: PickInsertResetDatasetGenerator) -> None:
        progress = self.document["progress"]
        progress["attempt_counts"] = list(generator.attempt_counts)
        progress["rejection_counts"] = {
            str(phase): dict(sorted(generator.rejection_counts[phase].items())) for phase in PICK_INSERT_RESET_PHASE_IDS
        }
        progress["accepted_oracle_metrics"] = {
            str(phase): _checkpoint_plain_value(generator.accepted_oracle_metrics[phase])
            for phase in PICK_INSERT_RESET_PHASE_IDS
        }
        progress["logical_ik_solve_call_count"] = self._logical_ik_count(generator)
        progress["row_rng_state"] = _encode_checkpoint_tensor(generator.random.get_state())

    def commit_batch(
        self,
        generator: PickInsertResetDatasetGenerator,
        phase: int,
        batch_index: int,
        accepted_chunk: Mapping[str, torch.Tensor] | None,
    ) -> None:
        """Durably commit one and only one fully completed production batch."""
        if self.status not in {"goal-ready", "generating"}:
            raise RuntimeError(f"Cannot append a batch to generation checkpoint status {self.status!r}.")
        progress = self.document["progress"]
        ordinal = len(progress["completed_batches"])
        if batch_index != self.next_batch_index(phase):
            raise RuntimeError("Generation checkpoint callback did not receive the next phase-local batch.")
        prior_phase_rows = sum(
            len(record["row_ids"]) for record in progress["completed_batches"] if record["phase"] == phase
        )
        accepted_count = 0 if accepted_chunk is None else int(accepted_chunk["phase"].shape[0])
        rows_per_phase = int(self.expected_metadata["artifact_contract"]["rows_per_phase"])
        row_id_start = phase * rows_per_phase + prior_phase_rows
        row_ids = list(range(row_id_start, row_id_start + accepted_count))
        progress["completed_batches"].append(
            {
                "ordinal": ordinal,
                "phase": phase,
                "phase_batch_index": batch_index,
                "row_ids": row_ids,
            }
        )
        if accepted_chunk is not None:
            if set(accepted_chunk) != set(RESET_DATASET_STATE_NAMES):
                raise RuntimeError("Generation checkpoint callback received an incomplete accepted row chunk.")
            progress["accepted_chunks"].append(
                {
                    "ordinal": ordinal,
                    "phase": phase,
                    "row_ids": row_ids,
                    "states": {
                        name: _encode_checkpoint_tensor(accepted_chunk[name]) for name in RESET_DATASET_STATE_NAMES
                    },
                }
            )
        progress["status"] = "generating"
        self._copy_generator_evidence(generator)
        self._persist()

    def mark_rows_complete(self, generator: PickInsertResetDatasetGenerator) -> None:
        """Commit the all-rows boundary before drawing the final permutation."""
        if self.status in {"artifact-ready", "stable-published"}:
            return
        rows_per_phase = int(self.expected_metadata["artifact_contract"]["rows_per_phase"])
        expected_rows = rows_per_phase * len(PICK_INSERT_RESET_PHASE_IDS)
        accepted_rows = sum(len(chunk["row_ids"]) for chunk in self.accepted_chunks)
        if accepted_rows != expected_rows:
            raise RuntimeError(f"Cannot mark {accepted_rows}/{expected_rows} checkpoint rows complete.")
        self.document["progress"]["status"] = "rows-complete"
        self.document["progress"]["final_artifact"] = None
        self._copy_generator_evidence(generator)
        self._persist()

    def final_permutation(
        self,
        generator: PickInsertResetDatasetGenerator,
        row_count: int,
    ) -> torch.Tensor:
        """Draw the final permutation once, or reuse the artifact-ready record."""
        final_artifact = self.document["progress"]["final_artifact"]
        if final_artifact is not None:
            return torch.tensor(final_artifact["permutation"], device=generator.device, dtype=torch.int64)
        if self.status != "rows-complete":
            raise RuntimeError("Final row permutation requires a rows-complete checkpoint boundary.")
        return torch.randperm(row_count, device=generator.device, generator=generator.random)

    def mark_artifact_ready(
        self,
        generator: PickInsertResetDatasetGenerator,
        payload: Mapping[str, Any],
        permutation: torch.Tensor,
    ) -> None:
        """Bind the validated final artifact digest and its sole permutation."""
        final_record = {
            "content_sha256": payload["content_sha256"],
            "row_count": int(permutation.numel()),
            "permutation": permutation.detach().cpu().tolist(),
        }
        existing = self.document["progress"]["final_artifact"]
        if existing is not None:
            if existing != final_record:
                raise RuntimeError("Resumed artifact payload differs from the checkpointed final artifact.")
            return
        if self.status != "rows-complete":
            raise RuntimeError("Only a rows-complete checkpoint can become artifact-ready.")
        self.document["progress"]["status"] = "artifact-ready"
        self.document["progress"]["final_artifact"] = final_record
        self._copy_generator_evidence(generator)
        self._persist()

    def mark_stable_published(
        self,
        generator: PickInsertResetDatasetGenerator,
        certificate: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> None:
        """Durably record that the exact final artifact is on disk."""
        self._require_live_contract(generator, certificate)
        final_artifact = self.document["progress"]["final_artifact"]
        if not isinstance(final_artifact, Mapping) or final_artifact["content_sha256"] != payload["content_sha256"]:
            raise RuntimeError("Published reset dataset does not match the checkpoint final artifact.")
        if self.status == "stable-published":
            return
        if self.status != "artifact-ready":
            raise RuntimeError("Only an artifact-ready checkpoint can become stable-published.")
        self.document["progress"]["status"] = "stable-published"
        self._persist()


def _load_matching_published_reset_dataset(
    output: Path,
    *,
    expected_content_sha256: str,
    expected_task_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Load and prove one already-published idempotent checkpoint output."""
    try:
        payload = torch.load(output, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not safely load checkpointed reset dataset {output}: {exc}") from exc
    reset_dataset_validate_runtime(
        payload,
        expected_content_sha256=expected_content_sha256,
        expected_task_contract=expected_task_contract,
    )
    return payload


def _generate_and_save_reset_dataset_artifact(
    generator: PickInsertResetDatasetGenerator,
    *,
    output: Path,
    certificate_input: Path | None,
    pre_environment_source_sha256: Mapping[str, str] | None = None,
    checkpoint_path: Path | None = None,
    resuming_checkpoint: bool = False,
) -> dict[str, Any]:
    """Generate and atomically save rows, guarding certificate-input provenance."""
    certificate: dict[str, Any] | None = None
    validation_snapshot: dict[str, Any] | None = None
    if certificate_input is not None:
        if pre_environment_source_sha256 is None:
            raise RuntimeError("Certificate-input generation requires a pre-environment source snapshot.")
        validation_snapshot = generator._canonical_goal_certificate_validation_kwargs()
        _require_unchanged_canonical_goal_validation_snapshot(
            {"expected_source_sha256": pre_environment_source_sha256},
            {"expected_source_sha256": validation_snapshot["expected_source_sha256"]},
            operation="environment construction",
        )
        load_goal_certificate = getattr(generator, "load_goal_certificate", None)
        if callable(load_goal_certificate):
            certificate = load_goal_certificate(certificate_input)
        else:
            certificate = _load_canonical_goal_certificate(certificate_input, **validation_snapshot)

    generation_checkpoint: _GenerationCheckpoint | None = None
    if checkpoint_path is not None:
        if certificate is None:
            raise RuntimeError("Generation checkpointing requires a validated canonical-goal certificate.")
        generation_checkpoint = _GenerationCheckpoint.open(
            generator,
            certificate,
            path=checkpoint_path,
            resuming=resuming_checkpoint,
        )
    if generation_checkpoint is None:
        payload = generator.generate(certificate)
    else:
        payload = generator.generate(certificate, generation_checkpoint=generation_checkpoint)
    if certificate is not None:
        assert validation_snapshot is not None
        live_validation = generator._canonical_goal_certificate_validation_kwargs()
        _require_unchanged_canonical_goal_validation_snapshot(
            validation_snapshot,
            live_validation,
            operation="certificate-input reset-row generation",
        )
        validate_goal_certificate = getattr(generator, "validate_goal_certificate", None)
        if callable(validate_goal_certificate):
            validate_goal_certificate(certificate)
        else:
            _validate_canonical_goal_certificate(certificate, **live_validation)
    output = output.expanduser().resolve()
    if generation_checkpoint is None:
        save_torch_atomic(payload, output)
    else:
        final_artifact = generation_checkpoint.document["progress"]["final_artifact"]
        assert isinstance(final_artifact, Mapping)
        if generation_checkpoint.status == "stable-published" and not output.is_file():
            raise RuntimeError("Stable-published generation checkpoint output is missing; checkpoint was retained.")
        if output.exists():
            _load_matching_published_reset_dataset(
                output,
                expected_content_sha256=final_artifact["content_sha256"],
                expected_task_contract=live_validation["expected_task_contract"],
            )
        else:
            save_torch_atomic(payload, output)
        generation_checkpoint.mark_stable_published(generator, certificate, payload)
    return payload


def _restore_certificate_backed_diagnostic_inputs(
    generator: PickInsertResetDatasetGenerator,
    certificate: Mapping[str, Any],
    *,
    diagnostic_name: str,
) -> dict[str, torch.Tensor]:
    """Restore the certified row stream before any diagnostic IK solve or scene sample."""
    if generator._ik_solve_call_count != 0:
        raise RuntimeError(
            f"Certificate-backed {diagnostic_name} diagnostic requires a fresh, unadvanced row IK stream."
        )
    row_rng_state = certificate["row_rng_state"].detach().cpu().clone().contiguous()
    try:
        generator.random.set_state(row_rng_state)
    except RuntimeError as exc:
        raise ValueError("Canonical-goal certificate row RNG state is incompatible with this device.") from exc
    return {
        name: certificate["goal_state"][name].detach().cpu().clone().contiguous()
        for name in RESET_DATASET_GOAL_STATE_NAMES
    }


def _run_save_disabled_diagnostic(
    generator: PickInsertResetDatasetGenerator,
    args: argparse.Namespace,
    canonical_goal_certificate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Route one no-artifact diagnostic without entering dataset generation."""
    diagnostic_recovery_phase = getattr(args, "diagnostic_recovery_phase", None)
    if diagnostic_recovery_phase is not None:
        if canonical_goal_certificate is None:
            raise RuntimeError("Recovery diagnostic mode requires a validated canonical goal.")
        canonical_goal = _restore_certificate_backed_diagnostic_inputs(
            generator,
            canonical_goal_certificate,
            diagnostic_name="recovery",
        )
        return generator.run_diagnostic_recovery_phase_once(diagnostic_recovery_phase, canonical_goal)
    if bool(getattr(args, "diagnostic_phase0_transport_only", False)):
        if canonical_goal_certificate is None:
            raise RuntimeError("Phase-0 transport-only diagnostic mode requires a validated canonical goal.")
        canonical_goal = _restore_certificate_backed_diagnostic_inputs(
            generator,
            canonical_goal_certificate,
            diagnostic_name="phase-0",
        )
        return generator.run_diagnostic_phase0_transport_once(canonical_goal)
    if bool(getattr(args, "diagnostic_pickup_only", False)):
        evidence = generator.run_diagnostic_pickup_once()
        print(f"[PICK-INSERT PICKUP-ONLY COMPLETE] {json.dumps(evidence, sort_keys=True)}", flush=True)
        return evidence
    _, evidence = generator.derive_goal()
    print(f"[PICK-INSERT GOAL-ONLY COMPLETE] {evidence}", flush=True)
    return evidence


def _finalize_certificate_backed_diagnostic(
    generator: PickInsertResetDatasetGenerator,
    certificate: Mapping[str, Any],
    validation_snapshot: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    operation: str,
    completion_label: str,
) -> None:
    """Revalidate live certificate inputs before publishing diagnostic completion."""
    live_validation = generator._canonical_goal_certificate_validation_kwargs()
    _require_unchanged_canonical_goal_validation_snapshot(
        validation_snapshot,
        live_validation,
        operation=operation,
    )
    _validate_canonical_goal_certificate(certificate, **live_validation)
    print(f"[PICK-INSERT {completion_label} COMPLETE] {json.dumps(evidence, sort_keys=True)}", flush=True)


def _finalize_certificate_backed_phase0_diagnostic(
    generator: PickInsertResetDatasetGenerator,
    certificate: Mapping[str, Any],
    validation_snapshot: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> None:
    """Preserve the phase-0 diagnostic completion contract."""
    _finalize_certificate_backed_diagnostic(
        generator,
        certificate,
        validation_snapshot,
        evidence,
        operation="certificate-backed phase-0 diagnostic",
        completion_label="PHASE0-TRANSPORT-ONLY",
    )


def _finalize_certificate_backed_recovery_diagnostic(
    generator: PickInsertResetDatasetGenerator,
    certificate: Mapping[str, Any],
    validation_snapshot: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> None:
    """Publish recovery evidence only after final live-source and certificate proof."""
    phase = int(evidence["phase"])
    _finalize_certificate_backed_diagnostic(
        generator,
        certificate,
        validation_snapshot,
        evidence,
        operation=f"certificate-backed phase-{phase} recovery diagnostic",
        completion_label=f"RECOVERY-PHASE{phase}",
    )


def _configure_generation_reset_dataset_shape(env_cfg: FrankaRJ45PickInsertEnvCfg, rows_per_phase: int) -> None:
    """Bind generated row cardinality and its diversity gate into the task contract."""
    env_cfg.reset_dataset_rows_per_phase = rows_per_phase
    if rows_per_phase == _CANONICAL_ROWS_PER_PHASE:
        minimum_unique = 90
    elif rows_per_phase == _REFERENCE_FAST_ROWS_PER_PHASE:
        minimum_unique = 3_000
    else:
        minimum_unique = max(1, min(rows_per_phase, math.ceil(0.90 * rows_per_phase)))
    env_cfg.reset_dataset_min_unique_full_pick_rows = minimum_unique


def _execute_parsed_invocation(
    args: argparse.Namespace,
    *,
    reset_dataset_saving_disabled: bool,
    certificate_input_source_snapshot: Mapping[str, str] | None,
    checkpoint_path: Path | None,
    resuming_checkpoint: bool,
) -> None:
    """Execute one parsed invocation while the optional checkpoint lock is held."""
    generator_cfg = GeneratorCfg(
        generation_mode=getattr(args, "generation_mode", _GENERATION_MODE_PHYSICAL_ORACLE),
        rows_per_phase=1 if args.quick else args.rows_per_phase,
        batch_size=1 if args.quick else args.batch_size,
        seed=args.seed,
        max_batches_per_phase=min(12, args.max_batches_per_phase) if args.quick else args.max_batches_per_phase,
        quick=args.quick,
        diagnostic_reset_abcd=args.diagnostic_reset_abcd,
        diagnostic_reset_e_only=args.diagnostic_reset_e,
        diagnostic_p_relax_reseat=args.diagnostic_p_relax_reseat,
        diagnostic_zero_finger_close_target=args.diagnostic_zero_finger_close_target,
        diagnostic_forward_grasp_offset=args.diagnostic_forward_grasp_offset,
        diagnostic_effective_grasp_friction_three=args.diagnostic_effective_grasp_friction_three,
        finger_closed_target=PICK_INSERT_CLOSED_FINGER_POSITION,
    )
    env_cfg = FrankaRJ45PickInsertEnvCfg()
    _configure_generation_reset_dataset_shape(env_cfg, generator_cfg.rows_per_phase)
    if args.diagnostic_forward_grasp_offset:
        env_cfg.plug_grasp_offset = (0.0, -0.020, 0.010)
    env_cfg.scene.num_envs = generator_cfg.batch_size
    env_cfg.sim.device = args.device
    env_cfg.seed = generator_cfg.seed
    env_cfg.validate_config()

    payload: dict[str, Any] | None = None
    certificate_payload: dict[str, Any] | None = None
    with launch_simulation(env_cfg, args):
        try:
            grasp_proxy_friction = (
                PICK_INSERT_GRASP_PROXY_FRICTION if args.diagnostic_effective_grasp_friction_three else None
            )
            env = RJ45PickInsertResetToolEnv(env_cfg, grasp_proxy_friction=grasp_proxy_friction)
        except Exception as exc:
            raise RuntimeError(
                "Could not construct the real coupled Franka RJ45 pick-insert tool environment; "
                "no reset artifact was emitted."
            ) from exc
        try:
            generator = PickInsertResetDatasetGenerator(env, generator_cfg)
            if args.canonical_goal_certificate_output is not None:
                certificate_payload = generator.derive_goal_certificate()
                save_torch_atomic(certificate_payload, args.canonical_goal_certificate_output)
            elif reset_dataset_saving_disabled:
                diagnostic_certificate: dict[str, Any] | None = None
                diagnostic_validation_snapshot: dict[str, Any] | None = None
                diagnostic_recovery_phase = getattr(args, "diagnostic_recovery_phase", None)
                certificate_backed_diagnostic = (
                    args.diagnostic_phase0_transport_only or diagnostic_recovery_phase is not None
                )
                if certificate_backed_diagnostic:
                    if certificate_input_source_snapshot is None:
                        raise RuntimeError("Certificate-backed diagnostic requires a source snapshot.")
                    assert args.canonical_goal_certificate_input is not None
                    diagnostic_validation_snapshot = generator._canonical_goal_certificate_validation_kwargs()
                    _require_unchanged_canonical_goal_validation_snapshot(
                        {"expected_source_sha256": certificate_input_source_snapshot},
                        {"expected_source_sha256": diagnostic_validation_snapshot["expected_source_sha256"]},
                        operation="diagnostic environment construction",
                    )
                    diagnostic_certificate = _load_canonical_goal_certificate(
                        args.canonical_goal_certificate_input,
                        **diagnostic_validation_snapshot,
                    )
                diagnostic_evidence = _run_save_disabled_diagnostic(generator, args, diagnostic_certificate)
                if diagnostic_certificate is not None:
                    assert diagnostic_validation_snapshot is not None
                    if diagnostic_recovery_phase is None:
                        _finalize_certificate_backed_phase0_diagnostic(
                            generator,
                            diagnostic_certificate,
                            diagnostic_validation_snapshot,
                            diagnostic_evidence,
                        )
                    else:
                        _finalize_certificate_backed_recovery_diagnostic(
                            generator,
                            diagnostic_certificate,
                            diagnostic_validation_snapshot,
                            diagnostic_evidence,
                        )
            else:
                payload = _generate_and_save_reset_dataset_artifact(
                    generator,
                    output=args.output,
                    certificate_input=args.canonical_goal_certificate_input,
                    pre_environment_source_sha256=certificate_input_source_snapshot,
                    checkpoint_path=checkpoint_path,
                    resuming_checkpoint=resuming_checkpoint,
                )
        finally:
            env.close()

    if args.canonical_goal_certificate_output is not None:
        assert certificate_payload is not None
        certificate_output = args.canonical_goal_certificate_output.expanduser().resolve()
        print(f"[INFO] Wrote canonical-goal certificate to {certificate_output}.")
        print(f"[INFO] Certificate SHA-256: {certificate_payload['content_sha256']}")
        return
    if reset_dataset_saving_disabled:
        return
    assert payload is not None
    output = args.output.expanduser().resolve()
    print(f"[INFO] Wrote {len(payload['states']['phase'])} pick-insert reset rows to {output}.")
    print(f"[INFO] Content SHA-256: {payload['content_sha256']}")
    if args.validate:
        import subprocess
        import sys

        validator_name = (
            "validate_franka_rj45_pick_insert_fast_resets.py"
            if generator_cfg.generation_mode == _GENERATION_MODE_FAST_IK
            else "validate_franka_rj45_pick_insert_resets.py"
        )
        command = [sys.executable, str(Path(__file__).with_name(validator_name)), "--input", str(output)]
        if generator_cfg.generation_mode == _GENERATION_MODE_FAST_IK:
            if args.validation_output_dir is not None:
                command.extend(("--output", str(Path(args.validation_output_dir) / "reset_validation.json")))
        else:
            command.extend(("--device", str(args.device)))
            if args.quick:
                command.append("--quick")
            if args.validation_output_dir is not None:
                command.extend(("--output-dir", str(args.validation_output_dir)))
            if getattr(args, "headless", False):
                command.append("--headless")
        subprocess.run(command, check=True)
    if checkpoint_path is not None and not args.keep_checkpoint:
        _unlink_generation_checkpoint_durable(checkpoint_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=(
            "Artifact path. The default training-bank path is reserved for the 20,004-row reference fast profile; "
            "legacy physical-oracle generation requires an explicit noncanonical path."
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--generation-mode",
        choices=_GENERATION_MODES,
        default=_GENERATION_MODE_PHYSICAL_ORACLE,
        help=(
            "Use the legacy physical oracle or the bounded zero-step fast-IK screen. The reference fast bank uses "
            "--rows-per-phase 3334 --batch-size 256."
        ),
    )
    parser.add_argument(
        "--rows-per-phase",
        type=int,
        default=_CANONICAL_ROWS_PER_PHASE,
        help="Rows in each of six balanced phases; use 3334 for the 20,004-row reference fast bank.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Simulation width; defaults to 4 in certifier mode and 24 otherwise.",
    )
    parser.add_argument("--max-batches-per-phase", type=int, default=96)
    parser.add_argument("--quick", action="store_true", help="Generate one row in each of the six phases.")
    parser.add_argument("--validate", action="store_true", help="Run the independent validator after saving.")
    certificate_group = parser.add_mutually_exclusive_group()
    certificate_group.add_argument(
        "--canonical-goal-certificate-output",
        type=Path,
        help="Atomically write a production canonical-goal certificate with one or four environments; no dataset.",
    )
    certificate_group.add_argument(
        "--canonical-goal-certificate-input",
        type=Path,
        help=(
            "Safely load a production canonical-goal certificate for legacy batch-24 physical generation, "
            "scalable fast-IK generation, or the phase-0 transport and phase-1/2/4/5 recovery diagnostics."
        ),
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--checkpoint",
        type=Path,
        help="Create an atomic JSON checkpoint after every complete production generation batch.",
    )
    checkpoint_group.add_argument(
        "--resume-from",
        type=Path,
        help="Resume exact production generation from an existing atomic JSON checkpoint.",
    )
    parser.add_argument(
        "--keep-checkpoint",
        action="store_true",
        help="Retain a stable-published checkpoint after saving and any requested validation.",
    )
    parser.add_argument(
        "--diagnostic-goal-only",
        action="store_true",
        help="Prove the canonical goal without generating or writing reset rows.",
    )
    parser.add_argument(
        "--diagnostic-pickup-only",
        action="store_true",
        help="Time and report exactly one randomized batch pickup attempt without writing an artifact.",
    )
    parser.add_argument(
        "--diagnostic-phase0-transport-only",
        action="store_true",
        help=(
            "Run and report exactly one certificate-backed randomized production pickup plus phase-0 loaded "
            "transport batch without writing an artifact; requires --canonical-goal-certificate-input."
        ),
    )
    parser.add_argument(
        "--diagnostic-recovery-phase",
        type=int,
        choices=(1, 2, 4, 5),
        help=(
            "Run exactly one certificate-backed production batch for phase 1, 2, 4, or 5 through pickup, "
            "realization, "
            "cold replay, semantics, and recovery oracle without checkpointing, saving, or validation."
        ),
    )
    parser.add_argument(
        "--diagnostic-reset-abcd",
        action="store_true",
        help="Compare one canonical candidate under continuous/raw/zero/effective-velocity reset histories.",
    )
    parser.add_argument(
        "--diagnostic-reset-e",
        action="store_true",
        help="Compare a C4-style continuous endpoint against a cold reset restoring both VBD pose histories.",
    )
    parser.add_argument(
        "--diagnostic-p-relax-reseat",
        action="store_true",
        help=(
            "Run a no-artifact continuous P relaxation, one authored-pose reseat, and strict 30/60-second "
            "cold-proof discriminator."
        ),
    )
    parser.add_argument(
        "--diagnostic-zero-finger-close-target",
        action="store_true",
        help=(
            "Compatibility assertion for the production 0 m per-finger close target; only legal on the "
            "no-artifact P relax/reseat discriminator."
        ),
    )
    parser.add_argument(
        "--diagnostic-forward-grasp-offset",
        action="store_true",
        help=(
            "Use the plug-local (0, -0.020, 0.010) m grasp offset only with the zero-close, no-artifact "
            "P relax/reseat discriminator; production remains fixed at (0, -0.025, 0.010) m."
        ),
    )
    parser.add_argument(
        "--diagnostic-effective-grasp-friction-three",
        action="store_true",
        help=(
            "Compatibility assertion for the production raw GraspProxy friction 4.5/effective pair friction 3; "
            "only legal on the zero-close no-artifact P relax/reseat discriminator."
        ),
    )
    parser.add_argument(
        "--validation-output-dir",
        type=Path,
        help="Optional report directory forwarded to the independent validator.",
    )
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = 4 if args.canonical_goal_certificate_output is not None else _CANONICAL_BATCH_SIZE
    reset_dataset_saving_disabled = _validate_parsed_artifact_policy(args)
    checkpoint_path, resuming_checkpoint = _validate_generation_checkpoint_invocation(args)
    certificate_input_source_snapshot = (
        _canonical_goal_source_digests() if args.canonical_goal_certificate_input is not None else None
    )
    lock_protected_paths = tuple(
        path
        for path in (
            checkpoint_path,
            args.output,
            args.canonical_goal_certificate_input,
            _DEFAULT_STABLE_VALIDATION_REPORT_PATH if args.validate else None,
        )
        if path is not None
    )
    lock_context = (
        nullcontext()
        if checkpoint_path is None
        else _GenerationCheckpointLock(checkpoint_path, protected_paths=lock_protected_paths)
    )
    with lock_context:
        _execute_parsed_invocation(
            args,
            reset_dataset_saving_disabled=reset_dataset_saving_disabled,
            certificate_input_source_snapshot=certificate_input_source_snapshot,
            checkpoint_path=checkpoint_path,
            resuming_checkpoint=resuming_checkpoint,
        )


if __name__ == "__main__":
    main()
