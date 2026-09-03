# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish the zero-step fast-IK reset gate for Franka RJ45 pick-and-insert."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from isaaclab.utils import math as math_utils

from isaaclab_tasks.contrib.franka_rj45_insertion.asset_provenance import (
    configured_franka_rj45_asset_closure,
    franka_rj45_asset_contract,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.dual_rack_env_cfg import (
    FrankaRJ45DualRackInsertEnvCfg,
    dual_rack_reset_dataset_task_contract,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.gb300_env_cfg import (
    FrankaRJ45Gb300InsertEnvCfg,
    gb300_reset_dataset_task_contract,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_env_cfg import (
    FrankaRJ45PickInsertEnvCfg,
    pick_insert_phase_0_reverse_curriculum_sampling_contract,
    pick_insert_reset_dataset_task_contract,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_reset_dataset_io import (
    FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY,
    FRANKA_RJ45_PICK_INSERT_FAST_RESET_VALIDATION_FORMAT,
    FRANKA_RJ45_PICK_INSERT_FAST_RESET_VALIDATION_SCHEMA_VERSION,
    PICK_INSERT_FAST_RESET_PHASE_0_BAND_ACCEPTANCE_CONTRACT,
    PICK_INSERT_FAST_RESET_ROW_BINDING_CONTRACT,
    PICK_INSERT_RESET_PHASE_IDS,
    fast_reset_validation_report_validate_runtime,
    franka_rj45_validation_source_sha256,
    pick_insert_fast_reset_phase_0_band_fraction_tolerance,
    pick_insert_reset_dataset_row_digest,
    reset_dataset_digest,
    reset_dataset_validate_full_pick_diversity,
    reset_dataset_validate_phase_row_counts,
    reset_dataset_validate_runtime,
    reset_validation_report_content_digest,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.task_success import rj45_insertion_success

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INPUT = _REPO_ROOT / "datasets/franka_rj45_pick_insert/reset_dataset.pt"
_DEFAULT_OUTPUT = _REPO_ROOT / "logs/rsl_rl/franka_rj45_pick_insert/validation/reset_validation.json"
_ARM_LOWER = torch.tensor((-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973))
_ARM_UPPER = torch.tensor((2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973))
_REFERENCE_FAST_ROWS_PER_PHASE = 3_334
_REFERENCE_FAST_BATCH_SIZE = 256
_REFERENCE_FAST_MAXIMUM_BATCHES_PER_PHASE = 96


def _normalized_digest(value: Any) -> str:
    return reset_dataset_digest(json.loads(json.dumps(value, allow_nan=False)))


def _write_json_atomic(payload: Mapping[str, Any], output: Path) -> Path:
    output = output.expanduser().resolve()
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
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _copy_atomic(source: Path, output: Path) -> Path:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if source == output:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _artifact_bound_env_cfg(
    payload: Mapping[str, Any],
) -> FrankaRJ45PickInsertEnvCfg | FrankaRJ45DualRackInsertEnvCfg | FrankaRJ45Gb300InsertEnvCfg:
    """Configure only artifact-cardinality fields before exact contract validation."""
    metadata = payload.get("metadata")
    states = payload.get("states")
    if not isinstance(metadata, Mapping) or not isinstance(states, Mapping):
        raise ValueError("Reset artifact must define metadata and states mappings.")
    task_contract = metadata.get("task_contract")
    if not isinstance(task_contract, Mapping):
        raise ValueError("Reset artifact metadata must define a task_contract mapping.")
    pick_insert = task_contract.get("pick_insert")
    if not isinstance(pick_insert, Mapping):
        raise ValueError("Reset artifact task contract must define pick_insert.")
    rows_per_phase = pick_insert.get("reset_dataset_rows_per_phase")
    if type(rows_per_phase) is not int or rows_per_phase < 1:
        raise ValueError("Artifact reset_dataset_rows_per_phase must be a positive plain integer.")

    phases = states.get("phase")
    if not isinstance(phases, torch.Tensor) or phases.dtype != torch.int64 or phases.ndim != 1:
        raise ValueError("Artifact states.phase must be a one-dimensional torch.int64 tensor.")
    reset_dataset_validate_phase_row_counts(phases, expected_rows_per_phase=rows_per_phase)

    diversity = pick_insert.get("full_pick_diversity")
    if not isinstance(diversity, Mapping):
        raise ValueError("Reset artifact task contract must define full_pick_diversity.")
    unique_minima = tuple(
        diversity.get(name)
        for name in (
            "minimum_unique_socket_rows",
            "minimum_unique_plug_rows",
            "minimum_unique_arm_rows",
        )
    )
    task_variant = task_contract.get("task_variant")
    if any(type(value) is not int for value in unique_minima) or any(
        not 1 <= value <= rows_per_phase for value in unique_minima
    ):
        raise ValueError("Artifact full-pick unique-row minima must be positive and no larger than the phase pool.")
    if task_variant == "franka-rj45-gb300-insert":
        if unique_minima[0] != 8 or unique_minima[1] != unique_minima[2]:
            raise ValueError("GB300 diversity requires eight socket anchors and shared plug/arm minima.")
        env_cfg = FrankaRJ45Gb300InsertEnvCfg()
        shared_minimum = unique_minima[1]
    elif task_variant == "franka-rj45-dual-rack-insert":
        if unique_minima[0] != 1 or unique_minima[1] != unique_minima[2]:
            raise ValueError("Dual-rack diversity requires one fixed socket row and shared plug/arm minima.")
        env_cfg = FrankaRJ45DualRackInsertEnvCfg()
        shared_minimum = unique_minima[1]
    else:
        if task_variant != "franka-rj45-pick-insert" or len(set(unique_minima)) != 1:
            raise ValueError("Pick-insert diversity requires one shared socket/plug/arm minimum.")
        env_cfg = FrankaRJ45PickInsertEnvCfg()
        shared_minimum = unique_minima[0]
    env_cfg.reset_dataset_rows_per_phase = rows_per_phase
    env_cfg.reset_dataset_min_unique_full_pick_rows = shared_minimum
    env_cfg.validate_config()
    return env_cfg


def _task_contract_for_cfg(
    env_cfg: FrankaRJ45PickInsertEnvCfg | FrankaRJ45DualRackInsertEnvCfg | FrankaRJ45Gb300InsertEnvCfg,
) -> dict[str, object]:
    """Return the exact variant contract selected by the artifact-bound config."""
    if isinstance(env_cfg, FrankaRJ45Gb300InsertEnvCfg):
        return gb300_reset_dataset_task_contract(env_cfg)
    if isinstance(env_cfg, FrankaRJ45DualRackInsertEnvCfg):
        return dual_rack_reset_dataset_task_contract(env_cfg)
    return pick_insert_reset_dataset_task_contract(env_cfg)


def _require_reference_promotion(payload: Mapping[str, Any]) -> None:
    """Allow canonical promotion only for the exact live 20,004-row profile."""
    metadata = payload.get("metadata")
    states = payload.get("states")
    if not isinstance(metadata, Mapping) or not isinstance(states, Mapping):
        raise ValueError("Canonical promotion requires reset artifact metadata and states mappings.")
    task_contract = metadata.get("task_contract")
    phases = states.get("phase")
    initial_state_policy = metadata.get("initial_state_policy")
    profile = initial_state_policy.get("reset_bank_profile") if isinstance(initial_state_policy, Mapping) else None

    live_cfg = FrankaRJ45PickInsertEnvCfg()
    live_cfg.validate_config()
    live_contract = pick_insert_reset_dataset_task_contract(live_cfg)
    expected_profile = {
        "contract_version": 1,
        "profile": "balanced-20004-v1",
        "reference_profile": True,
        "rows_per_phase": _REFERENCE_FAST_ROWS_PER_PHASE,
        "phase_counts": (_REFERENCE_FAST_ROWS_PER_PHASE,) * len(PICK_INSERT_RESET_PHASE_IDS),
        "total_rows": _REFERENCE_FAST_ROWS_PER_PHASE * len(PICK_INSERT_RESET_PHASE_IDS),
        "batch_size": _REFERENCE_FAST_BATCH_SIZE,
        "maximum_batches_per_phase": _REFERENCE_FAST_MAXIMUM_BATCHES_PER_PHASE,
        "simulation_steps_per_row": 0,
    }
    try:
        reset_dataset_validate_phase_row_counts(
            phases,
            expected_rows_per_phase=_REFERENCE_FAST_ROWS_PER_PHASE,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Canonical promotion requires the exact 20,004-row reference fast profile.") from error
    if (
        not isinstance(task_contract, Mapping)
        or _normalized_digest(task_contract) != _normalized_digest(live_contract)
        or not isinstance(profile, Mapping)
        or _normalized_digest(profile) != _normalized_digest(expected_profile)
        or not isinstance(initial_state_policy, Mapping)
        or initial_state_policy.get("generation_mode") != "fast-ik"
        or _normalized_digest(initial_state_policy.get("fast_reset_policy"))
        != _normalized_digest(FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY)
        or _normalized_digest(initial_state_policy.get("accepted_row_binding"))
        != _normalized_digest(PICK_INSERT_FAST_RESET_ROW_BINDING_CONTRACT)
    ):
        raise ValueError("Canonical promotion requires the exact reference profile and default live task contract.")
    # Promotion must stand on the artifact's current per-row fast evidence even
    # when a caller also supplied a legacy replay report to build the report.
    _fast_metadata_evidence(metadata, states)


def _legacy_evidence(
    path: Path,
    *,
    artifact_content_sha256: str,
    task_contract: Mapping[str, Any],
    phases: torch.Tensor,
) -> tuple[dict[int, dict[str, bool]], str]:
    report = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if report.get("content_sha256") != reset_validation_report_content_digest(report):
        raise ValueError("Prior reset replay report content digest is invalid.")
    if report.get("artifact_content_sha256") != artifact_content_sha256:
        raise ValueError("Prior reset replay report is bound to a different artifact.")
    if _normalized_digest(report.get("task_contract")) != _normalized_digest(task_contract):
        raise ValueError("Prior reset replay report is bound to a different task contract.")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != len(phases):
        raise ValueError("Prior reset replay report does not cover every artifact row.")
    evidence: dict[int, dict[str, bool]] = {}
    for row in rows:
        row_id = row.get("row_id")
        checks = row.get("checks")
        metrics = row.get("metrics")
        if (
            isinstance(row_id, bool)
            or not isinstance(row_id, int)
            or not 0 <= row_id < len(phases)
            or not isinstance(checks, Mapping)
            or not isinstance(metrics, Mapping)
            or row.get("phase") != int(phases[row_id])
        ):
            raise ValueError("Prior reset replay report row identity is invalid.")
        evidence[row_id] = {
            "finite": checks.get("reset_all_post_step_finite") is True and checks.get("reset_stable") is True,
            "ik_solved": metrics.get("initial_goal_error_matches") is True
            and metrics.get("initial_tcp_distance_matches") is True,
            "collision_filtered": checks.get("reset_all_post_step_collision_valid") is True,
            "phase_semantics": checks.get("phase_semantics") is True,
        }
    if set(evidence) != set(range(len(phases))) or not all(all(values.values()) for values in evidence.values()):
        raise ValueError("Prior replay did not pass every reset-only admission check.")
    return evidence, str(report["content_sha256"])


def _plain_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _bound_fast_record(
    value: Any,
    *,
    states: Mapping[str, torch.Tensor],
    phases: torch.Tensor,
    phase: int,
    seen_row_ids: Mapping[int, Any],
) -> tuple[Mapping[str, Any], int]:
    """Validate and return one exact final-row evidence binding."""
    if not isinstance(value, Mapping):
        raise ValueError("Fast-IK accepted-row evidence must be a mapping.")
    final_row_id = value.get("final_row_id")
    state_sha256 = value.get("state_sha256")
    if (
        type(final_row_id) is not int
        or not 0 <= final_row_id < len(phases)
        or final_row_id in seen_row_ids
        or int(phases[final_row_id]) != phase
        or not isinstance(state_sha256, str)
        or state_sha256 != pick_insert_reset_dataset_row_digest(states, final_row_id)
    ):
        raise ValueError("Fast-IK accepted-row evidence is not bound to its exact final artifact row.")
    return value, final_row_id


def _record_phase_0_evidence(
    record: Mapping[str, Any],
    *,
    band_names: tuple[str, ...],
    band_ranges: tuple[tuple[float, float], ...],
    band_counts: dict[str, int],
    shortfalls: list[float],
) -> None:
    """Validate and accumulate one phase-0 accepted-row record."""
    band_name = record.get("phase_0_reverse_curriculum_band")
    shortfall = record.get("phase_0_axial_shortfall_m")
    if not isinstance(band_name, str) or band_name not in band_counts:
        raise ValueError("Fast-IK phase-0 row has an unknown reverse-curriculum band.")
    if not isinstance(shortfall, int | float) or isinstance(shortfall, bool) or not math.isfinite(float(shortfall)):
        raise ValueError("Fast-IK phase-0 axial shortfall must be finite.")
    band_index = band_names.index(band_name)
    lower, upper = band_ranges[band_index]
    upper_valid = float(shortfall) <= upper if band_index == len(band_names) - 1 else float(shortfall) < upper
    if not lower <= float(shortfall) or not upper_valid:
        raise ValueError("Fast-IK phase-0 axial shortfall lies outside its recorded band.")
    if record.get("initial_runtime_geometric_success") is not False:
        raise ValueError("Fast-IK phase-0 row is already a geometric success at reset.")
    band_counts[band_name] += 1
    shortfalls.append(float(shortfall))


def _fast_metadata_evidence(
    metadata: Mapping[str, Any], states: Mapping[str, torch.Tensor]
) -> tuple[dict[int, dict[str, bool]], str]:
    phases = states.get("phase")
    if not isinstance(phases, torch.Tensor):
        raise ValueError("Fast-IK evidence validation requires states.phase.")
    initial_state_policy = metadata.get("initial_state_policy")
    if not isinstance(initial_state_policy, Mapping):
        raise ValueError("Fast-IK artifact has no initial_state_policy mapping.")
    policy = initial_state_policy.get("fast_reset_policy")
    if (
        initial_state_policy.get("generation_mode") != FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY["generation_mode"]
        or not isinstance(policy, Mapping)
        or _normalized_digest(policy) != _normalized_digest(FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY)
    ):
        raise ValueError("Fast-IK metadata does not bind the current fast reset policy.")

    phase_counts = tuple(int((phases == phase).sum()) for phase in PICK_INSERT_RESET_PHASE_IDS)
    if len(set(phase_counts)) != 1:
        raise ValueError("Fast-IK metadata evidence requires a balanced reset bank.")
    rows_per_phase = phase_counts[0]
    profile = initial_state_policy.get("reset_bank_profile")
    if not isinstance(profile, Mapping):
        raise ValueError("Fast-IK metadata has no reset_bank_profile mapping.")
    batch_size = profile.get("batch_size")
    maximum_batches = profile.get("maximum_batches_per_phase")
    profile_phase_counts = profile.get("phase_counts")
    reference_profile = rows_per_phase == _REFERENCE_FAST_ROWS_PER_PHASE and batch_size == _REFERENCE_FAST_BATCH_SIZE
    expected_profile_name = "balanced-20004-v1" if reference_profile else "custom-balanced-fast-ik"
    if not (
        type(profile.get("contract_version")) is int
        and profile.get("contract_version") == 1
        and profile.get("profile") == expected_profile_name
        and profile.get("reference_profile") is reference_profile
        and type(profile.get("rows_per_phase")) is int
        and profile.get("rows_per_phase") == rows_per_phase
        and isinstance(profile_phase_counts, list | tuple)
        and all(type(count) is int for count in profile_phase_counts)
        and tuple(profile_phase_counts) == phase_counts
        and type(profile.get("total_rows")) is int
        and profile.get("total_rows") == len(phases)
        and type(batch_size) is int
        and batch_size >= 1
        and type(maximum_batches) is int
        and maximum_batches >= 1
        and maximum_batches * batch_size >= rows_per_phase
        and type(profile.get("simulation_steps_per_row")) is int
        and profile.get("simulation_steps_per_row") == 0
    ):
        raise ValueError("Fast-IK reset_bank_profile is inconsistent with the artifact rows.")
    metadata_phase_counts = metadata.get("phase_counts")
    if (
        not isinstance(metadata_phase_counts, list | tuple)
        or not all(type(count) is int for count in metadata_phase_counts)
        or tuple(metadata_phase_counts) != phase_counts
    ):
        raise ValueError("Fast-IK metadata phase_counts do not match the artifact rows.")

    sampler_contract = initial_state_policy.get("phase_0_reverse_curriculum_sampling")
    expected_sampler_contract = pick_insert_phase_0_reverse_curriculum_sampling_contract()
    if not isinstance(sampler_contract, Mapping) or _normalized_digest(sampler_contract) != _normalized_digest(
        expected_sampler_contract
    ):
        raise ValueError("Fast-IK metadata does not bind the current phase-0 reverse-curriculum sampler.")
    row_binding = initial_state_policy.get("accepted_row_binding")
    if not isinstance(row_binding, Mapping) or _normalized_digest(row_binding) != _normalized_digest(
        PICK_INSERT_FAST_RESET_ROW_BINDING_CONTRACT
    ):
        raise ValueError("Fast-IK metadata does not bind the current final-row evidence contract.")
    band_acceptance = initial_state_policy.get("phase_0_accepted_band_proportions")
    if not isinstance(band_acceptance, Mapping) or _normalized_digest(band_acceptance) != _normalized_digest(
        PICK_INSERT_FAST_RESET_PHASE_0_BAND_ACCEPTANCE_CONTRACT
    ):
        raise ValueError("Fast-IK metadata does not bind the current phase-0 accepted-band tolerance.")

    accepted = metadata.get("accepted_fast_reset_metrics")
    if not isinstance(accepted, Mapping):
        raise ValueError("Artifact has no accepted_fast_reset_metrics and no prior replay report was supplied.")
    newton_query = FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY["collision_filter"]["newton_query"]
    collision_evidence_fields = tuple(newton_query["accepted_row_evidence_fields"])
    band_names = tuple(expected_sampler_contract["band_names"])
    band_ranges = tuple(expected_sampler_contract["axial_offset_ranges_m"])
    phase_0_band_counts = {name: 0 for name in band_names}
    phase_0_shortfalls: list[float] = []
    evidence: dict[int, dict[str, bool]] = {}
    for phase in PICK_INSERT_RESET_PHASE_IDS:
        records = accepted.get(str(phase))
        expected_count = phase_counts[phase]
        if not isinstance(records, list) or len(records) != expected_count:
            raise ValueError(f"Fast-IK metadata does not contain every accepted phase-{phase} row.")
        for value in records:
            record, final_row_id = _bound_fast_record(
                value,
                states=states,
                phases=phases,
                phase=phase,
                seen_row_ids=evidence,
            )
            checks = record.get("checks")
            if not isinstance(checks, Mapping) or checks != FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY["checks"]:
                raise ValueError("Fast-IK accepted-row checks do not match the production policy.")
            if any(name not in record for name in collision_evidence_fields):
                raise ValueError("Fast-IK accepted row has incomplete Newton collide-only evidence.")
            invalid_count = record["collide_only_invalid_contact_count"]
            grasp_count = record["collide_only_grasp_contact_count"]
            left_count = record["collide_only_left_grasp_contact_count"]
            right_count = record["collide_only_right_grasp_contact_count"]
            contact_overflow = record["collide_only_contact_overflow"]
            contact_counts = (invalid_count, grasp_count, left_count, right_count)
            if not all(_plain_nonnegative_int(value) for value in contact_counts):
                raise ValueError("Fast-IK Newton collide-only contact counts must be non-negative plain integers.")
            if contact_overflow is not False or invalid_count != 0:
                raise ValueError("Fast-IK accepted row has invalid or overflowing Newton contacts.")
            if phase <= 3:
                if left_count < 1 or right_count < 1 or grasp_count != left_count + right_count:
                    raise ValueError("Fast-IK starts-grasped row lacks bilateral Newton proxy-contact evidence.")
            elif grasp_count != 0 or left_count != 0 or right_count != 0:
                raise ValueError("Fast-IK open-gripper row has Newton proxy contacts.")

            if phase == 0:
                _record_phase_0_evidence(
                    record,
                    band_names=band_names,
                    band_ranges=band_ranges,
                    band_counts=phase_0_band_counts,
                    shortfalls=phase_0_shortfalls,
                )
            evidence[final_row_id] = {
                "finite": True,
                "ik_solved": True,
                "collision_filtered": True,
                "phase_semantics": True,
            }

    if set(evidence) != set(range(len(phases))):
        raise ValueError("Fast-IK final-row evidence does not cover every artifact row exactly once.")

    phase_0_evidence = initial_state_policy.get("phase_0_reverse_curriculum_evidence")
    if not isinstance(phase_0_evidence, Mapping) or not phase_0_shortfalls:
        raise ValueError("Fast-IK metadata has no phase-0 reverse-curriculum evidence.")
    band_fractions = {name: count / rows_per_phase for name, count in phase_0_band_counts.items()}
    maximum_fraction_error = max(
        abs(band_fractions[name] - weight)
        for name, weight in zip(band_names, expected_sampler_contract["band_weights"], strict=True)
    )
    allowed_fraction_error = pick_insert_fast_reset_phase_0_band_fraction_tolerance(rows_per_phase)
    if maximum_fraction_error > allowed_fraction_error:
        raise ValueError("Fast-IK accepted phase-0 band proportions exceed the deterministic tolerance.")
    if not (
        phase_0_evidence.get("accepted_row_count") == rows_per_phase
        and phase_0_evidence.get("accepted_band_counts") == phase_0_band_counts
        and phase_0_evidence.get("accepted_band_fractions") == band_fractions
        and phase_0_evidence.get("maximum_absolute_band_fraction_error") == maximum_fraction_error
        and phase_0_evidence.get("allowed_absolute_band_fraction_error") == allowed_fraction_error
        and phase_0_evidence.get("band_proportions_within_tolerance") is True
        and maximum_fraction_error <= allowed_fraction_error
        and phase_0_evidence.get("minimum_axial_shortfall_m") == min(phase_0_shortfalls)
        and phase_0_evidence.get("maximum_axial_shortfall_m") == max(phase_0_shortfalls)
        and phase_0_evidence.get("initial_runtime_geometric_success_count") == 0
        and phase_0_evidence.get("all_rows_preseat_and_outside_geometric_success") is True
        and phase_0_evidence.get("simulation_steps") == 0
    ):
        raise ValueError("Fast-IK phase-0 reverse-curriculum summary does not match its accepted rows.")
    return evidence, reset_dataset_digest(accepted)


def _phase_0_reverse_curriculum_semantics(
    states: Mapping[str, torch.Tensor],
    task_contract: Mapping[str, Any],
    env_cfg: FrankaRJ45PickInsertEnvCfg,
) -> torch.Tensor:
    """Check the live phase-0 pre-seat bands and exclude terminal reset rows."""
    task_pose = states["task_body_pose"]
    goal_pose = states["goal_task_body_pose"]
    layout = task_contract["rj45_physics"]["task_layout"]
    plug_body_index = int(layout["plug_body_index"])
    latch_body_index = int(layout["latch_body_index"])
    goal_plug = goal_pose[:, plug_body_index]
    plug_error_local = math_utils.quat_apply_inverse(
        goal_plug[:, 3:7],
        task_pose[:, plug_body_index, :3] - goal_plug[:, :3],
    )
    axial_shortfall = -plug_error_local[:, 1]
    sampler_contract = pick_insert_phase_0_reverse_curriculum_sampling_contract()
    in_band = torch.zeros(len(task_pose), dtype=torch.bool, device=task_pose.device)
    ranges = tuple(sampler_contract["axial_offset_ranges_m"])
    for band_index, (lower, upper) in enumerate(ranges):
        upper_bound = axial_shortfall <= upper if band_index == len(ranges) - 1 else axial_shortfall < upper
        in_band |= (axial_shortfall >= lower) & upper_bound
    initial_success = rj45_insertion_success(
        task_pose,
        states["task_body_velocity"],
        goal_pose,
        axial_tolerance=env_cfg.success_axial_tolerance,
        axial_overtravel_tolerance=env_cfg.success_axial_overtravel_tolerance,
        radial_tolerance=env_cfg.success_radial_tolerance,
        plug_angle_tolerance=env_cfg.success_plug_angle_tolerance,
        latch_angle_tolerance=env_cfg.success_latch_angle_tolerance,
        maximum_plug_spatial_speed=env_cfg.success_max_plug_speed,
        plug_body_index=plug_body_index,
        latch_body_index=latch_body_index,
    ).mask
    return (plug_error_local[:, 1] < 0.0) & in_band & ~initial_success


def _screen_rows(
    states: Mapping[str, torch.Tensor],
    task_contract: Mapping[str, Any],
    bound_evidence: Mapping[int, Mapping[str, bool]],
    env_cfg: FrankaRJ45PickInsertEnvCfg,
) -> list[dict[str, Any]]:
    phase = states["phase"]
    row_count = len(phase)
    finite = torch.ones(row_count, dtype=torch.bool)
    for tensor in states.values():
        if tensor.is_floating_point():
            finite &= torch.isfinite(tensor).reshape(row_count, -1).all(dim=-1)
    for name in (
        "task_body_pose",
        "task_body_previous_pose",
        "task_body_coupling_previous_pose",
        "goal_task_body_pose",
    ):
        norm = torch.linalg.vector_norm(states[name][..., 3:7], dim=-1)
        finite &= (torch.abs(norm - 1.0) <= 1.0e-3).all(dim=-1)

    margin = float(FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY["joint_limit_margin_rad"])
    arm_lower = _ARM_LOWER + margin
    arm_upper = _ARM_UPPER - margin
    arm_margin = torch.full((row_count,), torch.inf)
    joint_limits = torch.ones(row_count, dtype=torch.bool)
    for name in ("arm_joint_position", "arm_joint_target"):
        values = states[name]
        arm_margin = torch.minimum(arm_margin, torch.minimum(values - _ARM_LOWER, _ARM_UPPER - values).amin(dim=-1))
        joint_limits &= ((values >= arm_lower) & (values <= arm_upper)).all(dim=-1)

    geometry = task_contract["validation_geometry"]
    task_pose = states["task_body_pose"]
    positions = task_pose[..., :3]
    workspace_lower = torch.tensor(geometry["task_body_workspace_lower"])
    workspace_upper = torch.tensor(geometry["task_body_workspace_upper"])
    workspace_margin = torch.minimum(positions - workspace_lower, workspace_upper - positions).amin(dim=(1, 2))
    workspace = workspace_margin >= 0.0

    physics = task_contract["rj45_physics"]
    layout = physics["task_layout"]
    cable_begin, cable_end = layout["cable_body_range"]
    cable = positions[:, cable_begin:cable_end]
    cable_radius = float(physics["cable_radius"])
    static_scene = task_contract["static_scene"]
    slab_position = static_scene["table_contact_initial_state"]["pos"]
    slab_size = static_scene["table_contact_spawn"]["size"]
    table_top = float(slab_position[2]) + 0.5 * float(slab_size[2])
    cable_support_clearance = cable[..., 2].amin(dim=-1) - (table_top + cable_radius)
    distances = torch.cdist(cable, cable)
    cable_indices = torch.arange(cable.shape[1])
    nonadjacent = torch.abs(cable_indices[:, None] - cable_indices[None, :]) > 1
    nonadjacent_separation = distances[:, nonadjacent].amin(dim=-1)
    socket = positions[:, int(layout["socket_body_index"])]
    cable_socket_distance = torch.linalg.vector_norm(cable - socket[:, None], dim=-1).amin(dim=-1)
    collision_policy = FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY["collision_filter"]
    collision_filtered = (
        (cable_support_clearance >= -collision_policy["maximum_table_penetration_m"])
        & (nonadjacent_separation >= 2.0 * cable_radius + collision_policy["minimum_nonadjacent_cable_surface_gap_m"])
        & (cable_socket_distance >= collision_policy["minimum_cable_socket_center_distance_m"])
    )
    phase_semantics = states["starts_grasped"] == (phase <= 3)

    phase_0 = phase == 0
    phase_semantics &= ~phase_0 | _phase_0_reverse_curriculum_semantics(states, task_contract, env_cfg)

    rows: list[dict[str, Any]] = []
    for row_id in range(row_count):
        prior = bound_evidence[row_id]
        checks = {
            "finite": bool(finite[row_id]) and prior["finite"],
            "ik_solved": prior["ik_solved"],
            "joint_limits": bool(joint_limits[row_id]),
            "workspace": bool(workspace[row_id]),
            "collision_filtered": bool(collision_filtered[row_id]) and prior["collision_filtered"],
            "phase_semantics": bool(phase_semantics[row_id]) and prior["phase_semantics"],
        }
        rows.append(
            {
                "row_id": row_id,
                "phase": int(phase[row_id]),
                "passed": all(checks.values()),
                "checks": checks,
                "metrics": {
                    "minimum_joint_limit_margin_rad": float(arm_margin[row_id]),
                    "minimum_workspace_margin_m": float(workspace_margin[row_id]),
                    "minimum_cable_support_clearance_m": float(cable_support_clearance[row_id]),
                    "minimum_nonadjacent_cable_separation_m": float(nonadjacent_separation[row_id]),
                    "minimum_cable_socket_center_distance_m": float(cable_socket_distance[row_id]),
                },
            }
        )
    return rows


def build_report(payload: Mapping[str, Any], *, prior_validation_report: Path | None) -> dict[str, Any]:
    env_cfg = _artifact_bound_env_cfg(payload)
    task_contract = _task_contract_for_cfg(env_cfg)
    metadata, states, _ = reset_dataset_validate_runtime(payload, expected_task_contract=task_contract)
    reset_dataset_validate_phase_row_counts(
        states["phase"], expected_rows_per_phase=env_cfg.reset_dataset_rows_per_phase
    )
    diversity = reset_dataset_validate_full_pick_diversity(states, task_contract=task_contract)
    if prior_validation_report is None:
        bound_evidence, evidence_sha256 = _fast_metadata_evidence(metadata, states)
        evidence_method = "current-source-cpu-static-plus-fast-ik-metadata"
    else:
        bound_evidence, evidence_sha256 = _legacy_evidence(
            prior_validation_report,
            artifact_content_sha256=payload["content_sha256"],
            task_contract=task_contract,
            phases=states["phase"],
        )
        evidence_method = "current-source-cpu-static-plus-bound-prior-reset-replay"
    rows = _screen_rows(states, task_contract, bound_evidence, env_cfg)
    failed_row_ids = [row["row_id"] for row in rows if not row["passed"]]
    source_sha256 = franka_rj45_validation_source_sha256(_REPO_ROOT, include_fast_validator=True)
    configured_franka_rj45_asset_closure(required=True)
    asset_closure = franka_rj45_asset_contract()
    report: dict[str, Any] = {
        "format": FRANKA_RJ45_PICK_INSERT_FAST_RESET_VALIDATION_FORMAT,
        "schema_version": FRANKA_RJ45_PICK_INSERT_FAST_RESET_VALIDATION_SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "artifact_content_sha256": payload["content_sha256"],
        "task_contract": task_contract,
        "validation_policy": FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY,
        "source_sha256": source_sha256,
        "asset_closure": asset_closure,
        "evidence_origin": {
            "method": evidence_method,
            "bound_evidence_content_sha256": evidence_sha256,
        },
        "simulation_steps": 0,
        "dynamics_replay": False,
        "scripted_recovery": False,
        "dataset_row_count": len(states["phase"]),
        "selected_row_ids": list(range(len(states["phase"]))),
        "phase_counts": [int((states["phase"] == phase).sum()) for phase in PICK_INSERT_RESET_PHASE_IDS],
        "full_pick_diversity": diversity,
        "rows": rows,
        "failed_row_ids": failed_row_ids,
        "passed": not failed_row_ids,
    }
    report["content_sha256"] = reset_validation_report_content_digest(report)
    fast_reset_validation_report_validate_runtime(
        report,
        expected_content_sha256=payload["content_sha256"],
        expected_row_count=len(states["phase"]),
        expected_phases=states["phase"],
        expected_task_contract=task_contract,
        expected_validation_policy=FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY,
        expected_source_sha256=source_sha256,
        expected_asset_closure=asset_closure,
        expected_full_pick_diversity=diversity,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT)
    parser.add_argument("--prior-validation-report", type=Path)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--promote-output",
        type=Path,
        help="Promote only a passing canonical 20,004-row reference bank.",
    )
    args = parser.parse_args()
    input_path = args.input.expanduser().resolve()
    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    report = build_report(payload, prior_validation_report=args.prior_validation_report)
    if not report["passed"]:
        raise RuntimeError(f"Fast reset screen failed for rows {report['failed_row_ids']}.")
    if args.promote_output is not None:
        _require_reference_promotion(payload)
        promoted = _copy_atomic(input_path, args.promote_output)
        print(f"[INFO] Promoted reset dataset: {promoted}")
    output = _write_json_atomic(report, args.output)
    print(f"[INFO] Fast reset validation gate: {output}")
    print(f"[INFO] Passed={report['passed']} rows={report['dataset_row_count']} simulation_steps=0")


if __name__ == "__main__":
    main()
