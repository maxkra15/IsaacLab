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

from isaaclab_tasks.contrib.franka_rj45_insertion.asset_provenance import (
    configured_franka_rj45_asset_closure,
    franka_rj45_asset_contract,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_env_cfg import (
    FrankaRJ45PickInsertEnvCfg,
    pick_insert_reset_dataset_task_contract,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_reset_dataset_io import (
    FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY,
    FRANKA_RJ45_PICK_INSERT_FAST_RESET_VALIDATION_FORMAT,
    FRANKA_RJ45_PICK_INSERT_FAST_RESET_VALIDATION_SCHEMA_VERSION,
    PICK_INSERT_RESET_PHASE_IDS,
    fast_reset_validation_report_validate_runtime,
    franka_rj45_validation_source_sha256,
    reset_dataset_digest,
    reset_dataset_validate_full_pick_diversity,
    reset_dataset_validate_phase_row_counts,
    reset_dataset_validate_runtime,
    reset_validation_report_content_digest,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INPUT = _REPO_ROOT / "datasets/franka_rj45_pick_insert/reset_dataset.pt"
_DEFAULT_OUTPUT = _REPO_ROOT / "logs/rsl_rl/franka_rj45_pick_insert/validation/reset_validation.json"
_ARM_LOWER = torch.tensor((-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973))
_ARM_UPPER = torch.tensor((2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973))


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


def _fast_metadata_evidence(metadata: Mapping[str, Any], phases: torch.Tensor) -> tuple[dict[int, dict[str, bool]], str]:
    accepted = metadata.get("accepted_fast_reset_metrics")
    if not isinstance(accepted, Mapping):
        raise ValueError("Artifact has no accepted_fast_reset_metrics and no prior replay report was supplied.")
    for phase in PICK_INSERT_RESET_PHASE_IDS:
        records = accepted.get(str(phase))
        expected_count = int((phases == phase).sum())
        if not isinstance(records, list) or len(records) != expected_count:
            raise ValueError(f"Fast-IK metadata does not contain every accepted phase-{phase} row.")
        for record in records:
            checks = record.get("checks") if isinstance(record, Mapping) else None
            if not isinstance(checks, Mapping) or checks != FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY["checks"]:
                raise ValueError("Fast-IK accepted-row checks do not match the production policy.")
    evidence = {
        row_id: {
            "finite": True,
            "ik_solved": True,
            "collision_filtered": True,
            "phase_semantics": True,
        }
        for row_id in range(len(phases))
    }
    return evidence, reset_dataset_digest(accepted)


def _screen_rows(
    states: Mapping[str, torch.Tensor],
    task_contract: Mapping[str, Any],
    bound_evidence: Mapping[int, Mapping[str, bool]],
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
        & (
            nonadjacent_separation
            >= 2.0 * cable_radius + collision_policy["minimum_nonadjacent_cable_surface_gap_m"]
        )
        & (cable_socket_distance >= collision_policy["minimum_cable_socket_center_distance_m"])
    )
    phase_semantics = states["starts_grasped"] == (phase <= 3)

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
    env_cfg = FrankaRJ45PickInsertEnvCfg()
    task_contract = pick_insert_reset_dataset_task_contract(env_cfg)
    metadata, states, _ = reset_dataset_validate_runtime(payload, expected_task_contract=task_contract)
    reset_dataset_validate_phase_row_counts(
        states["phase"], expected_rows_per_phase=env_cfg.reset_dataset_rows_per_phase
    )
    diversity = reset_dataset_validate_full_pick_diversity(states, task_contract=task_contract)
    if prior_validation_report is None:
        bound_evidence, evidence_sha256 = _fast_metadata_evidence(metadata, states["phase"])
        evidence_method = "current-source-cpu-static-plus-fast-ik-metadata"
    else:
        bound_evidence, evidence_sha256 = _legacy_evidence(
            prior_validation_report,
            artifact_content_sha256=payload["content_sha256"],
            task_contract=task_contract,
            phases=states["phase"],
        )
        evidence_method = "current-source-cpu-static-plus-bound-prior-reset-replay"
    rows = _screen_rows(states, task_contract, bound_evidence)
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
    parser.add_argument("--promote-output", type=Path)
    args = parser.parse_args()
    input_path = args.input.expanduser().resolve()
    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    report = build_report(payload, prior_validation_report=args.prior_validation_report)
    if not report["passed"]:
        raise RuntimeError(f"Fast reset screen failed for rows {report['failed_row_ids']}.")
    if args.promote_output is not None:
        promoted = _copy_atomic(input_path, args.promote_output)
        print(f"[INFO] Promoted reset dataset: {promoted}")
    output = _write_json_atomic(report, args.output)
    print(f"[INFO] Fast reset validation gate: {output}")
    print(f"[INFO] Passed={report['passed']} rows={report['dataset_row_count']} simulation_steps=0")


if __name__ == "__main__":
    main()
