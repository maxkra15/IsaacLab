# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Versioned reset-state artifact boundary for Franka RJ45 insertion.

The artifact stores maximal-coordinate state for every Newton body owned by the
RJ45 assembly.  This is deliberate: reconstructing a cable from only its plug
pose does not restore its settled bend, joint, or contact state.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping
from io import BytesIO
from typing import Any

import torch

FRANKA_RJ45_RESET_DATASET_FORMAT = "isaaclab-franka-rj45-reset-dataset"
FRANKA_RJ45_RESET_DATASET_SCHEMA_VERSION = 1
FRANKA_RJ45_RESET_VALIDATION_FORMAT = "isaaclab-franka-rj45-reset-validation"
FRANKA_RJ45_RESET_VALIDATION_SCHEMA_VERSION = 2
RJ45_TASK_BODY_COUNT = 37

RESET_DATASET_STATE_NAMES = (
    "arm_joint_position",
    "arm_joint_target",
    "arm_joint_velocity",
    "finger_joint_position",
    "finger_joint_velocity",
    "finger_joint_target",
    "task_body_pose",
    "task_body_velocity",
    "phase",
    "difficulty",
    "initial_goal_error",
    "progress_threshold",
)

_STATE_TENSOR_SPECS: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {
    "arm_joint_position": (torch.float32, (7,)),
    "arm_joint_target": (torch.float32, (7,)),
    "arm_joint_velocity": (torch.float32, (7,)),
    "finger_joint_position": (torch.float32, (2,)),
    "finger_joint_velocity": (torch.float32, (2,)),
    "finger_joint_target": (torch.float32, (2,)),
    "task_body_pose": (torch.float32, (RJ45_TASK_BODY_COUNT, 7)),
    "task_body_velocity": (torch.float32, (RJ45_TASK_BODY_COUNT, 6)),
    "phase": (torch.int64, ()),
    "difficulty": (torch.float32, ()),
    "initial_goal_error": (torch.float32, ()),
    "progress_threshold": (torch.float32, ()),
}

_GOAL_TENSOR_SPECS: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {
    "arm_joint_position": (torch.float32, (7,)),
    "arm_joint_target": (torch.float32, (7,)),
    "arm_joint_velocity": (torch.float32, (7,)),
    "finger_joint_position": (torch.float32, (2,)),
    "finger_joint_velocity": (torch.float32, (2,)),
    "finger_joint_target": (torch.float32, (2,)),
    "task_body_pose": (torch.float32, (RJ45_TASK_BODY_COUNT, 7)),
    "task_body_velocity": (torch.float32, (RJ45_TASK_BODY_COUNT, 6)),
}


def _write_bytes(sink: BytesIO, value: bytes) -> None:
    sink.write(struct.pack("<Q", len(value)))
    sink.write(value)


def _write_value(sink: BytesIO, value: Any) -> None:
    """Encode supported values canonically for a cross-process content digest."""
    if value is None:
        sink.write(b"none")
    elif isinstance(value, bool):
        sink.write(b"bool1" if value else b"bool0")
    elif isinstance(value, int):
        sink.write(b"int")
        _write_bytes(sink, str(value).encode())
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Reset-dataset digest values must be finite.")
        sink.write(b"float")
        sink.write(struct.pack("<d", value))
    elif isinstance(value, str):
        sink.write(b"str")
        _write_bytes(sink, value.encode())
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        sink.write(b"tensor")
        _write_value(sink, str(tensor.dtype))
        _write_value(sink, tuple(tensor.shape))
        _write_bytes(sink, tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, Mapping):
        sink.write(b"mapping")
        encoded: list[tuple[bytes, Any]] = []
        for key, item in value.items():
            key_sink = BytesIO()
            _write_value(key_sink, key)
            encoded.append((key_sink.getvalue(), item))
        encoded.sort(key=lambda pair: pair[0])
        _write_value(sink, len(encoded))
        for key_bytes, item in encoded:
            _write_bytes(sink, key_bytes)
            _write_value(sink, item)
    elif isinstance(value, (tuple, list)):
        sink.write(b"tuple" if isinstance(value, tuple) else b"list")
        _write_value(sink, len(value))
        for item in value:
            _write_value(sink, item)
    else:
        raise TypeError(f"Unsupported reset-dataset digest value: {type(value).__name__}.")


def reset_dataset_digest(value: Any) -> str:
    """Return a deterministic SHA-256 digest of tensors and plain containers."""
    sink = BytesIO()
    _write_value(sink, value)
    return hashlib.sha256(sink.getvalue()).hexdigest()


def reset_dataset_content_digest(payload: Mapping[str, Any]) -> str:
    """Digest every artifact field except the digest itself."""
    return reset_dataset_digest({key: value for key, value in payload.items() if key != "content_sha256"})


def _validate_tensor_mapping(
    values: Mapping[str, Any],
    specs: Mapping[str, tuple[torch.dtype, tuple[int, ...]]],
    *,
    leading_count: int | None,
    path: str,
) -> int | None:
    for name, (dtype, trailing_shape) in specs.items():
        tensor = values.get(name)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{path}.{name} must be a torch.Tensor.")
        if leading_count is None and path == "states":
            leading_count = int(tensor.shape[0]) if tensor.ndim else 0
            if leading_count < 1:
                raise ValueError("Reset dataset must contain at least one row.")
        expected_shape = trailing_shape if path == "goal_state" else (leading_count, *trailing_shape)
        if tensor.dtype != dtype or tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{path}.{name} must have dtype {dtype} and shape {expected_shape}, "
                f"got {tensor.dtype} and {tuple(tensor.shape)}."
            )
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{path}.{name} must contain only finite values.")
    return leading_count


def _validate_task_contract(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key, expected_value in expected.items():
        if key not in actual:
            raise ValueError(f"Reset dataset task contract is missing {key!r}.")
        actual_value = actual[key]
        if isinstance(expected_value, Mapping):
            if not isinstance(actual_value, Mapping):
                raise ValueError(f"Reset dataset task contract field {key!r} is not a mapping.")
            _validate_task_contract(actual_value, expected_value)
        elif reset_dataset_digest(actual_value) != reset_dataset_digest(expected_value):
            raise ValueError(f"Reset dataset task contract field {key!r} does not match the runtime.")


def _json_normalize(value: Any) -> Any:
    """Normalize tuple/list distinctions lost when a validation report is JSON encoded."""
    if isinstance(value, Mapping):
        return {str(key): _json_normalize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_normalize(item) for item in value]
    return value


def reset_dataset_validate_runtime(
    payload: Mapping[str, Any],
    *,
    expected_content_sha256: str | None = None,
    expected_task_contract: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    """Validate a loaded artifact and return metadata, reset rows, and fixed goal."""
    if not isinstance(payload, Mapping):
        raise TypeError("Reset dataset payload must be a mapping.")
    if payload.get("format") != FRANKA_RJ45_RESET_DATASET_FORMAT:
        raise ValueError("Reset dataset format is not Franka RJ45 insertion.")
    if payload.get("schema_version") != FRANKA_RJ45_RESET_DATASET_SCHEMA_VERSION:
        raise ValueError("Reset dataset schema version is unsupported.")

    digest = payload.get("content_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Reset dataset content_sha256 must be a SHA-256 digest.")
    actual_digest = reset_dataset_content_digest(payload)
    if digest != actual_digest:
        raise ValueError("Reset dataset content digest does not match its payload.")
    if expected_content_sha256 is not None and digest != expected_content_sha256:
        raise ValueError("Reset dataset digest does not match the configured artifact.")

    metadata = payload.get("metadata")
    states = payload.get("states")
    goal_state = payload.get("goal_state")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping.")
    if not isinstance(states, Mapping):
        raise TypeError("states must be a mapping.")
    if not isinstance(goal_state, Mapping):
        raise TypeError("goal_state must be a mapping.")
    if expected_task_contract is not None:
        contract = metadata.get("task_contract")
        if not isinstance(contract, Mapping):
            raise TypeError("metadata.task_contract must be a mapping.")
        _validate_task_contract(contract, expected_task_contract)

    row_count = _validate_tensor_mapping(states, _STATE_TENSOR_SPECS, leading_count=None, path="states")
    _validate_tensor_mapping(goal_state, _GOAL_TENSOR_SPECS, leading_count=None, path="goal_state")
    assert row_count is not None
    phase = states["phase"]
    difficulty = states["difficulty"]
    initial_error = states["initial_goal_error"]
    progress_threshold = states["progress_threshold"]
    if not bool(torch.all((phase >= 0) & (phase <= 4))):
        raise ValueError("states.phase must contain identifiers in [0, 4].")
    if not torch.equal(torch.unique(phase, sorted=True), torch.arange(5, dtype=phase.dtype, device=phase.device)):
        raise ValueError("states.phase must represent every reset phase in [0, 4].")
    if not bool(torch.all((difficulty >= 0.0) & (difficulty <= 1.0))):
        raise ValueError("states.difficulty must lie in [0, 1].")
    if not bool(torch.all(initial_error >= 0.0)):
        raise ValueError("states.initial_goal_error must be non-negative.")
    if not bool(torch.all(progress_threshold > 0.0)):
        raise ValueError("states.progress_threshold must be positive.")

    for path, pose in (
        ("states.task_body_pose", states["task_body_pose"]),
        ("goal_state.task_body_pose", goal_state["task_body_pose"]),
    ):
        quat = pose[..., 3:7]
        norm = torch.linalg.vector_norm(quat, dim=-1)
        if not bool(torch.all(torch.abs(norm - 1.0) <= 1.0e-3)):
            raise ValueError(f"{path} quaternions must be normalized.")

    return metadata, states, goal_state


def reset_validation_report_validate_runtime(
    report: Mapping[str, Any],
    *,
    expected_content_sha256: str,
    expected_row_count: int,
    expected_task_contract: Mapping[str, Any] | None = None,
) -> dict[str, bool]:
    """Require complete physical replay evidence for one reset artifact.

    A self-consistent tensor artifact is not sufficient evidence that its
    maximal-coordinate cable state survives a cold coupled-solver reset.  This
    validator binds the full replay report to the exact artifact digest and
    runtime task contract before training or evaluation can consume it.

    Returns:
        Named checks, all true when validation succeeds.
    """
    if not isinstance(report, Mapping):
        raise TypeError("Reset validation report must be a mapping.")
    if report.get("format") != FRANKA_RJ45_RESET_VALIDATION_FORMAT:
        raise ValueError("Reset validation report format is not Franka RJ45 insertion.")
    if report.get("schema_version") != FRANKA_RJ45_RESET_VALIDATION_SCHEMA_VERSION:
        raise ValueError("Reset validation report schema version is unsupported.")
    if isinstance(expected_row_count, bool) or not isinstance(expected_row_count, int) or expected_row_count < 1:
        raise ValueError("expected_row_count must be a positive integer.")

    goal_replay = report.get("goal_replay")
    if not isinstance(goal_replay, Mapping):
        raise TypeError("Reset validation report goal_replay must be a mapping.")
    failed_row_ids = report.get("failed_row_ids")
    selected_row_ids = report.get("selected_row_ids")
    rows = report.get("rows")
    if not isinstance(failed_row_ids, list):
        raise TypeError("Reset validation report failed_row_ids must be a list.")
    if not isinstance(selected_row_ids, list):
        raise TypeError("Reset validation report selected_row_ids must be a list.")
    if not isinstance(rows, list):
        raise TypeError("Reset validation report rows must be a list.")

    report_contract = report.get("task_contract")
    if expected_task_contract is not None:
        if not isinstance(report_contract, Mapping):
            raise TypeError("Reset validation report task_contract must be a mapping.")
        _validate_task_contract(_json_normalize(report_contract), _json_normalize(expected_task_contract))

    expected_ids = list(range(expected_row_count))
    selected_ids_valid = all(isinstance(row_id, int) and not isinstance(row_id, bool) for row_id in selected_row_ids)
    row_ids = [row.get("row_id") for row in rows if isinstance(row, Mapping)]
    row_ids_valid = all(isinstance(row_id, int) and not isinstance(row_id, bool) for row_id in row_ids)

    goal_required_steps = goal_replay.get("exact_runtime_success_required_dwell_steps")
    goal_final_steps = goal_replay.get("exact_runtime_success_final_consecutive_steps_by_world")
    goal_simulation_steps = goal_replay.get("simulation_steps")
    goal_simulation_time_s = goal_replay.get("simulation_time_s")
    goal_exact_success_evidence = (
        goal_replay.get("stored_capture_exact_runtime_success") is True
        and goal_replay.get("all_post_step_exact_runtime_success") is True
        and goal_replay.get("exact_runtime_success_dwell_satisfied") is True
        and isinstance(goal_required_steps, int)
        and not isinstance(goal_required_steps, bool)
        and goal_required_steps >= 1
        and isinstance(goal_simulation_steps, int)
        and not isinstance(goal_simulation_steps, bool)
        and goal_simulation_steps >= goal_required_steps
        and isinstance(goal_simulation_time_s, (int, float))
        and not isinstance(goal_simulation_time_s, bool)
        and math.isfinite(goal_simulation_time_s)
        and goal_simulation_time_s >= 10.0
        and isinstance(goal_final_steps, list)
        and len(goal_final_steps) >= 1
        and all(
            isinstance(step_count, int) and not isinstance(step_count, bool) and step_count >= goal_required_steps
            for step_count in goal_final_steps
        )
    )

    row_exact_success_evidence = True
    for row in rows:
        if not isinstance(row, Mapping):
            row_exact_success_evidence = False
            break
        row_checks = row.get("checks")
        exact = row.get("recovery_exact_runtime_success")
        if not isinstance(row_checks, Mapping) or not isinstance(exact, Mapping):
            row_exact_success_evidence = False
            break
        required_steps = exact.get("required_dwell_steps")
        sample_steps = exact.get("sample_steps")
        final_steps = exact.get("final_consecutive_steps")
        maximum_steps = exact.get("maximum_consecutive_steps")
        if not (
            row_checks.get("recovery_exact_capture_success") is True
            and row_checks.get("recovery_exact_all_post_step_samples") is True
            and row_checks.get("recovery_exact_success_dwell") is True
            and exact.get("stored_capture_success") is True
            and exact.get("all_post_step_samples_success") is True
            and exact.get("dwell_satisfied") is True
            and isinstance(required_steps, int)
            and not isinstance(required_steps, bool)
            and required_steps >= 1
            and isinstance(sample_steps, int)
            and not isinstance(sample_steps, bool)
            and sample_steps >= required_steps
            and isinstance(final_steps, int)
            and not isinstance(final_steps, bool)
            and final_steps >= required_steps
            and isinstance(maximum_steps, int)
            and not isinstance(maximum_steps, bool)
            and maximum_steps >= required_steps
        ):
            row_exact_success_evidence = False
            break

    checks = {
        "passed": report.get("passed") is True,
        "evidence_complete": report.get("evidence_complete") is True,
        "full_dataset_replay": report.get("full_dataset_replay") is True,
        "not_quick": report.get("quick") is False,
        "content_digest_matches": report.get("artifact_content_sha256") == expected_content_sha256,
        "selected_row_count_matches": report.get("selected_row_count") == expected_row_count,
        "dataset_row_count_matches": report.get("dataset_row_count") == expected_row_count,
        "selected_every_row_once": (
            selected_ids_valid
            and len(selected_row_ids) == expected_row_count
            and sorted(selected_row_ids) == expected_ids
        ),
        "goal_replay_passed": goal_replay.get("passed") is True,
        "goal_exact_runtime_success_evidence": goal_exact_success_evidence,
        "no_failed_rows": len(failed_row_ids) == 0,
        "every_row_passed": (
            len(rows) == expected_row_count
            and row_ids_valid
            and sorted(row_ids) == expected_ids
            and all(isinstance(row, Mapping) and row.get("passed") is True for row in rows)
        ),
        "every_row_exact_runtime_success_evidence": row_exact_success_evidence,
    }
    if not all(checks.values()):
        raise ValueError(f"Reset validation evidence is incomplete or incompatible: {checks}")
    return checks


__all__ = [
    "FRANKA_RJ45_RESET_DATASET_FORMAT",
    "FRANKA_RJ45_RESET_DATASET_SCHEMA_VERSION",
    "FRANKA_RJ45_RESET_VALIDATION_FORMAT",
    "FRANKA_RJ45_RESET_VALIDATION_SCHEMA_VERSION",
    "RESET_DATASET_STATE_NAMES",
    "RJ45_TASK_BODY_COUNT",
    "reset_dataset_content_digest",
    "reset_dataset_digest",
    "reset_dataset_validate_runtime",
    "reset_validation_report_validate_runtime",
]
