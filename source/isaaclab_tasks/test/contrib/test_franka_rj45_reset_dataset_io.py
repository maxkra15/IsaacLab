# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Franka RJ45 reset-artifact boundary."""

import json

import pytest
import torch

from isaaclab_tasks.contrib.franka_rj45_insertion.reset_dataset_io import (
    _GOAL_TENSOR_SPECS,
    _STATE_TENSOR_SPECS,
    FRANKA_RJ45_RESET_DATASET_FORMAT,
    FRANKA_RJ45_RESET_DATASET_SCHEMA_VERSION,
    FRANKA_RJ45_RESET_VALIDATION_FORMAT,
    FRANKA_RJ45_RESET_VALIDATION_SCHEMA_VERSION,
    RJ45_TASK_BODY_COUNT,
    reset_dataset_content_digest,
    reset_dataset_validate_runtime,
    reset_validation_report_validate_runtime,
)


def _payload(row_count: int = 5) -> dict:
    states = {
        name: torch.zeros((row_count, *shape), dtype=dtype) for name, (dtype, shape) in _STATE_TENSOR_SPECS.items()
    }
    states["task_body_pose"][..., 6] = 1.0
    states["progress_threshold"].fill_(0.001)
    states["phase"] = torch.arange(row_count, dtype=torch.int64) % 5
    goal = {name: torch.zeros(shape, dtype=dtype) for name, (dtype, shape) in _GOAL_TENSOR_SPECS.items()}
    goal["task_body_pose"][..., 6] = 1.0
    payload = {
        "format": FRANKA_RJ45_RESET_DATASET_FORMAT,
        "schema_version": FRANKA_RJ45_RESET_DATASET_SCHEMA_VERSION,
        "metadata": {"task_contract": {"task_body_count": RJ45_TASK_BODY_COUNT, "patterns": ("a", "b")}},
        "states": states,
        "goal_state": goal,
    }
    payload["content_sha256"] = reset_dataset_content_digest(payload)
    return payload


def test_valid_artifact_preserves_fixed_goal_separately_from_reset_rows():
    payload = _payload()

    metadata, states, goal = reset_dataset_validate_runtime(
        payload,
        expected_task_contract={"task_body_count": 37, "patterns": ("a", "b")},
    )

    assert metadata is payload["metadata"]
    assert states["task_body_pose"].shape == (5, 37, 7)
    assert goal["task_body_pose"].shape == (37, 7)
    assert goal["arm_joint_position"].shape == (7,)
    assert goal["arm_joint_target"].shape == (7,)
    assert goal["finger_joint_target"].shape == (2,)


def test_content_mutation_is_rejected():
    payload = _payload()
    payload["states"]["arm_joint_position"][0, 0] = 0.25

    with pytest.raises(ValueError, match="content digest does not match"):
        reset_dataset_validate_runtime(payload)


def test_non_unit_body_quaternion_is_rejected_after_digest_verification():
    payload = _payload()
    payload["states"]["task_body_pose"][0, 0, 6] = 0.5
    payload["content_sha256"] = reset_dataset_content_digest(payload)

    with pytest.raises(ValueError, match="quaternions must be normalized"):
        reset_dataset_validate_runtime(payload)


def test_goal_is_not_a_randomizable_row_field():
    assert "goal_state" not in _STATE_TENSOR_SPECS
    assert set(_GOAL_TENSOR_SPECS) == {
        "arm_joint_position",
        "arm_joint_target",
        "arm_joint_velocity",
        "finger_joint_position",
        "finger_joint_velocity",
        "finger_joint_target",
        "task_body_pose",
        "task_body_velocity",
    }


def test_artifact_requires_coverage_of_every_reset_phase():
    payload = _payload()
    payload["states"]["phase"].fill_(0)
    payload["content_sha256"] = reset_dataset_content_digest(payload)

    with pytest.raises(ValueError, match="every reset phase"):
        reset_dataset_validate_runtime(payload)


def _validation_report(payload: dict) -> dict:
    row_count = len(payload["states"]["phase"])
    exact_success = {
        "stored_capture_success": True,
        "all_post_step_samples_success": True,
        "dwell_satisfied": True,
        "required_dwell_steps": 9,
        "sample_steps": 9,
        "final_consecutive_steps": 9,
        "maximum_consecutive_steps": 9,
    }
    return {
        "format": FRANKA_RJ45_RESET_VALIDATION_FORMAT,
        "schema_version": FRANKA_RJ45_RESET_VALIDATION_SCHEMA_VERSION,
        "artifact_content_sha256": payload["content_sha256"],
        "task_contract": payload["metadata"]["task_contract"],
        "quick": False,
        "full_dataset_replay": True,
        "evidence_complete": True,
        "selected_row_count": row_count,
        "dataset_row_count": row_count,
        "selected_row_ids": list(range(row_count)),
        "goal_replay": {
            "passed": True,
            "simulation_steps": 600,
            "simulation_time_s": 10.0,
            "stored_capture_exact_runtime_success": True,
            "all_post_step_exact_runtime_success": True,
            "exact_runtime_success_dwell_satisfied": True,
            "exact_runtime_success_required_dwell_steps": 9,
            "exact_runtime_success_final_consecutive_steps_by_world": [600],
        },
        "rows": [
            {
                "row_id": row_id,
                "passed": True,
                "checks": {
                    "recovery_exact_capture_success": True,
                    "recovery_exact_all_post_step_samples": True,
                    "recovery_exact_success_dwell": True,
                },
                "recovery_exact_runtime_success": dict(exact_success),
            }
            for row_id in range(row_count)
        ],
        "failed_row_ids": [],
        "passed": True,
    }


def test_full_validation_report_is_bound_to_artifact_and_contract():
    payload = _payload()
    report = json.loads(json.dumps(_validation_report(payload)))

    checks = reset_validation_report_validate_runtime(
        report,
        expected_content_sha256=payload["content_sha256"],
        expected_row_count=5,
        expected_task_contract={"task_body_count": 37},
    )

    assert checks and all(checks.values())


@pytest.mark.parametrize(
    ("field", "invalid"),
    (("quick", True), ("evidence_complete", False), ("full_dataset_replay", False), ("passed", False)),
)
def test_incomplete_validation_evidence_is_rejected(field: str, invalid: object):
    payload = _payload()
    report = _validation_report(payload)
    report[field] = invalid

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        reset_validation_report_validate_runtime(
            report,
            expected_content_sha256=payload["content_sha256"],
            expected_row_count=5,
        )


def test_legacy_validation_schema_is_rejected():
    payload = _payload()
    report = _validation_report(payload)
    report["schema_version"] = FRANKA_RJ45_RESET_VALIDATION_SCHEMA_VERSION - 1

    with pytest.raises(ValueError, match="schema version is unsupported"):
        reset_validation_report_validate_runtime(
            report,
            expected_content_sha256=payload["content_sha256"],
            expected_row_count=5,
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("stored_capture_exact_runtime_success", False),
        ("all_post_step_exact_runtime_success", False),
        ("exact_runtime_success_dwell_satisfied", False),
        ("exact_runtime_success_final_consecutive_steps_by_world", [8]),
        ("simulation_time_s", 9.99),
    ),
)
def test_goal_replay_requires_exact_runtime_success_evidence(field: str, invalid: object):
    payload = _payload()
    report = _validation_report(payload)
    report["goal_replay"][field] = invalid

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        reset_validation_report_validate_runtime(
            report,
            expected_content_sha256=payload["content_sha256"],
            expected_row_count=5,
        )


@pytest.mark.parametrize(
    ("path", "invalid"),
    (
        (("checks", "recovery_exact_capture_success"), False),
        (("checks", "recovery_exact_all_post_step_samples"), False),
        (("checks", "recovery_exact_success_dwell"), False),
        (("recovery_exact_runtime_success", "stored_capture_success"), False),
        (("recovery_exact_runtime_success", "all_post_step_samples_success"), False),
        (("recovery_exact_runtime_success", "dwell_satisfied"), False),
        (("recovery_exact_runtime_success", "final_consecutive_steps"), 8),
    ),
)
def test_every_row_requires_exact_runtime_success_evidence(path: tuple[str, str], invalid: object):
    payload = _payload()
    report = _validation_report(payload)
    report["rows"][0][path[0]][path[1]] = invalid

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        reset_validation_report_validate_runtime(
            report,
            expected_content_sha256=payload["content_sha256"],
            expected_row_count=5,
        )


def test_validation_report_must_cover_each_row_exactly_once():
    payload = _payload()
    report = _validation_report(payload)
    report["selected_row_ids"][-1] = 0

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        reset_validation_report_validate_runtime(
            report,
            expected_content_sha256=payload["content_sha256"],
            expected_row_count=5,
        )
