# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Franka RJ45 pick-and-insert reset-artifact boundary."""

import json
import math

import pytest
import torch

from isaaclab_tasks.contrib.franka_rj45_insertion.asset_provenance import franka_rj45_asset_contract
from isaaclab_tasks.contrib.franka_rj45_insertion.franka_robot_cfg import (
    PICK_INSERT_ARM_TARGET_TRACKING_LIMITS,
    franka_pick_insert_control_contract,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_reset_dataset_io import (
    FRANKA_RJ45_PICK_INSERT_RESET_DATASET_FORMAT,
    FRANKA_RJ45_PICK_INSERT_RESET_DATASET_SCHEMA_VERSION,
    FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_FORMAT,
    FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY,
    FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_SCHEMA_VERSION,
    PICK_INSERT_GOAL_MAX_ARM_JOINT_SPEED_RAD_S,
    PICK_INSERT_GOAL_MAX_CABLE_SPEED_M_S,
    PICK_INSERT_GOAL_MAX_FINGER_JOINT_SPEED_M_S,
    PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD,
    PICK_INSERT_GOAL_MAX_SOCKET_DRIFT_M,
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
    franka_rj45_validation_source_sha256,
    reset_dataset_content_digest,
    reset_dataset_validate_full_pick_diversity,
    reset_dataset_validate_phase_row_counts,
    reset_dataset_validate_runtime,
    reset_validation_report_content_digest,
    reset_validation_report_validate_runtime,
)

_ROW_CHECK_NAMES = (
    "reset_stable",
    "reset_zero_action_unclamped",
    "reset_absolute_target_stable",
    "reset_target_tracking_bounded",
    "reset_replay_sample_count_exact",
    "reset_all_post_step_finite",
    "reset_all_post_step_collision_valid",
    "reset_all_post_step_contact_state_valid",
    "reset_all_post_step_drives_disabled",
    "reset_vbd_pose_history_queued",
    "reset_vbd_pose_history_applied_exactly_once",
    "reset_vbd_pose_history_body_order_exact",
    "reset_robot_speed_bounded",
    "reset_cable_speed_bounded",
    "reset_body_excursion_bounded",
    "phase_semantics",
    "socket_stable",
    "drive_disabled",
    "collision_valid",
    "expected_initial_grasp_state",
    "oracle_grasp_acquired",
    "oracle_full_recovery",
    "exact_insertion_success_dwell",
    "oracle_all_samples_collision_free",
    "oracle_all_samples_bilateral_grasp",
    "oracle_all_samples_finite",
)
_ORACLE_EVIDENCE_NAMES = (
    "drive_disabled",
    "physical_grasp_acquired",
    "exact_success_dwell",
    "no_invalid_contacts",
    "all_samples_collision_free",
    "all_samples_bilateral_grasp",
    "all_samples_finite",
)
_SOURCE_SHA256_FIXTURE = {
    "scripts/tools/validate_franka_rj45_pick_insert_resets.py": "1" * 64,
    "scripts/tools/_franka_rj45_reset_tools.py": "2" * 64,
    "uv.lock": "3" * 64,
    "source/isaaclab/isaaclab/__init__.py": "4" * 64,
    "source/isaaclab_newton/isaaclab_newton/__init__.py": "5" * 64,
    "source/isaaclab_contrib/isaaclab_contrib/__init__.py": "6" * 64,
    "source/isaaclab_assets/isaaclab_assets/__init__.py": "7" * 64,
    "source/isaaclab_tasks/isaaclab_tasks/__init__.py": "8" * 64,
}


def test_pick_insert_artifact_schema_versions_remain_stable():
    assert FRANKA_RJ45_PICK_INSERT_RESET_DATASET_SCHEMA_VERSION == 3
    assert FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_SCHEMA_VERSION == 5


def test_pick_insert_validation_policy_binds_incremental_cartesian_recovery_without_schema_bump():
    assert FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY["contract_version"] == 4
    assert (
        FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY["scripted_recovery"] == PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY
    )
    assert PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["contract_version"] == 3
    assert PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["motion_policy"] == "incremental-cartesian-c2-v3"
    assert (
        PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["initial_route_endpoint_policy"]
        == "canonical-goal-c2-stop-before-bounded-compensation"
    )
    assert PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["phase_route_modes"] == {
        "0": "insertion-corridor",
        "1": "insertion-corridor",
        "2": "clearance-via-preinsert",
        "3": "clearance-via-preinsert",
        "4": "clearance-via-preinsert",
        "5": "clearance-via-preinsert",
    }
    assert PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["clearance_route"]["plug_pose_waypoints"] == (
        "vertical-lift",
        "high-midpoint",
        "overhead-preinsert",
        "preinsert",
        "canonical-goal",
    )
    planning = PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["planning"]
    assert planning["maximum_translation_step_m"] == 0.002
    assert planning["maximum_rotation_step_rad"] == math.radians(2.0)
    assert planning["maximum_raw_ik_joint_step_rad"] == 0.02
    assert planning["maximum_commanded_joint_step_rad"] == 0.02
    assert planning["maximum_waypoints"] == 430
    assert planning["maximum_waypoints_scope"] == "post-densification-executed-global-unique-knots"
    assert planning["endpoint_command_bias_policy"] == "linear-start-to-goal-over-global-unique-route-knots"
    assert planning["exact_start_target"] is True
    assert planning["exact_canonical_endpoint"] is True
    assert planning["command_interval_densification"] == "deterministic-collinear-joint-subknots"
    assert planning["densification_step_limit_rad"] == 0.02
    assert planning["compensation_bias_policy"] == "constant-start-bias"
    assert PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["compensation"] == {
        "subdivide_with_same_cartesian_policy": True,
        "proactive_overtravel": False,
        "trigger": "settled-goal-error-above-tolerance",
        "canonical_return_uses_same_endpoint_bias_blend": True,
    }
    assert PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["transient_cable_speed"]["rejection_gate"] is False


def test_pick_insert_post_settle_limits_are_training_oriented_but_explicit():
    assert PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M == PICK_INSERT_RESET_MAX_BODY_EXCURSION_M == 0.012
    assert PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD == 0.10


def test_public_validation_source_helper_hashes_the_complete_canonical_closure(tmp_path):
    roots = (
        "source/isaaclab/isaaclab",
        "source/isaaclab_newton/isaaclab_newton",
        "source/isaaclab_contrib/isaaclab_contrib",
        "source/isaaclab_assets/isaaclab_assets",
        "source/isaaclab_tasks/isaaclab_tasks",
    )
    for index, relative_root in enumerate(roots):
        source = tmp_path / relative_root / "module.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"VALUE = {index}\n", encoding="utf-8")
    for relative_name in (
        "scripts/tools/validate_franka_rj45_pick_insert_resets.py",
        "scripts/tools/_franka_rj45_reset_tools.py",
        "scripts/tools/generate_franka_rj45_dual_rack_reset_dataset.py",
        "scripts/tools/generate_franka_rj45_gb300_reset_dataset.py",
        "scripts/tools/generate_franka_rj45_pick_insert_reset_dataset.py",
        "scripts/tools/validate_franka_rj45_pick_insert_fast_resets.py",
        "uv.lock",
    ):
        source = tmp_path / relative_name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(relative_name, encoding="utf-8")

    first = franka_rj45_validation_source_sha256(str(tmp_path))
    changed = tmp_path / roots[-1] / "module.py"
    changed.write_text("VALUE = 'changed'\n", encoding="utf-8")
    second = franka_rj45_validation_source_sha256(tmp_path)
    fast = franka_rj45_validation_source_sha256(tmp_path, include_fast_validator=True)

    assert len(first) == len(roots) + 3
    assert set(first) == set(second)
    assert all(len(digest) == 64 for digest in first.values())
    assert first[changed.relative_to(tmp_path).as_posix()] != second[changed.relative_to(tmp_path).as_posix()]
    assert set(fast) == {
        *first,
        "scripts/tools/generate_franka_rj45_dual_rack_reset_dataset.py",
        "scripts/tools/generate_franka_rj45_gb300_reset_dataset.py",
        "scripts/tools/generate_franka_rj45_pick_insert_reset_dataset.py",
        "scripts/tools/validate_franka_rj45_pick_insert_fast_resets.py",
    }


def test_public_validation_source_helper_fails_closed_on_partial_checkout(tmp_path):
    with pytest.raises(FileNotFoundError, match="complete Franka RJ45 validation source closure"):
        franka_rj45_validation_source_sha256(tmp_path)


def _payload(*, task_body_count: int = 4) -> dict:
    row_count = len(PICK_INSERT_RESET_PHASE_IDS)
    states = {
        "arm_joint_position": torch.zeros((row_count, 7), dtype=torch.float32),
        "arm_joint_target": torch.zeros((row_count, 7), dtype=torch.float32),
        "arm_joint_velocity": torch.zeros((row_count, 7), dtype=torch.float32),
        "finger_joint_position": torch.zeros((row_count, 2), dtype=torch.float32),
        "finger_joint_velocity": torch.zeros((row_count, 2), dtype=torch.float32),
        "finger_joint_target": torch.zeros((row_count, 2), dtype=torch.float32),
        "task_body_pose": torch.zeros((row_count, task_body_count, 7), dtype=torch.float32),
        "task_body_previous_pose": torch.zeros((row_count, task_body_count, 7), dtype=torch.float32),
        "task_body_coupling_previous_pose": torch.zeros((row_count, task_body_count, 7), dtype=torch.float32),
        "task_body_velocity": torch.zeros((row_count, task_body_count, 6), dtype=torch.float32),
        "goal_task_body_pose": torch.zeros((row_count, task_body_count, 7), dtype=torch.float32),
        "goal_arm_joint_target": torch.zeros((row_count, 7), dtype=torch.float32),
        "phase": torch.tensor(PICK_INSERT_RESET_PHASE_IDS, dtype=torch.int64),
        "starts_grasped": torch.tensor([True, True, True, True, False, False], dtype=torch.bool),
        "difficulty": torch.linspace(0.0, 1.0, row_count, dtype=torch.float32),
        "initial_goal_error": torch.ones(row_count, dtype=torch.float32),
        "initial_tcp_grasp_distance": torch.ones(row_count, dtype=torch.float32),
        "progress_threshold": torch.full((row_count,), 0.001, dtype=torch.float32),
    }
    states["task_body_pose"][..., 6] = 1.0
    states["task_body_previous_pose"][..., 6] = 1.0
    states["task_body_coupling_previous_pose"][..., 6] = 1.0
    states["goal_task_body_pose"][..., 6] = 1.0
    goal_state = {
        "arm_joint_position": torch.zeros(7, dtype=torch.float32),
        "arm_joint_target": torch.zeros(7, dtype=torch.float32),
        "arm_joint_velocity": torch.zeros(7, dtype=torch.float32),
        "finger_joint_position": torch.zeros(2, dtype=torch.float32),
        "finger_joint_velocity": torch.zeros(2, dtype=torch.float32),
        "finger_joint_target": torch.zeros(2, dtype=torch.float32),
        "task_body_pose": torch.zeros((task_body_count, 7), dtype=torch.float32),
        "task_body_previous_pose": torch.zeros((task_body_count, 7), dtype=torch.float32),
        "task_body_coupling_previous_pose": torch.zeros((task_body_count, 7), dtype=torch.float32),
        "task_body_velocity": torch.zeros((task_body_count, 6), dtype=torch.float32),
    }
    goal_state["task_body_pose"][..., 6] = 1.0
    goal_state["task_body_previous_pose"][..., 6] = 1.0
    goal_state["task_body_coupling_previous_pose"][..., 6] = 1.0
    payload = {
        "format": FRANKA_RJ45_PICK_INSERT_RESET_DATASET_FORMAT,
        "schema_version": FRANKA_RJ45_PICK_INSERT_RESET_DATASET_SCHEMA_VERSION,
        "metadata": {
            "task_contract": {
                "task_body_count": task_body_count,
                "task_body_order": tuple(f"body_{index}" for index in range(task_body_count)),
                "reset_state_representation": {
                    "contract_version": 2,
                    "task_body_pose_frame": "environment-local-xyzw",
                    "task_body_velocity_frame": "world-linear-angular",
                    "vbd_entry_name": "rj45",
                    "vbd_body_order_source": "task_body_order",
                    "vbd_previous_pose_field": "task_body_previous_pose",
                    "vbd_coupling_previous_pose_field": "task_body_coupling_previous_pose",
                    "vbd_pose_history_frame": "environment-local-xyzw",
                    "restore_semantics": "deferred-one-shot-after-input-and-proxy-rebaseline-before-first-vbd-solve",
                    "preserved_input_task_body_range_half_open": (3, task_body_count - 1),
                    "preserved_input_semantics": "scatter-history-without-pose-delta-velocity-injection-or-rewind",
                },
                "asset": "rj45",
                "external_assets": franka_rj45_asset_contract(),
                "robot": {"reset_control_convention": franka_pick_insert_control_contract()},
                "pick_insert": {"finger_closed_position": 0.0},
                "rj45_physics": {
                    "grasp_proxy_raw_friction": 100.0,
                    "grasp_contact_effective_friction": 10.0,
                },
                "validation_geometry": {"success_max_plug_speed": 0.10},
                "runtime_physics_versions": {"newton": "1.0", "warp-lang": "2.0"},
            }
        },
        "states": states,
        "goal_state": goal_state,
    }
    payload["content_sha256"] = reset_dataset_content_digest(payload)
    return payload


def _refresh_digest(payload: dict) -> None:
    payload["content_sha256"] = reset_dataset_content_digest(payload)


def _full_pick_diversity_fixture() -> tuple[dict[str, torch.Tensor], dict]:
    row_count = 96
    fraction = torch.linspace(0.0, 1.0, row_count, dtype=torch.float32)
    socket_yaw = -0.4 + 0.8 * fraction
    plug_yaw = -1.1 + 2.2 * fraction
    task_pose = torch.zeros((row_count, 2, 7), dtype=torch.float32)
    task_pose[:, 0, 0] = 0.52 + 0.14 * fraction
    task_pose[:, 0, 1] = 0.08 + 0.14 * fraction.flip(0)
    task_pose[:, 0, 5] = torch.sin(0.5 * socket_yaw)
    task_pose[:, 0, 6] = torch.cos(0.5 * socket_yaw)
    task_pose[:, 1, 0] = 0.34 + 0.23 * fraction.flip(0)
    task_pose[:, 1, 1] = -0.20 + 0.185 * fraction
    task_pose[:, 1, 5] = torch.sin(0.5 * plug_yaw)
    task_pose[:, 1, 6] = torch.cos(0.5 * plug_yaw)
    arm_position = torch.stack(
        [(-0.12 + 0.24 * fraction.roll(joint_index)) for joint_index in range(7)],
        dim=-1,
    )
    states = {
        "phase": torch.full((row_count,), 5, dtype=torch.int64),
        "task_body_pose": task_pose,
        "arm_joint_position": arm_position,
        "initial_tcp_grasp_distance": torch.linspace(0.10, 0.25, row_count, dtype=torch.float32),
    }
    task_contract = {
        "task_body_order": ("socket", "plug"),
        "pick_insert": {
            "reset_dataset_rows_per_phase": row_count,
            "arm_reset_joint_noise": 0.12,
            "socket_position_lower": (0.52, 0.08, 0.0),
            "socket_position_upper": (0.66, 0.22, 0.0),
            "socket_yaw_range": (-0.436, 0.436),
            "pickup_position_lower": (0.34, -0.20, 0.0105),
            "pickup_position_upper": (0.57, -0.015, 0.0145),
            "pickup_yaw_range": (-1.22, 1.22),
            "full_pick_diversity": {
                "round_decimals": 4,
                "minimum_unique_socket_rows": 90,
                "minimum_unique_plug_rows": 90,
                "minimum_unique_arm_rows": 90,
                "minimum_socket_span_fraction": 0.60,
                "minimum_pickup_span_fraction": 0.60,
                "minimum_arm_joint_span_fraction": 0.50,
                "minimum_tcp_grasp_distance_span_m": 0.10,
            },
        },
    }
    return states, task_contract


def _reset_replay_fixture(phase: int, *, task_body_count: int = 4) -> dict:
    starts_grasped = phase <= 3
    return {
        "simulation_time_s": PICK_INSERT_RESET_REPLAY_DURATION_S,
        "simulation_steps": PICK_INSERT_RESET_REPLAY_POST_STEP_SAMPLES,
        "post_step_samples": PICK_INSERT_RESET_REPLAY_POST_STEP_SAMPLES,
        "required_simulation_time_s": PICK_INSERT_RESET_REPLAY_DURATION_S,
        "required_post_step_samples": PICK_INSERT_RESET_REPLAY_POST_STEP_SAMPLES,
        "starts_grasped": starts_grasped,
        "contact_expectation": "bilateral-proxy" if starts_grasped else "zero-proxy",
        "vbd_pose_history_restore_queued": True,
        "vbd_pose_history_pending_at_queue": True,
        "vbd_previous_pose_queued": True,
        "vbd_coupling_previous_pose_queued": True,
        "vbd_pose_history_applied_exactly_once": True,
        "vbd_pose_history_failed": False,
        "vbd_pose_history_superseded": False,
        "vbd_pose_history_pending_after_first_solve": False,
        "vbd_pose_history_application_count_delta": 1,
        "vbd_pose_history_expected_body_count": task_body_count,
        "vbd_pose_history_body_application_count_delta": task_body_count,
        "vbd_pose_history_generation": 1,
        "vbd_pose_history_body_order_exact": True,
        "vbd_pose_history_world_order_exact": True,
        "vbd_pose_history_entry_name": "rj45",
        "vbd_pose_history_body_count": task_body_count,
        "stored_state_finite": True,
        "stored_task_state_finite_and_normalized": True,
        "stored_drive_disabled": True,
        "stored_maximum_cable_speed_m_s": 0.01,
        "stored_maximum_arm_joint_speed_rad_s": 0.02,
        "stored_maximum_finger_joint_speed_m_s": 0.01,
        "stored_maximum_arm_target_tracking_error_rad": 0.02,
        "stored_arm_target_tracking_error_by_joint_rad": [0.02] * 7,
        "stored_arm_target_tracking_bounded": True,
        "all_post_step_state_finite": True,
        "all_post_step_task_state_finite_and_normalized": True,
        "all_post_step_collision_free": True,
        "all_post_step_drive_disabled": True,
        "all_post_step_expected_contact_state": True,
        "all_post_step_bilateral_grasp": starts_grasped,
        "all_post_step_proxy_bilateral_contact": starts_grasped,
        "all_post_step_zero_proxy_contacts": not starts_grasped,
        "all_post_step_arm_target_tracking_bounded": True,
        "maximum_body_excursion_m": 0.001,
        "maximum_plug_excursion_m": 0.001,
        "maximum_socket_excursion_m": 0.0,
        "maximum_post_step_cable_speed_m_s": 0.02,
        "maximum_post_step_arm_joint_speed_rad_s": 0.03,
        "maximum_post_step_finger_joint_speed_m_s": 0.02,
        "maximum_post_step_arm_target_tracking_error_rad": 0.02,
        "maximum_arm_target_tracking_error_by_joint_rad": [0.02] * 7,
        "arm_target_semantics": "persistent-absolute",
        "arm_target_tracking_limits_rad": list(PICK_INSERT_ARM_TARGET_TRACKING_LIMITS),
        "maximum_arm_target_drift_rad": 0.0,
        "absolute_target_stable": True,
        "final_cable_speed_m_s": 0.01,
        "final_arm_joint_speed_rad_s": 0.02,
        "final_finger_joint_speed_m_s": 0.01,
        "minimum_left_proxy_contact_count": 1 if starts_grasped else 0,
        "minimum_right_proxy_contact_count": 1 if starts_grasped else 0,
        "maximum_left_proxy_contact_count": 2 if starts_grasped else 0,
        "maximum_right_proxy_contact_count": 2 if starts_grasped else 0,
        "maximum_invalid_contact_count": 0,
        "no_contact_overflow": True,
        "invalid_contact_pairs": [],
        "maximum_arm_target_clamp_delta_rad": 0.0,
        "zero_action_unclamped": True,
        "maximum_allowed_arm_joint_speed_rad_s": PICK_INSERT_RESET_MAX_ARM_JOINT_SPEED_RAD_S,
        "maximum_allowed_finger_joint_speed_m_s": PICK_INSERT_RESET_MAX_FINGER_JOINT_SPEED_M_S,
        "maximum_allowed_cable_speed_m_s": PICK_INSERT_RESET_MAX_CABLE_SPEED_M_S,
        "maximum_allowed_body_excursion_m": PICK_INSERT_RESET_MAX_BODY_EXCURSION_M,
        "maximum_allowed_plug_excursion_m": PICK_INSERT_RESET_MAX_PLUG_EXCURSION_M,
        "maximum_allowed_socket_excursion_m": PICK_INSERT_RESET_MAX_SOCKET_EXCURSION_M,
        "maximum_allowed_arm_target_clamp_delta_rad": PICK_INSERT_RESET_MAX_ARM_TARGET_CLAMP_DELTA_RAD,
    }


def _validation_cfg_fixture() -> dict:
    return {
        "seed": 2027,
        "quick": False,
        "sample_count": None,
        "ik_sampler": "none",
        "ik_seed_count": 1,
        "ik_iterations": 160,
        "ik_noise_std": 0.0,
        "goal_replay_s": 60.0,
        "row_settle_s": PICK_INSERT_RESET_REPLAY_DURATION_S,
        "grasp_approach_s": 2.5,
        "grasp_close_s": 0.8,
        "grasp_hold_s": 1.5,
        "grasp_post_contact_settle_s": 1.0,
        "grasp_open_clearance_m": 0.045,
        "grasp_route_world_height_m": 0.22,
        "grasp_route_maximum_translation_step_m": 0.05,
        "grasp_coarse_descent_step_m": 0.005,
        "grasp_near_descent_step_m": 0.001,
        "grasp_descent_waypoint_motion_s": 0.15,
        "grasp_descent_waypoint_settle_s": 1.0 / 30.0,
        "grasp_descent_tracking_recovery_s": 0.05,
        "grasp_near_correction_step_m": 0.001,
        "grasp_near_correction_max_iterations": 3,
        "grasp_near_maximum_raw_ik_joint_step_rad": 0.02,
        "grasp_clearance_calibration_step_m": 0.001,
        "grasp_clearance_calibration_max_iterations": 24,
        "maximum_open_approach_plug_drift_m": 5.0e-4,
        "finger_open_position": 0.04,
        "finger_closed_target": 0.0,
        "recovery_motion_s": 4.0,
        "recovery_settle_s": 0.75,
        "recovery_compensation_iterations": 6,
        "recovery_compensation_tolerance_m": 0.0015,
        "maximum_ik_joint_step_rad": 0.6,
        "maximum_goal_socket_drift_m": PICK_INSERT_GOAL_MAX_SOCKET_DRIFT_M,
        "maximum_goal_body_drift_m": PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M,
        "maximum_goal_cable_speed_m_s": PICK_INSERT_GOAL_MAX_CABLE_SPEED_M_S,
        "maximum_goal_arm_joint_speed_rad_s": PICK_INSERT_GOAL_MAX_ARM_JOINT_SPEED_RAD_S,
        "maximum_goal_finger_joint_speed_m_s": PICK_INSERT_GOAL_MAX_FINGER_JOINT_SPEED_M_S,
        "maximum_goal_authored_seat_error_m": 0.001,
        "maximum_goal_authored_plug_angle_rad": math.radians(3.0),
        "maximum_goal_plug_relative_latch_angle_rad": PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD,
        "maximum_row_socket_drift_m": PICK_INSERT_RESET_MAX_SOCKET_EXCURSION_M,
        "maximum_row_plug_drift_m": PICK_INSERT_RESET_MAX_PLUG_EXCURSION_M,
        "maximum_row_body_drift_m": PICK_INSERT_RESET_MAX_BODY_EXCURSION_M,
        "maximum_row_cable_speed_m_s": PICK_INSERT_RESET_MAX_CABLE_SPEED_M_S,
        "maximum_row_arm_joint_speed_rad_s": PICK_INSERT_RESET_MAX_ARM_JOINT_SPEED_RAD_S,
        "maximum_row_finger_joint_speed_m_s": PICK_INSERT_RESET_MAX_FINGER_JOINT_SPEED_M_S,
    }


def _offline_diversity_fixture() -> dict:
    return {
        "phase": 5,
        "row_count": 90,
        "required_row_count": 90,
        "round_decimals": 4,
        "unique_socket_rows": 90,
        "unique_plug_rows": 90,
        "unique_arm_rows": 90,
        "required_unique_socket_rows": 90,
        "required_unique_plug_rows": 90,
        "required_unique_arm_rows": 90,
        "socket_xy_span_m": [0.10, 0.10],
        "required_socket_xy_span_m": [0.05, 0.05],
        "socket_yaw_span_rad": 0.50,
        "required_socket_yaw_span_rad": 0.25,
        "plug_pickup_xy_span_m": [0.15, 0.15],
        "required_plug_pickup_xy_span_m": [0.05, 0.05],
        "plug_pickup_yaw_span_rad": 1.0,
        "required_plug_pickup_yaw_span_rad": 0.50,
        "arm_joint_span_rad": [0.20] * 7,
        "required_each_arm_joint_span_rad": 0.10,
        "initial_tcp_grasp_distance_span_m": 0.15,
        "required_initial_tcp_grasp_distance_span_m": 0.10,
        "passed": True,
        "failures": [],
    }


def _started_acquisition_fixture() -> dict:
    return {
        "started_with_physical_bilateral_grasp": [True],
        "last_arm_target": [[0.0] * 7],
        "last_finger_target": [[0.0] * 2],
    }


def _open_acquisition_fixture() -> dict:
    return {
        "open_clearance_above_cfg_target_m": 0.045,
        "route_world_height_m": 0.22,
        "approach_abort_reason": None,
        "approach_samples": 1,
        "clearance_approach_valid": [True],
        "clearance_tcp_error_m": [0.001],
        "open_descent_valid": [True],
        "maximum_open_descent_tcp_error_m": [0.001],
        "open_approach_all_samples_collision_free": [True],
        "open_approach_all_samples_zero_proxy_contacts": [True],
        "open_approach_all_samples_finite": [True],
        "open_approach_all_samples_drives_disabled": [True],
        "open_approach_maximum_plug_drift_m": [0.0001],
        "open_approach_maximum_left_proxy_contacts": [0],
        "open_approach_maximum_right_proxy_contacts": [0],
        "open_approach_any_contact_overflow": False,
        "open_approach_invalid_pairs": [],
        "contact_preclose_invalid_contacts": [0],
        "contact_preclose_tcp_error_m": [0.001],
        "maximum_tcp_distance_m": 0.01,
        "minimum_bilateral_deflection_m": 0.001,
        "left_proxy_contacts": [1],
        "right_proxy_contacts": [1],
        "invalid_contacts": [0],
        "post_contact_settle": {
            "all_samples_finite": [True],
            "all_samples_collision_free": [True],
            "all_samples_bilateral_proxy_contact": [True],
            "all_samples_drives_disabled": [True],
            "any_contact_overflow": False,
            "invalid_contact_pairs": [],
            "maximum_cable_speed_m_s": [0.001],
            "maximum_plug_linear_speed_m_s": [0.001],
            "maximum_plug_angular_speed_rad_s": [0.001],
            "final_cable_speed_m_s": [0.001],
            "final_plug_linear_speed_m_s": [0.001],
            "final_plug_angular_speed_rad_s": [0.001],
            "final_arm_joint_speed_rad_s": [0.001],
            "final_finger_joint_speed_m_s": [0.001],
        },
        "lane_failure_masks": {},
        "last_arm_target": [[0.0] * 7],
        "last_finger_target": [[0.0] * 2],
    }


def _row_metrics_fixture(phase: int, row_id: int, reset_replay: dict) -> dict:
    phase_five_index = max(0, row_id - 5)
    fraction = phase_five_index / 89.0 if phase == 5 else 0.0
    return {
        "initial_goal_error_artifact": 0.1,
        "initial_goal_error_replayed": 0.1,
        "initial_goal_error_matches": True,
        "initial_tcp_distance_artifact_m": 0.11,
        "initial_tcp_distance_replayed_m": 0.11,
        "initial_tcp_distance_matches": True,
        "initial_tcp_xyz_replayed_m": [0.06 * fraction, 0.07 * fraction, 0.08 * fraction],
        "settle_socket_drift_m": reset_replay["maximum_socket_excursion_m"],
        "settle_plug_drift_m": reset_replay["maximum_plug_excursion_m"],
        "settle_max_body_drift_m": reset_replay["maximum_body_excursion_m"],
        "capture_max_cable_speed_m_s": reset_replay["stored_maximum_cable_speed_m_s"],
        "settled_max_cable_speed_m_s": reset_replay["final_cable_speed_m_s"],
        "maximum_reset_arm_target_clamp_delta_rad": reset_replay["maximum_arm_target_clamp_delta_rad"],
        "maximum_reset_arm_target_drift_rad": reset_replay["maximum_arm_target_drift_rad"],
        "reset_maximum_invalid_contacts": reset_replay["maximum_invalid_contact_count"],
        "recovery_invalid_contacts": 0,
        "recovery_goal_error": 0.001,
        "recovery_plug_speed": 0.01,
        "recovery_maximum_body_excursion_m": 0.001,
        "recovery_maximum_cable_linear_speed_m_s": 0.001,
        "recovery_maximum_arm_joint_speed_rad_s": 0.01,
        "recovery_maximum_finger_joint_speed_m_s": 0.01,
        "acquisition": _started_acquisition_fixture() if phase <= 3 else _open_acquisition_fixture(),
    }


def _validation_row_fixture(row_id: int, phase: int, *, task_body_count: int) -> dict:
    reset_replay = _reset_replay_fixture(phase, task_body_count=task_body_count)
    return {
        "row_id": row_id,
        "phase": phase,
        "passed": True,
        "checks": {name: True for name in _ROW_CHECK_NAMES},
        "oracle": {name: True for name in _ORACLE_EVIDENCE_NAMES},
        "reset_replay": reset_replay,
        "metrics": _row_metrics_fixture(phase, row_id, reset_replay),
    }


def _expand_validation_payload(payload: dict) -> None:
    """Give schema-5 report fixtures enough phase-5 rows for raw diversity checks."""
    if int((payload["states"]["phase"] == 5).sum()) >= 90:
        return
    source_rows = torch.tensor([0, 1, 2, 3, 4, *([5] * 90)], dtype=torch.int64)
    payload["states"] = {name: value[source_rows].clone() for name, value in payload["states"].items()}
    _refresh_digest(payload)


def _validation_report(payload: dict) -> dict:
    _expand_validation_payload(payload)
    phases = payload["states"]["phase"].tolist()
    offline_diversity = _offline_diversity_fixture()
    report = {
        "format": FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_FORMAT,
        "schema_version": FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_SCHEMA_VERSION,
        "created_utc": "2026-08-17T12:00:00+00:00",
        "artifact_content_sha256": payload["content_sha256"],
        "task_contract": payload["metadata"]["task_contract"],
        "validation_cfg": _validation_cfg_fixture(),
        "physical_contract": {
            "finger_closed_target_m": 0.0,
            "live_finger_close_position_m": 0.0,
            "configured_grasp_proxy_raw_friction": 100.0,
            "live_grasp_proxy_raw_friction": 100.0,
            "effective_finger_proxy_friction": 10.0,
            "success_max_plug_speed": 0.10,
        },
        "physics_versions": {"newton": "1.0", "warp": "2.0", "isaaclab": "3.0"},
        "validation_policy": json.loads(json.dumps(FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY)),
        "source_sha256": dict(_SOURCE_SHA256_FIXTURE),
        "asset_closure": json.loads(json.dumps(payload["metadata"]["task_contract"]["external_assets"])),
        "ik_solve_call_count": 1,
        "passed": True,
        "evidence_complete": True,
        "full_dataset_replay": True,
        "quick": False,
        "selected_row_count": len(phases),
        "dataset_row_count": len(phases),
        "selected_row_ids": list(range(len(phases))),
        "failed_row_ids": [],
        "phase_counts": [phases.count(phase) for phase in PICK_INSERT_RESET_PHASE_IDS],
        "goal_replay": {
            "passed": True,
            "drive_disabled": True,
            "vbd_pose_history_restore_queued": True,
            "vbd_pose_history_pending_at_queue": True,
            "vbd_previous_pose_queued": True,
            "vbd_coupling_previous_pose_queued": True,
            "vbd_pose_history_applied_exactly_once": True,
            "vbd_pose_history_failed": False,
            "vbd_pose_history_superseded": False,
            "vbd_pose_history_pending_after_first_solve": False,
            "vbd_pose_history_minimum_application_count_delta": 1,
            "vbd_pose_history_maximum_application_count_delta": 1,
            "vbd_pose_history_expected_body_count": payload["metadata"]["task_contract"]["task_body_count"],
            "vbd_pose_history_minimum_body_application_count_delta": payload["metadata"]["task_contract"][
                "task_body_count"
            ],
            "vbd_pose_history_maximum_body_application_count_delta": payload["metadata"]["task_contract"][
                "task_body_count"
            ],
            "vbd_pose_history_generation": 1,
            "vbd_pose_history_body_order_exact": True,
            "vbd_pose_history_world_order_exact": True,
            "vbd_pose_history_entry_name": "rj45",
            "vbd_pose_history_body_count": payload["metadata"]["task_contract"]["task_body_count"],
            "socket_stable": True,
            "whole_cable_stable": True,
            "exact_runtime_success_dwell": True,
            "simulation_steps": 1800,
            "simulation_time_s": 60.0,
            "contact_count_after_history_reset": 0,
            "all_samples_collision_free": True,
            "all_samples_bilateral_grasp": True,
            "all_samples_proxy_bilateral_contact": True,
            "all_samples_finite": True,
            "minimum_left_proxy_contact_count": 1,
            "minimum_right_proxy_contact_count": 1,
            "maximum_invalid_contact_count": 0,
            "any_contact_overflow": False,
            "no_contact_overflow": True,
            "sampled_invalid_contact_pairs": [],
            "stored_capture_exact_success": True,
            "all_post_step_exact_success": True,
            "required_dwell_steps": 1,
            "final_consecutive_steps": [1],
            "collision_valid": True,
            "closed_bilateral_grasp": True,
            "maximum_socket_drift_m": 5.0e-6,
            "maximum_task_body_drift_m": 0.002,
            "sampled_maximum_task_body_excursion_m": 0.003,
            "maximum_start_cable_speed_m_s": 0.004,
            "maximum_final_cable_speed_m_s": 0.005,
            "sampled_maximum_cable_speed_m_s": 0.006,
            "sampled_maximum_arm_joint_speed_rad_s": 0.05,
            "sampled_maximum_finger_joint_speed_m_s": 0.02,
            "maximum_allowed_socket_drift_m": PICK_INSERT_GOAL_MAX_SOCKET_DRIFT_M,
            "maximum_allowed_task_body_drift_m": PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M,
            "maximum_allowed_cable_speed_m_s": PICK_INSERT_GOAL_MAX_CABLE_SPEED_M_S,
            "maximum_allowed_arm_joint_speed_rad_s": PICK_INSERT_GOAL_MAX_ARM_JOINT_SPEED_RAD_S,
            "maximum_allowed_finger_joint_speed_m_s": PICK_INSERT_GOAL_MAX_FINGER_JOINT_SPEED_M_S,
            "robot_equilibrium": True,
            "zero_action_unclamped": True,
            "maximum_arm_target_clamp_delta_rad": 0.0,
            "controller_semantics": "persistent-absolute",
            "absolute_target_stable": True,
            "maximum_arm_target_drift_rad": 0.0,
            "all_samples_arm_target_tracking_bounded": True,
            "maximum_arm_target_tracking_error_by_joint_rad": [0.02] * 7,
            "authored_goal_geometry_valid": True,
            "maximum_allowed_authored_seat_error_m": 0.001,
            "maximum_allowed_authored_plug_angle_rad": math.radians(3.0),
            "maximum_allowed_plug_relative_latch_angle_rad": PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD,
            "maximum_stored_authored_seat_error_m": 0.0002,
            "maximum_final_authored_seat_error_m": 0.0003,
            "maximum_stored_authored_plug_angle_rad": 0.01,
            "maximum_final_authored_plug_angle_rad": 0.02,
            "maximum_stored_plug_relative_latch_angle_rad": 0.01,
            "maximum_final_plug_relative_latch_angle_rad": 0.02,
        },
        "full_pick_diversity": {
            "passed": True,
            "skipped_due_to_quick": False,
            "required_phase": 5,
            "required_rows": 90,
            "observed_rows": 90,
            "round_decimals": 4,
            "minimum_unique_tcp_positions": 90,
            "unique_tcp_positions": 90,
            "minimum_tcp_xyz_span_m": [0.05, 0.05, 0.05],
            "tcp_xyz_min_m": [0.0, 0.0, 0.0],
            "tcp_xyz_max_m": [0.06, 0.07, 0.08],
            "tcp_xyz_span_m": [0.06, 0.07, 0.08],
            "minimum_tcp_to_grasp_distance_m": 0.10,
            "observed_minimum_tcp_to_grasp_distance_m": 0.11,
            "offline_artifact": offline_diversity,
            "failures": [],
        },
        "rows": [
            _validation_row_fixture(
                row_id,
                phase,
                task_body_count=payload["metadata"]["task_contract"]["task_body_count"],
            )
            for row_id, phase in enumerate(phases)
        ],
    }
    report["content_sha256"] = reset_validation_report_content_digest(report)
    return report


def _validate_report(report: dict, payload: dict) -> dict[str, bool]:
    return reset_validation_report_validate_runtime(
        report,
        expected_content_sha256=payload["content_sha256"],
        expected_row_count=len(payload["states"]["phase"]),
        expected_phases=payload["states"]["phase"],
        expected_task_contract=payload["metadata"]["task_contract"],
        expected_validation_policy=FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY,
        expected_source_sha256=_SOURCE_SHA256_FIXTURE,
        expected_asset_closure=payload["metadata"]["task_contract"]["external_assets"],
        expected_full_pick_diversity=_offline_diversity_fixture(),
    )


def test_valid_artifact_uses_dynamic_task_body_count_and_independent_schema():
    payload = _payload(task_body_count=4)

    metadata, states, goal_state = reset_dataset_validate_runtime(
        payload,
        expected_content_sha256=payload["content_sha256"],
        expected_task_contract=payload["metadata"]["task_contract"],
    )

    assert metadata is payload["metadata"]
    assert tuple(states) == RESET_DATASET_STATE_NAMES
    assert tuple(goal_state) == RESET_DATASET_GOAL_STATE_NAMES
    assert states["task_body_pose"].shape == (6, 4, 7)
    assert states["task_body_previous_pose"].shape == (6, 4, 7)
    assert states["task_body_coupling_previous_pose"].shape == (6, 4, 7)
    assert states["goal_task_body_pose"].shape == (6, 4, 7)
    assert states["goal_arm_joint_target"].shape == (6, 7)
    assert states["starts_grasped"].dtype == torch.bool
    assert goal_state["task_body_pose"].shape == (4, 7)
    assert goal_state["task_body_previous_pose"].shape == (4, 7)
    assert goal_state["task_body_coupling_previous_pose"].shape == (4, 7)


def test_artifact_round_trips_through_restricted_torch_load(tmp_path):
    payload = _payload(task_body_count=3)
    artifact = tmp_path / "pick_insert_resets.pt"
    torch.save(payload, artifact)

    loaded = torch.load(artifact, map_location="cpu", weights_only=True)
    _, states, goal_state = reset_dataset_validate_runtime(loaded)

    assert states["task_body_pose"].shape == (6, 3, 7)
    assert goal_state["task_body_pose"].shape == (3, 7)


@pytest.mark.parametrize(
    ("container", "field", "index"),
    (
        ("states", "arm_joint_position", (0, 0)),
        ("states", "task_body_previous_pose", (0, 0, 0)),
        ("states", "task_body_coupling_previous_pose", (0, 0, 0)),
        ("goal_state", "task_body_previous_pose", (0, 0)),
        ("goal_state", "task_body_coupling_previous_pose", (0, 0)),
    ),
)
def test_content_mutation_is_rejected(container: str, field: str, index: tuple[int, ...]):
    payload = _payload()
    payload[container][field][index] = 0.25

    with pytest.raises(ValueError, match="content digest does not match"):
        reset_dataset_validate_runtime(payload)


@pytest.mark.parametrize("task_body_count", (0, True))
def test_task_body_count_must_be_a_dynamic_positive_integer(task_body_count: object):
    payload = _payload()
    payload["metadata"]["task_contract"]["task_body_count"] = task_body_count
    _refresh_digest(payload)

    with pytest.raises(ValueError, match="task_body_count must be a positive integer"):
        reset_dataset_validate_runtime(payload)


def test_runtime_task_body_count_must_match_artifact_contract():
    payload = _payload(task_body_count=4)

    with pytest.raises(ValueError, match="task contract does not exactly match"):
        reset_dataset_validate_runtime(payload, expected_task_contract={"task_body_count": 5})


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("missing_representation", "reset_state_representation"),
        ("duplicate_body_order", "uniquely name every task body"),
        ("wrong_entry", "incompatible with VBD replay"),
    ),
)
def test_artifact_requires_exact_vbd_history_representation(mutation: str, match: str):
    payload = _payload(task_body_count=4)
    contract = payload["metadata"]["task_contract"]
    if mutation == "missing_representation":
        del contract["reset_state_representation"]
    elif mutation == "duplicate_body_order":
        contract["task_body_order"] = ("same",) * 4
    else:
        contract["reset_state_representation"]["vbd_entry_name"] = "wrong"
    _refresh_digest(payload)

    with pytest.raises(ValueError, match=match):
        reset_dataset_validate_runtime(payload)


def test_runtime_task_contract_rejects_unexpected_artifact_fields():
    payload = _payload(task_body_count=4)
    expected_contract = dict(payload["metadata"]["task_contract"])
    payload["metadata"]["task_contract"]["unexpected"] = True
    _refresh_digest(payload)

    with pytest.raises(ValueError, match="task contract does not exactly match"):
        reset_dataset_validate_runtime(payload, expected_task_contract=expected_contract)


def test_runtime_task_contract_normalizes_json_tuple_and_list_representations():
    payload = _payload(task_body_count=4)
    payload["metadata"]["task_contract"]["body_order"] = ("socket", "plug", "latch", "cable")
    _refresh_digest(payload)
    expected_contract = dict(payload["metadata"]["task_contract"])
    expected_contract["body_order"] = ["socket", "plug", "latch", "cable"]
    expected_contract["task_body_order"] = list(expected_contract["task_body_order"])

    reset_dataset_validate_runtime(payload, expected_task_contract=expected_contract)


def test_artifact_requires_all_and_only_six_reset_phases():
    payload = _payload()
    payload["states"]["phase"][-1] = 4
    _refresh_digest(payload)

    with pytest.raises(ValueError, match="every reset phase exactly within \\[0, 5\\]"):
        reset_dataset_validate_runtime(payload)


def test_production_phase_count_gate_rejects_six_row_quick_artifact():
    phases = torch.tensor(PICK_INSERT_RESET_PHASE_IDS, dtype=torch.int64)

    with pytest.raises(ValueError, match="expected .*96.*got"):
        reset_dataset_validate_phase_row_counts(phases, expected_rows_per_phase=96)


def test_production_phase_count_gate_accepts_96_rows_per_phase_without_full_state_tensors():
    phases = torch.arange(len(PICK_INSERT_RESET_PHASE_IDS), dtype=torch.int64).repeat_interleave(96)

    counts = reset_dataset_validate_phase_row_counts(phases, expected_rows_per_phase=96)

    assert counts == (96,) * len(PICK_INSERT_RESET_PHASE_IDS)


def test_full_pick_diversity_gate_accepts_production_coverage():
    states, task_contract = _full_pick_diversity_fixture()

    evidence = reset_dataset_validate_full_pick_diversity(states, task_contract=task_contract)

    assert evidence["passed"] is True
    assert evidence["row_count"] == 96
    assert evidence["unique_socket_rows"] == 96
    assert evidence["unique_plug_rows"] == 96
    assert evidence["unique_arm_rows"] == 96


def test_full_pick_diversity_gate_rejects_repeated_rows_even_when_spans_are_wide():
    states, task_contract = _full_pick_diversity_fixture()
    source_rows = torch.linspace(0, 95, 10).round().long()
    repeated_rows = source_rows[torch.arange(96) % len(source_rows)]
    states["task_body_pose"] = states["task_body_pose"][repeated_rows].clone()
    states["arm_joint_position"] = states["arm_joint_position"][repeated_rows].clone()

    with pytest.raises(ValueError, match="unique_socket_rows.*unique_plug_rows.*unique_arm_rows"):
        reset_dataset_validate_full_pick_diversity(states, task_contract=task_contract)


def test_full_pick_diversity_gate_rejects_narrow_tcp_distance_span():
    states, task_contract = _full_pick_diversity_fixture()
    states["initial_tcp_grasp_distance"].fill_(0.15)

    with pytest.raises(ValueError, match="initial_tcp_grasp_distance_span"):
        reset_dataset_validate_full_pick_diversity(states, task_contract=task_contract)


@pytest.mark.parametrize(
    "pose_path",
    (
        ("states", "task_body_pose"),
        ("states", "task_body_previous_pose"),
        ("states", "task_body_coupling_previous_pose"),
        ("states", "goal_task_body_pose"),
        ("goal_state", "task_body_pose"),
        ("goal_state", "task_body_previous_pose"),
        ("goal_state", "task_body_coupling_previous_pose"),
    ),
)
def test_every_task_body_pose_requires_unit_quaternions(pose_path: tuple[str, str]):
    payload = _payload()
    pose = payload[pose_path[0]][pose_path[1]]
    pose.reshape(-1, 7)[0, 6] = 0.5
    _refresh_digest(payload)

    with pytest.raises(ValueError, match="quaternions must be normalized"):
        reset_dataset_validate_runtime(payload)


def test_nonfinite_distance_is_rejected_after_digest_verification():
    payload = _payload()
    payload["states"]["initial_tcp_grasp_distance"][0] = torch.nan
    _refresh_digest(payload)

    with pytest.raises(ValueError, match="initial_tcp_grasp_distance must contain only finite values"):
        reset_dataset_validate_runtime(payload)


def test_noncontiguous_artifact_tensor_is_rejected():
    payload = _payload()
    payload["states"]["arm_joint_position"] = torch.zeros((7, 6), dtype=torch.float32).T
    _refresh_digest(payload)

    with pytest.raises(ValueError, match="contiguous dense, strided, unquantized CPU tensor"):
        reset_dataset_validate_runtime(payload)


@pytest.mark.parametrize(
    ("container", "field"),
    (
        ("states", "task_body_previous_pose"),
        ("states", "task_body_coupling_previous_pose"),
        ("goal_state", "task_body_previous_pose"),
        ("goal_state", "task_body_coupling_previous_pose"),
    ),
)
def test_vbd_history_requires_exact_float32_dtype(container: str, field: str):
    payload = _payload()
    payload[container][field] = payload[container][field].to(dtype=torch.float64)
    _refresh_digest(payload)

    with pytest.raises(ValueError, match=rf"{container}\.{field} must have dtype torch\.float32"):
        reset_dataset_validate_runtime(payload)


@pytest.mark.parametrize(
    ("container", "field"),
    (
        ("states", "task_body_previous_pose"),
        ("states", "task_body_coupling_previous_pose"),
        ("goal_state", "task_body_previous_pose"),
        ("goal_state", "task_body_coupling_previous_pose"),
    ),
)
def test_vbd_history_requires_exact_body_pose_shape(container: str, field: str):
    payload = _payload()
    payload[container][field] = payload[container][field][..., :-1].contiguous()
    _refresh_digest(payload)

    with pytest.raises(ValueError, match=rf"{container}\.{field} must have dtype .* and shape"):
        reset_dataset_validate_runtime(payload)


@pytest.mark.parametrize(
    ("container", "field"),
    (
        ("states", "task_body_previous_pose"),
        ("states", "task_body_coupling_previous_pose"),
        ("goal_state", "task_body_previous_pose"),
        ("goal_state", "task_body_coupling_previous_pose"),
    ),
)
def test_vbd_history_requires_contiguous_storage(container: str, field: str):
    payload = _payload(task_body_count=4)
    value = payload[container][field]
    transposed = value.transpose(-2, -1).contiguous().transpose(-2, -1)
    assert transposed.shape == value.shape and not transposed.is_contiguous()
    payload[container][field] = transposed
    _refresh_digest(payload)

    with pytest.raises(ValueError, match="contiguous dense, strided, unquantized CPU tensor"):
        reset_dataset_validate_runtime(payload)


def test_unrecognized_state_field_is_rejected():
    payload = _payload()
    payload["states"]["legacy_category"] = torch.zeros(6, dtype=torch.int64)
    _refresh_digest(payload)

    with pytest.raises(ValueError, match="states fields do not match the schema"):
        reset_dataset_validate_runtime(payload)


def test_complete_validation_report_is_bound_to_artifact_rows_phases_and_contract():
    payload = _payload()
    report = json.loads(json.dumps(_validation_report(payload)))

    checks = _validate_report(report, payload)

    assert checks and all(checks.values())


def test_validation_report_accepts_recovery_body_distance_and_observed_cable_peak():
    payload = _payload()
    report = _validation_report(payload)
    metrics = report["rows"][0]["metrics"]
    recovery_speed_gates = PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["per_step_rejection_gates"]
    metrics["recovery_plug_speed"] = 0.03
    metrics["recovery_maximum_body_excursion_m"] = 0.55
    metrics["recovery_maximum_cable_linear_speed_m_s"] = 0.05
    metrics["recovery_maximum_arm_joint_speed_rad_s"] = recovery_speed_gates["arm_joint_speed_rad_s"]
    metrics["recovery_maximum_finger_joint_speed_m_s"] = recovery_speed_gates["finger_joint_speed_m_s"]
    report["content_sha256"] = reset_validation_report_content_digest(report)

    checks = _validate_report(report, payload)

    assert checks["every_row_metrics"] is True


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("recovery_invalid_contacts", 1),
        ("recovery_plug_speed", 0.030001),
        ("recovery_maximum_body_excursion_m", -1.0e-9),
        ("recovery_maximum_cable_linear_speed_m_s", -1.0e-9),
        (
            "recovery_maximum_arm_joint_speed_rad_s",
            PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["per_step_rejection_gates"]["arm_joint_speed_rad_s"] + 1.0e-6,
        ),
        (
            "recovery_maximum_finger_joint_speed_m_s",
            PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["per_step_rejection_gates"]["finger_joint_speed_m_s"] + 1.0e-6,
        ),
    ),
)
def test_validation_report_rejects_invalid_recovery_metrics(field: str, invalid: object):
    payload = _payload()
    report = _validation_report(payload)
    report["rows"][0]["metrics"][field] = invalid
    report["content_sha256"] = reset_validation_report_content_digest(report)

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        _validate_report(report, payload)


@pytest.mark.parametrize(
    ("container_path", "mutation"),
    (
        ((), "extra"),
        (("validation_cfg",), "missing"),
        (("physical_contract",), "extra"),
        (("physics_versions",), "missing"),
        (("goal_replay",), "extra"),
        (("full_pick_diversity",), "extra"),
        (("full_pick_diversity", "offline_artifact"), "extra"),
        (("rows", 0), "extra"),
        (("rows", 0, "metrics"), "missing"),
        (("rows", 0, "reset_replay"), "extra"),
    ),
)
def test_validation_report_requires_exact_schema_five_nested_key_sets(
    container_path: tuple[object, ...],
    mutation: str,
):
    payload = _payload()
    report = _validation_report(payload)
    container: object = report
    for part in container_path:
        container = container[part]
    assert isinstance(container, dict)
    if mutation == "extra":
        container["unexpected_schema_field"] = 0
    else:
        container.pop(next(iter(container)))
    report["content_sha256"] = reset_validation_report_content_digest(report)

    with pytest.raises(ValueError, match="fields do not match the schema"):
        _validate_report(report, payload)


@pytest.mark.parametrize(
    ("path", "invalid"),
    (
        (("full_pick_diversity", "unique_tcp_positions"), 91),
        (("rows", 0, "metrics", "initial_goal_error_matches"), False),
        (("rows", 0, "metrics", "settle_socket_drift_m"), 1.0e-6),
        (("goal_replay", "maximum_invalid_contact_count"), 1),
        (("goal_replay", "any_contact_overflow"), True),
    ),
)
def test_validation_report_rejects_cross_field_and_raw_evidence_contradictions(
    path: tuple[object, ...],
    invalid: object,
):
    payload = _payload()
    report = _validation_report(payload)
    container: object = report
    for part in path[:-1]:
        container = container[part]
    assert isinstance(container, dict)
    container[path[-1]] = invalid
    report["content_sha256"] = reset_validation_report_content_digest(report)

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        _validate_report(report, payload)


def test_validation_report_offline_diversity_must_match_independent_runtime_evidence():
    payload = _payload()
    report = _validation_report(payload)
    report["full_pick_diversity"]["offline_artifact"]["unique_arm_rows"] = 89
    report["content_sha256"] = reset_validation_report_content_digest(report)

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        _validate_report(report, payload)


@pytest.mark.parametrize("snapshot", ("validation_policy", "source_sha256", "asset_closure"))
def test_validation_report_schema_five_binds_policy_source_and_asset_snapshots(snapshot: str):
    payload = _payload()
    report = _validation_report(payload)
    if snapshot == "validation_policy":
        report[snapshot]["ik"]["seed_count"] = 2
    elif snapshot == "source_sha256":
        report[snapshot]["scripts/tools/validate_franka_rj45_pick_insert_resets.py"] = "f" * 64
    else:
        report[snapshot]["tree_sha256"] = "f" * 64
    report["content_sha256"] = reset_validation_report_content_digest(report)

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        _validate_report(report, payload)


def test_validation_report_rejects_changed_scripted_recovery_policy():
    payload = _payload()
    report = _validation_report(payload)
    report["validation_policy"]["scripted_recovery"]["planning"]["maximum_translation_step_m"] = 0.003
    report["content_sha256"] = reset_validation_report_content_digest(report)

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        _validate_report(report, payload)


def test_validation_report_content_digest_rejects_post_publication_tampering():
    payload = _payload()
    report = _validation_report(payload)
    report["rows"][0]["metrics"] = {"tampered": True}

    with pytest.raises(ValueError, match="metrics fields do not match the schema"):
        _validate_report(report, payload)


def test_validation_report_cfg_and_ik_call_evidence_must_match_sampler_free_policy():
    payload = _payload()
    report = _validation_report(payload)
    report["validation_cfg"]["ik_sampler"] = "gauss"
    report["content_sha256"] = reset_validation_report_content_digest(report)
    with pytest.raises(ValueError, match="incomplete or incompatible"):
        _validate_report(report, payload)

    report = _validation_report(payload)
    report["ik_solve_call_count"] = -1
    report["content_sha256"] = reset_validation_report_content_digest(report)
    with pytest.raises(ValueError, match="incomplete or incompatible"):
        _validate_report(report, payload)


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("passed", False),
        ("skipped_due_to_quick", True),
        ("required_phase", 4),
        ("observed_rows", 0),
        ("round_decimals", 3),
        ("minimum_unique_tcp_positions", 89),
        ("unique_tcp_positions", 89),
        ("minimum_tcp_xyz_span_m", [0.04, 0.05, 0.05]),
        ("tcp_xyz_span_m", [0.049, 0.06, 0.06]),
        ("minimum_tcp_to_grasp_distance_m", 0.09),
        ("observed_minimum_tcp_to_grasp_distance_m", 0.099),
    ),
)
def test_validation_report_rejects_forged_full_pick_diversity(field: str, invalid: object):
    payload = _payload()
    report = _validation_report(payload)
    report["full_pick_diversity"][field] = invalid

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        _validate_report(report, payload)


def test_validation_report_requires_full_pick_diversity_mapping():
    payload = _payload()
    report = _validation_report(payload)
    del report["full_pick_diversity"]

    with pytest.raises(ValueError, match=r"missing=\['full_pick_diversity'\]"):
        _validate_report(report, payload)


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("passed", False),
        ("evidence_complete", False),
        ("full_dataset_replay", False),
        ("quick", True),
    ),
)
def test_validation_report_requires_complete_full_replay(field: str, invalid: bool):
    payload = _payload()
    report = _validation_report(payload)
    report[field] = invalid

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        _validate_report(report, payload)


def test_validation_report_requires_exact_content_and_task_contract():
    payload = _payload()
    report = _validation_report(payload)
    report["artifact_content_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        _validate_report(report, payload)

    report = _validation_report(payload)
    report["task_contract"] = {**report["task_contract"], "unexpected": True}

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        _validate_report(report, payload)


@pytest.mark.parametrize("field", ("selected_row_ids", "rows"))
def test_validation_report_requires_every_row_id_exactly_once(field: str):
    payload = _payload()
    report = _validation_report(payload)
    if field == "selected_row_ids":
        report[field][-1] = 0
    else:
        report[field][-1]["row_id"] = 0

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        _validate_report(report, payload)


def test_validation_report_rejects_any_failed_row_id():
    payload = _payload()
    report = _validation_report(payload)
    report["failed_row_ids"] = [3]

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        _validate_report(report, payload)


@pytest.mark.parametrize("source", ("summary", "row"))
def test_validation_report_requires_exact_phase_counts(source: str):
    payload = _payload()
    report = _validation_report(payload)
    if source == "summary":
        report["phase_counts"][0] = 0
    else:
        report["rows"][0]["phase"] = 1

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        _validate_report(report, payload)


def test_validation_report_rejects_phase_permutation_with_unchanged_counts():
    payload = _payload()
    report = _validation_report(payload)
    report["rows"][0]["phase"], report["rows"][1]["phase"] = (
        report["rows"][1]["phase"],
        report["rows"][0]["phase"],
    )

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        _validate_report(report, payload)


@pytest.mark.parametrize(
    "expected_phases",
    (
        torch.tensor(PICK_INSERT_RESET_PHASE_IDS, dtype=torch.int32),
        [0, 1, 2, 3, 4, 4],
        [0, 1, 2, 3, 4],
    ),
)
def test_validation_report_requires_strict_exact_expected_phases(expected_phases: object):
    payload = _payload()
    report = _validation_report(payload)

    with pytest.raises((TypeError, ValueError)):
        reset_validation_report_validate_runtime(
            report,
            expected_content_sha256=payload["content_sha256"],
            expected_row_count=6,
            expected_phases=expected_phases,
            expected_task_contract=payload["metadata"]["task_contract"],
            expected_validation_policy=FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY,
            expected_source_sha256=_SOURCE_SHA256_FIXTURE,
            expected_asset_closure=payload["metadata"]["task_contract"]["external_assets"],
            expected_full_pick_diversity=_offline_diversity_fixture(),
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("passed", False),
        ("drive_disabled", False),
        ("vbd_pose_history_restore_queued", False),
        ("vbd_pose_history_pending_at_queue", False),
        ("vbd_previous_pose_queued", False),
        ("vbd_coupling_previous_pose_queued", False),
        ("vbd_pose_history_applied_exactly_once", False),
        ("vbd_pose_history_failed", True),
        ("vbd_pose_history_superseded", True),
        ("vbd_pose_history_pending_after_first_solve", True),
        ("vbd_pose_history_minimum_application_count_delta", 0),
        ("vbd_pose_history_maximum_application_count_delta", 2),
        ("vbd_pose_history_generation", True),
        ("vbd_pose_history_expected_body_count", 3),
        ("vbd_pose_history_minimum_body_application_count_delta", 3),
        ("vbd_pose_history_maximum_body_application_count_delta", 3),
        ("vbd_pose_history_body_order_exact", False),
        ("vbd_pose_history_world_order_exact", False),
        ("vbd_pose_history_entry_name", "wrong"),
        ("vbd_pose_history_body_count", 3),
        ("socket_stable", False),
        ("whole_cable_stable", False),
        ("exact_runtime_success_dwell", False),
        ("all_samples_collision_free", False),
        ("all_samples_bilateral_grasp", False),
        ("all_samples_proxy_bilateral_contact", False),
        ("all_samples_finite", False),
        ("no_contact_overflow", False),
        ("stored_capture_exact_success", False),
        ("all_post_step_exact_success", False),
        ("robot_equilibrium", False),
        ("zero_action_unclamped", False),
        ("sampled_maximum_arm_joint_speed_rad_s", 0.1001),
        ("sampled_maximum_finger_joint_speed_m_s", 0.0501),
        ("maximum_arm_target_clamp_delta_rad", 1.01e-7),
        ("controller_semantics", "measured-state-relative-reset-bias"),
        ("absolute_target_stable", False),
        ("maximum_arm_target_drift_rad", 1.01e-7),
        ("all_samples_arm_target_tracking_bounded", False),
        ("authored_goal_geometry_valid", False),
        ("maximum_allowed_authored_seat_error_m", 0.0011),
        ("maximum_allowed_authored_plug_angle_rad", math.radians(3.1)),
        (
            "maximum_allowed_plug_relative_latch_angle_rad",
            PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD + 0.001,
        ),
        ("maximum_stored_authored_seat_error_m", 0.0011),
        ("maximum_final_authored_seat_error_m", 0.0011),
        ("maximum_stored_authored_plug_angle_rad", math.radians(3.1)),
        ("maximum_final_authored_plug_angle_rad", math.radians(3.1)),
        (
            "maximum_stored_plug_relative_latch_angle_rad",
            PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD + 0.001,
        ),
        (
            "maximum_final_plug_relative_latch_angle_rad",
            PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD + 0.001,
        ),
        ("simulation_time_s", 59.99),
    ),
)
def test_goal_replay_requires_stable_drive_disabled_exact_success(field: str, invalid: object):
    payload = _payload()
    report = _validation_report(payload)
    report["goal_replay"][field] = invalid

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        _validate_report(report, payload)


@pytest.mark.parametrize(
    ("field", "required"),
    (
        ("maximum_allowed_socket_drift_m", PICK_INSERT_GOAL_MAX_SOCKET_DRIFT_M),
        ("maximum_allowed_task_body_drift_m", PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M),
        ("maximum_allowed_cable_speed_m_s", PICK_INSERT_GOAL_MAX_CABLE_SPEED_M_S),
        ("maximum_allowed_arm_joint_speed_rad_s", PICK_INSERT_GOAL_MAX_ARM_JOINT_SPEED_RAD_S),
        ("maximum_allowed_finger_joint_speed_m_s", PICK_INSERT_GOAL_MAX_FINGER_JOINT_SPEED_M_S),
    ),
)
@pytest.mark.parametrize("invalid_kind", ("bool", "nan", "negative", "over_limit"))
def test_goal_replay_requires_exact_reported_limits(
    field: str,
    required: float,
    invalid_kind: str,
):
    payload = _payload()
    report = _validation_report(payload)
    invalid = {
        "bool": True,
        "nan": float("nan"),
        "negative": -required,
        "over_limit": required + max(1.0e-9, required * 1.0e-3),
    }[invalid_kind]
    report["goal_replay"][field] = invalid

    expected_error = "finite" if invalid_kind == "nan" else "incomplete or incompatible"
    with pytest.raises(ValueError, match=expected_error):
        _validate_report(report, payload)


@pytest.mark.parametrize(
    ("field", "limit"),
    (
        ("maximum_socket_drift_m", PICK_INSERT_GOAL_MAX_SOCKET_DRIFT_M),
        ("maximum_task_body_drift_m", PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M),
        ("sampled_maximum_task_body_excursion_m", PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M),
        ("maximum_start_cable_speed_m_s", PICK_INSERT_GOAL_MAX_CABLE_SPEED_M_S),
        ("maximum_final_cable_speed_m_s", PICK_INSERT_GOAL_MAX_CABLE_SPEED_M_S),
        ("sampled_maximum_cable_speed_m_s", PICK_INSERT_GOAL_MAX_CABLE_SPEED_M_S),
    ),
)
@pytest.mark.parametrize("invalid_kind", ("bool", "nan", "negative", "over_limit"))
def test_goal_replay_rejects_invalid_stability_numeric_evidence(
    field: str,
    limit: float,
    invalid_kind: str,
):
    payload = _payload()
    report = _validation_report(payload)
    invalid = {
        "bool": True,
        "nan": float("nan"),
        "negative": -1.0e-9,
        "over_limit": limit + max(1.0e-9, limit * 1.0e-3),
    }[invalid_kind]
    report["goal_replay"][field] = invalid

    expected_error = "finite" if invalid_kind == "nan" else "incomplete or incompatible"
    with pytest.raises(ValueError, match=expected_error):
        _validate_report(report, payload)


@pytest.mark.parametrize("check_name", _ROW_CHECK_NAMES)
def test_every_row_requires_each_physical_replay_check(check_name: str):
    payload = _payload()
    report = _validation_report(payload)
    report["rows"][2]["checks"][check_name] = False

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        _validate_report(report, payload)


@pytest.mark.parametrize("evidence_name", _ORACLE_EVIDENCE_NAMES)
def test_every_row_requires_each_oracle_evidence_flag(evidence_name: str):
    payload = _payload()
    report = _validation_report(payload)
    report["rows"][2]["oracle"][evidence_name] = False

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        _validate_report(report, payload)


def test_dataset_requires_canonical_phase_grasp_mapping():
    payload = _payload()
    payload["states"]["starts_grasped"][0] = False
    _refresh_digest(payload)

    with pytest.raises(ValueError, match="starts_grasped"):
        reset_dataset_validate_runtime(payload)


def test_dataset_rejects_arm_target_outside_persistent_tracking_envelope():
    payload = _payload()
    payload["states"]["arm_joint_target"][0, 4] = PICK_INSERT_ARM_TARGET_TRACKING_LIMITS[4] + 1.0e-4
    _refresh_digest(payload)

    with pytest.raises(ValueError, match="target-tracking envelope"):
        reset_dataset_validate_runtime(payload)


def test_every_row_requires_raw_reset_replay_evidence():
    payload = _payload()
    report = _validation_report(payload)
    del report["rows"][0]["reset_replay"]

    with pytest.raises(ValueError, match=r"rows\[\] fields do not match the schema"):
        _validate_report(report, payload)


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("simulation_steps", 15.0),
        ("post_step_samples", True),
        ("stored_maximum_cable_speed_m_s", True),
        ("stored_maximum_arm_joint_speed_rad_s", -0.01),
        ("maximum_post_step_finger_joint_speed_m_s", 0.051),
        ("maximum_body_excursion_m", 0.0121),
        ("maximum_plug_excursion_m", 0.0061),
        ("maximum_socket_excursion_m", 1.1e-5),
        ("maximum_arm_target_clamp_delta_rad", 1.1e-7),
        ("maximum_arm_target_drift_rad", 1.1e-7),
        ("absolute_target_stable", False),
        ("arm_target_semantics", "measured-state-relative-reset-bias"),
        ("stored_arm_target_tracking_bounded", False),
        ("all_post_step_arm_target_tracking_bounded", False),
        ("maximum_invalid_contact_count", 1),
        ("no_contact_overflow", False),
        ("maximum_allowed_cable_speed_m_s", 0.041),
        ("vbd_pose_history_restore_queued", False),
        ("vbd_pose_history_pending_at_queue", False),
        ("vbd_previous_pose_queued", False),
        ("vbd_coupling_previous_pose_queued", False),
        ("vbd_pose_history_applied_exactly_once", False),
        ("vbd_pose_history_failed", True),
        ("vbd_pose_history_superseded", True),
        ("vbd_pose_history_pending_after_first_solve", True),
        ("vbd_pose_history_application_count_delta", 0),
        ("vbd_pose_history_generation", True),
        ("vbd_pose_history_expected_body_count", 3),
        ("vbd_pose_history_body_application_count_delta", 3),
        ("vbd_pose_history_body_order_exact", False),
        ("vbd_pose_history_world_order_exact", False),
        ("vbd_pose_history_entry_name", "wrong"),
        ("vbd_pose_history_body_count", 3),
    ),
)
def test_row_reset_replay_rejects_forged_numeric_or_contract_evidence(field: str, invalid: object):
    payload = _payload()
    report = _validation_report(payload)
    report["rows"][0]["reset_replay"][field] = invalid

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        _validate_report(report, payload)


def test_row_reset_replay_rejects_nonfinite_numeric_evidence():
    payload = _payload()
    report = _validation_report(payload)
    report["rows"][0]["reset_replay"]["maximum_body_excursion_m"] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        _validate_report(report, payload)


@pytest.mark.parametrize(
    ("row_id", "field", "invalid"),
    (
        (0, "maximum_left_proxy_contact_count", 0),
        (0, "all_post_step_zero_proxy_contacts", True),
        (4, "minimum_left_proxy_contact_count", 1),
        (4, "all_post_step_bilateral_grasp", True),
        (4, "all_post_step_proxy_bilateral_contact", True),
        (4, "starts_grasped", True),
    ),
)
def test_row_reset_replay_rejects_phase_inconsistent_contact_evidence(
    row_id: int,
    field: str,
    invalid: object,
):
    payload = _payload()
    report = _validation_report(payload)
    report["rows"][row_id]["reset_replay"][field] = invalid

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        _validate_report(report, payload)


def test_row_reset_replay_rejects_reported_invalid_contact_pairs():
    payload = _payload()
    report = _validation_report(payload)
    report["rows"][0]["reset_replay"]["invalid_contact_pairs"] = ["finger <-> table"]

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        _validate_report(report, payload)


def test_validation_report_rejects_nonfinite_or_non_json_evidence():
    payload = _payload()
    report = _validation_report(payload)
    report["generator_metric"] = float("nan")

    with pytest.raises(ValueError, match="must contain only finite values"):
        _validate_report(report, payload)

    report = _validation_report(payload)
    report["generator_metric"] = torch.zeros(1)

    with pytest.raises(TypeError, match="unsupported value type Tensor"):
        _validate_report(report, payload)
