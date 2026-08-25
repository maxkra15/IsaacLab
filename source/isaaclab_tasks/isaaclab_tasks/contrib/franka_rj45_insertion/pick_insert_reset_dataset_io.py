# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-state artifact boundary for Franka RJ45 pick-and-insert training."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from .franka_robot_cfg import PICK_INSERT_ARM_TARGET_TRACKING_LIMITS
from .reset_dataset_io import reset_dataset_content_digest, reset_dataset_digest

FRANKA_RJ45_PICK_INSERT_RESET_DATASET_FORMAT = "isaaclab-franka-rj45-pick-insert-reset-dataset"
FRANKA_RJ45_PICK_INSERT_RESET_DATASET_SCHEMA_VERSION = 3
FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_FORMAT = "isaaclab-franka-rj45-pick-insert-reset-validation"
FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_SCHEMA_VERSION = 5
FRANKA_RJ45_PICK_INSERT_FAST_RESET_VALIDATION_FORMAT = "isaaclab-franka-rj45-pick-insert-fast-reset-validation"
FRANKA_RJ45_PICK_INSERT_FAST_RESET_VALIDATION_SCHEMA_VERSION = 1
PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY = {
    "contract_version": 3,
    "motion_policy": "incremental-cartesian-c2-v3",
    "legacy_default_motion_policy": "legacy-direct-joint",
    "initial_route_endpoint_policy": "canonical-goal-c2-stop-before-bounded-compensation",
    "phase_route_modes": {
        "0": "insertion-corridor",
        "1": "insertion-corridor",
        "2": "clearance-via-preinsert",
        "3": "clearance-via-preinsert",
        "4": "clearance-via-preinsert",
        "5": "clearance-via-preinsert",
    },
    "insertion_corridor": {
        "maximum_axial_distance_before_goal_m": 0.10,
        "maximum_axial_overtravel_m": 0.006,
        "maximum_radial_error_m": 0.006,
        "maximum_orientation_error_rad": math.radians(3.0),
    },
    "clearance_route": {
        "preinsert_axial_offset_m": 0.08,
        "clearance_above_preinsert_m": 0.10,
        "transport_height": "max(live-plug-z,preinsert-z+clearance)",
        "plug_pose_waypoints": (
            "vertical-lift",
            "high-midpoint",
            "overhead-preinsert",
            "preinsert",
            "canonical-goal",
        ),
    },
    "planning": {
        "maximum_translation_step_m": 0.002,
        "maximum_rotation_step_rad": math.radians(2.0),
        "maximum_raw_ik_joint_step_rad": 0.02,
        "maximum_commanded_joint_step_rad": 0.02,
        "joint_limit_margin_rad": 0.02,
        "maximum_waypoints": 430,
        "maximum_waypoints_scope": "post-densification-executed-global-unique-knots",
        "endpoint_command_bias_policy": "linear-start-to-goal-over-global-unique-route-knots",
        "exact_start_target": True,
        "exact_canonical_endpoint": True,
        "command_interval_densification": "deterministic-collinear-joint-subknots",
        "densification_step_limit_rad": 0.02,
        "compensation_bias_policy": "constant-start-bias",
    },
    "execution": {
        "minimum_duration_per_knot_s": 0.20,
        "c2_ramp_fraction": 0.10,
        "time_law_endpoint_continuity": "C2",
        "joint_path_interpolation": "piecewise-linear-through-precomputed-ik-knots",
        "joint_path_internal_knot_continuity": "C0-with-bounded-target-velocity-jumps",
        "internal_knot_settles": 0,
        "segment_end_settle_s": 1.0 / 30.0,
    },
    "per_step_rejection_gates": {
        "finite": True,
        "collision": True,
        "bilateral_proxy_contact": True,
        "grasp_geometry": True,
        "construction_drives_disabled": True,
        "plug_linear_speed_m_s": 0.04,
        "plug_angular_speed_rad_s": 0.35,
        "arm_joint_speed_rad_s": 0.5,
        "finger_joint_speed_m_s": 0.05,
        "contact_overflow_is_global_fatal": True,
    },
    "transient_cable_speed": {
        "sampled": True,
        "reset_limit_m_s": 0.04,
        "rejection_gate": False,
    },
    "compensation": {
        "subdivide_with_same_cartesian_policy": True,
        "proactive_overtravel": False,
        "trigger": "settled-goal-error-above-tolerance",
        "canonical_return_uses_same_endpoint_bias_blend": True,
    },
    "final_recovery_gates_unchanged": True,
}
"""Immutable pick-only scripted-recovery policy shared by generation and validation."""
FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY = {
    "contract_version": 4,
    "ik": {
        "implementation": "FrankaResetIK",
        "owner_count": 1,
        "sampler": "none",
        "seed_count": 1,
        "iterations": 160,
        "noise_std": 0.0,
        "resume_solver_state": "fresh-sampler-free-owner",
        "solve_call_count_semantics": "evidence-only",
    },
    "scripted_recovery": PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY,
}
"""Immutable independent-validator solver policy embedded in schema-5 reports."""
FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY = {
    "contract_version": 2,
    "generation_mode": "fast-ik",
    "checks": {
        "finite": True,
        "ik_solved": True,
        "joint_limits": True,
        "workspace": True,
        "collision_filtered": True,
        "phase_semantics": True,
    },
    "joint_limit_margin_rad": 0.02,
    "collision_filter": {
        "method": "analytic-plus-newton-collide-only",
        "maximum_table_penetration_m": 5.0e-4,
        "minimum_nonadjacent_cable_surface_gap_m": -5.0e-4,
        "minimum_cable_socket_center_distance_m": 0.02,
        "newton_query": {
            "outer_pipeline": True,
            "proxy_pipeline": True,
            "penetration_tolerance_m": 5.0e-4,
            "starts_grasped_requires_bilateral_proxy_contact": True,
            "open_gripper_requires_zero_proxy_contact": True,
            "advances_solver": False,
            "consumes_proxy_collision_cadence": False,
            "accepted_row_evidence_fields": (
                "collide_only_invalid_contact_count",
                "collide_only_grasp_contact_count",
                "collide_only_left_grasp_contact_count",
                "collide_only_right_grasp_contact_count",
                "collide_only_contact_overflow",
            ),
        },
    },
    "simulation_steps": 0,
    "dynamics_replay": False,
    "scripted_recovery": False,
}
"""Fast initial-state admission policy; it deliberately excludes dynamics and recovery."""
PICK_INSERT_RESET_PHASE_IDS = tuple(range(6))
PICK_INSERT_RESET_REPLAY_DURATION_S = 0.5
PICK_INSERT_RESET_REPLAY_POST_STEP_SAMPLES = 15
PICK_INSERT_RESET_MAX_ARM_JOINT_SPEED_RAD_S = 0.1
PICK_INSERT_RESET_MAX_FINGER_JOINT_SPEED_M_S = 0.05
PICK_INSERT_RESET_MAX_CABLE_SPEED_M_S = 0.04
PICK_INSERT_RESET_MAX_BODY_EXCURSION_M = 0.012
PICK_INSERT_RESET_MAX_PLUG_EXCURSION_M = 0.006
PICK_INSERT_RESET_MAX_SOCKET_EXCURSION_M = 1.0e-5
PICK_INSERT_RESET_MAX_ARM_TARGET_CLAMP_DELTA_RAD = 1.0e-7
PICK_INSERT_VBD_POSE_HISTORY_CONTRACT_VERSION = 2
PICK_INSERT_VBD_POSE_HISTORY_ENTRY_NAME = "rj45"
PICK_INSERT_GOAL_MAX_ARM_JOINT_SPEED_RAD_S = 0.1
PICK_INSERT_GOAL_MAX_FINGER_JOINT_SPEED_M_S = 0.05
PICK_INSERT_GOAL_MAX_SOCKET_DRIFT_M = 1.0e-5
PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M = 0.012
PICK_INSERT_GOAL_MAX_CABLE_SPEED_M_S = 0.01
PICK_INSERT_GOAL_MAX_AUTHORED_SEAT_ERROR_M = 0.001
PICK_INSERT_GOAL_MAX_AUTHORED_PLUG_ANGLE_RAD = math.radians(3.0)
PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD = 0.10
_FULL_PICK_PHASE = 5
_FULL_PICK_TCP_ROUND_DECIMALS = 4
_FULL_PICK_MINIMUM_UNIQUE_TCP_POSITIONS = 90
_FULL_PICK_MINIMUM_TCP_SPAN_M = (0.05, 0.05, 0.05)
_FULL_PICK_MINIMUM_TCP_TO_GRASP_DISTANCE_M = 0.10
_VALIDATION_SOURCE_ROOTS = (
    "source/isaaclab/isaaclab/",
    "source/isaaclab_newton/isaaclab_newton/",
    "source/isaaclab_contrib/isaaclab_contrib/",
    "source/isaaclab_assets/isaaclab_assets/",
    "source/isaaclab_tasks/isaaclab_tasks/",
)
_VALIDATION_SOURCE_FILES = (
    "scripts/tools/validate_franka_rj45_pick_insert_resets.py",
    "scripts/tools/_franka_rj45_reset_tools.py",
    "uv.lock",
)
_FAST_VALIDATION_SOURCE_FILES = (
    *_VALIDATION_SOURCE_FILES,
    "scripts/tools/generate_franka_rj45_pick_insert_reset_dataset.py",
    "scripts/tools/validate_franka_rj45_pick_insert_fast_resets.py",
)

_FAST_VALIDATION_REPORT_NAMES = {
    "format",
    "schema_version",
    "created_utc",
    "artifact_content_sha256",
    "task_contract",
    "validation_policy",
    "source_sha256",
    "asset_closure",
    "evidence_origin",
    "simulation_steps",
    "dynamics_replay",
    "scripted_recovery",
    "dataset_row_count",
    "selected_row_ids",
    "phase_counts",
    "full_pick_diversity",
    "rows",
    "failed_row_ids",
    "passed",
    "content_sha256",
}
_FAST_VALIDATION_ROW_NAMES = {"row_id", "phase", "passed", "checks", "metrics"}
_FAST_VALIDATION_CHECK_NAMES = set(FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY["checks"])
_FAST_VALIDATION_METRIC_NAMES = {
    "minimum_joint_limit_margin_rad",
    "minimum_workspace_margin_m",
    "minimum_cable_support_clearance_m",
    "minimum_nonadjacent_cable_separation_m",
    "minimum_cable_socket_center_distance_m",
}

_VALIDATION_REPORT_NAMES = {
    "format",
    "schema_version",
    "created_utc",
    "artifact_content_sha256",
    "task_contract",
    "validation_cfg",
    "validation_policy",
    "physical_contract",
    "physics_versions",
    "source_sha256",
    "asset_closure",
    "ik_solve_call_count",
    "quick",
    "full_dataset_replay",
    "evidence_complete",
    "selected_row_count",
    "dataset_row_count",
    "selected_row_ids",
    "phase_counts",
    "goal_replay",
    "full_pick_diversity",
    "rows",
    "failed_row_ids",
    "passed",
    "content_sha256",
}
_VALIDATION_CFG_NAMES = {
    "seed",
    "quick",
    "sample_count",
    "ik_sampler",
    "ik_seed_count",
    "ik_iterations",
    "ik_noise_std",
    "goal_replay_s",
    "row_settle_s",
    "grasp_approach_s",
    "grasp_close_s",
    "grasp_hold_s",
    "grasp_post_contact_settle_s",
    "grasp_open_clearance_m",
    "grasp_route_world_height_m",
    "grasp_route_maximum_translation_step_m",
    "grasp_coarse_descent_step_m",
    "grasp_near_descent_step_m",
    "grasp_descent_waypoint_motion_s",
    "grasp_descent_waypoint_settle_s",
    "grasp_descent_tracking_recovery_s",
    "grasp_near_correction_step_m",
    "grasp_near_correction_max_iterations",
    "grasp_near_maximum_raw_ik_joint_step_rad",
    "grasp_clearance_calibration_step_m",
    "grasp_clearance_calibration_max_iterations",
    "maximum_open_approach_plug_drift_m",
    "finger_open_position",
    "finger_closed_target",
    "recovery_motion_s",
    "recovery_settle_s",
    "recovery_compensation_iterations",
    "recovery_compensation_tolerance_m",
    "maximum_ik_joint_step_rad",
    "maximum_goal_socket_drift_m",
    "maximum_goal_body_drift_m",
    "maximum_goal_cable_speed_m_s",
    "maximum_goal_arm_joint_speed_rad_s",
    "maximum_goal_finger_joint_speed_m_s",
    "maximum_goal_authored_seat_error_m",
    "maximum_goal_authored_plug_angle_rad",
    "maximum_goal_plug_relative_latch_angle_rad",
    "maximum_row_socket_drift_m",
    "maximum_row_plug_drift_m",
    "maximum_row_body_drift_m",
    "maximum_row_cable_speed_m_s",
    "maximum_row_arm_joint_speed_rad_s",
    "maximum_row_finger_joint_speed_m_s",
}
_VALIDATION_CFG_CANONICAL_VALUES = {
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
    "maximum_goal_authored_seat_error_m": PICK_INSERT_GOAL_MAX_AUTHORED_SEAT_ERROR_M,
    "maximum_goal_authored_plug_angle_rad": PICK_INSERT_GOAL_MAX_AUTHORED_PLUG_ANGLE_RAD,
    "maximum_goal_plug_relative_latch_angle_rad": PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD,
    "maximum_row_socket_drift_m": PICK_INSERT_RESET_MAX_SOCKET_EXCURSION_M,
    "maximum_row_plug_drift_m": PICK_INSERT_RESET_MAX_PLUG_EXCURSION_M,
    "maximum_row_body_drift_m": PICK_INSERT_RESET_MAX_BODY_EXCURSION_M,
    "maximum_row_cable_speed_m_s": PICK_INSERT_RESET_MAX_CABLE_SPEED_M_S,
    "maximum_row_arm_joint_speed_rad_s": PICK_INSERT_RESET_MAX_ARM_JOINT_SPEED_RAD_S,
    "maximum_row_finger_joint_speed_m_s": PICK_INSERT_RESET_MAX_FINGER_JOINT_SPEED_M_S,
}
_PHYSICAL_CONTRACT_NAMES = {
    "finger_closed_target_m",
    "live_finger_close_position_m",
    "configured_grasp_proxy_raw_friction",
    "live_grasp_proxy_raw_friction",
    "effective_finger_proxy_friction",
    "success_max_plug_speed",
}
_PHYSICS_VERSION_NAMES = {"newton", "warp", "isaaclab"}
_GOAL_REPLAY_NAMES = {
    "passed",
    "drive_disabled",
    "vbd_pose_history_restore_queued",
    "vbd_pose_history_pending_at_queue",
    "vbd_previous_pose_queued",
    "vbd_coupling_previous_pose_queued",
    "vbd_pose_history_applied_exactly_once",
    "vbd_pose_history_failed",
    "vbd_pose_history_superseded",
    "vbd_pose_history_pending_after_first_solve",
    "vbd_pose_history_minimum_application_count_delta",
    "vbd_pose_history_maximum_application_count_delta",
    "vbd_pose_history_expected_body_count",
    "vbd_pose_history_minimum_body_application_count_delta",
    "vbd_pose_history_maximum_body_application_count_delta",
    "vbd_pose_history_generation",
    "vbd_pose_history_body_order_exact",
    "vbd_pose_history_world_order_exact",
    "vbd_pose_history_entry_name",
    "vbd_pose_history_body_count",
    "socket_stable",
    "whole_cable_stable",
    "exact_runtime_success_dwell",
    "simulation_steps",
    "simulation_time_s",
    "contact_count_after_history_reset",
    "maximum_socket_drift_m",
    "maximum_task_body_drift_m",
    "maximum_start_cable_speed_m_s",
    "maximum_final_cable_speed_m_s",
    "sampled_maximum_task_body_excursion_m",
    "sampled_maximum_cable_speed_m_s",
    "sampled_maximum_arm_joint_speed_rad_s",
    "sampled_maximum_finger_joint_speed_m_s",
    "maximum_allowed_socket_drift_m",
    "maximum_allowed_task_body_drift_m",
    "maximum_allowed_cable_speed_m_s",
    "maximum_allowed_arm_joint_speed_rad_s",
    "maximum_allowed_finger_joint_speed_m_s",
    "robot_equilibrium",
    "zero_action_unclamped",
    "maximum_arm_target_clamp_delta_rad",
    "controller_semantics",
    "absolute_target_stable",
    "maximum_arm_target_drift_rad",
    "all_samples_arm_target_tracking_bounded",
    "maximum_arm_target_tracking_error_by_joint_rad",
    "authored_goal_geometry_valid",
    "maximum_allowed_authored_seat_error_m",
    "maximum_allowed_authored_plug_angle_rad",
    "maximum_allowed_plug_relative_latch_angle_rad",
    "maximum_stored_authored_seat_error_m",
    "maximum_final_authored_seat_error_m",
    "maximum_stored_authored_plug_angle_rad",
    "maximum_final_authored_plug_angle_rad",
    "maximum_stored_plug_relative_latch_angle_rad",
    "maximum_final_plug_relative_latch_angle_rad",
    "all_samples_collision_free",
    "all_samples_bilateral_grasp",
    "all_samples_proxy_bilateral_contact",
    "all_samples_finite",
    "minimum_left_proxy_contact_count",
    "minimum_right_proxy_contact_count",
    "maximum_invalid_contact_count",
    "any_contact_overflow",
    "no_contact_overflow",
    "sampled_invalid_contact_pairs",
    "stored_capture_exact_success",
    "all_post_step_exact_success",
    "required_dwell_steps",
    "final_consecutive_steps",
    "collision_valid",
    "closed_bilateral_grasp",
}
_FULL_PICK_DIVERSITY_NAMES = {
    "required_phase",
    "required_rows",
    "round_decimals",
    "minimum_unique_tcp_positions",
    "minimum_tcp_xyz_span_m",
    "minimum_tcp_to_grasp_distance_m",
    "observed_rows",
    "offline_artifact",
    "skipped_due_to_quick",
    "tcp_xyz_min_m",
    "tcp_xyz_max_m",
    "tcp_xyz_span_m",
    "unique_tcp_positions",
    "observed_minimum_tcp_to_grasp_distance_m",
    "passed",
    "failures",
}
_OFFLINE_DIVERSITY_NAMES = {
    "phase",
    "row_count",
    "required_row_count",
    "round_decimals",
    "unique_socket_rows",
    "unique_plug_rows",
    "unique_arm_rows",
    "required_unique_socket_rows",
    "required_unique_plug_rows",
    "required_unique_arm_rows",
    "socket_xy_span_m",
    "required_socket_xy_span_m",
    "socket_yaw_span_rad",
    "required_socket_yaw_span_rad",
    "plug_pickup_xy_span_m",
    "required_plug_pickup_xy_span_m",
    "plug_pickup_yaw_span_rad",
    "required_plug_pickup_yaw_span_rad",
    "arm_joint_span_rad",
    "required_each_arm_joint_span_rad",
    "initial_tcp_grasp_distance_span_m",
    "required_initial_tcp_grasp_distance_span_m",
    "passed",
    "failures",
}
_ROW_NAMES = {"row_id", "phase", "passed", "checks", "oracle", "reset_replay", "metrics"}

RESET_DATASET_STATE_NAMES = (
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
    "goal_task_body_pose",
    "goal_arm_joint_target",
    "phase",
    "starts_grasped",
    "difficulty",
    "initial_goal_error",
    "initial_tcp_grasp_distance",
    "progress_threshold",
)

RESET_DATASET_GOAL_STATE_NAMES = (
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

PICK_INSERT_FAST_RESET_ROW_BINDING_CONTRACT = {
    "contract_version": 1,
    "row_id_field": "final_row_id",
    "state_digest_field": "state_sha256",
    "state_digest_algorithm": "reset_dataset_digest",
    "state_names": RESET_DATASET_STATE_NAMES,
}
"""Binding between fast-reset evidence records and final artifact rows."""
PICK_INSERT_FAST_RESET_PHASE_0_BAND_ACCEPTANCE_CONTRACT = {
    "contract_version": 1,
    "maximum_absolute_fraction_error": 0.05,
    "small_bank_discretization_allowance_rows": 1,
}
"""Admission tolerance for accepted phase-0 reverse-curriculum bands."""

_PAYLOAD_NAMES = {"format", "schema_version", "metadata", "states", "goal_state", "content_sha256"}
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
_ROW_RESET_REPLAY_NAMES = {
    "simulation_time_s",
    "simulation_steps",
    "post_step_samples",
    "required_simulation_time_s",
    "required_post_step_samples",
    "starts_grasped",
    "contact_expectation",
    "vbd_pose_history_restore_queued",
    "vbd_pose_history_pending_at_queue",
    "vbd_previous_pose_queued",
    "vbd_coupling_previous_pose_queued",
    "vbd_pose_history_applied_exactly_once",
    "vbd_pose_history_failed",
    "vbd_pose_history_superseded",
    "vbd_pose_history_pending_after_first_solve",
    "vbd_pose_history_application_count_delta",
    "vbd_pose_history_expected_body_count",
    "vbd_pose_history_body_application_count_delta",
    "vbd_pose_history_generation",
    "vbd_pose_history_body_order_exact",
    "vbd_pose_history_world_order_exact",
    "vbd_pose_history_entry_name",
    "vbd_pose_history_body_count",
    "stored_state_finite",
    "stored_task_state_finite_and_normalized",
    "stored_drive_disabled",
    "stored_maximum_cable_speed_m_s",
    "stored_maximum_arm_joint_speed_rad_s",
    "stored_maximum_finger_joint_speed_m_s",
    "stored_maximum_arm_target_tracking_error_rad",
    "stored_arm_target_tracking_error_by_joint_rad",
    "stored_arm_target_tracking_bounded",
    "all_post_step_state_finite",
    "all_post_step_task_state_finite_and_normalized",
    "all_post_step_collision_free",
    "all_post_step_drive_disabled",
    "all_post_step_expected_contact_state",
    "all_post_step_bilateral_grasp",
    "all_post_step_proxy_bilateral_contact",
    "all_post_step_zero_proxy_contacts",
    "all_post_step_arm_target_tracking_bounded",
    "maximum_body_excursion_m",
    "maximum_plug_excursion_m",
    "maximum_socket_excursion_m",
    "maximum_post_step_cable_speed_m_s",
    "maximum_post_step_arm_joint_speed_rad_s",
    "maximum_post_step_finger_joint_speed_m_s",
    "maximum_post_step_arm_target_tracking_error_rad",
    "maximum_arm_target_tracking_error_by_joint_rad",
    "arm_target_semantics",
    "arm_target_tracking_limits_rad",
    "maximum_arm_target_drift_rad",
    "absolute_target_stable",
    "final_cable_speed_m_s",
    "final_arm_joint_speed_rad_s",
    "final_finger_joint_speed_m_s",
    "minimum_left_proxy_contact_count",
    "minimum_right_proxy_contact_count",
    "maximum_left_proxy_contact_count",
    "maximum_right_proxy_contact_count",
    "maximum_invalid_contact_count",
    "no_contact_overflow",
    "invalid_contact_pairs",
    "maximum_arm_target_clamp_delta_rad",
    "zero_action_unclamped",
    "maximum_allowed_arm_joint_speed_rad_s",
    "maximum_allowed_finger_joint_speed_m_s",
    "maximum_allowed_cable_speed_m_s",
    "maximum_allowed_body_excursion_m",
    "maximum_allowed_plug_excursion_m",
    "maximum_allowed_socket_excursion_m",
    "maximum_allowed_arm_target_clamp_delta_rad",
}
_ROW_METRIC_NAMES = {
    "initial_goal_error_artifact",
    "initial_goal_error_replayed",
    "initial_goal_error_matches",
    "initial_tcp_distance_artifact_m",
    "initial_tcp_distance_replayed_m",
    "initial_tcp_distance_matches",
    "initial_tcp_xyz_replayed_m",
    "settle_socket_drift_m",
    "settle_plug_drift_m",
    "settle_max_body_drift_m",
    "capture_max_cable_speed_m_s",
    "settled_max_cable_speed_m_s",
    "maximum_reset_arm_target_clamp_delta_rad",
    "maximum_reset_arm_target_drift_rad",
    "reset_maximum_invalid_contacts",
    "recovery_invalid_contacts",
    "recovery_goal_error",
    "recovery_plug_speed",
    "recovery_maximum_body_excursion_m",
    "recovery_maximum_cable_linear_speed_m_s",
    "recovery_maximum_arm_joint_speed_rad_s",
    "recovery_maximum_finger_joint_speed_m_s",
    "acquisition",
}
_STARTED_GRASP_ACQUISITION_NAMES = {
    "started_with_physical_bilateral_grasp",
    "last_arm_target",
    "last_finger_target",
}
_OPEN_GRASP_ACQUISITION_NAMES = {
    "open_clearance_above_cfg_target_m",
    "route_world_height_m",
    "approach_abort_reason",
    "approach_samples",
    "clearance_approach_valid",
    "clearance_tcp_error_m",
    "open_descent_valid",
    "maximum_open_descent_tcp_error_m",
    "open_approach_all_samples_collision_free",
    "open_approach_all_samples_zero_proxy_contacts",
    "open_approach_all_samples_finite",
    "open_approach_all_samples_drives_disabled",
    "open_approach_maximum_plug_drift_m",
    "open_approach_maximum_left_proxy_contacts",
    "open_approach_maximum_right_proxy_contacts",
    "open_approach_any_contact_overflow",
    "open_approach_invalid_pairs",
    "contact_preclose_invalid_contacts",
    "contact_preclose_tcp_error_m",
    "maximum_tcp_distance_m",
    "minimum_bilateral_deflection_m",
    "left_proxy_contacts",
    "right_proxy_contacts",
    "invalid_contacts",
    "post_contact_settle",
    "lane_failure_masks",
    "last_arm_target",
    "last_finger_target",
}
_POST_CONTACT_SETTLE_NAMES = {
    "all_samples_finite",
    "all_samples_collision_free",
    "all_samples_bilateral_proxy_contact",
    "all_samples_drives_disabled",
    "any_contact_overflow",
    "invalid_contact_pairs",
    "maximum_cable_speed_m_s",
    "maximum_plug_linear_speed_m_s",
    "maximum_plug_angular_speed_rad_s",
    "final_cable_speed_m_s",
    "final_plug_linear_speed_m_s",
    "final_plug_angular_speed_rad_s",
    "final_arm_joint_speed_rad_s",
    "final_finger_joint_speed_m_s",
}


def franka_rj45_validation_source_sha256(
    repo_root: str | Path | None = None,
    *,
    include_fast_validator: bool = False,
) -> dict[str, str]:
    """Hash the complete repository source closure used by physical validation.

    Args:
        repo_root: Repository root containing ``uv.lock`` and the configured
            package source roots. When omitted, the root is discovered from
            this module's location.

    Returns:
        Repository-relative POSIX paths mapped to lowercase SHA-256 digests.

    Raises:
        FileNotFoundError: If no complete source checkout can be resolved.
    """
    if repo_root is None:
        candidates = Path(__file__).resolve().parents
    else:
        candidates = (Path(repo_root).expanduser().resolve(),)
    source_files = _FAST_VALIDATION_SOURCE_FILES if include_fast_validator else _VALIDATION_SOURCE_FILES
    root: Path | None = None
    for candidate in candidates:
        if all((candidate / name).is_dir() for name in _VALIDATION_SOURCE_ROOTS) and all(
            (candidate / name).is_file() for name in source_files
        ):
            root = candidate
            break
    if root is None:
        requested = "this module's parents" if repo_root is None else str(Path(repo_root).expanduser())
        raise FileNotFoundError(
            f"Could not resolve the complete Franka RJ45 validation source closure from {requested}."
        )

    relative_names = set(source_files)
    for relative_root in _VALIDATION_SOURCE_ROOTS:
        source_root = root / relative_root
        relative_names.update(
            source.relative_to(root).as_posix()
            for source in source_root.rglob("*")
            if source.is_file() and source.suffix in {".py", ".pyi"}
        )
    return {
        relative_name: hashlib.sha256((root / relative_name).read_bytes()).hexdigest()
        for relative_name in sorted(relative_names)
    }


def reset_dataset_validate_runtime(
    payload: Mapping[str, Any],
    *,
    expected_content_sha256: str | None = None,
    expected_task_contract: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    """Validate a Franka RJ45 pick-and-insert reset dataset for runtime replay.

    The task-body count is part of the task contract instead of this schema so
    asset topology changes cannot silently reinterpret tensor dimensions.

    Args:
        payload: Safely loaded reset-dataset payload.
        expected_content_sha256: Optional configured artifact content digest.
        expected_task_contract: Optional exact runtime task contract.

    Returns:
        The validated metadata, reset rows, and canonical goal state.

    Raises:
        TypeError: If a required container or tensor has the wrong type.
        ValueError: If the artifact is malformed or incompatible with the runtime.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("Reset dataset payload must be a mapping.")
    if expected_content_sha256 is not None:
        _validate_sha256(expected_content_sha256, "expected_content_sha256")
    if expected_task_contract is not None and not isinstance(expected_task_contract, Mapping):
        raise TypeError("expected_task_contract must be a mapping or None.")

    if payload.get("format") != FRANKA_RJ45_PICK_INSERT_RESET_DATASET_FORMAT:
        raise ValueError(f"Expected reset dataset format {FRANKA_RJ45_PICK_INSERT_RESET_DATASET_FORMAT!r}.")
    if payload.get("schema_version") != FRANKA_RJ45_PICK_INSERT_RESET_DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"Expected reset dataset schema version {FRANKA_RJ45_PICK_INSERT_RESET_DATASET_SCHEMA_VERSION}."
        )
    _validate_exact_names(payload, _PAYLOAD_NAMES, path="payload")

    metadata = _require_mapping(payload, "metadata")
    states = _require_mapping(payload, "states")
    goal_state = _require_mapping(payload, "goal_state")
    task_contract = _require_mapping(metadata, "task_contract", path="metadata")
    _validate_plain_value(metadata, path="metadata")
    _validate_tensor_mapping_safety(states, path="states")
    _validate_tensor_mapping_safety(goal_state, path="goal_state")

    content_sha256 = payload.get("content_sha256")
    _validate_sha256(content_sha256, "content_sha256")
    if content_sha256 != reset_dataset_content_digest(payload):
        raise ValueError("Reset dataset content digest does not match its payload.")
    if expected_content_sha256 is not None and content_sha256 != expected_content_sha256:
        raise ValueError("Reset dataset content digest does not match the configured digest.")

    if expected_task_contract is not None:
        _validate_plain_value(expected_task_contract, path="expected_task_contract")
        if reset_dataset_digest(_json_normalize(task_contract)) != reset_dataset_digest(
            _json_normalize(expected_task_contract)
        ):
            raise ValueError("Reset dataset task contract does not exactly match the runtime task contract.")
    task_body_count = _resolve_task_body_count(task_contract, expected_task_contract)
    _validate_vbd_pose_history_contract(task_contract, task_body_count=task_body_count)

    row_count = _validate_tensors(
        states,
        _state_tensor_specs(task_body_count),
        leading_count=None,
        path="states",
    )
    _validate_tensors(
        goal_state,
        _goal_tensor_specs(task_body_count),
        leading_count=0,
        path="goal_state",
    )
    assert row_count is not None
    _validate_state_semantics(states)
    _validate_persistent_target_semantics(task_contract, states, goal_state)
    _validate_pose_quaternions(states["task_body_pose"], path="states.task_body_pose")
    _validate_pose_quaternions(states["task_body_previous_pose"], path="states.task_body_previous_pose")
    _validate_pose_quaternions(
        states["task_body_coupling_previous_pose"],
        path="states.task_body_coupling_previous_pose",
    )
    _validate_pose_quaternions(states["goal_task_body_pose"], path="states.goal_task_body_pose")
    _validate_pose_quaternions(goal_state["task_body_pose"], path="goal_state.task_body_pose")
    _validate_pose_quaternions(goal_state["task_body_previous_pose"], path="goal_state.task_body_previous_pose")
    _validate_pose_quaternions(
        goal_state["task_body_coupling_previous_pose"],
        path="goal_state.task_body_coupling_previous_pose",
    )

    return metadata, states, goal_state


def reset_dataset_validate_phase_row_counts(
    phases: torch.Tensor,
    *,
    expected_rows_per_phase: int,
) -> tuple[int, ...]:
    """Require an exact, balanced production row count across all six phases."""
    if not _is_plain_int(expected_rows_per_phase) or expected_rows_per_phase < 1:
        raise ValueError("expected_rows_per_phase must be a positive plain integer.")
    if not isinstance(phases, torch.Tensor) or phases.dtype != torch.int64 or phases.ndim != 1:
        raise ValueError("phases must be a one-dimensional torch.int64 tensor.")
    if not bool(torch.all((phases >= 0) & (phases < len(PICK_INSERT_RESET_PHASE_IDS)))):
        raise ValueError("phases must contain only phase identifiers in [0, 5].")

    counts_tensor = torch.bincount(phases, minlength=len(PICK_INSERT_RESET_PHASE_IDS))
    counts = tuple(int(count) for count in counts_tensor.detach().cpu().tolist())
    expected = (expected_rows_per_phase,) * len(PICK_INSERT_RESET_PHASE_IDS)
    if counts != expected:
        raise ValueError(
            "Reset dataset phase counts do not match the configured production size: "
            f"expected {expected}, got {counts}."
        )
    return counts


def pick_insert_reset_dataset_row_digest(states: Mapping[str, torch.Tensor], row_id: int) -> str:
    """Digest every stored state tensor for one final artifact row.

    Args:
        states: Complete reset-dataset state tensor mapping.
        row_id: Final artifact row index.

    Returns:
        Lowercase hexadecimal SHA-256 digest for the row.

    Raises:
        ValueError: If the mapping or row index is invalid.
    """
    if not isinstance(states, Mapping) or set(states) != set(RESET_DATASET_STATE_NAMES):
        raise ValueError("states must contain exactly the pick-insert reset-dataset state tensors.")
    if not _is_plain_int(row_id) or row_id < 0:
        raise ValueError("row_id must be a non-negative plain integer.")
    row_count: int | None = None
    row: dict[str, torch.Tensor] = {}
    for name in RESET_DATASET_STATE_NAMES:
        tensor = states[name]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1:
            raise ValueError(f"states.{name} must be a tensor with a leading row dimension.")
        if row_count is None:
            row_count = int(tensor.shape[0])
            if row_id >= row_count:
                raise ValueError("row_id lies outside the reset-dataset state tensors.")
        elif int(tensor.shape[0]) != row_count:
            raise ValueError("All state tensors must have the same leading row count.")
        row[name] = tensor[row_id]
    return reset_dataset_digest(row)


def pick_insert_fast_reset_phase_0_band_fraction_tolerance(row_count: int) -> float:
    """Return the deterministic phase-0 band-fraction tolerance for one bank size.

    Args:
        row_count: Number of accepted phase-0 rows.

    Returns:
        Maximum allowed absolute deviation from a configured band weight.

    Raises:
        ValueError: If ``row_count`` is not a positive plain integer.
    """
    if not _is_plain_int(row_count) or row_count < 1:
        raise ValueError("row_count must be a positive plain integer.")
    contract = PICK_INSERT_FAST_RESET_PHASE_0_BAND_ACCEPTANCE_CONTRACT
    return max(
        float(contract["maximum_absolute_fraction_error"]),
        int(contract["small_bank_discretization_allowance_rows"]) / row_count,
    )


def reset_dataset_validate_full_pick_diversity(
    states: Mapping[str, torch.Tensor],
    *,
    task_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate offline phase-5 coverage and return JSON-safe evidence."""
    if not isinstance(states, Mapping) or not isinstance(task_contract, Mapping):
        raise TypeError("states and task_contract must be mappings.")
    required_state_names = {"phase", "task_body_pose", "arm_joint_position", "initial_tcp_grasp_distance"}
    missing = sorted(required_state_names - set(states))
    if missing:
        raise ValueError(f"Full-pick diversity validation is missing state tensors: {missing}.")

    phases = states["phase"]
    task_pose = states["task_body_pose"]
    arm_position = states["arm_joint_position"]
    tcp_distance = states["initial_tcp_grasp_distance"]
    if not isinstance(phases, torch.Tensor) or phases.dtype != torch.int64 or phases.ndim != 1:
        raise ValueError("states.phase must be a one-dimensional torch.int64 tensor.")
    row_count = phases.numel()
    if (
        not isinstance(task_pose, torch.Tensor)
        or task_pose.ndim != 3
        or task_pose.shape[0] != row_count
        or task_pose.shape[-1] != 7
    ):
        raise ValueError("states.task_body_pose must have shape (row_count, body_count, 7).")
    if not isinstance(arm_position, torch.Tensor) or tuple(arm_position.shape) != (row_count, 7):
        raise ValueError("states.arm_joint_position must have shape (row_count, 7).")
    if not isinstance(tcp_distance, torch.Tensor) or tuple(tcp_distance.shape) != (row_count,):
        raise ValueError("states.initial_tcp_grasp_distance must have shape (row_count,).")

    body_order = task_contract.get("task_body_order")
    pick_contract = task_contract.get("pick_insert")
    if not isinstance(body_order, (tuple, list)) or not isinstance(pick_contract, Mapping):
        raise ValueError("task_contract must define task_body_order and pick_insert mappings.")
    try:
        socket_index = body_order.index("socket")
        plug_index = body_order.index("plug")
    except ValueError as exc:
        raise ValueError("task_contract.task_body_order must contain socket and plug bodies.") from exc
    if max(socket_index, plug_index) >= task_pose.shape[1]:
        raise ValueError("task_contract body indices exceed states.task_body_pose.")

    diversity_contract = pick_contract.get("full_pick_diversity")
    if not isinstance(diversity_contract, Mapping):
        raise ValueError("task_contract.pick_insert.full_pick_diversity must be a mapping.")
    rows_per_phase = pick_contract.get("reset_dataset_rows_per_phase")
    round_decimals = diversity_contract.get("round_decimals")
    unique_socket_minimum = diversity_contract.get("minimum_unique_socket_rows")
    unique_plug_minimum = diversity_contract.get("minimum_unique_plug_rows")
    unique_arm_minimum = diversity_contract.get("minimum_unique_arm_rows")
    for name, value in (
        ("reset_dataset_rows_per_phase", rows_per_phase),
        ("round_decimals", round_decimals),
        ("minimum_unique_socket_rows", unique_socket_minimum),
        ("minimum_unique_plug_rows", unique_plug_minimum),
        ("minimum_unique_arm_rows", unique_arm_minimum),
    ):
        minimum = 0 if name == "round_decimals" else 1
        if not _is_plain_int(value) or value < minimum:
            raise ValueError(f"task_contract pick-insert diversity field {name} is invalid.")
    if round_decimals > 8:
        raise ValueError("task_contract pick-insert diversity field round_decimals must be no greater than 8.")
    if any(minimum > rows_per_phase for minimum in (unique_socket_minimum, unique_plug_minimum, unique_arm_minimum)):
        raise ValueError("task_contract unique-row minima cannot exceed reset_dataset_rows_per_phase.")

    socket_fraction = _finite_contract_scalar(
        diversity_contract,
        "minimum_socket_span_fraction",
        positive=True,
        maximum=1.0,
    )
    pickup_fraction = _finite_contract_scalar(
        diversity_contract,
        "minimum_pickup_span_fraction",
        positive=True,
        maximum=1.0,
    )
    arm_fraction = _finite_contract_scalar(
        diversity_contract,
        "minimum_arm_joint_span_fraction",
        positive=True,
        maximum=1.0,
    )
    tcp_distance_minimum = _finite_contract_scalar(
        diversity_contract,
        "minimum_tcp_grasp_distance_span_m",
        positive=True,
    )
    arm_noise = _finite_contract_scalar(pick_contract, "arm_reset_joint_noise", positive=True)
    socket_lower = _finite_contract_vector(pick_contract, "socket_position_lower", length=3)
    socket_upper = _finite_contract_vector(pick_contract, "socket_position_upper", length=3)
    pickup_lower = _finite_contract_vector(pick_contract, "pickup_position_lower", length=3)
    pickup_upper = _finite_contract_vector(pick_contract, "pickup_position_upper", length=3)
    socket_yaw_range = _finite_contract_vector(pick_contract, "socket_yaw_range", length=2)
    pickup_yaw_range = _finite_contract_vector(pick_contract, "pickup_yaw_range", length=2)
    for name, lower, upper, indices in (
        ("socket_position", socket_lower, socket_upper, (0, 1)),
        ("pickup_position", pickup_lower, pickup_upper, (0, 1)),
    ):
        if any(upper[index] <= lower[index] for index in indices):
            raise ValueError(f"task_contract {name} XY ranges must have positive span.")
    if socket_yaw_range[1] <= socket_yaw_range[0] or pickup_yaw_range[1] <= pickup_yaw_range[0]:
        raise ValueError("task_contract socket/pickup yaw ranges must have positive span.")

    phase_mask = phases == PICK_INSERT_RESET_PHASE_IDS[-1]
    full_pick_count = int(phase_mask.sum().item())
    socket_pose = task_pose[phase_mask, socket_index].to(dtype=torch.float64)
    plug_pose = task_pose[phase_mask, plug_index].to(dtype=torch.float64)
    arm_rows = arm_position[phase_mask].to(dtype=torch.float64)
    tcp_rows = tcp_distance[phase_mask].to(dtype=torch.float64)

    unique_socket_rows = _rounded_unique_pose_count(socket_pose, decimals=round_decimals)
    unique_plug_rows = _rounded_unique_pose_count(plug_pose, decimals=round_decimals)
    unique_arm_rows = _rounded_unique_row_count(arm_rows, decimals=round_decimals)
    socket_xy_span = _column_span(socket_pose[:, :2])
    plug_xy_span = _column_span(plug_pose[:, :2])
    socket_yaw_span = _quaternion_yaw_span(socket_pose[:, 3:7])
    plug_yaw_span = _quaternion_yaw_span(plug_pose[:, 3:7])
    arm_joint_span = _column_span(arm_rows)
    tcp_distance_span = _column_span(tcp_rows.unsqueeze(-1))[0]

    socket_required_xy_span = [socket_fraction * (socket_upper[index] - socket_lower[index]) for index in (0, 1)]
    pickup_required_xy_span = [pickup_fraction * (pickup_upper[index] - pickup_lower[index]) for index in (0, 1)]
    socket_required_yaw_span = socket_fraction * (socket_yaw_range[1] - socket_yaw_range[0])
    pickup_required_yaw_span = pickup_fraction * (pickup_yaw_range[1] - pickup_yaw_range[0])
    arm_required_span = arm_fraction * 2.0 * arm_noise

    evidence: dict[str, Any] = {
        "phase": PICK_INSERT_RESET_PHASE_IDS[-1],
        "row_count": full_pick_count,
        "required_row_count": rows_per_phase,
        "round_decimals": round_decimals,
        "unique_socket_rows": unique_socket_rows,
        "unique_plug_rows": unique_plug_rows,
        "unique_arm_rows": unique_arm_rows,
        "required_unique_socket_rows": unique_socket_minimum,
        "required_unique_plug_rows": unique_plug_minimum,
        "required_unique_arm_rows": unique_arm_minimum,
        "socket_xy_span_m": socket_xy_span,
        "required_socket_xy_span_m": socket_required_xy_span,
        "socket_yaw_span_rad": socket_yaw_span,
        "required_socket_yaw_span_rad": socket_required_yaw_span,
        "plug_pickup_xy_span_m": plug_xy_span,
        "required_plug_pickup_xy_span_m": pickup_required_xy_span,
        "plug_pickup_yaw_span_rad": plug_yaw_span,
        "required_plug_pickup_yaw_span_rad": pickup_required_yaw_span,
        "arm_joint_span_rad": arm_joint_span,
        "required_each_arm_joint_span_rad": arm_required_span,
        "initial_tcp_grasp_distance_span_m": tcp_distance_span,
        "required_initial_tcp_grasp_distance_span_m": tcp_distance_minimum,
    }
    failures: list[str] = []
    if full_pick_count != rows_per_phase:
        failures.append("row_count")
    if unique_socket_rows < unique_socket_minimum:
        failures.append("unique_socket_rows")
    if unique_plug_rows < unique_plug_minimum:
        failures.append("unique_plug_rows")
    if unique_arm_rows < unique_arm_minimum:
        failures.append("unique_arm_rows")
    if any(observed < required for observed, required in zip(socket_xy_span, socket_required_xy_span, strict=True)):
        failures.append("socket_xy_span")
    if socket_yaw_span < socket_required_yaw_span:
        failures.append("socket_yaw_span")
    if any(observed < required for observed, required in zip(plug_xy_span, pickup_required_xy_span, strict=True)):
        failures.append("plug_pickup_xy_span")
    if plug_yaw_span < pickup_required_yaw_span:
        failures.append("plug_pickup_yaw_span")
    if any(observed < arm_required_span for observed in arm_joint_span):
        failures.append("arm_joint_span")
    if tcp_distance_span < tcp_distance_minimum:
        failures.append("initial_tcp_grasp_distance_span")
    evidence["passed"] = not failures
    evidence["failures"] = failures
    if failures:
        raise ValueError(f"Full-pick reset dataset diversity gate failed: {failures}; evidence={evidence}")
    return evidence


def _validate_validation_cfg(values: Mapping[str, Any]) -> bool:
    """Require the complete canonical full-validator configuration."""
    _validate_exact_names(values, _VALIDATION_CFG_NAMES, path="report.validation_cfg")
    seed = values.get("seed")
    if not _is_plain_int(seed):
        return False
    return all(values.get(name) == expected for name, expected in _VALIDATION_CFG_CANONICAL_VALUES.items())


def _validate_physical_contract(values: Mapping[str, Any], task_contract: Mapping[str, Any]) -> bool:
    """Cross-check the live tool snapshot against the path-independent task contract."""
    _validate_exact_names(values, _PHYSICAL_CONTRACT_NAMES, path="report.physical_contract")
    pick_insert = task_contract.get("pick_insert")
    physics = task_contract.get("rj45_physics")
    geometry = task_contract.get("validation_geometry")
    if not all(isinstance(item, Mapping) for item in (pick_insert, physics, geometry)):
        return False
    expected = {
        "finger_closed_target_m": pick_insert.get("finger_closed_position"),
        "live_finger_close_position_m": pick_insert.get("finger_closed_position"),
        "configured_grasp_proxy_raw_friction": physics.get("grasp_proxy_raw_friction"),
        "live_grasp_proxy_raw_friction": physics.get("grasp_proxy_raw_friction"),
        "effective_finger_proxy_friction": physics.get("grasp_contact_effective_friction"),
        "success_max_plug_speed": geometry.get("success_max_plug_speed"),
    }
    return all(values.get(name) == expected_value for name, expected_value in expected.items())


def _validate_physics_versions(values: Mapping[str, Any], task_contract: Mapping[str, Any]) -> bool:
    """Require complete, non-empty version evidence and cross-check shared packages."""
    _validate_exact_names(values, _PHYSICS_VERSION_NAMES, path="report.physics_versions")
    if any(not isinstance(value, str) or not value for value in values.values()):
        return False
    runtime = task_contract.get("runtime_physics_versions")
    return (
        isinstance(runtime, Mapping)
        and values.get("newton") == runtime.get("newton")
        and values.get("warp") == runtime.get("warp-lang")
    )


def _validate_exact_finite_vector(value: Any, *, length: int, nonnegative: bool = False) -> bool:
    if not isinstance(value, list | tuple) or len(value) != length:
        return False
    return all(
        isinstance(item, int | float)
        and not isinstance(item, bool)
        and math.isfinite(float(item))
        and (not nonnegative or float(item) >= 0.0)
        for item in value
    )


def _validate_finite_sequence(value: Any, *, nonnegative: bool = False) -> bool:
    if not isinstance(value, list | tuple) or not value:
        return False
    return all(
        isinstance(item, int | float)
        and not isinstance(item, bool)
        and math.isfinite(float(item))
        and (not nonnegative or float(item) >= 0.0)
        for item in value
    )


def _validate_bool_sequence(value: Any, *, require_any: bool = False) -> bool:
    return (
        isinstance(value, list | tuple)
        and bool(value)
        and all(isinstance(item, bool) for item in value)
        and (not require_any or any(value))
    )


def _validate_nonnegative_int_sequence(value: Any) -> bool:
    return isinstance(value, list | tuple) and bool(value) and all(_plain_nonnegative_int(item) for item in value)


def _validate_offline_diversity_evidence(value: Mapping[str, Any], *, expected_row_count: int) -> bool:
    """Validate the internally consistent result of independent artifact analysis."""
    count_pairs = (
        ("unique_socket_rows", "required_unique_socket_rows"),
        ("unique_plug_rows", "required_unique_plug_rows"),
        ("unique_arm_rows", "required_unique_arm_rows"),
    )
    if not (
        value.get("phase") == _FULL_PICK_PHASE
        and _is_plain_int(value.get("row_count"))
        and value["row_count"] == expected_row_count
        and _is_plain_int(value.get("required_row_count"))
        and value["required_row_count"] == expected_row_count
        and _is_plain_int(value.get("round_decimals"))
        and value["round_decimals"] == _FULL_PICK_TCP_ROUND_DECIMALS
        and value.get("passed") is True
        and value.get("failures") == []
    ):
        return False
    for observed_name, required_name in count_pairs:
        observed = value.get(observed_name)
        required = value.get(required_name)
        if not (
            _is_plain_int(observed) and _is_plain_int(required) and 1 <= required <= observed <= expected_row_count
        ):
            return False
    for observed_name, required_name, length in (
        ("socket_xy_span_m", "required_socket_xy_span_m", 2),
        ("plug_pickup_xy_span_m", "required_plug_pickup_xy_span_m", 2),
    ):
        observed = value.get(observed_name)
        required = value.get(required_name)
        if not (
            _validate_exact_finite_vector(observed, length=length, nonnegative=True)
            and _validate_exact_finite_vector(required, length=length, nonnegative=True)
            and all(float(actual) >= float(minimum) for actual, minimum in zip(observed, required, strict=True))
        ):
            return False
    arm_span = value.get("arm_joint_span_rad")
    arm_required = value.get("required_each_arm_joint_span_rad")
    return (
        _finite_nonnegative_number(value.get("socket_yaw_span_rad"))
        and _finite_nonnegative_number(value.get("required_socket_yaw_span_rad"))
        and value["socket_yaw_span_rad"] >= value["required_socket_yaw_span_rad"]
        and _finite_nonnegative_number(value.get("plug_pickup_yaw_span_rad"))
        and _finite_nonnegative_number(value.get("required_plug_pickup_yaw_span_rad"))
        and value["plug_pickup_yaw_span_rad"] >= value["required_plug_pickup_yaw_span_rad"]
        and _validate_exact_finite_vector(arm_span, length=7, nonnegative=True)
        and _finite_nonnegative_number(arm_required)
        and all(float(actual) >= float(arm_required) for actual in arm_span)
        and _finite_nonnegative_number(value.get("initial_tcp_grasp_distance_span_m"))
        and _finite_nonnegative_number(value.get("required_initial_tcp_grasp_distance_span_m"))
        and value["initial_tcp_grasp_distance_span_m"] >= value["required_initial_tcp_grasp_distance_span_m"]
    )


def _validate_started_grasp_acquisition(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    _validate_exact_names(value, _STARTED_GRASP_ACQUISITION_NAMES, path="report.rows[].metrics.acquisition")
    return (
        _validate_bool_sequence(value.get("started_with_physical_bilateral_grasp"), require_any=True)
        and isinstance(value.get("last_arm_target"), list)
        and bool(value["last_arm_target"])
        and all(_validate_exact_finite_vector(row, length=7) for row in value["last_arm_target"])
        and isinstance(value.get("last_finger_target"), list)
        and bool(value["last_finger_target"])
        and all(_validate_exact_finite_vector(row, length=2) for row in value["last_finger_target"])
    )


def _validate_open_grasp_acquisition(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    _validate_exact_names(value, _OPEN_GRASP_ACQUISITION_NAMES, path="report.rows[].metrics.acquisition")
    post_settle = value.get("post_contact_settle")
    lane_failures = value.get("lane_failure_masks")
    if not isinstance(post_settle, Mapping) or not isinstance(lane_failures, Mapping):
        return False
    _validate_exact_names(
        post_settle,
        _POST_CONTACT_SETTLE_NAMES,
        path="report.rows[].metrics.acquisition.post_contact_settle",
    )
    boolean_sequences = (
        "clearance_approach_valid",
        "open_descent_valid",
        "open_approach_all_samples_collision_free",
        "open_approach_all_samples_zero_proxy_contacts",
        "open_approach_all_samples_finite",
        "open_approach_all_samples_drives_disabled",
    )
    numeric_sequences = (
        "clearance_tcp_error_m",
        "maximum_open_descent_tcp_error_m",
        "open_approach_maximum_plug_drift_m",
        "contact_preclose_tcp_error_m",
    )
    count_sequences = (
        "open_approach_maximum_left_proxy_contacts",
        "open_approach_maximum_right_proxy_contacts",
        "contact_preclose_invalid_contacts",
        "left_proxy_contacts",
        "right_proxy_contacts",
        "invalid_contacts",
    )
    post_boolean_sequences = (
        "all_samples_finite",
        "all_samples_collision_free",
        "all_samples_bilateral_proxy_contact",
        "all_samples_drives_disabled",
    )
    post_numeric_sequences = (
        "maximum_cable_speed_m_s",
        "maximum_plug_linear_speed_m_s",
        "maximum_plug_angular_speed_rad_s",
        "final_cable_speed_m_s",
        "final_plug_linear_speed_m_s",
        "final_plug_angular_speed_rad_s",
        "final_arm_joint_speed_rad_s",
        "final_finger_joint_speed_m_s",
    )
    lane_count = len(value["last_arm_target"]) if isinstance(value.get("last_arm_target"), list) else 0
    lane_sequences = (*boolean_sequences, *numeric_sequences, *count_sequences)
    post_sequences = (*post_boolean_sequences, *post_numeric_sequences)
    return (
        value.get("approach_abort_reason") is None
        and _is_plain_int(value.get("approach_samples"))
        and value["approach_samples"] > 0
        and value.get("open_clearance_above_cfg_target_m") == _VALIDATION_CFG_CANONICAL_VALUES["grasp_open_clearance_m"]
        and value.get("route_world_height_m") == _VALIDATION_CFG_CANONICAL_VALUES["grasp_route_world_height_m"]
        and all(_validate_bool_sequence(value.get(name), require_any=True) for name in boolean_sequences)
        and all(_validate_finite_sequence(value.get(name), nonnegative=True) for name in numeric_sequences)
        and all(_validate_nonnegative_int_sequence(value.get(name)) for name in count_sequences)
        and value.get("open_approach_any_contact_overflow") is False
        and value.get("open_approach_invalid_pairs") == []
        and _finite_nonnegative_number(value.get("maximum_tcp_distance_m"))
        and _finite_nonnegative_number(value.get("minimum_bilateral_deflection_m"))
        and not lane_failures
        and isinstance(value.get("last_arm_target"), list)
        and bool(value["last_arm_target"])
        and all(_validate_exact_finite_vector(row, length=7) for row in value["last_arm_target"])
        and isinstance(value.get("last_finger_target"), list)
        and bool(value["last_finger_target"])
        and all(_validate_exact_finite_vector(row, length=2) for row in value["last_finger_target"])
        and lane_count > 0
        and len(value["last_finger_target"]) == lane_count
        and all(len(value[name]) == lane_count for name in lane_sequences)
        and all(_validate_bool_sequence(post_settle.get(name), require_any=True) for name in post_boolean_sequences)
        and all(_validate_finite_sequence(post_settle.get(name), nonnegative=True) for name in post_numeric_sequences)
        and all(len(post_settle[name]) == lane_count for name in post_sequences)
        and post_settle.get("any_contact_overflow") is False
        and post_settle.get("invalid_contact_pairs") == []
    )


def _validate_row_metrics(value: Any, *, phase: int, reset_replay: Mapping[str, Any]) -> bool:
    """Validate raw row diagnostics and their duplicated reset evidence."""
    if not isinstance(value, Mapping):
        return False
    _validate_exact_names(value, _ROW_METRIC_NAMES, path="report.rows[].metrics")
    scalar_names = _ROW_METRIC_NAMES - {
        "initial_goal_error_matches",
        "initial_tcp_distance_matches",
        "initial_tcp_xyz_replayed_m",
        "acquisition",
        "reset_maximum_invalid_contacts",
        "recovery_invalid_contacts",
    }
    if not all(_finite_nonnegative_number(value.get(name)) for name in scalar_names):
        return False
    if not _validate_exact_finite_vector(value.get("initial_tcp_xyz_replayed_m"), length=3):
        return False
    goal_matches = abs(value["initial_goal_error_artifact"] - value["initial_goal_error_replayed"]) <= 5.0e-4
    tcp_matches = abs(value["initial_tcp_distance_artifact_m"] - value["initial_tcp_distance_replayed_m"]) <= 2.0e-3
    reset_pairs = {
        "settle_socket_drift_m": "maximum_socket_excursion_m",
        "settle_plug_drift_m": "maximum_plug_excursion_m",
        "settle_max_body_drift_m": "maximum_body_excursion_m",
        "capture_max_cable_speed_m_s": "stored_maximum_cable_speed_m_s",
        "settled_max_cable_speed_m_s": "final_cable_speed_m_s",
        "maximum_reset_arm_target_clamp_delta_rad": "maximum_arm_target_clamp_delta_rad",
        "maximum_reset_arm_target_drift_rad": "maximum_arm_target_drift_rad",
        "reset_maximum_invalid_contacts": "maximum_invalid_contact_count",
    }
    duplicates_match = all(value.get(metric) == reset_replay.get(raw) for metric, raw in reset_pairs.items())
    recovery_speed_gates = PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["per_step_rejection_gates"]
    # Body distance includes randomized cable geometry, and transient cable speed is observation-only.
    recovery_valid = (
        _plain_nonnegative_int(value.get("reset_maximum_invalid_contacts"))
        and value["reset_maximum_invalid_contacts"] == 0
        and _plain_nonnegative_int(value.get("recovery_invalid_contacts"))
        and value["recovery_invalid_contacts"] == 0
        and value["recovery_plug_speed"] <= 0.03
        and value["recovery_maximum_arm_joint_speed_rad_s"] <= recovery_speed_gates["arm_joint_speed_rad_s"]
        and value["recovery_maximum_finger_joint_speed_m_s"] <= recovery_speed_gates["finger_joint_speed_m_s"]
    )
    acquisition_valid = (
        _validate_started_grasp_acquisition(value.get("acquisition"))
        if phase <= 3
        else _validate_open_grasp_acquisition(value.get("acquisition"))
    )
    return (
        value.get("initial_goal_error_matches") is goal_matches
        and goal_matches
        and value.get("initial_tcp_distance_matches") is tcp_matches
        and tcp_matches
        and duplicates_match
        and recovery_valid
        and acquisition_valid
    )


def _full_pick_live_evidence_matches_rows(value: Mapping[str, Any], rows: list[Any], *, required_rows: int) -> bool:
    """Recompute canonical live TCP diversity from per-row measurements."""
    phase_rows = [row for row in rows if isinstance(row, Mapping) and row.get("phase") == _FULL_PICK_PHASE]
    if len(phase_rows) != required_rows:
        return False
    xyz = [row.get("metrics", {}).get("initial_tcp_xyz_replayed_m") for row in phase_rows]
    distance = [row.get("metrics", {}).get("initial_tcp_distance_replayed_m") for row in phase_rows]
    if not all(_validate_exact_finite_vector(position, length=3) for position in xyz) or not all(
        _finite_nonnegative_number(item) for item in distance
    ):
        return False
    xyz_min = [min(float(position[axis]) for position in xyz) for axis in range(3)]
    xyz_max = [max(float(position[axis]) for position in xyz) for axis in range(3)]
    xyz_span = [upper - lower for lower, upper in zip(xyz_min, xyz_max, strict=True)]
    scale = 10**_FULL_PICK_TCP_ROUND_DECIMALS
    unique_positions = len({tuple(round(float(item) * scale) for item in position) for position in xyz})
    return (
        value.get("tcp_xyz_min_m") == xyz_min
        and value.get("tcp_xyz_max_m") == xyz_max
        and value.get("tcp_xyz_span_m") == xyz_span
        and value.get("unique_tcp_positions") == unique_positions
        and value.get("observed_minimum_tcp_to_grasp_distance_m") == min(float(item) for item in distance)
    )


def fast_reset_validation_report_validate_runtime(
    report: Mapping[str, Any],
    *,
    expected_content_sha256: str,
    expected_row_count: int,
    expected_phases: torch.Tensor | list[int] | tuple[int, ...],
    expected_task_contract: Mapping[str, Any],
    expected_validation_policy: Mapping[str, Any],
    expected_source_sha256: Mapping[str, Any],
    expected_asset_closure: Mapping[str, Any],
    expected_full_pick_diversity: Mapping[str, Any],
) -> dict[str, bool]:
    """Validate a source-bound, CPU-only fast reset admission certificate.

    This gate proves only initial-state admissibility: finite normalized state,
    bounded IK joints, workspace membership, analytic collision clearance, and
    phase semantics. It intentionally does not claim rollout stability or
    scripted task recovery.
    """
    if not isinstance(report, Mapping):
        raise TypeError("Fast reset validation report must be a mapping.")
    _validate_sha256(expected_content_sha256, "expected_content_sha256")
    if not _is_plain_int(expected_row_count) or expected_row_count < 1:
        raise ValueError("expected_row_count must be a positive integer.")
    for name, value in (
        ("expected_task_contract", expected_task_contract),
        ("expected_validation_policy", expected_validation_policy),
        ("expected_source_sha256", expected_source_sha256),
        ("expected_asset_closure", expected_asset_closure),
        ("expected_full_pick_diversity", expected_full_pick_diversity),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping.")

    expected_phase_by_row = _normalize_expected_phases(expected_phases, expected_row_count=expected_row_count)
    expected_counts = {phase: expected_phase_by_row.count(phase) for phase in PICK_INSERT_RESET_PHASE_IDS}
    _validate_report_value(report, path="report")
    _validate_exact_names(report, _FAST_VALIDATION_REPORT_NAMES, path="report")
    if report.get("format") != FRANKA_RJ45_PICK_INSERT_FAST_RESET_VALIDATION_FORMAT:
        raise ValueError(
            f"Expected fast reset validation format {FRANKA_RJ45_PICK_INSERT_FAST_RESET_VALIDATION_FORMAT!r}."
        )
    if report.get("schema_version") != FRANKA_RJ45_PICK_INSERT_FAST_RESET_VALIDATION_SCHEMA_VERSION:
        raise ValueError(
            "Expected fast reset validation schema version "
            f"{FRANKA_RJ45_PICK_INSERT_FAST_RESET_VALIDATION_SCHEMA_VERSION}."
        )

    content_sha256 = report.get("content_sha256")
    _validate_sha256(content_sha256, "content_sha256")
    artifact_sha256 = report.get("artifact_content_sha256")
    _validate_sha256(artifact_sha256, "artifact_content_sha256")
    report_contract = _require_mapping(report, "task_contract", path="report")
    policy = _require_mapping(report, "validation_policy", path="report")
    source_sha256 = _require_mapping(report, "source_sha256", path="report")
    asset_closure = _require_mapping(report, "asset_closure", path="report")
    diversity = _require_mapping(report, "full_pick_diversity", path="report")
    evidence_origin = _require_mapping(report, "evidence_origin", path="report")
    _validate_exact_names(
        evidence_origin,
        {"method", "bound_evidence_content_sha256"},
        path="report.evidence_origin",
    )
    if evidence_origin.get("method") not in {
        "current-source-cpu-static-plus-fast-ik-metadata",
        "current-source-cpu-static-plus-bound-prior-reset-replay",
    }:
        raise ValueError("Fast reset validation evidence_origin.method is not recognized.")
    _validate_sha256(evidence_origin.get("bound_evidence_content_sha256"), "bound_evidence_content_sha256")
    _validate_fast_source_sha256_snapshot(source_sha256)

    created_utc = report.get("created_utc")
    try:
        parsed_created_utc = datetime.fromisoformat(created_utc) if isinstance(created_utc, str) else None
    except ValueError:
        parsed_created_utc = None
    rows = report.get("rows")
    selected_row_ids = report.get("selected_row_ids")
    failed_row_ids = report.get("failed_row_ids")
    if not isinstance(rows, list) or not isinstance(selected_row_ids, list) or not isinstance(failed_row_ids, list):
        raise TypeError("Fast reset report rows and row-id fields must be lists.")

    cable_radius = report_contract.get("rj45_physics", {}).get("cable_radius")
    if not _finite_nonnegative_number(cable_radius):
        raise ValueError("Fast reset report task contract must define a finite cable radius.")
    policy_filter = policy.get("collision_filter")
    if not isinstance(policy_filter, Mapping):
        raise ValueError("Fast reset validation policy must define collision_filter.")
    expected_minimum_nonadjacent_separation = 2.0 * float(cable_radius) + float(
        policy_filter["minimum_nonadjacent_cable_surface_gap_m"]
    )
    row_ids: list[int] = []
    row_phases: list[int] = []
    rows_valid = True
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("Fast reset report rows must be mappings.")
        _validate_exact_names(row, _FAST_VALIDATION_ROW_NAMES, path="report.rows[]")
        checks = _require_mapping(row, "checks", path="report.rows[]")
        metrics = _require_mapping(row, "metrics", path="report.rows[]")
        _validate_exact_names(checks, _FAST_VALIDATION_CHECK_NAMES, path="report.rows[].checks")
        _validate_exact_names(metrics, _FAST_VALIDATION_METRIC_NAMES, path="report.rows[].metrics")
        row_id = row.get("row_id")
        phase = row.get("phase")
        if not _is_plain_int(row_id) or not 0 <= row_id < expected_row_count:
            raise ValueError("Fast reset report row_id is outside the artifact row range.")
        if not _is_plain_int(phase) or phase not in PICK_INSERT_RESET_PHASE_IDS:
            raise ValueError("Fast reset report phase must lie in [0, 5].")
        support_clearance = metrics.get("minimum_cable_support_clearance_m")
        metric_values_valid = (
            isinstance(support_clearance, int | float)
            and not isinstance(support_clearance, bool)
            and math.isfinite(float(support_clearance))
            and all(
                _finite_nonnegative_number(metrics.get(name))
                for name in _FAST_VALIDATION_METRIC_NAMES - {"minimum_cable_support_clearance_m"}
            )
        )
        metric_thresholds_valid = (
            metric_values_valid
            and metrics["minimum_joint_limit_margin_rad"] >= policy["joint_limit_margin_rad"]
            and metrics["minimum_workspace_margin_m"] >= 0.0
            and metrics["minimum_cable_support_clearance_m"] >= -policy_filter["maximum_table_penetration_m"]
            and metrics["minimum_nonadjacent_cable_separation_m"] >= expected_minimum_nonadjacent_separation
            and metrics["minimum_cable_socket_center_distance_m"]
            >= policy_filter["minimum_cable_socket_center_distance_m"]
        )
        row_valid = (
            row.get("passed") is True
            and set(checks) == _FAST_VALIDATION_CHECK_NAMES
            and all(checks.get(name) is True for name in _FAST_VALIDATION_CHECK_NAMES)
            and metric_thresholds_valid
        )
        rows_valid &= row_valid
        row_ids.append(row_id)
        row_phases.append(phase)

    report_counts = _normalize_phase_counts(report.get("phase_counts"), path="report.phase_counts", positive=True)
    observed_counts = {phase: row_phases.count(phase) for phase in PICK_INSERT_RESET_PHASE_IDS}
    checks = {
        "report_content_digest_matches": content_sha256 == reset_validation_report_content_digest(report),
        "created_utc_valid": parsed_created_utc is not None and parsed_created_utc.utcoffset() is not None,
        "artifact_content_digest_matches": artifact_sha256 == expected_content_sha256,
        "task_contract_matches": reset_dataset_digest(_json_normalize(report_contract))
        == reset_dataset_digest(_json_normalize(expected_task_contract)),
        "validation_policy_matches_schema": reset_dataset_digest(_json_normalize(policy))
        == reset_dataset_digest(_json_normalize(FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY)),
        "validation_policy_matches_runtime": reset_dataset_digest(_json_normalize(policy))
        == reset_dataset_digest(_json_normalize(expected_validation_policy)),
        "source_snapshot_matches_runtime": reset_dataset_digest(_json_normalize(source_sha256))
        == reset_dataset_digest(_json_normalize(expected_source_sha256)),
        "asset_closure_matches_task": reset_dataset_digest(_json_normalize(asset_closure))
        == reset_dataset_digest(_json_normalize(report_contract.get("external_assets"))),
        "asset_closure_matches_runtime": reset_dataset_digest(_json_normalize(asset_closure))
        == reset_dataset_digest(_json_normalize(expected_asset_closure)),
        "simulation_steps_zero": report.get("simulation_steps") == 0,
        "dynamics_replay_disabled": report.get("dynamics_replay") is False,
        "scripted_recovery_disabled": report.get("scripted_recovery") is False,
        "dataset_row_count_matches": report.get("dataset_row_count") == expected_row_count,
        "selected_every_row_once": selected_row_ids == list(range(expected_row_count)),
        "row_ids_exactly_once": row_ids == list(range(expected_row_count)),
        "row_phases_match_artifact": len(row_phases) == expected_row_count
        and all(row_phases[row_id] == expected_phase_by_row[row_id] for row_id in range(expected_row_count)),
        "reported_phase_counts_match": report_counts == expected_counts,
        "row_phase_counts_match": observed_counts == expected_counts,
        "full_pick_diversity_matches": reset_dataset_digest(_json_normalize(diversity))
        == reset_dataset_digest(_json_normalize(expected_full_pick_diversity))
        and _validate_offline_diversity_evidence(diversity, expected_row_count=expected_counts[_FULL_PICK_PHASE]),
        "every_row_passed": rows_valid,
        "no_failed_rows": failed_row_ids == [],
        "passed": report.get("passed") is True,
    }
    if not all(checks.values()):
        raise ValueError(f"Fast reset validation evidence is incomplete or incompatible: {checks}")
    return checks


def reset_validation_report_validate_runtime(
    report: Mapping[str, Any],
    *,
    expected_content_sha256: str,
    expected_row_count: int,
    expected_phases: torch.Tensor | list[int] | tuple[int, ...],
    expected_task_contract: Mapping[str, Any],
    expected_validation_policy: Mapping[str, Any],
    expected_source_sha256: Mapping[str, Any],
    expected_asset_closure: Mapping[str, Any],
    expected_full_pick_diversity: Mapping[str, Any],
) -> dict[str, bool]:
    """Require complete physical replay evidence for one pick-and-insert artifact.

    Args:
        report: JSON-safe full-dataset validation report.
        expected_content_sha256: Exact reset-artifact content digest.
        expected_row_count: Number of reset rows in the artifact.
        expected_phases: Exact reset-artifact phase for every row.
        expected_task_contract: Exact runtime task contract.
        expected_validation_policy: Exact live validator policy.
        expected_source_sha256: Exact live source snapshot.
        expected_asset_closure: Exact live external-asset closure.
        expected_full_pick_diversity: Exact independently recomputed artifact
            diversity evidence.

    Returns:
        Named validation checks, all true when validation succeeds.

    Raises:
        TypeError: If a required report container has the wrong type.
        ValueError: If the report is malformed or its evidence is incomplete.
    """
    if not isinstance(report, Mapping):
        raise TypeError("Reset validation report must be a mapping.")
    _validate_sha256(expected_content_sha256, "expected_content_sha256")
    if isinstance(expected_row_count, bool) or not isinstance(expected_row_count, int) or expected_row_count < 1:
        raise ValueError("expected_row_count must be a positive integer.")
    if not isinstance(expected_task_contract, Mapping):
        raise TypeError("expected_task_contract must be a mapping.")
    for name, value in (
        ("expected_validation_policy", expected_validation_policy),
        ("expected_source_sha256", expected_source_sha256),
        ("expected_asset_closure", expected_asset_closure),
        ("expected_full_pick_diversity", expected_full_pick_diversity),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping.")
    expected_phase_by_row = _normalize_expected_phases(expected_phases, expected_row_count=expected_row_count)
    expected_counts = {phase: expected_phase_by_row.count(phase) for phase in PICK_INSERT_RESET_PHASE_IDS}

    _validate_report_value(report, path="report")
    _validate_exact_names(report, _VALIDATION_REPORT_NAMES, path="report")
    if report.get("format") != FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_FORMAT:
        raise ValueError(f"Expected reset validation format {FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_FORMAT!r}.")
    if report.get("schema_version") != FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_SCHEMA_VERSION:
        raise ValueError(
            f"Expected reset validation schema version {FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_SCHEMA_VERSION}."
        )

    report_digest = report.get("content_sha256")
    _validate_sha256(report_digest, "content_sha256")
    report_digest_matches = report_digest == reset_validation_report_content_digest(report)

    report_content_sha256 = report.get("artifact_content_sha256")
    _validate_sha256(report_content_sha256, "artifact_content_sha256")
    report_contract = _require_mapping(report, "task_contract", path="report")
    validation_cfg = _require_mapping(report, "validation_cfg", path="report")
    validation_policy = _require_mapping(report, "validation_policy", path="report")
    physical_contract = _require_mapping(report, "physical_contract", path="report")
    physics_versions = _require_mapping(report, "physics_versions", path="report")
    source_sha256 = _require_mapping(report, "source_sha256", path="report")
    asset_closure = _require_mapping(report, "asset_closure", path="report")
    _validate_source_sha256_snapshot(source_sha256)
    _validate_plain_value(expected_task_contract, path="expected_task_contract")
    task_contract_matches = reset_dataset_digest(_json_normalize(report_contract)) == reset_dataset_digest(
        _json_normalize(expected_task_contract)
    )
    policy_matches_schema = reset_dataset_digest(_json_normalize(validation_policy)) == reset_dataset_digest(
        _json_normalize(FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY)
    )
    policy_matches_runtime = reset_dataset_digest(_json_normalize(validation_policy)) == reset_dataset_digest(
        _json_normalize(expected_validation_policy)
    )
    expected_ik = FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY["ik"]
    validation_cfg_matches_policy = _validate_validation_cfg(validation_cfg) and (
        validation_cfg.get("ik_sampler") == expected_ik["sampler"]
        and validation_cfg.get("ik_seed_count") == expected_ik["seed_count"]
        and validation_cfg.get("ik_iterations") == expected_ik["iterations"]
        and validation_cfg.get("ik_noise_std") == expected_ik["noise_std"]
    )
    ik_solve_call_count = report.get("ik_solve_call_count")
    ik_solve_call_count_valid = _is_plain_int(ik_solve_call_count) and ik_solve_call_count >= 0
    source_matches_runtime = reset_dataset_digest(_json_normalize(source_sha256)) == reset_dataset_digest(
        _json_normalize(expected_source_sha256)
    )
    task_asset_closure = report_contract.get("external_assets")
    asset_matches_task = isinstance(task_asset_closure, Mapping) and reset_dataset_digest(
        _json_normalize(asset_closure)
    ) == reset_dataset_digest(_json_normalize(task_asset_closure))
    asset_matches_runtime = reset_dataset_digest(_json_normalize(asset_closure)) == reset_dataset_digest(
        _json_normalize(expected_asset_closure)
    )
    physical_contract_matches_task = _validate_physical_contract(physical_contract, report_contract)
    physics_versions_match_task = _validate_physics_versions(physics_versions, report_contract)
    task_body_count = _resolve_task_body_count(report_contract, expected_task_contract)
    _validate_vbd_pose_history_contract(report_contract, task_body_count=task_body_count)

    selected_row_ids = report.get("selected_row_ids")
    failed_row_ids = report.get("failed_row_ids")
    rows = report.get("rows")
    goal_replay = report.get("goal_replay")
    full_pick_diversity = report.get("full_pick_diversity")
    if not isinstance(selected_row_ids, list):
        raise TypeError("report.selected_row_ids must be a list.")
    if not isinstance(failed_row_ids, list):
        raise TypeError("report.failed_row_ids must be a list.")
    if not isinstance(rows, list):
        raise TypeError("report.rows must be a list.")
    if not isinstance(goal_replay, Mapping):
        raise TypeError("report.goal_replay must be a mapping.")
    if not isinstance(full_pick_diversity, Mapping):
        raise TypeError("report.full_pick_diversity must be a mapping.")
    _validate_exact_names(goal_replay, _GOAL_REPLAY_NAMES, path="report.goal_replay")
    _validate_exact_names(full_pick_diversity, _FULL_PICK_DIVERSITY_NAMES, path="report.full_pick_diversity")
    offline_diversity = _require_mapping(full_pick_diversity, "offline_artifact", path="report.full_pick_diversity")
    _validate_exact_names(
        offline_diversity,
        _OFFLINE_DIVERSITY_NAMES,
        path="report.full_pick_diversity.offline_artifact",
    )
    report_counts = _normalize_phase_counts(report.get("phase_counts"), path="report.phase_counts", positive=False)

    created_utc = report.get("created_utc")
    try:
        parsed_created_utc = datetime.fromisoformat(created_utc) if isinstance(created_utc, str) else None
    except ValueError:
        parsed_created_utc = None
    created_utc_valid = parsed_created_utc is not None and parsed_created_utc.utcoffset() is not None
    dataset_row_count_valid = (
        _is_plain_int(report.get("dataset_row_count")) and report.get("dataset_row_count") == expected_row_count
    )
    selected_row_count_valid = (
        _is_plain_int(report.get("selected_row_count")) and report.get("selected_row_count") == expected_row_count
    )

    expected_row_ids = list(range(expected_row_count))
    selected_ids_valid = all(_is_plain_int(row_id) for row_id in selected_row_ids)
    selected_every_row_once = (
        selected_ids_valid
        and len(selected_row_ids) == expected_row_count
        and sorted(selected_row_ids) == expected_row_ids
    )

    row_ids: list[int] = []
    observed_counts = {phase: 0 for phase in PICK_INSERT_RESET_PHASE_IDS}
    every_row_passed = len(rows) == expected_row_count
    every_row_phase_matches = len(rows) == expected_row_count
    every_row_checks = len(rows) == expected_row_count
    every_row_oracle = len(rows) == expected_row_count
    every_row_reset_replay = len(rows) == expected_row_count
    every_row_metrics = len(rows) == expected_row_count
    for row in rows:
        if not isinstance(row, Mapping):
            every_row_passed = False
            every_row_phase_matches = False
            every_row_checks = False
            every_row_oracle = False
            every_row_reset_replay = False
            every_row_metrics = False
            continue
        _validate_exact_names(row, _ROW_NAMES, path="report.rows[]")
        row_id = row.get("row_id")
        phase = row.get("phase")
        if _is_plain_int(row_id):
            row_ids.append(row_id)
        else:
            every_row_passed = False
            every_row_phase_matches = False
        if _is_plain_int(phase) and phase in PICK_INSERT_RESET_PHASE_IDS:
            observed_counts[phase] += 1
        else:
            every_row_passed = False
            every_row_phase_matches = False
        if (
            not _is_plain_int(row_id)
            or not 0 <= row_id < expected_row_count
            or not _is_plain_int(phase)
            or phase != expected_phase_by_row[row_id]
        ):
            every_row_phase_matches = False
        if row.get("passed") is not True:
            every_row_passed = False

        row_checks = row.get("checks")
        if isinstance(row_checks, Mapping):
            _validate_exact_names(row_checks, set(_ROW_CHECK_NAMES), path="report.rows[].checks")
        if not isinstance(row_checks, Mapping) or not _required_true_fields(row_checks, _ROW_CHECK_NAMES):
            every_row_checks = False
        oracle = row.get("oracle")
        if isinstance(oracle, Mapping):
            _validate_exact_names(oracle, set(_ORACLE_EVIDENCE_NAMES), path="report.rows[].oracle")
        if not isinstance(oracle, Mapping) or not _required_true_fields(oracle, _ORACLE_EVIDENCE_NAMES):
            every_row_oracle = False
        reset_replay = row.get("reset_replay")
        if isinstance(reset_replay, Mapping):
            _validate_exact_names(
                reset_replay,
                _ROW_RESET_REPLAY_NAMES,
                path="report.rows[].reset_replay",
            )
        if (
            not _is_plain_int(phase)
            or phase not in PICK_INSERT_RESET_PHASE_IDS
            or not _validate_row_reset_replay_evidence(
                reset_replay,
                phase=phase,
                task_body_count=task_body_count,
            )
        ):
            every_row_reset_replay = False
        if (
            not _is_plain_int(phase)
            or phase not in PICK_INSERT_RESET_PHASE_IDS
            or not isinstance(reset_replay, Mapping)
            or not _validate_row_metrics(row.get("metrics"), phase=phase, reset_replay=reset_replay)
        ):
            every_row_metrics = False

    row_ids_exactly_once = len(row_ids) == expected_row_count and sorted(row_ids) == expected_row_ids
    goal_simulation_time_s = goal_replay.get("simulation_time_s")
    goal_duration_sufficient = (
        isinstance(goal_simulation_time_s, (int, float))
        and not isinstance(goal_simulation_time_s, bool)
        and math.isfinite(goal_simulation_time_s)
        and goal_simulation_time_s >= 60.0
        and goal_simulation_time_s >= validation_cfg["goal_replay_s"]
    )
    goal_required_dwell_steps = goal_replay.get("required_dwell_steps")
    goal_final_consecutive_steps = goal_replay.get("final_consecutive_steps")
    goal_raw_counts_valid = (
        _is_plain_int(goal_replay.get("simulation_steps"))
        and goal_replay["simulation_steps"] > 0
        and _plain_nonnegative_int(goal_replay.get("contact_count_after_history_reset"))
        and goal_replay["contact_count_after_history_reset"] == 0
        and _plain_nonnegative_int(goal_replay.get("minimum_left_proxy_contact_count"))
        and goal_replay["minimum_left_proxy_contact_count"] >= 1
        and _plain_nonnegative_int(goal_replay.get("minimum_right_proxy_contact_count"))
        and goal_replay["minimum_right_proxy_contact_count"] >= 1
        and _plain_nonnegative_int(goal_replay.get("maximum_invalid_contact_count"))
        and goal_replay["maximum_invalid_contact_count"] == 0
        and goal_replay.get("any_contact_overflow") is False
        and goal_replay.get("no_contact_overflow") is True
        and goal_replay.get("sampled_invalid_contact_pairs") == []
        and _is_plain_int(goal_required_dwell_steps)
        and goal_required_dwell_steps > 0
        and isinstance(goal_final_consecutive_steps, list)
        and bool(goal_final_consecutive_steps)
        and all(_is_plain_int(value) and value >= goal_required_dwell_steps for value in goal_final_consecutive_steps)
        and goal_replay.get("collision_valid") is True
        and goal_replay.get("closed_bilateral_grasp") is True
    )
    goal_arm_speed = goal_replay.get("sampled_maximum_arm_joint_speed_rad_s")
    goal_finger_speed = goal_replay.get("sampled_maximum_finger_joint_speed_m_s")
    goal_clamp_delta = goal_replay.get("maximum_arm_target_clamp_delta_rad")
    goal_reported_limits_valid = (
        goal_replay.get("maximum_allowed_socket_drift_m") == PICK_INSERT_GOAL_MAX_SOCKET_DRIFT_M
        and goal_replay.get("maximum_allowed_task_body_drift_m") == PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M
        and goal_replay.get("maximum_allowed_cable_speed_m_s") == PICK_INSERT_GOAL_MAX_CABLE_SPEED_M_S
        and goal_replay.get("maximum_allowed_arm_joint_speed_rad_s") == PICK_INSERT_GOAL_MAX_ARM_JOINT_SPEED_RAD_S
        and goal_replay.get("maximum_allowed_finger_joint_speed_m_s") == PICK_INSERT_GOAL_MAX_FINGER_JOINT_SPEED_M_S
    )
    goal_stability_metric_limits = {
        "maximum_socket_drift_m": PICK_INSERT_GOAL_MAX_SOCKET_DRIFT_M,
        "maximum_task_body_drift_m": PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M,
        "sampled_maximum_task_body_excursion_m": PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M,
        "maximum_start_cable_speed_m_s": PICK_INSERT_GOAL_MAX_CABLE_SPEED_M_S,
        "maximum_final_cable_speed_m_s": PICK_INSERT_GOAL_MAX_CABLE_SPEED_M_S,
        "sampled_maximum_cable_speed_m_s": PICK_INSERT_GOAL_MAX_CABLE_SPEED_M_S,
    }
    goal_stability_metrics_valid = all(
        _finite_nonnegative_number(goal_replay.get(name), maximum=limit)
        for name, limit in goal_stability_metric_limits.items()
    )
    goal_robot_speed_valid = (
        isinstance(goal_arm_speed, int | float)
        and not isinstance(goal_arm_speed, bool)
        and math.isfinite(goal_arm_speed)
        and 0.0 <= float(goal_arm_speed) <= PICK_INSERT_GOAL_MAX_ARM_JOINT_SPEED_RAD_S
        and isinstance(goal_finger_speed, int | float)
        and not isinstance(goal_finger_speed, bool)
        and math.isfinite(goal_finger_speed)
        and 0.0 <= float(goal_finger_speed) <= PICK_INSERT_GOAL_MAX_FINGER_JOINT_SPEED_M_S
    )
    goal_zero_action_unclamped = (
        goal_replay.get("zero_action_unclamped") is True
        and isinstance(goal_clamp_delta, int | float)
        and not isinstance(goal_clamp_delta, bool)
        and math.isfinite(goal_clamp_delta)
        and 0.0 <= float(goal_clamp_delta) <= 1.0e-7
    )
    goal_target_drift = goal_replay.get("maximum_arm_target_drift_rad")
    goal_tracking_errors = goal_replay.get("maximum_arm_target_tracking_error_by_joint_rad")
    goal_absolute_target_valid = (
        goal_replay.get("controller_semantics") == "persistent-absolute"
        and goal_replay.get("absolute_target_stable") is True
        and goal_replay.get("all_samples_arm_target_tracking_bounded") is True
        and _finite_nonnegative_number(
            goal_target_drift,
            maximum=PICK_INSERT_RESET_MAX_ARM_TARGET_CLAMP_DELTA_RAD,
        )
        and isinstance(goal_tracking_errors, list | tuple)
        and len(goal_tracking_errors) == len(PICK_INSERT_ARM_TARGET_TRACKING_LIMITS)
        and all(
            _finite_nonnegative_number(error, maximum=limit)
            for error, limit in zip(goal_tracking_errors, PICK_INSERT_ARM_TARGET_TRACKING_LIMITS, strict=True)
        )
    )
    goal_vbd_pose_history_valid = (
        goal_replay.get("vbd_pose_history_restore_queued") is True
        and goal_replay.get("vbd_pose_history_pending_at_queue") is True
        and goal_replay.get("vbd_previous_pose_queued") is True
        and goal_replay.get("vbd_coupling_previous_pose_queued") is True
        and goal_replay.get("vbd_pose_history_applied_exactly_once") is True
        and goal_replay.get("vbd_pose_history_failed") is False
        and goal_replay.get("vbd_pose_history_superseded") is False
        and goal_replay.get("vbd_pose_history_pending_after_first_solve") is False
        and _is_plain_int(goal_replay.get("vbd_pose_history_minimum_application_count_delta"))
        and goal_replay.get("vbd_pose_history_minimum_application_count_delta") == 1
        and _is_plain_int(goal_replay.get("vbd_pose_history_maximum_application_count_delta"))
        and goal_replay.get("vbd_pose_history_maximum_application_count_delta") == 1
        and _is_plain_int(goal_replay.get("vbd_pose_history_generation"))
        and goal_replay.get("vbd_pose_history_generation") > 0
        and _is_plain_int(goal_replay.get("vbd_pose_history_expected_body_count"))
        and goal_replay.get("vbd_pose_history_expected_body_count") == task_body_count
        and _is_plain_int(goal_replay.get("vbd_pose_history_minimum_body_application_count_delta"))
        and goal_replay.get("vbd_pose_history_minimum_body_application_count_delta") == task_body_count
        and _is_plain_int(goal_replay.get("vbd_pose_history_maximum_body_application_count_delta"))
        and goal_replay.get("vbd_pose_history_maximum_body_application_count_delta") == task_body_count
        and goal_replay.get("vbd_pose_history_body_order_exact") is True
        and goal_replay.get("vbd_pose_history_world_order_exact") is True
        and goal_replay.get("vbd_pose_history_entry_name") == PICK_INSERT_VBD_POSE_HISTORY_ENTRY_NAME
        and _is_plain_int(goal_replay.get("vbd_pose_history_body_count"))
        and goal_replay.get("vbd_pose_history_body_count") == task_body_count
    )
    goal_authored_metric_limits = {
        "maximum_stored_authored_seat_error_m": PICK_INSERT_GOAL_MAX_AUTHORED_SEAT_ERROR_M,
        "maximum_final_authored_seat_error_m": PICK_INSERT_GOAL_MAX_AUTHORED_SEAT_ERROR_M,
        "maximum_stored_authored_plug_angle_rad": PICK_INSERT_GOAL_MAX_AUTHORED_PLUG_ANGLE_RAD,
        "maximum_final_authored_plug_angle_rad": PICK_INSERT_GOAL_MAX_AUTHORED_PLUG_ANGLE_RAD,
        "maximum_stored_plug_relative_latch_angle_rad": PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD,
        "maximum_final_plug_relative_latch_angle_rad": PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD,
    }
    goal_authored_geometry_valid = (
        goal_replay.get("authored_goal_geometry_valid") is True
        and goal_replay.get("maximum_allowed_authored_seat_error_m") == PICK_INSERT_GOAL_MAX_AUTHORED_SEAT_ERROR_M
        and goal_replay.get("maximum_allowed_authored_plug_angle_rad") == PICK_INSERT_GOAL_MAX_AUTHORED_PLUG_ANGLE_RAD
        and goal_replay.get("maximum_allowed_plug_relative_latch_angle_rad")
        == PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD
        and all(
            isinstance(goal_replay.get(name), int | float)
            and not isinstance(goal_replay.get(name), bool)
            and math.isfinite(goal_replay[name])
            and 0.0 <= float(goal_replay[name]) <= limit
            for name, limit in goal_authored_metric_limits.items()
        )
    )
    tcp_span = full_pick_diversity.get("tcp_xyz_span_m")
    tcp_minimum = full_pick_diversity.get("tcp_xyz_min_m")
    tcp_maximum = full_pick_diversity.get("tcp_xyz_max_m")
    tcp_span_valid = (
        isinstance(tcp_span, tuple | list)
        and len(tcp_span) == 3
        and all(
            isinstance(observed, int | float) and not isinstance(observed, bool) and float(observed) >= required
            for observed, required in zip(tcp_span, _FULL_PICK_MINIMUM_TCP_SPAN_M, strict=True)
        )
    )
    tcp_bounds_valid = (
        _validate_exact_finite_vector(tcp_minimum, length=3)
        and _validate_exact_finite_vector(tcp_maximum, length=3)
        and tcp_span_valid
        and all(
            math.isclose(float(upper) - float(lower), float(span), rel_tol=0.0, abs_tol=1.0e-12)
            for lower, upper, span in zip(tcp_minimum, tcp_maximum, tcp_span, strict=True)
        )
    )
    unique_tcp_positions = full_pick_diversity.get("unique_tcp_positions")
    minimum_tcp_distance = full_pick_diversity.get("observed_minimum_tcp_to_grasp_distance_m")
    full_pick_diversity_valid = (
        full_pick_diversity.get("passed") is True
        and full_pick_diversity.get("skipped_due_to_quick") is False
        and full_pick_diversity.get("required_phase") == _FULL_PICK_PHASE
        and full_pick_diversity.get("required_rows") == expected_counts[_FULL_PICK_PHASE]
        and full_pick_diversity.get("observed_rows") == expected_counts[_FULL_PICK_PHASE]
        and full_pick_diversity.get("round_decimals") == _FULL_PICK_TCP_ROUND_DECIMALS
        and full_pick_diversity.get("minimum_unique_tcp_positions") == _FULL_PICK_MINIMUM_UNIQUE_TCP_POSITIONS
        and full_pick_diversity.get("minimum_tcp_xyz_span_m") == list(_FULL_PICK_MINIMUM_TCP_SPAN_M)
        and full_pick_diversity.get("minimum_tcp_to_grasp_distance_m") == _FULL_PICK_MINIMUM_TCP_TO_GRASP_DISTANCE_M
        and _is_plain_int(unique_tcp_positions)
        and _FULL_PICK_MINIMUM_UNIQUE_TCP_POSITIONS <= unique_tcp_positions <= expected_counts[_FULL_PICK_PHASE]
        and tcp_bounds_valid
        and isinstance(minimum_tcp_distance, int | float)
        and not isinstance(minimum_tcp_distance, bool)
        and float(minimum_tcp_distance) >= _FULL_PICK_MINIMUM_TCP_TO_GRASP_DISTANCE_M
        and full_pick_diversity.get("failures") == []
        and reset_dataset_digest(_json_normalize(offline_diversity))
        == reset_dataset_digest(_json_normalize(expected_full_pick_diversity))
        and _validate_offline_diversity_evidence(
            offline_diversity,
            expected_row_count=expected_counts[_FULL_PICK_PHASE],
        )
        and _full_pick_live_evidence_matches_rows(
            full_pick_diversity,
            rows,
            required_rows=expected_counts[_FULL_PICK_PHASE],
        )
    )
    checks = {
        "report_content_digest_matches": report_digest_matches,
        "created_utc_valid": created_utc_valid,
        "dataset_row_count_matches": dataset_row_count_valid,
        "selected_row_count_matches": selected_row_count_valid,
        "validation_policy_matches_schema": policy_matches_schema,
        "validation_policy_matches_runtime": policy_matches_runtime,
        "validation_cfg_matches_policy": validation_cfg_matches_policy,
        "physical_contract_matches_task": physical_contract_matches_task,
        "physics_versions_match_task": physics_versions_match_task,
        "ik_solve_call_count_is_evidence": ik_solve_call_count_valid,
        "source_snapshot_matches_runtime": source_matches_runtime,
        "asset_closure_matches_task": asset_matches_task,
        "asset_closure_matches_runtime": asset_matches_runtime,
        "passed": report.get("passed") is True,
        "evidence_complete": report.get("evidence_complete") is True,
        "full_dataset_replay": report.get("full_dataset_replay") is True,
        "not_quick": report.get("quick") is False,
        "content_digest_matches": report_content_sha256 == expected_content_sha256,
        "task_contract_matches": task_contract_matches,
        "selected_every_row_once": selected_every_row_once,
        "no_failed_rows": failed_row_ids == [],
        "row_ids_exactly_once": row_ids_exactly_once,
        "reported_phase_counts_match": report_counts == expected_counts,
        "row_phase_counts_match": observed_counts == expected_counts,
        "row_phases_match_artifact": every_row_phase_matches,
        "goal_replay_passed": goal_replay.get("passed") is True,
        "goal_drive_disabled": goal_replay.get("drive_disabled") is True,
        "goal_socket_stable": goal_replay.get("socket_stable") is True,
        "goal_whole_cable_stable": goal_replay.get("whole_cable_stable") is True,
        "goal_exact_runtime_success_dwell": goal_replay.get("exact_runtime_success_dwell") is True,
        "goal_all_samples_collision_free": goal_replay.get("all_samples_collision_free") is True,
        "goal_all_samples_bilateral_grasp": goal_replay.get("all_samples_bilateral_grasp") is True,
        "goal_all_samples_proxy_bilateral_contact": goal_replay.get("all_samples_proxy_bilateral_contact") is True,
        "goal_all_samples_finite": goal_replay.get("all_samples_finite") is True,
        "goal_no_contact_overflow": goal_replay.get("no_contact_overflow") is True,
        "goal_stored_capture_exact_success": goal_replay.get("stored_capture_exact_success") is True,
        "goal_all_post_step_exact_success": goal_replay.get("all_post_step_exact_success") is True,
        "goal_reported_limits": goal_reported_limits_valid,
        "goal_stability_metrics": goal_stability_metrics_valid,
        "goal_robot_equilibrium": goal_replay.get("robot_equilibrium") is True and goal_robot_speed_valid,
        "goal_zero_action_unclamped": goal_zero_action_unclamped,
        "goal_persistent_absolute_target": goal_absolute_target_valid,
        "goal_vbd_pose_history": goal_vbd_pose_history_valid,
        "goal_authored_geometry": goal_authored_geometry_valid,
        "goal_duration_sufficient": goal_duration_sufficient,
        "goal_raw_counts_consistent": goal_raw_counts_valid,
        "full_pick_diversity": full_pick_diversity_valid,
        "every_row_passed": every_row_passed,
        "every_row_checks": every_row_checks,
        "every_row_oracle": every_row_oracle,
        "every_row_reset_replay": every_row_reset_replay,
        "every_row_metrics": every_row_metrics,
    }
    if not all(checks.values()):
        raise ValueError(f"Reset validation evidence is incomplete or incompatible: {checks}")
    return checks


def reset_validation_report_content_digest(report: Mapping[str, Any]) -> str:
    """Return the canonical digest of a validation certificate excluding itself."""
    if not isinstance(report, Mapping):
        raise TypeError("Reset validation report must be a mapping.")
    unsigned = dict(report)
    unsigned.pop("content_sha256", None)
    return reset_dataset_digest(_json_normalize(unsigned))


def _state_tensor_specs(task_body_count: int) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    """Return reset-row tensor specifications for one task-body topology."""
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


def _goal_tensor_specs(task_body_count: int) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    """Return canonical-goal tensor specifications for one task-body topology."""
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
    }


def _require_mapping(container: Mapping[str, Any], key: str, *, path: str = "payload") -> Mapping[str, Any]:
    """Return one required mapping field."""
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}.{key} must be a mapping.")
    return value


def _validate_exact_names(values: Mapping[str, Any], expected: set[str], *, path: str) -> None:
    """Reject omitted and unrecognized schema fields."""
    actual = set(values)
    missing = sorted(expected - actual, key=repr)
    unexpected = sorted(actual - expected, key=repr)
    if missing or unexpected:
        raise ValueError(f"{path} fields do not match the schema: missing={missing}, unexpected={unexpected}.")


def _validate_sha256(value: Any, name: str) -> None:
    """Validate one lowercase hexadecimal SHA-256 string."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")


def _validate_source_sha256_snapshot(value: Mapping[str, Any]) -> None:
    """Require a path-independent digest for the full validator source closure."""
    if not value:
        raise ValueError("report.source_sha256 must not be empty.")
    for relative_name, digest in value.items():
        if (
            not isinstance(relative_name, str)
            or not relative_name
            or relative_name.startswith("/")
            or "\\" in relative_name
            or any(part in ("", ".", "..") for part in relative_name.split("/"))
        ):
            raise ValueError("report.source_sha256 keys must be canonical repository-relative POSIX paths.")
        _validate_sha256(digest, f"report.source_sha256[{relative_name!r}]")
    missing_files = [name for name in _VALIDATION_SOURCE_FILES if name not in value]
    missing_roots = [root for root in _VALIDATION_SOURCE_ROOTS if not any(name.startswith(root) for name in value)]
    if missing_files or missing_roots:
        raise ValueError(
            "report.source_sha256 is not a full validator source closure: "
            f"missing_files={missing_files}, missing_roots={missing_roots}."
        )


def _validate_fast_source_sha256_snapshot(value: Mapping[str, Any]) -> None:
    """Require the legacy closure plus the explicit fast-validator entrypoint."""
    _validate_source_sha256_snapshot(value)
    missing = [name for name in _FAST_VALIDATION_SOURCE_FILES if name not in value]
    if missing:
        raise ValueError(f"report.source_sha256 is missing fast-validator files: {missing}.")


def _validate_plain_value(value: Any, *, path: str) -> None:
    """Require metadata to remain in the restricted-unpickler value domain."""
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite values.")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings.")
            _validate_plain_value(item, path=f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _validate_plain_value(item, path=f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}.")


def _validate_tensor_mapping_safety(values: Mapping[str, Any], *, path: str) -> None:
    """Require artifact tensor mappings to contain only safe CPU tensors."""
    for name, value in values.items():
        if not isinstance(name, str):
            raise TypeError(f"{path} mapping keys must be strings.")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{path}.{name} must be a torch.Tensor.")
        if (
            value.layout != torch.strided
            or value.is_quantized
            or value.device.type != "cpu"
            or value.requires_grad
            or not value.is_contiguous()
        ):
            raise ValueError(f"{path}.{name} must be a contiguous dense, strided, unquantized CPU tensor.")


def _resolve_task_body_count(task_contract: Mapping[str, Any], expected_task_contract: Mapping[str, Any] | None) -> int:
    """Resolve and cross-check the dynamic task-body count."""
    task_body_count = task_contract.get("task_body_count")
    if isinstance(task_body_count, bool) or not isinstance(task_body_count, int) or task_body_count < 1:
        raise ValueError("metadata.task_contract.task_body_count must be a positive integer.")
    if expected_task_contract is not None and "task_body_count" in expected_task_contract:
        expected_count = expected_task_contract["task_body_count"]
        if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 1:
            raise ValueError("expected_task_contract.task_body_count must be a positive integer.")
        if task_body_count != expected_count:
            raise ValueError("Reset dataset task body count does not match the runtime task contract.")
    return task_body_count


def _validate_vbd_pose_history_contract(task_contract: Mapping[str, Any], *, task_body_count: int) -> None:
    """Require an exact, ordered, two-buffer VBD reset representation."""
    body_order = task_contract.get("task_body_order")
    if (
        not isinstance(body_order, tuple | list)
        or len(body_order) != task_body_count
        or any(not isinstance(name, str) or not name for name in body_order)
        or len(set(body_order)) != task_body_count
    ):
        raise ValueError("metadata.task_contract.task_body_order must uniquely name every task body in order.")
    representation = task_contract.get("reset_state_representation")
    if not isinstance(representation, Mapping):
        raise ValueError("metadata.task_contract.reset_state_representation must be a mapping.")
    expected = {
        "contract_version": PICK_INSERT_VBD_POSE_HISTORY_CONTRACT_VERSION,
        "task_body_pose_frame": "environment-local-xyzw",
        "task_body_velocity_frame": "world-linear-angular",
        "vbd_entry_name": PICK_INSERT_VBD_POSE_HISTORY_ENTRY_NAME,
        "vbd_body_order_source": "task_body_order",
        "vbd_previous_pose_field": "task_body_previous_pose",
        "vbd_coupling_previous_pose_field": "task_body_coupling_previous_pose",
        "vbd_pose_history_frame": "environment-local-xyzw",
        "restore_semantics": "deferred-one-shot-after-input-and-proxy-rebaseline-before-first-vbd-solve",
        "preserved_input_task_body_range_half_open": (3, task_body_count - 1),
        "preserved_input_semantics": "scatter-history-without-pose-delta-velocity-injection-or-rewind",
    }
    if _json_normalize(representation) != _json_normalize(expected):
        raise ValueError("metadata.task_contract.reset_state_representation is incompatible with VBD replay.")


def _normalize_expected_phases(
    value: torch.Tensor | list[int] | tuple[int, ...],
    *,
    expected_row_count: int,
) -> list[int]:
    """Return a strict per-row phase list and require all six phases."""
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.int64 or value.ndim != 1:
            raise ValueError("expected_phases tensor must have dtype torch.int64 and shape (expected_row_count,).")
        phases = value.detach().cpu().tolist()
    elif isinstance(value, (list, tuple)):
        phases = list(value)
    else:
        raise TypeError("expected_phases must be a torch.Tensor, list, or tuple.")
    if len(phases) != expected_row_count:
        raise ValueError(f"expected_phases must contain exactly {expected_row_count} rows.")
    if any(not _is_plain_int(phase) or phase not in PICK_INSERT_RESET_PHASE_IDS for phase in phases):
        raise ValueError("expected_phases must contain only phase identifiers in [0, 5].")
    if set(phases) != set(PICK_INSERT_RESET_PHASE_IDS):
        raise ValueError("expected_phases must represent every phase in [0, 5].")
    return phases


def _validate_tensors(
    values: Mapping[str, Any],
    specs: Mapping[str, tuple[torch.dtype, tuple[int, ...]]],
    *,
    leading_count: int | None,
    path: str,
) -> int | None:
    """Validate required tensor names, dtypes, shapes, and finite values."""
    _validate_exact_names(values, set(specs), path=path)
    for name, (dtype, trailing_shape) in specs.items():
        tensor = values[name]
        if leading_count is None:
            leading_count = int(tensor.shape[0]) if tensor.ndim > 0 else 0
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


def _validate_state_semantics(states: Mapping[str, torch.Tensor]) -> None:
    """Validate phase coverage and bounded reset-sampling scalars."""
    phase = states["phase"]
    expected_phases = torch.tensor(PICK_INSERT_RESET_PHASE_IDS, dtype=phase.dtype, device=phase.device)
    if not torch.equal(torch.unique(phase, sorted=True), expected_phases):
        raise ValueError("states.phase must represent every reset phase exactly within [0, 5].")
    expected_starts_grasped = phase <= 3
    if not torch.equal(states["starts_grasped"], expected_starts_grasped):
        raise ValueError("states.starts_grasped must be true exactly for reset phases 0 through 3.")
    if not bool(torch.all((states["difficulty"] >= 0.0) & (states["difficulty"] <= 1.0))):
        raise ValueError("states.difficulty must lie in [0, 1].")
    if not bool(torch.all(states["initial_goal_error"] >= 0.0)):
        raise ValueError("states.initial_goal_error must be non-negative.")
    if not bool(torch.all(states["initial_tcp_grasp_distance"] >= 0.0)):
        raise ValueError("states.initial_tcp_grasp_distance must be non-negative.")
    if not bool(torch.all(states["progress_threshold"] > 0.0)):
        raise ValueError("states.progress_threshold must be positive.")


def _validate_persistent_target_semantics(
    task_contract: Mapping[str, Any],
    states: Mapping[str, torch.Tensor],
    goal_state: Mapping[str, torch.Tensor],
) -> None:
    """Require pick artifacts to encode bounded persistent absolute arm targets."""
    robot = task_contract.get("robot")
    control = robot.get("reset_control_convention") if isinstance(robot, Mapping) else None
    if not isinstance(control, Mapping):
        raise ValueError("task_contract.robot.reset_control_convention must define pick controller semantics.")
    if (
        control.get("contract_version") != 1
        or control.get("target_semantics") != "persistent-absolute-integrated-once-per-policy-step"
        or control.get("zero_action_semantics") != "clear-ema-tail-and-hold-absolute-target-bitwise"
        or control.get("native_gravity_compensation") != "mjwarp-joint-actuatorgravcomp"
        or control.get("native_gravity_compensation_scope") != "pick-insert-franka-only"
        or control.get("action_inverse_dynamics_gravity_compensation") is not False
        or control.get("global_inverse_dynamics_gravity_compensation") is not False
        or tuple(control.get("target_tracking_error_limits_rad", ())) != PICK_INSERT_ARM_TARGET_TRACKING_LIMITS
    ):
        raise ValueError("task_contract does not match the persistent-target/native-gravity pick controller.")
    limits = torch.tensor(PICK_INSERT_ARM_TARGET_TRACKING_LIMITS, dtype=torch.float32)
    row_error = torch.abs(states["arm_joint_target"].cpu() - states["arm_joint_position"].cpu())
    goal_error = torch.abs(goal_state["arm_joint_target"].cpu() - goal_state["arm_joint_position"].cpu())
    if not bool((row_error <= limits).all()):
        raise ValueError("states arm targets exceed the pick controller target-tracking envelope.")
    if not bool((goal_error <= limits).all()):
        raise ValueError("goal_state arm target exceeds the pick controller target-tracking envelope.")


def _validate_pose_quaternions(pose: torch.Tensor, *, path: str) -> None:
    """Require normalized XYZW task-body quaternions."""
    norm = torch.linalg.vector_norm(pose[..., 3:7], dim=-1)
    if not bool(torch.all(torch.abs(norm - 1.0) <= 1.0e-3)):
        raise ValueError(f"{path} quaternions must be normalized.")


def _validate_report_value(value: Any, *, path: str) -> None:
    """Require validation evidence to remain in a finite JSON-safe domain."""
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite values.")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) and not _is_plain_int(key):
                raise TypeError(f"{path} mapping keys must be strings or integers.")
            _validate_report_value(item, path=f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _validate_report_value(item, path=f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}.")


def _normalize_phase_counts(
    value: Any,
    *,
    path: str,
    positive: bool,
) -> dict[int, int]:
    """Normalize a six-phase count sequence or mapping."""
    if isinstance(value, Mapping):
        counts: dict[int, int] = {}
        for key, count in value.items():
            if _is_plain_int(key):
                phase = key
            elif isinstance(key, str) and key in {str(phase) for phase in PICK_INSERT_RESET_PHASE_IDS}:
                phase = int(key)
            else:
                raise ValueError(f"{path} keys must be phase identifiers in [0, 5].")
            if phase in counts:
                raise ValueError(f"{path} contains duplicate phase {phase}.")
            counts[phase] = count
    elif isinstance(value, (tuple, list)):
        if len(value) != len(PICK_INSERT_RESET_PHASE_IDS):
            raise ValueError(f"{path} must contain exactly six phase counts.")
        counts = dict(enumerate(value))
    else:
        raise TypeError(f"{path} must be a mapping or six-element sequence.")

    if set(counts) != set(PICK_INSERT_RESET_PHASE_IDS):
        raise ValueError(f"{path} must represent every phase in [0, 5].")
    minimum = 1 if positive else 0
    if any(not _is_plain_int(count) or count < minimum for count in counts.values()):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{path} values must be {qualifier} integers.")
    return counts


def _finite_contract_scalar(
    values: Mapping[str, Any],
    name: str,
    *,
    positive: bool,
    maximum: float | None = None,
) -> float:
    """Read one finite numeric contract field."""
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"task_contract field {name} must be a finite number.")
    result = float(value)
    if positive and result <= 0.0:
        raise ValueError(f"task_contract field {name} must be positive.")
    if maximum is not None and result > maximum:
        raise ValueError(f"task_contract field {name} must be no greater than {maximum}.")
    return result


def _finite_contract_vector(values: Mapping[str, Any], name: str, *, length: int) -> list[float]:
    """Read one fixed-length finite numeric contract vector."""
    value = values.get(name)
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError(f"task_contract field {name} must contain exactly {length} values.")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"task_contract field {name} must contain only numbers.")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"task_contract field {name} must contain only finite values.")
    return result


def _column_span(values: torch.Tensor) -> list[float]:
    """Return max-minus-min for every column, or zeros for an empty selection."""
    if values.ndim != 2:
        raise ValueError("Diversity values must be a two-dimensional tensor.")
    if values.shape[0] == 0:
        return [0.0] * values.shape[1]
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Diversity values must contain only finite values.")
    span = values.amax(dim=0) - values.amin(dim=0)
    return [float(value) for value in span.detach().cpu().tolist()]


def _rounded_unique_row_count(values: torch.Tensor, *, decimals: int) -> int:
    """Count unique rows after deterministic fixed-decimal quantization."""
    if values.ndim != 2:
        raise ValueError("Rounded diversity values must be a two-dimensional tensor.")
    if values.shape[0] == 0:
        return 0
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Rounded diversity values must contain only finite values.")
    scale = float(10**decimals)
    quantized = torch.round(values * scale).to(dtype=torch.int64)
    return int(torch.unique(quantized, dim=0).shape[0])


def _rounded_unique_pose_count(pose: torch.Tensor, *, decimals: int) -> int:
    """Count unique poses after canonicalizing quaternion sign and rounding."""
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError("Rounded pose diversity values must have shape (row_count, 7).")
    if pose.shape[0] == 0:
        return 0
    canonical = pose.clone()
    quaternion = canonical[:, 3:7]
    pivot_index = quaternion.abs().argmax(dim=-1, keepdim=True)
    pivot = torch.gather(quaternion, 1, pivot_index)
    sign = torch.where(pivot < 0.0, -torch.ones_like(pivot), torch.ones_like(pivot))
    canonical[:, 3:7] = quaternion * sign
    return _rounded_unique_row_count(canonical, decimals=decimals)


def _quaternion_yaw_span(quaternion_xyzw: torch.Tensor) -> float:
    """Return the shortest circular arc containing all quaternion yaw angles."""
    if quaternion_xyzw.ndim != 2 or quaternion_xyzw.shape[1] != 4:
        raise ValueError("Yaw diversity quaternions must have shape (row_count, 4).")
    if quaternion_xyzw.shape[0] <= 1:
        return 0.0
    if not bool(torch.isfinite(quaternion_xyzw).all()):
        raise ValueError("Yaw diversity quaternions must contain only finite values.")
    x, y, z, w = quaternion_xyzw.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    angles = torch.sort(torch.remainder(yaw, 2.0 * math.pi)).values
    gaps = torch.cat((angles[1:] - angles[:-1], angles[:1] + 2.0 * math.pi - angles[-1:]))
    return float((2.0 * math.pi - gaps.max()).detach().cpu())


def _required_true_fields(values: Mapping[str, Any], names: tuple[str, ...]) -> bool:
    """Return whether every named evidence flag is exactly true."""
    return all(values.get(name) is True for name in names)


def _finite_nonnegative_number(value: Any, *, maximum: float | None = None) -> bool:
    """Return whether a report scalar is a finite non-Boolean number in bounds."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0.0:
        return False
    return maximum is None or float(value) <= maximum


def _plain_nonnegative_int(value: Any) -> bool:
    """Return whether a report count is a non-negative plain integer."""
    return _is_plain_int(value) and value >= 0


def _validate_row_reset_replay_evidence(value: Any, *, phase: int, task_body_count: int) -> bool:
    """Validate raw per-row mechanical evidence independently of producer flags."""
    if not isinstance(value, Mapping):
        return False
    _validate_exact_names(value, _ROW_RESET_REPLAY_NAMES, path="report.rows[].reset_replay")
    starts_grasped = phase <= 3
    required_true = (
        "stored_state_finite",
        "stored_task_state_finite_and_normalized",
        "stored_drive_disabled",
        "all_post_step_state_finite",
        "all_post_step_task_state_finite_and_normalized",
        "all_post_step_collision_free",
        "all_post_step_drive_disabled",
        "all_post_step_expected_contact_state",
        "no_contact_overflow",
        "zero_action_unclamped",
        "absolute_target_stable",
        "stored_arm_target_tracking_bounded",
        "all_post_step_arm_target_tracking_bounded",
        "vbd_pose_history_restore_queued",
        "vbd_pose_history_pending_at_queue",
        "vbd_previous_pose_queued",
        "vbd_coupling_previous_pose_queued",
        "vbd_pose_history_applied_exactly_once",
        "vbd_pose_history_body_order_exact",
        "vbd_pose_history_world_order_exact",
    )
    exact_contract = (
        value.get("simulation_time_s") == PICK_INSERT_RESET_REPLAY_DURATION_S
        and value.get("required_simulation_time_s") == PICK_INSERT_RESET_REPLAY_DURATION_S
        and _is_plain_int(value.get("simulation_steps"))
        and value["simulation_steps"] == PICK_INSERT_RESET_REPLAY_POST_STEP_SAMPLES
        and _is_plain_int(value.get("post_step_samples"))
        and value["post_step_samples"] == PICK_INSERT_RESET_REPLAY_POST_STEP_SAMPLES
        and _is_plain_int(value.get("required_post_step_samples"))
        and value["required_post_step_samples"] == PICK_INSERT_RESET_REPLAY_POST_STEP_SAMPLES
        and value.get("starts_grasped") is starts_grasped
        and value.get("contact_expectation") == ("bilateral-proxy" if starts_grasped else "zero-proxy")
        and value.get("maximum_allowed_arm_joint_speed_rad_s") == PICK_INSERT_RESET_MAX_ARM_JOINT_SPEED_RAD_S
        and value.get("maximum_allowed_finger_joint_speed_m_s") == PICK_INSERT_RESET_MAX_FINGER_JOINT_SPEED_M_S
        and value.get("maximum_allowed_cable_speed_m_s") == PICK_INSERT_RESET_MAX_CABLE_SPEED_M_S
        and value.get("maximum_allowed_body_excursion_m") == PICK_INSERT_RESET_MAX_BODY_EXCURSION_M
        and value.get("maximum_allowed_plug_excursion_m") == PICK_INSERT_RESET_MAX_PLUG_EXCURSION_M
        and value.get("maximum_allowed_socket_excursion_m") == PICK_INSERT_RESET_MAX_SOCKET_EXCURSION_M
        and value.get("maximum_allowed_arm_target_clamp_delta_rad") == PICK_INSERT_RESET_MAX_ARM_TARGET_CLAMP_DELTA_RAD
        and value.get("arm_target_semantics") == "persistent-absolute"
        and value.get("arm_target_tracking_limits_rad") == list(PICK_INSERT_ARM_TARGET_TRACKING_LIMITS)
        and value.get("vbd_pose_history_entry_name") == PICK_INSERT_VBD_POSE_HISTORY_ENTRY_NAME
        and _is_plain_int(value.get("vbd_pose_history_body_count"))
        and value.get("vbd_pose_history_body_count") == task_body_count
        and value.get("vbd_pose_history_failed") is False
        and value.get("vbd_pose_history_superseded") is False
        and value.get("vbd_pose_history_pending_after_first_solve") is False
        and _is_plain_int(value.get("vbd_pose_history_application_count_delta"))
        and value.get("vbd_pose_history_application_count_delta") == 1
        and _is_plain_int(value.get("vbd_pose_history_generation"))
        and value.get("vbd_pose_history_generation") > 0
        and _is_plain_int(value.get("vbd_pose_history_expected_body_count"))
        and value.get("vbd_pose_history_expected_body_count") == task_body_count
        and _is_plain_int(value.get("vbd_pose_history_body_application_count_delta"))
        and value.get("vbd_pose_history_body_application_count_delta") == task_body_count
        and value.get("no_contact_overflow") is True
    )
    if not exact_contract or not _required_true_fields(value, required_true):
        return False
    if not all(
        _plain_nonnegative_int(value.get(name))
        for name in (
            "minimum_left_proxy_contact_count",
            "minimum_right_proxy_contact_count",
            "maximum_left_proxy_contact_count",
            "maximum_right_proxy_contact_count",
            "maximum_invalid_contact_count",
        )
    ):
        return False
    invalid_pairs = value.get("invalid_contact_pairs")
    if (
        not isinstance(invalid_pairs, list | tuple)
        or invalid_pairs
        or not all(isinstance(pair, str) for pair in invalid_pairs)
    ):
        return False
    if value.get("maximum_invalid_contact_count") != 0:
        return False
    left_minimum = value["minimum_left_proxy_contact_count"]
    right_minimum = value["minimum_right_proxy_contact_count"]
    left_maximum = value["maximum_left_proxy_contact_count"]
    right_maximum = value["maximum_right_proxy_contact_count"]
    if left_minimum > left_maximum or right_minimum > right_maximum:
        return False
    speed_limits = {
        "stored_maximum_cable_speed_m_s": PICK_INSERT_RESET_MAX_CABLE_SPEED_M_S,
        "maximum_post_step_cable_speed_m_s": PICK_INSERT_RESET_MAX_CABLE_SPEED_M_S,
        "final_cable_speed_m_s": PICK_INSERT_RESET_MAX_CABLE_SPEED_M_S,
        "stored_maximum_arm_joint_speed_rad_s": PICK_INSERT_RESET_MAX_ARM_JOINT_SPEED_RAD_S,
        "maximum_post_step_arm_joint_speed_rad_s": PICK_INSERT_RESET_MAX_ARM_JOINT_SPEED_RAD_S,
        "final_arm_joint_speed_rad_s": PICK_INSERT_RESET_MAX_ARM_JOINT_SPEED_RAD_S,
        "stored_maximum_finger_joint_speed_m_s": PICK_INSERT_RESET_MAX_FINGER_JOINT_SPEED_M_S,
        "maximum_post_step_finger_joint_speed_m_s": PICK_INSERT_RESET_MAX_FINGER_JOINT_SPEED_M_S,
        "final_finger_joint_speed_m_s": PICK_INSERT_RESET_MAX_FINGER_JOINT_SPEED_M_S,
        "maximum_body_excursion_m": PICK_INSERT_RESET_MAX_BODY_EXCURSION_M,
        "maximum_plug_excursion_m": PICK_INSERT_RESET_MAX_PLUG_EXCURSION_M,
        "maximum_socket_excursion_m": PICK_INSERT_RESET_MAX_SOCKET_EXCURSION_M,
        "maximum_arm_target_clamp_delta_rad": PICK_INSERT_RESET_MAX_ARM_TARGET_CLAMP_DELTA_RAD,
        "maximum_arm_target_drift_rad": PICK_INSERT_RESET_MAX_ARM_TARGET_CLAMP_DELTA_RAD,
    }
    if not all(_finite_nonnegative_number(value.get(name), maximum=limit) for name, limit in speed_limits.items()):
        return False
    tracking_values: dict[str, list[float]] = {}
    for name in (
        "stored_arm_target_tracking_error_by_joint_rad",
        "maximum_arm_target_tracking_error_by_joint_rad",
    ):
        errors = value.get(name)
        if (
            not isinstance(errors, list | tuple)
            or len(errors) != len(PICK_INSERT_ARM_TARGET_TRACKING_LIMITS)
            or any(
                not _finite_nonnegative_number(error, maximum=limit)
                for error, limit in zip(errors, PICK_INSERT_ARM_TARGET_TRACKING_LIMITS, strict=True)
            )
        ):
            return False
        tracking_values[name] = [float(error) for error in errors]
    if not (
        _finite_nonnegative_number(value.get("stored_maximum_arm_target_tracking_error_rad"))
        and value["stored_maximum_arm_target_tracking_error_rad"]
        == max(tracking_values["stored_arm_target_tracking_error_by_joint_rad"])
        and _finite_nonnegative_number(value.get("maximum_post_step_arm_target_tracking_error_rad"))
        and value["maximum_post_step_arm_target_tracking_error_rad"]
        == max(tracking_values["maximum_arm_target_tracking_error_by_joint_rad"])
    ):
        return False
    if starts_grasped:
        return (
            value.get("all_post_step_bilateral_grasp") is True
            and value.get("all_post_step_proxy_bilateral_contact") is True
            and value.get("all_post_step_zero_proxy_contacts") is False
            and left_minimum >= 1
            and right_minimum >= 1
        )
    return (
        value.get("all_post_step_bilateral_grasp") is False
        and value.get("all_post_step_proxy_bilateral_contact") is False
        and value.get("all_post_step_zero_proxy_contacts") is True
        and left_minimum == 0
        and right_minimum == 0
        and left_maximum == 0
        and right_maximum == 0
    )


def _is_plain_int(value: Any) -> bool:
    """Return whether a value is an integer but not a Boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def _json_normalize(value: Any) -> Any:
    """Normalize tuple/list distinctions lost by JSON report serialization."""
    if isinstance(value, Mapping):
        return {str(key): _json_normalize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_normalize(item) for item in value]
    return value


__all__ = [
    "FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY",
    "FRANKA_RJ45_PICK_INSERT_FAST_RESET_VALIDATION_FORMAT",
    "FRANKA_RJ45_PICK_INSERT_FAST_RESET_VALIDATION_SCHEMA_VERSION",
    "FRANKA_RJ45_PICK_INSERT_RESET_DATASET_FORMAT",
    "FRANKA_RJ45_PICK_INSERT_RESET_DATASET_SCHEMA_VERSION",
    "FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_FORMAT",
    "FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY",
    "FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_SCHEMA_VERSION",
    "PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY",
    "PICK_INSERT_FAST_RESET_PHASE_0_BAND_ACCEPTANCE_CONTRACT",
    "PICK_INSERT_FAST_RESET_ROW_BINDING_CONTRACT",
    "PICK_INSERT_RESET_PHASE_IDS",
    "PICK_INSERT_RESET_REPLAY_DURATION_S",
    "PICK_INSERT_RESET_REPLAY_POST_STEP_SAMPLES",
    "PICK_INSERT_RESET_MAX_ARM_JOINT_SPEED_RAD_S",
    "PICK_INSERT_RESET_MAX_FINGER_JOINT_SPEED_M_S",
    "PICK_INSERT_RESET_MAX_CABLE_SPEED_M_S",
    "PICK_INSERT_RESET_MAX_BODY_EXCURSION_M",
    "PICK_INSERT_RESET_MAX_PLUG_EXCURSION_M",
    "PICK_INSERT_RESET_MAX_SOCKET_EXCURSION_M",
    "PICK_INSERT_RESET_MAX_ARM_TARGET_CLAMP_DELTA_RAD",
    "PICK_INSERT_VBD_POSE_HISTORY_CONTRACT_VERSION",
    "PICK_INSERT_VBD_POSE_HISTORY_ENTRY_NAME",
    "PICK_INSERT_GOAL_MAX_SOCKET_DRIFT_M",
    "PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M",
    "PICK_INSERT_GOAL_MAX_CABLE_SPEED_M_S",
    "PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD",
    "RESET_DATASET_GOAL_STATE_NAMES",
    "RESET_DATASET_STATE_NAMES",
    "franka_rj45_validation_source_sha256",
    "fast_reset_validation_report_validate_runtime",
    "pick_insert_fast_reset_phase_0_band_fraction_tolerance",
    "pick_insert_reset_dataset_row_digest",
    "reset_dataset_content_digest",
    "reset_dataset_digest",
    "reset_dataset_validate_full_pick_diversity",
    "reset_dataset_validate_phase_row_counts",
    "reset_dataset_validate_runtime",
    "reset_validation_report_content_digest",
    "reset_validation_report_validate_runtime",
]
