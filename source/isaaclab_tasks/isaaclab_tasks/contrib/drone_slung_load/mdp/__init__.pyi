# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# This package stub intentionally consists of public re-exports.
# ruff: noqa: F401, F403

from isaaclab.envs.mdp import *

from .actions import CollectiveThrustBodyRateAction, CollectiveThrustBodyRateActionCfg
from .bodies import (
    link_ang_vel_b,
    link_com_vel_w,
    link_height_below_minimum,
    link_lin_vel_w,
    link_pos_e,
    link_pos_w,
    link_pose_w,
    link_quat_w,
)
from .commands import WaypointSequenceCommand, WaypointSequenceCommandCfg
from .controllers import (
    FlareController,
    FlareControlOutput,
    allocate_rotor_thrusts,
    apply_motor_lag,
    body_rate_torque,
    quadrotor_allocation_matrix,
    quadrotor_rotor_geometry,
    reconstruct_wrench,
    scale_flare_actions,
)
from .curriculums import (
    DirectCTBRCurriculumV14,
    DirectCTBRRouteCurriculum,
    PrecisionSpeedCurriculum,
    PrecisionSpeedCurriculumV13,
)
from .episode_metrics import EpisodeMetricAccumulator
from .events import ResetSlungLoadEvent, reset_drone_state_on_annulus, reset_drone_state_uniform
from .geometry import (
    attachment_kinematics,
    cable_constraint_errors,
    cable_features,
    rotation_matrix_flat,
    straight_end_point,
    straight_segment_poses,
    swing_features,
    transverse_velocity,
)
from .observations import (
    body_ang_vel_normalized,
    body_lin_vel_normalized,
    body_rotation_matrix,
    cable_integrity_errors,
    cable_joint_error,
    cable_relative_separation,
    path_cross_track_error_b_normalized,
    path_curvature_b_normalized,
    path_preview_b_normalized,
    path_progress_fraction,
    path_speed_features,
    path_tangent_b,
    path_tracking_features_b,
    payload_attachment_b,
    payload_attachment_b_normalized,
    payload_lin_vel_w_normalized,
    payload_transverse_velocity_b,
    previous_action,
    swing_angles,
    swing_angles_normalized,
    swing_angular_velocity,
    swing_angular_velocity_normalized,
    total_swing_angle,
    total_swing_angle_normalized,
    upper_cable_tangent_b,
    waypoint_offsets_normalized,
    world_lin_vel_normalized,
)
from .rewards import (
    NormalizedActionAccelerationL2,
    RouteCompletionImpulse,
    WaypointAdvanceImpulse,
    action_delta_l2,
    body_angular_velocity_l2,
    body_tilt_exp,
    body_tilt_l2,
    crash_impulse,
    indexed_cross_track_error_exp,
    indexed_cross_track_error_l2,
    path_arc_length_progress,
    path_tangent_speed_tracking_l2,
    path_tracking_precision_exp,
    path_tracking_precision_log1p,
    path_transverse_speed_l2,
    path_velocity_tracking_l2,
    payload_transverse_speed_l2,
    record_episode_metrics,
    swing_safety_impulse,
    total_swing_angle_l2,
    unsafe_termination_impulse,
    waypoint_distance_progress,
    waypoint_progress,
)
from .spawners import (
    DroneCuboidCfg,
    PhysicsAttachmentCfg,
    spawn_cable_with_color,
    spawn_drone_cuboid,
    spawn_physics_attachment,
    spawn_sphere_with_color,
)
from .terminations import (
    active_waypoint_error_out_of_bounds,
    cable_integrity_violation,
    illegal_action,
    illegal_cable_state,
    illegal_link_state,
    out_of_workspace,
    path_corridor_violation,
    route_completed,
)
