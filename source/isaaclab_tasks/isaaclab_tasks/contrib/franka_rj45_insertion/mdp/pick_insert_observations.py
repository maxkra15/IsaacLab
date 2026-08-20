# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal-conditioned observations for the full Franka RJ45 pick-and-insert task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from ..pick_insert_env import FrankaRJ45PickInsertEnv


def _canonical_pose(pose: torch.Tensor) -> torch.Tensor:
    pose = torch.nan_to_num(pose)
    return torch.cat((pose[..., :3], math_utils.quat_unique(pose[..., 3:7])), dim=-1)


def socket_pose_obs(env: FrankaRJ45PickInsertEnv) -> torch.Tensor:
    """Current reset-conditioned socket pose in the environment frame."""
    return _canonical_pose(env.socket_pose_e())


def goal_plug_pose_obs(env: FrankaRJ45PickInsertEnv) -> torch.Tensor:
    """Solved plug pose associated with the current randomized socket."""
    return _canonical_pose(env.goal_plug_pose_e())


def tcp_velocity_obs(env: FrankaRJ45PickInsertEnv) -> torch.Tensor:
    """TCP linear and angular velocity, bounded against transient contact spikes."""
    return torch.nan_to_num(env.tcp_velocity_e()).clamp_(-20.0, 20.0)


def arm_target_error_obs(env: FrankaRJ45PickInsertEnv) -> torch.Tensor:
    """Persistent absolute arm target minus measured joint position [rad]."""
    action = env.action_manager.get_term("arm_action")
    return torch.nan_to_num(action.target_tracking_error).clamp_(-1.0, 1.0)


def tcp_grasp_error_obs(env: FrankaRJ45PickInsertEnv) -> torch.Tensor:
    """TCP translation and rotation error relative to the plug's grasp frame."""
    tcp = env.tcp_pose_e()
    target = env.desired_tcp_grasp_pose_e()
    translation = math_utils.quat_apply_inverse(target[:, 3:7], tcp[:, :3] - target[:, :3])
    rotation = math_utils.quat_unique(math_utils.quat_mul(math_utils.quat_conjugate(target[:, 3:7]), tcp[:, 3:7]))
    axis_angle = math_utils.axis_angle_from_quat(rotation)
    return torch.nan_to_num(torch.cat((translation, axis_angle), dim=-1)).clamp_(-1.0, 1.0)


def plug_goal_pose_error_obs(env: FrankaRJ45PickInsertEnv) -> torch.Tensor:
    """Plug-to-solved-goal SE(3) error in the randomized goal frame plus latch angle."""
    translation = env.plug_goal_translation_error_local()
    rotation = env.plug_goal_orientation_error_axis_angle()
    _, _, latch_angle = env.goal_error_components()
    return torch.nan_to_num(torch.cat((translation, rotation, latch_angle[:, None]), dim=-1)).clamp_(-1.0, 1.0)


def sampled_cable_positions_obs(env: FrankaRJ45PickInsertEnv) -> torch.Tensor:
    """Cable centers in the socket frame, including the extended free span and pinned tail."""
    current = env.task_body_pose_e()[:, env._cable_observation_body_indices, :3]
    socket = env.socket_pose_e()
    relative = math_utils.quat_apply_inverse(
        socket[:, None, 3:7].expand(-1, current.shape[1], -1),
        current - socket[:, None, :3],
    )
    limit = float(env.cfg.max_cable_socket_offset)
    return torch.nan_to_num(relative.reshape(env.num_envs, -1)).clamp_(-limit, limit)


def sampled_cable_linear_velocities_obs(env: FrankaRJ45PickInsertEnv) -> torch.Tensor:
    """Sampled cable linear velocities in the randomized socket frame [m/s]."""
    limit = float(env.cfg.max_task_body_linear_speed)
    linear_velocity_w = env.task_body_velocity()[:, env._cable_observation_body_indices, :3]
    linear_velocity_w = torch.nan_to_num(
        linear_velocity_w,
        nan=0.0,
        posinf=limit,
        neginf=-limit,
    ).clamp_(-limit, limit)
    socket = env.socket_pose_e()
    relative = math_utils.quat_apply_inverse(
        socket[:, None, 3:7].expand(-1, linear_velocity_w.shape[1], -1),
        linear_velocity_w,
    )
    return torch.nan_to_num(relative.reshape(env.num_envs, -1)).clamp_(-limit, limit)


def grasp_stage_obs(env: FrankaRJ45PickInsertEnv) -> torch.Tensor:
    """Current gated physical grasp and latched acquisition state."""
    if not hasattr(env, "termination_manager"):
        current = acquired = torch.zeros(env.num_envs, device=env.device)
    else:
        tracker = env.pick_insert_stage_tracker()
        current = tracker.current_grasp.float()
        acquired = tracker.ever_grasped.float()
    return torch.stack((current, acquired), dim=-1)


def grasp_proxy_contact_obs(env: FrankaRJ45PickInsertEnv) -> torch.Tensor:
    """Whether both fingers currently contact the plug's dedicated grasp proxy."""
    if not hasattr(env, "termination_manager"):
        contact = torch.zeros(env.num_envs, device=env.device)
    else:
        contact = env.pick_insert_stage_tracker().proxy_contact.float()
    return contact.unsqueeze(-1)


def reset_phase_obs(env: FrankaRJ45PickInsertEnv) -> torch.Tensor:
    """Reset recipe for the asymmetric critic only."""
    rows = env.reset_dataset_row_id.clamp_min(0)
    return env._reset_dataset_states["phase"][rows].float().unsqueeze(-1) / 5.0


__all__ = [
    "arm_target_error_obs",
    "goal_plug_pose_obs",
    "grasp_proxy_contact_obs",
    "grasp_stage_obs",
    "plug_goal_pose_error_obs",
    "reset_phase_obs",
    "sampled_cable_linear_velocities_obs",
    "sampled_cable_positions_obs",
    "socket_pose_obs",
    "tcp_grasp_error_obs",
    "tcp_velocity_obs",
]
