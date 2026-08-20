# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compact robot, connector, latch, and cable observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from ..rj45_env import FrankaRJ45InsertionEnv


def _canonical_pose(pose: torch.Tensor) -> torch.Tensor:
    pose = torch.nan_to_num(pose)
    return torch.cat((pose[..., :3], math_utils.quat_unique(pose[..., 3:7])), dim=-1)


def tcp_pose_obs(env: FrankaRJ45InsertionEnv) -> torch.Tensor:
    """TCP pose in the environment frame."""
    return _canonical_pose(env.tcp_pose_e())


def plug_goal_error_obs(env: FrankaRJ45InsertionEnv) -> torch.Tensor:
    """Signed plug translation error in the fixed socket frame plus latch error."""
    _, _, latch_angle = env.goal_error_components()
    return torch.nan_to_num(torch.cat((env.plug_goal_translation_error(), latch_angle[:, None]), dim=-1))


def plug_pose_obs(env: FrankaRJ45InsertionEnv) -> torch.Tensor:
    """Plug pose in the fixed environment frame."""
    return _canonical_pose(env.plug_pose_e())


def plug_velocity_obs(env: FrankaRJ45InsertionEnv) -> torch.Tensor:
    """Plug angular/linear spatial velocity."""
    return torch.nan_to_num(env.task_body_velocity()[:, getattr(env, "_plug_task_body_index", 0)]).clamp_(-20.0, 20.0)


def sampled_cable_positions_obs(env: FrankaRJ45InsertionEnv) -> torch.Tensor:
    """Seven cable-segment centers relative to their canonical goal positions."""
    current = env.task_body_pose_e()[:, env._cable_observation_body_indices, :3]
    goal = env.goal_task_body_pose[env._cable_observation_body_indices, :3]
    limit = float(env.cfg.max_cable_goal_offset)
    return torch.nan_to_num((current - goal).reshape(env.num_envs, -1)).clamp_(-limit, limit)


def finger_position_obs(env: FrankaRJ45InsertionEnv) -> torch.Tensor:
    return torch.nan_to_num(env._robot.data.joint_pos.torch[:, env._finger_joint_ids])


def finger_velocity_obs(env: FrankaRJ45InsertionEnv) -> torch.Tensor:
    return torch.nan_to_num(env._robot.data.joint_vel.torch[:, env._finger_joint_ids])


def gripper_target_obs(env: FrankaRJ45InsertionEnv) -> torch.Tensor:
    return torch.nan_to_num(env.action_manager.get_term("gripper_action").commanded_position)


def gripper_contact_obs(env: FrankaRJ45InsertionEnv) -> torch.Tensor:
    return torch.nan_to_num(env.action_manager.get_term("gripper_action").contact_deflection)


def time_remaining_obs(env: FrankaRJ45InsertionEnv) -> torch.Tensor:
    progress = env.episode_length_buf.float() / max(int(env.max_episode_length), 1)
    return torch.clamp(1.0 - progress, 0.0, 1.0).unsqueeze(-1)


def reset_difficulty_obs(env: FrankaRJ45InsertionEnv) -> torch.Tensor:
    rows = env.reset_dataset_row_id.clamp_min(0)
    return env._reset_dataset_states["difficulty"][rows].unsqueeze(-1)


def success_dwell_obs(env: FrankaRJ45InsertionEnv) -> torch.Tensor:
    return torch.clamp(env._success_dwell_count.float() / env._success_dwell_steps, 0.0, 1.0).unsqueeze(-1)
