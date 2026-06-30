# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dense, physically staged rewards for grasping and pouring MPM media."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..pour_env import FrankaPourEnv


def _reach_quality(env: FrankaPourEnv, std: float) -> torch.Tensor:
    distance = torch.linalg.norm(env.tcp_pos_e() - env.cup_grasp_point_e(), dim=-1)
    return 1.0 - torch.tanh(distance / max(float(std), 1.0e-6))


def _closure_quality(env: FrankaPourEnv) -> torch.Tensor:
    travel = max(float(env.gripper_open_width) - float(env.gripper_grasp_width), 1.0e-4)
    return torch.clamp((float(env.gripper_open_width) - env.gripper_width()) / travel, 0.0, 1.0)


def reach_cup(env: FrankaPourEnv, std: float = 0.10) -> torch.Tensor:
    """Smooth Cartesian reach signal centred on the cup's actual grasp height."""
    return _reach_quality(env, std)


def grasp_cup(env: FrankaPourEnv, reach_std: float = 0.06) -> torch.Tensor:
    """Reward closing only while the TCP is aligned with the source cup."""
    return _reach_quality(env, reach_std) * _closure_quality(env)


def lift_cup(env: FrankaPourEnv, target_height: float = 0.12, reach_std: float = 0.07) -> torch.Tensor:
    """Reward source-cup elevation, gated by a nearby closed gripper."""
    height = torch.clamp(
        (env.cup_pose_e()[:, 2] - float(env.cup_reset_height)) / max(float(target_height), 1.0e-6),
        0.0,
        1.0,
    )
    return height * grasp_cup(env, reach_std=reach_std)


def lift_command_progress(env: FrankaPourEnv, target_height: float = 0.12, reach_std: float = 0.07) -> torch.Tensor:
    """Immediate curriculum credit for commanding upward motion while physically grasped.

    The credit fades to zero as the cup reaches ``target_height`` so repeatedly pushing into a
    workspace limit cannot outscore actually completing the lift.
    """
    height = torch.clamp(
        (env.cup_pose_e()[:, 2] - float(env.cup_reset_height)) / max(float(target_height), 1.0e-6),
        0.0,
        1.0,
    )
    upward = torch.clamp(env.action_manager.action[:, 2], 0.0, 1.0)
    return grasp_cup(env, reach_std=reach_std) * (1.0 - height) * upward


def align_cup_over_target(env: FrankaPourEnv, lift_height: float = 0.06, std: float = 0.12) -> torch.Tensor:
    """Reward horizontal source-to-receiver alignment only after lifting off the table."""
    cup = env.cup_pose_e()[:, :3]
    target = env.target_pose_e()[:, :3]
    lifted = torch.clamp((cup[:, 2] - float(env.cup_reset_height)) / max(float(lift_height), 1.0e-6), 0.0, 1.0)
    distance_xy = torch.linalg.norm(cup[:, :2] - target[:, :2], dim=-1)
    aligned = 1.0 - torch.tanh(distance_xy / max(float(std), 1.0e-6))
    return lifted * aligned


def align_command_progress(env: FrankaPourEnv, lift_height: float = 0.06, std: float = 0.12) -> torch.Tensor:
    """Immediate credit for a lifted cup's Cartesian action pointing toward the receiver."""
    cup = env.cup_pose_e()[:, :3]
    target = env.target_pose_e()[:, :3]
    lifted = torch.clamp((cup[:, 2] - float(env.cup_reset_height)) / max(float(lift_height), 1.0e-6), 0.0, 1.0)
    delta_xy = target[:, :2] - cup[:, :2]
    distance = torch.linalg.norm(delta_xy, dim=-1)
    direction = delta_xy / torch.clamp(distance[:, None], min=1.0e-6)
    toward = torch.clamp(torch.sum(env.action_manager.action[:, :2] * direction, dim=-1), 0.0, 1.0)
    remaining = torch.tanh(distance / max(float(std), 1.0e-6))
    return lifted * remaining * toward


def _cup_up_z(env: FrankaPourEnv) -> torch.Tensor:
    """World/env z component of the source cup's local +z axis."""
    quat = env.cup_pose_e()[:, 3:7]
    up = torch.zeros((quat.shape[0], 3), device=quat.device, dtype=quat.dtype)
    up[:, 2] = 1.0
    xyz = quat[:, :3]
    cross = 2.0 * torch.cross(xyz, up, dim=-1)
    rotated = up + quat[:, 3:4] * cross + torch.cross(xyz, cross, dim=-1)
    return rotated[:, 2]


def tilt_over_target(env: FrankaPourEnv, lift_height: float = 0.06, align_std: float = 0.10) -> torch.Tensor:
    """Reward tilting the lifted cup only after it is positioned over the receiver."""
    cup = env.cup_pose_e()[:, :3]
    lifted = torch.clamp((cup[:, 2] - float(env.cup_reset_height)) / max(float(lift_height), 1.0e-6), 0.0, 1.0)
    distance_xy = torch.linalg.norm(cup[:, :2] - env.target_pose_e()[:, :2], dim=-1)
    aligned = 1.0 - torch.tanh(distance_xy / max(float(align_std), 1.0e-6))
    # Zero below 60 degrees, one by 90 degrees. Upright loitering receives no tilt reward.
    tilt = torch.clamp((math.cos(math.pi / 3.0) - _cup_up_z(env)) / math.cos(math.pi / 3.0), 0.0, 1.0)
    return lifted * aligned * tilt


def tilt_command_progress(env: FrankaPourEnv, lift_height: float = 0.06, align_std: float = 0.10) -> torch.Tensor:
    """Immediate credit for the demonstrated +x wrist rotation once lifted and aligned."""
    cup = env.cup_pose_e()[:, :3]
    lifted = torch.clamp((cup[:, 2] - float(env.cup_reset_height)) / max(float(lift_height), 1.0e-6), 0.0, 1.0)
    distance_xy = torch.linalg.norm(cup[:, :2] - env.target_pose_e()[:, :2], dim=-1)
    aligned = 1.0 - torch.tanh(distance_xy / max(float(align_std), 1.0e-6))
    tilt = torch.clamp((math.cos(math.pi / 3.0) - _cup_up_z(env)) / math.cos(math.pi / 3.0), 0.0, 1.0)
    rotate_toward_pour = torch.clamp(env.action_manager.action[:, 3], 0.0, 1.0)
    return lifted * aligned * (1.0 - tilt) * rotate_toward_pour


def particles_in_target(env: FrankaPourEnv) -> torch.Tensor:
    return env.count_in_target() / max(env.num_particles, 1)


def particles_in_source(env: FrankaPourEnv) -> torch.Tensor:
    return env.count_in_source() / max(env.num_particles, 1)


def spilled_particles(env: FrankaPourEnv) -> torch.Tensor:
    return torch.clamp(1.0 - particles_in_source(env) - particles_in_target(env), min=0.0, max=1.0)


def pour_success_bonus(env: FrankaPourEnv) -> torch.Tensor:
    return (particles_in_target(env) >= float(env.pour_target_frac)).float()


def action_l2(env: FrankaPourEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action), dim=-1)
