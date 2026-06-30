# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the physical-grasp, two-cup Franka pour task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..pour_env import FrankaPourEnv


def tcp_pose_obs(env: FrankaPourEnv) -> torch.Tensor:
    return torch.nan_to_num(env.tcp_pose_e())


def ee_pose_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Backward-compatible hand-pose observation used by older diagnostic scripts."""
    return torch.nan_to_num(env.ee_pose_e())


def cup_pose_obs(env: FrankaPourEnv) -> torch.Tensor:
    return torch.nan_to_num(env.cup_pose_e())


def target_pose_obs(env: FrankaPourEnv) -> torch.Tensor:
    return torch.nan_to_num(env.target_pose_e())


def tcp_to_grasp_obs(env: FrankaPourEnv) -> torch.Tensor:
    return torch.nan_to_num(env.cup_grasp_point_e() - env.tcp_pos_e())


def cup_to_target_obs(env: FrankaPourEnv) -> torch.Tensor:
    return torch.nan_to_num(env.target_pose_e()[:, :3] - env.cup_pose_e()[:, :3])


def gripper_width_obs(env: FrankaPourEnv) -> torch.Tensor:
    return torch.nan_to_num(env.gripper_width()).unsqueeze(-1)


def particle_fractions_obs(env: FrankaPourEnv) -> torch.Tensor:
    scale = max(env.num_particles, 1)
    source = env.count_in_source() / scale
    target = env.count_in_target() / scale
    spilled = torch.clamp(1.0 - source - target, min=0.0, max=1.0)
    return torch.stack((source, target, spilled), dim=-1)
