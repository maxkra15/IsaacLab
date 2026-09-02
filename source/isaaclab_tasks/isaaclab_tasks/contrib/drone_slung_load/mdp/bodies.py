# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read Newton VBD maximal-coordinate rigid-body state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, quat_unique

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def link_pose_w(asset: RigidObject) -> torch.Tensor:
    """Root-link pose ``[position, quaternion_xyzw]`` from ``body_q``, shape ``(N, 7)``."""
    pose = asset.data.body_link_pose_w.torch
    return pose[:, 0] if pose.ndim == 3 else pose


def link_com_vel_w(asset: RigidObject) -> torch.Tensor:
    """Root-CoM spatial velocity ``[linear, angular]`` from ``body_qd``, shape ``(N, 6)``."""
    velocity = asset.data.body_com_vel_w.torch
    return velocity[:, 0] if velocity.ndim == 3 else velocity


def link_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root-link position in the world frame [m]."""
    return link_pose_w(env.scene[asset_cfg.name])[:, :3]


def link_pos_e(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root-link position in the environment frame [m]."""
    return link_pos_w(env, asset_cfg) - env.scene.env_origins


def link_quat_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Sign-standardized root-link orientation ``(x, y, z, w)`` in world."""
    return quat_unique(link_pose_w(env.scene[asset_cfg.name])[:, 3:7])


def link_lin_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root-CoM linear velocity in the world frame [m/s]."""
    return link_com_vel_w(env.scene[asset_cfg.name])[:, :3]


def link_ang_vel_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root-CoM angular velocity in the body frame [rad/s]."""
    angular_velocity_w = link_com_vel_w(env.scene[asset_cfg.name])[:, 3:6]
    return quat_apply_inverse(link_quat_w(env, asset_cfg), angular_velocity_w)


def link_height_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when root-link world height reaches ``minimum_height`` [m]."""
    return link_pos_w(env, asset_cfg)[:, 2] <= minimum_height
