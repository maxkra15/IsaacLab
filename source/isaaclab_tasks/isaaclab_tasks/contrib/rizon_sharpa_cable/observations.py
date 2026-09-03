# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local cable observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def cable_segment_positions(env: ManagerBasedEnv, asset_name: str = "cable") -> torch.Tensor:
    """Return all cable segment positions in world coordinates [m]."""
    return env.scene[asset_name].data.segment_pose_w.torch[..., :3]


def connector_pose(env: ManagerBasedEnv, asset_name: str = "cable") -> torch.Tensor:
    """Return the RJ45 connector-center pose as ``xyz + xyzw`` in world coordinates."""
    position, quaternion = env.scene[asset_name].get_connector_pose_w()
    return torch.cat((position, quaternion), dim=-1)


def insertion_socket_pose(env: ManagerBasedEnv, asset_name: str = "cable") -> torch.Tensor:
    """Return the fixed socket pose as ``xyz + xyzw`` in world coordinates."""
    spawn_cfg = env.scene[asset_name].cfg.spawn
    position_e = torch.tensor(
        spawn_cfg.insertion_target_position_e,
        device=env.device,
        dtype=env.scene.env_origins.dtype,
    )
    quaternion_e = torch.tensor(
        spawn_cfg.insertion_target_rotation_xyzw,
        device=env.device,
        dtype=env.scene.env_origins.dtype,
    )
    position_w = env.scene.env_origins + position_e.unsqueeze(0)
    quaternion_w = quaternion_e.unsqueeze(0).expand(env.num_envs, -1)
    return torch.cat((position_w, quaternion_w), dim=-1)


__all__ = ["cable_segment_positions", "connector_pose", "insertion_socket_pose"]
