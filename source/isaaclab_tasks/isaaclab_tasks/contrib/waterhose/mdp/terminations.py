# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waterhose-specific termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import normalize, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def plug_inserted_in_socket(
    env: ManagerBasedRLEnv,
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable1"),
    socket_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    socket_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    connector_tip_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radial_threshold: float = 0.006,
    min_depth: float = -0.018,
    max_depth: float = -0.010,
    alignment_threshold: float = 0.95,
) -> torch.Tensor:
    """Return true when the connector head is aligned with and inserted into the socket mouth.

    The socket pose is specified in each environment's local frame. Depth is measured along the
    socket's local +Z axis. The predicate composes the connector transform with the cable-head body
    and measures the tip along connector local +Z.

    Args:
        env: The RL environment instance.
        cable_cfg: Waterhose cable asset whose attached connector is the reference.
        socket_pos: Socket-mouth position in the environment frame [m].
        socket_quat: Socket orientation as ``(x, y, z, w)``.
        connector_tip_offset: Physical connector-face centre in the connector frame [m].
        radial_threshold: Max allowed tip distance from the bore axis [m].
        min_depth: Min seated depth along the socket +Z axis [m].
        max_depth: Max seated depth along the socket +Z axis [m].
        alignment_threshold: Min cosine between the connector and socket axes (dimensionless).

    Returns:
        Boolean tensor of shape ``(num_envs,)``, true where the connector is seated and aligned.
    """

    cable = env.scene[cable_cfg.name]
    connector_pos_w, connector_quat_w = cable.get_connector_pose_w()

    device = connector_pos_w.device
    dtype = connector_pos_w.dtype
    num_envs = connector_pos_w.shape[0]

    local_z = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype).expand(num_envs, -1)
    connector_axis_w = normalize(quat_apply(connector_quat_w, local_z))
    connector_tip_offset_l = torch.tensor(connector_tip_offset, device=device, dtype=dtype).expand(num_envs, -1)
    connector_tip_pos_w = connector_pos_w + quat_apply(connector_quat_w, connector_tip_offset_l)

    socket_pos_l = torch.tensor(socket_pos, device=device, dtype=dtype).unsqueeze(0)
    socket_pos_w = socket_pos_l + env.scene.env_origins.to(device=device, dtype=dtype)
    socket_quat_w = torch.tensor(socket_quat, device=device, dtype=dtype).unsqueeze(0).expand(num_envs, -1)
    socket_axis_w = normalize(quat_apply(socket_quat_w, local_z))

    tip_delta_w = connector_tip_pos_w - socket_pos_w
    axial_depth = torch.sum(tip_delta_w * socket_axis_w, dim=-1)
    radial_delta_w = tip_delta_w - axial_depth.unsqueeze(-1) * socket_axis_w
    radial_dist = torch.linalg.vector_norm(radial_delta_w, dim=-1)
    axis_alignment = torch.sum(connector_axis_w * socket_axis_w, dim=-1)

    return (
        (radial_dist <= radial_threshold)
        & (axial_depth >= min_depth)
        & (axial_depth <= max_depth)
        & (axis_alignment >= alignment_threshold)
    )
