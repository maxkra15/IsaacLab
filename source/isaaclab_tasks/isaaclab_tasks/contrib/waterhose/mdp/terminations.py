# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waterhose-specific termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import normalize, quat_apply

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def plug_inserted_in_socket(
    env: ManagerBasedRLEnv,
    plug_cfg: SceneEntityCfg = SceneEntityCfg("plug1"),
    cable_cfg: SceneEntityCfg | None = None,
    socket_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    socket_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    connector_tip_len: float = 0.0,
    cable_tip_offset: float = 0.0,
    radial_threshold: float = 0.012,
    min_depth: float = 0.003,
    max_depth: float = 0.010,
    alignment_threshold: float = 0.75,
) -> torch.Tensor:
    """Return true when the connector head is aligned with and inserted into the socket mouth.

    The socket pose is specified in each environment's local frame. Depth is measured along the
    socket's local +Z axis. When ``cable_cfg`` is provided, the predicate uses the cable registry's
    segment-0 Newton body as the hose head and treats cable local ``-Z`` as the connector insertion
    axis. This matches the scripted insertion controller, where cable local ``+Z`` points back along
    the hose.
    """

    if cable_cfg is not None:
        cable = env.scene[cable_cfg.name]
        registry_entry = cable._registry_entry
        if len(registry_entry.segment_body_indices) < env.num_envs:
            raise RuntimeError(
                f"Cable registry for '{cable_cfg.name}' has {len(registry_entry.segment_body_indices)} worlds, "
                f"but the environment has {env.num_envs} envs."
            )
        body_indices_tuple = tuple(int(segments[0]) for segments in registry_entry.segment_body_indices[: env.num_envs])

        body_q = wp.to_torch(NewtonManager.get_state_0().body_q)
        cache = getattr(plug_inserted_in_socket, "_body_id_cache", {})
        cache_key = (id(registry_entry), body_indices_tuple, str(body_q.device))
        body_indices = cache.get(cache_key)
        if body_indices is None:
            body_indices = torch.tensor(body_indices_tuple, device=body_q.device, dtype=torch.long)
            cache[cache_key] = body_indices
            setattr(plug_inserted_in_socket, "_body_id_cache", cache)

        connector_pose_w = body_q[body_indices]
        connector_pos_w = connector_pose_w[:, :3]
        connector_quat_w = normalize(connector_pose_w[:, 3:])
        connector_axis_sign = -1.0
        axis_offset = float(cable_tip_offset)
    else:
        plug: RigidObject = env.scene[plug_cfg.name]
        plug_pose_w = plug.data.root_link_pose_w.torch
        connector_pos_w = plug_pose_w[:, :3]
        connector_quat_w = normalize(plug_pose_w[:, 3:])
        connector_axis_sign = 1.0
        axis_offset = float(connector_tip_len)

    device = connector_pos_w.device
    dtype = connector_pos_w.dtype
    num_envs = connector_pos_w.shape[0]

    local_z = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype).expand(num_envs, -1)
    connector_axis_w = normalize(quat_apply(connector_quat_w, local_z)) * connector_axis_sign
    connector_tip_pos_w = connector_pos_w + connector_axis_w * axis_offset

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
