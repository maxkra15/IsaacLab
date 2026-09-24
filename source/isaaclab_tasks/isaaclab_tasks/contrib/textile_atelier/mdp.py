# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small state-based MDP terms for the textile atelier."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cloth_edge_positions(env: ManagerBasedRLEnv, cloth_cfg: SceneEntityCfg = SceneEntityCfg("cloth")) -> torch.Tensor:
    """Return left and right cloth-edge centroids in the environment frame.

    Args:
        env: Manager-based environment containing the VBD sheet.
        cloth_cfg: Scene entity selector for the VBD sheet.

    Returns:
        Edge-centroid positions [m], shape [N, 6].
    """
    nodes = env.scene[cloth_cfg.name].data.nodal_pos_w.torch
    edge_count = max(1, nodes.shape[1] // 8)
    left_ids = nodes[..., 0].topk(edge_count, dim=1, largest=False).indices
    right_ids = nodes[..., 0].topk(edge_count, dim=1, largest=True).indices
    left = nodes.gather(1, left_ids.unsqueeze(-1).expand(-1, -1, 3)).mean(dim=1)
    right = nodes.gather(1, right_ids.unsqueeze(-1).expand(-1, -1, 3)).mean(dim=1)
    origins = env.scene.env_origins
    return torch.cat((left - origins, right - origins), dim=-1)


def cloth_press_patch_masks(reference_nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Select mirrored positive-y material patches.

    Args:
        reference_nodes: Reference cloth-node positions [m], shape [N, V, 3].

    Returns:
        Left and right material masks, each shaped [N, V].
    """
    x = reference_nodes[..., 0]
    y = reference_nodes[..., 1]
    x_min = x.amin(dim=1, keepdim=True)
    x_max = x.amax(dim=1, keepdim=True)
    y_min = y.amin(dim=1, keepdim=True)
    y_max = y.amax(dim=1, keepdim=True)
    positive_y = y >= y_min + 0.675 * (y_max - y_min)
    left = (x <= x_min + 0.25 * (x_max - x_min)) & positive_y
    right = (x >= x_max - 0.25 * (x_max - x_min)) & positive_y
    return left, right


def _press_patch_heights(nodes: torch.Tensor, reference_nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    left, right = cloth_press_patch_masks(reference_nodes)
    left_z = (nodes[..., 2] * left).sum(dim=1) / left.sum(dim=1).clamp_min(1)
    right_z = (nodes[..., 2] * right).sum(dim=1) / right.sum(dim=1).clamp_min(1)
    return left_z, right_z


def cloth_press_profile(env: ManagerBasedRLEnv, cloth_cfg: SceneEntityCfg = SceneEntityCfg("cloth")) -> torch.Tensor:
    """Return left/right press-patch heights and their difference [m], shape [N, 3].

    Args:
        env: Manager-based environment containing the VBD sheet.
        cloth_cfg: Scene entity selector for the VBD sheet.

    Returns:
        Left height, right height, and right-minus-left height [m].
    """
    cloth = env.scene[cloth_cfg.name]
    nodes = cloth.data.nodal_pos_w.torch
    reference_nodes = cloth.data.default_nodal_state_w.torch[..., :3]
    left_z, right_z = _press_patch_heights(nodes, reference_nodes)
    origin_z = env.scene.env_origins[:, 2]
    return torch.stack((left_z - origin_z, right_z - origin_z, right_z - left_z), dim=-1)


def hand_cloth_proximity(
    env: ManagerBasedRLEnv,
    std: float,
    kuka_cfg: SceneEntityCfg = SceneEntityCfg("kuka", body_names=["palm_link", "(index|middle|thumb)_link_3"]),
    humanoid_cfg: SceneEntityCfg = SceneEntityCfg(
        "humanoid", body_names=["right_hand_pitch_link", "R_(index|middle|thumb).*_link"]
    ),
    cloth_cfg: SceneEntityCfg = SceneEntityCfg("cloth"),
) -> torch.Tensor:
    """Reward both end effectors for approaching real cloth nodes.

    Args:
        env: Manager-based environment containing the robots and VBD sheet.
        std: Exponential distance scale [m].
        kuka_cfg: Scene entity selector for the KUKA collidable hand links.
        humanoid_cfg: Scene entity selector for the GR1T2 hand links.
        cloth_cfg: Scene entity selector for the VBD sheet.

    Returns:
        Per-environment proximity reward, shape [N].
    """
    nodes = env.scene[cloth_cfg.name].data.nodal_pos_w.torch
    kuka_pos = env.scene[kuka_cfg.name].data.body_pos_w.torch[:, kuka_cfg.body_ids, :]
    humanoid_pos = env.scene[humanoid_cfg.name].data.body_pos_w.torch[:, humanoid_cfg.body_ids, :]
    kuka_distance = torch.cdist(kuka_pos, nodes).amin(dim=(1, 2))
    humanoid_distance = torch.cdist(humanoid_pos, nodes).amin(dim=(1, 2))
    return torch.exp(-(kuka_distance + humanoid_distance) / std)


def cloth_press_target(
    env: ManagerBasedRLEnv,
    target_asymmetry: float,
    std: float,
    cloth_cfg: SceneEntityCfg = SceneEntityCfg("cloth"),
) -> torch.Tensor:
    """Reward local left-side draping toward a positive height asymmetry.

    Args:
        env: Manager-based environment containing the VBD sheet.
        target_asymmetry: Desired right-minus-left patch height [m].
        std: Exponential reward scale [m].
        cloth_cfg: Scene entity selector for the VBD sheet.

    Returns:
        Per-environment cloth-shape reward, shape [N].
    """
    cloth = env.scene[cloth_cfg.name]
    nodes = cloth.data.nodal_pos_w.torch
    reference_nodes = cloth.data.default_nodal_state_w.torch[..., :3]
    left_z, right_z = _press_patch_heights(nodes, reference_nodes)
    return torch.exp(-torch.abs(right_z - left_z - target_asymmetry) / std)


def cloth_outside_workspace(
    env: ManagerBasedRLEnv,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    cloth_cfg: SceneEntityCfg = SceneEntityCfg("cloth"),
) -> torch.Tensor:
    """Terminate if any cloth node exits the environment-frame workspace box.

    Args:
        env: Manager-based environment containing the VBD sheet.
        x_bounds: Allowed x interval [m].
        y_bounds: Allowed y interval [m].
        z_bounds: Allowed z interval [m].
        cloth_cfg: Scene entity selector for the VBD sheet.

    Returns:
        Per-environment out-of-workspace flags, shape [N].
    """
    nodes = env.scene[cloth_cfg.name].data.nodal_pos_w.torch - env.scene.env_origins.unsqueeze(1)
    outside = (
        (nodes[..., 0] < x_bounds[0])
        | (nodes[..., 0] > x_bounds[1])
        | (nodes[..., 1] < y_bounds[0])
        | (nodes[..., 1] > y_bounds[1])
        | (nodes[..., 2] < z_bounds[0])
        | (nodes[..., 2] > z_bounds[1])
    )
    return outside.any(dim=1)
