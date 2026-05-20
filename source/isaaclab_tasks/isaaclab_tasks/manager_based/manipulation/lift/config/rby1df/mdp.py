# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RBY1DF-specific lift MDP terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def reset_joints_to_positions(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    joint_positions: tuple[float, ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset selected articulation joints and their control targets to fixed positions."""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=asset.device)
    else:
        env_ids = torch.as_tensor(env_ids, device=asset.device)

    joint_ids = asset_cfg.joint_ids
    joint_count = asset.num_joints if isinstance(joint_ids, slice) else len(joint_ids)
    if len(joint_positions) != joint_count:
        raise ValueError(f"Expected {joint_count} joint positions, got {len(joint_positions)}.")

    joint_pos = torch.tensor(joint_positions, device=asset.device).repeat(len(env_ids), 1)
    joint_vel = torch.zeros_like(joint_pos)
    asset.write_joint_position_to_sim_index(position=joint_pos, joint_ids=joint_ids, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=joint_vel, joint_ids=joint_ids, env_ids=env_ids)
    asset.set_joint_position_target_index(target=joint_pos, joint_ids=joint_ids, env_ids=env_ids)
    asset.set_joint_velocity_target_index(target=joint_vel, joint_ids=joint_ids, env_ids=env_ids)


def reset_articulations_to_default(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    reset_joint_state: bool = False,
    reset_joint_targets: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset articulation targets without also writing joint or cube root state.

    RBY1DF is fixed-base in this task. Sparse Newton/MJWarp resets become
    unstable when the same timeout pass teleports the fixed-base robot root and
    the cube root, so this term never writes the articulation root state. It can
    still reset joint state and control targets to keep the robot initialized at
    the posture its PD controller is asked to hold.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=asset.device)
    else:
        env_ids = torch.as_tensor(env_ids, device=asset.device)

    joint_ids = asset_cfg.joint_ids
    default_joint_pos = asset.data.default_joint_pos.torch[env_ids].clone()
    default_joint_vel = asset.data.default_joint_vel.torch[env_ids].clone()
    if not isinstance(joint_ids, slice):
        default_joint_pos = default_joint_pos[:, joint_ids]
        default_joint_vel = default_joint_vel[:, joint_ids]

    if reset_joint_state:
        asset.write_joint_position_to_sim_index(position=default_joint_pos, joint_ids=joint_ids, env_ids=env_ids)
        asset.write_joint_velocity_to_sim_index(velocity=default_joint_vel, joint_ids=joint_ids, env_ids=env_ids)

    if reset_joint_targets:
        asset.set_joint_position_target_index(target=default_joint_pos, joint_ids=joint_ids, env_ids=env_ids)
        asset.set_joint_velocity_target_index(target=default_joint_vel, joint_ids=joint_ids, env_ids=env_ids)
