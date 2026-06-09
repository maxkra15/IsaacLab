# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms: heightfield + proprioception (actor); particle counts (critic)."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..scoop_env import FrankaScoopEnv


# ---- actor (realistic) observations ----
def heightfield_obs(env: FrankaScoopEnv) -> torch.Tensor:
    """Top-down max-height grid over both containers (depth-camera-like), flattened."""
    return torch.nan_to_num(env.heightfield(), nan=0.0)


def particle_summary_obs(env: FrankaScoopEnv) -> torch.Tensor:
    """Compact particle-location summary for learning: centroids relative to the scoop + counts."""
    bowl = env.bowl_pos_e()
    src_rel = (env.source_media_centroid_e() - bowl) / 0.3
    held_rel = (env.bowl_media_centroid_e() - bowl) / 0.3
    all_rel = (env.all_media_centroid_e() - bowl) / 0.3
    scale = max(float(env.cfg.success_particle_count), 1.0)
    counts = torch.stack(
        (
            env.count_in_bowl() / scale,
            env.count_in_source() / max(env._num_particles, 1),
            env.count_in_target() / scale,
        ),
        dim=-1,
    )
    return torch.nan_to_num(torch.cat((src_rel, held_rel, all_rel, counts), dim=-1), nan=0.0).clamp(-2.0, 2.0)


def arm_joint_pos_norm(env: FrankaScoopEnv) -> torch.Tensor:
    return torch.nan_to_num(env.arm_joint_q() - env._default_arm_q, nan=0.0).clamp(-3.14, 3.14)


def arm_joint_vel_scaled(env: FrankaScoopEnv) -> torch.Tensor:
    return torch.nan_to_num(0.1 * env.arm_joint_qd(), nan=0.0).clamp(-1.0, 1.0)


def bowl_pose_obs(env: FrankaScoopEnv) -> torch.Tensor:
    """Normalized bowl position + tilt (sin/cos of pitch)."""
    center = 0.5 * (env._ws_lo + env._ws_hi)
    half = torch.clamp(0.5 * (env._ws_hi - env._ws_lo), min=1e-6)
    pos = torch.nan_to_num((env.bowl_pos_e() - center) / half, nan=0.0).clamp(-2.0, 2.0)
    pitch = env._pitch
    return torch.cat((pos, torch.sin(pitch).unsqueeze(-1), torch.cos(pitch).unsqueeze(-1)), dim=-1)


def bowl_to_source_obs(env: FrankaScoopEnv) -> torch.Tensor:
    return torch.nan_to_num((env._src_center - env.bowl_pos_e()) / 0.3, nan=0.0).clamp(-2.0, 2.0)


def bowl_to_target_obs(env: FrankaScoopEnv) -> torch.Tensor:
    return torch.nan_to_num((env._tgt_center - env.bowl_pos_e()) / 0.3, nan=0.0).clamp(-2.0, 2.0)


# ---- privileged (critic-only) observations ----
def count_in_bowl_obs(env: FrankaScoopEnv) -> torch.Tensor:
    return (env.count_in_bowl() / max(env._num_particles, 1)).unsqueeze(-1)


def count_in_source_obs(env: FrankaScoopEnv) -> torch.Tensor:
    return (env.count_in_source() / max(env._num_particles, 1)).unsqueeze(-1)


def count_in_target_obs(env: FrankaScoopEnv) -> torch.Tensor:
    return (env.count_in_target() / max(env._num_particles, 1)).unsqueeze(-1)
