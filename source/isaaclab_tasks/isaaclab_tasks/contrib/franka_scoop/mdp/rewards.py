# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the scoop source->target transfer task.

Simple and strict: reach the source media (bootstrap), hold media in the bowl,
deliver media to the target (the dense objective), a sparse success bonus, and a
mild action penalty. The curriculum tightens the success criterion over time.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .terminations import transfer_success_mask

if TYPE_CHECKING:
    from ..scoop_env import FrankaScoopEnv


def reach_source(env: FrankaScoopEnv, std: float = 0.12) -> torch.Tensor:
    """Reward an EMPTY bowl for approaching the source-media centroid (go scoop)."""
    empty = torch.exp(-env.count_in_bowl())  # ~1 when empty, ->0 once media is held
    dist = torch.linalg.norm(env.bowl_pos_e() - env.source_media_centroid_e(), dim=-1)
    return empty * torch.exp(-dist / std)


def carry_to_target(env: FrankaScoopEnv, std: float = 0.12) -> torch.Tensor:
    """Reward a LOADED bowl for approaching the target container (carry the scoop over)."""
    full = 1.0 - torch.exp(-env.count_in_bowl())  # ~0 empty, ->1 when media is held
    dist = torch.linalg.norm(env.bowl_pos_e() - env._tgt_center, dim=-1)
    return full * torch.exp(-dist / std)


def _success_scale(env: FrankaScoopEnv) -> float:
    return max(float(env.cfg.success_particle_count), 1.0)


def particles_in_bowl(env: FrankaScoopEnv) -> torch.Tensor:
    """Dense scoop reward: each particle in the scoop bowl matters up to the final success scale."""
    return torch.clamp(env.count_in_bowl() / _success_scale(env), max=1.0)


def particles_in_target(env: FrankaScoopEnv) -> torch.Tensor:
    """Dense delivery reward: each particle delivered to the target matters."""
    return env.count_in_target() / _success_scale(env)


def transfer_success_bonus(env: FrankaScoopEnv) -> torch.Tensor:
    """Sparse strict bonus once the required amount has reached the target."""
    return transfer_success_mask(env).float()


def removed_from_source(env: FrankaScoopEnv, norm: float = 100.0) -> torch.Tensor:
    """Fill proxy: fraction of a cupful of media scooped OUT of the source since reset.

    Uses only the reliable static source-container counter (not the moving-cup bowl counter, which is
    unreliable here). Rewards getting media out of the source as a bootstrap toward the delivery objective.
    """
    removed = torch.clamp(env._init_source_count - env.count_in_source(), min=0.0)
    return torch.clamp(removed / max(float(norm), 1.0), max=1.0)


def action_l2(env: FrankaScoopEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action), dim=-1)
