# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms + transfer-success helper for the scoop task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..scoop_env import FrankaScoopEnv


def transfer_success_mask(env: FrankaScoopEnv) -> torch.Tensor:
    """True where the cup holds at least the curriculum's required amount of media (a successful scoop).

    Called every step (from the success reward). Success does NOT terminate the episode, so the policy
    can keep scooping; the per-episode flag is latched for the curriculum. The objective is FILLING the
    cup from the pile (``count_in_bowl``), not delivery to a target.
    """
    in_target = env.count_in_target()
    in_bowl = env.count_in_bowl()
    env.ep_max_in_target = torch.maximum(env.ep_max_in_target, in_target)
    env.ep_max_in_bowl = torch.maximum(env.ep_max_in_bowl, in_bowl)
    mask = in_bowl >= env.scoop_target_count
    env.episode_succeeded |= mask
    return mask


def delivered_success(env: FrankaScoopEnv) -> torch.Tensor:
    """Terminate the episode (success) once more than ``target_success_count`` particles reach the target bowl.

    Unlike :func:`transfer_success_mask` (the fill objective, which does NOT end the episode), delivering to
    the target ends it. The per-episode success flag and ``ep_max_in_target`` are latched for the curriculum
    and metrics.
    """
    in_target = env.count_in_target()
    env.ep_max_in_target = torch.maximum(env.ep_max_in_target, in_target)
    env.ep_max_in_bowl = torch.maximum(env.ep_max_in_bowl, env.count_in_bowl())
    mask = in_target > float(env.cfg.target_success_count)
    env.episode_succeeded |= mask
    return mask


def nonfinite_failure(env: FrankaScoopEnv) -> torch.Tensor:
    """Terminate on non-finite simulation state (instability guard)."""
    return ~env.state_finite()
