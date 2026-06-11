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


def delivery_success_mask(env: FrankaScoopEnv) -> torch.Tensor:
    """True where more than ``scoop_target_count`` particles currently sit in the target bowl.

    The threshold is the runtime ``env.scoop_target_count``, set per stage by the curriculum
    (``curriculum_target_count``), so the delivery requirement ramps as training progresses.
    Called every step (from the success reward). Success deliberately does NOT terminate the
    episode: a success terminal would cut off the dense post-delivery reward stream, making
    "hold the media and run out the clock" out-pay delivering. Instead the per-episode success
    flag and the ``ep_max_*`` metrics are latched here for the curriculum and logging, and the
    dense delivered/success rewards keep paying for the rest of the episode (deliver earlier ->
    collect more).
    """
    in_target = env.count_in_target()
    env.ep_max_in_target = torch.maximum(env.ep_max_in_target, in_target)
    env.ep_max_in_bowl = torch.maximum(env.ep_max_in_bowl, env.count_in_bowl())
    mask = in_target > float(env.scoop_target_count)
    env.episode_succeeded |= mask
    return mask


def nonfinite_failure(env: FrankaScoopEnv) -> torch.Tensor:
    """Terminate on non-finite simulation state (instability guard)."""
    return ~env.state_finite()
