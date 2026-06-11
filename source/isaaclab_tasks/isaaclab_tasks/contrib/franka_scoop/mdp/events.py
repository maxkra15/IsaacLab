# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event (reset) terms for the scoop transfer task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..scoop_env import FrankaScoopEnv


def reset_scoop_scene(env: FrankaScoopEnv, env_ids: torch.Tensor) -> None:
    """Reset the arm to home and re-pile the source media (curriculum-aware) for ``env_ids``."""
    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)
    env.reset_scoop_scene(env_ids.long())


def spawn_scoop_kit_visuals(env: FrankaScoopEnv, env_ids) -> None:
    """Author visual-only USD prims for task-specific custom Newton bodies."""
    del env_ids
    env.spawn_kit_visuals()
