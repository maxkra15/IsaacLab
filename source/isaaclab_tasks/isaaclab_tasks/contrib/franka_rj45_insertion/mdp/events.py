# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for Franka RJ45 insertion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..rj45_env import FrankaRJ45InsertionEnv


def reset_rj45_scene(env: FrankaRJ45InsertionEnv, env_ids: torch.Tensor) -> None:
    """Restore the reset rows selected by the adaptive curriculum."""
    env.reset_rj45_scene(env_ids)
