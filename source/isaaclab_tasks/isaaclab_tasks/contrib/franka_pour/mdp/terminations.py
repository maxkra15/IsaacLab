# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the pour task.

Only ``time_out`` (from the standard IsaacLab terms) and ``nonfinite_failure`` end episodes; the
latter is the instability guard that makes a divergence reset rather than poison the batch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..pour_env import FrankaPourEnv


def _state_finite(
    robot_joint_pos: torch.Tensor,
    cup_body_q: torch.Tensor,
    particle_pos: torch.Tensor,
) -> torch.Tensor:
    """Return a per-environment finite-state mask using unsanitized simulation tensors."""
    robot_ok = torch.isfinite(robot_joint_pos).all(dim=-1)
    cup_ok = torch.isfinite(cup_body_q).all(dim=-1)
    media_ok = torch.isfinite(particle_pos).all(dim=(1, 2))
    return robot_ok & cup_ok & media_ok


def nonfinite_failure(env: FrankaPourEnv) -> torch.Tensor:
    """Terminate on non-finite simulation state (instability guard)."""
    return ~env.state_finite()
