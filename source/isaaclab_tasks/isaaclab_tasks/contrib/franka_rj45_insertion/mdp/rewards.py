# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terminal-sparse policy rewards for RJ45 insertion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..rj45_env import FrankaRJ45InsertionEnv


def insertion_success_bonus(env: FrankaRJ45InsertionEnv) -> torch.Tensor:
    """Return a one-shot, control-rate-independent successful terminal pulse."""
    if "success" not in env.termination_manager.active_terms:
        success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    else:
        success = env.termination_manager.get_term("success")
    return success.float() / max(float(env.step_dt), 1.0e-6)


def terminal_failure(env: FrankaRJ45InsertionEnv, include_time_out: bool = False) -> torch.Tensor:
    """Return a one-shot unsuccessful terminal penalty."""
    completed = env.termination_manager.dones if include_time_out else env.termination_manager.terminated
    failed = completed & ~env.episode_succeeded
    return failed.float() / max(float(env.step_dt), 1.0e-6)
