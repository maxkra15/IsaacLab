# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Anchored-end integrity checks for the dual-rack cable task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..dual_rack_env import FrankaRJ45DualRackInsertEnv


def anchored_cable_disconnected(env: FrankaRJ45DualRackInsertEnv) -> torch.Tensor:
    """Terminate if the nominally fixed cable endpoint leaves its seated plug."""
    error = torch.linalg.vector_norm(
        env.anchored_cable_endpoint_position_e() - env.anchored_cable_target_position_e(),
        dim=-1,
    )
    return ~torch.isfinite(error) | (error > float(env.cfg.anchored_cable_endpoint_tolerance_m))


__all__ = ["anchored_cable_disconnected"]
