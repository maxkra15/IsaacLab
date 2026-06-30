# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small tensor utilities shared by Franka Pour reset orchestration."""

import torch


def boolean_selection_mask(count: int, selected: torch.Tensor) -> torch.Tensor:
    """Return a fixed-size boolean mask selecting the supplied indices."""
    if count < 0:
        raise ValueError(f"Mask length must be non-negative, got {count}.")
    mask = torch.zeros(count, dtype=torch.bool, device=selected.device)
    mask[selected.reshape(-1).long()] = True
    return mask
