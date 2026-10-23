# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exploration bookkeeping contracts at the RSL environment boundary."""

import torch

from examples.mpm_world_model.collect_franka_pour import reset_exploration_state


def test_integer_done_flags_reset_only_ended_worlds():
    """RSL integer 0/1 flags select worlds as a mask, never as row indices."""
    done = torch.zeros(64, dtype=torch.long)
    done[[3, 45]] = 1
    noise = torch.ones((64, 8))
    selected = torch.ones((64, 1), dtype=torch.bool)
    result = reset_exploration_state(done, noise, selected, torch.Generator().manual_seed(7), 0.0)
    expected = torch.ones(64, dtype=torch.bool)
    expected[[3, 45]] = False
    assert result.dtype == torch.bool
    assert torch.equal(result.nonzero().flatten(), torch.tensor([3, 45]))
    assert torch.equal(selected[:, 0], expected)
    assert torch.equal(noise, expected[:, None].expand(-1, 8).float())
