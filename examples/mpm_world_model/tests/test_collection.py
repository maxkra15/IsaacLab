# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exploration bookkeeping contracts at the RSL environment boundary."""

from types import SimpleNamespace

import torch

from examples.mpm_world_model.collect_franka_pour import override_loaded_learning_rate, reset_exploration_state
from examples.mpm_world_model.recording import EpisodeMetrics


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


def test_finetune_learning_rate_overrides_loaded_optimizer():
    """An explicit fine-tune LR changes the actual update; default keeps resume semantics."""
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.3)
    algorithm = SimpleNamespace(optimizer=optimizer, learning_rate=0.3)
    assert override_loaded_learning_rate(algorithm, None) == [0.3]
    assert override_loaded_learning_rate(algorithm, 1.0e-5) == [1.0e-5]
    assert algorithm.learning_rate == 1.0e-5
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    torch.testing.assert_close(parameter, torch.tensor(0.99999), atol=1.0e-7, rtol=0.0)


def test_episode_metrics_keep_terminal_lengths_and_censoring():
    """Autoreseeding cannot overwrite a first episode's row, return or length."""
    metrics = EpisodeMetrics(2, 1 / 30)
    metrics.reset([0, 1], torch.tensor([17, 29]))
    metrics.update(
        torch.tensor([1.0, 2.0]),
        torch.tensor([1, 1]),
        torch.tensor([17, 29]),
        torch.tensor([True, False]),
        torch.tensor([False, False]),
        torch.tensor([True, False]),
        torch.tensor([[True], [False]]),
        ["success"],
        warmup=True,
    )
    metrics.reset([0], torch.tensor([31]))
    metrics.update(
        torch.tensor([3.0, 4.0]),
        torch.tensor([1, 2]),
        torch.tensor([31, 29]),
        torch.tensor([False, False]),
        torch.tensor([True, False]),
        torch.tensor([False, False]),
        torch.tensor([[False], [False]]),
        ["success"],
    )
    metrics.reset([0], torch.tensor([47]))
    result = metrics.summary()
    assert result["first_cohort_records"][0]["reset_row"] == 17
    assert result["first_cohort_records"][0]["episode_return"] == 1.0
    assert result["first_cohort_records"][1]["outcome"] == "unfinished"
    assert result["first_cohort_records"][1]["episode_return"] == 6.0
    assert result["completed_episode_length_sum_policy_steps"] == 1
    assert result["completed_episode_return_sum"] == 3.0
    assert result["completed_episode_records"][-1]["outcome"] == "timeout"
    assert result["mean_final_episode_length_policy_steps"] == 1.0
