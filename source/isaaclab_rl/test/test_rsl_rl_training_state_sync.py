# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-free tests for distributed RSL-RL training-state synchronization."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import pytest

from isaaclab_rl.rsl_rl import _training_state


class _FakeEnv(gym.Env):
    """Minimal environment that records wrapper call ordering."""

    def __init__(self):
        self.events: list[str] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del seed, options
        self.events.append("reset")
        return 0, {}

    def step(self, action: Any):
        del action
        self.events.append("step")
        return 0, 0.0, False, False, {}


class _SyncEnv(_FakeEnv):
    """Fake environment that opts in to training-state synchronization."""

    def synchronize_training_state(self):
        self.events.append("sync")


def test_sync_wrapper_runs_after_each_rollout_and_reset_restarts_interval():
    """Synchronization runs after the final rollout step and explicit reset restarts the interval."""
    env = _SyncEnv()
    wrapped = _training_state._wrap_distributed_training_state_sync(
        env,
        distributed=True,
        step_interval=3,
    )

    wrapped.step(None)
    wrapped.step(None)
    wrapped.reset()
    wrapped.step(None)
    wrapped.step(None)
    assert env.events == ["step", "step", "reset", "step", "step"]

    wrapped.step(None)
    assert env.events == ["step", "step", "reset", "step", "step", "step", "sync"]

    wrapped.step(None)
    wrapped.step(None)
    wrapped.step(None)
    assert env.events[-4:] == ["step", "step", "step", "sync"]


@pytest.mark.parametrize("step_interval", [0, -1, True, 1.5])
def test_sync_wrapper_rejects_invalid_intervals(step_interval: int | float | bool):
    """Synchronization intervals must contain at least one environment step."""
    env = _SyncEnv()

    with pytest.raises(ValueError, match="positive integer"):
        _training_state._TrainingStateSyncWrapper(
            env,
            step_interval=step_interval,
            synchronize_training_state=env.synchronize_training_state,
        )


def test_sync_wrapper_is_disabled_for_non_distributed_training():
    """Non-distributed training leaves an opt-in environment unwrapped."""
    env = _SyncEnv()

    wrapped = _training_state._wrap_distributed_training_state_sync(
        env,
        distributed=False,
        step_interval=3,
    )

    assert wrapped is env


def test_sync_wrapper_is_disabled_when_environment_has_no_hook():
    """Distributed training leaves environments without the synchronization hook unwrapped."""
    env = _FakeEnv()

    wrapped = _training_state._wrap_distributed_training_state_sync(
        env,
        distributed=True,
        step_interval=3,
    )

    assert wrapped is env
