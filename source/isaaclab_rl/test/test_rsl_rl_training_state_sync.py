# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-free tests for distributed RSL-RL training-state synchronization."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from isaaclab_rl.rsl_rl import _training_state


class _SyncEnv(gym.Env):
    """Minimal environment that records steps and synchronization calls."""

    def __init__(self):
        self.events: list[str] = []

    def step(self, action: Any):
        del action
        self.events.append("step")
        return 0, 0.0, False, False, {}

    def synchronize_training_state(self):
        self.events.append("sync")


def test_sync_wrapper_runs_at_rollout_boundaries_and_requires_opt_in():
    """Distributed opt-in environments synchronize once per rollout; all others remain unchanged."""
    env = _SyncEnv()
    wrapped = _training_state._wrap_distributed_training_state_sync(
        env,
        distributed=True,
        step_interval=3,
    )

    wrapped.step(None)
    wrapped.step(None)
    wrapped.step(None)
    wrapped.step(None)
    wrapped.step(None)
    wrapped.step(None)
    assert env.events == ["step", "step", "step", "sync", "step", "step", "step", "sync"]

    assert _training_state._wrap_distributed_training_state_sync(env, distributed=False, step_interval=3) is env
    env_without_hook = gym.Env()
    assert (
        _training_state._wrap_distributed_training_state_sync(
            env_without_hook,
            distributed=True,
            step_interval=3,
        )
        is env_without_hook
    )
