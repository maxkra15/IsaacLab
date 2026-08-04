# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Internal rollout-boundary synchronization for distributed RSL-RL environments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym


def _wrap_distributed_training_state_sync(
    env: gym.Env,
    *,
    distributed: bool,
    step_interval: int,
) -> gym.Env:
    """Wrap an opt-in environment when distributed synchronization is required."""
    if not distributed:
        return env

    synchronize_training_state = getattr(env.unwrapped, "synchronize_training_state", None)
    if not callable(synchronize_training_state):
        return env

    return _TrainingStateSyncWrapper(
        env,
        step_interval=step_interval,
        synchronize_training_state=synchronize_training_state,
    )


class _TrainingStateSyncWrapper(gym.Wrapper):
    """Call an environment synchronization hook at fixed rollout intervals."""

    def __init__(
        self,
        env: gym.Env,
        *,
        step_interval: int,
        synchronize_training_state: Callable[[], None],
    ) -> None:
        """Initialize the rollout-boundary synchronization wrapper.

        Args:
            env: Gymnasium environment to wrap.
            step_interval: Number of environment steps in one RSL-RL rollout.
            synchronize_training_state: Callback that synchronizes queued environment training state.

        Raises:
            ValueError: If :paramref:`step_interval` is not a positive integer.
        """
        if not isinstance(step_interval, int) or isinstance(step_interval, bool) or step_interval < 1:
            raise ValueError("step_interval must be a positive integer.")

        super().__init__(env)
        self._step_interval = step_interval
        self._synchronize_training_state = synchronize_training_state
        self._step_count = 0

    def reset(self, **kwargs: Any) -> Any:
        """Reset the wrapped environment and restart the synchronization interval."""
        self._step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> Any:
        """Step the environment and synchronize after each completed rollout interval."""
        result = self.env.step(action)
        self._step_count += 1
        if self._step_count == self._step_interval:
            self._step_count = 0
            self._synchronize_training_state()
        return result
