# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recorder terms for the waterhose robot demo."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass

from .manager import NewtonWaterhoseManager


def _slice_env_ids(value, env_ids: Sequence[int] | None):
    if env_ids is None:
        return value
    if isinstance(value, dict):
        return {key: _slice_env_ids(sub_value, env_ids) for key, sub_value in value.items()}
    if isinstance(value, torch.Tensor):
        return value[env_ids]
    return value


class WaterhoseInitialStateRecorder(RecorderTerm):
    """Record the Newton runtime state after reset."""

    def record_post_reset(self, env_ids: Sequence[int] | None):
        state = NewtonWaterhoseManager.get_recording_state(device=self._env.device)
        return "initial_state/waterhose", _slice_env_ids(state, env_ids)


class WaterhosePostStepStateRecorder(RecorderTerm):
    """Record the Newton runtime state after every environment step."""

    def record_post_step(self):
        return "states/waterhose", NewtonWaterhoseManager.get_recording_state(device=self._env.device)


@configclass
class WaterhoseInitialStateRecorderCfg(RecorderTermCfg):
    """Configuration for the waterhose initial-state recorder."""

    class_type: type[RecorderTerm] = WaterhoseInitialStateRecorder


@configclass
class WaterhosePostStepStateRecorderCfg(RecorderTermCfg):
    """Configuration for the waterhose post-step state recorder."""

    class_type: type[RecorderTerm] = WaterhosePostStepStateRecorder


@configclass
class WaterhoseActionStateRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Action/state recorder with task-local Newton runtime state."""

    record_initial_waterhose_state = WaterhoseInitialStateRecorderCfg()
    record_post_step_waterhose_state = WaterhosePostStepStateRecorderCfg()
