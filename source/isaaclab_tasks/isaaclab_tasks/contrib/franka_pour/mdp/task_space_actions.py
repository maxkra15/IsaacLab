# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-space actions for the Franka pour task."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils.modifiers import DigitalFilter, DigitalFilterCfg

from .actions import DifferentialInverseKinematicsActionMovingAverageCfg


class DifferentialInverseKinematicsActionMovingAverage(DifferentialInverseKinematicsAction):
    """Differential IK action with a three-sample moving average on Cartesian commands.

    Filtering is applied only to commands sent to the IK controller. The raw policy-action
    buffer remains unchanged so action observations and rewards retain their usual semantics.
    """

    cfg: DifferentialInverseKinematicsActionMovingAverageCfg

    def __init__(self, cfg: DifferentialInverseKinematicsActionMovingAverageCfg, env) -> None:
        super().__init__(cfg, env)
        self._action_filter = DigitalFilter(
            DigitalFilterCfg(A=[0.0], B=[1.0 / 3.0] * 3),
            data_dim=self.raw_actions.shape,
            device=self.device,
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        """Filter Cartesian commands without replacing the raw policy actions."""
        super().process_actions(self._action_filter(actions))
        self._raw_actions.copy_(actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear raw actions and filter history for the selected environments."""
        super().reset(env_ids)
        self._action_filter.reset(env_ids)
