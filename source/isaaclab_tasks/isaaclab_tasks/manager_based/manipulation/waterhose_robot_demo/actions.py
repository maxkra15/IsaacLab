# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""No-op action term for the scripted Newton reference demo wrapper."""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor
from isaaclab.managers import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.utils.configclass import configclass


@configclass
class ReferenceDemoNoOpActionCfg(ActionTermCfg):
    """Single-dimension no-op action for manager compatibility."""

    class_type: type[ActionTerm] = MISSING
    asset_name: str = "reference_demo"


class ReferenceDemoNoOpAction(ActionTerm):
    """Accepts actions but leaves the reference Newton demo fully scripted."""

    cfg: ReferenceDemoNoOpActionCfg

    def __init__(self, cfg: ReferenceDemoNoOpActionCfg, env):
        ManagerTermBase.__init__(self, cfg, env)
        self._IO_descriptor = GenericActionIODescriptor()
        self._export_IO_descriptor = True
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions.reshape(self.num_envs, self.action_dim)
        self._processed_actions.copy_(self._raw_actions)

    def apply_actions(self) -> None:
        # The reference demo's state machine writes all robot controls.
        return

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0

