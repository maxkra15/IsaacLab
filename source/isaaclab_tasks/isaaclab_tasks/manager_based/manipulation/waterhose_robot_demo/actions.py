# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for the scripted waterhose robot demo."""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor
from isaaclab.managers import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.utils.configclass import configclass

@configclass
class ScriptedDemoActionCfg(ActionTermCfg):
    """SE(3) command action for the demo task."""

    class_type: type[ActionTerm] = MISSING
    asset_name: str = "waterhose_demo"
    command_dim: int = 7
    position_scale: float = 0.04
    rotation_scale: float = 0.25
    max_target_step: float = 0.018
    input_deadzone: float = 1.0e-6


class ScriptedDemoAction(ActionTerm):
    """Accepts relative end-effector commands from standard IsaacLab teleop scripts."""

    cfg: ScriptedDemoActionCfg

    def __init__(self, cfg: ScriptedDemoActionCfg, env):
        ManagerTermBase.__init__(self, cfg, env)
        self._IO_descriptor = GenericActionIODescriptor()
        self._export_IO_descriptor = True
        self._raw_actions = torch.zeros(self.num_envs, int(self.cfg.command_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._position_step_norm = torch.empty((self.num_envs, 1), device=self.device)
        self._position_step_scale = torch.empty_like(self._position_step_norm)

    @property
    def action_dim(self) -> int:
        return int(self.cfg.command_dim)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions.reshape(self.num_envs, self.action_dim).clamp(-1.0, 1.0)
        self._processed_actions[:, :3] = self._raw_actions[:, :3] * float(self.cfg.position_scale)
        max_target_step = float(self.cfg.max_target_step)
        if max_target_step > 0.0:
            position_step = self._processed_actions[:, :3]
            torch.linalg.vector_norm(position_step, dim=-1, keepdim=True, out=self._position_step_norm)
            self._position_step_norm.clamp_min_(1.0e-12)
            torch.div(max_target_step, self._position_step_norm, out=self._position_step_scale)
            self._position_step_scale.clamp_(max=1.0)
            position_step.mul_(self._position_step_scale)
        self._processed_actions[:, 3:6] = self._raw_actions[:, 3:6] * float(self.cfg.rotation_scale)
        self._processed_actions[:, 6:] = self._raw_actions[:, 6:]

    def apply_actions(self) -> None:
        from .coupled_manager import NewtonWaterhoseCoupledManager  # noqa: PLC0415

        command = self._processed_actions[0]
        if NewtonWaterhoseCoupledManager.teleop_enabled():
            NewtonWaterhoseCoupledManager.apply_teleop_command(command)
            return
        has_user_command = bool(torch.any(torch.abs(command) > float(self.cfg.input_deadzone)).item())
        if has_user_command:
            NewtonWaterhoseCoupledManager.set_teleop_enabled(True)
            NewtonWaterhoseCoupledManager.apply_teleop_command(command)
            return
        NewtonWaterhoseCoupledManager.apply_scripted_control()

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0


@configclass
class CoupledScriptedDemoActionCfg(ScriptedDemoActionCfg):
    """SE(3) command action for the coupled demo task."""

    pass


class CoupledScriptedDemoAction(ScriptedDemoAction):
    """Runs the coupled scripted controller, or routes non-zero commands to teleop."""

    cfg: CoupledScriptedDemoActionCfg
