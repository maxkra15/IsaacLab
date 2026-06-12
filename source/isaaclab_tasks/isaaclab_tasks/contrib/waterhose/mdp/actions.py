# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waterhose-specific action terms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.envs.mdp.actions.newton_ik_actions import NewtonInverseKinematicsAction
from isaaclab_newton.ik.newton_ik_objectives_cfg import NewtonIKPoseObjectiveCfg


@configclass
class WaterhoseGripperPositionActionCfg(ActionTermCfg):
    """One-dimensional continuous position command for the RBY1 right gripper."""

    class_type: type[ActionTerm] | str = "{DIR}.actions:WaterhoseGripperPositionAction"

    joint_names: list[str] = MISSING
    """Right gripper driver and finger joints to command explicitly."""

    open_command_expr: dict[str, float] = MISSING
    """Joint position targets for a normalized action of ``+1``."""

    close_command_expr: dict[str, float] = MISSING
    """Joint position targets for a normalized action of ``-1``."""


class WaterhoseGripperPositionAction(ActionTerm):
    """Interpolates one scalar into explicit right-gripper joint position targets."""

    cfg: WaterhoseGripperPositionActionCfg

    def __init__(self, cfg: WaterhoseGripperPositionActionCfg, env):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)

        self._raw_actions = torch.ones(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        self._open_command = torch.zeros(self._num_joints, device=self.device)
        indices, names, values = string_utils.resolve_matching_names_values(
            self.cfg.open_command_expr, self._joint_names
        )
        if len(indices) != self._num_joints:
            raise ValueError(f"Missing open gripper targets for: {set(self._joint_names) - set(names)}")
        self._open_command[indices] = torch.tensor(values, device=self.device)

        self._close_command = torch.zeros_like(self._open_command)
        indices, names, values = string_utils.resolve_matching_names_values(
            self.cfg.close_command_expr, self._joint_names
        )
        if len(indices) != self._num_joints:
            raise ValueError(f"Missing close gripper targets for: {set(self._joint_names) - set(names)}")
        self._close_command[indices] = torch.tensor(values, device=self.device)

        self._processed_actions[:] = self._open_command

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
        self._raw_actions[:] = actions
        close_alpha = torch.clamp((1.0 - actions) * 0.5, 0.0, 1.0)
        self._processed_actions[:] = self._open_command + close_alpha * (self._close_command - self._open_command)

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target_index(target=self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 1.0
        self._processed_actions[env_ids] = self._open_command


class WaterhoseLocalFrameNewtonInverseKinematicsAction(NewtonInverseKinematicsAction):
    """Newton IK action that applies relative orientation deltas in the end-effector frame.

    The base action's relative mode composes the rotation delta in the root frame
    (``delta * ee``); teleop devices express the delta in the end-effector frame
    (``ee * delta``). The two compositions agree once the delta's rotation vector is
    rotated from the end-effector frame into the root frame, so this subclass rotates
    the primary pose objective's rotation-vector slice and delegates to the base.
    """

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        pose_cfgs = [obj for obj in cfg.objectives if isinstance(obj, NewtonIKPoseObjectiveCfg)]
        if not pose_cfgs or cfg.objectives[0] is not pose_cfgs[0]:
            raise ValueError("WaterhoseLocalFrameNewtonInverseKinematicsAction expects the primary pose objective first.")
        primary = pose_cfgs[0]
        self._local_frame_active = primary.command_type == "pose" and primary.use_relative_mode
        self._ee_offset_quat = torch.tensor(
            primary.body_offset_rot, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self._primary_body_idx = self._resolve_isaac_body_index(primary.body_name)

    def process_actions(self, actions: torch.Tensor) -> None:
        if self._local_frame_active:
            # Root-frame end-effector orientation (data and math are both (x, y, z, w)).
            body_quat_w = self._asset.data.body_quat_w.torch[:, self._primary_body_idx]
            root_quat_w = self._asset.data.root_quat_w.torch
            ee_quat_b = math_utils.quat_mul(
                math_utils.quat_inv(root_quat_w), math_utils.quat_mul(body_quat_w, self._ee_offset_quat)
            )
            actions = actions.clone()
            actions[:, 3:6] = math_utils.quat_apply(ee_quat_b, actions[:, 3:6])
        super().process_actions(actions)
