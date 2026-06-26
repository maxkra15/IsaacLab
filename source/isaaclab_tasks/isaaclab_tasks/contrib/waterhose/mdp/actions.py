# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waterhose-specific action terms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab_newton.envs.mdp.actions.newton_ik_actions import NewtonInverseKinematicsAction
from isaaclab_newton.ik.newton_ik_objectives_cfg import NewtonIKPoseObjectiveCfg

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils.configclass import configclass

from . import contact_debug

# Matches ``WaterhoseSpaceMouseCfg.twist_sign`` and the AVP pipeline convention.
_TELEOP_TWIST_SIGN = -1.0
_EE_FRAME_TWIST_TOL = 1.0e-5


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

    max_joint_delta_per_step: float | None = None
    """Maximum per-step joint target change [m or rad, depending on joint type]."""


def _rate_limit_joint_targets(
    previous_targets: torch.Tensor, desired_targets: torch.Tensor, max_delta_per_step: float
) -> torch.Tensor:
    """Clamp desired joint targets to a per-step delta from previous targets."""

    max_delta = max(0.0, float(max_delta_per_step))
    delta = torch.clamp(desired_targets - previous_targets, min=-max_delta, max=max_delta)
    return previous_targets + delta


def _remap_teleop_rotvec_to_local_ee_roll(rotvec: torch.Tensor) -> torch.Tensor:
    """Map teleop rotation vectors to a single local EE roll channel.

    SpaceMouse already emits the desired gripper twist as ``[0, 0, twist]``.
    AVP wrist roll arrives from ``Se3RelRetargeter`` on the first rotation-vector
    component, with wrist pitch/yaw mixed into the remaining components. The
    waterhose teleop task only needs the gripper-roll component, so AVP roll is
    converted into the same local z-axis twist command that SpaceMouse uses.
    """

    is_ee_frame_twist = (rotvec[:, 0].abs() < _EE_FRAME_TWIST_TOL) & (rotvec[:, 1].abs() < _EE_FRAME_TWIST_TOL)
    twist = torch.where(is_ee_frame_twist, rotvec[:, 2], rotvec[:, 0] * _TELEOP_TWIST_SIGN)
    remapped = torch.zeros_like(rotvec)
    remapped[:, 2] = twist
    return remapped


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
        # Optional per-step contact logging (WATERHOSE_DEBUG_CONTACTS); a no-op when unset. The gripper
        # action runs every step in both the scripted demo and teleop, so this is the shared hook.
        contact_debug.log_contacts_if_enabled()
        self._raw_actions[:] = actions
        close_alpha = torch.clamp((1.0 - actions) * 0.5, 0.0, 1.0)
        desired_actions = self._open_command + close_alpha * (self._close_command - self._open_command)
        if self.cfg.max_joint_delta_per_step is None:
            self._processed_actions[:] = desired_actions
        else:
            self._processed_actions[:] = _rate_limit_joint_targets(
                self._processed_actions,
                desired_actions,
                self.cfg.max_joint_delta_per_step,
            )

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
            raise ValueError(
                "WaterhoseLocalFrameNewtonInverseKinematicsAction expects the primary pose objective first."
            )
        primary = pose_cfgs[0]
        self._local_frame_active = primary.command_type == "pose" and primary.use_relative_mode
        self._ee_offset_quat = torch.tensor(primary.body_offset_rot, dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )
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
            local_ee_roll = _remap_teleop_rotvec_to_local_ee_roll(actions[:, 3:6])
            actions[:, 3:6] = math_utils.quat_apply(ee_quat_b, local_ee_roll)
        super().process_actions(actions)


class WaterhoseTeleopPinnedNewtonIkAction(WaterhoseLocalFrameNewtonInverseKinematicsAction):
    """Teleop Newton IK that pins the torso and left gripper, like the scripted demo.

    The operator drives only the primary (right end-effector) relative pose. The demo's ``left_hold``
    and ``torso_hold`` pose objectives are kept, but their targets are captured once from the bodies'
    settled poses and held fixed, so the torso and left gripper stay put while the right arm tracks the
    teleop command -- the same multi-body hold IK the scripted demo uses, with the holds captured at
    teleop start instead of written each step by the state machine.

    The external action dimension is just the right-end-effector slice; the hold slices are filled
    internally, so it matches what the teleop pipeline emits (the pipeline drives only the
    end-effector). ``WaterhoseLocalFrameNewtonInverseKinematicsAction`` applies the primary
    orientation delta in the end-effector frame before the base solve, matching
    :class:`~isaaclab_tasks.contrib.waterhose.teleop.WaterhoseSpaceMouse`.
    """

    _HOLD_OBJECTIVE_NAMES = ("left_hold", "torso_hold")

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._hold_drivers = [d for d in self._drivers if d.objective.name in self._HOLD_OBJECTIVE_NAMES]
        if not self._hold_drivers:
            raise ValueError(
                "WaterhoseTeleopPinnedNewtonIkAction requires the 'left_hold' and 'torso_hold' pose objectives."
            )
        # The operator command covers only the primary (first) objective -- the right end-effector.
        # The hold slices are auto-filled, so the externally-exposed action dim excludes them and
        # matches what the teleop pipeline produces.
        self._teleop_action_dim = int(self._drivers[0].objective.action_dim)
        self._full_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._hold_targets_b = torch.zeros(self.num_envs, len(self._hold_drivers), 7, device=self.device)
        self._holds_captured = False

    @property
    def action_dim(self) -> int:
        return self._teleop_action_dim

    def _capture_hold_targets(self) -> None:
        # Pin each hold body at its current pose, expressed in the root frame as (pos, xyzw) -- the
        # absolute-pose convention the base IK expects for these objectives.
        body_pos_w = self._asset.data.body_pos_w.torch
        body_quat_w = self._asset.data.body_quat_w.torch
        root_pos_w = self._asset.data.root_pos_w.torch
        root_quat_w = self._asset.data.root_quat_w.torch
        for index, driver in enumerate(self._hold_drivers):
            pos_b, quat_b = math_utils.subtract_frame_transforms(
                root_pos_w, root_quat_w, body_pos_w[:, driver.body_idx], body_quat_w[:, driver.body_idx]
            )
            self._hold_targets_b[:, index, :3] = pos_b
            self._hold_targets_b[:, index, 3:] = quat_b

    def process_actions(self, actions: torch.Tensor) -> None:
        if not self._holds_captured:
            self._capture_hold_targets()
            self._holds_captured = True
        self._full_actions.zero_()
        self._full_actions[:, : self._teleop_action_dim] = actions[:, : self._teleop_action_dim]
        for index, driver in enumerate(self._hold_drivers):
            offset = driver.action_offset
            self._full_actions[:, offset : offset + 7] = self._hold_targets_b[:, index]
        super().process_actions(self._full_actions)
