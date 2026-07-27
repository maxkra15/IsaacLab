# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waterhose-specific action terms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
import warp as wp
from isaaclab_newton.envs.mdp.actions.newton_ik_actions import NewtonInverseKinematicsAction
from isaaclab_newton.ik.newton_ik_objectives_cfg import NewtonIKPoseObjectiveCfg

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils.configclass import configclass


@configclass
class WaterhoseGripperPositionActionCfg(ActionTermCfg):
    """One-dimensional continuous position command for one RBY1 gripper."""

    class_type: type[ActionTerm] | str = "{DIR}.actions:WaterhoseGripperPositionAction"

    joint_names: list[str] = MISSING
    """Gripper driver and finger joints to command explicitly."""

    open_command_expr: dict[str, float] = MISSING
    """Joint position targets for a normalized action of ``+1``."""

    close_command_expr: dict[str, float] = MISSING
    """Joint position targets for a normalized action of ``-1``."""

    max_joint_delta_per_step: float | None = None
    """Maximum per-step joint target change [m or rad, depending on joint type]."""


def _rate_limit_normalized_command(
    previous_command: torch.Tensor, desired_command: torch.Tensor, max_delta_per_step: float
) -> torch.Tensor:
    """Clamp a normalized command while preserving its multi-joint interpolation."""

    max_delta = max(0.0, float(max_delta_per_step))
    delta = torch.clamp(desired_command - previous_command, min=-max_delta, max=max_delta)
    return previous_command + delta


def _absolute_pose_is_uninitialized(commands: torch.Tensor, tolerance: float = 1.0e-6) -> torch.Tensor:
    """Return which absolute poses are IsaacTeleop's invalid origin sentinel."""

    return torch.all(torch.abs(commands[..., :3]) <= tolerance, dim=-1)


def _absolute_pose_is_valid(commands: torch.Tensor, tolerance: float = 1.0e-6) -> torch.Tensor:
    """Return which absolute poses contain a usable tracked position and quaternion."""

    finite = torch.all(torch.isfinite(commands), dim=-1)
    quaternion_valid = torch.linalg.vector_norm(commands[..., 3:7], dim=-1) > tolerance
    return finite & quaternion_valid & ~_absolute_pose_is_uninitialized(commands, tolerance)


def _rebase_absolute_pose_delta(
    commands: torch.Tensor,
    source_reference: torch.Tensor,
    target_reference: torch.Tensor,
) -> torch.Tensor:
    """Apply a tracked pose delta to a calibrated robot pose.

    Translation is mapped one-for-one. Rotation uses the spatial delta
    ``command * inverse(source_reference)`` and pre-multiplies the robot
    reference, which preserves all three wrist rotation axes while cancelling
    a constant tracker/tool-frame offset.
    """

    source_quat = math_utils.normalize(commands[..., 3:7])
    source_reference_quat = math_utils.normalize(source_reference[..., 3:7])
    target_reference_quat = math_utils.normalize(target_reference[..., 3:7])

    # Keep the quaternion representative continuous across q/-q tracker output.
    opposite_hemisphere = torch.sum(source_quat * source_reference_quat, dim=-1, keepdim=True) < 0.0
    source_quat = torch.where(opposite_hemisphere, -source_quat, source_quat)

    rotation_delta = math_utils.quat_mul(source_quat, math_utils.quat_inv(source_reference_quat))
    target_quat = math_utils.normalize(math_utils.quat_mul(rotation_delta, target_reference_quat))
    target_pos = target_reference[..., :3] + commands[..., :3] - source_reference[..., :3]
    return torch.cat((target_pos, target_quat), dim=-1)


def _update_clutched_absolute_pose(
    commands: torch.Tensor,
    current_targets: torch.Tensor,
    source_references: torch.Tensor,
    target_references: torch.Tensor,
    tracking_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update calibrated targets while making acquisition and reacquisition continuous."""

    valid = _absolute_pose_is_valid(commands)
    newly_valid = valid & ~tracking_valid
    source_references = torch.where(newly_valid.unsqueeze(-1), commands, source_references)
    target_references = torch.where(newly_valid.unsqueeze(-1), current_targets, target_references)

    # Keep invalid rows numerically well-formed even though their result is masked
    # out below. This avoids normalizing an all-zero startup quaternion.
    identity_pose = torch.zeros_like(commands)
    identity_pose[..., 6] = 1.0
    math_commands = torch.where(valid.unsqueeze(-1), commands, identity_pose)
    math_source_references = torch.where(valid.unsqueeze(-1), source_references, identity_pose)
    math_target_references = torch.where(valid.unsqueeze(-1), target_references, current_targets)
    rebased = _rebase_absolute_pose_delta(math_commands, math_source_references, math_target_references)
    updated_targets = torch.where(valid.unsqueeze(-1), rebased, current_targets)
    return updated_targets, source_references, target_references, valid


class WaterhoseGripperPositionAction(ActionTerm):
    """Interpolates one scalar into explicit gripper joint position targets."""

    cfg: WaterhoseGripperPositionActionCfg

    def __init__(self, cfg: WaterhoseGripperPositionActionCfg, env):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._joint_ids_warp = wp.array(self._joint_ids, dtype=wp.int32, device=self.device)
        self._num_joints = len(self._joint_ids)

        self._raw_actions = torch.ones(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._processed_close_alpha = torch.zeros(self.num_envs, 1, device=self.device)

        self._open_command = torch.zeros(self._num_joints, device=self.device)
        indices, names, values = string_utils.resolve_matching_names_values(
            self.cfg.open_command_expr, self._joint_names
        )
        if len(indices) != self._num_joints:
            raise ValueError(f"Missing open gripper targets for: {set(self._joint_names) - set(names)}")
        self._open_command[indices] = torch.tensor(values, device=self.device)
        open_values_by_index = {int(index): float(value) for index, value in zip(indices, values, strict=True)}

        self._close_command = torch.zeros_like(self._open_command)
        indices, names, values = string_utils.resolve_matching_names_values(
            self.cfg.close_command_expr, self._joint_names
        )
        if len(indices) != self._num_joints:
            raise ValueError(f"Missing close gripper targets for: {set(self._joint_names) - set(names)}")
        self._close_command[indices] = torch.tensor(values, device=self.device)
        close_values_by_index = {int(index): float(value) for index, value in zip(indices, values, strict=True)}

        self._processed_actions[:] = self._open_command
        self._max_close_alpha_delta = None
        if self.cfg.max_joint_delta_per_step is not None:
            # Compute this static configuration value on the host. Reading the device target tensor
            # here would introduce an unnecessary CUDA synchronization during environment setup.
            max_joint_travel = max(
                abs(close_values_by_index[index] - open_values_by_index[index]) for index in range(self._num_joints)
            )
            if max_joint_travel <= 0.0:
                raise ValueError("Open and close gripper commands must differ when rate limiting is enabled.")
            self._max_close_alpha_delta = max(0.0, float(self.cfg.max_joint_delta_per_step)) / max_joint_travel

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
        desired_close_alpha = torch.clamp((1.0 - actions) * 0.5, 0.0, 1.0)
        if self._max_close_alpha_delta is None:
            self._processed_close_alpha[:] = desired_close_alpha
        else:
            self._processed_close_alpha[:] = _rate_limit_normalized_command(
                self._processed_close_alpha,
                desired_close_alpha,
                self._max_close_alpha_delta,
            )
        self._processed_actions[:] = self._open_command + self._processed_close_alpha * (
            self._close_command - self._open_command
        )

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target_index(target=self._processed_actions, joint_ids=self._joint_ids_warp)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 1.0
        self._processed_close_alpha[env_ids] = 0.0
        self._processed_actions[env_ids] = self._open_command


@configclass
class WaterhoseDirectGripperJointPositionActionCfg(ActionTermCfg):
    """Direct joint-position targets for one RBY1 gripper."""

    class_type: type[ActionTerm] | str = "{DIR}.actions:WaterhoseDirectGripperJointPositionAction"

    joint_names: list[str] = MISSING
    """Gripper driver and finger joints to command explicitly."""

    open_command_expr: dict[str, float] = MISSING
    """Open joint-position targets, also used as one endpoint of the valid command range."""

    close_command_expr: dict[str, float] = MISSING
    """Closed joint-position targets, also used as the other endpoint of the valid command range."""


class WaterhoseDirectGripperJointPositionAction(ActionTerm):
    """Apply one explicit position target per RBY1 gripper joint."""

    cfg: WaterhoseDirectGripperJointPositionActionCfg

    def __init__(self, cfg: WaterhoseDirectGripperJointPositionActionCfg, env):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._joint_ids_warp = wp.array(self._joint_ids, dtype=wp.int32, device=self.device)
        self._num_joints = len(self._joint_ids)

        self._open_command = self._resolve_command(self.cfg.open_command_expr, "open")
        self._close_command = self._resolve_command(self.cfg.close_command_expr, "close")
        self._command_lower = torch.minimum(self._open_command, self._close_command)
        self._command_upper = torch.maximum(self._open_command, self._close_command)

        self._raw_actions = self._open_command.unsqueeze(0).repeat(self.num_envs, 1)
        self._processed_actions = self._raw_actions.clone()

    def _resolve_command(self, command_expr: dict[str, float], command_name: str) -> torch.Tensor:
        """Resolve one named command against the configured joint ordering."""

        command = torch.zeros(self._num_joints, device=self.device)
        indices, names, values = string_utils.resolve_matching_names_values(command_expr, self._joint_names)
        if len(indices) != self._num_joints:
            raise ValueError(f"Missing {command_name} gripper targets for: {set(self._joint_names) - set(names)}")
        command[indices] = torch.tensor(values, device=self.device)
        return command

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(actions, min=self._command_lower, max=self._command_upper)

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target_index(target=self._processed_actions, joint_ids=self._joint_ids_warp)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = self._open_command
        self._processed_actions[env_ids] = self._open_command


class WaterhoseBimanualTeleopNewtonIkAction(NewtonInverseKinematicsAction):
    """Calibrate both absolute wrist targets while holding the RBY1 torso fixed.

    IsaacTeleop rebases the two Apple Vision Pro wrist poses into the robot-root frame. This action
    clutches each newly tracked wrist to the robot's current wrist pose, then preserves subsequent
    translation and spatial rotation deltas one-for-one. Tracking loss holds the last target and
    reacquisition starts a fresh clutch, avoiding startup and dropout jumps. Newton can therefore
    use every joint in both arm chains while the torso remains stable.
    """

    _OPERATOR_OBJECTIVE_NAMES = ("right_ee", "left_ee")
    _HOLD_OBJECTIVE_NAMES = ("torso_hold",)

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        driver_names = tuple(driver.objective.name for driver in self._drivers)
        expected_names = self._OPERATOR_OBJECTIVE_NAMES + self._HOLD_OBJECTIVE_NAMES
        if driver_names != expected_names:
            raise ValueError(
                f"WaterhoseBimanualTeleopNewtonIkAction expects pose objectives {expected_names}, got {driver_names}."
            )

        self._operator_drivers = self._drivers[: len(self._OPERATOR_OBJECTIVE_NAMES)]
        self._hold_drivers = self._drivers[len(self._OPERATOR_OBJECTIVE_NAMES) :]
        operator_cfgs = [obj for obj in cfg.objectives if isinstance(obj, NewtonIKPoseObjectiveCfg)][
            : len(self._OPERATOR_OBJECTIVE_NAMES)
        ]
        self._teleop_action_dim = sum(driver.objective.action_dim for driver in self._operator_drivers)
        self._teleop_raw_actions = torch.zeros(self.num_envs, self._teleop_action_dim, device=self.device)
        self._teleop_processed_actions = torch.zeros_like(self._teleop_raw_actions)
        self._full_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._operator_offsets_pos = torch.tensor(
            [obj.body_offset_pos for obj in operator_cfgs], dtype=torch.float32, device=self.device
        )
        self._operator_offsets_quat = torch.tensor(
            [obj.body_offset_rot for obj in operator_cfgs], dtype=torch.float32, device=self.device
        )
        self._operator_hold_targets_b = torch.zeros(self.num_envs, len(self._operator_drivers), 7, device=self.device)
        self._operator_source_references_b = torch.zeros_like(self._operator_hold_targets_b)
        self._operator_target_references_b = torch.zeros_like(self._operator_hold_targets_b)
        self._operator_tracking_valid = torch.zeros(
            self.num_envs, len(self._operator_drivers), dtype=torch.bool, device=self.device
        )
        self._hold_targets_b = torch.zeros(self.num_envs, len(self._hold_drivers), 7, device=self.device)
        self._holds_captured = False

    @property
    def action_dim(self) -> int:
        return self._teleop_action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._teleop_raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._teleop_processed_actions

    def _capture_hold_targets(self) -> None:
        body_pos_w = self._asset.data.body_pos_w.torch
        body_quat_w = self._asset.data.body_quat_w.torch
        root_pos_w = self._asset.data.root_pos_w.torch
        root_quat_w = self._asset.data.root_quat_w.torch
        for index, driver in enumerate(self._operator_drivers):
            pos_b, quat_b = math_utils.subtract_frame_transforms(
                root_pos_w, root_quat_w, body_pos_w[:, driver.body_idx], body_quat_w[:, driver.body_idx]
            )
            pos_b, quat_b = math_utils.combine_frame_transforms(
                pos_b,
                quat_b,
                self._operator_offsets_pos[index].expand(self.num_envs, -1),
                self._operator_offsets_quat[index].expand(self.num_envs, -1),
            )
            self._operator_hold_targets_b[:, index, :3] = pos_b
            self._operator_hold_targets_b[:, index, 3:] = quat_b
            offset = self._operator_drivers[index].action_offset
            not_tracking = ~self._operator_tracking_valid[:, index]
            self._teleop_processed_actions[not_tracking, offset : offset + 7] = self._operator_hold_targets_b[
                not_tracking, index
            ]
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

        self._teleop_raw_actions[:] = actions
        for index, driver in enumerate(self._operator_drivers):
            offset = driver.action_offset
            command = self._teleop_raw_actions[:, offset : offset + 7]
            current_target = self._teleop_processed_actions[:, offset : offset + 7]
            (
                updated_target,
                self._operator_source_references_b[:, index],
                self._operator_target_references_b[:, index],
                self._operator_tracking_valid[:, index],
            ) = _update_clutched_absolute_pose(
                command,
                current_target,
                self._operator_source_references_b[:, index],
                self._operator_target_references_b[:, index],
                self._operator_tracking_valid[:, index],
            )
            self._teleop_processed_actions[:, offset : offset + 7] = updated_target
        self._full_actions.zero_()
        self._full_actions[:, : self._teleop_action_dim] = self._teleop_processed_actions
        for index, driver in enumerate(self._hold_drivers):
            offset = driver.action_offset
            self._full_actions[:, offset : offset + 7] = self._hold_targets_b[:, index]
        NewtonInverseKinematicsAction.process_actions(self, self._full_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        NewtonInverseKinematicsAction.reset(self, env_ids)
        self._teleop_raw_actions[env_ids] = 0.0
        self._teleop_processed_actions[env_ids] = 0.0
        self._operator_source_references_b[env_ids] = 0.0
        self._operator_target_references_b[env_ids] = 0.0
        self._operator_tracking_valid[env_ids] = False
        self._holds_captured = False


def _write_direct_bimanual_ik_actions(
    full_actions: torch.Tensor,
    wrist_targets: torch.Tensor,
    hold_targets: torch.Tensor,
    hold_action_offsets: Sequence[int],
) -> torch.Tensor:
    """Pack direct wrist targets and fixed-body holds into the full Newton IK action."""

    full_actions.zero_()
    full_actions[:, : wrist_targets.shape[-1]] = wrist_targets
    for index, action_offset in enumerate(hold_action_offsets):
        full_actions[:, action_offset : action_offset + 7] = hold_targets[:, index]
    return full_actions


def _capture_pending_hold_targets_b(
    hold_targets_b: torch.Tensor,
    holds_captured: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    hold_body_indices: Sequence[int],
) -> None:
    """Capture root-relative hold poses only for environments marked pending."""

    pending = ~holds_captured
    for hold_index, body_index in enumerate(hold_body_indices):
        pos_b, quat_b = math_utils.subtract_frame_transforms(
            root_pos_w,
            root_quat_w,
            body_pos_w[:, body_index],
            body_quat_w[:, body_index],
        )
        captured_pose_b = torch.cat((pos_b, quat_b), dim=-1)
        hold_targets_b[:, hold_index] = torch.where(
            pending.unsqueeze(-1),
            captured_pose_b,
            hold_targets_b[:, hold_index],
        )
    holds_captured.fill_(True)


class WaterhoseDirectBimanualNewtonIkAction(NewtonInverseKinematicsAction):
    """Drive two robot-side wrist targets directly while holding the RBY1 torso fixed."""

    _WRIST_OBJECTIVE_NAMES = ("right_ee", "left_ee")
    _HOLD_OBJECTIVE_NAMES = ("torso_hold",)

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        driver_names = tuple(driver.objective.name for driver in self._drivers)
        expected_names = self._WRIST_OBJECTIVE_NAMES + self._HOLD_OBJECTIVE_NAMES
        if driver_names != expected_names:
            raise ValueError(
                f"WaterhoseDirectBimanualNewtonIkAction expects pose objectives {expected_names}, got {driver_names}."
            )

        wrist_driver_count = len(self._WRIST_OBJECTIVE_NAMES)
        self._wrist_drivers = self._drivers[:wrist_driver_count]
        self._hold_drivers = self._drivers[wrist_driver_count:]
        self._direct_action_dim = sum(driver.objective.action_dim for driver in self._wrist_drivers)
        if self._direct_action_dim != 14:
            raise ValueError(f"Expected two seven-dimensional wrist targets, got {self._direct_action_dim} values.")

        self._direct_raw_actions = torch.zeros(self.num_envs, self._direct_action_dim, device=self.device)
        self._direct_processed_actions = torch.zeros_like(self._direct_raw_actions)
        self._full_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._hold_targets_b = torch.zeros(self.num_envs, len(self._hold_drivers), 7, device=self.device)
        self._hold_action_offsets = tuple(driver.action_offset for driver in self._hold_drivers)
        self._hold_body_indices = tuple(driver.body_idx for driver in self._hold_drivers)
        self._holds_captured = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hold_capture_pending = True

    @property
    def action_dim(self) -> int:
        return self._direct_action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._direct_raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._direct_processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """Describe the two external wrist targets without the internal torso hold."""

        descriptor = super().IO_descriptor
        descriptor.shape = (self._direct_action_dim,)
        descriptor.action_type = "WaterhoseDirectBimanualNewtonIkAction"
        descriptor.extras["objective_names"] = [driver.objective.name for driver in self._wrist_drivers]
        descriptor.extras["coordinate_names"] = [
            f"{driver.objective.name}/{coordinate}"
            for driver in self._wrist_drivers
            for coordinate in driver.objective.command_coordinate_names()
        ]
        return descriptor

    def _capture_hold_targets(self) -> None:
        _capture_pending_hold_targets_b(
            self._hold_targets_b,
            self._holds_captured,
            self._asset.data.root_pos_w.torch,
            self._asset.data.root_quat_w.torch,
            self._asset.data.body_pos_w.torch,
            self._asset.data.body_quat_w.torch,
            self._hold_body_indices,
        )
        self._hold_capture_pending = False

    def process_actions(self, actions: torch.Tensor) -> None:
        if self._hold_capture_pending:
            self._capture_hold_targets()

        self._direct_raw_actions[:] = actions
        self._direct_processed_actions[:] = self._direct_raw_actions
        _write_direct_bimanual_ik_actions(
            self._full_actions,
            self._direct_processed_actions,
            self._hold_targets_b,
            self._hold_action_offsets,
        )
        NewtonInverseKinematicsAction.process_actions(self, self._full_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        NewtonInverseKinematicsAction.reset(self, env_ids)
        self._direct_raw_actions[env_ids] = 0.0
        self._direct_processed_actions[env_ids] = 0.0
        self._holds_captured[env_ids] = False
        self._hold_capture_pending = True
