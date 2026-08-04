# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Filtered arm and continuous symmetric-gripper actions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.managers import ActionTerm

if TYPE_CHECKING:
    from .actions_cfg import (
        CurriculumGripperPositionActionCfg,
        EMARelativeJointPositionActionCfg,
    )

_GRIPPER_POSITION_TOLERANCE = 1.0e-6


class EMARelativeJointPositionAction(RelativeJointPositionAction):
    """Low-pass filter relative joint deltas without changing their policy-space meaning."""

    cfg: EMARelativeJointPositionActionCfg

    def __init__(self, cfg: EMARelativeJointPositionActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._alpha = float(cfg.alpha)
        if not 0.0 < self._alpha <= 1.0:
            raise ValueError(f"Moving-average weight must lie in (0, 1], got {self._alpha}.")
        self._previous_delta = torch.zeros_like(self._processed_actions)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Affine-map the raw action, then smooth only the commanded joint delta."""
        super().process_actions(actions)
        self._processed_actions.lerp_(self._previous_delta, 1.0 - self._alpha)
        self._previous_delta.copy_(self._processed_actions)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Clear selected command history so a reset pose receives exactly zero delta."""
        selected = slice(None) if env_ids is None else env_ids
        super().reset(selected)
        self._processed_actions[selected] = 0.0
        self._previous_delta[selected] = 0.0


class CurriculumGripperPositionAction(ActionTerm):
    """Filtered symmetric finger-position command with residual and binary modes."""

    cfg: CurriculumGripperPositionActionCfg

    def __init__(self, cfg: CurriculumGripperPositionActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        self._num_joints = len(self._joint_ids)
        if self._num_joints == 0:
            raise ValueError("CurriculumGripperPositionAction resolved no joints.")

        self._scale = float(cfg.scale)
        self._alpha = float(cfg.alpha)
        self._binary_threshold = cfg.binary_threshold
        self._close_position = float(cfg.close_position)
        self._neutral_position = float(cfg.neutral_position)
        self._default_position = self._close_position if cfg.default_position is None else float(cfg.default_position)
        self._contact_min_deflection = float(cfg.contact_min_deflection)
        if not math.isfinite(self._scale) or self._scale <= 0.0:
            raise ValueError("Curriculum gripper action scale must be finite and positive.")
        if not 0.0 < self._alpha <= 1.0:
            raise ValueError(f"Moving-average weight must lie in (0, 1], got {self._alpha}.")
        if self._binary_threshold is not None:
            if (
                isinstance(self._binary_threshold, bool)
                or not math.isfinite(self._binary_threshold)
                or not -1.0 < self._binary_threshold < 1.0
            ):
                raise ValueError("Binary gripper threshold must be finite and lie strictly between -1 and 1.")
        if (
            not math.isfinite(self._close_position)
            or not math.isfinite(self._neutral_position)
            or not self._close_position <= self._neutral_position
        ):
            raise ValueError("Curriculum gripper positions must be finite with close_position <= neutral_position.")
        if not math.isfinite(self._default_position) or not (
            self._close_position <= self._default_position <= self._neutral_position
        ):
            raise ValueError("default_position must lie in [close_position, neutral_position].")
        if not math.isfinite(self._contact_min_deflection) or self._contact_min_deflection <= 0.0:
            raise ValueError("contact_min_deflection must be finite and positive.")
        self._raw_actions = torch.zeros((self.num_envs, 1), device=self.device)
        self._action_offset = torch.full(
            (self.num_envs, 1),
            self._default_position,
            device=self.device,
        )
        self._processed_actions = self._action_offset.expand(-1, self._num_joints).clone()

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def action_offset(self) -> torch.Tensor:
        """Per-environment residual-mode target and initialization position [m]."""
        return self._action_offset

    @property
    def commanded_position(self) -> torch.Tensor:
        """Current symmetric per-finger position target [m]."""
        return self._processed_actions[:, :1]

    @property
    def contact_deflection(self) -> torch.Tensor:
        """Per-finger position-drive deflection caused by contact [m]."""
        joint_position = self._asset.data.joint_pos.torch[:, self._joint_ids]
        finite = torch.isfinite(joint_position) & torch.isfinite(self._processed_actions)
        return torch.where(
            finite,
            torch.clamp(joint_position - self._processed_actions, min=0.0),
            torch.zeros_like(joint_position),
        )

    @property
    def bilateral_contact(self) -> torch.Tensor:
        """Whether both fingers remain deflected against the commanded cup contact."""
        joint_position = self._asset.data.joint_pos.torch[:, self._joint_ids]
        deflection = self.contact_deflection
        finite = torch.isfinite(joint_position).all(dim=-1) & torch.isfinite(self._processed_actions).all(dim=-1)
        command_valid = self._processed_actions.amax(dim=-1) <= (self._neutral_position + _GRIPPER_POSITION_TOLERANCE)
        return finite & command_valid & (deflection.amin(dim=-1) >= self._contact_min_deflection)

    @property
    def contact_quality(self) -> torch.Tensor:
        """Smooth bilateral drive-deflection quality in ``[0, 1]``.

        Finger velocity is deliberately excluded.  It is useful when deciding whether a grasp has
        settled, but it is not a measure of whether the cup is physically between the fingers.
        Including it in dense shaping creates a large artificial potential drop while a restored
        contact relaxes during its first simulation step.
        """
        deflection = self.contact_deflection
        deflection_quality = torch.clamp(deflection.amin(dim=-1) / self._contact_min_deflection, 0.0, 1.0)
        command_valid = self._processed_actions.amax(dim=-1) <= (self._neutral_position + _GRIPPER_POSITION_TOLERANCE)
        return deflection_quality * command_valid.float()

    @property
    def IO_descriptor(self):
        descriptor = super().IO_descriptor
        descriptor.shape = (1,)
        descriptor.dtype = str(self.raw_actions.dtype)
        descriptor.action_type = "JointAction"
        descriptor.joint_names = self._joint_names
        descriptor.scale = self._scale
        return descriptor

    def set_action_offset(
        self,
        offset: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Set selected environments' residual-mode target and initialization position [m]."""
        selected = slice(None) if env_ids is None else env_ids
        expected_shape = self._action_offset[selected].shape
        if offset.shape != expected_shape:
            raise ValueError(f"Action offset shape {tuple(offset.shape)} does not match {tuple(expected_shape)}.")
        offset = offset.to(device=self._action_offset.device, dtype=self._action_offset.dtype)
        self._action_offset[selected] = offset
        self._processed_actions[selected] = offset.expand(-1, self._num_joints)

    def set_reset_position(
        self,
        position: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Align the filtered target with selected physical reset positions [m]."""
        selected = slice(None) if env_ids is None else env_ids
        expected_shape = self._action_offset[selected].shape
        if position.shape != expected_shape:
            raise ValueError(f"Reset-position shape {tuple(position.shape)} does not match {tuple(expected_shape)}.")
        position = position.to(device=self.device, dtype=self._processed_actions.dtype)
        expanded = position.expand(-1, self._num_joints)
        self._processed_actions[selected] = expanded

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions.copy_(actions)
        bounded_actions = torch.clamp(actions, -1.0, 1.0)
        if self._binary_threshold is None:
            target = torch.clamp(
                self._action_offset + self._scale * bounded_actions,
                min=self._close_position,
                max=self._neutral_position,
            )
        else:
            target = torch.where(
                bounded_actions < self._binary_threshold,
                torch.full_like(bounded_actions, self._close_position),
                torch.full_like(bounded_actions, self._neutral_position),
            )
        self._processed_actions.lerp_(target.expand(-1, self._num_joints), self._alpha)

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target_index(target=self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        self._raw_actions[selected] = 0.0
