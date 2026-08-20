# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-aware arm action for the gravity-loaded RJ45 grasp."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.franka_pour.mdp.actions import EMARelativeJointPositionAction
from isaaclab_tasks.contrib.franka_pour.mdp.actions_cfg import EMARelativeJointPositionActionCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ResetTargetEMARelativeJointPositionAction(EMARelativeJointPositionAction):
    """Apply policy deltas around a reset-specific gravity-compensated target.

    Newton's position actuators settle at a measured configuration that differs
    slightly from their target under gravity.  A plain measured-state relative
    action discards that preload on the first zero-action policy step.  This
    term retains the demonstrated target-minus-state bias, filters the policy
    delta with the shared Franka EMA implementation, and holds one target per
    policy step as in the reset-driven stack controller.
    """

    def __init__(self, cfg: ResetTargetEMARelativeJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        if cfg.max_delta <= 0.0:
            raise ValueError("max_delta must be positive.")
        if cfg.joint_limit_margin < 0.0:
            raise ValueError("joint_limit_margin must be non-negative.")
        self._reset_target_bias = torch.zeros_like(self._processed_actions)
        self._position_targets = self._asset.data.joint_pos.torch[:, self._joint_ids].clone()
        resolved_joint_ids = (
            list(range(self._asset.num_joints)) if isinstance(self._joint_ids, slice) else self._joint_ids
        )
        self._gravity_joint_ids = [joint_id + self._asset.num_base_dofs for joint_id in resolved_joint_ids]

    @property
    def reset_target_bias(self) -> torch.Tensor:
        """Per-environment target-minus-state bias restored from the reset row."""
        return self._reset_target_bias

    def set_reset_target(
        self,
        target: torch.Tensor,
        measured_position: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Restore the actuator preload associated with a measured reset state."""
        selected = slice(None) if env_ids is None else env_ids
        expected_shape = self._reset_target_bias[selected].shape
        if target.shape != expected_shape or measured_position.shape != expected_shape:
            raise ValueError(
                "Reset arm target/state shapes must both match "
                f"{tuple(expected_shape)}, got {tuple(target.shape)}/{tuple(measured_position.shape)}."
            )
        target = target.to(device=self.device, dtype=self._reset_target_bias.dtype)
        measured_position = measured_position.to(device=self.device, dtype=self._reset_target_bias.dtype)
        self._reset_target_bias[selected] = target - measured_position
        self._position_targets[selected] = target

    def process_actions(self, actions: torch.Tensor) -> None:
        """Create one reset-biased target and hold it through all physics substeps."""
        super().process_actions(actions)
        target_delta = torch.clamp(self._processed_actions, -self.cfg.max_delta, self.cfg.max_delta)
        current_position = self._asset.data.joint_pos.torch[:, self._joint_ids]
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        lower = limits[..., 0] + self.cfg.joint_limit_margin
        upper = limits[..., 1] - self.cfg.joint_limit_margin
        self._position_targets = torch.clamp(
            current_position + self._reset_target_bias + target_delta,
            min=lower,
            max=upper,
        )

    def apply_actions(self) -> None:
        """Apply the fixed position target and stack-style gravity feedforward."""
        self._asset.set_joint_position_target_index(target=self._position_targets, joint_ids=self._joint_ids)
        if self.cfg.gravity_compensation:
            gravity = self._asset.data.gravity_compensation_forces.torch[:, self._gravity_joint_ids]
            gravity = torch.where(torch.isfinite(gravity), gravity, torch.zeros_like(gravity))
            self._asset.set_joint_effort_target_index(target=gravity, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Clear policy history while retaining the reset-specific actuator target."""
        super().reset(env_ids)
        selected = slice(None) if env_ids is None else env_ids
        current_position = self._asset.data.joint_pos.torch[:, self._joint_ids]
        self._position_targets[selected] = current_position[selected] + self._reset_target_bias[selected]


class PersistentResetTargetEMAJointPositionAction(EMARelativeJointPositionAction):
    """Integrate policy deltas into a reset-authored absolute joint target.

    Unlike :class:`ResetTargetEMARelativeJointPositionAction`, the commanded
    position is never re-anchored to the measured joint position.  A reset row
    supplies the absolute actuator target, policy deltas move that target once
    per control step, and an exactly zero policy component clears its EMA tail
    and preserves the corresponding target bit-for-bit.  This retains the
    position-drive restoring stiffness required by a gravity-loaded cable
    grasp while preserving velocity-like residual actions for long motions.
    """

    cfg: PersistentResetTargetEMAJointPositionActionCfg

    def __init__(self, cfg: PersistentResetTargetEMAJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        if cfg.max_delta <= 0.0:
            raise ValueError("max_delta must be positive.")
        if cfg.joint_limit_margin < 0.0:
            raise ValueError("joint_limit_margin must be non-negative.")
        if cfg.gravity_compensation:
            raise ValueError(
                "Persistent RJ45 control requires robot-native MJWarp gravity compensation; "
                "action-level inverse-dynamics gravity compensation must remain disabled."
            )
        tracking_limits = torch.as_tensor(cfg.tracking_error_limits, device=self.device, dtype=torch.float32)
        if tracking_limits.shape != (self.action_dim,):
            raise ValueError(f"tracking_error_limits must contain one value per controlled joint ({self.action_dim}).")
        if not bool(torch.isfinite(tracking_limits).all()) or not bool((tracking_limits > 0.0).all()):
            raise ValueError("tracking_error_limits must contain finite positive values.")
        self._tracking_error_limits = tracking_limits
        self._position_targets = self._asset.data.joint_pos.torch[:, self._joint_ids].clone()
        self._target_clamp_delta = torch.zeros_like(self._position_targets)

    @property
    def position_targets(self) -> torch.Tensor:
        """Persistent absolute arm position targets [rad]."""
        return self._position_targets

    @property
    def target_tracking_error(self) -> torch.Tensor:
        """Absolute target-minus-measured joint error [rad]."""
        position = self._asset.data.joint_pos.torch[:, self._joint_ids]
        return self._position_targets - position

    @property
    def tracking_error_limits(self) -> torch.Tensor:
        """Per-joint target tracking-error envelope [rad]."""
        return self._tracking_error_limits

    @property
    def tracking_error_violation(self) -> torch.Tensor:
        """Whether any joint is outside its configured tracking envelope."""
        return torch.abs(self.target_tracking_error) > self._tracking_error_limits

    @property
    def target_clamp_delta(self) -> torch.Tensor:
        """Magnitude removed from the latest proposed target by safety clamps [rad]."""
        return self._target_clamp_delta

    def set_reset_target(
        self,
        target: torch.Tensor,
        measured_position: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Restore the reset row's absolute target without deriving a relative bias."""
        selected = slice(None) if env_ids is None else env_ids
        expected_shape = self._position_targets[selected].shape
        if target.shape != expected_shape or measured_position.shape != expected_shape:
            raise ValueError(
                "Reset arm target/state shapes must both match "
                f"{tuple(expected_shape)}, got {tuple(target.shape)}/{tuple(measured_position.shape)}."
            )
        target = target.to(device=self.device, dtype=self._position_targets.dtype)
        measured_position = measured_position.to(device=self.device, dtype=self._position_targets.dtype)
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids][selected]
        lower = limits[..., 0] + self.cfg.joint_limit_margin
        upper = limits[..., 1] - self.cfg.joint_limit_margin
        if bool(torch.any(~torch.isfinite(target))) or bool(torch.any((target < lower) | (target > upper))):
            raise ValueError("Reset arm target must be finite and inside the configured soft-limit margin.")
        if bool(torch.any(torch.abs(target - measured_position) > self._tracking_error_limits)):
            raise ValueError("Reset arm target exceeds the configured target-tracking envelope.")
        self._position_targets[selected] = target
        self._target_clamp_delta[selected] = 0.0

    def process_actions(self, actions: torch.Tensor) -> None:
        """Accumulate one bounded policy delta into the absolute target."""
        super().process_actions(actions)

        # A literal zero is the reset/evaluation hold command.  Clearing both
        # the emitted delta and the EMA history makes this invariant exact even
        # immediately after motion instead of integrating a decaying tail.
        zero_command = actions == 0.0
        self._processed_actions.masked_fill_(zero_command, 0.0)
        self._previous_delta.masked_fill_(zero_command, 0.0)
        target_delta = torch.clamp(self._processed_actions, -self.cfg.max_delta, self.cfg.max_delta)

        position = self._asset.data.joint_pos.torch[:, self._joint_ids]
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        lower = torch.maximum(
            limits[..., 0] + self.cfg.joint_limit_margin,
            position - self._tracking_error_limits,
        )
        upper = torch.minimum(
            limits[..., 1] - self.cfg.joint_limit_margin,
            position + self._tracking_error_limits,
        )
        proposed = self._position_targets + target_delta
        bounded = torch.clamp(proposed, min=lower, max=upper)
        next_target = torch.where(zero_command, self._position_targets, bounded)
        self._target_clamp_delta.copy_(torch.where(zero_command, 0.0, torch.abs(proposed - bounded)))
        self._position_targets.copy_(next_target)

    def apply_actions(self) -> None:
        """Hold the persistent absolute target through every physics substep."""
        self._asset.set_joint_position_target_index(target=self._position_targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Clear policy/EMA history while retaining the event-restored target."""
        super().reset(env_ids)
        selected = slice(None) if env_ids is None else env_ids
        self._target_clamp_delta[selected] = 0.0


@configclass
class ResetTargetEMARelativeJointPositionActionCfg(EMARelativeJointPositionActionCfg):
    """Configuration for :class:`ResetTargetEMARelativeJointPositionAction`."""

    class_type: type[ResetTargetEMARelativeJointPositionAction] = ResetTargetEMARelativeJointPositionAction
    max_delta: float = 0.025
    """Maximum target change from measured state per policy step [rad]."""

    joint_limit_margin: float = 0.02
    """Distance retained from each soft joint limit [rad]."""

    gravity_compensation: bool = True
    """Add the model's configuration-dependent gravity effort feedforward."""


@configclass
class PersistentResetTargetEMAJointPositionActionCfg(ResetTargetEMARelativeJointPositionActionCfg):
    """Configuration for :class:`PersistentResetTargetEMAJointPositionAction`."""

    class_type: type[PersistentResetTargetEMAJointPositionAction] = PersistentResetTargetEMAJointPositionAction
    gravity_compensation: bool = False
    """Never invoke model-wide RNEA; the pick robot uses native MJWarp compensation."""

    tracking_error_limits: tuple[float, ...] = (0.145, 0.145, 0.145, 0.145, 0.048, 0.080, 0.240)
    """Per-joint target-minus-measured envelope [rad], derived from effort limit / stiffness."""


__all__ = [
    "PersistentResetTargetEMAJointPositionAction",
    "PersistentResetTargetEMAJointPositionActionCfg",
    "ResetTargetEMARelativeJointPositionAction",
    "ResetTargetEMARelativeJointPositionActionCfg",
]
