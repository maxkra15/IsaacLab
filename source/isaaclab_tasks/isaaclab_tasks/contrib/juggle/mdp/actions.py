# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Juggle-specific compact and reset-preload action specializations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.stack.mdp.actions import (
    ResetPreservingRelativeJointPositionAction,
    WorkspaceBoundedRelativeJointPositionAction,
)
from isaaclab_tasks.contrib.stack.mdp.actions_cfg import (
    ResetPreservingRelativeJointPositionActionCfg,
    WorkspaceBoundedRelativeJointPositionActionCfg,
)
from isaaclab_tasks.contrib.stack.mdp.kuka_allegro_reset import kuka_allegro_tool_pose

from .runtime import get_juggle_runtime_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_KUKA_ARM_DOF = 7
_TASK_SPACE_DIM = 3
_TASK_SPACE_JACOBIAN_EPSILON = 1.0e-4


def normalized_kuka_allegro_translation_joint_action(
    joint_positions: torch.Tensor,
    translations: torch.Tensor,
    *,
    tool_offset: Sequence[float] | torch.Tensor,
    damping: float,
) -> torch.Tensor:
    """Map XYZ commands to normalized KUKA joint commands with translational DLS.

    The palm-fixed tool Jacobian is evaluated from the same analytic KUKA FK as
    the reset catalog.  Each DLS result is normalized independently so its
    maximum absolute joint command equals the maximum absolute XYZ input.  The
    mapping therefore preserves the policy-owned magnitude without embedding a
    preferred direction, pose, or timing sequence.

    Args:
        joint_positions: KUKA arm joint positions, shape ``(..., 7)`` [rad].
        translations: Environment-local translation commands, shape ``(..., 3)``.
        tool_offset: Controlled point in the palm-link frame [m].
        damping: Positive DLS regularization added to ``J J^T``.

    Returns:
        Normalized KUKA joint commands, shape ``(..., 7)``.
    """
    if joint_positions.ndim < 1 or joint_positions.shape[-1] != _KUKA_ARM_DOF:
        raise ValueError("KUKA arm joint_positions must have shape (..., 7).")
    if translations.ndim < 1 or translations.shape[-1] != _TASK_SPACE_DIM:
        raise ValueError("Task-space translations must have shape (..., 3).")
    if joint_positions.shape[:-1] != translations.shape[:-1]:
        raise ValueError("KUKA joint positions and translations must have matching batch dimensions.")
    if not torch.is_floating_point(joint_positions) or not torch.is_floating_point(translations):
        raise TypeError("KUKA joint positions and translations must be floating-point tensors.")
    if joint_positions.device != translations.device:
        raise ValueError("KUKA joint positions and translations must use the same device.")
    if isinstance(damping, bool) or not isinstance(damping, Real) or not math.isfinite(damping) or damping <= 0.0:
        raise ValueError("Task-space DLS damping must be finite and positive.")

    if isinstance(tool_offset, torch.Tensor):
        if tool_offset.shape != (_TASK_SPACE_DIM,) or not bool(torch.all(torch.isfinite(tool_offset))):
            raise ValueError("Task-space tool_offset must contain three finite numeric values.")
    elif (
        not isinstance(tool_offset, Sequence)
        or isinstance(tool_offset, (str, bytes))
        or len(tool_offset) != _TASK_SPACE_DIM
        or any(isinstance(value, bool) or not isinstance(value, Real) for value in tool_offset)
        or not all(math.isfinite(value) for value in tool_offset)
    ):
        raise ValueError("Task-space tool_offset must contain three finite numeric values.")
    offset = torch.as_tensor(tool_offset, dtype=torch.float64, device=joint_positions.device)

    # Float64 keeps the 0.1-mm central difference independent of TF32 policy
    # settings.  Leading dimensions remain batched, including thousands of
    # parallel environments.
    joint_positions_64 = joint_positions.to(dtype=torch.float64)
    translations_64 = translations.to(dtype=torch.float64)
    joint_offsets = (
        torch.eye(_KUKA_ARM_DOF, dtype=torch.float64, device=joint_positions.device) * _TASK_SPACE_JACOBIAN_EPSILON
    )
    positions_after, _ = kuka_allegro_tool_pose(joint_positions_64.unsqueeze(-2) + joint_offsets, offset)
    positions_before, _ = kuka_allegro_tool_pose(joint_positions_64.unsqueeze(-2) - joint_offsets, offset)
    jacobian = ((positions_after - positions_before) / (2.0 * _TASK_SPACE_JACOBIAN_EPSILON)).transpose(-1, -2)

    task_metric = torch.matmul(jacobian, jacobian.transpose(-1, -2))
    task_metric = task_metric + float(damping) * torch.eye(
        _TASK_SPACE_DIM,
        dtype=torch.float64,
        device=joint_positions.device,
    )
    joint_solution = torch.matmul(
        jacobian.transpose(-1, -2),
        torch.linalg.solve(task_metric, translations_64.unsqueeze(-1)),
    ).squeeze(-1)

    translation_magnitude = torch.amax(torch.abs(translations_64), dim=-1, keepdim=True)
    solution_magnitude = torch.amax(torch.abs(joint_solution), dim=-1, keepdim=True)
    normalization = torch.where(
        solution_magnitude > torch.finfo(torch.float64).eps,
        translation_magnitude / solution_magnitude.clamp_min(torch.finfo(torch.float64).eps),
        torch.zeros_like(solution_magnitude),
    )
    return (joint_solution * normalization).to(dtype=joint_positions.dtype)


class JuggleTaskSpaceTranslationAction(WorkspaceBoundedRelativeJointPositionAction):
    """Control seven KUKA joints through one normalized XYZ translation.

    Raw XYZ commands are converted at the measured pose with a translational
    damped-least-squares Jacobian, then passed to the inherited measured-state,
    workspace-bounded relative target.  Zero maps exactly to a held joint pose.
    """

    cfg: JuggleTaskSpaceTranslationActionCfg

    def __init__(self, cfg: JuggleTaskSpaceTranslationActionCfg, env: ManagerBasedEnv) -> None:
        # The inherited joint action must allocate seven-dimensional target and
        # workspace tensors before this term exposes its three policy inputs.
        self._initializing_full_joint_action = True
        super().__init__(cfg, env)
        self._initializing_full_joint_action = False

        expected_joint_names = tuple(f"iiwa7_joint_{joint_id}" for joint_id in range(1, _KUKA_ARM_DOF + 1))
        if self._num_joints != _KUKA_ARM_DOF or tuple(self._joint_names) != expected_joint_names:
            raise ValueError(
                "Task-space translation requires all seven KUKA iiwa joints in kinematic order; "
                f"resolved {tuple(self._joint_names)}."
            )
        if isinstance(cfg.scale, bool) or not isinstance(cfg.scale, Real) or not math.isfinite(cfg.scale):
            raise ValueError("Task-space translation requires a finite positive scalar scale.")
        if cfg.scale <= 0.0:
            raise ValueError("Task-space translation requires a finite positive scalar scale.")
        if isinstance(cfg.offset, bool) or not isinstance(cfg.offset, Real) or float(cfg.offset) != 0.0:
            raise ValueError("Task-space translation requires a zero scalar offset so zero holds.")
        if (
            isinstance(cfg.max_delta, bool)
            or not isinstance(cfg.max_delta, Real)
            or not math.isfinite(cfg.max_delta)
            or cfg.max_delta <= 0.0
        ):
            raise ValueError("Task-space translation requires a finite positive max_delta.")
        if not isinstance(cfg.body_name, str) or not cfg.body_name:
            raise ValueError("Task-space translation requires a non-empty palm body_name.")
        body_ids, body_names = self._asset.find_bodies(cfg.body_name)
        if len(body_ids) != 1 or body_names != ["palm_link"]:
            raise ValueError(
                "KUKA-Allegro analytic task-space translation requires body_name to resolve exactly palm_link; "
                f"resolved {body_names}."
            )
        if (
            not isinstance(cfg.tool_offset, Sequence)
            or isinstance(cfg.tool_offset, (str, bytes))
            or len(cfg.tool_offset) != _TASK_SPACE_DIM
            or any(isinstance(value, bool) or not isinstance(value, Real) for value in cfg.tool_offset)
            or not all(math.isfinite(value) for value in cfg.tool_offset)
        ):
            raise ValueError("Task-space tool_offset must contain three finite numeric values.")
        if (
            isinstance(cfg.damping, bool)
            or not isinstance(cfg.damping, Real)
            or not math.isfinite(cfg.damping)
            or cfg.damping <= 0.0
        ):
            raise ValueError("Task-space DLS damping must be finite and positive.")

        self._tool_offset = tuple(float(value) for value in cfg.tool_offset)
        self._damping = float(cfg.damping)
        self._raw_actions = torch.zeros(self.num_envs, _TASK_SPACE_DIM, device=self.device)

    @property
    def action_dim(self) -> int:
        """Return three after the inherited joint-space initialization."""
        if getattr(self, "_initializing_full_joint_action", False):
            return super().action_dim
        return _TASK_SPACE_DIM

    def process_actions(self, actions: torch.Tensor) -> None:
        """Convert XYZ inputs to normalized joint residuals and bound targets."""
        if actions.shape != (self.num_envs, _TASK_SPACE_DIM):
            raise ValueError(
                f"Expected task-space translation actions with shape {(self.num_envs, _TASK_SPACE_DIM)}, "
                f"got {actions.shape}."
            )
        self._raw_actions[:] = actions
        current_position = self._asset.data.joint_pos.torch[:, self._joint_ids]
        joint_actions = normalized_kuka_allegro_translation_joint_action(
            current_position,
            actions,
            tool_offset=self._tool_offset,
            damping=self._damping,
        )

        # Temporarily provide the inherited controller its seven joint inputs;
        # preserve only XYZ on the public action/observation interface.
        raw_actions = self._raw_actions
        self._raw_actions = joint_actions
        try:
            super().process_actions(joint_actions)
        finally:
            self._raw_actions = raw_actions
        self._raw_actions[:] = actions


class JuggleResetPreservingRelativeJointPositionAction(ResetPreservingRelativeJointPositionAction):
    """Anchor the sphere cradle until the policy deliberately opens the hand."""

    cfg: JuggleResetPreservingRelativeJointPositionActionCfg

    def _reset_grasp_pair_ids(self) -> torch.Tensor:
        """Return the one sphere-cradle calibration ID for every environment."""
        state = get_juggle_runtime_state(self._env)
        return torch.zeros_like(state.current_phases)

    def _reset_preload_active_mask(self) -> torch.Tensor:
        """Protect only reset rows explicitly authored with preload assistance."""
        return get_juggle_runtime_state(self._env).preload_assist_start

    def process_actions(self, actions: torch.Tensor) -> None:
        """Apply the reset anchor once, then optionally end its assistance."""
        super().process_actions(actions)
        if self.cfg.release_preload_after_first_action:
            self._preload_assist_active.zero_()


class JuggleHandSynergyAction(JuggleResetPreservingRelativeJointPositionAction):
    """Control the full Allegro hand with one relative aperture command.

    The scalar command expands along the calibrated open-minus-preload pose
    direction before the inherited measured-state and reset-preload logic.
    Positive commands open, negative commands close, and zero holds the
    measured reset pose. This gives PPO one learnable release/catch coordinate
    while retaining all 16 physical finger joints.
    """

    cfg: JuggleHandSynergyActionCfg

    def __init__(self, cfg: JuggleHandSynergyActionCfg, env: ManagerBasedEnv) -> None:
        # JointAction constructs joint-shaped scale/offset tensors from
        # ``action_dim``. Let that initialization complete in the full joint
        # space, then expose the scalar policy interface.
        self._initializing_full_joint_action = True
        super().__init__(cfg, env)
        self._initializing_full_joint_action = False

        if isinstance(cfg.scale, bool) or not isinstance(cfg.scale, (int, float)) or cfg.scale <= 0.0:
            raise ValueError("A hand synergy requires a positive scalar scale.")
        if len(cfg.joint_directions) != self._num_joints:
            raise ValueError(f"joint_directions must contain one value per controlled joint ({self._num_joints}).")
        directions = torch.tensor(cfg.joint_directions, dtype=torch.float32, device=self.device)
        if not torch.all(torch.isfinite(directions)) or torch.any(torch.abs(directions) > 1.0):
            raise ValueError("joint_directions must be finite values inside [-1, 1].")
        if torch.any(torch.abs(directions) < 1.0e-6):
            raise ValueError("Every hand-synergy joint direction must be nonzero.")
        self._joint_directions = directions
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def action_dim(self) -> int:
        """Return one after the inherited joint-space initialization."""
        if getattr(self, "_initializing_full_joint_action", False):
            return super().action_dim
        return 1

    def process_actions(self, actions: torch.Tensor) -> None:
        """Expand one aperture command into relative targets for all fingers."""
        if actions.shape != (self.num_envs, 1):
            raise ValueError(f"Expected hand-synergy actions with shape {(self.num_envs, 1)}, got {actions.shape}.")
        self._raw_actions[:] = actions
        joint_actions = actions * self._joint_directions.unsqueeze(0)

        # Temporarily restore the joint-shaped raw-action tensor expected by
        # the inherited relative controller. The policy-facing tensor remains
        # scalar before and after this call.
        raw_actions = self._raw_actions
        self._raw_actions = joint_actions
        try:
            super().process_actions(joint_actions)
        finally:
            self._raw_actions = raw_actions
        self._raw_actions[:] = actions


@configclass
class JuggleTaskSpaceTranslationActionCfg(WorkspaceBoundedRelativeJointPositionActionCfg):
    """Configuration for normalized KUKA-Allegro XYZ translation control."""

    class_type: type[JuggleTaskSpaceTranslationAction] | str = "{DIR}.actions:JuggleTaskSpaceTranslationAction"

    body_name: str = "palm_link"
    """Articulation body whose palm-fixed point is controlled."""

    tool_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Controlled point expressed in the palm-link frame [m]."""

    damping: float = 2.5e-3
    """DLS regularization added to the translational task metric."""


@configclass
class JuggleResetPreservingRelativeJointPositionActionCfg(ResetPreservingRelativeJointPositionActionCfg):
    """Configuration for the sphere-cradle preload handoff."""

    class_type: type[JuggleResetPreservingRelativeJointPositionAction] | str = (
        "{DIR}.actions:JuggleResetPreservingRelativeJointPositionAction"
    )

    release_preload_after_first_action: bool = False
    """Whether the reset preload anchor expires after the first policy action."""


@configclass
class JuggleHandSynergyActionCfg(JuggleResetPreservingRelativeJointPositionActionCfg):
    """Configuration for one-dimensional Allegro-hand aperture control."""

    class_type: type[JuggleHandSynergyAction] | str = "{DIR}.actions:JuggleHandSynergyAction"

    joint_directions: tuple[float, ...] = ()
    """Open-minus-preload joint direction, normalized to unit maximum magnitude."""
