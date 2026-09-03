# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical-palm Newton IK with absolute, dropout-safe XR targets."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab_newton.envs.mdp.actions.newton_ik_actions import NewtonInverseKinematicsAction
from isaaclab_newton.ik.newton_ik_objectives_cfg import NewtonIKPoseObjectiveCfg

import isaaclab.utils.math as math_utils


def absolute_pose_is_valid(commands: torch.Tensor, tolerance: float = 1.0e-6) -> torch.Tensor:
    """Return which ``xyz + xyzw`` commands contain a tracked pose."""
    finite = torch.all(torch.isfinite(commands), dim=-1)
    has_position = torch.any(torch.abs(commands[..., :3]) > tolerance, dim=-1)
    has_quaternion = torch.linalg.vector_norm(commands[..., 3:7], dim=-1) > tolerance
    return finite & has_position & has_quaternion


def update_absolute_pose_with_dropout(
    commands: torch.Tensor,
    current_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Follow valid absolute poses and hold the last target during tracking loss.

    Quaternion signs are selected to remain in the current target's hemisphere.
    This is representational continuity only; the commanded rotation is unchanged.
    """
    valid = absolute_pose_is_valid(commands)
    identity_quat = torch.zeros_like(commands[..., 3:7])
    identity_quat[..., 3] = 1.0
    safe_quat = torch.where(valid.unsqueeze(-1), commands[..., 3:7], identity_quat)
    command_quat = math_utils.normalize(safe_quat)
    opposite_hemisphere = torch.sum(command_quat * current_targets[..., 3:7], dim=-1, keepdim=True) < 0.0
    command_quat = torch.where(opposite_hemisphere, -command_quat, command_quat)
    absolute_targets = torch.cat((commands[..., :3], command_quat), dim=-1)
    return torch.where(valid.unsqueeze(-1), absolute_targets, current_targets), valid


class RizonSharpaTeleopNewtonIkAction(NewtonInverseKinematicsAction):
    """Drive Sharpa's canonical palm from AVP tracking with Newton IK."""

    _OBJECTIVE_NAME = "right_palm"

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        if len(self._drivers) != 1 or self._drivers[0].objective.name != self._OBJECTIVE_NAME:
            names = tuple(driver.objective.name for driver in self._drivers)
            raise ValueError(f"Expected one {self._OBJECTIVE_NAME!r} pose objective, got {names}.")
        pose_cfgs = [objective for objective in cfg.objectives if isinstance(objective, NewtonIKPoseObjectiveCfg)]
        if len(pose_cfgs) != 1 or self._action_dim != 7:
            raise ValueError("Rizon Sharpa teleoperation requires one absolute seven-dimensional pose objective.")

        self._teleop_raw_actions = torch.zeros(self.num_envs, 7, device=self.device)
        self._teleop_processed_actions = torch.zeros_like(self._teleop_raw_actions)
        self._target_captured = False
        self._offset_pos = torch.tensor(pose_cfgs[0].body_offset_pos, device=self.device, dtype=torch.float32)
        self._offset_quat = torch.tensor(pose_cfgs[0].body_offset_rot, device=self.device, dtype=torch.float32)

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._teleop_raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._teleop_processed_actions

    def _capture_live_palm_target(self) -> None:
        driver = self._drivers[0]
        pos_b, quat_b = math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w.torch,
            self._asset.data.root_quat_w.torch,
            self._asset.data.body_pos_w.torch[:, driver.body_idx],
            self._asset.data.body_quat_w.torch[:, driver.body_idx],
        )
        pos_b, quat_b = math_utils.combine_frame_transforms(
            pos_b,
            quat_b,
            self._offset_pos.expand(self.num_envs, -1),
            self._offset_quat.expand(self.num_envs, -1),
        )
        self._teleop_processed_actions[:, :3] = pos_b
        self._teleop_processed_actions[:, 3:] = quat_b
        self._target_captured = True

    def process_actions(self, actions: torch.Tensor) -> None:
        if not self._target_captured:
            self._capture_live_palm_target()
        self._teleop_raw_actions[:] = actions
        targets, _ = update_absolute_pose_with_dropout(
            self._teleop_raw_actions,
            self._teleop_processed_actions,
        )
        self._teleop_processed_actions.copy_(targets)
        NewtonInverseKinematicsAction.process_actions(self, self._teleop_processed_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        NewtonInverseKinematicsAction.reset(self, ids)
        self._teleop_raw_actions[ids] = 0.0
        self._teleop_processed_actions[ids] = 0.0
        self._target_captured = False


__all__ = [
    "RizonSharpaTeleopNewtonIkAction",
    "absolute_pose_is_valid",
    "update_absolute_pose_with_dropout",
]
