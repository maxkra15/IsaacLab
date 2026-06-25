# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleoperation devices for the waterhose manipulation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.devices.device_base import DeviceBase, DeviceCfg
from isaaclab.devices.spacemouse import Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from collections.abc import Callable


@configclass
class WaterhoseSpaceMouseCfg(DeviceCfg):
    """SpaceMouse mapping tuned for hose grasping and insertion."""

    gripper_term: bool = True
    """Whether the device emits a trailing gripper command element."""

    pos_sensitivity: float = 0.05
    """Translation scale, end-effector metres per unit puck deflection [m]."""

    rot_sensitivity: float = 0.15
    """Rotation scale, end-effector radians per unit puck twist [rad]."""

    translation_signs: tuple[float, float, float] = (-1.0, -1.0, 1.0)
    """Per-axis sign applied to the (x, y, z) translation deltas to match the task frame."""

    twist_sign: float = -1.0
    """Sign applied to the retained cap-twist (gripper-roll) channel."""

    deadzone: float = 1.0e-3
    """Per-axis magnitude below which a raw command component is zeroed [m or rad]."""

    twist_deadzone: float = 1.0e-2
    """Twist magnitude below which cap-twist cross-talk during translation is rejected [rad]."""

    retargeters: None = None
    class_type: type[WaterhoseSpaceMouse] | str = "{DIR}.teleop:WaterhoseSpaceMouse"


class WaterhoseSpaceMouse(DeviceBase):
    """Waterhose-specific SpaceMouse wrapper.

    The stock SpaceMouse device exposes full 6-DoF deltas. For this hose task the useful manual control
    is translation plus a cap twist that rolls the gripper about its own approach axis (spinning the
    held plug to line its keying up with the bore); the wrist pitch and yaw make the plug hard to keep
    aligned, so they are suppressed. Translation and twist are independent: a deliberate twist of the
    cap always rolls the gripper, even while translating. ``twist_deadzone`` rejects the small twist
    cross-talk the cap reports during a translation push. The relative IK action applies this twist in
    the end-effector frame (see
    :class:`~isaaclab_tasks.contrib.waterhose.mdp.actions.WaterhoseLocalFrameNewtonInverseKinematicsAction`),
    so the roll is about the gripper's current approach axis.
    """

    def __init__(self, cfg: WaterhoseSpaceMouseCfg):
        super().__init__(retargeters=None)
        self._cfg = cfg
        self._device = Se3SpaceMouse(
            Se3SpaceMouseCfg(
                gripper_term=cfg.gripper_term,
                pos_sensitivity=cfg.pos_sensitivity,
                rot_sensitivity=cfg.rot_sensitivity,
                sim_device=cfg.sim_device,
            )
        )
        self._translation_signs = torch.tensor(cfg.translation_signs, dtype=torch.float32, device=cfg.sim_device)
        self._twist_sign = float(cfg.twist_sign)
        self._deadzone = max(0.0, float(cfg.deadzone))
        self._twist_deadzone = max(0.0, float(cfg.twist_deadzone))

    def __str__(self) -> str:
        return f"{self._device} (waterhose XYZ + gripper-roll mapping)"

    def reset(self) -> None:
        self._device.reset()

    def add_callback(self, key: str, func: Callable[[], None]) -> None:
        self._device.add_callback(key, func)

    def advance(self) -> torch.Tensor:
        command = self._device.advance().clone()
        # Reject small per-axis noise.
        if self._deadzone > 0.0:
            command[torch.abs(command) < self._deadzone] = 0.0

        # Translation: move the gripper in XYZ with the task's sign convention.
        command[:3] *= self._translation_signs

        # Rotation: keep only the cap twist and map it to a gripper roll (the relative IK action
        # applies this delta in the end-effector frame, about the gripper's approach axis). The wrist
        # pitch and yaw are suppressed.
        command[3:5] = 0.0
        twist = command[5] * self._twist_sign
        # A deliberate twist passes through; the small twist cross-talk reported during a translation
        # push is rejected, so the operator can translate and roll without one cancelling the other.
        command[5] = torch.where(torch.abs(twist) > self._twist_deadzone, twist, torch.zeros_like(twist))
        return command
