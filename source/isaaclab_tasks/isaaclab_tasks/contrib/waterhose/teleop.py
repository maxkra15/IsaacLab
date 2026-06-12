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
    pos_sensitivity: float = 0.05
    rot_sensitivity: float = 0.15
    translation_signs: tuple[float, float, float] = (-1.0, -1.0, 1.0)
    yaw_sign: float = -1.0
    deadzone: float = 1.0e-3
    yaw_translation_lock: bool = False
    retargeters: None = None
    class_type: type[WaterhoseSpaceMouse] | str = "{DIR}.teleop:WaterhoseSpaceMouse"


class WaterhoseSpaceMouse(DeviceBase):
    """Waterhose-specific SpaceMouse wrapper.

    The stock SpaceMouse device exposes full 6-DoF deltas. For this hose task the useful manual control is
    translation plus twist around the gripper insertion axis; roll and pitch make the plug hard to keep aligned.
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
        self._yaw_sign = float(cfg.yaw_sign)
        self._deadzone = max(0.0, float(cfg.deadzone))
        self._yaw_translation_lock = bool(cfg.yaw_translation_lock)

    def __str__(self) -> str:
        return f"{self._device} (waterhose XYZ+yaw mapping)"

    def reset(self) -> None:
        self._device.reset()

    def add_callback(self, key: str, func: Callable[[], None]) -> None:
        self._device.add_callback(key, func)

    def advance(self) -> torch.Tensor:
        command = self._device.advance().clone()
        if self._deadzone > 0.0:
            command[torch.abs(command) < self._deadzone] = 0.0

        command[:3] *= self._translation_signs
        command[3:5] = 0.0
        command[5] *= self._yaw_sign

        translation_norm = torch.linalg.vector_norm(command[:3])
        yaw_abs = torch.abs(command[5])
        translation_active = translation_norm > self._deadzone
        yaw_active = yaw_abs > self._deadzone

        # Translation wins over yaw unless explicitly locked. This avoids accidental wrist spin from cap noise.
        command[5] = torch.where(translation_active, torch.zeros_like(command[5]), command[5])
        zero_translation = yaw_active & (translation_active.logical_not() | bool(self._yaw_translation_lock))
        command[:3] = torch.where(zero_translation, torch.zeros_like(command[:3]), command[:3])
        return command
