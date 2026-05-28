# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleoperation devices for the waterhose robot demo."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import torch

from isaaclab.devices.device_base import DeviceBase, DeviceCfg
from isaaclab.devices.spacemouse import Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from collections.abc import Callable


@configclass
class WaterhoseSpaceMouseCfg(DeviceCfg):
    """SpaceMouse mapping used by the waterhose demo."""

    gripper_term: bool = True
    pos_sensitivity: float = 0.05
    rot_sensitivity: float = 0.15
    translation_signs: tuple[float, float, float] = (-1.0, -1.0, 1.0)
    yaw_sign: float = -1.0
    deadzone: float = 1.0e-3
    yaw_translation_lock: bool = False
    retargeters: None = None
    class_type: type["WaterhoseSpaceMouse"] | str = "{DIR}.teleop:WaterhoseSpaceMouse"


class WaterhoseSpaceMouse(DeviceBase):
    """Restrict SpaceMouse control to XYZ translation plus gripper yaw/spin."""

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
        self._translation_signs = tuple(float(value) for value in cfg.translation_signs)
        self._yaw_sign = float(cfg.yaw_sign)
        self._deadzone = max(0.0, float(cfg.deadzone))
        self._yaw_translation_lock = bool(cfg.yaw_translation_lock)

    def __str__(self) -> str:
        return f"{self._device} (waterhose XYZ+yaw mode)"

    def reset(self) -> None:
        self._device.reset()

    def add_callback(self, key: str, func: "Callable[[], None]") -> None:
        self._device.add_callback(key, func)

    def advance(self) -> torch.Tensor:
        command = self._device.advance().clone()
        if self._deadzone > 0.0:
            command[torch.abs(command) < self._deadzone] = 0.0
        for axis, sign in enumerate(self._translation_signs):
            command[axis] *= sign
        command[3:5] = 0.0
        command[5] *= self._yaw_sign
        translation_norm = torch.linalg.vector_norm(command[:3])
        yaw_abs = torch.abs(command[5])
        translation_active = translation_norm > self._deadzone
        yaw_active = yaw_abs > self._deadzone
        command[5] = torch.where(translation_active, torch.zeros_like(command[5]), command[5])
        zero_translation = yaw_active & (translation_active.logical_not() | bool(self._yaw_translation_lock))
        command[:3] = torch.where(zero_translation, torch.zeros_like(command[:3]), command[:3])
        return command


def add_waterhose_spacemouse_args(parser: argparse.ArgumentParser) -> None:
    """Add demo-runner controls for the waterhose SpaceMouse mapping."""

    parser.add_argument("--spacemouse_pos_sensitivity", type=float, default=None)
    parser.add_argument("--spacemouse_rot_sensitivity", type=float, default=None)
    parser.add_argument("--spacemouse_simple_x_sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument("--spacemouse_simple_y_sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument("--spacemouse_simple_z_sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--spacemouse_simple_yaw_sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument("--spacemouse_simple_deadzone", type=float, default=1.0e-3)
    parser.add_argument(
        "--spacemouse_simple_yaw_translation_lock",
        action=argparse.BooleanOptionalAction,
        default=False,
    )


def create_waterhose_spacemouse_device(args_cli: argparse.Namespace, sensitivity: float) -> WaterhoseSpaceMouse:
    """Create the SpaceMouse device from demo-runner arguments."""

    pos_sensitivity = (
        float(args_cli.spacemouse_pos_sensitivity)
        if args_cli.spacemouse_pos_sensitivity is not None
        else 0.05 * sensitivity
    )
    rot_sensitivity = (
        float(args_cli.spacemouse_rot_sensitivity)
        if args_cli.spacemouse_rot_sensitivity is not None
        else 0.15 * sensitivity
    )
    cfg = WaterhoseSpaceMouseCfg(
        pos_sensitivity=pos_sensitivity,
        rot_sensitivity=rot_sensitivity,
        translation_signs=(
            float(args_cli.spacemouse_simple_x_sign),
            float(args_cli.spacemouse_simple_y_sign),
            float(args_cli.spacemouse_simple_z_sign),
        ),
        yaw_sign=float(args_cli.spacemouse_simple_yaw_sign),
        deadzone=float(args_cli.spacemouse_simple_deadzone),
        yaw_translation_lock=bool(args_cli.spacemouse_simple_yaw_translation_lock),
    )
    return WaterhoseSpaceMouse(cfg)
