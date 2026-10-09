# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured connector alignment and insertion from the task's physical pregrasp."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .contacts import GraspContacts, grasp_contact_failure

_GRIP_LOCAL_M = np.array((0.0, 0.0, 0.001))


class CableInstallStateMachine:
    """Align, insert, release and verify retention using ordinary robot targets.

    The reset supplies a housing pinch, not an attachment. Measured relative poses
    check slip and rotation; optional contact diagnostics additionally require a
    bilateral grasp. Connector and clip poses are never written.

    Args:
        tcp_pose_w: Initial TCP pose [m, XYZW].
        goal_pose_w: Seated connector pose [m, XYZW].
        dt: Controller interval [s].
        retreat_offset_w: World-frame displacement after release [m].
        grasp_penetration_limit: Maximum fingertip-pad/housing overlap [m].
        approach_angle_tolerance: Alignment tolerance before axial feed [rad].
    """

    def __init__(
        self,
        tcp_pose_w: np.ndarray,
        goal_pose_w: np.ndarray,
        dt: float,
        *,
        retreat_offset_w: tuple[float, float, float] = (0.0, -0.08, 0.0),
        grasp_penetration_limit: float = 0.003,
        approach_angle_tolerance: float = 0.02,
    ):
        for name, value in (
            ("dt", dt),
            ("grasp_penetration_limit", grasp_penetration_limit),
            ("approach_angle_tolerance", approach_angle_tolerance),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        self.retreat_offset_w = np.asarray(retreat_offset_w, dtype=np.float64).copy()
        if self.retreat_offset_w.shape != (3,) or not np.isfinite(self.retreat_offset_w).all():
            raise ValueError("retreat_offset_w must contain three finite world-frame displacements.")
        self.grasp_penetration_limit = grasp_penetration_limit
        self.approach_angle_tolerance = approach_angle_tolerance
        self.command = np.asarray(tcp_pose_w, dtype=np.float64).copy()
        self.goal = np.asarray(goal_pose_w, dtype=np.float64).copy()
        self.target = self.command.copy()
        self.dt = dt
        self.phase = "HOLD_GRASP"
        self.elapsed = self.dwell = self.width = self.max_penetration_m = self.contact_loss_s = 0.0
        self.grasp_rotation: Rotation | None = None
        self.grasp_offset_m: np.ndarray | None = None
        self.carry_target: np.ndarray | None = None
        self.failure: str | None = None

    def update(
        self,
        tcp_pose_w: np.ndarray,
        head_pose_w: np.ndarray,
        contacts: GraspContacts | None = None,
        *,
        latch_engaged: bool,
        hand_open: bool = False,
    ) -> tuple[np.ndarray, float]:
        """Return TCP pose [m, XYZW] and release opening [m], from 0 to 0.04.

        Without contact diagnostics, ``hand_open`` must confirm measured finger opening
        before release. Commanded opening alone is not sufficient.
        """
        if self.phase in ("DONE", "FAILED"):
            return self.command.copy(), self.width
        self.elapsed += self.dt
        tcp = np.asarray(tcp_pose_w, dtype=np.float64)
        head = np.asarray(head_pose_w, dtype=np.float64)
        grip = head[:3] + Rotation.from_quat(head[3:]).apply(_GRIP_LOCAL_M)
        if contacts is not None:
            self.max_penetration_m = max(self.max_penetration_m, contacts.penetration_m)
            failure = grasp_contact_failure(contacts, self.grasp_penetration_limit)
            if failure is not None:
                self._fail(failure)
                return self.command.copy(), self.width
        if self.elapsed > (30.0 if self.phase in ("RAISE", "ORIENT", "ALIGN", "INSERT") else 15.0):
            self._fail(f"{self.phase} timed out")
            return self.command.copy(), self.width

        if self.phase == "HOLD_GRASP":
            grasp_ready = contacts is None or (contacts.left and contacts.right)
            self.dwell = self.dwell + self.dt if grasp_ready else 0.0
            if self.elapsed > 1.0 and self.dwell > 0.5:
                rotation = Rotation.from_quat(tcp[3:]).inv()
                self.grasp_offset_m = rotation.apply(grip - tcp[:3])
                self.grasp_rotation = rotation * Rotation.from_quat(head[3:])
                self.carry_target = head.copy()
                self.carry_target[[0, 2]] = self.goal[[0, 2]]
                self._transition("RAISE")
        if self.phase in ("RAISE", "ORIENT", "ALIGN", "INSERT"):
            grip_relative = Rotation.from_quat(tcp[3:]).inv().apply(grip - tcp[:3])
            if contacts is not None:
                self.contact_loss_s = 0.0 if contacts.left and contacts.right else self.contact_loss_s + self.dt
            if np.linalg.norm(grip_relative - self.grasp_offset_m) > 0.008 or self.contact_loss_s > 0.25:
                self._fail("Connector grasp lost during transport")
                return self.command.copy(), self.width
            relative_rotation = Rotation.from_quat(tcp[3:]).inv() * Rotation.from_quat(head[3:])
            if (self.grasp_rotation.inv() * relative_rotation).magnitude() > 0.6:
                self._fail("Connector rolled between the fingers during transport")
                return self.command.copy(), self.width
            angular_error = self._update_transport_target(tcp, head)
            seat_position = self.goal[:3] if self.phase == "INSERT" else self.carry_target[:3]
            position_error = np.linalg.norm(head[:3] - seat_position)
            position_limit, angular_limit = {
                "ALIGN": (0.0002, self.approach_angle_tolerance),
                "INSERT": (0.0005, 0.01),
            }.get(self.phase, (0.003, 0.25 if self.phase == "RAISE" else 0.035))
            depth_ok = self.phase != "INSERT" or head[1] >= self.goal[1] - 0.0005
            latch_ok = self.phase != "INSERT" or latch_engaged
            self.dwell = (
                self.dwell + self.dt
                if position_error < position_limit and angular_error < angular_limit and depth_ok and latch_ok
                else 0.0
            )
            if self.dwell > 0.5:
                next_phase = {"RAISE": "ORIENT", "ORIENT": "ALIGN", "ALIGN": "INSERT", "INSERT": "RELEASE"}[self.phase]
                self._transition(next_phase)
                if next_phase == "ORIENT":
                    self.carry_target[3:] = self.goal[3:]
                elif next_phase == "ALIGN":
                    self.carry_target = self.goal.copy()
                    self.carry_target[1] -= 0.05
                elif next_phase == "INSERT":
                    # Small preload: the measured cavity has a rigid back.
                    self.carry_target = self.goal.copy()
                    self.carry_target[1] += 0.0002
                elif next_phase == "RELEASE":
                    # Stop insertion preload before opening beside the seated plug.
                    self.target = tcp.copy()
                    self.command = tcp.copy()
        if self.phase in ("RELEASE", "RETREAT"):
            self.width = min(0.04, self.width + 0.01 * self.dt)
        if self.phase == "RELEASE":
            released = hand_open if contacts is None else not contacts.left and not contacts.right
            if self.width >= 0.025 and released:
                self.target = tcp.copy()
                self.target[:3] += self.retreat_offset_w
                self._transition("RETREAT")
        if self.phase == "RETREAT" and self.elapsed > 5.0:
            seated_position = np.linalg.norm(head[:3] - self.goal[:3]) < 0.003
            seated_rotation = (
                Rotation.from_quat(head[3:]).inv() * Rotation.from_quat(self.goal[3:])
            ).magnitude() < 0.08
            if seated_position and seated_rotation and latch_engaged:
                self._transition("DONE")
            else:
                self._fail("Plug did not remain seated after release")

        self._advance_command(tcp, head)
        return self.command.copy(), self.width

    def _transition(self, phase: str) -> None:
        self.phase = phase
        self.elapsed = self.dwell = 0.0

    def _fail(self, reason: str) -> None:
        self.failure = reason
        self._transition("FAILED")

    def _update_transport_target(self, tcp: np.ndarray, head: np.ndarray) -> float:
        tcp_rotation = Rotation.from_quat(tcp[3:])
        head_rotation = Rotation.from_quat(head[3:])
        relative_rotation = tcp_rotation.inv() * head_rotation
        head_offset = tcp_rotation.inv().apply(head[:3] - tcp[:3])
        rotation = Rotation.from_quat(self.carry_target[3:]) * relative_rotation.inv()
        desired_head = self.carry_target[:3].copy()
        angular_error = (head_rotation.inv() * Rotation.from_quat(self.carry_target[3:])).magnitude()
        if self.phase == "INSERT":
            lateral_error = np.linalg.norm(head[[0, 2]] - self.goal[[0, 2]])
            if lateral_error > 0.0003 or angular_error > max(0.01, self.approach_angle_tolerance):
                desired_head[1] = head[1]
        self.target = np.r_[desired_head - rotation.apply(head_offset), rotation.as_quat()]
        return angular_error

    def _advance_command(self, tcp: np.ndarray, head: np.ndarray) -> None:
        desired = self.target.copy()
        position_lead = self.target[:3] - self.command[:3]
        close = np.abs(position_lead) < 0.02 if self.phase == "INSERT" else np.linalg.norm(position_lead) < 0.02
        desired[:3] = np.where(
            close, self.command[:3] + min(1.0, 4.0 * self.dt) * (self.target[:3] - tcp[:3]), desired[:3]
        )
        delta = desired[:3] - self.command[:3]
        if self.phase == "INSERT":
            remaining = self.goal[1] - head[1]
            axial_speed = np.interp(remaining, (0.015, 0.030), (0.006, 0.040))
            limits = np.array((0.01, axial_speed, 0.01)) * self.dt
            self.command[:3] += np.clip(delta, -limits, limits)
        else:
            self.command[:3] += delta * min(1.0, 0.04 * self.dt / max(np.linalg.norm(delta), 1e-9))
        current = Rotation.from_quat(self.command[3:])
        desired_rotation = Rotation.from_quat(self.target[3:])
        if (current.inv() * desired_rotation).magnitude() < 0.1:
            error_w = (desired_rotation * Rotation.from_quat(tcp[3:]).inv()).as_rotvec()
            desired_rotation = Rotation.from_rotvec(min(1.0, 4.0 * self.dt) * error_w) * current
        angle = (current.inv() * desired_rotation).as_rotvec()
        self.command[3:] = (
            current * Rotation.from_rotvec(angle * min(1.0, 0.5 * self.dt / max(np.linalg.norm(angle), 1e-9)))
        ).as_quat()
