# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured housing-pinch and clip-press Ethernet removal with optional contact diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from .contacts import GraspContacts, grasp_contact_failure

if TYPE_CHECKING:
    from .config import RemovalPostures


def removal_hand_targets(cfg: RemovalPostures, blends: tuple[float, float], grip_blend: float) -> np.ndarray:
    """Return canonical 22-joint targets [rad] for a supported thumb clip press."""
    opened, support, press = (
        np.asarray(values) for values in (cfg.removal_hand_open, cfg.removal_hand_support, cfg.removal_hand_press)
    )
    fingers = opened + blends[0] * (support - opened)
    fingers += blends[1] * (press - fingers)
    housing = np.asarray((*cfg.removal_index_grasp, *cfg.removal_middle_grasp))
    fingers[5:13] += grip_blend * (housing - fingers[5:13])
    return fingers


class CableRemovalStateMachine:
    """Support the housing, depress its real clip, then withdraw along socket -Y.

    The caller maps two returned hand blends to its open, supporting and clip-press
    joint targets. Measured clip clearance gates withdrawal, and relative poses
    check slip and rotation. Optional contact diagnostics additionally check
    underside support and thumb/clip contact. Neither actuator targets nor elapsed
    time alone establish an unclipped connector.

    Args:
        tcp_pose_w: Initial measured TCP pose [m, XYZW].
        seated_head_pose_w: Measured seated connector pose [m, XYZW].
        grasp_tcp_pose_w: Calibrated TCP pose for housing-under/clip-over grasp [m, XYZW].
        dt: Controller interval [s].
        settle_duration_s: Minimum stationary establishing-shot duration [s].
        withdrawal_distance_m: Straight axial extraction distance [m].
        withdrawal_duration_s: Duration of the smooth axial extraction ramp [s].
        grasp_penetration_limit: Allowed hydroelastic pad/housing overlap [m].
        housing_grasp_body_names: Pads that must each contact the rigid housing
            before and during extraction; clip contact is checked separately.
        thumb_clearance_reserve_m: Extra clip clearance required before pulling and targeted by feedback [m].
            Release, grasp and collision guards are unchanged.
    """

    def __init__(
        self,
        tcp_pose_w: np.ndarray,
        seated_head_pose_w: np.ndarray,
        grasp_tcp_pose_w: np.ndarray,
        dt: float,
        *,
        settle_duration_s: float = 1.0,
        withdrawal_distance_m: float = 0.06,
        withdrawal_duration_s: float = 3.0,
        grasp_penetration_limit: float = 0.003,
        housing_grasp_body_names: tuple[str, ...] = ("right_index_elastomer",),
        thumb_clearance_reserve_m: float = 0.0,
    ):
        for name, value in (
            ("dt", dt),
            ("withdrawal_distance_m", withdrawal_distance_m),
            ("withdrawal_duration_s", withdrawal_duration_s),
            ("grasp_penetration_limit", grasp_penetration_limit),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isfinite(settle_duration_s) or settle_duration_s < 0.0:
            raise ValueError("settle_duration_s must be finite and nonnegative.")
        if not housing_grasp_body_names:
            raise ValueError("At least one housing-grasp pad is required.")
        if not np.isfinite(thumb_clearance_reserve_m) or thumb_clearance_reserve_m < 0.0:
            raise ValueError("thumb_clearance_reserve_m must be finite and nonnegative.")
        self.command = _pose(tcp_pose_w)
        self.settle_pose = self.command.copy()
        self.goal = _pose(seated_head_pose_w)
        self.grasp = _pose(grasp_tcp_pose_w)
        self.dt = dt
        self.settle_duration_s = settle_duration_s
        self.withdrawal_distance_m = withdrawal_distance_m
        self.withdrawal_duration_s = withdrawal_duration_s
        self.grasp_penetration_limit = grasp_penetration_limit
        self.housing_grasp_body_names = housing_grasp_body_names
        self.thumb_clearance_reserve_m = thumb_clearance_reserve_m
        self.phase = "SETTLE"
        self.elapsed = self.dwell = self.contact_loss_s = 0.0
        self.latch_loss_s = self.withdrawal_progress_s = 0.0
        self.hand_blends = (0.0, 0.0)
        self.grip_blend = 0.0
        """Additional housing-pinch blend in [0, 1], established after measured clip release."""
        self.max_penetration_m = 0.0
        self.measured_withdrawal_m = 0.0
        self.withdrawal_start_distance_m = 0.0
        self.failure: str | None = None
        self.press_tcp: np.ndarray | None = None
        self.withdrawal_tcp: np.ndarray | None = None
        self.withdrawal_hold_tcp: np.ndarray | None = None
        self.grasp_offset_m: np.ndarray | None = None
        self.grasp_rotation: Rotation | None = None

    def update(
        self,
        tcp_pose_w: np.ndarray,
        head_pose_w: np.ndarray,
        contacts: GraspContacts | None = None,
        *,
        latch_angle_rad: float,
        capture_margins_m: np.ndarray,
        thumb_latch_contact: bool | None = None,
        keeper_release_margin_m: float | None = None,
        keeper_exit_margin_m: float | None = None,
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """Return TCP pose [m, XYZW] and support/clip-press hand blends in [0, 1].

        Args:
            tcp_pose_w: Measured TCP pose [m, XYZW].
            head_pose_w: Measured connector pose [m, XYZW].
            contacts: Optional physical contacts, with underside housing support in ``right``.
                ``None`` skips contact diagnostics, not measured pose or clip-clearance checks.
            latch_angle_rad: Measured clip hinge deflection [rad]; inward is negative.
            capture_margins_m: Measured axial, lateral and vertical keeper overlap [m].
            thumb_latch_contact: Optional measured thumb/clip contact for contact diagnostics.
                Defaults to the supplied contacts' clip measurement.
            keeper_release_margin_m: Worst vertical overlap of both clip barbs [m].
                Defaults to the capture margin for callers that do not measure both.
            keeper_exit_margin_m: Conservative axial overlap of the entire clip with
                the keeper front plane [m]. Without it, retain the 18 mm withdrawal
                threshold. Measured geometry must clear the plane by 0.3 mm.

        Returns:
            Arm pose and two hand blends. The second blend only presses the thumb;
            it stops at measured clip clearance and is retained during withdrawal.
            Supporting finger joints remain closed while it changes.
        """
        if self.phase in ("DONE", "FAILED"):
            return self.command.copy(), self.hand_blends
        tcp, head = _pose(tcp_pose_w), _pose(head_pose_w)
        margins = np.asarray(capture_margins_m, dtype=np.float64)
        if (
            margins.shape != (3,)
            or not np.isfinite(margins).all()
            or not np.isfinite(latch_angle_rad)
            or (keeper_release_margin_m is not None and not np.isfinite(keeper_release_margin_m))
            or (keeper_exit_margin_m is not None and not np.isfinite(keeper_exit_margin_m))
        ):
            self._fail("Non-finite or invalid measured latch state")
            return self.command.copy(), self.hand_blends
        release_margin = margins[2] if keeper_release_margin_m is None else keeper_release_margin_m
        self.elapsed += self.dt
        if contacts is not None:
            self.max_penetration_m = max(self.max_penetration_m, contacts.penetration_m)
            failure = grasp_contact_failure(contacts, self.grasp_penetration_limit)
            if failure is not None:
                self._fail(failure)
                return self.command.copy(), self.hand_blends
            if thumb_latch_contact is None:
                thumb_latch_contact = contacts.thumb_latch
        self.measured_withdrawal_m = self.goal[1] - head[1]
        if self.elapsed > (self.settle_duration_s + 15.0 if self.phase == "SETTLE" else 15.0):
            self._fail(f"{self.phase} timed out")
            return self.command.copy(), self.hand_blends

        target = self.settle_pose.copy()
        # Keep the chosen grasp heading while following the housing's approach translation.
        grasp = self.grasp.copy()
        grasp[:3] += head[:3] - self.goal[:3]
        support_ready = contacts is None or contacts.right
        grasp_ready = contacts is None or _housing_purchase(contacts, self.housing_grasp_body_names)
        # A loaded plug can pitch in its jack and release at a different hinge
        # angle. Require physical depression and actual clearance of both barbs.
        clip_clear = latch_angle_rad < 0.0 and release_margin < -0.00002
        unclipped = clip_clear and (contacts is None or (support_ready and thumb_latch_contact))
        if self.phase == "SETTLE":
            seated = self._pose_error(head, self.goal, 0.003, 0.08)
            captured = margins[0] > -0.00005 and np.all(margins[1:] > 0.0)
            if self._dwelled(seated and captured, 0.3) and self.elapsed >= self.settle_duration_s:
                self._transition("ORIENT")
        elif self.phase in ("ORIENT", "APPROACH"):
            target = self.grasp if self.phase == "ORIENT" else grasp
            hand_ready = self.phase != "ORIENT" or self.elapsed >= 1.0
            if self._dwelled(self._pose_error(tcp, target, 0.002, 0.03), 0.3) and hand_ready:
                self._transition("APPROACH" if self.phase == "ORIENT" else "SUPPORT")
        elif self.phase == "SUPPORT":
            # Chasing a connector lifted by the closing fingers amplifies its
            # motion against the jack roof and changes the clip-release geometry.
            if self.press_tcp is None:
                self.press_tcp = tcp.copy()
            target = self.press_tcp
            self.hand_blends = (_smooth(self.elapsed / 2.0), 0.0)
            # Let support settle before the thumb releases the latch.
            if self.elapsed >= 2.0 and self._dwelled(support_ready and grasp_ready, 0.3):
                self._transition("PRESS")
        elif self.phase in ("PRESS", "PINCH"):
            # Clip motion must not make the supported palm pivot after the plug.
            target = self.press_tcp
            self.contact_loss_s = 0.0 if support_ready else self.contact_loss_s + self.dt
            if not support_ready:
                self.dwell = 0.0
                if self.contact_loss_s > 0.2:
                    self._fail("Underside housing support lost during clip depression")
                self._advance_command(target, tcp)
                return self.command.copy(), self.hand_blends
            self._update_thumb_press(latch_angle_rad, release_margin)
            if self.phase == "PRESS":
                if self.elapsed >= 1.0 and self._dwelled(unclipped, 0.2):
                    self._transition("PINCH")
            else:
                # Tighten the housing pinch only with opposing thumb contact after release.
                self.grip_blend = _smooth(self.elapsed / 1.0)
                reserved = release_margin + self.thumb_clearance_reserve_m < -0.00002
                if self._dwelled(self.grip_blend == 1.0 and unclipped and grasp_ready and reserved, 0.2):
                    rotation = Rotation.from_quat(tcp[3:]).inv()
                    self.grasp_offset_m = rotation.apply(head[:3] - tcp[:3])
                    self.grasp_rotation = rotation * Rotation.from_quat(head[3:])
                    self.withdrawal_tcp = tcp.copy()
                    self.withdrawal_start_distance_m = self.measured_withdrawal_m
                    self._transition("WITHDRAW")
        elif self.phase in ("WITHDRAW", "HOLD"):
            clear = (
                self.measured_withdrawal_m > 0.018 if keeper_exit_margin_m is None else keeper_exit_margin_m < -0.0003
            )
            opposition_ready = contacts is None or thumb_latch_contact or contacts.left
            held = grasp_ready and (not clear or opposition_ready)
            self.contact_loss_s = 0.0 if held else self.contact_loss_s + self.dt
            relative = Rotation.from_quat(tcp[3:]).inv()
            slip = np.linalg.norm(relative.apply(head[:3] - tcp[:3]) - self.grasp_offset_m)
            roll = (self.grasp_rotation.inv() * relative * Rotation.from_quat(head[3:])).magnitude()
            if self.contact_loss_s > 0.2 or slip > 0.005 or roll > 0.15:
                self._fail("Physical housing purchase or connector grasp lost during removal")
                return self.command.copy(), self.hand_blends
            if not clear:
                # A moving grasp needs the same measured latch feedback as a
                # stationary press; freezing the target lets the clip rebound.
                self._update_thumb_press(latch_angle_rad, release_margin)
            target = self.withdrawal_tcp.copy()
            if self.phase == "WITHDRAW":
                reserved = clip_clear and release_margin + self.thumb_clearance_reserve_m <= -0.00005
                target = self._withdrawal_target(tcp, clear, reserved)
                if self.phase == "FAILED":
                    return self.command.copy(), self.hand_blends
            else:
                target[1] -= max(0.0, self.withdrawal_distance_m - self.withdrawal_start_distance_m)
            if self.phase == "WITHDRAW" and self.withdrawal_hold_tcp is None:
                if self.withdrawal_progress_s >= self.withdrawal_duration_s and self._dwelled(
                    self.measured_withdrawal_m >= self.withdrawal_distance_m - 0.003
                    and self._pose_error(tcp, target, 0.003, 0.05),
                    0.3,
                ):
                    self._transition("HOLD")
            elif self.phase == "HOLD" and self._dwelled(
                clear and self.measured_withdrawal_m >= self.withdrawal_distance_m - 0.003, 2.0
            ):
                self._transition("DONE")

        self._advance_command(target, tcp)
        return self.command.copy(), self.hand_blends

    def _withdrawal_target(self, tcp: np.ndarray, clear: bool, reserved: bool) -> np.ndarray:
        """Pause at measured TCP and smoothly restart the remaining axial travel [m, XYZW]."""
        if not clear and not reserved and self.withdrawal_hold_tcp is None:
            # Cancel tracking lead, retaining the original physical grasp reference.
            self.withdrawal_hold_tcp = tcp.copy()
            self.command = tcp.copy()
            self.dwell = self.latch_loss_s = 0.0
        if self.withdrawal_hold_tcp is not None:
            self.latch_loss_s += self.dt
            if self._dwelled(clear or reserved, 0.2):
                self.withdrawal_tcp = tcp.copy()
                self.withdrawal_start_distance_m = self.measured_withdrawal_m
                self.withdrawal_progress_s = self.latch_loss_s = self.dwell = 0.0
                self.withdrawal_hold_tcp = None
            else:
                if self.latch_loss_s > 2.0:
                    self._fail("Clip rebounded before clearing the socket keepers")
                return self.withdrawal_hold_tcp.copy()
        else:
            self.withdrawal_progress_s += self.dt
        target = self.withdrawal_tcp.copy()
        remaining = max(0.0, self.withdrawal_distance_m - self.withdrawal_start_distance_m)
        target[1] -= remaining * _smooth(self.withdrawal_progress_s / self.withdrawal_duration_s)
        return target

    def _update_thumb_press(self, latch_angle_rad: float, release_margin_m: float) -> None:
        release_margin_m += self.thumb_clearance_reserve_m
        moving = self.phase == "WITHDRAW"
        clearance = -0.00005 if moving else -0.00002
        released = latch_angle_rad < 0.0 and release_margin_m <= clearance
        thumb_blend = self.hand_blends[1]
        if latch_angle_rad <= -0.23:
            thumb_blend = max(0.0, thumb_blend - 0.15 * self.dt)
        elif not released:
            # Full keyframe closure can overload the clip's compliant stop.
            target = _smooth(self.elapsed / 2.0) if self.phase == "PRESS" else 1.0
            rate = 0.3
            thumb_blend = max(thumb_blend, min(target, thumb_blend + rate * self.dt))
        self.hand_blends = (1.0, thumb_blend)

    def _transition(self, phase: str) -> None:
        self.phase = phase
        self.elapsed = self.dwell = self.contact_loss_s = 0.0
        self.latch_loss_s = 0.0

    def _fail(self, reason: str) -> None:
        self.failure = reason
        self._transition("FAILED")

    def _dwelled(self, condition: bool, duration_s: float) -> bool:
        self.dwell = self.dwell + self.dt if condition else 0.0
        return self.dwell >= duration_s

    def _advance_command(self, target: np.ndarray, tcp: np.ndarray) -> None:
        desired = target.copy()
        if np.linalg.norm(target[:3] - self.command[:3]) < 0.02:
            desired[:3] = self.command[:3] + min(1.0, 4.0 * self.dt) * (target[:3] - tcp[:3])
        delta = desired[:3] - self.command[:3]
        self.command[:3] += delta * min(1.0, 0.04 * self.dt / max(np.linalg.norm(delta), 1e-9))
        rotation = Rotation.from_quat(self.command[3:])
        desired_rotation = Rotation.from_quat(desired[3:])
        if (rotation.inv() * desired_rotation).magnitude() < 0.1:
            error = (desired_rotation * Rotation.from_quat(tcp[3:]).inv()).as_rotvec()
            desired_rotation = Rotation.from_rotvec(min(1.0, 4.0 * self.dt) * error) * rotation
        angle = (rotation.inv() * desired_rotation).as_rotvec()
        self.command[3:] = (
            rotation * Rotation.from_rotvec(angle * min(1.0, 0.5 * self.dt / max(np.linalg.norm(angle), 1e-9)))
        ).as_quat()

    @staticmethod
    def _pose_error(pose: np.ndarray, target: np.ndarray, position_m: float, angle_rad: float) -> bool:
        return (
            np.linalg.norm(pose[:3] - target[:3]) < position_m
            and (Rotation.from_quat(pose[3:]).inv() * Rotation.from_quat(target[3:])).magnitude() < angle_rad
        )


def _smooth(progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))


def _housing_purchase(contacts: GraspContacts, body_names: tuple[str, ...]) -> bool:
    if not contacts.right:
        return False
    purchased = set()
    for pair, depth in contacts.pair_penetrations:
        labels = pair.split(" | ")
        if depth <= 0.0 or not any(label.endswith("/Cable/head/grip") for label in labels):
            continue
        for body_name in body_names:
            if any(f"/Robot/{body_name}/" in label and label.endswith("/hydroelastic_grasp") for label in labels):
                purchased.add(body_name)
    return all(body_name in purchased for body_name in body_names)


def _pose(value: np.ndarray) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64).copy()
    if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-9:
        raise ValueError("TCP and connector poses must contain finite [m, XYZW] values and a nonzero quaternion.")
    return pose
