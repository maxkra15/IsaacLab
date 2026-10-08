# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compose insertion and removal controllers in one physical episode."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from .contacts import GraspContacts, grasp_contact_failure
from .extraction import CableRemovalStateMachine, removal_hand_targets
from .insertion import CableInstallStateMachine

if TYPE_CHECKING:
    from .config import RemovalPostures


class CableCycleStateMachine:
    """Insert, release, regrasp and extract without resetting or writing body state.

    Both underlying controllers keep their measured seating, slip and retention
    checks; contact diagnostics are optional. Only robot action targets are returned.
    Release holds the measured seated palm pose. A nearby coordinated palm
    correction reaches the clip while fully open, without an axial retreat.

    Args:
        insertion: Insertion controller initialized from the actual TCP pose.
        postures: Measured removal postures, independent of the reset state.
        closed_hand: Insertion finger targets in canonical action order [rad].
        settle_duration_s: Initial stationary physical grasp hold [s].
    """

    def __init__(
        self,
        insertion: CableInstallStateMachine,
        postures: RemovalPostures,
        closed_hand: tuple[float, ...],
        *,
        settle_duration_s: float = 0.0,
    ):
        if not np.isfinite(settle_duration_s) or settle_duration_s < 0.0:
            raise ValueError("settle_duration_s must be finite and nonnegative.")
        self.insertion = insertion
        self.removal: CableRemovalStateMachine | None = None
        self.postures = postures
        self.closed_hand = np.asarray(closed_hand, dtype=np.float64)
        self.released_hand = np.asarray(postures.removal_hand_open, dtype=np.float64)
        if any(hand.shape != (22,) or not np.isfinite(hand).all() for hand in (self.closed_hand, self.released_hand)):
            raise ValueError("Hand targets must contain 22 finite canonical joint positions.")
        self.settle_duration_s = settle_duration_s
        self.elapsed = 0.0
        self.failure: str | None = None
        self.transitions: list[str] = []
        self._prior_phase = self.phase

    @property
    def goal(self) -> np.ndarray:
        """Seated connector goal [m, XYZW], unchanged between demonstrations."""
        return self.insertion.goal

    @property
    def phase(self) -> str:
        """Combined phase; only removal completion is terminal ``DONE``."""
        if self.failure is not None:
            return "FAILED"
        if self.elapsed < self.settle_duration_s:
            return "SETTLE"
        machine = self.active_machine
        if machine.phase == "FAILED" or (self.removal is not None and machine.phase == "DONE"):
            return machine.phase
        return f"{self.demonstration.upper()}/{machine.phase}"

    @property
    def demonstration(self) -> str:
        """Active half of the physical sequence, for frame-aligned telemetry."""
        return "insertion" if self.removal is None else "removal"

    @property
    def active_machine(self) -> CableInstallStateMachine | CableRemovalStateMachine:
        """Underlying physical controller used by the current half."""
        return self.insertion if self.removal is None else self.removal

    @property
    def max_penetration_m(self) -> float:
        """Maximum measured contact penetration across the entire episode [m]."""
        return max(self.insertion.max_penetration_m, self.active_machine.max_penetration_m)

    @property
    def grasp_penetration_limit(self) -> float:
        """Unchanged hydroelastic pad/housing overlap limit [m]."""
        return self.insertion.grasp_penetration_limit

    @property
    def measured_withdrawal_m(self) -> float:
        """Measured extraction displacement [m], zero before the removal half."""
        return 0.0 if self.removal is None else self.removal.measured_withdrawal_m

    def update(
        self,
        tcp_pose_w: np.ndarray,
        head_pose_w: np.ndarray,
        contacts: GraspContacts | None = None,
        *,
        latch_engaged: bool,
        latch_angle_rad: float,
        capture_margins_m: np.ndarray,
        keeper_release_margin_m: float,
        keeper_exit_margin_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return normal TCP [m, XYZW] and finger action targets [rad]."""
        self.elapsed += self.insertion.dt
        if self.elapsed < self.settle_duration_s:
            if contacts is not None:
                self.insertion.max_penetration_m = max(self.insertion.max_penetration_m, contacts.penetration_m)
                failure = grasp_contact_failure(contacts, self.grasp_penetration_limit)
                if failure is not None:
                    self._fail(failure)
            return self.insertion.command.copy(), self.closed_hand.copy()
        if self.removal is None:
            command, width = self.insertion.update(tcp_pose_w, head_pose_w, contacts, latch_engaged=latch_engaged)
            fingers = self.closed_hand + (width / 0.04) * (self.released_hand - self.closed_hand)
            if self.insertion.phase == "DONE":
                self._begin_removal(tcp_pose_w, head_pose_w)
        else:
            command, blends = self.removal.update(
                tcp_pose_w,
                head_pose_w,
                contacts,
                latch_angle_rad=latch_angle_rad,
                capture_margins_m=capture_margins_m,
                keeper_release_margin_m=keeper_release_margin_m,
                keeper_exit_margin_m=keeper_exit_margin_m,
            )
            fingers = removal_hand_targets(self.postures, blends, self.removal.grip_blend)
        if self.active_machine.failure is not None:
            self.failure = self.active_machine.failure
        if self.phase != self._prior_phase:
            self.transitions.append(self.phase)
            self._prior_phase = self.phase
        return command, fingers

    def _begin_removal(self, tcp_pose_w: np.ndarray, head_pose_w: np.ndarray) -> None:
        # Retarget once while open, not after contact-induced motion during closure.
        rotation = Rotation.from_quat(head_pose_w[3:]) * Rotation.from_quat(self.goal[3:]).inv()
        grasp = np.r_[
            head_pose_w[:3] + rotation.apply(self.postures.removal_tcp_offset_w),
            (rotation * Rotation.from_quat(self.postures.removal_tcp_orientation_w)).as_quat(),
        ]
        self.removal = CableRemovalStateMachine(
            tcp_pose_w,
            head_pose_w,
            grasp,
            self.insertion.dt,
            settle_duration_s=0.0,
            grasp_penetration_limit=self.grasp_penetration_limit,
            housing_grasp_body_names=("right_index_elastomer", "right_middle_elastomer"),
            thumb_clearance_reserve_m=self.postures.removal_thumb_clearance_reserve_m,
        )

    def _fail(self, reason: str) -> None:
        self.failure = reason
        self.transitions.append("FAILED")
