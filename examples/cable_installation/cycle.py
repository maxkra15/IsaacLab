# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compose insertion and removal controllers in one physical episode."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .contacts import GraspContacts, grasp_contact_failure
from .extraction import CableRemovalStateMachine, removal_hand_targets
from .insertion import CableInstallStateMachine

if TYPE_CHECKING:
    from .config import RemovalPostures


class CableCycleStateMachine:
    """Insert, release, regrasp and extract without resetting or writing body state.

    Both underlying controllers keep their contact, seating, slip and retention
    checks. Only robot action targets are returned. The hand first leaves the
    rack without wrist rotation, then changes to the thumb-over removal grasp.

    Args:
        insertion: Insertion controller initialized from the actual TCP pose.
        postures: Measured removal postures, independent of the reset state.
        closed_hand: Insertion finger targets in canonical action order [rad].
        released_hand: Insertion release targets in canonical action order [rad].
        settle_duration_s: Stationary opening-shot duration [s].
    """

    def __init__(
        self,
        insertion: CableInstallStateMachine,
        postures: RemovalPostures,
        closed_hand: tuple[float, ...],
        released_hand: tuple[float, ...],
        *,
        settle_duration_s: float = 0.0,
    ):
        if not np.isfinite(settle_duration_s) or settle_duration_s < 0.0:
            raise ValueError("settle_duration_s must be finite and nonnegative.")
        self.insertion = insertion
        self.removal: CableRemovalStateMachine | None = None
        self.postures = postures
        self.closed_hand = np.asarray(closed_hand, dtype=np.float64)
        self.released_hand = np.asarray(released_hand, dtype=np.float64)
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
    def approach_angle_tolerance(self) -> float:
        """Insertion approach tolerance [rad]; final seating checks are unchanged."""
        return self.insertion.approach_angle_tolerance

    @property
    def measured_withdrawal_m(self) -> float:
        """Measured extraction displacement [m], zero before the removal half."""
        return 0.0 if self.removal is None else self.removal.measured_withdrawal_m

    def update(
        self,
        tcp_pose_w: np.ndarray,
        head_pose_w: np.ndarray,
        contacts: GraspContacts,
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
            self.insertion.max_penetration_m = max(self.insertion.max_penetration_m, contacts.penetration_m)
            failure = grasp_contact_failure(contacts, self.grasp_penetration_limit)
            if failure is not None:
                self._fail(failure)
            return self.insertion.command.copy(), self.closed_hand.copy()
        if self.removal is None:
            command, width = self.insertion.update(tcp_pose_w, head_pose_w, contacts, latch_engaged=latch_engaged)
            fingers = self.closed_hand + (width / 0.04) * (self.released_hand - self.closed_hand)
            if self.insertion.phase == "DONE":
                self._begin_removal(tcp_pose_w)
        else:
            command, blends = self.removal.update(
                tcp_pose_w,
                head_pose_w,
                contacts,
                latch_angle_rad=latch_angle_rad,
                capture_margins_m=capture_margins_m,
                thumb_latch_contact=contacts.thumb_latch,
                keeper_release_margin_m=keeper_release_margin_m,
                keeper_exit_margin_m=keeper_exit_margin_m,
            )
            fingers = removal_hand_targets(self.postures, blends, self.removal.grip_blend)
            if self.removal.phase in ("SETTLE", "CLEAR"):
                fingers = self.released_hand.copy()
            elif self.removal.phase == "ORIENT":
                # Fold to the removal posture only after retreating clear of the rack.
                progress = min(self.removal.elapsed, 1.0)
                fraction = progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))
                fingers = self.released_hand + fraction * (fingers - self.released_hand)
        if self.active_machine.failure is not None:
            self.failure = self.active_machine.failure
        if self.phase != self._prior_phase:
            self.transitions.append(self.phase)
            self._prior_phase = self.phase
        return command, fingers

    def _begin_removal(self, tcp_pose_w: np.ndarray) -> None:
        clear = np.asarray(tcp_pose_w, dtype=np.float64).copy()
        clear[1] = min(clear[1], self.goal[1] - 0.25)
        grasp = np.r_[self.goal[:3] + self.postures.removal_tcp_offset_w, self.postures.removal_tcp_orientation_w]
        approach = np.r_[clear[:3], grasp[3:]]
        staging = grasp.copy()
        staging[1] = clear[1]
        self.removal = CableRemovalStateMachine(
            tcp_pose_w,
            self.goal,
            grasp,
            self.insertion.dt,
            clear_tcp_pose_w=clear,
            approach_tcp_pose_w=approach,
            staging_tcp_pose_w=staging,
            settle_duration_s=0.0,
            grasp_penetration_limit=self.grasp_penetration_limit,
            housing_grasp_body_names=("right_index_elastomer", "right_middle_elastomer"),
            thumb_clearance_reserve_m=self.postures.removal_thumb_clearance_reserve_m,
        )

    def _fail(self, reason: str) -> None:
        self.failure = reason
        self.transitions.append("FAILED")
