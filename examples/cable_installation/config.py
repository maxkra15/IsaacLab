# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured hand postures for housing support and physical clip depression."""

from dataclasses import dataclass

_TUCKED_FINGERS_RAD = (1.1, 0.0, 1.4, 1.0, 0.0, 1.1, 0.0, 1.4, 1.0)
_OPEN_HAND_RAD = (
    0.70216371,
    -0.13133197,
    0.77240945,
    -0.17230791,
    0.00004912,
    0.78591669,
    -0.03844880,
    0.86681288,
    0.12026159,
    1.22023412,
    -0.13107619,
    0.37545449,
    0.70664729,
    *_TUCKED_FINGERS_RAD,
)
_SUPPORT_HAND_RAD = (
    # Oppose the underside fingers with the established housing-grasp thumb posture.
    0.84772129,
    -0.10509452,
    0.67820725,
    -0.10205317,
    0.00001000,
    0.84205546,
    -0.09530088,
    0.77374883,
    0.15160327,
    1.23633955,
    -0.17374855,
    0.41702808,
    0.50194808,
    *_TUCKED_FINGERS_RAD,
)


@dataclass(frozen=True)
class RemovalPostures:
    """Thumb-over/index-and-middle-under keyframes in canonical hand-joint order.

    Finger positions are [rad]. Only the thumb changes between support and
    clip depression; the remaining fingers continue supporting the housing.
    """

    removal_tcp_offset_w: tuple[float, float, float] = (0.06986748, -0.11711917, 0.03615930)
    """Palm displacement at the canonical seated orientation, in world coordinates [m]."""
    removal_tcp_orientation_w: tuple[float, float, float, float] = (
        0.64710128,
        0.39442167,
        -0.52808549,
        -0.38316730,
    )
    """Palm world orientation at the canonical seated connector pose, in XYZW order."""
    removal_hand_open: tuple[float, ...] = _OPEN_HAND_RAD
    removal_hand_support: tuple[float, ...] = _SUPPORT_HAND_RAD
    removal_hand_press: tuple[float, ...] = (
        0.921236652,
        -0.087092333,
        0.572781487,
        -0.034258682,
        0.044998275,
        *_SUPPORT_HAND_RAD[5:],
    )
    """Available thumb travel [rad]; measured clearance stops closure before this endpoint."""
    removal_index_grasp: tuple[float, ...] = (0.85705546, *_SUPPORT_HAND_RAD[6:9])
    removal_middle_grasp: tuple[float, ...] = (1.25133955, -0.18874855, *_SUPPORT_HAND_RAD[11:13])
    """Firmer housing pinch [rad], centered on the housing after measured clip release."""
    removal_thumb_clearance_reserve_m: float = 0.0001
    """Extra measured clip clearance required before withdrawal [m]."""
