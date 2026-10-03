# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured hand postures for housing support and physical clip depression."""

from dataclasses import dataclass

_TUCKED_FINGERS_RAD = (1.1, 0.0, 1.4, 1.0, 0.0, 1.1, 0.0, 1.4, 1.0)
# Move the index below the seated housing before retreating from the rack.
RELEASE_HAND_RAD = (
    0.70216371,
    -0.13133197,
    0.77240945,
    -0.17230791,
    0.00004912,
    0.96448826,
    -0.03295866,
    0.28314956,
    0.87137673,
    1.24267003,
    -0.15347209,
    0.36637417,
    0.69207724,
    *_TUCKED_FINGERS_RAD,
)
_OPEN_HAND_RAD = (
    0.70216371,
    -0.13133197,
    0.77240945,
    -0.17230791,
    0.00004912,
    0.79493352,
    -0.06493176,
    0.87773075,
    0.12447356,
    1.24267003,
    -0.15347209,
    0.36637417,
    0.69207724,
    *_TUCKED_FINGERS_RAD,
)
_SUPPORT_HAND_RAD = (
    *_OPEN_HAND_RAD[:5],
    0.86205546,
    -0.09530088,
    0.77374883,
    0.15160327,
    1.26633955,
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
    """Palm displacement from the seated connector in world coordinates [m]."""
    removal_tcp_orientation_w: tuple[float, float, float, float] = (
        0.64710128,
        0.39442167,
        -0.52808549,
        -0.38316730,
    )
    """Canonical-palm world orientation in XYZW order."""
    removal_hand_open: tuple[float, ...] = _OPEN_HAND_RAD
    removal_hand_support: tuple[float, ...] = _SUPPORT_HAND_RAD
    removal_hand_press: tuple[float, ...] = (
        0.92024842,
        -0.13336721,
        0.65438666,
        0.00508673,
        0.00001000,
        *_SUPPORT_HAND_RAD[5:],
    )
    removal_index_grasp: tuple[float, ...] = _SUPPORT_HAND_RAD[5:9]
    removal_middle_grasp: tuple[float, ...] = _SUPPORT_HAND_RAD[9:13]
    removal_thumb_clearance_reserve_m: float = 0.0001
    """Extra measured clip clearance required before withdrawal [m]."""
