# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dependency-light physical constants for the FLARE slung-load system."""

from __future__ import annotations

import math

# Reported or directly measured from FLARE (arXiv:2508.09797v1, Fig. 8/Sec. IV).
DRONE_MASS = 0.305
PAYLOAD_MASS = 0.070
CABLE_NOMINAL_LENGTH = 0.50
MAX_THRUST_TO_WEIGHT = 3.5
GRAVITY = 9.81

# The later FLARE scenario-one release authors these vehicle properties in
# ``assets/suspended_system/real_sus_exp.urdf``. The paper's reported 305 g mass
# remains authoritative here; the release rounds that one value to 300 g.
DRONE_COLLIDER_SIZE = (0.08, 0.08, 0.035)
DRONE_DIAGONAL_INERTIA = (5.6e-4, 5.6e-4, 8.6e-4)
ROTOR_ARM_LENGTH = 0.08
ROTOR_HEIGHT = 0.02
ROTOR_THRUST_COEFFICIENT = 3.16e-10
ROTOR_MOMENT_COEFFICIENT = 7.94e-12
ROTOR_YAW_COEFFICIENT = ROTOR_MOMENT_COEFFICIENT / ROTOR_THRUST_COEFFICIENT
# Reserve roll/pitch authority for the geometric path controller while the
# learned residual remains substantially more agile than ordinary flight rates.
# The complete prior-plus-residual command is still capped by FLARE's published
# (15, 15, 5) rad/s envelope.
ENHANCED_RESIDUAL_BODY_RATE_LIMITS = (10.0, 10.0, 2.5)

# Explicit modeling choices where neither the paper nor release specifies a
# directly reusable value for this AVBD representation.
PAYLOAD_RADIUS = 0.04
CABLE_NUM_POINTS = 9
CABLE_THICKNESS = 0.002
CABLE_DENSITY = 1150.0
CABLE_STRETCH_MODULUS = 5.0e8
CABLE_BEND_MODULUS = 0.0
CABLE_TWIST_MODULUS = 0.0
# The later FLARE release assigns 0.1 N*m*s/rad to each approximately
# 0.48/31 m revolute cable link. Scaling inversely with our 0.50/8 m segment
# length preserves the corresponding continuum bending viscosity.
CABLE_BEND_DAMPING = 0.025
CABLE_MASS = CABLE_DENSITY * math.pi * (0.5 * CABLE_THICKNESS) ** 2 * CABLE_NOMINAL_LENGTH


def nominal_hover_action(max_thrust_to_weight: float = MAX_THRUST_TO_WEIGHT) -> float:
    """Return the normalized collective that balances the nominal loaded system."""
    loaded_thrust_to_drone_weight = (DRONE_MASS + PAYLOAD_MASS + CABLE_MASS) / DRONE_MASS
    return 2.0 * loaded_thrust_to_drone_weight / max_thrust_to_weight - 1.0


def nominal_drone_hover_action(max_thrust_to_weight: float = MAX_THRUST_TO_WEIGHT) -> float:
    """Return the normalized collective that balances the unloaded drone."""
    return 2.0 / max_thrust_to_weight - 1.0
