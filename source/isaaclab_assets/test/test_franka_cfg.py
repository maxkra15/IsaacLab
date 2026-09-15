# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Franka asset configuration contracts."""

from isaaclab_assets import FRANKA_PANDA_MENAGERIE_CFG


def test_franka_menagerie_actuators_define_backend_invariant_properties() -> None:
    """Solver-specific USD payloads must not change passive damping or effort limits."""
    arm = FRANKA_PANDA_MENAGERIE_CFG.actuators["panda_arm"]
    hand = FRANKA_PANDA_MENAGERIE_CFG.actuators["panda_hand"]

    assert arm.joint_effort_limit == {"panda_joint[1-4]": 100.0, "panda_joint[5-7]": 12.0}
    assert arm.viscous_friction == 0.0
    assert hand.joint_effort_limit == {"panda_finger_joint1": 200.0, "panda_finger_joint2": 1.0e6}
    assert hand.viscous_friction == 0.0
