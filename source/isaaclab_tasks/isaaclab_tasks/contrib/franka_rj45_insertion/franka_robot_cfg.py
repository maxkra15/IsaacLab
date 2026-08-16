# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka impedance configuration shared with the reset-driven stack workflow."""

from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_MENAGERIE_CFG

_ARM_EFFORT_LIMITS = {"panda_joint[1-4]": 87.0, "panda_joint[5-7]": 12.0}
_ARM_VELOCITY_LIMITS = {"panda_joint[1-4]": 2.175, "panda_joint[5-7]": 2.61}
_ARM_STIFFNESS = {
    "panda_joint[1-4]": 600.0,
    "panda_joint5": 250.0,
    "panda_joint6": 150.0,
    "panda_joint7": 50.0,
}
_ARM_DAMPING = {
    "panda_joint[1-4]": 50.0,
    "panda_joint5": 30.0,
    "panda_joint6": 25.0,
    "panda_joint7": 15.0,
}
_ARM_ARMATURE = {
    "panda_joint[1-2]": 0.6057,
    "panda_joint[3-4]": 0.4625,
    "panda_joint[5-7]": 0.2055,
}

FRANKA_RJ45_CFG = FRANKA_PANDA_MENAGERIE_CFG.copy()
FRANKA_RJ45_CFG.spawn.rigid_props.disable_gravity = False
FRANKA_RJ45_CFG.actuators = {
    "panda_arm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-7]"],
        effort_limit_sim=_ARM_EFFORT_LIMITS,
        velocity_limit_sim=_ARM_VELOCITY_LIMITS,
        stiffness=_ARM_STIFFNESS,
        damping=_ARM_DAMPING,
        armature=_ARM_ARMATURE,
    ),
    "panda_hand": ImplicitActuatorCfg(
        joint_names_expr=["panda_finger_joint[1-2]"],
        effort_limit_sim=70.0,
        velocity_limit_sim=2.0,
        stiffness=350.0,
        damping=175.0,
        armature=0.1,
    ),
}


def franka_reset_control_contract() -> dict[str, object]:
    """Return serialization-stable controls that invalidate reset snapshots."""
    return {
        "contract_version": 1,
        "source_workflow": "maximiliank/franka-newton-stack",
        "arm_effort_limits": tuple(_ARM_EFFORT_LIMITS.items()),
        "arm_velocity_limits": tuple(_ARM_VELOCITY_LIMITS.items()),
        "arm_stiffness": tuple(_ARM_STIFFNESS.items()),
        "arm_damping": tuple(_ARM_DAMPING.items()),
        "arm_armature": tuple(_ARM_ARMATURE.items()),
        "finger_effort_limit": 70.0,
        "finger_velocity_limit": 2.0,
        "finger_stiffness": 350.0,
        "finger_damping": 175.0,
        "finger_armature": 0.1,
        "gravity_compensation": False,
        "gravity_compensation_compatibility": (
            "Newton 1.5 inverse dynamics rejects global models containing JointType.CABLE"
        ),
        "relative_target_semantics": "measured-state-once-per-policy-step-with-reset-bias",
    }


__all__ = ["FRANKA_RJ45_CFG", "franka_reset_control_contract"]
