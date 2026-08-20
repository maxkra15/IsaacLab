# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka impedance configuration shared with the reset-driven stack workflow."""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_MENAGERIE_CFG

from .asset_provenance import (
    FRANKA_RJ45_ASSET_CLOSURE_TREE_SHA256,
    FRANKA_RJ45_FRANKA_LOGICAL_URI,
    FrankaRJ45AssetClosure,
)

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
PICK_INSERT_ARM_TARGET_TRACKING_LIMITS = (0.145, 0.145, 0.145, 0.145, 0.048, 0.080, 0.240)
"""Pick-task target-error limits [rad], derived from effort limit / stiffness per joint."""

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


def configure_franka_rj45_external_asset(
    robot_cfg: ArticulationCfg,
    verified_closure: FrankaRJ45AssetClosure | None = None,
) -> None:
    """Use the logical Franka identity for diagnostics or a verified local entrypoint for production."""
    if verified_closure is None:
        robot_cfg.spawn.usd_path = FRANKA_RJ45_FRANKA_LOGICAL_URI
        return
    if verified_closure.tree_sha256 != FRANKA_RJ45_ASSET_CLOSURE_TREE_SHA256:
        raise ValueError("Cannot bind a Franka asset from the wrong external-asset closure.")
    robot_cfg.spawn.usd_path = str(verified_closure.franka_usd_path)


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


def franka_pick_insert_control_contract() -> dict[str, object]:
    """Return the pick-only persistent-target and native-gravity control contract."""
    return {
        "contract_version": 1,
        "source_workflow": "maximiliank/franka-newton-stack",
        "target_semantics": "persistent-absolute-integrated-once-per-policy-step",
        "policy_delta_filter": "ema",
        "zero_action_semantics": "clear-ema-tail-and-hold-absolute-target-bitwise",
        "reset_target_semantics": "restore-stored-absolute-actuator-target",
        "target_tracking_error_limits_rad": PICK_INSERT_ARM_TARGET_TRACKING_LIMITS,
        "target_tracking_failure": "terminate-on-nonfinite-or-envelope-violation",
        "native_gravity_compensation": "mjwarp-joint-actuatorgravcomp",
        "native_gravity_compensation_scope": "pick-insert-franka-only",
        "action_inverse_dynamics_gravity_compensation": False,
        "global_inverse_dynamics_gravity_compensation": False,
    }


__all__ = [
    "FRANKA_RJ45_CFG",
    "PICK_INSERT_ARM_TARGET_TRACKING_LIMITS",
    "configure_franka_rj45_external_asset",
    "franka_pick_insert_control_contract",
    "franka_reset_control_contract",
]
