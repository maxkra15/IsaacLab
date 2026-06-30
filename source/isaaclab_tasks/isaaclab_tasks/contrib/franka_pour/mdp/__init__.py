# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the Franka pour task (grasp a dynamic cup of MPM media and pour)."""

from isaaclab.envs.mdp import (  # noqa: F401
    AbsBinaryJointPositionActionCfg,
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    time_out,
)

from .events import reset_pour_scene  # noqa: F401
from .observations import (  # noqa: F401
    cup_pose_obs,
    cup_to_target_obs,
    ee_pose_obs,
    gripper_width_obs,
    particle_fractions_obs,
    target_pose_obs,
    tcp_pose_obs,
    tcp_to_grasp_obs,
)
from .rewards import (  # noqa: F401
    action_l2,
    align_command_progress,
    align_cup_over_target,
    grasp_cup,
    lift_command_progress,
    lift_cup,
    particles_in_source,
    particles_in_target,
    pour_success_bonus,
    reach_cup,
    spilled_particles,
    tilt_command_progress,
    tilt_over_target,
)
from .terminations import nonfinite_failure  # noqa: F401
