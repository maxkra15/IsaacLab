# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for one-ball KUKA-Allegro juggling."""

from isaaclab.envs.mdp import action_rate_l2, joint_pos_rel, joint_vel_rel, last_action, time_out

from isaaclab_tasks.contrib.stack.mdp.actions_cfg import WorkspaceBoundedRelativeJointPositionActionCfg

from .actions import JuggleResetPreservingRelativeJointPositionActionCfg
from .curriculum import JuggleResetCurriculum
from .reset import (
    BALL_MASS,
    BALL_RADIUS,
    GRAVITY_Z,
    JUGGLE_SPHERE_CENTER_OFFSET,
    JUGGLE_SPHERE_CONTACT_HAND_POSITION,
    JUGGLE_SPHERE_FLIGHT_GATE_HAND_POSITION,
    JUGGLE_SPHERE_OPEN_HAND_POSITION,
    JUGGLE_SPHERE_PRELOAD_HAND_POSITION,
    KUKA_ALLEGRO_JUGGLE_ARM_WORKSPACE_LOWER,
    KUKA_ALLEGRO_JUGGLE_ARM_WORKSPACE_UPPER,
    KUKA_ARM_JOINT_NAMES,
    JuggleLocalGoal,
    JugglePhase,
    JuggleResetEvent,
    JuggleResetStateSource,
    ballistic_state,
    kuka_allegro_tool_point_velocity,
    local_goal_for_phase,
)
from .runtime import (
    JuggleRuntimeState,
    create_juggle_runtime_state,
    get_juggle_runtime_state,
    initialize_juggle_episode_state,
)
from .terms import (
    JuggleProgressContext,
    ball_height_and_velocity,
    ball_angular_velocity,
    ball_out_of_workspace,
    ball_position_relative_to_tool,
    ball_velocity_relative_to_tool,
    cycle_success,
    first_ascent_apex_crossing,
    fingertip_velocities_relative_to_ball,
    fingertips_relative_to_ball,
    full_cycle_pulse,
    hand_closure,
    local_transition_pulse,
    noncanonical_local_goal_success,
    nonfinite_state,
    palm_twist,
    phase_one_hot,
    sphere_support_from_fingertips,
    tool_axes,
    tool_state,
)

__all__ = [
    "BALL_MASS",
    "BALL_RADIUS",
    "GRAVITY_Z",
    "JUGGLE_SPHERE_CENTER_OFFSET",
    "JUGGLE_SPHERE_CONTACT_HAND_POSITION",
    "JUGGLE_SPHERE_FLIGHT_GATE_HAND_POSITION",
    "JUGGLE_SPHERE_OPEN_HAND_POSITION",
    "JUGGLE_SPHERE_PRELOAD_HAND_POSITION",
    "JugglePhase",
    "JuggleLocalGoal",
    "JuggleProgressContext",
    "JuggleResetCurriculum",
    "JuggleResetEvent",
    "JuggleResetPreservingRelativeJointPositionActionCfg",
    "JuggleResetStateSource",
    "JuggleRuntimeState",
    "KUKA_ALLEGRO_JUGGLE_ARM_WORKSPACE_LOWER",
    "KUKA_ALLEGRO_JUGGLE_ARM_WORKSPACE_UPPER",
    "KUKA_ARM_JOINT_NAMES",
    "WorkspaceBoundedRelativeJointPositionActionCfg",
    "action_rate_l2",
    "ball_height_and_velocity",
    "ball_angular_velocity",
    "ball_out_of_workspace",
    "ball_position_relative_to_tool",
    "ball_velocity_relative_to_tool",
    "ballistic_state",
    "create_juggle_runtime_state",
    "cycle_success",
    "first_ascent_apex_crossing",
    "fingertips_relative_to_ball",
    "fingertip_velocities_relative_to_ball",
    "full_cycle_pulse",
    "get_juggle_runtime_state",
    "hand_closure",
    "initialize_juggle_episode_state",
    "joint_pos_rel",
    "joint_vel_rel",
    "kuka_allegro_tool_point_velocity",
    "last_action",
    "local_transition_pulse",
    "local_goal_for_phase",
    "noncanonical_local_goal_success",
    "nonfinite_state",
    "palm_twist",
    "phase_one_hot",
    "sphere_support_from_fingertips",
    "time_out",
    "tool_axes",
    "tool_state",
]
