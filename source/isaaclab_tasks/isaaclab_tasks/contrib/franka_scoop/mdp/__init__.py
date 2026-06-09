# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the Franka scoop two-container transfer task."""

from isaaclab.envs.mdp import last_action, time_out  # noqa: F401

from .actions import ScoopAction, ScoopActionCfg  # noqa: F401
from .curriculums import ScoopCurriculum  # noqa: F401
from .events import reset_scoop_scene, spawn_scoop_kit_visuals  # noqa: F401
from .observations import (  # noqa: F401
    arm_joint_pos_norm,
    arm_joint_vel_scaled,
    bowl_pose_obs,
    bowl_to_source_obs,
    bowl_to_target_obs,
    count_in_bowl_obs,
    count_in_source_obs,
    count_in_target_obs,
    heightfield_obs,
    particle_summary_obs,
)
from .rewards import (  # noqa: F401
    action_l2,
    carry_to_target,
    particles_in_bowl,
    particles_in_target,
    reach_source,
    removed_from_source,
    transfer_success_bonus,
)
from .terminations import nonfinite_failure, transfer_success_mask  # noqa: F401
