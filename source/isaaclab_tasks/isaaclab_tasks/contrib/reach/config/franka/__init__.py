# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from isaaclab_tasks.core.reach.config.franka import agents as core_agents

##
# Register Gym environments.
##

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Reach-Franka-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaReachEnvCfg",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Reach-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:FrankaReachEnvCfg",
    },
    disable_env_checker=True,
)

##
# Newton Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Reach-Franka-Newton-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_newton_env_cfg:FrankaReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{core_agents.__name__}.rsl_rl_ppo_cfg:FrankaReachNewtonIKPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Reach-Franka-Newton-IK-Rel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_newton_env_cfg:FrankaReachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{core_agents.__name__}.rsl_rl_ppo_cfg:FrankaReachNewtonIKPPORunnerCfg",
    },
    disable_env_checker=True,
)
