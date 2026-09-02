# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="IsaacContrib-DroneSlungLoad-Waypoint-FLARE",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.contrib.drone_slung_load.drone_slung_load_env_cfg:DroneSlungLoadWaypointEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DroneSlungLoadWaypointPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-DroneSlungLoad-Waypoint-FLARE-Enhanced",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.contrib.drone_slung_load.drone_slung_load_env_cfg:DroneSlungLoadWaypointEnhancedEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_ppo_cfg:DroneSlungLoadWaypointEnhancedPPORunnerCfg"),
    },
)

gym.register(
    id="IsaacContrib-DroneSlungLoad-Waypoint-FLARE-DirectCTBR",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.contrib.drone_slung_load.drone_slung_load_env_cfg:DroneSlungLoadWaypointDirectCTBREnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_ppo_cfg:DroneSlungLoadWaypointDirectCTBRPPORunnerCfg"),
    },
)

gym.register(
    id="IsaacContrib-Drone-Waypoint-FLARE-DirectCTBR",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.contrib.drone_slung_load.drone_direct_ctbr_env_cfg:DroneWaypointDirectCTBREnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_ppo_cfg:DroneWaypointDirectCTBRPPORunnerCfg"),
    },
)
