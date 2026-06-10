# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


# Two-way proxy coupling (the gripper proxies are finite-mass bodies that exchange force with the
# cable in both directions). Stable for a single env; the plug grasp can be marginal for num_envs>1.
gym.register(
    id="Isaac-Waterhose-Proxy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.coupled_env_cfg:WaterhoseCoupledEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WaterhosePPORunnerCfg",
    },
)

# Deprecated alias for the two-way proxy task above; kept so existing scripts/CLIs keep working.
gym.register(
    id="Isaac-Waterhose-Coupled-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.coupled_env_cfg:WaterhoseCoupledEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WaterhosePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Waterhose-Admm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.admm_env_cfg:WaterhoseAdmmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WaterhosePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Waterhose-Coupled-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teleop_env_cfg:WaterhoseTeleopEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WaterhosePPORunnerCfg",
    },
)
