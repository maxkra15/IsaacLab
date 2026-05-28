# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym


_PKG = "isaaclab_tasks.manager_based.manipulation.waterhose_robot_demo"


gym.register(
    id="Isaac-Waterhose-Robot-Demo-v0",
    entry_point=f"{_PKG}.env:WaterhoseRobotDemoEnv",
    kwargs={"env_cfg_entry_point": f"{_PKG}.env_cfg:WaterhoseRobotDemoEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Waterhose-Robot-Demo-Mimic-v0",
    entry_point=f"{_PKG}.mimic_env:WaterhoseRobotDemoMimicEnv",
    kwargs={"env_cfg_entry_point": f"{_PKG}.mimic_env_cfg:WaterhoseRobotDemoMimicEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Waterhose-Robot-Demo-Admm-Experimental-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{_PKG}.admm_env_cfg:WaterhoseRobotDemoAdmmEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Waterhose-Robot-Demo-OneWay-Coupled-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{_PKG}.one_way_env_cfg:WaterhoseRobotDemoOneWayEnvCfg"},
    disable_env_checker=True,
)
