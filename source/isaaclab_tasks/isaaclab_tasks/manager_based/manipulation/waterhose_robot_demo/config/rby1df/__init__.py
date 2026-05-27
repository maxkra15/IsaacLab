# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym


gym.register(
    id="Isaac-Waterhose-Robot-Demo-v0",
    entry_point="isaaclab_tasks.manager_based.manipulation.waterhose_robot_demo.waterhose_robot_demo_env:WaterhoseRobotDemoEnv",
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.manager_based.manipulation.waterhose_robot_demo."
            "waterhose_robot_demo_env_cfg:WaterhoseRobotDemoEnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Waterhose-Robot-Demo-Play-v0",
    entry_point="isaaclab_tasks.manager_based.manipulation.waterhose_robot_demo.waterhose_robot_demo_env:WaterhoseRobotDemoEnv",
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.manager_based.manipulation.waterhose_robot_demo."
            "waterhose_robot_demo_env_cfg:WaterhoseRobotDemoEnvCfg_PLAY"
        ),
    },
    disable_env_checker=True,
)

