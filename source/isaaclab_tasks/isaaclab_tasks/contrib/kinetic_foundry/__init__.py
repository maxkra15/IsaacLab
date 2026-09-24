# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A dressed MPM pouring scene with a reusable manager-based PPO task."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-Kinetic-Foundry",
    entry_point="isaaclab_tasks.contrib.franka_pour.pour_env:FrankaPourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.kinetic_foundry.foundry_env_cfg:KineticFoundryEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.contrib.kinetic_foundry.agents.rsl_rl_ppo_cfg:KineticFoundryPPORunnerCfg"
        ),
    },
)
