# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the Franka two-bowl pour MPM task."""

import gymnasium as gym

from . import agents

_ENV = "isaaclab_tasks.contrib.franka_pour.pour_env:FrankaPourEnv"
_CFG = "isaaclab_tasks.contrib.franka_pour.pour_env_cfg"
_AGENT = f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPourPPORunnerCfg"

gym.register(
    id="Isaac-Pour-Franka-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}:FrankaPourEnvCfg",
        "rsl_rl_cfg_entry_point": _AGENT,
    },
)

gym.register(
    id="Isaac-Pour-Franka-Teleop-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}:FrankaPourEnvCfg_TELEOP",
    },
)

gym.register(
    id="Isaac-Pour-Franka-Play-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}:FrankaPourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _AGENT,
    },
)
