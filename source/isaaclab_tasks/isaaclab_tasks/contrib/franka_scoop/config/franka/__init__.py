# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the Franka scoop two-container MPM transfer task."""

import gymnasium as gym

from . import agents

_ENV = "isaaclab_tasks.contrib.franka_scoop.scoop_env:FrankaScoopEnv"
_CFG = "isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg"
_AGENT = f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaScoopPPORunnerCfg"

gym.register(
    id="Isaac-Scoop-Franka-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}:FrankaScoopEnvCfg",
        "rsl_rl_cfg_entry_point": _AGENT,
    },
)

gym.register(
    id="Isaac-Scoop-Franka-Teleop-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}:FrankaScoopEnvCfg_TELEOP",
        "rsl_rl_cfg_entry_point": _AGENT,
    },
)

gym.register(
    id="Isaac-Scoop-Franka-Play-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}:FrankaScoopEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _AGENT,
    },
)
