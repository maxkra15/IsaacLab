# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coupled-rigid/VBD textile atelier with two robot arms."""

import gymnasium as gym

from . import agents

gym.register(
    id="IsaacContrib-Textile-Atelier-Kuka-GR1T2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.textile_atelier_env_cfg:TextileAtelierEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TextileAtelierPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
