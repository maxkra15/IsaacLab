# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""KUKA-Allegro one-ball juggling task registration."""

import gymnasium as gym

from . import agents

gym.register(
    id="IsaacContrib-Juggle-Ball-KukaAllegro-RL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.meter_juggle_env_cfg:KukaAllegroJuggleRLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KukaAllegroJugglePPORunnerCfg",
    },
    disable_env_checker=True,
)
