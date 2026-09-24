# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""KUKA-Allegro and Fourier GR1T2 rigid-ball relay task."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-RelayJuggle-KukaAllegro-GR1T2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.relay_env_cfg:RelayJuggleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:RelayJugglePPORunnerCfg",
    },
    disable_env_checker=True,
)
