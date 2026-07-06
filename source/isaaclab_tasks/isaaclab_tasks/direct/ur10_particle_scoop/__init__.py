# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10 particle-scooping direct RL environment."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-UR10-Particle-Scoop-Direct-v0",
    entry_point=f"{__name__}.ur10_particle_scoop_env:UR10ParticleScoopEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10_particle_scoop_env_cfg:UR10ParticleScoopEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10ParticleScoopPPORunnerCfg",
    },
)
