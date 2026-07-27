# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##


# Two-way proxy coupling: finite-mass gripper proxies exchange force with the cable in both
# directions. Replicated multi-env smoke runs work, but the coupled VBD/contact workload is
# throughput-bound; keep interactive runs at one environment.
gym.register(
    id="Isaac-Waterhose-Coupled-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.coupled_env_cfg:WaterhoseCoupledEnvCfg",
    },
)

gym.register(
    id="Isaac-Waterhose-Coupled-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teleop_env_cfg:WaterhoseTeleopEnvCfg",
    },
)

gym.register(
    id="Isaac-Waterhose-Coupled-Mimic-v0",
    entry_point="isaaclab_tasks.contrib.waterhose.waterhose_mimic_env:WaterhoseMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:WaterhoseMimicEnvCfg",
    },
)
