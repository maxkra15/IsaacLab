# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


# Two-way proxy coupling (the gripper proxies are finite-mass bodies that exchange force with the
# cable in both directions). This is the default customer demo path. Replicated multi-env smoke
# runs work, but the coupled VBD/proxy-contact workload is throughput-bound rather than linearly
# scaling with env count; keep XR/demo runs at one env and profile before using it for RL batches.
gym.register(
    id="Isaac-Waterhose-Coupled-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.coupled_env_cfg:WaterhoseCoupledEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WaterhosePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Waterhose-Admm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.admm_env_cfg:WaterhoseAdmmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WaterhosePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Waterhose-Coupled-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teleop_env_cfg:WaterhoseTeleopEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WaterhosePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Waterhose-Coupled-Teleop-Mimic-v0",
    entry_point="isaaclab_tasks.contrib.waterhose.waterhose_mimic_env:WaterhoseMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:WaterhoseMimicEnvCfg",
    },
)
