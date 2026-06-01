# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


_PKG = "isaaclab_tasks.manager_based.manipulation.waterhose_robot_demo"
_WATERHOSE_PKG = "isaaclab_tasks.manager_based.manipulation.waterhose"
_WATERHOSE_AGENTS = f"{_WATERHOSE_PKG}.agents"


gym.register(
    id="Isaac-Waterhose-Robot-Demo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_PKG}.coupled_env_cfg:WaterhoseRobotDemoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WaterhoseRobotDemoPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Waterhose-Robot-Demo-Mimic-v0",
    entry_point=f"{_PKG}.mimic_env:WaterhoseRobotDemoMimicEnv",
    kwargs={"env_cfg_entry_point": f"{_PKG}.mimic_env_cfg:WaterhoseRobotDemoMimicEnvCfg"},
    disable_env_checker=True,
)

# ADMM-coupled task: use the same scene-config based setup as the canonical
# Isaac-Waterhose-v0 task. The stable client demo remains
# Isaac-Waterhose-Robot-Demo-v0.
gym.register(
    id="Isaac-Waterhose-Robot-Demo-Coupled-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_WATERHOSE_PKG}.waterhose_env_cfg:WaterhoseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_WATERHOSE_AGENTS}.rsl_rl_ppo_cfg:WaterhosePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Waterhose-Robot-Demo-Proxy-Coupled-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_WATERHOSE_PKG}.waterhose_env_cfg:WaterhoseProxyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_WATERHOSE_AGENTS}.rsl_rl_ppo_cfg:WaterhosePPORunnerCfg",
    },
    disable_env_checker=True,
)
