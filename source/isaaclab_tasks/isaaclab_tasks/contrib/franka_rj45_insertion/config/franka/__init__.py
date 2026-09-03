# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the Franka RJ45 insertion task."""

import gymnasium as gym

from . import agents

gym.register(
    id="IsaacContrib-Franka-RJ45-Insertion",
    entry_point="isaaclab_tasks.contrib.franka_rj45_insertion.rj45_env:FrankaRJ45InsertionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ("isaaclab_tasks.contrib.franka_rj45_insertion.rj45_env_cfg:FrankaRJ45InsertionEnvCfg"),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaRJ45InsertionPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Franka-RJ45-Dual-Rack-Insert",
    entry_point="isaaclab_tasks.contrib.franka_rj45_insertion.dual_rack_env:FrankaRJ45DualRackInsertEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.contrib.franka_rj45_insertion.dual_rack_env_cfg:FrankaRJ45DualRackInsertEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.dual_rack_rsl_rl_ppo_cfg:FrankaRJ45DualRackInsertPPORunnerCfg"),
    },
)

gym.register(
    id="IsaacContrib-Franka-RJ45-GB300-Insert",
    entry_point="isaaclab_tasks.contrib.franka_rj45_insertion.gb300_env:FrankaRJ45Gb300InsertEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.contrib.franka_rj45_insertion.gb300_env_cfg:FrankaRJ45Gb300InsertEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.gb300_rsl_rl_ppo_cfg:FrankaRJ45Gb300InsertPPORunnerCfg"),
    },
)

gym.register(
    id="IsaacContrib-Franka-RJ45-Pick-Insert",
    entry_point="isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_env:FrankaRJ45PickInsertEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_env_cfg:FrankaRJ45PickInsertEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.pick_insert_rsl_rl_ppo_cfg:FrankaRJ45PickInsertPPORunnerCfg"),
    },
)
