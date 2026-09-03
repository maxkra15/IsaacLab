# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone Rizon4s Sharpa cable teleoperation task."""

import gymnasium as gym

RIZON_SHARPA_CABLE_TASK_ID = "IsaacContrib-Rizon-Sharpa-Hanging-RJ45-XR-Teleop"

gym.register(
    id=RIZON_SHARPA_CABLE_TASK_ID,
    entry_point="isaaclab_tasks.contrib.rizon_sharpa_cable.env:RizonSharpaCableEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.rizon_sharpa_cable.env_cfg:RizonSharpaCableEnvCfg",
    },
)

__all__ = ["RIZON_SHARPA_CABLE_TASK_ID"]
