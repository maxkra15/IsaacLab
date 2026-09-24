# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO configuration for the MPM kinetic foundry task."""

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.franka_pour.config.franka.agents.rsl_rl_ppo_cfg import (
    FrankaPourResetDatasetPPORunnerCfg,
)


@configclass
class KineticFoundryPPORunnerCfg(FrankaPourResetDatasetPPORunnerCfg):
    """Train the same pouring policy in the foundry scene."""

    experiment_name = "kinetic_foundry"
    run_name = "mpm_pour"
