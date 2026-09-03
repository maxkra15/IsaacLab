# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO configuration for the native-port GB300 RJ45 task."""

from isaaclab.utils.configclass import configclass

from .dual_rack_rsl_rl_ppo_cfg import FrankaRJ45DualRackInsertPPORunnerCfg


@configclass
class FrankaRJ45Gb300InsertPPORunnerCfg(FrankaRJ45DualRackInsertPPORunnerCfg):
    """Retain the dual-ended cable ABI under a separate experiment name."""

    experiment_name = "franka_rj45_gb300_insert"
    run_name = "simready_gb300_native_sn2201_port_bank"


__all__ = ["FrankaRJ45Gb300InsertPPORunnerCfg"]
