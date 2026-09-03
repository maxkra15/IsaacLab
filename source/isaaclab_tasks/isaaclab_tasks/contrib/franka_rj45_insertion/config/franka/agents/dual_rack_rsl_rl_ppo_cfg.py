# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO configuration for the two-ended RJ45 rack-routing task."""

from isaaclab.utils.configclass import configclass

from .pick_insert_rsl_rl_ppo_cfg import FrankaRJ45PickInsertPPORunnerCfg


@configclass
class FrankaRJ45DualRackInsertPPORunnerCfg(FrankaRJ45PickInsertPPORunnerCfg):
    """Use the proven six-phase PPO schedule with a distinct checkpoint ABI."""

    experiment_name = "franka_rj45_dual_rack_insert"
    run_name = "two_ended_cable_t_slot_workcell"


__all__ = ["FrankaRJ45DualRackInsertPPORunnerCfg"]
