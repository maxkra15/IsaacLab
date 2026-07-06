# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RBY1DF waterhose task using Newton proxy coupling and Isaac Lab Mimic metadata."""

from isaaclab.utils.configclass import configclass

from ...waterhose_env_cfg import WaterhoseMimicEnvCfg as _WaterhoseMimicEnvCfg


@configclass
class WaterhoseMimicEnvCfg(_WaterhoseMimicEnvCfg):
    """RBY1DF waterhose teleop task with Mimic-compatible datagen metadata."""
