# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RBY1DF waterhose task using Newton ADMM coupling."""

from isaaclab.utils.configclass import configclass

from ...waterhose_env_cfg import WaterhoseAdmmIkEnvCfg


@configclass
class WaterhoseAdmmEnvCfg(WaterhoseAdmmIkEnvCfg):
    """Client-facing RBY1DF waterhose task using Newton ADMM coupling."""
