# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RBY1DF waterhose task using Newton proxy coupling and native IsaacLab teleop devices."""

from isaaclab.utils.configclass import configclass

from ...waterhose_env_cfg import WaterhoseProxyTeleopEnvCfg


@configclass
class WaterhoseTeleopEnvCfg(WaterhoseProxyTeleopEnvCfg):
    """RBY1DF waterhose task with relative differential IK for keyboard and SpaceMouse teleop."""
