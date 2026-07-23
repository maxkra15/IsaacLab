# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RBY1DF waterhose task using Newton proxy coupling and bimanual XR teleoperation."""

from isaaclab.utils.configclass import configclass

from ...waterhose_env_cfg import WaterhoseProxyTeleopEnvCfg


@configclass
class WaterhoseTeleopEnvCfg(WaterhoseProxyTeleopEnvCfg):
    """RBY1DF waterhose task with absolute two-wrist Newton IK for Apple Vision Pro."""
