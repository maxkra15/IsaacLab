# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical one-way coupled config for the waterhose robot demo."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from .coupled_env_cfg import WaterhoseRobotDemoCoupledEnvCfg


@configclass
class WaterhoseRobotDemoOneWayEnvCfg(WaterhoseRobotDemoCoupledEnvCfg):
    """One-way proxy coupling (recommended, stable).

    The MuJoCo robot drives kinematic gripper proxies embedded in the VBD
    cable world; cable/gripper contact is resolved entirely inside VBD and the
    harvested proxy feedback is discarded, so the cable cannot push the robot.
    This mirrors the proven reference demo's contact architecture and is the
    stable default for the coupled waterhose task.
    """
