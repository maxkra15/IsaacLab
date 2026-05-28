# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental two-way proxy coupling config for the waterhose robot demo."""

from __future__ import annotations

from pathlib import Path

from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import NewtonCfg

from .coupled_env_cfg import WaterhoseRobotDemoCoupledEnvCfg
from .coupled_manager import WaterhoseTwoWaySolverCfg


_DEFAULT_ASSET_ROOT = str(
    Path(__file__).resolve().parents[5] / "isaaclab_assets" / "data" / "WaterhoseDemo"
)


@configclass
class WaterhoseRobotDemoTwoWayEnvCfg(WaterhoseRobotDemoCoupledEnvCfg):
    """Experimental two-way proxy coupling.

    Same embedded gripper proxies as the one-way config, but harvested proxy
    contact wrenches feed back into the MuJoCo robot. Newton applies the full
    wrench (including tangential friction), so the robot reacts more strongly
    than the one-way default. Treat as experimental.
    """

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 100.0,
        render_interval=1,
        physics=NewtonCfg(
            solver_cfg=WaterhoseTwoWaySolverCfg(asset_root=_DEFAULT_ASSET_ROOT),
            num_substeps=10,
            use_cuda_graph=True,
        ),
    )
