# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental ADMM cross-solver coupling config for the waterhose robot demo."""

from __future__ import annotations

from pathlib import Path

from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import NewtonCfg

from .coupled_env_cfg import WaterhoseCoupledSceneCfg, WaterhoseRobotDemoCoupledEnvCfg
from .coupled_manager import WaterhoseAdmmSolverCfg


_DEFAULT_ASSET_ROOT = str(
    Path(__file__).resolve().parents[5] / "isaaclab_assets" / "data" / "WaterhoseDemo"
)


@configclass
class WaterhoseRobotDemoAdmmEnvCfg(WaterhoseRobotDemoCoupledEnvCfg):
    """Experimental ADMM cross-solver contact coupling.

    The robot (MuJoCo) and cable (VBD) are stepped as separate solvers and
    their gripper/cable contact is reconciled by linearized ADMM each step.
    This is stiffer than the one-way proxy path and can go unstable on first
    gripper contact; prefer the one-way config for stable runs.
    """

    scene: WaterhoseCoupledSceneCfg = WaterhoseCoupledSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 100.0,
        render_interval=1,
        physics=NewtonCfg(
            solver_cfg=WaterhoseAdmmSolverCfg(asset_root=_DEFAULT_ASSET_ROOT),
            num_substeps=10,
            use_cuda_graph=True,
        ),
    )
