# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One-way coupled-manager configuration for the waterhose robot demo."""

from __future__ import annotations

from pathlib import Path

from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import NewtonCfg

from .admm_env_cfg import (
    ActionsCfg,
    EmptyManagerCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
    WaterhoseAdmmSceneCfg,
)
from .admm_manager import WaterhoseOneWaySolverCfg


_DEFAULT_ASSET_ROOT = str(
    Path(__file__).resolve().parents[5] / "isaaclab_assets" / "data" / "WaterhoseDemo"
)


@configclass
class WaterhoseRobotDemoOneWayEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based waterhose task using one-way Newton proxy coupling."""

    scene: WaterhoseAdmmSceneCfg = WaterhoseAdmmSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EmptyManagerCfg = EmptyManagerCfg()
    curriculum: EmptyManagerCfg = EmptyManagerCfg()
    commands: EmptyManagerCfg = EmptyManagerCfg()

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 100.0,
        render_interval=1,
        physics=NewtonCfg(
            solver_cfg=WaterhoseOneWaySolverCfg(asset_root=_DEFAULT_ASSET_ROOT),
            num_substeps=10,
            use_cuda_graph=True,
        ),
    )
    viewer = ViewerCfg(eye=(-2.55, -7.1, 2.3), lookat=(0.55, -0.42, 0.9))

    episode_length_s = 30.0
    decimation = 1

    def __post_init__(self):
        self.sim.physics.solver_cfg.num_envs = int(self.scene.num_envs)
        self.sim.physics.solver_cfg.env_spacing = float(self.scene.env_spacing)
        self.sim.dt = 1.0 / 100.0
        self.sim.render_interval = self.decimation
