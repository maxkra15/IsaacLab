# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ADMM coupled-manager configuration for the waterhose robot demo."""

from __future__ import annotations

from pathlib import Path

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import NewtonCfg

from . import admm_mdp
from .actions import AdmmScriptedDemoAction, AdmmScriptedDemoActionCfg
from .admm_manager import WaterhoseAdmmSolverCfg


_DEFAULT_ASSET_ROOT = str(
    Path(__file__).resolve().parents[5] / "isaaclab_assets" / "data" / "WaterhoseDemo"
)


@configclass
class ActionsCfg:
    """Action terms for scripted and teleoperated demo control."""

    demo = AdmmScriptedDemoActionCfg(class_type=AdmmScriptedDemoAction)


@configclass
class EmptyManagerCfg:
    """Empty manager configuration used when the task has no terms."""

    pass


@configclass
class ObservationsCfg:
    """Observations exposing the waterhose task state."""

    @configclass
    class PolicyCfg(ObsGroup):
        sim_time = ObsTerm(func=admm_mdp.sim_time)
        phase = ObsTerm(func=admm_mdp.phase)
        right_ee_pose = ObsTerm(func=admm_mdp.right_ee_pose)
        plug_pose = ObsTerm(func=admm_mdp.plug_pose)
        tip_pose = ObsTerm(func=admm_mdp.tip_pose)
        finite = ObsTerm(func=admm_mdp.finite)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Placeholder reward terms for manager compatibility."""

    alive = RewTerm(func=admm_mdp.alive, weight=0.0)


@configclass
class TerminationsCfg:
    """Task termination terms."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    failed = DoneTerm(func=admm_mdp.failed)


@configclass
class WaterhoseAdmmSceneCfg(InteractiveSceneCfg):
    """Scene shell for the task-local Newton ADMM model."""

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class WaterhoseRobotDemoAdmmEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based waterhose task using Newton ADMM coupling."""

    scene: WaterhoseAdmmSceneCfg = WaterhoseAdmmSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
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
            solver_cfg=WaterhoseAdmmSolverCfg(asset_root=_DEFAULT_ASSET_ROOT),
            num_substeps=10,
            use_cuda_graph=True,
        ),
    )
    viewer = ViewerCfg(eye=(-2.55, -7.1, 2.3), lookat=(0.55, -0.42, 0.9))

    episode_length_s = 30.0
    decimation = 1

    def __post_init__(self):
        self.scene.num_envs = 1
        self.sim.dt = 1.0 / 100.0
        self.sim.render_interval = self.decimation
