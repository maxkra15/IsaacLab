# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared config and base env for the coupled waterhose robot demo.

Variant configs (one-way, two-way, ADMM) subclass
:class:`WaterhoseRobotDemoCoupledEnvCfg` and only swap the Newton coupling.
"""

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
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import NewtonCfg
from isaaclab_teleop.xr_cfg import XrCfg

from . import coupled_mdp
from .actions import CoupledScriptedDemoAction, CoupledScriptedDemoActionCfg
from .coupled_manager import WaterhoseOneWaySolverCfg
from .isaac_teleop import make_waterhose_isaac_teleop_cfg
from .teleop import WaterhoseSpaceMouseCfg


_DEFAULT_ASSET_ROOT = str(
    Path(__file__).resolve().parents[5] / "isaaclab_assets" / "data" / "WaterhoseDemo"
)


@configclass
class ActionsCfg:
    """Action terms for scripted and teleoperated demo control."""

    demo = CoupledScriptedDemoActionCfg(class_type=CoupledScriptedDemoAction)


@configclass
class EmptyManagerCfg:
    """Empty manager configuration used when the task has no terms."""

    pass


@configclass
class ObservationsCfg:
    """Observations exposing the waterhose task state."""

    @configclass
    class PolicyCfg(ObsGroup):
        sim_time = ObsTerm(func=coupled_mdp.sim_time)
        phase = ObsTerm(func=coupled_mdp.phase)
        right_ee_pose = ObsTerm(func=coupled_mdp.right_ee_pose)
        plug_pose = ObsTerm(func=coupled_mdp.plug_pose)
        tip_pose = ObsTerm(func=coupled_mdp.tip_pose)
        finite = ObsTerm(func=coupled_mdp.finite)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Placeholder reward terms for manager compatibility."""

    alive = RewTerm(func=coupled_mdp.alive, weight=0.0)


@configclass
class TerminationsCfg:
    """Task termination terms."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    failed = DoneTerm(func=coupled_mdp.failed)


@configclass
class WaterhoseCoupledSceneCfg(InteractiveSceneCfg):
    """Scene shell for the task-local Newton coupled model."""

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class WaterhoseRobotDemoCoupledEnvCfg(ManagerBasedRLEnvCfg):
    """Base manager-based waterhose task on the Newton coupled manager.

    Defaults to the stable one-way proxy coupling. Variants override
    ``sim.physics.solver_cfg`` (and the scene where needed) to select a
    different Newton coupling.
    """

    scene: WaterhoseCoupledSceneCfg = WaterhoseCoupledSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
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
    xr: XrCfg = XrCfg(anchor_pos=(0.55, -0.42, 0.9), anchor_rot=(0.0, 0.0, 0.0, 1.0))

    episode_length_s = 30.0
    decimation = 1
    isaac_teleop: object | None = None

    def __post_init__(self):
        solver_cfg = self.sim.physics.solver_cfg
        if hasattr(solver_cfg, "num_envs"):
            solver_cfg.num_envs = int(self.scene.num_envs)
        if hasattr(solver_cfg, "env_spacing"):
            solver_cfg.env_spacing = float(self.scene.env_spacing)
        self.sim.dt = 1.0 / 100.0
        self.sim.render_interval = self.decimation
        # Native Isaac Lab teleop devices (keyboard/SpaceMouse) + optional XR via
        # IsaacTeleop, so both coupled setups run under scripts/.../teleop_se3_agent.py.
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(pos_sensitivity=0.02, rot_sensitivity=0.05, sim_device="cpu"),
                "spacemouse": WaterhoseSpaceMouseCfg(pos_sensitivity=0.05, rot_sensitivity=0.15, sim_device="cpu"),
            }
        )
        self.isaac_teleop = make_waterhose_isaac_teleop_cfg(self.sim.device, self.xr)
