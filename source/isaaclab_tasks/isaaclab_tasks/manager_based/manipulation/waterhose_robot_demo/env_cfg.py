# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the waterhose robot demo task."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import NewtonCfg

from . import mdp
from .actions import ScriptedDemoAction, ScriptedDemoActionCfg
from .manager import WaterhoseNewtonSolverCfg
from .recorders import WaterhoseActionStateRecorderManagerCfg
from .teleop import WaterhoseSpaceMouseCfg


_DEFAULT_ASSET_ROOT = str(
    Path(__file__).resolve().parents[5] / "isaaclab_assets" / "data" / "WaterhoseDemo"
)


@configclass
class ActionsCfg:
    """Action terms for the scripted demo."""

    demo = ScriptedDemoActionCfg(class_type=ScriptedDemoAction)


@configclass
class ObservationsCfg:
    """Observations exposing key demo state."""

    @configclass
    class PolicyCfg(ObsGroup):
        sim_time = ObsTerm(func=mdp.sim_time)
        phase = ObsTerm(func=mdp.phase)
        right_ee_pose = ObsTerm(func=mdp.right_ee_pose)
        plug_pose = ObsTerm(func=mdp.plug_pose)
        tip_pose = ObsTerm(func=mdp.tip_pose)
        finite = ObsTerm(func=mdp.finite)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Placeholder rewards for manager compatibility."""

    alive = RewTerm(func=mdp.alive, weight=0.0)


@configclass
class TerminationsCfg:
    """Terminate when the scripted rollout finishes or an optional step cap is reached."""

    success = DoneTerm(func=mdp.success)
    demo_done = DoneTerm(func=mdp.done)


@configclass
class EventCfg:
    """Reset hooks for the local Newton simulation."""

    reset_demo = EventTerm(func=mdp.reset_demo, mode="reset")


@configclass
class WaterhoseSceneCfg(InteractiveSceneCfg):
    """Scene-level assets for the waterhose robot demo."""

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class WaterhoseRobotDemoEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-style waterhose robot demo using a local Newton simulation."""

    scene: WaterhoseSceneCfg = WaterhoseSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None
    commands = None

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 100.0,
        render_interval=1,
        physics=NewtonCfg(
            solver_cfg=WaterhoseNewtonSolverCfg(asset_root=_DEFAULT_ASSET_ROOT),
            num_substeps=1,
            use_cuda_graph=False,
        ),
    )
    viewer = ViewerCfg(eye=(-2.55, -7.1, 2.3), lookat=(0.55, -0.42, 0.9))

    episode_length_s = 30.0
    decimation = 1

    # Safety bound for scripted rollouts. 0 means run until the controller's DONE phase.
    max_demo_steps: int = 0

    def make_recorder_manager_cfg(self) -> WaterhoseActionStateRecorderManagerCfg:
        """Return the recorder config used by standard demo-recording scripts."""

        return WaterhoseActionStateRecorderManagerCfg()

    def __post_init__(self):
        self.scene.num_envs = 1
        self.sim.dt = 1.0 / 100.0
        self.sim.render_interval = self.decimation
        self.sim.physics.solver_cfg.max_demo_steps = int(self.max_demo_steps)
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.02,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": WaterhoseSpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.15,
                    sim_device=self.sim.device,
                ),
            }
        )
