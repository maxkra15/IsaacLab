# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-style task config for the literal Newton waterhose robot success demo."""

from __future__ import annotations

from isaaclab_newton.physics import NewtonCfg, XPBDSolverCfg

from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from . import mdp
from .actions import ReferenceDemoNoOpAction, ReferenceDemoNoOpActionCfg


@configclass
class ActionsCfg:
    """No-op action term; the reference demo state machine owns control."""

    demo = ReferenceDemoNoOpActionCfg(class_type=ReferenceDemoNoOpAction)


@configclass
class ObservationsCfg:
    """Observations exposing key reference-demo state."""

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
    """Terminate when reference demo finishes or max frame count is reached."""

    demo_done = DoneTerm(func=mdp.done)


@configclass
class EventCfg:
    """No reset events; the reference Newton demo owns all state."""

    pass


@configclass
class WaterhoseRobotDemoEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-style wrapper around `example_waterhose_scene2_insert_extract_success.py`."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None
    commands = None

    # IsaacLab still owns an app/sim context for manager lifecycle, but the
    # reference Newton Example owns the actual robot/cable simulation.
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 100.0,
        render_interval=1,
        # Dummy kitless physics manager for IsaacLab lifecycle only. The real
        # robot/cable simulation is owned by the reference Newton Example.
        # XPBD tolerates the empty IsaacLab scene; the default MJWarp solver
        # does not because it requires at least one joint.
        physics=NewtonCfg(solver_cfg=XPBDSolverCfg(), use_cuda_graph=False),
    )
    viewer = ViewerCfg(eye=(-2.55, -7.1, 2.3), lookat=(0.55, -0.42, 0.9))

    episode_length_s = 30.0
    decimation = 1

    # Path to the local Newton checkout containing the reference script.
    reference_newton_root: str = "/home/maximiliank/Work/newton"
    reference_module: str = "newton.examples.cable_robot.example_waterhose_scene2_insert_extract_success"
    reference_viewer: str = "gl"
    reference_primary_view: str = "mujoco"
    reference_headless: bool = False
    reference_num_frames: int = 100000
    reference_quiet: bool = True
    reference_device: str | None = None

    # Safety bound for scripted rollouts. 0 means run until state machine DONE.
    max_demo_steps: int = 0

    def __post_init__(self):
        self.scene.num_envs = 1
        self.sim.dt = 1.0 / 100.0
        self.sim.render_interval = self.decimation


@configclass
class WaterhoseRobotDemoEnvCfg_PLAY(WaterhoseRobotDemoEnvCfg):
    """Play/debug variant using Newton GL + MuJoCo primary view."""

    def __post_init__(self):
        super().__post_init__()
        self.reference_viewer = "gl"
        self.reference_primary_view = "mujoco"
        self.reference_headless = False

