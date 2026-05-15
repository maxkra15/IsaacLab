# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_newton.physics import NewtonCfg
from isaaclab_newton.physics.newton_manager_cfg import NewtonSolverCfg

from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

from . import mdp
from .actions import NewtonTaskSpaceIKAction, NewtonTaskSpaceIKActionCfg


@configclass
class ActionsCfg:
    """Action terms for task-space waterhose control."""

    task_space = NewtonTaskSpaceIKActionCfg(class_type=NewtonTaskSpaceIKAction)


@configclass
class ObservationsCfg:
    """Observation terms for waterhose manipulation."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        eef_pos = ObsTerm(func=mdp.eef_pos)
        eef_quat = ObsTerm(func=mdp.eef_quat)
        plug_pos = ObsTerm(func=mdp.plug_pos)
        plug_quat = ObsTerm(func=mdp.plug_quat)
        tip_pos = ObsTerm(func=mdp.tip_pos)
        tip_quat = ObsTerm(func=mdp.tip_quat)
        socket_pose = ObsTerm(func=mdp.socket_pose)
        alignment = ObsTerm(func=mdp.alignment)
        proxy_wrench_norm = ObsTerm(func=mdp.proxy_wrench_norm)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        approach = ObsTerm(func=mdp.approach_done)
        grasp = ObsTerm(func=mdp.grasp_done)
        align = ObsTerm(func=mdp.align_done)
        insert = ObsTerm(func=mdp.insert_done)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class RewardsCfg:
    """Reward terms for the waterhose task."""

    reach_hose = RewTerm(func=mdp.reach_hose, weight=1.0)
    align_tip = RewTerm(func=mdp.align_tip, weight=2.0)
    insert_tip = RewTerm(func=mdp.insert_tip, weight=30.0)
    success_bonus = RewTerm(func=mdp.success_bonus, weight=20.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    """Termination terms for the waterhose task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.success)


@configclass
class EventCfg:
    """No USD reset events are needed; Newton owns all simulated state."""

    pass


@configclass
class RBY1DFWaterhoseEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based Newton proxy-coupled RBY1 waterhose task."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None
    commands = None

    # The concrete coupled solver cfg is built from the waterhose assets in RBY1DFWaterhoseEnv.__init__.
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 300.0,
        render_interval=1,
        physics=NewtonCfg(solver_cfg=NewtonSolverCfg(), num_substeps=10, use_cuda_graph=True),
    )
    viewer = ViewerCfg(eye=(-1.2, -2.8, 1.6), lookat=(0.55, -0.42, 0.55))

    episode_length_s = 30.0
    decimation = 1

    asset_root: str = f"{ISAACLAB_ASSETS_DATA_DIR}/Props/Waterhose"
    robot_urdf: str | None = None
    scene_usd: str | None = None
    cable_usd: str | None = None
    cable_prims: str = "/World/cable001/curve_0,/World/cable002/curve_0"
    cable_prim: str | None = None
    fps: float = 300.0
    sim_substeps: int = 10
    rigid_substeps: int = 1
    proxy_iterations: int = 1
    vbd_iterations: int = 15
    disable_cuda_graph: bool = False
    hose_radius: float = 0.003
    gripper_drive_scale: float = 0.5
    grasp_friction: float = 1.0e6
    grasp_margin: float = 0.001
    grasp_contact_ke: float = 2.0e5
    success_lateral_threshold: float = 0.0008
    success_axis_cosine: float = -0.995
    success_insert_depth: float = 0.025
    insert_start_depth: float = 0.005

    def __post_init__(self):
        self.scene.num_envs = max(1, int(self.scene.num_envs))
        self.sim.dt = 1.0 / float(self.fps)
        self.sim.render_interval = self.decimation
