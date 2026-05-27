# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

# Newton's USD importer is sensitive to import order, so initialize it before
# the broader Isaac Lab stack is imported below.  Kit launches are the
# exception: SimulationApp must start before Newton/PXR-facing imports.
from . import waterhose_core as core  # isort: skip
from .launch import should_defer_newton_import  # isort: skip

if not should_defer_newton_import():
    core.import_newton_dependencies()

from isaaclab_newton.physics import NewtonCfg

from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

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
    """No reset events are needed; Newton owns the waterhose state."""

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
    # Keep `dt = 1/100` from the success demo; the reference cable_pendulum
    # uses 1/60 but its cable / plug regime is much softer than ours, and
    # the lower outer-step rate destabilised our stiff cable + heavy plug.
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 100.0,
        render_interval=1,
        physics=NewtonCfg(num_substeps=10, use_cuda_graph=True),
    )
    viewer = ViewerCfg(eye=(-1.2, -2.8, 1.6), lookat=(0.55, -0.42, 0.55))

    episode_length_s = 30.0
    decimation = 1

    asset_root: str = str(core.default_waterhose_asset_root())
    robot_urdf: str | None = None
    # None uses the authored Cable008 scene.
    scene_usd: str | None = None
    # None uses Cable008/curve/cable_SRA_curve03.usda for both authored cable curves.
    cable_usds: str | None = None
    cable_usd: str | None = None
    cable_prims: str | None = None
    cable_prim: str | None = None
    sim_substeps: int = 10
    rigid_substeps: int = 1
    # Linearised ADMM cross-solver coupling. The MJC robot entry and the
    # VBD cable / scene entry exchange Lagrange-multiplier forces on the
    # configured contact pair every iteration; baumgarte stabilises the
    # contact-distance constraint position-drift.
    admm_iterations: int = 5
    admm_rho: float = 30.0
    admm_gamma: float = 0.1
    admm_baumgarte: float = 0.005
    admm_contact_distance: float = 0.003
    admm_detection_margin: float = 0.01
    # Fixed-k VBD (no AVBD ramping) with 15 iterations matches the
    # success demo on our stiff (stretch=1e6) cable. The reference uses
    # 20+ramping but for a much softer cable; ramping cable-joint
    # stiffness from 1e2 toward 1e6 on a heavy chain shakes the cable
    # apart on early iterations.
    vbd_iterations: int = 15
    rigid_contact_max: int = 100000
    mujoco_iterations: int = 20
    mujoco_ls_iterations: int = 10
    mujoco_ls_parallel: bool = True
    mujoco_impratio: float = 1000.0
    mujoco_use_mujoco_contacts: bool = False
    disable_cuda_graph: bool = False
    hose_radius: float = 0.003
    gripper_drive_scale: float = 1.0
    robot_shape_margin: float = 0.0
    robot_shape_gap: float = 0.002
    robot_shape_ke: float = 5.0e4
    robot_shape_kd: float = 5.0e2
    robot_shape_mu: float = 2.0
    robot_joint_target_ke: float = 120000.0
    robot_joint_target_kd: float = 12000.0
    robot_joint_effort_limit: float = 10000.0
    robot_joint_armature: float = 0.2
    gripper_joint_target_ke: float = 10000.0
    gripper_joint_target_kd: float = 1000.0
    gripper_joint_effort_limit: float = 100000.0
    gripper_joint_armature: float = 0.5
    gripper_finger_target_ke: float = 500000.0
    gripper_finger_target_kd: float = 10000.0
    gripper_finger_effort_limit: float = 500000.0
    gripper_finger_armature: float = 0.5
    # ADMM applies Lagrange-multiplier forces consistently across solvers,
    # so the mu=1.0 default is what we want — the 1e6 we needed under
    # the lagged-proxy friction-drop hack is gone. Keeping ke aligned
    # with the cable/head pair (1e3) gives a symmetric gripper<->plug
    # contact instead of a 1e3-vs-1e5 mismatch.
    grasp_friction: float = 1.0
    grasp_margin: float = 0.001
    grasp_contact_ke: float = 1.0e3
    vbd_default_contact_ke: float = 1.0e3
    vbd_default_contact_kd: float = 0.0
    vbd_default_contact_margin: float = 0.001
    vbd_solver_friction_epsilon: float = 0.1
    vbd_rigid_contact_hard: bool = False
    vbd_rigid_contact_buffer_size: int = 1024
    vbd_rigid_body_particle_contact_buffer_size: int = 1
    # Cable density 1000 kg/m^3 (matches the success demo and keeps the
    # cable's spring period above substep_dt for our stretch=1e6 cable).
    # mu=1.0 is ADMM-friendly — large mu values produce big Lagrange
    # multipliers that destabilise the cross-solver solve.
    vbd_cable_density: float = 1000.0
    vbd_cable_mu: float = 1.0
    vbd_cable_margin: float = 0.0
    vbd_cable_gap: float = 0.002
    vbd_static_margin: float = 0.0
    vbd_static_gap: float = 0.002
    # Head/plug mesh tuning. Keep the success-demo mass (~3 g from
    # density * volume) and ke=1e3 — bumping ke to 1e5 over a 3 g body
    # pushed ke/m near the stability bound for substep_dt and made the
    # plug bounce out of the jaw. mu=1.0 stays (ADMM-friendly).
    vbd_head_mass: float = 0.0
    vbd_head_mesh_ke: float = 1.0e3
    vbd_head_mesh_kd: float = 0.0
    vbd_head_mesh_mu: float = 1.0
    vbd_head_mesh_margin: float = 0.0
    vbd_head_mesh_xy_scale: float = 0.95
    vbd_static_mesh_use_sdf: bool = True
    # SDF query cost scales with res^3 per shape; the fridge scene has
    # ~250 mesh shapes so halving from 64 -> 32 is a major perf win and
    # has no visible impact on cable<->scene contact behavior.
    vbd_static_mesh_sdf_max_resolution: int = 32
    # Static scene collision representation for cable contacts:
    #   - "proxy"   : 2 static boxes (tabletop + socket region). Default.
    #                 Avoids 247 convex-hull collisions from the fridge
    #                 USD's V-HACD authoring, giving ~100x fewer broad
    #                 phase pairs and 247 fewer SDF builds at startup.
    #   - "usd_sdf" : load `Cable008_Body.usda` colliders + build SDFs.
    #                 Use only if the cable must contact arbitrary
    #                 fridge geometry beyond the table + socket region.
    kit_static_contact_mode: str = "proxy"
    # When `kit_static_contact_mode="proxy"`, also load the fridge USD
    # purely for visualisation (every loaded shape has its COLLIDE
    # flags stripped, so the broad phase still only sees the proxy
    # boxes). Lets the Newton GL viewer render the fridge alongside
    # the cheap collision proxies. Disable for headless / null-viewer /
    # Kit-visualiser runs.
    kit_static_visual_meshes: bool = True
    vbd_near_tip_mu: float = 1.0e1
    vbd_far_tip_mu: float = 1.0e5
    vbd_ground_mu: float = 1.0e5
    # Fixed-k AVBD (beta=0) matches the success demo. Ramping joint
    # penalties on our stiff cable causes early-iteration wobble.
    vbd_rigid_avbd_beta: float = 0.0
    vbd_rigid_contact_history: bool = False
    vbd_rigid_contact_k_start: float = 1.0e2
    vbd_rigid_joint_linear_ke: float = 1.0e6
    vbd_rigid_joint_angular_ke: float = 1.0e6
    vbd_rigid_joint_linear_k_start: float = 1.0e2
    vbd_rigid_joint_angular_k_start: float = 1.0e1
    cable_stretch_stiffness: float = 1.0e6
    cable_stretch_damping: float = 1.0e-5
    cable_num_segments: int = 0
    cable_bend_stiffness: float = 2.0e1
    cable_bend_rigidity: float = 1.5e-1
    cable_bend_damping: float = 1.0e0
    success_lateral_threshold: float = 0.0008
    success_axis_cosine: float = -0.995
    success_insert_depth: float = 0.025
    insert_start_depth: float = 0.005

    def __post_init__(self):
        self.scene.num_envs = max(1, int(self.scene.num_envs))
        self.sync_waterhose_sim_cfg()

    def sync_waterhose_sim_cfg(self) -> None:
        """Synchronize derived Newton simulation settings from task cfg fields."""
        self.sim.render_interval = self.decimation
        self.sim.physics.num_substeps = int(self.sim_substeps)
        self.sim.physics.use_cuda_graph = not bool(self.disable_cuda_graph)

    def waterhose_scene_kwargs(self) -> dict[str, object]:
        """Return arguments used to build the Newton waterhose scene."""
        return {
            "fps": 1.0 / float(self.sim.dt),
            "num_envs": int(self.scene.num_envs),
            "env_spacing": float(self.scene.env_spacing),
            "asset_root": self.asset_root,
            "robot_urdf": self.robot_urdf,
            "scene_usd": self.scene_usd,
            "cable_usds": self.cable_usds,
            "cable_usd": self.cable_usd,
            "cable_prims": self.cable_prims,
            "cable_prim": self.cable_prim,
            "hose_radius": float(self.hose_radius),
            "gripper_drive_scale": float(self.gripper_drive_scale),
            "grasp_friction": float(self.grasp_friction),
            "grasp_margin": float(self.grasp_margin),
            "grasp_contact_ke": float(self.grasp_contact_ke),
            "sim_substeps": int(self.sim_substeps),
            "rigid_substeps": int(self.rigid_substeps),
            "admm_iterations": int(self.admm_iterations),
            "admm_rho": float(self.admm_rho),
            "admm_gamma": float(self.admm_gamma),
            "admm_baumgarte": float(self.admm_baumgarte),
            "admm_contact_distance": float(self.admm_contact_distance),
            "admm_detection_margin": float(self.admm_detection_margin),
            "vbd_iterations": int(self.vbd_iterations),
            "rigid_contact_max": int(self.rigid_contact_max),
            "mujoco_iterations": int(self.mujoco_iterations),
            "mujoco_ls_iterations": int(self.mujoco_ls_iterations),
            "mujoco_ls_parallel": bool(self.mujoco_ls_parallel),
            "mujoco_impratio": float(self.mujoco_impratio),
            "mujoco_use_mujoco_contacts": bool(self.mujoco_use_mujoco_contacts),
            "robot_shape_margin": float(self.robot_shape_margin),
            "robot_shape_gap": float(self.robot_shape_gap),
            "robot_shape_ke": float(self.robot_shape_ke),
            "robot_shape_kd": float(self.robot_shape_kd),
            "robot_shape_mu": float(self.robot_shape_mu),
            "robot_joint_target_ke": float(self.robot_joint_target_ke),
            "robot_joint_target_kd": float(self.robot_joint_target_kd),
            "robot_joint_effort_limit": float(self.robot_joint_effort_limit),
            "robot_joint_armature": float(self.robot_joint_armature),
            "gripper_joint_target_ke": float(self.gripper_joint_target_ke),
            "gripper_joint_target_kd": float(self.gripper_joint_target_kd),
            "gripper_joint_effort_limit": float(self.gripper_joint_effort_limit),
            "gripper_joint_armature": float(self.gripper_joint_armature),
            "gripper_finger_target_ke": float(self.gripper_finger_target_ke),
            "gripper_finger_target_kd": float(self.gripper_finger_target_kd),
            "gripper_finger_effort_limit": float(self.gripper_finger_effort_limit),
            "gripper_finger_armature": float(self.gripper_finger_armature),
            "vbd_default_contact_ke": float(self.vbd_default_contact_ke),
            "vbd_default_contact_kd": float(self.vbd_default_contact_kd),
            "vbd_default_contact_margin": float(self.vbd_default_contact_margin),
            "vbd_solver_friction_epsilon": float(self.vbd_solver_friction_epsilon),
            "vbd_rigid_contact_hard": bool(self.vbd_rigid_contact_hard),
            "vbd_rigid_contact_buffer_size": int(self.vbd_rigid_contact_buffer_size),
            "vbd_rigid_body_particle_contact_buffer_size": int(self.vbd_rigid_body_particle_contact_buffer_size),
            "vbd_cable_density": float(self.vbd_cable_density),
            "vbd_cable_mu": float(self.vbd_cable_mu),
            "vbd_cable_margin": float(self.vbd_cable_margin),
            "vbd_cable_gap": float(self.vbd_cable_gap),
            "vbd_static_margin": float(self.vbd_static_margin),
            "vbd_static_gap": float(self.vbd_static_gap),
            "vbd_head_mass": float(self.vbd_head_mass),
            "vbd_head_mesh_ke": float(self.vbd_head_mesh_ke),
            "vbd_head_mesh_kd": float(self.vbd_head_mesh_kd),
            "vbd_head_mesh_mu": float(self.vbd_head_mesh_mu),
            "vbd_head_mesh_margin": float(self.vbd_head_mesh_margin),
            "vbd_head_mesh_xy_scale": float(self.vbd_head_mesh_xy_scale),
            "vbd_static_mesh_use_sdf": bool(self.vbd_static_mesh_use_sdf),
            "vbd_static_mesh_sdf_max_resolution": int(self.vbd_static_mesh_sdf_max_resolution),
            "kit_static_contact_mode": self.kit_static_contact_mode,
            "kit_static_visual_meshes": bool(self.kit_static_visual_meshes),
            "vbd_near_tip_mu": float(self.vbd_near_tip_mu),
            "vbd_far_tip_mu": float(self.vbd_far_tip_mu),
            "vbd_ground_mu": float(self.vbd_ground_mu),
            "vbd_rigid_avbd_beta": float(self.vbd_rigid_avbd_beta),
            "vbd_rigid_contact_history": bool(self.vbd_rigid_contact_history),
            "vbd_rigid_contact_k_start": float(self.vbd_rigid_contact_k_start),
            "vbd_rigid_joint_linear_ke": float(self.vbd_rigid_joint_linear_ke),
            "vbd_rigid_joint_angular_ke": float(self.vbd_rigid_joint_angular_ke),
            "vbd_rigid_joint_linear_k_start": float(self.vbd_rigid_joint_linear_k_start),
            "vbd_rigid_joint_angular_k_start": float(self.vbd_rigid_joint_angular_k_start),
            "cable_stretch_stiffness": float(self.cable_stretch_stiffness),
            "cable_stretch_damping": float(self.cable_stretch_damping),
            "cable_num_segments": int(self.cable_num_segments),
            "cable_bend_stiffness": float(self.cable_bend_stiffness),
            "cable_bend_rigidity": float(self.cable_bend_rigidity),
            "cable_bend_damping": float(self.cable_bend_damping),
            "disable_cuda_graph": bool(self.disable_cuda_graph),
            "device": str(self.sim.device),
        }
