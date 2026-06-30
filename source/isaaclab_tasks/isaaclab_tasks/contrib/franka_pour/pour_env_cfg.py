# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka grasp-a-cup-of-MPM-media-and-pour, on the stable Isaac-Lift-Cube-Franka foundation.

Scene assets are borrowed from the lift task (standard Franka + the SeattleLab table USD) for a
stable, familiar base. On top we add a coupled Newton solver with **proxy coupling**:

* an MJWarp ``arm`` entry owns the robot AND a single rigid **dynamic** cup body, and
* an implicit ``media`` entry owns the MPM particles.

The cup is a real dynamic rigid body resting on the table: the Franka grasps it with its fingers
through MuJoCo friction contacts (exactly like the lift task grasps its cube), and a Newton proxy
mapping exposes the cup's ``COLLIDE_PARTICLES`` cavity mesh to the MPM solver as an auto-pose-synced
collider that retains/pours the media. This replaces the earlier welded-kinematic-cup design.

The cup carries two co-located shapes on the same body: a SOLID grasp box (``COLLIDE_SHAPES``,
arm-entry-only) the fingers can actually grip, and a hollow cavity mesh (``COLLIDE_PARTICLES``) the
proxy bridges to MPM. Arm control is relative DiffIK plus a binary gripper open/close action.
"""

from __future__ import annotations

from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import (
    CoupledProxyCfg,
    CoupledSolverCfg,
    CoupledSolverEntryCfg,
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonCfg,
    ProxyCouplingCfg,
)
from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from . import mdp

RIGID_ENTRY = "arm"
MPM_ENTRY = "media"
CUP_LABEL_PATTERN = r".*/Cup$"
TARGET_CUP_LABEL_PATTERN = r".*/TargetCup$"
TARGET_CUP_RIGID_LABEL_PATTERN = r".*/TargetCupRigid$"
SPILL_FLOOR_LABEL_PATTERN = r".*/SpillFloor$"


def _mpm_solver_cfg(cfg: FrankaPourEnvCfg) -> MPMSolverCfg:
    """Return the task's unique implicit-MPM solver config."""
    entries = [entry for entry in cfg.sim.physics.solver_cfg.entries if entry.name == MPM_ENTRY]
    if len(entries) != 1:
        raise ValueError(f"Expected exactly one {MPM_ENTRY!r} solver entry, found {len(entries)}.")
    return entries[0].solver_cfg


def _resolve_mpm_cell_cap(cfg: FrankaPourEnvCfg) -> int:
    """Resolve sparse MPM capacity for the final environment count without mutating ``cfg``.

    This runs during environment construction, after command-line or Hydra
    overrides have set ``scene.num_envs``. For sparse grids, the larger of the
    per-world estimate and per-world floor is multiplied by the world count.
    Fixed and dense grids retain their configured solver capacity unless an
    explicit total override is provided.

    Returns:
        The total capacity to assign to the MPM solver entry.
    """
    solver_cfg = _mpm_solver_cfg(cfg)
    override = cfg.mpm_cell_cap_override
    if override is not None:
        capacity = int(override)
    elif solver_cfg.grid_type == "sparse":
        num_envs = int(cfg.scene.num_envs)
        per_world = max(int(cfg.mpm_cells_per_env), int(cfg.mpm_min_cells_per_env))
        capacity = per_world * num_envs
    else:
        capacity = int(solver_cfg.max_active_cell_count)

    if capacity <= 0:
        raise ValueError(f"Franka Pour MPM capacity must be positive, got {capacity}.")
    return capacity


@configclass
class PourSceneCfg(InteractiveSceneCfg):
    """Lift-task scene assets (Franka + SeattleLab table) plus the declarative MPM media entity."""

    # SeattleLab table (top at env z=0), exactly as the Isaac-Lift-Cube-Franka scene.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0, 0, 0.707, 0.707]),  # wxyz, matches lift
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)
    )
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Built at environment-construction time so spawn points fill the resting cup cavity.
    media: MPMObjectCfg | None = None


@configclass
class ActionsCfg:
    """Relative DiffIK for the arm (6-dim EE delta) plus a binary gripper open/close (1-dim).

    The gripper is now a real grasp interface: closing it grips the dynamic cup through MuJoCo
    friction contacts (the cup is no longer welded)."""

    arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
        # Match the proven Isaac-Lift task scale. The previous 0.5 m/action made the initial
        # unit-variance PPO policy command half-metre Cartesian jumps at 60 Hz.
        scale=0.1,
    )
    # Bias the zero-centred initial policy toward a closed physical grasp. With a sign-threshold
    # binary action, exploration noise toggled the gripper every step and PPO learned to flee the
    # cup. The policy can still deliberately open by commanding > 0.25.
    gripper_action = mdp.AbsBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
        threshold=0.25,
        positive_threshold=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Robot, gripper, and two-cup geometry available to the actor."""

        arm_q = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])}
        )
        arm_qd = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])}
        )
        tcp_pose = ObsTerm(func=mdp.tcp_pose_obs)
        cup_pose = ObsTerm(func=mdp.cup_pose_obs)
        target_pose = ObsTerm(func=mdp.target_pose_obs)
        tcp_to_grasp = ObsTerm(func=mdp.tcp_to_grasp_obs)
        cup_to_target = ObsTerm(func=mdp.cup_to_target_obs)
        gripper_width = ObsTerm(func=mdp.gripper_width_obs)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Exact MPM fill fractions used by the critic, but not required by the deployed actor."""

        particle_fractions = ObsTerm(func=mdp.particle_fractions_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class RewardsCfg:
    # Each later physical stage is gated by the earlier one, while particle delivery remains the
    # dominant objective. This avoids the old run's zero-success sparse-reward plateau.
    reach = RewTerm(func=mdp.reach_cup, weight=1.0, params={"std": 0.10})
    grasp = RewTerm(func=mdp.grasp_cup, weight=2.0, params={"reach_std": 0.06})
    lift = RewTerm(func=mdp.lift_cup, weight=10.0, params={"target_height": 0.12, "reach_std": 0.07})
    lift_command = RewTerm(
        func=mdp.lift_command_progress,
        weight=15.0,
        params={"target_height": 0.12, "reach_std": 0.07},
    )
    align = RewTerm(func=mdp.align_cup_over_target, weight=8.0, params={"lift_height": 0.06, "std": 0.12})
    align_command = RewTerm(func=mdp.align_command_progress, weight=8.0, params={"lift_height": 0.06, "std": 0.12})
    tilt = RewTerm(func=mdp.tilt_over_target, weight=6.0, params={"lift_height": 0.06, "align_std": 0.10})
    tilt_command = RewTerm(
        func=mdp.tilt_command_progress,
        weight=4.0,
        params={"lift_height": 0.06, "align_std": 0.10},
    )
    delivered = RewTerm(func=mdp.particles_in_target, weight=30.0)
    success = RewTerm(func=mdp.pour_success_bonus, weight=25.0)
    # A full-cup spill previously applied -3 for every remaining step and dominated all staged
    # rewards, so the first 100-iteration baseline learned to move away. Keep it a penalty without
    # making exploration failure irreversible.
    spill = RewTerm(func=mdp.spilled_particles, weight=-0.2)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.001)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    failure = DoneTerm(func=mdp.nonfinite_failure)


@configclass
class EventsCfg:
    reset_scene = EventTerm(func=mdp.reset_pour_scene, mode="reset")


@configclass
class FrankaPourEnvCfg(ManagerBasedRLEnvCfg):
    """Franka grasping a dynamic cup of MPM media on the lift foundation, proxy-coupled solver."""

    scene: PourSceneCfg = PourSceneCfg(num_envs=2, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    # ---- Franka layout / reset (gripper above the cup, ready to grasp) ----
    # Initial grasp curriculum: fingers open and centred around the cup walls. This configuration
    # was measured from the same DiffIK trajectory used by ``pour_grasp_smoke.py``; the policy must
    # still close on the physical contacts, lift, carry, and pour, but does not spend the first
    # hundreds of low-throughput MPM iterations discovering a 30 cm free-space approach.
    arm_home: tuple = (0.00144, 0.56318, -0.00085, -2.59404, 0.00120, 3.69399, 0.74187)
    # Coupled-pipeline arm tuning: heavy PD damping + reflected rotor inertia keep the distal joints
    # from limit-cycling under the coupled solver.
    arm_stiffness: float = 600.0
    arm_damping: float = 50.0
    arm_armature: float = 0.5
    # Gripper actuator gains for a stable friction grasp of the LIGHT cup. Per this task's grasp
    # findings: stiffness 400-600 slips, >= 1500 ejects the light cup; ~800 is the sweet spot.
    finger_stiffness: float = 800.0
    finger_damping: float = 50.0
    finger_armature: float = 0.5
    gripper_open_pos: float = 0.04  # finger position the cup is grasped from (fingers start open)

    # ---- source cup (dynamic, grasped by the fingers) ----
    # A 55 x 55 x 33 mm hollow cube is small enough for the Panda's 80 mm opening. Its visible outer
    # wall and solid grasp proxy have exactly the same extents; the old visual was ~160 mm wide while
    # its hidden proxy was only 36 mm, which made a correct contact look like finger tunnelling.
    source_cup_inner_width: float = 0.037
    source_cup_inner_depth: float = 0.037
    # Keep the source shallower than its opening: the physical Panda grasp remains stable through
    # roughly 120 degrees, and a deeper 45 mm cavity retained the settled granular bed at that limit.
    source_cup_cavity_depth: float = 0.024
    source_cup_wall_thickness: float = 0.009
    source_cup_bottom_thickness: float = 0.009
    source_cup_friction: float = 0.9
    cup_mass: float = 0.05
    # The x/y faces coincide with the visible 55 mm cup. The taller 80 mm rigid grip band prevents
    # the media-loaded cup from rolling between flat fingers at >90 deg; a 54 mm near-square proxy
    # could rotate corner-through-contact once the hand reached the pour pose.
    cup_grasp_box_half: tuple = (0.0275, 0.0275, 0.040)
    cup_grasp_height: float = 0.032
    # A/B grasp testing: mu=1 let the media-loaded cup roll out during the carry, while mu=2 kept
    # bilateral contact through the full carry/tilt path without raising tangential stiffness.
    cup_grasp_box_friction: float = 2.0
    # The Newton default (ke=2.5e3 N/m) permits several millimeters of penetration under the
    # finger position drive.  That is enough for the fingers to cross the narrow grasp proxy and
    # lose the contact manifold entirely.  Match Newton's rigid-robot contact recipe instead.
    # A/B sweep: 50 kN/m retained the cup but allowed ~4 mm corner penetration at mu=2;
    # 100 kN/m reduced typical penetration to ~2 mm and stayed stable; 200 kN/m ejected the cup.
    grasp_contact_ke: float = 1.0e5
    grasp_contact_kd: float = 5.0e2
    grasp_contact_kf: float = 1.0e3
    # Cup reset pose, in the env frame. The cup rests on the table (top z=0) directly under the home
    # gripper, opening up. z is the cup base height (the body local origin sits at the outer base).
    cup_reset_pos: tuple = (0.5, 0.0, 0.0)

    # ---- receiving cup (fixed, represented once per solver to keep ownership disjoint) ----
    # The receiver is wider than the source during the initial learning curriculum. It remains a
    # proper hollow cup while making early particle-delivery experience substantially less sparse.
    target_cup_inner_width: float = 0.140
    target_cup_inner_depth: float = 0.140
    target_cup_cavity_depth: float = 0.065
    target_cup_wall_thickness: float = 0.009
    target_cup_bottom_thickness: float = 0.009
    target_cup_friction: float = 0.8
    # Keep enough initial clearance for the Panda finger collision meshes. Moving this wide receiver
    # to y=-0.12 made its rigid rim touch the pre-grasp hand and explosively eject media at step 0.
    target_cup_reset_pos: tuple = (0.5, -0.18, 0.0)
    collider_margin: float = 0.002

    # First-stage curriculum threshold: the deterministic physical reference pour reaches ~41%.
    # The continuous delivered-fraction reward remains active above this threshold, so the policy
    # is still rewarded for emptying more of the source rather than stopping at the bonus.
    pour_target_frac: float = 0.4
    particle_count_margin: float = 0.003

    # ---- media (granular sand inside the cup) ----
    media_fill_frac: float = 0.70
    media_material: MPMParticleMaterialCfg = MPMParticleMaterialCfg(
        density=1500.0,
        friction=0.7,
        yield_pressure=1.0e12,
    )

    # ---- MPM ----
    voxel_size: float = 0.006
    particles_per_cell: float = 2.0
    mpm_iterations: int = 24
    # ``max_active_cell_count`` is a total shared reserve in Newton. Derive it
    # exactly from a per-world estimate/floor unless a total override is set.
    mpm_min_cells_per_env: int = 16000
    mpm_cells_per_env: int = 4000
    mpm_cell_cap_override: int | None = None
    num_substeps: int = 4
    coupling_type: str = "proxy"
    proxy_iterations: int = 1
    proxy_mass_scale: float = 1.0
    use_cuda_graph: bool = True

    def __post_init__(self):
        self.decimation = 2
        # The conservative physical reference begins draining after ~8.7 s. A 12 s horizon makes
        # actual delivery reachable rather than timing out midway through the first tilt.
        self.episode_length_s = 12.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.use_newton_actuators = False
        self.viewer.eye = (1.4, 1.4, 0.9)
        self.viewer.lookat = (0.5, 0.0, 0.1)

        self.scene.robot.init_state.joint_pos.update(
            dict(zip([f"panda_joint{i}" for i in range(1, 8)], self.arm_home, strict=True))
        )
        self.scene.robot.init_state.joint_pos["panda_finger_joint.*"] = self.gripper_open_pos
        # Re-enable the gripper as a real grasp interface (keep the panda_hand actuator). Use the
        # coupled-stable arm gains on the arm and the light-cup grasp gains on the fingers.
        for actuator_name in ("panda_shoulder", "panda_forearm"):
            self.scene.robot.actuators[actuator_name].stiffness = self.arm_stiffness
            self.scene.robot.actuators[actuator_name].damping = self.arm_damping
            self.scene.robot.actuators[actuator_name].armature = self.arm_armature
        hand = self.scene.robot.actuators["panda_hand"]
        hand.stiffness = self.finger_stiffness
        hand.damping = self.finger_damping
        hand.armature = self.finger_armature

        self.sim.physics = NewtonCfg(
            solver_cfg=CoupledSolverCfg(
                coupling_type=self.coupling_type,
                scene_cfg=self.scene,
                entries=[
                    CoupledSolverEntryCfg(
                        name=RIGID_ENTRY,
                        # Proxy coupling keeps the MPM stable, so the arm integrator can be the faster
                        # "implicitfast" (unlike base coupling, which needed "euler"). The cup is a
                        # dynamic rigid body owned by this entry; the fingers grasp it via MuJoCo
                        # contacts and the proxy bridges its cavity mesh to the MPM solver.
                        solver_cfg=MJWarpSolverCfg(
                            use_mujoco_contacts=True, integrator="implicitfast", njmax=510, nconmax=400
                        ),
                        body_entities=[SceneEntityCfg("robot")],
                        body_label_patterns=[CUP_LABEL_PATTERN, TARGET_CUP_RIGID_LABEL_PATTERN],
                        include_static_shapes=True,
                        substeps=self.num_substeps,
                    ),
                    CoupledSolverEntryCfg(
                        name=MPM_ENTRY,
                        solver_cfg=MPMSolverCfg(
                            voxel_size=self.voxel_size,
                            grid_type="sparse",
                            grid_padding=0,
                            max_active_cell_count=120000,
                            strain_basis="P0",
                            transfer_scheme="apic",
                            max_iterations=self.mpm_iterations,
                            # "forward": the moving cup carries its media ("backward" drains it).
                            collider_velocity_mode="forward",
                            # Jacobi is the validated nonlinear path for outer capture. Combined
                            # with a positive sparse capacity and zero padding, the NanoVDB grid
                            # rebuilds in place while each RL environment stays grid-isolated.
                            solver="jacobi",
                            separate_worlds=True,
                            project_outside_colliders=True,
                        ),
                        all_particles=True,
                        # The source cup is proxied from the rigid solver. The separate particle-only
                        # target body is owned here; its co-located rigid copy belongs to the arm entry.
                        body_label_patterns=[TARGET_CUP_LABEL_PATTERN, SPILL_FLOOR_LABEL_PATTERN],
                        include_static_shapes=False,
                        include_child_joints=False,
                        in_place=True,
                    ),
                ],
                proxy_coupling=ProxyCouplingCfg(
                    proxies=[
                        CoupledProxyCfg(
                            source=RIGID_ENTRY,
                            destination=MPM_ENTRY,
                            body_label_patterns=[CUP_LABEL_PATTERN],
                            mass_scale=self.proxy_mass_scale,
                            mode="lagged",
                        )
                    ],
                    iterations=self.proxy_iterations,
                ),
                use_collision_pipeline=False,
            ),
            num_substeps=self.num_substeps,
            use_cuda_graph=self.use_cuda_graph,
        )


@configclass
class FrankaPourEnvCfg_PLAY(FrankaPourEnvCfg):
    def __post_init__(self):
        self.use_cuda_graph = True
        self.mpm_min_cells_per_env = 24000
        super().__post_init__()
        self.scene.num_envs = 4


@configclass
class FrankaPourEnvCfg_TELEOP(FrankaPourEnvCfg_PLAY):
    """Teleop preset: 1 env, no RL time-out (operator resets manually)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.terminations.time_out = None
        self.episode_length_s = 3600.0
