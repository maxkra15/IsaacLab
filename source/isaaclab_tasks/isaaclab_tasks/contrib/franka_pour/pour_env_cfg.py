# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka grasp-a-cup-of-MPM-media-and-pour, on the stable Isaac-Lift-Cube-Franka foundation.

Scene assets are borrowed from the lift task (standard Franka + the SeattleLab table USD) for a
stable, familiar base. On top we add a coupled Newton solver with **proxy coupling**:

* an MJWarp ``arm`` entry owns the robot, the dynamic source cup, and the fixed receiver, and
* an implicit ``media`` entry owns the MPM particles.

The source cup is a real dynamic rigid body resting on the table: the Franka grasps it with its fingers
through Newton-generated friction contacts resolved by MJWarp, and a Newton proxy mapping exposes
both cups' ``COLLIDE_PARTICLES`` cavity meshes to the MPM solver as auto-pose-synced colliders.
This replaces the earlier welded-kinematic-cup design.

The source cup carries two co-located shapes on the same body: a solid grasp box (``COLLIDE_SHAPES``,
arm-entry-only) the fingers can actually grip, and a hollow cavity mesh (``COLLIDE_PARTICLES``) the
proxy bridges to MPM. The policy commands trajectory phase, bounded joint residuals, and one
continuous symmetric finger target.
"""

from __future__ import annotations

import math
from copy import deepcopy

from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonCollisionPipelineCfg,
)
from isaaclab_newton.sim.schemas import MujocoJointCfg
from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas import MassCfg, UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import CoupledProxyCfg, CoupledProxySolverCfg, CoupledSolverEntryCfg
from isaaclab_contrib.deformable.newton_manager_cfg import CoupledNewtonCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from . import mdp
from .cube_bowl_mesh import cube_bowl_inner_bounds
from .cube_bowl_spawner_cfg import CubeBowlSpawnerCfg
from .cup_media import build_media_object_cfg, cup_cavity_lattice

RIGID_ENTRY = "arm"
MPM_ENTRY = "media"
SPILL_FLOOR_LABEL_PATTERN = r".*/SpillFloor$"
CURRICULUM_STAGE_NAMES = ("pour", "carry", "grasp", "full", "randomized")


def _mpm_solver_cfg(cfg: FrankaPourEnvCfg) -> MPMSolverCfg:
    """Return the task's unique implicit-MPM solver config."""
    entries = [entry for entry in cfg.sim.physics.solver_cfg.entries if entry.name == MPM_ENTRY]
    if len(entries) != 1:
        raise ValueError(f"Expected exactly one {MPM_ENTRY!r} solver entry, found {len(entries)}.")
    return entries[0].solver_cfg


def _resolve_mpm_cell_cap(cfg: FrankaPourEnvCfg) -> int:
    """Resolve sparse MPM capacity for the final environment count without mutating ``cfg``.

    This runs during environment construction, after command-line or Hydra
    overrides have set ``scene.num_envs``. A sparse cell can contain one or
    more particles, so the fixed particle count is a hard upper bound on the
    number of active cells. Round that count up for allocator-friendly
    headroom, then multiply by the world count. Fixed and dense grids retain
    their configured solver capacity unless an explicit total override is
    provided.

    Returns:
        The total capacity to assign to the MPM solver entry.
    """
    solver_cfg = _mpm_solver_cfg(cfg)
    override = cfg.mpm_cell_cap_override
    if override is not None:
        capacity = int(override)
    elif solver_cfg.grid_type == "sparse":
        alignment = int(cfg.mpm_cell_capacity_alignment)
        if alignment <= 0:
            raise ValueError(f"Franka Pour MPM cell-capacity alignment must be positive, got {alignment}.")
        particle_count = int(cup_cavity_lattice(cfg)[0].shape[0])
        per_world = ((particle_count + alignment - 1) // alignment) * alignment
        num_envs = int(cfg.scene.num_envs)
        capacity = per_world * num_envs
    else:
        capacity = int(solver_cfg.max_active_cell_count)

    if capacity <= 0:
        raise ValueError(f"Franka Pour MPM capacity must be positive, got {capacity}.")
    return capacity


def _resolve_mpm_upper_node_cap(cfg: FrankaPourEnvCfg) -> int:
    """Resolve a total NanoVDB upper-node reserve from bounded per-world motion.

    Warp packs isolated environments along the grid x axis with three guard
    cells between them. An upper NanoVDB node spans 4096 voxels per axis. The
    workspace is expanded by the maximum velocity-clamped displacement before
    Isaac Lab next evaluates terminations, then the worst-case number of
    intersected upper nodes is rounded to a power of two. This keeps small
    viewer runs at the automatic 32-node floor while scaling RL batches.

    Returns:
        The total upper-node capacity shared by all isolated environments.
    """
    solver_cfg = _mpm_solver_cfg(cfg)
    override = cfg.mpm_upper_node_cap_override
    if override is not None:
        capacity = int(override)
    elif solver_cfg.grid_type == "sparse" and solver_cfg.separate_worlds:
        voxel_size = float(solver_cfg.voxel_size)
        if not math.isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError(f"Franka Pour MPM voxel size must be finite and positive, got {voxel_size}.")

        # One velocity-bounded MPM advection and one velocity-bounded collider projection may
        # each move a particle during every substep interval.
        displacement = 2.0 * float(cfg.particle_max_velocity) * float(cfg.sim.dt) * int(cfg.decimation)
        lower = tuple(value - displacement for value in cfg.particle_workspace_lower_bound)
        upper = tuple(value + displacement for value in cfg.particle_workspace_upper_bound)
        voxel_spans = tuple(
            math.floor(hi / voxel_size) - math.floor(lo / voxel_size) + 1 for lo, hi in zip(lower, upper, strict=True)
        )

        upper_node_width = 8 * 16 * 32
        packed_x_span = int(cfg.scene.num_envs) * (voxel_spans[0] + 3)

        def worst_case_block_count(span: int) -> int:
            return math.ceil((span + upper_node_width - 1) / upper_node_width)

        required = worst_case_block_count(packed_x_span)
        required *= worst_case_block_count(voxel_spans[1])
        required *= worst_case_block_count(voxel_spans[2])
        capacity = max(32, 1 << (required - 1).bit_length())
    else:
        capacity = int(solver_cfg.max_upper_node_count)

    if capacity <= 0:
        raise ValueError(f"Franka Pour MPM upper-node capacity must be positive, got {capacity}.")
    return capacity


@configclass
class PourSceneCfg(InteractiveSceneCfg):
    """Lift-task scene assets plus resolved cups and MPM media."""

    # SeattleLab table (top at env z=0), exactly as the Isaac-Lift-Cube-Franka scene.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0, 0, 0.707, 0.707]),  # xyzw, matches Lift
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
    robot.spawn.joint_drive_props = [MujocoJointCfg(actuatorgravcomp=True)]
    # Built by :meth:`FrankaPourEnvCfg.finalize` from the final override values.
    source_cup: RigidObjectCfg | None = None
    target_cup: RigidObjectCfg | None = None
    media: MPMObjectCfg | None = None


@configclass
class ActionsCfg:
    """Permanent trajectory-phase, joint-residual, and symmetric-gripper controls."""

    arm_action = mdp.TrajectoryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[f"panda_joint{i}" for i in range(1, 8)],
        preserve_order=True,
        waypoint_count=6,
        # Residuals are always live and always retain their per-joint meaning. A conservative scale
        # keeps initial exploration close to the physically validated loaded-cup trajectory.
        residual_scale=(0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03),
        # The cubic reference is already smooth; filter only exploratory joint residuals so the
        # physical arm does not lag the task-space waypoints by an extra controller time constant.
        alpha=0.10,
        phase_rate=0.40,
        # Reach the guarded capture pose promptly, then keep lift/carry moderate enough that the
        # lightweight loaded cup remains inside the coupled-contact basin.
        approach_phase_rate=0.40,
        transport_phase_rate=0.40,
        waypoint_phases=(0.0, 0.12, 0.24, 0.40, 0.62, 1.0),
        grasp_waypoint=2,
        lift_waypoint=3,
        align_waypoint=4,
        # The carry-stage reset supplies geometric contact but still needs a short preload dwell
        # before moving the media-loaded cup.
        grasp_gate_stage=1,
        approach_max_lateral_distance=0.01,
        approach_max_joint_error=0.08,
        approach_dwell_steps=10,
        approach_max_linear_velocity=0.01,
        approach_max_angular_velocity=0.1,
        align_max_distance=0.06,
        # The fingers close around a 56 mm cup while the TCP is centered between them.  An 18 mm
        # held tolerance covers the measured contact-settling spread without making progression
        # optimistic: the following lift gate still requires the cup itself to rise with the hand.
        grasp_max_tcp_distance=0.018,
        grasp_dwell_steps=5,
        grasp_max_linear_velocity=0.10,
        grasp_max_angular_velocity=1.0,
        clip={
            "panda_joint1": (-2.8973, 2.8973),
            "panda_joint2": (-1.7628, 1.7628),
            "panda_joint3": (-2.8973, 2.8973),
            "panda_joint4": (-3.0718, -0.0698),
            "panda_joint5": (-2.8973, 2.8973),
            "panda_joint6": (-0.0175, 3.7525),
            "panda_joint7": (-2.8973, 2.8973),
        },
    )
    gripper_action = mdp.CurriculumGripperPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        # Zero action follows the validated close target; the policy supplies a filtered position
        # residual and may relax toward the demonstrated preload by at most the validated 1 mm
        # contact-safe interval. This avoids requiring same-sign samples merely to discover grasp.
        # Opening for acquisition is handled separately by the capture interlock below.
        scale=0.001,
        alpha=0.2,
        close_position=0.024,
        neutral_position=0.025,
        open_position=0.04,
        force_open_before_phase_stage=2,
        force_open_before_phase=0.24,
        # Do not begin closing from an off-axis approach. A 10 mm release tolerance lets one
        # finger contact first and translate this very light cup; the centered 5 mm condition is
        # the validated capture envelope for randomized starts.
        capture_max_lateral_distance=0.005,
        capture_max_vertical_distance=0.010,
        capture_max_joint_error=0.08,
        capture_dwell_steps=5,
        capture_max_linear_velocity=0.02,
        capture_max_angular_velocity=0.2,
        # The lift gate still requires persistent bilateral deflection and actual cup motion. A
        # 5 cm/s finger-settling threshold avoids spending most of a five-second attempt waiting
        # for sub-millimetre drive oscillations to decay.
        contact_max_velocity=0.05,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Sensor-compatible robot, gripper, and cup geometry available to the actor."""

        arm_q = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
            scale=0.3,
        )
        arm_qd = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
            scale=0.05,
        )
        arm_reference_phase = ObsTerm(func=mdp.arm_reference_phase_obs)
        # Observe the applied filtered command so the stateful residual controller remains Markov
        # under reset-bank randomization.
        arm_reference_error = ObsTerm(func=mdp.arm_reference_error_obs, scale=0.3)
        trajectory_status = ObsTerm(func=mdp.trajectory_status_obs)
        time_remaining = ObsTerm(func=mdp.time_remaining_obs)
        pour_target_fraction = ObsTerm(func=mdp.pour_target_fraction_obs)
        tcp_pose = ObsTerm(func=mdp.tcp_pose_obs)
        cup_pose = ObsTerm(func=mdp.cup_pose_obs)
        tcp_to_grasp_position_c = ObsTerm(func=mdp.tcp_to_grasp_position_c_obs, scale=10.0)
        grasp_to_tcp_quat = ObsTerm(func=mdp.grasp_to_tcp_quat_obs)
        target_position_c = ObsTerm(func=mdp.target_position_c_obs, scale=5.0)
        finger_position = ObsTerm(func=mdp.finger_position_obs, scale=25.0)
        finger_velocity = ObsTerm(func=mdp.finger_velocity_obs, scale=5.0)
        gripper_target = ObsTerm(func=mdp.gripper_target_obs, scale=25.0)
        gripper_contact = ObsTerm(func=mdp.gripper_contact_obs, scale=250.0)
        last_action = ObsTerm(func=mdp.last_action, scale=0.2)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Exact simulation state available only to the asymmetric critic."""

        success_dwell = ObsTerm(func=mdp.success_dwell_obs)
        lost_grasp_dwell = ObsTerm(func=mdp.lost_grasp_dwell_obs)
        cup_velocity = ObsTerm(func=mdp.cup_velocity_obs, scale=0.1)
        particle_fractions = ObsTerm(func=mdp.particle_fractions_obs)
        particle_transfer = ObsTerm(func=mdp.particle_transfer_obs)
        held_delivery_history = ObsTerm(func=mdp.held_delivery_history_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class RewardsCfg:
    # One discounted hierarchical physical potential spans every backward-curriculum reset. It is
    # policy-invariant at PPO's gamma, so holding or wiggling cannot improve discounted return.
    task_progress = RewTerm(
        func=mdp.PourTaskProgress,
        weight=5.0,
        params={
            "target_height": 0.12,
            "reach_std": 0.07,
            "grasp_reach_std": 0.015,
            "grasp_preload_position": 0.025,
            "lift_height": 0.06,
            "align_std": 0.12,
            "source_offset_xy": (0.0, 0.05),
            "target_tilt": math.radians(150.0),
            "pour_direction_xy": (0.0, -1.0),
            "source_mouth_height": 0.036,
            "alignment_radius": 0.15,
            # Tilt is an exploration bootstrap only for the supplied-grasp stages. Full-task
            # policies are optimized by actual held particle transfer rather than a prescribed pose.
            "active_through_stage": 1,
            "min_lift_height": 0.05,
            "max_tcp_distance": 0.018,
            "max_gripper_width_error": 0.006,
            "max_gripper_command": 0.025,
            # Must match PPO gamma for policy-invariant discounted potential shaping.
            "discount_factor": 0.99,
        },
    )
    # Signed held-delivery progress is capped at the active success threshold. Particles leaving
    # the receiver repay their credit, and an unsuccessful episode repays any credit still held.
    delivered = RewTerm(
        func=mdp.HeldDeliveryProgress,
        weight=30.0,
        params={
            "min_lift_height": 0.05,
            "max_tcp_distance": 0.018,
            "max_gripper_width_error": 0.006,
            "max_gripper_command": 0.025,
        },
    )
    success = RewTerm(func=mdp.pour_success_bonus, weight=25.0)
    # Airborne transfer is excluded; each particle is penalized once after reaching the table
    # outside both cups. Termination bounds the failure at just over ten percent.
    spill = RewTerm(func=mdp.NewlySpilledParticles, weight=-30.0)
    # Count overlapping failures and an unsuccessful deadline once. This keeps a transient dump
    # that misses the stable-success predicate strictly worse than completing the task.
    failure = RewTerm(func=mdp.terminal_failure, weight=-35.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-4)
    # Every action is a residual around a complete nominal behavior. Magnitude regularization keeps
    # masked or weakly identified coordinates at that reference across curriculum promotions.
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.05)


@configclass
class TerminationsCfg:
    failure = DoneTerm(func=mdp.nonfinite_failure)
    extreme_rigid_state = DoneTerm(func=mdp.extreme_rigid_state)
    lost_grasp = DoneTerm(
        func=mdp.lost_lifted_grasp,
        params={
            "dwell_time_s": 0.05,
            "max_tcp_distance": 0.018,
            "max_gripper_width_error": 0.006,
            "max_gripper_command": 0.025,
        },
    )
    spill = DoneTerm(func=mdp.excessive_spill)
    particle_out_of_bounds = DoneTerm(func=mdp.particle_out_of_bounds)
    # Success follows every failure predicate, then the custom timeout excludes same-step success.
    success = DoneTerm(
        func=mdp.stable_pour_success,
        params={
            "dwell_time_s": 0.15,
            "min_lift_height": 0.05,
            "max_tcp_distance": 0.018,
            "max_gripper_width_error": 0.006,
            "max_gripper_command": 0.025,
        },
    )
    time_out = DoneTerm(func=mdp.unsuccessful_time_out, time_out=True)


@configclass
class EventsCfg:
    reset_scene = EventTerm(func=mdp.reset_pour_scene, mode="reset")


@configclass
class CurriculumCfg:
    stage = CurrTerm(func=mdp.PourCurriculum)


@configclass
class FrankaPourEnvCfg(ManagerBasedRLEnvCfg):
    """Franka grasping a dynamic cup of MPM media on the lift foundation, proxy-coupled solver."""

    scene: PourSceneCfg = PourSceneCfg(num_envs=2, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # ---- Franka layout / reset (gripper above the cup, ready to grasp) ----
    # Initial grasp curriculum: fingers open and centred around the cup walls. This configuration
    # was measured from the same Cartesian reference trajectory used by ``pour_grasp_smoke.py``;
    # the policy must still close on the physical contacts, lift, carry, and pour, but does not
    # spend the first hundreds of low-throughput MPM iterations discovering a 30 cm free-space
    # approach.
    arm_home: tuple[float, float, float, float, float, float, float] = (
        0.00144,
        0.56318,
        -0.00085,
        -2.59404,
        0.00120,
        3.69399,
        0.74187,
    )
    # Task-space metadata is independent of the policy action representation. SpaceMouse teleop
    # uses the same frame for its input-only IK adapter, while PPO commands joint positions.
    tcp_body_name: str = "panda_hand"
    tcp_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.107)
    tcp_offset_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    # Coupled-pipeline arm tuning: heavy PD damping + reflected rotor inertia keep the distal joints
    # from limit-cycling under the coupled solver.
    arm_stiffness: float = 800.0
    arm_damping: float = 50.0
    arm_armature: float = 0.5
    # Slightly softer, proportionally damped finger gains close the 50 g source cup without the
    # unilateral impulse produced by the stock drive, while still acquiring within the horizon.
    finger_stiffness: float = 1500.0
    finger_damping: float = 75.0
    finger_armature: float = 0.0
    gripper_open_pos: float = 0.04  # finger position the cup is grasped from (fingers start open)
    # A target inside the 0.03 m geometric contact position proves active squeeze/preload instead
    # of open fingers passively compressed by cup contact.
    # Three millimetres of target penetration per finger is ample at mu=2. The close target is
    # derived from this value and ``gripper_close_offset`` during finalization.
    gripper_preload_pos: float = 0.025
    gripper_close_offset: float = 0.001

    # ---- source cup (dynamic, grasped by the fingers) ----
    # A 60 x 60 x 36 mm hollow cube leaves 20 mm of clearance in the Panda's 80 mm opening. Its
    # visible outer wall and solid grasp proxy have exactly the same extents; the old visual was
    # ~160 mm wide while its hidden proxy was only 36 mm, which made a correct contact look like
    # finger tunnelling.
    source_cup_inner_width: float = 0.042
    source_cup_inner_depth: float = 0.042
    # Keep the source shallower than its opening: the physical Panda grasp remains stable through
    # roughly 120 degrees, and a deeper 45 mm cavity retained the settled granular bed at that limit.
    source_cup_cavity_depth: float = 0.027
    source_cup_wall_thickness: float = 0.007
    source_cup_bottom_thickness: float = 0.009
    source_cup_friction: float = 0.9
    cup_mass: float = 0.05
    # Match the visible cup's exact 60 x 60 x 36 mm outer envelope. Extending this rigid-only proxy
    # above the rim creates phantom finger/table contacts even though particles ignore the proxy.
    cup_grasp_box_half: tuple[float, float, float] = (0.028, 0.028, 0.018)
    cup_grasp_height: float = 0.030
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
    cup_reset_pos: tuple[float, float, float] = (0.5, 0.0, 0.0)

    # ---- receiving cup (fixed, represented once and proxied between solvers) ----
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
    target_cup_reset_pos: tuple[float, float, float] = (0.5, -0.18, 0.0)
    # The grasped source origin moves about 5 cm toward environment -y during the deep +x tilt.
    # Starting behind the receiver keeps the draining mouth centered throughout that motion.
    pour_source_offset_xy: tuple[float, float] = (0.0, 0.05)
    collider_margin: float = 0.002

    # Full-task threshold kept as a standalone compatibility knob. Earlier curriculum stages use
    # the three values below; :attr:`curriculum_target_frac` combines both sources.
    # A 30% transfer is 74 of the 245 particles. The demonstrated trajectory reaches about 41%,
    # leaving enough margin that the success predicate measures manipulation rather than the
    # lower tail of large-batch MPM/contact variation.
    pour_target_frac: float = 0.30
    particle_count_margin: float = 0.003
    # Particle point samples resting on the z=0 MPM spill plane settle within the containment
    # margin above it. Only points in that contact band and outside both cups are true spills.
    spill_table_height: float = 0.0
    max_spill_fraction: float = 0.10
    # A transfer must remain above its stage threshold for this duration before successful
    # termination. This rejects transient particle crossings and aligns reward with curriculum.
    # Nine consecutive control steps reject transient particle crossings while leaving enough of
    # the finite horizon for the terminal event after a late, valid randomized grasp.
    success_dwell_time_s: float = 0.15
    lost_grasp_dwell_time_s: float = 0.05
    """Continuous post-lift grasp loss required before failure [s]."""
    success_min_lift_height: float = 0.05
    success_max_tcp_distance: float = 0.018
    # A contact-free hand reaches a 64 mm measured gap at the bounded 24 mm command, exactly 8 mm
    # wider than this 56 mm cup. Requiring <=6 mm distinguishes real bilateral cup contact while
    # retaining roughly 3 mm of measured true-grasp variation.
    success_max_gripper_width_error: float = 0.006
    # Reset extreme but finite rigid state before it can enter actor observation normalization.
    state_bound_joint_position_margin: float = 0.05
    state_bound_max_joint_velocity: float = 20.0
    state_bound_max_cup_linear_velocity: float = 10.0
    state_bound_max_cup_angular_velocity: float = 50.0
    # Keep finite escaped particles from expanding the rebuildable NanoVDB hierarchy throughout
    # an episode. Bounds are in each environment's local frame and comfortably contain both cups,
    # every curriculum reset, and the robot workspace.
    particle_workspace_lower_bound: tuple[float, float, float] = (-0.5, -1.0, -0.5)
    particle_workspace_upper_bound: tuple[float, float, float] = (1.5, 1.0, 1.5)

    # ---- success-driven backward curriculum ----
    # The first reset starts already grasping an upright loaded cup over the receiver (learn pour),
    # then moves it to a source-side hover (carry + pour), to the table centered between open fingers
    # (grasp + lift + carry + pour), and to the normal open-finger pre-grasp (full task). The final stage
    # adds a randomized open-hand approach and randomized source/target positions.
    # Reset IK is solved once into a bank; asynchronous resets only select prevalidated rows.
    curriculum_stage_names: tuple[str, ...] = CURRICULUM_STAGE_NAMES
    curriculum_pour_arm_q: tuple[float, float, float, float, float, float, float] = (
        -1.564917803,
        0.088611290,
        1.410412073,
        -2.594748974,
        0.802014828,
        3.137426376,
        -0.237847745,
    )
    curriculum_pour_target_arm_q: tuple[float, float, float, float, float, float, float] = (
        -1.743660808,
        1.478038549,
        0.972973883,
        -2.170451641,
        1.137111664,
        1.747459650,
        0.088701032,
    )
    curriculum_carry_arm_q: tuple[float, float, float, float, float, float, float] = (
        -0.02525926,
        -0.02049879,
        0.02077409,
        -2.71968818,
        -0.02134521,
        3.23598433,
        0.75964969,
    )
    # The deterministic physical reference reaches ~41% on the full task. First-time particle
    # delivery remains rewarded above every success threshold.
    curriculum_early_target_frac: tuple[float, float, float] = (0.10, 0.20, 0.30)
    curriculum_randomized_pour_target_frac: float = 0.30
    curriculum_randomized_source_position_range: tuple[float, float] = (0.12, 0.10)
    # Once lifted, pull only the new outer X reset cells into the previously validated carry
    # workspace. This retains the larger reach problem without placing the carry IK near joint 6's
    # lower limit.
    curriculum_randomized_carry_position_range: tuple[float, float] = (0.10, 0.10)
    # Keep the loaded source upright on the table and vary only its yaw. Fifteen degrees changes
    # which opposing faces the hand must align with without approaching the square cup's redundant
    # 90-degree symmetry or exhausting the receiver-clearance reserve.
    curriculum_randomized_source_yaw_range: float = math.radians(15.0)
    # The full receiver remains randomized by +/-5 cm, but its stage-four center is shifted 3 cm
    # behind the nominal curriculum pose. This preserves the requested range while keeping the
    # large receiver rim clear of the open hand at the source-y=-10 cm edge.
    curriculum_randomized_target_center_xy: tuple[float, float] = (0.50, -0.21)
    curriculum_randomized_target_position_range: tuple[float, float] = (0.05, 0.05)
    curriculum_randomized_cup_clearance: float = 0.04
    # The randomized stage starts above the grasp point. Several bounded Cartesian variants per
    # source location make the arm approach non-deterministic while keeping
    # the fixed robot base and all reset-time work fully batched on-device.
    curriculum_randomized_reset_tcp_standoff: tuple[float, float, float] = (0.0, 0.0, 0.12)
    curriculum_randomized_reset_tcp_jitter: tuple[float, float, float] = (0.04, 0.04, 0.02)
    curriculum_randomized_reset_tcp_min_grasp_distance: float = 0.09
    # Keep the held source proxy above the receiver rim throughout the independently solved
    # pour-to-tilt joint interpolation. Without this reserve the source corner and two collider
    # margins overlap even though both endpoint IK poses are valid.
    curriculum_randomized_pour_clearance: float = 0.010
    # Match the stationary table-contact equilibrium, which lies below the authored base pose.
    # The W1 dwell lets the cup settle before the arm descends to this fixed, graph-safe W2.
    curriculum_grasp_descent_overshoot: float = 0.007
    # An odd source grid includes the authored nominal pose and both configured XY extrema. Newton
    # IK solves this bank once at startup; asynchronous resets only gather prevalidated rows.
    curriculum_randomized_reset_ik_grid_size: int = 7
    curriculum_randomized_reset_ik_samples_per_source: int = 5
    curriculum_randomized_reset_ik_iterations: int = 96
    curriculum_randomized_reset_ik_max_cost: float = 1.0e-3
    curriculum_randomized_reset_ik_joint_margin: float = 0.02
    # Introduce the final stage through nested, normalized Chebyshev extents across source pose,
    # receiver pose, and reset-TCP jitter. The last level contains the complete prevalidated bank;
    # no reset-time IK is required.
    curriculum_randomization_extent_levels: tuple[float, ...] = (2.0 / 3.0, 5.0 / 6.0, 1.0)
    # Stateful manager progress is not part of RSL-RL checkpoints. Set both start controls to the
    # last logged values when resuming within the randomized stage.
    curriculum_randomization_start_level: int = 0
    curriculum_success_threshold: float = 0.8
    # Eight complete 512-world episode cohorts provide a statistically useful frontier estimate
    # without consuming most of the 3,000-iteration budget before final-stage optimization.
    curriculum_min_resets_per_stage: int = 4096
    # Retain the immediately preceding nested task after promotion so the policy does not forget
    # already-solved behavior while learning the newly introduced prerequisite.
    curriculum_previous_stage_replay_fraction: float = 0.1
    # Set this to the last logged ``Curriculum/stage/stage`` value when resuming training.
    curriculum_start_stage: int = 0
    curriculum_freeze: bool = False

    # ---- media (granular sand inside the cup) ----
    media_fill_frac: float = 0.70
    # Normal rollouts peak near 2.2 m/s. This generous clamp prevents a numerically launched
    # particle from crossing many NanoVDB upper regions within one manager step, before the
    # workspace termination can selectively reset its environment.
    particle_max_velocity: float = 10.0
    media_material: MPMParticleMaterialCfg = MPMParticleMaterialCfg(
        density=1500.0,
        friction=0.7,
        yield_pressure=1.0e12,
    )

    # ---- MPM ----
    voxel_size: float = 0.01
    particles_per_cell: float = 2.0
    mpm_iterations: int = 24
    # ``max_active_cell_count`` is a total shared reserve in Newton. Each fixed particle activates
    # at most one sparse cell, so round that hard per-world bound up to this alignment.
    mpm_cell_capacity_alignment: int = 256
    mpm_cell_cap_override: int | None = None
    mpm_upper_node_cap_override: int | None = None
    num_substeps: int = 2
    proxy_iterations: int = 1
    proxy_mass_scale: float = 1.0
    use_cuda_graph: bool = True
    # Warp FEM partitions topology by environment ID, so isolated worlds need not occupy distinct
    # physical coordinates. Colocating headless-scale batches avoids float32 cancellation in MPM
    # and contact calculations; smaller interactive layouts retain visible environment spacing.
    colocate_physics_min_envs: int | None = 1024
    """Minimum environment count at which isolated physics worlds share one origin.

    Set to ``None`` to preserve the requested spacing for large interactive layouts.
    """

    @property
    def curriculum_target_frac(self) -> tuple[float, ...]:
        """Per-stage delivered-particle success fractions."""
        return (
            *self.curriculum_early_target_frac,
            float(self.pour_target_frac),
            float(self.curriculum_randomized_pour_target_frac),
        )

    @curriculum_target_frac.setter
    def curriculum_target_frac(self, values: tuple[float, ...]) -> None:
        if len(values) != len(CURRICULUM_STAGE_NAMES):
            raise ValueError(f"curriculum_target_frac must contain {len(CURRICULUM_STAGE_NAMES)} values.")
        self.curriculum_early_target_frac = tuple(values[:3])
        self.pour_target_frac = float(values[3])
        self.curriculum_randomized_pour_target_frac = float(values[4])

    def __post_init__(self):
        self.actions.gripper_action.close_position = max(0.0, self.gripper_preload_pos - self.gripper_close_offset)
        self.actions.gripper_action.open_position = self.gripper_open_pos
        self.actions.gripper_action.neutral_position = self.gripper_preload_pos
        self.rewards.task_progress.params["grasp_preload_position"] = self.gripper_preload_pos
        self.rewards.task_progress.params["max_gripper_command"] = self.gripper_preload_pos
        self.rewards.task_progress.params["source_offset_xy"] = self.pour_source_offset_xy
        self.rewards.task_progress.params["source_mouth_height"] = (
            self.source_cup_bottom_thickness + self.source_cup_cavity_depth
        )
        self.rewards.delivered.params["max_gripper_command"] = self.gripper_preload_pos
        self.terminations.lost_grasp.params["dwell_time_s"] = self.lost_grasp_dwell_time_s
        self.terminations.lost_grasp.params["max_gripper_command"] = self.gripper_preload_pos
        self.terminations.success.params["max_gripper_command"] = self.gripper_preload_pos
        self.decimation = 2
        # Recycle failed attempts promptly after the expected manipulation sequence.
        self.episode_length_s = 5.0
        # The deadline is part of the task: an attempt that has not poured within five seconds is
        # a failed finite-horizon episode and must not be value-bootstrapped by RL wrappers.
        self.is_finite_horizon = True
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.use_newton_actuators = False
        self.viewer.eye = (1.4, 1.4, 0.9)
        self.viewer.lookat = (0.5, 0.0, 0.1)
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0

        self._validate_curriculum_cfg()
        self._validate_particle_workspace_cfg()
        self._apply_robot_cfg()

        self.sim.physics = CoupledNewtonCfg(
            scene_cfg=self.scene,
            solver_cfg=CoupledProxySolverCfg(
                entries=[
                    CoupledSolverEntryCfg(
                        name=RIGID_ENTRY,
                        # Proxy coupling keeps the MPM stable, so the arm integrator can be the faster
                        # "implicitfast" (unlike base coupling, which needed "euler"). The cup is a
                        # dynamic rigid body owned by this entry; Newton generates its contacts and
                        # the proxy bridges its cavity mesh to the MPM solver.
                        solver_cfg=MJWarpSolverCfg(
                            use_mujoco_contacts=False, integrator="implicitfast", njmax=510, nconmax=400
                        ),
                        bodies=[
                            SceneEntityCfg("robot"),
                            SceneEntityCfg("source_cup"),
                            SceneEntityCfg("target_cup"),
                        ],
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
                            # Active-cell and hierarchy capacities bound different resources. The
                            # final environment count and bounded workspace resolve this separately.
                            max_upper_node_count=32,
                            strain_basis="P0",
                            transfer_scheme="apic",
                            max_iterations=self.mpm_iterations,
                            # Keep Q1 velocity unknowns while sampling colliders at particle points.
                            # PIC27 caps local collider nodes at 27 per cell and avoids the
                            # capacity-sized grid-node reserve of the former S2 collider basis.
                            velocity_basis="Q1",
                            collider_basis="pic27",
                            # "forward": the moving cup carries its media ("backward" drains it).
                            collider_velocity_mode="forward",
                            # Jacobi is the validated nonlinear path for outer capture. Combined
                            # with a positive sparse capacity and zero padding, the NanoVDB grid
                            # rebuilds in place while each RL environment stays grid-isolated.
                            solver="jacobi",
                            separate_worlds=True,
                            project_outside_colliders=False,
                        ),
                        all_particles=True,
                        bodies=[SPILL_FLOOR_LABEL_PATTERN],
                        include_static_shapes=False,
                        include_child_joints=False,
                        in_place=True,
                    ),
                ],
                proxies=[
                    CoupledProxyCfg(
                        source=RIGID_ENTRY,
                        destination=MPM_ENTRY,
                        bodies=[SceneEntityCfg("source_cup"), SceneEntityCfg("target_cup")],
                        mass_scale=self.proxy_mass_scale,
                        mode="lagged",
                    )
                ],
                iterations=self.proxy_iterations,
                use_collision_pipeline=True,
            ),
            # Rigid contacts use Newton's outer pipeline. Implicit MPM handles particle/shape
            # collisions internally, so allocating outer soft contacts would waste O(P*S) work.
            collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
            num_substeps=self.num_substeps,
            use_cuda_graph=self.use_cuda_graph,
        )

    def _validate_gripper_action_cfg(self) -> None:
        """Validate the reset and action targets against the Panda finger range."""
        gripper_action = self.actions.gripper_action
        if (
            not math.isfinite(self.gripper_open_pos)
            or not 0.0 < self.gripper_open_pos <= 0.04
            or not math.isfinite(gripper_action.close_position)
            or not 0.0 <= gripper_action.close_position < self.gripper_open_pos
            or not math.isfinite(gripper_action.scale)
            or gripper_action.scale <= 0.0
            or not math.isfinite(gripper_action.alpha)
            or not 0.0 < gripper_action.alpha <= 1.0
        ):
            raise ValueError(
                "Gripper action positions must fit the Panda finger range [0, 0.04] with positive scale and a "
                "moving-average weight in (0, 1]."
            )
        if not math.isclose(gripper_action.open_position, self.gripper_open_pos, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("The gripper action open position must match gripper_open_pos.")
        if (
            not math.isfinite(self.gripper_preload_pos)
            or not gripper_action.close_position <= self.gripper_preload_pos < self.cup_grasp_box_half[1]
        ):
            raise ValueError("gripper_preload_pos must lie between the closed and geometric contact positions.")
        if (
            not math.isfinite(self.gripper_close_offset)
            or not 0.0 <= self.gripper_close_offset <= self.gripper_preload_pos
        ):
            raise ValueError("gripper_close_offset must lie in [0, gripper_preload_pos].")
        max_action_position = self.gripper_preload_pos if gripper_action.limit_to_preload else self.gripper_open_pos
        if not math.isclose(
            gripper_action.neutral_position,
            max_action_position,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("Gripper action maximum does not match its configured operating interval.")
        action_span = max_action_position - gripper_action.close_position
        if gripper_action.scale > action_span + 1.0e-9:
            raise ValueError("Gripper action scale must not exceed its configured operating interval.")
        if gripper_action.default_position is not None and not (
            gripper_action.close_position <= gripper_action.default_position <= gripper_action.neutral_position
        ):
            raise ValueError("Gripper default position must lie within its configured operating interval.")

    def _validate_curriculum_progress_cfg(self, stage_count: int) -> None:
        """Validate success statistics and promotion controls."""
        if not 0.0 < self.curriculum_success_threshold <= 1.0:
            raise ValueError("curriculum_success_threshold must lie in (0, 1].")
        if self.curriculum_min_resets_per_stage <= 0:
            raise ValueError("curriculum_min_resets_per_stage must be positive.")
        replay_fraction = self.curriculum_previous_stage_replay_fraction
        if not math.isfinite(replay_fraction) or replay_fraction < 0.0 or replay_fraction >= 1.0:
            raise ValueError("curriculum_previous_stage_replay_fraction must lie in [0, 1).")
        if self.curriculum_start_stage < 0 or self.curriculum_start_stage >= stage_count:
            raise ValueError(f"curriculum_start_stage must lie in [0, {stage_count - 1}].")
        extent_levels = self.curriculum_randomization_extent_levels
        if not extent_levels:
            raise ValueError("curriculum_randomization_extent_levels must not be empty.")
        if any(not math.isfinite(level) or level <= 0.0 or level > 1.0 for level in extent_levels):
            raise ValueError("curriculum_randomization_extent_levels must lie in (0, 1].")
        if any(previous >= current for previous, current in zip(extent_levels, extent_levels[1:])):
            raise ValueError("curriculum_randomization_extent_levels must be strictly increasing.")
        if not math.isclose(extent_levels[-1], 1.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("curriculum_randomization_extent_levels must end at 1.0.")
        if self.curriculum_randomization_start_level < 0 or self.curriculum_randomization_start_level >= len(
            extent_levels
        ):
            raise ValueError("curriculum_randomization_start_level must index curriculum_randomization_extent_levels.")
        tilt_params = self.rewards.task_progress.params
        target_tilt = float(tilt_params["target_tilt"])
        pour_direction_xy = tilt_params["pour_direction_xy"]
        source_offset_xy = self.pour_source_offset_xy
        source_mouth_height = float(tilt_params["source_mouth_height"])
        alignment_radius = float(tilt_params["alignment_radius"])
        active_through_stage = int(tilt_params["active_through_stage"])
        discount_factor = float(tilt_params["discount_factor"])
        if not math.isfinite(target_tilt) or not 0.0 < target_tilt < math.pi:
            raise ValueError("task_progress target_tilt must lie in (0, pi).")
        if (
            len(pour_direction_xy) != 2
            or any(not math.isfinite(value) for value in pour_direction_xy)
            or math.hypot(float(pour_direction_xy[0]), float(pour_direction_xy[1])) <= 0.0
        ):
            raise ValueError("task_progress pour_direction_xy must contain two finite values and be nonzero.")
        if len(source_offset_xy) != 2 or any(not math.isfinite(value) for value in source_offset_xy):
            raise ValueError("pour_source_offset_xy must contain two finite values.")
        if not math.isfinite(source_mouth_height) or source_mouth_height <= 0.0:
            raise ValueError("task_progress source_mouth_height must be finite and positive.")
        if not math.isfinite(alignment_radius) or alignment_radius <= 0.0:
            raise ValueError("task_progress alignment_radius must be finite and positive.")
        if active_through_stage < 0 or active_through_stage >= stage_count:
            raise ValueError(f"task_progress active_through_stage must lie in [0, {stage_count - 1}].")
        if not math.isfinite(discount_factor) or not 0.0 < discount_factor <= 1.0:
            raise ValueError("task_progress discount_factor must lie in (0, 1].")

    def _validate_arm_action_cfg(self) -> None:
        """Validate controller fields used by the task's authored six-waypoint trajectory."""
        arm_action = self.actions.arm_action
        if not math.isfinite(arm_action.alpha) or not 0.0 < arm_action.alpha <= 1.0:
            raise ValueError("Arm action moving-average weight must lie in (0, 1].")
        if not isinstance(arm_action, mdp.TrajectoryJointPositionActionCfg):
            return
        waypoint_phases = tuple(arm_action.waypoint_phases)
        if arm_action.waypoint_count != 6 or len(waypoint_phases) != 6:
            raise ValueError("Franka Pour requires exactly six authored arm waypoints and phases.")
        milestone_indices = (
            arm_action.approach_waypoint,
            arm_action.grasp_waypoint,
            arm_action.lift_waypoint,
            arm_action.align_waypoint,
        )
        if milestone_indices != (1, 2, 3, 4):
            raise ValueError("Franka Pour requires approach, grasp, lift, and align waypoint indices 1, 2, 3, 4.")
        if (
            waypoint_phases[0] != 0.0
            or waypoint_phases[-1] != 1.0
            or any(right <= left for left, right in zip(waypoint_phases, waypoint_phases[1:]))
        ):
            raise ValueError("Arm action waypoint phases must increase strictly from 0 to 1.")
        for field_name in ("phase_rate", "approach_phase_rate", "transport_phase_rate"):
            value = getattr(arm_action, field_name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Arm action {field_name} must be finite and positive.")

    def _validate_curriculum_cfg(self) -> None:
        """Validate the aligned per-stage backward-curriculum settings."""
        self._validate_gripper_action_cfg()
        self._validate_arm_action_cfg()
        stage_count = len(self.curriculum_stage_names)
        if self.curriculum_stage_names != CURRICULUM_STAGE_NAMES:
            raise ValueError(f"curriculum_stage_names must be {CURRICULUM_STAGE_NAMES!r}.")
        if len(self.curriculum_target_frac) != stage_count:
            raise ValueError(
                f"curriculum_target_frac has {len(self.curriculum_target_frac)} values for {stage_count} stages."
            )
        if any(
            not math.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0 for fraction in self.curriculum_target_frac
        ):
            raise ValueError("Curriculum target fractions must lie in (0, 1].")
        if tuple(sorted(self.curriculum_target_frac)) != self.curriculum_target_frac:
            raise ValueError("Curriculum target fractions must be nondecreasing.")
        self._validate_curriculum_progress_cfg(stage_count)
        if self.cup_grasp_box_half[1] < 0.0 or self.cup_grasp_box_half[1] > self.gripper_open_pos:
            raise ValueError("The curriculum contact position must fit within the open gripper.")

        for field_name in (
            "curriculum_randomized_source_position_range",
            "curriculum_randomized_carry_position_range",
            "curriculum_randomized_target_position_range",
        ):
            values = getattr(self, field_name)
            if len(values) != 2 or any(not math.isfinite(value) or value < 0.0 for value in values):
                raise ValueError(f"{field_name} must contain two finite nonnegative values.")
        if any(
            carry_range > source_range
            for carry_range, source_range in zip(
                self.curriculum_randomized_carry_position_range,
                self.curriculum_randomized_source_position_range,
                strict=True,
            )
        ):
            raise ValueError(
                "curriculum_randomized_carry_position_range must not exceed "
                "curriculum_randomized_source_position_range."
            )
        if len(self.curriculum_randomized_target_center_xy) != 2 or any(
            not math.isfinite(value) for value in self.curriculum_randomized_target_center_xy
        ):
            raise ValueError("curriculum_randomized_target_center_xy must contain two finite values.")
        if (
            not math.isfinite(self.curriculum_randomized_source_yaw_range)
            or self.curriculum_randomized_source_yaw_range < 0.0
            or self.curriculum_randomized_source_yaw_range > math.pi / 4.0
        ):
            raise ValueError("curriculum_randomized_source_yaw_range must lie in [0, pi / 4].")
        if not math.isfinite(self.curriculum_randomized_cup_clearance) or self.curriculum_randomized_cup_clearance < 0:
            raise ValueError("curriculum_randomized_cup_clearance must be finite and nonnegative.")
        for field_name in (
            "curriculum_randomized_reset_tcp_standoff",
            "curriculum_randomized_reset_tcp_jitter",
        ):
            values = getattr(self, field_name)
            if len(values) != 3 or any(not math.isfinite(value) for value in values):
                raise ValueError(f"{field_name} must contain three finite values.")
        if any(value < 0.0 for value in self.curriculum_randomized_reset_tcp_jitter):
            raise ValueError("curriculum_randomized_reset_tcp_jitter must contain three finite nonnegative values.")
        if (
            not math.isfinite(self.curriculum_randomized_reset_tcp_min_grasp_distance)
            or self.curriculum_randomized_reset_tcp_min_grasp_distance <= 0.0
        ):
            raise ValueError("curriculum_randomized_reset_tcp_min_grasp_distance must be finite and positive.")
        if (
            not math.isfinite(self.curriculum_randomized_pour_clearance)
            or self.curriculum_randomized_pour_clearance < 0.0
        ):
            raise ValueError("curriculum_randomized_pour_clearance must be finite and nonnegative.")
        if not math.isfinite(self.curriculum_grasp_descent_overshoot) or self.curriculum_grasp_descent_overshoot <= 0.0:
            raise ValueError("curriculum_grasp_descent_overshoot must be finite and positive.")
        if self.curriculum_randomized_reset_tcp_standoff[2] - self.curriculum_randomized_reset_tcp_jitter[2] <= 0.0:
            raise ValueError(
                "curriculum_randomized_reset_tcp_standoff and curriculum_randomized_reset_tcp_jitter "
                "must keep every reset TCP above the source-cup grasp point."
            )
        minimum_standoff = math.sqrt(
            sum(
                max(abs(offset) - jitter, 0.0) ** 2
                for offset, jitter in zip(
                    self.curriculum_randomized_reset_tcp_standoff,
                    self.curriculum_randomized_reset_tcp_jitter,
                    strict=True,
                )
            )
        )
        if minimum_standoff + 1.0e-9 < self.curriculum_randomized_reset_tcp_min_grasp_distance:
            raise ValueError(
                "curriculum_randomized_reset_tcp_standoff and curriculum_randomized_reset_tcp_jitter "
                "cannot guarantee curriculum_randomized_reset_tcp_min_grasp_distance."
            )
        if self.curriculum_randomized_reset_ik_grid_size < 3 or self.curriculum_randomized_reset_ik_grid_size % 2 == 0:
            raise ValueError("curriculum_randomized_reset_ik_grid_size must be an odd integer of at least three.")
        if (
            self.curriculum_randomized_reset_ik_samples_per_source < 3
            or self.curriculum_randomized_reset_ik_samples_per_source % 2 == 0
        ):
            raise ValueError(
                "curriculum_randomized_reset_ik_samples_per_source must be an odd integer of at least three."
            )
        if self.curriculum_randomized_reset_ik_iterations <= 0:
            raise ValueError("curriculum_randomized_reset_ik_iterations must be positive.")
        for field_name in (
            "curriculum_randomized_reset_ik_max_cost",
            "curriculum_randomized_reset_ik_joint_margin",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and nonnegative.")

        source_outer_half_x = self.source_cup_inner_width / 2.0 + self.source_cup_wall_thickness
        source_outer_half_y = self.source_cup_inner_depth / 2.0 + self.source_cup_wall_thickness
        target_outer_half_y = self.target_cup_inner_depth / 2.0 + self.target_cup_wall_thickness
        maximum_projection_yaw = min(
            self.curriculum_randomized_source_yaw_range,
            math.atan2(source_outer_half_x, source_outer_half_y),
        )
        maximum_source_half_y = source_outer_half_x * math.sin(maximum_projection_yaw) + source_outer_half_y * math.cos(
            maximum_projection_yaw
        )
        minimum_separation = maximum_source_half_y + target_outer_half_y + self.curriculum_randomized_cup_clearance
        minimum_source_y = self.cup_reset_pos[1] - self.curriculum_randomized_source_position_range[1]
        minimum_target_y = (
            self.curriculum_randomized_target_center_xy[1] - self.curriculum_randomized_target_position_range[1]
        )
        if minimum_source_y - minimum_separation < minimum_target_y - 1.0e-6:
            raise ValueError(
                "curriculum_randomized_target_position_range leaves no collision-free target y-position "
                "at the minimum randomized source y-position."
            )

        arm_configs = (
            self.curriculum_pour_arm_q,
            self.curriculum_pour_target_arm_q,
            self.curriculum_carry_arm_q,
            self.arm_home,
        )
        for arm_q in arm_configs:
            if len(arm_q) != 7:
                raise ValueError("Every curriculum arm configuration must contain seven joint positions.")
            for joint_name, position in zip(self.actions.arm_action.joint_names, arm_q, strict=True):
                lower, upper = self.actions.arm_action.clip[joint_name]
                if not math.isfinite(position) or position < lower or position > upper:
                    raise ValueError(
                        f"Curriculum joint position {joint_name}={position} lies outside [{lower}, {upper}]."
                    )

    def _validate_particle_workspace_cfg(self) -> None:
        """Validate finite local particle bounds and all configured media reset poses."""
        if not math.isfinite(self.particle_max_velocity) or self.particle_max_velocity <= 0.0:
            raise ValueError("particle_max_velocity must be finite and positive.")
        if not math.isfinite(self.spill_table_height):
            raise ValueError("spill_table_height must be finite.")
        if not 0.0 < self.max_spill_fraction < 1.0:
            raise ValueError("max_spill_fraction must lie in (0, 1).")
        if not math.isfinite(self.success_dwell_time_s) or self.success_dwell_time_s <= 0.0:
            raise ValueError("success_dwell_time_s must be finite and positive.")
        if not math.isfinite(self.lost_grasp_dwell_time_s) or self.lost_grasp_dwell_time_s <= 0.0:
            raise ValueError("lost_grasp_dwell_time_s must be finite and positive.")
        for field_name in (
            "success_min_lift_height",
            "success_max_tcp_distance",
            "success_max_gripper_width_error",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive.")
        if not math.isfinite(self.state_bound_joint_position_margin) or self.state_bound_joint_position_margin < 0.0:
            raise ValueError("state_bound_joint_position_margin must be finite and nonnegative.")
        for field_name in (
            "state_bound_max_joint_velocity",
            "state_bound_max_cup_linear_velocity",
            "state_bound_max_cup_angular_velocity",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive.")
        if not math.isfinite(self.particle_count_margin) or self.particle_count_margin < 0.0:
            raise ValueError("particle_count_margin must be finite and nonnegative.")
        lower = self.particle_workspace_lower_bound
        upper = self.particle_workspace_upper_bound
        if len(lower) != 3 or len(upper) != 3:
            raise ValueError("particle_workspace bounds must each contain three coordinates.")
        if any(not math.isfinite(value) for value in (*lower, *upper)):
            raise ValueError("particle_workspace bounds must be finite.")
        if any(lo >= hi for lo, hi in zip(lower, upper, strict=True)):
            raise ValueError("particle_workspace lower bounds must be smaller than upper bounds.")
        if not lower[2] <= self.spill_table_height <= upper[2]:
            raise ValueError("spill_table_height must lie inside the particle workspace z bounds.")

        local_points = cup_cavity_lattice(self)[0]
        local_lo = local_points.min(axis=0)
        local_hi = local_points.max(axis=0)
        source_range = (*self.curriculum_randomized_source_position_range, 0.0)
        # A radial XY envelope is conservative for every configured upright yaw and keeps this
        # validation independent of the reset bank's finite angular samples.
        local_xy_radius = max(math.hypot(float(point[0]), float(point[1])) for point in local_points)
        source_lo = (
            self.cup_reset_pos[0] - source_range[0] - local_xy_radius,
            self.cup_reset_pos[1] - source_range[1] - local_xy_radius,
            float(local_lo[2] + self.cup_reset_pos[2]),
        )
        source_hi = (
            self.cup_reset_pos[0] + source_range[0] + local_xy_radius,
            self.cup_reset_pos[1] + source_range[1] + local_xy_radius,
            float(local_hi[2] + self.cup_reset_pos[2]),
        )
        target_local_lo, target_local_hi = cube_bowl_inner_bounds(
            self.target_cup_inner_width,
            self.target_cup_inner_depth,
            self.target_cup_cavity_depth,
            self.target_cup_bottom_thickness,
        )
        target_range = (*self.curriculum_randomized_target_position_range, 0.0)
        target_lo = tuple(
            float(point + position - extent - self.particle_count_margin)
            for point, position, extent in zip(
                target_local_lo,
                self.target_cup_reset_pos,
                target_range,
                strict=True,
            )
        )
        target_hi = tuple(
            float(point + position + extent + self.particle_count_margin)
            for point, position, extent in zip(
                target_local_hi,
                self.target_cup_reset_pos,
                target_range,
                strict=True,
            )
        )
        for region_name, region_lo, region_hi in (
            ("randomized source media", source_lo, source_hi),
            ("randomized target cavity", target_lo, target_hi),
        ):
            if any(value < bound for value, bound in zip(region_lo, lower, strict=True)) or any(
                value > bound for value, bound in zip(region_hi, upper, strict=True)
            ):
                raise ValueError(f"particle_workspace bounds do not contain the {region_name}.")

    def _apply_robot_cfg(self) -> None:
        """Apply final task fields to the scene robot and its joint-position action offset."""
        self.scene.robot.init_state.joint_pos.update(
            dict(zip([f"panda_joint{i}" for i in range(1, 8)], self.arm_home, strict=True))
        )
        self.scene.robot.init_state.joint_pos["panda_finger_joint.*"] = self.gripper_open_pos
        for actuator_name in ("panda_shoulder", "panda_forearm"):
            self.scene.robot.actuators[actuator_name].stiffness = self.arm_stiffness
            self.scene.robot.actuators[actuator_name].damping = self.arm_damping
            self.scene.robot.actuators[actuator_name].armature = self.arm_armature
        hand = self.scene.robot.actuators["panda_hand"]
        hand.stiffness = self.finger_stiffness
        hand.damping = self.finger_damping
        hand.armature = self.finger_armature

    def _colocate_large_environment_batch(self) -> None:
        """Colocate isolated physics worlds for numerically stable large-batch training."""
        if self.scene.num_envs <= 0:
            raise ValueError("scene.num_envs must be positive.")
        if self.colocate_physics_min_envs is None:
            return
        if not isinstance(self.colocate_physics_min_envs, int) or isinstance(self.colocate_physics_min_envs, bool):
            raise TypeError("colocate_physics_min_envs must be an integer.")
        if self.colocate_physics_min_envs <= 0:
            raise ValueError("colocate_physics_min_envs must be positive.")
        if self.scene.num_envs >= self.colocate_physics_min_envs and _mpm_solver_cfg(self).separate_worlds:
            self.scene.env_spacing = 0.0

    def _apply_solver_cfg_overrides(self) -> None:
        """Propagate final top-level controls into the constructed coupled-solver config."""
        coupled_cfg = self.sim.physics.solver_cfg
        arm_entries = [entry for entry in coupled_cfg.entries if entry.name == RIGID_ENTRY]
        if len(arm_entries) != 1:
            raise ValueError(f"Expected exactly one {RIGID_ENTRY!r} solver entry, found {len(arm_entries)}.")
        proxies = [
            proxy for proxy in coupled_cfg.proxies if proxy.source == RIGID_ENTRY and proxy.destination == MPM_ENTRY
        ]
        if len(proxies) != 1:
            raise ValueError(f"Expected exactly one {RIGID_ENTRY!r}-to-{MPM_ENTRY!r} proxy, found {len(proxies)}.")

        mpm_solver_cfg = _mpm_solver_cfg(self)
        mpm_solver_cfg.voxel_size = self.voxel_size
        mpm_solver_cfg.max_iterations = self.mpm_iterations
        arm_entries[0].substeps = self.num_substeps
        self.sim.physics.num_substeps = self.num_substeps
        self.sim.physics.use_cuda_graph = self.use_cuda_graph
        coupled_cfg.iterations = self.proxy_iterations
        proxies[0].mass_scale = self.proxy_mass_scale

    def finalize(self) -> FrankaPourEnvCfg:
        """Return an independent config with all derived scene assets resolved."""
        resolved = deepcopy(self)
        # Hydra and command-line overrides are applied after ``__post_init__`` constructs the
        # nested Newton solver tree. Reapply every public top-level solver control before derived
        # capacities inspect that tree.
        resolved._apply_solver_cfg_overrides()
        resolved._colocate_large_environment_batch()
        # Command-line overrides are applied after ``__post_init__``. Re-resolve the custom action
        # bound so its open target cannot diverge from the physical reset configuration.
        resolved.actions.gripper_action.close_position = max(
            0.0, resolved.gripper_preload_pos - resolved.gripper_close_offset
        )
        resolved.actions.gripper_action.open_position = resolved.gripper_open_pos
        if resolved.actions.gripper_action.limit_to_preload:
            resolved.actions.gripper_action.neutral_position = resolved.gripper_preload_pos
        else:
            resolved.actions.gripper_action.neutral_position = resolved.gripper_open_pos
            resolved.actions.gripper_action.default_position = resolved.gripper_open_pos
            resolved.actions.gripper_action.scale = (
                resolved.gripper_open_pos - resolved.actions.gripper_action.close_position
            )
        progress_params = resolved.rewards.task_progress.params
        progress_params["grasp_preload_position"] = resolved.gripper_preload_pos
        progress_params["source_offset_xy"] = resolved.pour_source_offset_xy
        progress_params["source_mouth_height"] = resolved.source_cup_bottom_thickness + resolved.source_cup_cavity_depth
        progress_params["min_lift_height"] = resolved.success_min_lift_height
        progress_params["max_tcp_distance"] = resolved.success_max_tcp_distance
        progress_params["max_gripper_width_error"] = resolved.success_max_gripper_width_error
        progress_params["max_gripper_command"] = resolved.gripper_preload_pos
        resolved.rewards.delivered.params["min_lift_height"] = resolved.success_min_lift_height
        resolved.rewards.delivered.params["max_tcp_distance"] = resolved.success_max_tcp_distance
        resolved.rewards.delivered.params["max_gripper_width_error"] = resolved.success_max_gripper_width_error
        resolved.rewards.delivered.params["max_gripper_command"] = resolved.gripper_preload_pos
        resolved.terminations.lost_grasp.params["dwell_time_s"] = resolved.lost_grasp_dwell_time_s
        resolved.terminations.lost_grasp.params["max_tcp_distance"] = resolved.success_max_tcp_distance
        resolved.terminations.lost_grasp.params["max_gripper_width_error"] = resolved.success_max_gripper_width_error
        resolved.terminations.lost_grasp.params["max_gripper_command"] = resolved.gripper_preload_pos
        if isinstance(resolved.actions.arm_action, mdp.TrajectoryJointPositionActionCfg):
            resolved.actions.arm_action.grasp_max_tcp_distance = resolved.success_max_tcp_distance
        resolved._validate_curriculum_cfg()
        resolved._validate_particle_workspace_cfg()
        resolved._apply_robot_cfg()
        resolved.terminations.success.params["dwell_time_s"] = resolved.success_dwell_time_s
        resolved.terminations.success.params["min_lift_height"] = resolved.success_min_lift_height
        resolved.terminations.success.params["max_tcp_distance"] = resolved.success_max_tcp_distance
        resolved.terminations.success.params["max_gripper_width_error"] = resolved.success_max_gripper_width_error
        resolved.terminations.success.params["max_gripper_command"] = resolved.gripper_preload_pos
        resolved.scene.source_cup = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/SourceCup",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=resolved.cup_reset_pos,
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
            spawn=CubeBowlSpawnerCfg(
                inner_width=resolved.source_cup_inner_width,
                inner_depth=resolved.source_cup_inner_depth,
                cavity_depth=resolved.source_cup_cavity_depth,
                wall_thickness=resolved.source_cup_wall_thickness,
                bottom_thickness=resolved.source_cup_bottom_thickness,
                display_color=(0.95, 0.82, 0.16),
                grasp_proxy_half_extents=resolved.cup_grasp_box_half,
                mass_props=MassCfg(mass=resolved.cup_mass),
                rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=False),
                collision_props=UsdPhysicsCollisionCfg(collision_enabled=True),
                physics_material=RigidBodyMaterialBaseCfg(
                    static_friction=resolved.cup_grasp_box_friction,
                    dynamic_friction=resolved.cup_grasp_box_friction,
                ),
            ),
        )
        resolved.scene.target_cup = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TargetCup",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=resolved.target_cup_reset_pos,
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
            spawn=CubeBowlSpawnerCfg(
                inner_width=resolved.target_cup_inner_width,
                inner_depth=resolved.target_cup_inner_depth,
                cavity_depth=resolved.target_cup_cavity_depth,
                wall_thickness=resolved.target_cup_wall_thickness,
                bottom_thickness=resolved.target_cup_bottom_thickness,
                display_color=(0.20, 0.55, 0.90),
                grasp_proxy_half_extents=None,
                rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
                physics_material=RigidBodyMaterialBaseCfg(
                    static_friction=resolved.target_cup_friction,
                    dynamic_friction=resolved.target_cup_friction,
                ),
            ),
        )
        resolved.scene.media = build_media_object_cfg(
            resolved,
            resolved.cup_reset_pos,
            (0.0, 0.0, 0.0, 1.0),
        )
        _mpm_solver_cfg(resolved).max_active_cell_count = _resolve_mpm_cell_cap(resolved)
        _mpm_solver_cfg(resolved).max_upper_node_count = _resolve_mpm_upper_node_cap(resolved)
        resolved.sim.physics.scene_cfg = resolved.scene
        return resolved


@configclass
class FrankaPourEnvCfg_PLAY(FrankaPourEnvCfg):
    def __post_init__(self):
        self.use_cuda_graph = True
        super().__post_init__()
        self.scene.num_envs = 4
        self.curriculum_start_stage = len(self.curriculum_stage_names) - 1
        self.curriculum_randomization_start_level = len(self.curriculum_randomization_extent_levels) - 1
        self.curriculum_freeze = True


@configclass
class FrankaPourEnvCfg_TELEOP(FrankaPourEnvCfg_PLAY):
    """Teleop preset: 1 env, no RL time-out (operator resets manually)."""

    def __post_init__(self):
        super().__post_init__()
        # SpaceMouse IK emits direct seven-joint targets; keep that operator-only interface
        # separate from the policy's phase-plus-residual action representation.
        joint_clip = self.actions.arm_action.clip
        self.actions.arm_action = mdp.CurriculumJointPositionActionCfg(
            asset_name="robot",
            joint_names=[f"panda_joint{i}" for i in range(1, 8)],
            scale=0.5,
            alpha=0.2,
            project_reference_through_stage=-1,
            use_default_offset=True,
            preserve_order=True,
            clip=joint_clip,
        )
        self.actions.gripper_action.force_open_before_phase_stage = -1
        self.actions.gripper_action.limit_to_preload = False
        self.actions.gripper_action.neutral_position = self.gripper_open_pos
        self.actions.gripper_action.default_position = self.gripper_open_pos
        self.actions.gripper_action.scale = self.gripper_open_pos - self.actions.gripper_action.close_position
        self.scene.num_envs = 1
        self.terminations.time_out = None
        self.episode_length_s = 3600.0
