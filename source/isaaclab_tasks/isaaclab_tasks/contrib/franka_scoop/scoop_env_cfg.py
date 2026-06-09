# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka scoop two-container MPM transfer environment."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from isaaclab_newton.physics import (
    CoupledSolverCfg,
    CoupledSolverEntryCfg,
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonCfg,
)

from . import mdp

RIGID_ENTRY = "arm"
MPM_ENTRY = "media"


@configclass
class ScoopSceneCfg(InteractiveSceneCfg):
    """Scene with a standard IsaacLab Franka plus task-specific MPM extras."""

    # The task frame follows the Franka reach tasks: table top at z=0, floor below the table.
    # ``FrankaScoopEnvCfg.__post_init__`` sets the exact floor height from ``table_half``.
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light",
                         spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0))
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ActionsCfg:
    scoop = mdp.ScoopActionCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Realistic actor observations: top-down heightfield + proprioception."""

        heightfield = ObsTerm(func=mdp.heightfield_obs)
        particle_summary = ObsTerm(func=mdp.particle_summary_obs)
        arm_q = ObsTerm(func=mdp.arm_joint_pos_norm)
        arm_qd = ObsTerm(func=mdp.arm_joint_vel_scaled)
        bowl_pose = ObsTerm(func=mdp.bowl_pose_obs)
        to_source = ObsTerm(func=mdp.bowl_to_source_obs)
        to_target = ObsTerm(func=mdp.bowl_to_target_obs)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Privileged sim state for the critic only (not available on a real robot)."""

        in_bowl = ObsTerm(func=mdp.count_in_bowl_obs)
        in_source = ObsTerm(func=mdp.count_in_source_obs)
        in_target = ObsTerm(func=mdp.count_in_target_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class RewardsCfg:
    # Sparse, outcome-only rewards for FILLING the cup from the pile. No distance/pose shaping: the easy
    # start (cup tilted right in front of the pile) makes the objective reachable by exploration.
    #   fill          = particles currently in the cup (the objective; the bowl counter is reliable here).
    #   removed_source = light bootstrap for getting media moving out of the pile.
    #   success        = sparse stage bonus once the cup holds the curriculum's required count.
    # Dense reach/carry and target-delivery are dropped (delivery becomes a later-stage extension).
    fill = RewTerm(func=mdp.particles_in_bowl, weight=6.0)
    removed_source = RewTerm(func=mdp.removed_from_source, weight=1.0, params={"norm": 100.0})
    success = RewTerm(func=mdp.transfer_success_bonus, weight=10.0)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.002)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    failure = DoneTerm(func=mdp.nonfinite_failure)


@configclass
class EventsCfg:
    reset_scene = EventTerm(func=mdp.reset_scoop_scene, mode="reset")


@configclass
class CurriculumCfg:
    stage = CurrTerm(func=mdp.ScoopCurriculum)


@configclass
class FrankaScoopEnvCfg(ManagerBasedRLEnvCfg):
    """Franka scoop transferring MPM media from a source to a target container."""

    # env_spacing packs envs tightly so the dense fixed MPM grid (which spans the multi-env bbox) stays small;
    # 0.9 m clears the pile/box footprint. Required for fixed grid + cuda graph to fit at high env counts.
    scene: ScoopSceneCfg = ScoopSceneCfg(num_envs=2, env_spacing=0.9, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # ---- Franka layout ----
    hand_home_pos: tuple = (0.52, -0.23, 0.258)  # env frame: reset over the source bucket, above the table
    hand_home_quat: tuple = (0.70710678, 0.0, 0.70710678, 0.0)  # xyzw; hand +X/cup opening is world-up
    # Reset/spawn arm config (used directly as the reset, since the one-shot reset IK diverges). This is the
    # user's teleop-found scoop-start: cup at ~(0.18,-0.18,0.18) tilted ~2.06 rad, poised above the pile,
    # empty -- it descends + digs to fill. Captured via the stable DiffIK to 5 mm. See capture_reset_config.py.
    arm_home: tuple = (-0.3268, -0.3751, -0.2812, -2.8079, -0.3870, 1.9367, 0.4461)
    arm_stiffness: float = 600.0
    arm_damping: float = 50.0
    hide_robot_visual_shapes_in_newton: bool = False  # show the robot's visual meshes in the Newton viewer too.

    # ---- scoop-bowl EE (gripped cup; home orientation is exactly opening-up) ----
    gripper_open_pos: float = 0.04                 # fixed Panda finger opening; not exposed as an action
    bowl_reach: float = 0.0584                     # fallback procedural bowl centre along +Z_hand [m]
    bowl_home_offset: tuple = (0.06, 0.0, -0.078)  # home bowl-centre target offset from hand (env frame) [m]
    home_pitch: float = 0.0                        # neutral/ready bowl tilt: opening points up to hold media [rad]
    ee_bowl_scale: float = 0.20                    # fallback scale for the procedural pour-demo bowl
    ee_bowl_friction: float = 0.05
    collider_margin: float = 0.002                 # pour-demo style MPM collider margin
    # Watertight scoop cup (own asset, cup_mesh.make_cup_collision_mesh). Walls + bottom are >= ~1.5 MPM
    # voxels (voxel 0.01) so the cup is a solid barrier on the grid and media cannot tunnel through; the
    # ~74 mm cavity spans ~7 cells for retention. It is a COLLIDE_PARTICLES-only shape, so the wider outer
    # wall harmlessly overlaps the (rigid-only) fingers.
    ee_cup_inner_bottom_radius: float = 0.029
    ee_cup_inner_top_radius: float = 0.037
    ee_cup_wall_thickness: float = 0.012
    ee_cup_height: float = 0.096               # total: bottom_thickness (0.016) + cavity_depth (0.080)
    ee_cup_bottom_thickness: float = 0.016
    gripped_cup_hand_front_z: float = 0.066       # front edge of Panda hand collision in panda_hand frame
    gripped_cup_base_clearance: float = 0.005      # near cup wall sits 5 mm in front of panda_hand base
    gripped_cup_usd_path: str = (
        "omniverse://isaac-dev.ov.nvidia.com/Isaac/SimReady/Residential/Kitchen/Dishware/"
        "Coffee_Cup_A01/sm_food_beverage_coffeeCup_a01_01.usd"
    )
    gripped_cup_usd_prim_path: str = (
        "/RootNode/Geometry/sm_food_beverage_coffeeCup_a01_body_obj_00/"
        "sm_food_beverage_coffeeCup_a01_body_mesh_00"
    )
    gripped_cup_visual_scale: float = 1.0           # multiplier after auto-fit to the collision proxy diameter
    gripped_cup_visual_offset: tuple = (0.0, 0.0, 0.0)
    gripped_cup_visual_quat: tuple = (0.0, 0.0, 0.0, 1.0)
    gripped_cup_auto_fit_visual: bool = True        # center/bottom-align referenced USD with the open collision proxy

    # ---- containers + media ----
    # Two simple source/target buckets sitting flat on the table top (env-frame z=0). ``source_center`` and
    # ``target_center`` are the bucket interior centers used by observations/rewards; bucket base z is
    # ``center_z - 0.5 * bucket_height``.
    # Scoop-from-a-pile layout: the SOURCE is a free granular PILE on the table (a natural cone at the
    # media's angle of repose), retained only by a very shallow 4-wall box so the base does not spill.
    # ``source_center`` z == container_inner_half z so the box floor sits on the table top (env z=0).
    # ``container_inner_half`` is the SOURCE-region footprint (xy must clear the cone base) and the
    # particle-counting region half-height (z must cover the full pile). The retaining-box WALL height is
    # the separate, much shallower ``pile_box_wall_half``. The target box is parked clear (later-stage pour).
    # Pile sits under where the cup's home/ready pose naturally rests (~x=0.16, y=-0.23) so the per-reset
    # IK only has to make a SMALL move (down + tilt) onto it -- a large one-shot reset move makes the DLS
    # solver diverge. Target box is parked clear on the +y side (later-stage pour).
    source_center: tuple = (0.32, -0.23, 0.060)
    target_center: tuple = (0.32, 0.22, 0.060)
    container_inner_half: tuple = (0.14, 0.14, 0.060)
    container_wall: float = 0.012
    media_fill_frac: float = 0.80
    # ---- granular pile (source) ----
    pile_box_wall_half: float = 0.015         # retaining-box wall half-height [m] -> 3 cm walls
    pile_height: float = 0.150                # natural pile (cone apex) height above the table [m]
    pile_jitter: float = 0.004                # per-particle surface noise on the spawned pile [m]
    # Pile side slope = angle of repose; ~atan(sand_friction) for dry cohesionless granular media.

    # ---- procedural containers + bolt-on table ----
    container_geometry: str = "box"          # "box" (shallow pile retainer), "bucket", or "pour_bowl"
    use_pour_bowl_mesh: bool = False         # legacy alias; use container_geometry="pour_bowl" for old bowls
    bucket_inner_radius: float = 0.120
    bucket_wall_thickness: float = 0.017
    bucket_height: float = 0.160
    bucket_bottom_thickness: float = 0.017
    bucket_mesh_segments: int = 32
    bucket_rigid_wall_segments: int = 16
    bowl_target_diameter: float = 0.20       # uniform-scale source/target bowl outer rim diameter [m]
    rigid_bowl_wall_segments: int = 16       # open rigid bowl proxies; avoids MuJoCo convex mesh caps
    pedestal_half: tuple = (0.055, 0.055)    # only used by legacy helper/fallback geometry
    table_half: tuple = (0.45, 0.55, 0.50)   # compact table; enough room for bowls, smaller fixed MPM grid
    table_center_xy: tuple = (0.50, 0.00)    # centered under the source/target bowl pair, top at env-frame z=0
    sand_density: float = 1500.0
    # Free-flowing granular sand, matching Newton's MPM granular example: cohesionless and
    # undamped, moderate friction, finite yield pressure so it shears/flows under load. Cohesion
    # (tensile_yield_ratio > 0) or a large damping relaxation time make the media clump into a
    # sticky blob that "clunks" together when the scoop bowl passes through it.
    sand_friction: float = 0.7
    sand_damping: float = 0.0              # elastic relaxation time [s]; 0 = undamped granular flow
    sand_tensile_yield_ratio: float = 0.0  # 0 = cohesionless sand (no tensile strength)
    sand_yield_pressure: float = 1.0e12    # [Pa] pressure cap; below the 1e15 default so it yields/flows
    sand_young_modulus: float = 1.0e15     # [Pa] near-incompressibility penalty (Newton MPM default)

    # ---- MPM ----
    # voxel must resolve the small 1/4-scale scoop bowl: at 0.01 m the rim spans ~10 cells;
    # at 0.03 m it is only ~3 cells and particles tunnel/leak. Particle spacing is tied to the voxel via
    # particles_per_cell (Newton MPM best practice), so particle size always tracks voxel_size.
    voxel_size: float = 0.015
    particles_per_cell: float = 2.0  # MPM particle samples per voxel per axis (media spacing = voxel_size / this)
    mpm_iterations: int = 24
    mpm_grid_padding: int = 8
    # For a FIXED grid this hard-preallocates the FEM/BSR matrices. The pile_height=0.10 pile is ~14k
    # particles/env (~1.36M total at 96 envs) -> ~300k active cells after settling; 380k covers it with
    # headroom while keeping the preallocation within the shared GPU's free memory (~24 GB).
    mpm_max_active_cells: int = 380000
    num_substeps: int = 2                   # finer physics substep (dt/4) for better scoop-bowl<->MPM contact
    coupling_type: str = "base"            # robot is rigid-solved; MPM sees kinematic/static colliders
    grid_type: str = "fixed"
    use_cuda_graph: bool = True

    # ---- IK (bowl position + pitch) ----
    # Table-level reach band (env frame, above the table top): the arm reaches forward-and-down to
    # the bowls sitting on the table, dipping the cup to ~z=0.03 to scoop and lifting to ~0.20 to
    # clear and carry between bowls.
    workspace_lo: tuple = (0.10, -0.34, 0.020)
    workspace_hi: tuple = (0.52, 0.34, 0.30)
    ik_position_weight: float = 10.0
    ik_rotation_weight: float = 5.0          # hold the bowl orientation steady during motion (anti-spill)
    ik_joint_limit_weight: float = 10.0
    ik_lambda_initial: float = 0.1        # LM damping (multi-seed escapes local minima; lambda just stabilizes)
    ik_step_size: float = 0.5             # LM step
    ik_iterations: int = 12
    reset_ik_iterations: int = 60          # LM iters per seed at reset (x n_seeds; reset is occasional)
    ik_backend: str = "diffik"             # "diffik" for smooth runtime tracking, "newton" for full-pose IK
    diffik_lambda: float = 0.05
    diffik_max_delta: float = 0.05         # per-step joint delta clamp for DiffIK runtime tracking [rad]
    max_ik_delta: float = 0.05             # full-IK runtime joint delta clamp [rad]; 0 disables it.
    cartesian_action_scale: float = 0.30     # m/s per unit action
    pitch_action_scale: float = 3.0          # rad/s per unit action
    min_pitch: float = -1.4                  # tilt bowl opening toward -X [rad]
    max_pitch: float = 2.6                   # max bowl tilt [rad] (allows near-inversion to pour)
    action_smoothing: float = 0.4

    # ---- heightfield obs (env frame) ----
    heightfield_lo: tuple = (0.10, -0.40, -0.02)
    heightfield_hi: tuple = (0.46, 0.40, 0.26)
    heightfield_size: int = 16

    # ---- success / curriculum (mutated by ScoopCurriculum) ----
    # The watertight cup holds thousands of particles, so success is counted in hundreds, not single digits.
    # ``curriculum_target_count`` is particles-in-cup required for success per stage (placeholder scale to
    # tune once training shows a typical scoop size); ``success_particle_count`` normalizes the fill reward.
    success_particle_count: float = 500.0    # fill-reward normalization scale (a "good scoop" ~ this many)
    scoop_target_count: float = 50.0         # runtime curriculum target (particles in cup); set by the curriculum
    curriculum_target_count: tuple = (50, 120, 250, 400, 600)
    # Reset scoop target offset relative to the actual reset source-particle centroid. Stage 0 starts
    # essentially inside the source pile; later stages back the arm away toward the normal home hover.
    # Easy start: the cup resets OPENING-UP just above the source-media surface (offset is relative to the
    # source pile centroid). z offsets keep the cup cavity floor ABOVE the media (not spawned inside it);
    # the policy pitches to dip in. Difficulty ramps mainly via target count + pile jitter below.
    # Cup resets TILTED, just IN FRONT of the pile (robot side), not inside it -> a small move into the pile
    # fills it. Offsets are relative to the source-pile centroid. Reset uses a damped IK (small step + many
    # iters) so these close targets converge instead of the DLS overshooting/diverging.
    reset_start: str = "source_curriculum"
    curriculum_start_bowl_offset: tuple = (
        (-0.12, 0.00, 0.060),
        (-0.13, 0.00, 0.080),
        (-0.14, 0.00, 0.100),
        (-0.15, 0.00, 0.130),
        (-0.16, 0.00, 0.160),
    )
    # Pitch STATE at reset must match the fixed arm_home config's tilt (~2.06) so the action term HOLDS the
    # scoop-tilt instead of righting the cup. Difficulty ramps via target count + pile jitter, not start pose.
    curriculum_start_pitch: tuple = (2.06, 2.06, 2.06, 2.06, 2.06)
    curriculum_pile_xy_jitter: tuple = (0.0, 0.005, 0.01, 0.02, 0.03)
    curriculum_success_threshold: float = 0.35
    curriculum_min_resets_per_stage: int = 150
    curriculum_success_ema_alpha: float = 0.05

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 7.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        table_floor_z = -2.0 * float(self.table_half[2])
        self.scene.ground.init_state = AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, table_floor_z))
        self.viewer.eye = (1.4, 1.4, 0.9)
        self.viewer.lookat = (0.50, 0.00, 0.06)
        self.scene.robot.init_state.joint_pos.update(dict(zip(
            ("panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"),
            self.arm_home,
            strict=True,
        )))
        self.scene.robot.init_state.joint_pos["panda_finger_joint.*"] = self.gripper_open_pos
        # The cup is gripped by construction. The Panda fingers are fixed-open joints in this task,
        # not controlled DoFs, so the high-gain default gripper actuator must not fight the state pin.
        self.scene.robot.actuators.pop("panda_hand", None)
        for actuator_name in ("panda_shoulder", "panda_forearm"):
            self.scene.robot.actuators[actuator_name].stiffness = self.arm_stiffness
            self.scene.robot.actuators[actuator_name].damping = self.arm_damping
            self.scene.robot.actuators[actuator_name].armature = 0.1

        self.sim.physics = NewtonCfg(
            solver_cfg=CoupledSolverCfg(
                coupling_type=self.coupling_type,
                scene_cfg=self.scene,
                entries=[
                    CoupledSolverEntryCfg(
                        name=RIGID_ENTRY,
                        solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True, njmax=510, nconmax=400),
                        # Franka collides with the table/ground static shapes and with rigid-only
                        # source/target bowl copies. Normal robot links remain out of MPM particle collision.
                        body_entities=[SceneEntityCfg("robot")],
                        body_label_patterns=[r".*/Source(?:Bowl|Bucket)Rigid$", r".*/Target(?:Bowl|Bucket)Rigid$"],
                        include_static_shapes=True,
                        substeps=self.num_substeps),
                    CoupledSolverEntryCfg(
                        name=MPM_ENTRY,
                        solver_cfg=MPMSolverCfg(voxel_size=self.voxel_size, grid_type=self.grid_type,
                                                grid_padding=self.mpm_grid_padding,
                                                max_active_cell_count=self.mpm_max_active_cells,
                                                strain_basis="P0", transfer_scheme="apic",
                                                max_iterations=self.mpm_iterations, collider_velocity_mode="backward",
                                                solver="gauss-seidel"),
                        all_particles=True,
                        body_label_patterns=[r".*/ScoopBowl$", r".*/Source(?:Bowl|Bucket)$", r".*/Target(?:Bowl|Bucket)$"],
                        # MUST be False: the giant static ground plane as an MPM collider overflows the
                        # collider rasterization at high env counts (CUDA error 700). The pile-retaining
                        # boxes are kinematic bodies selected via body_label_patterns instead.
                        include_static_shapes=False,
                        include_child_joints=False,
                        in_place=True),
                ],
                use_collision_pipeline=False,
            ),
            num_substeps=self.num_substeps, use_cuda_graph=self.use_cuda_graph,
        )


@configclass
class FrankaScoopEnvCfg_PLAY(FrankaScoopEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
