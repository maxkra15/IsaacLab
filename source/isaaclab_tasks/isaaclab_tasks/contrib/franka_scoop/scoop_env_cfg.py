# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka scoop two-container MPM transfer environment."""

from __future__ import annotations

from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    MPMSolverCfg,
)
from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg

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

from isaaclab_contrib.coupling import CoupledSolverCfg, CoupledSolverEntryCfg
from isaaclab_contrib.deformable.newton_manager_cfg import CoupledNewtonCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from . import mdp

RIGID_ENTRY = "arm"
MPM_ENTRY = "media"


@configclass
class ScoopSceneCfg(InteractiveSceneCfg):
    """Scene with a standard IsaacLab Franka plus task-specific MPM extras."""

    # The task frame follows the Franka reach tasks: table top at z=0, floor below the table.
    # ``FrankaScoopEnvCfg.__post_init__`` sets the exact floor height from ``table_half``.
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0)
    )
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # The MPM media as a declarative scene entity. The spawn is built at env-construction time
    # (``FrankaScoopEnv._prepare_newton_extras`` -> ``build_media_spawn_cfg``) so post-construction
    # cfg overrides (hydra, play/train scripts) still reach the particle bed. Particle reads/
    # writes/resets go through the :class:`~isaaclab_newton.assets.MPMObject` asset API; Kit
    # point-cloud visualization is native.
    media: MPMObjectCfg | None = None


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
    # Scoop-AND-DUMP, OUTCOME-only (no distance/pose shaping -- the curriculum, not shaping, bridges the
    # exploration gap; early stages start the cup pre-loaded so dumping is reachable):
    #   delivered      = particles currently in the target bowl (the objective; dense outcome). Pays every
    #                    step after delivery, so delivering EARLY collects more -- this, not a success
    #                    terminal, is what makes delivering beat holding the media until time-out.
    #   success        = per-step bonus while > scoop_target_count particles are delivered; also latches
    #                    the per-episode success flag for the curriculum (delivery_success_mask).
    #   fill           = particles in the cup -- a bootstrap that only matters on the scoop stages.
    #   removed_source = light bootstrap for getting media out of the pile (scoop stages).
    delivered = RewTerm(func=mdp.particles_in_target, weight=6.0)
    success = RewTerm(func=mdp.delivery_success_bonus, weight=10.0)
    fill = RewTerm(func=mdp.particles_in_bowl, weight=2.0)
    removed_source = RewTerm(func=mdp.removed_from_source, weight=0.5, params={"norm": 100.0})
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.002)


@configclass
class TerminationsCfg:
    # Delivery success is deliberately NOT a terminal: ending the episode at success would cut off the
    # dense post-delivery reward stream and make "hold the media until time-out" out-pay delivering.
    # Success is latched per episode by the success reward (delivery_success_mask) for the curriculum.
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
    # Reflected rotor inertia added to every arm joint. The coupled solver runs explicit Euler at
    # the 1/120 s substep (implicitfast diverges the MPM coupling), so the PD damping must satisfy
    # kd*dt/I < 2. At the stock armature 0.1 the low-inertia distal joints sat ABOVE that bound and
    # the arm held a sustained ~1 rad/s limit-cycle jitter (the visible teleop "wobble") that
    # sloshed the cup empty. 0.5 lifts I 5x -> kd*dt/I ~ 0.7 with margin, dropping the held-pose
    # jitter ~28x without lowering the damping authority needed for tracking.
    arm_armature: float = 0.5
    hide_robot_visual_shapes_in_newton: bool = False  # show the robot's visual meshes in the Newton viewer too.

    # ---- scoop-bowl EE (gripped cup; home orientation is exactly opening-up) ----
    # Fixed Panda finger opening; not exposed as an action. 1 mm INSIDE the 0.04 joint limit so the
    # PD-held fingers do not sit exactly on the limit constraint (limit chatter under disturbances).
    gripper_open_pos: float = 0.039
    bowl_reach: float = 0.0584  # fallback procedural bowl centre along +Z_hand [m]
    bowl_home_offset: tuple = (0.06, 0.0, -0.078)  # home bowl-centre target offset from hand (env frame) [m]
    home_pitch: float = 0.0  # neutral/ready bowl tilt: opening points up to hold media [rad]
    ee_bowl_scale: float = 0.20  # fallback scale for the procedural pour-demo bowl
    ee_bowl_friction: float = 0.05
    collider_margin: float = 0.002  # pour-demo style MPM collider margin
    # Watertight scoop cup (own asset, cup_mesh.make_cup_collision_mesh). Walls + bottom are >= ~1.5 MPM
    # voxels (voxel 0.01) so the cup is a solid barrier on the grid and media cannot tunnel through; the
    # ~74 mm cavity spans ~7 cells for retention. It is a COLLIDE_PARTICLES-only shape, so the wider outer
    # wall harmlessly overlaps the (rigid-only) fingers.
    # Scoop shape: "hemisphere" = simple thick spherical-shell ladle (grid-robust; cavity depth == radius,
    # shell thickness == wall, walls stay solid on the MPM grid at voxel 0.015). "mug" = the flared
    # make_cup_collision_mesh cup below (with the decorative coffee-cup USD visual when gripped_cup_usd_path set).
    ee_cup_shape: str = "hemisphere"
    ee_ladle_radius: float = 0.045  # hemisphere cavity radius == cavity depth [m]; opening diameter = 2x
    ee_ladle_wall_thickness: float = 0.024  # shell thickness [m]; >= ~1.5*voxel_size (0.015) so media can't tunnel
    ee_cup_inner_bottom_radius: float = 0.029  # --- "mug" shape params (ee_cup_shape="mug") ---
    ee_cup_inner_top_radius: float = 0.037
    ee_cup_wall_thickness: float = 0.012
    ee_cup_height: float = 0.096  # total: bottom_thickness (0.016) + cavity_depth (0.080)
    ee_cup_bottom_thickness: float = 0.016
    gripped_cup_hand_front_z: float = 0.066  # front edge of Panda hand collision in panda_hand frame
    gripped_cup_base_clearance: float = 0.005  # near cup wall sits 5 mm in front of panda_hand base
    gripped_cup_usd_path: str = (
        "omniverse://isaac-dev.ov.nvidia.com/Isaac/SimReady/Residential/Kitchen/Dishware/"
        "Coffee_Cup_A01/sm_food_beverage_coffeeCup_a01_01.usd"
    )
    gripped_cup_usd_prim_path: str = (
        "/RootNode/Geometry/sm_food_beverage_coffeeCup_a01_body_obj_00/sm_food_beverage_coffeeCup_a01_body_mesh_00"
    )
    gripped_cup_visual_scale: float = 1.0  # multiplier after auto-fit to the collision proxy diameter
    gripped_cup_visual_offset: tuple = (0.0, 0.0, 0.0)
    gripped_cup_visual_quat: tuple = (0.0, 0.0, 0.0, 1.0)
    gripped_cup_auto_fit_visual: bool = True  # center/bottom-align referenced USD with the open collision proxy

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
    # NOTE the layout constraint: source/target centers are 0.45 m apart (and the -y neighbour
    # env's target box sits 0.45 m beyond the source box at env_spacing 0.9), so the boxes need
    # 2*(inner_half_y + wall) < 0.45 with >= ~3 voxels clearance -- inner_half_y <= 0.175. At
    # (0.2, 0.2) the two boxes of one env physically overlapped and neighbouring envs' boxes came
    # within 2 mm (< 1 voxel) of each other in the shared MPM grid.
    container_inner_half: tuple = (0.2, 0.175, 0.060)
    # Wall thickness must stay >= ~1.5 MPM voxels or pile-edge particles seep through the
    # retaining box (the MPM grid cannot represent a sub-voxel solid wall).
    container_wall: float = 0.024
    media_fill_frac: float = 0.80
    # ---- granular pile (source) ----
    pile_box_wall_half: float = 0.015  # retaining-box wall half-height [m] -> 3 cm walls
    pile_height: float = 0.50  # natural pile (cone apex) height above the table [m]
    pile_jitter: float = 0.004  # per-particle surface noise on the spawned pile [m]
    # Pile side slope = angle of repose; ~atan(media_material.friction) for dry cohesionless granular media.

    # ---- procedural containers + bolt-on table ----
    container_geometry: str = "box"  # "box" (shallow pile retainer), "bucket", or "pour_bowl"
    use_pour_bowl_mesh: bool = False  # legacy alias; use container_geometry="pour_bowl" for old bowls
    bucket_inner_radius: float = 0.120
    bucket_wall_thickness: float = 0.017
    bucket_height: float = 0.160
    bucket_bottom_thickness: float = 0.017
    bucket_mesh_segments: int = 32
    bucket_rigid_wall_segments: int = 16
    bowl_target_diameter: float = 0.20  # uniform-scale source/target bowl outer rim diameter [m]
    rigid_bowl_wall_segments: int = 16  # open rigid bowl proxies; avoids MuJoCo convex mesh caps
    pedestal_half: tuple = (0.055, 0.055)  # only used by legacy helper/fallback geometry
    table_half: tuple = (0.43, 0.55, 0.50)  # compact table; enough room for bowls, smaller fixed MPM grid
    # Shifted -x so the Franka base (bolted at env x=0) sits ON the table: front edge = 0.37 - 0.43 = -0.06,
    # just behind the ~6 cm base footprint; back edge 0.80 still clears the containers (max x ~0.47). Top at z=0.
    table_center_xy: tuple = (0.30, 0.00)
    # Free-flowing granular sand, matching Newton's MPM granular example: cohesionless and
    # undamped, moderate friction, finite yield pressure (below the 1e15 default) so it
    # shears/flows under load. Cohesion (tensile_yield_ratio > 0) or a large damping
    # relaxation time make the media clump into a sticky blob that "clunks" together when
    # the scoop bowl passes through it. ``friction`` also sets the pile's angle of repose.
    media_material: MPMParticleMaterialCfg = MPMParticleMaterialCfg(
        density=1500.0,
        friction=0.7,
        yield_pressure=1.0e12,
    )

    # ---- MPM ----
    # voxel must resolve the small 1/4-scale scoop bowl: at 0.01 m the rim spans ~10 cells;
    # at 0.03 m it is only ~3 cells and particles tunnel/leak. Particle spacing is tied to the voxel via
    # particles_per_cell (Newton MPM best practice), so particle size always tracks voxel_size.
    voxel_size: float = 0.015
    particles_per_cell: float = 2.0  # MPM particle samples per voxel per axis (media spacing = voxel_size / this)
    mpm_iterations: int = 24
    mpm_grid_padding: int = 32
    # On episode reset, restore the per-particle MPM state (elastic strain/transform -> identity, Jp -> 1,
    # APIC velocity gradient/stress -> 0) for the reset envs to its rest values, for a deterministic stress-free
    # fresh pile. NOTE: for the current granular rheology this is effectively a no-op (plastic flow keeps the
    # elastic strain at identity); it does NOT explain the post-reset "pop" after a dynamic episode -- that
    # residual was traced to the solver's grid scratchpad (not cleanly resettable from here). Default on.
    reset_mpm_particle_state: bool = True
    # For a FIXED grid this hard-preallocates the FEM/BSR matrices. The pile_height=0.10 pile is ~14k
    # particles/env (~1.36M total at 96 envs) -> ~300k active cells after settling; 380k covers it with
    # headroom while keeping the preallocation within the shared GPU's free memory (~24 GB).
    mpm_max_active_cells: int = 380000
    num_substeps: int = 2  # finer physics substep (dt/4) for better scoop-bowl<->MPM contact
    # Training default is the SPARSE grid: at training-scale env counts it beats the fixed grid
    # on both throughput and memory (see _scratch/reports/scoop_env_scaling_report.tex). Sparse
    # requires grid_padding=0 (its allocator dilates per voxel) and is not CUDA-graph capturable;
    # __post_init__ derives both from grid_type. The PLAY cfg switches to fixed + CUDA graph.
    grid_type: str = "sparse"
    use_cuda_graph: bool = True  # honored for the fixed grid only

    # ---- IK (bowl position + pitch) ----
    # Table-level reach band (env frame, above the table top): the arm reaches forward-and-down to
    # the bowls sitting on the table, dipping the cup to ~z=0.03 to scoop and lifting to ~0.20 to
    # clear and carry between bowls.
    workspace_lo: tuple = (0.10, -0.34, 0.020)
    workspace_hi: tuple = (0.52, 0.34, 0.30)
    ik_position_weight: float = 10.0
    ik_rotation_weight: float = 5.0  # hold the bowl orientation steady during motion (anti-spill)
    ik_joint_limit_weight: float = 10.0
    ik_lambda_initial: float = 0.1  # LM damping (multi-seed escapes local minima; lambda just stabilizes)
    ik_step_size: float = 0.5  # LM step
    ik_iterations: int = 12
    reset_ik_iterations: int = 60  # LM iters per seed at reset (x n_seeds; reset is occasional)
    # Multi-seed reset IK: the loaded "target_up"/"home_up" curriculum poses need branch-finding seeds
    # (a single warm-started seed rails the wrist over the +y target box). Runtime stays single-seed.
    reset_ik_seeds: int = 32
    # Joint-limit residual weight for the RESET solver only. Much stronger than the runtime
    # tracking weight: a reset pose railed on a joint limit puts every env on the limit
    # constraint's knife edge, and parallel-sim floating-point noise then diverges the
    # (identical) envs macroscopically within a few steps.
    reset_ik_joint_limit_weight: float = 200.0
    # Runtime control IK ONLY. The reset/initial pose always uses the Newton multi-seed solver
    # (branch-finding for the loaded curriculum poses); ``ik_backend`` selects how the per-step
    # bowl-pose command is tracked during the episode. Default "diffik": one damped-least-squares
    # Jacobian step per control step (the standard IsaacLab teleop/RL controller, ~3x cheaper than
    # the Newton LM solve and stateless-feeling). "newton" = full-pose LM (single warm-started seed),
    # kept available for the highest orientation-tracking fidelity.
    ik_backend: str = "diffik"
    diffik_lambda: float = 0.05
    diffik_lambda: float = 0.05
    diffik_max_delta: float = 0.05  # per-step joint delta clamp for DiffIK runtime tracking [rad]
    max_ik_delta: float = 0.05  # full-IK runtime joint delta clamp [rad]; 0 disables it.
    cartesian_action_scale: float = 0.30  # m/s per unit action
    pitch_action_scale: float = 3.0  # rad/s per unit action
    yaw_action_scale: float = 2.0  # rad/s per unit action (aims the pour direction about vertical)
    min_yaw: float = -1.57  # yaw range about world Z [rad]; modest so the wrist stays off its rails
    max_yaw: float = 1.57
    # Teleop: hold the commanded target through zero-action frames (the arm keeps converging to the
    # last command). RL keeps this False: a zero action snaps the target to the achieved pose and
    # skips the IK solve, so it is a true no-op.
    teleop_hold_target: bool = False
    min_pitch: float = -1.4  # tilt bowl opening toward -X [rad]
    max_pitch: float = 2.6  # max bowl tilt [rad] (allows near-inversion to pour)
    action_smoothing: float = 0.4

    # ---- heightfield obs (env frame) ----
    heightfield_lo: tuple = (0.10, -0.40, -0.02)
    heightfield_hi: tuple = (0.46, 0.40, 0.26)
    heightfield_size: int = 16
    # Kit-only debug: draw the live policy observations (heightfield grid colored by height + the source/held/
    # all-media centroids and the bowl target) as USD points that update every render. Off by default (no cost).
    debug_vis_obs: bool = False

    # ---- success / curriculum (mutated by ScoopCurriculum) ----
    success_particle_count: float = 500.0  # fill-reward normalization scale (a "good scoop" ~ this many)
    # Per-stage DELIVERY success threshold: the success bonus pays (and the per-episode success flag
    # latches, via ``delivery_success_mask``) while more than ``scoop_target_count`` particles sit in
    # the target box. ``scoop_target_count`` is the runtime value, set from
    # ``curriculum_target_count[stage]`` by ScoopCurriculum; early (pre-loaded) stages keep it
    # lenient, the full-scoop stages ramp it up.
    scoop_target_count: float = 10.0
    curriculum_target_count: tuple = (10, 10, 10, 25, 50)
    # Pitch STATE at the "pile" reset must match the fixed arm_home config's tilt (~2.06) so the action
    # term HOLDS the scoop-tilt instead of righting the cup.
    pile_start_pitch: float = 2.06
    curriculum_pile_xy_jitter: tuple = (0.0, 0.005, 0.01, 0.02, 0.03)
    # Staged DUMP-FIRST scoop curriculum (per stage). The early stages reset the cup OPENING-UP and
    # PRE-LOADED with a cupful so the policy experiences delivery success (tilt -> particles fall into the
    # target) within a few steps of episode start; later stages reset empty at the pile and train the full
    # scoop->carry->dump against the same delivery objective.
    #   reset_pose: "target_up" = pre-loaded cup opening-up directly ABOVE the target box (dump_hover_z):
    #                             one tilt away from success. Solved by the multi-seed reset IK -- the old
    #                             single-seed solve railed the wrist at this pose; multi-seed
    #                             (``reset_ik_seeds``) lets a non-railed branch win, and the env warns at
    #                             reset if the solved pose is still rail-adjacent.
    #               "home_up"   = pre-loaded cup opening-up at ``loaded_hover_xy`` (central, clear of the
    #                             source pile so the cupful never spawns inside media): carry +y, then dump.
    #               "pile"      = the scoop start (cup tilted at the pile, fixed ``arm_home``; cup empty).
    #   cup_fill_count: particles pre-loaded into the cup cavity at reset (0 = empty, must scoop).
    # Pre-load counts must fit the LOADABLE part of the cavity at the MPM particle spacing or the
    # overlap pressure ejects the media on the first solve ("pop"). For the hemisphere ladle the
    # sampled cone band holds ~160 non-overlapping particles at voxel 0.015 / 2 per cell -> stay under.
    curriculum_reset_pose: tuple = ("target_up", "home_up", "pile", "pile", "pile")
    curriculum_cup_fill_count: tuple = (120, 80, 0, 0, 0)
    # Per-stage reset cup pitch [rad]: stage 0 starts PRE-TILTED partway toward the dump ("just
    # before dumping"), so the policy only has to turn a little further to deliver and succeed
    # within a few steps. Must stay clearly below the spill tilt ~(pi/2 - angle of repose) ~ 0.96
    # or the pre-load pours out during the reset settle. Later loaded stages start opening-up.
    curriculum_reset_pitch: tuple = (0.6, 0.0, 0.0, 0.0, 0.0)
    dump_hover_z: float = 0.22  # env-frame z the opening-up cup hovers at over a container before dumping [m]
    # The loaded "home_up" reset hovers the pre-loaded cup HERE: a central spot clear of the source pile (so the
    # cup is not embedded in media -> no reset pop) and reachable without railing the wrist. Policy carries +y.
    loaded_hover_xy: tuple = (0.40, 0.0)
    # Loaded stages must require a REAL dump: the success threshold is at least this fraction of
    # the actually pre-loaded (capacity-clamped, tilt-derated) cup content, not the bare per-stage
    # count -- delivering a token ~10 particles of a 73-particle cupful is not success.
    curriculum_loaded_target_frac: float = 0.5
    # Advance only after the success-rate EMA shows the stage is genuinely mastered, and only
    # after enough resets to make the EMA meaningful. NOTE: at ~400 envs one reset batch already
    # exceeds a small reset minimum -- size it in BATCHES of resets, not single episodes.
    curriculum_success_threshold: float = 0.8
    curriculum_min_resets_per_stage: int = 1500
    curriculum_success_ema_alpha: float = 0.02
    # Initial stage (clamped to the stage count) and an advancement freeze -- set e.g.
    # ``env.curriculum_start_stage=2 env.curriculum_freeze=True`` on the CLI to train/play a
    # specific difficulty level.
    curriculum_start_stage: int = 0
    curriculum_freeze: bool = False

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 7.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        table_floor_z = -2.0 * float(self.table_half[2])
        self.scene.ground.init_state = AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, table_floor_z))
        self.viewer.eye = (1.4, 1.4, 0.9)
        self.viewer.lookat = (0.50, 0.00, 0.06)
        self.scene.robot.init_state.joint_pos.update(
            dict(
                zip(
                    (
                        "panda_joint1",
                        "panda_joint2",
                        "panda_joint3",
                        "panda_joint4",
                        "panda_joint5",
                        "panda_joint6",
                        "panda_joint7",
                    ),
                    self.arm_home,
                    strict=True,
                )
            )
        )
        self.scene.robot.init_state.joint_pos["panda_finger_joint.*"] = self.gripper_open_pos
        # The cup is gripped by construction; the Panda fingers are not action-controlled. The
        # "panda_hand" actuator group must stay POPPED and the USD-imported finger drive must stay
        # untouched: ANY change to the finger actuation (an IsaacLab actuator group at any gains,
        # zeroed drive gains, or added joint friction) makes the whole ARM diverge in the coupled
        # pipeline -- deterministic exponential joint-error growth against a railed corrective
        # actuator (A/B-verified pipeline bug). The fixed-open fingers are therefore state-pinned
        # per substep (env._pin_gripper_open_state); their shapes are non-colliding and the pin is
        # re-applied before rendering, so the old visible "gripper glitch" is gone.
        self.scene.robot.actuators.pop("panda_hand", None)
        for actuator_name in ("panda_shoulder", "panda_forearm"):
            self.scene.robot.actuators[actuator_name].stiffness = self.arm_stiffness
            self.scene.robot.actuators[actuator_name].damping = self.arm_damping
            self.scene.robot.actuators[actuator_name].armature = self.arm_armature

        self.sim.physics = CoupledNewtonCfg(
            solver_cfg=CoupledSolverCfg(
                entries=[
                    CoupledSolverEntryCfg(
                        name=RIGID_ENTRY,
                        # NOTE: integrator must stay "euler" -- "implicitfast" diverges in the coupled
                        # pipeline (arm drifts to the upright pose within ~4 steps despite correct ctrl
                        # targets). Actuator gains must therefore satisfy the explicit stability bounds
                        # sqrt(kp/m)*dt < 2 and kd*dt/m < 2 at the 1/120 s substep (see the gripper
                        # actuator override above).
                        solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True, njmax=510, nconmax=400),
                        # Franka collides with the table/ground static shapes and with rigid-only
                        # source/target bowl copies. Normal robot links remain out of MPM particle collision.
                        bodies=[
                            SceneEntityCfg("robot"),
                            r".*/Source(?:Bowl|Bucket)Rigid$",
                            r".*/Target(?:Bowl|Bucket)Rigid$",
                        ],
                        include_static_shapes=True,
                        substeps=self.num_substeps,
                    ),
                    CoupledSolverEntryCfg(
                        name=MPM_ENTRY,
                        solver_cfg=MPMSolverCfg(
                            voxel_size=self.voxel_size,
                            grid_type=self.grid_type,
                            # Sparse: padding must be 0 (per-voxel dilation) and the cell cap is
                            # ignored by Newton's sparse path; both are fixed-grid settings.
                            grid_padding=0 if self.grid_type == "sparse" else self.mpm_grid_padding,
                            max_active_cell_count=-1 if self.grid_type == "sparse" else self.mpm_max_active_cells,
                            strain_basis="P0",
                            transfer_scheme="apic",
                            max_iterations=self.mpm_iterations,
                            collider_velocity_mode="backward",
                            solver="gauss-seidel",
                            project_outside_colliders=True,
                        ),
                        all_particles=True,
                        bodies=[
                            r".*/ScoopBowl$",
                            r".*/Source(?:Bowl|Bucket)$",
                            r".*/Target(?:Bowl|Bucket)$",
                        ],
                        # MUST be False: the giant static ground plane as an MPM collider overflows the
                        # collider rasterization at high env counts (CUDA error 700). The pile-retaining
                        # boxes are kinematic bodies selected via full-label body regexes instead.
                        include_static_shapes=False,
                        include_child_joints=False,
                        in_place=True,
                    ),
                ],
                use_collision_pipeline=False,
            ),
            scene_cfg=self.scene,
            num_substeps=self.num_substeps,
            use_cuda_graph=self.use_cuda_graph and self.grid_type == "fixed",
        )


@configclass
class FrankaScoopEnvCfg_PLAY(FrankaScoopEnvCfg):
    def __post_init__(self):
        # Play/eval favors low per-step latency at few envs: fixed grid + CUDA graph (the graph
        # replay floor beats sparse's eager floor below the ~16-env crossover). Set BEFORE the
        # base __post_init__ bakes the solver entry.
        self.grid_type = "fixed"
        self.use_cuda_graph = True
        super().__post_init__()
        self.scene.num_envs = 4
        # Play/eval defaults to the REAL task: the final curriculum stage (empty cup at the pile,
        # full scoop->carry->dump, final delivery threshold), with stage advancement frozen.
        # Override ``env.curriculum_start_stage`` on the CLI to inspect another stage (e.g. =0 for
        # the pre-loaded dump-only start above the target box).
        self.curriculum_start_stage = len(self.curriculum_reset_pose) - 1
        self.curriculum_freeze = True


@configclass
class FrankaScoopEnvCfg_TELEOP(FrankaScoopEnvCfg_PLAY):
    """Lift-style teleop feel: low-lag, stateless-feeling DLS control.

    Mirrors the Franka cube-lift ``ik_rel`` teleop interface: the inherited ``diffik`` runtime
    backend (single-step damped-least-squares, the same controller the lift teleop uses), no
    action smoothing, a faster Cartesian rate, a looser per-step joint clamp, and the commanded
    target HELD through zero-action frames so the arm keeps converging to the last command
    instead of snapping to the achieved pose. The reset/initial pose still uses the Newton
    multi-seed solver (branch-finding for the loaded curriculum start).
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.action_smoothing = 0.0
        self.cartesian_action_scale = 0.6
        self.yaw_action_scale = 1.5
        self.diffik_max_delta = 0.08
        self.max_ik_delta = 0.08
        self.teleop_hold_target = True
        # No RL time-out mid-session: the operator resets explicitly (device button). Keep the
        # nonfinite_failure guard. Same pattern as the lift ik_rel teleop agent.
        self.terminations.time_out = None
        self.episode_length_s = 3600.0
