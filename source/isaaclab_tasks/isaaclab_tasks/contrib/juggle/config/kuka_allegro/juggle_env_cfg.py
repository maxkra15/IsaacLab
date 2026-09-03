# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton manager-based configuration for one-ball KUKA-Allegro juggling."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_newton.sim.schemas import NewtonCollisionPropertiesCfg, NewtonMaterialPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas import RigidBodyBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.contrib.juggle import mdp
from isaaclab_tasks.contrib.stack.mdp.kuka_allegro_reset import KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES
from isaaclab_tasks.utils.reset_sampling import (
    AdaptiveResetSamplerCfg,
    ContinuousAdaptiveResetSamplerCfg,
    RollingOutcomeMonitorCfg,
)

from isaaclab_assets.robots import KUKA_ALLEGRO_CFG

_TOOL_OFFSET = mdp.JUGGLE_SPHERE_CENTER_OFFSET
_FINGERTIP_BODY_NAMES = (
    "index_biotac_tip",
    "middle_biotac_tip",
    "ring_biotac_tip",
    "thumb_biotac_tip",
)
_NOMINAL_ARM_POSITION = (
    2.09426120,
    1.15379100,
    -2.19415376,
    1.48760476,
    -2.65542972,
    1.64765074,
    -2.88137803,
)


def _make_robot_cfg() -> ArticulationCfg:
    """Return the rotated, fully actuated KUKA-Allegro asset."""
    joint_positions = {
        **dict(zip(mdp.KUKA_ARM_JOINT_NAMES, _NOMINAL_ARM_POSITION, strict=True)),
        **dict(
            zip(
                KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES,
                mdp.JUGGLE_SPHERE_PRELOAD_HAND_POSITION,
                strict=True,
            )
        ),
    }
    cfg = KUKA_ALLEGRO_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            # The authored arm reaches negative X; this fixed-base rotation places the task at +X.
            rot=(0.0, 0.0, 1.0, 0.0),
            joint_pos=joint_positions,
        ),
    )
    cfg.spawn.rigid_props.disable_gravity = False
    hand_expression = "(index|middle|ring|thumb)_joint_(0|1|2|3)"
    cfg.actuators["kuka_allegro_actuators"].stiffness[hand_expression] = 20.0
    cfg.actuators["kuka_allegro_actuators"].damping[hand_expression] = 0.5
    return cfg


@configclass
class JuggleSceneCfg(InteractiveSceneCfg):
    """KUKA-Allegro, one procedural ball, a floor, and lighting."""

    robot: ArticulationCfg = _make_robot_cfg()
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.50, 0.0, 0.30)),
        spawn=sim_utils.SphereCfg(
            radius=mdp.BALL_RADIUS,
            rigid_props=RigidBodyBaseCfg(disable_gravity=False),
            collision_props=NewtonCollisionPropertiesCfg(contact_margin=0.0, contact_gap=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=mdp.BALL_MASS),
            physics_material=NewtonMaterialPropertiesCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
                torsional_friction=0.002,
                rolling_friction=0.0001,
                contact_stiffness=1.0e4,
                contact_damping=120.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.35, 0.05),
                roughness=0.55,
            ),
            semantic_tags=[("class", "ball")],
        ),
    )
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0),
    )


@configclass
class ActionsCfg:
    """Measured-state residual targets for all seven arm and sixteen hand joints."""

    arm_action = mdp.WorkspaceBoundedRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(mdp.KUKA_ARM_JOINT_NAMES),
        preserve_order=True,
        scale=0.60,
        max_delta=0.60,
        workspace_lower=mdp.KUKA_ALLEGRO_JUGGLE_ARM_WORKSPACE_LOWER,
        workspace_upper=mdp.KUKA_ALLEGRO_JUGGLE_ARM_WORKSPACE_UPPER,
        gravity_compensation=True,
    )
    hand_action = mdp.JuggleResetPreservingRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES),
        preserve_order=True,
        scale=0.10,
        max_delta=0.10,
        reset_preload_joint_names=KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES,
        reset_preload_commands_by_pair=(mdp.JUGGLE_SPHERE_PRELOAD_HAND_POSITION,),
        reset_open_commands_by_pair=(mdp.JUGGLE_SPHERE_OPEN_HAND_POSITION,),
        preload_release_threshold=0.01,
        preload_release_steps=1,
    )


@configclass
class ObservationsCfg:
    """Complete proprioception, ball motion, grasp geometry, and physical phase."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        ball_relative_position = ObsTerm(
            func=mdp.ball_position_relative_to_tool,
            params={
                "tool_body_cfg": SceneEntityCfg("robot", body_names=["palm_link"]),
                "tool_offset": _TOOL_OFFSET,
            },
        )
        ball_relative_velocity = ObsTerm(
            func=mdp.ball_velocity_relative_to_tool,
            params={
                "tool_body_cfg": SceneEntityCfg("robot", body_names=["palm_link"]),
                "tool_offset": _TOOL_OFFSET,
            },
        )
        ball_height_and_velocity = ObsTerm(func=mdp.ball_height_and_velocity)
        fingertips_relative_to_ball = ObsTerm(
            func=mdp.fingertips_relative_to_ball,
            params={
                "fingertip_cfg": SceneEntityCfg(
                    "robot",
                    body_names=list(_FINGERTIP_BODY_NAMES),
                    preserve_order=True,
                )
            },
        )
        fingertip_velocities_relative_to_ball = ObsTerm(
            func=mdp.fingertip_velocities_relative_to_ball,
            params={
                "fingertip_cfg": SceneEntityCfg(
                    "robot",
                    body_names=list(_FINGERTIP_BODY_NAMES),
                    preserve_order=True,
                )
            },
        )
        palm_twist = ObsTerm(
            func=mdp.palm_twist,
            params={
                "tool_body_cfg": SceneEntityCfg("robot", body_names=["palm_link"]),
                "tool_offset": _TOOL_OFFSET,
            },
        )
        ball_angular_velocity = ObsTerm(func=mdp.ball_angular_velocity)
        tool_axes = ObsTerm(
            func=mdp.tool_axes,
            params={
                "tool_body_cfg": SceneEntityCfg("robot", body_names=["palm_link"]),
                "tool_offset": _TOOL_OFFSET,
            },
        )
        hand_closure = ObsTerm(
            func=mdp.hand_closure,
            params={
                "hand_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES),
                    preserve_order=True,
                )
            },
        )
        phase = ObsTerm(func=mdp.phase_one_hot)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Sparse local/full-cycle impulses plus a tiny action-rate regularizer."""

    local_transition = RewTerm(func=mdp.local_transition_pulse, weight=1.0)
    full_cycle = RewTerm(func=mdp.full_cycle_pulse, weight=2.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-5)


@configclass
class TerminationsCfg:
    """Progress context, physical failure, full-cycle success, and neutral timeout."""

    progress_context = DoneTerm(
        func=mdp.JuggleProgressContext,
        params={
            "fingertip_cfg": SceneEntityCfg(
                "robot",
                body_names=list(_FINGERTIP_BODY_NAMES),
                preserve_order=True,
            ),
            "tool_body_cfg": SceneEntityCfg("robot", body_names=["palm_link"]),
            "tool_offset": _TOOL_OFFSET,
            "release_separation_distance": 0.03,
            "release_clear_steps": 2,
            "apex_height_gain": 0.06,
            "catch_approach_distance": 0.12,
            "stable_catch_steps": 15,
        },
    )
    ball_out_of_workspace = DoneTerm(func=mdp.ball_out_of_workspace)
    nonfinite_state = DoneTerm(func=mdp.nonfinite_state)
    local_goal_success = DoneTerm(func=mdp.noncanonical_local_goal_success)
    success = DoneTerm(func=mdp.cycle_success)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventCfg:
    """Apply one complete robot/ball state from the reset catalog."""

    reset_from_catalog = EventTerm(
        func=mdp.JuggleResetEvent,
        mode="reset",
        params={
            "rows_per_phase": 64,
            "fixed_phase": None,
            "static_held_only": False,
            "sampling_mode": "semantic",
            "continuous_seed": 0,
        },
    )


@configclass
class CurriculumCfg:
    """Configure uniform, semantic-item, or continuous success-model reset sampling."""

    reset_sampling = CurrTerm(
        func=mdp.JuggleResetCurriculum,
        params={
            # A full rank can reset thousands of environments together.  Retain several recent
            # batches per phase instead of keeping an ordering-biased tail of only 50 outcomes.
            "outcome_monitor": RollingOutcomeMonitorCfg(history_length=8192, prior_strength=2.0),
            # The sampler owns the non-canonical 65%; 15/65 makes exact coverage 15% globally.
            "adaptive_sampler": AdaptiveResetSamplerCfg(
                target_success_rate=0.50,
                kappa=1.0,
                temperature=1.0,
                coverage_fraction=0.15 / 0.65,
                epsilon=1.0e-4,
            ),
            # Used only by the third, continuously parameterized mode. The
            # coverage stream walks the complete Sobol proposal bank, while
            # Gaussian regression shares outcomes between nearby resets but
            # never across physical phase discontinuities.
            "continuous_sampler": ContinuousAdaptiveResetSamplerCfg(
                target_success_rate=0.50,
                kappa=1.0,
                temperature=1.0,
                coverage_fraction=0.15 / 0.65,
                epsilon=1.0e-4,
                history_length=4096,
                prior_strength=2.0,
                kernel_bandwidth=0.35,
                prediction_chunk_size=1024,
            ),
            "canonical_fraction": 0.35,
            "sampling_mode": "semantic",
        },
    )


@configclass
class _KukaAllegroJuggleBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Internal fully actuated foundation for the standard Juggle task."""

    decimation = 2
    episode_length_s = 3.0
    is_finite_horizon = False
    scene: JuggleSceneCfg = JuggleSceneCfg(
        num_envs=4096,
        env_spacing=2.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                njmax=300,
                nconmax=200,
                impratio=1.0,
                cone="pyramidal",
                update_data_interval=2,
                iterations=100,
                ls_iterations=15,
                use_mujoco_contacts=False,
                ccd_iterations=50,
            ),
            num_substeps=4,
            collision_decimation=1,
            use_cuda_graph=True,
            collision_cfg=NewtonCollisionPipelineCfg(
                broad_phase="explicit",
                reduce_contacts=True,
                rigid_contact_max=4_000_000,
            ),
            default_shape_cfg=NewtonShapeCfg(margin=0.0, gap=0.0),
        ),
        default_visualizer_cfg=VisualizerCfg(
            eye=(1.35, 1.35, 0.90),
            lookat=(0.50, 0.0, 0.32),
        ),
    )
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    commands = None

    def validate_config(self) -> None:
        """Validate the Newton timing, reset mixture, and 23-action contract."""
        if not isinstance(self.sim.physics, NewtonCfg) or not isinstance(self.sim.physics.solver_cfg, MJWarpSolverCfg):
            raise TypeError("KUKA-Allegro juggling requires the Newton MJWarp solver.")
        if self.sim.dt != 1.0 / 120.0 or self.decimation != 2 or self.sim.render_interval != self.decimation:
            raise ValueError("Juggling requires 120 Hz simulation and 60 Hz policy/render cadence.")
        if self.sim.physics.num_substeps != 4 or self.sim.physics.solver_cfg.ccd_iterations != 50:
            raise ValueError("Juggling requires four Newton substeps and 50 CCD iterations.")
        if not self.scene.replicate_physics:
            raise ValueError("Vectorized juggling requires replicated physics.")
        if len(self.actions.arm_action.joint_names) != 7 or len(self.actions.hand_action.joint_names) != 16:
            raise ValueError("Juggling actions must expose all seven arm and sixteen hand joints.")
        if not self.actions.arm_action.gravity_compensation:
            raise ValueError("The KUKA arm action must enable model-based gravity compensation.")
        if self.curriculum is not None:
            sampling_mode = self.curriculum.reset_sampling.params.get("sampling_mode", "semantic")
            reset_sampling_mode = self.events.reset_from_catalog.params.get("sampling_mode", "semantic")
            if sampling_mode not in ("uniform", "semantic", "continuous"):
                raise ValueError("Reset sampling mode must be 'uniform', 'semantic', or 'continuous'.")
            if reset_sampling_mode != sampling_mode:
                raise ValueError("The reset event and curriculum must use the same sampling mode.")
            canonical = float(self.curriculum.reset_sampling.params["canonical_fraction"])
            sampler_name = "continuous_sampler" if sampling_mode == "continuous" else "adaptive_sampler"
            frontier_coverage = float(self.curriculum.reset_sampling.params[sampler_name].coverage_fraction)
            if abs(canonical - 0.35) > 1.0e-9:
                raise ValueError("Reset sampling must retain 35% canonical held starts.")
            if (
                sampling_mode in ("semantic", "continuous")
                and abs((1.0 - canonical) * frontier_coverage - 0.15) > 1.0e-9
            ):
                raise ValueError("Adaptive reset sampling must retain 15% global uniform coverage.")

    def play_mode(self) -> None:
        """Evaluate from the canonical held start without adaptive reassignment."""
        super().play_mode()
        self.curriculum = None
        self.events.reset_from_catalog.params["fixed_phase"] = int(mdp.JugglePhase.HELD_PRETHROW)
        self.events.reset_from_catalog.params["static_held_only"] = True
