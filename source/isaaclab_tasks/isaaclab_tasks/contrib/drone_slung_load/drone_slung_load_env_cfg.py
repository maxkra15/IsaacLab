# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FLARE waypoint passing with a Newton AVBD cable-suspended payload."""

from __future__ import annotations

import math

from isaaclab_newton.physics import NewtonCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, CableObjectCfg, RigidObjectCfg
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

from isaaclab_contrib.deformable import VBDSolverCfg

import isaaclab_tasks.contrib.drone_slung_load.mdp as mdp
from isaaclab_tasks.contrib.drone_slung_load.system import (
    CABLE_BEND_DAMPING,
    CABLE_BEND_MODULUS,
    CABLE_DENSITY,
    CABLE_MASS,
    CABLE_NOMINAL_LENGTH,
    CABLE_NUM_POINTS,
    CABLE_STRETCH_MODULUS,
    CABLE_THICKNESS,
    CABLE_TWIST_MODULUS,
    DRONE_COLLIDER_SIZE,
    DRONE_DIAGONAL_INERTIA,
    DRONE_MASS,
    ENHANCED_RESIDUAL_BODY_RATE_LIMITS,
    GRAVITY,
    MAX_THRUST_TO_WEIGHT,
    PAYLOAD_MASS,
    PAYLOAD_RADIUS,
    ROTOR_ARM_LENGTH,
    ROTOR_HEIGHT,
    ROTOR_YAW_COEFFICIENT,
)

# FLARE's ``l`` is the quadrotor-state to point-payload-state distance. Both
# hard attachments therefore act at the corresponding centers of mass: this
# avoids adding unreported payload rotational dynamics and keeps ``l = 0.50 m``
# for every cable direction, including randomized initial swing.
# VBD reconstructs velocity from float32 pose differences at every substep. Keep
# the flight volume close to the numerical origin so near-hover accelerations do
# not fall below position resolution. Moving the ground instead of the vehicle
# preserves the physical 1.5 m clearance and every relative task quantity.
GROUND_HEIGHT = -1.5
HOVER_HEIGHT = 0.0
ATTACH_OFFSET_Z = 0.0
DRONE_CRASH_HEIGHT = GROUND_HEIGHT + 0.18
PAYLOAD_CRASH_HEIGHT = GROUND_HEIGHT + PAYLOAD_RADIUS + 0.005
MAX_CABLE_RELATIVE_SEPARATION = 0.05
MAX_CABLE_JOINT_ERROR = 0.01
# FLARE defines an invalid workspace and an unsafe-angle threshold but does not
# publish their numerical bounds. These baseline-task bounds are therefore
# implementation choices, not values attributed to the paper.
WORKSPACE_X_BOUND = 6.0
WORKSPACE_Y_BOUND = 3.0
WORKSPACE_Z_MAX = GROUND_HEIGHT + 4.0
SWING_SAFETY_ANGLE = 1.0
# The later released scenario-one configuration bounds active-target-relative
# XYZ error at 3 m, 3 m, and 2 m, respectively.
ENHANCED_TARGET_ERROR_X_BOUND = 3.0
ENHANCED_TARGET_ERROR_Y_BOUND = 3.0
ENHANCED_TARGET_ERROR_Z_BOUND = 2.0


def _figure_eight_waypoint_offsets(
    *, laps: int = 3, samples_per_lap: int = 16, longitudinal_radius: float = 2.0, lateral_radius: float = 1.0
) -> tuple[tuple[float, float, float], ...]:
    """Return a reset-relative horizontal Gerono lemniscate [m].

    FLARE reports an ``Eight - 3 laps`` benchmark but does not publish its
    waypoint coordinates. This deterministic route is therefore an explicit
    Isaac Lab benchmark choice, with the reset pose at the left tip.
    """
    offsets = []
    for sample in range(1, laps * samples_per_lap + 1):
        phase = -0.5 * math.pi + math.tau * sample / samples_per_lap
        x = longitudinal_radius * (1.0 + math.sin(phase))
        y = lateral_radius * math.sin(2.0 * phase)
        offsets.append((x, y, 0.0))
    return tuple(offsets)


FIGURE_EIGHT_EVAL_WAYPOINT_OFFSETS = _figure_eight_waypoint_offsets()
"""Three-lap figure-eight evaluation route, sampled at sixteen waypoints per lap [m]."""

PAPER_BASELINE_TRAIN_WAYPOINT_OFFSETS = (
    (1.0, 0.0, 0.0),
    (2.0, 0.75, 0.0),
    (3.0, -0.75, 0.0),
    (4.0, 0.0, 0.0),
)
"""Fixed four-point route retained for the published-MDP baseline."""


def _cable_positions(length: float, num_points: int) -> list[tuple[float, float, float]]:
    """Return uniformly spaced cable control points along local -Z [m]."""
    segment_length = length / (num_points - 1)
    return [(0.0, 0.0, -index * segment_length) for index in range(num_points)]


_CABLE_SPAWN_Z = HOVER_HEIGHT + ATTACH_OFFSET_Z
_PAYLOAD_SPAWN_Z = _CABLE_SPAWN_Z - CABLE_NOMINAL_LENGTH


def _colored_sphere(**kwargs) -> sim_utils.SphereCfg:
    color = kwargs.pop("display_color")
    cfg = sim_utils.SphereCfg(**kwargs)
    cfg.func = mdp.spawn_sphere_with_color
    cfg.display_color = color
    return cfg


def _colored_cable(**kwargs) -> sim_utils.CableCfg:
    color = kwargs.pop("display_color")
    cfg = sim_utils.CableCfg(**kwargs)
    cfg.func = mdp.spawn_cable_with_color
    cfg.display_color = color
    return cfg


@configclass
class DroneSlungLoadSceneCfg(InteractiveSceneCfg):
    """One rigid drone, one rigid payload, and an AVBD string with hard end attachments."""

    robot = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=mdp.DroneCuboidCfg(
            size=DRONE_COLLIDER_SIZE,
            arm_length=ROTOR_ARM_LENGTH,
            rotor_z=ROTOR_HEIGHT,
            diagonal_inertia=DRONE_DIAGONAL_INERTIA,
            rigid_props=sim_utils.RigidBodyBaseCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=DRONE_MASS),
            collision_props=sim_utils.CollisionBaseCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.12, 0.16)),
            display_color=(0.08, 0.12, 0.16),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, HOVER_HEIGHT)),
    )
    payload = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Payload",
        spawn=_colored_sphere(
            radius=PAYLOAD_RADIUS,
            rigid_props=sim_utils.RigidBodyBaseCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=PAYLOAD_MASS),
            collision_props=sim_utils.CollisionBaseCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.50, 0.08)),
            display_color=(0.95, 0.50, 0.08),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, _PAYLOAD_SPAWN_Z)),
    )
    cable = CableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cable",
        spawn=_colored_cable(
            positions=_cable_positions(CABLE_NOMINAL_LENGTH, CABLE_NUM_POINTS),
            physics_material=sim_utils.CableMaterialCfg(
                thickness=CABLE_THICKNESS,
                density=CABLE_DENSITY,
                stretch_stiffness=CABLE_STRETCH_MODULUS,
                bend_stiffness=CABLE_BEND_MODULUS,
                # Unauthored shear falls back to the stretch modulus in Newton.
                shear_stiffness=None,
                twist_stiffness=CABLE_TWIST_MODULUS,
            ),
            # Cable contact is intentionally disabled for this open-space waypoint
            # task. The rigid payload and drone retain ground contact.
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False)],
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.12, 0.14)),
            display_color=(0.24, 0.24, 0.27),
        ),
        init_state=CableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, _CABLE_SPAWN_Z)),
    )
    # These are hard position-only ball attachments outside the cable articulation.
    drone_cable_attach = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DroneCableAttach",
        spawn=mdp.PhysicsAttachmentCfg(
            src0="../Cable/geometry/mesh",
            src1="../Robot",
            indices0=(0,),
            coords1=((0.0, 0.0, ATTACH_OFFSET_Z),),
        ),
    )
    cable_payload_attach = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CablePayloadAttach",
        spawn=mdp.PhysicsAttachmentCfg(
            src0="../Cable/geometry/mesh",
            src1="../Payload",
            indices0=(CABLE_NUM_POINTS - 1,),
            coords1=((0.0, 0.0, 0.0),),
        ),
    )
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(color=(0.18, 0.18, 0.20)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, GROUND_HEIGHT)),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.90, 0.90, 0.95)),
    )


@configclass
class CommandsCfg:
    """Paper-style sequence exposing the active and following waypoint."""

    route = mdp.WaypointSequenceCommandCfg(
        asset_name="robot",
        waypoint_offsets=PAPER_BASELINE_TRAIN_WAYPOINT_OFFSETS,
        acceptance_radius=0.5,
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    """FLARE collective-thrust and body-rate action applied at four rotor sites."""

    # FLARE publishes the action limits but not the inner SI rate gains, torque
    # limits, or motor time constants. The later scenario-one release provides
    # the rotor geometry and thrust/moment coefficients used here.
    thrust = mdp.CollectiveThrustBodyRateActionCfg(
        asset_name="robot",
        max_thrust_to_weight=MAX_THRUST_TO_WEIGHT,
        max_body_rates=(15.0, 15.0, 5.0),
        rate_gains=(0.016, 0.016, 0.028),
        torque_limits=(0.20, 0.20, 0.08),
        arm_length=ROTOR_ARM_LENGTH,
        rotor_z=ROTOR_HEIGHT,
        yaw_coeff=ROTOR_YAW_COEFFICIENT,
        rotor_thrust_limits=(0.0, MAX_THRUST_TO_WEIGHT * DRONE_MASS * GRAVITY / 4.0),
        tau_up=0.03,
        tau_down=0.03,
        dt=0.01,
    )


@configclass
class PolicyCfg(ObsGroup):
    """FLARE actor observation: 14 general + 6 waypoint + 4 previous-action values."""

    drone_velocity = ObsTerm(func=mdp.world_lin_vel_normalized)
    body_rotation = ObsTerm(func=mdp.body_rotation_matrix)
    swing_angles = ObsTerm(func=mdp.swing_angles_normalized)
    waypoint_offsets = ObsTerm(func=mdp.waypoint_offsets_normalized, params={"command_name": "route"})
    previous_action = ObsTerm(func=mdp.previous_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class PrivilegedCfg(ObsGroup):
    """AVBD state available only to the asymmetric training critic."""

    body_angular_velocity = ObsTerm(func=mdp.body_ang_vel_normalized)
    payload_attachment = ObsTerm(func=mdp.payload_attachment_b_normalized)
    total_swing = ObsTerm(func=mdp.total_swing_angle_normalized)
    transverse_velocity = ObsTerm(func=mdp.payload_transverse_velocity_b)
    upper_cable_tangent = ObsTerm(
        func=mdp.upper_cable_tangent_b,
        params={"nominal_length": CABLE_NOMINAL_LENGTH},
    )
    cable_separation = ObsTerm(
        func=mdp.cable_relative_separation,
        params={"nominal_length": CABLE_NOMINAL_LENGTH},
    )
    payload_velocity = ObsTerm(func=mdp.payload_lin_vel_w_normalized)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    """Paper-aligned actor observations and AVBD-aware privileged critic observations."""

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Reset at a fixed hover pose and randomize only the minor initial cable swing."""

    reset_base = EventTerm(
        func=mdp.reset_drone_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (HOVER_HEIGHT, HOVER_HEIGHT),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_slung_load = EventTerm(
        func=mdp.ResetSlungLoadEvent,
        mode="reset",
        params={
            "cable_length": CABLE_NOMINAL_LENGTH,
            "robot_cfg": SceneEntityCfg("robot"),
            "cable_cfg": SceneEntityCfg("cable"),
            "payload_cfg": SceneEntityCfg("payload"),
            "attach_offset_z": ATTACH_OFFSET_Z,
            "max_initial_swing": 0.10,
        },
    )


@configclass
class RewardsCfg:
    """FLARE waypoint reward from Eqs. (7)--(11) and Table I."""

    # This zero-valued side-effect term captures the terminal sample before autoreset.
    episode_metrics = RewTerm(func=mdp.record_episode_metrics, weight=1.0)
    progress = RewTerm(
        func=mdp.waypoint_progress,
        weight=10.0,
        params={"command_name": "route", "asset_cfg": SceneEntityCfg("robot")},
    )
    action_smoothness = RewTerm(func=mdp.action_delta_l2, weight=-1.0e-4)
    swing_safety = RewTerm(
        func=mdp.swing_safety_impulse,
        weight=-3.0,
        params={"threshold": SWING_SAFETY_ANGLE},
    )
    crash = RewTerm(func=mdp.crash_impulse, weight=-10.0)


@configclass
class TerminationsCfg:
    """Timeout, ground-contact proxy, finite-state, and workspace guards."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    drone_crash = DoneTerm(
        func=mdp.link_height_below_minimum,
        params={"minimum_height": DRONE_CRASH_HEIGHT, "asset_cfg": SceneEntityCfg("robot")},
    )
    payload_crash = DoneTerm(
        func=mdp.link_height_below_minimum,
        params={"minimum_height": PAYLOAD_CRASH_HEIGHT, "asset_cfg": SceneEntityCfg("payload")},
    )
    illegal_drone = DoneTerm(func=mdp.illegal_link_state, params={"asset_cfg": SceneEntityCfg("robot")})
    illegal_payload = DoneTerm(func=mdp.illegal_link_state, params={"asset_cfg": SceneEntityCfg("payload")})
    illegal_cable = DoneTerm(func=mdp.illegal_cable_state, params={"cable_cfg": SceneEntityCfg("cable")})
    illegal_action = DoneTerm(func=mdp.illegal_action)
    cable_integrity = DoneTerm(
        func=mdp.cable_integrity_violation,
        params={
            "nominal_length": CABLE_NOMINAL_LENGTH,
            "max_relative_separation": MAX_CABLE_RELATIVE_SEPARATION,
            "max_joint_error": MAX_CABLE_JOINT_ERROR,
            "cable_cfg": SceneEntityCfg("cable"),
            "robot_cfg": SceneEntityCfg("robot"),
            "payload_cfg": SceneEntityCfg("payload"),
        },
    )
    drone_out_of_workspace = DoneTerm(
        func=mdp.out_of_workspace,
        params={
            "x_bound": WORKSPACE_X_BOUND,
            "y_bound": WORKSPACE_Y_BOUND,
            "z_max": WORKSPACE_Z_MAX,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    payload_out_of_workspace = DoneTerm(
        func=mdp.out_of_workspace,
        params={
            "x_bound": WORKSPACE_X_BOUND,
            "y_bound": WORKSPACE_Y_BOUND,
            "z_max": WORKSPACE_Z_MAX,
            "asset_cfg": SceneEntityCfg("payload"),
        },
    )


@configclass
class DroneSlungLoadWaypointEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based FLARE waypoint task using the Newton AVBD backend."""

    # Newton isolates replicated environments by ``body_world``. Co-locating
    # their numerical origins avoids float32 precision loss from a display-only
    # grid translation; interactive play uses one world below.
    scene: DroneSlungLoadSceneCfg = DroneSlungLoadSceneCfg(num_envs=32, env_spacing=0.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def _newton_physics_cfg(self) -> NewtonCfg:
        """Build the coupled rigid/cable solver used by slung-load variants."""
        return NewtonCfg(
            solver_cfg=VBDSolverCfg(iterations=32),
            num_substeps=8,
            use_cuda_graph=True,
        )

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = 12.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        # The eight-by-thirty-two solve keeps the thin eight-segment string
        # finite and limits loaded-hover drift without inflating cable stiffness.
        self.sim.physics = self._newton_physics_cfg()
        self.sim.default_visualizer_cfg = NewtonGLVisualizerCfg(
            eye=(6.5, -6.5, 1.9),
            lookat=(2.0, 0.0, -0.2),
            enable_markers=True,
        )

    def play_mode(self):
        super().play_mode()
        # Training worlds intentionally overlap numerically for VBD precision;
        # render one world so the detailed drone and cable remain unambiguous.
        self.scene.num_envs = 1
        self.episode_length_s = 30.0
        if self.events.reset_slung_load is not None:
            self.events.reset_slung_load.params["max_initial_swing"] = 0.0
        self.evaluation_mode()
        self.commands.route.debug_vis = True

    def evaluation_mode(self):
        """Select the deterministic three-lap figure-eight benchmark route."""
        self.commands.route.waypoint_offsets = FIGURE_EIGHT_EVAL_WAYPOINT_OFFSETS


@configclass
class EnhancedCommandsCfg:
    """Bounded randomized ellipse and figure-eight routes for precise tracking."""

    route = mdp.WaypointSequenceCommandCfg(
        asset_name="robot",
        waypoint_offsets=PAPER_BASELINE_TRAIN_WAYPOINT_OFFSETS,
        randomize_waypoints=True,
        regenerate_on_completion=False,
        # One randomized 24-sample lap is roughly 27--30 m. It is completable
        # within 15 s at the curvature-aware reference speed, so training and
        # evaluation optimize the same finite-route success objective.
        random_waypoint_count=24,
        route_family="bounded_template_mix",
        samples_per_lap=24,
        aspect_ratio_range=(0.94, 1.0),
        vertical_amplitude_range=(0.0, 0.15),
        figure_eight_probability=0.5,
        random_waypoint_ranges=mdp.WaypointSequenceCommandCfg.Ranges(
            pos_x=(-4.0, 4.0),
            pos_y=(-4.0, 4.0),
            pos_z=(-0.4, 0.4),
        ),
        minimum_waypoint_separation=0.75,
        maximum_waypoint_separation=1.5,
        nominal_heading_change=math.radians(40.0),
        maximum_heading_change=math.radians(60.0),
        maximum_vertical_step=0.15,
        random_sampling_attempts=8,
        route_sampling_attempts=4,
        acceptance_radius=0.5,
        spline_enabled=True,
        spline_tangent_scale=1.0,
        spline_projection_samples=12,
        spline_progressive_advancement=True,
        spline_plane_crossing_lateral_tolerance=0.30,
        spline_max_waypoint_advances_per_step=4,
        target_cruise_speed=1.25,
        maximum_lateral_acceleration=3.0,
        maximum_braking_acceleration=4.0,
        speed_lookahead_distances=(0.0, 0.75, 1.50),
        debug_vis=False,
    )


@configclass
class EnhancedPolicyCfg(ObsGroup):
    """Later-release FLARE state augmented with cable rate and spline preview."""

    drone_velocity = ObsTerm(func=mdp.world_lin_vel_normalized)
    body_rotation = ObsTerm(func=mdp.body_rotation_matrix)
    swing_angles = ObsTerm(func=mdp.swing_angles_normalized)
    swing_angular_velocity = ObsTerm(func=mdp.swing_angular_velocity_normalized)
    waypoint_offsets = ObsTerm(func=mdp.waypoint_offsets_normalized, params={"command_name": "route"})
    path_tracking = ObsTerm(
        func=mdp.path_tracking_features_b,
        params={
            "command_name": "route",
            "lookahead_distances": (0.75, 1.50),
            "cross_track_scale": 0.20,
            "preview_scale": (2.0, 2.0, 1.0),
            "curvature_scale": 1.0,
        },
    )
    path_speed = ObsTerm(
        func=mdp.path_speed_features,
        params={"command_name": "route", "speed_scale": 3.5, "asset_cfg": SceneEntityCfg("robot")},
    )
    previous_action = ObsTerm(func=mdp.previous_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class EnhancedObservationsCfg(ObservationsCfg):
    """Forty-four actor values plus the unchanged 17-value AVBD critic state."""

    policy: EnhancedPolicyCfg = EnhancedPolicyCfg()


@configclass
class DirectCTBRPolicyCfg(ObsGroup):
    """Policy-owned CTBR state with measured body rates for the inner-loop interface."""

    drone_velocity = ObsTerm(func=mdp.body_lin_vel_normalized, params={"speed_scale": 4.5})
    body_rotation = ObsTerm(func=mdp.body_rotation_matrix)
    body_angular_velocity = ObsTerm(func=mdp.body_ang_vel_normalized)
    swing_angles = ObsTerm(func=mdp.swing_angles_normalized)
    swing_angular_velocity = ObsTerm(func=mdp.swing_angular_velocity_normalized)
    waypoint_offsets = ObsTerm(func=mdp.waypoint_offsets_normalized, params={"command_name": "route"})
    path_tracking = ObsTerm(
        func=mdp.path_tracking_features_b,
        params={
            "command_name": "route",
            "lookahead_distances": (0.75, 1.50),
            "cross_track_scale": 1.00,
            "preview_scale": (2.0, 2.0, 1.0),
            "curvature_scale": 1.0,
        },
    )
    path_speed = ObsTerm(
        func=mdp.path_speed_features,
        params={"command_name": "route", "speed_scale": 4.5, "asset_cfg": SceneEntityCfg("robot")},
    )
    previous_action = ObsTerm(func=mdp.previous_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class DirectCTBRPrivilegedCfg(PrivilegedCfg):
    """Critic-only AVBD state without the body-rate signal already seen by the actor."""

    body_angular_velocity = None


@configclass
class DirectCTBRObservationsCfg(ObservationsCfg):
    """Forty-seven actor values plus fourteen additional critic values."""

    policy: DirectCTBRPolicyCfg = DirectCTBRPolicyCfg()
    privileged: DirectCTBRPrivilegedCfg = DirectCTBRPrivilegedCfg()


@configclass
class EnhancedRewardsCfg(RewardsCfg):
    """FLARE rewards plus continuous route and suspended-system stability costs.

    These costs are explicit Isaac Lab extensions: the paper and later release
    do not define a continuous cross-track or anti-swing objective. Each term is
    dimensionless after normalization and is integrated by RewardManager at the
    100 Hz control cadence.
    """

    # Replace the inherited point-potential term with indexed spline arc length.
    # The curriculum raises the rewarded rate from 1.25 to 3.50 m/s. A local
    # 3.0 m/s^2 limit preserves high speed on gentle arcs while removing the
    # incentive to attack tight figure-eight crossings too fast.
    progress = None
    path_progress = RewTerm(
        func=mdp.path_arc_length_progress,
        weight=3.0,
        params={"command_name": "route", "maximum_rate": 1.25, "maximum_lateral_acceleration": 3.0},
    )
    path_speed = RewTerm(
        func=mdp.path_tangent_speed_tracking_l2,
        weight=-1.0,
        params={
            "command_name": "route",
            "speed_error_scale": 1.0,
            "underspeed_weight": 1.0,
            "overspeed_weight": 2.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    completion = RewTerm(
        func=mdp.RouteCompletionImpulse,
        weight=50.0,
        params={"command_name": "route", "reference_completion_time": 15.0, "early_completion_scale": 1.0},
    )
    crash = RewTerm(
        func=mdp.unsafe_termination_impulse,
        weight=-100.0,
        params={
            "unsafe_term_names": (
                "drone_crash",
                "payload_crash",
                "illegal_drone",
                "illegal_payload",
                "illegal_cable",
                "illegal_action",
                "cable_integrity",
                "drone_out_of_workspace",
                "payload_out_of_workspace",
                "path_corridor",
            )
        },
    )
    path_precision = RewTerm(
        func=mdp.path_tracking_precision_log1p,
        weight=-2.0,
        params={
            "command_name": "route",
            "cross_track_scale": 0.50,
            "transverse_velocity_scale": 1.00,
            "transverse_speed_weight": 0.25,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    swing_magnitude = RewTerm(
        func=mdp.total_swing_angle_l2,
        weight=-0.25,
        params={"scale": 0.35},
    )
    transverse_speed = RewTerm(
        func=mdp.payload_transverse_speed_l2,
        weight=-0.05,
        params={"scale": 1.0},
    )
    body_rate = RewTerm(
        func=mdp.body_angular_velocity_l2,
        weight=-0.005,
        params={"scale": math.pi},
    )
    body_tilt = RewTerm(
        func=mdp.body_tilt_exp,
        weight=-0.1,
        params={"scale": 0.35},
    )
    action_acceleration = RewTerm(
        func=mdp.NormalizedActionAccelerationL2,
        weight=-0.005,
        params={"action_name": "thrust"},
    )


@configclass
class DirectCTBRRewardsCfg(EnhancedRewardsCfg):
    """Route-first objective shared by drone-only and slung-load Direct CTBR.

    Gated continuous progress and path-velocity tracking teach traversal and
    recovery without a separately exploitable waypoint impulse. There is no
    per-step speed target or early-completion multiplier: speed is introduced
    only by the curriculum after the policy has seen every route-acceptance
    stage. Bounded, low-weight stabilization costs cannot make an early
    corridor failure more attractive than continuing a difficult rollout.
    """

    path_progress = RewTerm(
        func=mdp.path_arc_length_progress,
        weight=1.0,
        params={
            "command_name": "route",
            "maximum_rate": 1.00,
            "maximum_lateral_acceleration": 6.0,
            "positive_progress_gate_distance": 0.75,
        },
    )
    action_smoothness = None
    swing_safety = None
    waypoint_advance = None
    path_speed = None
    path_velocity = RewTerm(
        func=mdp.path_velocity_tracking_l2,
        weight=-0.20,
        params={
            "command_name": "route",
            "cross_track_gain": 1.5,
            "maximum_cross_track_speed": 0.75,
            "velocity_error_scale": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    completion = RewTerm(
        func=mdp.RouteCompletionImpulse,
        weight=25.0,
        params={"command_name": "route", "early_completion_scale": 0.0},
    )
    path_precision = RewTerm(
        func=mdp.path_tracking_precision_exp,
        weight=-0.10,
        params={
            "command_name": "route",
            "cross_track_scale": 1.00,
            "transverse_velocity_scale": 1.50,
            "transverse_speed_weight": 0.25,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    swing_magnitude = RewTerm(func=mdp.total_swing_angle_l2, weight=-0.05, params={"scale": 0.35})
    transverse_speed = RewTerm(func=mdp.payload_transverse_speed_l2, weight=-0.01, params={"scale": 1.0})
    body_rate = RewTerm(
        func=mdp.body_angular_velocity_l2,
        weight=-0.001,
        params={"scale": math.pi},
    )
    body_tilt = RewTerm(func=mdp.body_tilt_exp, weight=-0.02, params={"scale": 0.35})
    action_acceleration = RewTerm(
        func=mdp.NormalizedActionAccelerationL2,
        weight=-0.001,
        params={"action_name": "thrust"},
    )


@configclass
class EnhancedTerminationsCfg(TerminationsCfg):
    """Baseline safety, route success, and a curriculum-tightened corridor."""

    path_corridor = DoneTerm(
        func=mdp.path_corridor_violation,
        params={"command_name": "route", "maximum_distance": 1.50},
    )
    route_completed = DoneTerm(func=mdp.route_completed, params={"command_name": "route"})


@configclass
class EnhancedCurriculumCfg:
    """Tighten tracking precision while exposing progressively higher speed."""

    precision_speed = CurrTerm(func=mdp.PrecisionSpeedCurriculumV13, update_mode="step")


@configclass
class DirectCTBRCurriculumCfg:
    """Stage route acceptance and geometry before Direct-CTBR speed."""

    precision_speed = CurrTerm(
        func=mdp.DirectCTBRRouteCurriculum,
        update_mode="step",
    )


@configclass
class DroneSlungLoadWaypointEnhancedEnvCfg(DroneSlungLoadWaypointEnvCfg):
    """Stable all-heading extension of the published FLARE waypoint MDP."""

    observations: EnhancedObservationsCfg = EnhancedObservationsCfg()
    commands: EnhancedCommandsCfg = EnhancedCommandsCfg()
    rewards: EnhancedRewardsCfg = EnhancedRewardsCfg()
    terminations: EnhancedTerminationsCfg = EnhancedTerminationsCfg()
    curriculum: EnhancedCurriculumCfg | None = EnhancedCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # A finite randomized ellipse or figure-eight lap uses the same route
        # distribution in training and evaluation and terminates on success.
        self.episode_length_s = 15.0
        # Passive per-joint bending damping matches the later released chain at
        # this task's coarser segment length. It damps cable flex without adding
        # world drag or an external wrench to the payload.
        if self.scene.cable is not None:
            self.scene.cable.spawn.physics_material.bend_damping = CABLE_BEND_DAMPING
        # The enhanced policy commands residual body rates around a conventional
        # yaw-invariant attitude-hold inner loop. This gives zero action a stable
        # physical meaning while the paper-aligned baseline remains pure rate
        # control with the configuration defaults of zero. The geometric
        # velocity loop requests a bounded thrust-axis tilt from -K_v v_xy.
        # At the 4.3--4.8 m ellipse radius, float32 VBD needs about 0.2 m/s^2
        # of radial acceleration to escape its pose-difference deadband. The
        # calibrated K_v=4/s, K_R=12/s pair arrests 0.05 m/s radial drift
        # without oscillation. The residual retains 10 rad/s roll/pitch
        # authority while reserving headroom inside the published 15 rad/s
        # final envelope for path tracking and curvature feedforward. The
        # vertical loop applies a calibrated K_vz=2/s total supported-mass
        # force correction and preserves vertical force through bounded tilt.
        self.actions.thrust.residual_body_rate_limits = ENHANCED_RESIDUAL_BODY_RATE_LIMITS
        self.actions.thrust.attitude_hold_gain = 12.0
        self.actions.thrust.horizontal_velocity_damping_gain = 4.0
        self.actions.thrust.vertical_velocity_damping_gain = 2.0
        self.actions.thrust.path_velocity_command_name = "route"
        self.actions.thrust.path_velocity_cross_track_gain = 1.5
        self.actions.thrust.path_velocity_maximum_cross_track_speed = 0.75
        self.actions.thrust.path_velocity_curvature_feedforward_gain = 1.0
        self.actions.thrust.suspended_mass = PAYLOAD_MASS + CABLE_MASS
        self.actions.thrust.tilt_compensation = True
        self.actions.thrust.maximum_velocity_hold_tilt = 0.42
        self.actions.thrust.maximum_tilt_compensation_angle = 0.5
        # Reset on the bounded route's outer tip. Event terms run in declaration order,
        # so ResetSlungLoadEvent subsequently hangs a strain-free cable from the
        # sampled drone pose and the command term reads that same post-reset
        # anchor when it samples the route.
        self.events.reset_base.func = mdp.reset_drone_state_on_annulus
        self.events.reset_base.params = {
            "radius_range": (4.3, 4.8),
            "height": HOVER_HEIGHT,
            "roll_range": (-0.05, 0.05),
            "pitch_range": (-0.05, 0.05),
            "yaw": 0.0,
            "asset_cfg": SceneEntityCfg("robot"),
        }
        # The later FLARE release bounds per-axis error from the active target.
        # Apply the same translation-invariant guard to both bodies named by the
        # paper while retaining the baseline task's fixed global workspace.
        for term, asset_name in (
            (self.terminations.drone_out_of_workspace, "robot"),
            (self.terminations.payload_out_of_workspace, "payload"),
        ):
            if term is None:
                continue
            term.func = mdp.active_waypoint_error_out_of_bounds
            term.params = {
                "x_bound": ENHANCED_TARGET_ERROR_X_BOUND,
                "y_bound": ENHANCED_TARGET_ERROR_Y_BOUND,
                "z_bound": ENHANCED_TARGET_ERROR_Z_BOUND,
                "command_name": "route",
                "asset_cfg": SceneEntityCfg(asset_name),
            }

    def evaluation_mode(self):
        """Score one finite randomized ellipse or figure-eight at final precision."""
        # Evaluation always uses the final precision/speed specification rather
        # than depending on the training process's current curriculum phase.
        self.curriculum = None
        self.commands.route.randomize_waypoints = True
        self.commands.route.regenerate_on_completion = False
        self.commands.route.random_waypoint_count = 24
        self.commands.route.acceptance_radius = 0.15
        self.commands.route.target_cruise_speed = 3.50
        self.rewards.path_progress.params["maximum_rate"] = 3.50
        self.rewards.path_precision.weight = -4.0
        self.rewards.path_precision.params["cross_track_scale"] = 0.20
        self.rewards.path_precision.params["transverse_velocity_scale"] = 0.40
        self.terminations.path_corridor.params["maximum_distance"] = 0.75
        self.episode_length_s = 15.0


@configclass
class DroneSlungLoadWaypointDirectCTBREnvCfg(DroneSlungLoadWaypointEnhancedEnvCfg):
    """Policy-owned collective-thrust/body-rate control on the v13 route geometry.

    The actor emits the complete normalized FLARE command. Only the conventional
    body-rate PID, saturation-aware rotor mixer, and motor lag remain below it;
    route geometry is consumed by observations and rewards, never by actuation.
    """

    observations: DirectCTBRObservationsCfg = DirectCTBRObservationsCfg()
    rewards: DirectCTBRRewardsCfg = DirectCTBRRewardsCfg()
    curriculum: DirectCTBRCurriculumCfg | None = DirectCTBRCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # Sharp turns share the same 24-waypoint geometry as the rigid task but
        # need a longer physical horizon and a cable-compatible acceleration
        # envelope. A 3 m/s^2 lateral limit corresponds to about 17 degrees of
        # quasi-static cable deflection; the extra preview point gives the
        # policy enough distance to brake before a randomized corner.
        self.episode_length_s = 20.0
        self.commands.route.maximum_lateral_acceleration = 3.0
        self.commands.route.maximum_braking_acceleration = 4.0
        self.commands.route.speed_lookahead_distances = (0.0, 0.75, 1.50, 2.25)
        self.rewards.path_progress.params["maximum_lateral_acceleration"] = 3.0
        action = self.actions.thrust
        action.residual_body_rate_limits = None
        action.attitude_hold_gain = 0.0
        action.horizontal_velocity_damping_gain = 0.0
        action.vertical_velocity_damping_gain = 0.0
        action.path_velocity_command_name = None
        action.path_velocity_cross_track_gain = 0.0
        action.path_velocity_curvature_feedforward_gain = 0.0
        action.suspended_mass = 0.0
        action.tilt_compensation = False
        # Conventional inner-loop CTBR PID. Gains are expressed in this
        # simulator's SI torque units; they intentionally are not copied from a
        # firmware's dimensionless tuning scale.
        action.rate_integral_gains = (0.040, 0.040, 0.070)
        action.rate_derivative_gains = (8.0e-5, 8.0e-5, 1.4e-4)
        action.rate_integral_error_limits = (0.50, 0.50, 0.11)
        action.rate_derivative_cutoff_hz = 20.0
        action.allocation_mode = "rate_priority"

        # Keep the reset domain fixed while the route-first curriculum changes
        # exactly one route-learning variable at a time.
        self.events.reset_base.params["roll_range"] = (-0.005, 0.005)
        self.events.reset_base.params["pitch_range"] = (-0.005, 0.005)
        if self.events.reset_slung_load is not None:
            self.events.reset_slung_load.params["max_initial_swing"] = 0.02

        # Begin with random-heading planar straight routes. The fixed W24
        # tensor shape lets the curriculum switch to the final periodic family
        # on a later reset without rebuilding the command term.
        route = self.commands.route
        route.randomize_waypoints = True
        route.regenerate_on_completion = False
        route.random_waypoint_count = 24
        route.route_family = "random_walk"
        route.random_waypoint_ranges.pos_x = (-12.5, 12.5)
        route.random_waypoint_ranges.pos_y = (-12.5, 12.5)
        route.random_waypoint_ranges.pos_z = (0.0, 0.0)
        route.minimum_waypoint_separation = 0.49
        route.maximum_waypoint_separation = 0.51
        route.nominal_heading_change = 0.0
        route.maximum_heading_change = math.radians(5.0)
        route.maximum_vertical_step = 0.0
        route.figure_eight_probability = 0.0
        route.vertical_amplitude_range = (0.0, 0.0)
        route.acceptance_radius = 1.0
        route.target_cruise_speed = 1.00
        route.maximum_lateral_acceleration = 3.0
        route.maximum_braking_acceleration = 4.0
        self.terminations.path_corridor.params["maximum_distance"] = 2.5

    def evaluation_mode(self):
        """Evaluate a seeded figure-eight and bounded random-corner mix."""
        super().evaluation_mode()
        self.events.reset_base.params["roll_range"] = (-0.05, 0.05)
        self.events.reset_base.params["pitch_range"] = (-0.05, 0.05)
        if self.events.reset_slung_load is not None:
            self.events.reset_slung_load.params["max_initial_swing"] = 0.10
        self.commands.route.route_family = "bounded_hard_mix"
        self.commands.route.figure_eight_probability = 0.50
        self.commands.route.vertical_amplitude_range = (0.0, 0.15)
        self.commands.route.random_waypoint_ranges.pos_x = (-4.80, 4.80)
        self.commands.route.random_waypoint_ranges.pos_y = (-4.80, 4.80)
        self.commands.route.random_waypoint_ranges.pos_z = (-0.40, 0.40)
        self.commands.route.minimum_waypoint_separation = 0.90
        self.commands.route.maximum_waypoint_separation = 1.35
        self.commands.route.nominal_heading_change = math.radians(100.0)
        self.commands.route.maximum_heading_change = math.radians(110.0)
        self.commands.route.maximum_vertical_step = 0.15
        self.commands.route.random_sampling_attempts = 32
        self.commands.route.route_sampling_attempts = 16
        self.commands.route.random_heading_change_interval = 3
        self.commands.route.acceptance_radius = 0.50
        self.commands.route.target_cruise_speed = 3.50
        self.rewards.path_progress.params["maximum_rate"] = 3.50
        self.rewards.path_precision.weight = -0.10
        self.rewards.path_precision.params["cross_track_scale"] = 1.00
        self.rewards.path_precision.params["transverse_velocity_scale"] = 1.50
        self.terminations.path_corridor.params["maximum_distance"] = 1.50
        self.episode_length_s = 20.0


# Concise conventional name for scripts that import the only environment directly.
DroneSlungLoadEnvCfg = DroneSlungLoadWaypointEnvCfg
