# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for randomized Franka pickup and insertion of Newton's RJ45 cable."""

from __future__ import annotations

import math
from typing import Literal

from isaaclab_newton.sim.schemas import MujocoJointCfg

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.franka_pour.reset_sampler import ResetDatasetSamplerCfg

from . import mdp
from .asset_provenance import (
    FRANKA_RJ45_FRANKA_LOGICAL_URI,
    FRANKA_RJ45_SEATTLE_TABLE_LOGICAL_URI,
    franka_rj45_asset_contract,
)
from .franka_robot_cfg import (
    PICK_INSERT_ARM_TARGET_TRACKING_LIMITS,
    configure_franka_rj45_external_asset,
    franka_pick_insert_control_contract,
)
from .rj45_env_cfg import (
    RJ45_ENTRY,
    ActionsCfg,
    CurriculumCfg,
    FrankaRJ45InsertionEnvCfg,
    RJ45SceneCfg,
    _config_contract,
    reset_dataset_task_contract,
)
from .table_scene_cfg import configure_seattle_table_external_asset, make_seattle_table_scene_assets

PICK_INSERT_TASK_TRANSLATION = (0.58, 0.15, 0.0)
PICK_INSERT_TASK_ROTATION_XYZW = (0.0, 0.0, 0.0, 1.0)
PICK_INSERT_RJ45_ENTRY_BODY_PATTERNS = (
    r"/World/envs/env_[^/]+/Rj45Assembly",
    r"/World/envs/env_[^/]+/TableContactSurface",
)
"""VBD-owned task bodies, including the exact kinematic Seattle support slab."""

PICK_INSERT_GRASP_OFFSET = (0.0, -0.025, 0.010)
PICK_INSERT_CLOSED_FINGER_POSITION = 0.0
"""Per-finger Franka joint position for a physically closed pick-insert gripper [m]."""
PICK_INSERT_OPEN_FINGER_POSITION = 0.04
"""Per-finger Franka joint position for a physically open pick-insert gripper [m]."""
PICK_INSERT_GRASP_PROXY_FRICTION = 4.5
"""Raw Coulomb friction of the pick-only plug grasp proxy."""
PICK_INSERT_GRASP_PROXY_FACE_TOLERANCE_M = 5.0e-4
"""Tolerance for classifying a contact on a grasp-proxy face [m]."""
PICK_INSERT_PHASE_4_PREGRASP_HEIGHT_M = 0.045
"""Vertical clearance of phase-4 open-pregrasp reset rows [m]."""
PICK_INSERT_PHASE_4_PREGRASP_ORIENTATION_SAMPLER_VERSION = 1
"""Version of the phase-4 tool-local orientation sampler contract."""
PICK_INSERT_PHASE_4_PREGRASP_MAXIMUM_TOP_DOWN_TILT_ERROR_RAD = math.radians(25.0)
"""Maximum phase-4 top-down tool-axis error [rad]."""
PICK_INSERT_PHASE_4_PREGRASP_MAXIMUM_CLOSING_AXIS_TWIST_ERROR_RAD = math.radians(60.0)
"""Maximum symmetric phase-4 finger-closing-axis twist error [rad]."""
PICK_INSERT_EFFECTIVE_GRASP_FRICTION = 3.0
"""Effective finger/proxy friction under Newton's geometric-mean combine rule."""
PICK_INSERT_SUCCESS_MAX_PLUG_SPEED = 0.10
"""Maximum plug six-vector velocity norm accepted by success [m/s and rad/s components]."""
# Table-clearance top-down grasp in the plug frame: tool Y keeps the Franka
# finger closing axis across plug X while tool Z points away from the slab.
PICK_INSERT_GRASP_QUAT_PLUG_XYZW = (
    math.sqrt(0.5),
    math.sqrt(0.5),
    0.0,
    0.0,
)
PICK_INSERT_PHASE_NAMES = (
    "near_insertion",
    "preinsertion",
    "transport",
    "postgrasp",
    "pregrasp",
    "full_pick",
)
PICK_INSERT_RESET_PHASE_FRACTIONS = (0.30, 0.15, 0.08, 0.06, 0.06, 0.35)
"""Long-run reset assignment fractions for phases zero through five."""
PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_SAMPLER_VERSION = 1
PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_BAND_NAMES = ("immediate", "quick", "boundary")
PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_AXIAL_RANGES_M = (
    (0.0010, 0.0016),
    (0.0016, 0.0035),
    (0.0035, 0.0120),
)
PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_WEIGHTS = (0.35, 0.35, 0.30)
_FRANKA_STACK_ARM_HOME = {
    "panda_joint1": 0.0444,
    "panda_joint2": -0.1894,
    "panda_joint3": -0.1107,
    "panda_joint4": -2.5148,
    "panda_joint5": 0.0044,
    "panda_joint6": 2.3775,
    "panda_joint7": 0.6952,
}

_SEATTLE_TABLE, _SEATTLE_CONTACT_SURFACE = make_seattle_table_scene_assets()


@configclass
class PickInsertSceneCfg(RJ45SceneCfg):
    """Seattle-table scene; task connector bodies are injected by the Newton hook."""

    table = _SEATTLE_TABLE
    table_contact_surface = _SEATTLE_CONTACT_SURFACE

    def __post_init__(self) -> None:
        super().__post_init__()
        configure_franka_rj45_external_asset(self.robot)
        configure_seattle_table_external_asset(self.table)
        # Keep native compensation scoped to this MJWarp-owned Franka.  The
        # legacy insertion scene continues using the unmodified robot config,
        # and the VBD-owned cable never enters an inverse-dynamics pass.
        self.robot.spawn.joint_drive_props = [MujocoJointCfg(actuatorgravcomp=True)]
        self.robot.init_state.joint_pos.update(_FRANKA_STACK_ARM_HOME)


@configclass
class PickInsertActionsCfg(ActionsCfg):
    """Larger stack-style residuals plus an open-by-default binary gripper."""

    arm_action = mdp.PersistentResetTargetEMAJointPositionActionCfg(
        asset_name="robot",
        joint_names=[f"panda_joint{index}" for index in range(1, 8)],
        preserve_order=True,
        scale=0.05,
        use_zero_offset=True,
        alpha=0.25,
        max_delta=0.05,
        joint_limit_margin=0.02,
        gravity_compensation=False,
        tracking_error_limits=PICK_INSERT_ARM_TARGET_TRACKING_LIMITS,
    )
    gripper_action = mdp.CurriculumGripperPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        alpha=0.35,
        close_position=PICK_INSERT_CLOSED_FINGER_POSITION,
        neutral_position=PICK_INSERT_OPEN_FINGER_POSITION,
        default_position=PICK_INSERT_OPEN_FINGER_POSITION,
        contact_min_deflection=0.0005,
    )


@configclass
class PickInsertObservationsCfg:
    """Goal-conditioned actor observations and reset labels for the critic."""

    @configclass
    class PolicyCfg(ObsGroup):
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
        arm_target_error = ObsTerm(func=mdp.arm_target_error_obs, scale=4.0)
        tcp_pose = ObsTerm(func=mdp.tcp_pose_obs)
        tcp_velocity = ObsTerm(func=mdp.tcp_velocity_obs, scale=0.1)
        plug_pose = ObsTerm(func=mdp.plug_pose_obs)
        socket_pose = ObsTerm(func=mdp.socket_pose_obs)
        goal_plug_pose = ObsTerm(func=mdp.goal_plug_pose_obs)
        tcp_grasp_error = ObsTerm(func=mdp.tcp_grasp_error_obs, scale=10.0)
        plug_goal_error = ObsTerm(func=mdp.plug_goal_pose_error_obs, scale=10.0)
        plug_velocity = ObsTerm(func=mdp.plug_velocity_obs, scale=0.1)
        cable_shape = ObsTerm(func=mdp.sampled_cable_positions_obs, scale=4.0)
        cable_velocity = ObsTerm(func=mdp.sampled_cable_linear_velocities_obs, scale=0.1)
        finger_position = ObsTerm(func=mdp.finger_position_obs, scale=40.0)
        finger_velocity = ObsTerm(func=mdp.finger_velocity_obs, scale=2.0)
        gripper_target = ObsTerm(func=mdp.gripper_target_obs, scale=40.0)
        gripper_deflection = ObsTerm(func=mdp.gripper_contact_obs, scale=100.0)
        grasp_proxy_contact = ObsTerm(func=mdp.grasp_proxy_contact_obs)
        grasp_stage = ObsTerm(func=mdp.grasp_stage_obs)
        time_remaining = ObsTerm(func=mdp.time_remaining_obs)
        last_action = ObsTerm(func=mdp.last_action, scale=0.2)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        reset_phase = ObsTerm(func=mdp.reset_phase_obs)
        reset_difficulty = ObsTerm(func=mdp.reset_difficulty_obs)
        success_dwell = ObsTerm(func=mdp.success_dwell_obs)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class PickInsertRewardsCfg:
    """Signed stage progress, grasp acquisition, and terminal task outcome."""

    progress = RewTerm(func=mdp.PickInsertProgressReward, weight=1.0)
    grasp_acquired = RewTerm(func=mdp.grasp_acquisition_bonus, weight=0.5)
    success = RewTerm(func=mdp.insertion_success_bonus, weight=10.0)
    failure = RewTerm(func=mdp.terminal_failure, weight=-2.0, params={"include_time_out": True})
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-5.0e-5)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-4)


@configclass
class PickInsertTerminationsCfg:
    """Full-task stage state, physical failures, insertion success, and timeout."""

    stage_context = DoneTerm(func=mdp.PickInsertStageContext)
    nonfinite = DoneTerm(func=mdp.nonfinite_failure)
    task_out_of_bounds = DoneTerm(func=mdp.pick_insert_task_out_of_bounds)
    arm_target_tracking = DoneTerm(func=mdp.arm_target_tracking_failure)
    lost_grasp = DoneTerm(func=mdp.lost_acquired_grasp, params={"minimum_episode_steps": 5})
    success = DoneTerm(func=mdp.stable_pick_insert_success)
    learning_progress_context = DoneTerm(
        func=mdp.PickInsertResetLearningProgress,
        params={"minimum_episode_steps": 3},
    )
    time_out = DoneTerm(func=mdp.unsuccessful_time_out, time_out=True)


@configclass
class PickInsertCurriculumCfg(CurriculumCfg):
    reset_dataset = CurrTerm(func=mdp.RJ45PickInsertResetDatasetCurriculum)


@configclass
class FrankaRJ45PickInsertEnvCfg(FrankaRJ45InsertionEnvCfg):
    """Randomized socket, cable pickup, grasp, transport, and insertion task."""

    scene: PickInsertSceneCfg = PickInsertSceneCfg(num_envs=2, env_spacing=2.0, replicate_physics=True)
    observations: PickInsertObservationsCfg = PickInsertObservationsCfg()
    actions: PickInsertActionsCfg = PickInsertActionsCfg()
    rewards: PickInsertRewardsCfg = PickInsertRewardsCfg()
    terminations: PickInsertTerminationsCfg = PickInsertTerminationsCfg()
    curriculum: PickInsertCurriculumCfg = PickInsertCurriculumCfg()

    task_translation: tuple[float, float, float] = PICK_INSERT_TASK_TRANSLATION
    task_rotation_xyzw: tuple[float, float, float, float] = PICK_INSERT_TASK_ROTATION_XYZW
    rj45_entry_body_patterns: tuple[str, ...] = PICK_INSERT_RJ45_ENTRY_BODY_PATTERNS
    resettable_socket: bool = True
    free_plug_rotation: bool = True
    extra_cable_segments: int = 10
    include_task_support_plane: bool = False
    plug_grasp_offset: tuple[float, float, float] = PICK_INSERT_GRASP_OFFSET
    plug_grasp_orientation_xyzw: tuple[float, float, float, float] = PICK_INSERT_GRASP_QUAT_PLUG_XYZW
    grasp_proxy_friction: float = PICK_INSERT_GRASP_PROXY_FRICTION
    """Raw Coulomb friction assigned only to the pick-insert grasp proxy."""
    grasp_proxy_face_tolerance_m: float = PICK_INSERT_GRASP_PROXY_FACE_TOLERANCE_M
    """Tolerance for opposing local-X grasp-proxy surface contacts [m]."""
    success_max_plug_speed: float = PICK_INSERT_SUCCESS_MAX_PLUG_SPEED
    """Maximum success velocity norm with [m/s] linear and [rad/s] angular components."""

    reset_dataset_path: str = "datasets/franka_rj45_pick_insert/reset_dataset.pt"
    reset_validation_report_path: str = "logs/rsl_rl/franka_rj45_pick_insert/validation/reset_validation.json"
    reset_source: Literal["dataset", "procedural"] = "dataset"
    """Reset source. Interactive play switches to fresh procedural full-pick starts."""
    procedural_reset_max_sampling_attempts: int = 32
    """Maximum bounded rejection-sampling attempts per procedural reset."""
    reset_dataset_rows_per_phase: int = 3334
    reset_dataset_diversity_round_decimals: int = 4
    reset_dataset_min_unique_full_pick_rows: int = 3000
    reset_dataset_min_socket_span_fraction: float = 0.60
    reset_dataset_min_pickup_span_fraction: float = 0.60
    reset_dataset_min_arm_joint_span_fraction: float = 0.50
    reset_dataset_min_full_pick_tcp_distance_span: float = 0.10
    reset_dataset_sampler: ResetDatasetSamplerCfg = ResetDatasetSamplerCfg(
        monitored_history_len=50,
        target_success_rate=0.5,
        kappa=1.0,
        epsilon=1.0e-4,
        uniform_fraction=0.35,
    )
    full_pick_start_fraction: float = 0.35
    reset_dataset_phase_fractions: tuple[float, ...] = PICK_INSERT_RESET_PHASE_FRACTIONS
    """Long-run reset assignment fractions ordered like :data:`PICK_INSERT_PHASE_NAMES`."""

    socket_position_lower: tuple[float, float, float] = (0.52, 0.08, 0.0)
    socket_position_upper: tuple[float, float, float] = (0.66, 0.22, 0.0)
    socket_yaw_range: tuple[float, float] = (-math.radians(25.0), math.radians(25.0))
    pickup_position_lower: tuple[float, float, float] = (0.34, -0.20, 0.0105)
    pickup_position_upper: tuple[float, float, float] = (0.57, -0.015, 0.0145)
    pickup_yaw_range: tuple[float, float] = (-math.radians(70.0), math.radians(70.0))
    minimum_pickup_socket_distance: float = 0.14
    arm_reset_joint_noise: float = 0.12

    grasp_acquisition_distance_m: float = 0.02
    grasp_acquisition_axis_tolerance_rad: float = math.radians(15.0)
    """Maximum tool and finger-axis error for grasp acquisition [rad]."""
    grasp_retention_axis_tolerance_rad: float = math.radians(25.0)
    """Maximum tool and finger-axis error after grasp acquisition [rad]."""
    grasp_loss_grace_steps: int = 3
    transport_stage_distance_m: float = 0.08
    preinsert_stage_distance_m: float = 0.035
    reach_reward_scale_m: float = 0.08
    reach_orientation_reward_scale_rad: float = math.radians(45.0)
    reach_orientation_reward_weight: float = 0.25
    transport_reward_scale_m: float = 0.12
    insertion_reward_scale: float = 0.025
    max_tcp_grasp_distance: float = 0.045
    max_cable_socket_offset: float = 0.60

    task_workspace_lower: tuple[float, float, float] = (0.25, -0.30, -0.02)
    task_workspace_upper: tuple[float, float, float] = (0.78, 0.30, 0.45)
    task_body_workspace_lower: tuple[float, float, float] = (-0.05, -0.48, -0.08)
    task_body_workspace_upper: tuple[float, float, float] = (0.95, 0.45, 0.70)
    max_cable_goal_offset: float = 0.60

    def __post_init__(self) -> None:
        super().__post_init__()
        from isaaclab_visualizers.kit import KitVisualizerCfg

        self.episode_length_s = 12.0
        # The deadline is part of the task objective and carries an explicit
        # failure pulse, so PPO must not bootstrap it as an external truncation.
        self.is_finite_horizon = True
        self.sim.default_visualizer_cfg = KitVisualizerCfg(
            eye=(1.12, -0.82, 0.72),
            lookat=(0.50, 0.0, 0.08),
            origin_type="env",
            origin_env_index=0,
        )

    def play_mode(self) -> None:
        """Use one freshly randomized full-pick start for interactive inference."""
        super().play_mode()
        self.scene.num_envs = 1
        self.reset_source = "procedural"
        self.curriculum = None

    def validate_config(self) -> None:
        if tuple(self.rj45_entry_body_patterns) != PICK_INSERT_RJ45_ENTRY_BODY_PATTERNS:
            raise ValueError(
                "Franka RJ45 pick-insert requires the exact validated RJ45/TableContactSurface VBD ownership selectors."
            )
        if self.is_finite_horizon is not True:
            raise ValueError("Franka RJ45 pick-insert requires a finite task horizon without timeout bootstrapping.")
        _validate_pick_insert_reset_source(self)
        super().validate_config()
        from .physics import RJ45_PICK_INSERT_TOPOLOGY, make_rj45_task_layout

        topology = pick_insert_topology_cfg(self)
        if topology != RJ45_PICK_INSERT_TOPOLOGY:
            raise ValueError("Franka RJ45 pick-insert requires the validated extended movable-socket topology.")
        layout = make_rj45_task_layout(topology)
        if layout.body_count != 48 or layout.socket_body_index != 0:
            raise ValueError("Franka RJ45 pick-insert requires socket, plug, latch, and 45 cable bodies.")
        if (
            float(self.actions.gripper_action.neutral_position) != PICK_INSERT_OPEN_FINGER_POSITION
            or float(self.actions.gripper_action.default_position) != PICK_INSERT_OPEN_FINGER_POSITION
        ):
            raise ValueError("Franka RJ45 pick-insert requires the exact 0.04 m open neutral/default gripper posture.")
        _validate_pick_insert_grasp_contract(self)
        if float(self.success_max_plug_speed) != PICK_INSERT_SUCCESS_MAX_PLUG_SPEED:
            raise ValueError("Franka RJ45 pick-insert requires the exact 0.10 plug success-speed limit.")
        if not isinstance(self.actions.arm_action, mdp.PersistentResetTargetEMAJointPositionActionCfg):
            raise ValueError("Franka RJ45 pick-insert requires the persistent absolute-target arm action.")
        if self.actions.arm_action.gravity_compensation:
            raise ValueError("Pick-insert action-level inverse-dynamics gravity compensation must remain disabled.")
        if tuple(self.actions.arm_action.tracking_error_limits) != PICK_INSERT_ARM_TARGET_TRACKING_LIMITS:
            raise ValueError("Pick-insert requires the validated per-joint target-tracking envelope.")
        joint_drive_props = self.scene.robot.spawn.joint_drive_props
        if (
            not isinstance(joint_drive_props, list)
            or len(joint_drive_props) != 1
            or not isinstance(joint_drive_props[0], MujocoJointCfg)
            or joint_drive_props[0].actuatorgravcomp is not True
        ):
            raise ValueError("Pick-insert requires robot-scoped native MJWarp actuator gravity compensation.")
        for lower_name, upper_name in (
            ("socket_position_lower", "socket_position_upper"),
            ("pickup_position_lower", "pickup_position_upper"),
        ):
            lower = tuple(float(value) for value in getattr(self, lower_name))
            upper = tuple(float(value) for value in getattr(self, upper_name))
            if len(lower) != 3 or len(upper) != 3 or any(low > high for low, high in zip(lower, upper, strict=True)):
                raise ValueError(f"{lower_name}/{upper_name} must contain ordered xyz bounds.")
        for name in ("socket_yaw_range", "pickup_yaw_range"):
            bounds = tuple(float(value) for value in getattr(self, name))
            if len(bounds) != 2 or not bounds[0] < bounds[1]:
                raise ValueError(f"{name} must contain increasing bounds.")
        for name in (
            "minimum_pickup_socket_distance",
            "arm_reset_joint_noise",
            "grasp_acquisition_distance_m",
            "grasp_proxy_face_tolerance_m",
            "transport_stage_distance_m",
            "preinsert_stage_distance_m",
            "reach_reward_scale_m",
            "reach_orientation_reward_scale_rad",
            "transport_reward_scale_m",
            "insertion_reward_scale",
            "max_cable_socket_offset",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        acquisition_tolerance = float(self.grasp_acquisition_axis_tolerance_rad)
        retention_tolerance = float(self.grasp_retention_axis_tolerance_rad)
        if not (
            math.isfinite(acquisition_tolerance)
            and math.isfinite(retention_tolerance)
            and 0.0 < acquisition_tolerance < retention_tolerance < math.pi / 2.0
        ):
            raise ValueError("grasp axis tolerances must be finite and satisfy 0 < acquisition < retention < pi/2.")
        from .physics import GRASP_PROXY_HALF_EXTENTS

        if float(self.grasp_proxy_face_tolerance_m) >= min(GRASP_PROXY_HALF_EXTENTS):
            raise ValueError("grasp_proxy_face_tolerance_m must be smaller than every proxy half extent.")
        if isinstance(self.grasp_loss_grace_steps, bool) or int(self.grasp_loss_grace_steps) < 1:
            raise ValueError("grasp_loss_grace_steps must be a positive integer.")
        if not 0.0 < float(self.reach_orientation_reward_weight) < 1.0:
            raise ValueError("reach_orientation_reward_weight must lie strictly inside (0, 1).")
        if type(self.reset_dataset_rows_per_phase) is not int or self.reset_dataset_rows_per_phase < 1:
            raise ValueError("reset_dataset_rows_per_phase must be a positive plain integer.")
        if (
            type(self.reset_dataset_min_unique_full_pick_rows) is not int
            or not 1 <= self.reset_dataset_min_unique_full_pick_rows <= self.reset_dataset_rows_per_phase
        ):
            raise ValueError(
                "reset_dataset_min_unique_full_pick_rows must be a positive plain integer no larger than "
                "reset_dataset_rows_per_phase."
            )
        if (
            type(self.reset_dataset_diversity_round_decimals) is not int
            or not 0 <= self.reset_dataset_diversity_round_decimals <= 8
        ):
            raise ValueError("reset_dataset_diversity_round_decimals must be a plain integer in [0, 8].")
        for name in (
            "reset_dataset_min_socket_span_fraction",
            "reset_dataset_min_pickup_span_fraction",
            "reset_dataset_min_arm_joint_span_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite and lie in (0, 1].")
        if (
            not math.isfinite(float(self.reset_dataset_min_full_pick_tcp_distance_span))
            or self.reset_dataset_min_full_pick_tcp_distance_span <= 0.0
        ):
            raise ValueError("reset_dataset_min_full_pick_tcp_distance_span must be finite and positive.")
        _validate_pick_insert_reset_phase_fractions(self)
        quat_norm = math.sqrt(sum(float(value) ** 2 for value in self.plug_grasp_orientation_xyzw))
        if not math.isclose(quat_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError("plug_grasp_orientation_xyzw must be normalized.")
        if (
            not isinstance(self.scene.table.spawn, sim_utils.UsdFileCfg)
            or not self.scene.table.spawn.make_uninstanceable
        ):
            raise ValueError("The Seattle table must be recursively editable so authored collision can be disabled.")


def _validate_pick_insert_reset_source(cfg: FrankaRJ45PickInsertEnvCfg) -> None:
    """Validate the mutually exclusive training and play reset paths."""
    if cfg.reset_source not in ("dataset", "procedural"):
        raise ValueError("reset_source must be either 'dataset' or 'procedural'.")
    if cfg.reset_source == "dataset" and cfg.curriculum is None:
        raise ValueError("Dataset resets require the reset-dataset curriculum to assign rows.")
    if cfg.reset_source == "procedural" and cfg.curriculum is not None:
        raise ValueError("Procedural resets must not construct the reset-dataset curriculum.")
    if type(cfg.procedural_reset_max_sampling_attempts) is not int or cfg.procedural_reset_max_sampling_attempts < 1:
        raise ValueError("procedural_reset_max_sampling_attempts must be a positive plain integer.")


def _validate_pick_insert_reset_phase_fractions(cfg: FrankaRJ45PickInsertEnvCfg) -> None:
    """Validate phase weights and their legacy full-pick alias."""
    if not 0.0 < float(cfg.full_pick_start_fraction) < 1.0:
        raise ValueError("full_pick_start_fraction must lie strictly inside (0, 1).")
    try:
        phase_fractions = tuple(float(value) for value in cfg.reset_dataset_phase_fractions)
    except (TypeError, ValueError) as error:
        raise ValueError("reset_dataset_phase_fractions must contain six finite numeric fractions.") from error
    if len(phase_fractions) != len(PICK_INSERT_PHASE_NAMES):
        raise ValueError("reset_dataset_phase_fractions must contain exactly six fractions.")
    if any(
        isinstance(value, bool) or not math.isfinite(fraction) or fraction < 0.0
        for value, fraction in zip(cfg.reset_dataset_phase_fractions, phase_fractions, strict=True)
    ):
        raise ValueError("reset_dataset_phase_fractions must contain finite nonnegative fractions.")
    if not math.isclose(math.fsum(phase_fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("reset_dataset_phase_fractions must sum exactly to 1.0.")
    if phase_fractions[5] != float(cfg.full_pick_start_fraction):
        raise ValueError("reset_dataset_phase_fractions phase-5 fraction must equal full_pick_start_fraction.")


def _validate_pick_insert_grasp_contract(cfg: FrankaRJ45PickInsertEnvCfg) -> None:
    """Reject any material or gripper mutation outside the validated pick-only grasp."""
    from .physics import GRASP_FRICTION

    close_position = cfg.actions.gripper_action.close_position
    if (
        isinstance(close_position, bool)
        or not isinstance(close_position, int | float)
        or not math.isfinite(close_position)
        or float(close_position) != PICK_INSERT_CLOSED_FINGER_POSITION
    ):
        raise ValueError("Franka RJ45 pick-insert requires the exact 0.0 m closed gripper target.")
    if (
        isinstance(cfg.grasp_proxy_friction, bool)
        or not isinstance(cfg.grasp_proxy_friction, int | float)
        or not math.isfinite(cfg.grasp_proxy_friction)
        or float(cfg.grasp_proxy_friction) != PICK_INSERT_GRASP_PROXY_FRICTION
    ):
        raise ValueError("Franka RJ45 pick-insert requires the exact 4.5 grasp-proxy friction.")
    if float(GRASP_FRICTION) != 2.0:
        raise ValueError("Franka RJ45 pick-insert requires the unchanged raw Franka finger friction 2.0.")
    effective_grasp_friction = math.sqrt(float(GRASP_FRICTION) * float(cfg.grasp_proxy_friction))
    if not math.isclose(
        effective_grasp_friction,
        PICK_INSERT_EFFECTIVE_GRASP_FRICTION,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Franka RJ45 pick-insert requires exact effective finger/proxy friction 3.0.")


def pick_insert_reset_dataset_task_contract(cfg: FrankaRJ45PickInsertEnvCfg) -> dict[str, object]:
    """Return the exact physical, scene, goal, and learning contract for reset artifacts."""
    from .physics import GRASP_FRICTION, make_rj45_task_layout, rj45_reset_physics_contract
    from .pick_insert_reset_dataset_io import (
        PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD,
        PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M,
    )

    contract = reset_dataset_task_contract(cfg)
    base_contract_version = int(contract["contract_version"])
    topology = pick_insert_topology_cfg(cfg)
    layout = make_rj45_task_layout(topology)
    contract["contract_version"] = 8
    contract["base_contract_version"] = base_contract_version
    contract["task_variant"] = "franka-rj45-pick-insert"
    # Runtime binds verified absolute paths immediately before scene startup.
    # Artifacts retain only stable logical identities so preflight and live
    # contracts are identical across hosts and cache roots.
    contract["robot"]["asset"] = FRANKA_RJ45_FRANKA_LOGICAL_URI
    contract["robot"]["spawn"]["usd_path"] = FRANKA_RJ45_FRANKA_LOGICAL_URI
    contract["static_scene"]["table_spawn"]["usd_path"] = FRANKA_RJ45_SEATTLE_TABLE_LOGICAL_URI
    contract["external_assets"] = franka_rj45_asset_contract()
    contract["task_body_count"] = layout.body_count
    contract["task_body_order"] = layout.body_names
    contract["reset_state_representation"] = {
        "contract_version": 2,
        "task_body_pose_frame": "environment-local-xyzw",
        "task_body_velocity_frame": "world-linear-angular",
        "vbd_entry_name": RJ45_ENTRY,
        "vbd_body_order_source": "task_body_order",
        "vbd_previous_pose_field": "task_body_previous_pose",
        "vbd_coupling_previous_pose_field": "task_body_coupling_previous_pose",
        "vbd_pose_history_frame": "environment-local-xyzw",
        "restore_semantics": "deferred-one-shot-after-input-and-proxy-rebaseline-before-first-vbd-solve",
        "preserved_input_task_body_range_half_open": (
            layout.cable_body_slice.start,
            layout.cable_body_slice.stop - 1,
        ),
        "preserved_input_semantics": "scatter-history-without-pose-delta-velocity-injection-or-rewind",
    }
    effective_grasp_friction = math.sqrt(float(GRASP_FRICTION) * float(cfg.grasp_proxy_friction))
    pick_physics_contract = rj45_reset_physics_contract(topology)
    pick_physics_contract.update(
        {
            "contract_version": 6,
            "franka_finger_raw_friction": float(GRASP_FRICTION),
            "grasp_proxy_raw_friction": float(cfg.grasp_proxy_friction),
            "grasp_contact_friction_combine_rule": "geometric-mean",
            "grasp_contact_effective_friction": effective_grasp_friction,
        }
    )
    contract["rj45_physics"] = pick_physics_contract
    contract["robot"]["reset_control_convention"] = franka_pick_insert_control_contract()
    contract["simulation"]["control_decimation"] = int(cfg.decimation)
    contract["simulation"]["control_step_dt"] = float(cfg.sim.dt) * int(cfg.decimation)
    contract["static_scene"]["table_contact_initial_state"] = _config_contract(
        cfg.scene.table_contact_surface.init_state
    )
    contract["static_scene"]["table_contact_spawn"] = _config_contract(
        cfg.scene.table_contact_surface.spawn,
        exclude=("spawn_path",),
    )
    from .task_success import RJ45_GOAL_LOCAL_SUCCESS_PREDICATE_VERSION

    contract["validation_geometry"]["grasp"] = {
        "contract_version": 2,
        "acquisition_axis_tolerance_rad": float(cfg.grasp_acquisition_axis_tolerance_rad),
        "retention_axis_tolerance_rad": float(cfg.grasp_retention_axis_tolerance_rad),
        "tool_axis": (0.0, 0.0, 1.0),
        "tool_axis_comparison": "signed",
        "finger_closing_axis": (0.0, 1.0, 0.0),
        "finger_closing_axis_comparison": "signed-coupled-to-proxy-face-assignment",
        "proxy_contact_faces": "exclusive-opposing-local-x-surface",
        "canonical_proxy_face_assignment": "left:+local-x,right:-local-x",
        "canonical_closing_axis_comparison": "relative-rotation-yy>=cos(tolerance)",
        "swapped_proxy_face_assignment": "left:-local-x,right:+local-x",
        "swapped_closing_axis_comparison": "relative-rotation-yy<=-cos(tolerance)",
        "proxy_face_tolerance_m": float(cfg.grasp_proxy_face_tolerance_m),
    }

    contract["pick_insert"] = {
        "semantics_version": 8,
        "goal_local_success_predicate_version": RJ45_GOAL_LOCAL_SUCCESS_PREDICATE_VERSION,
        "phase_names": PICK_INSERT_PHASE_NAMES,
        "plug_grasp_orientation_xyzw": tuple(cfg.plug_grasp_orientation_xyzw),
        "finger_closed_position": float(cfg.actions.gripper_action.close_position),
        "finger_open_position": float(cfg.actions.gripper_action.default_position),
        "goal_max_task_body_drift_m": PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M,
        "goal_max_plug_relative_latch_angle_rad": PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD,
        "socket_position_lower": tuple(cfg.socket_position_lower),
        "socket_position_upper": tuple(cfg.socket_position_upper),
        "socket_yaw_range": tuple(cfg.socket_yaw_range),
        "pickup_position_lower": tuple(cfg.pickup_position_lower),
        "pickup_position_upper": tuple(cfg.pickup_position_upper),
        "pickup_yaw_range": tuple(cfg.pickup_yaw_range),
        "minimum_pickup_socket_distance": float(cfg.minimum_pickup_socket_distance),
        "arm_reset_joint_noise": float(cfg.arm_reset_joint_noise),
        "reset_dataset_rows_per_phase": int(cfg.reset_dataset_rows_per_phase),
        "reset_dataset_phase_fractions": tuple(float(value) for value in cfg.reset_dataset_phase_fractions),
        "full_pick_diversity": {
            "round_decimals": int(cfg.reset_dataset_diversity_round_decimals),
            "minimum_unique_socket_rows": int(cfg.reset_dataset_min_unique_full_pick_rows),
            "minimum_unique_plug_rows": int(cfg.reset_dataset_min_unique_full_pick_rows),
            "minimum_unique_arm_rows": int(cfg.reset_dataset_min_unique_full_pick_rows),
            "minimum_socket_span_fraction": float(cfg.reset_dataset_min_socket_span_fraction),
            "minimum_pickup_span_fraction": float(cfg.reset_dataset_min_pickup_span_fraction),
            "minimum_arm_joint_span_fraction": float(cfg.reset_dataset_min_arm_joint_span_fraction),
            "minimum_tcp_grasp_distance_span_m": float(cfg.reset_dataset_min_full_pick_tcp_distance_span),
        },
        "full_pick_start_fraction": float(cfg.full_pick_start_fraction),
        "phase_0_reverse_curriculum_sampling": pick_insert_phase_0_reverse_curriculum_sampling_contract(),
        "phase_4_pregrasp_orientation_sampling": pick_insert_phase_4_pregrasp_orientation_sampling_contract(),
        "grasp_acquisition_distance_m": float(cfg.grasp_acquisition_distance_m),
        "grasp_loss_grace_steps": int(cfg.grasp_loss_grace_steps),
        "transport_stage_distance_m": float(cfg.transport_stage_distance_m),
        "preinsert_stage_distance_m": float(cfg.preinsert_stage_distance_m),
        "reach_reward_scale_m": float(cfg.reach_reward_scale_m),
        "reach_orientation_reward_scale_rad": float(cfg.reach_orientation_reward_scale_rad),
        "reach_orientation_reward_weight": float(cfg.reach_orientation_reward_weight),
        "transport_reward_scale_m": float(cfg.transport_reward_scale_m),
        "insertion_reward_scale": float(cfg.insertion_reward_scale),
        "max_cable_socket_offset": float(cfg.max_cable_socket_offset),
        "episode_length_s": float(cfg.episode_length_s),
        "is_finite_horizon": bool(cfg.is_finite_horizon),
        "observations": _config_contract(cfg.observations),
        "rewards": _config_contract(cfg.rewards),
        "terminations": _config_contract(cfg.terminations),
    }
    return contract


def pick_insert_phase_0_reverse_curriculum_sampling_contract() -> dict[str, object]:
    """Return the immutable near-insertion reverse-curriculum sampler contract."""
    return {
        "sampler_version": PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_SAMPLER_VERSION,
        "phase": 0,
        "phase_name": PICK_INSERT_PHASE_NAMES[0],
        "frame": "goal-plug-local",
        "axial_offset_semantics": "positive-pre-seat-distance-along-negative-local-y",
        "band_names": PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_BAND_NAMES,
        "axial_offset_ranges_m": PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_AXIAL_RANGES_M,
        "band_weights": PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_WEIGHTS,
        "geometric_success_at_reset": False,
        "rng_owner": "PickInsertResetDatasetGenerator.random",
    }


def pick_insert_phase_4_pregrasp_orientation_sampling_contract() -> dict[str, object]:
    """Return the immutable phase-4 open-pregrasp reset sampler contract."""
    maximum_tilt = PICK_INSERT_PHASE_4_PREGRASP_MAXIMUM_TOP_DOWN_TILT_ERROR_RAD
    maximum_twist = PICK_INSERT_PHASE_4_PREGRASP_MAXIMUM_CLOSING_AXIS_TWIST_ERROR_RAD
    return {
        "sampler_version": PICK_INSERT_PHASE_4_PREGRASP_ORIENTATION_SAMPLER_VERSION,
        "phase": 4,
        "phase_name": PICK_INSERT_PHASE_NAMES[4],
        "starts_grasped": False,
        "clearance_height_m": PICK_INSERT_PHASE_4_PREGRASP_HEIGHT_M,
        "frame": "canonical-grasp-tool-local",
        "composition": "canonical-tcp * top-down-tilt * closing-axis-twist",
        "top_down_tilt_distribution": "uniform-solid-angle-cone",
        "top_down_tilt_range_rad": (0.0, maximum_tilt),
        "tilt_azimuth_distribution": "uniform",
        "tilt_azimuth_range_rad": (-math.pi, math.pi),
        "closing_axis_twist_distribution": "uniform",
        "closing_axis_twist_range_rad": (-maximum_twist, maximum_twist),
        "sampled_once_per_candidate": True,
        "rng_owner": "PickInsertResetDatasetGenerator.random",
        "additional_ik_solves_per_candidate": 0,
        "additional_simulation_steps_per_candidate": 0,
        "starts_grasped_phases_use_canonical_orientation": (0, 1, 2, 3),
        "full_pick_phase_5_orientation_sampling": "unchanged-away-pose",
    }


def pick_insert_play_reset_seed_contract(cfg: FrankaRJ45PickInsertEnvCfg) -> dict[str, object]:
    """Return the physics subset that a packaged play-reset seed must match."""
    contract = pick_insert_reset_dataset_task_contract(cfg)
    static_scene = contract["static_scene"]
    return {
        "contract_version": 1,
        "task_variant": contract["task_variant"],
        "task_translation": contract["task_translation"],
        "task_rotation_xyzw": contract["task_rotation_xyzw"],
        "task_body_count": contract["task_body_count"],
        "task_body_order": contract["task_body_order"],
        "runtime_physics_versions": contract["runtime_physics_versions"],
        "reset_state_representation": contract["reset_state_representation"],
        "rj45_physics": contract["rj45_physics"],
        "simulation": contract["simulation"],
        "coupler": contract["coupler"],
        "table_contact_initial_state": static_scene["table_contact_initial_state"],
        "table_contact_spawn": static_scene["table_contact_spawn"],
    }


def pick_insert_topology_cfg(cfg: FrankaRJ45PickInsertEnvCfg):
    """Create the immutable physics topology lazily to keep config discovery light."""
    from .physics import RJ45_PICK_INSERT_PLUG_PASSIVE_ANGULAR_DAMPING_RATE, Rj45AssemblyTopologyCfg

    return Rj45AssemblyTopologyCfg(
        resettable_socket=bool(cfg.resettable_socket),
        free_plug_rotation=bool(cfg.free_plug_rotation),
        extra_cable_segments=int(cfg.extra_cable_segments),
        include_task_support_plane=bool(cfg.include_task_support_plane),
        plug_passive_angular_damping_rate=RJ45_PICK_INSERT_PLUG_PASSIVE_ANGULAR_DAMPING_RATE,
    )


__all__ = [
    "FrankaRJ45PickInsertEnvCfg",
    "PICK_INSERT_CLOSED_FINGER_POSITION",
    "PICK_INSERT_EFFECTIVE_GRASP_FRICTION",
    "PICK_INSERT_GRASP_OFFSET",
    "PICK_INSERT_GRASP_PROXY_FRICTION",
    "PICK_INSERT_GRASP_QUAT_PLUG_XYZW",
    "PICK_INSERT_OPEN_FINGER_POSITION",
    "PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_AXIAL_RANGES_M",
    "PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_BAND_NAMES",
    "PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_SAMPLER_VERSION",
    "PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_WEIGHTS",
    "PICK_INSERT_PHASE_NAMES",
    "PICK_INSERT_RESET_PHASE_FRACTIONS",
    "PICK_INSERT_RJ45_ENTRY_BODY_PATTERNS",
    "PICK_INSERT_SUCCESS_MAX_PLUG_SPEED",
    "PICK_INSERT_TASK_ROTATION_XYZW",
    "PICK_INSERT_TASK_TRANSLATION",
    "PickInsertSceneCfg",
    "pick_insert_phase_0_reverse_curriculum_sampling_contract",
    "pick_insert_play_reset_seed_contract",
    "pick_insert_reset_dataset_task_contract",
    "pick_insert_topology_cfg",
]
