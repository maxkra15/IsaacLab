# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for reset-driven Franka insertion of Newton's RJ45 cable."""

from __future__ import annotations

import importlib.metadata
import math
from collections.abc import Mapping
from typing import Any, Literal

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
    VBDSolverCfg,
)

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import spawn_franka_with_arm_collisions
from isaaclab_tasks.contrib.franka_pour.reset_sampler import ResetDatasetSamplerCfg

from . import mdp
from .franka_robot_cfg import FRANKA_RJ45_CFG, franka_reset_control_contract
from .task_success import RJ45_SUCCESS_PREDICATE_VERSION

RIGID_ENTRY = "franka"
RJ45_ENTRY = "rj45"

_RIGID_ENTRY_BODY_PATTERNS = (r"/World/envs/env_[^/]+/Robot",)
_RJ45_ENTRY_BODY_PATTERNS = (r"/World/envs/env_[^/]+/Rj45Assembly",)
_PROXY_BODY_PATTERNS = (
    r"/World/envs/env_[^/]+/Robot/Geometry/.*panda_hand",
    r"/World/envs/env_[^/]+/Robot/Geometry/.*panda_(left|right)finger",
)

RJ45_TASK_TRANSLATION = (0.55, 0.0, 0.25)
RJ45_TASK_ROTATION_XYZW = (0.0, 0.0, 0.0, 1.0)
RJ45_PLUG_GRASP_OFFSET = (0.0, -0.025, 0.0)
_RIGID_CONTACTS_PER_WORLD = 1024

_ARM_HOME = (-1.05, 0.55, 0.55, -2.65, 2.35, 1.55, 0.25)
_ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))


def _coupler_cfg(cfg: FrankaRJ45InsertionEnvCfg) -> CouplerProxyCfg:
    coupler = cfg.sim.physics.solver_cfg
    if not isinstance(coupler, CouplerProxyCfg):
        raise TypeError("Franka RJ45 requires CouplerProxyCfg.")
    if len(coupler.entries) != 2 or len(coupler.proxies) != 1:
        raise ValueError("Franka RJ45 requires exactly two coupled entries and one directed proxy.")
    return coupler


def _coupler_entry(coupler: CouplerProxyCfg, name: str) -> CouplerEntryCfg:
    matches = [entry for entry in coupler.entries if entry.name == name]
    if len(matches) != 1:
        raise ValueError(f"Franka RJ45 requires exactly one coupled entry named {name!r}.")
    return matches[0]


def _coupler_proxy(coupler: CouplerProxyCfg) -> CouplerProxyMappingCfg:
    matches = [proxy for proxy in coupler.proxies if proxy.source == RIGID_ENTRY and proxy.destination == RJ45_ENTRY]
    if len(matches) != 1:
        raise ValueError(f"Franka RJ45 requires exactly one {RIGID_ENTRY!r}->{RJ45_ENTRY!r} proxy.")
    return matches[0]


def _stable_contract_value(value: Any) -> Any:
    """Convert config values into the reset artifact's canonical value domain."""
    if value is None:
        return value
    # Artifact loading deliberately uses ``weights_only=True``. Normalize
    # config primitive subclasses (notably ResolvableString) to the exact safe
    # built-ins accepted by PyTorch's restricted unpickler.
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}:{value.__qualname__}"
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        qualname = getattr(value, "__qualname__", type(value).__qualname__)
        return f"{module}:{qualname}"
    if isinstance(value, Mapping):
        return {str(key): _stable_contract_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        converted = tuple(_stable_contract_value(item) for item in value)
        return converted if isinstance(value, tuple) else list(converted)
    if hasattr(value, "__dict__"):
        return {key: _stable_contract_value(item) for key, item in vars(value).items()}
    raise TypeError(f"Unsupported reset-contract configuration value: {type(value).__name__}.")


def _config_contract(config: Any, *, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    return {key: _stable_contract_value(value) for key, value in vars(config).items() if key not in exclude}


def _runtime_physics_versions() -> dict[str, str]:
    versions = {}
    for name in ("newton", "warp-lang", "mujoco-warp", "isaaclab-newton"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


@configclass
class RJ45SceneCfg(InteractiveSceneCfg):
    """Franka scene; the RJ45 assembly is injected by a Newton builder hook."""

    robot: ArticulationCfg = FRANKA_RJ45_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.func = spawn_franka_with_arm_collisions
    robot.spawn.articulation_props.enabled_self_collisions = True
    robot.init_state.joint_pos.update(dict(zip(_ARM_JOINT_NAMES, _ARM_HOME, strict=True)))
    robot.init_state.joint_pos["panda_finger_joint.*"] = 0.008

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, -0.525)),
        spawn=sim_utils.CuboidCfg(
            size=(1.3, 0.9, 1.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.20, 0.22)),
        ),
    )
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=900.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class ActionsCfg:
    """Measured-state arm deltas and a filtered binary grasp command."""

    arm_action = mdp.ResetTargetEMARelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[f"panda_joint{index}" for index in range(1, 8)],
        preserve_order=True,
        scale=0.025,
        use_zero_offset=True,
        alpha=0.25,
        max_delta=0.025,
        joint_limit_margin=0.02,
        gravity_compensation=False,
    )
    gripper_action = mdp.CurriculumGripperPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        alpha=0.35,
        close_position=0.004,
        neutral_position=0.012,
        default_position=0.005,
        contact_min_deflection=0.0005,
    )


@configclass
class ObservationsCfg:
    """Actor state plus reset/goal state reserved for the asymmetric critic."""

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
        tcp_pose = ObsTerm(func=mdp.tcp_pose_obs)
        plug_pose = ObsTerm(func=mdp.plug_pose_obs)
        plug_goal_error = ObsTerm(func=mdp.plug_goal_error_obs, scale=40.0)
        plug_velocity = ObsTerm(func=mdp.plug_velocity_obs, scale=0.1)
        cable_goal_offsets = ObsTerm(func=mdp.sampled_cable_positions_obs, scale=10.0)
        finger_position = ObsTerm(func=mdp.finger_position_obs, scale=40.0)
        finger_velocity = ObsTerm(func=mdp.finger_velocity_obs, scale=2.0)
        gripper_target = ObsTerm(func=mdp.gripper_target_obs, scale=40.0)
        gripper_contact = ObsTerm(func=mdp.gripper_contact_obs, scale=100.0)
        time_remaining = ObsTerm(func=mdp.time_remaining_obs)
        last_action = ObsTerm(func=mdp.last_action, scale=0.2)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        reset_difficulty = ObsTerm(func=mdp.reset_difficulty_obs)
        success_dwell = ObsTerm(func=mdp.success_dwell_obs)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class RewardsCfg:
    """Sparse success/failure objective with small smooth-control costs."""

    success = RewTerm(func=mdp.insertion_success_bonus, weight=10.0)
    # Every reset row is physically validated as recoverable within the five-
    # second horizon. Treat an unsuccessful timeout as a terminal failure so
    # holding the cable motionless cannot dominate insertion exploration.
    failure = RewTerm(func=mdp.terminal_failure, weight=-1.0, params={"include_time_out": True})
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-1.0e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-2.0e-4)


@configclass
class TerminationsCfg:
    """Instability, grasp loss, stable insertion, local progress, and timeout."""

    nonfinite = DoneTerm(func=mdp.nonfinite_failure)
    task_out_of_bounds = DoneTerm(func=mdp.task_out_of_bounds)
    lost_grasp = DoneTerm(func=mdp.lost_grasp, params={"minimum_episode_steps": 3})
    success = DoneTerm(func=mdp.stable_insertion_success)
    learning_progress_context = DoneTerm(
        func=mdp.InsertionResetLearningProgress,
        params={"minimum_episode_steps": 3},
    )
    time_out = DoneTerm(func=mdp.unsuccessful_time_out, time_out=True)


@configclass
class EventsCfg:
    reset_scene = EventTerm(func=mdp.reset_rj45_scene, mode="reset")


@configclass
class CurriculumCfg:
    reset_dataset = CurrTerm(func=mdp.RJ45ResetDatasetCurriculum)


@configclass
class FrankaRJ45InsertionEnvCfg(ManagerBasedRLEnvCfg):
    """Fixed-goal, reset-dataset Franka RJ45 insertion environment."""

    scene: RJ45SceneCfg = RJ45SceneCfg(num_envs=2, env_spacing=1.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    tcp_body_name: str = "panda_hand"
    tcp_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.1034)
    tcp_offset_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    task_translation: tuple[float, float, float] = RJ45_TASK_TRANSLATION
    task_rotation_xyzw: tuple[float, float, float, float] = RJ45_TASK_ROTATION_XYZW
    plug_grasp_offset: tuple[float, float, float] = RJ45_PLUG_GRASP_OFFSET

    reset_dataset_path: str = "datasets/franka_rj45_insertion/reset_dataset.pt"
    reset_validation_report_path: str = "logs/rsl_rl/franka_rj45_insertion/validation/reset_validation.json"
    reset_dataset_content_sha256: str | None = None
    reset_dataset_sampling_mode: Literal["adaptive", "uniform"] = "adaptive"
    reset_dataset_sampler: ResetDatasetSamplerCfg = ResetDatasetSamplerCfg(
        monitored_history_len=50,
        target_success_rate=0.5,
        kappa=1.0,
        epsilon=1.0e-4,
        uniform_fraction=0.35,
    )
    curriculum_freeze: bool = False

    success_axial_tolerance: float = 8.0e-4
    success_axial_overtravel_tolerance: float = 2.0e-4
    success_radial_tolerance: float = 7.5e-4
    success_plug_angle_tolerance: float = math.radians(3.0)
    success_latch_angle_tolerance: float = math.radians(3.0)
    success_max_plug_speed: float = 0.01
    success_dwell_time_s: float = 0.15
    max_tcp_grasp_distance: float = 0.02
    max_plug_spatial_speed: float = 20.0
    task_workspace_lower: tuple[float, float, float] = (0.48, -0.12, 0.18)
    task_workspace_upper: tuple[float, float, float] = (0.62, 0.06, 0.32)
    task_body_workspace_lower: tuple[float, float, float] = (0.25, -0.40, -0.05)
    task_body_workspace_upper: tuple[float, float, float] = (0.85, 0.30, 0.65)
    max_task_body_angular_speed: float = 50.0
    max_task_body_linear_speed: float = 20.0
    max_cable_goal_offset: float = 0.25

    def __post_init__(self) -> None:
        from isaaclab_visualizers.kit import KitVisualizerCfg

        self.decimation = 4
        self.episode_length_s = 5.0
        self.is_finite_horizon = False
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.use_newton_actuators = True
        self.sim.default_visualizer_cfg = KitVisualizerCfg(
            eye=(1.05, -0.75, 0.58),
            lookat=(0.55, -0.02, 0.25),
            origin_type="env",
            origin_env_index=0,
        )
        self.sim.physics = NewtonCfg(
            solver_cfg=CouplerProxyCfg(
                entries=[
                    CouplerEntryCfg(
                        name=RIGID_ENTRY,
                        solver_cfg=MJWarpSolverCfg(
                            use_mujoco_contacts=False,
                            cone="elliptic",
                            ls_iterations=20,
                            integrator="implicitfast",
                            njmax=2560,
                            nconmax=512,
                        ),
                        bodies=list(_RIGID_ENTRY_BODY_PATTERNS),
                    ),
                    CouplerEntryCfg(
                        name=RJ45_ENTRY,
                        solver_cfg=VBDSolverCfg(
                            iterations=12,
                            rigid_compliant_alm=True,
                            rigid_contact_hard=False,
                            rigid_body_contact_buffer_size=256,
                        ),
                        bodies=list(_RJ45_ENTRY_BODY_PATTERNS),
                        include_static_shapes=True,
                    ),
                ],
                proxies=[
                    CouplerProxyMappingCfg(
                        source=RIGID_ENTRY,
                        destination=RJ45_ENTRY,
                        bodies=list(_PROXY_BODY_PATTERNS),
                        mode="staggered",
                        collide_interval=1,
                        collision_pipeline=NewtonCollisionPipelineCfg(),
                    )
                ],
                iterations=1,
            ),
            default_shape_cfg=NewtonShapeCfg(ke=1.0e5, kd=0.0, mu=0.0, gap=0.002),
            collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=_RIGID_CONTACTS_PER_WORLD * self.scene.num_envs),
            num_substeps=3,
            collision_decimation=1,
            use_cuda_graph=True,
        )

    def validate_config(self) -> None:
        """Reject overrides that break the fixed-goal/reset contract."""
        from .physics import RJ45_REFERENCE_SIM_DT, validate_rj45_vbd_solver_cfg

        if self.reset_dataset_sampling_mode not in ("adaptive", "uniform"):
            raise ValueError("reset_dataset_sampling_mode must be 'adaptive' or 'uniform'.")
        for name in ("reset_dataset_path", "reset_validation_report_path"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string.")
        if not isinstance(self.curriculum_freeze, bool):
            raise TypeError("curriculum_freeze must be a bool.")
        self.reset_dataset_sampler.validate_values()
        for name in (
            "success_axial_tolerance",
            "success_axial_overtravel_tolerance",
            "success_radial_tolerance",
            "success_plug_angle_tolerance",
            "success_latch_angle_tolerance",
            "success_max_plug_speed",
            "success_dwell_time_s",
            "max_tcp_grasp_distance",
            "max_plug_spatial_speed",
            "max_task_body_angular_speed",
            "max_task_body_linear_speed",
            "max_cable_goal_offset",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("success_plug_angle_tolerance", "success_latch_angle_tolerance"):
            if float(getattr(self, name)) > math.pi:
                raise ValueError(f"{name} must not exceed pi radians.")
        for lower_name, upper_name in (
            ("task_workspace_lower", "task_workspace_upper"),
            ("task_body_workspace_lower", "task_body_workspace_upper"),
        ):
            lower = tuple(getattr(self, lower_name))
            upper = tuple(getattr(self, upper_name))
            if len(lower) != 3 or len(upper) != 3 or not all(math.isfinite(float(value)) for value in (*lower, *upper)):
                raise ValueError(f"{lower_name}/{upper_name} must contain three finite values.")
            if any(float(low) >= float(high) for low, high in zip(lower, upper, strict=True)):
                raise ValueError(f"{lower_name} must be strictly smaller than {upper_name}.")
        if tuple(self.task_rotation_xyzw) != RJ45_TASK_ROTATION_XYZW:
            raise ValueError("The validated RJ45 insertion geometry currently requires identity task rotation.")
        coupler = _coupler_cfg(self)
        rigid_entry = _coupler_entry(coupler, RIGID_ENTRY)
        rj45_entry = _coupler_entry(coupler, RJ45_ENTRY)
        proxy = _coupler_proxy(coupler)
        if not isinstance(rigid_entry.solver_cfg, MJWarpSolverCfg):
            raise TypeError(f"Coupled entry {RIGID_ENTRY!r} must use MJWarpSolverCfg.")
        if not isinstance(rj45_entry.solver_cfg, VBDSolverCfg):
            raise TypeError(f"Coupled entry {RJ45_ENTRY!r} must use VBDSolverCfg.")
        if tuple(rigid_entry.bodies) != _RIGID_ENTRY_BODY_PATTERNS:
            raise ValueError(f"Coupled entry {RIGID_ENTRY!r} body selectors do not match the validated task.")
        if tuple(rj45_entry.bodies) != _RJ45_ENTRY_BODY_PATTERNS:
            raise ValueError(f"Coupled entry {RJ45_ENTRY!r} body selectors do not match the validated task.")
        if tuple(proxy.bodies) != _PROXY_BODY_PATTERNS:
            raise ValueError("The Franka-to-RJ45 proxy body selectors do not match the validated task.")
        validate_rj45_vbd_solver_cfg(rj45_entry.solver_cfg)
        solver_dt = float(self.sim.dt) / int(self.sim.physics.num_substeps)
        if not math.isclose(solver_dt, RJ45_REFERENCE_SIM_DT, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"Franka RJ45 requires Newton's 1/360 s solver step; got sim.dt/num_substeps={solver_dt}.")
        if int(self.sim.physics.collision_decimation) != 1:
            raise ValueError("Franka RJ45 requires collision detection on every solver step.")
        if self.actions.arm_action.gravity_compensation:
            raise ValueError(
                "Franka RJ45 cannot use global inverse-dynamics gravity compensation while Newton "
                "JointType.CABLE articulations are present; use the reset target bias instead."
            )


def reset_dataset_task_contract(cfg: FrankaRJ45InsertionEnvCfg) -> dict[str, object]:
    """Return exact runtime fields whose changes invalidate stored reset states."""
    # Import task physics only when producing/validating an artifact contract;
    # ordinary config discovery remains import-light.
    from .physics import rj45_reset_physics_contract

    coupler = _coupler_cfg(cfg)
    rigid_entry = _coupler_entry(coupler, RIGID_ENTRY)
    rj45_entry = _coupler_entry(coupler, RJ45_ENTRY)
    proxy = _coupler_proxy(coupler)
    outer_collision = _config_contract(cfg.sim.physics.collision_cfg, exclude=("rigid_contact_max",))
    return {
        "contract_version": 3,
        "task_body_count": 37,
        "task_translation": tuple(cfg.task_translation),
        "task_rotation_xyzw": tuple(cfg.task_rotation_xyzw),
        "plug_grasp_offset": tuple(cfg.plug_grasp_offset),
        "runtime_physics_versions": _runtime_physics_versions(),
        "robot": {
            "asset": str(cfg.scene.robot.spawn.usd_path),
            # Scene construction mutates SpawnCfg.spawn_path from None to the
            # resolved prim path. It is placement bookkeeping rather than a
            # physical parameter, and would make preflight and live contracts
            # differ despite constructing the same scene.
            "spawn": _config_contract(cfg.scene.robot.spawn, exclude=("spawn_path",)),
            "rigid_properties": _config_contract(cfg.scene.robot.spawn.rigid_props),
            "articulation_properties": _config_contract(cfg.scene.robot.spawn.articulation_props),
            "initial_state": _config_contract(cfg.scene.robot.init_state),
            "initial_joint_positions": _stable_contract_value(cfg.scene.robot.init_state.joint_pos),
            "actuators": {name: _config_contract(actuator) for name, actuator in cfg.scene.robot.actuators.items()},
            "reset_control_convention": franka_reset_control_contract(),
            "tcp_body_name": cfg.tcp_body_name,
            "tcp_offset_pos": tuple(cfg.tcp_offset_pos),
            "tcp_offset_rot": tuple(cfg.tcp_offset_rot),
        },
        "static_scene": {
            "table_initial_state": _config_contract(cfg.scene.table.init_state),
            "table_spawn": _config_contract(cfg.scene.table.spawn, exclude=("spawn_path",)),
            "ground_initial_state": _config_contract(cfg.scene.ground.init_state),
            "ground_spawn": _config_contract(cfg.scene.ground.spawn, exclude=("spawn_path",)),
        },
        "actions": {
            "arm": _config_contract(cfg.actions.arm_action),
            "gripper": _config_contract(cfg.actions.gripper_action),
        },
        "simulation": {
            "sim_dt": float(cfg.sim.dt),
            "newton_substeps": int(cfg.sim.physics.num_substeps),
            "collision_decimation": int(cfg.sim.physics.collision_decimation),
            "use_newton_actuators": bool(cfg.sim.use_newton_actuators),
            "use_cuda_graph": bool(cfg.sim.physics.use_cuda_graph),
            "deterministic_mode": cfg.sim.physics.deterministic_mode,
            "gravity": tuple(cfg.sim.gravity),
            "default_shape": _config_contract(cfg.sim.physics.default_shape_cfg),
            "outer_collision": outer_collision,
            "outer_rigid_contacts_per_world": _RIGID_CONTACTS_PER_WORLD,
        },
        "coupler": {
            "iterations": int(coupler.iterations),
            "rigid_entry": {
                "name": rigid_entry.name,
                "bodies": tuple(rigid_entry.bodies),
                "particles": tuple(rigid_entry.particles),
                "all_particles": bool(rigid_entry.all_particles),
                "include_static_shapes": bool(rigid_entry.include_static_shapes),
                "include_child_joints": bool(rigid_entry.include_child_joints),
                "include_body_shapes": bool(rigid_entry.include_body_shapes),
                "shape_label_patterns": tuple(rigid_entry.shape_label_patterns),
                "substeps": int(rigid_entry.substeps),
                "in_place": bool(rigid_entry.in_place),
                "solver": _config_contract(rigid_entry.solver_cfg),
            },
            "rj45_entry": {
                "name": rj45_entry.name,
                "bodies": tuple(rj45_entry.bodies),
                "particles": tuple(rj45_entry.particles),
                "all_particles": bool(rj45_entry.all_particles),
                "include_static_shapes": bool(rj45_entry.include_static_shapes),
                "include_child_joints": bool(rj45_entry.include_child_joints),
                "include_body_shapes": bool(rj45_entry.include_body_shapes),
                "shape_label_patterns": tuple(rj45_entry.shape_label_patterns),
                "substeps": int(rj45_entry.substeps),
                "in_place": bool(rj45_entry.in_place),
                "solver": _config_contract(rj45_entry.solver_cfg),
            },
            "proxy": {
                "source": proxy.source,
                "destination": proxy.destination,
                # Coupler resolution rewrites this field to world-count-dependent
                # body ids, so the validated immutable selectors are recorded.
                "body_patterns": _PROXY_BODY_PATTERNS,
                "particles": tuple(proxy.particles),
                "mode": proxy.mode,
                "mass_scale": float(proxy.mass_scale),
                "collide_interval": proxy.collide_interval,
                "collision_pipeline": (
                    None if proxy.collision_pipeline is None else _config_contract(proxy.collision_pipeline)
                ),
            },
        },
        "validation_geometry": {
            "success_predicate_version": RJ45_SUCCESS_PREDICATE_VERSION,
            "success_axial_tolerance": float(cfg.success_axial_tolerance),
            "success_axial_overtravel_tolerance": float(cfg.success_axial_overtravel_tolerance),
            "success_radial_tolerance": float(cfg.success_radial_tolerance),
            "success_plug_angle_tolerance": float(cfg.success_plug_angle_tolerance),
            "success_latch_angle_tolerance": float(cfg.success_latch_angle_tolerance),
            "success_max_plug_speed": float(cfg.success_max_plug_speed),
            "success_dwell_time_s": float(cfg.success_dwell_time_s),
            "max_tcp_grasp_distance": float(cfg.max_tcp_grasp_distance),
            "max_plug_spatial_speed": float(cfg.max_plug_spatial_speed),
            "task_workspace_lower": tuple(cfg.task_workspace_lower),
            "task_workspace_upper": tuple(cfg.task_workspace_upper),
            "task_body_workspace_lower": tuple(cfg.task_body_workspace_lower),
            "task_body_workspace_upper": tuple(cfg.task_body_workspace_upper),
            "max_task_body_angular_speed": float(cfg.max_task_body_angular_speed),
            "max_task_body_linear_speed": float(cfg.max_task_body_linear_speed),
            "max_cable_goal_offset": float(cfg.max_cable_goal_offset),
        },
        "rj45_physics": rj45_reset_physics_contract(),
    }


def configure_rj45_capacities(cfg: FrankaRJ45InsertionEnvCfg) -> None:
    """Resolve contact storage after command-line world-count overrides."""
    collision_cfg = cfg.sim.physics.collision_cfg
    if not isinstance(collision_cfg, NewtonCollisionPipelineCfg):
        raise TypeError("Franka RJ45 requires an explicit NewtonCollisionPipelineCfg.")
    collision_cfg.rigid_contact_max = _RIGID_CONTACTS_PER_WORLD * int(cfg.scene.num_envs)
