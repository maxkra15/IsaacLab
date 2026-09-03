# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clean single-cable Rizon4s Sharpa XR teleoperation environment."""

from __future__ import annotations

from isaaclab_newton.envs.mdp.actions.newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg
from isaaclab_newton.ik.newton_ik_objectives_cfg import (
    NewtonIKJointLimitObjectiveCfg,
    NewtonIKJointPostureObjectiveCfg,
    NewtonIKPoseObjectiveCfg,
)
from isaaclab_newton.ik.newton_ik_solver_cfg import NewtonIKSolverCfg
from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
    VBDSolverCfg,
)

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from . import RIZON_SHARPA_CABLE_TASK_ID
from .cable_cfg import (
    CABLE_CONNECTOR_RIGID_SPAN_M,
    CABLE_DIAMETER_M,
    CABLE_FLEX_SEGMENT_LENGTH_M,
    CABLE_INITIAL_LATERAL_OFFSET_M,
    CABLE_INITIAL_VERTICAL_SPAN_M,
    CABLE_LENGTH_M,
    CABLE_SEGMENT_COUNT,
    HangingRj45CableSpawnerCfg,
    RizonSharpaCableObjectCfg,
    hanging_cable_positions,
)
from .observations import cable_segment_positions, connector_pose, insertion_socket_pose
from .robot_asset import (
    RIZON_SHARPA_ARM_HOME_RAD,
    RIZON_SHARPA_ARM_JOINT_NAMES,
    RIZON_SHARPA_BASE_LINK_NAME,
    RIZON_SHARPA_END_EFFECTOR_BODY_NAME,
    RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES,
    RIZON_SHARPA_RIGHT_HAND_LIMITS_RAD,
    RIZON_SHARPA_RIGHT_HAND_OPEN_RAD,
    default_rizon_sharpa_bundle_root,
    rizon_sharpa_asset_contract,
)
from .teleop import SHARPA_THUMB_RETARGETING_GAINS, build_rizon_sharpa_teleop_pipeline

# Marker consumed by task tests and the teleoperation launcher.
_TELEOP_AVAILABLE = True

ROBOT_BASE_POSITION_E = (0.0, -0.32, 0.50)
ROBOT_BASE_ROTATION_XYZW = (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)
STAND_CENTER_E = (0.0, -0.32, 0.25)
STAND_SIZE_M = (0.54, 0.62, 0.50)
CABLE_FREE_END_POSITION_E = (0.48, -0.16, 1.60)
CABLE_ANCHOR_POSITION_E = (
    CABLE_FREE_END_POSITION_E[0] + CABLE_INITIAL_LATERAL_OFFSET_M,
    CABLE_FREE_END_POSITION_E[1],
    CABLE_FREE_END_POSITION_E[2] + CABLE_INITIAL_VERTICAL_SPAN_M,
)
INSERTION_SOCKET_POSITION_E = (0.48, 0.02, 1.60)
INSERTION_SOCKET_ROTATION_XYZW = (0.0, 0.0, 0.0, 1.0)
# X/Y align a neutral tracked right hand with the simulated home workspace.
# CloudXR's floor-relative height already carries the operator's real height;
# keeping the anchor at world z=0 avoids the former 26.6 cm downward bias in
# both the headset camera and hand-debug spheres.
XR_ANCHOR_POSITION_E = (-0.137, -0.083, 0.0)
XR_ANCHOR_ROTATION_XYZW = (0.0, 0.0, 0.0, 1.0)
CAMERA_EYE_E = (1.35, -3.10, 2.30)
CAMERA_LOOKAT_E = (0.28, -0.04, 1.50)

_ROBOT_ENTRY = "rizon_sharpa"
_CABLE_ENTRY = "hanging_cable"
_ROBOT_BODY_PATTERN = r"/World/envs/env_[^/]+/Robot"
_CABLE_BODY_PATTERN = r"/World/envs/env_[^/]+/Cable/.*"
_SOCKET_BODY_PATTERN = r"/World/envs/env_[^/]+/InsertionTarget"
_HAND_PROXY_PATTERN = r"/World/envs/env_[^/]+/Robot/right_(?:hand_C_MC|thumb_.*|index_.*|middle_.*|ring_.*|pinky_.*)"
_STAND_SHAPE_PATTERN = r".*/Stand.*"
_HAND_BODY_LABEL_TOKENS = tuple(
    f"/Robot/{name}"
    for name in (
        "right_hand_C_MC",
        "right_thumb_",
        "right_index_",
        "right_middle_",
        "right_ring_",
        "right_pinky_",
    )
)
_GRIP_SHAPE_LABEL_TOKENS = ("/connector_grip", "/geometry/mesh_edge_capsule_")
_CONTACT_CAPACITY = 16_384
# The hand is a one-way teleoperation source: cable impulses never feed back
# into MJWarp. High-authority thumb gains therefore remove visible servo lag
# without allowing cable contacts to destabilize the tracked hand.
_THUMB_EFFORT_LIMIT_N_M = 100.0
_THUMB_STIFFNESS_N_M_RAD = 500.0
_THUMB_DAMPING_N_M_S_RAD = 100.0
_FINGER_EFFORT_LIMIT_N_M = 3.3
_FINGER_STIFFNESS_N_M_RAD = 24.0
_FINGER_DAMPING_N_M_S_RAD = 1.2


def _make_hand_cable_collision_pipeline(model):
    """Keep destination contact to the complete right hand against the cable."""
    import numpy as np
    import warp as wp
    from newton import CollisionPipeline, ShapeFlags

    hand_bodies = {
        body_id
        for body_id, label in enumerate(model.body_label)
        if any(token in str(label) for token in _HAND_BODY_LABEL_TOKENS)
    }
    if not hand_bodies:
        raise RuntimeError("Sharpa cable coupling found no mirrored right-hand bodies.")
    palm_bodies = {body_id for body_id, label in enumerate(model.body_label) if "/Robot/right_hand_C_MC" in str(label)}
    if not palm_bodies:
        raise RuntimeError("Sharpa cable coupling requires the native right-palm body.")
    shape_body = model.shape_body.numpy()
    flags = model.shape_flags.numpy().copy()
    collide_shapes = int(ShapeFlags.COLLIDE_SHAPES)
    collide_particles = int(ShapeFlags.COLLIDE_PARTICLES)
    hand_shapes = {
        shape_id
        for shape_id, body_id in enumerate(shape_body)
        if int(body_id) in hand_bodies and int(flags[shape_id]) & collide_shapes
    }
    if not hand_shapes:
        raise RuntimeError("Sharpa cable coupling found no collidable right-hand shapes.")
    palm_shapes = {
        shape_id
        for shape_id, body_id in enumerate(shape_body)
        if int(body_id) in palm_bodies and int(flags[shape_id]) & collide_shapes
    }
    if not palm_shapes:
        raise RuntimeError("Sharpa cable coupling requires a collidable native right-palm shape.")
    for shape_id in hand_shapes:
        flags[shape_id] &= ~collide_particles
    model.shape_flags = wp.array(flags, dtype=wp.int32, device=model.device)

    labels = [str(label) for label in model.shape_label]
    cable_shapes = {
        shape_id for shape_id, label in enumerate(labels) if any(token in label for token in _GRIP_SHAPE_LABEL_TOKENS)
    }
    if not cable_shapes:
        raise RuntimeError("Sharpa cable coupling found no connector or cable collision shapes.")
    base_pairs = getattr(model, "shape_contact_pairs", None)
    if base_pairs is None:
        raise RuntimeError("Sharpa cable coupling requires explicit shape_contact_pairs.")
    kept: list[tuple[int, int]] = []
    for shape_a, shape_b in base_pairs.numpy().reshape(-1, 2):
        shape_a, shape_b = int(shape_a), int(shape_b)
        a_hand = shape_a in hand_shapes
        b_hand = shape_b in hand_shapes
        if a_hand or b_hand:
            other = shape_b if a_hand else shape_a
            if a_hand == b_hand or other not in cable_shapes:
                continue
        kept.append((shape_a, shape_b))
    if not any((a in hand_shapes) != (b in hand_shapes) for a, b in kept):
        raise RuntimeError("Sharpa cable coupling retained no hand-to-cable contact pairs.")
    return CollisionPipeline(
        model,
        broad_phase="explicit",
        shape_pairs_filtered=wp.array(
            np.asarray(kept, dtype=np.int32).reshape(-1, 2),
            dtype=wp.vec2i,
            device=model.device,
        ),
        rigid_contact_max=_CONTACT_CAPACITY,
        contact_matching="latest",
        contact_matching_pos_threshold=0.005,
        contact_matching_normal_dot_threshold=0.95,
    )


def _joint_command(names: tuple[str, ...], values: tuple[float, ...]) -> dict[str, float]:
    return dict(zip(names, values, strict=True))


def rizon_sharpa_articulation_cfg() -> ArticulationCfg:
    """Return the fixed-base arm and physical 22-DoF right hand."""
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(default_rizon_sharpa_bundle_root() / "rizon4s_sharpa_no_spheres_generated.usd"),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_BASE_POSITION_E,
            rot=ROBOT_BASE_ROTATION_XYZW,
            joint_pos={
                **_joint_command(RIZON_SHARPA_ARM_JOINT_NAMES, RIZON_SHARPA_ARM_HOME_RAD),
                **_joint_command(RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES, RIZON_SHARPA_RIGHT_HAND_OPEN_RAD),
            },
        ),
        actuators={
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-2]"],
                joint_effort_limit=123.0,
                joint_velocity_limit=2.094,
                stiffness=6000.0,
                damping=108.5,
            ),
            "elbow": ImplicitActuatorCfg(
                joint_names_expr=["joint[3-4]"],
                joint_effort_limit=64.0,
                joint_velocity_limit=2.443,
                stiffness=4200.0,
                damping=90.7,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["joint[5-7]"],
                joint_effort_limit=39.0,
                joint_velocity_limit=4.887,
                stiffness=1500.0,
                damping=54.2,
            ),
            "right_thumb": ImplicitActuatorCfg(
                joint_names_expr=["right_thumb_.*"],
                joint_effort_limit=_THUMB_EFFORT_LIMIT_N_M,
                joint_velocity_limit=6.0,
                stiffness=_THUMB_STIFFNESS_N_M_RAD,
                damping=_THUMB_DAMPING_N_M_S_RAD,
            ),
            "right_fingers": ImplicitActuatorCfg(
                joint_names_expr=["right_(?:index|middle|ring|pinky)_.*"],
                joint_effort_limit=_FINGER_EFFORT_LIMIT_N_M,
                joint_velocity_limit=6.0,
                stiffness=_FINGER_STIFFNESS_N_M_RAD,
                damping=_FINGER_DAMPING_N_M_S_RAD,
            ),
        },
        soft_joint_pos_limit_factor=0.95,
    )


@configclass
class RizonSharpaCableSceneCfg(InteractiveSceneCfg):
    """One robot, one pedestal, and one top-anchored half-meter cable."""

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(color=(0.93, 0.94, 0.96)),
    )
    stand = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Stand",
        init_state=AssetBaseCfg.InitialStateCfg(pos=STAND_CENTER_E),
        spawn=sim_utils.CuboidCfg(
            size=STAND_SIZE_M,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.12, 0.14, 0.17),
                roughness=0.22,
                metallic=0.55,
            ),
        ),
    )
    cable_anchor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CableAnchor",
        init_state=AssetBaseCfg.InitialStateCfg(pos=CABLE_ANCHOR_POSITION_E),
        spawn=sim_utils.CuboidCfg(
            size=(0.10, 0.10, 0.06),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.08, 0.10, 0.13),
                roughness=0.20,
                metallic=0.65,
            ),
        ),
    )
    robot = rizon_sharpa_articulation_cfg()
    cable = RizonSharpaCableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cable",
        spawn=HangingRj45CableSpawnerCfg(
            positions=hanging_cable_positions(),
            insertion_target_position_e=INSERTION_SOCKET_POSITION_E,
            insertion_target_rotation_xyzw=INSERTION_SOCKET_ROTATION_XYZW,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.018, 0.075, 0.18),
                roughness=0.30,
            ),
            physics_material=sim_utils.CableMaterialCfg(
                thickness=CABLE_DIAMETER_M,
                density=1100.0,
                stretch_stiffness=1.0e7,
                bend_stiffness=3.0e5,
                shear_stiffness=1.0e7,
                twist_stiffness=2.0e5,
            ),
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
        ),
        init_state=RizonSharpaCableObjectCfg.InitialStateCfg(pos=CABLE_FREE_END_POSITION_E),
        tail_anchor_prim_path="/World/envs/env_.*/CableAnchor",
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(color=(1.0, 0.96, 0.92), intensity=2600.0, angle=0.45),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.25, -0.34, -0.12, 0.90)),
    )
    fill_light = AssetBaseCfg(
        prim_path="/World/FillLight",
        spawn=sim_utils.DomeLightCfg(color=(0.82, 0.90, 1.0), intensity=700.0),
    )


@configclass
class RizonSharpaCableActionsCfg:
    """Absolute canonical-palm IK target followed by 22 finger-joint targets."""

    right_palm = NewtonInverseKinematicsActionCfg(
        class_type="isaaclab_tasks.contrib.rizon_sharpa_cable.actions:RizonSharpaTeleopNewtonIkAction",
        asset_name="robot",
        joint_names=list(RIZON_SHARPA_ARM_JOINT_NAMES),
        controller=NewtonIKSolverCfg(
            optimizer="lm",
            jacobian_mode="analytic",
            iterations=4,
            lambda_initial=0.1,
        ),
        objectives=[
            NewtonIKPoseObjectiveCfg(
                name="right_palm",
                body_name=RIZON_SHARPA_END_EFFECTOR_BODY_NAME,
                body_offset_pos=(0.0, 0.0, 0.0),
                body_offset_rot=(0.0, 0.0, 0.0, 1.0),
                command_type="pose",
                use_relative_mode=False,
                position_weight=1.0,
                rotation_weight=1.0,
            ),
            # The seven-axis arm has one wrist-pose null-space DoF. Match the
            # Waterhose-v2 recipe with a weak proximal posture preference so a
            # stationary tracked hand cannot make the elbow turn continuously.
            NewtonIKJointPostureObjectiveCfg(
                joint_names=list(RIZON_SHARPA_ARM_JOINT_NAMES[:4]),
                target_positions=RIZON_SHARPA_ARM_HOME_RAD[:4],
                weight=0.01,
            ),
            NewtonIKJointLimitObjectiveCfg(weight=0.1),
        ],
    )
    right_hand = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES),
        preserve_order=True,
        use_default_offset=False,
        scale=1.0,
        offset=0.0,
        clip=dict(zip(RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES, RIZON_SHARPA_RIGHT_HAND_LIMITS_RAD, strict=True)),
    )


@configclass
class RizonSharpaCableObservationsCfg:
    """Robot, connector, and full cable state for recording and diagnostics."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        right_palm_pose = ObsTerm(
            func=mdp.body_pose_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[RIZON_SHARPA_END_EFFECTOR_BODY_NAME])},
        )
        connector_pose = ObsTerm(func=connector_pose)
        insertion_socket_pose = ObsTerm(func=insertion_socket_pose)
        cable_segment_positions = ObsTerm(func=cable_segment_positions)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class RizonSharpaCableEventsCfg:
    """Restore robot, actuator targets, cable pose history, and velocities exactly."""

    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


@configclass
class RizonSharpaCableTerminationsCfg:
    """Long fallback timeout; the teleoperation runner removes it."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class RizonSharpaCableEnvCfg(ManagerBasedRLEnvCfg):
    """Standalone right-hand XR teleoperation of a single hanging RJ45 cable."""

    scene: RizonSharpaCableSceneCfg = RizonSharpaCableSceneCfg(
        num_envs=1,
        env_spacing=3.0,
        replicate_physics=True,
    )
    actions: RizonSharpaCableActionsCfg = RizonSharpaCableActionsCfg()
    observations: RizonSharpaCableObservationsCfg = RizonSharpaCableObservationsCfg()
    events: RizonSharpaCableEventsCfg = RizonSharpaCableEventsCfg()
    terminations: RizonSharpaCableTerminationsCfg = RizonSharpaCableTerminationsCfg()
    commands = None
    rewards = None
    curriculum = None
    xr: XrCfg = XrCfg(
        anchor_pos=XR_ANCHOR_POSITION_E,
        anchor_rot=XR_ANCHOR_ROTATION_XYZW,
        near_plane=0.05,
    )

    def __post_init__(self) -> None:
        """Configure coupled physics, Newton IK, and paused AVP teleoperation."""
        try:
            from isaaclab_teleop import IsaacTeleopCfg
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Rizon Sharpa XR teleoperation requires Isaac Lab's optional 'teleop' extra."
            ) from error

        self.decimation = 1
        self.episode_length_s = 3600.0
        # Keep the simulation clock aligned with the measured interactive
        # control budget. A 90 Hz clock made a healthy ~55 Hz XR loop appear
        # to run in slow motion even though every tracked pose was consumed.
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.81)
        self.sim.use_newton_actuators = True
        self.sim.physics = NewtonCfg(
            # This interactive one-world task builds its isolated IK model after
            # physics starts. Keep the coupled solver eager so that model setup,
            # operator pause/resume, and reset remain independent of graph state.
            use_cuda_graph=False,
            solver_cfg=CouplerProxyCfg(
                entries=[
                    CouplerEntryCfg(
                        name=_ROBOT_ENTRY,
                        solver_cfg=MJWarpSolverCfg(
                            use_mujoco_contacts=False,
                            cone="elliptic",
                            ls_iterations=10,
                            integrator="implicitfast",
                            njmax=4096,
                            nconmax=4096,
                        ),
                        bodies=[_ROBOT_BODY_PATTERN],
                        include_body_shapes=True,
                        shape_label_patterns=[_STAND_SHAPE_PATTERN],
                    ),
                    CouplerEntryCfg(
                        name=_CABLE_ENTRY,
                        solver_cfg=VBDSolverCfg(
                            iterations=10,
                            rigid_contact_hard=True,
                            rigid_contact_k_start=1.0e3,
                            rigid_body_contact_buffer_size=4096,
                        ),
                        bodies=[_CABLE_BODY_PATTERN, _SOCKET_BODY_PATTERN],
                        include_body_shapes=True,
                        include_static_shapes=False,
                    ),
                ],
                proxies=[
                    CouplerProxyMappingCfg(
                        source=_ROBOT_ENTRY,
                        destination=_CABLE_ENTRY,
                        bodies=[_HAND_PROXY_PATTERN],
                        mode="staggered",
                        proxy_relaxation=0.0,
                        mass_scale=1_000.0,
                        collide_interval=1,
                        collision_pipeline=_make_hand_cable_collision_pipeline,
                    )
                ],
                iterations=1,
            ),
            num_substeps=1,
            collision_decimation=0,
            collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=_CONTACT_CAPACITY),
            default_shape_cfg=NewtonShapeCfg(ke=1.0e4, kd=0.1, mu=0.8, gap=0.004),
        )

        from isaaclab_visualizers.kit import KitVisualizerCfg

        self.sim.default_visualizer_cfg = KitVisualizerCfg(
            eye=CAMERA_EYE_E,
            lookat=CAMERA_LOOKAT_E,
            focal_length=22.0,
            origin_type="env",
            origin_env_index=0,
        )
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=build_rizon_sharpa_teleop_pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,
            app_name="RizonSharpaCableTeleop",
            target_frame_prim_path=f"/World/envs/env_0/Robot/{RIZON_SHARPA_BASE_LINK_NAME}",
            teleoperation_active_default=False,
        )

    def play_mode(self) -> None:
        """Keep the standalone scene deterministic and single-world."""
        self.scene.num_envs = 1
        self.terminations.time_out = None


def rizon_sharpa_cable_contract() -> dict[str, object]:
    """Return the stable task topology and control semantics."""
    from .sharpa_hand_retargeting import (
        SHARPA_OPENXR_TO_CANONICAL_PALM_RPY_DEG,
        sharpa_hand_retargeting_contract,
    )
    from .showroom import rizon_sharpa_showroom_contract

    return {
        "task_id": RIZON_SHARPA_CABLE_TASK_ID,
        "scene": {
            "robot": "rizon4s-sharpa-right-hand",
            "cable_count": 1,
            "cable_length_m": CABLE_LENGTH_M,
            "cable_segment_count": CABLE_SEGMENT_COUNT,
            "connector_rigid_span_m": CABLE_CONNECTOR_RIGID_SPAN_M,
            "flex_segment_length_m": CABLE_FLEX_SEGMENT_LENGTH_M,
            "free_end_position_e_m": CABLE_FREE_END_POSITION_E,
            "anchor_position_e_m": CABLE_ANCHOR_POSITION_E,
            "floating_socket_position_e_m": INSERTION_SOCKET_POSITION_E,
            "floating_socket_rotation_xyzw": INSERTION_SOCKET_ROTATION_XYZW,
            "props": ("glossy-white-ground", "single-pedestal", "small-cable-anchor", "white-backwall"),
            "gb300": rizon_sharpa_showroom_contract(),
            "franka": "absent",
        },
        "physics": {
            "solver": "coupled-proxy",
            "execution": "eager-interactive",
            "robot_entry": "MJWarp",
            "cable_entry": "VBD-native-cable",
            "proxy": "complete-right-hand-to-one-plug-and-one-cable",
            "mode": "staggered-one-way-source-to-cable",
            "proxy_feedback_relaxation": 0.0,
            "proxy_mass_scale": 1_000.0,
            "connector_attachment": "same-body-rigid-strain-relief;first-flex-joint-behind-plug",
            "insertion_geometry": "exact-floating-plug-and-socket-narrow-band-sdf-pair",
            "control_frequency_hz": 60,
            "substeps": 1,
            "vbd_iterations": 10,
        },
        "control": {
            "arm": "newton-ik-absolute-canonical-palm-pose",
            "ik_optimizer": "lm",
            "ik_jacobian": "analytic",
            "ik_iterations": 4,
            "pose_tracking": "absolute-position-and-orientation; dropout-hold",
            "orientation": "openxr-wrist-to-canonical-r-palm-ctrl-absolute",
            "palm_control_frame": RIZON_SHARPA_END_EFFECTOR_BODY_NAME,
            "tracker_offsets_rpy_deg": SHARPA_OPENXR_TO_CANONICAL_PALM_RPY_DEG,
            "hand": "nvidia-isaacteleop-dexpilot-openxr-to-22-independent-joints",
            "thumb_retargeting_gains": dict(SHARPA_THUMB_RETARGETING_GAINS),
            "thumb_actuator": {
                "effort_limit_n_m": _THUMB_EFFORT_LIMIT_N_M,
                "stiffness_n_m_rad": _THUMB_STIFFNESS_N_M_RAD,
                "damping_n_m_s_rad": _THUMB_DAMPING_N_M_S_RAD,
            },
            "hand_retargeting": sharpa_hand_retargeting_contract(),
            "operator_hand": "right",
            "action_dim": 29,
            "autostart": False,
        },
        "reset": "scene-default-robot-targets-cable-poses-and-velocities",
        "robot_asset": rizon_sharpa_asset_contract(),
    }


__all__ = [
    "CABLE_ANCHOR_POSITION_E",
    "CABLE_FREE_END_POSITION_E",
    "CAMERA_EYE_E",
    "CAMERA_LOOKAT_E",
    "INSERTION_SOCKET_POSITION_E",
    "INSERTION_SOCKET_ROTATION_XYZW",
    "ROBOT_BASE_POSITION_E",
    "ROBOT_BASE_ROTATION_XYZW",
    "RizonSharpaCableActionsCfg",
    "RizonSharpaCableEnvCfg",
    "RizonSharpaCableEventsCfg",
    "RizonSharpaCableObservationsCfg",
    "RizonSharpaCableSceneCfg",
    "rizon_sharpa_articulation_cfg",
    "rizon_sharpa_cable_contract",
]
