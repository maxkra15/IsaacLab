# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waterhose manipulation environment.

MDP config groups are copied from the Franka cable-plug task
(:mod:`isaaclab_tasks.core.lift_franka_soft`), whose
``mdp`` functions are reused by import.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import MISSING

from isaaclab_newton.physics import (
    AdmmContactPairCfg,
    AdmmCouplingCfg,
    CoupledProxyCfg,
    CoupledSolverCfg,
    CoupledSolverEntryCfg,
    MJWarpSolverCfg,
    NewtonCollisionPipelineCfg,
    ProxyCouplingCfg,
)
from isaaclab_newton.envs.mdp.actions.newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg
from isaaclab_newton.ik.newton_ik_manager_cfg import NewtonIKManagerCfg
from isaaclab_newton.sim.spawners.materials.physics_materials_cfg import NewtonCableMaterialCfg
from isaaclab_visualizers.kit.kit_visualizer_cfg import KitVisualizerCfg
from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.sim import schemas as sim_schemas
from isaaclab.sim.schemas.schemas_cfg import MeshCollisionBaseCfg
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.utils import clone
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers.manager_term_cfg import ActionTermCfg
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
from isaaclab_contrib.cable.cable_object_cfg import CableAttachmentCfg, CableObjectCfg
from isaaclab_contrib.deformable.newton_manager_cfg import (
    CoupledNewtonCfg,
    NewtonModelCfg,
    VBDSolverCfg,
)

from .teleop import WaterhoseSpaceMouseCfg

try:
    import isaacteleop  # noqa: F401 -- IsaacTeleop pipeline builders need this at runtime.
    from isaaclab_teleop import IsaacTeleopCfg, XrCfg

    _TELEOP_AVAILABLE = True
except ImportError:
    _TELEOP_AVAILABLE = False
    logging.getLogger(__name__).warning("isaaclab_teleop is not installed. XR teleoperation is disabled.")

WATERHOSE_ASSETS_DIR = os.environ.get(
    "WATERHOSE_ASSETS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets"),
)

# rby1df robot: URDF converted to USD (scripts/tools/convert_urdf.py) then flattened
# into a single self-contained asset.
_RBY1_USD = os.path.join(WATERHOSE_ASSETS_DIR, "rby1df", "rby1df.usda")
# EE contact frame offset from right_gripper_base. The finger pad spans base-z [-0.0735, -0.1355]
# (base..tip); -0.125 grips in the TIP third of the pad (not the pad centre at -0.1045), so the
# flat fingertip surface closes on the plug. The plug's long axis lies along the finger width
# (gripper-Y, 32 mm pad) so the pad spans the 22 mm plug length once centered.
_RIGHT_GRIPPER_EE_FRAME_POS = (0.0, 0.0, -0.125)
# USD stores xformOp:orient as (w, x, y, z); IsaacLab action offsets use (x, y, z, w).
_RIGHT_GRIPPER_EE_FRAME_ROT = (0.70710677, 0.70710677, 0.0, 0.0)

# add_rod_graph places each segment's body frame at the edge's start node u
# (edge (u, v), +Z from u->v), so cable_local_pos=(0, 0, 0) welds at u and the head
# plug weld's local offset is authored against segment 0's start frame. cable001's
# last segment is edge (42, 43) -> u=42, so the tail weld pins node 42.
_FRIDGE_POS = (0.0, 0.0, 0.5)
_CABLE1_TAIL_NODE_42 = (-0.18810473382472992, 0.3453156650066376, -0.25986239314079285)
_CABLE1_ANCHOR_NODE = _CABLE1_TAIL_NODE_42
# World position of the cable tail node = the per-env kinematic anchor body. The cable
# welds to this per-env body rather than the shared static world body (-1): a fixed joint
# to the global world body corrupts the multi-env coupled MJWarp+VBD solve (robot joints
# go NaN at step 0).
_ANCHOR_POS = tuple(p + n for p, n in zip(_FRIDGE_POS, _CABLE1_ANCHOR_NODE))

# Hand-authored initial visualizer cameras. These used to live in
# scripts/environments/waterhose/run_robot_demo.py; keep them in the task config
# so IsaacLab's official --visualizer/--viz path uses the same starting views.
# KitVisualizer uses eye/lookat; this lookat is one meter along camera-local -Z
# from the authored Kit transform translate=(-0.9, 0.6, 0.3), rotateXYZ=(73.32259, 0, -112.30437).
_KIT_CAMERA_EYE = (-0.9, 0.6, 0.3)
_KIT_CAMERA_LOOKAT = (-0.013736291, 0.236437794, 0.013017143)
_NEWTON_CAMERA_EYE = (-2.55, -7.1, 2.3)
_NEWTON_CAMERA_LOOKAT = (0.55, -0.42, 0.9)
_ROBOT_BASE_PRIM_PATH_ENV0 = "/World/envs/env_0/Robot/origin"


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}

_RIGHT_GRIPPER_FINGER_BODY_TOKENS = (
    "right_gripper_leftfinger",
    "right_gripper_rightfinger",
    "right_gripper_left_finger",
    "right_gripper_right_finger",
)
_RIGHT_GRIPPER_ACTUAL_MESH_COLLISION = MeshCollisionBaseCfg(mesh_approximation_name="none")
_RIGHT_GRIPPER_SDF_COLLISION = sim_utils.NewtonSDFCollisionPropertiesCfg(
    sdf_max_resolution=64,
    sdf_narrow_band_inner=0.002,
    sdf_narrow_band_outer=0.006,
    sdf_texture_format="float32",
    sdf_padding=0.001,
    hydroelastic_enabled=False,
)

# ----- Insertion socket (hollow-cylinder collider the plug connector inserts into) -----
# Single source of truth for the socket mouth pose (env-local; env_origins added at runtime).
# The socket is deliberately placed/oriented to MATCH the direction the grasped plug's connector
# naturally presents after the settle phase (measured from the scripted demo), so insertion is a
# short straight push along the connector axis instead of an infeasible ~106 deg arm reorientation.
# The bore axis is the connector axis (0.8613, -0.3011, -0.4092); the mouth sits a standoff ahead
# of the post-settle connector tip. The scripted state machine (scripted_state_machine.py) uses the
# SAME pose -- keep them in sync. NOTE: this trades the exact fridge-socket location for a working
# insert/extract; re-measure if the grasp/settle motion changes.
_SOCKET_MOUTH_POS = (-0.259345, 0.344709, 0.28698)
# init_state.rot (w, x, y, z): 20 deg about +X, so the authored bore axis +Z maps onto the real
# fridge-socket hole axis (0, -sin20, cos20). This is the visually-correct fridge socket location.
_SOCKET_ROT = (0.984808, 0.173648, 0.0, 0.0)
_SOCKET_USD = os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "socket_collision.usda")
# SDF guides the plug into the bore. Higher resolution than the gripper because the 3 mm bore
# wall is a thin feature. Enable hydroelastic too when WATERHOSE_SOCKET_HYDRO is set.
_SOCKET_SDF_COLLISION = sim_utils.NewtonSDFCollisionPropertiesCfg(
    sdf_max_resolution=128,
    sdf_narrow_band_inner=0.004,
    sdf_narrow_band_outer=0.006,
    sdf_texture_format="float32",
    sdf_padding=0.001,
    hydroelastic_enabled=_env_flag("WATERHOSE_SOCKET_HYDRO", False),
    hydroelastic_stiffness=1.0e7,
)


@clone
def spawn_socket_collider(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the hollow-cylinder socket and (optionally) give its mesh Newton SDF collision."""
    from pxr import Usd, UsdPhysics

    prim = spawn_from_usd.__wrapped__(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    # SDF is OFF by default: on the current Newton build, plug(plain mesh) vs socket(mesh+texture
    # SDF) routes through the texture-SDF mesh-mesh narrow-phase kernel that raises a CUDA illegal
    # memory access the instant the plug nears the socket (the same bug that keeps the gripper-finger
    # SDF opt-in). Empirically, the plain-mesh socket uses the BVH mesh-mesh fallback and the plug
    # inserts/holds/extracts by friction without crashing. Re-enable WATERHOSE_SOCKET_SDF (and/or
    # WATERHOSE_SOCKET_HYDRO, which also needs the plug made hydroelastic) once that kernel is fixed.
    if not _env_flag("WATERHOSE_SOCKET_SDF", False):
        return prim
    stage = prim.GetStage()
    for child in Usd.PrimRange(prim):
        if child.GetTypeName() == "Mesh" and child.HasAPI(UsdPhysics.CollisionAPI):
            sim_schemas.modify_collision_properties(child.GetPath().pathString, _SOCKET_SDF_COLLISION, stage=stage)
    return prim


@configclass
class WaterhoseGripperPositionActionCfg(ActionTermCfg):
    """One-dimensional continuous position command for the RBY1 right gripper."""

    class_type: str = "isaaclab_tasks.contrib.waterhose.mdp.actions:WaterhoseGripperPositionAction"

    joint_names: list[str] = MISSING
    """Right gripper driver and finger joints to command explicitly."""

    open_command_expr: dict[str, float] = MISSING
    """Joint position targets for a normalized action of ``+1``."""

    close_command_expr: dict[str, float] = MISSING
    """Joint position targets for a normalized action of ``-1``."""


def _is_right_gripper_finger_collision_instance(prim) -> bool:
    """Return true for right gripper finger collision instances in the rby1df USD."""
    if not prim.IsInstance():
        return False
    path = prim.GetPath().pathString.lower()
    if not any(token in path for token in _RIGHT_GRIPPER_FINGER_BODY_TOKENS):
        return False
    name = prim.GetName().lower()
    parent_name = prim.GetParent().GetName().lower()
    return name.endswith("_collision") or (name == parent_name and name.endswith("finger"))


def _apply_collision_overrides_to_right_gripper_fingers(robot_prim) -> None:
    """Use actual mesh collision and optional Newton SDF only for the right gripper finger meshes."""
    from pxr import Usd, UsdPhysics

    stage = robot_prim.GetStage()
    # Newton's current texture-SDF mesh contact path hits a CUDA illegal access
    # when the plug reaches these proxy finger meshes. Keep SDF opt-in until
    # that solver path is fixed; actual mesh collision stays enabled either way.
    enable_sdf = _env_flag("WATERHOSE_RIGHT_GRIPPER_SDF", False)
    collision_instance_paths = [
        prim.GetPath().pathString
        for prim in Usd.PrimRange(robot_prim)
        if _is_right_gripper_finger_collision_instance(prim)
    ]

    modified_meshes: list[str] = []
    for instance_path in collision_instance_paths:
        instance_prim = stage.GetPrimAtPath(instance_path)
        instance_prim.SetInstanceable(False)
        for child_prim in Usd.PrimRange(instance_prim):
            if child_prim.GetTypeName() != "Mesh" or not child_prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            mesh_path = child_prim.GetPath().pathString
            sim_schemas.modify_mesh_collision_properties(mesh_path, _RIGHT_GRIPPER_ACTUAL_MESH_COLLISION, stage=stage)
            if enable_sdf:
                sim_schemas.modify_collision_properties(mesh_path, _RIGHT_GRIPPER_SDF_COLLISION, stage=stage)
            modified_meshes.append(mesh_path)

    if not modified_meshes:
        logging.warning("Did not find right gripper finger collision meshes to override.")


@clone
def spawn_rby1df_with_right_gripper_finger_collision_overrides(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn rby1df and override only the right gripper finger collision meshes."""
    prim = spawn_from_usd.__wrapped__(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    _apply_collision_overrides_to_right_gripper_fingers(prim)
    return prim


def _build_waterhose_teleop_pipeline():
    """Build the IsaacTeleop pipeline for the absolute Waterhose IK action space."""

    from isaacteleop.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))

    se3_cfg = Se3RetargeterConfig(
        input_device=ControllersSource.RIGHT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=False,
        use_wrist_position=False,
        target_offset_roll=90.0,
        target_offset_pitch=0.0,
        target_offset_yaw=0.0,
    )
    se3 = Se3AbsRetargeter(se3_cfg, name="ee_pose")
    connected_se3 = se3.connect(
        {
            ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
        }
    )

    gripper_cfg = GripperRetargeterConfig(hand_side="right")
    gripper = GripperRetargeter(gripper_cfg, name="gripper")
    connected_gripper = gripper.connect(
        {
            ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
            HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
        }
    )

    ee_pose_elements = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
    gripper_elements = ["gripper_value"]
    reorderer = TensorReorderer(
        input_config={
            "ee_pose": ee_pose_elements,
            "gripper_command": gripper_elements,
        },
        output_order=ee_pose_elements + gripper_elements,
        name="action_reorderer",
        input_types={"ee_pose": "array", "gripper_command": "scalar"},
    )
    connected_reorderer = reorderer.connect(
        {
            "ee_pose": connected_se3.output("ee_pose"),
            "gripper_command": connected_gripper.output("gripper_command"),
        }
    )

    return OutputCombiner({"action": connected_reorderer.output("output")})


def _build_waterhose_relative_teleop_pipeline():
    """Build the IsaacTeleop pipeline for the relative Waterhose IK teleop action space."""

    import numpy as np

    from isaacteleop.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        Se3RelRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import BaseRetargeter, OutputCombiner, TensorGroupType, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import DLDataType, NDArrayType, TransformMatrix

    class WaterhoseDeltaFrameRemapper(BaseRetargeter):
        """Rotate AVP hand deltas into the waterhose robot's relative IK frame."""

        def input_spec(self):
            return {
                "ee_delta": TensorGroupType(
                    "ee_delta",
                    [NDArrayType("delta", shape=(6,), dtype=DLDataType.FLOAT, dtype_bits=32)],
                )
            }

        def output_spec(self):
            return {
                "ee_delta": TensorGroupType(
                    "ee_delta",
                    [NDArrayType("delta", shape=(6,), dtype=DLDataType.FLOAT, dtype_bits=32)],
                )
            }

        def _compute_fn(self, inputs, outputs, context) -> None:
            delta = np.asarray(inputs["ee_delta"][0], dtype=np.float32).flatten()
            remapped = delta.copy()
            remapped[0] = -delta[1]
            remapped[1] = delta[0]
            remapped[3] = delta[3]
            remapped[4] = 0.0
            remapped[5] = 0.0
            outputs["ee_delta"][0] = remapped

    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")

    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_hands = hands.transformed(transform_input.output(ValueInput.VALUE))

    se3_cfg = Se3RetargeterConfig(
        input_device=HandsSource.RIGHT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=True,
        delta_pos_scale_factor=15.0,
        delta_rot_scale_factor=2.0,
        alpha_pos=0.5,
        alpha_rot=0.5,
    )
    se3 = Se3RelRetargeter(se3_cfg, name="ee_delta")
    connected_se3 = se3.connect({HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT)})
    delta_remapper = WaterhoseDeltaFrameRemapper(name="waterhose_delta_frame")
    connected_delta = delta_remapper.connect({"ee_delta": connected_se3.output("ee_delta")})

    gripper_cfg = GripperRetargeterConfig(hand_side="right")
    gripper = GripperRetargeter(gripper_cfg, name="gripper")
    connected_gripper = gripper.connect(
        {
            "hand_right": hands.output(HandsSource.RIGHT),
            "controller_right": controllers.output(ControllersSource.RIGHT),
        }
    )

    delta_elements = ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]
    gripper_elements = ["gripper"]
    reorderer = TensorReorderer(
        input_config={
            "ee_delta": delta_elements,
            "gripper": gripper_elements,
        },
        output_order=delta_elements + gripper_elements,
        name="action_reorderer",
        input_types={
            "ee_delta": "array",
            "gripper": "scalar",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "ee_delta": connected_delta.output("ee_delta"),
            "gripper": connected_gripper.output("gripper_command"),
        }
    )

    return OutputCombiner({"action": connected_reorderer.output("output")})


def _make_proxy_collision_pipeline(model):
    """Build the destination-view collision pipeline used by Newton proxy coupling."""
    from newton import CollisionPipeline

    return CollisionPipeline(
        model,
        broad_phase="explicit",
        rigid_contact_max=30000,
        contact_matching="sticky",
        contact_matching_pos_threshold=0.005,
        contact_matching_normal_dot_threshold=0.95,
    )


_RBY1_IK_INITIAL_JOINT_POS = {
    "torso_joint_1": 0.0,
    "torso_joint_2": 0.872664213180542,
    "torso_joint_3": -1.5707811117172241,
    "torso_joint_4": 0.6981245279312134,
    "torso_joint_5": 3.796982127823867e-06,
    "torso_joint_6": 0.0,
    "right_arm_joint_1": 0.3021828234195709,
    "right_arm_joint_2": -0.013802030123770237,
    "right_arm_joint_3": -0.09509921818971634,
    "right_arm_joint_4": -2.2242417335510254,
    "right_arm_joint_5": -0.7117632627487183,
    "right_arm_joint_6": 0.14113007485866547,
    "right_arm_joint_7": 0.5137608647346497 + math.pi / 2.0,
    "left_arm_joint_1": -0.4555884897708893,
    "left_arm_joint_2": 0.2500312626361847,
    "left_arm_joint_3": -0.665743887424469,
    "left_arm_joint_4": -1.3314952850341797,
    "left_arm_joint_5": -0.19328542053699493,
    "left_arm_joint_6": -0.5307496786117554,
    "left_arm_joint_7": 0.6565361022949219 - math.pi / 2.0,
    "right_gripper_finger_joint_1": 0.09138019700534642,
    "right_gripper_left_finger_joint": -0.09138019700534642 / 2.0,
    "right_gripper_right_finger_joint": 0.09138019700534642 / 2.0,
    "left_gripper_finger_joint_1": 0.09098683297634125,
    "left_gripper_left_finger_joint": -0.09098683297634125 / 2.0,
    "left_gripper_right_finger_joint": 0.09098683297634125 / 2.0,
}


##
# Scene
##


@configclass
class WaterhoseSceneCfg(InteractiveSceneCfg):
    """Cable + plug with the cable tail welded to a kinematic anchor; sky light and ground."""

    ### Static fridge body
    fridge = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Fridge",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "fridge.usda"),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=_FRIDGE_POS),
    )

    ### rby1df robot (28-DOF, fixed base). Drive gains match the reference example.
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_RBY1_USD,
            func=spawn_rby1df_with_right_gripper_finger_collision_overrides,
        ),
        articulation_root_prim_path="/Geometry/origin",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 1.0, -1.0),
            rot=(0.0, 0.0, -0.70710678, 0.70710678),  # 90 deg about +Z (x, y, z, w)
        ),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=["torso_joint_.*", "left_arm_joint_.*", "right_arm_joint_.*", "head_joint_.*"],
                stiffness=120000.0,
                damping=12000.0,
                effort_limit=10000.0,
                armature=0.2,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[".*_gripper_finger_joint_1"],
                stiffness=10000.0,
                damping=1000.0,
                effort_limit=100000.0,
                armature=0.5,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*_gripper_(left|right)_finger_joint"],
                stiffness=500000.0,
                damping=10000.0,
                effort_limit=500000.0,
                armature=0.5,
            ),
        },
    )

    ### Cable 1 (graspable plug welded to the head; tail welded to a kinematic anchor sphere)

    plug1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plug1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "cable", "plug.usda"),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.38398558, 0.34585292, 0.5 - 0.36874688),
            rot=(0.0, -0.57096256, 0.0, 0.8209761),
        ),
    )

    # Cable tail anchor. The cable tail is pinned to a 1 mm per-env kinematic sphere
    # so it does not fall under gravity while the robot grasps and inserts the head.
    # The cable tail node is welded to this body with a Newton `add_joint_fixed`
    # constraint (see the `CableAttachmentCfg` on `cable1`). A per-env body is used
    # rather than the global world body (-1): a fixed joint to the shared world body
    # corrupts the multi-env coupled MJWarp+VBD solve (robot joints go NaN at step 0).
    anchor1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Anchor1",
        spawn=sim_utils.SphereCfg(
            radius=0.001,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_ANCHOR_POS),
    )

    # Insertion socket: the REAL fridge socket collider (``socket_collision.usda``), a single
    # concave mesh (``physics:approximation = "none"``) forming a tube with an inner bore Ø~6 mm and
    # outer Ø~11.7 mm. NO SDF: plug/cable-vs-mesh uses Newton's BVH mesh narrow-phase (verified
    # stable); a texture SDF on this mesh hits the CUDA-illegal-access bug. The mesh points are
    # authored in the fridge ``/root`` frame, so it spawns at the fridge pose (identity rot). It is
    # a static ``body=-1`` shape pulled into the VBD solver via the vbd entry's ``shape_label_patterns``.
    # NOTE: the Ø6 mm bore matches the cable (Ø6 mm); the Ø11 mm plug connector only tip-seats on the
    # mouth -- a shallow male/female mate, not a deep slide.
    socket1 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Socket1",
        spawn=sim_utils.UsdFileCfg(usd_path=_SOCKET_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=_FRIDGE_POS, rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # The deformable cable, simulated as a Cosserat rod by the VBD solver. Material values are
    # tuned for the plug-grasp path:
    #   stretch_stiffness -- axial EA; 1e8 resists stretching without exploding (1e12 blew up).
    #   bend_stiffness    -- resistance to bending; keeps the hose from kinking unnaturally.
    #   stretch/bend_damping -- velocity damping; small values quell jitter without over-damping.
    #   density           -- mass per unit volume; balanced against the gripper proxy mass_scale
    #                        so contact does not pump energy into the rod.
    cable1 = CableObjectCfg(
        prim_path="/World/envs/env_.*/Cable1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "cable", "cable001.usda"),
            physics_material=NewtonCableMaterialCfg(
                stretch_stiffness=1e8,
                bend_stiffness=30.0,
                stretch_damping=1e-3,
                bend_damping=1e0,
                density=10000.0,
            ),
        ),
        init_state=CableObjectCfg.InitialStateCfg(
            pos=_FRIDGE_POS,
        ),
        # Keep the authored node spacing: the head plug weld's 22 mm offset is authored against
        # the original segment-0 frame, so resampling would invalidate it. None = no resampling.
        resample_segment_length=None,
        attachments=[
            # Head weld: cable segment-0 head node -> graspable Plug1 rigid body.
            CableAttachmentCfg(
                target_prim_path="/World/envs/env_.*/Plug1",
                cable_anchor=0,
                cable_local_pos=(0.0, 0.0, 0.022),  # the head node is 22mm along +Z from the head body center
            ),
            # Tail weld: cable last-segment start node (42) -> kinematic Anchor1 sphere.
            CableAttachmentCfg(
                target_prim_path="/World/envs/env_.*/Anchor1",
                cable_anchor=42,  # last segment start node; Anchor1 sits exactly there
            ),
        ],
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
        spawn=GroundPlaneCfg(),
    )


##
# MDP overrides
##


@configclass
class ActionsCfg:
    """Joint-position control of the rby1df robot.

    Actions are offsets from the default joint pose (``use_default_offset=True``), so a
    zero action holds the rest configuration. Only the gripper *driver* joints
    (``*_gripper_finger_joint_1``) are actuated; the left/right finger joints follow
    them via the USD mimic joints.
    """

    body_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["torso_joint_.*", "left_arm_joint_.*", "right_arm_joint_.*", "head_joint_.*"],
        scale=0.1,
        use_default_offset=True,
    )
    gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_gripper_finger_joint_1"],
        scale=80.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Policy observations for the cable plug task."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events for the cable plug task."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    gripper_finger_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["right_gripper_leftfinger", "right_gripper_rightfinger"],
            ),
            "static_friction_range": (1.0e6, 1.0e6),
            "dynamic_friction_range": (1.0e6, 1.0e6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )


@configclass
class RewardsCfg:
    """Reach-and-track reward shaping for the rigid plug."""

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-2)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1e-4)


@configclass
class TerminationsCfg:
    """Time-out only; the cable is anchored so the plug cannot escape the workspace."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class WaterhoseEnvCfg(ManagerBasedRLEnvCfg):
    """Waterhose environment reusing the cable-plug MDP on an externally loaded scene."""

    scene: WaterhoseSceneCfg = WaterhoseSceneCfg(num_envs=8, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        # general settings
        self.decimation = 1
        self.episode_length_s = 1.0

        # simulation settings
        self.sim.dt = 1 / 100.0
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.81)

        kit_view = dict(
            eye=_KIT_CAMERA_EYE,
            lookat=_KIT_CAMERA_LOOKAT,
            window_width=1600,
            window_height=1600,
        )
        newton_view = dict(
            eye=_NEWTON_CAMERA_EYE,
            lookat=_NEWTON_CAMERA_LOOKAT,
            window_width=1600,
            window_height=1600,
            show_static=True,
        )
        self.sim.visualizer_cfgs = [KitVisualizerCfg(**kit_view), NewtonVisualizerCfg(**newton_view)]

        # Resolution of `--video` recordings (independent of the on-screen visualizer windows above).
        self.video_recorder.window_width = 1600
        self.video_recorder.window_height = 1600

        # Coupled physics: MuJoCo-Warp solves the articulated rby1df robot, VBD solves the
        # deformable cable (and its welded plug/anchor bodies). The two are joined by "proxy"
        # coupling: the gripper bodies are mirrored as proxy bodies in the VBD solver so the cable
        # collides against them.
        #
        # NOTE on coupling direction & stability: a default proxy is a fully-dynamic, finite-mass
        # body (mass = ``mass_scale * effective mass``) that exchanges force with the cable in
        # BOTH directions -- this is genuine two-way proxy coupling (the robot "feels" the cable).
        # It is stable for a single env, but when the gripper closes on the rigid ~1 g plug the
        # plug is squeezed between two finite-mass proxy "walls" that themselves move, forming a
        # two-sided plug<->proxy feedback loop; the few-iteration AVBD solve cannot equilibrate the
        # opposing stiff penalties and (for num_envs > 1) the marginal squeeze can blow up. For a
        # rock-solid grasp the cleaner model is ONE-WAY coupling, where the gripper proxies are
        # KINEMATIC colliders (immovable, inverse mass 0) that drive the cable but are never pushed
        # back -- see :class:`WaterhoseKinematicIkEnvCfg` (task ``Isaac-Waterhose-Kinematic-v0``),
        # which sets ``CoupledProxyCfg.immovable=True``. This task keeps the two-way proxy.
        self.sim.physics = CoupledNewtonCfg(
            scene_cfg=self.scene,
            use_cuda_graph=_env_flag("WATERHOSE_USE_CUDA_GRAPH", True),
            solver_cfg=CoupledSolverCfg(
                coupling_type="proxy",
                scene_cfg=self.scene,
                entries=[
                    CoupledSolverEntryCfg(
                        name="mjc",
                        solver_cfg=MJWarpSolverCfg(
                            cone="elliptic",
                            ls_parallel=True,
                            ls_iterations=20,
                            integrator="implicitfast",
                            use_mujoco_contacts=False,
                        ),
                        body_entities=[SceneEntityCfg("robot")],
                    ),
                    CoupledSolverEntryCfg(
                        name="vbd",
                        # Keep plug-era cable welds compliant: the cable head->Plug1 and
                        # tail->Anchor1 fixed joints have small authored offsets, so hard
                        # AVBD joints can inject a large startup impulse and explode the cable.
                        solver_cfg=VBDSolverCfg(
                            iterations=10,
                            friction_epsilon=0.1,
                            rigid_contact_hard=False,
                            rigid_joint_hard=False,
                            rigid_avbd_beta=1.0e5,
                            rigid_avbd_gamma=0.999,
                            rigid_contact_history=False,
                            rigid_contact_k_start=1.0e2,
                            # The generic cable_robot example uses 128, but the authored
                            # cable/gripper contact can exceed 600 contacts on one body
                            # during grasp, so smaller buffers overflow and poison the solve.
                            rigid_body_contact_buffer_size=768,
                            rigid_joint_linear_ke=1.0e5,
                            rigid_joint_angular_ke=1.0e5,
                            rigid_joint_linear_k_start=1.0e4,
                            rigid_joint_angular_k_start=1.0e1,
                            rigid_joint_linear_kd=0.0,
                            rigid_joint_angular_kd=0.0,
                        ),
                        solver_class="newton.solvers:SolverVBD",
                        body_entities=[
                            SceneEntityCfg("cable1"),
                            # Weld targets must live in the VBD model so the cable fixed joint
                            # (head -> Plug1, tail -> Anchor1) can be created against them.
                            SceneEntityCfg("plug1"),
                            SceneEntityCfg("anchor1"),
                        ],
                        all_particles=True,
                        include_static_shapes=False,
                        # Pull ONLY the static socket collider into the VBD solver so the
                        # VBD-owned plug collides with the bore (include_static_shapes=False keeps
                        # the cable from colliding with the ground/fridge). Matching against the
                        # full Newton shape label of the spawned Socket1 mesh.
                        shape_label_patterns=[r".*/Socket1/.*"],
                    ),
                ],
                proxy_coupling=ProxyCouplingCfg(
                    proxies=[
                        CoupledProxyCfg(
                            source="mjc",
                            destination="vbd",
                            body_entities=[
                                SceneEntityCfg(
                                    "robot",
                                    body_names=[
                                        "right_gripper_base",
                                        "right_gripper_leftfinger",
                                        "right_gripper_rightfinger",
                                        "left_gripper_base",
                                        "left_gripper_leftfinger",
                                        "left_gripper_rightfinger",
                                    ],
                                )
                            ],
                            # Use the source solver's end-of-step pose and velocity together.
                            # The lagged path copies begin pose + end velocity, which can inject
                            # inconsistent proxy motion while the gripper closes on the plug.
                            mode="staggered",
                            mass_scale=1.0,
                            collision_pipeline_factory=_make_proxy_collision_pipeline,
                            collide_interval=1,
                            shape_material_ke=2.0e5,
                            shape_material_kd=1.0e-1,
                            # mu=3 is ample Coulomb friction to hold the clamped plug (mu>1 means
                            # friction can exceed the normal load); the previous 1e6 is unphysical.
                            shape_material_mu=3.0,
                            shape_margin=0.0,
                        )
                    ],
                    iterations=1,
                ),
            ),
            num_substeps=10,
            collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=65536),
            model_cfg=NewtonModelCfg(
                shape_material_ke=1.0e5,
                shape_material_kd=1.0e-1,
                soft_contact_mu=1.0,
                shape_material_mu=2.0,
            ),
        )


@configclass
class WaterhoseIkActionsCfg:
    """Absolute right end-effector pose plus normalized right-gripper action for scripted demos."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["right_arm_joint_.*"],
        body_name="right_gripper_base",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.05},
        ),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=_RIGHT_GRIPPER_EE_FRAME_POS,
            rot=_RIGHT_GRIPPER_EE_FRAME_ROT,
        ),
    )
    gripper_action = WaterhoseGripperPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_gripper_finger_joint_1",
            "right_gripper_left_finger_joint",
            "right_gripper_right_finger_joint",
        ],
        open_command_expr={
            "right_gripper_finger_joint_1": 0.09,
            "right_gripper_left_finger_joint": -0.045,
            "right_gripper_right_finger_joint": 0.045,
        },
        # Firm clamp on the ~14.6 mm plug flange: ~11 mm gap (each finger +/-5.5 mm) -> ~1.8 mm
        # interference per side against the kinematic gripper proxy, so the plug is pinned and does
        # not rotate/slip in the grip. (Requires the corrected EE offset -0.075 so the fingers are
        # centered on the flange in the first place.)
        close_command_expr={
            "right_gripper_finger_joint_1": 0.013,
            "right_gripper_left_finger_joint": -0.0065,
            "right_gripper_right_finger_joint": 0.0065,
        },
    )


@configclass
class WaterhoseIkEnvCfg(WaterhoseEnvCfg):
    """Waterhose variant with an IK action space for the scripted RBY1 demo."""

    actions: WaterhoseIkActionsCfg = WaterhoseIkActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 30.0
        self.scene.robot.init_state.joint_pos = _RBY1_IK_INITIAL_JOINT_POS


@configclass
class WaterhoseNewtonIkActionsCfg(WaterhoseIkActionsCfg):
    """Newton-native absolute right end-effector pose plus normalized right-gripper action."""

    arm_action = NewtonInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["torso_joint_.*", "right_arm_joint_.*"],
        body_name="right_gripper_base",
        controller=NewtonIKManagerCfg(
            command_type="pose",
            use_relative_mode=False,
            iterations=24,
            lambda_initial=0.1,
            jacobian_mode="analytic",
            joint_limit_weight=10.0,
            use_persistent_seed=True,
        ),
        ik_model_source="asset_usd",
        fixed_body_names=["left_gripper_base", "torso_hip_yaw"],
        fixed_body_weights=[1.0, 50.0],
        body_offset=NewtonInverseKinematicsActionCfg.OffsetCfg(
            pos=_RIGHT_GRIPPER_EE_FRAME_POS,
            rot=_RIGHT_GRIPPER_EE_FRAME_ROT,
        ),
    )


@configclass
class WaterhoseNewtonRelativeIkActionsCfg(WaterhoseIkActionsCfg):
    """Newton-native relative end-effector delta pose plus normalized right-gripper action."""

    arm_action = NewtonInverseKinematicsActionCfg(
        class_type="isaaclab_tasks.contrib.waterhose.mdp.actions:WaterhoseLocalFrameNewtonInverseKinematicsAction",
        asset_name="robot",
        joint_names=["torso_joint_.*", "right_arm_joint_.*"],
        body_name="right_gripper_base",
        controller=NewtonIKManagerCfg(
            command_type="pose",
            use_relative_mode=True,
            iterations=24,
            lambda_initial=0.1,
            jacobian_mode="analytic",
            joint_limit_weight=10.0,
            use_persistent_seed=True,
        ),
        ik_model_source="asset_usd",
        fixed_body_names=["left_gripper_base", "torso_hip_yaw"],
        fixed_body_weights=[1.0, 50.0],
        body_offset=NewtonInverseKinematicsActionCfg.OffsetCfg(
            pos=_RIGHT_GRIPPER_EE_FRAME_POS,
            rot=_RIGHT_GRIPPER_EE_FRAME_ROT,
        ),
    )


@configclass
class WaterhoseProxyIkEnvCfg(WaterhoseEnvCfg):
    """Waterhose task with Newton proxy coupling and the scripted IK action space."""

    actions: WaterhoseNewtonIkActionsCfg = WaterhoseNewtonIkActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 30.0
        self.scene.robot.init_state.joint_pos = _RBY1_IK_INITIAL_JOINT_POS
        if _TELEOP_AVAILABLE:
            self.xr = XrCfg(
                anchor_pos=(0.0, 0.9, -1),
                # XrCfg quaternions are xyzw. Rotate the simulation 180 deg
                # around world up so the headset initially faces the fridge.
                anchor_rot=(0.0, 0.0, 1.0, 0.0),
            )
            self.isaac_teleop = IsaacTeleopCfg(
                pipeline_builder=_build_waterhose_teleop_pipeline,
                sim_device=self.sim.device,
                xr_cfg=self.xr,
                target_frame_prim_path=_ROBOT_BASE_PRIM_PATH_ENV0,
                teleoperation_active_default=True,
                control_channel_uuid=None,
            )


@configclass
class WaterhoseKinematicIkEnvCfg(WaterhoseProxyIkEnvCfg):
    """Waterhose task with ONE-WAY kinematic-proxy coupling (the stable demo path).

    Same scene, solvers, and scripted IK as :class:`WaterhoseProxyIkEnvCfg`, but the gripper
    proxies are flagged ``immovable`` so they are KINEMATIC colliders in the VBD solve: they track
    the gripper pose and drive the cable, but the cable can never push them back and no reaction is
    harvested onto the robot. This removes the two-sided dynamic-proxy feedback loop that makes the
    default finite-mass two-way proxy marginal when the gripper squeezes the rigid plug, so the full
    grasp+retract+insert demo stays stable (verified for num_envs up to 8). Choose this variant when
    you want a robust grasp and do not need the cable's reaction force fed back to the robot; use the
    two-way :class:`WaterhoseProxyIkEnvCfg` when that force feedback matters.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        for proxy in self.sim.physics.solver_cfg.proxy_coupling.proxies:
            proxy.immovable = True


@configclass
class WaterhoseAdmmIkEnvCfg(WaterhoseProxyIkEnvCfg):
    """Waterhose task variant using Newton ADMM contact coupling instead of proxy bodies."""

    def __post_init__(self) -> None:
        super().__post_init__()
        solver_cfg = self.sim.physics.solver_cfg
        solver_cfg.coupling_type = "admm"
        solver_cfg.use_collision_pipeline = False
        # NOTE: ADMM contact coupling is fragile for this plug grasp. With rho=50 the cable
        # explodes the instant the fingers close on the plug (each augmented-Lagrangian iteration
        # overshoots the stiff contact, so *more* iterations make it worse). rho=5 lets the grasp
        # itself succeed, but the scene still destabilizes later during transport (the plug blows
        # up while the gripper carries it toward the socket). Unlike the proxy path -- where
        # making the proxy immovable (large mass_scale) gives a fully stable full-demo run -- ADMM
        # has no comparable single-knob fix here. Prefer the proxy task (Isaac-Waterhose-Coupled-v0)
        # for a stable demo; treat this ADMM variant as experimental until the ADMM contact
        # solve is hardened (lower/auto rho, contact damping, or a robust complementarity step).
        solver_cfg.admm_coupling = AdmmCouplingCfg(
            iterations=5,
            rho=5.0,
            gamma=0.1,
            baumgarte=0.01,
            contact_pairs=[
                AdmmContactPairCfg(
                    source="mjc",
                    destination="vbd",
                    detection_margin=0.005,
                )
            ],
        )


@configclass
class WaterhoseProxyTeleopEnvCfg(WaterhoseProxyIkEnvCfg):
    """Waterhose task variant for native IsaacLab keyboard and SpaceMouse teleoperation."""

    actions: WaterhoseNewtonRelativeIkActionsCfg = WaterhoseNewtonRelativeIkActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        if _TELEOP_AVAILABLE:
            self.isaac_teleop = IsaacTeleopCfg(
                pipeline_builder=_build_waterhose_relative_teleop_pipeline,
                sim_device=self.sim.device,
                xr_cfg=self.xr,
                app_name="WaterhoseTeleop",
                target_frame_prim_path=_ROBOT_BASE_PRIM_PATH_ENV0,
                teleoperation_active_default=True,
                control_channel_uuid=None,
            )
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.02,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": WaterhoseSpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.15,
                    sim_device=self.sim.device,
                ),
            }
        )
