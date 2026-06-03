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

from isaaclab_newton.physics import (
    CoupledProxyCfg,
    CoupledSolverCfg,
    CoupledSolverEntryCfg,
    HydroelasticSDFCfg,
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
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.sim.utils import clone
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab_contrib.cable.cable_object_cfg import CableAttachmentCfg, CableObjectCfg
from isaaclab_contrib.deformable.newton_manager_cfg import (
    CoupledNewtonCfg,
    NewtonModelCfg,
    VBDSolverCfg,
)

from . import mdp
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
_RIGHT_GRIPPER_EE_FRAME_POS = (0.0, 0.0, -0.1055)
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

_GRIPPER_ACTUAL_MESH_COLLISION = MeshCollisionBaseCfg(mesh_approximation_name="none")
_GRIPPER_SDF_COLLISION = sim_utils.NewtonSDFCollisionPropertiesCfg(
    sdf_max_resolution=64,
    sdf_narrow_band_inner=0.002,
    sdf_narrow_band_outer=0.006,
    sdf_texture_format="float32",
    sdf_padding=0.001,
    hydroelastic_enabled=True,
    hydroelastic_stiffness=1.0e8,
)


def _is_gripper_collision_instance(prim: Usd.Prim) -> bool:
    """Return true for rby1df instanceable gripper collision Xforms."""
    if not prim.IsInstance():
        return False
    path = prim.GetPath().pathString
    if "gripper" not in path:
        return False
    name = prim.GetName()
    parent_name = prim.GetParent().GetName()
    return name.endswith("_collision") or (name == parent_name and name.endswith("finger"))


def _apply_actual_mesh_collision_to_grippers(robot_prim: Usd.Prim) -> None:
    """Use real mesh/SDF collision for gripper meshes while leaving the rest of the robot unchanged."""
    from pxr import Usd, UsdPhysics

    stage = robot_prim.GetStage()
    gripper_collision_instances = [
        prim.GetPath().pathString for prim in Usd.PrimRange(robot_prim) if _is_gripper_collision_instance(prim)
    ]

    modified_meshes = []
    for instance_path in gripper_collision_instances:
        instance_prim = stage.GetPrimAtPath(instance_path)
        instance_prim.SetInstanceable(False)
        for child_prim in Usd.PrimRange(instance_prim):
            if child_prim.GetTypeName() != "Mesh" or not child_prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            mesh_path = child_prim.GetPath().pathString
            sim_schemas.modify_mesh_collision_properties(mesh_path, _GRIPPER_ACTUAL_MESH_COLLISION, stage=stage)
            sim_schemas.modify_collision_properties(mesh_path, _GRIPPER_SDF_COLLISION, stage=stage)
            modified_meshes.append(mesh_path)

    if not modified_meshes:
        logging.warning("Did not find any rby1df gripper collision meshes to switch to actual mesh collision.")


@clone
def spawn_rby1df_with_gripper_mesh_collision(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn rby1df and override only gripper collision meshes to use Newton SDF on real mesh geometry."""
    prim = spawn_from_usd.__wrapped__(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    _apply_actual_mesh_collision_to_grippers(prim)
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

    from isaacteleop.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        Se3RelRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")

    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_hands = hands.transformed(transform_input.output(ValueInput.VALUE))

    se3_cfg = Se3RetargeterConfig(
        input_device=HandsSource.RIGHT,
        zero_out_xy_rotation=True,
        use_wrist_rotation=True,
        use_wrist_position=True,
        delta_pos_scale_factor=15.0,
        delta_rot_scale_factor=2.0,
        alpha_pos=0.5,
        alpha_rot=0.5,
    )
    se3 = Se3RelRetargeter(se3_cfg, name="ee_delta")
    connected_se3 = se3.connect({HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT)})

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
            "ee_delta": connected_se3.output("ee_delta"),
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
    """Cable + plug with the cable tail captured by SDF contact; sky light and ground."""

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
        spawn=sim_utils.UsdFileCfg(usd_path=_RBY1_USD, func=spawn_rby1df_with_gripper_mesh_collision),
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
            collision_props=sim_utils.NewtonSDFCollisionPropertiesCfg(
                sdf_max_resolution=64,
                sdf_narrow_band_inner=0.002,
                sdf_narrow_band_outer=0.006,
                sdf_texture_format="float32",
                sdf_padding=0.001,
                hydroelastic_enabled=True,
                hydroelastic_stiffness=1.0e8,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.38398558, 0.34585292, 0.5 - 0.36874688),
            rot=(0.0, -0.57096256, 0.0, 0.8209761),
        ),
    )

    # Cable tail anchor.
    #
    # The cable tail is pinned to the world so it does not fall under gravity while the robot
    # grasps and inserts the head. Two mechanisms can do this; we use the *weld* (option A):
    #
    #   Option A -- fixed-joint weld (used here).
    #     A 1 mm kinematic sphere (`Anchor1`) is placed exactly at the tail node's world pose
    #     and the cable tail node is welded to it with a Newton `add_joint_fixed` constraint
    #     (see the `CableAttachmentCfg` on `cable1`). Pros: exact, drift-free, cheap (two
    #     constraint rows, no collision geometry). With the softened VBD joints used here
    #     (`rigid_joint_hard=False` + `k_start` ramps) the weld is compliant rather than
    #     infinitely stiff, so it no longer pumps energy on contact the way a hard weld did.
    #     A per-env body is used rather than the global world body (-1): a fixed joint to the
    #     shared world body corrupts the multi-env coupled MJWarp+VBD solve (robot joints go
    #     NaN at step 0).
    #
    #   Option B -- static SDF capture (alternative; see `CableSdfCaptureCfg`).
    #     Instead of a joint, generate a static collision fixture (a "retaining cup" at the tail
    #     tip plus several "through-sleeves" around the last segments) as signed-distance-field
    #     meshes built once and reused across envs. The cable is mechanically trapped inside the
    #     fixture by contact rather than constrained by a joint. To switch to it, drop the tail
    #     `CableAttachmentCfg` below and instead pass e.g.::
    #
    #         sdf_captures=[
    #             CableSdfCaptureCfg(cable_anchor=-1, label_suffix="tail_sdf_capture", ...),
    #             *(CableSdfCaptureCfg(cable_anchor=i, through_sleeve=True, ...) for i in range(-2, -8, -1)),
    #         ]
    #
    #     and add the generated shapes to the VBD entry's `shape_label_patterns`
    #     (e.g. r"/World/envs/env_\\d+/Cable1/tail_sdf_.*"). Pros: contact can slip/settle via
    #     friction and only constrains the cable where it actually touches, so it yields more
    #     gracefully than a stiff weld. Cons: 8 collision shapes per cable cost more contact work
    #     at runtime than the two-row weld, and the tail is held only as firmly as friction allows.
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

    # Physical socket collider.
    #
    # The fridge socket mouth/bore baked into one standalone triangle mesh
    # (assets/fridge/socket_collision.usda, generated by
    # scripts/environments/waterhose/create_socket_collision_asset.py from the
    # Cable008 socket collider fragments). Its mesh points live in the fridge /root
    # frame, so spawning it at _FRIDGE_POS lands it exactly over the fridge socket.
    # A Newton SDF collider is built on it so cable/connector contact can be
    # tested against the real socket bore.
    socket_collision = AssetBaseCfg(
        prim_path="/World/envs/env_.*/SocketCollision",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "socket_collision.usda"),
            collision_props=sim_utils.NewtonSDFCollisionPropertiesCfg(
                sdf_max_resolution=64,
                sdf_narrow_band_inner=0.002,
                sdf_narrow_band_outer=0.006,
                sdf_texture_format="float32",
                sdf_padding=0.001,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=_FRIDGE_POS),
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
                density=1000.0,
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
        sdf_captures=[],
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
        # deformable cable (and its welded plug/anchor bodies). The two are joined by one-way
        # "proxy" coupling: the gripper bodies are mirrored as immovable proxy shapes in the VBD
        # solver so the cable collides against them, but the cable cannot push the robot back.
        # This avoids two-way energy pumping while still letting the gripper grasp the cable.
        self.sim.physics = CoupledNewtonCfg(
            scene_cfg=self.scene,
            use_cuda_graph=True,
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
                            rigid_contact_hard=True,
                            rigid_joint_hard=False,
                            rigid_avbd_beta=1.0e5,
                            rigid_avbd_gamma=0.999,
                            rigid_contact_history=False,
                            rigid_contact_k_start=1.0e2,
                            # The generic cable_robot example uses 128, but this IsaacLab
                            # scene produces >300 per-body cable/gripper/socket contacts during
                            # grasp, so 128 overflows and poisons the solve.
                            rigid_body_contact_buffer_size=512,
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
                        # Pull the physical socket collider into the VBD solve for direct
                        # cable/socket contact tests.
                        shape_label_patterns=[
                            r"/World/envs/env_\d+/SocketCollision/.*",
                        ],
                        all_particles=True,
                        include_static_shapes=False,
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
                            mode="lagged",
                            # Match Newton's generic MuJoCo+VBD cable_robot proxy-coupled example.
                            mass_scale=1.0,
                            collision_pipeline_factory=_make_proxy_collision_pipeline,
                            collide_interval=5,
                            shape_material_ke=2.0e5,
                            shape_material_kd=1.0e-1,
                            shape_material_mu=3.0e6,
                            shape_margin=0.001,
                        )
                    ],
                    iterations=1,
                ),
            ),
            num_substeps=10,
            collision_cfg=NewtonCollisionPipelineCfg(
                rigid_contact_max=65536,
                sdf_hydroelastic_config=HydroelasticSDFCfg(),
            ),
            model_cfg=NewtonModelCfg(
                shape_material_ke=1.0e5,
                shape_material_kd=1.0e-1,
                soft_contact_mu=0.5,
                shape_material_mu=1.0,
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
    gripper_action = mdp.WaterhoseGripperPositionActionCfg(
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
        # Sized for the ~7.3 mm-radius Plug1 (not the 3 mm cable): each finger closes to
        # +/-7 mm, ~0.3 mm inside the plug surface, so the fingers squeeze the plug through
        # the soft VBD contact instead of being driven ~7 mm past the surface.
        close_command_expr={
            "right_gripper_finger_joint_1": 0.014,
            "right_gripper_left_finger_joint": -0.007,
            "right_gripper_right_finger_joint": 0.007,
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
        class_type=mdp.WaterhoseLocalFrameNewtonInverseKinematicsAction,
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
                anchor_pos=(0.0, 0.0, 0.0),
                anchor_rot=(0.0, 0.0, 0.0, 1.0),
            )
            self.isaac_teleop = IsaacTeleopCfg(
                pipeline_builder=_build_waterhose_teleop_pipeline,
                sim_device=self.sim.device,
                xr_cfg=self.xr,
                target_frame_prim_path=_ROBOT_BASE_PRIM_PATH_ENV0,
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
