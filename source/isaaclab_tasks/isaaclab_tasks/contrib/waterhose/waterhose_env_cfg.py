# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waterhose manipulation environment.

An RBY1DF robot grasps a hose connector and inserts it into a fridge socket under a Newton coupled
solver (MuJoCo-Warp rigid robot + VBD deformable hose). The MDP uses standard
:mod:`isaaclab.envs.mdp` terms plus the task-local action/termination terms in :mod:`.mdp`.
"""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path

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
)
from isaaclab_newton.sim.schemas.schemas_cfg import NewtonSDFCollisionPropertiesCfg
from isaaclab_newton.sim.spawners.materials.physics_materials_cfg import NewtonCableMaterialCfg
from isaaclab_teleop import IsaacTeleopCfg, XrCfg
from isaaclab_visualizers.kit.kit_visualizer_cfg import KitVisualizerCfg
from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonVisualizerCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import schemas as sim_schemas
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.sim.utils import clone
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import (
    CouplerAdmmCfg,
    CouplerEntryCfg,
    CouplerProxyCfg,
    CouplerProxyMappingCfg,
)
from isaaclab_contrib.deformable.newton_manager_cfg import (
    NewtonModelCfg,
    VBDSolverCfg,
)

from .cable import WaterhoseCableObjectCfg
from .geometry import (
    ANCHOR_POS,
    CONNECTOR_LOCAL_POS,
    CONNECTOR_LOCAL_QUAT_XYZW,
    CONNECTOR_MASS,
    CONNECTOR_TIP_LOCAL_POS,
    FRIDGE_POS,
    RIGHT_GRIPPER_EE_FRAME_POS,
    RIGHT_GRIPPER_EE_FRAME_QUAT_XYZW,
    SOCKET_COLLISION_MESH_PATTERN,
    SOCKET_COLLISION_MESH_SUFFIX,
    SOCKET_MOUTH_POS,
    SOCKET_ROT_QUAT_XYZW,
    SOCKET_SEATED_TIP_DEPTH,
)
from .mdp.actions import WaterhoseGripperPositionActionCfg
from .mdp.events import reset_cable_to_default
from .mdp.terminations import plug_inserted_in_socket

# The bimanual pipeline exposes complete absolute poses for both AVP wrists.
from .teleop_pipelines import build_waterhose_bimanual_teleop_pipeline

_LOCAL_ASSETS_DIR = Path(__file__).resolve().parent / "assets"

_FRIDGE_USD = str(_LOCAL_ASSETS_DIR / "fridge" / "fridge_waterhose.usda")
_RBY1_USD = str(_LOCAL_ASSETS_DIR / "rby1df" / "rby1df_waterhose.usda")
_PLUG_USD = str(_LOCAL_ASSETS_DIR / "fridge" / "cable" / "plug.usda")
_CABLE1_USD = str(_LOCAL_ASSETS_DIR / "fridge" / "cable" / "cable001.usda")
_FRIDGE_COLLISION_PROXY_USD = str(_LOCAL_ASSETS_DIR / "fridge_clearanced_collision.usda")
# Keep the sky and ground local so the demo needs no S3 or Nucleus connection at runtime. The ground
# USD uses relative texture paths and the locally resolved ``OmniPBR.mdl``.
_SKY_HDR = str(_LOCAL_ASSETS_DIR / "skies" / "kloofendal_43d_clear_puresky_4k.hdr")
_GROUND_USD = str(_LOCAL_ASSETS_DIR / "ground" / "default_environment.usd")

# Side-biased scene camera shared by the environment controller and every visualizer. The robot
# stands behind the fridge along +Y, so a nearly head-on view from -Y lets the fridge occlude it.
_SCENE_CAMERA_EYE = (-4.0, -1.0, 2.2)
_SCENE_CAMERA_LOOKAT = (0.0, 0.5, 0.2)
_ROBOT_BASE_PRIM_PATH_ENV0 = "/World/envs/env_0/Robot/Geometry/origin"
_ROBOT_BODY_PATTERN = r"/World/envs/env_.*/Robot"
_CABLE_BODY_PATTERN = r"/World/envs/env_.*/Cable1"
_GRIPPER_FINGER_BODY_PATTERN = r"/World/envs/env_.*/Robot/.*/(left|right)_gripper_(left|right)finger"
_FRIDGE_ROBOT_COLLISION_PATTERN = r".*/FridgeRobotCollision/Housing.*"
_FRIDGE_CABLE_COLLISION_PATTERN = r".*/FridgeCableCollision/Housing.*"
_SOCKET_CONTACT_PROPERTIES = NewtonSDFCollisionPropertiesCfg(contact_gap=0.0)


@clone
def spawn_fridge_without_embedded_body_collision(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the canonical fridge with task-specific housing and socket contact geometry."""

    prim = spawn_from_usd.__wrapped__(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    body_collision = prim.GetStage().GetPrimAtPath(f"{prim_path}/Cable008/BodyCollision")
    if not body_collision.IsValid():
        raise RuntimeError(f"Fridge body-collision scope is missing under {prim_path!r}.")
    body_collision.SetActive(False)

    # The model-wide Newton default is a 10 mm contact gap. That is useful for the hose and coarse
    # scene geometry, but it makes the connector stop about 10 mm in front of this tiny socket and
    # slide through the gripper. Keep the socket SDF padding authored by the canonical asset, while
    # overriding only its broad contact gap so the physical flange can dock at the mesh surface.
    socket_mesh_path = f"{prim_path}{SOCKET_COLLISION_MESH_SUFFIX}"
    socket_mesh = prim.GetStage().GetPrimAtPath(socket_mesh_path)
    if not socket_mesh.IsValid():
        raise RuntimeError(f"Fridge socket collision mesh is missing at {socket_mesh_path!r}.")
    sim_schemas.modify_collision_properties(socket_mesh_path, _SOCKET_CONTACT_PROPERTIES, stage=prim.GetStage())
    return prim


@clone
def _spawn_rby1_with_fabric_compatible_geometry(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn RBY1 with ordinary references so GPU body transforms reach its render meshes.

    The imported RBY1 USD marks every visual and collision reference instanceable. Kit 110's
    Fabric hierarchy does not propagate Newton-authored parent world matrices across those
    instance-proxy boundaries. De-instancing once during stage authoring preserves the referenced
    geometry and physics while allowing Cubric and RTX to keep the per-frame path entirely on GPU.
    """
    from pxr import Usd  # noqa: PLC0415

    prim = spawn_from_usd.__wrapped__(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    # Collect before editing because de-instancing expands each referenced subtree
    # and therefore invalidates an active PrimRange iterator.
    instance_roots = [descendant for descendant in Usd.PrimRange(prim) if descendant.IsInstanceable()]
    for instance_root in instance_roots:
        instance_root.SetInstanceable(False)
    return prim


_GRIPPER_DRIVER_STIFFNESS = 1.0e4
_GRIPPER_DRIVER_DAMPING = 1.0e3
_GRIPPER_DRIVER_EFFORT_LIMIT = 150.0
_GRIPPER_FINGER_STIFFNESS = 4.0e4
_GRIPPER_FINGER_DAMPING = 2.0e3
_GRIPPER_FINGER_EFFORT_LIMIT = 120.0
# Model-wide VBD contact material [N/m, N·s/m] and friction (ke/kd from Newton's
# franka_cable_ik_pick_place reference grasp). With hard (augmented-Lagrangian) contacts the duals
# enforce non-penetration, so ke is the penalty seed rather than the sole restoring stiffness.
# The default shape friction governs the plug<->socket-bore contact during insertion: too high and
# the plug stick-slips jerkily against the bore wall, so it is kept moderate. Per-asset materials
# override this fallback when authored. Lower it for a smoother push or raise it for more retention.
_VBD_CONTACT_STIFFNESS = 1.0e4
_VBD_CONTACT_DAMPING = 1.0e-1
_VBD_DEFAULT_SHAPE_FRICTION = 0.5
_VBD_SOFT_CONTACT_FRICTION = 0.6
# Preserve the validated predecessor's grip recipe. The 10 mm gap is only the
# broad-phase detection shell; the 1 mm margin is the physical contact surface.
# A friction coefficient of 20 is needed while the light connector is rotated
# and carried against the tail load. Unit proxy mass preserves the authored
# finger inertia instead of amplifying its contact wrench.
_VBD_GRIPPER_PROXY_FRICTION = 20.0
_VBD_GRIPPER_PROXY_MARGIN = 0.001
_VBD_GRIPPER_PROXY_GAP = 0.01
_GRIPPER_PROXY_MASS_SCALE = 1.0
_PROXY_RIGID_CONTACT_MAX = 30000
_RIGHT_GRIPPER_JOINT_NAMES = [
    "right_gripper_finger_joint_1",
    "right_gripper_left_finger_joint",
    "right_gripper_right_finger_joint",
]
# Teleop actions arrive as per-step relative commands. Bound them so high-gain IK targets stay finite
# while still feeling responsive during free teleop.
_TELEOP_MAX_EE_TRANSLATION_DELTA = 0.07
_TELEOP_MAX_EE_ROTATION_DELTA = 0.1
_TELEOP_MAX_GRIPPER_JOINT_DELTA = 0.15


def _env_positive_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value, 10)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer; got {raw_value!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {raw_value!r}.")
    return value


_RIGHT_GRIPPER_OPEN_COMMAND = {
    "right_gripper_finger_joint_1": 0.09,
    "right_gripper_left_finger_joint": -0.045,
    "right_gripper_right_finger_joint": 0.045,
}
_RIGHT_GRIPPER_CLOSE_COMMAND = {
    # Close to 14.8 mm on the ~14.5 mm connector flange and grip through the contact margin band
    # rather than commanding geometric interference (which would drive the fingers through it).
    "right_gripper_finger_joint_1": 0.014,
    "right_gripper_left_finger_joint": -0.007,
    "right_gripper_right_finger_joint": 0.007,
}


_GRIPPER_FINGER_BODY_TOKENS = (
    "left_gripper_leftfinger",
    "left_gripper_rightfinger",
    "right_gripper_leftfinger",
    "right_gripper_rightfinger",
)
# Shape-label suffix for the connector mesh lumped directly into the cable head.
_CONNECTOR_SHAPE_TOKEN = "waterhose_connector"
_CABLE_ROD_SHAPE_TOKEN = "/Cable1/cable_edge_capsule_"
_CABLE_HOUSING_SHAPE_TOKEN = "/FridgeCableCollision/Housing"
_VBD_TIGHT_GEOMETRY_GAP = 0.001


def _make_proxy_collision_pipeline(model):
    """Build the destination-view collision pipeline used by Newton proxy coupling.

    Both grippers constrain the connector mesh lumped into the cable-head body and the articulated
    cable rods.  Finger-to-housing, finger-to-socket, and finger-to-finger pairs are omitted because
    MJWarp already owns robot-to-housing contact. Non-finger cable, housing, socket, and self-contact
    pairs remain enabled.

    ``model`` here is the VBD destination view; this runs once at pipeline construction (after per-entry
    shape visibility is set), so it is CUDA-graph-safe.
    """
    import warp as wp
    from newton import CollisionPipeline, ShapeFlags

    body_label = model.body_label
    finger_bodies = {
        i for i, lbl in enumerate(body_label) if any(str(lbl).endswith(t) for t in _GRIPPER_FINGER_BODY_TOKENS)
    }
    if not finger_bodies:
        raise RuntimeError(
            "waterhose proxy pipeline: no gripper finger proxy bodies found; "
            f"proxy view body labels did not end with any of {_GRIPPER_FINGER_BODY_TOKENS}."
        )
    shape_body = model.shape_body.numpy()
    flags = model.shape_flags.numpy().copy()
    shape_collision_bit = int(ShapeFlags.COLLIDE_SHAPES)
    particle_bit = int(ShapeFlags.COLLIDE_PARTICLES)
    finger_shapes = {
        shape_id
        for shape_id, body_id in enumerate(shape_body)
        if int(body_id) in finger_bodies and (int(flags[shape_id]) & shape_collision_bit)
    }
    if not finger_shapes:
        raise RuntimeError("waterhose proxy pipeline: gripper proxy bodies have no collision shapes.")
    dropped = 0
    for shape_id in finger_shapes:
        if int(flags[shape_id]) & particle_bit:
            flags[shape_id] &= ~particle_bit  # keep rigid shape contact for the articulated hose and plug
            dropped += 1
    model.shape_flags = wp.array(flags, dtype=wp.int32, device=model.device)
    if dropped:
        print(f"[waterhose] Disabled unrelated particle collision on {dropped} finger proxy shape(s).", flush=True)

    # Scope the finger proxies to the cable grip path. This destination-view pipeline otherwise collides
    # the fingers against the housing and socket explicitly routed into the VBD entry, duplicating the
    # robot<->housing contact owned by MJWarp and pushing the gripper away from the bore. Keep every
    # non-finger path plus finger contact with the connector mesh and articulated cable rods.
    import numpy as np

    shape_label = [str(x) for x in model.shape_label] if getattr(model, "shape_label", None) is not None else None
    if shape_label is None:
        raise RuntimeError("waterhose proxy pipeline: model has no shape_label; cannot scope the finger grip pairs.")
    base_pairs = getattr(model, "shape_contact_pairs", None)
    if base_pairs is None:
        raise RuntimeError(
            "waterhose proxy pipeline: model.shape_contact_pairs is None; the explicit broad phase needs it."
        )
    grip_shapes = {
        shape_id for shape_id, label in enumerate(shape_label) if _CONNECTOR_SHAPE_TOKEN.lower() in label.lower()
    }
    if not grip_shapes:
        raise RuntimeError(
            f"waterhose proxy pipeline: no connector collider shape found by label ({_CONNECTOR_SHAPE_TOKEN!r}); "
            "the finger grip-pair filter cannot be built."
        )
    cable_rod_shapes = {
        shape_id for shape_id, label in enumerate(shape_label) if _CABLE_ROD_SHAPE_TOKEN.lower() in label.lower()
    }
    cable_housing_shapes = {
        shape_id for shape_id, label in enumerate(shape_label) if _CABLE_HOUSING_SHAPE_TOKEN.lower() in label.lower()
    }
    socket_shapes = {
        shape_id for shape_id, label in enumerate(shape_label) if SOCKET_COLLISION_MESH_SUFFIX.lower() in label.lower()
    }
    if not cable_rod_shapes or not cable_housing_shapes or not socket_shapes:
        raise RuntimeError(
            "waterhose proxy pipeline: tight-contact shapes are missing "
            f"(cable rods={len(cable_rod_shapes)}, cable housing={len(cable_housing_shapes)}, "
            f"socket={len(socket_shapes)})."
        )
    grip_path_shapes = grip_shapes | cable_rod_shapes
    # Proxy mappings now preserve the source shapes' material arrays verbatim.
    # Apply the task-local grasp material to the destination VBD view before its
    # collision pipeline captures those arrays. Only collision shapes are
    # changed; render shapes sharing a finger body stay untouched.
    material_ke = model.shape_material_ke.numpy().copy()
    material_kd = model.shape_material_kd.numpy().copy()
    material_mu = model.shape_material_mu.numpy().copy()
    shape_margin = model.shape_margin.numpy().copy()
    shape_gap = model.shape_gap.numpy().copy()
    # The predecessor manager applied the task material to the finalized model.
    # PR 5834 correctly preserves imported per-shape arrays instead, but these
    # legacy USDs then retain Newton's standalone defaults (ke=2500, kd=100,
    # mu=1), including 1000x too much damping for the VBD hose path. Normalize
    # only the task-owned connector corridor; unrelated imported scene shapes
    # retain their authored materials.
    task_contact_shapes = grip_shapes | cable_rod_shapes | cable_housing_shapes | socket_shapes
    for shape_id in task_contact_shapes:
        material_ke[shape_id] = _VBD_CONTACT_STIFFNESS
        material_kd[shape_id] = _VBD_CONTACT_DAMPING
        material_mu[shape_id] = _VBD_DEFAULT_SHAPE_FRICTION
    for shape_id in finger_shapes:
        # PR 5834 preserves source-shape materials when it mirrors proxy bodies.
        # The imported RBY1 USD therefore arrives with Newton's standalone
        # ke=2500, kd=100 defaults; the predecessor proxy API overrode both
        # explicitly. Restore the task's VBD contact units here, especially the
        # 1000x lower damping required for a connector that rotates in the grip.
        material_ke[shape_id] = _VBD_CONTACT_STIFFNESS
        material_kd[shape_id] = _VBD_CONTACT_DAMPING
        material_mu[shape_id] = _VBD_GRIPPER_PROXY_FRICTION
        shape_margin[shape_id] = _VBD_GRIPPER_PROXY_MARGIN
        shape_gap[shape_id] = _VBD_GRIPPER_PROXY_GAP
    for shape_id in cable_rod_shapes | cable_housing_shapes:
        # The connector corridor and the 6 mm hose nose are millimetre-scale geometry. The
        # model-wide 10 mm broad-phase gap reaches across that intentional clearance and creates
        # phantom candidates against the housing, so scope only this tight path to 1 mm.
        shape_gap[shape_id] = _VBD_TIGHT_GEOMETRY_GAP
    model.shape_material_ke = wp.array(material_ke, dtype=wp.float32, device=model.device)
    model.shape_material_kd = wp.array(material_kd, dtype=wp.float32, device=model.device)
    model.shape_material_mu = wp.array(material_mu, dtype=wp.float32, device=model.device)
    model.shape_margin = wp.array(shape_margin, dtype=wp.float32, device=model.device)
    model.shape_gap = wp.array(shape_gap, dtype=wp.float32, device=model.device)

    if os.environ.get("WATERHOSE_DEBUG_CONTACTS", "").strip().lower() not in {"", "0", "false", "no", "off"}:
        debug_shapes = sorted(finger_shapes | grip_shapes)
        summary = ", ".join(
            f"{shape_label[shape_id].rsplit('/', 1)[-1]}[{shape_id}]: "
            f"ke={float(material_ke[shape_id]):.3g}, "
            f"kd={float(material_kd[shape_id]):.3g}, "
            f"mu={float(material_mu[shape_id]):.3g}, "
            f"margin={1.0e3 * float(shape_margin[shape_id]):.3g}mm, "
            f"gap={1.0e3 * float(shape_gap[shape_id]):.3g}mm"
            for shape_id in debug_shapes
        )
        print(f"[waterhose contacts] grip inputs: {summary}", flush=True)

    kept_pairs = []
    dropped_unrelated_finger_pairs = 0
    for shape_a, shape_b in base_pairs.numpy().reshape(-1, 2):
        shape_a, shape_b = int(shape_a), int(shape_b)
        a_is_finger, b_is_finger = shape_a in finger_shapes, shape_b in finger_shapes
        if a_is_finger or b_is_finger:
            # Keep only a single-finger pair whose other shape belongs to the cable grip path.
            other, single_finger = (shape_b, not b_is_finger) if a_is_finger else (shape_a, not a_is_finger)
            if not (single_finger and other in grip_path_shapes):
                dropped_unrelated_finger_pairs += 1
                continue
        kept_pairs.append((shape_a, shape_b))
    shape_pairs_filtered = wp.array(
        np.asarray(kept_pairs, dtype=np.int32).reshape(-1, 2), dtype=wp.vec2i, device=model.device
    )
    print(
        f"[waterhose] Scoped gripper proxies to the cable grip path: kept {len(kept_pairs)} contact pair(s), "
        f"dropped {dropped_unrelated_finger_pairs} finger-vs-unrelated pair(s).",
        flush=True,
    )

    return CollisionPipeline(
        model,
        broad_phase="explicit",
        shape_pairs_filtered=shape_pairs_filtered,
        rigid_contact_max=_PROXY_RIGID_CONTACT_MAX,
        # Match the current geometry while retaining VBD's previous hard-contact
        # multipliers; the warm start is required for a continuous carry grip.
        contact_matching="latest",
        contact_matching_pos_threshold=0.005,
        contact_matching_normal_dot_threshold=0.95,
        # Newton #3262's water-tight path generates contacts from a soft mesh's edges
        # and faces. This hose is a rigid-capsule rod graph, and SolverCoupledProxy
        # cannot yet harvest full-surface VBD forces, so enabling it here would add no
        # hose contacts and latest Newton intentionally rejects that combination.
        enable_rigid_soft_full_surface_contact=False,
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
_RBY1_TELEOP_POSTURE_JOINT_NAMES = [f"{side}_arm_joint_{index}" for side in ("left", "right") for index in range(1, 5)]
_RBY1_TELEOP_POSTURE = tuple(_RBY1_IK_INITIAL_JOINT_POS[joint_name] for joint_name in _RBY1_TELEOP_POSTURE_JOINT_NAMES)


##
# Scene
##


@configclass
class WaterhoseSceneCfg(InteractiveSceneCfg):
    """Cable + plug with the cable tail welded to a kinematic anchor; sky light and ground."""

    ### Static fridge body (the canonical Newton wrapper authors its socket SDF)
    fridge = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Fridge",
        spawn=sim_utils.UsdFileCfg(usd_path=_FRIDGE_USD, func=spawn_fridge_without_embedded_body_collision),
        init_state=AssetBaseCfg.InitialStateCfg(pos=FRIDGE_POS),
    )

    # Keep solver-specific collision surfaces around the canonical visual fridge. Only a 1.5 cm-radius
    # corridor at the socket is removed; robot and cable contact remain active everywhere else.
    fridge_robot_collision = AssetBaseCfg(
        prim_path="/World/envs/env_.*/FridgeRobotCollision",
        spawn=sim_utils.UsdFileCfg(usd_path=_FRIDGE_COLLISION_PROXY_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=FRIDGE_POS),
    )
    fridge_cable_collision = AssetBaseCfg(
        prim_path="/World/envs/env_.*/FridgeCableCollision",
        spawn=sim_utils.UsdFileCfg(usd_path=_FRIDGE_COLLISION_PROXY_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=FRIDGE_POS),
    )

    ### rby1df robot (28-DOF, fixed base). Gripper drives are force-limited so VBD contacts can stop the fingers.
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=_RBY1_USD, func=_spawn_rby1_with_fabric_compatible_geometry),
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
                effort_limit_sim=10000.0,
                armature=0.2,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[".*_gripper_finger_joint_1"],
                stiffness=_GRIPPER_DRIVER_STIFFNESS,
                damping=_GRIPPER_DRIVER_DAMPING,
                effort_limit_sim=_GRIPPER_DRIVER_EFFORT_LIMIT,
                armature=0.5,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*_gripper_(left|right)_finger_joint"],
                stiffness=_GRIPPER_FINGER_STIFFNESS,
                damping=_GRIPPER_FINGER_DAMPING,
                effort_limit_sim=_GRIPPER_FINGER_EFFORT_LIMIT,
                armature=0.5,
            ),
        },
    )

    ### Cable 1 (connector lumped into the head; tail welded to a kinematic anchor sphere)

    # Static target for the cable-tail weld. This demo intentionally runs one environment.
    anchor1 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Anchor1",
        spawn=sim_utils.SphereCfg(
            radius=0.001,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=ANCHOR_POS),
    )

    # The deformable cable, simulated as a Cosserat rod by the VBD solver. The stretch stiffness is
    # firm enough to hold the hose taut without making it behave like a rigid rod during plug motion.
    cable1 = WaterhoseCableObjectCfg(
        prim_path="/World/envs/env_.*/Cable1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_CABLE1_USD,
            physics_material=NewtonCableMaterialCfg(
                # Current VBD uses absolute physical damping units. Keep the stretch stiffness and low
                # density of Newton's proxy-coupled pick/place cable, while retaining enough bending
                # resistance for this long, tail-anchored hose to move as a hose instead of a loose chain.
                stretch_stiffness=1.0e6,
                stretch_damping=1.0e-2,
                bend_stiffness=3.0e-1,
                bend_damping=2.0e-2,
                density=100.0,
            ),
        ),
        init_state=WaterhoseCableObjectCfg.InitialStateCfg(
            pos=FRIDGE_POS,
        ),
        connector_usd_path=_PLUG_USD,
        connector_mass=CONNECTOR_MASS,
        connector_local_pos=CONNECTOR_LOCAL_POS,
        connector_local_quat=CONNECTOR_LOCAL_QUAT_XYZW,
        connector_shape_label=_CONNECTOR_SHAPE_TOKEN,
        connector_ke=_VBD_CONTACT_STIFFNESS,
        connector_kd=_VBD_CONTACT_DAMPING,
        connector_mu=_VBD_DEFAULT_SHAPE_FRICTION,
        connector_margin=0.0,
        # The socket clearance is only 15 mm in radius. A 10 mm broad contact gap would make the
        # 7.3 mm-radius connector contact that wall even while perfectly centred, so use a local
        # millimetre-scale gap for the connector mesh.
        connector_gap=0.001,
        tail_anchor_prim_path="/World/envs/env_.*/Anchor1",
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            # Local HDR (bundled) -- no S3/Nucleus fetch. See _SKY_HDR.
            texture_file=_SKY_HDR,
        ),
    )

    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
        # Local grid-ground USD (bundled) -- overrides GroundPlaneCfg's default S3/Nucleus usd_path.
        spawn=GroundPlaneCfg(usd_path=_GROUND_USD),
    )


##
# MDP overrides
##


@configclass
class ActionsCfg:
    """Joint-position control of the rby1df robot.

    Actions are offsets from the default joint pose (``use_default_offset=True``), so a
    zero action holds the rest configuration. The right gripper is commanded explicitly
    through driver and follower finger joints because the imported USD mimic signs are
    disabled for Newton.
    """

    body_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["torso_joint_.*", "left_arm_joint_.*", "right_arm_joint_.*", "head_joint_.*"],
        scale=0.1,
        use_default_offset=True,
    )
    gripper_action = WaterhoseGripperPositionActionCfg(
        asset_name="robot",
        joint_names=_RIGHT_GRIPPER_JOINT_NAMES,
        open_command_expr=dict(_RIGHT_GRIPPER_OPEN_COMMAND),
        close_command_expr=dict(_RIGHT_GRIPPER_CLOSE_COMMAND),
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

    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    reset_cable = EventTerm(
        func=reset_cable_to_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("cable1")},
    )


@configclass
class RewardsCfg:
    """Joint-regularization penalties only (velocity, torque, acceleration).

    The demo is scripted/teleoperated, so no reach-or-insert reward is learned; these terms only keep
    the motion smooth. A real training task would add task-progress rewards here.
    """

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-2)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1e-4)


@configclass
class TerminationsCfg:
    """Termination terms for the waterhose task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=plug_inserted_in_socket,
        params={
            "cable_cfg": SceneEntityCfg("cable1"),
            "socket_pos": SOCKET_MOUTH_POS,
            "socket_quat": SOCKET_ROT_QUAT_XYZW,
            "radial_threshold": 0.001,
            "connector_tip_offset": CONNECTOR_TIP_LOCAL_POS,
            "min_depth": SOCKET_SEATED_TIP_DEPTH - 0.004,
            "max_depth": SOCKET_SEATED_TIP_DEPTH + 0.004,
            "alignment_threshold": 0.95,
        },
    )


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
        substeps = _env_positive_int("WATERHOSE_SUBSTEPS", 10)
        vbd_iterations = _env_positive_int("WATERHOSE_VBD_ITERS", 20)

        # general settings
        self.decimation = 1
        self.episode_length_s = 1.0

        # simulation settings
        self.sim.dt = 1 / 100.0
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.81)

        visualizer_view = dict(
            eye=_SCENE_CAMERA_EYE,
            lookat=_SCENE_CAMERA_LOOKAT,
            window_width=1600,
            window_height=1600,
        )
        self.sim.visualizer_cfgs = [KitVisualizerCfg(**visualizer_view), NewtonVisualizerCfg(**visualizer_view)]

        # ManagerBasedEnv's ViewportCameraController applies ViewerCfg to every active visualizer.
        # Keep its pose aligned with the visualizer configs so startup order cannot change the view.
        self.viewer.eye = _SCENE_CAMERA_EYE
        self.viewer.lookat = _SCENE_CAMERA_LOOKAT
        self.viewer.origin_type = "world"
        self.viewer.resolution = (1600, 1000)

        # Resolution of `--video` recordings (independent of the on-screen visualizer windows above).
        self.video_recorder.window_width = 1600
        self.video_recorder.window_height = 1600

        # Coupled physics: MuJoCo-Warp (MJWarp) solves the articulated rby1df robot, VBD solves the
        # deformable cable, its compound connector head, and the welded tail anchor. The two are joined
        # by proxy coupling: the gripper bodies are mirrored as proxy bodies in the VBD solver so the
        # cable and connector collide against them. The solver entries route the canonical socket SDF
        # and task-local clearanced housing explicitly rather than importing the whole static scene.
        self.sim.physics = NewtonCfg(
            use_cuda_graph=True,
            # The plug must insert into the authored socket bore: convex-hull mesh
            # approximation would fill the bore (and the gripper-finger cavities).
            simplify_meshes=False,
            solver_cfg=CouplerProxyCfg(
                entries=[
                    CouplerEntryCfg(
                        name="mjc",
                        solver_cfg=MJWarpSolverCfg(
                            cone="elliptic",
                            ls_iterations=20,
                            integrator="implicitfast",
                            # Newton generates contacts against the explicit clearanced housing proxy;
                            # MuJoCo-Warp resolves those contacts without convexifying the mesh.
                            use_mujoco_contacts=False,
                            # Match the standalone demo's contact and constraint capacity.
                            nconmax=4096,
                            njmax=1024,
                        ),
                        # The gripper proxy should carry the selected fingers'
                        # authored mass/inertia. MJWarp's articulated effective
                        # inertia is appropriate for two-way generalized-force
                        # coupling, but makes this tiny compliant grasp resist
                        # the commanded wrist rotation.
                        use_solver_effective_mass=False,
                        bodies=[_ROBOT_BODY_PATTERN],
                        # Route only the robot's clearanced housing. Explicit selection excludes the
                        # canonical SDF socket from the robot view.
                        include_body_shapes=True,
                        shape_label_patterns=[_FRIDGE_ROBOT_COLLISION_PATTERN],
                    ),
                    CouplerEntryCfg(
                        name="vbd",
                        # Contact/solver recipe matched to Newton's franka_cable_ik_pick_place reference
                        # grasp: HARD (augmented-Lagrangian) contacts enforce non-penetration of the plug
                        # against the gripper and socket, paired with a gentle penalty ramp (beta=1e2) and
                        # enough VBD iterations for the contact duals to converge. The cable-tail anchor
                        # stays a soft fixed joint; the connector itself is part of segment 0 and needs no weld.
                        # iterations x num_substeps (below) is the primary performance/stability
                        # trade-off. The validated high-fidelity default is 20 x 10; use
                        # WATERHOSE_VBD_ITERS / WATERHOSE_SUBSTEPS only for explicit experiments.
                        solver_cfg=VBDSolverCfg(
                            iterations=vbd_iterations,
                            friction_epsilon=0.1,
                            rigid_contact_hard=True,
                            rigid_joint_hard=False,
                            rigid_avbd_beta=1.0e2,
                            rigid_avbd_gamma=0.999,
                            # Warm-start the hard-contact multipliers across substeps. Without history,
                            # the tiny connector grip restarts from zero each refresh and is lost under
                            # the cable-tail load during LIFT/CARRY.
                            rigid_contact_history=True,
                            rigid_contact_k_start=1.0e3,
                            # The cable/gripper/socket contacts can exceed 1000 contacts on a single
                            # body, so size the per-body contact buffer well above the default.
                            rigid_body_contact_buffer_size=4096,
                            # Tail-anchor weld penalty. The connector has no joint: its mesh and inertia
                            # are lumped directly into the cable-head body.
                            rigid_joint_linear_ke=1.0e9,
                            rigid_joint_angular_ke=1.0e9,
                            rigid_joint_linear_k_start=1.0e4,
                            rigid_joint_angular_k_start=1.0e1,
                            rigid_joint_linear_kd=0.0,
                            rigid_joint_angular_kd=0.0,
                        ),
                        bodies=[_CABLE_BODY_PATTERN],
                        # VBD owns the canonical socket SDF plus the locally clearanced housing copy.
                        include_body_shapes=True,
                        include_static_shapes=False,
                        shape_label_patterns=[SOCKET_COLLISION_MESH_PATTERN, _FRIDGE_CABLE_COLLISION_PATTERN],
                    ),
                ],
                proxies=[
                    CouplerProxyMappingCfg(
                        source="mjc",
                        destination="vbd",
                        bodies=[_GRIPPER_FINGER_BODY_PATTERN],
                        # Drive the proxy bodies from the source solver's end-of-step pose and
                        # velocity together, so the gripper proxies stay consistent as the
                        # fingers close on the plug.
                        mode="staggered",
                        mass_scale=_GRIPPER_PROXY_MASS_SCALE,
                        collision_pipeline=_make_proxy_collision_pipeline,
                        collide_interval=1,
                    )
                ],
                iterations=1,
                model_cfg=NewtonModelCfg(
                    soft_contact_mu=_VBD_SOFT_CONTACT_FRICTION,
                ),
            ),
            num_substeps=substeps,
            # Refresh the robot/fridge and proxy contact manifolds at the same 1 ms cadence as the
            # high-fidelity solver substeps. Reusing one 10 ms-old manifold produces visible
            # penetration and a late, bouncy correction when a finger first reaches the housing.
            collision_decimation=1,
            # This explicit capacity is visible before SolverVBD construction. Keep it
            # at least as large as the proxy pipeline so rigid-contact history can be
            # allocated before CUDA graph capture on current Newton.
            collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=65536),
            default_shape_cfg=NewtonShapeCfg(
                ke=_VBD_CONTACT_STIFFNESS,
                kd=_VBD_CONTACT_DAMPING,
                mu=_VBD_DEFAULT_SHAPE_FRICTION,
            ),
        )


@configclass
class WaterhoseIkActionsCfg:
    """Absolute right end-effector pose plus normalized right-gripper action for scripted demos.

    Base (DiffIK) action space, retained for the unregistered :class:`WaterhoseIkEnvCfg` reference
    variant and as the shared gripper-action parent for the Newton-IK subclasses below.
    """

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
            pos=RIGHT_GRIPPER_EE_FRAME_POS,
            rot=RIGHT_GRIPPER_EE_FRAME_QUAT_XYZW,
        ),
    )
    gripper_action = WaterhoseGripperPositionActionCfg(
        asset_name="robot",
        joint_names=_RIGHT_GRIPPER_JOINT_NAMES,
        open_command_expr=dict(_RIGHT_GRIPPER_OPEN_COMMAND),
        close_command_expr=dict(_RIGHT_GRIPPER_CLOSE_COMMAND),
    )


@configclass
class WaterhoseNewtonIkActionsCfg(WaterhoseIkActionsCfg):
    """Newton-native absolute right end-effector pose plus normalized right-gripper action.

    The IK is a multi-body objective solve: the command-driven right end-effector pose objective is
    followed by two hold objectives (left gripper, torso hip) whose target poses the scripted state
    machine writes into their action slices each step. The holds keep the shared torso/left joints
    from swinging the uncommanded bodies while the right arm tracks its target. Action layout:
    ``[right_ee pose(7), left_hold pose(7), torso_hold pose(7)]`` -- root-frame positions plus
    ``(x, y, z, w)`` quaternions -- followed by the gripper action.
    """

    arm_action = NewtonInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["torso_joint_.*", "left_arm_joint_.*", "right_arm_joint_.*"],
        controller=NewtonIKSolverCfg(optimizer="lm", jacobian_mode="analytic", iterations=24),
        objectives=[
            NewtonIKPoseObjectiveCfg(
                name="right_ee",
                body_name="right_gripper_base",
                body_offset_pos=RIGHT_GRIPPER_EE_FRAME_POS,
                body_offset_rot=RIGHT_GRIPPER_EE_FRAME_QUAT_XYZW,
                command_type="pose",
                use_relative_mode=False,
            ),
            NewtonIKPoseObjectiveCfg(
                name="left_hold",
                body_name="left_gripper_base",
                command_type="pose",
                use_relative_mode=False,
                position_weight=1.0,
                rotation_weight=1.0,
            ),
            NewtonIKPoseObjectiveCfg(
                name="torso_hold",
                body_name="torso_hip_yaw",
                command_type="pose",
                use_relative_mode=False,
                position_weight=50.0,
                rotation_weight=50.0,
            ),
            NewtonIKJointLimitObjectiveCfg(weight=0.1),
        ],
    )


@configclass
class WaterhoseNewtonRelativeIkActionsCfg(WaterhoseIkActionsCfg):
    """Newton-native relative end-effector teleop with the torso + left gripper pinned.

    Same multi-body IK as the scripted demo (:class:`WaterhoseNewtonIkActionsCfg`) -- a right
    end-effector objective plus ``left_hold``/``torso_hold`` objectives over the full upper-body joint
    set -- so the torso and left gripper stay pinned while the right arm tracks the operator. The only
    differences from the demo: the right end-effector is driven by *relative* deltas (the operator
    streams pose deltas, applied in the end-effector frame) instead of absolute waypoints, and the
    hold targets are captured once at teleop start instead of written each step by the state machine.
    The action class (:class:`WaterhoseTeleopPinnedNewtonIkAction`) exposes only the right-ee slice as
    the action dimension and fills the hold slices internally, so it matches what the teleop pipeline
    emits.
    """

    arm_action = NewtonInverseKinematicsActionCfg(
        class_type="isaaclab_tasks.contrib.waterhose.mdp.actions:WaterhoseTeleopPinnedNewtonIkAction",
        asset_name="robot",
        joint_names=["torso_joint_.*", "left_arm_joint_.*", "right_arm_joint_.*"],
        controller=NewtonIKSolverCfg(optimizer="lm", jacobian_mode="analytic", iterations=24),
        clip={
            "right_ee/x": (-_TELEOP_MAX_EE_TRANSLATION_DELTA, _TELEOP_MAX_EE_TRANSLATION_DELTA),
            "right_ee/y": (-_TELEOP_MAX_EE_TRANSLATION_DELTA, _TELEOP_MAX_EE_TRANSLATION_DELTA),
            "right_ee/z": (-_TELEOP_MAX_EE_TRANSLATION_DELTA, _TELEOP_MAX_EE_TRANSLATION_DELTA),
            "right_ee/roll": (-_TELEOP_MAX_EE_ROTATION_DELTA, _TELEOP_MAX_EE_ROTATION_DELTA),
            "right_ee/pitch": (-_TELEOP_MAX_EE_ROTATION_DELTA, _TELEOP_MAX_EE_ROTATION_DELTA),
            "right_ee/yaw": (-_TELEOP_MAX_EE_ROTATION_DELTA, _TELEOP_MAX_EE_ROTATION_DELTA),
        },
        objectives=[
            NewtonIKPoseObjectiveCfg(
                name="right_ee",
                body_name="right_gripper_base",
                body_offset_pos=RIGHT_GRIPPER_EE_FRAME_POS,
                body_offset_rot=RIGHT_GRIPPER_EE_FRAME_QUAT_XYZW,
                command_type="pose",
                use_relative_mode=True,
            ),
            NewtonIKPoseObjectiveCfg(
                name="left_hold",
                body_name="left_gripper_base",
                command_type="pose",
                use_relative_mode=False,
                position_weight=1.0,
                rotation_weight=1.0,
            ),
            NewtonIKPoseObjectiveCfg(
                name="torso_hold",
                body_name="torso_hip_yaw",
                command_type="pose",
                use_relative_mode=False,
                position_weight=50.0,
                rotation_weight=50.0,
            ),
            NewtonIKJointLimitObjectiveCfg(weight=0.1),
        ],
    )
    gripper_action = WaterhoseGripperPositionActionCfg(
        asset_name="robot",
        joint_names=_RIGHT_GRIPPER_JOINT_NAMES,
        open_command_expr=dict(_RIGHT_GRIPPER_OPEN_COMMAND),
        close_command_expr=dict(_RIGHT_GRIPPER_CLOSE_COMMAND),
        max_joint_delta_per_step=_TELEOP_MAX_GRIPPER_JOINT_DELTA,
    )


@configclass
class WaterhoseNewtonBimanualIkActionsCfg(WaterhoseIkActionsCfg):
    """Absolute Apple Vision Pro wrist poses for both complete RBY1 arm chains.

    IsaacTeleop outputs two poses in the robot-root frame. Newton IK tracks the right and left
    gripper-base wrist frames using the torso and every joint in both arms, while an internal hold
    objective keeps the shared torso stable. The scripted insertion action keeps using its pad/contact
    offset; wrist teleoperation intentionally pivots at the physical robot wrist. External action
    layout: ``[right pose(7), left pose(7)]`` followed by the right gripper scalar.
    """

    arm_action = NewtonInverseKinematicsActionCfg(
        class_type="isaaclab_tasks.contrib.waterhose.mdp.actions:WaterhoseBimanualTeleopNewtonIkAction",
        asset_name="robot",
        joint_names=["torso_joint_.*", "left_arm_joint_.*", "right_arm_joint_.*"],
        controller=NewtonIKSolverCfg(optimizer="lm", jacobian_mode="analytic", iterations=24),
        objectives=[
            NewtonIKPoseObjectiveCfg(
                name="right_ee",
                body_name="right_gripper_base",
                body_offset_pos=(0.0, 0.0, 0.0),
                body_offset_rot=(0.0, 0.0, 0.0, 1.0),
                command_type="pose",
                use_relative_mode=False,
            ),
            NewtonIKPoseObjectiveCfg(
                name="left_ee",
                body_name="left_gripper_base",
                body_offset_pos=(0.0, 0.0, 0.0),
                body_offset_rot=(0.0, 0.0, 0.0, 1.0),
                command_type="pose",
                use_relative_mode=False,
            ),
            NewtonIKPoseObjectiveCfg(
                name="torso_hold",
                body_name="torso_hip_yaw",
                command_type="pose",
                use_relative_mode=False,
                position_weight=50.0,
                rotation_weight=50.0,
            ),
            # Both arms have one redundant elbow-swivel DoF after their wrist
            # poses are fixed. A weak shoulder/elbow reference resolves that
            # null space toward the task's natural bent-arm reset posture while
            # leaving the three wrist joints governed by the pose objectives.
            NewtonIKJointPostureObjectiveCfg(
                joint_names=_RBY1_TELEOP_POSTURE_JOINT_NAMES,
                target_positions=_RBY1_TELEOP_POSTURE,
                weight=0.01,
            ),
            NewtonIKJointLimitObjectiveCfg(weight=0.1),
        ],
    )
    gripper_action = WaterhoseGripperPositionActionCfg(
        asset_name="robot",
        joint_names=_RIGHT_GRIPPER_JOINT_NAMES,
        open_command_expr=dict(_RIGHT_GRIPPER_OPEN_COMMAND),
        close_command_expr=dict(_RIGHT_GRIPPER_CLOSE_COMMAND),
        max_joint_delta_per_step=_TELEOP_MAX_GRIPPER_JOINT_DELTA,
    )


@configclass
class WaterhoseIkEnvCfg(WaterhoseEnvCfg):
    """Coupled waterhose variant with the base DiffIK action space (unregistered reference).

    Not gym-registered; kept as a reference for the differential-IK action variant. It still inherits
    the full MJWarp+VBD coupled solver from :class:`WaterhoseEnvCfg`. The registered demo/teleop tasks
    use the Newton-IK variants (:class:`WaterhoseProxyIkEnvCfg`, :class:`WaterhoseProxyTeleopEnvCfg`).
    """

    actions: WaterhoseIkActionsCfg = WaterhoseIkActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 30.0
        self.scene.robot.init_state.joint_pos = _RBY1_IK_INITIAL_JOINT_POS


@configclass
class WaterhoseProxyIkEnvCfg(WaterhoseEnvCfg):
    """Waterhose task with Newton proxy coupling and the scripted Newton-IK action space.

    The right arm tracks an absolute end-effector pose with Newton's multi-body-objective IK. The
    torso and left gripper share the IK joint set but are pinned by hold objectives, so the
    uncommanded bodies stay put while the right arm reaches the coaxial-grasp orientations.
    """

    actions: WaterhoseNewtonIkActionsCfg = WaterhoseNewtonIkActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 30.0
        self.scene.robot.init_state.joint_pos = _RBY1_IK_INITIAL_JOINT_POS
        # No teleop on this (scripted) env: it uses the multi-body IK action (right_ee + left/torso
        # holds = 22 dims) that the scripted state machine fills. Teleop is the dedicated
        # ``Isaac-Waterhose-Coupled-Teleop-v0`` task (``WaterhoseProxyTeleopEnvCfg``), whose
        # two-wrist action matches the bimanual Apple Vision Pro pipeline.


@configclass
class WaterhoseAdmmIkEnvCfg(WaterhoseProxyIkEnvCfg):
    """Waterhose task variant using Newton ADMM contact coupling instead of proxy bodies."""

    def __post_init__(self) -> None:
        super().__post_init__()
        proxy_solver_cfg = self.sim.physics.solver_cfg
        if not isinstance(proxy_solver_cfg, CouplerProxyCfg):
            raise TypeError("WaterhoseAdmmIkEnvCfg expects the base environment to configure a CouplerProxyCfg.")
        # ADMM contact coupling uses a stiff penalty (rho) with a small proximal term (gamma) and
        # Baumgarte stabilization, plus frame-to-frame contact matching that warm-starts the ADMM
        # dual so the grasp contact stays consistent as the gripper closes and carries the plug.
        # The finite-mass proxy task (Isaac-Waterhose-Coupled-v0) is the primary demo path; this
        # ADMM variant is provided for solver comparison.
        entries = copy.deepcopy(proxy_solver_cfg.entries)
        for entry in entries:
            if isinstance(entry.solver_cfg, VBDSolverCfg):
                # ADMM owns cross-entry contact matching. The VBD entry's local pipeline has no
                # matching buffers, so its independent rigid-contact history must stay disabled.
                entry.solver_cfg.rigid_contact_history = False
        self.sim.physics.solver_cfg = CouplerAdmmCfg(
            model_cfg=copy.deepcopy(proxy_solver_cfg.model_cfg),
            entries=entries,
            iterations=5,
            rho=200.0,
            gamma=0.001,
            baumgarte=0.5,
            rigid_contact_matching="latest",
            contact_matching_force_scale=0.9,
            contact_pairs=[("mjc", "vbd")],
        )


@configclass
class WaterhoseProxyTeleopEnvCfg(WaterhoseProxyIkEnvCfg):
    """Waterhose task variant for bimanual Apple Vision Pro hand teleoperation."""

    actions: WaterhoseNewtonBimanualIkActionsCfg = WaterhoseNewtonBimanualIkActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.xr = XrCfg(
            anchor_pos=(0.0, 0.9, -1),
            # XrCfg quaternions are xyzw. Rotate the simulation 180 deg around world up so the
            # headset initially faces the fridge.
            anchor_rot=(0.0, 0.0, 1.0, 0.0),
        )
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: build_waterhose_bimanual_teleop_pipeline()[0],
            sim_device=self.sim.device,
            xr_cfg=self.xr,
            app_name="WaterhoseTeleop",
            target_frame_prim_path=_ROBOT_BASE_PRIM_PATH_ENV0,
            teleoperation_active_default=True,
            control_channel_uuid=None,
        )
        # Native single-arm devices emit seven actions and therefore do not match this bimanual
        # 15D contract. Keep those devices on the legacy relative action configuration instead of
        # silently padding or freezing one arm here.
        self.teleop_devices = DevicesCfg(devices={})


@configclass
class WaterhoseMimicEnvCfg(WaterhoseProxyTeleopEnvCfg, MimicEnvCfg):
    """Waterhose teleop task variant with Isaac Lab Mimic data-generation metadata."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.datagen_config.name = "waterhose_coupled_teleop_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_relative = True
        self.datagen_config.generation_joint_pos = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        self.subtask_configs = {
            "right": [
                SubTaskConfig(
                    object_ref="socket",
                    subtask_term_signal=None,
                    subtask_term_offset_range=(0, 0),
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 3},
                    action_noise=0.003,
                    num_interpolation_steps=3,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                    description="Insert the waterhose connector into the fridge socket",
                )
            ]
        }

    def make_recorder_manager_cfg(self):
        """Return recorder terms that make teleop recordings usable by Isaac Lab Mimic."""

        from .waterhose_mimic_env import WaterhoseMimicRecorderManagerCfg

        return WaterhoseMimicRecorderManagerCfg()
