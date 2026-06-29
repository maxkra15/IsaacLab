# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waterhose manipulation environment.

An RBY1DF robot grasps a hose connector and inserts it into a fridge socket under a Newton coupled
solver (MuJoCo-Warp rigid robot + VBD deformable hose). The MDP uses standard
:mod:`isaaclab.envs.mdp` terms plus the task-local action/termination terms in :mod:`.mdp`. See the
package ``README.md`` for the task list, run commands, and the batchability/performance summary.
"""

from __future__ import annotations

import logging
import math
import os

from isaaclab_newton.envs.mdp.actions.newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg
from isaaclab_newton.ik.newton_ik_objectives_cfg import NewtonIKJointLimitObjectiveCfg, NewtonIKPoseObjectiveCfg
from isaaclab_newton.ik.newton_ik_solver_cfg import NewtonIKSolverCfg
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
from isaaclab_newton.sim.spawners.materials.physics_materials_cfg import NewtonCableMaterialCfg
from isaaclab_teleop import IsaacTeleopCfg, XrCfg
from isaaclab_visualizers.kit.kit_visualizer_cfg import KitVisualizerCfg
from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonVisualizerCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
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

from isaaclab_contrib.cable.cable_object_cfg import CableAttachmentCfg, CableObjectCfg
from isaaclab_contrib.deformable.newton_manager_cfg import (
    CoupledNewtonCfg,
    NewtonModelCfg,
    VBDSolverCfg,
)

from .geometry import (
    ANCHOR_POS,
    CABLE_HEAD_TO_PLUG_ORIGIN_LOCAL_Z,
    FRIDGE_FLOOR_COLLISION_TOKEN,
    # Retained for the intentionally disabled collision box below.
    FRIDGE_FLOOR_POS,  # noqa: F401
    FRIDGE_FLOOR_SIZE,  # noqa: F401
    FRIDGE_HOUSING_COLLISION_MESH_PATTERN,
    FRIDGE_POS,
    RIGHT_GRIPPER_EE_FRAME_POS,
    RIGHT_GRIPPER_EE_FRAME_QUAT_XYZW,
    SOCKET_COLLISION_MESH_SUFFIX,
    SOCKET_MOUTH_POS,
    SOCKET_ROT_QUAT_XYZW,
)
from .mdp.actions import WaterhoseGripperPositionActionCfg
from .mdp.events import reset_cable_to_default
from .mdp.terminations import plug_inserted_in_socket
from .teleop import WaterhoseSpaceMouseCfg

# Best-practices IsaacTeleop pipelines. The previous known-working variants are preserved in
# ``teleop_pipelines_legacy`` (same function names); switch this import to that module to
# restore the exact prior XR behavior if a refactor here regresses the live session.
from .teleop_pipelines import build_waterhose_relative_teleop_pipeline


def _resolve_waterhose_assets_dir(module_dir: str) -> str:
    env_assets_dir = os.environ.get("WATERHOSE_ASSETS_DIR")
    if env_assets_dir:
        return env_assets_dir

    setup_assets_dir = os.path.abspath(
        os.path.join(module_dir, "..", "..", "..", "..", "isaaclab_assets", "data", "WaterhoseDemo")
    )
    if os.path.isdir(setup_assets_dir):
        return setup_assets_dir

    return os.path.join(module_dir, "assets")


WATERHOSE_ASSETS_DIR = _resolve_waterhose_assets_dir(os.path.dirname(os.path.abspath(__file__)))

_FRIDGE_USD = os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "fridge_waterhose.usda")
_RBY1_USD = os.path.join(WATERHOSE_ASSETS_DIR, "rby1df", "rby1df_waterhose.usda")
_PLUG_USD = os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "cable", "plug.usda")
_CABLE1_USD = os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "cable", "cable001.usda")
# Cosmetic scene assets bundled locally (sky dome HDR + grid ground USD) so the demo needs NO external
# (S3/Nucleus) connection at runtime. Upstream these default to the Isaac cloud asset root
# (``ISAAC_NUCLEUS_DIR`` -> AWS S3); we ship local copies in the asset bundle instead. The ground USD
# uses relative texture paths under ``ground/Materials/Textures`` and the core ``OmniPBR.mdl`` (resolved
# locally), so it is fully self-contained offline. See docs/waterhose_robot_demo.md (offline assets).
_SKY_HDR = os.path.join(WATERHOSE_ASSETS_DIR, "skies", "kloofendal_43d_clear_puresky_4k.hdr")
_GROUND_USD = os.path.join(WATERHOSE_ASSETS_DIR, "ground", "default_environment.usd")

# Initial visualizer cameras, defined in the task config so the --visualizer/--viz path uses
# these starting views. KitVisualizer uses eye/lookat; the Kit lookat is one meter along the
# camera-local -Z from the authored Kit transform translate=(-0.9, 0.6, 0.3),
# rotateXYZ=(73.32259, 0, -112.30437).
_KIT_CAMERA_EYE = (-0.9, 0.6, 0.3)
_KIT_CAMERA_LOOKAT = (-0.013736291, 0.236437794, 0.013017143)
_NEWTON_CAMERA_EYE = (-2.55, -7.1, 2.3)
_NEWTON_CAMERA_LOOKAT = (0.55, -0.42, 0.9)
_ROBOT_BASE_PRIM_PATH_ENV0 = "/World/envs/env_0/Robot/Geometry/origin"


_RBY1_GRIPPER_MIMIC_JOINT_TOKENS = (
    "gripper_left_finger_joint",
    "gripper_right_finger_joint",
)
_GRIPPER_PAD_STATIC_FRICTION = 1.5
_GRIPPER_PAD_DYNAMIC_FRICTION = 1.0
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
# the plug stick-slips jerkily against the bore wall, so it is kept moderate (the gripper override
# below keeps the grip firm regardless). Lower it further for a smoother push, raise it for more
# seated-plug retention after release.
_VBD_CONTACT_STIFFNESS = 1.0e4
_VBD_CONTACT_DAMPING = 1.0e-1
_VBD_DEFAULT_SHAPE_FRICTION = 0.5
_VBD_SOFT_CONTACT_FRICTION = 0.6
# Gripper-proxy contact friction. The task PUSHES the plug into a tight socket bore, so the grip
# needs enough tangential hold that the bore insertion resistance does not slide the gripper off the
# plug during INSERT. mu=1.0 (the reference pick-and-place value) is marginal here; weld-grade mu
# (~1e3) holds but, combined with the hard contacts, over-constrains the seated plug and oscillates
# it at HOLD_INSERTED. This moderate value gives a firm grip with margin against both failure modes.
# Tune up if the grip still slips during INSERT, down if the seated plug flips at HOLD_INSERTED.
_VBD_GRIPPER_PROXY_FRICTION = 2.0e1
_VBD_GRIPPER_PROXY_MARGIN = 0.001
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


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


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


# Connector-housing collision: the welded housing mesh is a world-static collider SHARED by both coupled
# entries, so the robot collides it directly in MJWarp AND the cable collides it in VBD -- one mesh, no
# duplication. The MJWarp entry lists it explicitly (so the robot collides only the housing, not the
# socket bore -- the gripper must not fight insertion); the VBD entry's empty shape list auto-includes it
# (shared visibility). The socket is collided only by the cable (plug insertion). Set
# WATERHOSE_FRIDGE_COLLISION=0 to drop the housing contact: ``_disable_fridge_body_collision`` clears the
# housing mesh's collide flags for both sides (~1.5x faster at scale, removing the hose<->body soft
# contacts) while leaving the socket so the plug still inserts.
_FRIDGE_COLLISION = _env_flag("WATERHOSE_FRIDGE_COLLISION", True)


# Optional texture-SDF collision for the fridge's embedded socket collider. The socket is authored as
# a plain triangle-mesh (BVH) collider; ``spawn_fridge_with_socket_sdf`` upgrades it to this texture-SDF
# only when WATERHOSE_SOCKET_SDF is enabled (the flag is the single source of truth -- the SDF is no
# longer baked into the USD). An SDF gives the bore a smooth, correctly-signed gradient field that
# guides the plug tip during insertion, at the cost of a per-env SDF build at startup; the plain mesh is
# faster to start and the scripted demo inserts on it (the state machine drives the pose, so it does not
# rely on the SDF gradient). Resolution 128 resolves the thin (~3 mm) bore wall. Default OFF; set
# WATERHOSE_SOCKET_SDF=1 for the smoother high-fidelity contact.
_SOCKET_SDF_COLLISION = sim_utils.NewtonSDFCollisionPropertiesCfg(
    sdf_max_resolution=128,
    sdf_narrow_band_inner=0.004,
    sdf_narrow_band_outer=0.006,
    sdf_texture_format="float32",
    sdf_padding=0.001,
    hydroelastic_enabled=False,
)
# The socket contact stiffness comes from the model-wide ``NewtonModelCfg.shape_material_*`` fill
# below: those fields fill the whole model at build time, so per-shape contact materials authored
# on the socket via USD attributes or bound physics materials would be overwritten and have no effect.


@clone
def spawn_fridge_with_socket_sdf(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the fridge and give its embedded socket collider Newton texture-SDF collision."""
    from pxr import Usd, UsdPhysics

    prim = spawn_from_usd.__wrapped__(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    if not _env_flag("WATERHOSE_SOCKET_SDF", False):
        return prim
    stage = prim.GetStage()
    modified = 0
    for child in Usd.PrimRange(prim):
        path = child.GetPath().pathString
        if SOCKET_COLLISION_MESH_SUFFIX not in path:
            continue
        if child.GetTypeName() == "Mesh" and child.HasAPI(UsdPhysics.CollisionAPI):
            sim_schemas.modify_collision_properties(path, _SOCKET_SDF_COLLISION, stage=stage)
            modified += 1
    if modified == 0:
        raise RuntimeError(
            f"spawn_fridge_with_socket_sdf: no socket collision meshes matched "
            f"'{SOCKET_COLLISION_MESH_SUFFIX}' under {prim_path}; the socket would "
            "silently fall back to BVH mesh contacts."
        )
    print(
        f"[waterhose] Applied Newton texture-SDF collision to {modified} socket prim(s) under {prim_path}.", flush=True
    )
    return prim


_RIGHT_GRIPPER_OPEN_COMMAND = {
    "right_gripper_finger_joint_1": 0.09,
    "right_gripper_left_finger_joint": -0.045,
    "right_gripper_right_finger_joint": 0.045,
}
_RIGHT_GRIPPER_CLOSE_COMMAND = {
    # Close to 14.8 mm on the ~14.5 mm plug flange and grip through the contact margin band
    # rather than commanding geometric interference (which would drive the fingers through the plug).
    "right_gripper_finger_joint_1": 0.014,
    "right_gripper_left_finger_joint": -0.007,
    "right_gripper_right_finger_joint": 0.007,
}


def _disable_rby1df_gripper_mimic_constraints(_payload=None) -> None:
    """Disable imported RBY1 gripper mimic constraints before Newton finalizes the model."""

    from isaaclab_newton.physics import NewtonManager

    builder = getattr(NewtonManager, "_builder", None)
    if builder is None:
        raise RuntimeError("Newton builder is unavailable while disabling RBY1 gripper mimic constraints.")

    labels = getattr(builder, "constraint_mimic_label", None)
    enabled = getattr(builder, "constraint_mimic_enabled", None)
    if labels is None or enabled is None:
        raise RuntimeError("Newton builder is missing gripper mimic constraint arrays.")

    disabled = 0
    for index, label in enumerate(labels):
        if any(token in str(label) for token in _RBY1_GRIPPER_MIMIC_JOINT_TOKENS):
            enabled[index] = False
            disabled += 1

    if disabled == 0:
        raise RuntimeError("No RBY1 gripper mimic constraints matched the expected right/left finger joint labels.")

    logging.debug("Disabled %d RBY1 gripper mimic constraints for explicit finger control.", disabled)


def _register_rby1df_gripper_mimic_override() -> None:
    """Match Newton's waterhose examples by using explicit gripper drives, not mimic equality."""

    from isaaclab_newton.physics import NewtonManager

    from isaaclab.physics import PhysicsEvent

    NewtonManager.register_callback(
        _disable_rby1df_gripper_mimic_constraints,
        PhysicsEvent.MODEL_INIT,
        order=10,
        name="waterhose_disable_rby1df_gripper_mimics",
        wrap_weak_ref=False,
    )


# RBY1 link-label tokens used to identify the robot's own collision shapes (vs. the cable/plug/anchor
# bodies). The two right-gripper fingers are the only robot bodies that need to collide.
_RBY1_LINK_TOKENS = ("torso", "left_arm", "right_arm", "head", "left_gripper", "right_gripper")
_RIGHT_GRIPPER_FINGER_BODY_TOKENS = ("right_gripper_leftfinger", "right_gripper_rightfinger")

# Shape-label tokens of the fridge connector-housing COLLIDERS (welded body mesh, socket). Their
# VISIBLE flag is cleared at model-init so the Newton viewer draws them only under its "Collisions"
# toggle, not on top of the fridge visual mesh under "Visuals" (see ``_hide_fridge_collider_visuals``).
# The fridge visual mesh (``Cable008/Visuals``) is not listed, so it stays under "Visuals".
_FRIDGE_COLLIDER_SHAPE_TOKENS = ("Cable008_BodyCollision", "Cable008_SocketCollision", FRIDGE_FLOOR_COLLISION_TOKEN)
# Shape-label token of the connector-housing body mesh alone (NOT the socket). Used to gate the
# housing contact off via ``WATERHOSE_FRIDGE_COLLISION`` while leaving the socket so the plug still
# inserts (see ``_disable_fridge_body_collision``).
_FRIDGE_HOUSING_SHAPE_TOKEN = "Cable008_BodyCollision"

# Body-label suffix of the kinematic cable-tail anchor (a collision-free weld target).
_ANCHOR_BODY_TOKEN = "Anchor1"
# Body-label suffixes for merging the connector into the cable head: the plug donates its collision
# shape to the cable's head-segment body so the connector is rigidly part of the rod (no separate
# welded body, no soft weld to stretch). See ``_merge_plug_shape_into_cable_head``.
_PLUG_BODY_TOKEN = "Plug1"
_CABLE_HEAD_BODY_TOKEN = "Cable1/cable_edge_body_0"


def _restrict_rby1df_collision_to_right_gripper(_payload=None) -> None:
    """Keep collision only on the right-gripper fingers; clear it on the rest of the robot."""

    from isaaclab_newton.physics import NewtonManager
    from newton import ShapeFlags

    builder = getattr(NewtonManager, "_builder", None)
    if builder is None:
        raise RuntimeError("Newton builder is unavailable while restricting RBY1 collision.")

    collide_mask = int(ShapeFlags.COLLIDE_SHAPES) | int(ShapeFlags.COLLIDE_PARTICLES)
    cleared = 0
    for shape_id, body_id in enumerate(builder.shape_body):
        if body_id < 0:
            continue
        label = str(builder.body_label[body_id])
        is_robot_link = any(token in label for token in _RBY1_LINK_TOKENS)
        is_right_finger = any(label.endswith(token) for token in _RIGHT_GRIPPER_FINGER_BODY_TOKENS)
        if is_robot_link and not is_right_finger and (builder.shape_flags[shape_id] & collide_mask):
            builder.shape_flags[shape_id] &= ~collide_mask
            cleared += 1

    if cleared == 0:
        raise RuntimeError("No non-gripper RBY1 collision shapes matched; the robot link labels changed.")

    logging.debug("Restricted RBY1 collision to the right gripper (cleared %d non-finger shapes).", cleared)


def _disable_anchor_collision(_payload=None) -> None:
    """Clear the cable-tail anchor's collision flags so it acts as a pure weld target.

    The anchor keeps a minimal collider in USD so Newton imports it as a rigid body (the body the
    cable tail welds to), but it should never generate contacts. Clear both ``COLLIDE_SHAPES`` and
    ``COLLIDE_PARTICLES`` on its shapes so the 1 mm sphere produces no rigid or particle contact pairs
    against the cable, plug, socket, or robot.
    """

    from isaaclab_newton.physics import NewtonManager
    from newton import ShapeFlags

    builder = getattr(NewtonManager, "_builder", None)
    if builder is None:
        raise RuntimeError("Newton builder is unavailable while disabling the anchor collider.")

    collide_mask = int(ShapeFlags.COLLIDE_SHAPES) | int(ShapeFlags.COLLIDE_PARTICLES)
    cleared = 0
    for shape_id, body_id in enumerate(builder.shape_body):
        if body_id < 0:
            continue
        if str(builder.body_label[body_id]).endswith(_ANCHOR_BODY_TOKEN) and (
            builder.shape_flags[shape_id] & collide_mask
        ):
            builder.shape_flags[shape_id] &= ~collide_mask
            cleared += 1

    if cleared == 0:
        raise RuntimeError("No anchor collision shapes matched; the anchor body label changed.")

    logging.debug("Disabled cable-tail anchor collision (cleared %d shapes).", cleared)


def _merge_plug_shape_into_cable_head(_payload=None) -> None:
    """Re-parent the plug's collision shape onto the cable head-segment body before finalize.

    The connector is then rigidly part of the rod's head segment -- one body, no separate ``Plug1``
    body and no soft head->plug weld to stretch under the carry/insert drag. Each shape on the plug
    body is moved to the matching per-env cable head body (``.../Cable1/cable_edge_body_0``) with its
    world pose preserved (so the connector keeps its authored offset/orientation relative to the head
    node). Runs at MODEL_INIT while the builder is still mutable, like the collision callbacks above.
    """

    import warp as wp
    from isaaclab_newton.physics import NewtonManager

    builder = getattr(NewtonManager, "_builder", None)
    if builder is None:
        raise RuntimeError("Newton builder is unavailable while merging the plug into the cable head.")

    # Resolve per-env plug and cable-head bodies by their shared prim prefix.
    plug_bodies: dict[str, int] = {}
    head_bodies: dict[str, int] = {}
    for body_id, label in enumerate(builder.body_label):
        text = str(label)
        if text.endswith(_PLUG_BODY_TOKEN):
            plug_bodies[text[: -len(_PLUG_BODY_TOKEN)]] = body_id
        elif text.endswith(_CABLE_HEAD_BODY_TOKEN):
            head_bodies[text[: -len(_CABLE_HEAD_BODY_TOKEN)]] = body_id

    merged = 0
    for prefix, plug_body in plug_bodies.items():
        head_body = head_bodies.get(prefix)
        if head_body is None:
            raise RuntimeError(f"No cable head body matching plug prefix {prefix!r} while merging connector.")
        head_inv = wp.transform_inverse(builder.body_q[head_body])
        plug_world = builder.body_q[plug_body]
        for shape_id, body_id in enumerate(builder.shape_body):
            if body_id != plug_body:
                continue
            shape_world = wp.transform_multiply(plug_world, builder.shape_transform[shape_id])
            builder.shape_transform[shape_id] = wp.transform_multiply(head_inv, shape_world)
            builder.shape_body[shape_id] = head_body
            merged += 1

    if merged == 0:
        raise RuntimeError("No plug shapes matched to merge into the cable head; body labels changed.")
    logging.debug("Merged %d plug shape(s) onto the cable head segment.", merged)


def _hide_fridge_collider_visuals(_payload=None) -> None:
    """Clear the ``VISIBLE`` flag on the fridge connector-housing collider shapes (render-only).

    The fridge's housing colliders (the welded body mesh and the socket) are authored visible in the
    USD, so the Newton viewer otherwise draws them on top of the fridge visual mesh and they clutter the
    "Visuals" toggle. Clear their ``VISIBLE`` bit so they render only under the viewer's "Collisions"
    toggle, like the robot's colliders already do -- leaving the fridge visual mesh as the sole "Visuals"
    representation. Only ``ShapeFlags.VISIBLE`` is touched; ``COLLIDE_SHAPES``/``COLLIDE_PARTICLES`` are
    left as-is, so collision behavior is unchanged. The cable/plug keep their visuals (they have no
    separate render mesh -- the rod/plug geometry is what you watch being manipulated).
    """

    from isaaclab_newton.physics import NewtonManager
    from newton import ShapeFlags

    builder = getattr(NewtonManager, "_builder", None)
    if builder is None:
        raise RuntimeError("Newton builder is unavailable while hiding fridge collider visuals.")

    visible = int(ShapeFlags.VISIBLE)
    cleared = 0
    for shape_id, label in enumerate(builder.shape_label):
        if any(token in str(label) for token in _FRIDGE_COLLIDER_SHAPE_TOKENS) and (
            builder.shape_flags[shape_id] & visible
        ):
            builder.shape_flags[shape_id] &= ~visible
            cleared += 1

    if cleared == 0:
        raise RuntimeError("No fridge collider shapes matched to hide; the shape labels changed.")

    logging.debug("Cleared the VISIBLE flag on %d fridge collider shape(s) (collision-only).", cleared)


def _disable_fridge_body_collision(_payload=None) -> None:
    """Clear the connector-housing mesh's collide flags when ``WATERHOSE_FRIDGE_COLLISION`` is off.

    Both coupled entries auto-include the world-static fridge colliders (each entry's resolved shape
    list is empty, so Newton makes every ``shape_body < 0`` shape visible to it). That shared collider
    is what gives the robot AND the cable a direct contact against the single welded housing mesh. When
    the flag is off, drop that contact for both sides by clearing the housing mesh's ``COLLIDE_SHAPES``
    (robot/rigid) and ``COLLIDE_PARTICLES`` (cable/deformable) flags. The socket collider is untouched,
    so the plug still inserts -- the flag governs only the housing-body contact. No-op when the flag is
    on (the housing keeps its collide flags and both sides collide it).
    """

    if _FRIDGE_COLLISION:
        return

    from isaaclab_newton.physics import NewtonManager
    from newton import ShapeFlags

    builder = getattr(NewtonManager, "_builder", None)
    if builder is None:
        raise RuntimeError("Newton builder is unavailable while disabling the fridge body collision.")

    collide_mask = int(ShapeFlags.COLLIDE_SHAPES) | int(ShapeFlags.COLLIDE_PARTICLES)
    cleared = 0
    for shape_id, label in enumerate(builder.shape_label):
        if _FRIDGE_HOUSING_SHAPE_TOKEN in str(label) and (builder.shape_flags[shape_id] & collide_mask):
            builder.shape_flags[shape_id] &= ~collide_mask
            cleared += 1

    if cleared == 0:
        raise RuntimeError("No fridge housing collider shape matched to disable; the shape label changed.")

    logging.debug("Cleared COLLIDE flags on %d fridge housing shape(s) (WATERHOSE_FRIDGE_COLLISION=0).", cleared)


def _register_rby1df_collision_restriction() -> None:
    """Limit active colliders to what the task needs before Newton finalizes the model.

    Restrict the robot's colliders to the right gripper, turn the cable-tail anchor into a
    collision-free weld target, merge the connector shape onto the cable head segment, hide the
    fridge collider shapes from the viewer's "Visuals" toggle (they stay under "Collisions"), and
    optionally drop the housing contact when ``WATERHOSE_FRIDGE_COLLISION=0``.
    """

    from isaaclab_newton.physics import NewtonManager

    from isaaclab.physics import PhysicsEvent

    NewtonManager.register_callback(
        _restrict_rby1df_collision_to_right_gripper,
        PhysicsEvent.MODEL_INIT,
        order=10,
        name="waterhose_restrict_rby1df_collision",
        wrap_weak_ref=False,
    )
    NewtonManager.register_callback(
        _disable_anchor_collision,
        PhysicsEvent.MODEL_INIT,
        order=10,
        name="waterhose_disable_anchor_collision",
        wrap_weak_ref=False,
    )
    NewtonManager.register_callback(
        _merge_plug_shape_into_cable_head,
        PhysicsEvent.MODEL_INIT,
        order=10,
        name="waterhose_merge_plug_into_cable_head",
        wrap_weak_ref=False,
    )
    NewtonManager.register_callback(
        _hide_fridge_collider_visuals,
        PhysicsEvent.MODEL_INIT,
        order=10,
        name="waterhose_hide_fridge_collider_visuals",
        wrap_weak_ref=False,
    )
    NewtonManager.register_callback(
        _disable_fridge_body_collision,
        PhysicsEvent.MODEL_INIT,
        order=10,
        name="waterhose_disable_fridge_body_collision",
        wrap_weak_ref=False,
    )


def _make_proxy_collision_pipeline(model):
    """Build the destination-view collision pipeline used by Newton proxy coupling.

    The pipeline is scoped so the gripper finger proxies interact with the deformable connector ONLY,
    along two independent contact paths:

    * Soft (particle) path: ``COLLIDE_PARTICLES`` is cleared on the finger proxy shapes so the deformable
      hose (VBD particles) does NOT collide with -- and penetrate -- the fingers. This bit is generated
      over every particle x shape with no shape-pair filter, so only the flag can target it.
    * Rigid path: an explicit ``shape_pairs_filtered`` list keeps every non-finger pair (cable<->housing,
      plug<->socket, cable self-contacts, ...) but, for the fingers, keeps only finger<->connector. Without
      this the destination view auto-includes the world-static housing and socket, so the finger proxies
      would collide the fridge inside the VBD solve -- duplicating the robot<->housing contact the MJWarp
      entry owns and fighting the socket bore. The connector grip carries the firm tangential hold set by
      ``_VBD_GRIPPER_PROXY_FRICTION``.

    ``model`` here is the VBD destination view; this runs once at pipeline construction (after per-entry
    shape visibility is set), so it is CUDA-graph-safe.
    """
    import warp as wp
    from newton import CollisionPipeline, ShapeFlags

    body_label = model.body_label
    finger_bodies = {
        i for i, lbl in enumerate(body_label) if any(str(lbl).endswith(t) for t in _RIGHT_GRIPPER_FINGER_BODY_TOKENS)
    }
    if not finger_bodies:
        raise RuntimeError(
            "waterhose proxy pipeline: no right_gripper finger proxy bodies found to drop particle collision; "
            f"proxy view body labels did not end with any of {_RIGHT_GRIPPER_FINGER_BODY_TOKENS}."
        )
    shape_body = model.shape_body.numpy()
    flags = model.shape_flags.numpy().copy()
    particle_bit = int(ShapeFlags.COLLIDE_PARTICLES)
    dropped = 0
    for shape_id, body_id in enumerate(shape_body):
        if int(body_id) in finger_bodies and (int(flags[shape_id]) & particle_bit):
            flags[shape_id] &= ~particle_bit  # keep COLLIDE_SHAPES (the Plug1 grip); drop only the hose path
            dropped += 1
    if dropped == 0:
        raise RuntimeError(
            "waterhose proxy pipeline: no gripper finger shapes had COLLIDE_PARTICLES set "
            "(proxy view shape layout changed?); the hose-vs-finger penetration filter did not apply."
        )
    model.shape_flags = wp.array(flags, dtype=wp.int32, device=model.device)
    print(f"[waterhose] Disabled hose-vs-gripper particle collision on {dropped} finger proxy shape(s).", flush=True)

    # Scope the finger proxies to the GRIP ONLY. This destination-view pipeline otherwise collides the
    # finger proxies against every world-static collider auto-included into the VBD entry -- the welded
    # housing mesh and the socket bore -- so the gripper fights the fridge inside the VBD solve. That
    # duplicates the robot<->housing contact the MJWarp entry already owns and pushes the gripper off the
    # bore during insertion. Robot<->housing belongs to MJWarp; the proxy is only the grip on the
    # connector. Build an explicit pair list that keeps every non-finger pair (cable<->housing,
    # plug<->socket, cable self-contacts, ...) but, for the fingers, keeps only finger<->connector. The
    # connector collider is identified by SHAPE label: ``_merge_plug_shape_into_cable_head`` re-parents it
    # onto the cable head BODY (so it reads as a cable body), but its shape label still names the plug.
    import numpy as np

    shape_label = [str(x) for x in model.shape_label] if getattr(model, "shape_label", None) is not None else None
    if shape_label is None:
        raise RuntimeError("waterhose proxy pipeline: model has no shape_label; cannot scope the finger grip pairs.")
    base_pairs = getattr(model, "shape_contact_pairs", None)
    if base_pairs is None:
        raise RuntimeError(
            "waterhose proxy pipeline: model.shape_contact_pairs is None; the explicit broad phase needs it."
        )
    finger_shapes = {shape_id for shape_id, body_id in enumerate(shape_body) if int(body_id) in finger_bodies}
    grip_shapes = {shape_id for shape_id, label in enumerate(shape_label) if _PLUG_BODY_TOKEN.lower() in label.lower()}
    if not grip_shapes:
        raise RuntimeError(
            f"waterhose proxy pipeline: no connector collider shape found by label ({_PLUG_BODY_TOKEN!r}); "
            "the finger grip-pair filter cannot be built."
        )
    kept_pairs = []
    dropped_finger_pairs = 0
    for shape_a, shape_b in base_pairs.numpy().reshape(-1, 2):
        shape_a, shape_b = int(shape_a), int(shape_b)
        a_is_finger, b_is_finger = shape_a in finger_shapes, shape_b in finger_shapes
        if a_is_finger or b_is_finger:
            # Keep only a single-finger pair whose other shape is the connector grip target.
            other, single_finger = (shape_b, not b_is_finger) if a_is_finger else (shape_a, not a_is_finger)
            if not (single_finger and other in grip_shapes):
                dropped_finger_pairs += 1
                continue
        kept_pairs.append((shape_a, shape_b))
    shape_pairs_filtered = wp.array(
        np.asarray(kept_pairs, dtype=np.int32).reshape(-1, 2), dtype=wp.vec2i, device=model.device
    )
    print(
        f"[waterhose] Scoped gripper proxy to the connector grip: kept {len(kept_pairs)} contact pair(s), "
        f"dropped {dropped_finger_pairs} finger-vs-non-connector pair(s).",
        flush=True,
    )

    return CollisionPipeline(
        model,
        broad_phase="explicit",
        shape_pairs_filtered=shape_pairs_filtered,
        rigid_contact_max=30000,
        # "sticky" replays previous-frame contact geometry; that is useful for
        # hard-contact demos, but makes the hose feel glued to the fingers here.
        contact_matching="latest",
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

    ### Static fridge body (socket collider gets texture-SDF collision, see spawn func)
    fridge = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Fridge",
        spawn=sim_utils.UsdFileCfg(usd_path=_FRIDGE_USD, func=spawn_fridge_with_socket_sdf),
        init_state=AssetBaseCfg.InitialStateCfg(pos=FRIDGE_POS),
    )

    ### rby1df robot (28-DOF, fixed base). Gripper drives are force-limited so VBD contacts can stop the fingers.
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=_RBY1_USD),
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

    ### Cable 1 (graspable plug welded to the head; tail welded to a kinematic anchor sphere)

    plug1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plug1",
        spawn=sim_utils.UsdFileCfg(usd_path=_PLUG_USD),
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
    # The anchor is purely a weld target. It keeps a minimal collider so Newton imports
    # it as a rigid body (the body the cable tail welds to), but the collision flags are
    # cleared at model-init (see ``_disable_anchor_collision``) so it never generates
    # contact pairs against the cable, plug, or robot.
    anchor1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Anchor1",
        spawn=sim_utils.SphereCfg(
            radius=0.001,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=ANCHOR_POS),
    )

    # Below-socket collision wall (collider-only -- not a visual). A per-env kinematic SOLID box filling
    # the fridge body below/behind the socket so the robot gripper cannot dip/tunnel into the concave
    # connector housing while inserting the plug. Spawned as a ``GeoType.BOX`` primitive (analytic solid
    # contact); it MUST stay thick in every dimension -- MuJoCo-Warp has no continuous collision, so a
    # thin slab is stepped through by the stiff gripper even though the contact pair is generated (see the
    # geometry-constant comment). It is added to the MJWarp robot entry below
    # (``SceneEntityCfg("fridge_floor")``) so the gripper -- and only the gripper -- collides it; being a
    # body shape (not world-static) it is invisible to the VBD entry, so it never touches the cable or
    # plug. Cloned per-env by the replicator. Its VISIBLE flag is cleared at model-init
    # (``_hide_fridge_collider_visuals``), so the Newton viewer draws it only under "Collisions". Tune the
    # size/pose (``FRIDGE_FLOOR_SIZE`` / ``FRIDGE_FLOOR_POS`` in ``geometry.py``) live in the viewer.
    # TEMPORARILY DISABLED: the bottom-of-cavity collision box. Uncomment to restore it (and re-add
    # ``SceneEntityCfg("fridge_floor")`` to the MJWarp entry's ``body_entities`` below).
    # fridge_floor = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/FridgeFloor",
    #     spawn=sim_utils.CuboidCfg(
    #         size=FRIDGE_FLOOR_SIZE,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 0.8)),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=FRIDGE_FLOOR_POS),
    # )

    # The deformable cable, simulated as a Cosserat rod by the VBD solver. The stretch stiffness is
    # firm enough to hold the hose taut without making it behave like a rigid rod during plug motion.
    cable1 = CableObjectCfg(
        prim_path="/World/envs/env_.*/Cable1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_CABLE1_USD,
            physics_material=NewtonCableMaterialCfg(
                # Hose feel: stiff in stretch (a hose barely elongates) but flexible in bending so it
                # holds a loose curve and follows the connector without fighting the grip. bend_damping
                # is kept a small fraction of bend_stiffness (was 2.0, ~0.4x stiffness -- a sluggish,
                # rod-like response that dumped force into the grasp through the weld). Tune
                # bend_stiffness up for a firmer hose / down for a floppier one.
                stretch_stiffness=1.0e6,
                stretch_damping=1.0e-2,
                bend_stiffness=3.0e-1,
                bend_damping=2.0e-2,
                density=100.0,
            ),
        ),
        init_state=CableObjectCfg.InitialStateCfg(
            pos=FRIDGE_POS,
        ),
        # Keep the authored node spacing: the head plug weld's 22 mm offset is authored against
        # the original segment-0 frame, so resampling would invalidate it. None = no resampling.
        resample_segment_length=None,
        attachments=[
            # Head weld: cable segment-0 head node -> graspable Plug1 rigid body.
            CableAttachmentCfg(
                target_prim_path="/World/envs/env_.*/Plug1",
                cable_anchor=0,
                cable_local_pos=(0.0, 0.0, CABLE_HEAD_TO_PLUG_ORIGIN_LOCAL_Z),
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
    gripper_finger_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["right_gripper_leftfinger", "right_gripper_rightfinger"],
            ),
            "static_friction_range": (_GRIPPER_PAD_STATIC_FRICTION, _GRIPPER_PAD_STATIC_FRICTION),
            "dynamic_friction_range": (_GRIPPER_PAD_DYNAMIC_FRICTION, _GRIPPER_PAD_DYNAMIC_FRICTION),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
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
            "plug_cfg": SceneEntityCfg("plug1"),
            "cable_cfg": SceneEntityCfg("cable1"),
            "socket_pos": SOCKET_MOUTH_POS,
            "socket_quat": SOCKET_ROT_QUAT_XYZW,
            "radial_threshold": 0.012,
            # Measure the seated connector at a point 4 mm ahead of the cable-head body.
            # The scripted insertion target is also +4 mm; using this tip point lets the
            # demo terminate before the gripper keeps pushing into the socket contact.
            "cable_tip_offset": 0.004,
            "min_depth": -0.001,
            "max_depth": 0.010,
            "alignment_threshold": 0.75,
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

        _register_rby1df_gripper_mimic_override()
        _register_rby1df_collision_restriction()

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
            # show_static=False so static shapes (the fridge, ground) obey the viewer's Visuals/Collisions
            # toggles via their VISIBLE/COLLIDE_SHAPES flags, instead of being force-drawn regardless. This
            # lets the fridge visual be toggled off with the "Visuals" checkbox, and routes the fridge
            # colliders (VISIBLE cleared by _hide_fridge_collider_visuals) to the "Collisions" checkbox.
            show_static=False,
        )
        self.sim.visualizer_cfgs = [KitVisualizerCfg(**kit_view), NewtonVisualizerCfg(**newton_view)]

        # Resolution of `--video` recordings (independent of the on-screen visualizer windows above).
        self.video_recorder.window_width = 1600
        self.video_recorder.window_height = 1600

        # Coupled physics: MuJoCo-Warp (MJWarp) solves the articulated rby1df robot, VBD solves the
        # deformable cable and its welded plug/anchor bodies. The two are joined by proxy coupling:
        # the gripper bodies are mirrored as proxy bodies in the VBD solver so the cable and plug
        # collide against them. The solver entries are spelled out explicitly so the task can route
        # only the authored socket and connector-housing colliders from the fridge USD into the VBD
        # entry rather than the whole static scene.
        self.sim.physics = CoupledNewtonCfg(
            scene_cfg=self.scene,
            use_cuda_graph=True,
            # The plug must insert into the authored socket bore: convex-hull mesh
            # approximation would fill the bore (and the gripper-finger cavities).
            simplify_meshes=False,
            solver_cfg=CoupledSolverCfg(
                coupling_type="proxy",
                scene_cfg=self.scene,
                entries=[
                    CoupledSolverEntryCfg(
                        name="mjc",
                        solver_cfg=MJWarpSolverCfg(
                            cone="elliptic",
                            ls_iterations=20,
                            integrator="implicitfast",
                            # use_mujoco_contacts=False routes the robot's contacts to Newton's
                            # collision pipeline, so the static housing collider is the single concave
                            # welded mesh (Newton collides it directly; MuJoCo-Warp's compiled convex
                            # geom for it is inert). MuJoCo never sees the cavity-filling convex hull.
                            use_mujoco_contacts=False,
                            # The robot's restricted gripper auto-includes the world-static fridge
                            # colliders (housing + socket) into this entry and nestles into the socket
                            # region during insertion, so the contact count there can exceed the default
                            # nconmax (48), which silently DROPS the surplus. Size the contact + constraint
                            # buffers with headroom; buffers only cap capacity, so the spare is cheap.
                            nconmax=4096,
                            njmax=1024,
                        ),
                        # NOTE: the bottom-of-cavity collision box (``fridge_floor``) is temporarily
                        # disabled (asset commented out below); re-add ``SceneEntityCfg("fridge_floor")``
                        # here to restore it.
                        body_entities=[SceneEntityCfg("robot")],
                        # The robot collides the connector HOUSING but not the socket: the gripper should
                        # not fight the socket bore during insertion (only the plug seats into it). Listing
                        # the housing explicitly here scopes this entry to it -- a non-empty shape list also
                        # suppresses Newton's world-static auto-include, so the socket (a separate
                        # world-static shape) is NOT pulled in. The housing is still SHARED with the VBD
                        # entry: explicit listing here makes this entry *own* the housing, but the VBD entry
                        # (empty shape list) auto-includes it by visibility -- Newton's _entry_visible_shapes
                        # shows every ``shape_body < 0`` shape to any entry with no owned shapes, and that is
                        # shared visibility, not ownership (the "owned by >1 entry" error only fires on
                        # explicit listings, and only this entry lists it). So robot↔housing runs in MJWarp
                        # and cable↔housing in VBD, from one mesh. ``include_body_shapes=True`` adds the
                        # robot's (and fridge_floor's) colliders to this entry's resolved shape list so
                        # the entry-local Newton collision pipeline (``use_mujoco_contacts=False``) actually
                        # broad-phases robot↔housing pairs. Set ``WATERHOSE_FRIDGE_COLLISION=0`` to drop the
                        # housing contact for both sides (``_disable_fridge_body_collision`` clears its collide flags).
                        include_body_shapes=True,
                        shape_label_patterns=[FRIDGE_HOUSING_COLLISION_MESH_PATTERN],
                    ),
                    CoupledSolverEntryCfg(
                        name="vbd",
                        # Contact/solver recipe matched to Newton's franka_cable_ik_pick_place reference
                        # grasp: HARD (augmented-Lagrangian) contacts enforce non-penetration of the plug
                        # against the gripper and socket, paired with a gentle penalty ramp (beta=1e2) and
                        # enough VBD iterations for the contact duals to converge. The cable welds stay SOFT
                        # (rigid_joint_hard=False): the head->Plug1 and tail->Anchor1 fixed joints have
                        # small authored offsets, and a hard joint solve would inject a large startup
                        # impulse into the cable.
                        # iterations x num_substeps (below) is the throughput knob. The high-fidelity
                        # default is 20 x 10; 16 x 8 is ~1.2x faster with a byte-identical scripted-demo
                        # arc, while 12 x 6 is ~1.44x faster (good for training; the plug seats a little
                        # slower in the demo). Select those lower-cost modes per run via
                        # WATERHOSE_VBD_ITERS / WATERHOSE_SUBSTEPS.
                        solver_cfg=VBDSolverCfg(
                            iterations=vbd_iterations,
                            friction_epsilon=0.1,
                            rigid_contact_hard=True,
                            rigid_joint_hard=False,
                            rigid_avbd_beta=1.0e2,
                            rigid_avbd_gamma=0.999,
                            # Warm-start contact state across the per-substep contact refresh. The proxy
                            # pipeline re-collides every substep; history restores the hard-contact lambda
                            # (plus penalty k and sticky anchors) so the constraint stays converged instead
                            # of restarting each substep. Requires contact_matching="latest" on the proxy
                            # collision pipeline (see _make_proxy_collision_pipeline).
                            rigid_contact_history=True,
                            rigid_contact_k_start=1.0e3,
                            # The cable/gripper/socket contacts can exceed 1000 contacts on a single
                            # body, so size the per-body contact buffer well above the default.
                            rigid_body_contact_buffer_size=4096,
                            # Weld (head->Plug1, tail->Anchor1) penalty stiffness. Matched to the cable
                            # stretch stiffness (1e6) so the head->plug weld is not the weak link in the
                            # head(gripped)<->tail(anchored) chain: at the previous 1e5 the weld -- 10x
                            # softer than the cable -- stretched under the carry/insert drag, so the
                            # plug pulled away from the cable head (the "plug dragged, cable lagging"
                            # separation) and the stiffness discontinuity at the weld oscillated. The
                            # ramp (k_start) stays low so the authored 22 mm weld offset does not inject
                            # a startup impulse; the joints remain soft-mode (rigid_joint_hard=False).
                            rigid_joint_linear_ke=1.0e9,
                            rigid_joint_angular_ke=1.0e9,
                            rigid_joint_linear_k_start=1.0e4,
                            rigid_joint_angular_k_start=1.0e1,
                            rigid_joint_linear_kd=0.0,
                            rigid_joint_angular_kd=0.0,
                        ),
                        solver_class="newton.solvers:SolverVBD",
                        body_entities=[
                            SceneEntityCfg("cable1"),
                            # Weld targets must live in the VBD model so the cable fixed joints
                            # (head -> Plug1, tail -> Anchor1) can be created against them.
                            SceneEntityCfg("plug1"),
                            SceneEntityCfg("anchor1"),
                        ],
                        all_particles=True,
                        # ``include_body_shapes=False`` + ``include_static_shapes=False`` + no explicit
                        # shapes/patterns leaves this entry's resolved shape list EMPTY, so Newton
                        # auto-includes EVERY world-static fridge collider here: the welded housing mesh
                        # (shared with the MJWarp entry, which owns it explicitly) AND the socket bore. So
                        # the cable particles graze the fridge body (cable↔housing) and the plug seats into
                        # the socket (plug↔socket); the housing is the same single mesh the robot also
                        # collides. The socket is collided only by this entry (the robot's non-empty shape
                        # list excludes it, so the gripper does not fight the bore). The cable/plug/anchor
                        # bodies stay visible via body ownership.
                        include_body_shapes=False,
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
                                        "right_gripper_leftfinger",
                                        "right_gripper_rightfinger",
                                    ],
                                )
                            ],
                            # Drive the proxy bodies from the source solver's end-of-step pose and
                            # velocity together, so the gripper proxies stay consistent as the
                            # fingers close on the plug.
                            mode="staggered",
                            mass_scale=1.0,
                            collision_pipeline_factory=_make_proxy_collision_pipeline,
                            collide_interval=1,
                            shape_material_ke=_VBD_CONTACT_STIFFNESS,
                            shape_material_kd=_VBD_CONTACT_DAMPING,
                            shape_material_mu=_VBD_GRIPPER_PROXY_FRICTION,
                            shape_margin=_VBD_GRIPPER_PROXY_MARGIN,
                        )
                    ],
                    iterations=1,
                ),
            ),
            num_substeps=substeps,
            collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=65536),
            model_cfg=NewtonModelCfg(
                shape_material_ke=_VBD_CONTACT_STIFFNESS,
                shape_material_kd=_VBD_CONTACT_DAMPING,
                soft_contact_mu=_VBD_SOFT_CONTACT_FRICTION,
                shape_material_mu=_VBD_DEFAULT_SHAPE_FRICTION,
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
        # right-ee-only action matches what the teleop pipelines emit.


@configclass
class WaterhoseAdmmIkEnvCfg(WaterhoseProxyIkEnvCfg):
    """Waterhose task variant using Newton ADMM contact coupling instead of proxy bodies."""

    def __post_init__(self) -> None:
        super().__post_init__()
        solver_cfg = self.sim.physics.solver_cfg
        solver_cfg.coupling_type = "admm"
        solver_cfg.use_collision_pipeline = False
        # ADMM contact coupling uses a stiff penalty (rho) with a small proximal term (gamma) and
        # Baumgarte stabilization, plus frame-to-frame contact matching that warm-starts the ADMM
        # dual so the grasp contact stays consistent as the gripper closes and carries the plug.
        # The finite-mass proxy task (Isaac-Waterhose-Coupled-v0) is the primary demo path; this
        # ADMM variant is provided for solver comparison.
        solver_cfg.admm_coupling = AdmmCouplingCfg(
            iterations=5,
            rho=200.0,
            gamma=0.001,
            baumgarte=0.5,
            rigid_contact_matching="latest",
            contact_matching_force_scale=0.9,
            contact_pairs=[AdmmContactPairCfg(source="mjc", destination="vbd")],
        )


@configclass
class WaterhoseProxyTeleopEnvCfg(WaterhoseProxyIkEnvCfg):
    """Waterhose task variant for native IsaacLab keyboard and SpaceMouse teleoperation."""

    actions: WaterhoseNewtonRelativeIkActionsCfg = WaterhoseNewtonRelativeIkActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.xr = XrCfg(
            anchor_pos=(0.0, 0.9, -1),
            # XrCfg quaternions are xyzw. Rotate the simulation 180 deg around world up so the
            # headset initially faces the fridge.
            anchor_rot=(0.0, 0.0, 1.0, 0.0),
        )
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=build_waterhose_relative_teleop_pipeline,
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
