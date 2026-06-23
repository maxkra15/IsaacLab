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
from isaaclab_newton.ik.newton_ik_objectives_cfg import NewtonIKJointLimitObjectiveCfg, NewtonIKPoseObjectiveCfg
from isaaclab_newton.ik.newton_ik_solver_cfg import NewtonIKSolverCfg
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
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
    FRIDGE_BODY_COLLISION_MESH_PATTERN,
    FRIDGE_BODY_WELDED_MESH_PATTERN,
    FRIDGE_POS,
    RIGHT_GRIPPER_EE_FRAME_POS,
    RIGHT_GRIPPER_EE_FRAME_QUAT_XYZW,
    SOCKET_COLLISION_MESH_PATTERN,
    SOCKET_COLLISION_MESH_SUFFIX,
    SOCKET_MOUTH_POS,
    SOCKET_ROT_QUAT_XYZW,
)
from .mdp.actions import WaterhoseGripperPositionActionCfg
from .mdp.terminations import plug_inserted_in_socket
from .teleop import WaterhoseSpaceMouseCfg

# Best-practices IsaacTeleop pipelines. The previous known-working variants are preserved in
# ``teleop_pipelines_legacy`` (same function names); switch this import to that module to
# restore the exact prior XR behavior if a refactor here regresses the live session.
from .teleop_pipelines import build_waterhose_relative_teleop_pipeline, build_waterhose_teleop_pipeline

WATERHOSE_ASSETS_DIR = os.environ.get(
    "WATERHOSE_ASSETS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets"),
)

_FRIDGE_USD = os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "fridge_waterhose.usda")
_RBY1_USD = os.path.join(WATERHOSE_ASSETS_DIR, "rby1df", "rby1df_waterhose.usda")
_PLUG_USD = os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "cable", "plug.usda")
_CABLE1_USD = os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "cable", "cable001.usda")

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
_VBD_GRIPPER_PROXY_FRICTION = 1.0e1
_VBD_GRIPPER_PROXY_MARGIN = 0.001
_RIGHT_GRIPPER_JOINT_NAMES = [
    "right_gripper_finger_joint_1",
    "right_gripper_left_finger_joint",
    "right_gripper_right_finger_joint",
]


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


# Connector-housing collision. The robot collides with the per-fragment convex housing hulls through
# the MJWarp entry (cheap in MuJoCo-Warp), and when this flag is enabled the deformable hose collides
# with the housing through a single welded body collider routed into the VBD/cable entry (one shape
# keeps the per-substep particle-vs-shape soft-contact cost low). Set WATERHOSE_FRIDGE_BODY_COLLISION=0
# to route no housing collision into the hose entry (socket-only; the hose is authored to sit flush
# against the body).
_FRIDGE_BODY_COLLISION = _env_flag("WATERHOSE_FRIDGE_BODY_COLLISION", True)


# Texture-SDF collision for the fridge's embedded socket collider. An SDF gives the bore a smooth,
# correctly-signed gradient field that guides the plug tip during insertion better than reduced
# BVH mesh-mesh contacts. Resolution 128 resolves the thin (~3 mm) bore wall. Set
# WATERHOSE_SOCKET_SDF=0 to fall back to the plain BVH mesh contact path.
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
    if not _env_flag("WATERHOSE_SOCKET_SDF", True):
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

# Body-label suffix of the kinematic cable-tail anchor (a collision-free weld target).
_ANCHOR_BODY_TOKEN = "Anchor1"


def _restrict_rby1df_collision_to_right_gripper(_payload=None) -> None:
    """Keep collision only on the right-gripper fingers; clear it on the rest of the robot.

    The task only needs the robot to collide where it grasps and inserts -- the right gripper. The
    imported RBY1 USD authors a collider on every link, and the Newton importer keeps those colliders
    active, so the whole arm/torso/head would otherwise collide against the fridge housing every step.
    Clear both the ``COLLIDE_SHAPES`` (rigid) and ``COLLIDE_PARTICLES`` (deformable) flags on every
    robot shape whose body is not a right-gripper finger, so no part of the robot except the gripper
    generates rigid or hose-particle contacts. The cable, plug, and anchor colliders are left
    untouched (they are not robot links).
    """

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


def _register_rby1df_collision_restriction() -> None:
    """Limit active colliders to what the task needs before Newton finalizes the model.

    Restrict the robot's colliders to the right gripper, and turn the cable-tail anchor into a
    collision-free weld target.
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


def _make_proxy_collision_pipeline(model):
    """Build the destination-view collision pipeline used by Newton proxy coupling.

    Before building it, drop ``COLLIDE_PARTICLES`` on the gripper finger proxy shapes so the
    deformable hose (VBD particles) does NOT collide with -- and penetrate -- the fingers, while the
    plug grip is untouched. The two interactions ride independent contact paths gated by independent
    shape-flag bits: the Plug1 grip is a RIGID-vs-rigid contact (``COLLIDE_SHAPES``, carrying the
    weld-grade mu=1e6 tangential hold), and the hose-vs-finger penetration is a SOFT particle contact
    (``COLLIDE_PARTICLES``, generated over every particle x shape with no shape-pair filter, so a
    shape-pair filter cannot target it). Clearing only the particle bit removes the hose contacts and
    leaves the rigid grip byte-identical. ``model`` here is the VBD destination view; this runs once at
    pipeline construction (after per-entry shape visibility is set), so it is CUDA-graph-safe.
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

    return CollisionPipeline(
        model,
        broad_phase="explicit",
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
            "static_friction_range": (_GRIPPER_PAD_STATIC_FRICTION, _GRIPPER_PAD_STATIC_FRICTION),
            "dynamic_friction_range": (_GRIPPER_PAD_DYNAMIC_FRICTION, _GRIPPER_PAD_DYNAMIC_FRICTION),
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
            show_static=True,
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
                            use_mujoco_contacts=False,
                            # The fridge-housing colliders raise the robot's peak contact count to
                            # ~126 (gripper/plug nestling into the socket region, which is densely
                            # surrounded by convex housing pieces). The default nconmax (48) silently
                            # DROPS the surplus, so size the contact + constraint buffers above the
                            # peak with headroom. Buffers only cap capacity -- solver cost scales with
                            # the actual contact count, so the spare capacity is cheap (1 env).
                            nconmax=4096,
                            njmax=1024,
                        ),
                        body_entities=[SceneEntityCfg("robot")],
                        # The robot collides with the per-fragment housing convex hulls
                        # (Cable008_Collider*), selected explicitly by label. ``include_body_shapes=True``
                        # keeps the robot's own shapes in this entry's resolved shape list, which
                        # suppresses the coupled solver's blanket world-static auto-include; the housing
                        # hulls are then the only static shapes this entry owns. That keeps the single
                        # welded body mesh (which the hose uses) out of the robot's view -- as a concave
                        # mesh it would otherwise be convexified by MuJoCo-Warp and fill the socket
                        # cavity. The socket collider is owned by the VBD entry (the plug inserts against
                        # it); a world-static shape can only be owned by one entry, so it is not listed
                        # here. With WATERHOSE_FRIDGE_BODY_COLLISION=0 the robot collides with its own
                        # shapes only.
                        include_body_shapes=True,
                        shape_label_patterns=([FRIDGE_BODY_COLLISION_MESH_PATTERN] if _FRIDGE_BODY_COLLISION else []),
                    ),
                    CoupledSolverEntryCfg(
                        name="vbd",
                        # Contact/solver recipe matched to Newton's franka_cable_ik_pick_place reference
                        # grasp: HARD (augmented-Lagrangian) contacts enforce non-penetration of the plug
                        # against the gripper and socket, paired with a gentle penalty ramp (beta=1e2) and
                        # 20 VBD iterations so the contact duals converge. The cable welds stay SOFT
                        # (rigid_joint_hard=False): the head->Plug1 and tail->Anchor1 fixed joints have
                        # small authored offsets, and a hard joint solve would inject a large startup
                        # impulse into the cable.
                        solver_cfg=VBDSolverCfg(
                            iterations=20,
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
                            rigid_joint_linear_ke=1.0e6,
                            rigid_joint_angular_ke=1.0e6,
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
                        include_static_shapes=False,
                        # Route the socket collider (texture-SDF bore, so the plug inserts against a
                        # smooth field) and, when WATERHOSE_FRIDGE_BODY_COLLISION is enabled, the single
                        # welded housing-body mesh into the VBD entry, so the cable and plug collide with
                        # the fridge body. The deformable-hose soft-contact pass runs over every
                        # particle x shape, so routing one welded mesh here instead of the full hull set
                        # keeps that cost low. The robot uses the per-fragment hulls via the MJWarp entry.
                        shape_label_patterns=(
                            [SOCKET_COLLISION_MESH_PATTERN, FRIDGE_BODY_WELDED_MESH_PATTERN]
                            if _FRIDGE_BODY_COLLISION
                            else [SOCKET_COLLISION_MESH_PATTERN]
                        ),
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
            num_substeps=10,
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

    Base (DiffIK) action space, retained for the non-coupled :class:`WaterhoseIkEnvCfg` variant and
    as the shared gripper-action parent for the Newton-IK subclasses below.
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
    """Newton-native relative end-effector delta pose plus normalized right-gripper action.

    Teleop variant (matches the original multi-body Newton-IK teleop): a single command-driven
    relative pose objective (translation deltas applied in the root frame; orientation deltas applied
    in the end-effector frame via :class:`WaterhoseLocalFrameNewtonInverseKinematicsAction`) plus the
    soft joint-limit objective, with no hold objectives -- the teleop operator owns the posture. The
    torso, left arm, and right arm all share the IK joint set so the end-effector cleanly tracks the
    commanded direction (a right-arm-only set pushes the EE off-axis near the arm's reach limit); the
    soft joint-limit objective keeps the redundant joints from drifting.
    """

    arm_action = NewtonInverseKinematicsActionCfg(
        class_type="isaaclab_tasks.contrib.waterhose.mdp.actions:WaterhoseLocalFrameNewtonInverseKinematicsAction",
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
                use_relative_mode=True,
            ),
            NewtonIKJointLimitObjectiveCfg(weight=0.1),
        ],
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
        self.xr = XrCfg(
            anchor_pos=(0.0, 0.9, -1),
            # XrCfg quaternions are xyzw. Rotate the simulation 180 deg
            # around world up so the headset initially faces the fridge.
            anchor_rot=(0.0, 0.0, 1.0, 0.0),
        )
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=build_waterhose_teleop_pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,
            target_frame_prim_path=_ROBOT_BASE_PRIM_PATH_ENV0,
            teleoperation_active_default=True,
            control_channel_uuid=None,
        )


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
