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
from isaaclab_newton.ik.newton_ik_objectives_cfg import NewtonIKJointLimitObjectiveCfg, NewtonIKPoseObjectiveCfg
from isaaclab_newton.ik.newton_ik_solver_cfg import NewtonIKSolverCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCollisionPipelineCfg
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

from isaaclab_contrib.cable.cable_object_cfg import CableAttachmentCfg
from isaaclab_contrib.coupling import (
    CoupledAdmmContactPairCfg,
    CoupledAdmmSolverCfg,
    CoupledProxyCfg,
    CoupledProxySolverCfg,
    CoupledSolverEntryCfg,
)
from isaaclab_contrib.deformable.newton_manager_cfg import (
    CoupledNewtonCfg,
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
from .teleop import WaterhoseSpaceMouseCfg

# Best-practices IsaacTeleop pipelines. The previous known-working variants are preserved in
# ``teleop_pipelines_legacy`` (same function names); switch this import to that module to
# restore the exact prior XR behavior if a refactor here regresses the live session.
from .teleop_pipelines import build_waterhose_relative_teleop_pipeline

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

# Wide scene-fit camera shared by the environment controller and every visualizer.
_SCENE_CAMERA_EYE = (-2.55, -7.1, 2.3)
_SCENE_CAMERA_LOOKAT = (0.55, -0.42, 0.9)
_ROBOT_BASE_PRIM_PATH_ENV0 = "/World/envs/env_0/Robot/Geometry/origin"
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


_RIGHT_GRIPPER_FINGER_BODY_TOKENS = ("right_gripper_leftfinger", "right_gripper_rightfinger")
# Shape-label suffix for the connector mesh lumped directly into the cable head.
_CONNECTOR_SHAPE_TOKEN = "waterhose_connector"


def _make_proxy_collision_pipeline(model):
    """Build the destination-view collision pipeline used by Newton proxy coupling.

    The pipeline is scoped so the gripper finger proxies interact with the deformable connector ONLY,
    along two independent contact paths:

    * Soft (particle) path: ``COLLIDE_PARTICLES`` is cleared on the finger proxy shapes so the deformable
      hose (VBD particles) does NOT collide with -- and penetrate -- the fingers. This bit is generated
      over every particle x shape with no shape-pair filter, so only the flag can target it.
    * Rigid path: an explicit ``shape_pairs_filtered`` list keeps cable<->housing,
      connector<->housing, connector<->socket, and cable self-contact, while keeping only
      finger<->connector for finger pairs. The localized clearance in the housing mesh is therefore
      the only place around the socket where connector-to-housing contact is absent.

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
            flags[shape_id] &= ~particle_bit  # keep COLLIDE_SHAPES (the connector grip); drop only the hose path
            dropped += 1
    if dropped == 0:
        raise RuntimeError(
            "waterhose proxy pipeline: no gripper finger shapes had COLLIDE_PARTICLES set "
            "(proxy view shape layout changed?); the hose-vs-finger penetration filter did not apply."
        )
    model.shape_flags = wp.array(flags, dtype=wp.int32, device=model.device)
    print(f"[waterhose] Disabled hose-vs-gripper particle collision on {dropped} finger proxy shape(s).", flush=True)

    # Scope the finger proxies to the GRIP ONLY. This destination-view pipeline otherwise collides the
    # finger proxies against the housing and socket explicitly routed into the VBD entry, so the gripper
    # fights the fridge inside the VBD solve. That
    # duplicates the robot<->housing contact the MJWarp entry already owns and pushes the gripper off the
    # bore during insertion. Robot<->housing belongs to MJWarp; the proxy is only the grip on the
    # connector. Build an explicit pair list that keeps every non-finger path and drops every finger
    # pair except finger<->connector.
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
    grip_shapes = {
        shape_id for shape_id, label in enumerate(shape_label) if _CONNECTOR_SHAPE_TOKEN.lower() in label.lower()
    }
    if not grip_shapes:
        raise RuntimeError(
            f"waterhose proxy pipeline: no connector collider shape found by label ({_CONNECTOR_SHAPE_TOKEN!r}); "
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
        rigid_contact_max=_PROXY_RIGID_CONTACT_MAX,
        # "sticky" replays previous-frame contact geometry; that is useful for
        # hard-contact demos, but makes the hose feel glued to the fingers here.
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
        init_state=WaterhoseCableObjectCfg.InitialStateCfg(
            pos=FRIDGE_POS,
        ),
        # Keep the authored node spacing because the connector transform is expressed in segment 0's frame.
        resample_segment_length=None,
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
        attachments=[
            # Tail weld: final cable-segment start node -> kinematic Anchor1 sphere. A negative index
            # keeps this attached to the tail if the canonical curve is resampled in the future.
            CableAttachmentCfg(
                target_prim_path="/World/envs/env_.*/Anchor1",
                cable_anchor=-1,
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
        self.sim.physics = CoupledNewtonCfg(
            scene_cfg=self.scene,
            use_cuda_graph=True,
            # The plug must insert into the authored socket bore: convex-hull mesh
            # approximation would fill the bore (and the gripper-finger cavities).
            simplify_meshes=False,
            solver_cfg=CoupledProxySolverCfg(
                entries=[
                    CoupledSolverEntryCfg(
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
                        # Preserve the selected finger links' authored mass/inertia for their
                        # contact proxies. MuJoCo's articulated effective inertia is intentionally
                        # much larger and makes this small compliant grasp overly aggressive.
                        use_solver_effective_mass=False,
                        bodies=[SceneEntityCfg("robot")],
                        # Route only the robot's clearanced housing. Explicit selection excludes the
                        # canonical SDF socket from the robot view.
                        include_body_shapes=True,
                        shape_label_patterns=[_FRIDGE_ROBOT_COLLISION_PATTERN],
                    ),
                    CoupledSolverEntryCfg(
                        name="vbd",
                        # Contact/solver recipe matched to Newton's franka_cable_ik_pick_place reference
                        # grasp: HARD (augmented-Lagrangian) contacts enforce non-penetration of the plug
                        # against the gripper and socket, paired with a gentle penalty ramp (beta=1e2) and
                        # enough VBD iterations for the contact duals to converge. The cable-tail anchor
                        # stays a soft fixed joint; the connector itself is part of segment 0 and needs no weld.
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
                            # Tail-anchor weld penalty. The connector has no joint: its mesh and inertia
                            # are lumped directly into the cable-head body.
                            rigid_joint_linear_ke=1.0e9,
                            rigid_joint_angular_ke=1.0e9,
                            rigid_joint_linear_k_start=1.0e4,
                            rigid_joint_angular_k_start=1.0e1,
                            rigid_joint_linear_kd=0.0,
                            rigid_joint_angular_kd=0.0,
                        ),
                        bodies=[SceneEntityCfg("cable1")],
                        all_particles=True,
                        # VBD owns the canonical socket SDF plus the locally clearanced housing copy.
                        include_body_shapes=True,
                        include_static_shapes=False,
                        shape_label_patterns=[SOCKET_COLLISION_MESH_PATTERN, _FRIDGE_CABLE_COLLISION_PATTERN],
                    ),
                ],
                proxies=[
                    CoupledProxyCfg(
                        source="mjc",
                        destination="vbd",
                        bodies=[
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
            num_substeps=substeps,
            # This explicit capacity is visible before SolverVBD construction. Keep it
            # at least as large as the proxy pipeline so rigid-contact history can be
            # allocated before CUDA graph capture on current Newton.
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
        proxy_solver_cfg = self.sim.physics.solver_cfg
        if not isinstance(proxy_solver_cfg, CoupledProxySolverCfg):
            raise TypeError("WaterhoseAdmmIkEnvCfg expects the base environment to configure a CoupledProxySolverCfg.")
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
        self.sim.physics.solver_cfg = CoupledAdmmSolverCfg(
            entries=entries,
            use_collision_pipeline=False,
            iterations=5,
            rho=200.0,
            gamma=0.001,
            baumgarte=0.5,
            rigid_contact_matching="latest",
            contact_matching_force_scale=0.9,
            contact_pairs=[CoupledAdmmContactPairCfg(source="mjc", destination="vbd")],
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
