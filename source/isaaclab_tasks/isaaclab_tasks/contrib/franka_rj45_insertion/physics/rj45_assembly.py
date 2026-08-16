# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-scoped Newton assembly for the RJ45 plug, latch, socket, and cable.

The geometry and physical parameters match Newton's ``contacts_rj45_plug``
example at commit ``7bb6d02d8eeab2cffc3adfa453ddd63799a2ac6a``. The assembly is
added per world through :func:`isaaclab_newton.cloner.newton_builder_world_hook`.
It retains the example's task-local support plane; the host Isaac Lab scene
separately owns the robot table and global ground.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import newton
import newton.usd
import newton.utils
import numpy as np
import warp as wp
from newton.solvers import SolverVBD

from pxr import Usd, UsdGeom

from ._kernels import (
    align_cable_orientations,
    apply_connector_forces,
    reset_task_bodies,
    restore_goal_targets_masked,
    set_drive_enabled_masked,
    sync_cable_anchors,
    write_drive_targets_masked,
    write_task_body_state,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


NEWTON_RJ45_SOURCE_COMMIT = "7bb6d02d8eeab2cffc3adfa453ddd63799a2ac6a"
"""Newton commit from which the task geometry and physics were ported."""

RJ45_ASSET_SHA256 = "50c95bcfb63544777f9148d548aac6f16b62f65cacbaaa9316453d579de4b4fa"
"""SHA-256 of the unmodified Newton ``rj45_plug.usd`` asset."""

RJ45_ASSET_PATH = Path(__file__).resolve().parent / "assets" / "rj45_plug.usd"
"""Canonical task-local Newton RJ45 USD asset."""

CONTACT_KE = 1.0e5
"""Connector and cable normal contact stiffness [N/m]."""

CONTACT_KD = 0.0
"""Connector and cable normal contact damping [N·s/m]."""

CONNECTOR_GAP = 0.002
"""Connector contact detection gap [m]."""

CONNECTOR_DENSITY = 1.0e6
"""Plug and latch mesh density [kg/m³]."""

RIGID_GAP = 0.005
"""Newton rigid broad-phase gap used by the reference example [m]."""

SUPPORT_PLANE_LOCAL_HEIGHT = 0.0
"""Authored support-plane height in the RJ45 assembly frame [m]."""

SUPPORT_PLANE_CONTACT_KE = 2.5e3
"""Reference Newton default support-plane contact stiffness [N/m]."""

SUPPORT_PLANE_CONTACT_KD = 100.0
"""Reference Newton default support-plane contact damping [N·s/m]."""

SUPPORT_PLANE_CONTACT_KF = 1000.0
"""Reference Newton default support-plane tangential stiffness [N/m]."""

SUPPORT_PLANE_MU = 1.0
"""Reference Newton default support-plane Coulomb friction coefficient."""

MESH_SDF_MAX_RESOLUTION = 128
"""Maximum axis resolution of each connector mesh SDF."""

MESH_SDF_NARROW_BAND_RANGE = (-2.0 * CONNECTOR_GAP, 2.0 * CONNECTOR_GAP)
"""Signed-distance range retained around each connector surface [m]."""

PLUG_Y_OFFSET = -0.025
"""Unplugged initial displacement along the local insertion axis [m]."""

INSERTION_DISTANCE = -PLUG_Y_OFFSET
"""Distance from the offset start to the USD-authored plug pose [m].

The mechanically seated reset oracle deliberately travels farther than this
nominal displacement so the latch can clear and return to its rest angle.
"""

CABLE_RADIUS = 0.00325
"""Cable capsule radius [m]."""

CABLE_SEGMENT_COUNT = 35
"""Number of rigid capsule segments in the Newton cable."""

CABLE_KINEMATIC_COUNT = 4
"""Number of leading cable segments rigidly synchronized to the plug."""

CABLE_MU = 2.0
"""Cable Coulomb friction coefficient."""

CABLE_BEND_STIFFNESS = 1.0e1
"""Cable-joint bend and twist stiffness [N·m/rad]."""

CABLE_BEND_DAMPING = 1.0
"""Cable-joint bend and twist damping [N·m·s/rad]."""

GRASP_PROXY_CENTER = (0.0, -0.025, -0.0006)
"""Rear-housing grasp-proxy center in the plug body frame [m]."""

GRASP_PROXY_HALF_EXTENTS = (0.00725, 0.010, 0.00475)
"""Rear-housing grasp-proxy half extents [m]."""

GRASP_FRICTION = 2.0
"""Friction coefficient used by the grasp proxy and Franka fingers."""

LATCH_LIMIT_LOWER = -0.2
"""Maximum inward latch deflection [rad]."""

LATCH_LIMIT_UPPER = 0.3
"""Maximum outward latch deflection [rad]."""

LATCH_SPRING_KE = 0.15
"""Latch angular return-spring stiffness [N·m/rad]."""

LATCH_SPRING_KD = 0.03
"""Latch angular return-spring damping [N·m·s/rad]."""

LATCH_LIMIT_KD = 1.0e-4
"""Latch angular-limit damping [N·m·s/rad]."""

RJ45_REFERENCE_FRAME_DT = 1.0 / 60.0
"""Reference-example rendered frame duration [s]."""

RJ45_REFERENCE_SUBSTEPS = 6
"""Reference-example solver substeps per rendered frame."""

RJ45_REFERENCE_SIM_DT = RJ45_REFERENCE_FRAME_DT / RJ45_REFERENCE_SUBSTEPS
"""Reference-example VBD substep duration [s]."""

RJ45_VBD_ITERATIONS = 12
"""Minimum VBD iterations used by the reference assembly."""

RJ45_VBD_CONTACT_BUFFER_SIZE = 256
"""Minimum per-body rigid-contact buffer used by the reference assembly."""

RJ45_VBD_LEGACY_CONTACT_HARD = False
"""Newton 1.5 soft-contact fallback matching the pre-compliant-ALM example."""

TASK_BODY_COUNT = 2 + CABLE_SEGMENT_COUNT
"""Number of resettable bodies per world: plug, latch, then cable segments."""

_CONNECTOR_SHAPE_CFG = newton.ModelBuilder.ShapeConfig(
    mu=0.0,
    ke=CONTACT_KE,
    kd=CONTACT_KD,
    gap=CONNECTOR_GAP,
    density=CONNECTOR_DENSITY,
    mu_torsional=0.0,
    mu_rolling=0.0,
)

_SUPPORT_PLANE_SHAPE_CFG = newton.ModelBuilder.ShapeConfig(
    ke=SUPPORT_PLANE_CONTACT_KE,
    kd=SUPPORT_PLANE_CONTACT_KD,
    kf=SUPPORT_PLANE_CONTACT_KF,
    mu=SUPPORT_PLANE_MU,
    mu_torsional=0.005,
    mu_rolling=0.0001,
    margin=0.0,
    gap=None,
)

_GRASP_PROXY_SHAPE_CFG = newton.ModelBuilder.ShapeConfig(
    density=0.0,
    ke=CONTACT_KE,
    kd=CONTACT_KD,
    mu=GRASP_FRICTION,
    mu_torsional=0.0,
    mu_rolling=0.0,
    gap=CONNECTOR_GAP,
    has_particle_collision=False,
    is_visible=False,
)

_VBD_ATTRIBUTE_NAMES = ("vbd:joint_is_hard", "vbd:dahl_eps_max", "vbd:dahl_tau")
_FRANKA_HAND_BODY_SUFFIX = "/panda_hand"
_FRANKA_FINGER_BODY_SUFFIXES = ("/panda_leftfinger", "/panda_rightfinger")
_FRANKA_GRASP_BODY_SUFFIXES = (_FRANKA_HAND_BODY_SUFFIX, *_FRANKA_FINGER_BODY_SUFFIXES)


@dataclass(frozen=True)
class Rj45InsertionDriveCfg:
    """Parameters for the optional goal-generation insertion drive.

    The runtime always cancels gravity on the plug and latch, matching the
    Newton example. The positional drive is disabled at construction and must
    be explicitly enabled for solved-state generation.

    Attributes:
        stiffness: Goal-drive translational stiffness [N/m].
        damping: Goal-drive translational damping [N·s/m].
    """

    stiffness: float = 50.0
    damping: float = 10.0

    def __post_init__(self) -> None:
        """Validate finite non-negative drive gains."""
        if not math.isfinite(self.stiffness) or self.stiffness < 0.0:
            raise ValueError(f"RJ45 drive stiffness must be finite and non-negative, got {self.stiffness}.")
        if not math.isfinite(self.damping) or self.damping < 0.0:
            raise ValueError(f"RJ45 drive damping must be finite and non-negative, got {self.damping}.")


@dataclass(frozen=True)
class Rj45WorldBodyIds:
    """Stable builder indices for one task-local RJ45 world."""

    world_id: int
    support_plane_shape_id: int
    socket_shape_id: int
    plug_body_id: int
    plug_shape_id: int
    grasp_proxy_shape_id: int
    latch_body_id: int
    latch_shape_id: int
    d6_joint_id: int
    latch_joint_id: int
    cable_body_ids: tuple[int, ...]
    cable_joint_ids: tuple[int, ...]

    @property
    def task_body_ids(self) -> tuple[int, ...]:
        """Body indices in persisted reset-state order."""
        return (self.plug_body_id, self.latch_body_id, *self.cable_body_ids)

    @property
    def cable_anchor_body_ids(self) -> tuple[int, ...]:
        """Plug-relative kinematic cable body indices."""
        return self.cable_body_ids[:CABLE_KINEMATIC_COUNT]

    @property
    def pinned_cable_body_id(self) -> int:
        """Fixed far-end cable body index."""
        return self.cable_body_ids[-1]


@dataclass(frozen=True)
class FrankaGraspShapeIds:
    """Imported Franka hand/finger colliders used by task-local filtering.

    A body may own multiple colliders, so each field is a tuple even though
    the standard Franka asset currently imports one collider per body.
    """

    hand_shape_ids: tuple[int, ...]
    left_finger_shape_ids: tuple[int, ...]
    right_finger_shape_ids: tuple[int, ...]

    @property
    def finger_shape_ids(self) -> tuple[int, ...]:
        """All collidable finger shapes, left then right."""
        return (*self.left_finger_shape_ids, *self.right_finger_shape_ids)

    @property
    def all_shape_ids(self) -> tuple[int, ...]:
        """All coupled grasp shapes, hand then left/right fingers."""
        return (*self.hand_shape_ids, *self.finger_shape_ids)


@dataclass(frozen=True)
class _Rj45Geometry:
    socket_mesh: newton.Mesh
    socket_position: wp.vec3
    plug_mesh: newton.Mesh
    plug_position: wp.vec3
    latch_mesh: newton.Mesh
    latch_position: wp.vec3
    cable_points: tuple[wp.vec3, ...]
    cable_quaternions: tuple[wp.quat, ...]
    anchor_offsets: tuple[wp.vec3, ...]
    anchor_rotations: tuple[wp.quat, ...]
    align_next_start_offsets: tuple[wp.vec3, ...]


@dataclass(frozen=True)
class _WorldBuildRecord:
    ids: Rj45WorldBodyIds
    root_label: str
    default_body_q: tuple[tuple[float, ...], ...]
    goal_target_w: tuple[float, float, float]


def verify_rj45_asset(path: str | Path = RJ45_ASSET_PATH) -> None:
    """Verify that an asset is the exact Newton RJ45 source file.

    Args:
        path: Asset path to validate.

    Raises:
        FileNotFoundError: If the asset does not exist.
        RuntimeError: If its SHA-256 differs from the pinned Newton asset.
    """
    asset_path = Path(path)
    if not asset_path.is_file():
        raise FileNotFoundError(f"Newton RJ45 asset not found: {asset_path}")
    digest = hashlib.sha256(asset_path.read_bytes()).hexdigest()
    if digest != RJ45_ASSET_SHA256:
        raise RuntimeError(
            f"Newton RJ45 asset digest mismatch for {asset_path}: expected {RJ45_ASSET_SHA256}, got {digest}."
        )


def validate_rj45_vbd_solver_cfg(solver_cfg: Any) -> None:
    """Validate the VBD settings required by the compliant-ALM RJ45 assembly.

    Isaac Lab develop pins Newton ``release-1.5``, where the official RJ45
    example used soft legacy AVBD contacts. Requiring both flags lets the
    manager forward ``rigid_contact_hard=False`` on 1.5 and select unified
    compliant ALM automatically when the 1.6 argument is available.

    Args:
        solver_cfg: Object exposing the fields of
            :class:`isaaclab_newton.physics.VBDSolverCfg`.

    Raises:
        ValueError: If the solver would use legacy constraints, external rigid
            integration, too few iterations, or too small a contact buffer.
    """
    errors: list[str] = []
    if getattr(solver_cfg, "rigid_compliant_alm", None) is not True:
        errors.append("rigid_compliant_alm must be True")
    if getattr(solver_cfg, "rigid_contact_hard", None) is not RJ45_VBD_LEGACY_CONTACT_HARD:
        errors.append("rigid_contact_hard must be False for the Newton 1.5 fallback")
    if bool(getattr(solver_cfg, "integrate_with_external_rigid_solver", False)):
        errors.append("integrate_with_external_rigid_solver must be False")
    iterations = int(getattr(solver_cfg, "iterations", 0))
    if iterations < RJ45_VBD_ITERATIONS:
        errors.append(f"iterations must be >= {RJ45_VBD_ITERATIONS}, got {iterations}")
    contact_buffer_size = int(getattr(solver_cfg, "rigid_body_contact_buffer_size", 0))
    if contact_buffer_size < RJ45_VBD_CONTACT_BUFFER_SIZE:
        errors.append(
            f"rigid_body_contact_buffer_size must be >= {RJ45_VBD_CONTACT_BUFFER_SIZE}, got {contact_buffer_size}"
        )
    if errors:
        raise ValueError("Invalid RJ45 VBD solver configuration: " + "; ".join(errors) + ".")


def rj45_reset_physics_contract() -> dict[str, object]:
    """Return physical constants that invalidate persisted RJ45 reset states.

    The result contains only serialization-stable Python scalars and tuples so
    dataset manifests can hash it directly.
    """
    drive_cfg = Rj45InsertionDriveCfg()
    return {
        "contract_version": 1,
        "newton_source_commit": NEWTON_RJ45_SOURCE_COMMIT,
        "asset_sha256": RJ45_ASSET_SHA256,
        "task_body_order": ("plug", "latch", "cable_segment_00..34"),
        "task_body_count": TASK_BODY_COUNT,
        "connector_contact_ke": CONTACT_KE,
        "connector_contact_kd": CONTACT_KD,
        "connector_gap": CONNECTOR_GAP,
        "connector_density": CONNECTOR_DENSITY,
        "rigid_gap": RIGID_GAP,
        "support_plane_local_height": SUPPORT_PLANE_LOCAL_HEIGHT,
        "support_plane_contact_ke": SUPPORT_PLANE_CONTACT_KE,
        "support_plane_contact_kd": SUPPORT_PLANE_CONTACT_KD,
        "support_plane_contact_kf": SUPPORT_PLANE_CONTACT_KF,
        "support_plane_friction": SUPPORT_PLANE_MU,
        "support_plane_robot_collision_policy": "all-robot-shapes-filtered",
        "mesh_sdf_max_resolution": MESH_SDF_MAX_RESOLUTION,
        "mesh_sdf_narrow_band_range": MESH_SDF_NARROW_BAND_RANGE,
        "plug_y_offset": PLUG_Y_OFFSET,
        "insertion_distance": INSERTION_DISTANCE,
        "cable_radius": CABLE_RADIUS,
        "cable_segment_count": CABLE_SEGMENT_COUNT,
        "cable_kinematic_count": CABLE_KINEMATIC_COUNT,
        "cable_friction": CABLE_MU,
        "cable_bend_stiffness": CABLE_BEND_STIFFNESS,
        "cable_bend_damping": CABLE_BEND_DAMPING,
        "latch_limit_lower": LATCH_LIMIT_LOWER,
        "latch_limit_upper": LATCH_LIMIT_UPPER,
        "latch_spring_ke": LATCH_SPRING_KE,
        "latch_spring_kd": LATCH_SPRING_KD,
        "latch_limit_kd": LATCH_LIMIT_KD,
        "grasp_proxy_center": GRASP_PROXY_CENTER,
        "grasp_proxy_half_extents": GRASP_PROXY_HALF_EXTENTS,
        "grasp_proxy_density": 0.0,
        "grasp_friction": GRASP_FRICTION,
        "grasp_robot_body_suffixes": _FRANKA_GRASP_BODY_SUFFIXES,
        "grasp_collision_policy": "finger-proxy-only",
        "goal_drive_stiffness": drive_cfg.stiffness,
        "goal_drive_damping": drive_cfg.damping,
        "vbd_rigid_compliant_alm": True,
        "vbd_legacy_rigid_contact_hard": RJ45_VBD_LEGACY_CONTACT_HARD,
        "vbd_iterations": RJ45_VBD_ITERATIONS,
        "vbd_rigid_body_contact_buffer_size": RJ45_VBD_CONTACT_BUFFER_SIZE,
    }


def _current_world_range(builder: newton.ModelBuilder, prefix: str, env_id: int) -> range:
    """Return the contiguous tail assigned to the currently open world."""
    worlds = getattr(builder, f"{prefix}_world")
    stop = len(worlds)
    start = stop
    while start > 0 and int(worlds[start - 1]) == env_id:
        start -= 1
    return range(start, stop)


def resolve_franka_grasp_shape_ids(builder: newton.ModelBuilder, env_id: int) -> FrankaGraspShapeIds:
    """Resolve the imported Franka hand and finger collision shapes.

    Body suffixes are used deliberately: Isaac Lab's Newton importer may put
    collision geometry beneath an extra ``Geometry`` scope, while the terminal
    Franka link names remain stable. A present robot must have exactly one body
    and at least one collidable shape for each hand/finger link.

    Args:
        builder: Newton builder with the current Isaac Lab world imported.
        env_id: Currently open environment/world index.

    Returns:
        Shape ids grouped by hand, left finger, and right finger. All groups
        are empty when building the assembly without a robot (for example in
        isolated physics tests).

    Raises:
        RuntimeError: If a world containing the robot has missing/duplicate
            grasp bodies or a grasp body has no collidable shape.
    """
    body_ids = _current_world_range(builder, "body", env_id)
    shape_ids = _current_world_range(builder, "shape", env_id)
    body_ids_by_suffix = {
        suffix: tuple(body_id for body_id in body_ids if str(builder.body_label[body_id]).endswith(suffix))
        for suffix in _FRANKA_GRASP_BODY_SUFFIXES
    }
    robot_is_present = any("/Robot/" in str(builder.body_label[body_id]) for body_id in body_ids)
    if not robot_is_present:
        return FrankaGraspShapeIds((), (), ())
    if any(len(ids) != 1 for ids in body_ids_by_suffix.values()):
        raise RuntimeError(f"RJ45 grasp filtering expected one Franka body per suffix, found {body_ids_by_suffix}.")

    collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
    shape_ids_by_suffix = {
        suffix: tuple(
            shape_id
            for shape_id in shape_ids
            if int(builder.shape_body[shape_id]) == ids[0] and int(builder.shape_flags[shape_id]) & collide_shapes
        )
        for suffix, ids in body_ids_by_suffix.items()
    }
    missing_shapes = {suffix: ids for suffix, ids in shape_ids_by_suffix.items() if not ids}
    if missing_shapes:
        raise RuntimeError(
            f"RJ45 grasp filtering requires collidable shapes on each Franka grasp body: {missing_shapes}."
        )
    return FrankaGraspShapeIds(
        hand_shape_ids=shape_ids_by_suffix[_FRANKA_HAND_BODY_SUFFIX],
        left_finger_shape_ids=shape_ids_by_suffix[_FRANKA_FINGER_BODY_SUFFIXES[0]],
        right_finger_shape_ids=shape_ids_by_suffix[_FRANKA_FINGER_BODY_SUFFIXES[1]],
    )


def configure_franka_finger_contact_material(builder: newton.ModelBuilder, env_id: int) -> tuple[int, ...]:
    """Set high friction on imported Franka finger collision shapes.

    Only friction is changed so the source rigid solver retains its authored
    contact-stiffness interpretation. The returned ids are the only imported
    shapes allowed to contact the task's invisible grasp proxy.

    Args:
        builder: Newton builder with the current Isaac Lab world imported.
        env_id: Currently open environment/world index.

    Returns:
        Collidable left/right finger shape indices.

    Raises:
        RuntimeError: If a world containing the Franka robot does not expose
            collidable shapes for both expected finger bodies.
    """
    finger_shape_ids = resolve_franka_grasp_shape_ids(builder, env_id).finger_shape_ids
    for shape_id in finger_shape_ids:
        builder.shape_material_mu[shape_id] = GRASP_FRICTION
    return finger_shape_ids


def _load_mesh(stage: Usd.Stage, prim_path: str) -> tuple[newton.Mesh, wp.vec3]:
    """Load one source mesh at its prim origin and construct its narrow-band SDF."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"Newton RJ45 asset is missing mesh prim {prim_path!r}.")
    usd_mesh = newton.usd.get_mesh(prim, load_normals=True)
    prim_transform = newton.usd.get_transform(prim, local=False)
    prim_position = wp.transform_get_translation(prim_transform)
    vertices = np.asarray(usd_mesh.vertices, dtype=np.float32)
    indices = np.asarray(usd_mesh.indices, dtype=np.int32)
    normals = np.asarray(usd_mesh.normals, dtype=np.float32) if usd_mesh.normals is not None else None
    mesh = newton.Mesh(vertices, indices, normals=normals)
    mesh.build_sdf(
        max_resolution=MESH_SDF_MAX_RESOLUTION,
        narrow_band_range=MESH_SDF_NARROW_BAND_RANGE,
        margin=CONNECTOR_GAP,
    )
    return mesh, prim_position


def _load_cable_centerline(stage: Usd.Stage) -> tuple[wp.vec3, ...]:
    """Load the exact source centerline and apply the unplugged Y offset."""
    prim = stage.GetPrimAtPath("/World/CableCurve")
    if not prim or not prim.IsA(UsdGeom.BasisCurves):
        raise RuntimeError("Newton RJ45 asset is missing /World/CableCurve.")
    points = UsdGeom.BasisCurves(prim).GetPointsAttr().Get()
    prim_position = wp.transform_get_translation(newton.usd.get_transform(prim, local=False))
    cable_points = tuple(
        wp.vec3(
            float(point[0]) + float(prim_position[0]),
            float(point[1]) + float(prim_position[1]) + PLUG_Y_OFFSET,
            float(point[2]) + float(prim_position[2]),
        )
        for point in points
    )
    if len(cable_points) != CABLE_SEGMENT_COUNT + 1:
        raise RuntimeError(f"Newton RJ45 cable must contain {CABLE_SEGMENT_COUNT + 1} points, got {len(cable_points)}.")
    return cable_points


def _load_geometry(asset_path: Path) -> _Rj45Geometry:
    """Load and validate all immutable task geometry once."""
    verify_rj45_asset(asset_path)
    stage = Usd.Stage.Open(str(asset_path))
    if stage is None:
        raise RuntimeError(f"Failed to open Newton RJ45 asset: {asset_path}")
    socket_mesh, socket_position = _load_mesh(stage, "/World/Socket")
    plug_mesh, plug_position = _load_mesh(stage, "/World/Plug")
    latch_mesh, latch_position = _load_mesh(stage, "/World/Latch")
    cable_points = _load_cable_centerline(stage)
    cable_quaternions = tuple(newton.utils.create_parallel_transport_cable_quaternions(cable_points))
    unplugged_plug_position = plug_position + wp.vec3(0.0, PLUG_Y_OFFSET, 0.0)
    anchor_offsets = tuple(
        0.5 * (cable_points[index] + cable_points[index + 1]) - unplugged_plug_position
        for index in range(CABLE_KINEMATIC_COUNT)
    )
    anchor_rotations = cable_quaternions[:CABLE_KINEMATIC_COUNT]
    align_start = CABLE_KINEMATIC_COUNT - 1
    align_next_start_offsets = tuple(
        wp.vec3(0.0, 0.0, -0.5 * float(wp.length(cable_points[index + 2] - cable_points[index + 1])))
        for index in range(align_start, CABLE_SEGMENT_COUNT - 1)
    )
    return _Rj45Geometry(
        socket_mesh=socket_mesh,
        socket_position=socket_position,
        plug_mesh=plug_mesh,
        plug_position=plug_position,
        latch_mesh=latch_mesh,
        latch_position=latch_position,
        cable_points=cable_points,
        cable_quaternions=cable_quaternions,
        anchor_offsets=anchor_offsets,
        anchor_rotations=anchor_rotations,
        align_next_start_offsets=align_next_start_offsets,
    )


def _as_transform_tuple(transform: wp.transform) -> tuple[float, ...]:
    """Convert a Warp transform to a stable xyzw tuple without device work."""
    return tuple(float(value) for value in transform)


def _as_vec3_tuple(vector: wp.vec3) -> tuple[float, float, float]:
    """Convert a Warp vector to a stable tuple without device work."""
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def _world_transform(position: Sequence[float], quaternion: Sequence[float]) -> wp.transform:
    """Construct and validate one hook-provided environment transform."""
    if len(position) != 3 or len(quaternion) != 4:
        raise ValueError("RJ45 world placement requires xyz position and xyzw quaternion.")
    values = tuple(float(value) for value in (*position, *quaternion))
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"RJ45 world placement must be finite, got {values}.")
    quaternion_norm = math.sqrt(sum(value * value for value in values[3:]))
    if quaternion_norm < 1.0e-8:
        raise ValueError("RJ45 world placement quaternion must have non-zero norm.")
    rotation = wp.quat(*(value / quaternion_norm for value in values[3:]))
    return wp.transform(wp.vec3(*values[:3]), rotation)


def _compose_transform(world_transform: wp.transform, position: wp.vec3, rotation: wp.quat) -> wp.transform:
    """Compose an asset-local transform with its environment placement."""
    return wp.transform_multiply(world_transform, wp.transform(position, rotation))


def _ensure_vbd_attributes(builder: newton.ModelBuilder) -> None:
    """Idempotently register the VBD joint attributes used by the reference."""
    present = tuple(builder.has_custom_attribute(name) for name in _VBD_ATTRIBUTE_NAMES)
    if all(present):
        return
    if any(present):
        missing = [name for name, is_present in zip(_VBD_ATTRIBUTE_NAMES, present, strict=True) if not is_present]
        raise RuntimeError(f"Newton builder has a partial VBD attribute registration; missing {missing}.")
    SolverVBD.register_custom_attributes(builder)


class Rj45NewtonAssemblyBuilder:
    """Add one exact Newton RJ45 assembly to each replicated Isaac Lab world.

    Create one instance before constructing the environment, pass
    :meth:`world_hook` to ``newton_builder_world_hook``, and call :meth:`bind`
    after Newton finalizes the model. One builder instance is intentionally
    single-use so recorded model indices cannot silently refer to another
    builder.

    Args:
        asset_path: Task-local Newton RJ45 USD asset.
        drive_cfg: Optional goal-generation drive gains.
        task_translation: Assembly translation in the environment frame [m].
        task_rotation_xyzw: Assembly orientation in the environment frame.
    """

    def __init__(
        self,
        asset_path: str | Path = RJ45_ASSET_PATH,
        drive_cfg: Rj45InsertionDriveCfg | None = None,
        task_translation: Sequence[float] = (0.0, 0.0, 0.0),
        task_rotation_xyzw: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    ) -> None:
        self.asset_path = Path(asset_path).resolve()
        self.drive_cfg = drive_cfg or Rj45InsertionDriveCfg()
        self._task_transform = _world_transform(task_translation, task_rotation_xyzw)
        self._geometry: _Rj45Geometry | None = None
        self._builder: newton.ModelBuilder | None = None
        self._records: dict[int, _WorldBuildRecord] = {}
        self._runtime: Rj45NewtonAssembly | None = None
        self._global_collision_shape_ids: tuple[int, ...] | None = None

    @property
    def world_body_ids(self) -> tuple[Rj45WorldBodyIds, ...]:
        """Per-world indices in ascending world order."""
        return tuple(record.ids for _, record in sorted(self._records.items()))

    def world_hook(
        self,
        builder: newton.ModelBuilder,
        env_id: int,
        position: list[float],
        quaternion: list[float],
    ) -> None:
        """Add an RJ45 assembly to the currently open Newton world.

        Args:
            builder: Builder owned by Newton replication.
            env_id: Zero-based environment/world index.
            position: Environment origin [m].
            quaternion: Environment orientation quaternion in xyzw order.

        Raises:
            RuntimeError: If reused with a second builder, called outside the
                matching world, or called twice for one world.
        """
        if self._runtime is not None:
            raise RuntimeError("Cannot add RJ45 worlds after binding the finalized Newton model.")
        if self._builder is None:
            self._builder = builder
        elif builder is not self._builder:
            raise RuntimeError("Rj45NewtonAssemblyBuilder is single-use and cannot extend multiple builders.")
        if builder.current_world != env_id:
            raise RuntimeError(f"RJ45 world hook expected builder.current_world={env_id}, got {builder.current_world}.")
        if env_id in self._records:
            raise RuntimeError(f"RJ45 world {env_id} was already added.")

        _ensure_vbd_attributes(builder)
        builder.rigid_gap = RIGID_GAP
        if self._geometry is None:
            self._geometry = _load_geometry(self.asset_path)
        franka_grasp_shape_ids = resolve_franka_grasp_shape_ids(builder, env_id)
        finger_shape_ids = franka_grasp_shape_ids.finger_shape_ids
        for shape_id in finger_shape_ids:
            builder.shape_material_mu[shape_id] = GRASP_FRICTION
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        current_shape_ids = _current_world_range(builder, "shape", env_id)
        robot_collision_shape_ids = tuple(
            shape_id
            for shape_id in current_shape_ids
            if int(builder.shape_flags[shape_id]) & collide_shapes
            and int(builder.shape_body[shape_id]) >= 0
            and "/Robot/" in str(builder.body_label[int(builder.shape_body[shape_id])])
        )
        if self._global_collision_shape_ids is None:
            self._global_collision_shape_ids = tuple(
                shape_id
                for shape_id, world_id in enumerate(builder.shape_world)
                if int(world_id) == -1 and int(builder.shape_flags[shape_id]) & collide_shapes
            )
        finger_shape_id_set = set(finger_shape_ids)
        nonfinger_collision_shape_ids = (
            *self._global_collision_shape_ids,
            *(
                shape_id
                for shape_id in current_shape_ids
                if shape_id not in finger_shape_id_set and int(builder.shape_flags[shape_id]) & collide_shapes
            ),
        )
        self._records[env_id] = self._add_world(
            builder,
            env_id,
            position,
            quaternion,
            self._geometry,
            franka_grasp_shape_ids,
            robot_collision_shape_ids,
            nonfinger_collision_shape_ids,
        )

    def _add_world(
        self,
        builder: newton.ModelBuilder,
        env_id: int,
        position: Sequence[float],
        quaternion: Sequence[float],
        geometry: _Rj45Geometry,
        franka_grasp_shape_ids: FrankaGraspShapeIds,
        robot_collision_shape_ids: tuple[int, ...],
        nonfinger_collision_shape_ids: tuple[int, ...],
    ) -> _WorldBuildRecord:
        """Construct one world while preserving Newton's reference ordering."""
        env_tf = _world_transform(position, quaternion)
        world_tf = wp.transform_multiply(env_tf, self._task_transform)
        world_rotation = wp.transform_get_rotation(world_tf)
        root_label = f"/World/envs/env_{env_id}/Rj45Assembly"

        # The source example places its cable on an infinite ground plane at
        # assembly-local z=0. Using the complete task transform preserves that
        # boundary after Isaac Lab translates a world or the task itself.
        support_plane_shape = builder.add_shape_plane(
            xform=_compose_transform(world_tf, wp.vec3(0.0, 0.0, SUPPORT_PLANE_LOCAL_HEIGHT), wp.quat_identity()),
            width=0.0,
            length=0.0,
            cfg=_SUPPORT_PLANE_SHAPE_CFG,
            label=f"{root_label}/SupportPlane/Collision",
        )
        # The source plane belongs only to the RJ45 assembly. Its infinite
        # extent would otherwise cut through the separately coupled Franka,
        # whose base is below the translated task-local z=0 surface.
        for robot_shape_id in robot_collision_shape_ids:
            builder.add_shape_collision_filter_pair(robot_shape_id, support_plane_shape)

        socket_tf = _compose_transform(world_tf, geometry.socket_position, wp.quat_identity())
        socket_shape = builder.add_shape_mesh(
            -1,
            mesh=geometry.socket_mesh,
            xform=socket_tf,
            cfg=_CONNECTOR_SHAPE_CFG,
            label=f"{root_label}/Socket/Collision",
        )

        unplugged_plug_position = geometry.plug_position + wp.vec3(0.0, PLUG_Y_OFFSET, 0.0)
        plug_tf = _compose_transform(world_tf, unplugged_plug_position, wp.quat_identity())
        plug_body = builder.add_link(xform=plug_tf, label=f"{root_label}/Plug")
        plug_shape = builder.add_shape_mesh(
            plug_body,
            mesh=geometry.plug_mesh,
            cfg=_CONNECTOR_SHAPE_CFG,
            label=f"{root_label}/Plug/Collision",
        )
        # The exact Newton plug SDF remains the insertion geometry. This
        # massless, invisible box only makes the rear housing reliably
        # graspable through the coupled Franka finger proxies.
        plug_mass_properties = (
            builder.body_mass[plug_body],
            builder.body_inv_mass[plug_body],
            builder.body_com[plug_body],
            builder.body_inertia[plug_body],
            builder.body_inv_inertia[plug_body],
        )
        grasp_proxy_shape = builder.add_shape_box(
            plug_body,
            xform=wp.transform(wp.vec3(*GRASP_PROXY_CENTER), wp.quat_identity()),
            hx=GRASP_PROXY_HALF_EXTENTS[0],
            hy=GRASP_PROXY_HALF_EXTENTS[1],
            hz=GRASP_PROXY_HALF_EXTENTS[2],
            cfg=_GRASP_PROXY_SHAPE_CFG,
            label=f"{root_label}/Plug/GraspProxy",
        )
        (
            builder.body_mass[plug_body],
            builder.body_inv_mass[plug_body],
            builder.body_com[plug_body],
            builder.body_inertia[plug_body],
            builder.body_inv_inertia[plug_body],
        ) = plug_mass_properties

        unplugged_latch_position = geometry.latch_position + wp.vec3(0.0, PLUG_Y_OFFSET, 0.0)
        latch_tf = _compose_transform(world_tf, unplugged_latch_position, wp.quat_identity())
        latch_body = builder.add_link(xform=latch_tf, label=f"{root_label}/Latch")
        latch_shape = builder.add_shape_mesh(
            latch_body,
            mesh=geometry.latch_mesh,
            cfg=_CONNECTOR_SHAPE_CFG,
            label=f"{root_label}/Latch/Collision",
        )
        connector_shapes = (socket_shape, plug_shape, latch_shape)

        joint_dof = newton.ModelBuilder.JointDofConfig
        d6_joint = builder.add_joint_d6(
            parent=-1,
            child=plug_body,
            linear_axes=(
                joint_dof(axis=(1.0, 0.0, 0.0)),
                joint_dof(axis=(0.0, 1.0, 0.0)),
                joint_dof(axis=(0.0, 0.0, 1.0)),
            ),
            angular_axes=None,
            parent_xform=plug_tf,
            child_xform=wp.transform_identity(),
            label=f"{root_label}/Plug/TranslationJoint",
            custom_attributes={"vbd:joint_is_hard": 0},
        )
        latch_joint = builder.add_joint_revolute(
            parent=plug_body,
            child=latch_body,
            axis=(-1.0, 0.0, 0.0),
            parent_xform=wp.transform(geometry.latch_position - geometry.plug_position, wp.quat_identity()),
            child_xform=wp.transform_identity(),
            target_ke=LATCH_SPRING_KE,
            target_kd=LATCH_SPRING_KD,
            limit_lower=LATCH_LIMIT_LOWER,
            limit_upper=LATCH_LIMIT_UPPER,
            limit_kd=LATCH_LIMIT_KD,
            collision_filter_parent=True,
            label=f"{root_label}/Latch/HingeJoint",
            custom_attributes={"vbd:joint_is_hard": 0},
        )
        builder.add_articulation(
            [d6_joint, latch_joint],
            label=f"{root_label}/ConnectorArticulation",
        )

        cable_points_w = tuple(wp.transform_point(world_tf, point) for point in geometry.cable_points)
        cable_quaternions_w = tuple(
            wp.normalize(wp.mul(world_rotation, rotation)) for rotation in geometry.cable_quaternions
        )
        cable_bodies, cable_joints = builder.add_rod(
            positions=cable_points_w,
            quaternions=cable_quaternions_w,
            radius=CABLE_RADIUS,
            cfg=dataclasses.replace(
                builder.default_shape_cfg,
                ke=CONTACT_KE,
                kd=CONTACT_KD,
                mu=CABLE_MU,
            ),
            bend_stiffness=CABLE_BEND_STIFFNESS,
            bend_damping=CABLE_BEND_DAMPING,
            label=f"{root_label}/Cable",
            body_frame_origin="com",
        )
        if len(cable_bodies) != CABLE_SEGMENT_COUNT or len(cable_joints) != CABLE_SEGMENT_COUNT - 1:
            raise RuntimeError(
                "Newton RJ45 rod topology changed: "
                f"expected {CABLE_SEGMENT_COUNT} bodies/{CABLE_SEGMENT_COUNT - 1} joints, "
                f"got {len(cable_bodies)} bodies/{len(cable_joints)} joints."
            )

        for segment_id, body_id in enumerate(cable_bodies):
            segment_label = f"{root_label}/Cable/Segment_{segment_id:02d}"
            builder.body_label[body_id] = segment_label
            segment_shapes = builder.body_shapes[body_id]
            if len(segment_shapes) != 1:
                raise RuntimeError(f"RJ45 cable segment {segment_id} must own exactly one capsule shape.")
            builder.shape_label[segment_shapes[0]] = f"{segment_label}/Collision"
        for joint_id, cable_joint_id in enumerate(cable_joints):
            builder.joint_label[cable_joint_id] = f"{root_label}/Cable/Joint_{joint_id:02d}_{joint_id + 1:02d}"
        cable_articulation_id = int(builder.joint_articulation[cable_joints[0]])
        builder.articulation_label[cable_articulation_id] = f"{root_label}/Cable/Articulation"

        for body_id in cable_bodies[:CABLE_KINEMATIC_COUNT]:
            for cable_shape in builder.body_shapes[body_id]:
                for connector_shape in connector_shapes:
                    builder.add_shape_collision_filter_pair(cable_shape, connector_shape)

        # The hand/finger bodies are staggered rigid proxies from the robot's
        # MJWarp entry. Letting those proxy meshes also collide with Newton's
        # thin official connector meshes over-constrains a grasp and can eject
        # the plug after an IK reset. Their only task contact is therefore the
        # controlled finger-to-GraspProxy interface. Socket clearance is also
        # filtered so a seated plug cannot transmit a second, coupled impulse
        # through the nearby hand; socket/plug/latch/cable physics themselves
        # remain exactly the official Newton assembly.
        official_task_shapes = [support_plane_shape, socket_shape, plug_shape, latch_shape]
        official_task_shapes.extend(builder.body_shapes[body_id][0] for body_id in cable_bodies)
        for robot_shape_id in franka_grasp_shape_ids.all_shape_ids:
            for task_shape_id in official_task_shapes:
                builder.add_shape_collision_filter_pair(robot_shape_id, task_shape_id)

        # The proxy is an Isaac Lab grasp aid, not part of Newton's connector,
        # and must not introduce a second plug/support contact patch.
        builder.add_shape_collision_filter_pair(grasp_proxy_shape, support_plane_shape)

        # The grasp proxy is deliberately finger-only. Its overlap with the
        # official connector/cable geometry must not alter insertion physics.
        for shape_id in (*nonfinger_collision_shape_ids, *connector_shapes):
            builder.add_shape_collision_filter_pair(grasp_proxy_shape, shape_id)
        for body_id in cable_bodies:
            for cable_shape in builder.body_shapes[body_id]:
                builder.add_shape_collision_filter_pair(grasp_proxy_shape, cable_shape)

        for body_id in (*cable_bodies[:CABLE_KINEMATIC_COUNT], cable_bodies[-1]):
            builder.body_mass[body_id] = 0.0
            builder.body_inv_mass[body_id] = 0.0
            builder.body_inertia[body_id] = wp.mat33(0.0)
            builder.body_inv_inertia[body_id] = wp.mat33(0.0)

        ids = Rj45WorldBodyIds(
            world_id=env_id,
            support_plane_shape_id=support_plane_shape,
            socket_shape_id=socket_shape,
            plug_body_id=plug_body,
            plug_shape_id=plug_shape,
            grasp_proxy_shape_id=grasp_proxy_shape,
            latch_body_id=latch_body,
            latch_shape_id=latch_shape,
            d6_joint_id=d6_joint,
            latch_joint_id=latch_joint,
            cable_body_ids=tuple(cable_bodies),
            cable_joint_ids=tuple(cable_joints),
        )
        default_body_q = tuple(_as_transform_tuple(builder.body_q[body_id]) for body_id in ids.task_body_ids)
        goal_target_w = _as_vec3_tuple(wp.transform_point(world_tf, geometry.plug_position))
        return _WorldBuildRecord(
            ids=ids,
            root_label=root_label,
            default_body_q=default_body_q,
            goal_target_w=goal_target_w,
        )

    def bind(self, model: newton.Model) -> Rj45NewtonAssembly:
        """Bind recorded indices and immutable defaults to a finalized model.

        Args:
            model: Finalized Newton model built from the extended builder.

        Returns:
            Device-resident graph-safe RJ45 runtime helpers.

        Raises:
            RuntimeError: If no worlds were built, world indices are not dense,
                or finalization did not preserve the recorded body labels.
        """
        if self._runtime is not None:
            if self._runtime.model is model:
                return self._runtime
            # A hard Newton reset re-finalizes the same builder. Recorded ids
            # remain stable, but every device array must bind to the new model.
            self._runtime = None
        if not self._records:
            raise RuntimeError("Cannot bind an RJ45 assembly before its world hook builds any worlds.")
        records = tuple(record for _, record in sorted(self._records.items()))
        world_ids = tuple(record.ids.world_id for record in records)
        if world_ids != tuple(range(len(records))):
            raise RuntimeError(f"RJ45 world ids must be dense and zero-based, got {world_ids}.")
        if model.body_label is None:
            raise RuntimeError("Finalized Newton model does not expose body labels.")
        for record in records:
            expected_labels = (
                f"{record.root_label}/Plug",
                f"{record.root_label}/Latch",
                *(f"{record.root_label}/Cable/Segment_{index:02d}" for index in range(CABLE_SEGMENT_COUNT)),
            )
            for body_id, expected_label in zip(record.ids.task_body_ids, expected_labels, strict=True):
                if body_id >= model.body_count or str(model.body_label[body_id]) != expected_label:
                    actual = None if body_id >= model.body_count else str(model.body_label[body_id])
                    raise RuntimeError(
                        f"RJ45 model body index {body_id} changed during finalization: "
                        f"expected {expected_label!r}, got {actual!r}."
                    )
        if self._geometry is None:
            raise RuntimeError("RJ45 geometry was unexpectedly released before model binding.")
        self._runtime = Rj45NewtonAssembly(model, records, self._geometry, self.drive_cfg)
        return self._runtime


class Rj45NewtonAssembly:
    """Device-resident runtime controls and reset layout for batched RJ45 worlds.

    Stepping methods only launch fixed-shape Warp kernels and are safe to call
    from Newton callbacks captured in a CUDA graph. Call :meth:`prepare_step`
    after forces are cleared and before each VBD solve, and call
    :meth:`align_after_step` on the resulting state after each solve.
    """

    def __init__(
        self,
        model: newton.Model,
        records: tuple[_WorldBuildRecord, ...],
        geometry: _Rj45Geometry,
        drive_cfg: Rj45InsertionDriveCfg,
    ) -> None:
        self.model = model
        self.num_worlds = len(records)
        self.drive_cfg = drive_cfg
        device = model.device
        world_ids = [record.ids.world_id for record in records]
        cable_body_ids = [record.ids.cable_body_ids for record in records]
        align_start = CABLE_KINEMATIC_COUNT - 1

        self.world_ids = wp.array(world_ids, dtype=wp.int32, device=device)
        self.socket_shape_ids = wp.array(
            [record.ids.socket_shape_id for record in records], dtype=wp.int32, device=device
        )
        self.support_plane_shape_ids = wp.array(
            [record.ids.support_plane_shape_id for record in records], dtype=wp.int32, device=device
        )
        self.plug_body_ids = wp.array([record.ids.plug_body_id for record in records], dtype=wp.int32, device=device)
        self.plug_shape_ids = wp.array([record.ids.plug_shape_id for record in records], dtype=wp.int32, device=device)
        self.grasp_proxy_shape_ids = wp.array(
            [record.ids.grasp_proxy_shape_id for record in records], dtype=wp.int32, device=device
        )
        self.latch_body_ids = wp.array([record.ids.latch_body_id for record in records], dtype=wp.int32, device=device)
        self.latch_shape_ids = wp.array(
            [record.ids.latch_shape_id for record in records], dtype=wp.int32, device=device
        )
        self.cable_body_ids = wp.array(cable_body_ids, dtype=wp.int32, device=device)
        self.cable_anchor_body_ids = wp.array(
            [ids[:CABLE_KINEMATIC_COUNT] for ids in cable_body_ids], dtype=wp.int32, device=device
        )
        self.pinned_cable_body_ids = wp.array([ids[-1] for ids in cable_body_ids], dtype=wp.int32, device=device)
        self.task_body_ids = wp.array([record.ids.task_body_ids for record in records], dtype=wp.int32, device=device)
        self.default_body_q = wp.array([record.default_body_q for record in records], dtype=wp.transform, device=device)
        self.default_goal_target_w = wp.array(
            [record.goal_target_w for record in records], dtype=wp.vec3, device=device
        )
        self.drive_target_w = wp.clone(self.default_goal_target_w)
        self.drive_enabled = wp.zeros(self.num_worlds, dtype=wp.bool, device=device)
        self._all_env_mask = wp.ones(self.num_worlds, dtype=wp.bool, device=device)
        self._anchor_offsets = wp.array(geometry.anchor_offsets, dtype=wp.vec3, device=device)
        self._anchor_rotations = wp.array(geometry.anchor_rotations, dtype=wp.quat, device=device)
        self._align_body_ids = wp.array([ids[align_start:-1] for ids in cable_body_ids], dtype=wp.int32, device=device)
        self._align_next_body_ids = wp.array(
            [ids[align_start + 1 :] for ids in cable_body_ids], dtype=wp.int32, device=device
        )
        self._align_next_start_offsets = wp.array(geometry.align_next_start_offsets, dtype=wp.vec3, device=device)
        self._align_count = len(geometry.align_next_start_offsets)

    def prepare_step(self, state: newton.State) -> None:
        """Apply anti-gravity/drive forces and synchronize cable anchors.

        Args:
            state: Newton input state whose force buffer has just been cleared.
        """
        if self.model.gravity is None:
            raise RuntimeError("Newton model gravity is unavailable for RJ45 anti-gravity.")
        wp.launch(
            apply_connector_forces,
            dim=self.num_worlds,
            inputs=[
                state.body_q,
                state.body_qd,
                state.body_f,
                self.model.body_mass,
                self.model.gravity,
                self.world_ids,
                self.plug_body_ids,
                self.latch_body_ids,
                self.drive_enabled,
                self.drive_target_w,
                self.drive_cfg.stiffness,
                self.drive_cfg.damping,
            ],
            device=self.model.device,
        )
        self.sync_anchors(state)

    def sync_anchors(self, state: newton.State) -> None:
        """Teleport the four kinematic cable anchors to their plug-relative poses."""
        wp.launch(
            sync_cable_anchors,
            dim=(self.num_worlds, CABLE_KINEMATIC_COUNT),
            inputs=[
                state.body_q,
                state.body_qd,
                self.plug_body_ids,
                self.cable_anchor_body_ids,
                self._anchor_offsets,
                self._anchor_rotations,
            ],
            device=self.model.device,
        )

    def align_after_step(self, state: newton.State) -> None:
        """Align cable capsule collision/render orientation after integration."""
        wp.launch(
            align_cable_orientations,
            dim=(self.num_worlds, self._align_count),
            inputs=[
                state.body_q,
                self._align_body_ids,
                self._align_next_body_ids,
                self._align_next_start_offsets,
            ],
            device=self.model.device,
        )

    def reset_to_default(
        self,
        states: newton.State | Iterable[newton.State],
        env_mask: wp.array[wp.bool] | None = None,
    ) -> None:
        """Restore the unplugged authored state in selected worlds.

        Args:
            states: One state or all double-buffered Newton states to update.
            env_mask: Selected environments, shape ``[num_worlds]``. Defaults
                to all worlds.

        Note:
            After direct state writes, the caller must invoke
            ``NewtonManager.invalidate_body_state(env_mask=...)`` so VBD clears
            per-world constraint/contact history.
        """
        mask = self._resolve_env_mask(env_mask)
        for state in self._iter_states(states):
            wp.launch(
                reset_task_bodies,
                dim=(self.num_worlds, TASK_BODY_COUNT),
                inputs=[mask, self.task_body_ids, self.default_body_q, state.body_q, state.body_qd],
                device=self.model.device,
            )

    def write_state(
        self,
        states: newton.State | Iterable[newton.State],
        body_q: wp.array2d[wp.transform],
        body_qd: wp.array2d[wp.spatial_vector],
        env_mask: wp.array[wp.bool] | None = None,
    ) -> None:
        """Write a complete persisted RJ45 state in stable task-body order.

        Args:
            states: One state or all double-buffered Newton states to update.
            body_q: Body poses [m and unitless xyzw], shape
                ``[num_worlds, TASK_BODY_COUNT]``.
            body_qd: COM velocities [m/s, rad/s], shape
                ``[num_worlds, TASK_BODY_COUNT]``.
            env_mask: Selected environments, shape ``[num_worlds]``. Defaults
                to all worlds.
        """
        expected_shape = (self.num_worlds, TASK_BODY_COUNT)
        if body_q.shape != expected_shape or body_qd.shape != expected_shape:
            raise ValueError(
                f"RJ45 state arrays must have shape {expected_shape}, got body_q={body_q.shape}, "
                f"body_qd={body_qd.shape}."
            )
        mask = self._resolve_env_mask(env_mask)
        for state in self._iter_states(states):
            wp.launch(
                write_task_body_state,
                dim=expected_shape,
                inputs=[mask, self.task_body_ids, body_q, body_qd, state.body_q, state.body_qd],
                device=self.model.device,
            )

    def set_drive_enabled(self, enabled: bool, env_mask: wp.array[wp.bool] | None = None) -> None:
        """Enable or disable the insertion drive for selected worlds.

        Args:
            enabled: Whether to apply the goal-generation drive.
            env_mask: Selected environments. Defaults to all worlds.
        """
        mask = self._resolve_env_mask(env_mask)
        wp.launch(
            set_drive_enabled_masked,
            dim=self.num_worlds,
            inputs=[mask, bool(enabled), self.drive_enabled],
            device=self.model.device,
        )

    def restore_goal_drive_targets(self, env_mask: wp.array[wp.bool] | None = None) -> None:
        """Restore the USD-authored nominal plug targets in selected worlds."""
        mask = self._resolve_env_mask(env_mask)
        wp.launch(
            restore_goal_targets_masked,
            dim=self.num_worlds,
            inputs=[mask, self.default_goal_target_w, self.drive_target_w],
            device=self.model.device,
        )

    def write_drive_targets(
        self,
        targets_w: wp.array[wp.vec3],
        env_mask: wp.array[wp.bool] | None = None,
    ) -> None:
        """Set world-frame positional drive targets [m] for selected worlds."""
        if targets_w.shape != (self.num_worlds,):
            raise ValueError(f"RJ45 drive targets must have shape {(self.num_worlds,)}, got {targets_w.shape}.")
        mask = self._resolve_env_mask(env_mask)
        wp.launch(
            write_drive_targets_masked,
            dim=self.num_worlds,
            inputs=[mask, targets_w, self.drive_target_w],
            device=self.model.device,
        )

    def _resolve_env_mask(self, env_mask: wp.array[wp.bool] | None) -> wp.array[wp.bool]:
        """Validate and return a full-width environment mask."""
        if env_mask is None:
            return self._all_env_mask
        if env_mask.shape != (self.num_worlds,):
            raise ValueError(f"RJ45 environment mask must have shape {(self.num_worlds,)}, got {env_mask.shape}.")
        return env_mask

    @staticmethod
    def _iter_states(states: newton.State | Iterable[newton.State]) -> tuple[newton.State, ...]:
        """Normalize one or more Newton states without touching device data."""
        if hasattr(states, "body_q") and hasattr(states, "body_qd"):
            return (states,)  # type: ignore[return-value]
        normalized = tuple(states)  # type: ignore[arg-type]
        if not normalized:
            raise ValueError("At least one Newton state is required.")
        return normalized
