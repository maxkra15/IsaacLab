# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cable / 1D-rod asset class, registry entry, and replicate-hook plumbing."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import newton
import numpy as np
import warp as wp
from isaaclab_newton.assets.articulation.articulation import Articulation
from isaaclab_newton.physics import NewtonManager as SimulationManager

import isaaclab.sim as sim_utils

if TYPE_CHECKING:
    from .cable_object_cfg import CableObjectCfg

logger = logging.getLogger(__name__)


def _rotate_vector_by_quat_np(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector by an ``xyzw`` quaternion."""

    quat = np.asarray(quat_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return vec
    quat = quat / norm
    q_vec = quat[:3]
    q_w = quat[3]
    t = 2.0 * np.cross(q_vec, vec)
    return vec + q_w * t + np.cross(q_vec, t)


def _resample_open_polyline(
    node_positions: list[wp.vec3],
    edges: list[tuple[int, int]],
    segment_length: float,
) -> tuple[list[wp.vec3], list[tuple[int, int]]]:
    """Arc-length-resample an ordered open polyline to ~uniform segment length.

    The two endpoints are preserved exactly. Only a simple sequential chain
    (edges ``(0, 1), (1, 2), ...``) is resampled; any other topology or a
    degenerate/too-short curve is returned unchanged. This mirrors the uniform
    resampling the Newton ``cable_robot`` reference applies to its rod, which
    avoids the stiffness/mass discontinuities that destabilize a rod built from
    non-uniform authored control points (e.g. a long lead-in segment).
    """
    n = len(node_positions)
    if n < 3 or not segment_length or segment_length <= 0.0:
        return node_positions, edges
    if list(edges) != [(i, i + 1) for i in range(n - 1)]:
        return node_positions, edges
    pts = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in node_positions], dtype=np.float64)
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
    total = float(arc[-1])
    if total <= 0.0:
        return node_positions, edges
    num_segments = max(1, int(round(total / float(segment_length))))
    if num_segments == n - 1:
        return node_positions, edges
    targets = np.linspace(0.0, total, num_segments + 1)
    resampled = np.empty((num_segments + 1, 3), dtype=np.float64)
    for axis in range(3):
        resampled[:, axis] = np.interp(targets, arc, pts[:, axis])
    new_nodes = [wp.vec3(float(x), float(y), float(z)) for x, y, z in resampled]
    new_edges = [(i, i + 1) for i in range(num_segments)]
    return new_nodes, new_edges


@dataclass
class CableRegistryEntry:
    """Mutable bridge between :class:`CableObject` and the per-world replicate hook."""

    prim_path: str
    node_positions: list[wp.vec3]
    edges: list[tuple[int, int]]
    radius: float
    curve_prim_path: str = ""

    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    init_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    stretch_stiffness: float = 1.0e9
    bend_stiffness: float = 0.0
    stretch_damping: float = 0.0
    bend_damping: float = 0.0
    density: float = 1500.0

    body_offsets: list[int] = field(default_factory=list)
    last_edge_length: float = 0.0
    # Per-env Newton body indices of each cable segment in edge order; outer list
    # indexed by ``world_idx``, inner indexed by ``CableAttachmentCfg.cable_anchor``.
    segment_body_indices: list[list[int]] = field(default_factory=list)


_SDF_CAPTURE_MESH_CACHE: dict[tuple, newton.Mesh] = {}


def _quat_from_z_axis_to_vector_np(vec: np.ndarray) -> wp.quat:
    """Quaternion rotating local +Z onto ``vec``."""

    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return wp.quat_identity()
    target = vec / norm
    source = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 1.0 - 1.0e-10:
        return wp.quat_identity()
    if dot < -1.0 + 1.0e-10:
        return wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi)
    axis = np.cross(source, target)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-12:
        return wp.quat_identity()
    axis /= axis_norm
    angle = math.acos(dot)
    return wp.quat_from_axis_angle(wp.vec3(float(axis[0]), float(axis[1]), float(axis[2])), angle)


def _make_sdf_capture_mesh(cable_radius: float, segment_length: float, capture_cfg) -> newton.Mesh:
    """Build or return a cached closed SDF connector mesh for a cable segment."""

    retainer_hole_radius = (
        0.75 * float(cable_radius)
        if capture_cfg.retainer_hole_radius is None
        else float(capture_cfg.retainer_hole_radius)
    )
    retainer_hole_radius = max(0.2 * float(cable_radius), min(retainer_hole_radius, 0.95 * float(cable_radius)))
    bore_radius = max(float(cable_radius) + float(capture_cfg.bore_clearance), 1.05 * float(cable_radius))
    outer_radius = max(float(capture_cfg.outer_radius), bore_radius + 0.002)
    radial_segments = max(12, int(capture_cfg.radial_segments))
    retainer_offset = max(float(capture_cfg.retainer_offset), 1.2 * float(cable_radius))
    retainer_thickness = max(float(capture_cfg.retainer_thickness), 0.25 * float(cable_radius))
    tail_clearance = max(float(capture_cfg.tail_clearance), 1.0e-4)
    sdf_max_resolution = int(capture_cfg.sdf_max_resolution)
    device_key = str(wp.get_device())
    key = (
        round(float(cable_radius), 7),
        round(float(segment_length), 7),
        round(outer_radius, 7),
        round(bore_radius, 7),
        round(retainer_hole_radius, 7),
        round(retainer_offset, 7),
        round(retainer_thickness, 7),
        round(tail_clearance, 7),
        bool(capture_cfg.through_sleeve),
        radial_segments,
        sdf_max_resolution,
        device_key,
    )
    cached = _SDF_CAPTURE_MESH_CACHE.get(key)
    if cached is not None:
        return cached

    if bool(capture_cfg.through_sleeve):
        stations = (
            (-retainer_offset, bore_radius),
            (float(segment_length) + tail_clearance, bore_radius),
        )
    else:
        z0 = -retainer_offset - retainer_thickness
        z1 = -retainer_offset
        z2 = -1.05 * float(cable_radius)
        if z2 <= z1:
            z2 = 0.5 * (z1 + 0.0)
        z3 = float(segment_length) + 0.67 * tail_clearance
        z4 = float(segment_length) + tail_clearance
        stations = (
            (z0, retainer_hole_radius),
            (z1, retainer_hole_radius),
            (z2, bore_radius),
            (z3, bore_radius),
            (z4, 0.0),
        )

    vertices: list[tuple[float, float, float]] = []
    outer_rings: list[list[int]] = []
    inner_rings: list[list[int] | None] = []
    center_index: int | None = None

    angles = [2.0 * math.pi * i / radial_segments for i in range(radial_segments)]
    for z, inner_radius in stations:
        outer_ring = []
        for theta in angles:
            outer_ring.append(len(vertices))
            vertices.append((outer_radius * math.cos(theta), outer_radius * math.sin(theta), z))
        outer_rings.append(outer_ring)

        if inner_radius > 1.0e-8:
            inner_ring = []
            for theta in angles:
                inner_ring.append(len(vertices))
                vertices.append((inner_radius * math.cos(theta), inner_radius * math.sin(theta), z))
            inner_rings.append(inner_ring)
        else:
            center_index = len(vertices)
            vertices.append((0.0, 0.0, z))
            inner_rings.append(None)

    indices: list[tuple[int, int, int]] = []
    for ring_idx in range(len(stations) - 1):
        outer_a = outer_rings[ring_idx]
        outer_b = outer_rings[ring_idx + 1]
        for i in range(radial_segments):
            j = (i + 1) % radial_segments
            indices.append((outer_a[i], outer_a[j], outer_b[j]))
            indices.append((outer_a[i], outer_b[j], outer_b[i]))

        inner_a = inner_rings[ring_idx]
        inner_b = inner_rings[ring_idx + 1]
        if inner_a is not None and inner_b is not None:
            for i in range(radial_segments):
                j = (i + 1) % radial_segments
                indices.append((inner_a[i], inner_b[j], inner_a[j]))
                indices.append((inner_a[i], inner_b[i], inner_b[j]))
        elif inner_a is not None and inner_b is None and center_index is not None:
            for i in range(radial_segments):
                j = (i + 1) % radial_segments
                indices.append((inner_a[i], center_index, inner_a[j]))

    front_outer = outer_rings[0]
    front_inner = inner_rings[0]
    if front_inner is not None:
        for i in range(radial_segments):
            j = (i + 1) % radial_segments
            indices.append((front_outer[i], front_inner[j], front_outer[j]))
            indices.append((front_outer[i], front_inner[i], front_inner[j]))

    back_outer = outer_rings[-1]
    back_inner = inner_rings[-1]
    if center_index is not None:
        for i in range(radial_segments):
            j = (i + 1) % radial_segments
            indices.append((back_outer[i], back_outer[j], center_index))
    elif back_inner is not None:
        for i in range(radial_segments):
            j = (i + 1) % radial_segments
            indices.append((back_outer[i], back_outer[j], back_inner[j]))
            indices.append((back_outer[i], back_inner[j], back_inner[i]))

    mesh = newton.Mesh(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(indices, dtype=np.int32),
        compute_inertia=True,
        is_solid=True,
    )
    if wp.get_device().is_cuda:
        mesh.build_sdf(max_resolution=sdf_max_resolution)
    _SDF_CAPTURE_MESH_CACHE[key] = mesh
    return mesh


def add_cable_entry_to_builder(
    builder,
    entry: CableRegistryEntry,
    env_idx: int,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
    cable_idx: int = 0,
) -> None:
    """Add one cable to a Newton ``ModelBuilder`` for one environment.

    Args:
        builder: The Newton ``ModelBuilder``.
        entry: Registry entry describing the cable's geometry and material.
        env_idx: Zero-based environment (world) index.
        env_position: World translation ``[x, y, z]`` [m] for this environment.
        env_rotation: World orientation quaternion ``(x, y, z, w)`` for this environment.
        cable_idx: Zero-based cable index within :attr:`SimulationManager._cable_registry`.
    """
    if env_idx == 0:
        entry.body_offsets.clear()
        entry.segment_body_indices.clear()
        entry.last_edge_length = 0.0

    env_pos = wp.vec3(float(env_position[0]), float(env_position[1]), float(env_position[2]))
    env_rot = wp.quat(
        float(env_rotation[0]),
        float(env_rotation[1]),
        float(env_rotation[2]),
        float(env_rotation[3]),
    )
    init_pos = wp.vec3(float(entry.init_pos[0]), float(entry.init_pos[1]), float(entry.init_pos[2]))
    init_rot = wp.quat(
        float(entry.init_rot[0]),
        float(entry.init_rot[1]),
        float(entry.init_rot[2]),
        float(entry.init_rot[3]),
    )

    composed_pos = env_pos + wp.quat_rotate(env_rot, init_pos)
    composed_rot = env_rot * init_rot

    world_nodes: list[wp.vec3] = []
    for node in entry.node_positions:
        rotated = wp.quat_rotate(composed_rot, node)
        world_nodes.append(composed_pos + rotated)

    shape_cfg = newton.ModelBuilder.ShapeConfig()
    shape_cfg.density = float(entry.density)
    # Unique negative group disables segment-vs-segment self-collision (Newton
    # filters same-negative pairs) while keeping negative-vs-positive collisions.
    shape_cfg.collision_group = -(1 + cable_idx)

    # Pre-expand ``env_.*`` here: the cloner's ``_rename_builder_labels`` does
    # not visit builder-hook bodies, so we must produce the per-env label ourselves.
    expanded_prim_path = entry.prim_path.replace("env_.*", f"env_{env_idx}")
    entry.body_offsets.append(builder.body_count)
    # :meth:`add_rod_graph` only joins adjacent segments, leaving the chain root as an
    # orphan body that breaks :class:`SolverMuJoCo` topological walks (``KeyError`` on
    # ``body_mapping[parent]``). Anchor it with a free joint in its own articulation.
    rod_body_indices, _rod_joint_indices = builder.add_rod_graph(
        node_positions=world_nodes,
        edges=entry.edges,
        radius=entry.radius,
        cfg=shape_cfg,
        stretch_stiffness=entry.stretch_stiffness,
        stretch_damping=entry.stretch_damping,
        bend_stiffness=entry.bend_stiffness,
        bend_damping=entry.bend_damping,
        label=f"{expanded_prim_path}/cable",
    )
    # NOTE: if statement below can be removed once mujoco solver stops parsing cable bodies
    # even though it cannot solve cable joint.
    if rod_body_indices:
        root_joint_id = builder.add_joint_free(
            child=int(rod_body_indices[0]),
            label=f"{expanded_prim_path}/cable_root_free",
        )
        builder.add_articulation([root_joint_id], label=f"{expanded_prim_path}/cable_root_articulation")

    entry.segment_body_indices.append(list(rod_body_indices))
    if env_idx == 0:
        u, v = entry.edges[-1]
        entry.last_edge_length = float(wp.length(entry.node_positions[v] - entry.node_positions[u]))


def add_registered_cables_to_builder(
    builder,
    world_idx: int,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
) -> None:
    """Per-world hook that registers all cables in :attr:`SimulationManager._cable_registry`."""
    for cable_idx, entry in enumerate(SimulationManager._cable_registry):
        add_cable_entry_to_builder(builder, entry, world_idx, env_position, env_rotation, cable_idx=cable_idx)


def add_cable_sdf_captures_to_builder(
    builder,
    world_idx: int,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
) -> list[int]:
    """Per-world hook that adds static SDF connector meshes for cable captures."""

    pending = getattr(SimulationManager, "_pending_cable_sdf_captures", None)
    if not pending:
        return []

    env_pos = wp.vec3(float(env_position[0]), float(env_position[1]), float(env_position[2]))
    env_rot = wp.quat(
        float(env_rotation[0]),
        float(env_rotation[1]),
        float(env_rotation[2]),
        float(env_rotation[3]),
    )

    shape_ids: list[int] = []
    for cable_idx, capture in pending:
        entry = SimulationManager._cable_registry[cable_idx]
        num_segments = len(entry.edges)
        anchor_idx = int(capture.cable_anchor)
        if not -num_segments <= anchor_idx < num_segments:
            raise ValueError(
                f"CableSdfCaptureCfg.cable_anchor={anchor_idx} is out of range for cable"
                f" '{entry.prim_path}' with {num_segments} segments;"
                f" valid range is [-{num_segments}, {num_segments - 1}]."
            )
        edge_idx = anchor_idx if anchor_idx >= 0 else num_segments + anchor_idx
        u, v = entry.edges[edge_idx]
        local_start = entry.node_positions[u]
        local_end = entry.node_positions[v]
        local_edge = local_end - local_start
        segment_length = float(wp.length(local_edge))
        if not np.isfinite(segment_length) or segment_length <= 1.0e-8:
            continue

        init_pos = wp.vec3(float(entry.init_pos[0]), float(entry.init_pos[1]), float(entry.init_pos[2]))
        init_rot = wp.quat(
            float(entry.init_rot[0]),
            float(entry.init_rot[1]),
            float(entry.init_rot[2]),
            float(entry.init_rot[3]),
        )
        composed_pos = env_pos + wp.quat_rotate(env_rot, init_pos)
        composed_rot = env_rot * init_rot
        world_start = composed_pos + wp.quat_rotate(composed_rot, local_start)
        world_edge = wp.quat_rotate(composed_rot, local_edge)
        world_edge_np = np.array(
            [float(world_edge[0]), float(world_edge[1]), float(world_edge[2])],
            dtype=np.float64,
        )
        shape_xform = wp.transform(world_start, _quat_from_z_axis_to_vector_np(world_edge_np))

        mesh = _make_sdf_capture_mesh(float(entry.radius), segment_length, capture)
        shape_cfg = builder.default_shape_cfg.copy()
        shape_cfg.density = 0.0
        shape_cfg.margin = float(capture.margin)
        shape_cfg.gap = float(capture.gap)
        shape_cfg.mu = float(capture.mu)

        expanded_prim_path = entry.prim_path.replace("env_.*", f"env_{world_idx}")
        shape_ids.append(
            int(
                builder.add_shape_mesh(
                    body=-1,
                    mesh=mesh,
                    xform=shape_xform,
                    cfg=shape_cfg,
                    label=f"{expanded_prim_path}/{capture.label_suffix}",
                )
            )
        )

    return shape_ids


def _resolve_static_target_xform(
    target_path: str,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
) -> wp.transform | None:
    """World transform for a cable-attachment target that is a USD-only (non-body) prim.

    Static scene prims (e.g. an ``AssetBaseCfg`` world-static collider) have no Newton
    body. Resolve the source prim's pose relative to its ``env_N`` root, then place it
    at this world's env offset. Returns ``None`` if no prim matches ``target_path``.
    """
    prim = sim_utils.find_first_matching_prim(target_path)
    if prim is None:
        return None
    match = re.match(r"(?P<env>.*/env_\d+)", prim.GetPath().pathString)
    env_prim = prim.GetStage().GetPrimAtPath(match.group("env")) if match else None
    ref = env_prim if env_prim is not None and env_prim.IsValid() else None
    target_xform = wp.transform(*sim_utils.resolve_prim_pose(prim, ref_prim=ref))
    if ref is None:
        return target_xform
    return wp.transform_multiply(wp.transform(env_position, env_rotation), target_xform)


def apply_cable_attachments_to_builder(
    builder,
    world_idx: int,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
) -> list[int]:
    """Per-world hook that realizes pending cable attachments as Newton fixed joints.

    Args:
        builder: The Newton ``ModelBuilder`` for the current scene.
        world_idx: Zero-based environment (world) index for this invocation.
        env_position: World translation ``[x, y, z]`` [m] for this environment.
        env_rotation: World orientation quaternion ``(x, y, z, w)`` for this environment.
    """
    pending = getattr(SimulationManager, "_pending_cable_attachments", None)
    if not pending:
        return []

    joint_ids: list[int] = []
    joint_ids_by_cable: dict[int, list[int]] = {}
    ordered_pending = sorted(pending, key=lambda item: 0 if item[1].add_to_articulation else 1)
    for cable_idx, attachment in ordered_pending:
        entry = SimulationManager._cable_registry[cable_idx]
        segments_in_world = entry.segment_body_indices[world_idx]
        num_segments = len(segments_in_world)
        anchor_idx = attachment.cable_anchor
        if not -num_segments <= anchor_idx < num_segments:
            raise ValueError(
                f"CableAttachmentCfg.cable_anchor={anchor_idx} is out of range for cable"
                f" '{entry.prim_path}' with {num_segments} segments;"
                f" valid range is [-{num_segments}, {num_segments - 1}]."
            )
        cable_body_idx = segments_in_world[anchor_idx]

        # Match labels via the regex template *and* its per-env expansion:
        # rigid bodies added by the cloner via ``add_builder(proto)`` carry the
        # prototype's ``env_0/...`` label until the cloner's post-build rewrite
        # (which runs after these hooks), while builder-hook bodies (e.g. cables)
        # are already per-env expanded. The regex pattern accepts both. Filter
        # by ``body_world``: ``-1`` is Newton's "global" sentinel for bodies
        # added outside any ``begin_world``/``end_world`` block (e.g. single-env
        # flat path); world-specific matches win over global ones.
        target_path = attachment.target_prim_path
        expanded_target_path = target_path.replace("env_.*", f"env_{world_idx}")
        target_re = re.compile(target_path)
        body_label = builder.body_label
        body_world = builder.body_world
        # Only this world's bodies (and Newton's ``-1`` global sentinel) can match.
        # Filter on ``body_world`` first so the regex never runs against the other
        # worlds' bodies already accumulated in the shared builder: this hook runs
        # per world during replication, and scanning the full (growing) ``body_label``
        # made attachment resolution O(num_worlds**2) and dominated startup at scale.
        candidate_indices = np.flatnonzero((np.asarray(body_world) == world_idx) | (np.asarray(body_world) == -1))
        target_body_idx = -1
        for body_idx in candidate_indices:
            body_idx = int(body_idx)
            label = body_label[body_idx]
            if label == expanded_target_path or target_re.fullmatch(label):
                target_body_idx = body_idx
                break
        # No matching Newton body: the target may be a USD-only static scene prim
        # (e.g. an AssetBaseCfg world-static collider). Resolve its world pose so
        # the cable can be welded to that fixed point instead.
        static_target_xform = None
        if target_body_idx < 0:
            static_target_xform = _resolve_static_target_xform(target_path, env_position, env_rotation)
        if target_body_idx < 0 and static_target_xform is None:
            available_in_world = [body_label[i] for i in range(len(body_label)) if body_world[i] in (world_idx, -1)]
            searched = (
                target_path
                if target_path == expanded_target_path
                else f"{target_path!r} (also tried {expanded_target_path!r})"
            )
            raise ValueError(
                f"CableAttachmentCfg.target_prim_path {searched} did not match any body in world {world_idx}."
                f" Available body labels in this world: {available_in_world}."
            )

        target_xform = wp.transform(attachment.target_local_pos, attachment.target_local_quat)
        if static_target_xform is not None:
            # Weld to the world at the static target's pose (parent = world body -1).
            target_body_idx = -1
            target_xform = wp.transform_multiply(static_target_xform, target_xform)
        cable_xform = wp.transform(attachment.cable_local_pos, attachment.cable_local_quat)

        # Keep the articulated cable segment as the loop-joint child. Targets may be
        # static scene bodies (or world, -1) with no articulation, while the cable
        # segment is already reachable through the cable root articulation.
        suffix = attachment.label_suffix or f"attachment_seg{anchor_idx}"
        joint_id = builder.add_joint_fixed(
            parent=target_body_idx,
            child=cable_body_idx,
            parent_xform=target_xform,
            child_xform=cable_xform,
            label=f"{entry.prim_path}/{suffix}_w{world_idx}",
            collision_filter_parent=True,
            enabled=bool(attachment.enabled),
        )
        joint_id = int(joint_id)
        joint_ids.append(joint_id)
        if attachment.add_to_articulation:
            joint_ids_by_cable.setdefault(int(cable_idx), []).append(joint_id)

    for cable_idx, cable_joint_ids in joint_ids_by_cable.items():
        if not cable_joint_ids:
            continue
        entry = SimulationManager._cable_registry[cable_idx]
        expanded_prim_path = entry.prim_path.replace("env_.*", f"env_{world_idx}")
        builder.add_articulation(cable_joint_ids, label=f"{expanded_prim_path}/attachment_articulation")

    return joint_ids


def install_cable_builder_hooks() -> None:
    """Reset the cable registry and install the per-world cable + attachment hooks."""
    SimulationManager._cable_registry = []
    SimulationManager._pending_cable_attachments = []
    SimulationManager._pending_cable_sdf_captures = []
    if not hasattr(SimulationManager, "_per_world_builder_hooks"):
        SimulationManager._per_world_builder_hooks = []
    SimulationManager._per_world_builder_hooks = [
        hook
        for hook in SimulationManager._per_world_builder_hooks
        if hook
        not in (
            add_registered_cables_to_builder,
            add_cable_sdf_captures_to_builder,
            apply_cable_attachments_to_builder,
        )
    ]
    SimulationManager._per_world_builder_hooks.append(add_registered_cables_to_builder)
    SimulationManager._per_world_builder_hooks.append(add_cable_sdf_captures_to_builder)
    SimulationManager._per_world_builder_hooks.append(apply_cable_attachments_to_builder)
    SimulationManager.register_pre_render_callback("cable_curve_sync", sync_registered_cable_curves_to_usd)


def sync_registered_cable_curves_to_usd() -> None:
    """Update registered cable ``UsdGeomBasisCurves.points`` from Newton body poses.

    Cable segment bodies are created by builder hooks and intentionally do not
    have authored USD prims. Kit renders the authored cable curve USD, so the
    curve control points must be updated explicitly from the live Newton rod
    segment body transforms before each render.
    """
    if getattr(SimulationManager, "_clone_physics_only", False):
        return
    state = SimulationManager._state_0
    if state is None or state.body_q is None:
        return
    registry = getattr(SimulationManager, "_cable_registry", None)
    if not registry:
        return

    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    if stage is None:
        return

    try:
        from pxr import Gf, UsdGeom, Vt  # noqa: PLC0415
    except Exception:
        return

    try:
        body_q = state.body_q.numpy()
    except Exception:
        return

    xform_cache = UsdGeom.XformCache()
    for entry in registry:
        curve_template_path = entry.curve_prim_path or f"{entry.prim_path}/geometry/mesh"
        point_count = len(entry.edges) + 1
        if point_count < 2:
            continue
        for inst_idx, body_offset in enumerate(entry.body_offsets):
            if int(body_offset) + len(entry.edges) > body_q.shape[0]:
                continue
            resolved = re.sub(r"(?<=[Ee]nv_)\.\*", str(inst_idx), curve_template_path)
            resolved = re.sub(r"\.\*", str(inst_idx), resolved)
            curve_prim = stage.GetPrimAtPath(resolved)
            if not curve_prim or not curve_prim.IsValid():
                logger.debug("[cable_curve_sync] curve prim not found at %s", resolved)
                continue

            world_to_local = xform_cache.GetLocalToWorldTransform(curve_prim).GetInverse()
            local_points = []
            for point_idx in range(len(entry.edges)):
                pose = body_q[int(body_offset) + point_idx]
                point_world = Gf.Vec3d(float(pose[0]), float(pose[1]), float(pose[2]))
                local_points.append(world_to_local.Transform(point_world))

            tail_pose = body_q[int(body_offset) + len(entry.edges) - 1]
            tail_offset = _rotate_vector_by_quat_np(
                np.asarray(tail_pose[3:7], dtype=np.float64),
                np.asarray((0.0, 0.0, float(entry.last_edge_length)), dtype=np.float64),
            )
            tail_world_np = np.asarray(tail_pose[:3], dtype=np.float64) + tail_offset
            tail_world = Gf.Vec3d(float(tail_world_np[0]), float(tail_world_np[1]), float(tail_world_np[2]))
            local_points.append(world_to_local.Transform(tail_world))

            curve = UsdGeom.BasisCurves(curve_prim)
            curve.GetPointsAttr().Set(
                Vt.Vec3fArray([Gf.Vec3f(float(point[0]), float(point[1]), float(point[2])) for point in local_points])
            )
            vertex_counts_attr = curve.GetCurveVertexCountsAttr()
            vertex_counts = vertex_counts_attr.Get() if vertex_counts_attr else None
            if vertex_counts is None or sum(int(count) for count in vertex_counts) != point_count:
                vertex_counts_attr.Set(Vt.IntArray([point_count]))


def _active_solver_cfg_supports_cables() -> bool:
    """Return whether the active Newton solver cfg contains a VBD cable-owning solver."""
    from isaaclab_newton.physics import CoupledSolverCfg

    from isaaclab.physics import PhysicsManager

    from isaaclab_contrib.deformable.newton_manager_cfg import (
        CoupledFeatherstoneVBDSolverCfg,
        CoupledMJWarpVBDSolverCfg,
        VBDSolverCfg,
    )

    physics_cfg = PhysicsManager._cfg
    solver_cfg = getattr(physics_cfg, "solver_cfg", None)
    if isinstance(solver_cfg, (VBDSolverCfg, CoupledMJWarpVBDSolverCfg, CoupledFeatherstoneVBDSolverCfg)):
        return True
    if isinstance(solver_cfg, CoupledSolverCfg):
        return any(isinstance(getattr(entry_cfg, "solver_cfg", None), VBDSolverCfg) for entry_cfg in solver_cfg.entries)
    return False


class CableObject(Articulation):
    """Cable / 1D-rod asset (Newton backend)."""

    cfg: CableObjectCfg

    def __init__(self, cfg: CableObjectCfg):
        """Initialize the cable object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

        self._registry_entry = self._register_cable()

        # Look up by identity via ``list.index`` rather than ``len(registry) - 1``
        # to stay robust if the base init ever mutates the registry concurrently.
        cable_idx = SimulationManager._cable_registry.index(self._registry_entry)
        for attachment in self.cfg.attachments:
            if attachment.dormant and not attachment.enabled:
                raise ValueError(
                    f"CableAttachmentCfg(label_suffix={attachment.label_suffix!r}) sets dormant=True with"
                    " enabled=False. Dormant latches must be built enabled (a joint enabled only at runtime"
                    " exerts no force); dormancy comes from zeroing the penalty gains at initialization."
                )
            SimulationManager._pending_cable_attachments.append((cable_idx, attachment))
        for capture in self.cfg.sdf_captures:
            SimulationManager._pending_cable_sdf_captures.append((cable_idx, capture))

    def _initialize_impl(self):
        super()._initialize_impl()
        # Assets initialize before the Newton solver is built; the registered callback
        # fires once the solver (and its AVBD penalty arrays) exist, before graph capture.
        SimulationManager.register_post_solver_init_callback(
            f"cable_dormant_attachments_{self.cfg.prim_path}", self._make_dormant_attachments_dormant
        )

    def _make_dormant_attachments_dormant(self) -> None:
        """Zero the AVBD joint-penalty gains of attachments flagged ``dormant``.

        Dormant attachments are built enabled so the solver allocates their
        build-time joint structures, with whatever default penalty gains the
        solver seeds. This zeroes ``joint_penalty_k``/``k_min``/``k_max``/``kd``
        for their constraint slots before the first physics step, writing in
        place into the existing warp arrays (CUDA-graph safe: captured kernels
        re-read them every step). Runtime code re-engages a latch by writing the
        gains back (see the waterhose scripted state machine for an example).

        Raises:
            RuntimeError: If no AVBD-style solver exposing the penalty arrays is
                active, or a dormant attachment's joint label does not resolve to
                exactly one joint per environment.
        """
        dormant = [attachment for attachment in self.cfg.attachments if attachment.dormant]
        if not dormant:
            return

        solver = SimulationManager._solver
        if solver is None:
            raise RuntimeError("Cannot make cable attachments dormant: the Newton solver is not initialized yet.")

        # Collect (sub-solver, joint labels) pairs that expose the AVBD joint-penalty
        # path: every entry of a coupled solver, or the flat solver itself.
        penalty_attrs = ("joint_penalty_k", "joint_penalty_k_min", "joint_penalty_k_max", "joint_penalty_kd")
        pairs = []
        entry_names = getattr(solver, "_entries", None)
        if entry_names:
            for name in entry_names:
                sub_solver = solver.solver(name)
                if all(hasattr(sub_solver, attr) for attr in penalty_attrs):
                    labels = [str(label) for label in solver.view(name).joint_label]
                    pairs.append((sub_solver, labels))
        elif all(hasattr(solver, attr) for attr in penalty_attrs):
            labels = [str(label) for label in SimulationManager.get_model().joint_label]
            pairs.append((solver, labels))
        if not pairs:
            raise RuntimeError(
                "CableAttachmentCfg.dormant requires an AVBD-style VBD solver exposing the joint_penalty_*"
                " arrays; the active solver has none."
            )

        num_worlds = len(self._registry_entry.body_offsets)
        tokens = [
            f"{attachment.label_suffix or f'attachment_seg{attachment.cable_anchor}'}_w{world_idx}"
            for attachment in dormant
            for world_idx in range(num_worlds)
        ]
        matched: dict[str, int] = dict.fromkeys(tokens, 0)
        for sub_solver, labels in pairs:
            joints = [j for j, label in enumerate(labels) for token in tokens if label.endswith(token)]
            for j, label in ((j, labels[j]) for j in joints):
                for token in tokens:
                    if label.endswith(token):
                        matched[token] += 1
            if not joints:
                continue
            constraint_start = sub_solver.joint_constraint_start.numpy()
            for attr in penalty_attrs:
                arr = getattr(sub_solver, attr)
                host = arr.numpy()
                for j in joints:
                    slot_begin = int(constraint_start[j])
                    slot_end = int(constraint_start[j + 1]) if j + 1 < len(constraint_start) else host.shape[0]
                    host[slot_begin:slot_end] = 0.0
                arr.assign(host)

        unresolved = {token: count for token, count in matched.items() if count != 1}
        if unresolved:
            raise RuntimeError(
                f"Dormant cable attachments on '{self.cfg.prim_path}' did not resolve to exactly one joint"
                f" each: {unresolved}."
            )

    def _register_cable(self) -> CableRegistryEntry:
        """Read cable geometry + material from the spawned USD prim and append to the registry.

        Returns:
            The registry entry (also appended to :attr:`SimulationManager._cable_registry`).

        Raises:
            ValueError: If the template prim has no ``UsdGeomBasisCurves``
                descendant, or the curve is missing its ``widths`` attribute.
            NotImplementedError: If more than one ``UsdGeomBasisCurves``
                descendant is found under the template prim — multi-curve
                cables under a single :class:`CableObject` are not supported.
            RuntimeError: If the template prim cannot be located, or the active
                Newton solver is not a VBD variant (only :class:`VBDSolverCfg`
                and its coupled variants register the cable builder hooks; no
                other Newton solver steps :attr:`newton.JointType.CABLE`).

        Note:
            ``pxr`` imports are deferred to this method (not module level) so
            that ``resolve_task_config`` can import the env-cfg module before
            Kit starts without polluting the ``pxr`` module cache.
        """
        # ``pxr`` import is deferred so ``resolve_task_config`` can import the
        # env-cfg module before Kit starts without polluting ``pxr`` caches.
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

        if not hasattr(SimulationManager, "_cable_registry") and _active_solver_cfg_supports_cables():
            install_cable_builder_hooks()

        if not hasattr(SimulationManager, "_cable_registry"):
            raise RuntimeError(
                "CableObject can only be simulated under the Newton VBD solver"
                " (`VBDSolverCfg`, a generic `CoupledSolverCfg` with a `VBDSolverCfg` entry,"
                " or one of the legacy coupled variants:"
                " `CoupledMJWarpVBDSolverCfg`, `CoupledFeatherstoneVBDSolverCfg`)."
                " The cable registry is installed by the VBD manager's `initialize()`"
                " hook via `install_cable_builder_hooks()`, and `JointType.CABLE`"
                " is not stepped by any other Newton solver. Switch the solver cfg"
                " or remove the CableObject from the scene."
            )

        if self.cfg.spawn is None:
            raise ValueError(
                f"CableObjectCfg(prim_path='{self.cfg.prim_path}') has no `spawn` configuration."
                " CableObject requires a `CableCfg` (or compatible USD-loading cfg) to register"
                " cable geometry; pass one via `CableObjectCfg.spawn`."
            )

        lookup_path = self.cfg.spawn.spawn_path if self.cfg.spawn.spawn_path is not None else self.cfg.prim_path
        template_prim = sim_utils.find_first_matching_prim(lookup_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find cable template prim for expression: '{lookup_path}'.")
        template_prim_path = template_prim.GetPrimPath()

        stage = template_prim.GetStage()
        curve_prims = [
            descendant for descendant in Usd.PrimRange(template_prim) if descendant.GetTypeName() == "BasisCurves"
        ]
        if not curve_prims:
            raise ValueError(f"No UsdGeomBasisCurves prim found under '{template_prim_path}'.")
        if len(curve_prims) > 1:
            paths = ", ".join(str(p.GetPrimPath()) for p in curve_prims)
            raise NotImplementedError(
                f"Found {len(curve_prims)} BasisCurves prims under '{template_prim_path}' ({paths}); "
                "multi-curve cables under a single CableObject are not supported yet."
            )
        curve_prim = curve_prims[0]
        curves = UsdGeom.BasisCurves(curve_prim)

        # Bake the curve prim's xform into node positions so the replicate hook
        # only needs to apply the env transform.
        xform_cache = UsdGeom.XformCache()
        curve_to_parent_frame = (
            xform_cache.GetLocalToWorldTransform(curve_prim)
            * xform_cache.GetLocalToWorldTransform(template_prim.GetParent()).GetInverse()
        )
        raw_points = curves.GetPointsAttr().Get()
        node_positions: list[wp.vec3] = []
        for p in raw_points:
            q = curve_to_parent_frame.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
            node_positions.append(wp.vec3(float(q[0]), float(q[1]), float(q[2])))

        raw_widths = curves.GetWidthsAttr().Get()
        if raw_widths is None or len(raw_widths) == 0:
            raise ValueError(f"UsdGeomBasisCurves at '{curve_prim.GetPrimPath()}' is missing the `widths` attribute.")
        widths_list = [float(w) for w in raw_widths]
        if max(widths_list) - min(widths_list) > 1e-9:
            raise ValueError(
                f"UsdGeomBasisCurves at '{curve_prim.GetPrimPath()}' has non-uniform `widths`"
                f" (min={min(widths_list)}, max={max(widths_list)}); tapered cables are not supported."
                " Author a constant width across all control points."
            )
        radius = widths_list[0] / 2.0

        connections_attr = curve_prim.GetAttribute("connections")
        if not connections_attr.IsValid() or connections_attr.Get() is None:
            raise ValueError(
                f"UsdGeomBasisCurves at '{curve_prim.GetPrimPath()}' is missing the `connections`"
                " attribute (expected `int2[]` listing each edge as a pair of control-point indices)."
                " Author this attribute on the curve prim — `spawn_cable` writes it automatically;"
                " user-imported curve USDs must add it explicitly."
            )
        edges = [(int(e[0]), int(e[1])) for e in connections_attr.Get()]

        resample_segment_length = getattr(self.cfg, "resample_segment_length", None)
        if resample_segment_length:
            node_positions, edges = _resample_open_polyline(node_positions, edges, float(resample_segment_length))

        # Material binding requires ``UsdPhysics.CollisionAPI`` on the curve;
        # without it the spawner's bind silently no-ops.
        material_targets = (
            UsdShade.MaterialBindingAPI(curve_prim).GetDirectBindingRel("physics").GetTargets()
            if curve_prim.HasAPI(UsdShade.MaterialBindingAPI)
            else []
        )
        material_prim = None
        for mat_path in material_targets:
            mat_prim = stage.GetPrimAtPath(mat_path)
            if mat_prim.GetAttribute("newton:density").IsValid():
                material_prim = mat_prim
                break
        if material_prim is None:
            has_collision_api = curve_prim.HasAPI(UsdPhysics.CollisionAPI)
            hint = (
                ""
                if has_collision_api
                else (
                    " Hint: the curve has no `UsdPhysics.CollisionAPI`, which `bind_physics_material`"
                    " requires; set `CableCfg.collision_props = sim_utils.CollisionPropertiesCfg()` so"
                    " `spawn_cable` applies the API (cables are currently Newton-only and the API has"
                    " no PhysX runtime effect)."
                )
            )
            raise ValueError(
                f"Could not find a Newton cable physics material bound to '{curve_prim.GetPrimPath()}'." + hint
            )

        def _get_material_attr(name: str, default):
            attr = material_prim.GetAttribute(name)
            return attr.Get() if attr.IsValid() else default

        stretch_stiffness = _get_material_attr("newton:stretchStiffness", CableRegistryEntry.stretch_stiffness)
        bend_stiffness = _get_material_attr("newton:bendStiffness", CableRegistryEntry.bend_stiffness)
        stretch_damping = _get_material_attr("newton:stretchDamping", CableRegistryEntry.stretch_damping)
        bend_damping = _get_material_attr("newton:bendDamping", CableRegistryEntry.bend_damping)
        density = _get_material_attr("newton:density", CableRegistryEntry.density)

        # init_pos/init_rot stay identity: the template xform is already baked
        # into ``node_positions``; the replicate hook only adds the env transform.
        entry = CableRegistryEntry(
            prim_path=self.cfg.prim_path,
            curve_prim_path=str(curve_prim.GetPrimPath()),
            node_positions=node_positions,
            edges=edges,
            radius=radius,
            stretch_stiffness=float(stretch_stiffness),
            bend_stiffness=float(bend_stiffness),
            stretch_damping=float(stretch_damping),
            bend_damping=float(bend_damping),
            density=float(density),
        )
        SimulationManager._cable_registry.append(entry)
        return entry
