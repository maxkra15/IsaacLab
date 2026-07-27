# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the watertight waterhose fridge collision proxies.

The source fridge contains hundreds of convex-decomposition fragments. Runtime use of every
fragment is unnecessarily expensive, while concatenating and deleting triangles produces an open,
non-manifold mesh with undefined inside/outside queries. This tool instead:

1. reconstructs each active authored fragment as its intended convex hull;
2. removes exact duplicate hulls;
3. computes a topology-preserving solid union;
4. subtracts solver-specific closed insertion clearances;
5. simplifies the result without changing its topology; and
6. validates both topological and Newton-quantized geometric manifoldness.

Run from the repository root with the Isaac Lab environment:

.. code-block:: bash

    uv run --with manifold3d==3.5.2 python \
      scripts/environments/waterhose/create_fridge_collision_proxy.py
"""

from __future__ import annotations

import argparse
import operator
from functools import reduce
from pathlib import Path

import manifold3d
import numpy as np
import trimesh

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ASSETS_DIR = _REPO_ROOT / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "contrib" / "waterhose" / "assets"
_DEFAULT_SOURCE = _ASSETS_DIR / "fridge" / "fridge.usda"
_DEFAULT_ROBOT_OUTPUT = _ASSETS_DIR / "fridge_robot_manifold_collision.usda"
_DEFAULT_CABLE_OUTPUT = _ASSETS_DIR / "fridge_cable_manifold_collision.usda"

_COLLISION_SCOPE = "/root/Cable008/Collisions/"
_CORRIDOR_START = np.asarray((-0.259345, 0.36352012, -0.2647031), dtype=np.float64)
_CORRIDOR_END = np.asarray((-0.259345, 0.33718455, -0.19234677), dtype=np.float64)
_CABLE_CORRIDOR_RADIUS = 0.015
# The articulated gripper reaches roughly 53 mm from the connector axis. Leave another 12 mm for
# its 10 mm broad-phase gap and compliant contact deflection. The cable uses the smaller corridor
# above so housing contact is retained outside the local plug-and-bent-hose insertion path.
_ROBOT_CORRIDOR_RADIUS = 0.065
_CORRIDOR_END_PADDING = 0.01
_CORRIDOR_SECTIONS = 48
_SIMPLIFY_TOLERANCE = 2.0e-4
_COINCIDENT_VERTEX_SEPARATION = 2.0e-6
_NEWTON_VERTEX_DECIMALS = 7


def _to_manifold(mesh: trimesh.Trimesh) -> manifold3d.Manifold:
    """Convert a validated triangle mesh to Manifold's float32 representation."""

    manifold_mesh = manifold3d.Mesh(
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.uint32),
    )
    result = manifold3d.Manifold(manifold_mesh)
    if result.status() != manifold3d.Error.NoError:
        raise RuntimeError(f"Manifold rejected an authored convex hull: {result.status()}.")
    return result


def _mesh_points_in_root_frame(
    mesh: UsdGeom.Mesh,
    xform_cache: UsdGeom.XformCache,
    world_to_root: Gf.Matrix4d,
) -> np.ndarray:
    """Return a USD mesh's points expressed in the fridge root frame."""

    local_to_world = xform_cache.GetLocalToWorldTransform(mesh.GetPrim())
    points = []
    for point in mesh.GetPointsAttr().Get():
        world_point = local_to_world.Transform(Gf.Vec3d(*point))
        root_point = world_to_root.Transform(world_point)
        points.append(tuple(root_point))
    return np.asarray(points, dtype=np.float64)


def _load_unique_source_hulls(source_path: Path) -> tuple[list[trimesh.Trimesh], int]:
    """Load and deduplicate the active authored fridge collision hulls."""

    stage = Usd.Stage.Open(str(source_path))
    if stage is None:
        raise RuntimeError(f"Could not open fridge source USD: {source_path}")
    root_prim = stage.GetPrimAtPath("/root")
    if not root_prim.IsValid():
        raise RuntimeError(f"Fridge source USD has no /root prim: {source_path}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_to_root = xform_cache.GetLocalToWorldTransform(root_prim).GetInverse()
    hulls: list[trimesh.Trimesh] = []
    signatures: set[tuple[tuple[float, float, float], ...]] = set()
    source_count = 0

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Mesh) or not path.startswith(_COLLISION_SCOPE):
            continue
        source_count += 1
        points = _mesh_points_in_root_frame(UsdGeom.Mesh(prim), xform_cache, world_to_root)
        if len(points) < 4:
            raise RuntimeError(f"Collider {path} has fewer than four points.")

        # These prims are authored with physics:approximation = "convexHull". Reconstructing that
        # intended geometry also repairs the few source triangle soups that are themselves open.
        hull = trimesh.convex.convex_hull(points)
        if not hull.is_volume:
            raise RuntimeError(f"Collider {path} did not produce a closed, outward convex hull.")

        # The source contains repeated fragment geometry at identical transforms. Remove it before
        # Boolean union so duplicates cannot create coincident surfaces.
        signature = tuple(sorted(tuple(point) for point in np.round(hull.vertices, decimals=7)))
        if signature in signatures:
            continue
        signatures.add(signature)
        hulls.append(hull)

    if not hulls:
        raise RuntimeError(f"No active collider meshes found under {_COLLISION_SCOPE!r}.")
    return hulls, source_count


def _cylinder_between(start: np.ndarray, end: np.ndarray, radius: float) -> trimesh.Trimesh:
    """Create a closed cylinder between two points."""

    axis = end - start
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        raise RuntimeError("Collision-proxy cylinder endpoints must be distinct.")
    transform = trimesh.geometry.align_vectors((0.0, 0.0, 1.0), axis / length)
    transform[:3, 3] = (start + end) * 0.5
    return trimesh.creation.cylinder(
        radius=radius,
        height=length,
        sections=_CORRIDOR_SECTIONS,
        transform=transform,
    )


def _separate_kissing_vertices(mesh: trimesh.Trimesh) -> int:
    """Separate topologically distinct vertices that occupy one quantized point.

    A solid Boolean may retain two closed sheets that only kiss along an edge. Manifold treats those
    sheets as topologically valid, but Newton intentionally canonicalizes float32 positions to 100 nm
    when checking a collision mesh. Move each sheet by at most one micron along its own outward vertex
    normal so the exported surface is also geometrically manifold at Newton's tolerance.
    """

    quantized = np.round(np.asarray(mesh.vertices, dtype=np.float32), decimals=_NEWTON_VERTEX_DECIMALS)
    _, canonical_ids, counts = np.unique(quantized, axis=0, return_inverse=True, return_counts=True)
    duplicate_groups = np.flatnonzero(counts > 1)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    for group_id in duplicate_groups:
        vertex_ids = np.flatnonzero(canonical_ids == group_id)
        midpoint = 0.5 * (len(vertex_ids) - 1)
        for rank, vertex_id in enumerate(vertex_ids):
            normal = normals[vertex_id]
            norm = float(np.linalg.norm(normal))
            if norm <= 1.0e-12:
                normal = np.asarray((1.0, 0.0, 0.0))
            else:
                normal = normal / norm
            offset = (rank - midpoint) * _COINCIDENT_VERTEX_SEPARATION
            mesh.vertices[vertex_id] += offset * normal
    return len(duplicate_groups)


def _build_housing(
    hulls: list[trimesh.Trimesh],
    corridor_radius: float,
    simplify_tolerance: float,
) -> tuple[trimesh.Trimesh, int]:
    """Union the source hulls and subtract one closed insertion clearance."""

    housing = reduce(operator.add, (_to_manifold(hull) for hull in hulls))
    corridor_axis = _CORRIDOR_END - _CORRIDOR_START
    corridor_length = float(np.linalg.norm(corridor_axis))
    corridor_padding = _CORRIDOR_END_PADDING * corridor_axis / corridor_length
    corridor = _cylinder_between(
        _CORRIDOR_START - corridor_padding,
        _CORRIDOR_END + corridor_padding,
        corridor_radius,
    )

    housing = housing - _to_manifold(corridor)
    housing = housing.simplify(float(simplify_tolerance))
    if housing.status() != manifold3d.Error.NoError:
        raise RuntimeError(f"Housing Boolean failed: {housing.status()}.")

    result = housing.to_mesh()
    mesh = trimesh.Trimesh(
        vertices=np.asarray(result.vert_properties[:, :3], dtype=np.float64),
        faces=np.asarray(result.tri_verts, dtype=np.int64),
        process=False,
    )
    separated_vertex_groups = _separate_kissing_vertices(mesh)
    edge_counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    quantized = np.round(np.asarray(mesh.vertices, dtype=np.float32), decimals=_NEWTON_VERTEX_DECIMALS)
    _, canonical_ids = np.unique(quantized, axis=0, return_inverse=True)
    canonical_faces = canonical_ids[np.asarray(mesh.faces)]
    geometric_edges = np.concatenate(
        (
            np.sort(canonical_faces[:, (0, 1)], axis=1),
            np.sort(canonical_faces[:, (1, 2)], axis=1),
            np.sort(canonical_faces[:, (0, 2)], axis=1),
        ),
        axis=0,
    )
    _, geometric_edge_counts = np.unique(geometric_edges, axis=0, return_counts=True)
    if (
        not mesh.is_watertight
        or not mesh.is_winding_consistent
        or not mesh.is_volume
        or not np.all(edge_counts == 2)
        or not np.all(geometric_edge_counts == 2)
    ):
        raise RuntimeError("Generated housing is not a closed, consistently wound manifold volume.")
    return mesh, separated_vertex_groups


def _write_housing(
    output_path: Path,
    housing: trimesh.Trimesh,
    *,
    source_path: Path,
    source_count: int,
    unique_count: int,
    corridor_radius: float,
    simplify_tolerance: float,
    separated_vertex_groups: int,
    author_mjwarp_solref: bool,
) -> None:
    """Write one solver-specific collision-only USD."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    root = UsdGeom.Xform.Define(stage, "/FridgeCollision")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/FridgeCollision/Housing")

    vertices = np.asarray(housing.vertices, dtype=np.float32)
    faces = np.asarray(housing.faces, dtype=np.int32)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
    mesh.CreateExtentAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(*(float(value) for value in np.min(vertices, axis=0))),
                Gf.Vec3f(*(float(value) for value in np.max(vertices, axis=0))),
            ]
        )
    )
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
    mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)

    collision_api = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    collision_api.CreateCollisionEnabledAttr(True)
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision_api.CreateApproximationAttr("none")
    if author_mjwarp_solref:
        mesh.GetPrim().CreateAttribute("mjc:solref", Sdf.ValueTypeNames.DoubleArray).Set(Vt.DoubleArray([0.005, 1.0]))

    attributes = {
        "waterhose:clearanceCenter": (Gf.Vec3f, (_CORRIDOR_START + _CORRIDOR_END) * 0.5),
        "waterhose:corridorEnd": (Gf.Vec3f, _CORRIDOR_END),
        "waterhose:corridorRadius": (float, corridor_radius),
        "waterhose:corridorStart": (Gf.Vec3f, _CORRIDOR_START),
        "waterhose:coincidentVertexSeparation": (float, _COINCIDENT_VERTEX_SEPARATION),
        "waterhose:simplifyTolerance": (float, simplify_tolerance),
    }
    for name, (value_type, value) in attributes.items():
        sdf_type = Sdf.ValueTypeNames.Float3 if value_type is Gf.Vec3f else Sdf.ValueTypeNames.Float
        authored_value = Gf.Vec3f(*(float(component) for component in value)) if value_type is Gf.Vec3f else value
        mesh.GetPrim().CreateAttribute(name, sdf_type, custom=True).Set(authored_value)
    mesh.GetPrim().CreateAttribute("waterhose:source", Sdf.ValueTypeNames.String, custom=True).Set(
        f"Manifold union of active convex hulls from {source_path.name}"
    )
    mesh.GetPrim().CreateAttribute("waterhose:sourceColliderCount", Sdf.ValueTypeNames.Int, custom=True).Set(
        source_count
    )
    mesh.GetPrim().CreateAttribute("waterhose:separatedVertexGroups", Sdf.ValueTypeNames.Int, custom=True).Set(
        separated_vertex_groups
    )
    mesh.GetPrim().CreateAttribute("waterhose:uniqueColliderCount", Sdf.ValueTypeNames.Int, custom=True).Set(
        unique_count
    )
    stage.GetRootLayer().Save()


def main() -> None:
    """Generate and validate the task-local fridge collision proxies."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=_DEFAULT_SOURCE, help="Source fridge USD.")
    parser.add_argument("--robot-output", type=Path, default=_DEFAULT_ROBOT_OUTPUT, help="Robot proxy output.")
    parser.add_argument("--cable-output", type=Path, default=_DEFAULT_CABLE_OUTPUT, help="Cable proxy output.")
    parser.add_argument(
        "--robot-corridor-radius",
        type=float,
        default=_ROBOT_CORRIDOR_RADIUS,
        help="Robot insertion-approach clearance radius in meters.",
    )
    parser.add_argument(
        "--cable-corridor-radius",
        type=float,
        default=_CABLE_CORRIDOR_RADIUS,
        help="Cable insertion clearance radius in meters.",
    )
    parser.add_argument(
        "--simplify-tolerance",
        type=float,
        default=_SIMPLIFY_TOLERANCE,
        help="Maximum topology-preserving surface displacement in meters.",
    )
    args = parser.parse_args()

    source_path = args.source.resolve()
    hulls, source_count = _load_unique_source_hulls(source_path)
    outputs = (
        (args.robot_output.resolve(), args.robot_corridor_radius, True),
        (args.cable_output.resolve(), args.cable_corridor_radius, False),
    )
    for output_path, corridor_radius, author_mjwarp_solref in outputs:
        housing, separated_vertex_groups = _build_housing(
            hulls,
            corridor_radius,
            args.simplify_tolerance,
        )
        _write_housing(
            output_path,
            housing,
            source_path=source_path,
            source_count=source_count,
            unique_count=len(hulls),
            corridor_radius=corridor_radius,
            simplify_tolerance=args.simplify_tolerance,
            separated_vertex_groups=separated_vertex_groups,
            author_mjwarp_solref=author_mjwarp_solref,
        )
        print(
            f"Wrote {output_path} from {source_count} active colliders "
            f"({len(hulls)} unique): {len(housing.vertices)} vertices, {len(housing.faces)} triangles."
        )


if __name__ == "__main__":
    main()
