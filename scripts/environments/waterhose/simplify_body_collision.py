# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Decimate the welded ``Cable008_BodyCollision`` mesh in ``fridge.usda`` to a low-poly collider.

This single welded mesh is the connector-housing collider for the robot (the MJWarp entry collides it
through Newton's pipeline; see ``WATERHOSE_FRIDGE_COLLISION``). Keeping it low-poly keeps the per-env
mesh-triangle candidate count small: deterministic contact matching hard-caps the global
``triangle_pairs`` buffer at ``2**20``, so the welded mesh's ~103k triangles would overflow it at scale.
This rewrites that mesh in place as a coarse vertex-clustered shell (a few hundred triangles) so the
contact batches to thousands of envs. The socket collider is untouched -- only this one body mesh changes.

Vertex clustering is used (no external decimation backend needed): snap every vertex to a coarse grid
cell, replace it with the cell centroid, and drop the faces that collapse. It is robust to the welded
union-of-hulls triangle soup, and the mesh is authored ``doubleSided`` so any flipped winding is moot.

    ./isaaclab.sh -p scripts/environments/waterhose/simplify_body_collision.py [--grid-res 48]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from pxr import Gf, Usd, UsdGeom, Vt

_DEFAULT_FRIDGE = (
    Path(__file__).resolve().parents[3]
    / "source/isaaclab_tasks/isaaclab_tasks/contrib/waterhose/assets/fridge/fridge.usda"
)
_BODY_MESH_PATH = "/root/Cable008/BodyCollision/Cable008_BodyCollision"


def cluster_decimate(vertices: np.ndarray, faces: np.ndarray, grid_res: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertex-cluster a triangle soup onto a ``grid_res**3`` lattice over its bounding box."""
    lo = vertices.min(axis=0)
    span = vertices.max(axis=0) - lo
    cell = np.where(span > 0.0, span / grid_res, 1.0)
    quant = np.clip(np.floor((vertices - lo) / cell).astype(np.int64), 0, grid_res - 1)
    key = (quant[:, 0] * grid_res + quant[:, 1]) * grid_res + quant[:, 2]
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)

    new_vertices = np.zeros((counts.shape[0], 3), dtype=np.float64)
    np.add.at(new_vertices, inverse, vertices)
    new_vertices /= counts[:, None]

    remapped = inverse[faces]
    non_degenerate = (
        (remapped[:, 0] != remapped[:, 1]) & (remapped[:, 1] != remapped[:, 2]) & (remapped[:, 0] != remapped[:, 2])
    )
    remapped = remapped[non_degenerate]
    _, unique_face_idx = np.unique(np.sort(remapped, axis=1), axis=0, return_index=True)
    return new_vertices, remapped[unique_face_idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Decimate the welded fridge-body collision mesh in place.")
    parser.add_argument("--fridge", type=Path, default=_DEFAULT_FRIDGE, help="Path to fridge.usda.")
    parser.add_argument("--mesh-path", type=str, default=_BODY_MESH_PATH, help="Prim path of the body mesh.")
    parser.add_argument(
        "--grid-res", type=int, default=48, help="Clustering grid resolution per axis (higher = finer)."
    )
    parser.add_argument("--dry-run", action="store_true", help="Report the result without writing the USD.")
    args = parser.parse_args()

    stage = Usd.Stage.Open(str(args.fridge))
    if stage is None:
        raise SystemExit(f"Could not open {args.fridge}")
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(args.mesh_path))
    if not mesh:
        raise SystemExit(f"No UsdGeom.Mesh at {args.mesh_path}")

    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if not np.all(counts == 3):
        raise SystemExit("Body mesh has non-triangle faces; triangulate before decimating.")
    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
    print(f"[simplify] in : {len(vertices)} verts, {len(faces)} faces, extent {vertices.min(0)} -> {vertices.max(0)}")

    new_vertices, new_faces = cluster_decimate(vertices, faces, args.grid_res)
    print(
        f"[simplify] out: {len(new_vertices)} verts, {len(new_faces)} faces "
        f"({len(faces) / max(len(new_faces), 1):.0f}x fewer), extent {new_vertices.min(0)} -> {new_vertices.max(0)}"
    )
    if args.dry_run:
        print("[simplify] dry-run: USD not modified.")
        return

    mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in new_vertices]))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(new_faces)))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([int(i) for i in new_faces.reshape(-1)]))
    mesh.GetExtentAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(*map(float, new_vertices.min(0))), Gf.Vec3f(*map(float, new_vertices.max(0)))])
    )
    stage.GetRootLayer().Save()
    print(f"[simplify] wrote {os.fspath(args.fridge)}")


if __name__ == "__main__":
    main()
