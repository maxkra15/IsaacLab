# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate a simple hollow-cylinder (cup) "socket" collision mesh USD.

Replaces the complex multi-fragment ``socket_collision.usda`` with a single clean
tube the plug connector friction-fits into. Authored at the origin with the bore
axis along +Z and the open mouth at z=0; the bore extends to +Z and is closed at
the far end (z=depth), forming a closed manifold suitable for an SDF.

    ./isaaclab.sh -p scripts/environments/waterhose/generate_socket_cylinder.py

Geometry (defaults): bore radius 5.6mm (friction-fit the ~11mm-dia, r5.5 plug
shaft), wall 3mm (outer radius 8.6mm), depth 12mm.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics, Vt

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / "source/isaaclab_tasks/isaaclab_tasks/contrib/waterhose/assets/fridge/socket_cylinder.usda"


def build_cup(bore_r, outer_r, depth, seg, floor=0.002, lead_in=0.0015, lead_depth=0.0025):
    """Return (points, tris) for a watertight blind tube (cup) with a chamfered mouth.

    Open mouth at z=0 widened by ``lead_in`` and tapering to ``bore_r`` over ``lead_depth``
    (a lead-in chamfer so the plug self-centers and enters); straight bore to z=depth-floor;
    outer shell to z=depth. Each ring edge is shared by exactly two triangles (watertight).
    """
    bore_depth = max(depth - floor, 0.5 * depth)
    pts: list[tuple[float, float, float]] = []
    ang = [2.0 * math.pi * i / seg for i in range(seg)]

    def ring(r, z):
        base = len(pts)
        for a in ang:
            pts.append((r * math.cos(a), r * math.sin(a), z))
        return list(range(base, base + seg))

    out0 = ring(outer_r, 0.0)  # outer wall, mouth
    out1 = ring(outer_r, depth)  # outer wall, bottom
    inn0 = ring(bore_r + lead_in, 0.0)  # mouth opening (chamfer top, widened)
    innA = ring(bore_r, lead_depth)  # chamfer bottom = start of straight bore
    innB = ring(bore_r, bore_depth)  # bore wall, floor
    ext_c = len(pts)
    pts.append((0.0, 0.0, depth))  # exterior bottom center
    bore_c = len(pts)
    pts.append((0.0, 0.0, bore_depth))  # bore floor center

    tris: list[tuple[int, int, int]] = []
    for i in range(seg):
        j = (i + 1) % seg
        # outer wall (outward normals)
        tris.append((out0[i], out0[j], out1[j]))
        tris.append((out0[i], out1[j], out1[i]))
        # chamfer lead-in (inward) inn0 -> innA
        tris.append((inn0[i], innA[i], innA[j]))
        tris.append((inn0[i], innA[j], inn0[j]))
        # straight bore wall (inward) innA -> innB
        tris.append((innA[i], innB[i], innB[j]))
        tris.append((innA[i], innB[j], innA[j]))
        # mouth rim annulus z=0 (faces -Z toward the approaching plug): out0 -> inn0
        tris.append((out0[i], inn0[i], inn0[j]))
        tris.append((out0[i], inn0[j], out0[j]))
        # exterior bottom disk z=depth (faces +Z)
        tris.append((out1[i], ext_c, out1[j]))
        # bore floor disk z=bore_depth (faces -Z, into the cavity)
        tris.append((innB[i], innB[j], bore_c))
    return pts, tris


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bore_radius", type=float, default=0.0056)
    p.add_argument("--outer_radius", type=float, default=0.0086)
    p.add_argument("--depth", type=float, default=0.012)
    p.add_argument("--segments", type=int, default=48)
    p.add_argument("--out", type=str, default=str(OUT))
    a = p.parse_args()

    pts, tris = build_cup(a.bore_radius, a.outer_radius, a.depth, a.segments)
    counts = [3] * len(tris)
    flat = [i for t in tris for i in t]

    if Path(a.out).exists():
        Path(a.out).unlink()
    stage = Usd.Stage.CreateNew(a.out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Socket")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Socket/socket_mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray([tuple(map(float, q)) for q in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(flat))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    # Static collider: collision API + (mesh) approximation = none (use actual triangles).
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mca = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mca.CreateApproximationAttr(UsdPhysics.Tokens.none)
    stage.GetRootLayer().Save()
    print(
        f"wrote {a.out}  verts={len(pts)} tris={len(tris)} "
        f"bore_r={a.bore_radius} outer_r={a.outer_radius} depth={a.depth}"
    )


if __name__ == "__main__":
    main()
