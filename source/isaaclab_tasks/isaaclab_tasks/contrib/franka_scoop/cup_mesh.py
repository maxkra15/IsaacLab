# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clean, watertight, low-poly cup collision mesh for the scoop end-effector.

The cup is a surface of revolution (open-top mug, no handle) generated as a single closed,
consistently outward-wound triangle mesh, so an MPM/SDF collider built from it has an unambiguous
inside/outside. Crucially the walls and bottom must be **at least ~1 grid voxel thick**: MPM resolves
mesh colliders at grid-node resolution (a thin sub-voxel wall has no solid interior on the grid, so
particles tunnel through it). Keep ``wall_thickness`` and ``bottom_thickness`` >= ~1.5 * voxel_size.

Local frame: ``z=0`` is the outer base (table-facing); the cavity floor is at ``z=bottom_thickness``;
the rim is at ``z=bottom_thickness+cavity_depth``. The cup is centred on the z axis.
"""

from __future__ import annotations

import math

import numpy as np


def make_cup_collision_mesh(
    *,
    inner_bottom_radius: float,
    inner_top_radius: float,
    wall_thickness: float,
    cavity_depth: float,
    bottom_thickness: float,
    num_segments: int = 32,
    validate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a watertight, outward-wound cup collision mesh (surface of revolution).

    Args:
        inner_bottom_radius: Cavity radius at its floor [m].
        inner_top_radius: Cavity radius at the rim [m] (>= ``inner_bottom_radius`` for a flared cup).
        wall_thickness: Side-wall thickness [m]; keep >= ~1.5 * MPM voxel_size to avoid tunnelling.
        cavity_depth: Inner cavity height from floor to rim [m].
        bottom_thickness: Base thickness below the cavity floor [m]; keep >= ~1.5 * voxel_size.
        num_segments: Radial tessellation; 32 is smooth and low-poly.
        validate: If True, assert the mesh is closed (every edge shared by exactly two triangles).

    Returns:
        ``(vertices, indices)`` with vertices ``(V, 3)`` float32 and triangle indices ``(3F,)`` int32.
    """
    n = int(num_segments)
    total_h = float(bottom_thickness) + float(cavity_depth)
    outer_bottom_r = float(inner_bottom_radius) + float(wall_thickness)
    outer_top_r = float(inner_top_radius) + float(wall_thickness)

    theta = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    def ring(radius: float, z: float) -> np.ndarray:
        return np.column_stack([radius * cos_t, radius * sin_t, np.full(n, z)])

    ob = ring(outer_bottom_r, 0.0)                       # [0*n : 1*n) outer base ring
    ot = ring(outer_top_r, total_h)                      # [1*n : 2*n) outer rim ring
    it = ring(float(inner_top_radius), total_h)          # [2*n : 3*n) inner rim ring
    if_ = ring(float(inner_bottom_radius), float(bottom_thickness))  # [3*n : 4*n) cavity-floor ring
    obc = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)              # outer base centre
    ifc = np.array([[0.0, 0.0, float(bottom_thickness)]], dtype=np.float32)  # cavity-floor centre
    vertices = np.vstack([ob, ot, it, if_, obc, ifc]).astype(np.float32)
    OBC, IFC = 4 * n, 4 * n + 1

    faces: list[int] = []
    for i in range(n):
        j = (i + 1) % n
        ob_i, ob_j = i, j
        ot_i, ot_j = n + i, n + j
        it_i, it_j = 2 * n + i, 2 * n + j
        if_i, if_j = 3 * n + i, 3 * n + j
        # Outer side wall (normal +radial, outward).
        faces += [ob_i, ob_j, ot_i, ot_i, ob_j, ot_j]
        # Top rim annulus (normal +z).
        faces += [it_i, ot_i, it_j, it_j, ot_i, ot_j]
        # Inner cavity wall (normal -radial, toward the cavity).
        faces += [if_i, it_i, if_j, if_j, it_i, it_j]
        # Cavity floor disk (normal +z, into the cavity).
        faces += [IFC, if_i, if_j]
        # Outer base disk (normal -z).
        faces += [OBC, ob_j, ob_i]
    indices = np.asarray(faces, dtype=np.int32)

    if validate:
        edges: dict[tuple[int, int], int] = {}
        tris = indices.reshape(-1, 3)
        for tri in tris:
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                key = (int(min(a, b)), int(max(a, b)))
                edges[key] = edges.get(key, 0) + 1
        open_edges = [e for e, count in edges.items() if count != 2]
        if open_edges:
            raise RuntimeError(f"Cup mesh is not watertight: {len(open_edges)} edges not shared by 2 triangles.")

    return vertices, indices
