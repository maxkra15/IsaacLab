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

    outer_base = ring(outer_bottom_r, 0.0)  # [0*n : 1*n) outer base ring
    outer_rim = ring(outer_top_r, total_h)  # [1*n : 2*n) outer rim ring
    inner_rim = ring(float(inner_top_radius), total_h)  # [2*n : 3*n) inner rim ring
    cavity_floor = ring(float(inner_bottom_radius), float(bottom_thickness))  # [3*n : 4*n) cavity-floor ring
    obc = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)  # outer base centre
    ifc = np.array([[0.0, 0.0, float(bottom_thickness)]], dtype=np.float32)  # cavity-floor centre
    vertices = np.vstack([outer_base, outer_rim, inner_rim, cavity_floor, obc, ifc]).astype(np.float32)
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
        _assert_closed_oriented(vertices, indices, "Cup")

    return vertices, indices


def _assert_closed_oriented(vertices: np.ndarray, indices: np.ndarray, name: str) -> None:
    """Assert the mesh is a closed, consistently-oriented manifold (every directed edge appears once)."""
    tris = indices.reshape(-1, 3)
    directed: dict[tuple[int, int], int] = {}
    for tri in tris:
        for a, b in ((int(tri[0]), int(tri[1])), (int(tri[1]), int(tri[2])), (int(tri[2]), int(tri[0]))):
            directed[(a, b)] = directed.get((a, b), 0) + 1
    dup = [e for e, c in directed.items() if c != 1]
    if dup:
        raise RuntimeError(f"{name} mesh winding is inconsistent: {len(dup)} directed edges not unique.")
    open_edges = [(a, b) for (a, b) in directed if (b, a) not in directed]
    if open_edges:
        raise RuntimeError(f"{name} mesh is not watertight: {len(open_edges)} boundary edges.")


def make_hemisphere_scoop_mesh(
    *,
    inner_radius: float,
    wall_thickness: float,
    num_segments: int = 32,
    num_rings: int = 10,
    validate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a watertight, outward-wound thick hemispherical ladle/scoop (spherical shell, open top).

    A simpler, more grid-robust alternative to :func:`make_cup_collision_mesh`: the cavity is the lower half
    of a sphere of radius ``inner_radius``; the outer surface is a concentric hemisphere offset by
    ``wall_thickness``; an annular rim closes the shell at the top. Keep ``wall_thickness`` >= ~1.5 * MPM
    voxel_size so the shell is solid on the grid (else particles tunnel).

    Local frame: ``z=0`` is the outer base (table-facing); the cavity floor is at ``z=wall_thickness``; the
    rim is at ``z = inner_radius + wall_thickness``. Centred on the z axis, opening up (+z).

    Args:
        inner_radius: Cavity (inner hemisphere) radius [m]; also the cavity depth.
        wall_thickness: Shell thickness [m]; keep >= ~1.5 * MPM voxel_size to avoid tunnelling.
        num_segments: Radial tessellation (around the z axis).
        num_rings: Polar tessellation (apex to rim along the arc).
        validate: If True, assert the mesh is a closed, consistently-oriented manifold.

    Returns:
        ``(vertices, indices)`` with vertices ``(V, 3)`` float32 and triangle indices ``(3F,)`` int32.
    """
    r_in = float(inner_radius)
    w = float(wall_thickness)
    r_out = r_in + w
    n = int(num_segments)
    m = int(num_rings)

    ang = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    theta = np.linspace(0.0, 0.5 * math.pi, m + 1)  # 0 = nadir (apex), pi/2 = rim

    def ring(radius: float, z: float) -> np.ndarray:
        return np.column_stack([radius * ca, radius * sa, np.full(n, float(z))])

    outer_apex = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)  # idx 0
    inner_apex = np.array([[0.0, 0.0, w]], dtype=np.float64)  # idx 1
    outer_rings = [ring(r_out * math.sin(t), r_out * (1.0 - math.cos(t))) for t in theta[1:]]  # j=1..m
    inner_rings = [ring(r_in * math.sin(t), r_out - r_in * math.cos(t)) for t in theta[1:]]  # j=1..m
    vertices = np.vstack([outer_apex, inner_apex, *outer_rings, *inner_rings]).astype(np.float32)

    OA, IA = 0, 1
    o0, i0 = 2, 2 + m * n

    def o(j: int, i: int) -> int:  # outer ring j in 1..m
        return o0 + (j - 1) * n + (i % n)

    def iv(j: int, i: int) -> int:  # inner ring j in 1..m
        return i0 + (j - 1) * n + (i % n)

    faces: list[int] = []
    for i in range(n):
        faces += [OA, o(1, i + 1), o(1, i)]  # outer apex fan (outward, -z at nadir)
        faces += [IA, iv(1, i), iv(1, i + 1)]  # inner apex fan (cavity-facing, reversed)
    for j in range(1, m):
        for i in range(n):
            faces += [o(j, i), o(j, i + 1), o(j + 1, i), o(j + 1, i), o(j, i + 1), o(j + 1, i + 1)]  # outer band
            faces += [iv(j, i), iv(j + 1, i), iv(j, i + 1), iv(j, i + 1), iv(j + 1, i), iv(j + 1, i + 1)]  # inner band
    for i in range(n):  # rim annulus closing the shell at the top (wound to oppose the adjacent bands)
        faces += [o(m, i), o(m, i + 1), iv(m, i), o(m, i + 1), iv(m, i + 1), iv(m, i)]
    indices = np.asarray(faces, dtype=np.int32)

    # Auto-orient to outward (positive signed volume); flips consistently if my hand-winding is inward.
    tris = vertices[indices.reshape(-1, 3)]
    signed_vol = float(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum())
    if signed_vol < 0.0:
        indices = indices.reshape(-1, 3)[:, ::-1].reshape(-1).astype(np.int32)

    if validate:
        _assert_closed_oriented(vertices, indices, "Hemisphere scoop")

    return vertices, indices
