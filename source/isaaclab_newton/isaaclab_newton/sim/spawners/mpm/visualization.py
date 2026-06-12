# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np


class MPMParticleVisualization(NamedTuple):
    """USD point prims used by NewtonManager's Fabric particle sync."""

    base_path: str
    prim_paths: list[str]


def create_mpm_particle_visualization(
    prim_path: str,
    positions,
    particle_offsets: Sequence[int],
    widths: Sequence[float],
    color: Sequence[float],
    sync_frequency: int = 1,
) -> MPMParticleVisualization:
    """Create tagged USD ``Points`` prims for Newton MPM particle rendering in Kit.

    The created prims are static USD containers: per-frame position updates are
    handled by :meth:`isaaclab_newton.physics.NewtonManager.sync_particles_to_usd`.
    """
    from pxr import Gf, Sdf, UsdGeom, Vt  # noqa: PLC0415

    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    _ensure_xform_prim(sim_utils, stage, prim_path.rsplit("/", 1)[0])
    _ensure_xform_prim(sim_utils, stage, prim_path)

    positions_np = _as_batched_positions(positions)
    if len(particle_offsets) != positions_np.shape[0]:
        raise ValueError(
            "particle_offsets must contain one entry per particle-position batch. "
            f"Got {len(particle_offsets)} offsets for {positions_np.shape[0]} batches."
        )

    color = tuple(float(value) for value in color)
    if len(color) != 3:
        raise ValueError(f"MPM particle visualization color must contain three values. Got {color}.")
    if sync_frequency < 1:
        raise ValueError(f"MPM particle visualization sync_frequency must be >= 1. Got {sync_frequency}.")

    if len(widths) != positions_np.shape[1]:
        raise ValueError(
            "MPM particle visualization widths must contain one value per particle. "
            f"Got {len(widths)} widths for {positions_np.shape[1]} particles."
        )

    prim_paths: list[str] = []
    point_prims = []
    for env_idx in range(positions_np.shape[0]):
        env_prim_path = f"{prim_path}/env_{env_idx}"
        points = UsdGeom.Points.Define(stage, env_prim_path)
        # The point positions are written (and later synced) in WORLD space, but the prim may be
        # parented under a translated env prim (e.g. /World/envs/env_i/Media): reset the xform
        # stack so ancestor transforms are not applied on top of the world-space points.
        UsdGeom.Xformable(points.GetPrim()).SetResetXformStack(True)
        point_prims.append((env_prim_path, points))

    with Sdf.ChangeBlock():
        for env_idx, (env_prim_path, points) in enumerate(point_prims):
            points.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(positions_np[env_idx]))
            points.CreateWidthsAttr(Vt.FloatArray([float(width) for width in widths]))
            points.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
            points_prim = points.GetPrim()
            points_prim.CreateAttribute("newton:particleOffset", Sdf.ValueTypeNames.UInt, custom=True).Set(
                int(particle_offsets[env_idx])
            )
            points_prim.CreateAttribute("newton:particleCount", Sdf.ValueTypeNames.UInt, custom=True).Set(
                int(positions_np.shape[1])
            )
            points_prim.CreateAttribute("newton:particleSyncFrequency", Sdf.ValueTypeNames.UInt, custom=True).Set(
                int(sync_frequency)
            )
            prim_paths.append(env_prim_path)

    return MPMParticleVisualization(base_path=prim_path, prim_paths=prim_paths)


def _as_batched_positions(positions) -> np.ndarray:
    if hasattr(positions, "detach"):
        positions = positions.detach().cpu().numpy()

    positions_np = np.ascontiguousarray(positions, dtype=np.float32)
    if positions_np.ndim == 2:
        positions_np = positions_np.reshape(1, *positions_np.shape)
    if positions_np.ndim != 3 or positions_np.shape[-1] != 3:
        raise ValueError(
            f"MPM particle visualization positions must have shape (N, 3) or (E, N, 3). Got {positions_np.shape}."
        )
    return positions_np


def _ensure_xform_prim(sim_utils, stage, prim_path: str) -> None:
    if prim_path in ("", "/"):
        return
    parent_path = prim_path.rsplit("/", 1)[0]
    if parent_path and parent_path != prim_path:
        _ensure_xform_prim(sim_utils, stage, parent_path)
    if not stage.GetPrimAtPath(prim_path).IsValid():
        sim_utils.create_prim(prim_path, "Xform", stage=stage)
