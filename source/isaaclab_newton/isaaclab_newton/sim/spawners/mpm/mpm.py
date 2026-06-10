# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from pxr import Usd

from .mpm_cfg import MPMGridCfg, MPMParticleSpawnerCfg, MPMPointsCfg


def _clone_after_kit_start(func: Callable) -> Callable:
    cloned_func: Callable | None = None

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal cloned_func
        if cloned_func is None:
            from isaaclab.sim.utils import clone

            cloned_func = clone(func)
        return cloned_func(*args, **kwargs)

    return wrapper


@_clone_after_kit_start
def spawn_mpm_particles(
    prim_path: str,
    cfg: MPMParticleSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a lightweight placeholder prim for a Newton MPM particle object.

    MPM particles are inserted directly into the Newton model builder during
    Newton replication. The USD prim exists so Isaac Lab's normal asset spawning
    and clone-planning machinery can reason about the scene entity.
    """
    from isaaclab.sim.utils import create_prim, get_current_stage

    stage = get_current_stage()
    return create_prim(
        prim_path,
        prim_type="Xform",
        translation=translation,
        orientation=orientation,
        stage=stage,
    )


def emit_mpm_particles(
    builder,
    cfg: MPMParticleSpawnerCfg,
    *,
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
) -> None:
    """Emit particles described by ``cfg`` into a Newton ``ModelBuilder``."""
    if isinstance(cfg, MPMGridCfg):
        _emit_grid(builder, cfg, position=position, orientation=orientation)
    elif isinstance(cfg, MPMPointsCfg):
        _emit_points(builder, cfg, position=position, orientation=orientation)
    else:
        raise TypeError(f"Unsupported MPM particle spawner config type: {type(cfg).__name__}")


def _emit_grid(
    builder,
    cfg: MPMGridCfg,
    *,
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
) -> None:
    lower = np.asarray(cfg.lower, dtype=np.float32)
    upper = np.asarray(cfg.upper, dtype=np.float32)
    extent = upper - lower
    if np.any(extent <= 0.0):
        raise ValueError(f"MPMGridCfg upper corner must be greater than lower corner. Got {cfg.lower=} {cfg.upper=}.")
    if cfg.voxel_size <= 0.0:
        raise ValueError(f"MPMGridCfg voxel_size must be positive. Got {cfg.voxel_size}.")
    if cfg.particles_per_cell <= 0.0:
        raise ValueError(f"MPMGridCfg particles_per_cell must be positive. Got {cfg.particles_per_cell}.")

    resolution = np.maximum(np.ceil(cfg.particles_per_cell * extent / cfg.voxel_size), 1).astype(np.int32)
    cell_size = extent / resolution
    cell_volume = float(np.prod(cell_size))
    mass = float(cfg.mass) if cfg.mass is not None else cell_volume * float(cfg.material.density)
    radius = float(cfg.radius) if cfg.radius is not None else 0.5 * float(np.max(cell_size))

    world_pos = _transform_point(lower, position, orientation)
    builder.add_particle_grid(
        pos=wp.vec3(*world_pos.tolist()),
        rot=wp.quat(*orientation),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=int(resolution[0]) + 1,
        dim_y=int(resolution[1]) + 1,
        dim_z=int(resolution[2]) + 1,
        cell_x=float(cell_size[0]),
        cell_y=float(cell_size[1]),
        cell_z=float(cell_size[2]),
        mass=mass,
        jitter=float(cfg.jitter),
        radius_mean=radius,
        custom_attributes=cfg.material.to_custom_attributes(),
    )


def _emit_points(
    builder,
    cfg: MPMPointsCfg,
    *,
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
) -> None:
    points = np.asarray(cfg.positions, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"MPMPointsCfg positions must have shape (N, 3). Got {points.shape}.")
    if points.shape[0] == 0:
        raise ValueError("MPMPointsCfg positions must contain at least one particle.")

    velocities = np.zeros_like(points) if cfg.velocities is None else np.asarray(cfg.velocities, dtype=np.float32)
    if velocities.shape != points.shape:
        raise ValueError(f"MPMPointsCfg velocities must match positions shape {points.shape}. Got {velocities.shape}.")

    world_points = _transform_points(points, position, orientation)
    world_velocities = _rotate_vectors(velocities, orientation)

    builder.add_particles(
        pos=world_points.tolist(),
        vel=world_velocities.tolist(),
        mass=_expand_scalar_or_sequence(cfg.mass, points.shape[0], "mass"),
        radius=_expand_scalar_or_sequence(cfg.radius, points.shape[0], "radius"),
        custom_attributes=cfg.material.to_custom_attributes(),
    )


def _expand_scalar_or_sequence(value: float | Sequence[float], count: int, name: str) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * count
    if len(value) != count:
        raise ValueError(f"MPMPointsCfg {name} must be scalar or have one value per particle. Got {len(value)} values.")
    return [float(v) for v in value]


def _transform_point(
    point: np.ndarray,
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
) -> np.ndarray:
    return _transform_points(point.reshape(1, 3), position, orientation)[0]


def _transform_points(
    points: np.ndarray,
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
) -> np.ndarray:
    return _rotate_vectors(points, orientation) + np.asarray(position, dtype=np.float32)


def _rotate_vectors(vectors: np.ndarray, orientation: tuple[float, float, float, float]) -> np.ndarray:
    q = np.asarray(orientation, dtype=np.float32)
    q_vec = q[:3]
    q_w = q[3]
    uv = np.cross(q_vec, vectors)
    uuv = np.cross(q_vec, uv)
    return vectors + 2.0 * (q_w * uv + uuv)
