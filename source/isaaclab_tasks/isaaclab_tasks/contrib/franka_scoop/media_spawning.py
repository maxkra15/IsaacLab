# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config-side MPM media spawn-point generation for the Franka scoop task.

The media is declared as an :class:`~isaaclab_newton.assets.MPMObjectCfg` scene entity;
this module computes the env-frame spawn points (and per-particle mass/radius) for the
configured container geometry so the env config can build the spawner declaratively,
instead of the env class injecting particles into the Newton builder imperatively.

Particle spacing follows the MPM grid voxel: ``particles_per_cell`` samples per voxel per
axis (Newton MPM best practice), so the media is always resolved consistently with the
solver grid and particle size tracks ``voxel_size``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from isaaclab_newton.sim.spawners.mpm import MPMPointsCfg

if TYPE_CHECKING:
    from .scoop_env_cfg import FrankaScoopEnvCfg

# Procedural pour-demo bowl proportions (unit scale); the rigid/MPM bowl proxies and the
# media fill must agree on these so particles seed inside the collider.
POUR_BOWL_INNER_BOTTOM_RADIUS = 0.045
POUR_BOWL_INNER_TOP_RADIUS = 0.19
POUR_BOWL_WALL_THICKNESS = 0.025
POUR_BOWL_HEIGHT = 0.13
POUR_BOWL_BOTTOM_THICKNESS = 0.025

MEDIA_SPAWN_SEED = 7
"""Fixed seed for media spawn sampling: every env must emit identical particles."""


def container_geometry_kind(cfg: FrankaScoopEnvCfg) -> str:
    """Normalize ``cfg.container_geometry`` to one of ``"box"``, ``"bucket"``, ``"pour_bowl"``."""
    kind = str(getattr(cfg, "container_geometry", "") or "").strip().lower()
    if not kind:
        kind = "pour_bowl" if getattr(cfg, "use_pour_bowl_mesh", False) else "box"
    aliases = {
        "cylinder": "bucket",
        "cylindrical": "bucket",
        "buckets": "bucket",
        "bowl": "pour_bowl",
        "pour-bowl": "pour_bowl",
        "pour_bowls": "pour_bowl",
    }
    kind = aliases.get(kind, kind)
    if kind not in {"bucket", "pour_bowl", "box"}:
        raise ValueError(f"Unsupported container_geometry={kind!r}; expected 'bucket', 'pour_bowl', or 'box'.")
    return kind


def container_bowl_scale(cfg: FrankaScoopEnvCfg) -> float:
    """Uniform scale that maps the unit pour-demo bowl to ``cfg.bowl_target_diameter``."""
    outer_diameter = 2.0 * (POUR_BOWL_INNER_TOP_RADIUS + POUR_BOWL_WALL_THICKNESS)
    return float(cfg.bowl_target_diameter) / outer_diameter


def bucket_base_z(cfg: FrankaScoopEnvCfg, center) -> float:
    """Env-frame z of a bucket's outer base for a bucket centered at *center*."""
    return float(center[2]) - 0.5 * float(cfg.bucket_height)


def _particle_spacing(cfg: FrankaScoopEnvCfg) -> float:
    return float(cfg.voxel_size) / max(float(cfg.particles_per_cell), 1.0)


def _pile_points(cfg: FrankaScoopEnvCfg) -> tuple[np.ndarray, np.ndarray]:
    """Conical source pile (angle of repose) on the table, inside the retaining box."""
    from .pile_sampling import sample_conical_pile

    cx, cy, cz = cfg.source_center
    ihz = float(cfg.container_inner_half[2])
    floor_z = cz - ihz  # box floor == table top (env z=0)
    spacing = _particle_spacing(cfg)
    angle = math.atan(max(float(cfg.media_material.friction), 0.05))  # angle of repose ~ atan(friction)
    height = float(cfg.pile_height)
    base_radius = height / max(math.tan(angle), 1.0e-3)
    # Keep the pile base inside the retaining box footprint.
    base_radius = min(base_radius, float(cfg.container_inner_half[0]) - 2.0 * spacing)
    # Respect the angle of repose: when the box clips the base radius, cap the height too.
    # An over-steep cone slumps on the first solves and surges over the shallow retaining
    # wall (observed as particles spilling off the table right after reset).
    height = min(height, base_radius * math.tan(angle))
    cone_volume = (math.pi / 3.0) * base_radius * base_radius * height
    count = max(int(cone_volume / (spacing**3)), 64)
    points = sample_conical_pile(
        count,
        (float(cx), float(cy), float(floor_z) + 0.5 * spacing),
        height=height,
        base_radius=base_radius,
        jitter=float(cfg.pile_jitter),
        seed=MEDIA_SPAWN_SEED,
        device="cpu",
    )
    cell = np.full(3, spacing, dtype=np.float32)
    return points, cell


def _source_interior(cfg: FrankaScoopEnvCfg, kind: str) -> dict:
    """Interior bounds of the source container (mirrors the env's collider construction)."""
    cx, cy, cz = cfg.source_center
    if kind == "bucket":
        base_z = bucket_base_z(cfg, cfg.source_center)
        return {
            "kind": "bucket",
            "cx": float(cx),
            "cy": float(cy),
            "floor_z": base_z + float(cfg.bucket_bottom_thickness),
            "rim_z": base_z + float(cfg.bucket_height),
            "inner_radius": float(cfg.bucket_inner_radius),
        }
    scale = container_bowl_scale(cfg)
    base_z = float(cz) - float(cfg.container_inner_half[2])
    return {
        "kind": "pour_bowl",
        "cx": float(cx),
        "cy": float(cy),
        "floor_z": base_z + POUR_BOWL_BOTTOM_THICKNESS * scale,
        "rim_z": base_z + POUR_BOWL_HEIGHT * scale,
        "inner_bottom_r": POUR_BOWL_INNER_BOTTOM_RADIUS * scale,
        "inner_top_r": POUR_BOWL_INNER_TOP_RADIUS * scale,
    }


def _lattice_points(cfg: FrankaScoopEnvCfg, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Jittered lattice fill of the source container interior (bucket / pour_bowl / legacy box)."""
    interior = _source_interior(cfg, kind) if kind in {"bucket", "pour_bowl"} else None
    if kind == "bucket":
        clearance = max(float(cfg.voxel_size), 3.0 * float(cfg.collider_margin))
        interior_height = max(float(interior["rim_z"]) - float(interior["floor_z"]), 1.0e-6)
        min_depth = 0.25 * float(cfg.voxel_size)
        fill_top = float(interior["floor_z"]) + interior_height * float(cfg.media_fill_frac)
        fill_top = min(fill_top, float(interior["rim_z"]) - clearance)
        radius = max(float(interior["inner_radius"]) - clearance, 0.25 * float(cfg.voxel_size))
        floor = float(interior["floor_z"]) + clearance
        depth = max(min_depth, fill_top - floor)
        lo = np.array([interior["cx"] - radius, interior["cy"] - radius, floor], dtype=np.float32)
        hi = np.array([interior["cx"] + radius, interior["cy"] + radius, floor + depth], dtype=np.float32)
        center_xy = np.array([interior["cx"], interior["cy"]], dtype=np.float32)
        radius_mode = "constant"
    elif kind == "pour_bowl":
        # Seed like the pour demo: at least one voxel / several margins away from the collider,
        # small jitter, then filter to the bowl's cylinder instead of relying on large grid jitter.
        clearance = max(float(cfg.voxel_size), 3.0 * float(cfg.collider_margin))
        bowl_height = max(float(interior["rim_z"]) - float(interior["floor_z"]), 1.0e-6)
        depth = max(0.0, bowl_height * float(cfg.media_fill_frac))
        depth = min(depth, max(bowl_height - 2.0 * clearance, 0.25 * float(cfg.voxel_size)))
        top_t = np.clip(depth / bowl_height, 0.0, 1.0)
        top_radius = float(interior["inner_bottom_r"]) + top_t * (
            float(interior["inner_top_r"]) - float(interior["inner_bottom_r"])
        )
        radius = max(top_radius - clearance, 0.25 * float(cfg.voxel_size))
        floor = float(interior["floor_z"]) + clearance
        lo = np.array([interior["cx"] - radius, interior["cy"] - radius, floor], dtype=np.float32)
        hi = np.array([interior["cx"] + radius, interior["cy"] + radius, floor + depth], dtype=np.float32)
        center_xy = np.array([interior["cx"], interior["cy"]], dtype=np.float32)
        radius_mode = "frustum"
    else:
        cx, cy, cz = cfg.source_center
        ihx, ihy, ihz = cfg.container_inner_half
        clearance = max(float(cfg.voxel_size), 3.0 * float(cfg.collider_margin), 0.015)
        lo = np.array([cx - ihx + clearance, cy - ihy + clearance, cz - ihz + clearance], dtype=np.float32)
        hi = np.array(
            [cx + ihx - clearance, cy + ihy - clearance, cz - ihz + clearance + 2 * ihz * cfg.media_fill_frac],
            dtype=np.float32,
        )
        center_xy = None
        radius_mode = "box"

    res = np.maximum(np.ceil(cfg.particles_per_cell * (hi - lo) / cfg.voxel_size), 1).astype(int)
    cell = (hi - lo) / res
    px = np.arange(int(res[0]) + 1, dtype=np.float32) * cell[0]
    py = np.arange(int(res[1]) + 1, dtype=np.float32) * cell[1]
    pz = np.arange(int(res[2]) + 1, dtype=np.float32) * cell[2]
    points = np.stack(np.meshgrid(px, py, pz, indexing="ij")).reshape(3, -1).T
    rng = np.random.default_rng(MEDIA_SPAWN_SEED)
    points += (rng.random(points.shape, dtype=np.float32) - 0.5) * (0.10 * float(np.max(cell)))
    points += lo
    if center_xy is not None and radius_mode == "frustum":
        z_t = np.clip(
            (points[:, 2] - float(interior["floor_z"]))
            / max(float(interior["rim_z"]) - float(interior["floor_z"]), 1.0e-6),
            0.0,
            1.0,
        )
        local_radius = (
            float(interior["inner_bottom_r"])
            + z_t * (float(interior["inner_top_r"]) - float(interior["inner_bottom_r"]))
            - clearance
        )
        local_radius = np.maximum(local_radius, 0.25 * float(cfg.voxel_size))
        normalized_xy = (points[:, :2] - center_xy) / local_radius[:, None]
        points = points[np.sum(normalized_xy * normalized_xy, axis=1) < 1.0]
    elif center_xy is not None and radius_mode == "constant":
        normalized_xy = (points[:, :2] - center_xy) / max(radius, 1.0e-6)
        points = points[np.sum(normalized_xy * normalized_xy, axis=1) < 1.0]
    if points.shape[0] == 0:
        raise RuntimeError("Particle initialization produced no media particles; reduce voxel size or clearance.")
    return points, cell


def compute_media_spawn_points(cfg: FrankaScoopEnvCfg) -> tuple[np.ndarray, np.ndarray]:
    """Env-frame media spawn points for the configured container geometry.

    Returns:
        ``(points, cell)`` with points ``(N, 3)`` float32 in the env frame and the
        lattice cell size ``(3,)`` float32 [m] used for mass/radius derivation.
    """
    kind = container_geometry_kind(cfg)
    if kind == "box":
        return _pile_points(cfg)
    return _lattice_points(cfg, kind)


def build_media_spawn_cfg(cfg: FrankaScoopEnvCfg) -> MPMPointsCfg:
    """Build the declarative media particle spawner from the env config."""
    points, cell = compute_media_spawn_points(cfg)
    radius = float(np.max(cell) * 0.45)
    mass = float(np.prod(cell) * cfg.media_material.density)
    return MPMPointsCfg(
        positions=points.astype(np.float32, copy=False).tolist(),
        mass=mass,
        radius=radius,
        material=cfg.media_material,
        visual_color=(0.85, 0.72, 0.45),
    )
