# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config-side MPM media generation for the dynamic hollow-cube source cup."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.spawners.mpm import MPMPointsCfg

from .cube_bowl_mesh import cube_bowl_inner_bounds
from .media_fill import cube_fill_points

if TYPE_CHECKING:
    from .pour_env_cfg import FrankaPourResetDatasetEnvCfg

MEDIA_SPAWN_SEED = 7
"""Fixed seed for media spawn sampling: every env must emit identical particles."""


def particle_spacing(cfg: FrankaPourResetDatasetEnvCfg) -> float:
    """Return the task's MPM particle-lattice spacing [m].

    Particle sampling and solver-grid resolution are deliberately independent.  This lets the
    collision grid be refined without silently changing the represented media volume, particle
    count, mass, or render radius.
    """
    return float(cfg.media_particle_spacing)


def particle_mass_and_radius(cfg: FrankaPourResetDatasetEnvCfg) -> tuple[float, float]:
    """Return mass [kg] and radius [m] for one MPM lattice cell.

    Newton's implicit MPM backend treats each particle as a cube with volume
    ``8 * radius**3``.  A radius of half the lattice spacing therefore makes the
    represented particle volume exactly match the cell volume used to compute
    mass, preserving the configured material density.
    """
    spacing = particle_spacing(cfg)
    volume = spacing**3
    return float(volume * cfg.media_material.density), float(0.5 * spacing)


def cup_cavity_lattice(cfg: FrankaPourResetDatasetEnvCfg) -> tuple[np.ndarray, np.ndarray]:
    """Jittered lattice filling the source cup cavity in its local frame.

    Args:
        cfg: The pour env config (cup geometry + MPM spacing fields).

    Returns:
        ``(points, cell)`` with cup-local points ``(N, 3)`` float32 and the lattice cell size
        ``(3,)`` float32 [m] used for per-particle mass/radius derivation.
    """
    spacing = particle_spacing(cfg)
    # Keep particle centres at least one lattice spacing / MPM collider margin from the wall. The
    # default 4 mm particle margin remains below the 5 mm lattice spacing, so separating it from the
    # rigid-contact margin does not change the default particle positions, count, or mass.
    clearance = max(spacing, float(cfg.mpm_collider_margin))
    lo, hi = cube_bowl_inner_bounds(
        float(cfg.source_cup_inner_width),
        float(cfg.source_cup_inner_depth),
        float(cfg.source_cup_cavity_depth),
        float(cfg.source_cup_bottom_thickness),
    )
    # ``cube_fill_points.fill_frac`` controls seed *height* inside an inset footprint, whereas the
    # task config describes the represented MPM *volume*. Choose the nearest whole number of z
    # layers whose cubic particle volumes match that requested cavity-volume fraction.
    spans = np.maximum((hi - lo)[:2] - 2.0 * clearance, 0.0)
    nx, ny = (int(np.floor(span / spacing)) + 1 for span in spans)
    cavity_volume = float(np.prod(hi - lo))
    target_count = float(cfg.media_fill_frac) * cavity_volume / spacing**3
    nz = max(1, int(round(target_count / max(nx * ny, 1))))
    fill_depth = min((nz - 1) * spacing + 1.0e-6 * spacing, float(hi[2] - lo[2]) - 2.0 * clearance)
    seed_height_frac = max(fill_depth, 0.0) / float(hi[2] - lo[2])
    points = cube_fill_points(
        lo,
        hi,
        spacing=spacing,
        fill_frac=seed_height_frac,
        clearance=clearance,
        jitter=0.05,
        seed=MEDIA_SPAWN_SEED,
    )
    if points.shape[0] == 0:
        raise RuntimeError("Cup media initialization produced no particles; reduce voxel size or clearance.")
    cell = np.full(3, spacing, dtype=np.float32)
    return points.astype(np.float32, copy=False), cell


def build_media_object_cfg(cfg: FrankaPourResetDatasetEnvCfg, cup_pos, cup_quat_xyzw) -> MPMObjectCfg:
    """Build the declarative cup-media :class:`MPMObjectCfg` from the env config.

    Args:
        cfg: The pour env config.
        cup_pos: Environment-local initial position [m] of the cup body (the cup-local frame origin).
        cup_quat_xyzw: Environment-local initial orientation (xyzw quaternion) of the cup body.

    Returns:
        An :class:`MPMObjectCfg` whose spawn points fill the cup cavity at the reset pose, with
        per-particle mass/radius derived from the lattice cell.
    """
    local_points, _ = cup_cavity_lattice(cfg)
    mass, radius = particle_mass_and_radius(cfg)
    return MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Media",
        spawn=MPMPointsCfg(
            positions=local_points.tolist(),
            mass=mass,
            radius=radius,
            material=cfg.media_material,
            visual_color=(0.85, 0.72, 0.45),
        ),
        init_state=MPMObjectCfg.InitialStateCfg(pos=cup_pos, rot=cup_quat_xyzw),
    )
