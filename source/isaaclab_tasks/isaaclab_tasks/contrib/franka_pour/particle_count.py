# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fresh full-MPM configurations for externally orchestrated particle-count stages."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import numpy as np
from isaaclab_newton.sim.spawners.mpm import MPMPointsCfg

from isaaclab.utils import validate

from .geometry import SOURCE_CUP_GEOMETRY
from .pour_env_cfg import _resolve_pour_solver_tree, configure_mpm_capacities

if TYPE_CHECKING:
    from .pour_env_cfg import FrankaPourResetDatasetEnvCfg


def particle_count_env_cfg(
    base_cfg: FrankaPourResetDatasetEnvCfg,
    points: MPMPointsCfg,
    *,
    reference_mass: float,
    reference_volume: float,
) -> FrankaPourResetDatasetEnvCfg:
    """Copy the task configuration for a genuinely different, uniform particle cloud.

    Positions and velocities are source-cup-local. The caller supplies the cloud;
    this helper does not invent particles or fit their distribution. Uniform masses
    keep existing count fractions equivalent to mass fractions. Reference volume is
    ``sum(8 * radius**3)``, not a measurement of instantaneous bulk volume.

    Close the previous environment and construct a new one from the result at a
    training boundary. All worlds and material histories restart; policy and optimizer
    state belong to the external training driver. No live arrays are resized.

    Sparse cell and particle-contact capacities are rounded up to powers of two.
    GIMP reserves up to eight cells per particle's finite domain; PIC reserves one.
    Contacts use particle centers in both cases. Multiple worlds must be isolated
    so each cell contains at most one world's points. Other solver parameters and
    material values come from ``base_cfg``. Projection and simultaneous resolution curricula are rejected.
    This prototype supports at most 512 points per world. Five large-support MPM
    quadrature points are an underresolved discretization, not five hard spheres.
    """
    solver = _resolve_pour_solver_tree(base_cfg).media_solver
    if solver.project_outside_colliders:
        raise ValueError("Particle-count stages require project_outside_colliders=False.")
    if solver.grid_type != "sparse":
        raise ValueError("Particle-count stages require the sparse MPM grid.")
    if base_cfg.scene.num_envs > 1 and not solver.separate_worlds:
        raise ValueError("Multiple particle-count worlds require separate_worlds=True.")
    if base_cfg.mpm_resolution_stages:
        raise ValueError("Particle-count stages cannot run a simultaneous MPM resolution curriculum.")
    if not isinstance(points, MPMPointsCfg):
        raise TypeError("Particle-count stages require MPMPointsCfg.")
    q = np.asarray(points.positions, dtype=np.float64)
    if q.ndim != 2 or q.shape[1:] != (3,) or not 1 <= len(q) <= 512 or not np.isfinite(q).all():
        raise ValueError("Particle-count positions must be finite, with shape (N, 3) and 1 <= N <= 512.")
    if points.velocities is not None:
        velocity = np.asarray(points.velocities, dtype=np.float64)
        if velocity.shape != q.shape or not np.isfinite(velocity).all():
            raise ValueError("Particle-count velocities must be finite and match positions.")
    lower, upper = SOURCE_CUP_GEOMETRY.inner_bounds
    if not ((q >= lower) & (q <= upper)).all():
        raise ValueError("Particle-count centers must start inside the source-cup cavity.")
    masses, radii = [], []
    for name, value, destination in (("mass", points.mass, masses), ("radius", points.radius, radii)):
        values = np.asarray(value, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(len(q), values.item())
        if values.shape != (len(q),) or not np.isfinite(values).all() or not (values > 0).all():
            raise ValueError(f"Particle-count {name} must be positive and scalar or one value per point.")
        if not (values == values[0]).all():
            raise ValueError(f"Particle-count {name} must be uniform for count-based observations.")
        destination.extend(values)
    totals = (float(np.sum(masses)), float(np.sum(8 * np.asarray(radii) ** 3)))
    for name, actual, expected in zip(("mass", "reference volume"), totals, (reference_mass, reference_volume)):
        if not math.isfinite(expected) or expected <= 0 or not math.isclose(actual, expected, rel_tol=2e-6):
            raise ValueError(f"Particle-count {name} does not preserve its finite positive reference value.")
    # configclass.copy() re-runs the task's __post_init__, replacing caller solver overrides.
    cfg = copy.deepcopy(base_cfg)
    cloud = points.copy()
    # Keep exported task configurations plain-data even when the source cloud is NumPy-backed.
    cloud.positions = q.tolist()
    if points.velocities is not None:
        cloud.velocities = velocity.tolist()
    cloud.mass = float(masses[0])
    cloud.radius = float(radii[0])
    cloud.material = base_cfg.scene.media.spawn.material.copy()
    cfg.scene.media.spawn = cloud
    cfg.mpm_cell_cap_override = None
    cell_budget = len(q) * (8 if solver.integration_scheme == "gimp" else 1)
    cfg.mpm_cell_capacity_alignment = 1 << (cell_budget - 1).bit_length()
    _resolve_pour_solver_tree(cfg).media_solver.collider_basis = f"pic{1 << (len(q) - 1).bit_length()}"
    configure_mpm_capacities(cfg)
    validate(cfg)
    return cfg
