# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Franka pour granular-media fill (no simulator)."""

from types import SimpleNamespace

import numpy as np
from isaaclab_newton.assets.mpm_object.mpm_object import MPMObjectRegistryEntry, add_mpm_entry_to_builder

from isaaclab_tasks.contrib.franka_pour.cube_bowl_mesh import cube_bowl_inner_bounds
from isaaclab_tasks.contrib.franka_pour.cup_media import (
    build_media_object_cfg,
    cup_cavity_lattice,
    particle_mass_and_radius,
)
from isaaclab_tasks.contrib.franka_pour.media_fill import cube_fill_points
from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import FrankaPourResetDatasetEnvCfg

LO, HI = cube_bowl_inner_bounds(0.037, 0.037, 0.045, 0.009)
CLR = max(0.003, 3 * 0.002)


class _RecordingParticleBuilder:
    """Minimal Newton builder recording emitted point particles."""

    def __init__(self):
        self.particle_count = 0
        self.particle_kwargs = None

    def add_particles(self, **kwargs):
        self.particle_kwargs = kwargs
        self.particle_count += len(kwargs["pos"])


def _min_neighbor_distance(pts: np.ndarray, sample: int = 200) -> float:
    """Smallest nearest-neighbour distance over a subsample (numpy-only, no scipy)."""
    idx = np.linspace(0, len(pts) - 1, min(sample, len(pts))).astype(int)
    sub = pts[idx]
    best = np.inf
    for i in range(len(sub)):
        d = np.linalg.norm(pts - sub[i], axis=1)
        d[np.argmin(d)] = np.inf  # drop the self-distance (0)
        best = min(best, float(d.min()))
    return best


def test_points_nonempty_and_inside_cavity():
    pts = cube_fill_points(LO, HI, spacing=0.003, fill_frac=1.0, jitter=0.0)
    assert pts.dtype == np.float32 and pts.shape[1] == 3 and len(pts) > 200
    assert np.all(pts[:, 0] >= LO[0] + CLR - 1e-6) and np.all(pts[:, 0] <= HI[0] - CLR + 1e-6)
    assert np.all(pts[:, 1] >= LO[1] + CLR - 1e-6) and np.all(pts[:, 1] <= HI[1] - CLR + 1e-6)
    assert np.all(pts[:, 2] >= LO[2] + CLR - 1e-6)


def test_fill_frac_limits_height():
    full = cube_fill_points(LO, HI, spacing=0.003, fill_frac=1.0, jitter=0.0)
    half = cube_fill_points(LO, HI, spacing=0.003, fill_frac=0.5, jitter=0.0)
    assert float(half[:, 2].max()) < float(full[:, 2].max())
    assert len(half) < len(full)


def test_deterministic_seed():
    a = cube_fill_points(LO, HI, spacing=0.003, seed=7)
    b = cube_fill_points(LO, HI, spacing=0.003, seed=7)
    assert np.array_equal(a, b)


def test_no_overlap_min_spacing():
    pts = cube_fill_points(LO, HI, spacing=0.003, jitter=0.0)
    assert _min_neighbor_distance(pts) > 0.5 * 0.003


def test_particle_mass_and_radius_represent_one_full_lattice_cell():
    """Implicit MPM derives particle volume as 8*r^3, so r must be half the lattice spacing."""
    cfg = SimpleNamespace(
        media_particle_spacing=0.003,
        media_material=SimpleNamespace(density=1500.0),
    )
    mass, radius = particle_mass_and_radius(cfg)
    spacing = cfg.media_particle_spacing
    represented_volume = 8.0 * radius**3

    assert np.isclose(radius, 0.5 * spacing)
    assert np.isclose(represented_volume, spacing**3)
    assert np.isclose(mass / represented_volume, cfg.media_material.density)


def test_task_fill_fraction_means_represented_cavity_volume_not_inset_point_height():
    cfg = FrankaPourResetDatasetEnvCfg()
    # Fill-fraction fidelity needs multiple lattice layers. The task's intentionally coarse
    # rollout resolution is covered separately by the environment-config regression tests.
    cfg.media_particle_spacing = 0.003
    points, cell = cup_cavity_lattice(cfg)
    cavity_volume = cfg.source_cup_inner_width * cfg.source_cup_inner_depth * cfg.source_cup_cavity_depth
    represented_fill = len(points) * float(np.prod(cell)) / cavity_volume
    points_per_layer = len(np.unique(np.round(points[:, :2], decimals=5), axis=0))
    one_layer_fraction = points_per_layer * float(np.prod(cell)) / cavity_volume

    # The safe wall/rim inset can leave the nearest higher layer infeasible in a shallow cup.
    assert abs(represented_fill - cfg.media_fill_frac) <= one_layer_fraction


def test_solver_voxel_refinement_preserves_particle_layout_and_volume():
    cfg = FrankaPourResetDatasetEnvCfg()
    coarse_points, coarse_cell = cup_cavity_lattice(cfg)
    coarse_mass_radius = particle_mass_and_radius(cfg)

    cfg.voxel_size = 0.005
    fine_points, fine_cell = cup_cavity_lattice(cfg)

    assert len(coarse_points) == len(fine_points) == 245
    assert np.array_equal(coarse_points, fine_points)
    assert np.array_equal(coarse_cell, fine_cell)
    assert particle_mass_and_radius(cfg) == coarse_mass_radius


def test_default_mpm_collider_margin_preserves_particle_layout():
    """A sub-spacing MPM margin change must not silently alter the cached media layout."""
    cfg = FrankaPourResetDatasetEnvCfg()
    assert cfg.mpm_collider_margin < cfg.media_particle_spacing
    default_points, default_cell = cup_cavity_lattice(cfg)

    cfg.collider_margin = 0.010
    rigid_margin_points, rigid_margin_cell = cup_cavity_lattice(cfg)
    assert np.array_equal(default_points, rigid_margin_points)
    assert np.array_equal(default_cell, rigid_margin_cell)

    cfg.mpm_collider_margin = 0.001
    thinner_margin_points, thinner_margin_cell = cup_cavity_lattice(cfg)
    assert len(default_points) == len(thinner_margin_points) == 245
    assert np.array_equal(default_points, thinner_margin_points)
    assert np.array_equal(default_cell, thinner_margin_cell)

    cfg.mpm_collider_margin = 0.010
    thicker_margin_points, _ = cup_cavity_lattice(cfg)
    assert not np.array_equal(default_points, thicker_margin_points)
    assert np.max(np.abs(thicker_margin_points[:, :2])) < np.max(np.abs(default_points[:, :2]))


def test_media_object_emits_unchanged_world_points_from_local_nonidentity_pose():
    """The asset pose must transform the exact task-local lattice during Newton emission."""
    cfg = FrankaPourResetDatasetEnvCfg()
    cup_pos = (0.375, -0.1875, 0.25)
    cup_quat = (0.0, 0.0, 1.0, 0.0)
    local_points, _ = cup_cavity_lattice(cfg)
    media_cfg = build_media_object_cfg(cfg, cup_pos, cup_quat)

    builder = _RecordingParticleBuilder()
    entry = MPMObjectRegistryEntry(cfg=media_cfg)
    add_mpm_entry_to_builder(
        builder,
        entry,
        env_idx=0,
        env_position=[0.0, 0.0, 0.0],
        env_rotation=(0.0, 0.0, 0.0, 1.0),
    )

    assert builder.particle_kwargs is not None
    emitted_world_points = np.asarray(builder.particle_kwargs["pos"], dtype=np.float32)
    legacy_world_points = np.column_stack((-local_points[:, 0], -local_points[:, 1], local_points[:, 2])).astype(
        np.float64
    )
    legacy_world_points = (legacy_world_points + np.asarray(cup_pos, dtype=np.float64)).astype(np.float32)

    assert len(local_points) == len(emitted_world_points) == 245
    np.testing.assert_array_equal(np.asarray(media_cfg.spawn.positions, dtype=np.float32), local_points)
    assert media_cfg.init_state.pos == cup_pos
    assert media_cfg.init_state.rot == cup_quat
    np.testing.assert_array_equal(emitted_world_points, legacy_world_points)
