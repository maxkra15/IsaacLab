# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime MPM controls through real model-bound backends and coupled views."""

import newton
import numpy as np
import pytest
from isaaclab_newton.physics import MPMSolverCfg, NewtonCfg, XPBDSolverCfg
from isaaclab_newton.physics.newton_backend import NewtonBackend
from isaaclab_newton.physics.newton_backend_cfg import NewtonBackendCfg
from newton.solvers import SolverImplicitMPM

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

pytestmark = pytest.mark.kitless


def _backend(*, coupled=False, partial_owner=False):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    for x, mass in ((0.0, 0.01), (0.05, 0.02)):
        builder.add_particle(pos=(x, 0.0, 0.0), vel=(0.1, 0.0, 0.0), mass=mass, radius=0.02)
    mpm = MPMSolverCfg(grid_type="dense", grid_padding=1, voxel_size=0.1, max_iterations=1)
    config = mpm
    if coupled:
        config = CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="media", particles=[0] if partial_owner else [1, 0], in_place=True, solver_cfg=mpm
                ),
                CouplerEntryCfg(
                    name="other", particles=[1] if partial_owner else [], solver_cfg=XPBDSolverCfg(iterations=1)
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="other" if partial_owner else "media",
                    destination="media" if partial_owner else "other",
                    particles=[1],
                    mode="lagged" if partial_owner else "staggered",
                    collision_pipeline=None,
                )
            ],
        )
    backend = NewtonBackend(
        NewtonBackendCfg(device="cpu", physics_cfg=NewtonCfg(use_cuda_graph=False, solver_cfg=config)),
        builder,
        dt=0.001,
    )
    backend.init_solver()
    return backend


@pytest.mark.parametrize("coupled", (False, True), ids=("direct", "coupled"))
def test_resolution_preserves_live_state_and_updates_coupled_models(coupled):
    backend = _backend(coupled=coupled)
    solver = backend.solver.solver("media") if coupled else backend.solver
    state = backend.solver.entry_state("media") if coupled else backend.state_0
    kwargs = {"entry_name": "media"} if coupled else {}
    solver.step(state, state, None, None, 0.001)
    state.particle_q.assign(state.particle_q.numpy() + np.array([1.0, 2.0, 3.0], dtype=np.float32))
    positions, velocities = state.particle_q.numpy().copy(), state.particle_qd.numpy().copy()
    masses = backend.model.particle_mass.numpy().copy()
    state.mpm.particle_Jp.fill_(0.9)
    backend.graphs = {"step": object()}
    with pytest.raises(ValueError):
        backend.set_voxel_size(float("nan"), **kwargs)
    assert backend.graphs
    backend.set_voxel_size(0.05, **kwargs)
    assert not backend.graphs
    assert solver.voxel_size == 0.05
    config = backend.cfg.solver_cfg.entries[0].solver_cfg if coupled else backend.cfg.solver_cfg
    assert config.voxel_size == 0.05
    grid = solver._scratchpad.grid
    assert np.all(np.asarray(grid.origin) <= positions.min(axis=0))
    assert np.all(np.asarray(grid.origin) + np.asarray(grid.res) * np.asarray(grid.cell_size) >= positions.max(axis=0))
    backend.refit_bvh(7)
    backend.graphs = {"step": object()}
    backend.set_particle_radius(0.015, **kwargs)
    backend.refit_bvh(7)
    np.testing.assert_allclose(
        backend.model.bvh_particles.uppers.numpy() - backend.model.bvh_particles.lowers.numpy(), 0.03, atol=1.0e-6
    )
    assert not backend.graphs
    models = [backend.model]
    if coupled:
        models += [backend.solver.view(name) for name in backend.solver.entry_names()]
    for model in models:
        np.testing.assert_allclose(model.particle_radius.numpy(), 0.015)
        assert model.particle_max_radius == 0.015
    np.testing.assert_allclose(solver._mpm_model.particle_volume.numpy(), 8 * 0.015**3)
    np.testing.assert_allclose(
        solver._mpm_model.particle_density.numpy(), solver.model.particle_mass.numpy() / (8 * 0.015**3)
    )
    np.testing.assert_array_equal(state.mpm.particle_Jp.numpy(), np.full(2, 0.9, dtype=np.float32))
    np.testing.assert_array_equal(state.particle_q.numpy(), positions)
    np.testing.assert_array_equal(state.particle_qd.numpy(), velocities)
    np.testing.assert_array_equal(backend.model.particle_mass.numpy(), masses)
    solver.step(state, state, None, None, 0.001)
    assert np.isfinite(state.particle_q.numpy()).all()
    assert np.linalg.norm(state.particle_q.numpy() - positions) > 0


def test_radius_rejects_foreign_proxies_without_partial_mutation():
    backend = _backend(coupled=True, partial_owner=True)
    solver = backend.solver.solver("media")
    assert solver.model.particle_count == backend.model.particle_count
    before = backend.model.particle_radius.numpy().copy()
    volume = solver._mpm_model.particle_volume.numpy().copy()
    graph = backend.graphs = {"step": object()}
    with pytest.raises(ValueError, match="own.*all|all.*particle|subset"):
        backend.set_particle_radius(0.03, entry_name="media")
    np.testing.assert_array_equal(backend.model.particle_radius.numpy(), before)
    np.testing.assert_array_equal(solver._mpm_model.particle_volume.numpy(), volume)
    assert backend.graphs is graph
