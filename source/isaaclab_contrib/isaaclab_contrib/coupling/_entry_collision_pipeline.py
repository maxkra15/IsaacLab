# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collision-pipeline adapter for coupled-solver entries."""

from __future__ import annotations

import warp as wp
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import CollisionPipeline
from newton.solvers import SolverBase, SolverMuJoCo
from newton.solvers.experimental.coupled import CouplingInterface


class _EntryCollisionPipelineSolver(CouplingInterface):
    """Run a sub-solver with contacts generated against its own model view."""

    def __init__(self, solver: SolverBase, model_view, *, use_solver_effective_mass: bool = True):
        self._solver = solver
        self._use_solver_effective_mass = use_solver_effective_mass
        collision_cfg = NewtonManager._collision_cfg
        if collision_cfg is None:
            self._collision_pipeline = CollisionPipeline(model_view, broad_phase="explicit")
        else:
            self._collision_pipeline = CollisionPipeline(model_view, **collision_cfg.to_pipeline_args())
        self._contacts = self._collision_pipeline.contacts()
        self.coupling_unsupported = getattr(solver, "coupling_unsupported", frozenset())
        self._prepare_cuda_graph_buffers()

    def _prepare_cuda_graph_buffers(self) -> None:
        """Allocate MuJoCo's external-contact caches before CUDA graph capture."""
        solver = self._solver
        if not isinstance(solver, SolverMuJoCo) or solver._use_mujoco_contacts:
            return
        if solver.newton_shape_to_mjc_geom is None:
            solver._create_inverse_shape_mapping()
        launch_dim = min(self._contacts.rigid_contact_max, solver.mjw_data.naconmax)
        contact_map = solver._contact_tid_to_cid
        if contact_map is None or contact_map.shape[0] < launch_dim:
            solver._contact_tid_to_cid = wp.full(launch_dim, -1, dtype=wp.int32, device=solver.model.device)

    def __getattr__(self, name):
        return getattr(self._solver, name)

    def coupling_eval_effective_mass(self, *args, **kwargs):
        if getattr(self, "_use_solver_effective_mass", True):
            return self._solver.coupling_eval_effective_mass(*args, **kwargs)
        return CouplingInterface.coupling_eval_effective_mass(self, *args, **kwargs)

    def coupling_eval_effective_mass_block(self, *args, **kwargs):
        if getattr(self, "_use_solver_effective_mass", True):
            return self._solver.coupling_eval_effective_mass_block(*args, **kwargs)
        return CouplingInterface.coupling_eval_effective_mass_block(self, *args, **kwargs)

    def coupling_notify_input_state_update(self, *args, **kwargs):
        return self._solver.coupling_notify_input_state_update(*args, **kwargs)

    def coupling_supports_inertial_property_refresh(self, *args, **kwargs):
        return self._solver.coupling_supports_inertial_property_refresh(*args, **kwargs)

    def coupling_eval_gravity_acceleration(self, *args, **kwargs):
        return self._solver.coupling_eval_gravity_acceleration(*args, **kwargs)

    def coupling_rewind_proxy_body(self, *args, **kwargs):
        return self._solver.coupling_rewind_proxy_body(*args, **kwargs)

    def coupling_rewind_proxy_particle(self, *args, **kwargs):
        return self._solver.coupling_rewind_proxy_particle(*args, **kwargs)

    def coupling_harvest_proxy_wrenches(self, *args, **kwargs):
        return self._solver.coupling_harvest_proxy_wrenches(*args, **kwargs)

    def coupling_harvest_proxy_particle_forces(self, *args, **kwargs):
        return self._solver.coupling_harvest_proxy_particle_forces(*args, **kwargs)

    def coupling_prepare_proxy_contacts(self, *args, **kwargs):
        return self._solver.coupling_prepare_proxy_contacts(*args, **kwargs)

    def step(self, state_in, state_out, control, contacts, dt):
        del contacts
        self._collision_pipeline.collide(state_in, self._contacts)
        self._solver.step(state_in, state_out, control, self._contacts, dt)
