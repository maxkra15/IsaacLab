# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implicit MPM Newton manager."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import warp as wp
from newton import (
    BodyFlags,
    Contacts,
    Control,
    GeoType,
    Model,
    ModelBuilder,
    ModelFlags,
    ParticleFlags,
    State,
    StateFlags,
)
from newton.solvers import SolverImplicitMPM
from warp.fem import TemporaryStore

from isaaclab.physics import PhysicsManager

from .mpm_manager_cfg import MPMSolverCfg
from .newton_manager import NewtonManager


@wp.kernel(enable_backward=False)
def _scatter_particle_active_flag_index(
    active: wp.array2d(dtype=wp.bool),
    env_ids: wp.array(dtype=wp.int32),
    particle_offsets: wp.array(dtype=wp.int32),
    full_data: bool,
    active_flag: int,
    particle_flags: wp.array(dtype=wp.int32),
):
    row, particle = wp.tid()
    env_id = env_ids[row]
    source_row = env_id if full_data else row
    particle_id = particle_offsets[env_id] + particle
    if active[source_row, particle]:
        particle_flags[particle_id] = particle_flags[particle_id] | active_flag
    else:
        particle_flags[particle_id] = particle_flags[particle_id] & (~active_flag)


@dataclass(frozen=True)
class _MPMParticleActivationTarget:
    """Public Newton objects required to change fixed-capacity MPM particle activity."""

    solver: SolverImplicitMPM
    model_particle_flags: tuple[wp.array(dtype=wp.int32), ...]


def _make_solver_config(solver_cfg: MPMSolverCfg) -> SolverImplicitMPM.Config:
    """Build Newton's implicit MPM solver config from Isaac Lab's cfg."""
    collider_velocity_mode = solver_cfg.collider_velocity_mode
    deprecated_velocity_modes = {
        "instantaneous": "forward",
        "finite_difference": "backward",
    }
    if replacement := deprecated_velocity_modes.get(collider_velocity_mode):
        warnings.warn(
            f"collider_velocity_mode={collider_velocity_mode!r} is deprecated; use {replacement!r} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        collider_velocity_mode = replacement

    return SolverImplicitMPM.Config(
        max_iterations=solver_cfg.max_iterations,
        tolerance=solver_cfg.tolerance,
        solver=solver_cfg.solver,
        warmstart_mode=solver_cfg.warmstart_mode,
        collider_velocity_mode=collider_velocity_mode,
        voxel_size=solver_cfg.voxel_size,
        grid_type=solver_cfg.grid_type,
        grid_padding=solver_cfg.grid_padding,
        max_active_cell_count=solver_cfg.max_active_cell_count,
        max_leaf_node_count=solver_cfg.max_leaf_node_count,
        max_lower_node_count=solver_cfg.max_lower_node_count,
        max_upper_node_count=solver_cfg.max_upper_node_count,
        separate_worlds=solver_cfg.separate_worlds,
        transfer_scheme=solver_cfg.transfer_scheme,
        integration_scheme=solver_cfg.integration_scheme,
        critical_fraction=solver_cfg.critical_fraction,
        air_drag=solver_cfg.air_drag,
        collider_normal_from_sdf_gradient=solver_cfg.collider_normal_from_sdf_gradient,
        collider_basis=solver_cfg.collider_basis,
        strain_basis=solver_cfg.strain_basis,
        velocity_basis=solver_cfg.velocity_basis,
    )


class NewtonMPMManager(NewtonManager):
    """:class:`NewtonManager` specialization for Newton's implicit MPM solver.

    MPM advances particle materials in-place and treats rigid geometry as
    colliders, so it does not consume Newton's rigid-body collision pipeline
    and steps with a single :class:`State`.
    """

    _project_outside_colliders: bool = False
    """Whether :meth:`_step_solver` projects particles out of colliders each substep.

    Set from :attr:`MPMSolverCfg.project_outside_colliders` in
    :meth:`_build_solver` and read in :meth:`_step_solver`.
    """

    @classmethod
    def _register_builder_attributes(cls, builder: ModelBuilder) -> None:
        """Register the particle custom attributes required by :class:`SolverImplicitMPM`.

        Implicit MPM materials are configured per-particle through Newton
        custom attributes (``mpm:young_modulus``, ``mpm:viscosity``, ...).
        These must be present on the builder *before* particles are added so
        that ``add_particles(custom_attributes=...)`` succeeds and so that
        ``builder.finalize()`` allocates the matching model arrays.

        Idempotent: ``has_custom_attribute`` guards against re-registration
        when the hook is invoked multiple times (e.g. once via
        :meth:`create_builder` and again via :meth:`start_simulation`).
        """
        if not builder.has_custom_attribute("mpm:young_modulus"):
            SolverImplicitMPM.register_custom_attributes(builder)

    @classmethod
    def _prepare_builder_for_finalize(cls, builder: ModelBuilder) -> None:
        """Normalize rigid colliders before MPM solver construction.

        Newton's implicit MPM solver treats positive-mass body colliders as
        finite-mass colliders. Isaac Lab kinematic assets can import with a
        computed mass, so clear mass and inertia for kinematic bodies to match
        Newton's direct-builder MPM examples. The solver consumes mesh vertices
        and indices but only accepts the triangle-mesh geometry type, so classify
        convex meshes as meshes without changing their geometry.
        """
        kinematic_flag = int(BodyFlags.KINEMATIC)
        for body_id, flags in enumerate(builder.body_flags):
            if int(flags) & kinematic_flag:
                builder.body_mass[body_id] = 0.0
                builder.body_inv_mass[body_id] = 0.0
                builder.body_inertia[body_id] = wp.mat33()
                builder.body_inv_inertia[body_id] = wp.mat33()
        for shape_id, shape_type in enumerate(builder.shape_type):
            if shape_type == GeoType.CONVEX_MESH:
                builder.shape_type[shape_id] = GeoType.MESH

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: MPMSolverCfg) -> SolverImplicitMPM:
        """Construct the configured implicit MPM solver."""
        return SolverImplicitMPM(
            model,
            _make_solver_config(solver_cfg),
            temporary_store=TemporaryStore(),
        )

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: MPMSolverCfg) -> None:
        """Construct :class:`SolverImplicitMPM` and populate the base-class slots.

        MPM steps in-place on a single :class:`State` and runs collision
        handling internally, so it neither double-buffers state nor drives
        Newton's :class:`CollisionPipeline`.

        Args:
            model: Finalized Newton model the solver should run on.
            solver_cfg: Implicit MPM solver configuration.
        """
        NewtonManager._solver = cls._create_solver(model, solver_cfg)
        NewtonManager._use_single_state = True
        NewtonManager._needs_collision_pipeline = False
        cls._project_outside_colliders = solver_cfg.project_outside_colliders

    @classmethod
    def _defer_standard_graph_capture(cls) -> bool:
        """Defer capture until reset-dependent sparse topology has been initialized."""
        return cls._solver.grid_type == "sparse"

    @classmethod
    def _implicit_mpm_solvers(cls) -> tuple[SolverImplicitMPM, ...]:
        """Return direct or coupled implicit-MPM solvers without importing the coupler."""
        root_solver = NewtonManager._solver
        if isinstance(root_solver, SolverImplicitMPM):
            return (root_solver,)
        if root_solver is None or not hasattr(root_solver, "entry_names") or not hasattr(root_solver, "solver"):
            return ()
        return tuple(
            entry_solver
            for name in root_solver.entry_names()
            if isinstance((entry_solver := root_solver.solver(name)), SolverImplicitMPM)
        )

    @classmethod
    def _resolve_particle_activation_target(
        cls,
        particle_offsets: tuple[int, ...],
        particles_per_instance: int,
    ) -> _MPMParticleActivationTarget:
        """Resolve the owning solver for eager runtime activation of one MPM object.

        Newton derives its transfer and material masks from the public model
        particle flags. Updating those flags followed by
        :meth:`SolverImplicitMPM.notify_model_changed` keeps Isaac Lab on
        Newton's supported API. Newton currently rebuilds solver-owned arrays
        during that notification, so runtime activation requires eager
        execution rather than a previously captured CUDA graph.
        """
        if not particle_offsets or particles_per_instance < 1:
            raise ValueError("Particle activation requires non-empty instance offsets and a positive slot count.")
        cfg = PhysicsManager._cfg
        if cfg is not None and cfg.use_cuda_graph:
            raise RuntimeError(
                "Runtime MPM particle activation is not compatible with CUDA graph capture because Newton "
                "rebuilds solver-owned material arrays when particle flags change. Set NewtonCfg.use_cuda_graph=False."
            )
        master_model = NewtonManager.get_model()
        active_flag = int(ParticleFlags.ACTIVE)
        owning_solvers: list[SolverImplicitMPM] = []
        for solver in cls._implicit_mpm_solvers():
            solver_flags = solver.model.particle_flags.numpy()
            owns_object = True
            for offset in particle_offsets:
                selected = (solver_flags[offset : offset + particles_per_instance] & active_flag) != 0
                if np.any(selected) and not np.all(selected):
                    raise RuntimeError("An MPM object is only partially active in one coupled solver entry.")
                owns_object &= bool(np.all(selected))
            if owns_object:
                owning_solvers.append(solver)
        if len(owning_solvers) != 1:
            raise RuntimeError(
                "Runtime MPM particle activation requires exactly one owning implicit-MPM solver entry; "
                f"found {len(owning_solvers)}."
            )

        solver = owning_solvers[0]
        required_count = max(particle_offsets) + particles_per_instance
        arrays = (master_model.particle_flags, solver.model.particle_flags)
        if any(array is None or array.shape[0] < required_count for array in arrays):
            raise RuntimeError("The owning MPM solver does not preserve the object's global particle layout.")

        model_particle_flags: list[wp.array] = []
        for flags in (master_model.particle_flags, solver.model.particle_flags):
            if all(flags is not existing for existing in model_particle_flags):
                model_particle_flags.append(flags)
        # The Isaac Lab Newton visualizer normally caches its all-active fast path. Mark this
        # model so runtime-varying flags keep using Newton's device compaction.
        master_model._isaaclab_particle_flags_dynamic = True
        return _MPMParticleActivationTarget(
            solver=solver,
            model_particle_flags=tuple(model_particle_flags),
        )

    @classmethod
    def _write_particle_active_mask(
        cls,
        target: _MPMParticleActivationTarget,
        active: wp.array(dtype=wp.bool),
        env_ids: wp.array(dtype=wp.int32),
        particle_offsets: wp.array(dtype=wp.int32),
        *,
        full_data: bool,
        particles_per_instance: int,
    ) -> None:
        """Write a fixed-capacity active mask and refresh Newton's derived MPM material data."""
        launch_dim = (env_ids.shape[0], particles_per_instance)
        active_flag = int(ParticleFlags.ACTIVE)
        for particle_flags in target.model_particle_flags:
            wp.launch(
                _scatter_particle_active_flag_index,
                dim=launch_dim,
                inputs=[active, env_ids, particle_offsets, full_data, active_flag],
                outputs=[particle_flags],
                device=particle_flags.device,
            )
        target.solver.notify_model_changed(ModelFlags.MODEL_PROPERTIES)
        NewtonManager._mark_particles_dirty()

    @classmethod
    def _step_solver(
        cls, state_0: State, state_1: State, control: Control, contacts: Contacts | None, substep_dt: float
    ) -> None:
        """Run one implicit MPM substep, optionally projecting particles out of colliders.

        The implicit solve already resolves colliders at the grid level. When
        :attr:`MPMSolverCfg.project_outside_colliders` is set, the manager also
        runs ``project_outside`` after the step (as in Newton's MPM examples) to
        hard-project particles out of collider interiors. The flag is evaluated
        when the step is first run, so the chosen branch is baked into any
        captured CUDA graph.
        """
        cls._solver.step(state_0, state_1, control, contacts, substep_dt)
        if cls._project_outside_colliders:
            cls._solver.project_outside(state_1, state_1, substep_dt)

    @classmethod
    def _reset_solver_internals(cls, world_mask: wp.array | None) -> None:
        """Preserve the existing implicit-MPM behavior for automatic asset resets.

        Shared-grid MPM configurations cannot reset solver history for only a
        subset of worlds. Tasks that use independent MPM worlds and require an
        exact history reset call :meth:`reset_solver_state`
        explicitly after authoring their complete state.

        Args:
            world_mask: Per-world reset mask, intentionally ignored.
        """

    @classmethod
    def reset_solver_state(
        cls,
        state: State | None = None,
        world_mask: wp.array(dtype=wp.bool) | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Reset MPM and coupled-solver history after task state is rewritten.

        When :paramref:`state` is omitted, both distinct manager state buffers
        are reset so a later buffer swap cannot restore stale history. A mask
        follows Newton's canonical ``world_count + 1`` contract, where the last
        entry selects global entities in world -1. A selected single local world
        is promoted to a full reset because a one-world MPM grid has no
        environment offsets.

        Args:
            state: State whose solver-owned history should be reset. If omitted,
                reset both manager states.
            world_mask: Canonical per-world mask, including the final global-world entry.
            flags: State components whose solver-owned history should reset.

        Raises:
            RuntimeError: If the MPM solver or a usable state is not initialized.
            ValueError: If :paramref:`world_mask` does not use Newton's canonical shape.
        """
        solver = NewtonManager._solver
        model = NewtonManager._model
        if solver is None or model is None or not cls._implicit_mpm_solvers():
            raise RuntimeError("An implicit MPM solver is not initialized; cannot reset solver state.")

        reset_mask = world_mask
        if world_mask is not None:
            expected_shape = (model.world_count + 1,)
            if world_mask.shape != expected_shape:
                raise ValueError(f"world_mask must have shape {expected_shape}; got {world_mask.shape}.")
            if model.world_count == 1:
                selected = world_mask.numpy()
                if not selected.any():
                    return
                if selected[0] and not selected[-1]:
                    reset_mask = None

        candidates = (state,) if state is not None else (NewtonManager._state_1, NewtonManager._state_0)
        states: list[State] = []
        seen: set[int] = set()
        for candidate in candidates:
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            states.append(candidate)
        if not states:
            raise RuntimeError("Newton state is not initialized; provide an explicit state to reset.")

        for candidate in states:
            solver.reset(candidate, world_mask=reset_mask, flags=flags)

    @classmethod
    def _solver_specific_clear(cls) -> None:
        """Reset MPM-specific class state on teardown.

        :meth:`_build_solver` sets :attr:`_project_outside_colliders` from the
        active config. Resetting it here keeps a teardown-only :meth:`clear`
        (without a follow-up rebuild) from leaving a stale value on the class,
        mirroring how :meth:`NewtonManager.clear` resets the base-class flags.
        """
        cls._project_outside_colliders = False
