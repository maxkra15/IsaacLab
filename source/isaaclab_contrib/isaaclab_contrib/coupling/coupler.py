# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton coupler for named solver configurations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any

import warp as wp
from isaaclab_newton.physics import (
    KaminoDVISolverCfg,
    KaminoPADMMSolverCfg,
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonCollisionPipelineCfg,
    NewtonSolverCfg,
)
from isaaclab_newton.physics.mpm_manager import NewtonMPMManager
from isaaclab_newton.physics.newton_manager import NewtonManager
from isaaclab_newton.physics.vbd_manager import NewtonVBDManager
from newton import BodyFlags, CollisionPipeline, Model, ModelBuilder, ShapeFlags, StateFlags
from newton.math import quat_velocity
from newton.solvers import SolverVBD
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM, SolverCoupledProxy

from isaaclab.physics import PhysicsManager
from isaaclab.utils.string import resolve_matching_names

from .coupler_cfg import (
    CouplerAdmmCfg,
    CouplerCfg,
    CouplerEntryCfg,
    CouplerProxyCfg,
    CouplerProxyMappingCfg,
)

if TYPE_CHECKING:
    from newton import Contacts, Control, State
    from newton.solvers.experimental.coupled import ModelView


@wp.kernel
def _gather_vbd_pose_history(
    entry_local_body_ids: wp.array(dtype=wp.int32),
    source_body_q_prev: wp.array(dtype=wp.transform),
    source_coupling_body_q_prev: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    coupling_body_q_prev: wp.array(dtype=wp.transform),
) -> None:
    """Gather both VBD previous-pose histories in caller body order."""
    tid = wp.tid()
    local_body_id = entry_local_body_ids[tid]
    body_q_prev[tid] = source_body_q_prev[local_body_id]
    coupling_body_q_prev[tid] = source_coupling_body_q_prev[local_body_id]


@wp.kernel
def _queue_vbd_pose_history_bodies(
    entry_local_body_ids: wp.array(dtype=wp.int32),
    body_q_prev: wp.array(dtype=wp.transform),
    coupling_body_q_prev: wp.array(dtype=wp.transform),
    pending_body_q_prev: wp.array(dtype=wp.transform),
    pending_coupling_body_q_prev: wp.array(dtype=wp.transform),
    body_pending: wp.array(dtype=wp.bool),
) -> None:
    """Stage selected histories in stable entry-local graph buffers."""
    tid = wp.tid()
    local_body_id = entry_local_body_ids[tid]
    pending_body_q_prev[local_body_id] = body_q_prev[tid]
    pending_coupling_body_q_prev[local_body_id] = coupling_body_q_prev[tid]
    body_pending[local_body_id] = True


@wp.kernel
def _queue_vbd_pose_history_worlds(
    selected_world_ids: wp.array(dtype=wp.int32),
    generation: int,
    world_pending: wp.array(dtype=wp.bool),
    queued_generation: wp.array(dtype=wp.int32),
) -> None:
    """Mark selected worlds as holding one pending restore generation."""
    world_id = selected_world_ids[wp.tid()]
    queued_generation[world_id] = generation
    world_pending[world_id] = True


@wp.kernel
def _apply_pending_vbd_pose_history_bodies(
    dt: float,
    body_flags: wp.array(dtype=wp.int32),
    kinematic_flag: int,
    body_world: wp.array(dtype=wp.int32),
    body_pending: wp.array(dtype=wp.bool),
    world_pending: wp.array(dtype=wp.bool),
    rigid_pose_rebaseline_mask: wp.array(dtype=wp.bool),
    pending_body_q_prev: wp.array(dtype=wp.transform),
    pending_coupling_body_q_prev: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    coupling_body_q_prev: wp.array(dtype=wp.transform),
    state_body_q: wp.array(dtype=wp.transform),
    state_body_qd: wp.array(dtype=wp.spatial_vector),
    body_application_count: wp.array(dtype=wp.int32),
) -> None:
    """Rebaseline pending worlds, then replay selected continuous VBD input."""
    local_body_id = wp.tid()
    world_id = body_world[local_body_id]
    if world_id < 0:
        world_id = world_pending.shape[0] - 1
    if world_pending[world_id] and rigid_pose_rebaseline_mask[world_id]:
        q_current = state_body_q[local_body_id]
        body_q_prev[local_body_id] = q_current
        coupling_body_q_prev[local_body_id] = q_current
        if body_pending[local_body_id]:
            restored_q_prev = pending_body_q_prev[local_body_id]
            body_q_prev[local_body_id] = restored_q_prev
            coupling_body_q_prev[local_body_id] = pending_coupling_body_q_prev[local_body_id]
            if (body_flags[local_body_id] & kinematic_flag) == 0:
                dv = (wp.transform_get_translation(q_current) - wp.transform_get_translation(restored_q_prev)) / dt
                dw = quat_velocity(
                    wp.transform_get_rotation(q_current),
                    wp.transform_get_rotation(restored_q_prev),
                    dt,
                )
                state_body_qd[local_body_id] += wp.spatial_vector(dv, dw)
                state_body_q[local_body_id] = restored_q_prev
            wp.atomic_add(body_application_count, world_id, 1)
    if body_pending[local_body_id]:
        body_pending[local_body_id] = False


@wp.kernel
def _apply_pending_vbd_pose_history_bodies_preserved(
    dt: float,
    body_flags: wp.array(dtype=wp.int32),
    kinematic_flag: int,
    body_world: wp.array(dtype=wp.int32),
    body_pending: wp.array(dtype=wp.bool),
    world_pending: wp.array(dtype=wp.bool),
    rigid_pose_rebaseline_mask: wp.array(dtype=wp.bool),
    preserve_input_mask: wp.array(dtype=wp.bool),
    pending_body_q_prev: wp.array(dtype=wp.transform),
    pending_coupling_body_q_prev: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    coupling_body_q_prev: wp.array(dtype=wp.transform),
    state_body_q: wp.array(dtype=wp.transform),
    state_body_qd: wp.array(dtype=wp.spatial_vector),
    body_application_count: wp.array(dtype=wp.int32),
) -> None:
    """Replay pending histories without consuming preserved input poses."""
    local_body_id = wp.tid()
    world_id = body_world[local_body_id]
    if world_id < 0:
        world_id = world_pending.shape[0] - 1
    if world_pending[world_id] and rigid_pose_rebaseline_mask[world_id]:
        q_current = state_body_q[local_body_id]
        body_q_prev[local_body_id] = q_current
        coupling_body_q_prev[local_body_id] = q_current
        if body_pending[local_body_id]:
            restored_q_prev = pending_body_q_prev[local_body_id]
            body_q_prev[local_body_id] = restored_q_prev
            coupling_body_q_prev[local_body_id] = pending_coupling_body_q_prev[local_body_id]
            if not preserve_input_mask[local_body_id] and (body_flags[local_body_id] & kinematic_flag) == 0:
                dv = (wp.transform_get_translation(q_current) - wp.transform_get_translation(restored_q_prev)) / dt
                dw = quat_velocity(
                    wp.transform_get_rotation(q_current),
                    wp.transform_get_rotation(restored_q_prev),
                    dt,
                )
                state_body_qd[local_body_id] += wp.spatial_vector(dv, dw)
                state_body_q[local_body_id] = restored_q_prev
            wp.atomic_add(body_application_count, world_id, 1)
    if body_pending[local_body_id]:
        body_pending[local_body_id] = False


@wp.kernel
def _mark_vbd_preserved_input_pose_bodies(
    entry_local_body_ids: wp.array(dtype=wp.int32),
    preserve_input_mask: wp.array(dtype=wp.bool),
) -> None:
    """Mark entry-local bodies whose authored input pose must be preserved."""
    preserve_input_mask[entry_local_body_ids[wp.tid()]] = True


@wp.kernel
def _save_and_neutralize_vbd_preserved_input_pose(
    entry_local_body_ids: wp.array(dtype=wp.int32),
    body_world: wp.array(dtype=wp.int32),
    rigid_pose_rebaseline_mask: wp.array(dtype=wp.bool),
    neutral_body_q_prev: wp.array(dtype=wp.transform),
    state_body_q: wp.array(dtype=wp.transform),
    state_body_qd: wp.array(dtype=wp.spatial_vector),
    saved_body_q: wp.array(dtype=wp.transform),
    saved_body_qd: wp.array(dtype=wp.spatial_vector),
) -> None:
    """Save authored state and hide its pose delta from VBD's input hook."""
    tid = wp.tid()
    local_body_id = entry_local_body_ids[tid]
    saved_body_q[tid] = state_body_q[local_body_id]
    saved_body_qd[tid] = state_body_qd[local_body_id]
    world_id = body_world[local_body_id]
    if world_id < 0:
        world_id = rigid_pose_rebaseline_mask.shape[0] - 1
    if not rigid_pose_rebaseline_mask[world_id]:
        state_body_q[local_body_id] = neutral_body_q_prev[local_body_id]


@wp.kernel
def _restore_vbd_preserved_input_pose(
    entry_local_body_ids: wp.array(dtype=wp.int32),
    saved_body_q: wp.array(dtype=wp.transform),
    saved_body_qd: wp.array(dtype=wp.spatial_vector),
    state_body_q: wp.array(dtype=wp.transform),
    state_body_qd: wp.array(dtype=wp.spatial_vector),
) -> None:
    """Restore authored pose and velocity after VBD updates its histories."""
    tid = wp.tid()
    local_body_id = entry_local_body_ids[tid]
    state_body_q[local_body_id] = saved_body_q[tid]
    state_body_qd[local_body_id] = saved_body_qd[tid]


@wp.kernel
def _copy_parent_body_q_to_vbd_entry(
    parent_global_body_ids: wp.array(dtype=wp.int32),
    entry_local_body_ids: wp.array(dtype=wp.int32),
    parent_body_q: wp.array(dtype=wp.transform),
    entry_body_q: wp.array(dtype=wp.transform),
) -> None:
    """Publish projected parent poses back to the named entry output state."""
    tid = wp.tid()
    entry_body_q[entry_local_body_ids[tid]] = parent_body_q[parent_global_body_ids[tid]]


@wp.kernel
def _consume_pending_vbd_pose_history_worlds(
    world_pending: wp.array(dtype=wp.bool),
    queued_generation: wp.array(dtype=wp.int32),
    rigid_pose_rebaseline_mask: wp.array(dtype=wp.bool),
    applied_generation: wp.array(dtype=wp.int32),
    failed_generation: wp.array(dtype=wp.int32),
    application_count: wp.array(dtype=wp.int32),
) -> None:
    """Consume each pending world once after its normal baseline is ready."""
    world_id = wp.tid()
    if not world_pending[world_id]:
        return
    generation = queued_generation[world_id]
    if not rigid_pose_rebaseline_mask[world_id]:
        failed_generation[world_id] = generation
        world_pending[world_id] = False
        return
    rigid_pose_rebaseline_mask[world_id] = False
    applied_generation[world_id] = generation
    application_count[world_id] += 1
    world_pending[world_id] = False


@dataclass(frozen=True)
class VBDPoseHistoryRestoreStatus:
    """Immutable ticket and observable status for one deferred restore.

    ``body_ids`` and ``world_ids`` preserve the caller's accepted order.
    ``expected_body_counts`` and both application-count deltas use
    ``world_ids`` order. Pass any status returned for this request back to
    :meth:`NewtonCouplerManager.get_vbd_pose_history_restore_status` to refresh
    its device-observed state after a coupled solve.
    """

    entry_name: str
    generation: int
    body_ids: tuple[int, ...] = field(repr=False)
    world_ids: tuple[int, ...]
    expected_body_counts: tuple[int, ...]
    pending_world_ids: tuple[int, ...]
    applied_world_ids: tuple[int, ...]
    failed_world_ids: tuple[int, ...]
    superseded_world_ids: tuple[int, ...]
    application_count_deltas: tuple[int, ...]
    body_application_count_deltas: tuple[int, ...]
    _application_counts_before: tuple[int, ...] = field(repr=False, compare=False)
    _body_application_counts_before: tuple[int, ...] = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    @property
    def pending(self) -> bool:
        """Return whether any requested world still awaits its first solve."""
        return bool(self.pending_world_ids)

    @property
    def applied_exactly_once(self) -> bool:
        """Return whether every selected body wrote both histories exactly once."""
        return (
            not self.pending_world_ids
            and not self.failed_world_ids
            and not self.superseded_world_ids
            and self.applied_world_ids == self.world_ids
            and all(count == 1 for count in self.application_count_deltas)
            and self.body_application_count_deltas == self.expected_body_counts
        )


@dataclass(frozen=True)
class VBDPreservedInputPoseProjectionHandle:
    """Lifecycle handle for one deferred VBD input-pose projection."""

    name: str
    _registration_id: int = field(repr=False)
    _issuer: object = field(repr=False, compare=False)

    def deregister(self) -> bool:
        """Remove this registration from future solver builds.

        Returns:
            ``True`` if the registration was active, or ``False`` if it was
            already removed or its owning physics-manager lifecycle was
            cleared. A projection already captured into an active solver stays
            bound until that solver is cleared or rebuilt.
        """
        return NewtonCouplerManager._deregister_vbd_preserved_input_pose_projection(self)


@dataclass(frozen=True)
class _VBDPreservedInputPoseProjectionRegistration:
    """Deferred public registration tied to one finalized parent model."""

    registration_id: int
    name: str
    entry_name: str
    body_ids: tuple[int, ...]
    callback: Callable[[State], None] = field(repr=False, compare=False)
    model: Model = field(repr=False, compare=False)


@dataclass(frozen=True)
class _VBDPreservedInputPoseProjectionBinding:
    """Stable resolved arrays and output state used during graph capture."""

    name: str
    entry_name: str
    callback: Callable[[State], None] = field(repr=False, compare=False)
    parent_global_body_ids: wp.array = field(repr=False, compare=False)
    entry_local_body_ids: wp.array = field(repr=False, compare=False)
    entry_output_body_q: wp.array = field(repr=False, compare=False)


@dataclass(frozen=True)
class _VBDPreservedInputPoseBuffers:
    """Stable scratch arrays used by the wrapped VBD input notification."""

    entry_local_body_ids: wp.array
    saved_body_q: wp.array
    saved_body_qd: wp.array


@dataclass(frozen=True)
class _VBDPoseHistorySelectionLayout:
    """Bounded host cache for immutable coupled-view ownership topology."""

    storage_key: tuple[int, ...]
    owned_body_ids: frozenset[int]
    global_to_local: Any
    parent_body_world: Any
    entry_body_world: Any


@dataclass
class _VBDPoseHistoryRestoreBuffers:
    """Stable device storage consumed from the captured VBD step graph."""

    body_pending: wp.array
    world_pending: wp.array
    preserve_input_mask: wp.array
    body_q_prev: wp.array
    coupling_body_q_prev: wp.array
    queued_generation: wp.array
    applied_generation: wp.array
    failed_generation: wp.array
    application_count: wp.array
    body_application_count: wp.array
    preserve_input_active: bool = False
    next_generation: int = 0
    status_issuer: object = field(default_factory=object)
    selection_layout: _VBDPoseHistorySelectionLayout | None = None


class NewtonCouplerManager(NewtonVBDManager):
    """Couple named Newton solver entries through proxy or ADMM interfaces."""

    _vbd_preserved_input_pose_projection_registrations: dict[int, _VBDPreservedInputPoseProjectionRegistration] = {}
    _vbd_preserved_input_pose_projection_names: dict[str, int] = {}
    _vbd_preserved_input_pose_projection_bindings: tuple[_VBDPreservedInputPoseProjectionBinding, ...] = ()
    _vbd_preserved_input_pose_projection_next_id: int = 0
    _vbd_preserved_input_pose_projection_issuer: object = object()

    @dataclass
    class _ResolvedEntry:
        """Entry configuration with model selectors resolved to indices."""

        config: CouplerEntryCfg
        bodies: list[int]
        particles: list[int]
        joints: list[int]
        shapes: list[int]

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: CouplerCfg):
        """Reject recursive use as a nested coupled-solver entry."""
        del model, solver_cfg
        raise NotImplementedError("Nested Newton couplers are not supported.")

    @staticmethod
    def _requires_external_contacts(solver_cfg: NewtonSolverCfg) -> bool:
        """Return whether a sub-solver expects contacts from Newton's collision pipeline.

        Unknown solver configs conservatively opt in to external contacts.
        """
        if isinstance(solver_cfg, MJWarpSolverCfg):
            return not solver_cfg.use_mujoco_contacts
        if isinstance(solver_cfg, (KaminoPADMMSolverCfg, KaminoDVISolverCfg)):
            return not solver_cfg.use_collision_detector
        if isinstance(solver_cfg, MPMSolverCfg):
            return False
        return True

    @classmethod
    def get_proxy_contact_data(
        cls,
        source: str,
        destination: str,
    ) -> tuple[Contacts | None, ModelView, State]:
        """Return one proxy interface's contacts and matching destination-local layout.

        Proxy collision pipelines operate on the destination entry's
        :class:`~newton.solvers.experimental.coupled.ModelView`. Consequently,
        shape/body indices in their contact buffers must be interpreted against
        that view and its entry-local state, not the parent Newton model/state.

        Args:
            source: Name of the proxy source entry.
            destination: Name of the proxy destination entry.

        Returns:
            The proxy-local contact buffer (or ``None`` when the mapping uses
            outer contacts), destination model view, and its current state.

        Raises:
            RuntimeError: If the active Newton solver is not a proxy coupler.
            KeyError: If either entry name is not present in the active coupler.
        """
        solver = NewtonManager._solver
        if not isinstance(solver, SolverCoupledProxy):
            raise RuntimeError("Proxy contact data requires an active SolverCoupledProxy.")
        entry_names = solver.entry_names()
        for role, name in (("source", source), ("destination", destination)):
            if name not in entry_names:
                raise KeyError(f"Unknown proxy {role} entry {name!r}; available entries are {entry_names}.")
        return (
            solver.get_proxy_contacts(source, destination),
            solver.view(destination),
            solver.entry_state(destination, phase="current"),
        )

    @classmethod
    def register_vbd_preserved_input_pose_projection(
        cls,
        *,
        name: str,
        entry_name: str,
        body_ids: wp.array,
        callback: Callable[[State], None],
    ) -> VBDPreservedInputPoseProjectionHandle:
        """Register a graph-safe post-solver pose projection for named VBD bodies.

        Registration is intentionally deferred. Call this from
        ``PhysicsEvent.PHYSICS_READY`` after the parent model and body ids exist
        but before solver initialization. The named entry and its parent-to-
        local ownership mapping are resolved only after the coupled solver has
        been constructed and before CUDA graph capture.

        On every solver substep, ``callback`` receives the parent output state
        immediately after the coupled solve and before state-buffer swapping or
        mid-loop collision detection. It may author ``body_q`` only for the
        registered ids and must use fixed-shape, graph-safe operations. The
        projected poses are then copied into the named entry's output state.
        During the next coupled input distribution, VBD accepts those poses as
        direct solver inputs without converting their pose delta into velocity
        or rewinding them to its previous-pose history. Their input ``body_qd``
        and both VBD histories remain unchanged.

        Args:
            name: Unique lifecycle name for this projection.
            entry_name: Coupled solver entry that owns every selected body and
                is backed by rigid-integrating :class:`newton.SolverVBD`.
            body_ids: Parent-model body ids, shape ``[N]``, ``wp.int32``, on
                the finalized model device. Id order is preserved.
            callback: Graph-safe q-only projector invoked on the parent output
                state once per solver substep.

        Returns:
            A tolerant lifecycle handle. Deregistration prevents binding on a
            future rebuild; an already captured binding remains active until
            the current solver is cleared or rebuilt.

        Raises:
            RuntimeError: If no finalized model exists or solver construction
                has already started for that model.
            TypeError: If names, callback, or index storage are incompatible.
            ValueError: If ids are empty, duplicated, out of range, or the
                lifecycle name is already registered.
        """
        del cls
        manager = NewtonCouplerManager
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if not name:
            raise ValueError("name must be a non-empty string.")
        if not isinstance(entry_name, str):
            raise TypeError("entry_name must be a string.")
        if not entry_name:
            raise ValueError("entry_name must be a non-empty string.")
        if not callable(callback):
            raise TypeError("callback must be callable.")

        model = NewtonManager._model
        if model is None:
            raise RuntimeError("Register the VBD input-pose projection after the parent model is finalized.")
        active_solver = NewtonManager._solver
        if active_solver is not None and getattr(active_solver, "model", None) is model:
            raise RuntimeError("Register the VBD input-pose projection before coupled solver initialization.")
        manager._validate_vbd_index_array(body_ids, name="body_ids", device=model.device)
        if body_ids.shape[0] == 0:
            raise ValueError("body_ids must select at least one body.")
        body_ids_array = body_ids.numpy().astype("int64", copy=False)
        if len(set(body_ids_array.tolist())) != body_ids.shape[0]:
            raise ValueError("body_ids must not contain duplicates.")
        if ((body_ids_array < 0) | (body_ids_array >= int(model.body_count))).any():
            raise ValueError(f"body_ids must lie in [0, {int(model.body_count)}).")
        if name in manager._vbd_preserved_input_pose_projection_names:
            raise ValueError(f"A VBD input-pose projection named {name!r} is already registered.")
        selected_body_ids = set(int(value) for value in body_ids_array)
        for registered in manager._vbd_preserved_input_pose_projection_registrations.values():
            if registered.model is not model:
                continue
            overlaps = sorted(selected_body_ids.intersection(registered.body_ids))
            if overlaps:
                raise ValueError(
                    f"VBD input-pose projection {name!r} overlaps projection {registered.name!r} "
                    f"on parent bodies {overlaps}."
                )

        registration_id = manager._vbd_preserved_input_pose_projection_next_id + 1
        if registration_id > 2_147_483_647:
            raise RuntimeError("VBD input-pose projection registration ids are exhausted.")
        registration = _VBDPreservedInputPoseProjectionRegistration(
            registration_id=registration_id,
            name=name,
            entry_name=entry_name,
            body_ids=tuple(int(value) for value in body_ids_array),
            callback=callback,
            model=model,
        )
        manager._vbd_preserved_input_pose_projection_registrations[registration_id] = registration
        manager._vbd_preserved_input_pose_projection_names[name] = registration_id
        manager._vbd_preserved_input_pose_projection_next_id = registration_id
        return VBDPreservedInputPoseProjectionHandle(
            name=name,
            _registration_id=registration_id,
            _issuer=manager._vbd_preserved_input_pose_projection_issuer,
        )

    @staticmethod
    def _deregister_vbd_preserved_input_pose_projection(
        handle: VBDPreservedInputPoseProjectionHandle,
    ) -> bool:
        """Remove a valid handle from the deferred-registration registry."""
        manager = NewtonCouplerManager
        if not isinstance(handle, VBDPreservedInputPoseProjectionHandle):
            raise TypeError("handle must be a VBDPreservedInputPoseProjectionHandle.")
        if handle._issuer is not manager._vbd_preserved_input_pose_projection_issuer:
            return False
        registration = manager._vbd_preserved_input_pose_projection_registrations.pop(handle._registration_id, None)
        if registration is None:
            return False
        if registration.name != handle.name:
            raise RuntimeError("VBD input-pose projection handle registry is inconsistent.")
        manager._vbd_preserved_input_pose_projection_names.pop(registration.name, None)
        return True

    @classmethod
    def capture_vbd_pose_history(
        cls,
        entry_name: str,
        body_ids: wp.array,
        world_ids: wp.array,
    ) -> tuple[wp.array, wp.array]:
        """Capture selected rigid-pose histories from a named VBD entry.

        The body identifiers use the parent Newton model's global body
        namespace. Returned transforms follow ``body_ids`` exactly, even when
        the coupled entry uses a compact entry-local model view. Both arrays
        are required for faithful replay because VBD's current previous-pose
        history and its coupled-iteration restart snapshot are not generally
        identical.

        Args:
            entry_name: Name of the coupled solver entry containing VBD.
            body_ids: Parent-model body ids, shape ``[N]``, ``wp.int32``.
                Every body must be owned by ``entry_name``.
            world_ids: Unique parent-model world ids represented by
                ``body_ids``, shape ``[W]``, ``wp.int32``.

        Returns:
            A pair containing VBD previous body poses [m] and coupled-restart
            previous body poses [m]. Each is a ``wp.transform`` array of shape
            ``[N]`` on the active model device and follows ``body_ids`` order.

        Raises:
            RuntimeError: If the active solver is not a proxy/ADMM coupler, the
                named entry is not a rigid-integrating VBD solver, or required
                VBD history storage is unavailable.
            KeyError: If ``entry_name`` is not present in the active coupler.
            TypeError: If an input is not a one-dimensional ``wp.int32`` array
                on the active model device.
            ValueError: If ids are empty, duplicated, out of range, unowned, or
                do not describe exactly the same selected worlds.
        """
        vbd_solver, entry_local_body_ids, _, _, _, _ = cls._resolve_vbd_pose_history_selection(
            entry_name,
            body_ids,
            world_ids,
        )
        body_q_prev = wp.empty(body_ids.shape[0], dtype=wp.transform, device=body_ids.device)
        coupling_body_q_prev = wp.empty_like(body_q_prev)
        wp.launch(
            _gather_vbd_pose_history,
            dim=body_ids.shape[0],
            inputs=[
                entry_local_body_ids,
                vbd_solver.body_q_prev,
                vbd_solver._coupling_body_q_prev_snapshot,
                body_q_prev,
                coupling_body_q_prev,
            ],
            device=body_ids.device,
        )
        return body_q_prev, coupling_body_q_prev

    @classmethod
    def queue_vbd_pose_history_restore(
        cls,
        entry_name: str,
        body_ids: wp.array,
        world_ids: wp.array,
        body_q_prev: wp.array,
        coupling_body_q_prev: wp.array,
    ) -> VBDPoseHistoryRestoreStatus:
        """Queue selected histories for one deferred named-VBD restore.

        Call this after the normal public reset/forward path and before the
        selected worlds' first coupled solve. The restore is staged in stable
        device buffers and consumed exactly once from the captured solver graph.
        At the first VBD solve boundary, normal input distribution and proxy
        synchronization first establish histories for unsaved bodies. Only then
        are the selected histories scattered and those worlds' pending VBD
        rebaseline flags consumed. A world may have only one pending request;
        callers must let it solve or explicitly fail rather than silently
        replacing serialized history.

        Args:
            entry_name: Name of the coupled solver entry containing VBD.
            body_ids: Parent-model body ids, shape ``[N]``, ``wp.int32``.
                Array order defines the order of both history inputs.
            world_ids: Unique parent-model world ids represented by
                ``body_ids``, shape ``[W]``, ``wp.int32``.
            body_q_prev: VBD previous body poses [m], shape ``[N]``,
                ``wp.transform``.
            coupling_body_q_prev: Coupled-restart previous body poses [m],
                shape ``[N]``, ``wp.transform``.

        Returns:
            Pending restore status containing the accepted body/world order and
            a generation token for later status queries. Pass this object to
            :meth:`get_vbd_pose_history_restore_status` after the first solve.

        Raises:
            RuntimeError: If the active solver is not a proxy/ADMM coupler, the
                named entry is not a rigid-integrating VBD solver, or required
                VBD history storage is unavailable, a requested world does not
                await rebaselining, or that world already has a queued restore.
            KeyError: If ``entry_name`` is not present in the active coupler.
            TypeError: If an input has an incompatible array type, dtype, rank,
                or device.
            ValueError: If ids, worlds, or history shapes are incompatible.
        """
        (
            vbd_solver,
            entry_local_body_ids,
            pending,
            body_ids_host,
            world_ids_host,
            selected_body_worlds,
        ) = cls._resolve_vbd_pose_history_selection(
            entry_name,
            body_ids,
            world_ids,
        )
        cls._validate_vbd_pose_history_array(
            body_q_prev,
            name="body_q_prev",
            length=body_ids.shape[0],
            device=body_ids.device,
        )
        cls._validate_vbd_pose_history_array(
            coupling_body_q_prev,
            name="coupling_body_q_prev",
            length=body_ids.shape[0],
            device=body_ids.device,
        )

        pending_worlds_host = pending.world_pending.numpy()
        conflicts = tuple(world for world in world_ids_host if bool(pending_worlds_host[world]))
        if conflicts:
            raise RuntimeError(f"Coupled VBD entry {entry_name!r} already has pending restores for worlds {conflicts}.")
        rebaseline_host = vbd_solver._rigid_pose_rebaseline_mask.numpy()
        unavailable = tuple(world for world in world_ids_host if not bool(rebaseline_host[world]))
        if unavailable:
            raise RuntimeError(
                f"Coupled VBD entry {entry_name!r} worlds {unavailable} do not await pose rebaselining; "
                "queue immediately after their normal reset/forward path."
            )

        generation = pending.next_generation + 1
        if generation > 2_147_483_647:
            raise RuntimeError(f"Coupled VBD entry {entry_name!r} exhausted restore generation ids.")
        application_count = pending.application_count.numpy()
        body_application_count = pending.body_application_count.numpy()
        application_counts_before = tuple(int(application_count[world]) for world in world_ids_host)
        body_application_counts_before = tuple(int(body_application_count[world]) for world in world_ids_host)
        expected_body_counts = tuple(selected_body_worlds.count(world) for world in world_ids_host)

        wp.launch(
            _queue_vbd_pose_history_bodies,
            dim=body_ids.shape[0],
            inputs=[
                entry_local_body_ids,
                body_q_prev,
                coupling_body_q_prev,
                pending.body_q_prev,
                pending.coupling_body_q_prev,
                pending.body_pending,
            ],
            device=body_ids.device,
        )
        wp.launch(
            _queue_vbd_pose_history_worlds,
            dim=world_ids.shape[0],
            inputs=[
                world_ids,
                generation,
                pending.world_pending,
                pending.queued_generation,
            ],
            device=body_ids.device,
        )
        pending.next_generation = generation
        return VBDPoseHistoryRestoreStatus(
            entry_name=entry_name,
            generation=generation,
            body_ids=body_ids_host,
            world_ids=world_ids_host,
            expected_body_counts=expected_body_counts,
            pending_world_ids=world_ids_host,
            applied_world_ids=(),
            failed_world_ids=(),
            superseded_world_ids=(),
            application_count_deltas=tuple(0 for _ in world_ids_host),
            body_application_count_deltas=tuple(0 for _ in world_ids_host),
            _application_counts_before=application_counts_before,
            _body_application_counts_before=body_application_counts_before,
            _issuer=pending.status_issuer,
        )

    @classmethod
    def restore_vbd_pose_history(
        cls,
        entry_name: str,
        body_ids: wp.array,
        world_ids: wp.array,
        body_q_prev: wp.array,
        coupling_body_q_prev: wp.array,
    ) -> VBDPoseHistoryRestoreStatus:
        """Queue a graph-safe one-shot VBD pose-history restore.

        This compatibility spelling has the same deferred semantics as
        :meth:`queue_vbd_pose_history_restore`; it never mutates active VBD
        histories or clears rebaseline flags eagerly.
        """
        return cls.queue_vbd_pose_history_restore(
            entry_name,
            body_ids,
            world_ids,
            body_q_prev,
            coupling_body_q_prev,
        )

    @classmethod
    def get_vbd_pose_history_restore_status(
        cls,
        request: VBDPoseHistoryRestoreStatus,
    ) -> VBDPoseHistoryRestoreStatus:
        """Return public ordering and one-shot evidence for a queued restore.

        Args:
            request: Status ticket returned by
                :meth:`queue_vbd_pose_history_restore` or
                :meth:`restore_vbd_pose_history`.

        Returns:
            Current device-observed status for that restore generation.
            A status queried after a newer request has reused one of its worlds
            reports that world as superseded. Retain a completed status if
            longer-lived audit evidence is required.

        Raises:
            RuntimeError: If the active solver or entry cannot service deferred
                VBD history restores.
            TypeError: If ``request`` is not a restore status ticket.
            ValueError: If the ticket metadata is internally inconsistent.
        """
        if not isinstance(request, VBDPoseHistoryRestoreStatus):
            raise TypeError("request must be a VBDPoseHistoryRestoreStatus returned by the restore API.")
        if request.generation < 1 or request.generation > 2_147_483_647:
            raise ValueError("request has an invalid restore generation.")
        if (
            not request.world_ids
            or len(request.world_ids) != len(request._application_counts_before)
            or len(request.world_ids) != len(request._body_application_counts_before)
            or len(request.world_ids) != len(request.expected_body_counts)
            or len(set(request.world_ids)) != len(request.world_ids)
            or any(count < 1 for count in request.expected_body_counts)
            or sum(request.expected_body_counts) != len(request.body_ids)
        ):
            raise ValueError("request has inconsistent world metadata.")
        _, pending = cls._resolve_vbd_pose_history_entry(request.entry_name)
        if request._issuer is not pending.status_issuer:
            raise ValueError(f"Restore status was not issued for active entry {request.entry_name!r}.")
        if request.generation > pending.next_generation:
            raise ValueError(
                f"Restore generation {request.generation} was not issued for entry {request.entry_name!r}."
            )

        world_pending = pending.world_pending.numpy()
        queued_generation = pending.queued_generation.numpy()
        applied_generation = pending.applied_generation.numpy()
        failed_generation = pending.failed_generation.numpy()
        application_count = pending.application_count.numpy()
        body_application_count = pending.body_application_count.numpy()
        world_slot_count = pending.world_pending.shape[0]
        if any(world < 0 or world >= world_slot_count - 1 for world in request.world_ids):
            raise ValueError("request contains an out-of-range world id.")
        pending_world_ids = tuple(
            world
            for world in request.world_ids
            if bool(world_pending[world]) and int(queued_generation[world]) == request.generation
        )
        applied_world_ids = tuple(
            world for world in request.world_ids if int(applied_generation[world]) == request.generation
        )
        failed_world_ids = tuple(
            world for world in request.world_ids if int(failed_generation[world]) == request.generation
        )
        observed_worlds = set(pending_world_ids) | set(applied_world_ids) | set(failed_world_ids)
        superseded_world_ids = tuple(world for world in request.world_ids if world not in observed_worlds)
        application_count_deltas = tuple(
            int(application_count[world]) - before
            for world, before in zip(request.world_ids, request._application_counts_before)
        )
        body_application_count_deltas = tuple(
            int(body_application_count[world]) - before
            for world, before in zip(request.world_ids, request._body_application_counts_before)
        )
        return VBDPoseHistoryRestoreStatus(
            entry_name=request.entry_name,
            generation=request.generation,
            body_ids=request.body_ids,
            world_ids=request.world_ids,
            expected_body_counts=request.expected_body_counts,
            pending_world_ids=pending_world_ids,
            applied_world_ids=applied_world_ids,
            failed_world_ids=failed_world_ids,
            superseded_world_ids=superseded_world_ids,
            application_count_deltas=application_count_deltas,
            body_application_count_deltas=body_application_count_deltas,
            _application_counts_before=request._application_counts_before,
            _body_application_counts_before=request._body_application_counts_before,
            _issuer=request._issuer,
        )

    @classmethod
    def _resolve_vbd_pose_history_selection(
        cls,
        entry_name: str,
        body_ids: wp.array,
        world_ids: wp.array,
    ) -> tuple[
        SolverVBD,
        wp.array,
        _VBDPoseHistoryRestoreBuffers,
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]:
        """Validate a public history selection and return its entry-local ids."""
        vbd_solver, pending = cls._resolve_vbd_pose_history_entry(entry_name)
        solver = NewtonManager._solver
        model = solver.model
        device = model.device
        cls._validate_vbd_index_array(body_ids, name="body_ids", device=device)
        cls._validate_vbd_index_array(world_ids, name="world_ids", device=device)
        if body_ids.shape[0] == 0:
            raise ValueError("body_ids must select at least one body.")
        if world_ids.shape[0] == 0:
            raise ValueError("world_ids must select at least one world.")

        body_ids_host = body_ids.numpy().astype("int64", copy=False)
        world_ids_host = world_ids.numpy().astype("int64", copy=False)
        if len(set(body_ids_host.tolist())) != body_ids.shape[0]:
            raise ValueError("body_ids must not contain duplicates.")
        if len(set(world_ids_host.tolist())) != world_ids.shape[0]:
            raise ValueError("world_ids must not contain duplicates.")
        if ((body_ids_host < 0) | (body_ids_host >= int(model.body_count))).any():
            raise ValueError(f"body_ids must lie in [0, {int(model.body_count)}).")
        if ((world_ids_host < 0) | (world_ids_host >= int(model.world_count))).any():
            raise ValueError(f"world_ids must lie in [0, {int(model.world_count)}).")

        entries = getattr(solver, "_entries", None)
        entry: Any = entries.get(entry_name) if isinstance(entries, dict) else None
        if entry is None:
            raise RuntimeError(f"Active coupler has no runtime mapping for entry {entry_name!r}.")
        cls._validate_vbd_internal_index_array(
            entry.body_indices,
            name=f"entry {entry_name!r} owned body ids",
            device=device,
        )
        cls._validate_vbd_internal_index_array(
            entry.body_global_to_local,
            name=f"entry {entry_name!r} global-to-local body map",
            device=device,
            length=int(model.body_count),
        )
        cls._validate_vbd_internal_index_array(
            model.body_world,
            name="model body-world map",
            device=device,
            length=int(model.body_count),
        )
        cls._validate_vbd_internal_index_array(
            vbd_solver.model.body_world,
            name=f"entry {entry_name!r} VBD body-world map",
            device=device,
            length=int(vbd_solver.model.body_count),
        )
        layout = cls._vbd_pose_history_selection_layout(entry_name, model, entry, vbd_solver, pending)
        unowned_body_ids = [int(value) for value in body_ids_host if int(value) not in layout.owned_body_ids]
        if unowned_body_ids:
            raise ValueError(f"body_ids contains bodies not owned by coupled entry {entry_name!r}: {unowned_body_ids}.")
        global_to_local = layout.global_to_local
        entry_local_body_ids_host = global_to_local[body_ids_host].astype("int32", copy=False)
        if (
            (entry_local_body_ids_host < 0).any()
            or (entry_local_body_ids_host >= int(vbd_solver.model.body_count)).any()
            or len(set(entry_local_body_ids_host.tolist())) != body_ids.shape[0]
        ):
            raise RuntimeError(f"Coupled entry {entry_name!r} has an invalid body ownership mapping.")

        selected_body_worlds = layout.parent_body_world[body_ids_host].astype("int64", copy=False)
        entry_local_body_worlds = layout.entry_body_world[entry_local_body_ids_host]
        if not (entry_local_body_worlds == selected_body_worlds).all():
            raise RuntimeError(f"Coupled entry {entry_name!r} has an inconsistent entry-local body-world mapping.")
        represented_worlds = set(selected_body_worlds.tolist())
        requested_worlds = set(world_ids_host.tolist())
        if represented_worlds != requested_worlds:
            raise ValueError(
                "world_ids must exactly match the worlds represented by body_ids; "
                f"represented={sorted(represented_worlds)}, requested={sorted(requested_worlds)}."
            )

        cls._validate_vbd_pose_history_array(
            getattr(vbd_solver, "body_q_prev", None),
            name=f"entry {entry_name!r} VBD body history",
            length=int(vbd_solver.model.body_count),
            device=device,
            internal=True,
        )
        cls._validate_vbd_pose_history_array(
            getattr(vbd_solver, "_coupling_body_q_prev_snapshot", None),
            name=f"entry {entry_name!r} VBD coupling history",
            length=int(vbd_solver.model.body_count),
            device=device,
            internal=True,
        )
        rebaseline_mask = getattr(vbd_solver, "_rigid_pose_rebaseline_mask", None)
        cls._validate_vbd_bool_array(
            rebaseline_mask,
            name=f"entry {entry_name!r} VBD rebaseline mask",
            length=int(model.world_count) + 1,
            device=device,
        )
        return (
            vbd_solver,
            wp.array(entry_local_body_ids_host, dtype=wp.int32, device=device),
            pending,
            tuple(int(value) for value in body_ids_host),
            tuple(int(value) for value in world_ids_host),
            tuple(int(value) for value in selected_body_worlds),
        )

    @staticmethod
    def _vbd_pose_history_selection_layout(
        entry_name: str,
        model: Model,
        entry: Any,
        vbd_solver: SolverVBD,
        pending: _VBDPoseHistoryRestoreBuffers,
    ) -> _VBDPoseHistorySelectionLayout:
        """Cache immutable ownership maps without retaining restore requests."""
        storage_key = (
            int(entry.body_indices.ptr),
            int(entry.body_global_to_local.ptr),
            int(model.body_world.ptr),
            int(vbd_solver.model.body_world.ptr),
            int(model.body_count),
            int(model.world_count),
            int(vbd_solver.model.body_count),
            int(vbd_solver.model.world_count),
        )
        layout = pending.selection_layout
        if layout is not None:
            if layout.storage_key != storage_key:
                raise RuntimeError(f"Coupled entry {entry_name!r} changed its VBD ownership storage.")
            return layout

        owned_body_ids_array = entry.body_indices.numpy().astype("int64", copy=False)
        if ((owned_body_ids_array < 0) | (owned_body_ids_array >= int(model.body_count))).any() or len(
            set(owned_body_ids_array.tolist())
        ) != owned_body_ids_array.shape[0]:
            raise RuntimeError(f"Coupled entry {entry_name!r} has invalid owned body ids.")
        layout = _VBDPoseHistorySelectionLayout(
            storage_key=storage_key,
            owned_body_ids=frozenset(int(value) for value in owned_body_ids_array),
            global_to_local=entry.body_global_to_local.numpy(),
            parent_body_world=model.body_world.numpy(),
            entry_body_world=vbd_solver.model.body_world.numpy(),
        )
        pending.selection_layout = layout
        return layout

    @classmethod
    def _resolve_vbd_pose_history_entry(
        cls,
        entry_name: str,
    ) -> tuple[SolverVBD, _VBDPoseHistoryRestoreBuffers]:
        """Resolve an active named rigid VBD entry and its deferred buffers."""
        solver = NewtonManager._solver
        if not isinstance(solver, (SolverCoupledProxy, SolverCoupledADMM)):
            raise RuntimeError("VBD pose history requires an active SolverCoupledProxy or SolverCoupledADMM.")
        if not isinstance(entry_name, str) or not entry_name:
            raise TypeError("entry_name must be a non-empty string.")
        entry_names = solver.entry_names()
        if entry_name not in entry_names:
            raise KeyError(f"Unknown coupled solver entry {entry_name!r}; available entries are {entry_names}.")

        vbd_solver = solver.solver(entry_name)
        if not isinstance(vbd_solver, SolverVBD):
            raise RuntimeError(f"Coupled solver entry {entry_name!r} uses {type(vbd_solver).__name__}, not SolverVBD.")
        if not getattr(vbd_solver, "_coupling_has_rigid_avbd_state", False):
            raise RuntimeError(f"Coupled VBD entry {entry_name!r} does not integrate rigid-body pose history.")
        pending = getattr(vbd_solver, "_isaaclab_vbd_pose_history_restore", None)
        if not isinstance(pending, _VBDPoseHistoryRestoreBuffers):
            raise RuntimeError(f"Coupled VBD entry {entry_name!r} has no deferred pose-history restore hook.")
        cls._validate_vbd_pose_history_restore_buffers(entry_name, vbd_solver, pending)
        return vbd_solver, pending

    @classmethod
    def _install_vbd_pose_history_restore_hook(cls, solver: SolverVBD) -> None:
        """Install stable one-shot restore nodes before this VBD solver's step."""
        if isinstance(getattr(solver, "_isaaclab_vbd_pose_history_restore", None), _VBDPoseHistoryRestoreBuffers):
            return
        if not getattr(solver, "_coupling_has_rigid_avbd_state", False):
            return

        body_count = int(solver.model.body_count)
        world_slot_count = int(solver.model.world_count) + 1
        device = solver.device
        pending = _VBDPoseHistoryRestoreBuffers(
            body_pending=wp.zeros(body_count, dtype=wp.bool, device=device),
            world_pending=wp.zeros(world_slot_count, dtype=wp.bool, device=device),
            preserve_input_mask=wp.zeros(body_count, dtype=wp.bool, device=device),
            body_q_prev=wp.empty(body_count, dtype=wp.transform, device=device),
            coupling_body_q_prev=wp.empty(body_count, dtype=wp.transform, device=device),
            queued_generation=wp.zeros(world_slot_count, dtype=wp.int32, device=device),
            applied_generation=wp.zeros(world_slot_count, dtype=wp.int32, device=device),
            failed_generation=wp.zeros(world_slot_count, dtype=wp.int32, device=device),
            application_count=wp.zeros(world_slot_count, dtype=wp.int32, device=device),
            body_application_count=wp.zeros(world_slot_count, dtype=wp.int32, device=device),
        )
        original_step = solver.step

        def step_with_deferred_pose_history(_solver, state_in, state_out, control, contacts, dt):
            cls._apply_pending_vbd_pose_history_restore(_solver, pending, state_in, dt)
            return original_step(state_in, state_out, control, contacts, dt)

        solver.step = MethodType(step_with_deferred_pose_history, solver)
        solver._isaaclab_vbd_pose_history_restore = pending

    @staticmethod
    def _apply_pending_vbd_pose_history_restore(
        solver: SolverVBD,
        pending: _VBDPoseHistoryRestoreBuffers,
        state_in: State,
        dt: float,
    ) -> None:
        """Record graph-safe restore and input-replay nodes at the VBD solve boundary."""
        device = solver.device
        kernel = _apply_pending_vbd_pose_history_bodies
        inputs = [
            float(dt),
            solver.model.body_flags,
            int(BodyFlags.KINEMATIC),
            solver.model.body_world,
            pending.body_pending,
            pending.world_pending,
            solver._rigid_pose_rebaseline_mask,
        ]
        if pending.preserve_input_active:
            kernel = _apply_pending_vbd_pose_history_bodies_preserved
            inputs.append(pending.preserve_input_mask)
        inputs.extend(
            [
                pending.body_q_prev,
                pending.coupling_body_q_prev,
                solver.body_q_prev,
                solver._coupling_body_q_prev_snapshot,
                state_in.body_q,
                state_in.body_qd,
                pending.body_application_count,
            ]
        )
        wp.launch(
            kernel,
            dim=int(solver.model.body_count),
            inputs=inputs,
            device=device,
        )
        wp.launch(
            _consume_pending_vbd_pose_history_worlds,
            dim=int(solver.model.world_count) + 1,
            inputs=[
                pending.world_pending,
                pending.queued_generation,
                solver._rigid_pose_rebaseline_mask,
                pending.applied_generation,
                pending.failed_generation,
                pending.application_count,
            ],
            device=device,
        )

    @staticmethod
    def _validate_vbd_index_array(value: object, *, name: str, device: wp.context.Device) -> None:
        """Validate a caller-owned one-dimensional index array."""
        if not isinstance(value, wp.array):
            raise TypeError(f"{name} must be a Warp array.")
        if value.dtype != wp.int32 or value.ndim != 1:
            raise TypeError(f"{name} must be a one-dimensional wp.int32 array.")
        if value.device != device:
            raise TypeError(f"{name} must be on model device {device}, got {value.device}.")

    @staticmethod
    def _validate_vbd_internal_index_array(
        value: object,
        *,
        name: str,
        device: wp.context.Device,
        length: int | None = None,
    ) -> None:
        """Fail closed when a coupled solver's internal index storage is incompatible."""
        try:
            NewtonCouplerManager._validate_vbd_index_array(value, name=name, device=device)
        except TypeError as error:
            raise RuntimeError(str(error)) from error
        if length is not None and value.shape != (length,):
            raise RuntimeError(f"{name} must have shape {(length,)}, got {value.shape}.")

    @staticmethod
    def _validate_vbd_pose_history_array(
        value: object,
        *,
        name: str,
        length: int,
        device: wp.context.Device,
        internal: bool = False,
    ) -> None:
        """Validate one public or solver-owned VBD pose-history array."""
        error_type = RuntimeError if internal else TypeError
        if not isinstance(value, wp.array):
            raise error_type(f"{name} must be a Warp array.")
        if value.dtype != wp.transform or value.ndim != 1:
            raise error_type(f"{name} must be a one-dimensional wp.transform array.")
        if value.device != device:
            raise error_type(f"{name} must be on model device {device}, got {value.device}.")
        if value.shape != (length,):
            shape_error_type = RuntimeError if internal else ValueError
            raise shape_error_type(f"{name} must have shape {(length,)}, got {value.shape}.")

    @staticmethod
    def _validate_vbd_bool_array(
        value: object,
        *,
        name: str,
        length: int,
        device: wp.context.Device,
    ) -> None:
        """Validate solver-owned per-world VBD mask storage."""
        if not isinstance(value, wp.array):
            raise RuntimeError(f"{name} must be a Warp array.")
        if value.dtype != wp.bool or value.ndim != 1 or value.device != device or value.shape != (length,):
            raise RuntimeError(f"{name} must be a one-dimensional wp.bool array of shape {(length,)} on {device}.")

    @classmethod
    def _validate_vbd_pose_history_restore_buffers(
        cls,
        entry_name: str,
        solver: SolverVBD,
        pending: _VBDPoseHistoryRestoreBuffers,
    ) -> None:
        """Fail closed if stable graph buffers no longer match their VBD view."""
        body_count = int(solver.model.body_count)
        world_slot_count = int(solver.model.world_count) + 1
        device = solver.device
        cls._validate_vbd_bool_array(
            pending.body_pending,
            name=f"entry {entry_name!r} pending-body mask",
            length=body_count,
            device=device,
        )
        cls._validate_vbd_bool_array(
            pending.world_pending,
            name=f"entry {entry_name!r} pending-world mask",
            length=world_slot_count,
            device=device,
        )
        cls._validate_vbd_bool_array(
            pending.preserve_input_mask,
            name=f"entry {entry_name!r} preserved-input mask",
            length=body_count,
            device=device,
        )
        cls._validate_vbd_pose_history_array(
            pending.body_q_prev,
            name=f"entry {entry_name!r} staged VBD body history",
            length=body_count,
            device=device,
            internal=True,
        )
        cls._validate_vbd_pose_history_array(
            pending.coupling_body_q_prev,
            name=f"entry {entry_name!r} staged VBD coupling history",
            length=body_count,
            device=device,
            internal=True,
        )
        for name, value in (
            ("queued generations", pending.queued_generation),
            ("applied generations", pending.applied_generation),
            ("failed generations", pending.failed_generation),
            ("application counts", pending.application_count),
            ("body application counts", pending.body_application_count),
        ):
            cls._validate_vbd_internal_index_array(
                value,
                name=f"entry {entry_name!r} {name}",
                device=device,
                length=world_slot_count,
            )
        if isinstance(pending.next_generation, bool) or not isinstance(pending.next_generation, int):
            raise RuntimeError(f"Coupled VBD entry {entry_name!r} has an invalid restore generation counter.")
        if not 0 <= pending.next_generation <= 2_147_483_647:
            raise RuntimeError(f"Coupled VBD entry {entry_name!r} restore generation counter is out of range.")
        if not isinstance(pending.preserve_input_active, bool):
            raise RuntimeError(f"Coupled VBD entry {entry_name!r} has an invalid preserved-input state.")
        if pending.selection_layout is not None and not isinstance(
            pending.selection_layout, _VBDPoseHistorySelectionLayout
        ):
            raise RuntimeError(f"Coupled VBD entry {entry_name!r} has an invalid ownership-layout cache.")

    @classmethod
    def _bind_vbd_preserved_input_pose_projections(cls, model: Model) -> None:
        """Resolve deferred projection registrations against the built coupler."""
        del cls
        manager = NewtonCouplerManager
        manager._vbd_preserved_input_pose_projection_bindings = ()
        registrations = tuple(manager._vbd_preserved_input_pose_projection_registrations.values())
        if not registrations:
            return

        coupled_solver = NewtonManager._solver
        if not isinstance(coupled_solver, (SolverCoupledProxy, SolverCoupledADMM)):
            raise RuntimeError("VBD input-pose projections require a proxy or ADMM coupled solver.")
        if coupled_solver.model is not model:
            raise RuntimeError("Active coupled solver does not own the finalized projection model.")

        bindings: list[_VBDPreservedInputPoseProjectionBinding] = []
        preserved_by_entry: dict[str, tuple[SolverVBD, _VBDPoseHistoryRestoreBuffers, list[int]]] = {}
        claimed_global_bodies: dict[int, str] = {}
        for registration in registrations:
            if registration.model is not model:
                raise RuntimeError(
                    f"VBD input-pose projection {registration.name!r} belongs to a stale parent model; "
                    "deregister it and register the rebuilt model's body ids."
                )

            overlaps = {
                body_id: claimed_global_bodies[body_id]
                for body_id in registration.body_ids
                if body_id in claimed_global_bodies
            }
            if overlaps:
                raise ValueError(
                    f"VBD input-pose projection {registration.name!r} overlaps bodies already claimed by "
                    f"other projections: {overlaps}."
                )

            vbd_solver, pending = manager._resolve_vbd_pose_history_entry(registration.entry_name)
            entries = getattr(coupled_solver, "_entries", None)
            entry: Any = entries.get(registration.entry_name) if isinstance(entries, dict) else None
            if entry is None:
                raise RuntimeError(f"Active coupler has no runtime mapping for entry {registration.entry_name!r}.")
            manager._validate_vbd_internal_index_array(
                entry.body_indices,
                name=f"entry {registration.entry_name!r} owned body ids",
                device=model.device,
            )
            manager._validate_vbd_internal_index_array(
                entry.body_global_to_local,
                name=f"entry {registration.entry_name!r} global-to-local body map",
                device=model.device,
                length=int(model.body_count),
            )
            manager._validate_vbd_internal_index_array(
                model.body_world,
                name="model body-world map",
                device=model.device,
                length=int(model.body_count),
            )
            manager._validate_vbd_internal_index_array(
                vbd_solver.model.body_world,
                name=f"entry {registration.entry_name!r} VBD body-world map",
                device=model.device,
                length=int(vbd_solver.model.body_count),
            )
            layout = manager._vbd_pose_history_selection_layout(
                registration.entry_name, model, entry, vbd_solver, pending
            )
            unowned_body_ids = [body_id for body_id in registration.body_ids if body_id not in layout.owned_body_ids]
            if unowned_body_ids:
                raise ValueError(
                    f"VBD input-pose projection {registration.name!r} contains bodies not owned by coupled "
                    f"entry {registration.entry_name!r}: {unowned_body_ids}."
                )
            entry_local_body_ids_host = layout.global_to_local[list(registration.body_ids)].astype("int32", copy=False)
            if (
                (entry_local_body_ids_host < 0).any()
                or (entry_local_body_ids_host >= int(vbd_solver.model.body_count)).any()
                or len(set(entry_local_body_ids_host.tolist())) != len(registration.body_ids)
            ):
                raise RuntimeError(f"Coupled entry {registration.entry_name!r} has an invalid body ownership mapping.")
            parent_worlds = layout.parent_body_world[list(registration.body_ids)]
            entry_worlds = layout.entry_body_world[entry_local_body_ids_host]
            if not (parent_worlds == entry_worlds).all():
                raise RuntimeError(f"Coupled entry {registration.entry_name!r} has an inconsistent body-world mapping.")

            entry_output_state = coupled_solver.entry_state(registration.entry_name, phase="output")
            manager._validate_vbd_pose_history_array(
                getattr(entry_output_state, "body_q", None),
                name=f"entry {registration.entry_name!r} output body poses",
                length=int(vbd_solver.model.body_count),
                device=model.device,
                internal=True,
            )
            parent_global_body_ids = wp.array(registration.body_ids, dtype=wp.int32, device=model.device)
            entry_local_body_ids = wp.array(entry_local_body_ids_host, dtype=wp.int32, device=model.device)
            bindings.append(
                _VBDPreservedInputPoseProjectionBinding(
                    name=registration.name,
                    entry_name=registration.entry_name,
                    callback=registration.callback,
                    parent_global_body_ids=parent_global_body_ids,
                    entry_local_body_ids=entry_local_body_ids,
                    entry_output_body_q=entry_output_state.body_q,
                )
            )
            preserved_entry = preserved_by_entry.setdefault(registration.entry_name, (vbd_solver, pending, []))
            preserved_entry[2].extend(int(value) for value in entry_local_body_ids_host)
            for body_id in registration.body_ids:
                claimed_global_bodies[body_id] = registration.name

        for entry_name, (vbd_solver, pending, entry_local_body_ids) in preserved_by_entry.items():
            manager._install_vbd_preserved_input_pose_hook(
                entry_name,
                vbd_solver,
                pending,
                tuple(entry_local_body_ids),
            )
        manager._vbd_preserved_input_pose_projection_bindings = tuple(bindings)

    @classmethod
    def _install_vbd_preserved_input_pose_hook(
        cls,
        entry_name: str,
        solver: SolverVBD,
        pending: _VBDPoseHistoryRestoreBuffers,
        entry_local_body_ids_host: tuple[int, ...],
    ) -> None:
        """Wrap one VBD input notification with exact selected-row preservation."""
        del cls
        existing = getattr(solver, "_isaaclab_vbd_preserved_input_pose", None)
        if existing is not None:
            raise RuntimeError(f"Coupled VBD entry {entry_name!r} already has a preserved-input hook.")
        if not entry_local_body_ids_host:
            raise RuntimeError(f"Coupled VBD entry {entry_name!r} has an empty preserved-input selection.")

        device = solver.device
        entry_local_body_ids = wp.array(entry_local_body_ids_host, dtype=wp.int32, device=device)
        buffers = _VBDPreservedInputPoseBuffers(
            entry_local_body_ids=entry_local_body_ids,
            saved_body_q=wp.empty(len(entry_local_body_ids_host), dtype=wp.transform, device=device),
            saved_body_qd=wp.empty(len(entry_local_body_ids_host), dtype=wp.spatial_vector, device=device),
        )
        wp.launch(
            _mark_vbd_preserved_input_pose_bodies,
            dim=len(entry_local_body_ids_host),
            inputs=[entry_local_body_ids, pending.preserve_input_mask],
            device=device,
        )
        pending.preserve_input_active = True
        original_notify = solver.coupling_notify_input_state_update

        def notify_with_preserved_input_pose(
            _solver,
            state,
            flags,
            *,
            iteration_restart=False,
            dt=0.0,
        ):
            flags_int = int(flags)
            should_preserve = (
                float(dt) > 0.0
                and bool(flags_int & int(StateFlags.BODY_Q))
                and state.body_q is not None
                and state.body_qd is not None
                and bool(getattr(_solver, "_coupling_has_rigid_avbd_state", False))
            )
            if not should_preserve:
                original_notify(state, flags, iteration_restart=iteration_restart, dt=dt)
                return

            neutral_body_q_prev = _solver._coupling_body_q_prev_snapshot if iteration_restart else _solver.body_q_prev
            wp.launch(
                _save_and_neutralize_vbd_preserved_input_pose,
                dim=entry_local_body_ids.shape[0],
                inputs=[
                    entry_local_body_ids,
                    _solver.model.body_world,
                    _solver._rigid_pose_rebaseline_mask,
                    neutral_body_q_prev,
                    state.body_q,
                    state.body_qd,
                    buffers.saved_body_q,
                    buffers.saved_body_qd,
                ],
                device=_solver.device,
            )
            original_notify(state, flags, iteration_restart=iteration_restart, dt=dt)
            wp.launch(
                _restore_vbd_preserved_input_pose,
                dim=entry_local_body_ids.shape[0],
                inputs=[
                    entry_local_body_ids,
                    buffers.saved_body_q,
                    buffers.saved_body_qd,
                    state.body_q,
                    state.body_qd,
                ],
                device=_solver.device,
            )
            return

        solver.coupling_notify_input_state_update = MethodType(notify_with_preserved_input_pose, solver)
        solver._isaaclab_vbd_preserved_input_pose = buffers

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: CouplerCfg) -> None:
        """Resolve ownership and construct the selected coupled solver."""
        if NewtonManager._report_contacts:
            raise NotImplementedError(
                "Newton contact sensors are not yet supported by coupled solvers because contact forces live "
                "in per-entry buffers. Remove the contact sensor."
            )

        cls._validate_config(solver_cfg)
        resolved_entries = [cls._resolve_entry(model, entry) for entry in solver_cfg.entries]
        proxies: list[CouplerProxyMappingCfg] = []
        active_proxy_destinations: set[str] = set()
        if isinstance(solver_cfg, CouplerProxyCfg):
            proxies = [cls._resolve_proxy(model, proxy) for proxy in solver_cfg.proxies]
            active_proxy_destinations = {proxy.destination for proxy in proxies if proxy.bodies or proxy.particles}
        cls._validate_resolved_entries(model, resolved_entries, solver_cfg, active_proxy_destinations)
        entries = [cls._build_entry(entry) for entry in resolved_entries]

        if isinstance(solver_cfg, CouplerProxyCfg):
            solver = cls._build_proxy_coupled_solver(model, entries, proxies, solver_cfg)
            directions = {(proxy.source, proxy.destination) for proxy in proxies}
            proxy_destinations = {destination for _, destination in directions}
            outer_contact_entries = {
                entry.config.name for entry in resolved_entries if entry.config.name not in proxy_destinations
            }
            outer_contact_entries.update(source for source, _ in directions)
            outer_contact_entries.update(
                destination
                for source, destination in directions
                if solver.get_proxy_contacts(source, destination) is None
            )
            NewtonManager._solver = solver
            needs_collision_pipeline = any(
                entry.config.name in outer_contact_entries and cls._requires_external_contacts(entry.config.solver_cfg)
                for entry in resolved_entries
            )
        else:
            NewtonManager._solver = cls._build_admm_coupled_solver(model, entries, solver_cfg)
            needs_collision_pipeline = True

        cls._bind_vbd_preserved_input_pose_projections(model)
        NewtonManager._use_single_state = False
        NewtonManager._supports_contact_sensors = False
        NewtonManager._needs_collision_pipeline = needs_collision_pipeline
        NewtonManager._supports_rigid_body_force_input = True

    @classmethod
    def _validate_config(cls, solver_cfg: CouplerCfg) -> None:
        """Validate adapter-specific nested-manager constraints before construction."""
        if not isinstance(solver_cfg, (CouplerProxyCfg, CouplerAdmmCfg)):
            raise TypeError(
                f"CouplerCfg subclass {type(solver_cfg).__name__!r} is not supported; "
                "use CouplerProxyCfg or CouplerAdmmCfg."
            )
        if not solver_cfg.entries:
            raise ValueError("CouplerCfg.entries must contain at least one solver entry.")

        if any(not isinstance(entry.name, str) or not entry.name for entry in solver_cfg.entries):
            raise ValueError("CouplerCfg entry names must be non-empty strings.")

        for entry in solver_cfg.entries:
            nested_cfg = entry.solver_cfg
            if not isinstance(nested_cfg, NewtonSolverCfg):
                raise TypeError(
                    f"CouplerEntryCfg {entry.name!r} solver_cfg must be a NewtonSolverCfg, "
                    f"got {type(nested_cfg).__name__}."
                )
            if isinstance(nested_cfg, CouplerCfg):
                raise ValueError(
                    f"CouplerEntryCfg {entry.name!r} contains a nested CouplerCfg; nested couplers are not supported."
                )
            manager = nested_cfg.class_type
            factory = getattr(manager, "_create_solver", None)
            if not callable(factory) or getattr(factory, "__func__", factory) is NewtonManager._create_solver.__func__:
                raise TypeError(
                    f"CouplerEntryCfg {entry.name!r} uses {type(nested_cfg).__name__}, whose manager "
                    "does not implement nested solver construction."
                )
            if isinstance(nested_cfg, (KaminoPADMMSolverCfg, KaminoDVISolverCfg)):
                raise NotImplementedError(
                    f"CouplerEntryCfg {entry.name!r} uses a Kamino solver config, whose manager-specific FK/reset "
                    "lifecycle cannot yet be preserved inside Newton's coupled-solver entry API."
                )
            if isinstance(nested_cfg, MPMSolverCfg) and nested_cfg.project_outside_colliders:
                raise NotImplementedError(
                    f"CouplerEntryCfg {entry.name!r} enables MPMSolverCfg.project_outside_colliders, whose "
                    "manager-level post-step projection cannot yet run inside a coupled-solver entry."
                )
            if isinstance(nested_cfg, MPMSolverCfg) and not entry.in_place:
                raise ValueError(f"CouplerEntryCfg {entry.name!r} uses MPMSolverCfg and must set in_place=True.")
            if isinstance(nested_cfg, MJWarpSolverCfg) and nested_cfg.use_mujoco_cpu:
                raise NotImplementedError(
                    f"CouplerEntryCfg {entry.name!r} enables MJWarpSolverCfg.use_mujoco_cpu, whose global reset "
                    "state cannot yet preserve the manager's per-world reset-mask lifecycle inside a coupled entry."
                )

    @classmethod
    def _validate_resolved_entries(
        cls,
        model: Model,
        entries: list[_ResolvedEntry],
        solver_cfg: CouplerCfg,
        active_proxy_destinations: set[str] | None = None,
    ) -> None:
        """Reject entries that neither own nor receive model elements."""
        active_proxy_destinations = active_proxy_destinations or set()
        for entry in entries:
            owns_any = any((entry.bodies, entry.particles, entry.joints, entry.shapes))
            if not owns_any and entry.config.name not in active_proxy_destinations:
                raise ValueError(f"CouplerEntryCfg {entry.config.name!r} neither owns nor receives any model elements.")

        if isinstance(solver_cfg, CouplerProxyCfg):
            cls._validate_no_cross_entry_proxy_joints(model, {entry.config.name: entry for entry in entries})

    @classmethod
    def _register_builder_attributes(cls, builder: ModelBuilder) -> None:
        """Register custom attributes required by nested coupled entries."""
        super()._register_builder_attributes(builder)
        solver_cfg = getattr(PhysicsManager._cfg, "solver_cfg", None)
        if any(isinstance(entry.solver_cfg, MPMSolverCfg) for entry in getattr(solver_cfg, "entries", ())):
            NewtonMPMManager._register_builder_attributes(builder)

    @classmethod
    def _prepare_builder_for_finalize(cls, builder: ModelBuilder) -> None:
        """Normalize kinematic colliders when a coupled entry uses implicit MPM."""
        super()._prepare_builder_for_finalize(builder)
        solver_cfg = getattr(PhysicsManager._cfg, "solver_cfg", None)
        if any(isinstance(entry.solver_cfg, MPMSolverCfg) for entry in getattr(solver_cfg, "entries", ())):
            NewtonMPMManager._prepare_builder_for_finalize(builder)

    @classmethod
    def _initialize_contacts(cls) -> None:
        """Initialize contacts and entry-local buffers before CUDA graph capture."""
        super()._initialize_contacts()
        if cls._contacts is not None and hasattr(NewtonManager._solver, "prepare_contacts"):
            NewtonManager._solver.prepare_contacts(cls._contacts)

    @classmethod
    def _step_solver(
        cls,
        state_0: State,
        state_1: State,
        control: Control,
        contacts: Contacts | None,
        substep_dt: float,
    ) -> None:
        """Run the coupled solve, then project and republish preserved VBD poses."""
        super()._step_solver(state_0, state_1, control, contacts, substep_dt)
        bindings = NewtonCouplerManager._vbd_preserved_input_pose_projection_bindings
        if not bindings:
            return
        if state_1.body_q is None:
            raise RuntimeError("VBD input-pose projection requires parent output body poses.")
        for binding in bindings:
            binding.callback(state_1)
            wp.launch(
                _copy_parent_body_q_to_vbd_entry,
                dim=binding.parent_global_body_ids.shape[0],
                inputs=[
                    binding.parent_global_body_ids,
                    binding.entry_local_body_ids,
                    state_1.body_q,
                    binding.entry_output_body_q,
                ],
                device=state_1.body_q.device,
            )

    @classmethod
    def _check_solver_status(cls) -> None:
        """Raise asynchronous failures from nested implicit-MPM solvers."""
        NewtonMPMManager._check_solver_status()

    @classmethod
    def _solver_specific_clear(cls) -> None:
        """Clear VBD hooks and cached nested-MPM solver references."""
        super()._solver_specific_clear()
        NewtonMPMManager._solver_specific_clear()
        manager = NewtonCouplerManager
        manager._vbd_preserved_input_pose_projection_registrations = {}
        manager._vbd_preserved_input_pose_projection_names = {}
        manager._vbd_preserved_input_pose_projection_bindings = ()
        manager._vbd_preserved_input_pose_projection_next_id = 0
        manager._vbd_preserved_input_pose_projection_issuer = object()

    @classmethod
    def _requires_initial_reset_before_graph_capture(cls) -> bool:
        """Capture coupled MPM only after the task authors its initial particle state."""
        return bool(NewtonMPMManager._implicit_mpm_solvers())

    @classmethod
    def _supports_cuda_graph_capture(cls) -> bool:
        """Reject capture when a nested MPM solver has dynamic storage."""
        return all(
            NewtonMPMManager._solver_supports_cuda_graph_capture(solver)
            for solver in NewtonMPMManager._implicit_mpm_solvers()
        )

    @classmethod
    def _reset_solver_internals(cls, world_mask: wp.array | None) -> None:
        """Promote a selected single MPM world to the solver's full-reset path."""
        model = NewtonManager._model
        solver_cfg = getattr(PhysicsManager._cfg, "solver_cfg", None)
        has_mpm_entry = any(isinstance(entry.solver_cfg, MPMSolverCfg) for entry in getattr(solver_cfg, "entries", ()))
        if world_mask is not None and model is not None and model.world_count == 1 and has_mpm_entry:
            selected = world_mask.numpy()
            if not selected.any():
                return
            if selected[0] and not selected[-1]:
                NewtonManager._solver.reset(NewtonManager._state_0, world_mask=None, flags=0)
                return
        super()._reset_solver_internals(world_mask)

    @classmethod
    def _resolve_entry(
        cls,
        model: Model,
        entry_cfg: CouplerEntryCfg,
    ) -> _ResolvedEntry:
        """Resolve one entry's selectors and derived ownership."""
        bodies = cls._resolve_entities_to_body_ids(model, entry_cfg.bodies, f"entry {entry_cfg.name!r}")

        particles = list(dict.fromkeys(map(int, entry_cfg.particles)))
        if entry_cfg.all_particles:
            particles = list(dict.fromkeys([*particles, *range(int(model.particle_count))]))

        joints: list[int] = []
        if entry_cfg.include_child_joints and int(model.joint_count):
            body_set = set(bodies)
            parents = model.joint_parent.numpy()
            joints = [
                joint
                for joint, child in enumerate(model.joint_child.numpy())
                if int(child) in body_set and (int(parents[joint]) < 0 or int(parents[joint]) in body_set)
            ]

        shapes: list[int] = []
        if entry_cfg.include_body_shapes or entry_cfg.include_static_shapes:
            body_set = set(bodies)
            for shape, body_raw in enumerate(model.shape_body.numpy()):
                body = int(body_raw)
                if (entry_cfg.include_body_shapes and body in body_set) or (
                    entry_cfg.include_static_shapes and body < 0
                ):
                    shapes.append(shape)
        if entry_cfg.shape_label_patterns:
            labels = list(getattr(model, "shape_label", ()) or ())
            labeled_shapes = [(index, label) for index, label in enumerate(labels) if label is not None]
            try:
                matched_shapes, _ = resolve_matching_names(
                    entry_cfg.shape_label_patterns, [label for _, label in labeled_shapes]
                )
            except ValueError as error:
                raise ValueError(
                    f"CouplerEntryCfg {entry_cfg.name!r}: failed to resolve shape-label patterns."
                ) from error
            shapes.extend(labeled_shapes[index][0] for index in matched_shapes)

        return cls._ResolvedEntry(
            config=entry_cfg,
            bodies=bodies,
            particles=particles,
            joints=list(dict.fromkeys(joints)),
            shapes=list(dict.fromkeys(shapes)),
        )

    @classmethod
    def _resolve_proxy(
        cls,
        model: Model,
        proxy_cfg: CouplerProxyMappingCfg,
    ) -> CouplerProxyMappingCfg:
        """Resolve a proxy mapping's selectors to collidable body ids, writing them into the config in place."""
        selected = cls._resolve_entities_to_body_ids(
            model, proxy_cfg.bodies, f"proxy {proxy_cfg.source!r}->{proxy_cfg.destination!r}"
        )
        collide_flag = int(ShapeFlags.COLLIDE_SHAPES)
        collide_bodies = {
            int(body)
            for body, flags in zip(model.shape_body.numpy(), model.shape_flags.numpy())
            if int(body) >= 0 and int(flags) & collide_flag
        }
        bodies = [body for body in selected if body in collide_bodies]
        if proxy_cfg.bodies and not bodies:
            raise ValueError(
                f"CouplerProxyMappingCfg {proxy_cfg.source!r}->{proxy_cfg.destination!r} selected no bodies "
                "with ShapeFlags.COLLIDE_SHAPES."
            )
        proxy_cfg.bodies = bodies
        proxy_cfg.particles = list(dict.fromkeys(map(int, proxy_cfg.particles)))
        return proxy_cfg

    @classmethod
    def _resolve_entities_to_body_ids(
        cls,
        model: Model,
        specs: list[str | int],
        field: str,
    ) -> list[int]:
        """Resolve body-label regexes or raw body ids to unique, order-preserving body ids."""
        labels = list(model.body_label)
        body_ids: list[int] = []
        for spec in specs:
            if isinstance(spec, int):
                if not 0 <= spec < len(labels):
                    raise ValueError(f"CouplerCfg {field}: body id {spec} is out of range [0, {len(labels)}).")
                body_ids.append(spec)
                continue
            if isinstance(spec, str):
                matched, _ = resolve_matching_names(f"(?:{spec})(?:/.*)?", labels, raise_when_no_match=False)
                if not matched:
                    raise ValueError(f"CouplerCfg {field}: body-label regex {spec!r} matched no Newton bodies.")
                body_ids.extend(matched)
                continue

            raise TypeError(
                f"CouplerCfg {field}: expected a full-label regex string or raw body id; got {type(spec).__name__}."
            )

        return list(dict.fromkeys(body_ids))

    @classmethod
    def _build_entry(cls, entry: _ResolvedEntry) -> SolverCoupled.Entry:
        entry_cfg = entry.config

        def solver_factory(model_view):
            solver = entry_cfg.solver_cfg.class_type._create_solver(model_view, entry_cfg.solver_cfg)
            if isinstance(solver, SolverVBD):
                cls._install_vbd_pose_history_restore_hook(solver)
            return solver

        return SolverCoupled.Entry(
            name=entry_cfg.name,
            solver=solver_factory,
            bodies=entry.bodies,
            particles=entry.particles,
            joints=entry.joints,
            shapes=entry.shapes,
            substeps=entry_cfg.substeps,
            in_place=entry_cfg.in_place,
        )

    @classmethod
    def _build_proxy_coupled_solver(
        cls,
        model: Model,
        entries: list[SolverCoupled.Entry],
        proxy_cfgs: list[CouplerProxyMappingCfg],
        solver_cfg: CouplerProxyCfg,
    ) -> SolverCoupledProxy:
        proxies = []
        for proxy_cfg in proxy_cfgs:
            values = vars(proxy_cfg).copy()
            if isinstance(values["collision_pipeline"], NewtonCollisionPipelineCfg):
                collision_cfg = values["collision_pipeline"]
                values["collision_pipeline"] = partial(CollisionPipeline, **collision_cfg.to_pipeline_args())
            proxies.append(SolverCoupledProxy.Proxy(**values))
        coupling_values = cls._filter_solver_kwargs(SolverCoupledProxy.Config, solver_cfg)
        coupling_values["proxies"] = proxies
        coupling = SolverCoupledProxy.Config(**coupling_values)
        return SolverCoupledProxy(model=model, entries=entries, coupling=coupling)

    @classmethod
    def _build_admm_coupled_solver(
        cls,
        model: Model,
        entries: list[SolverCoupled.Entry],
        solver_cfg: CouplerAdmmCfg,
    ) -> SolverCoupledADMM:
        values = cls._filter_solver_kwargs(SolverCoupledADMM.Config, solver_cfg)
        if solver_cfg.contact_pairs is None:
            values["contact_pairs"] = SolverCoupledADMM.auto_detect_contact_pairs(entries)
        else:
            values["contact_pairs"] = [
                SolverCoupledADMM.ContactPair(source=source, destination=destination)
                for source, destination in solver_cfg.contact_pairs
            ]
        coupling = SolverCoupledADMM.Config(**values)
        return SolverCoupledADMM(model=model, entries=entries, coupling=coupling)

    @staticmethod
    def _validate_no_cross_entry_proxy_joints(model: Model, entries: dict[str, _ResolvedEntry]) -> None:
        body_owner = {int(body): name for name, entry in entries.items() for body in entry.bodies}
        for joint, (parent_raw, child_raw) in enumerate(zip(model.joint_parent.numpy(), model.joint_child.numpy())):
            parent = int(parent_raw)
            child = int(child_raw)
            parent_owner = body_owner.get(parent)
            child_owner = body_owner.get(child)
            if parent >= 0 and parent_owner is not None and child_owner is not None and parent_owner != child_owner:
                raise ValueError(
                    f"CouplerProxyCfg does not support cross-entry joint {joint} between "
                    f"{parent_owner!r} and {child_owner!r}; keep the articulation in one entry "
                    "or use ADMM coupling."
                )
