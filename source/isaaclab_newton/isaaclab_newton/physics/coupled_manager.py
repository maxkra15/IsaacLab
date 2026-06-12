# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coupled Newton multi-solver manager."""

from __future__ import annotations

import copy
import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from newton import CollisionPipeline, Model, eval_fk
from newton._src.solvers.coupled.proxy_utils import sync_proxy_particles_kernel, sync_proxy_states_kernel
from newton.solvers.experimental.coupled import (
    CouplingInterface,
    SolverCoupled,
    SolverCoupledADMM,
    SolverCoupledProxy,
)

from isaaclab.managers import SceneEntityCfg
from isaaclab.physics import PhysicsManager

from .coupled_manager_cfg import (
    AdmmContactPairCfg,
    AdmmCouplingCfg,
    CoupledProxyCfg,
    CoupledSolverCfg,
    CoupledSolverEntryCfg,
)
from .mjwarp_manager import apply_mujoco_warp_model_overrides
from .newton_manager import NewtonManager
from .solver_factory import (
    resolve_class_or_callable,
    resolve_newton_solver_class_and_kwargs,
    solver_cfg_needs_external_contacts,
)

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveSceneCfg

logger = logging.getLogger(__name__)


@wp.kernel(enable_backward=False)
def _int_mask_to_bool_mask_kernel(src: wp.array(dtype=wp.int32), dst: wp.array(dtype=wp.bool)):
    tid = wp.tid()
    dst[tid] = src[tid] != 0


class _EntryCollisionPipelineSolver(CouplingInterface):
    """Run a solver with a collision pipeline scoped to its model view."""

    def __init__(self, solver, model_view):
        self._solver = solver
        self._collision_pipeline = self._make_collision_pipeline(model_view)
        self._contacts = self._collision_pipeline.contacts()
        self.coupling_unsupported = getattr(solver, "coupling_unsupported", frozenset())

    @staticmethod
    def _make_collision_pipeline(model_view):
        collision_cfg = NewtonManager._collision_cfg
        if collision_cfg is not None:
            return CollisionPipeline(model_view, **collision_cfg.to_pipeline_args())
        return CollisionPipeline(model_view, broad_phase="explicit")

    def __getattr__(self, name):
        return getattr(self._solver, name)

    def step(self, state_in, state_out, control, contacts, dt):
        del contacts
        self._collision_pipeline.collide(state_in, self._contacts)
        self._solver.step(state_in, state_out, control, self._contacts, dt)


class NewtonCoupledManager(NewtonManager):
    """:class:`NewtonManager` specialization for Newton coupled solvers.

    The manager is intentionally thin: Isaac Lab owns lifecycle, state buffers,
    collision-pipeline refresh, and visualization, while Newton's coupled
    solvers own per-solver ``ModelView`` construction and
    cross-entry force or constraint exchange.
    """

    _bool_world_reset_mask: wp.array | None = None
    """Boolean copy of :attr:`NewtonManager._world_reset_mask` for sub-solver reset kernels."""

    _teleport_protocol_streak: int = 0
    """Consecutive steps on which the teleport-reset protocol fired (per-step-write detector)."""

    @classmethod
    def get_entry_solver(cls, name: str):
        """Return a named sub-solver from the active coupled solver."""
        solver = NewtonManager._solver
        if solver is None:
            raise RuntimeError("Newton coupled solver is not initialized.")
        return solver.solver(name)

    @classmethod
    def get_entry_view(cls, name: str):
        """Return a named sub-solver model view from the active coupled solver."""
        solver = NewtonManager._solver
        if solver is None:
            raise RuntimeError("Newton coupled solver is not initialized.")
        return solver.view(name)

    @classmethod
    def get_proxy_body_wrenches(cls, source: str, destination: str):
        """Return proxy body feedback wrenches when the active Newton solver exposes them."""
        solver = NewtonManager._solver
        if solver is None:
            return None
        for mapping in getattr(solver, "_proxy_mappings", ()):
            if mapping.src_name == source and mapping.dst_name == destination:
                return mapping.coupling_forces
        return None

    @classmethod
    def step(cls) -> None:
        """Step coupled physics, re-seeding solver history first if state was teleported."""
        sim = PhysicsManager._sim
        if NewtonManager._state_teleport_pending and sim is not None and sim.is_playing():
            cls._reset_coupled_solver_history()
            cls._teleport_protocol_streak += 1
            if cls._teleport_protocol_streak == 16:
                logger.warning(
                    "The coupled teleport-reset protocol has fired on 16 consecutive steps: some"
                    " event or action term writes asset state every step. Each run clears contact"
                    " warm-start history and sub-solver warm starts, which degrades contact quality"
                    " and performance. Route continuous targets through actions/controls instead of"
                    " state writes."
                )
        else:
            cls._teleport_protocol_streak = 0
        super().step()

    @classmethod
    def _reset_coupled_solver_history(cls) -> None:
        """Run the coupled solver's discontinuity protocol after asset state writes.

        The coupled pipeline is history-based: every substep the proxy sync converts
        source-body pose deltas into destination-proxy velocities, and history-keeping
        entry solvers (e.g. ``SolverVBD``) reference per-body previous poses for
        friction and velocity finalization. A state write that teleports bodies (an
        env reset) must not flow through that incremental path: a 1 m jump at a 1 ms
        substep becomes a 1000 m/s proxy sweep through resting contacts. This applies
        Newton's own reset/teleport contracts from the manager side:

        1. FK so the parent state's body poses match the teleported joint state.
        2. :meth:`SolverCoupled.reset` — distributes the parent state to entry views
           without velocity folding (``dt=0``), resets sub-solver internals (e.g.
           MuJoCo warm starts), and clears lagged proxy feedback forces and
           contact-matching buffers.
        3. Re-syncs proxy body poses/velocities from their teleported source bodies
           (the reset cascade clears proxy transients but does not move proxies).
        4. Re-seeds each entry solver's previous-pose history, per ``SolverVBD``'s
           documented teleport contract ("Dynamic teleportation: also set
           ``body_q_prev`` and ``body_qd``").

        Known cost: Newton's reset API clears coupling forces and contact/matching
        buffers globally (all worlds), so a partial multi-env reset briefly restarts
        contact warm-start history for non-reset envs as well. This is a bounded
        one-step transient; per-world clearing would need an upstream Newton change.
        """
        solver = NewtonManager._solver
        state = NewtonManager._state_0
        if solver is None or state is None:
            return
        with wp.ScopedDevice(PhysicsManager._device):
            # Sub-solver reset kernels expect a boolean world mask. The manager's
            # accumulated mask is bool on current upstream; convert when an older
            # int32 (Kamino-convention) allocation is active.
            world_mask = NewtonManager._world_reset_mask
            if world_mask is not None and world_mask.dtype is not wp.bool:
                if cls._bool_world_reset_mask is None or cls._bool_world_reset_mask.shape != world_mask.shape:
                    cls._bool_world_reset_mask = wp.zeros(world_mask.shape, dtype=wp.bool, device=world_mask.device)
                wp.launch(
                    _int_mask_to_bool_mask_kernel,
                    dim=world_mask.shape[0],
                    inputs=[world_mask, cls._bool_world_reset_mask],
                )
                world_mask = cls._bool_world_reset_mask
            eval_fk(cls._model, state.joint_q, state.joint_qd, state, cls._filtered_fk_reset_mask())
            # Newton bug workaround: SolverVBD.rebuild_bvh (reached via
            # SolverCoupled.reset -> _rebuild_entry_solver_state_caches) reads
            # ``particle_enable_self_contact``, which SolverVBD.__init__ only assigns
            # when the model has particles. Pre-seed the documented default (False)
            # so reset also works for rigid-only VBD entries.
            for entry in solver._entries.values():
                if callable(getattr(entry.solver, "rebuild_bvh", None)) and not hasattr(
                    entry.solver, "particle_enable_self_contact"
                ):
                    entry.solver.particle_enable_self_contact = False
            # flags=0: reset NO state quantities — Isaac Lab owns the sim state and
            # its reset events already wrote the desired values into the parent
            # state (distributed above); sub-solver state reset would restore MODEL
            # defaults instead (e.g. SolverMuJoCo snaps joints to USD defaults).
            # Sub-solvers still clear their internal buffers (MuJoCo warm starts,
            # applied forces, actuator activations) regardless of flags.
            # NOTE: with update_data_interval != 1 the immediate qpos push in
            # SolverMuJoCo.reset is gated on the JOINT_Q flag; the default
            # interval (1) re-syncs state -> qpos every step, which is what the
            # coupled configs use.
            solver.reset(state, world_mask=world_mask, flags=0)
            for proxy in getattr(solver, "_proxy_mappings", ()):
                src = solver._entries[proxy.src_name]
                dst = solver._entries[proxy.dst_name]
                for dst_state in (dst.state_0, dst.state_1):
                    if dst_state is None:
                        continue
                    wp.launch(
                        sync_proxy_states_kernel,
                        dim=proxy.source_local_to_proxy_local.shape[0],
                        inputs=[
                            src.state_0.body_q,
                            src.state_0.body_qd,
                            proxy.source_local_to_proxy_local,
                            dst_state.body_q,
                            dst_state.body_qd,
                        ],
                    )
            for proxy in getattr(solver, "_proxy_particle_mappings", ()):
                src = solver._entries[proxy.src_name]
                dst = solver._entries[proxy.dst_name]
                for dst_state in (dst.state_0, dst.state_1):
                    if dst_state is None:
                        continue
                    wp.launch(
                        sync_proxy_particles_kernel,
                        dim=proxy.source_local_to_proxy_local.shape[0],
                        inputs=[
                            src.state_0.particle_q,
                            src.state_0.particle_qd,
                            proxy.source_local_to_proxy_local,
                            dst_state.particle_q,
                            dst_state.particle_qd,
                        ],
                    )
            for entry in solver._entries.values():
                body_q_prev = getattr(entry.solver, "body_q_prev", None)
                if body_q_prev is not None:
                    wp.copy(dest=body_q_prev, src=entry.state_0.body_q)

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: CoupledSolverCfg) -> None:
        """Construct a Newton coupled solver and populate the base-class slots."""
        solver_cfg = cls._resolve_solver_cfg(model, solver_cfg)
        cls._validate_solver_cfg(solver_cfg)

        cls._apply_coupled_model_cfg(model)
        cls._apply_proxy_shape_overrides(model, solver_cfg.proxy_coupling.proxies)

        entries = [cls._build_entry(entry_cfg) for entry_cfg in solver_cfg.entries]
        if solver_cfg.coupling_type == "base":
            NewtonManager._solver = SolverCoupled(model=model, entries=entries)
        elif solver_cfg.coupling_type == "proxy":
            NewtonManager._solver = SolverCoupledProxy(
                model=model,
                entries=entries,
                coupling=SolverCoupledProxy.Config(
                    proxies=[cls._build_proxy(proxy_cfg) for proxy_cfg in solver_cfg.proxy_coupling.proxies],
                    iterations=solver_cfg.proxy_coupling.iterations,
                ),
            )
        elif solver_cfg.coupling_type == "admm":
            NewtonManager._solver = SolverCoupledADMM(
                model=model,
                entries=entries,
                coupling=cls._build_admm(solver_cfg.admm_coupling, entries),
            )
        else:
            raise ValueError(f"Unsupported Newton coupling_type {solver_cfg.coupling_type!r}.")

        cls._apply_entry_solver_overrides(solver_cfg.entries)
        cls._apply_vbd_joint_constraint_modes(solver_cfg.entries)
        cls._configure_fk_articulation_filter(model, solver_cfg.entries)
        if hasattr(NewtonManager._solver, "prepare_graph_capture"):
            NewtonManager._solver.prepare_graph_capture()
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = cls._needs_external_collision_pipeline(solver_cfg)

    @classmethod
    def _resolve_solver_cfg(cls, model: Model, solver_cfg: CoupledSolverCfg) -> CoupledSolverCfg:
        """Return a shallow copy of ``solver_cfg`` with selector fields resolved to ids."""
        scene_cfg = cls._resolve_scene_cfg(solver_cfg)
        resolved_cfg = copy.copy(solver_cfg)
        resolved_cfg.entries = [cls._resolve_entry_cfg(model, entry_cfg, scene_cfg) for entry_cfg in solver_cfg.entries]
        resolved_proxy_coupling = copy.copy(solver_cfg.proxy_coupling)
        resolved_proxy_coupling.proxies = [
            cls._resolve_proxy_cfg(model, proxy_cfg, scene_cfg) for proxy_cfg in solver_cfg.proxy_coupling.proxies
        ]
        resolved_cfg.proxy_coupling = resolved_proxy_coupling
        return resolved_cfg

    @staticmethod
    def _resolve_scene_cfg(solver_cfg: CoupledSolverCfg):
        """Resolve the scene cfg used by ``SceneEntityCfg`` selectors."""
        if solver_cfg.scene_cfg is not None:
            return solver_cfg.scene_cfg
        return getattr(PhysicsManager._cfg, "scene_cfg", None)

    @classmethod
    def _resolve_entry_cfg(
        cls, model: Model, entry_cfg: CoupledSolverEntryCfg, scene_cfg: InteractiveSceneCfg | None
    ) -> CoupledSolverEntryCfg:
        """Resolve one entry's front-end selectors into raw Newton index lists."""
        resolved = copy.copy(entry_cfg)
        body_selector_used = cls._uses_body_selectors(entry_cfg)
        selected_bodies = cls._resolve_body_selectors(model, entry_cfg, scene_cfg, f"entry {entry_cfg.name!r}")
        bodies = cls._unique_ints([*entry_cfg.bodies, *selected_bodies])
        joints = list(entry_cfg.joints)
        shapes = list(entry_cfg.shapes)
        shapes.extend(
            cls._resolve_shape_label_patterns(
                model,
                entry_cfg.shape_label_patterns,
                f"entry {entry_cfg.name!r}",
            )
        )
        if body_selector_used:
            if entry_cfg.include_child_joints:
                joints.extend(cls._child_joints_for_bodies(model, bodies))
            if entry_cfg.include_body_shapes or entry_cfg.include_static_shapes:
                shapes.extend(
                    cls._shapes_for_bodies(
                        model,
                        bodies,
                        include_body_shapes=entry_cfg.include_body_shapes,
                        include_static_shapes=entry_cfg.include_static_shapes,
                    )
                )

        resolved.bodies = bodies
        resolved.joints = cls._unique_ints(joints)
        resolved.shapes = cls._unique_ints(shapes)
        resolved.particles = cls._resolve_particles(
            model,
            explicit=entry_cfg.particles,
            particle_range=entry_cfg.particle_range,
            all_particles=entry_cfg.all_particles,
            field=f"CoupledSolverEntryCfg {entry_cfg.name!r}",
        )
        return resolved

    @classmethod
    def _resolve_proxy_cfg(
        cls, model: Model, proxy_cfg: CoupledProxyCfg, scene_cfg: InteractiveSceneCfg | None
    ) -> CoupledProxyCfg:
        """Resolve one proxy cfg's selectors into raw Newton index lists."""
        resolved = copy.copy(proxy_cfg)
        selected_bodies = cls._resolve_body_selectors(
            model,
            proxy_cfg,
            scene_cfg,
            f"proxy {proxy_cfg.source!r}->{proxy_cfg.destination!r}",
        )
        resolved.bodies = cls._unique_ints([*proxy_cfg.bodies, *selected_bodies])
        selected_proxy_bodies = [
            *cls._resolve_body_label_patterns(
                model,
                proxy_cfg.proxy_body_label_patterns,
                f"proxy {proxy_cfg.source!r}->{proxy_cfg.destination!r} destination",
            ),
            *cls._resolve_body_name_patterns(
                model,
                proxy_cfg.proxy_body_name_patterns,
                f"proxy {proxy_cfg.source!r}->{proxy_cfg.destination!r} destination",
            ),
        ]
        if selected_proxy_bodies:
            resolved.proxy_bodies = cls._unique_ints([*(proxy_cfg.proxy_bodies or []), *selected_proxy_bodies])
        resolved.particles = cls._resolve_particles(
            model,
            explicit=proxy_cfg.particles,
            particle_range=proxy_cfg.particle_range,
            all_particles=proxy_cfg.all_particles,
            field=f"CoupledProxyCfg {proxy_cfg.source!r}->{proxy_cfg.destination!r}",
        )
        selected_proxy_particles = cls._resolve_particles(
            model,
            explicit=[] if proxy_cfg.proxy_particles is None else proxy_cfg.proxy_particles,
            particle_range=proxy_cfg.proxy_particle_range,
            all_particles=False,
            field=f"CoupledProxyCfg {proxy_cfg.source!r}->{proxy_cfg.destination!r} proxy",
        )
        if selected_proxy_particles:
            resolved.proxy_particles = selected_proxy_particles
        return resolved

    @staticmethod
    def _uses_body_selectors(cfg: CoupledSolverEntryCfg | CoupledProxyCfg) -> bool:
        return bool(cfg.body_entities or cfg.body_label_patterns or cfg.body_name_patterns)

    @classmethod
    def _resolve_body_selectors(
        cls,
        model: Model,
        cfg: CoupledSolverEntryCfg | CoupledProxyCfg,
        scene_cfg: InteractiveSceneCfg | None,
        field: str,
    ) -> list[int]:
        body_ids: list[int] = []
        if cfg.body_entities:
            if scene_cfg is None:
                raise ValueError(
                    f"{type(cfg).__name__} {field} uses body_entities, but CoupledSolverCfg.scene_cfg is not set. "
                    "Set scene_cfg=self.scene in the coupled solver cfg or use body_label_patterns/body_name_patterns."
                )
            for entity_cfg in cfg.body_entities:
                body_ids.extend(cls._resolve_entity_to_body_ids(model, entity_cfg, scene_cfg, field))
        body_ids.extend(cls._resolve_body_label_patterns(model, cfg.body_label_patterns, field))
        body_ids.extend(cls._resolve_body_name_patterns(model, cfg.body_name_patterns, field))
        return cls._unique_ints(body_ids)

    @classmethod
    def _resolve_entity_to_body_ids(
        cls,
        model: Model,
        entity_cfg: SceneEntityCfg,
        scene_cfg: InteractiveSceneCfg,
        field: str,
    ) -> list[int]:
        """Resolve one ``SceneEntityCfg`` to Newton body ids."""
        asset_cfg = getattr(scene_cfg, entity_cfg.name, None)
        if asset_cfg is None or not hasattr(asset_cfg, "prim_path"):
            raise ValueError(
                f"CoupledSolverCfg {field} references scene entity {entity_cfg.name!r}, "
                "which is not present on scene_cfg or lacks prim_path."
            )

        asset_pattern = str(asset_cfg.prim_path).replace("{ENV_REGEX_NS}", r"/World/envs/env_.*")
        asset_regex = re.compile(rf"^{asset_pattern}(/|$)")
        labels = cls._body_labels(model)
        candidate_ids = [body_id for body_id, label in enumerate(labels) if asset_regex.match(label)]
        patterns = entity_cfg.body_names
        if isinstance(patterns, str):
            patterns = [patterns]
        if patterns is None:
            body_ids = cls._select_entity_body_ids(candidate_ids, entity_cfg.body_ids, field, entity_cfg.name)
            if not candidate_ids:
                raise ValueError(
                    f"CoupledSolverCfg {field}: scene entity {entity_cfg.name!r} matched no Newton bodies "
                    f"under prim_path regex {asset_pattern!r}."
                )
            if not body_ids:
                raise ValueError(
                    f"CoupledSolverCfg {field}: scene entity {entity_cfg.name!r} body_ids selected no bodies "
                    f"from {len(candidate_ids)} candidate Newton bodies."
                )
            return body_ids
        if not cls._is_all_slice(entity_cfg.body_ids):
            raise ValueError(
                f"CoupledSolverCfg {field}: scene entity {entity_cfg.name!r} sets both body_names and body_ids. "
                "Use only one selector to avoid ambiguous Newton body ownership."
            )

        compiled = [re.compile(pattern) for pattern in patterns]
        matched = [False] * len(compiled)
        body_ids: list[int] = []
        if entity_cfg.preserve_order:
            for index, pattern in enumerate(compiled):
                matches = [
                    body_id for body_id in candidate_ids if pattern.fullmatch(labels[body_id].rsplit("/", 1)[-1])
                ]
                if matches:
                    matched[index] = True
                    body_ids.extend(matches)
        else:
            for body_id in candidate_ids:
                short_name = labels[body_id].rsplit("/", 1)[-1]
                hit = next((index for index, pattern in enumerate(compiled) if pattern.fullmatch(short_name)), None)
                if hit is None:
                    continue
                matched[hit] = True
                body_ids.append(body_id)

        unmatched = [pattern for pattern, ok in zip(patterns, matched) if not ok]
        if unmatched:
            raise ValueError(
                f"CoupledSolverCfg {field}: scene entity {entity_cfg.name!r} has no Newton bodies matching "
                f"{unmatched}. Check the regexes against body short names."
            )
        return cls._unique_ints(body_ids)

    @staticmethod
    def _is_all_slice(value) -> bool:
        return isinstance(value, slice) and value.start is None and value.stop is None and value.step is None

    @classmethod
    def _select_entity_body_ids(cls, candidate_ids: list[int], selector, field: str, entity_name: str) -> list[int]:
        """Apply an entity-local ``body_ids`` selector to candidate Newton body ids."""
        if cls._is_all_slice(selector):
            return candidate_ids
        if isinstance(selector, int):
            selector = [selector]
        if isinstance(selector, slice):
            return candidate_ids[selector]

        body_ids: list[int] = []
        for raw_index in selector:
            local_index = int(raw_index)
            if local_index < 0:
                raise ValueError(
                    f"CoupledSolverCfg {field}: scene entity {entity_name!r} body_ids index {local_index} is "
                    "negative. Use non-negative entity-local body ids."
                )
            try:
                body_ids.append(candidate_ids[local_index])
            except IndexError as exc:
                raise ValueError(
                    f"CoupledSolverCfg {field}: scene entity {entity_name!r} body_ids index {local_index} is "
                    f"outside the matched Newton body range [0, {len(candidate_ids)})."
                ) from exc
        return body_ids

    @classmethod
    def _resolve_body_label_patterns(cls, model: Model, patterns: list[str], field: str) -> list[int]:
        """Resolve full-body-label regexes to body ids."""
        labels = cls._body_labels(model)
        return cls._resolve_body_patterns(labels, patterns, field, "body_label_patterns")

    @classmethod
    def _resolve_shape_label_patterns(cls, model: Model, patterns: list[str], field: str) -> list[int]:
        """Resolve full-shape-label regexes to shape ids."""
        if not patterns:
            return []
        labels = getattr(model, "shape_label", None)
        if labels is None:
            raise ValueError("Newton model does not expose shape_label; shape selectors cannot be resolved.")
        return cls._resolve_body_patterns(
            [str(label) for label in labels],
            patterns,
            field,
            "shape_label_patterns",
        )

    @classmethod
    def _resolve_body_name_patterns(cls, model: Model, patterns: list[str], field: str) -> list[int]:
        """Resolve short-body-name regexes to body ids."""
        labels = cls._body_labels(model)
        short_names = [label.rsplit("/", 1)[-1] for label in labels]
        return cls._resolve_body_patterns(short_names, patterns, field, "body_name_patterns")

    @staticmethod
    def _resolve_body_patterns(
        match_values: list[str], patterns: list[str], field: str, selector_name: str
    ) -> list[int]:
        body_ids: list[int] = []
        for pattern in patterns:
            regex = re.compile(pattern)
            matches = [body_id for body_id, value in enumerate(match_values) if regex.fullmatch(value)]
            if not matches:
                raise ValueError(f"CoupledSolverCfg {field}: {selector_name} pattern {pattern!r} matched no bodies.")
            body_ids.extend(matches)
        return body_ids

    @staticmethod
    def _body_labels(model: Model) -> list[str]:
        labels = getattr(model, "body_label", None) or getattr(model, "body_key", None)
        if labels is None:
            raise ValueError("Newton model does not expose body_label/body_key; body selectors cannot be resolved.")
        return [str(label) for label in labels]

    @classmethod
    def _child_joints_for_bodies(cls, model: Model, body_ids: list[int]) -> list[int]:
        """Return joints whose child body is in ``body_ids``."""
        if int(getattr(model, "joint_count", 0)) <= 0 or getattr(model, "joint_child", None) is None:
            return []
        owned = set(body_ids)
        return [joint_id for joint_id, child in enumerate(model.joint_child.numpy()) if int(child) in owned]

    @classmethod
    def _shapes_for_bodies(
        cls,
        model: Model,
        body_ids: list[int],
        *,
        include_body_shapes: bool,
        include_static_shapes: bool,
    ) -> list[int]:
        """Return shapes attached to selected bodies and optionally static shapes."""
        if int(getattr(model, "shape_count", 0)) <= 0 or getattr(model, "shape_body", None) is None:
            return []
        owned = set(body_ids)
        shape_ids: list[int] = []
        for shape_id, body_id_raw in enumerate(model.shape_body.numpy()):
            body_id = int(body_id_raw)
            if (include_body_shapes and body_id in owned) or (include_static_shapes and body_id < 0):
                shape_ids.append(shape_id)
        return shape_ids

    @classmethod
    def _resolve_particles(
        cls,
        model: Model,
        *,
        explicit: list[int],
        particle_range: tuple[int | None, int | None] | None,
        all_particles: bool,
        field: str,
    ) -> list[int]:
        particle_count = int(getattr(model, "particle_count", 0))
        particles = list(explicit)
        if all_particles:
            particles.extend(range(particle_count))
        if particle_range is not None:
            start_raw, end_raw = particle_range
            start = 0 if start_raw is None else int(start_raw)
            end = particle_count if end_raw is None else int(end_raw)
            if start < 0 or end < start or end > particle_count:
                raise ValueError(
                    f"{field}.particle_range must satisfy 0 <= start <= end <= particle_count "
                    f"({particle_count}), got ({start}, {end})."
                )
            particles.extend(range(start, end))
        return cls._unique_ints(particles)

    @staticmethod
    def _unique_ints(values) -> list[int]:
        seen: set[int] = set()
        result: list[int] = []
        for value in values:
            index = int(value)
            if index in seen:
                continue
            seen.add(index)
            result.append(index)
        return result

    @classmethod
    def _apply_entry_solver_overrides(cls, entries: list[CoupledSolverEntryCfg]) -> None:
        """Apply post-construction solver cfg overrides for coupled sub-solvers."""
        for entry_cfg in entries:
            if getattr(entry_cfg.solver_cfg, "solver_type", None) != "mujoco_warp":
                continue
            apply_mujoco_warp_model_overrides(NewtonManager._solver.solver(entry_cfg.name), entry_cfg.solver_cfg)

    @classmethod
    def _apply_vbd_joint_constraint_modes(cls, entries: list[CoupledSolverEntryCfg]) -> None:
        """Soften VBD structural joints to penalty-only when ``rigid_joint_hard=False``.

        Mirrors the Newton waterhose reference, which sets every VBD joint to
        penalty-only (no augmented-Lagrangian dual) to avoid lambda accumulation
        against cable bend torques that otherwise blows up the solve.
        """
        for entry_cfg in entries:
            if getattr(entry_cfg.solver_cfg, "rigid_joint_hard", True):
                continue
            solver_class, _ = resolve_newton_solver_class_and_kwargs(
                entry_cfg.solver_cfg,
                entry_cfg.solver_class,
                entry_cfg.solver_kwargs,
            )
            if getattr(solver_class, "__name__", "") != "SolverVBD":
                continue
            sub_solver = NewtonManager._solver.solver(entry_cfg.name)
            joint_count = int(getattr(getattr(sub_solver, "model", None), "joint_count", 0))
            for joint_index in range(joint_count):
                sub_solver.set_joint_constraint_mode(joint_index, hard=False)

    @classmethod
    def _configure_fk_articulation_filter(cls, model: Model, entries: list[CoupledSolverEntryCfg]) -> None:
        """Exclude solver-owned VBD articulations from NewtonManager's generic FK path."""
        if model.articulation_count <= 0 or getattr(model, "joint_articulation", None) is None:
            NewtonManager._set_fk_articulation_filter(None)
            return

        fk_mask = np.ones(int(model.articulation_count), dtype=bool)
        joint_articulation = model.joint_articulation.numpy()
        disabled_any = False
        for entry_cfg in entries:
            solver_class, _ = resolve_newton_solver_class_and_kwargs(
                entry_cfg.solver_cfg,
                entry_cfg.solver_class,
                entry_cfg.solver_kwargs,
            )
            if getattr(solver_class, "__name__", "") != "SolverVBD":
                continue
            for joint_id in entry_cfg.joints:
                joint_index = int(joint_id)
                if joint_index < 0 or joint_index >= joint_articulation.shape[0]:
                    continue
                articulation_id = int(joint_articulation[joint_index])
                if articulation_id < 0:
                    continue
                fk_mask[articulation_id] = False
                disabled_any = True

        NewtonManager._set_fk_articulation_filter(fk_mask if disabled_any else None)

    @classmethod
    def _build_entry(cls, entry_cfg: CoupledSolverEntryCfg) -> SolverCoupled.Entry:
        """Build a Newton ``SolverCoupled.Entry`` from an Isaac Lab entry cfg."""
        solver_class, solver_kwargs = resolve_newton_solver_class_and_kwargs(
            entry_cfg.solver_cfg,
            entry_cfg.solver_class,
            entry_cfg.solver_kwargs,
        )
        configure_view = (
            None if entry_cfg.configure_view is None else resolve_class_or_callable(entry_cfg.configure_view)
        )

        entry_kwargs = dict(
            name=entry_cfg.name,
            solver=cls._make_entry_solver_factory(entry_cfg, solver_class, solver_kwargs),
            bodies=list(entry_cfg.bodies),
            particles=list(entry_cfg.particles),
            joints=list(entry_cfg.joints),
            shapes=list(entry_cfg.shapes),
            configure_view=configure_view,
            substeps=entry_cfg.substeps,
            in_place=entry_cfg.in_place,
        )

        return SolverCoupled.Entry(**entry_kwargs)

    @classmethod
    def _make_entry_solver_factory(
        cls, entry_cfg: CoupledSolverEntryCfg, solver_class: Callable, solver_kwargs: dict
    ) -> Callable:
        """Bind constructor kwargs into a Newton coupled entry solver factory."""
        use_entry_collision_pipeline = cls._entry_uses_local_collision_pipeline(entry_cfg)

        def _factory(model_view):
            solver = solver_class(model_view, **solver_kwargs)
            if use_entry_collision_pipeline:
                return _EntryCollisionPipelineSolver(solver, model_view)
            return solver

        _factory.__name__ = getattr(solver_class, "__name__", type(solver_class).__name__)
        return _factory

    @staticmethod
    def _entry_uses_local_collision_pipeline(entry_cfg: CoupledSolverEntryCfg) -> bool:
        solver_cfg = entry_cfg.solver_cfg
        return getattr(solver_cfg, "solver_type", None) == "mujoco_warp" and not getattr(
            solver_cfg, "use_mujoco_contacts", True
        )

    @classmethod
    def _build_proxy(cls, proxy_cfg: CoupledProxyCfg) -> SolverCoupledProxy.Proxy:
        """Build a Newton proxy mapping from an Isaac Lab proxy cfg."""
        if not proxy_cfg.source or not proxy_cfg.destination:
            raise ValueError("CoupledProxyCfg source and destination must be non-empty.")
        if not proxy_cfg.bodies and not proxy_cfg.particles:
            raise ValueError("CoupledProxyCfg must map at least one body or particle.")

        return SolverCoupledProxy.Proxy(
            source=proxy_cfg.source,
            destination=proxy_cfg.destination,
            bodies=list(proxy_cfg.bodies),
            proxy_bodies=None if proxy_cfg.proxy_bodies is None else list(proxy_cfg.proxy_bodies),
            mass_scale=proxy_cfg.mass_scale,
            mode=cls._build_proxy_mode(proxy_cfg.mode),
            particles=list(proxy_cfg.particles),
            proxy_particles=None if proxy_cfg.proxy_particles is None else list(proxy_cfg.proxy_particles),
            collision_pipeline=proxy_cfg.collision_pipeline_factory,
            collide_interval=proxy_cfg.collide_interval,
        )

    @staticmethod
    def _set_model_array_indices(model: Model, attr_name: str, indices: list[int], value: float | None) -> None:
        """Set selected entries in a Newton model array when the attribute exists."""

        if value is None or not indices:
            return
        data = getattr(model, attr_name, None)
        if data is None:
            return
        values = data.numpy()
        values[np.asarray(indices, dtype=np.int32)] = float(value)
        data.assign(values)

    @classmethod
    def _apply_coupled_model_cfg(cls, model: Model) -> None:
        """Apply global NewtonModelCfg overrides for coupled-solver models."""

        from isaaclab.physics import PhysicsManager

        cfg = PhysicsManager._cfg
        model_cfg = getattr(cfg, "model_cfg", None) if cfg is not None else None
        if model_cfg is None:
            return

        model.soft_contact_ke = float(model_cfg.soft_contact_ke)
        model.soft_contact_kd = float(model_cfg.soft_contact_kd)
        model.soft_contact_mu = float(model_cfg.soft_contact_mu)
        if model_cfg.shape_material_ke is not None:
            model.shape_material_ke.fill_(float(model_cfg.shape_material_ke))
        if model_cfg.shape_material_kd is not None:
            model.shape_material_kd.fill_(float(model_cfg.shape_material_kd))
        if model_cfg.shape_material_mu is not None:
            model.shape_material_mu.fill_(float(model_cfg.shape_material_mu))

    @classmethod
    def _apply_proxy_shape_overrides(cls, model: Model, proxies: list[CoupledProxyCfg]) -> None:
        """Apply per-proxy contact material overrides to source body shapes."""

        for proxy_cfg in proxies:
            shape_ids = cls._shapes_for_bodies(
                model,
                list(proxy_cfg.bodies),
                include_body_shapes=True,
                include_static_shapes=False,
            )
            cls._set_model_array_indices(model, "shape_material_ke", shape_ids, proxy_cfg.shape_material_ke)
            cls._set_model_array_indices(model, "shape_material_kd", shape_ids, proxy_cfg.shape_material_kd)
            cls._set_model_array_indices(model, "shape_material_mu", shape_ids, proxy_cfg.shape_material_mu)
            cls._set_model_array_indices(model, "shape_margin", shape_ids, proxy_cfg.shape_margin)
            cls._set_model_array_indices(model, "shape_gap", shape_ids, proxy_cfg.shape_gap)

    @staticmethod
    def _build_proxy_mode(mode: str | int) -> str:
        """Return the Newton proxy mode string for an Isaac Lab proxy cfg mode."""
        if isinstance(mode, str):
            return mode
        if mode == 0:
            return "lagged"
        if mode == 1:
            return "staggered"
        raise ValueError(f"Unsupported CoupledProxyCfg mode {mode!r}; expected 'lagged', 'staggered', 0, or 1.")

    @classmethod
    def _build_admm(
        cls, admm_cfg: AdmmCouplingCfg, entries: list[SolverCoupled.Entry] | None = None
    ) -> SolverCoupledADMM.Config:
        """Build a Newton ADMM coupling config from an Isaac Lab cfg."""
        contact_pairs = [cls._build_admm_contact_pair(pair_cfg) for pair_cfg in admm_cfg.contact_pairs]
        if admm_cfg.auto_contact_pairs:
            if entries is None:
                raise ValueError("AdmmCouplingCfg.auto_contact_pairs requires coupled solver entries.")
            contact_pairs.extend(
                SolverCoupledADMM.auto_detect_contact_pairs(
                    entries,
                    contact_distance=admm_cfg.auto_contact_distance,
                    detection_margin=admm_cfg.auto_detection_margin,
                )
            )

        return SolverCoupledADMM.Config(
            iterations=admm_cfg.iterations,
            rho=admm_cfg.rho,
            gamma=admm_cfg.gamma,
            baumgarte=admm_cfg.baumgarte,
            joint_stiffness=admm_cfg.joint_stiffness,
            joint_damping=admm_cfg.joint_damping,
            joint_angular_stiffness=admm_cfg.joint_angular_stiffness,
            joint_angular_damping=admm_cfg.joint_angular_damping,
            contact_pairs=contact_pairs,
        )

    @staticmethod
    def _build_admm_contact_pair(pair_cfg: AdmmContactPairCfg) -> SolverCoupledADMM.ContactPair:
        """Build a Newton ADMM contact-pair config from an Isaac Lab cfg."""
        return SolverCoupledADMM.ContactPair(
            source=pair_cfg.source,
            destination=pair_cfg.destination,
            contact_distance=pair_cfg.contact_distance,
            detection_margin=pair_cfg.detection_margin,
        )

    @classmethod
    def _validate_solver_cfg(cls, solver_cfg: CoupledSolverCfg) -> None:
        """Validate coupled-solver config before constructing Newton objects."""
        if solver_cfg.coupling_type not in ("base", "proxy", "admm"):
            raise ValueError(f"Unsupported Newton coupling_type {solver_cfg.coupling_type!r}.")
        if len(solver_cfg.entries) < 2:
            raise ValueError("Newton coupled solver requires at least two solver entries.")
        cls._validate_entries(solver_cfg.entries)
        if solver_cfg.coupling_type == "base":
            return
        if solver_cfg.coupling_type == "proxy":
            cls._validate_proxy_coupling(solver_cfg.entries, solver_cfg.proxy_coupling)
        else:
            cls._validate_admm_coupling(solver_cfg.admm_coupling)

    @classmethod
    def _validate_entries(cls, entries: list[CoupledSolverEntryCfg]) -> None:
        names: set[str] = set()
        for entry in entries:
            if not entry.name:
                raise ValueError("CoupledSolverEntryCfg.name must be non-empty.")
            if entry.name in names:
                raise ValueError(f"Duplicate CoupledSolverEntryCfg name {entry.name!r}.")
            names.add(entry.name)
            if entry.substeps < 1:
                raise ValueError(f"CoupledSolverEntryCfg {entry.name!r} substeps must be >= 1.")
            if entry.in_place and entry.substeps != 1:
                raise ValueError(f"CoupledSolverEntryCfg {entry.name!r} in_place requires substeps=1.")
        for field_name in ("bodies", "particles", "joints", "shapes"):
            cls._validate_unique_entry_ownership(entries, field_name)

    @staticmethod
    def _validate_unique_entry_ownership(entries: list[CoupledSolverEntryCfg], field_name: str) -> None:
        owners: dict[int, str] = {}
        for entry in entries:
            for raw_index in getattr(entry, field_name):
                index = int(raw_index)
                owner = owners.get(index)
                if owner is not None:
                    raise ValueError(
                        f"CoupledSolverEntryCfg {field_name} index {index} is owned by both "
                        f"{owner!r} and {entry.name!r}."
                    )
                owners[index] = entry.name

    @classmethod
    def _validate_proxy_coupling(cls, entries: list[CoupledSolverEntryCfg], proxy_coupling) -> None:
        if len(entries) > 2:
            raise ValueError("Newton proxy coupling currently supports at most two solver entries.")
        if not proxy_coupling.proxies:
            raise ValueError("Newton proxy coupling requires at least one proxy mapping.")
        if proxy_coupling.iterations < 1:
            raise ValueError("ProxyCouplingCfg.iterations must be >= 1.")
        cls._validate_proxy_mappings(entries, proxy_coupling.proxies)

    @classmethod
    def _validate_proxy_mappings(cls, entries: list[CoupledSolverEntryCfg], proxies: list[CoupledProxyCfg]) -> None:
        entry_names = {entry.name for entry in entries}
        for proxy in proxies:
            if proxy.source not in entry_names:
                raise ValueError(f"CoupledProxyCfg source {proxy.source!r} does not match a coupled entry.")
            if proxy.destination not in entry_names:
                raise ValueError(f"CoupledProxyCfg destination {proxy.destination!r} does not match a coupled entry.")
            if proxy.source == proxy.destination:
                raise ValueError("CoupledProxyCfg source and destination must be different entries.")
            if not proxy.bodies and not proxy.particles:
                raise ValueError("CoupledProxyCfg must map at least one body or particle.")
            if proxy.proxy_bodies is not None and len(proxy.proxy_bodies) != len(proxy.bodies):
                raise ValueError("CoupledProxyCfg proxy_bodies must match bodies length.")
            if proxy.proxy_particles is not None and len(proxy.proxy_particles) != len(proxy.particles):
                raise ValueError("CoupledProxyCfg proxy_particles must match particles length.")
            if proxy.mass_scale <= 0.0:
                raise ValueError("CoupledProxyCfg mass_scale must be > 0.")
            if proxy.collide_interval is not None and proxy.collide_interval < 1:
                raise ValueError("CoupledProxyCfg collide_interval must be >= 1.")
            cls._build_proxy_mode(proxy.mode)

    @staticmethod
    def _validate_admm_coupling(admm_cfg: AdmmCouplingCfg) -> None:
        if admm_cfg.iterations < 1:
            raise ValueError("AdmmCouplingCfg.iterations must be >= 1.")
        if admm_cfg.rho <= 0.0:
            raise ValueError("AdmmCouplingCfg.rho must be > 0.")
        if admm_cfg.gamma < 0.0:
            raise ValueError("AdmmCouplingCfg.gamma must be >= 0.")
        if admm_cfg.auto_contact_distance is not None and admm_cfg.auto_contact_distance < 0.0:
            raise ValueError("AdmmCouplingCfg.auto_contact_distance must be >= 0.")
        if admm_cfg.auto_detection_margin is not None and admm_cfg.auto_detection_margin < 0.0:
            raise ValueError("AdmmCouplingCfg.auto_detection_margin must be >= 0.")
        for pair in admm_cfg.contact_pairs:
            if pair.source == pair.destination:
                raise ValueError("AdmmContactPairCfg source and destination must be different.")
            if pair.contact_distance is not None and pair.contact_distance < 0.0:
                raise ValueError("AdmmContactPairCfg.contact_distance must be >= 0.")
            if pair.detection_margin is not None and pair.detection_margin < 0.0:
                raise ValueError("AdmmContactPairCfg.detection_margin must be >= 0.")

    @classmethod
    def _needs_external_collision_pipeline(cls, solver_cfg: CoupledSolverCfg) -> bool:
        """Return whether the coupled solver should receive external contacts."""
        if solver_cfg.use_collision_pipeline is not None:
            return solver_cfg.use_collision_pipeline
        return any(
            solver_cfg_needs_external_contacts(entry.solver_cfg) and not cls._entry_uses_local_collision_pipeline(entry)
            for entry in solver_cfg.entries
        )
