# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton manager for named coupled-solver configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import warp as wp
from isaaclab_newton.physics import (
    FeatherstoneSolverCfg,
    MJWarpSolverCfg,
    NewtonSolverCfg,
    XPBDSolverCfg,
)
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import CollisionPipeline, Model, ShapeFlags, eval_fk
from newton.solvers import SolverBase, SolverFeatherstone, SolverMuJoCo, SolverVBD, SolverXPBD
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM, SolverCoupledProxy

from isaaclab.managers import SceneEntityCfg
from isaaclab.physics import PhysicsManager
from isaaclab.utils.string import resolve_matching_names

from ..deformable.newton_manager_cfg import CoupledNewtonCfg, VBDSolverCfg
from ..deformable.vbd_manager import NewtonVBDManager
from ._entry_collision_pipeline import _EntryCollisionPipelineSolver
from .coupled_manager_cfg import (
    CoupledAdmmSolverCfg,
    CoupledProxyCfg,
    CoupledProxySolverCfg,
    CoupledSolverCfg,
    CoupledSolverEntryCfg,
)

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveSceneCfg


@wp.kernel(enable_backward=False)
def _and_bool_masks(
    left: wp.array(dtype=wp.bool),
    right: wp.array(dtype=wp.bool),
    result: wp.array(dtype=wp.bool),
):
    index = wp.tid()
    result[index] = left[index] and right[index]


class NewtonCoupledSolverManager(NewtonVBDManager):
    """Build and manage Newton proxy or ADMM coupling from named entries."""

    @dataclass
    class _ResolvedEntry:
        """Entry configuration with model selectors resolved to indices."""

        config: CoupledSolverEntryCfg
        bodies: list[int]
        particles: list[int]
        joints: list[int]
        shapes: list[int]

    @dataclass
    class _ResolvedProxy:
        """Proxy configuration with source selectors resolved to indices."""

        config: CoupledProxyCfg
        bodies: list[int]
        particles: list[int]

    _SOLVER_CLASS_BY_CFG_TYPE: ClassVar[dict[type[NewtonSolverCfg], type[SolverBase]]] = {
        MJWarpSolverCfg: SolverMuJoCo,
        VBDSolverCfg: SolverVBD,
        FeatherstoneSolverCfg: SolverFeatherstone,
        XPBDSolverCfg: SolverXPBD,
    }
    _fk_articulation_filter: wp.array | None = None
    _combined_fk_mask: wp.array | None = None

    @classmethod
    def _resolve_solver_class(cls, sub_cfg: NewtonSolverCfg) -> type[SolverBase]:
        """Resolve a supported Isaac Lab solver config to its Newton solver class."""
        try:
            return cls._SOLVER_CLASS_BY_CFG_TYPE[type(sub_cfg)]
        except KeyError:
            known = ", ".join(sorted(t.__name__ for t in cls._SOLVER_CLASS_BY_CFG_TYPE))
            raise ValueError(
                f"No Newton solver registered for sub-cfg type {type(sub_cfg).__name__!r}. Known config types: {known}."
            ) from None

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: CoupledSolverCfg) -> None:
        """Resolve ownership and construct the selected coupled solver."""
        outer_cfg = PhysicsManager._cfg
        scene_cfg = outer_cfg.scene_cfg if isinstance(outer_cfg, CoupledNewtonCfg) else None
        resolved_entries = [cls._resolve_entry(model, entry, scene_cfg) for entry in solver_cfg.entries]
        resolved_proxies = (
            [cls._resolve_proxy(model, proxy, scene_cfg) for proxy in solver_cfg.proxies]
            if isinstance(solver_cfg, CoupledProxySolverCfg)
            else []
        )
        cls._validate_solver_cfg(model, solver_cfg, resolved_entries, resolved_proxies)

        if isinstance(solver_cfg, CoupledProxySolverCfg):
            cls._apply_proxy_shape_overrides(model, resolved_proxies)

        needs_outer_pipeline = (
            solver_cfg.use_collision_pipeline
            if solver_cfg.use_collision_pipeline is not None
            else isinstance(solver_cfg, CoupledAdmmSolverCfg)
        )
        proxy_destinations = (
            {proxy.config.destination for proxy in resolved_proxies}
            if isinstance(solver_cfg, CoupledProxySolverCfg)
            else set()
        )
        entries = []
        for entry in resolved_entries:
            entry_solver_cfg = entry.config.solver_cfg
            needs_external_contacts = not (
                isinstance(entry_solver_cfg, MJWarpSolverCfg) and entry_solver_cfg.use_mujoco_contacts
            )
            entries.append(
                cls._build_entry(
                    entry,
                    local_collision=(
                        not needs_outer_pipeline
                        and needs_external_contacts
                        and entry.config.name not in proxy_destinations
                    ),
                )
            )
        if isinstance(solver_cfg, CoupledProxySolverCfg):
            NewtonManager._solver = cls._build_proxy_coupled_solver(model, entries, solver_cfg, resolved_proxies)
        elif isinstance(solver_cfg, CoupledAdmmSolverCfg):
            NewtonManager._solver = cls._build_admm_coupled_solver(model, entries, solver_cfg)
        else:
            raise TypeError(
                f"CoupledSolverCfg subclass {type(solver_cfg).__name__!r} is not supported; "
                "use CoupledProxySolverCfg or CoupledAdmmSolverCfg."
            )

        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = needs_outer_pipeline
        NewtonManager._supports_contact_sensors = False
        if NewtonManager._report_contacts:
            raise NotImplementedError(
                "Newton contact sensors are not yet supported by coupled solvers because contact forces live "
                "in per-entry buffers. Remove the contact sensor."
            )
        cls._configure_fk_articulation_filter(model, resolved_entries)

    @classmethod
    def _resolve_entry(
        cls,
        model: Model,
        entry_cfg: CoupledSolverEntryCfg,
        scene_cfg: InteractiveSceneCfg | None,
    ) -> _ResolvedEntry:
        """Resolve one entry's selectors and derived ownership."""
        bodies = cls._resolve_body_selectors(model, entry_cfg.bodies, scene_cfg, f"entry {entry_cfg.name!r}")
        particles = cls._unique_ints(entry_cfg.particles)
        if entry_cfg.all_particles:
            particles = cls._unique_ints([*particles, *range(int(model.particle_count))])

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
                    f"CoupledSolverEntryCfg {entry_cfg.name!r}: failed to resolve shape-label patterns."
                ) from error
            shapes.extend(labeled_shapes[index][0] for index in matched_shapes)

        return cls._ResolvedEntry(
            config=entry_cfg,
            bodies=bodies,
            particles=particles,
            joints=cls._unique_ints(joints),
            shapes=cls._unique_ints(shapes),
        )

    @classmethod
    def _resolve_proxy(
        cls,
        model: Model,
        proxy_cfg: CoupledProxyCfg,
        scene_cfg: InteractiveSceneCfg | None,
    ) -> _ResolvedProxy:
        """Resolve one proxy mapping's body selectors to collidable body ids."""
        selected = cls._resolve_body_selectors(
            model,
            proxy_cfg.bodies,
            scene_cfg,
            f"proxy {proxy_cfg.source!r}->{proxy_cfg.destination!r}",
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
                f"CoupledProxyCfg {proxy_cfg.source!r}->{proxy_cfg.destination!r} selected no bodies "
                "with ShapeFlags.COLLIDE_SHAPES."
            )
        return cls._ResolvedProxy(
            config=proxy_cfg,
            bodies=bodies,
            particles=cls._unique_ints(proxy_cfg.particles),
        )

    @classmethod
    def _resolve_body_selectors(
        cls,
        model: Model,
        selectors: list[SceneEntityCfg | str],
        scene_cfg: InteractiveSceneCfg | None,
        field: str,
    ) -> list[int]:
        body_ids: list[int] = []
        for selector in selectors:
            body_ids.extend(cls._resolve_entity_to_body_ids(model, selector, scene_cfg, field))
        return cls._unique_ints(body_ids)

    @classmethod
    def _resolve_entity_to_body_ids(
        cls,
        model: Model,
        spec: SceneEntityCfg | str,
        scene_cfg: InteractiveSceneCfg | None,
        field: str,
    ) -> list[int]:
        """Resolve one scene entity or full body-label regex."""
        labels = list(model.body_label)
        if isinstance(spec, str):
            body_ids, _ = resolve_matching_names(f"(?:{spec})(?:/.*)?", labels, raise_when_no_match=False)
            if not body_ids:
                raise ValueError(f"CoupledSolverCfg {field}: body-label regex {spec!r} matched no Newton bodies.")
            return body_ids

        asset_cfg = getattr(scene_cfg, spec.name, None) if scene_cfg is not None else None
        if asset_cfg is None or not hasattr(asset_cfg, "prim_path"):
            raise ValueError(f"CoupledSolverCfg {field}: scene entity {spec.name!r} is not on the attached scene cfg.")
        asset_body_ids, _ = resolve_matching_names(
            f"(?:{asset_cfg.prim_path})(?:/.*)?", labels, raise_when_no_match=False
        )
        if not asset_body_ids:
            raise ValueError(f"CoupledSolverCfg {field}: scene entity {spec.name!r} matched no Newton bodies.")
        if spec.body_names is None:
            return asset_body_ids

        body_patterns = [spec.body_names] if isinstance(spec.body_names, str) else spec.body_names
        short_names = [labels[index].rsplit("/", 1)[-1] for index in asset_body_ids]
        try:
            local_body_ids, _ = resolve_matching_names(body_patterns, short_names)
        except ValueError as error:
            raise ValueError(
                f"CoupledSolverCfg {field}: scene entity {spec.name!r} could not match body patterns {body_patterns}."
            ) from error
        return [asset_body_ids[index] for index in local_body_ids]

    @staticmethod
    def _unique_ints(values) -> list[int]:
        return list(dict.fromkeys(map(int, values)))

    @classmethod
    def _build_entry(cls, entry: _ResolvedEntry, *, local_collision: bool = False) -> SolverCoupled.Entry:
        entry_cfg = entry.config
        solver_cls = cls._resolve_solver_class(entry_cfg.solver_cfg)
        solver_kwargs = cls._filter_solver_kwargs(solver_cls, entry_cfg.solver_cfg)

        def solver_factory(
            model_view,
            _solver_cls=solver_cls,
            _kwargs=solver_kwargs,
            _local=local_collision,
            _use_solver_effective_mass=entry_cfg.use_solver_effective_mass,
            _soft_vbd_joints=isinstance(entry_cfg.solver_cfg, VBDSolverCfg)
            and not entry_cfg.solver_cfg.rigid_joint_hard,
        ):
            solver = _solver_cls(model=model_view, **_kwargs)
            if _soft_vbd_joints:
                cls._set_all_vbd_joints_soft(solver)
            return (
                _EntryCollisionPipelineSolver(
                    solver,
                    model_view,
                    use_solver_effective_mass=_use_solver_effective_mass,
                )
                if _local
                else solver
            )

        return SolverCoupled.Entry(
            name=entry_cfg.name,
            solver=solver_factory,
            bodies=entry.bodies,
            particles=entry.particles,
            joints=entry.joints,
            shapes=entry.shapes,
        )

    @classmethod
    def _build_proxy_coupled_solver(
        cls,
        model: Model,
        entries: list[SolverCoupled.Entry],
        solver_cfg: CoupledProxySolverCfg,
        proxies: list[_ResolvedProxy],
    ) -> SolverCoupledProxy:
        proxy_mappings = [
            SolverCoupledProxy.Proxy(
                source=proxy.config.source,
                destination=proxy.config.destination,
                bodies=proxy.bodies,
                particles=proxy.particles,
                mode=proxy.config.mode,
                mass_scale=float(proxy.config.mass_scale),
                collision_pipeline=proxy.config.collision_pipeline_factory
                or (lambda model_view: CollisionPipeline(model_view, broad_phase="explicit")),
                collide_interval=proxy.config.collide_interval,
            )
            for proxy in proxies
        ]
        return SolverCoupledProxy(
            model=model,
            entries=entries,
            coupling=SolverCoupledProxy.Config(proxies=proxy_mappings, iterations=int(solver_cfg.iterations)),
        )

    @classmethod
    def _build_admm_coupled_solver(
        cls,
        model: Model,
        entries: list[SolverCoupled.Entry],
        solver_cfg: CoupledAdmmSolverCfg,
    ) -> SolverCoupledADMM:
        contact_pairs = [
            SolverCoupledADMM.ContactPair(source=pair.source, destination=pair.destination)
            for pair in solver_cfg.contact_pairs
        ]
        coupling = SolverCoupledADMM.Config(
            iterations=int(solver_cfg.iterations),
            rho=float(solver_cfg.rho),
            gamma=float(solver_cfg.gamma),
            baumgarte=float(solver_cfg.baumgarte),
            joint_stiffness=float(solver_cfg.joint_stiffness),
            joint_damping=float(solver_cfg.joint_damping),
            joint_angular_stiffness=float(solver_cfg.joint_angular_stiffness),
            joint_angular_damping=float(solver_cfg.joint_angular_damping),
            joint_proximal_bodies=bool(solver_cfg.joint_proximal_bodies),
            joint_proximal_destination_entries=solver_cfg.joint_proximal_destination_entries,
            joint_proximal_mass_scale=float(solver_cfg.joint_proximal_mass_scale),
            rigid_contact_matching=solver_cfg.rigid_contact_matching,
            contact_matching_pos_threshold=solver_cfg.contact_matching_pos_threshold,
            contact_matching_normal_dot_threshold=solver_cfg.contact_matching_normal_dot_threshold,
            contact_matching_force_scale=float(solver_cfg.contact_matching_force_scale),
            contact_pairs=contact_pairs,
        )
        return SolverCoupledADMM(model=model, entries=entries, coupling=coupling)

    @classmethod
    def _validate_solver_cfg(
        cls,
        model: Model,
        solver_cfg: CoupledSolverCfg,
        entries: list[_ResolvedEntry],
        proxies: list[_ResolvedProxy] | None = None,
    ) -> None:
        if len(entries) < 2:
            raise ValueError("A coupled solver requires at least two named entries.")
        names = [entry.config.name for entry in entries]
        if any(not name for name in names):
            raise ValueError("CoupledSolverEntryCfg.name must be non-empty.")
        if len(set(names)) != len(names):
            raise ValueError(f"Coupled solver entry names must be unique, got {names!r}.")

        cls._validate_ownership(model, entries, "bodies", int(model.body_count), require_complete=True)
        cls._validate_ownership(model, entries, "particles", int(model.particle_count), require_complete=True)
        cls._validate_ownership(model, entries, "joints", int(model.joint_count))
        cls._validate_ownership(model, entries, "shapes", int(model.shape_count))

        if isinstance(solver_cfg, CoupledProxySolverCfg):
            if len(entries) > 2:
                raise ValueError("Newton proxy coupling currently supports at most two solver entries.")
            if solver_cfg.iterations < 1:
                raise ValueError("CoupledProxySolverCfg.iterations must be >= 1.")
            if not proxies:
                raise ValueError("CoupledProxySolverCfg requires at least one proxy mapping.")
            entries_by_name = {entry.config.name: entry for entry in entries}
            cls._validate_no_cross_entry_proxy_joints(model, entries_by_name)
            for proxy in proxies:
                cls._validate_proxy(proxy, entries_by_name)
        elif isinstance(solver_cfg, CoupledAdmmSolverCfg):
            if solver_cfg.iterations < 1:
                raise ValueError("CoupledAdmmSolverCfg.iterations must be >= 1.")
            for pair in solver_cfg.contact_pairs:
                if pair.source not in names or pair.destination not in names:
                    raise ValueError(
                        f"ADMM contact-pair endpoints {pair.source!r}->{pair.destination!r} must name coupled entries."
                    )
                if pair.source == pair.destination:
                    raise ValueError("ADMM contact-pair source and destination must differ.")

    @staticmethod
    def _validate_ownership(
        model: Model,
        entries: list[_ResolvedEntry],
        field: str,
        count: int,
        *,
        require_complete: bool = False,
    ) -> None:
        owners: dict[int, str] = {}
        for entry in entries:
            for index in getattr(entry, field, ()):
                if index < 0 or index >= count:
                    raise ValueError(f"Coupled entry {entry.config.name!r} owns out-of-range {field} index {index}.")
                if index in owners:
                    raise ValueError(
                        f"{field} index {index} is owned by both {owners[index]!r} and {entry.config.name!r}."
                    )
                owners[index] = entry.config.name
        if require_complete and (unclaimed := [index for index in range(count) if index not in owners]):
            labels = getattr(model, "body_label", None) if field == "bodies" else None
            preview = [labels[index] if labels is not None else index for index in unclaimed[:5]]
            raise ValueError(f"Coupled solver has {len(unclaimed)} unclaimed {field} (first few: {preview!r}).")

    @staticmethod
    def _validate_proxy(proxy: _ResolvedProxy, entries: dict[str, _ResolvedEntry]) -> None:
        proxy_cfg = proxy.config
        if proxy_cfg.source not in entries or proxy_cfg.destination not in entries:
            raise ValueError(
                f"CoupledProxyCfg endpoints {proxy_cfg.source!r}->{proxy_cfg.destination!r} must name coupled entries."
            )
        if proxy_cfg.source == proxy_cfg.destination:
            raise ValueError("CoupledProxyCfg source and destination must differ.")
        if not proxy.bodies and not proxy.particles:
            raise ValueError("CoupledProxyCfg must map at least one body or particle.")
        if not set(proxy.bodies).issubset(entries[proxy_cfg.source].bodies):
            raise ValueError("CoupledProxyCfg bodies must be owned by its source entry.")
        if not set(proxy.particles).issubset(entries[proxy_cfg.source].particles):
            raise ValueError("CoupledProxyCfg particles must be owned by its source entry.")
        if proxy_cfg.mass_scale <= 0.0:
            raise ValueError("CoupledProxyCfg.mass_scale must be > 0.")
        if proxy_cfg.collide_interval is not None and proxy_cfg.collide_interval < 1:
            raise ValueError("CoupledProxyCfg.collide_interval must be >= 1.")
        if proxy_cfg.mode not in ("lagged", "staggered"):
            raise ValueError("CoupledProxyCfg.mode must be 'lagged' or 'staggered'.")

    @staticmethod
    def _validate_no_cross_entry_proxy_joints(model: Model, entries: dict[str, _ResolvedEntry]) -> None:
        body_owner = {int(body): name for name, entry in entries.items() for body in entry.bodies}
        for joint, (parent_raw, child_raw) in enumerate(zip(model.joint_parent.numpy(), model.joint_child.numpy())):
            parent = int(parent_raw)
            child = int(child_raw)
            if parent >= 0 and body_owner[parent] != body_owner[child]:
                raise ValueError(
                    f"CoupledProxySolverCfg does not support cross-entry joint {joint} between "
                    f"{body_owner[parent]!r} and {body_owner[child]!r}; keep the articulation in one entry "
                    "or use ADMM coupling."
                )

    @classmethod
    def _apply_proxy_shape_overrides(cls, model: Model, proxies: list[_ResolvedProxy]) -> None:
        shape_bodies = model.shape_body.numpy()
        for proxy in proxies:
            body_set = set(proxy.bodies)
            shape_ids = [shape for shape, body in enumerate(shape_bodies) if int(body) in body_set]
            for name in ("shape_material_ke", "shape_material_kd", "shape_material_mu", "shape_margin", "shape_gap"):
                value = getattr(proxy.config, name)
                array = getattr(model, name, None)
                if value is not None and shape_ids and array is not None:
                    values = array.numpy()
                    values[np.asarray(shape_ids, dtype=np.int32)] = float(value)
                    array.assign(values)

    @classmethod
    def _configure_fk_articulation_filter(cls, model: Model, entries: list[_ResolvedEntry]) -> None:
        """Exclude VBD-owned articulations from generic reduced-coordinate FK."""
        if int(model.articulation_count) == 0 or getattr(model, "joint_articulation", None) is None:
            cls._fk_articulation_filter = None
            return
        allowed = np.ones(int(model.articulation_count), dtype=bool)
        joint_articulation = model.joint_articulation.numpy()
        vbd_joints: set[int] = set()
        for entry in entries:
            if isinstance(entry.config.solver_cfg, VBDSolverCfg):
                vbd_joints.update(entry.joints)
        for articulation in range(int(model.articulation_count)):
            articulation_joints = {
                joint for joint, owner in enumerate(joint_articulation) if int(owner) == articulation
            }
            if articulation_joints and articulation_joints.issubset(vbd_joints):
                allowed[articulation] = False
        cls._fk_articulation_filter = wp.array(allowed, dtype=wp.bool, device=model.device)
        cls._combined_fk_mask = wp.zeros_like(cls._fk_articulation_filter)

    @classmethod
    def _eval_fk_impl(cls, world_reset_mask: wp.array | None, fk_mask: wp.array | None) -> None:
        del world_reset_mask
        allowed = cls._fk_articulation_filter
        if allowed is None:
            mask = fk_mask
        elif fk_mask is None:
            mask = allowed
        else:
            if cls._combined_fk_mask is None or cls._combined_fk_mask.shape != fk_mask.shape:
                cls._combined_fk_mask = wp.zeros_like(fk_mask)
            wp.launch(
                _and_bool_masks,
                dim=fk_mask.shape[0],
                inputs=[fk_mask, allowed],
                outputs=[cls._combined_fk_mask],
                device=cls._model.device,
            )
            mask = cls._combined_fk_mask
        eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, mask)

    @classmethod
    def _reset_solver_internals(cls, world_mask: wp.array | None) -> None:
        """Update FK, distribute teleported state, and clear coupled history."""
        if world_mask is None:
            return

        with wp.ScopedDevice(PhysicsManager._device):
            cls._eval_fk_impl(world_mask, NewtonManager._fk_reset_mask)
            NewtonManager._solver.reset(NewtonManager._state_0, world_mask=world_mask, flags=0)

    @classmethod
    def _solver_specific_clear(cls):
        cls._fk_articulation_filter = None
        cls._combined_fk_mask = None
        super()._solver_specific_clear()
