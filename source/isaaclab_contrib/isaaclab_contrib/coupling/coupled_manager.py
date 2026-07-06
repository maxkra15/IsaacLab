# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton manager for named coupled-solver configurations."""

from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import warp as wp
from isaaclab_newton.physics import (
    FeatherstoneSolverCfg,
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonSolverCfg,
    XPBDSolverCfg,
)
from isaaclab_newton.physics.mjwarp_manager import apply_mujoco_warp_model_overrides
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import BodyFlags, CollisionPipeline, Contacts, Control, Model, ModelBuilder, ShapeFlags, State, eval_fk
from newton._src.solvers.coupled.proxy_utils import sync_proxy_particles_kernel, sync_proxy_states_kernel
from newton.solvers import SolverBase, SolverFeatherstone, SolverImplicitMPM, SolverMuJoCo, SolverVBD, SolverXPBD
from newton.solvers.experimental.coupled import CouplingInterface, SolverCoupled, SolverCoupledADMM, SolverCoupledProxy
from warp.fem import TemporaryStore

from isaaclab.managers import SceneEntityCfg
from isaaclab.physics import PhysicsManager

from ..deformable.newton_manager_cfg import CoupledNewtonCfg, VBDSolverCfg
from ..deformable.vbd_manager import NewtonVBDManager
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


def _default_proxy_collision_pipeline(model_view):
    """Build the explicit proxy-local pipeline used by the original two-entry manager."""
    return CollisionPipeline(model_view, broad_phase="explicit")


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


class NewtonCoupledSolverManager(NewtonVBDManager):
    """Build and manage Newton proxy or ADMM coupling from named entries."""

    _SOLVER_CLASS_BY_CFG_TYPE: ClassVar[dict[type[NewtonSolverCfg], type[SolverBase]]] = {
        MJWarpSolverCfg: SolverMuJoCo,
        MPMSolverCfg: SolverImplicitMPM,
        VBDSolverCfg: SolverVBD,
        FeatherstoneSolverCfg: SolverFeatherstone,
        XPBDSolverCfg: SolverXPBD,
    }
    _fk_articulation_filter: wp.array | None = None
    _combined_fk_mask: wp.array | None = None
    _mpm_project_outside_entries: tuple[str, ...] = ()

    @classmethod
    def get_entry_solver(cls, name: str):
        """Return a named sub-solver from the active coupled solver."""
        if NewtonManager._solver is None:
            raise RuntimeError("Newton coupled solver is not initialized.")
        return NewtonManager._solver.solver(name)

    @classmethod
    def get_entry_view(cls, name: str):
        """Return a named sub-solver model view from the active coupled solver."""
        if NewtonManager._solver is None:
            raise RuntimeError("Newton coupled solver is not initialized.")
        return NewtonManager._solver.view(name)

    @classmethod
    def get_proxy_body_wrenches(cls, source: str, destination: str):
        """Return body feedback wrenches for one proxy mapping, when available."""
        solver = NewtonManager._solver
        if solver is None:
            return None
        for mapping in getattr(solver, "_proxy_mappings", ()):
            if mapping.src_name == source and mapping.dst_name == destination:
                return mapping.coupling_forces
        return None

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
        resolved_cfg = cls._resolve_solver_cfg(model, solver_cfg)
        cls._validate_solver_cfg(model, resolved_cfg)

        if isinstance(resolved_cfg, CoupledProxySolverCfg):
            cls._apply_proxy_shape_overrides(model, resolved_cfg.proxies)

        needs_outer_pipeline = (
            resolved_cfg.use_collision_pipeline
            if resolved_cfg.use_collision_pipeline is not None
            else (
                isinstance(resolved_cfg, CoupledAdmmSolverCfg)
                or (
                    type(resolved_cfg) is CoupledSolverCfg
                    and any(cls._solver_cfg_needs_external_contacts(entry.solver_cfg) for entry in resolved_cfg.entries)
                )
            )
        )
        proxy_destinations = (
            {proxy.destination for proxy in resolved_cfg.proxies}
            if isinstance(resolved_cfg, CoupledProxySolverCfg)
            else set()
        )
        entries = [
            cls._build_entry(
                entry_cfg,
                local_collision=(
                    not needs_outer_pipeline
                    and cls._solver_cfg_needs_external_contacts(entry_cfg.solver_cfg)
                    and entry_cfg.name not in proxy_destinations
                ),
            )
            for entry_cfg in resolved_cfg.entries
        ]
        if type(resolved_cfg) is CoupledSolverCfg:
            NewtonManager._solver = SolverCoupled(model=model, entries=entries)
        elif isinstance(resolved_cfg, CoupledProxySolverCfg):
            NewtonManager._solver = cls._build_proxy_coupled_solver(model, entries, resolved_cfg)
        elif isinstance(resolved_cfg, CoupledAdmmSolverCfg):
            NewtonManager._solver = cls._build_admm_coupled_solver(model, entries, resolved_cfg)
        else:
            raise TypeError(
                f"CoupledSolverCfg subclass {type(resolved_cfg).__name__!r} is not supported; "
                "use CoupledSolverCfg, CoupledProxySolverCfg, or CoupledAdmmSolverCfg."
            )

        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = needs_outer_pipeline
        NewtonManager._needs_fk_before_step = any(
            isinstance(entry.solver_cfg, MPMSolverCfg) for entry in resolved_cfg.entries
        )
        NewtonManager._requires_teleport_reset = True
        NewtonManager._supports_contact_sensors = False
        if NewtonManager._report_contacts:
            raise NotImplementedError(
                "Newton contact sensors are not yet supported by coupled solvers because contact forces live "
                "in per-entry buffers. Remove the contact sensor."
            )
        cls._apply_vbd_joint_constraint_modes(resolved_cfg.entries)
        cls._configure_fk_articulation_filter(model, resolved_cfg.entries)
        cls._mpm_project_outside_entries = tuple(
            entry.name
            for entry in resolved_cfg.entries
            if isinstance(entry.solver_cfg, MPMSolverCfg) and entry.solver_cfg.project_outside_colliders
        )

    @classmethod
    def _prepare_builder_for_finalize(cls, builder: ModelBuilder) -> None:
        """Normalize kinematic colliders when an entry uses implicit MPM."""
        super()._prepare_builder_for_finalize(builder)
        physics_cfg = PhysicsManager._cfg
        solver_cfg = getattr(physics_cfg, "solver_cfg", None)
        if not any(isinstance(entry.solver_cfg, MPMSolverCfg) for entry in getattr(solver_cfg, "entries", ())):
            return

        kinematic_flag = int(BodyFlags.KINEMATIC)
        for body_id, flags in enumerate(builder.body_flags):
            if int(flags) & kinematic_flag:
                builder.body_mass[body_id] = 0.0
                builder.body_inv_mass[body_id] = 0.0
                builder.body_inertia[body_id] = wp.mat33()
                builder.body_inv_inertia[body_id] = wp.mat33()

    @classmethod
    def _register_builder_attributes(cls, builder: ModelBuilder) -> None:
        """Register custom attributes required by concrete coupled algorithms."""
        super()._register_builder_attributes(builder)
        physics_cfg = PhysicsManager._cfg
        if isinstance(getattr(physics_cfg, "solver_cfg", None), CoupledAdmmSolverCfg):
            SolverCoupledADMM.register_custom_attributes(builder)

    @classmethod
    def _resolve_solver_cfg(cls, model: Model, solver_cfg: CoupledSolverCfg) -> CoupledSolverCfg:
        """Return a shallow copy with entry and proxy selectors resolved to indices."""
        resolved = copy.copy(solver_cfg)
        scene_cfg = cls._resolve_scene_cfg()
        resolved.entries = [cls._resolve_entry_cfg(model, entry, scene_cfg) for entry in solver_cfg.entries]
        if isinstance(solver_cfg, CoupledProxySolverCfg):
            resolved.proxies = [cls._resolve_proxy_cfg(model, proxy, scene_cfg) for proxy in solver_cfg.proxies]
        return resolved

    @staticmethod
    def _resolve_scene_cfg():
        outer_cfg = PhysicsManager._cfg
        return outer_cfg.scene_cfg if isinstance(outer_cfg, CoupledNewtonCfg) else None

    @classmethod
    def _resolve_entry_cfg(
        cls,
        model: Model,
        entry_cfg: CoupledSolverEntryCfg,
        scene_cfg: InteractiveSceneCfg | None,
    ) -> CoupledSolverEntryCfg:
        """Resolve one entry's selectors and derived ownership."""
        resolved = copy.copy(entry_cfg)
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
        shapes.extend(cls._resolve_shape_label_patterns(model, entry_cfg.shape_label_patterns, entry_cfg.name))

        resolved.bodies = bodies
        resolved.particles = particles
        resolved.joints = cls._unique_ints(joints)
        resolved.shapes = cls._unique_ints(shapes)
        return resolved

    @classmethod
    def _resolve_proxy_cfg(
        cls,
        model: Model,
        proxy_cfg: CoupledProxyCfg,
        scene_cfg: InteractiveSceneCfg | None,
    ) -> CoupledProxyCfg:
        """Resolve one proxy mapping's body selectors to collidable body ids."""
        resolved = copy.copy(proxy_cfg)
        selected = cls._resolve_body_selectors(
            model,
            proxy_cfg.bodies,
            scene_cfg,
            f"proxy {proxy_cfg.source!r}->{proxy_cfg.destination!r}",
        )
        collide_bodies = cls._collidable_body_ids(model)
        resolved.bodies = [body for body in selected if body in collide_bodies]
        resolved.particles = cls._unique_ints(proxy_cfg.particles)
        if proxy_cfg.bodies and not resolved.bodies:
            raise ValueError(
                f"CoupledProxyCfg {proxy_cfg.source!r}->{proxy_cfg.destination!r} selected no bodies "
                "with ShapeFlags.COLLIDE_SHAPES."
            )
        return resolved

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
        if isinstance(spec, str):
            asset_pattern = spec
            body_patterns = None
            description = f"body-label regex {spec!r}"
        else:
            asset_cfg = getattr(scene_cfg, spec.name, None) if scene_cfg is not None else None
            if asset_cfg is None or not hasattr(asset_cfg, "prim_path"):
                raise ValueError(
                    f"CoupledSolverCfg {field}: scene entity {spec.name!r} is not on the attached scene cfg."
                )
            asset_pattern = str(asset_cfg.prim_path).replace("{ENV_REGEX_NS}", r"/World/envs/env_.*")
            body_patterns = [spec.body_names] if isinstance(spec.body_names, str) else spec.body_names
            description = f"scene entity {spec.name!r}"

        asset_regex = re.compile(rf"^{asset_pattern}(/|$)")
        compiled = [re.compile(pattern) for pattern in (body_patterns or [r".*"])]
        matched_patterns = [False] * len(compiled)
        body_ids: list[int] = []
        for body, label in enumerate(model.body_label):
            if not asset_regex.match(label):
                continue
            short_name = label.rsplit("/", 1)[-1]
            for pattern_index, pattern in enumerate(compiled):
                if pattern.fullmatch(short_name):
                    matched_patterns[pattern_index] = True
                    body_ids.append(body)
                    break

        if body_patterns is not None:
            unmatched = [pattern for pattern, matched in zip(body_patterns, matched_patterns) if not matched]
            if unmatched:
                raise ValueError(f"CoupledSolverCfg {field}: {description} has no bodies matching {unmatched}.")
        elif not body_ids:
            raise ValueError(f"CoupledSolverCfg {field}: {description} matched no Newton bodies.")
        return body_ids

    @staticmethod
    def _resolve_shape_label_patterns(model: Model, patterns: list[str], entry_name: str) -> list[int]:
        labels = list(getattr(model, "shape_label", ()) or ())
        result: list[int] = []
        for pattern_text in patterns:
            pattern = re.compile(pattern_text)
            matched = [index for index, label in enumerate(labels) if label is not None and pattern.fullmatch(label)]
            if not matched:
                raise ValueError(
                    f"CoupledSolverEntryCfg {entry_name!r}: shape-label pattern {pattern_text!r} matched no shapes."
                )
            result.extend(matched)
        return result

    @staticmethod
    def _collidable_body_ids(model: Model) -> set[int]:
        collide_flag = int(ShapeFlags.COLLIDE_SHAPES)
        return {
            int(body)
            for body, flags in zip(model.shape_body.numpy(), model.shape_flags.numpy())
            if int(body) >= 0 and int(flags) & collide_flag
        }

    @staticmethod
    def _unique_ints(values) -> list[int]:
        result: list[int] = []
        seen: set[int] = set()
        for value in values:
            index = int(value)
            if index not in seen:
                result.append(index)
                seen.add(index)
        return result

    @classmethod
    def _build_entry(cls, entry_cfg: CoupledSolverEntryCfg, *, local_collision: bool = False) -> SolverCoupled.Entry:
        solver_cls = cls._resolve_solver_class(entry_cfg.solver_cfg)
        solver_kwargs = (
            None
            if isinstance(entry_cfg.solver_cfg, MPMSolverCfg)
            else cls._filter_solver_kwargs(solver_cls, entry_cfg.solver_cfg)
        )

        def solver_factory(
            model_view,
            _solver_cls=solver_cls,
            _kwargs=solver_kwargs,
            _solver_cfg=entry_cfg.solver_cfg,
            _local=local_collision,
            _use_solver_effective_mass=entry_cfg.use_solver_effective_mass,
        ):
            if isinstance(_solver_cfg, MPMSolverCfg):
                solver = _solver_cls(
                    model_view,
                    _solver_cfg.to_solver_config(),
                    temporary_store=TemporaryStore(),
                )
            else:
                solver = _solver_cls(model=model_view, **_kwargs)
                if isinstance(_solver_cfg, MJWarpSolverCfg):
                    apply_mujoco_warp_model_overrides(solver, _solver_cfg)
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
            bodies=list(entry_cfg.bodies),
            particles=list(entry_cfg.particles),
            joints=list(getattr(entry_cfg, "joints", ())),
            shapes=list(getattr(entry_cfg, "shapes", ())),
            substeps=int(entry_cfg.substeps),
            in_place=bool(entry_cfg.in_place),
        )

    @staticmethod
    def _solver_cfg_needs_external_contacts(solver_cfg: NewtonSolverCfg) -> bool:
        if isinstance(solver_cfg, MJWarpSolverCfg):
            return not solver_cfg.use_mujoco_contacts
        return isinstance(solver_cfg, (VBDSolverCfg, XPBDSolverCfg, FeatherstoneSolverCfg))

    @classmethod
    def _build_proxy_coupled_solver(
        cls,
        model: Model,
        entries: list[SolverCoupled.Entry],
        solver_cfg: CoupledProxySolverCfg,
    ) -> SolverCoupledProxy:
        entry_cfgs = {entry.name: entry for entry in solver_cfg.entries}
        proxies = [
            SolverCoupledProxy.Proxy(
                source=proxy.source,
                destination=proxy.destination,
                bodies=list(proxy.bodies),
                particles=list(proxy.particles),
                mode=proxy.mode,
                mass_scale=float(proxy.mass_scale),
                proxy_relaxation=float(proxy.proxy_relaxation),
                collision_pipeline=(
                    proxy.collision_pipeline_factory
                    or (
                        _default_proxy_collision_pipeline
                        if proxy.destination not in entry_cfgs
                        or cls._solver_cfg_needs_external_contacts(entry_cfgs[proxy.destination].solver_cfg)
                        else None
                    )
                ),
                collide_interval=proxy.collide_interval,
            )
            for proxy in solver_cfg.proxies
        ]
        return SolverCoupledProxy(
            model=model,
            entries=entries,
            coupling=SolverCoupledProxy.Config(proxies=proxies, iterations=int(solver_cfg.iterations)),
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
    def _validate_solver_cfg(cls, model: Model, solver_cfg: CoupledSolverCfg) -> None:
        if len(solver_cfg.entries) < 2:
            raise ValueError("A coupled solver requires at least two named entries.")
        names = [entry.name for entry in solver_cfg.entries]
        if any(not name for name in names):
            raise ValueError("CoupledSolverEntryCfg.name must be non-empty.")
        if len(set(names)) != len(names):
            raise ValueError(f"Coupled solver entry names must be unique, got {names!r}.")
        for entry in solver_cfg.entries:
            if entry.substeps < 1:
                raise ValueError(f"CoupledSolverEntryCfg {entry.name!r} substeps must be >= 1.")
            if entry.in_place and entry.substeps != 1:
                raise ValueError(f"CoupledSolverEntryCfg {entry.name!r} in_place requires substeps=1.")

        cls._validate_complete_ownership(model, solver_cfg.entries, "bodies", int(model.body_count))
        cls._validate_complete_ownership(model, solver_cfg.entries, "particles", int(model.particle_count))
        cls._validate_unique_ownership(solver_cfg.entries, "joints")
        cls._validate_unique_ownership(solver_cfg.entries, "shapes")

        if isinstance(solver_cfg, CoupledProxySolverCfg):
            if len(solver_cfg.entries) > 2:
                raise ValueError("Newton proxy coupling currently supports at most two solver entries.")
            if solver_cfg.iterations < 1:
                raise ValueError("CoupledProxySolverCfg.iterations must be >= 1.")
            if not solver_cfg.proxies:
                raise ValueError("CoupledProxySolverCfg requires at least one proxy mapping.")
            entries = {entry.name: entry for entry in solver_cfg.entries}
            cls._validate_no_cross_entry_proxy_joints(model, entries)
            for proxy in solver_cfg.proxies:
                cls._validate_proxy(proxy, entries)
        elif isinstance(solver_cfg, CoupledAdmmSolverCfg):
            cls._validate_admm(solver_cfg, set(names))

    @classmethod
    def _validate_complete_ownership(
        cls,
        model: Model,
        entries: list[CoupledSolverEntryCfg],
        field: str,
        count: int,
    ) -> None:
        owners: dict[int, str] = {}
        for entry in entries:
            for index in getattr(entry, field):
                if index < 0 or index >= count:
                    raise ValueError(f"Coupled entry {entry.name!r} owns out-of-range {field} index {index}.")
                if index in owners:
                    raise ValueError(f"{field} index {index} is owned by both {owners[index]!r} and {entry.name!r}.")
                owners[index] = entry.name
        if unclaimed := [index for index in range(count) if index not in owners]:
            labels = getattr(model, "body_label", None) if field == "bodies" else None
            preview = [labels[index] if labels is not None else index for index in unclaimed[:5]]
            raise ValueError(f"Coupled solver has {len(unclaimed)} unclaimed {field} (first few: {preview!r}).")

    @staticmethod
    def _validate_unique_ownership(entries: list[CoupledSolverEntryCfg], field: str) -> None:
        owners: dict[int, str] = {}
        for entry in entries:
            for index in getattr(entry, field, ()):
                if index in owners:
                    raise ValueError(f"{field} index {index} is owned by both {owners[index]!r} and {entry.name!r}.")
                owners[index] = entry.name

    @staticmethod
    def _validate_proxy(proxy: CoupledProxyCfg, entries: dict[str, CoupledSolverEntryCfg]) -> None:
        if proxy.source not in entries or proxy.destination not in entries:
            raise ValueError(
                f"CoupledProxyCfg endpoints {proxy.source!r}->{proxy.destination!r} must name coupled entries."
            )
        if proxy.source == proxy.destination:
            raise ValueError("CoupledProxyCfg source and destination must differ.")
        if not proxy.bodies and not proxy.particles:
            raise ValueError("CoupledProxyCfg must map at least one body or particle.")
        if not set(proxy.bodies).issubset(entries[proxy.source].bodies):
            raise ValueError("CoupledProxyCfg bodies must be owned by its source entry.")
        if not set(proxy.particles).issubset(entries[proxy.source].particles):
            raise ValueError("CoupledProxyCfg particles must be owned by its source entry.")
        if proxy.mass_scale <= 0.0:
            raise ValueError("CoupledProxyCfg.mass_scale must be > 0.")
        if not np.isfinite(proxy.proxy_relaxation) or proxy.proxy_relaxation < 0.0:
            raise ValueError("CoupledProxyCfg.proxy_relaxation must be finite and >= 0.")
        if proxy.collide_interval is not None and proxy.collide_interval < 1:
            raise ValueError("CoupledProxyCfg.collide_interval must be >= 1.")
        if proxy.mode not in ("lagged", "staggered"):
            raise ValueError("CoupledProxyCfg.mode must be 'lagged' or 'staggered'.")

    @staticmethod
    def _validate_no_cross_entry_proxy_joints(model: Model, entries: dict[str, CoupledSolverEntryCfg]) -> None:
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

    @staticmethod
    def _validate_admm(solver_cfg: CoupledAdmmSolverCfg, entry_names: set[str]) -> None:
        if solver_cfg.iterations < 1:
            raise ValueError("CoupledAdmmSolverCfg.iterations must be >= 1.")
        for pair in solver_cfg.contact_pairs:
            if pair.source not in entry_names or pair.destination not in entry_names:
                raise ValueError(
                    f"ADMM contact-pair endpoints {pair.source!r}->{pair.destination!r} must name coupled entries."
                )
            if pair.source == pair.destination:
                raise ValueError("ADMM contact-pair source and destination must differ.")

    @staticmethod
    def _set_model_array_indices(model: Model, name: str, indices: list[int], value: float | None) -> None:
        if value is None or not indices:
            return
        array = getattr(model, name, None)
        if array is None:
            return
        values = array.numpy()
        values[np.asarray(indices, dtype=np.int32)] = float(value)
        array.assign(values)

    @classmethod
    def _apply_proxy_shape_overrides(cls, model: Model, proxies: list[CoupledProxyCfg]) -> None:
        shape_bodies = model.shape_body.numpy()
        for proxy in proxies:
            body_set = set(proxy.bodies)
            shape_ids = [shape for shape, body in enumerate(shape_bodies) if int(body) in body_set]
            cls._set_model_array_indices(model, "shape_material_ke", shape_ids, proxy.shape_material_ke)
            cls._set_model_array_indices(model, "shape_material_kd", shape_ids, proxy.shape_material_kd)
            cls._set_model_array_indices(model, "shape_material_mu", shape_ids, proxy.shape_material_mu)
            cls._set_model_array_indices(model, "shape_margin", shape_ids, proxy.shape_margin)
            cls._set_model_array_indices(model, "shape_gap", shape_ids, proxy.shape_gap)

    @classmethod
    def _apply_vbd_joint_constraint_modes(cls, entries: list[CoupledSolverEntryCfg]) -> None:
        """Apply the optional Isaac Lab VBD all-joints soft-mode setting."""
        for entry in entries:
            if not isinstance(entry.solver_cfg, VBDSolverCfg) or getattr(entry.solver_cfg, "rigid_joint_hard", True):
                continue
            cls._set_all_vbd_joints_soft(NewtonManager._solver.solver(entry.name))

    @classmethod
    def _configure_fk_articulation_filter(cls, model: Model, entries: list[CoupledSolverEntryCfg]) -> None:
        """Exclude VBD-owned articulations from generic reduced-coordinate FK."""
        if int(model.articulation_count) == 0 or getattr(model, "joint_articulation", None) is None:
            cls._fk_articulation_filter = None
            return
        allowed = np.ones(int(model.articulation_count), dtype=bool)
        joint_articulation = model.joint_articulation.numpy()
        vbd_joints: set[int] = set()
        for entry in entries:
            if isinstance(entry.solver_cfg, VBDSolverCfg):
                vbd_joints.update(int(joint) for joint in getattr(entry, "joints", ()))
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
    def _step_solver(
        cls,
        state_0: State,
        state_1: State,
        control: Control,
        contacts: Contacts | None,
        substep_dt: float,
    ) -> None:
        """Run one coupled step and configured MPM particle projections."""
        super()._step_solver(state_0, state_1, control, contacts, substep_dt)
        solver = NewtonManager._solver
        reconcile_entry = getattr(solver, "reconcile_entry_state", None)
        needs_full_reconcile = False
        for entry_name in cls._mpm_project_outside_entries:
            entry_solver = solver.solver(entry_name)
            entry_state = solver.entry_state(entry_name, phase="output")
            entry_solver.project_outside(entry_state, entry_state, substep_dt)
            if callable(reconcile_entry):
                reconcile_entry(entry_name, state_1, phase="output")
            else:
                needs_full_reconcile = True
        if needs_full_reconcile:
            solver._reconcile_state(state_1)

    @classmethod
    def step(cls) -> None:
        """Reset history after state teleports, then run the normal Newton step."""
        sim = PhysicsManager._sim
        if NewtonManager._state_teleport_pending and sim is not None and sim.is_playing():
            cls._reset_coupled_solver_history()
        super().step()

    @classmethod
    def _reset_coupled_solver_history(cls) -> None:
        """Distribute teleported state and clear sub-solver/coupling history."""
        solver = NewtonManager._solver
        state = NewtonManager._state_0
        if solver is None or state is None:
            return

        with wp.ScopedDevice(PhysicsManager._device):
            cls._eval_fk_impl(NewtonManager._world_reset_mask, NewtonManager._fk_reset_mask)
            for entry in solver._entries.values():
                entry_solver = getattr(entry.solver, "_solver", entry.solver)
                if callable(getattr(entry_solver, "rebuild_bvh", None)) and not hasattr(
                    entry_solver, "particle_enable_self_contact"
                ):
                    entry_solver.particle_enable_self_contact = False
            solver.reset(state, world_mask=NewtonManager._world_reset_mask, flags=0)

            for proxy in getattr(solver, "_proxy_mappings", ()):
                source = solver._entries[proxy.src_name]
                destination = solver._entries[proxy.dst_name]
                for destination_state in (destination.state_0, destination.state_1):
                    if destination_state is not None:
                        wp.launch(
                            sync_proxy_states_kernel,
                            dim=proxy.source_local_to_proxy_local.shape[0],
                            inputs=[
                                source.state_0.body_q,
                                source.state_0.body_qd,
                                proxy.source_local_to_proxy_local,
                                destination_state.body_q,
                                destination_state.body_qd,
                            ],
                            device=cls._model.device,
                        )
            for proxy in getattr(solver, "_proxy_particle_mappings", ()):
                source = solver._entries[proxy.src_name]
                destination = solver._entries[proxy.dst_name]
                for destination_state in (destination.state_0, destination.state_1):
                    if destination_state is not None:
                        wp.launch(
                            sync_proxy_particles_kernel,
                            dim=proxy.source_local_to_proxy_local.shape[0],
                            inputs=[
                                source.state_0.particle_q,
                                source.state_0.particle_qd,
                                proxy.source_local_to_proxy_local,
                                destination_state.particle_q,
                                destination_state.particle_qd,
                            ],
                            device=cls._model.device,
                        )
            for entry in solver._entries.values():
                body_q_prev = getattr(entry.solver, "body_q_prev", None)
                if body_q_prev is not None:
                    wp.copy(dest=body_q_prev, src=entry.state_0.body_q)

    @classmethod
    def _solver_specific_clear(cls):
        cls._fk_articulation_filter = None
        cls._combined_fk_mask = None
        cls._mpm_project_outside_entries = ()
        super()._solver_specific_clear()
