# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coupled Newton multi-solver manager."""

from __future__ import annotations

from collections.abc import Callable

from newton import Model
from newton.solvers import SolverAdmmCoupled, SolverCoupled, SolverProxyCoupled

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


class NewtonCoupledManager(NewtonManager):
    """:class:`NewtonManager` specialization for Newton coupled solvers.

    The manager is intentionally thin: Isaac Lab owns lifecycle, state buffers,
    collision-pipeline refresh, and visualization, while Newton's coupled
    solvers own per-solver ``ModelView`` construction and
    cross-entry force or constraint exchange.
    """

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
    def _build_solver(cls, model: Model, solver_cfg: CoupledSolverCfg) -> None:
        """Construct a Newton coupled solver and populate the base-class slots."""
        cls._validate_solver_cfg(solver_cfg)

        entries = [cls._build_entry(entry_cfg) for entry_cfg in solver_cfg.entries]
        if solver_cfg.coupling_type == "proxy":
            NewtonManager._solver = SolverProxyCoupled(
                model=model,
                entries=entries,
                coupling=SolverProxyCoupled.Config(
                    proxies=[cls._build_proxy(proxy_cfg) for proxy_cfg in solver_cfg.proxy_coupling.proxies],
                    iterations=solver_cfg.proxy_coupling.iterations,
                ),
            )
        elif solver_cfg.coupling_type == "admm":
            NewtonManager._solver = SolverAdmmCoupled(
                model=model,
                entries=entries,
                coupling=cls._build_admm(solver_cfg.admm_coupling, entries),
            )
        else:
            raise ValueError(f"Unsupported Newton coupling_type {solver_cfg.coupling_type!r}.")

        cls._apply_entry_solver_overrides(solver_cfg.entries)
        if hasattr(NewtonManager._solver, "prepare_graph_capture"):
            NewtonManager._solver.prepare_graph_capture()
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = cls._needs_external_collision_pipeline(solver_cfg)

    @classmethod
    def _apply_entry_solver_overrides(cls, entries: list[CoupledSolverEntryCfg]) -> None:
        """Apply post-construction solver cfg overrides for coupled sub-solvers."""
        for entry_cfg in entries:
            if getattr(entry_cfg.solver_cfg, "solver_type", None) != "mujoco_warp":
                continue
            apply_mujoco_warp_model_overrides(NewtonManager._solver.solver(entry_cfg.name), entry_cfg.solver_cfg)

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
            solver=cls._make_entry_solver_factory(solver_class, solver_kwargs),
            bodies=list(entry_cfg.bodies),
            particles=list(entry_cfg.particles),
            joints=list(entry_cfg.joints),
            shapes=list(entry_cfg.shapes),
            configure_view=configure_view,
            substeps=entry_cfg.substeps,
            in_place=entry_cfg.in_place,
        )

        return SolverCoupled.Entry(**entry_kwargs)

    @staticmethod
    def _make_entry_solver_factory(solver_class: Callable, solver_kwargs: dict) -> Callable:
        """Bind constructor kwargs into a Newton coupled entry solver factory."""

        def _factory(model_view):
            return solver_class(model_view, **solver_kwargs)

        _factory.__name__ = getattr(solver_class, "__name__", type(solver_class).__name__)
        return _factory

    @staticmethod
    def _build_proxy(proxy_cfg: CoupledProxyCfg) -> SolverProxyCoupled.Proxy:
        """Build a Newton proxy mapping from an Isaac Lab proxy cfg."""
        if not proxy_cfg.source or not proxy_cfg.destination:
            raise ValueError("CoupledProxyCfg source and destination must be non-empty.")
        if not proxy_cfg.bodies and not proxy_cfg.particles:
            raise ValueError("CoupledProxyCfg must map at least one body or particle.")

        return SolverProxyCoupled.Proxy(
            source=proxy_cfg.source,
            destination=proxy_cfg.destination,
            bodies=list(proxy_cfg.bodies),
            proxy_bodies=None if proxy_cfg.proxy_bodies is None else list(proxy_cfg.proxy_bodies),
            mass_scale=proxy_cfg.mass_scale,
            mode=proxy_cfg.mode,
            particles=list(proxy_cfg.particles),
            proxy_particles=None if proxy_cfg.proxy_particles is None else list(proxy_cfg.proxy_particles),
            collision_pipeline=proxy_cfg.collision_pipeline_factory,
            collide_interval=proxy_cfg.collide_interval,
        )

    @classmethod
    def _build_admm(
        cls, admm_cfg: AdmmCouplingCfg, entries: list[SolverCoupled.Entry] | None = None
    ) -> SolverAdmmCoupled.Config:
        """Build a Newton ADMM coupling config from an Isaac Lab cfg."""
        contact_pairs = [cls._build_admm_contact_pair(pair_cfg) for pair_cfg in admm_cfg.contact_pairs]
        if admm_cfg.auto_contact_pairs:
            if entries is None:
                raise ValueError("AdmmCouplingCfg.auto_contact_pairs requires coupled solver entries.")
            contact_pairs.extend(
                SolverAdmmCoupled.auto_detect_contact_pairs(
                    entries,
                    contact_distance=admm_cfg.auto_contact_distance,
                    detection_margin=admm_cfg.auto_detection_margin,
                )
            )

        return SolverAdmmCoupled.Config(
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
    def _build_admm_contact_pair(pair_cfg: AdmmContactPairCfg) -> SolverAdmmCoupled.ContactPair:
        """Build a Newton ADMM contact-pair config from an Isaac Lab cfg."""
        return SolverAdmmCoupled.ContactPair(
            source=pair_cfg.source,
            destination=pair_cfg.destination,
            contact_distance=pair_cfg.contact_distance,
            detection_margin=pair_cfg.detection_margin,
        )

    @classmethod
    def _validate_solver_cfg(cls, solver_cfg: CoupledSolverCfg) -> None:
        """Validate coupled-solver config before constructing Newton objects."""
        if solver_cfg.coupling_type not in ("proxy", "admm"):
            raise ValueError(f"Unsupported Newton coupling_type {solver_cfg.coupling_type!r}.")
        if len(solver_cfg.entries) < 2:
            raise ValueError("Newton coupled solver requires at least two solver entries.")
        cls._validate_entries(solver_cfg.entries)
        if solver_cfg.coupling_type == "proxy":
            cls._validate_proxy_coupling(solver_cfg)
        else:
            cls._validate_admm_coupling(solver_cfg.admm_coupling)

    @staticmethod
    def _validate_entries(entries: list[CoupledSolverEntryCfg]) -> None:
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

    @classmethod
    def _validate_proxy_coupling(cls, solver_cfg: CoupledSolverCfg) -> None:
        if len(solver_cfg.entries) > 2:
            raise ValueError("Newton proxy coupling currently supports at most two solver entries.")
        if not solver_cfg.proxy_coupling.proxies:
            raise ValueError("Newton proxy coupling requires at least one proxy mapping.")
        if solver_cfg.proxy_coupling.iterations < 1:
            raise ValueError("ProxyCouplingCfg.iterations must be >= 1.")
        entry_names = {entry.name for entry in solver_cfg.entries}
        for proxy in solver_cfg.proxy_coupling.proxies:
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
        return any(solver_cfg_needs_external_contacts(entry.solver_cfg) for entry in solver_cfg.entries)
