# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coupled Newton multi-solver manager."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable

from newton import Model
from newton.solvers import (
    SolverFeatherstone,
    SolverImplicitMPM,
    SolverKamino,
    SolverMuJoCo,
    SolverProxyCoupled,
    SolverXPBD,
)

from .coupled_manager_cfg import CoupledProxyCfg, CoupledSolverCfg, CoupledSolverEntryCfg
from .mjwarp_manager import resolve_mujoco_solver_kwargs
from .newton_manager import NewtonManager
from .newton_manager_cfg import NewtonSolverCfg


class NewtonCoupledManager(NewtonManager):
    """:class:`NewtonManager` specialization for Newton coupled solvers.

    The manager is intentionally thin: Isaac Lab owns lifecycle, state buffers,
    collision-pipeline refresh, and visualization, while Newton's
    ``SolverProxyCoupled`` owns per-solver ``ModelView`` construction and
    proxy-force exchange.
    """

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: CoupledSolverCfg) -> None:
        """Construct a Newton coupled solver and populate the base-class slots."""
        if solver_cfg.coupling_type != "proxy":
            raise ValueError(f"Unsupported Newton coupling_type {solver_cfg.coupling_type!r}; expected 'proxy'.")
        if len(solver_cfg.entries) < 2:
            raise ValueError("Newton coupled solver requires at least two solver entries.")
        if not solver_cfg.proxy_coupling.proxies:
            raise ValueError("Newton proxy coupling requires at least one proxy mapping.")

        NewtonManager._solver = SolverProxyCoupled(
            model=model,
            entries=[cls._build_entry(entry_cfg) for entry_cfg in solver_cfg.entries],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[cls._build_proxy(proxy_cfg) for proxy_cfg in solver_cfg.proxy_coupling.proxies],
                iterations=solver_cfg.proxy_coupling.iterations,
            ),
        )
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = cls._needs_external_collision_pipeline(solver_cfg)

    @classmethod
    def _build_entry(cls, entry_cfg: CoupledSolverEntryCfg) -> SolverProxyCoupled.Entry:
        """Build a Newton ``SolverCoupled.Entry`` from an Isaac Lab entry cfg."""
        if not entry_cfg.name:
            raise ValueError("CoupledSolverEntryCfg.name must be non-empty.")
        if entry_cfg.substeps < 1:
            raise ValueError(f"CoupledSolverEntryCfg {entry_cfg.name!r} substeps must be >= 1.")

        solver_class = cls._resolve_solver_class(entry_cfg)
        solver_kwargs = cls._solver_kwargs(entry_cfg.solver_cfg)
        solver_kwargs.update(entry_cfg.solver_kwargs)

        return SolverProxyCoupled.Entry(
            name=entry_cfg.name,
            solver=solver_class,
            bodies=list(entry_cfg.bodies),
            particles=list(entry_cfg.particles),
            joints=list(entry_cfg.joints),
            shapes=list(entry_cfg.shapes),
            solver_kwargs=solver_kwargs,
            substeps=entry_cfg.substeps,
        )

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
    def _resolve_solver_class(cls, entry_cfg: CoupledSolverEntryCfg) -> type | Callable:
        """Resolve the Newton solver class for an entry."""
        if entry_cfg.solver_class is not None:
            return cls._resolve_class_or_callable(entry_cfg.solver_class)

        solver_type = getattr(entry_cfg.solver_cfg, "solver_type", None)
        if solver_type == "mujoco_warp":
            return SolverMuJoCo
        if solver_type == "implicit_mpm":
            return SolverImplicitMPM
        if solver_type == "xpbd":
            return SolverXPBD
        if solver_type == "featherstone":
            return SolverFeatherstone
        if solver_type == "kamino":
            return SolverKamino
        raise ValueError(
            f"Cannot infer coupled entry solver class from solver_type={solver_type!r}. "
            "Set CoupledSolverEntryCfg.solver_class for custom solvers."
        )

    @staticmethod
    def _resolve_class_or_callable(value: type | Callable | str) -> type | Callable:
        """Resolve a callable, LazyType-like object, or ``module:attr`` string."""
        if isinstance(value, str):
            module_name, _, attr = value.partition(":")
            if not module_name or not attr:
                raise ValueError(f"Expected solver_class as 'module:Class', got {value!r}.")
            return getattr(importlib.import_module(module_name), attr)
        if hasattr(value, "_resolve"):
            return value._resolve()
        return value

    @staticmethod
    def _solver_kwargs(solver_cfg: NewtonSolverCfg) -> dict:
        """Translate an Isaac Lab solver cfg into Newton constructor kwargs."""
        solver_type = getattr(solver_cfg, "solver_type", None)
        if solver_type == "implicit_mpm":
            return {"config": solver_cfg.to_solver_config()}
        if solver_type == "mujoco_warp":
            return resolve_mujoco_solver_kwargs(solver_cfg)

        solver_class_by_type = {
            "xpbd": SolverXPBD,
            "featherstone": SolverFeatherstone,
            "kamino": SolverKamino,
        }
        solver_class = solver_class_by_type.get(solver_type)
        if solver_class is None:
            return {}

        valid = set(inspect.signature(solver_class.__init__).parameters) - {"self", "model"}
        return {key: value for key, value in solver_cfg.to_dict().items() if key in valid}

    @classmethod
    def _needs_external_collision_pipeline(cls, solver_cfg: CoupledSolverCfg) -> bool:
        """Return whether the coupled solver should receive external contacts."""
        if solver_cfg.use_collision_pipeline is not None:
            return solver_cfg.use_collision_pipeline
        return any(cls._entry_needs_external_contacts(entry.solver_cfg) for entry in solver_cfg.entries)

    @staticmethod
    def _entry_needs_external_contacts(solver_cfg: NewtonSolverCfg) -> bool:
        """Infer external contact needs for known Newton sub-solvers."""
        solver_type = getattr(solver_cfg, "solver_type", None)
        if solver_type == "mujoco_warp":
            return not getattr(solver_cfg, "use_mujoco_contacts", True)
        if solver_type == "kamino":
            return not getattr(solver_cfg, "use_collision_detector", True)
        if solver_type in ("xpbd", "featherstone"):
            return True
        return False
