# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton coupled multi-solver managers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from typing import TYPE_CHECKING, Any, Literal

from isaaclab.utils import configclass

from .newton_manager_cfg import NewtonSolverCfg

if TYPE_CHECKING:
    from newton.solvers import SolverBase

    from isaaclab_newton.physics import NewtonManager


@configclass
class CoupledSolverEntryCfg:
    """Configuration for one sub-solver entry inside a coupled Newton solver.

    Ownership is expressed in parent-model indices. The coupled solver uses
    these lists to build a :class:`newton.solvers.ModelView` per entry and to
    reconcile the owned state back into Isaac Lab's canonical state.
    """

    name: str = ""
    """Unique name for this sub-solver entry."""

    solver_cfg: NewtonSolverCfg = field(default_factory=NewtonSolverCfg)
    """Isaac Lab Newton solver cfg used to construct the entry's Newton solver."""

    solver_class: type[SolverBase] | Callable | str | None = None
    """Optional explicit solver class or ``"module:Class"`` path.

    When ``None``, the solver class and constructor kwargs are inferred from
    :attr:`solver_cfg`. This escape hatch lets experiments wire new Newton
    solvers before Isaac Lab has a dedicated ``*SolverCfg`` wrapper.
    """

    bodies: list[int] = field(default_factory=list)
    """Parent-model body indices owned by this entry."""

    particles: list[int] = field(default_factory=list)
    """Parent-model particle indices owned by this entry."""

    joints: list[int] = field(default_factory=list)
    """Parent-model joint indices owned by this entry."""

    shapes: list[int] = field(default_factory=list)
    """Parent-model shape indices visible to this entry.

    Leave empty to let Newton's coupled solver keep default shape visibility.
    """

    solver_kwargs: dict[str, Any] = field(default_factory=dict)
    """Extra keyword arguments forwarded to the sub-solver constructor.

    These override kwargs inferred from :attr:`solver_cfg`.
    """

    substeps: int = 1
    """Number of equal substeps this entry runs inside one coupled step."""


@configclass
class CoupledProxyCfg:
    """Configuration for one lagged-impulse proxy mapping."""

    source: str = ""
    """Entry name that owns the source objects."""

    destination: str = ""
    """Entry name that receives the proxy objects."""

    bodies: list[int] = field(default_factory=list)
    """Source body ids mapped into the destination as proxy bodies."""

    proxy_bodies: list[int] | None = None
    """Destination proxy body ids. ``None`` mirrors :attr:`bodies`."""

    particles: list[int] = field(default_factory=list)
    """Source particle ids mapped into the destination as proxy particles."""

    proxy_particles: list[int] | None = None
    """Destination proxy particle ids. ``None`` mirrors :attr:`particles`."""

    mass_scale: float = 1.0
    """Scale factor for proxy mass/inertia in the destination view."""

    mode: Literal["lagged", "staggered"] | int = "lagged"
    """Proxy transfer mode passed to Newton's ``SolverProxyCoupled``."""

    collision_pipeline_factory: Callable | None = None
    """Optional factory for a proxy-local collision pipeline.

    The callable is passed directly to Newton as ``collision_pipeline`` and is
    invoked as ``factory(destination_model_view)``.
    """

    collide_interval: int | None = None
    """Proxy-local collision refresh interval when a factory is supplied."""


@configclass
class ProxyCouplingCfg:
    """Lagged-impulse proxy coupling configuration."""

    proxies: list[CoupledProxyCfg] = field(default_factory=list)
    """Proxy mappings used by ``SolverProxyCoupled``."""

    iterations: int = 1
    """Number of proxy relaxation passes per coupled step."""


@configclass
class CoupledSolverCfg(NewtonSolverCfg):
    """Configuration for Newton multi-solver coupling.

    This initial Isaac Lab wrapper targets Newton's
    :class:`newton.solvers.SolverProxyCoupled`, but the entry cfgs are solver
    agnostic: each entry declares what it owns and which solver should advance
    it. Future coupled algorithms can reuse the same ownership model with a
    different manager or ``coupling_type``.
    """

    class_type: type[NewtonManager] | str = "{DIR}.coupled_manager:NewtonCoupledManager"
    """Manager class for Newton coupled solvers."""

    solver_type: str = "coupled"
    """Solver type metadata. Can be ``"coupled"``."""

    coupling_type: Literal["proxy"] = "proxy"
    """Coupling algorithm to construct. Only ``"proxy"`` is currently supported."""

    entries: list[CoupledSolverEntryCfg] = field(default_factory=list)
    """Ordered sub-solver entries."""

    proxy_coupling: ProxyCouplingCfg = field(default_factory=ProxyCouplingCfg)
    """Configuration for ``coupling_type="proxy"``."""

    use_collision_pipeline: bool | None = None
    """Whether Isaac Lab should run Newton's external collision pipeline.

    If ``None``, the manager infers the value from the sub-solver cfgs. For
    example, MuJoCo entries with ``use_mujoco_contacts=False`` and XPBD entries
    need the external pipeline, while implicit MPM does not.
    """
