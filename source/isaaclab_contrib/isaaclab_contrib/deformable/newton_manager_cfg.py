# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for VBD and global Newton model parameters."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonSolverCfg

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class NewtonModelCfg:
    """Global Newton model parameters applied after builder finalization.

    These control model-level contact behavior shared across all objects.
    """

    soft_contact_ke: float = 1.0e3
    """Body-particle and particle self-contact stiffness [N/m].

    Effective per-contact stiffness is the average of this value and the rigid
    shape's material stiffness.
    """

    soft_contact_kd: float = 1.0e-2
    """Body-particle contact damping [N*s/m]."""

    soft_contact_mu: float = 0.5
    """Body-particle contact friction coefficient [dimensionless].

    Effective per-contact friction is ``sqrt(soft_contact_mu * shape_mu)``, where
    ``shape_mu`` is the rigid shape's own friction coefficient (from its per-asset
    material or :attr:`~isaaclab_newton.physics.NewtonShapeCfg.mu` default), not
    set by this config.
    """


@configclass
class NewtonModelSolverCfg(NewtonSolverCfg):
    """Base for solver configs whose manager applies :class:`NewtonModelCfg` to the finalized model.

    TODO: Temporary. This base only exists because :class:`NewtonModelCfg` lives in
    ``isaaclab_contrib`` while :class:`NewtonSolverCfg` is in ``isaaclab_newton`` core.
    Once these model params move into core, ``model_cfg`` should live on
    :class:`NewtonSolverCfg` (or ``NewtonCfg``) directly and this class can be removed.
    """

    model_cfg: NewtonModelCfg | None = None
    """Global Newton model parameters applied after builder finalization."""


@configclass
class VBDSolverCfg(NewtonModelSolverCfg):
    """Configuration for the Vertex Block Descent (VBD) solver.

    Supports cloth, soft bodies, and coupled rigid-body systems. Requires
    ``ModelBuilder.color()`` before ``finalize()`` to build the vertex coloring.
    """

    class_type: type[NewtonManager] | str = "{DIR}.vbd_manager:NewtonVBDManager"
    """Manager class for the VBD solver."""

    iterations: int = 10
    """Number of particle-VBD and rigid-AVBD iterations per substep.

    Newton does not expose separate rigid-joint iteration or convergence-
    tolerance controls; tune this shared iteration count and the joint penalty
    parameters below instead.
    """

    friction_epsilon: float = 1.0e-2
    """Relative-velocity threshold used to smooth contact friction [m/s]."""

    integrate_with_external_rigid_solver: bool = False
    """Whether rigid bodies are integrated by an external solver (one-way coupling).

    Set to ``True`` when coupling cloth with a separate rigid-body solver so VBD
    only integrates the cloth particles.
    """

    particle_enable_self_contact: bool = False
    """Whether to enable VBD deformable's self-contact."""

    particle_self_contact_radius: float = 0.005
    """Particle radius used for self-contact detection [m]."""

    particle_self_contact_margin: float = 0.005
    """Self-contact detection margin [m]. Should be >= particle_self_contact_radius."""

    particle_collision_detection_interval: int = -1
    """How often particle self-contact detection is applied.

    ``< 0``: once before initialization. ``0``: once before and once after
    initialization. ``k >= 1``: before every ``k`` VBD iterations.
    """

    particle_vertex_contact_buffer_size: int = 32
    """Preallocation size for each vertex's vertex-triangle collision buffer."""

    particle_edge_contact_buffer_size: int = 64
    """Preallocation size for each edge's edge-edge collision buffer."""

    particle_topological_contact_filter_threshold: int = 2
    """Maximum topological distance (in rings) below which self-contacts are discarded.

    Only used when ``particle_enable_self_contact`` is ``True``. Values > 3
    significantly increase computation time.
    """

    particle_rest_shape_contact_exclusion_radius: float = 0.0
    """Rest-configuration separation threshold for filtering close primitives [m].

    Only used when ``particle_enable_self_contact`` is ``True``.
    """

    rigid_contact_k_start: float = 1.0e2
    """Initial stiffness seed for all rigid body contacts [N/m]."""

    rigid_contact_hard: bool = True
    """Whether body-body contacts use augmented-Lagrangian hard constraints."""

    rigid_contact_history: bool = False
    """Whether to warm-start body-body contacts from collision-pipeline matching.

    The collision pipeline must populate contact-match indices, for example by
    setting ``contact_matching="latest"``.
    """

    rigid_body_contact_buffer_size: int = 64
    """Maximum number of body-body contacts tracked per rigid body."""

    rigid_body_particle_contact_buffer_size: int = 256
    """Maximum number of body-particle and full-surface contacts tracked per rigid body."""

    rigid_avbd_beta: float = 0.0
    """Per-iteration AVBD penalty-stiffness ramp rate.

    The value has units of [N/m per iteration] for linear constraints and
    [N*m/rad per iteration] for angular constraints. ``0.0`` uses fixed
    stiffness.
    """

    rigid_avbd_gamma: float = 0.999
    """Per-step decay for AVBD penalty stiffness and hard-mode multipliers."""

    rigid_joint_linear_ke: float = 1.0e5
    """Maximum linear penalty stiffness for VBD rigid joints [N/m]."""

    rigid_joint_angular_ke: float = 1.0e5
    """Maximum angular penalty stiffness for VBD rigid joints [N*m/rad]."""

    rigid_joint_linear_k_start: float = 1.0e2
    """Initial linear penalty seed for VBD rigid joints [N/m]."""

    rigid_joint_angular_k_start: float = 1.0e1
    """Initial angular penalty seed for VBD rigid joints [N*m/rad]."""

    rigid_joint_linear_kd: float = 0.0
    """Damping coefficient for non-cable linear joint constraints [N*s/m]."""

    rigid_joint_angular_kd: float = 0.0
    """Damping coefficient for non-cable angular joint constraints [N*m*s/rad]."""


@configclass
class CoupledMJWarpVBDSolverCfg(NewtonModelSolverCfg):
    """Deprecated configuration for the coupled MJWarp and VBD solver.

    .. deprecated:: 0.5.0
        Use :class:`isaaclab_contrib.custom_coupling.CoupledMJWarpVBDSolverCfg`.
    """

    class_type: type[NewtonManager] | str = (
        "isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_manager:NewtonCoupledMJWarpVBDManager"
    )
    """Manager class for the coupled MJWarp and VBD solver."""

    rigid_solver_cfg: MJWarpSolverCfg = MJWarpSolverCfg()
    """Rigid-body sub-solver configuration."""

    soft_solver_cfg: VBDSolverCfg = VBDSolverCfg(integrate_with_external_rigid_solver=True)
    """VBD sub-solver configuration."""

    coupling_mode: Literal["one_way", "two_way"] = "two_way"
    """Coupling direction between the rigid and VBD solvers."""

    def __post_init__(self) -> None:
        warnings.warn(
            "isaaclab_contrib.deformable.CoupledMJWarpVBDSolverCfg is deprecated. "
            "Use isaaclab_contrib.custom_coupling.CoupledMJWarpVBDSolverCfg.",
            DeprecationWarning,
            stacklevel=2,
        )
