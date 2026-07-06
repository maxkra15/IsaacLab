# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the per-solver :class:`NewtonManager` abstraction.

Covers:

* :attr:`NewtonSolverCfg.class_type` resolves to the matching manager subclass.
* :meth:`NewtonCfg.__post_init__` propagates ``solver_cfg.class_type`` onto
  :attr:`NewtonCfg.class_type` so that ``SimulationContext`` picks the right
  manager.
* Each leaf manager subclasses :class:`NewtonManager` and implements
  :meth:`_build_solver` (with the abstract base raising ``NotImplementedError``).
* The cross-config validation in :meth:`NewtonMJWarpManager._build_solver`
  rejects the ``MJWarp + use_mujoco_contacts=True + collision_cfg`` combination.
* Manager name dispatch (used by :class:`InteractiveScene` and the various
  factory dispatchers) still starts with ``"newton"``.
* End-to-end: spinning up a simulation with each solver builds the correct
  solver, sets the right ``_use_single_state`` / ``_needs_collision_pipeline``
  flags, and lands canonical state on :class:`NewtonManager` so that external
  ``NewtonManager._foo`` reads keep working.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import (
    AdmmContactPairCfg,
    AdmmCouplingCfg,
    CoupledProxyCfg,
    CoupledSolverCfg,
    CoupledSolverEntryCfg,
    FeatherstoneSolverCfg,
    KaminoSolverCfg,
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonCoupledManager,
    NewtonFeatherstoneManager,
    NewtonKaminoManager,
    NewtonManager,
    NewtonMJWarpManager,
    NewtonMPMManager,
    NewtonSolverCfg,
    NewtonXPBDManager,
    ProxyCouplingCfg,
    XPBDSolverCfg,
)
from newton.solvers import SolverFeatherstone, SolverImplicitMPM, SolverKamino, SolverMuJoCo, SolverXPBD
from newton.solvers.experimental.coupled import (
    SolverCoupled,
    SolverCoupledADMM,
    SolverCoupledProxy,
)

from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg, build_simulation_context

# ---------------------------------------------------------------------------
# Lightweight (no sim) parametrisation
# ---------------------------------------------------------------------------

# (solver_cfg_factory, expected_manager, expected_solver_cls,
#  expected_use_single_state, expected_needs_collision_pipeline)
SOLVER_MATRIX = [
    pytest.param(
        lambda: MJWarpSolverCfg(use_mujoco_contacts=True),
        NewtonMJWarpManager,
        SolverMuJoCo,
        True,
        False,
        id="mjwarp_internal_contacts",
    ),
    pytest.param(
        lambda: MJWarpSolverCfg(use_mujoco_contacts=False),
        NewtonMJWarpManager,
        SolverMuJoCo,
        True,
        True,
        id="mjwarp_newton_pipeline",
    ),
    pytest.param(
        lambda: XPBDSolverCfg(),
        NewtonXPBDManager,
        SolverXPBD,
        False,
        True,
        id="xpbd",
    ),
    pytest.param(
        lambda: FeatherstoneSolverCfg(),
        NewtonFeatherstoneManager,
        SolverFeatherstone,
        False,
        True,
        id="featherstone",
    ),
    pytest.param(
        lambda: KaminoSolverCfg(use_collision_detector=True),
        NewtonKaminoManager,
        SolverKamino,
        False,
        False,
        id="kamino_internal_contacts",
    ),
    pytest.param(
        lambda: KaminoSolverCfg(use_collision_detector=False),
        NewtonKaminoManager,
        SolverKamino,
        False,
        True,
        id="kamino_newton_pipeline",
    ),
    pytest.param(
        lambda: MPMSolverCfg(max_iterations=2, voxel_size=0.05),
        NewtonMPMManager,
        SolverImplicitMPM,
        True,
        False,
        id="implicit_mpm",
    ),
    pytest.param(
        lambda: CoupledSolverCfg(
            coupling_type="base",
            entries=[
                CoupledSolverEntryCfg(name="rigid", solver_cfg=XPBDSolverCfg(iterations=1), bodies=[0]),
                CoupledSolverEntryCfg(
                    name="particle",
                    solver_cfg=XPBDSolverCfg(iterations=1),
                    particles=[0],
                    in_place=True,
                ),
            ],
        ),
        NewtonCoupledManager,
        SolverCoupled,
        False,
        True,
        marks=pytest.mark.skipif(SolverCoupled is None, reason="Newton SolverCoupled is unavailable"),
        id="base_coupled_xpbd_body_particle",
    ),
    pytest.param(
        lambda: CoupledSolverCfg(
            entries=[
                CoupledSolverEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(
                        use_mujoco_contacts=False,
                        njmax=100,
                        nconmax=100,
                        iterations=2,
                        ls_iterations=2,
                    ),
                    bodies=[0],
                    joints=[0],
                ),
                CoupledSolverEntryCfg(
                    name="sand",
                    solver_cfg=MPMSolverCfg(max_iterations=2, voxel_size=0.05),
                    particles=list(range(8)),
                ),
            ],
            proxy_coupling=ProxyCouplingCfg(
                proxies=[
                    CoupledProxyCfg(
                        source="rigid",
                        destination="sand",
                        bodies=[0],
                    )
                ],
            ),
        ),
        NewtonCoupledManager,
        SolverCoupledProxy,
        False,
        True,
        id="proxy_coupled_mjwarp_mpm",
    ),
    pytest.param(
        lambda: CoupledSolverCfg(
            coupling_type="admm",
            entries=[
                CoupledSolverEntryCfg(
                    name="rigid",
                    solver_cfg=XPBDSolverCfg(iterations=1),
                    bodies=[0],
                ),
                CoupledSolverEntryCfg(
                    name="particle",
                    solver_cfg=XPBDSolverCfg(iterations=1),
                    particles=[0],
                    in_place=True,
                ),
            ],
            admm_coupling=AdmmCouplingCfg(iterations=1, rho=1.0, gamma=0.0),
            use_collision_pipeline=False,
        ),
        NewtonCoupledManager,
        SolverCoupledADMM,
        False,
        False,
        id="admm_coupled_xpbd_body_particle",
    ),
]


# ---------------------------------------------------------------------------
# class_type wiring (no SimulationContext required)
# ---------------------------------------------------------------------------


def test_newton_manager_clear_discards_mpm_object_registrations():
    """Closed simulations must not re-emit particle assets from a prior scene."""
    NewtonManager._mpm_object_registry = [object()]

    NewtonManager.clear()

    assert NewtonManager._mpm_object_registry == []


@pytest.mark.parametrize(
    "solver_cfg_factory, expected_manager, _solver_cls, _single_state, _pipeline",
    SOLVER_MATRIX,
)
def test_solver_cfg_class_type_resolves_to_subclass(
    solver_cfg_factory, expected_manager, _solver_cls, _single_state, _pipeline
):
    """Each ``*SolverCfg.class_type`` resolves to its matching manager subclass."""
    solver_cfg = solver_cfg_factory()
    # ``class_type`` is a lazy ``"module:Class"`` reference; calling its
    # ``_resolve()`` returns the actual class. ``__name__`` works without
    # forcing import (LazyType caches metadata) and is sufficient identity.
    assert solver_cfg.class_type.__name__ == expected_manager.__name__


@pytest.mark.parametrize(
    "solver_cfg_factory, expected_manager, _solver_cls, _single_state, _pipeline",
    SOLVER_MATRIX,
)
def test_newton_cfg_post_init_propagates_class_type(
    solver_cfg_factory, expected_manager, _solver_cls, _single_state, _pipeline
):
    """``NewtonCfg.__post_init__`` lifts ``solver_cfg.class_type`` onto ``NewtonCfg.class_type``."""
    cfg = NewtonCfg(solver_cfg=solver_cfg_factory())
    assert cfg.class_type.__name__ == expected_manager.__name__


def test_mpm_custom_attribute_registration_is_idempotent(monkeypatch):
    """MPM custom attributes are registered once per builder."""

    original_register = SolverImplicitMPM.register_custom_attributes
    register_call_count = 0

    def counted_register(builder):
        nonlocal register_call_count
        register_call_count += 1
        original_register(builder)

    monkeypatch.setattr(SolverImplicitMPM, "register_custom_attributes", counted_register)

    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(solver_cfg=MPMSolverCfg(max_iterations=2, voxel_size=0.05), use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg):
        builder = NewtonManager.create_builder()
        NewtonManager._register_solver_custom_attributes(builder)

    assert register_call_count == 1


def test_kamino_eval_fk_uses_public_reset_config_when_available(monkeypatch):
    """Kamino's current reset API receives the parent state and a joint-derived config."""
    from isaaclab.physics import PhysicsManager

    calls = []
    state = SimpleNamespace(joint_q=object(), joint_qd=object())
    world_mask = object()
    solver = SimpleNamespace(reset=lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        SimpleNamespace(solver_cfg=KaminoSolverCfg(use_fk_solver=True)),
        raising=False,
    )
    monkeypatch.setattr(NewtonManager, "_solver", solver, raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", state, raising=False)

    NewtonKaminoManager._eval_fk_impl(world_reset_mask=world_mask, fk_mask=None)

    assert calls[0][0] == ()
    assert calls[0][1]["state"] is state
    assert calls[0][1]["world_mask"] is world_mask
    assert isinstance(calls[0][1]["config"], SolverKamino.ResetConfig)


def test_kamino_eval_fk_falls_back_to_legacy_joint_reset(monkeypatch):
    """The manager remains compatible with the upstream-pinned legacy Kamino signature."""
    from isaaclab.physics import PhysicsManager

    calls = []
    state = SimpleNamespace(joint_q=object(), joint_qd=object())
    world_mask = object()
    solver = SimpleNamespace(reset=lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        SimpleNamespace(solver_cfg=KaminoSolverCfg(use_fk_solver=True)),
        raising=False,
    )
    monkeypatch.setattr(NewtonManager, "_solver", solver, raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", state, raising=False)
    monkeypatch.setattr(SolverKamino, "ResetConfig", None, raising=False)

    NewtonKaminoManager._eval_fk_impl(world_reset_mask=world_mask, fk_mask=None)

    assert calls == [
        (
            (state,),
            {"joint_q": state.joint_q, "joint_u": state.joint_qd, "world_mask": world_mask},
        )
    ]


@pytest.mark.parametrize(
    "num_substeps, collision_decimation, should_warn",
    [
        (8, 0, False),  # Default: feature disabled, no warning.
        (8, 1, False),  # Valid: re-collide every substep.
        (8, 2, False),  # Valid: re-collide every 2 substeps.
        (8, 7, False),  # Valid edge: one mid-loop re-collide at i=6.
        (8, 8, True),  # Equal to num_substeps: gate never fires.
        (8, 16, True),  # Larger than num_substeps: gate never fires.
    ],
)
def test_newton_cfg_collision_decimation_warning(num_substeps, collision_decimation, should_warn, caplog):
    """``NewtonCfg.__post_init__`` warns when ``collision_decimation >= num_substeps``."""
    import logging

    with caplog.at_level(logging.WARNING, logger="isaaclab_newton.physics.newton_manager_cfg"):
        cfg = NewtonCfg(num_substeps=num_substeps, collision_decimation=collision_decimation)
    warned = any("collision_decimation" in rec.getMessage() for rec in caplog.records)
    assert warned is should_warn
    # Cfg field round-trips regardless of warning.
    assert cfg.collision_decimation == collision_decimation


def test_mpm_solver_cfg_maps_only_newton_solver_fields():
    """MPM config forwarding ignores Isaac Lab metadata fields explicitly."""

    solver_cfg = MPMSolverCfg(
        max_iterations=7,
        voxel_size=0.04,
        solver_type="isaaclab_metadata_should_not_forward",
    )

    newton_cfg = solver_cfg.to_solver_config()

    assert newton_cfg.max_iterations == 7
    assert newton_cfg.voxel_size == 0.04
    assert not hasattr(newton_cfg, "class_type")
    assert not hasattr(newton_cfg, "solver_type")
    # Manager-level stepping option must not leak into the Newton solver config.
    assert not hasattr(newton_cfg, "project_outside_colliders")


def test_mpm_solver_cfg_requires_world_isolation_support(monkeypatch):
    """Missing upstream isolation support fails instead of silently sharing one grid."""
    import newton.solvers

    class LegacyConfig:
        max_iterations = 250

    monkeypatch.setattr(newton.solvers.SolverImplicitMPM, "Config", LegacyConfig)

    with pytest.raises(RuntimeError, match="separate_worlds"):
        MPMSolverCfg().to_solver_config()


def test_mpm_solver_cfg_rejects_missing_requested_hierarchy_capacity_support(monkeypatch):
    """A non-default node reserve must not be silently dropped by an older Newton."""
    import newton.solvers

    class LegacyConfig:
        separate_worlds = True

    monkeypatch.setattr(newton.solvers.SolverImplicitMPM, "Config", LegacyConfig)

    with pytest.raises(RuntimeError, match="max_upper_node_count"):
        MPMSolverCfg(max_upper_node_count=128).to_solver_config()


# Tuples of ``(field_name, non_default_value)`` covering every solver-tunable
# field on :class:`MPMSolverCfg`. Each entry exercises the implementation-side
# SolverImplicitMPM.Config construction so a Newton field rename or accidental
# drop is caught here instead of silently producing wrong-physics runs.
_MPM_FIELD_VALUES = [
    ("max_iterations", 13),
    ("tolerance", 5.0e-5),
    ("solver", "gauss-seidel"),
    ("warmstart_mode", "particles"),
    ("collider_velocity_mode", "backward"),
    ("voxel_size", 0.0375),
    ("grid_type", "dense"),
    ("grid_padding", 4),
    ("max_active_cell_count", 1024),
    ("max_leaf_node_count", 768),
    ("max_lower_node_count", 96),
    ("max_upper_node_count", 24),
    ("transfer_scheme", "pic"),
    ("integration_scheme", "gimp"),
    ("critical_fraction", 0.25),
    ("air_drag", 0.5),
    ("collider_normal_from_sdf_gradient", True),
    ("collider_basis", "Q1"),
    ("strain_basis", "P1d"),
    ("velocity_basis", "B2"),
    ("separate_worlds", False),
]


@pytest.mark.parametrize("field_name, value", _MPM_FIELD_VALUES)
def test_mpm_solver_cfg_forwards_every_solver_field(field_name, value):
    """Every tunable MPM cfg field round-trips into ``SolverImplicitMPM.Config``.

    Guards against MPM manager construction dropping or mis-naming a field if
    Newton's config surface changes.
    """
    solver_cfg = MPMSolverCfg(**{field_name: value})
    newton_cfg = solver_cfg.to_solver_config()
    assert hasattr(newton_cfg, field_name), (
        f"{field_name!r} disappeared from SolverImplicitMPM.Config — MPMSolverCfg needs to drop or rename it."
    )
    assert getattr(newton_cfg, field_name) == value


def test_mpm_register_builder_attributes_is_idempotent():
    """The MPM custom-attribute hook is a no-op when attributes are already registered."""
    import newton

    builder = newton.ModelBuilder()
    assert not builder.has_custom_attribute("mpm:young_modulus")

    NewtonMPMManager._register_builder_attributes(builder)
    assert builder.has_custom_attribute("mpm:young_modulus")

    # Second call must be a no-op (no exceptions, attribute still present).
    NewtonMPMManager._register_builder_attributes(builder)
    assert builder.has_custom_attribute("mpm:young_modulus")


def test_mpm_prepare_builder_makes_kinematic_bodies_massless():
    """Kinematic bodies must be massless so MPM treats them as kinematic colliders."""
    import newton

    builder = newton.ModelBuilder()
    kinematic_body = builder.add_body(
        mass=0.35,
        inertia=wp.mat33(1.0),
        is_kinematic=True,
        label="kinematic_collider",
    )
    dynamic_body = builder.add_body(
        mass=1.2,
        inertia=wp.mat33(2.0),
        is_kinematic=False,
        label="dynamic_body",
    )

    NewtonMPMManager._prepare_builder_for_finalize(builder)

    assert builder.body_flags[kinematic_body] & int(newton.BodyFlags.KINEMATIC)
    assert builder.body_mass[kinematic_body] == 0.0
    assert builder.body_inv_mass[kinematic_body] == 0.0
    assert np.allclose(np.array(builder.body_inertia[kinematic_body]), 0.0)
    assert np.allclose(np.array(builder.body_inv_inertia[kinematic_body]), 0.0)

    assert builder.body_mass[dynamic_body] == pytest.approx(1.2)
    assert builder.body_inv_mass[dynamic_body] == pytest.approx(1.0 / 1.2)
    assert np.allclose(np.array(builder.body_inertia[dynamic_body]), 2.0)


def test_coupled_prepare_builder_makes_mpm_kinematic_bodies_massless(monkeypatch):
    """Coupled MPM colliders use the same massless kinematic convention."""
    import newton

    from isaaclab.physics import PhysicsManager

    builder = newton.ModelBuilder()
    kinematic_body = builder.add_body(mass=0.35, inertia=wp.mat33(1.0), is_kinematic=True)
    dynamic_body = builder.add_body(mass=1.2, inertia=wp.mat33(2.0), is_kinematic=False)
    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        SimpleNamespace(solver_cfg=CoupledSolverCfg(entries=[CoupledSolverEntryCfg(solver_cfg=MPMSolverCfg())])),
        raising=False,
    )

    NewtonCoupledManager._prepare_builder_for_finalize(builder)

    assert builder.body_mass[kinematic_body] == 0.0
    assert builder.body_inv_mass[kinematic_body] == 0.0
    assert np.allclose(np.array(builder.body_inertia[kinematic_body]), 0.0)
    assert np.allclose(np.array(builder.body_inv_inertia[kinematic_body]), 0.0)
    assert builder.body_mass[dynamic_body] == pytest.approx(1.2)


def test_coupled_prepare_builder_preserves_non_mpm_kinematic_mass(monkeypatch):
    """Pure rigid/soft coupled configurations must not inherit MPM collider normalization."""
    import newton

    from isaaclab.physics import PhysicsManager

    builder = newton.ModelBuilder()
    kinematic_body = builder.add_body(mass=0.35, inertia=wp.mat33(1.0), is_kinematic=True)
    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        SimpleNamespace(solver_cfg=CoupledSolverCfg(entries=[CoupledSolverEntryCfg(solver_cfg=XPBDSolverCfg())])),
        raising=False,
    )

    NewtonCoupledManager._prepare_builder_for_finalize(builder)

    assert builder.body_mass[kinematic_body] == pytest.approx(0.35)
    assert builder.body_inv_mass[kinematic_body] == pytest.approx(1.0 / 0.35)
    assert np.allclose(np.array(builder.body_inertia[kinematic_body]), 1.0)


def test_coupled_mpm_projection_entries_are_selected_from_config():
    """Only coupled MPM entries that opt in run particle projection."""
    entries = [
        CoupledSolverEntryCfg(
            name="projected",
            solver_cfg=MPMSolverCfg(project_outside_colliders=True),
        ),
        CoupledSolverEntryCfg(
            name="implicit_only",
            solver_cfg=MPMSolverCfg(project_outside_colliders=False),
        ),
        CoupledSolverEntryCfg(name="rigid", solver_cfg=XPBDSolverCfg()),
    ]

    assert NewtonCoupledManager._mpm_project_outside_entry_names(entries) == ("projected",)


def test_coupled_step_projects_selected_mpm_entries(monkeypatch):
    """Coupled projection updates authoritative entry state and reconciles it publicly."""
    calls = []
    entry_state = object()
    mpm_solver = SimpleNamespace(
        project_outside=lambda state_in, state_out, dt: calls.append((state_in, state_out, dt)),
    )
    coupled_solver = SimpleNamespace(
        step=lambda *args: calls.append("step"),
        solver=lambda name: mpm_solver,
        entry_state=lambda name, phase: calls.append(("entry_state", name, phase)) or entry_state,
        reconcile_entry_state=lambda name, state, phase: calls.append(("reconcile", name, state, phase)),
    )
    state_in = object()
    state_out = object()
    monkeypatch.setattr(NewtonManager, "_solver", coupled_solver, raising=False)
    monkeypatch.setattr(NewtonCoupledManager, "_mpm_project_outside_entries", ("sand",), raising=False)

    NewtonCoupledManager._step_solver(state_in, state_out, None, None, 0.01)

    assert calls == [
        "step",
        ("entry_state", "sand", "output"),
        (entry_state, entry_state, 0.01),
        ("reconcile", "sand", state_out, "output"),
    ]


def test_coupled_clear_removes_mpm_projection_entries(monkeypatch):
    """Coupled teardown does not leak MPM entry selection into the next scene."""
    monkeypatch.setattr(NewtonCoupledManager, "_mpm_project_outside_entries", ("sand",), raising=False)

    NewtonCoupledManager._solver_specific_clear()

    assert NewtonCoupledManager._mpm_project_outside_entries == ()


@pytest.mark.parametrize("solver_cfg, expected", [(MPMSolverCfg(), True), (XPBDSolverCfg(), False)])
def test_coupled_build_requests_fk_for_mpm_entries(monkeypatch, solver_cfg, expected):
    """Coupled MPM asks the manager to refresh kinematic collider transforms."""
    import isaaclab_newton.physics.coupled_manager as coupled_manager

    cfg = CoupledSolverCfg(
        coupling_type="base",
        entries=[CoupledSolverEntryCfg(name="entry", solver_cfg=solver_cfg)],
    )
    monkeypatch.setattr(NewtonCoupledManager, "_resolve_solver_cfg", classmethod(lambda cls, model, value: value))
    monkeypatch.setattr(NewtonCoupledManager, "_validate_solver_cfg", classmethod(lambda cls, value: None))
    monkeypatch.setattr(NewtonCoupledManager, "_build_entry", classmethod(lambda cls, value: object()))
    monkeypatch.setattr(NewtonCoupledManager, "_apply_entry_solver_overrides", classmethod(lambda cls, value: None))
    monkeypatch.setattr(
        NewtonCoupledManager,
        "_configure_fk_articulation_filter",
        classmethod(lambda cls, model, value: None),
    )
    monkeypatch.setattr(
        NewtonCoupledManager,
        "_needs_external_collision_pipeline",
        classmethod(lambda cls, value: False),
    )
    monkeypatch.setattr(coupled_manager, "SolverCoupled", lambda **kwargs: SimpleNamespace())

    NewtonCoupledManager._build_solver(SimpleNamespace(), cfg)

    assert NewtonManager._needs_fk_before_step is expected


def test_active_manager_create_builder_registers_mpm_attributes():
    """The active MPM manager registers solver-specific builder attributes."""
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(solver_cfg=MPMSolverCfg(max_iterations=2, voxel_size=0.05), use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = sim.physics_manager.create_builder()

    assert builder.has_custom_attribute("mpm:young_modulus")


def test_mpm_end_to_end_with_particle_custom_attributes():
    """End-to-end MPM step using ``add_particles(custom_attributes=...)`` — the production path."""
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MPMSolverCfg(max_iterations=2, voxel_size=0.05),
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = sim.physics_manager.create_builder()
        # MPM custom attrs must exist on the builder before particles use them.
        assert builder.has_custom_attribute("mpm:young_modulus")

        positions = [(0.0, 0.0, 0.10), (0.05, 0.0, 0.10), (0.0, 0.05, 0.10)]
        builder.add_particles(
            pos=positions,
            vel=[(0.0, 0.0, 0.0)] * len(positions),
            mass=[0.01] * len(positions),
            radius=[0.02] * len(positions),
            custom_attributes={
                "mpm:viscosity": 50.0,
                "mpm:friction": 0.0,
                "mpm:tensile_yield_ratio": 1.0,
                "mpm:yield_pressure": 1.0e15,
                "mpm:yield_stress": 0.0,
                "mpm:young_modulus": 1.0e15,
                "mpm:damping": 0.0,
            },
        )
        NewtonManager.set_builder(builder)

        sim.reset()
        assert isinstance(NewtonManager._solver, SolverImplicitMPM)
        sim.step(render=False)


@pytest.mark.parametrize("project_outside", [True, False])
def test_mpm_project_outside_colliders_gates_projection(project_outside):
    """``project_outside_colliders`` controls whether ``project_outside`` runs per substep.

    Wraps the solver's ``project_outside`` with a counter after ``sim.reset()``
    (``use_cuda_graph=False`` keeps the Python callable on the step path) and
    runs one tick. The call count is positive only when the flag is set.
    """
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MPMSolverCfg(max_iterations=2, voxel_size=0.05, project_outside_colliders=project_outside),
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = sim.physics_manager.create_builder()
        builder.add_particles(
            pos=[(0.0, 0.0, 0.10), (0.05, 0.0, 0.10), (0.0, 0.05, 0.10)],
            vel=[(0.0, 0.0, 0.0)] * 3,
            mass=[0.01] * 3,
            radius=[0.02] * 3,
            custom_attributes={
                "mpm:viscosity": 50.0,
                "mpm:friction": 0.0,
                "mpm:tensile_yield_ratio": 1.0,
                "mpm:yield_pressure": 1.0e15,
                "mpm:yield_stress": 0.0,
                "mpm:young_modulus": 1.0e15,
                "mpm:damping": 0.0,
            },
        )
        NewtonManager.set_builder(builder)
        sim.reset()

        calls = {"n": 0}
        original_project = NewtonManager._solver.project_outside

        def counting_project(*args, **kwargs):
            calls["n"] += 1
            return original_project(*args, **kwargs)

        NewtonManager._solver.project_outside = counting_project
        try:
            sim.step(render=False)
        finally:
            NewtonManager._solver.project_outside = original_project

        if project_outside:
            assert calls["n"] >= 1
        else:
            assert calls["n"] == 0


@pytest.mark.parametrize("supported", [True, False])
def test_newton_cuda_graph_capture_capability_is_owned_by_solver(monkeypatch, supported):
    """The manager delegates graph capability to Newton's public solver contract."""

    monkeypatch.setattr(
        NewtonManager,
        "_solver",
        SimpleNamespace(supports_cuda_graph_capture=supported),
        raising=False,
    )

    assert NewtonManager._supports_cuda_graph_capture() is supported


def test_legacy_solver_without_cuda_graph_contract_stays_eager(monkeypatch):
    """Pinned Newton solvers without a capability contract never enter graph capture."""
    from isaaclab.physics import PhysicsManager

    monkeypatch.setattr(PhysicsManager, "_cfg", SimpleNamespace(use_cuda_graph=True), raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", "cuda:0", raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", SimpleNamespace(), raising=False)
    monkeypatch.setattr(NewtonManager, "_graph", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_graph_capture_pending", True, raising=False)

    NewtonManager._capture_or_defer_graph()

    assert NewtonManager._supports_cuda_graph_capture() is False
    assert NewtonManager._graph is None
    assert NewtonManager._graph_capture_pending is False


@pytest.mark.parametrize(
    "grid_type, advertised, expected",
    [
        ("fixed", False, False),
        ("fixed", True, True),
        ("sparse", False, False),
        ("sparse", True, True),
        ("dense", True, False),
    ],
)
def test_mpm_cuda_graph_capture_preserves_fixed_policy_and_accepts_sparse_opt_in(
    monkeypatch, grid_type, advertised, expected
):
    """Fixed grids remain supported while a custom sparse solver can explicitly opt in."""
    monkeypatch.setattr(
        NewtonManager,
        "_solver",
        SimpleNamespace(grid_type=grid_type, supports_cuda_graph_capture=advertised),
        raising=False,
    )

    assert NewtonMPMManager._supports_cuda_graph_capture() is expected


@pytest.mark.parametrize("grid_type", ["fixed", "sparse"])
def test_legacy_mpm_solver_without_cuda_graph_contract_stays_eager(monkeypatch, grid_type):
    """Pinned fixed and sparse MPM solvers do not inherit an implicit graph capability."""
    monkeypatch.setattr(NewtonManager, "_solver", SimpleNamespace(grid_type=grid_type), raising=False)

    assert NewtonMPMManager._supports_cuda_graph_capture() is False


@pytest.mark.parametrize("graph, expected", [(None, False), (object(), True)])
def test_cuda_graph_active_reports_recorded_manager_graph(monkeypatch, graph, expected):
    """Callers can verify graph activation without reading manager internals."""
    monkeypatch.setattr(NewtonManager, "_graph", graph, raising=False)

    assert NewtonManager.is_cuda_graph_active() is expected


def test_forward_pending_evaluates_and_consumes_only_filtered_dirty_articulations(monkeypatch):
    """Selective asset writes must not turn an RL reset into full-scene FK."""
    filtered_mask = object()
    world_mask = object()
    events = []
    fk_calls = []
    reset_mask = SimpleNamespace(zero_=lambda: events.append("zero"))

    monkeypatch.setattr(NewtonManager, "_world_reset_mask", world_mask, raising=False)
    monkeypatch.setattr(NewtonManager, "_fk_reset_mask", reset_mask, raising=False)
    monkeypatch.setattr(NewtonManager, "_usdrt_stage", object(), raising=False)
    monkeypatch.setattr(
        NewtonManager,
        "_eval_fk",
        staticmethod(lambda received_world_mask, fk_mask: fk_calls.append((received_world_mask, fk_mask))),
    )
    monkeypatch.setattr(
        NewtonManager,
        "_filtered_fk_reset_mask",
        classmethod(lambda cls: filtered_mask),
    )
    monkeypatch.setattr(
        NewtonManager,
        "_mark_transforms_dirty",
        classmethod(lambda cls: events.append("dirty")),
    )

    NewtonManager.forward_pending()

    assert fk_calls == [(world_mask, filtered_mask)]
    assert events == ["dirty", "zero"]


def test_forward_flushes_fabric_transforms_after_fk(monkeypatch):
    """Explicit forward must publish fresh link poses before returning to a Kit caller."""
    events = []
    world_mask = SimpleNamespace(zero_=lambda: None)

    monkeypatch.setattr(NewtonManager, "_world_reset_mask", world_mask, raising=False)
    monkeypatch.setattr(NewtonManager, "_fk_reset_mask", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_state_teleport_pending", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_requires_teleport_reset", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_usdrt_stage", object(), raising=False)
    monkeypatch.setattr(
        NewtonManager,
        "_eval_fk",
        staticmethod(lambda received_world_mask, fk_mask: events.append(("fk", received_world_mask, fk_mask))),
    )
    monkeypatch.setattr(
        NewtonManager,
        "_filtered_fk_reset_mask",
        classmethod(lambda cls: "filter"),
    )
    monkeypatch.setattr(
        NewtonManager,
        "_mark_transforms_dirty",
        classmethod(lambda cls: events.append(("dirty",))),
    )
    monkeypatch.setattr(
        NewtonManager,
        "sync_transforms_to_usd",
        classmethod(lambda cls: events.append(("sync",))),
    )

    NewtonManager.forward()

    assert events == [
        ("fk", world_mask, "filter"),
        ("dirty",),
        ("sync",),
    ]


def _configure_manager_step_test(monkeypatch, *, use_graph=False, externally_capturing=False, status_error=None):
    from isaaclab.physics import PhysicsManager

    events = []

    def check_status():
        events.append("status")
        if status_error is not None:
            raise status_error

    device = "cuda:0" if use_graph else "cpu"
    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(is_playing=lambda: True), raising=False)
    monkeypatch.setattr(PhysicsManager, "_cfg", SimpleNamespace(use_cuda_graph=use_graph), raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", device, raising=False)
    monkeypatch.setattr(PhysicsManager, "_sim_time", 0.0, raising=False)
    monkeypatch.setattr(NewtonManager, "_model_changes", set(), raising=False)
    monkeypatch.setattr(NewtonManager, "_graph_capture_pending", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_graph", object() if use_graph else None, raising=False)
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_needs_fk_before_step", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_world_reset_mask", SimpleNamespace(zero_=lambda: None), raising=False)
    monkeypatch.setattr(NewtonManager, "_fk_reset_mask", SimpleNamespace(zero_=lambda: None), raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", SimpleNamespace(check_status=check_status))
    monkeypatch.setattr(NewtonManager, "_is_all_graphable", classmethod(lambda cls: True))
    monkeypatch.setattr(
        NewtonManager,
        "_is_outer_cuda_graph_capture_active",
        classmethod(lambda cls: externally_capturing),
        raising=False,
    )
    monkeypatch.setattr(NewtonManager, "_simulate_full", classmethod(lambda cls: events.append("eager")))
    monkeypatch.setattr(wp, "capture_launch", lambda graph: events.append("replay"))
    monkeypatch.setattr(NewtonManager, "_usdrt_stage", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_particle_visual_prims", {}, raising=False)
    monkeypatch.setattr(NewtonManager, "_log_solver_debug", classmethod(lambda cls: events.append("debug")))
    return events


@pytest.mark.parametrize("use_graph", [False, True])
def test_manager_checks_solver_status_after_eager_or_captured_tick(monkeypatch, use_graph):
    """Sticky device failures are inspected only after physics execution leaves capture."""
    events = _configure_manager_step_test(monkeypatch, use_graph=use_graph)

    NewtonManager.step()

    assert events == ["replay" if use_graph else "eager", "status", "debug"]


@pytest.mark.parametrize(
    "solver", [SimpleNamespace(), SimpleNamespace(check_status=None)], ids=["missing", "non-callable"]
)
def test_legacy_solver_without_callable_status_hook_is_safe_after_eager_tick(monkeypatch, solver):
    """The eager host boundary is a no-op when the pinned solver has no callable status hook."""
    events = _configure_manager_step_test(monkeypatch)
    monkeypatch.setattr(NewtonManager, "_solver", solver, raising=False)

    NewtonManager.step()

    assert events == ["eager", "debug"]


def test_manager_does_not_publish_time_when_solver_status_fails(monkeypatch):
    """A failed sparse rebuild is rejected before simulation-time bookkeeping."""
    from isaaclab.physics import PhysicsManager

    _configure_manager_step_test(monkeypatch, status_error=RuntimeError("sparse capacity exceeded"))

    with pytest.raises(RuntimeError, match="sparse capacity exceeded"):
        NewtonManager.step()

    assert PhysicsManager._sim_time == 0.0


def test_outer_capture_defers_status_to_public_host_boundary(monkeypatch):
    """Application-owned capture records the step and checks status only after replay."""
    events = _configure_manager_step_test(monkeypatch, externally_capturing=True)

    NewtonManager.step()
    assert events == ["eager", "debug"]

    NewtonManager.check_solver_status()
    assert events == ["eager", "debug", "status"]


@pytest.mark.parametrize("supported", [True, False])
def test_initialize_solver_prepares_only_supported_cuda_graphs(monkeypatch, supported):
    """Graph preparation precedes prewarming and unsupported solvers stay eager."""
    from isaaclab.physics import PhysicsManager

    events = []
    contacts = object()
    solver = SimpleNamespace(
        supports_cuda_graph_capture=supported,
        prepare_cuda_graph_capture=lambda received: events.append(("prepare", received)),
        check_status=lambda: events.append(("status", None)),
    )
    cfg = SimpleNamespace(
        num_substeps=1,
        collision_decimation=0,
        collision_cfg=None,
        solver_cfg=object(),
        use_cuda_graph=True,
    )

    monkeypatch.setattr(PhysicsManager, "_cfg", cfg, raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", "cuda:0", raising=False)
    monkeypatch.setattr(PhysicsManager, "_sim_time", 1.0, raising=False)
    monkeypatch.setattr(NewtonManager, "_model", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_contacts", contacts, raising=False)
    monkeypatch.setattr(NewtonManager, "_usdrt_stage", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_use_newton_actuators_active", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_post_solver_init_callbacks", {}, raising=False)
    monkeypatch.setattr(NewtonManager, "get_physics_dt", classmethod(lambda cls: 1.0 / 120.0))
    monkeypatch.setattr(
        NewtonManager,
        "_eval_fk_impl",
        classmethod(lambda cls, world_mask, fk_mask: events.append(("fk", world_mask, fk_mask))),
    )

    def build_solver(cls, model, solver_cfg):
        NewtonManager._solver = solver

    monkeypatch.setattr(NewtonManager, "_build_solver", classmethod(build_solver))
    monkeypatch.setattr(NewtonManager, "_initialize_contacts", classmethod(lambda cls: None))
    monkeypatch.setattr(
        NewtonManager,
        "_prewarm_cuda_graph_allocations",
        classmethod(lambda cls: events.append(("prewarm", None))),
    )
    monkeypatch.setattr(
        NewtonManager,
        "_capture_or_defer_graph",
        classmethod(lambda cls: events.append(("capture", None))),
    )

    NewtonManager.initialize_solver()

    if supported:
        assert events == [
            ("fk", None, None),
            ("prepare", contacts),
            ("prewarm", None),
            ("status", None),
            ("capture", None),
        ]
    else:
        assert events == [("fk", None, None), ("capture", None)]


def test_prepare_cuda_graph_capture_keeps_graph_ownership_external(monkeypatch):
    """Public preparation makes eager stepping capturable without creating a nested graph."""
    from isaaclab.physics import PhysicsManager

    events = []
    contacts = object()
    solver = SimpleNamespace(
        supports_cuda_graph_capture=True,
        prepare_cuda_graph_capture=lambda received: events.append(("prepare", received)),
        check_status=lambda: events.append(("status", None)),
    )
    monkeypatch.setattr(PhysicsManager, "_cfg", SimpleNamespace(use_cuda_graph=False), raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", "cuda:0", raising=False)
    monkeypatch.setattr(PhysicsManager, "_sim_time", 0.0, raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", solver, raising=False)
    monkeypatch.setattr(NewtonManager, "_contacts", contacts, raising=False)
    monkeypatch.setattr(NewtonManager, "_graph", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_graph_capture_pending", False, raising=False)
    monkeypatch.setattr(
        NewtonManager,
        "_prewarm_cuda_graph_allocations",
        classmethod(lambda cls: events.append(("prewarm", None))),
    )

    NewtonManager.prepare_cuda_graph_capture()

    assert events == [("prepare", contacts), ("prewarm", None), ("status", None)]
    assert NewtonManager.is_cuda_graph_active() is False
    assert NewtonManager._graph_capture_pending is False


@pytest.mark.parametrize(
    "solver",
    [
        SimpleNamespace(supports_cuda_graph_capture=True),
        SimpleNamespace(supports_cuda_graph_capture=True, prepare_cuda_graph_capture=None),
    ],
    ids=["missing", "non-callable"],
)
def test_prepare_cuda_graph_capture_rejects_missing_solver_prepare_hook(monkeypatch, solver):
    """A solver cannot advertise capture without providing a callable preparation hook."""
    from isaaclab.physics import PhysicsManager

    monkeypatch.setattr(PhysicsManager, "_device", "cuda:0", raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", solver, raising=False)
    monkeypatch.setattr(NewtonManager, "_is_outer_cuda_graph_capture_active", classmethod(lambda cls: False))

    with pytest.raises(RuntimeError, match="prepare_cuda_graph_capture"):
        NewtonManager._prepare_cuda_graph_capture_resources()


def test_prepare_cuda_graph_capture_rejects_an_active_trajectory(monkeypatch):
    """Allocation warmup cannot silently reset MPM history after simulation has advanced."""
    from isaaclab.physics import PhysicsManager

    solver = SimpleNamespace(supports_cuda_graph_capture=True)
    monkeypatch.setattr(PhysicsManager, "_device", "cuda:0", raising=False)
    monkeypatch.setattr(PhysicsManager, "_sim_time", 0.25, raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", solver, raising=False)

    with pytest.raises(RuntimeError, match="before the first simulation step"):
        NewtonManager.prepare_cuda_graph_capture()


def test_builder_world_hook_registration_is_public_and_idempotent(monkeypatch):
    """Tasks can extend each replicated Newton world without mutating manager internals."""

    def hook(builder, env_id, position, quaternion):
        pass

    monkeypatch.setattr(NewtonManager, "_per_world_builder_hooks", [], raising=False)

    NewtonManager.register_builder_world_hook(hook)
    NewtonManager.register_builder_world_hook(hook)
    assert NewtonManager._per_world_builder_hooks == [hook]

    NewtonManager.unregister_builder_world_hook(hook)
    NewtonManager.unregister_builder_world_hook(hook)
    assert NewtonManager._per_world_builder_hooks == []


def test_builder_world_hooks_dispatch_in_registration_order(monkeypatch):
    """Every live replication path can use one mutation-safe hook dispatcher."""
    events = []
    builder = object()

    def first(received_builder, env_id, position, quaternion):
        events.append(("first", received_builder, env_id, position, quaternion))
        NewtonManager.unregister_builder_world_hook(first)

    def second(received_builder, env_id, position, quaternion):
        events.append(("second", received_builder, env_id, position, quaternion))

    monkeypatch.setattr(NewtonManager, "_per_world_builder_hooks", [first, second], raising=False)

    NewtonManager._run_builder_world_hooks(builder, 3, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])

    assert events == [
        ("first", builder, 3, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]),
        ("second", builder, 3, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]),
    ]
    assert NewtonManager._per_world_builder_hooks == [second]


def test_external_capture_odd_substep_copies_without_swapping_python_state(monkeypatch):
    """Outer-graph replay must always advance the same canonical input buffer."""
    from isaaclab.physics import PhysicsManager

    events = []

    class FakeState:
        def __init__(self, name):
            self.name = name

        def assign(self, other):
            events.append(("assign", self.name, other.name))

        def clear_forces(self):
            events.append(("clear", self.name))

    state_0 = FakeState("state_0")
    state_1 = FakeState("state_1")
    monkeypatch.setattr(PhysicsManager, "_cfg", SimpleNamespace(use_cuda_graph=False), raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", state_0, raising=False)
    monkeypatch.setattr(NewtonManager, "_state_1", state_1, raising=False)
    monkeypatch.setattr(NewtonManager, "_num_substeps", 1, raising=False)
    monkeypatch.setattr(NewtonManager, "_collision_decimation", 0, raising=False)
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_use_single_state", False, raising=False)
    monkeypatch.setattr(
        NewtonManager,
        "_is_outer_cuda_graph_capture_active",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        NewtonManager,
        "_step_solver",
        classmethod(
            lambda cls, state_in, state_out, control, contacts, dt: events.append(("step", state_in, state_out))
        ),
    )

    NewtonManager._run_solver_substeps(contacts=None)

    assert events == [("step", state_0, state_1), ("assign", "state_0", "state_1"), ("clear", "state_0")]
    assert NewtonManager._state_0 is state_0
    assert NewtonManager._state_1 is state_1


def test_cuda_graph_warmup_restores_parent_and_solver_private_state(monkeypatch):
    """Allocation warmup cannot leak a future coupled state into the first real step."""

    class FakeState:
        def __init__(self, value=None):
            self.value = value

        def assign(self, other):
            self.value = other.value

    state_0 = FakeState("input")
    state_1 = FakeState("output")
    solver = SimpleNamespace(private_state="input")
    resets = []

    def reset(state, *, world_mask, flags):
        assert world_mask is None and flags is None
        solver.private_state = state.value
        resets.append(state.value)
        state.value = "solver-default"

    solver.reset = reset
    solver.check_status = lambda: None
    monkeypatch.setattr(NewtonManager, "_model", SimpleNamespace(state=FakeState), raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", state_0, raising=False)
    monkeypatch.setattr(NewtonManager, "_state_1", state_1, raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", solver, raising=False)

    def simulate(cls):
        cls._state_0.value = "future-input"
        cls._state_1.value = "future-output"
        solver.private_state = "future-private"

    monkeypatch.setattr(NewtonManager, "_simulate_physics_only", classmethod(simulate))
    monkeypatch.setattr(wp, "synchronize_device", lambda: None)

    NewtonManager._simulate_once_for_cuda_graph_warmup()

    assert state_0.value == "input"
    assert state_1.value == "output"
    assert resets == ["output", "input"]
    assert solver.private_state == "input"


def test_cuda_graph_warmup_restores_parent_when_solver_reset_fails(monkeypatch):
    """A reset error cannot expose the warmup or reset-mutated public state."""

    class FakeState:
        def __init__(self, value=None):
            self.value = value

        def assign(self, other):
            self.value = other.value

    state_0 = FakeState("input")
    state_1 = FakeState("output")

    def reset(state, *, world_mask, flags):
        state.value = "solver-default"
        raise RuntimeError("reset failed")

    solver = SimpleNamespace(reset=reset, check_status=lambda: None)
    monkeypatch.setattr(NewtonManager, "_model", SimpleNamespace(state=FakeState), raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", state_0, raising=False)
    monkeypatch.setattr(NewtonManager, "_state_1", state_1, raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", solver, raising=False)
    monkeypatch.setattr(
        NewtonManager,
        "_simulate_physics_only",
        classmethod(lambda cls: setattr(cls._state_0, "value", "future-input")),
    )
    monkeypatch.setattr(wp, "synchronize_device", lambda: None)

    with pytest.raises(RuntimeError, match="reset failed"):
        NewtonManager._simulate_once_for_cuda_graph_warmup()

    assert state_0.value == "input"
    assert state_1.value == "output"


def _configure_capture_state_preservation_test(monkeypatch):
    """Install minimal manager state whose public and private values can be tracked."""

    class FakeState:
        def __init__(self, value=None):
            self.value = value

        def assign(self, other):
            self.value = other.value

    class FakeSolver:
        supports_cuda_graph_capture = True

        def __init__(self):
            self.private_state = "input"
            self.scratch_allocations = []

        def reset(self, state, *, world_mask, flags):
            assert world_mask is None and flags is None
            self.private_state = state.value
            state.value = "solver-default"

        def check_status(self):
            pass

    state_0 = FakeState("input")
    state_1 = FakeState("output")
    solver = FakeSolver()
    monkeypatch.setattr(NewtonManager, "_model", SimpleNamespace(state=FakeState), raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", state_0, raising=False)
    monkeypatch.setattr(NewtonManager, "_state_1", state_1, raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", solver, raising=False)
    monkeypatch.setattr(wp, "synchronize_device", lambda: None)
    return state_0, state_1, solver, FakeSolver


@pytest.mark.parametrize("fails", [False, True])
def test_relaxed_capture_warmup_preserves_state(monkeypatch, fails):
    """Relaxed-capture allocation warmup restores state after success or failure."""
    import isaaclab_newton.physics.newton_manager as newton_manager_module

    state_0, state_1, solver, _ = _configure_capture_state_preservation_test(monkeypatch)
    scratch_allocations = solver.scratch_allocations

    def warmup(cls):
        solver.scratch_allocations.append(object())
        cls._state_0.value = "future-input"
        cls._state_1.value = "future-output"
        cls._state_0, cls._state_1 = cls._state_1, cls._state_0
        solver.private_state = "future-private"
        if fails:
            raise RuntimeError("warmup failed")

    fake_cudart = SimpleNamespace(cudaStreamCreateWithFlags=lambda *_args: 1)
    monkeypatch.setattr(newton_manager_module, "_cudart", fake_cudart)
    monkeypatch.setattr(NewtonManager, "_is_all_graphable", classmethod(lambda cls: False))
    monkeypatch.setattr(NewtonManager, "_simulate_physics_only", classmethod(warmup))
    monkeypatch.setattr(wp, "get_stream", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(wp, "synchronize_stream", lambda *_args, **_kwargs: None)

    if fails:
        with pytest.raises(RuntimeError, match="warmup failed"):
            NewtonManager._capture_relaxed_graph("cpu")
    else:
        assert NewtonManager._capture_relaxed_graph("cpu") is None

    assert NewtonManager._state_0 is state_0
    assert NewtonManager._state_1 is state_1
    assert state_0.value == "input"
    assert state_1.value == "output"
    assert solver.private_state == "input"
    assert solver.scratch_allocations is scratch_allocations
    assert len(scratch_allocations) == 1


@pytest.mark.parametrize("fails", [False, True])
def test_kamino_post_capture_replay_preserves_state(monkeypatch, fails):
    """Kamino's allocation-pinning replay restores state after success or failure."""
    import isaaclab_newton.physics.newton_manager as newton_manager_module

    from isaaclab.physics import PhysicsManager

    state_0, state_1, solver, fake_solver_type = _configure_capture_state_preservation_test(monkeypatch)
    graph = object()
    scratch_allocations = solver.scratch_allocations

    class FakeCapture:
        def __init__(self, *, device):
            assert device == "cuda:0"

        def __enter__(self):
            self.graph = graph
            return self

        def __exit__(self, *_args):
            return False

    def replay(_graph):
        solver.scratch_allocations.append(object())
        NewtonManager._state_0.value = "future-input"
        NewtonManager._state_1.value = "future-output"
        NewtonManager._state_0, NewtonManager._state_1 = NewtonManager._state_1, NewtonManager._state_0
        solver.private_state = "future-private"
        if fails:
            raise RuntimeError("replay failed")

    monkeypatch.setattr(newton_manager_module, "SolverKamino", fake_solver_type)
    monkeypatch.setattr(PhysicsManager, "_cfg", SimpleNamespace(use_cuda_graph=True), raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", "cuda:0", raising=False)
    monkeypatch.setattr(NewtonManager, "_usdrt_stage", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_graph", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_graph_capture_pending", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_is_all_graphable", classmethod(lambda cls: False))
    monkeypatch.setattr(NewtonManager, "_simulate_physics_only", classmethod(lambda cls: None))
    monkeypatch.setattr(wp, "ScopedCapture", FakeCapture)
    monkeypatch.setattr(wp, "capture_launch", replay)

    if fails:
        with pytest.raises(RuntimeError, match="replay failed"):
            NewtonManager._capture_or_defer_graph()
    else:
        NewtonManager._capture_or_defer_graph()

    assert NewtonManager._graph is graph
    assert NewtonManager._state_0 is state_0
    assert NewtonManager._state_1 is state_1
    assert state_0.value == "input"
    assert state_1.value == "output"
    assert solver.private_state == "input"
    assert solver.scratch_allocations is scratch_allocations
    assert len(scratch_allocations) == 1


def _configure_kamino_deferred_step_test(monkeypatch, *, replay_fails):
    """Install a deferred Kamino step whose graph replay exposes state advances."""
    from isaaclab.physics import PhysicsManager

    state_0, state_1, solver, _ = _configure_capture_state_preservation_test(monkeypatch)
    state_0.value = 0
    state_1.value = 10
    solver.private_state = 0
    scratch_allocations = solver.scratch_allocations
    graph = object()
    launches = []

    def replay(captured_graph):
        assert captured_graph is graph
        launches.append(captured_graph)
        if not solver.scratch_allocations:
            solver.scratch_allocations.append(object())
        NewtonManager._state_0.value += 1
        NewtonManager._state_1.value += 1
        solver.private_state += 1
        if replay_fails:
            raise RuntimeError("allocation replay failed")

    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(is_playing=lambda: True), raising=False)
    monkeypatch.setattr(PhysicsManager, "_cfg", SimpleNamespace(use_cuda_graph=True), raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", "cuda:0", raising=False)
    monkeypatch.setattr(PhysicsManager, "_sim_time", 0.0, raising=False)
    monkeypatch.setattr(NewtonManager, "_model_changes", set(), raising=False)
    monkeypatch.setattr(NewtonManager, "_graph_capture_pending", True, raising=False)
    monkeypatch.setattr(NewtonManager, "_graph", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_needs_fk_before_step", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_use_newton_actuators_active", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_adapter", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_post_actuator_callbacks", [], raising=False)
    monkeypatch.setattr(NewtonManager, "_state_teleport_pending", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_world_reset_mask", SimpleNamespace(zero_=lambda: None), raising=False)
    monkeypatch.setattr(NewtonManager, "_fk_reset_mask", SimpleNamespace(zero_=lambda: None), raising=False)
    monkeypatch.setattr(NewtonManager, "_solver_dt", 0.25, raising=False)
    monkeypatch.setattr(NewtonManager, "_num_substeps", 2, raising=False)
    monkeypatch.setattr(NewtonManager, "_usdrt_stage", object(), raising=False)
    monkeypatch.setattr(
        NewtonKaminoManager,
        "_capture_relaxed_graph",
        classmethod(lambda cls, device: graph),
    )
    monkeypatch.setattr(NewtonKaminoManager, "_mark_state_dirty", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonKaminoManager, "_log_solver_debug", classmethod(lambda cls: None))
    monkeypatch.setattr(wp, "capture_launch", replay)

    return state_0, state_1, solver, scratch_allocations, graph, launches


def test_kamino_deferred_capture_replay_does_not_double_advance(monkeypatch):
    """Kamino's deferred allocation replay is invisible to the first real step."""
    from isaaclab.physics import PhysicsManager

    state_0, state_1, solver, scratch_allocations, graph, launches = _configure_kamino_deferred_step_test(
        monkeypatch, replay_fails=False
    )

    NewtonKaminoManager.step()

    assert launches == [graph]
    assert NewtonManager._state_0 is state_0
    assert NewtonManager._state_1 is state_1
    assert state_0.value == 1
    assert state_1.value == 11
    assert solver.private_state == 1
    assert solver.scratch_allocations is scratch_allocations
    assert len(scratch_allocations) == 1
    assert PhysicsManager._sim_time == 0.5


def test_kamino_deferred_capture_replay_failure_does_not_advance_time(monkeypatch):
    """A failed real graph replay propagates without publishing elapsed simulation time."""
    from isaaclab.physics import PhysicsManager

    state_0, state_1, solver, scratch_allocations, graph, launches = _configure_kamino_deferred_step_test(
        monkeypatch, replay_fails=True
    )

    with pytest.raises(RuntimeError, match="allocation replay failed"):
        NewtonKaminoManager.step()

    assert launches == [graph]
    assert NewtonManager._state_0 is state_0
    assert NewtonManager._state_1 is state_1
    assert state_0.value == 1
    assert state_1.value == 11
    assert solver.private_state == 1
    assert solver.scratch_allocations is scratch_allocations
    assert len(scratch_allocations) == 1
    assert PhysicsManager._sim_time == 0.0


def test_reset_solver_state_forwards_public_reset_contract(monkeypatch):
    """Default reset finishes from the authoritative input buffer; explicit reset stays singular."""
    calls = []
    current_input_state = object()
    current_output_state = object()
    explicit_state = object()
    world_mask = object()
    solver = SimpleNamespace(
        reset=lambda state, *, world_mask, flags: calls.append((state, world_mask, flags)),
    )
    monkeypatch.setattr(NewtonManager, "_solver", solver, raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", current_input_state, raising=False)
    monkeypatch.setattr(NewtonManager, "_state_1", current_output_state, raising=False)

    NewtonManager.reset_solver_state(world_mask=world_mask, flags=17)
    NewtonManager.reset_solver_state(state=explicit_state, world_mask=world_mask, flags=None)

    assert calls == [
        (current_output_state, world_mask, 17),
        (current_input_state, world_mask, 17),
        (explicit_state, world_mask, None),
    ]


def test_get_clone_prototype_model_copies_mutable_builder_state_and_geometry(monkeypatch):
    """Auxiliary prototype finalization cannot mutate geometry shared with the live model."""

    class FakeGeometry:
        def __init__(self, name):
            self.name = name

        def copy(self):
            return FakeGeometry(f"{self.name}-copy")

    class FakeBuilder:
        def __init__(self):
            self.values = [1]
            self.mapping = {"rows": [2]}
            self.labels = {"source"}
            self.shape_source = [FakeGeometry("mesh"), None]

        def finalize(self, device):
            self.values.append(3)
            self.mapping["rows"].append(4)
            self.labels.add("finalized")
            return SimpleNamespace(device=device, shape_source=self.shape_source)

    prototype = FakeBuilder()
    monkeypatch.setattr(NewtonManager, "_model", SimpleNamespace(device="cuda:0"), raising=False)
    monkeypatch.setattr(NewtonManager, "_cl_protos", {"/World/envs/env_0": prototype}, raising=False)

    model = NewtonManager.get_clone_prototype_model("/World/envs/env_0", device="cuda:1")

    assert model.device == "cuda:1"
    assert model.shape_source[0].name == "mesh-copy"
    assert model.shape_source[0] is not prototype.shape_source[0]
    assert prototype.values == [1]
    assert prototype.mapping == {"rows": [2]}
    assert prototype.labels == {"source"}


def test_get_clone_prototype_model_rejects_uninitialized_or_unknown_source(monkeypatch):
    monkeypatch.setattr(NewtonManager, "_model", None, raising=False)
    with pytest.raises(RuntimeError, match="not initialized"):
        NewtonManager.get_clone_prototype_model("/World/envs/env_0")

    monkeypatch.setattr(NewtonManager, "_model", SimpleNamespace(device="cuda:0"), raising=False)
    monkeypatch.setattr(NewtonManager, "_cl_protos", {"/World/envs/env_0": object()}, raising=False)
    with pytest.raises(RuntimeError, match="Available: /World/envs/env_0"):
        NewtonManager.get_clone_prototype_model("/World/envs/env_1")


def test_reset_solver_state_deduplicates_in_place_manager_buffer(monkeypatch):
    """An in-place solver must receive one reset when both manager states alias."""
    calls = []
    shared_state = object()
    solver = SimpleNamespace(reset=lambda state, *, world_mask, flags: calls.append(state))
    monkeypatch.setattr(NewtonManager, "_solver", solver, raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", shared_state, raising=False)
    monkeypatch.setattr(NewtonManager, "_state_1", shared_state, raising=False)

    NewtonManager.reset_solver_state()

    assert calls == [shared_state]


def test_reset_solver_state_rejects_uninitialized_solver(monkeypatch):
    """Resetting before solver initialization reports a clear lifecycle error."""
    monkeypatch.setattr(NewtonManager, "_solver", None, raising=False)

    with pytest.raises(RuntimeError, match="not initialized"):
        NewtonManager.reset_solver_state()


def test_mpm_unsupported_cuda_graph_capture_uses_eager_execution(monkeypatch):
    """An unsupported MPM configuration should not enter a graph capture window."""
    from isaaclab.physics import PhysicsManager

    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        NewtonCfg(solver_cfg=MPMSolverCfg(grid_type="sparse"), use_cuda_graph=True),
        raising=False,
    )
    monkeypatch.setattr(PhysicsManager, "_device", "cuda:0", raising=False)
    monkeypatch.setattr(
        NewtonManager,
        "_solver",
        SimpleNamespace(grid_type="sparse", supports_cuda_graph_capture=False),
        raising=False,
    )
    monkeypatch.setattr(NewtonManager, "_graph", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_graph_capture_pending", True, raising=False)

    NewtonMPMManager._capture_or_defer_graph()

    assert NewtonManager._graph is None
    assert NewtonManager._graph_capture_pending is False


# ---------------------------------------------------------------------------
# Manager class hierarchy and factory contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "manager",
    [
        NewtonMJWarpManager,
        NewtonXPBDManager,
        NewtonFeatherstoneManager,
        NewtonKaminoManager,
        NewtonMPMManager,
        NewtonCoupledManager,
    ],
)
def test_subclass_of_newton_manager(manager):
    """All concrete managers inherit from :class:`NewtonManager`."""
    assert issubclass(manager, NewtonManager)
    # Subclasses must override the abstract factory.
    assert manager._build_solver is not NewtonManager._build_solver


def test_abstract_build_solver_raises():
    """Calling :meth:`_build_solver` on the abstract base raises."""
    with pytest.raises(NotImplementedError):
        NewtonManager._build_solver(model=None, solver_cfg=NewtonSolverCfg())


@pytest.mark.parametrize(
    "manager",
    [
        NewtonMJWarpManager,
        NewtonXPBDManager,
        NewtonFeatherstoneManager,
        NewtonKaminoManager,
        NewtonMPMManager,
        NewtonCoupledManager,
    ],
)
def test_manager_name_starts_with_newton(manager):
    """The ``"newton"`` prefix is required by :class:`InteractiveScene` and the
    various backend factories that dispatch on ``physics_manager.__name__.lower()``.
    """
    assert manager.__name__.lower().startswith("newton")


def test_coupled_entry_threads_generic_entry_options():
    """Isaac Lab entry cfg exposes Newton's generic SolverCoupled.Entry options."""

    def _configure_view(_view):
        return None

    entry = NewtonCoupledManager._build_entry(
        CoupledSolverEntryCfg(
            name="xpbd",
            solver_cfg=XPBDSolverCfg(iterations=1),
            particles=[0],
            configure_view=_configure_view,
            in_place=True,
            preserve_shape_ids=False,
        )
    )
    assert entry.configure_view is _configure_view
    assert callable(entry.solver)
    assert entry.in_place is True
    assert entry.preserve_shape_ids is False


def test_coupled_proxy_int_mode_is_normalized():
    """Integer proxy modes are normalized before constructing Newton proxy configs."""
    proxy = NewtonCoupledManager._build_proxy(CoupledProxyCfg(source="src", destination="dst", particles=[0], mode=1))
    assert proxy.mode == "staggered"


def test_coupled_proxy_threads_relaxation():
    """Proxy relaxation is forwarded to Newton's proxy mapping config."""
    proxy = NewtonCoupledManager._build_proxy(
        CoupledProxyCfg(source="src", destination="dst", particles=[0], proxy_relaxation=0.5)
    )
    assert proxy.proxy_relaxation == 0.5


def test_coupled_selectors_resolve_bodies_shapes_joints_particles():
    """Front-end selectors resolve to the raw ids Newton coupled solvers expect."""
    builder = NewtonManager.create_builder()
    base = builder.add_body(mass=1.0, label="/World/envs/env_0/Robot/base")
    finger = builder.add_body(mass=1.0, label="/World/envs/env_0/Robot/finger")
    joint = builder.add_joint_revolute(parent=base, child=finger, axis=(0, 0, 1))
    base_shape = builder.add_shape_box(base, hx=0.05, hy=0.05, hz=0.05)
    finger_shape = builder.add_shape_box(finger, hx=0.02, hy=0.02, hz=0.02)
    builder.add_ground_plane()
    builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.1), vel=wp.vec3(0.0), mass=0.1, radius=0.02)
    builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.2), vel=wp.vec3(0.0), mass=0.1, radius=0.02)
    model = builder.finalize(device="cpu")

    scene_cfg = SimpleNamespace(robot=SimpleNamespace(prim_path="/World/envs/env_.*/Robot"))
    entry = NewtonCoupledManager._resolve_entry_cfg(
        model,
        CoupledSolverEntryCfg(
            name="rigid",
            solver_cfg=XPBDSolverCfg(iterations=1),
            body_entities=[SceneEntityCfg("robot")],
            particle_range=(0, None),
            include_static_shapes=True,
        ),
        scene_cfg,
    )
    assert entry.bodies == [base, finger]
    assert joint in entry.joints
    assert entry.shapes == []
    assert entry.particles == [0, 1]

    body_shape_entry = NewtonCoupledManager._resolve_entry_cfg(
        model,
        CoupledSolverEntryCfg(
            name="rigid",
            solver_cfg=XPBDSolverCfg(iterations=1),
            body_entities=[SceneEntityCfg("robot")],
            include_static_shapes=False,
        ),
        scene_cfg,
    )
    assert body_shape_entry.shapes == [base_shape, finger_shape]

    proxy = NewtonCoupledManager._resolve_proxy_cfg(
        model,
        CoupledProxyCfg(
            source="rigid",
            destination="soft",
            body_entities=[SceneEntityCfg("robot", body_names=["finger"])],
            particle_range=(1, None),
        ),
        scene_cfg,
    )
    assert proxy.bodies == [finger]
    assert proxy.particles == [1]

    local_id_entry = NewtonCoupledManager._resolve_entry_cfg(
        model,
        CoupledSolverEntryCfg(
            name="finger",
            solver_cfg=XPBDSolverCfg(iterations=1),
            body_entities=[SceneEntityCfg("robot", body_ids=[1])],
        ),
        scene_cfg,
    )
    assert local_id_entry.bodies == [finger]


def test_coupled_scene_entity_selectors_require_scene_cfg():
    """SceneEntityCfg selectors fail early when the solver cfg has no scene cfg."""
    builder = NewtonManager.create_builder()
    builder.add_body(mass=1.0, label="/World/envs/env_0/Robot/base")
    model = builder.finalize(device="cpu")

    with pytest.raises(ValueError, match="scene_cfg"):
        NewtonCoupledManager._resolve_entry_cfg(
            model,
            CoupledSolverEntryCfg(name="rigid", solver_cfg=XPBDSolverCfg(), body_entities=[SceneEntityCfg("robot")]),
            None,
        )


@pytest.mark.parametrize(
    "cfg, match",
    [
        (
            CoupledSolverCfg(
                entries=[
                    CoupledSolverEntryCfg(name="a", solver_cfg=XPBDSolverCfg()),
                    CoupledSolverEntryCfg(name="a", solver_cfg=XPBDSolverCfg()),
                ],
            ),
            "Duplicate",
        ),
        (
            CoupledSolverCfg(
                entries=[
                    CoupledSolverEntryCfg(name="a", solver_cfg=XPBDSolverCfg(), in_place=True, substeps=2),
                    CoupledSolverEntryCfg(name="b", solver_cfg=XPBDSolverCfg()),
                ],
            ),
            "in_place requires substeps=1",
        ),
        (
            CoupledSolverCfg(
                entries=[
                    CoupledSolverEntryCfg(name="a", solver_cfg=XPBDSolverCfg(), shapes=[0]),
                    CoupledSolverEntryCfg(name="b", solver_cfg=XPBDSolverCfg(), shapes=[0]),
                ],
            ),
            "shapes index 0 is owned by both",
        ),
        (
            CoupledSolverCfg(
                entries=[
                    CoupledSolverEntryCfg(name="a", solver_cfg=XPBDSolverCfg()),
                    CoupledSolverEntryCfg(name="b", solver_cfg=XPBDSolverCfg()),
                ],
                proxy_coupling=ProxyCouplingCfg(
                    proxies=[CoupledProxyCfg(source="missing", destination="b", particles=[0])]
                ),
            ),
            "source 'missing'",
        ),
        (
            CoupledSolverCfg(
                entries=[
                    CoupledSolverEntryCfg(name="a", solver_cfg=XPBDSolverCfg()),
                    CoupledSolverEntryCfg(name="b", solver_cfg=XPBDSolverCfg()),
                ],
                proxy_coupling=ProxyCouplingCfg(
                    proxies=[CoupledProxyCfg(source="a", destination="b", particles=[0], mode=2)]
                ),
            ),
            "Unsupported CoupledProxyCfg mode",
        ),
        (
            CoupledSolverCfg(
                entries=[
                    CoupledSolverEntryCfg(name="a", solver_cfg=XPBDSolverCfg()),
                    CoupledSolverEntryCfg(name="b", solver_cfg=XPBDSolverCfg()),
                ],
                proxy_coupling=ProxyCouplingCfg(
                    proxies=[CoupledProxyCfg(source="a", destination="b", particles=[0], proxy_relaxation=-0.5)]
                ),
            ),
            "proxy_relaxation must be finite",
        ),
        (
            CoupledSolverCfg(
                coupling_type="admm",
                entries=[
                    CoupledSolverEntryCfg(name="a", solver_cfg=XPBDSolverCfg()),
                    CoupledSolverEntryCfg(name="b", solver_cfg=XPBDSolverCfg()),
                ],
                admm_coupling=AdmmCouplingCfg(contact_pairs=[AdmmContactPairCfg(source="a", destination="a")]),
            ),
            "source and destination",
        ),
    ],
)
def test_coupled_cfg_validation_rejects_invalid_configs(cfg, match):
    """Invalid coupled configs fail before Newton constructs sub-solvers."""
    with pytest.raises(ValueError, match=match):
        NewtonCoupledManager._validate_solver_cfg(cfg)


# ---------------------------------------------------------------------------
# End-to-end: build each solver via SimulationContext
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver_cfg_factory, expected_manager, expected_solver_cls,"
    " expected_use_single_state, expected_needs_collision_pipeline",
    SOLVER_MATRIX,
)
def test_initialize_solver_populates_canonical_state(
    solver_cfg_factory,
    expected_manager,
    expected_solver_cls,
    expected_use_single_state,
    expected_needs_collision_pipeline,
):
    """End-to-end: ``SimulationContext`` resolves the right manager subclass and
    ``initialize_solver`` lands the right solver + flags on :class:`NewtonManager`.

    External code reads :class:`NewtonManager` attributes directly (``_solver``,
    ``_use_single_state``, ``_needs_collision_pipeline``).  Even though dispatch
    runs through a leaf subclass (e.g. :class:`NewtonMJWarpManager`), shared
    state is assigned through the explicit base class so that those reads keep
    working regardless of which leaf is active.  This test is the regression
    guard for that contract.

    The builder is pre-populated directly (instead of relying on a USD stage)
    with either a minimal particle grid for MPM or a one-body / one-joint
    scene for rigid/articulation solvers:

    1. :class:`SolverImplicitMPM` requires particles and MPM custom attributes
       registered on the builder before particle creation.
    2. :class:`SolverMuJoCo` requires at least one joint to convert the model
       to MJCF; a ground-plane-only scene fails MJCF conversion.
    3. Pre-populating ``NewtonManager._builder`` causes
       :meth:`NewtonManager.start_simulation` to skip
       :meth:`instantiate_builder_from_stage`, so the test does not depend on
       USD asset packages.
    """
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(solver_cfg=solver_cfg_factory(), use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        # Resolved manager class matches the expected leaf.
        resolved_manager = sim.physics_manager
        # ``physics_manager`` is a LazyType proxy — compare by ``__name__`` to
        # avoid forcing identity-by-id checks against the unresolved proxy.
        assert resolved_manager.__name__ == expected_manager.__name__
        assert resolved_manager.__name__.lower().startswith("newton")

        builder = resolved_manager.create_builder()
        if expected_solver_cls is SolverImplicitMPM:
            assert builder.has_custom_attribute("mpm:young_modulus")
            builder.add_particle_grid(
                pos=wp.vec3(-0.05, -0.05, 0.10),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=2,
                dim_y=2,
                dim_z=2,
                cell_x=0.05,
                cell_y=0.05,
                cell_z=0.05,
                mass=0.01,
                jitter=0.0,
                radius_mean=0.02,
            )
        elif expected_solver_cls is SolverCoupledProxy:
            assert builder.has_custom_attribute("mpm:young_modulus")
            body = builder.add_body(mass=1.0)
            builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
            builder.add_ground_plane()
            builder.add_particle_grid(
                pos=wp.vec3(-0.05, -0.05, 0.10),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=2,
                dim_y=2,
                dim_z=2,
                cell_x=0.05,
                cell_y=0.05,
                cell_z=0.05,
                mass=0.01,
                jitter=0.0,
                radius_mean=0.02,
            )
        elif expected_solver_cls is SolverCoupled:
            body = builder.add_body(mass=1.0)
            builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
            builder.add_particle(
                pos=wp.vec3(0.0, 0.0, 0.1),
                vel=wp.vec3(0.0),
                mass=0.1,
                radius=0.02,
            )
        elif expected_solver_cls is SolverCoupledADMM:
            assert builder.has_custom_attribute("coupling:body_particle_attachment_body")
            body = builder.add_body(mass=1.0)
            particle = builder.add_particle(
                pos=wp.vec3(0.0, 0.0, 0.0),
                vel=wp.vec3(0.0),
                mass=0.1,
                radius=0.02,
            )
            SolverCoupledADMM.add_body_particle_attachment(builder, body, particle, stiffness=10.0)
        else:
            # Pre-populate the builder with a minimal scene so MJCF conversion has
            # something to work with.
            body = builder.add_body(mass=1.0)
            builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
        NewtonManager.set_builder(builder)

        # Force resolution and bring up the solver.
        sim.reset()

        # Canonical state lives on the base class.
        assert NewtonManager._solver is not None
        assert isinstance(NewtonManager._solver, expected_solver_cls)
        if expected_solver_cls is SolverCoupled:
            assert NewtonCoupledManager.get_entry_solver("rigid") is not None
            assert NewtonCoupledManager.get_entry_solver("particle") is not None
        if expected_solver_cls is SolverCoupledProxy:
            assert NewtonCoupledManager.get_entry_solver("rigid") is not None
            assert NewtonCoupledManager.get_entry_solver("sand") is not None
        if expected_solver_cls is SolverCoupledADMM:
            assert NewtonCoupledManager.get_entry_solver("rigid") is not None
            assert NewtonCoupledManager.get_entry_solver("particle") is not None
        assert NewtonManager._use_single_state is expected_use_single_state
        assert NewtonManager._needs_collision_pipeline is expected_needs_collision_pipeline

        # ``_contacts`` is allocated whichever way contacts are handled
        # (MuJoCo internal buffer or Newton pipeline output).
        # Kamino with internal contacts does not currently set NewtonManager._contacts.
        if expected_needs_collision_pipeline and expected_solver_cls not in (SolverKamino, SolverImplicitMPM):
            assert NewtonManager._contacts is not None

        # One step should not raise — proves the dispatch wiring lines up
        # end-to-end.  (We do not assert physics; that's covered by the
        # asset/sensor test suites.)
        sim.step(render=False)


def test_mjwarp_internal_contacts_with_collision_cfg_raises():
    """Combining ``use_mujoco_contacts=True`` with a ``collision_cfg`` is rejected.

    The check lives in :meth:`NewtonMJWarpManager._build_solver` because it
    needs both the solver cfg subtype and the parent :class:`NewtonCfg`, so it
    fires during :meth:`NewtonManager.initialize_solver` (i.e. on
    ``sim.reset()``) rather than at cfg construction time.
    """
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True),
            collision_cfg=NewtonCollisionPipelineCfg(),
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = sim.physics_manager.create_builder()
        body = builder.add_body(mass=1.0)
        builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
        NewtonManager.set_builder(builder)

        with pytest.raises(ValueError, match="collision_cfg cannot be set"):
            sim.reset()


@pytest.mark.parametrize(
    "num_substeps, collision_decimation, expected_mid_loop_collides",
    [
        (8, 0, 0),  # Feature disabled.
        (8, 2, 3),  # Re-collide after substeps 2, 4, 6 (skip last).
        (8, 4, 1),  # Re-collide after substep 4 only.
        (8, 7, 1),  # Re-collide after substep 7 only.
        (8, 8, 0),  # Gated off (>= num_substeps).
    ],
)
def test_collision_decimation_invokes_mid_loop_collide(num_substeps, collision_decimation, expected_mid_loop_collides):
    """``_run_solver_substeps`` re-invokes ``collide`` at the expected substeps.

    Wraps :attr:`NewtonManager._collision_pipeline.collide` with a counter and
    runs one physics tick. The collide-call count is ``1`` (top-of-tick) plus
    one per matching mid-loop substep, excluding the last substep.

    The scene has a free-joint sphere falling onto a ground plane so the
    broadphase actually generates pairs — guards against a future change
    that skips ``collide()`` when there are no collidable shapes.
    """
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False),
            num_substeps=num_substeps,
            collision_decimation=collision_decimation,
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = sim.physics_manager.create_builder()
        body = builder.add_body(mass=1.0)
        builder.add_joint_free(child=body)
        builder.add_shape_sphere(body=body, radius=0.05)
        builder.add_ground_plane()
        # Lift the sphere to 0.5 m above the plane so the scene is non-degenerate.
        # joint_q for a free joint is [tx, ty, tz, qx, qy, qz, qw].
        builder.joint_q[-7:] = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
        NewtonManager.set_builder(builder)
        sim.reset()

        # Wrap collide() with a counter — must run after sim.reset() so the
        # pipeline is allocated, and use_cuda_graph=False so the wrapped
        # Python callable isn't bypassed by a captured graph.
        calls = {"n": 0}
        original_collide = NewtonManager._collision_pipeline.collide

        def counting_collide(state, contacts):
            calls["n"] += 1
            return original_collide(state, contacts)

        NewtonManager._collision_pipeline.collide = counting_collide
        try:
            sim.step(render=False)
        finally:
            NewtonManager._collision_pipeline.collide = original_collide

        # Expect: 1 (top-of-tick) + expected_mid_loop_collides.
        assert calls["n"] == 1 + expected_mid_loop_collides


# ---------------------------------------------------------------------------
# Regression: an env reset written through the data layer must land in the
# manager's canonical _state_0 after an odd number of steps when CUDA graphs
# are disabled (the use_cuda_graph state-swap gating bug).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_steps", [1, 3])
def test_reset_lands_in_state_0_after_odd_kamino_steps_without_cuda_graph(num_steps):
    """An env reset written through the data-layer binding lands in ``_state_0``.

    Kamino is double-buffered (``_use_single_state=False``), so each substep
    ping-pongs ``_state_0`` / ``_state_1``. With a single substep the loop must
    copy the result back into ``_state_0`` instead of swapping, otherwise after
    an *odd* number of steps the canonical ``_state_0`` ends up on the other
    buffer. This copy-on-last was previously gated on ``use_cuda_graph``, so with
    CUDA graphs disabled ``_state_0`` flipped buffers and env-reset writes landed
    in the stale buffer.

    :class:`~isaaclab_newton.assets.ArticulationData` binds its joint-state write
    target to ``_state_0.joint_q`` once at setup (``_sim_bind_joint_pos``) and
    never re-binds on env resets, so a flipped ``_state_0`` makes reset writes
    miss the live state. This test reproduces that contract without a full USD
    articulation: it caches the same ``_state_0.joint_q`` binding, steps Kamino an
    odd number of times, writes a sentinel through the cached binding (mimicking
    the reset write), and asserts the manager's ``_state_0`` observes it.

    Without the fix the swap-on-last flips ``_state_0`` for odd ``num_steps`` and
    the sentinel lands in ``_state_1`` instead, so the final assertion fails.
    """
    sentinel = 1.2345
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=KaminoSolverCfg(),
            num_substeps=1,
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = NewtonManager.create_builder()
        body = builder.add_body(mass=1.0)
        builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
        NewtonManager.set_builder(builder)
        sim.reset()

        # Kamino keeps separate input/output states; the bug only exists there.
        assert NewtonManager._use_single_state is False
        # The data layer binds its joint-state write target to _state_0 at setup.
        reset_target = NewtonManager._state_0.joint_q
        assert reset_target.shape[0] > 0  # guard against a vacuous assertion

        for _ in range(num_steps):
            sim.step(render=False)

        # An env reset writes joint state through the (still bound) target.
        reset_target.fill_(sentinel)

        # The reset must be visible in the manager's canonical _state_0; if the
        # buffer flipped it landed in _state_1 instead.
        canonical_joint_q = NewtonManager._state_0.joint_q.numpy()
        assert np.allclose(canonical_joint_q, sentinel), (
            f"reset write did not land in _state_0 after {num_steps} steps: {canonical_joint_q}"
        )
