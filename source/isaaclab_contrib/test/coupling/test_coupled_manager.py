# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Pure-Python tests for named Newton coupled-solver configuration.

The tests use small model fakes and never start Isaac Sim. They exercise the
selector, ownership, proxy, and ADMM translation performed before Newton
constructs the coupled solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, XPBDSolverCfg
from newton import ShapeFlags
from newton.solvers.experimental.coupled import CouplingInterface, SolverCoupledADMM, SolverCoupledProxy

from isaaclab.managers import SceneEntityCfg

from isaaclab_contrib.coupling import coupled_manager
from isaaclab_contrib.coupling.coupled_manager import NewtonCoupledSolverManager
from isaaclab_contrib.coupling.coupled_manager_cfg import (
    CoupledAdmmContactPairCfg,
    CoupledAdmmSolverCfg,
    CoupledProxyCfg,
    CoupledProxySolverCfg,
    CoupledSolverCfg,
    CoupledSolverEntryCfg,
)
from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg


@dataclass
class _FakeArray:
    """Minimal mutable stand-in for a Warp array."""

    data: np.ndarray

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    def assign(self, values: np.ndarray) -> None:
        self.data = np.asarray(values).copy()


@dataclass
class _FakeModel:
    """Fields consulted by the manager's pure configuration helpers."""

    body_count: int = 3
    body_label: list[str] = field(
        default_factory=lambda: [
            "/World/envs/env_0/Robot/base",
            "/World/envs/env_0/Robot/hand",
            "/World/envs/env_0/Cable/link",
        ]
    )
    joint_count: int = 2
    joint_child: _FakeArray = field(default_factory=lambda: _FakeArray(np.asarray([1, 2], dtype=np.int32)))
    joint_parent: _FakeArray = field(default_factory=lambda: _FakeArray(np.asarray([0, -1], dtype=np.int32)))
    shape_count: int = 4
    shape_body: _FakeArray = field(default_factory=lambda: _FakeArray(np.asarray([0, 1, 2, -1], dtype=np.int32)))
    shape_flags: _FakeArray = field(
        default_factory=lambda: _FakeArray(np.full(4, int(ShapeFlags.COLLIDE_SHAPES), dtype=np.int32))
    )
    shape_label: list[str] = field(
        default_factory=lambda: [
            "/World/envs/env_0/Robot/base_collision",
            "/World/envs/env_0/Robot/hand_collision",
            "/World/envs/env_0/Cable/cable_collision",
            "/World/ground",
        ]
    )
    particle_count: int = 3
    shape_material_ke: _FakeArray = field(default_factory=lambda: _float_array(4, 1.0))
    shape_material_kd: _FakeArray = field(default_factory=lambda: _float_array(4, 2.0))
    shape_material_mu: _FakeArray = field(default_factory=lambda: _float_array(4, 3.0))
    shape_margin: _FakeArray = field(default_factory=lambda: _float_array(4, 4.0))
    shape_gap: _FakeArray = field(default_factory=lambda: _float_array(4, 5.0))


def _float_array(size: int, value: float) -> _FakeArray:
    return _FakeArray(np.full(size, value, dtype=np.float32))


@dataclass
class _FakeAsset:
    prim_path: str


@dataclass
class _FakeSceneCfg:
    robot: _FakeAsset = field(default_factory=lambda: _FakeAsset("/World/envs/env_.*/Robot"))
    cable: _FakeAsset = field(default_factory=lambda: _FakeAsset("/World/envs/env_.*/Cable"))


def _entry(
    name: str,
    *,
    bodies: list[int] | None = None,
    particles: list[int] | None = None,
) -> CoupledSolverEntryCfg:
    """Build an already-resolved entry for validation tests."""
    entry = CoupledSolverEntryCfg(
        name=name,
        solver_cfg=XPBDSolverCfg(),
        bodies=list(bodies or []),
        particles=list(particles or []),
    )
    entry.joints = []
    entry.shapes = []
    return entry


def _valid_proxy_cfg(*, proxy: CoupledProxyCfg | None = None) -> CoupledProxySolverCfg:
    """Build a complete two-entry proxy configuration."""
    return CoupledProxySolverCfg(
        entries=[
            _entry("rigid", bodies=[0, 1], particles=[0]),
            _entry("soft", bodies=[2], particles=[1, 2]),
        ],
        proxies=[proxy or CoupledProxyCfg(source="rigid", destination="soft", bodies=[1])],
    )


def test_scene_entity_and_string_selectors_resolve_full_body_labels():
    """Scene selectors filter short names while strings match full labels."""
    model = _FakeModel()
    scene_cfg = _FakeSceneCfg()

    assert NewtonCoupledSolverManager._resolve_entity_to_body_ids(
        model,
        SceneEntityCfg("robot", body_names=["hand"]),
        scene_cfg,
        "entry 'rigid'",
    ) == [1]
    assert NewtonCoupledSolverManager._resolve_entity_to_body_ids(
        model,
        "/World/envs/env_.*/Cable/link",
        None,
        "entry 'cable'",
    ) == [2]


def test_scene_entity_selector_expands_environment_namespace_macro():
    """Scene selectors accept the standard unresolved Isaac Lab environment macro."""
    scene_cfg = _FakeSceneCfg(robot=_FakeAsset("{ENV_REGEX_NS}/Robot"))

    assert NewtonCoupledSolverManager._resolve_entity_to_body_ids(
        _FakeModel(),
        SceneEntityCfg("robot"),
        scene_cfg,
        "entry 'rigid'",
    ) == [0, 1]


def test_scene_entity_selector_reports_unmatched_body_pattern():
    with pytest.raises(ValueError, match="no bodies matching"):
        NewtonCoupledSolverManager._resolve_entity_to_body_ids(
            _FakeModel(),
            SceneEntityCfg("robot", body_names=["missing"]),
            _FakeSceneCfg(),
            "entry 'rigid'",
        )


def test_raw_body_label_selector_reports_no_matches():
    with pytest.raises(ValueError, match="matched no Newton bodies"):
        NewtonCoupledSolverManager._resolve_entity_to_body_ids(
            _FakeModel(),
            "/World/envs/env_.*/Missing",
            None,
            "entry 'missing'",
        )


def test_three_named_entries_partition_bodies_joints_shapes_and_particles():
    """Derived joint/shape ownership follows each entry's selected bodies."""
    model = _FakeModel()
    scene_cfg = _FakeSceneCfg()
    entries = [
        CoupledSolverEntryCfg(
            name="rigid",
            solver_cfg=XPBDSolverCfg(),
            bodies=[SceneEntityCfg("robot")],
        ),
        CoupledSolverEntryCfg(
            name="cable",
            solver_cfg=XPBDSolverCfg(),
            bodies=["/World/envs/env_.*/Cable"],
            all_particles=True,
        ),
        CoupledSolverEntryCfg(
            name="world",
            solver_cfg=XPBDSolverCfg(),
            include_static_shapes=True,
        ),
    ]

    resolved = [NewtonCoupledSolverManager._resolve_entry_cfg(model, entry, scene_cfg) for entry in entries]

    assert resolved[0].bodies == [0, 1]
    assert getattr(resolved[0], "joints") == [0]
    assert getattr(resolved[0], "shapes") == [0, 1]
    assert resolved[1].bodies == [2]
    assert getattr(resolved[1], "joints") == [1]
    assert getattr(resolved[1], "shapes") == [2]
    assert resolved[1].particles == [0, 1, 2]
    assert resolved[2].bodies == []
    assert getattr(resolved[2], "joints") == []
    assert getattr(resolved[2], "shapes") == [3]

    cfg = CoupledAdmmSolverCfg(entries=resolved)
    NewtonCoupledSolverManager._validate_solver_cfg(model, cfg)


def test_cross_entry_joint_is_left_unowned_for_admm_attachment():
    """A joint spanning two entries must remain visible to the ADMM coupler."""
    model = _FakeModel()
    model.joint_parent = _FakeArray(np.asarray([0, 1], dtype=np.int32))
    entries = [
        CoupledSolverEntryCfg(
            name="robot",
            solver_cfg=XPBDSolverCfg(),
            bodies=[SceneEntityCfg("robot")],
        ),
        CoupledSolverEntryCfg(
            name="cable",
            solver_cfg=XPBDSolverCfg(),
            bodies=[SceneEntityCfg("cable")],
            all_particles=True,
            include_static_shapes=True,
        ),
    ]

    resolved = [NewtonCoupledSolverManager._resolve_entry_cfg(model, entry, _FakeSceneCfg()) for entry in entries]

    assert resolved[0].joints == [0]
    assert resolved[1].joints == []


def test_proxy_validation_rejects_cross_entry_joint():
    model = _FakeModel()
    model.joint_parent = _FakeArray(np.asarray([0, 1], dtype=np.int32))
    cfg = _valid_proxy_cfg()

    with pytest.raises(ValueError, match="does not support cross-entry joint"):
        NewtonCoupledSolverManager._validate_solver_cfg(model, cfg)


def test_explicit_particle_ownership_accepts_a_complete_partition():
    model = _FakeModel()
    cfg = CoupledAdmmSolverCfg(
        entries=[
            _entry("a", bodies=[0, 1], particles=[0, 2]),
            _entry("b", bodies=[2], particles=[1]),
        ]
    )
    NewtonCoupledSolverManager._validate_solver_cfg(model, cfg)


@pytest.mark.parametrize(
    ("entries", "match"),
    [
        (
            [_entry("a", bodies=[0, 1], particles=[0]), _entry("b", bodies=[1, 2], particles=[1, 2])],
            "bodies index 1 is owned by both",
        ),
        (
            [_entry("a", bodies=[0], particles=[0]), _entry("b", bodies=[1], particles=[1, 2])],
            "unclaimed bodies",
        ),
        (
            [_entry("a", bodies=[0, 1], particles=[0, 1]), _entry("b", bodies=[2], particles=[1, 2])],
            "particles index 1 is owned by both",
        ),
        (
            [_entry("a", bodies=[0, 1], particles=[0]), _entry("b", bodies=[2], particles=[1])],
            "unclaimed particles",
        ),
    ],
)
def test_validation_rejects_duplicate_or_unclaimed_body_and_particle_ownership(entries, match):
    with pytest.raises(ValueError, match=match):
        NewtonCoupledSolverManager._validate_solver_cfg(_FakeModel(), CoupledAdmmSolverCfg(entries=entries))


@pytest.mark.parametrize(
    ("entries", "match"),
    [
        ([_entry("a", bodies=[0, 1], particles=[0, 1, 2])], "at least two named entries"),
        (
            [
                _entry("", bodies=[0, 1], particles=[0]),
                _entry("b", bodies=[2], particles=[1, 2]),
            ],
            "name must be non-empty",
        ),
        (
            [
                _entry("same", bodies=[0, 1], particles=[0]),
                _entry("same", bodies=[2], particles=[1, 2]),
            ],
            "names must be unique",
        ),
        (
            [
                _entry("a", bodies=[0, 1, 3], particles=[0]),
                _entry("b", bodies=[2], particles=[1, 2]),
            ],
            "out-of-range bodies index 3",
        ),
        (
            [
                _entry("a", bodies=[0, 1], particles=[0, 3]),
                _entry("b", bodies=[2], particles=[1, 2]),
            ],
            "out-of-range particles index 3",
        ),
    ],
)
def test_validation_rejects_invalid_entry_names_counts_and_indices(entries, match):
    with pytest.raises(ValueError, match=match):
        NewtonCoupledSolverManager._validate_solver_cfg(_FakeModel(), CoupledAdmmSolverCfg(entries=entries))


def test_shape_label_patterns_and_static_shape_selection_are_additive():
    entry = NewtonCoupledSolverManager._resolve_entry_cfg(
        _FakeModel(),
        CoupledSolverEntryCfg(
            name="special",
            solver_cfg=XPBDSolverCfg(),
            bodies=[SceneEntityCfg("robot", body_names=["base"])],
            include_body_shapes=False,
            include_static_shapes=True,
            shape_label_patterns=[r".*/Cable/cable_collision"],
        ),
        _FakeSceneCfg(),
    )
    assert entry.bodies == [0]
    assert getattr(entry, "shapes") == [3, 2]


@pytest.mark.parametrize(
    ("proxy", "match"),
    [
        (CoupledProxyCfg(source="missing", destination="soft", bodies=[0]), "must name coupled entries"),
        (CoupledProxyCfg(source="rigid", destination="rigid", bodies=[0]), "must differ"),
        (CoupledProxyCfg(source="rigid", destination="soft", bodies=[2]), "owned by its source"),
        (CoupledProxyCfg(source="rigid", destination="soft", particles=[2]), "owned by its source"),
    ],
)
def test_proxy_validation_checks_endpoints_and_source_ownership(proxy, match):
    with pytest.raises(ValueError, match=match):
        NewtonCoupledSolverManager._validate_solver_cfg(_FakeModel(), _valid_proxy_cfg(proxy=proxy))


@pytest.mark.parametrize(
    ("proxy", "match"),
    [
        (CoupledProxyCfg(source="rigid", destination="soft"), "map at least one body or particle"),
        (
            CoupledProxyCfg(source="rigid", destination="soft", bodies=[0], mode="invalid"),
            "mode must be 'lagged' or 'staggered'",
        ),
        (CoupledProxyCfg(source="rigid", destination="soft", bodies=[0], mass_scale=0.0), "mass_scale must be > 0"),
        (
            CoupledProxyCfg(source="rigid", destination="soft", bodies=[0], proxy_relaxation=-0.1),
            "proxy_relaxation must be finite and >= 0",
        ),
        (
            CoupledProxyCfg(source="rigid", destination="soft", bodies=[0], proxy_relaxation=float("inf")),
            "proxy_relaxation must be finite and >= 0",
        ),
        (
            CoupledProxyCfg(source="rigid", destination="soft", bodies=[0], proxy_relaxation=float("nan")),
            "proxy_relaxation must be finite and >= 0",
        ),
        (
            CoupledProxyCfg(source="rigid", destination="soft", bodies=[0], collide_interval=0),
            "collide_interval must be >= 1",
        ),
    ],
)
def test_proxy_validation_rejects_invalid_mapping_options(proxy, match):
    with pytest.raises(ValueError, match=match):
        NewtonCoupledSolverManager._validate_solver_cfg(_FakeModel(), _valid_proxy_cfg(proxy=proxy))


def test_proxy_validation_rejects_more_than_two_entries():
    cfg = CoupledProxySolverCfg(
        entries=[
            _entry("a", bodies=[0], particles=[0]),
            _entry("b", bodies=[1], particles=[1]),
            _entry("c", bodies=[2], particles=[2]),
        ],
        proxies=[CoupledProxyCfg(source="a", destination="b", bodies=[0])],
    )
    with pytest.raises(ValueError, match="at most two solver entries"):
        NewtonCoupledSolverManager._validate_solver_cfg(_FakeModel(), cfg)


def test_proxy_resolution_keeps_only_collidable_selected_bodies():
    model = _FakeModel()
    model.shape_flags = _FakeArray(
        np.asarray([int(ShapeFlags.COLLIDE_SHAPES), 0, int(ShapeFlags.COLLIDE_SHAPES), 0], dtype=np.int32)
    )
    proxy = NewtonCoupledSolverManager._resolve_proxy_cfg(
        model,
        CoupledProxyCfg(source="rigid", destination="soft", bodies=[SceneEntityCfg("robot")]),
        _FakeSceneCfg(),
    )
    assert proxy.bodies == [0]


class _RecordingProxy:
    """Capture proxy construction while retaining Newton's config dataclasses."""

    Proxy = SolverCoupledProxy.Proxy
    Config = SolverCoupledProxy.Config

    def __init__(self, *, model, entries, coupling):
        self.model = model
        self.entries = entries
        self.coupling = coupling


def test_proxy_build_uses_custom_and_default_collision_pipelines(monkeypatch):
    def custom_pipeline(model_view):
        return model_view

    cfg = CoupledProxySolverCfg(
        entries=[
            CoupledSolverEntryCfg(name="rigid", solver_cfg=XPBDSolverCfg()),
            CoupledSolverEntryCfg(name="soft", solver_cfg=XPBDSolverCfg()),
        ],
        proxies=[
            CoupledProxyCfg(
                source="rigid",
                destination="soft",
                bodies=[0],
                proxy_relaxation=0.35,
                collision_pipeline_factory=custom_pipeline,
            ),
            CoupledProxyCfg(source="soft", destination="rigid", particles=[0]),
        ],
        iterations=3,
    )
    monkeypatch.setattr(coupled_manager, "SolverCoupledProxy", _RecordingProxy)

    solver = NewtonCoupledSolverManager._build_proxy_coupled_solver(object(), [], cfg)

    assert solver.coupling.iterations == 3
    assert solver.coupling.proxies[0].proxy_relaxation == pytest.approx(0.35)
    assert solver.coupling.proxies[0].collision_pipeline is custom_pipeline
    assert solver.coupling.proxies[1].collision_pipeline is coupled_manager._default_proxy_collision_pipeline


def test_proxy_build_omits_default_collision_pipeline_for_implicit_mpm_destination(monkeypatch):
    """Implicit MPM handles proxy colliders internally and needs no external contact pipeline."""
    cfg = CoupledProxySolverCfg(
        entries=[
            CoupledSolverEntryCfg(name="rigid", solver_cfg=XPBDSolverCfg()),
            CoupledSolverEntryCfg(name="media", solver_cfg=MPMSolverCfg()),
        ],
        proxies=[CoupledProxyCfg(source="rigid", destination="media", bodies=[0])],
    )
    monkeypatch.setattr(coupled_manager, "SolverCoupledProxy", _RecordingProxy)

    solver = NewtonCoupledSolverManager._build_proxy_coupled_solver(object(), [], cfg)

    assert solver.coupling.proxies[0].collision_pipeline is None


def test_implicit_mpm_entry_builds_solver_with_config_and_temporary_store(monkeypatch):
    """The MPM adapter uses its structured config and an entry-local temporary store."""
    model_view = object()
    solver_config = object()
    temporary_store = object()

    class _RecordingImplicitMPM:
        def __init__(self, model, config, *, temporary_store):
            self.model = model
            self.config = config
            self.temporary_store = temporary_store

    monkeypatch.setitem(
        NewtonCoupledSolverManager._SOLVER_CLASS_BY_CFG_TYPE,
        MPMSolverCfg,
        _RecordingImplicitMPM,
    )
    monkeypatch.setattr(MPMSolverCfg, "to_solver_config", lambda self: solver_config)
    monkeypatch.setattr(coupled_manager, "TemporaryStore", lambda: temporary_store)
    entry_cfg = CoupledSolverEntryCfg(
        name="media",
        solver_cfg=MPMSolverCfg(),
        particles=[0, 1],
        in_place=True,
    )

    entry = NewtonCoupledSolverManager._build_entry(entry_cfg)
    solver = entry.solver(model_view)

    assert isinstance(solver, _RecordingImplicitMPM)
    assert solver.model is model_view
    assert solver.config is solver_config
    assert solver.temporary_store is temporary_store


@pytest.mark.parametrize(("substeps", "in_place"), [(4, False), (1, True)])
def test_entry_build_forwards_substeps_and_in_place(substeps, in_place):
    entry_cfg = CoupledSolverEntryCfg(
        name="entry",
        solver_cfg=XPBDSolverCfg(),
        substeps=substeps,
        in_place=in_place,
    )

    entry = NewtonCoupledSolverManager._build_entry(entry_cfg)

    assert entry.substeps == substeps
    assert entry.in_place is in_place


@pytest.mark.parametrize(
    ("substeps", "in_place", "match"),
    [
        (0, False, "substeps must be >= 1"),
        (2, True, "in_place requires substeps=1"),
    ],
)
def test_entry_validation_rejects_invalid_substeps_and_in_place(substeps, in_place, match):
    cfg = _valid_proxy_cfg()
    cfg.entries[0].substeps = substeps
    cfg.entries[0].in_place = in_place

    with pytest.raises(ValueError, match=match):
        NewtonCoupledSolverManager._validate_solver_cfg(_FakeModel(), cfg)


def test_proxy_shape_overrides_apply_only_to_selected_body_shapes():
    model = _FakeModel()
    proxy = CoupledProxyCfg(
        source="rigid",
        destination="soft",
        bodies=[1],
        shape_material_ke=10.0,
        shape_material_kd=20.0,
        shape_material_mu=0.75,
        shape_margin=0.015,
        shape_gap=0.002,
    )

    NewtonCoupledSolverManager._apply_proxy_shape_overrides(model, [proxy])

    np.testing.assert_allclose(model.shape_material_ke.data, [1.0, 10.0, 1.0, 1.0])
    np.testing.assert_allclose(model.shape_material_kd.data, [2.0, 20.0, 2.0, 2.0])
    np.testing.assert_allclose(model.shape_material_mu.data, [3.0, 0.75, 3.0, 3.0])
    np.testing.assert_allclose(model.shape_margin.data, [4.0, 0.015, 4.0, 4.0])
    np.testing.assert_allclose(model.shape_gap.data, [5.0, 0.002, 5.0, 5.0])


@pytest.mark.parametrize(
    ("solver_cfg", "expected"),
    [
        (XPBDSolverCfg(), True),
        (MJWarpSolverCfg(use_mujoco_contacts=False), True),
        (MJWarpSolverCfg(use_mujoco_contacts=True), False),
    ],
)
def test_solver_cfg_external_contact_requirements(solver_cfg, expected):
    assert NewtonCoupledSolverManager._solver_cfg_needs_external_contacts(solver_cfg) is expected


def test_algorithm_defaults_route_outer_and_entry_local_collision_pipelines(monkeypatch):
    """Base infers its outer path, proxy defaults local, and ADMM defaults outer."""
    model = _FakeModel()
    proxy_cfg = _valid_proxy_cfg()
    recorded_entries: list[tuple[str, bool]] = []

    monkeypatch.setattr(
        NewtonCoupledSolverManager,
        "_resolve_solver_cfg",
        classmethod(lambda cls, model, cfg: cfg),
    )
    monkeypatch.setattr(
        NewtonCoupledSolverManager,
        "_build_entry",
        classmethod(
            lambda cls, entry_cfg, *, local_collision=False: (
                recorded_entries.append((entry_cfg.name, local_collision)) or entry_cfg.name
            )
        ),
    )
    monkeypatch.setattr(
        NewtonCoupledSolverManager,
        "_build_proxy_coupled_solver",
        classmethod(lambda cls, model, entries, cfg: SimpleNamespace(kind="proxy")),
    )
    monkeypatch.setattr(
        NewtonCoupledSolverManager,
        "_build_admm_coupled_solver",
        classmethod(lambda cls, model, entries, cfg: SimpleNamespace(kind="admm")),
    )
    monkeypatch.setattr(
        coupled_manager,
        "SolverCoupled",
        lambda *, model, entries: SimpleNamespace(kind="base", model=model, entries=entries),
    )
    monkeypatch.setattr(
        NewtonCoupledSolverManager,
        "_apply_proxy_shape_overrides",
        classmethod(lambda cls, model, proxies: None),
    )
    monkeypatch.setattr(
        NewtonCoupledSolverManager,
        "_apply_vbd_joint_constraint_modes",
        classmethod(lambda cls, entries: None),
    )
    monkeypatch.setattr(
        NewtonCoupledSolverManager,
        "_configure_fk_articulation_filter",
        classmethod(lambda cls, model, entries: None),
    )

    old_solver = coupled_manager.NewtonManager._solver
    old_outer = coupled_manager.NewtonManager._needs_collision_pipeline
    try:
        base_cfg = CoupledSolverCfg(entries=proxy_cfg.entries)
        NewtonCoupledSolverManager._build_solver(model, base_cfg)
        assert coupled_manager.NewtonManager._solver.kind == "base"
        assert coupled_manager.NewtonManager._solver.model is model
        assert coupled_manager.NewtonManager._solver.entries == ["rigid", "soft"]
        assert coupled_manager.NewtonManager._needs_collision_pipeline is True
        assert recorded_entries == [("rigid", False), ("soft", False)]

        recorded_entries.clear()
        internal_entries = [
            _entry("rigid", bodies=[0, 1], particles=[0]),
            _entry("soft", bodies=[2], particles=[1, 2]),
        ]
        for entry in internal_entries:
            entry.solver_cfg = MJWarpSolverCfg(use_mujoco_contacts=True)
        NewtonCoupledSolverManager._build_solver(model, CoupledSolverCfg(entries=internal_entries))
        assert coupled_manager.NewtonManager._solver.kind == "base"
        assert coupled_manager.NewtonManager._needs_collision_pipeline is False
        assert recorded_entries == [("rigid", False), ("soft", False)]

        recorded_entries.clear()
        NewtonCoupledSolverManager._build_solver(model, proxy_cfg)
        assert coupled_manager.NewtonManager._needs_collision_pipeline is False
        assert recorded_entries == [("rigid", True), ("soft", False)]

        recorded_entries.clear()
        admm_cfg = CoupledAdmmSolverCfg(entries=proxy_cfg.entries)
        NewtonCoupledSolverManager._build_solver(model, admm_cfg)
        assert coupled_manager.NewtonManager._needs_collision_pipeline is True
        assert recorded_entries == [("rigid", False), ("soft", False)]

        recorded_entries.clear()
        admm_cfg.use_collision_pipeline = False
        NewtonCoupledSolverManager._build_solver(model, admm_cfg)
        assert coupled_manager.NewtonManager._needs_collision_pipeline is False
        assert recorded_entries == [("rigid", True), ("soft", True)]
    finally:
        coupled_manager.NewtonManager._solver = old_solver
        coupled_manager.NewtonManager._needs_collision_pipeline = old_outer


class _RecordingCouplingHooks:
    """Record calls whose CouplingInterface defaults must not swallow behavior."""

    def __init__(self):
        self.calls: list[tuple[str, tuple, dict]] = []

    def _record(self, name, args, kwargs):
        self.calls.append((name, args, kwargs))
        return f"{name}-result"

    def coupling_eval_effective_mass_block(self, *args, **kwargs):
        return self._record("mass-block", args, kwargs)

    def coupling_eval_gravity_acceleration(self, *args, **kwargs):
        return self._record("gravity", args, kwargs)

    def coupling_notify_input_state_update(self, *args, **kwargs):
        return self._record("notify", args, kwargs)


def test_entry_collision_pipeline_wrapper_forwards_coupling_hooks():
    """Concrete CouplingInterface defaults never replace sub-solver hooks."""
    wrapped = coupled_manager._EntryCollisionPipelineSolver.__new__(coupled_manager._EntryCollisionPipelineSolver)
    solver = _RecordingCouplingHooks()
    wrapped._solver = solver

    assert wrapped.coupling_eval_effective_mass_block("state", bodies=[1]) == "mass-block-result"
    assert wrapped.coupling_eval_gravity_acceleration("model", 0.01) == "gravity-result"
    assert wrapped.coupling_notify_input_state_update("state") == "notify-result"
    assert solver.calls == [
        ("mass-block", ("state",), {"bodies": [1]}),
        ("gravity", ("model", 0.01), {}),
        ("notify", ("state",), {}),
    ]


def test_entry_collision_pipeline_wrapper_can_use_generic_effective_mass(monkeypatch):
    """An entry can deliberately retain its model-view mass and inertia."""
    wrapped = coupled_manager._EntryCollisionPipelineSolver.__new__(coupled_manager._EntryCollisionPipelineSolver)
    wrapped._solver = _RecordingCouplingHooks()
    wrapped._use_solver_effective_mass = False
    monkeypatch.setattr(
        CouplingInterface,
        "coupling_eval_effective_mass_block",
        lambda self, *args, **kwargs: "generic-mass-block",
    )

    assert wrapped.coupling_eval_effective_mass_block("state", bodies=[1]) == "generic-mass-block"


class _RecordingAdmm:
    """Capture ADMM construction while retaining Newton's config dataclasses."""

    ContactPair = SolverCoupledADMM.ContactPair
    Config = SolverCoupledADMM.Config

    def __init__(self, *, model, entries, coupling):
        self.model = model
        self.entries = entries
        self.coupling = coupling


def test_admm_build_forwards_multiple_pairs_matching_and_proximal_options(monkeypatch):
    cfg = CoupledAdmmSolverCfg(
        contact_pairs=[
            CoupledAdmmContactPairCfg(source="robot", destination="cable"),
            CoupledAdmmContactPairCfg(source="cable", destination="world"),
        ],
        iterations=7,
        joint_proximal_bodies=False,
        joint_proximal_destination_entries=["cable", "world"],
        joint_proximal_mass_scale=0.25,
        rigid_contact_matching="latest",
        contact_matching_pos_threshold=0.01,
        contact_matching_normal_dot_threshold=0.8,
        contact_matching_force_scale=0.7,
    )
    monkeypatch.setattr(coupled_manager, "SolverCoupledADMM", _RecordingAdmm)

    solver = NewtonCoupledSolverManager._build_admm_coupled_solver(object(), [], cfg)

    assert [(pair.source, pair.destination) for pair in solver.coupling.contact_pairs] == [
        ("robot", "cable"),
        ("cable", "world"),
    ]
    assert solver.coupling.iterations == 7
    assert solver.coupling.joint_proximal_bodies is False
    assert solver.coupling.joint_proximal_destination_entries == ["cable", "world"]
    assert solver.coupling.joint_proximal_mass_scale == pytest.approx(0.25)
    assert solver.coupling.rigid_contact_matching == "latest"
    assert solver.coupling.contact_matching_pos_threshold == pytest.approx(0.01)
    assert solver.coupling.contact_matching_normal_dot_threshold == pytest.approx(0.8)
    assert solver.coupling.contact_matching_force_scale == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("pair", "match"),
    [
        (CoupledAdmmContactPairCfg(source="missing", destination="b"), "must name coupled entries"),
        (CoupledAdmmContactPairCfg(source="a", destination="a"), "source and destination must differ"),
    ],
)
def test_admm_validation_rejects_invalid_contact_pairs(pair, match):
    cfg = CoupledAdmmSolverCfg(
        entries=[
            _entry("a", bodies=[0, 1], particles=[0]),
            _entry("b", bodies=[2], particles=[1, 2]),
        ],
        contact_pairs=[pair],
    )
    with pytest.raises(ValueError, match=match):
        NewtonCoupledSolverManager._validate_solver_cfg(_FakeModel(), cfg)


def test_fk_filter_masks_only_articulations_whose_every_joint_is_vbd_owned():
    """A mixed-solver articulation must remain eligible for generic FK."""
    model = SimpleNamespace(
        articulation_count=2,
        joint_articulation=wp.array([0, 0, 1], dtype=wp.int32, device="cpu"),
        device="cpu",
    )
    vbd = CoupledSolverEntryCfg(name="vbd", solver_cfg=VBDSolverCfg())
    vbd.joints = [0, 2]
    rigid = CoupledSolverEntryCfg(name="rigid", solver_cfg=XPBDSolverCfg())
    rigid.joints = [1]

    try:
        NewtonCoupledSolverManager._configure_fk_articulation_filter(model, [vbd, rigid])
        assert NewtonCoupledSolverManager._fk_articulation_filter.numpy().tolist() == [True, False]
    finally:
        NewtonCoupledSolverManager._fk_articulation_filter = None
        NewtonCoupledSolverManager._combined_fk_mask = None
