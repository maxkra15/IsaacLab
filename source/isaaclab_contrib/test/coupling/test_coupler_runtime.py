# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Kitless runtime tests for Newton's coupled-solver configurations."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import NewtonManager, VBDSolverCfg, XPBDSolverCfg
from newton import BodyFlags, CollisionPipeline, Model, ModelBuilder, StateFlags
from newton.solvers import SolverVBD, SolverXPBD
from newton.solvers.experimental.coupled import SolverCoupledADMM, SolverCoupledProxy

from isaaclab.physics import PhysicsManager

from isaaclab_contrib.coupling import (
    CouplerAdmmCfg,
    CouplerEntryCfg,
    CouplerProxyCfg,
    CouplerProxyMappingCfg,
    NewtonCouplerManager,
)


@pytest.fixture
def isolated_newton_manager(monkeypatch: pytest.MonkeyPatch):
    """Isolate every global manager slot touched by coupler construction."""
    clean_values = {
        "_model": None,
        "_solver": None,
        "_use_single_state": None,
        "_contacts": None,
        "_collision_pipeline": None,
        "_collision_cfg": None,
        "_needs_collision_pipeline": False,
        "_supports_contact_sensors": True,
        "_supports_rigid_body_force_input": False,
        "_report_contacts": False,
    }
    for name, value in clean_values.items():
        monkeypatch.setattr(NewtonManager, name, value)
    projection_values = {
        "_vbd_preserved_input_pose_projection_registrations": {},
        "_vbd_preserved_input_pose_projection_names": {},
        "_vbd_preserved_input_pose_projection_bindings": (),
        "_vbd_preserved_input_pose_projection_next_id": 0,
        "_vbd_preserved_input_pose_projection_issuer": object(),
    }
    for name, value in projection_values.items():
        monkeypatch.setattr(NewtonCouplerManager, name, value)
    yield


@wp.kernel
def _project_pose_parity_bodies(
    body_q: wp.array(dtype=wp.transform),
    anchor_body: int,
    spin_body: int,
) -> None:
    """Translate one authored anchor and swing one dynamic body."""
    anchor_q = body_q[anchor_body]
    body_q[anchor_body] = wp.transform(
        wp.transform_get_translation(anchor_q) + wp.vec3(0.01, 0.0, 0.0),
        wp.transform_get_rotation(anchor_q),
    )
    spin_q = body_q[spin_body]
    swing = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.04)
    body_q[spin_body] = wp.transform(
        wp.transform_get_translation(spin_q),
        swing * wp.transform_get_rotation(spin_q),
    )


def _build_pose_parity_model(device: str) -> tuple[Model, int, int]:
    """Build a dynamic-flagged zero-inertia anchor and one free body."""
    builder = ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body_ids = []
    for x, label in ((0.0, "/anchor"), (0.2, "/spin")):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3, dtype=np.float32)),
            label=label,
        )
        builder.add_shape_sphere(body, radius=0.01, label=f"{label}/shape")
        body_ids.append(body)
    anchor_body, spin_body = body_ids
    builder.body_mass[anchor_body] = 0.0
    builder.body_inv_mass[anchor_body] = 0.0
    builder.body_inertia[anchor_body] = wp.mat33(0.0)
    builder.body_inv_inertia[anchor_body] = wp.mat33(0.0)
    builder.color()
    model = builder.finalize(device=device)
    assert int(model.body_flags.numpy()[anchor_body]) & int(BodyFlags.DYNAMIC)
    assert model.body_mass.numpy()[anchor_body] == 0.0
    assert model.body_inv_mass.numpy()[anchor_body] == 0.0
    np.testing.assert_array_equal(model.body_inv_inertia.numpy()[anchor_body], 0.0)
    return model, anchor_body, spin_body


def _build_overlapping_body_model() -> Model:
    """Build two labeled free bodies with one rigid contact on the CPU."""
    builder = ModelBuilder(gravity=-9.81)
    for x, label in ((-0.09, "/World/Source/body"), (0.09, "/World/Destination/body")):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(x, 0.0, 1.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3, dtype=np.float32)),
            label=label,
        )
        builder.add_shape_sphere(body=body, radius=0.1, label=f"{label}/shape")
    builder.color()
    return builder.finalize(device="cpu")


def _entry_configs() -> list[CouplerEntryCfg]:
    """Return fresh entry configs so selector resolution cannot leak between cases."""
    return [
        CouplerEntryCfg(
            name="source",
            solver_cfg=XPBDSolverCfg(iterations=2),
            bodies=[r"/World/Source/body"],
        ),
        CouplerEntryCfg(
            name="destination",
            solver_cfg=XPBDSolverCfg(iterations=2),
            bodies=[r"/World/Destination/body"],
        ),
    ]


def _build_multi_world_pose_history_model(device: str) -> Model:
    """Build three worlds with task, unsaved table, and proxy-source bodies."""
    template = ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for x, label in (
        (0.0, "/World/VBD/task"),
        (0.25, "/World/VBD/table"),
        (0.5, "/World/Source/proxy"),
    ):
        body = template.add_body(
            xform=wp.transform(wp.vec3(x, 0.0, 1.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3, dtype=np.float32)),
            label=label,
        )
        template.add_shape_sphere(body, radius=0.04, label=f"{label}/shape")
    template.color()

    builder = ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for world in range(3):
        builder.add_world(
            template,
            xform=wp.transform(wp.vec3(float(world), 0.0, 0.0), wp.quat_identity()),
        )
    builder.color()
    return builder.finalize(device=device)


def _build_pose_history_coupler(
    model: Model, algorithm: str, *, iterations: int = 1
) -> tuple[SolverCoupledProxy | SolverCoupledADMM, SolverVBD]:
    """Build VBD task/table ownership plus a disjoint XPBD proxy source."""
    entries = [
        CouplerEntryCfg(name="vbd", solver_cfg=VBDSolverCfg(iterations=1), bodies=[0, 1, 3, 4, 6, 7]),
        CouplerEntryCfg(name="source", solver_cfg=XPBDSolverCfg(iterations=1), bodies=[2, 5, 8]),
    ]
    solver_cfg = (
        CouplerProxyCfg(
            entries=entries,
            iterations=iterations,
            proxies=[
                CouplerProxyMappingCfg(
                    source="source",
                    destination="vbd",
                    bodies=[2, 5, 8],
                    mode="staggered",
                )
            ],
        )
        if algorithm == "proxy"
        else CouplerAdmmCfg(entries=entries, iterations=iterations)
    )
    NewtonManager._model = model
    NewtonCouplerManager._build_solver(model, solver_cfg)
    coupled_solver = NewtonManager._solver
    assert isinstance(coupled_solver, (SolverCoupledProxy, SolverCoupledADMM))
    vbd_solver = coupled_solver.solver("vbd")
    assert isinstance(vbd_solver, SolverVBD)
    return coupled_solver, vbd_solver


def _pose_history_values(worlds: np.ndarray, offset: float) -> np.ndarray:
    """Return distinct valid transforms indexed by entry-local body world."""
    values = np.zeros((len(worlds), 7), dtype=np.float32)
    values[:, 0] = offset + worlds
    values[:, 1] = 0.01 * worlds
    values[:, 6] = 1.0
    return values


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", id="cpu"),
        pytest.param(
            "cuda:0",
            id="cuda",
            marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA is unavailable"),
        ),
    ],
)
def test_vbd_preserved_input_pose_matches_direct_vbd_across_steps(
    device: str,
    isolated_newton_manager,
):
    """Authored anchors and swings retain direct standalone VBD semantics."""
    model, anchor_body, spin_body = _build_pose_parity_model(device)
    NewtonManager._model = model
    body_ids = wp.array([anchor_body, spin_body], dtype=wp.int32, device=device)

    def project(state):
        wp.launch(
            _project_pose_parity_bodies,
            dim=1,
            inputs=[state.body_q, anchor_body, spin_body],
            device=device,
        )

    handle = NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
        name="pose-parity",
        entry_name="vbd",
        body_ids=body_ids,
        callback=project,
    )
    assert NewtonManager._solver is None
    NewtonCouplerManager._build_solver(
        model,
        CouplerAdmmCfg(
            entries=[
                CouplerEntryCfg(
                    name="vbd",
                    solver_cfg=VBDSolverCfg(iterations=1),
                    bodies=[anchor_body, spin_body],
                )
            ],
            contact_pairs=[],
            iterations=1,
        ),
    )
    coupled_solver = NewtonManager._solver
    assert isinstance(coupled_solver, SolverCoupledADMM)
    nested_solver = coupled_solver.solver("vbd")
    assert isinstance(nested_solver, SolverVBD)
    direct_solver = SolverVBD(model, iterations=1)
    dt = 1.0 / 120.0

    direct_0, direct_1 = model.state(), model.state()
    coupled_0, coupled_1 = model.state(), model.state()
    direct_solver.step(direct_0, direct_1, model.control(), None, dt)
    project(direct_1)
    NewtonCouplerManager._step_solver(coupled_0, coupled_1, model.control(), None, dt)
    wp.synchronize()

    tolerance = 0.0 if device == "cpu" else 2.0e-7
    np.testing.assert_allclose(coupled_1.body_q.numpy(), direct_1.body_q.numpy(), atol=tolerance, rtol=0.0)
    np.testing.assert_allclose(
        nested_solver.body_q_prev.numpy(), direct_solver.body_q_prev.numpy(), atol=tolerance, rtol=0.0
    )
    np.testing.assert_allclose(
        nested_solver._coupling_body_q_prev_snapshot.numpy(),
        direct_solver.body_q_prev.numpy(),
        atol=tolerance,
        rtol=0.0,
    )
    assert not np.array_equal(direct_1.body_q.numpy(), direct_solver.body_q_prev.numpy())
    entry_output_q = coupled_solver.entry_state("vbd", phase="output").body_q.numpy()
    np.testing.assert_allclose(entry_output_q, coupled_1.body_q.numpy(), atol=tolerance, rtol=0.0)

    authored_qd = np.array(
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [-0.1, -0.2, -0.3, -0.4, -0.5, 0.3]],
        dtype=np.float32,
    )
    direct_1.body_qd.assign(authored_qd)
    coupled_1.body_qd.assign(authored_qd)
    direct_2, coupled_2 = model.state(), model.state()
    direct_solver.step(direct_1, direct_2, model.control(), None, dt)
    project(direct_2)
    NewtonCouplerManager._step_solver(coupled_1, coupled_2, model.control(), None, dt)
    wp.synchronize()

    np.testing.assert_allclose(coupled_2.body_q.numpy(), direct_2.body_q.numpy(), atol=tolerance, rtol=0.0)
    np.testing.assert_allclose(coupled_2.body_qd.numpy(), direct_2.body_qd.numpy(), atol=tolerance, rtol=0.0)
    np.testing.assert_allclose(
        nested_solver.body_q_prev.numpy(), direct_solver.body_q_prev.numpy(), atol=tolerance, rtol=0.0
    )
    np.testing.assert_allclose(
        nested_solver._coupling_body_q_prev_snapshot.numpy(),
        direct_solver._coupling_body_q_prev_snapshot.numpy(),
        atol=tolerance,
        rtol=0.0,
    )
    assert direct_2.body_q.numpy()[anchor_body, 0] == pytest.approx(0.02, abs=2.0e-7)
    assert direct_2.body_qd.numpy()[anchor_body, 0] == pytest.approx(1.2, abs=2.0e-6)
    assert direct_2.body_qd.numpy()[spin_body, 5] == pytest.approx(5.1, abs=2.0e-5)
    np.testing.assert_allclose(
        coupled_solver.entry_state("vbd", phase="output").body_q.numpy(),
        coupled_2.body_q.numpy(),
        atol=tolerance,
        rtol=0.0,
    )
    assert handle.deregister() is True
    assert handle.deregister() is False


def test_vbd_preserved_input_pose_registration_validation_and_overlap(
    isolated_newton_manager,
):
    """Deferred registration rejects malformed and multiply-owned selections."""
    model, anchor_body, spin_body = _build_pose_parity_model("cpu")
    NewtonManager._model = model
    valid_ids = wp.array([anchor_body], dtype=wp.int32, device="cpu")

    with pytest.raises(TypeError, match="name must be a string"):
        NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
            name=None, entry_name="vbd", body_ids=valid_ids, callback=lambda state: None
        )
    with pytest.raises(ValueError, match="name must be a non-empty"):
        NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
            name="", entry_name="vbd", body_ids=valid_ids, callback=lambda state: None
        )
    with pytest.raises(TypeError, match="entry_name must be a string"):
        NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
            name="pose", entry_name=None, body_ids=valid_ids, callback=lambda state: None
        )
    with pytest.raises(TypeError, match="callback must be callable"):
        NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
            name="pose", entry_name="vbd", body_ids=valid_ids, callback=None
        )
    with pytest.raises(TypeError, match="body_ids must be a Warp array"):
        NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
            name="pose", entry_name="vbd", body_ids=[anchor_body], callback=lambda state: None
        )
    with pytest.raises(TypeError, match="one-dimensional wp.int32"):
        NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
            name="pose",
            entry_name="vbd",
            body_ids=wp.array([float(anchor_body)], dtype=wp.float32, device="cpu"),
            callback=lambda state: None,
        )
    with pytest.raises(TypeError, match="one-dimensional wp.int32"):
        NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
            name="pose",
            entry_name="vbd",
            body_ids=wp.array([[anchor_body]], dtype=wp.int32, device="cpu"),
            callback=lambda state: None,
        )
    with pytest.raises(ValueError, match="select at least one"):
        NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
            name="pose",
            entry_name="vbd",
            body_ids=wp.empty(0, dtype=wp.int32, device="cpu"),
            callback=lambda state: None,
        )
    with pytest.raises(ValueError, match="must not contain duplicates"):
        NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
            name="pose",
            entry_name="vbd",
            body_ids=wp.array([anchor_body, anchor_body], dtype=wp.int32, device="cpu"),
            callback=lambda state: None,
        )
    with pytest.raises(ValueError, match="must lie"):
        NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
            name="pose",
            entry_name="vbd",
            body_ids=wp.array([model.body_count], dtype=wp.int32, device="cpu"),
            callback=lambda state: None,
        )

    handle = NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
        name="pose",
        entry_name="vbd",
        body_ids=valid_ids,
        callback=lambda state: None,
    )
    with pytest.raises(ValueError, match="already registered"):
        NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
            name="pose",
            entry_name="vbd",
            body_ids=wp.array([spin_body], dtype=wp.int32, device="cpu"),
            callback=lambda state: None,
        )
    with pytest.raises(ValueError, match="overlaps projection"):
        NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
            name="other",
            entry_name="vbd",
            body_ids=valid_ids,
            callback=lambda state: None,
        )
    assert handle.deregister() is True
    assert handle.deregister() is False


def test_vbd_preserved_input_pose_lifecycle_is_deferred_and_clear_safe(
    isolated_newton_manager,
):
    """Pre-build deregistration is effective and clear invalidates old handles."""
    model, anchor_body, spin_body = _build_pose_parity_model("cpu")
    NewtonManager._model = model
    handle = NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
        name="cancelled",
        entry_name="vbd",
        body_ids=wp.array([anchor_body], dtype=wp.int32, device="cpu"),
        callback=lambda state: None,
    )
    assert handle.deregister() is True
    NewtonCouplerManager._build_solver(
        model,
        CouplerAdmmCfg(
            entries=[
                CouplerEntryCfg(
                    name="vbd",
                    solver_cfg=VBDSolverCfg(iterations=1),
                    bodies=[anchor_body, spin_body],
                )
            ],
            contact_pairs=[],
            iterations=1,
        ),
    )
    assert NewtonCouplerManager._vbd_preserved_input_pose_projection_bindings == ()
    nested_solver = NewtonManager._solver.solver("vbd")
    assert not hasattr(nested_solver, "_isaaclab_vbd_preserved_input_pose")
    np.testing.assert_array_equal(
        nested_solver._isaaclab_vbd_pose_history_restore.preserve_input_mask.numpy(), [False, False]
    )
    with pytest.raises(RuntimeError, match="before coupled solver initialization"):
        NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
            name="late",
            entry_name="vbd",
            body_ids=wp.array([anchor_body], dtype=wp.int32, device="cpu"),
            callback=lambda state: None,
        )

    NewtonManager._solver = None
    cleared_handle = NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
        name="clear-me",
        entry_name="vbd",
        body_ids=wp.array([anchor_body], dtype=wp.int32, device="cpu"),
        callback=lambda state: None,
    )
    NewtonCouplerManager._solver_specific_clear()
    assert cleared_handle.deregister() is False
    assert NewtonCouplerManager._vbd_preserved_input_pose_projection_bindings == ()


def test_vbd_preserved_input_pose_rejects_stale_model_registration(
    isolated_newton_manager,
):
    """A hard rebuild cannot reinterpret body ids registered for an old model."""
    old_model, old_anchor, _ = _build_pose_parity_model("cpu")
    NewtonManager._model = old_model
    NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
        name="stale",
        entry_name="vbd",
        body_ids=wp.array([old_anchor], dtype=wp.int32, device="cpu"),
        callback=lambda state: None,
    )
    new_model, new_anchor, new_spin = _build_pose_parity_model("cpu")
    NewtonManager._model = new_model
    with pytest.raises(RuntimeError, match="stale parent model"):
        NewtonCouplerManager._build_solver(
            new_model,
            CouplerAdmmCfg(
                entries=[
                    CouplerEntryCfg(
                        name="vbd",
                        solver_cfg=VBDSolverCfg(iterations=1),
                        bodies=[new_anchor, new_spin],
                    )
                ],
                contact_pairs=[],
                iterations=1,
            ),
        )


@pytest.mark.parametrize(
    ("scenario", "error_type", "match"),
    [
        pytest.param("missing", KeyError, "Unknown coupled solver entry", id="missing-entry"),
        pytest.param("non-vbd", RuntimeError, "not SolverVBD", id="non-vbd-entry"),
        pytest.param("unowned", ValueError, "not owned", id="unowned-body"),
    ],
)
def test_vbd_preserved_input_pose_binding_validates_entry_ownership(
    scenario: str,
    error_type: type[Exception],
    match: str,
    isolated_newton_manager,
):
    """Deferred resolution fails closed on missing, incompatible, or unowned rows."""
    model, anchor_body, spin_body = _build_pose_parity_model("cpu")
    NewtonManager._model = model
    entry_name = "missing" if scenario == "missing" else "vbd"
    selected_body = spin_body if scenario == "unowned" else anchor_body
    NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
        name="invalid-binding",
        entry_name=entry_name,
        body_ids=wp.array([selected_body], dtype=wp.int32, device="cpu"),
        callback=lambda state: None,
    )
    nested_cfg = XPBDSolverCfg(iterations=1) if scenario == "non-vbd" else VBDSolverCfg(iterations=1)
    owned_bodies = [anchor_body] if scenario == "unowned" else [anchor_body, spin_body]
    with pytest.raises(error_type, match=match):
        NewtonCouplerManager._build_solver(
            model,
            CouplerAdmmCfg(
                entries=[CouplerEntryCfg(name="vbd", solver_cfg=nested_cfg, bodies=owned_bodies)],
                contact_pairs=[],
                iterations=1,
            ),
        )


def test_vbd_preserved_input_pose_callback_exception_propagates(
    isolated_newton_manager,
):
    """Projection failures surface at the solver boundary without being hidden."""
    model, anchor_body, spin_body = _build_pose_parity_model("cpu")
    NewtonManager._model = model

    def fail_projection(state):
        del state
        raise RuntimeError("projection failed")

    NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
        name="failing",
        entry_name="vbd",
        body_ids=wp.array([anchor_body], dtype=wp.int32, device="cpu"),
        callback=fail_projection,
    )
    NewtonCouplerManager._build_solver(
        model,
        CouplerAdmmCfg(
            entries=[
                CouplerEntryCfg(
                    name="vbd",
                    solver_cfg=VBDSolverCfg(iterations=1),
                    bodies=[anchor_body, spin_body],
                )
            ],
            contact_pairs=[],
            iterations=1,
        ),
    )
    with pytest.raises(RuntimeError, match="projection failed"):
        NewtonCouplerManager._step_solver(model.state(), model.state(), model.control(), None, 1.0 / 120.0)


def test_preserved_input_pose_precedes_swap_copy_and_midloop_collision(
    isolated_newton_manager,
    monkeypatch: pytest.MonkeyPatch,
):
    """Every consumer in the manager loop observes the post-solver projection."""
    model, anchor_body, spin_body = _build_pose_parity_model("cpu")
    NewtonManager._model = model

    def project(state):
        wp.launch(
            _project_pose_parity_bodies,
            dim=1,
            inputs=[state.body_q, anchor_body, spin_body],
            device="cpu",
        )

    NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
        name="loop-order",
        entry_name="vbd",
        body_ids=wp.array([anchor_body, spin_body], dtype=wp.int32, device="cpu"),
        callback=project,
    )
    NewtonCouplerManager._build_solver(
        model,
        CouplerAdmmCfg(
            entries=[
                CouplerEntryCfg(
                    name="vbd",
                    solver_cfg=VBDSolverCfg(iterations=1),
                    bodies=[anchor_body, spin_body],
                )
            ],
            contact_pairs=[],
            iterations=1,
        ),
    )
    NewtonCouplerManager._initialize_contacts()
    contacts = NewtonManager._contacts
    assert contacts is not None
    collision_observations: list[float] = []

    class RecordingCollisionPipeline:
        def collide(self, state, received_contacts):
            assert received_contacts is contacts
            collision_observations.append(float(state.body_q.numpy()[anchor_body, 0]))

    monkeypatch.setattr(PhysicsManager, "_cfg", SimpleNamespace())
    monkeypatch.setattr(NewtonManager, "_state_0", model.state())
    monkeypatch.setattr(NewtonManager, "_state_1", model.state())
    monkeypatch.setattr(NewtonManager, "_control", model.control())
    monkeypatch.setattr(NewtonManager, "_num_substeps", 3)
    monkeypatch.setattr(NewtonManager, "_collision_decimation", 1)
    monkeypatch.setattr(NewtonManager, "_solver_dt", 1.0 / 120.0)
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", True)
    monkeypatch.setattr(NewtonManager, "_state_force_callbacks", [])
    monkeypatch.setattr(NewtonManager, "_collision_pipeline", RecordingCollisionPipeline())

    NewtonCouplerManager._run_solver_substeps(contacts)
    wp.synchronize()

    np.testing.assert_allclose(collision_observations, [0.01, 0.02], atol=2.0e-7, rtol=0.0)
    assert NewtonManager._state_0.body_q.numpy()[anchor_body, 0] == pytest.approx(0.03, abs=2.0e-7)
    local_anchor = NewtonManager._solver._entries["vbd"].body_global_to_local.numpy()[anchor_body]
    assert NewtonManager._solver.entry_state("vbd", phase="output").body_q.numpy()[local_anchor, 0] == pytest.approx(
        0.03, abs=2.0e-7
    )


@pytest.mark.parametrize("algorithm", ["proxy", "admm"])
@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", id="cpu"),
        pytest.param(
            "cuda:0",
            id="cuda",
            marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA is unavailable"),
        ),
    ],
)
def test_no_registration_keeps_upstream_vbd_input_hook_bitwise_identical(
    algorithm: str,
    device: str,
    isolated_newton_manager,
):
    """The default path installs no wrapper and matches upstream array bits."""
    model = _build_multi_world_pose_history_model(device)
    coupled_solver, nested_solver = _build_pose_history_coupler(model, algorithm)
    assert NewtonCouplerManager._vbd_preserved_input_pose_projection_bindings == ()
    assert not hasattr(nested_solver, "_isaaclab_vbd_preserved_input_pose")
    pending = nested_solver._isaaclab_vbd_pose_history_restore
    assert pending.preserve_input_active is False
    np.testing.assert_array_equal(
        pending.preserve_input_mask.numpy(), np.zeros(nested_solver.model.body_count, dtype=bool)
    )
    assert nested_solver.coupling_notify_input_state_update.__func__ is SolverVBD.coupling_notify_input_state_update

    reference_solver = SolverVBD(nested_solver.model, iterations=1)
    body_count = nested_solver.model.body_count
    body_q_prev = _pose_history_values(nested_solver.model.body_world.numpy(), 2.0)
    coupling_q_prev = _pose_history_values(nested_solver.model.body_world.numpy(), 4.0)
    rebaseline_mask = np.zeros(nested_solver.model.world_count + 1, dtype=bool)
    for solver in (nested_solver, reference_solver):
        solver.body_q_prev.assign(body_q_prev)
        solver._coupling_body_q_prev_snapshot.assign(coupling_q_prev)
        solver._rigid_pose_rebaseline_mask.assign(rebaseline_mask)

    nested_state = nested_solver.model.state()
    reference_state = nested_solver.model.state()
    input_q = nested_state.body_q.numpy()
    input_q[:, 0] += np.arange(body_count, dtype=np.float32) * 0.01 + 0.125
    input_qd = np.arange(body_count * 6, dtype=np.float32).reshape(body_count, 6) * 0.03125
    nested_state.body_q.assign(input_q)
    reference_state.body_q.assign(input_q)
    nested_state.body_qd.assign(input_qd)
    reference_state.body_qd.assign(input_qd)

    nested_solver.coupling_notify_input_state_update(
        nested_state, StateFlags.BODY_Q | StateFlags.BODY_QD, dt=1.0 / 120.0
    )
    reference_solver.coupling_notify_input_state_update(
        reference_state, StateFlags.BODY_Q | StateFlags.BODY_QD, dt=1.0 / 120.0
    )
    wp.synchronize()

    np.testing.assert_array_equal(nested_state.body_q.numpy(), reference_state.body_q.numpy())
    np.testing.assert_array_equal(nested_state.body_qd.numpy(), reference_state.body_qd.numpy())
    np.testing.assert_array_equal(nested_solver.body_q_prev.numpy(), reference_solver.body_q_prev.numpy())
    np.testing.assert_array_equal(
        nested_solver._coupling_body_q_prev_snapshot.numpy(),
        reference_solver._coupling_body_q_prev_snapshot.numpy(),
    )


@pytest.mark.parametrize("algorithm", ["proxy", "admm"])
@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", id="cpu"),
        pytest.param(
            "cuda:0",
            id="cuda",
            marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA is unavailable"),
        ),
    ],
)
def test_vbd_pose_history_capture_restore_preserves_order_and_selected_worlds(
    algorithm: str,
    device: str,
    isolated_newton_manager,
):
    """Deferred restore survives final distribution and applies only once."""
    model = _build_multi_world_pose_history_model(device)
    coupled_solver, vbd_solver = _build_pose_history_coupler(model, algorithm)
    np.testing.assert_array_equal(model.body_world.numpy(), [0, 0, 0, 1, 1, 1, 2, 2, 2])

    local_worlds = vbd_solver.model.body_world.numpy()
    previous_values = _pose_history_values(local_worlds, 10.0)
    coupling_values = _pose_history_values(local_worlds, 20.0)
    vbd_solver.body_q_prev.assign(previous_values)
    vbd_solver._coupling_body_q_prev_snapshot.assign(coupling_values)
    vbd_solver._rigid_pose_rebaseline_mask.fill_(True)

    body_ids = wp.array([6, 0], dtype=wp.int32, device=device)
    world_ids = wp.array([2, 0], dtype=wp.int32, device=device)
    local_body_ids = coupled_solver._entries["vbd"].body_global_to_local.numpy()[[6, 0]]
    captured_previous, captured_coupling = NewtonCouplerManager.capture_vbd_pose_history("vbd", body_ids, world_ids)

    np.testing.assert_allclose(captured_previous.numpy(), previous_values[local_body_ids])
    np.testing.assert_allclose(captured_coupling.numpy(), coupling_values[local_body_ids])
    np.testing.assert_array_equal(vbd_solver._rigid_pose_rebaseline_mask.numpy(), [True, True, True, True])

    destination_previous = _pose_history_values(local_worlds, 30.0)
    destination_coupling = _pose_history_values(local_worlds, 40.0)
    vbd_solver.body_q_prev.assign(destination_previous)
    vbd_solver._coupling_body_q_prev_snapshot.assign(destination_coupling)
    queued = NewtonCouplerManager.queue_vbd_pose_history_restore(
        "vbd",
        body_ids,
        world_ids,
        captured_previous,
        captured_coupling,
    )

    assert queued.body_ids == (6, 0)
    assert queued.world_ids == (2, 0)
    assert queued.expected_body_counts == (1, 1)
    assert queued.pending_world_ids == (2, 0)
    assert queued.applied_world_ids == ()
    assert queued.failed_world_ids == ()
    assert queued.superseded_world_ids == ()
    assert queued.pending is True
    assert queued.applied_exactly_once is False
    assert queued.body_application_count_deltas == (0, 0)
    np.testing.assert_allclose(vbd_solver.body_q_prev.numpy(), destination_previous)
    np.testing.assert_allclose(vbd_solver._coupling_body_q_prev_snapshot.numpy(), destination_coupling)
    np.testing.assert_array_equal(vbd_solver._rigid_pose_rebaseline_mask.numpy(), [True, True, True, True])

    state_0 = model.state()
    state_1 = model.state()
    state_0_body_q = state_0.body_q.numpy()
    state_0_body_q[[2, 5, 8], 0] += np.array([0.7, 0.8, 0.9], dtype=np.float32)
    state_0.body_q.assign(state_0_body_q)
    coupled_solver.step(state_0, state_1, model.control(), None, 1.0 / 120.0)
    status = NewtonCouplerManager.get_vbd_pose_history_restore_status(queued)

    assert status.pending is False
    assert status.applied_world_ids == (2, 0)
    assert status.failed_world_ids == ()
    assert status.application_count_deltas == (1, 1)
    assert status.body_application_count_deltas == (1, 1)
    assert status.applied_exactly_once is True
    np.testing.assert_allclose(
        vbd_solver._coupling_body_q_prev_snapshot.numpy()[local_body_ids],
        captured_coupling.numpy(),
    )
    global_to_local = coupled_solver._entries["vbd"].body_global_to_local.numpy()
    unsaved_owned_body_ids = np.array([1, 3, 4, 7])
    np.testing.assert_allclose(
        vbd_solver._coupling_body_q_prev_snapshot.numpy()[global_to_local[unsaved_owned_body_ids]],
        state_0_body_q[unsaved_owned_body_ids],
    )
    if algorithm == "proxy":
        proxy_body_ids = np.array([2, 5, 8])
        proxy_local_body_ids = global_to_local[proxy_body_ids]
        np.testing.assert_allclose(
            vbd_solver._coupling_body_q_prev_snapshot.numpy()[proxy_local_body_ids],
            state_0_body_q[proxy_body_ids],
        )
        np.testing.assert_allclose(
            coupled_solver._entries["vbd"].state_0.body_qd.numpy()[proxy_local_body_ids],
            0.0,
            atol=1.0e-6,
        )

    coupled_solver.step(state_1, state_0, model.control(), None, 1.0 / 120.0)
    second_status = NewtonCouplerManager.get_vbd_pose_history_restore_status(status)
    assert second_status.applied_exactly_once is True
    assert second_status.application_count_deltas == (1, 1)
    assert second_status.body_application_count_deltas == (1, 1)


@pytest.mark.parametrize("algorithm", ["proxy", "admm"])
@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", id="cpu"),
        pytest.param(
            "cuda:0",
            id="cuda",
            marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA is unavailable"),
        ),
    ],
)
def test_deferred_vbd_restore_runs_after_pass_zero_and_drives_iteration_restart(
    algorithm: str,
    device: str,
    isolated_newton_manager,
    monkeypatch: pytest.MonkeyPatch,
):
    """The hook replays continuous input while preserving final unsaved baselines."""
    step_observations: list[dict[str, np.ndarray]] = []
    original_step = SolverVBD.step

    def observe_step(self, state_in, state_out, control, contacts, dt):
        step_observations.append(
            {
                "body_q": state_in.body_q.numpy().copy(),
                "body_qd": state_in.body_qd.numpy().copy(),
                "body_q_prev": self.body_q_prev.numpy().copy(),
                "coupling_body_q_prev": self._coupling_body_q_prev_snapshot.numpy().copy(),
                "rebaseline_mask": self._rigid_pose_rebaseline_mask.numpy().copy(),
            }
        )
        return original_step(self, state_in, state_out, control, contacts, dt)

    monkeypatch.setattr(SolverVBD, "step", observe_step)
    model = _build_multi_world_pose_history_model(device)
    coupled_solver, vbd_solver = _build_pose_history_coupler(model, algorithm, iterations=2)
    local_worlds = vbd_solver.model.body_world.numpy()
    captured_previous_values = _pose_history_values(local_worlds, 10.0)
    captured_coupling_values = _pose_history_values(local_worlds, 20.0)
    vbd_solver.body_q_prev.assign(captured_previous_values)
    vbd_solver._coupling_body_q_prev_snapshot.assign(captured_coupling_values)
    vbd_solver._rigid_pose_rebaseline_mask.fill_(True)
    body_ids = wp.array([6, 0], dtype=wp.int32, device=device)
    world_ids = wp.array([2, 0], dtype=wp.int32, device=device)
    local_body_ids = coupled_solver._entries["vbd"].body_global_to_local.numpy()[[6, 0]]
    captured_previous, captured_coupling = NewtonCouplerManager.capture_vbd_pose_history("vbd", body_ids, world_ids)

    vbd_solver.body_q_prev.assign(_pose_history_values(local_worlds, 30.0))
    vbd_solver._coupling_body_q_prev_snapshot.assign(_pose_history_values(local_worlds, 40.0))
    queued = NewtonCouplerManager.restore_vbd_pose_history(
        "vbd", body_ids, world_ids, captured_previous, captured_coupling
    )
    dt = 1.0 / 120.0
    state_0 = model.state()
    state_0_body_q = state_0.body_q.numpy()
    selected_body_ids = np.array([6, 0])
    selected_angles = np.array([0.02, -0.03], dtype=np.float32)
    state_0_body_q[selected_body_ids, 3:7] = np.stack(
        [
            np.zeros_like(selected_angles),
            np.zeros_like(selected_angles),
            np.sin(0.5 * selected_angles),
            np.cos(0.5 * selected_angles),
        ],
        axis=1,
    )
    state_0_body_q[[2, 5, 8], 0] += np.array([0.7, 0.8, 0.9], dtype=np.float32)
    state_0.body_q.assign(state_0_body_q)
    state_0_body_qd = state_0.body_qd.numpy()
    state_0_body_qd[selected_body_ids] = np.array(
        [[0.5, -0.25, 0.125, 0.1, -0.2, 0.3], [-0.5, 0.25, -0.125, -0.1, 0.2, -0.3]],
        dtype=np.float32,
    )
    state_0_body_qd[[2, 5, 8], 0] = np.array([0.12, 0.24, 0.36], dtype=np.float32)
    state_0.body_qd.assign(state_0_body_qd)
    coupled_solver.step(state_0, model.state(), model.control(), None, dt)

    assert len(step_observations) == 2
    first_solve, restarted_solve = step_observations
    np.testing.assert_array_equal(first_solve["rebaseline_mask"], [False, True, False, True])
    np.testing.assert_allclose(first_solve["body_q_prev"][local_body_ids], captured_previous.numpy())
    np.testing.assert_allclose(
        first_solve["coupling_body_q_prev"][local_body_ids],
        captured_coupling.numpy(),
    )
    expected_first_qd = state_0_body_qd[selected_body_ids].copy()
    expected_first_qd[:, :3] += (state_0_body_q[selected_body_ids, :3] - captured_previous.numpy()[:, :3]) / dt
    expected_first_qd[:, 5] += selected_angles / dt
    np.testing.assert_allclose(first_solve["body_q"][local_body_ids], captured_previous.numpy())
    np.testing.assert_allclose(first_solve["body_qd"][local_body_ids], expected_first_qd, rtol=1.0e-5)
    np.testing.assert_allclose(restarted_solve["body_q"][local_body_ids], captured_coupling.numpy())

    global_to_local = coupled_solver._entries["vbd"].body_global_to_local.numpy()
    unsaved_owned_body_ids = np.array([1, 3, 4, 7])
    unsaved_owned_local_ids = global_to_local[unsaved_owned_body_ids]
    np.testing.assert_allclose(
        first_solve["body_q_prev"][unsaved_owned_local_ids],
        first_solve["body_q"][unsaved_owned_local_ids],
    )
    np.testing.assert_allclose(
        first_solve["coupling_body_q_prev"][unsaved_owned_local_ids],
        first_solve["body_q"][unsaved_owned_local_ids],
    )
    if algorithm == "proxy":
        proxy_body_ids = np.array([2, 5, 8])
        proxy_local_body_ids = global_to_local[proxy_body_ids]
        np.testing.assert_allclose(
            first_solve["body_q_prev"][proxy_local_body_ids],
            first_solve["body_q"][proxy_local_body_ids],
        )
        np.testing.assert_allclose(
            first_solve["coupling_body_q_prev"][proxy_local_body_ids],
            first_solve["body_q"][proxy_local_body_ids],
        )
        assert not np.allclose(
            first_solve["body_q"][proxy_local_body_ids, :3],
            state_0_body_q[proxy_body_ids, :3],
        )
        np.testing.assert_allclose(
            first_solve["body_qd"][proxy_local_body_ids],
            state_0_body_qd[proxy_body_ids],
            atol=1.0e-6,
        )
    assert NewtonCouplerManager.get_vbd_pose_history_restore_status(queued).applied_exactly_once is True


@pytest.mark.parametrize("algorithm", ["proxy", "admm"])
@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", id="cpu"),
        pytest.param(
            "cuda:0",
            id="cuda",
            marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA is unavailable"),
        ),
    ],
)
def test_preserved_input_pose_survives_queued_restore_and_iteration_restart(
    algorithm: str,
    device: str,
    isolated_newton_manager,
    monkeypatch: pytest.MonkeyPatch,
):
    """Selected q/qd stay authored while raw histories scatter and unsaved rows replay."""
    observations: list[dict[str, np.ndarray]] = []
    original_step = SolverVBD.step

    def observe_step(self, state_in, state_out, control, contacts, dt):
        observations.append(
            {
                "body_q": state_in.body_q.numpy().copy(),
                "body_qd": state_in.body_qd.numpy().copy(),
                "body_q_prev": self.body_q_prev.numpy().copy(),
                "coupling_body_q_prev": self._coupling_body_q_prev_snapshot.numpy().copy(),
            }
        )
        return original_step(self, state_in, state_out, control, contacts, dt)

    monkeypatch.setattr(SolverVBD, "step", observe_step)
    model = _build_multi_world_pose_history_model(device)
    NewtonManager._model = model
    selected_global_body_ids = np.array([6, 0], dtype=np.int32)
    selected_body_ids = wp.array(selected_global_body_ids, dtype=wp.int32, device=device)
    NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
        name="restore-parity",
        entry_name="vbd",
        body_ids=selected_body_ids,
        callback=lambda state: None,
    )
    coupled_solver, vbd_solver = _build_pose_history_coupler(model, algorithm, iterations=2)
    global_to_local = coupled_solver._entries["vbd"].body_global_to_local.numpy()
    selected_local_body_ids = global_to_local[selected_global_body_ids]
    initial_history = vbd_solver.model.body_q.numpy()
    vbd_solver.body_q_prev.assign(initial_history)
    vbd_solver._coupling_body_q_prev_snapshot.assign(initial_history)
    vbd_solver._rigid_pose_rebaseline_mask.assign(np.array([True, False, True, False]))

    restored_previous = wp.array(
        _pose_history_values(np.array([2, 0]), 10.0),
        dtype=wp.transform,
        device=device,
    )
    restored_coupling = wp.array(
        _pose_history_values(np.array([2, 0]), 20.0),
        dtype=wp.transform,
        device=device,
    )
    world_ids = wp.array([2, 0], dtype=wp.int32, device=device)
    queued = NewtonCouplerManager.restore_vbd_pose_history(
        "vbd",
        selected_body_ids,
        world_ids,
        restored_previous,
        restored_coupling,
    )

    state_in = model.state()
    body_q = state_in.body_q.numpy()
    body_q[selected_global_body_ids, 0] += np.array([0.02, -0.03], dtype=np.float32)
    angles = np.array([0.04, -0.05], dtype=np.float32)
    body_q[selected_global_body_ids, 3:7] = np.stack(
        [
            np.zeros_like(angles),
            np.zeros_like(angles),
            np.sin(0.5 * angles),
            np.cos(0.5 * angles),
        ],
        axis=1,
    )
    unsaved_global_body = 3
    unsaved_local_body = int(global_to_local[unsaved_global_body])
    body_q[unsaved_global_body, 0] += 0.125
    state_in.body_q.assign(body_q)
    body_qd = state_in.body_qd.numpy()
    selected_qd = np.array(
        [[0.5, -0.25, 0.125, 0.1, -0.2, 0.3], [-0.5, 0.25, -0.125, -0.1, 0.2, -0.3]],
        dtype=np.float32,
    )
    body_qd[selected_global_body_ids] = selected_qd
    body_qd[unsaved_global_body, 0] = 0.75
    state_in.body_qd.assign(body_qd)

    coupled_solver.step(state_in, model.state(), model.control(), None, 1.0 / 120.0)
    wp.synchronize()

    assert len(observations) == 2
    first_solve, restarted_solve = observations
    np.testing.assert_allclose(first_solve["body_q"][selected_local_body_ids], body_q[selected_global_body_ids])
    np.testing.assert_array_equal(first_solve["body_qd"][selected_local_body_ids], selected_qd)
    np.testing.assert_allclose(first_solve["body_q_prev"][selected_local_body_ids], restored_previous.numpy())
    np.testing.assert_allclose(first_solve["coupling_body_q_prev"][selected_local_body_ids], restored_coupling.numpy())
    np.testing.assert_allclose(restarted_solve["body_q"][selected_local_body_ids], body_q[selected_global_body_ids])
    np.testing.assert_array_equal(restarted_solve["body_qd"][selected_local_body_ids], selected_qd)
    np.testing.assert_allclose(restarted_solve["body_q_prev"][selected_local_body_ids], restored_coupling.numpy())
    np.testing.assert_allclose(
        restarted_solve["coupling_body_q_prev"][selected_local_body_ids], restored_coupling.numpy()
    )

    np.testing.assert_allclose(first_solve["body_q"][unsaved_local_body], initial_history[unsaved_local_body])
    expected_unsaved_qd_x = 0.75 + (body_q[unsaved_global_body, 0] - initial_history[unsaved_local_body, 0]) / (
        1.0 / 120.0
    )
    assert first_solve["body_qd"][unsaved_local_body, 0] == pytest.approx(expected_unsaved_qd_x)
    assert NewtonCouplerManager.get_vbd_pose_history_restore_status(queued).applied_exactly_once is True


@pytest.mark.parametrize("algorithm", ["proxy", "admm"])
@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA is unavailable")
def test_deferred_vbd_restore_is_consumed_by_preexisting_cuda_graph(
    algorithm: str,
    isolated_newton_manager,
):
    """Stable restore nodes accept data queued after graph capture."""
    device = "cuda:0"
    model = _build_multi_world_pose_history_model(device)
    coupled_solver, vbd_solver = _build_pose_history_coupler(model, algorithm)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    coupled_solver.step(state_0, state_1, control, None, 1.0 / 120.0)
    wp.synchronize()

    vbd_solver._rigid_pose_rebaseline_mask.fill_(True)
    with wp.ScopedCapture(device=device) as capture:
        coupled_solver.step(state_0, state_1, control, None, 1.0 / 120.0)

    body_ids = wp.array([6, 0], dtype=wp.int32, device=device)
    world_ids = wp.array([2, 0], dtype=wp.int32, device=device)
    restored_previous = wp.array(
        _pose_history_values(np.array([2, 0]), 50.0),
        dtype=wp.transform,
        device=device,
    )
    restored_coupling = wp.array(
        _pose_history_values(np.array([2, 0]), 60.0),
        dtype=wp.transform,
        device=device,
    )
    vbd_solver._rigid_pose_rebaseline_mask.fill_(True)
    queued = NewtonCouplerManager.restore_vbd_pose_history(
        "vbd",
        body_ids,
        world_ids,
        restored_previous,
        restored_coupling,
    )

    wp.capture_launch(capture.graph)
    status = NewtonCouplerManager.get_vbd_pose_history_restore_status(queued)
    assert status.applied_exactly_once is True
    local_body_ids = coupled_solver._entries["vbd"].body_global_to_local.numpy()[[6, 0]]
    np.testing.assert_allclose(
        vbd_solver._coupling_body_q_prev_snapshot.numpy()[local_body_ids],
        restored_coupling.numpy(),
    )

    wp.capture_launch(capture.graph)
    second_status = NewtonCouplerManager.get_vbd_pose_history_restore_status(status)
    assert second_status.applied_exactly_once is True
    assert second_status.application_count_deltas == (1, 1)
    assert second_status.body_application_count_deltas == (1, 1)


@pytest.mark.parametrize("algorithm", ["proxy", "admm"])
@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA is unavailable")
def test_preserved_input_pose_and_queued_restore_replay_in_cuda_graph(
    algorithm: str,
    isolated_newton_manager,
):
    """Captured projection and preserve nodes consume later restore data."""
    device = "cuda:0"
    model = _build_multi_world_pose_history_model(device)
    NewtonManager._model = model
    body_ids = wp.array([6, 0], dtype=wp.int32, device=device)

    def project(state):
        wp.launch(
            _project_pose_parity_bodies,
            dim=1,
            inputs=[state.body_q, 6, 0],
            device=device,
        )

    NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
        name="graphed-pose",
        entry_name="vbd",
        body_ids=body_ids,
        callback=project,
    )
    coupled_solver, vbd_solver = _build_pose_history_coupler(model, algorithm)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    NewtonCouplerManager._step_solver(state_0, state_1, control, None, 1.0 / 120.0)
    wp.synchronize()

    with wp.ScopedCapture(device=device) as capture:
        NewtonCouplerManager._step_solver(state_0, state_1, control, None, 1.0 / 120.0)

    restored_previous = wp.array(
        _pose_history_values(np.array([2, 0]), 50.0),
        dtype=wp.transform,
        device=device,
    )
    restored_coupling = wp.array(
        _pose_history_values(np.array([2, 0]), 60.0),
        dtype=wp.transform,
        device=device,
    )
    world_ids = wp.array([2, 0], dtype=wp.int32, device=device)
    vbd_solver._rigid_pose_rebaseline_mask.assign(np.array([True, False, True, False]))
    queued = NewtonCouplerManager.restore_vbd_pose_history(
        "vbd",
        body_ids,
        world_ids,
        restored_previous,
        restored_coupling,
    )

    wp.capture_launch(capture.graph)
    wp.synchronize()

    status = NewtonCouplerManager.get_vbd_pose_history_restore_status(queued)
    assert status.applied_exactly_once is True
    local_body_ids = coupled_solver._entries["vbd"].body_global_to_local.numpy()[[6, 0]]
    np.testing.assert_allclose(
        vbd_solver._coupling_body_q_prev_snapshot.numpy()[local_body_ids],
        restored_coupling.numpy(),
    )
    parent_projected_q = state_1.body_q.numpy()[[6, 0]]
    entry_projected_q = coupled_solver.entry_state("vbd", phase="output").body_q.numpy()[local_body_ids]
    np.testing.assert_allclose(entry_projected_q, parent_projected_q)
    assert not np.array_equal(parent_projected_q, restored_previous.numpy())


def test_vbd_pose_history_restore_validation_fails_before_mutation(isolated_newton_manager):
    """An invalid history shape must leave histories and every mask slot intact."""
    model = _build_multi_world_pose_history_model("cpu")
    _, vbd_solver = _build_pose_history_coupler(model, "proxy")
    local_worlds = vbd_solver.model.body_world.numpy()
    previous_values = _pose_history_values(local_worlds, 10.0)
    coupling_values = _pose_history_values(local_worlds, 20.0)
    vbd_solver.body_q_prev.assign(previous_values)
    vbd_solver._coupling_body_q_prev_snapshot.assign(coupling_values)
    vbd_solver._rigid_pose_rebaseline_mask.fill_(True)

    body_ids = wp.array([6, 0], dtype=wp.int32, device="cpu")
    world_ids = wp.array([2, 0], dtype=wp.int32, device="cpu")
    invalid_previous = wp.array(previous_values[:1], dtype=wp.transform, device="cpu")
    valid_coupling = wp.array(coupling_values[[2, 0]], dtype=wp.transform, device="cpu")

    with pytest.raises(ValueError, match="body_q_prev must have shape"):
        NewtonCouplerManager.restore_vbd_pose_history(
            "vbd",
            body_ids,
            world_ids,
            invalid_previous,
            valid_coupling,
        )

    np.testing.assert_allclose(vbd_solver.body_q_prev.numpy(), previous_values)
    np.testing.assert_allclose(vbd_solver._coupling_body_q_prev_snapshot.numpy(), coupling_values)
    np.testing.assert_array_equal(vbd_solver._rigid_pose_rebaseline_mask.numpy(), [True, True, True, True])


def test_vbd_pose_history_rejects_second_pending_restore_without_replacing_first(isolated_newton_manager):
    """Consecutive public resets must fail closed instead of mixing histories."""
    model = _build_multi_world_pose_history_model("cpu")
    coupled_solver, vbd_solver = _build_pose_history_coupler(model, "proxy")
    body_ids = wp.array([6, 0], dtype=wp.int32, device="cpu")
    world_ids = wp.array([2, 0], dtype=wp.int32, device="cpu")
    local_body_ids = coupled_solver._entries["vbd"].body_global_to_local.numpy()[[6, 0]]
    first_previous = wp.array(_pose_history_values(np.array([2, 0]), 10.0), dtype=wp.transform, device="cpu")
    first_coupling = wp.array(_pose_history_values(np.array([2, 0]), 20.0), dtype=wp.transform, device="cpu")
    second_previous = wp.array(_pose_history_values(np.array([2, 0]), 30.0), dtype=wp.transform, device="cpu")
    second_coupling = wp.array(_pose_history_values(np.array([2, 0]), 40.0), dtype=wp.transform, device="cpu")
    vbd_solver._rigid_pose_rebaseline_mask.fill_(True)

    first = NewtonCouplerManager.restore_vbd_pose_history("vbd", body_ids, world_ids, first_previous, first_coupling)
    with pytest.raises(RuntimeError, match="already has pending restores"):
        NewtonCouplerManager.restore_vbd_pose_history("vbd", body_ids, world_ids, second_previous, second_coupling)

    assert NewtonCouplerManager.get_vbd_pose_history_restore_status(first).pending is True
    state_0 = model.state()
    coupled_solver.step(state_0, model.state(), model.control(), None, 1.0 / 120.0)
    completed = NewtonCouplerManager.get_vbd_pose_history_restore_status(first)
    assert completed.applied_exactly_once is True
    np.testing.assert_allclose(
        vbd_solver._coupling_body_q_prev_snapshot.numpy()[local_body_ids],
        first_coupling.numpy(),
    )


@pytest.mark.parametrize("algorithm", ["proxy", "admm"])
def test_deferred_vbd_restore_fails_closed_if_rebaseline_is_consumed_before_solve(
    algorithm: str,
    isolated_newton_manager,
    monkeypatch: pytest.MonkeyPatch,
):
    """A lost reset boundary consumes the request without scattering it."""
    solve_boundary_histories: list[tuple[np.ndarray, np.ndarray]] = []
    original_step = SolverVBD.step

    def observe_step(self, state_in, state_out, control, contacts, dt):
        solve_boundary_histories.append(
            (
                self.body_q_prev.numpy().copy(),
                self._coupling_body_q_prev_snapshot.numpy().copy(),
            )
        )
        return original_step(self, state_in, state_out, control, contacts, dt)

    monkeypatch.setattr(SolverVBD, "step", observe_step)
    model = _build_multi_world_pose_history_model("cpu")
    coupled_solver, vbd_solver = _build_pose_history_coupler(model, algorithm)
    body_ids = wp.array([6, 0], dtype=wp.int32, device="cpu")
    world_ids = wp.array([2, 0], dtype=wp.int32, device="cpu")
    local_body_ids = coupled_solver._entries["vbd"].body_global_to_local.numpy()[[6, 0]]
    restored_previous = wp.array(
        _pose_history_values(np.array([2, 0]), 50.0),
        dtype=wp.transform,
        device="cpu",
    )
    restored_coupling = wp.array(
        _pose_history_values(np.array([2, 0]), 60.0),
        dtype=wp.transform,
        device="cpu",
    )
    initial_body_q = vbd_solver.model.body_q.numpy()
    vbd_solver.body_q_prev.assign(initial_body_q)
    vbd_solver._coupling_body_q_prev_snapshot.assign(initial_body_q)
    vbd_solver._rigid_pose_rebaseline_mask.fill_(True)
    queued = NewtonCouplerManager.restore_vbd_pose_history(
        "vbd", body_ids, world_ids, restored_previous, restored_coupling
    )

    vbd_solver._rigid_pose_rebaseline_mask.assign(np.array([False, True, False, True]))
    state_0 = model.state()
    coupled_solver.step(state_0, model.state(), model.control(), None, 1.0 / 120.0)
    status = NewtonCouplerManager.get_vbd_pose_history_restore_status(queued)

    assert status.pending is False
    assert status.applied_world_ids == ()
    assert status.failed_world_ids == (2, 0)
    assert status.superseded_world_ids == ()
    assert status.application_count_deltas == (0, 0)
    assert status.body_application_count_deltas == (0, 0)
    assert status.applied_exactly_once is False
    assert len(solve_boundary_histories) == 1
    boundary_previous, boundary_coupling = solve_boundary_histories[0]
    assert not np.array_equal(boundary_previous[local_body_ids], restored_previous.numpy())
    assert not np.array_equal(boundary_coupling[local_body_ids], restored_coupling.numpy())
    pending = vbd_solver._isaaclab_vbd_pose_history_restore
    np.testing.assert_array_equal(pending.world_pending.numpy(), [False, False, False, False])
    np.testing.assert_array_equal(pending.body_pending.numpy(), np.zeros(vbd_solver.model.body_count, dtype=bool))


def test_vbd_pose_history_rejects_index_dtype_rank_and_device(isolated_newton_manager):
    """Public index arrays must use the active model's exact CUDA-safe layout."""
    model = _build_multi_world_pose_history_model("cpu")
    _build_pose_history_coupler(model, "proxy")
    body_ids = wp.array([0], dtype=wp.int32, device="cpu")
    world_ids = wp.array([0], dtype=wp.int32, device="cpu")

    with pytest.raises(TypeError, match="body_ids must be a one-dimensional wp.int32 array"):
        NewtonCouplerManager.capture_vbd_pose_history(
            "vbd",
            wp.array([0.0], dtype=wp.float32, device="cpu"),
            world_ids,
        )
    with pytest.raises(TypeError, match="world_ids must be a one-dimensional wp.int32 array"):
        NewtonCouplerManager.capture_vbd_pose_history(
            "vbd",
            body_ids,
            wp.array([[0]], dtype=wp.int32, device="cpu"),
        )
    if wp.is_cuda_available():
        with pytest.raises(TypeError, match="body_ids must be on model device"):
            NewtonCouplerManager.capture_vbd_pose_history(
                "vbd",
                wp.array([0], dtype=wp.int32, device="cuda:0"),
                world_ids,
            )


def test_vbd_pose_history_rejects_history_dtype_rank_and_device_before_queue(isolated_newton_manager):
    """Both serialized history arrays must match the exact solver layout."""
    model = _build_multi_world_pose_history_model("cpu")
    _, vbd_solver = _build_pose_history_coupler(model, "proxy")
    body_ids = wp.array([0], dtype=wp.int32, device="cpu")
    world_ids = wp.array([0], dtype=wp.int32, device="cpu")
    valid_history = wp.array(_pose_history_values(np.array([0]), 10.0), dtype=wp.transform, device="cpu")
    vbd_solver._rigid_pose_rebaseline_mask.fill_(True)

    with pytest.raises(TypeError, match="body_q_prev must be a one-dimensional wp.transform array"):
        NewtonCouplerManager.restore_vbd_pose_history(
            "vbd",
            body_ids,
            world_ids,
            wp.zeros(1, dtype=wp.float32, device="cpu"),
            valid_history,
        )
    with pytest.raises(TypeError, match="coupling_body_q_prev must be a one-dimensional wp.transform array"):
        NewtonCouplerManager.restore_vbd_pose_history(
            "vbd",
            body_ids,
            world_ids,
            valid_history,
            wp.empty((1, 1), dtype=wp.transform, device="cpu"),
        )
    if wp.is_cuda_available():
        with pytest.raises(TypeError, match="body_q_prev must be on model device"):
            NewtonCouplerManager.restore_vbd_pose_history(
                "vbd",
                body_ids,
                world_ids,
                wp.array(valid_history.numpy(), dtype=wp.transform, device="cuda:0"),
                valid_history,
            )

    pending = vbd_solver._isaaclab_vbd_pose_history_restore
    np.testing.assert_array_equal(pending.world_pending.numpy(), [False, False, False, False])
    np.testing.assert_array_equal(pending.body_pending.numpy(), np.zeros(vbd_solver.model.body_count, dtype=bool))
    np.testing.assert_array_equal(vbd_solver._rigid_pose_rebaseline_mask.numpy(), [True, True, True, True])


@pytest.mark.parametrize(
    ("body_values", "world_values", "error_type", "match"),
    [
        ([0, 0], [0], ValueError, "body_ids must not contain duplicates"),
        ([2], [0], ValueError, "not owned"),
        ([0], [1], ValueError, "exactly match"),
        ([9], [0], ValueError, "must lie"),
    ],
)
def test_vbd_pose_history_rejects_invalid_body_world_selection(
    body_values: list[int],
    world_values: list[int],
    error_type: type[Exception],
    match: str,
    isolated_newton_manager,
):
    """Body ownership and represented-world checks reject ambiguous selections."""
    model = _build_multi_world_pose_history_model("cpu")
    _build_pose_history_coupler(model, "proxy")
    body_ids = wp.array(body_values, dtype=wp.int32, device="cpu")
    world_ids = wp.array(world_values, dtype=wp.int32, device="cpu")

    with pytest.raises(error_type, match=match):
        NewtonCouplerManager.capture_vbd_pose_history("vbd", body_ids, world_ids)


def test_proxy_destination_can_receive_only_proxy_bodies(isolated_newton_manager):
    model = _build_overlapping_body_model()
    solver_cfg = CouplerProxyCfg(
        entries=[
            CouplerEntryCfg(
                name="source",
                solver_cfg=XPBDSolverCfg(iterations=2),
                bodies=[r"/World/Source/body"],
            ),
            CouplerEntryCfg(name="destination", solver_cfg=XPBDSolverCfg(iterations=2)),
        ],
        proxies=[
            CouplerProxyMappingCfg(
                source="source",
                destination="destination",
                bodies=[r"/World/Source/body"],
            )
        ],
    )

    NewtonManager._model = model
    NewtonCouplerManager._build_solver(model, solver_cfg)

    assert NewtonManager._solver._entries["destination"].proxy_body_local_indices.numpy().tolist() == [0]


@pytest.mark.parametrize(
    ("algorithm", "expected_solver_type"),
    [
        pytest.param("proxy", SolverCoupledProxy, id="proxy"),
        pytest.param("admm", SolverCoupledADMM, id="admm"),
    ],
)
def test_real_coupler_constructs_resets_and_steps(
    algorithm: str,
    expected_solver_type: type,
    isolated_newton_manager,
):
    """Construct, prepare contacts, reset, and step the pinned Newton solver."""
    model = _build_overlapping_body_model()
    entries = _entry_configs()
    if algorithm == "proxy":
        solver_cfg = CouplerProxyCfg(
            entries=entries,
            proxies=[
                CouplerProxyMappingCfg(
                    source="source",
                    destination="destination",
                    bodies=[0],
                )
            ],
            iterations=1,
        )
    else:
        solver_cfg = CouplerAdmmCfg(entries=entries, iterations=1)

    NewtonManager._model = model
    NewtonCouplerManager._build_solver(model, solver_cfg)
    solver = NewtonManager._solver

    assert isinstance(solver, expected_solver_type)
    assert solver.entry_names() == ("source", "destination")
    for name in solver.entry_names():
        nested_solver = solver.solver(name)
        assert isinstance(nested_solver, SolverXPBD)
        assert nested_solver.model is solver.view(name)

    NewtonCouplerManager._initialize_contacts()
    collision_pipeline = NewtonManager._collision_pipeline
    contacts = NewtonManager._contacts
    assert isinstance(collision_pipeline, CollisionPipeline)
    assert contacts is not None
    assert set(solver._entry_contact_buffers) == {"source", "destination"}

    state_0 = model.state()
    state_1 = model.state()
    solver.reset(state_0)
    assert solver.entry_output_state_valid() is False

    collision_pipeline.collide(state_0, contacts)
    assert int(contacts.rigid_contact_count.numpy()[0]) >= 1
    body_q_before = state_0.body_q.numpy().copy()

    solver.step(state_0, state_1, model.control(), contacts, 1.0 / 60.0)

    body_q_after = state_1.body_q.numpy()
    assert solver.entry_output_state_valid() is True
    assert np.all(np.isfinite(body_q_after))
    assert np.all(np.isfinite(state_1.body_qd.numpy()))
    assert np.any(body_q_after[:, 2] < body_q_before[:, 2])

    solver.reset(state_1)
    assert solver.entry_output_state_valid() is False
