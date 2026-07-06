# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the waterhose physics configuration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from newton import GeoType, ModelBuilder, ShapeFlags
from newton.geometry import compute_inertia_shape, transform_inertia

from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.contrib.waterhose import waterhose_env_cfg
from isaaclab_tasks.contrib.waterhose.cable import WaterhoseCableObject, WaterhoseCableObjectCfg
from isaaclab_tasks.contrib.waterhose.geometry import (
    CONNECTOR_LOCAL_POS,
    CONNECTOR_LOCAL_QUAT_XYZW,
    CONNECTOR_MASS,
    CONNECTOR_TIP_LEN,
    CONNECTOR_TIP_LOCAL_POS,
    SOCKET_ALIGN_TIP_DEPTH,
    SOCKET_SEATED_TIP_DEPTH,
)


def _solver_tuning(monkeypatch, *, substeps: str | None = None, vbd_iters: str | None = None) -> tuple[int, int]:
    if substeps is None:
        monkeypatch.delenv("WATERHOSE_SUBSTEPS", raising=False)
    else:
        monkeypatch.setenv("WATERHOSE_SUBSTEPS", substeps)
    if vbd_iters is None:
        monkeypatch.delenv("WATERHOSE_VBD_ITERS", raising=False)
    else:
        monkeypatch.setenv("WATERHOSE_VBD_ITERS", vbd_iters)

    cfg = waterhose_env_cfg.WaterhoseEnvCfg()
    entries = {entry.name: entry for entry in cfg.sim.physics.solver_cfg.entries}
    return cfg.sim.physics.num_substeps, entries["vbd"].solver_cfg.iterations


def test_waterhose_uses_high_fidelity_solver_tuning_defaults(monkeypatch):
    assert _solver_tuning(monkeypatch) == (10, 20)


def test_waterhose_proxy_uses_authored_finger_inertia():
    cfg = waterhose_env_cfg.WaterhoseEnvCfg()
    entries = {entry.name: entry for entry in cfg.sim.physics.solver_cfg.entries}

    assert entries["mjc"].use_solver_effective_mass is False


def test_waterhose_admm_disables_unmatched_vbd_contact_history():
    cfg = waterhose_env_cfg.WaterhoseAdmmIkEnvCfg()
    entries = {entry.name: entry for entry in cfg.sim.physics.solver_cfg.entries}

    assert entries["vbd"].solver_cfg.rigid_contact_history is False


def test_waterhose_solver_tuning_accepts_environment_overrides(monkeypatch):
    assert _solver_tuning(monkeypatch, substeps="6", vbd_iters="12") == (6, 12)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("WATERHOSE_SUBSTEPS", "not-an-int"),
        ("WATERHOSE_SUBSTEPS", "0"),
        ("WATERHOSE_VBD_ITERS", "-1"),
    ],
)
def test_waterhose_solver_tuning_rejects_invalid_values(monkeypatch, name, value):
    monkeypatch.delenv("WATERHOSE_SUBSTEPS", raising=False)
    monkeypatch.delenv("WATERHOSE_VBD_ITERS", raising=False)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=rf"{name} must be a positive integer"):
        waterhose_env_cfg.WaterhoseEnvCfg()


def test_robot_asset_enables_only_right_finger_colliders():
    builder = ModelBuilder()
    builder.add_usd(
        waterhose_env_cfg._RBY1_USD,
        floating=False,
        load_visual_shapes=False,
        hide_collision_shapes=False,
        parse_mujoco_options=False,
    )
    collide_mask = int(ShapeFlags.COLLIDE_SHAPES) | int(ShapeFlags.COLLIDE_PARTICLES)
    collision_bodies = {
        builder.shape_body[shape] for shape, flags in enumerate(builder.shape_flags) if flags & collide_mask
    }
    collision_labels = {str(builder.body_label[body]).rsplit("/", 1)[-1] for body in collision_bodies}

    assert collision_labels == {"right_gripper_leftfinger", "right_gripper_rightfinger"}
    assert builder.constraint_mimic_enabled == []


def test_scene_imports_authored_assets_without_model_init_callbacks():
    scene = waterhose_env_cfg.WaterhoseSceneCfg()

    assert scene.fridge.spawn.usd_path == waterhose_env_cfg._FRIDGE_USD
    assert scene.robot.spawn.usd_path == waterhose_env_cfg._RBY1_USD
    assert scene.cable1.spawn.usd_path == waterhose_env_cfg._CABLE1_USD
    assert scene.anchor1.spawn.rigid_props is None
    assert scene.anchor1.spawn.collision_props is None
    assert not hasattr(scene, "connector_visual")
    assert waterhose_env_cfg._SOCKET_CONTACT_PROPERTIES.contact_gap == 0.0
    assert scene.cable1.connector_gap == pytest.approx(0.001)
    assert scene.cable1.attachments[0].cable_anchor == -1


def test_compound_connector_aligns_clear_of_the_success_region():
    cfg = waterhose_env_cfg.WaterhoseEnvCfg()
    success = cfg.terminations.success.params

    assert pytest.approx(-0.030) == SOCKET_ALIGN_TIP_DEPTH
    assert pytest.approx(-0.004) == SOCKET_SEATED_TIP_DEPTH
    assert success["min_depth"] > SOCKET_ALIGN_TIP_DEPTH
    assert success["min_depth"] < SOCKET_SEATED_TIP_DEPTH < success["max_depth"]
    assert success["radial_threshold"] == pytest.approx(0.001)


def test_canonical_assets_author_socket_sdf_and_render_only_connector():
    from pxr import Usd

    fridge_stage = Usd.Stage.Open(waterhose_env_cfg._FRIDGE_USD)
    socket = fridge_stage.GetPrimAtPath("/root/Cable008/SocketCollision/Cable008_SocketCollision")
    api_schemas = socket.GetMetadata("apiSchemas").GetAddedOrExplicitItems()
    assert "NewtonSDFCollisionAPI" in api_schemas
    assert socket.GetAttribute("newton:sdfMaxResolution").Get() == 128
    assert socket.GetAttribute("visibility").Get() == "invisible"

    cable_stage = Usd.Stage.Open(waterhose_env_cfg._CABLE1_USD)
    connector = cable_stage.GetPrimAtPath("/cable001/cable_edge_body_0/connector")
    connector_mesh = connector.GetChild("plug_mesh")
    assert connector.IsValid()
    assert connector_mesh.IsValid()
    assert connector.GetAppliedSchemas() == []
    assert connector_mesh.GetAppliedSchemas() == []

    builder = ModelBuilder()
    result = builder.add_usd(waterhose_env_cfg._PLUG_USD, floating=False, load_visual_shapes=True)
    connector_shape = next(iter(result["path_shape_map"].values()))
    vertices = np.asarray(builder.shape_source[connector_shape].vertices)
    assert pytest.approx(float(vertices[:, 2].max()), abs=1.0e-7) == CONNECTOR_TIP_LEN
    assert CONNECTOR_TIP_LOCAL_POS[2] == CONNECTOR_TIP_LEN


def test_connector_mesh_is_lumped_into_the_cable_head():
    builder = ModelBuilder()
    head_mass = 1.3585861e-4
    head_com = wp.vec3(0.0, 0.0, 0.0)
    head_inertia = wp.mat33(1.0e-7, 0.0, 0.0, 0.0, 1.0e-7, 0.0, 0.0, 0.0, 1.0e-7)
    head_body = builder.add_body(
        mass=head_mass,
        com=head_com,
        inertia=head_inertia,
        label="/World/envs/env_0/Cable1/cable_edge_body_0",
    )
    cfg = WaterhoseCableObjectCfg(
        prim_path="/World/envs/env_.*/Cable1",
        connector_usd_path=waterhose_env_cfg._PLUG_USD,
        connector_mass=CONNECTOR_MASS,
        connector_local_pos=CONNECTOR_LOCAL_POS,
        connector_local_quat=CONNECTOR_LOCAL_QUAT_XYZW,
        connector_shape_label=waterhose_env_cfg._CONNECTOR_SHAPE_TOKEN,
    )
    cable = SimpleNamespace(
        cfg=cfg,
        _registry_entry=SimpleNamespace(
            prim_path=cfg.prim_path,
            segment_body_indices=[[head_body]],
        ),
        _connector_shape_indices=[],
        _connector_head_body_ids=None,
        _connector_local_pose=None,
        _connector_geometry=None,
        _cable_registry_index=0,
    )
    cable._load_connector_geometry = lambda: WaterhoseCableObject._load_connector_geometry(cable)
    mesh, scale, connector_xform, density, is_solid, _ = cable._load_connector_geometry()
    connector_mass, connector_com, connector_inertia = compute_inertia_shape(
        GeoType.MESH,
        scale,
        mesh,
        density,
        is_solid,
        cfg.connector_margin,
    )
    connector_com_head = wp.transform_point(connector_xform, connector_com)
    expected_mass = head_mass + connector_mass
    expected_com = (head_mass * head_com + connector_mass * connector_com_head) / expected_mass
    expected_inertia = transform_inertia(
        head_mass,
        head_inertia,
        head_com - expected_com,
        wp.quat_identity(),
    ) + transform_inertia(
        connector_mass,
        connector_inertia,
        connector_com_head - expected_com,
        wp.transform_get_rotation(connector_xform),
    )

    body_count = builder.body_count
    joint_count = builder.joint_count
    WaterhoseCableObject._add_connector_to_builder(cable, builder, 0, [0.0] * 3, [0.0, 0.0, 0.0, 1.0])

    assert builder.body_count == body_count
    assert builder.joint_count == joint_count
    assert len(cable._connector_shape_indices) == 1
    connector_shape = cable._connector_shape_indices[0]
    assert builder.shape_body[connector_shape] == head_body
    assert builder.shape_collision_group[connector_shape] == -1
    assert builder.body_mass[head_body] == pytest.approx(expected_mass, rel=1.0e-6)
    np.testing.assert_allclose(np.asarray(builder.body_com[head_body]), np.asarray(expected_com), rtol=1.0e-6)
    inertia = np.asarray(builder.body_inertia[head_body]).reshape(3, 3)
    np.testing.assert_allclose(inertia, np.asarray(expected_inertia).reshape(3, 3), rtol=1.0e-6, atol=1.0e-12)
    assert np.all(np.linalg.eigvalsh(inertia) > 0.0)


def test_waterhose_reset_restores_the_full_scene_and_joint_targets():
    events = waterhose_env_cfg.EventCfg()

    assert events.reset_scene.func is waterhose_env_cfg.mdp.reset_scene_to_default
    assert events.reset_scene.params == {"reset_joint_targets": True}
    assert not hasattr(events, "reset_robot_joints")


def test_waterhose_cable_reset_event_runs_after_full_scene_reset():
    events = waterhose_env_cfg.EventCfg()
    reset_cable_to_default = getattr(waterhose_env_cfg, "reset_cable_to_default", None)

    assert reset_cable_to_default is not None
    assert events.reset_cable.func is reset_cable_to_default
    assert events.reset_cable.params == {"asset_cfg": SceneEntityCfg("cable1")}
    term_names = list(events.__dict__)
    assert term_names.index("reset_scene") < term_names.index("reset_cable")
    assert term_names == ["reset_scene", "reset_cable"]


@pytest.mark.parametrize(
    ("selected_env_ids", "expected_env_ids"),
    [
        (torch.tensor([1], dtype=torch.int64), [1]),
        ([1], [1]),
        (slice(None), [0, 1]),
    ],
    ids=["tensor", "sequence", "full-reset-slice"],
)
def test_waterhose_cable_reset_restores_selected_segment_state(monkeypatch, selected_env_ids, expected_env_ids):
    reset_cable_to_default = getattr(waterhose_env_cfg, "reset_cable_to_default", None)
    assert reset_cable_to_default is not None

    default_body_q = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    disturbed_body_q = default_body_q.copy()
    disturbed_body_q[:, 1] += 0.5
    disturbed_body_q[:, 3:7] = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    disturbed_body_qd = np.full((4, 6), 3.0, dtype=np.float32)
    model = SimpleNamespace(body_q=wp.array(default_body_q, dtype=wp.transformf, device="cpu"))
    state = SimpleNamespace(
        body_q=wp.array(disturbed_body_q, dtype=wp.transformf, device="cpu"),
        body_qd=wp.array(disturbed_body_qd, dtype=wp.spatial_vectorf, device="cpu"),
    )
    articulation_ids = object()
    cable = SimpleNamespace(
        _registry_entry=SimpleNamespace(segment_body_indices=[[0, 1], [2, 3]]),
        _root_view=SimpleNamespace(articulation_ids=articulation_ids),
    )
    env = SimpleNamespace(scene={"cable1": cable}, num_envs=2, device="cpu")
    asset_cfg = SceneEntityCfg("cable1")
    cfg = waterhose_env_cfg.EventTerm(
        func=reset_cable_to_default,
        mode="reset",
        params={"asset_cfg": asset_cfg},
    )
    invalidations = []
    monkeypatch.setattr(NewtonManager, "get_model", classmethod(lambda cls: model))
    monkeypatch.setattr(NewtonManager, "get_state_0", classmethod(lambda cls: state))
    monkeypatch.setattr(
        NewtonManager,
        "invalidate_fk",
        classmethod(lambda cls, **kwargs: invalidations.append(kwargs)),
    )

    term = reset_cable_to_default(cfg, env)
    term(env, selected_env_ids, asset_cfg=asset_cfg)

    body_q_after = state.body_q.numpy()
    body_qd_after = state.body_qd.numpy()
    expected_body_q = disturbed_body_q.copy()
    expected_body_qd = disturbed_body_qd.copy()
    expected_body_ids = np.array([[0, 1], [2, 3]], dtype=np.int64)[expected_env_ids].reshape(-1)
    expected_body_q[expected_body_ids] = default_body_q[expected_body_ids]
    expected_body_qd[expected_body_ids] = 0.0
    np.testing.assert_array_equal(body_q_after, expected_body_q)
    np.testing.assert_array_equal(body_qd_after, expected_body_qd)
    assert len(invalidations) == 1
    np.testing.assert_array_equal(invalidations[0]["env_ids"].numpy(), np.array(expected_env_ids, dtype=np.int32))
    assert invalidations[0]["articulation_ids"] is articulation_ids
