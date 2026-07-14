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
from newton import GeoType, ModelBuilder, ShapeFlags, solvers
from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx
from newton.geometry import compute_inertia_shape, transform_inertia

from isaaclab.managers import SceneEntityCfg

from isaaclab_contrib.cable import cable_object as cable_object_module

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


def test_waterhose_refreshes_outer_contacts_each_solver_substep():
    """MJWarp must not reuse a stale robot/fridge manifold for the whole 10 ms step."""
    cfg = waterhose_env_cfg.WaterhoseEnvCfg()

    assert cfg.sim.physics.num_substeps > 1
    assert cfg.sim.physics.collision_decimation == 1


@pytest.mark.parametrize(
    ("asset_path", "expected_shape_count"),
    [
        (waterhose_env_cfg._RBY1_USD, 4),
        (waterhose_env_cfg._FRIDGE_COLLISION_PROXY_USD, 1),
    ],
    ids=["rby1-fingers", "fridge-housing"],
)
def test_mjwarp_contact_shapes_author_raw_critically_damped_solref(asset_path, expected_shape_count):
    """MJWarp contact shapes use a direct response instead of the VBD ke/kd fallback."""
    builder = ModelBuilder()
    solvers.SolverMuJoCo.register_custom_attributes(builder)
    result = builder.add_usd(
        asset_path,
        floating=False,
        load_visual_shapes=False,
        hide_collision_shapes=False,
        parse_mujoco_options=False,
        schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
    )
    shape_indices = list(result["path_shape_map"].values())
    solref_modes = builder.custom_attributes["mujoco:solref_mode"].values
    solrefs = builder.custom_attributes["mujoco:solref"].values

    assert len(shape_indices) == expected_shape_count
    np.testing.assert_array_equal(
        np.asarray([solref_modes[index] for index in shape_indices], dtype=np.int32),
        np.full(expected_shape_count, SOLREF_MODE_RAW, dtype=np.int32),
    )
    np.testing.assert_allclose(
        np.asarray([solrefs[index] for index in shape_indices], dtype=np.float32),
        np.tile((0.005, 1.0), (expected_shape_count, 1)),
    )


def test_waterhose_proxy_uses_authored_finger_inertia():
    """The grasp proxy must not substitute MJWarp's articulated effective inertia."""
    cfg = waterhose_env_cfg.WaterhoseEnvCfg()
    entries = {entry.name: entry for entry in cfg.sim.physics.solver_cfg.entries}

    assert entries["mjc"].use_solver_effective_mass is False


def test_cable_uses_post_replicate_coloring_hook(monkeypatch):
    """Procedural rods follow the same post-replication coloring pattern as deformables."""

    monkeypatch.setattr(NewtonManager, "_per_world_builder_hooks", [], raising=False)
    monkeypatch.setattr(NewtonManager, "_post_replicate_hooks", [], raising=False)
    monkeypatch.setattr(NewtonManager, "_pre_render_callbacks", {}, raising=False)

    cable_object_module.install_cable_builder_hooks()

    assert cable_object_module.color_registered_cables in NewtonManager._post_replicate_hooks
    assert (
        NewtonManager._pre_render_callbacks["cable_curve_sync"]
        is cable_object_module.sync_registered_cable_curves_to_usd
    )
    color_calls = []
    NewtonManager._cable_registry.append(object())
    cable_object_module.color_registered_cables(SimpleNamespace(color=lambda: color_calls.append(True)))
    assert color_calls == [True]


def test_waterhose_proxy_contacts_warm_start_the_grip():
    cfg = waterhose_env_cfg.WaterhoseEnvCfg()
    entries = {entry.name: entry for entry in cfg.sim.physics.solver_cfg.entries}

    assert cfg.sim.physics.use_cuda_graph is True
    assert entries["vbd"].solver_cfg.rigid_contact_history is True
    assert cfg.sim.physics.collision_cfg.rigid_contact_max >= waterhose_env_cfg._PROXY_RIGID_CONTACT_MAX


def test_waterhose_proxy_pipeline_restores_task_materials_on_imported_contact_shapes(monkeypatch, capsys):
    """Both grippers keep connector/rod contacts while unrelated proxy pairs are filtered."""

    class RecordingPipeline:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

    import newton

    monkeypatch.setattr(newton, "CollisionPipeline", RecordingPipeline)
    collide = int(ShapeFlags.COLLIDE_SHAPES) | int(ShapeFlags.COLLIDE_PARTICLES)
    model = SimpleNamespace(
        device="cpu",
        body_label=[
            "/World/envs/env_0/Robot/left_gripper_leftfinger",
            "/World/envs/env_0/Robot/left_gripper_rightfinger",
            "/World/envs/env_0/Robot/right_gripper_leftfinger",
            "/World/envs/env_0/Robot/right_gripper_rightfinger",
            "/World/envs/env_0/Cable1/cable_edge_body_0",
            "/World/envs/env_0/Cable1/cable_edge_body_1",
            "/World/envs/env_0/FridgeCableCollision/Housing",
        ],
        # Shape 0 is a render shape on the left-finger body. It must not be
        # promoted to a collision shape or have its render material rewritten.
        shape_body=wp.array([0, 0, 1, 2, 3, 4, 5, 6, -1], dtype=wp.int32, device="cpu"),
        shape_flags=wp.array(
            [int(ShapeFlags.VISIBLE), collide, collide, collide, collide, collide, collide, collide, collide],
            dtype=wp.int32,
            device="cpu",
        ),
        shape_label=[
            "left_visual",
            "left_left_collision",
            "left_right_collision",
            "right_left_collision",
            "right_right_collision",
            "/World/envs/env_0/Cable1/waterhose_connector",
            "/World/envs/env_0/Cable1/cable_edge_capsule_1",
            "/World/envs/env_0/FridgeCableCollision/Housing",
            "/World/envs/env_0/Fridge/Cable008/SocketCollision/Cable008_SocketCollision",
        ],
        shape_contact_pairs=wp.array(
            [
                (1, 5),
                (2, 5),
                (3, 5),
                (4, 5),
                (1, 6),
                (2, 6),
                (3, 6),
                (4, 6),
                (1, 7),
                (2, 7),
                (3, 7),
                (4, 7),
                (1, 8),
                (2, 8),
                (3, 8),
                (4, 8),
                (1, 2),
                (3, 4),
                (6, 7),
                (5, 8),
                (6, 8),
            ],
            dtype=wp.vec2i,
            device="cpu",
        ),
        shape_material_ke=wp.array([17.0] + [2500.0] * 8, dtype=wp.float32),
        shape_material_kd=wp.array([19.0] + [100.0] * 8, dtype=wp.float32),
        shape_material_mu=wp.array([7.0] + [1.0] * 8, dtype=wp.float32, device="cpu"),
        shape_margin=wp.array([0.0] * 9, dtype=wp.float32, device="cpu"),
        shape_gap=wp.array([0.1, 0.01, 0.01, 0.01, 0.01, 0.001, 0.01, 0.01, 0.0], dtype=wp.float32),
    )

    pipeline = waterhose_env_cfg._make_proxy_collision_pipeline(model)

    np.testing.assert_allclose(model.shape_material_ke.numpy(), [17.0] + [1.0e4] * 8)
    np.testing.assert_allclose(model.shape_material_kd.numpy(), [19.0] + [0.1] * 8)
    np.testing.assert_allclose(model.shape_material_mu.numpy(), [7.0, 20.0, 20.0, 20.0, 20.0, 0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(model.shape_margin.numpy(), [0.0, 0.001, 0.001, 0.001, 0.001, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(model.shape_gap.numpy(), [0.1, 0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 0.0])
    np.testing.assert_array_equal(
        pipeline.kwargs["shape_pairs_filtered"].numpy(),
        np.asarray(
            [
                (1, 5),
                (2, 5),
                (3, 5),
                (4, 5),
                (1, 6),
                (2, 6),
                (3, 6),
                (4, 6),
                (6, 7),
                (5, 8),
                (6, 8),
            ],
            dtype=np.int32,
        ),
    )
    assert "kept 11 contact pair(s), dropped 10 finger-vs-unrelated pair(s)" in capsys.readouterr().out
    assert pipeline.kwargs["contact_matching"] == "latest"


def test_waterhose_vbd_owns_rod_bodies_not_particles():
    """The hose follows PR 5641's articulated rod representation."""

    cfg = waterhose_env_cfg.WaterhoseEnvCfg()
    entries = {entry.name: entry for entry in cfg.sim.physics.solver_cfg.entries}

    assert entries["mjc"].bodies == [waterhose_env_cfg._ROBOT_BODY_PATTERN]
    assert entries["mjc"].include_body_shapes is True
    assert entries["mjc"].shape_label_patterns == [waterhose_env_cfg._FRIDGE_ROBOT_COLLISION_PATTERN]
    assert entries["vbd"].bodies == [waterhose_env_cfg._CABLE_BODY_PATTERN]
    assert entries["vbd"].particles == []
    assert entries["vbd"].all_particles is False

    proxy = cfg.sim.physics.solver_cfg.proxies[0]
    assert proxy.bodies == [r"/World/envs/env_.*/Robot/.*/(left|right)_gripper_(left|right)finger"]
    assert not hasattr(cfg.sim.physics.solver_cfg, "scene_cfg")


def test_waterhose_active_visualizers_share_the_environment_camera_view():
    cfg = waterhose_env_cfg.WaterhoseEnvCfg()

    assert {visualizer.visualizer_type for visualizer in cfg.sim.visualizer_cfgs} == {"kit", "newton"}
    for visualizer in cfg.sim.visualizer_cfgs:
        assert tuple(visualizer.eye) == tuple(cfg.viewer.eye)
        assert tuple(visualizer.lookat) == tuple(cfg.viewer.lookat)
    assert cfg.viewer.origin_type == "world"


def test_waterhose_camera_keeps_robot_clear_of_the_fridge():
    """The shared camera must not put the robot behind the fridge again."""

    eye = np.asarray(waterhose_env_cfg._SCENE_CAMERA_EYE, dtype=np.float64)
    lookat = np.asarray(waterhose_env_cfg._SCENE_CAMERA_LOOKAT, dtype=np.float64)
    forward = lookat - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)

    fridge_center = np.asarray(waterhose_env_cfg.FRIDGE_POS, dtype=np.float64)
    scene_cfg = waterhose_env_cfg.WaterhoseSceneCfg(num_envs=1, env_spacing=2.5)
    robot_center = np.asarray(scene_cfg.robot.init_state.pos, dtype=np.float64)

    def projected_x(point: np.ndarray) -> float:
        camera_point = point - eye
        return float(camera_point @ right / (camera_point @ forward))

    # Their centers were separated by only 0.04 in the previous head-on projection, leaving the
    # robot almost entirely hidden. The side view makes their ordering immediately apparent in Kit
    # and in Newton's viewer while retaining both objects in one scene-fit view.
    assert abs(projected_x(robot_center) - projected_x(fridge_center)) > 0.15


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


def test_robot_asset_enables_both_grippers_finger_colliders():
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

    assert collision_labels == {
        "left_gripper_leftfinger",
        "left_gripper_rightfinger",
        "right_gripper_leftfinger",
        "right_gripper_rightfinger",
    }
    assert builder.constraint_mimic_enabled == []


def test_robot_asset_imports_visible_render_shapes_for_newton():
    builder = ModelBuilder()
    result = builder.add_usd(
        waterhose_env_cfg._RBY1_USD,
        floating=False,
        load_visual_shapes=True,
        hide_collision_shapes=True,
        parse_mujoco_options=False,
    )
    visible_shapes = [
        shape_index
        for shape_index in result["path_shape_map"].values()
        if int(builder.shape_flags[shape_index]) & int(ShapeFlags.VISIBLE)
    ]

    # The two wheels are visual children of the torso, not redundant fixed rigid bodies.
    assert builder.body_count == 35
    assert len(visible_shapes) >= 30
    assert all(builder.shape_body[shape_index] >= 0 for shape_index in visible_shapes)
    wheel_shapes = [
        shape_index for shape_index in visible_shapes if "wheel" in str(builder.shape_label[shape_index]).lower()
    ]
    assert len(wheel_shapes) == 2
    assert {builder.shape_body[shape_index] for shape_index in wheel_shapes} == {0}


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
    assert scene.cable1.tail_anchor_prim_path == "/World/envs/env_.*/Anchor1"


def test_waterhose_uses_current_vbd_units_for_anchored_cable_material():
    """The anchored hose must not lever itself out of the gripper with legacy bend damping."""

    material = waterhose_env_cfg.WaterhoseSceneCfg().cable1.spawn.physics_material

    assert material.stretch_stiffness == pytest.approx(1.0e6)
    assert material.stretch_damping == pytest.approx(1.0e-2)
    assert material.bend_stiffness == pytest.approx(3.0e-1)
    assert material.bend_damping == pytest.approx(2.0e-2)
    assert material.density == pytest.approx(100.0)


def test_compound_connector_aligns_clear_of_the_success_region():
    cfg = waterhose_env_cfg.WaterhoseEnvCfg()
    success = cfg.terminations.success.params

    assert pytest.approx(-0.030) == SOCKET_ALIGN_TIP_DEPTH
    assert pytest.approx(-0.004) == SOCKET_SEATED_TIP_DEPTH
    assert success["min_depth"] > SOCKET_ALIGN_TIP_DEPTH
    assert success["min_depth"] < SOCKET_SEATED_TIP_DEPTH < success["max_depth"]
    assert success["radial_threshold"] == pytest.approx(0.001)


def test_canonical_assets_author_socket_sdf_and_render_only_connector():
    from pxr import Usd, UsdGeom

    fridge_stage = Usd.Stage.Open(waterhose_env_cfg._FRIDGE_USD)
    socket = fridge_stage.GetPrimAtPath("/root/Cable008/SocketCollision/Cable008_SocketCollision")
    api_schemas = socket.GetMetadata("apiSchemas").GetAddedOrExplicitItems()
    assert "NewtonSDFCollisionAPI" in api_schemas
    assert socket.GetAttribute("newton:sdfMaxResolution").Get() == 128
    assert socket.GetAttribute("visibility").Get() == "invisible"

    cable_stage = Usd.Stage.Open(waterhose_env_cfg._CABLE1_USD)
    cable_curve = UsdGeom.BasisCurves(cable_stage.GetPrimAtPath("/cable001/curve_0"))
    assert cable_curve.GetNormalsAttr().Get() == [(0.0, 1.0, 0.0)]
    assert cable_curve.GetNormalsInterpolation() == UsdGeom.Tokens.constant
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
            body_offsets=[head_body],
            edges=[(0, 1)],
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


def test_waterhose_gripper_proxy_uses_requested_mass_scale():
    cfg = waterhose_env_cfg.WaterhoseEnvCfg()
    proxy = cfg.sim.physics.solver_cfg.proxies[0]

    assert proxy.mass_scale == pytest.approx(1.0)
    assert proxy.mass_scale == pytest.approx(waterhose_env_cfg._GRIPPER_PROXY_MASS_SCALE)


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
        _registry_entry=SimpleNamespace(body_offsets=[0, 2], edges=[(0, 1), (1, 2)]),
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
