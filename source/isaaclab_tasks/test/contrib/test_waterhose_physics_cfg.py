# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the waterhose physics configuration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import NewtonManager
from newton import GeoType, ModelBuilder, ShapeFlags, solvers
from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx
from newton.geometry import compute_inertia_shape

import isaaclab.sim as sim_utils
from isaaclab.assets import CableObjectCfg

from isaaclab_tasks.contrib.waterhose import waterhose_env, waterhose_env_cfg
from isaaclab_tasks.contrib.waterhose.cable import (
    WaterhoseCableBuilderExtension,
    WaterhoseCableObjectCfg,
    spawn_waterhose_cable_from_usd,
)
from isaaclab_tasks.contrib.waterhose.geometry import (
    CONNECTOR_TIP_LEN,
    CONNECTOR_TIP_LOCAL_POS,
    SOCKET_ALIGN_TIP_DEPTH,
    SOCKET_COLLISION_MESH_PATTERN,
    SOCKET_RETAINED_AXIS_COS,
    SOCKET_RETAINED_MAX_TIP_DEPTH,
    SOCKET_RETAINED_MIN_TIP_DEPTH,
    SOCKET_RETAINED_RADIAL_TOLERANCE,
    SOCKET_SEATED_TIP_DEPTH,
)


def _solver_tuning(
    monkeypatch,
    *,
    substeps: str | None = None,
    vbd_iters: str | None = None,
    coupling_iters: str | None = None,
) -> tuple[int, int, int]:
    if substeps is None:
        monkeypatch.delenv("WATERHOSE_SUBSTEPS", raising=False)
    else:
        monkeypatch.setenv("WATERHOSE_SUBSTEPS", substeps)
    if vbd_iters is None:
        monkeypatch.delenv("WATERHOSE_VBD_ITERS", raising=False)
    else:
        monkeypatch.setenv("WATERHOSE_VBD_ITERS", vbd_iters)
    if coupling_iters is None:
        monkeypatch.delenv("WATERHOSE_COUPLING_ITERS", raising=False)
    else:
        monkeypatch.setenv("WATERHOSE_COUPLING_ITERS", coupling_iters)

    cfg = waterhose_env_cfg.WaterhoseEnvCfg()
    entries = {entry.name: entry for entry in cfg.sim.physics.solver_cfg.entries}
    return (
        cfg.sim.physics.num_substeps,
        entries["vbd"].solver_cfg.iterations,
        cfg.sim.physics.solver_cfg.iterations,
    )


def test_waterhose_uses_high_fidelity_solver_tuning_defaults(monkeypatch):
    assert _solver_tuning(monkeypatch) == (10, 20, 1)


def test_waterhose_refreshes_outer_contacts_each_solver_substep():
    """MJWarp must not reuse a stale robot/fridge manifold for the whole 10 ms step."""
    cfg = waterhose_env_cfg.WaterhoseEnvCfg()

    assert cfg.sim.physics.num_substeps > 1
    assert cfg.sim.physics.collision_decimation == 1


@pytest.mark.parametrize(
    ("asset_path", "expected_shape_count"),
    [
        (waterhose_env_cfg._RBY1_USD, 4),
        (waterhose_env_cfg._FRIDGE_ROBOT_COLLISION_PROXY_USD, 1),
    ],
    ids=["rby1-fingers", "fridge-robot-housing"],
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


def test_waterhose_proxy_contacts_warm_start_the_grip():
    cfg = waterhose_env_cfg.WaterhoseEnvCfg()
    entries = {entry.name: entry for entry in cfg.sim.physics.solver_cfg.entries}

    assert cfg.sim.physics.use_cuda_graph is True
    assert entries["vbd"].solver_cfg.rigid_contact_history is True
    assert entries["vbd"].solver_cfg.rigid_contact_hard is True
    assert entries["vbd"].solver_cfg.rigid_avbd_beta == pytest.approx(1.0e2)
    assert entries["vbd"].solver_cfg.rigid_contact_k_start == pytest.approx(1.0e3)
    assert entries["vbd"].solver_cfg.friction_epsilon == pytest.approx(0.1)
    assert entries["vbd"].solver_cfg.rigid_joint_linear_ke == pytest.approx(1.0e9)
    assert entries["vbd"].solver_cfg.rigid_joint_angular_ke == pytest.approx(1.0e9)
    assert entries["vbd"].solver_cfg.rigid_joint_linear_k_start == pytest.approx(1.0e4)
    assert entries["vbd"].solver_cfg.rigid_joint_angular_k_start == pytest.approx(1.0e1)
    assert cfg.sim.physics.collision_cfg.rigid_contact_max == waterhose_env_cfg._DEFAULT_RIGID_CONTACT_MAX
    assert cfg.sim.physics.collision_cfg.rigid_contact_max >= waterhose_env_cfg._PROXY_RIGID_CONTACT_MIN


def test_waterhose_rigid_contact_capacity_can_scale_for_larger_batches(monkeypatch):
    monkeypatch.setenv("WATERHOSE_RIGID_CONTACT_MAX", "262144")

    cfg = waterhose_env_cfg.WaterhoseEnvCfg()

    assert cfg.sim.physics.collision_cfg.rigid_contact_max == 262_144


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
            "/World/envs/env_0/Cable1/curve_0_edge_capsule_0",
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
                (5, 7),
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
    np.testing.assert_allclose(
        model.shape_material_mu.numpy(),
        [7.0, 20.0, 20.0, 20.0, 20.0, 0.5, 0.5, 0.5, 0.5],
    )
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
                (5, 7),
                (6, 7),
                (5, 8),
                (6, 8),
            ],
            dtype=np.int32,
        ),
    )
    assert "kept 12 contact pair(s), dropped 10 finger-vs-unrelated pair(s)" in capsys.readouterr().out
    assert pipeline.kwargs["contact_matching"] == "latest"
    assert pipeline.kwargs["contact_matching_pos_threshold"] == pytest.approx(0.005)
    assert pipeline.kwargs["contact_matching_normal_dot_threshold"] == pytest.approx(0.95)
    assert pipeline.kwargs["rigid_contact_max"] == waterhose_env_cfg._PROXY_RIGID_CONTACT_MIN


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
    assert entries["vbd"].include_body_shapes is True
    assert entries["vbd"].include_static_shapes is False
    assert entries["vbd"].shape_label_patterns == [
        SOCKET_COLLISION_MESH_PATTERN,
        waterhose_env_cfg._FRIDGE_CABLE_COLLISION_PATTERN,
    ]

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


def test_waterhose_solver_tuning_accepts_environment_overrides(monkeypatch):
    assert _solver_tuning(monkeypatch, substeps="6", vbd_iters="12", coupling_iters="3") == (6, 12, 3)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("WATERHOSE_SUBSTEPS", "not-an-int"),
        ("WATERHOSE_SUBSTEPS", "0"),
        ("WATERHOSE_VBD_ITERS", "-1"),
        ("WATERHOSE_COUPLING_ITERS", "0"),
    ],
)
def test_waterhose_solver_tuning_rejects_invalid_values(monkeypatch, name, value):
    monkeypatch.delenv("WATERHOSE_SUBSTEPS", raising=False)
    monkeypatch.delenv("WATERHOSE_VBD_ITERS", raising=False)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=rf"{name} must be a positive integer"):
        waterhose_env_cfg.WaterhoseEnvCfg()


def test_robot_asset_enables_both_grippers_finger_colliders():
    """The immutable source asset only authors the four cable-proxy finger colliders."""
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


def test_robot_spawn_enables_bilateral_fridge_collision_shells():
    """Runtime authoring enables the existing palm/flange/wrist meshes without editing the asset."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(waterhose_env_cfg._RBY1_USD)
    robot_prim = stage.GetDefaultPrim()
    instance_roots = [descendant for descendant in Usd.PrimRange(robot_prim) if descendant.IsInstanceable()]
    for instance_root in instance_roots:
        instance_root.SetInstanceable(False)

    enabled_paths = waterhose_env_cfg._enable_rby1_robot_fridge_colliders(robot_prim)

    assert len(enabled_paths) == len(waterhose_env_cfg._ROBOT_FRIDGE_COLLISION_ROOT_NAMES) == 8
    assert {
        next(name for name in waterhose_env_cfg._ROBOT_FRIDGE_COLLISION_ROOT_NAMES if f"/{name}/" in path)
        for path in enabled_paths
    } == waterhose_env_cfg._ROBOT_FRIDGE_COLLISION_ROOT_NAMES
    for mesh_path in enabled_paths:
        mesh = stage.GetPrimAtPath(mesh_path)
        assert UsdPhysics.CollisionAPI(mesh)
        assert UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Get() is True
        assert UsdPhysics.MeshCollisionAPI(mesh)
        assert UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() == "none"
        assert tuple(mesh.GetAttribute("mjc:solref").Get()) == waterhose_env_cfg._ROBOT_FRIDGE_CONTACT_SOLREF
        assert mesh.GetAttribute("newton:contactMargin").Get() == pytest.approx(0.0)
        assert mesh.GetAttribute("newton:contactGap").Get() == pytest.approx(0.01)
        assert mesh.GetAttribute("newton:sdfMaxResolution").Get() == 64
        assert mesh.GetAttribute("newton:hydroelasticEnabled").Get() is False


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


def test_scene_uses_native_cable_config_and_runtime_schema_adapter():
    scene = waterhose_env_cfg.WaterhoseSceneCfg()

    assert scene.fridge.spawn.usd_path == waterhose_env_cfg._FRIDGE_USD
    assert scene.robot.spawn.usd_path == waterhose_env_cfg._RBY1_USD
    assert scene.cable1.spawn.usd_path == waterhose_env_cfg._CABLE1_USD
    assert isinstance(scene.cable1, CableObjectCfg)
    assert isinstance(scene.cable1, WaterhoseCableObjectCfg)
    assert scene.cable1.spawn.func is spawn_waterhose_cable_from_usd
    assert scene.cable1.class_type.endswith(":WaterhoseCableObject")
    assert scene.anchor1.spawn.rigid_props is None
    assert scene.anchor1.spawn.collision_props is None
    assert not hasattr(scene, "connector_visual")
    assert waterhose_env_cfg._SOCKET_CONTACT_PROPERTIES.contact_gap == 0.0
    assert scene.cable1.connector_gap == pytest.approx(0.001)
    assert scene.cable1.tail_anchor_prim_path == "/World/envs/env_.*/Anchor1"
    assert scene.cable1.curve_prim_suffix == "/curve_0"


def test_waterhose_native_material_converts_to_validated_per_joint_stiffness():
    """Native elastic moduli preserve the prior rod stiffness in Newton's physical units."""

    from pxr import UsdGeom

    cable_cfg = waterhose_env_cfg.WaterhoseSceneCfg().cable1
    material = cable_cfg.spawn.physics_material
    cable_stage = sim_utils.create_new_stage()
    with sim_utils.use_stage(cable_stage):
        cable_cfg.spawn.func("/World/Cable1", cable_cfg.spawn)
    curve = UsdGeom.BasisCurves(cable_stage.GetPrimAtPath("/World/Cable1/curve_0"))
    points = np.asarray(curve.GetPointsAttr().Get(), dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    mean_segment_length = float(segment_lengths.mean())
    flexible_segment_length = float(segment_lengths[1:].mean())
    radius = 0.5 * material.thickness
    area = np.pi * radius**2
    area_moment = np.pi * radius**4 / 4.0
    polar_moment = 2.0 * area_moment

    # The long head segment is the rigid span inside the connector. Only the flexible tail should be
    # uniformly sampled; splitting the head would attach the plug to a body that no longer spans it.
    assert segment_lengths[0] > 5.0 * flexible_segment_length
    np.testing.assert_allclose(segment_lengths[1:], flexible_segment_length, rtol=0.0, atol=5.0e-6)
    # Resampling a curved polyline changes its total chord length by a few parts in 100,000.
    assert material.stretch_stiffness * area / mean_segment_length == pytest.approx(1.0e6, rel=5.0e-5)
    assert material.bend_stiffness * area_moment / mean_segment_length == pytest.approx(3.0e-1, rel=5.0e-5)
    assert material.twist_stiffness * polar_moment / mean_segment_length == pytest.approx(9.0e-2, rel=5.0e-5)
    assert cable_cfg.stretch_shear_damping == pytest.approx(1.0e-2)
    assert cable_cfg.bend_twist_damping == pytest.approx(2.0e-2)
    assert material.density == pytest.approx(100.0)
    assert material.thickness == pytest.approx(0.006)
    assert not hasattr(material, "stretch_damping")
    assert not hasattr(material, "bend_damping")


def test_spawn_adapter_authors_native_schema_without_modifying_the_asset():
    """The distributed legacy USD stays immutable while its stage instance becomes a native cable."""

    from pxr import Usd, UsdGeom, UsdShade

    source_stage = Usd.Stage.Open(waterhose_env_cfg._CABLE1_USD)
    source_curve_prim = source_stage.GetPrimAtPath("/cable001/curve_0")
    source_connector = source_stage.GetPrimAtPath("/cable001/cable_edge_body_0/connector")
    source_connector_world = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(source_connector))
    assert "PhysicsCurvesDeformableSimAPI" not in source_curve_prim.GetPrimTypeInfo().GetAppliedAPISchemas()

    stage = sim_utils.create_new_stage()
    cable_cfg = waterhose_env_cfg.WaterhoseSceneCfg().cable1
    root = cable_cfg.spawn.func("/World/Cable1", cable_cfg.spawn)
    curve_prim = stage.GetPrimAtPath("/World/Cable1/curve_0")
    curve = UsdGeom.BasisCurves(curve_prim)

    assert root.GetPath().pathString == "/World/Cable1"
    assert "PhysicsCurvesDeformableSimAPI" in curve_prim.GetPrimTypeInfo().GetAppliedAPISchemas()
    assert curve.GetNormalsAttr().Get() is None
    binding = UsdShade.MaterialBindingAPI(curve_prim).GetDirectBinding("physics")
    assert binding.GetMaterialPath() == "/World/Cable1/material"
    assert binding.GetMaterialPurpose() == "physics"
    material_prim = stage.GetPrimAtPath(binding.GetMaterialPath())
    material_cfg = cable_cfg.spawn.physics_material
    assert material_prim.GetAttribute("physics:thickness").Get() == pytest.approx(material_cfg.thickness)
    assert material_prim.GetAttribute("physics:density").Get() == pytest.approx(material_cfg.density)
    assert material_prim.GetAttribute("physics:stretchStiffness").Get() == pytest.approx(material_cfg.stretch_stiffness)
    assert material_prim.GetAttribute("physics:bendStiffness").Get() == pytest.approx(material_cfg.bend_stiffness)
    assert material_prim.GetAttribute("physics:twistStiffness").Get() == pytest.approx(material_cfg.twist_stiffness)

    # The adapter changes the parent from the old segment-start frame to the
    # native segment-center frame, but counter-translates the visual child.
    spawned_connector = stage.GetPrimAtPath("/World/Cable1/cable_edge_body_0/connector")
    spawned_connector_world = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(spawned_connector))
    np.testing.assert_allclose(spawned_connector_world, source_connector_world, rtol=0.0, atol=5.0e-8)

    # The connector origin must remain inside its rigid head segment. Resampling the complete curve
    # shortens this segment to about 7 mm while leaving the connector about 25 mm from its start,
    # causing the plug to overlap independently moving cable bodies and appear to detach under bend.
    points = curve.GetPointsAttr().Get()
    head_length = float((points[1] - points[0]).GetLength())
    connector_local = spawned_connector.GetAttribute("xformOp:translate").Get()
    connector_from_head_start = float(connector_local[2]) + 0.5 * head_length
    assert 0.0 < connector_from_head_start < head_length


def test_compound_connector_aligns_clear_of_the_success_region():
    cfg = waterhose_env_cfg.WaterhoseEnvCfg()
    success = cfg.terminations.success.params

    assert pytest.approx(-0.030) == SOCKET_ALIGN_TIP_DEPTH
    assert pytest.approx(-0.004) == SOCKET_SEATED_TIP_DEPTH
    assert success["min_depth"] == pytest.approx(SOCKET_RETAINED_MIN_TIP_DEPTH)
    assert success["max_depth"] == pytest.approx(SOCKET_RETAINED_MAX_TIP_DEPTH)
    assert SOCKET_ALIGN_TIP_DEPTH < success["min_depth"] < SOCKET_SEATED_TIP_DEPTH < success["max_depth"]
    assert success["radial_threshold"] == pytest.approx(SOCKET_RETAINED_RADIAL_TOLERANCE)
    assert success["alignment_threshold"] == pytest.approx(SOCKET_RETAINED_AXIS_COS)


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
    assert builder.body_count == 1
    assert len(result["path_shape_map"]) == 1
    connector_shape = next(iter(result["path_shape_map"].values()))
    vertices = np.asarray(builder.shape_source[connector_shape].vertices)
    assert pytest.approx(float(vertices[:, 2].max()), abs=1.0e-7) == CONNECTOR_TIP_LEN
    assert CONNECTOR_TIP_LOCAL_POS[2] == CONNECTOR_TIP_LEN


@pytest.mark.parametrize(
    ("proxy_path", "corridor_radius"),
    [
        (waterhose_env_cfg._FRIDGE_ROBOT_COLLISION_PROXY_USD, 0.065),
        (waterhose_env_cfg._FRIDGE_CABLE_COLLISION_PROXY_USD, 0.015),
    ],
)
def test_fridge_housing_proxy_is_a_watertight_manifold(proxy_path, corridor_radius):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(proxy_path)
    housing = UsdGeom.Mesh(stage.GetPrimAtPath("/FridgeCollision/Housing"))
    points = np.asarray(housing.GetPointsAttr().Get(), dtype=np.float64)
    faces = np.asarray(housing.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
    assert housing.GetPrim().GetAttribute("waterhose:corridorRadius").Get() == pytest.approx(corridor_radius)

    undirected_edges = np.sort(
        np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0),
        axis=1,
    )
    _, edge_counts = np.unique(undirected_edges, axis=0, return_counts=True)
    signed_volume = (
        np.einsum(
            "ij,ij->i",
            points[faces[:, 0]],
            np.cross(points[faces[:, 1]], points[faces[:, 2]]),
        ).sum()
        / 6.0
    )

    assert np.all(edge_counts == 2)
    assert signed_volume > 0.0

    builder = ModelBuilder()
    result = builder.add_usd(
        proxy_path,
        floating=False,
        load_visual_shapes=False,
    )
    housing_shape = next(iter(result["path_shape_map"].values()))
    assert builder.shape_source[housing_shape].is_watertight


def test_cable_housing_proxy_has_no_mjwarp_contact_metadata():
    from pxr import Usd

    stage = Usd.Stage.Open(waterhose_env_cfg._FRIDGE_CABLE_COLLISION_PROXY_USD)
    housing = stage.GetPrimAtPath("/FridgeCollision/Housing")

    assert not housing.HasAttribute("mjc:solref")


def test_fridge_props_are_part_of_only_the_robot_housing_proxy():
    """The rigid robot sees both fridge props while the deformable cable does not."""
    from pxr import Usd

    expected_sources = [
        "/root/Cable008_Prop001/Collisions/Cable008_Prop001_Collider1",
        "/root/Cable008_Prop002/Collisions/Cable008_Prop002_Collider1",
    ]
    fridge_stage = Usd.Stage.Open(waterhose_env_cfg._FRIDGE_USD)
    robot_stage = Usd.Stage.Open(waterhose_env_cfg._FRIDGE_ROBOT_COLLISION_PROXY_USD)
    cable_stage = Usd.Stage.Open(waterhose_env_cfg._FRIDGE_CABLE_COLLISION_PROXY_USD)
    robot_housing = robot_stage.GetPrimAtPath("/FridgeCollision/Housing")
    cable_housing = cable_stage.GetPrimAtPath("/FridgeCollision/Housing")

    for prop_name in ("Cable008_Prop001", "Cable008_Prop002"):
        prop_collision_scope = fridge_stage.GetPrimAtPath(f"/root/{prop_name}/Collisions")
        assert not prop_collision_scope.IsActive()
        assert prop_collision_scope.GetAttribute("physics:collisionEnabled").Get() is False

    robot_sources = robot_housing.GetAttribute("waterhose:postClearanceColliderSources").Get()
    cable_sources = cable_housing.GetAttribute("waterhose:postClearanceColliderSources").Get()

    assert list(robot_sources) == expected_sources
    assert list(cable_sources) == []


@pytest.fixture
def native_waterhose_builder():
    """Import one runtime-adapted cable and apply the task's scoped extension."""

    from pxr import UsdGeom

    stage = sim_utils.create_new_stage()
    scene_cfg = waterhose_env_cfg.WaterhoseSceneCfg()
    cable_cfg = scene_cfg.cable1
    cable_cfg.spawn.func("/World/envs/env_0/Cable1", cable_cfg.spawn)
    anchor = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Anchor1")
    anchor.AddTranslateOp().Set(scene_cfg.anchor1.init_state.pos)

    builder = ModelBuilder()
    solvers.SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)
    builder.begin_world()
    builder.add_usd(
        stage,
        root_path="/World/envs/env_0",
        load_visual_shapes=True,
        return_deformable_results=True,
    )
    extension = WaterhoseCableBuilderExtension(cable_cfg)
    group = extension._curve_group_index(builder, world_idx=0)
    body_start = int(builder._cable_body_start[group])
    body_end = int(builder._cable_body_end[group])
    joint_start = int(builder._cable_joint_start[group])
    joint_end = int(builder._cable_joint_end[group])
    before = SimpleNamespace(
        body_count=builder.body_count,
        joint_count=builder.joint_count,
        shape_count=builder.shape_count,
        head_mass=float(builder.body_mass[body_start]),
        head_com=np.asarray(builder.body_com[body_start], dtype=np.float64),
    )
    extension.add_to_builder(
        builder,
        world_idx=0,
        env_position=[0.0, 0.0, 0.0],
        env_rotation=(0.0, 0.0, 0.0, 1.0),
    )
    builder.end_world()
    return SimpleNamespace(
        builder=builder,
        cable_cfg=cable_cfg,
        extension=extension,
        body_start=body_start,
        body_end=body_end,
        joint_start=joint_start,
        joint_end=joint_end,
        before=before,
    )


def test_native_builder_extension_lumps_offset_connector_into_head(native_waterhose_builder):
    built = native_waterhose_builder
    builder = built.builder
    cfg = built.cable_cfg
    head_body = built.body_start
    connector_prefix = f"/World/envs/env_0/Cable1/{waterhose_env_cfg._CONNECTOR_SHAPE_TOKEN}_"
    connector_shapes = [
        shape_id for shape_id, label in enumerate(builder.shape_label) if str(label).startswith(connector_prefix)
    ]

    mesh, scale, source_xform, density, is_solid, _ = built.extension._load_connector_geometry()
    expected_base_xform = wp.transform_multiply(
        wp.transform((0.0, 0.0, -built.extension.head_segment_half_length), wp.quat_identity()),
        source_xform,
    )
    for (_, radius, half_height, center_z), shape_id in zip(
        cfg.connector_collision_primitives, connector_shapes, strict=True
    ):
        expected_xform = wp.transform_multiply(
            expected_base_xform,
            wp.transform((0.0, 0.0, center_z), wp.quat_identity()),
        )
        assert builder.shape_type[shape_id] == GeoType.CYLINDER
        np.testing.assert_allclose(np.asarray(builder.shape_scale[shape_id]), [radius, half_height, 0.0], atol=1.0e-9)
        np.testing.assert_allclose(
            np.asarray(builder.shape_transform[shape_id]), np.asarray(expected_xform), atol=1.0e-8
        )

    connector_mass, connector_com, _ = compute_inertia_shape(
        GeoType.MESH,
        scale,
        mesh,
        density,
        is_solid,
        cfg.connector_margin,
    )
    connector_com_head = np.asarray(wp.transform_point(expected_base_xform, connector_com), dtype=np.float64)
    expected_mass = built.before.head_mass + connector_mass
    expected_com = (
        built.before.head_mass * built.before.head_com + connector_mass * connector_com_head
    ) / expected_mass

    assert builder.body_count == built.before.body_count
    assert builder.joint_count == built.before.joint_count + 1
    assert builder.shape_count == built.before.shape_count + 4
    assert len(connector_shapes) == 3
    assert [builder.shape_label[shape_id] for shape_id in connector_shapes] == [
        f"{connector_prefix}{name}" for name, *_ in cfg.connector_collision_primitives
    ]
    for connector_shape in connector_shapes:
        assert builder.shape_body[connector_shape] == head_body
        assert builder.shape_collision_group[connector_shape] == -1
        assert builder.shape_material_ke[connector_shape] == pytest.approx(cfg.connector_ke)
        assert builder.shape_material_kd[connector_shape] == pytest.approx(cfg.connector_kd)
        assert builder.shape_material_mu[connector_shape] == pytest.approx(cfg.connector_mu)
        assert builder.shape_margin[connector_shape] == pytest.approx(cfg.connector_margin)
        assert builder.shape_gap[connector_shape] == pytest.approx(cfg.connector_gap)
        assert not int(builder.shape_flags[connector_shape]) & int(ShapeFlags.VISIBLE)
    dynamic_visual = builder.shape_label.index("/World/envs/env_0/Cable1/waterhose_plug_visual")
    assert builder.shape_type[dynamic_visual] == GeoType.MESH
    assert builder.shape_body[dynamic_visual] == head_body
    assert int(builder.shape_flags[dynamic_visual]) & int(ShapeFlags.VISIBLE)
    assert not int(builder.shape_flags[dynamic_visual]) & int(ShapeFlags.COLLIDE_SHAPES)
    assert not int(builder.shape_flags[dynamic_visual]) & int(ShapeFlags.COLLIDE_PARTICLES)
    np.testing.assert_allclose(
        np.asarray(builder.shape_transform[dynamic_visual]), np.asarray(expected_base_xform), atol=1.0e-8
    )
    assert connector_mass == pytest.approx(cfg.connector_mass, rel=1.0e-6)
    assert builder.body_mass[head_body] == pytest.approx(expected_mass, rel=1.0e-6)
    np.testing.assert_allclose(np.asarray(builder.body_com[head_body]), expected_com, rtol=1.0e-6, atol=1.0e-9)
    assert np.all(np.linalg.eigvalsh(np.asarray(builder.body_inertia[head_body]).reshape(3, 3)) > 0.0)

    static_visual = builder.shape_label.index("/World/envs/env_0/Cable1/cable_edge_body_0/connector/plug_mesh")
    assert not int(builder.shape_flags[static_visual]) & int(ShapeFlags.VISIBLE)
    for body_id in range(built.body_start, built.body_end):
        assert all(
            builder.shape_collision_group[shape_id] == -1
            for shape_id in builder.body_shapes[body_id]
            if shape_id != dynamic_visual
        )


def test_native_builder_extension_restores_per_dof_cable_damping(native_waterhose_builder):
    built = native_waterhose_builder
    builder = built.builder
    expected = [
        built.cable_cfg.stretch_shear_damping,
        built.cable_cfg.stretch_shear_damping,
        built.cable_cfg.bend_twist_damping,
        built.cable_cfg.bend_twist_damping,
    ]

    assert built.joint_end - built.joint_start == built.body_end - built.body_start - 1
    for joint_id in range(built.joint_start, built.joint_end):
        assert tuple(builder.joint_dof_dim[joint_id]) == (2, 2)
        dof_start = int(builder.joint_qd_start[joint_id])
        np.testing.assert_allclose(builder.joint_target_kd[dof_start : dof_start + 4], expected)


def test_native_builder_extension_authors_tail_constraint_in_matching_world(native_waterhose_builder):
    built = native_waterhose_builder
    builder = built.builder
    tail_body = built.body_end - 1
    tail_joint = builder.joint_label.index("/World/envs/env_0/Cable1/tail_attachment_w0")

    assert builder.joint_parent[tail_joint] == -1
    assert builder.joint_child[tail_joint] == tail_body
    assert builder.joint_collision_filter_parent[tail_joint] is True
    assert builder.custom_attributes["vbd:joint_is_hard"].values == {tail_joint: 0}
    assert "/World/envs/env_0/Cable1/tail_attachment_articulation" in builder.articulation_label

    parent_frame_w = builder.joint_X_p[tail_joint]
    child_frame_w = wp.transform_multiply(builder.body_q[tail_body], builder.joint_X_c[tail_joint])
    np.testing.assert_allclose(np.asarray(child_frame_w), np.asarray(parent_frame_w), rtol=0.0, atol=1.0e-7)


def test_waterhose_environment_scopes_and_binds_native_builder_extension(monkeypatch):
    """The task-local world hook exists only during scene replication."""

    class FakeCable:
        def __init__(self):
            self.bound_extension = None

        def bind_builder_extension(self, extension):
            self.bound_extension = extension

    class FakeBaseEnv:
        def _init_sim(self):
            assert len(NewtonManager._per_world_builder_hooks) == 1
            self.hook_owner_during_init = NewtonManager._per_world_builder_hooks[0].__self__
            self.scene = {"cable1": FakeCable()}

    class Harness(waterhose_env.WaterhoseCableEnvMixin, FakeBaseEnv):
        def __init__(self):
            self.cfg = SimpleNamespace(scene=SimpleNamespace(cable1=waterhose_env_cfg.WaterhoseSceneCfg().cable1))

    monkeypatch.setattr(NewtonManager, "_per_world_builder_hooks", [])
    monkeypatch.setattr(waterhose_env, "WaterhoseCableObject", FakeCable)
    harness = Harness()

    harness._init_sim()

    extension = harness._waterhose_cable_builder_extension
    assert harness.hook_owner_during_init is extension
    assert NewtonManager._per_world_builder_hooks == []
    assert harness.scene["cable1"].bound_extension is extension


def test_waterhose_reset_restores_the_full_scene_and_joint_targets():
    events = waterhose_env_cfg.EventCfg()

    assert events.reset_scene.func is waterhose_env_cfg.mdp.reset_scene_to_default
    assert events.reset_scene.params == {"reset_joint_targets": True}
    assert list(events.__dict__) == ["reset_scene"]


def test_waterhose_gripper_proxy_uses_requested_mass_scale():
    cfg = waterhose_env_cfg.WaterhoseEnvCfg()
    proxy = cfg.sim.physics.solver_cfg.proxies[0]

    assert proxy.mass_scale == pytest.approx(1.0)
    assert proxy.mass_scale == pytest.approx(waterhose_env_cfg._GRIPPER_PROXY_MASS_SCALE)
