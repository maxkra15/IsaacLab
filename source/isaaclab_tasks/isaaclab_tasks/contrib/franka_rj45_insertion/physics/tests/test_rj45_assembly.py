# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Non-simulation contract tests for the Newton RJ45 assembly."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import newton
import numpy as np
import pytest
import warp as wp

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab_tasks.contrib.franka_rj45_insertion.dual_rack_workcell import (
    DUAL_RACK_CABLE_CONTACT_DAMPING_N_S_M,
    DUAL_RACK_CABLE_TABLE_CENTERLINE_HEIGHT_M,
    DUAL_RACK_TARGET_TASK_TRANSLATION,
    DUAL_RACK_WORKCELL_CFG,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.gb300_workcell import (
    GB300_DEFAULT_TARGET_TASK_TRANSLATION,
    GB300_WORKCELL_CFG,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.physics.rj45_assembly import (
    CABLE_EXTENSION_SEGMENT_COUNT,
    CABLE_KINEMATIC_COUNT,
    CABLE_RADIUS,
    CABLE_SEGMENT_COUNT,
    GRASP_FRICTION,
    INSERTION_DISTANCE,
    RJ45_ASSET_PATH,
    RJ45_DEFAULT_TOPOLOGY,
    RJ45_PICK_INSERT_PLUG_PASSIVE_ANGULAR_DAMPING_RATE,
    RJ45_PICK_INSERT_TOPOLOGY,
    RJ45_PRESENTATION_SWITCH_USD_PATH,
    RJ45_VBD_CONTACT_BUFFER_SIZE,
    RJ45_VBD_ITERATIONS,
    SUPPORT_PLANE_CONTACT_KD,
    SUPPORT_PLANE_CONTACT_KE,
    SUPPORT_PLANE_MU,
    TASK_BODY_COUNT,
    Rj45AssemblyTopologyCfg,
    Rj45NewtonAssemblyBuilder,
    configure_franka_finger_contact_material,
    make_rj45_task_layout,
    resolve_franka_grasp_shape_ids,
    rj45_reset_physics_contract,
    validate_rj45_vbd_solver_cfg,
    verify_rj45_asset,
)


def test_canonical_newton_asset_is_unmodified() -> None:
    """Verify the task-local USD is the pinned Newton source asset."""
    verify_rj45_asset()
    assert RJ45_ASSET_PATH.stat().st_size == 145_138


def test_reset_physics_contract_covers_grasp_and_cable_topology() -> None:
    """Expose serialization-stable reset-invalidating physics values."""
    contract = rj45_reset_physics_contract()
    assert contract["task_body_count"] == TASK_BODY_COUNT
    assert contract["cable_segment_count"] == CABLE_SEGMENT_COUNT
    assert contract["grasp_friction"] == GRASP_FRICTION
    assert contract["grasp_collision_policy"] == "finger-proxy-plus-visible-plug-and-cable"
    assert contract["support_plane_local_height"] == 0.0
    assert contract["support_plane_friction"] == SUPPORT_PLANE_MU
    assert contract["vbd_rigid_compliant_alm"] is True
    assert contract["vbd_legacy_rigid_contact_hard"] is False


def test_pick_insert_layout_and_physics_contract_are_explicit() -> None:
    """Keep the legacy layout exact while exposing the opt-in full-task topology."""
    default = make_rj45_task_layout()
    pick_insert = make_rj45_task_layout(RJ45_PICK_INSERT_TOPOLOGY)

    assert Rj45AssemblyTopologyCfg() == RJ45_DEFAULT_TOPOLOGY
    assert default.body_count == TASK_BODY_COUNT
    assert default.socket_body_index is None
    assert default.plug_body_index == 0
    assert default.latch_body_index == 1
    assert default.cable_body_slice == slice(2, 37)
    assert default.cable_sample_body_indices == (2, 6, 11, 16, 21, 28, 36)
    assert pick_insert.body_count == 1 + 2 + CABLE_SEGMENT_COUNT + CABLE_EXTENSION_SEGMENT_COUNT
    assert pick_insert.body_names[:3] == ("socket", "plug", "latch")
    assert pick_insert.socket_body_index == 0
    assert pick_insert.plug_body_index == 1
    assert pick_insert.latch_body_index == 2
    assert pick_insert.cable_body_slice == slice(3, 48)
    assert pick_insert.cable_sample_body_indices == (3, 10, 18, 25, 32, 40, 47)

    legacy_contract = rj45_reset_physics_contract()
    variant_contract = rj45_reset_physics_contract(RJ45_PICK_INSERT_TOPOLOGY)
    assert legacy_contract["contract_version"] == 1
    assert "task_layout" not in legacy_contract
    assert "goal_orientation_hold_stiffness" not in legacy_contract
    assert "plug_root_passive_angular_damping_rate_s_inv" not in legacy_contract
    assert RJ45_DEFAULT_TOPOLOGY.plug_passive_angular_damping_rate == 0.0
    assert variant_contract["contract_version"] == 5
    assert variant_contract["task_body_count"] == pick_insert.body_count
    assert variant_contract["task_body_order"] == pick_insert.body_names
    assert variant_contract["socket_body_mode"] == "zero-mass-resettable"
    assert variant_contract["plug_root_angular_dofs"] == 3
    assert variant_contract["task_support_plane_enabled"] is False
    assert variant_contract["cable_extension_segment_count"] == CABLE_EXTENSION_SEGMENT_COUNT
    assert variant_contract["goal_orientation_hold_default_enabled"] is False
    assert variant_contract["goal_orientation_hold_requires_translation_drive"] is True
    assert variant_contract["goal_orientation_hold_target"] == "authored-plug-orientation-per-world"
    assert variant_contract["plug_root_passive_angular_damping_rate_s_inv"] == pytest.approx(
        RJ45_PICK_INSERT_PLUG_PASSIVE_ANGULAR_DAMPING_RATE
    )
    assert variant_contract["plug_root_angular_damping_model"] == ("body-torque=-rate*inertia*angular-velocity")
    assert variant_contract["cable_alignment_timing"] == "post-solver-pre-swap-and-collision"
    assert variant_contract["cable_alignment_segment_range_half_open"] == (3, 44)
    assert variant_contract["cable_coupled_input_pose_policy"] == "preserve-q-qd-history-no-delta-or-rewind"
    assert variant_contract["cable_preserved_input_segment_range_half_open"] == (0, 44)
    assert variant_contract["cable_pinned_segment_index"] == 44


@pytest.mark.parametrize("value", (-1, 1.5, True))
def test_topology_rejects_invalid_extra_cable_segment_counts(value: object) -> None:
    with pytest.raises(ValueError, match="extra_cable_segments"):
        Rj45AssemblyTopologyCfg(extra_cable_segments=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", (-1.0, float("nan"), True))
def test_topology_rejects_invalid_passive_angular_damping(value: object) -> None:
    with pytest.raises(ValueError, match="plug_passive_angular_damping_rate"):
        Rj45AssemblyTopologyCfg(  # type: ignore[arg-type]
            free_plug_rotation=True,
            plug_passive_angular_damping_rate=value,
        )

    if isinstance(value, float) and value < 0.0:
        with pytest.raises(ValueError, match="requires free_plug_rotation"):
            Rj45AssemblyTopologyCfg(plug_passive_angular_damping_rate=-value)


def test_compliant_alm_solver_contract() -> None:
    """Accept the reference VBD settings and reject legacy constraint mode."""
    cfg = SimpleNamespace(
        iterations=RJ45_VBD_ITERATIONS,
        rigid_compliant_alm=True,
        rigid_contact_hard=False,
        integrate_with_external_rigid_solver=False,
        rigid_body_contact_buffer_size=RJ45_VBD_CONTACT_BUFFER_SIZE,
    )
    validate_rj45_vbd_solver_cfg(cfg)
    cfg.rigid_compliant_alm = None
    with pytest.raises(ValueError, match="rigid_compliant_alm must be True"):
        validate_rj45_vbd_solver_cfg(cfg)


def test_franka_finger_material_helper_changes_only_friction() -> None:
    """Tune both imported finger colliders without changing contact gains."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    hand_body = builder.add_link(label="/World/envs/env_0/Robot/panda_hand")
    hand_shape = builder.add_shape_box(hand_body, hx=0.01, hy=0.01, hz=0.01)
    finger_shapes = []
    for name in ("panda_leftfinger", "panda_rightfinger"):
        body_id = builder.add_link(label=f"/World/envs/env_0/Robot/{name}")
        finger_shapes.append(builder.add_shape_box(body_id, hx=0.01, hy=0.01, hz=0.01))
    builder.end_world()
    ke_before = tuple(builder.shape_material_ke[shape_id] for shape_id in finger_shapes)
    kd_before = tuple(builder.shape_material_kd[shape_id] for shape_id in finger_shapes)
    hand_mu_before = builder.shape_material_mu[hand_shape]

    configured = configure_franka_finger_contact_material(builder, 0)
    resolved = resolve_franka_grasp_shape_ids(builder, 0)

    assert configured == tuple(finger_shapes)
    assert resolved.hand_shape_ids == (hand_shape,)
    assert resolved.finger_shape_ids == tuple(finger_shapes)
    assert all(builder.shape_material_mu[shape_id] == pytest.approx(GRASP_FRICTION) for shape_id in configured)
    assert builder.shape_material_mu[hand_shape] == hand_mu_before
    assert tuple(builder.shape_material_ke[shape_id] for shape_id in configured) == ke_before
    assert tuple(builder.shape_material_kd[shape_id] for shape_id in configured) == kd_before


def test_builder_grasp_proxy_friction_override_is_explicit_and_validated() -> None:
    """Keep the default exact while allowing one proxy-only diagnostic value."""
    assert Rj45NewtonAssemblyBuilder().grasp_proxy_friction == GRASP_FRICTION
    assert Rj45NewtonAssemblyBuilder(grasp_proxy_friction=4.5).grasp_proxy_friction == 4.5
    for invalid in (-1.0, float("nan"), True):
        with pytest.raises(ValueError, match="grasp_proxy_friction"):
            Rj45NewtonAssemblyBuilder(grasp_proxy_friction=invalid)  # type: ignore[arg-type]


def test_workcell_builder_allows_only_cable_length_topology_override() -> None:
    """A hanging workcell may lengthen its cable without changing task semantics."""
    hanging_topology = dataclasses.replace(RJ45_PICK_INSERT_TOPOLOGY, extra_cable_segments=170)
    assembly = Rj45NewtonAssemblyBuilder(
        topology_cfg=hanging_topology,
        workcell_cfg=GB300_WORKCELL_CFG,
    )
    assert assembly.topology_cfg == hanging_topology

    incompatible = dataclasses.replace(
        hanging_topology,
        free_plug_rotation=False,
        plug_passive_angular_damping_rate=0.0,
    )
    with pytest.raises(ValueError, match="pick-insert semantics"):
        Rj45NewtonAssemblyBuilder(topology_cfg=incompatible, workcell_cfg=GB300_WORKCELL_CFG)


def test_builder_topology_and_runtime_layout() -> None:
    """Build the exact connector/cable topology and bind its stable reset layout."""
    if wp.get_cuda_device_count() == 0:
        pytest.skip("RJ45 texture SDF finalization requires CUDA.")

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    assembly = Rj45NewtonAssemblyBuilder()
    builder.begin_world()
    robot_shapes = {}
    for name in ("panda_link1", "panda_hand", "panda_leftfinger", "panda_rightfinger"):
        body_id = builder.add_link(label=f"/World/envs/env_0/Robot/{name}")
        robot_shapes[name] = builder.add_shape_box(body_id, hx=0.003, hy=0.01, hz=0.005)
    assembly.world_hook(builder, 0, [0.55, 0.0, 0.25], [0.0, 0.0, 0.0, 1.0])
    builder.end_world()

    (ids,) = assembly.world_body_ids
    assert len(ids.cable_body_ids) == CABLE_SEGMENT_COUNT
    assert len(ids.cable_joint_ids) == CABLE_SEGMENT_COUNT - 1
    assert len(ids.task_body_ids) == TASK_BODY_COUNT
    assert ids.cable_anchor_body_ids == ids.cable_body_ids[:CABLE_KINEMATIC_COUNT]
    assert ids.pinned_cable_body_id == ids.cable_body_ids[-1]
    assert ids.pinned_cable_body_ids == ids.cable_body_ids[-1:]
    assert builder.body_label[ids.plug_body_id] == "/World/envs/env_0/Rj45Assembly/Plug"
    assert builder.shape_label[ids.support_plane_shape_id].endswith("/SupportPlane/Collision")
    assert tuple(builder.shape_transform[ids.support_plane_shape_id])[:3] == pytest.approx((0.55, 0.0, 0.25))
    assert builder.shape_material_ke[ids.support_plane_shape_id] == pytest.approx(SUPPORT_PLANE_CONTACT_KE)
    assert builder.shape_material_kd[ids.support_plane_shape_id] == pytest.approx(SUPPORT_PLANE_CONTACT_KD)
    assert builder.shape_material_mu[ids.support_plane_shape_id] == pytest.approx(SUPPORT_PLANE_MU)
    assert builder.shape_label[ids.grasp_proxy_shape_id].endswith("/Plug/GraspProxy")
    assert int(builder.shape_body[ids.grasp_proxy_shape_id]) == ids.plug_body_id
    assert not int(builder.shape_flags[ids.grasp_proxy_shape_id]) & int(newton.ShapeFlags.VISIBLE)
    assert builder.shape_material_mu[ids.grasp_proxy_shape_id] == pytest.approx(GRASP_FRICTION)
    assert builder.body_label[ids.cable_body_ids[-1]].endswith("/Cable/Segment_34")
    cable_visual_path = "/World/envs/env_0/Rj45Assembly/Cable/geometry/mesh"
    for segment_id, body_id in enumerate(ids.cable_body_ids):
        (shape_id,) = builder.body_shapes[body_id]
        assert builder.shape_label[shape_id] == f"{cable_visual_path}_edge_capsule_{segment_id}"
    assert int(builder.joint_type[ids.d6_joint_id]) == int(newton.JointType.D6)
    assert int(builder.joint_type[ids.latch_joint_id]) == int(newton.JointType.REVOLUTE)
    for body_id in (*ids.cable_anchor_body_ids, ids.pinned_cable_body_id):
        assert builder.body_mass[body_id] == 0.0
    filtered_pairs = {tuple(sorted(pair)) for pair in builder.shape_collision_filter_pairs}
    assert tuple(sorted((ids.grasp_proxy_shape_id, ids.support_plane_shape_id))) in filtered_pairs
    assert tuple(sorted((ids.grasp_proxy_shape_id, ids.socket_shape_id))) in filtered_pairs
    assert tuple(sorted((ids.grasp_proxy_shape_id, ids.latch_shape_id))) in filtered_pairs
    for body_id in ids.cable_body_ids:
        cable_shape_id = builder.body_shapes[body_id][0]
        assert tuple(sorted((ids.grasp_proxy_shape_id, cable_shape_id))) in filtered_pairs
        assert tuple(sorted((robot_shapes["panda_hand"], cable_shape_id))) in filtered_pairs
        for finger_name in ("panda_leftfinger", "panda_rightfinger"):
            assert tuple(sorted((robot_shapes[finger_name], cable_shape_id))) not in filtered_pairs
    official_connector_shapes = (ids.socket_shape_id, ids.plug_shape_id, ids.latch_shape_id)
    for robot_shape_id in robot_shapes.values():
        assert tuple(sorted((robot_shape_id, ids.support_plane_shape_id))) in filtered_pairs
    for connector_shape_id in official_connector_shapes:
        assert tuple(sorted((robot_shapes["panda_hand"], connector_shape_id))) in filtered_pairs
    for finger_name in ("panda_leftfinger", "panda_rightfinger"):
        robot_shape_id = robot_shapes[finger_name]
        assert tuple(sorted((robot_shape_id, ids.socket_shape_id))) in filtered_pairs
        assert tuple(sorted((robot_shape_id, ids.latch_shape_id))) in filtered_pairs
        assert tuple(sorted((robot_shape_id, ids.plug_shape_id))) not in filtered_pairs
    assert tuple(sorted((ids.grasp_proxy_shape_id, robot_shapes["panda_hand"]))) in filtered_pairs
    for finger_name in ("panda_leftfinger", "panda_rightfinger"):
        assert tuple(sorted((ids.grasp_proxy_shape_id, robot_shapes[finger_name]))) not in filtered_pairs

    model = builder.finalize(device="cuda:0")
    model_counts = (model.body_count, model.shape_count, model.joint_count, model.articulation_count)
    stage = Usd.Stage.CreateInMemory()
    assembly.author_render_prims(stage)
    for body_name in ("Socket", "Plug", "Latch"):
        body_path = f"/World/envs/env_0/Rj45Assembly/{body_name}"
        visual = stage.GetPrimAtPath(f"{body_path}/geometry/mesh")
        assert visual.IsA(UsdGeom.Mesh)
        assert not visual.IsInstanceable()
        assert not visual.GetAppliedSchemas()
        assert UsdGeom.Xformable(visual).GetLocalTransformation() == Gf.Matrix4d(1.0)
        visual_mesh = UsdGeom.Mesh(visual)
        assert len(visual_mesh.GetPointsAttr().Get()) > 0
        assert len(visual_mesh.GetFaceVertexCountsAttr().Get()) > 0
        assert len(visual_mesh.GetFaceVertexIndicesAttr().Get()) > 0
        extent = visual_mesh.GetExtentAttr().Get()
        assert extent is not None
        assert all(float(extent[1][axis] - extent[0][axis]) > 0.0 for axis in range(3))
    cable_visual = UsdGeom.BasisCurves(stage.GetPrimAtPath(cable_visual_path))
    assert cable_visual
    assert cable_visual.GetCurveVertexCountsAttr().Get() == [CABLE_SEGMENT_COUNT + 1]
    assert cable_visual.GetWidthsAttr().Get() == pytest.approx([2.0 * CABLE_RADIUS])
    assert "PhysicsCurvesDeformableSimAPI" in cable_visual.GetPrim().GetPrimTypeInfo().GetAppliedAPISchemas()
    assert not stage.GetPrimAtPath("/World/envs/env_0/Rj45Assembly/Plug/GraspProxy").IsValid()
    assert not stage.GetPrimAtPath("/World/envs/env_0/Rj45Assembly/Socket/Presentation").IsValid()
    authored_layer = stage.GetRootLayer().ExportToString()
    assembly.author_render_prims(stage)
    assert stage.GetRootLayer().ExportToString() == authored_layer
    assert (model.body_count, model.shape_count, model.joint_count, model.articulation_count) == model_counts

    runtime = assembly.bind(model)
    assert runtime.task_body_ids.shape == (1, TASK_BODY_COUNT)
    assert runtime.support_plane_shape_ids.shape == (1,)
    assert runtime.default_body_q.shape == (1, TASK_BODY_COUNT)
    assert runtime.cable_body_ids.shape == (1, CABLE_SEGMENT_COUNT)
    assert runtime.grasp_proxy_shape_ids.shape == (1,)
    assert not bool(runtime.drive_enabled.numpy()[0])

    default_pose = runtime.default_body_q.numpy()[0, 0]
    goal_position = runtime.default_goal_target_w.numpy()[0]
    assert goal_position[1] - default_pose[1] == pytest.approx(INSERTION_DISTANCE, abs=1.0e-7)

    states = (model.state(), model.state())
    task_qd_values = np.zeros((1, TASK_BODY_COUNT, 6), dtype=np.float32)
    task_qd_values[0, runtime.layout.plug_body_index, 3:] = (0.2, -0.1, 0.3)
    runtime.write_state(
        states,
        runtime.default_body_q,
        wp.array(task_qd_values, dtype=wp.spatial_vector, device=model.device),
    )
    state = states[0]
    state.clear_forces()
    runtime.prepare_step(state)
    assert state.body_f.numpy()[ids.plug_body_id, 3:] == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-9)
    runtime.align_after_step(state)
    runtime.reset_to_default(state)


def test_pick_insert_builder_has_resettable_socket_free_rotation_and_extended_tail() -> None:
    """Build, bind, and write the opt-in full-task maximal-coordinate layout."""
    if wp.get_cuda_device_count() == 0:
        pytest.skip("RJ45 texture SDF finalization requires CUDA.")

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    assembly = Rj45NewtonAssemblyBuilder(topology_cfg=RJ45_PICK_INSERT_TOPOLOGY)
    builder.begin_world()
    for name in ("panda_link1", "panda_hand", "panda_leftfinger", "panda_rightfinger"):
        body_id = builder.add_link(label=f"/World/envs/env_0/Robot/{name}")
        builder.add_shape_box(body_id, hx=0.003, hy=0.01, hz=0.005)
    assembly.world_hook(builder, 0, [0.55, 0.0, 0.25], [0.0, 0.0, 0.0, 1.0])
    builder.end_world()

    (ids,) = assembly.world_body_ids
    layout = assembly.layout
    assert ids.support_plane_shape_id is None
    assert ids.socket_body_id is not None
    assert ids.task_body_ids[0] == ids.socket_body_id
    assert builder.body_mass[ids.socket_body_id] == 0.0
    assert builder.body_inv_mass[ids.socket_body_id] == 0.0
    assert len(ids.cable_body_ids) == CABLE_SEGMENT_COUNT + CABLE_EXTENSION_SEGMENT_COUNT
    assert len(ids.task_body_ids) == layout.body_count == 48
    assert builder.joint_dof_dim[ids.d6_joint_id] == (3, 3)
    assert not any("SupportPlane" in str(label) for label in builder.shape_label)

    assert assembly._geometry is not None
    cable_points = assembly._geometry.cable_points
    authored_terminal_step = cable_points[CABLE_SEGMENT_COUNT] - cable_points[CABLE_SEGMENT_COUNT - 1]
    for index in range(CABLE_SEGMENT_COUNT, len(cable_points) - 1):
        extension_step = cable_points[index + 1] - cable_points[index]
        assert tuple(extension_step) == pytest.approx(tuple(authored_terminal_step), abs=1.0e-8)

    model = builder.finalize(device="cuda:0")
    model_counts = (model.body_count, model.shape_count, model.joint_count, model.articulation_count)
    builder_labels = (tuple(builder.body_label), tuple(builder.shape_label), tuple(builder.joint_label))
    collision_filters = tuple(builder.shape_collision_filter_pairs)
    runtime = assembly.bind(model)
    assert runtime.layout == layout
    assert runtime.topology_cfg == RJ45_PICK_INSERT_TOPOLOGY
    assert runtime.task_body_ids.shape == (1, 48)
    assert runtime.cable_body_ids.shape == (1, 45)
    assert runtime.cable_preserved_input_body_ids.numpy().tolist() == list(ids.cable_body_ids[:-1])
    assert runtime.pinned_cable_tail_body_ids.numpy().tolist() == [list(ids.cable_body_ids[-1:])]
    assert runtime.socket_body_ids.numpy().tolist() == [ids.socket_body_id]
    assert runtime.support_plane_shape_ids.numpy().tolist() == [-1]
    assert runtime.default_orientation_target_w.shape == (1,)
    assert runtime.orientation_target_w.shape == (1,)
    assert runtime.orientation_hold_enabled.numpy().tolist() == [False]

    stage = Usd.Stage.CreateInMemory()
    assembly.author_render_prims(stage)
    cable_visual_path = "/World/envs/env_0/Rj45Assembly/Cable/geometry/mesh"
    cable_visual = UsdGeom.BasisCurves(stage.GetPrimAtPath(cable_visual_path))
    assert cable_visual.GetCurveVertexCountsAttr().Get() == [46]
    assert len(cable_visual.GetPointsAttr().Get()) == 46
    for segment_id, body_id in enumerate(ids.cable_body_ids):
        (shape_id,) = builder.body_shapes[body_id]
        assert builder.shape_label[shape_id] == f"{cable_visual_path}_edge_capsule_{segment_id}"

    socket_path = "/World/envs/env_0/Rj45Assembly/Socket"
    presentation_path = f"{socket_path}/Presentation"
    presentation = stage.GetPrimAtPath(presentation_path)
    assert presentation.IsA(UsdGeom.Xform)
    for prim in Usd.PrimRange(presentation):
        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert not prim.HasAPI(UsdPhysics.CollisionAPI)
        assert not prim.HasAPI(UsdPhysics.MassAPI)
        assert not prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        assert not prim.IsA(UsdPhysics.Joint)
        if prim.IsA(UsdGeom.Xformable):
            assert not UsdGeom.Xformable(prim).GetResetXformStack()

    active_socket_visual = stage.GetPrimAtPath(f"{socket_path}/geometry/mesh")
    assert active_socket_visual.IsA(UsdGeom.Mesh)
    assert builder.body_label[ids.socket_body_id] == socket_path
    assert builder.shape_label[ids.socket_shape_id] == f"{socket_path}/Collision"
    assert (model.body_count, model.shape_count, model.joint_count, model.articulation_count) == model_counts
    assert (tuple(builder.body_label), tuple(builder.shape_label), tuple(builder.joint_label)) == builder_labels
    assert tuple(builder.shape_collision_filter_pairs) == collision_filters
    assert not any("Presentation" in str(label) for label in (*builder.body_label, *builder.shape_label))

    network_switch = stage.GetPrimAtPath(f"{presentation_path}/NetworkSwitch")
    assert network_switch.IsA(UsdGeom.Xform)
    assert not network_switch.IsInstanceable()
    references = network_switch.GetMetadata("references").prependedItems
    assert len(references) == 1
    assert references[0].assetPath == RJ45_PRESENTATION_SWITCH_USD_PATH
    assert references[0].primPath == "/AS4610"
    (switch_matrix_op,) = UsdGeom.Xformable(network_switch).GetOrderedXformOps()
    switch_matrix = switch_matrix_op.Get()
    assert tuple(switch_matrix.Transform(Gf.Vec3d(-0.3902528441, 21.82628255, 2.4888706))) == pytest.approx(
        (0.0, 0.0, 0.0), abs=1.0e-8
    )
    assert tuple(switch_matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))) == pytest.approx((0.0, -0.01, 0.0), abs=1.0e-8)
    authored_layer = stage.GetRootLayer().ExportToString()
    assembly.author_render_prims(stage)
    assert stage.GetRootLayer().ExportToString() == authored_layer

    xform_cache = UsdGeom.XformCache()
    switch_world_before = xform_cache.GetLocalToWorldTransform(network_switch).ExtractTranslation()
    socket_xform = UsdGeom.Xformable(stage.GetPrimAtPath(socket_path))
    (socket_matrix_op,) = socket_xform.GetOrderedXformOps()
    socket_matrix = socket_matrix_op.Get()
    socket_matrix.SetTranslateOnly(socket_matrix.ExtractTranslation() + Gf.Vec3d(0.1, -0.2, 0.3))
    socket_matrix_op.Set(socket_matrix)
    xform_cache.Clear()
    switch_world_after = xform_cache.GetLocalToWorldTransform(network_switch).ExtractTranslation()
    assert tuple(switch_world_after - switch_world_before) == pytest.approx((0.1, -0.2, 0.3))

    body_q_values = runtime.default_body_q.numpy()
    body_q_values[0, layout.socket_body_index, 0] += 0.03
    body_q_values[0, layout.plug_body_index, 3:7] = (0.0, 0.0, 2.0**-0.5, 2.0**-0.5)
    body_q = wp.array(body_q_values, dtype=wp.transform, device=model.device)
    body_qd = wp.zeros((1, layout.body_count), dtype=wp.spatial_vector, device=model.device)
    states = (model.state(), model.state())
    runtime.write_state(states, body_q, body_qd)

    for state in states:
        state_values = state.body_q.numpy()
        assert state_values[ids.socket_body_id, 0] == pytest.approx(body_q_values[0, layout.socket_body_index, 0])
        assert state_values[ids.plug_body_id, 3:7] == pytest.approx(body_q_values[0, layout.plug_body_index, 3:7])

    state = states[0]
    runtime.set_orientation_hold_enabled(True)
    state.clear_forces()
    runtime.prepare_step(state)
    assert np.linalg.norm(state.body_f.numpy()[ids.plug_body_id, 3:]) == pytest.approx(0.0, abs=1.0e-9)

    runtime.set_drive_enabled(True)
    runtime.set_orientation_hold_enabled(False)
    state.clear_forces()
    runtime.prepare_step(state)
    assert np.linalg.norm(state.body_f.numpy()[ids.plug_body_id, 3:]) == pytest.approx(0.0, abs=1.0e-9)

    runtime.set_orientation_hold_enabled(True)
    state.clear_forces()
    runtime.prepare_step(state)
    restoring_torque = state.body_f.numpy()[ids.plug_body_id, 3:]
    assert np.linalg.norm(restoring_torque) > 0.0
    assert restoring_torque[2] < 0.0

    runtime.set_drive_enabled(False)
    assert runtime.drive_enabled.numpy().tolist() == [False]
    assert runtime.orientation_hold_enabled.numpy().tolist() == [False]
    angular_velocity_w = np.asarray((0.2, -0.1, 0.3), dtype=np.float32)
    body_qd_values = np.zeros((1, layout.body_count, 6), dtype=np.float32)
    body_qd_values[0, layout.plug_body_index, 3:] = angular_velocity_w
    runtime.write_state(
        states,
        body_q,
        wp.array(body_qd_values, dtype=wp.spatial_vector, device=model.device),
    )
    state.clear_forces()
    runtime.prepare_step(state)
    passive_torque_w = state.body_f.numpy()[ids.plug_body_id, 3:]
    # The test plug is rotated +90 degrees about z.  Reproduce the body-space
    # inertia-scaled damper independently and rotate its torque back to world.
    world_from_body = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    inertia_body = model.body_inertia.numpy()[ids.plug_body_id]
    expected_torque_w = world_from_body @ (
        -RJ45_PICK_INSERT_PLUG_PASSIVE_ANGULAR_DAMPING_RATE * inertia_body @ (world_from_body.T @ angular_velocity_w)
    )
    assert passive_torque_w == pytest.approx(expected_torque_w, abs=1.0e-8)
    assert np.dot(passive_torque_w, angular_velocity_w) < 0.0

    runtime.set_drive_enabled(True)
    runtime.set_orientation_hold_enabled(True)
    runtime.reset_to_default(states)
    assert runtime.drive_enabled.numpy().tolist() == [False]
    assert runtime.orientation_hold_enabled.numpy().tolist() == [False]
    assert runtime.orientation_target_w.numpy()[0] == pytest.approx(runtime.default_orientation_target_w.numpy()[0])


def test_dual_rack_builder_adds_static_second_connector_and_open_workcell_shells() -> None:
    """Keep the learned 48-body ABI while adding exact source-end and cuboid contacts."""
    if wp.get_cuda_device_count() == 0:
        pytest.skip("RJ45 texture SDF finalization requires CUDA.")

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    assembly = Rj45NewtonAssemblyBuilder(
        topology_cfg=RJ45_PICK_INSERT_TOPOLOGY,
        task_translation=DUAL_RACK_TARGET_TASK_TRANSLATION,
        workcell_cfg=DUAL_RACK_WORKCELL_CFG,
    )
    builder.begin_world()
    robot_shapes = {}
    for name in ("panda_link1", "panda_hand", "panda_leftfinger", "panda_rightfinger"):
        body_id = builder.add_link(label=f"/World/envs/env_0/Robot/{name}")
        robot_shapes[name] = builder.add_shape_box(body_id, hx=0.003, hy=0.01, hz=0.005)
    assembly.world_hook(builder, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    builder.end_world()

    (ids,) = assembly.world_body_ids
    assert len(ids.task_body_ids) == 48
    assert len(ids.cable_body_ids) == 45
    anchored_shapes = (
        ids.anchored_socket_shape_id,
        ids.anchored_plug_shape_id,
        ids.anchored_latch_shape_id,
    )
    assert all(shape_id is not None for shape_id in anchored_shapes)
    assert len(ids.workcell_shape_ids) == 34
    assert all(int(builder.shape_body[shape_id]) == -1 for shape_id in (*anchored_shapes, *ids.workcell_shape_ids))
    assert builder.shape_label[ids.anchored_socket_shape_id].endswith("/AnchoredConnector/Socket/Collision")
    assert builder.shape_label[ids.anchored_plug_shape_id].endswith("/AnchoredConnector/Plug/Collision")
    assert builder.shape_label[ids.anchored_latch_shape_id].endswith("/AnchoredConnector/Latch/Collision")
    assert {builder.shape_label[shape_id].split("/Workcell/", 1)[1] for shape_id in ids.workcell_shape_ids} == {
        box.name for box in DUAL_RACK_WORKCELL_CFG.boxes if box.collidable
    }

    filtered = {tuple(sorted(pair)) for pair in builder.shape_collision_filter_pairs}
    # Rack/frame cuboids and seated source meshes remain physical for every
    # robot collider; only the invisible grasp aid is excluded from them.
    for robot_shape in robot_shapes.values():
        for fixture_shape in (*anchored_shapes, *ids.workcell_shape_ids):
            assert tuple(sorted((robot_shape, fixture_shape))) not in filtered
    for fixture_shape in (*anchored_shapes, *ids.workcell_shape_ids):
        assert tuple(sorted((ids.grasp_proxy_shape_id, fixture_shape))) in filtered
    for cable_body in ids.cable_body_ids[-CABLE_KINEMATIC_COUNT:]:
        assert builder.body_mass[cable_body] == 0.0
        assert builder.body_inv_mass[cable_body] == 0.0
        (cable_shape,) = builder.body_shapes[cable_body]
        for anchored_shape in anchored_shapes:
            assert tuple(sorted((cable_shape, anchored_shape))) in filtered
    assert all(
        builder.shape_material_kd[builder.body_shapes[body_id][0]]
        == pytest.approx(DUAL_RACK_CABLE_CONTACT_DAMPING_N_S_M)
        for body_id in ids.cable_body_ids
    )

    record = assembly._records[0]
    routed = np.asarray(record.render_cable_points_e)
    assert routed.shape == (46, 3)
    assert record.anchored_cable_endpoint_w is not None
    assert routed[-1] == pytest.approx(record.anchored_cable_endpoint_w, abs=1.0e-8)
    assert assembly._geometry is not None
    expected_lengths = np.asarray(
        [
            float(wp.length(assembly._geometry.cable_points[index + 1] - assembly._geometry.cable_points[index]))
            for index in range(45)
        ]
    )
    assert np.linalg.norm(np.diff(routed, axis=0), axis=-1) == pytest.approx(expected_lengths, abs=5.0e-8)
    assert routed[:, 2].min() >= DUAL_RACK_CABLE_TABLE_CENTERLINE_HEIGHT_M - 1.0e-9

    model = builder.finalize(device="cuda:0")
    runtime = assembly.bind(model)
    assert runtime.task_body_ids.shape == (1, 48)
    assert runtime.anchored_connector_q_w is not None
    assert runtime.anchored_connector_q_w.shape == (1, 3)
    assert runtime.anchored_cable_endpoint_w is not None
    assert runtime.anchored_cable_endpoint_w.shape == (1,)
    assert runtime.cable_segment_lengths.shape == (45,)
    assert runtime.cable_prefix_point_offsets.shape == (CABLE_KINEMATIC_COUNT + 1,)
    assert runtime.cable_prefix_rotations.shape == (CABLE_KINEMATIC_COUNT,)
    assert ids.pinned_cable_body_ids == ids.cable_body_ids[-CABLE_KINEMATIC_COUNT:]
    assert runtime.pinned_cable_tail_body_ids.numpy().tolist() == [list(ids.cable_body_ids[-CABLE_KINEMATIC_COUNT:])]
    assert runtime.cable_preserved_input_body_ids.numpy().tolist() == list(ids.cable_body_ids[:-CABLE_KINEMATIC_COUNT])
    assert runtime._align_body_ids.numpy().tolist() == [
        list(ids.cable_body_ids[CABLE_KINEMATIC_COUNT - 1 : -CABLE_KINEMATIC_COUNT])
    ]
    assert runtime._align_next_body_ids.numpy().tolist() == [
        list(ids.cable_body_ids[CABLE_KINEMATIC_COUNT : -CABLE_KINEMATIC_COUNT + 1])
    ]

    stage = Usd.Stage.CreateInMemory()
    assembly.author_render_prims(stage)
    assert stage.GetPrimAtPath(
        "/World/envs/env_0/Rj45Assembly/AnchoredConnector/Socket/Presentation/NetworkSwitch"
    ).IsA(UsdGeom.Xform)
    assert not stage.GetPrimAtPath("/World/envs/env_0/Rj45Assembly/Workcell/Frame/Posts/LeftFront").IsValid()
    assert stage.GetPrimAtPath("/World/envs/env_0/Rj45Assembly/Workcell/TSlotVisual/Frame/Posts/LeftFront/Web0").IsA(
        UsdGeom.Cube
    )
    assert stage.GetPrimAtPath(
        "/World/envs/env_0/Rj45Assembly/Workcell/TSlotVisual/Frame/RackSupports/AnchoredFront/Web1"
    ).IsA(UsdGeom.Cube)


def test_gb300_builder_keeps_inactive_port_candidates_out_of_physics() -> None:
    """Only the selected target and occupied far end may own socket SDFs."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    assembly = Rj45NewtonAssemblyBuilder(
        topology_cfg=RJ45_PICK_INSERT_TOPOLOGY,
        task_translation=GB300_DEFAULT_TARGET_TASK_TRANSLATION,
        workcell_cfg=GB300_WORKCELL_CFG,
    )
    builder.begin_world()
    for name in ("panda_link1", "panda_hand", "panda_leftfinger", "panda_rightfinger"):
        body_id = builder.add_link(label=f"/World/envs/env_0/Robot/{name}")
        builder.add_shape_box(body_id, hx=0.003, hy=0.01, hz=0.005)
    assembly.world_hook(builder, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    builder.end_world()

    (ids,) = assembly.world_body_ids
    socket_labels = tuple(label for label in builder.shape_label if str(label).endswith("/Socket/Collision"))
    assert len(socket_labels) == 2
    assert ids.socket_shape_id is not None
    assert ids.anchored_socket_shape_id is not None
    assert int(builder.shape_flags[ids.socket_shape_id]) & int(newton.ShapeFlags.COLLIDE_SHAPES)
    assert int(builder.shape_flags[ids.anchored_socket_shape_id]) & int(newton.ShapeFlags.COLLIDE_SHAPES)
    assert len(ids.workcell_shape_ids) == 6
    assert not any("CandidatePort" in str(builder.shape_label[shape_id]) for shape_id in ids.workcell_shape_ids)

    stage = Usd.Stage.CreateInMemory()
    assembly.author_render_prims(stage, include_network_switch_presentation=False)
    root = "/World/envs/env_0/Rj45Assembly"
    assert not stage.GetPrimAtPath(f"{root}/Socket/geometry/mesh").IsValid()
    assert not stage.GetPrimAtPath(f"{root}/AnchoredConnector/Socket/geometry/mesh").IsValid()
    assert stage.GetPrimAtPath(f"{root}/Plug/geometry/mesh").IsA(UsdGeom.Mesh)
    assert stage.GetPrimAtPath(f"{root}/AnchoredConnector/Plug/geometry/mesh").IsA(UsdGeom.Mesh)


def test_gb300_builder_can_partition_identical_workcell_proxies_between_solvers() -> None:
    """Give MJWarp and VBD exclusive copies of the same cabinet collision shell."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    assembly = Rj45NewtonAssemblyBuilder(
        topology_cfg=RJ45_PICK_INSERT_TOPOLOGY,
        task_translation=GB300_DEFAULT_TARGET_TASK_TRANSLATION,
        workcell_cfg=GB300_WORKCELL_CFG,
    )
    builder.begin_world()
    for name in ("panda_link1", "panda_hand", "panda_leftfinger", "panda_rightfinger"):
        body_id = builder.add_link(label=f"/World/envs/env_0/Robot/{name}")
        builder.add_shape_box(body_id, hx=0.003, hy=0.01, hz=0.005)
    assembly.world_hook(builder, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    rigid_shape_ids = assembly.add_workcell_collision_copy(
        builder,
        0,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        label_scope="RobotWorkcell",
    )
    builder.end_world()

    (ids,) = assembly.world_body_ids
    assert len(rigid_shape_ids) == len(ids.workcell_shape_ids) == 6
    assert set(rigid_shape_ids).isdisjoint(ids.workcell_shape_ids)
    assert all(int(builder.shape_body[shape_id]) == -1 for shape_id in rigid_shape_ids)
    assert all("/RobotWorkcell/" in str(builder.shape_label[shape_id]) for shape_id in rigid_shape_ids)
    assert all("/Workcell/" in str(builder.shape_label[shape_id]) for shape_id in ids.workcell_shape_ids)

    filtered = {tuple(sorted(pair)) for pair in builder.shape_collision_filter_pairs}
    cable_shape_ids = tuple(shape_id for body_id in ids.cable_body_ids for shape_id in builder.body_shapes[body_id])
    for rigid_shape_id in rigid_shape_ids:
        assert tuple(sorted((rigid_shape_id, ids.grasp_proxy_shape_id))) in filtered
        assert all(tuple(sorted((rigid_shape_id, cable_shape_id))) in filtered for cable_shape_id in cable_shape_ids)
