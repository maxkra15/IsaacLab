# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Non-simulation contract tests for the Newton RJ45 assembly."""

from __future__ import annotations

from types import SimpleNamespace

import newton
import pytest
import warp as wp

from isaaclab_tasks.contrib.franka_rj45_insertion.physics.rj45_assembly import (
    CABLE_KINEMATIC_COUNT,
    CABLE_SEGMENT_COUNT,
    GRASP_FRICTION,
    INSERTION_DISTANCE,
    RJ45_ASSET_PATH,
    RJ45_VBD_CONTACT_BUFFER_SIZE,
    RJ45_VBD_ITERATIONS,
    SUPPORT_PLANE_CONTACT_KD,
    SUPPORT_PLANE_CONTACT_KE,
    SUPPORT_PLANE_MU,
    TASK_BODY_COUNT,
    Rj45NewtonAssemblyBuilder,
    configure_franka_finger_contact_material,
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
    assert contract["grasp_collision_policy"] == "finger-proxy-only"
    assert contract["support_plane_local_height"] == 0.0
    assert contract["support_plane_friction"] == SUPPORT_PLANE_MU
    assert contract["vbd_rigid_compliant_alm"] is True
    assert contract["vbd_legacy_rigid_contact_hard"] is False


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
        for grasp_shape_name in ("panda_hand", "panda_leftfinger", "panda_rightfinger"):
            robot_shape_id = robot_shapes[grasp_shape_name]
            assert tuple(sorted((robot_shape_id, cable_shape_id))) in filtered_pairs
    official_connector_shapes = (ids.socket_shape_id, ids.plug_shape_id, ids.latch_shape_id)
    for robot_shape_id in robot_shapes.values():
        assert tuple(sorted((robot_shape_id, ids.support_plane_shape_id))) in filtered_pairs
    for grasp_shape_name in ("panda_hand", "panda_leftfinger", "panda_rightfinger"):
        for connector_shape_id in official_connector_shapes:
            robot_shape_id = robot_shapes[grasp_shape_name]
            assert tuple(sorted((robot_shape_id, connector_shape_id))) in filtered_pairs
    assert tuple(sorted((ids.grasp_proxy_shape_id, robot_shapes["panda_hand"]))) in filtered_pairs
    for finger_name in ("panda_leftfinger", "panda_rightfinger"):
        assert tuple(sorted((ids.grasp_proxy_shape_id, robot_shapes[finger_name]))) not in filtered_pairs

    model = builder.finalize(device="cuda:0")
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

    state = model.state()
    state.clear_forces()
    runtime.prepare_step(state)
    runtime.align_after_step(state)
    runtime.reset_to_default(state)
