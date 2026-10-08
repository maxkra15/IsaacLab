# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Patch-lead length and CAD contact clearance at the spring-latching socket."""

import newton
import numpy as np
import pytest
import torch
import trimesh
import warp as wp
from isaaclab_newton.physics import VBDSolverCfg
from newton._src.geometry.sdf_texture import TextureSDFData, texture_sample_sdf
from newton._src.solvers.coupled.model_view import ModelView
from scipy.spatial.transform import Rotation

from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_tasks.contrib.gb300_sharpa_insertion.assets import LATCH_LIMIT_LOWER
from isaaclab_tasks.contrib.gb300_sharpa_insertion.cable import (
    _connector_closed_mesh,
    _pin_seated_tail,
    hanging_cable_positions,
)
from isaaclab_tasks.contrib.gb300_sharpa_insertion.endpoint import NewtonEndpointVBDManager
from isaaclab_tasks.contrib.gb300_sharpa_insertion.hydroelastic import (
    _one_way_cable_collision_pipeline,
    configure_hydroelastic_shapes,
)
from isaaclab_tasks.contrib.gb300_sharpa_insertion.socket_geometry import (
    latch_engagement_margins,
    latch_release_margin,
    local_socket_mesh,
)
from isaaclab_tasks.contrib.gb300_sharpa_insertion.workcell import (
    UPPER_SOCKET_POSITION_E,
    head_goal_for_socket,
)


@wp.kernel
def _sample_distances(sdf: TextureSDFData, points: wp.array(dtype=wp.vec3), distances: wp.array(dtype=float)):
    index = wp.tid()
    distances[index] = texture_sample_sdf(sdf, points[index])


@wp.kernel
def _sample_shape_distances(
    sdfs: wp.array(dtype=TextureSDFData),
    sdf_indices: wp.array(dtype=int),
    transforms: wp.array(dtype=wp.transform),
    scales: wp.array(dtype=wp.vec3),
    shape: int,
    points: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=float),
):
    index = wp.tid()
    local_point = wp.transform_point(wp.transform_inverse(transforms[shape]), points[index])
    local_point = wp.cw_div(local_point, scales[shape])
    distances[index] = texture_sample_sdf(sdfs[sdf_indices[shape]], local_point)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_native_pad_field_preserves_scaled_surface_mass_and_material(device):
    """Cooking grasp copies also accelerates the unchanged ordinary collision surface."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        proximal = builder.add_body()
        distal = builder.add_body()
        body = builder.add_body()
        other_finger = builder.add_body()
        proximal_shape = builder.add_shape_sphere(
            proximal, radius=0.004, xform=wp.transform((0.02, 0.039, 0.04), wp.quat_identity())
        )
        other_shape = builder.add_shape_sphere(
            other_finger, radius=0.004, xform=wp.transform((0.02, 0.039, 0.04), wp.quat_identity())
        )
        builder.add_joint_revolute(proximal, distal, collision_filter_parent=True)
        builder.add_joint_fixed(distal, body, collision_filter_parent=True)
        pad = builder.add_shape_mesh(
            body,
            mesh=newton.Mesh.create_box(0.004, 0.005, 0.006, duplicate_vertices=False),
            scale=(2.0, 3.0, 4.0),
            xform=wp.transform((0.02, 0.03, 0.04), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 2)),
            cfg=newton.ModelBuilder.ShapeConfig(density=1250.0, ke=12345.0, kd=0.7, mu=0.61, gap=0.007),
            label="/World/envs/env_0/Robot/right_index_elastomer/collision",
        )
        configure_hydroelastic_shapes(builder, ("right_index_elastomer",))
        model = builder.finalize()
    assert model._shape_sdf_index.numpy()[pad] >= 0, "Ordinary pad contacts must use a cooked distance field."
    # The rotated cuboid measures 30 x 16 x 48 mm in body axes.
    points = wp.array(
        (
            (0.02, 0.0385, 0.04),
            (0.02, 0.0375, 0.04),
            (0.0045, 0.03, 0.04),
            (0.0055, 0.03, 0.04),
            (0.02, 0.03, 0.0645),
            (0.02, 0.03, 0.0635),
        ),
        dtype=wp.vec3,
        device=device,
    )
    distances = wp.empty(len(points), dtype=float, device=device)
    wp.launch(
        _sample_shape_distances,
        dim=len(points),
        inputs=[
            model._texture_sdf_data,
            model._shape_sdf_index,
            model.shape_transform,
            model.shape_scale,
            pad,
            points,
            distances,
        ],
        device=device,
    )
    np.testing.assert_allclose(distances.numpy(), (0.0005, -0.0005) * 3, atol=2e-5)
    mass = 1250 * 0.03 * 0.016 * 0.048
    assert model.body_mass.numpy()[body] == pytest.approx(mass, rel=2e-5)
    assert model.shape_material_ke.numpy()[pad] == pytest.approx(12345.0)
    assert model.shape_material_kd.numpy()[pad] == pytest.approx(0.7)
    assert model.shape_material_mu.numpy()[pad] == pytest.approx(0.61)
    assert model.shape_gap.numpy()[pad] == pytest.approx(0.007)

    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    pairs = {
        tuple(sorted(pair))
        for pair in zip(
            contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count], strict=True
        )
    }
    assert tuple(sorted((pad, proximal_shape))) not in pairs
    assert tuple(sorted((pad, other_shape))) in pairs, "Cross-finger collisions must remain active."


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_cable_collision_scope_preserves_source_hydroelastic_and_dynamic_contacts(device):
    """A one-way cable view needs no rigid-only pressure field or source-pair mutation."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        hand = builder.add_body()
        plug = builder.add_body(xform=wp.transform((0.015, 0, 0), wp.quat_identity()))
        cable = builder.add_body(xform=wp.transform((0, 0, 0.008), wp.quat_identity()))
        hydro_cfg = newton.ModelBuilder.ShapeConfig(is_hydroelastic=True, sdf_max_resolution=32, kh=1e8)
        hand_shape = builder.add_shape_sphere(hand, radius=0.01, cfg=hydro_cfg)
        plug_shape = builder.add_shape_sphere(plug, radius=0.01, cfg=hydro_cfg)
        cable_shape = builder.add_shape_sphere(cable, radius=0.01)
        rack_shape = builder.add_shape_box(
            -1, hx=0.02, hy=0.02, hz=0.003, xform=wp.transform((0, 0, 0.018), wp.quat_identity())
        )
        ground_shape = builder.add_ground_plane()
        model = builder.finalize()
        source_pairs = model.shape_contact_pairs.numpy().copy()
        source = newton.CollisionPipeline(model)
        source_contacts = source.contacts()
        source.collide(model.state(), source_contacts)

        view = ModelView(model, "cable")
        flags = model.body_flags.numpy()
        flags[[hand, plug]] = int(newton.BodyFlags.PROXY | newton.BodyFlags.KINEMATIC)
        flags[cable] |= int(newton.BodyFlags.PROXY)
        view.body_flags = wp.array(flags, dtype=model.body_flags.dtype, device=device)
        destination = _one_way_cable_collision_pipeline(view)
        contacts = destination.contacts()
        destination.collide(model.state(), contacts)

    count = int(contacts.rigid_contact_count.numpy()[0])
    pairs = {
        tuple(sorted(pair))
        for pair in zip(
            contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count], strict=True
        )
    }
    assert (hand_shape, cable_shape) in pairs
    assert (cable_shape, rack_shape) in pairs
    assert (cable_shape, ground_shape) in pairs
    assert (hand_shape, plug_shape) not in pairs
    assert destination.hydroelastic_sdf is None, "Prescribed rigid contacts must not allocate a second pressure field."
    np.testing.assert_array_equal(model.shape_contact_pairs.numpy(), source_pairs)
    source.collide(model.state(), source_contacts)
    count = int(source_contacts.rigid_contact_count.numpy()[0])
    pairs = {
        tuple(sorted(pair))
        for pair in zip(
            source_contacts.rigid_contact_shape0.numpy()[:count],
            source_contacts.rigid_contact_shape1.numpy()[:count],
            strict=True,
        )
    }
    assert (hand_shape, plug_shape) in pairs, "The source must retain real hydroelastic grasp contacts."


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_cable_endpoint_follows_measured_plug_twist_through_captured_step_and_reset(device):
    """Prescribed plug motion survives VBD gravity and drives the cable boundary after reset."""
    quarter_turn = np.sqrt(0.5)
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
        plug = builder.add_body(
            com=(0.01, 0, 0),
            mass=0.02,
            inertia=wp.mat33(np.eye(3) * 1e-5),
            label="/World/envs/env_0/Cable/head",
        )
        endpoint = builder.add_body(
            xform=wp.transform((0.04, 0, 0), (quarter_turn, 0, 0, quarter_turn)),
            com=(0, 0.02, 0),
            is_kinematic=True,
            label="/World/envs/env_0/Cable/geometry/plug_endpoint",
        )
        tail = builder.add_body(
            xform=wp.transform((0.0, 0.25, 0.5), wp.quat_identity()),
            label="/World/envs/env_0/Cable/tail",
        )
        tail_shape = builder.add_shape_capsule(tail, radius=0.00325, half_height=0.02)
        _pin_seated_tail(builder, tail)
        builder.color()
        model = builder.finalize()
        original_flags = model.body_flags.numpy()
        proxy_flags = original_flags.copy()
        proxy_flags[plug] |= int(newton.BodyFlags.PROXY)
        view = ModelView(model, "cable")
        view.body_flags = wp.array(proxy_flags, dtype=model.body_flags.dtype)
        solver = NewtonEndpointVBDManager._create_solver(view, VBDSolverCfg(iterations=2, rigid_compliant_alm=True))
        state_in, state_out = model.state(), model.state()
        control = model.control()
        dt = 1 / 240

        def advance():
            solver.coupling_notify_input_state_update(
                state_in, newton.StateFlags.BODY_Q | newton.StateFlags.BODY_QD, dt=dt
            )
            solver.step(state_in, state_out, control, None, dt)

        advance()
        with wp.ScopedCapture(device=device) as capture:
            advance()

        poses, velocities = model.body_q.numpy(), model.body_qd.numpy()
        poses[plug] = (1, 2, 3, 0, 0, quarter_turn, quarter_turn)
        velocities[plug] = (1, 2, 3, 0, 0, 4)
        state_in.body_q.assign(poses)
        state_in.body_qd.assign(velocities)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(state_out.body_q.numpy()[endpoint], (1, 2.04, 3, 0.5, 0.5, 0.5, 0.5), atol=1e-6)
        # Endpoint COM minus plug COM is (0, 0.03, 0.02); omega cross offset is (-0.12, 0, 0).
        for state in (state_in, state_out):
            np.testing.assert_allclose(state.body_q.numpy()[plug], poses[plug], atol=1e-6)
            np.testing.assert_allclose(state.body_qd.numpy()[plug], velocities[plug], atol=1e-6)
            np.testing.assert_allclose(state.body_qd.numpy()[endpoint], (0.88, 2, 3, 0, 0, 4), atol=1e-6)
            np.testing.assert_allclose(state.body_q.numpy()[tail], (0, 0.25, 0.5, 0, 0, 0, 1), atol=1e-6)
            np.testing.assert_allclose(state.body_qd.numpy()[tail], 0, atol=1e-6)
        assert not model.shape_flags.numpy()[tail_shape] & int(newton.ShapeFlags.COLLIDE_SHAPES)

        poses[plug] = (-2, 0.5, 1, 0, 0, 0, 1)
        velocities.fill(0)
        state_in.body_q.assign(poses)
        state_in.body_qd.assign(velocities)
        solver.reset(state_in, flags=0)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(
            state_out.body_q.numpy()[endpoint], (-1.96, 0.5, 1, quarter_turn, 0, 0, quarter_turn), atol=1e-6
        )
        for state in (state_in, state_out):
            np.testing.assert_allclose(state.body_q.numpy()[plug], poses[plug], atol=1e-6)
            np.testing.assert_allclose(state.body_qd.numpy()[plug], 0, atol=1e-6)
            np.testing.assert_allclose(state.body_qd.numpy()[endpoint], 0, atol=1e-6)
            np.testing.assert_allclose(state.body_q.numpy()[tail], (0, 0.25, 0.5, 0, 0, 0, 1), atol=1e-6)
        np.testing.assert_array_equal(model.body_flags.numpy(), original_flags)


def test_patch_lead_preserves_length_and_both_connector_exit_datums():
    """Socket placement cannot stretch the flexible jacket or roll the seated connector."""
    start = np.array((0.52, 0.26, 2.04))
    positions = np.asarray(hanging_cable_positions(tuple(start), UPPER_SOCKET_POSITION_E))
    world = positions[:, (0, 2, 1)] * (1, -1, 1) + start
    lengths = np.linalg.norm(np.diff(world, axis=0), axis=1)
    np.testing.assert_allclose(world[:2], (start, start - (0, 0.04, 0)), atol=1e-12)
    # Independent CAD registration of the already seated upper connector.
    seated = np.array((0.3770352, 0.3041757, 2.1235837718685))
    np.testing.assert_allclose(world[-2:], (seated - (0, 0.02, 0), seated + (0, 0.02, 0)), atol=1e-12)
    np.testing.assert_allclose(lengths[[0, -1]], 0.04, atol=1e-12)
    assert lengths[1:-1].sum() == pytest.approx(0.24, abs=1e-9)
    assert np.all((lengths[1:-1] > 0) & (lengths[1:-1] <= 0.0101))
    with pytest.raises(ValueError, match="must exceed the distance"):
        hanging_cable_positions(tuple(start), UPPER_SOCKET_POSITION_E, flexible_length_m=0.02)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_socket_field_clears_cad_entry_and_retains_only_the_unpressed_clip(device):
    """The actual signed collision volume leaves entry clearance and solid latch shoulders."""
    with wp.ScopedDevice(device):
        socket = local_socket_mesh()
    assert socket.is_watertight
    probe = np.array(
        [
            (-0.0028, -0.0117, 0.0035),  # Below the left keeper.
            (-0.0028, -0.0117, 0.0042),  # Inside the left keeper.
            (0.0024, -0.0117, 0.0042),  # Inside the right keeper.
            (-0.0028, -0.0105, 0.0042),  # Behind the keeper.
            (0.0, -0.007, 0.0),  # Open throat.
            (0.007, -0.007, 0.0),  # Cabinet-side wall.
            (0.0, -0.007, -0.006),  # Cavity floor.
            (0.0, 0.0, 0.0),  # Electrical-contact backstop.
            (-0.0045, -0.0038, 0.0013),  # Native rear guide, left of the keyhole.
            (0.0042, -0.0038, 0.0013),  # Native rear guide, right of the keyhole.
            (0.0, -0.0038, 0.0013),  # Central nose still clears the rear keyhole.
            (0.0, -0.011, -0.00469),  # Above the measured lower entry lip.
            (0.0, -0.011, -0.00489),  # Inside its 0.2 mm raised support.
            (0.0, -0.0104, -0.00489),  # Ordinary floor behind the lip stays open.
        ]
    )
    seat = np.asarray(head_goal_for_socket((0, 0, 0)))
    # Measured source hinge and the physical inward clip limit, not a synthetic hook.
    hinge = np.array((0, 0.004677868448, -0.030711488256))
    head_rotation = Rotation.from_euler("x", 90, degrees=True)
    compression = Rotation.from_rotvec((-LATCH_LIMIT_LOWER, 0, 0))
    samples = [probe]
    meshes = {name: _connector_closed_mesh(name) for name in ("plug", "latch")}
    for name, mesh in meshes.items():
        assert mesh.is_watertight and mesh.mass > 0
        triangles = mesh.vertices[np.asarray(mesh.indices).reshape(-1, 3)]
        surface = np.vstack((mesh.vertices, triangles.mean(axis=1)))
        if name == "latch":
            # Exclude the supported hinge root; the free press beam must not enter the housing.
            beam = surface[:, 2] > -0.025
            surface = compression.apply(surface - hinge) + hinge
            # Millimeters avoid thin-triangle tolerances in the independent mesh query.
            housing = trimesh.Trimesh(
                vertices=meshes["plug"].vertices * 1000,
                faces=np.asarray(meshes["plug"].indices).reshape(-1, 3),
                process=False,
            )
            clearance = -trimesh.proximity.signed_distance(housing, surface[beam] * 1000) * 0.001
            assert clearance.min() > 0.00005
        surface = head_rotation.apply(surface) + seat
        samples.append(np.concatenate([surface + (0, depth, 0) for depth in np.linspace(-0.015, 0.0004, 78)]))
    query = wp.array(np.vstack(samples), dtype=wp.vec3, device=device)
    distances = wp.empty(len(query), dtype=float, device=device)
    wp.launch(
        _sample_distances,
        dim=len(query),
        inputs=[socket.sdf.to_texture_kernel_data(), query, distances],
        device=device,
    )
    measured = distances.numpy()
    assert np.all(measured[[0, 3, 4, 10]] > 0.00015)
    assert np.all(measured[[1, 2, 5, 6, 7, 8, 9]] < -0.0003)
    assert np.all(measured[[11, 13]] > 0.00007)
    assert measured[12] < -0.00007
    assert measured[len(probe) :].min() > 0

    poses = torch.tensor(np.tile(np.r_[seat, head_rotation.as_quat()], (2, 1)), dtype=torch.float64)
    poses[1, :3] += torch.from_numpy(head_rotation.apply(hinge - compression.apply(hinge)))
    poses[1, 3:] = torch.from_numpy((head_rotation * compression).as_quat())
    assert torch.all(latch_engagement_margins(poses)[0] > 0)
    assert latch_release_margin(poses)[0] > 0
    assert latch_release_margin(poses)[1] < 0
    # The native barb vertices independently prove physical shoulder clearance.
    vertices = meshes["latch"].vertices
    barbs = vertices[np.isclose(vertices[:, 2], -0.026459, atol=1e-6)]
    pressed = head_rotation.apply(compression.apply(barbs - hinge) + hinge) + seat
    assert 0.003730 - pressed[:, 2].max() >= 0.00012
