# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Patch-lead length and CAD contact clearance at the spring-latching socket."""

import numpy as np
import pytest
import torch
import warp as wp
from newton._src.geometry.sdf_texture import TextureSDFData, texture_sample_sdf
from scipy.spatial.transform import Rotation

from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_tasks.contrib.gb300_sharpa_insertion.cable import (
    _connector_closed_mesh,
    hanging_cable_positions,
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


def test_patch_lead_preserves_length_and_both_connector_exit_datums():
    """Socket placement cannot stretch the flexible jacket or roll the seated connector."""
    start = np.array((0.52, 0.26, 2.04))
    positions = np.asarray(hanging_cable_positions(tuple(start), UPPER_SOCKET_POSITION_E))
    world = positions[:, (0, 2, 1)] * (1, -1, 1) + start
    lengths = np.linalg.norm(np.diff(world, axis=0), axis=1)
    np.testing.assert_allclose(world[:2], (start, start - (0, 0.04, 0)), atol=1e-12)
    # Independent CAD registration of the already seated upper connector.
    seated = np.array((0.3767752, 0.3041757, 2.1235837718685))
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
        ]
    )
    seat = np.asarray(head_goal_for_socket((0, 0, 0)))
    # Measured source hinge and the physical inward clip limit, not a synthetic hook.
    hinge = np.array((0, 0.004677868448, -0.030711488256))
    head_rotation = Rotation.from_euler("x", 90, degrees=True)
    compression = Rotation.from_rotvec((0.2, 0, 0))
    samples = [probe]
    for name in ("plug", "latch"):
        mesh = _connector_closed_mesh(name)
        assert mesh.is_watertight and mesh.mass > 0
        triangles = mesh.vertices[np.asarray(mesh.indices).reshape(-1, 3)]
        surface = np.vstack((mesh.vertices, triangles.mean(axis=1)))
        if name == "latch":
            surface = compression.apply(surface - hinge) + hinge
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
    assert np.all(measured[[0, 3, 4]] > 0.00015)
    assert np.all(measured[[1, 2, 5, 6, 7]] < -0.0003)
    assert measured[len(probe) :].min() > 0

    poses = torch.tensor(np.tile(np.r_[seat, head_rotation.as_quat()], (2, 1)), dtype=torch.float64)
    poses[1, :3] += torch.from_numpy(head_rotation.apply(hinge - compression.apply(hinge)))
    poses[1, 3:] = torch.from_numpy((head_rotation * compression).as_quat())
    assert torch.all(latch_engagement_margins(poses)[0] > 0)
    assert latch_release_margin(poses)[0] > 0
    assert latch_release_margin(poses)[1] < 0
    # The native barb vertices independently prove physical shoulder clearance.
    vertices = _connector_closed_mesh("latch").vertices
    barbs = vertices[np.isclose(vertices[:, 2], -0.026459, atol=1e-6)]
    pressed = head_rotation.apply(compression.apply(barbs - hinge) + hinge) + seat
    assert 0.00002 < 0.003730 - pressed[:, 2].max() < 0.0001
