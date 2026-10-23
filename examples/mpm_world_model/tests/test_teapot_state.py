# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent frame, all-mass and physically bounded reference contracts."""

from types import SimpleNamespace

import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation

from examples.mpm_world_model.collect_teapot import _accumulate_rigid_capacity, _reset_split
from examples.mpm_world_model.recording import EpisodeWriter
from examples.mpm_world_model.teapot_state import (
    TeapotResidualController,
    particle_summary,
    point_in_body_frame,
    pour_reward,
    rigid_state,
    spatial_sample,
)


def test_sampling_retains_identity_and_represents_all_mass():
    """Spatial sample reconstruction weights retain unsampled particle mass."""
    position = torch.tensor([[float(i), float(i % 2), 0] for i in range(9)])
    mass = torch.arange(1, 10, dtype=torch.float32)
    indices, weights = spatial_sample(position, mass, 3, seed=4)
    assert indices.shape == (3,) and len(indices.unique()) == 3
    assert weights.sum() == 45
    other_indices, other_weights = spatial_sample(position, mass, 3, seed=4)
    torch.testing.assert_close(indices, other_indices)
    torch.testing.assert_close(weights, other_weights)


def test_full_cloud_mass_partition_and_reward():
    """Airborne and invalid mass cannot silently become delivered or spilled."""
    q = torch.tensor([[0.0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [float("nan"), 0, 5]])
    v = torch.zeros_like(q)
    mass = torch.tensor([1.0, 2, 3, 4, 5])
    source = torch.tensor([True, False, False, False, False])
    target = torch.tensor([False, True, False, False, True])
    spill = torch.tensor([False, False, True, False, True])
    summary, finite = particle_summary(q, v, mass, source, target, spill)
    torch.testing.assert_close(summary[7:12], torch.tensor([1, 2, 3, 4, 5]) / 15)
    assert not finite and torch.isfinite(summary).all()
    # Transfer four units from in-flight to the receiver; invalid five stay invalid.
    target[3] = True
    delivered, _ = particle_summary(q, v, mass, source, target, spill)
    torch.testing.assert_close(pour_reward(summary, delivered, torch.zeros(6)), torch.tensor(100 * 4 / 15))
    spill[0] = True
    source[0] = False
    spilled, _ = particle_summary(q, v, mass, source, target, spill)
    torch.testing.assert_close(pour_reward(delivered, spilled, torch.zeros(6)), torch.tensor(-200 / 15))


def test_rigid_dataset_frame_and_quaternion_convention():
    """A known quarter-turn pose is exported in WXYZ with unchanged world twists."""
    root = torch.tensor([3.0, 4, 5, 0, 0, 2**-0.5, 2**-0.5])
    twist = torch.tensor([1.0, 2, 3, 4, 5, 6])
    result = rigid_state(root, twist, torch.tensor([1.0, 2, 3]))
    torch.testing.assert_close(result, torch.tensor([2.0, 2, 2, 2**-0.5, 0, 0, 2**-0.5, 1, 2, 3, 4, 5, 6]))
    # Translate and rotate the entire held pair: grasp drift must remain zero.
    local_palm = torch.tensor([0.1, 0.2, -0.3], dtype=torch.float64)
    first_pot = torch.tensor([3.0, 4, 5, 0, 0, 0, 1], dtype=torch.float64)
    moved_pot = torch.tensor([-2.0, 1, 7, 0, 0, 2**-0.5, 2**-0.5], dtype=torch.float64)
    first_palm = torch.tensor([3.1, 4.2, 4.7], dtype=torch.float64)
    moved_palm = torch.tensor([-2.2, 1.1, 6.7], dtype=torch.float64)
    before = point_in_body_frame(first_palm, first_pot)
    after = point_in_body_frame(moved_palm, moved_pot)
    torch.testing.assert_close(before, local_palm)
    torch.testing.assert_close(after - before, torch.zeros(3, dtype=torch.float64), atol=1.0e-12, rtol=0)


def test_residual_reference_is_bounded_and_has_consistent_twist():
    """The actual continuous pose derivative agrees with the IK reference twist."""
    angular_velocity = np.array([0.3, -0.2, 0.4])

    def nominal(time):
        return (
            np.array([time, 0.0, 1]),
            Rotation.from_rotvec(angular_velocity * time).as_quat(),
            np.r_[1, 0, 0, angular_velocity],
        )

    motion = SimpleNamespace(_pose_at_time=nominal)
    residual = TeapotResidualController(motion)
    residual.apply_action(np.array([10, -10, 1, 10, -10, 1]), 0.0)
    epsilon, time = 1.0e-7, 0.004
    p, quaternion, twist = motion._pose_at_time(time)
    p_plus, q_plus, _ = motion._pose_at_time(time + epsilon)
    p_minus, q_minus, _ = motion._pose_at_time(time - epsilon)
    np.testing.assert_allclose((np.array(p_plus) - p_minus) / (2 * epsilon), twist[:3], atol=1.0e-7)
    omega = (Rotation.from_quat(q_plus) * Rotation.from_quat(q_minus).inv()).as_rotvec() / (2 * epsilon)
    np.testing.assert_allclose(omega, twist[3:], atol=1.0e-7)
    assert np.max(np.abs(np.array(p) - nominal(time)[0])) <= 0.02
    assert np.max(np.abs(residual.action)) == 1
    # Endpoints join continuously even as the next action reverses direction.
    end_pose = motion._pose_at_time(0.01)
    residual.apply_action(-np.ones(6), 0.01)
    next_pose = motion._pose_at_time(0.01)
    np.testing.assert_allclose(next_pose[0], end_pose[0], atol=1.0e-15)
    np.testing.assert_allclose(next_pose[1], end_pose[1], atol=1.0e-15)
    np.testing.assert_allclose(next_pose[2], end_pose[2], atol=1.0e-12)
    residual.reset()
    for actual, expected in zip(motion._pose_at_time(time), nominal(time)):
        np.testing.assert_allclose(actual, expected, atol=1.0e-15)
    np.testing.assert_array_equal(residual.state(), np.zeros(12))


def test_prephysics_split_selection_matches_actual_recorder(tmp_path):
    """Candidate admission predicts the recorder's labels without relabeling data."""
    writer = EpisodeWriter(tmp_path)
    state = {"observation": torch.zeros(121), "particle_state": torch.zeros(1, 6)}
    predictions = []
    for group in ("teapot-reset-0", "teapot-reset-9", "teapot-reset-29"):
        predictions.append(_reset_split(group))
        writer.start_episode(0, state, {"task_id": "teapot", "reset_group": group})
        writer.append_transition(0, torch.zeros(6), 0.0, False, True, state)
        writer.finish_episode(0)
    writer.close()
    assert {entry["split"] for entry in writer.manifest["episodes"]} == {"train", "validation", "test"}
    assert predictions == [entry["split"] for entry in writer.manifest["episodes"]]


def test_native_capacity_monitor_retains_transient_overflow_and_resets():
    """One dropped-constraint microstep remains invalid after counts fall again."""
    nefc = wp.array([180], dtype=wp.int32, device="cpu")
    ncon = wp.array([32], dtype=wp.int32, device="cpu")
    overflow = wp.array([1 << 9], dtype=wp.int32, device="cpu")
    statistics = wp.zeros(7, dtype=wp.int32, device="cpu")

    def sample():
        wp.launch(
            _accumulate_rigid_capacity, dim=1, inputs=[nefc, ncon, overflow, 128, 256, 1, statistics], device="cpu"
        )

    sample()
    nefc.assign(np.array([20], dtype=np.int32))
    ncon.assign(np.array([4], dtype=np.int32))
    overflow.zero_()
    sample()
    values = statistics.numpy()
    assert values.tolist() == [180, 32, 1 << 28, 1 << 9, 2, 20, 4]
    # Native iteration limits remain visible without becoming buffer overflow.
    statistics.zero_()
    overflow.assign(np.array([1 << 9], dtype=np.int32))
    sample()
    assert statistics.numpy()[2] == 0 and statistics.numpy()[3] == 1 << 9
    # Native capacity flags remain invalid even with small post-drop counts.
    overflow.assign(np.array([1], dtype=np.int32))
    sample()
    assert statistics.numpy()[2] == 1
    statistics.zero_()
    overflow.zero_()
    ncon.assign(np.array([256], dtype=np.int32))
    sample()
    assert statistics.numpy()[2] == 1 << 29
