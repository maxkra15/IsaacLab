# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cable reset geometry tests for the Newton slung-load task."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.utils.math import euler_xyz_from_quat, quat_apply

from isaaclab_tasks.contrib.drone_slung_load.mdp.events import ResetSlungLoadEvent, reset_drone_state_on_annulus
from isaaclab_tasks.contrib.drone_slung_load.mdp.geometry import straight_end_point, straight_segment_poses

pytestmark = pytest.mark.unit


def test_straight_segment_poses_preserve_length_spacing_and_direction():
    attach = torch.tensor([[1.0, 2.0, 3.0]])
    direction = torch.tensor([[0.6, 0.0, -0.8]])
    poses = straight_segment_poses(attach, torch.tensor([0.5]), num_segments=5, direction=direction)
    end = straight_end_point(attach, torch.tensor([0.5]), direction)

    assert poses.shape == (1, 5, 7)
    torch.testing.assert_close(torch.linalg.vector_norm(end - attach, dim=-1), torch.tensor([0.5]))
    centers = poses[..., :3]
    torch.testing.assert_close(
        torch.linalg.vector_norm(torch.diff(centers, dim=1), dim=-1),
        torch.full((1, 4), 0.1),
    )
    torch.testing.assert_close(torch.linalg.vector_norm(centers[:, 0] - attach, dim=-1), torch.tensor([0.05]))
    torch.testing.assert_close(torch.linalg.vector_norm(end - centers[:, -1], dim=-1), torch.tensor([0.05]))
    local_z = torch.tensor([[0.0, 0.0, 1.0]])
    torch.testing.assert_close(quat_apply(poses[:, 0, 3:7], local_z), direction)


def test_hanging_segment_quaternion_maps_local_positive_z_to_world_down():
    poses = straight_segment_poses(
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([0.5]),
        num_segments=4,
        direction=torch.tensor([[0.0, 0.0, -1.0]]),
    )

    torch.testing.assert_close(
        poses[..., 3:7],
        torch.tensor([1.0, 0.0, 0.0, 0.0]).expand_as(poses[..., 3:7]),
    )


def test_reset_slung_load_starts_at_authored_rest_length_without_velocity():
    segment_poses: list[torch.Tensor] = []
    segment_velocities: list[torch.Tensor] = []
    payload_poses: list[torch.Tensor] = []
    payload_velocities: list[torch.Tensor] = []
    rest_poses: list[torch.Tensor] = []
    event = object.__new__(ResetSlungLoadEvent)
    event.robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=torch.tensor([[1.0, -2.0, 3.0]])),
            root_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]])),
        )
    )
    event.cable = SimpleNamespace(
        num_segments=4,
        write_segment_pose_to_sim_index=lambda segment_pose, env_ids: segment_poses.append(segment_pose),
        write_segment_velocity_to_sim_index=lambda segment_velocity, env_ids: segment_velocities.append(
            segment_velocity
        ),
    )
    event.payload = SimpleNamespace(
        write_root_pose_to_sim_index=lambda root_pose, env_ids: payload_poses.append(root_pose),
        write_root_velocity_to_sim_index=lambda root_velocity, env_ids: payload_velocities.append(root_velocity),
    )
    event._identity_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    event._write_cable_rest_pose = lambda segment_pose, env_ids: rest_poses.append(segment_pose)
    env = SimpleNamespace(num_envs=1, device="cpu")

    event(
        env,
        torch.tensor([0]),
        cable_length=0.50,
        robot_cfg=None,
        cable_cfg=None,
        payload_cfg=None,
        attach_offset_z=0.0,
        max_initial_swing=0.0,
    )

    expected_centers = torch.tensor(
        [[1.0, -2.0, 2.9375], [1.0, -2.0, 2.8125], [1.0, -2.0, 2.6875], [1.0, -2.0, 2.5625]]
    )
    torch.testing.assert_close(segment_poses[0][0, :, :3], expected_centers)
    torch.testing.assert_close(rest_poses[0], segment_poses[0])
    torch.testing.assert_close(payload_poses[0][0, :3], torch.tensor([1.0, -2.0, 2.5]))
    torch.testing.assert_close(segment_velocities[0], torch.zeros_like(segment_velocities[0]))
    torch.testing.assert_close(payload_velocities[0], torch.zeros_like(payload_velocities[0]))


def test_annulus_reset_is_local_stationary_and_matches_ellipse_anchor_contract():
    poses: list[torch.Tensor] = []
    velocities: list[torch.Tensor] = []
    asset = SimpleNamespace(
        write_root_pose_to_sim_index=lambda root_pose, env_ids: poses.append(root_pose),
        write_root_velocity_to_sim_index=lambda root_velocity, env_ids: velocities.append(root_velocity),
    )
    env = SimpleNamespace(
        device="cpu",
        scene=SimpleNamespace(
            env_origins=torch.tensor([[0.0, 0.0, 0.0], [10.0, -3.0, 1.0], [-2.0, 4.0, 0.5]]),
            __getitem__=lambda _self, _name: asset,
        ),
    )
    # SimpleNamespace does not dispatch special methods supplied as attributes.
    env.scene = type(
        "SceneStub",
        (),
        {
            "env_origins": env.scene.env_origins,
            "__getitem__": lambda _self, _name: asset,
        },
    )()
    env_ids = torch.tensor([0, 2])

    torch.manual_seed(20260817)
    reset_drone_state_on_annulus(
        env,
        env_ids,
        radius_range=(4.3, 4.8),
        height=0.0,
        roll_range=(-0.05, 0.05),
        pitch_range=(-0.05, 0.05),
        yaw=0.0,
    )

    pose_e = poses[0][:, :3] - env.scene.env_origins[env_ids]
    radius = torch.linalg.vector_norm(pose_e[:, :2], dim=-1)
    assert torch.all((radius >= 4.3) & (radius <= 4.8))
    torch.testing.assert_close(pose_e[:, 2], torch.zeros(2))
    torch.testing.assert_close(velocities[0], torch.zeros(2, 6))
    roll, pitch, yaw = euler_xyz_from_quat(poses[0][:, 3:7])
    assert torch.all(torch.abs(roll) <= 0.05 + 1.0e-6)
    assert torch.all(torch.abs(pitch) <= 0.05 + 1.0e-6)
    torch.testing.assert_close(yaw, torch.zeros_like(yaw), atol=1.0e-6, rtol=0.0)


@pytest.mark.parametrize("radius_range", ((0.0, 4.8), (4.8, 4.3), (4.3, float("inf"))))
def test_annulus_reset_rejects_invalid_radius_range(radius_range):
    env = SimpleNamespace(device="cpu", scene={"robot": SimpleNamespace()})

    with pytest.raises(ValueError, match="radius_range"):
        reset_drone_state_on_annulus(env, torch.tensor([0]), radius_range=radius_range)


def test_straight_geometry_rejects_zero_segments():
    with pytest.raises(ValueError, match="num_segments"):
        straight_segment_poses(
            torch.zeros(1, 3),
            torch.ones(1),
            num_segments=0,
            direction=torch.tensor([[0.0, 0.0, -1.0]]),
        )
