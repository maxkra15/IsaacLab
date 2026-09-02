# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused pure-unit coverage for FLARE observations, commands, and rewards."""

import math
from types import SimpleNamespace

import pytest
import torch

import isaaclab_tasks.contrib.drone_slung_load.mdp as mdp
import isaaclab_tasks.contrib.drone_slung_load.mdp.commands as commands_module

pytestmark = pytest.mark.unit


def _pose(position: tuple[float, float, float], quaternion: tuple[float, float, float, float]) -> torch.Tensor:
    return torch.tensor([(*position, *quaternion)], dtype=torch.float32)


def _rigid_body(position: torch.Tensor, velocity: torch.Tensor, quaternion: torch.Tensor | None = None):
    num_envs = position.shape[0]
    if quaternion is None:
        quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1)
    pose = torch.cat((position, quaternion), dim=-1)
    return SimpleNamespace(
        data=SimpleNamespace(
            body_link_pose_w=SimpleNamespace(torch=pose),
            body_com_pos_b=SimpleNamespace(torch=torch.zeros(num_envs, 1, 3)),
            body_com_vel_w=SimpleNamespace(torch=velocity),
        )
    )


def _slung_load_env(
    payload_position: torch.Tensor,
    payload_velocity: torch.Tensor,
    robot_velocity: torch.Tensor | None = None,
):
    num_envs = payload_position.shape[0]
    if robot_velocity is None:
        robot_velocity = torch.zeros(num_envs, 6)
    robot = _rigid_body(torch.zeros(num_envs, 3), robot_velocity)
    payload = _rigid_body(payload_position, payload_velocity)
    return SimpleNamespace(num_envs=num_envs, device="cpu", scene={"robot": robot, "payload": payload})


def test_attachment_kinematics_uses_actor_offset_and_com_velocity():
    pose = _pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    point_pos, point_vel = mdp.attachment_kinematics(
        pose,
        com_pos_w=torch.tensor([[0.5, 0.0, 0.0]]),
        com_vel_w=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 2.0]]),
        local_offset=(1.0, 0.0, 0.0),
    )

    torch.testing.assert_close(point_pos, torch.tensor([[1.0, 0.0, 0.0]]))
    torch.testing.assert_close(point_vel, torch.tensor([[1.0, 1.0, 0.0]]))


def test_swing_geometry_matches_flare_angle_convention():
    attachment_b = torch.tensor([[1.0, 1.0, -1.0], [0.0, 0.0, -2.0]])

    angles, total = mdp.swing_features(attachment_b)

    torch.testing.assert_close(angles[0], torch.tensor([math.pi / 4.0, math.pi / 4.0]))
    torch.testing.assert_close(total[0], torch.tensor([math.acos(1.0 / math.sqrt(3.0))]))
    torch.testing.assert_close(angles[1], torch.zeros(2))
    torch.testing.assert_close(total[1], torch.zeros(1))


def test_analytic_swing_angular_velocity_uses_point_velocity_and_rotating_body_frame():
    env = _slung_load_env(
        payload_position=torch.tensor([[0.0, 0.0, -2.0], [1.0, 0.0, -1.0]]),
        payload_velocity=torch.tensor([[2.0, 4.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 6]),
        robot_velocity=torch.tensor([[0.0] * 6, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
    )

    result = mdp.swing_angular_velocity(env)

    # First sample: direct point motion gives phi_dot=vy/L and theta_dot=vx/L.
    # Second sample: a stationary world vector moves at -omega x r in the body frame.
    torch.testing.assert_close(result, torch.tensor([[2.0, 1.0], [-1.0, 0.0]]))
    torch.testing.assert_close(mdp.swing_angular_velocity_normalized(env), result / 10.0)
    torch.testing.assert_close(
        mdp.swing_angular_velocity_normalized(env, scale=(2.0, 4.0)),
        torch.tensor([[1.0, 0.25], [-0.5, 0.0]]),
    )


def test_swing_angular_velocity_is_finite_for_degenerate_and_illegal_state():
    env = _slung_load_env(
        payload_position=torch.zeros(2, 3),
        payload_velocity=torch.tensor([[float("nan")] * 6, [float("inf")] * 6]),
        robot_velocity=torch.tensor([[0.0] * 6, [0.0, 0.0, 0.0, float("nan"), 0.0, 0.0]]),
    )

    result = mdp.swing_angular_velocity_normalized(env)

    torch.testing.assert_close(result, torch.zeros(2, 2))
    assert torch.isfinite(result).all()
    with pytest.raises(ValueError, match="finite positive"):
        mdp.swing_angular_velocity_normalized(env, scale=0.0)


def test_transverse_velocity_removes_cable_axis_component():
    result = mdp.transverse_velocity(
        relative_velocity_b=torch.tensor([[3.0, 4.0, -5.0]]),
        attachment_vector_b=torch.tensor([[0.0, 0.0, -2.0]]),
    )

    torch.testing.assert_close(result, torch.tensor([[3.0, 4.0, 0.0]]))


def test_rotation_matrix_is_body_to_world_and_row_major():
    yaw_90 = torch.tensor([[0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]])

    result = mdp.rotation_matrix_flat(yaw_90)

    torch.testing.assert_close(
        result,
        torch.tensor([[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_cable_features_report_capsule_tangent_and_zero_joint_separation():
    segment_poses = torch.tensor([[[0.0, 0.0, -0.125, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -0.375, 1.0, 0.0, 0.0, 0.0]]])
    tangent, relative_separation = mdp.cable_features(
        segment_pose_w=segment_poses,
        drone_attachment_w=torch.zeros(1, 3),
        payload_attachment_w=torch.tensor([[0.0, 0.0, -0.5]]),
        drone_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        nominal_length=0.5,
    )

    torch.testing.assert_close(tangent, torch.tensor([[0.0, 0.0, -1.0]]))
    torch.testing.assert_close(relative_separation, torch.zeros(1, 1))


def test_cable_constraint_errors_detect_separation_and_detached_joint():
    rest_poses = torch.zeros(3, 4, 7)
    rest_poses[..., :3] = torch.tensor(
        [[0.0, 0.0, -0.0625], [0.0, 0.0, -0.1875], [0.0, 0.0, -0.3125], [0.0, 0.0, -0.4375]]
    )
    # Rotate local +Z to world -Z for all hanging capsules.
    rest_poses[..., 3] = 1.0
    rest_poses[1, 1, 0] = 0.2
    rest_poses[2, 0, 2] = -0.10

    relative_separation, joint_error = mdp.cable_constraint_errors(
        rest_poses,
        drone_attachment_w=torch.zeros(3, 3),
        payload_attachment_w=torch.tensor([[0.0, 0.0, -0.5]]).repeat(3, 1),
        nominal_length=0.5,
    )

    torch.testing.assert_close(relative_separation[0], torch.tensor(0.0))
    torch.testing.assert_close(joint_error[0], torch.tensor(0.0))
    assert relative_separation[1] > 0.05
    assert joint_error[2] > 0.01


def test_geometry_features_are_finite_for_degenerate_state():
    zeros = torch.zeros(2, 3)
    tangent, relative_separation = mdp.cable_features(
        torch.zeros(2, 2, 7),
        zeros,
        zeros,
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
        nominal_length=0.0,
    )
    angles, total = mdp.swing_features(zeros)

    for value in (tangent, relative_separation, angles, total):
        assert torch.isfinite(value).all()


def test_world_velocity_observation_does_not_rotate_into_body_frame():
    pose = _pose((0.0, 0.0, 0.0), (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)))
    velocity = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_link_pose_w=SimpleNamespace(torch=pose),
            body_com_vel_w=SimpleNamespace(torch=velocity),
        )
    )
    env = SimpleNamespace(scene={"robot": robot})

    result = mdp.world_lin_vel_normalized(env)

    torch.testing.assert_close(result, torch.tensor([[0.1, 0.0, 0.0]]))


def test_body_velocity_observation_rotates_world_velocity_and_uses_one_speed_scale():
    pose = _pose((0.0, 0.0, 0.0), (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)))
    velocity = torch.tensor([[4.5, 0.0, 2.25, 0.0, 0.0, 0.0]])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_link_pose_w=SimpleNamespace(torch=pose),
            body_com_vel_w=SimpleNamespace(torch=velocity),
        )
    )
    env = SimpleNamespace(scene={"robot": robot})

    result = mdp.body_lin_vel_normalized(env, speed_scale=4.5)

    # Inverse yaw(+90): world +X becomes body -Y; body Z is unchanged.
    torch.testing.assert_close(result, torch.tensor([[0.0, -1.0, 0.5]]), atol=1.0e-6, rtol=0.0)


@pytest.mark.parametrize("speed_scale", [True, 0.0, -1.0, float("nan"), float("inf"), "4.5"])
def test_body_velocity_observation_rejects_invalid_speed_scale(speed_scale):
    pose = _pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    velocity = torch.zeros(1, 6)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_link_pose_w=SimpleNamespace(torch=pose),
            body_com_vel_w=SimpleNamespace(torch=velocity),
        )
    )
    env = SimpleNamespace(scene={"robot": robot})

    with pytest.raises(ValueError, match="speed_scale must be finite and positive"):
        mdp.body_lin_vel_normalized(env, speed_scale=speed_scale)


def test_waypoint_offsets_remain_in_environment_frame_and_use_table_scales():
    robot_pose = _pose((1.0, 2.0, 1.5), (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)))
    robot = SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=robot_pose)))

    class _Scene(dict):
        def __init__(self):
            super().__init__(robot=robot)
            self.env_origins = torch.zeros(1, 3)

    env = SimpleNamespace(
        num_envs=1,
        scene=_Scene(),
        command_manager=SimpleNamespace(get_command=lambda name: torch.tensor([[6.0, 2.0, 2.5, 1.0, 7.0, 0.5]])),
    )

    result = mdp.waypoint_offsets_normalized(env)

    torch.testing.assert_close(result, torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, -1.0]]))


def test_compact_path_tracking_observation_has_documented_body_frame_order():
    robot_pose = _pose((1.0, 2.0, 0.0), (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)))
    robot = SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=robot_pose)))

    class _Scene(dict):
        def __init__(self):
            super().__init__(robot=robot)
            self.env_origins = torch.zeros(1, 3)

    command_term = SimpleNamespace(
        path_cross_track_error_e=torch.tensor([[0.20, 0.0, 0.0]]),
        path_tangent_e=torch.tensor([[0.0, 1.0, 0.0]]),
        path_curvature_e=torch.tensor([[0.0, 0.0, 2.0]]),
        path_progress_fraction=torch.tensor([0.25]),
        path_preview_e=lambda _lookaheads: (
            torch.tensor([[[1.0, 4.0, 0.0], [-1.0, 2.0, 0.0]]]),
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, 3),
        ),
    )
    env = SimpleNamespace(
        num_envs=1,
        scene=_Scene(),
        command_manager=SimpleNamespace(get_term=lambda _name: command_term),
    )

    result = mdp.path_tracking_features_b(
        env,
        lookahead_distances=(0.75, 1.50),
        cross_track_scale=0.20,
        preview_scale=1.0,
        curvature_scale=2.0,
    )

    assert result.shape == (1, 15)
    # Inverse yaw(+90): world +X -> body -Y; world +Y -> body +X.
    torch.testing.assert_close(result[:, 0:3], torch.tensor([[0.0, -1.0, 0.0]]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(result[:, 3:6], torch.tensor([[1.0, 0.0, 0.0]]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(result[:, 6:9], torch.tensor([[0.0, 0.0, 1.0]]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(result[:, 9:15], torch.tensor([[2.0, 0.0, 0.0, 0.0, 2.0, 0.0]]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(mdp.path_progress_fraction(env), torch.tensor([[0.25]]))


def test_path_speed_observation_exposes_live_reference_signed_speed_and_progress():
    env = _slung_load_env(
        payload_position=torch.zeros(2, 3),
        payload_velocity=torch.zeros(2, 6),
        robot_velocity=torch.tensor([[1.75, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.70, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    command_term = SimpleNamespace(
        path_speed_reference=torch.tensor([3.5, 1.75]),
        path_tangent_e=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        path_progress_fraction=torch.tensor([0.25, 0.75]),
        path_state_valid=torch.tensor([True, True]),
    )
    env.command_manager = SimpleNamespace(get_term=lambda _name: command_term)

    result = mdp.path_speed_features(env, speed_scale=3.5)

    torch.testing.assert_close(result, torch.tensor([[1.0, 0.5, 0.25], [0.5, -0.2, 0.75]]))
    command_term.path_state_valid[1] = False
    torch.testing.assert_close(mdp.path_speed_features(env, speed_scale=3.5)[1], torch.zeros(3))


def test_previous_action_is_unscaled_and_finite():
    env = SimpleNamespace(action_manager=SimpleNamespace(action=torch.tensor([[2.0, -2.0, float("nan"), 1.0]])))
    torch.testing.assert_close(mdp.previous_action(env), torch.tensor([[1.0, -1.0, 0.0, 1.0]]))


def test_waypoint_sequence_rotates_by_reset_yaw_and_advances_exactly_once():
    term = object.__new__(commands_module.WaypointSequenceCommand)
    term.cfg = SimpleNamespace(waypoint_offsets=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)), acceptance_radius=0.5)
    pose = _pose((10.0, 20.0, 1.5), (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)))
    term.robot = SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=pose)))
    term._env = SimpleNamespace(num_envs=1, device="cpu", scene=SimpleNamespace(env_origins=torch.zeros(1, 3)))
    term.route_anchor_e = torch.zeros(1, 3)
    term.waypoints_e = torch.zeros(1, 2, 3)
    term.current_index = torch.zeros(1, dtype=torch.long)
    term.completed = torch.zeros(1, dtype=torch.bool)
    term.previous_distance_sq = torch.zeros(1)

    term._resample_command([0])
    torch.testing.assert_close(term.command[0, :3], torch.tensor([10.0, 21.0, 1.5]), atol=1.0e-5, rtol=0.0)
    torch.testing.assert_close(term.command[0, 3:], torch.tensor([10.0, 22.0, 1.5]), atol=1.0e-5, rtol=0.0)

    pose[:, :3] = torch.tensor([[10.0, 20.51, 1.5]])
    term._update_command()
    assert term.current_index.item() == 1
    next_distance_sq = torch.sum(torch.square(term.waypoints_e[:, 1] - pose[:, :3]), dim=-1)
    torch.testing.assert_close(term.previous_distance_sq, next_distance_sq)
    term._update_command()
    torch.testing.assert_close(term.previous_distance_sq, next_distance_sq)

    pose[:, :3] = term.waypoints_e[:, 1]
    term._update_command()
    assert term.completed.item()
    torch.testing.assert_close(term.command[:, :3], term.command[:, 3:])


def test_waypoint_switch_seeds_new_squared_distance_reward_potential():
    pose = _pose((0.4, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    robot = SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=pose)))

    class _Scene(dict):
        def __init__(self):
            super().__init__(robot=robot)
            self.env_origins = torch.zeros(1, 3)

    term = object.__new__(commands_module.WaypointSequenceCommand)
    term.cfg = SimpleNamespace(acceptance_radius=0.5)
    term.robot = robot
    term.waypoints_e = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    term.current_index = torch.zeros(1, dtype=torch.long)
    term.completed = torch.zeros(1, dtype=torch.bool)
    term.previous_distance_sq = torch.tensor([0.6**2])
    env = SimpleNamespace(num_envs=1, device="cpu", step_dt=0.01, scene=_Scene())
    term._env = env
    env.command_manager = SimpleNamespace(get_term=lambda name: term, get_command=lambda name: term.command)

    pose[:, 0] = 0.6
    old_target_reward = mdp.waypoint_progress(env) * 10.0 * env.step_dt
    term._update_command()

    assert term.current_index.item() == 1
    torch.testing.assert_close(old_target_reward, torch.tensor([2.0]))
    torch.testing.assert_close(term.previous_distance_sq, torch.tensor([1.4**2]))

    pose[:, 0] = 0.7
    new_target_reward = mdp.waypoint_progress(env) * 10.0 * env.step_dt
    torch.testing.assert_close(new_target_reward, torch.tensor([2.7]))


def test_route_segment_markers_join_waypoints_with_z_axis_cylinders():
    waypoints = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, -3.0]]], dtype=torch.float32)

    midpoint, orientation, scale = commands_module._route_segment_marker_transforms(waypoints)

    torch.testing.assert_close(midpoint, torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, -1.5]]]))
    torch.testing.assert_close(scale, torch.tensor([[[1.0, 1.0, 2.0], [1.0, 1.0, 3.0]]]))
    # +Z -> +X is +90 degrees about Y; +Z -> -Z uses the stable 180-degree-X branch.
    torch.testing.assert_close(
        orientation,
        torch.tensor([[[0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)], [1.0, 0.0, 0.0, 0.0]]]),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_waypoint_debug_visualization_marks_route_progress_and_world_origin():
    class _CaptureVisualizer:
        def __init__(self):
            self.kwargs = None

        def visualize(self, **kwargs):
            self.kwargs = kwargs

    term = object.__new__(commands_module.WaypointSequenceCommand)
    term.robot = SimpleNamespace(is_initialized=True)
    term._env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene=SimpleNamespace(env_origins=torch.tensor([[10.0, 20.0, 30.0]])),
    )
    term.route_anchor_e = torch.tensor([[0.0, 0.0, 0.0]])
    term.waypoints_e = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]]])
    term.current_index = torch.tensor([1])
    term.completed = torch.tensor([False])
    term.waypoint_visualizer = _CaptureVisualizer()
    term.route_segment_visualizer = _CaptureVisualizer()

    term._debug_vis_callback(None)

    torch.testing.assert_close(
        term.waypoint_visualizer.kwargs["translations"],
        torch.tensor([[10.0, 20.0, 30.0], [11.0, 20.0, 30.0], [12.0, 21.0, 30.0]]),
    )
    # Prototype order: active=0, completed=1, future=2.
    assert term.waypoint_visualizer.kwargs["marker_indices"].tolist() == [1, 0, 2]
    # Prototype order: completed=0, future=1.
    assert term.route_segment_visualizer.kwargs["marker_indices"].tolist() == [0, 1, 1]

    term.current_index[:] = 2
    term.completed[:] = True
    term._debug_vis_callback(None)
    assert term.waypoint_visualizer.kwargs["marker_indices"].tolist() == [1, 1, 1]
    assert term.route_segment_visualizer.kwargs["marker_indices"].tolist() == [0, 0, 0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"waypoint_offsets": ((1.0, 0.0, 0.0),)},
        {"waypoint_offsets": ((1.0, 0.0), (2.0, 0.0, 0.0))},
        {"waypoint_offsets": ((1.0, 0.0, 0.0), (2.0, 0.0, float("nan")))},
        {"acceptance_radius": 0.0},
    ],
)
def test_waypoint_command_rejects_invalid_geometry(kwargs):
    cfg = commands_module.WaypointSequenceCommandCfg(**kwargs)
    env = SimpleNamespace(num_envs=1, device="cpu", scene={"robot": SimpleNamespace()})
    with pytest.raises(ValueError):
        commands_module.WaypointSequenceCommand(cfg, env)


def test_paper_impulse_rewards_cancel_manager_step_scaling():
    robot_pose = _pose((1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    robot = SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=robot_pose)))

    class _Scene(dict):
        def __init__(self):
            super().__init__(robot=robot)
            self.env_origins = torch.zeros(1, 3)

    command_term = SimpleNamespace(previous_distance_sq=torch.tensor([4.0]))
    env = SimpleNamespace(
        step_dt=0.01,
        scene=_Scene(),
        action_manager=SimpleNamespace(
            action=torch.tensor([[1.0, -1.0, 0.5, 0.0]]),
            prev_action=torch.zeros(1, 4),
        ),
        command_manager=SimpleNamespace(
            get_term=lambda name: command_term,
            get_command=lambda name: torch.tensor([[2.0, 0.0, 0.0, 2.0, 1.0, 0.0]]),
        ),
        termination_manager=SimpleNamespace(terminated=torch.tensor([True])),
    )

    progress = mdp.waypoint_progress(env) * 10.0 * env.step_dt
    linear_progress = mdp.waypoint_distance_progress(env)
    smoothness = mdp.action_delta_l2(env) * -1.0e-4 * env.step_dt
    safety = mdp.swing_safety_impulse(env, threshold=1.0, angles=torch.tensor([[1.01, 0.0]])) * -3.0 * env.step_dt
    crash = mdp.crash_impulse(env) * -10.0 * env.step_dt

    torch.testing.assert_close(progress, torch.tensor([30.0]))
    torch.testing.assert_close(linear_progress, torch.tensor([100.0]))
    torch.testing.assert_close(smoothness, torch.tensor([-1.5e-4]))
    torch.testing.assert_close(safety, torch.tensor([-3.0]))
    torch.testing.assert_close(crash, torch.tensor([-10.0]))


def test_continuous_stability_costs_are_normalized_squared_errors():
    env = _slung_load_env(
        payload_position=torch.tensor([[1.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        payload_velocity=torch.tensor([[0.0] * 6, [3.0, 4.0, 5.0, 0.0, 0.0, 0.0]]),
        robot_velocity=torch.tensor([[0.0, 0.0, 0.0, math.pi, 2.0 * math.pi, 3.0 * math.pi], [0.0] * 6]),
    )

    swing = mdp.total_swing_angle_l2(env, scale=math.pi / 4.0)
    transverse_speed = mdp.payload_transverse_speed_l2(env, scale=5.0)
    body_rate = mdp.body_angular_velocity_l2(env, scale=(math.pi, 2.0 * math.pi, 3.0 * math.pi))

    torch.testing.assert_close(swing, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(transverse_speed, torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(body_rate, torch.tensor([3.0, 0.0]))
    for value in (swing, transverse_speed, body_rate):
        assert torch.isfinite(value).all()


def test_body_tilt_cost_is_yaw_invariant_smooth_and_finite():
    half_angle = math.pi / 8.0
    quaternion = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)],
            [math.sin(half_angle), 0.0, 0.0, math.cos(half_angle)],
            [float("nan"), 0.0, 0.0, 1.0],
        ]
    )
    asset = SimpleNamespace(
        data=SimpleNamespace(
            body_link_pose_w=SimpleNamespace(torch=torch.cat((torch.zeros(4, 3), quaternion), dim=-1)[:, None])
        )
    )
    env = SimpleNamespace(scene={"robot": asset})

    result = mdp.body_tilt_l2(env, scale=math.pi / 4.0)

    expected_tilt = 2.0 * (1.0 - math.cos(math.pi / 4.0)) / (math.pi / 4.0) ** 2
    torch.testing.assert_close(result, torch.tensor([0.0, 0.0, expected_tilt, 0.0]))
    assert torch.isfinite(result).all()

    robust = mdp.body_tilt_exp(env, scale=math.pi / 4.0)
    torch.testing.assert_close(robust, -torch.expm1(-result))
    assert torch.all((robust >= 0.0) & (robust <= 1.0))


def test_indexed_cross_track_cost_uses_command_segment_distance_and_is_finite():
    command_term = SimpleNamespace(cross_track_distance=torch.tensor([0.25, float("nan"), float("inf")]))
    env = SimpleNamespace(
        num_envs=3,
        command_manager=SimpleNamespace(get_term=lambda name: command_term),
    )

    result = mdp.indexed_cross_track_error_l2(env, scale=0.5)

    torch.testing.assert_close(result, torch.tensor([0.25, 0.0, 0.0]))
    assert torch.isfinite(result).all()
    with pytest.raises(ValueError, match="finite positive"):
        mdp.indexed_cross_track_error_l2(env, scale=float("nan"))


def test_indexed_cross_track_exp_cost_is_locally_quadratic_and_bounded():
    command_term = SimpleNamespace(cross_track_distance=torch.tensor([0.0, 0.05, 0.5, 5.0, float("nan")]))
    env = SimpleNamespace(
        num_envs=5,
        command_manager=SimpleNamespace(get_term=lambda name: command_term),
    )

    result = mdp.indexed_cross_track_error_exp(env, scale=0.5)

    expected = -torch.expm1(-torch.tensor([0.0, 0.01, 1.0, 100.0, 0.0]))
    torch.testing.assert_close(result, expected)
    assert torch.all((result >= 0.0) & (result <= 1.0))


def test_spline_progress_and_precision_rewards_have_rate_and_cost_semantics():
    env = _slung_load_env(
        payload_position=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        payload_velocity=torch.zeros(2, 6),
        robot_velocity=torch.tensor([[3.0, 1.0, 0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    command_term = SimpleNamespace(
        cross_track_distance=torch.tensor([0.10, float("inf")]),
        path_tangent_e=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        path_progress=torch.tensor([0.13, 0.49]),
        previous_path_progress=torch.tensor([0.10, 0.50]),
        path_state_valid=torch.tensor([True, True]),
    )
    env.step_dt = 0.01
    env.command_manager = SimpleNamespace(get_term=lambda _name: command_term)

    progress_rate = mdp.path_arc_length_progress(env)
    transverse_cost = mdp.path_transverse_speed_l2(env, scale=2.0)
    precision_cost = mdp.path_tracking_precision_exp(
        env,
        cross_track_scale=0.20,
        transverse_velocity_scale=2.0,
        transverse_speed_weight=0.5,
    )

    torch.testing.assert_close(progress_rate, torch.tensor([3.0, -1.0]), atol=1.0e-5, rtol=0.0)
    torch.testing.assert_close(progress_rate * env.step_dt, torch.tensor([0.03, -0.01]), atol=1.0e-7, rtol=0.0)
    torch.testing.assert_close(transverse_cost, torch.tensor([0.25, 0.0]))
    expected_first_cost = -math.expm1(-(0.5**2 + 0.5 * 0.5**2))
    torch.testing.assert_close(precision_cost, torch.tensor([expected_first_cost, 0.0]))
    assert torch.all((precision_cost >= 0.0) & (precision_cost <= 1.0))

    stricter_cost = mdp.path_tracking_precision_exp(
        env,
        cross_track_scale=0.10,
        transverse_velocity_scale=2.0,
        transverse_speed_weight=0.5,
    )
    assert stricter_cost[0] > precision_cost[0]
    torch.testing.assert_close(mdp.path_arc_length_progress(env, maximum_rate=2.0), torch.tensor([2.0, -1.0]))

    with pytest.raises(ValueError, match="finite and positive"):
        mdp.path_arc_length_progress(env, maximum_rate=0.0)
    with pytest.raises(ValueError, match="maximum_lateral_acceleration must be finite and positive"):
        mdp.path_arc_length_progress(env, maximum_lateral_acceleration=0.0)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        mdp.path_tracking_precision_exp(env, transverse_speed_weight=-1.0)


def test_spline_progress_uses_a_signed_curvature_aware_rate_cap():
    command_term = SimpleNamespace(
        path_progress=torch.tensor([0.05, -0.05, 0.05, -0.05]),
        previous_path_progress=torch.zeros(4),
        path_curvature_e=torch.tensor([[0.0, 0.0, 0.20], [0.0, 0.0, 0.93], [0.0, 0.0, 0.93], [0.0, 0.0, 0.0]]),
        path_state_valid=torch.ones(4, dtype=torch.bool),
    )
    env = SimpleNamespace(
        num_envs=4,
        step_dt=0.01,
        command_manager=SimpleNamespace(get_term=lambda _name: command_term),
    )

    result = mdp.path_arc_length_progress(
        env,
        maximum_rate=3.5,
        maximum_lateral_acceleration=3.0,
    )

    crossing_rate = math.sqrt(3.0 / 0.93)
    torch.testing.assert_close(result, torch.tensor([3.5, -crossing_rate, crossing_rate, -3.5]))

    command_term.path_curvature_e[0, 0] = float("nan")
    assert mdp.path_arc_length_progress(env, maximum_lateral_acceleration=3.0)[0] == 0.0


def test_spline_progress_c1_gate_applies_only_to_positive_progress_after_caps():
    command_term = SimpleNamespace(
        path_progress=torch.tensor([0.05, 0.05, 0.05, 0.05, -0.05, -0.05, -0.05, -0.05]),
        previous_path_progress=torch.zeros(8),
        path_curvature_e=torch.tensor(
            [
                [0.0, 0.0, 0.75],
                [0.0, 0.0, 3.00],
                [0.0, 0.0, 0.75],
                [0.0, 0.0, 0.75],
                [0.0, 0.0, 0.75],
                [0.0, 0.0, 3.00],
                [0.0, 0.0, 0.75],
                [0.0, 0.0, 0.75],
            ]
        ),
        path_cross_track_distance=torch.tensor([0.0, 0.375, 0.75, 1.0, 0.0, 0.375, 0.75, 1.0]),
        path_state_valid=torch.ones(8, dtype=torch.bool),
    )
    env = SimpleNamespace(
        num_envs=8,
        step_dt=0.01,
        command_manager=SimpleNamespace(get_term=lambda _name: command_term),
    )

    result = mdp.path_arc_length_progress(
        env,
        maximum_rate=2.0,
        maximum_lateral_acceleration=3.0,
        positive_progress_gate_distance=0.75,
    )

    # At half the gate distance, (1 - 0.5^2)^2 = 0.5625. Applying
    # this after the 1 m/s curvature cap distinguishes the intended ordering.
    torch.testing.assert_close(result, torch.tensor([2.0, 0.5625, 0.0, 0.0, -2.0, -1.0, -2.0, -2.0]))


def test_spline_progress_gate_default_preserves_legacy_without_cross_track_state():
    command_term = SimpleNamespace(
        path_progress=torch.tensor([0.05, -0.05]),
        previous_path_progress=torch.zeros(2),
        path_state_valid=torch.ones(2, dtype=torch.bool),
    )
    env = SimpleNamespace(
        num_envs=2,
        step_dt=0.01,
        command_manager=SimpleNamespace(get_term=lambda _name: command_term),
    )

    torch.testing.assert_close(mdp.path_arc_length_progress(env, maximum_rate=2.0), torch.tensor([2.0, -2.0]))


def test_spline_progress_gate_masks_nonfinite_distance_and_validates_shape():
    command_term = SimpleNamespace(
        path_progress=torch.full((4,), 0.01),
        previous_path_progress=torch.zeros(4),
        path_cross_track_distance=torch.tensor([0.0, float("nan"), float("inf"), -float("inf")]),
        path_state_valid=torch.ones(4, dtype=torch.bool),
    )
    env = SimpleNamespace(
        num_envs=4,
        step_dt=0.01,
        command_manager=SimpleNamespace(get_term=lambda _name: command_term),
    )

    result = mdp.path_arc_length_progress(env, positive_progress_gate_distance=0.75)

    torch.testing.assert_close(result, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.isfinite(result).all()

    command_term.path_cross_track_distance = torch.zeros(4, 1)
    with pytest.raises(ValueError, match="path_cross_track_distance must have shape"):
        mdp.path_arc_length_progress(env, positive_progress_gate_distance=0.75)


@pytest.mark.parametrize("gate_distance", [0.0, -0.75, float("nan"), float("inf"), True])
def test_spline_progress_gate_rejects_invalid_distance(gate_distance):
    env = SimpleNamespace(num_envs=1)

    with pytest.raises(ValueError, match="positive_progress_gate_distance must be finite and positive"):
        mdp.path_arc_length_progress(env, positive_progress_gate_distance=gate_distance)


def test_log1p_path_precision_cost_is_separable_locally_quadratic_and_has_robust_tails():
    env = _slung_load_env(
        payload_position=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        payload_velocity=torch.zeros(2, 6),
        robot_velocity=torch.tensor([[3.0, 1.0, 0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    command_term = SimpleNamespace(
        cross_track_distance=torch.tensor([0.10, 2.0]),
        path_tangent_e=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    env.command_manager = SimpleNamespace(get_term=lambda _name: command_term)

    cost = mdp.path_tracking_precision_log1p(
        env,
        cross_track_scale=0.20,
        transverse_velocity_scale=2.0,
        transverse_speed_weight=0.5,
    )

    expected = torch.tensor([math.log1p(0.5**2) + 0.5 * math.log1p(0.5**2), math.log1p(10.0**2)])
    torch.testing.assert_close(cost, expected)
    assert cost[1] > 1.0
    assert torch.isfinite(cost).all()

    with pytest.raises(ValueError, match="finite and nonnegative"):
        mdp.path_tracking_precision_log1p(env, transverse_speed_weight=-1.0)


def test_path_corridor_violation_is_strict_finite_and_shape_checked():
    command_term = SimpleNamespace(path_cross_track_distance=torch.tensor([0.75, 0.751, float("nan")]))
    env = SimpleNamespace(
        num_envs=3,
        command_manager=SimpleNamespace(get_term=lambda name: command_term),
    )

    assert mdp.path_corridor_violation(env, maximum_distance=0.75).tolist() == [False, True, True]

    command_term.path_cross_track_distance = torch.zeros(3, 2)
    with pytest.raises(ValueError, match="path_cross_track_distance must have shape"):
        mdp.path_corridor_violation(env, maximum_distance=0.75)
    with pytest.raises(ValueError, match="finite and positive"):
        mdp.path_corridor_violation(env, maximum_distance=0.0)


def test_normalized_action_acceleration_is_clamped_and_reset_safe():
    raw_action = torch.tensor([[2.0, float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    action_term = SimpleNamespace(raw_actions=raw_action)

    class _ActionManager:
        # Deliberately different from the term buffer: the reward should use the
        # action term that applies the task's physical clamp.
        action = torch.full_like(raw_action, -10.0)

        @staticmethod
        def get_term(name):
            assert name == "thrust"
            return action_term

    env = SimpleNamespace(num_envs=2, device="cpu", action_manager=_ActionManager())
    reward = mdp.NormalizedActionAccelerationL2(SimpleNamespace(params={}), env)

    torch.testing.assert_close(reward(env), torch.zeros(2))

    raw_action[:] = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(reward(env), torch.tensor([1.0, 0.25]))

    reward.reset(torch.tensor([0]))
    raw_action[:] = torch.tensor([[-1.0, 0.0, 0.0, 0.0], [0.75, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(reward(env), torch.tensor([0.0, 0.0625]))


def test_progress_and_action_rewards_remain_finite_on_illegal_terminal_sample():
    robot_pose = _pose((float("nan"), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    robot = SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=robot_pose)))

    class _Scene(dict):
        def __init__(self):
            super().__init__(robot=robot)
            self.env_origins = torch.zeros(1, 3)

    env = SimpleNamespace(
        step_dt=0.01,
        scene=_Scene(),
        action_manager=SimpleNamespace(
            action=torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]),
            prev_action=torch.zeros(1, 4),
        ),
        command_manager=SimpleNamespace(
            get_term=lambda name: SimpleNamespace(previous_distance_sq=torch.tensor([1.0])),
            get_command=lambda name: torch.tensor([[1.0, 0.0, 0.0, 2.0, 0.0, 0.0]]),
        ),
    )

    torch.testing.assert_close(mdp.waypoint_progress(env), torch.zeros(1))
    torch.testing.assert_close(mdp.waypoint_distance_progress(env), torch.zeros(1))
    assert torch.isfinite(mdp.action_delta_l2(env)).all()


def test_path_tangent_speed_tracking_uses_live_curvature_and_braking_reference_with_signed_speed():
    env = _slung_load_env(
        payload_position=torch.zeros(4, 3),
        payload_velocity=torch.zeros(4, 6),
        robot_velocity=torch.tensor(
            [
                [4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2.3, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    command_term = SimpleNamespace(
        # Respectively cruise-, curvature-, and braking-limited references,
        # followed by a reference used to expose signed backtracking.
        path_speed_reference=torch.tensor([3.5, 1.8, 1.0, 2.0]),
        path_tangent_e=torch.tensor([[1.0, 0.0, 0.0]]).repeat(4, 1),
        path_state_valid=torch.ones(4, dtype=torch.bool),
    )
    env.command_manager = SimpleNamespace(get_term=lambda _name: command_term)

    cost = mdp.path_tangent_speed_tracking_l2(
        env,
        speed_error_scale=0.5,
        underspeed_weight=1.0,
        overspeed_weight=2.0,
    )

    # Both local caps are obeyed: 0.5 m/s overspeed costs twice the normalized
    # squared error. A -1 m/s backtrack remains -1 rather than becoming abs(-1).
    torch.testing.assert_close(cost, torch.tensor([2.0, 2.0, 0.0, 36.0]))
    command_term.path_state_valid[1] = False
    assert mdp.path_tangent_speed_tracking_l2(env)[1] == 0.0


def test_path_velocity_tracking_l2_uses_inward_recovery_vector_cap_and_rotation():
    env = _slung_load_env(
        payload_position=torch.zeros(6, 3),
        payload_velocity=torch.zeros(6, 6),
        robot_velocity=torch.tensor(
            [
                [2.0, -0.75, 0.0, 0.0, 0.0, 0.0],
                [2.0, 0.75, 0.0, 0.0, 0.0, 0.0],
                [0.75, 2.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 0.375, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    command_term = SimpleNamespace(
        path_speed_reference=torch.tensor([2.0, 2.0, 2.0, 2.0, 1.0, 2.0]),
        path_tangent_e=torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        path_cross_track_error_e=torch.tensor(
            [
                [0.0, 10.0, 0.0],
                [0.0, -10.0, 0.0],
                [-10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.0, 0.25, 0.0],
            ]
        ),
        path_state_valid=torch.ones(6, dtype=torch.bool),
    )
    env.command_manager = SimpleNamespace(get_term=lambda _name: command_term)

    cost = mdp.path_velocity_tracking_l2(env)

    # Positive projection-to-vehicle Y error requests negative Y recovery;
    # negative error reverses it. Rotating every vector 90 degrees is invariant.
    torch.testing.assert_close(cost[:3], torch.zeros(3))
    # The 10 m error is independently capped at 0.75 m/s.
    assert cost[3] == pytest.approx(2.0**2 + 0.75**2)
    # Below the cap, -gain*error is exact in 3D. The final sample deliberately
    # flies away from the path, producing a 0.75 m/s signed recovery error.
    assert cost[4] == pytest.approx(1.0**2 + 0.30**2)
    assert cost[5] == pytest.approx(0.75**2)
    torch.testing.assert_close(mdp.path_velocity_tracking_l2(env, velocity_error_scale=0.5), 4.0 * cost)


def test_path_velocity_tracking_l2_distinguishes_desired_hover_and_backtracking_and_reads_live_reference():
    env = _slung_load_env(
        payload_position=torch.zeros(3, 3),
        payload_velocity=torch.zeros(3, 6),
        robot_velocity=torch.tensor(
            [
                [2.25, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-2.25, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    command_term = SimpleNamespace(
        path_speed_reference=torch.full((3,), 2.25),
        path_tangent_e=torch.tensor([[1.0, 0.0, 0.0]]).repeat(3, 1),
        path_cross_track_error_e=torch.zeros(3, 3),
        path_state_valid=torch.ones(3, dtype=torch.bool),
    )
    env.command_manager = SimpleNamespace(get_term=lambda _name: command_term)

    torch.testing.assert_close(
        mdp.path_velocity_tracking_l2(env),
        torch.tensor([0.0, 2.25**2, (2.0 * 2.25) ** 2]),
    )

    command_term.path_speed_reference[0] = 3.0
    assert mdp.path_velocity_tracking_l2(env)[0] == pytest.approx(0.75**2)


def test_path_velocity_tracking_l2_masks_nonfinite_and_invalid_path_samples():
    env = _slung_load_env(
        payload_position=torch.zeros(6, 3),
        payload_velocity=torch.zeros(6, 6),
        robot_velocity=torch.zeros(6, 6),
    )
    command_term = SimpleNamespace(
        path_speed_reference=torch.full((6,), 2.0),
        path_tangent_e=torch.tensor([[1.0, 0.0, 0.0]]).repeat(6, 1),
        path_cross_track_error_e=torch.zeros(6, 3),
        path_state_valid=torch.ones(6, dtype=torch.bool),
    )
    env.command_manager = SimpleNamespace(get_term=lambda _name: command_term)
    command_term.path_speed_reference[0] = float("nan")
    command_term.path_tangent_e[1, 0] = float("inf")
    command_term.path_cross_track_error_e[2, 0] = float("nan")
    env.scene["robot"].data.body_com_vel_w.torch[3, 0] = float("inf")
    command_term.path_tangent_e[4] = 0.0
    command_term.path_state_valid[5] = False

    cost = mdp.path_velocity_tracking_l2(env)

    torch.testing.assert_close(cost, torch.zeros(6))
    assert torch.isfinite(cost).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cross_track_gain": -1.0}, "cross_track_gain must be finite and nonnegative"),
        ({"cross_track_gain": float("nan")}, "cross_track_gain must be finite and nonnegative"),
        ({"maximum_cross_track_speed": 0.0}, "maximum_cross_track_speed must be finite and positive"),
        ({"velocity_error_scale": 0.0}, "velocity_error_scale must be finite and positive"),
    ],
)
def test_path_velocity_tracking_l2_rejects_invalid_parameters(kwargs, message):
    env = _slung_load_env(torch.zeros(1, 3), torch.zeros(1, 6))
    env.command_manager = SimpleNamespace(
        get_term=lambda _name: SimpleNamespace(
            path_speed_reference=torch.ones(1),
            path_tangent_e=torch.tensor([[1.0, 0.0, 0.0]]),
            path_cross_track_error_e=torch.zeros(1, 3),
        )
    )

    with pytest.raises(ValueError, match=message):
        mdp.path_velocity_tracking_l2(env, **kwargs)


def test_path_velocity_tracking_l2_validates_command_tensor_contract():
    env = _slung_load_env(torch.zeros(1, 3), torch.zeros(1, 6))
    command_term = SimpleNamespace(
        path_speed_reference=torch.ones(1, 1),
        path_tangent_e=torch.tensor([[1.0, 0.0, 0.0]]),
        path_cross_track_error_e=torch.zeros(1, 3),
    )
    env.command_manager = SimpleNamespace(get_term=lambda _name: command_term)

    with pytest.raises(ValueError, match="path speed reference must have shape"):
        mdp.path_velocity_tracking_l2(env)
    command_term.path_speed_reference = torch.ones(1)
    command_term.path_tangent_e = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="path tangent must have shape"):
        mdp.path_velocity_tracking_l2(env)
    command_term.path_tangent_e = torch.tensor([[1.0, 0.0, 0.0]])
    command_term.path_cross_track_error_e = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="path cross-track error must have shape"):
        mdp.path_velocity_tracking_l2(env)
    command_term.path_cross_track_error_e = torch.zeros(1, 3)
    command_term.path_state_valid = torch.ones(1)
    with pytest.raises(ValueError, match="path_state_valid must be a bool tensor"):
        mdp.path_velocity_tracking_l2(env)


def test_route_completion_impulse_emits_once_and_preserves_early_completion_incentive():
    command_term = SimpleNamespace(
        completed=torch.zeros(3, dtype=torch.bool),
        metrics={"waypoint_completion_time": torch.tensor([8.0, 10.0, 14.0])},
    )
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        step_dt=0.01,
        command_manager=SimpleNamespace(get_term=lambda _name: command_term),
    )
    reward = mdp.RouteCompletionImpulse(SimpleNamespace(params={}), env)

    # The first post-reset observation only seeds transition state.
    torch.testing.assert_close(reward(env), torch.zeros(3))
    command_term.completed[:] = torch.tensor([True, False, True])
    result = reward(env, reference_completion_time=15.0, early_completion_scale=1.0)
    torch.testing.assert_close(result * env.step_dt, torch.tensor([1.0 + 7.0 / 15.0, 0.0, 1.0 + 1.0 / 15.0]))
    torch.testing.assert_close(
        reward(env, reference_completion_time=15.0, early_completion_scale=1.0),
        torch.zeros(3),
    )

    # RewardManager resets before CommandManager. Re-arm without reading the
    # still-true command, then seed false after CommandManager has reset it.
    reward.reset(torch.tensor([0]))
    command_term.completed[0] = False
    torch.testing.assert_close(reward(env), torch.zeros(3))
    command_term.completed[0] = True
    assert reward(env)[0] * env.step_dt == pytest.approx(1.0)


def test_waypoint_advance_impulse_counts_multi_knot_progress_once_and_rearms_on_reset():
    command_term = SimpleNamespace(
        current_index=torch.zeros(3, dtype=torch.long),
        completed=torch.zeros(3, dtype=torch.bool),
    )
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        step_dt=0.01,
        command_manager=SimpleNamespace(get_term=lambda _name: command_term),
    )
    reward = mdp.WaypointAdvanceImpulse(SimpleNamespace(params={}), env)

    torch.testing.assert_close(reward(env), torch.zeros(3))
    command_term.current_index[:] = torch.tensor([1, 3, 0])
    torch.testing.assert_close(reward(env) * env.step_dt, torch.tensor([1.0, 3.0, 0.0]))
    torch.testing.assert_close(reward(env), torch.zeros(3))

    # The completion bit represents the final knot after the index can no
    # longer advance, so it contributes exactly one additional event.
    command_term.completed[0] = True
    torch.testing.assert_close(reward(env) * env.step_dt, torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(reward(env), torch.zeros(3))

    reward.reset(torch.tensor([0, 1]))
    command_term.current_index[:2] = 0
    command_term.completed[:2] = False
    torch.testing.assert_close(reward(env), torch.zeros(3))
    command_term.current_index[1] = 1
    assert reward(env)[1] * env.step_dt == pytest.approx(1.0)


def test_unsafe_termination_impulse_omits_success_but_keeps_simultaneous_failure():
    term_values = {
        "drone_crash": torch.tensor([False, True, False]),
        "path_corridor": torch.tensor([False, False, True]),
        "route_completed": torch.tensor([True, True, False]),
    }

    class _TerminationManager:
        @staticmethod
        def get_term(name):
            return term_values[name]

        @staticmethod
        def get_term_cfg(_name):
            return SimpleNamespace(time_out=False)

    env = SimpleNamespace(num_envs=3, device="cpu", step_dt=0.01, termination_manager=_TerminationManager())

    impulse = mdp.unsafe_termination_impulse(env, unsafe_term_names=("drone_crash", "path_corridor"))

    # Pure success is free; success plus crash is still unsafe.
    torch.testing.assert_close(impulse * env.step_dt, torch.tensor([0.0, 1.0, 1.0]))


def test_route_completed_termination_returns_command_transition_state():
    completed = torch.tensor([False, True, True])
    env = SimpleNamespace(
        num_envs=3,
        command_manager=SimpleNamespace(get_term=lambda _name: SimpleNamespace(completed=completed)),
    )

    result = mdp.route_completed(env)

    assert result is completed
