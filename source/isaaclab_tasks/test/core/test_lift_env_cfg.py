# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral tests for the unified dexterous Lift and Reorient tasks."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import CommandTerm

from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.core.lift.adr_curriculum import CurriculumCfg
from isaaclab_tasks.core.lift.config.franka_soft.agents.rsl_rl_ppo_cfg import FrankaDeformableCameraPPORunnerCfg
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import FrankaSoftCameraEnvCfg
from isaaclab_tasks.core.lift.mdp.commands.pose_commands import (
    CableUniformPoseCommand,
    DeformableUniformPoseCommand,
    ObjectUniformPoseCommand,
)


class _MarkerSpy:
    def __init__(self, _cfg=None):
        self.calls: list[tuple[tuple, dict]] = []

    def set_visibility(self, _visible: bool) -> None:
        pass

    def visualize(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


class _FakeScene(dict):
    def __init__(self, environment_ids: torch.Tensor, **assets):
        super().__init__(assets)
        self._ALL_INDICES = environment_ids
        self.env_origins = torch.zeros((len(environment_ids), 3))


def _tensor_data(value: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(torch=value)


def test_point_cloud_noise_curriculum_is_symmetric() -> None:
    """Point-cloud noise should widen evenly around the uncorrupted observation."""
    curriculum = CurriculumCfg()

    minimum = curriculum.object_obs_unoise_min_adr.params["modify_params"]
    maximum = curriculum.object_obs_unoise_max_adr.params["modify_params"]

    assert minimum["initial_value"] == maximum["initial_value"] == 0.0
    assert minimum["final_value"] == -maximum["final_value"] == -0.01


def test_camera_normalization_is_stationary() -> None:
    """RGB and depth normalization must not depend on per-frame statistics."""
    rgb = torch.tensor([0.0, 127.5, 255.0])
    depth = torch.tensor([0.0, 2.0])

    assert torch.allclose(mdp.vision_camera._rgb_norm(None, rgb), torch.tensor([-0.5, 0.0, 0.5]))
    assert torch.allclose(mdp.vision_camera._depth_norm(None, depth), torch.tanh(depth / 2) - 0.5)


def test_franka_camera_checkpoint_contract() -> None:
    """The Franka camera task must reproduce the observation and agent contract used for training."""
    env_cfg = FrankaSoftCameraEnvCfg()
    agent_cfg = FrankaDeformableCameraPPORunnerCfg()

    image_term = env_cfg.observations.base_image.image
    assert image_term.func is mdp.vision_camera
    assert image_term.params["sensor_cfg"].name == "base_camera"
    assert agent_cfg.actor.class_name.endswith(":SpatialSoftmaxCNNModel")
    assert agent_cfg.algorithm.learning_rate == pytest.approx(7.0e-5)
    assert agent_cfg.algorithm.schedule == "fixed"


def test_lift_pose_markers_forward_environment_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every lift pose, goal, and success marker should retain its environment ownership."""
    num_envs = 3
    environment_ids = torch.arange(num_envs)
    identity_quat = torch.zeros((num_envs, 4))
    identity_quat[:, 0] = 1.0
    root_pos_w = torch.zeros((num_envs, 3))
    root_pose_w = torch.cat((root_pos_w, identity_quat), dim=-1)

    robot = SimpleNamespace(
        is_initialized=True,
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=root_pos_w),
            root_quat_w=SimpleNamespace(torch=identity_quat),
        ),
    )
    object_asset = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=root_pos_w),
            root_quat_w=SimpleNamespace(torch=identity_quat),
            root_link_pose_w=SimpleNamespace(torch=root_pose_w),
        )
    )
    success_asset = SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=root_pos_w)))
    scene = _FakeScene(environment_ids, robot=robot, object=object_asset, table=success_asset)
    env = SimpleNamespace(num_envs=num_envs, device="cpu", scene=scene)
    cfg = SimpleNamespace(
        asset_name="robot",
        object_name="object",
        success_vis_asset_name="table",
        success_visualizer_cfg=object(),
        goal_pose_visualizer_cfg=object(),
        curr_pose_visualizer_cfg=object(),
        position_only=True,
        cmd_kind=None,
        element_names=None,
    )

    def _initialize_command_term(command, command_cfg, command_env) -> None:
        command.cfg = command_cfg
        command._env = command_env
        command.metrics = {}

    monkeypatch.setattr(CommandTerm, "__init__", _initialize_command_term)
    monkeypatch.setattr("isaaclab.markers.VisualizationMarkers", _MarkerSpy)

    command = ObjectUniformPoseCommand(cfg, env)
    command._set_debug_vis_impl(True)
    command._debug_vis_callback(None)
    command.cfg.position_only = False
    command._debug_vis_callback(None)
    command._update_metrics()
    DeformableUniformPoseCommand._update_metrics(command)
    command._segment_position_w = lambda: root_pos_w
    CableUniformPoseCommand._update_metrics(command)
    CableUniformPoseCommand._debug_vis_callback(command, None)

    expected_call_counts = {
        command.success_visualizer: 4,
        command.goal_visualizer: 3,
        command.curr_visualizer: 3,
    }
    for visualizer, expected_count in expected_call_counts.items():
        assert len(visualizer.calls) == expected_count
        for _, kwargs in visualizer.calls:
            assert torch.equal(kwargs["environment_ids"], environment_ids)


def test_lift_point_cloud_markers_repeat_environment_ids_per_point() -> None:
    """Flattened point-cloud markers should retain env-major ownership."""
    num_envs = 3
    num_points = 4
    identity_quat = torch.zeros((num_envs, 4))
    identity_quat[:, 0] = 1.0
    root_pos_w = torch.zeros((num_envs, 3))
    points_local = torch.arange(num_envs * num_points * 3, dtype=torch.float32).view(num_envs, num_points, 3)

    term = object.__new__(mdp.object_point_cloud_b)
    term.object = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=root_pos_w),
            root_quat_w=SimpleNamespace(torch=identity_quat),
        )
    )
    term.ref_asset = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=root_pos_w),
            root_quat_w=SimpleNamespace(torch=identity_quat),
        )
    )
    term.points_local = points_local
    term.points_w = torch.zeros_like(points_local)
    term.visualizer = _MarkerSpy()
    env = SimpleNamespace(num_envs=num_envs)

    term(env, num_points=num_points, visualize=True)

    assert len(term.visualizer.calls) == 1
    _, kwargs = term.visualizer.calls[0]
    assert torch.equal(kwargs["translations"], term.points_w.view(-1, 3))
    assert torch.equal(kwargs["environment_ids"], torch.arange(num_envs).repeat_interleave(num_points))


def test_lift_terminations_detect_nonfinite_state() -> None:
    """Invalid rigid, deformable, cable, and robot state should request an immediate reset."""
    environment_ids = torch.arange(3)
    rigid_positions = torch.tensor(
        [
            [0.5, 0.0, 0.1],
            [float("nan"), 0.0, 0.1],
            [1.1, 0.0, 0.1],
        ]
    )
    rigid_quaternions = torch.zeros((3, 4))
    rigid_quaternions[:, 3] = 1.0
    rigid_quaternions[1, 0] = float("nan")
    rigid_velocities = torch.zeros((3, 6))
    deformable_positions = torch.tensor(
        [
            [[0.5, 0.0, 0.1], [0.6, 0.0, 0.1]],
            [[0.5, 0.0, 0.1], [float("nan"), 0.0, 0.1]],
            [[1.1, 0.0, 0.1], [0.6, 0.0, 0.1]],
        ]
    )
    cable_poses = torch.zeros((3, 2, 7))
    cable_poses[:, :, 2] = 0.1
    cable_poses[1, 0, 0] = float("inf")
    cable_poses[2, 0, 1] = 0.6
    joint_pos = torch.zeros((3, 2))
    joint_pos[1, 0] = float("nan")
    joint_vel = torch.zeros((3, 2))
    joint_vel[2, 1] = 2.0
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=_tensor_data(joint_pos),
            joint_vel=_tensor_data(joint_vel),
            joint_vel_limits=_tensor_data(torch.ones((3, 2))),
        )
    )
    scene = _FakeScene(
        environment_ids,
        object=SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=_tensor_data(rigid_positions),
                root_quat_w=_tensor_data(rigid_quaternions),
                root_vel_w=_tensor_data(rigid_velocities),
            )
        ),
        deformable=SimpleNamespace(data=SimpleNamespace(nodal_pos_w=_tensor_data(deformable_positions))),
        cable=SimpleNamespace(data=SimpleNamespace(segment_pose_w=_tensor_data(cable_poses))),
        robot=robot,
    )
    env = SimpleNamespace(scene=scene)

    rigid_termination = object.__new__(mdp.out_of_bound)
    rigid_termination._object = scene["object"]
    rigid_termination._origins = scene.env_origins
    rigid_termination._lower = scene.env_origins.clone()
    rigid_termination._upper = scene.env_origins.clone()
    rigid_termination._cached_axis = [None, None, None]
    assert torch.equal(
        rigid_termination(env, in_bound_range={"x": (0.0, 1.0), "y": (-0.5, 0.5), "z": (-0.02, 1.0)}),
        torch.tensor([False, True, True]),
    )
    assert torch.equal(
        mdp.deformable_outside_bounds(env, x_bounds=(0.0, 1.0), y_bounds=(-0.5, 0.5), z_bounds=(-0.02, 1.0)),
        torch.tensor([False, True, True]),
    )
    assert torch.equal(
        mdp.cable_outside_bounds(env, x_bounds=(0.0, 1.0), y_bounds=(-0.5, 0.5), z_bounds=(-0.02, 1.0)),
        torch.tensor([False, True, True]),
    )
    assert torch.equal(mdp.joint_vel_out_of_sim_limit(env), torch.tensor([False, True, True]))


def test_soft_asset_rewards_zero_nonfinite_state() -> None:
    """A terminal invalid physics state should not propagate NaN into the RL batch."""
    environment_ids = torch.arange(2)
    deformable_root_positions = torch.tensor([[0.5, 0.0, 0.2], [float("nan"), 0.0, 0.2]])
    deformable_nodal_positions = deformable_root_positions[:, None, :].repeat(1, 2, 1)
    cable_poses = torch.zeros((2, 2, 7))
    cable_poses[:, :, :3] = torch.tensor([0.5, 0.0, 0.2])
    cable_poses[1, 0, 0] = float("inf")
    ee_positions = torch.tensor([[[0.5, 0.0, 0.2]], [[0.5, 0.0, 0.2]]])
    identity_quat = torch.zeros((2, 4))
    identity_quat[:, 3] = 1.0
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=_tensor_data(torch.zeros((2, 3))),
            root_quat_w=_tensor_data(identity_quat),
        )
    )
    scene = _FakeScene(
        environment_ids,
        deformable=SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=_tensor_data(deformable_root_positions),
                nodal_pos_w=_tensor_data(deformable_nodal_positions),
            )
        ),
        cable=SimpleNamespace(data=SimpleNamespace(segment_pose_w=_tensor_data(cable_poses))),
        ee_frame=SimpleNamespace(data=SimpleNamespace(target_pos_w=_tensor_data(ee_positions))),
        robot=robot,
    )
    command = torch.zeros((2, 7))
    command[:, 0] = 0.5
    command[:, 2] = 0.2
    env = SimpleNamespace(scene=scene, command_manager=SimpleNamespace(get_command=lambda _name: command))

    rewards = {
        "deformable_lifting": mdp.deformable_lifting(env, std=0.1, minimal_height=0.02),
        "deformable_ee_distance": mdp.deformable_ee_distance(env, std=0.1),
        "deformable_com_ee_distance": mdp.deformable_com_ee_distance(env, std=0.1),
        "deformable_com_goal_reached": mdp.deformable_com_goal_reached(
            env, minimal_height=0.0, command_name="deformable_pose", success_threshold=0.05
        ),
        "cable_lifting": mdp.cable_lifting(env, std=0.1, minimal_height=0.02),
        "cable_ee_distance": mdp.cable_ee_distance(env, std=0.1),
        "cable_segment_goal_reached": mdp.cable_segment_goal_reached(
            env, command_name="cable_pose", success_threshold=0.05, segment_index=0
        ),
    }
    for name, reward in rewards.items():
        assert torch.isfinite(reward).all(), name
        assert reward[0] > 0.0, name
        assert reward[1] == 0.0, name


def test_rigid_object_rewards_zero_nonfinite_state() -> None:
    """A terminal invalid rigid-object state should not propagate NaN into the RL batch."""
    num_envs = 2
    environment_ids = torch.arange(num_envs)
    identity_quat = torch.zeros((num_envs, 4))
    identity_quat[:, 3] = 1.0
    object_positions = torch.tensor([[0.5, 0.0, 0.2], [float("nan"), 0.0, 0.2]])
    object_quaternions = identity_quat.clone()
    object_quaternions[1, 0] = float("nan")
    robot_positions = torch.zeros((num_envs, 3))
    robot_body_positions = object_positions.nan_to_num()[:, None, :]
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=_tensor_data(robot_body_positions),
            root_pos_w=_tensor_data(robot_positions),
            root_quat_w=_tensor_data(identity_quat),
            root_link_quat_w=_tensor_data(identity_quat),
        )
    )
    rigid_object = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=_tensor_data(object_positions),
            root_quat_w=_tensor_data(object_quaternions),
        )
    )
    scene = _FakeScene(environment_ids, robot=robot, object=rigid_object)
    contact_forces = torch.zeros((num_envs, 3))
    contact_forces[:, 0] = 1.0
    contact_sensor = SimpleNamespace(data=SimpleNamespace(normal_force_matrix_w=_tensor_data(contact_forces)))
    scene.sensors = {"thumb": contact_sensor, "finger": contact_sensor}
    command = torch.zeros((num_envs, 7))
    command[:, :3] = object_positions.nan_to_num()
    command[:, 3:] = identity_quat
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=scene,
        command_manager=SimpleNamespace(get_command=lambda _name: command),
    )
    robot_cfg = SimpleNamespace(name="robot", body_ids=[0])
    object_cfg = SimpleNamespace(name="object")

    success = object.__new__(mdp.success_reward)
    success.succeeded = torch.zeros(num_envs, dtype=torch.bool)
    position_progress = object.__new__(mdp.position_command_progress)
    position_progress.best_error = torch.full((num_envs,), float("inf"))
    position_progress._prev_command = None
    orientation_progress = object.__new__(mdp.orientation_command_progress)
    orientation_progress.best_error = torch.full((num_envs,), float("inf"))
    orientation_progress._prev_command = None

    rewards = {
        "object_ee_distance": mdp.object_ee_distance(
            env,
            std=0.4,
            thumb_name="thumb",
            finger_names=["finger"],
            asset_cfg=robot_cfg,
            object_cfg=object_cfg,
        ),
        "success": success(
            env,
            command_name="object_pose",
            asset_cfg=robot_cfg,
            align_asset_cfg=object_cfg,
            pos_std=0.05,
            rot_std=0.5,
            thumb_name="thumb",
            finger_names=["finger"],
        ),
        "position_progress": position_progress(
            env,
            command_name="object_pose",
            asset_cfg=robot_cfg,
            align_asset_cfg=object_cfg,
            min_improvement=0.0025,
            thumb_name="thumb",
            finger_names=["finger"],
        ),
        "orientation_progress": orientation_progress(
            env,
            command_name="object_pose",
            asset_cfg=robot_cfg,
            align_asset_cfg=object_cfg,
            min_improvement=0.015,
            thumb_name="thumb",
            finger_names=["finger"],
        ),
    }
    for name, reward in rewards.items():
        assert torch.isfinite(reward).all(), name
        assert reward[1] == 0.0, name
    assert torch.isinf(position_progress.best_error[1])
    assert torch.isinf(orientation_progress.best_error[1])
