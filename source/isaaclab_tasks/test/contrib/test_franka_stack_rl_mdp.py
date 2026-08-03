# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.stack.mdp import (
    ResetBufferedGripperAction,
    WorkspaceBoundedRelativeJointPositionAction,
    WorkspaceBoundedRelativeJointPositionActionCfg,
    curriculums,
    goal_context,
    observations,
    reset_events,
    rewards,
    robot_state,
    runtime_state,
    terminations,
)


class _DummyIndexProxy:
    """Minimal test double for Isaac Lab's cached joint-index proxy."""

    def __init__(self, indices: list[int]):
        self.torch = torch.tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.torch)


class _DummyRobot:
    def __init__(
        self,
        joint_positions: torch.Tensor,
        ee_positions: torch.Tensor | None = None,
        hand_velocities: torch.Tensor | None = None,
    ):
        self.data = SimpleNamespace(joint_pos=SimpleNamespace(torch=joint_positions))
        if ee_positions is not None:
            body_positions = ee_positions.clone()
            body_positions[:, 2] -= 0.1034
            body_orientations = torch.zeros((joint_positions.shape[0], 1, 4))
            body_orientations[:, 0, 3] = 1.0
            self.data.body_pos_w = SimpleNamespace(torch=body_positions.unsqueeze(1))
            self.data.body_quat_w = SimpleNamespace(torch=body_orientations)
            if hand_velocities is None:
                hand_velocities = torch.zeros((joint_positions.shape[0], 6))
            self.data.body_vel_w = SimpleNamespace(torch=hand_velocities.unsqueeze(1))

    def find_joints(self, _joint_names, preserve_order=False, *, as_proxy=False):
        del preserve_order
        indices = [0, 1]
        return (_DummyIndexProxy(indices) if as_proxy else indices), ["finger_left", "finger_right"]

    def find_bodies(self, _body_names):
        return [0], ["panda_hand"]


class _DummyResetRobot:
    def __init__(self, num_envs: int):
        self.data = SimpleNamespace(default_joint_pos=SimpleNamespace(torch=torch.zeros((num_envs, 9))))

    def set_joint_position_target_index(self, *, target, env_ids):
        self.position_target = target.clone()

    def set_joint_velocity_target_index(self, *, target, env_ids):
        self.velocity_target = target.clone()

    def write_joint_position_to_sim_index(self, *, position, env_ids):
        self.joint_position = position.clone()

    def write_joint_velocity_to_sim_index(self, *, velocity, env_ids):
        self.joint_velocity = velocity.clone()


class _DummyActionRobot:
    def __init__(self, num_envs: int):
        self.num_joints = 7
        self.num_base_dofs = 0
        self.data = SimpleNamespace(
            joint_pos=SimpleNamespace(torch=torch.zeros((num_envs, 7))),
            default_joint_pos=SimpleNamespace(torch=torch.zeros((num_envs, 7))),
            gravity_compensation_forces=SimpleNamespace(torch=torch.arange(1.0, 8.0).unsqueeze(0).repeat(num_envs, 1)),
            soft_joint_pos_limits=SimpleNamespace(
                torch=torch.tensor([[-2.0, 2.0]] * 7).unsqueeze(0).repeat(num_envs, 1, 1)
            ),
        )

    def find_joints(self, _joint_names, preserve_order=False, *, as_proxy=False):
        del preserve_order
        indices = list(range(7))
        return (_DummyIndexProxy(indices) if as_proxy else indices), [f"panda_joint{i + 1}" for i in range(7)]

    def set_joint_position_target_index(self, *, target, joint_ids):
        self.position_target = target.clone()

    def set_joint_effort_target_index(self, *, target, joint_ids):
        self.effort_target = target.clone()


class _DummyResetCube:
    def __init__(self, num_envs: int):
        default_pose = torch.zeros((num_envs, 7))
        default_pose[:, 3] = 1.0
        self.data = SimpleNamespace(default_root_pose=SimpleNamespace(torch=default_pose))

    def write_root_pose_to_sim_index(self, *, root_pose, env_ids):
        self.root_pose = root_pose.clone()

    def write_root_velocity_to_sim_index(self, *, root_velocity, env_ids):
        self.root_velocity = root_velocity.clone()


class _DummyResetScene(dict):
    def __init__(self, num_envs: int):
        super().__init__()
        self.env_origins = torch.zeros((num_envs, 3))


def _dummy_rigid_object(positions: torch.Tensor):
    velocities = torch.zeros((positions.shape[0], 6))
    poses = torch.zeros((positions.shape[0], 7))
    poses[:, :3] = positions
    poses[:, 3] = 1.0
    return SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=positions),
            root_quat_w=SimpleNamespace(torch=poses[:, 3:7]),
            root_pose_w=SimpleNamespace(torch=poses),
            root_vel_w=SimpleNamespace(torch=velocities),
        )
    )


def test_stack_success_context_terminates_after_stability_and_emits_one_reward_pulse():
    num_envs = 2
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        episode_length_buf=torch.full((num_envs,), 10, dtype=torch.long),
        extras={},
        scene={"robot": object()},
    )
    context = goal_context.StableOrderInvariantStackGoal(SimpleNamespace(params={}), env)
    env.termination_manager = SimpleNamespace(
        get_term_cfg=lambda _name: SimpleNamespace(func=context),
    )
    cube_velocities = torch.zeros((num_envs, 3, 6))

    with (
        patch.object(goal_context, "order_invariant_stack_progress", return_value=torch.full((num_envs,), 2.0)),
        patch.object(
            goal_context,
            "_order_invariant_cube_state",
            return_value=(torch.zeros((num_envs, 3, 3)), cube_velocities),
        ),
        patch.object(goal_context, "_gripper_is_released", return_value=torch.ones(num_envs, dtype=torch.bool)),
    ):
        # Solved, physically released rows intentionally bootstrap the success
        # signal; the adaptive reset sampler expands from this easy frontier.
        for _ in range(4):
            assert not context(env, minimum_episode_steps=3, hold_steps=5, maximum_cube_velocity=0.1).any()
            assert not context.new_success.any()

        assert not context(env, minimum_episode_steps=3, hold_steps=5, maximum_cube_velocity=0.1).any()
        assert context.is_success.all()
        assert context.new_success.all()
        assert context.ever_success.all()
        assert torch.equal(rewards.stack_success_pulse(env), torch.ones(num_envs))

        assert not context(env, minimum_episode_steps=3, hold_steps=5, maximum_cube_velocity=0.1).any()
        assert not context.new_success.any()
        assert not rewards.stack_success_pulse(env).any()

    env.step_dt = 0.02
    env.episode_length_buf = torch.tensor([4, 5])
    assert torch.equal(
        terminations.success_after_minimum_horizon(env, minimum_episode_length_s=0.1),
        torch.tensor([False, True]),
    )


def test_irrecoverable_stack_failure_ignores_timeout_and_success():
    env = SimpleNamespace(
        reset_terminated=torch.tensor([True, True, False, False]),
        termination_manager=SimpleNamespace(
            get_term=lambda _name: torch.tensor([True, False, False, False]),
        ),
    )

    assert torch.equal(rewards.irrecoverable_stack_failure(env), torch.tensor([0.0, 1.0, 0.0, 0.0]))


def test_finite_joint_velocity_penalty_cannot_emit_nan_or_infinity():
    joint_velocities = torch.tensor(
        [
            [float("nan"), 2.0],
            [float("inf"), -float("inf")],
            [3.0, 4.0],
        ]
    )
    robot = SimpleNamespace(data=SimpleNamespace(joint_vel=SimpleNamespace(torch=joint_velocities)))
    env = SimpleNamespace(scene={"robot": robot})
    asset_cfg = SimpleNamespace(name="robot", joint_ids=[0, 1])

    penalty = rewards.finite_joint_velocity_l2(env, asset_cfg=asset_cfg, maximum_velocity=3.0)

    assert torch.isfinite(penalty).all()
    assert torch.equal(penalty, torch.tensor([4.0, 18.0, 18.0]))


def test_relative_action_uses_measured_joints_once_and_clamps_to_workspace():
    robot = _DummyActionRobot(num_envs=2)
    env = SimpleNamespace(num_envs=2, device="cpu", scene={"robot": robot})
    cfg = WorkspaceBoundedRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.05,
        max_delta=0.10,
        workspace_lower=(-0.20,) * 7,
        workspace_upper=(0.15,) * 7,
    )
    action = WorkspaceBoundedRelativeJointPositionAction(cfg, env)

    action.process_actions(torch.full((2, 7), 4.0))
    assert torch.allclose(action.processed_actions, torch.full((2, 7), 0.10))

    robot.data.joint_pos.torch[:] = 0.04
    for _ in range(5):
        action.apply_actions()
        assert torch.allclose(robot.position_target, torch.full((2, 7), 0.10))

    action.process_actions(torch.full((2, 7), -1.0))
    assert torch.allclose(action.processed_actions, torch.full((2, 7), -0.01))
    action.process_actions(torch.full((2, 7), 4.0))
    assert torch.allclose(action.processed_actions, torch.full((2, 7), 0.14))
    action.process_actions(torch.full((2, 7), 4.0))
    assert torch.allclose(action.processed_actions, torch.full((2, 7), 0.14))

    robot.data.joint_pos.torch[:] = 0.30
    action.reset(torch.tensor([0]))
    action.process_actions(torch.zeros((2, 7)))
    assert torch.allclose(action.processed_actions[0], torch.full((7,), 0.15))
    assert torch.allclose(action.processed_actions[1], torch.full((7,), 0.15))


def test_relative_action_applies_finite_model_based_gravity_feedforward():
    robot = _DummyActionRobot(num_envs=2)
    robot.data.gravity_compensation_forces = SimpleNamespace(
        torch=torch.tensor(
            [
                [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0],
                [-1.0, -2.0, -3.0, -4.0, -5.0, float("inf"), -7.0],
            ]
        )
    )
    env = SimpleNamespace(num_envs=2, device="cpu", scene={"robot": robot})
    cfg = WorkspaceBoundedRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        gravity_compensation=True,
    )
    action = WorkspaceBoundedRelativeJointPositionAction(cfg, env)

    action.process_actions(torch.zeros((2, 7)))
    action.apply_actions()

    assert torch.equal(
        robot.effort_target,
        torch.tensor(
            [
                [1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0],
                [-1.0, -2.0, -3.0, -4.0, -5.0, 0.0, -7.0],
            ]
        ),
    )


def test_relative_action_waits_for_a_nearby_commanded_grasp_to_settle():
    term = WorkspaceBoundedRelativeJointPositionAction.__new__(WorkspaceBoundedRelativeJointPositionAction)
    term.cfg = SimpleNamespace(
        grasp_close_interlock_steps=2,
        grasp_close_distance=0.05,
        grasp_cube_names=("cube_1", "cube_2", "cube_3"),
        gripper_action_index=-1,
    )
    term._grasp_close_steps = torch.zeros(1, dtype=torch.long)
    term._env = SimpleNamespace(
        action_manager=SimpleNamespace(action=torch.tensor([[0.0] * 7 + [-1.0]])),
        scene={
            "cube_1": _dummy_rigid_object(torch.tensor([[0.70, 0.0, 0.02]])),
            "cube_2": _dummy_rigid_object(torch.tensor([[0.50, 0.0, 0.04]])),
            "cube_3": _dummy_rigid_object(torch.tensor([[0.30, 0.0, 0.02]])),
        },
    )
    command = torch.full((1, 7), 0.25)

    with patch(
        "isaaclab_tasks.contrib.stack.mdp.actions.franka_end_effector_pose",
        return_value=(torch.tensor([[0.50, 0.0, 0.04]]), torch.tensor([[0.0, 0.0, 0.0, 1.0]])),
    ):
        assert torch.count_nonzero(term._apply_grasp_close_interlock(command)) == 0
        assert torch.count_nonzero(term._apply_grasp_close_interlock(command)) == 0
        assert torch.equal(term._apply_grasp_close_interlock(command), command)

        term._env.action_manager.action[:, -1] = 1.0
        assert torch.equal(term._apply_grasp_close_interlock(command), command)
        assert torch.equal(term._grasp_close_steps, torch.zeros(1, dtype=torch.long))


def test_acquisition_resets_hold_physical_gripper_closed_only_during_reset_grace():
    term = ResetBufferedGripperAction.__new__(ResetBufferedGripperAction)
    term.cfg = SimpleNamespace(
        force_close_steps=5,
        clip=None,
    )
    term._raw_actions = torch.zeros((6, 1))
    term._processed_actions = torch.zeros((6, 2))
    term._open_command = torch.full((2,), 0.04)
    term._close_command = torch.full((2,), 0.014)
    term._env = SimpleNamespace(
        stack_reset_held_cube_ids=torch.tensor([1, 2, 1, -1, -1, -1]),
        episode_length_buf=torch.tensor([0, 4, 5, 0, 0, 0]),
    )

    term.process_actions(torch.ones((6, 1)))

    assert torch.allclose(term.raw_actions, torch.ones((6, 1)))
    assert torch.allclose(term.processed_actions[0], term._close_command)
    assert torch.allclose(term.processed_actions[1], term._close_command)
    assert torch.allclose(term.processed_actions[2], term._open_command)
    assert torch.allclose(term.processed_actions[3], term._open_command)
    assert torch.allclose(term.processed_actions[4], term._open_command)
    assert torch.allclose(term.processed_actions[5], term._open_command)


def test_role_conditioned_observation_is_color_invariant_and_temporally_stable():
    role_positions = torch.tensor(
        (
            (0.48, 0.00, 0.0205),
            (0.48, -0.10, 0.0205),
            (0.48, 0.10, 0.0205),
        )
    )
    # Environment one assigns the same physical roles to different colored
    # assets. Gathering through role_to_cube must produce the same policy state.
    asset_positions = (
        torch.stack((role_positions[0], role_positions[1])),
        torch.stack((role_positions[1], role_positions[2])),
        torch.stack((role_positions[2], role_positions[0])),
    )
    cubes = tuple(_dummy_rigid_object(positions) for positions in asset_positions)
    ee = torch.tensor([[0.48, -0.10, 0.08], [0.48, -0.10, 0.08]])
    env = SimpleNamespace(
        device="cpu",
        num_envs=2,
        cfg=SimpleNamespace(gripper_joint_names=["panda_finger_.*"]),
        scene=_DummyResetScene(2),
        stack_reset_role_to_cube=torch.tensor(((0, 1, 2), (2, 0, 1))),
    )
    env.scene.update(
        {
            "robot": _DummyRobot(torch.full((2, 2), 0.04), ee),
            "cube_1": cubes[0],
            "cube_2": cubes[1],
            "cube_3": cubes[2],
        }
    )

    observation = observations.role_conditioned_stack_obs(env)

    assert observation.shape == (2, 64)
    assert torch.allclose(observation[0], observation[1])
    assert torch.allclose(observation[:, :3], role_positions[0].expand(2, -1))


def test_franka_ee_velocity_reports_the_offset_tool_center_twist():
    hand_velocity = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    robot = _DummyRobot(
        torch.full((1, 2), 0.04),
        torch.tensor([[0.48, 0.0, 0.10]]),
        hand_velocities=hand_velocity,
    )
    env = SimpleNamespace(
        device="cpu",
        num_envs=1,
        scene={"robot": robot},
    )

    velocity = observations.franka_ee_velocity(env)

    tool_offset = torch.tensor([[0.0, 0.0, 0.1034]])
    expected_linear = hand_velocity[:, :3] + torch.linalg.cross(
        hand_velocity[:, 3:],
        tool_offset,
    )
    assert velocity.shape == (1, 6)
    assert torch.allclose(velocity[:, :3], expected_linear)
    assert torch.allclose(velocity[:, 3:], hand_velocity[:, 3:])


def test_franka_ee_axes_use_a_continuous_six_dimensional_orientation():
    robot = _DummyRobot(
        torch.full((1, 2), 0.04),
        torch.tensor([[0.48, 0.0, 0.10]]),
    )
    env = SimpleNamespace(
        device="cpu",
        num_envs=1,
        scene={"robot": robot},
    )

    axes = observations.franka_ee_axes(env)

    assert axes.shape == (1, 6)
    assert torch.allclose(axes, torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))


def test_robot_neutral_tool_state_resolves_each_body_and_offset_independently():
    body_positions = torch.tensor([[[0.2, 0.0, 0.3], [0.5, 0.1, 0.4]]])
    body_orientations = torch.zeros((1, 2, 4))
    body_orientations[..., 3] = 1.0
    body_velocities = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])

    class _TwoToolRobot:
        def __init__(self):
            self.data = SimpleNamespace(
                body_pos_w=SimpleNamespace(torch=body_positions),
                body_quat_w=SimpleNamespace(torch=body_orientations),
                body_vel_w=SimpleNamespace(torch=body_velocities),
            )

        def find_bodies(self, body_name):
            body_id = {"wrist": 0, "allegro_mount": 1}[body_name]
            return [body_id], [body_name]

    env = SimpleNamespace(
        device="cpu",
        num_envs=1,
        scene={"robot": _TwoToolRobot()},
    )
    wrist_position, _ = robot_state.end_effector_pose(
        env,
        body_name="wrist",
        body_offset=(0.0, 0.0, 0.0),
    )
    tool_position, _ = robot_state.end_effector_pose(
        env,
        body_name="allegro_mount",
        body_offset=(0.0, 0.0, 0.1),
    )
    tool_velocity = robot_state.end_effector_velocity(
        env,
        body_name="allegro_mount",
        body_offset=(0.0, 0.0, 0.1),
    )

    assert torch.allclose(wrist_position, body_positions[:, 0])
    assert torch.allclose(tool_position, body_positions[:, 1] + torch.tensor([[0.0, 0.0, 0.1]]))
    assert torch.allclose(
        tool_velocity[:, :3],
        body_velocities[:, 1, :3] + torch.linalg.cross(body_velocities[:, 1, 3:], torch.tensor([[0.0, 0.0, 0.1]])),
    )
    assert torch.allclose(tool_velocity[:, 3:], body_velocities[:, 1, 3:])
    assert len(env._stack_end_effector_cache) == 2


def test_two_finger_posture_features_and_release_are_joint_direction_aware():
    open_posture = (0.0,) * 8
    closed_posture = (0.1, 1.0, -1.0, 0.5, 0.1, -1.0, 1.0, -0.5)
    midpoint = tuple(
        0.5 * (open_position + closed_position)
        for open_position, closed_position in zip(open_posture, closed_posture, strict=True)
    )
    # A small-range proximal joint can be far from its command while the
    # physically relevant synergy has converged. Per-joint amin would report
    # 0.459 here; least-squares projection correctly remains above 0.99.
    contact_posture = (0.0459, 1.0, -1.0, 0.5, 0.0459, -1.0, 1.0, -0.5)
    joint_positions = torch.tensor((open_posture, closed_posture, midpoint, contact_posture))
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=joint_positions)))
    gripper_cfg = SimpleNamespace(name="robot", joint_ids=list(range(8)))
    env = SimpleNamespace(scene={"robot": robot})

    posture = observations.two_finger_gripper_posture(
        env,
        open_joint_positions=open_posture,
        closed_joint_positions=closed_posture,
        asset_cfg=gripper_cfg,
    )
    released = rewards._gripper_is_released(
        env,
        robot,
        gripper_cfg=gripper_cfg,
        open_joint_positions=open_posture,
        closed_joint_positions=closed_posture,
    )
    release_progress = rewards._gripper_release_progress(
        env,
        robot,
        gripper_cfg=gripper_cfg,
        open_joint_positions=open_posture,
        closed_joint_positions=closed_posture,
    )
    closed = rewards._gripper_is_closed(
        env,
        robot,
        maximum_finger_position=0.03,
        gripper_cfg=gripper_cfg,
        open_joint_positions=open_posture,
        closed_joint_positions=closed_posture,
    )

    assert torch.allclose(
        posture,
        torch.tensor(((0.0, 0.0), (1.0, 1.0), (0.5, 0.5), (0.9976, 0.9976))),
        atol=1.0e-4,
    )
    assert torch.equal(released, torch.tensor((True, False, False, False)))
    assert torch.allclose(release_progress, torch.tensor((1.0, 0.0, 0.5, 0.0)))
    assert torch.equal(closed, torch.tensor((False, True, False, True)))


def test_legacy_franka_release_thresholds_remain_strict():
    finger_positions = torch.tensor(((0.023, 0.024), (0.024, 0.024), (0.029, 0.029), (0.030, 0.029)))
    robot = _DummyRobot(finger_positions)
    env = SimpleNamespace(
        cfg=SimpleNamespace(gripper_joint_names=["panda_finger_.*"]),
    )

    assert torch.equal(
        rewards._gripper_is_released(env, robot),
        torch.tensor((False, True, True, True)),
    )
    assert torch.equal(
        rewards._gripper_is_closed(env, robot, maximum_finger_position=0.03),
        torch.tensor((True, True, True, False)),
    )


def test_stable_stack_goal_accepts_posture_based_release_configuration():
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        episode_length_buf=torch.tensor([10]),
        extras={},
        scene={"robot": object()},
    )
    context = goal_context.StableOrderInvariantStackGoal(SimpleNamespace(params={}), env)
    cube_velocities = torch.zeros((1, 3, 6))
    gripper_cfg = SimpleNamespace(name="robot", joint_ids=list(range(8)))
    closed_posture = tuple(float(index) for index in range(8))
    open_posture = tuple(-value for value in closed_posture)

    with (
        patch.object(goal_context, "order_invariant_stack_progress", return_value=torch.tensor([2.0])),
        patch.object(
            goal_context,
            "_order_invariant_cube_state",
            return_value=(torch.zeros((1, 3, 3)), cube_velocities),
        ),
        patch.object(
            goal_context,
            "_gripper_is_released",
            return_value=torch.tensor([True]),
        ) as released_mock,
    ):
        context(
            env,
            hold_steps=1,
            gripper_cfg=gripper_cfg,
            open_gripper_joint_positions=open_posture,
            closed_gripper_joint_positions=closed_posture,
        )

    released_mock.assert_called_once_with(
        env,
        env.scene["robot"],
        minimum_finger_position=0.023,
        gripper_cfg=gripper_cfg,
        open_joint_positions=open_posture,
        closed_joint_positions=closed_posture,
        finger_joint_counts=(4, 4),
        maximum_gripper_closure=0.2,
    )
    assert context.is_success.item()


def test_stack_success_uses_explicit_cube_com_alignment_thresholds():
    lower_com = torch.zeros((3, 3))
    upper_com = torch.tensor(
        [
            [0.019, 0.000, 0.040],
            [0.021, 0.000, 0.040],
            [0.000, 0.000, 0.051],
        ]
    )

    aligned = rewards.cube_com_pair_aligned(upper_com, lower_com)

    assert rewards.STACK_COM_XY_THRESHOLD == 0.02
    assert rewards.STACK_COM_HEIGHT_THRESHOLD == 0.01
    assert torch.equal(aligned, torch.tensor([True, False, False]))


def _build_reset_state_table_for_test() -> reset_events.StackResetStateTable:
    term = reset_events.StackResetStateTable.__new__(reset_events.StackResetStateTable)
    term._env = SimpleNamespace(device="cpu")
    term._arm_anchors = torch.tensor(reset_events._STATE_TABLE_ARM_POSES, dtype=torch.float32)
    term._build_table(
        closed_finger_position=0.020,
        placed_finger_position=0.021,
        open_finger_position=0.040,
    )
    term._validate_table()
    term._neighbor_indices = term._build_neighbor_graph(k=8, partition_by_goal=False)
    return term


def test_order_invariant_reset_table_is_complete_supported_and_physically_connected():
    term = _build_reset_state_table_for_test()

    assert term.row_count == 6786
    assert torch.equal(
        torch.bincount(term.recipe_ids),
        torch.tensor([324, 594, 1170, 594, 594, 594, 1170, 594, 1152]),
    )
    assert torch.equal(torch.bincount(term.layout_ids), torch.full((18,), 377))
    assert torch.all(term._role_positions[..., 2] >= 0.0205 - 1.0e-6)
    assert torch.all(term._goal_pairs == 2)
    held_rows = term._held_roles >= 0
    held_positions = term._role_positions[held_rows, term._held_roles[held_rows]]
    expected_held_positions = torch.stack(
        tuple(reset_events._franka_tool_position(joints) for joints in term._arm_positions[held_rows])
    )
    assert torch.allclose(held_positions, expected_held_positions, atol=1.0e-6)
    expected_target_by_recipe = torch.tensor([10.0, 10.0, 8.0, 6.0, 6.0, 5.0, 3.0, 1.0, 1.0])
    assert torch.equal(term._target_potentials, expected_target_by_recipe[term.recipe_ids])

    final_release_rows = term.recipe_ids == int(reset_events.StackResetRecipe.FINAL_RELEASE)
    held_final_rows = final_release_rows & (term._held_roles == 2)
    released_final_rows = final_release_rows & (term._held_roles == -1)
    assert held_final_rows.sum() == 18
    assert released_final_rows.sum() == 306
    assert torch.allclose(term._finger_positions[held_final_rows], torch.full((18,), 0.020))
    assert torch.all(term._finger_positions[released_final_rows] >= 0.021)
    assert torch.all(term._finger_positions[released_final_rows] <= 0.040)
    assert torch.any(term._finger_positions[released_final_rows] > 0.035)
    for layout_rows in torch.nonzero(released_final_rows, as_tuple=False).flatten().reshape(18, 17):
        assert torch.allclose(term._progress[layout_rows], torch.linspace(0.0, 1.0, 17))
        assert torch.allclose(
            term._role_positions[layout_rows],
            term._role_positions[layout_rows[:1]].expand(17, -1, -1),
        )

    for recipe in (reset_events.StackResetRecipe.SECOND_PLACE, reset_events.StackResetRecipe.FIRST_PLACE):
        endpoints = (term.recipe_ids == int(recipe)) & (term._progress == 1.0)
        assert endpoints.sum() == 18
        assert torch.all(term._held_roles[endpoints] == -1)
        assert torch.allclose(term._finger_positions[endpoints], torch.full((18,), 0.021))

    pair_ready_rows = term.recipe_ids == int(reset_events.StackResetRecipe.PAIR_READY)
    assert pair_ready_rows.sum() == 594
    for layout_rows in torch.nonzero(pair_ready_rows, as_tuple=False).flatten().reshape(18, 33):
        assert torch.allclose(term._progress[layout_rows], torch.linspace(0.0, 1.0, 33))
        assert torch.all(term._held_roles[layout_rows] == -1)
        assert torch.allclose(term._finger_positions[layout_rows], torch.full((33,), 0.040))
        # The bridge moves only the arm. The supported first pair and the
        # remaining table cube stay in the same physical state.
        assert torch.allclose(
            term._role_positions[layout_rows],
            term._role_positions[layout_rows[:1]].expand(33, -1, -1),
        )

    for recipe, held_role in (
        (reset_events.StackResetRecipe.SECOND_PICK, 2),
        (reset_events.StackResetRecipe.FIRST_PICK, 1),
    ):
        pick_rows = torch.nonzero(term.recipe_ids == int(recipe), as_tuple=False).flatten().reshape(18, 33)
        for layout_rows in pick_rows:
            assert torch.allclose(term._progress[layout_rows], torch.linspace(0.0, 1.0, 33))
            assert torch.all(term._finger_positions[layout_rows[:17]] == 0.040)
            assert torch.all(term._finger_positions[layout_rows[17:]] < 0.040)
            closing_finger_positions = term._finger_positions[layout_rows[17:]]
            assert torch.all(closing_finger_positions[1:] < closing_finger_positions[:-1])
            assert term._finger_positions[layout_rows[-1]] == pytest.approx(0.020)
            assert torch.all(term._held_roles[layout_rows[:-1]] == -1)
            assert term._held_roles[layout_rows[-1]] == held_role

    for recipe, held_role in (
        (reset_events.StackResetRecipe.SECOND_TRANSPORT, 2),
        (reset_events.StackResetRecipe.FIRST_TRANSPORT, 1),
    ):
        transport_rows = torch.nonzero(term.recipe_ids == int(recipe), as_tuple=False).flatten().reshape(18, 65)
        for layout_rows in transport_rows:
            assert torch.allclose(
                term._progress[layout_rows],
                torch.linspace(0.0, 1.0, 65),
            )
            assert torch.all(term._held_roles[layout_rows] == held_role)
            # The first half is a monotonic vertical lift at the source. The
            # second half continues forward from its endpoint into transport.
            held_heights = term._role_positions[layout_rows, held_role, 2]
            assert torch.all(held_heights[1:33] > held_heights[:32])

    table_rows = term.recipe_ids == int(reset_events.StackResetRecipe.TABLE)
    assert table_rows.sum() == 1152
    table_positions = term._role_positions[table_rows]
    pair_indices = torch.triu_indices(3, 3, offset=1)
    table_distances = torch.cdist(table_positions[..., :2], table_positions[..., :2])
    assert torch.all(table_distances[:, pair_indices[0], pair_indices[1]] >= 0.085 - 1.0e-6)
    assert torch.allclose(table_positions[..., 2], torch.full((1152, 3), 0.0205))
    assert table_positions[..., 0].amin() >= 0.40
    assert table_positions[..., 0].amax() <= 0.56
    assert table_positions[..., 1].amin() >= -0.18
    assert table_positions[..., 1].amax() <= 0.18
    # Every physical role, including the future base, spans the complete
    # workspace instead of occupying a prescribed center/left/right slot.
    assert torch.all(table_positions[..., :2].std(dim=0) > 0.035)
    base_positions = term._role_positions[:, 0, :2]
    expected_stack_sites = torch.tensor(reset_events._STATE_TABLE_ANCHORS)
    for stack_site in expected_stack_sites:
        assert torch.any(torch.linalg.vector_norm(base_positions - stack_site, dim=1) < 1.0e-6)


def test_full_goal_reset_graph_uses_one_deployment_objective():
    term = _build_reset_state_table_for_test()

    neighbors = term._build_neighbor_graph(k=8, partition_by_goal=False)
    assert torch.all(term._goal_pairs == 2)
    assert neighbors.shape == (term.row_count, 8)
    assert torch.all((neighbors >= 0) & (neighbors < term.row_count))


def test_reset_table_keeps_full_goal_and_preserves_each_rows_learning_target():
    term = _build_reset_state_table_for_test()
    num_envs = 4
    robot = _DummyResetRobot(num_envs)
    cubes = tuple(_DummyResetCube(num_envs) for _ in range(3))
    scene = _DummyResetScene(num_envs)
    local_rows = torch.nonzero(term._target_potentials < 10.0, as_tuple=False).flatten()[:num_envs]
    env = SimpleNamespace(
        device="cpu",
        scene=scene,
        stack_reset_row_ids=local_rows.clone(),
        stack_reset_recipes=torch.zeros(num_envs, dtype=torch.long),
        stack_previous_reset_recipes=torch.zeros(num_envs, dtype=torch.long),
        stack_reset_goal_pairs=torch.zeros(num_envs, dtype=torch.long),
        stack_reset_target_potentials=torch.zeros(num_envs),
        stack_continue_to_final=torch.zeros(num_envs, dtype=torch.bool),
        stack_reset_held_cube_ids=torch.full((num_envs,), -1, dtype=torch.long),
        stack_reset_role_to_cube=torch.arange(3).repeat(num_envs, 1),
        stack_reset_initialized=torch.zeros(num_envs, dtype=torch.bool),
        stack_previous_reset_initialized=torch.zeros(num_envs, dtype=torch.bool),
        stack_reset_sample_counts=torch.zeros(term.row_count, dtype=torch.long),
    )
    term._env = env
    term._robot = robot
    term._cubes = cubes
    term._arm_joint_ids = list(range(7))
    term._finger_joint_ids = [7, 8]
    term._role_permutations = torch.tensor(tuple(reset_events.permutations(range(3))), dtype=torch.long)
    env_ids = torch.arange(num_envs)

    term(
        env,
        env_ids,
        fixed_role_permutation=0,
        fixed_continue_to_final=False,
        force_full_goal=True,
    )

    assert env.stack_continue_to_final.all()
    assert torch.equal(env.stack_reset_goal_pairs, torch.full((num_envs,), 2))
    assert torch.equal(env.stack_reset_target_potentials, term._target_potentials[local_rows])
    for cube in cubes:
        assert not hasattr(cube, "permanent_wrench_composer")


def test_stack_reset_runtime_state_owns_legacy_tensor_aliases():
    env = SimpleNamespace(num_envs=3, device="cpu")

    state = runtime_state.create_stack_reset_runtime_state(env, row_count=7)

    assert env.stack_reset_state is state
    assert env.stack_reset_row_ids is state.row_ids
    assert env.stack_reset_role_to_cube is state.role_to_cube
    assert env.stack_reset_target_potentials is state.target_potentials
    assert env.stack_reset_sample_counts is state.sample_counts


def test_reset_table_continuously_randomizes_table_robot_and_cube_states():
    term = _build_reset_state_table_for_test()
    num_envs = 32
    robot = _DummyResetRobot(num_envs)
    cubes = tuple(_DummyResetCube(num_envs) for _ in range(3))
    scene = _DummyResetScene(num_envs)
    table_row = int(torch.nonzero(term.recipe_ids == int(reset_events.StackResetRecipe.TABLE))[0])
    env = SimpleNamespace(
        device="cpu",
        scene=scene,
        stack_reset_row_ids=torch.full((num_envs,), table_row, dtype=torch.long),
        stack_reset_recipes=torch.zeros(num_envs, dtype=torch.long),
        stack_previous_reset_recipes=torch.zeros(num_envs, dtype=torch.long),
        stack_reset_goal_pairs=torch.zeros(num_envs, dtype=torch.long),
        stack_reset_target_potentials=torch.zeros(num_envs),
        stack_continue_to_final=torch.zeros(num_envs, dtype=torch.bool),
        stack_reset_held_cube_ids=torch.full((num_envs,), -1, dtype=torch.long),
        stack_reset_role_to_cube=torch.arange(3).repeat(num_envs, 1),
        stack_reset_initialized=torch.zeros(num_envs, dtype=torch.bool),
        stack_previous_reset_initialized=torch.zeros(num_envs, dtype=torch.bool),
        stack_reset_sample_counts=torch.zeros(term.row_count, dtype=torch.long),
    )
    term._env = env
    term._robot = robot
    term._cubes = cubes
    term._arm_joint_ids = list(range(7))
    term._finger_joint_ids = [7, 8]
    term._role_permutations = torch.tensor(tuple(reset_events.permutations(range(3))), dtype=torch.long)

    torch.manual_seed(7)
    term(
        env,
        torch.arange(num_envs),
        fixed_role_permutation=0,
        force_full_goal=True,
        arm_joint_noise_range=0.020,
        table_arm_joint_noise_range=0.080,
        table_cube_planar_translation_range=0.015,
        table_cube_rotation_range=0.45,
    )

    cached_arm = term._arm_positions[table_row].expand(num_envs, -1)
    arm_delta = robot.joint_position[:, :7] - cached_arm
    assert torch.all(torch.abs(arm_delta) <= 0.080 + 1.0e-6)
    assert torch.unique(robot.joint_position[:, :7], dim=0).shape[0] == num_envs

    cube_positions = torch.stack(tuple(cube.root_pose[:, :3] for cube in cubes), dim=1)
    cached_positions = term._role_positions[table_row].expand(num_envs, -1, -1)
    # Coherent translation and rotation preserve safe pairwise separation.
    expected_distances = torch.cdist(cached_positions[:, :, :2], cached_positions[:, :, :2])
    actual_distances = torch.cdist(cube_positions[:, :, :2], cube_positions[:, :, :2])
    assert torch.allclose(actual_distances, expected_distances, atol=1.0e-6)
    assert torch.unique(cube_positions[:, :, :2].reshape(num_envs, -1), dim=0).shape[0] == num_envs
    assert torch.allclose(cube_positions[:, :, 2], cached_positions[:, :, 2])
    for cube in cubes:
        assert torch.allclose(torch.linalg.vector_norm(cube.root_pose[:, 3:7], dim=1), torch.ones(num_envs))


def test_reset_table_joint_randomization_preserves_held_cube_fk_attachment():
    term = _build_reset_state_table_for_test()
    num_envs = 16
    robot = _DummyResetRobot(num_envs)
    cubes = tuple(_DummyResetCube(num_envs) for _ in range(3))
    scene = _DummyResetScene(num_envs)
    final_row = int(torch.nonzero(term.recipe_ids == int(reset_events.StackResetRecipe.FINAL_RELEASE))[0])
    env = SimpleNamespace(
        device="cpu",
        scene=scene,
        stack_reset_row_ids=torch.full((num_envs,), final_row, dtype=torch.long),
        stack_reset_recipes=torch.zeros(num_envs, dtype=torch.long),
        stack_previous_reset_recipes=torch.zeros(num_envs, dtype=torch.long),
        stack_reset_goal_pairs=torch.zeros(num_envs, dtype=torch.long),
        stack_reset_target_potentials=torch.zeros(num_envs),
        stack_continue_to_final=torch.zeros(num_envs, dtype=torch.bool),
        stack_reset_held_cube_ids=torch.full((num_envs,), -1, dtype=torch.long),
        stack_reset_role_to_cube=torch.arange(3).repeat(num_envs, 1),
        stack_reset_initialized=torch.zeros(num_envs, dtype=torch.bool),
        stack_previous_reset_initialized=torch.zeros(num_envs, dtype=torch.bool),
        stack_reset_sample_counts=torch.zeros(term.row_count, dtype=torch.long),
    )
    term._env = env
    term._robot = robot
    term._cubes = cubes
    term._arm_joint_ids = list(range(7))
    term._finger_joint_ids = [7, 8]
    term._role_permutations = torch.tensor(tuple(reset_events.permutations(range(3))), dtype=torch.long)

    torch.manual_seed(11)
    term(
        env,
        torch.arange(num_envs),
        fixed_role_permutation=0,
        force_full_goal=True,
        arm_joint_noise_range=0.020,
        table_arm_joint_noise_range=0.080,
        table_cube_planar_translation_range=0.015,
        table_cube_rotation_range=0.45,
    )

    expected_tool_positions = reset_events._franka_tool_position(robot.joint_position[:, :7])
    assert torch.allclose(cubes[2].root_pose[:, :3], expected_tool_positions, atol=1.0e-6)
    assert torch.equal(env.stack_reset_held_cube_ids, torch.full((num_envs,), 2))


def test_reset_table_curriculum_records_learning_and_full_task_outcomes_separately():
    reset_term = _build_reset_state_table_for_test()
    num_envs = 4
    curriculum = curriculums.StackResetTableCurriculum.__new__(curriculums.StackResetTableCurriculum)
    curriculum._reset_term = reset_term
    curriculum._sampler = curriculums._EpsilonResetTableSampler(
        reset_term.row_count,
        "cpu",
        monitored_history_len=50,
        target_success_rate=0.5,
        kappa=1.0,
        epsilon=1.0e-4,
    )
    curriculum._continuation_attempts = torch.zeros((), dtype=torch.long)
    curriculum._continuation_successes = torch.zeros((), dtype=torch.long)
    curriculum._table_sampling_probability = 0.35
    curriculum._table_rows = reset_term.recipe_ids == int(reset_events.StackResetRecipe.TABLE)
    curriculum._full_task_attempts_by_row = torch.zeros(reset_term.row_count, dtype=torch.long)
    curriculum._full_task_successes_by_row = torch.zeros(reset_term.row_count, dtype=torch.long)
    learning_succeeded = torch.tensor([True, True, False, False])
    final_succeeded = torch.tensor([False, True, False, False])

    def get_context(name):
        values = learning_succeeded if name == "learning_progress_context" else final_succeeded
        return SimpleNamespace(func=SimpleNamespace(ever_success=values))

    env = SimpleNamespace(
        device="cpu",
        stack_reset_initialized=torch.ones(num_envs, dtype=torch.bool),
        stack_reset_row_ids=torch.arange(num_envs),
        stack_continue_to_final=torch.tensor([False, True, False, True]),
        episode_length_buf=torch.ones(num_envs, dtype=torch.long),
        termination_manager=SimpleNamespace(get_term_cfg=get_context),
    )

    metrics = curriculum(env, torch.arange(num_envs))

    assert torch.equal(curriculum._sampler.total_attempts[:4], torch.tensor([1, 1, 1, 1]))
    assert torch.equal(curriculum._sampler.total_successes[:4], torch.tensor([1, 1, 0, 0]))
    assert metrics["full_task_attempts"] == 4
    assert metrics["full_task_success_rate"] == 0.25
    assert metrics["batch_success_rate"] == 0.5
    assert metrics["batch_full_task_success_rate"] == 0.25
    assert metrics["table_probability"] == pytest.approx(0.35)
    probabilities = curriculum._sampling_probabilities()
    intermediate_probabilities = probabilities.clone()
    intermediate_probabilities[curriculum._table_rows] = 0.0
    layout_probability = torch.zeros(len(reset_events._STATE_TABLE_LAYOUTS))
    layout_probability.scatter_add_(0, reset_term.layout_ids, intermediate_probabilities)
    assert torch.allclose(
        layout_probability,
        torch.full_like(layout_probability, 0.65 / len(reset_events._STATE_TABLE_LAYOUTS)),
    )


def test_epsilon_reset_sampler_checkpoint_round_trip():
    term = _build_reset_state_table_for_test()
    sampler = curriculums._EpsilonResetTableSampler(
        term.row_count,
        "cpu",
        monitored_history_len=50,
        target_success_rate=0.5,
        kappa=1.0,
        epsilon=1.0e-4,
    )
    rows = sampler.sample(41)
    sampler.record(rows, rows.remainder(3) == 0)
    state = sampler.get_state()

    restored = curriculums._EpsilonResetTableSampler(
        term.row_count,
        "cpu",
        monitored_history_len=50,
        target_success_rate=0.5,
        kappa=1.0,
        epsilon=1.0e-4,
    )
    restored.set_state(state)

    for key, value in state.items():
        assert torch.equal(restored.get_state()[key], value)

    malformed = dict(state)
    malformed["history_pointer"] = torch.full_like(state["history_pointer"], 50)
    with pytest.raises(ValueError, match="invalid history pointer"):
        restored.set_state(malformed)


def test_epsilon_reset_sampler_uses_exact_rolling_success_window():
    sampler = curriculums._EpsilonResetTableSampler(
        3,
        "cpu",
        monitored_history_len=4,
        target_success_rate=0.5,
        kappa=1.0,
        epsilon=1.0e-4,
    )

    sampler.record(
        torch.tensor([0, 0, 0, 0, 1, 1]),
        torch.tensor([True, True, False, False, True, True]),
    )
    assert torch.allclose(sampler.success_rates, torch.tensor([0.5, 1.0, 0.0]))

    # Five new failures replace the complete four-outcome row-0 history.
    sampler.record(torch.zeros(5, dtype=torch.long), torch.zeros(5, dtype=torch.bool))
    assert sampler.success_rates[0] == 0.0
    assert sampler.history_size[0] == 4
    assert sampler.history_success_count[0] == 0
    assert torch.equal(sampler.success_history[0], torch.zeros(4, dtype=torch.bool))


def test_epsilon_reset_sampler_focuses_half_solved_rows_without_starving_others():
    sampler = curriculums._EpsilonResetTableSampler(
        3,
        "cpu",
        monitored_history_len=50,
        target_success_rate=0.5,
        kappa=1.0,
        epsilon=1.0e-4,
    )
    sampler.success_rates.copy_(torch.tensor([0.0, 0.5, 1.0]))

    probabilities = sampler.probabilities()

    assert probabilities[1] > 0.999
    assert probabilities[0] > 0.0
    assert probabilities[2] > 0.0
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0))


def test_role_conditioned_potential_allows_either_side_cube_order():
    base = torch.tensor([[0.5, 0.0, 0.0205], [0.5, 0.0, 0.0205]])
    first_side = torch.tensor([[0.5, 0.0, 0.0605], [0.5, -0.10, 0.0205]])
    second_side = torch.tensor([[0.5, 0.10, 0.0205], [0.5, 0.0, 0.0605]])
    ee = torch.tensor([[0.5, 0.10, 0.08], [0.5, -0.10, 0.08]])
    env = SimpleNamespace(
        device="cpu",
        num_envs=2,
        scene={
            "robot": _DummyRobot(torch.full((2, 2), 0.04), ee),
            "cube_1": _dummy_rigid_object(base),
            "cube_2": _dummy_rigid_object(first_side),
            "cube_3": _dummy_rigid_object(second_side),
        },
        cfg=SimpleNamespace(
            gripper_joint_names=["panda_finger_.*"],
            gripper_open_val=0.04,
            gripper_threshold=0.005,
        ),
        stack_reset_role_to_cube=torch.arange(3).repeat(2, 1),
    )

    potential = rewards.role_conditioned_stack_potential(env)

    assert torch.all((potential > 5.0) & (potential < 6.0))
    assert torch.allclose(potential[0], potential[1])


def test_reset_learning_progress_bootstraps_rows_without_terminating():
    term = goal_context.StackResetLearningProgress.__new__(goal_context.StackResetLearningProgress)
    term._initial_potential = torch.zeros(2)
    term._target_potential = torch.zeros(2)
    term.is_success = torch.zeros(2, dtype=torch.bool)
    term.new_success = torch.zeros(2, dtype=torch.bool)
    term.ever_success = torch.zeros(2, dtype=torch.bool)
    term._no_termination = torch.zeros(2, dtype=torch.bool)
    env = SimpleNamespace(
        stack_reset_target_potentials=torch.tensor([1.0, 8.0]),
        episode_length_buf=torch.tensor([3, 3]),
    )

    with patch.object(
        goal_context,
        "role_conditioned_stack_potential",
        side_effect=(
            torch.tensor([0.2, 7.9]),
            torch.tensor([0.9, 8.2]),
            torch.tensor([1.1, 8.3]),
        ),
    ):
        term._env = env
        term.reset()
        first = term(env)
        second = term(env)

    assert torch.allclose(term._target_potential, torch.tensor([1.0, 8.15]))
    assert not first.any()
    assert not second.any()
    assert term.ever_success.all()
