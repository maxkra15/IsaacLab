# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registration tests for the Kuka-Allegro cube-stack variant."""

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.stack import mdp
from isaaclab_tasks.contrib.stack.config.franka.agents.rsl_rl_ppo_cfg import FrankaStackPPORunnerCfg
from isaaclab_tasks.contrib.stack.config.kuka_allegro.agents.rsl_rl_ppo_cfg import (
    KukaAllegroGaussianDistribution,
    KukaAllegroStackPPORunnerCfg,
)
from isaaclab_tasks.contrib.stack.mdp import curriculums, goal_context, observations, reset_events
from isaaclab_tasks.contrib.stack.mdp.actions import ResetPreservingRelativeJointPositionAction
from isaaclab_tasks.contrib.stack.mdp.actions_cfg import ResetPreservingRelativeJointPositionActionCfg
from isaaclab_tasks.contrib.stack.mdp.kuka_allegro_reset import (
    KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES,
    KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_CONTACT_POSES,
    KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_OPEN_COMMANDS,
    KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_OPEN_POSES,
    KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_PRELOAD_COMMANDS,
    KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_TOOL_OFFSETS,
    KUKA_ALLEGRO_GRASP_PAIR_CLOSED_COMMANDS,
    KUKA_ALLEGRO_GRASP_PAIR_RESET_CLOSED_COMMANDS,
    KUKA_ALLEGRO_LARGE_CUBE_EDGE_LENGTH,
    KUKA_ALLEGRO_LARGE_CUBE_GRASP_PAIR_CLOSED_COMMANDS,
    KUKA_ALLEGRO_LARGE_CUBE_GRASP_PAIR_OPEN_COMMANDS,
    KUKA_ALLEGRO_LARGE_CUBE_PALM_TO_HELD_CUBE_QUATERNIONS_XYZW,
    KUKA_ALLEGRO_LARGE_CUBE_RESTING_HEIGHT,
    KUKA_ALLEGRO_STACK_ARM_POSES,
    KUKA_ALLEGRO_STACK_ARM_WORKSPACE_LOWER,
    KUKA_ALLEGRO_STACK_ARM_WORKSPACE_UPPER,
    kuka_allegro_pinch_position,
    matrix_from_quaternion_xyzw,
)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


class _DummyIndexProxy:
    """Minimal test double for Isaac Lab's cached joint-index proxy."""

    def __init__(self, indices: list[int]):
        self.torch = torch.tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.torch)


class _DummyScene(dict):
    def __init__(self, num_envs: int):
        super().__init__()
        self.env_origins = torch.zeros((num_envs, 3))


class _DummyKukaAllegro:
    body_names = ("palm_link", "index_tip", "thumb_tip")

    def __init__(self, num_envs: int):
        joint_pos = torch.linspace(-0.2, 0.2, 15).repeat(num_envs, 1)
        joint_vel = torch.linspace(-0.1, 0.1, 15).repeat(num_envs, 1)
        body_pos = torch.zeros((num_envs, 3, 3))
        body_pos[:, 0] = torch.tensor((0.45, 0.0, 0.10))
        pinch_center = body_pos[:, 0] + torch.tensor((0.0570965, -0.0375159, 0.0498749))
        body_pos[:, 1] = pinch_center + torch.tensor((0.0, 0.018, 0.0))
        body_pos[:, 2] = pinch_center + torch.tensor((0.0, -0.018, 0.0))
        body_pos[-1, 1, 0] += 0.01
        body_quat = torch.zeros((num_envs, 3, 4))
        body_quat[..., 3] = 1.0
        self.data = SimpleNamespace(
            joint_pos=SimpleNamespace(torch=joint_pos),
            default_joint_pos=SimpleNamespace(torch=torch.zeros_like(joint_pos)),
            joint_vel=SimpleNamespace(torch=joint_vel),
            default_joint_vel=SimpleNamespace(torch=torch.zeros_like(joint_vel)),
            body_pos_w=SimpleNamespace(torch=body_pos),
            body_quat_w=SimpleNamespace(torch=body_quat),
            body_vel_w=SimpleNamespace(torch=torch.zeros((num_envs, 3, 6))),
        )

    def find_bodies(self, body_name):
        body_id = self.body_names.index(body_name)
        return [body_id], [body_name]


class _DummyFullKukaAllegro:
    body_names = (
        "palm_link",
        "index_biotac_tip",
        "middle_biotac_tip",
        "ring_biotac_tip",
        "thumb_biotac_tip",
    )

    def __init__(self, num_envs: int):
        joint_pos = torch.linspace(-0.3, 0.3, 23).repeat(num_envs, 1)
        joint_vel = torch.linspace(-0.2, 0.2, 23).repeat(num_envs, 1)
        body_pos = torch.zeros((num_envs, 5, 3))
        body_pos[:, 0] = torch.tensor((0.45, 0.0, 0.10))
        for body_id, offset in enumerate(
            ((0.08, 0.04, 0.02), (0.09, 0.01, 0.02), (0.08, -0.02, 0.02), (0.04, -0.05, 0.01)),
            start=1,
        ):
            body_pos[:, body_id] = body_pos[:, 0] + torch.tensor(offset)
        body_quat = torch.zeros((num_envs, 5, 4))
        body_quat[..., 3] = 1.0
        self.data = SimpleNamespace(
            joint_pos=SimpleNamespace(torch=joint_pos),
            default_joint_pos=SimpleNamespace(torch=torch.zeros_like(joint_pos)),
            joint_vel=SimpleNamespace(torch=joint_vel),
            default_joint_vel=SimpleNamespace(torch=torch.zeros_like(joint_vel)),
            soft_joint_pos_limits=SimpleNamespace(torch=torch.tensor((-1.0, 1.0)).repeat(num_envs, 23, 1)),
            body_pos_w=SimpleNamespace(torch=body_pos),
            body_quat_w=SimpleNamespace(torch=body_quat),
            body_vel_w=SimpleNamespace(torch=torch.zeros((num_envs, 5, 6))),
        )
        self.num_joints = 23
        self.position_targets = None

    def find_joints(self, joint_names, preserve_order=False, *, as_proxy=False):
        del preserve_order
        all_joint_names = tuple(f"iiwa7_joint_{joint_id}" for joint_id in range(1, 8)) + (
            KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES
        )
        joint_ids = [all_joint_names.index(joint_name) for joint_name in joint_names]
        indices = _DummyIndexProxy(joint_ids) if as_proxy else joint_ids
        return indices, [all_joint_names[joint_id] for joint_id in joint_ids]

    def find_bodies(self, body_name):
        body_id = self.body_names.index(body_name)
        return [body_id], [body_name]

    def set_joint_position_target_index(self, target, joint_ids):
        self.position_targets = (target.clone(), joint_ids)


def _dummy_cube(positions: torch.Tensor, quaternions: torch.Tensor):
    return SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=positions),
            root_quat_w=SimpleNamespace(torch=quaternions),
            root_vel_w=SimpleNamespace(torch=torch.zeros((positions.shape[0], 6))),
        )
    )


def _dummy_observation_env():
    num_envs = 3
    positions = (
        torch.tensor((0.45, 0.00, 0.0205)).repeat(num_envs, 1),
        torch.tensor((0.45, -0.10, 0.0205)).repeat(num_envs, 1),
        torch.tensor((0.45, 0.10, 0.0205)).repeat(num_envs, 1),
    )
    sqrt_half = 2.0**-0.5
    q_0 = torch.tensor((0.0, 0.0, 0.0, 1.0))
    q_45 = torch.tensor((0.0, 0.0, 0.38268343, 0.92387953))
    q_90 = torch.tensor((0.0, 0.0, sqrt_half, sqrt_half))
    q_180 = torch.tensor((0.0, 0.0, 1.0, 0.0))
    # Row one permutes physical assets while preserving role orientations.
    quaternions = (
        torch.stack((q_0, q_90, q_45)),
        torch.stack((q_90, q_180, q_90)),
        torch.stack((q_180, q_0, q_180)),
    )
    scene = _DummyScene(num_envs)
    scene.update(
        {
            "robot": _DummyKukaAllegro(num_envs),
            **{
                f"cube_{cube_id + 1}": _dummy_cube(cube_positions, cube_quaternions)
                for cube_id, (cube_positions, cube_quaternions) in enumerate(zip(positions, quaternions, strict=True))
            },
        }
    )
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=scene,
        stack_reset_role_to_cube=torch.tensor(((0, 1, 2), (2, 0, 1), (0, 1, 2))),
    )


def _small_diverse_reset_curriculum():
    """Build a CPU-only curriculum with five distinct non-table reset modes."""
    # The first mode deliberately has three rows. Every subsequent mode differs
    # from it in exactly one dimension, making omissions from the stratum key
    # observable as unequal probability mass.
    reset_term = SimpleNamespace(
        row_count=8,
        recipe_names=("phase", "table"),
        recipe_ids=torch.tensor((0, 0, 0, 0, 0, 0, 0, 1)),
        layout_ids=torch.tensor((0, 0, 0, 0, 0, 0, 0, 1)),
        layout_count=2,
        grasp_pair_ids=torch.tensor((0, 0, 0, 0, 0, 0, 1, 2)),
        orientation_bin_ids=torch.tensor((0, 0, 0, 0, 0, 1, 0, 7)),
        # Resolved angles deliberately merge here. Sampling must use the
        # separately authored categories below, not a mixed metadata key.
        tilt_azimuth_bin_ids=torch.zeros(8, dtype=torch.long),
        authored_tilt_azimuth_bin_ids=torch.tensor((0, 0, 0, 0, 1, 0, 0, 7)),
        tilt_magnitude_bin_ids=torch.tensor((0, 0, 0, 1, 0, 0, 0, 3)),
    )
    curriculum = curriculums.StackResetTableCurriculum.__new__(curriculums.StackResetTableCurriculum)
    curriculum._reset_term = reset_term
    curriculum._sampler = curriculums._EpsilonResetTableSampler(
        reset_term.row_count,
        "cpu",
        monitored_history_len=4,
        target_success_rate=0.5,
        kappa=1.0,
        epsilon=1.0e-4,
    )
    curriculum._table_sampling_probability = 0.35
    curriculum._balance_recipes = True
    curriculum._balance_reset_modes = True
    curriculum._global_sampling = False
    curriculum._table_rows = reset_term.recipe_ids == 1
    curriculum._layout_count = reset_term.layout_count
    curriculum._recipe_rows = tuple(reset_term.recipe_ids == recipe_id for recipe_id in range(2))
    curriculum._grasp_pair_rows = tuple(reset_term.grasp_pair_ids == pair_id for pair_id in range(3))
    curriculum._orientation_rows = tuple(
        reset_term.orientation_bin_ids == orientation_id for orientation_id in range(8)
    )
    curriculum._tilt_azimuth_rows = tuple(
        reset_term.authored_tilt_azimuth_bin_ids == azimuth_id for azimuth_id in range(8)
    )
    curriculum._resolved_tilt_azimuth_rows = tuple(
        reset_term.tilt_azimuth_bin_ids == azimuth_id for azimuth_id in range(8)
    )
    curriculum._tilt_magnitude_rows = tuple(
        reset_term.tilt_magnitude_bin_ids == magnitude_id for magnitude_id in range(4)
    )
    curriculum._continuation_attempts = torch.zeros((), dtype=torch.long)
    curriculum._continuation_successes = torch.zeros((), dtype=torch.long)
    curriculum._full_task_attempts_by_row = torch.zeros(reset_term.row_count, dtype=torch.long)
    curriculum._full_task_successes_by_row = torch.zeros(reset_term.row_count, dtype=torch.long)
    return curriculum


def test_pair_close_targets_preload_contact_without_interpenetrating_resets():
    reset_positions = torch.tensor(KUKA_ALLEGRO_GRASP_PAIR_RESET_CLOSED_COMMANDS)
    close_targets = torch.tensor(KUKA_ALLEGRO_GRASP_PAIR_CLOSED_COMMANDS)
    delta = close_targets - reset_positions

    assert reset_positions.shape == close_targets.shape == (3, 16)
    assert torch.equal(
        torch.nonzero(delta, as_tuple=False),
        torch.tensor(((0, 1), (1, 4), (1, 5), (2, 8))),
    )
    assert torch.allclose(delta[delta != 0.0], torch.tensor((0.04, 0.02, 0.04, 0.02)))
    # Thumb and parked fingers remain at their geometrically authored reset.
    assert torch.allclose(delta[:, 12:16], torch.zeros((3, 4)))


def test_kuka_allegro_stack_registration_exposes_production_task_only():
    task_id = "IsaacContrib-Stack-Cube-KukaAllegro-RL"
    spec = gym.spec(task_id)
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")

    registered_kuka_tasks = {
        task_spec.id
        for task_spec in gym.registry.values()
        if task_spec.id.startswith("IsaacContrib-Stack-Cube-KukaAllegro")
    }
    assert registered_kuka_tasks == {task_id}
    assert spec.kwargs["env_cfg_entry_point"].endswith(":KukaAllegroCubeStackRLEnvCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":KukaAllegroStackPPORunnerCfg")
    assert issubclass(KukaAllegroStackPPORunnerCfg, FrankaStackPPORunnerCfg)
    assert agent_cfg.experiment_name == "kuka_allegro_stack_full_hand"
    assert agent_cfg.actor.distribution_cfg.class_name.endswith(":KukaAllegroGaussianDistribution")

    assert len(env_cfg.actions.arm_action.joint_names) == 1
    assert isinstance(env_cfg.actions.gripper_action, ResetPreservingRelativeJointPositionActionCfg)
    assert tuple(env_cfg.actions.gripper_action.joint_names) == KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES
    assert env_cfg.actions.gripper_action.scale == pytest.approx(0.10)
    assert env_cfg.actions.gripper_action.max_delta == pytest.approx(0.10)
    assert env_cfg.actions.gripper_action.preserve_order is True
    assert tuple(env_cfg.actions.gripper_action.reset_preload_joint_names) == KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES
    assert (
        env_cfg.actions.gripper_action.reset_preload_commands_by_pair
        == KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_PRELOAD_COMMANDS
    )
    assert env_cfg.actions.gripper_action.reset_open_commands_by_pair == KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_OPEN_COMMANDS
    assert env_cfg.actions.gripper_action.preload_release_threshold == pytest.approx(0.5)
    assert env_cfg.actions.gripper_action.preload_release_steps == 2
    # Seven arm joints plus sixteen independently actuated hand joints.
    assert 7 + len(env_cfg.actions.gripper_action.joint_names) == 23
    assert env_cfg.rewards.reset_progress.func is mdp.stack_success_pulse
    assert env_cfg.rewards.reset_progress.params == {"context_term_name": "learning_progress_context"}
    assert env_cfg.rewards.reset_progress.weight == pytest.approx(25.0)
    assert env_cfg.rewards.success.weight == pytest.approx(100.0)

    actuator = env_cfg.scene.robot.actuators["kuka_allegro_actuators"]
    all_hand_expression = "(index|middle|ring|thumb)_joint_(0|1|2|3)"
    assert actuator.stiffness[all_hand_expression] == pytest.approx(20.0)
    assert actuator.damping[all_hand_expression] == pytest.approx(0.5)
    assert env_cfg.curriculum.reset_sampling.params["balance_recipes"] is False
    assert env_cfg.curriculum.reset_sampling.params["balance_reset_modes"] is False
    assert env_cfg.curriculum.reset_sampling.params["global_sampling"] is True
    assert env_cfg.events.reset_from_state_buffer.func is (
        reset_events.FullHandLargeCubeDiverseKukaAllegroStackResetStateTable
    )
    assert env_cfg.events.reset_from_state_buffer.params["table_arm_joint_noise_range"] == pytest.approx(0.04)
    assert env_cfg.events.reset_from_state_buffer.params["table_target_potential"] == pytest.approx(1.05)
    assert env_cfg.scene.plane is None
    assert env_cfg.observations.policy.object.params["grasp_pair_tool_offsets"] == (
        KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_TOOL_OFFSETS
    )
    assert env_cfg.observations.policy.gripper_pos.params["open_joint_positions_by_pair"] == (
        KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_OPEN_POSES
    )
    assert env_cfg.observations.policy.gripper_pos.params["closed_joint_positions_by_pair"] == (
        KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_CONTACT_POSES
    )
    assert all(
        cube.spawn.mass_props.mass == pytest.approx(0.05)
        for cube in (env_cfg.scene.cube_1, env_cfg.scene.cube_2, env_cfg.scene.cube_3)
    )
    assert all(
        cube.spawn.size == (0.08, 0.08, 0.08)
        and cube.init_state.pos[2] == pytest.approx(KUKA_ALLEGRO_LARGE_CUBE_RESTING_HEIGHT)
        for cube in (env_cfg.scene.cube_1, env_cfg.scene.cube_2, env_cfg.scene.cube_3)
    )
    assert env_cfg.terminations.progress_context.func is mdp.StableFullHandOrderInvariantStackGoal
    assert env_cfg.terminations.progress_context.params["maximum_cube_linear_velocity"] == pytest.approx(0.10)
    assert env_cfg.terminations.progress_context.params["maximum_cube_angular_velocity"] == pytest.approx(1.0)
    assert env_cfg.terminations.progress_context.params["minimum_fingertip_cube_clearance"] == pytest.approx(0.010)
    assert env_cfg.terminations.progress_context.params["fingertip_cfg"].body_names == (
        "index_biotac_tip",
        "middle_biotac_tip",
        "ring_biotac_tip",
        "thumb_biotac_tip",
    )

    assert agent_cfg.algorithm.entropy_coef == pytest.approx(1.0e-4)
    assert agent_cfg.algorithm.learning_rate == pytest.approx(5.0e-5)
    assert agent_cfg.algorithm.schedule == "fixed"
    assert agent_cfg.actor.distribution_cfg.init_std == pytest.approx(0.35)
    assert agent_cfg.actor.distribution_cfg.hand_init_std == pytest.approx(0.15)
    assert agent_cfg.actor.distribution_cfg.std_range == pytest.approx((0.08, 0.45))


def test_full_hand_release_clearance_uses_oriented_cube_surfaces():
    points = torch.tensor(
        (
            (
                (0.00, 0.00, 0.00),
                (0.04, 0.00, 0.00),
                (0.06, 0.00, 0.00),
                (0.06, 0.06, 0.00),
            ),
        )
    )
    centers = torch.zeros((1, 1, 3))
    yaw_45 = torch.tensor((0.0, 0.0, 0.38268343, 0.92387953)).reshape(1, 1, 4)

    distances = goal_context._oriented_box_point_signed_distance(
        points,
        centers,
        yaw_45,
        half_extent=0.04,
    )

    assert distances.shape == (1, 4, 1)
    assert distances[0, 0, 0] == pytest.approx(-0.04)
    # At 45 degrees, the world-X face is sqrt(2) * half_extent away.
    assert distances[0, 1, 0] < 0.0
    assert distances[0, 2, 0] == pytest.approx(0.06 - 2.0**0.5 * 0.04, abs=1.0e-6)
    assert distances[0, 3, 0] == pytest.approx(2.0**0.5 * 0.06 - 0.04, abs=1.0e-6)


def test_full_hand_success_rejects_nearby_fingertips_and_spinning_cubes():
    num_envs = 3
    identity = torch.tensor((0.0, 0.0, 0.0, 1.0)).repeat(num_envs, 1)
    cube_positions = (
        torch.tensor((0.45, 0.0, 0.037)).repeat(num_envs, 1),
        torch.tensor((0.45, 0.0, 0.117)).repeat(num_envs, 1),
        torch.tensor((0.45, 0.0, 0.197)).repeat(num_envs, 1),
    )
    cubes = tuple(_dummy_cube(positions, identity) for positions in cube_positions)
    cubes[0].data.root_vel_w.torch[2, 3] = 2.0
    fingertip_positions = torch.full((num_envs, 4, 3), 0.8)
    # Five millimeters from cube one's +X face is below the 10 mm release margin.
    fingertip_positions[1, 0] = torch.tensor((0.495, 0.0, 0.037))
    scene = {
        "robot": SimpleNamespace(data=SimpleNamespace(body_pos_w=SimpleNamespace(torch=fingertip_positions))),
        **{f"cube_{cube_id + 1}": cube for cube_id, cube in enumerate(cubes)},
    }
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        episode_length_buf=torch.full((num_envs,), 10, dtype=torch.long),
        extras={},
        scene=scene,
    )
    context = goal_context.StableFullHandOrderInvariantStackGoal(SimpleNamespace(params={}), env)
    fingertip_cfg = SimpleNamespace(
        name="robot",
        body_names=("index", "middle", "ring", "thumb"),
        body_ids=[0, 1, 2, 3],
    )

    context(env, hold_steps=1, fingertip_cfg=fingertip_cfg)

    assert torch.equal(context.is_success, torch.tensor((True, False, False)))


def test_full_hand_observations_expose_every_joint_and_fingertip():
    task_id = "IsaacContrib-Stack-Cube-KukaAllegro-RL"
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")

    assert env_cfg.observations.policy.pinch_joint_pos is None
    assert env_cfg.observations.policy.pinch_joint_vel is None
    assert env_cfg.observations.policy.pinch_tip_positions is None
    assert tuple(env_cfg.observations.policy.hand_joint_pos.params["asset_cfg"].joint_names) == (
        KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES
    )
    assert tuple(env_cfg.observations.policy.hand_joint_vel.params["asset_cfg"].joint_names) == (
        KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES
    )
    assert env_cfg.observations.policy.hand_tip_positions.params["body_cfg"].body_names == (
        "index_biotac_tip",
        "middle_biotac_tip",
        "ring_biotac_tip",
        "thumb_biotac_tip",
    )
    assert env_cfg.observations.policy.grasp_pair.func is mdp.grasp_pair_one_hot
    assert env_cfg.observations.policy.grasp_pair.params == {"num_pairs": 3}

    num_envs = 3
    scene = _DummyScene(num_envs)
    scene["robot"] = _DummyFullKukaAllegro(num_envs)
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=scene,
        stack_reset_state=SimpleNamespace(grasp_pair_ids=torch.tensor((0, 2, 1))),
    )
    hand_cfg = SimpleNamespace(name="robot", joint_ids=list(range(7, 23)))
    hand_joint_pos = mdp.joint_pos(env, asset_cfg=hand_cfg)
    hand_joint_vel = mdp.joint_vel(env, asset_cfg=hand_cfg)
    tip_positions = observations.body_positions_relative_to_tool(
        env,
        body_cfg=SimpleNamespace(name="robot", body_ids=[1, 2, 3, 4]),
        tool_body_name="palm_link",
        tool_offset=(0.0, 0.0, 0.0),
    )

    assert hand_joint_pos.shape == (num_envs, 16)
    assert hand_joint_vel.shape == (num_envs, 16)
    assert tip_positions.shape == (num_envs, 12)
    assert torch.equal(
        mdp.grasp_pair_one_hot(env),
        torch.tensor(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))),
    )


def test_full_hand_action_preserves_reset_pose_and_uses_measured_relative_targets():
    num_envs = 2
    robot = _DummyFullKukaAllegro(num_envs)
    env = SimpleNamespace(num_envs=num_envs, device="cpu", scene={"robot": robot})
    cfg = ResetPreservingRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES),
        preserve_order=True,
        scale=0.1,
        max_delta=0.1,
        joint_limit_margin=0.02,
    )
    action = ResetPreservingRelativeJointPositionAction(cfg, env)
    reset_pose = robot.data.joint_pos.torch[:, 7:].clone()

    action.reset()
    assert torch.allclose(action.processed_actions, reset_pose)

    raw_actions = torch.stack((torch.ones(16), -torch.ones(16)))
    action.process_actions(raw_actions)
    expected_target = reset_pose + 0.1 * raw_actions
    assert torch.allclose(action.raw_actions, raw_actions)
    assert torch.allclose(action.processed_actions, expected_target)

    # Match DexSuite and Isaac Lab's standard relative action: the next target
    # is based on the latest measured joint position, never an unobserved
    # persistent command from the previous policy step.
    robot.data.joint_pos.torch[:, 7:] += 0.05
    action.process_actions(torch.zeros_like(raw_actions))
    expected_target = robot.data.joint_pos.torch[:, 7:].clone()
    assert torch.allclose(action.processed_actions, expected_target)

    action.apply_actions()
    first_applied_target = robot.position_targets[0]
    action.apply_actions()
    assert torch.allclose(robot.position_targets[0], first_applied_target)
    assert torch.allclose(robot.position_targets[0], expected_target)


def test_full_hand_action_anchors_only_held_resets_until_deliberate_release():
    num_envs = 2
    robot = _DummyFullKukaAllegro(num_envs)
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene={"robot": robot},
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        stack_reset_state=SimpleNamespace(
            held_cube_ids=torch.tensor((1, -1)),
            grasp_pair_ids=torch.tensor((1, 2)),
        ),
    )
    cfg = ResetPreservingRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES),
        preserve_order=True,
        scale=0.10,
        max_delta=0.10,
        joint_limit_margin=0.02,
        reset_preload_joint_names=KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES,
        reset_preload_commands_by_pair=KUKA_ALLEGRO_LARGE_CUBE_GRASP_PAIR_CLOSED_COMMANDS,
        reset_open_commands_by_pair=KUKA_ALLEGRO_LARGE_CUBE_GRASP_PAIR_OPEN_COMMANDS,
        preload_release_threshold=0.5,
        preload_release_steps=2,
    )
    action = ResetPreservingRelativeJointPositionAction(cfg, env)
    reset_pose = robot.data.joint_pos.torch[:, 7:].clone()
    action.reset()

    raw_actions = torch.zeros((num_envs, 16))
    action.process_actions(raw_actions)
    preload_command = torch.tensor(KUKA_ALLEGRO_LARGE_CUBE_GRASP_PAIR_CLOSED_COMMANDS[1])
    preload = torch.clamp(preload_command, min=-0.98, max=0.98)
    assert torch.allclose(action.processed_actions[0], preload)
    assert torch.allclose(action.processed_actions[1], reset_pose[1])
    assert torch.equal(action.raw_actions, raw_actions)

    # The held reset remains anchored despite elapsed time or contact
    # deflection; the unheld row always uses measured-state semantics.
    env.episode_length_buf[:] = 50
    robot.data.joint_pos.torch[:, 7:] += 0.05
    action.process_actions(raw_actions)
    assert torch.allclose(action.processed_actions[0], preload)
    assert torch.allclose(action.processed_actions[1], reset_pose[1] + 0.05)

    # Two consecutive pair-projected opening commands release the one-way
    # anchor. The first command is still expressed around the preload, while
    # the second immediately switches to the measured-state target.
    opening_direction = torch.tensor(KUKA_ALLEGRO_LARGE_CUBE_GRASP_PAIR_OPEN_COMMANDS[1]) - preload_command
    opening_actions = torch.zeros_like(raw_actions)
    opening_actions[0] = 0.5 * torch.sign(opening_direction)
    action.process_actions(opening_actions)
    first_open_target = torch.clamp(preload_command + 0.05 * torch.sign(opening_direction), min=-0.98, max=0.98)
    assert torch.allclose(action.processed_actions[0], first_open_target)
    action.process_actions(opening_actions)
    measured_open_target = torch.clamp(
        robot.data.joint_pos.torch[0, 7:] + 0.05 * torch.sign(opening_direction),
        min=-0.98,
        max=0.98,
    )
    assert torch.allclose(action.processed_actions[0], measured_open_target)
    assert not action._preload_assist_active[0]

    action.process_actions(raw_actions)
    assert torch.allclose(action.processed_actions[0], robot.data.joint_pos.torch[0, 7:])

    # A reset rearms held rows and leaves unheld rows unassisted.
    action.reset()
    assert action._preload_assist_active.tolist() == [True, False]


def test_full_hand_distribution_uses_continuous_group_specific_exploration():
    distribution = KukaAllegroGaussianDistribution(output_dim=23)
    distribution.update(torch.zeros((4, 23)))

    assert distribution.sample().shape == (4, 23)
    assert torch.allclose(distribution.std[:, :7], torch.full((4, 7), 0.35))
    assert torch.allclose(distribution.std[:, 7:], torch.full((4, 16), 0.15))
    assert distribution.params[0].shape == (4, 23)
    assert distribution.params[1].shape == (4, 23)


def test_large_cube_reset_geometry_preserves_clearances_and_yaw_safe_layouts():
    reset_cls = reset_events.LargeCubeDiverseKukaAllegroStackResetStateTable

    assert pytest.approx(KUKA_ALLEGRO_LARGE_CUBE_EDGE_LENGTH) == reset_cls._CUBE_HEIGHT
    assert pytest.approx(KUKA_ALLEGRO_LARGE_CUBE_RESTING_HEIGHT) == reset_cls._TABLE_HEIGHT
    assert reset_cls._pick_pregrasp_height() == pytest.approx(0.1965)
    assert reset_cls._pick_contact_height() == pytest.approx(0.037)
    assert reset_cls._pick_supported_height() == pytest.approx(0.200)
    assert reset_cls._transport_height(second_pick=False) == pytest.approx(0.1565)
    expected_second_transport_height = (
        reset_cls._table_surface_height() + 0.5 * reset_cls._CUBE_HEIGHT + reset_cls._SECOND_TRANSPORT_BOTTOM_CLEARANCE
    )
    assert reset_cls._transport_height(second_pick=True) == pytest.approx(expected_second_transport_height)
    assert reset_cls._transport_height(second_pick=True) == pytest.approx(0.2262842712)
    assert reset_cls._pair_ready_source_height() == pytest.approx(0.1965)
    assert reset_cls._table_approach_minimum_height() == pytest.approx(0.1965)
    assert reset_cls._ring_transport_minimum_height() == pytest.approx(0.1115)
    assert pytest.approx(torch.deg2rad(torch.tensor(45.0)).item()) == reset_cls._GLOBAL_TILT_LIMIT
    assert pytest.approx(torch.deg2rad(torch.tensor(45.0)).item()) == reset_cls._PLACE_TILT_LIMIT

    required_separation = 2.0**0.5 * KUKA_ALLEGRO_LARGE_CUBE_EDGE_LENGTH
    for table_start, count in ((False, reset_cls._SEMANTIC_LAYOUT_COUNT), (True, reset_cls._TABLE_ROWS)):
        layouts = (reset_cls._sample_layout(layout_id, table_start=table_start) for layout_id in range(count))
        minimum_separation = min(
            torch.dist(torch.tensor(layout[first]), torch.tensor(layout[second])).item()
            for layout in layouts
            for first in range(3)
            for second in range(first + 1, 3)
        )
        assert minimum_separation >= reset_cls._LAYOUT_MINIMUM_SEPARATION
        assert minimum_separation > required_separation


def test_large_cube_pick_separates_partial_supported_states_from_safe_held_resets():
    source_positions = torch.tensor(
        (
            (0.50, 0.00, 0.037),
            (0.50, 0.00, 0.037),
            (0.50, 0.00, 0.037),
            (0.50, 0.00, 0.037),
            (0.50, 0.00, 0.037),
            (0.50, 0.00, 0.037),
            (0.50, 0.00, 0.037),
            (0.50, 0.00, 0.037),
            (0.50, 0.00, 0.037),
            (0.50, 0.00, 0.037),
        )
    )
    progress = torch.tensor((0.0, 0.25, 0.50, 0.625, 0.70, 0.75, 0.8125, 0.875, 0.9375, 1.0))
    ring_pair_ids = torch.full((progress.numel(),), 2, dtype=torch.long)
    large_term = reset_events.LargeCubeDiverseKukaAllegroStackResetStateTable.__new__(
        reset_events.LargeCubeDiverseKukaAllegroStackResetStateTable
    )

    targets, closure, held, maximum_tilt = large_term._pick_phase(
        source_positions,
        progress,
        ring_pair_ids,
    )

    assert torch.allclose(closure, torch.tensor((0.0, 0.0, 0.0, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0)))
    assert torch.equal(
        held,
        torch.tensor((False, False, False, False, False, True, True, True, True, True)),
    )
    assert torch.allclose(
        targets[:, 2],
        torch.tensor((0.1965, 0.19825, 0.2000, 0.2000, 0.2000, 0.1050, 0.106625, 0.10825, 0.109875, 0.1115)),
    )
    assert torch.allclose(
        torch.rad2deg(maximum_tilt),
        torch.tensor((45.0, 30.0, 15.0, 7.5, 3.0, 15.0, 15.0, 15.0, 15.0, 15.0)),
        atol=1.0e-5,
    )

    legacy_term = reset_events.DiverseKukaAllegroStackResetStateTable.__new__(
        reset_events.DiverseKukaAllegroStackResetStateTable
    )
    _, legacy_closure, legacy_held, _ = legacy_term._pick_phase(
        source_positions,
        progress,
        ring_pair_ids,
    )
    assert torch.allclose(legacy_closure, torch.clamp(2.0 * progress - 1.0, min=0.0, max=1.0))
    assert torch.equal(legacy_held, progress >= 1.0 - 1.0e-6)


def test_full_hand_pick_bridge_is_continuous_and_uses_object_axis_approach():
    reset_cls = reset_events.FullHandLargeCubeDiverseKukaAllegroStackResetStateTable
    term = reset_cls.__new__(reset_cls)
    progress = torch.tensor((0.0, 0.25, 0.50, 0.625, 0.75, 0.875, 1.0))
    source_positions = torch.tensor((0.50, 0.00, 0.037)).expand(progress.numel(), -1).clone()
    pair_ids = torch.zeros(progress.numel(), dtype=torch.long)

    targets, closure, held, maximum_tilt = term._pick_phase(source_positions, progress, pair_ids)

    assert torch.allclose(closure, torch.tensor((0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0)))
    assert torch.equal(held, torch.tensor((False, False, False, False, True, True, True)))
    assert targets[3, 2] == pytest.approx(0.0405)
    assert targets[4, 2] == pytest.approx(0.0405)
    assert targets[5, 2] == pytest.approx(0.0655)
    assert targets[6, 2] == pytest.approx(0.1115)
    assert torch.allclose(maximum_tilt, torch.zeros_like(maximum_tilt))

    term._recipe_ids = torch.full((progress.numel(),), int(reset_events.StackResetRecipe.FIRST_PICK))
    term._held_roles = torch.where(held, torch.ones_like(pair_ids), -torch.ones_like(pair_ids))
    term._grasp_pair_ids = pair_ids
    term._orientation_ids = torch.full_like(pair_ids, 2)
    term._progress = progress
    term._role_positions = source_positions[:, None].expand(-1, 3, -1).clone()
    term._layout_ids = torch.zeros_like(pair_ids)
    identity = torch.eye(3).expand(progress.numel(), -1, -1)
    adjusted = term._adjust_target_positions_for_rotation(targets, identity)
    approach_axis = torch.tensor(KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_TOOL_OFFSETS[0])
    approach_axis /= torch.linalg.vector_norm(approach_axis)
    contact_positions = source_positions.clone()
    contact_positions[:, 2] = reset_cls._pick_contact_height()

    assert torch.allclose(adjusted[0], contact_positions[0] - 0.10 * approach_axis)
    assert torch.allclose(adjusted[1], contact_positions[1] - 0.05 * approach_axis)
    assert torch.allclose(adjusted[2], contact_positions[2])
    assert torch.allclose(adjusted[3], contact_positions[3])
    assert torch.allclose(adjusted[4:], targets[4:])
    assert torch.equal(
        term._active_grasp_pair_ids(torch.tensor((0, 1, 2, 2))),
        torch.zeros(4, dtype=torch.long),
    )
    assert reset_cls._orientation_ids_for_recipe(reset_events.StackResetRecipe.FIRST_PICK) == (2, 6)
    assert reset_cls._orientation_ids_for_recipe(reset_events.StackResetRecipe.PAIR_READY) == (2, 6)
    assert reset_cls._orientation_ids_for_recipe(reset_events.StackResetRecipe.TABLE) == (2, 3, 6)
    assert reset_cls._orientation_ids_for_recipe(reset_events.StackResetRecipe.FIRST_TRANSPORT) == tuple(range(8))
    assert reset_cls._TABLE_USES_OBJECT_AXIS_APPROACH is True
    assert reset_cls._TABLE_X_LOWER == reset_cls._SEMANTIC_X_LOWER
    assert reset_cls._TABLE_X_EXTENT == reset_cls._SEMANTIC_X_EXTENT
    assert reset_cls._TABLE_Y_LOWER == reset_cls._SEMANTIC_Y_LOWER
    assert reset_cls._TABLE_Y_EXTENT == reset_cls._SEMANTIC_Y_EXTENT


def test_full_hand_pick_bridge_densely_covers_contact_to_retained_lift():
    reset_cls = reset_events.FullHandLargeCubeDiverseKukaAllegroStackResetStateTable
    term = reset_cls.__new__(reset_cls)
    progress = torch.linspace(0.75, 0.875, 33)
    source_positions = torch.tensor((0.50, 0.00, 0.037)).expand(progress.numel(), -1).clone()
    pair_ids = torch.zeros(progress.numel(), dtype=torch.long)

    targets, closure, held, maximum_tilt = term._pick_phase(source_positions, progress, pair_ids)

    expected_lift = 0.025 * (progress - progress[0]) / (progress[-1] - progress[0])
    contact_height = reset_cls._pick_contact_height()
    assert torch.all(held)
    assert torch.allclose(closure, torch.ones_like(closure))
    assert torch.allclose(targets[:, 2] - contact_height, expected_lift, atol=1.0e-6)
    assert torch.all(targets[1:, 2] > targets[:-1, 2])
    assert float(torch.diff(targets[:, 2]).max()) < 8.0e-4
    assert torch.allclose(maximum_tilt, torch.zeros_like(maximum_tilt))


def test_full_hand_acquisition_targets_split_table_lift_ladder_from_retained_lift():
    full_hand_cls = reset_events.FullHandLargeCubeDiverseKukaAllegroStackResetStateTable
    legacy_classes = (
        reset_events.DiverseKukaAllegroStackResetStateTable,
        reset_events.LargeCubeDiverseKukaAllegroStackResetStateTable,
    )

    assert full_hand_cls._target_potential(reset_events.StackResetRecipe.TABLE) == pytest.approx(1.05)
    assert full_hand_cls._target_potential(reset_events.StackResetRecipe.FIRST_PICK) == pytest.approx(1.25)
    for recipe in (reset_events.StackResetRecipe.TABLE, reset_events.StackResetRecipe.FIRST_PICK):
        assert all(reset_cls._target_potential(recipe) == pytest.approx(1.0) for reset_cls in legacy_classes)
    for recipe in (reset_events.StackResetRecipe.PAIR_READY, reset_events.StackResetRecipe.SECOND_PICK):
        assert full_hand_cls._target_potential(recipe) == pytest.approx(6.25)
        assert all(reset_cls._target_potential(recipe) == pytest.approx(6.0) for reset_cls in legacy_classes)
    assert full_hand_cls._target_potential(reset_events.StackResetRecipe.FIRST_TRANSPORT) == pytest.approx(3.0)


def test_table_target_potential_override_changes_only_table_rows():
    reset_cls = reset_events.FullHandLargeCubeDiverseKukaAllegroStackResetStateTable
    term = reset_cls.__new__(reset_cls)
    term._recipe_ids = torch.tensor(
        (
            int(reset_events.StackResetRecipe.TABLE),
            int(reset_events.StackResetRecipe.FIRST_PICK),
            int(reset_events.StackResetRecipe.TABLE),
        )
    )
    term._target_potentials = torch.tensor((1.05, 1.25, 1.05))

    term._apply_table_target_potential(None)
    assert torch.allclose(term._target_potentials, torch.tensor((1.05, 1.25, 1.05)))

    term._apply_table_target_potential(1.10)

    assert torch.allclose(term._target_potentials, torch.tensor((1.10, 1.25, 1.10)))


@pytest.mark.parametrize("target_potential", (0.0, -1.0, 10.01, float("inf"), float("nan")))
def test_table_target_potential_override_rejects_invalid_values(target_potential: float):
    term = reset_events.StackResetStateTable.__new__(reset_events.StackResetStateTable)
    term._recipe_ids = torch.tensor((int(reset_events.StackResetRecipe.TABLE),))
    term._target_potentials = torch.tensor((1.0,))

    with pytest.raises(ValueError, match="table_target_potential"):
        term._apply_table_target_potential(target_potential)


def test_full_hand_semantic_planner_balances_orders_and_clears_acquisition_corridors():
    reset_cls = reset_events.FullHandLargeCubeDiverseKukaAllegroStackResetStateTable
    term = reset_cls.__new__(reset_cls)
    term._env = SimpleNamespace(device="cpu")
    term._arm_anchors = torch.tensor(KUKA_ALLEGRO_STACK_ARM_POSES)
    semantic_layouts = term._arm_anchors.new_tensor(
        tuple(term._sample_layout(layout_id, table_start=False) for layout_id in range(term._SEMANTIC_LAYOUT_COUNT))
    )
    physical_layouts = semantic_layouts.clone()

    term._prepare_semantic_reset_plans(semantic_layouts)

    assert torch.equal(
        torch.bincount(term._semantic_first_physical_role_ids, minlength=3),
        torch.tensor((0, 128, 128)),
    )
    selected_physical_first = physical_layouts.gather(
        1,
        term._semantic_first_physical_role_ids.view(-1, 1, 1).expand(-1, 1, 2),
    ).squeeze(1)
    selected_physical_second = physical_layouts.gather(
        1,
        (3 - term._semantic_first_physical_role_ids).view(-1, 1, 1).expand(-1, 1, 2),
    ).squeeze(1)
    assert torch.allclose(semantic_layouts[:, 1], selected_physical_first)
    assert torch.allclose(semantic_layouts[:, 2], selected_physical_second)
    minimum_clearance = term._MINIMUM_ACQUISITION_CORRIDOR_CENTER_DISTANCE
    assert float(term._first_pick_corridor_center_distances.min()) >= minimum_clearance - 1.0e-6
    assert float(term._second_pick_corridor_center_distances.min()) >= minimum_clearance - 1.0e-6
    assert set(term._first_pick_orientation_ids_by_layout.tolist()) == {2, 6}
    assert set(term._second_pick_orientation_ids_by_layout.tolist()) == {2, 6}
    assert reset_cls._orientation_ids_for_recipe(reset_events.StackResetRecipe.SECOND_TRANSPORT) == tuple(range(8))


def test_oriented_cube_validation_catches_rotated_corner_overlap():
    first_centers = torch.zeros((3, 3))
    second_centers = torch.tensor(
        (
            (0.090, 0.0, 0.0),
            (0.100, 0.0, 0.0),
            (0.000, 0.0, 0.080),
        )
    )
    identity = torch.eye(3)
    yaw_45 = matrix_from_quaternion_xyzw(
        torch.tensor((0.0, 0.0, 0.38268343, 0.92387953)),
        first_centers,
    )
    first_rotations = identity.expand(3, -1, -1)
    second_rotations = torch.stack((yaw_45, yaw_45, identity))

    intersections = reset_events._oriented_cube_pair_intersections(
        first_centers,
        first_rotations,
        second_centers,
        second_rotations,
        edge_length=0.08,
    )

    # A 45-degree cube reaches sqrt(2) * half-edge along X, so the 9 cm
    # center spacing overlaps even though the old 8 cm center test accepted it.
    assert torch.equal(intersections, torch.tensor((True, False, False)))


def test_segment_oriented_box_validation_catches_swept_hand_overlap():
    starts = torch.tensor(
        (
            (-0.10, 0.00, 0.00),
            (-0.10, 0.06, 0.00),
            (-0.10, 0.06, 0.00),
        )
    )
    ends = torch.tensor(
        (
            (0.10, 0.00, 0.00),
            (0.10, 0.06, 0.00),
            (0.10, 0.06, 0.00),
        )
    )
    centers = torch.zeros_like(starts)
    rotations = torch.eye(3).expand(3, -1, -1).clone()
    rotations[2] = matrix_from_quaternion_xyzw(
        torch.tensor((0.0, 0.0, 0.38268343, 0.92387953)),
        starts[2],
    )

    intersections = reset_events._segment_oriented_box_intersections(
        starts,
        ends,
        centers,
        rotations,
        half_extent=0.05,
    )

    assert torch.equal(intersections, torch.tensor((True, False, True)))


def test_full_hand_second_transport_envelope_rejects_stack_crossing_palm():
    reset_cls = reset_events.FullHandLargeCubeDiverseKukaAllegroStackResetStateTable
    term = reset_cls.__new__(reset_cls)
    term._env = SimpleNamespace(device="cpu")
    term._arm_anchors = torch.tensor(KUKA_ALLEGRO_STACK_ARM_POSES)
    row_id = 15215
    term._grasp_pair_ids = torch.zeros(row_id + 1, dtype=torch.long)
    term._role_positions = torch.zeros((row_id + 1, 3, 3))
    term._role_positions[row_id, :2] = torch.tensor(
        (
            (0.56287998, -0.00122449, 0.037),
            (0.56287998, -0.00122449, 0.117),
        )
    )
    tool_position = torch.tensor((0.55028254, -0.13089603, 0.11926484))
    candidate_rotations, _ = term._target_wrist_rotations(
        torch.tensor((7, 7)),
        torch.deg2rad(torch.tensor((33.75, 33.75))),
        torch.tensor((7, 1)),
        torch.zeros(2, dtype=torch.long),
    )

    collision_free = term._second_transport_stack_clearance(
        torch.tensor((row_id,)),
        candidate_rotations.unsqueeze(0),
        tool_position.expand(2, -1).unsqueeze(0),
        hand_clearance=term._SECOND_TRANSPORT_HAND_STACK_CLEARANCE,
        cube_clearance=term._SECOND_TRANSPORT_CUBE_STACK_CLEARANCE,
    )

    # Both held cubes clear the first stack, but azimuth seven routes the
    # palm-to-pinch segment through it. The alternate tilt direction keeps the
    # same yaw and held center while clearing the complete envelope.
    assert torch.equal(collision_free, torch.tensor(((False, True),)))


def test_full_hand_final_release_envelope_excludes_non_tip_top_and_lower_contacts():
    reset_cls = reset_events.FullHandLargeCubeDiverseKukaAllegroStackResetStateTable
    term = reset_cls.__new__(reset_cls)
    term._env = SimpleNamespace(device="cpu")
    term._arm_anchors = torch.tensor(KUKA_ALLEGRO_STACK_ARM_POSES)
    term._grasp_pair_ids = torch.zeros(1, dtype=torch.long)
    term._progress = torch.zeros(1)
    term._role_positions = torch.tensor(
        (
            (
                (0.50, 0.00, 0.037),
                (0.50, 0.00, 0.117),
                (0.50, 0.00, 0.197),
            ),
        )
    )
    palm_rotation, _ = term._target_wrist_rotations(
        torch.tensor((0,)),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
    )
    palm_rotation = palm_rotation[0]
    index_offset = torch.tensor(reset_cls._FINAL_RELEASE_INDEX_LINK_2_CLOSED_OFFSET)
    tool_offset = torch.tensor(KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_TOOL_OFFSETS[0])
    top_center = term._role_positions[0, 2]
    middle_center = term._role_positions[0, 1]
    safe_palm = top_center + torch.tensor((0.0, 0.0, 0.20))
    top_contact_palm = top_center - palm_rotation @ index_offset
    lower_contact_palm = middle_center - palm_rotation @ index_offset
    palm_positions = torch.stack((safe_palm, top_contact_palm, lower_contact_palm))
    tool_positions = palm_positions + palm_rotation @ tool_offset

    collision_free = term._final_release_stack_clearance(
        torch.tensor((0,)),
        palm_rotation.expand(3, -1, -1).unsqueeze(0),
        tool_positions.unsqueeze(0),
    )

    assert torch.equal(collision_free, torch.tensor(((True, False, False),)))


def test_full_hand_final_release_retreat_preserves_authored_orientations(monkeypatch):
    reset_cls = reset_events.FullHandLargeCubeDiverseKukaAllegroStackResetStateTable
    term = reset_cls.__new__(reset_cls)
    term._env = SimpleNamespace(device="cpu")
    term._arm_anchors = torch.tensor(KUKA_ALLEGRO_STACK_ARM_POSES)
    term._arm_positions = torch.zeros((1, 7))
    term._recipe_ids = torch.tensor((int(reset_events.StackResetRecipe.FINAL_RELEASE),))
    term._grasp_pair_ids = torch.zeros(1, dtype=torch.long)
    term._orientation_ids = torch.tensor((6,))
    term._authored_orientation_ids = term._orientation_ids.clone()
    term._tilt_azimuth_ids = torch.tensor((7,))
    term._authored_tilt_azimuth_ids = term._tilt_azimuth_ids.clone()
    term._resolved_tilt_angles = torch.deg2rad(torch.tensor((45.0,)))

    def fake_reset_ik(
        seed_positions,
        target_positions,
        target_rotations,
        tool_offsets,
        **kwargs,
    ):
        del seed_positions, target_rotations, tool_offsets, kwargs
        arm_positions = torch.zeros((target_positions.shape[0], 7))
        arm_positions[:, :3] = target_positions
        zeros = target_positions.new_zeros(target_positions.shape[0])
        return arm_positions, zeros, zeros

    def fake_grasp_pair_pose(arm_positions, grasp_pair_ids):
        del grasp_pair_ids
        rotations = torch.eye(3).expand(arm_positions.shape[0], -1, -1).clone()
        return arm_positions[:, :3], rotations

    def fake_final_release_clearance(release_rows, palm_rotations, tool_positions):
        del release_rows, palm_rotations
        return tool_positions[..., 2] >= 0.020 - 1.0e-6

    monkeypatch.setattr(reset_events, "solve_kuka_allegro_reset_ik", fake_reset_ik)
    term._grasp_pair_pose = fake_grasp_pair_pose
    term._final_release_stack_clearance = fake_final_release_clearance

    arm_positions = torch.zeros((1, 7))
    position_residuals = torch.zeros(1)
    rotation_residuals = torch.zeros(1)
    target_positions = torch.zeros((1, 3))
    target_rotations, _ = term._target_wrist_rotations(
        term._orientation_ids,
        term._resolved_tilt_angles,
        term._tilt_azimuth_ids,
        term._grasp_pair_ids,
    )
    tool_offsets = torch.tensor((KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_TOOL_OFFSETS[0],))
    joint_lower = torch.full((7,), -3.05)
    joint_upper = torch.full((7,), 3.05)

    repaired = term._repair_final_release_clearance(
        arm_positions,
        position_residuals,
        rotation_residuals,
        target_positions,
        target_rotations,
        tool_offsets,
        joint_lower,
        joint_upper,
    )

    assert repaired[0][0, 2] == pytest.approx(0.020)
    assert term._ik_final_release_repair_count == 1
    assert term._ik_final_release_orientation_repair_count == 0
    assert term._final_release_retreat_lifts.tolist() == pytest.approx([0.020])
    assert term._orientation_ids.tolist() == [6]
    assert term._authored_orientation_ids.tolist() == [6]
    assert term._tilt_azimuth_ids.tolist() == [7]
    assert term._authored_tilt_azimuth_ids.tolist() == [7]
    assert torch.rad2deg(term._resolved_tilt_angles).item() == pytest.approx(45.0)


def test_full_hand_final_release_orientation_fallback_preserves_authored_bins(monkeypatch):
    reset_cls = reset_events.FullHandLargeCubeDiverseKukaAllegroStackResetStateTable
    term = reset_cls.__new__(reset_cls)
    term._env = SimpleNamespace(device="cpu")
    term._arm_anchors = torch.tensor(KUKA_ALLEGRO_STACK_ARM_POSES)
    term._recipe_ids = torch.tensor((int(reset_events.StackResetRecipe.FINAL_RELEASE),))
    term._grasp_pair_ids = torch.zeros(1, dtype=torch.long)
    term._orientation_ids = torch.tensor((6,))
    term._authored_orientation_ids = term._orientation_ids.clone()
    term._tilt_azimuth_ids = torch.tensor((7,))
    term._authored_tilt_azimuth_ids = term._tilt_azimuth_ids.clone()
    term._resolved_tilt_angles = torch.deg2rad(torch.tensor((45.0,)))

    def fake_reset_ik(
        seed_positions,
        target_positions,
        target_rotations,
        tool_offsets,
        **kwargs,
    ):
        del seed_positions, tool_offsets, kwargs
        arm_positions = torch.zeros((target_positions.shape[0], 7))
        arm_positions[:, :3] = target_positions
        arm_positions[:, 3] = target_rotations[:, 2, 2]
        zeros = target_positions.new_zeros(target_positions.shape[0])
        return arm_positions, zeros, zeros

    def fake_grasp_pair_pose(arm_positions, grasp_pair_ids):
        del grasp_pair_ids
        rotations = torch.eye(3).expand(arm_positions.shape[0], -1, -1).clone()
        rotations[:, 2, 2] = arm_positions[:, 3]
        return arm_positions[:, :3], rotations

    def fake_final_release_clearance(release_rows, palm_rotations, tool_positions):
        del release_rows, tool_positions
        return palm_rotations[..., 2, 2] >= 1.0 - 1.0e-6

    def fake_target_wrist_rotations(
        orientation_ids,
        tilt_angles,
        tilt_azimuth_ids=None,
        grasp_pair_ids=None,
    ):
        del orientation_ids, tilt_azimuth_ids, grasp_pair_ids
        cosine = torch.cos(tilt_angles)
        sine = torch.sin(tilt_angles)
        rotations = torch.eye(3).expand(tilt_angles.numel(), -1, -1).clone()
        rotations[:, 1, 1] = cosine
        rotations[:, 1, 2] = -sine
        rotations[:, 2, 1] = sine
        rotations[:, 2, 2] = cosine
        return rotations, torch.zeros_like(tilt_angles)

    monkeypatch.setattr(reset_events, "solve_kuka_allegro_reset_ik", fake_reset_ik)
    term._grasp_pair_pose = fake_grasp_pair_pose
    term._final_release_stack_clearance = fake_final_release_clearance
    term._target_wrist_rotations = fake_target_wrist_rotations

    target_rotations, _ = term._target_wrist_rotations(
        term._orientation_ids,
        term._resolved_tilt_angles,
        term._tilt_azimuth_ids,
        term._grasp_pair_ids,
    )
    repaired = term._repair_final_release_clearance(
        torch.zeros((1, 7)),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros((1, 3)),
        target_rotations,
        torch.tensor((KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_TOOL_OFFSETS[0],)),
        torch.full((7,), -3.05),
        torch.full((7,), 3.05),
    )

    assert repaired[0][0, 3] == pytest.approx(1.0)
    assert term._ik_final_release_repair_count == 1
    assert term._ik_final_release_orientation_repair_count == 1
    assert term._final_release_retreat_lifts.tolist() == pytest.approx([0.0])
    assert term._orientation_ids.tolist() == [6]
    assert term._authored_orientation_ids.tolist() == [6]
    assert term._tilt_azimuth_ids.tolist() == [7]
    assert term._authored_tilt_azimuth_ids.tolist() == [7]
    assert torch.rad2deg(term._resolved_tilt_angles).item() == pytest.approx(0.0)


def test_full_hand_second_transport_yaw_fallback_preserves_authored_balance(monkeypatch):
    reset_cls = reset_events.FullHandLargeCubeDiverseKukaAllegroStackResetStateTable
    term = reset_cls.__new__(reset_cls)
    term._env = SimpleNamespace(device="cpu")
    term._arm_anchors = torch.tensor(KUKA_ALLEGRO_STACK_ARM_POSES)
    term._recipe_ids = torch.tensor((int(reset_events.StackResetRecipe.SECOND_TRANSPORT),))
    term._grasp_pair_ids = torch.zeros(1, dtype=torch.long)
    term._orientation_ids = torch.tensor((3,))
    term._authored_orientation_ids = term._orientation_ids.clone()
    term._tilt_azimuth_ids = torch.tensor((5,))
    term._authored_tilt_azimuth_ids = term._tilt_azimuth_ids.clone()
    term._resolved_tilt_angles = torch.deg2rad(torch.tensor((15.0,)))
    term._resolved_tilt_limits = torch.deg2rad(torch.tensor((45.0,)))
    term._held_roles = torch.tensor((2,))
    term._role_positions = torch.zeros((1, 3, 3))

    def fake_parent_solve(
        self,
        seed_positions,
        target_positions,
        target_rotations,
        tool_offsets,
        joint_lower,
        joint_upper,
    ):
        del self, target_positions, tool_offsets, joint_lower, joint_upper
        zeros = seed_positions.new_zeros(seed_positions.shape[0])
        return seed_positions.clone(), zeros, zeros, target_rotations.clone()

    def fake_reset_ik(
        seed_positions,
        target_positions,
        target_rotations,
        tool_offsets,
        **kwargs,
    ):
        del target_positions, target_rotations, tool_offsets, kwargs
        zeros = seed_positions.new_zeros(seed_positions.shape[0])
        return seed_positions.clone(), zeros, zeros

    def fake_grasp_pair_pose(arm_positions, grasp_pair_ids):
        del grasp_pair_ids
        positions = arm_positions.new_zeros((arm_positions.shape[0], 3))
        rotations = torch.eye(3, dtype=arm_positions.dtype).expand(arm_positions.shape[0], -1, -1).clone()
        return positions, rotations

    def fake_collision_free(row_ids, target_positions, palm_rotations, tool_positions=None):
        del row_ids, target_positions, tool_positions
        valid = torch.zeros(palm_rotations.shape[:2], dtype=torch.bool)
        if palm_rotations.shape[1] > 1:
            # Candidate 57 is yaw 4, azimuth 7, tilt 5 degrees for the
            # deterministic shift/grid ordering in the FullHand fallback.
            valid[:, 57] = True
        return valid

    monkeypatch.setattr(
        reset_events.LargeCubeDiverseKukaAllegroStackResetStateTable,
        "_solve_diverse_arm_targets",
        fake_parent_solve,
    )
    monkeypatch.setattr(reset_events, "solve_kuka_allegro_reset_ik", fake_reset_ik)
    term._grasp_pair_pose = fake_grasp_pair_pose
    term._collision_free_wrist_rotations = fake_collision_free

    target_rotations, _ = term._target_wrist_rotations(
        term._orientation_ids,
        term._resolved_tilt_angles,
        term._tilt_azimuth_ids,
        term._grasp_pair_ids,
    )
    seed_positions = torch.zeros((1, 7))
    target_positions = torch.zeros((1, 3))
    tool_offsets = torch.tensor((KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_TOOL_OFFSETS[0],))
    joint_lower = torch.full((7,), -3.05)
    joint_upper = torch.full((7,), 3.05)

    term._solve_diverse_arm_targets(
        seed_positions,
        target_positions,
        target_rotations,
        tool_offsets,
        joint_lower,
        joint_upper,
    )

    assert term._ik_yaw_stack_repair_count == 1
    assert term._authored_orientation_ids.tolist() == [3]
    assert term._orientation_ids.tolist() == [4]
    assert term._tilt_azimuth_ids.tolist() == [7]
    assert torch.rad2deg(term._resolved_tilt_angles).item() == pytest.approx(5.0)


def test_large_cube_pair_rotations_map_nominal_palm_to_upright_cube():
    rotations = matrix_from_quaternion_xyzw(
        KUKA_ALLEGRO_LARGE_CUBE_PALM_TO_HELD_CUBE_QUATERNIONS_XYZW,
        torch.empty(0),
    )

    assert torch.allclose(
        rotations @ rotations.transpose(-1, -2),
        torch.eye(3).expand(3, -1, -1),
        atol=1.0e-6,
    )
    assert torch.allclose(torch.linalg.det(rotations), torch.ones(3), atol=1.0e-6)
    # The nominal downward-facing palm maps palm -X to world +Z.
    assert torch.allclose(
        rotations[:, :, 2],
        torch.tensor((-1.0, 0.0, 0.0)).expand(3, -1),
        atol=1.0e-6,
    )


def test_large_cube_progress_uses_eight_centimeter_stack_spacing():
    positions = torch.tensor(
        (
            ((0.45, 0.0, 0.037), (0.45, 0.0, 0.117), (0.45, 0.0, 0.197)),
            ((0.45, 0.0, 0.0205), (0.45, 0.0, 0.0605), (0.45, 0.0, 0.1005)),
        )
    )
    quaternions = torch.zeros((2, 4))
    quaternions[:, 3] = 1.0
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene={f"cube_{cube_id + 1}": _dummy_cube(positions[:, cube_id], quaternions) for cube_id in range(3)},
    )

    large_progress = mdp.order_invariant_stack_progress(env, cube_height=0.08)
    legacy_progress = mdp.order_invariant_stack_progress(env, cube_height=0.04)

    assert torch.equal(large_progress, torch.tensor((2.0, 1.0)))
    assert torch.equal(legacy_progress, torch.tensor((0.0, 2.0)))


def test_extended_cube_axes_preserve_roles_and_tip_geometry_adds_missing_sensitivity():
    env = _dummy_observation_env()
    tool_params = {"tool_body_name": "palm_link", "tool_offset": (0.0570965, -0.0375159, 0.0498749)}

    base_object = observations.role_conditioned_stack_obs(env, **tool_params)
    cube_x_axes = observations.role_conditioned_cube_x_axes(env)
    tip_positions = observations.body_positions_relative_to_tool(
        env,
        body_cfg=SimpleNamespace(name="robot", body_ids=[1, 2]),
        **tool_params,
    )

    assert cube_x_axes.shape == (3, 9)
    assert torch.allclose(cube_x_axes[0], cube_x_axes[1])
    assert not torch.allclose(cube_x_axes[0], cube_x_axes[2])
    # Existing cube z-axis observations cannot distinguish pure yaw.
    assert torch.allclose(base_object[0], base_object[2])
    assert tip_positions.shape == (3, 6)
    assert torch.allclose(tip_positions[0], tip_positions[1])
    assert not torch.allclose(tip_positions[0], tip_positions[2])


def test_kuka_allegro_reset_anchors_reach_the_shared_stack_workspace():
    arm_positions = torch.tensor(KUKA_ALLEGRO_STACK_ARM_POSES)
    actual_positions = kuka_allegro_pinch_position(arm_positions)
    target_xy = torch.tensor(reset_events._STATE_TABLE_ANCHORS).unsqueeze(1).expand(-1, 5, -1)
    target_z = torch.tensor((0.04, 0.08, 0.14, 0.17, 0.1175)).view(1, 5, 1).expand(9, -1, -1)
    target_positions = torch.cat((target_xy, target_z), dim=2)

    assert torch.allclose(actual_positions, target_positions, atol=6.0e-4)
    assert torch.all(arm_positions >= torch.tensor(KUKA_ALLEGRO_STACK_ARM_WORKSPACE_LOWER))
    assert torch.all(arm_positions <= torch.tensor(KUKA_ALLEGRO_STACK_ARM_WORKSPACE_UPPER))


def test_diverse_wrist_targets_span_independent_yaw_and_tilt_directions():
    term = reset_events.DiverseKukaAllegroStackResetStateTable.__new__(
        reset_events.DiverseKukaAllegroStackResetStateTable
    )
    term._env = SimpleNamespace(device="cpu")
    term._arm_anchors = torch.tensor(KUKA_ALLEGRO_STACK_ARM_POSES)
    yaw_ids = torch.arange(8).repeat_interleave(8)
    azimuth_ids = torch.arange(8).repeat(8)
    tilt_angles = torch.deg2rad(torch.full((64,), 20.0))

    rotations, yaw_angles = term._target_wrist_rotations(yaw_ids, tilt_angles, azimuth_ids)

    identity = torch.eye(3).expand(64, -1, -1)
    assert torch.allclose(rotations @ rotations.transpose(-1, -2), identity, atol=1.0e-5)
    assert torch.allclose(torch.linalg.det(rotations), torch.ones(64), atol=1.0e-5)
    assert torch.unique(rotations.round(decimals=5).reshape(64, -1), dim=0).shape[0] == 64
    assert torch.unique(yaw_angles).shape[0] == 8
    # For any fixed yaw, all eight local-XY axes change the palm normal.
    palm_normals = rotations[:, :, 2].reshape(8, 8, 3)
    assert all(torch.unique(normals.round(decimals=5), dim=0).shape[0] == 8 for normals in palm_normals)


def test_diverse_pair_ready_bridge_uses_clearance_arc_and_safe_pregrasp_endpoint():
    table_height = reset_events.StackResetStateTable._TABLE_HEIGHT
    progress = torch.tensor((0.0, 0.01, 0.5, 0.99, 1.0))
    base_positions = torch.tensor((0.42, -0.10, table_height)).expand(progress.numel(), -1)
    second_sources = torch.tensor((0.54, 0.10, table_height)).expand(progress.numel(), -1)

    targets = reset_events.DiverseKukaAllegroStackResetStateTable._pair_ready_targets(
        base_positions,
        second_sources,
        progress,
    )

    starts = base_positions + torch.tensor((0.0, 0.0, 0.12))
    ends = second_sources.clone()
    ends[:, 2] = reset_events.DiverseKukaAllegroStackResetStateTable._pair_ready_source_height()
    linear_targets = torch.lerp(starts, ends, progress.unsqueeze(1))
    clearance = targets[:, 2] - linear_targets[:, 2]

    assert torch.allclose(targets[[0, -1]], linear_targets[[0, -1]], atol=1.0e-6)
    assert torch.allclose(targets[:, :2], linear_targets[:, :2])
    # SECOND_PICK owns the final descent from this safe loose-cube approach.
    assert targets[-1, 2] > reset_events.DiverseKukaAllegroStackResetStateTable._pick_pregrasp_height()
    assert clearance[2] == pytest.approx(
        reset_events.DiverseKukaAllegroStackResetStateTable._PAIR_READY_CLEARANCE_ARC_HEIGHT
    )
    # Broad shoulders protect the full open-hand envelope even one percent
    # away from either exact semantic endpoint.
    assert clearance[1] > 0.04
    assert clearance[3] > 0.04


def test_diverse_table_approaches_only_balanced_movable_cube_roles():
    row_ids = torch.arange(reset_events.DiverseKukaAllegroStackResetStateTable._TABLE_ROWS)

    selected_roles = reset_events.DiverseKukaAllegroStackResetStateTable._table_approach_role_ids(row_ids)

    assert torch.equal(torch.unique(selected_roles), torch.tensor((1, 2)))
    assert torch.equal(torch.bincount(selected_roles, minlength=3), torch.tensor((0, 8192, 8192)))
    for orientation_id in range(8):
        orientation_roles = selected_roles[row_ids.remainder(8) == orientation_id]
        assert torch.equal(torch.bincount(orientation_roles, minlength=3), torch.tensor((0, 1024, 1024)))


def test_diverse_reset_curriculum_balances_every_authored_reset_mode():
    curriculum = _small_diverse_reset_curriculum()

    probabilities = curriculum._sampling_probabilities()

    # Each occupied non-table (recipe, pair, yaw, azimuth, magnitude) mode
    # receives one fifth of the 0.65 adaptive mass, independent of row count.
    expected_mode_mass = torch.tensor(0.65 / 5)
    mode_rows = (
        torch.tensor((0, 1, 2)),
        torch.tensor((3,)),
        torch.tensor((4,)),
        torch.tensor((5,)),
        torch.tensor((6,)),
    )
    assert all(torch.isclose(probabilities[rows].sum(), expected_mode_mass) for rows in mode_rows)
    assert torch.isclose(probabilities[curriculum._table_rows].sum(), torch.tensor(0.35))
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0))

    metrics = curriculum(SimpleNamespace(device="cpu"), torch.empty(0, dtype=torch.long))
    assert torch.isclose(metrics["tilt_azimuth_0_probability"], torch.tensor(4 * 0.65 / 5))
    assert torch.isclose(metrics["tilt_azimuth_1_probability"], expected_mode_mass)
    assert torch.isclose(metrics["tilt_azimuth_7_probability"], torch.tensor(0.35))
    assert torch.isclose(metrics["resolved_tilt_azimuth_0_probability"], torch.tensor(1.0))


def test_diverse_reset_curriculum_can_use_global_row_sampling():
    curriculum = _small_diverse_reset_curriculum()
    curriculum._global_sampling = True
    curriculum._balance_recipes = False
    curriculum._balance_reset_modes = False
    # Six active rows share layout zero while one owns layout one. Legacy
    # layout balancing would give the singleton half the adaptive mass;
    # The global accumulator gives all unseen rows equal mass.
    curriculum._reset_term.layout_ids = torch.tensor((0, 0, 0, 0, 0, 0, 1, 2))
    curriculum._reset_term.layout_count = 3
    curriculum._layout_count = 3

    probabilities = curriculum._sampling_probabilities()

    assert torch.allclose(probabilities[:7], torch.full((7,), 0.65 / 7))
    assert torch.isclose(probabilities[7], torch.tensor(0.35))


def test_diverse_reset_curriculum_reports_tilt_magnitude_metrics():
    curriculum = _small_diverse_reset_curriculum()
    curriculum._sampler.record(
        torch.tensor((0, 0, 3, 3, 3, 3, 4, 5, 6)),
        torch.tensor((True, False, True, True, True, False, False, True, False)),
    )
    curriculum._full_task_attempts_by_row[3] = 4
    curriculum._full_task_successes_by_row[3] = 3
    metrics = curriculum(SimpleNamespace(device="cpu"), torch.empty(0, dtype=torch.long))

    assert metrics["recipe_phase_attempts"] == 9
    assert metrics["recipe_phase_full_stack_attempts"] == 4
    for magnitude_id in range(4):
        assert f"tilt_magnitude_{magnitude_id}_probability" in metrics
        assert f"tilt_magnitude_{magnitude_id}_curriculum_success" in metrics
        assert f"tilt_magnitude_{magnitude_id}_full_stack_success" in metrics
    assert torch.isclose(metrics["tilt_magnitude_1_probability"], torch.tensor(0.65 / 5))
    assert torch.isclose(metrics["tilt_magnitude_1_curriculum_success"], torch.tensor(0.75))
    assert torch.isclose(metrics["tilt_magnitude_1_full_stack_success"], torch.tensor(0.75))
    assert torch.isclose(
        sum(metrics[f"tilt_magnitude_{magnitude_id}_probability"] for magnitude_id in range(4)),
        torch.tensor(1.0),
    )
