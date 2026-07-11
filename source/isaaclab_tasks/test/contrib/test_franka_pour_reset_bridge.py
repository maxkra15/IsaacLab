# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused CPU tests for the Franka Pour reset-mixture grasp bridge."""

from types import MethodType, SimpleNamespace

import pytest
import torch

import isaaclab_tasks.contrib.franka_pour.pour_env as pour_env_module
from isaaclab_tasks.contrib.franka_pour.mdp.reset_mixture import RESET_MIXTURE_REGION_NAMES
from isaaclab_tasks.contrib.franka_pour.pour_env import FrankaPourEnv


def test_near_object_bank_preserves_open_and_preloaded_population_mass(monkeypatch):
    preload_probability = 0.4
    source_count = 8
    source_x = torch.arange(source_count, dtype=torch.float32) * 0.01 + 0.45
    source_positions = torch.stack((source_x, torch.zeros_like(source_x), torch.zeros_like(source_x)), dim=-1)
    cfg = SimpleNamespace(
        curriculum=SimpleNamespace(reset_mixture=object()),
        curriculum_randomized_reset_ik_grid_size=2,
        curriculum_randomized_reset_ik_samples_per_source=2,
        curriculum_independent_sample_attempts=1,
        curriculum_independent_arm_min_tcp_distance=0.1,
        curriculum_randomized_min_source_cell_fraction=1.0,
        curriculum_randomized_min_reset_variants_per_source=1,
        curriculum_randomized_source_radius_range=None,
        curriculum_randomized_target_center_xy=(0.5, -0.2),
        curriculum_randomized_target_position_range=(0.1, 0.1),
        curriculum_randomized_cup_clearance=0.04,
        source_cup_inner_width=0.04,
        source_cup_inner_depth=0.04,
        source_cup_wall_thickness=0.005,
        target_cup_inner_depth=0.10,
        target_cup_wall_thickness=0.005,
        target_cup_reset_pos=(0.5, -0.2, 0.0),
        reset_mixture_near_object_open_phase_probabilities=(0.05, 0.05, 0.90),
        reset_mixture_near_object_preloaded_probability=preload_probability,
        cup_grasp_height=0.05,
        gripper_open_pos=0.04,
        cup_grasp_box_half=(0.03, 0.03, 0.06),
    )
    env = SimpleNamespace(
        cfg=cfg,
        device="cpu",
        _randomized_extent_index_pools=(torch.arange(source_count),),
        _randomized_source_pos_bank_t=source_positions,
        _randomized_source_quat_bank_t=torch.tensor(((0.0, 0.0, 0.0, 1.0),) * source_count),
        _randomized_source_yaw_bank_t=torch.zeros(source_count),
        _randomized_target_pos_bank_t=source_positions + torch.tensor((0.0, -0.2, 0.0)),
        _randomized_tcp_pos_bank_t=source_positions + torch.tensor((0.0, 0.2, 0.2)),
        _randomized_arm_q_bank_t=torch.zeros((source_count, 7)),
        _randomized_pregrasp_arm_q_bank_t=torch.ones((source_count, 7)),
        _randomized_midgrasp_arm_q_bank_t=torch.full((source_count, 7), 2.0),
        _randomized_grasp_arm_q_bank_t=torch.full((source_count, 7), 3.0),
        _target_vertices=None,
        _target_indices=None,
        _collider_margin=0.0,
    )
    env._independent_target_clearance = lambda source, target: torch.ones_like(target, dtype=torch.float32)
    monkeypatch.setattr(
        pour_env_module,
        "collision_free_reset_candidates",
        lambda _builder, robot_q, *_args, **_kwargs: torch.ones(robot_q.shape[0], dtype=torch.bool),
    )

    FrankaPourEnv._build_reset_mixture_banks(
        env,
        object(),
        torch.zeros((source_count, 9)),
        torch.zeros((source_count, 7)),
        torch.arange(source_count),
        torch.arange(7),
        torch.tensor((7, 8)),
    )

    preload = env._reset_mixture_near_object_preloaded_t
    weights = env._reset_mixture_near_object_weights_t
    assert preload.dtype == torch.bool
    assert preload.shape == weights.shape == env._reset_mixture_near_object_source_rows_t.shape
    assert int(preload.sum()) == source_count
    assert int((~preload).sum()) == 25 * source_count
    assert float(weights[preload].sum()) == pytest.approx(preload_probability)
    assert float(weights[~preload].sum()) == pytest.approx(1.0 - preload_probability)
    assert env._reset_mixture_near_object_source_rows_t[preload].tolist() == list(range(source_count))
    torch.testing.assert_close(
        env._reset_mixture_near_object_arm_q_t[preload],
        env._randomized_grasp_arm_q_bank_t,
    )
    for source_row in range(source_count):
        source = env._reset_mixture_near_object_source_rows_t == source_row
        assert float(weights[source & preload].sum()) == pytest.approx(preload_probability / source_count)
        assert float(weights[source & ~preload].sum()) == pytest.approx((1.0 - preload_probability) / source_count)
    source_rows = env._reset_mixture_near_object_source_rows_t
    cell_ids = torch.div(source_rows, 2, rounding_mode="floor")
    for cell_id in range(4):
        cell = cell_ids == cell_id
        assert float(weights[cell & preload].sum()) == pytest.approx(preload_probability / 4.0)
        assert float(weights[cell & ~preload].sum()) == pytest.approx((1.0 - preload_probability) / 4.0)

    for endpoint in (0.0, 1.0):
        cfg.reset_mixture_near_object_preloaded_probability = endpoint
        FrankaPourEnv._build_reset_mixture_banks(
            env,
            object(),
            torch.zeros((source_count, 9)),
            torch.zeros((source_count, 7)),
            torch.arange(source_count),
            torch.arange(7),
            torch.tensor((7, 8)),
        )
        preload = env._reset_mixture_near_object_preloaded_t
        weights = env._reset_mixture_near_object_weights_t
        assert bool(torch.all(torch.isfinite(weights) & (weights >= 0.0)))
        assert float(weights[preload].sum()) == pytest.approx(endpoint)
        assert float(weights[~preload].sum()) == pytest.approx(1.0 - endpoint)
        for source_row in range(source_count):
            source = env._reset_mixture_near_object_source_rows_t == source_row
            assert float(weights[source].sum()) == pytest.approx(1.0 / source_count)

    cfg.reset_mixture_near_object_preloaded_probability = preload_probability

    def reject_one_exact_grasp(_builder, robot_q, *_args, **_kwargs):
        collision_free = torch.ones(robot_q.shape[0], dtype=torch.bool)
        if robot_q.shape[0] > 24:
            collision_free[24] = False
        return collision_free

    monkeypatch.setattr(pour_env_module, "collision_free_reset_candidates", reject_one_exact_grasp)
    FrankaPourEnv._build_reset_mixture_banks(
        env,
        object(),
        torch.zeros((source_count, 9)),
        torch.zeros((source_count, 7)),
        torch.arange(source_count),
        torch.arange(7),
        torch.tensor((7, 8)),
    )

    preload = env._reset_mixture_near_object_preloaded_t
    weights = env._reset_mixture_near_object_weights_t
    source_rows = env._reset_mixture_near_object_source_rows_t
    assert source_rows[preload].tolist() == list(range(1, source_count))
    assert float(weights[preload].sum()) == pytest.approx(preload_probability)
    assert float(weights[~preload].sum()) == pytest.approx(1.0 - preload_probability)
    for source_row in range(source_count):
        source = source_rows == source_row
        assert float(weights[source].sum()) == pytest.approx(1.0 / source_count)
    assert float(weights[(source_rows == 0) & preload].sum()) == 0.0
    assert float(weights[(source_rows == 0) & ~preload].sum()) == pytest.approx(1.0 / source_count)
    for source_row in range(1, source_count):
        source = source_rows == source_row
        assert float(weights[source & preload].sum()) == pytest.approx(preload_probability / (source_count - 1))
        assert float(weights[source & ~preload].sum()) == pytest.approx(
            1.0 / source_count - preload_probability / (source_count - 1)
        )

    cfg.reset_mixture_near_object_preloaded_probability = 0.9
    with pytest.raises(RuntimeError, match="requested=0.900000, maximum=0.875000"):
        FrankaPourEnv._build_reset_mixture_banks(
            env,
            object(),
            torch.zeros((source_count, 9)),
            torch.zeros((source_count, 7)),
            torch.arange(source_count),
            torch.arange(7),
            torch.tensor((7, 8)),
        )


def test_grasped_bank_covers_both_screened_segments_at_eighths(monkeypatch):
    env_count = 16
    env = SimpleNamespace(
        device="cpu",
        curriculum_randomization_level=torch.zeros(env_count, dtype=torch.long),
        _randomized_extent_index_pools=(torch.tensor((0,)),),
        _randomized_extent_index_weights=(torch.ones(1),),
        _last_source_bank_index=torch.full((env_count,), -1, dtype=torch.long),
        _last_arm_bank_index=torch.full((env_count,), -1, dtype=torch.long),
        _last_target_bank_index=torch.full((env_count,), -1, dtype=torch.long),
        _randomized_source_pos_bank_t=torch.zeros((1, 3)),
        _randomized_source_quat_bank_t=torch.tensor(((0.0, 0.0, 0.0, 1.0),)),
        _randomized_target_pos_bank_t=torch.zeros((1, 3)),
        _randomized_grasp_arm_q_bank_t=torch.zeros((1, 7)),
        _randomized_carry_arm_q_bank_t=torch.full((1, 7), 8.0),
        _randomized_pour_arm_q_bank_t=torch.full((1, 7), 16.0),
        _randomized_tilt_arm_q_bank_t=torch.full((1, 7), 24.0),
    )
    monkeypatch.setattr(
        pour_env_module,
        "sample_index_pools",
        lambda *_args, **_kwargs: torch.zeros(env_count, dtype=torch.long),
    )

    def sample_every_phase(high, size, *, device):
        assert high == 16
        assert size == (env_count,)
        return torch.arange(env_count, device=device)

    monkeypatch.setattr(torch, "randint", sample_every_phase)
    arm_q = torch.full((env_count, 7), -1.0)
    preloaded = FrankaPourEnv._apply_reset_mixture_bank(
        env,
        torch.arange(env_count),
        torch.full((env_count,), RESET_MIXTURE_REGION_NAMES.index("grasped"), dtype=torch.long),
        arm_q,
        torch.full((env_count, 3), -1.0),
        torch.full((env_count, 4), -1.0),
        torch.full((env_count, 3), -1.0),
    )

    expected = torch.arange(env_count, dtype=torch.float32).unsqueeze(-1).expand(-1, 7)
    torch.testing.assert_close(arm_q, expected)
    assert not bool(preloaded.any())


class _RobotRecorder:
    def __init__(self):
        self.position_writes = []
        self.position_targets = []
        self.data = SimpleNamespace(body_link_pose_w=torch.empty(0))

    def write_joint_position_to_sim_index(self, *, position, **_kwargs):
        self.position_writes.append(position.clone())

    def write_joint_velocity_to_sim_index(self, **_kwargs):
        pass

    def set_joint_position_target_index(self, *, target, **_kwargs):
        self.position_targets.append(target.clone())


class _RigidRecorder:
    def __init__(self):
        self.root_pose = None

    def write_root_pose_to_sim_index(self, *, root_pose, **_kwargs):
        self.root_pose = root_pose.clone()

    def write_root_velocity_to_sim_index(self, **_kwargs):
        pass


class _MediaRecorder:
    def write_particle_pos_to_sim_index(self, *_args, **_kwargs):
        pass

    def write_particle_velocity_to_sim_index(self, *_args, **_kwargs):
        pass


def test_preloaded_near_object_mask_sets_contact_fingers_and_drive_target(monkeypatch):
    near_object = RESET_MIXTURE_REGION_NAMES.index("near_object")
    robot = _RobotRecorder()
    source_cup = _RigidRecorder()
    gripper_targets = []
    gripper_action = SimpleNamespace(
        set_reset_position=lambda target, **_kwargs: gripper_targets.append(target.clone())
    )
    env = SimpleNamespace(
        device="cpu",
        num_envs=2,
        cfg=SimpleNamespace(
            cup_reset_pos=(0.5, 0.0, 0.0),
            target_cup_reset_pos=(0.5, -0.2, 0.0),
            cup_grasp_height=0.05,
            cup_grasp_box_half=(0.03, 0.03, 0.06),
            gripper_preload_pos=0.024,
            gripper_open_pos=0.04,
        ),
        curriculum_stage=torch.ones(2, dtype=torch.long),
        _curriculum_arm_q_t=torch.zeros((5, 7)),
        _curriculum_cup_quat_t=torch.tensor(((0.0, 0.0, 0.0, 1.0),) * 5),
        _curriculum_finger_pos_t=torch.full((5,), 0.04),
        _grasp_stage_index=1,
        _approach_stage_index=2,
        _full_stage_index=3,
        _randomized_stage_index=4,
        _reach_grasp_arm_q_bank_t=torch.zeros((1, 7)),
        _last_source_bank_index=torch.full((2,), -1, dtype=torch.long),
        _last_arm_bank_index=torch.full((2,), -1, dtype=torch.long),
        _last_target_bank_index=torch.full((2,), -1, dtype=torch.long),
        _uses_reset_mixture=True,
        reset_region_id=torch.full((2,), near_object, dtype=torch.long),
        _reset_mixture_near_object_weights_t=torch.ones(2),
        _reset_mixture_near_object_source_rows_t=torch.zeros(2, dtype=torch.long),
        _reset_mixture_near_object_arm_q_t=torch.zeros((2, 7)),
        _reset_mixture_near_object_preloaded_t=torch.tensor((False, True)),
        _randomized_source_pos_bank_t=torch.tensor(((0.5, 0.0, 0.0),)),
        _randomized_source_quat_bank_t=torch.tensor(((0.0, 0.0, 0.0, 1.0),)),
        _randomized_target_pos_bank_t=torch.tensor(((0.5, -0.2, 0.0),)),
        _robot=robot,
        _arm_joint_ids=torch.arange(7),
        _finger_joint_ids=torch.tensor((7, 8)),
        action_manager=SimpleNamespace(
            get_term=lambda name: gripper_action if name == "gripper_action" else SimpleNamespace()
        ),
        env_origins=torch.zeros((2, 3)),
        _desired_grasp_tcp_quat_c=torch.tensor(((0.0, 0.0, 0.0, 1.0),) * 2),
        _source_cup=source_cup,
        _target_cup=_RigidRecorder(),
        _media=_MediaRecorder(),
        _particle_region_cache=object(),
        _particle_region_cache_step=1,
        episode_succeeded=torch.ones(2, dtype=torch.bool),
        ep_max_target_frac=torch.ones(2),
        _success_dwell_count=torch.ones(2, dtype=torch.long),
        _lost_grasp_dwell_count=torch.ones(2, dtype=torch.long),
        _lifted_grasp_seen=torch.ones(2, dtype=torch.bool),
        _target_entry_seen=torch.ones((2, 1), dtype=torch.bool),
        _held_delivered=torch.ones((2, 1), dtype=torch.bool),
        _held_delivery_tracker_step=1,
    )
    env._apply_reset_mixture_bank = MethodType(FrankaPourEnv._apply_reset_mixture_bank, env)
    env.tcp_pose_e = lambda: torch.tensor(((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),) * 2)
    env._sample_cup_media = lambda cup_pos, _cup_quat: torch.zeros((cup_pos.shape[0], 1, 3))
    monkeypatch.setattr(torch, "multinomial", lambda *_args, **_kwargs: torch.tensor((0, 1)))
    monkeypatch.setattr(pour_env_module.NewtonManager, "reset_solver_state", lambda **_kwargs: None)
    FrankaPourEnv.reset_pour_scene(env, torch.arange(2))

    torch.testing.assert_close(robot.position_writes[1], torch.tensor(((0.04, 0.04), (0.03, 0.03))))
    torch.testing.assert_close(robot.position_targets[1], torch.tensor(((0.04, 0.04), (0.024, 0.024))))
    torch.testing.assert_close(gripper_targets[0], torch.tensor(((0.04,), (0.024,))))
    torch.testing.assert_close(source_cup.root_pose[:, :3], torch.tensor(((0.5, 0.0, 0.0),) * 2))
