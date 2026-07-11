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
    source_count = 8
    source_x = torch.arange(source_count, dtype=torch.float32) * 0.01 + 0.45
    source_positions = torch.stack((source_x, torch.zeros_like(source_x), torch.zeros_like(source_x)), dim=-1)
    half_sqrt = 2.0**-0.5
    source_quaternions = torch.tensor(((0.0, 0.0, half_sqrt, half_sqrt),) * source_count)
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
        reset_mixture_near_object_preloaded_probability=0.15,
        reset_mixture_near_object_local_position_half_range=(0.02, 0.02, 0.02),
        reset_mixture_near_object_local_rotation_half_range=(0.1, 0.1, 0.1),
        reset_mixture_near_object_local_sample_count=25,
        curriculum_randomized_reset_ik_iterations=4,
        curriculum_randomized_reset_ik_max_cost=1.0,
        curriculum_randomized_reset_ik_joint_margin=0.01,
        tcp_body_name="panda_hand",
        tcp_offset_pos=(0.0, 0.0, 0.0),
        tcp_offset_rot=(0.0, 0.0, 0.0, 1.0),
        cup_grasp_tcp_quat_c=(0.0, 0.0, 0.0, 1.0),
        cup_grasp_height=0.05,
        gripper_open_pos=0.04,
        cup_grasp_box_half=(0.03, 0.03, 0.06),
    )
    env = SimpleNamespace(
        cfg=cfg,
        device="cpu",
        _randomized_extent_index_pools=(torch.arange(source_count),),
        _randomized_source_pos_bank_t=source_positions,
        _randomized_source_quat_bank_t=source_quaternions,
        _randomized_source_yaw_bank_t=torch.full((source_count,), torch.pi / 2.0),
        _randomized_target_pos_bank_t=source_positions + torch.tensor((0.0, -0.2, 0.0)),
        _randomized_tcp_pos_bank_t=source_positions + torch.tensor((0.0, 0.2, 0.2)),
        _randomized_arm_q_bank_t=torch.zeros((source_count, 7)),
        _randomized_pregrasp_arm_q_bank_t=torch.ones((source_count, 7)),
        _randomized_midgrasp_arm_q_bank_t=torch.full((source_count, 7), 2.0),
        _randomized_grasp_arm_q_bank_t=torch.full((source_count, 7), 3.0),
        _target_vertices=None,
        _target_indices=None,
        _collider_margin=0.0,
        _joint_pos_limits_t=torch.tensor([[[-100.0, 100.0]] * 7]),
        _arm_joint_ids=torch.arange(7),
    )
    env._independent_target_clearance = lambda source, target: torch.ones_like(target, dtype=torch.float32)
    env._collision_free_ik_candidates = lambda _builder, waypoints: torch.ones(waypoints[0].shape[:2], dtype=torch.bool)

    class FakePoseComponent:
        def __init__(self):
            self.targets = []

        def set_target_positions(self, targets):
            self.targets.append(pour_env_module.wp.to_torch(targets).clone())

        def set_target_rotations(self, targets):
            self.targets.append(pour_env_module.wp.to_torch(targets).clone())

    class FakePoseObjective:
        def __init__(self):
            self.position_objective = FakePoseComponent()
            self.rotation_objective = FakePoseComponent()

    class FakeIKSolver:
        nonfinite_row = None
        over_cost_row = None
        near_limit_row = None
        instances = []

        def __init__(self, solver_cfg, *, num_envs, objectives, **_kwargs):
            assert solver_cfg.sampler == "none"
            assert solver_cfg.n_seeds == 1
            self.cfg = solver_cfg
            self.objectives_by_name = {objectives[0].name: FakePoseObjective()}
            self.costs = pour_env_module.wp.zeros(num_envs, dtype=pour_env_module.wp.float32, device="cpu")
            if self.over_cost_row is not None:
                pour_env_module.wp.to_torch(self.costs)[self.over_cost_row] = 2.0
            self.instances.append(self)

        def solve(self, initial_guess):
            solved = pour_env_module.wp.to_torch(initial_guess).clone()
            solved[:, 0] = 10.0 + torch.arange(solved.shape[0], dtype=solved.dtype) * 0.001
            if self.nonfinite_row is not None:
                solved[self.nonfinite_row, 0] = float("nan")
            if self.near_limit_row is not None:
                solved[self.near_limit_row, 0] = 99.995
            return pour_env_module.wp.from_torch(solved.contiguous(), dtype=pour_env_module.wp.float32)

    monkeypatch.setattr(pour_env_module, "NewtonIKSolver", FakeIKSolver)
    model = SimpleNamespace(device=pour_env_module.wp.get_device("cpu"))
    collision_calls = []

    def build_bank(
        *,
        reject_exact=False,
        reject_local=False,
        reject_first_source_local=False,
        reject_all_local=False,
        nonfinite_local=False,
        over_cost_local=False,
        near_limit_local=False,
    ):
        collision_calls.clear()
        FakeIKSolver.instances.clear()
        FakeIKSolver.nonfinite_row = 0 if nonfinite_local else None
        FakeIKSolver.over_cost_row = 0 if over_cost_local else None
        FakeIKSolver.near_limit_row = 1 if near_limit_local else None

        def collision_screen(_builder, robot_q, *_args, **_kwargs):
            call = len(collision_calls)
            collision_calls.append(robot_q.shape[0])
            accepted = torch.ones(robot_q.shape[0], dtype=torch.bool)
            if call == 1 and reject_exact:
                accepted[24] = False
            if call == 2 and reject_all_local:
                accepted[:] = False
            elif call == 2 and reject_first_source_local:
                accepted[:24] = False
            elif call == 2 and reject_local:
                accepted[0] = False
            return accepted

        monkeypatch.setattr(pour_env_module, "collision_free_reset_candidates", collision_screen)
        FrankaPourEnv._build_reset_mixture_banks(
            env,
            object(),
            model,
            0,
            torch.zeros((source_count, 9)),
            torch.zeros((source_count, 7)),
            torch.arange(source_count),
            torch.arange(7),
            torch.tensor((7, 8)),
        )

    build_bank()
    preload_probability = cfg.reset_mixture_near_object_preloaded_probability
    preload = env._reset_mixture_near_object_preloaded_t
    weights = env._reset_mixture_near_object_weights_t
    source_rows = env._reset_mixture_near_object_source_rows_t
    assert preload.dtype == torch.bool
    assert preload.shape == weights.shape == source_rows.shape
    assert int(preload.sum()) == source_count
    assert int((~preload).sum()) == (22 + 25) * source_count
    assert float(weights[preload].sum()) == pytest.approx(preload_probability)
    assert float(weights[~preload].sum()) == pytest.approx(1.0 - preload_probability)
    assert source_rows[preload].tolist() == list(range(source_count))
    torch.testing.assert_close(env._reset_mixture_near_object_arm_q_t[preload], torch.full((source_count, 7), 3.0))
    local_targets = (
        FakeIKSolver.instances[0]
        .objectives_by_name["near_object_local_tcp"]
        .position_objective.targets[0]
        .reshape(source_count, 24, 3)
    )
    local_rotations = (
        FakeIKSolver.instances[0]
        .objectives_by_name["near_object_local_tcp"]
        .rotation_objective.targets[0]
        .reshape(source_count, 24, 4)
    )
    assert local_targets.shape == (source_count, 24, 3)
    torch.testing.assert_close(
        local_targets[1] - local_targets[0],
        (source_positions[1] - source_positions[0]).expand_as(local_targets[0]),
    )
    local_samples = pour_env_module.symmetric_local_pose_samples(
        cfg.reset_mixture_near_object_local_position_half_range,
        cfg.reset_mixture_near_object_local_rotation_half_range,
        cfg.reset_mixture_near_object_local_sample_count,
    )[1:]
    exact_tcp_position = source_positions[0] + pour_env_module.math_utils.quat_apply(
        source_quaternions[0], torch.tensor((0.0, 0.0, cfg.cup_grasp_height))
    )
    local_delta_quaternions = pour_env_module.math_utils.quat_from_euler_xyz(
        local_samples[:, 3], local_samples[:, 4], local_samples[:, 5]
    )
    expected_local_targets, expected_local_rotations = pour_env_module.math_utils.combine_frame_transforms(
        exact_tcp_position.expand(24, -1),
        source_quaternions[0].expand(24, -1),
        local_samples[:, :3],
        local_delta_quaternions,
    )
    torch.testing.assert_close(local_targets[0], expected_local_targets)
    torch.testing.assert_close(local_rotations[0], expected_local_rotations)
    open_weights = weights[~preload]
    bridge_groups = (torch.arange(22).repeat(source_count) >= 16).long()
    open_groups = torch.cat((bridge_groups, torch.full((25 * source_count,), 2)))
    for group, expected_mass in enumerate(cfg.reset_mixture_near_object_open_phase_probabilities):
        assert float(open_weights[open_groups == group].sum()) == pytest.approx(
            (1.0 - preload_probability) * expected_mass
        )
    for source_row in range(source_count):
        source = source_rows == source_row
        assert float(weights[source & preload].sum()) == pytest.approx(preload_probability / source_count)
        assert float(weights[source & ~preload].sum()) == pytest.approx((1.0 - preload_probability) / source_count)
    cell_ids = torch.div(source_rows, 2, rounding_mode="floor")
    for cell_id in range(4):
        cell = cell_ids == cell_id
        assert float(weights[cell & preload].sum()) == pytest.approx(preload_probability / 4.0)
        assert float(weights[cell & ~preload].sum()) == pytest.approx((1.0 - preload_probability) / 4.0)

    for endpoint in (0.0, 1.0):
        cfg.reset_mixture_near_object_preloaded_probability = endpoint
        build_bank()
        preload = env._reset_mixture_near_object_preloaded_t
        weights = env._reset_mixture_near_object_weights_t
        assert bool(torch.all(torch.isfinite(weights) & (weights >= 0.0)))
        assert float(weights[preload].sum()) == pytest.approx(endpoint)
        assert float(weights[~preload].sum()) == pytest.approx(1.0 - endpoint)

    cfg.reset_mixture_near_object_preloaded_probability = preload_probability
    build_bank(reject_exact=True)
    preload = env._reset_mixture_near_object_preloaded_t
    weights = env._reset_mixture_near_object_weights_t
    source_rows = env._reset_mixture_near_object_source_rows_t
    assert not bool(torch.any(source_rows == 0))
    assert source_rows[preload].tolist() == list(range(1, source_count))
    for source_row in range(1, source_count):
        source = source_rows == source_row
        expected_source_mass = 0.25 if source_row == 1 else 0.125
        assert float(weights[source].sum()) == pytest.approx(expected_source_mass)
    retained_cells = torch.div(source_rows, 2, rounding_mode="floor")
    for cell_id in range(4):
        assert float(weights[retained_cells == cell_id].sum()) == pytest.approx(0.25)

    build_bank(reject_local=True, nonfinite_local=True)
    preload = env._reset_mixture_near_object_preloaded_t
    weights = env._reset_mixture_near_object_weights_t
    source_rows = env._reset_mixture_near_object_source_rows_t
    arm_q = env._reset_mixture_near_object_arm_q_t
    assert int((~preload).sum()) == (22 + 25) * source_count - 2
    assert bool(torch.all(torch.isfinite(weights) & (weights >= 0.0)))
    assert not bool(torch.any(~torch.isfinite(arm_q)))
    assert not bool(torch.any(torch.isclose(arm_q[:, 0], torch.tensor(10.0))))
    assert not bool(torch.any(torch.isclose(arm_q[:, 0], torch.tensor(10.001))))
    assert bool(torch.any(torch.isclose(arm_q[:, 0], torch.tensor(10.002))))
    assert float(weights[preload].sum()) == pytest.approx(preload_probability)
    assert float(weights[~preload].sum()) == pytest.approx(1.0 - preload_probability)

    build_bank(over_cost_local=True, near_limit_local=True)
    preload = env._reset_mixture_near_object_preloaded_t
    arm_q = env._reset_mixture_near_object_arm_q_t
    assert int((~preload).sum()) == (22 + 25) * source_count - 2
    assert collision_calls[-1] == source_count * 24 - 2
    assert not bool(torch.any(torch.isclose(arm_q[:, 0], torch.tensor(10.0))))
    assert not bool(torch.any(torch.isclose(arm_q[:, 0], torch.tensor(99.995))))
    assert bool(torch.any(torch.isclose(arm_q[:, 0], torch.tensor(10.002))))

    build_bank(reject_first_source_local=True)
    source_rows = env._reset_mixture_near_object_source_rows_t
    weights = env._reset_mixture_near_object_weights_t
    assert not bool(torch.any(source_rows == 0))
    retained_cells = torch.div(source_rows, 2, rounding_mode="floor")
    for cell_id in range(4):
        assert float(weights[retained_cells == cell_id].sum()) == pytest.approx(0.25)

    with pytest.raises(RuntimeError, match="removed every perturbed source row"):
        build_bank(reject_all_local=True)


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
        self.data = _PoseDataRecorder()

    def write_joint_position_to_sim_index(self, *, position, **_kwargs):
        self.position_writes.append(position.clone())

    def write_joint_velocity_to_sim_index(self, **_kwargs):
        pass

    def set_joint_position_target_index(self, *, target, **_kwargs):
        self.position_targets.append(target.clone())


class _RigidRecorder:
    def __init__(self):
        self.root_pose = None
        self.data = _PoseDataRecorder()

    def write_root_pose_to_sim_index(self, *, root_pose, **_kwargs):
        self.root_pose = root_pose.clone()

    def write_root_velocity_to_sim_index(self, **_kwargs):
        pass


class _MediaRecorder:
    def write_particle_pos_to_sim_index(self, *_args, **_kwargs):
        pass

    def write_particle_velocity_to_sim_index(self, *_args, **_kwargs):
        pass


class _PoseDataRecorder:
    def __init__(self):
        self.body_link_pose_reads = 0

    @property
    def body_link_pose_w(self):
        self.body_link_pose_reads += 1
        return torch.empty(0)


def test_preloaded_near_object_mask_sets_contact_fingers_and_drive_target(monkeypatch):
    near_object = RESET_MIXTURE_REGION_NAMES.index("near_object")
    robot = _RobotRecorder()
    source_cup = _RigidRecorder()
    target_cup = _RigidRecorder()
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
        _target_cup=target_cup,
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
    solver_resets = []
    monkeypatch.setattr(
        pour_env_module.NewtonManager,
        "reset_solver_state",
        lambda **kwargs: solver_resets.append(kwargs),
    )
    FrankaPourEnv.reset_pour_scene(env, torch.arange(2))

    torch.testing.assert_close(robot.position_writes[1], torch.tensor(((0.04, 0.04), (0.03, 0.03))))
    torch.testing.assert_close(robot.position_targets[1], torch.tensor(((0.04, 0.04), (0.024, 0.024))))
    torch.testing.assert_close(gripper_targets[0], torch.tensor(((0.04,), (0.024,))))
    torch.testing.assert_close(source_cup.root_pose[:, :3], torch.tensor(((0.5, 0.0, 0.0),) * 2))
    assert robot.data.body_link_pose_reads == 0
    assert source_cup.data.body_link_pose_reads == 1
    assert target_cup.data.body_link_pose_reads == 0
    assert len(solver_resets) == 1
    assert solver_resets[0]["flags"] == pour_env_module.newton.StateFlags.PARTICLE
