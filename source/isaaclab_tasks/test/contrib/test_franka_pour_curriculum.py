# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Franka Pour curriculum and reset-relative actions."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import CurriculumTermCfg

from isaaclab_tasks.contrib.franka_pour.mdp.actions import (
    CurriculumGripperPositionAction,
    CurriculumJointPositionAction,
    TrajectoryJointPositionAction,
    _bilateral_gripper_preload,
)
from isaaclab_tasks.contrib.franka_pour.mdp.curriculums import PourCurriculum


class FakeCurriculumEnv:
    """Minimal vectorized environment state consumed by :class:`PourCurriculum`."""

    def __init__(
        self,
        *,
        frozen: bool = False,
        start_stage: int = 0,
        start_randomization_level: int = 0,
        replay_fraction: float = 0.0,
    ):
        self.num_envs = 4
        self.device = "cpu"
        self.cfg = SimpleNamespace(
            curriculum_stage_names=("pour", "carry", "grasp", "full", "randomized"),
            curriculum_target_frac=(0.1, 0.2, 0.3, 0.4, 0.4),
            curriculum_start_stage=start_stage,
            curriculum_randomization_extent_levels=(2.0 / 3.0, 5.0 / 6.0, 1.0),
            curriculum_randomization_start_level=start_randomization_level,
            curriculum_freeze=frozen,
            curriculum_success_threshold=0.75,
            curriculum_min_resets_per_stage=2,
            curriculum_previous_stage_replay_fraction=replay_fraction,
        )
        self.curriculum_stage = torch.zeros(self.num_envs, dtype=torch.long)
        self.curriculum_randomization_level = torch.zeros(self.num_envs, dtype=torch.long)
        self.pour_target_frac = torch.zeros(self.num_envs)
        self.episode_succeeded = torch.zeros(self.num_envs, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.ep_max_target_frac = torch.zeros(self.num_envs)

    def set_curriculum_stage(self, env_ids, stage: int) -> None:
        self.curriculum_stage[env_ids] = stage
        self.pour_target_frac[env_ids] = self.cfg.curriculum_target_frac[stage]

    def set_curriculum_randomization_level(self, env_ids, level: int) -> None:
        self.curriculum_randomization_level[env_ids] = level

    @staticmethod
    def curriculum_randomization_bank_size(level: int) -> int:
        return (37, 88, 245)[level]


def test_curriculum_ignores_initial_reset_and_advances_only_reset_worlds():
    env = FakeCurriculumEnv()
    term = PourCurriculum(CurriculumTermCfg(func=PourCurriculum), env)

    initial = term(env, torch.arange(env.num_envs))
    assert set(initial) == {
        "stage",
        "randomization_level",
        "randomization_extent_fraction",
        "eligible_bank_rows",
        "success_rate",
        "target_frac",
        "completed_episodes",
        "mastered",
    }
    assert term.resets_in_stage == 0

    env.episode_length_buf[:2] = 10
    env.episode_succeeded[:2] = True
    env.ep_max_target_frac[:2] = torch.tensor([0.4, 0.5])
    metrics = term(env, torch.tensor([0, 1]))

    assert term.stage == 1
    assert term.resets_in_stage == 0
    assert env.curriculum_stage.tolist() == [1, 1, 0, 0]
    assert env.pour_target_frac.tolist() == pytest.approx([0.2, 0.2, 0.1, 0.1])
    assert metrics == pytest.approx(
        {
            "stage": 1.0,
            "randomization_level": 0.0,
            "randomization_extent_fraction": 2.0 / 3.0,
            "eligible_bank_rows": 37.0,
            "success_rate": 0.0,
            "target_frac": 0.2,
            "completed_episodes": 0.0,
            "mastered": 0.0,
            "mean_peak_target_frac": 0.45,
        }
    )


def test_curriculum_requires_configured_number_of_completed_episodes():
    env = FakeCurriculumEnv()
    env.cfg.curriculum_min_resets_per_stage = 500
    term = PourCurriculum(CurriculumTermCfg(func=PourCurriculum), env)
    env.episode_length_buf[:] = 10
    env.episode_succeeded[:] = True

    for _ in range(124):
        term(env, torch.arange(env.num_envs))
    term(env, torch.tensor([0, 1, 2]))
    assert term.resets_in_stage == 499
    assert term.stage == 0

    term(env, torch.tensor([3]))
    assert term.stage == 1
    assert term.resets_in_stage == 0


def test_lagging_old_stage_episode_does_not_change_new_stage_statistics():
    env = FakeCurriculumEnv()
    term = PourCurriculum(CurriculumTermCfg(func=PourCurriculum), env)
    env.episode_length_buf[:2] = 10
    env.episode_succeeded[:2] = True
    term(env, torch.tensor([0, 1]))

    env.episode_length_buf[2] = 10
    env.episode_succeeded[2] = True
    term(env, torch.tensor([2]))

    assert env.curriculum_stage.tolist() == [1, 1, 1, 0]
    assert term.stage == 1
    assert term.resets_in_stage == 0
    assert term.success_rate == 0.0


def test_curriculum_replays_previous_stage_without_counting_it(monkeypatch):
    env = FakeCurriculumEnv(start_stage=1, replay_fraction=0.5)
    term = PourCurriculum(CurriculumTermCfg(func=PourCurriculum), env)
    draws = iter((torch.tensor([0.1, 0.9, 0.2, 0.8]), torch.tensor([0.9, 0.9])))
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: next(draws))

    term(env, torch.arange(env.num_envs))
    assert env.curriculum_stage.tolist() == [0, 1, 0, 1]

    env.episode_length_buf[[0, 2]] = 10
    env.episode_succeeded[[0, 2]] = True
    term(env, torch.tensor([0, 2]))
    assert term.resets_in_stage == 0


def test_randomized_curriculum_replays_previous_extent_without_counting_it(monkeypatch):
    env = FakeCurriculumEnv(start_stage=4, start_randomization_level=1, replay_fraction=0.5)
    term = PourCurriculum(CurriculumTermCfg(func=PourCurriculum), env)
    draws = iter((torch.tensor([0.1, 0.9, 0.2, 0.8]), torch.tensor([0.9, 0.9])))
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: next(draws))

    term(env, torch.arange(env.num_envs))
    assert env.curriculum_stage.tolist() == [4, 4, 4, 4]
    assert env.curriculum_randomization_level.tolist() == [0, 1, 0, 1]

    env.episode_length_buf[[0, 2]] = 10
    env.episode_succeeded[[0, 2]] = True
    term(env, torch.tensor([0, 2]))
    assert term.resets_in_stage == 0


def test_frozen_curriculum_stays_at_configured_stage(monkeypatch):
    env = FakeCurriculumEnv(
        frozen=True,
        start_stage=4,
        start_randomization_level=2,
        replay_fraction=0.5,
    )
    term = PourCurriculum(CurriculumTermCfg(func=PourCurriculum), env)

    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: pytest.fail("frozen curriculum sampled replay"))
    term(env, torch.arange(env.num_envs))
    assert env.curriculum_stage.tolist() == [4, 4, 4, 4]
    assert env.curriculum_randomization_level.tolist() == [2, 2, 2, 2]


def test_curriculum_success_window_weights_each_completed_episode_equally():
    env = FakeCurriculumEnv(frozen=True)
    env.cfg.curriculum_min_resets_per_stage = 5
    term = PourCurriculum(CurriculumTermCfg(func=PourCurriculum), env)
    env.episode_length_buf[:] = 10

    env.episode_succeeded[:] = True
    term(env, torch.arange(env.num_envs))
    env.episode_succeeded[0] = False
    metrics = term(env, torch.tensor([0]))

    assert term.resets_in_stage == 5
    assert term.success_rate == pytest.approx(0.8)
    assert metrics["success_rate"] == pytest.approx(0.8)


def test_curriculum_success_window_evicts_oldest_completed_episodes():
    env = FakeCurriculumEnv(frozen=True)
    env.cfg.curriculum_min_resets_per_stage = 4
    term = PourCurriculum(CurriculumTermCfg(func=PourCurriculum), env)
    env.episode_length_buf[:] = 10

    env.episode_succeeded[:] = True
    term(env, torch.arange(env.num_envs))
    env.episode_succeeded[:2] = False
    term(env, torch.tensor([0, 1]))

    assert term.resets_in_stage == 6
    assert term.success_rate == pytest.approx(0.5)


def test_curriculum_promotes_at_exact_window_threshold():
    env = FakeCurriculumEnv()
    env.cfg.curriculum_min_resets_per_stage = 5
    env.cfg.curriculum_success_threshold = 0.8
    term = PourCurriculum(CurriculumTermCfg(func=PourCurriculum), env)
    env.episode_length_buf[:] = 10

    env.episode_succeeded[:] = True
    term(env, torch.arange(env.num_envs))
    env.episode_succeeded[0] = False
    term(env, torch.tensor([0]))

    assert term.stage == 1
    assert term.resets_in_stage == 0
    assert term.success_rate == 0.0


def test_curriculum_mastery_requires_a_full_success_window():
    env = FakeCurriculumEnv(frozen=True, start_stage=4, start_randomization_level=2)
    env.cfg.curriculum_min_resets_per_stage = 4
    term = PourCurriculum(CurriculumTermCfg(func=PourCurriculum), env)
    env.episode_length_buf[:] = 10

    env.episode_succeeded[0] = True
    metrics = term(env, torch.tensor([0]))
    assert metrics["success_rate"] == 1.0
    assert metrics["mastered"] == 0.0

    env.episode_succeeded[1:4] = torch.tensor([True, True, False])
    metrics = term(env, torch.tensor([1, 2, 3]))
    assert metrics["success_rate"] == pytest.approx(0.75)
    assert metrics["mastered"] == 1.0


def test_final_stage_advances_nested_randomization_frontiers_before_mastery():
    env = FakeCurriculumEnv(start_stage=3)
    env.cfg.curriculum_success_threshold = 1.0
    term = PourCurriculum(CurriculumTermCfg(func=PourCurriculum), env)
    env.episode_length_buf[:] = 10
    env.episode_succeeded[:] = True

    metrics = term(env, torch.tensor([0, 1]))
    assert term.stage == 4
    assert term.randomization_level == 0
    assert term.resets_in_stage == 0
    assert metrics["mastered"] == 0.0
    assert env.curriculum_randomization_level[:2].tolist() == [0, 0]

    # Reset one lagging stage-three world onto level zero without counting its old episode.
    term(env, torch.tensor([2]))
    assert env.curriculum_stage[2].item() == 4
    assert env.curriculum_randomization_level[2].item() == 0

    metrics = term(env, torch.tensor([0, 1]))
    assert term.stage == 4
    assert term.randomization_level == 1
    assert term.resets_in_stage == 0
    assert metrics["randomization_extent_fraction"] == pytest.approx(5.0 / 6.0)
    assert metrics["eligible_bank_rows"] == 88.0
    assert env.curriculum_randomization_level[:2].tolist() == [1, 1]

    # This in-flight episode began at level zero. Its completion must not enter level one's
    # statistics, but its replacement episode must receive the current frontier tag.
    assert env.curriculum_randomization_level[2].item() == 0
    term(env, torch.tensor([2]))
    assert term.randomization_level == 1
    assert term.resets_in_stage == 0
    assert term.success_rate == 0.0
    assert env.curriculum_randomization_level[2].item() == 1

    term(env, torch.tensor([0, 1]))
    assert term.randomization_level == 2
    assert term.resets_in_stage == 0
    assert term.success_rate == 0.0

    metrics = term(env, torch.tensor([0, 1]))
    assert term.stage == 4
    assert term.randomization_level == 2
    assert term.resets_in_stage == 2
    assert term.success_rate == 1.0
    assert metrics["mastered"] == 1.0
    assert metrics["randomization_extent_fraction"] == 1.0
    assert metrics["eligible_bank_rows"] == 245.0


def test_curriculum_rejects_nonpositive_success_window():
    env = FakeCurriculumEnv()
    env.cfg.curriculum_min_resets_per_stage = 0

    with pytest.raises(ValueError, match="curriculum_min_resets_per_stage must be positive"):
        PourCurriculum(CurriculumTermCfg(func=PourCurriculum), env)


def test_curriculum_joint_action_offset_updates_selected_worlds_only():
    action = CurriculumJointPositionAction.__new__(CurriculumJointPositionAction)
    action._offset = torch.zeros((4, 7))
    target = torch.arange(14, dtype=torch.float32).reshape(2, 7)

    action.set_action_offset(target, env_ids=torch.tensor([1, 3]))

    torch.testing.assert_close(action.action_offset[[1, 3]], target)
    torch.testing.assert_close(action.action_offset[[0, 2]], torch.zeros((2, 7)))
    with pytest.raises(ValueError, match="shape"):
        action.set_action_offset(torch.zeros((1, 7)), env_ids=torch.tensor([1, 3]))


def test_trajectory_phase_action_modulates_around_nominal_speed():
    phase_action = torch.tensor([-1.0, 0.0, 1.0])

    phase_speed = TrajectoryJointPositionAction._phase_speed_command(phase_action)

    torch.testing.assert_close(phase_speed, torch.tensor([0.75, 1.0, 1.25]))


def test_phase_gate_never_rewinds_a_later_curriculum_reset():
    current = torch.tensor([0.0, 0.12, 0.24, 0.40, 0.62])

    approach_limit = TrajectoryJointPositionAction._monotonic_gate_limit(current, 0.12)
    grasp_limit = TrajectoryJointPositionAction._monotonic_gate_limit(current, 0.24)

    torch.testing.assert_close(approach_limit, torch.tensor([0.12, 0.12, 0.24, 0.40, 0.62]))
    torch.testing.assert_close(grasp_limit, torch.tensor([0.24, 0.24, 0.24, 0.40, 0.62]))


def test_carry_stage_requires_fresh_grasp_dwell_despite_starting_after_grasp_waypoint():
    action = TrajectoryJointPositionAction.__new__(TrajectoryJointPositionAction)
    action._env = SimpleNamespace(curriculum_stage=torch.tensor([0, 1]))
    action._waypoint_count = 6
    action._num_joints = 7
    action._grasp_gate_stage = 1
    action._approach_phase = 0.12
    action._grasp_phase = 0.24
    action._lift_phase = 0.40
    action._align_phase = 0.62
    action._reference_waypoints = torch.zeros((2, 6, 7))
    action._reference_phase = torch.zeros(2)
    action._minimum_phase = torch.zeros(2)
    action._grasp_dwell_count = torch.zeros(2, dtype=torch.long)
    action._approach_dwell_count = torch.zeros(2, dtype=torch.long)
    action._approach_unlocked = torch.zeros(2, dtype=torch.bool)
    action._grasp_unlocked = torch.zeros(2, dtype=torch.bool)
    action._lift_unlocked = torch.zeros(2, dtype=torch.bool)
    action._align_unlocked = torch.zeros(2, dtype=torch.bool)
    action._processed_actions = torch.zeros((2, 7))
    action._filtered_residual = torch.zeros((2, 7))
    phase = torch.tensor([0.40, 0.40])

    action.set_reference(
        torch.zeros((2, 6, 7)),
        phase,
        torch.zeros((2, 7)),
    )

    assert action._grasp_unlocked.tolist() == [True, False]
    assert action._lift_unlocked.tolist() == [True, False]
    assert action._align_unlocked.tolist() == [False, False]


def test_bilateral_gripper_preload_rejects_unilateral_empty_transient_and_open_states():
    target = torch.tensor(
        [
            [0.024, 0.024],
            [0.024, 0.024],
            [0.024, 0.024],
            [0.024, 0.024],
            [0.040, 0.040],
            [0.024, 0.024],
        ]
    )
    position = torch.tensor(
        [
            [0.026, 0.026],
            [0.027, 0.0242],
            [0.024, 0.024],
            [0.027, 0.027],
            [0.042, 0.042],
            [float("nan"), 0.026],
        ]
    )
    velocity = torch.zeros_like(position)
    velocity[3] = 0.04

    deflection, bilateral = _bilateral_gripper_preload(
        position,
        velocity,
        target,
        min_deflection=0.001,
        max_velocity=0.005,
        max_command=0.025,
    )

    assert bilateral.tolist() == [True, False, False, False, False, False]
    torch.testing.assert_close(deflection[0], torch.tensor([0.002, 0.002]))
    torch.testing.assert_close(deflection[-1], torch.tensor([0.0, 0.002]))


def test_curriculum_joint_action_smooths_targets_and_reset_clears_history():
    action = CurriculumJointPositionAction.__new__(CurriculumJointPositionAction)
    action.cfg = SimpleNamespace(clip=None)
    action._raw_actions = torch.zeros((2, 2))
    action._processed_actions = torch.zeros((2, 2))
    action._previous_target = torch.zeros((2, 2))
    action._offset = torch.zeros((2, 2))
    action._scale = 0.5
    action._alpha = 0.2
    action._project_reference_through_stage = -1
    action._reference_action_magnitude = 1.0
    action._reference_action_index = 0
    action._reference_target = None

    action.process_actions(torch.ones((2, 2)))
    torch.testing.assert_close(action.processed_actions, torch.full((2, 2), 0.1))
    action.process_actions(torch.ones((2, 2)))
    torch.testing.assert_close(action.processed_actions, torch.full((2, 2), 0.18))

    action.set_action_offset(torch.tensor([[0.3, 0.4]]), env_ids=torch.tensor([1]))
    action.reset(torch.tensor([1]))
    torch.testing.assert_close(action.processed_actions[1], torch.tensor([0.3, 0.4]))
    torch.testing.assert_close(action._previous_target[1], torch.tensor([0.3, 0.4]))


def test_curriculum_joint_action_projects_only_early_stage_onto_validated_segment():
    action = CurriculumJointPositionAction.__new__(CurriculumJointPositionAction)
    action.cfg = SimpleNamespace(clip=None)
    action._env = SimpleNamespace(curriculum_stage=torch.tensor([0, 1, 0]))
    action._raw_actions = torch.zeros((3, 2))
    action._processed_actions = torch.zeros((3, 2))
    action._previous_target = torch.zeros((3, 2))
    action._offset = torch.zeros((3, 2))
    action._scale = torch.tensor([[2.0, 1.0]]).repeat(3, 1)
    action._alpha = 1.0
    action._project_reference_through_stage = 0
    action._reference_action_magnitude = 1.0
    action._reference_action_index = 0
    action._reference_target = torch.tensor([[2.0, 2.0]]).repeat(3, 1)

    # The first coordinate is a stage-stable scalar phase; stage one keeps the normal full-rank action.
    raw = torch.tensor([[0.2, 0.9], [0.4, -0.2], [-1.0, 2.0]])
    action.process_actions(raw)

    torch.testing.assert_close(action.raw_actions, raw)
    torch.testing.assert_close(action.processed_actions[0], torch.tensor([0.4, 0.4]))
    torch.testing.assert_close(action.processed_actions[1], torch.tensor([0.8, -0.2]))
    torch.testing.assert_close(action.processed_actions[2], torch.zeros(2))

    # A later low command cannot reverse an early-stage pour, while the unrestricted stage still
    # follows its ordinary reset-relative joint command.
    action.process_actions(torch.zeros_like(raw))
    torch.testing.assert_close(action.processed_actions[0], torch.tensor([0.4, 0.4]))
    torch.testing.assert_close(action.processed_actions[1], torch.zeros(2))


def test_curriculum_gripper_zero_action_tracks_nominal_preload_after_reset():
    action = CurriculumGripperPositionAction.__new__(CurriculumGripperPositionAction)
    action._env = SimpleNamespace(device="cpu")
    action._raw_actions = torch.zeros((4, 1))
    action._processed_actions = torch.zeros((4, 2))
    action._action_offset = torch.full((4, 1), 0.024)
    action._scale = 0.001
    action._alpha = 1.0
    action._close_position = 0.024
    action._neutral_position = 0.025
    action._open_position = 0.04
    action._force_open_stage = -1
    action._capture_unlocked = torch.ones(4, dtype=torch.bool)
    action._capture_dwell_count = torch.zeros(4, dtype=torch.long)
    action._num_joints = 2

    action.set_reset_position(torch.tensor([[0.04], [0.04]]), env_ids=torch.tensor([1, 3]))
    action.reset(torch.arange(4))
    action.process_actions(torch.zeros((4, 1)))

    torch.testing.assert_close(action.processed_actions, torch.full((4, 2), 0.024))
    action.process_actions(torch.tensor([[0.0], [-0.25], [0.0], [1.0]]))
    torch.testing.assert_close(action.processed_actions[1], torch.full((2,), 0.024))
    torch.testing.assert_close(action.processed_actions[3], torch.full((2,), 0.025))


def test_curriculum_gripper_action_filters_bounded_position_residual():
    action = CurriculumGripperPositionAction.__new__(CurriculumGripperPositionAction)
    action._env = SimpleNamespace()
    action._raw_actions = torch.zeros((2, 1))
    action._processed_actions = torch.full((2, 2), 0.024)
    action._action_offset = torch.full((2, 1), 0.024)
    action._scale = 0.001
    action._alpha = 0.2
    action._close_position = 0.024
    action._neutral_position = 0.025
    action._open_position = 0.04
    action._force_open_stage = -1
    action._capture_unlocked = torch.ones(2, dtype=torch.bool)
    action._capture_dwell_count = torch.zeros(2, dtype=torch.long)
    action._num_joints = 2

    action.process_actions(torch.ones((2, 1)))
    torch.testing.assert_close(action.processed_actions, torch.full((2, 2), 0.0242))
    action.process_actions(torch.ones((2, 1)))
    torch.testing.assert_close(action.processed_actions, torch.full((2, 2), 0.02436))

    action.reset(torch.tensor([1]))
    torch.testing.assert_close(action.processed_actions[1], torch.full((2,), 0.02436))


def test_curriculum_gripper_caps_policy_opening_at_safe_preload_in_every_stage():
    action = CurriculumGripperPositionAction.__new__(CurriculumGripperPositionAction)
    action._env = SimpleNamespace(step_dt=1.0 / 60.0, curriculum_stage=torch.tensor([0, 2, 3, 4]))
    action._raw_actions = torch.zeros((4, 1))
    action._processed_actions = torch.full((4, 2), 0.025)
    action._action_offset = torch.full((4, 1), 0.024)
    action._scale = 0.001
    action._alpha = 0.2
    action._close_position = 0.024
    action._neutral_position = 0.025
    action._open_position = 0.04
    action._force_open_stage = -1
    action._capture_unlocked = torch.ones(4, dtype=torch.bool)
    action._capture_dwell_count = torch.zeros(4, dtype=torch.long)
    action._num_joints = 2

    action.process_actions(torch.ones((4, 1)))

    torch.testing.assert_close(action.raw_actions, torch.ones((4, 1)))
    torch.testing.assert_close(
        action.processed_actions,
        torch.tensor(
            [
                [0.025, 0.025],
                [0.025, 0.025],
                [0.025, 0.025],
                [0.025, 0.025],
            ]
        ),
    )
