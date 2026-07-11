# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Franka Pour static reset mixture."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import CurriculumTermCfg

from isaaclab_tasks.contrib.franka_pour import pour_env as pour_env_module
from isaaclab_tasks.contrib.franka_pour.mdp.reset_mixture import (
    RESET_MIXTURE_REGION_NAMES,
    RESET_MIXTURE_STAGE_NAMES,
    PourResetMixture,
)
from isaaclab_tasks.contrib.franka_pour.pour_env import FrankaPourEnv
from isaaclab_tasks.contrib.franka_pour.reset_utils import (
    balanced_reset_triples,
    hierarchical_reset_sampling_weights,
)

_STAGE_NAMES = (
    "drain",
    "deep_tilt",
    "tilt",
    "pour",
    "near_carry",
    "mid_carry",
    "carry",
    "grasp",
    "approach_1",
    "approach_2",
    "approach_3",
    "approach_4",
    "approach_5",
    "approach_6",
    "full",
    "randomized",
)


class FakeTerminationManager:
    def __init__(self, num_envs: int):
        self.terminated = torch.zeros(num_envs, dtype=torch.bool)
        self.time_outs = torch.zeros(num_envs, dtype=torch.bool)
        self._terms = {
            "failure": torch.zeros(num_envs, dtype=torch.bool),
            "extreme_rigid_state": torch.zeros(num_envs, dtype=torch.bool),
            "particle_out_of_bounds": torch.zeros(num_envs, dtype=torch.bool),
            "spill": torch.zeros(num_envs, dtype=torch.bool),
            "lost_grasp": torch.zeros(num_envs, dtype=torch.bool),
            "success": torch.zeros(num_envs, dtype=torch.bool),
        }

    def get_term(self, name: str) -> torch.Tensor:
        return self._terms[name]


class FakeResetMixtureEnv:
    def __init__(
        self,
        probabilities: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
        target_fraction: float = 0.30,
        statistics_window_size: int = 4096,
    ):
        self.num_envs = 4
        self.device = "cpu"
        self.cfg = SimpleNamespace(
            reset_mixture_probabilities=probabilities,
            reset_mixture_statistics_window_size=statistics_window_size,
            success_dwell_time_s=0.15,
            lost_grasp_dwell_time_s=0.05,
            max_spill_fraction=0.10,
            pour_target_frac=target_fraction,
            curriculum_stage_names=_STAGE_NAMES,
            curriculum_target_frac=(0.30,) * len(_STAGE_NAMES),
            curriculum_randomization_extent_levels=(0.0, 0.5, 1.0),
        )
        self.reset_region_id = torch.full((self.num_envs,), -1, dtype=torch.long)
        self.curriculum_stage = torch.zeros(self.num_envs, dtype=torch.long)
        self.curriculum_randomization_level = torch.zeros(self.num_envs, dtype=torch.long)
        self.pour_target_frac = torch.zeros(self.num_envs)
        self.step_dt = 1.0 / 60.0
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.episode_succeeded = torch.zeros(self.num_envs, dtype=torch.bool)
        self._success_dwell_count = torch.zeros(self.num_envs, dtype=torch.long)
        self._lost_grasp_dwell_count = torch.zeros(self.num_envs, dtype=torch.long)
        self.ep_max_target_frac = torch.zeros(self.num_envs)
        self.termination_manager = FakeTerminationManager(self.num_envs)
        self.spill_fractions = torch.zeros(self.num_envs)
        self.spilled_fraction_call_count = 0

    def set_curriculum_stage(self, env_ids, stage: int) -> None:
        self.curriculum_stage[env_ids] = stage
        self.pour_target_frac[env_ids] = self.cfg.curriculum_target_frac[stage]

    def set_curriculum_randomization_level(self, env_ids, level: int) -> None:
        self.curriculum_randomization_level[env_ids] = level

    def spilled_fraction(self) -> torch.Tensor:
        self.spilled_fraction_call_count += 1
        return self.spill_fractions


def _term(env: FakeResetMixtureEnv) -> PourResetMixture:
    return PourResetMixture(CurriculumTermCfg(func=PourResetMixture), env)


def test_reset_mixture_samples_one_region_per_selected_environment(monkeypatch):
    env = FakeResetMixtureEnv()
    term = _term(env)
    selected_ids = torch.tensor([3, 0, 2, 1])
    monkeypatch.setattr(torch, "multinomial", lambda *_args, **_kwargs: torch.tensor([0, 1, 2, 3]))

    metrics = term(env, selected_ids)

    assert env.reset_region_id[selected_ids].tolist() == [0, 1, 2, 3]
    assert env.curriculum_stage[selected_ids].tolist() == [
        _STAGE_NAMES.index(stage_name) for stage_name in RESET_MIXTURE_STAGE_NAMES
    ]
    assert env.curriculum_randomization_level.tolist() == [2, 2, 2, 2]
    assert env.pour_target_frac.tolist() == pytest.approx([0.30] * env.num_envs)
    assert metrics["reaching_sample_fraction"] == pytest.approx(0.25)
    assert metrics["near_goal_sampled_resets"] == 1.0
    assert RESET_MIXTURE_STAGE_NAMES[-1] == "tilt"


def test_reset_mixture_ignores_initial_reset_outcomes(monkeypatch):
    env = FakeResetMixtureEnv()
    term = _term(env)
    env.episode_succeeded[:] = True
    env.termination_manager.terminated[:] = True
    env.spilled_fraction = lambda: pytest.fail("initial reset queried particle outcomes")
    monkeypatch.setattr(torch, "multinomial", lambda *_args, **_kwargs: torch.tensor([0, 1, 2, 3]))

    metrics = term(env, slice(None))

    assert all(metrics[f"{name}_window_completed_episodes"] == 0.0 for name in RESET_MIXTURE_REGION_NAMES)
    assert all(metrics[f"{name}_total_completed_episodes"] == 0.0 for name in RESET_MIXTURE_REGION_NAMES)
    assert all(metrics[f"{name}_success_rate"] == 0.0 for name in RESET_MIXTURE_REGION_NAMES)
    assert all(metrics[f"{name}_ever_success_rate"] == 0.0 for name in RESET_MIXTURE_REGION_NAMES)
    assert all(metrics[f"{name}_raw_peak_target_rate"] == 0.0 for name in RESET_MIXTURE_REGION_NAMES)


def test_reset_mixture_attributes_outcomes_before_sampling_next_regions(monkeypatch):
    env = FakeResetMixtureEnv()
    term = _term(env)
    draws = iter((torch.tensor([0, 1, 2, 3]), torch.tensor([3, 2, 1, 0])))
    monkeypatch.setattr(torch, "multinomial", lambda *_args, **_kwargs: next(draws))
    term(env, torch.arange(env.num_envs))

    env.episode_length_buf[:] = torch.tensor([10, 20, 30, 40])
    # Region one succeeded transiently, then lost the final stable state before termination.
    env.episode_succeeded[:] = torch.tensor([True, True, False, False])
    env._success_dwell_count[0] = 9
    env.ep_max_target_frac[:] = torch.tensor([0.30, 0.20, 0.10, 0.05])
    env.spill_fractions[:] = torch.tensor([0.01, 0.20, 0.03, 0.04])
    env.termination_manager.terminated[:] = torch.tensor([True, True, True, False])
    env.termination_manager.time_outs[3] = True
    env._lost_grasp_dwell_count[2] = 3

    metrics = term(env, torch.arange(env.num_envs))

    assert metrics["reaching_success_rate"] == 1.0
    assert metrics["reaching_ever_success_rate"] == 1.0
    assert metrics["reaching_raw_peak_target_rate"] == 1.0
    assert metrics["near_object_failure_rate"] == 1.0
    assert metrics["near_object_success_rate"] == 0.0
    assert metrics["near_object_ever_success_rate"] == 1.0
    assert metrics["near_object_raw_peak_target_rate"] == 0.0
    assert metrics["near_object_spill_rate"] == 1.0
    assert metrics["near_object_mean_spill_fraction"] == pytest.approx(0.20)
    assert metrics["grasped_lost_grasp_rate"] == 1.0
    assert metrics["near_goal_timeout_rate"] == 1.0
    assert metrics["near_goal_mean_episode_length_steps"] == 40.0
    assert metrics["reaching_estimated_transition_fraction"] == pytest.approx(0.10)
    assert metrics["near_object_estimated_transition_fraction"] == pytest.approx(0.20)
    assert metrics["grasped_estimated_transition_fraction"] == pytest.approx(0.30)
    assert metrics["near_goal_estimated_transition_fraction"] == pytest.approx(0.40)
    assert env.spilled_fraction_call_count == 1
    assert env.reset_region_id.tolist() == [3, 2, 1, 0]
    assert env.curriculum_stage.tolist() == [
        _STAGE_NAMES.index("tilt"),
        _STAGE_NAMES.index("carry"),
        _STAGE_NAMES.index("grasp"),
        _STAGE_NAMES.index("randomized"),
    ]


def test_reset_mixture_attributes_abnormal_causes_and_episode_age_by_region(monkeypatch):
    env = FakeResetMixtureEnv()
    term = _term(env)
    monkeypatch.setattr(torch, "multinomial", lambda *_args, **_kwargs: torch.arange(env.num_envs))
    term(env, torch.arange(env.num_envs))

    env.episode_length_buf[:] = torch.tensor([4, 8, 9, 12])
    env.termination_manager.terminated[:] = True
    env.termination_manager.get_term("failure")[0] = True
    env.termination_manager.get_term("extreme_rigid_state")[1] = True
    env.termination_manager.get_term("particle_out_of_bounds")[2] = True
    env.termination_manager.get_term("success")[3] = True
    env._success_dwell_count[3] = 9

    metrics = term(env, torch.arange(env.num_envs))

    assert metrics["reaching_nonfinite_failure_rate"] == 1.0
    assert metrics["reaching_abnormal_completion_rate"] == 1.0
    assert metrics["reaching_early_abnormal_completion_rate"] == 1.0
    assert metrics["reaching_mean_abnormal_episode_length_steps"] == 4.0
    assert metrics["near_object_extreme_rigid_state_rate"] == 1.0
    assert metrics["near_object_early_abnormal_completion_rate"] == 1.0
    assert metrics["near_object_mean_abnormal_episode_length_steps"] == 8.0
    assert metrics["grasped_particle_out_of_bounds_rate"] == 1.0
    assert metrics["grasped_early_abnormal_completion_rate"] == 0.0
    assert metrics["grasped_mean_abnormal_episode_length_steps"] == 9.0
    assert metrics["near_goal_success_termination_rate"] == 1.0
    assert metrics["near_goal_abnormal_completion_rate"] == 0.0
    assert metrics["near_goal_mean_abnormal_episode_length_steps"] == 0.0


def test_reset_mixture_statistics_window_evicts_oldest_outcome(monkeypatch):
    env = FakeResetMixtureEnv(statistics_window_size=2)
    term = _term(env)
    draws = iter(
        (
            torch.zeros(2, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
        )
    )
    monkeypatch.setattr(torch, "multinomial", lambda *_args, **_kwargs: next(draws))

    term(env, torch.tensor([0, 1]))
    env.episode_length_buf[:2] = 10
    env.episode_succeeded[:2] = torch.tensor([True, False])
    env._success_dwell_count[:2] = torch.tensor([9, 0])
    env.termination_manager.terminated[:2] = True
    metrics = term(env, torch.tensor([0]))
    assert metrics["reaching_success_rate"] == 1.0
    assert metrics["reaching_ever_success_rate"] == 1.0

    # Environment one still carries its original reaching label.
    env.episode_succeeded[1] = False
    env._success_dwell_count[1] = 0
    metrics = term(env, torch.tensor([1]))
    assert metrics["reaching_window_completed_episodes"] == 2.0
    assert metrics["reaching_total_completed_episodes"] == 2.0
    assert metrics["reaching_success_rate"] == pytest.approx(0.5)
    assert metrics["reaching_ever_success_rate"] == pytest.approx(0.5)

    # A third failure wraps the ring and evicts the oldest success.
    env.episode_succeeded[0] = False
    env._success_dwell_count[0] = 0
    metrics = term(env, torch.tensor([0]))
    assert metrics["reaching_window_completed_episodes"] == 2.0
    assert metrics["reaching_total_completed_episodes"] == 3.0
    assert metrics["reaching_success_rate"] == 0.0
    assert metrics["reaching_ever_success_rate"] == 0.0


def test_reset_mixture_counts_current_stable_state_as_success_at_fixed_deadline(monkeypatch):
    env = FakeResetMixtureEnv()
    term = _term(env)
    monkeypatch.setattr(torch, "multinomial", lambda *_args, **_kwargs: torch.arange(env.num_envs))
    term(env, torch.arange(env.num_envs))

    env.episode_length_buf[:] = 300
    env.episode_succeeded[:] = torch.tensor([True, True, False, False])
    env._success_dwell_count[:] = torch.tensor([9, 0, 0, 0])
    env.termination_manager.time_outs[:] = True

    metrics = term(env, torch.arange(env.num_envs))

    assert metrics["reaching_success_rate"] == 1.0
    assert metrics["reaching_failure_rate"] == 0.0
    assert metrics["reaching_timeout_rate"] == 1.0
    assert metrics["near_object_success_rate"] == 0.0
    assert metrics["near_object_ever_success_rate"] == 1.0
    assert metrics["near_object_failure_rate"] == 1.0


def test_estimated_transition_fraction_preserves_nonuniform_reset_probabilities(monkeypatch):
    env = FakeResetMixtureEnv(probabilities=(0.7, 0.1, 0.1, 0.1))
    term = _term(env)
    monkeypatch.setattr(torch, "multinomial", lambda *_args, **_kwargs: torch.arange(4))
    term(env, torch.arange(4))
    env.episode_length_buf[:] = 10
    env.termination_manager.terminated[:] = True

    metrics = term(env, torch.arange(4))

    for probability, name in zip((0.7, 0.1, 0.1, 0.1), RESET_MIXTURE_REGION_NAMES, strict=True):
        assert metrics[f"{name}_estimated_transition_fraction"] == pytest.approx(probability)


def test_estimated_transition_fraction_normalizes_sparse_window_warmup(monkeypatch):
    env = FakeResetMixtureEnv(probabilities=(0.1, 0.3, 0.3, 0.3))
    term = _term(env)
    monkeypatch.setattr(torch, "multinomial", lambda *_args, **_kwargs: torch.zeros(4, dtype=torch.long))
    term(env, torch.arange(4))
    env.episode_length_buf[:] = 1
    env.termination_manager.terminated[:] = True

    metrics = term(env, torch.arange(4))

    assert metrics["reaching_estimated_transition_fraction"] == pytest.approx(1.0)
    assert all(metrics[f"{name}_estimated_transition_fraction"] == 0.0 for name in RESET_MIXTURE_REGION_NAMES[1:])


def test_reset_mixture_bank_samples_reaching_atomically_and_keeps_nonreaching_correlated(monkeypatch):
    env = SimpleNamespace(
        device="cpu",
        curriculum_randomization_level=torch.full((4,), 2, dtype=torch.long),
        _randomized_extent_index_pools=(torch.arange(3),),
        _randomized_extent_index_weights=(torch.ones(3),),
        _last_source_bank_index=torch.full((4,), -1, dtype=torch.long),
        _last_arm_bank_index=torch.full((4,), -1, dtype=torch.long),
        _last_target_bank_index=torch.full((4,), -1, dtype=torch.long),
        _reset_mixture_reaching_triples_t=torch.tensor(((0, 2, 0), (2, 0, 1))),
        _reset_mixture_reaching_weights_t=torch.ones(2),
        _reset_mixture_near_object_source_rows_t=torch.tensor((0, 2, 2)),
        _reset_mixture_near_object_weights_t=torch.ones(3),
        _reset_mixture_near_object_preloaded_t=torch.tensor((False, False, True)),
    )
    env._randomized_source_pos_bank_t = torch.arange(9, dtype=torch.float32).reshape(3, 3) + 10.0
    env._randomized_source_quat_bank_t = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 20.0
    env._randomized_target_pos_bank_t = torch.arange(9, dtype=torch.float32).reshape(3, 3) + 30.0
    env._reset_mixture_reaching_target_pos_t = env._randomized_target_pos_bank_t.clone()
    env._randomized_grasp_arm_q_bank_t = torch.arange(21, dtype=torch.float32).reshape(3, 7) + 40.0
    env._reset_mixture_near_object_arm_q_t = torch.stack(
        (
            env._randomized_grasp_arm_q_bank_t[0] + 1000.0,
            env._randomized_grasp_arm_q_bank_t[2] + 2000.0,
            env._randomized_grasp_arm_q_bank_t[2] + 3000.0,
        )
    )
    env._randomized_carry_arm_q_bank_t = torch.arange(21, dtype=torch.float32).reshape(3, 7) + 70.0
    env._randomized_pour_arm_q_bank_t = torch.arange(21, dtype=torch.float32).reshape(3, 7) + 85.0
    env._randomized_tilt_arm_q_bank_t = torch.arange(21, dtype=torch.float32).reshape(3, 7) + 100.0
    env._randomized_arm_q_bank_t = torch.arange(21, dtype=torch.float32).reshape(3, 7) + 130.0
    sample_calls = 0

    def sample_correlated(*_args, **_kwargs):
        nonlocal sample_calls
        sample_calls += 1
        return torch.tensor([0, 1])

    monkeypatch.setattr(
        pour_env_module,
        "sample_index_pools",
        sample_correlated,
    )

    def sample_reset_candidate(weights, *_args, **_kwargs):
        if weights.data_ptr() == env._reset_mixture_reaching_weights_t.data_ptr():
            return torch.tensor([1])
        if weights.data_ptr() == env._reset_mixture_near_object_weights_t.data_ptr():
            return torch.tensor([2])
        raise AssertionError("Unexpected reset-candidate distribution.")

    monkeypatch.setattr(torch, "multinomial", sample_reset_candidate)
    # Exercise both segment endpoints, then interior collision-screened eighths.
    path_draws = iter(
        (
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([7]),
            torch.tensor([8]),
            torch.tensor([3]),
            torch.tensor([4]),
        )
    )
    path_highs = []

    def sample_path(high, *_args, **_kwargs):
        path_highs.append(high)
        return next(path_draws)

    monkeypatch.setattr(torch, "randint", sample_path)
    env_ids = torch.tensor([2, 0, 3, 1])
    reset_regions = torch.tensor([0, 1, 2, 3])
    arm_q = torch.full((4, 7), -1.0)
    cup_pos_e = torch.full((4, 3), -1.0)
    source_quat = torch.full((4, 4), -1.0)
    target_pos_e = torch.full((4, 3), -1.0)

    preloaded_table_grasp = FrankaPourEnv._apply_reset_mixture_bank(
        env,
        env_ids,
        reset_regions,
        arm_q,
        cup_pos_e,
        source_quat,
        target_pos_e,
    )

    assert torch.equal(arm_q[0], env._randomized_arm_q_bank_t[0])
    assert torch.equal(arm_q[1], env._reset_mixture_near_object_arm_q_t[2])
    assert torch.equal(arm_q[2], env._randomized_grasp_arm_q_bank_t[0])
    assert torch.equal(arm_q[3], env._randomized_pour_arm_q_bank_t[1])
    assert preloaded_table_grasp.tolist() == [False, True, False, False]

    preloaded_table_grasp = FrankaPourEnv._apply_reset_mixture_bank(
        env,
        env_ids,
        reset_regions,
        arm_q,
        cup_pos_e,
        source_quat,
        target_pos_e,
    )
    torch.testing.assert_close(
        arm_q[2],
        torch.lerp(env._randomized_grasp_arm_q_bank_t[0], env._randomized_carry_arm_q_bank_t[0], 7.0 / 8.0),
    )
    assert torch.equal(arm_q[3], env._randomized_tilt_arm_q_bank_t[1])
    assert preloaded_table_grasp.tolist() == [False, True, False, False]

    preloaded_table_grasp = FrankaPourEnv._apply_reset_mixture_bank(
        env,
        env_ids,
        reset_regions,
        arm_q,
        cup_pos_e,
        source_quat,
        target_pos_e,
    )
    torch.testing.assert_close(
        arm_q[2],
        torch.lerp(env._randomized_grasp_arm_q_bank_t[0], env._randomized_carry_arm_q_bank_t[0], 3.0 / 8.0),
    )
    torch.testing.assert_close(
        arm_q[3],
        torch.lerp(env._randomized_pour_arm_q_bank_t[1], env._randomized_tilt_arm_q_bank_t[1], 1.0 / 2.0),
    )
    assert torch.equal(cup_pos_e[0], env._randomized_source_pos_bank_t[2])
    assert torch.equal(target_pos_e[0], env._randomized_target_pos_bank_t[1])
    assert torch.equal(cup_pos_e[1:], env._randomized_source_pos_bank_t[torch.tensor([2, 0, 1])])
    assert torch.equal(source_quat[1:], env._randomized_source_quat_bank_t[torch.tensor([2, 0, 1])])
    assert torch.equal(target_pos_e[1:], env._randomized_target_pos_bank_t[torch.tensor([2, 0, 1])])
    assert env._last_source_bank_index.tolist() == [2, 1, 2, 0]
    assert env._last_arm_bank_index.tolist() == [2, 1, 0, 0]
    assert env._last_target_bank_index.tolist() == [2, 1, 1, 0]
    assert preloaded_table_grasp.tolist() == [False, True, False, False]
    assert path_highs == [16, 9, 16, 9, 16, 9]
    assert sample_calls == 3


def test_balanced_reset_triples_preserve_every_marginal():
    pool = torch.tensor((7, 11, 19, 23))

    triples = balanced_reset_triples(pool, attempt_count=3)

    expected = torch.tensor(
        (
            (7, 7, 7),
            (7, 11, 23),
            (7, 19, 19),
            (11, 11, 11),
            (11, 19, 7),
            (11, 23, 23),
            (19, 19, 19),
            (19, 23, 11),
            (19, 7, 7),
            (23, 23, 23),
            (23, 7, 19),
            (23, 11, 11),
        )
    )
    assert torch.equal(triples, expected)
    assert torch.equal(triples, balanced_reset_triples(pool, attempt_count=3))
    for marginal in triples.unbind(dim=1):
        assert torch.equal(torch.bincount(marginal, minlength=24)[pool], torch.full((4,), 3))


def test_hierarchical_reset_sampling_weights_balance_cells_rows_and_candidates():
    triples = torch.tensor(
        (
            (10, 0, 0),
            (11, 0, 0),
            (11, 1, 0),
            (11, 2, 0),
            (20, 0, 0),
            (20, 1, 0),
        )
    )
    cells = torch.tensor((0, 0, 0, 0, 1, 1))

    weights = hierarchical_reset_sampling_weights(triples[:, 0], cells)

    torch.testing.assert_close(
        weights,
        torch.tensor((1 / 4, 1 / 12, 1 / 12, 1 / 12, 1 / 4, 1 / 4)),
    )
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
    torch.testing.assert_close(weights[cells == 0].sum(), torch.tensor(0.5))
    torch.testing.assert_close(weights[cells == 1].sum(), torch.tensor(0.5))
