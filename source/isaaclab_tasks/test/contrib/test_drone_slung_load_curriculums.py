# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused unit coverage for the slung-load precision/speed curriculum."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import CurriculumManager, CurriculumTermCfg, ManagerTermBase

from isaaclab_tasks.contrib.drone_slung_load.mdp.curriculums import (
    DirectCTBRCurriculumV14,
    DirectCTBRRouteCurriculum,
    PrecisionSpeedCurriculum,
    PrecisionSpeedCurriculumV13,
)

pytestmark = pytest.mark.unit


class _CommandManager:
    def __init__(self):
        self.term = SimpleNamespace(
            cfg=SimpleNamespace(
                acceptance_radius=0.50,
                target_cruise_speed=1.25,
                route_family="random_walk",
                figure_eight_probability=0.50,
                vertical_amplitude_range=(0.0, 0.15),
                random_waypoint_ranges=SimpleNamespace(
                    pos_x=(-12.5, 12.5),
                    pos_y=(-12.5, 12.5),
                    pos_z=(0.0, 0.0),
                ),
                minimum_waypoint_separation=0.49,
                maximum_waypoint_separation=0.51,
                nominal_heading_change=0.0,
                maximum_heading_change=math.radians(5.0),
                maximum_vertical_step=0.0,
                random_sampling_attempts=8,
                route_sampling_attempts=4,
                random_heading_change_interval=1,
            ),
            metrics={
                "cross_track_error_rms": torch.zeros(4),
                "route_traversal_fraction": torch.zeros(4),
            },
        )

    def get_term(self, name: str):
        if name != "route":
            raise KeyError(name)
        return self.term


class _RewardManager:
    def __init__(self):
        self.configs = {
            "path_progress": SimpleNamespace(params={"maximum_rate": 1.25, "maximum_lateral_acceleration": 3.0}),
            "path_precision": SimpleNamespace(
                params={"cross_track_scale": 0.50, "transverse_velocity_scale": 1.00}, weight=-2.0
            ),
        }
        self.set_calls: list[str] = []

    def get_term_cfg(self, name: str):
        if name not in self.configs:
            raise ValueError(name)
        return self.configs[name]

    def set_term_cfg(self, name: str, cfg) -> None:
        self.configs[name] = cfg
        self.set_calls.append(name)


class _TerminationManager:
    def __init__(self):
        self.configs = {"path_corridor": SimpleNamespace(params={"maximum_distance": 1.50}, time_out=False)}
        self.set_calls: list[str] = []

    def get_term_cfg(self, name: str):
        if name not in self.configs:
            raise ValueError(name)
        return self.configs[name]

    def set_term_cfg(self, name: str, cfg) -> None:
        self.configs[name] = cfg
        self.set_calls.append(name)


class _EventManager:
    def __init__(self):
        self.configs = {
            "reset_base": SimpleNamespace(
                params={
                    "radius_range": (4.3, 4.8),
                    "roll_range": (-0.05, 0.05),
                    "pitch_range": (-0.05, 0.05),
                }
            ),
            "reset_slung_load": SimpleNamespace(params={"cable_length": 0.50, "max_initial_swing": 0.10}),
        }
        self.set_calls: list[str] = []

    def get_term_cfg(self, name: str):
        if name not in self.configs:
            raise ValueError(name)
        return self.configs[name]

    def set_term_cfg(self, name: str, cfg) -> None:
        self.configs[name] = cfg
        self.set_calls.append(name)


def _make_env(step: int = 0):
    return SimpleNamespace(
        common_step_counter=step,
        command_manager=_CommandManager(),
        reward_manager=_RewardManager(),
        termination_manager=_TerminationManager(),
        event_manager=_EventManager(),
        sim=SimpleNamespace(is_playing=lambda: True),
        num_envs=4,
        device="cpu",
        step_curriculum_calls=[],
    )


def _record_step_curriculum_call(env, env_ids):
    env.step_curriculum_calls.append((env.common_step_counter, env_ids))
    return {"global_step": float(env.common_step_counter)}


def _make_term(env, **params) -> PrecisionSpeedCurriculum:
    cfg = CurriculumTermCfg(func=PrecisionSpeedCurriculum, params=params)
    return PrecisionSpeedCurriculum(cfg, env)


def _make_v13_term(env, **params) -> PrecisionSpeedCurriculumV13:
    cfg = CurriculumTermCfg(func=PrecisionSpeedCurriculumV13, params=params)
    return PrecisionSpeedCurriculumV13(cfg, env)


def _make_v14_term(env, **params) -> DirectCTBRCurriculumV14:
    cfg = CurriculumTermCfg(func=DirectCTBRCurriculumV14, params=params)
    return DirectCTBRCurriculumV14(cfg, env)


def _make_route_term(env, **params) -> DirectCTBRRouteCurriculum:
    cfg = CurriculumTermCfg(func=DirectCTBRRouteCurriculum, params=params)
    return DirectCTBRRouteCurriculum(cfg, env)


def test_precision_speed_curriculum_runs_through_curriculum_manager_and_logs_each_value():
    env = _make_env(step=70_000)
    manager = CurriculumManager(
        {"precision_speed": CurriculumTermCfg(func=PrecisionSpeedCurriculum)},
        env,
    )

    manager.compute()

    assert manager.active_terms == ["precision_speed"]
    assert manager.reset() == pytest.approx(
        {
            "Curriculum/precision_speed/acceptance_radius": 0.325,
            "Curriculum/precision_speed/maximum_rate": 2.375,
            "Curriculum/precision_speed/cross_track_scale": 0.35,
            "Curriculum/precision_speed/transverse_velocity_scale": 0.70,
            "Curriculum/precision_speed/precision_reward_weight": -8.5,
            "Curriculum/precision_speed/corridor_distance": 1.125,
        }
    )


def test_v13_step_curriculum_is_rank_deterministic_and_never_recomputed_by_local_resets():
    envs = [_make_env(step=130_000), _make_env(step=130_000)]
    managers = [
        CurriculumManager(
            {
                "precision_speed": CurriculumTermCfg(func=PrecisionSpeedCurriculumV13, update_mode="step"),
                "step_probe": CurriculumTermCfg(func=_record_step_curriculum_call, update_mode="step"),
            },
            env,
        )
        for env in envs
    ]

    for manager in managers:
        manager.compute_step()

    expected_state = {
        "acceptance_radius": pytest.approx(0.15),
        "target_cruise_speed": pytest.approx(2.375),
        "maximum_rate": pytest.approx(2.375),
        "precision_reward_weight": pytest.approx(-4.0),
        "corridor_distance": pytest.approx(0.75),
    }
    for env in envs:
        assert env.command_manager.term.cfg.acceptance_radius == expected_state["acceptance_radius"]
        assert env.command_manager.term.cfg.target_cruise_speed == expected_state["target_cruise_speed"]
        assert env.reward_manager.configs["path_progress"].params["maximum_rate"] == expected_state["maximum_rate"]
        assert env.reward_manager.configs["path_precision"].weight == expected_state["precision_reward_weight"]
        assert (
            env.termination_manager.configs["path_corridor"].params["maximum_distance"]
            == expected_state["corridor_distance"]
        )
        assert len(env.step_curriculum_calls) == 1
        call_step, call_env_ids = env.step_curriculum_calls[0]
        assert call_step == 130_000
        assert call_env_ids == slice(None)

    reward_calls = [list(env.reward_manager.set_calls) for env in envs]
    termination_calls = [list(env.termination_manager.set_calls) for env in envs]
    # Simulate different asynchronous reset subsets on two DDP ranks.
    managers[0].compute(env_ids=[0])
    managers[0].reset(env_ids=[0])
    managers[1].compute(env_ids=[1, 2, 3])
    managers[1].reset(env_ids=[1, 2, 3])
    # A restore-time refresh and the normal pre-action hook may call at the same counter.
    managers[0].compute_step()
    managers[1].compute_step()

    assert envs[0].step_curriculum_calls == envs[1].step_curriculum_calls
    assert len(envs[0].step_curriculum_calls) == 1
    assert [env.reward_manager.set_calls for env in envs] == reward_calls
    assert [env.termination_manager.set_calls for env in envs] == termination_calls


def test_step_curriculum_applies_at_global_step_zero_and_rejects_unknown_update_mode():
    env = _make_env(step=0)
    manager = CurriculumManager(
        {"step_probe": CurriculumTermCfg(func=_record_step_curriculum_call, update_mode="step")},
        env,
    )

    manager.compute_step()
    manager.compute_step()

    assert len(env.step_curriculum_calls) == 1
    assert env.step_curriculum_calls[0][0] == 0

    with pytest.raises(ValueError, match="invalid update mode 'episode'"):
        CurriculumManager(
            {"invalid": CurriculumTermCfg(func=_record_step_curriculum_call, update_mode="episode")},
            _make_env(),
        )


def test_precision_speed_curriculum_is_a_manager_term_with_exact_linear_schedule():
    env = _make_env()
    term = _make_term(env)

    assert isinstance(term, ManagerTermBase)
    assert term(env, slice(None)) == {
        "acceptance_radius": 0.50,
        "maximum_rate": 1.25,
        "cross_track_scale": 0.50,
        "transverse_velocity_scale": 1.00,
        "precision_reward_weight": -2.0,
        "corridor_distance": 1.50,
    }

    env.common_step_counter = 20_000
    assert term(env, [0]) == pytest.approx(
        {
            "acceptance_radius": 0.50,
            "maximum_rate": 1.25,
            "cross_track_scale": 0.50,
            "transverse_velocity_scale": 1.00,
            "precision_reward_weight": -2.0,
            "corridor_distance": 1.50,
        }
    )

    env.common_step_counter = 70_000
    state = term(env, [0, 2])
    assert state == pytest.approx(
        {
            "acceptance_radius": 0.325,
            "maximum_rate": 2.375,
            "cross_track_scale": 0.35,
            "transverse_velocity_scale": 0.70,
            "precision_reward_weight": -8.5,
            "corridor_distance": 1.125,
        }
    )
    assert env.command_manager.term.cfg.acceptance_radius == pytest.approx(0.325)
    assert env.reward_manager.configs["path_progress"].params["maximum_rate"] == pytest.approx(2.375)
    assert env.reward_manager.configs["path_progress"].params["maximum_lateral_acceleration"] == pytest.approx(3.0)
    assert env.reward_manager.configs["path_precision"].params == pytest.approx(
        {"cross_track_scale": 0.35, "transverse_velocity_scale": 0.70}
    )
    assert env.reward_manager.configs["path_precision"].weight == pytest.approx(-8.5)
    assert env.termination_manager.configs["path_corridor"].params["maximum_distance"] == pytest.approx(1.125)

    env.common_step_counter = 120_000
    assert term(env, []) == pytest.approx(
        {
            "acceptance_radius": 0.15,
            "maximum_rate": 3.50,
            "cross_track_scale": 0.20,
            "transverse_velocity_scale": 0.40,
            "precision_reward_weight": -15.0,
            "corridor_distance": 0.75,
        }
    )


def test_precision_speed_curriculum_resumes_from_global_step_and_skips_unchanged_updates():
    env = _make_env(step=95_000)
    term = _make_term(env)

    state = term(env, [1])
    assert state == pytest.approx(
        {
            "acceptance_radius": 0.2375,
            "maximum_rate": 2.9375,
            "cross_track_scale": 0.275,
            "transverse_velocity_scale": 0.55,
            "precision_reward_weight": -11.75,
            "corridor_distance": 0.9375,
        }
    )
    assert env.common_step_counter == 95_000
    assert env.reward_manager.set_calls == ["path_progress", "path_precision"]
    assert env.termination_manager.set_calls == ["path_corridor"]

    progress_cfg = env.reward_manager.configs["path_progress"]
    precision_cfg = env.reward_manager.configs["path_precision"]
    assert term(env, [0, 1, 2, 3]) == pytest.approx(state)
    assert env.reward_manager.set_calls == ["path_progress", "path_precision"]
    assert env.termination_manager.set_calls == ["path_corridor"]
    assert env.reward_manager.configs["path_progress"] is progress_cfg
    assert env.reward_manager.configs["path_precision"] is precision_cfg

    env.common_step_counter = 150_000
    term(env, [0])
    env.reward_manager.set_calls.clear()
    env.termination_manager.set_calls.clear()
    assert term(env, [1]) == pytest.approx(
        {
            "acceptance_radius": 0.15,
            "maximum_rate": 3.50,
            "cross_track_scale": 0.20,
            "transverse_velocity_scale": 0.40,
            "precision_reward_weight": -15.0,
            "corridor_distance": 0.75,
        }
    )
    assert env.reward_manager.set_calls == []
    assert env.termination_manager.set_calls == []


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"warmup_steps": -1}, "warmup_steps must be a non-negative integer"),
        ({"ramp_steps": 0}, "ramp_steps must be a positive integer"),
        ({"final_acceptance_radius": 0.60}, "final_acceptance_radius must not exceed"),
        ({"final_maximum_rate": 1.00}, "final_maximum_rate must not be less"),
        ({"initial_cross_track_scale": float("nan")}, "initial_cross_track_scale must be positive and finite"),
        (
            {"final_transverse_velocity_scale": 1.10},
            "final_transverse_velocity_scale must not exceed",
        ),
        ({"initial_precision_reward_weight": 0.0}, "initial_precision_reward_weight must be negative"),
        ({"final_precision_reward_weight": -1.0}, "final_precision_reward_weight must not exceed"),
        ({"final_corridor_distance": 2.0}, "final_corridor_distance must not exceed"),
    ],
)
def test_precision_speed_curriculum_rejects_invalid_schedule_parameters(params, message):
    with pytest.raises(ValueError, match=message):
        _make_term(_make_env(), **params)


def test_precision_speed_curriculum_validates_reward_parameter_bindings_before_mutation():
    env = _make_env(step=70_000)
    env.reward_manager.configs["path_precision"].params.pop("cross_track_scale")

    with pytest.raises(ValueError, match="missing parameters: cross_track_scale"):
        _make_term(env)

    assert env.command_manager.term.cfg.acceptance_radius == pytest.approx(0.50)
    assert env.reward_manager.set_calls == []


def test_precision_speed_curriculum_validates_corridor_binding_before_mutation():
    env = _make_env(step=70_000)
    env.termination_manager.configs["path_corridor"].params.clear()

    with pytest.raises(ValueError, match="has no 'maximum_distance' parameter"):
        _make_term(env)

    assert env.command_manager.term.cfg.acceptance_radius == pytest.approx(0.50)
    assert env.reward_manager.set_calls == []
    assert env.termination_manager.set_calls == []

    env = _make_env(step=70_000)
    env.termination_manager.configs["path_corridor"].time_out = True
    with pytest.raises(ValueError, match="must not be a time-out"):
        _make_term(env)


def test_v13_curriculum_separates_geometry_weight_and_speed_at_exact_stage_boundaries():
    env = _make_env()
    term = _make_v13_term(env)

    expected = {
        0: (0.0, 0.50, -2.0, 1.25),
        20_000: (1.0, 0.50, -2.0, 1.25),
        40_000: (1.0, 0.325, -2.0, 1.25),
        60_000: (2.0, 0.15, -2.0, 1.25),
        70_000: (2.0, 0.15, -3.0, 1.25),
        80_000: (3.0, 0.15, -4.0, 1.25),
        100_000: (4.0, 0.15, -4.0, 1.25),
        130_000: (4.0, 0.15, -4.0, 2.375),
        160_000: (5.0, 0.15, -4.0, 3.50),
    }
    for step, (stage, acceptance_radius, precision_weight, target_speed) in expected.items():
        env.common_step_counter = step
        state = term(env, slice(None))
        assert state["stage"] == stage
        assert state["acceptance_radius"] == pytest.approx(acceptance_radius)
        assert state["precision_reward_weight"] == pytest.approx(precision_weight)
        assert state["target_cruise_speed"] == pytest.approx(target_speed)
        assert state["maximum_rate"] == pytest.approx(target_speed)

    assert env.command_manager.term.cfg.target_cruise_speed == pytest.approx(3.50)
    assert env.reward_manager.configs["path_precision"].weight == pytest.approx(-4.0)
    assert env.reward_manager.configs["path_progress"].params["maximum_lateral_acceleration"] == 3.0


def test_v13_optional_performance_gate_uses_traversal_and_starts_a_late_speed_ramp_from_zero():
    env = _make_env(step=130_000)
    term = _make_v13_term(env, performance_gate_enabled=True)

    blocked = term(env, slice(None), performance_gate_enabled=True)
    assert blocked["stage"] == 3.0
    assert blocked["gate_open"] == 0.0
    assert blocked["speed_fraction"] == 0.0
    assert blocked["target_cruise_speed"] == pytest.approx(1.25)

    env.command_manager.term.metrics["cross_track_error_rms"][:] = torch.tensor([0.2, 0.2, 0.4, 0.1])
    env.command_manager.term.metrics["route_traversal_fraction"][:] = torch.tensor([0.3, 0.3, 0.8, 0.1])
    opened = term(env, slice(None), performance_gate_enabled=True)
    assert opened["gate_pass_fraction"] == pytest.approx(0.5)
    assert opened["gate_open"] == 1.0
    assert opened["speed_stage_start_step"] == 130_000.0
    assert opened["speed_fraction"] == 0.0
    assert opened["target_cruise_speed"] == pytest.approx(1.25)

    env.common_step_counter = 160_000
    halfway = term(env, [], performance_gate_enabled=True)
    assert halfway["stage"] == 4.0
    assert halfway["speed_fraction"] == pytest.approx(0.5)
    assert halfway["target_cruise_speed"] == pytest.approx(2.375)


def test_v13_curriculum_validates_command_speed_binding_and_updates_only_on_change():
    env = _make_env(step=130_000)
    del env.command_manager.term.cfg.target_cruise_speed
    with pytest.raises(ValueError, match="no mutable 'target_cruise_speed'"):
        _make_v13_term(env)

    env = _make_env(step=130_000)
    term = _make_v13_term(env)
    first = term(env, slice(None))
    reward_set_calls = list(env.reward_manager.set_calls)
    termination_set_calls = list(env.termination_manager.set_calls)
    assert term(env, slice(None)) == pytest.approx(first)
    assert env.reward_manager.set_calls == reward_set_calls
    assert env.termination_manager.set_calls == termination_set_calls


def test_v14_direct_ctbr_curriculum_has_exact_speed_precision_and_hold_boundaries():
    env = _make_env()
    term = _make_v14_term(env)

    expected = {
        0: (0.0, 1.25, 0.50, -1.0, 1.50),
        10_000: (1.0, 1.25, 0.50, -1.0, 1.50),
        20_000: (1.0, 1.75, 0.50, -1.0, 1.50),
        30_000: (2.0, 2.25, 0.50, -1.0, 1.50),
        55_000: (2.0, 3.375, 0.50, -1.0, 1.50),
        80_000: (3.0, 4.50, 0.50, -1.0, 1.50),
        100_000: (3.0, 4.50, 0.325, -1.0, 1.125),
        120_000: (4.0, 4.50, 0.15, -1.0, 0.75),
        130_000: (4.0, 4.50, 0.15, -2.5, 0.75),
        140_000: (5.0, 4.50, 0.15, -4.0, 0.75),
        200_000: (5.0, 4.50, 0.15, -4.0, 0.75),
    }
    for step, (stage, speed, acceptance_radius, precision_weight, corridor_distance) in expected.items():
        env.common_step_counter = step
        state = term(env, slice(None))
        assert state["stage"] == stage
        assert state["maximum_rate"] == pytest.approx(speed)
        assert state["target_cruise_speed"] == pytest.approx(speed)
        assert state["acceptance_radius"] == pytest.approx(acceptance_radius)
        assert state["precision_reward_weight"] == pytest.approx(precision_weight)
        assert state["corridor_distance"] == pytest.approx(corridor_distance)
        assert env.command_manager.term.cfg.target_cruise_speed == pytest.approx(speed)
        assert env.reward_manager.configs["path_progress"].params["maximum_rate"] == pytest.approx(speed)

    assert env.reward_manager.configs["path_progress"].params["maximum_lateral_acceleration"] == 3.0
    assert env.reward_manager.configs["path_precision"].params == pytest.approx(
        {"cross_track_scale": 0.20, "transverse_velocity_scale": 0.40}
    )


def test_v14_direct_ctbr_curriculum_ramps_reset_domain_over_first_eighty_thousand_steps():
    env = _make_env()
    term = _make_v14_term(env)

    expected = {
        0: (0.0, 0.005, 0.020),
        40_000: (0.5, 0.0275, 0.060),
        80_000: (1.0, 0.050, 0.100),
        140_000: (1.0, 0.050, 0.100),
    }
    for step, (fraction, tilt_limit, max_initial_swing) in expected.items():
        env.common_step_counter = step
        state = term(env, slice(None))
        assert state["reset_fraction"] == pytest.approx(fraction)
        assert state["reset_tilt_limit"] == pytest.approx(tilt_limit)
        assert state["max_initial_swing"] == pytest.approx(max_initial_swing)
        reset_base = env.event_manager.configs["reset_base"]
        assert reset_base.params["roll_range"] == pytest.approx((-tilt_limit, tilt_limit))
        assert reset_base.params["pitch_range"] == pytest.approx((-tilt_limit, tilt_limit))
        assert env.event_manager.configs["reset_slung_load"].params["max_initial_swing"] == pytest.approx(
            max_initial_swing
        )


def test_v14_direct_ctbr_curriculum_copies_event_configs_and_skips_unchanged_updates():
    env = _make_env(step=40_000)
    original_reset_base = env.event_manager.configs["reset_base"]
    original_reset_slung_load = env.event_manager.configs["reset_slung_load"]
    term = _make_v14_term(env)

    state = term(env, slice(None))

    assert original_reset_base.params["roll_range"] == (-0.05, 0.05)
    assert original_reset_base.params["pitch_range"] == (-0.05, 0.05)
    assert original_reset_slung_load.params["max_initial_swing"] == 0.10
    assert env.event_manager.configs["reset_base"] is not original_reset_base
    assert env.event_manager.configs["reset_slung_load"] is not original_reset_slung_load
    first_event_calls = list(env.event_manager.set_calls)
    first_reward_calls = list(env.reward_manager.set_calls)
    first_termination_calls = list(env.termination_manager.set_calls)

    assert term(env, [0, 2]) == pytest.approx(state)
    assert env.event_manager.set_calls == first_event_calls
    assert env.reward_manager.set_calls == first_reward_calls
    assert env.termination_manager.set_calls == first_termination_calls


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"warmup_steps": -1}, "warmup_steps must be a non-negative integer"),
        (
            {"completion_speed_ramp_steps": 0},
            "completion_speed_ramp_steps must be a positive integer",
        ),
        ({"completion_speed": 1.0}, "completion_speed must not be less than initial_speed"),
        ({"final_speed": 2.0}, "final_speed must not be less than completion_speed"),
        ({"final_acceptance_radius": 0.6}, "final_acceptance_radius must not exceed"),
        ({"final_precision_reward_weight": -0.5}, "final_precision_reward_weight must not exceed"),
        ({"final_reset_tilt_limit": 0.001}, "final_reset_tilt_limit must not be less"),
        ({"final_max_initial_swing": 0.01}, "final_max_initial_swing must not be less"),
    ],
)
def test_v14_direct_ctbr_curriculum_rejects_invalid_schedule_parameters(params, message):
    with pytest.raises(ValueError, match=message):
        _make_v14_term(_make_env(), **params)


def test_v14_direct_ctbr_curriculum_validates_all_reset_bindings_before_mutation():
    env = _make_env(step=40_000)
    env.event_manager.configs["reset_base"].params.pop("pitch_range")

    with pytest.raises(ValueError, match="missing parameter 'pitch_range'"):
        _make_v14_term(env)

    assert env.command_manager.term.cfg.acceptance_radius == 0.50
    assert env.reward_manager.set_calls == []
    assert env.termination_manager.set_calls == []
    assert env.event_manager.set_calls == []

    env = _make_env(step=40_000)
    env.event_manager.configs.pop("reset_slung_load")
    with pytest.raises(ValueError, match="Reset event term 'reset_slung_load'.*was not found"):
        _make_v14_term(env)


def test_direct_ctbr_route_curriculum_preserves_v17_then_stages_hard_routes():
    env = _make_env()
    term = _make_route_term(env)
    expected = {
        0: (0.0, "random_walk", 1.00, 2.50, 0.00, 0.000, 1.000),
        19_999: (0.0, "random_walk", 1.00, 2.50, 0.00, 0.000, 1.000),
        20_000: (1.0, "bounded_template_mix", 1.00, 2.50, 0.00, 0.000, 1.000),
        39_999: (1.0, "bounded_template_mix", 1.00, 2.50, 0.00, 0.000, 1.000),
        40_000: (2.0, "bounded_template_mix", 1.00, 2.50, 0.00, 0.000, 1.000),
        55_000: (2.0, "bounded_template_mix", 0.75, 2.50, 0.00, 0.000, 1.000),
        70_000: (3.0, "bounded_template_mix", 0.50, 2.50, 0.00, 0.000, 1.000),
        80_000: (3.0, "bounded_template_mix", 0.50, 2.00, 0.00, 0.000, 1.000),
        90_000: (4.0, "bounded_template_mix", 0.50, 1.50, 0.00, 0.000, 1.000),
        105_000: (4.0, "bounded_template_mix", 0.50, 1.50, 0.25, 0.000, 1.000),
        120_000: (5.0, "bounded_template_mix", 0.50, 1.50, 0.50, 0.000, 1.000),
        130_000: (5.0, "bounded_template_mix", 0.50, 1.50, 0.50, 0.075, 1.000),
        140_000: (6.0, "bounded_template_mix", 0.50, 1.50, 0.50, 0.150, 1.000),
        160_000: (6.0, "bounded_template_mix", 0.50, 1.50, 0.50, 0.150, 2.250),
        180_000: (7.0, "bounded_template_mix", 0.50, 1.50, 0.50, 0.150, 3.500),
        199_999: (7.0, "bounded_template_mix", 0.50, 1.50, 0.50, 0.150, 3.500),
        200_000: (8.0, "bounded_hard_mix", 0.50, 1.50, 1.00, 0.150, 2.250),
        220_000: (9.0, "bounded_hard_mix", 0.50, 1.50, 1.00, 0.150, 2.250),
        240_000: (9.0, "bounded_hard_mix", 0.50, 1.50, 0.75, 0.150, 2.250),
        260_000: (10.0, "bounded_hard_mix", 0.50, 1.50, 0.50, 0.150, 2.250),
        290_000: (10.0, "bounded_hard_mix", 0.50, 1.50, 0.50, 0.150, 2.875),
        320_000: (11.0, "bounded_hard_mix", 0.50, 1.50, 0.50, 0.150, 3.500),
        340_000: (12.0, "bounded_hard_mix", 0.50, 1.50, 0.50, 0.150, 3.500),
    }
    for step, (stage, route_family, acceptance, corridor, figure_eight, vertical, speed) in expected.items():
        env.common_step_counter = step
        state = term(env, slice(None))
        assert state["stage"] == stage
        assert env.command_manager.term.cfg.route_family == route_family
        assert state["acceptance_radius"] == pytest.approx(acceptance)
        assert state["corridor_distance"] == pytest.approx(corridor)
        assert state["figure_eight_probability"] == pytest.approx(figure_eight)
        assert state["maximum_vertical_amplitude"] == pytest.approx(vertical)
        assert state["target_cruise_speed"] == pytest.approx(speed)
        assert state["maximum_rate"] == pytest.approx(speed)

        if step < 200_000:
            assert state["nominal_heading_change"] == pytest.approx(0.0)
            assert state["maximum_heading_change"] == pytest.approx(0.0)
        elif step <= 220_000:
            assert state["nominal_heading_change"] == pytest.approx(math.radians(40.0))
            assert state["maximum_heading_change"] == pytest.approx(math.radians(60.0))
        elif step == 240_000:
            assert state["nominal_heading_change"] == pytest.approx(math.radians(70.0))
            assert state["maximum_heading_change"] == pytest.approx(math.radians(85.0))
        else:
            assert state["nominal_heading_change"] == pytest.approx(math.radians(100.0))
            assert state["maximum_heading_change"] == pytest.approx(math.radians(110.0))

    command_cfg = env.command_manager.term.cfg
    assert command_cfg.acceptance_radius == pytest.approx(0.50)
    assert command_cfg.figure_eight_probability == pytest.approx(0.50)
    assert command_cfg.vertical_amplitude_range == pytest.approx((0.0, 0.15))
    assert command_cfg.random_waypoint_ranges.pos_x == pytest.approx((-4.8, 4.8))
    assert command_cfg.random_waypoint_ranges.pos_y == pytest.approx((-4.8, 4.8))
    assert command_cfg.random_waypoint_ranges.pos_z == pytest.approx((-0.4, 0.4))
    assert command_cfg.minimum_waypoint_separation == pytest.approx(0.9)
    assert command_cfg.maximum_waypoint_separation == pytest.approx(1.35)
    assert command_cfg.maximum_vertical_step == pytest.approx(0.15)
    assert command_cfg.random_sampling_attempts == 32
    assert command_cfg.route_sampling_attempts == 16
    assert command_cfg.random_heading_change_interval == 3
    assert command_cfg.target_cruise_speed == pytest.approx(3.50)
    assert env.reward_manager.configs["path_progress"].params["maximum_rate"] == pytest.approx(3.50)
    assert env.termination_manager.configs["path_corridor"].params["maximum_distance"] == pytest.approx(1.50)


def test_direct_ctbr_route_curriculum_is_event_independent_rank_deterministic_and_idempotent():
    envs = [_make_env(step=105_000), _make_env(step=105_000)]
    for env in envs:
        del env.event_manager
    terms = [_make_route_term(env) for env in envs]

    states = [term(env, slice(None)) for term, env in zip(terms, envs, strict=True)]
    assert states[0] == pytest.approx(states[1])
    assert states[0]["figure_eight_probability"] == pytest.approx(0.25)
    first_reward_calls = [list(env.reward_manager.set_calls) for env in envs]
    first_termination_calls = [list(env.termination_manager.set_calls) for env in envs]

    # Different local reset subsets cannot affect the global schedule, and a
    # repeated call at one counter does not replace unchanged manager configs.
    assert terms[0](envs[0], [0]) == pytest.approx(states[0])
    assert terms[1](envs[1], [1, 2, 3]) == pytest.approx(states[1])
    assert [env.reward_manager.set_calls for env in envs] == first_reward_calls
    assert [env.termination_manager.set_calls for env in envs] == first_termination_calls


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"foundation_steps": -1}, "foundation_steps must be a non-negative integer"),
        ({"straight_foundation_steps": -1}, "straight_foundation_steps must be a non-negative integer"),
        (
            {"straight_foundation_steps": 40_001},
            "straight_foundation_steps must not exceed foundation_steps",
        ),
        ({"acceptance_ramp_steps": 0}, "acceptance_ramp_steps must be a positive integer"),
        ({"final_acceptance_radius": 1.1}, "final_acceptance_radius must not exceed"),
        ({"final_figure_eight_probability": -0.1}, "must be finite and in"),
        ({"final_maximum_vertical_amplitude": -0.1}, "must be finite and non-negative"),
        ({"final_speed": 0.5}, "final_speed must not be less than initial_speed"),
        ({"initial_hard_speed": 4.0}, "initial_hard_speed must not exceed final_speed"),
        (
            {"final_hard_figure_eight_probability": 1.0, "initial_hard_figure_eight_probability": 0.5},
            "final_hard_figure_eight_probability must not exceed",
        ),
        (
            {"hard_maximum_waypoint_separation": 0.8},
            "hard_maximum_waypoint_separation must not be less",
        ),
        (
            {"final_hard_nominal_heading_change": math.pi},
            "final_hard_nominal_heading_change must not exceed",
        ),
        ({"hard_heading_change_interval": 0}, "hard_heading_change_interval must be a positive integer"),
    ],
)
def test_direct_ctbr_route_curriculum_rejects_invalid_schedule_parameters(params, message):
    with pytest.raises(ValueError, match=message):
        _make_route_term(_make_env(), **params)


def test_direct_ctbr_route_curriculum_rejects_incompatible_geometry_before_mutation():
    env = _make_env(step=105_000)
    env.command_manager.term.cfg.route_family = "bounded_ellipse"

    with pytest.raises(ValueError, match="must use route_family='random_walk'.*bounded staged route mixes"):
        _make_route_term(env)

    assert env.command_manager.term.cfg.acceptance_radius == pytest.approx(0.50)
    assert env.reward_manager.set_calls == []
    assert env.termination_manager.set_calls == []
