# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode metrics, bounded PPO, and fail-closed evaluation tests."""

import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.modules import MLP
from rsl_rl.utils import resolve_callable

import isaaclab_tasks.contrib.drone_slung_load.mdp as mdp
import isaaclab_tasks.contrib.drone_slung_load.mdp.commands as commands_module
from isaaclab_tasks.contrib.drone_slung_load.drone_slung_load_env_cfg import (
    DroneSlungLoadWaypointEnvCfg,
)
from isaaclab_tasks.contrib.drone_slung_load.system import (
    CABLE_MASS,
    DRONE_MASS,
    PAYLOAD_MASS,
)

pytestmark = pytest.mark.unit


def _metric_sample(**overrides):
    sample = {
        "position_error": torch.tensor([1.0, 2.0]),
        "swing_angle": torch.tensor([0.1, 0.3]),
        "transverse_speed": torch.tensor([2.0, 4.0]),
        "drone_speed": torch.tensor([3.0, 5.0]),
        "payload_speed": torch.tensor([4.0, 6.0]),
        "cable_relative_separation": torch.tensor([0.01, 0.03]),
        "cable_joint_error": torch.tensor([0.001, 0.003]),
    }
    sample.update(overrides)
    return sample


def test_episode_accumulator_reports_time_means_rms_and_maxima():
    accumulator = mdp.EpisodeMetricAccumulator(num_envs=2, device="cpu")
    accumulator.update(**_metric_sample())
    accumulator.update(
        **_metric_sample(
            position_error=torch.tensor([3.0, 4.0]),
            swing_angle=torch.tensor([0.5, 0.7]),
            transverse_speed=torch.tensor([4.0, 6.0]),
            drone_speed=torch.tensor([7.0, 9.0]),
            payload_speed=torch.tensor([8.0, 10.0]),
            cable_relative_separation=torch.tensor([0.05, 0.07]),
            cable_joint_error=torch.tensor([0.005, 0.007]),
        )
    )

    metrics = accumulator.metrics
    torch.testing.assert_close(metrics["position_rmse"], torch.tensor([math.sqrt(5.0), math.sqrt(10.0)]))
    torch.testing.assert_close(metrics["position_error_max"], torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(metrics["swing_angle_mean"], torch.tensor([0.3, 0.5]))
    torch.testing.assert_close(metrics["swing_angle_rms"], torch.tensor([math.sqrt(0.13), math.sqrt(0.29)]))
    torch.testing.assert_close(metrics["swing_angle_max"], torch.tensor([0.5, 0.7]))
    torch.testing.assert_close(metrics["transverse_speed_rms"], torch.tensor([math.sqrt(10.0), math.sqrt(26.0)]))
    torch.testing.assert_close(metrics["drone_speed_mean"], torch.tensor([5.0, 7.0]))
    torch.testing.assert_close(metrics["drone_speed_max"], torch.tensor([7.0, 9.0]))
    torch.testing.assert_close(metrics["payload_speed_mean"], torch.tensor([6.0, 8.0]))
    torch.testing.assert_close(metrics["payload_speed_max"], torch.tensor([8.0, 10.0]))
    torch.testing.assert_close(metrics["cable_relative_separation_mean"], torch.tensor([0.03, 0.05]))
    torch.testing.assert_close(metrics["cable_relative_separation_max"], torch.tensor([0.05, 0.07]))
    torch.testing.assert_close(metrics["cable_joint_error_mean"], torch.tensor([0.003, 0.005]))
    torch.testing.assert_close(metrics["cable_joint_error_max"], torch.tensor([0.005, 0.007]))


def test_episode_accumulator_tracks_first_completion_time_and_subset_reset():
    accumulator = mdp.EpisodeMetricAccumulator(num_envs=2, device="cpu")
    common = _metric_sample(
        position_error=torch.ones(2),
        swing_angle=torch.zeros(2),
        transverse_speed=torch.zeros(2),
        drone_speed=torch.zeros(2),
        payload_speed=torch.zeros(2),
        cable_relative_separation=torch.zeros(2),
        cable_joint_error=torch.zeros(2),
    )
    accumulator.update(
        **common,
        waypoint_fraction=torch.tensor([0.5, 0.25]),
        waypoint_completed=torch.tensor([False, False]),
        elapsed_time=torch.ones(2),
    )
    accumulator.update(
        **common,
        waypoint_fraction=torch.tensor([1.0, 0.75]),
        waypoint_completed=torch.tensor([True, False]),
        elapsed_time=torch.full((2,), 2.0),
    )
    accumulator.update(
        **common,
        waypoint_fraction=torch.ones(2),
        waypoint_completed=torch.ones(2, dtype=torch.bool),
        elapsed_time=torch.full((2,), 3.0),
    )

    torch.testing.assert_close(accumulator.metrics["waypoint_completion_time"], torch.tensor([2.0, 3.0]))
    preserved = {name: value[1].clone() for name, value in accumulator.metrics.items()}
    accumulator.reset(torch.tensor([0]))
    assert all(value[0].item() == 0.0 for value in accumulator.metrics.values())
    for name, value in accumulator.metrics.items():
        torch.testing.assert_close(value[1], preserved[name])


def test_episode_accumulator_freezes_trajectory_metrics_after_first_completion():
    accumulator = mdp.EpisodeMetricAccumulator(num_envs=2, device="cpu")
    accumulator.update(
        **_metric_sample(),
        waypoint_fraction=torch.ones(2),
        waypoint_completed=torch.ones(2, dtype=torch.bool),
        elapsed_time=torch.ones(2),
    )
    frozen = {name: value.clone() for name, value in accumulator.metrics.items()}
    accumulator.update(
        **_metric_sample(
            position_error=torch.full((2,), 100.0),
            swing_angle=torch.full((2,), 2.0),
            cable_relative_separation=torch.full((2,), 1.0),
            cable_joint_error=torch.full((2,), 1.0),
        ),
        waypoint_fraction=torch.ones(2),
        waypoint_completed=torch.ones(2, dtype=torch.bool),
        elapsed_time=torch.full((2,), 2.0),
    )

    for name, value in accumulator.metrics.items():
        torch.testing.assert_close(value, frozen[name])


def _patch_metric_state(monkeypatch, num_envs: int) -> None:
    monkeypatch.setattr(commands_module, "payload_transverse_velocity_b", lambda env: torch.zeros(num_envs, 3))
    monkeypatch.setattr(commands_module, "total_swing_angle", lambda env: torch.zeros(num_envs, 1))
    monkeypatch.setattr(commands_module, "link_lin_vel_w", lambda env, asset_cfg=None: torch.zeros(num_envs, 3))
    monkeypatch.setattr(commands_module, "cable_relative_separation", lambda env: torch.zeros(num_envs, 1))
    monkeypatch.setattr(commands_module, "cable_joint_error", lambda env: torch.zeros(num_envs, 1))


def test_waypoint_completion_is_previewed_and_snapshotted_before_autoreset(monkeypatch):
    _patch_metric_state(monkeypatch, num_envs=1)
    pose = torch.tensor([[2.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0]])
    robot = SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=pose)))
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.01,
        episode_length_buf=torch.tensor([25]),
        scene=SimpleNamespace(env_origins=torch.zeros(1, 3)),
    )
    term = object.__new__(commands_module.WaypointSequenceCommand)
    term._env = env
    term.robot = robot
    term.cfg = SimpleNamespace(resampling_time_range=(1.0e8, 1.0e8), acceptance_radius=0.5)
    term.metrics = {}
    term.time_left = torch.full((1,), 1.0e8)
    term.command_counter = torch.zeros(1, dtype=torch.long)
    term._initialize_episode_metrics()
    term._resample_command = lambda env_ids: None
    term._update_command = lambda: None
    term.route_anchor_e = torch.tensor([[0.0, 0.0, 1.5]])
    term.waypoints_e = torch.tensor([[[1.0, 0.0, 1.5], [2.0, 0.0, 1.5]]])
    term.current_index = torch.tensor([1])
    term.completed = torch.tensor([False])
    env.command_manager = SimpleNamespace(get_term=lambda name: term)

    mdp.record_episode_metrics(env)
    assert not term.completed.item()
    assert term.metrics["waypoint_completed"].item() == 1.0
    assert term.metrics["waypoint_completion_time"].item() == pytest.approx(0.25)

    term.reset(torch.tensor([0]))
    assert term.last_episode_metrics["waypoint_completed"].item() == 1.0
    assert term.last_episode_metrics["waypoint_completion_time"].item() == pytest.approx(0.25)


def test_hover_biased_beta_is_bounded_and_initializes_at_loaded_system_hover():
    from isaaclab_tasks.contrib.drone_slung_load.config.newton_drone.agents.bounded_beta_distribution import (
        HoverBiasedBetaDistribution,
    )

    hover = 2.0 * ((DRONE_MASS + PAYLOAD_MASS + CABLE_MASS) / DRONE_MASS) / 3.5 - 1.0
    target = torch.tensor([hover, 0.0, 0.0, 0.0])
    distribution = HoverBiasedBetaDistribution(output_dim=4, initial_mean=target.tolist(), concentration=1000.0)
    mlp = MLP(input_dim=5, output_dim=distribution.input_dim, hidden_dims=[8], activation="elu")
    distribution.init_mlp_weights(mlp)

    observations = torch.randn(32, 5)
    torch.testing.assert_close(
        distribution.deterministic_output(mlp(observations)),
        target.expand(32, -1),
        atol=1.0e-5,
        rtol=0.0,
    )
    distribution.update(mlp(observations))
    samples = distribution.sample()
    assert torch.all((samples >= -1.0) & (samples <= 1.0))
    assert torch.isfinite(mlp[-2].bias).all()


def test_waypoint_ppo_configuration_is_serializable_and_asymmetric():
    from isaaclab_rl.entrypoints.backends.cli_args_rsl_rl import update_rsl_rl_cfg

    from isaaclab_tasks.contrib.drone_slung_load import (
        DIRECT_CTBR_HARD_ROUTES_EXPERIMENT_NAME,
        DRONE_SLUNG_LOAD_WANDB_PROJECT,
        ENHANCED_EXPERIMENT_NAME,
    )
    from isaaclab_tasks.contrib.drone_slung_load.config.newton_drone.agents.rsl_rl_ppo_cfg import (
        DroneSlungLoadWaypointDirectCTBRPPORunnerCfg,
        DroneSlungLoadWaypointEnhancedPPORunnerCfg,
        DroneSlungLoadWaypointPPORunnerCfg,
    )

    cfg = DroneSlungLoadWaypointPPORunnerCfg()
    assert cfg.actor.hidden_dims == [128, 128]
    assert cfg.critic.hidden_dims == [128, 128]
    assert cfg.obs_groups == {"actor": ["policy"], "critic": ["policy", "privileged"]}
    assert cfg.actor.distribution_cfg.action_range == (-1.0, 1.0)
    assert resolve_callable(cfg.actor.distribution_cfg.to_dict()["class_name"]).__name__ == (
        "HoverBiasedBetaDistribution"
    )
    assert cfg.logger == "wandb"
    assert cfg.wandb_project == DRONE_SLUNG_LOAD_WANDB_PROJECT == "drone_slung_load_waypoint_flare"
    assert cfg.experiment_name == "drone_slung_load_waypoint_flare"
    assert cfg.num_steps_per_env == 128
    assert cfg.algorithm.gamma == pytest.approx(0.997)
    assert cfg.algorithm.lam == pytest.approx(0.99)

    enhanced = DroneSlungLoadWaypointEnhancedPPORunnerCfg()
    assert enhanced.experiment_name == ENHANCED_EXPERIMENT_NAME
    assert enhanced.experiment_name == "drone_slung_load_waypoint_flare_enhanced_curvature_speed_v13"
    assert enhanced.wandb_project == cfg.wandb_project
    assert enhanced.max_iterations == 2000
    assert enhanced.save_interval == 10
    assert not enhanced.init_at_random_ep_len
    assert enhanced.num_steps_per_env == 100
    assert enhanced.actor.distribution_cfg.init_std == pytest.approx((0.03, 0.03, 0.03, 0.03))
    assert enhanced.actor.distribution_cfg.std_range == pytest.approx((0.005, 0.5))
    assert enhanced.algorithm.entropy_coef == pytest.approx(0.002)
    assert enhanced.algorithm.num_learning_epochs == 2
    assert enhanced.algorithm.learning_rate == pytest.approx(1.0e-4)
    assert enhanced.algorithm.final_learning_rate == pytest.approx(3.0e-5)
    assert enhanced.algorithm.learning_rate_decay_updates == 400
    assert enhanced.algorithm.learning_rate_decay_start_update == 1_600
    assert enhanced.algorithm.final_entropy_coef == pytest.approx(5.0e-4)
    assert enhanced.algorithm.entropy_decay_updates == 400
    assert enhanced.algorithm.entropy_decay_start_update == 1_600
    assert enhanced.algorithm.kl_acceptance_lr_recovery_factor == pytest.approx(1.01)
    assert enhanced.algorithm.kl_guard_threshold == pytest.approx(0.015)
    assert enhanced.algorithm.schedule == "fixed"
    assert enhanced.algorithm.gamma == pytest.approx(0.997)
    assert enhanced.algorithm.lam == pytest.approx(0.99)

    direct = DroneSlungLoadWaypointDirectCTBRPPORunnerCfg()
    assert direct.experiment_name == DIRECT_CTBR_HARD_ROUTES_EXPERIMENT_NAME
    assert direct.experiment_name == "drone_slung_load_waypoint_flare_direct_ctbr_hard_routes_v18"
    assert direct.wandb_project == DRONE_SLUNG_LOAD_WANDB_PROJECT
    assert direct.num_steps_per_env == 500
    assert direct.max_iterations == 680
    assert direct.save_interval == 5
    assert direct.actor.distribution_cfg.init_std == pytest.approx((0.03, 0.03, 0.03, 0.03))
    assert direct.actor.distribution_cfg.std_range == pytest.approx((0.005, 0.15))
    assert direct.algorithm.physical_body_rate_limits == pytest.approx((15.0, 15.0, 5.0))
    assert direct.algorithm.entropy_coef == pytest.approx(0.001)
    assert direct.algorithm.final_entropy_coef == pytest.approx(0.0002)
    assert direct.algorithm.num_learning_epochs == 2
    assert direct.algorithm.num_mini_batches == 20
    assert direct.algorithm.learning_rate == pytest.approx(1.0e-4)
    assert direct.algorithm.learning_rate_decay_start_update == 639
    assert direct.algorithm.learning_rate_decay_updates == 40
    assert direct.algorithm.entropy_decay_start_update == 639
    assert direct.algorithm.entropy_decay_updates == 40
    assert direct.algorithm.gamma == pytest.approx(0.999)
    assert direct.algorithm.lam == pytest.approx(0.999)

    cli_defaults = SimpleNamespace(
        seed=None,
        resume=False,
        load_run=None,
        checkpoint=None,
        experiment_name=None,
        run_name=None,
        logger=None,
        log_project_name=None,
    )
    configured = update_rsl_rl_cfg(enhanced, cli_defaults)
    assert configured.logger == "wandb"
    assert configured.wandb_project == DRONE_SLUNG_LOAD_WANDB_PROJECT


def test_direct_ctbr_hard_route_trace_and_decay_match_control_step_boundaries():
    from isaaclab_tasks.contrib.drone_slung_load.config.newton_drone.agents.durability_ppo import (
        exponential_decay,
    )
    from isaaclab_tasks.contrib.drone_slung_load.config.newton_drone.agents.rsl_rl_ppo_cfg import (
        DroneSlungLoadWaypointDirectCTBRPPORunnerCfg,
    )
    from isaaclab_tasks.contrib.drone_slung_load.drone_slung_load_env_cfg import (
        DroneSlungLoadWaypointDirectCTBREnvCfg,
    )

    runner = DroneSlungLoadWaypointDirectCTBRPPORunnerCfg()
    env = DroneSlungLoadWaypointDirectCTBREnvCfg()
    rollout_steps = runner.num_steps_per_env
    final_curriculum_step = 320_000
    decay_start = runner.algorithm.learning_rate_decay_start_update
    decay_updates = runner.algorithm.learning_rate_decay_updates

    assert env.decimation * env.sim.dt == pytest.approx(0.01)
    assert decay_start * rollout_steps == 319_500
    assert (decay_start + 1) * rollout_steps == final_curriculum_step
    assert decay_updates * rollout_steps == 20_000
    assert decay_start + decay_updates == runner.max_iterations - 1
    assert runner.max_iterations * rollout_steps == 340_000
    assert runner.algorithm.entropy_decay_start_update == decay_start
    assert runner.algorithm.entropy_decay_updates == decay_updates
    for initial, final in (
        (runner.algorithm.learning_rate, runner.algorithm.final_learning_rate),
        (runner.algorithm.entropy_coef, runner.algorithm.final_entropy_coef),
    ):
        first_decay = initial * (final / initial) ** (1.0 / decay_updates)
        assert exponential_decay(initial, final, decay_start, decay_updates, decay_start) == pytest.approx(initial)
        assert exponential_decay(initial, final, 640, decay_updates, decay_start) == pytest.approx(first_decay)
        assert exponential_decay(initial, final, 679, decay_updates, decay_start) == pytest.approx(final)
        assert exponential_decay(initial, final, 680, decay_updates, decay_start) == pytest.approx(final)

    default_rollout_samples = env.scene.num_envs * rollout_steps
    production_rollout_samples = 2_048 * rollout_steps
    assert env.scene.num_envs == 32
    assert default_rollout_samples == 16_000
    assert default_rollout_samples % runner.algorithm.num_mini_batches == 0
    assert production_rollout_samples == 1_024_000
    assert production_rollout_samples % runner.algorithm.num_mini_batches == 0

    trace = runner.algorithm.gamma * runner.algorithm.lam
    episode_steps = round(env.episode_length_s / (env.decimation * env.sim.dt))
    assert episode_steps == 2_000
    assert trace == pytest.approx(0.998001)
    assert trace**100 == pytest.approx(0.8186488294786379)
    assert trace**300 == pytest.approx(0.5486469074855014)
    assert trace**rollout_steps == pytest.approx(0.367695424770969)
    assert runner.algorithm.gamma**episode_steps == pytest.approx(0.13519992539749945)
    assert trace**episode_steps == pytest.approx(0.018279019827490466)


def _load_evaluation_module():
    script = Path(__file__).parents[4] / "scripts" / "reinforcement_learning" / "rsl_rl" / "evaluate_checkpoints.py"
    spec = importlib.util.spec_from_file_location("drone_slung_load_evaluate", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe_counts(**overrides):
    counts = {
        "drone_crashes": 0,
        "payload_crashes": 0,
        "illegal_state_terminations": 0,
        "workspace_exits": 0,
        "cable_integrity_failures": 0,
        "cable_relative_separation_mean": 0.0005,
        "cable_relative_separation_max": 0.001,
        "cable_joint_error_mean": 0.0005,
        "cable_joint_error_max": 0.001,
    }
    counts.update(overrides)
    return counts


def _candidate(
    name: str,
    *,
    completion: float,
    completion_time: float,
    rmse: float,
    safety=None,
    swing=0.1,
    rms_swing=None,
    peak_swing=0.2,
    transverse_speed=0.1,
    cross_track_rms=0.1,
    cross_track_max=0.2,
    completion_fraction=None,
    arrival_rate=1.0,
    drone_speed=2.0,
):
    fraction = completion if completion_fraction is None else completion_fraction
    passed = 24.0 * fraction

    def metric_groups(episode_count: int):
        completions = completion * episode_count
        return {
            "tracking": {
                "position_rmse": rmse,
                "max_position_error": 2.0 * rmse,
                "cross_track_error_mean": cross_track_rms,
                "cross_track_error_rms": cross_track_rms,
                "cross_track_error_max": cross_track_max,
            },
            "swing": {
                "mean_angle": swing,
                "rms_angle": swing if rms_swing is None else rms_swing,
                "max_angle": peak_swing,
                "transverse_speed_rms": transverse_speed,
            },
            "speed": {
                "drone_mean": drone_speed,
                "drone_max": 2.0 * drone_speed,
                "payload_mean": drone_speed,
                "payload_max": 2.0 * drone_speed,
            },
            "route": {
                "targets_per_route": 24,
                "waypoints_passed_total": passed * episode_count,
                "waypoints_passed_mean": passed,
                "traversal_fraction": fraction,
                "arc_length_traversed_total": arrival_rate * 12.0,
                "arc_length_traversed_mean": arrival_rate * 12.0 / episode_count,
                "episode_arc_length_rate": arrival_rate,
                "active_arc_length_rate": arrival_rate,
                "mean_reported_arc_length_rate": arrival_rate,
                "route_completions_total": completions,
                "completion_rate": completion,
                "completion_time": completion_time,
                "episode_time_total": 12.0,
                "active_time_total": completion_time if completion > 0.0 else 12.0,
            },
            "precision": {
                "hits_total": passed * episode_count,
                "hits_mean": passed,
                "hit_fraction_of_passed": 1.0 if passed > 0.0 else 0.0,
                "mean_episode_hit_fraction": 1.0 if passed > 0.0 else 0.0,
                "misses_total": 0.0,
                "misses_mean": 0.0,
                "miss_distance_mean": 0.0,
                "miss_distance_max": 0.0,
                "episode_hit_rate_hz": arrival_rate if passed > 0.0 else 0.0,
                "active_hit_rate_hz": arrival_rate if passed > 0.0 else 0.0,
                "mean_interarrival_time": 1.0 / arrival_rate if passed > 0.0 else 0.0,
                "min_interarrival_time": 1.0 / arrival_rate if passed > 0.0 else 0.0,
                "max_interarrival_time": 1.0 / arrival_rate if passed > 0.0 else 0.0,
                "target_distance_total": passed,
                "episode_target_distance_rate": arrival_rate if passed > 0.0 else 0.0,
                "active_target_distance_rate": arrival_rate if passed > 0.0 else 0.0,
            },
            "termination": {
                "success_terminations": completions,
                "path_corridor_exits": 0,
                "path_corridor_exit_rate": 0.0,
            },
            "physics_safety": _safe_counts() if safety is None else safety,
        }

    result = {
        "seed": 42,
        "task_id": "randomized-task",
        "episode_count": 10,
        "checkpoint_path": name,
        "evaluation_suite": {
            "kind": "seeded_randomized_finite_routes",
            "route_suite_sha256": "a" * 64,
            "route_family_counts": {"ellipse": 5, "figure_eight": 5},
        },
        **metric_groups(10),
    }
    result["route_families"] = {
        family: {"episode_count": 5, **copy.deepcopy(metric_groups(5))} for family in ("ellipse", "figure_eight")
    }
    return result


def _hard_candidate(name: str, **kwargs):
    """Convert the legacy candidate fixture to the final hard-route capability."""
    result = _candidate(name, **kwargs)
    families = result["route_families"]
    result["evaluation_suite"]["route_family_counts"] = {"figure_eight": 5, "random_corner": 5}
    result["route_families"] = {
        "figure_eight": families["figure_eight"],
        "random_corner": families["ellipse"],
    }
    return result


def _drone_episode(route_family_id: float):
    evaluation = _load_evaluation_module()
    episode = {name: 0.0 for name in evaluation._COMMON_REQUIRED_EPISODE_KEYS}
    episode.update(
        slung_load_metrics_available=0.0,
        route_family_id=route_family_id,
        route_waypoints_passed=24.0,
        route_traversal_fraction=1.0,
        route_arc_length_traversed=27.0,
        route_arc_length_traversal_rate=2.7,
        waypoint_completion_fraction=1.0,
        waypoint_completed=1.0,
        waypoint_completion_time=10.0,
        waypoint_count=24.0,
        waypoint_arrivals=24.0,
        waypoint_precision_hits=24.0,
        waypoint_precision_hit_fraction=1.0,
        waypoint_arrival_time_mean=0.4,
        waypoint_arrival_time_min=0.2,
        waypoint_arrival_time_max=0.6,
        waypoint_throughput=2.4,
        route_completions=1.0,
        target_distance_completed=27.0,
        episode_duration=10.0,
        success_termination=1.0,
        drone_speed_mean=2.8,
        drone_speed_max=4.0,
    )
    return episode


def test_evaluation_aggregation_is_json_serializable_and_reports_safety():
    evaluation = _load_evaluation_module()
    episodes = [
        {
            "position_rmse": 0.4,
            "position_error_max": 0.8,
            "cross_track_error_mean": 0.2,
            "cross_track_error_rms": 0.25,
            "cross_track_error_max": 0.6,
            "swing_angle_mean": 0.1,
            "swing_angle_rms": 0.12,
            "swing_angle_max": 0.3,
            "transverse_speed_rms": 0.5,
            "drone_speed_mean": 2.0,
            "drone_speed_max": 4.0,
            "payload_speed_mean": 1.5,
            "payload_speed_max": 3.0,
            "cable_relative_separation_mean": 0.01,
            "cable_relative_separation_max": 0.03,
            "cable_joint_error_mean": 0.001,
            "cable_joint_error_max": 0.003,
            "route_family_id": 0.0,
            "route_waypoints_passed": 12.0,
            "route_traversal_fraction": 1.0,
            "route_arc_length_traversed": 13.0,
            "route_arc_length_traversal_rate": 3.25,
            "waypoint_completion_fraction": 1.0,
            "waypoint_completed": 1.0,
            "waypoint_completion_time": 4.0,
            "waypoint_count": 12.0,
            "waypoint_arrivals": 10.0,
            "waypoint_precision_hits": 10.0,
            "waypoint_precision_hit_fraction": 10.0 / 12.0,
            "waypoint_precision_misses": 2.0,
            "waypoint_precision_miss_distance_mean": 0.2,
            "waypoint_precision_miss_distance_max": 0.3,
            "waypoint_arrival_time_mean": 1.0 / 3.0,
            "waypoint_arrival_time_min": 0.2,
            "waypoint_arrival_time_max": 0.6,
            "waypoint_throughput": 1.0,
            "route_completions": 1.0,
            "target_distance_completed": 13.0,
            "episode_duration": 12.0,
            "drone_crash": 0,
            "payload_crash": 1,
            "illegal_state": 0,
            "workspace_exit": 0,
            "path_corridor_exit": 0,
            "cable_integrity_failure": 0,
            "success_termination": 1,
        },
        {
            "position_rmse": 0.2,
            "position_error_max": 0.7,
            "cross_track_error_mean": 0.4,
            "cross_track_error_rms": 0.5,
            "cross_track_error_max": 0.7,
            "swing_angle_mean": 0.2,
            "swing_angle_rms": 0.24,
            "swing_angle_max": 0.4,
            "transverse_speed_rms": 0.7,
            "drone_speed_mean": 1.0,
            "drone_speed_max": 3.0,
            "payload_speed_mean": 0.75,
            "payload_speed_max": 2.5,
            "cable_relative_separation_mean": 0.03,
            "cable_relative_separation_max": 0.05,
            "cable_joint_error_mean": 0.002,
            "cable_joint_error_max": 0.004,
            "route_family_id": 1.0,
            "route_waypoints_passed": 9.0,
            "route_traversal_fraction": 0.75,
            "route_arc_length_traversed": 9.0,
            "route_arc_length_traversal_rate": 0.75,
            "waypoint_completion_fraction": 0.75,
            "waypoint_completed": 0.0,
            "waypoint_completion_time": 0.0,
            "waypoint_count": 12.0,
            "waypoint_arrivals": 6.0,
            "waypoint_precision_hits": 6.0,
            "waypoint_precision_hit_fraction": 2.0 / 3.0,
            "waypoint_precision_misses": 3.0,
            "waypoint_precision_miss_distance_mean": 0.4,
            "waypoint_precision_miss_distance_max": 0.6,
            "waypoint_arrival_time_mean": 1.2,
            "waypoint_arrival_time_min": 0.5,
            "waypoint_arrival_time_max": 2.0,
            "waypoint_throughput": 0.75,
            "route_completions": 0.0,
            "target_distance_completed": 9.0,
            "episode_duration": 12.0,
            "drone_crash": 1,
            "payload_crash": 0,
            "illegal_state": 1,
            "workspace_exit": 1,
            "path_corridor_exit": 1,
            "cable_integrity_failure": 1,
            "success_termination": 0,
        },
    ]

    result = evaluation.aggregate_checkpoint_results(
        episodes,
        task_id="IsaacContrib-DroneSlungLoad-Waypoint-FLARE",
        checkpoint_path="/tmp/model_1200.pt",
        seed=42,
        route_suite_sha256="a" * 64,
    )

    assert result["tracking"]["position_rmse"] == pytest.approx(0.3)
    assert result["tracking"]["max_position_error"] == pytest.approx(0.8)
    assert result["tracking"]["cross_track_error_mean"] == pytest.approx(0.3)
    assert result["tracking"]["cross_track_error_rms"] == pytest.approx(0.375)
    assert result["tracking"]["cross_track_error_max"] == pytest.approx(0.7)
    assert result["route"]["completion_rate"] == pytest.approx(0.5)
    assert result["route"]["traversal_fraction"] == pytest.approx(0.875)
    assert result["route"]["waypoints_passed_total"] == 21
    assert result["route"]["arc_length_traversed_total"] == pytest.approx(22.0)
    assert result["route"]["active_arc_length_rate"] == pytest.approx(22.0 / 16.0)
    assert result["precision"]["hits_total"] == 16
    assert result["precision"]["misses_total"] == 5
    assert result["precision"]["hit_fraction_of_passed"] == pytest.approx(16.0 / 21.0)
    assert result["precision"]["mean_interarrival_time"] == pytest.approx((10.0 / 3.0 + 7.2) / 16.0)
    assert result["precision"]["target_distance_total"] == pytest.approx(22.0)
    assert result["precision"]["active_target_distance_rate"] == pytest.approx(22.0 / 16.0)
    assert result["speed"]["drone_mean"] == pytest.approx(1.5)
    assert result["speed"]["drone_max"] == pytest.approx(4.0)
    assert result["physics_safety"]["drone_crashes"] == 1
    assert result["physics_safety"]["illegal_state_terminations"] == 1
    assert result["physics_safety"]["cable_integrity_failures"] == 1
    assert result["termination"] == {
        "success_terminations": 1,
        "path_corridor_exits": 1,
        "path_corridor_exit_rate": 0.5,
    }
    assert result["route_families"]["ellipse"]["route"]["completion_rate"] == 1.0
    assert result["route_families"]["figure_eight"]["route"]["traversal_fraction"] == pytest.approx(0.75)
    json.dumps(result)


def test_evaluation_aggregation_fails_closed_when_episode_metric_is_missing():
    evaluation = _load_evaluation_module()
    with pytest.raises(ValueError, match="missing required metrics"):
        evaluation.aggregate_checkpoint_results(
            [{"waypoint_completed": 0.0}],
            task_id="task",
            checkpoint_path="model.pt",
            seed=1,
            route_suite_sha256="a" * 64,
        )


def test_evaluation_aggregation_marks_drone_only_load_metrics_not_applicable():
    evaluation = _load_evaluation_module()

    result = evaluation.aggregate_checkpoint_results(
        [_drone_episode(1.0), _drone_episode(2.0)],
        task_id="IsaacContrib-Drone-Waypoint-FLARE-DirectCTBR",
        checkpoint_path="model_20.pt",
        seed=7,
        route_suite_sha256="b" * 64,
        route_family_ids=(1, 2),
    )

    assert result["task_profile"] == "drone_only"
    assert result["evaluation_suite"]["task_profile"] == "drone_only"
    assert "swing" not in result
    assert result["speed"] == {"drone_mean": 2.8, "drone_max": 4.0}
    assert result["physics_safety"] == {
        "drone_crashes": 0,
        "illegal_state_terminations": 0,
        "workspace_exits": 0,
    }
    assert result["evaluation_suite"]["route_family_counts"] == {"figure_eight": 1, "random_corner": 1}
    assert set(result["route_families"]) == {"figure_eight", "random_corner"}
    assert (
        evaluation.select_checkpoint(
            [result],
            min_completion_rate=1.0,
            min_traversal_fraction=1.0,
            min_active_arc_rate=2.5,
        )["checkpoint_path"]
        == "model_20.pt"
    )


def test_evaluation_capability_defaults_to_strict_slung_load_and_must_be_uniform():
    evaluation = _load_evaluation_module()
    missing_load_metrics = {name: 0.0 for name in evaluation._COMMON_REQUIRED_EPISODE_KEYS}
    missing_load_metrics.update(route_family_id=0.0, waypoint_count=24.0, episode_duration=15.0)
    with pytest.raises(ValueError, match="missing required metrics"):
        evaluation.aggregate_checkpoint_results(
            [missing_load_metrics],
            task_id="legacy-slung-task",
            checkpoint_path="model.pt",
            seed=1,
            route_suite_sha256="a" * 64,
        )

    drone = dict(missing_load_metrics, slung_load_metrics_available=0.0)
    slung = dict(missing_load_metrics, slung_load_metrics_available=1.0)
    with pytest.raises(ValueError, match="disagree about slung-load metric availability"):
        evaluation.aggregate_checkpoint_results(
            [drone, slung],
            task_id="mixed-task",
            checkpoint_path="model.pt",
            seed=1,
            route_suite_sha256="a" * 64,
        )


def test_evaluation_aggregation_requires_both_seeded_route_families():
    evaluation = _load_evaluation_module()
    episode = {name: 0.0 for name in evaluation._REQUIRED_EPISODE_KEYS}
    episode.update(
        route_family_id=0.0,
        waypoint_count=24.0,
        episode_duration=15.0,
    )

    with pytest.raises(ValueError, match="missing route families: figure_eight"):
        evaluation.aggregate_checkpoint_results(
            [episode],
            task_id="task",
            checkpoint_path="model.pt",
            seed=1,
            route_suite_sha256="a" * 64,
        )

    with pytest.raises(ValueError, match="missing route families: random_corner"):
        evaluation.aggregate_checkpoint_results(
            [_drone_episode(1.0)],
            task_id="hard-task",
            checkpoint_path="model.pt",
            seed=1,
            route_suite_sha256="a" * 64,
            route_family_ids=(1, 2),
        )


@pytest.mark.parametrize("route_family_ids", [(0, 2), (0, 1, 2), (1, 1), (True, 1), (1, 3)])
def test_evaluation_aggregation_rejects_unknown_or_mixed_route_capabilities(route_family_ids):
    evaluation = _load_evaluation_module()
    with pytest.raises(ValueError, match="route_family_ids"):
        evaluation.aggregate_checkpoint_results(
            [],
            task_id="task",
            checkpoint_path="model.pt",
            seed=1,
            route_suite_sha256="a" * 64,
            route_family_ids=route_family_ids,
        )


@pytest.mark.parametrize(
    ("route_family_id", "message"),
    [(3.0, "invalid route family ID"), (1.5, "invalid route family ID"), (0.0, "configured route-family")],
)
def test_evaluation_aggregation_rejects_invalid_or_cross_capability_episode_ids(route_family_id, message):
    evaluation = _load_evaluation_module()
    with pytest.raises(ValueError, match=message):
        evaluation.aggregate_checkpoint_results(
            [_drone_episode(route_family_id), _drone_episode(2.0)],
            task_id="hard-task",
            checkpoint_path="model.pt",
            seed=1,
            route_suite_sha256="a" * 64,
            route_family_ids=(1, 2),
        )


def test_evaluation_route_family_modes_resolve_to_exact_legacy_and_hard_capabilities():
    evaluation = _load_evaluation_module()

    assert evaluation._route_family_ids_for_cfg("bounded_template_mix") == (0, 1)
    assert evaluation._route_family_ids_for_cfg("bounded_hard_mix") == (1, 2)
    assert evaluation._ROUTE_FAMILY_NAMES == {0: "ellipse", 1: "figure_eight", 2: "random_corner"}

    for unsupported in ("random_walk", "bounded_ellipse", None):
        with pytest.raises(ValueError, match="requires route_family"):
            evaluation._route_family_ids_for_cfg(unsupported)


def test_checkpoint_selection_rejects_mixed_or_missing_randomized_route_suites():
    evaluation = _load_evaluation_module()
    first = _candidate("first", completion=1.0, completion_time=4.0, rmse=0.1)
    different_route = _candidate("different", completion=1.0, completion_time=3.0, rmse=0.1)
    different_route["evaluation_suite"]["route_suite_sha256"] = "b" * 64
    different_mix = _candidate("different-mix", completion=1.0, completion_time=3.0, rmse=0.1)
    different_mix["evaluation_suite"]["route_family_counts"] = {"ellipse": 6, "figure_eight": 4}
    missing_suite = _candidate("missing", completion=1.0, completion_time=2.0, rmse=0.1)
    del missing_suite["evaluation_suite"]

    with pytest.raises(ValueError, match="randomized route suite digest"):
        evaluation.select_checkpoint([first, different_route])
    with pytest.raises(ValueError, match="randomized route suite digest"):
        evaluation.select_checkpoint([first, different_mix])
    with pytest.raises(ValueError, match="randomized route suite digest"):
        evaluation.select_checkpoint([missing_suite])


def test_checkpoint_selection_accepts_hard_suite_and_rejects_cross_capability_comparison():
    evaluation = _load_evaluation_module()
    legacy = _candidate("legacy", completion=1.0, completion_time=4.0, rmse=0.1)
    hard = _hard_candidate("hard", completion=1.0, completion_time=3.0, rmse=0.1)

    assert evaluation.select_checkpoint([hard])["checkpoint_path"] == "hard"
    with pytest.raises(ValueError, match="randomized route suite digest"):
        evaluation.select_checkpoint([legacy, hard])


@pytest.mark.parametrize(
    "family_counts",
    [
        {"ellipse": 5, "random_corner": 5},
        {"ellipse": 5, "figure_eight": 5, "random_corner": 5},
        {"figure_eight": 5, "random_corner": 0},
    ],
)
def test_checkpoint_selection_rejects_unknown_or_mixed_family_count_capabilities(family_counts):
    evaluation = _load_evaluation_module()
    result = _candidate("invalid", completion=1.0, completion_time=4.0, rmse=0.1)
    result["episode_count"] = sum(family_counts.values())
    result["evaluation_suite"]["route_family_counts"] = family_counts

    with pytest.raises(ValueError, match="randomized route suite digest"):
        evaluation.select_checkpoint([result])


def test_route_suite_digest_includes_ordered_route_family_assignments():
    evaluation = _load_evaluation_module()
    command_term = SimpleNamespace(
        waypoints_e=torch.arange(36, dtype=torch.float32).reshape(2, 6, 3),
        route_anchor_e=torch.zeros(2, 3),
        route_family_id=torch.tensor([1, 2], dtype=torch.long),
    )
    mixed_digest = evaluation._route_suite_sha256(command_term, 2, (1, 2))
    command_term.route_family_id[:] = torch.tensor([2, 1])
    swapped_digest = evaluation._route_suite_sha256(command_term, 2, (1, 2))

    assert mixed_digest != swapped_digest

    bool_only_command = SimpleNamespace(
        waypoints_e=command_term.waypoints_e,
        route_anchor_e=command_term.route_anchor_e,
        route_is_figure_eight=torch.tensor([False, True]),
    )
    with pytest.raises(ValueError, match="integer route_family_id"):
        evaluation._route_suite_sha256(bool_only_command, 2, (1, 2))


@pytest.mark.parametrize(
    ("family_ids", "message"),
    [
        (torch.tensor([1.0, 2.0]), "rank-one integer"),
        (torch.tensor([[1, 2]], dtype=torch.long), "rank-one integer"),
        (torch.tensor([True, False]), "rank-one integer"),
        (torch.tensor([1, 3], dtype=torch.long), "unknown route family IDs"),
        (torch.tensor([0, 1], dtype=torch.long), "configured evaluation capability"),
        (torch.tensor([1, 1], dtype=torch.long), "missing route families: random_corner"),
        (torch.tensor([1], dtype=torch.long), "shorter than the scored episode count"),
    ],
)
def test_route_suite_family_tensor_validation_fails_closed(family_ids, message):
    evaluation = _load_evaluation_module()
    with pytest.raises(ValueError, match=message):
        evaluation._command_route_family_ids(SimpleNamespace(route_family_id=family_ids), 2, (1, 2))


def test_checkpoint_selection_prioritizes_completion_then_throughput_time_and_speed():
    evaluation = _load_evaluation_module()
    results = [
        _candidate("low-rate", completion=0.8, completion_time=2.0, rmse=0.1),
        _candidate("slow", completion=1.0, completion_time=4.0, arrival_rate=2.0, rmse=0.1),
        _candidate("best", completion=1.0, completion_time=3.0, arrival_rate=3.0, rmse=0.4),
        _candidate("same-throughput-slower", completion=1.0, completion_time=3.5, arrival_rate=3.0, rmse=0.1),
    ]
    assert evaluation.select_checkpoint(results)["checkpoint_path"] == "best"


def test_checkpoint_selection_uses_partial_route_fraction_when_explicitly_allowed():
    evaluation = _load_evaluation_module()
    results = [
        _candidate(
            "nearly-done",
            completion=0.0,
            completion_fraction=23.0 / 24.0,
            completion_time=0.0,
            arrival_rate=2.0,
            rmse=0.2,
        ),
        _candidate("hover", completion=0.0, completion_fraction=0.0, completion_time=0.0, rmse=0.01),
    ]

    selected = evaluation.select_checkpoint(results, min_completion_rate=0.0)

    assert selected["checkpoint_path"] == "nearly-done"


def test_checkpoint_selection_enforces_traversal_and_active_arc_rate_overall_and_per_family():
    evaluation = _load_evaluation_module()
    weak_traversal = _candidate(
        "weak-traversal",
        completion=0.0,
        completion_fraction=0.85,
        completion_time=0.0,
        arrival_rate=3.0,
        rmse=0.1,
    )
    slow = _candidate("slow", completion=1.0, completion_time=8.0, arrival_rate=1.5, rmse=0.1)
    fast = _candidate("fast", completion=1.0, completion_time=7.0, arrival_rate=2.5, rmse=0.1)

    assert (
        evaluation.select_checkpoint(
            [weak_traversal, slow, fast],
            min_completion_rate=0.0,
            min_traversal_fraction=0.90,
            min_active_arc_rate=2.0,
        )["checkpoint_path"]
        == "fast"
    )

    fast["route_families"]["figure_eight"]["route"]["active_arc_length_rate"] = 1.9
    with pytest.raises(ValueError, match="route-speed"):
        evaluation.select_checkpoint(
            [fast],
            min_completion_rate=0.0,
            min_traversal_fraction=0.90,
            min_active_arc_rate=2.0,
        )


def test_checkpoint_selection_defaults_require_performance_without_overfitting_tracking_limits():
    evaluation = _load_evaluation_module()
    hover = _candidate("hover", completion=0.0, completion_time=0.0, rmse=0.01)
    useful = _candidate(
        "useful",
        completion=0.6875,
        completion_time=13.0,
        rmse=0.5,
        cross_track_rms=0.59,
        cross_track_max=1.74,
    )

    selected = evaluation.select_checkpoint([hover, useful])

    assert selected["checkpoint_path"] == "useful"


def test_checkpoint_selection_requires_each_route_family_to_pass_completion_and_tracking_gates():
    evaluation = _load_evaluation_module()
    weak_eight = _candidate("weak-eight", completion=1.0, completion_time=5.0, rmse=0.1)
    weak_eight["route_families"]["figure_eight"]["route"]["completion_rate"] = 0.4
    weak_eight["route_families"]["figure_eight"]["route"]["route_completions_total"] = 2.0
    weak_eight["route_families"]["figure_eight"]["termination"]["success_terminations"] = 2.0
    weak_eight["route_families"]["figure_eight"]["tracking"]["cross_track_error_rms"] = 0.7
    balanced = _candidate("balanced", completion=0.8, completion_time=6.0, rmse=0.2)

    assert evaluation.select_checkpoint([weak_eight, balanced])["checkpoint_path"] == "balanced"


def test_checkpoint_selection_requires_strict_precision_hits_overall_and_per_family():
    evaluation = _load_evaluation_module()
    misses_overall = _candidate("misses-overall", completion=1.0, completion_time=4.0, rmse=0.1)
    misses_overall["precision"]["hit_fraction_of_passed"] = 0.75
    misses_eight = _candidate("misses-eight", completion=1.0, completion_time=4.0, rmse=0.1)
    misses_eight["route_families"]["figure_eight"]["precision"]["hit_fraction_of_passed"] = 0.75
    precise = _candidate("precise", completion=0.8, completion_time=5.0, rmse=0.2)

    assert evaluation.select_checkpoint([misses_overall, misses_eight, precise])["checkpoint_path"] == "precise"
    assert (
        evaluation.select_checkpoint([misses_eight], min_precision_hit_fraction=0.70)["checkpoint_path"]
        == "misses-eight"
    )


def test_checkpoint_selection_gates_corridor_exits_separately_from_physical_safety():
    evaluation = _load_evaluation_module()
    corridor_exit = _candidate("corridor", completion=1.0, completion_time=4.0, rmse=0.1)
    corridor_exit["termination"].update(path_corridor_exits=1, path_corridor_exit_rate=0.1)
    corridor_exit["route_families"]["figure_eight"]["termination"].update(
        path_corridor_exits=1, path_corridor_exit_rate=0.2
    )
    clean = _candidate("clean", completion=0.8, completion_time=5.0, rmse=0.2)

    assert corridor_exit["physics_safety"]["workspace_exits"] == 0
    assert evaluation.select_checkpoint([corridor_exit, clean])["checkpoint_path"] == "clean"
    assert evaluation.select_checkpoint([corridor_exit], max_corridor_exit_rate=0.2)["checkpoint_path"] == "corridor"


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "drone_crashes",
        "payload_crashes",
        "illegal_state_terminations",
        "workspace_exits",
        "cable_integrity_failures",
    ],
)
def test_checkpoint_selection_rejects_every_unsafe_category(unsafe_name):
    evaluation = _load_evaluation_module()
    unsafe = _candidate(
        "unsafe",
        completion=1.0,
        completion_time=1.0,
        rmse=0.01,
        safety=_safe_counts(**{unsafe_name: 1}),
    )
    safe = _candidate("safe", completion=1.0, completion_time=5.0, rmse=1.0)
    assert evaluation.select_checkpoint([unsafe, safe])["checkpoint_path"] == "safe"


def test_checkpoint_selection_fails_closed_on_missing_safety_or_excess_swing():
    evaluation = _load_evaluation_module()
    missing = _candidate("missing", completion=1.0, completion_time=1.0, rmse=0.1, safety={})
    high_swing = _candidate("swing", completion=1.0, completion_time=1.0, rmse=0.1, swing=0.5)
    high_rms_swing = _candidate(
        "rms-swing", completion=1.0, completion_time=1.0, rmse=0.1, rms_swing=math.radians(21.0)
    )
    high_peak = _candidate("peak", completion=1.0, completion_time=1.0, rmse=0.1, peak_swing=1.1)
    high_transverse = _candidate("transverse", completion=1.0, completion_time=1.0, rmse=0.1, transverse_speed=1.1)
    high_cross_track_rms = _candidate("cross-rms", completion=1.0, completion_time=1.0, rmse=0.1, cross_track_rms=0.61)
    high_cross_track_max = _candidate("cross-max", completion=1.0, completion_time=1.0, rmse=0.1, cross_track_max=1.76)
    detached = _candidate(
        "detached",
        completion=1.0,
        completion_time=1.0,
        rmse=0.1,
        safety=_safe_counts(cable_joint_error_max=0.01),
    )
    nonfinite = _candidate("nan", completion=1.0, completion_time=1.0, rmse=float("nan"))
    incomplete = _candidate("incomplete", completion=0.99, completion_time=1.0, rmse=0.1)
    missing_completion_time = _candidate("no-time", completion=1.0, completion_time=0.0, rmse=0.1)

    with pytest.raises(ValueError, match="zero-unsafe-event"):
        evaluation.select_checkpoint([missing])
    with pytest.raises(ValueError, match="zero-unsafe-event"):
        evaluation.select_checkpoint([high_swing])
    with pytest.raises(ValueError, match="zero-unsafe-event"):
        evaluation.select_checkpoint([high_rms_swing])
    with pytest.raises(ValueError, match="zero-unsafe-event"):
        evaluation.select_checkpoint([high_peak])
    with pytest.raises(ValueError, match="zero-unsafe-event"):
        evaluation.select_checkpoint([high_transverse])
    with pytest.raises(ValueError, match="zero-unsafe-event"):
        evaluation.select_checkpoint([high_cross_track_rms])
    with pytest.raises(ValueError, match="zero-unsafe-event"):
        evaluation.select_checkpoint([high_cross_track_max])
    with pytest.raises(ValueError, match="zero-unsafe-event"):
        evaluation.select_checkpoint([detached])
    with pytest.raises(ValueError, match="zero-unsafe-event"):
        evaluation.select_checkpoint([nonfinite])
    with pytest.raises(ValueError, match="zero-unsafe-event"):
        evaluation.select_checkpoint([incomplete], min_completion_rate=1.0)
    with pytest.raises(ValueError, match="zero-unsafe-event"):
        evaluation.select_checkpoint([missing_completion_time])


def test_episode_extraction_reads_autoreset_snapshot_and_terminal_terms():
    evaluation = _load_evaluation_module()
    command_term = SimpleNamespace(
        last_episode_metrics={
            "position_rmse": torch.tensor([0.25, 0.75]),
            "waypoint_completed": torch.tensor([1.0, 0.0]),
            "waypoint_completion_fraction": torch.tensor([1.0, 0.25]),
            "route_traversal_fraction": torch.tensor([1.0, 0.25]),
            "route_arc_length_traversed": torch.tensor([27.0, 6.0]),
            "waypoint_arrivals": torch.tensor([7.0, 2.0]),
            "waypoint_precision_hits": torch.tensor([7.0, 2.0]),
            "route_family_id": torch.tensor([1.0, 0.0]),
        },
        waypoints_e=torch.zeros(2, 24, 3),
    )
    terminal_terms = {
        "drone_crash": torch.tensor([False, False]),
        "payload_crash": torch.tensor([True, False]),
        "illegal_drone": torch.tensor([False, False]),
        "illegal_payload": torch.tensor([False, False]),
        "illegal_cable": torch.tensor([False, False]),
        "illegal_action": torch.tensor([False, False]),
        "cable_integrity": torch.tensor([True, False]),
        "drone_out_of_workspace": torch.tensor([False, False]),
        "payload_out_of_workspace": torch.tensor([False, False]),
        "path_corridor": torch.tensor([True, False]),
        "route_completed": torch.tensor([True, False]),
    }
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda name: command_term),
            termination_manager=SimpleNamespace(
                active_terms=list(terminal_terms), get_term=lambda name: terminal_terms[name]
            ),
        )
    )

    episode = evaluation._episode_from_env(env, env_id=0, episode_duration=4.0)

    assert episode["position_rmse"] == pytest.approx(0.25)
    assert episode["waypoint_completed"] == pytest.approx(1.0)
    assert episode["waypoint_count"] == pytest.approx(24.0)
    assert episode["waypoint_arrivals"] == pytest.approx(7.0)
    assert episode["route_arc_length_traversed"] == pytest.approx(27.0)
    assert episode["route_family_id"] == pytest.approx(1.0)
    assert episode["episode_duration"] == pytest.approx(4.0)
    assert episode["payload_crash"] == 1
    assert episode["workspace_exit"] == 0
    assert episode["path_corridor_exit"] == 1
    assert episode["success_termination"] == 1
    assert episode["drone_crash"] == 0
    assert episode["illegal_state"] == 0
    assert episode["cable_integrity_failure"] == 1


def test_evaluation_writes_raw_results_before_reporting_selection_failure(monkeypatch, tmp_path):
    evaluation = _load_evaluation_module()
    result = _candidate("model_10.pt", completion=0.0, completion_time=0.0, rmse=0.1)
    monkeypatch.setattr(evaluation, "_evaluate_runtime", lambda args, checkpoint: result)
    output_path = tmp_path / "evaluation.json"

    with pytest.raises(ValueError, match="Raw evaluation results were written"):
        evaluation.main(
            [
                "--task",
                "randomized-task",
                "--checkpoint",
                "model_10.pt",
                "--episodes",
                "10",
                "--output",
                str(output_path),
            ]
        )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["evaluation_suite"]["route_family_counts"] == {"ellipse": 5, "figure_eight": 5}
    assert payload["results"] == [result]
    assert payload["selected_checkpoint"] is None
    assert "No checkpoint satisfies" in payload["selection_error"]


def test_evaluation_preserves_training_horizon_and_seeded_swing_distribution():
    evaluation = _load_evaluation_module()
    from isaaclab_tasks.contrib.drone_slung_load.drone_slung_load_env_cfg import (
        DroneSlungLoadWaypointEnhancedEnvCfg,
    )

    cfg = DroneSlungLoadWaypointEnvCfg()
    training_waypoint_count = cfg.commands.route.random_waypoint_count

    evaluation.prepare_evaluation_env_cfg(cfg, num_envs=23)

    assert cfg.scene.num_envs == 23
    assert cfg.events.reset_slung_load.params["max_initial_swing"] == pytest.approx(0.10)
    assert cfg.episode_length_s == pytest.approx(12.0)
    assert cfg.commands.route.randomize_waypoints
    assert cfg.commands.route.random_waypoint_count == training_waypoint_count
    assert not cfg.commands.route.regenerate_on_completion
    assert not cfg.commands.route.debug_vis

    enhanced = DroneSlungLoadWaypointEnhancedEnvCfg()
    enhanced_horizon = enhanced.episode_length_s
    enhanced_count = enhanced.commands.route.random_waypoint_count
    assert enhanced_count == 24
    assert enhanced.commands.route.randomize_waypoints
    evaluation.prepare_evaluation_env_cfg(enhanced, num_envs=23)
    assert enhanced.commands.route.randomize_waypoints
    assert enhanced.commands.route.route_family == "bounded_template_mix"
    assert enhanced.commands.route.figure_eight_probability == pytest.approx(0.5)
    assert enhanced.commands.route.random_waypoint_count == 24
    assert enhanced.commands.route.samples_per_lap == 24
    assert not enhanced.commands.route.regenerate_on_completion
    assert enhanced.commands.route.acceptance_radius == pytest.approx(0.15)
    assert enhanced.commands.route.target_cruise_speed == pytest.approx(3.50)
    assert enhanced.rewards.path_progress.params["maximum_rate"] == pytest.approx(3.50)
    assert enhanced.rewards.path_progress.params["maximum_lateral_acceleration"] == pytest.approx(3.0)
    assert enhanced.rewards.path_precision.weight == pytest.approx(-4.0)
    assert enhanced.rewards.path_precision.params["cross_track_scale"] == pytest.approx(0.20)
    assert enhanced.terminations.path_corridor.params["maximum_distance"] == pytest.approx(0.75)
    assert enhanced.curriculum is None
    assert enhanced.episode_length_s == pytest.approx(enhanced_horizon)
