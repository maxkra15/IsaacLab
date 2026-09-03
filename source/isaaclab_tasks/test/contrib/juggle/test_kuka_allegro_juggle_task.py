# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public-contract tests for one-ball KUKA-Allegro juggling."""

import inspect
import math
from dataclasses import replace
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.managers import CurriculumTermCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.juggle import mdp
from isaaclab_tasks.contrib.juggle.config.kuka_allegro.agents.rsl_rl_ppo_cfg import (
    KukaAllegroJugglePPORunnerCfg,
)
from isaaclab_tasks.contrib.juggle.config.kuka_allegro.juggle_env_cfg import (
    _KukaAllegroJuggleBaseEnvCfg,
)
from isaaclab_tasks.contrib.juggle.mdp.actions import JuggleResetPreservingRelativeJointPositionAction
from isaaclab_tasks.contrib.juggle.mdp.reset import _interpolate_arm_anchor_with_yaw
from isaaclab_tasks.contrib.stack.mdp.kuka_allegro_reset import (
    kuka_allegro_tool_pose,
)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
from isaaclab_tasks.utils.reset_sampling import (
    AdaptiveResetSamplerCfg,
    ContinuousAdaptiveResetSamplerCfg,
    RollingOutcomeMonitorCfg,
)

from isaaclab_assets.robots import KUKA_ALLEGRO_CFG

TASK_NAME = "IsaacContrib-Juggle-Ball-KukaAllegro-RL"


def test_extended_manager_terms_preserve_existing_positional_parameters():
    """Juggle options append to rather than reorder existing manager-term calls."""
    runtime_parameters = tuple(inspect.signature(mdp.JuggleRuntimeState).parameters)
    assert runtime_parameters == (
        "row_ids",
        "start_phases",
        "current_phases",
        "visited_phase_bits",
        "stable_catch_steps",
        "release_clear_steps",
        "release_heights",
        "first_ascent_active",
        "seen_initial_ascent",
        "seen_release",
        "seen_apex",
        "static_held_start",
        "local_success",
        "new_local_success",
        "cycle_success",
        "new_cycle_success",
        "initialized",
    )
    initializer_parameters = tuple(inspect.signature(mdp.initialize_juggle_episode_state).parameters)
    assert initializer_parameters[:5] == (
        "state",
        "env_ids",
        "phases",
        "release_heights",
        "static_held_start",
    )
    assert initializer_parameters[-1] == "local_goal_ids"
    reset_parameters = tuple(inspect.signature(mdp.JuggleResetEvent.__call__).parameters)
    assert reset_parameters[:6] == (
        "self",
        "env",
        "env_ids",
        "rows_per_phase",
        "fixed_phase",
        "static_held_only",
    )
    progress_parameters = tuple(inspect.signature(mdp.JuggleProgressContext.__call__).parameters)
    assert progress_parameters[:14] == (
        "self",
        "env",
        "tool_body_cfg",
        "fingertip_cfg",
        "ball_cfg",
        "tool_offset",
        "release_separation_distance",
        "release_clear_steps",
        "apex_height_gain",
        "catch_approach_distance",
        "catch_distance",
        "contact_maximum_relative_speed",
        "stable_maximum_relative_speed",
        "stable_catch_steps",
    )


def test_single_juggle_task_is_continuous_one_metre_ppo():
    """The sole public Juggle task owns the complete repeated-cycle contract."""
    registered = sorted(task_id for task_id in gym.registry if task_id.startswith("IsaacContrib-Juggle-Ball-Kuka"))
    assert registered == [TASK_NAME]
    spec = gym.spec(TASK_NAME)
    assert spec.kwargs["env_cfg_entry_point"].endswith("meter_juggle_env_cfg:KukaAllegroJuggleRLEnvCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":KukaAllegroJugglePPORunnerCfg")

    cfg = parse_env_cfg(TASK_NAME, device="cpu", num_envs=8)
    cfg.validate_config()
    runner = load_cfg_from_registry(TASK_NAME, "rsl_rl_cfg_entry_point")

    assert cfg.scene.num_envs == 8
    assert isinstance(cfg.sim.physics, NewtonCfg)
    assert isinstance(cfg.sim.physics.solver_cfg, MJWarpSolverCfg)
    assert cfg.sim.dt == 1.0 / 120.0
    assert cfg.decimation == cfg.sim.render_interval == 2
    assert cfg.sim.physics.num_substeps == 4
    assert cfg.sim.physics.solver_cfg.ccd_iterations == 50
    assert cfg.scene.ball.spawn.radius == mdp.BALL_RADIUS
    assert cfg.scene.ball.spawn.mass_props.mass == mdp.BALL_MASS
    assert cfg.scene.ball.spawn.physics_material.restitution == 0.0
    assert cfg.scene.ball.spawn.physics_material.contact_stiffness == 5.0e4
    assert cfg.scene.ball.spawn.physics_material.contact_damping == 120.0
    assert cfg.events.reset_from_catalog.params["profile"] == mdp.METER_TOSS_RESET_PROFILE.name
    assert cfg.events.reset_from_catalog.params["sampling_mode"] == "continuous"
    assert cfg.events.reset_from_catalog.params["continuous_seed"] == 17
    assert cfg.events.reset_from_catalog.params["rows_per_phase"] == 128
    curriculum_params = cfg.curriculum.reset_sampling.params
    assert set(curriculum_params) == {
        "outcome_monitor",
        "adaptive_sampler",
        "continuous_sampler",
        "canonical_fraction",
        "sampling_mode",
    }
    assert curriculum_params["sampling_mode"] == "continuous"
    assert isinstance(curriculum_params["continuous_sampler"], ContinuousAdaptiveResetSamplerCfg)
    assert curriculum_params["continuous_sampler"].target_success_rate == 0.5
    assert curriculum_params["continuous_sampler"].coverage_fraction == pytest.approx(0.15 / 0.65)
    assert curriculum_params["canonical_fraction"] == 0.35
    assert (1.0 - curriculum_params["canonical_fraction"]) * curriculum_params[
        "continuous_sampler"
    ].coverage_fraction == pytest.approx(0.15)
    assert (1.0 - curriculum_params["canonical_fraction"]) * (
        1.0 - curriculum_params["continuous_sampler"].coverage_fraction
    ) == pytest.approx(0.50)
    assert cfg.terminations.progress_context.params["apex_height_gain"] == 1.0
    assert cfg.terminations.progress_context.params["apex_maximum_horizontal_displacement"] == 0.15
    assert cfg.terminations.progress_context.params["track_supported_release_reference"] is True
    assert cfg.terminations.progress_context.params["rearm_after_stable_catch"] is True
    assert cfg.terminations.success is None
    assert cfg.terminations.local_goal_success is None
    assert cfg.terminations.time_out is not None
    assert cfg.terminations.ball_out_of_workspace.params == {
        "workspace_lower": (0.20, -0.40, 0.08),
        "workspace_upper": (0.80, 0.40, 2.00),
    }
    assert cfg.scene.ground is None
    assert cfg.episode_length_s == 5.0
    assert isinstance(cfg.actions.arm_action, mdp.JuggleTaskSpaceTranslationActionCfg)
    assert cfg.actions.arm_action.scale == cfg.actions.arm_action.max_delta == 2.0
    assert cfg.actions.arm_action.body_name == "palm_link"
    assert cfg.actions.arm_action.tool_offset == mdp.JUGGLE_SPHERE_CENTER_OFFSET
    assert cfg.actions.arm_action.damping == 2.5e-3
    assert cfg.actions.hand_action.release_preload_after_first_action
    meter_effort = cfg.scene.robot.actuators["kuka_allegro_actuators"].effort_limit_sim
    assert isinstance(meter_effort, dict)
    assert meter_effort["iiwa7_joint_(1|2)"] == 352.0
    assert meter_effort["iiwa7_joint_(3|4|5)"] == 220.0
    assert meter_effort["iiwa7_joint_(6|7)"] == 80.0
    assert meter_effort["(index|middle|ring|thumb)_joint_(0|1|2|3)"] == 0.7
    assert KUKA_ALLEGRO_CFG.actuators["kuka_allegro_actuators"].effort_limit_sim["iiwa7_joint_(1|2)"] == 176.0
    assert cfg.observations.policy.ball_height_and_velocity.func is mdp.ball_height_above_release_hand_and_velocity
    assert cfg.observations.policy.ball_height_and_velocity.params["target_height_gain"] == 1.0
    assert cfg.observations.policy.actions.func is mdp.last_action
    assert cfg.rewards.physical_progress.func is mdp.JugglePhysicalProgressReward
    assert cfg.rewards.physical_progress.weight == 1.0
    assert cfg.rewards.physical_progress.params == {
        "gamma": runner.algorithm.gamma,
        "target_height_gain": 1.0,
        "apex_maximum_horizontal_displacement": 0.15,
        "catch_distance_scale": 0.12,
        "catch_relative_speed_scale": 0.45,
        "canonical_launch_fraction": 0.5,
    }
    assert cfg.rewards.local_transition is None
    assert cfg.rewards.apex_height.func is mdp.apex_height_pulse
    assert cfg.rewards.apex_height.weight == 1.0
    assert cfg.rewards.dropped_ball.func is mdp.ball_out_of_workspace_pulse
    assert cfg.rewards.dropped_ball.weight == -2.0
    maximum_discount_steps = math.ceil(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation)) - 1
    assert maximum_discount_steps == 299
    throw_then_drop_return = cfg.rewards.apex_height.weight + (
        runner.algorithm.gamma**maximum_discount_steps * cfg.rewards.dropped_ball.weight
    )
    assert throw_then_drop_return < 0.0
    assert isinstance(runner, KukaAllegroJugglePPORunnerCfg)
    assert runner.experiment_name == "kuka_allegro_juggle"
    assert runner.wandb_project == "kuka_allegro_juggle"
    assert runner.num_steps_per_env == 192
    assert runner.save_interval == 10
    assert runner.actor.distribution_cfg.init_std == 0.30
    assert runner.actor.distribution_cfg.hand_init_std == 0.40
    assert runner.actor.distribution_cfg.arm_action_dim == 3
    assert runner.actor.distribution_cfg.std_range == (0.002, 0.50)
    assert not runner.actor.obs_normalization
    assert not runner.critic.obs_normalization
    assert runner.algorithm.entropy_coef == 1.0e-3

    assert runner.actor.hidden_dims == [512, 256, 128]
    assert isinstance(cfg.actions.hand_action, mdp.JuggleHandSynergyActionCfg)
    assert runner.actor.distribution_cfg.arm_action_dim + 1 == 4


def test_standard_task_play_mode_keeps_only_failure_resets():
    """Playing the standard task converts it to an endless physical rally."""
    cfg = parse_env_cfg(TASK_NAME, device="cpu", num_envs=1)
    cfg.play_mode()
    assert cfg.curriculum is None
    assert cfg.events.reset_from_catalog.params["fixed_phase"] == int(mdp.JugglePhase.HELD_PRETHROW)
    assert cfg.events.reset_from_catalog.params["static_held_only"]
    assert cfg.terminations.progress_context.params["rearm_after_stable_catch"] is True
    assert cfg.terminations.local_goal_success is None
    assert cfg.terminations.success is None
    assert cfg.terminations.time_out is None
    assert cfg.terminations.ball_out_of_workspace is not None
    assert cfg.terminations.nonfinite_state is not None
    assert cfg.scene.ground is None
    assert cfg.scene.play_stage_deck.spawn.size == (1.60, 1.40, 0.08)
    assert cfg.scene.play_workspace_mat.spawn.size == (0.60, 0.80, 0.008)
    assert cfg.scene.play_robot_pedestal.spawn.size == (0.34, 0.34, 0.12)
    assert cfg.scene.play_stage_deck.spawn.display_color == (0.055, 0.070, 0.095)
    assert cfg.scene.play_workspace_mat.spawn.display_color == (0.035, 0.24, 0.36)
    assert cfg.scene.play_robot_pedestal.spawn.display_color == (0.20, 0.24, 0.30)
    for asset in (
        cfg.scene.play_stage_deck,
        cfg.scene.play_workspace_mat,
        cfg.scene.play_robot_pedestal,
    ):
        assert asset.spawn.rigid_props is None
        assert asset.spawn.collision_props is None
        assert asset.spawn.mass_props is None
        assert asset.spawn.physics_material is None
    assert cfg.sim.default_visualizer_cfg.eye == (1.75, 1.75, 1.35)
    assert cfg.sim.default_visualizer_cfg.lookat == (0.45, 0.0, 0.70)
    cfg.validate_config()


def test_juggle_config_validation_rejects_broken_contracts():
    """Config validation protects the compact physical Juggle contract."""
    base_cfg = _KukaAllegroJuggleBaseEnvCfg()
    ground_plane = base_cfg.scene.ground
    invalid_cases = (
        (lambda cfg: setattr(cfg.actions.arm_action, "scale", 0.60), "palm-center task-space"),
        (lambda cfg: setattr(cfg.actions.arm_action, "max_delta", 0.60), "palm-center task-space"),
        (lambda cfg: setattr(cfg.actions.arm_action, "body_name", "iiwa7_link_7"), "palm-center task-space"),
        (lambda cfg: setattr(cfg.actions.arm_action, "tool_offset", (0.0, 0.0, 0.0)), "palm-center task-space"),
        (lambda cfg: setattr(cfg.actions.arm_action, "damping", 0.01), "palm-center task-space"),
        (lambda cfg: setattr(cfg.scene, "ground", ground_plane), "ground plane"),
        (lambda cfg: setattr(cfg.rewards.physical_progress, "weight", 0.5), "physical-progress reward"),
        (
            lambda cfg: cfg.rewards.physical_progress.params.update({"gamma": 0.99}),
            "physical-progress reward",
        ),
        (lambda cfg: setattr(cfg.rewards.apex_height, "weight", 0.5), "apex-height reward"),
        (
            lambda cfg: setattr(cfg.rewards.apex_height, "func", mdp.full_cycle_pulse),
            "apex-height reward",
        ),
        (lambda cfg: setattr(cfg.rewards.dropped_ball, "weight", -1.0), "dropped-ball reward"),
        (
            lambda cfg: setattr(cfg.rewards.dropped_ball, "func", mdp.full_cycle_pulse),
            "dropped-ball reward",
        ),
        (
            lambda cfg: setattr(cfg.rewards, "local_transition", base_cfg.rewards.local_transition),
            "adaptive-reset label",
        ),
        (
            lambda cfg: setattr(
                cfg.terminations,
                "local_goal_success",
                base_cfg.terminations.local_goal_success,
            ),
            "phase-local success",
        ),
        (
            lambda cfg: cfg.curriculum.reset_sampling.params.update({"canonical_fraction": 0.0}),
            "35% canonical",
        ),
        (
            lambda cfg: setattr(
                cfg.curriculum.reset_sampling.params["continuous_sampler"],
                "coverage_fraction",
                0.15,
            ),
            "15% global uniform coverage",
        ),
        (
            lambda cfg: cfg.terminations.progress_context.params.update({"apex_maximum_horizontal_displacement": 0.30}),
            "launch-relative apex corridor",
        ),
        (
            lambda cfg: setattr(cfg.observations.policy.ball_height_and_velocity, "func", mdp.ball_height_and_velocity),
            "relative to the latched release hand",
        ),
        (
            lambda cfg: cfg.terminations.ball_out_of_workspace.params.update({"workspace_lower": (0.90, -0.35, 0.08)}),
            "workspace bounds",
        ),
        (
            lambda cfg: cfg.terminations.ball_out_of_workspace.params.update({"workspace_upper": (0.80, 0.40)}),
            "three-dimensional numeric workspace vector",
        ),
        (
            lambda cfg: cfg.terminations.ball_out_of_workspace.params.update({"workspace_upper": 2.0}),
            "three-dimensional numeric workspace vector",
        ),
        (
            lambda cfg: cfg.terminations.ball_out_of_workspace.params.update({"workspace_upper": (0.80, 0.40, "high")}),
            "three-dimensional numeric workspace vector",
        ),
        (
            lambda cfg: cfg.scene.robot.actuators["kuka_allegro_actuators"].effort_limit_sim.update(
                {"(index|middle|ring|thumb)_joint_(0|1|2|3)": 1.4}
            ),
            "Allegro effort",
        ),
    )
    for mutate, message in invalid_cases:
        cfg = parse_env_cfg(TASK_NAME, device="cpu", num_envs=8)
        mutate(cfg)
        with pytest.raises(ValueError, match=message):
            cfg.validate_config()


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"difficulty_band_count": 1.5}, "positive integer"),
        ({"minimum_apex_height_gain": -1.0}, "finite and positive"),
        ({"release_speed_range": (float("nan"), 1.4)}, "two finite"),
        (
            {"minimum_apex_height_gain": 1.0, "release_speed_range": (1.1, 1.4)},
            "cannot reach",
        ),
    ),
)
def test_reset_profile_rejects_invalid_physical_ranges(changes, message):
    """Reset profiles reject malformed or physically contradictory ranges."""
    with pytest.raises(ValueError, match=message):
        replace(mdp.LOW_TOSS_RESET_PROFILE, name="invalid_test_profile", profile_id=99, **changes)


def test_meter_reset_profile_is_a_small_randomized_physical_curriculum():
    """Meter PPO uses physical phase distributions without trajectory labels."""
    source = mdp.JuggleResetStateSource(rows_per_phase=64, profile=mdp.METER_TOSS_RESET_PROFILE, device="cpu")

    assert source.catalog.item_count == 9
    assert source.item_names == (
        "held_prethrow",
        "release",
        "ascending",
        "apex",
        "descending",
        "catch_approach",
        "catch_contact",
        "stable_catch",
        "moving_held_launch",
    )
    held_rows = source.phase_rows[int(mdp.JugglePhase.HELD_PRETHROW)]
    static_rows = held_rows[source.static_held_rows[held_rows]]
    moving_rows = held_rows[~source.static_held_rows[held_rows]]
    assert static_rows.numel() == moving_rows.numel() == 32
    assert source.canonical_start_rows[static_rows].all()
    assert not source.canonical_start_rows[moving_rows].any()
    assert (source.item_ids[static_rows] == int(mdp.JugglePhase.HELD_PRETHROW)).all()
    assert (source.item_ids[moving_rows] == len(mdp.JugglePhase)).all()
    torch.testing.assert_close(
        source.canonical_item_mask,
        torch.tensor((True, False, False, False, False, False, False, False, False)),
    )
    torch.testing.assert_close(
        source.adaptive_item_mask,
        torch.tensor((False, True, True, True, True, True, True, False, True)),
    )

    expected_goals = torch.tensor(
        (
            int(mdp.JuggleLocalGoal.FLIGHT_APEX),
            int(mdp.JuggleLocalGoal.STABLE_CATCH),
            int(mdp.JuggleLocalGoal.STABLE_CATCH),
            int(mdp.JuggleLocalGoal.STABLE_CATCH),
            int(mdp.JuggleLocalGoal.STABLE_CATCH),
            int(mdp.JuggleLocalGoal.STABLE_CATCH),
            int(mdp.JuggleLocalGoal.STABLE_CATCH),
            int(mdp.JuggleLocalGoal.FLIGHT_APEX),
        )
    )
    for phase in mdp.JugglePhase:
        phase_rows = source.phase_rows[int(phase)]
        assert (source.local_goal_ids[phase_rows] == expected_goals[int(phase)]).all()
        torch.testing.assert_close(torch.bincount(source.difficulty_band_ids[phase_rows]), torch.full((4,), 16))
    torch.testing.assert_close(source.catalog.metadata["local_goal"], source.local_goal_ids)
    assert not hasattr(source, "previous_actions")
    assert not hasattr(source, "local_goal_deadline_steps")


def test_meter_reset_catalog_is_deterministic_fk_consistent_and_workspace_randomized():
    """Repeated catalogs agree while arm yaw broadens every physical phase."""
    first = mdp.JuggleResetStateSource(rows_per_phase=64, profile=mdp.METER_TOSS_RESET_PROFILE, device="cpu")
    second = mdp.JuggleResetStateSource(rows_per_phase=64, profile=mdp.METER_TOSS_RESET_PROFILE, device="cpu")
    for name in (
        "arm_positions",
        "arm_velocities",
        "hand_positions",
        "ball_positions",
        "ball_velocities",
        "phase_ids",
        "item_ids",
        "local_goal_ids",
    ):
        torch.testing.assert_close(getattr(first, name), getattr(second, name))

    held_rows = first.phase_rows[int(mdp.JugglePhase.HELD_PRETHROW)]
    static_rows = held_rows[first.static_held_rows[held_rows]]
    moving_rows = held_rows[~first.static_held_rows[held_rows]]
    assert not first.arm_velocities[static_rows].any()
    assert not first.ball_velocities[static_rows].any()
    tool_position, _ = kuka_allegro_tool_pose(first.arm_positions[held_rows].double(), mdp.JUGGLE_SPHERE_CENTER_OFFSET)
    expected_velocity = mdp.kuka_allegro_tool_point_velocity(
        first.arm_positions[held_rows].double(), first.arm_velocities[held_rows].double()
    )
    torch.testing.assert_close(first.ball_positions[held_rows].double(), tool_position, atol=2.0e-6, rtol=0.0)
    torch.testing.assert_close(first.ball_velocities[held_rows, :3].double(), expected_velocity, atol=2.0e-5, rtol=0.0)
    assert (first.ball_velocities[moving_rows, 2] >= -1.0e-6).all()
    assert (first.ball_velocities[moving_rows, 2] <= 1.01).all()
    assert first.ball_velocities[moving_rows, 2].min() < 0.05
    assert first.ball_velocities[moving_rows, 2].max() > 0.95
    assert first.ball_positions[static_rows, 1].max() - first.ball_positions[static_rows, 1].min() > 0.08
    assert first.ball_positions[static_rows, 2].max() - first.ball_positions[static_rows, 2].min() > 0.05

    arm_lower = torch.tensor(mdp.KUKA_ALLEGRO_JUGGLE_ARM_WORKSPACE_LOWER)
    arm_upper = torch.tensor(mdp.KUKA_ALLEGRO_JUGGLE_ARM_WORKSPACE_UPPER)
    assert (first.arm_positions >= arm_lower).all()
    assert (first.arm_positions <= arm_upper).all()


def test_continuous_meter_reset_proposals_are_seeded_and_cover_the_full_parameter_box():
    """The third mode uses a reproducible Sobol proposal bank rather than competence bins."""
    first = mdp.JuggleResetStateSource(
        rows_per_phase=64,
        profile=mdp.METER_TOSS_RESET_PROFILE,
        device="cpu",
        parameter_sampling="continuous",
        continuous_seed=17,
    )
    repeated = mdp.JuggleResetStateSource(
        rows_per_phase=64,
        profile=mdp.METER_TOSS_RESET_PROFILE,
        device="cpu",
        parameter_sampling="continuous",
        continuous_seed=17,
    )
    different = mdp.JuggleResetStateSource(
        rows_per_phase=64,
        profile=mdp.METER_TOSS_RESET_PROFILE,
        device="cpu",
        parameter_sampling="continuous",
        continuous_seed=18,
    )

    assert first.parameter_sampling == "continuous"
    assert first.reset_parameters.shape == (first.row_count, mdp.JUGGLE_RESET_PARAMETER_DIM)
    assert first.model_features.shape == first.reset_parameters.shape
    assert first.reset_parameters.min() < 2.0e-4
    assert first.reset_parameters.max() > 0.999
    torch.testing.assert_close(first.reset_parameters, repeated.reset_parameters)
    torch.testing.assert_close(first.arm_positions, repeated.arm_positions)
    assert not torch.equal(first.reset_parameters, different.reset_parameters)
    for phase in mdp.JugglePhase:
        assert first.phase_rows[int(phase)].numel() == 64
    held_rows = first.phase_rows[int(mdp.JugglePhase.HELD_PRETHROW)]
    assert int(first.static_held_rows[held_rows].sum()) == 32
    assert first.catalog.metadata["reset_parameters"] is first.reset_parameters


def test_continuous_meter_held_starts_randomize_robot_and_ball_on_the_palm_manifold():
    """Meter held starts vary arm posture and ball placement while keeping FK consistency."""
    source = mdp.JuggleResetStateSource(
        rows_per_phase=128,
        profile=mdp.METER_TOSS_RESET_PROFILE,
        device="cpu",
        parameter_sampling="continuous",
        continuous_seed=17,
    )
    held_rows = source.phase_rows[int(mdp.JugglePhase.HELD_PRETHROW)]
    parameters = source.reset_parameters[held_rows]

    height = 0.28 + 0.28 * parameters[:, 0]
    workspace_yaw = 0.24 * (2.0 * parameters[:, 7] - 1.0)
    nominal = torch.tensor(
        [
            _interpolate_arm_anchor_with_yaw(float(row_height), float(row_yaw))
            for row_height, row_yaw in zip(height, workspace_yaw, strict=True)
        ]
    )
    perturbation = source.arm_positions[held_rows, 1:4] - nominal[:, 1:4]
    amplitudes = torch.tensor((0.12, 0.10, 0.10))
    assert (torch.abs(perturbation) <= amplitudes + 1.0e-6).all()
    assert (torch.max(perturbation, dim=0).values > 0.85 * amplitudes).all()
    assert (torch.min(perturbation, dim=0).values < -0.85 * amplitudes).all()

    center, palm_rotation = kuka_allegro_tool_pose(
        source.arm_positions[held_rows].double(), mdp.JUGGLE_SPHERE_CENTER_OFFSET
    )
    local_offset = torch.matmul(
        palm_rotation.transpose(-1, -2),
        (source.ball_positions[held_rows].double() - center).unsqueeze(-1),
    ).squeeze(-1)
    radial_offset = torch.linalg.vector_norm(local_offset[:, :2], dim=1)
    assert (radial_offset <= 0.005001).all()
    assert radial_offset.max() > 0.0044
    assert (torch.abs(local_offset[:, 2]) <= 0.002001).all()
    assert local_offset[:, 2].max() > 0.0016
    assert local_offset[:, 2].min() < -0.0016

    actual_tool_offsets = torch.tensor(mdp.JUGGLE_SPHERE_CENTER_OFFSET, dtype=torch.float64) + local_offset
    expected_velocity = torch.stack(
        [
            mdp.kuka_allegro_tool_point_velocity(
                arm_position,
                arm_velocity,
                tuple(float(value) for value in tool_offset),
            )
            for arm_position, arm_velocity, tool_offset in zip(
                source.arm_positions[held_rows].double(),
                source.arm_velocities[held_rows].double(),
                actual_tool_offsets,
                strict=True,
            )
        ]
    )
    torch.testing.assert_close(
        source.ball_velocities[held_rows, :3].double(), expected_velocity, atol=2.0e-7, rtol=5.0e-6
    )

    arm_lower = torch.tensor(mdp.KUKA_ALLEGRO_JUGGLE_ARM_WORKSPACE_LOWER)
    arm_upper = torch.tensor(mdp.KUKA_ALLEGRO_JUGGLE_ARM_WORKSPACE_UPPER)
    assert (source.arm_positions[held_rows] >= arm_lower).all()
    assert (source.arm_positions[held_rows] <= arm_upper).all()
    static_rows = held_rows[source.static_held_rows[held_rows]]
    static_span = source.ball_positions[static_rows].amax(dim=0) - source.ball_positions[static_rows].amin(dim=0)
    assert static_span[0] > 0.07
    assert static_span[1] > 0.27
    assert static_span[2] > 0.34


def test_explicit_catalog_mode_preserves_the_default_reset_source_exactly():
    """Adding continuous proposals does not perturb either existing sampler mode."""
    default = mdp.JuggleResetStateSource(rows_per_phase=16, device="cpu")
    explicit = mdp.JuggleResetStateSource(rows_per_phase=16, device="cpu", parameter_sampling="catalog")

    for name in (
        "reset_parameters",
        "arm_positions",
        "arm_velocities",
        "hand_positions",
        "ball_positions",
        "ball_velocities",
        "phase_ids",
        "item_ids",
        "local_goal_ids",
    ):
        torch.testing.assert_close(getattr(default, name), getattr(explicit, name))


def test_meter_launch_and_catch_rows_match_the_live_screened_control_basin():
    """Meter reset speeds, gaps, and aperture stay inside the reachable bracket."""
    source = mdp.JuggleResetStateSource(rows_per_phase=64, profile=mdp.METER_TOSS_RESET_PROFILE, device="cpu")
    assert source.profile.moving_held_speed_range == (0.0, 1.0)
    assert source.profile.catch_approach_tool_speed_range == (0.0, 1.0)
    assert source.profile.catch_approach_relative_speed_range == (0.05, 0.15)
    assert source.profile.catch_contact_tool_speed_range == (0.0, 1.0)
    assert source.profile.catch_contact_relative_speed_range == (0.05, 0.15)
    assert source.profile.catch_hand_open_fraction == 0.50

    preload = torch.tensor(mdp.JUGGLE_SPHERE_PRELOAD_HAND_POSITION)
    opened = torch.tensor(mdp.JUGGLE_SPHERE_FLIGHT_GATE_HAND_POSITION)
    expected_hand = preload + 0.50 * (opened - preload)
    expected_gap_ranges = {
        mdp.JugglePhase.CATCH_APPROACH: (0.050, 0.065),
        mdp.JugglePhase.CATCH_CONTACT: (0.045, 0.055),
    }
    for phase, (minimum_gap, maximum_gap) in expected_gap_ranges.items():
        rows = source.phase_rows[int(phase)]
        tool_position, _ = kuka_allegro_tool_pose(source.arm_positions[rows], mdp.JUGGLE_SPHERE_CENTER_OFFSET)
        tool_velocity = mdp.kuka_allegro_tool_point_velocity(
            source.arm_positions[rows].double(), source.arm_velocities[rows].double()
        ).float()
        offset = source.ball_positions[rows] - tool_position
        relative_fall_speed = tool_velocity[:, 2] - source.ball_velocities[rows, 2]

        assert (offset[:, 2] >= minimum_gap - 1.0e-6).all()
        assert (offset[:, 2] <= maximum_gap + 1.0e-6).all()
        assert (torch.abs(offset[:, 0]) <= 0.0051).all()
        assert (torch.abs(offset[:, 1]) <= 0.0051).all()
        assert (tool_velocity[:, 2] <= 1.0e-5).all()
        assert (tool_velocity[:, 2] >= -1.01).all()
        assert (relative_fall_speed >= 0.049).all()
        assert (relative_fall_speed <= 0.151).all()
        torch.testing.assert_close(source.hand_positions[rows], expected_hand.expand(len(rows), -1))


def test_every_reset_phase_is_present_and_physical_tensors_are_finite():
    """The physical catalog is phase-balanced and contains no invalid state."""
    source = mdp.JuggleResetStateSource(rows_per_phase=16, device="cpu")

    assert source.row_count == 16 * len(mdp.JugglePhase)
    assert source.catalog.item_count == len(mdp.JugglePhase)
    torch.testing.assert_close(
        torch.bincount(source.phase_ids),
        torch.full((len(mdp.JugglePhase),), 16, dtype=torch.long),
    )
    for phase in mdp.JugglePhase:
        assert source.phase_rows[int(phase)].numel() == 16
        phase_rows = source.phase_rows[int(phase)]
        physical_rows = torch.cat(
            (
                source.arm_positions[phase_rows],
                source.arm_velocities[phase_rows],
                source.hand_positions[phase_rows],
                source.ball_positions[phase_rows],
                source.ball_velocities[phase_rows],
            ),
            dim=1,
        )
        assert torch.unique(physical_rows, dim=0).shape[0] == 16
    held_rows = source.phase_rows[int(mdp.JugglePhase.HELD_PRETHROW)]
    torch.testing.assert_close(source.item_ids, source.phase_ids)
    torch.testing.assert_close(source.canonical_start_rows, source.phase_ids == int(mdp.JugglePhase.HELD_PRETHROW))
    assert source.item_names == tuple(phase.name.lower() for phase in mdp.JugglePhase)
    torch.testing.assert_close(
        source.local_goal_ids,
        torch.tensor([int(mdp.local_goal_for_phase(phase)) for phase in mdp.JugglePhase]).repeat_interleave(16),
    )
    torch.testing.assert_close(source.adaptive_item_mask, ~source.canonical_item_mask)
    assert int(source.static_held_rows[held_rows].sum()) == len(held_rows) // 2
    assert source.preload_assist_rows[held_rows].all()
    stable_rows = source.phase_rows[int(mdp.JugglePhase.STABLE_CATCH)]
    assert source.preload_assist_rows[stable_rows].all()
    assert not source.static_held_rows[~torch.isin(torch.arange(source.row_count), held_rows)].any()
    for values in (
        source.arm_positions,
        source.arm_velocities,
        source.hand_positions,
        source.hand_velocities,
        source.ball_positions,
        source.ball_quaternions,
        source.ball_velocities,
    ):
        assert torch.isfinite(values).all()


def test_free_flight_rows_obey_the_ballistic_invariant():
    """Every authored flight row comes from one analytic release state."""
    source = mdp.JuggleResetStateSource(rows_per_phase=32, device="cpu")
    mask = source.ballistic_rows

    expected_position, expected_velocity = mdp.ballistic_state(
        source.release_positions[mask],
        source.release_velocities[mask],
        source.flight_times[mask],
    )

    torch.testing.assert_close(source.ball_positions[mask], expected_position)
    torch.testing.assert_close(source.ball_velocities[mask, :3], expected_velocity)
    assert (source.ball_positions[source.phase_rows[int(mdp.JugglePhase.APEX)], 2] > 0.41).all()
    apex_rows = source.phase_rows[int(mdp.JugglePhase.APEX)]
    assert (source.ball_positions[apex_rows, 2] - source.release_positions[apex_rows, 2] > 0.06).all()
    lateral_speed = torch.linalg.vector_norm(source.release_velocities[:, :2], dim=1)
    assert (lateral_speed >= 0.189).all()
    assert (lateral_speed <= 0.341).all()


def test_attached_reset_rows_use_one_fk_for_ball_position_and_velocity():
    """Physically attached rows contain no kinematic placement or velocity slip."""
    source = mdp.JuggleResetStateSource(rows_per_phase=16, device="cpu")
    attached = torch.zeros(source.row_count, dtype=torch.bool)
    for phase in (
        mdp.JugglePhase.HELD_PRETHROW,
        mdp.JugglePhase.STABLE_CATCH,
    ):
        attached[source.phase_rows[int(phase)]] = True

    arm_position = source.arm_positions[attached].double()
    arm_velocity = source.arm_velocities[attached].double()
    expected_position, _ = kuka_allegro_tool_pose(arm_position, mdp.JUGGLE_SPHERE_CENTER_OFFSET)
    expected_velocity = mdp.kuka_allegro_tool_point_velocity(arm_position, arm_velocity)

    torch.testing.assert_close(source.ball_positions[attached].double(), expected_position, atol=2.0e-6, rtol=0.0)
    torch.testing.assert_close(source.ball_velocities[attached, :3].double(), expected_velocity, atol=2.0e-5, rtol=0.0)


def test_flight_and_capture_rows_have_valid_staging_geometry():
    """Palm-up flight rows clear the gate and catch rows follow the intercept envelope."""
    source = mdp.JuggleResetStateSource(rows_per_phase=32, device="cpu")
    # The lateral corridor grows with ballistic flight time.  These bounds
    # preserve realistic 0.19--0.34 m/s launch drift while keeping every row
    # inside the empirically reachable palm-up interception envelope.
    lateral_bounds = {
        mdp.JugglePhase.RELEASE: (0.006, 0.006),
        mdp.JugglePhase.ASCENDING: (0.035, 0.020),
        mdp.JugglePhase.APEX: (0.055, 0.025),
        mdp.JugglePhase.DESCENDING: (0.075, 0.035),
    }
    for phase, (x_bound, y_bound) in lateral_bounds.items():
        rows = source.phase_rows[int(phase)]
        tool_position, _ = kuka_allegro_tool_pose(source.arm_positions[rows], mdp.JUGGLE_SPHERE_CENTER_OFFSET)
        clearance = source.ball_positions[rows] - tool_position
        assert (torch.abs(clearance[:, 0]) < x_bound).all()
        assert (torch.abs(clearance[:, 1]) < y_bound).all()
        assert (clearance[:, 2] > 0.035).all()
        torch.testing.assert_close(
            source.hand_positions[rows],
            source.hand_positions.new_tensor(mdp.JUGGLE_SPHERE_FLIGHT_GATE_HAND_POSITION).expand(len(rows), -1),
        )
    descending_rows = source.phase_rows[int(mdp.JugglePhase.DESCENDING)]
    descending_tool, _ = kuka_allegro_tool_pose(source.arm_positions[descending_rows], mdp.JUGGLE_SPHERE_CENTER_OFFSET)
    assert (torch.linalg.vector_norm(source.ball_positions[descending_rows] - descending_tool, dim=1) > 0.12).all()

    rows = source.phase_rows[int(mdp.JugglePhase.CATCH_APPROACH)]
    tool_position, _ = kuka_allegro_tool_pose(source.arm_positions[rows], mdp.JUGGLE_SPHERE_CENTER_OFFSET)
    approach_offset = source.ball_positions[rows] - tool_position
    assert (torch.abs(approach_offset[:, 0]) <= 0.006).all()
    assert (torch.abs(approach_offset[:, 1]) <= 0.006).all()
    assert (approach_offset[:, 2] >= 0.079).all()
    assert (approach_offset[:, 2] <= 0.121).all()
    assert (source.ball_velocities[rows, 2] >= -0.451).all()
    assert (source.ball_velocities[rows, 2] <= -0.149).all()
    torch.testing.assert_close(
        source.hand_positions[rows],
        source.hand_positions.new_tensor(mdp.JUGGLE_SPHERE_CONTACT_HAND_POSITION).expand(len(rows), -1),
    )
    assert not source.ballistic_rows[rows].any()

    rows = source.phase_rows[int(mdp.JugglePhase.CATCH_CONTACT)]
    tool_position, _ = kuka_allegro_tool_pose(source.arm_positions[rows], mdp.JUGGLE_SPHERE_CENTER_OFFSET)
    contact_offset = source.ball_positions[rows] - tool_position
    assert (torch.abs(contact_offset[:, 0]) <= 0.005).all()
    assert (torch.abs(contact_offset[:, 1]) <= 0.005).all()
    assert (contact_offset[:, 2] >= 0.039).all()
    assert (contact_offset[:, 2] <= 0.076).all()
    assert (source.ball_velocities[rows, 2] >= -0.351).all()
    assert (source.ball_velocities[rows, 2] <= -0.099).all()
    torch.testing.assert_close(
        source.hand_positions[rows],
        source.hand_positions.new_tensor(mdp.JUGGLE_SPHERE_CONTACT_HAND_POSITION).expand(len(rows), -1),
    )


def test_sphere_support_proxy_accepts_opposed_pairs_and_rejects_same_side_contacts():
    """The catch proxy is finger-agnostic but requires geometric support."""
    relative_positions = torch.tensor(
        (
            ((0.050, 0.000, -0.010), (-0.050, 0.000, 0.020), (0.12, 0.0, 0.0), (0.13, 0.0, 0.0)),
            ((0.040, 0.000, -0.010), (0.050, 0.005, -0.010), (0.12, 0.0, 0.0), (0.13, 0.0, 0.0)),
            ((0.090, 0.000, -0.010), (-0.090, 0.000, -0.010), (0.12, 0.0, 0.0), (0.13, 0.0, 0.0)),
        ),
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        mdp.sphere_support_from_fingertips(relative_positions),
        torch.tensor((True, False, False)),
    )


def test_apex_crossing_latch_rejects_a_later_bounce():
    """Only the first ascent-to-descent crossing can satisfy the apex event."""
    torch.testing.assert_close(
        mdp.first_ascent_apex_crossing(
            torch.tensor((True, False, True)),
            torch.tensor((True, True, False)),
            torch.tensor((-0.01, -0.50, -0.10)),
        ),
        torch.tensor((True, False, False)),
    )


def test_held_release_latches_last_supported_hand_height():
    """Pre-release lifting and post-release hand lowering cannot reduce the height target."""
    env, context = _make_progress_harness(2)
    state = env.juggle_runtime_state
    ids = torch.arange(2)
    mdp.initialize_juggle_episode_state(
        state,
        ids,
        torch.full((2,), int(mdp.JugglePhase.HELD_PRETHROW)),
        torch.tensor((0.20, 0.90)),
        torch.ones(2, dtype=torch.bool),
    )

    _set_progress_geometry(env, palm_height=0.60, ball_height=0.60, ball_vertical_velocity=0.0, supported=True)
    _step_progress(context, env, apex_height_gain=1.0, track_supported_release_reference=True)
    torch.testing.assert_close(state.release_heights, torch.full((2,), 0.60))
    torch.testing.assert_close(state.release_origins_xy, torch.tensor(((0.50, 0.0), (0.50, 0.0))))

    # Both hands move down after support is lost. The first ball's 1.465 m
    # predicted apex would pass against its authored 0.20 m reset reference,
    # but correctly fails against the latched 0.60 m hand reference. The
    # second ball has enough speed to pass the same invariant target.
    _set_progress_geometry(
        env,
        palm_height=0.20,
        ball_height=0.65,
        ball_vertical_velocity=torch.tensor((4.0, 4.5)),
        supported=False,
    )
    _step_progress(context, env, apex_height_gain=1.0, track_supported_release_reference=True)

    torch.testing.assert_close(state.release_heights, torch.full((2,), 0.60))
    assert state.current_phases.tolist() == [int(mdp.JugglePhase.HELD_PRETHROW), int(mdp.JugglePhase.RELEASE)]
    assert state.seen_release.tolist() == [False, True]


def test_short_toss_keeps_its_authored_release_reference():
    """The original task does not inherit the meter-only moving hand reference."""
    env, context = _make_progress_harness(1)
    state = env.juggle_runtime_state
    mdp.initialize_juggle_episode_state(
        state,
        torch.tensor((0,)),
        torch.tensor((int(mdp.JugglePhase.HELD_PRETHROW),)),
        torch.tensor((0.20,)),
        torch.ones(1, dtype=torch.bool),
    )
    _set_progress_geometry(env, palm_height=0.60, ball_height=0.60, ball_vertical_velocity=0.0, supported=True)

    _step_progress(context, env, apex_height_gain=0.06)

    torch.testing.assert_close(state.release_heights, torch.tensor((0.20,)))


def test_noncanonical_reference_is_preserved_and_authored_apex_is_not_rewarded():
    """Phase resets retain authored references and cannot grant a reset-time height pulse."""
    env, context = _make_progress_harness(2)
    state = env.juggle_runtime_state
    mdp.initialize_juggle_episode_state(
        state,
        torch.arange(2),
        torch.tensor((int(mdp.JugglePhase.RELEASE), int(mdp.JugglePhase.APEX))),
        torch.tensor((0.37, 0.42)),
        torch.zeros(2, dtype=torch.bool),
    )
    _set_progress_geometry(
        env,
        palm_height=torch.tensor((0.20, 0.30)),
        ball_height=torch.tensor((0.50, 1.50)),
        ball_vertical_velocity=torch.tensor((1.0, 0.0)),
        supported=True,
    )

    _step_progress(context, env, apex_height_gain=1.0)

    torch.testing.assert_close(state.release_heights, torch.tensor((0.37, 0.42)))
    assert not state.height_success.any()
    assert not state.new_height_success.any()
    assert not mdp.apex_height_pulse(env).any()


def test_apex_height_pulse_is_one_shot_and_local_pulse_excludes_it():
    """A valid first apex emits one height impulse without duplicating the local reward."""
    env, context = _make_progress_harness(1)
    state = env.juggle_runtime_state
    mdp.initialize_juggle_episode_state(
        state,
        torch.tensor((0,)),
        torch.tensor((int(mdp.JugglePhase.ASCENDING),)),
        torch.tensor((0.40,)),
        torch.zeros(1, dtype=torch.bool),
    )
    _set_progress_geometry(env, palm_height=0.40, ball_height=1.39, ball_vertical_velocity=0.10, supported=False)
    _step_progress(context, env, apex_height_gain=1.0)
    _set_progress_geometry(env, palm_height=0.20, ball_height=1.41, ball_vertical_velocity=0.0, supported=False)
    _step_progress(context, env, apex_height_gain=1.0)

    torch.testing.assert_close(mdp.apex_height_pulse(env), torch.tensor((60.0,)))
    torch.testing.assert_close(mdp.local_transition_pulse(env), torch.tensor((60.0,)))
    torch.testing.assert_close(mdp.non_height_local_transition_pulse(env), torch.tensor((0.0,)))
    assert state.height_success.item()

    _step_progress(context, env, apex_height_gain=1.0)
    torch.testing.assert_close(mdp.apex_height_pulse(env), torch.tensor((0.0,)))
    assert state.height_success.item()

    state.new_local_success[:] = True
    state.new_height_success[:] = False
    torch.testing.assert_close(mdp.non_height_local_transition_pulse(env), torch.tensor((60.0,)))


def test_meter_flight_reset_gets_no_passive_apex_reward_or_local_success():
    """A meter flight reset must still execute a fresh stable catch."""
    env, context = _make_progress_harness(2)
    state = env.juggle_runtime_state
    mdp.initialize_juggle_episode_state(
        state,
        torch.arange(2),
        torch.full((2,), int(mdp.JugglePhase.ASCENDING)),
        torch.full((2,), 0.40),
        torch.zeros(2, dtype=torch.bool),
        local_goal_ids=torch.tensor((int(mdp.JuggleLocalGoal.FLIGHT_APEX), int(mdp.JuggleLocalGoal.STABLE_CATCH))),
    )
    _set_progress_geometry(env, palm_height=0.40, ball_height=1.39, ball_vertical_velocity=0.10, supported=False)
    _step_progress(context, env, apex_height_gain=1.0)
    _set_progress_geometry(env, palm_height=0.20, ball_height=1.41, ball_vertical_velocity=0.0, supported=False)
    _step_progress(context, env, apex_height_gain=1.0)

    assert state.height_success.all()
    assert state.local_success.tolist() == [True, False]
    torch.testing.assert_close(mdp.apex_height_pulse(env), torch.tensor((60.0, 0.0)))
    assert not mdp.non_height_local_transition_pulse(env).any()


def test_meter_catch_goal_requires_a_fresh_stable_retention_dwell():
    """Authored contact is not success until the ball is retained for 15 steps."""
    env, context = _make_progress_harness(1)
    state = env.juggle_runtime_state
    mdp.initialize_juggle_episode_state(
        state,
        torch.tensor((0,)),
        torch.tensor((int(mdp.JugglePhase.CATCH_CONTACT),)),
        torch.tensor((0.40,)),
        torch.zeros(1, dtype=torch.bool),
        local_goal_ids=torch.tensor((int(mdp.JuggleLocalGoal.STABLE_CATCH),)),
    )
    _set_progress_geometry(env, palm_height=0.40, ball_height=0.40, ball_vertical_velocity=0.0, supported=True)

    for _ in range(14):
        _step_progress(context, env, apex_height_gain=1.0)
        assert not state.local_success.item()
    _step_progress(context, env, apex_height_gain=1.0)

    assert state.local_success.item()
    assert state.new_local_success.item()
    assert state.current_phases.item() == int(mdp.JugglePhase.STABLE_CATCH)
    torch.testing.assert_close(mdp.non_height_local_transition_pulse(env), torch.tensor((60.0,)))


def test_continuous_play_rearms_a_completed_catch_without_a_physical_reset():
    """A stable catch becomes a fresh held phase while preserving its success pulse."""
    env, context = _make_progress_harness(1)
    state = env.juggle_runtime_state
    mdp.initialize_juggle_episode_state(
        state,
        torch.tensor((0,)),
        torch.tensor((int(mdp.JugglePhase.HELD_PRETHROW),)),
        torch.tensor((0.40,)),
        torch.ones(1, dtype=torch.bool),
    )
    state.current_phases[:] = int(mdp.JugglePhase.CATCH_CONTACT)
    state.visited_phase_bits[:] = sum(
        1 << int(phase)
        for phase in mdp.JugglePhase
        if phase not in (mdp.JugglePhase.HELD_PRETHROW, mdp.JugglePhase.STABLE_CATCH)
    )
    state.stable_catch_steps[:] = 14
    state.release_clear_steps[:] = 3
    state.first_ascent_active[:] = False
    state.seen_initial_ascent[:] = True
    state.seen_release[:] = True
    state.seen_apex[:] = True
    state.height_success[:] = True
    state.local_success[:] = True
    _set_progress_geometry(env, palm_height=0.40, ball_height=0.40, ball_vertical_velocity=0.0, supported=True)

    _step_progress(context, env, apex_height_gain=1.0, rearm_after_stable_catch=True)

    assert env.extras["successes"].item()
    assert state.new_cycle_success.item()
    assert state.current_phases.item() == int(mdp.JugglePhase.HELD_PRETHROW)
    assert state.visited_phase_bits.item() == 1 << int(mdp.JugglePhase.HELD_PRETHROW)
    assert state.first_ascent_active.item()
    assert state.local_success.item()
    assert state.cycle_success.item()
    for tensor in (
        state.stable_catch_steps,
        state.release_clear_steps,
        state.seen_initial_ascent,
        state.seen_release,
        state.seen_apex,
        state.height_success,
    ):
        assert not tensor.any()

    _step_progress(context, env, apex_height_gain=1.0, rearm_after_stable_catch=True)
    assert not state.new_cycle_success.item()
    assert state.current_phases.item() == int(mdp.JugglePhase.HELD_PRETHROW)

    # A second physical cycle emits another pulse while the episode-level
    # success latches remain true for curriculum credit at the eventual reset.
    state.current_phases[:] = int(mdp.JugglePhase.CATCH_CONTACT)
    state.visited_phase_bits[:] = sum(
        1 << int(phase)
        for phase in mdp.JugglePhase
        if phase not in (mdp.JugglePhase.HELD_PRETHROW, mdp.JugglePhase.STABLE_CATCH)
    )
    state.stable_catch_steps[:] = 14
    state.seen_release[:] = True
    state.seen_apex[:] = True
    state.height_success[:] = True
    _step_progress(context, env, apex_height_gain=1.0, rearm_after_stable_catch=True)

    assert state.new_cycle_success.item()
    assert state.cycle_success.item()
    assert state.local_success.item()
    torch.testing.assert_close(mdp.full_cycle_pulse(env), torch.tensor((60.0,)))


def test_meter_apex_corridor_rejects_an_uncatchable_sideways_throw():
    """The one-metre pulse requires a first apex inside its launch-relative XY corridor."""
    env, context = _make_progress_harness(2)
    state = env.juggle_runtime_state
    mdp.initialize_juggle_episode_state(
        state,
        torch.arange(2),
        torch.full((2,), int(mdp.JugglePhase.ASCENDING)),
        torch.full((2,), 0.40),
        torch.zeros(2, dtype=torch.bool),
        release_origins_xy=torch.tensor(((0.50, 0.0), (0.50, 0.0))),
    )
    _set_progress_geometry(
        env,
        palm_height=0.40,
        ball_height=1.39,
        ball_vertical_velocity=0.10,
        supported=False,
        ball_x=torch.tensor((0.61, 0.63)),
    )
    _step_progress(context, env, apex_height_gain=1.0, apex_maximum_horizontal_displacement=0.12)
    _set_progress_geometry(
        env,
        palm_height=0.40,
        ball_height=1.41,
        ball_vertical_velocity=0.0,
        supported=False,
        ball_x=torch.tensor((0.61, 0.63)),
    )
    _step_progress(context, env, apex_height_gain=1.0, apex_maximum_horizontal_displacement=0.12)

    assert state.height_success.tolist() == [True, False]
    assert state.local_success.tolist() == [True, False]
    with pytest.raises(ValueError, match="horizontal-displacement"):
        _step_progress(context, env, apex_height_gain=1.0, apex_maximum_horizontal_displacement=0.0)


def test_release_relative_height_observation_preserves_width_and_normalizes_height():
    """The Markov release reference replaces absolute height without changing observation width."""
    env, _ = _make_progress_harness(2)
    env.juggle_runtime_state.release_heights[:] = torch.tensor((0.40, 0.70))
    env.scene["ball"].data.root_pos_w.torch[:, 2] = torch.tensor((1.40, 1.20))
    env.scene["ball"].data.root_lin_vel_w.torch[:] = torch.tensor(((0.1, 0.2, 0.3), (-0.1, -0.2, -0.3)))

    observation = mdp.ball_height_above_release_hand_and_velocity(env, target_height_gain=0.50)

    assert observation.shape == (2, 4)
    torch.testing.assert_close(observation[:, 0], torch.tensor((2.0, 1.0)))
    torch.testing.assert_close(observation[:, 1:], env.scene["ball"].data.root_lin_vel_w.torch)
    with pytest.raises(ValueError, match="target_height_gain"):
        mdp.ball_height_above_release_hand_and_velocity(env, target_height_gain=0.0)


def test_ball_out_of_workspace_pulse_is_unit_integral_and_validates_term_name():
    """The terminal drop signal contributes exactly one configured reward impulse."""
    env, _ = _make_progress_harness(2)
    env.termination_manager = _FakeTerminationManager({"ball_out_of_workspace": torch.tensor((True, False))})

    torch.testing.assert_close(mdp.ball_out_of_workspace_pulse(env), torch.tensor((60.0, 0.0)))
    with pytest.raises(ValueError, match="Unknown termination term"):
        mdp.ball_out_of_workspace_pulse(env, termination_term_name="missing")
    with pytest.raises(ValueError, match="must not be empty"):
        mdp.ball_out_of_workspace_pulse(env, termination_term_name="")


def test_reset_initialization_never_counts_the_authored_state_as_success():
    """Even contact/stable rows begin with uncredited fresh-event latches."""
    env = SimpleNamespace(num_envs=4, device="cpu", episode_length_buf=torch.zeros(4, dtype=torch.long))
    state = mdp.create_juggle_runtime_state(env)
    assert state.preload_assist_start.all()
    env_ids = torch.arange(4)
    phases = torch.tensor(
        (
            int(mdp.JugglePhase.HELD_PRETHROW),
            int(mdp.JugglePhase.RELEASE),
            int(mdp.JugglePhase.CATCH_CONTACT),
            int(mdp.JugglePhase.STABLE_CATCH),
        )
    )

    static_held = torch.tensor((True, False, False, False))
    mdp.initialize_juggle_episode_state(state, env_ids, phases, torch.full((4,), 0.35), static_held)

    assert not state.local_success.any()
    assert not state.new_local_success.any()
    assert not state.height_success.any()
    assert not state.new_height_success.any()
    assert not state.cycle_success.any()
    assert not state.new_cycle_success.any()
    assert not state.seen_apex.any()
    assert (state.local_goal_ids == -1).all()
    assert state.stable_catch_steps.count_nonzero() == 0

    assert state.static_held_start.tolist() == static_held.tolist()
    assert state.start_item_ids.tolist() == phases.tolist()
    assert state.canonical_start.tolist() == [True, False, False, False]
    assert state.preload_assist_start.tolist() == [True, False, False, True]
    assert mdp.local_goal_for_phase(mdp.JugglePhase.HELD_PRETHROW) is mdp.JuggleLocalGoal.FLIGHT_APEX
    assert mdp.local_goal_for_phase(mdp.JugglePhase.STABLE_CATCH) is mdp.JuggleLocalGoal.FLIGHT_APEX
    assert mdp.local_goal_for_phase(mdp.JugglePhase.RELEASE) is mdp.JuggleLocalGoal.FLIGHT_APEX
    assert mdp.local_goal_for_phase(mdp.JugglePhase.ASCENDING) is mdp.JuggleLocalGoal.FLIGHT_APEX
    for approach_phase in (mdp.JugglePhase.APEX, mdp.JugglePhase.DESCENDING):
        assert mdp.local_goal_for_phase(approach_phase) is mdp.JuggleLocalGoal.CATCH_APPROACH
    assert mdp.local_goal_for_phase(mdp.JugglePhase.CATCH_APPROACH) is mdp.JuggleLocalGoal.CATCH_CONTACT
    assert mdp.local_goal_for_phase(mdp.JugglePhase.CATCH_CONTACT) is mdp.JuggleLocalGoal.STABLE_CATCH

    env.episode_length_buf[:] = 1
    assert not mdp.noncanonical_local_goal_success(env).any()
    state.local_success[:] = True
    torch.testing.assert_close(
        mdp.noncanonical_local_goal_success(env),
        torch.tensor((False, True, True, True)),
    )


def test_explicit_preload_assist_latch_overrides_legacy_phase_inference():
    """Launch-ready held rows stay policy-owned while assisted cradle rows remain anchored."""
    env = SimpleNamespace(num_envs=4, device="cpu")
    state = mdp.create_juggle_runtime_state(env)
    phases = torch.tensor(
        (
            int(mdp.JugglePhase.HELD_PRETHROW),
            int(mdp.JugglePhase.HELD_PRETHROW),
            int(mdp.JugglePhase.RELEASE),
            int(mdp.JugglePhase.STABLE_CATCH),
        )
    )
    explicit_assist = torch.tensor((False, True, False, True))
    mdp.initialize_juggle_episode_state(
        state,
        torch.arange(4),
        phases,
        torch.full((4,), 0.35),
        torch.tensor((False, True, False, False)),
        explicit_assist,
    )

    action = object.__new__(JuggleResetPreservingRelativeJointPositionAction)
    action._env = env
    torch.testing.assert_close(action._reset_preload_active_mask(), explicit_assist)
    # The reset latch, rather than a later phase transition, owns whether the
    # one-shot action assist is armed.
    state.current_phases[:] = int(mdp.JugglePhase.HELD_PRETHROW)
    torch.testing.assert_close(action._reset_preload_active_mask(), explicit_assist)

    with pytest.raises(RuntimeError, match="valid only"):
        mdp.initialize_juggle_episode_state(
            state,
            torch.tensor((2,)),
            torch.tensor((int(mdp.JugglePhase.RELEASE),)),
            torch.tensor((0.35,)),
            torch.tensor((False,)),
            torch.tensor((True,)),
        )


def test_moving_reset_does_not_leave_a_persistent_velocity_target():
    """Authored arm motion is physical state, not a hidden actuator command."""
    source = mdp.JuggleResetStateSource(rows_per_phase=8, device="cpu")
    robot = _RecordingRobot()
    ball = _RecordingBall()
    scene = _FakeScene(robot=robot, ball=ball)
    env = SimpleNamespace(num_envs=1, device="cpu", scene=scene)
    runtime = mdp.create_juggle_runtime_state(env)
    # The second half of HELD_PRETHROW rows carries nonzero bootstrap motion.
    runtime.row_ids[0] = 5
    reset = object.__new__(mdp.JuggleResetEvent)
    reset.source = source
    reset.row_count = source.row_count
    reset.runtime = runtime
    reset._robot = robot
    reset._ball = ball
    reset._arm_joint_ids = list(range(7))
    reset._hand_joint_ids = list(range(7, 23))

    reset(env, torch.tensor([0]))

    assert robot.sim_velocity[:, :7].abs().max() > 0.0
    torch.testing.assert_close(robot.velocity_target, torch.zeros_like(robot.velocity_target))
    assert runtime.preload_assist_start.item() == source.preload_assist_rows[5].item()
    assert runtime.start_item_ids.item() == source.item_ids[5].item()
    assert runtime.canonical_start.item() == source.canonical_start_rows[5].item()
    torch.testing.assert_close(runtime.release_origins_xy[0], source.release_origins_xy[5])
    assert runtime.local_goal_ids.item() == source.local_goal_ids[5].item()


def test_phase_sampler_has_exact_stream_mixture_and_checkpoint_roundtrip():
    """All phases stay sampleable and restoring state reproduces the next complete draw."""
    env, curriculum, cfg = _make_fake_curriculum(num_envs=200)
    ids = torch.arange(env.num_envs)

    metrics = curriculum(env, ids, **cfg.params)
    phases = curriculum._reset_term.source.phase_ids[env.juggle_runtime_state.row_ids]
    assert int((phases == int(mdp.JugglePhase.HELD_PRETHROW)).sum()) == 70
    assert set(phases.unique().tolist()) == set(range(len(mdp.JugglePhase)))
    probabilities = curriculum.phase_probabilities()
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))
    torch.testing.assert_close(probabilities[0], torch.tensor(0.35))
    for phase in mdp.JugglePhase:
        phase_name = phase.name.lower()
        assert f"recent_{phase_name}_local_success_rate" in metrics
        assert f"{phase_name}_local_success_rate" in metrics
        assert f"{phase_name}_sampling_probability" in metrics
    for name in (
        "recent_static_held_attempts",
        "recent_static_held_local_successes",
        "recent_static_held_local_success_rate",
        "recent_static_held_full_cycle_successes",
        "recent_static_held_full_cycle_success_rate",
        "static_held_attempts",
        "static_held_local_successes",
        "static_held_local_success_rate",
        "static_held_full_cycle_successes",
        "static_held_full_cycle_success_rate",
    ):
        assert name in metrics

    saved = curriculum.get_state()
    curriculum(env, ids, **cfg.params)
    expected_rows = env.juggle_runtime_state.row_ids.clone()
    curriculum.set_state(saved)
    curriculum(env, ids, **cfg.params)
    torch.testing.assert_close(env.juggle_runtime_state.row_ids, expected_rows)
    assert all(isinstance(value, torch.Tensor) for value in saved.values())


def test_meter_curriculum_samples_only_physical_trainable_items():
    """Meter PPO mixes canonical starts with eligible physical local goals."""
    env, curriculum, cfg = _make_fake_curriculum(num_envs=200, profile=mdp.METER_TOSS_RESET_PROFILE)
    curriculum(env, torch.arange(env.num_envs), **cfg.params)
    source = curriculum._reset_term.source
    rows = env.juggle_runtime_state.row_ids
    canonical = source.canonical_start_rows[rows]

    assert int(canonical.sum()) == 70
    assert source.static_held_rows[rows[canonical]].all()
    sampled_local_items = source.item_ids[rows[~canonical]]
    assert source.adaptive_item_mask[sampled_local_items].all()
    assert not (sampled_local_items == int(mdp.JugglePhase.STABLE_CATCH)).any()

    item_probabilities = curriculum.item_probabilities()
    phase_probabilities = curriculum.phase_probabilities()
    torch.testing.assert_close(item_probabilities.sum(), torch.tensor(1.0))
    torch.testing.assert_close(phase_probabilities.sum(), torch.tensor(1.0))
    torch.testing.assert_close(item_probabilities[0], torch.tensor(0.35))
    torch.testing.assert_close(item_probabilities[7], torch.tensor(0.0))
    assert item_probabilities[8] > 0.0
    assert phase_probabilities[int(mdp.JugglePhase.HELD_PRETHROW)] > 0.35


def test_continuous_curriculum_learns_from_rows_and_roundtrips_exactly():
    """Continuous outcomes train the success model while canonical and coverage shares remain exact."""
    env, curriculum, cfg = _make_fake_curriculum(
        num_envs=200,
        profile=mdp.METER_TOSS_RESET_PROFILE,
        sampling_mode="continuous",
    )
    ids = torch.arange(env.num_envs)
    state = env.juggle_runtime_state
    source = curriculum._reset_term.source

    curriculum(env, ids, **cfg.params)
    sampled_rows = state.row_ids.clone()
    sampled_items = source.item_ids[sampled_rows]
    sampled_canonical = source.canonical_start_rows[sampled_rows]
    trainable_items = source.adaptive_item_mask | source.canonical_item_mask
    assert int(sampled_canonical.sum()) >= 70
    assert trainable_items[sampled_items].all()
    assert not (sampled_items == int(mdp.JugglePhase.STABLE_CATCH)).any()

    state.initialized[:] = True
    state.start_item_ids[:] = source.item_ids[sampled_rows]
    state.start_phases[:] = source.phase_ids[sampled_rows]
    state.static_held_start[:] = source.static_held_rows[sampled_rows]
    # Supply a deterministic, continuous outcome surface for this CPU manager
    # harness; production outcomes come from the physical local-goal state.
    state.local_success[:] = source.reset_parameters[sampled_rows, 1] > 0.5
    state.cycle_success.zero_()
    env.episode_length_buf[:] = 1
    metrics = curriculum(env, ids, **cfg.params)

    assert curriculum._continuous_sampler is not None
    assert curriculum._continuous_sampler.history_count == 200
    assert metrics["continuous_history_count"] == 200.0
    assert 0.0 <= metrics["continuous_predicted_success_minimum"] <= 1.0
    assert 0.0 <= metrics["continuous_predicted_success_maximum"] <= 1.0
    torch.testing.assert_close(curriculum.item_probabilities().sum(), torch.tensor(1.0))

    saved = curriculum.get_state()
    assert saved["curriculum_compatibility_signature"][0] == 11
    assert any(name.startswith("continuous__") for name in saved)
    expected = curriculum._sample_rows(512)
    curriculum.set_state(saved)
    torch.testing.assert_close(curriculum._sample_rows(512), expected)


def test_continuous_curriculum_signature_covers_parameters_features_and_seed():
    """Learned evidence cannot cross a different continuous proposal distribution."""
    _, curriculum, _ = _make_fake_curriculum(
        num_envs=8,
        profile=mdp.METER_TOSS_RESET_PROFILE,
        sampling_mode="continuous",
    )
    names = tuple(name for name, _ in curriculum._reset_source_signature_fields())
    saved = curriculum.get_state()

    assert names[-3:] == ("reset_parameters", "model_features", "continuous_seed")
    curriculum._reset_term.source.reset_parameters[0, 0] += 0.01
    with pytest.raises(ValueError, match="curriculum_compatibility_signature"):
        curriculum.set_state(saved)


def test_curriculum_signature_covers_physical_catalog_and_local_goals():
    """Saved evidence is invalidated by any physical reset or goal change."""
    expected_names = (
        "arm_positions",
        "arm_velocities",
        "hand_positions",
        "hand_velocities",
        "ball_positions",
        "ball_quaternions",
        "ball_velocities",
        "release_positions",
        "release_origins_xy",
        "release_velocities",
        "launch_reference_heights",
        "ballistic_rows",
        "flight_times",
        "difficulty_band_ids",
        "phase_ids",
        "item_ids",
        "item_phase_ids",
        "static_held_rows",
        "preload_assist_rows",
        "canonical_start_rows",
        "canonical_row_ids",
        "canonical_item_mask",
        "adaptive_item_mask",
        "local_goal_ids",
    )
    _, curriculum, _ = _make_fake_curriculum(num_envs=8, profile=mdp.METER_TOSS_RESET_PROFILE)
    names = tuple(name for name, _ in curriculum._reset_source_signature_fields())
    source = curriculum._reset_term.source
    saved = curriculum.get_state()

    assert names == expected_names
    assert saved["curriculum_compatibility_signature"][0] == 9
    source.local_goal_ids[0] = int(mdp.JuggleLocalGoal.STABLE_CATCH)
    with pytest.raises(ValueError, match="curriculum_compatibility_signature"):
        curriculum.set_state(saved)


@pytest.mark.parametrize(
    "profile",
    (
        pytest.param(mdp.LOW_TOSS_RESET_PROFILE, id="low"),
        pytest.param(mdp.METER_TOSS_RESET_PROFILE, id="meter"),
    ),
)
def test_curriculum_snapshot_rejects_changed_outcome_semantics(profile):
    """LOW and meter evidence cannot cross changed success or failure thresholds."""
    env, curriculum, _ = _make_fake_curriculum(num_envs=8, profile=profile)
    saved = curriculum.get_state()
    progress_cfg = env.termination_manager.get_term_cfg("progress_context")
    workspace_cfg = env.termination_manager.get_term_cfg("ball_out_of_workspace")

    progress_changes = {
        "release_separation_distance": 0.031,
        "release_clear_steps": 3,
        "apex_height_gain": float(progress_cfg.params["apex_height_gain"]) + 0.01,
        "apex_maximum_horizontal_displacement": 0.13,
        "rearm_after_stable_catch": not progress_cfg.params.get("rearm_after_stable_catch", False),
    }
    for name, changed_value in progress_changes.items():
        original_present = name in progress_cfg.params
        original_value = progress_cfg.params.get(name)
        progress_cfg.params[name] = changed_value
        with pytest.raises(ValueError, match="curriculum_compatibility_signature"):
            curriculum.set_state(saved)
        if original_present:
            progress_cfg.params[name] = original_value
        else:
            progress_cfg.params.pop(name)

    original_workspace = workspace_cfg.params.get("workspace_upper")
    workspace_cfg.params["workspace_upper"] = (0.80, 0.35, 2.01)
    with pytest.raises(ValueError, match="curriculum_compatibility_signature"):
        curriculum.set_state(saved)
    if original_workspace is None:
        workspace_cfg.params.pop("workspace_upper")
    else:
        workspace_cfg.params["workspace_upper"] = original_workspace


def test_low_curriculum_snapshot_rejects_changed_reset_source_semantics():
    """The non-ordered LOW curriculum fingerprints the physical reset source too."""
    _, curriculum, _ = _make_fake_curriculum(num_envs=8)
    source = curriculum._reset_term.source
    saved = curriculum.get_state()
    original_joint = source.arm_positions[0, 0].clone()
    source.arm_positions[0, 0] += 0.01

    with pytest.raises(ValueError, match="curriculum_compatibility_signature"):
        curriculum.set_state(saved)

    source.arm_positions[0, 0] = original_joint


@pytest.mark.parametrize(
    "profile",
    (
        pytest.param(mdp.LOW_TOSS_RESET_PROFILE, id="low"),
        pytest.param(mdp.METER_TOSS_RESET_PROFILE, id="meter"),
    ),
)
def test_curriculum_checkpoint_generators_fork_deterministically_by_ddp_rank(profile):
    """Authoritative evidence is shared while restored sampling streams differ by rank."""
    _, curriculum, _ = _make_fake_curriculum(num_envs=8, profile=profile)
    saved = curriculum.get_state()

    def restored_draw(global_rank: int) -> torch.Tensor:
        _, restored, _ = _make_fake_curriculum(num_envs=8, profile=profile)
        restored.set_state(saved)
        restored.reseed_checkpoint_generators(global_rank)
        torch.testing.assert_close(restored._attempts, saved["attempts"])
        return restored._sample_rows(512)

    rank_one_first = restored_draw(1)
    rank_one_second = restored_draw(1)
    rank_two = restored_draw(2)

    torch.testing.assert_close(rank_one_first, rank_one_second)
    assert not torch.equal(rank_one_first, rank_two)


def test_semantic_mode_uses_one_adaptive_sampler_and_exact_stream_distribution():
    """Both profiles retain the existing canonical-plus-semantic option."""
    for profile in (mdp.LOW_TOSS_RESET_PROFILE, mdp.METER_TOSS_RESET_PROFILE):
        _, curriculum, _ = _make_fake_curriculum(num_envs=200, profile=profile)
        assert hasattr(curriculum, "_sampler")
        state = curriculum.get_state()
        assert any(name.startswith("sampler__") for name in state)
        assert not any("ordered" in name for name in state)
        expected = curriculum._canonical_item_probabilities * 0.35
        expected += 0.65 * curriculum._sampler.sampling_probabilities(curriculum._monitor.success_rates)
        torch.testing.assert_close(curriculum.item_probabilities(), expected)


def test_curriculum_snapshot_rejects_a_different_reset_profile_without_mutation():
    """Adaptive evidence cannot silently cross physical reset-profile boundaries."""
    _, low_curriculum, _ = _make_fake_curriculum(num_envs=8, profile=mdp.LOW_TOSS_RESET_PROFILE)
    _, meter_curriculum, _ = _make_fake_curriculum(num_envs=8, profile=mdp.METER_TOSS_RESET_PROFILE)
    assert mdp.LOW_TOSS_RESET_PROFILE.profile_id == 0
    assert mdp.METER_TOSS_RESET_PROFILE.profile_id == 1
    low_state = low_curriculum.get_state()
    meter_state = meter_curriculum.get_state()

    with pytest.raises(ValueError, match="Reset profile mismatch"):
        meter_curriculum.set_state(low_state)

    unchanged_state = meter_curriculum.get_state()
    assert set(unchanged_state) == set(meter_state)
    for name in meter_state:
        torch.testing.assert_close(unchanged_state[name], meter_state[name])


def test_curriculum_snapshot_rejects_incompatible_semantic_item_count_without_mutation():
    """A profile-matched snapshot cannot cross a different semantic catalog shape."""
    _, curriculum, _ = _make_fake_curriculum(num_envs=8, profile=mdp.METER_TOSS_RESET_PROFILE)
    original = curriculum.get_state()
    incompatible = {name: value.clone() for name, value in original.items()}
    incompatible["catalog_item_count"] += 1

    with pytest.raises(ValueError, match="reset catalog item count"):
        curriculum.set_state(incompatible)

    unchanged = curriculum.get_state()
    for name in original:
        torch.testing.assert_close(unchanged[name], original[name])


def test_curriculum_snapshot_restore_is_atomic_after_a_late_validation_error():
    """A late malformed metric rolls back earlier monitor, sampler, RNG, and counter restores."""
    _, curriculum, _ = _make_fake_curriculum(
        num_envs=8,
        profile=mdp.METER_TOSS_RESET_PROFILE,
    )
    donor_env, donor, donor_cfg = _make_fake_curriculum(
        num_envs=8,
        profile=mdp.METER_TOSS_RESET_PROFILE,
    )
    donor_runtime = donor_env.juggle_runtime_state
    completed_ids = torch.arange(4)
    launch_item = 8
    donor_runtime.initialized[completed_ids] = True
    donor_runtime.start_item_ids[completed_ids] = launch_item
    donor_runtime.start_phases[completed_ids] = donor._reset_term.source.item_phase_ids[launch_item]
    donor_runtime.local_success[completed_ids] = torch.tensor((True, False, True, True))
    donor_env.episode_length_buf[completed_ids] = 1
    donor(donor_env, completed_ids, **donor_cfg.params)

    before = curriculum.get_state()
    malformed = donor.get_state()
    assert not torch.equal(malformed["row_generator_state"], before["row_generator_state"])
    assert not torch.equal(malformed["monitor__history"], before["monitor__history"])
    assert not torch.equal(malformed["attempts"], before["attempts"])
    malformed["cached_metrics"][0] = torch.nan

    with pytest.raises(ValueError, match="cached_metrics must be finite"):
        curriculum.set_state(malformed)

    after = curriculum.get_state()
    assert set(after) == set(before)
    for name in before:
        assert torch.equal(after[name], before[name]), name


def test_static_held_metrics_use_completed_episodes_and_roundtrip_partial_window():
    """Static deployment metrics count episode outcomes and preserve an unfinished report window."""
    env, curriculum, cfg = _make_fake_curriculum(num_envs=8)
    state = env.juggle_runtime_state
    ids = torch.arange(env.num_envs)

    state.initialized[:] = True
    state.start_phases[:] = torch.tensor((0, 0, 0, 1, 0, 2, 3, 0))
    state.start_item_ids.copy_(state.start_phases)
    state.static_held_start[:] = torch.tensor((True, True, True, False, True, False, False, True))
    state.local_success[:] = torch.tensor((True, True, True, True, False, False, False, False))
    state.cycle_success[:] = torch.tensor((True, False, True, True, False, False, False, False))
    env.episode_length_buf[:] = 1
    metrics = curriculum(env, ids, **cfg.params)

    assert metrics["recent_static_held_attempts"] == 5.0
    assert metrics["recent_static_held_local_successes"] == 3.0
    assert metrics["recent_static_held_local_success_rate"] == pytest.approx(0.6)
    assert metrics["recent_static_held_full_cycle_successes"] == 2.0
    assert metrics["recent_static_held_full_cycle_success_rate"] == pytest.approx(0.4)
    assert metrics["static_held_attempts"] == 5.0
    assert metrics["static_held_local_successes"] == 3.0
    assert metrics["static_held_local_success_rate"] == pytest.approx(0.6)
    assert metrics["static_held_full_cycle_successes"] == 2.0
    assert metrics["static_held_full_cycle_success_rate"] == pytest.approx(0.4)

    partial_ids = ids[:4]
    state.start_phases[partial_ids] = torch.tensor((0, 0, 0, 1))
    state.start_item_ids[partial_ids] = state.start_phases[partial_ids]
    state.static_held_start[partial_ids] = torch.tensor((True, True, True, False))
    state.local_success[partial_ids] = torch.tensor((True, True, False, True))
    state.cycle_success[partial_ids] = torch.tensor((True, False, False, True))
    curriculum(env, partial_ids, **cfg.params)
    saved = curriculum.get_state()

    final_ids = ids[4:]
    state.start_phases[final_ids] = torch.tensor((0, 2, 3, 4))
    state.start_item_ids[final_ids] = state.start_phases[final_ids]
    state.static_held_start[final_ids] = torch.tensor((True, False, False, False))
    state.local_success[final_ids] = torch.tensor((True, False, False, False))
    state.cycle_success[final_ids] = torch.tensor((True, False, False, False))
    expected_metrics = curriculum(env, final_ids, **cfg.params).copy()
    expected_state = curriculum.get_state()

    curriculum.set_state(saved)
    roundtripped_partial_state = curriculum.get_state()
    for name in saved:
        torch.testing.assert_close(roundtripped_partial_state[name], saved[name])
    restored_metrics = curriculum(env, final_ids, **cfg.params)
    assert restored_metrics == expected_metrics
    restored_state = curriculum.get_state()
    assert set(restored_state) == set(expected_state)
    for name in expected_state:
        torch.testing.assert_close(restored_state[name], expected_state[name])

    assert restored_metrics["recent_static_held_attempts"] == 4.0
    assert restored_metrics["recent_static_held_local_successes"] == 3.0
    assert restored_metrics["recent_static_held_local_success_rate"] == pytest.approx(0.75)
    assert restored_metrics["recent_static_held_full_cycle_successes"] == 2.0
    assert restored_metrics["recent_static_held_full_cycle_success_rate"] == pytest.approx(0.5)
    assert restored_metrics["static_held_attempts"] == 9.0
    assert restored_metrics["static_held_local_successes"] == 6.0
    assert restored_metrics["static_held_local_success_rate"] == pytest.approx(2.0 / 3.0)
    assert restored_metrics["static_held_full_cycle_successes"] == 4.0
    assert restored_metrics["static_held_full_cycle_success_rate"] == pytest.approx(4.0 / 9.0)


def _make_fake_curriculum(
    num_envs: int,
    profile: mdp.JuggleResetProfile = mdp.LOW_TOSS_RESET_PROFILE,
    sampling_mode: str = "semantic",
):
    """Build a CPU-only manager-term harness without a simulator."""
    source = mdp.JuggleResetStateSource(
        rows_per_phase=32 if profile.profile_id == mdp.METER_TOSS_RESET_PROFILE.profile_id else 16,
        device="cpu",
        profile=profile,
        parameter_sampling="continuous" if sampling_mode == "continuous" else "catalog",
        continuous_seed=17,
    )
    reset_term = object.__new__(mdp.JuggleResetEvent)
    reset_term.source = source
    reset_term.catalog = source.catalog
    reset_term.row_count = source.row_count
    reset_term.phase_ids = source.phase_ids
    event_manager = SimpleNamespace(
        get_term_cfg=lambda name: SimpleNamespace(func=reset_term) if name == "reset_from_catalog" else None
    )
    if profile.profile_id == mdp.METER_TOSS_RESET_PROFILE.profile_id:
        task_cfg = parse_env_cfg(TASK_NAME, device="cpu", num_envs=num_envs)
    else:
        task_cfg = _KukaAllegroJuggleBaseEnvCfg()
    termination_cfgs = {
        "progress_context": task_cfg.terminations.progress_context,
        "ball_out_of_workspace": task_cfg.terminations.ball_out_of_workspace,
    }
    termination_manager = SimpleNamespace(get_term_cfg=termination_cfgs.__getitem__)
    step_dt = task_cfg.sim.dt * task_cfg.decimation
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        event_manager=event_manager,
        termination_manager=termination_manager,
        step_dt=step_dt,
        max_episode_length=math.ceil(task_cfg.episode_length_s / step_dt),
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
    )
    mdp.create_juggle_runtime_state(env)
    canonical_fraction = 0.35
    coverage_fraction = 0.15 / 0.65
    cfg = CurriculumTermCfg(
        func=mdp.JuggleResetCurriculum,
        params={
            "outcome_monitor": RollingOutcomeMonitorCfg(
                history_length=8,
                prior_strength=2.0,
            ),
            "adaptive_sampler": AdaptiveResetSamplerCfg(
                target_success_rate=0.5,
                coverage_fraction=0.15 / 0.65,
            ),
            "continuous_sampler": ContinuousAdaptiveResetSamplerCfg(
                target_success_rate=0.5,
                coverage_fraction=coverage_fraction,
                history_length=256,
                prior_strength=2.0,
                kernel_bandwidth=0.35,
            ),
            "canonical_fraction": canonical_fraction,
            "sampling_mode": sampling_mode,
        },
    )
    return env, mdp.JuggleResetCurriculum(cfg, env), cfg


def _make_progress_harness(num_envs: int):
    """Build a CPU-only progress-term harness with mutable public asset data."""
    body_position = torch.zeros((num_envs, 5, 3))
    body_quaternion = torch.zeros((num_envs, 5, 4))
    body_quaternion[..., 3] = 1.0
    body_velocity = torch.zeros((num_envs, 5, 3))
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=SimpleNamespace(torch=body_position),
            body_quat_w=SimpleNamespace(torch=body_quaternion),
            body_link_lin_vel_w=SimpleNamespace(torch=body_velocity),
            body_link_ang_vel_w=SimpleNamespace(torch=torch.zeros_like(body_velocity)),
            joint_pos=SimpleNamespace(torch=torch.zeros((num_envs, 23))),
            joint_vel=SimpleNamespace(torch=torch.zeros((num_envs, 23))),
        )
    )
    ball = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=torch.zeros((num_envs, 3))),
            root_lin_vel_w=SimpleNamespace(torch=torch.zeros((num_envs, 3))),
        )
    )
    scene = {"robot": robot, "ball": ball}
    scene = _ProgressScene(scene, num_envs=num_envs)
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=scene,
        step_dt=1.0 / 60.0,
        episode_length_buf=torch.ones(num_envs, dtype=torch.long),
        extras={},
    )
    mdp.create_juggle_runtime_state(env)
    context = object.__new__(mdp.JuggleProgressContext)
    context._env = env
    context._no_termination = torch.zeros(num_envs, dtype=torch.bool)
    context._local_goals = torch.tensor(
        [int(mdp.local_goal_for_phase(phase)) for phase in mdp.JugglePhase],
        dtype=torch.long,
    )
    return env, context


def _set_progress_geometry(
    env,
    *,
    palm_height: float | torch.Tensor,
    ball_height: float | torch.Tensor,
    ball_vertical_velocity: float | torch.Tensor,
    supported: bool,
    palm_x: float | torch.Tensor = 0.50,
    ball_x: float | torch.Tensor = 0.50,
) -> None:
    """Set palm, ball, and fingertip geometry for one progress update."""
    num_envs = env.num_envs
    palm_height = torch.as_tensor(palm_height, dtype=torch.float32).expand(num_envs)
    ball_height = torch.as_tensor(ball_height, dtype=torch.float32).expand(num_envs)
    ball_vertical_velocity = torch.as_tensor(ball_vertical_velocity, dtype=torch.float32).expand(num_envs)
    palm_x = torch.as_tensor(palm_x, dtype=torch.float32).expand(num_envs)
    ball_x = torch.as_tensor(ball_x, dtype=torch.float32).expand(num_envs)
    robot_position = env.scene["robot"].data.body_pos_w.torch
    robot_position.zero_()
    robot_position[:, 0, 0] = palm_x
    robot_position[:, 0, 2] = palm_height
    ball_position = env.scene["ball"].data.root_pos_w.torch
    ball_position.zero_()
    ball_position[:, 0] = ball_x
    ball_position[:, 2] = ball_height
    ball_velocity = env.scene["ball"].data.root_lin_vel_w.torch
    ball_velocity.zero_()
    ball_velocity[:, 2] = ball_vertical_velocity
    if supported:
        relative_tip_positions = torch.tensor(
            ((0.050, 0.000, -0.010), (-0.050, 0.000, 0.000), (0.12, 0.0, 0.0), (0.13, 0.0, 0.0))
        )
    else:
        relative_tip_positions = torch.tensor(
            ((0.12, 0.00, 0.00), (0.13, 0.00, 0.00), (0.12, 0.02, 0.0), (0.13, -0.02, 0.0))
        )
    robot_position[:, 1:] = ball_position.unsqueeze(1) + relative_tip_positions.unsqueeze(0)


def _step_progress(
    context,
    env,
    *,
    apex_height_gain: float,
    apex_maximum_horizontal_displacement: float | None = None,
    track_supported_release_reference: bool = False,
    rearm_after_stable_catch: bool = False,
) -> None:
    """Run one progress update with the harness entity selections."""
    context(
        env,
        tool_body_cfg=SimpleNamespace(name="robot", body_ids=[0]),
        fingertip_cfg=SimpleNamespace(name="robot", body_ids=[1, 2, 3, 4]),
        tool_offset=(0.0, 0.0, 0.0),
        apex_height_gain=apex_height_gain,
        apex_maximum_horizontal_displacement=apex_maximum_horizontal_displacement,
        track_supported_release_reference=track_supported_release_reference,
        rearm_after_stable_catch=rearm_after_stable_catch,
    )


class _ProgressScene(dict):
    """Scene mapping with per-environment origins for progress-term tests."""

    def __init__(self, assets, *, num_envs: int):
        super().__init__(assets)
        self.env_origins = torch.zeros((num_envs, 3))


class _FakeTerminationManager:
    """Return already-computed named termination masks."""

    def __init__(self, terms):
        self._terms = terms

    def get_term(self, name):
        return self._terms[name]


class _FakeScene(dict):
    """Minimal scene mapping for the reset-event regression test."""

    def __init__(self, **assets):
        super().__init__(assets)
        self.env_origins = torch.zeros((1, 3))


class _RecordingRobot:
    """Record target and simulated joint-state writes."""

    def __init__(self):
        defaults = torch.zeros((1, 23))
        self.data = SimpleNamespace(default_joint_pos=SimpleNamespace(torch=defaults))

    def set_joint_position_target_index(self, *, target, env_ids):
        self.position_target = target.clone()

    def set_joint_velocity_target_index(self, *, target, env_ids):
        self.velocity_target = target.clone()

    def write_joint_position_to_sim_index(self, *, position, env_ids):
        self.sim_position = position.clone()

    def write_joint_velocity_to_sim_index(self, *, velocity, env_ids):
        self.sim_velocity = velocity.clone()


class _RecordingBall:
    """Accept reset pose and velocity writes."""

    def __init__(self):
        default_pose = torch.zeros((1, 7))
        default_pose[:, 3] = 1.0
        self.data = SimpleNamespace(default_root_pose=SimpleNamespace(torch=default_pose))

    def write_root_pose_to_sim_index(self, *, root_pose, env_ids):
        self.root_pose = root_pose.clone()

    def write_root_velocity_to_sim_index(self, *, root_velocity, env_ids):
        self.root_velocity = root_velocity.clone()
