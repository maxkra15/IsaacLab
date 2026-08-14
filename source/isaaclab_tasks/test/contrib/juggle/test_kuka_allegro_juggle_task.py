# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public-contract tests for one-ball KUKA-Allegro juggling."""

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
from isaaclab_tasks.contrib.stack.mdp.kuka_allegro_reset import (
    KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES,
    kuka_allegro_tool_pose,
)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
from isaaclab_tasks.utils.reset_sampling import AdaptiveResetSamplerCfg, RollingOutcomeMonitorCfg

TASK_NAME = "IsaacContrib-Juggle-Ball-KukaAllegro-RL"


def test_task_registration_and_23_action_config():
    """The task resolves a standalone Newton environment and normal RSL-RL config."""
    spec = gym.spec(TASK_NAME)
    assert spec.kwargs["env_cfg_entry_point"].endswith(":KukaAllegroJuggleRLEnvCfg")
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
    assert cfg.actions.arm_action.gravity_compensation
    assert cfg.actions.arm_action.scale == cfg.actions.arm_action.max_delta == 0.60
    assert len(cfg.actions.arm_action.joint_names) == 7
    assert tuple(cfg.actions.hand_action.joint_names) == KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES
    assert len(cfg.actions.hand_action.joint_names) == 16
    assert isinstance(cfg.actions.hand_action, mdp.JuggleResetPreservingRelativeJointPositionActionCfg)
    assert cfg.actions.hand_action.reset_preload_commands_by_pair == (mdp.JUGGLE_SPHERE_PRELOAD_HAND_POSITION,)
    assert cfg.actions.hand_action.reset_open_commands_by_pair == (mdp.JUGGLE_SPHERE_OPEN_HAND_POSITION,)
    assert cfg.actions.hand_action.preload_release_threshold == 0.01
    assert cfg.actions.hand_action.preload_release_steps == 1
    opened = torch.tensor(mdp.JUGGLE_SPHERE_OPEN_HAND_POSITION)
    torch.testing.assert_close(torch.tensor(mdp.JUGGLE_SPHERE_CONTACT_HAND_POSITION), opened)
    torch.testing.assert_close(torch.tensor(mdp.JUGGLE_SPHERE_FLIGHT_GATE_HAND_POSITION), opened)
    assert cfg.scene.ball.spawn.radius == mdp.BALL_RADIUS
    assert cfg.scene.ball.spawn.mass_props.mass == mdp.BALL_MASS
    assert cfg.scene.ball.spawn.physics_material.restitution == 0.0
    assert cfg.terminations.progress_context.params["release_separation_distance"] == 0.03
    assert cfg.terminations.progress_context.params["release_clear_steps"] == 2
    assert cfg.terminations.progress_context.params["apex_height_gain"] == 0.06
    assert cfg.terminations.progress_context.params["catch_approach_distance"] == 0.12
    assert cfg.terminations.progress_context.params["stable_catch_steps"] == 15
    assert cfg.rewards.local_transition.weight == 1.0
    assert cfg.rewards.full_cycle.weight == 2.0
    assert isinstance(runner, KukaAllegroJugglePPORunnerCfg)
    assert runner.wandb_project == "kuka_allegro_juggle"
    assert runner.num_steps_per_env == 32
    assert runner.algorithm.learning_rate == 5.0e-5
    assert runner.algorithm.gamma == 0.998
    assert runner.actor.distribution_cfg.init_std == 0.12

    cfg.play_mode()
    assert cfg.curriculum is None
    assert cfg.events.reset_from_catalog.params["fixed_phase"] == int(mdp.JugglePhase.HELD_PRETHROW)
    assert cfg.events.reset_from_catalog.params["static_held_only"]


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
    assert int(source.static_held_rows[held_rows].sum()) == len(held_rows) // 2
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


def test_reset_initialization_never_counts_the_authored_state_as_success():
    """Even contact/stable rows begin with uncredited fresh-event latches."""
    env = SimpleNamespace(num_envs=4, device="cpu")
    state = mdp.create_juggle_runtime_state(env)
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
    assert not state.cycle_success.any()
    assert not state.new_cycle_success.any()
    assert not state.seen_apex.any()
    assert state.stable_catch_steps.count_nonzero() == 0

    assert state.static_held_start.tolist() == static_held.tolist()
    assert mdp.local_goal_for_phase(mdp.JugglePhase.HELD_PRETHROW) is mdp.JuggleLocalGoal.FLIGHT_APEX
    assert mdp.local_goal_for_phase(mdp.JugglePhase.STABLE_CATCH) is mdp.JuggleLocalGoal.FLIGHT_APEX
    assert mdp.local_goal_for_phase(mdp.JugglePhase.RELEASE) is mdp.JuggleLocalGoal.FLIGHT_APEX
    assert mdp.local_goal_for_phase(mdp.JugglePhase.ASCENDING) is mdp.JuggleLocalGoal.FLIGHT_APEX
    for approach_phase in (mdp.JugglePhase.APEX, mdp.JugglePhase.DESCENDING):
        assert mdp.local_goal_for_phase(approach_phase) is mdp.JuggleLocalGoal.CATCH_APPROACH
    assert mdp.local_goal_for_phase(mdp.JugglePhase.CATCH_APPROACH) is mdp.JuggleLocalGoal.CATCH_CONTACT
    assert mdp.local_goal_for_phase(mdp.JugglePhase.CATCH_CONTACT) is mdp.JuggleLocalGoal.STABLE_CATCH

    state.local_success[:] = True
    torch.testing.assert_close(
        mdp.noncanonical_local_goal_success(env),
        torch.tensor((False, True, True, True)),
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
        "recent_static_held_full_cycle_successes",
        "recent_static_held_full_cycle_success_rate",
        "static_held_attempts",
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


def test_static_held_metrics_use_completed_episodes_and_roundtrip_partial_window():
    """Static deployment metrics count episode outcomes and preserve an unfinished report window."""
    env, curriculum, cfg = _make_fake_curriculum(num_envs=8)
    state = env.juggle_runtime_state
    ids = torch.arange(env.num_envs)

    state.initialized[:] = True
    state.start_phases[:] = torch.tensor((0, 0, 0, 1, 0, 2, 3, 0))
    state.static_held_start[:] = torch.tensor((True, True, True, False, True, False, False, True))
    state.cycle_success[:] = torch.tensor((True, False, True, True, False, False, False, False))
    env.episode_length_buf[:] = 1
    metrics = curriculum(env, ids, **cfg.params)

    assert metrics["recent_static_held_attempts"] == 5.0
    assert metrics["recent_static_held_full_cycle_successes"] == 2.0
    assert metrics["recent_static_held_full_cycle_success_rate"] == pytest.approx(0.4)
    assert metrics["static_held_attempts"] == 5.0
    assert metrics["static_held_full_cycle_successes"] == 2.0
    assert metrics["static_held_full_cycle_success_rate"] == pytest.approx(0.4)

    partial_ids = ids[:4]
    state.start_phases[partial_ids] = torch.tensor((0, 0, 0, 1))
    state.static_held_start[partial_ids] = torch.tensor((True, True, True, False))
    state.cycle_success[partial_ids] = torch.tensor((True, False, False, True))
    curriculum(env, partial_ids, **cfg.params)
    saved = curriculum.get_state()

    final_ids = ids[4:]
    state.start_phases[final_ids] = torch.tensor((0, 2, 3, 4))
    state.static_held_start[final_ids] = torch.tensor((True, False, False, False))
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
    assert restored_metrics["recent_static_held_full_cycle_successes"] == 2.0
    assert restored_metrics["recent_static_held_full_cycle_success_rate"] == pytest.approx(0.5)
    assert restored_metrics["static_held_attempts"] == 9.0
    assert restored_metrics["static_held_full_cycle_successes"] == 4.0
    assert restored_metrics["static_held_full_cycle_success_rate"] == pytest.approx(4.0 / 9.0)


def _make_fake_curriculum(num_envs: int):
    """Build a CPU-only manager-term harness without a simulator."""
    source = mdp.JuggleResetStateSource(rows_per_phase=16, device="cpu")
    reset_term = object.__new__(mdp.JuggleResetEvent)
    reset_term.source = source
    reset_term.catalog = source.catalog
    reset_term.row_count = source.row_count
    reset_term.phase_ids = source.phase_ids
    event_manager = SimpleNamespace(
        get_term_cfg=lambda name: SimpleNamespace(func=reset_term) if name == "reset_from_catalog" else None
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        event_manager=event_manager,
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
    )
    mdp.create_juggle_runtime_state(env)
    cfg = CurriculumTermCfg(
        func=mdp.JuggleResetCurriculum,
        params={
            "outcome_monitor": RollingOutcomeMonitorCfg(history_length=8, prior_strength=2.0),
            "adaptive_sampler": AdaptiveResetSamplerCfg(
                target_success_rate=0.5,
                coverage_fraction=0.15 / 0.65,
            ),
            "canonical_fraction": 0.35,
        },
    )
    return env, mdp.JuggleResetCurriculum(cfg, env), cfg


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
