# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for reset-aware physical juggling progress shaping."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import RewardTermCfg

from isaaclab_tasks.contrib.juggle import mdp
from isaaclab_tasks.contrib.juggle.mdp import terms as juggle_terms


class _Scene(dict):
    def __init__(self, ball: object, num_envs: int):
        super().__init__(ball=ball)
        self.env_origins = torch.zeros((num_envs, 3))


class _TerminationManager:
    def __init__(self, num_envs: int):
        self.dropped = torch.zeros(num_envs, dtype=torch.bool)

    def get_term(self, name: str) -> torch.Tensor:
        if name != "ball_out_of_workspace":
            raise KeyError(name)
        return self.dropped


def _fake_env(num_envs: int) -> SimpleNamespace:
    ball_position = torch.zeros((num_envs, 3))
    ball_velocity = torch.zeros((num_envs, 3))
    ball = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=ball_position),
            root_lin_vel_w=SimpleNamespace(torch=ball_velocity),
        )
    )
    runtime = SimpleNamespace(
        start_phases=torch.zeros(num_envs, dtype=torch.long),
        release_heights=torch.zeros(num_envs),
        release_origins_xy=torch.zeros((num_envs, 2)),
        local_goal_ids=torch.full((num_envs,), int(mdp.JuggleLocalGoal.FLIGHT_APEX), dtype=torch.long),
        canonical_start=torch.zeros(num_envs, dtype=torch.bool),
        local_success=torch.zeros(num_envs, dtype=torch.bool),
        new_local_success=torch.zeros(num_envs, dtype=torch.bool),
        height_success=torch.zeros(num_envs, dtype=torch.bool),
        new_cycle_success=torch.zeros(num_envs, dtype=torch.bool),
        current_phases=torch.zeros(num_envs, dtype=torch.long),
    )
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        step_dt=1.0 / 60.0,
        scene=_Scene(ball, num_envs),
        juggle_runtime_state=runtime,
        termination_manager=_TerminationManager(num_envs),
        tool_position=torch.zeros((num_envs, 3)),
        tool_velocity=torch.zeros((num_envs, 3)),
    )


@pytest.fixture
def fake_tool_state(monkeypatch: pytest.MonkeyPatch):
    def tool_state(env, *_args, **_kwargs):
        zeros = torch.zeros_like(env.tool_position)
        quaternion = torch.zeros((env.num_envs, 4))
        quaternion[:, 3] = 1.0
        return env.tool_position, quaternion, env.tool_velocity, zeros

    monkeypatch.setattr(juggle_terms, "tool_state", tool_state)


def test_physical_progress_potential_is_bounded_and_uses_full_local_ranges(fake_tool_state):
    env = _fake_env(3)
    ball_position = env.scene["ball"].data.root_pos_w.torch
    ball_velocity = env.scene["ball"].data.root_lin_vel_w.torch

    # A quarter-target vertical ballistic prediction with no lateral drift.
    ball_velocity[0, 2] = math.sqrt(2.0 * 9.81 * 0.25)
    # Distance and relative speed are each exactly one shaping scale.
    env.juggle_runtime_state.local_goal_ids[1] = int(mdp.JuggleLocalGoal.STABLE_CATCH)
    ball_position[1, 2] = 0.12
    ball_velocity[1, 2] = 0.45
    # Canonical launch reserves only the first half of the full-cycle potential.
    env.juggle_runtime_state.canonical_start[2] = True
    ball_velocity[2, 2] = math.sqrt(2.0 * 9.81 * 0.25)

    potential = mdp.juggle_physical_progress_potential(env)

    torch.testing.assert_close(potential, torch.tensor((0.25, 0.25, 0.125)), atol=1.0e-6, rtol=0.0)
    assert torch.isfinite(potential).all()
    assert ((potential >= 0.0) & (potential <= 1.0)).all()


def test_canonical_apex_to_catch_stage_has_no_artificial_half_point_drop(fake_tool_state):
    env = _fake_env(1)
    ball_position = env.scene["ball"].data.root_pos_w.torch
    ball_velocity = env.scene["ball"].data.root_lin_vel_w.torch
    env.juggle_runtime_state.canonical_start[:] = True
    ball_velocity[:, 2] = math.sqrt(2.0 * 9.81)
    before = mdp.juggle_physical_progress_potential(env)

    env.juggle_runtime_state.height_success[:] = True
    ball_position[:, 0] = 100.0
    ball_velocity.zero_()
    after = mdp.juggle_physical_progress_potential(env)

    torch.testing.assert_close(before, torch.tensor((0.5,)), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(after, torch.tensor((0.5,)), atol=1.0e-6, rtol=0.0)


def test_completed_local_goal_hands_off_to_continuous_launch_without_a_stage_drop(fake_tool_state):
    """A catch reset must train the following throw instead of ending or rewarding a hold."""
    env = _fake_env(1)
    state = env.juggle_runtime_state
    state.local_goal_ids[:] = int(mdp.JuggleLocalGoal.STABLE_CATCH)
    state.local_success[:] = True
    state.new_local_success[:] = True
    state.current_phases[:] = int(mdp.JugglePhase.HELD_PRETHROW)
    env.scene["ball"].data.root_lin_vel_w.torch[:, 2] = math.sqrt(2.0 * 9.81 * 0.25)

    # Once the reset-local catch is complete, the same episode uses the first
    # half of the continuous-cycle potential for its next launch.
    torch.testing.assert_close(
        mdp.juggle_physical_progress_potential(env),
        torch.tensor((0.125,)),
        atol=1.0e-6,
        rtol=0.0,
    )

    cfg = RewardTermCfg(func=mdp.JugglePhysicalProgressReward, weight=1.0)
    term = mdp.JugglePhysicalProgressReward(cfg, env)
    term._previous[:] = 0.95
    term._baseline_valid[:] = True
    term._skip_next[:] = False
    term._previous_phase[:] = int(mdp.JugglePhase.CATCH_CONTACT)
    torch.testing.assert_close(term(env, gamma=0.998), torch.zeros(1))

    state.new_local_success.zero_()
    env.scene["ball"].data.root_lin_vel_w.torch[:, 2] = math.sqrt(2.0 * 9.81 * 0.50)
    assert term(env, gamma=1.0).item() > 0.0


def test_physical_progress_has_finite_nonzero_launch_and_catch_gradients(fake_tool_state):
    env = _fake_env(2)
    env.juggle_runtime_state.local_goal_ids[1] = int(mdp.JuggleLocalGoal.STABLE_CATCH)
    ball_position = torch.tensor(((0.01, 0.0, 0.10), (0.0, 0.0, 0.08)), requires_grad=True)
    ball_velocity = torch.tensor(((0.02, 0.0, 2.00), (0.0, 0.0, 0.20)), requires_grad=True)
    env.scene["ball"].data.root_pos_w.torch = ball_position
    env.scene["ball"].data.root_lin_vel_w.torch = ball_velocity

    mdp.juggle_physical_progress_potential(env).sum().backward()

    assert ball_position.grad is not None and ball_velocity.grad is not None
    assert torch.isfinite(ball_position.grad).all() and torch.isfinite(ball_velocity.grad).all()
    assert ball_velocity.grad[0, 2] > 0.0
    assert ball_position.grad[1, 2] < 0.0
    assert ball_velocity.grad[1, 2] < 0.0


def test_progress_reward_rebaselines_then_pays_and_repays_signed_progress(fake_tool_state):
    env = _fake_env(1)
    env.juggle_runtime_state.local_goal_ids[:] = int(mdp.JuggleLocalGoal.STABLE_CATCH)
    ball_position = env.scene["ball"].data.root_pos_w.torch
    ball_velocity = env.scene["ball"].data.root_lin_vel_w.torch
    ball_position[:, 2] = 0.12
    ball_velocity[:, 2] = 0.45
    cfg = RewardTermCfg(func=mdp.JugglePhysicalProgressReward, weight=1.0)
    term = mdp.JugglePhysicalProgressReward(cfg, env)

    term.reset()
    # First manager evaluation is a settling re-baseline, not authored credit.
    torch.testing.assert_close(term(env), torch.zeros(1))

    ball_position[:, 2] = 0.06
    ball_velocity[:, 2] = 0.225
    positive_rate = term(env)
    torch.testing.assert_close(positive_rate * env.step_dt, torch.tensor((0.39,)), atol=1.0e-6, rtol=0.0)

    ball_position[:, 2] = 0.12
    ball_velocity[:, 2] = 0.45
    negative_rate = term(env)
    torch.testing.assert_close(negative_rate * env.step_dt, torch.tensor((-0.39,)), atol=1.0e-6, rtol=0.0)

    ball_position[:, 2] = 0.06
    ball_velocity[:, 2] = 0.225
    term(env)
    env.termination_manager.dropped[:] = True
    terminal_rate = term(env)
    # Terminal zero potential repays the complete 0.64 potential on this step.
    torch.testing.assert_close(terminal_rate * env.step_dt, torch.tensor((-0.64,)), atol=1.0e-6, rtol=0.0)
    assert torch.isfinite(terminal_rate).all()


def test_progress_reward_rebases_a_completed_continuous_cycle_without_a_stage_drop(fake_tool_state):
    """Catch-to-relaunch bookkeeping must not create an artificial negative shaping pulse."""
    env = _fake_env(1)
    env.juggle_runtime_state.canonical_start[:] = True
    env.juggle_runtime_state.new_cycle_success[:] = True
    env.juggle_runtime_state.current_phases[:] = int(mdp.JugglePhase.HELD_PRETHROW)
    cfg = RewardTermCfg(func=mdp.JugglePhysicalProgressReward, weight=1.0)
    term = mdp.JugglePhysicalProgressReward(cfg, env)
    term._previous[:] = 0.95
    term._baseline_valid[:] = True
    term._skip_next[:] = False

    torch.testing.assert_close(term(env, gamma=0.998), torch.zeros(1))
    torch.testing.assert_close(term._previous, torch.zeros(1))


def test_discount_and_subset_reset_are_exact_and_validate_configuration(fake_tool_state):
    env = _fake_env(2)
    env.juggle_runtime_state.local_goal_ids[:] = int(mdp.JuggleLocalGoal.STABLE_CATCH)
    env.scene["ball"].data.root_pos_w.torch[:, 2] = 0.12
    env.scene["ball"].data.root_lin_vel_w.torch[:, 2] = 0.45
    cfg = RewardTermCfg(func=mdp.JugglePhysicalProgressReward, weight=1.0)
    term = mdp.JugglePhysicalProgressReward(cfg, env)

    term.reset(env_ids=torch.tensor((0,)))
    torch.testing.assert_close(term(env, gamma=0.998), torch.zeros(2))
    discounted_rate = term(env, gamma=0.998)
    torch.testing.assert_close(
        discounted_rate * env.step_dt,
        torch.full((2,), -0.0005),
        atol=1.0e-7,
        rtol=0.0,
    )

    with pytest.raises(ValueError, match="gamma"):
        term(env, gamma=0.0)
    with pytest.raises(ValueError, match="Unknown termination"):
        term(env, drop_termination_name="missing")
    with pytest.raises(ValueError, match="target_height_gain"):
        mdp.juggle_physical_progress_potential(env, target_height_gain=0.0)
