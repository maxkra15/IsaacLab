# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for manager-based RL environments."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumManager, CurriculumTermCfg

pytestmark = pytest.mark.unit


def _record_step_curriculum(env, env_ids):
    env.step_call_order.append(("curriculum", env.common_step_counter))
    return float(env.common_step_counter)


def _make_env_with_policy_obs_terms(
    num_envs: int,
    terms: list[tuple[str, tuple[int, ...], tuple[float, float] | None]],
) -> ManagerBasedRLEnv:
    """Build an uninitialized env whose observation manager stubs drive space setup.

    Args:
        num_envs: Number of vectorized environments.
        terms: Non-concatenated policy observation terms as ``(name, shape, clip)``
            where ``clip`` is ``(low, high)`` or ``None``.

    Returns:
        An uninitialized :class:`~isaaclab.envs.ManagerBasedRLEnv` ready for
        :meth:`~isaaclab.envs.ManagerBasedRLEnv._configure_gym_env_spaces`.
    """
    term_names = [name for name, _, _ in terms]
    term_dims = [shape for _, shape, _ in terms]
    term_cfgs = [SimpleNamespace(clip=clip) for _, _, clip in terms]

    env = object.__new__(ManagerBasedRLEnv)
    env._is_closed = True
    env.scene = SimpleNamespace(num_envs=num_envs)
    env.observation_manager = SimpleNamespace(
        active_terms={"policy": term_names},
        group_obs_concatenate={"policy": False},
        group_obs_dim={"policy": term_dims},
        _group_obs_term_cfgs={"policy": term_cfgs},
    )
    env.action_manager = SimpleNamespace(action_term_dim=[0])
    return env


def test_non_concatenated_obs_groups_contain_all_terms():
    """Non-concatenated observation groups expose every term in the Dict space (issue #3133).

    Before the fix, only the last term in each non-concatenated group would be present
    in the observation space Dict. This test ensures all terms are correctly included.
    """
    terms = [
        ("image", (4, 5, 3), (0.0, 1.0)),
        ("matrix", (2, 3), (-2.0, 2.0)),
        ("vector", (3,), None),
    ]
    env = _make_env_with_policy_obs_terms(num_envs=2, terms=terms)
    ManagerBasedRLEnv._configure_gym_env_spaces(env)

    assert isinstance(env.observation_space, gym.spaces.Dict)
    policy_space = env.observation_space.spaces["policy"]
    assert isinstance(policy_space, gym.spaces.Dict)

    expected_policy_terms = ["image", "matrix", "vector"]
    assert list(policy_space.spaces) == expected_policy_terms
    for term_name in expected_policy_terms:
        assert isinstance(policy_space.spaces[term_name], gym.spaces.Box)


def test_obs_space_follows_clip_constraint():
    """Observation space bounds reflect the clip constraint on each non-concatenated term."""
    terms = [
        ("vector", (3,), None),
        ("matrix", (2, 3), (-2.0, 2.0)),
        ("image", (4, 5, 3), (0.0, 1.0)),
    ]
    expected_shapes = {
        "vector": (2, 3),
        "matrix": (2, 2, 3),
        "image": (2, 4, 5, 3),
    }
    term_clips = {name: clip for name, _, clip in terms}
    env = _make_env_with_policy_obs_terms(num_envs=2, terms=terms)
    ManagerBasedRLEnv._configure_gym_env_spaces(env)

    for group_name, group_space in env.observation_space.spaces.items():
        assert isinstance(group_space, gym.spaces.Dict)
        for term_name, term_space in group_space.spaces.items():
            clip = term_clips[term_name]
            low = -np.inf if clip is None else clip[0]
            high = np.inf if clip is None else clip[1]
            assert isinstance(term_space, gym.spaces.Box), (
                f"Expected Box space for {term_name} in {group_name}, got {type(term_space)}"
            )
            assert term_space.shape == expected_shapes[term_name]
            assert np.all(term_space.low == low)
            assert np.all(term_space.high == high)


def test_step_curriculum_runs_before_action_processing_at_current_global_step():
    """Step-triggered curricula update before the action manager reads live configuration."""
    env = object.__new__(ManagerBasedRLEnv)
    call_order: list[tuple[str, int]] = []
    env._is_closed = True
    env.step_call_order = call_order
    env.common_step_counter = 37
    env.episode_length_buf = torch.zeros(1, dtype=torch.long)
    env._sim_step_counter = 0
    env._physics_handles_decimation = True
    env.cfg = SimpleNamespace(decimation=1, sim=SimpleNamespace(dt=0.01, render_interval=1))
    env.action_manager = SimpleNamespace(
        process_action=lambda action: call_order.append(("action", env.common_step_counter)),
        apply_action=lambda: None,
    )
    env.recorder_manager = SimpleNamespace(
        active_terms=(),
        record_pre_step=lambda: None,
        record_post_physics_decimation_step=lambda: None,
    )
    env.sim = SimpleNamespace(
        device="cpu",
        is_rendering=False,
        is_playing=lambda: True,
        step=lambda *, render: None,
        consume_reset_request=lambda: False,
    )
    env.scene = SimpleNamespace(
        num_envs=1,
        write_data_to_sim=lambda: None,
        update=lambda *, dt: None,
    )
    done = torch.zeros(1, dtype=torch.bool)
    env.termination_manager = SimpleNamespace(
        compute=lambda: done,
        terminated=done,
        time_outs=done,
    )
    env.reward_manager = SimpleNamespace(compute=lambda *, dt: torch.zeros(1))
    env.command_manager = SimpleNamespace(compute=lambda *, dt: call_order.append(("command", env.common_step_counter)))
    env.event_manager = SimpleNamespace(available_modes=())
    env.video_recorders = ()
    env.observation_manager = SimpleNamespace(
        compute=lambda **kwargs: (
            call_order.append(("observation", env.common_step_counter)) or {"policy": torch.zeros(1, 1)}
        )
    )
    env.extras = {}
    env.curriculum_manager = CurriculumManager(
        {"step_probe": CurriculumTermCfg(func=_record_step_curriculum, update_mode="step")},
        env,
    )

    ManagerBasedRLEnv.step(env, torch.zeros(1, 1))
    ManagerBasedRLEnv.step(env, torch.zeros(1, 1))

    assert call_order == [
        ("curriculum", 37),
        ("action", 37),
        ("curriculum", 38),
        ("command", 38),
        ("observation", 38),
        # The second pre-action hook at counter 38 is idempotent.
        ("action", 38),
        ("curriculum", 39),
        ("command", 39),
        ("observation", 39),
    ]
    assert env.common_step_counter == 39
