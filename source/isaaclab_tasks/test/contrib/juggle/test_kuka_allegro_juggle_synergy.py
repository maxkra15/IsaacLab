# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-PPO and hand-synergy contracts for one-metre KUKA-Allegro juggling."""

from types import SimpleNamespace

import pytest
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.juggle import mdp
from isaaclab_tasks.contrib.juggle.config.kuka_allegro.agents.rsl_rl_ppo_cfg import (
    JuggleGaussianDistribution,
    KukaAllegroJugglePPORunnerCfg,
)
from isaaclab_tasks.contrib.juggle.mdp.actions import (
    JuggleHandSynergyAction,
    JuggleTaskSpaceTranslationAction,
)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

TASK_NAME = "IsaacContrib-Juggle-Ball-KukaAllegro-RL"


def test_juggle_uses_compact_physical_actions_and_from_scratch_ppo():
    """Juggle PPO uses three palm translations plus one hand aperture."""
    cfg = parse_env_cfg(TASK_NAME, device="cpu", num_envs=8)
    runner = load_cfg_from_registry(TASK_NAME, "rsl_rl_cfg_entry_point")

    assert isinstance(cfg.actions.hand_action, mdp.JuggleHandSynergyActionCfg)
    assert isinstance(cfg.actions.arm_action, mdp.JuggleTaskSpaceTranslationActionCfg)
    assert len(cfg.actions.arm_action.joint_names) == 7
    assert len(cfg.actions.hand_action.joint_names) == 16
    assert cfg.actions.arm_action.scale == cfg.actions.arm_action.max_delta == 2.0
    assert cfg.actions.arm_action.body_name == "palm_link"
    assert cfg.actions.arm_action.tool_offset == mdp.JUGGLE_SPHERE_CENTER_OFFSET
    assert cfg.actions.arm_action.damping == 2.5e-3
    assert cfg.actions.hand_action.scale == cfg.actions.hand_action.max_delta == 0.10
    assert cfg.observations.policy.actions.func is mdp.last_action

    preload = torch.tensor(mdp.JUGGLE_SPHERE_PRELOAD_HAND_POSITION)
    opened = torch.tensor(mdp.JUGGLE_SPHERE_OPEN_HAND_POSITION)
    expected_direction = opened - preload
    expected_direction /= expected_direction.abs().max()
    directions = torch.tensor(cfg.actions.hand_action.joint_directions)
    torch.testing.assert_close(directions, expected_direction)
    assert directions.abs().max().item() == 1.0
    assert torch.dot(directions, opened - preload).item() > 0.0

    assert isinstance(runner, KukaAllegroJugglePPORunnerCfg)
    assert runner.num_steps_per_env == 192
    assert runner.save_interval == 10
    assert not runner.actor.obs_normalization
    assert not runner.critic.obs_normalization
    assert runner.actor.distribution_cfg.arm_action_dim == 3
    assert runner.actor.distribution_cfg.init_std == 0.30
    assert runner.actor.distribution_cfg.hand_init_std == 0.40
    assert runner.actor.distribution_cfg.std_range == (0.002, 0.50)
    assert runner.algorithm.entropy_coef == 1.0e-3
    assert runner.algorithm.learning_rate == 5.0e-5
    assert runner.algorithm.num_learning_epochs == 5
    assert runner.algorithm.num_mini_batches == 8

    distribution = JuggleGaussianDistribution(
        output_dim=4,
        init_std=0.30,
        hand_init_std=0.40,
        arm_action_dim=3,
        std_range=(0.002, 0.50),
    )
    distribution.update(torch.zeros((1, 4)))
    torch.testing.assert_close(distribution.std[0, :3], torch.full((3,), 0.30))
    torch.testing.assert_close(distribution.std[0, 3:], torch.full((1,), 0.40))
    with torch.no_grad():
        distribution.std_param.fill_(-100.0)
    distribution.update(torch.zeros((1, 4)))
    assert 0.0019 < distribution.std.max().item() < 0.015


def test_task_space_translation_matches_the_vetted_dynamic_pose_dls_direction():
    """The analytic action reproduces the live controller without storing its direction."""
    joint_positions = torch.tensor(
        (
            (
                2.0526866912841797,
                1.2092933654785156,
                -2.0302281379699707,
                1.411957859992981,
                -2.5589497089385986,
                1.8656471967697144,
                -2.5877718925476074,
            ),
        ),
        dtype=torch.float64,
    )
    translation = torch.tensor(((0.0, 0.46, 1.0),), dtype=torch.float64)
    expected = torch.tensor(
        ((0.4419886768, -0.6677613854, 1.0, -0.1431256384, 0.5499856472, -0.0276448224, -0.0482892953),),
        dtype=torch.float64,
    )

    joint_action = mdp.normalized_kuka_allegro_translation_joint_action(
        joint_positions,
        translation,
        tool_offset=mdp.JUGGLE_SPHERE_CENTER_OFFSET,
        damping=2.5e-3,
    )

    torch.testing.assert_close(joint_action, expected, atol=2.0e-7, rtol=0.0)
    torch.testing.assert_close(joint_action.abs().amax(dim=1), translation.abs().amax(dim=1))


def test_task_space_translation_preserves_xyz_magnitude_and_zero_holds():
    """Each environment keeps its own XYZ magnitude and an exact zero target."""
    joint_position = torch.tensor(
        (
            2.0526866912841797,
            1.2092933654785156,
            -2.0302281379699707,
            1.411957859992981,
            -2.5589497089385986,
            1.8656471967697144,
            -2.5877718925476074,
        )
    )
    joint_positions = joint_position.repeat(3, 1)
    translations = torch.tensor(((0.0, 0.0, 0.0), (0.0, 0.23, 0.50), (-0.17, 0.08, 0.04)))
    joint_actions = mdp.normalized_kuka_allegro_translation_joint_action(
        joint_positions,
        translations,
        tool_offset=mdp.JUGGLE_SPHERE_CENTER_OFFSET,
        damping=2.5e-3,
    )
    torch.testing.assert_close(joint_actions[0], torch.zeros(7))
    torch.testing.assert_close(
        joint_actions.abs().amax(dim=1),
        translations.abs().amax(dim=1),
        atol=1.0e-6,
        rtol=0.0,
    )

    action = object.__new__(JuggleTaskSpaceTranslationAction)
    action._env = SimpleNamespace(num_envs=2, device="cpu")
    action.cfg = SimpleNamespace(clip=None, joint_limit_margin=0.02, max_delta=2.0)
    action._joint_ids = slice(None)
    action._asset = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=SimpleNamespace(torch=joint_positions[:2]),
            soft_joint_pos_limits=SimpleNamespace(torch=torch.tensor((-4.0, 4.0)).repeat(2, 7, 1)),
        )
    )
    action._workspace_lower = torch.tensor(mdp.KUKA_ALLEGRO_JUGGLE_ARM_WORKSPACE_LOWER)
    action._workspace_upper = torch.tensor(mdp.KUKA_ALLEGRO_JUGGLE_ARM_WORKSPACE_UPPER)
    action._scale = 2.0
    action._offset = 0.0
    action._raw_actions = torch.zeros((2, 3))
    action._processed_actions = torch.zeros((2, 7))
    action._position_targets = joint_positions[:2].clone()
    action._tool_offset = mdp.JUGGLE_SPHERE_CENTER_OFFSET
    action._damping = 2.5e-3
    commands = translations[:2]

    action.process_actions(commands)

    assert action.action_dim == 3
    torch.testing.assert_close(action.raw_actions, commands)
    torch.testing.assert_close(action.processed_actions[0], joint_positions[0])
    target_delta = action.processed_actions[1] - joint_positions[1]
    torch.testing.assert_close(target_delta, joint_actions[1] * 2.0, atol=2.0e-6, rtol=0.0)
    assert target_delta.abs().max().item() == pytest.approx(1.0, abs=2.0e-6)


def test_hand_synergy_zero_holds_preload_and_signed_commands_open_and_close():
    """The scalar interface preserves a safe zero command and the calibrated aperture sign."""
    preload = torch.tensor(mdp.JUGGLE_SPHERE_PRELOAD_HAND_POSITION)
    opened = torch.tensor(mdp.JUGGLE_SPHERE_OPEN_HAND_POSITION)
    direction = opened - preload
    direction /= direction.abs().max()
    num_envs = 3

    env = SimpleNamespace(num_envs=num_envs, device="cpu")
    mdp.create_juggle_runtime_state(env)
    action = object.__new__(JuggleHandSynergyAction)
    action._env = env
    action.cfg = SimpleNamespace(
        clip=None,
        joint_limit_margin=0.02,
        max_delta=0.10,
        preload_release_steps=1,
        preload_release_threshold=0.01,
        release_preload_after_first_action=True,
    )
    action._joint_ids = slice(None)
    joint_positions = torch.stack((preload, preload, opened))
    action._asset = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=SimpleNamespace(torch=joint_positions),
            soft_joint_pos_limits=SimpleNamespace(torch=torch.tensor((-2.0, 2.0)).repeat(num_envs, preload.numel(), 1)),
        )
    )
    action._pair_reset_preload_commands = preload.unsqueeze(0)
    action._pair_reset_open_commands = opened.unsqueeze(0)
    action._joint_directions = direction
    action._scale = 0.10
    action._offset = 0.0
    action._raw_actions = torch.zeros((num_envs, 1))
    action._processed_actions = torch.zeros_like(joint_positions)
    action._position_targets = joint_positions.clone()
    action._preload_assist_active = torch.tensor((True, True, False))
    action._preload_open_intent_steps = torch.zeros(num_envs, dtype=torch.long)

    commands = torch.tensor(((0.0,), (1.0,), (-1.0,)))
    action.process_actions(commands)

    assert action.action_dim == 1
    torch.testing.assert_close(action.raw_actions, commands)
    torch.testing.assert_close(action.processed_actions[0], preload)
    opening_delta = action.processed_actions[1] - preload
    closing_delta = action.processed_actions[2] - opened
    assert torch.dot(opening_delta, direction).item() > 0.0
    assert torch.dot(closing_delta, direction).item() < 0.0
    assert opening_delta.abs().max().item() <= 0.100001
    assert closing_delta.abs().max().item() <= 0.100001
    assert not action._preload_assist_active.any()
