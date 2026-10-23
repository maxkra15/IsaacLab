# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise the actual ensemble and PPO boundaries around artificial resets."""

from argparse import Namespace

import pytest
import torch
from mpm_world_model.dataset import fit_normalizers, load_episodes, task_specs
from mpm_world_model.learned_env import LearnedParticleEnv, bootstrap_truncations
from mpm_world_model.models import CoupledWorldModel, WorldModelConfig
from mpm_world_model.recording import EpisodeWriter
from mpm_world_model.teapot_state import pour_reward
from mpm_world_model.train_policy import ActionBoundedPolicy, default_runner_config, train
from rsl_rl.algorithms import PPO


def _episodes(root):
    writer = EpisodeWriter(root)
    for world in range(20):

        def state(step):
            x = torch.tensor([world * 0.01 + step * 0.1, 0.1])
            return {
                "observation": x,
                "critic_observation": x.clone(),
                "particle_state": torch.zeros(2, 6),
                "rigid_state": torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                "finite_mask": torch.tensor(True),
            }

        writer.start_episode(
            world,
            state(0),
            {
                "task_id": "boundary_test",
                "reset_group": f"row-{world}",
                "context": torch.tensor([0.1]),
            },
        )
        for step in range(3):
            writer.append_transition(world, torch.zeros(1), float(step), False, False, state(step + 1))
        writer.finish_episode(world)
    writer.close()
    return load_episodes(root)


def test_artificial_cut_preserves_predicted_endpoint_and_ppo_bootstrap(tmp_path):
    """A short model rollout bootstraps its final state, never the new replay seed."""
    torch.set_num_threads(2)
    episodes = _episodes(tmp_path)
    model = CoupledWorldModel(WorldModelConfig(task_specs(episodes), hidden_dim=16, entity_dim=8))
    fit_normalizers(model, [episode for episode in episodes if episode["split"] == "train"])
    env = LearnedParticleEnv(
        [model],
        episodes,
        task_id="boundary_test",
        num_envs=8,
        horizon=1,
        seed_bank_size=16,
        device="cpu",
        uncertainty_limit=10,
    )
    initial = env.get_observations()["policy"].clone()
    obs, reward, done, info = env.step(torch.zeros(8, 1))
    assert done.all() and info["truncated"].all() and not info["terminated"].any()
    expected = initial + model.task("boundary_test").observation_delta_norm.mean
    torch.testing.assert_close(info["terminal_observations"]["policy"], expected)
    assert not torch.equal(obs["policy"], expected)
    assert not env.episode_length_buf.any()

    cfg = default_runner_config()
    cfg.update(num_steps_per_env=1, multi_gpu=None)
    cfg["algorithm"].update(num_mini_batches=1, num_learning_epochs=1)
    alg = PPO.construct_algorithm(obs, env, cfg, "cpu")
    with torch.no_grad():
        action = alg.act(obs)
        next_obs, reward, done, info = env.step(action)
        terminal_value = alg.critic(info["terminal_observations"]).squeeze(-1)
        corrected = bootstrap_truncations(reward, info["truncated"], terminal_value, alg.gamma)
        alg.process_env_step(next_obs, corrected, done, {})
        alg.compute_returns(next_obs)
    # With one done transition, the stored return is exactly corrected reward;
    # the critic value of the unrelated reset snapshot contributes nothing.
    torch.testing.assert_close(alg.storage.returns[0, :, 0], corrected)
    torch.testing.assert_close(corrected, reward + alg.gamma * terminal_value)
    # Exercise the same actual RSL export used by the training CLI, including
    # the policy's fitted observation-normalization module.
    native_actor = alg.get_policy().as_jit().cpu().eval()
    scripted_actor = torch.jit.script(ActionBoundedPolicy(native_actor, 0.1))
    with torch.no_grad():
        torch.testing.assert_close(scripted_actor(initial), native_actor(initial).clamp(-0.1, 0.1))
    assert not torch.cuda.is_initialized()


def test_invalid_termination_head_is_rejected_before_ppo(tmp_path):
    """A nonfinite auxiliary prediction cannot silently continue an imagination."""
    episodes = _episodes(tmp_path)
    model = CoupledWorldModel(WorldModelConfig(task_specs(episodes), hidden_dim=16, entity_dim=8))
    fit_normalizers(model, [episode for episode in episodes if episode["split"] == "train"])
    with torch.no_grad():
        model.task("boundary_test").output[-1].bias[-1] = float("nan")
    env = LearnedParticleEnv(
        [model],
        episodes,
        task_id="boundary_test",
        num_envs=4,
        horizon=8,
        seed_bank_size=4,
        device="cpu",
    )
    obs, reward, done, info = env.step(torch.zeros(4, 1))
    assert done.all() and info["nonfinite"].all() and info["terminated"].all()
    assert not info["truncated"].any()
    assert torch.isfinite(obs["policy"]).all()
    # Invalid physical terminals are not bootstrapped, including NaN values in
    # the discarded branch: multiplying NaN by a zero mask would poison PPO.
    corrected = bootstrap_truncations(reward, info["truncated"], torch.full_like(reward, float("nan")), 0.99)
    torch.testing.assert_close(corrected, reward)


def test_teapot_imagination_uses_the_native_mass_reward(tmp_path):
    """The policy reward follows delivered/spilled mass rather than a reward head."""
    episodes = _episodes(tmp_path)
    for episode in episodes:
        episode["task_id"] = "rizon_sharpa_teapot_held_pour"
        count = len(episode["observation"])
        episode["dt"] = 0.01
        observation = torch.zeros(count, 121)
        observation[:, [61, 74]] = 1.0
        observation[:, 91] = 1.0 - torch.arange(count) * 0.1
        observation[:, 92] = torch.arange(count) * 0.1
        observation[:, 118] = torch.arange(count) * 0.01
        observation[:, 119] = 1.0 - torch.arange(count) * 0.01 / 16.0
        observation[:, 120] = 0.5
        episode["observation"] = observation
        episode["critic_observation"] = observation.clone()
        episode["action"] = torch.zeros(count - 1, 6)
        episode["reward"] = torch.full((count - 1,), 10.0)
    model = CoupledWorldModel(
        WorldModelConfig(task_specs(episodes), hidden_dim=16, entity_dim=8, physical_decoder="typed")
    )
    fit_normalizers(model, [episode for episode in episodes if episode["split"] == "train"])
    env = LearnedParticleEnv(
        [model],
        episodes,
        task_id=episodes[0]["task_id"],
        num_envs=4,
        horizon=1,
        seed_bank_size=4,
        device="cpu",
        action_limit=0.1,
    )
    env.reward_min, env.reward_max = -1000.0, 1000.0
    previous = env.get_observations()["policy"].clone()
    actions = torch.full((4, 6), 0.5)
    _, reward, _, info = env.step(actions)
    endpoint = info["terminal_observations"]["policy"]
    torch.testing.assert_close(endpoint[:, 112:118], actions.clamp(-0.1, 0.1))
    expected = torch.stack(
        [
            pour_reward(before[84:106], after[84:106], action)
            for before, after, action in zip(previous, endpoint, actions.clamp(-0.1, 0.1))
        ]
    )
    torch.testing.assert_close(reward, expected)
    assert not info["inadmissible"].any()
    for limit in (0.0, 1.1):
        with pytest.raises(ValueError, match="Action limit"):
            LearnedParticleEnv([model], episodes, task_id=episodes[0]["task_id"], device="cpu", action_limit=limit)


def test_teapot_native_timeout_preserves_endpoint_and_finite_precedence(tmp_path):
    """Truncate at native clock limits before replay reset, preserving real failures."""
    episodes = _episodes(tmp_path)
    for episode in episodes:
        count = len(episode["observation"])
        observation = torch.zeros(count, 121)
        observation[:, [61, 74, 91]] = 1.0
        observation[:, 118] = torch.arange(count) * 0.01
        observation[:, 119] = 1.0 - torch.arange(count) * 0.01 / 16.0
        observation[:, 120] = 0.5
        episode.update(
            task_id="rizon_sharpa_teapot_held_pour",
            dt=0.01,
            observation=observation,
            critic_observation=observation.clone(),
            action=torch.zeros(count - 1, 6),
        )
    task_id = episodes[0]["task_id"]
    model = CoupledWorldModel(
        WorldModelConfig(task_specs(episodes), hidden_dim=16, entity_dim=8, physical_decoder="typed")
    )
    env = LearnedParticleEnv([model], episodes, task_id=task_id, num_envs=5, horizon=8, seed_bank_size=8, device="cpu")
    initial_clocks = torch.tensor([[15.99, 0.000625], [15.99, 0.5], [1.0, 0.000625], [1.0, 0.5], [15.99, 0.000625]])
    for values in (env.observations, env.critics):
        values[0][:, 118:120] = initial_clocks
    # A numerical failure at the same endpoint remains a non-bootstrapped
    # terminal, rather than being hidden by the known native timeout.
    env.observations[0][4, 0] = float("nan")
    observations, reward, done, info = env.step(torch.zeros(5, 6))
    assert done.tolist() == [True, True, True, False, True]
    assert info["truncated"].tolist() == [True, True, True, False, False]
    assert info["terminated"].tolist() == [False, False, False, False, True]
    assert info["native_timeout"].tolist() == [True, True, True, False, False]
    endpoint = info["terminal_observations"]["policy"]
    torch.testing.assert_close(endpoint[:4, 118], initial_clocks[:4, 0] + 0.01)
    torch.testing.assert_close(endpoint[:4, 119], (initial_clocks[:4, 1] - 0.01 / 16.0).clamp(0.0, 1.0))
    assert (observations["policy"][[0, 1, 2, 4], 118] < 0.1).all()
    assert env.episode_length_buf.tolist() == [0, 0, 0, 1, 0]
    endpoint_value = torch.tensor([1.0, 2.0, 3.0, 4.0, float("nan")])
    corrected = bootstrap_truncations(reward, info["truncated"], endpoint_value, 0.99)
    torch.testing.assert_close(corrected[:4], reward[:4] + torch.tensor([0.99, 1.98, 2.97, 0.0]))
    torch.testing.assert_close(corrected[4], reward[4])


def test_exported_action_limit_bounds_outputs_without_changing_raw_policy(tmp_path):
    """Persist the executed actuator limit in TorchScript, retaining raw weights."""
    actor = torch.nn.Linear(2, 2)
    with torch.no_grad():
        actor.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, -2.0]]))
        actor.bias.zero_()
    observation = torch.tensor([[0.5, 0.25], [-0.05, -0.02], [0.0, 0.0]])
    expected_raw = torch.tensor([[0.5, -0.5], [-0.05, 0.04], [0.0, 0.0]])
    path = tmp_path / "bounded_policy.pt"
    torch.jit.script(ActionBoundedPolicy(actor, 0.1)).save(str(path))
    restored = torch.jit.load(str(path))
    torch.testing.assert_close(restored(observation), expected_raw.clamp(-0.1, 0.1))
    torch.testing.assert_close(actor(observation), expected_raw)
    torch.testing.assert_close(ActionBoundedPolicy(actor)(torch.tensor([[2.0, -3.0]])), torch.ones(1, 2))
    for limit in (0.0, -0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="Action limit"):
            ActionBoundedPolicy(actor, limit)
    # The native Franka consumer loads the raw RSL checkpoint, so accepting a
    # JIT-only smaller bound would change imagined and native action semantics.
    with pytest.raises(ValueError, match="Franka native RSL"):
        train(Namespace(task="franka_pour", action_limit=0.1))
