# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise offline policy fitting and the actual RSL/native deployment contract."""

import hashlib
import json
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from mpm_world_model.learned_env import observation_groups
from mpm_world_model.recording import EpisodeWriter
from mpm_world_model.train_imitation import train
from mpm_world_model.train_policy import default_runner_config
from rsl_rl.algorithms import PPO


def _demonstrations(root):
    writer = EpisodeWriter(root)
    validation = None
    train_observation = None
    for world, split in enumerate(("train", "validation", "test")):
        count = 256 if split == "train" else 128
        generator = torch.Generator().manual_seed(101 + world)
        observation = torch.rand(count + 1, 3, generator=generator) * 2 - 1
        # A nonlinear, stateless teacher: previous-action persistence is not an
        # oracle for this mapping, and validation contains unseen inputs.
        actions = torch.stack(
            (0.7 * torch.sin(2 * observation[:-1, 0]), 0.6 * observation[:-1, 1] * observation[:-1, 2]), -1
        )
        if split == "train":
            train_observation = observation[: count - 17].clone()
            observation[-16:] = 10000
            actions[-16:] = 10000
        if split == "validation":
            validation = (observation[:-1].clone(), actions.clone())
        if split == "test":
            # Holdout must not affect normalization, fitting, or selection.
            observation[:] = float("nan")
            actions[:] = float("nan")

        def state(step):
            return {
                "observation": observation[step],
                "critic_observation": observation[step].clone(),
                "particle_state": torch.zeros(1, 6),
                "rigid_state": torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                "finite_mask": torch.tensor(True),
                "rigid_capacity_valid": torch.tensor(split != "train" or step < count - 16),
            }

        writer.start_episode(
            world,
            state(0),
            {"task_id": "imitation_boundary", "reset_group": f"split-{split}", "context": torch.tensor([0.1])},
        )
        for step in range(count):
            writer.append_transition(world, actions[step], 0.0, False, False, state(step + 1))
        writer.finish_episode(world)
        writer.manifest["episodes"][-1]["split"] = split
    writer.close()
    return train_observation, validation


def test_imitation_generalizes_and_preserves_native_checkpoint_contract(tmp_path):
    """Learn unseen teacher actions without fitting invalid or holdout states."""
    train_observation, (validation_obs, validation_action) = _demonstrations(tmp_path / "data")
    cfg = default_runner_config()
    cfg["actor"]["hidden_dims"] = cfg["critic"]["hidden_dims"] = [32, 32]
    runner = tmp_path / "runner.json"
    runner.write_text(json.dumps(cfg))
    output = tmp_path / "policy"
    result = train(
        Namespace(
            data=[str(tmp_path / "data")],
            output=str(output),
            task="imitation_boundary",
            runner_config=str(runner),
            device="cpu",
            updates=400,
            batch_size=64,
            validate_every=100,
            learning_rate=0.003,
            action_limit=0.8,
            observation_mode="full",
            threads=2,
            seed=11,
        )
    )
    selected = result["selected_validation"]
    assert selected["action_rmse"] < 0.06
    assert selected["action_rmse"] < result["initial_validation"]["action_rmse"] * 0.15
    assert selected["action_permutation_rmse"] > selected["action_rmse"] * 5
    assert selected["previous_action_rmse"] > selected["action_rmse"] * 5
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["transition_counts"]["train"]["valid"] == len(train_observation)
    assert provenance["normalization_split"] == "train" and not provenance["test_split_evaluated"]
    assert (
        provenance["datasets"][0]["manifest_sha256"]
        == hashlib.sha256((tmp_path / "data" / "manifest.json").read_bytes()).hexdigest()
    )

    # Strictly reload both native RSL modules as the subsequent PPO entrypoint
    # does. The critic is deliberately untrained, with a fresh PPO optimizer.
    saved = torch.load(output / "model_imitation.pt", map_location="cpu", weights_only=False)
    obs = observation_groups(validation_obs, validation_obs, "imitation_boundary")
    restored_cfg = json.loads((output / "runner_cfg.json").read_text())
    alg = PPO.construct_algorithm(obs[:1], SimpleNamespace(num_envs=1, num_actions=2), restored_cfg, "cpu")
    alg.load(saved, {"actor": True, "critic": True}, strict=True)
    alg.eval_mode()
    torch.testing.assert_close(alg.actor.obs_normalizer.mean, train_observation.mean(0))
    assert int(alg.actor.obs_normalizer.count) == len(train_observation)
    assert saved["optimizer_state_dict"]["state"] == {}
    with torch.no_grad():
        native = alg.actor(obs).clamp(-0.8, 0.8)
        scripted = torch.jit.load(str(output / "policy.pt"))
        torch.testing.assert_close(scripted(validation_obs), native)
        assert float((native - validation_action).square().mean().sqrt()) == pytest.approx(selected["action_rmse"])
        assert scripted(torch.full((1, 3), 10000.0)).abs().max() <= 0.8
    assert not torch.cuda.is_initialized()


def test_imitation_rejects_reset_group_leakage_across_manifests(tmp_path):
    """Separate individually valid manifests cannot split one reset group."""
    for name in ("first", "second"):
        _demonstrations(tmp_path / name)
    manifest_path = tmp_path / "second" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["episodes"] = [manifest["episodes"][0]]
    manifest["episodes"][0]["split"] = "validation"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Reset groups cannot cross"):
        train(
            Namespace(
                data=[str(tmp_path / "first"), str(tmp_path / "second")],
                output=str(tmp_path / "policy"),
                task="imitation_boundary",
                runner_config=None,
                device="cpu",
                updates=1,
                batch_size=1,
                validate_every=1,
                learning_rate=0.001,
                action_limit=1.0,
                observation_mode="full",
                threads=2,
                seed=0,
            )
        )


@pytest.mark.parametrize("mode", ["teapot_reduced", "teapot_phase"])
def test_masked_teapot_policy_ignores_excluded_fields_after_native_reload(tmp_path, mode):
    """Masked feedback stays inert while retained pose or clock drives learned actions."""
    data = tmp_path / "data"
    _demonstrations(data)
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["episodes"]:
        path = data / entry["file"]
        episode = torch.load(path, weights_only=False)
        old_observation = episode["observation"]
        count = len(episode["action"])
        action = torch.zeros(count, 30)
        action[:, 0] = (
            0.6 * (torch.arange(count) * 0.01 - 0.8) if mode == "teapot_phase" else 0.4 * old_observation[:-1, 0].tanh()
        )
        action[:, 29] = -1
        observation = torch.zeros(count + 1, 182)
        observation[:, :3] = old_observation
        observation[:, 29:58] = old_observation[:, :1]
        observation[:, [61, 74, 91, 169]] = 1
        observation[:, 106:135] = old_observation[:, 1:2]
        observation[1:, 135:165] = action
        observation[:, 165] = 24
        observation[:, 173:179] = old_observation[:, 2:3]
        observation[:, 179] = torch.arange(count + 1) * 0.01
        observation[:, 180] = 1 - observation[:, 179] / 18
        observation[:, 181] = 0.5
        episode.update(
            task_id="rizon_sharpa_teapot_full",
            dt=0.01,
            observation=observation,
            critic_observation=observation.clone(),
            controller_state=observation[:, 106:166].clone(),
            action=action,
            observation_names=[f"feature_{index}" for index in range(182)],
        )
        entry["task_id"] = episode["task_id"]
        torch.save(episode, path)
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    cfg = default_runner_config()
    cfg["actor"]["hidden_dims"] = cfg["critic"]["hidden_dims"] = [16, 16]
    runner = tmp_path / "runner.json"
    runner.write_text(json.dumps(cfg))
    output = tmp_path / mode
    train(
        Namespace(
            data=[str(data)],
            output=str(output),
            task="rizon_sharpa_teapot_full",
            runner_config=str(runner),
            device="cpu",
            updates=300,
            batch_size=64,
            validate_every=100,
            learning_rate=0.003,
            action_limit=1.0,
            observation_mode=mode,
            threads=2,
            seed=3,
        )
    )
    scripted = torch.jit.load(str(output / "policy.pt"))
    probe = torch.zeros(2, 182)
    probe[:, 0] = torch.tensor([-0.8, 0.8])
    probe[:, [61, 74, 91, 169]] = 1
    probe[:, 165], probe[:, 180], probe[:, 181] = 24, 1, 0.5
    if mode == "teapot_phase":
        probe[:, 179] = torch.tensor([0.2, 1.2])
        probe[:, 180] = 1 - probe[:, 179] / 18
        excluded = [index for index in range(182) if index != 179]
    else:
        excluded = [*range(29, 58), *range(106, 166), *range(173, 179)]
    perturbed = probe.clone()
    perturbed[:, excluded] = torch.linspace(-1000, 1000, len(excluded))
    with torch.no_grad():
        prediction = scripted(probe)
        torch.testing.assert_close(scripted(perturbed), prediction, atol=0, rtol=0)
        assert prediction[1, 0] - prediction[0, 0] > 0.3
    saved = torch.load(output / "model_imitation.pt", weights_only=False)
    observations = observation_groups(probe, probe, "rizon_sharpa_teapot_full")
    alg = PPO.construct_algorithm(
        observations[:1],
        SimpleNamespace(num_envs=1, num_actions=30),
        json.loads((output / "runner_cfg.json").read_text()),
        "cpu",
    )
    alg.load(saved, {"actor": True, "critic": True}, strict=True)
    alg.eval_mode()
    with torch.no_grad():
        torch.testing.assert_close(alg.actor(observations).clamp(-1, 1), prediction)
