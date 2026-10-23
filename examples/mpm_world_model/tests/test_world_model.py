# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Protect dataset isolation, entity invariance, and real autoregressive learning."""

import hashlib
import json

import pytest
import torch
from mpm_world_model.dataset import EpisodeWindows, fit_normalizers, load_episodes, task_specs
from mpm_world_model.evaluate_world_model import physical_observation_errors
from mpm_world_model.models import (
    CoupledWorldModel,
    TaskSpec,
    WorldModelConfig,
    checkpoint_payload,
    load_checkpoint,
    rollout_loss,
)
from mpm_world_model.train_world_model import evaluate_models


def _episode(seed, *, transitions=16):
    generator = torch.Generator().manual_seed(seed)
    action = torch.randn(transitions, 1, generator=generator).clamp(-1, 1)
    x = torch.randn((), generator=generator) * 0.2
    velocity = torch.randn((), generator=generator) * 0.2
    states = [torch.stack((x, velocity))]
    for u in action:
        velocity = 0.8 * velocity + 0.25 * u[0]
        x = x + 0.1 * velocity
        states.append(torch.stack((x, velocity)))
    observation = torch.stack(states)
    particles = torch.zeros(transitions + 1, 2, 6)
    particles[:, :, 0] = observation[:, :1] + torch.tensor([[-0.01, 0.01]])
    particles[:, :, 3] = observation[:, 1:]
    rigid = torch.zeros(transitions + 1, 1, 13)
    rigid[:, 0, 0] = 0.5 * observation[:, 0]
    rigid[:, 0, 3] = 1.0
    return {
        "task_id": "analytic_damped_motion",
        "reset_group": f"initial-state-{seed}",
        "observation": observation,
        "critic_observation": torch.cat((observation, torch.ones(transitions + 1, 1)), -1),
        "action": action,
        "reward": -observation[1:, 0].square() - 0.1 * action[:, 0].square(),
        "terminated": torch.zeros(transitions, dtype=torch.bool),
        "truncated": torch.zeros(transitions, dtype=torch.bool),
        "particle_state": particles,
        "rigid_state": rigid,
        "controller_state": torch.cat((torch.zeros(1, 1), action), 0),
        "context": torch.tensor([0.1]),
    }


def _write(root, episodes):
    root.mkdir(exist_ok=True)
    entries = []
    for index, (episode, split) in enumerate(episodes):
        path = root / f"episode-{index}.pt"
        torch.save(episode, path)
        entries.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "split": split})
    (root / "manifest.json").write_text(json.dumps({"episodes": entries, "complete": True}))
    return load_episodes(root)


def test_reset_group_isolation_and_terminal_windows(tmp_path):
    """Reject held-out reset leakage and retain a finite terminal transition once."""
    train = _episode(1, transitions=3)
    train["terminated"][-1] = True
    train["observation"][0] = torch.tensor([2.0, 4.0])
    validation = _episode(2, transitions=3)
    validation["observation"].fill_(1e6)
    episodes = _write(tmp_path, [(train, "train"), (validation, "validation")])
    model = CoupledWorldModel(WorldModelConfig(task_specs(episodes), hidden_dim=16, entity_dim=8))
    with pytest.raises(ValueError, match="training episodes only"):
        fit_normalizers(model, episodes)
    fit_normalizers(model, episodes[:1])
    assert model.task(train["task_id"]).observation_norm.mean.max() < 5.0
    windows = EpisodeWindows(episodes, task_id=train["task_id"], split="train", horizon=8)
    batch = windows.sample(16)
    for mask, terminated in zip(batch["transition_mask"], batch["terminated"]):
        count = int(mask.sum())
        assert count <= 3
        assert count >= 1
        assert terminated[count - 1]
        assert not mask[count:].any()
    leaked = dict(validation, reset_group=train["reset_group"])
    with pytest.raises(ValueError, match="different dataset splits"):
        _write(tmp_path, [(train, "train"), (leaked, "validation")])
    crossing = _episode(3, transitions=3)
    crossing["terminated"][0] = True
    with pytest.raises(ValueError, match="autoreset/terminal boundary"):
        _write(tmp_path, [(crossing, "train")])


def test_entity_permutation_and_padding_do_not_change_predictions(tmp_path):
    """Preserve predictions under particle reordering and masked invalid slots."""
    torch.manual_seed(5)
    episodes = _write(tmp_path, [(_episode(5), "train")])
    model = CoupledWorldModel(WorldModelConfig(task_specs(episodes), hidden_dim=16, entity_dim=8))
    fit_normalizers(model, episodes)
    batch = EpisodeWindows(episodes, task_id=episodes[0]["task_id"], split="train", horizon=3).sample(4)
    with torch.no_grad():
        expected = model.encode_batch(batch, task_id=episodes[0]["task_id"])
        permuted = {key: value.clone() for key, value in batch.items()}
        permuted["particle_state"] = permuted["particle_state"][:, :, [1, 0]]
        actual = model.encode_batch(permuted, task_id=episodes[0]["task_id"])
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        padded = {key: value.clone() for key, value in batch.items()}
        padded["particle_state"] = torch.cat((padded["particle_state"], torch.full((4, 1, 1, 6), float("nan"))), 2)
        padded["particle_mask"] = torch.cat((padded["particle_mask"], torch.zeros(4, 1, dtype=torch.bool)), 1)
        padded["particle_weights"] = torch.cat((padded["particle_weights"], torch.zeros(4, 1)), 1)
        actual = model.encode_batch(padded, task_id=episodes[0]["task_id"])
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        # Future recorded geometry is unavailable during imagination.
        padded["particle_state"] = torch.cat(
            (padded["particle_state"], torch.full_like(padded["particle_state"], 1e6)), 1
        )
        padded["rigid_state"] = torch.cat((padded["rigid_state"], torch.full_like(padded["rigid_state"], -1e6)), 1)
        loss1, _ = rollout_loss(model, batch, task_id=episodes[0]["task_id"])
        loss2, _ = rollout_loss(model, padded, task_id=episodes[0]["task_id"])
        torch.testing.assert_close(loss1, loss2)


def test_full_cloud_history_and_legacy_weight_units(tmp_path):
    """Fit authoritative material aggregates and convert legacy weights to mass."""
    episode = _episode(7)
    episode["task_id"] = "franka_pour"
    episode["particle_mass"] = torch.tensor([0.03, 0.07])
    episode["particle_weights"] = torch.tensor([2.0, 3.0])
    episode["material_state"] = torch.zeros(17, 2, 28)
    episode["material_state_valid"] = torch.ones(17, 2, dtype=torch.bool)
    # Full-cloud stress/history can differ from the spatial sample.
    episode["material_summary"] = torch.full((17, 29), 2.5)
    episode["material_summary"][:3, 0] = torch.tensor([1.0, 2.0, 1e35])
    episode["material_summary"][:, 1] = -2.5
    episode["material_summary"][:, -1] = 1.0
    episodes = _write(tmp_path, [(episode, "train")])
    assert float(episodes[0]["particle_weights"].sum()) == pytest.approx(0.27)
    model = CoupledWorldModel(WorldModelConfig(task_specs(episodes), hidden_dim=16, entity_dim=8))
    fit_normalizers(model, episodes)
    # The encoder applies signed-log to the authoritative aggregate, rather
    # than taking means of logs from the sampled zero-history particles.
    torch.testing.assert_close(model.material_norm.mean[2:28], torch.full((26,), 1.252762968))
    torch.testing.assert_close(model.material_norm.mean[1], torch.tensor(-1.252762968))
    assert model.material_norm.mean[-1] == 1.0
    encoded = model.material_norm(model.material_features(episodes[0]["material_summary"]))
    assert torch.isfinite(encoded).all()
    assert (encoded[0, 0] - encoded[1, 0]).abs() > 1e-3


def test_invalid_terminal_tail_cannot_poison_masked_training(tmp_path):
    """Ignore infinite or overflowing targets after the last valid transition."""
    episode = _episode(8, transitions=3)
    episode["observation"][-1] = float("inf")
    episode["critic_observation"][-1] = float("inf")
    episode["terminated"][-1] = True
    episodes = _write(tmp_path, [(episode, "train")])
    task = episodes[0]["task_id"]
    model = CoupledWorldModel(WorldModelConfig(task_specs(episodes), hidden_dim=16, entity_dim=8))
    fit_normalizers(model, episodes)
    batch = EpisodeWindows(episodes, task_id=task, split="train", horizon=5).sample(8)
    clean_loss, _ = rollout_loss(model, batch, task_id=task)
    assert torch.isfinite(clean_loss)
    corrupted = {key: value.clone() for key, value in batch.items()}
    padding = torch.cat((torch.zeros(8, 1, dtype=torch.bool), ~batch["transition_mask"].bool()), 1)
    for key in ("observation", "critic_observation"):
        corrupted[key][padding] = torch.finfo(torch.float32).max
    corrupted["reward"][~batch["transition_mask"].bool()] = float("inf")
    corrupted_loss, _ = rollout_loss(model, corrupted, task_id=task)
    torch.testing.assert_close(corrupted_loss, clean_loss)
    corrupted_loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


def test_teapot_diagnostics_detect_physical_violations():
    """Detect lost mass, negative energy, invalid covariance and quaternion drift."""
    observation = torch.zeros(3, 121)
    observation[:, 61] = observation[:, 74] = 1.0
    observation[:, 91:96] = torch.tensor([0.6, 0.2, 0.1, 0.1, 0.0])
    observation[:, 90] = 1.0
    observation[:, 96:98] = torch.tensor([1.0, 2.0])
    observation[:, 98:100] = torch.tensor([0.0, 1.0])
    observation[:, 100:103] = 0.01
    observation[1, 90] = -1.0
    observation[1, 91] = -0.1
    observation[2, 61] = 2.0
    observation[2, 100] = -0.01
    errors = physical_observation_errors(observation, task_id="rizon_sharpa_teapot_held_pour")
    assert errors["mass_fraction_sum_absolute_error"][0] < 1e-6
    assert errors["mass_fraction_sum_absolute_error"][1] > 0.69
    assert errors["mass_fraction_out_of_bounds"].tolist() == [0.0, 1.0, 0.0]
    assert errors["negative_kinetic_energy"].tolist() == [0.0, 1.0, 0.0]
    assert errors["non_psd_position_covariance"].tolist() == [0.0, 0.0, 1.0]
    assert errors["nonunit_quaternion"].tolist() == [0.0, 0.0, 1.0]


def test_franka_known_action_and_clock_preserve_observation_contract():
    """Advance Franka's known action memory and timer exactly before critic sharing."""
    spec = TaskSpec(
        "franka_pour",
        84,
        86,
        8,
        1,
        shared_observation_prefix_dim=84,
        last_action_observation_start=54,
        last_action_scale=0.2,
        time_remaining_index=14,
        time_remaining_decrement=1.0 / 360,
    )
    model = CoupledWorldModel(WorldModelConfig([spec], hidden_dim=16, entity_dim=8))
    observation, critic = torch.zeros(2, 84), torch.zeros(2, 86)
    observation[:, 14] = critic[:, 14] = 0.5
    action = torch.tensor([[-2.0, 2.0, 0.5, -0.5, 0.0, 0.1, 0.2, 0.3]]).expand(2, -1)
    result = model.step(observation, critic, action, torch.zeros(2, 1), torch.zeros(2, 16), task_id=spec.task_id)
    expected_action = torch.tensor([[-0.2, 0.2, 0.1, -0.1, 0.0, 0.02, 0.04, 0.06]]).expand(2, -1)
    torch.testing.assert_close(result["observation"][:, 54:62], expected_action)
    torch.testing.assert_close(result["observation"][:, 14], torch.full((2,), 0.497222222))
    torch.testing.assert_close(result["critic_observation"][:, :84], result["observation"], rtol=0, atol=0)


def test_learn_held_out_action_response_and_reload(tmp_path):
    """Learn damped action dynamics across real unrolls and preserve a saved model."""
    torch.set_num_threads(2)
    torch.manual_seed(19)
    episodes = _write(tmp_path, [(_episode(i), "train" if i < 20 else "validation") for i in range(24)])
    task = episodes[0]["task_id"]
    config = WorldModelConfig(task_specs(episodes), hidden_dim=32, entity_dim=8, material_transform="identity")
    model = CoupledWorldModel(config)
    fit_normalizers(model, [episode for episode in episodes if episode["split"] == "train"])
    train = EpisodeWindows(episodes, task_id=task, split="train", horizon=4, seed=13)
    validation = EpisodeWindows(episodes, task_id=task, split="validation", horizon=4)
    before = evaluate_models([model], validation, task_id=task, batch_size=32, batches=2, device="cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(180):
        loss, _ = rollout_loss(model, train.sample(32), task_id=task)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    after = evaluate_models([model], validation, task_id=task, batch_size=32, batches=2, device="cpu")
    assert after["observation_rmse"] < before["observation_rmse"] * 0.35, (before, after)
    assert after["observation_rmse"] < after["persistence_observation_rmse"] * 0.35
    batch = validation.sample(4)
    with torch.no_grad():
        prediction = model.step(
            batch["observation"][:, 0],
            batch["critic_observation"][:, 0],
            batch["action"][:, 0],
            batch["context"],
            model.encode_batch(batch, task_id=task),
            task_id=task,
        )
    torch.testing.assert_close(prediction["critic_observation"][:, :2], prediction["observation"], rtol=0, atol=0)
    path = tmp_path / "trained.pt"
    payload = checkpoint_payload([model], step=180)
    # Legacy version-1 files predate the explicit material-transform marker.
    del payload["config"]["material_transform"]
    torch.save(payload, path)
    restored, checkpoint = load_checkpoint(path)
    assert checkpoint["step"] == 180
    assert restored[0].config.material_transform == "identity"
    reloaded = evaluate_models(restored, validation, task_id=task, batch_size=32, batches=2, device="cpu")
    assert reloaded == after
