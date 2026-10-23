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
from mpm_world_model.evaluate_world_model import (
    _accumulate,
    _finish,
    _statistics,
    evaluate_task,
    physical_observation_errors,
)
from mpm_world_model.models import (
    CoupledWorldModel,
    TaskSpec,
    WorldModelConfig,
    checkpoint_payload,
    load_checkpoint,
    quaternion_apply_local_delta,
    quaternion_geodesic_error,
    quaternion_local_delta,
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


def test_native_capacity_loss_excludes_finite_tail(tmp_path):
    """Do not train on finite states after the native solver drops constraints."""
    episode = _episode(44, transitions=4)
    episode["rigid_capacity_valid"] = torch.tensor([True, True, False, False, False])
    episode["observation"][2:] = 1e6
    episodes = _write(tmp_path, [(episode, "train")])
    loaded = episodes[0]
    assert loaded["finite_mask"].all()
    assert loaded["valid_mask"].tolist() == [True, True, False, False, False]
    model = CoupledWorldModel(WorldModelConfig(task_specs(episodes), hidden_dim=16, entity_dim=8))
    fit_normalizers(model, episodes)
    torch.testing.assert_close(model.task(loaded["task_id"]).observation_norm.mean, episode["observation"][:2].mean(0))
    windows = EpisodeWindows(episodes, task_id=loaded["task_id"], split="train", horizon=4)
    batch = windows.sample(8)
    torch.testing.assert_close(batch["transition_mask"], torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(8, -1))
    assert torch.isfinite(rollout_loss(model, batch, task_id=loaded["task_id"])[0])
    recovering = dict(episode, rigid_capacity_valid=torch.tensor([True, False, True, True, True]))
    with pytest.raises(ValueError, match="cannot recover"):
        _write(tmp_path, [(recovering, "train")])


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
    torch.testing.assert_close(errors["mass_fraction_bound_violation_max"], torch.tensor([0.0, 0.1, 0.0]))
    assert errors["mass_fraction_bound_violation_mean_square"][1] == pytest.approx(0.002)
    assert errors["negative_kinetic_energy"].tolist() == [0.0, 1.0, 0.0]
    assert errors["non_psd_position_covariance"].tolist() == [0.0, 0.0, 1.0]
    assert errors["nonunit_quaternion"].tolist() == [0.0, 0.0, 1.0]


def test_evaluator_distinguishes_orientation_angle_from_quaternion_norm():
    """Measure signed-equivalent poses equally and report physical angular error."""
    reference = torch.zeros(1, 121)
    reference[:, 61] = reference[:, 74] = 1.0
    predicted = reference.clone()
    predicted[:, 61] = -2.0  # Identical rotation with a different sign and norm.
    predicted[:, 74] = predicted[:, 77] = 2.0**-0.5  # Ninety degrees about Z.
    statistics = _statistics()
    _accumulate(
        statistics, predicted, reference, torch.ones(121), torch.ones(1), task_id="rizon_sharpa_teapot_held_pour"
    )
    result = _finish(statistics)
    assert result["quaternion_angular_rmse_rad"] == pytest.approx(torch.pi / (2 * 2.0**0.5))
    assert result["quaternion_angular_count"] == 2
    errors = physical_observation_errors(predicted, task_id="rizon_sharpa_teapot_held_pour")
    assert errors["quaternion_norm_absolute_error"] == pytest.approx(0.5)


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
    del payload["config"]["physical_decoder"]
    for task_config in payload["config"]["tasks"]:
        for field in (
            "mass_fraction_indices",
            "quaternion_starts",
            "quaternion_order",
            "constant_observation_indices",
            "zero_observation_indices",
            "typed_action_observation_start",
            "typed_elapsed_time_index",
            "typed_elapsed_time_increment",
            "typed_countdown_index",
            "typed_countdown_decrement",
        ):
            del task_config[field]
    torch.save(payload, path)
    restored, checkpoint = load_checkpoint(path)
    assert checkpoint["step"] == 180
    assert restored[0].config.material_transform == "identity"
    assert restored[0].config.physical_decoder == "none"
    reloaded = evaluate_models(restored, validation, task_id=task, batch_size=32, batches=2, device="cpu")
    assert reloaded == after


def test_local_quaternion_updates_follow_rotation_composition():
    """Respect quaternion order, sign equivalence and local-frame composition."""
    half = 2.0**-0.5
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    local_x = torch.tensor([[torch.pi / 2, 0.0, 0.0]])
    x_turn = quaternion_apply_local_delta(identity, local_x)
    torch.testing.assert_close(x_turn, torch.tensor([[half, half, 0.0, 0.0]]))
    world_z = torch.tensor([[half, 0.0, 0.0, half]])
    composed = quaternion_apply_local_delta(world_z, local_x)
    torch.testing.assert_close(composed, torch.full((1, 4), 0.5))
    torch.testing.assert_close(quaternion_apply_local_delta(world_z, torch.zeros_like(local_x)), world_z)
    xyzw = quaternion_apply_local_delta(world_z[:, [1, 2, 3, 0]], local_x, order="xyzw")
    torch.testing.assert_close(xyzw, composed[:, [1, 2, 3, 0]])
    torch.testing.assert_close(quaternion_local_delta(world_z, composed), local_x)
    torch.testing.assert_close(quaternion_geodesic_error(composed, -composed), torch.zeros(1), atol=1e-6, rtol=0)
    torch.testing.assert_close(quaternion_geodesic_error(identity, world_z), torch.tensor([torch.pi / 2]))
    rotation = torch.zeros(1, 3, requires_grad=True)
    angle = quaternion_geodesic_error(quaternion_apply_local_delta(identity, rotation), identity)
    angle.square().sum().backward()
    assert torch.isfinite(rotation.grad).all()


def test_typed_rollouts_preserve_mass_pose_and_declared_constants():
    """Transport mass into empty regions while retaining unit rotations and goals."""
    spec = TaskSpec(
        "controlled_reservoir",
        9,
        11,
        1,
        1,
        shared_observation_prefix_dim=9,
        mass_fraction_indices=[0, 1, 2],
        quaternion_starts=[3],
        constant_observation_indices=[7],
        zero_observation_indices=[8],
    )
    model = CoupledWorldModel(WorldModelConfig([spec], hidden_dim=16, entity_dim=8, physical_decoder="typed"))
    observation = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 42.0, 0.0]]).expand(4, -1).clone()
    critic = torch.cat((observation, torch.zeros(4, 2)), -1)
    hidden = torch.zeros(4, 16)
    with torch.no_grad():
        model.task(spec.task_id).output[-1].bias[-3:] = torch.tensor([0.1, 0.2, -0.15])
        for _ in range(64):
            result = model.step(observation, critic, torch.ones(4, 1), torch.zeros(4, 1), hidden, task_id=spec.task_id)
            observation, critic, hidden = result["observation"], result["critic_observation"], result["hidden"]
            assert (observation[:, :3] >= 0).all()
            # Row-stochastic multiplication conserves the seed sum up to
            # accumulated float32 roundoff; no post-hoc renormalizer is used.
            torch.testing.assert_close(observation[:, :3].sum(-1), torch.ones(4), atol=1e-5, rtol=0)
            assert (observation[:, 1:3] > 0).all()
            torch.testing.assert_close(observation[:, 3:7].norm(dim=-1), torch.ones(4), atol=2e-6, rtol=0)
            torch.testing.assert_close(observation[:, 7:9], torch.tensor([[42.0, 0.0]]).expand(4, -1))
            torch.testing.assert_close(critic[:, :9], observation, rtol=0, atol=0)


def test_typed_encoder_rejects_invalid_physical_seeds():
    """Reject invalid actor groups once without inspecting masked entity padding."""
    spec = TaskSpec(
        "seed", 8, 8, 1, 1, mass_fraction_indices=[0, 1, 2], quaternion_starts=[3], zero_observation_indices=[7]
    )
    typed = CoupledWorldModel(WorldModelConfig([spec], hidden_dim=16, entity_dim=8, physical_decoder="typed"))
    legacy = CoupledWorldModel(WorldModelConfig([spec], hidden_dim=16, entity_dim=8))
    observation = torch.tensor([[0.7, 0.2, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0]])

    def encode(model, actor):
        return model.encode(
            actor,
            actor,
            torch.full((1, 1, 6), float("nan")),
            torch.zeros(1, 1, dtype=torch.bool),
            torch.full((1, 1, 13), float("nan")),
            torch.zeros(1, 1, dtype=torch.bool),
            torch.zeros(1, 1),
            task_id=spec.task_id,
        )

    assert torch.isfinite(encode(typed, observation)).all()
    for index, value, message in (
        (3, 0.0, "quaternions"),
        (3, 1.1, "quaternions"),
        (0, -0.1, "mass fractions"),
        (0, 0.5, "sum to one"),
        (7, 0.1, "failure"),
    ):
        invalid = observation.clone()
        invalid[:, index] = value
        with pytest.raises(ValueError, match=message):
            encode(typed, invalid)
        assert torch.isfinite(encode(legacy, invalid)).all()


def test_typed_rotation_loss_ignores_quaternion_sign():
    """Train equal physical rotations equally despite opposite quaternion signs."""
    spec = TaskSpec("rotation", 4, 4, 1, 1, shared_observation_prefix_dim=4, quaternion_starts=[0])
    model = CoupledWorldModel(WorldModelConfig([spec], hidden_dim=16, entity_dim=8, physical_decoder="typed"))
    observation = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [2.0**-0.5, 2.0**-0.5, 0.0, 0.0]]])
    with torch.no_grad():
        # A 30-degree local-X prediction versus a 90-degree reference leaves a
        # 60-degree error. Unit per-axis scales give angular scale sqrt(3).
        model.task(spec.task_id).output[-1].bias[-3:] = torch.tensor([torch.pi / 6, 0.0, 0.0])
    batch = {
        "observation": observation,
        "critic_observation": observation.clone(),
        "action": torch.zeros(1, 1, 1),
        "reward": torch.zeros(1, 1),
        "terminated": torch.zeros(1, 1),
        "transition_mask": torch.ones(1, 1),
        "context": torch.zeros(1, 1),
        "particle_state": torch.zeros(1, 1, 1, 6),
        "particle_mask": torch.ones(1, 1, dtype=torch.bool),
        "particle_weights": torch.ones(1, 1),
        "rigid_state": torch.zeros(1, 1, 1, 13),
        "rigid_mask": torch.ones(1, 1, dtype=torch.bool),
        "controller_state": torch.zeros(1, 1, 0),
        "material_summary": torch.zeros(1, 1, 29),
    }
    expected, expected_components = rollout_loss(model, batch, task_id=spec.task_id)
    torch.testing.assert_close(expected_components["rotation"], torch.tensor(torch.pi**2 / 27))
    batch["observation"][:, 1].neg_()
    batch["critic_observation"][:, 1].neg_()
    actual, components = rollout_loss(model, batch, task_id=spec.task_id)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(components["rotation"], expected_components["rotation"])
    actual.backward()
    gradient = model.task(spec.task_id).output[-1].bias.grad[-3:]
    torch.testing.assert_close(gradient, torch.tensor([-2 * torch.pi / 9, 0.0, 0.0]), atol=1e-6, rtol=1e-5)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_teapot_typed_bookkeeping_retains_native_control_contract(tmp_path):
    """Advance exact action memory and clocks without changing legacy predictions."""
    episode = _episode(12, transitions=2)
    observation = torch.zeros(3, 121)
    observation[:, 61] = observation[:, 74] = observation[:, 91] = 1.0
    observation[:, 118] = torch.tensor([15.98, 15.99, 16.0])
    observation[:, 119] = torch.tensor([0.00125, 0.000625, 0.0])
    observation[:, 120] = 0.6
    action = torch.tensor([[0.2, -0.3, 0.4, 0.5, -0.6, 0.7], [-2.0, 2.0, 0.0, 0.1, 0.2, 0.3]])
    observation[1:, 112:118] = action.clamp(-1.0, 1.0)
    episode.update(
        task_id="rizon_sharpa_teapot_held_pour",
        observation=observation,
        critic_observation=observation.clone(),
        action=action,
        dt=0.01,
    )
    episodes = _write(tmp_path, [(episode, "train")])
    spec = task_specs(episodes)[0]
    assert spec.mass_fraction_indices == [91, 92, 93, 94]
    assert spec.zero_observation_indices == [95]
    initial = observation[1:2]
    for mode in ("none", "typed"):
        model = CoupledWorldModel(WorldModelConfig([spec], hidden_dim=16, entity_dim=8, physical_decoder=mode))
        result = model.step(initial, initial, action[1:2], torch.zeros(1, 1), torch.zeros(1, 16), task_id=spec.task_id)
        if mode == "typed":
            torch.testing.assert_close(result["observation"][:, 112:118], action[1:2].clamp(-1.0, 1.0))
            torch.testing.assert_close(result["observation"][:, 118:121], observation[2:3, 118:121])
            assert result["observation"][:, 95] == 0.0
        else:
            torch.testing.assert_close(result["observation"], initial, rtol=0, atol=0)
        torch.testing.assert_close(result["critic_observation"], result["observation"], rtol=0, atol=0)
    members = [
        CoupledWorldModel(WorldModelConfig([spec], hidden_dim=16, entity_dim=8, physical_decoder="typed"))
        for _ in range(2)
    ]
    with torch.no_grad():
        members[1].task(spec.task_id).output[-1].bias[-6:-3] = torch.tensor([1.0, 0.0, 0.0])
    report = evaluate_task(
        members,
        episodes,
        task_id=spec.task_id,
        split="train",
        horizon=1,
        batches=1,
        batch_size=4,
        device="cpu",
        seed=1,
    )
    # Averaging differing unit orientations is not itself a unit orientation.
    # The policy uses one member, so constraints must be graded per member.
    assert report["ensemble_mean_physical_observables"]["quaternion_norm_absolute_error"] > 0.01
    for diagnostic in report["per_member_diagnostics"]:
        assert diagnostic["physical_observables"]["quaternion_norm_absolute_error"] < 1e-6
        assert diagnostic["physical_observables"]["nonunit_quaternion"] == 0.0
        assert diagnostic["physical_observables"]["mass_fraction_out_of_bounds"] == 0.0
        assert diagnostic["rollout_average"]["quaternion_angular_count"] == 8


def test_typed_transport_learns_held_out_action_dependent_flow(tmp_path):
    """Learn reservoir transfers rather than merely imposing a conserved sum."""
    torch.set_num_threads(2)
    torch.manual_seed(31)
    samples = []
    for seed in range(24):
        episode = _episode(seed, transitions=12)
        episode["task_id"] = "controlled_reservoir"
        generator = torch.Generator().manual_seed(seed + 1000)
        action = torch.randint(2, (12, 1), generator=generator).float() * 2.0 - 1.0
        source = 0.4 + 0.55 * torch.rand((), generator=generator)
        fractions = [torch.stack((source, 1.0 - source, torch.zeros_like(source)))]
        for control in action:
            flow = 0.15 * (control[0] + 1.0) / 2.0 * fractions[-1][0]
            fractions.append(fractions[-1] + torch.stack((-flow, flow, torch.zeros_like(flow))))
        observation = torch.stack(fractions)
        episode.update(
            observation=observation,
            critic_observation=observation.clone(),
            action=action,
            reward=observation[1:, 1] - observation[:-1, 1],
        )
        samples.append((episode, "train" if seed < 20 else "validation"))
    episodes = _write(tmp_path, samples)
    spec = task_specs(episodes)[0]
    spec.mass_fraction_indices = [0, 1, 2]
    model = CoupledWorldModel(WorldModelConfig([spec], hidden_dim=32, entity_dim=8, physical_decoder="typed"))
    fit_normalizers(model, [episode for episode in episodes if episode["split"] == "train"])
    train = EpisodeWindows(episodes, task_id=spec.task_id, split="train", horizon=4, seed=13)
    validation = EpisodeWindows(episodes, task_id=spec.task_id, split="validation", horizon=4)
    before = evaluate_models([model], validation, task_id=spec.task_id, batch_size=32, batches=2, device="cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(280):
        loss, _ = rollout_loss(model, train.sample(32), task_id=spec.task_id)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    after = evaluate_models([model], validation, task_id=spec.task_id, batch_size=32, batches=2, device="cpu")
    assert after["observation_rmse"] < 0.4 * before["observation_rmse"], (before, after)
    assert after["observation_rmse"] < 0.4 * after["persistence_observation_rmse"], after
    batch = validation.sample(64)
    with torch.no_grad():
        initial = batch["observation"][:, 0]
        hidden = model.encode_batch(batch, task_id=spec.task_id)
        delivered = []
        for sign in (-1.0, 1.0):
            result = model.step(
                initial, initial, sign * torch.ones(64, 1), batch["context"], hidden, task_id=spec.task_id
            )
            delivered.append(result["observation"][:, 1])
    assert (delivered[1] - delivered[0]).mean() > 0.04
