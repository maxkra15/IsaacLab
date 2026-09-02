# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for the enhanced slung-load policy distribution."""

import math
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import MLP
from rsl_rl.utils import resolve_callable

from isaaclab_tasks.contrib.drone_slung_load.config.newton_drone.agents.bounded_tanh_gaussian_distribution import (
    HoverBiasedTanhGaussianDistribution,
)
from isaaclab_tasks.contrib.drone_slung_load.config.newton_drone.agents.durability_ppo import (
    DroneSlungLoadDurabilityPPO,
    exponential_decay,
)
from isaaclab_tasks.contrib.drone_slung_load.config.newton_drone.agents.exploration_telemetry_ppo import (
    DroneSlungLoadTelemetryPPO,
    compute_hover_exploration_metrics,
)
from isaaclab_tasks.contrib.drone_slung_load.config.newton_drone.agents.rsl_rl_ppo_cfg import (
    DroneSlungLoadWaypointDirectCTBRPPORunnerCfg,
    DroneSlungLoadWaypointEnhancedPPORunnerCfg,
    DroneSlungLoadWaypointPPORunnerCfg,
)
from isaaclab_tasks.contrib.drone_slung_load.drone_slung_load_env_cfg import DroneSlungLoadWaypointEnhancedEnvCfg
from isaaclab_tasks.contrib.drone_slung_load.system import nominal_hover_action

pytestmark = pytest.mark.unit


def _initialized_distribution() -> tuple[HoverBiasedTanhGaussianDistribution, MLP]:
    distribution = HoverBiasedTanhGaussianDistribution(output_dim=4, init_std=0.003)
    mlp = MLP(input_dim=26, output_dim=distribution.input_dim, hidden_dims=[16], activation="elu")
    distribution.init_mlp_weights(mlp)
    return distribution, mlp


def test_tanh_gaussian_is_bounded_hover_biased_and_directly_learnable():
    distribution, mlp = _initialized_distribution()
    observations = torch.randn(128, 26)
    mlp_output = mlp(observations)
    target = torch.tensor([nominal_hover_action(), 0.0, 0.0, 0.0]).expand_as(mlp_output)

    deterministic_action = distribution.deterministic_output(mlp_output)
    torch.testing.assert_close(deterministic_action, target, atol=1.0e-6, rtol=0.0)
    distribution.update(mlp_output)
    samples = distribution.sample()
    assert torch.all((samples > -1.0) & (samples < 1.0))
    assert distribution.std[:, 1:].mean().item() == pytest.approx(0.003, rel=1.0e-5)

    residual = torch.zeros(1, 4, requires_grad=True)
    roll_action = distribution.deterministic_output(residual)[0, 1]
    (roll_gradient,) = torch.autograd.grad(roll_action, residual)
    assert roll_gradient[0, 1].item() == pytest.approx(1.0)


def test_tanh_gaussian_log_prob_kl_entropy_and_export_match_base_normal():
    distribution, mlp = _initialized_distribution()
    observations = torch.randn(64, 26)
    mlp_output = mlp(observations)
    distribution.update(mlp_output)
    actions = distribution.sample()

    normalized = actions.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
    latent = torch.atanh(normalized)
    base_mean, base_std = distribution.params
    reference_log_prob = (
        torch.distributions.Normal(base_mean, base_std).log_prob(latent) - torch.log1p(-normalized.square())
    ).sum(dim=-1)
    torch.testing.assert_close(distribution.log_prob(actions), reference_log_prob)
    assert torch.isfinite(distribution.entropy).all()

    shifted_output = mlp_output + torch.tensor([1.0e-3, -2.0e-3, 3.0e-3, -4.0e-3])
    old_params = tuple(parameter.detach().clone() for parameter in distribution.params)
    distribution.update(shifted_output)
    new_params = distribution.params
    reference_kl = torch.distributions.kl_divergence(
        torch.distributions.Normal(*old_params), torch.distributions.Normal(*new_params)
    ).sum(dim=-1)
    torch.testing.assert_close(distribution.kl_divergence(old_params, new_params), reference_kl)

    export_module = distribution.as_deterministic_output_module()
    torch.testing.assert_close(export_module(shifted_output), distribution.deterministic_output(shifted_output))


def test_tanh_gaussian_log_prob_and_entropy_have_finite_gradients():
    distribution, mlp = _initialized_distribution()
    observations = torch.randn(512, 26)
    distribution.update(mlp(observations))
    actions = distribution.sample().detach()
    loss = -distribution.log_prob(actions).mean() - 0.002 * distribution.entropy.mean()

    loss.backward()

    assert all(parameter.grad is not None for parameter in mlp.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in mlp.parameters())
    assert distribution.log_std_param.grad is not None
    assert torch.isfinite(distribution.log_std_param.grad).all()


def test_enhanced_runner_uses_axis_calibrated_bounded_exploration():
    cfg = DroneSlungLoadWaypointEnhancedPPORunnerCfg()
    distribution_cfg = cfg.actor.distribution_cfg

    assert resolve_callable(distribution_cfg.to_dict()["class_name"]) is HoverBiasedTanhGaussianDistribution
    assert resolve_callable(cfg.to_dict()["algorithm"]["class_name"]) is DroneSlungLoadDurabilityPPO
    assert DroneSlungLoadWaypointPPORunnerCfg().to_dict()["algorithm"]["class_name"] == "PPO"
    assert distribution_cfg.init_std == pytest.approx((0.03, 0.03, 0.03, 0.03))
    assert distribution_cfg.std_range == pytest.approx((0.005, 0.5))
    assert distribution_cfg.initial_mean[0] == pytest.approx(nominal_hover_action() + 5.0e-4)
    assert distribution_cfg.initial_mean[0] - DroneSlungLoadWaypointPPORunnerCfg().actor.distribution_cfg.initial_mean[
        0
    ] == pytest.approx(5.0e-4)
    assert cfg.experiment_name.endswith("curvature_speed_v13")
    assert cfg.num_steps_per_env == 100
    assert cfg.algorithm.learning_rate == pytest.approx(1.0e-4)
    assert cfg.algorithm.num_learning_epochs == 2
    assert cfg.algorithm.entropy_coef == pytest.approx(0.002)
    assert cfg.algorithm.final_entropy_coef == pytest.approx(0.0005)
    assert cfg.algorithm.entropy_decay_updates == 400
    assert cfg.algorithm.entropy_decay_start_update == 1_600
    assert cfg.algorithm.final_learning_rate == pytest.approx(3.0e-5)
    assert cfg.algorithm.learning_rate_decay_updates == 400
    assert cfg.algorithm.learning_rate_decay_start_update == 1_600
    assert cfg.algorithm.kl_guard_threshold == pytest.approx(0.015)
    assert cfg.algorithm.kl_rejection_lr_factor == pytest.approx(0.5)
    assert cfg.algorithm.gamma == pytest.approx(0.997)
    assert cfg.algorithm.lam == pytest.approx(0.99)


def test_hover_exploration_telemetry_is_physical_detached_and_scalar_only():
    cfg = DroneSlungLoadWaypointEnhancedPPORunnerCfg()
    env_cfg = DroneSlungLoadWaypointEnhancedEnvCfg()
    distribution_cfg = cfg.actor.distribution_cfg.to_dict()
    distribution_cfg.pop("class_name")
    distribution = HoverBiasedTanhGaussianDistribution(output_dim=4, **distribution_cfg)
    sentinel_gradient = torch.tensor((1.0, 2.0, 3.0, 4.0))
    distribution.log_std_param.grad = sentinel_gradient.clone()

    metrics = compute_hover_exploration_metrics(distribution)

    assert all(type(value) is float for value in metrics.values())
    torch.testing.assert_close(distribution.log_std_param.grad, sentinel_gradient)
    assert metrics["exploration/std_latent/collective"] == pytest.approx(0.03)
    assert metrics["exploration/std_latent/roll"] == pytest.approx(0.03)
    assert metrics["exploration/std_latent/pitch"] == pytest.approx(0.03)
    assert metrics["exploration/std_latent/yaw"] == pytest.approx(0.03)
    hover_mean = distribution_cfg["initial_mean"][0]
    collective_action_std = 0.03 * (1.0 - hover_mean**2)
    assert metrics["exploration/std_action_local/collective"] == pytest.approx(collective_action_std)
    assert metrics["exploration/std_action_local/roll"] == pytest.approx(0.03)
    assert metrics["exploration/std_action_local/pitch"] == pytest.approx(0.03)
    assert metrics["exploration/std_action_local/yaw"] == pytest.approx(0.03)
    assert metrics["exploration/std_physical/collective_N"] == pytest.approx(
        collective_action_std * 0.5 * 3.5 * 0.305 * 9.81
    )
    roll_limit, pitch_limit, yaw_limit = env_cfg.actions.thrust.residual_body_rate_limits
    assert metrics["exploration/std_physical/roll_rate_rad_s"] == pytest.approx(0.03 * roll_limit)
    assert metrics["exploration/std_physical/pitch_rate_rad_s"] == pytest.approx(0.03 * pitch_limit)
    assert metrics["exploration/std_physical/yaw_rate_rad_s"] == pytest.approx(0.03 * yaw_limit)
    expected_entropy = 4.0 * (math.log(0.03) + 0.5 * math.log(2.0 * math.pi * math.e))
    assert metrics["exploration/base_normal_entropy_nats"] == pytest.approx(expected_entropy)


def test_direct_ctbr_telemetry_uses_the_full_published_rate_envelope():
    cfg = DroneSlungLoadWaypointDirectCTBRPPORunnerCfg()
    distribution_cfg = cfg.actor.distribution_cfg.to_dict()
    distribution_cfg.pop("class_name")
    distribution = HoverBiasedTanhGaussianDistribution(output_dim=4, **distribution_cfg)

    metrics = compute_hover_exploration_metrics(distribution, cfg.algorithm.physical_body_rate_limits)

    assert metrics["exploration/std_physical/roll_rate_rad_s"] == pytest.approx(0.45)
    assert metrics["exploration/std_physical/pitch_rate_rad_s"] == pytest.approx(0.45)
    assert metrics["exploration/std_physical/yaw_rate_rad_s"] == pytest.approx(0.15)


def test_telemetry_ppo_preserves_parent_update_fields(monkeypatch: pytest.MonkeyPatch):
    cfg = DroneSlungLoadWaypointEnhancedPPORunnerCfg()
    distribution_cfg = cfg.actor.distribution_cfg.to_dict()
    distribution_cfg.pop("class_name")
    distribution = HoverBiasedTanhGaussianDistribution(output_dim=4, **distribution_cfg)
    parent_fields = {"value": 1.25, "surrogate": -0.125, "entropy": -9.0}
    monkeypatch.setattr(PPO, "update", lambda _self: parent_fields.copy())
    algorithm = object.__new__(DroneSlungLoadTelemetryPPO)
    algorithm._raw_actor = SimpleNamespace(distribution=distribution)

    result = algorithm.update()

    assert {key: result[key] for key in parent_fields} == parent_fields
    assert set(result) > set(parent_fields)
    assert all(type(value) is float for value in result.values())


def test_durability_schedules_decay_monotonically_to_configured_floors():
    learning_rates = [exponential_decay(1.0e-4, 2.0e-5, update, 500) for update in range(0, 601)]
    entropy_coefficients = [exponential_decay(2.0e-3, 2.0e-4, update, 500) for update in range(0, 601)]

    assert learning_rates[0] == pytest.approx(1.0e-4)
    assert learning_rates[500] == pytest.approx(2.0e-5)
    assert learning_rates[-1] == pytest.approx(2.0e-5)
    assert entropy_coefficients[0] == pytest.approx(2.0e-3)
    assert entropy_coefficients[500] == pytest.approx(2.0e-4)
    assert entropy_coefficients[-1] == pytest.approx(2.0e-4)
    assert all(next_value <= value for value, next_value in zip(learning_rates[:-1], learning_rates[1:], strict=True))
    assert all(
        next_value <= value
        for value, next_value in zip(entropy_coefficients[:-1], entropy_coefficients[1:], strict=True)
    )


def test_durability_schedules_can_hold_through_the_curriculum_then_decay():
    learning_rates = [exponential_decay(1.0e-4, 3.0e-5, update, 400, 1_600) for update in range(2_001)]
    entropy_coefficients = [exponential_decay(2.0e-3, 5.0e-4, update, 400, 1_600) for update in range(2_001)]

    assert learning_rates[0] == learning_rates[1_599] == learning_rates[1_600] == pytest.approx(1.0e-4)
    assert learning_rates[2_000] == pytest.approx(3.0e-5)
    assert entropy_coefficients[0] == entropy_coefficients[1_600] == pytest.approx(2.0e-3)
    assert entropy_coefficients[2_000] == pytest.approx(5.0e-4)
    assert all(next_value <= value for value, next_value in zip(learning_rates[:-1], learning_rates[1:], strict=True))
    assert all(
        next_value <= value
        for value, next_value in zip(entropy_coefficients[:-1], entropy_coefficients[1:], strict=True)
    )


class _TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.distribution = HoverBiasedTanhGaussianDistribution(
            output_dim=4,
            initial_mean=(nominal_hover_action(), 0.0, 0.0, 0.0),
            init_std=(0.03, 0.02, 0.02, 0.03),
            std_range=(0.005, 0.5),
        )


def _durability_algorithm_for_update_test() -> DroneSlungLoadDurabilityPPO:
    algorithm = object.__new__(DroneSlungLoadDurabilityPPO)
    algorithm._raw_actor = _TinyPolicy()
    algorithm._raw_critic = torch.nn.Linear(1, 1)
    algorithm.optimizer = torch.optim.Adam(
        [*algorithm._raw_actor.parameters(), *algorithm._raw_critic.parameters()], lr=1.0e-4
    )
    algorithm.learning_rate = 1.0e-4
    algorithm.entropy_coef = 2.0e-3
    algorithm._initial_learning_rate = 1.0e-4
    algorithm._final_learning_rate = 2.0e-5
    algorithm._learning_rate_decay_updates = 500
    algorithm._learning_rate_decay_start_update = 0
    algorithm._initial_entropy_coef = 2.0e-3
    algorithm._final_entropy_coef = 2.0e-4
    algorithm._entropy_decay_updates = 500
    algorithm._entropy_decay_start_update = 0
    algorithm._kl_guard_threshold = 0.015
    algorithm._kl_rejection_lr_factor = 0.5
    algorithm._kl_acceptance_lr_recovery_factor = 1.0
    algorithm._completed_updates = 0
    algorithm._learning_rate_cap = 1.0e-4
    algorithm._kl_rejections = 0
    algorithm._environment_step_provider = None
    algorithm._restored_environment_common_step_counter = None
    return algorithm


def test_durability_checkpoint_round_trips_exact_environment_step(monkeypatch: pytest.MonkeyPatch):
    algorithm = _durability_algorithm_for_update_test()
    algorithm._completed_updates = 1_451
    algorithm.bind_environment_step_provider(lambda: 145_137)
    monkeypatch.setattr(PPO, "save", lambda _self: {"policy": "sentinel"})

    saved = algorithm.save()

    assert saved["policy"] == "sentinel"
    assert saved["drone_slung_load_durability_state"] == {
        "version": 1,
        "completed_updates": 1_451,
        "learning_rate_cap": pytest.approx(1.0e-4),
        "kl_rejections": 0,
        "environment_common_step_counter": 145_137,
    }

    restored = _durability_algorithm_for_update_test()
    monkeypatch.setattr(PPO, "load", lambda _self, _loaded, _cfg, _strict: True)

    assert restored.load(saved, load_cfg=None, strict=True)
    assert restored.completed_updates == 1_451
    assert restored.restored_environment_common_step_counter == 145_137


def test_durability_checkpoint_accepts_legacy_state_without_environment_step(monkeypatch: pytest.MonkeyPatch):
    algorithm = _durability_algorithm_for_update_test()
    monkeypatch.setattr(PPO, "load", lambda _self, _loaded, _cfg, _strict: True)
    legacy = {
        "drone_slung_load_durability_state": {
            "version": 1,
            "completed_updates": 73,
            "learning_rate_cap": 8.0e-5,
            "kl_rejections": 2,
        }
    }

    algorithm.load(legacy, load_cfg=None, strict=True)

    assert algorithm.completed_updates == 73
    assert algorithm.restored_environment_common_step_counter is None


@pytest.mark.parametrize("invalid_step", [True, -1, 1.5])
def test_durability_checkpoint_rejects_invalid_environment_step(
    monkeypatch: pytest.MonkeyPatch,
    invalid_step: object,
):
    algorithm = _durability_algorithm_for_update_test()
    monkeypatch.setattr(PPO, "load", lambda _self, _loaded, _cfg, _strict: True)
    checkpoint = {
        "drone_slung_load_durability_state": {
            "version": 1,
            "completed_updates": 1,
            "learning_rate_cap": 1.0e-4,
            "kl_rejections": 0,
            "environment_common_step_counter": invalid_step,
        }
    }

    with pytest.raises(ValueError, match="environment_common_step_counter"):
        algorithm.load(checkpoint, load_cfg=None, strict=True)


@pytest.mark.parametrize(("proposed_kl", "accepted"), [(0.01, True), (0.02, False)])
def test_durability_ppo_commits_only_updates_inside_kl_budget(
    monkeypatch: pytest.MonkeyPatch,
    proposed_kl: float,
    accepted: bool,
):
    algorithm = _durability_algorithm_for_update_test()
    actor_before = algorithm._raw_actor.weight.detach().clone()
    critic_before = algorithm._raw_critic.weight.detach().clone()

    def attempted_parent_update(self):
        with torch.no_grad():
            self._raw_actor.weight.add_(1.0)
            self._raw_critic.weight.add_(1.0)
        return {"value": 1.0, "surrogate": -0.1, "entropy": -9.0}

    monkeypatch.setattr(DroneSlungLoadTelemetryPPO, "update", attempted_parent_update)
    monkeypatch.setattr(DroneSlungLoadDurabilityPPO, "_measure_rollout_kl", lambda _self: (proposed_kl, proposed_kl))
    monkeypatch.setattr(DroneSlungLoadDurabilityPPO, "_refresh_policy_distribution", lambda _self: None)

    result = algorithm.update()

    if accepted:
        assert algorithm._raw_actor.weight.item() == pytest.approx(actor_before.item() + 1.0)
        assert algorithm._raw_critic.weight.item() == pytest.approx(critic_before.item() + 1.0)
        assert algorithm._learning_rate_cap == pytest.approx(1.0e-4)
    else:
        torch.testing.assert_close(algorithm._raw_actor.weight, actor_before)
        torch.testing.assert_close(algorithm._raw_critic.weight, critic_before)
        assert algorithm._learning_rate_cap == pytest.approx(5.0e-5)
    assert result["durability/update_accepted"] == float(accepted)
    assert result["durability/rollout_kl_proposed_mean"] == pytest.approx(proposed_kl)
    assert result["durability/rollout_kl_applied_mean"] == pytest.approx(proposed_kl if accepted else 0.0)
    assert algorithm.learning_rate <= 1.0e-4
    assert algorithm.entropy_coef < 2.0e-3


def test_tanh_gaussian_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="strictly inside"):
        HoverBiasedTanhGaussianDistribution(output_dim=1, initial_mean=(1.0,))
    with pytest.raises(ValueError, match="inside std_range"):
        HoverBiasedTanhGaussianDistribution(output_dim=1, initial_mean=(0.0,), init_std=math.inf)
