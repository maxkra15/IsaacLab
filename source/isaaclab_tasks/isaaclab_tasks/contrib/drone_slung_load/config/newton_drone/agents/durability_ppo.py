# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Monotonic optimization schedules and transactional KL guarding for slung-load PPO."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from typing import Any

import torch

from .exploration_telemetry_ppo import DroneSlungLoadTelemetryPPO, compute_hover_exploration_metrics

_CHECKPOINT_STATE_KEY = "drone_slung_load_durability_state"
_CHECKPOINT_STATE_VERSION = 1
_ENVIRONMENT_STEP_KEY = "environment_common_step_counter"


def exponential_decay(
    initial: float,
    final: float,
    completed_updates: int,
    decay_updates: int,
    start_update: int = 0,
) -> float:
    """Return a delayed exponential interpolation from ``initial`` to ``final``."""
    if not math.isfinite(initial) or not math.isfinite(final) or initial <= 0.0 or final <= 0.0:
        raise ValueError("Decay endpoints must be finite and positive.")
    if final > initial:
        raise ValueError("A monotonic decay requires final <= initial.")
    if completed_updates < 0:
        raise ValueError("completed_updates must be nonnegative.")
    if decay_updates <= 0:
        raise ValueError("decay_updates must be positive.")
    if start_update < 0:
        raise ValueError("start_update must be nonnegative.")
    elapsed_updates = max(completed_updates - start_update, 0)
    fraction = min(elapsed_updates, decay_updates) / decay_updates
    return initial * math.pow(final / initial, fraction)


class DroneSlungLoadDurabilityPPO(DroneSlungLoadTelemetryPPO):
    """Enhanced-task PPO with resume-safe decay and a transactional rollout-KL guard.

    RSL-RL's adaptive schedule observes minibatch KL but may increase the learning
    rate again when KL is small. This variant keeps the upstream PPO loss unchanged,
    evaluates the exact old-to-new policy KL on the complete rollout, and commits the
    update only when its global mean remains inside the configured budget. A rejected
    update restores the actor, critic, and optimizer together before reducing the
    learning-rate ceiling.

    Exploration pressure decays independently from the learned standard deviation.
    The distribution can therefore retain task-driven stochasticity instead of being
    forced toward a prescribed standard-deviation trajectory.
    """

    def __init__(
        self,
        *args: Any,
        final_learning_rate: float = 2.0e-5,
        learning_rate_decay_updates: int = 500,
        learning_rate_decay_start_update: int = 0,
        final_entropy_coef: float = 2.0e-4,
        entropy_decay_updates: int = 500,
        entropy_decay_start_update: int = 0,
        kl_guard_threshold: float = 0.015,
        kl_rejection_lr_factor: float = 0.5,
        kl_acceptance_lr_recovery_factor: float = 1.0,
        kl_evaluation_batch_size: int = 16_384,
        **kwargs: Any,
    ) -> None:
        """Initialize durability schedules around the standard RSL-RL PPO update."""
        super().__init__(*args, **kwargs)
        if self.schedule != "fixed":
            raise ValueError("DroneSlungLoadDurabilityPPO requires schedule='fixed'.")
        if not math.isfinite(final_learning_rate) or not 0.0 < final_learning_rate <= self.learning_rate:
            raise ValueError("final_learning_rate must lie in (0, learning_rate].")
        if learning_rate_decay_updates <= 0:
            raise ValueError("learning_rate_decay_updates must be positive.")
        if learning_rate_decay_start_update < 0:
            raise ValueError("learning_rate_decay_start_update must be nonnegative.")
        if not math.isfinite(final_entropy_coef) or not 0.0 < final_entropy_coef <= self.entropy_coef:
            raise ValueError("final_entropy_coef must lie in (0, entropy_coef].")
        if entropy_decay_updates <= 0:
            raise ValueError("entropy_decay_updates must be positive.")
        if entropy_decay_start_update < 0:
            raise ValueError("entropy_decay_start_update must be nonnegative.")
        if not math.isfinite(kl_guard_threshold) or kl_guard_threshold <= 0.0:
            raise ValueError("kl_guard_threshold must be finite and positive.")
        if not math.isfinite(kl_rejection_lr_factor) or not 0.0 < kl_rejection_lr_factor < 1.0:
            raise ValueError("kl_rejection_lr_factor must lie in (0, 1).")
        if not math.isfinite(kl_acceptance_lr_recovery_factor) or kl_acceptance_lr_recovery_factor < 1.0:
            raise ValueError("kl_acceptance_lr_recovery_factor must be finite and at least one.")
        if kl_evaluation_batch_size <= 0:
            raise ValueError("kl_evaluation_batch_size must be positive.")
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError("DroneSlungLoadDurabilityPPO currently supports feed-forward policies only.")

        self._initial_learning_rate = self.learning_rate
        self._final_learning_rate = final_learning_rate
        self._learning_rate_decay_updates = learning_rate_decay_updates
        self._learning_rate_decay_start_update = learning_rate_decay_start_update
        self._initial_entropy_coef = self.entropy_coef
        self._final_entropy_coef = final_entropy_coef
        self._entropy_decay_updates = entropy_decay_updates
        self._entropy_decay_start_update = entropy_decay_start_update
        self._kl_guard_threshold = kl_guard_threshold
        self._kl_rejection_lr_factor = kl_rejection_lr_factor
        self._kl_acceptance_lr_recovery_factor = kl_acceptance_lr_recovery_factor
        self._kl_evaluation_batch_size = kl_evaluation_batch_size
        self._completed_updates = 0
        self._learning_rate_cap = self.learning_rate
        self._kl_rejections = 0
        self._environment_step_provider: Callable[[], int] | None = None
        self._restored_environment_common_step_counter: int | None = None
        self._apply_schedules()

    @property
    def completed_updates(self) -> int:
        """Return the number of completed PPO updates restored from the checkpoint."""
        return self._completed_updates

    @property
    def restored_environment_common_step_counter(self) -> int | None:
        """Return the exact restored environment control step, when available."""
        return self._restored_environment_common_step_counter

    def bind_environment_step_provider(self, provider: Callable[[], int]) -> None:
        """Bind a live environment control-step provider for subsequent checkpoints."""
        if not callable(provider):
            raise TypeError("provider must be callable.")
        self._environment_step_provider = provider

    @staticmethod
    def _validate_environment_step(value: object) -> int:
        """Validate and return a serialized environment control-step counter."""
        if type(value) is not int or value < 0:
            raise ValueError("environment_common_step_counter must be a non-negative integer.")
        return value

    def _apply_schedules(self) -> None:
        scheduled_learning_rate = exponential_decay(
            self._initial_learning_rate,
            self._final_learning_rate,
            self._completed_updates,
            self._learning_rate_decay_updates,
            self._learning_rate_decay_start_update,
        )
        self.learning_rate = min(scheduled_learning_rate, self._learning_rate_cap)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate
        self.entropy_coef = exponential_decay(
            self._initial_entropy_coef,
            self._final_entropy_coef,
            self._completed_updates,
            self._entropy_decay_updates,
            self._entropy_decay_start_update,
        )

    def _measure_rollout_kl(self) -> tuple[float, float]:
        """Measure global mean and maximum old-to-new KL over the stored rollout."""
        if self.storage.distribution_params is None:
            raise RuntimeError("Rollout distribution parameters are unavailable for KL guarding.")

        observations = self.storage.observations.flatten(0, 1)
        old_distribution_params = tuple(parameter.flatten(0, 1) for parameter in self.storage.distribution_params)
        sample_count = observations.batch_size[0]
        local_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        local_max = torch.zeros((), device=self.device, dtype=torch.float64)

        with torch.inference_mode():
            for start in range(0, sample_count, self._kl_evaluation_batch_size):
                stop = min(start + self._kl_evaluation_batch_size, sample_count)
                self._raw_actor(observations[start:stop], stochastic_output=True)
                new_distribution_params = self._raw_actor.output_distribution_params
                old_chunk = tuple(parameter[start:stop] for parameter in old_distribution_params)
                kl = self._raw_actor.get_kl_divergence(old_chunk, new_distribution_params).reshape(-1)
                if not torch.isfinite(kl).all():
                    local_sum.fill_(torch.inf)
                    local_max.fill_(torch.inf)
                    break
                local_sum += kl.to(torch.float64).sum()
                local_max = torch.maximum(local_max, kl.to(torch.float64).amax())

        global_sum_and_count = torch.stack(
            (local_sum, torch.tensor(float(sample_count), device=self.device, dtype=torch.float64))
        )
        global_max = local_max.clone()
        if self.is_multi_gpu:
            torch.distributed.all_reduce(global_sum_and_count, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(global_max, op=torch.distributed.ReduceOp.MAX)
        global_mean = global_sum_and_count[0] / global_sum_and_count[1].clamp_min(1.0)
        return global_mean.item(), global_max.item()

    def _snapshot_training_state(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
        actor_state = {name: value.detach().clone() for name, value in self._raw_actor.state_dict().items()}
        critic_state = {name: value.detach().clone() for name, value in self._raw_critic.state_dict().items()}
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        return actor_state, critic_state, optimizer_state

    def _restore_training_state(
        self,
        snapshot: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]],
    ) -> None:
        actor_state, critic_state, optimizer_state = snapshot
        self._raw_actor.load_state_dict(actor_state)
        self._raw_critic.load_state_dict(critic_state)
        self.optimizer.load_state_dict(optimizer_state)

    def _refresh_policy_distribution(self) -> None:
        """Refresh cached output statistics after accepting or restoring parameters."""
        with torch.inference_mode():
            self._raw_actor(self.storage.observations[0], stochastic_output=True)

    def update(self) -> dict[str, float]:
        """Apply one PPO update transaction and report the exact rollout KL."""
        self._apply_schedules()
        attempted_learning_rate = self.learning_rate
        snapshot = self._snapshot_training_state()
        try:
            loss_dict = super().update()
            proposed_kl_mean, proposed_kl_max = self._measure_rollout_kl()
        except Exception:
            # Do not leave a partially applied update in memory if KL evaluation
            # or the upstream optimizer fails after touching model state.
            self._restore_training_state(snapshot)
            raise
        update_accepted = math.isfinite(proposed_kl_mean) and proposed_kl_mean <= self._kl_guard_threshold

        if not update_accepted:
            self._restore_training_state(snapshot)
            self._kl_rejections += 1
            self._learning_rate_cap = max(
                self._final_learning_rate,
                min(self._learning_rate_cap, attempted_learning_rate * self._kl_rejection_lr_factor),
            )
        else:
            # A single early outlier must not permanently starve a later
            # curriculum stage. Recover the cap slowly after accepted updates;
            # the delayed decay schedule remains the absolute ceiling.
            self._learning_rate_cap = min(
                self._initial_learning_rate,
                self._learning_rate_cap * self._kl_acceptance_lr_recovery_factor,
            )

        self._completed_updates += 1
        self._apply_schedules()
        self._refresh_policy_distribution()
        loss_dict.update(
            compute_hover_exploration_metrics(
                self.get_policy().distribution,
                getattr(self, "_physical_body_rate_limits", (10.0, 10.0, 2.5)),
            )
        )
        loss_dict.update(
            {
                "durability/rollout_kl_proposed_mean": proposed_kl_mean,
                "durability/rollout_kl_proposed_max": proposed_kl_max,
                "durability/rollout_kl_applied_mean": proposed_kl_mean if update_accepted else 0.0,
                "durability/update_accepted": float(update_accepted),
                "durability/kl_rejections_total": float(self._kl_rejections),
                "durability/learning_rate_next": float(self.learning_rate),
                "durability/entropy_coef_next": float(self.entropy_coef),
            }
        )
        return loss_dict

    def save(self) -> dict:
        """Save PPO state together with resume-critical durability counters."""
        saved_dict = super().save()
        durability_state = {
            "version": _CHECKPOINT_STATE_VERSION,
            "completed_updates": self._completed_updates,
            "learning_rate_cap": self._learning_rate_cap,
            "kl_rejections": self._kl_rejections,
        }
        if self._environment_step_provider is not None:
            durability_state[_ENVIRONMENT_STEP_KEY] = self._validate_environment_step(self._environment_step_provider())
        saved_dict[_CHECKPOINT_STATE_KEY] = durability_state
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Restore PPO and durability state, including safe legacy-checkpoint inference."""
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        durability_state = loaded_dict.get(_CHECKPOINT_STATE_KEY)
        self._restored_environment_common_step_counter = None
        if durability_state is None:
            self._completed_updates = max(int(loaded_dict.get("iter", -1)) + 1, 0)
            loaded_learning_rates = [float(group["lr"]) for group in self.optimizer.param_groups]
            self._learning_rate_cap = min(self._initial_learning_rate, *loaded_learning_rates)
            self._kl_rejections = 0
        else:
            if durability_state.get("version") != _CHECKPOINT_STATE_VERSION:
                raise ValueError("Unsupported drone slung-load durability checkpoint version.")
            self._completed_updates = int(durability_state["completed_updates"])
            self._learning_rate_cap = float(durability_state["learning_rate_cap"])
            self._kl_rejections = int(durability_state["kl_rejections"])
            if self._completed_updates < 0 or self._kl_rejections < 0:
                raise ValueError("Durability checkpoint counters must be nonnegative.")
            if not self._final_learning_rate <= self._learning_rate_cap <= self._initial_learning_rate:
                raise ValueError("Durability checkpoint learning-rate cap is outside the configured range.")
            if _ENVIRONMENT_STEP_KEY in durability_state:
                self._restored_environment_common_step_counter = self._validate_environment_step(
                    durability_state[_ENVIRONMENT_STEP_KEY]
                )
        self._apply_schedules()
        return load_iteration
