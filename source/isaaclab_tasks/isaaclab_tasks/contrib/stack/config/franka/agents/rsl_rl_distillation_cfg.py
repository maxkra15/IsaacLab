# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher-student distillation configuration for camera-based stacking."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms import Distillation
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlDistillationAlgorithmCfg, RslRlDistillationRunnerCfg, RslRlMLPModelCfg

from isaaclab_tasks.core.lift.config.kuka_allegro.agents.models import SpatialSoftmaxCNNModel

from .rsl_rl_ppo_cfg import (
    StackGaussianDistribution,
    StackGaussianDistributionCfg,
    StackSpatialSoftmaxCNNModelCfg,
)


class StackDistillationDistribution(StackGaussianDistribution):
    """Mixed stack distribution with a differentiable cloning output.

    Environment rollouts still sample an exact Bernoulli gripper action. For
    behavior cloning, the signed Bernoulli expectation provides a smooth path
    from the action loss to the gripper logit; the standard hard threshold has
    zero derivative and therefore cannot teach the gripper decision.
    """

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Return arm means and a differentiable signed gripper probability."""
        gripper_expectation = torch.tanh(0.5 * mlp_output[..., -1:])
        return torch.cat((mlp_output[..., :-1], gripper_expectation), dim=-1)


@configclass
class StackDistillationDistributionCfg(StackGaussianDistributionCfg):
    """Differentiable student distribution used only during distillation."""

    class_name: str = (
        "isaaclab_tasks.contrib.stack.config.franka.agents.rsl_rl_distillation_cfg:StackDistillationDistribution"
    )


class StackVisualDistillationModel(SpatialSoftmaxCNNModel):
    """Direct visual behavior-cloning policy with physical-state supervision.

    The policy head learns only the teacher's eight actions. A separate head
    predicts the 37 independent visual primitives used for representation
    learning and diagnostics; it is discarded when initializing PPO.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        cnn_cfg: dict | None = None,
        init_temperature: float = 1.0,
        auxiliary_hidden_dims: tuple[int, ...] | list[int] = (512, 256),
        visual_target_indices: tuple[int, ...] | list[int] = (),
        teacher_object_start: int = 22,
    ) -> None:
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
            cnn_cfg=cnn_cfg,
            init_temperature=init_temperature,
        )
        if not visual_target_indices:
            raise ValueError("visual_target_indices must select at least one teacher value.")
        self.teacher_object_start = int(teacher_object_start)
        self.register_buffer(
            "visual_target_indices", torch.tensor(visual_target_indices, dtype=torch.long), persistent=False
        )
        self.auxiliary_head = MLP(
            self._get_latent_dim(),
            len(visual_target_indices),
            auxiliary_hidden_dims,
            activation,
        )

    def predict_visual_state(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict normalized independent object-state primitives."""
        return self.auxiliary_head(latent)

    def select_visual_target(self, normalized_teacher_state: torch.Tensor) -> torch.Tensor:
        """Select matching primitive targets from the normalized teacher input."""
        return normalized_teacher_state.index_select(-1, self.visual_target_indices)


@configclass
class StackVisualDistillationModelCfg(StackSpatialSoftmaxCNNModelCfg):
    """Direct camera student with an auxiliary physical-state head."""

    class_name: str = (
        "isaaclab_tasks.contrib.stack.config.franka.agents.rsl_rl_distillation_cfg:StackVisualDistillationModel"
    )
    auxiliary_hidden_dims: list[int] = [512, 256]
    # Teacher object block: positions[0:9], up axes[18:27], spatial
    # velocities[27:45], and stack progress[63]. Redundant relative/pair
    # geometry is intentionally excluded from the learned target.
    visual_target_indices: tuple[int, ...] = (*range(22, 31), *range(40, 67), 85)
    teacher_object_start: int = 22


class StackDistillation(Distillation):
    """Feedback-controlled DAgger for the direct visual student policy.

    Teacher-only behavior cloning provides a short warm-up. A fixed fraction
    of student-controlled states is then collected with teacher labels; this
    corrects behavior-cloning covariate shift. Independent per-step controller
    mixing gives the requested occupancy without bias from unequal episode
    lengths, while evaluation environments remain student-controlled for
    complete episodes.

    The seven arm outputs and the binary gripper also use losses matching their
    semantics: Huber/MSE regression for the arm and binary cross entropy for
    the gripper logit. Treating all eight outputs as one regression diluted the
    gripper supervision and did not directly optimize its Bernoulli decision.
    """

    def __init__(
        self,
        student: MLPModel,
        teacher: MLPModel,
        storage: RolloutStorage,
        teacher_pretrain_updates: int = 40,
        student_control_fraction: float = 0.25,
        evaluation_success_ema_alpha: float = 0.25,
        arm_loss_weight: float = 1.0,
        gripper_loss_weight: float = 1.0,
        auxiliary_loss_weight: float = 0.0,
        action_clip: float = 1.0,
        evaluation_envs_per_recipe: int = 4,
        success_reward_threshold: float = 1.0,
        recipe_count: int = 9,
        recipe_names: tuple[str, ...] = (
            "final_release",
            "second_place",
            "second_transport",
            "second_pick",
            "pair_ready",
            "first_place",
            "first_transport",
            "first_pick",
            "table",
        ),
        recipe_balance: bool = True,
        table_recipe_weight: float = 3.0,
        student_state_loss_weight: float = 3.0,
        distillation_context_obs_group: str = "distillation_context",
        table_recipe_id: int = 8,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        adaptive_learning_rate_min: float = 1.0e-5,
        adaptive_learning_rate_max: float = 3.0e-4,
        adaptive_learning_rate_factor: float = 1.5,
        kl_measurement_samples: int = 2048,
        **kwargs,
    ) -> None:
        super().__init__(student, teacher, storage, **kwargs)
        if teacher_pretrain_updates < 0:
            raise ValueError("teacher_pretrain_updates must be non-negative.")
        if not 0.0 < student_control_fraction <= 1.0:
            raise ValueError("student_control_fraction must be in (0, 1].")
        if not 0.0 < evaluation_success_ema_alpha <= 1.0:
            raise ValueError("evaluation_success_ema_alpha must be in (0, 1].")
        if arm_loss_weight <= 0.0 or gripper_loss_weight <= 0.0 or auxiliary_loss_weight < 0.0:
            raise ValueError("Action losses must be positive and the auxiliary loss must be non-negative.")
        if action_clip <= 0.0:
            raise ValueError("action_clip must be positive.")
        if success_reward_threshold <= 0.0:
            raise ValueError("success_reward_threshold must be positive.")
        if (
            recipe_count <= 0
            or len(recipe_names) != recipe_count
            or not 0 <= table_recipe_id < recipe_count
            or table_recipe_weight <= 0.0
            or student_state_loss_weight < 1.0
        ):
            raise ValueError("Recipe and loss-weight settings are inconsistent.")
        if evaluation_envs_per_recipe <= 0:
            raise ValueError("evaluation_envs_per_recipe must be positive.")
        if schedule not in ("fixed", "adaptive"):
            raise ValueError("schedule must be either 'fixed' or 'adaptive'.")
        if desired_kl <= 0.0:
            raise ValueError("desired_kl must be positive.")
        if not 0.0 < adaptive_learning_rate_min <= adaptive_learning_rate_max:
            raise ValueError("Adaptive learning-rate bounds must be positive and increasing.")
        if (
            schedule == "adaptive"
            and not adaptive_learning_rate_min <= self.learning_rate <= adaptive_learning_rate_max
        ):
            raise ValueError("The initial learning rate must lie inside the adaptive bounds.")
        if adaptive_learning_rate_factor <= 1.0:
            raise ValueError("adaptive_learning_rate_factor must be greater than one.")
        if kl_measurement_samples <= 0:
            raise ValueError("kl_measurement_samples must be positive.")
        evaluation_env_count = evaluation_envs_per_recipe * recipe_count
        if evaluation_env_count >= storage.num_envs:
            raise ValueError("Per-recipe evaluation slots leave no environments for training.")
        if self.student.is_recurrent or self.teacher.is_recurrent:
            raise ValueError("StackDistillation currently supports only feed-forward teacher and student models.")
        if auxiliary_loss_weight > 0.0 and not callable(getattr(self.student, "predict_visual_state", None)):
            raise TypeError("Auxiliary supervision requires a student with predict_visual_state().")
        self.teacher_pretrain_updates = int(teacher_pretrain_updates)
        self.student_control_fraction = float(student_control_fraction)
        self.evaluation_success_ema_alpha = float(evaluation_success_ema_alpha)
        self.arm_loss_weight = float(arm_loss_weight)
        self.gripper_loss_weight = float(gripper_loss_weight)
        self.auxiliary_loss_weight = float(auxiliary_loss_weight)
        self.action_clip = float(action_clip)
        self.evaluation_envs_per_recipe = int(evaluation_envs_per_recipe)
        self.evaluation_env_count = evaluation_env_count
        self.success_reward_threshold = float(success_reward_threshold)
        self.recipe_count = int(recipe_count)
        self.recipe_names = tuple(recipe_names)
        self.recipe_balance = bool(recipe_balance)
        self.table_recipe_weight = float(table_recipe_weight)
        self.student_state_loss_weight = float(student_state_loss_weight)
        self.distillation_context_obs_group = distillation_context_obs_group
        self.table_recipe_id = int(table_recipe_id)
        self.schedule = schedule
        self.desired_kl = float(desired_kl)
        self.adaptive_learning_rate_min = float(adaptive_learning_rate_min)
        self.adaptive_learning_rate_max = float(adaptive_learning_rate_max)
        self.adaptive_learning_rate_factor = float(adaptive_learning_rate_factor)
        self.kl_measurement_samples = int(kl_measurement_samples)

        self._teacher_control_mask: torch.Tensor | None = None
        self._teacher_control_count = torch.zeros((), device=self.device)
        self._total_control_count = torch.zeros((), device=self.device)
        self._evaluation_successes = torch.zeros(self.recipe_count, device=self.device)
        self._evaluation_attempts = torch.zeros(self.recipe_count, device=self.device)
        self._evaluation_cumulative_successes = torch.zeros(self.recipe_count, device=self.device)
        self._evaluation_cumulative_attempts = torch.zeros(self.recipe_count, device=self.device)
        self._evaluation_success_ema = torch.zeros(self.recipe_count, device=self.device)
        self._evaluation_ema_initialized = torch.zeros(self.recipe_count, dtype=torch.bool, device=self.device)
        self._episode_success_seen = torch.zeros(storage.num_envs, dtype=torch.bool, device=self.device)
        self._rollout_student_control_masks: list[torch.Tensor] = []

    def _recipe_ids(self, obs: TensorDict) -> torch.Tensor | None:
        if self.distillation_context_obs_group not in obs.keys():
            return None
        return obs[self.distillation_context_obs_group].argmax(dim=-1)

    def _balanced_recipe_weights(self, recipe_ids: torch.Tensor) -> torch.Tensor:
        """Balance recipes while retaining extra loss mass for TABLE starts."""
        counts = torch.bincount(recipe_ids, minlength=self.recipe_count).to(dtype=torch.float32)
        present = counts > 0
        recipe_mass = torch.ones_like(counts)
        recipe_mass[self.table_recipe_id] = self.table_recipe_weight
        present_mass = recipe_mass[present].sum().clamp_min(1.0)
        weights_by_recipe = torch.zeros_like(counts)
        weights_by_recipe[present] = recipe_ids.numel() * recipe_mass[present] / (present_mass * counts[present])
        return weights_by_recipe[recipe_ids]

    def _dagger_collection_started(self) -> bool:
        """Return whether teacher-only behavior-cloning warm-up is complete."""
        return self.num_updates >= self.teacher_pretrain_updates

    def _student_control_probability(self) -> float:
        """Return the probability that the student controls a training step."""
        if not self._dagger_collection_started():
            return 0.0
        return self.student_control_fraction

    def _assign_pending_controllers(self, obs: TensorDict) -> None:
        """Choose a teacher or student independently for every training step."""
        if self.distillation_context_obs_group not in obs.keys():
            raise KeyError(f"Missing DAgger recipe observation group: {self.distillation_context_obs_group}")
        context = obs[self.distillation_context_obs_group]
        count = context.shape[0]
        if self._teacher_control_mask is None or self._teacher_control_mask.shape[0] != count:
            self._teacher_control_mask = torch.ones((count, 1), dtype=torch.bool, device=context.device)

        # Evaluation slots are always deterministic student rollouts and never
        # enter the optimization slice.
        self._teacher_control_mask[: self.evaluation_env_count] = False
        training_count = count - self.evaluation_env_count
        student_control = torch.rand(training_count, device=context.device) < self._student_control_probability()
        self._teacher_control_mask[self.evaluation_env_count :, 0] = ~student_control

    def _update_evaluation_statistics(self, outcomes: torch.Tensor) -> None:
        """Update held-out success estimates without gating data collection."""
        successes, attempts = outcomes
        self._evaluation_cumulative_successes.add_(successes)
        self._evaluation_cumulative_attempts.add_(attempts)
        observed = attempts > 0
        batch_rates = successes / attempts.clamp_min(1.0)
        uninitialized = observed & ~self._evaluation_ema_initialized
        self._evaluation_success_ema[uninitialized] = batch_rates[uninitialized]
        initialized = observed & self._evaluation_ema_initialized
        self._evaluation_success_ema[initialized] = torch.lerp(
            self._evaluation_success_ema[initialized],
            batch_rates[initialized],
            self.evaluation_success_ema_alpha,
        )
        self._evaluation_ema_initialized[observed] = True

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Collect deterministic student-state DAgger trajectories with teacher labels."""
        student_latent = self.student.get_latent(obs)
        student_raw_output = self.student.mlp(student_latent)
        if self.student.distribution is None:
            raise RuntimeError("StackDistillation requires a student action distribution.")
        # RSL-RL's runner logs output_std after every update. Refresh the
        # distribution from this rollout's output even though DAgger executes
        # its deterministic action; otherwise its statistics remain unset.
        self.student.distribution.update(student_raw_output)
        student_actions = self.student.distribution.deterministic_output(student_raw_output)
        student_actions = student_actions.detach().clamp(-self.action_clip, self.action_clip)
        # The RSL-RL environment wrapper clips actions before stepping physics.
        # Clone that physical action, not an unreachable raw teacher output.
        teacher_actions = self.teacher(obs).detach().clamp(-self.action_clip, self.action_clip)
        self._assign_pending_controllers(obs)
        if self._teacher_control_mask is None:
            raise RuntimeError("Controller assignment failed.")

        actions = torch.where(self._teacher_control_mask, teacher_actions, student_actions)
        self._rollout_student_control_masks.append((~self._teacher_control_mask[:, 0]).detach().clone())
        training_control_mask = self._teacher_control_mask[self.evaluation_env_count :]
        self._teacher_control_count.add_(training_control_mask.sum())
        self._total_control_count.add_(training_control_mask.numel())

        self.transition.actions = actions
        self.transition.privileged_actions = teacher_actions
        self.transition.observations = obs
        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Store transitions, evaluation outcomes, and episode controller boundaries."""
        self._episode_success_seen |= rewards.reshape(-1) >= self.success_reward_threshold
        evaluation_dones = (
            dones[: self.evaluation_env_count].reshape(self.recipe_count, self.evaluation_envs_per_recipe).bool()
        )
        evaluation_success_seen = self._episode_success_seen[: self.evaluation_env_count].reshape(
            self.recipe_count, self.evaluation_envs_per_recipe
        )
        self._evaluation_attempts.add_(evaluation_dones.sum(dim=1))
        self._evaluation_successes.add_((evaluation_dones & evaluation_success_seen).sum(dim=1))
        super().process_env_step(obs, rewards, dones, extras)
        done_mask = dones.reshape(-1).bool()
        self._episode_success_seen[done_mask] = False

    def _behavior_terms(
        self,
        student_raw_output: torch.Tensor,
        teacher_actions: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute type-correct arm and gripper supervision plus diagnostics."""
        student_arm = student_raw_output[..., :-1]
        student_gripper_logits = student_raw_output[..., -1:]
        teacher_arm = teacher_actions[..., :-1].clamp(-self.action_clip, self.action_clip)
        teacher_gripper_open = (teacher_actions[..., -1:] >= 0.0).to(student_raw_output.dtype)

        arm_loss_per_sample = self.loss_fn(student_arm, teacher_arm, reduction="none").mean(dim=-1)
        gripper_loss_per_sample = F.binary_cross_entropy_with_logits(
            student_gripper_logits,
            teacher_gripper_open,
            reduction="none",
        ).mean(dim=-1)
        if sample_weights is None:
            arm_loss = arm_loss_per_sample.mean()
            gripper_loss = gripper_loss_per_sample.mean()
        else:
            normalized_weights = sample_weights / sample_weights.sum().clamp_min(1.0e-8)
            arm_loss = torch.sum(arm_loss_per_sample * normalized_weights)
            gripper_loss = torch.sum(gripper_loss_per_sample * normalized_weights)
        arm_mae = (student_arm.detach().clamp(-self.action_clip, self.action_clip) - teacher_arm).abs().mean()
        gripper_accuracy = ((student_gripper_logits.detach() >= 0.0) == teacher_gripper_open.bool()).float().mean()
        teacher_gripper_open_rate = teacher_gripper_open.mean()
        return arm_loss, gripper_loss, arm_mae, gripper_accuracy, teacher_gripper_open_rate

    def _auxiliary_terms(
        self,
        student_latent: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute visual-state representation loss and mean absolute error."""
        if self.auxiliary_loss_weight == 0.0:
            zero = torch.zeros((), device=self.device)
            return zero, zero
        predictions = self.student.predict_visual_state(student_latent)
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Visual-state prediction shape {predictions.shape} does not match target shape {targets.shape}."
            )
        loss_per_sample = self.loss_fn(predictions, targets, reduction="none").mean(dim=-1)
        if sample_weights is None:
            auxiliary_loss = loss_per_sample.mean()
        else:
            normalized_weights = sample_weights / sample_weights.sum().clamp_min(1.0e-8)
            auxiliary_loss = torch.sum(loss_per_sample * normalized_weights)
        auxiliary_mae = (predictions.detach() - targets).abs().mean()
        return auxiliary_loss, auxiliary_mae

    def _camera_batch_diagnostics(self, observations: TensorDict) -> torch.Tensor:
        """Measure image diversity, temporal motion, and image/state alignment."""
        if "base_image" not in observations.keys():
            return torch.zeros(4, device=self.device)
        images = observations["base_image"].float()
        if images.ndim != 4 or images.shape[1] < 3:
            return torch.zeros(4, device=self.device)
        current = images[:, -3:]
        environment_variation = current.std(dim=0, unbiased=False).mean()
        dynamic_range = current.flatten(1).amax(dim=1).sub(current.flatten(1).amin(dim=1)).mean()
        temporal_delta = (current - images[:, -6:-3]).abs().mean() if images.shape[1] >= 6 else current.new_zeros(())

        # Saturated cube colors provide a renderer-independent correspondence
        # check. If images are accidentally permuted across environments, no
        # color centroid should correlate with the matching simulator state.
        alignment = current.new_zeros(())
        teacher_groups = getattr(self.teacher, "obs_groups", [])
        object_start = getattr(
            self.student,
            "teacher_object_start",
            getattr(getattr(self.student, "mlp", None), "teacher_object_start", -1),
        )
        if len(teacher_groups) == 1 and object_start >= 0:
            teacher_raw = observations[teacher_groups[0]]
            if teacher_raw.shape[-1] >= object_start + 9:
                rgb = current + 0.5
                red, green, blue = rgb.unbind(dim=1)
                scores = torch.stack(
                    (
                        torch.relu(red - torch.maximum(green, blue)),
                        torch.relu(green - torch.maximum(red, blue)),
                        torch.relu(blue - torch.maximum(red, green)),
                    ),
                    dim=1,
                )
                height, width = scores.shape[-2:]
                grid_y, grid_x = torch.meshgrid(
                    torch.linspace(-1.0, 1.0, height, device=scores.device),
                    torch.linspace(-1.0, 1.0, width, device=scores.device),
                    indexing="ij",
                )
                mass = scores.sum(dim=(-2, -1)).clamp_min(1.0e-6)
                centroids = torch.stack(
                    (
                        (scores * grid_x).sum(dim=(-2, -1)) / mass,
                        (scores * grid_y).sum(dim=(-2, -1)) / mass,
                    ),
                    dim=-1,
                )
                cube_xy = teacher_raw[..., object_start : object_start + 9].reshape(-1, 3, 3)[..., :2]
                candidate_correlations = []
                for color_id in range(3):
                    for cube_id in range(3):
                        for image_axis in range(2):
                            for world_axis in range(2):
                                x = centroids[:, color_id, image_axis]
                                y = cube_xy[:, cube_id, world_axis]
                                x = x - x.mean()
                                y = y - y.mean()
                                denominator = torch.sqrt(x.square().sum() * y.square().sum()).clamp_min(1.0e-8)
                                candidate_correlations.append((x * y).sum().abs() / denominator)
                alignment = torch.stack(candidate_correlations).amax()
        return torch.stack((environment_variation, dynamic_range, temporal_delta, alignment))

    def _capture_policy_distribution(self) -> tuple[TensorDict, tuple[torch.Tensor, ...]]:
        """Snapshot the student distribution on a representative rollout subset.

        Distillation storage contains images, so using the 16,384 samples from
        the lightweight state-policy PPO diagnostic would add substantial
        memory pressure. Two thousand samples per rank still give a 16,384
        sample estimate in the eight-GPU comparison run.
        """
        observations = self.storage.observations.flatten(0, 1)
        total_samples = observations.shape[0]
        sample_count = min(self.kl_measurement_samples, total_samples)
        if sample_count <= 0:
            raise RuntimeError("Cannot measure policy KL from an empty rollout.")
        sample_indices = (
            torch.arange(sample_count, device=self.device, dtype=torch.long) * total_samples // sample_count
        )
        sampled_observations = observations[sample_indices]
        with torch.no_grad():
            self.student(sampled_observations, stochastic_output=True)
            distribution_params = tuple(
                parameter.detach().clone() for parameter in self.student.output_distribution_params
            )
        return sampled_observations, distribution_params

    def _adapt_learning_rate(self, kl_mean: torch.Tensor) -> torch.Tensor:
        """Synchronize KL and apply RSL-RL's adaptive update rule once per rollout."""
        # ``update`` can be entered from an inference-mode rollout. Cloning
        # here guarantees a normal tensor that distributed collectives may
        # update in place on every rank.
        kl_mean = kl_mean.detach().clone()
        if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size

        if self.schedule == "adaptive" and self.gpu_global_rank == 0:
            if not torch.isfinite(kl_mean) or kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(
                    self.adaptive_learning_rate_min,
                    self.learning_rate / self.adaptive_learning_rate_factor,
                )
            elif 0.0 < kl_mean < self.desired_kl / 2.0:
                self.learning_rate = min(
                    self.adaptive_learning_rate_max,
                    self.learning_rate * self.adaptive_learning_rate_factor,
                )

        if self.is_multi_gpu:
            learning_rate = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(learning_rate, src=0)
            self.learning_rate = learning_rate.item()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate
        return kl_mean

    def update(self) -> dict[str, float]:
        """Optimize the student and expose deterministic imitation diagnostics."""
        kl_observations, old_distribution_params = self._capture_policy_distribution()
        self.num_updates += 1
        metric_sums = torch.zeros(8, device=self.device)
        camera_metric_sums = torch.zeros(4, device=self.device)
        evaluation_metric_sums = torch.zeros((self.recipe_count, 3), device=self.device)
        accumulated_loss: torch.Tensor | int = 0
        batch_count = 0

        self.optimizer.zero_grad()
        for _ in range(self.num_learning_epochs):
            self.student.reset(hidden_state=self.last_hidden_states[0])
            self.teacher.reset(hidden_state=self.last_hidden_states[1])
            self.student.detach_hidden_state()
            for step_index, batch in enumerate(self.storage.generator()):
                if batch.observations is None or batch.privileged_actions is None:
                    raise RuntimeError("Distillation storage yielded an incomplete batch.")

                student_latent = self.student.get_latent(batch.observations)
                student_raw_output = self.student.mlp(student_latent)
                training_slice = slice(self.evaluation_env_count, None)
                recipe_ids = self._recipe_ids(batch.observations)
                sample_weights = (
                    torch.ones_like(recipe_ids[training_slice], dtype=torch.float32) if recipe_ids is not None else None
                )
                if self.recipe_balance and recipe_ids is not None:
                    sample_weights = self._balanced_recipe_weights(recipe_ids[training_slice])
                if not self._rollout_student_control_masks:
                    raise RuntimeError("Missing DAgger controller masks for the collected rollout.")
                student_control = self._rollout_student_control_masks[
                    step_index % len(self._rollout_student_control_masks)
                ][training_slice]
                control_weights = torch.where(
                    student_control,
                    torch.full_like(student_control, self.student_state_loss_weight, dtype=torch.float32),
                    torch.ones_like(student_control, dtype=torch.float32),
                )
                sample_weights = control_weights if sample_weights is None else sample_weights * control_weights
                arm_loss, gripper_loss, arm_mae, gripper_accuracy, gripper_open_rate = self._behavior_terms(
                    student_raw_output[training_slice],
                    batch.privileged_actions[training_slice],
                    sample_weights=sample_weights,
                )
                if self.auxiliary_loss_weight > 0.0:
                    # Supervise independent physical primitives in addition to
                    # the teacher actions. The state head is representation
                    # training only and does not participate in inference.
                    with torch.no_grad():
                        normalized_teacher_state = self.teacher.get_latent(batch.observations)
                        auxiliary_targets = self.student.select_visual_target(normalized_teacher_state)
                else:
                    auxiliary_targets = student_latent.new_empty((student_latent.shape[0], 0))
                auxiliary_loss, auxiliary_mae = self._auxiliary_terms(
                    student_latent[training_slice],
                    auxiliary_targets[training_slice],
                    sample_weights=sample_weights,
                )
                for recipe in range(self.recipe_count):
                    first_env = recipe * self.evaluation_envs_per_recipe
                    last_env = first_env + self.evaluation_envs_per_recipe
                    evaluation_slice = slice(first_env, last_env)
                    evaluation_terms = self._behavior_terms(
                        student_raw_output[evaluation_slice].detach(),
                        batch.privileged_actions[evaluation_slice],
                    )
                    _, evaluation_auxiliary_mae = self._auxiliary_terms(
                        student_latent[evaluation_slice].detach(),
                        auxiliary_targets[evaluation_slice],
                    )
                    evaluation_metric_sums[recipe] += torch.stack(
                        (evaluation_terms[2], evaluation_terms[3], evaluation_auxiliary_mae)
                    )
                behavior_loss = (
                    self.arm_loss_weight * arm_loss
                    + self.gripper_loss_weight * gripper_loss
                    + self.auxiliary_loss_weight * auxiliary_loss
                )
                accumulated_loss = accumulated_loss + behavior_loss
                metric_sums += torch.stack(
                    (
                        behavior_loss.detach(),
                        arm_loss.detach(),
                        gripper_loss.detach(),
                        auxiliary_loss.detach(),
                        arm_mae,
                        gripper_accuracy,
                        gripper_open_rate,
                        auxiliary_mae,
                    )
                )
                camera_metric_sums += self._camera_batch_diagnostics(batch.observations[training_slice])
                batch_count += 1

                if batch_count % self.gradient_length == 0:
                    accumulated_loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.student.detach_hidden_state()
                    accumulated_loss = 0

                if batch.dones is not None:
                    dones = batch.dones.view(-1)
                    self.student.reset(dones)
                    self.teacher.reset(dones)
                    self.student.detach_hidden_state(dones)

        if batch_count % self.gradient_length != 0:
            accumulated_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            if self.max_grad_norm:
                nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
            self.optimizer.step()

        if batch_count == 0:
            raise RuntimeError("Cannot update StackDistillation from an empty rollout.")

        with torch.no_grad():
            self.student(kl_observations, stochastic_output=True)
            new_distribution_params = self.student.output_distribution_params
            kl_mean = self.student.get_kl_divergence(old_distribution_params, new_distribution_params).mean()
        kl_mean = self._adapt_learning_rate(kl_mean)

        metric_sums /= batch_count
        camera_metric_sums /= batch_count
        evaluation_metric_sums /= batch_count
        teacher_control_fraction = self._teacher_control_count / self._total_control_count.clamp_min(1)
        rollout_metrics = teacher_control_fraction.reshape(1)
        evaluation_outcomes = torch.stack((self._evaluation_successes, self._evaluation_attempts))
        if self.is_multi_gpu:
            torch.distributed.all_reduce(metric_sums, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(camera_metric_sums, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(evaluation_metric_sums, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(rollout_metrics, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(evaluation_outcomes, op=torch.distributed.ReduceOp.SUM)
            metric_sums /= self.gpu_world_size
            camera_metric_sums /= self.gpu_world_size
            evaluation_metric_sums /= self.gpu_world_size
            rollout_metrics /= self.gpu_world_size

        self._update_evaluation_statistics(evaluation_outcomes)
        evaluation_success_rates = evaluation_outcomes[0] / evaluation_outcomes[1].clamp_min(1.0)
        evaluation_cumulative_success_rates = self._evaluation_cumulative_successes / (
            self._evaluation_cumulative_attempts.clamp_min(1.0)
        )
        observed_student_fraction = 1.0 - rollout_metrics[0].item()
        student_control_probability = self._student_control_probability()

        self.storage.clear()
        self.last_hidden_states = (self.student.get_hidden_state(), self.teacher.get_hidden_state())
        self.student.detach_hidden_state()
        self._teacher_control_count.zero_()
        self._total_control_count.zero_()
        self._evaluation_successes.zero_()
        self._evaluation_attempts.zero_()
        self._rollout_student_control_masks.clear()

        metrics = {
            "behavior": metric_sums[0].item(),
            "behavior_arm": metric_sums[1].item(),
            "behavior_gripper": metric_sums[2].item(),
            "behavior_auxiliary": metric_sums[3].item(),
            "arm_mae": metric_sums[4].item(),
            "gripper_accuracy": metric_sums[5].item(),
            "teacher_gripper_open_rate": metric_sums[6].item(),
            "auxiliary_mae": metric_sums[7].item(),
            "camera_environment_variation": camera_metric_sums[0].item(),
            "camera_dynamic_range": camera_metric_sums[1].item(),
            "camera_temporal_delta": camera_metric_sums[2].item(),
            "camera_state_alignment": camera_metric_sums[3].item(),
            "teacher_control_fraction": rollout_metrics[0].item(),
            "student_control_fraction": observed_student_fraction,
            "student_control_probability": student_control_probability,
            "student_control_fraction_target": student_control_probability,
            "dagger_active": float(self._dagger_collection_started()),
            "kl": kl_mean.item(),
        }
        for recipe, name in enumerate(self.recipe_names):
            metrics[f"recipe_{name}_eval_arm_mae"] = evaluation_metric_sums[recipe, 0].item()
            metrics[f"recipe_{name}_eval_gripper_accuracy"] = evaluation_metric_sums[recipe, 1].item()
            metrics[f"recipe_{name}_eval_auxiliary_mae"] = evaluation_metric_sums[recipe, 2].item()
            metrics[f"recipe_{name}_eval_success_rate"] = evaluation_success_rates[recipe].item()
            metrics[f"recipe_{name}_eval_cumulative_success_rate"] = evaluation_cumulative_success_rates[recipe].item()
            metrics[f"recipe_{name}_eval_success_ema"] = self._evaluation_success_ema[recipe].item()
            metrics[f"recipe_{name}_eval_attempts"] = evaluation_outcomes[1, recipe].item()

        table_metrics = evaluation_metric_sums[self.table_recipe_id]
        metrics.update(
            {
                "table_eval_arm_mae": table_metrics[0].item(),
                "table_eval_gripper_accuracy": table_metrics[1].item(),
                "table_eval_auxiliary_mae": table_metrics[2].item(),
                "table_eval_success_rate": evaluation_success_rates[self.table_recipe_id].item(),
                "table_eval_cumulative_success_rate": evaluation_cumulative_success_rates[self.table_recipe_id].item(),
                "table_eval_attempts": evaluation_outcomes[1, self.table_recipe_id].item(),
            }
        )
        return metrics

    def save(self) -> dict:
        """Save DAgger progress and evaluation state alongside native models."""
        saved = super().save()
        saved["stack_distillation_state"] = {
            "num_updates": self.num_updates,
            "evaluation_cumulative_successes": self._evaluation_cumulative_successes.tolist(),
            "evaluation_cumulative_attempts": self._evaluation_cumulative_attempts.tolist(),
            "evaluation_success_ema": self._evaluation_success_ema.tolist(),
            "evaluation_ema_initialized": self._evaluation_ema_initialized.tolist(),
        }
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Resume collection scheduling, while PPO checkpoints still load only the teacher."""
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        self.learning_rate = float(self.optimizer.param_groups[0]["lr"])
        if load_iteration:
            state = loaded_dict["stack_distillation_state"]
            self.num_updates = int(state["num_updates"])
            for key, target in (
                ("evaluation_cumulative_successes", self._evaluation_cumulative_successes),
                ("evaluation_cumulative_attempts", self._evaluation_cumulative_attempts),
                ("evaluation_success_ema", self._evaluation_success_ema),
            ):
                value = torch.as_tensor(state[key], dtype=target.dtype, device=target.device).flatten()
                if value.numel() != target.numel():
                    raise ValueError(f"Saved {key} does not match the configured recipe count.")
                target.copy_(value)
            initialized = torch.as_tensor(
                state["evaluation_ema_initialized"],
                dtype=torch.bool,
                device=self.device,
            ).flatten()
            if initialized.numel() != self._evaluation_ema_initialized.numel():
                raise ValueError("Saved evaluation EMA state does not match the configured recipe count.")
            self._evaluation_ema_initialized.copy_(initialized)
            self._teacher_control_mask = None
            self._episode_success_seen.zero_()
            self._rollout_student_control_masks.clear()
        return load_iteration


@configclass
class StackDistillationAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    """Configuration for visual behavior cloning with per-step DAgger."""

    class_name: str = "isaaclab_tasks.contrib.stack.config.franka.agents.rsl_rl_distillation_cfg:StackDistillation"
    teacher_pretrain_updates: int = 40
    student_control_fraction: float = 0.25
    evaluation_success_ema_alpha: float = 0.25
    arm_loss_weight: float = 1.0
    gripper_loss_weight: float = 2.0
    auxiliary_loss_weight: float = 0.5
    action_clip: float = 1.0
    evaluation_envs_per_recipe: int = 4
    success_reward_threshold: float = 1.0
    recipe_count: int = 9
    recipe_names: tuple[str, ...] = (
        "final_release",
        "second_place",
        "second_transport",
        "second_pick",
        "pair_ready",
        "first_place",
        "first_transport",
        "first_pick",
        "table",
    )
    recipe_balance: bool = True
    table_recipe_weight: float = 3.0
    student_state_loss_weight: float = 3.0
    distillation_context_obs_group: str = "distillation_context"
    table_recipe_id: int = 8
    schedule: str = "fixed"
    desired_kl: float = 0.01
    adaptive_learning_rate_min: float = 1.0e-5
    adaptive_learning_rate_max: float = 3.0e-4
    adaptive_learning_rate_factor: float = 1.5
    kl_measurement_samples: int = 2048


@configclass
class FrankaStackCameraDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Clone the state teacher into the deployable RGB-plus-proprio actor."""

    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 50
    experiment_name = "franka_stack_camera"
    run_name = "distillation"
    clip_actions = 1.0
    init_at_random_ep_len = False
    obs_groups = {
        "student": ["policy", "base_image"],
        "teacher": ["teacher"],
    }
    student = StackVisualDistillationModelCfg(
        obs_normalization=True,
        hidden_dims=[1024, 512, 256],
        activation="elu",
        # Distillation collection is deterministic, but the RSL-RL model and
        # exported policy still require a valid action distribution.
        distribution_cfg=StackDistillationDistributionCfg(init_std=0.05, std_range=(0.02, 0.12)),
        cnn_cfg=StackVisualDistillationModelCfg.CNNCfg(
            output_channels=[32, 64, 64],
            kernel_size=[8, 4, 3],
            stride=[4, 2, 1],
            activation="elu",
        ),
    )
    teacher = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=StackGaussianDistributionCfg(init_std=0.45),
    )
    algorithm = StackDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-4,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="huber",
    )
