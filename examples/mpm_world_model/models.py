# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental compact coupled dynamics for policy training, without a particle decoder.

Entity encoders initialize recurrent memory from a real simulator snapshot. All
subsequent predictions use only that memory, predicted observations and actions;
future reference particle or rigid states never enter an imagined rollout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class TaskSpec:
    """Observation and actuator dimensions of one task-specific interface."""

    task_id: str
    observation_dim: int
    critic_observation_dim: int
    action_dim: int
    context_dim: int
    controller_dim: int = 0
    shared_observation_prefix_dim: int = 0
    last_action_observation_start: int = -1
    last_action_scale: float = 1.0
    time_remaining_index: int = -1
    time_remaining_decrement: float = 0.0


@dataclass
class WorldModelConfig:
    """Network configuration stored alongside trained weights."""

    tasks: list[TaskSpec]
    hidden_dim: int = 128
    entity_dim: int = 64
    material_dim: int = 29
    material_transform: str = "signed_log1p"


class FeatureNormalizer(nn.Module):
    """Accumulate real training-data moments; imagination never updates them."""

    def __init__(self, dimension: int, *, floor: float = 1e-3):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("std", torch.ones(dimension))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("m2", torch.zeros(dimension, dtype=torch.float64))
        self.floor = floor

    @torch.no_grad()
    def update(self, values: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """Merge finite, optionally masked samples from training episodes only."""
        if self.mean.numel() == 0:
            return
        values = values.detach().reshape(-1, self.mean.numel()).to(self.mean.device, torch.float64)
        if mask is not None:
            values = values[mask.reshape(-1).to(values.device, torch.bool)]
        values = values[torch.isfinite(values).all(-1)]
        if not len(values):
            return
        n = len(values)
        batch_mean = values.mean(0)
        batch_m2 = ((values - batch_mean) ** 2).sum(0)
        total = self.count + n
        delta = batch_mean - self.mean.double()
        merged_mean = self.mean.double() + delta * n / total
        self.m2.add_(batch_m2 + delta.square() * self.count * n / total)
        self.mean.copy_(merged_mean)
        self.count.copy_(total)
        self.std.copy_((self.m2 / total).sqrt().clamp_min(self.floor))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return ((values - self.mean) / self.std).clamp(-20.0, 20.0)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim)
    )


class _SetEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.normalizer = FeatureNormalizer(input_dim)
        self.network = _mlp(input_dim, output_dim, output_dim)

    def forward(self, state, mask, weights=None):
        # Remove padding before nonlinear operations: NaN * zero is still NaN.
        state = torch.where(mask.unsqueeze(-1), state, torch.zeros_like(state))
        encoded = self.network(self.normalizer(state))
        weight = mask.to(encoded.dtype) if weights is None else weights * mask
        mean = (encoded * weight.unsqueeze(-1)).sum(-2) / weight.sum(-1, keepdim=True).clamp_min(1e-12)
        maximum = encoded.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(-2)
        maximum = torch.where(mask.any(-1, keepdim=True), maximum, torch.zeros_like(maximum))
        return torch.cat((mean, maximum, torch.log1p(weight.sum(-1, keepdim=True))), dim=-1)


class _TaskHeads(nn.Module):
    def __init__(self, spec: TaskSpec, config: WorldModelConfig):
        super().__init__()
        self.observation_norm = FeatureNormalizer(spec.observation_dim)
        self.critic_norm = FeatureNormalizer(spec.critic_observation_dim)
        self.action_norm = FeatureNormalizer(spec.action_dim)
        self.context_norm = FeatureNormalizer(spec.context_dim)
        self.controller_norm = FeatureNormalizer(spec.controller_dim)
        self.observation_delta_norm = FeatureNormalizer(spec.observation_dim)
        self.critic_delta_norm = FeatureNormalizer(spec.critic_observation_dim)
        self.reward_norm = FeatureNormalizer(1)
        state_dim = spec.observation_dim + spec.critic_observation_dim + spec.context_dim
        self.initial = _mlp(
            state_dim + spec.controller_dim + 4 * config.entity_dim + 2 + config.material_dim,
            config.hidden_dim,
            config.hidden_dim,
        )
        self.transition_input = _mlp(state_dim + spec.action_dim, config.hidden_dim, config.hidden_dim)
        self.output = _mlp(
            config.hidden_dim,
            config.hidden_dim,
            spec.observation_dim + spec.critic_observation_dim - spec.shared_observation_prefix_dim + 2,
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        with torch.no_grad():
            self.output[-1].bias[-1] = -4.6


class CoupledWorldModel(nn.Module):
    """Experimental set encoder and recurrent action-conditioned policy surrogate.

    Positions must be relative to the environment origin in metres, velocities
    are in m/s, and rigid states use xyz, quaternion WXYZ, linear/angular velocity.
    Context names, geometry, material parameters and dt belong in dataset metadata.
    It predicts task observations rather than a complete physical particle state.
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        if not config.tasks or len({spec.task_id for spec in config.tasks}) != len(config.tasks):
            raise ValueError("World-model task identifiers must be nonempty and unique")
        for spec in config.tasks:
            if not 0 <= spec.shared_observation_prefix_dim <= min(spec.observation_dim, spec.critic_observation_dim):
                raise ValueError("Shared actor/critic prefix exceeds its observation dimensions")
        self.config = config
        if config.material_transform not in ("identity", "signed_log1p"):
            raise ValueError("Unsupported material aggregate transform")
        self._specs = {spec.task_id: spec for spec in config.tasks}
        self._task_indices = {spec.task_id: str(i) for i, spec in enumerate(config.tasks)}
        self.particle_encoder = _SetEncoder(6, config.entity_dim)
        self.rigid_encoder = _SetEncoder(13, config.entity_dim)
        self.material_norm = FeatureNormalizer(config.material_dim)
        self.heads = nn.ModuleDict({str(i): _TaskHeads(spec, config) for i, spec in enumerate(config.tasks)})
        self.transition = nn.GRUCell(config.hidden_dim, config.hidden_dim)

    def task(self, task_id: str) -> _TaskHeads:
        return self.heads[self._task_indices[task_id]]

    def observation_scale(self, task_id: str) -> torch.Tensor:
        """Return real training-observation scales for ensemble disagreement."""
        return self.task(task_id).observation_norm.std

    def material_features(self, summary: torch.Tensor) -> torch.Tensor:
        """Transform the supplied aggregate, preserving its final validity flag.

        This is signed-log of the full-cloud mean, not mean of particle logs.
        Raw recordings are retained; the versioned checkpoint fixes the encoding.
        """
        if self.config.material_transform == "identity":
            return summary
        physical = summary[..., :-1]
        return torch.cat((physical.sign() * physical.abs().log1p(), summary[..., -1:]), -1)

    def _state_features(self, observation, critic_observation, context, task_id):
        heads = self.task(task_id)
        return torch.cat(
            (heads.observation_norm(observation), heads.critic_norm(critic_observation), heads.context_norm(context)),
            -1,
        )

    def encode(
        self,
        observation,
        critic_observation,
        particle_state,
        particle_mask,
        rigid_state,
        rigid_mask,
        context,
        *,
        task_id: str,
        controller_state=None,
        material_summary=None,
        particle_weights=None,
    ):
        """Initialize memory from one real, masked simulator snapshot."""
        heads = self.task(task_id)
        if controller_state is None:
            controller_state = observation.new_zeros((*observation.shape[:-1], heads.controller_norm.mean.numel()))
        if material_summary is None:
            material_summary = observation.new_zeros((*observation.shape[:-1], self.config.material_dim))
        features = torch.cat(
            (
                self._state_features(observation, critic_observation, context, task_id),
                heads.controller_norm(controller_state),
                self.particle_encoder(particle_state, particle_mask, particle_weights),
                self.rigid_encoder(rigid_state, rigid_mask),
                self.material_norm(self.material_features(material_summary)),
            ),
            -1,
        )
        return torch.tanh(heads.initial(features))

    def step(self, observation, critic_observation, action, context, hidden, *, task_id: str):
        """Advance one recorded control interval without reference-state inputs."""
        heads = self.task(task_id)
        spec = self._specs[task_id]
        if spec.last_action_observation_start >= 0:
            action = action.clamp(-1.0, 1.0)
        inputs = torch.cat(
            (self._state_features(observation, critic_observation, context, task_id), heads.action_norm(action)), -1
        )
        hidden = self.transition(heads.transition_input(inputs), hidden)
        output = heads.output(hidden)
        actor_dim, critic_dim = observation.shape[-1], critic_observation.shape[-1]
        prefix = self._specs[task_id].shared_observation_prefix_dim
        actor_delta, critic_delta, reward, termination = output.split((actor_dim, critic_dim - prefix, 1, 1), -1)
        next_observation = (
            observation + actor_delta * heads.observation_delta_norm.std + heads.observation_delta_norm.mean
        )
        if spec.last_action_observation_start >= 0:
            start = spec.last_action_observation_start
            next_observation = torch.cat(
                (
                    next_observation[..., :start],
                    action * spec.last_action_scale,
                    next_observation[..., start + action.shape[-1] :],
                ),
                -1,
            )
        if spec.time_remaining_index >= 0:
            index = spec.time_remaining_index
            time_remaining = (observation[..., index : index + 1] - spec.time_remaining_decrement).clamp(0.0, 1.0)
            next_observation = torch.cat(
                (next_observation[..., :index], time_remaining, next_observation[..., index + 1 :]), -1
            )
        private_critic = (
            critic_observation[..., prefix:]
            + critic_delta * heads.critic_delta_norm.std[prefix:]
            + heads.critic_delta_norm.mean[prefix:]
        )
        return {
            "observation": next_observation,
            "critic_observation": torch.cat((next_observation[..., :prefix], private_critic), -1),
            "reward": (reward * heads.reward_norm.std + heads.reward_norm.mean).squeeze(-1),
            "termination_logit": termination.squeeze(-1),
            "hidden": hidden,
        }

    def encode_batch(self, batch, *, task_id: str):
        """Encode the first frame of a padded training window."""
        return self.encode(
            batch["observation"][:, 0],
            batch["critic_observation"][:, 0],
            batch["particle_state"][:, 0],
            batch["particle_mask"],
            batch["rigid_state"][:, 0],
            batch["rigid_mask"],
            batch["context"],
            task_id=task_id,
            controller_state=batch["controller_state"][:, 0],
            material_summary=batch["material_summary"][:, 0],
            particle_weights=batch["particle_weights"],
        )


def rollout_loss(model: CoupledWorldModel, batch: dict, *, task_id: str, burn_in: int = 0):
    """Fit a teacher-free unroll, masking padding and post-terminal transitions.

    Burn-in transitions use logged current observations to initialize memory;
    every optimized transition thereafter consumes the model's own observation.
    """
    hidden = model.encode_batch(batch, task_id=task_id)
    observation, critic_observation = batch["observation"][:, 0], batch["critic_observation"][:, 0]
    heads = model.task(task_id)
    losses = {key: observation.new_zeros(()) for key in ("observation", "critic_observation", "reward", "termination")}
    weight_sum = observation.new_zeros(())
    for step in range(batch["action"].shape[1]):
        if step < burn_in:
            observation, critic_observation = batch["observation"][:, step], batch["critic_observation"][:, step]
        prediction = model.step(
            observation,
            critic_observation,
            batch["action"][:, step],
            batch["context"],
            hidden,
            task_id=task_id,
        )
        hidden = prediction["hidden"]
        observation, critic_observation = prediction["observation"], prediction["critic_observation"]
        if step < burn_in:
            continue
        mask = batch["transition_mask"][:, step]
        valid = mask.bool()
        for key, scale in (
            ("observation", heads.observation_delta_norm.std),
            ("critic_observation", heads.critic_delta_norm.std),
        ):
            target = batch[key][:, step + 1]
            predicted = torch.where(valid[:, None], prediction[key], 0.0)
            target = torch.where(valid[:, None], target, 0.0)
            error = ((predicted - target) / scale).square().mean(-1)
            losses[key] = losses[key] + (error * mask).sum()
        predicted_reward = torch.where(valid, prediction["reward"], 0.0)
        target_reward = torch.where(valid, batch["reward"][:, step], 0.0)
        reward_error = ((predicted_reward - target_reward) / heads.reward_norm.std[0]).square()
        losses["reward"] = losses["reward"] + (reward_error * mask).sum()
        termination_error = nn.functional.binary_cross_entropy_with_logits(
            torch.where(valid, prediction["termination_logit"], 0.0),
            torch.where(valid, batch["terminated"][:, step].float(), 0.0),
            reduction="none",
        )
        losses["termination"] = losses["termination"] + (termination_error * mask).sum()
        weight_sum = weight_sum + mask.sum()
    losses = {key: value / weight_sum.clamp_min(1) for key, value in losses.items()}
    total = losses["observation"] + 0.5 * losses["critic_observation"] + losses["reward"] + losses["termination"]
    return total, losses


def load_checkpoint(path: str | Path, *, device: str = "cpu"):
    """Load trusted project weights and their training-data normalizers."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported coupled world-model checkpoint format")
    config_dict = dict(checkpoint["config"])
    # Version-1 checkpoints created before the explicit transform used raw
    # aggregate moments; never reinterpret their learned normalization silently.
    config_dict.setdefault("material_transform", "identity")
    config_dict["tasks"] = [TaskSpec(**spec) for spec in config_dict["tasks"]]
    models = []
    for weights in checkpoint["models"]:
        model = CoupledWorldModel(WorldModelConfig(**config_dict)).to(device)
        model.load_state_dict(weights)
        model.eval()
        models.append(model)
    return models, checkpoint


def checkpoint_payload(models: list[CoupledWorldModel], **metadata):
    return {
        "format_version": 1,
        "config": asdict(models[0].config),
        "models": [m.state_dict() for m in models],
        **metadata,
    }
