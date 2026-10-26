# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Short, independently seeded latent rollouts with an unchanged policy interface.

No reference future particles, actions or observations enter a rollout. This is
a training surrogate, not an MPM integrator or a particle renderer. Real physics
must determine whether a policy trained here improves the task.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from .dataset import EpisodeWindows, task_specs


def observation_groups(observation: torch.Tensor, critic: torch.Tensor, task_id: str) -> TensorDict:
    """Keep Franka's trained concatenation order; other tasks use their own head."""
    if task_id == "franka_pour":
        if observation.shape[-1] != 84 or critic.shape[-1] != 86:
            raise ValueError("Franka Pour requires the original 84/86 observation interface")
        groups = {"policy": observation[..., :62], "media": observation[..., 62:], "privileged": critic[..., 84:]}
    else:
        groups = {"policy": observation, "privileged": critic}
    return TensorDict(groups, batch_size=observation.shape[:-1], device=observation.device)


class LearnedParticleEnv:
    """Vectorized recurrent ensemble rollouts from a training-only real seed bank.

    Each member maintains its own autoregressive state. One member per world is
    sampled at reset; disagreement across members can shorten uncertain rollouts.
    Resetting never consults validation/test episodes or later reference frames.
    """

    def __init__(
        self,
        models,
        episodes,
        *,
        task_id: str,
        num_envs: int = 1024,
        horizon: int = 8,
        seed_bank_size: int = 4096,
        uncertainty_limit: float = 2.0,
        uncertainty_penalty: float = 0.0,
        mass_tolerance: float = 0.05,
        action_limit: float = 1.0,
        device: str = "cuda:0",
        seed: int = 42,
    ):
        if min(num_envs, horizon, seed_bank_size) < 1 or not models:
            raise ValueError("World, horizon, bank and ensemble sizes must be positive")
        if not 0.0 < action_limit <= 1.0:
            raise ValueError("Action limit must lie in (0, 1]")
        self.models = [model.to(device).eval().requires_grad_(False) for model in models]
        self.task_id, self.device, self.num_envs = task_id, device, num_envs
        spec = next(spec for spec in models[0].config.tasks if spec.task_id == task_id)
        if task_id == "rizon_sharpa_teapot_full":
            recorded_spec = task_specs([episode for episode in episodes if episode["task_id"] == task_id])
            if recorded_spec != [spec] or any(
                next(item for item in model.config.tasks if item.task_id == task_id) != spec for model in models
            ):
                raise ValueError("Full teapot model checkpoints must match the recorded native task interface")
        self.task_spec = spec
        self.num_actions, self.max_episode_length = spec.action_dim, horizon
        self.uncertainty_limit, self.uncertainty_penalty = uncertainty_limit, uncertainty_penalty
        self.mass_tolerance = mass_tolerance
        self.action_limit = float(action_limit)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.cfg = {
            "task_id": task_id,
            "surrogate": True,
            "imagination_horizon": horizon,
            "action_limit": self.action_limit,
        }
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        windows = EpisodeWindows(episodes, task_id=task_id, split="train", horizon=1, seed=seed)
        rewards = torch.cat(
            [
                episode["reward"][
                    episode.get("valid_mask", episode["finite_mask"])[:-1]
                    & episode.get("valid_mask", episode["finite_mask"])[1:]
                    & torch.isfinite(episode["action"]).all(-1)
                ]
                for episode in windows.episodes
            ]
        )
        rewards = rewards[torch.isfinite(rewards)]
        self.reward_min, self.reward_max = float(rewards.min()), float(rewards.max())
        banks = {key: [] for key in ("observation", "critic_observation", "context")}
        hidden = [[] for _ in models]
        with torch.no_grad():
            for start in range(0, seed_bank_size, 256):
                batch = windows.sample(min(256, seed_bank_size - start), device=device)
                for key in ("observation", "critic_observation"):
                    banks[key].append(batch[key][:, 0])
                banks["context"].append(batch["context"])
                for index, model in enumerate(models):
                    hidden[index].append(model.encode_batch(batch, task_id=task_id))
        self.bank = {key: torch.cat(values) for key, values in banks.items()}
        self.hidden_bank = [torch.cat(values) for values in hidden]
        self.observation_scale = models[0].task(task_id).observation_norm.std.clamp_min(0.01)
        self.observations = [self.bank["observation"].new_empty((num_envs, spec.observation_dim)) for _ in models]
        self.critics = [
            self.bank["critic_observation"].new_empty((num_envs, spec.critic_observation_dim)) for _ in models
        ]
        self.hidden = [value.new_empty((num_envs, value.shape[-1])) for value in self.hidden_bank]
        self.context = self.bank["context"].new_empty((num_envs, spec.context_dim))
        self.member = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.mass_indices = {
            "franka_pour": [68, 69, 70, 83],
            "rizon_sharpa_teapot_held_pour": [91, 92, 93, 94, 95],
            "rizon_sharpa_teapot_full": [91, 92, 93, 94, 95],
        }.get(task_id, list(getattr(spec, "mass_fraction_indices", [])))
        self._rows = torch.arange(num_envs, device=device)
        self._reset(self._rows)

    @torch.no_grad()
    def _reset(self, rows):
        indices = torch.randint(
            len(self.bank["observation"]), (len(rows),), generator=self.generator, device=self.device
        )
        self.context[rows] = self.bank["context"][indices]
        for index in range(len(self.models)):
            self.observations[index][rows] = self.bank["observation"][indices]
            self.critics[index][rows] = self.bank["critic_observation"][indices]
            self.hidden[index][rows] = self.hidden_bank[index][indices]
        self.member[rows] = torch.randint(len(self.models), (len(rows),), generator=self.generator, device=self.device)
        self.episode_length_buf[rows] = 0

    def _selected(self, values):
        return torch.stack(values)[self.member, self._rows]

    def get_observations(self):
        return observation_groups(self._selected(self.observations), self._selected(self.critics), self.task_id)

    @torch.no_grad()
    def step(self, actions):
        """Predict one action, expose its terminal endpoint, then reset finished worlds."""
        actions = actions.clamp(-self.action_limit, self.action_limit)
        previous = self._selected(self.observations)
        rewards, probabilities = [], []
        for index, model in enumerate(self.models):
            result = model.step(
                self.observations[index],
                self.critics[index],
                actions,
                self.context,
                self.hidden[index],
                task_id=self.task_id,
            )
            self.observations[index], self.critics[index], self.hidden[index] = (
                result["observation"],
                result["critic_observation"],
                result["hidden"],
            )
            rewards.append(result["reward"])
            probabilities.append(result["termination_logit"].sigmoid())
        ensemble = torch.stack(self.observations)
        disagreement = (ensemble.std(dim=0, unbiased=False) / self.observation_scale).square().mean(-1).sqrt()
        reward = self._selected(rewards)
        inadmissible = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.task_id == "rizon_sharpa_teapot_held_pour":
            selected = self._selected(self.observations)
            # These are the adapter's exact all-cloud mass fraction observations.
            # The reward formula matches collect_teapot.pour_reward; the model's
            # independent reward head is not allowed to invent pouring progress.
            reward = 100.0 * (selected[:, 92] - previous[:, 92])
            reward -= 200.0 * (selected[:, 93] - previous[:, 93]) + 0.001 * actions.square().sum(-1)
        elif self.task_id == "rizon_sharpa_teapot_full":
            from .teapot_full_task import full_task_reward

            # Share shaping algebra with native control. This does not establish
            # that predicted grasp/contact states are accurate enough for PPO.
            reward = full_task_reward(previous, self._selected(self.observations), actions)
        if self.mass_indices:
            fractions = ensemble[..., self.mass_indices]
            tolerance = self.mass_tolerance
            invalid = (fractions < -tolerance).any(-1) | (fractions > 1 + tolerance).any(-1)
            invalid |= (fractions.sum(-1) - 1).abs() > tolerance
            inadmissible = invalid.any(0)
        finite = torch.isfinite(ensemble).all(dim=(0, 2)) & torch.isfinite(disagreement) & torch.isfinite(reward)
        for values in (self.critics, self.hidden):
            finite &= torch.isfinite(torch.stack(values)).all(dim=(0, 2))
        for values in (rewards, probabilities):
            finite &= torch.isfinite(torch.stack(values)).all(dim=0)
        physical_done = self._selected(probabilities) > 0.5
        self.episode_length_buf += 1
        native_timeout = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        spec = self.task_spec
        if spec.typed_countdown_index >= 0 and spec.typed_countdown_decrement > 0.0:
            # Native task clocks remain separate from the shorter imagination
            # horizon. Their indices/units come from the validated task spec.
            selected = self._selected(self.observations)
            expired = selected[:, spec.typed_countdown_index] <= 0.0
            if spec.typed_elapsed_time_index >= 0:
                duration = spec.typed_elapsed_time_increment / spec.typed_countdown_decrement
                expired |= selected[:, spec.typed_elapsed_time_index] >= duration
            native_timeout = finite & expired
        cutoff = (self.episode_length_buf >= self.max_episode_length) | (disagreement > self.uncertainty_limit)
        cutoff |= inadmissible | native_timeout
        terminated = physical_done | ~finite
        truncated = cutoff & ~terminated
        done = terminated | truncated
        reward = torch.nan_to_num(reward, nan=self.reward_min).clamp(self.reward_min, self.reward_max)
        reward -= self.uncertainty_penalty * torch.nan_to_num(disagreement, nan=0.0)
        # Preserve the actual predicted endpoint before resetting. PPO uses this
        # value for truncation bootstrapping, never the unrelated replay seed.
        terminal_observations = self.get_observations().clone()
        terminal_observations = terminal_observations.apply(lambda x: torch.nan_to_num(x, posinf=0.0, neginf=0.0))
        extras = {
            "terminal_observations": terminal_observations,
            "truncated": truncated,
            "terminated": terminated,
            "disagreement": disagreement,
            "nonfinite": ~finite,
            "inadmissible": inadmissible,
            "native_timeout": native_timeout & ~terminated,
        }
        self._reset(done.nonzero().flatten())
        return self.get_observations(), reward, done, extras


def bootstrap_truncations(reward, truncated, terminal_value, gamma):
    """Bootstrap artificial rollout cuts at their endpoint, with no reset leakage."""
    return reward + torch.where(truncated, gamma * terminal_value.reshape_as(reward), 0.0)
