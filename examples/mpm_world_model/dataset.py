# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode-held-out data and terminal-safe windows for compact world models."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from .models import CoupledWorldModel, TaskSpec


def _metadata(episode, name, default=None):
    return episode.get(name, episode.get("metadata", {}).get(name, default))


def load_episodes(root: str | Path, *, verify_hashes: bool = True) -> list[dict]:
    """Load trusted recorder shards and reject reset-group train/test leakage."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    episodes, group_splits = [], {}
    for entry in manifest["episodes"]:
        path = (root / entry["file"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Episode paths must remain inside the dataset directory")
        if verify_hashes and entry.get("sha256"):
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Episode checksum mismatch: {path}")
        episode = torch.load(path, map_location="cpu", weights_only=False)
        episode["task_id"] = str(_metadata(episode, "task_id", entry.get("task_id")))
        episode["reset_group"] = str(_metadata(episode, "reset_group", entry.get("reset_group")))
        episode["episode_id"] = str(_metadata(episode, "episode_id", path.stem))
        if episode["task_id"] == "None" or episode["reset_group"] == "None":
            raise ValueError("Every episode needs task_id and reset_group metadata")
        episode["split"] = entry["split"]
        episode["source_file"] = str(path)
        group = (episode["task_id"], episode["reset_group"])
        if group in group_splits and group_splits[group] != episode["split"]:
            raise ValueError(f"Reset group occurs in different dataset splits: {group}")
        group_splits[group] = episode["split"]
        _validate_episode(episode)
        episodes.append(episode)
    if not episodes:
        raise ValueError("Dataset contains no completed episode shards")
    return episodes


def _validate_episode(episode):
    action = torch.as_tensor(episode["action"], dtype=torch.float32)
    count = len(action)
    if count < 1 or action.ndim != 2:
        raise ValueError("Episode actions must have shape [transitions, action_dim]")
    for key in ("observation", "critic_observation", "particle_state", "rigid_state"):
        value = torch.as_tensor(episode[key], dtype=torch.float32)
        if len(value) != count + 1:
            raise ValueError(f"{key} needs one more state than action transitions")
        episode[key] = value
    if episode["particle_state"].ndim != 3 or episode["particle_state"].shape[-1] != 6:
        raise ValueError("particle_state must be [states, sampled_particles, 6]")
    if episode["rigid_state"].ndim != 3 or episode["rigid_state"].shape[-1] != 13:
        raise ValueError("rigid_state must be [states, bodies, 13], using quaternion WXYZ")
    episode["action"] = action
    for key in ("reward", "terminated", "truncated"):
        value = torch.as_tensor(episode[key], dtype=torch.float32 if key == "reward" else torch.bool)
        if value.shape != (count,):
            raise ValueError(f"{key} must have shape [transitions]")
        episode[key] = value
    done = episode["terminated"] | episode["truncated"]
    if done[:-1].any():
        raise ValueError("Episode shard crosses an autoreset/terminal boundary")
    episode["context"] = torch.as_tensor(episode["context"], dtype=torch.float32)
    if episode["context"].ndim == 2:
        if len(episode["context"]) != count + 1 or not torch.equal(
            episode["context"], episode["context"][0].expand_as(episode["context"])
        ):
            raise ValueError("Repeated episode context must remain constant across time")
        episode["context"] = episode["context"][0]
    if episode["context"].ndim != 1 or not torch.isfinite(episode["context"]).all():
        raise ValueError("Episode context must be a finite static vector")
    finite = torch.as_tensor(episode.get("finite_mask", torch.ones(count + 1)), dtype=torch.bool)
    if finite.shape != (count + 1,):
        raise ValueError("finite_mask must contain one Boolean per state")
    for key in ("observation", "critic_observation"):
        finite = finite & torch.isfinite(episode[key]).flatten(1).all(-1)
    episode["finite_mask"] = finite
    controller = torch.as_tensor(episode.get("controller_state", torch.zeros(count + 1, 0)), dtype=torch.float32)
    if controller.ndim != 2 or len(controller) != count + 1:
        raise ValueError("controller_state must contain one vector per state")
    episode["finite_mask"] &= torch.isfinite(controller).all(-1)
    episode["controller_state"] = controller
    n, r = episode["particle_state"].shape[1], episode["rigid_state"].shape[1]
    if min(n, r) < 1:
        raise ValueError("Entity arrays need at least one allocated slot; use masks for absent entities")
    episode["particle_mask"] = torch.as_tensor(episode.get("particle_mask", torch.ones(n)), dtype=torch.bool)
    episode["rigid_mask"] = torch.as_tensor(episode.get("rigid_mask", torch.ones(r)), dtype=torch.bool)
    episode["particle_weights"] = torch.as_tensor(episode.get("particle_weights", torch.ones(n)), dtype=torch.float32)
    if episode["particle_mask"].shape != (n,) or episode["rigid_mask"].shape != (r,):
        raise ValueError("Entity masks must match the allocated particle and body counts")
    weights = episode["particle_weights"]
    if weights.shape != (n,) or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Particle weights must be a finite nonnegative vector")
    if episode["task_id"] == "franka_pour" and _metadata(episode, "particle_weight_units") is None:
        mass = torch.as_tensor(episode["particle_mass"], dtype=torch.float32)
        if mass.shape != (n,) or not torch.isfinite(mass).all() or (mass <= 0).any():
            raise ValueError("Legacy Franka sampling weights require positive per-particle masses")
        episode["particle_weights"] = weights * mass
        episode["particle_weight_units"] = "kg"
        episode["particle_weights_conversion"] = "legacy Franka multiplicity times particle mass"
    for key, mask in (("particle_state", episode["particle_mask"]), ("rigid_state", episode["rigid_mask"])):
        episode["finite_mask"] &= (torch.isfinite(episode[key]).all(-1) | ~mask[None]).all(-1)
    episode["material_summary"] = _material_summary(episode)


def _material_summary(episode):
    count = len(episode["observation"])
    if "material_summary" in episode:
        summary = torch.as_tensor(episode["material_summary"], dtype=torch.float32)
        if summary.shape != (count, 29) or not torch.isfinite(summary).all():
            raise ValueError("Explicit full-cloud material_summary must be finite with shape [states, 29]")
        return summary
    if "material_state" not in episode:
        return torch.zeros(count, 29)
    material = torch.as_tensor(episode["material_state"], dtype=torch.float32)
    if material.shape != (*episode["particle_state"].shape[:2], 28):
        raise ValueError("material_state must contain 28 values per sampled particle and state")
    valid = torch.as_tensor(episode.get("material_state_valid", torch.ones(material.shape[:2])), dtype=torch.bool)
    valid = valid & episode["particle_mask"][None] & torch.isfinite(material).all(-1)
    weights = valid * episode["particle_weights"][None]
    total = weights.sum(-1, keepdim=True)
    mean = (torch.where(valid[..., None], material, 0) * weights[..., None]).sum(1) / total.clamp_min(1e-12)
    available = (total > 0).float()
    return torch.cat((mean, available), -1)


def task_specs(episodes: list[dict]) -> list[TaskSpec]:
    """Infer task interfaces and reject incompatible episodes under one task ID."""
    specs = {}
    field_names = {}
    for episode in episodes:
        actor_dim = episode["observation"].shape[-1]
        finite = episode["finite_mask"]
        shared_prefix = 0
        if episode["critic_observation"].shape[-1] >= actor_dim and torch.equal(
            episode["observation"][finite], episode["critic_observation"][finite, :actor_dim]
        ):
            shared_prefix = actor_dim
        spec = TaskSpec(
            task_id=episode["task_id"],
            observation_dim=episode["observation"].shape[-1],
            critic_observation_dim=episode["critic_observation"].shape[-1],
            action_dim=episode["action"].shape[-1],
            context_dim=episode["context"].shape[-1],
            controller_dim=episode["controller_state"].shape[-1],
            shared_observation_prefix_dim=shared_prefix,
        )
        if spec.task_id == "franka_pour" and (spec.observation_dim, spec.critic_observation_dim, spec.action_dim) == (
            84,
            86,
            8,
        ):
            # The unchanged task uses 12 s episodes and a step counter in its
            # normalized remaining-time observation. Validate the recorded
            # algebra instead of learning it as uncertain material dynamics.
            dt = float(_metadata(episode, "dt"))
            max_steps = int(_metadata(episode, "max_episode_length", math.ceil(12.0 / dt)))
            spec.last_action_observation_start, spec.last_action_scale = 54, 0.2
            spec.time_remaining_index, spec.time_remaining_decrement = 14, 1.0 / max_steps
            valid = finite[:-1] & finite[1:]
            expected_action = 0.2 * episode["action"][valid].clamp(-1.0, 1.0)
            expected_time = (episode["observation"][:-1, 14][valid] - spec.time_remaining_decrement).clamp(0.0, 1.0)
            if not torch.allclose(episode["observation"][1:, 54:62][valid], expected_action, atol=2e-6, rtol=0):
                raise ValueError("Franka recorded last-action observations violate their declared scaling")
            if not torch.allclose(episode["observation"][1:, 14][valid], expected_time, atol=2e-6, rtol=0):
                raise ValueError("Franka recorded remaining-time observations violate their declared counter")
        if spec.task_id in specs and spec != specs[spec.task_id]:
            raise ValueError(f"Incompatible state/action dimensions for task {spec.task_id}")
        for field in ("context_names", "observation_names", "action_names"):
            names = tuple(_metadata(episode, field, ()))
            key = (spec.task_id, field)
            if key in field_names and names != field_names[key]:
                raise ValueError(f"Inconsistent {field} ordering for task {spec.task_id}")
            field_names[key] = names
        specs[spec.task_id] = spec
    return [specs[key] for key in sorted(specs)]


@torch.no_grad()
def fit_normalizers(model: CoupledWorldModel, episodes: list[dict]) -> None:
    """Fit scale statistics exclusively on finite training-state transitions."""
    if any(episode["split"] != "train" for episode in episodes):
        raise ValueError("Normalizer fitting accepts training episodes only")
    for episode in episodes:
        heads = model.task(episode["task_id"])
        finite = episode["finite_mask"]
        for key, normalizer in (
            ("observation", heads.observation_norm),
            ("critic_observation", heads.critic_norm),
            ("controller_state", heads.controller_norm),
        ):
            normalizer.update(episode[key], finite)
        model.material_norm.update(model.material_features(episode["material_summary"]), finite)
        valid_transition = finite[:-1] & finite[1:] & torch.isfinite(episode["action"]).all(-1)
        heads.action_norm.update(episode["action"], valid_transition)
        heads.context_norm.update(episode["context"].unsqueeze(0))
        heads.reward_norm.update(episode["reward"].unsqueeze(-1), valid_transition)
        heads.observation_delta_norm.update(episode["observation"][1:] - episode["observation"][:-1], valid_transition)
        heads.critic_delta_norm.update(
            episode["critic_observation"][1:] - episode["critic_observation"][:-1], valid_transition
        )
        model.particle_encoder.normalizer.update(
            episode["particle_state"], finite[:, None] & episode["particle_mask"][None]
        )
        model.rigid_encoder.normalizer.update(episode["rigid_state"], finite[:, None] & episode["rigid_mask"][None])


class EpisodeWindows:
    """Sample padded, finite windows from a single split and task.

    Valid terminal transitions are retained; later transitions are masked. A
    recording-budget cut is not converted into a physical terminal label.
    """

    def __init__(self, episodes: list[dict], *, task_id: str, split: str, horizon: int, seed: int = 0):
        if horizon < 1:
            raise ValueError("Window horizon must be positive")
        self.episodes = [episode for episode in episodes if episode["task_id"] == task_id and episode["split"] == split]
        self.horizon = horizon
        self.generator = torch.Generator().manual_seed(seed)
        self.starts = []
        for i, episode in enumerate(self.episodes):
            finite = episode["finite_mask"]
            valid = finite[:-1] & finite[1:]
            valid &= torch.isfinite(episode["action"]).all(-1) & torch.isfinite(episode["reward"])
            self.starts.extend((i, int(start)) for start in valid.nonzero().flatten())
        if not self.starts:
            raise ValueError(f"No finite {split} transitions for task {task_id}")

    def sample(self, batch_size: int, *, device: str = "cpu") -> dict[str, torch.Tensor]:
        """Draw real seed states and logged action sequences with terminal masks."""
        indices = torch.randint(len(self.starts), (batch_size,), generator=self.generator)
        samples = [self._window(*self.starts[int(index)]) for index in indices]
        output = {}
        for key in samples[0]:
            if key in ("particle_state", "rigid_state", "particle_mask", "rigid_mask", "particle_weights"):
                axis = 1 if key.endswith("state") else 0
                count = max(sample[key].shape[axis] for sample in samples)
                padded = []
                for sample in samples:
                    value = sample[key]
                    shape = list(value.shape)
                    shape[axis] = count - shape[axis]
                    padded.append(torch.cat((value, value.new_zeros(shape)), dim=axis))
                output[key] = torch.stack(padded).to(device)
            else:
                output[key] = torch.stack([sample[key] for sample in samples]).to(device)
        return output

    def _window(self, episode_index, start):
        episode = self.episodes[episode_index]
        count = len(episode["action"])
        state_indices = torch.arange(start, start + self.horizon + 1).clamp_max(count)
        transition_indices = torch.arange(start, start + self.horizon).clamp_max(count - 1)
        finite = episode["finite_mask"][state_indices]
        mask = finite[:-1] & finite[1:] & (torch.arange(self.horizon) < count - start)
        done = (episode["terminated"] | episode["truncated"])[transition_indices]
        valid_transition = torch.isfinite(episode["action"][transition_indices]).all(-1)
        valid_transition &= torch.isfinite(episode["reward"][transition_indices])
        mask &= valid_transition
        continuation = mask & ~done
        previous_alive = torch.cat((torch.ones(1, dtype=torch.bool), continuation[:-1])).cumprod(0).bool()
        mask &= previous_alive
        output = {
            key: torch.nan_to_num(episode[key][state_indices], nan=0.0, posinf=0.0, neginf=0.0)
            for key in ("observation", "critic_observation")
        }
        # Only seed geometry enters this compact latent model. Avoid transferring
        # future particle snapshots that the learner cannot use.
        output.update(
            {
                key: torch.nan_to_num(episode[key][start : start + 1], nan=0.0, posinf=0.0, neginf=0.0)
                for key in ("particle_state", "rigid_state", "controller_state", "material_summary")
            }
        )
        output.update(
            {
                key: torch.nan_to_num(episode[key][transition_indices], nan=0.0, posinf=0.0, neginf=0.0)
                for key in ("action", "reward", "terminated")
            }
        )
        output.update({key: episode[key] for key in ("context", "particle_mask", "rigid_mask", "particle_weights")})
        output["transition_mask"] = mask.float()
        return output
