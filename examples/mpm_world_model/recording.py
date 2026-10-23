# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Atomic episode shards shared by material world-model collectors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _owned(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, dict):
        return {key: _owned(item) for key, item in value.items()}
    return value


def atomic_json(path: Path, value: dict) -> None:
    """Replace a JSON manifest only after its complete contents are written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


class EpisodeWriter:
    """Own bounded per-world trajectories and export each episode independently.

    State tensors have no time dimension on input. A saved episode has T+1
    states and T transitions. Recording cuts never invent a task termination.
    """

    def __init__(self, output_dir: str | Path, metadata: dict | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "manifest.json"
        if self.manifest_path.exists():
            raise FileExistsError(f"Refusing to overwrite dataset: {self.manifest_path}")
        self.manifest = {"schema_version": 1, "metadata": metadata or {}, "episodes": [], "complete": False}
        self._active: dict[int, dict] = {}
        self._next_id = 0
        atomic_json(self.manifest_path, self.manifest)

    def start_episode(self, world_id: int, state: dict[str, torch.Tensor], episode_metadata: dict) -> None:
        """Record an initial state after reset, keeping static context separate."""
        if world_id in self._active:
            raise RuntimeError(f"World {world_id} already has an active episode.")
        if "observation" not in state or "particle_state" not in state:
            raise ValueError("An episode requires observation and particle_state.")
        self._active[world_id] = {
            "metadata": {**episode_metadata, "world_id": int(world_id), "episode_id": self._next_id},
            "states": {key: [_owned(value)] for key, value in state.items()},
            "transitions": {key: [] for key in ("action", "reward", "terminated", "truncated")},
            "extras": {},
        }
        self._next_id += 1

    def append_transition(
        self,
        world_id: int,
        action: torch.Tensor,
        reward: torch.Tensor | float,
        terminated: torch.Tensor | bool,
        truncated: torch.Tensor | bool,
        next_state: dict[str, torch.Tensor],
        extra: dict | None = None,
    ) -> None:
        """Append the pre-reset physical endpoint of one action."""
        episode = self._active[world_id]
        if next_state.keys() != episode["states"].keys():
            raise ValueError("State fields changed within an episode.")
        transitions = episode["transitions"]
        if transitions["terminated"] and bool(transitions["terminated"][-1] | transitions["truncated"][-1]):
            raise ValueError("A terminal episode must be closed before another transition.")
        for key, value in (
            ("action", action),
            ("reward", reward),
            ("terminated", terminated),
            ("truncated", truncated),
        ):
            transitions[key].append(_owned(torch.as_tensor(value)))
        for key, value in next_state.items():
            if value.shape != episode["states"][key][0].shape:
                raise ValueError(f"State shape changed within an episode: {key}")
            episode["states"][key].append(_owned(value))
        for key, value in (extra or {}).items():
            episode["extras"].setdefault(key, []).append(_owned(torch.as_tensor(value)))

    def finish_episode(self, world_id: int, recording_cut: bool = False) -> Path | None:
        """Close one episode; its reset state belongs to a new shard."""
        episode = self._active.pop(world_id, None)
        if episode is None or not episode["transitions"]["action"]:
            return None
        payload = {key: torch.stack(value) for key, value in episode["states"].items()}
        payload.update({key: torch.stack(value) for key, value in episode["transitions"].items()})
        length = len(payload["action"])
        if any(len(value) != length for value in episode["extras"].values()):
            raise ValueError("Transition extras must be present at every step of an episode.")
        payload["transition_info"] = {key: torch.stack(value) for key, value in episode["extras"].items()}
        metadata = episode["metadata"]
        payload.update(metadata)
        payload["recording_cut"] = bool(recording_cut)
        payload["schema_version"] = 1
        group = str(metadata["reset_group"])
        bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10
        split = "train" if bucket < 8 else ("validation" if bucket == 8 else "test")
        payload["split"] = split
        filename = f"episode-{metadata['episode_id']:06d}-world-{world_id:04d}.pt"
        path = self.output_dir / filename
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        entry = {
            "file": filename,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "transitions": length,
            "task_id": metadata["task_id"],
            "reset_group": group,
            "episode_id": metadata["episode_id"],
            "world_id": world_id,
            "split": split,
            "recording_cut": bool(recording_cut),
        }
        self.manifest["episodes"].append(entry)
        atomic_json(self.manifest_path, self.manifest)
        return path

    def close(self, complete: bool = True) -> None:
        """Flush partial trajectories and mark whether collection ended normally."""
        for world_id in list(self._active):
            self.finish_episode(world_id, recording_cut=True)
        self.manifest["complete"] = bool(complete)
        self.manifest["transition_count"] = sum(item["transitions"] for item in self.manifest["episodes"])
        atomic_json(self.manifest_path, self.manifest)


class EpisodeMetrics:
    """Keep pre-reset lengths/returns, including censored initial episodes."""

    def __init__(self, num_worlds: int, dt: float):
        self.dt = dt
        self.returns = torch.zeros(num_worlds, dtype=torch.float64)
        self.lengths = torch.zeros(num_worlds, dtype=torch.long)
        self.rows = torch.full((num_worlds,), -1, dtype=torch.long)
        self.first = [
            {
                "world_id": i,
                "reset_row": -1,
                "outcome": "unfinished",
                "episode_length_policy_steps": 0,
                "episode_return": 0.0,
            }
            for i in range(num_worlds)
        ]
        self.completed = []

    def reset(self, world_ids, rows):
        ids = torch.as_tensor(world_ids, dtype=torch.long).cpu()
        self.returns[ids] = 0.0
        self.lengths[ids] = 0
        self.rows[ids] = torch.as_tensor(rows, dtype=torch.long).cpu()
        for index in ids.tolist():
            if self.first[index]["reset_row"] < 0:
                self.first[index]["reset_row"] = int(self.rows[index])

    def update(self, rewards, lengths, rows, terminated, truncated, success, terms, term_names, *, warmup=False):
        rewards, lengths, rows = rewards.detach().cpu(), lengths.detach().cpu(), rows.detach().cpu()
        terminated, truncated, success = terminated.cpu().bool(), truncated.cpu().bool(), success.cpu().bool()
        terms = terms.cpu().bool()
        self.returns += rewards.double()
        self.lengths.copy_(lengths)
        self.rows.copy_(rows)
        done = terminated | truncated
        for world in range(len(self.first)):
            record = {
                "world_id": world,
                "reset_row": int(rows[world]),
                "episode_length_policy_steps": int(lengths[world]),
                "episode_length_seconds": int(lengths[world]) * self.dt,
                "episode_return": float(self.returns[world]),
                "warmup": bool(warmup),
                "terminated": bool(terminated[world]),
                "truncated": bool(truncated[world]),
                "outcome": "success" if success[world] else ("timeout" if truncated[world] else "failure"),
                "termination_reasons": [name for name, active in zip(term_names, terms[world]) if active],
            }
            if self.first[world]["outcome"] == "unfinished":
                self.first[world] = {**record, "outcome": record["outcome"] if done[world] else "unfinished"}
            if done[world]:
                self.completed.append(record)

    def summary(self) -> dict:
        completed = [record for record in self.completed if not record["warmup"]]
        length_sum = sum(record["episode_length_policy_steps"] for record in completed)
        return_sum = sum(record["episode_return"] for record in completed)
        return {
            "completed_episode_length_sum_policy_steps": length_sum,
            "completed_episode_return_sum": return_sum,
            "mean_completed_episode_length_policy_steps": length_sum / max(len(completed), 1),
            "mean_completed_episode_return": return_sum / max(len(completed), 1),
            "mean_final_episode_length_policy_steps": float(self.lengths.float().mean()),
            "mean_final_episode_return": float(self.returns.mean()),
            "first_cohort_records": self.first,
            "completed_episode_records": self.completed,
        }
