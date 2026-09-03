# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab extensions to the RSL-RL on-policy runner."""

from __future__ import annotations

import warnings
from typing import Any

from rsl_rl.runners import OnPolicyRunner

_ISAACLAB_ENV_STATE_KEY = "isaaclab_environment_state"
_ISAACLAB_ENV_STATE_VERSION = 1


class IsaacLabOnPolicyRunner(OnPolicyRunner):
    """RSL-RL runner that persists environment state exposed by its wrapper.

    Single-process checkpoints restore the opted-in environment state exactly.
    In distributed training RSL-RL writes checkpoints only from the primary
    rank. That rank's environment state is therefore authoritative, and every
    rank restores the same snapshot when loading. No distributed collective is
    used here, so non-primary ranks cannot deadlock while the primary saves.

    Checkpoints without Isaac Lab environment state retain upstream RSL-RL
    behavior and remain loadable.
    """

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save policy state and any environment state that explicitly opts in."""
        get_state = getattr(self.env, "get_checkpoint_state", None)
        environment_state = get_state() if callable(get_state) else {}
        if not environment_state:
            super().save(path, infos)
            return

        checkpoint_infos = dict(infos) if infos is not None else {}
        if _ISAACLAB_ENV_STATE_KEY in checkpoint_infos:
            raise ValueError(f"Checkpoint infos key '{_ISAACLAB_ENV_STATE_KEY}' is reserved by Isaac Lab.")
        checkpoint_infos[_ISAACLAB_ENV_STATE_KEY] = {
            "version": _ISAACLAB_ENV_STATE_VERSION,
            "source_world_size": self.gpu_world_size,
            "source_global_rank": self.gpu_global_rank,
            "source_num_envs": self.env.num_envs,
            "user_infos_was_none": infos is None,
            "state": environment_state,
        }
        super().save(path, checkpoint_infos)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict | None:
        """Load policy state and optionally restore the authoritative environment snapshot.

        Full resumes (``load_cfg is None``) restore environment state. Selective
        policy warm starts skip it unless ``load_cfg["environment_state"]`` is
        explicitly true.
        """
        if map_location is None:
            map_location = self.device
        infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
        if not isinstance(infos, dict) or _ISAACLAB_ENV_STATE_KEY not in infos:
            return infos

        checkpoint_infos = dict(infos)
        payload: Any = checkpoint_infos.pop(_ISAACLAB_ENV_STATE_KEY)
        if not isinstance(payload, dict) or payload.get("version") != _ISAACLAB_ENV_STATE_VERSION:
            raise ValueError("Unsupported Isaac Lab environment checkpoint payload.")
        required_keys = {
            "version",
            "source_world_size",
            "source_global_rank",
            "source_num_envs",
            "user_infos_was_none",
            "state",
        }
        if set(payload) != required_keys:
            raise ValueError("Isaac Lab environment checkpoint payload has incompatible fields.")

        for name in ("source_world_size", "source_global_rank", "source_num_envs"):
            value = payload[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Isaac Lab environment checkpoint field '{name}' must be a non-negative integer.")
        if (
            payload["source_world_size"] < 1
            or payload["source_global_rank"] >= payload["source_world_size"]
            or payload["source_num_envs"] < 1
        ):
            raise ValueError("Isaac Lab environment checkpoint has invalid distributed-rank metadata.")
        if not isinstance(payload["user_infos_was_none"], bool) or not isinstance(payload["state"], dict):
            raise ValueError("Isaac Lab environment checkpoint has invalid state metadata.")

        restore_environment_state = load_cfg is None or load_cfg.get("environment_state") is True
        source_world_size = payload["source_world_size"]
        if restore_environment_state and source_world_size != self.gpu_world_size:
            warnings.warn(
                "Restoring authoritative environment state from an RSL-RL checkpoint written with "
                f"world size {source_world_size}; the current world size is {self.gpu_world_size}.",
                RuntimeWarning,
                stacklevel=2,
            )
        if restore_environment_state and payload["source_num_envs"] != self.env.num_envs:
            warnings.warn(
                "Restoring environment state from an RSL-RL checkpoint written with "
                f"{payload['source_num_envs']} environments per rank; the current runner has {self.env.num_envs}. "
                "The state provider will validate whether its partial windows are compatible.",
                RuntimeWarning,
                stacklevel=2,
            )
        if restore_environment_state:
            set_state = getattr(self.env, "set_checkpoint_state", None)
            if not callable(set_state):
                raise RuntimeError(
                    "The checkpoint contains environment state, but the RSL-RL environment cannot restore it."
                )
            set_state(
                payload["state"],
                source_global_rank=payload["source_global_rank"],
                current_global_rank=self.gpu_global_rank,
            )

        if payload["user_infos_was_none"] and not checkpoint_infos:
            return None
        return checkpoint_infos
