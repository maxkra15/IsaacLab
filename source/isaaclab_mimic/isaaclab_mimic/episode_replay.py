# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Utilities for selecting actions when replaying Mimic source episodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from isaaclab.envs.mimic_env_cfg import SubTaskConfig
from isaaclab.utils.datasets import EpisodeData


def resolve_episode_replay_actions(
    episode: EpisodeData,
    action_key: str = "actions",
    expected_action_dim: int | None = None,
) -> torch.Tensor:
    """Resolve and validate the action sequence used to replay an episode.

    Args:
        episode: Episode containing the recorded action sequences.
        action_key: Top-level episode-data key containing the replay actions.
        expected_action_dim: Expected action dimension of the replay
            environment. If provided, the selected sequence must match it.

    Returns:
        The selected action sequence with shape ``(T, action_dim)``.

    Raises:
        ValueError: If ``action_key`` is empty, the selected sequence is not
            two-dimensional, or its action dimension does not match
            ``expected_action_dim``.
        KeyError: If the episode does not contain ``action_key``.
        TypeError: If the selected value is not a tensor.
    """
    if not isinstance(action_key, str) or not action_key:
        raise ValueError("Mimic annotation replay action key must be a non-empty string.")

    if action_key not in episode.data:
        available_keys = ", ".join(sorted(episode.data)) or "<none>"
        raise KeyError(
            f"Episode does not contain the configured replay action key '{action_key}'. "
            f"Available top-level keys: {available_keys}."
        )

    actions = episode.data[action_key]
    if not isinstance(actions, torch.Tensor):
        raise TypeError(
            f"Episode replay action field '{action_key}' must be a torch.Tensor, received {type(actions).__name__}."
        )
    if actions.ndim != 2:
        raise ValueError(
            f"Episode replay action field '{action_key}' must have shape (T, action_dim), "
            f"received {tuple(actions.shape)}."
        )
    if expected_action_dim is not None and actions.shape[-1] != expected_action_dim:
        raise ValueError(
            f"Episode replay action field '{action_key}' has action dimension {actions.shape[-1]}, "
            f"but the environment expects {expected_action_dim}."
        )

    return actions


def has_manual_subtask_annotations(
    subtask_term_signal_names: Mapping[str, Sequence[str]],
    subtask_start_signal_names: Mapping[str, Sequence[str]],
) -> bool:
    """Return whether manual annotation has any subtask boundaries to mark.

    Args:
        subtask_term_signal_names: Termination signal names grouped by end effector.
        subtask_start_signal_names: Start signal names grouped by end effector.

    Returns:
        ``True`` when at least one signal needs manual annotation.
    """
    eef_names = subtask_term_signal_names.keys() | subtask_start_signal_names.keys()
    return any(subtask_term_signal_names.get(name) or subtask_start_signal_names.get(name) for name in eef_names)


def get_required_subtask_term_signal_names(
    subtask_configs: Mapping[str, Sequence[SubTaskConfig]],
) -> set[str]:
    """Return termination signals required to split configured Mimic subtasks.

    The final subtask for each end effector always ends with the episode, so its
    configured termination signal is not required.

    Args:
        subtask_configs: Subtask configurations grouped by end effector.

    Returns:
        Names of termination signals required by non-final subtasks.

    Raises:
        ValueError: If a non-final subtask does not define a signal name.
    """
    required_signal_names = set()
    for eef_name, configs in subtask_configs.items():
        for subtask_index, config in enumerate(configs[:-1]):
            signal_name = config.subtask_term_signal
            if not isinstance(signal_name, str) or not signal_name:
                raise ValueError(
                    f"Non-final Mimic subtask {subtask_index} for end effector {eef_name!r} "
                    "must configure a non-empty subtask termination signal."
                )
            required_signal_names.add(signal_name)
    return required_signal_names


def resolve_episode_subtask_term_signals(
    episode: EpisodeData,
    required_signal_names: Sequence[str] | set[str] = (),
) -> dict[str, torch.Tensor]:
    """Resolve optional subtask termination signals from an annotated episode.

    Episodes with one final subtask per end effector do not need intermediate
    termination signals. Such episodes may omit the nested
    ``subtask_term_signals`` group because empty recorder dictionaries are not
    serialized.

    Args:
        episode: Episode containing Mimic datagen annotations.
        required_signal_names: Signal names required by non-final subtasks.

    Returns:
        Recorded subtask termination signals, or an empty dictionary when none
        are recorded or required.

    Raises:
        TypeError: If the recorded subtask termination signals are not a dictionary.
        ValueError: If a required signal is absent from the episode.
    """
    observations = episode.data.get("obs", {})
    datagen_info = observations.get("datagen_info", {})
    subtask_term_signals = datagen_info.get("subtask_term_signals", {})
    if subtask_term_signals is None:
        subtask_term_signals = {}
    if not isinstance(subtask_term_signals, dict):
        raise TypeError(
            "Episode Mimic subtask termination signals must be a dictionary, "
            f"received {type(subtask_term_signals).__name__}."
        )

    missing_signal_names = sorted(set(required_signal_names) - subtask_term_signals.keys())
    if missing_signal_names:
        raise ValueError(
            f"Episode is missing required Mimic subtask termination signal(s): {', '.join(missing_signal_names)}."
        )
    return subtask_term_signals
