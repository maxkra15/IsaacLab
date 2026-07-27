# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for loading annotated episodes into the Mimic datagen info pool."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mimic_env_cfg import SubTaskConfig
from isaaclab.utils.datasets import EpisodeData

from isaaclab_mimic.datagen.datagen_info_pool import DataGenInfoPool


def _make_env():
    return SimpleNamespace(actions_to_gripper_actions=lambda actions: {"right": actions[:, -1:]})


def _make_env_cfg(subtask_configs):
    return SimpleNamespace(
        subtask_configs={"right": subtask_configs},
        datagen_config=SimpleNamespace(use_skillgen=False),
    )


def _make_episode(include_empty_subtask_signals: bool = False) -> EpisodeData:
    episode = EpisodeData()
    datagen_info = {
        "eef_pose": {"right": torch.eye(4).repeat(3, 1, 1)},
        "object_pose": {},
        "target_eef_pose": {"right": torch.eye(4).repeat(3, 1, 1)},
    }
    if include_empty_subtask_signals:
        datagen_info["subtask_term_signals"] = {}
    episode.data = {
        "actions": torch.zeros(3, 2),
        "obs": {"datagen_info": datagen_info},
    }
    return episode


@pytest.mark.parametrize("include_empty_subtask_signals", [False, True])
def test_final_only_subtask_accepts_absent_or_empty_term_signals(include_empty_subtask_signals):
    """A final-only episode spans the full action sequence without a boundary signal."""
    pool = DataGenInfoPool(
        env=_make_env(),
        env_cfg=_make_env_cfg([SubTaskConfig(subtask_term_signal=None)]),
        device="cpu",
    )

    pool._add_episode(_make_episode(include_empty_subtask_signals))

    assert pool.num_datagen_infos == 1
    assert pool.datagen_infos[0].subtask_term_signals == {}
    assert pool.subtask_boundaries == {"right": [[(0, 3)]]}


def test_non_final_subtask_reports_missing_required_term_signal():
    """An omitted boundary for a configured non-final subtask raises a clear error."""
    pool = DataGenInfoPool(
        env=_make_env(),
        env_cfg=_make_env_cfg(
            [
                SubTaskConfig(subtask_term_signal="grasp"),
                SubTaskConfig(subtask_term_signal=None),
            ]
        ),
        device="cpu",
    )

    with pytest.raises(ValueError, match=r"required Mimic subtask termination signal\(s\): grasp"):
        pool._add_episode(_make_episode())
