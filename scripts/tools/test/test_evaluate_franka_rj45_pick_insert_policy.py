# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for deterministic Franka RJ45 pick-and-insert evaluation."""

import pytest
import torch

from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_env_cfg import PICK_INSERT_PHASE_NAMES
from isaaclab_tasks.contrib.franka_rj45_insertion.rj45_env import TERMINAL_OUTCOME_NAMES

from scripts.tools.evaluate_franka_rj45_pick_insert_policy import (
    _new_phase_counts,
    _record_completed_episodes,
    _summarize_counts,
)


def _terminal_batch() -> dict[str, object]:
    outcomes = {name: torch.zeros(6, dtype=torch.bool) for name in TERMINAL_OUTCOME_NAMES}
    outcomes["success"][0] = True
    outcomes["lost_grasp"][[1, 4]] = True
    outcomes["task_out_of_bounds"][3] = True
    outcomes["time_out"][[2, 5]] = True
    return {
        "phase_ids": torch.arange(6, dtype=torch.long),
        "episode_returns": torch.arange(1, 7, dtype=torch.float32),
        "episode_lengths": torch.arange(1, 7, dtype=torch.long),
        "terminal_outcomes": outcomes,
        "starts_grasped": torch.tensor([True, True, True, True, False, False]),
        "initial_stages": torch.tensor([3, 2, 1, 1, 0, 0]),
        "maximum_stages": torch.tensor([4, 2, 3, 1, 1, 0]),
        "ever_grasped": torch.tensor([True, True, True, True, True, False]),
        "learning_progress": torch.tensor([True, False, True, False, True, False]),
    }


def test_phase_balanced_recording_caps_each_phase_and_reports_pick_progress() -> None:
    counts = _new_phase_counts()
    batch = _terminal_batch()

    assert _record_completed_episodes(counts, **batch, episodes_per_phase=1) == 6
    assert _record_completed_episodes(counts, **batch, episodes_per_phase=1) == 0

    per_phase, overall = _summarize_counts(counts, PICK_INSERT_PHASE_NAMES)
    assert [row["episodes"] for row in per_phase] == [1] * 6
    assert overall["episodes"] == 6
    assert overall["success_rate"] == pytest.approx(1.0 / 6.0)
    assert overall["lost_grasp_rate"] == pytest.approx(2.0 / 6.0)
    assert overall["task_out_of_bounds_rate"] == pytest.approx(1.0 / 6.0)
    assert overall["time_out_rate"] == pytest.approx(2.0 / 6.0)
    assert overall["mean_return"] == pytest.approx(3.5)
    assert overall["mean_episode_length"] == pytest.approx(3.5)
    assert overall["ever_grasped_rate"] == pytest.approx(5.0 / 6.0)
    assert overall["grasp_acquisition_eligible_episodes"] == 2
    assert overall["grasp_acquisition_rate"] == pytest.approx(0.5)
    assert overall["stage_advance_rate"] == pytest.approx(0.5)
    assert overall["learning_progress_rate"] == pytest.approx(0.5)
    assert overall["maximum_stage_histogram"] == {"0": 1, "1": 2, "2": 1, "3": 1, "4": 1}
    assert per_phase[0]["grasp_acquisition_rate"] is None
    assert per_phase[4]["grasp_acquisition_rate"] == 1.0
    assert per_phase[5]["grasp_acquisition_rate"] == 0.0


def test_recording_rejects_terminal_episode_without_a_task_cause() -> None:
    batch = _terminal_batch()
    batch["terminal_outcomes"] = {name: torch.zeros(6, dtype=torch.bool) for name in TERMINAL_OUTCOME_NAMES}

    with pytest.raises(RuntimeError, match="no recognized terminal cause"):
        _record_completed_episodes(_new_phase_counts(), **batch, episodes_per_phase=20)


def test_recording_rejects_nonmonotonic_stage_snapshot() -> None:
    batch = _terminal_batch()
    batch["maximum_stages"] = torch.tensor([2, 2, 3, 1, 1, 0])

    with pytest.raises(RuntimeError, match="invalid stage progression 3->2"):
        _record_completed_episodes(_new_phase_counts(), **batch, episodes_per_phase=20)
