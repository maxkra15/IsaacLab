# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for selecting source actions during Mimic annotation."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp.recorders.recorders import PreStepActionsRecorder
from isaaclab.envs.mimic_env_cfg import SubTaskConfig
from isaaclab.managers import RecorderTermCfg
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

from isaaclab_mimic.episode_replay import (
    get_required_subtask_term_signal_names,
    has_manual_subtask_annotations,
    resolve_episode_replay_actions,
    resolve_episode_subtask_term_signals,
)


def _make_episode() -> EpisodeData:
    episode = EpisodeData()
    episode.data = {
        "actions": torch.zeros((4, 2)),
        "processed_actions": torch.arange(12, dtype=torch.float32).reshape(4, 3),
    }
    return episode


def test_resolve_episode_replay_actions_defaults_to_canonical_actions():
    """Raw actions remain the backward-compatible replay source."""
    episode = _make_episode()

    resolved = resolve_episode_replay_actions(episode, expected_action_dim=2)

    assert resolved is episode.data["actions"]


def test_resolve_episode_replay_actions_selects_processed_actions():
    """An alternate recorded action stream can be selected without copying it."""
    episode = _make_episode()

    resolved = resolve_episode_replay_actions(
        episode,
        action_key="processed_actions",
        expected_action_dim=3,
    )

    assert resolved is episode.data["processed_actions"]


@pytest.mark.parametrize(
    ("action_key", "expected_exception", "message"),
    [
        ("missing", KeyError, "configured replay action key 'missing'"),
        ("processed_actions", ValueError, "environment expects 2"),
    ],
)
def test_resolve_episode_replay_actions_reports_invalid_selection(action_key, expected_exception, message):
    """Invalid replay keys and environment action dimensions fail clearly."""
    with pytest.raises(expected_exception, match=message):
        resolve_episode_replay_actions(_make_episode(), action_key=action_key, expected_action_dim=2)


def test_resolve_episode_replay_actions_rejects_non_sequence_data():
    """Replay fields must be tensors with a time and action dimension."""
    episode = _make_episode()
    episode.data["processed_actions"] = torch.zeros(3)

    with pytest.raises(ValueError, match=r"shape \(T, action_dim\)"):
        resolve_episode_replay_actions(episode, action_key="processed_actions")


def test_pre_step_recorder_canonicalizes_replay_input_as_actions():
    """The selected env.step input is exported under Mimic's canonical action key."""
    replay_actions = torch.arange(3, dtype=torch.float32).reshape(1, 3)
    env = SimpleNamespace(action_manager=SimpleNamespace(action=replay_actions))

    key, value = PreStepActionsRecorder(RecorderTermCfg(), env).record_pre_step()

    assert key == "actions"
    assert value is replay_actions


def test_hdf5_processed_actions_become_canonical_actions(tmp_path):
    """A 16D/20D source file round-trips as a canonical 20D annotated file."""
    raw_actions = torch.arange(4 * 16, dtype=torch.float32).reshape(4, 16)
    processed_actions = torch.arange(4 * 20, dtype=torch.float32).reshape(4, 20) + 100.0
    source_episode = EpisodeData()
    source_episode.data = {
        "actions": raw_actions,
        "processed_actions": processed_actions,
    }

    source_path = str(tmp_path / "source.hdf5")
    source_handler = HDF5DatasetFileHandler()
    source_handler.create(source_path, "test-processed-action-task")
    source_handler.write_episode(source_episode)
    source_handler.close()

    source_handler.open(source_path)
    loaded_source = source_handler.load_episode("demo_0", device="cpu")
    source_handler.close()
    replay_actions = resolve_episode_replay_actions(
        loaded_source,
        action_key="processed_actions",
        expected_action_dim=20,
    )

    annotated_episode = EpisodeData()
    env = SimpleNamespace(action_manager=SimpleNamespace(action=None))
    recorder = PreStepActionsRecorder(RecorderTermCfg(), env)
    for action in replay_actions:
        env.action_manager.action = action.unsqueeze(0)
        key, value = recorder.record_pre_step()
        annotated_episode.add(key, value[0])
    annotated_episode.pre_export()

    output_path = str(tmp_path / "annotated.hdf5")
    output_handler = HDF5DatasetFileHandler()
    output_handler.create(output_path, "test-processed-action-task")
    output_handler.write_episode(annotated_episode)
    output_handler.close()

    output_handler.open(output_path)
    loaded_output = output_handler.load_episode("demo_0", device="cpu")
    output_handler.close()

    assert loaded_output.data.keys() == {"actions"}
    torch.testing.assert_close(loaded_output.data["actions"], processed_actions)


def test_manual_annotation_detection_allows_signal_free_final_subtasks():
    """One final subtask per end effector requires no manual boundary marks."""
    term_signal_names = {"right": [], "left": []}
    start_signal_names = {"right": [], "left": []}

    assert not has_manual_subtask_annotations(term_signal_names, start_signal_names)

    term_signal_names["right"].append("grasp")
    assert has_manual_subtask_annotations(term_signal_names, start_signal_names)


def test_required_subtask_signals_exclude_each_end_effectors_final_subtask():
    """Only non-final subtask boundaries require recorded termination signals."""
    subtask_configs = {
        "right": [
            SubTaskConfig(subtask_term_signal="grasp"),
            SubTaskConfig(subtask_term_signal=None),
        ],
        "left": [SubTaskConfig(subtask_term_signal=None)],
    }

    assert get_required_subtask_term_signal_names(subtask_configs) == {"grasp"}


def test_resolve_episode_subtask_signals_allows_omitted_empty_group():
    """The recorder may omit an empty signal dictionary for final-only tasks."""
    episode = EpisodeData()
    episode.data = {"obs": {"datagen_info": {"eef_pose": {}}}}

    assert resolve_episode_subtask_term_signals(episode) == {}


def test_resolve_episode_subtask_signals_reports_missing_required_signal():
    """Missing boundaries for configured non-final subtasks fail clearly."""
    episode = EpisodeData()
    episode.data = {"obs": {"datagen_info": {"subtask_term_signals": {}}}}

    with pytest.raises(ValueError, match=r"required Mimic subtask termination signal\(s\): grasp"):
        resolve_episode_subtask_term_signals(episode, required_signal_names={"grasp"})


def test_annotation_script_has_no_hardcoded_episode_action_source():
    """Every annotation path obtains its sequence through the configured resolver."""
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts/imitation_learning/isaaclab_mimic/annotate_demos.py"
    tree = ast.parse(script_path.read_text(), filename=str(script_path))

    hardcoded_action_subscripts = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "episode"
        and node.value.attr == "data"
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "actions"
    ]

    assert not hardcoded_action_subscripts

    function_sources = {
        node.name: ast.unparse(node) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_get_episode_replay_actions(env, episode)" in function_sources["replay_episode"]
    assert function_sources["annotate_episode_in_manual_mode"].count("_get_episode_replay_actions(env, episode)") == 2
    assert "has_manual_subtask_annotations" in function_sources["annotate_episode_in_manual_mode"]
    assert function_sources["annotate_episode_in_manual_mode"].count("replay_episode(env, episode, success_term)") == 2
    assert "resolve_episode_subtask_term_signals" in function_sources["annotate_episode_in_auto_mode"]
