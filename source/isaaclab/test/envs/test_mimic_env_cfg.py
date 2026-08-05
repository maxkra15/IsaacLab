# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the base Mimic environment configuration."""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg


def test_annotation_replay_action_key_defaults_to_actions():
    """The existing raw-action annotation behavior remains the default."""
    assert MimicEnvCfg().annotation_replay_action_key == "actions"


def test_annotation_replay_action_key_accepts_processed_actions():
    """Tasks can select a recorded controller-space action sequence."""
    cfg = MimicEnvCfg(annotation_replay_action_key="processed_actions")

    assert cfg.annotation_replay_action_key == "processed_actions"


def test_annotation_sim_buffer_reset_defaults_to_enabled():
    """Existing annotation tasks retain the historical hard-reset behavior."""
    assert MimicEnvCfg().annotation_reset_sim_buffer_each_episode


def test_annotation_sim_buffer_reset_can_be_disabled():
    """Tasks can preserve backend solver buffers across annotation episodes."""
    cfg = MimicEnvCfg(annotation_reset_sim_buffer_each_episode=False)

    assert not cfg.annotation_reset_sim_buffer_each_episode
