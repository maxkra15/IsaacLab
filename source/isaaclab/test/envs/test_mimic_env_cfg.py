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
