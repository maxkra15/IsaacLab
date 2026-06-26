# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for waterhose teleop command safety limits."""

import importlib

import pytest
import torch

from isaaclab_tasks.contrib.waterhose.waterhose_env_cfg import WaterhoseNewtonRelativeIkActionsCfg


def test_waterhose_relative_ik_action_clips_teleop_deltas():
    """Teleop relative IK bounds per-step EE targets so contacts are not overrun."""

    actions_cfg = WaterhoseNewtonRelativeIkActionsCfg()

    assert actions_cfg.arm_action.clip == {
        "right_ee/x": (-0.01, 0.01),
        "right_ee/y": (-0.01, 0.01),
        "right_ee/z": (-0.01, 0.01),
        "right_ee/roll": (-0.08, 0.08),
        "right_ee/pitch": (-0.08, 0.08),
        "right_ee/yaw": (-0.08, 0.08),
    }
    assert actions_cfg.gripper_action.max_joint_delta_per_step == pytest.approx(0.0015)


def test_rate_limit_joint_targets_clamps_each_joint_delta():
    """Joint target limiting moves each gripper joint by at most the configured step."""

    actions_module = importlib.import_module("isaaclab_tasks.contrib.waterhose.mdp.actions")
    assert hasattr(actions_module, "_rate_limit_joint_targets")

    previous = torch.tensor([[0.0, 1.0], [1.0, -1.0]])
    desired = torch.tensor([[0.01, 0.0], [0.999, -0.997]])

    limited = actions_module._rate_limit_joint_targets(previous, desired, 0.002)

    expected = torch.tensor([[0.002, 0.998], [0.999, -0.998]])
    assert torch.allclose(limited, expected)
