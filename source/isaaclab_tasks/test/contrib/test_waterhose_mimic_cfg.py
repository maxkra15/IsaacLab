# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the waterhose Isaac Lab Mimic environment configuration."""

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.waterhose.waterhose_env_cfg import WaterhoseNewtonRelativeIkActionsCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def test_waterhose_mimic_task_cfg_matches_teleop_action_space():
    """The Mimic task should reuse the teleop action space recorded by teleop demos."""

    env_cfg = parse_env_cfg("Isaac-Waterhose-Coupled-Teleop-Mimic-v0", num_envs=1)

    assert isinstance(env_cfg, MimicEnvCfg)
    assert isinstance(env_cfg.actions, WaterhoseNewtonRelativeIkActionsCfg)
    assert set(env_cfg.subtask_configs) == {"right"}

    subtask_cfg = env_cfg.subtask_configs["right"][0]
    assert subtask_cfg.object_ref == "socket"
    assert subtask_cfg.subtask_term_signal is None

    recorder_cfg = env_cfg.make_recorder_manager_cfg()
    assert recorder_cfg.record_pre_step_datagen_info is not None
    assert recorder_cfg.record_pre_step_subtask_term_signals is not None


def test_waterhose_mimic_relative_pose_helpers_preserve_local_rotation_convention():
    """Round-trip helper math should preserve Waterhose's local-frame rotation deltas."""

    from isaaclab_tasks.contrib.waterhose.waterhose_mimic_env import (
        relative_action_to_target_pose,
        target_pose_to_relative_action,
    )

    device = "cpu"
    dtype = torch.float32
    current_pos = torch.tensor([[0.4, -0.2, 0.7]], device=device, dtype=dtype)
    current_quat = PoseUtils.quat_from_euler_xyz(
        torch.tensor([0.0], device=device, dtype=dtype),
        torch.tensor([0.0], device=device, dtype=dtype),
        torch.tensor([1.57079632679], device=device, dtype=dtype),
    )
    current_pose = PoseUtils.make_pose(current_pos, PoseUtils.matrix_from_quat(current_quat))

    action = torch.tensor([[0.01, -0.02, 0.03, 0.10, -0.05, 0.20, -1.0]], device=device, dtype=dtype)
    target_pose = relative_action_to_target_pose(current_pose, action[:, :6])

    current_rot = PoseUtils.matrix_from_quat(current_quat)
    delta_quat = PoseUtils.quat_from_angle_axis(
        torch.linalg.norm(action[:, 3:6], dim=-1),
        action[:, 3:6] / torch.linalg.norm(action[:, 3:6], dim=-1, keepdim=True),
    )
    expected_target_rot = torch.matmul(current_rot, PoseUtils.matrix_from_quat(delta_quat))

    assert torch.allclose(target_pose[:, :3, 3], current_pos + action[:, :3])
    assert torch.allclose(target_pose[:, :3, :3], expected_target_rot, atol=1e-6)

    round_trip_action = target_pose_to_relative_action(current_pose, target_pose, action[:, -1:])
    assert torch.allclose(round_trip_action, action, atol=1e-6)
