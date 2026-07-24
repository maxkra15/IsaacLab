# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim-free contract tests for the bimanual waterhose XR pipeline."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("isaacteleop")

from isaaclab_teleop import TELEOP_CONTROL_CHANNEL_UUID
from isaacteleop.retargeting_engine.deviceio_source_nodes import HandsSource

from isaaclab_tasks.contrib.waterhose.teleop_pipelines import build_waterhose_bimanual_teleop_pipeline
from isaaclab_tasks.contrib.waterhose.waterhose_env_cfg import WaterhoseProxyIkEnvCfg, WaterhoseProxyTeleopEnvCfg


def test_waterhose_bimanual_pipeline_matches_the_16d_environment_action():
    """Two wrist poses and two grippers must flatten to the environment's 16D action."""

    pipeline, retargeters = build_waterhose_bimanual_teleop_pipeline()

    assert pipeline.output_types()["action"].types[0].shape == (16,)
    reorderer = pipeline.output_mapping["action"].module._target_module
    assert reorderer._output_order == [
        "r_pos_x",
        "r_pos_y",
        "r_pos_z",
        "r_quat_x",
        "r_quat_y",
        "r_quat_z",
        "r_quat_w",
        "l_pos_x",
        "l_pos_y",
        "l_pos_z",
        "l_quat_x",
        "l_quat_y",
        "l_quat_z",
        "l_quat_w",
        "right_gripper",
        "left_gripper",
    ]
    assert [retargeter._config.input_device for retargeter in retargeters] == [HandsSource.RIGHT, HandsSource.LEFT]


def test_waterhose_action_configs_pack_arm_before_grippers():
    """Scripted and bimanual commands must reach the matching action terms."""

    assert list(WaterhoseProxyIkEnvCfg().actions.__dict__) == ["arm_action", "gripper_action"]
    assert list(WaterhoseProxyTeleopEnvCfg().actions.__dict__) == [
        "arm_action",
        "gripper_action",
        "left_gripper_action",
    ]


def test_waterhose_teleop_config_exposes_the_pipeline_not_the_tuning_tuple():
    """The session lifecycle consumes an OutputCombiner directly from ``pipeline_builder``."""

    pipeline = WaterhoseProxyTeleopEnvCfg().isaac_teleop.pipeline_builder()

    assert pipeline.output_types()["action"].types[0].shape == (16,)


def test_waterhose_teleop_uses_standard_episode_controls():
    """AVP Play, Stop, and Reset must use IsaacTeleop's standard control channel."""

    teleop_cfg = WaterhoseProxyTeleopEnvCfg().isaac_teleop

    assert not teleop_cfg.teleoperation_active_default
    assert teleop_cfg.control_channel_uuid == TELEOP_CONTROL_CHANNEL_UUID


def test_waterhose_bimanual_pipeline_tracks_both_wrist_orientations_without_axis_suppression():
    """Both wrist retargeters must preserve every tracked rotation axis."""

    _, retargeters = build_waterhose_bimanual_teleop_pipeline()

    assert len(retargeters) == 2
    for retargeter in retargeters:
        assert retargeter._config.use_wrist_rotation
        assert retargeter._config.use_wrist_position
        assert not retargeter._config.zero_out_xy_rotation


def test_waterhose_tool_offsets_preserve_wrist_rotation_one_to_one():
    """A fixed tool offset must not change the wrist's three-axis rotation delta."""

    _, retargeters = build_waterhose_bimanual_teleop_pipeline()
    wrist_before = Rotation.from_euler("XYZ", [17.0, -23.0, 11.0], degrees=True)
    wrist_after = Rotation.from_euler("XYZ", [31.0, 5.0, -19.0], degrees=True)
    wrist_delta = wrist_after * wrist_before.inv()

    for retargeter in retargeters:
        tool_offset = Rotation.from_quat(retargeter._target_offset_rot)
        tool_before = wrist_before * tool_offset
        tool_after = wrist_after * tool_offset
        tool_delta = tool_after * tool_before.inv()
        assert tool_delta.as_rotvec() == pytest.approx(wrist_delta.as_rotvec(), abs=1.0e-7)


def test_waterhose_retargeter_emits_invalid_sentinel_during_tracking_loss():
    """A lost hand must re-clutch instead of replaying a stale absolute pose."""

    _, retargeters = build_waterhose_bimanual_teleop_pipeline()
    stale_pose = np.array([0.4, -0.2, 0.9, 0.1, 0.2, -0.3, 0.9], dtype=np.float32)
    expected_sentinel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    for retargeter in retargeters:
        retargeter._last_pose = stale_pose.copy()
        output = np.empty((1, 7), dtype=np.float32)
        inputs = {retargeter._config.input_device: SimpleNamespace(is_none=True)}
        context = SimpleNamespace(execution_events=SimpleNamespace(reset=False))

        retargeter._compute_fn(inputs, {"ee_pose": output}, context)

        np.testing.assert_array_equal(output[0], expected_sentinel)
