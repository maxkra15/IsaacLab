# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for waterhose teleop command safety limits."""

import importlib

import pytest
import torch
from isaaclab_newton.ik.newton_ik_objectives_cfg import NewtonIKJointPostureObjectiveCfg
from scipy.spatial.transform import Rotation

from isaaclab_tasks.contrib.waterhose import waterhose_env_cfg
from isaaclab_tasks.contrib.waterhose.waterhose_env_cfg import WaterhoseNewtonBimanualIkActionsCfg


def test_waterhose_bimanual_ik_drives_both_complete_arm_chains():
    """XR controls both wrist poses while Newton IK may use every joint in both arms."""

    actions_cfg = WaterhoseNewtonBimanualIkActionsCfg()
    arm_cfg = actions_cfg.arm_action

    assert arm_cfg.joint_names == ["torso_joint_.*", "left_arm_joint_.*", "right_arm_joint_.*"]
    assert [getattr(objective, "name", None) for objective in arm_cfg.objectives] == [
        "right_ee",
        "left_ee",
        "torso_hold",
        None,
        None,
    ]
    assert [objective.use_relative_mode for objective in arm_cfg.objectives[:3]] == [False, False, False]
    assert arm_cfg.class_type.endswith(":WaterhoseBimanualTeleopNewtonIkAction")


def test_waterhose_bimanual_teleop_commands_both_grippers_with_safe_close_endpoints():
    """Each hand gets one scalar and neither side commands a zero-gap closure."""

    actions_cfg = WaterhoseNewtonBimanualIkActionsCfg()
    assert actions_cfg.gripper_action.joint_names == [
        "right_gripper_finger_joint_1",
        "right_gripper_left_finger_joint",
        "right_gripper_right_finger_joint",
    ]
    assert actions_cfg.left_gripper_action.joint_names == [
        "left_gripper_finger_joint_1",
        "left_gripper_left_finger_joint",
        "left_gripper_right_finger_joint",
    ]
    assert tuple(actions_cfg.gripper_action.close_command_expr.values()) == pytest.approx((0.014, -0.007, 0.007))
    assert tuple(actions_cfg.left_gripper_action.close_command_expr.values()) == pytest.approx((0.014, -0.007, 0.007))
    assert actions_cfg.gripper_action.max_joint_delta_per_step == pytest.approx(0.15)
    assert actions_cfg.left_gripper_action.max_joint_delta_per_step == pytest.approx(0.15)


def test_waterhose_bimanual_ik_regularizes_shoulders_and_elbows_to_start_posture():
    """Redundant arm IK should prefer the task's natural bent-arm posture."""

    arm_cfg = WaterhoseNewtonBimanualIkActionsCfg().arm_action
    posture_objectives = [
        objective for objective in arm_cfg.objectives if isinstance(objective, NewtonIKJointPostureObjectiveCfg)
    ]
    expected_names = [f"{side}_arm_joint_{index}" for side in ("left", "right") for index in range(1, 5)]

    assert len(posture_objectives) == 1
    posture = posture_objectives[0]
    assert posture.joint_names == expected_names
    assert posture.target_positions == tuple(
        waterhose_env_cfg._RBY1_IK_INITIAL_JOINT_POS[name] for name in expected_names
    )
    assert posture.weight == pytest.approx(0.01)


def test_rate_limit_normalized_command_preserves_shared_gripper_interpolation():
    """Rate limiting keeps all gripper joints on one synchronized open/close interpolation."""

    actions_module = importlib.import_module("isaaclab_tasks.contrib.waterhose.mdp.actions")
    assert hasattr(actions_module, "_rate_limit_normalized_command")

    previous = torch.tensor([[0.0], [0.5]])
    desired = torch.tensor([[1.0], [0.0]])

    limited = actions_module._rate_limit_normalized_command(previous, desired, 0.25)

    expected = torch.tensor([[0.25], [0.25]])
    assert torch.allclose(limited, expected)


def test_bimanual_avp_calibration_preserves_wrist_pose_delta_one_to_one():
    """Calibration must remove morphology/tool offsets without changing wrist motion."""

    actions_module = importlib.import_module("isaaclab_tasks.contrib.waterhose.mdp.actions")
    rebase_pose_delta = actions_module._rebase_absolute_pose_delta

    wrist_reference_rot = Rotation.from_euler("XYZ", [24.0, -17.0, 31.0], degrees=True)
    wrist_delta_rot = Rotation.from_euler("XYZ", [-8.0, 13.0, 19.0], degrees=True)
    tool_offset_rot = Rotation.from_euler("X", 90.0, degrees=True)
    robot_reference_rot = Rotation.from_euler("XYZ", [-41.0, 22.0, 7.0], degrees=True)

    source_reference = torch.tensor(
        [[0.32, -0.18, 0.77, *((wrist_reference_rot * tool_offset_rot).as_quat())]], dtype=torch.float32
    )
    source_command = torch.tensor(
        [
            [
                0.38,
                -0.21,
                0.81,
                *((wrist_delta_rot * wrist_reference_rot * tool_offset_rot).as_quat()),
            ]
        ],
        dtype=torch.float32,
    )
    robot_reference = torch.tensor([[-0.27, 0.44, 0.63, *robot_reference_rot.as_quat()]], dtype=torch.float32)

    rebased = rebase_pose_delta(source_command, source_reference, robot_reference)

    assert rebased[0, :3].tolist() == pytest.approx([-0.21, 0.41, 0.67], abs=1.0e-6)
    expected_rotation = wrist_delta_rot * robot_reference_rot
    assert abs(float(torch.dot(rebased[0, 3:], torch.tensor(expected_rotation.as_quat(), dtype=torch.float32)))) == (
        pytest.approx(1.0, abs=1.0e-6)
    )


def test_bimanual_avp_clutch_prevents_acquisition_and_reacquisition_jumps():
    """A newly valid wrist must clutch onto the last robot target before moving."""

    actions_module = importlib.import_module("isaaclab_tasks.contrib.waterhose.mdp.actions")
    update_clutch = actions_module._update_clutched_absolute_pose
    robot_target = torch.tensor([[0.2, -0.3, 0.7, 0.0, 0.0, 0.0, 1.0]])
    source_reference = torch.zeros_like(robot_target)
    target_reference = torch.zeros_like(robot_target)
    tracking_valid = torch.tensor([False])
    first_wrist = torch.tensor([[0.5, 0.1, 1.1, 0.2, -0.1, 0.3, 0.9]])

    target, source_reference, target_reference, tracking_valid = update_clutch(
        first_wrist, robot_target, source_reference, target_reference, tracking_valid
    )
    assert torch.allclose(target, robot_target, atol=1.0e-6)
    assert tracking_valid.tolist() == [True]

    invalid_wrist = torch.tensor([[0.0, 0.0, 0.0, 0.7071068, 0.0, 0.0, 0.7071068]])
    held, source_reference, target_reference, tracking_valid = update_clutch(
        invalid_wrist, target, source_reference, target_reference, tracking_valid
    )
    assert torch.equal(held, target)
    assert tracking_valid.tolist() == [False]

    reacquired_wrist = torch.tensor([[-0.4, 0.8, 0.6, -0.3, 0.4, -0.1, 0.8]])
    reacquired, _, _, tracking_valid = update_clutch(
        reacquired_wrist, held, source_reference, target_reference, tracking_valid
    )
    assert torch.allclose(reacquired, held, atol=1.0e-6)
    assert tracking_valid.tolist() == [True]
