# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim-free contract tests for the direct bimanual Waterhose Mimic action."""

from types import SimpleNamespace

import gymnasium as gym
import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

from isaaclab_tasks.contrib.waterhose.config.rby1df.mimic_env_cfg import WaterhoseMimicEnvCfg
from isaaclab_tasks.contrib.waterhose.mdp.actions import (
    WaterhoseBimanualTeleopNewtonIkAction,
    WaterhoseDirectBimanualNewtonIkAction,
    _capture_pending_hold_targets_b,
    _write_direct_bimanual_ik_actions,
)
from isaaclab_tasks.contrib.waterhose.waterhose_env_cfg import WaterhoseProxyTeleopEnvCfg
from isaaclab_tasks.contrib.waterhose.waterhose_mimic_env import (
    WATERHOSE_MIMIC_ACTION_DIM,
    target_eef_poses_and_grippers_to_waterhose_mimic_action,
    waterhose_mimic_action_to_target_eef_poses,
    waterhose_mimic_actions_to_gripper_actions,
)


def test_mimic_config_uses_direct_20d_processed_actions_without_xr():
    """The Mimic task must not replay processed targets through the AVP clutch."""

    cfg = WaterhoseMimicEnvCfg()
    teleop_cfg = WaterhoseProxyTeleopEnvCfg()

    assert list(cfg.actions.__dict__) == ["arm_action", "gripper_action", "left_gripper_action"]
    assert cfg.actions.arm_action.class_type.endswith(":WaterhoseDirectBimanualNewtonIkAction")
    assert teleop_cfg.actions.arm_action.class_type.endswith(":WaterhoseBimanualTeleopNewtonIkAction")
    assert 14 + len(cfg.actions.gripper_action.joint_names) + len(cfg.actions.left_gripper_action.joint_names) == 20
    assert WATERHOSE_MIMIC_ACTION_DIM == 20
    assert cfg.annotation_replay_action_key == "processed_actions"
    assert cfg.terminations.success is not None
    assert cfg.xr is None
    assert cfg.isaac_teleop is None
    assert all(subtask.action_noise == 0.0 for eef_subtasks in cfg.subtask_configs.values() for subtask in eef_subtasks)


def test_bimanual_pose_and_three_joint_gripper_action_round_trip():
    """Both wrist poses and both explicit hand targets must preserve their order."""

    poses = {
        "right": PoseUtils.make_pose(
            torch.tensor([0.1, -0.2, 0.3]),
            PoseUtils.matrix_from_quat(PoseUtils.normalize(torch.tensor([0.1, -0.2, 0.3, 0.9]))),
        ),
        "left": PoseUtils.make_pose(
            torch.tensor([-0.4, 0.5, 0.6]),
            PoseUtils.matrix_from_quat(PoseUtils.normalize(torch.tensor([-0.3, 0.2, 0.1, 0.9]))),
        ),
    }
    grippers = {
        "right": torch.tensor([0.011, -0.012, 0.013]),
        "left": torch.tensor([0.021, -0.022, 0.023]),
    }

    action = target_eef_poses_and_grippers_to_waterhose_mimic_action(poses, grippers)
    unpacked_poses = waterhose_mimic_action_to_target_eef_poses(action)
    unpacked_grippers = waterhose_mimic_actions_to_gripper_actions(action)

    assert action.shape == (20,)
    torch.testing.assert_close(action[:3], torch.tensor([0.1, -0.2, 0.3]))
    torch.testing.assert_close(action[7:10], torch.tensor([-0.4, 0.5, 0.6]))
    for eef_name in ("right", "left"):
        torch.testing.assert_close(unpacked_poses[eef_name], poses[eef_name])
        torch.testing.assert_close(unpacked_grippers[eef_name], grippers[eef_name])


def test_gripper_extraction_preserves_arbitrary_leading_dimensions():
    """Mimic may pass an environment-by-time action tensor to the extractor."""

    actions = torch.arange(2 * 5 * 20, dtype=torch.float32).reshape(2, 5, 20)

    grippers = waterhose_mimic_actions_to_gripper_actions(actions)

    assert grippers["right"].shape == (2, 5, 3)
    assert grippers["left"].shape == (2, 5, 3)
    torch.testing.assert_close(grippers["right"], actions[..., 14:17])
    torch.testing.assert_close(grippers["left"], actions[..., 17:20])


def test_direct_ik_packing_has_no_teleop_clutch_or_rebase():
    """Recorded robot-side wrist targets must enter Newton IK unchanged."""

    wrists = torch.arange(28, dtype=torch.float32).reshape(2, 14)
    holds = torch.arange(14, dtype=torch.float32).reshape(2, 1, 7) + 100.0
    full_actions = torch.full((2, 21), -1.0)

    result = _write_direct_bimanual_ik_actions(full_actions, wrists, holds, (14,))

    assert result.data_ptr() == full_actions.data_ptr()
    torch.testing.assert_close(result[:, :14], wrists)
    torch.testing.assert_close(result[:, 14:21], holds[:, 0])
    assert not issubclass(WaterhoseDirectBimanualNewtonIkAction, WaterhoseBimanualTeleopNewtonIkAction)


def test_direct_ik_io_descriptor_describes_external_wrist_action():
    """The descriptor must exclude the internally generated torso-hold target."""

    pose_coordinates = ("x", "y", "z", "qx", "qy", "qz", "qw")

    def make_driver(name: str):
        objective = SimpleNamespace(name=name, command_coordinate_names=lambda: list(pose_coordinates))
        return SimpleNamespace(objective=objective)

    action = WaterhoseDirectBimanualNewtonIkAction.__new__(WaterhoseDirectBimanualNewtonIkAction)
    action._IO_descriptor = GenericActionIODescriptor()
    action._export_IO_descriptor = True
    action._action_dim = 21
    action._direct_action_dim = 14
    action._direct_raw_actions = torch.zeros(1, 14)
    action._joint_names = []
    action.cfg = SimpleNamespace(clip=None, controller=SimpleNamespace())
    action._wrist_drivers = [make_driver("right_ee"), make_driver("left_ee")]
    action._drivers = [*action._wrist_drivers, make_driver("torso_hold")]

    descriptor = action.IO_descriptor

    assert descriptor.shape == (14,)
    assert descriptor.action_type == "WaterhoseDirectBimanualNewtonIkAction"
    assert descriptor.extras["objective_names"] == ["right_ee", "left_ee"]
    assert descriptor.extras["coordinate_names"] == [
        f"{eef_name}/{coordinate}" for eef_name in ("right_ee", "left_ee") for coordinate in pose_coordinates
    ]


def test_direct_ik_partial_reset_recaptures_only_pending_environment_holds():
    """Recapturing a reset environment must preserve every active environment's hold."""

    identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
    root_pos_w = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    root_quat_w = identity_quat.repeat(3, 1)
    body_pos_w = torch.tensor(
        [
            [[101.0, 10.0, 0.0]],
            [[12.0, 20.0, 0.0]],
            [[103.0, 30.0, 0.0]],
        ]
    )
    body_quat_w = identity_quat.repeat(3, 1, 1)
    hold_targets_b = torch.tensor(
        [
            [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]],
            [[-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0]],
            [[3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 1.0]],
        ]
    )
    holds_captured = torch.tensor([True, False, True])
    preserved_env_0 = hold_targets_b[0].clone()
    preserved_env_2 = hold_targets_b[2].clone()

    _capture_pending_hold_targets_b(
        hold_targets_b,
        holds_captured,
        root_pos_w,
        root_quat_w,
        body_pos_w,
        body_quat_w,
        (0,),
    )

    torch.testing.assert_close(hold_targets_b[0], preserved_env_0)
    torch.testing.assert_close(hold_targets_b[2], preserved_env_2)
    torch.testing.assert_close(
        hold_targets_b[1, 0],
        torch.tensor([10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    assert torch.all(holds_captured)

    # A later partial reset of env 2 must likewise leave envs 0 and 1 untouched.
    preserved_env_0 = hold_targets_b[0].clone()
    preserved_env_1 = hold_targets_b[1].clone()
    holds_captured[2] = False
    body_pos_w[2, 0] = torch.tensor([8.0, 40.0, 0.0])
    _capture_pending_hold_targets_b(
        hold_targets_b,
        holds_captured,
        root_pos_w,
        root_quat_w,
        body_pos_w,
        body_quat_w,
        (0,),
    )

    torch.testing.assert_close(hold_targets_b[0], preserved_env_0)
    torch.testing.assert_close(hold_targets_b[1], preserved_env_1)
    torch.testing.assert_close(
        hold_targets_b[2, 0],
        torch.tensor([5.0, 40.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    )


def test_waterhose_mimic_task_is_registered():
    """The dedicated task must resolve to the Mimic wrapper and configuration."""

    spec = gym.spec("Isaac-Waterhose-Coupled-Mimic-v0")

    assert spec.entry_point == "isaaclab_tasks.contrib.waterhose.waterhose_mimic_env:WaterhoseMimicEnv"
    assert spec.kwargs["env_cfg_entry_point"].endswith(".mimic_env_cfg:WaterhoseMimicEnvCfg")
