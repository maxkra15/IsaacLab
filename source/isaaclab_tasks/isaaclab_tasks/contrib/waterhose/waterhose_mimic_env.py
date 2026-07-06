# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab Mimic wrapper for the RBY1DF waterhose teleop task."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

from .geometry import (
    RIGHT_GRIPPER_EE_FRAME_POS,
    RIGHT_GRIPPER_EE_FRAME_QUAT_XYZW,
    SOCKET_MOUTH_POS,
    SOCKET_ROT_QUAT_XYZW,
)

WATERHOSE_MIMIC_EEF_NAME = "right"
"""End-effector key used by the Waterhose Mimic task."""

_RIGHT_GRIPPER_BODY_NAME = "right_gripper_base"
_CABLE_HEAD_OBJECT_NAME = "plug"
_SOCKET_OBJECT_NAME = "socket"


def _as_batched_pose(pose: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return a batched pose tensor and whether the input was unbatched."""

    if pose.ndim == 2:
        return pose.unsqueeze(0), True
    return pose, False


def _as_batched_action(action: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return a batched action tensor and whether the input was unbatched."""

    if action.ndim == 1:
        return action.unsqueeze(0), True
    return action, False


def relative_action_to_target_pose(current_eef_pose: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Convert Waterhose's relative teleop pose action into a target end-effector pose.

    Translation deltas are interpreted in the robot/root frame. Rotation deltas are interpreted in
    the end-effector frame, matching :class:`WaterhoseTeleopPinnedNewtonIkAction`.
    """

    current_eef_pose, pose_was_unbatched = _as_batched_pose(current_eef_pose)
    action, _ = _as_batched_action(action)

    delta_position = action[:, :3]
    delta_rotation = action[:, 3:6]

    current_pos, current_rot = PoseUtils.unmake_pose(current_eef_pose)
    target_pos = current_pos + delta_position

    delta_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
    delta_axis = torch.zeros_like(delta_rotation)
    nonzero = delta_angle.squeeze(-1) > 1.0e-8
    delta_axis[nonzero] = delta_rotation[nonzero] / delta_angle[nonzero]
    delta_quat = PoseUtils.quat_from_angle_axis(delta_angle.squeeze(-1), delta_axis)
    target_rot = torch.matmul(current_rot, PoseUtils.matrix_from_quat(delta_quat))

    target_pose = PoseUtils.make_pose(target_pos, target_rot)
    return target_pose[0] if pose_was_unbatched else target_pose


def target_pose_to_relative_action(
    current_eef_pose: torch.Tensor,
    target_eef_pose: torch.Tensor,
    gripper_action: torch.Tensor,
    action_noise: float | None = None,
) -> torch.Tensor:
    """Convert a target pose into Waterhose's relative teleop action convention."""

    current_eef_pose, pose_was_unbatched = _as_batched_pose(current_eef_pose)
    target_eef_pose, _ = _as_batched_pose(target_eef_pose)
    gripper_action, _ = _as_batched_action(gripper_action)

    current_pos, current_rot = PoseUtils.unmake_pose(current_eef_pose)
    target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

    delta_position = target_pos - current_pos
    delta_rot_mat = torch.matmul(current_rot.transpose(-1, -2), target_rot)
    delta_rotation = PoseUtils.axis_angle_from_quat(PoseUtils.quat_from_matrix(delta_rot_mat))

    pose_action = torch.cat((delta_position, delta_rotation), dim=-1)
    if action_noise is not None:
        noise = float(action_noise) * torch.randn_like(pose_action)
        pose_action = torch.clamp(pose_action + noise, -1.0, 1.0)

    action = torch.cat((pose_action, gripper_action), dim=-1)
    return action[0] if pose_was_unbatched else action


class WaterhoseMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic-compatible wrapper for the Waterhose relative teleop task."""

    def _robot_root_pose(self, env_ids: Sequence[int] | slice) -> tuple[torch.Tensor, torch.Tensor]:
        robot = self.scene["robot"]
        return robot.data.root_pos_w.torch[env_ids], robot.data.root_quat_w.torch[env_ids]

    def _right_gripper_body_index(self) -> int:
        cached_index = getattr(self, "_waterhose_right_gripper_body_index", None)
        if cached_index is not None:
            return cached_index

        robot = self.scene["robot"]
        body_ids, body_names = robot.find_bodies(_RIGHT_GRIPPER_BODY_NAME)
        if len(body_ids) != 1:
            raise RuntimeError(f"Expected one {_RIGHT_GRIPPER_BODY_NAME!r} body, got {body_names}.")
        cached_index = int(body_ids[0])
        self._waterhose_right_gripper_body_index = cached_index
        return cached_index

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get the right end-effector pose in the robot/root frame."""

        if eef_name != WATERHOSE_MIMIC_EEF_NAME:
            raise ValueError(f"Unknown Waterhose end-effector {eef_name!r}; expected {WATERHOSE_MIMIC_EEF_NAME!r}.")
        if env_ids is None:
            env_ids = slice(None)

        robot = self.scene["robot"]
        body_index = self._right_gripper_body_index()
        root_pos_w, root_quat_w = self._robot_root_pose(env_ids)
        body_pos_w = robot.data.body_pos_w.torch[env_ids, body_index]
        body_quat_w = robot.data.body_quat_w.torch[env_ids, body_index]
        body_pos_b, body_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)

        offset_pos = torch.tensor(RIGHT_GRIPPER_EE_FRAME_POS, device=self.device, dtype=body_pos_b.dtype).expand(
            body_pos_b.shape[0], -1
        )
        offset_quat = torch.tensor(
            RIGHT_GRIPPER_EE_FRAME_QUAT_XYZW, device=self.device, dtype=body_quat_b.dtype
        ).expand(body_quat_b.shape[0], -1)
        eef_pos_b, eef_quat_b = combine_frame_transforms(body_pos_b, body_quat_b, offset_pos, offset_quat)
        return PoseUtils.make_pose(eef_pos_b, PoseUtils.matrix_from_quat(eef_quat_b))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert a Mimic target pose into the Waterhose 7D relative teleop action."""

        eef_name = WATERHOSE_MIMIC_EEF_NAME
        target_eef_pose = target_eef_pose_dict[eef_name]
        gripper_action = gripper_action_dict[eef_name]
        current_eef_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        action_noise = None if action_noise_dict is None else action_noise_dict[eef_name]
        return target_pose_to_relative_action(current_eef_pose, target_eef_pose, gripper_action, action_noise)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert recorded Waterhose teleop actions into target end-effector poses."""

        current_eef_pose = self.get_robot_eef_pose(WATERHOSE_MIMIC_EEF_NAME)
        target_pose = relative_action_to_target_pose(current_eef_pose, action[:, :6])
        return {WATERHOSE_MIMIC_EEF_NAME: target_pose}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract right-gripper commands from recorded Waterhose actions."""

        return {WATERHOSE_MIMIC_EEF_NAME: actions[..., -1:]}

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Get Mimic object reference poses in the robot/root frame."""

        if env_ids is None:
            env_ids = slice(None)

        root_pos_w, root_quat_w = self._robot_root_pose(env_ids)
        num_envs = root_pos_w.shape[0]
        device = root_pos_w.device
        dtype = root_pos_w.dtype

        socket_pos_l = torch.tensor(SOCKET_MOUTH_POS, device=device, dtype=dtype).expand(num_envs, -1)
        socket_pos_w = socket_pos_l + self.scene.env_origins[env_ids].to(device=device, dtype=dtype)
        socket_quat_w = torch.tensor(SOCKET_ROT_QUAT_XYZW, device=device, dtype=dtype).expand(num_envs, -1)
        socket_pos_b, socket_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, socket_pos_w, socket_quat_w)

        object_poses = {
            _SOCKET_OBJECT_NAME: PoseUtils.make_pose(socket_pos_b, PoseUtils.matrix_from_quat(socket_quat_b)),
        }
        object_poses[_CABLE_HEAD_OBJECT_NAME] = self._get_cable_head_pose(env_ids, root_pos_w, root_quat_w)
        return object_poses

    def _get_cable_head_pose(
        self,
        env_ids: Sequence[int] | slice,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        """Read the live connector pose from Newton and express it in the robot/root frame."""

        if isinstance(env_ids, slice):
            env_indices = list(range(self.num_envs))[env_ids]
        else:
            env_indices = [int(env_id) for env_id in env_ids]

        connector_pos_w, connector_quat_w = self.scene["cable1"].get_connector_pose_w()
        connector_pos_w = connector_pos_w[env_indices]
        connector_quat_w = connector_quat_w[env_indices]
        cable_head_pos_b, cable_head_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, connector_pos_w, connector_quat_w
        )
        return PoseUtils.make_pose(cable_head_pos_b, PoseUtils.matrix_from_quat(cable_head_quat_b))

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Return automatic subtask termination signals.

        The initial Waterhose Mimic cfg uses a single final subtask, so no intermediate termination
        signal is required. Manual annotation can introduce finer boundaries later.
        """

        return {}


class PreStepWaterhoseDatagenInfoRecorder(RecorderTerm):
    """Recorder term that stores Waterhose Mimic datagen info before each step."""

    def record_pre_step(self):
        eef_pose_dict = {}
        for eef_name in self._env.cfg.subtask_configs.keys():
            eef_pose_dict[eef_name] = self._env.get_robot_eef_pose(eef_name=eef_name)

        datagen_info = {
            "object_pose": self._env.get_object_poses(),
            "eef_pose": eef_pose_dict,
            "target_eef_pose": self._env.action_to_target_eef_pose(self._env.action_manager.action),
        }
        return "obs/datagen_info", datagen_info


@configclass
class PreStepWaterhoseDatagenInfoRecorderCfg(RecorderTermCfg):
    """Configuration for the Waterhose Mimic datagen-info recorder."""

    class_type: type[RecorderTerm] = PreStepWaterhoseDatagenInfoRecorder


class PreStepWaterhoseSubtaskTermsRecorder(RecorderTerm):
    """Recorder term that stores Waterhose Mimic subtask termination signals."""

    def record_pre_step(self):
        return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()


@configclass
class PreStepWaterhoseSubtaskTermsRecorderCfg(RecorderTermCfg):
    """Configuration for the Waterhose Mimic subtask termination recorder."""

    class_type: type[RecorderTerm] = PreStepWaterhoseSubtaskTermsRecorder


@configclass
class WaterhoseMimicRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Recorder configuration for Waterhose Mimic-compatible teleop demos."""

    record_pre_step_datagen_info = PreStepWaterhoseDatagenInfoRecorderCfg()
    record_pre_step_subtask_term_signals = PreStepWaterhoseSubtaskTermsRecorderCfg()
