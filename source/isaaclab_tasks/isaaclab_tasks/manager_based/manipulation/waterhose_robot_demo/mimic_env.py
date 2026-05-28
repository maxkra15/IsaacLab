# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab Mimic wrapper for the waterhose robot demo."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv

from .env import WaterhoseRobotDemoEnv
from .manager import NewtonWaterhoseManager


class WaterhoseRobotDemoMimicEnv(WaterhoseRobotDemoEnv, ManagerBasedRLMimicEnv):
    """Mimic API adapter for the stable one-way waterhose demo task."""

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        poses = self._poses_to_tensor(NewtonWaterhoseManager.get_right_ee_poses(), env_ids)
        return self._pose_tensor_to_matrix(poses)

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        target_eef_pose = target_eef_pose_dict[eef_name]
        target_eef_pose = target_eef_pose.to(device=self.device, dtype=torch.float32)
        target_eef_pose = target_eef_pose.reshape(-1, 4, 4)[0]
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)
        target_pos = target_pos.reshape(3)
        target_rot = target_rot.reshape(3, 3)

        current_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        current_pos, current_rot = PoseUtils.unmake_pose(current_pose)

        delta_pos = target_pos - current_pos
        delta_rot_mat = current_rot.transpose(-1, -2).matmul(target_rot)
        delta_rot = PoseUtils.axis_angle_from_quat(PoseUtils.quat_from_matrix(delta_rot_mat))

        cfg = self._demo_action_cfg()
        action = torch.zeros(self.action_manager.total_action_dim, device=self.device, dtype=torch.float32)
        action[:3] = delta_pos / float(cfg.position_scale)
        action[3:6] = delta_rot / float(cfg.rotation_scale)

        if action_noise_dict is not None:
            action[:6] += action_noise_dict[eef_name] * torch.randn_like(action[:6])

        gripper_action = gripper_action_dict[eef_name]
        action[-1:] = gripper_action.to(device=self.device, dtype=torch.float32).reshape(-1)[:1]
        return action.clamp(-1.0, 1.0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        action = action.to(device=self.device, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0)

        current_pose = self.get_robot_eef_pose(eef_name, env_ids=None)[: action.shape[0]]
        current_pos, current_rot = PoseUtils.unmake_pose(current_pose)
        delta_pos, delta_rot = self._action_to_pose_delta(action)
        target_pos = current_pos + delta_pos
        target_rot = current_rot.matmul(self._axis_angle_to_matrix(delta_rot))
        return {eef_name: PoseUtils.make_pose(target_pos, target_rot).clone()}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        return {eef_name: actions[..., -1:]}

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        return {
            name: self._pose_tensor_to_matrix(self._poses_to_tensor(poses, env_ids))
            for name, poses in NewtonWaterhoseManager.get_object_poses().items()
        }

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        return {
            name: self._slice_env_ids(torch.as_tensor(value, device=self.device, dtype=torch.bool), env_ids)
            for name, value in NewtonWaterhoseManager.get_subtask_term_signals().items()
        }

    def _demo_action_cfg(self):
        return self.action_manager.get_term("demo").cfg

    def _action_to_pose_delta(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self._demo_action_cfg()
        action = action.clamp(-1.0, 1.0)
        delta_pos = action[:, :3] * float(cfg.position_scale)
        max_target_step = float(cfg.max_target_step)
        if max_target_step > 0.0:
            norm = torch.linalg.vector_norm(delta_pos, dim=-1, keepdim=True).clamp_min(1.0e-12)
            delta_pos = delta_pos * torch.clamp(max_target_step / norm, max=1.0)
        delta_rot = action[:, 3:6] * float(cfg.rotation_scale)
        return delta_pos, delta_rot

    def _axis_angle_to_matrix(self, axis_angle: torch.Tensor) -> torch.Tensor:
        angle = torch.linalg.vector_norm(axis_angle, dim=-1)
        axis = torch.zeros_like(axis_angle)
        nonzero = angle > 1.0e-8
        axis[nonzero] = axis_angle[nonzero] / angle[nonzero].unsqueeze(-1)
        axis[~nonzero, 0] = 1.0
        return PoseUtils.matrix_from_quat(PoseUtils.quat_from_angle_axis(angle, axis))

    def _poses_to_tensor(self, poses, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        tensor = torch.as_tensor(poses, device=self.device, dtype=torch.float32).reshape(-1, 7)
        return self._slice_env_ids(tensor, env_ids)

    @staticmethod
    def _pose_tensor_to_matrix(poses: torch.Tensor) -> torch.Tensor:
        return PoseUtils.make_pose(poses[:, :3], PoseUtils.matrix_from_quat(poses[:, 3:7]))

    @staticmethod
    def _slice_env_ids(value: torch.Tensor, env_ids: Sequence[int] | None):
        if env_ids is None:
            return value
        return value[env_ids]
