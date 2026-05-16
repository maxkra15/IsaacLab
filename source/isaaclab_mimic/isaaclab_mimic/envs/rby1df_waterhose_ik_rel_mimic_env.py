# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

from isaaclab_tasks.manager_based.manipulation.waterhose import waterhose_core as core  # isort: skip

core.import_newton_dependencies()

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv

from isaaclab_tasks.manager_based.manipulation.waterhose.waterhose_env import RBY1DFWaterhoseEnv


class RBY1DFWaterhoseIKRelMimicEnv(RBY1DFWaterhoseEnv, ManagerBasedRLMimicEnv):
    """Isaac Lab Mimic wrapper for the RBY1 waterhose IK-relative task."""

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat_xyzw = self.obs_buf["policy"]["eef_quat"][env_ids]
        eef_quat_wxyz = torch.cat((eef_quat_xyzw[:, 3:4], eef_quat_xyzw[:, :3]), dim=-1)
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat_wxyz))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)
        target_quat_wxyz = PoseUtils.quat_from_matrix(target_rot)
        target_quat_xyzw = torch.cat((target_quat_wxyz[1:], target_quat_wxyz[:1]), dim=0)
        action = self.get_task_space_action_term().target_pose_to_action(target_pos, target_quat_xyzw, env_id=env_id)
        if action_noise_dict is not None:
            action[:6] += action_noise_dict[eef_name] * torch.randn_like(action[:6])
            action[:6] = torch.clamp(action[:6], -1.0, 1.0)
        (gripper_action,) = gripper_action_dict.values()
        action[-1:] = gripper_action.to(action.device)
        return action

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        target_pos, target_quat_xyzw = self.get_task_space_action_term().action_to_target_pose(action)
        target_quat_wxyz = torch.cat((target_quat_xyzw[:, 3:4], target_quat_xyzw[:, :3]), dim=-1)
        return {eef_name: PoseUtils.make_pose(target_pos, PoseUtils.matrix_from_quat(target_quat_wxyz)).clone()}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, -1:]}

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        object_pose_matrix = dict()
        for name, pos_key, quat_key in (
            ("hose_plug", "plug_pos", "plug_quat"),
            ("hose_tip", "tip_pos", "tip_quat"),
            ("socket", "socket_pose", "socket_pose"),
        ):
            if name == "socket":
                pose = self.obs_buf["policy"]["socket_pose"][env_ids]
                pos = pose[:, :3]
                quat_xyzw = pose[:, 3:7]
            else:
                pos = self.obs_buf["policy"][pos_key][env_ids]
                quat_xyzw = self.obs_buf["policy"][quat_key][env_ids]
            quat_wxyz = torch.cat((quat_xyzw[:, 3:4], quat_xyzw[:, :3]), dim=-1)
            object_pose_matrix[name] = PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat_wxyz))
        return object_pose_matrix

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        terms = self.obs_buf["subtask_terms"]
        return {
            "approach": terms["approach"][env_ids],
            "grasp": terms["grasp"][env_ids],
            "align": terms["align"][env_ids],
            "insert": terms["insert"][env_ids],
        }
