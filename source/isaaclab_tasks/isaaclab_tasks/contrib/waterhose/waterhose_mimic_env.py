# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab Mimic wrapper for direct bimanual RBY1 waterhose control."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.utils.math import subtract_frame_transforms

from .geometry import SOCKET_MOUTH_POS, SOCKET_ROT_QUAT_XYZW

WATERHOSE_MIMIC_EEF_NAMES = ("right", "left")
"""End-effector keys used by the bimanual Waterhose Mimic task."""

WATERHOSE_MIMIC_ACTION_DIM = 20
"""Dimension of the direct Waterhose Mimic action."""

_WRIST_ACTION_SLICES = {"right": slice(0, 7), "left": slice(7, 14)}
_GRIPPER_ACTION_SLICES = {"right": slice(14, 17), "left": slice(17, 20)}
_EEF_BODY_NAMES = {"right": "right_gripper_base", "left": "left_gripper_base"}
_CABLE_HEAD_OBJECT_NAME = "plug"
_SOCKET_OBJECT_NAME = "socket"


def waterhose_mimic_action_to_target_eef_poses(action: torch.Tensor) -> dict[str, torch.Tensor]:
    """Unpack direct wrist targets from a Waterhose Mimic action."""

    if action.shape[-1] != WATERHOSE_MIMIC_ACTION_DIM:
        raise ValueError(
            f"Expected a {WATERHOSE_MIMIC_ACTION_DIM}D Waterhose Mimic action, got shape {tuple(action.shape)}."
        )

    target_poses = {}
    for eef_name in WATERHOSE_MIMIC_EEF_NAMES:
        wrist_action = action[..., _WRIST_ACTION_SLICES[eef_name]]
        target_poses[eef_name] = PoseUtils.make_pose(
            wrist_action[..., :3],
            PoseUtils.matrix_from_quat(wrist_action[..., 3:7]),
        )
    return target_poses


def waterhose_mimic_actions_to_gripper_actions(actions: torch.Tensor) -> dict[str, torch.Tensor]:
    """Extract the two three-joint gripper targets from direct Mimic actions."""

    if actions.shape[-1] != WATERHOSE_MIMIC_ACTION_DIM:
        raise ValueError(f"Expected {WATERHOSE_MIMIC_ACTION_DIM} action values, got shape {tuple(actions.shape)}.")
    return {eef_name: actions[..., action_slice] for eef_name, action_slice in _GRIPPER_ACTION_SLICES.items()}


def target_eef_poses_and_grippers_to_waterhose_mimic_action(
    target_eef_pose_dict: dict[str, torch.Tensor],
    gripper_action_dict: dict[str, torch.Tensor],
    action_noise_dict: dict[str, torch.Tensor | float] | None = None,
) -> torch.Tensor:
    """Pack bimanual controller poses and explicit gripper targets into a direct action."""

    action_parts = []
    for eef_name in WATERHOSE_MIMIC_EEF_NAMES:
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose_dict[eef_name])
        target_quat = PoseUtils.normalize(PoseUtils.quat_from_matrix(target_rot))
        wrist_action = torch.cat((target_pos, target_quat), dim=-1)
        if action_noise_dict is not None:
            wrist_action = wrist_action + action_noise_dict[eef_name] * torch.randn_like(wrist_action)
            wrist_action = torch.cat(
                (wrist_action[..., :3], PoseUtils.normalize(wrist_action[..., 3:7])),
                dim=-1,
            )
        action_parts.append(wrist_action)

    for eef_name in WATERHOSE_MIMIC_EEF_NAMES:
        gripper_action = gripper_action_dict[eef_name]
        if gripper_action.shape[-1] != 3:
            raise ValueError(
                f"Expected three {eef_name} gripper joint targets, got shape {tuple(gripper_action.shape)}."
            )
        action_parts.append(gripper_action)

    action = torch.cat(action_parts, dim=-1)
    if action.shape[-1] != WATERHOSE_MIMIC_ACTION_DIM:
        raise RuntimeError(f"Packed an invalid Waterhose Mimic action with shape {tuple(action.shape)}.")
    return action


class WaterhoseMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic-compatible environment using direct robot-side bimanual targets."""

    def _robot_root_pose(self, env_ids: Sequence[int] | slice) -> tuple[torch.Tensor, torch.Tensor]:
        robot = self.scene["robot"]
        return robot.data.root_pos_w.torch[env_ids], robot.data.root_quat_w.torch[env_ids]

    def _eef_body_index(self, eef_name: str) -> int:
        if eef_name not in WATERHOSE_MIMIC_EEF_NAMES:
            raise ValueError(f"Unknown Waterhose end-effector {eef_name!r}; expected {WATERHOSE_MIMIC_EEF_NAMES}.")

        body_indices = getattr(self, "_waterhose_eef_body_indices", None)
        if body_indices is None:
            body_indices = {}
            self._waterhose_eef_body_indices = body_indices
        if eef_name not in body_indices:
            body_name = _EEF_BODY_NAMES[eef_name]
            body_ids, body_names = self.scene["robot"].find_bodies(body_name)
            if len(body_ids) != 1:
                raise RuntimeError(f"Expected one {body_name!r} body, got {body_names}.")
            body_indices[eef_name] = int(body_ids[0])
        return body_indices[eef_name]

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get one physical wrist pose in the robot-root frame."""

        if env_ids is None:
            env_ids = slice(None)
        body_index = self._eef_body_index(eef_name)
        robot = self.scene["robot"]
        root_pos_w, root_quat_w = self._robot_root_pose(env_ids)
        body_pos_b, body_quat_b = subtract_frame_transforms(
            root_pos_w,
            root_quat_w,
            robot.data.body_pos_w.torch[env_ids, body_index],
            robot.data.body_quat_w.torch[env_ids, body_index],
        )
        return PoseUtils.make_pose(body_pos_b, PoseUtils.matrix_from_quat(body_quat_b))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert two Mimic target poses directly into the 20D environment action."""

        del env_id
        return target_eef_poses_and_grippers_to_waterhose_mimic_action(
            target_eef_pose_dict,
            gripper_action_dict,
            action_noise_dict,
        )

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract both robot-side wrist targets from a direct environment action."""

        return waterhose_mimic_action_to_target_eef_poses(action)

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract both explicit three-joint hand targets from recorded actions."""

        return waterhose_mimic_actions_to_gripper_actions(actions)

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Get socket and connector poses in the robot-root frame."""

        if env_ids is None:
            env_ids = slice(None)
        root_pos_w, root_quat_w = self._robot_root_pose(env_ids)
        num_envs = root_pos_w.shape[0]
        device = root_pos_w.device
        dtype = root_pos_w.dtype

        socket_pos_w = torch.tensor(SOCKET_MOUTH_POS, device=device, dtype=dtype).expand(num_envs, -1)
        socket_pos_w = socket_pos_w + self.scene.env_origins[env_ids].to(device=device, dtype=dtype)
        socket_quat_w = torch.tensor(SOCKET_ROT_QUAT_XYZW, device=device, dtype=dtype).expand(num_envs, -1)
        socket_pos_b, socket_quat_b = subtract_frame_transforms(
            root_pos_w,
            root_quat_w,
            socket_pos_w,
            socket_quat_w,
        )

        connector_pos_w, connector_quat_w = self.scene["cable1"].get_connector_pose_w()
        connector_pos_w = connector_pos_w[env_ids]
        connector_quat_w = connector_quat_w[env_ids]
        connector_pos_b, connector_quat_b = subtract_frame_transforms(
            root_pos_w,
            root_quat_w,
            connector_pos_w,
            connector_quat_w,
        )

        return {
            _SOCKET_OBJECT_NAME: PoseUtils.make_pose(socket_pos_b, PoseUtils.matrix_from_quat(socket_quat_b)),
            _CABLE_HEAD_OBJECT_NAME: PoseUtils.make_pose(
                connector_pos_b,
                PoseUtils.matrix_from_quat(connector_quat_b),
            ),
        }
