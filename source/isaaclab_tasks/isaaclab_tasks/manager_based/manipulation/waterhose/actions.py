# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import torch

from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor
from isaaclab.managers import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.utils import configclass

from . import waterhose_core as core


@configclass
class NewtonTaskSpaceIKActionCfg(ActionTermCfg):
    """Task-space RBY1 action backed by Newton analytic IK."""

    class_type: type[ActionTerm] = MISSING
    asset_name: str = "newton_waterhose"
    position_scale: float = 0.04
    rotation_scale: float = 0.25
    max_target_step: float = 0.018
    max_joint_step: float = 0.02
    max_gripper_joint_step: float = 0.20
    ik_iterations: int = 12


class NewtonTaskSpaceIKAction(ActionTerm):
    """Apply relative end-effector pose and gripper commands to the Newton RBY1 robot."""

    cfg: NewtonTaskSpaceIKActionCfg

    def __init__(self, cfg: NewtonTaskSpaceIKActionCfg, env):
        # This task has no Isaac Lab Articulation asset; Newton owns the robot buffers.
        ManagerTermBase.__init__(self, cfg, env)
        self._IO_descriptor = GenericActionIODescriptor()
        self._export_IO_descriptor = True
        self._debug_vis_handle = None
        self._raw_actions = torch.zeros(self.num_envs, 7, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._scene_builder = env.waterhose_scene_builder
        self._model = core.NewtonManager.get_model()
        self._control = core.NewtonManager.get_control()
        self._single_robot_model = self._scene_builder.single_robot_model
        self._body_ids = _resolve_body_ids(self._model.body_label, core.RIGHT_EE, self.num_envs)
        self._left_body_ids = _resolve_body_ids(self._model.body_label, core.LEFT_EE, self.num_envs)
        self._torso_body_ids = _resolve_body_ids(self._model.body_label, core.TORSO, self.num_envs)
        self._right_open_targets, self._right_closed_targets = _gripper_target_pairs(
            self._model, self._scene_builder.right_gripper_dofs
        )
        self._left_open_targets, _ = _gripper_target_pairs(self._model, self._scene_builder.left_gripper_dofs)
        self._setup_ik()
        self._seed_control_targets()

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        self._processed_actions[:, :3] = self._raw_actions[:, :3] * float(self.cfg.position_scale)
        self._processed_actions[:, 3:6] = self._raw_actions[:, 3:6] * float(self.cfg.rotation_scale)
        self._processed_actions[:, 6:] = self._raw_actions[:, 6:]

    def apply_actions(self) -> None:
        core.apply_viewer_forces(self._env.sim)
        state = core.NewtonManager.get_state_0()
        body_q = state.body_q.numpy()
        joint_q = state.joint_q.numpy()
        action_np = self._processed_actions.detach().cpu().numpy()
        target_np = self._control.joint_target_pos.numpy().astype(np.float32, copy=True)
        for env_id, joint_coord_ids in enumerate(self._scene_builder.robot_joint_coord_ids_by_env):
            ee_q = body_q[self._body_ids[env_id]]
            target_pos = ee_q[:3].astype(np.float64) + action_np[env_id, :3].astype(np.float64)
            target_quat = _integrate_axis_angle(ee_q[3:].astype(np.float64), action_np[env_id, 3:6].astype(np.float64))
            q = self._solve_env(env_id, joint_q[joint_coord_ids].astype(np.float32), target_pos, target_quat)
            self._set_gripper_targets(q, float(action_np[env_id, 6]))
            max_step = np.full_like(q, float(self.cfg.max_joint_step))
            if self._scene_builder.gripper_dofs:
                max_step[self._scene_builder.gripper_dofs] = float(self.cfg.max_gripper_joint_step)
            q = self._last_control_q[env_id] + np.clip(q - self._last_control_q[env_id], -max_step, max_step)
            self._last_control_q[env_id] = q.astype(np.float32, copy=True)
            target_np[joint_coord_ids] = q
        core.wp.copy(
            self._control.joint_target_pos,
            core.wp.array(target_np, dtype=core.wp.float32, device=self._model.device),
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0

    def target_pose_to_action(
        self, target_pos: torch.Tensor, target_quat_xyzw: torch.Tensor, env_id: int = 0
    ) -> torch.Tensor:
        curr = core.NewtonManager.get_state_0().body_q.numpy()[self._body_ids[env_id]]
        curr_pos = torch.as_tensor(curr[:3], device=self.device, dtype=torch.float32)
        delta_pos = target_pos.to(self.device) - curr_pos
        curr_quat = torch.as_tensor(curr[3:], device=self.device, dtype=torch.float32)
        delta_axis = _torch_axis_angle_between(curr_quat, target_quat_xyzw.to(self.device))
        action = torch.zeros(7, device=self.device)
        action[:3] = delta_pos / float(self.cfg.position_scale)
        action[3:6] = delta_axis / float(self.cfg.rotation_scale)
        return action.clamp(-1.0, 1.0)

    def action_to_target_pose(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state_np = core.NewtonManager.get_state_0().body_q.numpy()
        pos = []
        quat = []
        for env_id in range(actions.shape[0]):
            ee_q = state_np[self._body_ids[env_id]]
            pos.append(torch.as_tensor(ee_q[:3], device=self.device, dtype=torch.float32) + actions[env_id, :3])
            quat_np = _integrate_axis_angle(ee_q[3:].astype(np.float64), actions[env_id, 3:6].detach().cpu().numpy())
            quat.append(torch.as_tensor(quat_np, device=self.device, dtype=torch.float32))
        return torch.stack(pos, dim=0), torch.stack(quat, dim=0)

    def _setup_ik(self) -> None:
        body_q_np = core.NewtonManager.get_state_0().body_q.numpy()
        weights = (1.0, 1.0, 50.0)
        self._pos_objs = []
        self._rot_objs = []
        body_ids = (self._body_ids[0], self._left_body_ids[0], self._torso_body_ids[0])
        for body_id, weight in zip(body_ids, weights):
            tf = core.wp.transform(*body_q_np[body_id])
            self._pos_objs.append(
                core.ik.IKObjectivePosition(
                    link_index=body_id,
                    link_offset=core.wp.vec3(0.0, 0.0, 0.0),
                    target_positions=core.wp.array([core.wp.transform_get_translation(tf)], dtype=core.wp.vec3),
                    weight=weight,
                )
            )
            quat = core.wp.transform_get_rotation(tf)
            self._rot_objs.append(
                core.ik.IKObjectiveRotation(
                    link_index=body_id,
                    link_offset_rotation=core.wp.quat_identity(),
                    target_rotations=core.wp.array([core._quat_to_vec4(quat)], dtype=core.wp.vec4),
                    weight=weight,
                )
            )
        joint_limits = core.ik.IKObjectiveJointLimit(
            self._single_robot_model.joint_limit_lower,
            self._single_robot_model.joint_limit_upper,
            weight=10.0,
        )
        initial = self._single_robot_model.joint_q.numpy().astype(np.float32, copy=False)
        self._ik_joint_q = core.wp.array(
            initial,
            shape=(1, self._single_robot_model.joint_coord_count),
            dtype=core.wp.float32,
            device=self._model.device,
        )
        self._ik_solver = core.ik.IKSolver(
            model=self._single_robot_model,
            n_problems=1,
            objectives=[*self._pos_objs, *self._rot_objs, joint_limits],
            lambda_initial=0.1,
            jacobian_mode=core.ik.IKJacobianType.ANALYTIC,
        )

    def _seed_control_targets(self) -> None:
        initial = self._single_robot_model.joint_q.numpy().astype(np.float32, copy=False)
        target_np = self._control.joint_target_pos.numpy().astype(np.float32, copy=True)
        self._last_control_q = np.tile(initial, (self.num_envs, 1)).astype(np.float32)
        for joint_coord_ids in self._scene_builder.robot_joint_coord_ids_by_env:
            target_np[joint_coord_ids] = initial
        core.wp.copy(
            self._control.joint_target_pos,
            core.wp.array(target_np, dtype=core.wp.float32, device=self._model.device),
        )

    def _solve_env(
        self, env_id: int, current_q: np.ndarray, target_pos: np.ndarray, target_quat: np.ndarray
    ) -> np.ndarray:
        self._pos_objs[0].set_target_position(0, core.wp.vec3(*[float(v) for v in target_pos]))
        self._rot_objs[0].set_target_rotation(0, core.wp.vec4(*[float(v) for v in target_quat]))
        hold_body_ids = (self._left_body_ids[env_id], self._torso_body_ids[env_id])
        for objective_idx, body_id in enumerate(hold_body_ids, start=1):
            q = core.NewtonManager.get_state_0().body_q.numpy()[body_id]
            self._pos_objs[objective_idx].set_target_position(0, core.wp.vec3(*[float(v) for v in q[:3]]))
            self._rot_objs[objective_idx].set_target_rotation(0, core.wp.vec4(*[float(v) for v in q[3:]]))
        core.wp.copy(
            self._ik_joint_q,
            core.wp.array(
                current_q,
                shape=(1, self._single_robot_model.joint_coord_count),
                dtype=core.wp.float32,
                device=self._model.device,
            ),
        )
        self._ik_solver.step(
            self._ik_joint_q,
            self._ik_joint_q,
            iterations=int(self.cfg.ik_iterations),
        )
        return self._ik_joint_q.numpy().reshape(-1).astype(np.float32, copy=True)

    def _set_gripper_targets(self, q: np.ndarray, gripper_action: float) -> None:
        right_alpha = 1.0 if gripper_action > 0.0 else 0.0
        for idx, dof in enumerate(self._scene_builder.right_gripper_dofs[:2]):
            q[dof] = (1.0 - right_alpha) * self._right_open_targets[idx] + right_alpha * self._right_closed_targets[idx]
        for idx, dof in enumerate(self._scene_builder.left_gripper_dofs[:2]):
            q[dof] = self._left_open_targets[idx]


def _resolve_body_ids(labels: list[str], short_name: str, num_envs: int) -> list[int]:
    suffix = "/" + short_name
    matches = [idx for idx, label in enumerate(labels) if label == short_name or label.endswith(suffix)]
    if len(matches) < num_envs:
        raise RuntimeError(f"Expected at least {num_envs} bodies named {short_name!r}, found {matches}.")
    return matches[:num_envs]


def _gripper_target_pairs(model, dofs: list[int]) -> tuple[list[float], list[float]]:
    if len(dofs) < 2:
        return [0.0] * len(dofs), [0.0] * len(dofs)
    lower = model.joint_limit_lower.numpy()[dofs]
    upper = model.joint_limit_upper.numpy()[dofs]
    eps = 1.0e-4
    return [-0.04, 0.04], [float(upper[0] - eps), float(lower[1] + eps)]


def _integrate_axis_angle(quat_xyzw: np.ndarray, axis_angle: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(axis_angle))
    if angle <= 1.0e-8:
        return quat_xyzw / max(np.linalg.norm(quat_xyzw), 1.0e-12)
    delta = core._np_quat_from_axis_angle(axis_angle / angle, angle)
    quat = core._np_quat_multiply(delta, quat_xyzw)
    return quat / max(np.linalg.norm(quat), 1.0e-12)


def _torch_axis_angle_between(curr_xyzw: torch.Tensor, target_xyzw: torch.Tensor) -> torch.Tensor:
    curr_np = curr_xyzw.detach().cpu().numpy().astype(np.float64)
    target_np = target_xyzw.detach().cpu().numpy().astype(np.float64)
    delta = core._np_quat_multiply(target_np, core._np_quat_inverse(curr_np))
    delta = delta / max(np.linalg.norm(delta), 1.0e-12)
    angle = 2.0 * np.arctan2(np.linalg.norm(delta[:3]), delta[3])
    if angle <= 1.0e-8:
        return torch.zeros(3, device=curr_xyzw.device)
    axis = delta[:3] / max(np.linalg.norm(delta[:3]), 1.0e-12)
    return torch.as_tensor(axis * angle, device=curr_xyzw.device, dtype=torch.float32)
