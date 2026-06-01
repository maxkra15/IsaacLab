# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted IK state machine for the RBY1 waterhose scene-config tasks."""

from __future__ import annotations

import torch

from isaaclab.utils.math import (
    combine_frame_transforms,
    normalize,
    quat_apply,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

# Current RBY1DF USD has no right_gripper_end_effector body; use the measured
# midpoint between the right finger bodies in right_gripper_base local frame.
_RIGHT_EE_FROM_BASE_POS = (0.0, 0.0, -0.075)
# USD stores xformOp:orient as (w, x, y, z); IsaacLab frame math uses (x, y, z, w).
_RIGHT_EE_FROM_BASE_QUAT = (0.70710677, 0.70710677, 0.0, 0.0)
_PLUG_TIP_OFFSET = (0.0, 0.0, -0.014106234)
_PLUG_GRASP_OFFSET = (0.0, 0.05, 0.0)
_CABLE1_PLUG_SEGMENT_ID = 0
_CABLE1_PLUG_LOCAL_POS = (0.0, 0.0, 0.022)
_GRIPPER_INITIAL_GRASP_COMMAND = -0.72
_GRIPPER_FALLBACK_GRASP_COMMAND = -0.86
_GRIPPER_MAX_FORCE_CLOSE_COMMAND = -0.94
_GRIPPER_FORCE_TARGET_N = 35.0
_GRIPPER_TIGHTEN_RATE = 0.45


def _normalize_vector(value: torch.Tensor) -> torch.Tensor:
    return value / torch.clamp(torch.linalg.vector_norm(value, dim=-1, keepdim=True), min=1.0e-8)


def _correction_quat_between_vectors(
    from_vector: torch.Tensor,
    to_vector: torch.Tensor,
    gain: float,
) -> torch.Tensor:
    from_vector = _normalize_vector(from_vector)
    to_vector = _normalize_vector(to_vector)
    axis = torch.cross(from_vector, to_vector, dim=-1)
    sin_angle = torch.linalg.vector_norm(axis, dim=-1)
    axis = axis / torch.clamp(sin_angle.unsqueeze(-1), min=1.0e-8)
    fallback_axis = torch.zeros_like(axis)
    fallback_axis[:, 0] = 1.0
    axis = torch.where(sin_angle.unsqueeze(-1) > 1.0e-8, axis, fallback_axis)
    cos_angle = torch.sum(from_vector * to_vector, dim=-1).clamp(-1.0, 1.0)
    angle = torch.atan2(sin_angle, cos_angle) * gain
    return normalize(quat_from_angle_axis(angle, axis))


class WaterhoseDemoState:
    """Per-environment scripted pick-and-insert state machine."""

    REST = 0
    APPROACH = 1
    ENGAGE = 2
    GRASP = 3
    HOLD_GRASP = 4
    RETRACT = 5
    SETTLE = 6
    APPROACH_TARGET = 7
    ALIGN_AXES = 8
    VERIFY_ALIGN = 9
    INSERT = 10
    RELEASE = 11
    WITHDRAW = 12
    DONE = 13

    PHASE_NAMES = (
        "REST",
        "APPROACH",
        "ENGAGE",
        "GRASP",
        "HOLD_GRASP",
        "RETRACT",
        "SETTLE",
        "APPROACH_TARGET",
        "ALIGN_AXES",
        "VERIFY_ALIGN",
        "INSERT",
        "RELEASE",
        "WITHDRAW",
        "DONE",
    )
    DURATIONS = (0.25, 1.0, 1.5, 0.5, 0.5, 1.5, 0.3, 5.0, 5.0, 2.0, 5.0, 1.0, 2.0, 1.0e6)

    def __init__(self, num_envs: int, step_dt: float, device: torch.device | str, settle_time: float, debug: bool):
        self.num_envs = int(num_envs)
        self.step_dt = float(step_dt)
        self.device = device
        self.debug = bool(debug)
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.elapsed = torch.zeros(self.num_envs, device=device)
        self.last_reported_phase = torch.full((self.num_envs,), -1, dtype=torch.long, device=device)
        self.phase_start_pose = torch.zeros((self.num_envs, 7), device=device)
        self.phase_start_pose_w = torch.zeros((self.num_envs, 7), device=device)
        self.command_pose = torch.zeros((self.num_envs, 7), device=device)
        self.command_pose[:, 6] = 1.0
        durations = list(self.DURATIONS)
        durations[self.REST] = max(float(settle_time), self.step_dt)
        self.durations = torch.tensor(durations, dtype=torch.float32, device=device)
        pos_tolerances = torch.full((len(self.PHASE_NAMES), 3), 999.0, dtype=torch.float32, device=device)
        rot_tolerances = torch.full((len(self.PHASE_NAMES),), 999.0, dtype=torch.float32, device=device)
        default_pos_tol = torch.tensor([0.001, 0.001, 0.002], dtype=torch.float32, device=device)
        default_rot_tol = 5.0 * torch.pi / 180.0
        for phase in (self.APPROACH, self.ENGAGE, self.GRASP, self.RETRACT, self.WITHDRAW):
            pos_tolerances[phase] = default_pos_tol
            rot_tolerances[phase] = default_rot_tol
        pos_tolerances[self.ENGAGE] = torch.tensor([0.005, 0.005, 0.005], dtype=torch.float32, device=device)
        pos_tolerances[self.GRASP] = torch.tensor([0.005, 0.005, 0.005], dtype=torch.float32, device=device)
        pos_tolerances[self.RETRACT] = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32, device=device)
        pos_tolerances[self.WITHDRAW] = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32, device=device)
        rot_tolerances[self.ENGAGE] = 10.0 * torch.pi / 180.0
        rot_tolerances[self.RETRACT] = 10.0 * torch.pi / 180.0
        rot_tolerances[self.WITHDRAW] = 10.0 * torch.pi / 180.0
        self.pos_tolerances = pos_tolerances
        self.rot_tolerances = rot_tolerances
        self.approach_offset = torch.tensor([0.0, 0.08, 0.0], dtype=torch.float32, device=device).repeat(
            self.num_envs, 1
        )
        self.engage_offset = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=device)
        self.retract_vector = torch.tensor([0.0, 0.05, 0.0], dtype=torch.float32, device=device).repeat(
            self.num_envs, 1
        )
        self.withdraw_offset = torch.tensor([-0.10, 0.0, 0.0], dtype=torch.float32, device=device).repeat(
            self.num_envs, 1
        )
        self.plug_tip_offset = torch.tensor(_PLUG_TIP_OFFSET, dtype=torch.float32, device=device).repeat(
            self.num_envs, 1
        )
        self.plug_cable_local_pos = torch.tensor(_CABLE1_PLUG_LOCAL_POS, dtype=torch.float32, device=device).repeat(
            self.num_envs, 1
        )
        self.insertion_start_depth = 0.005
        self.insert_final_depth = 0.035
        self.insert_duration = 4.0
        self.align_orientation_gain = 0.35
        self.insert_orientation_gain = 0.2
        self.verify_lateral_gain = 1.0
        self.insert_lateral_gain = 0.5
        self.insert_lateral_integral_gain = 5.0
        self.insert_lateral_integral = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=device)
        self.plug_grasp_fallback_offset = torch.tensor(_PLUG_GRASP_OFFSET, dtype=torch.float32, device=device).repeat(
            self.num_envs, 1
        )
        self.plug_grasp_offset = self.plug_grasp_fallback_offset.clone()
        z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device).repeat(self.num_envs, 1)
        x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device).repeat(self.num_envs, 1)
        q_rz = quat_from_angle_axis(
            torch.full((self.num_envs,), -1.5707963267948966, dtype=torch.float32, device=device), z_axis
        )
        q_rx = quat_from_angle_axis(
            torch.full((self.num_envs,), 1.5707963267948966, dtype=torch.float32, device=device), x_axis
        )
        self.grasp_orientation_offset = normalize(quat_mul(q_rx, q_rz))
        self.socket_pos_w = torch.tensor(
            [-0.259404, 0.362961, 0.5 - 0.262711],
            dtype=torch.float32,
            device=device,
        ).repeat(self.num_envs, 1)
        socket_angle = torch.full((self.num_envs,), 0.3490658503988659, dtype=torch.float32, device=device)
        socket_axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device).repeat(self.num_envs, 1)
        self.socket_quat_w = quat_from_angle_axis(socket_angle, socket_axis)
        self.ee_offset_pos = torch.tensor(_RIGHT_EE_FROM_BASE_POS, dtype=torch.float32, device=device).repeat(
            self.num_envs, 1
        )
        self.ee_offset_quat = torch.tensor(_RIGHT_EE_FROM_BASE_QUAT, dtype=torch.float32, device=device).repeat(
            self.num_envs, 1
        )
        self._plug_segment_body_ids = None
        self._plug_body_ids = None
        self._right_proxy_body_ids = None
        self.grip_command = torch.full(
            (self.num_envs, 1), _GRIPPER_INITIAL_GRASP_COMMAND, dtype=torch.float32, device=device
        )
        self.grip_force = torch.zeros((self.num_envs,), dtype=torch.float32, device=device)
        self.grip_feedback_available = False
        self._debug_markers = None

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.phase[env_ids] = self.REST
        self.elapsed[env_ids] = 0.0
        self.last_reported_phase[env_ids] = -1
        self.plug_grasp_offset[env_ids] = self.plug_grasp_fallback_offset[env_ids]
        self.grip_command[env_ids] = _GRIPPER_INITIAL_GRASP_COMMAND
        self.grip_force[env_ids] = 0.0

    def compute(self, env) -> torch.Tensor:
        robot = env.scene["robot"]
        plug = env.scene["plug1"]
        ee_body_id = robot.find_bodies("right_gripper_base")[0][0]

        root_pose_w = robot.data.root_link_pose_w.torch
        root_pos_w = root_pose_w[:, :3]
        root_quat_w = root_pose_w[:, 3:]
        ee_base_pos_w = robot.data.body_pos_w.torch[:, ee_body_id]
        ee_base_quat_w = robot.data.body_quat_w.torch[:, ee_body_id]
        ee_pos_w, ee_quat_w = combine_frame_transforms(
            ee_base_pos_w,
            ee_base_quat_w,
            self.ee_offset_pos,
            self.ee_offset_quat,
        )
        rigid_plug_pose_w = plug.data.root_link_pose_w.torch
        rigid_plug_pos_w = rigid_plug_pose_w[:, :3]
        plug_pos_w, plug_quat_w = self._get_live_plug_frame(env, rigid_plug_pose_w)
        plug_grasp_offset = self.plug_grasp_offset
        grasp_pos_w = plug_pos_w + quat_apply(plug_quat_w, plug_grasp_offset)
        grasp_quat_w = plug_quat_w

        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        plug_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, plug_pos_w, plug_quat_w)
        grasp_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, grasp_pos_w, grasp_quat_w)
        socket_pos_w = self.socket_pos_w
        env_origins = getattr(env.scene, "env_origins", None)
        if env_origins is not None:
            socket_pos_w = socket_pos_w + env_origins.to(device=self.device, dtype=socket_pos_w.dtype)
        socket_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, socket_pos_w, None)
        socket_z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )
        insertion_dir_w = _normalize_vector(quat_apply(self.socket_quat_w, socket_z_axis))
        plug_tip_pos_w = plug_pos_w + quat_apply(plug_quat_w, self.plug_tip_offset)
        plug_tip_z_w = _normalize_vector(quat_apply(plug_quat_w, socket_z_axis))
        plug_tip_target_w = socket_pos_w + self.insertion_start_depth * insertion_dir_w
        plug_tip_error_w = plug_tip_pos_w - plug_tip_target_w
        plug_tip_axial_error = torch.sum(plug_tip_error_w * insertion_dir_w, dim=-1, keepdim=True) * insertion_dir_w
        plug_tip_lateral_error = plug_tip_error_w - plug_tip_axial_error
        plug_tip_lateral_norm = torch.linalg.vector_norm(plug_tip_lateral_error, dim=-1)
        plug_axis_cosine = torch.sum(plug_tip_z_w * insertion_dir_w, dim=-1)

        current_pose = torch.cat((ee_pos_b, ee_quat_b), dim=-1)
        current_pose_w = torch.cat((ee_pos_w, ee_quat_w), dim=-1)
        first_step = self.elapsed == 0.0
        if torch.any(first_step):
            self.phase_start_pose[first_step] = current_pose[first_step]
            self.phase_start_pose_w[first_step] = current_pose_w[first_step]

        target_pose = self.phase_start_pose.clone()
        gripper = torch.ones((self.num_envs, 1), device=self.device)

        def set_world_target(mask: torch.Tensor, pos_w: torch.Tensor, quat_w: torch.Tensor) -> None:
            pos_b, quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, pos_w, normalize(quat_w))
            target_pose[mask, :3] = pos_b[mask]
            target_pose[mask, 3:] = quat_b[mask]

        plug_grasp_quat_w = normalize(quat_mul(grasp_quat_w, self.grasp_orientation_offset))
        socket_grasp_quat_w = normalize(quat_mul(self.socket_quat_w, self.grasp_orientation_offset))
        grip_command = self._update_grip_command()

        approach = self.phase == self.APPROACH
        approach_pos_w = plug_pos_w + quat_apply(grasp_quat_w, plug_grasp_offset + self.approach_offset)
        set_world_target(approach, approach_pos_w, plug_grasp_quat_w)

        engage = self.phase == self.ENGAGE
        set_world_target(engage, grasp_pos_w + self.engage_offset, plug_grasp_quat_w)

        grasp = self.phase == self.GRASP
        grasp_duration = torch.clamp(self.durations[self.GRASP], min=1.0e-6)
        grasp_alpha = torch.clamp((self.elapsed / grasp_duration).unsqueeze(-1), 0.0, 1.0)
        grasp_alpha = grasp_alpha * grasp_alpha * (3.0 - 2.0 * grasp_alpha)
        gripper[grasp] = 1.0 + (grip_command[grasp] - 1.0) * grasp_alpha[grasp]

        hold_grasp = self.phase == self.HOLD_GRASP
        gripper[hold_grasp] = grip_command[hold_grasp]

        retract = self.phase == self.RETRACT
        retract_pos_w = self.phase_start_pose_w[:, :3] + quat_apply(plug_quat_w, self.retract_vector)
        set_world_target(retract, retract_pos_w, self.phase_start_pose_w[:, 3:])
        gripper[retract] = grip_command[retract]

        settle = self.phase == self.SETTLE
        gripper[settle] = grip_command[settle]

        approach_target = self.phase == self.APPROACH_TARGET
        socket_start_pos_w = socket_pos_w + self.insertion_start_depth * insertion_dir_w
        set_world_target(approach_target, socket_start_pos_w, socket_grasp_quat_w)
        gripper[approach_target] = grip_command[approach_target]

        align_axes = self.phase == self.ALIGN_AXES
        desired_plug_z_w = -insertion_dir_w
        align_correction_w = _correction_quat_between_vectors(
            plug_tip_z_w,
            desired_plug_z_w,
            self.align_orientation_gain,
        )
        align_quat_w = normalize(quat_mul(align_correction_w, ee_quat_w))
        set_world_target(align_axes, ee_pos_w, align_quat_w)
        gripper[align_axes] = grip_command[align_axes]

        verify_align = self.phase == self.VERIFY_ALIGN
        verify_pos_w = ee_pos_w - self.verify_lateral_gain * plug_tip_lateral_error
        set_world_target(verify_align, verify_pos_w, self.phase_start_pose_w[:, 3:])
        gripper[verify_align] = grip_command[verify_align]

        insert = self.phase == self.INSERT
        insert_elapsed = torch.clamp(self.elapsed, min=0.0, max=self.insert_duration)
        insert_alpha = torch.clamp((insert_elapsed / self.insert_duration).unsqueeze(-1), 0.0, 1.0)
        insert_alpha = insert_alpha * insert_alpha * (3.0 - 2.0 * insert_alpha)
        insert_depth = self.insertion_start_depth + insert_alpha * (self.insert_final_depth - self.insertion_start_depth)
        insert_depth_travel = insert_depth - self.insertion_start_depth
        self.insert_lateral_integral[insert] += plug_tip_lateral_error[insert] * self.step_dt
        insert_correction_w = _correction_quat_between_vectors(
            plug_tip_z_w,
            desired_plug_z_w,
            self.insert_orientation_gain,
        )
        insert_quat_w = normalize(quat_mul(insert_correction_w, self.phase_start_pose_w[:, 3:]))
        insert_pos_w = (
            self.phase_start_pose_w[:, :3]
            + insert_depth_travel * insertion_dir_w
            - self.insert_lateral_gain * plug_tip_lateral_error
            - self.insert_lateral_integral_gain * self.insert_lateral_integral
        )
        set_world_target(insert, insert_pos_w, insert_quat_w)
        gripper[insert] = grip_command[insert]

        release = self.phase == self.RELEASE
        release_duration = torch.clamp(self.durations[self.RELEASE], min=1.0e-6)
        release_alpha = torch.clamp((self.elapsed / release_duration).unsqueeze(-1), 0.0, 1.0)
        release_alpha = release_alpha * release_alpha * (3.0 - 2.0 * release_alpha)
        gripper[release] = grip_command[release] + (1.0 - grip_command[release]) * release_alpha[release]

        withdraw = self.phase == self.WITHDRAW
        withdraw_pos_w = self.phase_start_pose_w[:, :3] + self.withdraw_offset
        set_world_target(withdraw, withdraw_pos_w, self.phase_start_pose_w[:, 3:])
        gripper[withdraw] = 1.0

        done = self.phase == self.DONE
        gripper[done] = 1.0

        durations = self.durations[self.phase].unsqueeze(-1)
        blend = torch.clamp((self.elapsed / durations.squeeze(-1)).unsqueeze(-1), 0.0, 1.0)
        blend = blend * blend * (3.0 - 2.0 * blend)
        self.command_pose[:, :3] = self.phase_start_pose[:, :3] * (1.0 - blend) + target_pose[:, :3] * blend
        start_quat = self.phase_start_pose[:, 3:]
        target_quat = target_pose[:, 3:]
        target_quat = torch.where(torch.sum(start_quat * target_quat, dim=-1, keepdim=True) < 0.0, -target_quat, target_quat)
        self.command_pose[:, 3:] = normalize(start_quat * (1.0 - blend) + target_quat * blend)

        actions = torch.cat((self.command_pose, gripper), dim=-1)

        position_error = torch.abs(target_pose[:, :3] - current_pose[:, :3])
        rotation_error = quat_error_magnitude(target_pose[:, 3:], current_pose[:, 3:])

        if self.debug:
            left_finger_id = robot.find_bodies("right_gripper_leftfinger")[0][0]
            right_finger_id = robot.find_bodies("right_gripper_rightfinger")[0][0]
            left_finger_pos_w = robot.data.body_pos_w.torch[:, left_finger_id]
            right_finger_pos_w = robot.data.body_pos_w.torch[:, right_finger_id]
            finger_mid_pos_w = 0.5 * (left_finger_pos_w + right_finger_pos_w)
            finger_gap = torch.linalg.vector_norm(left_finger_pos_w - right_finger_pos_w, dim=-1)
            finger_mid_error = torch.linalg.vector_norm(finger_mid_pos_w - grasp_pos_w, dim=-1)
            self._update_debug_markers(env, plug_pos_w, rigid_plug_pos_w, grasp_pos_w, finger_mid_pos_w)

            changed = self.phase != self.last_reported_phase
            if bool(changed[0].item()):
                phase_name = self.PHASE_NAMES[int(self.phase[0].item())]
                print(
                    f"[waterhose_ik] {phase_name}: "
                    f"ee_b={ee_pos_b[0].detach().cpu().tolist()} "
                    f"plug_w={plug_pos_w[0].detach().cpu().tolist()} "
                    f"plug_rigid_w={rigid_plug_pos_w[0].detach().cpu().tolist()} "
                    f"plug_b={plug_pos_b[0].detach().cpu().tolist()} "
                    f"grasp_w={grasp_pos_w[0].detach().cpu().tolist()} "
                    f"grasp_b={grasp_pos_b[0].detach().cpu().tolist()} "
                    f"socket_b={socket_pos_b[0].detach().cpu().tolist()} "
                    f"target_b={target_pose[0, :3].detach().cpu().tolist()} "
                    f"pos_err={position_error[0].detach().cpu().tolist()} "
                    f"rot_err={float(rotation_error[0].detach().cpu()):.4f} "
                    f"tip_lateral={float(plug_tip_lateral_norm[0].detach().cpu()):.4f} "
                    f"tip_axis_cos={float(plug_axis_cosine[0].detach().cpu()):.4f} "
                    f"finger_gap={float(finger_gap[0].detach().cpu()):.4f} "
                    f"finger_mid_err={float(finger_mid_error[0].detach().cpu()):.4f} "
                    f"grip_cmd={float(grip_command[0].detach().cpu()):.3f} "
                    f"grip_force={float(self.grip_force[0].detach().cpu()):.2f}",
                    flush=True,
                )
            self.last_reported_phase[changed] = self.phase[changed]

        self.elapsed += self.step_dt
        phase_pos_tolerance = self.pos_tolerances[self.phase]
        phase_rot_tolerance = self.rot_tolerances[self.phase]
        pose_converged = torch.all(position_error < phase_pos_tolerance, dim=-1) & (
            rotation_error < phase_rot_tolerance
        )
        if self.grip_feedback_available:
            grip_ready = (self.grip_force >= _GRIPPER_FORCE_TARGET_N) | (
                self.grip_command.squeeze(-1) <= _GRIPPER_MAX_FORCE_CLOSE_COMMAND + 1.0e-5
            )
            grip_wait = (self.phase == self.GRASP) | (self.phase == self.HOLD_GRASP)
            pose_converged = pose_converged & (grip_wait.logical_not() | grip_ready)
        timed_out = self.elapsed >= self.durations[self.phase]
        should_advance = timed_out & pose_converged
        custom_advance = approach_target | align_axes | verify_align | insert
        should_advance &= custom_advance.logical_not()
        align_converged = (self.elapsed >= 0.5) & (plug_axis_cosine < -0.90)
        verify_converged = (
            (self.elapsed >= 0.5)
            & (plug_tip_lateral_norm < 0.010)
            & (plug_axis_cosine < -0.90)
        )
        should_advance |= approach_target & timed_out
        should_advance |= align_axes & (align_converged | timed_out)
        should_advance |= verify_align & (verify_converged | timed_out)
        should_advance |= insert & (self.elapsed >= self.insert_duration)
        should_advance &= self.phase < self.DONE
        if torch.any(should_advance):
            self.phase[should_advance] += 1
            self.elapsed[should_advance] = 0.0
            self.phase_start_pose[should_advance] = current_pose[should_advance]
            self.phase_start_pose_w[should_advance] = current_pose_w[should_advance]
            entered_insert = should_advance & (self.phase == self.INSERT)
            if torch.any(entered_insert):
                self.insert_lateral_integral[entered_insert] = 0.0

        return actions

    def _update_grip_command(self) -> torch.Tensor:
        """Tighten the grasp command until proxy feedback indicates a confident grip."""

        active = (self.phase >= self.GRASP) & (self.phase <= self.INSERT)
        if not torch.any(active):
            return self.grip_command

        proxy_force = self._get_right_proxy_grip_force()
        if proxy_force is None:
            self.grip_feedback_available = False
            fallback = torch.full_like(self.grip_command, _GRIPPER_FALLBACK_GRASP_COMMAND)
            self.grip_command[active] = torch.minimum(self.grip_command, fallback)[active]
            return self.grip_command

        self.grip_feedback_available = True
        self.grip_force[:] = proxy_force
        tighten = active & (proxy_force < _GRIPPER_FORCE_TARGET_N)
        if torch.any(tighten):
            next_command = self.grip_command[tighten] - _GRIPPER_TIGHTEN_RATE * self.step_dt
            self.grip_command[tighten] = torch.clamp(next_command, min=_GRIPPER_MAX_FORCE_CLOSE_COMMAND)
        return self.grip_command

    def _get_right_proxy_grip_force(self) -> torch.Tensor | None:
        """Return summed right-finger proxy feedback force magnitudes for each env."""

        body_ids = self._resolve_right_proxy_body_ids()
        if body_ids is None:
            return None
        try:
            import warp as wp
            from isaaclab_newton.physics import NewtonCoupledManager

            wrenches = NewtonCoupledManager.get_proxy_body_wrenches("mjc", "vbd")
            if wrenches is None:
                return None
            wrench_tensor = wp.to_torch(wrenches).to(device=self.device)
            if wrench_tensor.ndim != 2 or wrench_tensor.shape[-1] < 3:
                return None
            if int(torch.max(body_ids).item()) >= wrench_tensor.shape[0]:
                return None
            finger_wrenches = wrench_tensor[body_ids]
            finger_forces = finger_wrenches[..., :3]
            return torch.linalg.vector_norm(finger_forces, dim=-1).sum(dim=-1)
        except (AttributeError, ImportError, RuntimeError):
            return None

    def _get_live_plug_frame(self, env, rigid_plug_pose_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the live Plug1 body frame from the Newton solver state."""

        plug_body_ids = self._resolve_plug_body_ids()
        plug_pose_w = self._newton_body_poses(plug_body_ids)
        if plug_pose_w is not None:
            return plug_pose_w[:, :3], normalize(plug_pose_w[:, 3:])

        cable = None
        try:
            cable = env.scene["cable1"]
        except KeyError:
            pass

        if cable is not None:
            body_ids = self._resolve_plug_segment_body_ids(cable)
            if body_ids is not None:
                cable_pose_w = self._newton_body_poses(body_ids)
                if cable_pose_w is not None:
                    cable_pos_w = cable_pose_w[:, :3]
                    cable_quat_w = normalize(cable_pose_w[:, 3:])
                    plug_pos_w = cable_pos_w + quat_apply(cable_quat_w, self.plug_cable_local_pos)
                    return plug_pos_w, cable_quat_w

            try:
                cable_pose_w = cable.data.body_link_pose_w.torch[:, _CABLE1_PLUG_SEGMENT_ID]
                cable_pos_w = cable_pose_w[:, :3]
                cable_quat_w = normalize(cable_pose_w[:, 3:])
                plug_pos_w = cable_pos_w + quat_apply(cable_quat_w, self.plug_cable_local_pos)
                return plug_pos_w, cable_quat_w
            except (AttributeError, IndexError, RuntimeError):
                pass

        return rigid_plug_pose_w[:, :3], normalize(rigid_plug_pose_w[:, 3:])

    def _newton_body_poses(self, body_ids: torch.Tensor | None) -> torch.Tensor | None:
        """Read global Newton body poses for a vector of body ids."""

        if body_ids is None:
            return None
        try:
            import warp as wp
            from isaaclab_newton.physics import NewtonManager

            state = NewtonManager.get_state_0()
            if state is None or state.body_q is None:
                return None
            body_q = wp.to_torch(state.body_q).to(device=self.device)
            if int(torch.max(body_ids).item()) >= body_q.shape[0]:
                return None
            return body_q[body_ids]
        except (AttributeError, ImportError, RuntimeError):
            return None

    def _resolve_plug_body_ids(self) -> torch.Tensor | None:
        """Resolve global Newton body ids for the simulated Plug1 rigid body."""

        if self._plug_body_ids is not None:
            return self._plug_body_ids

        try:
            from isaaclab_newton.physics import NewtonManager

            model = NewtonManager.get_model()
            body_labels = [str(label) for label in getattr(model, "body_label", [])]
        except (AttributeError, ImportError, RuntimeError):
            return None

        body_ids = []
        for env_idx in range(self.num_envs):
            target = f"/World/envs/env_{env_idx}/Plug1"
            try:
                body_ids.append(body_labels.index(target))
            except ValueError:
                return None
        self._plug_body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)
        return self._plug_body_ids

    def _resolve_right_proxy_body_ids(self) -> torch.Tensor | None:
        """Resolve global Newton body ids for the right gripper proxy/source bodies."""

        if self._right_proxy_body_ids is not None:
            return self._right_proxy_body_ids

        try:
            from isaaclab_newton.physics import NewtonManager

            model = NewtonManager.get_model()
            body_labels = [str(label) for label in getattr(model, "body_label", [])]
        except (AttributeError, ImportError, RuntimeError):
            return None

        if not body_labels:
            return None

        body_ids = []
        finger_names = ("right_gripper_leftfinger", "right_gripper_rightfinger")
        for env_idx in range(self.num_envs):
            env_ids = []
            env_token = f"/World/envs/env_{env_idx}/"
            for finger_name in finger_names:
                matches = [
                    body_id
                    for body_id, label in enumerate(body_labels)
                    if env_token in label and (label == finger_name or label.endswith(f"/{finger_name}"))
                ]
                if not matches:
                    return None
                env_ids.append(matches[0])
            body_ids.append(env_ids)

        self._right_proxy_body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)
        return self._right_proxy_body_ids

    def _resolve_plug_segment_body_ids(self, cable) -> torch.Tensor | None:
        """Resolve global Newton body ids for the Cable1 segment welded to Plug1."""

        if self._plug_segment_body_ids is not None:
            return self._plug_segment_body_ids

        entry = getattr(cable, "_registry_entry", None)
        segment_body_indices = getattr(entry, "segment_body_indices", None)
        if not segment_body_indices or len(segment_body_indices) < self.num_envs:
            return None

        body_ids = []
        for env_idx in range(self.num_envs):
            segments = segment_body_indices[env_idx]
            if len(segments) <= _CABLE1_PLUG_SEGMENT_ID:
                return None
            body_ids.append(int(segments[_CABLE1_PLUG_SEGMENT_ID]))
        self._plug_segment_body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)
        return self._plug_segment_body_ids

    def _update_debug_markers(
        self,
        env,
        plug_pos_w: torch.Tensor,
        rigid_plug_pos_w: torch.Tensor,
        grasp_pos_w: torch.Tensor,
        finger_mid_pos_w: torch.Tensor,
    ) -> None:
        """Visualize the live plug attachment, rigid plug root, policy target, and finger midpoint."""

        if not self.debug:
            return
        visualizers = getattr(getattr(env, "sim", None), "visualizers", ()) or ()
        if not any(
            getattr(visualizer, "supports_markers", lambda: False)()
            and getattr(getattr(visualizer, "cfg", None), "enable_markers", True)
            for visualizer in visualizers
        ):
            return

        if self._debug_markers is None:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/WaterhoseScriptedDebug/plug_origin",
                markers={
                    "live_plug_attachment": sim_utils.SphereCfg(
                        radius=0.018,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                    "rigid_plug_root": sim_utils.SphereCfg(
                        radius=0.012,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                    "grasp_target": sim_utils.SphereCfg(
                        radius=0.014,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.25, 1.0)),
                    ),
                    "finger_midpoint": sim_utils.SphereCfg(
                        radius=0.010,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0)),
                    ),
                },
            )
            self._debug_markers = VisualizationMarkers(marker_cfg)

        translations = torch.cat((plug_pos_w, rigid_plug_pos_w, grasp_pos_w, finger_mid_pos_w), dim=0)
        marker_indices = torch.cat(
            (
                torch.zeros(self.num_envs, dtype=torch.int32, device=self.device),
                torch.ones(self.num_envs, dtype=torch.int32, device=self.device),
                torch.full((self.num_envs,), 2, dtype=torch.int32, device=self.device),
                torch.full((self.num_envs,), 3, dtype=torch.int32, device=self.device),
            ),
            dim=0,
        )
        self._debug_markers.visualize(translations=translations, marker_indices=marker_indices)


def create_scripted_policy(env, *, settle_time: float = 4.0, debug: bool = False) -> WaterhoseDemoState:
    """Create the task-local scripted policy used by the demo launcher."""

    return WaterhoseDemoState(
        env.num_envs,
        env.step_dt,
        env.device,
        settle_time,
        debug,
    )
