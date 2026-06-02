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

# Center of the grasp contact region in right_gripper_base local frame.
_RIGHT_EE_FROM_BASE_POS = (0.0, 0.0, -0.1055)
# USD stores xformOp:orient as (w, x, y, z); IsaacLab frame math uses (x, y, z, w).
_RIGHT_EE_FROM_BASE_QUAT = (0.70710677, 0.70710677, 0.0, 0.0)
_RIGHT_FINGER_PAD_LOCAL_POS = (0.0015, 0.0, -0.0305)
_PLUG_TIP_OFFSET = (0.0, 0.0, -0.014106234)
_CABLE_RADIUS = 0.003
_GRASP_SHIFT = 0.01
_PLUG_GRASP_OFFSET = (0.0, -_CABLE_RADIUS + 0.002, _GRASP_SHIFT)
_CABLE1_PLUG_SEGMENT_ID = 0
_CABLE1_PLUG_LOCAL_POS = (0.0, 0.0, 0.022)
_GRIPPER_INITIAL_GRASP_COMMAND = -0.72
_GRIPPER_PREGRASP_COMMAND = -0.80
_GRIPPER_FALLBACK_GRASP_COMMAND = -1.0
_GRIPPER_MAX_FORCE_CLOSE_COMMAND = -1.0
_GRIPPER_FORCE_TARGET_N = 80.0
_GRIPPER_LOCK_FORCE_N = 80.0
_GRIPPER_TIGHTEN_RATE = 0.30
_GRIPPER_CENTERING_K = 0.4
_GRIPPER_CENTERING_MAX_STEP = 0.002


def _normalize_vector(value: torch.Tensor) -> torch.Tensor:
    return value / torch.clamp(torch.linalg.vector_norm(value, dim=-1, keepdim=True), min=1.0e-8)


def _clamp_vector_norm(value: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    scale = torch.clamp(float(max_norm) / torch.clamp(norm, min=1.0e-8), max=1.0)
    return value * scale


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
    ALIGN_AXES = 7
    APPROACH_TARGET = 8
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
        "ALIGN_AXES",
        "APPROACH_TARGET",
        "VERIFY_ALIGN",
        "INSERT",
        "RELEASE",
        "WITHDRAW",
        "DONE",
    )
    DURATIONS = (0.25, 3.0, 1.5, 0.5, 0.5, 1.5, 0.3, 5.0, 5.0, 2.0, 5.0, 1.0, 2.0, 1.0e6)

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
        self.phase_plug_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=device)
        self.phase_plug_quat_w = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=device)
        self.phase_plug_quat_w[:, 3] = 1.0
        self.phase_grasp_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=device)
        self.phase_grasp_quat_w = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=device)
        self.phase_grasp_quat_w[:, 3] = 1.0
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
        self.turn_orientation_gain = 0.7
        self.align_orientation_gain = 0.12
        self.insert_orientation_gain = 0.2
        self.verify_lateral_gain = 1.0
        self.insert_axial_gain = 0.8
        self.insert_lateral_gain = 0.5
        self.insert_lateral_integral_gain = 5.0
        self.verify_lateral_correction_limit = 0.010
        self.insert_axial_correction_limit = 0.015
        self.insert_lateral_correction_limit = 0.012
        self.insert_lateral_integral_limit = 0.002
        self.insert_lateral_integral = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=device)
        self.insert_ee_start_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=device)
        self.insert_ee_quat_w = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=device)
        self.insert_ee_quat_w[:, 3] = 1.0
        self.insert_t_paused = torch.zeros((self.num_envs,), dtype=torch.float32, device=device)
        self.insert_depth_paused = torch.zeros((self.num_envs,), dtype=torch.bool, device=device)
        self.insert_cos_pause_threshold = -0.95
        self.insert_cos_resume_threshold = -0.97
        # One-shot turn target that aligns the plug tip with the socket axis on ALIGN_AXES entry.
        self.align_target_quat_w = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=device)
        self.align_target_quat_w[:, 3] = 1.0
        self.align_ee_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=device)
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
        self.finger_pad_local_pos = torch.tensor(
            _RIGHT_FINGER_PAD_LOCAL_POS,
            dtype=torch.float32,
            device=device,
        ).repeat(self.num_envs, 1)
        self._plug_segment_body_ids = None
        self._tail_segment_body_ids = None
        self._plug_body_ids = None
        self._plug_body_ids_done = False
        self._right_proxy_body_ids = None
        self._ee_body_id = None
        self._left_finger_body_id = None
        self._right_finger_body_id = None
        self.grip_command = torch.full(
            (self.num_envs, 1), _GRIPPER_INITIAL_GRASP_COMMAND, dtype=torch.float32, device=device
        )
        self.grip_locked = torch.zeros((self.num_envs,), dtype=torch.bool, device=device)
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
        self.grip_locked[env_ids] = False
        self.grip_force[env_ids] = 0.0
        self.phase_plug_pos_w[env_ids] = 0.0
        self.phase_plug_quat_w[env_ids] = 0.0
        self.phase_plug_quat_w[env_ids, 3] = 1.0
        self.phase_grasp_pos_w[env_ids] = 0.0
        self.phase_grasp_quat_w[env_ids] = 0.0
        self.phase_grasp_quat_w[env_ids, 3] = 1.0
        self.insert_lateral_integral[env_ids] = 0.0
        self.insert_ee_start_pos_w[env_ids] = 0.0
        self.insert_ee_quat_w[env_ids] = 0.0
        self.insert_ee_quat_w[env_ids, 3] = 1.0
        self.insert_t_paused[env_ids] = 0.0
        self.insert_depth_paused[env_ids] = False
        self.align_target_quat_w[env_ids] = 0.0
        self.align_target_quat_w[env_ids, 3] = 1.0
        self.align_ee_pos_w[env_ids] = 0.0

    def compute(self, env) -> torch.Tensor:
        robot = env.scene["robot"]
        try:
            plug = env.scene["plug1"]
        except KeyError:
            plug = None  # cable-only debug: no plug spawned; fall back to the live cable frame
        if self._ee_body_id is None:
            self._ee_body_id = robot.find_bodies("right_gripper_base")[0][0]
            self._left_finger_body_id = robot.find_bodies("right_gripper_leftfinger")[0][0]
            self._right_finger_body_id = robot.find_bodies("right_gripper_rightfinger")[0][0]
        ee_body_id = self._ee_body_id

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
        if plug is not None:
            rigid_plug_pose_w = plug.data.root_link_pose_w.torch
        else:
            rigid_plug_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
            rigid_plug_pose_w[..., 6] = 1.0
        rigid_plug_pos_w = rigid_plug_pose_w[:, :3]
        plug_pos_w, plug_quat_w = self._get_live_plug_frame(env, rigid_plug_pose_w)
        first_step = self.elapsed == 0.0
        self._update_plug_grasp_offset(env, plug_pos_w, plug_quat_w, first_step & (self.phase <= self.ENGAGE))
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
        plug_tip_pos_w, plug_tip_quat_w = self._get_live_tip_frame(env, plug_pos_w, plug_quat_w)
        plug_tip_z_w = _normalize_vector(quat_apply(plug_tip_quat_w, socket_z_axis))
        plug_tip_target_w = socket_pos_w + self.insertion_start_depth * insertion_dir_w
        plug_tip_error_w = plug_tip_pos_w - plug_tip_target_w
        plug_tip_axial_error = torch.sum(plug_tip_error_w * insertion_dir_w, dim=-1, keepdim=True) * insertion_dir_w
        plug_tip_lateral_error = plug_tip_error_w - plug_tip_axial_error
        plug_tip_lateral_norm = torch.linalg.vector_norm(plug_tip_lateral_error, dim=-1)
        plug_axis_cosine = torch.sum(plug_tip_z_w * insertion_dir_w, dim=-1)
        desired_plug_z_w = -insertion_dir_w

        current_pose = torch.cat((ee_pos_b, ee_quat_b), dim=-1)
        current_pose_w = torch.cat((ee_pos_w, ee_quat_w), dim=-1)
        # Branch-free: masked writes are no-ops when ``first_step`` is empty, so we avoid the
        # ``torch.any(...)`` host sync that would otherwise stall the step on the GPU.
        self.phase_start_pose[first_step] = current_pose[first_step]
        self.phase_start_pose_w[first_step] = current_pose_w[first_step]

        target_pose = self.phase_start_pose.clone()
        gripper = torch.ones((self.num_envs, 1), device=self.device)

        def set_world_target(mask: torch.Tensor, pos_w: torch.Tensor, quat_w: torch.Tensor) -> None:
            pos_b, quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, pos_w, normalize(quat_w))
            target_pose[mask, :3] = pos_b[mask]
            target_pose[mask, 3:] = quat_b[mask]

        plug_grasp_quat_w = normalize(quat_mul(grasp_quat_w, self.grasp_orientation_offset))
        # Branch-free phase-entry bookkeeping (masked writes; no host sync).
        self.phase_plug_pos_w[first_step] = plug_pos_w[first_step]
        self.phase_plug_quat_w[first_step] = grasp_quat_w[first_step]
        self.phase_grasp_pos_w[first_step] = grasp_pos_w[first_step]
        self.phase_grasp_quat_w[first_step] = plug_grasp_quat_w[first_step]
        entered_align = first_step & (self.phase == self.ALIGN_AXES)
        self._init_align_axes_state(entered_align, ee_pos_w, ee_quat_w, plug_tip_z_w, desired_plug_z_w)
        entered_insert = first_step & (self.phase == self.INSERT)
        self.insert_ee_start_pos_w[entered_insert] = ee_pos_w[entered_insert]
        self.insert_ee_quat_w[entered_insert] = ee_quat_w[entered_insert]
        self.insert_lateral_integral[entered_insert] = 0.0
        self.insert_t_paused[entered_insert] = 0.0
        self.insert_depth_paused[entered_insert] = False
        grip_command = self._update_grip_command()

        approach = self.phase == self.APPROACH
        approach_pos_w = self.phase_plug_pos_w + quat_apply(
            self.phase_plug_quat_w,
            plug_grasp_offset + self.approach_offset,
        )
        set_world_target(approach, approach_pos_w, self.phase_grasp_quat_w)

        engage = self.phase == self.ENGAGE
        gripper[engage] = 1.0
        set_world_target(engage, self.phase_grasp_pos_w + self.engage_offset, self.phase_grasp_quat_w)

        grasp = self.phase == self.GRASP
        grasp_duration = torch.clamp(self.durations[self.GRASP], min=1.0e-6)
        grasp_alpha = torch.clamp((self.elapsed / grasp_duration).unsqueeze(-1), 0.0, 1.0)
        grasp_alpha = grasp_alpha * grasp_alpha * (3.0 - 2.0 * grasp_alpha)
        gripper[grasp] = 1.0 + (grip_command[grasp] - 1.0) * grasp_alpha[grasp]

        hold_grasp = self.phase == self.HOLD_GRASP
        gripper[hold_grasp] = grip_command[hold_grasp]

        retract = self.phase == self.RETRACT
        retract_pos_w = self.phase_start_pose_w[:, :3] + quat_apply(self.phase_plug_quat_w, self.retract_vector)
        set_world_target(retract, retract_pos_w, self.phase_start_pose_w[:, 3:])
        gripper[retract] = grip_command[retract]

        settle = self.phase == self.SETTLE
        set_world_target(settle, self.phase_start_pose_w[:, :3], self.phase_start_pose_w[:, 3:])
        gripper[settle] = grip_command[settle]

        approach_target = self.phase == self.APPROACH_TARGET
        socket_start_pos_w = socket_pos_w + self.insertion_start_depth * insertion_dir_w
        # The plug is aligned while still clear of the socket, then translated in with that
        # measured aligned orientation preserved. Driving a twisted plug into the socket proxy
        # first tends to knock it out of the fingers.
        set_world_target(approach_target, socket_start_pos_w, self.phase_start_pose_w[:, 3:])
        gripper[approach_target] = grip_command[approach_target]

        lost_grip = self.grip_feedback_available & (self.phase >= self.APPROACH_TARGET) & (self.phase <= self.INSERT) & (
            self.grip_force <= 5.0
        )

        align_axes = (self.phase == self.ALIGN_AXES) & lost_grip.logical_not()
        set_world_target(align_axes, self.align_ee_pos_w, self.align_target_quat_w)
        gripper[align_axes] = grip_command[align_axes]

        verify_align = (self.phase == self.VERIFY_ALIGN) & lost_grip.logical_not()
        verify_lateral_correction = _clamp_vector_norm(
            self.verify_lateral_gain * plug_tip_lateral_error,
            self.verify_lateral_correction_limit,
        )
        verify_pos_w = ee_pos_w - verify_lateral_correction
        set_world_target(verify_align, verify_pos_w, self.phase_start_pose_w[:, 3:])
        gripper[verify_align] = grip_command[verify_align]

        insert = (self.phase == self.INSERT) & lost_grip.logical_not()
        insert_correction_w = _correction_quat_between_vectors(
            plug_tip_z_w,
            desired_plug_z_w,
            self.insert_orientation_gain,
        )
        self.insert_ee_quat_w[insert] = normalize(quat_mul(insert_correction_w, self.insert_ee_quat_w))[insert]
        pause_now = insert & (plug_axis_cosine > self.insert_cos_pause_threshold)
        resume_now = insert & self.insert_depth_paused & (plug_axis_cosine < self.insert_cos_resume_threshold)
        self.insert_depth_paused[pause_now] = True
        self.insert_depth_paused[resume_now] = False
        self.insert_t_paused[insert & self.insert_depth_paused] += self.step_dt
        insert_elapsed = torch.clamp(self.elapsed - self.insert_t_paused, min=0.0, max=self.insert_duration)
        insert_alpha = torch.clamp((insert_elapsed / self.insert_duration).unsqueeze(-1), 0.0, 1.0)
        insert_alpha = insert_alpha * insert_alpha * (3.0 - 2.0 * insert_alpha)
        insert_depth = self.insertion_start_depth + insert_alpha * (self.insert_final_depth - self.insertion_start_depth)
        insert_depth_travel = insert_depth - self.insertion_start_depth
        insert_tip_target_w = socket_pos_w + insert_depth * insertion_dir_w
        insert_tip_error_w = plug_tip_pos_w - insert_tip_target_w
        insert_tip_axial_error = (
            torch.sum(insert_tip_error_w * insertion_dir_w, dim=-1, keepdim=True) * insertion_dir_w
        )
        insert_axial_correction = _clamp_vector_norm(
            self.insert_axial_gain * insert_tip_axial_error,
            self.insert_axial_correction_limit,
        )
        self.insert_lateral_integral[insert] += plug_tip_lateral_error[insert] * self.step_dt
        self.insert_lateral_integral[insert] = _clamp_vector_norm(
            self.insert_lateral_integral,
            self.insert_lateral_integral_limit,
        )[insert]
        insert_lateral_correction = _clamp_vector_norm(
            self.insert_lateral_gain * plug_tip_lateral_error
            + self.insert_lateral_integral_gain * self.insert_lateral_integral,
            self.insert_lateral_correction_limit,
        )
        insert_pos_w = (
            self.insert_ee_start_pos_w
            + insert_depth_travel * insertion_dir_w
            - insert_axial_correction
            - insert_lateral_correction
        )
        set_world_target(insert, insert_pos_w, self.insert_ee_quat_w)
        gripper[insert] = grip_command[insert]

        target_pose[lost_grip] = current_pose[lost_grip]
        gripper[lost_grip] = grip_command[lost_grip]

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
        direct_control = verify_align | insert
        self.command_pose[direct_control] = target_pose[direct_control]

        # Finger-pad frames (cached body ids, GPU-only reads). Computed once per step and reused
        # by both the centering nudge and the optional debug block.
        left_finger_pos_w = robot.data.body_pos_w.torch[:, self._left_finger_body_id]
        right_finger_pos_w = robot.data.body_pos_w.torch[:, self._right_finger_body_id]
        left_finger_quat_w = robot.data.body_quat_w.torch[:, self._left_finger_body_id]
        right_finger_quat_w = robot.data.body_quat_w.torch[:, self._right_finger_body_id]
        left_pad_pos_w = left_finger_pos_w + quat_apply(left_finger_quat_w, self.finger_pad_local_pos)
        right_pad_pos_w = right_finger_pos_w + quat_apply(right_finger_quat_w, self.finger_pad_local_pos)
        finger_mid_pos_w = 0.5 * (left_pad_pos_w + right_pad_pos_w)

        # Branch-free centering: nudge the EE so the finger midpoint tracks the grasp point.
        # The masked write affects only ENGAGE..ALIGN_AXES envs, so no host sync is needed.
        centering_active = (self.phase >= self.ENGAGE) & (self.phase <= self.ALIGN_AXES) & lost_grip.logical_not()
        centering_delta_b = quat_apply(
            quat_inv(root_quat_w),
            _clamp_vector_norm(_GRIPPER_CENTERING_K * (grasp_pos_w - finger_mid_pos_w), _GRIPPER_CENTERING_MAX_STEP),
        )
        self.command_pose[centering_active, :3] += centering_delta_b[centering_active]

        actions = torch.cat((self.command_pose, gripper), dim=-1)

        position_error = torch.abs(target_pose[:, :3] - current_pose[:, :3])
        rotation_error = quat_error_magnitude(target_pose[:, 3:], current_pose[:, 3:])

        if self.debug:
            finger_gap = torch.linalg.vector_norm(left_pad_pos_w - right_pad_pos_w, dim=-1)
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
            grip_ready = self.grip_locked | (self.grip_force >= _GRIPPER_LOCK_FORCE_N) | (
                self.grip_command.squeeze(-1) <= _GRIPPER_MAX_FORCE_CLOSE_COMMAND + 1.0e-5
            )
            grip_wait = (self.phase == self.GRASP) | (self.phase == self.HOLD_GRASP) | (self.phase == self.SETTLE)
            pose_converged = pose_converged & (grip_wait.logical_not() | grip_ready)
        timed_out = self.elapsed >= self.durations[self.phase]
        should_advance = timed_out & pose_converged
        custom_advance = (
            retract
            | approach_target
            | (self.phase == self.ALIGN_AXES)
            | (self.phase == self.VERIFY_ALIGN)
            | (self.phase == self.INSERT)
        )
        should_advance &= custom_advance.logical_not()
        align_converged = (self.elapsed >= 0.5) & (plug_axis_cosine < -0.90)
        verify_converged = (
            (self.elapsed >= 0.5)
            & (plug_tip_lateral_norm < 0.010)
            & (plug_axis_cosine < -0.90)
        )
        should_advance |= retract & timed_out
        should_advance |= approach_target & timed_out
        grasp_still_held = self.grip_force > 5.0 if self.grip_feedback_available else torch.ones_like(timed_out, dtype=torch.bool)
        should_advance |= align_axes & grasp_still_held & (align_converged | timed_out)
        should_advance |= verify_align & grasp_still_held & (verify_converged | timed_out)
        should_advance |= insert & (self.elapsed >= self.insert_duration)
        should_advance &= lost_grip.logical_not()
        should_advance &= self.phase < self.DONE
        # Branch-free phase advance (masked writes; no host sync).
        self.phase[should_advance] += 1
        self.elapsed[should_advance] = 0.0
        self.phase_start_pose[should_advance] = current_pose[should_advance]
        self.phase_start_pose_w[should_advance] = current_pose_w[should_advance]
        entered_insert = should_advance & (self.phase == self.INSERT)
        self.insert_lateral_integral[entered_insert] = 0.0

        return actions

    def _update_grip_command(self) -> torch.Tensor:
        """Tighten only during capture, then hold the locked command through insertion.

        Branch-free over envs: the proxy grip force is read every step (a cheap GPU-only
        gather) and every command update is masked, so there is no per-step host sync. Updates
        still only affect the GRASP/HOLD_GRASP/INSERT envs via the masks below.
        """

        tighten_active = ((self.phase == self.GRASP) | (self.phase == self.HOLD_GRASP)) & self.grip_locked.logical_not()

        proxy_force = self._get_right_proxy_grip_force()
        if proxy_force is None:
            self.grip_feedback_available = False
            fallback = torch.full_like(self.grip_command, _GRIPPER_FALLBACK_GRASP_COMMAND)
            self.grip_command[tighten_active] = torch.minimum(self.grip_command, fallback)[tighten_active]
            return self.grip_command

        self.grip_feedback_available = True
        self.grip_force[:] = proxy_force
        hold_active = (self.phase >= self.GRASP) & (self.phase <= self.INSERT)
        at_close_limit = self.grip_command.squeeze(-1) <= _GRIPPER_MAX_FORCE_CLOSE_COMMAND + 1.0e-5
        self.grip_locked[hold_active & ((proxy_force >= _GRIPPER_LOCK_FORCE_N) | at_close_limit)] = True
        tighten = tighten_active & self.grip_locked.logical_not() & (proxy_force < _GRIPPER_FORCE_TARGET_N)
        next_command = torch.clamp(
            self.grip_command - _GRIPPER_TIGHTEN_RATE * self.step_dt, min=_GRIPPER_MAX_FORCE_CLOSE_COMMAND
        )
        self.grip_command[tighten] = next_command[tighten]
        self.grip_locked[tighten & (self.grip_command.squeeze(-1) <= _GRIPPER_MAX_FORCE_CLOSE_COMMAND + 1.0e-5)] = True
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
            finger_forces = wrench_tensor[body_ids][..., :3]
            return torch.linalg.vector_norm(finger_forces, dim=-1).sum(dim=-1)
        except (AttributeError, ImportError, IndexError, RuntimeError):
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

    def _get_live_tip_frame(
        self,
        env,
        plug_pos_w: torch.Tensor,
        plug_quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the live plug insertion tip.

        Cable segment 0's Newton body origin is not the connector tip in this
        rig: the Plug1 weld uses cable_local_pos=(0, 0, 0.022).  Use the actual
        plug frame plus the plug mesh's local -Z tip offset for insertion
        lateral/depth checks.
        """

        return plug_pos_w + quat_apply(plug_quat_w, self.plug_tip_offset), plug_quat_w

    def _update_plug_grasp_offset(
        self,
        env,
        plug_pos_w: torch.Tensor,
        plug_quat_w: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Match the stable demo: grasp from the side and shift 1 cm toward the cable body.

        Branch-free: the offset write is masked, so this can run every step without a host
        sync; only ``first_step`` envs in the early phases actually update.
        """

        try:
            cable = env.scene["cable1"]
        except KeyError:
            return

        tail_ids = self._resolve_tail_segment_body_ids(cable)
        tail_pos_w = None
        if tail_ids is not None:
            tail_pose_w = self._newton_body_poses(tail_ids)
            if tail_pose_w is not None:
                tail_pos_w = tail_pose_w[:, :3]

        if tail_pos_w is None:
            try:
                tail_pos_w = cable.data.body_link_pose_w.torch[:, -1, :3]
            except (AttributeError, IndexError, RuntimeError):
                return

        toward_cable_w = _normalize_vector(tail_pos_w - plug_pos_w)
        toward_cable_local = quat_apply(quat_inv(plug_quat_w), toward_cable_w)
        offset = toward_cable_local * _GRASP_SHIFT
        offset[:, 1] += -_CABLE_RADIUS + 0.002
        self.plug_grasp_offset[mask] = offset[mask]

    def _init_align_axes_state(
        self,
        mask: torch.Tensor,
        ee_pos_w: torch.Tensor,
        ee_quat_w: torch.Tensor,
        plug_tip_z_w: torch.Tensor,
        desired_plug_z_w: torch.Tensor,
    ) -> None:
        """Capture the one-shot turn target that aligns the plug tip with the socket axis."""

        turn_correction_w = _correction_quat_between_vectors(plug_tip_z_w, desired_plug_z_w, 1.0)
        turn_target_quat_w = normalize(quat_mul(turn_correction_w, ee_quat_w))
        self.align_ee_pos_w[mask] = ee_pos_w[mask]
        self.align_target_quat_w[mask] = turn_target_quat_w[mask]

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
            return body_q[body_ids]
        except (AttributeError, ImportError, IndexError, RuntimeError):
            return None

    def _resolve_plug_body_ids(self) -> torch.Tensor | None:
        """Resolve global Newton body ids for the simulated Plug1 rigid body.

        Caches the result once the model is queryable -- including a negative result when Plug1
        is absent (cable-only scenes) -- so the body-label scan does not repeat every step.
        """

        if self._plug_body_ids_done:
            return self._plug_body_ids

        try:
            from isaaclab_newton.physics import NewtonManager

            model = NewtonManager.get_model()
            body_labels = [str(label) for label in getattr(model, "body_label", [])]
        except (AttributeError, ImportError, RuntimeError):
            return None
        if not body_labels:
            return None

        body_ids = []
        for env_idx in range(self.num_envs):
            target = f"/World/envs/env_{env_idx}/Plug1"
            matches = [
                body_id
                for body_id, label in enumerate(body_labels)
                if label == target or label.startswith(f"{target}/")
            ]
            if not matches:
                self._plug_body_ids = None
                self._plug_body_ids_done = True
                return None
            body_ids.append(matches[0])
        self._plug_body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)
        self._plug_body_ids_done = True
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

    def _resolve_tail_segment_body_ids(self, cable) -> torch.Tensor | None:
        """Resolve global Newton body ids for Cable1's tail segment."""

        if self._tail_segment_body_ids is not None:
            return self._tail_segment_body_ids

        entry = getattr(cable, "_registry_entry", None)
        segment_body_indices = getattr(entry, "segment_body_indices", None)
        if not segment_body_indices or len(segment_body_indices) < self.num_envs:
            return None

        body_ids = []
        for env_idx in range(self.num_envs):
            segments = segment_body_indices[env_idx]
            if not segments:
                return None
            body_ids.append(int(segments[-1]))
        self._tail_segment_body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)
        return self._tail_segment_body_ids

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
