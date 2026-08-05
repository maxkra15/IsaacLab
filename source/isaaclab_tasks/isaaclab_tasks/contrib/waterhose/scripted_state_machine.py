# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted IK state machine for the RBY1 waterhose grasp-and-insert demo.

Phases: ``REST -> APPROACH -> ENGAGE -> GRASP -> HOLD_GRASP -> RETRACT -> SETTLE ->
LIFT -> CARRY -> ALIGN -> INSERT -> HOLD_INSERTED -> DONE``.

Each phase has a fixed minimum *duration* and an end-effector *target pose*. The commanded pose is a
smoothstep blend from the entry pose to the target. Free-space phases advance after the end effector
converges, while insertion phases advance from the measured connector geometry because contact
compliance can keep the hand away from its unconstrained target. The terminal state holds the
inserted pose and closed grasp; the scripted demo does not release or retreat.

The output is the registered task's 22-D action vector:
``[right_ee pose(7), left_hold pose(7), torso_hold pose(7), gripper(1)]``. Positions are expressed in
the robot root frame and quaternions use ``(x, y, z, w)`` order. The two hold blocks pin the torso and
idle left gripper.
"""

from __future__ import annotations

import torch
import warp as wp

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

from .geometry import (
    CONNECTOR_TIP_LOCAL_POS,
    PLUG_GRASP_OFFSET,
    RIGHT_GRIPPER_EE_FRAME_POS,
    RIGHT_GRIPPER_EE_FRAME_QUAT_XYZW,
    SOCKET_ALIGN_TIP_DEPTH,
    SOCKET_MOUTH_POS,
    SOCKET_RETAINED_AXIS_COS,
    SOCKET_RETAINED_DEPTH_TOLERANCE,
    SOCKET_RETAINED_RADIAL_TOLERANCE,
    SOCKET_ROT_QUAT_XYZW,
    SOCKET_SEATED_TIP_DEPTH,
)
from .mdp.terminations import plug_inserted_in_socket

# Gripper command convention used by the IK action term: +1 fully open, -1 fully closed.
_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSED = -1.0
# Maximum change in the measured connector-tip offset in the end-effector frame while the plug is
# expected to be grasped [m]. A retained plug can settle between the compliant fingers by about
# 1 cm; a missed grasp separates by several centimetres as soon as RETRACT/LIFT starts.
_GRASP_FOLLOW_TOLERANCE = 0.04

_REST = 0
_APPROACH = 1
_ENGAGE = 2
_GRASP = 3
_HOLD_GRASP = 4
_RETRACT = 5
_SETTLE = 6
_LIFT = 7
_CARRY = 8
_ALIGN = 9
_INSERT = 10
_HOLD_INSERTED = 11
_DONE = 12


def connector_retained_mask(env) -> torch.Tensor:
    """Return one boolean per environment indicating a physically retained connector.

    This uses the connector's measured Newton pose rather than the end-effector command. It is
    intentionally side-effect free so both the controller and demo runner can use the same
    terminal criterion without changing contact or solver state.
    """

    return plug_inserted_in_socket(
        env,
        socket_pos=SOCKET_MOUTH_POS,
        socket_quat=SOCKET_ROT_QUAT_XYZW,
        connector_tip_offset=CONNECTOR_TIP_LOCAL_POS,
        radial_threshold=SOCKET_RETAINED_RADIAL_TOLERANCE,
        min_depth=SOCKET_SEATED_TIP_DEPTH - SOCKET_RETAINED_DEPTH_TOLERANCE,
        max_depth=SOCKET_SEATED_TIP_DEPTH + SOCKET_RETAINED_DEPTH_TOLERANCE,
        alignment_threshold=SOCKET_RETAINED_AXIS_COS,
    )


@wp.func
def _smoothstep_wp(value: float) -> float:
    value = wp.clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


@wp.func
def _rotation_error_angle_wp(target: wp.quat, current: wp.quat) -> float:
    error = wp.normalize(target * wp.quat_inverse(current))
    return 2.0 * wp.acos(wp.clamp(wp.abs(error[3]), 0.0, 1.0))


@wp.func
def _quat_from_two_vectors_wp(source: wp.vec3, target: wp.vec3) -> wp.quat:
    """Warp equivalent of the eager state machine's ``_quat_from_two_vectors``."""

    source = wp.normalize(source)
    target = wp.normalize(target)
    cross = wp.cross(source, target)
    cross_norm = wp.length(cross)
    dot = wp.clamp(wp.dot(source, target), -1.0, 1.0)

    fallback = wp.cross(source, wp.vec3(1.0, 0.0, 0.0))
    if wp.length(fallback) <= 1.0e-6:
        fallback = wp.cross(source, wp.vec3(0.0, 1.0, 0.0))
    fallback = wp.normalize(fallback)

    axis = fallback
    if cross_norm > 1.0e-6:
        axis = cross / cross_norm
    angle = wp.atan2(cross_norm, dot)
    if cross_norm <= 1.0e-6 and dot < 0.0:
        angle = wp.pi
    return wp.normalize(wp.quat_from_axis_angle(axis, angle))


@wp.kernel(enable_backward=False)
def _capture_hold_targets_wp(
    robot_body_q: wp.array2d(dtype=wp.transform),
    robot_root_q: wp.array(dtype=wp.transform),
    left_body: int,
    torso_body: int,
    left_position: wp.array(dtype=wp.vec3),
    left_rotation: wp.array(dtype=wp.vec4),
    torso_position: wp.array(dtype=wp.vec3),
    torso_rotation: wp.array(dtype=wp.vec4),
):
    env_id = wp.tid()
    prototype_root = robot_root_q[0]
    env_root_inv = wp.transform_inverse(robot_root_q[env_id])

    left_target = prototype_root * (env_root_inv * robot_body_q[env_id, left_body])
    left_pos = wp.transform_get_translation(left_target)
    left_rot = wp.transform_get_rotation(left_target)
    left_position[env_id] = left_pos
    left_rotation[env_id] = wp.vec4(left_rot[0], left_rot[1], left_rot[2], left_rot[3])

    torso_target = prototype_root * (env_root_inv * robot_body_q[env_id, torso_body])
    torso_pos = wp.transform_get_translation(torso_target)
    torso_rot = wp.transform_get_rotation(torso_target)
    torso_position[env_id] = torso_pos
    torso_rotation[env_id] = wp.vec4(torso_rot[0], torso_rot[1], torso_rot[2], torso_rot[3])


@wp.kernel(enable_backward=False)
def _update_state_machine_wp(
    robot_body_q: wp.array2d(dtype=wp.transform),
    robot_root_q: wp.array(dtype=wp.transform),
    cable_body_q: wp.array(dtype=wp.transform),
    cable_head_bodies: wp.array(dtype=wp.int32),
    env_origins: wp.array(dtype=wp.vec3),
    phase: wp.array(dtype=wp.int32),
    elapsed: wp.array(dtype=wp.float32),
    durations: wp.array(dtype=wp.float32),
    phase_ee: wp.array(dtype=wp.transform),
    phase_connector: wp.array(dtype=wp.transform),
    frozen_tip_offset: wp.array(dtype=wp.vec3),
    frozen_insert_rotation: wp.array(dtype=wp.quat),
    grasp_reference_valid: wp.array(dtype=wp.bool),
    right_ee_body: int,
    ee_local_xform: wp.transform,
    connector_local_xform: wp.transform,
    socket_position: wp.vec3,
    socket_rotation: wp.quat,
    plug_grasp_offset: wp.vec3,
    connector_tip_local_pos: wp.vec3,
    grasp_orientation_offset: wp.quat,
    static_tip_offset: wp.vec3,
    frame_dt: float,
    right_target_position: wp.array(dtype=wp.vec3),
    right_target_rotation: wp.array(dtype=wp.vec4),
    gripper_blend: wp.array(dtype=wp.float32),
    diagnostics: wp.array2d(dtype=wp.float32),
):
    env_id = wp.tid()
    p = phase[env_id]
    time = elapsed[env_id]
    first_step = time == 0.0

    ee_tf = robot_body_q[env_id, right_ee_body] * ee_local_xform
    connector_body_tf = cable_body_q[cable_head_bodies[env_id]]
    connector_body_position = wp.transform_get_translation(connector_body_tf)
    connector_body_rotation = wp.normalize(wp.transform_get_rotation(connector_body_tf))
    connector_local_position = wp.transform_get_translation(connector_local_xform)
    connector_local_rotation = wp.normalize(wp.transform_get_rotation(connector_local_xform))
    connector_position = connector_body_position + wp.quat_rotate(connector_body_rotation, connector_local_position)
    connector_rotation = connector_body_rotation * connector_local_rotation
    connector_tf = wp.transform(connector_position, connector_rotation)
    if first_step:
        phase_ee[env_id] = ee_tf
        phase_connector[env_id] = connector_tf

    ee_position = wp.transform_get_translation(ee_tf)
    ee_rotation = wp.transform_get_rotation(ee_tf)
    connector_axis = wp.quat_rotate(connector_rotation, wp.vec3(0.0, 0.0, 1.0))
    connector_tip_position = connector_position + wp.quat_rotate(connector_rotation, connector_tip_local_pos)

    start_tf = phase_ee[env_id]
    start_position = wp.transform_get_translation(start_tf)
    start_rotation = wp.transform_get_rotation(start_tf)
    entry_connector_tf = phase_connector[env_id]
    entry_connector_position = wp.transform_get_translation(entry_connector_tf)
    entry_connector_rotation = wp.transform_get_rotation(entry_connector_tf)
    entry_connector_axis = wp.normalize(wp.quat_rotate(entry_connector_rotation, wp.vec3(0.0, 0.0, 1.0)))

    # Freeze the pick target at phase entry. Chasing small cable motion throughout
    # APPROACH/ENGAGE shifts the final grasp along the tiny flange and leaves too
    # little pad overlap to transmit the CARRY rotation. REST provides a dedicated
    # settling period, so a stable phase-local target is the safer behavior.
    grasp_rotation = wp.normalize(entry_connector_rotation * grasp_orientation_offset)
    grasp_position = entry_connector_position + wp.quat_rotate(entry_connector_rotation, plug_grasp_offset)

    target_position = start_position
    target_rotation = start_rotation
    grip = 0.0

    socket_rotation = wp.normalize(socket_rotation)
    socket_pos_w = socket_position + env_origins[env_id]
    insertion_axis = wp.normalize(wp.quat_rotate(socket_rotation, wp.vec3(0.0, 0.0, 1.0)))
    socket_grasp_rotation = wp.normalize(socket_rotation * grasp_orientation_offset)
    coax_delta = _quat_from_two_vectors_wp(connector_axis, insertion_axis)
    coaxial_rotation = wp.normalize(coax_delta * ee_rotation)
    live_tip_offset = wp.quat_rotate(wp.quat_inverse(ee_rotation), connector_tip_position - ee_position)
    align_delta = _quat_from_two_vectors_wp(entry_connector_axis, insertion_axis)
    align_rotation = wp.normalize(align_delta * start_rotation)

    if p == _RETRACT and first_step:
        frozen_tip_offset[env_id] = live_tip_offset
        grasp_reference_valid[env_id] = True
    if p == _INSERT and first_step:
        frozen_tip_offset[env_id] = live_tip_offset
        frozen_insert_rotation[env_id] = coaxial_rotation

    tip_offset = static_tip_offset
    if p == _CARRY or p == _ALIGN:  # noqa: SIM109 - Warp kernels do not support membership.
        tip_offset = live_tip_offset
    elif p == _INSERT or p == _HOLD_INSERTED:  # noqa: SIM109 - Warp kernels do not support membership.
        tip_offset = frozen_tip_offset[env_id]

    align_tip_position = socket_pos_w + SOCKET_ALIGN_TIP_DEPTH * insertion_axis
    inserted_tip_position = socket_pos_w + SOCKET_SEATED_TIP_DEPTH * insertion_axis
    carry_position = align_tip_position - wp.quat_rotate(socket_grasp_rotation, tip_offset)
    # Keep the phase-entry orientation correction fixed, but centre the measured connector face
    # continuously. This corrects translational grasp slip without accumulating another rotation.
    align_position = align_tip_position - wp.quat_rotate(align_rotation, live_tip_offset)
    frozen_insert_position = inserted_tip_position - wp.quat_rotate(
        frozen_insert_rotation[env_id], frozen_tip_offset[env_id]
    )

    if p == _APPROACH:
        target_position = grasp_position + wp.quat_rotate(entry_connector_rotation, wp.vec3(0.0, 0.08, 0.0))
        target_rotation = grasp_rotation
    elif p == _ENGAGE:
        target_position = grasp_position + wp.vec3(0.01, 0.0, 0.0)
        target_rotation = grasp_rotation
    elif p == _GRASP:
        grip = _smoothstep_wp(time / durations[_GRASP])
    elif p == _HOLD_GRASP:
        grip = 1.0
    elif p == _RETRACT:
        target_position = start_position + wp.quat_rotate(entry_connector_rotation, wp.vec3(0.0, 0.05, 0.0))
        grip = 1.0
    elif p == _SETTLE:
        grip = 1.0
    elif p == _LIFT:
        target_position = start_position + wp.vec3(0.0, 0.0, 0.16)
        target_rotation = socket_grasp_rotation
        grip = 1.0
    elif p == _CARRY:
        target_position = carry_position
        target_rotation = socket_grasp_rotation
        grip = 1.0
    elif p == _ALIGN:
        target_position = align_position
        target_rotation = align_rotation
        grip = 1.0
    elif p >= _INSERT:  # INSERT, HOLD_INSERTED, and terminal DONE.
        target_position = frozen_insert_position
        target_rotation = frozen_insert_rotation[env_id]
        grip = 1.0

    blend = _smoothstep_wp(time / durations[p])
    command_position = (1.0 - blend) * start_position + blend * target_position
    aligned_target_rotation = target_rotation
    if wp.dot(start_rotation, target_rotation) < 0.0:
        aligned_target_rotation = -target_rotation
    command_rotation = wp.normalize((1.0 - blend) * start_rotation + blend * aligned_target_rotation)

    # Newton IK solves each clone against the env-0 prototype model. Move the
    # world command through this environment's root frame into the prototype.
    command_tf = wp.transform(command_position, command_rotation)
    command_b = wp.transform_inverse(robot_root_q[env_id]) * command_tf
    prototype_command = robot_root_q[0] * command_b
    prototype_position = wp.transform_get_translation(prototype_command)
    prototype_rotation = wp.transform_get_rotation(prototype_command)
    right_target_position[env_id] = prototype_position
    right_target_rotation[env_id] = wp.vec4(
        prototype_rotation[0], prototype_rotation[1], prototype_rotation[2], prototype_rotation[3]
    )
    gripper_blend[env_id] = grip

    position_error = target_position - ee_position
    converged = (
        wp.abs(position_error[0]) < 0.01
        and wp.abs(position_error[1]) < 0.01
        and wp.abs(position_error[2]) < 0.01
        and _rotation_error_angle_wp(target_rotation, ee_rotation) < 0.2617994
    )
    tip_delta = connector_tip_position - socket_pos_w
    tip_depth = wp.dot(tip_delta, insertion_axis)
    tip_radial_error = wp.length(tip_delta - tip_depth * insertion_axis)
    axis_cos = wp.dot(connector_axis, insertion_axis)
    diagnostics[env_id, 0] = wp.length(position_error)
    diagnostics[env_id, 1] = _rotation_error_angle_wp(target_rotation, ee_rotation)
    diagnostics[env_id, 2] = axis_cos
    diagnostics[env_id, 3] = tip_depth
    diagnostics[env_id, 4] = tip_radial_error
    diagnostics[env_id, 5] = wp.length(live_tip_offset - frozen_tip_offset[env_id])
    if p == _ALIGN:
        align_ready = (
            axis_cos > SOCKET_RETAINED_AXIS_COS
            and tip_radial_error < 0.001
            and wp.abs(tip_depth - SOCKET_ALIGN_TIP_DEPTH) < 0.003
        )
        if not align_ready:
            converged = False
    elif p >= _INSERT:
        retained = (
            axis_cos >= SOCKET_RETAINED_AXIS_COS
            and tip_radial_error <= SOCKET_RETAINED_RADIAL_TOLERANCE
            and wp.abs(tip_depth - SOCKET_SEATED_TIP_DEPTH) <= SOCKET_RETAINED_DEPTH_TOLERANCE
        )
        # Once contact has seated the connector, measured retention is the phase objective.
        converged = retained

    next_time = time + frame_dt
    minimum_time_met = next_time >= durations[p]
    hard_timeout = next_time >= 2.0 * durations[p]
    # Timeouts are only safe before the socket corridor. ALIGN and every retained-insertion phase
    # must complete from measured geometry; otherwise a lost connector can eventually reach DONE.
    # CARRY is also geometry-dependent: timing it out advances a missed grasp into ALIGN, where it
    # can never recover. RETRACT captures a tip-in-EE reference; retry the pick if the connector
    # stops following the end effector at any point through CARRY.
    grasp_follow_phase = p >= _RETRACT and p <= _CARRY
    grasp_lost = (
        grasp_follow_phase
        and grasp_reference_valid[env_id]
        and wp.length(live_tip_offset - frozen_tip_offset[env_id]) > _GRASP_FOLLOW_TOLERANCE
    )
    timeout_advance = hard_timeout and p < _CARRY
    if grasp_lost:
        phase[env_id] = _APPROACH
        elapsed[env_id] = 0.0
        grasp_reference_valid[env_id] = False
    elif p < _DONE and minimum_time_met and (converged or timeout_advance):
        phase[env_id] = p + 1
        elapsed[env_id] = 0.0
    elif hard_timeout and p == _ALIGN:
        # Contact compliance can change the connector-in-gripper orientation during
        # the one-shot alignment motion. Re-enter ALIGN from the measured pose so the
        # next correction removes that residual instead of waiting on a stale target.
        elapsed[env_id] = 0.0
    elif hard_timeout and p == _INSERT:
        # A failed axial push should back out to the standoff and re-align. Never
        # advance toward the terminal hold unless the measured connector is retained.
        phase[env_id] = _ALIGN
        elapsed[env_id] = 0.0
    else:
        elapsed[env_id] = next_time


@wp.kernel(enable_backward=False)
def _write_gripper_targets_wp(
    gripper_blend: wp.array(dtype=wp.float32),
    open_targets: wp.array(dtype=wp.float32),
    close_targets: wp.array(dtype=wp.float32),
    targets: wp.array2d(dtype=wp.float32),
):
    env_id, joint_id = wp.tid()
    blend = gripper_blend[env_id]
    targets[env_id, joint_id] = open_targets[joint_id] + blend * (close_targets[joint_id] - open_targets[joint_id])


def _smoothstep(alpha: torch.Tensor) -> torch.Tensor:
    """Classic 3a^2 - 2a^3 ease-in/ease-out on a clamped [0, 1] interpolant."""
    alpha = torch.clamp(alpha, 0.0, 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _blend_quat(start_quat: torch.Tensor, target_quat: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """Shortest-path normalized-lerp between two quaternions (per env)."""
    target_quat = torch.where(
        torch.sum(start_quat * target_quat, dim=-1, keepdim=True) < 0.0, -target_quat, target_quat
    )
    return normalize(start_quat * (1.0 - blend) + target_quat * blend)


def _quat_from_two_vectors(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Quaternion rotating normalized ``source`` vectors onto ``target`` vectors."""

    source = normalize(source)
    target = normalize(target)
    cross = torch.linalg.cross(source, target, dim=-1)
    cross_norm = torch.linalg.norm(cross, dim=-1, keepdim=True)
    dot = torch.sum(source * target, dim=-1, keepdim=True).clamp(-1.0, 1.0)

    x_axis = torch.zeros_like(source)
    x_axis[:, 0] = 1.0
    y_axis = torch.zeros_like(source)
    y_axis[:, 1] = 1.0
    fallback = torch.linalg.cross(source, x_axis, dim=-1)
    fallback_norm = torch.linalg.norm(fallback, dim=-1, keepdim=True)
    fallback_y = torch.linalg.cross(source, y_axis, dim=-1)
    fallback = torch.where(fallback_norm > 1.0e-6, fallback, fallback_y)
    fallback = normalize(fallback)

    axis = torch.where(cross_norm > 1.0e-6, cross / cross_norm.clamp_min(1.0e-6), fallback)
    angle = torch.atan2(cross_norm.squeeze(-1), dot.squeeze(-1))
    opposite = (cross_norm.squeeze(-1) <= 1.0e-6) & (dot.squeeze(-1) < 0.0)
    angle = torch.where(opposite, torch.full_like(angle, torch.pi), angle)
    return normalize(quat_from_angle_axis(angle, axis))


class WaterhoseDemoState:
    """Per-environment scripted grasp-and-insert state machine."""

    REST = _REST
    APPROACH = _APPROACH
    ENGAGE = _ENGAGE
    GRASP = _GRASP
    HOLD_GRASP = _HOLD_GRASP
    RETRACT = _RETRACT
    SETTLE = _SETTLE
    LIFT = _LIFT
    CARRY = _CARRY
    ALIGN = _ALIGN
    INSERT = _INSERT
    HOLD_INSERTED = _HOLD_INSERTED
    DONE = _DONE

    PHASE_NAMES = (
        "REST",
        "APPROACH",
        "ENGAGE",
        "GRASP",
        "HOLD_GRASP",
        "RETRACT",
        "SETTLE",
        "LIFT",
        "CARRY",
        "ALIGN",
        "INSERT",
        "HOLD_INSERTED",
        "DONE",
    )
    # Minimum time spent in each phase [s]. Free-space phases also require end-effector convergence;
    # insertion uses measured connector retention. DONE holds the seated grasp and is terminal.
    DURATIONS = (
        0.25,
        3.0,
        1.5,
        0.5,
        0.5,
        1.5,
        0.3,
        3.0,
        5.0,
        2.0,
        4.0,
        1.0,
        1.0e6,
    )

    def __init__(self, num_envs: int, step_dt: float, device: torch.device | str, settle_time: float, debug: bool):
        self.num_envs = int(num_envs)
        self.step_dt = float(step_dt)
        self.device = device
        self.debug = bool(debug)

        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.elapsed = torch.zeros(self.num_envs, device=device)
        self._last_reported_phase = -1
        self._step_count = 0

        # Phase-entry snapshots (world frame) and the commanded pose (base frame).
        self.phase_start_pos_w = torch.zeros((self.num_envs, 3), device=device)
        self.phase_start_quat_w = torch.zeros((self.num_envs, 4), device=device)
        self.phase_start_quat_w[:, 3] = 1.0
        self.phase_plug_pos_w = torch.zeros((self.num_envs, 3), device=device)
        self.phase_plug_quat_w = torch.zeros((self.num_envs, 4), device=device)
        self.phase_plug_quat_w[:, 3] = 1.0
        self.command_pose = torch.zeros((self.num_envs, 7), device=device)
        self.command_pose[:, 6] = 1.0
        # Multi-body Newton-IK hold targets: [left_gripper_base pose(7), torso_hip_yaw pose(7)],
        # root frame, quaternions in (x, y, z, w) per the Newton IK action convention. Captured once
        # from the settled pose and held for the whole demo so the torso (and idle left arm) stay put
        # while the right arm tracks the connector.
        self.hold_poses = torch.zeros((self.num_envs, 14), device=device)
        self.hold_poses[:, 6] = 1.0
        self.hold_poses[:, 13] = 1.0
        self._holds_captured = False
        self._left_hold_body_id = None
        self._torso_hold_body_id = None

        durations = list(self.DURATIONS)
        durations[self.REST] = max(float(settle_time), self.step_dt)
        self.durations = torch.tensor(durations, dtype=torch.float32, device=device)

        # Convergence tolerances (generous; combined with the min duration this gives smooth motion).
        self.pos_tolerance = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32, device=device)
        self.rot_tolerance = 15.0 * torch.pi / 180.0
        # Fixed geometric offsets.
        self.plug_grasp_offset = self._vec(PLUG_GRASP_OFFSET)
        self.approach_offset = self._vec((0.0, 0.08, 0.0))
        self.engage_offset = self._vec((0.01, 0.0, 0.0))
        self.retract_vector = self._vec((0.0, 0.05, 0.0))
        self.connector_axis_local = self._vec((0.0, 0.0, 1.0))

        # LIFT first raises the grasp clear. CARRY then moves to the original pre-insert waypoint,
        # where ALIGN corrects the compliant connector before INSERT begins.
        self.align_tip_depth = SOCKET_ALIGN_TIP_DEPTH
        self.lift_vector = self._vec((0.0, 0.0, 0.16))
        self.seated_tip_depth = SOCKET_SEATED_TIP_DEPTH
        self.connector_tip_local_pos = self._vec(CONNECTOR_TIP_LOCAL_POS)

        # End-effector orientation that grasps the plug from the side: Rx(+90) * Rz(-90).
        z_axis = self._vec((0.0, 0.0, 1.0))
        x_axis = self._vec((1.0, 0.0, 0.0))
        q_rz = quat_from_angle_axis(torch.full((self.num_envs,), -torch.pi / 2.0, device=device), z_axis)
        q_rx = quat_from_angle_axis(torch.full((self.num_envs,), torch.pi / 2.0, device=device), x_axis)
        self.grasp_orientation_offset = normalize(quat_mul(q_rx, q_rz))
        self.connector_tip_pos_in_ee = quat_apply(
            quat_inv(self.grasp_orientation_offset),
            self.connector_tip_local_pos - self.plug_grasp_offset,
        )
        # Connector-tip offsets in the end-effector frame. CARRY/ALIGN use the live offset for precise
        # centring, then INSERT freezes both offset and orientation for a straight axial push.
        self._tip_offset_frozen = self.connector_tip_pos_in_ee.clone()
        self._insert_quat_frozen = self._vec((0.0, 0.0, 0.0, 1.0))
        self._grasp_reference_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

        self.socket_pos_w = self._vec(SOCKET_MOUTH_POS)
        self.socket_quat_w = normalize(self._vec(SOCKET_ROT_QUAT_XYZW))

        self.ee_offset_pos = self._vec(RIGHT_GRIPPER_EE_FRAME_POS)
        self.ee_offset_quat = self._vec(RIGHT_GRIPPER_EE_FRAME_QUAT_XYZW)

        self._ee_body_id = None

    def _vec(self, values) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)

    def compute(self, env) -> torch.Tensor:
        if self.debug:
            self._step_count += 1

        robot = env.scene["robot"]

        if self._ee_body_id is None:
            self._ee_body_id = robot.find_bodies("right_gripper_base")[0][0]
        root_pose_w = robot.data.root_link_pose_w.torch
        root_pos_w = root_pose_w[:, :3]
        root_quat_w = root_pose_w[:, 3:]

        ee_base_pos_w = robot.data.body_pos_w.torch[:, self._ee_body_id]
        ee_base_quat_w = robot.data.body_quat_w.torch[:, self._ee_body_id]
        ee_pos_w, ee_quat_w = combine_frame_transforms(
            ee_base_pos_w, ee_base_quat_w, self.ee_offset_pos, self.ee_offset_quat
        )

        # Live connector pose from the compound cable-head body. The connector mesh's local transform
        # is composed with segment 0, so this remains correct as the hose bends and rotates.
        plug_pos_w, plug_quat_w = env.scene["cable1"].get_connector_pose_w()

        socket_pos_w = self.socket_pos_w + env.scene.env_origins.to(device=self.device, dtype=self.socket_pos_w.dtype)
        insertion_dir_w = normalize(quat_apply(self.socket_quat_w, self._vec((0.0, 0.0, 1.0))))

        # Phase-entry snapshots (branch-free masked writes; no host sync).
        first_step = self.elapsed == 0.0
        self.phase_start_pos_w[first_step] = ee_pos_w[first_step]
        self.phase_start_quat_w[first_step] = ee_quat_w[first_step]
        self.phase_plug_pos_w[first_step] = plug_pos_w[first_step]
        self.phase_plug_quat_w[first_step] = plug_quat_w[first_step]
        connector_dir = quat_apply(plug_quat_w, self.connector_axis_local)

        # The connector inserts along its local +Z axis; its tip is the mesh origin advanced by
        # the authored connector length.
        cable_tip_pos_w = plug_pos_w + quat_apply(plug_quat_w, self.connector_tip_local_pos)

        start_pos_w = self.phase_start_pos_w
        start_quat_w = self.phase_start_quat_w
        phase_plug_pos_w = self.phase_plug_pos_w
        phase_plug_quat_w = self.phase_plug_quat_w
        entry_connector_axis_w = normalize(quat_apply(phase_plug_quat_w, self.connector_axis_local))

        # Freeze the pick target at phase entry so APPROACH/ENGAGE cannot chase
        # cable motion and move the connector toward the edge of the finger pads.
        grasp_quat_w = normalize(quat_mul(phase_plug_quat_w, self.grasp_orientation_offset))
        grasp_pos_w = phase_plug_pos_w + quat_apply(phase_plug_quat_w, self.plug_grasp_offset)

        phase = self.phase
        target_pos_w = start_pos_w.clone()
        target_quat_w = start_quat_w.clone()
        t_grip = torch.zeros(self.num_envs, device=self.device)

        def set_target(mask, pos_w, quat_w, grip):
            target_pos_w[mask] = pos_w[mask]
            target_quat_w[mask] = quat_w[mask]
            t_grip[mask] = grip

        # --- Pick ---
        approach = phase == self.APPROACH
        set_target(approach, grasp_pos_w + quat_apply(phase_plug_quat_w, self.approach_offset), grasp_quat_w, 0.0)

        engage = phase == self.ENGAGE
        set_target(engage, grasp_pos_w + self.engage_offset, grasp_quat_w, 0.0)

        # GRASP: hold pose, close the gripper over the phase duration.
        grasp = phase == self.GRASP
        grasp_blend = _smoothstep(self.elapsed / torch.clamp(self.durations[self.GRASP], min=1.0e-6))
        target_pos_w[grasp] = start_pos_w[grasp]
        target_quat_w[grasp] = start_quat_w[grasp]
        t_grip[grasp] = grasp_blend[grasp]

        hold = phase == self.HOLD_GRASP
        t_grip[hold] = 1.0

        retract = phase == self.RETRACT
        set_target(retract, start_pos_w + quat_apply(phase_plug_quat_w, self.retract_vector), start_quat_w, 1.0)

        settle = phase == self.SETTLE
        t_grip[settle] = 1.0

        # --- Guided transfer and insert ---
        # Targets are computed from the connector tip pose. The gripper frame is offset behind the
        # tip, so aiming the EE itself at the socket mouth would overshoot and drive the plug into
        # the fridge. CARRY moves to the standoff; ALIGN/INSERT use the cable-tip axis so the hose,
        # not just the plug rigid body, becomes coaxial with the socket bore.
        ins_dir = insertion_dir_w  # bore axis into the socket = R(socket_quat) @ +Z
        socket_grasp_quat = normalize(quat_mul(self.socket_quat_w, self.grasp_orientation_offset))
        coax_delta_quat = _quat_from_two_vectors(connector_dir, ins_dir)
        coaxial_grasp_quat = normalize(quat_mul(coax_delta_quat, ee_quat_w))
        align_delta_quat = _quat_from_two_vectors(entry_connector_axis_w, ins_dir)
        fixed_align_quat = normalize(quat_mul(align_delta_quat, start_quat_w))

        # Connector-face offset in the EE frame for targeting. The static estimate assumes an
        # idealized grasp, while the connector can settle laterally between the fingers. The live
        # offset, derived from the current connector/EE poses, centres the real face on the socket axis:
        #   * CARRY/ALIGN: live offset, to centre the physical tip on the bore axis.
        #   * INSERT entry: freeze the centred offset and aligned quaternion through
        #     INSERT/HOLD_INSERTED so the push stays straight instead of chasing contact deflection.
        #   * Otherwise (REST..SETTLE): the static estimate.
        grasped_tip_offset_ee = quat_apply(quat_inv(ee_quat_w), cable_tip_pos_w - ee_pos_w)
        retract_entry = (phase == self.RETRACT) & first_step
        self._tip_offset_frozen[retract_entry] = grasped_tip_offset_ee[retract_entry]
        self._grasp_reference_valid[retract_entry] = True
        insert_entry = (phase == self.INSERT) & first_step
        self._tip_offset_frozen[insert_entry] = grasped_tip_offset_ee[insert_entry]
        self._insert_quat_frozen[insert_entry] = coaxial_grasp_quat[insert_entry]
        live_mask = (phase == self.CARRY) | (phase == self.ALIGN)
        frozen_mask = (phase == self.INSERT) | (phase == self.HOLD_INSERTED)
        tip_offset = self.connector_tip_pos_in_ee.clone()
        tip_offset = torch.where(live_mask.unsqueeze(-1), grasped_tip_offset_ee, tip_offset)
        tip_offset = torch.where(frozen_mask.unsqueeze(-1), self._tip_offset_frozen, tip_offset)

        def ee_pos_for_tip(target_tip_pos_w, target_ee_quat_w):
            return target_tip_pos_w - quat_apply(target_ee_quat_w, tip_offset)

        align_tip_pos = socket_pos_w + self.align_tip_depth * ins_dir
        inserted_tip_pos = socket_pos_w + self.seated_tip_depth * ins_dir
        carry_pos = ee_pos_for_tip(align_tip_pos, socket_grasp_quat)
        # Keep orientation fixed while using the live connector-to-EE translation to remove grasp slip.
        fixed_align_pos = align_tip_pos - quat_apply(fixed_align_quat, grasped_tip_offset_ee)
        frozen_inserted_pos = inserted_tip_pos - quat_apply(self._insert_quat_frozen, self._tip_offset_frozen)

        # LIFT: move vertically clear of the fridge while rotating toward the socket orientation.
        lift = phase == self.LIFT
        set_target(lift, start_pos_w + self.lift_vector, socket_grasp_quat, 1.0)

        # CARRY: move to the collision-free alignment standoff while rotating toward the socket.
        carry = phase == self.CARRY
        set_target(carry, carry_pos, socket_grasp_quat, 1.0)

        # ALIGN: centre and rotate the connector before its tip reaches the socket collision geometry.
        align = phase == self.ALIGN
        set_target(align, fixed_align_pos, fixed_align_quat, 1.0)

        # INSERT: push straight from the verified standoff with the aligned grasp pose frozen.
        insert = phase == self.INSERT
        set_target(insert, frozen_inserted_pos, self._insert_quat_frozen, 1.0)

        # HOLD_INSERTED: dwell at the seated pose before finishing the scripted motion.
        hold_ins = phase == self.HOLD_INSERTED
        set_target(hold_ins, frozen_inserted_pos, self._insert_quat_frozen, 1.0)

        # DONE: hold the inserted pose and keep the fingers closed. The demo intentionally does not
        # release, so the complete robot/fridge collision shells can remain enabled.
        done = phase == self.DONE
        set_target(done, frozen_inserted_pos, self._insert_quat_frozen, 1.0)

        # Smoothstep blend from the entry pose to the target pose (world frame).
        blend = _smoothstep(self.elapsed / self.durations[self.phase]).unsqueeze(-1)
        cmd_pos_w = start_pos_w * (1.0 - blend) + target_pos_w * blend
        cmd_quat_w = _blend_quat(start_quat_w, target_quat_w, blend)

        cmd_pos_b, cmd_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, cmd_pos_w, cmd_quat_w)
        self.command_pose[:, :3] = cmd_pos_b
        # Isaac Lab math and the Newton IK action both use (x, y, z, w).
        self.command_pose[:, 3:] = cmd_quat_b

        if not self._holds_captured:
            # Capture the multi-body hold targets (root frame, xyzw) once from the settled pose.
            if self._left_hold_body_id is None:
                self._left_hold_body_id = robot.find_bodies("left_gripper_base")[0][0]
                self._torso_hold_body_id = robot.find_bodies("torso_hip_yaw")[0][0]
            for slot, body_id in ((0, self._left_hold_body_id), (7, self._torso_hold_body_id)):
                hold_pos_b, hold_quat_b = subtract_frame_transforms(
                    root_pos_w,
                    root_quat_w,
                    robot.data.body_pos_w.torch[:, body_id],
                    robot.data.body_quat_w.torch[:, body_id],
                )
                self.hold_poses[:, slot : slot + 3] = hold_pos_b
                self.hold_poses[:, slot + 3 : slot + 7] = hold_quat_b
            self._holds_captured = True

        gripper = (_GRIPPER_OPEN + (_GRIPPER_CLOSED - _GRIPPER_OPEN) * t_grip).unsqueeze(-1)
        actions = torch.cat((self.command_pose, self.hold_poses, gripper), dim=-1)
        if actions.shape[-1] != env.action_manager.total_action_dim:
            raise ValueError(
                "The scripted waterhose policy requires the registered 22-D Newton IK action layout; "
                f"the environment exposes {env.action_manager.total_action_dim} dimensions."
            )

        # --- Advance: min duration met AND converged (or hard 2x timeout). ---
        position_error = torch.abs(target_pos_w - ee_pos_w)
        rotation_error = quat_error_magnitude(target_quat_w, ee_quat_w)
        converged = torch.all(position_error < self.pos_tolerance, dim=-1) & (rotation_error < self.rot_tolerance)
        # ALIGN may advance only when the measured connector is safely centred and coaxial. Its target
        # is a single phase-entry correction, so compliance cannot accumulate another rotation each frame.
        coax_cos = torch.sum(connector_dir * ins_dir, dim=-1)
        tip_delta = cable_tip_pos_w - socket_pos_w
        tip_depth = torch.sum(tip_delta * ins_dir, dim=-1, keepdim=True)
        tip_radial_error = torch.linalg.norm(tip_delta - tip_depth * ins_dir, dim=-1)
        align_phase = self.phase == self.ALIGN
        align_ready = (
            (coax_cos > SOCKET_RETAINED_AXIS_COS)
            & (tip_radial_error < 0.001)
            & (torch.abs(tip_depth[:, 0] - self.align_tip_depth) < 0.003)
        )
        converged = converged & (~align_phase | align_ready)
        retained_ready = (
            (coax_cos >= SOCKET_RETAINED_AXIS_COS)
            & (tip_radial_error <= SOCKET_RETAINED_RADIAL_TOLERANCE)
            & (torch.abs(tip_depth[:, 0] - self.seated_tip_depth) <= SOCKET_RETAINED_DEPTH_TOLERANCE)
        )
        retained_objective_phase = self.phase >= self.INSERT
        converged = torch.where(retained_objective_phase, retained_ready, converged)

        if self.debug:
            target_tip_offset = torch.where(align_phase.unsqueeze(-1), grasped_tip_offset_ee, tip_offset)
            target_tip_pos_w = target_pos_w + quat_apply(target_quat_w, target_tip_offset)
            target_tip_depth = torch.sum((target_tip_pos_w - socket_pos_w) * ins_dir, dim=-1)
            current_phase = int(self.phase[0].item())
            phase_changed = current_phase != self._last_reported_phase
            guided_phase = current_phase in (
                self.CARRY,
                self.ALIGN,
                self.INSERT,
                self.HOLD_INSERTED,
            )
            periodic_guided_report = guided_phase and self._step_count % 250 == 0
            if phase_changed or periodic_guided_report:
                name = self.PHASE_NAMES[current_phase]
                print(
                    f"[waterhose_ik] {name}: "
                    f"pos_err={position_error[0].detach().cpu().tolist()} "
                    f"rot_err={float(rotation_error[0].detach().cpu()):.4f} "
                    f"axis_cos={float(coax_cos[0].detach().cpu()):+.2f} "
                    f"tip_depth_mm={float(tip_depth[0, 0].detach().cpu()) * 1000.0:.1f} "
                    f"tip_radial_mm={float(tip_radial_error[0].detach().cpu()) * 1000.0:.1f} "
                    f"target_depth_mm={float(target_tip_depth[0].detach().cpu()) * 1000.0:.1f} "
                    f"grip={float(gripper[0, 0].detach().cpu()):.2f}",
                    flush=True,
                )
                self._last_reported_phase = current_phase

        self.elapsed += self.step_dt
        timed_out = self.elapsed >= self.durations[self.phase]
        hard_timeout = self.elapsed >= 2.0 * self.durations[self.phase]
        grasp_follow_phase = (self.phase >= self.RETRACT) & (self.phase <= self.CARRY)
        grasp_offset_error = torch.linalg.norm(grasped_tip_offset_ee - self._tip_offset_frozen, dim=-1)
        grasp_lost = grasp_follow_phase & self._grasp_reference_valid & (grasp_offset_error > _GRASP_FOLLOW_TOLERANCE)
        timeout_advance = hard_timeout & (self.phase < self.CARRY)
        should_advance = timed_out & (converged | timeout_advance) & (self.phase < self.DONE) & ~grasp_lost
        realign = hard_timeout & (self.phase == self.ALIGN) & ~converged
        retry_insert = hard_timeout & (self.phase == self.INSERT) & ~converged

        self.phase[should_advance] += 1
        self.elapsed[should_advance] = 0.0
        self.elapsed[realign] = 0.0
        self.phase[retry_insert] = self.ALIGN
        self.elapsed[retry_insert] = 0.0
        self.phase[grasp_lost] = self.APPROACH
        self.elapsed[grasp_lost] = 0.0
        self._grasp_reference_valid[grasp_lost] = False

        return actions


class WaterhoseGraphDemoState:
    """CUDA-graph-native version of the scripted waterhose controller.

    The graph contains the state machine, Newton IK solve, gripper interpolation,
    and writes into Newton's live control target. Physics remains in the Newton
    manager's graph, so the scripted runner performs two graph launches per frame
    without executing the RL manager stack.
    """

    REST = _REST
    APPROACH = _APPROACH
    ENGAGE = _ENGAGE
    GRASP = _GRASP
    HOLD_GRASP = _HOLD_GRASP
    RETRACT = _RETRACT
    SETTLE = _SETTLE
    LIFT = _LIFT
    CARRY = _CARRY
    ALIGN = _ALIGN
    INSERT = _INSERT
    HOLD_INSERTED = _HOLD_INSERTED
    DONE = _DONE
    PHASE_NAMES = WaterhoseDemoState.PHASE_NAMES
    DURATIONS = WaterhoseDemoState.DURATIONS
    is_control_graph_captured = True

    def __init__(self, env, *, settle_time: float):
        from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415

        self.num_envs = int(env.num_envs)
        self.step_dt = float(env.step_dt)
        self.device = wp.get_device(str(env.device))
        if not self.device.is_cuda:
            raise ValueError("WaterhoseGraphDemoState requires a CUDA device.")
        if NewtonManager._graph is None:
            raise RuntimeError("Newton's physics CUDA graph must be captured before the scripted controller graph.")

        robot = env.scene["robot"]
        cable = env.scene["cable1"]
        self._arm_action = env.action_manager.get_term("arm_action")
        self._gripper_action = env.action_manager.get_term("gripper_action")
        if self._gripper_action.cfg.max_joint_delta_per_step is not None:
            raise ValueError("The scripted controller graph requires an un-ratelimited gripper action.")

        self._robot_body_q = robot.data.body_link_pose_w.warp
        self._robot_root_q = robot.data.root_link_pose_w.warp
        self._cable_body_q = NewtonManager.get_state_0().body_q
        self._cable_head_bodies = cable.connector_head_body_ids_warp
        if self._cable_head_bodies.shape[0] != self.num_envs:
            raise RuntimeError(
                "Expected one cable head body per environment, got "
                f"{self._cable_head_bodies.shape[0]} for {self.num_envs} envs."
            )

        # Keep Torch owners alive for every zero-copy Warp view stored in the graph.
        self._env_origins_owner = env.scene.env_origins.to(device=str(env.device), dtype=torch.float32).contiguous()
        self._env_origins = wp.from_torch(self._env_origins_owner, dtype=wp.vec3)

        self._right_ee_body = robot.find_bodies("right_gripper_base")[0][0]
        left_hold_body = robot.find_bodies("left_gripper_base")[0][0]
        torso_hold_body = robot.find_bodies("torso_hip_yaw")[0][0]
        self._ee_local_xform = wp.transform(RIGHT_GRIPPER_EE_FRAME_POS, RIGHT_GRIPPER_EE_FRAME_QUAT_XYZW)
        self._connector_local_xform = wp.transform(
            cable.connector_local_pos_from_head_com, cable.cfg.connector_local_quat
        )

        q_rz = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -0.5 * wp.pi)
        q_rx = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * wp.pi)
        self._grasp_orientation_offset = wp.normalize(q_rx * q_rz)
        ideal_tip_from_grasp = wp.vec3(*CONNECTOR_TIP_LOCAL_POS) - wp.vec3(*PLUG_GRASP_OFFSET)
        self._static_tip_offset = wp.quat_rotate(wp.quat_inverse(self._grasp_orientation_offset), ideal_tip_from_grasp)

        durations = list(self.DURATIONS)
        durations[self.REST] = max(float(settle_time), self.step_dt)
        self._durations = wp.array(durations, dtype=wp.float32, device=self.device)
        self.phase = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        self._elapsed = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._phase_ee = wp.array([wp.transform()] * self.num_envs, dtype=wp.transform, device=self.device)
        self._phase_connector = wp.array([wp.transform()] * self.num_envs, dtype=wp.transform, device=self.device)
        self._initial_tip_offsets = wp.array(
            [self._static_tip_offset] * self.num_envs, dtype=wp.vec3, device=self.device
        )
        self._frozen_tip_offset = wp.empty_like(self._initial_tip_offsets)
        self._identity_rotations = wp.array([wp.quat_identity()] * self.num_envs, dtype=wp.quat, device=self.device)
        self._frozen_insert_rotation = wp.empty_like(self._identity_rotations)
        self._grasp_reference_valid = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._gripper_blend = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._diagnostics = wp.zeros((self.num_envs, 6), dtype=wp.float32, device=self.device)

        objectives = self._arm_action._ik_solver.objectives_by_name
        missing = {"right_ee", "left_hold", "torso_hold"} - objectives.keys()
        if missing:
            raise RuntimeError(f"Scripted controller graph is missing Newton IK objectives: {sorted(missing)}")
        right_objective = objectives["right_ee"]
        left_objective = objectives["left_hold"]
        torso_objective = objectives["torso_hold"]
        self._right_target_position = right_objective.position_objective.target_positions
        self._right_target_rotation = right_objective.rotation_objective.target_rotations

        wp.launch(
            _capture_hold_targets_wp,
            dim=self.num_envs,
            inputs=[
                self._robot_body_q,
                self._robot_root_q,
                left_hold_body,
                torso_hold_body,
                left_objective.position_objective.target_positions,
                left_objective.rotation_objective.target_rotations,
                torso_objective.position_objective.target_positions,
                torso_objective.rotation_objective.target_rotations,
            ],
            device=self.device,
        )

        self._open_targets_owner = self._gripper_action._open_command
        self._close_targets_owner = self._gripper_action._close_command
        self._gripper_targets_owner = self._gripper_action._processed_actions
        self._open_targets = wp.from_torch(self._open_targets_owner)
        self._close_targets = wp.from_torch(self._close_targets_owner)
        self._gripper_targets = wp.from_torch(self._gripper_targets_owner)
        self._lab_joint_targets = robot.data._joint_pos_target
        self._sim_joint_targets = robot.data._sim_bind_joint_position_target

        # Compile and allocate every IK path before capture. This changes only
        # command buffers; physics has not stepped yet. Reset the policy state
        # afterward so the first replay starts at REST, exactly like eager mode.
        self._reset_state()
        self._launch_control()
        wp.synchronize_device(self.device)
        self._reset_state()
        wp.synchronize_device(self.device)

        with wp.ScopedDevice(self.device), wp.ScopedCapture(device=self.device) as capture:
            self._launch_control()
        self.graph = capture.graph
        self._last_reported_phase = -1
        self._sim_step_counter = 0

    def _reset_state(self) -> None:
        self.phase.zero_()
        self._elapsed.zero_()
        self._gripper_blend.zero_()
        self._grasp_reference_valid.zero_()
        wp.copy(self._frozen_tip_offset, self._initial_tip_offsets)
        wp.copy(self._frozen_insert_rotation, self._identity_rotations)

    def _launch_control(self) -> None:
        wp.launch(
            _update_state_machine_wp,
            dim=self.num_envs,
            inputs=[
                self._robot_body_q,
                self._robot_root_q,
                self._cable_body_q,
                self._cable_head_bodies,
                self._env_origins,
                self.phase,
                self._elapsed,
                self._durations,
                self._phase_ee,
                self._phase_connector,
                self._frozen_tip_offset,
                self._frozen_insert_rotation,
                self._grasp_reference_valid,
                self._right_ee_body,
                self._ee_local_xform,
                self._connector_local_xform,
                wp.vec3(*SOCKET_MOUTH_POS),
                wp.quat(*SOCKET_ROT_QUAT_XYZW),
                wp.vec3(*PLUG_GRASP_OFFSET),
                wp.vec3(*CONNECTOR_TIP_LOCAL_POS),
                self._grasp_orientation_offset,
                self._static_tip_offset,
                self.step_dt,
                self._right_target_position,
                self._right_target_rotation,
                self._gripper_blend,
                self._diagnostics,
            ],
            device=self.device,
        )
        self._arm_action.apply_actions()
        wp.launch(
            _write_gripper_targets_wp,
            dim=(self.num_envs, self._gripper_targets.shape[1]),
            inputs=[self._gripper_blend, self._open_targets, self._close_targets, self._gripper_targets],
            device=self.device,
        )
        self._gripper_action.apply_actions()
        wp.copy(self._sim_joint_targets, self._lab_joint_targets)

    def step(self) -> None:
        """Replay the captured controller graph once."""

        wp.capture_launch(self.graph)

    def step_environment(self, env) -> None:
        """Advance the captured controller and physics without the RL manager stack.

        The graph writes action targets directly, so calling ``env.step`` would process and apply
        them a second time. The scripted task disables episode resets and does not consume
        observations, rewards, recorders, or interval events; this method keeps that optimized
        contract explicit and retains the environment's configured render cadence.
        """
        self.step()
        self._sim_step_counter += env.cfg.decimation
        env.sim.step(render=False)
        if self._sim_step_counter % env.cfg.sim.render_interval == 0 and env.sim.is_rendering:
            env.sim.render(skip_app_pumping=not env.render_enabled)

    def read_phases(self) -> list[int]:
        """Synchronously read phase values for occasional UI/termination polling."""

        return [int(value) for value in self.phase.numpy()]

    def report_phase(self, step: int) -> list[int]:
        """Print an env-0 transition and return all current phases."""

        phases = self.read_phases()
        current_phase = phases[0]
        phase_changed = current_phase != self._last_reported_phase
        periodic_guided_report = current_phase in (
            self.CARRY,
            self.ALIGN,
            self.INSERT,
            self.HOLD_INSERTED,
        ) and (step % 300 == 0)
        if phase_changed:
            print(f"[waterhose SM] step {step}: phase = {self.PHASE_NAMES[current_phase]}", flush=True)
            self._last_reported_phase = current_phase
        if phase_changed or periodic_guided_report:
            position_error, rotation_error, axis_cos, tip_depth, tip_radial_error, grasp_offset_error = (
                self._diagnostics.numpy()[0]
            )
            print(
                f"[waterhose SM] metrics: pos_err={position_error:.4f}m "
                f"rot_err={rotation_error:.4f}rad axis_cos={axis_cos:+.4f} "
                f"tip_depth={tip_depth * 1000.0:.1f}mm tip_radial={tip_radial_error * 1000.0:.1f}mm "
                f"grasp_offset_err={grasp_offset_error * 1000.0:.1f}mm",
                flush=True,
            )
        return phases


def create_scripted_policy(
    env, *, settle_time: float = 4.0, debug: bool = False
) -> WaterhoseDemoState | WaterhoseGraphDemoState:
    """Create the task-local scripted policy used by the demo launcher.

    Newton defers physics-graph capture until the first physics step so all managers can finish
    lazy CUDA initialization. Perform that one setup step here and immediately reset the complete
    scene so the controller graph can be captured before the timed rollout without retaining any
    warmup motion.
    """

    if not debug and "cuda" in str(env.device):
        from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415

        if NewtonManager._graph is None and NewtonManager._graph_capture_pending:
            env.sim.step(render=False)
            env.reset()
        if NewtonManager._graph is not None:
            return WaterhoseGraphDemoState(env, settle_time=settle_time)
    return WaterhoseDemoState(env.num_envs, env.step_dt, env.device, settle_time, debug)
