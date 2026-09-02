# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure geometry helpers for the AVBD slung-load task."""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_apply_inverse

_HANGING_QUAT_XYZW = (1.0, 0.0, 0.0, 0.0)
_EPS = 1.0e-6


def _safe_normalize(vector: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Return finite unit vectors, using zero for degenerate inputs."""
    finite = torch.nan_to_num(vector)
    norm = torch.linalg.vector_norm(finite, dim=-1, keepdim=True)
    return torch.where(norm > eps, finite / norm.clamp_min(eps), torch.zeros_like(finite))


def attachment_kinematics(
    link_pose_w: torch.Tensor,
    com_pos_w: torch.Tensor,
    com_vel_w: torch.Tensor,
    local_offset: tuple[float, float, float] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a rigid attachment point's world position and velocity.

    Args:
        link_pose_w: Actor-frame pose ``[position, quaternion_xyzw]`` [m], shape ``(N, 7)``.
        com_pos_w: Center-of-mass position [m], shape ``(N, 3)``.
        com_vel_w: Center-of-mass spatial velocity ``[linear, angular]``
            [m/s, rad/s], shape ``(N, 6)``.
        local_offset: Attachment offset in the actor frame [m], shape ``(3,)`` or ``(N, 3)``.

    Returns:
        Attachment position [m] and point velocity [m/s], each shape ``(N, 3)``.
    """
    offset = torch.as_tensor(local_offset, device=link_pose_w.device, dtype=link_pose_w.dtype)
    if offset.ndim == 1:
        offset = offset.unsqueeze(0).expand(link_pose_w.shape[0], -1)
    offset_w = quat_apply(link_pose_w[:, 3:7], offset)
    point_pos_w = link_pose_w[:, :3] + offset_w
    radius_w = point_pos_w - com_pos_w
    point_vel_w = com_vel_w[:, :3] + torch.cross(com_vel_w[:, 3:6], radius_w, dim=-1)
    return torch.nan_to_num(point_pos_w), torch.nan_to_num(point_vel_w)


def swing_features(attachment_vector_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FLARE swing angles and total deflection from body-frame down.

    ``phi = atan2(y_b, -z_b)`` and ``theta = atan2(x_b, -z_b)``. Degenerate
    vectors map to zero rather than producing undefined observations.
    """
    vector = torch.nan_to_num(attachment_vector_b)
    valid = torch.linalg.vector_norm(vector, dim=-1, keepdim=True) > _EPS
    angles = torch.stack(
        (torch.atan2(vector[:, 1], -vector[:, 2]), torch.atan2(vector[:, 0], -vector[:, 2])),
        dim=-1,
    )
    direction = _safe_normalize(vector)
    total = torch.acos(torch.clamp(-direction[:, 2:3], -1.0, 1.0))
    return torch.where(valid, angles, torch.zeros_like(angles)), torch.where(valid, total, torch.zeros_like(total))


def transverse_velocity(relative_velocity_b: torch.Tensor, attachment_vector_b: torch.Tensor) -> torch.Tensor:
    """Remove relative velocity parallel to the payload attachment vector [m/s]."""
    direction = _safe_normalize(attachment_vector_b)
    relative = torch.nan_to_num(relative_velocity_b)
    return relative - torch.sum(relative * direction, dim=-1, keepdim=True) * direction


def rotation_matrix_flat(quaternion_w: torch.Tensor) -> torch.Tensor:
    """Return the row-major body-to-world rotation matrix, shape ``(N, 9)``."""
    quaternion = torch.nan_to_num(quaternion_w)
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    identity = torch.zeros_like(quaternion)
    identity[:, 3] = 1.0
    quaternion = torch.where(norm > _EPS, quaternion / norm.clamp_min(_EPS), identity)
    return matrix_from_quat(quaternion).reshape(quaternion.shape[0], 9)


def cable_features(
    segment_pose_w: torch.Tensor,
    drone_attachment_w: torch.Tensor,
    payload_attachment_w: torch.Tensor,
    drone_quat_w: torch.Tensor,
    nominal_length: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return upper-cable tangent and relative AVBD joint separation.

    Newton cable capsules use local +Z along their centerline. Reading the first
    capsule orientation keeps the tangent correct even for a sharply folded,
    zero-bending string. ``relative_separation`` is the sum of all attachment/joint
    endpoint gaps divided by nominal length; unlike a center-polyline approximation,
    it does not report false compression merely because adjacent segments bend.
    """
    poses = torch.nan_to_num(segment_pose_w)
    local_z = torch.zeros(poses.shape[0], 3, device=poses.device, dtype=poses.dtype)
    local_z[:, 2] = 1.0
    upper_w = quat_apply(poses[:, 0, 3:7], local_z)
    tangent_b = quat_apply_inverse(drone_quat_w, _safe_normalize(upper_w))

    relative_separation, _ = cable_constraint_errors(
        segment_pose_w, drone_attachment_w, payload_attachment_w, nominal_length
    )
    return torch.nan_to_num(tangent_b), torch.nan_to_num(relative_separation).unsqueeze(-1)


def cable_constraint_errors(
    segment_pose_w: torch.Tensor,
    drone_attachment_w: torch.Tensor,
    payload_attachment_w: torch.Tensor,
    nominal_length: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw relative cable separation and maximum joint endpoint gap.

    This helper deliberately does not sanitize non-finite state: termination
    guards need to fail closed instead of turning a broken cable into zeros.
    Newton imports each segment as a rigid capsule whose local +Z axis spans its
    cylindrical centerline. The returned gaps therefore cover the two external
    attachments and every internal cable joint exactly, including folded cable
    configurations where center-to-center distances are not material stretch.
    """
    if segment_pose_w.ndim != 3 or segment_pose_w.shape[-1] != 7 or segment_pose_w.shape[1] < 1:
        raise ValueError(f"segment_pose_w must have shape (N, S, 7) with S >= 1, got {segment_pose_w.shape}.")
    if drone_attachment_w.shape != (segment_pose_w.shape[0], 3):
        raise ValueError("drone_attachment_w must have shape (N, 3).")
    if payload_attachment_w.shape != (segment_pose_w.shape[0], 3):
        raise ValueError("payload_attachment_w must have shape (N, 3).")

    nominal = torch.as_tensor(nominal_length, device=segment_pose_w.device, dtype=segment_pose_w.dtype)
    if nominal.ndim == 0:
        nominal = nominal.expand(segment_pose_w.shape[0])
    elif nominal.shape != (segment_pose_w.shape[0],):
        raise ValueError("nominal_length must be scalar or have shape (N,).")

    num_segments = segment_pose_w.shape[1]
    local_half_segment = torch.zeros_like(segment_pose_w[..., :3])
    local_half_segment[..., 2] = (0.5 * nominal / num_segments).unsqueeze(-1)
    half_segment_w = quat_apply(segment_pose_w[..., 3:7].reshape(-1, 4), local_half_segment.reshape(-1, 3)).reshape_as(
        local_half_segment
    )
    segment_start_w = segment_pose_w[..., :3] - half_segment_w
    segment_end_w = segment_pose_w[..., :3] + half_segment_w

    gap_vectors = [segment_start_w[:, :1] - drone_attachment_w.unsqueeze(1)]
    if num_segments > 1:
        gap_vectors.append(segment_start_w[:, 1:] - segment_end_w[:, :-1])
    gap_vectors.append(payload_attachment_w.unsqueeze(1) - segment_end_w[:, -1:])
    joint_gaps = torch.linalg.vector_norm(torch.cat(gap_vectors, dim=1), dim=-1)
    relative_separation = joint_gaps.sum(dim=-1) / nominal
    return relative_separation, joint_gaps.amax(dim=-1)


def straight_segment_poses(
    attach_pos_w: torch.Tensor,
    length: torch.Tensor,
    num_segments: int,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Return equally spaced capsule poses along a straight cable direction.

    Args:
        attach_pos_w: World-frame attachment points [m], shape ``(N, 3)``.
        length: Total cable length [m], shape ``(N,)``.
        num_segments: Number of capsule segments.
        direction: Cable direction from drone to payload, shape ``(N, 3)``.

    Returns:
        Segment poses ``[position, quaternion_xyzw]``, shape ``(N, num_segments, 7)``.
    """
    if num_segments < 1:
        raise ValueError(f"num_segments must be >= 1, got {num_segments}.")
    direction = _safe_normalize(direction)
    segment_length = length / num_segments
    offsets = (torch.arange(num_segments, device=attach_pos_w.device, dtype=attach_pos_w.dtype) + 0.5).view(1, -1, 1)
    pos = attach_pos_w.unsqueeze(1) + direction.unsqueeze(1) * offsets * segment_length.view(-1, 1, 1)

    horizontal = torch.linalg.vector_norm(direction[:, :2], dim=-1)
    tilt_from_down = torch.atan2(horizontal, -direction[:, 2])
    azimuth = torch.atan2(direction[:, 1], direction[:, 0])
    half_rotation = 0.5 * (math.pi - tilt_from_down)
    quat = torch.zeros(direction.shape[0], 4, device=direction.device, dtype=direction.dtype)
    quat[:, 0] = -torch.sin(azimuth) * torch.sin(half_rotation)
    quat[:, 1] = torch.cos(azimuth) * torch.sin(half_rotation)
    quat[:, 3] = torch.cos(half_rotation)
    hanging = horizontal <= _EPS
    quat[hanging] = torch.tensor(_HANGING_QUAT_XYZW, device=direction.device, dtype=direction.dtype)
    quat = quat.unsqueeze(1).expand(-1, num_segments, -1)
    return torch.cat((pos, quat), dim=-1)


def straight_end_point(attach_pos_w: torch.Tensor, length: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Return a straight cable endpoint while preserving exact cable length [m]."""
    return attach_pos_w + _safe_normalize(direction) * length.unsqueeze(-1)
