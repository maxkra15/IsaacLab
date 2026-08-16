# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for the task-local Newton RJ45 assembly."""

from __future__ import annotations

import warp as wp

from newton.math import quat_between_vectors_robust


@wp.kernel(enable_backward=False)
def apply_connector_forces(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_mass: wp.array[float],
    gravity: wp.array[wp.vec3],
    world_ids: wp.array[int],
    plug_body_ids: wp.array[int],
    latch_body_ids: wp.array[int],
    drive_enabled: wp.array[wp.bool],
    drive_target_w: wp.array[wp.vec3],
    drive_stiffness: float,
    drive_damping: float,
):
    """Cancel connector gravity and optionally drive each plug toward its target."""
    env_id = wp.tid()
    plug_idx = plug_body_ids[env_id]
    latch_idx = latch_body_ids[env_id]
    gravity_w = gravity[world_ids[env_id]]

    plug_mass = body_mass[plug_idx]
    latch_mass = body_mass[latch_idx]
    wp.atomic_add(
        body_f,
        plug_idx,
        wp.spatial_vector(-gravity_w * plug_mass, wp.vec3(0.0)),
    )
    wp.atomic_add(
        body_f,
        latch_idx,
        wp.spatial_vector(-gravity_w * latch_mass, wp.vec3(0.0)),
    )

    if not drive_enabled[env_id]:
        return

    target = drive_target_w[env_id]
    plug_pos = wp.transform_get_translation(body_q[plug_idx])
    plug_vel = wp.spatial_top(body_qd[plug_idx])
    plug_multiplier = 10.0 + plug_mass
    plug_force = plug_multiplier * (drive_stiffness * (target - plug_pos) - drive_damping * plug_vel)
    wp.atomic_add(body_f, plug_idx, wp.spatial_vector(plug_force, wp.vec3(0.0)))

    latch_vel = wp.spatial_top(body_qd[latch_idx])
    drive_acceleration = (target - plug_pos) * (plug_multiplier * drive_stiffness / plug_mass)
    latch_force = drive_acceleration * latch_mass - latch_vel * ((10.0 + latch_mass) * drive_damping)
    wp.atomic_add(body_f, latch_idx, wp.spatial_vector(latch_force, wp.vec3(0.0)))


@wp.kernel(enable_backward=False)
def sync_cable_anchors(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    plug_body_ids: wp.array[int],
    anchor_body_ids: wp.array2d[int],
    anchor_offsets: wp.array[wp.vec3],
    anchor_rotations: wp.array[wp.quat],
):
    """Copy each plug transform into its four plug-relative cable anchors."""
    env_id, anchor_id = wp.tid()
    plug_tf = body_q[plug_body_ids[env_id]]
    plug_pos = wp.transform_get_translation(plug_tf)
    plug_rot = wp.transform_get_rotation(plug_tf)
    body_idx = anchor_body_ids[env_id, anchor_id]
    anchor_world = plug_pos + wp.quat_rotate(plug_rot, anchor_offsets[anchor_id])
    cable_rot = wp.normalize(wp.mul(plug_rot, anchor_rotations[anchor_id]))
    body_q[body_idx] = wp.transform(anchor_world, cable_rot)
    body_qd[body_idx] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel(enable_backward=False)
def align_cable_orientations(
    body_q: wp.array[wp.transform],
    cable_body_ids: wp.array2d[int],
    cable_next_body_ids: wp.array2d[int],
    cable_next_start_offsets: wp.array[wp.vec3],
):
    """Swing each cable capsule's local +Z axis toward its deformed successor."""
    env_id, segment_id = wp.tid()
    body_idx = cable_body_ids[env_id, segment_id]
    next_idx = cable_next_body_ids[env_id, segment_id]

    body_tf = body_q[body_idx]
    body_pos = wp.transform_get_translation(body_tf)
    body_rot = wp.transform_get_rotation(body_tf)
    next_tf = body_q[next_idx]
    next_start = wp.transform_get_translation(next_tf) + wp.quat_rotate(
        wp.transform_get_rotation(next_tf), cable_next_start_offsets[segment_id]
    )
    segment = next_start - body_pos
    segment_length = wp.length(segment)
    if segment_length < 1.0e-10:
        return

    direction = segment / segment_length
    current_axis = wp.quat_rotate(body_rot, wp.vec3(0.0, 0.0, 1.0))
    swing = quat_between_vectors_robust(current_axis, direction)
    body_q[body_idx] = wp.transform(body_pos, wp.normalize(wp.mul(swing, body_rot)))


@wp.kernel(enable_backward=False)
def reset_task_bodies(
    env_mask: wp.array[wp.bool],
    task_body_ids: wp.array2d[int],
    default_body_q: wp.array2d[wp.transform],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Restore selected task bodies to their authored default state."""
    env_id, local_body_id = wp.tid()
    if env_mask[env_id]:
        body_idx = task_body_ids[env_id, local_body_id]
        body_q[body_idx] = default_body_q[env_id, local_body_id]
        body_qd[body_idx] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel(enable_backward=False)
def write_task_body_state(
    env_mask: wp.array[wp.bool],
    task_body_ids: wp.array2d[int],
    task_body_q: wp.array2d[wp.transform],
    task_body_qd: wp.array2d[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Scatter a complete task-local body state into selected Newton worlds."""
    env_id, local_body_id = wp.tid()
    if env_mask[env_id]:
        body_idx = task_body_ids[env_id, local_body_id]
        body_q[body_idx] = task_body_q[env_id, local_body_id]
        body_qd[body_idx] = task_body_qd[env_id, local_body_id]


@wp.kernel(enable_backward=False)
def set_drive_enabled_masked(
    env_mask: wp.array[wp.bool],
    value: bool,
    drive_enabled: wp.array[wp.bool],
):
    """Set the goal drive flag for selected worlds."""
    env_id = wp.tid()
    if env_mask[env_id]:
        drive_enabled[env_id] = value


@wp.kernel(enable_backward=False)
def restore_goal_targets_masked(
    env_mask: wp.array[wp.bool],
    default_goal_target_w: wp.array[wp.vec3],
    drive_target_w: wp.array[wp.vec3],
):
    """Restore the USD-authored nominal target for selected worlds."""
    env_id = wp.tid()
    if env_mask[env_id]:
        drive_target_w[env_id] = default_goal_target_w[env_id]


@wp.kernel(enable_backward=False)
def write_drive_targets_masked(
    env_mask: wp.array[wp.bool],
    targets_w: wp.array[wp.vec3],
    drive_target_w: wp.array[wp.vec3],
):
    """Write world-frame insertion-drive targets for selected worlds."""
    env_id = wp.tid()
    if env_mask[env_id]:
        drive_target_w[env_id] = targets_w[env_id]
