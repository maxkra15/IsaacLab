# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared physical success predicate for Franka RJ45 insertion."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.utils import math as math_utils

RJ45_SUCCESS_PREDICATE_VERSION = 1
"""Version of the geometry, sidedness, orientation, and speed predicate."""


@dataclass(frozen=True)
class RJ45SuccessResult:
    """Per-environment success mask and its physical components."""

    mask: torch.Tensor
    signed_axial_error: torch.Tensor
    axial_error: torch.Tensor
    radial_error: torch.Tensor
    plug_angle_error: torch.Tensor
    latch_angle_error: torch.Tensor
    plug_spatial_speed: torch.Tensor


def _goal_body(goal_pose: torch.Tensor, body_index: int, batch_size: int) -> torch.Tensor:
    """Return a fixed or batched goal body pose with an explicit batch dimension."""
    if goal_pose.ndim == 2:
        if goal_pose.shape[-1] != 7:
            raise ValueError(f"Goal body poses must end in 7 values, got {tuple(goal_pose.shape)}.")
        return goal_pose[body_index].unsqueeze(0).expand(batch_size, -1)
    if goal_pose.ndim == 3 and goal_pose.shape[-1] == 7 and goal_pose.shape[0] in (1, batch_size):
        return goal_pose[:, body_index].expand(batch_size, -1)
    raise ValueError(
        "Goal body poses must have shape (body_count, 7), (1, body_count, 7), or "
        f"({batch_size}, body_count, 7); got {tuple(goal_pose.shape)}."
    )


def _orientation_error(current_quat: torch.Tensor, goal_quat: torch.Tensor) -> torch.Tensor:
    goal_inverse = math_utils.quat_conjugate(goal_quat)
    error = math_utils.quat_unique(math_utils.quat_mul(goal_inverse, current_quat))
    return torch.linalg.vector_norm(math_utils.axis_angle_from_quat(error), dim=-1)


def rj45_insertion_success(
    task_body_pose: torch.Tensor,
    task_body_velocity: torch.Tensor,
    goal_task_body_pose: torch.Tensor,
    *,
    axial_tolerance: float,
    axial_overtravel_tolerance: float,
    radial_tolerance: float,
    plug_angle_tolerance: float,
    latch_angle_tolerance: float,
    maximum_plug_spatial_speed: float,
) -> RJ45SuccessResult:
    """Evaluate the exact runtime sparse-success geometry from raw task state.

    The insertion axis is local/world ``+Y`` because the validated task frame
    is identity-rotated. Positive signed axial error is overtravel beyond the
    fixed goal and therefore receives the tighter sided tolerance.
    """
    if task_body_pose.ndim != 3 or task_body_pose.shape[-1] != 7 or task_body_pose.shape[1] < 2:
        raise ValueError(f"Task body poses must have shape (N, >=2, 7), got {tuple(task_body_pose.shape)}.")
    if (
        task_body_velocity.ndim != 3
        or task_body_velocity.shape[:2] != task_body_pose.shape[:2]
        or task_body_velocity.shape[-1] != 6
    ):
        raise ValueError(
            "Task body velocities must have shape matching task poses with 6 spatial values; "
            f"got {tuple(task_body_velocity.shape)} for poses {tuple(task_body_pose.shape)}."
        )

    batch_size = task_body_pose.shape[0]
    plug = task_body_pose[:, 0]
    latch = task_body_pose[:, 1]
    goal_plug = _goal_body(goal_task_body_pose, 0, batch_size)
    goal_latch = _goal_body(goal_task_body_pose, 1, batch_size)

    translation_error = plug[:, :3] - goal_plug[:, :3]
    signed_axial_error = translation_error[:, 1]
    axial_error = signed_axial_error.abs()
    radial_error = torch.linalg.vector_norm(translation_error[:, (0, 2)], dim=-1)
    plug_angle_error = _orientation_error(plug[:, 3:7], goal_plug[:, 3:7])
    latch_angle_error = _orientation_error(latch[:, 3:7], goal_latch[:, 3:7])
    plug_spatial_speed = torch.linalg.vector_norm(task_body_velocity[:, 0], dim=-1)

    mask = (
        (axial_error <= float(axial_tolerance))
        & (signed_axial_error <= float(axial_overtravel_tolerance))
        & (radial_error <= float(radial_tolerance))
        & (plug_angle_error <= float(plug_angle_tolerance))
        & (latch_angle_error <= float(latch_angle_tolerance))
        & (plug_spatial_speed <= float(maximum_plug_spatial_speed))
    )
    return RJ45SuccessResult(
        mask=mask,
        signed_axial_error=signed_axial_error,
        axial_error=axial_error,
        radial_error=radial_error,
        plug_angle_error=plug_angle_error,
        latch_angle_error=latch_angle_error,
        plug_spatial_speed=plug_spatial_speed,
    )


__all__ = ["RJ45_SUCCESS_PREDICATE_VERSION", "RJ45SuccessResult", "rj45_insertion_success"]
