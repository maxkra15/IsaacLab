# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Role-explicit endpoint observations for the dual-rack RJ45 task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from ..dual_rack_env import FrankaRJ45DualRackInsertEnv


def _canonical_pose(pose: torch.Tensor) -> torch.Tensor:
    """Return finite xyz plus a unique xyzw quaternion representative."""
    pose = torch.nan_to_num(pose)
    return torch.cat((pose[..., :3], math_utils.quat_unique(pose[..., 3:7])), dim=-1)


def anchored_socket_pose_obs(env: FrankaRJ45DualRackInsertEnv) -> torch.Tensor:
    """Pose of the lower rack's occupied socket in the environment frame."""
    return _canonical_pose(env.anchored_socket_pose_e())


def anchored_plug_pose_obs(env: FrankaRJ45DualRackInsertEnv) -> torch.Tensor:
    """Pose of the cable end that starts and remains seated in the lower rack."""
    return _canonical_pose(env.anchored_plug_pose_e())


def anchored_cable_endpoint_error_obs(env: FrankaRJ45DualRackInsertEnv) -> torch.Tensor:
    """Pinned cable endpoint error in the anchored-plug frame [m]."""
    endpoint = env.anchored_cable_endpoint_position_e()
    target = env.anchored_cable_target_position_e()
    anchored_plug = env.anchored_plug_pose_e()
    local_error = math_utils.quat_apply_inverse(anchored_plug[:, 3:7], endpoint - target)
    return torch.nan_to_num(local_error).clamp_(-0.05, 0.05)


__all__ = [
    "anchored_cable_endpoint_error_obs",
    "anchored_plug_pose_obs",
    "anchored_socket_pose_obs",
]
