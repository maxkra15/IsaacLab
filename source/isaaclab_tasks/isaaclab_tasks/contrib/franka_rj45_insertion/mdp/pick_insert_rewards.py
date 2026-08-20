# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-aware progress shaping for reach, grasp, transport, and insertion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from ..pick_insert_env import FrankaRJ45PickInsertEnv


def pick_insert_potential(env: FrankaRJ45PickInsertEnv) -> torch.Tensor:
    """Return a bounded stage-continuous task potential.

    Reach shaping dominates before contact. Once a physical bilateral grasp is
    acquired, transport and insertion precision replace it. The potential is
    used only through signed differences, so holding a convenient state cannot
    accumulate reward.
    """
    tcp = env.tcp_pose_e()
    tcp_distance = torch.nan_to_num(
        torch.linalg.vector_norm(tcp[:, :3] - env.plug_grasp_position_e(), dim=-1),
        nan=torch.inf,
        posinf=torch.inf,
        neginf=torch.inf,
    )
    grasp_target = env.desired_tcp_grasp_pose_e()
    grasp_rotation_error = math_utils.quat_unique(
        math_utils.quat_mul(math_utils.quat_conjugate(grasp_target[:, 3:7]), tcp[:, 3:7])
    )
    grasp_orientation_error = torch.nan_to_num(
        torch.linalg.vector_norm(math_utils.axis_angle_from_quat(grasp_rotation_error), dim=-1),
        nan=torch.inf,
        posinf=torch.inf,
        neginf=torch.inf,
    )
    reach_translation = torch.exp(-tcp_distance / float(env.cfg.reach_reward_scale_m))
    reach_orientation = torch.exp(-grasp_orientation_error / float(env.cfg.reach_orientation_reward_scale_rad))
    orientation_weight = float(env.cfg.reach_orientation_reward_weight)
    reach = (1.0 - orientation_weight) * reach_translation + orientation_weight * reach_orientation
    tracker = env.pick_insert_stage_tracker()
    # Only the termination tracker may unlock post-grasp shaping. Raw finger
    # drive deflection can also come from the table or other scene geometry.
    grasp = tracker.ever_grasped.float()
    plug_goal_distance = torch.nan_to_num(
        torch.linalg.vector_norm(env.plug_goal_translation_error_local(), dim=-1),
        nan=torch.inf,
        posinf=torch.inf,
        neginf=torch.inf,
    )
    transport = torch.exp(-plug_goal_distance / float(env.cfg.transport_reward_scale_m))
    insertion_error = torch.nan_to_num(env.scalar_goal_error(), nan=torch.inf, posinf=torch.inf, neginf=torch.inf)
    insertion = torch.exp(-insertion_error / float(env.cfg.insertion_reward_scale))
    return torch.nan_to_num(reach + grasp * (1.0 + 1.5 * transport + 2.5 * insertion))


class PickInsertProgressReward(ManagerTermBase):
    """Signed potential difference across the complete manipulation sequence."""

    def __init__(self, cfg: RewardTermCfg, env: FrankaRJ45PickInsertEnv):
        super().__init__(cfg, env)
        self._previous = pick_insert_potential(env).clone()
        self._needs_baseline = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def __call__(self, env: FrankaRJ45PickInsertEnv) -> torch.Tensor:
        value = pick_insert_potential(env)
        delta = value - self._previous
        # ManagerBasedRLEnv resets rewards before terminations.  The stage
        # tracker therefore still describes the previous episode when
        # ``reset()`` runs.  Establish the new baseline on the first policy
        # step, after termination terms have consumed the new reset row, and
        # suppress only that otherwise-spurious cross-episode delta.
        delta = torch.where(self._needs_baseline, torch.zeros_like(delta), delta)
        self._previous.copy_(value)
        self._needs_baseline.zero_()
        return delta / max(float(env.step_dt), 1.0e-6)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        self._needs_baseline[selected] = True


def grasp_acquisition_bonus(env: FrankaRJ45PickInsertEnv) -> torch.Tensor:
    """One-shot event when the policy first establishes a bilateral grasp."""
    return env.pick_insert_stage_tracker().new_grasp.float() / max(float(env.step_dt), 1.0e-6)


__all__ = ["PickInsertProgressReward", "grasp_acquisition_bonus", "pick_insert_potential"]
