# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage tracking and failures for full RJ45 pick-and-insert episodes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

if TYPE_CHECKING:
    from ..pick_insert_env import FrankaRJ45PickInsertEnv


def pick_insert_task_out_of_bounds(env: FrankaRJ45PickInsertEnv) -> torch.Tensor:
    """Reject escaped connectors/cable states across the larger table workspace."""
    task_pose = env.task_body_pose_e()
    task_velocity = env.task_body_velocity()
    plug = task_pose[:, env._plug_task_body_index, :3]
    plug_velocity = task_velocity[:, env._plug_task_body_index]
    plug_ok = ((plug >= env._task_workspace_lower) & (plug <= env._task_workspace_upper)).all(dim=-1)
    plug_speed_ok = torch.linalg.vector_norm(plug_velocity, dim=-1) <= float(env.cfg.max_plug_spatial_speed)
    body_position_ok = (
        (task_pose[..., :3] >= env._task_body_workspace_lower) & (task_pose[..., :3] <= env._task_body_workspace_upper)
    ).all(dim=(1, 2))
    body_linear_speed_ok = torch.linalg.vector_norm(task_velocity[..., :3], dim=-1).amax(dim=-1) <= float(
        env.cfg.max_task_body_linear_speed
    )
    body_angular_speed_ok = torch.linalg.vector_norm(task_velocity[..., 3:], dim=-1).amax(dim=-1) <= float(
        env.cfg.max_task_body_angular_speed
    )
    return ~(plug_ok & plug_speed_ok & body_position_ok & body_linear_speed_ok & body_angular_speed_ok)


def arm_target_tracking_failure(env: FrankaRJ45PickInsertEnv) -> torch.Tensor:
    """Terminate when the persistent arm target leaves its bounded tracking envelope."""
    action = env.action_manager.get_term("arm_action")
    error = action.target_tracking_error
    return ~torch.isfinite(error).all(dim=-1) | action.tracking_error_violation.any(dim=-1)


class PickInsertStageContext(ManagerTermBase):
    """Track grasp acquisition and monotonic manipulation stages without terminating."""

    def __init__(self, cfg: TerminationTermCfg, env: FrankaRJ45PickInsertEnv):
        super().__init__(cfg, env)
        self.ever_grasped = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.new_grasp = torch.zeros_like(self.ever_grasped)
        self.proxy_contact = torch.zeros_like(self.ever_grasped)
        self.current_grasp = torch.zeros_like(self.ever_grasped)
        self.maximum_stage = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.loss_count = torch.zeros_like(self.maximum_stage)
        self._no_termination = torch.zeros_like(self.ever_grasped)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        rows = self._env.reset_dataset_row_id[selected].clamp_min(0)
        starts_grasped = self._env._reset_dataset_states["starts_grasped"][rows]
        self.ever_grasped[selected] = starts_grasped
        self.new_grasp[selected] = False
        # Contact buffers still describe the pre-reset physics step until the
        # next solve. Do not expose that stale contact as part of the new row.
        self.proxy_contact[selected] = False
        self.current_grasp[selected] = False
        plug_goal_distance = torch.linalg.vector_norm(self._env.plug_goal_translation_error_local()[selected], dim=-1)
        initial_stage = starts_grasped.long()
        initial_stage += (starts_grasped & (plug_goal_distance <= self._env.cfg.transport_stage_distance_m)).long()
        initial_stage += (starts_grasped & (plug_goal_distance <= self._env.cfg.preinsert_stage_distance_m)).long()
        initial_stage += (starts_grasped & self._env.insertion_success_mask()[selected]).long()
        self.maximum_stage[selected] = initial_stage
        self.loss_count[selected] = 0

    def __call__(self, env: FrankaRJ45PickInsertEnv) -> torch.Tensor:
        gripper = env.action_manager.get_term("gripper_action")
        tcp_pose = env.tcp_pose_e()
        tcp_distance = torch.linalg.vector_norm(tcp_pose[:, :3] - env.plug_grasp_position_e(), dim=-1)
        acquisition_tolerance = float(env.cfg.grasp_acquisition_axis_tolerance_rad)
        retention_tolerance = float(env.cfg.grasp_retention_axis_tolerance_rad)
        alignment_tolerance = acquisition_tolerance + self.ever_grasped.to(tcp_distance.dtype) * (
            retention_tolerance - acquisition_tolerance
        )
        contact_aligned = env.grasp_contact_alignment_mask(
            alignment_tolerance,
            tcp_orientation_xyzw=tcp_pose[:, 3:7],
            proxy_contact_mask_out=self.proxy_contact,
        )
        self.current_grasp.copy_(
            gripper.bilateral_contact & contact_aligned & (tcp_distance <= float(env.cfg.max_tcp_grasp_distance))
        )
        acquired = self.current_grasp & (tcp_distance <= float(env.cfg.grasp_acquisition_distance_m))
        self.new_grasp.copy_(acquired & ~self.ever_grasped)
        self.ever_grasped |= acquired

        lost_contact = self.ever_grasped & ~self.current_grasp
        self.loss_count[:] = torch.where(lost_contact, self.loss_count + 1, 0)

        plug_goal_distance = torch.linalg.vector_norm(env.plug_goal_translation_error_local(), dim=-1)
        stage = self.ever_grasped.long()
        stage += (self.ever_grasped & (plug_goal_distance <= env.cfg.transport_stage_distance_m)).long()
        stage += (self.ever_grasped & (plug_goal_distance <= env.cfg.preinsert_stage_distance_m)).long()
        stage += (self.ever_grasped & env.insertion_success_mask()).long()
        self.maximum_stage.copy_(torch.maximum(self.maximum_stage, stage))
        return self._no_termination


def lost_acquired_grasp(env: FrankaRJ45PickInsertEnv, minimum_episode_steps: int = 5) -> torch.Tensor:
    """Terminate only after an acquired/demonstrated grasp is persistently lost."""
    tracker = env.pick_insert_stage_tracker()
    return (
        tracker.ever_grasped
        & (tracker.loss_count >= int(env.cfg.grasp_loss_grace_steps))
        & (env.episode_length_buf >= int(minimum_episode_steps))
    )


def stable_pick_insert_success(env: FrankaRJ45PickInsertEnv) -> torch.Tensor:
    """Require a prior physical grasp before accepting stable insertion."""
    tracker = env.pick_insert_stage_tracker()
    instantaneous = tracker.ever_grasped & env.insertion_success_mask()
    env._success_dwell_count[:] = torch.where(
        instantaneous,
        torch.clamp(env._success_dwell_count + 1, max=env._success_dwell_steps),
        0,
    )
    success = env._success_dwell_count >= env._success_dwell_steps
    # Earlier failure terms take precedence over a coincident geometry match.
    success &= ~env.termination_manager.terminated
    env.episode_succeeded |= success
    return success


class PickInsertResetLearningProgress(ManagerTermBase):
    """Record local stage/error improvement for adaptive reset sampling only."""

    def __init__(self, cfg: TerminationTermCfg, env: FrankaRJ45PickInsertEnv):
        super().__init__(cfg, env)
        self.ever_success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.new_success = torch.zeros_like(self.ever_success)
        self._baseline_stage = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self._baseline_error = torch.zeros(env.num_envs, device=env.device)
        self._required_improvement = torch.zeros_like(self._baseline_error)
        self._no_termination = torch.zeros_like(self.ever_success)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        rows = self._env.reset_dataset_row_id[selected].clamp_min(0)
        tracker = self._env.pick_insert_stage_tracker()
        self._baseline_stage[selected] = tracker.maximum_stage[selected]
        self._baseline_error[selected] = self._env.phase_progress_error()[selected]
        self._required_improvement[selected] = self._env._reset_dataset_states["progress_threshold"][rows]
        self.ever_success[selected] = False
        self.new_success[selected] = False

    def __call__(self, env: FrankaRJ45PickInsertEnv, minimum_episode_steps: int = 3) -> torch.Tensor:
        tracker = env.pick_insert_stage_tracker()
        stage_advanced = tracker.maximum_stage > self._baseline_stage
        improved = (self._baseline_error - env.phase_progress_error()) >= self._required_improvement
        reached = (stage_advanced | improved) & (env.episode_length_buf >= int(minimum_episode_steps))
        reached &= ~env.termination_manager.terminated
        reached |= env.episode_succeeded
        self.new_success.copy_(reached & ~self.ever_success)
        self.ever_success |= reached
        return self._no_termination


__all__ = [
    "PickInsertResetLearningProgress",
    "PickInsertStageContext",
    "arm_target_tracking_failure",
    "lost_acquired_grasp",
    "pick_insert_task_out_of_bounds",
    "stable_pick_insert_success",
]
