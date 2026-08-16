# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical success, safety, and local curriculum progress terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

if TYPE_CHECKING:
    from ..rj45_env import FrankaRJ45InsertionEnv


def nonfinite_failure(env: FrankaRJ45InsertionEnv) -> torch.Tensor:
    task_finite = torch.isfinite(env.task_body_pose_e()).all(dim=(1, 2))
    task_finite &= torch.isfinite(env.task_body_velocity()).all(dim=(1, 2))
    robot_finite = torch.isfinite(env._robot.data.joint_pos.torch).all(dim=-1)
    robot_finite &= torch.isfinite(env._robot.data.joint_vel.torch).all(dim=-1)
    return ~(task_finite & robot_finite)


def task_out_of_bounds(env: FrankaRJ45InsertionEnv) -> torch.Tensor:
    task_pose = env.task_body_pose_e()
    task_velocity = env.task_body_velocity()
    plug = task_pose[:, 0, :3]
    plug_ok = ((plug >= env._task_workspace_lower) & (plug <= env._task_workspace_upper)).all(dim=-1)
    plug_velocity_ok = torch.linalg.vector_norm(task_velocity[:, 0], dim=-1) <= float(env.cfg.max_plug_spatial_speed)
    body_position_ok = (
        (task_pose[..., :3] >= env._task_body_workspace_lower) & (task_pose[..., :3] <= env._task_body_workspace_upper)
    ).all(dim=(1, 2))
    body_linear_speed_ok = torch.linalg.vector_norm(task_velocity[..., :3], dim=-1).amax(dim=-1) <= float(
        env.cfg.max_task_body_linear_speed
    )
    body_angular_speed_ok = torch.linalg.vector_norm(task_velocity[..., 3:], dim=-1).amax(dim=-1) <= float(
        env.cfg.max_task_body_angular_speed
    )
    return ~(plug_ok & plug_velocity_ok & body_position_ok & body_angular_speed_ok & body_linear_speed_ok)


def lost_grasp(env: FrankaRJ45InsertionEnv, minimum_episode_steps: int = 3) -> torch.Tensor:
    """Terminate when the plug has separated from the demonstrated TCP grasp."""
    distance = torch.linalg.vector_norm(env.tcp_pose_e()[:, :3] - env.plug_grasp_position_e(), dim=-1)
    gripper = env.action_manager.get_term("gripper_action")
    lost = (distance > float(env.cfg.max_tcp_grasp_distance)) | ~gripper.bilateral_contact
    return lost & (env.episode_length_buf >= int(minimum_episode_steps))


def stable_insertion_success(env: FrankaRJ45InsertionEnv) -> torch.Tensor:
    """Require a fully seated, slow, popped-latch state for a fixed dwell."""
    instantaneous = env.insertion_success_mask()
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


def unsuccessful_time_out(env: FrankaRJ45InsertionEnv) -> torch.Tensor:
    return (env.episode_length_buf >= env.max_episode_length) & ~env.episode_succeeded


class InsertionResetLearningProgress(ManagerTermBase):
    """Latch row-local insertion progress as curriculum evidence, never as reward."""

    def __init__(self, cfg: TerminationTermCfg, env: FrankaRJ45InsertionEnv):
        super().__init__(cfg, env)
        self.ever_success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.new_success = torch.zeros_like(self.ever_success)
        self._baseline_error = torch.zeros(env.num_envs, device=env.device)
        self._required_improvement = torch.zeros_like(self._baseline_error)
        self._no_termination = torch.zeros_like(self.ever_success)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        rows = self._env.reset_dataset_row_id[selected].clamp_min(0)
        states = self._env._reset_dataset_states
        self._baseline_error[selected] = self._env.scalar_goal_error()[selected]
        self._required_improvement[selected] = states["progress_threshold"][rows]
        self.ever_success[selected] = False
        self.new_success[selected] = False

    def __call__(
        self,
        env: FrankaRJ45InsertionEnv,
        minimum_episode_steps: int = 3,
    ) -> torch.Tensor:
        improvement = self._baseline_error - env.scalar_goal_error()
        reached = improvement >= self._required_improvement
        reached &= env.episode_length_buf >= int(minimum_episode_steps)
        reached &= ~env.termination_manager.terminated
        reached |= env.episode_succeeded
        self.new_success.copy_(reached & ~self.ever_success)
        self.ever_success |= reached
        return self._no_termination
