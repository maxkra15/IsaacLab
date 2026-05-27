# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation and termination helpers for the reference waterhose robot demo wrapper."""

from __future__ import annotations

import numpy as np
import torch


def _example(env):
    example = getattr(env, "reference_demo", None)
    if example is None:
        raise RuntimeError("WaterhoseRobotDemoEnv.reference_demo has not been initialized.")
    return example


def _body_pose(model, state, short_name: str) -> np.ndarray:
    labels = getattr(model, "body_label", [])
    suffix = "/" + short_name
    for body_id, label in enumerate(labels):
        if label == short_name or str(label).endswith(suffix):
            return state.body_q.numpy()[body_id]
    raise RuntimeError(f"Body {short_name!r} not found in Newton model.")


def phase(env) -> torch.Tensor:
    """Current scripted state-machine phase index."""
    ex = _example(env)
    task_idx = int(ex.sm_task_idx.numpy()[0]) if hasattr(ex, "sm_task_idx") else 0
    task_schedule = ex.sm_task_schedule.numpy() if hasattr(ex, "sm_task_schedule") else np.asarray([0])
    task_value = int(task_schedule[min(task_idx, len(task_schedule) - 1)])
    return torch.full((env.num_envs, 1), float(task_value), device=env.device)


def sim_time(env) -> torch.Tensor:
    """Reference demo simulation time [s]."""
    ex = _example(env)
    return torch.full((env.num_envs, 1), float(getattr(ex, "sim_time", 0.0)), device=env.device)


def plug_pose(env) -> torch.Tensor:
    """Plug/head pose as xyz + xyzw."""
    ex = _example(env)
    body_id = int(getattr(ex, "cable_head_body_idx", 0))
    pose = ex.vbd_state_0.body_q.numpy()[body_id]
    return torch.as_tensor(pose, device=env.device, dtype=torch.float32).reshape(1, 7).repeat(env.num_envs, 1)


def tip_pose(env) -> torch.Tensor:
    """Tip capsule body pose as xyz + xyzw."""
    ex = _example(env)
    body_id = int(getattr(ex, "tip_capsule_body_idx", 0))
    pose = ex.vbd_state_0.body_q.numpy()[body_id]
    return torch.as_tensor(pose, device=env.device, dtype=torch.float32).reshape(1, 7).repeat(env.num_envs, 1)


def right_ee_pose(env) -> torch.Tensor:
    """Right gripper end-effector pose as xyz + xyzw."""
    ex = _example(env)
    pose = _body_pose(ex.mujoco_model, ex.state_0, "right_gripper_end_effector")
    return torch.as_tensor(pose, device=env.device, dtype=torch.float32).reshape(1, 7).repeat(env.num_envs, 1)


def finite(env) -> torch.Tensor:
    """Whether all primary Newton state buffers are finite."""
    ex = _example(env)
    ok = (
        np.isfinite(ex.state_0.body_q.numpy()).all()
        and np.isfinite(ex.state_0.body_qd.numpy()).all()
        and np.isfinite(ex.vbd_state_0.body_q.numpy()).all()
        and np.isfinite(ex.vbd_state_0.body_qd.numpy()).all()
    )
    return torch.full((env.num_envs, 1), float(ok), device=env.device)


def alive(env) -> torch.Tensor:
    """Zero-weight placeholder reward."""
    return finite(env).squeeze(-1)


def done(env) -> torch.Tensor:
    """Terminate when the finite-state-machine reaches DONE or max frames are exhausted."""
    ex = _example(env)
    max_steps = int(getattr(env.cfg, "max_demo_steps", 0))
    timed_out = max_steps > 0 and int(getattr(ex, "frame_count", 0)) >= max_steps
    is_done = False
    if hasattr(ex, "sm_task_idx") and hasattr(ex, "sm_task_schedule"):
        idx = int(ex.sm_task_idx.numpy()[0])
        schedule = ex.sm_task_schedule.numpy()
        if idx < len(schedule):
            # TaskType.DONE is 18 in the reference script.
            is_done = int(schedule[idx]) == 18
    return torch.full((env.num_envs,), bool(timed_out or is_done), device=env.device, dtype=torch.bool)

