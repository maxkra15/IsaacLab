# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation, reward, and termination helpers for the waterhose robot demo."""

from __future__ import annotations

import torch

from .manager import NewtonWaterhoseManager


def _as_env_tensor(env, value, shape_tail: tuple[int, ...]) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=env.device, dtype=torch.float32)
    return tensor.reshape(env.num_envs, *shape_tail)


def phase(env) -> torch.Tensor:
    """Current scripted state-machine phase index."""
    return _as_env_tensor(env, NewtonWaterhoseManager.current_phases(), (1,))


def sim_time(env) -> torch.Tensor:
    """Local Newton simulation time [s]."""
    return _as_env_tensor(env, NewtonWaterhoseManager.get_sim_times(), (1,))


def plug_pose(env) -> torch.Tensor:
    """Plug/head pose as xyz + xyzw."""
    return _as_env_tensor(env, NewtonWaterhoseManager.get_plug_poses(), (7,))


def tip_pose(env) -> torch.Tensor:
    """Cable tip capsule pose as xyz + xyzw."""
    return _as_env_tensor(env, NewtonWaterhoseManager.get_tip_poses(), (7,))


def right_ee_pose(env) -> torch.Tensor:
    """Right gripper end-effector pose as xyz + xyzw."""
    return _as_env_tensor(env, NewtonWaterhoseManager.get_right_ee_poses(), (7,))


def finite(env) -> torch.Tensor:
    """Whether all primary Newton state buffers are finite."""
    return _as_env_tensor(env, NewtonWaterhoseManager.finite_mask(), (1,))


def _subtask_signal(env, name: str) -> torch.Tensor:
    signals = NewtonWaterhoseManager.get_subtask_term_signals()
    return torch.as_tensor(signals[name], device=env.device, dtype=torch.bool)


def approach_done(env) -> torch.Tensor:
    """Whether the end effector has reached the plug approach segment."""
    return _subtask_signal(env, "approach")


def grasp_done(env) -> torch.Tensor:
    """Whether the plug is likely grasped."""
    return _subtask_signal(env, "grasp")


def align_done(env) -> torch.Tensor:
    """Whether the tip is aligned near the socket."""
    return _subtask_signal(env, "align")


def insert_done(env) -> torch.Tensor:
    """Whether the tip is inserted into the socket."""
    return _subtask_signal(env, "insert")


def alive(env) -> torch.Tensor:
    """Zero-weight placeholder reward for manager compatibility."""
    return finite(env).squeeze(-1)


def done(env) -> torch.Tensor:
    """Terminate when the scripted rollout finishes or the configured step cap is reached."""
    solver_cfg = getattr(getattr(getattr(env.cfg, "sim", None), "physics", None), "solver_cfg", None)
    max_steps = int(getattr(solver_cfg, "max_demo_steps", 0))
    return torch.as_tensor(
        NewtonWaterhoseManager.done_mask(max_demo_steps=max_steps),
        device=env.device,
        dtype=torch.bool,
    )


def success(env) -> torch.Tensor:
    """Task success flag used by standard recording and imitation-learning scripts."""
    return torch.as_tensor(
        NewtonWaterhoseManager.success_mask(),
        device=env.device,
        dtype=torch.bool,
    )


def reset_demo(env, env_ids) -> None:
    """Rebuild the local Newton demo state on full environment resets."""
    NewtonWaterhoseManager.reset_runtime(env_ids)
