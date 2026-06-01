# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP helpers for the coupled waterhose task."""

from __future__ import annotations

import torch

from .coupled_manager import NewtonWaterhoseCoupledManager


def _pose_tensor(env, pose) -> torch.Tensor:
    tensor = torch.as_tensor(pose, dtype=torch.float32, device=env.device)
    if tensor.ndim == 1:
        return tensor.reshape(1, 7).repeat(env.num_envs, 1)
    if tensor.shape[0] == env.num_envs:
        return tensor.reshape(env.num_envs, 7)
    return tensor[:1].reshape(1, 7).repeat(env.num_envs, 1)


def sim_time(env) -> torch.Tensor:
    return torch.full((env.num_envs, 1), float(NewtonWaterhoseCoupledManager.get_sim_time()), device=env.device)


def phase(env) -> torch.Tensor:
    if hasattr(NewtonWaterhoseCoupledManager, "current_phases"):
        phases = torch.as_tensor(NewtonWaterhoseCoupledManager.current_phases(), dtype=torch.float32, device=env.device)
        return phases.reshape(-1, 1)[: env.num_envs]
    return torch.full((env.num_envs, 1), float(NewtonWaterhoseCoupledManager.current_phase()), device=env.device)


def right_ee_pose(env) -> torch.Tensor:
    if hasattr(NewtonWaterhoseCoupledManager, "get_right_ee_poses"):
        return _pose_tensor(env, NewtonWaterhoseCoupledManager.get_right_ee_poses())
    return _pose_tensor(env, NewtonWaterhoseCoupledManager.get_right_ee_pose())


def plug_pose(env) -> torch.Tensor:
    if hasattr(NewtonWaterhoseCoupledManager, "get_plug_poses"):
        return _pose_tensor(env, NewtonWaterhoseCoupledManager.get_plug_poses())
    return _pose_tensor(env, NewtonWaterhoseCoupledManager.get_plug_pose())


def tip_pose(env) -> torch.Tensor:
    if hasattr(NewtonWaterhoseCoupledManager, "get_tip_poses"):
        return _pose_tensor(env, NewtonWaterhoseCoupledManager.get_tip_poses())
    return _pose_tensor(env, NewtonWaterhoseCoupledManager.get_tip_pose())


def finite(env) -> torch.Tensor:
    if hasattr(NewtonWaterhoseCoupledManager, "finite_mask"):
        return torch.as_tensor(NewtonWaterhoseCoupledManager.finite_mask(), dtype=torch.float32, device=env.device).reshape(
            -1, 1
        )[: env.num_envs]
    return torch.full((env.num_envs, 1), float(NewtonWaterhoseCoupledManager.is_finite()), device=env.device)


def alive(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def failed(env) -> torch.Tensor:
    return torch.full((env.num_envs,), not NewtonWaterhoseCoupledManager.is_finite(), dtype=torch.bool, device=env.device)


def _subtask_signal(env, name: str) -> torch.Tensor:
    signals = NewtonWaterhoseCoupledManager.get_subtask_term_signals()
    values = signals.get(name)
    if values is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return torch.as_tensor(values, dtype=torch.bool, device=env.device).reshape(-1)[: env.num_envs]


def approach_done(env) -> torch.Tensor:
    return _subtask_signal(env, "approach")


def grasp_done(env) -> torch.Tensor:
    return _subtask_signal(env, "grasp")


def align_done(env) -> torch.Tensor:
    return _subtask_signal(env, "align")


def insert_done(env) -> torch.Tensor:
    return _subtask_signal(env, "insert")
