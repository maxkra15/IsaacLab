# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP helpers for the ADMM-backed waterhose task."""

from __future__ import annotations

import torch

from .admm_manager import NewtonWaterhoseAdmmManager


def _pose_tensor(env, pose) -> torch.Tensor:
    tensor = torch.as_tensor(pose, dtype=torch.float32, device=env.device)
    if tensor.ndim == 1:
        return tensor.reshape(1, 7).repeat(env.num_envs, 1)
    if tensor.shape[0] == env.num_envs:
        return tensor.reshape(env.num_envs, 7)
    return tensor[:1].reshape(1, 7).repeat(env.num_envs, 1)


def sim_time(env) -> torch.Tensor:
    return torch.full((env.num_envs, 1), float(NewtonWaterhoseAdmmManager.get_sim_time()), device=env.device)


def phase(env) -> torch.Tensor:
    return torch.full((env.num_envs, 1), float(NewtonWaterhoseAdmmManager.current_phase()), device=env.device)


def right_ee_pose(env) -> torch.Tensor:
    if hasattr(NewtonWaterhoseAdmmManager, "get_right_ee_poses"):
        return _pose_tensor(env, NewtonWaterhoseAdmmManager.get_right_ee_poses())
    return _pose_tensor(env, NewtonWaterhoseAdmmManager.get_right_ee_pose())


def plug_pose(env) -> torch.Tensor:
    if hasattr(NewtonWaterhoseAdmmManager, "get_plug_poses"):
        return _pose_tensor(env, NewtonWaterhoseAdmmManager.get_plug_poses())
    return _pose_tensor(env, NewtonWaterhoseAdmmManager.get_plug_pose())


def tip_pose(env) -> torch.Tensor:
    if hasattr(NewtonWaterhoseAdmmManager, "get_tip_poses"):
        return _pose_tensor(env, NewtonWaterhoseAdmmManager.get_tip_poses())
    return _pose_tensor(env, NewtonWaterhoseAdmmManager.get_tip_pose())


def finite(env) -> torch.Tensor:
    return torch.full((env.num_envs, 1), float(NewtonWaterhoseAdmmManager.is_finite()), device=env.device)


def alive(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def failed(env) -> torch.Tensor:
    return torch.full((env.num_envs,), not NewtonWaterhoseAdmmManager.is_finite(), dtype=torch.bool, device=env.device)
