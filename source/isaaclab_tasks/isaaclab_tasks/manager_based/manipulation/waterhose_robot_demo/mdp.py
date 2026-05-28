# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation, reward, and termination helpers for the waterhose robot demo."""

from __future__ import annotations

import torch

from .manager import NewtonWaterhoseManager


def _pose_tensor(env, pose) -> torch.Tensor:
    return torch.as_tensor(pose, device=env.device, dtype=torch.float32).reshape(1, 7).repeat(env.num_envs, 1)


def phase(env) -> torch.Tensor:
    """Current scripted state-machine phase index."""
    return torch.full((env.num_envs, 1), float(NewtonWaterhoseManager.current_phase()), device=env.device)


def sim_time(env) -> torch.Tensor:
    """Local Newton simulation time [s]."""
    return torch.full((env.num_envs, 1), float(NewtonWaterhoseManager.get_sim_time()), device=env.device)


def plug_pose(env) -> torch.Tensor:
    """Plug/head pose as xyz + xyzw."""
    return _pose_tensor(env, NewtonWaterhoseManager.get_plug_pose())


def tip_pose(env) -> torch.Tensor:
    """Cable tip capsule pose as xyz + xyzw."""
    return _pose_tensor(env, NewtonWaterhoseManager.get_tip_pose())


def right_ee_pose(env) -> torch.Tensor:
    """Right gripper end-effector pose as xyz + xyzw."""
    return _pose_tensor(env, NewtonWaterhoseManager.get_right_ee_pose())


def finite(env) -> torch.Tensor:
    """Whether all primary Newton state buffers are finite."""
    return torch.full((env.num_envs, 1), float(NewtonWaterhoseManager.is_finite()), device=env.device)


def alive(env) -> torch.Tensor:
    """Zero-weight placeholder reward for manager compatibility."""
    return finite(env).squeeze(-1)


def done(env) -> torch.Tensor:
    """Terminate when the scripted rollout finishes or the configured step cap is reached."""
    solver_cfg = getattr(getattr(getattr(env.cfg, "sim", None), "physics", None), "solver_cfg", None)
    max_steps = int(getattr(solver_cfg, "max_demo_steps", 0))
    return torch.full(
        (env.num_envs,),
        bool(NewtonWaterhoseManager.is_done(max_demo_steps=max_steps)),
        device=env.device,
        dtype=torch.bool,
    )


def reset_demo(env, env_ids) -> None:
    """Rebuild the local Newton demo state on full environment resets."""
    if env_ids is None or isinstance(env_ids, slice) or len(env_ids) >= env.num_envs:
        NewtonWaterhoseManager.reset_runtime()
