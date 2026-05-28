# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation, reward, and termination helpers for the waterhose robot demo."""

from __future__ import annotations

import torch

from .manager import NewtonWaterhoseManager


def _runtime_token() -> tuple[tuple[int, int], ...]:
    return tuple(
        (id(runtime), int(getattr(runtime, "frame_count", -1))) for runtime in NewtonWaterhoseManager.get_runtimes()
    )


def _policy_state(env) -> dict[str, torch.Tensor]:
    token = _runtime_token()
    cached = getattr(env, "_waterhose_policy_state_cache", None)
    if cached is None or cached[0] != token:
        state = {
            name: torch.as_tensor(value, device=env.device, dtype=torch.float32)
            for name, value in NewtonWaterhoseManager.get_policy_state().items()
        }
        setattr(env, "_waterhose_policy_state_cache", (token, state))
        return state
    return cached[1]


def _subtask_signals(env) -> dict[str, torch.Tensor]:
    token = _runtime_token()
    cached = getattr(env, "_waterhose_subtask_signal_cache", None)
    if cached is None or cached[0] != token:
        signals = {
            name: torch.as_tensor(value, device=env.device, dtype=torch.bool)
            for name, value in NewtonWaterhoseManager.get_subtask_term_signals().items()
        }
        setattr(env, "_waterhose_subtask_signal_cache", (token, signals))
        return signals
    return cached[1]


def phase(env) -> torch.Tensor:
    """Current scripted state-machine phase index."""
    return _policy_state(env)["phase"].reshape(env.num_envs, 1)


def sim_time(env) -> torch.Tensor:
    """Local Newton simulation time [s]."""
    return _policy_state(env)["sim_time"].reshape(env.num_envs, 1)


def plug_pose(env) -> torch.Tensor:
    """Plug/head pose as xyz + xyzw."""
    return _policy_state(env)["plug_pose"].reshape(env.num_envs, 7)


def tip_pose(env) -> torch.Tensor:
    """Cable tip capsule pose as xyz + xyzw."""
    return _policy_state(env)["tip_pose"].reshape(env.num_envs, 7)


def right_ee_pose(env) -> torch.Tensor:
    """Right gripper end-effector pose as xyz + xyzw."""
    return _policy_state(env)["right_ee_pose"].reshape(env.num_envs, 7)


def finite(env) -> torch.Tensor:
    """Whether all primary Newton state buffers are finite."""
    return _policy_state(env)["finite"].reshape(env.num_envs, 1)


def _subtask_signal(env, name: str) -> torch.Tensor:
    return _subtask_signals(env)[name]


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
