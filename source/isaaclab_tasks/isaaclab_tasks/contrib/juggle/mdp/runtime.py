# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed episode state shared by the juggling MDP terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@dataclass
class JuggleRuntimeState:
    """Episode-local reset assignment and phase progress."""

    row_ids: torch.Tensor
    start_phases: torch.Tensor
    current_phases: torch.Tensor
    visited_phase_bits: torch.Tensor
    stable_catch_steps: torch.Tensor
    release_clear_steps: torch.Tensor
    release_heights: torch.Tensor
    first_ascent_active: torch.Tensor
    seen_initial_ascent: torch.Tensor
    seen_release: torch.Tensor
    seen_apex: torch.Tensor
    static_held_start: torch.Tensor
    local_success: torch.Tensor
    new_local_success: torch.Tensor
    cycle_success: torch.Tensor
    new_cycle_success: torch.Tensor
    initialized: torch.Tensor


def create_juggle_runtime_state(env: ManagerBasedRLEnv) -> JuggleRuntimeState:
    """Create and attach the task's single typed runtime-state owner."""
    state = JuggleRuntimeState(
        row_ids=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        start_phases=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        current_phases=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        visited_phase_bits=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        stable_catch_steps=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        release_clear_steps=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        release_heights=torch.zeros(env.num_envs, dtype=torch.float32, device=env.device),
        first_ascent_active=torch.ones(env.num_envs, dtype=torch.bool, device=env.device),
        seen_initial_ascent=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        seen_release=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        seen_apex=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        static_held_start=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        local_success=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        new_local_success=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        cycle_success=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        new_cycle_success=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        initialized=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    env.juggle_runtime_state = state
    return state


def initialize_juggle_episode_state(
    state: JuggleRuntimeState,
    env_ids: torch.Tensor,
    phases: torch.Tensor,
    release_heights: torch.Tensor,
    static_held_start: torch.Tensor,
) -> None:
    """Initialize reset metadata without crediting the authored state as success.

    Args:
        state: Runtime state to update.
        env_ids: Environment indices, shape ``[N]``.
        phases: Authored reset phases, shape ``[N]``.
        release_heights: Launch-reference heights [m], shape ``[N]``.
        static_held_start: Whether each row is a canonical held-at-rest reset, shape ``[N]``.
    """
    if (
        env_ids.ndim != 1
        or phases.shape != env_ids.shape
        or release_heights.shape != env_ids.shape
        or static_held_start.shape != env_ids.shape
    ):
        raise ValueError("Episode reset ids and metadata must have matching one-dimensional shapes.")
    state.start_phases[env_ids] = phases
    state.current_phases[env_ids] = phases
    state.visited_phase_bits[env_ids] = torch.bitwise_left_shift(torch.ones_like(phases), phases)
    state.stable_catch_steps[env_ids] = 0
    state.release_clear_steps[env_ids] = 0
    state.release_heights[env_ids] = release_heights
    state.first_ascent_active[env_ids] = True
    state.seen_initial_ascent[env_ids] = (phases == 1) | (phases == 2)
    state.seen_release[env_ids] = (phases >= 1) & (phases <= 5)
    state.seen_apex[env_ids] = False
    state.static_held_start[env_ids] = static_held_start
    state.local_success[env_ids] = False
    state.new_local_success[env_ids] = False
    state.cycle_success[env_ids] = False
    state.new_cycle_success[env_ids] = False
    state.initialized[env_ids] = True


def get_juggle_runtime_state(env: ManagerBasedRLEnv) -> JuggleRuntimeState:
    """Return the runtime state created by the reset event."""
    state = getattr(env, "juggle_runtime_state", None)
    if state is None:
        raise AttributeError("Juggle runtime state is unavailable before the reset event is initialized.")
    return state
