# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed episode state shared by the juggling MDP terms."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    # These fields extend the public runtime state without changing its
    # original constructor. Direct construction remains supported, while the
    # reset initializer owns their episode-specific values.
    start_item_ids: torch.Tensor = field(init=False)
    canonical_start: torch.Tensor = field(init=False)
    release_origins_xy: torch.Tensor = field(init=False)
    local_goal_ids: torch.Tensor = field(init=False)
    height_success: torch.Tensor = field(init=False)
    new_height_success: torch.Tensor = field(init=False)
    preload_assist_start: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        """Initialize fields added after the original public constructor."""
        self.start_item_ids = torch.zeros_like(self.row_ids)
        self.canonical_start = torch.zeros_like(self.static_held_start)
        self.release_origins_xy = self.release_heights.new_zeros((self.row_ids.numel(), 2))
        self.local_goal_ids = torch.full_like(self.row_ids, -1)
        self.height_success = torch.zeros_like(self.local_success)
        self.new_height_success = torch.zeros_like(self.new_local_success)
        # Preserve the original constructor's HELD_PRETHROW/STABLE_CATCH
        # behavior for direct callers. The reset initializer may explicitly
        # disable this latch for policy-owned rows.
        self.preload_assist_start = (self.current_phases == 0) | (self.current_phases == 7)


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
    preload_assist_start: torch.Tensor | None = None,
    item_ids: torch.Tensor | None = None,
    canonical_start: torch.Tensor | None = None,
    release_origins_xy: torch.Tensor | None = None,
    local_goal_ids: torch.Tensor | None = None,
) -> None:
    """Initialize reset metadata without crediting the authored state as success.

    Args:
        state: Runtime state to update.
        env_ids: Environment indices, shape ``[N]``.
        phases: Authored reset phases, shape ``[N]``.
        release_heights: Authored launch-reference heights [m], shape ``[N]``. A task may
            configure the progress context to replace this with the last supported hand height.
        static_held_start: Whether each row is a canonical held-at-rest reset, shape ``[N]``.
        preload_assist_start: Whether the reset row requires a preload target anchor, shape ``[N]``.
            When omitted, held-prethrow and stable-catch phases retain the legacy assisted behavior.
        item_ids: Semantic reset-curriculum item IDs, shape ``[N]``. Defaults to physical phase IDs.
        canonical_start: Whether each reset must complete the uninterrupted full cycle, shape ``[N]``.
            Defaults to all held-prethrow starts for backward compatibility.
        release_origins_xy: Authored launch-reference positions [m], shape ``[N, 2]``. A task may
            update held starts to the last supported tool position. Defaults to the world-local origin.
        local_goal_ids: Fresh physical-event goal IDs, shape ``[N]``. A value of -1 selects the
            legacy phase-derived goal.
    """
    if (
        env_ids.ndim != 1
        or phases.shape != env_ids.shape
        or release_heights.shape != env_ids.shape
        or static_held_start.shape != env_ids.shape
        or (preload_assist_start is not None and preload_assist_start.shape != env_ids.shape)
        or (item_ids is not None and item_ids.shape != env_ids.shape)
        or (canonical_start is not None and canonical_start.shape != env_ids.shape)
        or (release_origins_xy is not None and release_origins_xy.shape != (env_ids.numel(), 2))
        or (local_goal_ids is not None and local_goal_ids.shape != env_ids.shape)
    ):
        raise ValueError("Episode reset ids and metadata must have matching one-dimensional shapes.")
    if static_held_start.dtype != torch.bool or static_held_start.device != phases.device:
        raise TypeError("static_held_start must be a Boolean tensor on the phase tensor's device.")
    if preload_assist_start is None:
        preload_assist_start = (phases == 0) | (phases == 7)
    elif preload_assist_start.dtype != torch.bool or preload_assist_start.device != phases.device:
        raise TypeError("preload_assist_start must be a Boolean tensor on the phase tensor's device.")
    if item_ids is None:
        item_ids = phases
    elif (
        item_ids.dtype == torch.bool
        or item_ids.is_floating_point()
        or item_ids.is_complex()
        or item_ids.device != phases.device
    ):
        raise TypeError("item_ids must be an integer tensor on the phase tensor's device.")
    if canonical_start is None:
        canonical_start = phases == 0
    elif canonical_start.dtype != torch.bool or canonical_start.device != phases.device:
        raise TypeError("canonical_start must be a Boolean tensor on the phase tensor's device.")
    if release_origins_xy is None:
        release_origins_xy = release_heights.new_zeros((env_ids.numel(), 2))
    elif release_origins_xy.device != phases.device or not release_origins_xy.is_floating_point():
        raise TypeError("release_origins_xy must be a floating-point tensor on the phase tensor's device.")
    if local_goal_ids is None:
        local_goal_ids = torch.full_like(phases, -1)
    elif (
        local_goal_ids.dtype == torch.bool
        or local_goal_ids.is_floating_point()
        or local_goal_ids.is_complex()
        or local_goal_ids.device != phases.device
    ):
        raise TypeError("local_goal_ids must be an integer tensor on the phase tensor's device.")
    preload_eligible = (phases == 0) | (phases == 7)
    # This initializer is on the vectorized reset hot path. Keep the internal
    # authoring invariant on-device instead of synchronizing CUDA with Python.
    torch._assert_async(
        ~(preload_assist_start & ~preload_eligible).any(),
        "Preload assistance is valid only for held-prethrow or stable-catch reset phases.",
    )
    torch._assert_async((item_ids >= 0).all(), "Reset curriculum item IDs must be non-negative.")
    torch._assert_async(
        ((local_goal_ids == -1) | ((local_goal_ids >= 0) & (local_goal_ids <= 3))).all(),
        "Physical local-goal ids are outside the supported range.",
    )
    torch._assert_async(
        ~(canonical_start & (phases != 0)).any(),
        "Only held-prethrow reset rows may be canonical full-cycle starts.",
    )
    state.start_item_ids[env_ids] = item_ids
    state.start_phases[env_ids] = phases
    state.canonical_start[env_ids] = canonical_start
    state.current_phases[env_ids] = phases
    state.visited_phase_bits[env_ids] = torch.bitwise_left_shift(torch.ones_like(phases), phases)
    state.stable_catch_steps[env_ids] = 0
    state.release_clear_steps[env_ids] = 0
    state.release_heights[env_ids] = release_heights
    state.release_origins_xy[env_ids] = release_origins_xy
    state.local_goal_ids[env_ids] = local_goal_ids
    state.height_success[env_ids] = False
    state.new_height_success[env_ids] = False
    state.first_ascent_active[env_ids] = True
    state.seen_initial_ascent[env_ids] = (phases == 1) | (phases == 2)
    state.seen_release[env_ids] = (phases >= 1) & (phases <= 5)
    state.seen_apex[env_ids] = False
    state.static_held_start[env_ids] = static_held_start
    state.preload_assist_start[env_ids] = preload_assist_start
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
