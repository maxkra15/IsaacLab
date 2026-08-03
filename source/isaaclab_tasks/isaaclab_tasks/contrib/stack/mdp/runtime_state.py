# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed runtime state shared by the Franka stack reset MDP terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@dataclass
class StackResetRuntimeState:
    """Episode-local reset metadata shared across stack MDP managers."""

    row_ids: torch.Tensor
    recipes: torch.Tensor
    previous_recipes: torch.Tensor
    goal_pairs: torch.Tensor
    target_potentials: torch.Tensor
    continue_to_final: torch.Tensor
    held_cube_ids: torch.Tensor
    grasp_pair_ids: torch.Tensor
    role_to_cube: torch.Tensor
    initialized: torch.Tensor
    previous_initialized: torch.Tensor
    sample_counts: torch.Tensor


_LEGACY_ALIASES = {
    "stack_reset_row_ids": "row_ids",
    "stack_reset_recipes": "recipes",
    "stack_previous_reset_recipes": "previous_recipes",
    "stack_reset_goal_pairs": "goal_pairs",
    "stack_reset_target_potentials": "target_potentials",
    "stack_continue_to_final": "continue_to_final",
    "stack_reset_held_cube_ids": "held_cube_ids",
    "stack_reset_grasp_pair_ids": "grasp_pair_ids",
    "stack_reset_role_to_cube": "role_to_cube",
    "stack_reset_initialized": "initialized",
    "stack_previous_reset_initialized": "previous_initialized",
    "stack_reset_sample_counts": "sample_counts",
}


def create_stack_reset_runtime_state(
    env: ManagerBasedRLEnv,
    row_count: int,
) -> StackResetRuntimeState:
    """Create and attach one typed reset-state owner to an environment."""
    state = StackResetRuntimeState(
        row_ids=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        recipes=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        previous_recipes=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        goal_pairs=torch.ones(env.num_envs, dtype=torch.long, device=env.device),
        target_potentials=torch.ones(env.num_envs, dtype=torch.float32, device=env.device),
        continue_to_final=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        held_cube_ids=torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
        grasp_pair_ids=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        role_to_cube=torch.arange(3, dtype=torch.long, device=env.device).repeat(env.num_envs, 1),
        initialized=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        previous_initialized=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        sample_counts=torch.zeros(row_count, dtype=torch.long, device=env.device),
    )
    env.stack_reset_state = state
    # Preserve the branch's initial public tensor names during a deprecation
    # window. All task implementation code uses the typed owner above; these
    # aliases reference the same tensors and therefore cannot diverge.
    for alias, field_name in _LEGACY_ALIASES.items():
        setattr(env, alias, getattr(state, field_name))
    return state


def get_stack_reset_runtime_state(env: ManagerBasedRLEnv) -> StackResetRuntimeState:
    """Return typed reset state, adapting the initial tensor aliases if needed."""
    state = getattr(env, "stack_reset_state", None)
    if state is not None:
        return state
    fields = {}
    for alias, field_name in _LEGACY_ALIASES.items():
        if field_name == "grasp_pair_ids" and not hasattr(env, alias):
            fields[field_name] = torch.zeros_like(env.stack_reset_held_cube_ids)
        else:
            fields[field_name] = getattr(env, alias)
    state = StackResetRuntimeState(**fields)
    env.stack_reset_state = state
    env.stack_reset_grasp_pair_ids = state.grasp_pair_ids
    return state
