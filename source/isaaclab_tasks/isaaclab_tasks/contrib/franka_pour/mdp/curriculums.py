# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success-driven curriculum for the Franka Pour task."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
    from ..pour_env import FrankaPourEnv


class PourCurriculum(ManagerTermBase):
    """Advance one shared stage-and-randomization frontier from completed episodes."""

    def __init__(self, cfg: CurriculumTermCfg, env: FrankaPourEnv):
        super().__init__(cfg, env)
        stage_count = len(env.cfg.curriculum_stage_names)
        if stage_count == 0 or len(env.cfg.curriculum_target_frac) != stage_count:
            raise ValueError("Curriculum stage names and target fractions must have equal nonzero length.")

        self.max_stage = stage_count - 1
        self.stage = int(env.cfg.curriculum_start_stage)
        if self.stage < 0 or self.stage > self.max_stage:
            raise ValueError(f"curriculum_start_stage must lie in [0, {self.max_stage}].")
        randomization_levels = env.cfg.curriculum_randomization_extent_levels
        if not randomization_levels:
            raise ValueError("curriculum_randomization_extent_levels must not be empty.")
        self.max_randomization_level = len(randomization_levels) - 1
        self.randomization_level = int(env.cfg.curriculum_randomization_start_level)
        if self.randomization_level < 0 or self.randomization_level > self.max_randomization_level:
            raise ValueError(f"curriculum_randomization_start_level must lie in [0, {self.max_randomization_level}].")

        self._success_window_size(env)
        self._success_window: deque[bool] = deque()
        self._success_count = 0
        self.success_rate = 0.0
        self.resets_in_stage = 0
        env.set_curriculum_stage(slice(None), self.stage)
        env.set_curriculum_randomization_level(slice(None), self.randomization_level)

    def __call__(
        self,
        env: FrankaPourEnv,
        env_ids: Sequence[int] | torch.Tensor | slice,
    ) -> dict[str, float]:
        """Update stage statistics and assign the next episodes."""
        ids = self._as_tensor(env, env_ids)
        window_size = self._success_window_size(env)
        self._trim_success_window(window_size)
        mean_peak_target_frac: float | None = None

        if ids.numel() > 0:
            # Count only completed episodes from the active frontier. This excludes the initial
            # reset, replay episodes, and asynchronous old-stage episodes after a promotion.
            completed = (env.episode_length_buf[ids] > 0) & (env.curriculum_stage[ids] == self.stage)
            if self.stage == self.max_stage:
                completed &= env.curriculum_randomization_level[ids] == self.randomization_level
            completed_ids = ids[completed]
            count = int(completed_ids.numel())
            if count > 0:
                outcomes = env.episode_succeeded[completed_ids].detach().to(device="cpu", dtype=torch.bool).tolist()
                self._update_success_window(outcomes, window_size)
                self.resets_in_stage += count
                mean_peak_target_frac = float(env.ep_max_target_frac[completed_ids].mean().item())

                promotion_threshold = float(env.cfg.curriculum_success_threshold)
                if self.stage == self.max_stage and self.randomization_level < self.max_randomization_level:
                    promotion_threshold = float(env.cfg.curriculum_randomization_promotion_threshold)
                frontier_mastered = (
                    not env.cfg.curriculum_freeze
                    and self.resets_in_stage >= window_size
                    and self.success_rate >= promotion_threshold
                )
                if frontier_mastered:
                    if self.stage < self.max_stage:
                        self.stage += 1
                        self._clear_stage_statistics()
                    elif self.randomization_level < self.max_randomization_level:
                        self.randomization_level += 1
                        self._clear_stage_statistics()

            env.set_curriculum_stage(ids, self.stage)
            env.set_curriculum_randomization_level(ids, self.randomization_level)
            replay_fraction = float(env.cfg.curriculum_previous_stage_replay_fraction)
            if not env.cfg.curriculum_freeze and self.stage > 0 and replay_fraction > 0.0:
                replay = torch.rand(ids.numel(), device=env.device) < replay_fraction
                if self.stage == self.max_stage and self.randomization_level > 0:
                    env.set_curriculum_randomization_level(ids[replay], self.randomization_level - 1)
                else:
                    env.set_curriculum_stage(ids[replay], self.stage - 1)

        mastered = (
            self.stage == self.max_stage
            and self.randomization_level == self.max_randomization_level
            and self.resets_in_stage >= window_size
            and self.success_rate >= env.cfg.curriculum_success_threshold
        )
        randomization_extent_fraction = float(env.cfg.curriculum_randomization_extent_levels[self.randomization_level])
        metrics = {
            "stage": float(self.stage),
            "randomization_level": float(self.randomization_level),
            "randomization_extent_fraction": randomization_extent_fraction,
            "eligible_bank_rows": float(env.curriculum_randomization_bank_size(self.randomization_level)),
            "eligible_source_cells": float(env.curriculum_randomization_source_cell_count(self.randomization_level)),
            "minimum_arm_variants_per_source": float(
                env.curriculum_randomization_minimum_variant_count(self.randomization_level)
            ),
            "success_rate": float(self.success_rate),
            "target_frac": float(env.cfg.curriculum_target_frac[self.stage]),
            "completed_episodes": float(self.resets_in_stage),
            "mastered": float(mastered),
        }
        if mean_peak_target_frac is not None:
            metrics["mean_peak_target_frac"] = mean_peak_target_frac
        return metrics

    @staticmethod
    def _success_window_size(env: FrankaPourEnv) -> int:
        """Return the configured number of recent frontier episodes used for promotion."""
        window_size = int(env.cfg.curriculum_min_resets_per_stage)
        if window_size <= 0:
            raise ValueError("curriculum_min_resets_per_stage must be positive.")
        return window_size

    def _update_success_window(self, outcomes: list[bool], window_size: int) -> None:
        """Append episode outcomes and update the exact recent-window success rate."""
        for outcome in outcomes:
            if len(self._success_window) == window_size:
                self._success_count -= int(self._success_window.popleft())
            outcome = bool(outcome)
            self._success_window.append(outcome)
            self._success_count += int(outcome)
        self.success_rate = self._success_count / len(self._success_window)

    def _trim_success_window(self, window_size: int) -> None:
        """Apply a runtime window-size change without retaining stale outcomes."""
        while len(self._success_window) > window_size:
            self._success_count -= int(self._success_window.popleft())
        self.success_rate = self._success_count / len(self._success_window) if self._success_window else 0.0

    def _clear_stage_statistics(self) -> None:
        """Clear frontier evidence after promotion to a new stage."""
        self._success_window.clear()
        self._success_count = 0
        self.success_rate = 0.0
        self.resets_in_stage = 0

    @staticmethod
    def _as_tensor(
        env: FrankaPourEnv,
        env_ids: Sequence[int] | torch.Tensor | slice,
    ) -> torch.Tensor:
        """Normalize manager environment indices to a one-dimensional device tensor."""
        if isinstance(env_ids, slice):
            return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
        return torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()
