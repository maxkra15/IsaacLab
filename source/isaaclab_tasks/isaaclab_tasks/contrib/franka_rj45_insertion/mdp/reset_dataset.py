# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success-aware replay of validated RJ45 near-goal reset rows."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers.manager_base import ManagerTermBase

from isaaclab_tasks.contrib.franka_pour.reset_sampler import ResetDatasetSamplerCfg, _ResetDatasetSampler

if TYPE_CHECKING:
    from ..rj45_env import FrankaRJ45InsertionEnv


class RJ45ResetDatasetCurriculum(ManagerTermBase):
    """Assign reset rows and adapt their probability from local progress outcomes."""

    def __init__(self, cfg: CurriculumTermCfg, env: FrankaRJ45InsertionEnv):
        super().__init__(cfg, env)
        row_count = int(env._reset_dataset_states["phase"].numel())
        sampler_cfg = env.cfg.reset_dataset_sampler.copy()
        if not isinstance(sampler_cfg, ResetDatasetSamplerCfg):
            raise TypeError("reset_dataset_sampler must be ResetDatasetSamplerCfg.")
        self._sampler = _ResetDatasetSampler(row_count, env.device, sampler_cfg)
        self._row_count = row_count
        self._metrics_cache = self._sampler.metrics()
        self._assignments_since_metrics = 0

    @staticmethod
    def _ids(env: FrankaRJ45InsertionEnv, env_ids: Sequence[int] | torch.Tensor | slice) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
        return torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()

    def __call__(
        self,
        env: FrankaRJ45InsertionEnv,
        env_ids: Sequence[int] | torch.Tensor | slice,
    ) -> dict[str, float]:
        ids = self._ids(env, env_ids)
        if ids.numel() == 0:
            return self._metrics_cache

        completed = (env.episode_length_buf[ids] > 0) & (env.reset_dataset_row_id[ids] >= 0)
        completed_ids = ids[completed]
        if completed_ids.numel() and not env.cfg.curriculum_freeze:
            progress = env.termination_manager.get_term_cfg("learning_progress_context").func
            self._sampler._record_validated(
                env.reset_dataset_row_id[completed_ids],
                progress.ever_success[completed_ids],
            )

        if env.cfg.curriculum_freeze or env.cfg.reset_dataset_sampling_mode == "uniform":
            rows = torch.randint(self._row_count, (ids.numel(),), device=env.device)
        else:
            rows = self._sampler._sample_with_uniform_replay(ids.numel())
        env.reset_dataset_row_id[ids] = rows

        self._assignments_since_metrics += ids.numel()
        if self._assignments_since_metrics >= env.num_envs:
            self._metrics_cache = self._sampler.metrics()
            self._assignments_since_metrics = 0
        return self._metrics_cache


class RJ45PickInsertResetDatasetCurriculum(RJ45ResetDatasetCurriculum):
    """Assign exact phase shares, then adapt or cyclically replay within each phase."""

    def __init__(self, cfg: CurriculumTermCfg, env: FrankaRJ45InsertionEnv):
        super().__init__(cfg, env)
        phase = env._reset_dataset_states["phase"]
        self._phase_fractions = tuple(float(value) for value in env.cfg.reset_dataset_phase_fractions)
        if len(self._phase_fractions) != 6:
            raise ValueError("Pick-insert reset sampling requires exactly six phase fractions.")
        self._phase_rows = tuple(torch.where(phase == phase_id)[0] for phase_id in range(6))
        if any(pool.numel() == 0 for pool in self._phase_rows):
            raise ValueError("Pick-insert reset rows must include every phase from 0 through 5.")

        self._phase_credits = [0.0] * len(self._phase_rows)
        self._phase_uniform_credits = [0.0] * len(self._phase_rows)
        self._phase_orders = [pool[torch.randperm(pool.numel(), device=env.device)] for pool in self._phase_rows]
        self._phase_cursors = [0] * len(self._phase_rows)
        self._assigned_phase_counts = [0] * len(self._phase_rows)
        self._assigned_phase_uniform_counts = [0] * len(self._phase_rows)
        self._assigned_phase_adaptive_counts = [0] * len(self._phase_rows)
        adaptive_enabled = not env.cfg.curriculum_freeze and env.cfg.reset_dataset_sampling_mode == "adaptive"
        self._metrics_cache = self._sampling_metrics(adaptive_enabled=adaptive_enabled)

    @staticmethod
    def _take_cyclic(
        pool: torch.Tensor,
        order: torch.Tensor,
        cursor: int,
        count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Take complete shuffled pool cycles and return updated order/cursor."""
        if count == 0:
            return pool.new_empty((0,), dtype=torch.long), order, cursor
        chunks: list[torch.Tensor] = []
        remaining = count
        while remaining:
            available = order.numel() - cursor
            take = min(remaining, available)
            chunks.append(order[cursor : cursor + take])
            cursor += take
            remaining -= take
            if cursor == order.numel():
                order = pool[torch.randperm(pool.numel(), device=pool.device)]
                cursor = 0
        return torch.cat(chunks), order, cursor

    def _phase_assignment_counts(self, count: int) -> list[int]:
        """Return weighted-fair phase counts with bounded long-run discrepancy."""
        if count == 0:
            return [0] * len(self._phase_rows)
        if not any(fraction > 0.0 for fraction in self._phase_fractions):
            raise RuntimeError("At least one pick-insert reset phase must have positive assignment weight.")

        desired = [
            credit + fraction * count
            for credit, fraction in zip(self._phase_credits, self._phase_fractions, strict=True)
        ]
        counts = [max(0, math.floor(value + 1.0e-12)) for value in desired]
        remainder = count - sum(counts)
        # Flooring leaves at most one unit of carry per phase, so repair is O(number of phases).
        while remainder > 0:
            selected = max(
                range(len(counts)),
                key=lambda phase_id: (desired[phase_id] - counts[phase_id], -phase_id),
            )
            counts[selected] += 1
            remainder -= 1
        while remainder < 0:
            selected = min(
                (phase_id for phase_id, phase_count in enumerate(counts) if phase_count > 0),
                key=lambda phase_id: (desired[phase_id] - counts[phase_id], phase_id),
            )
            counts[selected] -= 1
            remainder += 1
        self._phase_credits = [value - phase_count for value, phase_count in zip(desired, counts, strict=True)]
        return counts

    def _phase_uniform_assignment_count(self, phase_id: int, count: int) -> int:
        """Reserve the configured cyclic fraction independently within one phase."""
        credit = self._phase_uniform_credits[phase_id] + self._sampler.cfg.uniform_fraction * count
        uniform_count = min(count, math.floor(credit + 1.0e-12))
        self._phase_uniform_credits[phase_id] = credit - uniform_count
        return uniform_count

    def _sample_phase_partitioned_rows(self, count: int, *, adaptive_enabled: bool) -> torch.Tensor:
        """Sample exact long-run phase shares independently of per-phase pool size."""
        device = self._phase_rows[0].device
        rows = torch.empty(count, device=device, dtype=torch.long)
        if count == 0:
            return rows

        phase_counts = self._phase_assignment_counts(count)
        assignment_order = torch.randperm(count, device=device)
        position_cursor = 0
        probabilities = self._sampler._probabilities() if adaptive_enabled else None
        for phase_id, phase_count in enumerate(phase_counts):
            if phase_count == 0:
                continue
            positions = assignment_order[position_cursor : position_cursor + phase_count]
            position_cursor += phase_count
            uniform_count = (
                self._phase_uniform_assignment_count(phase_id, phase_count) if adaptive_enabled else phase_count
            )
            uniform_positions = positions[:uniform_count]
            adaptive_positions = positions[uniform_count:]
            uniform_rows, self._phase_orders[phase_id], self._phase_cursors[phase_id] = self._take_cyclic(
                self._phase_rows[phase_id],
                self._phase_orders[phase_id],
                self._phase_cursors[phase_id],
                uniform_count,
            )
            rows[uniform_positions] = uniform_rows
            if adaptive_positions.numel():
                phase_probabilities = probabilities[self._phase_rows[phase_id]]
                phase_probabilities /= phase_probabilities.sum()
                selected = torch.multinomial(phase_probabilities, adaptive_positions.numel(), replacement=True)
                rows[adaptive_positions] = self._phase_rows[phase_id][selected]
            self._assigned_phase_counts[phase_id] += phase_count
            self._assigned_phase_uniform_counts[phase_id] += uniform_count
            self._assigned_phase_adaptive_counts[phase_id] += phase_count - uniform_count
        return rows

    def _sample_training_rows(self, count: int) -> torch.Tensor:
        """Sample phase-partitioned adaptive rows with exact cyclic replay."""
        return self._sample_phase_partitioned_rows(count, adaptive_enabled=True)

    def _sample_uniform_rows(self, count: int) -> torch.Tensor:
        """Sample phase-partitioned rows using exact cyclic replay only."""
        return self._sample_phase_partitioned_rows(count, adaptive_enabled=False)

    def _sampling_metrics(self, *, adaptive_enabled: bool) -> dict[str, float]:
        """Return configured and realized phase/replay assignment metrics."""
        device = self._phase_rows[0].device
        probabilities = torch.zeros(self._row_count, dtype=torch.float32, device=device)
        replay_fraction = float(self._sampler.cfg.uniform_fraction) if adaptive_enabled else 1.0
        adaptive_probabilities = self._sampler._probabilities() if adaptive_enabled else None
        for phase_id, (fraction, pool) in enumerate(zip(self._phase_fractions, self._phase_rows, strict=True)):
            if fraction == 0.0:
                continue
            probabilities[pool] = fraction * replay_fraction / pool.numel()
            if adaptive_enabled:
                phase_probabilities = adaptive_probabilities[pool]
                phase_probabilities /= phase_probabilities.sum()
                probabilities[pool] += fraction * (1.0 - replay_fraction) * phase_probabilities

        top_count = max(1, math.ceil(0.01 * self._row_count))
        effective_pool_fraction = float(probabilities.square().sum().reciprocal() / self._row_count)
        top_mass = float(torch.topk(probabilities, top_count, sorted=False).values.sum())
        total_assignments = sum(self._assigned_phase_counts)
        total_uniform = sum(self._assigned_phase_uniform_counts)
        total_adaptive = sum(self._assigned_phase_adaptive_counts)
        metrics = {
            "sampler/effective_pool_fraction": effective_pool_fraction,
            "sampler/top_1_percent_mass": top_mass,
            "sampler/configured_uniform_replay_fraction": float(self._sampler.cfg.uniform_fraction),
            "sampler/uniform_replay_fraction": total_uniform / total_assignments if total_assignments else 0.0,
            "sampler/adaptive_sampling_fraction": total_adaptive / total_assignments if total_assignments else 0.0,
            "sampler/full_pick_start_fraction": self._phase_fractions[5],
        }
        for phase_id, fraction in enumerate(self._phase_fractions):
            phase_assignments = self._assigned_phase_counts[phase_id]
            metrics[f"sampler/configured_phase_{phase_id}_assignment_fraction"] = fraction
            metrics[f"sampler/realized_phase_{phase_id}_assignment_fraction"] = (
                phase_assignments / total_assignments if total_assignments else 0.0
            )
            metrics[f"sampler/phase_{phase_id}_uniform_replay_fraction"] = (
                self._assigned_phase_uniform_counts[phase_id] / phase_assignments if phase_assignments else 0.0
            )
            metrics[f"sampler/phase_{phase_id}_adaptive_sampling_fraction"] = (
                self._assigned_phase_adaptive_counts[phase_id] / phase_assignments if phase_assignments else 0.0
            )
        return metrics

    def __call__(
        self,
        env: FrankaRJ45InsertionEnv,
        env_ids: Sequence[int] | torch.Tensor | slice,
    ) -> dict[str, float]:
        ids = self._ids(env, env_ids)
        if ids.numel() == 0:
            return self._metrics_cache

        completed = (env.episode_length_buf[ids] > 0) & (env.reset_dataset_row_id[ids] >= 0)
        completed_ids = ids[completed]
        if completed_ids.numel() and not env.cfg.curriculum_freeze:
            progress = env.termination_manager.get_term_cfg("learning_progress_context").func
            self._sampler._record_validated(
                env.reset_dataset_row_id[completed_ids],
                progress.ever_success[completed_ids],
            )

        adaptive_enabled = not env.cfg.curriculum_freeze and env.cfg.reset_dataset_sampling_mode == "adaptive"
        rows = self._sample_phase_partitioned_rows(ids.numel(), adaptive_enabled=adaptive_enabled)
        env.reset_dataset_row_id[ids] = rows

        self._assignments_since_metrics += ids.numel()
        if self._assignments_since_metrics >= env.num_envs:
            self._metrics_cache = self._sampling_metrics(adaptive_enabled=adaptive_enabled)
            self._assignments_since_metrics = 0
        return self._metrics_cache


__all__ = [
    "RJ45PickInsertResetDatasetCurriculum",
    "RJ45ResetDatasetCurriculum",
    "ResetDatasetSamplerCfg",
]
