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
    """Keep a fixed deployment-start share while adapting intermediate rows.

    The Franka stack deliberately reserves a substantial fraction of resets
    for the complete table-start task.  Without that reservation, row-level
    adaptive sampling can become excellent at insertion continuations while
    rarely exercising the open-hand approach and grasp that the deployed
    policy must perform.  Phase 5 is the validated full-pick recipe; all other
    phases retain success-aware sampling plus exact cyclic replay.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: FrankaRJ45InsertionEnv):
        super().__init__(cfg, env)
        phase = env._reset_dataset_states["phase"]
        self._deployment_rows = torch.where(phase == 5)[0]
        self._continuation_rows = torch.where(phase != 5)[0]
        if self._deployment_rows.numel() == 0 or self._continuation_rows.numel() == 0:
            raise ValueError("Pick-insert reset rows must include phase 5 and continuation phases 0-4.")
        self._deployment_fraction = float(env.cfg.full_pick_start_fraction)
        if not 0.0 < self._deployment_fraction < 1.0:
            raise ValueError("full_pick_start_fraction must lie strictly inside (0, 1).")
        self._metrics_cache["sampler/full_pick_start_fraction"] = self._deployment_fraction
        self._deployment_credit = 0.0
        self._continuation_uniform_credit = 0.0
        self._deployment_order = self._deployment_rows[torch.randperm(self._deployment_rows.numel(), device=env.device)]
        self._continuation_order = self._continuation_rows[
            torch.randperm(self._continuation_rows.numel(), device=env.device)
        ]
        self._deployment_cursor = 0
        self._continuation_cursor = 0

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

    def _sample_training_rows(self, count: int) -> torch.Tensor:
        """Sample an exact long-run full-pick share and adaptive continuations."""
        device = self._deployment_rows.device
        rows = torch.empty(count, device=device, dtype=torch.long)
        deployment_credit = self._deployment_credit + self._deployment_fraction * count
        deployment_count = min(count, math.floor(deployment_credit + 1.0e-12))
        self._deployment_credit = deployment_credit - deployment_count
        assignment_order = torch.randperm(count, device=device)
        deployment_positions = assignment_order[:deployment_count]
        continuation_positions = assignment_order[deployment_count:]

        deployment, self._deployment_order, self._deployment_cursor = self._take_cyclic(
            self._deployment_rows,
            self._deployment_order,
            self._deployment_cursor,
            deployment_count,
        )
        rows[deployment_positions] = deployment

        continuation_count = int(continuation_positions.numel())
        uniform_credit = self._continuation_uniform_credit + self._sampler.cfg.uniform_fraction * continuation_count
        uniform_count = min(continuation_count, math.floor(uniform_credit + 1.0e-12))
        self._continuation_uniform_credit = uniform_credit - uniform_count
        continuation_assignment = torch.randperm(continuation_count, device=device)
        uniform_positions = continuation_positions[continuation_assignment[:uniform_count]]
        adaptive_positions = continuation_positions[continuation_assignment[uniform_count:]]
        uniform_rows, self._continuation_order, self._continuation_cursor = self._take_cyclic(
            self._continuation_rows,
            self._continuation_order,
            self._continuation_cursor,
            uniform_count,
        )
        rows[uniform_positions] = uniform_rows
        if adaptive_positions.numel():
            probability = self._sampler._probabilities()[self._continuation_rows]
            probability /= probability.sum()
            selected = torch.multinomial(probability, adaptive_positions.numel(), replacement=True)
            rows[adaptive_positions] = self._continuation_rows[selected]
        return rows

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
            rows = self._sample_training_rows(ids.numel())
        env.reset_dataset_row_id[ids] = rows

        self._assignments_since_metrics += ids.numel()
        if self._assignments_since_metrics >= env.num_envs:
            self._metrics_cache = self._sampler.metrics()
            self._metrics_cache["sampler/full_pick_start_fraction"] = self._deployment_fraction
            self._assignments_since_metrics = 0
        return self._metrics_cache


__all__ = [
    "RJ45PickInsertResetDatasetCurriculum",
    "RJ45ResetDatasetCurriculum",
    "ResetDatasetSamplerCfg",
]
