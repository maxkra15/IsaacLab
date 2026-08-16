# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success-aware replay of validated RJ45 near-goal reset rows."""

from __future__ import annotations

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


__all__ = ["RJ45ResetDatasetCurriculum", "ResetDatasetSamplerCfg"]
