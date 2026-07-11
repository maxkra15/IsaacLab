# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static reset-distribution mixture for the Franka Pour task."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
    from ..pour_env import FrankaPourEnv


RESET_MIXTURE_REGION_NAMES = ("reaching", "near_object", "grasped", "near_goal")
"""Reset regions ordered from the full task to a fully refilled, pour-ready state."""

RESET_MIXTURE_STAGE_NAMES = ("randomized", "grasp", "carry", "tilt")
"""Curriculum stages carrying the reward/action semantics for each reset region."""

_EARLY_ABNORMAL_COMPLETION_STEPS = 8


class PourResetMixture(ManagerTermBase):
    """Sample a static reset mixture and report outcomes for each reset region.

    Isaac Lab evaluates curriculum terms before reset events, while the completed episode state is
    still available. This term uses that lifecycle hook only for reset sampling and logging; it
    does not schedule or promote reset difficulty.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: FrankaPourEnv):
        super().__init__(cfg, env)
        probabilities = tuple(float(value) for value in env.cfg.reset_mixture_probabilities)
        target_fraction = float(env.cfg.pour_target_frac)

        probability_sum = sum(probabilities)
        self._probability_values = tuple(value / probability_sum for value in probabilities)
        self._probabilities = torch.as_tensor(self._probability_values, device=env.device, dtype=torch.float32)
        self._target_fraction = target_fraction
        self._stage_indices = tuple(env.cfg.curriculum_stage_names.index(name) for name in RESET_MIXTURE_STAGE_NAMES)
        self._randomization_level = len(env.cfg.curriculum_randomization_extent_levels) - 1
        self._statistics_window_size = int(env.cfg.reset_mixture_statistics_window_size)
        self._success_dwell_steps = max(
            1,
            math.ceil(float(env.cfg.success_dwell_time_s) / max(float(env.step_dt), 1.0e-6)),
        )
        self._lost_grasp_dwell_steps = max(
            1,
            math.ceil(float(env.cfg.lost_grasp_dwell_time_s) / max(float(env.step_dt), 1.0e-6)),
        )

        region_count = len(RESET_MIXTURE_REGION_NAMES)
        self._sample_count = torch.zeros(region_count, dtype=torch.long)
        self._total_episode_count = torch.zeros(region_count, dtype=torch.long)
        self._window_count = torch.zeros(region_count, dtype=torch.long)
        self._window_position = torch.zeros(region_count, dtype=torch.long)
        self._outcome_window = torch.zeros(
            (region_count, self._statistics_window_size, 16),
            dtype=torch.float64,
        )

    def __call__(
        self,
        env: FrankaPourEnv,
        env_ids: Sequence[int] | torch.Tensor | slice,
    ) -> dict[str, float]:
        """Record completed outcomes, then sample reset regions for the next episodes."""
        if isinstance(env_ids, slice):
            ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()

        if ids.numel() > 0:
            previous_regions = env.reset_region_id[ids]
            completed = (env.episode_length_buf[ids] > 0) & (previous_regions >= 0)
            completed_ids = ids[completed]
            if completed_ids.numel() > 0:
                regions = previous_regions[completed]
                if bool(torch.any(regions >= len(RESET_MIXTURE_REGION_NAMES))):
                    raise RuntimeError("A completed episode has an invalid reset region ID.")
                terminated = env.termination_manager.terminated[completed_ids]
                timeouts = env.termination_manager.time_outs[completed_ids]
                final_success = env._success_dwell_count[completed_ids] >= self._success_dwell_steps
                final_lost_grasp = env._lost_grasp_dwell_count[completed_ids] >= self._lost_grasp_dwell_steps
                spill_fraction = env.spilled_fraction()[completed_ids]
                final_spill = spill_fraction > float(env.cfg.max_spill_fraction)
                nonfinite_failure = env.termination_manager.get_term("failure")[completed_ids]
                extreme_rigid_state = env.termination_manager.get_term("extreme_rigid_state")[completed_ids]
                particle_out_of_bounds = env.termination_manager.get_term("particle_out_of_bounds")[completed_ids]
                success_termination = env.termination_manager.get_term("success")[completed_ids]
                abnormal_completion = nonfinite_failure | extreme_rigid_state | particle_out_of_bounds
                early_abnormal_completion = abnormal_completion & (
                    env.episode_length_buf[completed_ids] <= _EARLY_ABNORMAL_COMPLETION_STEPS
                )
                outcomes = (
                    torch.stack(
                        (
                            regions,
                            final_success,
                            (terminated | timeouts) & ~final_success,
                            final_spill,
                            final_lost_grasp,
                            timeouts,
                            env.episode_length_buf[completed_ids],
                            env.ep_max_target_frac[completed_ids],
                            spill_fraction,
                            env.episode_succeeded[completed_ids],
                            env.ep_max_target_frac[completed_ids] >= self._target_fraction,
                            nonfinite_failure,
                            extreme_rigid_state,
                            particle_out_of_bounds,
                            success_termination,
                            abnormal_completion,
                            early_abnormal_completion,
                        ),
                        dim=-1,
                    )
                    .detach()
                    .to(device="cpu", dtype=torch.float64)
                )
                for region in range(len(RESET_MIXTURE_REGION_NAMES)):
                    region_outcomes = outcomes[outcomes[:, 0] == region, 1:]
                    outcome_count = region_outcomes.shape[0]
                    if outcome_count == 0:
                        continue
                    self._total_episode_count[region] += outcome_count
                    if outcome_count >= self._statistics_window_size:
                        self._outcome_window[region].copy_(region_outcomes[-self._statistics_window_size :])
                        self._window_count[region] = self._statistics_window_size
                        self._window_position[region] = 0
                        continue
                    position = int(self._window_position[region])
                    slots = torch.arange(position, position + outcome_count).remainder_(self._statistics_window_size)
                    self._outcome_window[region, slots] = region_outcomes
                    self._window_position[region] = (position + outcome_count) % self._statistics_window_size
                    self._window_count[region] = min(
                        int(self._window_count[region]) + outcome_count,
                        self._statistics_window_size,
                    )

            sampled_regions = torch.multinomial(self._probabilities, ids.numel(), replacement=True)
            env.reset_region_id[ids] = sampled_regions
            sampled_regions_cpu = sampled_regions.to(device="cpu")
            self._sample_count.scatter_add_(0, sampled_regions_cpu, torch.ones_like(sampled_regions_cpu))
            for region, stage in enumerate(self._stage_indices):
                env.set_curriculum_stage(ids[sampled_regions == region], stage)
            env.set_curriculum_randomization_level(ids, self._randomization_level)
            env.pour_target_frac[ids] = self._target_fraction

        total_samples = max(int(self._sample_count.sum()), 1)
        metrics = {
            "target_fraction": self._target_fraction,
            "randomization_extent_fraction": float(
                env.cfg.curriculum_randomization_extent_levels[self._randomization_level]
            ),
        }
        window_means = tuple(
            self._outcome_window[region, : int(self._window_count[region])].mean(dim=0)
            if int(self._window_count[region]) > 0
            else self._outcome_window.new_zeros(16)
            for region in range(len(RESET_MIXTURE_REGION_NAMES))
        )
        transition_mass = tuple(
            self._probability_values[region] * float(window_means[region][5])
            for region in range(len(RESET_MIXTURE_REGION_NAMES))
        )
        transition_mass_total = max(sum(transition_mass), 1.0e-12)
        for region, name in enumerate(RESET_MIXTURE_REGION_NAMES):
            window_count = int(self._window_count[region])
            means = window_means[region]
            window = self._outcome_window[region, :window_count]
            abnormal_rows = window[:, 14] > 0.5
            mean_abnormal_episode_length = (
                float(window[abnormal_rows, 5].mean()) if bool(torch.any(abnormal_rows)) else 0.0
            )
            metrics.update(
                {
                    f"{name}_configured_probability": self._probability_values[region],
                    f"{name}_sample_fraction": int(self._sample_count[region]) / total_samples,
                    f"{name}_estimated_transition_fraction": transition_mass[region] / transition_mass_total,
                    f"{name}_sampled_resets": float(self._sample_count[region]),
                    f"{name}_window_completed_episodes": float(window_count),
                    f"{name}_total_completed_episodes": float(self._total_episode_count[region]),
                    f"{name}_success_rate": float(means[0]),
                    f"{name}_failure_rate": float(means[1]),
                    f"{name}_spill_rate": float(means[2]),
                    f"{name}_lost_grasp_rate": float(means[3]),
                    f"{name}_timeout_rate": float(means[4]),
                    f"{name}_mean_episode_length_steps": float(means[5]),
                    f"{name}_mean_peak_target_fraction": float(means[6]),
                    f"{name}_mean_spill_fraction": float(means[7]),
                    f"{name}_ever_success_rate": float(means[8]),
                    f"{name}_raw_peak_target_rate": float(means[9]),
                    f"{name}_nonfinite_failure_rate": float(means[10]),
                    f"{name}_extreme_rigid_state_rate": float(means[11]),
                    f"{name}_particle_out_of_bounds_rate": float(means[12]),
                    f"{name}_success_termination_rate": float(means[13]),
                    f"{name}_abnormal_completion_rate": float(means[14]),
                    f"{name}_early_abnormal_completion_rate": float(means[15]),
                    f"{name}_mean_abnormal_episode_length_steps": mean_abnormal_episode_length,
                }
            )
        return metrics
