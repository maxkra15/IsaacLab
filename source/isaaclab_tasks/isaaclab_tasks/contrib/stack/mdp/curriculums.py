# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adaptive epsilon reset sampling for cube stacking."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ring_append_bool_count_rate(
    data: torch.Tensor,
    stream_ids: torch.Tensor,
    values: torch.Tensor,
    pointer: torch.Tensor,
    size: torch.Tensor,
    true_count: torch.Tensor,
    rate: torch.Tensor,
) -> None:
    """Append outcomes to per-row Boolean rings and update rolling success rates.

    Duplicate row ids are kept in stable input order, including batches that
    contain more outcomes for one row than the complete history capacity.
    """
    if stream_ids.numel() == 0:
        return

    capacity = data.shape[1]
    unique_ids, inverse, counts = torch.unique(stream_ids, return_inverse=True, return_counts=True)
    if unique_ids.numel() == stream_ids.numel():
        columns = pointer[stream_ids].long()
        overwritten = torch.where(
            size[stream_ids] == capacity,
            data[stream_ids, columns].to(dtype=true_count.dtype),
            torch.zeros_like(true_count[stream_ids]),
        )
        new_true_counts = true_count[stream_ids] - overwritten + values.to(dtype=true_count.dtype)
        data[stream_ids, columns] = values
        pointer[stream_ids] = ((columns + 1) % capacity).to(dtype=pointer.dtype)
        size[stream_ids] = (size[stream_ids] + 1).clamp(max=capacity)
        true_count[stream_ids] = new_true_counts
        rate[stream_ids] = new_true_counts.to(rate.dtype) / size[stream_ids].clamp(min=1)
        return

    order = torch.argsort(inverse, stable=True)
    sorted_ids = stream_ids[order]
    sorted_values = values[order]
    group_starts = counts.cumsum(0) - counts
    local_rank = torch.arange(stream_ids.numel(), device=data.device) - torch.repeat_interleave(group_starts, counts)
    inverse_sorted = inverse[order]
    counts_sorted = counts[inverse_sorted]
    true_added = torch.zeros(unique_ids.shape, device=data.device, dtype=true_count.dtype)
    true_added.scatter_add_(0, inverse, values.to(dtype=true_count.dtype))

    keep_start = (counts - capacity).clamp(min=0)
    keep = local_rank >= torch.repeat_interleave(keep_start, counts)
    true_kept = torch.zeros_like(true_added)
    true_kept.scatter_add_(0, inverse_sorted[keep], sorted_values[keep].to(dtype=true_count.dtype))

    overwrite_start = capacity - size[sorted_ids].long()
    overwrite_mask = (counts_sorted < capacity) & (local_rank >= overwrite_start)
    overwritten = torch.zeros_like(true_added)
    overwrite_ids = sorted_ids[overwrite_mask]
    overwrite_columns = (pointer[overwrite_ids].long() + local_rank[overwrite_mask]) % capacity
    overwritten.scatter_add_(
        0,
        inverse_sorted[overwrite_mask],
        data[overwrite_ids, overwrite_columns].to(dtype=true_count.dtype),
    )

    kept_ids = sorted_ids[keep]
    kept_columns = (pointer[kept_ids].long() + local_rank[keep]) % capacity
    data[kept_ids, kept_columns] = sorted_values[keep]
    replace = counts >= capacity
    new_true_counts = torch.where(replace, true_kept, true_count[unique_ids] - overwritten + true_added)
    new_size = (size[unique_ids].long() + counts).clamp(max=capacity)
    pointer[unique_ids] = ((pointer[unique_ids].long() + counts) % capacity).to(dtype=pointer.dtype)
    size[unique_ids] = new_size.to(dtype=size.dtype)
    true_count[unique_ids] = new_true_counts
    rate[unique_ids] = new_true_counts.to(rate.dtype) / new_size.clamp(min=1).to(rate.dtype)


class _EpsilonResetTableSampler:
    """Beta-kernel curriculum with an epsilon floor over one reset table."""

    def __init__(
        self,
        row_count: int,
        device: str | torch.device,
        *,
        monitored_history_len: int,
        target_success_rate: float,
        kappa: float,
        epsilon: float,
    ) -> None:
        if row_count < 1:
            raise ValueError("row_count must be positive.")
        if monitored_history_len < 1:
            raise ValueError("monitored_history_len must be positive.")
        if not 0.0 < target_success_rate < 1.0:
            raise ValueError("target_success_rate must lie strictly between zero and one.")
        if kappa <= 0.0:
            raise ValueError("kappa must be positive.")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")

        self.row_count = row_count
        self.monitored_history_len = monitored_history_len
        self.target_success_rate = target_success_rate
        self.kappa = kappa
        self.epsilon = epsilon
        self._alpha_minus_one = kappa * target_success_rate
        self._beta_minus_one = kappa * (1.0 - target_success_rate)

        self.success_rates = torch.zeros(row_count, dtype=torch.float32, device=device)
        self.success_history = torch.zeros(
            (row_count, monitored_history_len),
            dtype=torch.bool,
            device=device,
        )
        self.history_pointer = torch.zeros(row_count, dtype=torch.int32, device=device)
        self.history_size = torch.zeros_like(self.history_pointer)
        self.history_success_count = torch.zeros_like(self.history_pointer)
        self.total_successes = torch.zeros(row_count, dtype=torch.long, device=device)
        self.total_attempts = torch.zeros_like(self.total_successes)

    def beta_scores(self) -> torch.Tensor:
        """Return the Beta-kernel score for each reset row."""
        return self.success_rates.pow(self._alpha_minus_one) * (1.0 - self.success_rates).pow(self._beta_minus_one)

    def probabilities(self) -> torch.Tensor:
        """Return normalized Beta scores with a per-row epsilon floor."""
        weights = self.beta_scores().add(self.epsilon)
        return weights / weights.sum()

    def sample(self, count: int) -> torch.Tensor:
        """Sample reset rows with replacement."""
        if count < 0:
            raise ValueError("count cannot be negative.")
        if count == 0:
            return torch.empty(0, dtype=torch.long, device=self.success_rates.device)
        return torch.multinomial(self.probabilities(), count, replacement=True)

    def record(self, rows: torch.Tensor, success: torch.Tensor) -> None:
        """Append completed outcomes to each reset row's exact rolling window."""
        if rows.shape != success.shape or rows.dtype != torch.long or success.dtype != torch.bool:
            raise ValueError("rows and success must be aligned long and Boolean vectors.")
        if rows.numel() == 0:
            return
        _ring_append_bool_count_rate(
            self.success_history,
            rows,
            success,
            self.history_pointer,
            self.history_size,
            self.history_success_count,
            self.success_rates,
        )
        self.total_attempts.add_(torch.bincount(rows, minlength=self.row_count))
        self.total_successes.add_(torch.bincount(rows[success], minlength=self.row_count))

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return the rolling monitor and cumulative diagnostics for a checkpoint."""
        return {
            "success_rates": self.success_rates.clone(),
            "success_history": self.success_history.clone(),
            "history_pointer": self.history_pointer.clone(),
            "history_size": self.history_size.clone(),
            "history_success_count": self.history_success_count.clone(),
            "total_successes": self.total_successes.clone(),
            "total_attempts": self.total_attempts.clone(),
        }

    def set_state(self, state: dict[str, torch.Tensor]) -> None:
        """Restore an exact rolling-monitor checkpoint."""
        targets = {
            "success_rates": self.success_rates,
            "success_history": self.success_history,
            "history_pointer": self.history_pointer,
            "history_size": self.history_size,
            "history_success_count": self.history_success_count,
            "total_successes": self.total_successes,
            "total_attempts": self.total_attempts,
        }
        for key, target in targets.items():
            if key not in state:
                raise KeyError(f"Reset-table curriculum checkpoint is missing '{key}'.")
            if state[key].shape != target.shape:
                raise ValueError(
                    f"Reset-table curriculum checkpoint '{key}' has shape {state[key].shape}; expected {target.shape}."
                )
        if bool(torch.any((state["history_pointer"] < 0) | (state["history_pointer"] >= self.monitored_history_len))):
            raise ValueError("Reset-table curriculum checkpoint contains an invalid history pointer.")
        if bool(torch.any((state["history_size"] < 0) | (state["history_size"] > self.monitored_history_len))):
            raise ValueError("Reset-table curriculum checkpoint contains an invalid history size.")
        for key, target in targets.items():
            target.copy_(state[key].to(device=target.device, dtype=target.dtype))


class StackResetTableCurriculum(ManagerTermBase):
    """Mix guaranteed table starts with adaptive epsilon reset sampling."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        reset_term = env.event_manager.get_term_cfg("reset_from_state_buffer").func
        if not hasattr(reset_term, "row_count"):
            raise RuntimeError("StackResetTableCurriculum requires StackResetStateTable.")
        self._reset_term = reset_term
        self._sampler = _EpsilonResetTableSampler(
            reset_term.row_count,
            env.device,
            monitored_history_len=int(cfg.params.get("monitored_history_len", 50)),
            target_success_rate=float(cfg.params.get("target_success_rate", 0.5)),
            kappa=float(cfg.params.get("kappa", 1.0)),
            epsilon=float(cfg.params.get("epsilon", 1.0e-4)),
        )
        self._table_sampling_probability = float(cfg.params.get("table_sampling_probability", 0.35))
        if not 0.0 < self._table_sampling_probability < 1.0:
            raise ValueError("table_sampling_probability must lie strictly between zero and one.")
        self._balance_recipes = bool(cfg.params.get("balance_recipes", False))
        self._balance_reset_modes = bool(cfg.params.get("balance_reset_modes", False))
        self._global_sampling = bool(cfg.params.get("global_sampling", False))
        if self._global_sampling and (self._balance_recipes or self._balance_reset_modes):
            raise ValueError("global_sampling cannot be combined with recipe or reset-mode balancing.")
        table_recipe_id = reset_term.recipe_names.index("table")
        self._table_rows = reset_term.recipe_ids == table_recipe_id
        if not bool(torch.any(self._table_rows)) or bool(torch.all(self._table_rows)):
            raise RuntimeError("The stack reset table must contain both table and intermediate rows.")
        self._layout_count = reset_term.layout_count
        self._recipe_rows = tuple(reset_term.recipe_ids == recipe for recipe in range(len(reset_term.recipe_names)))
        pair_ids = getattr(reset_term, "grasp_pair_ids", None)
        self._grasp_pair_rows = tuple(pair_ids == pair_id for pair_id in range(3)) if pair_ids is not None else None
        orientation_ids = getattr(reset_term, "orientation_bin_ids", None)
        self._orientation_rows = (
            tuple(orientation_ids == orientation_id for orientation_id in range(8))
            if orientation_ids is not None
            else None
        )
        resolved_tilt_azimuth_ids = getattr(reset_term, "tilt_azimuth_bin_ids", None)
        tilt_azimuth_ids = getattr(
            reset_term,
            "authored_tilt_azimuth_bin_ids",
            resolved_tilt_azimuth_ids,
        )
        self._tilt_azimuth_rows = (
            tuple(tilt_azimuth_ids == azimuth_id for azimuth_id in range(8)) if tilt_azimuth_ids is not None else None
        )
        self._resolved_tilt_azimuth_rows = (
            tuple(resolved_tilt_azimuth_ids == azimuth_id for azimuth_id in range(8))
            if hasattr(reset_term, "authored_tilt_azimuth_bin_ids")
            else None
        )
        tilt_magnitude_ids = getattr(reset_term, "tilt_magnitude_bin_ids", None)
        self._tilt_magnitude_rows = (
            tuple(tilt_magnitude_ids == magnitude_id for magnitude_id in range(4))
            if tilt_magnitude_ids is not None
            else None
        )
        self._continuation_attempts = torch.zeros((), dtype=torch.long, device=env.device)
        self._continuation_successes = torch.zeros((), dtype=torch.long, device=env.device)
        self._full_task_attempts_by_row = torch.zeros(reset_term.row_count, dtype=torch.long, device=env.device)
        self._full_task_successes_by_row = torch.zeros_like(self._full_task_attempts_by_row)

    def _sampling_distribution(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mixture probabilities and their underlying Beta scores.

        Layout-balanced tasks normalize within each workspace layout. Reset
        banks may instead normalize within recipe-layout strata or, for
        pair-conditioned hands, within recipe/pair/yaw/tilt strata. The latter
        guarantees that a hard finger or wrist mode cannot disappear while the
        Beta score selects the useful progress frontier inside that mode.
        """
        beta_scores = self._sampler.beta_scores()
        adaptive = beta_scores.add(self._sampler.epsilon)
        adaptive[self._table_rows] = 0.0
        layout_ids = self._reset_term.layout_ids
        layout_count = getattr(self, "_layout_count", self._reset_term.layout_count)
        if getattr(self, "_global_sampling", False):
            # Normalize one Beta-plus-epsilon score vector over the complete
            # active table without layout, recipe, or mode quotas.
            pass
        elif getattr(self, "_balance_reset_modes", False):
            pair_ids = self._reset_term.grasp_pair_ids
            orientation_ids = self._reset_term.orientation_bin_ids
            # Balance the deliberately authored reset modes. IK may repair an
            # infeasible row to a nearby physical azimuth; mixing that resolved
            # azimuth with an authored magnitude would create incoherent,
            # potentially empty strata.
            tilt_azimuth_ids = getattr(
                self._reset_term,
                "authored_tilt_azimuth_bin_ids",
                self._reset_term.tilt_azimuth_bin_ids,
            )
            tilt_magnitude_ids = self._reset_term.tilt_magnitude_bin_ids
            recipe_ids = self._reset_term.recipe_ids
            stratum_ids = (
                4 * (8 * (8 * (3 * recipe_ids + pair_ids) + orientation_ids) + tilt_azimuth_ids) + tilt_magnitude_ids
            )
            stratum_count = len(self._reset_term.recipe_names) * 3 * 8 * 8 * 4
            stratum_mass = torch.zeros(stratum_count, dtype=adaptive.dtype, device=adaptive.device)
            stratum_mass.scatter_add_(0, stratum_ids, adaptive)
            adaptive /= stratum_mass[stratum_ids].clamp_min(torch.finfo(adaptive.dtype).tiny)
        elif getattr(self, "_balance_recipes", False):
            recipe_ids = self._reset_term.recipe_ids
            stratum_ids = recipe_ids * layout_count + layout_ids
            stratum_count = len(self._reset_term.recipe_names) * layout_count
            stratum_mass = torch.zeros(stratum_count, dtype=adaptive.dtype, device=adaptive.device)
            stratum_mass.scatter_add_(0, stratum_ids, adaptive)
            adaptive /= stratum_mass[stratum_ids].clamp_min(torch.finfo(adaptive.dtype).tiny)
        else:
            layout_mass = torch.zeros(layout_count, dtype=adaptive.dtype, device=adaptive.device)
            layout_mass.scatter_add_(0, layout_ids, adaptive)
            adaptive /= layout_mass[layout_ids].clamp_min(torch.finfo(adaptive.dtype).tiny)
        adaptive[self._table_rows] = 0.0
        adaptive /= adaptive.sum()
        table = self._table_rows.to(dtype=adaptive.dtype)
        table /= table.sum()
        probabilities = (1.0 - self._table_sampling_probability) * adaptive + self._table_sampling_probability * table
        return probabilities, beta_scores

    def _sampling_probabilities(self) -> torch.Tensor:
        """Return the fixed-table/adaptive-intermediate sampling mixture."""
        return self._sampling_distribution()[0]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        success_context_name: str = "learning_progress_context",
        final_success_context_name: str = "progress_context",
        monitored_history_len: int = 50,
        target_success_rate: float = 0.5,
        kappa: float = 1.0,
        epsilon: float = 1.0e-4,
        table_sampling_probability: float = 0.35,
        balance_recipes: bool = False,
        balance_reset_modes: bool = False,
        global_sampling: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Record progress/final outcomes, select new rows, and report coverage."""
        del (
            monitored_history_len,
            target_success_rate,
            kappa,
            epsilon,
            table_sampling_probability,
            balance_recipes,
            balance_reset_modes,
            global_sampling,
        )
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
        batch_success_rate = torch.zeros((), device=env.device)
        batch_full_task_success_rate = torch.zeros((), device=env.device)
        batch_table_full_task_success_rate = torch.zeros((), device=env.device)
        batch_table_full_task_attempts = torch.zeros((), device=env.device)
        if ids.numel():
            state = getattr(env, "stack_reset_state", None)
            initialized = state.initialized if state is not None else env.stack_reset_initialized
            row_ids = state.row_ids if state is not None else env.stack_reset_row_ids
            completed = initialized[ids] & (env.episode_length_buf[ids] > 0)
            completed_ids = ids[completed]
            if completed_ids.numel():
                success_context = env.termination_manager.get_term_cfg(success_context_name).func
                succeeded = success_context.ever_success[completed_ids]
                final_success_context = env.termination_manager.get_term_cfg(final_success_context_name).func
                final_succeeded = final_success_context.ever_success[completed_ids]
                batch_success_rate = succeeded.float().mean()
                batch_full_task_success_rate = final_succeeded.float().mean()
                self._continuation_attempts.add_(completed_ids.numel())
                self._continuation_successes.add_(final_succeeded.sum())
                completed_rows = row_ids[completed_ids]
                completed_table = self._table_rows[completed_rows]
                batch_table_full_task_attempts = completed_table.sum()
                if bool(torch.any(completed_table)):
                    batch_table_full_task_success_rate = final_succeeded[completed_table].float().mean()
                self._sampler.record(
                    completed_rows,
                    succeeded,
                )
                self._full_task_attempts_by_row.add_(
                    torch.bincount(completed_rows, minlength=self._reset_term.row_count)
                )
                self._full_task_successes_by_row.add_(
                    torch.bincount(completed_rows[final_succeeded], minlength=self._reset_term.row_count)
                )

            probabilities, beta_scores = self._sampling_distribution()
            rows = torch.multinomial(probabilities, ids.numel(), replacement=True)
            row_ids[ids] = rows
        else:
            probabilities, beta_scores = self._sampling_distribution()

        attempts = self._sampler.total_attempts
        observed = attempts > 0
        success_rate = self._sampler.total_successes.sum().float() / attempts.sum().clamp_min(1)
        entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum()
        entropy /= math.log(probabilities.numel())
        active_rows = ~self._table_rows
        epsilon_mass = self._sampler.epsilon * active_rows.sum()
        epsilon_mass /= beta_scores[active_rows].sum() + epsilon_mass
        rolling_rates = self._sampler.success_rates
        observed_count = observed.sum().clamp_min(1)
        metrics: dict[str, torch.Tensor] = {
            "row_coverage": observed.float().mean(),
            "row_success_rate": success_rate,
            "rolling_row_success_rate": (rolling_rates * observed).sum() / observed_count,
            "batch_success_rate": batch_success_rate,
            "batch_full_task_success_rate": batch_full_task_success_rate,
            "batch_table_full_task_success_rate": batch_table_full_task_success_rate,
            "batch_table_full_task_attempts": batch_table_full_task_attempts,
            "sampling_entropy": entropy,
            "epsilon_probability_mass": epsilon_mass,
            "table_probability": probabilities[self._table_rows].sum(),
            "target_band_fraction": ((rolling_rates - self._sampler.target_success_rate).abs() <= 0.1).float().mean(),
            "full_task_attempts": self._continuation_attempts.float(),
            "full_task_success_rate": self._continuation_successes.float() / self._continuation_attempts.clamp_min(1),
            "table_curriculum_success_rate": self._sampler.total_successes[self._table_rows].sum().float()
            / self._sampler.total_attempts[self._table_rows].sum().clamp_min(1),
            "table_full_task_success_rate": self._full_task_successes_by_row[self._table_rows].sum().float()
            / self._full_task_attempts_by_row[self._table_rows].sum().clamp_min(1),
            "table_full_task_attempts": self._full_task_attempts_by_row[self._table_rows].sum().float(),
        }
        recipe_rows_by_id = getattr(
            self,
            "_recipe_rows",
            tuple(self._reset_term.recipe_ids == recipe for recipe in range(len(self._reset_term.recipe_names))),
        )
        for recipe, recipe_rows in enumerate(recipe_rows_by_id):
            recipe_attempts = attempts[recipe_rows].sum()
            recipe_full_task_attempts = self._full_task_attempts_by_row[recipe_rows].sum()
            recipe_name = self._reset_term.recipe_names[recipe]
            metrics[f"recipe_{recipe_name}_attempts"] = recipe_attempts
            metrics[f"recipe_{recipe_name}_full_stack_attempts"] = recipe_full_task_attempts
            metrics[f"recipe_{recipe_name}_curriculum_success"] = self._sampler.total_successes[
                recipe_rows
            ].sum().float() / recipe_attempts.clamp_min(1)
            metrics[f"recipe_{recipe_name}_full_stack_success"] = self._full_task_successes_by_row[
                recipe_rows
            ].sum().float() / recipe_full_task_attempts.clamp_min(1)
            metrics[f"recipe_{self._reset_term.recipe_names[recipe]}_probability"] = probabilities[recipe_rows].sum()
        pair_rows_by_id = getattr(self, "_grasp_pair_rows", None)
        if pair_rows_by_id is not None:
            for pair_rows, pair_name in zip(
                pair_rows_by_id,
                ("index_thumb", "middle_thumb", "ring_thumb"),
                strict=True,
            ):
                pair_attempts = attempts[pair_rows].sum()
                pair_full_attempts = self._full_task_attempts_by_row[pair_rows].sum()
                metrics[f"pair_{pair_name}_probability"] = probabilities[pair_rows].sum()
                metrics[f"pair_{pair_name}_curriculum_success"] = self._sampler.total_successes[
                    pair_rows
                ].sum().float() / pair_attempts.clamp_min(1)
                metrics[f"pair_{pair_name}_full_stack_success"] = self._full_task_successes_by_row[
                    pair_rows
                ].sum().float() / pair_full_attempts.clamp_min(1)
        orientation_rows_by_id = getattr(self, "_orientation_rows", None)
        if orientation_rows_by_id is not None:
            for orientation_id, orientation_rows in enumerate(orientation_rows_by_id):
                orientation_attempts = attempts[orientation_rows].sum()
                orientation_full_attempts = self._full_task_attempts_by_row[orientation_rows].sum()
                metrics[f"orientation_{orientation_id}_probability"] = probabilities[orientation_rows].sum()
                metrics[f"orientation_{orientation_id}_curriculum_success"] = self._sampler.total_successes[
                    orientation_rows
                ].sum().float() / orientation_attempts.clamp_min(1)
                metrics[f"orientation_{orientation_id}_full_stack_success"] = self._full_task_successes_by_row[
                    orientation_rows
                ].sum().float() / orientation_full_attempts.clamp_min(1)
        tilt_azimuth_rows_by_id = getattr(self, "_tilt_azimuth_rows", None)
        if tilt_azimuth_rows_by_id is not None:
            for azimuth_id, azimuth_rows in enumerate(tilt_azimuth_rows_by_id):
                azimuth_attempts = attempts[azimuth_rows].sum()
                azimuth_full_attempts = self._full_task_attempts_by_row[azimuth_rows].sum()
                metrics[f"tilt_azimuth_{azimuth_id}_probability"] = probabilities[azimuth_rows].sum()
                metrics[f"tilt_azimuth_{azimuth_id}_curriculum_success"] = self._sampler.total_successes[
                    azimuth_rows
                ].sum().float() / azimuth_attempts.clamp_min(1)
                metrics[f"tilt_azimuth_{azimuth_id}_full_stack_success"] = self._full_task_successes_by_row[
                    azimuth_rows
                ].sum().float() / azimuth_full_attempts.clamp_min(1)
        resolved_azimuth_rows_by_id = getattr(self, "_resolved_tilt_azimuth_rows", None)
        if resolved_azimuth_rows_by_id is not None:
            for azimuth_id, azimuth_rows in enumerate(resolved_azimuth_rows_by_id):
                azimuth_attempts = attempts[azimuth_rows].sum()
                azimuth_full_attempts = self._full_task_attempts_by_row[azimuth_rows].sum()
                metrics[f"resolved_tilt_azimuth_{azimuth_id}_probability"] = probabilities[azimuth_rows].sum()
                metrics[f"resolved_tilt_azimuth_{azimuth_id}_curriculum_success"] = self._sampler.total_successes[
                    azimuth_rows
                ].sum().float() / azimuth_attempts.clamp_min(1)
                metrics[f"resolved_tilt_azimuth_{azimuth_id}_full_stack_success"] = self._full_task_successes_by_row[
                    azimuth_rows
                ].sum().float() / azimuth_full_attempts.clamp_min(1)
        tilt_magnitude_rows_by_id = getattr(self, "_tilt_magnitude_rows", None)
        if tilt_magnitude_rows_by_id is not None:
            for magnitude_id, magnitude_rows in enumerate(tilt_magnitude_rows_by_id):
                magnitude_attempts = attempts[magnitude_rows].sum()
                magnitude_full_attempts = self._full_task_attempts_by_row[magnitude_rows].sum()
                metrics[f"tilt_magnitude_{magnitude_id}_probability"] = probabilities[magnitude_rows].sum()
                metrics[f"tilt_magnitude_{magnitude_id}_curriculum_success"] = self._sampler.total_successes[
                    magnitude_rows
                ].sum().float() / magnitude_attempts.clamp_min(1)
                metrics[f"tilt_magnitude_{magnitude_id}_full_stack_success"] = self._full_task_successes_by_row[
                    magnitude_rows
                ].sum().float() / magnitude_full_attempts.clamp_min(1)
        return metrics

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return adaptive evidence and replay coverage for an RL checkpoint."""
        state = self._sampler.get_state()
        state["continuation_attempts"] = self._continuation_attempts.clone()
        state["continuation_successes"] = self._continuation_successes.clone()
        state["full_task_attempts_by_row"] = self._full_task_attempts_by_row.clone()
        state["full_task_successes_by_row"] = self._full_task_successes_by_row.clone()
        return state

    def set_state(self, state: dict[str, torch.Tensor]) -> None:
        """Restore adaptive evidence and replay coverage from an RL checkpoint."""
        self._sampler.set_state(state)
        # Older checkpoints predate mixed local/full table resets.
        self._continuation_attempts.copy_(
            state.get("continuation_attempts", torch.zeros_like(self._continuation_attempts)).to(
                device=self._continuation_attempts.device,
                dtype=self._continuation_attempts.dtype,
            )
        )
        self._continuation_successes.copy_(
            state.get("continuation_successes", torch.zeros_like(self._continuation_successes)).to(
                device=self._continuation_successes.device,
                dtype=self._continuation_successes.dtype,
            )
        )
        self._full_task_attempts_by_row.copy_(
            state.get("full_task_attempts_by_row", torch.zeros_like(self._full_task_attempts_by_row)).to(
                device=self._full_task_attempts_by_row.device,
                dtype=self._full_task_attempts_by_row.dtype,
            )
        )
        self._full_task_successes_by_row.copy_(
            state.get("full_task_successes_by_row", torch.zeros_like(self._full_task_successes_by_row)).to(
                device=self._full_task_successes_by_row.device,
                dtype=self._full_task_successes_by_row.dtype,
            )
        )
