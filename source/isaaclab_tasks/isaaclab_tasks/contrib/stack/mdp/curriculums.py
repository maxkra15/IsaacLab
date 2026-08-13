# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success-monitored reset sampling for cube stacking."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

from isaaclab_tasks.core.lift.mdp.events_cfg import SuccessMonitorCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class StackResetTableCurriculum(ManagerTermBase):
    """Mix guaranteed table starts with target-rate reset sampling."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        reset_term = env.event_manager.get_term_cfg("reset_from_state_buffer").func
        if not hasattr(reset_term, "row_count"):
            raise RuntimeError("StackResetTableCurriculum requires StackResetStateTable.")
        self._reset_term = reset_term
        monitor_cfg = cfg.params.get("success_monitor")
        if not isinstance(monitor_cfg, SuccessMonitorCfg):
            raise TypeError("StackResetTableCurriculum requires a SuccessMonitorCfg.")
        self._progress_monitor = monitor_cfg.class_type(
            monitor_cfg,
            num_partitions=1,
            partition_size=reset_term.row_count,
            device=env.device,
        )
        self._attempts = torch.zeros(reset_term.row_count, dtype=torch.long, device=env.device)
        self._progress_successes = torch.zeros_like(self._attempts)
        self._table_sampling_probability = float(cfg.params.get("table_sampling_probability", 0.35))
        if not 0.0 < self._table_sampling_probability < 1.0:
            raise ValueError("table_sampling_probability must lie strictly between zero and one.")
        self._balance_recipes = bool(cfg.params.get("balance_recipes", False))
        self._balance_reset_modes = bool(cfg.params.get("balance_reset_modes", False))
        self._global_sampling = bool(cfg.params.get("global_sampling", False))
        self._evaluation_env_count = int(cfg.params.get("evaluation_env_count", 0))
        if not 0 <= self._evaluation_env_count < env.num_envs:
            raise ValueError("evaluation_env_count must leave at least one curriculum-controlled environment.")
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
        """Return mixture probabilities and target-rate weights.

        Layout-balanced tasks normalize within each workspace layout. Reset
        banks may instead normalize within recipe-layout strata or, for
        pair-conditioned hands, within recipe/pair/yaw/tilt strata. The latter
        guarantees that a hard finger or wrist mode cannot disappear while the
        target-rate weight selects the useful progress frontier inside that
        mode.
        """
        target_weights = self._progress_monitor.target_weights()
        adaptive = target_weights.clone()
        adaptive[self._table_rows] = 0.0
        layout_ids = self._reset_term.layout_ids
        layout_count = getattr(self, "_layout_count", self._reset_term.layout_count)
        if getattr(self, "_global_sampling", False):
            # Normalize one target-rate weight vector over the complete active
            # table without layout, recipe, or mode quotas.
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
        return probabilities, target_weights

    def _sampling_probabilities(self) -> torch.Tensor:
        """Return the fixed-table/adaptive-intermediate sampling mixture."""
        return self._sampling_distribution()[0]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        success_monitor: SuccessMonitorCfg,
        success_context_name: str = "learning_progress_context",
        final_success_context_name: str = "progress_context",
        table_sampling_probability: float = 0.35,
        balance_recipes: bool = False,
        balance_reset_modes: bool = False,
        global_sampling: bool = False,
        evaluation_env_count: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Record training outcomes, select new rows, and report coverage."""
        del (
            success_monitor,
            table_sampling_probability,
            balance_recipes,
            balance_reset_modes,
            global_sampling,
            evaluation_env_count,
        )
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
        # A fixed prefix can be owned by deterministic student evaluation.
        # Those rollouts are assigned directly by the reset event and must not
        # affect either the adaptive sampler's evidence or its next-row draws.
        training_ids = ids[ids >= getattr(self, "_evaluation_env_count", 0)]
        batch_success_rate = torch.zeros((), device=env.device)
        batch_full_task_success_rate = torch.zeros((), device=env.device)
        batch_table_full_task_success_rate = torch.zeros((), device=env.device)
        batch_table_full_task_attempts = torch.zeros((), device=env.device)
        if training_ids.numel():
            state = getattr(env, "stack_reset_state", None)
            initialized = state.initialized if state is not None else env.stack_reset_initialized
            row_ids = state.row_ids if state is not None else env.stack_reset_row_ids
            completed = initialized[training_ids] & (env.episode_length_buf[training_ids] > 0)
            completed_ids = training_ids[completed]
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
                self._progress_monitor.success_update(completed_rows, succeeded)
                self._attempts.add_(torch.bincount(completed_rows, minlength=self._reset_term.row_count))
                self._progress_successes.add_(
                    torch.bincount(completed_rows[succeeded], minlength=self._reset_term.row_count)
                )
                self._full_task_attempts_by_row.add_(
                    torch.bincount(completed_rows, minlength=self._reset_term.row_count)
                )
                self._full_task_successes_by_row.add_(
                    torch.bincount(completed_rows[final_succeeded], minlength=self._reset_term.row_count)
                )

            probabilities, _ = self._sampling_distribution()
            rows = torch.multinomial(probabilities, training_ids.numel(), replacement=True)
            row_ids[training_ids] = rows
        else:
            probabilities, _ = self._sampling_distribution()

        attempts = self._attempts
        observed = attempts > 0
        success_rate = self._progress_successes.sum().float() / attempts.sum().clamp_min(1)
        entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum()
        entropy /= math.log(probabilities.numel())
        active_rows = ~self._table_rows
        unseen_rows = active_rows & ~observed
        rolling_rates = self._progress_monitor.success_rate
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
            "unseen_row_probability_mass": probabilities[unseen_rows].sum(),
            "table_probability": probabilities[self._table_rows].sum(),
            "target_band_fraction": (
                ((rolling_rates - self._progress_monitor.cfg.target_success_rate).abs() <= 0.1) & observed
            ).sum()
            / observed_count,
            "full_task_attempts": self._continuation_attempts.float(),
            "full_task_success_rate": self._continuation_successes.float() / self._continuation_attempts.clamp_min(1),
            "table_curriculum_success_rate": self._progress_successes[self._table_rows].sum().float()
            / self._attempts[self._table_rows].sum().clamp_min(1),
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
            metrics[f"recipe_{recipe_name}_curriculum_success"] = self._progress_successes[
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
                metrics[f"pair_{pair_name}_curriculum_success"] = self._progress_successes[
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
                metrics[f"orientation_{orientation_id}_curriculum_success"] = self._progress_successes[
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
                metrics[f"tilt_azimuth_{azimuth_id}_curriculum_success"] = self._progress_successes[
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
                metrics[f"resolved_tilt_azimuth_{azimuth_id}_curriculum_success"] = self._progress_successes[
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
                metrics[f"tilt_magnitude_{magnitude_id}_curriculum_success"] = self._progress_successes[
                    magnitude_rows
                ].sum().float() / magnitude_attempts.clamp_min(1)
                metrics[f"tilt_magnitude_{magnitude_id}_full_stack_success"] = self._full_task_successes_by_row[
                    magnitude_rows
                ].sum().float() / magnitude_full_attempts.clamp_min(1)
        return metrics

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return monitor evidence and replay coverage for an RL checkpoint.

        The field names intentionally match checkpoints written by the former
        stack-local monitor so existing policies can resume with the shared
        :class:`SuccessMonitor` implementation.
        """
        history = self._progress_monitor.success_buf.bool()
        return {
            "success_rates": self._progress_monitor.success_rate.clone(),
            "success_history": history.clone(),
            "history_pointer": self._progress_monitor.success_pointer.clone(),
            "history_size": self._progress_monitor.success_size.clone(),
            "history_success_count": history.sum(dim=1).long(),
            "total_successes": self._progress_successes.clone(),
            "total_attempts": self._attempts.clone(),
            "continuation_attempts": self._continuation_attempts.clone(),
            "continuation_successes": self._continuation_successes.clone(),
            "full_task_attempts_by_row": self._full_task_attempts_by_row.clone(),
            "full_task_successes_by_row": self._full_task_successes_by_row.clone(),
        }

    def set_state(self, state: dict[str, torch.Tensor]) -> None:
        """Restore adaptive evidence and replay coverage from an RL checkpoint."""
        targets = {
            "success_rates": self._progress_monitor.success_rate,
            "success_history": self._progress_monitor.success_buf,
            "history_pointer": self._progress_monitor.success_pointer,
            "history_size": self._progress_monitor.success_size,
            "total_successes": self._progress_successes,
            "total_attempts": self._attempts,
        }
        for name, target in targets.items():
            if name not in state:
                raise KeyError(f"Reset-table curriculum checkpoint is missing '{name}'.")
            if state[name].shape != target.shape:
                raise ValueError(
                    f"Reset-table curriculum checkpoint '{name}' has shape {state[name].shape}; "
                    f"expected {target.shape}."
                )
        history_len = self._progress_monitor.cfg.monitored_history_len
        if bool(torch.any((state["history_pointer"] < 0) | (state["history_pointer"] >= history_len))):
            raise ValueError("Reset-table curriculum checkpoint contains an invalid history pointer.")
        if bool(torch.any((state["history_size"] < 0) | (state["history_size"] > history_len))):
            raise ValueError("Reset-table curriculum checkpoint contains an invalid history size.")
        for name, target in targets.items():
            target.copy_(state[name].to(device=target.device, dtype=target.dtype))
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
