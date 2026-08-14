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

from isaaclab_tasks.utils.reset_sampling import (
    AdaptiveResetSampler,
    AdaptiveResetSamplerCfg,
    ResetStateCatalog,
    RollingOutcomeMonitor,
    RollingOutcomeMonitorCfg,
)

from .runtime_state import get_stack_reset_runtime_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class StackResetTableCurriculum(ManagerTermBase):
    """Mix table starts, exact intermediate coverage, and adaptive frontier resets."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        reset_term = env.event_manager.get_term_cfg("reset_from_state_buffer").func
        catalog = getattr(reset_term, "catalog", None)
        if not isinstance(catalog, ResetStateCatalog):
            raise RuntimeError("StackResetTableCurriculum requires StackResetStateTable.")
        self._reset_term = reset_term
        self._catalog = catalog
        if catalog.item_count != catalog.row_count:
            raise ValueError("Stack reset sampling currently requires one competence item per physical row.")
        monitor_cfg = cfg.params.get("outcome_monitor")
        if not isinstance(monitor_cfg, RollingOutcomeMonitorCfg):
            raise TypeError("StackResetTableCurriculum requires a RollingOutcomeMonitorCfg.")
        sampler_cfg = cfg.params.get("adaptive_sampler")
        if not isinstance(sampler_cfg, AdaptiveResetSamplerCfg):
            raise TypeError("StackResetTableCurriculum requires an AdaptiveResetSamplerCfg.")
        self._progress_monitor = RollingOutcomeMonitor(
            catalog.item_count,
            monitor_cfg,
            env.device,
            prior_success_rate=sampler_cfg.target_success_rate,
        )
        self._attempts = torch.zeros(catalog.row_count, dtype=torch.long, device=env.device)
        self._progress_successes = torch.zeros_like(self._attempts)
        self._table_sampling_probability = float(cfg.params.get("table_sampling_probability", 0.35))
        if not 0.0 < self._table_sampling_probability < 1.0:
            raise ValueError("table_sampling_probability must lie strictly between zero and one.")
        expected_intermediate_fraction = 1.0 - self._table_sampling_probability
        if not math.isclose(
            expected_intermediate_fraction * sampler_cfg.coverage_fraction,
            0.15,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError(
                "The stack reset mixture must reserve exactly 15% overall probability for intermediate coverage."
            )
        self._global_sampling = bool(cfg.params.get("global_sampling", False))
        self._evaluation_env_count = int(cfg.params.get("evaluation_env_count", 0))
        if not 0 <= self._evaluation_env_count < env.num_envs:
            raise ValueError("evaluation_env_count must leave at least one curriculum-controlled environment.")
        recipe_ids = catalog.metadata.get("recipe_ids")
        layout_ids = catalog.metadata.get("layout_ids")
        if recipe_ids is None or layout_ids is None:
            raise ValueError("The stack reset catalog must provide recipe_ids and layout_ids metadata.")
        table_recipe_id = reset_term.recipe_names.index("table")
        self._table_rows = recipe_ids == table_recipe_id
        if not bool(torch.any(self._table_rows)) or bool(torch.all(self._table_rows)):
            raise RuntimeError("The stack reset table must contain both table and intermediate rows.")
        self._layout_count = reset_term.layout_count
        self._recipe_rows = tuple(recipe_ids == recipe for recipe in range(len(reset_term.recipe_names)))
        eligible_rows = ~self._table_rows
        base_weights = torch.zeros(catalog.item_count, dtype=torch.float32, device=env.device)
        if self._global_sampling:
            # KUKA authors exactly 6,144 rows for every non-table recipe, so
            # uniform row weights also give every recipe equal initial mass.
            base_weights[eligible_rows] = 1.0
        else:
            # Franka motion recipes intentionally use different interpolation
            # densities. Normalize each recipe/layout cell so authored row
            # count cannot make the two transport families dominate the
            # adaptive frontier merely because they contain more waypoints.
            cell_ids = recipe_ids * self._layout_count + layout_ids
            cell_count = len(reset_term.recipe_names) * self._layout_count
            eligible_per_cell = torch.zeros(cell_count, dtype=torch.float32, device=env.device)
            eligible_per_cell.scatter_add_(
                0,
                cell_ids[eligible_rows],
                torch.ones_like(cell_ids[eligible_rows], dtype=torch.float32),
            )
            base_weights[eligible_rows] = eligible_per_cell[cell_ids[eligible_rows]].reciprocal()
        self._sampler = AdaptiveResetSampler(
            catalog.item_count,
            sampler_cfg,
            env.device,
            eligible_mask=eligible_rows,
            base_weights=base_weights,
        )
        self._table_row_ids = torch.where(self._table_rows)[0]
        self._table_credit = 0.0
        self._table_assignments = 0
        self._table_generator = torch.Generator(device=env.device)
        self._table_generator.manual_seed(int(torch.randint(0, torch.iinfo(torch.int64).max, ()).item()))
        pair_ids = getattr(reset_term, "grasp_pair_ids", None)
        self._grasp_pair_rows = (pair_ids == 0,) if pair_ids is not None else None
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
        self._full_task_attempts_by_row = torch.zeros(catalog.row_count, dtype=torch.long, device=env.device)
        self._full_task_successes_by_row = torch.zeros_like(self._full_task_attempts_by_row)

    def _sampling_distribution(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the long-run 35/15/50 mixture and posterior success rates."""
        success_rates = self._progress_monitor.success_rates
        intermediate = self._sampler.sampling_probabilities(success_rates)
        table = self._table_rows.to(dtype=intermediate.dtype)
        table /= table.sum()
        probabilities = (1.0 - self._table_sampling_probability) * intermediate
        probabilities += self._table_sampling_probability * table
        return probabilities, success_rates

    def _sampling_probabilities(self) -> torch.Tensor:
        """Return the table/coverage/frontier sampling mixture."""
        return self._sampling_distribution()[0]

    def _sample_rows(self, count: int) -> torch.Tensor:
        """Draw rows with exact long-run table and intermediate-stream fractions."""
        rows = torch.empty(count, dtype=torch.long, device=self.device)
        if count == 0:
            return rows
        table_credit = self._table_credit + self._table_sampling_probability * count
        table_count = min(count, math.floor(table_credit + 1.0e-12))
        self._table_credit = table_credit - table_count
        assignment_order = torch.randperm(count, device=self.device, generator=self._table_generator)
        table_positions = assignment_order[:table_count]
        intermediate_positions = assignment_order[table_count:]
        if table_count:
            table_indices = torch.randint(
                self._table_row_ids.numel(),
                (table_count,),
                device=self.device,
                generator=self._table_generator,
            )
            rows[table_positions] = self._table_row_ids[table_indices]
            self._table_assignments += table_count
        if intermediate_positions.numel():
            rows[intermediate_positions] = self._sampler.sample(
                int(intermediate_positions.numel()),
                self._progress_monitor.success_rates,
            )
        return rows

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        outcome_monitor: RollingOutcomeMonitorCfg,
        adaptive_sampler: AdaptiveResetSamplerCfg,
        success_context_name: str = "learning_progress_context",
        final_success_context_name: str = "progress_context",
        table_sampling_probability: float = 0.35,
        global_sampling: bool = False,
        evaluation_env_count: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Record training outcomes, select new rows, and report coverage."""
        del (
            outcome_monitor,
            adaptive_sampler,
            table_sampling_probability,
            global_sampling,
            evaluation_env_count,
        )
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
        # A fixed prefix can be owned by deterministic student evaluation.
        # Those rollouts are assigned directly by the reset event and must not
        # affect either the adaptive sampler's evidence or its next-row draws.
        training_ids = ids[ids >= self._evaluation_env_count]
        batch_success_rate = torch.zeros((), device=env.device)
        batch_full_task_success_rate = torch.zeros((), device=env.device)
        batch_table_full_task_success_rate = torch.zeros((), device=env.device)
        batch_table_full_task_attempts = torch.zeros((), device=env.device)
        if training_ids.numel():
            state = get_stack_reset_runtime_state(env)
            initialized = state.initialized
            row_ids = state.row_ids
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
                completed_items = self._catalog.item_ids_for_rows(completed_rows)
                self._progress_monitor.record(completed_items, succeeded)
                self._attempts.add_(torch.bincount(completed_rows, minlength=self._catalog.row_count))
                self._progress_successes.add_(
                    torch.bincount(completed_rows[succeeded], minlength=self._catalog.row_count)
                )
                self._full_task_attempts_by_row.add_(torch.bincount(completed_rows, minlength=self._catalog.row_count))
                self._full_task_successes_by_row.add_(
                    torch.bincount(completed_rows[final_succeeded], minlength=self._catalog.row_count)
                )

            rows = self._sample_rows(training_ids.numel())
            row_ids[training_ids] = rows
        probabilities, _ = self._sampling_distribution()

        attempts = self._attempts
        observed = attempts > 0
        success_rate = self._progress_successes.sum().float() / attempts.sum().clamp_min(1)
        entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum()
        entropy /= math.log(probabilities.numel())
        active_rows = ~self._table_rows
        unseen_rows = active_rows & ~observed
        rolling_rates = self._progress_monitor.success_rates
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
                ((rolling_rates - self._sampler.cfg.target_success_rate).abs() <= 0.1) & observed
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
        for name, value in self._sampler.metrics(rolling_rates).items():
            metrics[f"sampler_{name}"] = torch.tensor(value, dtype=torch.float32, device=env.device)
        metrics["sampler_table_assignments"] = torch.tensor(
            self._table_assignments,
            dtype=torch.float32,
            device=env.device,
        )
        frontier_probabilities = self._sampler.adaptive_probabilities(rolling_rates)
        for recipe, recipe_rows in enumerate(self._recipe_rows):
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
            metrics[f"recipe_{recipe_name}_frontier_probability"] = frontier_probabilities[recipe_rows].sum()
        pair_rows_by_id = getattr(self, "_grasp_pair_rows", None)
        if pair_rows_by_id is not None:
            for pair_rows, pair_name in zip(pair_rows_by_id, ("index_thumb",), strict=True):
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
        """Return monitor evidence and replay coverage for an explicit state snapshot."""
        state = {
            "total_successes": self._progress_successes.clone(),
            "total_attempts": self._attempts.clone(),
            "continuation_attempts": self._continuation_attempts.clone(),
            "continuation_successes": self._continuation_successes.clone(),
            "full_task_attempts_by_row": self._full_task_attempts_by_row.clone(),
            "full_task_successes_by_row": self._full_task_successes_by_row.clone(),
            "table_credit": torch.tensor(self._table_credit, dtype=torch.float64, device=self.device),
            "table_assignments": torch.tensor(self._table_assignments, dtype=torch.long, device=self.device),
            "table_generator_state": self._table_generator.get_state().clone(),
        }
        state.update({f"monitor/{name}": value for name, value in self._progress_monitor.state_dict().items()})
        state.update({f"sampler/{name}": value for name, value in self._sampler.state_dict().items()})
        return state

    def set_state(self, state: dict[str, torch.Tensor]) -> None:
        """Restore adaptive evidence and replay coverage from an explicit state snapshot."""
        targets = {
            "total_successes": self._progress_successes,
            "total_attempts": self._attempts,
            "continuation_attempts": self._continuation_attempts,
            "continuation_successes": self._continuation_successes,
            "full_task_attempts_by_row": self._full_task_attempts_by_row,
            "full_task_successes_by_row": self._full_task_successes_by_row,
        }
        monitor_state = {
            name.removeprefix("monitor/"): value for name, value in state.items() if name.startswith("monitor/")
        }
        sampler_state = {
            name.removeprefix("sampler/"): value for name, value in state.items() if name.startswith("sampler/")
        }
        expected_names = set(targets) | {
            "table_credit",
            "table_assignments",
            "table_generator_state",
        }
        expected_names |= {f"monitor/{name}" for name in self._progress_monitor.state_dict()}
        expected_names |= {f"sampler/{name}" for name in self._sampler.state_dict()}
        if set(state) != expected_names:
            raise ValueError("Reset-table curriculum checkpoint does not match the current state schema.")
        for name, target in targets.items():
            if state[name].shape != target.shape:
                raise ValueError(
                    f"Reset-table curriculum checkpoint '{name}' has shape {state[name].shape}; "
                    f"expected {target.shape}."
                )
        table_credit = state["table_credit"]
        table_assignments = state["table_assignments"]
        table_generator_state = state["table_generator_state"]
        if table_credit.numel() != 1 or not table_credit.is_floating_point():
            raise ValueError("table_credit must be a scalar floating-point tensor.")
        resolved_table_credit = float(table_credit.item())
        if not math.isfinite(resolved_table_credit) or not 0.0 <= resolved_table_credit < 1.0 + 1.0e-9:
            raise ValueError("table_credit must lie in [0, 1).")
        if (
            table_assignments.numel() != 1
            or table_assignments.dtype == torch.bool
            or table_assignments.is_floating_point()
            or table_assignments.is_complex()
            or int(table_assignments.item()) < 0
        ):
            raise ValueError("table_assignments must be a non-negative scalar integer tensor.")
        if not isinstance(table_generator_state, torch.Tensor):
            raise TypeError("table_generator_state must be a tensor.")
        self._progress_monitor.load_state_dict(monitor_state)
        self._sampler.load_state_dict(sampler_state)
        for name, target in targets.items():
            target.copy_(state[name].to(device=target.device, dtype=target.dtype))
        self._table_credit = min(resolved_table_credit, math.nextafter(1.0, 0.0))
        self._table_assignments = int(table_assignments.item())
        self._table_generator.set_state(table_generator_state.cpu())
