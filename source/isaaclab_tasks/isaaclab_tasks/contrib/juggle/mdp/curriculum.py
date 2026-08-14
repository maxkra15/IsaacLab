# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adaptive phase-balanced reset sampling for one-ball juggling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

from isaaclab_tasks.utils.reset_sampling import (
    AdaptiveResetSampler,
    AdaptiveResetSamplerCfg,
    RollingOutcomeMonitor,
    RollingOutcomeMonitorCfg,
)

from .reset import JugglePhase, JuggleResetEvent
from .runtime import get_juggle_runtime_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class JuggleResetCurriculum(ManagerTermBase):
    """Mix held deployment starts, exact phase coverage, and an adaptive frontier."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        reset_term = env.event_manager.get_term_cfg("reset_from_catalog").func
        if not isinstance(reset_term, JuggleResetEvent):
            raise RuntimeError("JuggleResetCurriculum requires the JuggleResetEvent.")
        self._reset_term = reset_term
        monitor_cfg = cfg.params.get("outcome_monitor")
        sampler_cfg = cfg.params.get("adaptive_sampler")
        if not isinstance(monitor_cfg, RollingOutcomeMonitorCfg):
            raise TypeError("outcome_monitor must be a RollingOutcomeMonitorCfg.")
        if not isinstance(sampler_cfg, AdaptiveResetSamplerCfg):
            raise TypeError("adaptive_sampler must be an AdaptiveResetSamplerCfg.")
        self._canonical_fraction = float(cfg.params.get("canonical_fraction", 0.35))
        if not 0.0 < self._canonical_fraction < 1.0:
            raise ValueError("canonical_fraction must lie strictly between zero and one.")
        self._monitor = RollingOutcomeMonitor(
            item_count=reset_term.catalog.item_count,
            cfg=monitor_cfg,
            device=env.device,
            prior_success_rate=sampler_cfg.target_success_rate,
        )
        eligible = torch.ones(reset_term.catalog.item_count, dtype=torch.bool, device=env.device)
        eligible[int(JugglePhase.HELD_PRETHROW)] = False
        self._sampler = AdaptiveResetSampler(
            item_count=reset_term.catalog.item_count,
            cfg=sampler_cfg,
            device=env.device,
            eligible_mask=eligible,
        )
        self._phase_rows = torch.stack(reset_term.source.phase_rows)
        self._row_generator = torch.Generator(device=env.device)
        seed = int(torch.randint(0, torch.iinfo(torch.int64).max, (), device="cpu").item())
        self._row_generator.manual_seed(seed)
        self._canonical_credit = 0.0
        self._attempts = torch.zeros(reset_term.catalog.item_count, dtype=torch.long, device=env.device)
        self._local_successes = torch.zeros_like(self._attempts)
        self._cycle_successes = torch.zeros_like(self._attempts)
        self._static_held_attempts = torch.zeros((), dtype=torch.long, device=env.device)
        self._static_held_successes = torch.zeros_like(self._static_held_attempts)
        # Accumulate one environment-sized reporting window.  CurriculumManager converts every
        # tensor metric to a Python scalar, so returning cached floats avoids synchronizing once
        # per metric whenever only a few failed environments reset.
        self._completed_since_metrics = 0
        self._window_attempts = torch.zeros((), dtype=torch.long, device=env.device)
        self._window_local_successes = torch.zeros_like(self._window_attempts)
        self._window_cycle_successes = torch.zeros_like(self._window_attempts)
        self._window_phase_attempts = torch.zeros_like(self._attempts)
        self._window_phase_local_successes = torch.zeros_like(self._attempts)
        self._window_static_held_attempts = torch.zeros_like(self._window_attempts)
        self._window_static_held_successes = torch.zeros_like(self._window_attempts)
        self._cached_metrics: dict[str, float] = {
            "recent_local_success_rate": 0.0,
            "recent_full_cycle_success_rate": 0.0,
            "local_success_rate": 0.0,
            "full_cycle_success_rate": 0.0,
            "canonical_full_cycle_success_rate": 0.0,
            "phase_coverage": 0.0,
            "recent_static_held_attempts": 0.0,
            "recent_static_held_full_cycle_successes": 0.0,
            "recent_static_held_full_cycle_success_rate": 0.0,
            "static_held_attempts": 0.0,
            "static_held_full_cycle_successes": 0.0,
            "static_held_full_cycle_success_rate": 0.0,
        }
        for phase in JugglePhase:
            phase_name = phase.name.lower()
            self._cached_metrics[f"recent_{phase_name}_local_success_rate"] = 0.0
            self._cached_metrics[f"{phase_name}_local_success_rate"] = 0.0
            self._cached_metrics[f"{phase_name}_sampling_probability"] = 0.0
        self._metric_names = tuple(self._cached_metrics)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        outcome_monitor: RollingOutcomeMonitorCfg,
        adaptive_sampler: AdaptiveResetSamplerCfg,
        canonical_fraction: float = 0.35,
    ) -> dict[str, torch.Tensor | float]:
        """Credit completed episodes, assign new rows, and return training diagnostics."""
        del outcome_monitor, adaptive_sampler, canonical_fraction
        ids = _resolve_env_ids(env, env_ids)
        state = get_juggle_runtime_state(env)
        completed = state.initialized[ids] & (env.episode_length_buf[ids] > 0)
        completed_ids = ids[completed]
        if completed_ids.numel():
            phases = state.start_phases[completed_ids]
            local_success = state.local_success[completed_ids]
            cycle_success = state.cycle_success[completed_ids]
            static_held = state.static_held_start[completed_ids] & (phases == int(JugglePhase.HELD_PRETHROW))
            static_held_success = static_held & cycle_success
            self._monitor.record(phases, local_success)
            self._attempts.add_(torch.bincount(phases, minlength=len(JugglePhase)))
            self._local_successes.add_(torch.bincount(phases[local_success], minlength=len(JugglePhase)))
            self._cycle_successes.add_(torch.bincount(phases[cycle_success], minlength=len(JugglePhase)))
            self._static_held_attempts.add_(static_held.sum())
            self._static_held_successes.add_(static_held_success.sum())
            self._completed_since_metrics += completed_ids.numel()
            self._window_attempts.add_(completed_ids.numel())
            self._window_local_successes.add_(local_success.sum())
            self._window_cycle_successes.add_(cycle_success.sum())
            self._window_phase_attempts.add_(torch.bincount(phases, minlength=len(JugglePhase)))
            self._window_phase_local_successes.add_(torch.bincount(phases[local_success], minlength=len(JugglePhase)))
            self._window_static_held_attempts.add_(static_held.sum())
            self._window_static_held_successes.add_(static_held_success.sum())

        if ids.numel():
            state.row_ids[ids] = self._sample_rows(ids.numel())

        if self._completed_since_metrics >= env.num_envs:
            attempts = self._attempts
            aggregate_metrics = torch.stack(
                (
                    self._window_local_successes.float() / self._window_attempts.clamp_min(1),
                    self._window_cycle_successes.float() / self._window_attempts.clamp_min(1),
                    self._local_successes.sum().float() / attempts.sum().clamp_min(1),
                    self._cycle_successes.sum().float() / attempts.sum().clamp_min(1),
                    self._cycle_successes[int(JugglePhase.HELD_PRETHROW)].float()
                    / attempts[int(JugglePhase.HELD_PRETHROW)].clamp_min(1),
                    (attempts > 0).float().mean(),
                    self._window_static_held_attempts.float(),
                    self._window_static_held_successes.float(),
                    self._window_static_held_successes.float() / self._window_static_held_attempts.clamp_min(1),
                    self._static_held_attempts.float(),
                    self._static_held_successes.float(),
                    self._static_held_successes.float() / self._static_held_attempts.clamp_min(1),
                )
            )
            recent_phase_rates = self._window_phase_local_successes.float() / self._window_phase_attempts.clamp_min(1)
            cumulative_phase_rates = self._local_successes.float() / attempts.clamp_min(1)
            sampling_probabilities = self.phase_probabilities()
            phase_metrics = torch.stack(
                tuple(
                    value[phase_id]
                    for phase_id in range(len(JugglePhase))
                    for value in (recent_phase_rates, cumulative_phase_rates, sampling_probabilities)
                )
            )
            metric_values = torch.cat((aggregate_metrics, phase_metrics))
            self._cached_metrics = dict(zip(self._metric_names, metric_values.detach().cpu().tolist(), strict=True))
            self._completed_since_metrics = 0
            self._window_attempts.zero_()
            self._window_local_successes.zero_()
            self._window_cycle_successes.zero_()
            self._window_phase_attempts.zero_()
            self._window_phase_local_successes.zero_()
            self._window_static_held_attempts.zero_()
            self._window_static_held_successes.zero_()
        return self._cached_metrics

    def phase_probabilities(self) -> torch.Tensor:
        """Return the long-run phase probabilities of all three reset streams."""
        probabilities = torch.zeros(len(JugglePhase), dtype=torch.float32, device=self.device)
        probabilities[int(JugglePhase.HELD_PRETHROW)] = self._canonical_fraction
        probabilities += (1.0 - self._canonical_fraction) * self._sampler.sampling_probabilities(
            self._monitor.success_rates
        )
        return probabilities

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return a flat, tensor-only snapshot of all sampling state."""
        state = {
            "canonical_credit": torch.tensor(self._canonical_credit, dtype=torch.float64, device=self.device),
            "attempts": self._attempts.clone(),
            "local_successes": self._local_successes.clone(),
            "cycle_successes": self._cycle_successes.clone(),
            "static_held_attempts": self._static_held_attempts.clone(),
            "static_held_successes": self._static_held_successes.clone(),
            "completed_since_metrics": torch.tensor(
                self._completed_since_metrics, dtype=torch.long, device=self.device
            ),
            "window_attempts": self._window_attempts.clone(),
            "window_local_successes": self._window_local_successes.clone(),
            "window_cycle_successes": self._window_cycle_successes.clone(),
            "window_phase_attempts": self._window_phase_attempts.clone(),
            "window_phase_local_successes": self._window_phase_local_successes.clone(),
            "window_static_held_attempts": self._window_static_held_attempts.clone(),
            "window_static_held_successes": self._window_static_held_successes.clone(),
            "cached_metrics": torch.tensor(
                tuple(self._cached_metrics.values()), dtype=torch.float64, device=self.device
            ),
            "row_generator_state": self._row_generator.get_state().clone(),
        }
        state.update({f"monitor__{name}": value for name, value in self._monitor.state_dict().items()})
        state.update({f"sampler__{name}": value for name, value in self._sampler.state_dict().items()})
        return state

    def set_state(self, state: dict[str, torch.Tensor]) -> None:
        """Restore state returned by :meth:`get_state`."""
        monitor_keys = {f"monitor__{name}" for name in self._monitor.state_dict()}
        sampler_keys = {f"sampler__{name}" for name in self._sampler.state_dict()}
        expected = {
            "canonical_credit",
            "attempts",
            "local_successes",
            "cycle_successes",
            "static_held_attempts",
            "static_held_successes",
            "completed_since_metrics",
            "window_attempts",
            "window_local_successes",
            "window_cycle_successes",
            "window_phase_attempts",
            "window_phase_local_successes",
            "window_static_held_attempts",
            "window_static_held_successes",
            "cached_metrics",
            "row_generator_state",
            *monitor_keys,
            *sampler_keys,
        }
        if set(state) != expected:
            raise ValueError(f"Juggle curriculum state keys must be exactly {sorted(expected)}.")
        self._monitor.load_state_dict({name.removeprefix("monitor__"): state[name] for name in monitor_keys})
        self._sampler.load_state_dict({name.removeprefix("sampler__"): state[name] for name in sampler_keys})
        credit_tensor = state["canonical_credit"]
        if not isinstance(credit_tensor, torch.Tensor) or credit_tensor.numel() != 1:
            raise ValueError("canonical_credit must be a scalar tensor.")
        credit = float(credit_tensor.item())
        if not 0.0 <= credit < 1.0:
            raise ValueError("canonical_credit must lie in [0, 1).")
        self._canonical_credit = credit
        generator_state = state["row_generator_state"]
        if not isinstance(generator_state, torch.Tensor):
            raise TypeError("row_generator_state must be a tensor.")
        self._row_generator.set_state(generator_state.cpu())
        for name, target in (
            ("attempts", self._attempts),
            ("local_successes", self._local_successes),
            ("cycle_successes", self._cycle_successes),
            ("static_held_attempts", self._static_held_attempts),
            ("static_held_successes", self._static_held_successes),
            ("window_attempts", self._window_attempts),
            ("window_local_successes", self._window_local_successes),
            ("window_cycle_successes", self._window_cycle_successes),
            ("window_phase_attempts", self._window_phase_attempts),
            ("window_phase_local_successes", self._window_phase_local_successes),
            ("window_static_held_attempts", self._window_static_held_attempts),
            ("window_static_held_successes", self._window_static_held_successes),
        ):
            values = state[name]
            if not isinstance(values, torch.Tensor) or values.shape != target.shape:
                raise ValueError(f"{name} has an incompatible shape.")
            if bool((values < 0).any()):
                raise ValueError(f"{name} must be non-negative.")
            target.copy_(values.to(device=self.device, dtype=target.dtype))
        completed_tensor = state["completed_since_metrics"]
        if not isinstance(completed_tensor, torch.Tensor) or completed_tensor.numel() != 1:
            raise ValueError("completed_since_metrics must be a scalar tensor.")
        completed = int(completed_tensor.item())
        if not 0 <= completed < self.num_envs or completed != int(self._window_attempts.item()):
            raise ValueError("completed_since_metrics is inconsistent with the reporting window.")
        cached_metrics = state["cached_metrics"]
        if not isinstance(cached_metrics, torch.Tensor) or cached_metrics.shape != (len(self._metric_names),):
            raise ValueError("cached_metrics has an incompatible shape.")
        if not bool(torch.isfinite(cached_metrics).all()):
            raise ValueError("cached_metrics must be finite.")
        if bool((self._local_successes > self._attempts).any()) or bool((self._cycle_successes > self._attempts).any()):
            raise ValueError("cumulative successes cannot exceed attempts.")
        held_phase = int(JugglePhase.HELD_PRETHROW)
        if (
            self._static_held_successes > self._static_held_attempts
            or self._static_held_attempts > self._attempts[held_phase]
        ):
            raise ValueError("static held cumulative counts are inconsistent.")
        if self._window_local_successes > self._window_attempts or self._window_cycle_successes > self._window_attempts:
            raise ValueError("reporting-window successes cannot exceed attempts.")
        if (
            self._window_phase_attempts.sum() != self._window_attempts
            or self._window_phase_local_successes.sum() != self._window_local_successes
            or bool((self._window_phase_local_successes > self._window_phase_attempts).any())
            or self._window_static_held_successes > self._window_static_held_attempts
            or self._window_static_held_attempts > self._window_phase_attempts[held_phase]
        ):
            raise ValueError("reporting-window counts are inconsistent.")
        self._completed_since_metrics = completed
        self._cached_metrics = dict(
            zip(self._metric_names, cached_metrics.to(device="cpu", dtype=torch.float64).tolist(), strict=True)
        )

    def _sample_rows(self, count: int) -> torch.Tensor:
        """Sample physical rows while preserving the configured stream fractions."""
        credit = self._canonical_credit + self._canonical_fraction * count
        canonical_count = min(count, int(credit + 1.0e-12))
        self._canonical_credit = credit - canonical_count
        assignment_order = torch.randperm(count, device=self.device, generator=self._row_generator)
        rows = torch.empty(count, dtype=torch.long, device=self.device)
        canonical_positions = assignment_order[:canonical_count]
        frontier_positions = assignment_order[canonical_count:]
        held_rows = self._phase_rows[int(JugglePhase.HELD_PRETHROW)]
        if canonical_count:
            rows[canonical_positions] = held_rows[
                torch.randint(
                    held_rows.numel(),
                    (canonical_count,),
                    device=self.device,
                    generator=self._row_generator,
                )
            ]
        if frontier_positions.numel():
            phases = self._sampler.sample(int(frontier_positions.numel()), self._monitor.success_rates)
            row_offsets = torch.randint(
                self._phase_rows.shape[1],
                (frontier_positions.numel(),),
                device=self.device,
                generator=self._row_generator,
            )
            rows[frontier_positions] = self._phase_rows[phases, row_offsets]
        return rows


def _resolve_env_ids(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Return reset environment ids as a flat device tensor."""
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)[env_ids]
    return torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
