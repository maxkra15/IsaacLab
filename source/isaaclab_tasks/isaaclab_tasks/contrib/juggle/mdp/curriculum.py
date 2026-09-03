# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adaptive phase-balanced reset sampling for one-ball juggling."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

from isaaclab_tasks.utils.reset_sampling import (
    AdaptiveResetSampler,
    AdaptiveResetSamplerCfg,
    ContinuousAdaptiveResetSampler,
    ContinuousAdaptiveResetSamplerCfg,
    RollingOutcomeMonitor,
    RollingOutcomeMonitorCfg,
)

from .reset import JugglePhase, JuggleResetEvent
from .runtime import get_juggle_runtime_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class JuggleResetCurriculum(ManagerTermBase):
    """Mix randomized canonical starts with adaptive physical-phase resets."""

    checkpoint_state_enabled = True
    """Persist the adaptive reset state in training checkpoints."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        reset_term = env.event_manager.get_term_cfg("reset_from_catalog").func
        if not isinstance(reset_term, JuggleResetEvent):
            raise RuntimeError("JuggleResetCurriculum requires the JuggleResetEvent.")
        self._reset_term = reset_term
        monitor_cfg = cfg.params.get("outcome_monitor")
        sampler_cfg = cfg.params.get("adaptive_sampler")
        continuous_cfg = cfg.params.get("continuous_sampler")
        if not isinstance(monitor_cfg, RollingOutcomeMonitorCfg):
            raise TypeError("outcome_monitor must be a RollingOutcomeMonitorCfg.")
        if not isinstance(sampler_cfg, AdaptiveResetSamplerCfg):
            raise TypeError("adaptive_sampler must be an AdaptiveResetSamplerCfg.")
        self._sampling_mode = str(cfg.params.get("sampling_mode", "semantic"))
        if self._sampling_mode not in ("uniform", "semantic", "continuous"):
            raise ValueError("sampling_mode must be 'uniform', 'semantic', or 'continuous'.")
        if self._sampling_mode == "continuous" and not isinstance(continuous_cfg, ContinuousAdaptiveResetSamplerCfg):
            raise TypeError("continuous_sampler must be a ContinuousAdaptiveResetSamplerCfg in continuous mode.")
        if self._sampling_mode == "continuous" and reset_term.source.parameter_sampling != "continuous":
            raise ValueError("Continuous curriculum mode requires continuously parameterized reset proposals.")
        if self._sampling_mode != "continuous" and reset_term.source.parameter_sampling != "catalog":
            raise ValueError("Uniform and semantic curriculum modes require the catalog reset source.")
        self._canonical_fraction = float(cfg.params.get("canonical_fraction", 0.35))
        if not 0.0 <= self._canonical_fraction < 1.0:
            raise ValueError("canonical_fraction must lie in [0, 1).")
        self._monitor = RollingOutcomeMonitor(
            item_count=reset_term.catalog.item_count,
            cfg=monitor_cfg,
            device=env.device,
            prior_success_rate=sampler_cfg.target_success_rate,
        )
        self._item_phase_ids = reset_term.source.item_phase_ids
        self._canonical_item_mask = reset_term.source.canonical_item_mask
        adaptive_eligible = reset_term.source.adaptive_item_mask
        self._sampler = AdaptiveResetSampler(
            item_count=reset_term.catalog.item_count,
            cfg=sampler_cfg,
            device=env.device,
            eligible_mask=adaptive_eligible,
        )
        proposal_eligible = adaptive_eligible
        if self._sampling_mode == "continuous":
            # Continuous mode learns over the entire trainable reset domain.
            # Canonical starts are ordinary candidates rather than a separate
            # fixed stream; only deliberately non-trainable rows stay masked.
            proposal_eligible = adaptive_eligible | self._canonical_item_mask
        self._eligible_rows = torch.where(proposal_eligible[reset_term.source.item_ids])[0]
        self._continuous_sampler: ContinuousAdaptiveResetSampler | None = None
        if self._sampling_mode == "continuous":
            assert isinstance(continuous_cfg, ContinuousAdaptiveResetSamplerCfg)
            continuous_generator = torch.Generator(device=env.device)
            continuous_seed = int(torch.randint(0, torch.iinfo(torch.int64).max, (), device="cpu").item())
            continuous_generator.manual_seed(continuous_seed)
            self._continuous_sampler = ContinuousAdaptiveResetSampler(
                candidate_features=reset_term.source.model_features,
                group_ids=reset_term.source.item_ids,
                cfg=continuous_cfg,
                device=env.device,
                eligible_mask=proposal_eligible[reset_term.source.item_ids],
                generator=continuous_generator,
            )
        sampled_items = self._canonical_item_mask | proposal_eligible
        self._sampled_phase_mask = torch.bincount(
            self._item_phase_ids[sampled_items], minlength=len(JugglePhase)
        ).bool()
        self._canonical_rows = reset_term.source.canonical_row_ids
        canonical_item_ids = reset_term.source.item_ids[self._canonical_rows]
        self._canonical_item_probabilities = torch.bincount(
            canonical_item_ids,
            minlength=reset_term.catalog.item_count,
        ).float()
        self._canonical_item_probabilities /= self._canonical_item_probabilities.sum()

        item_row_counts = torch.tensor(
            [rows.numel() for rows in reset_term.source.item_rows],
            dtype=torch.long,
            device=env.device,
        )
        max_item_rows = int(item_row_counts.max().item())
        self._item_rows = torch.zeros(
            (reset_term.catalog.item_count, max_item_rows),
            dtype=torch.long,
            device=env.device,
        )
        for item_id, item_rows in enumerate(reset_term.source.item_rows):
            self._item_rows[item_id, : item_rows.numel()] = item_rows
        self._item_row_counts = item_row_counts
        self._uniform_item_row_count = bool((item_row_counts == item_row_counts[0]).all())
        self._uniform_rows_per_item = int(item_row_counts[0].item()) if self._uniform_item_row_count else 0
        self._row_generator = torch.Generator(device=env.device)
        seed = int(torch.randint(0, torch.iinfo(torch.int64).max, (), device="cpu").item())
        self._row_generator.manual_seed(seed)
        self._canonical_credit = 0.0
        self._attempts = torch.zeros(reset_term.catalog.item_count, dtype=torch.long, device=env.device)
        self._local_successes = torch.zeros_like(self._attempts)
        self._cycle_successes = torch.zeros_like(self._attempts)
        self._static_held_attempts = torch.zeros((), dtype=torch.long, device=env.device)
        self._static_held_local_successes = torch.zeros_like(self._static_held_attempts)
        self._static_held_successes = torch.zeros_like(self._static_held_attempts)
        # Accumulate one environment-sized reporting window.  CurriculumManager converts every
        # tensor metric to a Python scalar, so returning cached floats avoids synchronizing once
        # per metric whenever only a few failed environments reset.
        self._completed_since_metrics = 0
        self._window_attempts = torch.zeros((), dtype=torch.long, device=env.device)
        self._window_local_successes = torch.zeros_like(self._window_attempts)
        self._window_cycle_successes = torch.zeros_like(self._window_attempts)
        self._window_item_attempts = torch.zeros_like(self._attempts)
        self._window_item_local_successes = torch.zeros_like(self._attempts)
        self._window_phase_attempts = torch.zeros(len(JugglePhase), dtype=torch.long, device=env.device)
        self._window_phase_local_successes = torch.zeros_like(self._window_phase_attempts)
        self._window_static_held_attempts = torch.zeros_like(self._window_attempts)
        self._window_static_held_local_successes = torch.zeros_like(self._window_attempts)
        self._window_static_held_successes = torch.zeros_like(self._window_attempts)
        self._cached_metrics: dict[str, float] = {
            "recent_local_success_rate": 0.0,
            "recent_full_cycle_success_rate": 0.0,
            "local_success_rate": 0.0,
            "full_cycle_success_rate": 0.0,
            "canonical_full_cycle_success_rate": 0.0,
            "phase_coverage": 0.0,
            "recent_static_held_attempts": 0.0,
            "recent_static_held_local_successes": 0.0,
            "recent_static_held_local_success_rate": 0.0,
            "recent_static_held_full_cycle_successes": 0.0,
            "recent_static_held_full_cycle_success_rate": 0.0,
            "static_held_attempts": 0.0,
            "static_held_local_successes": 0.0,
            "static_held_local_success_rate": 0.0,
            "static_held_full_cycle_successes": 0.0,
            "static_held_full_cycle_success_rate": 0.0,
        }
        for phase in JugglePhase:
            phase_name = phase.name.lower()
            self._cached_metrics[f"recent_{phase_name}_local_success_rate"] = 0.0
            self._cached_metrics[f"{phase_name}_local_success_rate"] = 0.0
            self._cached_metrics[f"{phase_name}_sampling_probability"] = 0.0
        if self._continuous_sampler is not None:
            for metric_name in (
                "history_count",
                "predicted_success_mean",
                "predicted_success_minimum",
                "predicted_success_maximum",
                "effective_pool_fraction",
                "top_1_percent_mass",
                "realized_coverage_fraction",
            ):
                self._cached_metrics[f"continuous_{metric_name}"] = 0.0
        self._metric_names = tuple(self._cached_metrics)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        outcome_monitor: RollingOutcomeMonitorCfg,
        adaptive_sampler: AdaptiveResetSamplerCfg,
        canonical_fraction: float = 0.35,
        sampling_mode: str = "semantic",
        continuous_sampler: ContinuousAdaptiveResetSamplerCfg | None = None,
    ) -> dict[str, torch.Tensor | float]:
        """Credit completed episodes, assign new rows, and return training diagnostics."""
        del (
            outcome_monitor,
            adaptive_sampler,
            canonical_fraction,
            sampling_mode,
            continuous_sampler,
        )
        ids = _resolve_env_ids(env, env_ids)
        state = get_juggle_runtime_state(env)
        completed = state.initialized[ids] & (env.episode_length_buf[ids] > 0)
        completed_ids = ids[completed]
        if completed_ids.numel():
            item_ids = state.start_item_ids[completed_ids]
            phases = state.start_phases[completed_ids]
            local_success = state.local_success[completed_ids]
            cycle_success = state.cycle_success[completed_ids]
            static_held = state.static_held_start[completed_ids] & (phases == int(JugglePhase.HELD_PRETHROW))
            static_held_local_success = static_held & local_success
            static_held_success = static_held & cycle_success
            item_attempts = torch.bincount(item_ids, minlength=self._reset_term.catalog.item_count)
            item_local_successes = torch.bincount(
                item_ids[local_success],
                minlength=self._reset_term.catalog.item_count,
            )
            self._monitor.record(item_ids, local_success)
            if self._continuous_sampler is not None:
                self._continuous_sampler.record(state.row_ids[completed_ids], local_success)
            self._attempts.add_(item_attempts)
            self._local_successes.add_(item_local_successes)
            self._cycle_successes.add_(
                torch.bincount(item_ids[cycle_success], minlength=self._reset_term.catalog.item_count)
            )
            self._static_held_attempts.add_(static_held.sum())
            self._static_held_local_successes.add_(static_held_local_success.sum())
            self._static_held_successes.add_(static_held_success.sum())
            self._completed_since_metrics += completed_ids.numel()
            self._window_attempts.add_(completed_ids.numel())
            self._window_local_successes.add_(local_success.sum())
            self._window_cycle_successes.add_(cycle_success.sum())
            self._window_item_attempts.add_(item_attempts)
            self._window_item_local_successes.add_(item_local_successes)
            self._window_phase_attempts.add_(torch.bincount(phases, minlength=len(JugglePhase)))
            self._window_phase_local_successes.add_(torch.bincount(phases[local_success], minlength=len(JugglePhase)))
            self._window_static_held_attempts.add_(static_held.sum())
            self._window_static_held_local_successes.add_(static_held_local_success.sum())
            self._window_static_held_successes.add_(static_held_success.sum())

        if ids.numel():
            state.row_ids[ids] = self._sample_rows(ids.numel())

        if self._completed_since_metrics >= env.num_envs:
            attempts = self._attempts
            phase_attempts = _aggregate_items_by_phase(attempts, self._item_phase_ids)
            phase_local_successes = _aggregate_items_by_phase(self._local_successes, self._item_phase_ids)
            canonical_attempts = attempts[self._canonical_item_mask].sum()
            canonical_cycle_successes = self._cycle_successes[self._canonical_item_mask].sum()
            aggregate_metrics = torch.stack(
                (
                    self._window_local_successes.float() / self._window_attempts.clamp_min(1),
                    self._window_cycle_successes.float() / self._window_attempts.clamp_min(1),
                    self._local_successes.sum().float() / attempts.sum().clamp_min(1),
                    self._cycle_successes.sum().float() / attempts.sum().clamp_min(1),
                    canonical_cycle_successes.float() / canonical_attempts.clamp_min(1),
                    (phase_attempts[self._sampled_phase_mask] > 0).float().mean(),
                    self._window_static_held_attempts.float(),
                    self._window_static_held_local_successes.float(),
                    self._window_static_held_local_successes.float() / self._window_static_held_attempts.clamp_min(1),
                    self._window_static_held_successes.float(),
                    self._window_static_held_successes.float() / self._window_static_held_attempts.clamp_min(1),
                    self._static_held_attempts.float(),
                    self._static_held_local_successes.float(),
                    self._static_held_local_successes.float() / self._static_held_attempts.clamp_min(1),
                    self._static_held_successes.float(),
                    self._static_held_successes.float() / self._static_held_attempts.clamp_min(1),
                )
            )
            recent_phase_rates = self._window_phase_local_successes.float() / self._window_phase_attempts.clamp_min(1)
            cumulative_phase_rates = phase_local_successes.float() / phase_attempts.clamp_min(1)
            sampling_probabilities = self.phase_probabilities()
            phase_metrics = torch.stack(
                tuple(
                    value[phase_id]
                    for phase_id in range(len(JugglePhase))
                    for value in (recent_phase_rates, cumulative_phase_rates, sampling_probabilities)
                )
            )
            metric_values = torch.cat((aggregate_metrics, phase_metrics)).detach().cpu().tolist()
            if self._continuous_sampler is not None:
                continuous_metrics = self._continuous_sampler.metrics()
                metric_values.extend(
                    continuous_metrics[name]
                    for name in (
                        "history_count",
                        "predicted_success_mean",
                        "predicted_success_minimum",
                        "predicted_success_maximum",
                        "effective_pool_fraction",
                        "top_1_percent_mass",
                        "realized_coverage_fraction",
                    )
                )
            self._cached_metrics = dict(zip(self._metric_names, metric_values, strict=True))
            self._completed_since_metrics = 0
            self._window_attempts.zero_()
            self._window_local_successes.zero_()
            self._window_cycle_successes.zero_()
            self._window_item_attempts.zero_()
            self._window_item_local_successes.zero_()
            self._window_phase_attempts.zero_()
            self._window_phase_local_successes.zero_()
            self._window_static_held_attempts.zero_()
            self._window_static_held_local_successes.zero_()
            self._window_static_held_successes.zero_()
        return self._cached_metrics

    def item_probabilities(self) -> torch.Tensor:
        """Return long-run semantic-item probabilities of both reset streams."""
        probabilities = self._canonical_item_probabilities * self._canonical_fraction
        if self._sampling_mode == "semantic":
            local_probabilities = self._sampler.sampling_probabilities(self._monitor.success_rates)
        else:
            row_probabilities = torch.zeros(self._reset_term.source.row_count, device=self.device)
            if self._sampling_mode == "uniform":
                row_probabilities[self._eligible_rows] = 1.0 / self._eligible_rows.numel()
            else:
                assert self._continuous_sampler is not None
                row_probabilities = self._continuous_sampler.sampling_probabilities()
            local_probabilities = torch.zeros(self._reset_term.catalog.item_count, device=self.device)
            local_probabilities.scatter_add_(0, self._reset_term.source.item_ids, row_probabilities)
        probabilities = probabilities + (1.0 - self._canonical_fraction) * local_probabilities
        return probabilities

    def phase_probabilities(self) -> torch.Tensor:
        """Return long-run physical-phase probabilities aggregated over semantic items."""
        return _aggregate_items_by_phase(self.item_probabilities(), self._item_phase_ids)

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return a flat, tensor-only snapshot of all sampling state."""
        state = {
            "reset_profile_id": torch.tensor(
                self._reset_term.source.profile.profile_id,
                dtype=torch.long,
                device=self.device,
            ),
            "catalog_item_count": torch.tensor(
                self._reset_term.catalog.item_count,
                dtype=torch.long,
                device=self.device,
            ),
            "curriculum_compatibility_signature": self._curriculum_compatibility_signature(),
            "canonical_credit": torch.tensor(self._canonical_credit, dtype=torch.float64, device=self.device),
            "attempts": self._attempts.clone(),
            "local_successes": self._local_successes.clone(),
            "cycle_successes": self._cycle_successes.clone(),
            "static_held_attempts": self._static_held_attempts.clone(),
            "static_held_local_successes": self._static_held_local_successes.clone(),
            "static_held_successes": self._static_held_successes.clone(),
            "completed_since_metrics": torch.tensor(
                self._completed_since_metrics, dtype=torch.long, device=self.device
            ),
            "window_attempts": self._window_attempts.clone(),
            "window_local_successes": self._window_local_successes.clone(),
            "window_cycle_successes": self._window_cycle_successes.clone(),
            "window_item_attempts": self._window_item_attempts.clone(),
            "window_item_local_successes": self._window_item_local_successes.clone(),
            "window_phase_attempts": self._window_phase_attempts.clone(),
            "window_phase_local_successes": self._window_phase_local_successes.clone(),
            "window_static_held_attempts": self._window_static_held_attempts.clone(),
            "window_static_held_local_successes": self._window_static_held_local_successes.clone(),
            "window_static_held_successes": self._window_static_held_successes.clone(),
            "cached_metrics": torch.tensor(
                tuple(self._cached_metrics.values()), dtype=torch.float64, device=self.device
            ),
            "row_generator_state": self._row_generator.get_state().clone(),
        }
        state.update({f"monitor__{name}": value for name, value in self._monitor.state_dict().items()})
        state.update({f"sampler__{name}": value for name, value in self._sampler.state_dict().items()})
        if self._continuous_sampler is not None:
            state.update(
                {f"continuous__{name}": value for name, value in self._continuous_sampler.state_dict().items()}
            )
        return state

    def set_state(self, state: dict[str, torch.Tensor]) -> None:
        """Atomically restore state returned by :meth:`get_state`."""
        previous_state = self.get_state()
        try:
            self._set_state(state)
        except Exception:
            # Nested monitors and samplers validate independently, so a later
            # curriculum-level error can follow an otherwise valid component
            # restore. Roll back the complete tensor snapshot before exposing
            # that error to the caller.
            self._set_state(previous_state)
            raise

    def reseed_checkpoint_generators(self, global_rank: int) -> None:
        """Fork restored sampler streams deterministically for a non-source DDP rank."""
        if isinstance(global_rank, bool) or not isinstance(global_rank, int) or global_rank < 0:
            raise ValueError("global_rank must be a non-negative integer.")
        generator_states = [("row", self._row_generator.get_state())]
        samplers = (("adaptive", self._sampler),)
        generator_states.extend(
            (stream_name, sampler.state_dict()["generator_state"]) for stream_name, sampler in samplers
        )
        rank_seeds: dict[str, int] = {}
        for stream_name, generator_state in generator_states:
            restored_state = bytes(generator_state.cpu().tolist())
            digest = hashlib.sha256(
                b"isaaclab-juggle-curriculum\0"
                + stream_name.encode("ascii")
                + b"\0"
                + global_rank.to_bytes(8, byteorder="little")
                + restored_state
            ).digest()
            rank_seeds[stream_name] = int.from_bytes(digest[:8], byteorder="little") % torch.iinfo(torch.int64).max

        self._row_generator.manual_seed(rank_seeds["row"])
        for stream_name, sampler in samplers:
            sampler_state = sampler.state_dict()
            generator = torch.Generator(device=self.device)
            generator.manual_seed(rank_seeds[stream_name])
            sampler_state["generator_state"] = generator.get_state()
            sampler.load_state_dict(sampler_state)
        if self._continuous_sampler is not None:
            continuous_state = self._continuous_sampler.state_dict()
            restored_state = bytes(continuous_state["frontier__generator_state"].cpu().tolist())
            digest = hashlib.sha256(
                b"isaaclab-juggle-curriculum\0continuous\0"
                + global_rank.to_bytes(8, byteorder="little")
                + restored_state
            ).digest()
            seed = int.from_bytes(digest[:8], byteorder="little") % torch.iinfo(torch.int64).max
            self._continuous_sampler.reseed_generator(seed)

    def _set_state(self, state: dict[str, torch.Tensor]) -> None:
        """Validate and apply a snapshot, with rollback owned by :meth:`set_state`."""
        monitor_keys = {f"monitor__{name}" for name in self._monitor.state_dict()}
        sampler_keys = {f"sampler__{name}" for name in self._sampler.state_dict()}
        continuous_keys = (
            {f"continuous__{name}" for name in self._continuous_sampler.state_dict()}
            if self._continuous_sampler is not None
            else set()
        )
        expected = {
            "reset_profile_id",
            "catalog_item_count",
            "curriculum_compatibility_signature",
            "canonical_credit",
            "attempts",
            "local_successes",
            "cycle_successes",
            "static_held_attempts",
            "static_held_local_successes",
            "static_held_successes",
            "completed_since_metrics",
            "window_attempts",
            "window_local_successes",
            "window_cycle_successes",
            "window_item_attempts",
            "window_item_local_successes",
            "window_phase_attempts",
            "window_phase_local_successes",
            "window_static_held_attempts",
            "window_static_held_local_successes",
            "window_static_held_successes",
            "cached_metrics",
            "row_generator_state",
            *monitor_keys,
            *sampler_keys,
            *continuous_keys,
        }
        if set(state) != expected:
            raise ValueError(f"Juggle curriculum state keys must be exactly {sorted(expected)}.")
        profile_id = state["reset_profile_id"]
        integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if (
            not isinstance(profile_id, torch.Tensor)
            or profile_id.numel() != 1
            or profile_id.dtype not in integer_dtypes
        ):
            raise ValueError("reset_profile_id must be a scalar integer tensor.")
        restored_profile_id = int(profile_id.item())
        expected_profile_id = self._reset_term.source.profile.profile_id
        if restored_profile_id != expected_profile_id:
            raise ValueError(
                f"Reset profile mismatch: state uses {restored_profile_id}, environment uses {expected_profile_id}."
            )
        catalog_item_count = state["catalog_item_count"]
        if (
            not isinstance(catalog_item_count, torch.Tensor)
            or catalog_item_count.numel() != 1
            or catalog_item_count.dtype not in integer_dtypes
        ):
            raise ValueError("catalog_item_count must be a scalar integer tensor.")
        if int(catalog_item_count.item()) != self._reset_term.catalog.item_count:
            raise ValueError("Curriculum state is incompatible with this reset catalog item count.")
        signature = state["curriculum_compatibility_signature"]
        expected_signature = self._curriculum_compatibility_signature()
        if (
            not isinstance(signature, torch.Tensor)
            or signature.shape != expected_signature.shape
            or not torch.equal(signature.to(device=self.device, dtype=torch.float64), expected_signature)
        ):
            raise ValueError("curriculum_compatibility_signature is incompatible with this curriculum.")
        self._monitor.load_state_dict({name.removeprefix("monitor__"): state[name] for name in monitor_keys})
        self._sampler.load_state_dict({name.removeprefix("sampler__"): state[name] for name in sampler_keys})
        if self._continuous_sampler is not None:
            self._continuous_sampler.load_state_dict(
                {name.removeprefix("continuous__"): state[name] for name in continuous_keys}
            )
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
            ("static_held_local_successes", self._static_held_local_successes),
            ("static_held_successes", self._static_held_successes),
            ("window_attempts", self._window_attempts),
            ("window_local_successes", self._window_local_successes),
            ("window_cycle_successes", self._window_cycle_successes),
            ("window_item_attempts", self._window_item_attempts),
            ("window_item_local_successes", self._window_item_local_successes),
            ("window_phase_attempts", self._window_phase_attempts),
            ("window_phase_local_successes", self._window_phase_local_successes),
            ("window_static_held_attempts", self._window_static_held_attempts),
            ("window_static_held_local_successes", self._window_static_held_local_successes),
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
        held_item_mask = self._item_phase_ids == held_phase
        held_attempts = self._attempts[held_item_mask].sum()
        if (
            self._static_held_local_successes > self._static_held_attempts
            or self._static_held_successes > self._static_held_local_successes
            or self._static_held_attempts > held_attempts
        ):
            raise ValueError("static held cumulative counts are inconsistent.")
        if self._window_local_successes > self._window_attempts or self._window_cycle_successes > self._window_attempts:
            raise ValueError("reporting-window successes cannot exceed attempts.")
        if (
            self._window_item_attempts.sum() != self._window_attempts
            or self._window_item_local_successes.sum() != self._window_local_successes
            or bool((self._window_item_local_successes > self._window_item_attempts).any())
            or self._window_phase_attempts.sum() != self._window_attempts
            or self._window_phase_local_successes.sum() != self._window_local_successes
            or bool((self._window_phase_local_successes > self._window_phase_attempts).any())
            or not torch.equal(
                _aggregate_items_by_phase(self._window_item_attempts, self._item_phase_ids),
                self._window_phase_attempts,
            )
            or not torch.equal(
                _aggregate_items_by_phase(self._window_item_local_successes, self._item_phase_ids),
                self._window_phase_local_successes,
            )
            or self._window_static_held_local_successes > self._window_static_held_attempts
            or self._window_static_held_successes > self._window_static_held_local_successes
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
        local_positions = assignment_order[canonical_count:]
        if canonical_count:
            rows[canonical_positions] = self._canonical_rows[
                torch.randint(
                    self._canonical_rows.numel(),
                    (canonical_count,),
                    device=self.device,
                    generator=self._row_generator,
                )
            ]
        if local_positions.numel():
            local_count = int(local_positions.numel())
            if self._sampling_mode == "semantic":
                item_ids = self._sampler.sample(local_count, self._monitor.success_rates)
                rows[local_positions] = self._rows_for_items(item_ids)
            elif self._sampling_mode == "uniform":
                offsets = torch.randint(
                    self._eligible_rows.numel(),
                    (local_count,),
                    device=self.device,
                    generator=self._row_generator,
                )
                rows[local_positions] = self._eligible_rows[offsets]
            else:
                assert self._continuous_sampler is not None
                rows[local_positions] = self._continuous_sampler.sample(local_count)
        return rows

    def _rows_for_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Choose one physical reset row for every semantic item."""
        if self._uniform_item_row_count:
            row_offsets = torch.randint(
                self._uniform_rows_per_item,
                (item_ids.numel(),),
                device=self.device,
                generator=self._row_generator,
            )
        else:
            row_offsets = torch.floor(
                torch.rand(item_ids.numel(), device=self.device, generator=self._row_generator)
                * self._item_row_counts[item_ids]
            ).long()
        return self._item_rows[item_ids, row_offsets]

    def _curriculum_compatibility_signature(self) -> torch.Tensor:
        """Return the reset and outcome identity that gives saved evidence meaning."""
        source = self._reset_term.source
        signature_fields = (
            *self._reset_source_signature_fields(),
            *self._outcome_semantics_signature_fields(),
        )
        if self._sampling_mode == "semantic":
            configuration_values = (
                9,
                source.profile.profile_id,
                self._reset_term.catalog.item_count,
                source.row_count,
                self._canonical_fraction,
                len(signature_fields),
            )
        else:
            mode_id = 1 if self._sampling_mode == "uniform" else 2
            configuration_values = (
                11,
                source.profile.profile_id,
                self._reset_term.catalog.item_count,
                source.row_count,
                self._canonical_fraction,
                mode_id,
                len(signature_fields),
            )
        configuration = torch.tensor(configuration_values, dtype=torch.float64, device=self.device)
        encoded_fields: list[torch.Tensor] = []
        for field_id, (_, values) in enumerate(signature_fields):
            encoded_fields.append(
                torch.tensor(
                    (field_id, values.ndim, *values.shape, values.numel()),
                    dtype=torch.float64,
                    device=self.device,
                )
            )
            encoded_fields.append(values.detach().to(device=self.device, dtype=torch.float64).reshape(-1))
        return torch.cat((configuration, *encoded_fields))

    def _outcome_semantics_signature_fields(self) -> tuple[tuple[str, torch.Tensor], ...]:
        """Return configured evaluator values that define local and cycle success."""
        progress_names = (
            "tool_offset",
            "release_separation_distance",
            "release_clear_steps",
            "apex_height_gain",
            "apex_maximum_horizontal_displacement",
            "track_supported_release_reference",
            "catch_approach_distance",
            "catch_distance",
            "contact_maximum_relative_speed",
            "stable_maximum_relative_speed",
            "stable_catch_steps",
            "rearm_after_stable_catch",
        )
        workspace_names = ("workspace_lower", "workspace_upper")
        progress_values = self._resolved_termination_parameters("progress_context", progress_names)
        workspace_values = self._resolved_termination_parameters("ball_out_of_workspace", workspace_names)
        configured_values = (
            *((f"progress_context__{name}", value) for name, value in progress_values),
            *((f"ball_out_of_workspace__{name}", value) for name, value in workspace_values),
            ("environment__step_dt", self._env.step_dt),
            ("environment__max_episode_length", self._env.max_episode_length),
        )
        return tuple((name, self._encode_signature_value(value)) for name, value in configured_values)

    def _resolved_termination_parameters(
        self, term_name: str, parameter_names: tuple[str, ...]
    ) -> tuple[tuple[str, object], ...]:
        """Resolve explicit term parameters and callable defaults without duplicating them."""
        term_cfg = self._env.termination_manager.get_term_cfg(term_name)
        term_callable = term_cfg.func.__call__ if isinstance(term_cfg.func, type) else term_cfg.func
        parameters = inspect.signature(term_callable).parameters
        values: list[tuple[str, object]] = []
        for name in parameter_names:
            if name in term_cfg.params:
                value = term_cfg.params[name]
            else:
                parameter = parameters.get(name)
                if parameter is None or parameter.default is inspect.Parameter.empty:
                    raise RuntimeError(f"Termination term '{term_name}' has no resolved value for '{name}'.")
                value = parameter.default
            values.append((name, value))
        return tuple(values)

    def _encode_signature_value(self, value: object) -> torch.Tensor:
        """Encode one scalar, optional scalar, or numeric sequence without ambiguous sentinels."""
        if value is None:
            encoded = (0.0,)
        elif isinstance(value, bool):
            encoded = (1.0, float(value))
        elif isinstance(value, int):
            encoded = (2.0, float(value))
        elif isinstance(value, float):
            encoded = (3.0, value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            encoded = (4.0, float(len(value)), *(float(item) for item in value))
        else:
            raise TypeError(f"Unsupported curriculum signature value: {value!r}.")
        return torch.tensor(encoded, dtype=torch.float64, device=self.device)

    def _reset_source_signature_fields(self) -> tuple[tuple[str, torch.Tensor], ...]:
        """Return versioned physical reset fields that give saved evidence its meaning."""
        source = self._reset_term.source
        fields = (
            ("arm_positions", source.arm_positions),
            ("arm_velocities", source.arm_velocities),
            ("hand_positions", source.hand_positions),
            ("hand_velocities", source.hand_velocities),
            ("ball_positions", source.ball_positions),
            ("ball_quaternions", source.ball_quaternions),
            ("ball_velocities", source.ball_velocities),
            ("release_positions", source.release_positions),
            ("release_origins_xy", source.release_origins_xy),
            ("release_velocities", source.release_velocities),
            ("launch_reference_heights", source.launch_reference_heights),
            ("ballistic_rows", source.ballistic_rows),
            ("flight_times", source.flight_times),
            ("difficulty_band_ids", source.difficulty_band_ids),
            ("phase_ids", source.phase_ids),
            ("item_ids", source.item_ids),
            ("item_phase_ids", source.item_phase_ids),
            ("static_held_rows", source.static_held_rows),
            ("preload_assist_rows", source.preload_assist_rows),
            ("canonical_start_rows", source.canonical_start_rows),
            ("canonical_row_ids", source.canonical_row_ids),
            ("canonical_item_mask", source.canonical_item_mask),
            ("adaptive_item_mask", source.adaptive_item_mask),
            ("local_goal_ids", source.local_goal_ids),
        )
        if self._sampling_mode == "continuous":
            fields = (
                *fields,
                ("reset_parameters", source.reset_parameters),
                ("model_features", source.model_features),
                (
                    "continuous_seed",
                    torch.tensor(source.continuous_seed, dtype=torch.long, device=self.device),
                ),
            )
        return fields


def _resolve_env_ids(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Return reset environment ids as a flat device tensor."""
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)[env_ids]
    return torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()


def _aggregate_items_by_phase(values: torch.Tensor, item_phase_ids: torch.Tensor) -> torch.Tensor:
    """Sum one scalar per competence item into the eight physical phases."""
    phase_values = torch.zeros(len(JugglePhase), dtype=values.dtype, device=values.device)
    phase_values.scatter_add_(0, item_phase_ids, values)
    return phase_values
