# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Continuous success-model sampling over normalized reset parameters."""

from __future__ import annotations

import torch

from .cfg import ContinuousAdaptiveResetSamplerCfg
from .sampler import AdaptiveResetSampler


class ContinuousAdaptiveResetSampler:
    """Prefer continuously parameterized reset candidates near a target success rate.

    The sampler retains a rolling history of normalized reset parameters and Boolean outcomes.
    Gaussian kernel regression estimates success for every proposal, then
    :class:`AdaptiveResetSampler` mixes the target-rate frontier with exact uniform coverage.
    Categorical reset discontinuities are represented by ``group_ids`` and never smoothed across.
    """

    def __init__(
        self,
        candidate_features: torch.Tensor,
        group_ids: torch.Tensor,
        cfg: ContinuousAdaptiveResetSamplerCfg,
        device: str | torch.device,
        *,
        eligible_mask: torch.Tensor | None = None,
        base_weights: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        """Create a continuous reset sampler.

        Args:
            candidate_features: Normalized continuous reset parameters, shape ``[N, D]``.
            group_ids: Categorical physical-region identifiers, shape ``[N]``.
            cfg: Continuous success-model and target-frontier configuration.
            device: Device used for histories, regression, and sampling.
            eligible_mask: Optional fixed Boolean candidate mask, shape ``[N]``.
            base_weights: Optional fixed positive frontier weights, shape ``[N]``.
            generator: Optional device-compatible generator for proposal draws.
        """
        cfg.validate()
        self.cfg = cfg
        self.device = torch.device(device)
        self._candidate_features = self._prepare_features(candidate_features)
        self.candidate_count, self.feature_dim = self._candidate_features.shape
        self._group_ids = self._prepare_groups(group_ids)
        self._frontier = AdaptiveResetSampler(
            self.candidate_count,
            cfg,
            self.device,
            eligible_mask=eligible_mask,
            base_weights=base_weights,
            generator=generator,
        )
        self._history_features = torch.zeros(
            (cfg.history_length, self.feature_dim), dtype=torch.float32, device=self.device
        )
        self._history_groups = torch.zeros(cfg.history_length, dtype=torch.long, device=self.device)
        self._history_outcomes = torch.zeros(cfg.history_length, dtype=torch.bool, device=self.device)
        self._history_count = 0
        self._history_cursor = 0

    @property
    def candidate_features(self) -> torch.Tensor:
        """Return a copy of normalized candidate reset parameters."""
        return self._candidate_features.clone()

    @property
    def group_ids(self) -> torch.Tensor:
        """Return a copy of candidate categorical groups."""
        return self._group_ids.clone()

    @property
    def eligible_mask(self) -> torch.Tensor:
        """Return a copy of the fixed proposal eligibility mask."""
        return self._frontier.eligible_mask

    @property
    def history_count(self) -> int:
        """Return the number of valid outcomes retained in the rolling history."""
        return self._history_count

    def record(
        self,
        candidate_ids: torch.Tensor,
        outcomes: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> None:
        """Record outcomes for sampled continuous reset candidates."""
        ids, values, valid_mask = self._validate_record_inputs(candidate_ids, outcomes, valid)
        valid_mask &= self._frontier.eligible_mask[ids]
        ids = ids[valid_mask]
        values = values[valid_mask]
        if ids.numel() == 0:
            return
        history_length = self.cfg.history_length
        if ids.numel() > history_length:
            ids = ids[-history_length:]
            values = values[-history_length:]
        count = int(ids.numel())
        positions = (torch.arange(count, device=self.device, dtype=torch.long) + self._history_cursor) % history_length
        self._history_features[positions] = self._candidate_features[ids]
        self._history_groups[positions] = self._group_ids[ids]
        self._history_outcomes[positions] = values
        self._history_cursor = (self._history_cursor + count) % history_length
        self._history_count = min(history_length, self._history_count + count)

    def predicted_success_rates(self) -> torch.Tensor:
        """Estimate candidate success probabilities with Gaussian kernel regression."""
        target = float(self.cfg.target_success_rate)
        rates = torch.full((self.candidate_count,), target, dtype=torch.float32, device=self.device)
        if self._history_count == 0:
            return rates
        history_features = self._history_features[: self._history_count]
        history_groups = self._history_groups[: self._history_count]
        history_outcomes = self._history_outcomes[: self._history_count].float()
        inverse_two_variance = 0.5 / float(self.cfg.kernel_bandwidth) ** 2
        prior_strength = float(self.cfg.prior_strength)
        chunk_size = self.cfg.prediction_chunk_size
        for start in range(0, self.candidate_count, chunk_size):
            stop = min(self.candidate_count, start + chunk_size)
            features = self._candidate_features[start:stop]
            squared_distance = torch.cdist(features, history_features).square()
            same_group = self._group_ids[start:stop, None] == history_groups[None, :]
            weights = torch.exp(-squared_distance * inverse_two_variance) * same_group
            evidence = weights.sum(dim=1)
            successes = weights @ history_outcomes
            rates[start:stop] = (prior_strength * target + successes) / (prior_strength + evidence)
        return rates.clamp_(0.0, 1.0)

    def sampling_probabilities(self) -> torch.Tensor:
        """Return long-run candidate probabilities of frontier and uniform coverage."""
        return self._frontier.sampling_probabilities(self.predicted_success_rates())

    def sample(self, count: int) -> torch.Tensor:
        """Draw candidate identifiers from the learned 50%-success frontier."""
        return self._frontier.sample(count, self.predicted_success_rates())

    def metrics(self) -> dict[str, float]:
        """Return continuous-model and proposal-distribution diagnostics."""
        rates = self.predicted_success_rates()
        metrics = self._frontier.metrics(rates)
        eligible_rates = rates[self._frontier.eligible_mask]
        metrics.update(
            {
                "history_count": float(self._history_count),
                "predicted_success_mean": float(eligible_rates.mean()),
                "predicted_success_minimum": float(eligible_rates.min()),
                "predicted_success_maximum": float(eligible_rates.max()),
            }
        )
        return metrics

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a tensor-only snapshot of model evidence and proposal state."""
        signature = torch.tensor(
            (
                1,
                self.candidate_count,
                self.feature_dim,
                self.cfg.history_length,
                self.cfg.prior_strength,
                self.cfg.kernel_bandwidth,
                self.cfg.prediction_chunk_size,
            ),
            dtype=torch.float64,
            device=self.device,
        )
        state = {
            "signature": signature,
            "candidate_features": self._candidate_features.clone(),
            "group_ids": self._group_ids.clone(),
            "history_features": self._history_features.clone(),
            "history_groups": self._history_groups.clone(),
            "history_outcomes": self._history_outcomes.clone(),
            "history_count": torch.tensor(self._history_count, dtype=torch.long, device=self.device),
            "history_cursor": torch.tensor(self._history_cursor, dtype=torch.long, device=self.device),
        }
        state.update({f"frontier__{name}": value for name, value in self._frontier.state_dict().items()})
        return state

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Validate and restore a snapshot returned by :meth:`state_dict`."""
        frontier_keys = {f"frontier__{name}" for name in self._frontier.state_dict()}
        expected_keys = {
            "signature",
            "candidate_features",
            "group_ids",
            "history_features",
            "history_groups",
            "history_outcomes",
            "history_count",
            "history_cursor",
            *frontier_keys,
        }
        if set(state) != expected_keys:
            raise ValueError(f"continuous sampler state keys must be exactly {sorted(expected_keys)}.")
        current = self.state_dict()
        for name, dtype in (
            ("signature", torch.float64),
            ("candidate_features", torch.float32),
            ("group_ids", torch.long),
        ):
            loaded = _state_tensor(state, name).to(device=self.device, dtype=dtype)
            if loaded.shape != current[name].shape or not torch.equal(loaded, current[name]):
                raise ValueError(f"continuous sampler state {name} is incompatible with this sampler.")
        history_features = _state_tensor(state, "history_features").to(device=self.device, dtype=torch.float32)
        history_groups = _state_tensor(state, "history_groups").to(device=self.device, dtype=torch.long)
        history_outcomes = _state_tensor(state, "history_outcomes").to(device=self.device, dtype=torch.bool)
        if history_features.shape != self._history_features.shape:
            raise ValueError("history_features has an incompatible shape.")
        if history_groups.shape != self._history_groups.shape:
            raise ValueError("history_groups has an incompatible shape.")
        if history_outcomes.shape != self._history_outcomes.shape:
            raise ValueError("history_outcomes has an incompatible shape.")
        if not bool(torch.isfinite(history_features).all()):
            raise ValueError("history_features must be finite.")
        history_count = _scalar_int(state, "history_count")
        history_cursor = _scalar_int(state, "history_cursor")
        if not 0 <= history_count <= self.cfg.history_length:
            raise ValueError("history_count is outside the rolling history.")
        if not 0 <= history_cursor < self.cfg.history_length:
            raise ValueError("history_cursor is outside the rolling history.")
        if history_count < self.cfg.history_length and history_cursor != history_count:
            raise ValueError("history_cursor is inconsistent with a partially filled history.")
        frontier_state = {name.removeprefix("frontier__"): state[name] for name in frontier_keys}
        self._frontier.load_state_dict(frontier_state)
        self._history_features.copy_(history_features)
        self._history_groups.copy_(history_groups)
        self._history_outcomes.copy_(history_outcomes)
        self._history_count = history_count
        self._history_cursor = history_cursor

    def reseed_generator(self, seed: int) -> None:
        """Replace the proposal generator seed without changing learned evidence."""
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        state = self._frontier.state_dict()
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        state["generator_state"] = generator.get_state()
        self._frontier.load_state_dict(state)

    def _prepare_features(self, features: torch.Tensor) -> torch.Tensor:
        """Validate and copy normalized candidate parameters."""
        if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[0] < 1:
            raise ValueError("candidate_features must be a rank-two tensor with at least one row.")
        if not features.is_floating_point():
            raise TypeError("candidate_features must have a floating-point dtype.")
        values = features.to(device=self.device, dtype=torch.float32).clone()
        if not bool(torch.isfinite(values).all()) or bool(((values < 0.0) | (values > 1.0)).any()):
            raise ValueError("candidate_features must be finite and normalized to [0, 1].")
        return values

    def _prepare_groups(self, group_ids: torch.Tensor) -> torch.Tensor:
        """Validate and copy categorical candidate identifiers."""
        if not isinstance(group_ids, torch.Tensor) or group_ids.shape != (self.candidate_count,):
            raise ValueError(f"group_ids must have shape ({self.candidate_count},).")
        if group_ids.dtype == torch.bool or group_ids.is_floating_point() or group_ids.is_complex():
            raise TypeError("group_ids must have an integer dtype.")
        groups = group_ids.to(device=self.device, dtype=torch.long).clone()
        if bool((groups < 0).any()):
            raise ValueError("group_ids must be non-negative.")
        return groups

    def _validate_record_inputs(
        self,
        candidate_ids: torch.Tensor,
        outcomes: torch.Tensor,
        valid: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate outcome attribution inputs without changing sampler state."""
        if not isinstance(candidate_ids, torch.Tensor) or candidate_ids.ndim != 1:
            raise ValueError("candidate_ids must be a rank-one tensor.")
        if candidate_ids.dtype == torch.bool or candidate_ids.is_floating_point() or candidate_ids.is_complex():
            raise TypeError("candidate_ids must have an integer dtype.")
        if candidate_ids.device != self.device:
            raise ValueError(f"candidate_ids must be on {self.device}.")
        ids = candidate_ids.to(dtype=torch.long)
        if not isinstance(outcomes, torch.Tensor) or outcomes.shape != ids.shape or outcomes.dtype != torch.bool:
            raise ValueError("outcomes must be Boolean with the same shape as candidate_ids.")
        if outcomes.device != self.device:
            raise ValueError(f"outcomes must be on {self.device}.")
        torch._assert_async(
            ((ids >= 0) & (ids < self.candidate_count)).all(),
            "candidate_ids are outside the proposal bank.",
        )
        if valid is None:
            valid_mask = torch.ones_like(outcomes)
        else:
            if not isinstance(valid, torch.Tensor) or valid.shape != ids.shape or valid.dtype != torch.bool:
                raise ValueError("valid must be Boolean with the same shape as candidate_ids.")
            if valid.device != self.device:
                raise ValueError(f"valid must be on {self.device}.")
            valid_mask = valid
        return ids, outcomes, valid_mask


def _state_tensor(state: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    """Return a required tensor state value."""
    value = state[name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"state[{name!r}] must be a torch.Tensor.")
    return value


def _scalar_int(state: dict[str, torch.Tensor], name: str) -> int:
    """Read a scalar integer tensor."""
    value = _state_tensor(state, name)
    if value.numel() != 1 or value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise ValueError(f"state[{name!r}] must be a scalar integer tensor.")
    return int(value.item())
