# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adaptive target-rate sampling with explicit exact coverage."""

from __future__ import annotations

import math

import torch

from .cfg import AdaptiveResetSamplerCfg


class AdaptiveResetSampler:
    """Sample competence items from adaptive and exact-coverage streams.

    The adaptive stream favors items whose measured success rate is near the configured target.
    The coverage stream walks shuffled complete cycles, ensuring every eligible item appears once
    per cycle independently of its adaptive weight. Eligibility and base weights are fixed at
    construction so saved sampler state has one unambiguous meaning.
    """

    def __init__(
        self,
        item_count: int,
        cfg: AdaptiveResetSamplerCfg,
        device: str | torch.device,
        *,
        eligible_mask: torch.Tensor | None = None,
        base_weights: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        """Create an adaptive sampler.

        Args:
            item_count: Number of competence items.
            cfg: Adaptive target-rate and coverage configuration.
            device: Torch device used for sampling.
            eligible_mask: Optional fixed Boolean eligibility mask, shape ``[item_count]``.
            base_weights: Optional fixed positive adaptive weights, shape ``[item_count]``.
            generator: Optional Torch random-number generator used for every draw and shuffle.
        """
        if isinstance(item_count, bool) or not isinstance(item_count, int) or item_count < 1:
            raise ValueError("item_count must be a positive integer.")
        cfg.validate()
        self.cfg = cfg
        self.item_count = item_count
        self.device = torch.device(device)
        self._eligible_mask = self._prepare_eligible_mask(eligible_mask)
        self._eligible_ids = torch.where(self._eligible_mask)[0]
        self._base_weights = self._prepare_base_weights(base_weights)
        self._generator = self._prepare_generator(generator)

        self._coverage_order = self._new_coverage_order()
        self._coverage_cursor = 0
        self._coverage_credit = 0.0
        self._coverage_assignments = 0
        self._adaptive_assignments = 0

    @property
    def eligible_mask(self) -> torch.Tensor:
        """Return a copy of the fixed item eligibility mask."""
        return self._eligible_mask.clone()

    @property
    def base_weights(self) -> torch.Tensor:
        """Return a copy of the fixed adaptive base weights."""
        return self._base_weights.clone()

    def adaptive_probabilities(self, success_rates: torch.Tensor) -> torch.Tensor:
        """Return normalized probabilities for the adaptive stream."""
        rates = self._validate_success_rates(success_rates)
        target = float(self.cfg.target_success_rate)
        kappa = float(self.cfg.kappa)
        epsilon = float(self.cfg.epsilon)
        log_kernel = kappa * target * torch.log(rates + epsilon)
        log_kernel += kappa * (1.0 - target) * torch.log(1.0 - rates + epsilon)
        log_kernel /= float(self.cfg.temperature)

        log_weights = log_kernel + self._base_weights.log()
        log_weights = torch.where(self._eligible_mask, log_weights, log_weights.new_full((), -torch.inf))
        return torch.softmax(log_weights, dim=0)

    def sampling_probabilities(self, success_rates: torch.Tensor) -> torch.Tensor:
        """Return long-run marginal probabilities of the adaptive/coverage mixture."""
        adaptive = self.adaptive_probabilities(success_rates)
        coverage = torch.zeros_like(adaptive)
        coverage[self._eligible_mask] = 1.0 / self._eligible_ids.numel()
        fraction = float(self.cfg.coverage_fraction)
        return adaptive.mul(1.0 - fraction).add_(coverage, alpha=fraction)

    def sample(self, count: int, success_rates: torch.Tensor) -> torch.Tensor:
        """Draw competence-item ids, shape ``[count]``.

        The number assigned to coverage is accumulated fractionally across calls, so the exact
        long-run assignment count is ``floor(total_requested * coverage_fraction)``.
        """
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer.")
        probabilities = self.adaptive_probabilities(success_rates)
        rows = torch.empty(count, dtype=torch.long, device=self.device)
        if count == 0:
            return rows

        coverage_count = self._coverage_assignment_count(count)
        assignment_order = torch.randperm(count, device=self.device, generator=self._generator)
        coverage_positions = assignment_order[:coverage_count]
        adaptive_positions = assignment_order[coverage_count:]
        if coverage_count:
            rows[coverage_positions] = self._take_coverage_items(coverage_count)
        if adaptive_positions.numel():
            rows[adaptive_positions] = torch.multinomial(
                probabilities,
                adaptive_positions.numel(),
                replacement=True,
                generator=self._generator,
            )
        self._coverage_assignments += coverage_count
        self._adaptive_assignments += int(adaptive_positions.numel())
        return rows

    def metrics(self, success_rates: torch.Tensor) -> dict[str, float]:
        """Return distribution concentration and assignment metrics."""
        probabilities = self.sampling_probabilities(success_rates)
        eligible_count = self._eligible_ids.numel()
        top_count = max(1, math.ceil(0.01 * eligible_count))
        eligible_probabilities = probabilities[self._eligible_mask]
        effective_pool_fraction = eligible_probabilities.square().sum().reciprocal() / eligible_count
        top_mass = torch.topk(eligible_probabilities, top_count, sorted=False).values.sum()
        total_assignments = self._coverage_assignments + self._adaptive_assignments
        realized_coverage = self._coverage_assignments / total_assignments if total_assignments else 0.0
        return {
            "effective_pool_fraction": float(effective_pool_fraction),
            "top_1_percent_mass": float(top_mass),
            "configured_coverage_fraction": float(self.cfg.coverage_fraction),
            "realized_coverage_fraction": realized_coverage,
            "coverage_assignments": float(self._coverage_assignments),
            "adaptive_assignments": float(self._adaptive_assignments),
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a detached copy of the complete sampler state, including RNG state."""
        signature = torch.tensor(
            [
                self.item_count,
                self.cfg.target_success_rate,
                self.cfg.kappa,
                self.cfg.temperature,
                self.cfg.coverage_fraction,
                self.cfg.epsilon,
            ],
            dtype=torch.float64,
            device=self.device,
        )
        return {
            "signature": signature,
            "eligible_mask": self._eligible_mask.clone(),
            "base_weights": self._base_weights.clone(),
            "coverage_order": self._coverage_order.clone(),
            "coverage_cursor": torch.tensor(self._coverage_cursor, dtype=torch.long, device=self.device),
            "coverage_credit": torch.tensor(self._coverage_credit, dtype=torch.float64, device=self.device),
            "coverage_assignments": torch.tensor(self._coverage_assignments, dtype=torch.long, device=self.device),
            "adaptive_assignments": torch.tensor(self._adaptive_assignments, dtype=torch.long, device=self.device),
            "generator_state": self._generator.get_state().clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore state produced by :meth:`state_dict` for an identical sampler."""
        expected_keys = {
            "signature",
            "eligible_mask",
            "base_weights",
            "coverage_order",
            "coverage_cursor",
            "coverage_credit",
            "coverage_assignments",
            "adaptive_assignments",
            "generator_state",
        }
        if set(state) != expected_keys:
            raise ValueError(f"sampler state keys must be exactly {sorted(expected_keys)}.")

        current = self.state_dict()
        for name, dtype in (
            ("signature", torch.float64),
            ("eligible_mask", torch.bool),
            ("base_weights", torch.float32),
        ):
            loaded = _state_tensor(state, name).to(device=self.device, dtype=dtype)
            expected = current[name]
            if loaded.shape != expected.shape or not torch.equal(loaded, expected):
                raise ValueError(f"sampler state {name} is incompatible with this sampler.")

        order = _state_tensor(state, "coverage_order").to(device=self.device, dtype=torch.long)
        if order.shape != self._eligible_ids.shape or not torch.equal(order.sort().values, self._eligible_ids):
            raise ValueError("coverage_order must be a permutation of the eligible item ids.")
        cursor = _scalar_int(state, "coverage_cursor")
        credit = _scalar_float(state, "coverage_credit")
        coverage_assignments = _scalar_int(state, "coverage_assignments")
        adaptive_assignments = _scalar_int(state, "adaptive_assignments")
        if not 0 <= cursor < self._eligible_ids.numel():
            raise ValueError("coverage_cursor is outside the coverage cycle.")
        if not math.isfinite(credit) or not 0.0 <= credit < 1.0 + 1.0e-9:
            raise ValueError("coverage_credit must lie in [0, 1).")
        if coverage_assignments < 0 or adaptive_assignments < 0:
            raise ValueError("assignment counters must be non-negative.")

        generator_state = _state_tensor(state, "generator_state")
        self._generator.set_state(generator_state.cpu())
        self._coverage_order.copy_(order)
        self._coverage_cursor = cursor
        self._coverage_credit = min(credit, math.nextafter(1.0, 0.0))
        self._coverage_assignments = coverage_assignments
        self._adaptive_assignments = adaptive_assignments

    def _prepare_eligible_mask(self, eligible_mask: torch.Tensor | None) -> torch.Tensor:
        """Validate and copy the fixed eligibility mask."""
        if eligible_mask is None:
            return torch.ones(self.item_count, dtype=torch.bool, device=self.device)
        if not isinstance(eligible_mask, torch.Tensor):
            raise TypeError("eligible_mask must be a torch.Tensor or None.")
        if eligible_mask.shape != (self.item_count,) or eligible_mask.dtype != torch.bool:
            raise ValueError(f"eligible_mask must be Boolean with shape ({self.item_count},).")
        mask = eligible_mask.to(device=self.device).clone()
        if not bool(mask.any()):
            raise ValueError("eligible_mask must select at least one item.")
        return mask

    def _prepare_base_weights(self, base_weights: torch.Tensor | None) -> torch.Tensor:
        """Validate and copy fixed adaptive base weights."""
        if base_weights is None:
            return torch.ones(self.item_count, dtype=torch.float32, device=self.device)
        if not isinstance(base_weights, torch.Tensor):
            raise TypeError("base_weights must be a torch.Tensor or None.")
        if base_weights.shape != (self.item_count,):
            raise ValueError(f"base_weights must have shape ({self.item_count},).")
        weights = base_weights.to(device=self.device, dtype=torch.float32).clone()
        if not bool(torch.isfinite(weights).all()) or bool((weights[self._eligible_mask] <= 0.0).any()):
            raise ValueError("base_weights must be finite and positive for every eligible item.")
        if bool((weights[~self._eligible_mask] < 0.0).any()):
            raise ValueError("base_weights must be non-negative for ineligible items.")
        return weights

    def _prepare_generator(self, generator: torch.Generator | None) -> torch.Generator:
        """Return an owned or caller-provided device-compatible generator."""
        if generator is not None:
            if torch.device(generator.device) != self.device:
                raise ValueError(f"generator must be on {self.device}.")
            return generator
        generator = torch.Generator(device=self.device)
        seed = int(torch.randint(0, torch.iinfo(torch.int64).max, (), device="cpu").item())
        generator.manual_seed(seed)
        return generator

    def _validate_success_rates(self, success_rates: torch.Tensor) -> torch.Tensor:
        """Validate per-item success rates before sampling."""
        if not isinstance(success_rates, torch.Tensor):
            raise TypeError("success_rates must be a torch.Tensor.")
        if success_rates.shape != (self.item_count,):
            raise ValueError(f"success_rates must have shape ({self.item_count},).")
        if success_rates.device != self.device:
            raise ValueError(f"success_rates must be on {self.device}.")
        if not success_rates.is_floating_point():
            raise TypeError("success_rates must have a floating-point dtype.")
        rates = success_rates.to(dtype=torch.float32)
        # Sampling is a reset-hot-path operation.  Preserve the public invariant without
        # serializing CUDA against the host for every batch of environment resets.
        torch._assert_async(
            (torch.isfinite(rates) & (rates >= 0.0) & (rates <= 1.0)).all(),
            "success_rates must be finite and lie in [0, 1].",
        )
        return rates

    def _new_coverage_order(self) -> torch.Tensor:
        """Create one shuffled permutation of all eligible items."""
        permutation = torch.randperm(self._eligible_ids.numel(), device=self.device, generator=self._generator)
        return self._eligible_ids[permutation]

    def _coverage_assignment_count(self, count: int) -> int:
        """Accumulate fractional coverage assignments exactly across calls."""
        credit = self._coverage_credit + float(self.cfg.coverage_fraction) * count
        coverage_count = min(count, math.floor(credit + 1.0e-12))
        self._coverage_credit = max(0.0, credit - coverage_count)
        return coverage_count

    def _take_coverage_items(self, count: int) -> torch.Tensor:
        """Take items from shuffled complete cycles without replacement within a cycle."""
        remaining = count
        chunks: list[torch.Tensor] = []
        while remaining:
            available = self._eligible_ids.numel() - self._coverage_cursor
            taken = min(remaining, available)
            chunks.append(self._coverage_order[self._coverage_cursor : self._coverage_cursor + taken])
            self._coverage_cursor += taken
            remaining -= taken
            if self._coverage_cursor == self._eligible_ids.numel():
                self._coverage_order = self._new_coverage_order()
                self._coverage_cursor = 0
        return torch.cat(chunks)


def _state_tensor(state: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    """Return a required tensor state value."""
    value = state[name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"state[{name!r}] must be a torch.Tensor.")
    return value


def _scalar_int(state: dict[str, torch.Tensor], name: str) -> int:
    """Read a scalar integer state value."""
    value = _state_tensor(state, name)
    if value.numel() != 1 or value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise ValueError(f"state[{name!r}] must be a scalar integer tensor.")
    return int(value.item())


def _scalar_float(state: dict[str, torch.Tensor], name: str) -> float:
    """Read a scalar floating-point state value."""
    value = _state_tensor(state, name)
    if value.numel() != 1 or not value.is_floating_point():
        raise ValueError(f"state[{name!r}] must be a scalar floating-point tensor.")
    return float(value.item())
