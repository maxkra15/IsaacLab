# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rolling outcome monitoring for adaptive reset sampling."""

from __future__ import annotations

import math

import torch

from .cfg import RollingOutcomeMonitorCfg


class RollingOutcomeMonitor:
    """Track recent Boolean outcomes independently for each competence item.

    Repeated item ids in one update are appended in input order. The posterior success rate uses a
    prior centered on ``prior_success_rate`` so unseen items begin at the sampling target instead of
    being misclassified as failures.
    """

    def __init__(
        self,
        item_count: int,
        cfg: RollingOutcomeMonitorCfg,
        device: str | torch.device,
        prior_success_rate: float,
    ) -> None:
        """Allocate the per-item outcome rings.

        Args:
            item_count: Number of monitored competence items.
            cfg: Rolling-history configuration.
            device: Torch device holding monitor state.
            prior_success_rate: Prior mean success probability in ``[0, 1]``.
        """
        if isinstance(item_count, bool) or not isinstance(item_count, int) or item_count < 1:
            raise ValueError("item_count must be a positive integer.")
        cfg.validate()
        prior_success_rate = float(prior_success_rate)
        if not math.isfinite(prior_success_rate) or not 0.0 <= prior_success_rate <= 1.0:
            raise ValueError("prior_success_rate must lie in the closed interval [0, 1].")

        self.cfg = cfg
        self.item_count = item_count
        self.device = torch.device(device)
        self.prior_success_rate = prior_success_rate
        self._history = torch.zeros((item_count, cfg.history_length), dtype=torch.bool, device=self.device)
        self._pointers = torch.zeros(item_count, dtype=torch.long, device=self.device)
        self._sizes = torch.zeros(item_count, dtype=torch.long, device=self.device)
        self._success_counts = torch.zeros(item_count, dtype=torch.long, device=self.device)

    @property
    def success_rates(self) -> torch.Tensor:
        """Return posterior success rates, shape ``[item_count]``."""
        sizes = self._sizes.to(dtype=torch.float32)
        empirical_numerator = self._success_counts.to(dtype=torch.float32)
        prior_strength = float(self.cfg.prior_strength)
        if prior_strength == 0.0:
            empirical = empirical_numerator / sizes.clamp_min(1.0)
            return torch.where(sizes > 0.0, empirical, empirical.new_full((), self.prior_success_rate))
        return (empirical_numerator + prior_strength * self.prior_success_rate) / (sizes + prior_strength)

    @property
    def history_sizes(self) -> torch.Tensor:
        """Return the number of recorded outcomes retained for every item."""
        return self._sizes.clone()

    def record(
        self,
        item_ids: torch.Tensor,
        outcomes: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> None:
        """Append outcomes to their per-item rolling histories.

        Args:
            item_ids: Competence item for each outcome, shape ``[N]``.
            outcomes: Boolean episode outcomes, shape ``[N]``.
            valid: Optional Boolean mask selecting attributable outcomes, shape ``[N]``.
        """
        self._validate_update(item_ids, outcomes, valid)
        if valid is not None:
            item_ids = item_ids[valid]
            outcomes = outcomes[valid]
        if item_ids.numel() == 0:
            return
        _append_boolean_rings(
            self._history,
            item_ids.to(dtype=torch.long),
            outcomes,
            self._pointers,
            self._sizes,
            self._success_counts,
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a detached copy of the complete rolling-monitor state."""
        signature = torch.tensor(
            [self.item_count, self.cfg.history_length, self.cfg.prior_strength, self.prior_success_rate],
            dtype=torch.float64,
            device=self.device,
        )
        return {
            "signature": signature,
            "history": self._history.clone(),
            "pointers": self._pointers.clone(),
            "sizes": self._sizes.clone(),
            "success_counts": self._success_counts.clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore state produced by :meth:`state_dict` for an identical monitor."""
        expected_keys = {"signature", "history", "pointers", "sizes", "success_counts"}
        if set(state) != expected_keys:
            raise ValueError(f"monitor state keys must be exactly {sorted(expected_keys)}.")
        expected_signature = self.state_dict()["signature"]
        signature = _state_tensor(state, "signature").to(device=self.device, dtype=torch.float64)
        if signature.shape != expected_signature.shape or not torch.equal(signature, expected_signature):
            raise ValueError("monitor state is incompatible with this monitor configuration.")

        history = _state_tensor(state, "history").to(device=self.device)
        pointers = _state_tensor(state, "pointers").to(device=self.device, dtype=torch.long)
        sizes = _state_tensor(state, "sizes").to(device=self.device, dtype=torch.long)
        success_counts = _state_tensor(state, "success_counts").to(device=self.device, dtype=torch.long)
        if history.shape != self._history.shape or history.dtype != torch.bool:
            raise ValueError(f"history must be a Boolean tensor with shape {tuple(self._history.shape)}.")
        for name, values in (("pointers", pointers), ("sizes", sizes), ("success_counts", success_counts)):
            if values.shape != (self.item_count,):
                raise ValueError(f"{name} must have shape ({self.item_count},).")
        if bool(((pointers < 0) | (pointers >= self.cfg.history_length)).any()):
            raise ValueError("monitor pointers are outside the history ring.")
        if bool(((sizes < 0) | (sizes > self.cfg.history_length)).any()):
            raise ValueError("monitor sizes are outside the history capacity.")
        partially_filled = sizes < self.cfg.history_length
        if bool((pointers[partially_filled] != sizes[partially_filled]).any()):
            raise ValueError("monitor pointers must follow retained outcomes in partially filled rings.")
        history_successes = history.sum(dim=1, dtype=torch.long)
        if bool(((success_counts < 0) | (success_counts > sizes)).any()) or not torch.equal(
            success_counts, history_successes
        ):
            raise ValueError("monitor success counts are inconsistent with the stored history.")

        self._history.copy_(history)
        self._pointers.copy_(pointers)
        self._sizes.copy_(sizes)
        self._success_counts.copy_(success_counts)

    def _validate_update(
        self,
        item_ids: torch.Tensor,
        outcomes: torch.Tensor,
        valid: torch.Tensor | None,
    ) -> None:
        """Validate a public update before mutating state."""
        if item_ids.ndim != 1 or outcomes.ndim != 1 or item_ids.shape != outcomes.shape:
            raise ValueError("item_ids and outcomes must be one-dimensional tensors with equal shape.")
        if item_ids.device != self.device or outcomes.device != self.device:
            raise ValueError(f"item_ids and outcomes must be on {self.device}.")
        if item_ids.dtype == torch.bool or item_ids.is_floating_point() or item_ids.is_complex():
            raise TypeError("item_ids must have an integer dtype other than bool.")
        if outcomes.dtype != torch.bool:
            raise TypeError("outcomes must have Boolean dtype.")
        if item_ids.numel() > 0:
            # Invalid ids are an internal programming error.  Assert them on-device so a
            # vectorized reset does not introduce a CUDA-to-host synchronization.
            torch._assert_async(
                ((item_ids >= 0) & (item_ids < self.item_count)).all(),
                "item_ids contains an id outside the monitor.",
            )
        if valid is not None:
            if valid.shape != outcomes.shape or valid.ndim != 1:
                raise ValueError("valid must have the same one-dimensional shape as outcomes.")
            if valid.device != self.device or valid.dtype != torch.bool:
                raise TypeError(f"valid must be a Boolean tensor on {self.device}.")


def _append_boolean_rings(
    history: torch.Tensor,
    item_ids: torch.Tensor,
    outcomes: torch.Tensor,
    pointers: torch.Tensor,
    sizes: torch.Tensor,
    success_counts: torch.Tensor,
) -> None:
    """Append repeated ids to Boolean rings without last-write-wins indexing."""
    capacity = history.shape[1]
    unique_ids, inverse, counts = torch.unique(item_ids, return_inverse=True, return_counts=True)
    if unique_ids.numel() == item_ids.numel():
        columns = pointers[item_ids]
        overwritten = torch.where(
            sizes[item_ids] == capacity,
            history[item_ids, columns].to(dtype=torch.long),
            torch.zeros_like(item_ids, dtype=torch.long),
        )
        success_counts[item_ids] += outcomes.to(dtype=torch.long) - overwritten
        history[item_ids, columns] = outcomes
        pointers[item_ids] = (columns + 1) % capacity
        sizes[item_ids] = (sizes[item_ids] + 1).clamp(max=capacity)
        return

    order = torch.argsort(inverse, stable=True)
    sorted_ids = item_ids[order]
    sorted_outcomes = outcomes[order]
    group_starts = counts.cumsum(0) - counts
    local_rank = torch.arange(item_ids.numel(), device=history.device) - torch.repeat_interleave(group_starts, counts)
    inverse_sorted = inverse[order]
    repeated_counts = counts[inverse_sorted]

    added_successes = torch.zeros_like(unique_ids, dtype=torch.long)
    added_successes.scatter_add_(0, inverse, outcomes.to(dtype=torch.long))

    keep_start = (counts - capacity).clamp(min=0)
    keep = local_rank >= torch.repeat_interleave(keep_start, counts)
    kept_successes = torch.zeros_like(added_successes)
    kept_successes.scatter_add_(0, inverse_sorted[keep], sorted_outcomes[keep].to(dtype=torch.long))

    overwrite_start = capacity - sizes[sorted_ids]
    overwritten_mask = (repeated_counts < capacity) & (local_rank >= overwrite_start)
    overwritten_successes = torch.zeros_like(added_successes)
    overwritten_ids = sorted_ids[overwritten_mask]
    overwritten_columns = (pointers[overwritten_ids] + local_rank[overwritten_mask]) % capacity
    overwritten_successes.scatter_add_(
        0,
        inverse_sorted[overwritten_mask],
        history[overwritten_ids, overwritten_columns].to(dtype=torch.long),
    )

    kept_ids = sorted_ids[keep]
    kept_columns = (pointers[kept_ids] + local_rank[keep]) % capacity
    history[kept_ids, kept_columns] = sorted_outcomes[keep]
    success_counts[unique_ids] = torch.where(
        counts >= capacity,
        kept_successes,
        success_counts[unique_ids] - overwritten_successes + added_successes,
    )
    sizes[unique_ids] = (sizes[unique_ids] + counts).clamp(max=capacity)
    pointers[unique_ids] = (pointers[unique_ids] + counts) % capacity


def _state_tensor(state: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    """Return a required tensor state value."""
    value = state[name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"state[{name!r}] must be a torch.Tensor.")
    return value
