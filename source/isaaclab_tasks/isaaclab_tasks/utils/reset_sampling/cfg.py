# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for adaptive reset-state sampling."""

from __future__ import annotations

import math

from isaaclab.utils.configclass import configclass


@configclass
class RollingOutcomeMonitorCfg:
    """Configuration for a per-item rolling outcome monitor."""

    history_length: int = 50
    """Maximum number of recent Boolean outcomes retained for each item."""

    prior_strength: float = 2.0
    """Pseudo-observation count assigned to the caller-provided prior success rate.

    A positive value smooths sparsely measured items. Setting this to zero uses empirical rates as
    soon as an item is measured, while still using the prior for unseen items.
    """

    def __post_init__(self) -> None:
        """Validate configuration values."""
        self.validate()

    def validate(self) -> None:
        """Validate values after construction or runtime overrides."""
        if isinstance(self.history_length, bool) or not isinstance(self.history_length, int) or self.history_length < 1:
            raise ValueError("history_length must be a positive integer.")
        prior_strength = float(self.prior_strength)
        if isinstance(self.prior_strength, bool) or not math.isfinite(prior_strength) or prior_strength < 0.0:
            raise ValueError("prior_strength must be finite and non-negative.")


@configclass
class AdaptiveResetSamplerCfg:
    """Configuration for target-rate sampling with exact cyclic coverage."""

    target_success_rate: float = 0.5
    """Success rate at which the adaptive sampling kernel peaks."""

    kappa: float = 1.0
    """Non-negative concentration of the target-rate kernel."""

    temperature: float = 1.0
    """Positive temperature applied to adaptive log weights."""

    coverage_fraction: float = 0.15
    """Fraction of assignments reserved for shuffled complete cycles over eligible items."""

    epsilon: float = 1.0e-4
    """Positive numerical offset used when evaluating the target-rate kernel."""

    def __post_init__(self) -> None:
        """Validate configuration values."""
        self.validate()

    def validate(self) -> None:
        """Validate values after construction or runtime overrides."""
        self._validate_unit_interval("target_success_rate", self.target_success_rate)
        self._validate_unit_interval("coverage_fraction", self.coverage_fraction)

        kappa = float(self.kappa)
        if isinstance(self.kappa, bool) or not math.isfinite(kappa) or kappa < 0.0:
            raise ValueError("kappa must be finite and non-negative.")
        for name in ("temperature", "epsilon"):
            value = getattr(self, name)
            numeric_value = float(value)
            if isinstance(value, bool) or not math.isfinite(numeric_value) or numeric_value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")

    @staticmethod
    def _validate_unit_interval(name: str, value: float) -> None:
        numeric_value = float(value)
        if isinstance(value, bool) or not math.isfinite(numeric_value) or not 0.0 <= numeric_value <= 1.0:
            raise ValueError(f"{name} must lie in the closed interval [0, 1].")


@configclass
class ContinuousAdaptiveResetSamplerCfg(AdaptiveResetSamplerCfg):
    """Configuration for continuous kernel-regression reset sampling.

    Candidate reset parameters are normalized before they reach the sampler. Recent outcomes
    provide a non-parametric estimate of success probability, while the inherited target-rate
    kernel and coverage stream decide which candidates to draw.
    """

    history_length: int = 4096
    """Maximum number of recent reset parameter/outcome pairs retained."""

    prior_strength: float = 2.0
    """Kernel evidence assigned to the target-rate prior before observations arrive."""

    kernel_bandwidth: float = 0.20
    """Gaussian-kernel bandwidth in normalized reset-parameter coordinates."""

    prediction_chunk_size: int = 1024
    """Maximum number of candidate predictions evaluated in one tensor chunk."""

    def validate(self) -> None:
        """Validate inherited frontier values and continuous-model parameters."""
        super().validate()
        if isinstance(self.history_length, bool) or not isinstance(self.history_length, int):
            raise ValueError("history_length must be a positive integer.")
        if self.history_length < 1:
            raise ValueError("history_length must be a positive integer.")
        prior_strength = float(self.prior_strength)
        if isinstance(self.prior_strength, bool) or not math.isfinite(prior_strength) or prior_strength <= 0.0:
            raise ValueError("prior_strength must be finite and positive.")
        kernel_bandwidth = float(self.kernel_bandwidth)
        if isinstance(self.kernel_bandwidth, bool) or not math.isfinite(kernel_bandwidth) or kernel_bandwidth <= 0.0:
            raise ValueError("kernel_bandwidth must be finite and positive.")
        if (
            isinstance(self.prediction_chunk_size, bool)
            or not isinstance(self.prediction_chunk_size, int)
            or self.prediction_chunk_size < 1
        ):
            raise ValueError("prediction_chunk_size must be a positive integer.")
