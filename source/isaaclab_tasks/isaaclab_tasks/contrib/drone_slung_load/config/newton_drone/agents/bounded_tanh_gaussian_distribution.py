# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded tanh-Gaussian policy distribution with a loaded-hover initial mean."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from rsl_rl.modules.distribution import Distribution
from torch import nn
from torch.distributions import Normal

from isaaclab_tasks.contrib.drone_slung_load.system import nominal_hover_action

_NOMINAL_HOVER_ACTION = nominal_hover_action()
_ACTION_EPS = 1.0e-6


class HoverBiasedTanhGaussianDistribution(Distribution):
    """A bounded Gaussian policy centered on the loaded-hover action.

    The actor predicts a residual in pre-tanh space around ``initial_mean``.
    Sampling is therefore bounded without a downstream action clip, while the
    direct latent-mean head retains well-conditioned policy gradients. PPO uses
    the exact change-of-variables log probability and the base-Normal KL. The
    entropy bonus intentionally uses base-Normal entropy, which keeps exploration
    pressure independent of the asymmetric collective-thrust operating point.
    """

    def __init__(
        self,
        output_dim: int,
        action_range: tuple[float, float] = (-1.0, 1.0),
        initial_mean: Sequence[float] = (_NOMINAL_HOVER_ACTION, 0.0, 0.0, 0.0),
        init_std: float | Sequence[float] | torch.Tensor = 0.003,
        std_range: tuple[float, float] = (0.001, 0.15),
        learn_std: bool = True,
    ) -> None:
        """Initialize a hover-biased tanh-Gaussian distribution.

        Args:
            output_dim: Dimension of the action/output space.
            action_range: Open interval containing every initial action.
            initial_mean: Initial deterministic action in ``action_range``.
            init_std: Initial pre-tanh Normal standard deviation.
            std_range: Inclusive pre-tanh standard-deviation limits.
            learn_std: Whether PPO may learn the log standard deviation.

        Raises:
            ValueError: If the range, mean, or standard-deviation values are invalid.
        """
        super().__init__(output_dim)
        lower, upper = action_range
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("action_range must be finite and ordered.")
        if len(initial_mean) != output_dim:
            raise ValueError(f"Expected {output_dim} initial means, got {len(initial_mean)}.")
        if not all(math.isfinite(value) and lower < value < upper for value in initial_mean):
            raise ValueError("initial_mean values must be finite and lie strictly inside action_range.")
        std_lower, std_upper = std_range
        if not math.isfinite(std_lower) or not math.isfinite(std_upper) or std_lower <= 0.0 or std_lower > std_upper:
            raise ValueError("std_range must be finite, positive, and ordered.")

        initial_std = torch.as_tensor(init_std, dtype=torch.float32)
        if initial_std.ndim == 0:
            initial_std = initial_std.expand(output_dim).clone()
        if initial_std.shape != (output_dim,):
            raise ValueError(
                f"init_std must be a scalar or have shape ({output_dim},), got {tuple(initial_std.shape)}."
            )
        if (
            not torch.isfinite(initial_std).all()
            or torch.any(initial_std < std_lower)
            or torch.any(initial_std > std_upper)
        ):
            raise ValueError("init_std must contain finite values inside std_range.")

        self.action_range = action_range
        self.initial_mean = tuple(initial_mean)
        self.std_range = std_range
        self._action_scale = 0.5 * (upper - lower)
        self._action_offset = 0.5 * (upper + lower)
        self._log_action_scale = math.log(self._action_scale)
        self._log_std_range = (math.log(std_lower), math.log(std_upper))

        normalized_mean = (
            torch.as_tensor(initial_mean, dtype=torch.float32) - self._action_offset
        ) / self._action_scale
        self.register_buffer("initial_latent_mean", torch.atanh(normalized_mean))
        self.log_std_param = nn.Parameter(torch.log(initial_std), requires_grad=learn_std)
        self._distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def _latent_mean(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output + self.initial_latent_mean

    def _squash(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.tanh(latent) * self._action_scale + self._action_offset

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the base Normal from the actor's pre-tanh mean residual."""
        latent_mean = self._latent_mean(mlp_output)
        log_std = self.log_std_param.clamp(*self._log_std_range)
        self._distribution = Normal(latent_mean, torch.exp(log_std))

    def sample(self) -> torch.Tensor:
        """Sample an action strictly bounded by ``action_range``."""
        return self._squash(self._distribution.sample())  # type: ignore[union-attr]

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Return the squashed mean action for deterministic evaluation."""
        return self._squash(self._latent_mean(mlp_output))

    def as_deterministic_output_module(self) -> nn.Module:
        """Return an export-friendly module implementing the same mean projection."""
        return _TanhGaussianDeterministicOutput(
            self.initial_latent_mean.detach().clone(), self._action_scale, self._action_offset
        )

    @property
    def input_dim(self) -> int:
        """Return the direct latent-mean output dimension required from the MLP."""
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        """Return the deterministic action associated with the current base mean."""
        return self._squash(self._distribution.mean)  # type: ignore[union-attr]

    @property
    def std(self) -> torch.Tensor:
        """Return the local linearized action standard deviation."""
        normalized_mean = torch.tanh(self._distribution.mean)  # type: ignore[union-attr]
        return self._distribution.stddev * self._action_scale * (1.0 - normalized_mean.square())  # type: ignore[union-attr]

    @property
    def entropy(self) -> torch.Tensor:
        """Return base-Normal entropy summed over action dimensions."""
        return self._distribution.entropy().sum(dim=-1)  # type: ignore[union-attr]

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Return base-Normal mean and standard deviation for exact PPO KL."""
        return (self._distribution.mean, self._distribution.stddev)  # type: ignore[union-attr]

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Return the exact transformed log probability of bounded actions."""
        normalized = (outputs - self._action_offset) / self._action_scale
        normalized = normalized.clamp(-1.0 + _ACTION_EPS, 1.0 - _ACTION_EPS)
        latent = torch.atanh(normalized)
        log_abs_det_jacobian = self._log_action_scale + torch.log1p(-normalized.square())
        return (self._distribution.log_prob(latent) - log_abs_det_jacobian).sum(dim=-1)  # type: ignore[union-attr]

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Return exact transformed-policy KL via invariance of the shared bijection."""
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        return torch.distributions.kl_divergence(Normal(old_mean, old_std), Normal(new_mean, new_std)).sum(dim=-1)

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        """Initialize the actor to emit zero residual from loaded hover."""
        final_layer = mlp[-1]  # RSL-RL MLP with an integer output ends in Linear.
        if not isinstance(final_layer, nn.Linear):
            raise TypeError("Expected the RSL-RL MLP final parameter layer to be Linear.")
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)


class _TanhGaussianDeterministicOutput(nn.Module):
    """Exportable loaded-hover tanh projection."""

    def __init__(self, initial_latent_mean: torch.Tensor, action_scale: float, action_offset: float) -> None:
        super().__init__()
        self.register_buffer("initial_latent_mean", initial_latent_mean)
        self.action_scale = action_scale
        self.action_offset = action_offset

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Project latent-mean residuals into the bounded action interval."""
        return torch.tanh(mlp_output + self.initial_latent_mean) * self.action_scale + self.action_offset
