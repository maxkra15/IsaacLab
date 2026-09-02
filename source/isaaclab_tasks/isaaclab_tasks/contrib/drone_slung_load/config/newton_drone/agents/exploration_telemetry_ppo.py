# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Detached exploration telemetry for the enhanced slung-load PPO policy."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from rsl_rl.algorithms import PPO

from isaaclab_tasks.contrib.drone_slung_load.system import (
    DRONE_MASS,
    ENHANCED_RESIDUAL_BODY_RATE_LIMITS,
    GRAVITY,
    MAX_THRUST_TO_WEIGHT,
)

from .bounded_tanh_gaussian_distribution import HoverBiasedTanhGaussianDistribution

_ACTION_AXES = ("collective", "roll", "pitch", "yaw")
_PHYSICAL_ACTION_AXES = ("collective_N", "roll_rate_rad_s", "pitch_rate_rad_s", "yaw_rate_rad_s")
# These rate scales are the enhanced task's policy-residual envelope. The focused
# configuration test guards this telemetry metadata against drift from the MDP.
_COLLECTIVE_ACTION_SCALE = 0.5 * MAX_THRUST_TO_WEIGHT * DRONE_MASS * GRAVITY


def compute_hover_exploration_metrics(
    distribution: HoverBiasedTanhGaussianDistribution,
    body_rate_limits: Sequence[float] = ENHANCED_RESIDUAL_BODY_RATE_LIMITS,
) -> dict[str, float]:
    """Return detached exploration diagnostics at the configured hover mean.

    The bounded-action values use the same local tanh linearization exposed by
    :attr:`HoverBiasedTanhGaussianDistribution.std`. Physical values then apply
    the enhanced task's collective-thrust and residual-rate scales. The entropy
    is explicitly named as base-Normal differential entropy because it is not
    the entropy of the squashed action or its dimensional command.
    """
    if distribution.output_dim != len(_ACTION_AXES):
        raise ValueError(f"Expected {len(_ACTION_AXES)} action axes, got {distribution.output_dim}.")
    if len(body_rate_limits) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in body_rate_limits):
        raise ValueError("body_rate_limits must contain three finite positive values.")

    with torch.no_grad():
        std_lower, std_upper = distribution.std_range
        clamped_log_std = distribution.log_std_param.detach().clamp(math.log(std_lower), math.log(std_upper))
        latent_std = clamped_log_std.exp()
        lower, upper = distribution.action_range
        action_scale = 0.5 * (upper - lower)
        normalized_hover_mean = torch.tanh(distribution.initial_latent_mean.detach())
        action_std = latent_std * action_scale * (1.0 - normalized_hover_mean.square())
        physical_scales = torch.as_tensor(
            (_COLLECTIVE_ACTION_SCALE, *body_rate_limits),
            device=action_std.device,
            dtype=action_std.dtype,
        )
        physical_std = action_std * physical_scales
        base_normal_entropy = clamped_log_std.sum() + len(_ACTION_AXES) * 0.5 * math.log(2.0 * math.pi * math.e)

        metrics = {
            f"exploration/std_latent/{axis}": value.item() for axis, value in zip(_ACTION_AXES, latent_std, strict=True)
        }
        metrics.update(
            {
                f"exploration/std_action_local/{axis}": value.item()
                for axis, value in zip(_ACTION_AXES, action_std, strict=True)
            }
        )
        metrics.update(
            {
                f"exploration/std_physical/{axis}": value.item()
                for axis, value in zip(_PHYSICAL_ACTION_AXES, physical_std, strict=True)
            }
        )
        metrics["exploration/base_normal_entropy_nats"] = base_normal_entropy.item()
        return metrics


class DroneSlungLoadTelemetryPPO(PPO):
    """PPO with post-update, detached exploration telemetry for the enhanced task."""

    def __init__(
        self,
        *args: Any,
        physical_body_rate_limits: Sequence[float] = ENHANCED_RESIDUAL_BODY_RATE_LIMITS,
        **kwargs: Any,
    ) -> None:
        """Initialize PPO and bind telemetry to the policy's physical rate envelope."""
        if len(physical_body_rate_limits) != 3 or any(
            not math.isfinite(value) or value <= 0.0 for value in physical_body_rate_limits
        ):
            raise ValueError("physical_body_rate_limits must contain three finite positive values.")
        self._physical_body_rate_limits = tuple(float(value) for value in physical_body_rate_limits)
        super().__init__(*args, **kwargs)

    def update(self) -> dict[str, float]:
        """Run the unchanged PPO update, then append scalar-only diagnostics."""
        loss_dict = super().update()
        distribution = self.get_policy().distribution
        if not isinstance(distribution, HoverBiasedTanhGaussianDistribution):
            raise TypeError(
                "DroneSlungLoadTelemetryPPO requires HoverBiasedTanhGaussianDistribution, "
                f"got {type(distribution).__name__}."
            )
        body_rate_limits = getattr(self, "_physical_body_rate_limits", ENHANCED_RESIDUAL_BODY_RATE_LIMITS)
        loss_dict.update(compute_hover_exploration_metrics(distribution, body_rate_limits))
        return loss_dict
