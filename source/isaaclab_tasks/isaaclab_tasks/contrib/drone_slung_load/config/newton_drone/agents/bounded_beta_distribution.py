# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded RSL-RL policy distribution with a hover-biased initial mean."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from rsl_rl.modules.distribution import BetaDistribution

from isaaclab_tasks.contrib.drone_slung_load.system import nominal_hover_action

_NOMINAL_HOVER_ACTION = nominal_hover_action()


class HoverBiasedBetaDistribution(BetaDistribution):
    """Beta policy whose initial deterministic action is independent of observations.

    The inherited distribution preserves RSL-RL's bounded sampling, log-probability,
    and KL-divergence behavior. This subclass only initializes the final MLP layer
    so its initial Beta mean matches ``initial_mean``.
    """

    def __init__(
        self,
        output_dim: int,
        action_range: tuple[float, float] = (-1.0, 1.0),
        initial_mean: Sequence[float] = (_NOMINAL_HOVER_ACTION, 0.0, 0.0, 0.0),
        concentration: float = 1_000.0,
    ) -> None:
        """Initialize a hover-biased bounded distribution.

        Args:
            output_dim: Dimension of the action/output space.
            action_range: Interval to which Beta samples are linearly rescaled.
            initial_mean: Initial deterministic action in ``action_range``.
            concentration: Total initial Beta concentration. Larger values produce
                lower-variance initial samples.

        Raises:
            ValueError: If the range, mean dimension, mean values, or concentration
                are invalid.
        """
        if len(initial_mean) != output_dim:
            raise ValueError(f"Expected {output_dim} initial means, got {len(initial_mean)}.")
        if not math.isfinite(concentration) or concentration <= 0.0:
            raise ValueError("concentration must be finite and positive.")
        if not all(math.isfinite(value) for value in initial_mean):
            raise ValueError("initial_mean must contain only finite values.")
        if (
            not math.isfinite(action_range[0])
            or not math.isfinite(action_range[1])
            or action_range[0] >= action_range[1]
        ):
            raise ValueError("action_range must be finite and ordered.")
        if not all(action_range[0] < value < action_range[1] for value in initial_mean):
            raise ValueError("initial_mean values must lie strictly inside action_range.")
        unit_means = tuple((value - action_range[0]) / (action_range[1] - action_range[0]) for value in initial_mean)
        if concentration * min(*unit_means, *(1.0 - value for value in unit_means)) <= 1.0:
            raise ValueError("concentration is too low to keep all Beta shape parameters above one.")

        super().__init__(output_dim=output_dim, action_range=action_range)
        self.initial_mean = tuple(initial_mean)
        self.concentration = concentration

    def init_mlp_weights(self, mlp: torch.nn.Module) -> None:
        """Set the final MLP layer to emit fixed initial Beta shape parameters."""
        final_layer = mlp[-2]  # RSL-RL MLP ends a structured output with Linear, Unflatten.
        if not isinstance(final_layer, torch.nn.Linear):
            raise TypeError("Expected the RSL-RL MLP final parameter layer to be Linear.")

        mean = torch.tensor(self.initial_mean, device=final_layer.bias.device, dtype=final_layer.bias.dtype)
        unit_mean = (mean - self._range_offset) / self._range_scale
        alpha = self.concentration * unit_mean
        beta = self.concentration * (1.0 - unit_mean)
        shape_offsets = torch.cat((alpha - 1.0, beta - 1.0))
        # Stable inverse softplus: log(expm1(x)) overflows for the deliberately
        # concentrated low-variance initialization used by the flight policy.
        raw_shapes = shape_offsets + torch.log(-torch.expm1(-shape_offsets))

        torch.nn.init.zeros_(final_layer.weight)
        with torch.no_grad():
            final_layer.bias.copy_(raw_shapes)
