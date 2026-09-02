# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure Torch episode metric accumulation for slung-load command terms."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class EpisodeMetricAccumulator:
    """Accumulate finite per-environment slung-load episode summaries.

    Samples are assumed to arrive at the fixed control cadence, so sample means
    are also time means. All public metric tensors have shape ``(num_envs,)``.
    """

    def __init__(self, num_envs: int, device: torch.device | str):
        """Create zeroed accumulators.

        Args:
            num_envs: Number of vectorized environments.
            device: Torch device holding the episode state.
        """
        self._count = torch.zeros(num_envs, device=device)
        self._position_sq_sum = torch.zeros(num_envs, device=device)
        self._swing_sum = torch.zeros(num_envs, device=device)
        self._swing_sq_sum = torch.zeros(num_envs, device=device)
        self._transverse_sq_sum = torch.zeros(num_envs, device=device)
        self._drone_speed_sum = torch.zeros(num_envs, device=device)
        self._payload_speed_sum = torch.zeros(num_envs, device=device)
        self._cable_relative_separation_sum = torch.zeros(num_envs, device=device)
        self._cable_joint_error_sum = torch.zeros(num_envs, device=device)
        self._completion_time_recorded = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.metrics = {
            "position_rmse": torch.zeros(num_envs, device=device),
            "position_error_max": torch.zeros(num_envs, device=device),
            "swing_angle_mean": torch.zeros(num_envs, device=device),
            "swing_angle_rms": torch.zeros(num_envs, device=device),
            "swing_angle_max": torch.zeros(num_envs, device=device),
            "transverse_speed_rms": torch.zeros(num_envs, device=device),
            "drone_speed_mean": torch.zeros(num_envs, device=device),
            "drone_speed_max": torch.zeros(num_envs, device=device),
            "payload_speed_mean": torch.zeros(num_envs, device=device),
            "payload_speed_max": torch.zeros(num_envs, device=device),
            "cable_relative_separation_mean": torch.zeros(num_envs, device=device),
            "cable_relative_separation_max": torch.zeros(num_envs, device=device),
            "cable_joint_error_mean": torch.zeros(num_envs, device=device),
            "cable_joint_error_max": torch.zeros(num_envs, device=device),
            "waypoint_completion_fraction": torch.zeros(num_envs, device=device),
            "waypoint_completed": torch.zeros(num_envs, device=device),
            "waypoint_completion_time": torch.zeros(num_envs, device=device),
        }

    def update(
        self,
        *,
        position_error: torch.Tensor,
        swing_angle: torch.Tensor,
        transverse_speed: torch.Tensor,
        drone_speed: torch.Tensor,
        payload_speed: torch.Tensor,
        cable_relative_separation: torch.Tensor,
        cable_joint_error: torch.Tensor,
        waypoint_fraction: torch.Tensor | None = None,
        waypoint_completed: torch.Tensor | None = None,
        elapsed_time: torch.Tensor | None = None,
    ) -> None:
        """Add one control-step sample for every environment.

        Args:
            position_error: Active-waypoint distance [m].
            swing_angle: Total cable swing angle [rad].
            transverse_speed: Payload attachment transverse speed [m/s].
            drone_speed: Drone center-of-mass speed [m/s].
            payload_speed: Payload center-of-mass speed [m/s].
            cable_relative_separation: Sum of joint gaps divided by cable length.
            cable_joint_error: Maximum attachment/internal-joint endpoint gap [m].
            waypoint_fraction: Optional completed waypoint fraction.
            waypoint_completed: Optional sequence completion mask.
            elapsed_time: Optional current episode time [s].
        """
        values = {
            "position_error": position_error,
            "swing_angle": swing_angle,
            "transverse_speed": transverse_speed,
            "drone_speed": drone_speed,
            "payload_speed": payload_speed,
            "cable_relative_separation": cable_relative_separation.abs(),
            "cable_joint_error": cable_joint_error.abs(),
        }
        finite = {name: torch.nan_to_num(value).reshape(-1) for name, value in values.items()}
        # A completed route remains active until timeout for the paper-style MDP,
        # but evaluation metrics should describe the traversal rather than the
        # arbitrary post-completion hover interval. Include the first completion
        # sample, then freeze trajectory metrics for that environment.
        active = ~self._completion_time_recorded
        active_float = active.float()
        self._count += active_float
        self._position_sq_sum += active_float * finite["position_error"].square()
        self._swing_sum += active_float * finite["swing_angle"]
        self._swing_sq_sum += active_float * finite["swing_angle"].square()
        self._transverse_sq_sum += active_float * finite["transverse_speed"].square()
        self._drone_speed_sum += active_float * finite["drone_speed"]
        self._payload_speed_sum += active_float * finite["payload_speed"]
        self._cable_relative_separation_sum += active_float * finite["cable_relative_separation"]
        self._cable_joint_error_sum += active_float * finite["cable_joint_error"]
        count = self._count.clamp_min(1.0)
        self.metrics["position_rmse"][:] = torch.sqrt(self._position_sq_sum / count)
        self.metrics["position_error_max"][:] = torch.where(
            active,
            torch.maximum(self.metrics["position_error_max"], finite["position_error"]),
            self.metrics["position_error_max"],
        )
        self.metrics["swing_angle_mean"][:] = self._swing_sum / count
        self.metrics["swing_angle_rms"][:] = torch.sqrt(self._swing_sq_sum / count)
        self.metrics["swing_angle_max"][:] = torch.where(
            active,
            torch.maximum(self.metrics["swing_angle_max"], finite["swing_angle"]),
            self.metrics["swing_angle_max"],
        )
        self.metrics["transverse_speed_rms"][:] = torch.sqrt(self._transverse_sq_sum / count)
        self.metrics["drone_speed_mean"][:] = self._drone_speed_sum / count
        self.metrics["drone_speed_max"][:] = torch.where(
            active,
            torch.maximum(self.metrics["drone_speed_max"], finite["drone_speed"]),
            self.metrics["drone_speed_max"],
        )
        self.metrics["payload_speed_mean"][:] = self._payload_speed_sum / count
        self.metrics["payload_speed_max"][:] = torch.where(
            active,
            torch.maximum(self.metrics["payload_speed_max"], finite["payload_speed"]),
            self.metrics["payload_speed_max"],
        )
        self.metrics["cable_relative_separation_mean"][:] = self._cable_relative_separation_sum / count
        self.metrics["cable_relative_separation_max"][:] = torch.where(
            active,
            torch.maximum(self.metrics["cable_relative_separation_max"], finite["cable_relative_separation"]),
            self.metrics["cable_relative_separation_max"],
        )
        self.metrics["cable_joint_error_mean"][:] = self._cable_joint_error_sum / count
        self.metrics["cable_joint_error_max"][:] = torch.where(
            active,
            torch.maximum(self.metrics["cable_joint_error_max"], finite["cable_joint_error"]),
            self.metrics["cable_joint_error_max"],
        )

        if waypoint_fraction is not None:
            self.metrics["waypoint_completion_fraction"][:] = torch.nan_to_num(waypoint_fraction).reshape(-1)
        if waypoint_completed is not None:
            completed = waypoint_completed.reshape(-1).bool()
            self.metrics["waypoint_completed"][:] = completed.float()
            if elapsed_time is not None:
                first_completion = completed & ~self._completion_time_recorded
                self.metrics["waypoint_completion_time"][first_completion] = torch.nan_to_num(elapsed_time).reshape(-1)[
                    first_completion
                ]
                self._completion_time_recorded |= completed

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        """Reset selected episode state without changing other environments."""
        for value in (
            self._count,
            self._position_sq_sum,
            self._swing_sum,
            self._swing_sq_sum,
            self._transverse_sq_sum,
            self._drone_speed_sum,
            self._payload_speed_sum,
            self._cable_relative_separation_sum,
            self._cable_joint_error_sum,
            self._completion_time_recorded,
            *self.metrics.values(),
        ):
            value[env_ids] = 0
