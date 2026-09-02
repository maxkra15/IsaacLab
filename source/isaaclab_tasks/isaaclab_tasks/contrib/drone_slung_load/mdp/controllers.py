# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Torch FLARE command scaling, rate control, allocation, and motor dynamics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from ..system import ROTOR_ARM_LENGTH, ROTOR_YAW_COEFFICIENT

AxisValue = float | Sequence[float] | torch.Tensor
AllocationMode = Literal["collective_priority", "rate_priority"]


def _axis_tensor(value: AxisValue, reference: torch.Tensor, size: int, name: str) -> torch.Tensor:
    """Convert a scalar or axis sequence to a finite tensor matching ``reference``."""
    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if tensor.ndim == 0:
        tensor = tensor.expand(size)
    if tensor.shape != (size,):
        raise ValueError(f"{name} must be a scalar or have shape ({size},), got {tuple(tensor.shape)}.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def scale_flare_actions(
    raw_actions: torch.Tensor,
    drone_mass: torch.Tensor,
    *,
    gravity: float = 9.81,
    max_thrust_to_weight: float = 3.5,
    max_body_rates: AxisValue = (15.0, 15.0, 5.0),
) -> torch.Tensor:
    """Map FLARE policy actions to collective thrust and body rates.

    Args:
        raw_actions: Policy actions, shape ``(..., 4)``. Values are clamped to ``[-1, 1]``.
        drone_mass: Drone mass [kg], broadcastable to ``raw_actions[..., 0]``. Payload
            and cable mass do not change the vehicle's available rotor thrust.
        gravity: Gravity magnitude [m/s^2].
        max_thrust_to_weight: Maximum collective thrust-to-weight ratio.
        max_body_rates: Positive body-rate limits ``[roll, pitch, yaw]`` [rad/s].

    Returns:
        Command ``[collective thrust, roll rate, pitch rate, yaw rate]`` with units
        ``[N, rad/s, rad/s, rad/s]`` and shape ``(..., 4)``.
    """
    if raw_actions.shape[-1] != 4:
        raise ValueError(f"FLARE actions must have a final dimension of 4, got {tuple(raw_actions.shape)}.")
    if not math.isfinite(gravity) or gravity <= 0.0:
        raise ValueError("gravity must be finite and positive.")
    if not math.isfinite(max_thrust_to_weight) or max_thrust_to_weight <= 0.0:
        raise ValueError("max_thrust_to_weight must be finite and positive.")

    action = torch.nan_to_num(raw_actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    mass = torch.as_tensor(drone_mass, device=action.device, dtype=action.dtype)
    if not torch.isfinite(mass).all() or torch.any(mass < 0.0):
        raise ValueError("drone_mass must contain finite nonnegative values.")
    rate_limits = _axis_tensor(max_body_rates, action, 3, "max_body_rates")
    return _scale_flare_actions_unchecked(
        action,
        mass,
        gravity=gravity,
        half_max_thrust_to_weight=0.5 * max_thrust_to_weight,
        rate_limits=rate_limits,
    )


def _scale_flare_actions_unchecked(
    clamped_actions: torch.Tensor,
    drone_mass: torch.Tensor,
    *,
    gravity: float,
    half_max_thrust_to_weight: float,
    rate_limits: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scale already-sanitized FLARE actions without device-side validation."""
    if out is None:
        thrust_to_weight = (clamped_actions[..., 0] + 1.0) * half_max_thrust_to_weight
        collective_thrust = thrust_to_weight * drone_mass * gravity
        body_rate_command = clamped_actions[..., 1:4] * rate_limits
        return torch.cat((collective_thrust.unsqueeze(-1), body_rate_command), dim=-1)

    torch.nan_to_num(clamped_actions, nan=0.0, posinf=1.0, neginf=-1.0, out=out)
    out.clamp_(-1.0, 1.0)
    out[..., 0].add_(1.0).mul_(half_max_thrust_to_weight).mul_(drone_mass).mul_(gravity)
    out[..., 1:4].mul_(rate_limits)
    return out


def quadrotor_rotor_geometry(
    *,
    arm_length: float = ROTOR_ARM_LENGTH,
    rotor_z: float = 0.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rotor positions and yaw reaction directions for an X quadrotor.

    Rotors are ordered front-right, back-right, back-left, front-left. Positions
    are center-of-mass offsets in the body frame [m], shape ``(4, 3)``. Yaw
    directions are unitless ``[-1, +1, -1, +1]``, shape ``(4,)``.
    """
    if not math.isfinite(arm_length) or arm_length <= 0.0:
        raise ValueError("arm_length must be finite and positive.")
    if not math.isfinite(rotor_z):
        raise ValueError("rotor_z must be finite.")
    positions = torch.tensor(
        (
            (arm_length, -arm_length, rotor_z),
            (-arm_length, -arm_length, rotor_z),
            (-arm_length, arm_length, rotor_z),
            (arm_length, arm_length, rotor_z),
        ),
        device=device,
        dtype=dtype,
    )
    yaw_directions = torch.tensor((-1.0, 1.0, -1.0, 1.0), device=device, dtype=dtype)
    return positions, yaw_directions


def body_rate_torque(
    rate_command: torch.Tensor,
    body_rate: torch.Tensor,
    *,
    gains: AxisValue,
    torque_limits: AxisValue,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute clipped proportional body torque from body-rate error.

    Args:
        rate_command: Commanded body angular velocity [rad/s], shape ``(..., 3)``.
        body_rate: Measured body angular velocity [rad/s], shape ``(..., 3)``.
        gains: Proportional rate gains [N·m/(rad/s)].
        torque_limits: Symmetric body-torque limits ``[Mx, My, Mz]`` [N·m].

    Returns:
        Tuple of clipped body torque [N·m] and body-rate error [rad/s].
    """
    if rate_command.shape != body_rate.shape or rate_command.shape[-1] != 3:
        raise ValueError(
            f"rate_command and body_rate must have matching (..., 3) shapes, got "
            f"{tuple(rate_command.shape)} and {tuple(body_rate.shape)}."
        )
    gain = _axis_tensor(gains, rate_command, 3, "gains")
    limit = _axis_tensor(torque_limits, rate_command, 3, "torque_limits")
    if torch.any(gain < 0.0):
        raise ValueError("gains must be nonnegative.")
    if torch.any(limit < 0.0):
        raise ValueError("torque_limits must be nonnegative.")
    return _body_rate_torque_unchecked(rate_command, body_rate, gain=gain, lower_limit=-limit, upper_limit=limit)


def _body_rate_torque_unchecked(
    rate_command: torch.Tensor,
    body_rate: torch.Tensor,
    *,
    gain: torch.Tensor,
    lower_limit: torch.Tensor,
    upper_limit: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute body-rate torque with prevalidated, device-resident parameters."""
    error = rate_command - body_rate
    return torch.clamp(error * gain, min=lower_limit, max=upper_limit), error


def quadrotor_allocation_matrix(
    *,
    arm_length: float = ROTOR_ARM_LENGTH,
    yaw_coeff: float = ROTOR_YAW_COEFFICIENT,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the X-quadrotor rotor-to-wrench allocation matrix.

    Rotors are ordered front-right, back-right, back-left, front-left at body-frame
    ``(x, y)`` positions ``[(+l, -l), (-l, -l), (-l, +l), (+l, +l)]`` [m].
    Alternating yaw directions map rotor thrust [N] to yaw torque [N·m].

    Returns:
        Matrix with rows ``[collective thrust, Mx, My, Mz]``, shape ``(4, 4)``.
    """
    if not math.isfinite(yaw_coeff) or yaw_coeff <= 0.0:
        raise ValueError("yaw_coeff must be finite and positive.")
    rotor_positions_b, yaw_directions = quadrotor_rotor_geometry(
        arm_length=arm_length,
        device=device,
        dtype=dtype,
    )
    return torch.stack(
        (
            torch.ones(4, device=device, dtype=dtype),
            rotor_positions_b[:, 1],
            -rotor_positions_b[:, 0],
            yaw_coeff * yaw_directions,
        )
    )


def reconstruct_wrench(rotor_thrusts: torch.Tensor, allocation_matrix: torch.Tensor) -> torch.Tensor:
    """Reconstruct body wrench ``[collective thrust, Mx, My, Mz]`` from rotor thrusts [N]."""
    if rotor_thrusts.shape[-1] != 4 or allocation_matrix.shape != (4, 4):
        raise ValueError("rotor_thrusts must have shape (..., 4) and allocation_matrix must have shape (4, 4).")
    return rotor_thrusts @ allocation_matrix.to(device=rotor_thrusts.device, dtype=rotor_thrusts.dtype).T


def allocate_rotor_thrusts(
    wrench: torch.Tensor,
    allocation_matrix: torch.Tensor,
    *,
    thrust_limits: tuple[float, float] = (0.0, 2.62),
    allocation_mode: AllocationMode = "collective_priority",
) -> torch.Tensor:
    """Allocate a wrench with bounded collective- or rate-priority desaturation.

    Feasible wrenches are reconstructed exactly. ``"collective_priority"``
    preserves feasible collective thrust and uniformly scales all moment-producing
    differential thrust when necessary. ``"rate_priority"`` instead shifts the
    collective to preserve the requested moments, uniformly scaling the differential
    only when it cannot fit anywhere inside the rotor thrust range. Both modes avoid
    the moment-direction distortion caused by independently clipping four rotors.

    Args:
        wrench: Desired ``[collective thrust, Mx, My, Mz]`` with units
            ``[N, N·m, N·m, N·m]``, shape ``(..., 4)``.
        allocation_matrix: Rotor-thrust-to-wrench matrix, shape ``(4, 4)``.
        thrust_limits: Inclusive per-rotor thrust limits [N].
        allocation_mode: Saturation priority. See the function description.

    Returns:
        Bounded target rotor thrusts [N], shape ``(..., 4)``.
    """
    if wrench.shape[-1] != 4 or allocation_matrix.shape != (4, 4):
        raise ValueError("wrench must have shape (..., 4) and allocation_matrix must have shape (4, 4).")
    lower, upper = thrust_limits
    if not math.isfinite(lower) or not math.isfinite(upper) or lower < 0.0 or upper < lower:
        raise ValueError("thrust_limits must be finite, nonnegative, and ordered.")
    if allocation_mode not in ("collective_priority", "rate_priority"):
        raise ValueError(f"allocation_mode must be 'collective_priority' or 'rate_priority', got {allocation_mode!r}.")
    matrix = allocation_matrix.to(device=wrench.device, dtype=wrench.dtype)
    unconstrained = torch.linalg.solve(matrix, wrench.unsqueeze(-1)).squeeze(-1)

    # Row zero of the allocation matrix is collective thrust, so subtracting
    # the rotor mean isolates a zero-sum differential that creates moments.
    differential = unconstrained - unconstrained.mean(dim=-1, keepdim=True)
    if allocation_mode == "rate_priority":
        return _rate_priority_desaturation(
            wrench,
            differential,
            lower=lower,
            upper=upper,
        )

    collective_mean = (wrench[..., :1] / 4.0).clamp(lower, upper)
    positive_scale = torch.where(
        differential > 0.0,
        (upper - collective_mean) / differential.clamp_min(torch.finfo(wrench.dtype).eps),
        torch.full_like(differential, torch.inf),
    )
    negative_scale = torch.where(
        differential < 0.0,
        (collective_mean - lower) / (-differential).clamp_min(torch.finfo(wrench.dtype).eps),
        torch.full_like(differential, torch.inf),
    )
    scale = torch.minimum(positive_scale, negative_scale).amin(dim=-1, keepdim=True).clamp(0.0, 1.0)
    return (collective_mean + scale * differential).clamp(lower, upper)


def _rate_priority_desaturation(
    wrench: torch.Tensor,
    differential: torch.Tensor,
    *,
    lower: float,
    upper: float,
) -> torch.Tensor:
    """Shift collective around a uniformly scaled moment differential."""
    differential_min = differential.amin(dim=-1, keepdim=True)
    differential_max = differential.amax(dim=-1, keepdim=True)
    differential_span = differential_max - differential_min
    available_span = upper - lower
    scale = torch.where(
        differential_span > 0.0,
        torch.full_like(differential_span, available_span) / differential_span.clamp_min(torch.finfo(wrench.dtype).eps),
        torch.ones_like(differential_span),
    ).clamp(0.0, 1.0)
    scaled_differential = scale * differential

    # Any mean in this interval fits all four rotors. Choose the one closest to
    # the requested collective so only the collective sacrificed for moments moves.
    feasible_mean_min = lower - scaled_differential.amin(dim=-1, keepdim=True)
    feasible_mean_max = upper - scaled_differential.amax(dim=-1, keepdim=True)
    requested_mean = wrench[..., :1] / 4.0
    shifted_mean = torch.maximum(torch.minimum(requested_mean, feasible_mean_max), feasible_mean_min)
    return (shifted_mean + scaled_differential).clamp(lower, upper)


def _allocate_rotor_thrusts_unchecked(
    wrench: torch.Tensor,
    allocation_inverse_transpose: torch.Tensor,
    allocation_inf: torch.Tensor,
    *,
    lower: float,
    upper: float,
) -> torch.Tensor:
    """Allocate with cached matrix inverse and prevalidated thrust limits."""
    unconstrained = wrench @ allocation_inverse_transpose
    collective_mean = (wrench[..., :1] / 4.0).clamp(lower, upper)
    differential = unconstrained - unconstrained.mean(dim=-1, keepdim=True)
    positive_scale = torch.where(
        differential > 0.0,
        (upper - collective_mean) / differential.clamp_min(torch.finfo(wrench.dtype).eps),
        allocation_inf,
    )
    negative_scale = torch.where(
        differential < 0.0,
        (collective_mean - lower) / (-differential).clamp_min(torch.finfo(wrench.dtype).eps),
        allocation_inf,
    )
    scale = torch.minimum(positive_scale, negative_scale).amin(dim=-1, keepdim=True).clamp(0.0, 1.0)
    return (collective_mean + scale * differential).clamp(lower, upper)


def _allocate_rotor_thrusts_rate_priority_unchecked(
    wrench: torch.Tensor,
    allocation_inverse_transpose: torch.Tensor,
    *,
    lower: float,
    upper: float,
) -> torch.Tensor:
    """Allocate with cached inverse and rate-priority bounded desaturation."""
    unconstrained = wrench @ allocation_inverse_transpose
    differential = unconstrained - unconstrained.mean(dim=-1, keepdim=True)
    return _rate_priority_desaturation(wrench, differential, lower=lower, upper=upper)


def apply_motor_lag(
    target: torch.Tensor,
    current: torch.Tensor,
    *,
    dt: float = 0.01,
    tau_up: float = 0.03,
    tau_down: float = 0.03,
) -> torch.Tensor:
    """Advance asymmetric first-order rotor thrust dynamics by one controller step.

    Args:
        target: Target rotor thrust [N].
        current: Current rotor thrust [N], same shape as ``target``.
        dt: Controller time step [s].
        tau_up: Thrust-increase time constant [s].
        tau_down: Thrust-decrease time constant [s].
    """
    if target.shape != current.shape:
        raise ValueError(f"target and current must have matching shapes, got {target.shape} and {current.shape}.")
    if any(not math.isfinite(value) or value <= 0.0 for value in (dt, tau_up, tau_down)):
        raise ValueError("dt, tau_up, and tau_down must be finite and positive.")
    time_constant = torch.where(target >= current, tau_up, tau_down)
    alpha = 1.0 - torch.exp(torch.full_like(target, -dt) / time_constant)
    return current + alpha * (target - current)


def _apply_motor_lag_unchecked(
    target: torch.Tensor,
    current: torch.Tensor,
    *,
    alpha_up: torch.Tensor,
    alpha_down: torch.Tensor,
) -> torch.Tensor:
    """Apply motor lag using cached, device-resident rise and fall coefficients."""
    alpha = torch.where(target >= current, alpha_up, alpha_down)
    return current + alpha * (target - current)


@dataclass(frozen=True)
class FlareControlOutput:
    """One FLARE inner-controller update, with all physical values in SI units."""

    wrench: torch.Tensor
    """Equivalent realized ``[collective thrust, Mx, My, Mz]`` in ``[N, N·m, N·m, N·m]``."""

    rate_error: torch.Tensor
    """Body-rate error [rad/s]."""

    torque_command: torch.Tensor
    """Clipped desired body torque [N·m] before rotor allocation."""

    target_rotor_thrusts: torch.Tensor
    """Clipped target rotor thrusts [N]."""

    rotor_thrusts: torch.Tensor
    """Lagged applied rotor thrusts [N]."""


class FlareController:
    """Stateful FLARE rate controller, allocator, and motor lag.

    The default is the original proportional controller with collective-priority
    allocation. Nonzero integral or derivative gains opt into a conventional rate
    PID. Its derivative acts on filtered measured body rate, not the command, and
    its bounded integral uses conditional integration at the torque limits.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        dt: float = 0.01,
        rate_gains: AxisValue = (0.016, 0.016, 0.028),
        rate_integral_gains: AxisValue = 0.0,
        rate_derivative_gains: AxisValue = 0.0,
        rate_integral_error_limits: AxisValue = 0.0,
        rate_derivative_cutoff_hz: float = 30.0,
        torque_limits: AxisValue = (0.20, 0.20, 0.08),
        arm_length: float = ROTOR_ARM_LENGTH,
        yaw_coeff: float = ROTOR_YAW_COEFFICIENT,
        rotor_thrust_limits: tuple[float, float] = (0.0, 2.62),
        allocation_mode: AllocationMode = "collective_priority",
        tau_up: float = 0.03,
        tau_down: float = 0.03,
    ) -> None:
        """Initialize controller state for ``num_envs`` parallel environments.

        Args:
            num_envs: Number of parallel controller instances.
            device: Torch device for controller state.
            dt: Controller time step [s].
            rate_gains: Proportional rate gains [N·m/(rad/s)].
            rate_integral_gains: Integral rate gains [N·m/rad]. Zero disables
                the integral term on an axis.
            rate_derivative_gains: Measured-rate derivative gains
                [N·m/(rad/s²)]. Zero disables the derivative term on an axis.
            rate_integral_error_limits: Symmetric bounds on integrated rate error
                [rad]. Zero prevents integral accumulation on an axis.
            rate_derivative_cutoff_hz: First-order measured-rate derivative filter
                cutoff [Hz].
            torque_limits: Symmetric body-torque limits ``[Mx, My, Mz]`` [N·m].
            arm_length: Rotor x/y coordinate magnitude from the center of mass [m].
            yaw_coeff: Rotor yaw moment per unit thrust [N·m/N].
            rotor_thrust_limits: Inclusive per-rotor thrust limits [N].
            allocation_mode: ``"collective_priority"`` preserves collective under
                saturation; ``"rate_priority"`` shifts collective to preserve moments.
            tau_up: Thrust-increase first-order time constant [s].
            tau_down: Thrust-decrease first-order time constant [s].
        """
        if num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if any(not math.isfinite(value) or value <= 0.0 for value in (dt, tau_up, tau_down)):
            raise ValueError("dt, tau_up, and tau_down must be finite and positive.")
        if not math.isfinite(rate_derivative_cutoff_hz) or rate_derivative_cutoff_hz <= 0.0:
            raise ValueError("rate_derivative_cutoff_hz must be finite and positive.")
        lower, upper = rotor_thrust_limits
        if not math.isfinite(lower) or not math.isfinite(upper) or lower < 0.0 or upper < lower:
            raise ValueError("rotor_thrust_limits must be finite, nonnegative, and ordered.")
        if allocation_mode not in ("collective_priority", "rate_priority"):
            raise ValueError(
                f"allocation_mode must be 'collective_priority' or 'rate_priority', got {allocation_mode!r}."
            )
        self.dt = dt
        self.rate_gains = rate_gains
        self.rate_integral_gains = rate_integral_gains
        self.rate_derivative_gains = rate_derivative_gains
        self.rate_integral_error_limits = rate_integral_error_limits
        self.rate_derivative_cutoff_hz = rate_derivative_cutoff_hz
        self.torque_limits = torque_limits
        self.rotor_thrust_limits = rotor_thrust_limits
        self.allocation_mode = allocation_mode
        self._rotor_thrust_lower = lower
        self._rotor_thrust_upper = upper
        self.tau_up = tau_up
        self.tau_down = tau_down
        self.allocation_matrix = quadrotor_allocation_matrix(arm_length=arm_length, yaw_coeff=yaw_coeff, device=device)
        self.motor_thrusts = torch.zeros(num_envs, 4, device=device)
        self._rate_gain_tensor = _axis_tensor(rate_gains, self.motor_thrusts, 3, "rate_gains")
        self._rate_integral_gain_tensor = _axis_tensor(
            rate_integral_gains, self.motor_thrusts, 3, "rate_integral_gains"
        )
        self._rate_derivative_gain_tensor = _axis_tensor(
            rate_derivative_gains, self.motor_thrusts, 3, "rate_derivative_gains"
        )
        self._rate_integral_error_limit_tensor = _axis_tensor(
            rate_integral_error_limits, self.motor_thrusts, 3, "rate_integral_error_limits"
        )
        self._torque_limit_tensor = _axis_tensor(torque_limits, self.motor_thrusts, 3, "torque_limits")
        if torch.any(self._rate_gain_tensor < 0.0):
            raise ValueError("rate_gains must be nonnegative.")
        if torch.any(self._rate_integral_gain_tensor < 0.0):
            raise ValueError("rate_integral_gains must be nonnegative.")
        if torch.any(self._rate_derivative_gain_tensor < 0.0):
            raise ValueError("rate_derivative_gains must be nonnegative.")
        if torch.any(self._rate_integral_error_limit_tensor < 0.0):
            raise ValueError("rate_integral_error_limits must be nonnegative.")
        if torch.any(self._torque_limit_tensor < 0.0):
            raise ValueError("torque_limits must be nonnegative.")
        self._rate_integral_enabled_tensor = (self._rate_integral_gain_tensor != 0.0) & (
            self._rate_integral_error_limit_tensor != 0.0
        )
        self._use_rate_integral = bool(torch.any(self._rate_integral_enabled_tensor).item())
        self._use_rate_derivative = bool(torch.any(self._rate_derivative_gain_tensor != 0.0).item())
        self._use_rate_pid = self._use_rate_integral or self._use_rate_derivative
        self._negative_torque_limit_tensor = -self._torque_limit_tensor
        self._allocation_inverse_transpose = torch.linalg.inv(self.allocation_matrix).T.contiguous()
        self._allocation_matrix_transpose = self.allocation_matrix.T.contiguous()
        self._allocation_inf = torch.full_like(self.motor_thrusts, torch.inf)
        self._motor_alpha_up = 1.0 - torch.exp(
            torch.tensor(-dt / tau_up, device=self.motor_thrusts.device, dtype=self.motor_thrusts.dtype)
        )
        self._motor_alpha_down = 1.0 - torch.exp(
            torch.tensor(-dt / tau_down, device=self.motor_thrusts.device, dtype=self.motor_thrusts.dtype)
        )
        self._derivative_filter_decay = math.exp(-2.0 * math.pi * rate_derivative_cutoff_hz * dt)
        self.rate_error_integral = torch.zeros(num_envs, 3, device=device)
        self.filtered_body_rate_derivative = torch.zeros_like(self.rate_error_integral)
        self._previous_body_rate = torch.zeros_like(self.rate_error_integral)
        self._rate_measurement_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._desired_wrench = torch.empty_like(self.motor_thrusts)

    def _compute_rate_pid_torque(
        self,
        rate_command: torch.Tensor,
        body_rate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute PID torque and update integral and filtered measurement state."""
        error = rate_command - body_rate

        if self._use_rate_derivative:
            raw_derivative = (body_rate - self._previous_body_rate) / self.dt
            raw_derivative = torch.where(
                self._rate_measurement_initialized.unsqueeze(-1),
                raw_derivative,
                torch.zeros_like(raw_derivative),
            )
            self.filtered_body_rate_derivative.mul_(self._derivative_filter_decay).add_(
                raw_derivative,
                alpha=1.0 - self._derivative_filter_decay,
            )
            self._previous_body_rate.copy_(body_rate)
            self._rate_measurement_initialized.fill_(True)

        proportional_derivative = error * self._rate_gain_tensor
        proportional_derivative.sub_(self.filtered_body_rate_derivative * self._rate_derivative_gain_tensor)

        if self._use_rate_integral:
            candidate_integral = torch.clamp(
                self.rate_error_integral + self.dt * error,
                min=-self._rate_integral_error_limit_tensor,
                max=self._rate_integral_error_limit_tensor,
            )
            candidate_integral = torch.where(
                self._rate_integral_enabled_tensor,
                candidate_integral,
                self.rate_error_integral,
            )
            candidate_torque = proportional_derivative + candidate_integral * self._rate_integral_gain_tensor
            blocks_integration = ((candidate_torque > self._torque_limit_tensor) & (error > 0.0)) | (
                (candidate_torque < self._negative_torque_limit_tensor) & (error < 0.0)
            )
            self.rate_error_integral.copy_(
                torch.where(blocks_integration, self.rate_error_integral, candidate_integral)
            )

        torque = proportional_derivative + self.rate_error_integral * self._rate_integral_gain_tensor
        return torch.clamp(
            torque,
            min=self._negative_torque_limit_tensor,
            max=self._torque_limit_tensor,
        ), error

    def compute(self, command: torch.Tensor, body_rate: torch.Tensor) -> FlareControlOutput:
        """Advance the controller from a scaled FLARE command and measured body rate.

        Args:
            command: ``[collective thrust, roll rate, pitch rate, yaw rate]`` in
                ``[N, rad/s, rad/s, rad/s]``, shape ``(N, 4)``.
            body_rate: Maximal-coordinate body angular velocity [rad/s], shape ``(N, 3)``.
        """
        if command.shape != (self.motor_thrusts.shape[0], 4):
            raise ValueError(f"command must have shape {self.motor_thrusts.shape}, got {tuple(command.shape)}.")
        if body_rate.shape != (self.motor_thrusts.shape[0], 3):
            expected_shape = (self.motor_thrusts.shape[0], 3)
            raise ValueError(f"body_rate must have shape {expected_shape}, got {tuple(body_rate.shape)}.")
        if self._use_rate_pid:
            torque, rate_error = self._compute_rate_pid_torque(command[:, 1:4], body_rate)
        else:
            # Retain the original arithmetic path exactly when the new gains are zero.
            torque, rate_error = _body_rate_torque_unchecked(
                command[:, 1:4],
                body_rate,
                gain=self._rate_gain_tensor,
                lower_limit=self._negative_torque_limit_tensor,
                upper_limit=self._torque_limit_tensor,
            )
        self._desired_wrench[:, :1].copy_(command[:, :1])
        self._desired_wrench[:, 1:4].copy_(torque)
        if self.allocation_mode == "rate_priority":
            target_rotor_thrusts = _allocate_rotor_thrusts_rate_priority_unchecked(
                self._desired_wrench,
                self._allocation_inverse_transpose,
                lower=self._rotor_thrust_lower,
                upper=self._rotor_thrust_upper,
            )
        else:
            target_rotor_thrusts = _allocate_rotor_thrusts_unchecked(
                self._desired_wrench,
                self._allocation_inverse_transpose,
                self._allocation_inf,
                lower=self._rotor_thrust_lower,
                upper=self._rotor_thrust_upper,
            )
        self.motor_thrusts[:] = _apply_motor_lag_unchecked(
            target_rotor_thrusts,
            self.motor_thrusts,
            alpha_up=self._motor_alpha_up,
            alpha_down=self._motor_alpha_down,
        ).clamp(self._rotor_thrust_lower, self._rotor_thrust_upper)
        wrench = self.motor_thrusts @ self._allocation_matrix_transpose
        return FlareControlOutput(
            wrench=wrench,
            rate_error=rate_error,
            torque_command=torque,
            target_rotor_thrusts=target_rotor_thrusts,
            rotor_thrusts=self.motor_thrusts.clone(),
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Clear motor and PID state for selected environments."""
        if env_ids is None:
            index: slice | torch.Tensor = slice(None)
        elif isinstance(env_ids, torch.Tensor):
            index = env_ids.to(device=self.motor_thrusts.device)
        else:
            index = torch.as_tensor(env_ids, device=self.motor_thrusts.device, dtype=torch.long)
        self.motor_thrusts[index] = 0.0
        self.rate_error_integral[index] = 0.0
        self.filtered_body_rate_derivative[index] = 0.0
        self._previous_body_rate[index] = 0.0
        self._rate_measurement_initialized[index] = False
