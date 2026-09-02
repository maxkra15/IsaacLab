# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FLARE waypoint reward terms with policy-frequency-independent impulses."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply

from .bodies import link_ang_vel_b, link_lin_vel_w, link_pos_e, link_quat_w
from .observations import payload_transverse_velocity_b, swing_angles, total_swing_angle

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _positive_scale(scale: tuple[float, ...] | float, dimension: int, name: str) -> tuple[float, ...]:
    values = (scale,) * dimension if isinstance(scale, int | float) else scale
    if len(values) != dimension or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"{name} must contain {dimension} finite positive values.")
    return values


def _normalized_l2(value: torch.Tensor, scale: tuple[float, ...] | float, name: str) -> torch.Tensor:
    values = _positive_scale(scale, value.shape[-1], name)
    divisor = torch.as_tensor(values, device=value.device, dtype=value.dtype)
    normalized = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) / divisor
    return torch.nan_to_num(torch.sum(normalized.square(), dim=-1))


def _normalized_clamped_action(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Read the task action term and reproduce its normalized physical clamp."""
    manager = env.action_manager
    if hasattr(manager, "get_term"):
        term = manager.get_term(action_name)
        # The drone action's physical processing starts by applying this exact
        # clamp. Its public processed_actions are dimensional thrust/rate values,
        # so raw_actions plus the physical clamp is the normalized representation.
        action = term.raw_actions
    else:
        action = manager.action
    return torch.nan_to_num(action.detach(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


def record_episode_metrics(env: ManagerBasedRLEnv, command_name: str = "route") -> torch.Tensor:
    """Record one post-physics episode sample before autoreset and return zero reward."""
    env.command_manager.get_term(command_name).record_metrics_step()
    return torch.zeros(env.num_envs, device=env.device)


def waypoint_progress(
    env: ManagerBasedRLEnv,
    command_name: str = "route",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared-distance waypoint progress divided by ``step_dt``.

    Isaac Lab multiplies reward terms by ``step_dt``. Dividing here recovers
    FLARE's discrete-step reward from Eq. (11). RewardManager executes before
    CommandManager, so this evaluates the latest post-physics position directly.
    """
    term = env.command_manager.get_term(command_name)
    current_waypoint = env.command_manager.get_command(command_name)[:, :3]
    position = link_pos_e(env, asset_cfg)
    previous_distance_sq = term.previous_distance_sq
    finite = (
        torch.isfinite(current_waypoint).all(dim=-1)
        & torch.isfinite(position).all(dim=-1)
        & torch.isfinite(previous_distance_sq)
    )
    delta = torch.nan_to_num(current_waypoint - position, nan=0.0, posinf=0.0, neginf=0.0)
    distance_sq = torch.sum(torch.square(delta), dim=-1)
    reward = (torch.nan_to_num(previous_distance_sq) - distance_sq) / env.step_dt
    return torch.where(finite, torch.nan_to_num(reward), torch.zeros_like(reward))


def waypoint_distance_progress(
    env: ManagerBasedRLEnv,
    command_name: str = "route",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Linear active-waypoint distance progress divided by ``step_dt`` [m/s].

    Unlike the paper's squared-distance progress, this enhanced variant has a
    route-length-independent scale when route geometry is randomized.
    """
    term = env.command_manager.get_term(command_name)
    current_waypoint = env.command_manager.get_command(command_name)[:, :3]
    position = link_pos_e(env, asset_cfg)
    previous_distance_sq = term.previous_distance_sq
    finite = (
        torch.isfinite(current_waypoint).all(dim=-1)
        & torch.isfinite(position).all(dim=-1)
        & torch.isfinite(previous_distance_sq)
        & (previous_distance_sq >= 0.0)
    )
    delta = torch.nan_to_num(current_waypoint - position, nan=0.0, posinf=0.0, neginf=0.0)
    current_distance = torch.linalg.vector_norm(delta, dim=-1)
    previous_distance = torch.sqrt(torch.nan_to_num(previous_distance_sq).clamp_min(0.0))
    reward = (previous_distance - current_distance) / env.step_dt
    return torch.where(finite, torch.nan_to_num(reward), torch.zeros_like(reward))


def action_delta_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Euclidean norm of consecutive normalized actions divided by ``step_dt``."""
    current = torch.nan_to_num(env.action_manager.action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    previous = torch.nan_to_num(env.action_manager.prev_action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    delta = current - previous
    return torch.nan_to_num(torch.linalg.vector_norm(delta, dim=-1) / env.step_dt)


def indexed_cross_track_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "route",
    scale: float = 0.5,
) -> torch.Tensor:
    """Squared indexed-route cross-track distance normalized by ``scale`` [m].

    The command term owns segment indexing, which prevents shortcuts to a
    geometrically nearby but topologically different part of a crossing route.
    """
    scale_value = _positive_scale(scale, 1, "scale")[0]
    distance = env.command_manager.get_term(command_name).cross_track_distance
    if distance.shape == (env.num_envs, 1):
        distance = distance[:, 0]
    if distance.shape != (env.num_envs,):
        raise ValueError(f"cross_track_distance must have shape ({env.num_envs},), got {tuple(distance.shape)}.")
    normalized = torch.nan_to_num(distance, nan=0.0, posinf=0.0, neginf=0.0) / scale_value
    return torch.nan_to_num(normalized.square())


def indexed_cross_track_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "route",
    scale: float = 0.5,
) -> torch.Tensor:
    """Robust indexed-route cross-track cost in ``[0, 1)``.

    The cost is ``1 - exp(-(distance / scale)^2)``. It matches normalized L2
    locally, while smoothly limiting the contribution from trajectories that
    are already outside the useful tracking region. This keeps failed rollouts
    from dominating critic targets without hiding their failure termination.
    """
    squared_error = indexed_cross_track_error_l2(env, command_name=command_name, scale=scale)
    return torch.nan_to_num(-torch.expm1(-squared_error), nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


def path_arc_length_progress(
    env: ManagerBasedRLEnv,
    command_name: str = "route",
    maximum_rate: float | None = None,
    maximum_lateral_acceleration: float | None = None,
    positive_progress_gate_distance: float | None = None,
) -> torch.Tensor:
    """Indexed spline arc-length progress rate [m/s].

    RewardManager multiplies this per-second term by ``step_dt``, recovering the
    signed change in projected route arc length. CommandManager stores the
    current projection after every reward evaluation and reseeds the potential
    after waypoint switches, preventing an index-change impulse.

    Args:
        env: Manager-based RL environment.
        command_name: Spline-enabled waypoint command term.
        maximum_rate: Optional symmetric clipping magnitude [m/s]. This is read
            every call and can be changed by a curriculum.
        maximum_lateral_acceleration: Optional positive lateral-acceleration
            limit [m/s^2]. When provided, local path curvature further limits
            the symmetric rewarded rate to ``sqrt(a_lat / curvature)``.
        positive_progress_gate_distance: Optional cross-track distance [m] at
            which positive progress is fully suppressed. Between zero and this
            distance it is scaled by the C1 gate ``(1 - u^2)^2``. Negative
            progress is never suppressed.
    """
    if maximum_rate is not None and (not math.isfinite(maximum_rate) or maximum_rate <= 0.0):
        raise ValueError("maximum_rate must be finite and positive when provided.")
    if maximum_lateral_acceleration is not None and (
        not math.isfinite(maximum_lateral_acceleration) or maximum_lateral_acceleration <= 0.0
    ):
        raise ValueError("maximum_lateral_acceleration must be finite and positive when provided.")
    if positive_progress_gate_distance is not None and (
        isinstance(positive_progress_gate_distance, bool)
        or not isinstance(positive_progress_gate_distance, int | float)
        or not math.isfinite(positive_progress_gate_distance)
        or positive_progress_gate_distance <= 0.0
    ):
        raise ValueError("positive_progress_gate_distance must be finite and positive when provided.")
    term = env.command_manager.get_term(command_name)
    current = term.path_progress
    previous = term.previous_path_progress
    if current.shape != (env.num_envs,) or previous.shape != (env.num_envs,):
        raise ValueError(
            f"path progress buffers must have shape ({env.num_envs},), got "
            f"{tuple(current.shape)} and {tuple(previous.shape)}."
        )
    valid = term.path_state_valid & torch.isfinite(current) & torch.isfinite(previous)
    progress_rate = (torch.nan_to_num(current) - torch.nan_to_num(previous)) / env.step_dt
    if maximum_lateral_acceleration is not None:
        curvature = term.path_curvature_e
        if curvature.shape != (env.num_envs, 3):
            raise ValueError(f"path curvature must have shape ({env.num_envs}, 3), got {tuple(curvature.shape)}.")
        finite_curvature = torch.isfinite(curvature).all(dim=-1)
        valid &= finite_curvature
        curvature_magnitude = torch.linalg.vector_norm(torch.nan_to_num(curvature), dim=-1)
        local_rate_cap = torch.sqrt(
            maximum_lateral_acceleration / curvature_magnitude.clamp_min(torch.finfo(curvature_magnitude.dtype).eps)
        )
        if maximum_rate is not None:
            local_rate_cap.clamp_max_(maximum_rate)
        progress_rate = torch.maximum(torch.minimum(progress_rate, local_rate_cap), -local_rate_cap)
    elif maximum_rate is not None:
        progress_rate.clamp_(-maximum_rate, maximum_rate)
    if positive_progress_gate_distance is not None:
        distance = term.path_cross_track_distance
        if distance.shape != (env.num_envs,):
            raise ValueError(
                f"path_cross_track_distance must have shape ({env.num_envs},), got {tuple(distance.shape)}."
            )
        finite_distance = torch.isfinite(distance)
        valid &= finite_distance
        normalized_distance = torch.nan_to_num(
            distance,
            nan=positive_progress_gate_distance,
            posinf=positive_progress_gate_distance,
            neginf=positive_progress_gate_distance,
        )
        normalized_distance = (normalized_distance / positive_progress_gate_distance).clamp_(0.0, 1.0)
        positive_gate = (1.0 - normalized_distance.square()).square()
        progress_rate = progress_rate.clamp_max(0.0) + positive_gate * progress_rate.clamp_min(0.0)
    return torch.where(valid, torch.nan_to_num(progress_rate), torch.zeros_like(progress_rate))


def path_tangent_speed_tracking_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "route",
    speed_error_scale: float = 1.0,
    underspeed_weight: float = 1.0,
    overspeed_weight: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track the command term's curvature- and braking-aware path speed.

    The signed tangent speed is ``dot(v, tangent)``; taking an absolute speed
    here would incorrectly reward backtracking. The command term owns the
    reference calculation so observations, rewards, and evaluation share one
    definition of
    ``min(cruise, curvature limit, preview-braking limit, finish-braking limit)``.
    This function returns a nonnegative normalized squared cost, intended for a
    negative reward weight.

    Args:
        env: Manager-based RL environment.
        command_name: Spline-enabled waypoint command term.
        speed_error_scale: Normalization scale for tangent-speed error [m/s].
        underspeed_weight: Nonnegative multiplier below the reference speed.
        overspeed_weight: Nonnegative multiplier above the reference speed.
        asset_cfg: Asset whose world-frame velocity is tracked.

    Returns:
        Per-environment dimensionless speed-tracking cost.
    """
    positive_parameters = {"speed_error_scale": speed_error_scale}
    for name, value in positive_parameters.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    for name, value in {"underspeed_weight": underspeed_weight, "overspeed_weight": overspeed_weight}.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative.")

    term = env.command_manager.get_term(command_name)
    reference_speed = term.path_speed_reference
    tangent = term.path_tangent_e
    if reference_speed.shape != (env.num_envs,):
        raise ValueError(f"path speed reference must have shape ({env.num_envs},), got {tuple(reference_speed.shape)}.")
    if tangent.shape != (env.num_envs, 3):
        raise ValueError(f"path tangent must have shape ({env.num_envs}, 3), got {tuple(tangent.shape)}.")

    finite = torch.isfinite(reference_speed) & (reference_speed >= 0.0) & torch.isfinite(tangent).all(dim=-1)
    if hasattr(term, "path_state_valid"):
        valid = term.path_state_valid
        if valid.shape != (env.num_envs,):
            raise ValueError(f"path_state_valid must have shape ({env.num_envs},), got {tuple(valid.shape)}.")
        finite &= valid
    tangent = torch.nan_to_num(tangent)
    tangent_norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    finite &= tangent_norm[:, 0] > torch.finfo(tangent.dtype).eps
    tangent /= tangent_norm.clamp_min(torch.finfo(tangent.dtype).eps)
    velocity = link_lin_vel_w(env, asset_cfg)
    if velocity.shape != (env.num_envs, 3):
        raise ValueError(f"tracked velocity must have shape ({env.num_envs}, 3), got {tuple(velocity.shape)}.")
    finite &= torch.isfinite(velocity).all(dim=-1)
    signed_tangent_speed = torch.sum(torch.nan_to_num(velocity) * tangent, dim=-1)
    speed_error = signed_tangent_speed - torch.nan_to_num(reference_speed)
    multiplier = torch.where(speed_error > 0.0, overspeed_weight, underspeed_weight)
    cost = multiplier * torch.square(speed_error / speed_error_scale)
    return torch.where(finite, torch.nan_to_num(cost), torch.zeros_like(cost))


def path_velocity_tracking_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "route",
    cross_track_gain: float = 1.5,
    maximum_cross_track_speed: float = 0.75,
    velocity_error_scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track a 3D path velocity with bounded inward cross-track recovery.

    The desired velocity is the command's curvature- and braking-aware
    tangential reference plus ``-cross_track_gain * cross_track_error``. The
    command error points from the indexed path projection to the vehicle, so
    the explicit negative sign makes the recovery velocity point inward. Its
    Euclidean norm is capped independently by ``maximum_cross_track_speed``.

    This function returns a nonnegative normalized squared cost intended for a
    negative reward weight. Invalid terminal samples return zero instead of
    injecting non-finite values into the critic target.
    """
    for name, value in {
        "maximum_cross_track_speed": maximum_cross_track_speed,
        "velocity_error_scale": velocity_error_scale,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if (
        isinstance(cross_track_gain, bool)
        or not isinstance(cross_track_gain, int | float)
        or not math.isfinite(cross_track_gain)
        or cross_track_gain < 0.0
    ):
        raise ValueError("cross_track_gain must be finite and nonnegative.")

    term = env.command_manager.get_term(command_name)
    reference_speed = term.path_speed_reference
    tangent = term.path_tangent_e
    cross_track_error = term.path_cross_track_error_e
    velocity = link_lin_vel_w(env, asset_cfg)
    if reference_speed.shape != (env.num_envs,):
        raise ValueError(f"path speed reference must have shape ({env.num_envs},), got {tuple(reference_speed.shape)}.")
    if tangent.shape != (env.num_envs, 3):
        raise ValueError(f"path tangent must have shape ({env.num_envs}, 3), got {tuple(tangent.shape)}.")
    if cross_track_error.shape != (env.num_envs, 3):
        raise ValueError(
            f"path cross-track error must have shape ({env.num_envs}, 3), got {tuple(cross_track_error.shape)}."
        )
    if velocity.shape != (env.num_envs, 3):
        raise ValueError(f"tracked velocity must have shape ({env.num_envs}, 3), got {tuple(velocity.shape)}.")

    finite = (
        torch.isfinite(reference_speed)
        & (reference_speed >= 0.0)
        & torch.isfinite(tangent).all(dim=-1)
        & torch.isfinite(cross_track_error).all(dim=-1)
        & torch.isfinite(velocity).all(dim=-1)
    )
    if hasattr(term, "path_state_valid"):
        path_state_valid = term.path_state_valid
        if path_state_valid.shape != (env.num_envs,) or path_state_valid.dtype != torch.bool:
            raise ValueError(f"path_state_valid must be a bool tensor with shape ({env.num_envs},).")
        finite &= path_state_valid

    safe_tangent = torch.nan_to_num(tangent)
    tangent_norm = torch.linalg.vector_norm(safe_tangent, dim=-1, keepdim=True)
    finite &= tangent_norm[:, 0] > torch.finfo(safe_tangent.dtype).eps
    unit_tangent = safe_tangent / tangent_norm.clamp_min(torch.finfo(safe_tangent.dtype).eps)
    desired_velocity = unit_tangent * torch.nan_to_num(reference_speed).clamp_min(0.0).unsqueeze(-1)

    recovery_velocity = -cross_track_gain * torch.nan_to_num(cross_track_error)
    recovery_norm = torch.linalg.vector_norm(recovery_velocity, dim=-1, keepdim=True)
    recovery_scale = maximum_cross_track_speed / recovery_norm.clamp_min(torch.finfo(recovery_norm.dtype).eps)
    recovery_velocity *= recovery_scale.clamp_max(1.0)
    desired_velocity += recovery_velocity

    normalized_error = (torch.nan_to_num(velocity) - desired_velocity) / velocity_error_scale
    cost = torch.sum(normalized_error.square(), dim=-1)
    finite &= torch.isfinite(cost)
    return torch.where(finite, torch.nan_to_num(cost), torch.zeros_like(cost))


def path_transverse_speed_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "route",
    scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared drone speed transverse to the local spline tangent, normalized by ``scale`` [m/s]."""
    term = env.command_manager.get_term(command_name)
    tangent = torch.nan_to_num(term.path_tangent_e)
    tangent /= torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(torch.finfo(tangent.dtype).eps)
    velocity = torch.nan_to_num(link_lin_vel_w(env, asset_cfg), nan=0.0, posinf=0.0, neginf=0.0)
    transverse_velocity = velocity - torch.sum(velocity * tangent, dim=-1, keepdim=True) * tangent
    return _normalized_l2(transverse_velocity, scale, "scale")


def path_tracking_precision_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "route",
    cross_track_scale: float = 0.20,
    transverse_velocity_scale: float = 1.0,
    transverse_speed_weight: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Bounded precision cost combining spline error and transverse speed.

    The result is ``1 - exp(-e)`` in ``[0, 1]``, where ``e`` is normalized
    cross-track squared error plus a weighted normalized transverse-speed
    squared error. It is locally quadratic for precise tracking and saturates
    smoothly for failed trajectories, protecting critic targets. Use a negative
    reward weight because this function returns a nonnegative cost.

    All scales and the nonnegative mixing weight are validated and consumed on
    every call, making them safe targets for a runtime curriculum.
    """
    if not math.isfinite(transverse_speed_weight) or transverse_speed_weight < 0.0:
        raise ValueError("transverse_speed_weight must be finite and nonnegative.")
    cross_track_error = indexed_cross_track_error_l2(env, command_name=command_name, scale=cross_track_scale)
    transverse_speed = path_transverse_speed_l2(
        env, command_name=command_name, scale=transverse_velocity_scale, asset_cfg=asset_cfg
    )
    normalized_error = cross_track_error + transverse_speed_weight * transverse_speed
    return torch.nan_to_num(-torch.expm1(-normalized_error), nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


def path_tracking_precision_log1p(
    env: ManagerBasedRLEnv,
    command_name: str = "route",
    cross_track_scale: float = 0.20,
    transverse_velocity_scale: float = 1.0,
    transverse_speed_weight: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robust precision cost with independently persistent error gradients.

    The cost is
    ``log(1 + (d / cross_track_scale)^2) + transverse_speed_weight *
    log(1 + (v_perp / transverse_velocity_scale)^2)``. It is locally
    quadratic like the bounded exponential cost, but its logarithmic tails do
    not become flat after a trajectory leaves the precision tube. Keeping the
    two robust costs separable also prevents a large position error from
    suppressing the transverse-speed learning signal.

    Use a negative reward weight because this function returns a nonnegative
    cost. All scales and the nonnegative mixing weight are consumed on every
    call so a runtime curriculum can update them safely.
    """
    if not math.isfinite(transverse_speed_weight) or transverse_speed_weight < 0.0:
        raise ValueError("transverse_speed_weight must be finite and nonnegative.")
    cross_track_error = indexed_cross_track_error_l2(env, command_name=command_name, scale=cross_track_scale)
    transverse_speed = path_transverse_speed_l2(
        env, command_name=command_name, scale=transverse_velocity_scale, asset_cfg=asset_cfg
    )
    cost = torch.log1p(cross_track_error) + transverse_speed_weight * torch.log1p(transverse_speed)
    return torch.nan_to_num(cost, nan=0.0, posinf=torch.finfo(cost.dtype).max, neginf=0.0).clamp_min_(0.0)


def total_swing_angle_l2(
    env: ManagerBasedRLEnv,
    scale: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    robot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    payload_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Continuous squared total payload deflection, normalized by ``scale`` [rad]."""
    angle = total_swing_angle(env, robot_cfg, payload_cfg, robot_offset, payload_offset)
    return _normalized_l2(angle, scale, "scale")


def payload_transverse_speed_l2(
    env: ManagerBasedRLEnv,
    scale: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    robot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    payload_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Squared relative attachment speed transverse to the cable, normalized by ``scale`` [m/s]."""
    velocity = payload_transverse_velocity_b(env, robot_cfg, payload_cfg, robot_offset, payload_offset)
    return _normalized_l2(velocity, scale, "scale")


def body_angular_velocity_l2(
    env: ManagerBasedRLEnv,
    scale: tuple[float, float, float] | float = math.pi,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared body angular rate normalized per axis by ``scale`` [rad/s]."""
    return _normalized_l2(link_ang_vel_b(env, asset_cfg), scale, "scale")


def body_tilt_l2(
    env: ManagerBasedRLEnv,
    scale: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Smooth yaw-invariant upright cost normalized by ``scale`` [rad].

    ``2 * (1 - z_body · z_world)`` is equal to squared tilt angle locally and
    remains well conditioned at perfect hover. Unlike Euler-angle penalties it
    is continuous across yaw and still penalizes an inverted vehicle.
    """
    scale_value = _positive_scale(scale, 1, "scale")[0]
    quaternion = link_quat_w(env, asset_cfg)
    body_z = torch.zeros(quaternion.shape[0], 3, device=quaternion.device, dtype=quaternion.dtype)
    body_z[:, 2] = 1.0
    world_z_component = quat_apply(quaternion, body_z)[:, 2]
    finite = torch.isfinite(world_z_component)
    cost = 2.0 * (1.0 - torch.nan_to_num(world_z_component).clamp(-1.0, 1.0)) / (scale_value * scale_value)
    return torch.where(finite, torch.nan_to_num(cost), torch.zeros_like(cost))


def body_tilt_exp(
    env: ManagerBasedRLEnv,
    scale: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robust yaw-invariant upright cost in ``[0, 1)``."""
    squared_error = body_tilt_l2(env, scale=scale, asset_cfg=asset_cfg)
    return torch.nan_to_num(-torch.expm1(-squared_error), nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


class NormalizedActionAccelerationL2(ManagerTermBase):
    """Squared second difference of clamped normalized policy actions.

    This discrete regularizer intentionally operates in normalized action space,
    avoiding a unit-dependent mixture of collective thrust and body rates. The
    first sample after each environment reset is unpenalized.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        action_name = cfg.params.get("action_name", "thrust")
        action = _normalized_clamped_action(env, action_name)
        self._previous_action = torch.zeros_like(action)
        self._previous_previous_action = torch.zeros_like(action)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        index = slice(None) if env_ids is None else env_ids
        self._previous_action[index] = 0.0
        self._previous_previous_action[index] = 0.0
        self._initialized[index] = False

    def __call__(self, env: ManagerBasedRLEnv, action_name: str = "thrust") -> torch.Tensor:
        current_action = _normalized_clamped_action(env, action_name)
        if current_action.shape != self._previous_action.shape:
            raise ValueError(
                "Normalized action shape changed after reward initialization: "
                f"expected {tuple(self._previous_action.shape)}, got {tuple(current_action.shape)}."
            )

        acceleration = current_action - 2.0 * self._previous_action + self._previous_previous_action
        cost = torch.sum(acceleration.square(), dim=-1)
        cost = torch.where(self._initialized, torch.nan_to_num(cost), torch.zeros_like(cost))

        seeded_previous = torch.where(self._initialized.unsqueeze(-1), self._previous_action, current_action)
        self._previous_previous_action.copy_(seeded_previous)
        self._previous_action.copy_(current_action)
        self._initialized.fill_(True)
        return cost


class WaypointAdvanceImpulse(ManagerTermBase):
    """Emit one fixed impulse for every newly traversed indexed waypoint.

    The command index is monotonic within an episode and the completion bit
    accounts for the final waypoint, whose index cannot advance any farther.
    Comparing that count with the previous reward call supports the command
    term's bounded multi-knot advancement without rewarding the same knot
    twice. RewardManager multiplies terms by ``step_dt``, so dividing here
    preserves the configured reward weight as the per-waypoint bonus.

    This term deliberately depends only on the route command. It can therefore
    be shared by the drone-only and slung-load Direct-CTBR tasks.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._validate_command_name(cfg.params.get("command_name", "route"))
        self._previous_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Re-arm selected environments before their command indices reset."""
        index = slice(None) if env_ids is None else env_ids
        self._previous_count[index] = 0

    def __call__(self, env: ManagerBasedRLEnv, command_name: str = "route") -> torch.Tensor:
        """Return newly traversed waypoint count divided by ``step_dt``."""
        self._validate_command_name(command_name)
        term = env.command_manager.get_term(command_name)
        current_index = term.current_index
        completed = term.completed
        if current_index.shape != (env.num_envs,) or current_index.dtype != torch.long:
            raise ValueError(f"current_index must be a long tensor with shape ({env.num_envs},).")
        if completed.shape != (env.num_envs,) or completed.dtype != torch.bool:
            raise ValueError(f"completed must be a bool tensor with shape ({env.num_envs},).")

        count = current_index + completed.long()
        newly_traversed = (count - self._previous_count).clamp_min(0)
        self._previous_count.copy_(count)
        return newly_traversed.float() / env.step_dt

    @staticmethod
    def _validate_command_name(command_name: str) -> None:
        if not isinstance(command_name, str) or not command_name:
            raise ValueError("command_name must be a non-empty string.")


class RouteCompletionImpulse(ManagerTermBase):
    """Emit one fixed, optionally time-shaped impulse per route completion.

    Completion is detected from the command term's ``completed`` transition,
    not from a persistent terminal state or a cumulative metric. This makes the
    bonus safe when a successful non-timeout termination autoresets the
    environment immediately. RewardManager multiplies terms by ``step_dt``, so
    the returned impulse is divided by ``step_dt`` to preserve the configured
    reward weight as the per-completion bonus.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_completed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._seed_after_reset = torch.ones_like(self._previous_completed)
        self._validate_parameters(
            command_name=cfg.params.get("command_name", "route"),
            reference_completion_time=cfg.params.get("reference_completion_time"),
            early_completion_scale=cfg.params.get("early_completion_scale", 0.0),
            completion_time_metric_name=cfg.params.get("completion_time_metric_name", "waypoint_completion_time"),
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Re-arm selected environments without reading the not-yet-reset command."""
        index = slice(None) if env_ids is None else env_ids
        self._previous_completed[index] = False
        self._seed_after_reset[index] = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str = "route",
        reference_completion_time: float | None = None,
        early_completion_scale: float = 0.0,
        completion_time_metric_name: str = "waypoint_completion_time",
    ) -> torch.Tensor:
        """Return the transition impulse with an optional early-finish bonus.

        With ``reference_completion_time=T`` and ``early_completion_scale=s``,
        the event magnitude is ``1 + s * clamp((T - t_finish) / T, 0, 1)``.
        The RewardTerm weight therefore remains the base completion bonus.
        """
        self._validate_parameters(
            command_name=command_name,
            reference_completion_time=reference_completion_time,
            early_completion_scale=early_completion_scale,
            completion_time_metric_name=completion_time_metric_name,
        )
        term = env.command_manager.get_term(command_name)
        completed = term.completed
        if completed.shape != (env.num_envs,) or completed.dtype != torch.bool:
            raise ValueError(f"completed must be a bool tensor with shape ({env.num_envs},).")

        newly_completed = completed & ~self._previous_completed & ~self._seed_after_reset
        self._previous_completed.copy_(completed)
        self._seed_after_reset.fill_(False)
        magnitude = torch.ones(env.num_envs, device=env.device)
        if reference_completion_time is not None and early_completion_scale > 0.0:
            try:
                completion_time = term.metrics[completion_time_metric_name]
            except (AttributeError, KeyError) as error:
                raise ValueError(
                    f"Command term '{command_name}' has no completion-time metric '{completion_time_metric_name}'."
                ) from error
            if completion_time.shape != (env.num_envs,):
                raise ValueError(
                    f"completion-time metric must have shape ({env.num_envs},), got {tuple(completion_time.shape)}."
                )
            finite_time = torch.nan_to_num(
                completion_time,
                nan=reference_completion_time,
                posinf=reference_completion_time,
                neginf=reference_completion_time,
            ).clamp_min_(0.0)
            early_fraction = ((reference_completion_time - finite_time) / reference_completion_time).clamp_(0.0, 1.0)
            magnitude += early_completion_scale * early_fraction
        return newly_completed.float() * magnitude / env.step_dt

    @staticmethod
    def _validate_parameters(
        command_name: str,
        reference_completion_time: float | None,
        early_completion_scale: float,
        completion_time_metric_name: str,
    ) -> None:
        if not isinstance(command_name, str) or not command_name:
            raise ValueError("command_name must be a non-empty string.")
        if not isinstance(completion_time_metric_name, str) or not completion_time_metric_name:
            raise ValueError("completion_time_metric_name must be a non-empty string.")
        if reference_completion_time is not None and (
            isinstance(reference_completion_time, bool)
            or not isinstance(reference_completion_time, int | float)
            or not math.isfinite(reference_completion_time)
            or reference_completion_time <= 0.0
        ):
            raise ValueError("reference_completion_time must be finite and positive when provided.")
        if (
            isinstance(early_completion_scale, bool)
            or not isinstance(early_completion_scale, int | float)
            or not math.isfinite(early_completion_scale)
            or early_completion_scale < 0.0
        ):
            raise ValueError("early_completion_scale must be finite and nonnegative.")


def swing_safety_impulse(
    env: ManagerBasedRLEnv,
    threshold: float,
    angles: torch.Tensor | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    robot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    payload_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """FLARE unsafe-swing indicator divided by ``step_dt``."""
    if angles is None:
        angles = swing_angles(env, robot_cfg, payload_cfg, robot_offset, payload_offset)
    unsafe = torch.any(torch.abs(angles) > threshold, dim=-1)
    return unsafe.float() / env.step_dt


def crash_impulse(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Non-timeout termination indicator divided by ``step_dt``."""
    return env.termination_manager.terminated.float() / env.step_dt


def unsafe_termination_impulse(
    env: ManagerBasedRLEnv,
    unsafe_term_names: Sequence[str],
) -> torch.Tensor:
    """Explicit unsafe-termination union divided by ``step_dt``.

    Unlike :func:`crash_impulse`, this term does not treat every non-timeout
    terminal state as failure. Configure the names of all safety termination
    terms and deliberately omit successful terminal terms such as
    ``route_completed``. If success and an unsafe condition occur together,
    the unsafe event is still penalized.
    """
    names = (unsafe_term_names,) if isinstance(unsafe_term_names, str) else tuple(unsafe_term_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("unsafe_term_names must contain at least one non-empty string.")
    if len(set(names)) != len(names):
        raise ValueError("unsafe_term_names must not contain duplicates.")

    manager = env.termination_manager
    unsafe = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in names:
        try:
            value = manager.get_term(name)
        except (KeyError, ValueError) as error:
            raise ValueError(f"Unsafe termination term '{name}' was not found.") from error
        if value.shape != (env.num_envs,) or value.dtype != torch.bool:
            raise ValueError(f"Unsafe termination term '{name}' must be a bool tensor with shape ({env.num_envs},).")
        if hasattr(manager, "get_term_cfg") and getattr(manager.get_term_cfg(name), "time_out", False):
            raise ValueError(f"Unsafe termination term '{name}' must not be a time-out.")
        unsafe |= value
    return unsafe.float() / env.step_dt
