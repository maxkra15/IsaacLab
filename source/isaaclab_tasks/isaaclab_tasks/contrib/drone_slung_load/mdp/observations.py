# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FLARE actor observations and AVBD privileged critic state."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .bodies import link_ang_vel_b, link_lin_vel_w, link_pos_e, link_pose_w, link_quat_w
from .geometry import (
    attachment_kinematics,
    cable_constraint_errors,
    cable_features,
    rotation_matrix_flat,
    swing_features,
    transverse_velocity,
)

if TYPE_CHECKING:
    from isaaclab.assets import CableObject, RigidObject
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


# Element-wise normalization constants from FLARE Table I.
_RELATIVE_POSITION_SCALE = (5.0, 5.0, 1.0)
_DRONE_VELOCITY_SCALE = (10.0, 10.0, 3.0)
_SWING_ANGLE_SCALE = (1.5, 1.5)
# The later FLARE release multiplies cable-angle velocity by 0.1.
_SWING_ANGULAR_VELOCITY_SCALE = (10.0, 10.0)
# Privileged-only scales, chosen to keep AVBD features order one.
_ANGULAR_VELOCITY_SCALE = (5.0, 5.0, 5.0)
_TRANSVERSE_VELOCITY_SCALE = (1.0, 1.0, 1.0)
_KINEMATICS_EPS = 1.0e-6


def _scale_like(value: torch.Tensor, scale: tuple[float, ...] | float) -> torch.Tensor:
    """Divide a tensor by a finite scalar or last-axis scale and sanitize it."""
    divisor = torch.as_tensor(scale, device=value.device, dtype=value.dtype)
    return torch.nan_to_num(value / divisor)


def _positive_observation_scale(scale: tuple[float, ...] | float, dimension: int, name: str) -> tuple[float, ...]:
    """Validate a scalar or fixed-axis normalization scale."""
    values = (scale,) * dimension if isinstance(scale, int | float) else scale
    if len(values) != dimension or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"{name} must contain {dimension} finite positive values.")
    return values


def _body_com_pos_w(asset: RigidObject) -> torch.Tensor:
    # Derive the CoM from the live maximal-coordinate link pose. This remains
    # correct even when a backend exposes a root-pose convenience cache that is
    # updated later than ``body_link_pose_w``.
    link = link_pose_w(asset)
    com_b = asset.data.body_com_pos_b.torch
    com_b = com_b[:, 0] if com_b.ndim == 3 else com_b
    return link[:, :3] + quat_apply(link[:, 3:7], com_b)


def _attachment_states(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg,
    payload_cfg: SceneEntityCfg,
    robot_offset: tuple[float, float, float],
    payload_offset: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    robot: RigidObject = env.scene[robot_cfg.name]
    payload: RigidObject = env.scene[payload_cfg.name]
    robot_pose = link_pose_w(robot)
    payload_pose = link_pose_w(payload)
    robot_velocity = robot.data.body_com_vel_w.torch
    payload_velocity = payload.data.body_com_vel_w.torch
    robot_velocity = robot_velocity[:, 0] if robot_velocity.ndim == 3 else robot_velocity
    payload_velocity = payload_velocity[:, 0] if payload_velocity.ndim == 3 else payload_velocity
    robot_pos, robot_vel = attachment_kinematics(
        robot_pose,
        _body_com_pos_w(robot),
        robot_velocity,
        robot_offset,
    )
    payload_pos, payload_vel = attachment_kinematics(
        payload_pose,
        _body_com_pos_w(payload),
        payload_velocity,
        payload_offset,
    )
    attachment_b = quat_apply_inverse(robot_pose[:, 3:7], payload_pos - robot_pos)
    relative_velocity_b = quat_apply_inverse(robot_pose[:, 3:7], payload_vel - robot_vel)
    return robot_pos, payload_pos, attachment_b, relative_velocity_b, robot_pose[:, 3:7]


def world_lin_vel_normalized(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """World-frame drone velocity normalized by FLARE ``[10, 10, 3]`` [m/s]."""
    return _scale_like(link_lin_vel_w(env, asset_cfg), _DRONE_VELOCITY_SCALE)


def body_lin_vel_normalized(
    env: ManagerBasedEnv,
    speed_scale: float = 4.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Body-frame drone velocity normalized by one isotropic speed scale [m/s].

    A scalar divisor preserves the physical velocity direction after the
    world-to-body rotation. Direct-CTBR policies can therefore combine this
    observation directly with body-frame path tangents and errors.
    """
    if (
        isinstance(speed_scale, bool)
        or not isinstance(speed_scale, int | float)
        or not math.isfinite(speed_scale)
        or speed_scale <= 0.0
    ):
        raise ValueError("speed_scale must be finite and positive.")
    velocity_b = quat_apply_inverse(link_quat_w(env, asset_cfg), link_lin_vel_w(env, asset_cfg))
    return torch.nan_to_num(velocity_b / speed_scale)


def body_rotation_matrix(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Flattened row-major body-to-world rotation matrix."""
    return rotation_matrix_flat(link_quat_w(env, asset_cfg))


def body_ang_vel_normalized(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Body angular velocity normalized by the privileged 5 rad/s scale."""
    return _scale_like(link_ang_vel_b(env, asset_cfg), _ANGULAR_VELOCITY_SCALE)


def previous_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Previous normalized policy action after the physical action clamp."""
    action = torch.nan_to_num(env.action_manager.action, nan=0.0, posinf=1.0, neginf=-1.0)
    return action.clamp(-1.0, 1.0)


def payload_attachment_b(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    robot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    payload_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Payload-minus-drone attachment vector in the drone body frame [m]."""
    return _attachment_states(env, robot_cfg, payload_cfg, robot_offset, payload_offset)[2]


def payload_attachment_b_normalized(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    robot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    payload_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Body-frame attachment vector normalized by FLARE ``[5, 5, 1]`` [m]."""
    return _scale_like(
        payload_attachment_b(env, robot_cfg, payload_cfg, robot_offset, payload_offset),
        _RELATIVE_POSITION_SCALE,
    )


def swing_angles(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    robot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    payload_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """FLARE payload swing angles ``[phi, theta]`` [rad]."""
    vector = payload_attachment_b(env, robot_cfg, payload_cfg, robot_offset, payload_offset)
    return swing_features(vector)[0]


def swing_angles_normalized(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    robot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    payload_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """FLARE swing angles normalized element-wise by ``[1.5, 1.5]`` [rad]."""
    return _scale_like(
        swing_angles(env, robot_cfg, payload_cfg, robot_offset, payload_offset),
        _SWING_ANGLE_SCALE,
    )


def _swing_angular_velocity_from_kinematics(
    attachment_vector_b: torch.Tensor,
    relative_point_velocity_b: torch.Tensor,
    robot_angular_velocity_b: torch.Tensor,
) -> torch.Tensor:
    """Differentiate FLARE's two body-frame swing angles analytically."""
    finite = (
        torch.isfinite(attachment_vector_b).all(dim=-1)
        & torch.isfinite(relative_point_velocity_b).all(dim=-1)
        & torch.isfinite(robot_angular_velocity_b).all(dim=-1)
    )
    vector = torch.nan_to_num(attachment_vector_b)
    relative_velocity = torch.nan_to_num(relative_point_velocity_b)
    angular_velocity = torch.nan_to_num(robot_angular_velocity_b)

    # The point-velocity difference is expressed in the instantaneous body
    # frame. Subtract frame rotation to obtain the body-frame vector derivative.
    vector_dot = relative_velocity - torch.cross(angular_velocity, vector, dim=-1)
    x, y, z = vector.unbind(dim=-1)
    x_dot, y_dot, z_dot = vector_dot.unbind(dim=-1)
    phi_denominator = y.square() + z.square()
    theta_denominator = x.square() + z.square()
    eps_squared = _KINEMATICS_EPS**2
    phi_dot = torch.where(
        phi_denominator > eps_squared,
        (-z * y_dot + y * z_dot) / phi_denominator.clamp_min(eps_squared),
        torch.zeros_like(phi_denominator),
    )
    theta_dot = torch.where(
        theta_denominator > eps_squared,
        (-z * x_dot + x * z_dot) / theta_denominator.clamp_min(eps_squared),
        torch.zeros_like(theta_denominator),
    )
    rate = torch.stack((phi_dot, theta_dot), dim=-1)
    valid_vector = torch.sum(vector.square(), dim=-1) > eps_squared
    return torch.where((finite & valid_vector).unsqueeze(-1), torch.nan_to_num(rate), torch.zeros_like(rate))


def swing_angular_velocity(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    robot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    payload_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Analytic body-frame swing-angle velocity ``[phi_dot, theta_dot]`` [rad/s].

    The derivative uses live attachment-point velocities and includes the
    rotating-drone-frame transport term. It therefore has no observation
    history, delay, angle-wrap discontinuity, or reset transient.
    """
    _, _, vector_b, relative_velocity_b, _ = _attachment_states(
        env, robot_cfg, payload_cfg, robot_offset, payload_offset
    )
    angular_velocity_b = link_ang_vel_b(env, robot_cfg)
    return _swing_angular_velocity_from_kinematics(vector_b, relative_velocity_b, angular_velocity_b)


def swing_angular_velocity_normalized(
    env: ManagerBasedEnv,
    scale: tuple[float, float] | float = _SWING_ANGULAR_VELOCITY_SCALE,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    robot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    payload_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Swing-angle velocity divided by a configurable positive scale [rad/s].

    The default 10 rad/s scale matches the 0.1 multiplier in the later FLARE
    release. Passing a scalar applies the same normalization to both angles.
    """
    scale_values = (scale, scale) if isinstance(scale, int | float) else scale
    if len(scale_values) != 2 or any(not math.isfinite(value) or value <= 0.0 for value in scale_values):
        raise ValueError("scale must contain two finite positive angular-velocity scales.")
    return _scale_like(
        swing_angular_velocity(env, robot_cfg, payload_cfg, robot_offset, payload_offset),
        scale_values,
    )


def total_swing_angle(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    robot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    payload_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Total payload deflection from drone body -Z [rad]."""
    vector = payload_attachment_b(env, robot_cfg, payload_cfg, robot_offset, payload_offset)
    return swing_features(vector)[1]


def total_swing_angle_normalized(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    robot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    payload_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Total swing angle normalized by 1.5 rad."""
    return _scale_like(
        total_swing_angle(env, robot_cfg, payload_cfg, robot_offset, payload_offset),
        1.5,
    )


def payload_transverse_velocity_b(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    robot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    payload_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Relative attachment velocity transverse to the cable, normalized by 1 m/s."""
    _, _, vector, relative_velocity, _ = _attachment_states(env, robot_cfg, payload_cfg, robot_offset, payload_offset)
    return _scale_like(transverse_velocity(relative_velocity, vector), _TRANSVERSE_VELOCITY_SCALE)


def _cable_observation_features(
    env: ManagerBasedEnv,
    cable_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    payload_cfg: SceneEntityCfg,
    robot_offset: tuple[float, float, float],
    payload_offset: tuple[float, float, float],
    nominal_length: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    cable: CableObject = env.scene[cable_cfg.name]
    robot_pos, payload_pos, _, _, robot_quat = _attachment_states(
        env, robot_cfg, payload_cfg, robot_offset, payload_offset
    )
    return cable_features(
        cable.data.segment_pose_w.torch,
        robot_pos,
        payload_pos,
        robot_quat,
        nominal_length,
    )


def upper_cable_tangent_b(
    env: ManagerBasedEnv,
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    robot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    payload_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    nominal_length: float = 0.50,
) -> torch.Tensor:
    """Unit upper-cable tangent in the drone body frame."""
    return _cable_observation_features(
        env, cable_cfg, robot_cfg, payload_cfg, robot_offset, payload_offset, nominal_length
    )[0]


def cable_relative_separation(
    env: ManagerBasedEnv,
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    robot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    payload_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    nominal_length: float = 0.50,
) -> torch.Tensor:
    """Total AVBD joint/attachment gap divided by nominal cable length."""
    return _cable_observation_features(
        env, cable_cfg, robot_cfg, payload_cfg, robot_offset, payload_offset, nominal_length
    )[1]


def cable_integrity_errors(
    env: ManagerBasedEnv,
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    robot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    payload_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    nominal_length: float = 0.50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw relative cable separation and maximum joint endpoint gap."""
    cable: CableObject = env.scene[cable_cfg.name]
    robot_pos, payload_pos, _, _, _ = _attachment_states(env, robot_cfg, payload_cfg, robot_offset, payload_offset)
    return cable_constraint_errors(cable.data.segment_pose_w.torch, robot_pos, payload_pos, nominal_length)


def cable_joint_error(
    env: ManagerBasedEnv,
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    robot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    payload_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    nominal_length: float = 0.50,
) -> torch.Tensor:
    """Maximum external-attachment or internal-joint endpoint gap [m]."""
    error = cable_integrity_errors(
        env, cable_cfg, robot_cfg, payload_cfg, robot_offset, payload_offset, nominal_length
    )[1]
    return torch.nan_to_num(error).unsqueeze(-1)


def payload_lin_vel_w_normalized(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
) -> torch.Tensor:
    """World-frame payload velocity normalized by FLARE's velocity scale [m/s]."""
    return _scale_like(link_lin_vel_w(env, asset_cfg), _DRONE_VELOCITY_SCALE)


def waypoint_offsets_normalized(
    env: ManagerBasedRLEnv,
    command_name: str = "route",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Current and following waypoint offsets in the environment frame, FLARE-normalized."""
    waypoints_e = env.command_manager.get_command(command_name).reshape(env.num_envs, 2, 3)
    offsets_e = waypoints_e - link_pos_e(env, asset_cfg).unsqueeze(1)
    scale = torch.as_tensor(_RELATIVE_POSITION_SCALE, device=offsets_e.device, dtype=offsets_e.dtype)
    return torch.nan_to_num(offsets_e / scale).reshape(env.num_envs, 6)


def path_cross_track_error_b_normalized(
    env: ManagerBasedRLEnv,
    command_name: str = "route",
    scale: tuple[float, float, float] | float = 0.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Indexed spline cross-track error in the body frame, normalized by ``scale`` [m]."""
    scale_values = _positive_observation_scale(scale, 3, "scale")
    term = env.command_manager.get_term(command_name)
    error_b = quat_apply_inverse(link_quat_w(env, asset_cfg), term.path_cross_track_error_e)
    return _scale_like(error_b, scale_values)


def path_tangent_b(
    env: ManagerBasedRLEnv,
    command_name: str = "route",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Unit tangent of the indexed spline projection in the drone body frame."""
    term = env.command_manager.get_term(command_name)
    return torch.nan_to_num(quat_apply_inverse(link_quat_w(env, asset_cfg), term.path_tangent_e))


def path_curvature_b_normalized(
    env: ManagerBasedRLEnv,
    command_name: str = "route",
    scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Body-frame spline curvature binormal divided by ``scale`` [1/m]."""
    scale_value = _positive_observation_scale(scale, 1, "scale")[0]
    term = env.command_manager.get_term(command_name)
    curvature_b = quat_apply_inverse(link_quat_w(env, asset_cfg), term.path_curvature_e)
    return torch.nan_to_num(curvature_b / scale_value)


def path_preview_b_normalized(
    env: ManagerBasedRLEnv,
    command_name: str = "route",
    lookahead_distances: tuple[float, ...] = (0.75, 1.50),
    scale: tuple[float, float, float] | float = (2.0, 2.0, 1.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Body-frame offsets to forward arc-length samples, normalized and flattened.

    The default lookaheads correspond to 0.25 s and 0.50 s at the 3 m/s
    tracking target. Sampling stays on the current indexed route and clamps at
    the route endpoint.
    """
    scale_values = _positive_observation_scale(scale, 3, "scale")
    term = env.command_manager.get_term(command_name)
    preview_e, _, _ = term.path_preview_e(lookahead_distances)
    offset_e = preview_e - link_pos_e(env, asset_cfg).unsqueeze(1)
    quaternion = link_quat_w(env, asset_cfg).unsqueeze(1).expand(-1, offset_e.shape[1], -1)
    offset_b = quat_apply_inverse(quaternion, offset_e)
    divisor = torch.as_tensor(scale_values, device=offset_b.device, dtype=offset_b.dtype)
    return torch.nan_to_num(offset_b / divisor).reshape(env.num_envs, -1)


def path_progress_fraction(env: ManagerBasedRLEnv, command_name: str = "route") -> torch.Tensor:
    """Indexed projected arc length divided by total route length, shape ``(N, 1)``."""
    progress = env.command_manager.get_term(command_name).path_progress_fraction
    if progress.shape != (env.num_envs,):
        raise ValueError(f"path_progress_fraction must have shape ({env.num_envs},), got {tuple(progress.shape)}.")
    return torch.nan_to_num(progress).clamp_(0.0, 1.0).unsqueeze(-1)


def path_speed_features(
    env: ManagerBasedRLEnv,
    command_name: str = "route",
    speed_scale: float = 3.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return reference speed, signed tangent speed, and route progress.

    The three dimensionless values make a live speed curriculum observable to
    the actor. Signed tangent speed distinguishes forward tracking from
    backtracking, while progress fraction exposes finish-braking context.
    """
    if not math.isfinite(speed_scale) or speed_scale <= 0.0:
        raise ValueError("speed_scale must be finite and positive.")
    term = env.command_manager.get_term(command_name)
    reference_speed = term.path_speed_reference
    tangent = term.path_tangent_e
    progress_fraction = term.path_progress_fraction
    if reference_speed.shape != (env.num_envs,):
        raise ValueError(f"path_speed_reference must have shape ({env.num_envs},), got {tuple(reference_speed.shape)}.")
    if tangent.shape != (env.num_envs, 3):
        raise ValueError(f"path_tangent_e must have shape ({env.num_envs}, 3), got {tuple(tangent.shape)}.")
    if progress_fraction.shape != (env.num_envs,):
        raise ValueError(
            f"path_progress_fraction must have shape ({env.num_envs},), got {tuple(progress_fraction.shape)}."
        )
    velocity = link_lin_vel_w(env, asset_cfg)
    if velocity.shape != (env.num_envs, 3):
        raise ValueError(f"tracked velocity must have shape ({env.num_envs}, 3), got {tuple(velocity.shape)}.")

    finite = (
        torch.isfinite(reference_speed)
        & torch.isfinite(tangent).all(dim=-1)
        & torch.isfinite(progress_fraction)
        & torch.isfinite(velocity).all(dim=-1)
    )
    if hasattr(term, "path_state_valid"):
        finite &= term.path_state_valid
    tangent = torch.nan_to_num(tangent)
    tangent /= torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(torch.finfo(tangent.dtype).eps)
    signed_speed = torch.sum(torch.nan_to_num(velocity) * tangent, dim=-1)
    features = torch.stack(
        (
            torch.nan_to_num(reference_speed) / speed_scale,
            signed_speed / speed_scale,
            torch.nan_to_num(progress_fraction).clamp(0.0, 1.0),
        ),
        dim=-1,
    )
    return torch.where(finite.unsqueeze(-1), torch.nan_to_num(features), torch.zeros_like(features))


def path_tracking_features_b(
    env: ManagerBasedRLEnv,
    command_name: str = "route",
    lookahead_distances: tuple[float, ...] = (0.75, 1.50),
    cross_track_scale: tuple[float, float, float] | float = 0.20,
    preview_scale: tuple[float, float, float] | float = (2.0, 2.0, 1.0),
    curvature_scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compact body-frame spline state for high-speed precision tracking.

    Features are ordered as normalized cross-track error (3), unit tangent
    (3), normalized curvature binormal (3), and flattened normalized preview
    offsets (``3 * len(lookahead_distances)``). With the default two previews,
    the output has 15 elements. Every normalization argument is read on each
    call and can therefore be curriculum-controlled at runtime.
    """
    cross_track_scale_values = _positive_observation_scale(cross_track_scale, 3, "cross_track_scale")
    preview_scale_values = _positive_observation_scale(preview_scale, 3, "preview_scale")
    curvature_scale_value = _positive_observation_scale(curvature_scale, 1, "curvature_scale")[0]

    term = env.command_manager.get_term(command_name)
    preview_e, _, _ = term.path_preview_e(lookahead_distances)
    robot_pose = link_pose_w(env.scene[asset_cfg.name])
    preview_offset_e = preview_e - (robot_pose[:, :3] - env.scene.env_origins).unsqueeze(1)
    path_vector_e = torch.cat(
        (
            term.path_cross_track_error_e.unsqueeze(1),
            term.path_tangent_e.unsqueeze(1),
            term.path_curvature_e.unsqueeze(1),
            preview_offset_e,
        ),
        dim=1,
    )
    quaternion = robot_pose[:, 3:7].unsqueeze(1).expand(-1, path_vector_e.shape[1], -1)
    path_vector_b = torch.nan_to_num(quat_apply_inverse(quaternion, path_vector_e))
    cross_track_divisor = torch.as_tensor(
        cross_track_scale_values, device=path_vector_b.device, dtype=path_vector_b.dtype
    )
    preview_divisor = torch.as_tensor(preview_scale_values, device=path_vector_b.device, dtype=path_vector_b.dtype)
    return torch.cat(
        (
            path_vector_b[:, 0] / cross_track_divisor,
            path_vector_b[:, 1],
            path_vector_b[:, 2] / curvature_scale_value,
            (path_vector_b[:, 3:] / preview_divisor).reshape(env.num_envs, -1),
        ),
        dim=-1,
    )
