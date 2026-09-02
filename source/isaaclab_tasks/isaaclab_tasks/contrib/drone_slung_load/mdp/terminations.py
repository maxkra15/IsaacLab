# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Finite-state and workspace termination guards for the slung-load task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .bodies import link_com_vel_w, link_pos_e, link_pose_w
from .observations import cable_integrity_errors

if TYPE_CHECKING:
    from isaaclab.assets import CableObject
    from isaaclab.envs import ManagerBasedRLEnv


def out_of_workspace(
    env: ManagerBasedRLEnv,
    x_bound: float,
    y_bound: float,
    z_max: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when a rigid body leaves the environment-relative workspace [m]."""
    pos = link_pos_e(env, asset_cfg)
    return (pos[:, 0].abs() > x_bound) | (pos[:, 1].abs() > y_bound) | (pos[:, 2] > z_max)


def active_waypoint_error_out_of_bounds(
    env: ManagerBasedRLEnv,
    x_bound: float,
    y_bound: float,
    z_bound: float,
    command_name: str = "route",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when a body exceeds the active-waypoint error bounds [m].

    The bounds apply independently to the absolute XYZ error from the active
    waypoint. Consequently, translating a route within the environment frame
    does not change which relative tracking states are valid.
    """
    if any(not math.isfinite(value) or value <= 0.0 for value in (x_bound, y_bound, z_bound)):
        raise ValueError("Active-waypoint error bounds must be finite and positive.")
    active_waypoint = env.command_manager.get_command(command_name)[:, :3]
    error = link_pos_e(env, asset_cfg) - active_waypoint
    return (error[:, 0].abs() > x_bound) | (error[:, 1].abs() > y_bound) | (error[:, 2].abs() > z_bound)


def path_corridor_violation(
    env: ManagerBasedRLEnv,
    maximum_distance: float,
    command_name: str = "route",
) -> torch.Tensor:
    """Terminate when drone-to-spline cross-track distance exceeds a corridor [m]."""
    if not math.isfinite(maximum_distance) or maximum_distance <= 0.0:
        raise ValueError("maximum_distance must be finite and positive.")
    distance = env.command_manager.get_term(command_name).path_cross_track_distance
    if distance.shape == (env.num_envs, 1):
        distance = distance[:, 0]
    if distance.shape != (env.num_envs,):
        raise ValueError(f"path_cross_track_distance must have shape ({env.num_envs},), got {tuple(distance.shape)}.")
    return ~torch.isfinite(distance) | (distance > maximum_distance)


def route_completed(env: ManagerBasedRLEnv, command_name: str = "route") -> torch.Tensor:
    """Terminate successfully after traversing the finite indexed route."""
    completed = env.command_manager.get_term(command_name).completed
    if completed.shape != (env.num_envs,):
        raise ValueError(f"completed must have shape ({env.num_envs},), got {tuple(completed.shape)}.")
    return completed


def illegal_link_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when a maximal-coordinate rigid-body pose or velocity is non-finite."""
    pose = link_pose_w(env.scene[asset_cfg.name])
    velocity = link_com_vel_w(env.scene[asset_cfg.name])
    return ~(torch.isfinite(pose).all(dim=-1) & torch.isfinite(velocity).all(dim=-1))


def illegal_cable_state(
    env: ManagerBasedRLEnv,
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    max_linear_speed: float = 100.0,
    max_angular_speed: float = 1000.0,
    max_quaternion_norm_error: float = 0.01,
) -> torch.Tensor:
    """Terminate on non-finite or numerically exploded AVBD cable state."""
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (max_linear_speed, max_angular_speed, max_quaternion_norm_error)
    ):
        raise ValueError("Cable state limits must be finite and positive.")
    cable: CableObject = env.scene[cable_cfg.name]
    pose = cable.data.segment_pose_w.torch
    velocity = cable.data.segment_velocity_w.torch
    finite = torch.isfinite(pose).all(dim=(-1, -2)) & torch.isfinite(velocity).all(dim=(-1, -2))
    quaternion_norm = torch.linalg.vector_norm(torch.nan_to_num(pose[..., 3:7]), dim=-1)
    quaternion_valid = (quaternion_norm - 1.0).abs().amax(dim=-1) <= max_quaternion_norm_error
    linear_speed = torch.linalg.vector_norm(torch.nan_to_num(velocity[..., :3]), dim=-1).amax(dim=-1)
    angular_speed = torch.linalg.vector_norm(torch.nan_to_num(velocity[..., 3:6]), dim=-1).amax(dim=-1)
    bounded = (linear_speed <= max_linear_speed) & (angular_speed <= max_angular_speed)
    return ~(finite & quaternion_valid & bounded)


def illegal_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate when a policy emits a non-finite normalized action."""
    return ~torch.isfinite(env.action_manager.action).all(dim=-1)


def cable_integrity_violation(
    env: ManagerBasedRLEnv,
    nominal_length: float,
    max_relative_separation: float,
    max_joint_error: float,
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    robot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    payload_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Terminate when the string stretches or detaches beyond failure limits."""
    if any(not math.isfinite(value) or value <= 0.0 for value in (nominal_length, max_relative_separation)):
        raise ValueError("nominal_length and max_relative_separation must be finite and positive.")
    if not math.isfinite(max_joint_error) or max_joint_error < 0.0:
        raise ValueError("max_joint_error must be finite and nonnegative.")
    relative_separation, joint_error = cable_integrity_errors(
        env,
        cable_cfg,
        robot_cfg,
        payload_cfg,
        robot_offset,
        payload_offset,
        nominal_length,
    )
    finite = torch.isfinite(relative_separation) & torch.isfinite(joint_error)
    within_limits = (relative_separation <= max_relative_separation) & (joint_error <= max_joint_error)
    return ~(finite & within_limits)
