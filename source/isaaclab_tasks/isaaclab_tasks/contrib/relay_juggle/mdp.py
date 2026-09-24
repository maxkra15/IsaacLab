# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured ball-state terms for the rigid two-robot relay."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

from isaaclab_tasks.contrib.juggle.mdp.reset import JUGGLE_SPHERE_CENTER_OFFSET
from isaaclab_tasks.contrib.stack.mdp.kuka_allegro_reset import kuka_allegro_tool_pose

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_ball_in_kuka_hand(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    arm_joint_names: tuple[str, ...],
    kuka_base_position: tuple[float, float, float],
) -> None:
    """Place the ball in the calibrated KUKA cradle at an episode reset.

    Args:
        env: Manager-based relay environment.
        env_ids: Environment indices to reset.
        arm_joint_names: Ordered KUKA arm joint names.
        kuka_base_position: KUKA root translation in each environment [m].
    """
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()
    if ids.numel() == 0:
        return
    robot = env.scene["kuka"]
    ball = env.scene["ball"]
    arm_ids = robot.find_joints(list(arm_joint_names), preserve_order=True, as_proxy=True)[0].torch
    arm_positions = robot.data.default_joint_pos.torch[ids][:, arm_ids]
    ball_position_b, _ = kuka_allegro_tool_pose(arm_positions, JUGGLE_SPHERE_CENTER_OFFSET)
    ball_position_w = (
        ball_position_b + torch.as_tensor(kuka_base_position, device=env.device) + env.scene.env_origins[ids]
    )
    pose = ball.data.default_root_pose.torch[ids].clone()
    pose[:, :3] = ball_position_w
    velocity = torch.zeros((ids.numel(), 6), device=env.device, dtype=pose.dtype)
    ball.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    ball.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=ids)


def ball_position_local(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the ball center relative to environment origin [m]."""
    return env.scene["ball"].data.root_pos_w.torch - env.scene.env_origins


def ball_velocity_world(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return ball linear velocity in world coordinates [m/s]."""
    return env.scene["ball"].data.root_lin_vel_w.torch


def ball_relative_to_hand(
    env: ManagerBasedRLEnv,
    hand_cfg: SceneEntityCfg,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Return ball displacement from one hand frame [m].

    Args:
        env: Manager-based relay environment.
        hand_cfg: Articulation and one hand-link selection.
        offset: Hand-frame contact-point offset [m].
    """
    robot = env.scene[hand_cfg.name]
    hand_id = hand_cfg.body_ids[0]
    hand_pos = robot.data.body_link_pos_w.torch[:, hand_id]
    hand_quat = robot.data.body_link_quat_w.torch[:, hand_id]
    contact = hand_pos + quat_apply(hand_quat, torch.as_tensor(offset, device=env.device).expand_as(hand_pos))
    return env.scene["ball"].data.root_pos_w.torch - contact


def relay_target_direction(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the current receiver direction inferred from ball velocity."""
    velocity_x = env.scene["ball"].data.root_lin_vel_w.torch[:, :1]
    return torch.where(velocity_x < -0.15, -torch.ones_like(velocity_x), torch.ones_like(velocity_x))


def receiver_proximity(
    env: ManagerBasedRLEnv,
    kuka_hand_cfg: SceneEntityCfg,
    gr1_hand_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward proximity to the hand receiving the current flight."""
    kuka_distance = torch.linalg.vector_norm(
        ball_relative_to_hand(env, kuka_hand_cfg, JUGGLE_SPHERE_CENTER_OFFSET), dim=1
    )
    gr1_distance = torch.linalg.vector_norm(ball_relative_to_hand(env, gr1_hand_cfg), dim=1)
    receiver_distance = torch.where(relay_target_direction(env)[:, 0] > 0, gr1_distance, kuka_distance)
    return torch.exp(-4.0 * receiver_distance)


def outbound_flight_progress(
    env: ManagerBasedRLEnv,
    kuka_hand_cfg: SceneEntityCfg,
    gr1_hand_cfg: SceneEntityCfg,
    minimum_height: float = 0.70,
) -> torch.Tensor:
    """Reward forward ball flight only after a measured KUKA-hand separation.

    Args:
        env: Manager-based relay environment.
        kuka_hand_cfg: KUKA palm-link selection.
        gr1_hand_cfg: GR1T2 right-hand-link selection.
        minimum_height: Flight-height onset above the ground [m].
    """
    kuka_distance = torch.linalg.vector_norm(
        ball_relative_to_hand(env, kuka_hand_cfg, JUGGLE_SPHERE_CENTER_OFFSET), dim=1
    )
    gr1_distance = torch.linalg.vector_norm(ball_relative_to_hand(env, gr1_hand_cfg), dim=1)
    clearance = ((kuka_distance - 0.07) / 0.08).clamp(0.0, 1.0)
    height = ((ball_position_local(env)[:, 2] - minimum_height) / 0.25).clamp(0.0, 1.0)
    forward_speed = ball_velocity_world(env)[:, 0].clamp(0.0, 1.0)
    return clearance * height * forward_speed * torch.exp(-2.0 * gr1_distance)


def gr1_stable_ball_proximity(
    env: ManagerBasedRLEnv,
    gr1_hand_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward a nearby ball moving slowly relative to the GR1T2 hand.

    This is deliberately a kinematic retention proxy, not contact-verified success.
    """
    relative_position = ball_relative_to_hand(env, gr1_hand_cfg)
    hand_id = gr1_hand_cfg.body_ids[0]
    hand_velocity = env.scene[gr1_hand_cfg.name].data.body_link_lin_vel_w.torch[:, hand_id]
    relative_speed = torch.linalg.vector_norm(ball_velocity_world(env) - hand_velocity, dim=1)
    distance = torch.linalg.vector_norm(relative_position, dim=1)
    return torch.exp(-10.0 * distance - 3.0 * relative_speed)


def ball_dropped(env: ManagerBasedRLEnv, minimum_height: float = 0.10) -> torch.Tensor:
    """Terminate a relay after the ball center falls below the play volume [m]."""
    return ball_position_local(env)[:, 2] < minimum_height


def ball_out_of_bounds(env: ManagerBasedRLEnv, horizontal_limit: float = 2.0) -> torch.Tensor:
    """Terminate a relay when the ball exits the horizontal gallery [m]."""
    position = ball_position_local(env)
    return (position[:, :2].abs() > horizontal_limit).any(dim=1)


def nonfinite_relay_state(
    env: ManagerBasedRLEnv,
    kuka_cfg: SceneEntityCfg,
    gr1_cfg: SceneEntityCfg,
    maximum_jacobian: float = 100.0,
) -> torch.Tensor:
    """Terminate a world with invalid rigid state or an unphysical hand Jacobian.

    Args:
        env: Manager-based relay environment.
        kuka_cfg: KUKA arm joints and palm-link selection.
        gr1_cfg: GR1T2 arm joints and right-hand-link selection.
        maximum_jacobian: Largest acceptable linear/angular Jacobian entry [m/rad or 1].
    """
    fields: dict[str, tuple[torch.Tensor, float]] = {}
    ball = env.scene["ball"]
    ball_pose = ball.data.root_pose_w.torch
    fields["ball_pose"] = (torch.cat((ball_pose[:, :3] - env.scene.env_origins, ball_pose[:, 3:]), dim=1), 100.0)
    fields["ball_velocity"] = (ball.data.root_vel_w.torch, 100.0)
    for robot_cfg in (kuka_cfg, gr1_cfg):
        robot = env.scene[robot_cfg.name]
        fields[f"{robot_cfg.name}_joint_pos"] = (robot.data.joint_pos.torch, 100.0)
        fields[f"{robot_cfg.name}_joint_vel"] = (robot.data.joint_vel.torch, 100.0)
        jacobian = robot.data.body_link_jacobian_w.torch[:, robot_cfg.body_ids[0] - 1, :, robot_cfg.joint_ids]
        fields[f"{robot_cfg.name}_jacobian"] = (jacobian, maximum_jacobian)

    field_valid: dict[str, torch.Tensor] = {}
    for name, (value, maximum_abs) in fields.items():
        field_valid[name] = (torch.isfinite(value) & (value.abs() < maximum_abs)).flatten(1).all(dim=1)
    invalid = ~torch.stack(tuple(field_valid.values())).all(dim=0)

    # The demo opts into failure snapshots; PPO training incurs no GPU-to-CPU sync here.
    if getattr(env.cfg, "record_relay_invalid_details", False):
        details: dict[int, dict[str, dict[str, float | int | str]]] = {}
        for env_idx in invalid.nonzero(as_tuple=False).flatten().tolist():
            details[env_idx] = {}
            for name, (value, maximum_abs) in fields.items():
                if not bool(field_valid[name][env_idx]):
                    sample = value[env_idx]
                    finite = torch.isfinite(sample)
                    finite_magnitude = torch.where(finite, sample.abs(), torch.zeros_like(sample))
                    largest_index = int(finite_magnitude.flatten().argmax().item())
                    details[env_idx][name] = {
                        "nonfinite_count": int((~finite).sum().item()),
                        "max_finite_abs": float(finite_magnitude.amax().item()),
                        "max_abs_flat_index": largest_index,
                        "limit": maximum_abs,
                    }
                    if name.endswith("_joint_pos") or name.endswith("_joint_vel"):
                        robot_name = name.split("_joint_")[0]
                        robot = env.scene[robot_name]
                        details[env_idx][name]["joint_name"] = robot.joint_names[largest_index]
                        if name.endswith("_joint_vel"):
                            details[env_idx][name]["solver_velocity_limit"] = float(
                                robot.data.joint_vel_limits.torch[env_idx, largest_index].item()
                            )
                            details[env_idx][name]["soft_velocity_limit"] = float(
                                robot.data.soft_joint_vel_limits.torch[env_idx, largest_index].item()
                            )
        env._relay_invalid_state_details = details
    return invalid
