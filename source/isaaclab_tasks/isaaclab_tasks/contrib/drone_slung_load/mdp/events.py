# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset terms that hang the cable and payload under the drone."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, sample_uniform

from .geometry import straight_end_point, straight_segment_poses

if TYPE_CHECKING:
    from isaaclab.assets import CableObject, RigidObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg


def reset_drone_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the drone pose and velocity from independent uniform ranges.

    ``pose_range`` keys ``x``, ``y``, ``z`` are absolute environment-frame
    positions [m]. ``roll``, ``pitch``, ``yaw`` are orientation offsets [rad].
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    n = len(env_ids)
    device = env.device

    def _offset(key: str, default: tuple[float, float] = (0.0, 0.0)) -> torch.Tensor:
        low, high = pose_range.get(key, default)
        return sample_uniform(low, high, (n,), device=device)

    pos_e = torch.stack((_offset("x"), _offset("y"), _offset("z")), dim=-1)
    orientations = quat_from_euler_xyz(_offset("roll"), _offset("pitch"), _offset("yaw"))

    def _vel(key: str) -> torch.Tensor:
        low, high = velocity_range.get(key, (0.0, 0.0))
        return sample_uniform(low, high, (n,), device=device)

    velocities = torch.stack((_vel("x"), _vel("y"), _vel("z"), _vel("roll"), _vel("pitch"), _vel("yaw")), dim=-1)
    asset.write_root_pose_to_sim_index(
        root_pose=torch.cat((pos_e + env.scene.env_origins[env_ids], orientations), dim=-1),
        env_ids=env_ids,
    )
    asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)


def reset_drone_state_on_annulus(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    radius_range: tuple[float, float] = (4.3, 4.8),
    height: float = 0.0,
    roll_range: tuple[float, float] = (-0.05, 0.05),
    pitch_range: tuple[float, float] = (-0.05, 0.05),
    yaw: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset a stationary drone on a numerically local horizontal annulus.

    This reset pairs with the bounded periodic waypoint families. The command
    term runs after reset events and derives each route's scale and orientation
    from this post-event position, so no state needs to be shared between managers.

    Args:
        env: Manager-based environment.
        env_ids: Environment indices to reset.
        radius_range: Inclusive planar-radius sampling range [m].
        height: Absolute environment-frame height [m].
        roll_range: Inclusive roll-angle sampling range [rad].
        pitch_range: Inclusive pitch-angle sampling range [rad].
        yaw: Fixed yaw angle [rad].
        asset_cfg: Drone scene entity.
    """
    radius_low, radius_high = _validate_finite_range(radius_range, "radius_range", positive=True)
    roll_low, roll_high = _validate_finite_range(roll_range, "roll_range")
    pitch_low, pitch_high = _validate_finite_range(pitch_range, "pitch_range")
    if not math.isfinite(height):
        raise ValueError("height must be finite.")
    if not math.isfinite(yaw):
        raise ValueError("yaw must be finite.")

    asset: RigidObject = env.scene[asset_cfg.name]
    count = len(env_ids)
    radius = sample_uniform(radius_low, radius_high, (count,), device=env.device)
    azimuth = sample_uniform(-torch.pi, torch.pi, (count,), device=env.device)
    position_e = torch.stack(
        (radius * torch.cos(azimuth), radius * torch.sin(azimuth), torch.full_like(radius, height)), dim=-1
    )
    roll = sample_uniform(roll_low, roll_high, (count,), device=env.device)
    pitch = sample_uniform(pitch_low, pitch_high, (count,), device=env.device)
    orientation = quat_from_euler_xyz(roll, pitch, torch.full_like(radius, yaw))
    root_pose = torch.cat((position_e + env.scene.env_origins[env_ids], orientation), dim=-1)
    asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(count, 6, device=env.device, dtype=root_pose.dtype), env_ids=env_ids
    )


def _validate_finite_range(value: tuple[float, float], name: str, *, positive: bool = False) -> tuple[float, float]:
    """Return an ordered finite scalar range."""
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    low, high = value
    if not math.isfinite(low) or not math.isfinite(high) or low > high or (positive and low <= 0.0):
        qualifier = "positive, " if positive else ""
        raise ValueError(f"{name} must contain {qualifier}finite ordered values.")
    return low, high


class ResetSlungLoadEvent(ManagerTermBase):
    """Hang the cable and payload at their fixed nominal attachment geometry.

    The geometry is written to Newton state and to ``model.body_q`` so VBD starts
    from its authored cable rest length without a reset-time strain impulse.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: RigidObject = env.scene[cfg.params["robot_cfg"].name]
        self.cable: CableObject = env.scene[cfg.params["cable_cfg"].name]
        self.payload: RigidObject = env.scene[cfg.params["payload_cfg"].name]
        self._identity_quat = torch.zeros(env.num_envs, 4, device=env.device)
        self._identity_quat[:, 3] = 1.0

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        cable_length: float,
        robot_cfg: SceneEntityCfg,
        cable_cfg: SceneEntityCfg,
        payload_cfg: SceneEntityCfg,
        attach_offset_z: float,
        max_initial_swing: float = 0.0,
    ):
        del robot_cfg, cable_cfg, payload_cfg
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
        else:
            env_ids = env_ids.to(device=env.device, dtype=torch.int32)

        cable_length_t = torch.full((len(env_ids),), cable_length, device=env.device)

        drone_pos = self.robot.data.root_pos_w.torch[env_ids]
        drone_quat = self.robot.data.root_quat_w.torch[env_ids]
        offset = torch.zeros(len(env_ids), 3, device=env.device)
        offset[:, 2] = attach_offset_z
        attach_pos = drone_pos + quat_apply(drone_quat, offset)

        swing = sample_uniform(0.0, max_initial_swing, (len(env_ids),), device=env.device)
        azimuth = sample_uniform(-torch.pi, torch.pi, (len(env_ids),), device=env.device)
        direction = torch.stack(
            (
                torch.sin(swing) * torch.cos(azimuth),
                torch.sin(swing) * torch.sin(azimuth),
                -torch.cos(swing),
            ),
            dim=-1,
        )
        segment_pose = straight_segment_poses(attach_pos, cable_length_t, self.cable.num_segments, direction)
        zeros_vel = torch.zeros(len(env_ids), self.cable.num_segments, 6, device=env.device)
        self.cable.write_segment_pose_to_sim_index(segment_pose=segment_pose, env_ids=env_ids)
        self.cable.write_segment_velocity_to_sim_index(segment_velocity=zeros_vel, env_ids=env_ids)
        self._write_cable_rest_pose(segment_pose, env_ids)

        end_point = straight_end_point(attach_pos, cable_length_t, direction)
        payload_pose = torch.zeros(len(env_ids), 7, device=env.device)
        payload_pose[:, :3] = end_point
        payload_pose[:, 3:7] = self._identity_quat[env_ids]
        payload_vel = torch.zeros(len(env_ids), 6, device=env.device)
        self.payload.write_root_pose_to_sim_index(root_pose=payload_pose, env_ids=env_ids)
        self.payload.write_root_velocity_to_sim_index(root_velocity=payload_vel, env_ids=env_ids)

    def _write_cable_rest_pose(self, segment_pose: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Copy hanging poses into Newton's rest configuration when available."""
        try:
            import warp as wp
            from isaaclab_newton.assets.cable_object.kernels import set_segment_pose_to_sim_index
            from isaaclab_newton.physics import NewtonManager as SimulationManager
        except ImportError:
            return
        model = SimulationManager.get_model()
        rest = getattr(model, "body_q", None)
        if rest is None:
            return
        env_ids_wp = wp.from_torch(env_ids.contiguous(), dtype=wp.int32)
        wp.launch(
            set_segment_pose_to_sim_index,
            dim=(env_ids.shape[0], self.cable.num_segments),
            inputs=[
                segment_pose.contiguous(),
                env_ids_wp,
                self.cable.data._sim_bind_root_body_ids,
                self.cable.data._sim_bind_link_body_ids,
            ],
            outputs=[rest],
            device=self.cable.device,
        )
        from newton import ModelFlags

        SimulationManager.add_model_change(ModelFlags.BODY_PROPERTIES)
