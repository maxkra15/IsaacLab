# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Body-frame rotor actions for a single-rigid-body slung-load drone."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply_inverse

from ..system import ROTOR_ARM_LENGTH, ROTOR_HEIGHT, ROTOR_YAW_COEFFICIENT
from .bodies import link_com_vel_w, link_pose_w
from .controllers import (
    FlareController,
    _scale_flare_actions_unchecked,
    quadrotor_rotor_geometry,
    scale_flare_actions,
)

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedEnv


class CollectiveThrustBodyRateAction(ActionTerm):
    """Apply FLARE collective-thrust/body-rate commands through four drone rotors.

    Each nonnegative rotor thrust acts along body +Z at its body-frame rotor location.
    Its aerodynamic drag moment acts as a colocated yaw reaction torque. The wrench
    composer reduces those four contributions to the exactly equivalent wrench about
    the center of mass because the drone is modeled as one rigid body. No force or
    torque is written to the cable or payload; they move only through gravity,
    contacts, and the physical attachment constraints.
    """

    cfg: CollectiveThrustBodyRateActionCfg

    def __init__(self, cfg: CollectiveThrustBodyRateActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset: RigidObject = env.scene[cfg.asset_name]
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 4, device=self.device)

        # Rigid-object mass and action limits are constant for this task. Validate
        # them once, then keep the tensors on device for the per-step hot path.
        self._drone_mass_kg = self._drone_mass()
        scale_flare_actions(
            self._raw_actions,
            self._drone_mass_kg,
            gravity=cfg.gravity,
            max_thrust_to_weight=cfg.max_thrust_to_weight,
            max_body_rates=cfg.max_body_rates,
        )
        residual_body_rate_limits = (
            cfg.max_body_rates if cfg.residual_body_rate_limits is None else cfg.residual_body_rate_limits
        )
        scale_flare_actions(
            self._raw_actions,
            self._drone_mass_kg,
            gravity=cfg.gravity,
            max_thrust_to_weight=cfg.max_thrust_to_weight,
            max_body_rates=residual_body_rate_limits,
        )
        self._max_body_rate_tensor = torch.as_tensor(
            cfg.max_body_rates, device=self._processed_actions.device, dtype=self._processed_actions.dtype
        )
        self._residual_body_rate_tensor = torch.as_tensor(
            residual_body_rate_limits,
            device=self._processed_actions.device,
            dtype=self._processed_actions.dtype,
        )
        if torch.any(self._max_body_rate_tensor <= 0.0):
            raise ValueError("max_body_rates must contain only positive values.")
        if torch.any(self._residual_body_rate_tensor <= 0.0):
            raise ValueError("residual_body_rate_limits must contain only positive values.")
        if torch.any(self._residual_body_rate_tensor > self._max_body_rate_tensor):
            raise ValueError("residual_body_rate_limits must not exceed max_body_rates.")
        self._gravity = cfg.gravity
        self._half_max_thrust_to_weight = 0.5 * cfg.max_thrust_to_weight
        if not math.isfinite(cfg.attitude_hold_gain) or cfg.attitude_hold_gain < 0.0:
            raise ValueError("attitude_hold_gain must be finite and nonnegative.")
        if not math.isfinite(cfg.horizontal_velocity_damping_gain) or cfg.horizontal_velocity_damping_gain < 0.0:
            raise ValueError("horizontal_velocity_damping_gain must be finite and nonnegative.")
        if not math.isfinite(cfg.vertical_velocity_damping_gain) or cfg.vertical_velocity_damping_gain < 0.0:
            raise ValueError("vertical_velocity_damping_gain must be finite and nonnegative.")
        if cfg.path_velocity_command_name is not None and (
            not isinstance(cfg.path_velocity_command_name, str) or not cfg.path_velocity_command_name
        ):
            raise ValueError("path_velocity_command_name must be a nonempty string or None.")
        if not math.isfinite(cfg.path_velocity_cross_track_gain) or cfg.path_velocity_cross_track_gain < 0.0:
            raise ValueError("path_velocity_cross_track_gain must be finite and nonnegative.")
        if (
            not math.isfinite(cfg.path_velocity_maximum_cross_track_speed)
            or cfg.path_velocity_maximum_cross_track_speed < 0.0
        ):
            raise ValueError("path_velocity_maximum_cross_track_speed must be finite and nonnegative.")
        if cfg.path_velocity_cross_track_gain > 0.0 and cfg.path_velocity_maximum_cross_track_speed == 0.0:
            raise ValueError(
                "path_velocity_maximum_cross_track_speed must be positive when path_velocity_cross_track_gain is."
            )
        if (
            not math.isfinite(cfg.path_velocity_curvature_feedforward_gain)
            or cfg.path_velocity_curvature_feedforward_gain < 0.0
        ):
            raise ValueError("path_velocity_curvature_feedforward_gain must be finite and nonnegative.")
        if not math.isfinite(cfg.suspended_mass) or cfg.suspended_mass < 0.0:
            raise ValueError("suspended_mass must be finite and nonnegative.")
        if not isinstance(cfg.tilt_compensation, bool):
            raise ValueError("tilt_compensation must be a boolean.")
        if cfg.horizontal_velocity_damping_gain > 0.0 and cfg.attitude_hold_gain == 0.0:
            raise ValueError("horizontal_velocity_damping_gain requires a positive attitude_hold_gain.")
        if (
            not math.isfinite(cfg.maximum_velocity_hold_tilt)
            or cfg.maximum_velocity_hold_tilt <= 0.0
            or cfg.maximum_velocity_hold_tilt >= 0.5 * math.pi
        ):
            raise ValueError("maximum_velocity_hold_tilt must be finite and lie in (0, pi / 2).")
        if (
            not math.isfinite(cfg.maximum_tilt_compensation_angle)
            or cfg.maximum_tilt_compensation_angle <= 0.0
            or cfg.maximum_tilt_compensation_angle >= 0.5 * math.pi
        ):
            raise ValueError("maximum_tilt_compensation_angle must be finite and lie in (0, pi / 2).")
        if cfg.vertical_velocity_damping_gain > 0.0 and not cfg.tilt_compensation:
            raise ValueError("vertical_velocity_damping_gain requires tilt_compensation=True.")
        if cfg.path_velocity_command_name is None and (
            cfg.path_velocity_cross_track_gain > 0.0 or cfg.path_velocity_curvature_feedforward_gain > 0.0
        ):
            raise ValueError("path-velocity correction gains require path_velocity_command_name.")
        if cfg.path_velocity_curvature_feedforward_gain > 0.0 and cfg.attitude_hold_gain == 0.0:
            raise ValueError("path_velocity_curvature_feedforward_gain requires a positive attitude_hold_gain.")
        if cfg.path_velocity_curvature_feedforward_gain > 0.0 and not cfg.tilt_compensation:
            raise ValueError("path_velocity_curvature_feedforward_gain requires tilt_compensation=True.")
        if (
            (cfg.horizontal_velocity_damping_gain > 0.0 or cfg.path_velocity_curvature_feedforward_gain > 0.0)
            and cfg.tilt_compensation
            and cfg.maximum_tilt_compensation_angle < cfg.maximum_velocity_hold_tilt
        ):
            raise ValueError(
                "maximum_tilt_compensation_angle must cover maximum_velocity_hold_tilt when both loops are enabled."
            )
        self._attitude_hold_gain = cfg.attitude_hold_gain
        self._horizontal_velocity_damping_gain = cfg.horizontal_velocity_damping_gain
        self._vertical_velocity_damping_gain = cfg.vertical_velocity_damping_gain
        self._path_velocity_cross_track_gain = cfg.path_velocity_cross_track_gain
        self._path_velocity_maximum_cross_track_speed = cfg.path_velocity_maximum_cross_track_speed
        self._path_velocity_curvature_feedforward_gain = cfg.path_velocity_curvature_feedforward_gain
        self._tilt_compensation = cfg.tilt_compensation
        self._maximum_horizontal_acceleration = cfg.gravity * math.tan(cfg.maximum_velocity_hold_tilt)
        self._minimum_tilt_cosine = math.cos(cfg.maximum_tilt_compensation_angle)
        self._supported_mass_kg = self._drone_mass_kg + cfg.suspended_mass
        self._maximum_collective_thrust = self._drone_mass_kg * cfg.gravity * cfg.max_thrust_to_weight
        self._world_up = torch.zeros(self.num_envs, 3, device=self.device)
        self._world_up[:, 2] = 1.0
        self._desired_thrust_axis_w = self._world_up.clone()
        self._velocity_hold_scale = torch.ones(self.num_envs, 1, device=self.device)
        self._attitude_rate_correction = torch.zeros(self.num_envs, 2, device=self.device)
        self._tilt_cosine = torch.ones(self.num_envs, device=self.device)
        self._vertical_velocity = torch.zeros(self.num_envs, device=self.device)
        self._path_speed_reference = torch.zeros(self.num_envs, device=self.device)
        self._path_tangent_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._path_curvature_w = torch.zeros_like(self._path_tangent_w)
        self._desired_path_velocity_w = torch.zeros_like(self._path_tangent_w)
        self._desired_path_acceleration_w = torch.zeros_like(self._path_tangent_w)
        self._path_velocity_error_w = torch.zeros_like(self._path_tangent_w)
        self._path_scalar_buffer = torch.zeros(self.num_envs, device=self.device)
        self._path_vector_scale = torch.ones(self.num_envs, 1, device=self.device)

        self._path_velocity_command = None
        if cfg.path_velocity_command_name is not None:
            try:
                self._path_velocity_command = env.command_manager.get_term(cfg.path_velocity_command_name)
            except (AttributeError, KeyError) as error:
                raise ValueError(f"Unknown path_velocity_command_name {cfg.path_velocity_command_name!r}.") from error
            required_fields = ["path_speed_reference", "path_tangent_e"]
            if cfg.path_velocity_cross_track_gain > 0.0:
                required_fields.append("path_cross_track_error_e")
            if cfg.path_velocity_curvature_feedforward_gain > 0.0:
                required_fields.append("path_curvature_e")
            for field in required_fields:
                value = getattr(self._path_velocity_command, field, None)
                expected_shape = (self.num_envs,) if field == "path_speed_reference" else (self.num_envs, 3)
                if not isinstance(value, torch.Tensor) or value.shape != expected_shape:
                    actual_shape = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
                    raise ValueError(
                        f"Path command {field} must be a tensor with shape {expected_shape}, got {actual_shape}."
                    )

        rotor_positions_b, yaw_directions = quadrotor_rotor_geometry(
            arm_length=cfg.arm_length,
            rotor_z=cfg.rotor_z,
            device=self.device,
        )
        # WrenchComposer accepts one contribution per selected rigid body. Since all
        # rotors belong to the same body, retain one contiguous buffer per rotor and
        # accumulate four colocated contributions below.
        self._rotor_positions_b = rotor_positions_b[:, None, None, :].expand(-1, self.num_envs, -1, -1).clone()
        self._rotor_yaw_coefficients = yaw_directions * cfg.yaw_coeff
        self._rotor_forces_b = torch.zeros(4, self.num_envs, 1, 3, device=self.device)
        self._rotor_torques_b = torch.zeros_like(self._rotor_forces_b)
        self._needs_motor_init = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._controller = FlareController(
            self.num_envs,
            self.device,
            dt=cfg.dt,
            rate_gains=cfg.rate_gains,
            rate_integral_gains=cfg.rate_integral_gains,
            rate_derivative_gains=cfg.rate_derivative_gains,
            rate_integral_error_limits=cfg.rate_integral_error_limits,
            rate_derivative_cutoff_hz=cfg.rate_derivative_cutoff_hz,
            torque_limits=cfg.torque_limits,
            arm_length=cfg.arm_length,
            yaw_coeff=cfg.yaw_coeff,
            rotor_thrust_limits=cfg.rotor_thrust_limits,
            allocation_mode=cfg.allocation_mode,
            tau_up=cfg.tau_up,
            tau_down=cfg.tau_down,
        )

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def motor_thrusts(self) -> torch.Tensor:
        """Lagged rotor thrust state [N], shape ``(N, 4)``."""
        return self._controller.motor_thrusts

    def _drone_mass(self) -> torch.Tensor:
        """Live drone mass [kg], shape ``(N,)``."""
        return self._asset.data.body_mass.torch.reshape(self.num_envs, -1).sum(dim=-1)

    def _body_rate_b(self) -> torch.Tensor:
        """Drone angular velocity from Newton maximal-coordinate body state [rad/s]."""
        pose_w = link_pose_w(self._asset)
        angular_velocity_w = link_com_vel_w(self._asset)[:, 3:6]
        return quat_apply_inverse(pose_w[:, 3:7], angular_velocity_w)

    def _update_path_velocity_prior(self, linear_velocity_w: torch.Tensor | None) -> None:
        """Build the live 3D path-velocity acceleration prior in world coordinates."""
        term = self._path_velocity_command
        assert term is not None

        # The command term is the single owner of the curvature/braking speed
        # profile. Read it live so curriculum changes are immediately reflected
        # by the controller as well as the actor observations and reward.
        torch.nan_to_num(
            term.path_speed_reference,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
            out=self._path_speed_reference,
        )
        self._path_speed_reference.clamp_min_(0.0)
        torch.nan_to_num(term.path_tangent_e, nan=0.0, posinf=0.0, neginf=0.0, out=self._path_tangent_w)
        torch.linalg.vector_norm(self._path_tangent_w, dim=-1, keepdim=True, out=self._path_vector_scale)
        self._path_vector_scale.clamp_min_(torch.finfo(self._path_vector_scale.dtype).eps)
        self._path_tangent_w.div_(self._path_vector_scale)
        self._desired_path_velocity_w.copy_(self._path_tangent_w)
        self._desired_path_velocity_w.mul_(self._path_speed_reference.unsqueeze(-1))

        if self._path_velocity_cross_track_gain > 0.0:
            # cross_track_error is projection-to-vehicle, hence the negative
            # sign drives toward the indexed active branch even at crossings.
            torch.nan_to_num(
                term.path_cross_track_error_e,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
                out=self._desired_path_acceleration_w,
            )
            self._desired_path_acceleration_w.mul_(-self._path_velocity_cross_track_gain)
            torch.linalg.vector_norm(
                self._desired_path_acceleration_w,
                dim=-1,
                keepdim=True,
                out=self._path_vector_scale,
            )
            self._path_vector_scale.clamp_min_(torch.finfo(self._path_vector_scale.dtype).eps)
            self._path_vector_scale.reciprocal_().mul_(self._path_velocity_maximum_cross_track_speed).clamp_max_(1.0)
            self._desired_path_acceleration_w.mul_(self._path_vector_scale)
            self._desired_path_velocity_w.add_(self._desired_path_acceleration_w)

            # Enforce a second, explicit bound on the combined velocity. The
            # triangle-inequality value retains the commanded tangential speed
            # while preventing malformed geometry from growing the reference.
            torch.linalg.vector_norm(
                self._desired_path_velocity_w,
                dim=-1,
                keepdim=True,
                out=self._path_vector_scale,
            )
            self._path_vector_scale.clamp_min_(torch.finfo(self._path_vector_scale.dtype).eps)
            self._path_scalar_buffer.copy_(self._path_speed_reference).add_(
                self._path_velocity_maximum_cross_track_speed
            )
            self._path_vector_scale.reciprocal_().mul_(self._path_scalar_buffer.unsqueeze(-1)).clamp_max_(1.0)
            self._desired_path_velocity_w.mul_(self._path_vector_scale)

        if self._path_velocity_curvature_feedforward_gain > 0.0:
            # For curvature binormal k_b = t x dt/ds, cross(k_b, t) points
            # toward the instantaneous center of curvature. It therefore gives
            # the acceleration v^2 k without waiting for velocity error.
            torch.nan_to_num(
                term.path_curvature_e,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
                out=self._path_curvature_w,
            )
            torch.linalg.cross(
                self._path_curvature_w,
                self._path_tangent_w,
                dim=-1,
                out=self._desired_path_acceleration_w,
            )
            torch.mul(self._path_speed_reference, self._path_speed_reference, out=self._path_scalar_buffer)
            self._desired_path_acceleration_w.mul_(self._path_scalar_buffer.unsqueeze(-1))
            self._desired_path_acceleration_w.mul_(self._path_velocity_curvature_feedforward_gain)
        else:
            self._desired_path_acceleration_w.zero_()

        if self._horizontal_velocity_damping_gain > 0.0 or self._vertical_velocity_damping_gain > 0.0:
            assert linear_velocity_w is not None
            self._path_velocity_error_w.copy_(self._desired_path_velocity_w).sub_(linear_velocity_w)
            torch.nan_to_num(
                self._path_velocity_error_w,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
                out=self._path_velocity_error_w,
            )
        if self._horizontal_velocity_damping_gain > 0.0:
            self._desired_path_acceleration_w[:, :2].add_(
                self._path_velocity_error_w[:, :2],
                alpha=self._horizontal_velocity_damping_gain,
            )
        if self._vertical_velocity_damping_gain > 0.0:
            self._vertical_velocity.copy_(self._path_velocity_error_w[:, 2])
            self._desired_path_acceleration_w[:, 2].add_(
                self._vertical_velocity,
                alpha=self._vertical_velocity_damping_gain,
            )
        torch.nan_to_num(
            self._desired_path_acceleration_w,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
            out=self._desired_path_acceleration_w,
        )

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        _scale_flare_actions_unchecked(
            actions,
            self._drone_mass_kg,
            gravity=self._gravity,
            half_max_thrust_to_weight=self._half_max_thrust_to_weight,
            rate_limits=self._residual_body_rate_tensor,
            out=self._processed_actions,
        )
        needs_pose = self._attitude_hold_gain > 0.0 or self._tilt_compensation
        needs_velocity = self._horizontal_velocity_damping_gain > 0.0 or self._vertical_velocity_damping_gain > 0.0
        pose_w = link_pose_w(self._asset) if needs_pose else None
        linear_velocity_w = link_com_vel_w(self._asset)[:, :3] if needs_velocity else None

        path_velocity_enabled = self._path_velocity_command is not None
        if path_velocity_enabled:
            self._update_path_velocity_prior(linear_velocity_w)

        if path_velocity_enabled and (
            self._vertical_velocity_damping_gain > 0.0 or self._path_velocity_curvature_feedforward_gain > 0.0
        ):
            # Track the 3D path reference, including vertical tangent and
            # centripetal feedforward, before bounded tilt compensation.
            self._processed_actions[:, 0].addcmul_(
                self._supported_mass_kg,
                self._desired_path_acceleration_w[:, 2],
            )
        elif self._vertical_velocity_damping_gain > 0.0:
            # Add the force required to damp the supported system's world-frame
            # vertical velocity: delta_T = -m_total K_vz v_z. Non-finite state
            # disables only the feedback term and cannot poison the action.
            assert linear_velocity_w is not None
            torch.nan_to_num(
                linear_velocity_w[:, 2],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
                out=self._vertical_velocity,
            )
            self._processed_actions[:, 0].addcmul_(
                self._supported_mass_kg,
                self._vertical_velocity,
                value=-self._vertical_velocity_damping_gain,
            )

        if self._tilt_compensation:
            # Collective thrust acts along body +Z. Divide the complete desired
            # world-vertical force, including velocity feedback, by its vertical
            # projection. Bound the denominator to avoid a singular correction
            # after an invalid or inverted pose.
            assert pose_w is not None
            quaternion_xyzw = pose_w[:, 3:7]
            torch.mul(quaternion_xyzw[:, 0], quaternion_xyzw[:, 0], out=self._tilt_cosine)
            self._tilt_cosine.addcmul_(quaternion_xyzw[:, 1], quaternion_xyzw[:, 1])
            self._tilt_cosine.mul_(-2.0).add_(1.0)
            torch.nan_to_num(
                self._tilt_cosine,
                nan=1.0,
                posinf=1.0,
                neginf=self._minimum_tilt_cosine,
                out=self._tilt_cosine,
            )
            self._tilt_cosine.clamp_(self._minimum_tilt_cosine, 1.0)
            self._processed_actions[:, 0].div_(self._tilt_cosine)

        if self._tilt_compensation or self._vertical_velocity_damping_gain > 0.0:
            self._processed_actions[:, 0].clamp_min_(0.0)
            torch.minimum(
                self._processed_actions[:, 0],
                self._maximum_collective_thrust,
                out=self._processed_actions[:, 0],
            )

        if self._attitude_hold_gain > 0.0:
            # Standard geometric cascaded-flight-control prior: interpret the
            # policy's roll/pitch rates as residuals around a desired thrust
            # axis. With velocity damping disabled that axis is world-up. The
            # optional path branch tracks a bounded 3D velocity reference and
            # includes its vertical acceleration in the desired thrust axis.
            assert pose_w is not None
            desired_thrust_axis_w = self._world_up
            if path_velocity_enabled and (
                self._horizontal_velocity_damping_gain > 0.0 or self._path_velocity_curvature_feedforward_gain > 0.0
            ):
                self._desired_thrust_axis_w[:, :2].copy_(self._desired_path_acceleration_w[:, :2])
                self._desired_thrust_axis_w[:, 2].copy_(self._desired_path_acceleration_w[:, 2]).add_(self._gravity)
                self._desired_thrust_axis_w[:, 2].clamp_min_(torch.finfo(self._desired_thrust_axis_w.dtype).eps)
                torch.linalg.vector_norm(
                    self._desired_thrust_axis_w[:, :2],
                    dim=-1,
                    keepdim=True,
                    out=self._velocity_hold_scale,
                )
                self._velocity_hold_scale.clamp_min_(torch.finfo(self._velocity_hold_scale.dtype).eps)
                self._velocity_hold_scale.reciprocal_().mul_(self._desired_thrust_axis_w[:, 2:3])
                self._velocity_hold_scale.mul_(self._maximum_horizontal_acceleration / self._gravity).clamp_max_(1.0)
                self._desired_thrust_axis_w[:, :2].mul_(self._velocity_hold_scale)
                torch.linalg.vector_norm(
                    self._desired_thrust_axis_w,
                    dim=-1,
                    keepdim=True,
                    out=self._velocity_hold_scale,
                )
                self._desired_thrust_axis_w.div_(self._velocity_hold_scale)
                desired_thrust_axis_w = self._desired_thrust_axis_w
            elif self._horizontal_velocity_damping_gain > 0.0:
                assert linear_velocity_w is not None
                torch.nan_to_num(linear_velocity_w[:, :2], out=self._desired_thrust_axis_w[:, :2])
                self._desired_thrust_axis_w[:, :2].mul_(-self._horizontal_velocity_damping_gain)
                torch.linalg.vector_norm(
                    self._desired_thrust_axis_w[:, :2],
                    dim=-1,
                    keepdim=True,
                    out=self._velocity_hold_scale,
                )
                self._velocity_hold_scale.clamp_min_(torch.finfo(self._velocity_hold_scale.dtype).eps)
                self._velocity_hold_scale.reciprocal_().mul_(self._maximum_horizontal_acceleration).clamp_max_(1.0)
                self._desired_thrust_axis_w[:, :2].mul_(self._velocity_hold_scale)
                self._desired_thrust_axis_w[:, 2].fill_(self._gravity)
                torch.linalg.vector_norm(
                    self._desired_thrust_axis_w,
                    dim=-1,
                    keepdim=True,
                    out=self._velocity_hold_scale,
                )
                self._desired_thrust_axis_w.div_(self._velocity_hold_scale)
                desired_thrust_axis_w = self._desired_thrust_axis_w
            desired_thrust_axis_b = quat_apply_inverse(pose_w[:, 3:7], desired_thrust_axis_w)
            # e3 x b3_des = [-b_y, b_x, 0] is yaw invariant and gives the
            # shortest roll/pitch leveling direction for the desired axis.
            torch.nan_to_num(desired_thrust_axis_b[:, 1], out=self._attitude_rate_correction[:, 0])
            self._attitude_rate_correction[:, 0].mul_(-self._attitude_hold_gain)
            torch.nan_to_num(desired_thrust_axis_b[:, 0], out=self._attitude_rate_correction[:, 1])
            self._attitude_rate_correction[:, 1].mul_(self._attitude_hold_gain)
            self._processed_actions[:, 1:3].add_(self._attitude_rate_correction)
        # The policy controls a deliberately smoother residual envelope. Priors
        # are added first, then every axis is bounded by FLARE's published total
        # body-rate envelope before the low-level controller sees the command.
        torch.maximum(
            self._processed_actions[:, 1:4],
            -self._max_body_rate_tensor,
            out=self._processed_actions[:, 1:4],
        )
        torch.minimum(
            self._processed_actions[:, 1:4],
            self._max_body_rate_tensor,
            out=self._processed_actions[:, 1:4],
        )
        torch.where(
            self._needs_motor_init[:, None],
            self._processed_actions[:, 0:1] / 4.0,
            self._controller.motor_thrusts,
            out=self._controller.motor_thrusts,
        )
        self._needs_motor_init.zero_()

    def apply_actions(self):
        output = self._controller.compute(self._processed_actions, self._body_rate_b())
        self._rotor_forces_b[:, :, 0, 2].copy_(output.rotor_thrusts.T)
        self._rotor_torques_b[:, :, 0, 2].copy_(output.rotor_thrusts.T).mul_(self._rotor_yaw_coefficients[:, None])

        composer = self._asset.permanent_wrench_composer
        composer.set_forces_and_torques_index(
            forces=self._rotor_forces_b[0],
            torques=self._rotor_torques_b[0],
            positions=self._rotor_positions_b[0],
            is_global=False,
        )
        for rotor_index in range(1, 4):
            composer.add_forces_and_torques_index(
                forces=self._rotor_forces_b[rotor_index],
                torques=self._rotor_torques_b[rotor_index],
                positions=self._rotor_positions_b[rotor_index],
                is_global=False,
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        index = slice(None) if env_ids is None else env_ids
        self._raw_actions[index] = 0.0
        self._processed_actions[index] = 0.0
        self._rotor_forces_b[:, index] = 0.0
        self._rotor_torques_b[:, index] = 0.0
        self._desired_thrust_axis_w[index] = self._world_up[index]
        self._velocity_hold_scale[index] = 1.0
        self._attitude_rate_correction[index] = 0.0
        self._tilt_cosine[index] = 1.0
        self._vertical_velocity[index] = 0.0
        self._path_speed_reference[index] = 0.0
        self._path_tangent_w[index] = 0.0
        self._path_curvature_w[index] = 0.0
        self._desired_path_velocity_w[index] = 0.0
        self._desired_path_acceleration_w[index] = 0.0
        self._path_velocity_error_w[index] = 0.0
        self._path_scalar_buffer[index] = 0.0
        self._path_vector_scale[index] = 1.0
        self._needs_motor_init[index] = True
        self._controller.reset(env_ids)


@configclass
class CollectiveThrustBodyRateActionCfg(ActionTermCfg):
    """FLARE collective-thrust/body-rate control for a suspended-load drone."""

    class_type: type[CollectiveThrustBodyRateAction] = CollectiveThrustBodyRateAction

    asset_name: str = "robot"
    """Drone scene entity that receives the four rotor contributions."""

    gravity: float = 9.81
    """Gravity magnitude [m/s^2]."""

    max_thrust_to_weight: float = 3.5
    """Maximum collective rotor thrust divided by drone weight."""

    max_body_rates: tuple[float, float, float] = (15.0, 15.0, 5.0)
    """Final roll, pitch, and yaw body-rate limits [rad/s].

    These limits bound the complete command after any attitude prior is added.
    """

    residual_body_rate_limits: tuple[float, float, float] | None = None
    """Policy-residual roll, pitch, and yaw rate limits [rad/s].

    The normalized policy rate actions are scaled by these limits before the
    attitude prior is added. Values must not exceed :attr:`max_body_rates`.
    ``None`` preserves the original FLARE mapping by using
    :attr:`max_body_rates` for both operations.
    """

    attitude_hold_gain: float = 0.0
    """Yaw-invariant upright-hold rate gain [1/s]; zero preserves pure FLARE rate control."""

    horizontal_velocity_damping_gain: float = 0.0
    """Horizontal velocity-to-acceleration gain [1/s]; zero disables velocity hold.

    When positive, the geometric attitude prior targets the thrust direction
    required for ``a_xy = gain * (v_des_xy - v_xy)``. ``v_des`` is zero unless
    :attr:`path_velocity_command_name` selects a live path reference. This
    requires a positive :attr:`attitude_hold_gain` and leaves the policy's
    body-rate commands as residuals that command translational motion.
    """

    maximum_velocity_hold_tilt: float = 0.25
    """Maximum thrust-axis tilt requested by horizontal velocity damping [rad]."""

    vertical_velocity_damping_gain: float = 0.0
    """World vertical velocity damping gain [1/s]; zero disables vertical hold.

    The feedback adds ``m_total * gain * (v_des_z - v_z)`` to collective thrust,
    where ``v_des_z`` is zero unless :attr:`path_velocity_command_name` selects
    a 3D path reference and ``m_total`` is the drone mass plus
    :attr:`suspended_mass`. The policy's collective command remains a residual
    around this damping prior. A positive gain requires
    :attr:`tilt_compensation` so this is a world-vertical force.
    """

    suspended_mass: float = 0.0
    """Payload and cable mass supported by the drone [kg]."""

    tilt_compensation: bool = False
    """Whether to preserve vertical collective force as the drone tilts.

    The compensation divides collective thrust by the body +Z vertical
    projection, with the denominator bounded by
    :attr:`maximum_tilt_compensation_angle`. It is disabled by default so the
    paper-aligned baseline retains its exact FLARE action mapping.
    """

    path_velocity_command_name: str | None = None
    """Command term supplying the live 3D path tangent and speed reference.

    ``None`` preserves zero-world-velocity hold exactly. When set, horizontal
    and vertical damping track ``path_speed_reference * path_tangent_e`` from
    the command term. The command remains the single owner of curvature- and
    braking-aware speed planning, including curriculum changes.
    """

    path_velocity_cross_track_gain: float = 0.0
    """Cross-track error-to-convergence-velocity gain [1/s].

    The desired velocity receives ``-gain * path_cross_track_error_e`` before
    the correction is bounded by :attr:`path_velocity_maximum_cross_track_speed`.
    Zero disables convergence and preserves tangent-only tracking.
    """

    path_velocity_maximum_cross_track_speed: float = 0.75
    """Maximum norm of the cross-track convergence velocity [m/s]."""

    path_velocity_curvature_feedforward_gain: float = 0.0
    """Centripetal acceleration feedforward multiplier; zero disables it.

    The acceleration is ``gain * speed**2 * cross(curvature_binormal, tangent)``
    and shares the existing final tilt/collective safety limits. A positive
    value requires attitude hold and tilt compensation.
    """

    maximum_tilt_compensation_angle: float = 0.5
    """Maximum angle used by the bounded collective tilt compensation [rad]."""

    rate_gains: tuple[float, float, float] = (0.016, 0.016, 0.028)
    """Proportional body-rate gains [N·m/(rad/s)]."""

    rate_integral_gains: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Integral body-rate gains [N·m/rad]; zero preserves proportional control."""

    rate_derivative_gains: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Filtered measured-rate derivative gains [N·m/(rad/s²)]."""

    rate_integral_error_limits: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Symmetric integrated body-rate-error bounds [rad]."""

    rate_derivative_cutoff_hz: float = 30.0
    """First-order measured-rate derivative cutoff [Hz]."""

    torque_limits: tuple[float, float, float] = (0.20, 0.20, 0.08)
    """Symmetric roll, pitch, and yaw torque limits [N·m]."""

    arm_length: float = ROTOR_ARM_LENGTH
    """Rotor x/y coordinate from the drone center of mass [m]."""

    rotor_z: float = ROTOR_HEIGHT
    """Rotor z coordinate from the drone center of mass [m]."""

    yaw_coeff: float = ROTOR_YAW_COEFFICIENT
    """Alternating rotor yaw moment per unit thrust [N·m/N]."""

    rotor_thrust_limits: tuple[float, float] = (0.0, 2.62)
    """Per-rotor thrust limits [N]."""

    allocation_mode: str = "collective_priority"
    """Rotor saturation policy: ``collective_priority`` or ``rate_priority``."""

    tau_up: float = 0.03
    """Thrust-increase first-order time constant [s]."""

    tau_down: float = 0.03
    """Thrust-decrease first-order time constant [s]."""

    dt: float = 0.01
    """Controller and motor-model time step [s]."""
