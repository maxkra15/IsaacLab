# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured joint-space state machine for a KUKA toss and GR1T2 open-hand deflection."""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply

from isaaclab_tasks.contrib.juggle.mdp.reset import (
    JUGGLE_SPHERE_CENTER_OFFSET,
    JUGGLE_SPHERE_OPEN_HAND_POSITION,
    JUGGLE_SPHERE_PRELOAD_HAND_POSITION,
    KUKA_ARM_JOINT_NAMES,
)
from isaaclab_tasks.contrib.stack.mdp.kuka_allegro_reset import KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES

from .relay_env_cfg import GR1_RIGHT_ARM_JOINT_NAMES, GR1_RIGHT_HAND_JOINT_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_GR1_GRASP_LINK_NAMES = (
    "R_index_intermediate_link",
    "R_middle_intermediate_link",
    "R_ring_intermediate_link",
    "R_pinky_intermediate_link",
    "R_thumb_distal_link",
)


class RelayPhase(IntEnum):
    """Stages of the validated outbound deflection."""

    READY = 0
    LAUNCH = 1
    OUTBOUND = 2
    RECOVER = 3


class RelayStateMachine:
    """Drive a KUKA toss into a staged GR1T2 open hand without moving the ball.

    The controller commands only joint-position actions. KUKA release and the
    GR1T2 deflection occur through simulated contact; the ball is not caught.
    """

    def __init__(self, env: ManagerBasedRLEnv) -> None:
        """Bind the configured joint and hand-link indices.

        Args:
            env: Instantiated manager-based relay environment.
        """
        self.env = env
        self.device = torch.device(env.device)
        self.num_envs = env.num_envs
        self.kuka = env.scene["kuka"]
        self.gr1 = env.scene["gr1"]
        self.ball = env.scene["ball"]
        self.kuka_arm_ids = self._joint_ids(self.kuka, KUKA_ARM_JOINT_NAMES)
        self.kuka_hand_ids = self._joint_ids(self.kuka, KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES)
        self.gr1_arm_ids = self._joint_ids(self.gr1, GR1_RIGHT_ARM_JOINT_NAMES)
        self.gr1_hand_ids = self._joint_ids(self.gr1, GR1_RIGHT_HAND_JOINT_NAMES)
        self.kuka_hand_id = self._body_id(self.kuka, "palm_link")
        self.gr1_hand_id = self._body_id(self.gr1, "right_hand_pitch_link")
        self.gr1_grasp_ids = torch.tensor(
            [self._body_id(self.gr1, name) for name in _GR1_GRASP_LINK_NAMES], device=self.device
        )
        if not self.kuka.is_fixed_base or not self.gr1.is_fixed_base:
            raise RuntimeError("Relay Jacobian control requires both robots to have fixed roots.")
        self.phase = torch.full((self.num_envs,), int(RelayPhase.READY), dtype=torch.long, device=self.device)
        self.age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.launch_peak_vz = torch.zeros(self.num_envs, device=self.device)
        self.launch_peak_vx = torch.zeros(self.num_envs, device=self.device)
        self.released_this_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.jacobian_fault_reported = torch.zeros_like(self.released_this_episode)
        self.release_count = torch.zeros_like(self.age)
        self.closest_gr1 = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.closest_gr1_grasp = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.max_ball_height = torch.full((self.num_envs,), float("-inf"), device=self.device)
        self.peak_gr1_joint_speed = torch.zeros(self.num_envs, device=self.device)
        self.kuka_start = self._hand_point(self.kuka, self.kuka_hand_id, JUGGLE_SPHERE_CENTER_OFFSET).clone()
        self.gr1_ready = env.scene.env_origins + torch.tensor((-0.04, 0.14, 0.98), device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Restart only the finite-state controller after an environment reset."""
        ids = slice(None) if env_ids is None else env_ids
        self.phase[ids] = int(RelayPhase.READY)
        self.age[ids] = 0
        self.launch_peak_vz[ids] = 0.0
        self.launch_peak_vx[ids] = 0.0
        self.released_this_episode[ids] = False
        self.jacobian_fault_reported[ids] = False
        self.kuka_start[ids] = self._hand_point(self.kuka, self.kuka_hand_id, JUGGLE_SPHERE_CENTER_OFFSET)[ids]

    def act(self) -> torch.Tensor:
        """Return four action blocks in the environment's configured order [rad]."""
        ball_pos = self.ball.data.root_pos_w.torch
        ball_vel = self.ball.data.root_lin_vel_w.torch
        kuka_pos = self._hand_point(self.kuka, self.kuka_hand_id, JUGGLE_SPHERE_CENTER_OFFSET)
        gr1_pos = self._hand_point(self.gr1, self.gr1_hand_id)
        kuka_vel = self.kuka.data.body_link_lin_vel_w.torch[:, self.kuka_hand_id]
        self._assert_finite_state()
        self.peak_gr1_joint_speed = torch.maximum(
            self.peak_gr1_joint_speed, self.gr1.data.joint_vel.torch.abs().amax(dim=1)
        )
        kuka_distance = torch.linalg.vector_norm(ball_pos - kuka_pos, dim=1)
        gr1_distance = torch.linalg.vector_norm(ball_pos - gr1_pos, dim=1)
        gr1_grasp_distance, _ = self._gr1_grasp_metrics(ball_pos, ball_vel)
        self.closest_gr1 = torch.minimum(self.closest_gr1, gr1_distance)
        self.closest_gr1_grasp = torch.minimum(self.closest_gr1_grasp, gr1_grasp_distance)
        self.max_ball_height = torch.maximum(self.max_ball_height, ball_pos[:, 2] - self.env.scene.env_origins[:, 2])

        self.age += 1
        ready = (self.phase == int(RelayPhase.READY)) & (self.age >= 24) & (kuka_distance < 0.13)
        self._transition(ready, RelayPhase.LAUNCH)
        self.launch_peak_vz[ready] = 0.0
        self.launch_peak_vx[ready] = 0.0
        self.released_this_episode[ready] = False
        held_and_rising = (self.phase == int(RelayPhase.LAUNCH)) & (kuka_distance < 0.08)
        self.launch_peak_vz = torch.where(
            held_and_rising,
            torch.maximum(self.launch_peak_vz, ball_vel[:, 2]),
            self.launch_peak_vz,
        )
        self.launch_peak_vx = torch.where(
            held_and_rising,
            torch.maximum(self.launch_peak_vx, ball_vel[:, 0]),
            self.launch_peak_vx,
        )

        outward_clearance = (
            (self.phase == int(RelayPhase.LAUNCH))
            & (self.launch_peak_vx > 0.15)
            & (self.launch_peak_vz > 0.20)
            & (ball_pos[:, 2] > self.kuka_start[:, 2] + 0.08)
            & (kuka_distance > 0.055)
        )
        self._transition(outward_clearance, RelayPhase.OUTBOUND)
        released = (
            ~self.released_this_episode
            & (self.phase == int(RelayPhase.OUTBOUND))
            & (self.launch_peak_vx > 0.15)
            & (self.launch_peak_vz > 0.20)
            & (ball_vel[:, 0] > 0.0)
            & (ball_pos[:, 2] > self.kuka_start[:, 2] + 0.06)
            & (kuka_distance > 0.08)
        )
        self.release_count += released.long()
        self.released_this_episode |= released

        timed_out = (self.age > 210) & (self.phase != int(RelayPhase.RECOVER))
        self._transition(timed_out, RelayPhase.RECOVER)

        kuka_target = self.kuka_start.clone()
        # Holding the receiver steady avoids a high-speed finger impact during the outbound flight.
        gr1_target = self.gr1_ready.clone()
        # Do not recoil the sender while the newly released ball is still within hand reach.
        launching = (self.phase == int(RelayPhase.LAUNCH)) | (
            (self.phase == int(RelayPhase.OUTBOUND)) & (kuka_distance < 0.25)
        )
        kuka_target[launching] += torch.tensor((0.42, 0.0, 0.60), device=self.device)
        kuka_target[:, 2].clamp_(0.55, 1.45)
        gr1_target[:, 2].clamp_(0.65, 1.35)

        kuka_arm_action = self._arm_action(
            self.kuka,
            self.kuka_arm_ids,
            self.kuka_hand_id,
            kuka_target,
            JUGGLE_SPHERE_CENTER_OFFSET,
            relative_action=True,
        )
        gr1_arm_action = self._arm_action(self.gr1, self.gr1_arm_ids, self.gr1_hand_id, gr1_target)
        kuka_hand_open = (
            (self.phase == int(RelayPhase.LAUNCH))
            & (
                ((self.age >= 34) & (kuka_pos[:, 2] > self.kuka_start[:, 2] + 0.14) & (kuka_vel[:, 2] > 0.20))
                | (self.age >= 56)
            )
        ) | (self.phase == int(RelayPhase.OUTBOUND))
        kuka_hand_action = self._kuka_hand_action(kuka_hand_open)
        gr1_hand_action = torch.zeros_like(self.gr1.data.default_joint_pos.torch[:, self.gr1_hand_ids])
        return torch.cat((kuka_arm_action, kuka_hand_action, gr1_arm_action, gr1_hand_action), dim=1)

    def metrics(self) -> dict[str, int | float]:
        """Return measured outbound-release and proximity diagnostics."""
        result = {
            "physical_kuka_releases": int(self.release_count.sum().item()),
            "closest_gr1_m": float(self.closest_gr1.min().item()),
            "closest_gr1_grasp_link_m": float(self.closest_gr1_grasp.min().item()),
            "max_ball_height_m": float(self.max_ball_height.max().item()),
            "peak_gr1_joint_speed_rad_s": float(self.peak_gr1_joint_speed.max().item()),
        }
        result.update(
            {f"phase_{phase.name.lower()}": int((self.phase == int(phase)).sum().item()) for phase in RelayPhase}
        )
        return result

    def diagnostic_state(self, env_idx: int = 0) -> str:
        """Summarize live ball and hand kinematics for a single environment."""
        origin = self.env.scene.env_origins[env_idx]
        ball = self.ball.data.root_pos_w.torch[env_idx] - origin
        velocity = self.ball.data.root_lin_vel_w.torch[env_idx]
        kuka = self._hand_point(self.kuka, self.kuka_hand_id, JUGGLE_SPHERE_CENTER_OFFSET)[env_idx] - origin
        gr1 = self._hand_point(self.gr1, self.gr1_hand_id)[env_idx] - origin
        kuka_velocity = self.kuka.data.body_link_lin_vel_w.torch[env_idx, self.kuka_hand_id]
        gr1_velocity = self.gr1.data.body_link_lin_vel_w.torch[env_idx, self.gr1_hand_id]
        grasp_distance, grasp_relative_speed = self._gr1_grasp_metrics(
            self.ball.data.root_pos_w.torch, self.ball.data.root_lin_vel_w.torch
        )

        def xyz(vector: torch.Tensor) -> str:
            return "(" + ", ".join(f"{value:.3f}" for value in vector.tolist()) + ")"

        return (
            f"phase={RelayPhase(int(self.phase[env_idx])).name.lower()} age={int(self.age[env_idx])} "
            f"ball={xyz(ball)} m v={xyz(velocity)} m/s; "
            f"KUKA={xyz(kuka)} m v={xyz(kuka_velocity)} m/s d={torch.linalg.vector_norm(ball - kuka):.3f} m; "
            f"GR1T2={xyz(gr1)} m v={xyz(gr1_velocity)} m/s d={torch.linalg.vector_norm(ball - gr1):.3f} m; "
            f"grasp_d={grasp_distance[env_idx]:.3f} m grasp_rel_v={grasp_relative_speed[env_idx]:.3f} m/s"
        )

    def _transition(self, mask: torch.Tensor, phase: RelayPhase) -> None:
        self.phase[mask] = int(phase)
        self.age[mask] = 0

    def _assert_finite_state(self) -> None:
        """Fail visibly before invalid physics state reaches an action or DLS solve."""
        ball_pose = self.ball.data.root_pose_w.torch
        kuka_palm_pose = self.kuka.data.body_link_pose_w.torch[:, self.kuka_hand_id]
        gr1_hand_pose = self.gr1.data.body_link_pose_w.torch[:, self.gr1_hand_id]
        origins = self.env.scene.env_origins
        fields = {
            "ball_pose": (torch.cat((ball_pose[:, :3] - origins, ball_pose[:, 3:]), dim=1), 100.0),
            "ball_velocity": (self.ball.data.root_vel_w.torch, 100.0),
            "kuka_joint_pos": (self.kuka.data.joint_pos.torch, 100.0),
            "kuka_joint_vel": (self.kuka.data.joint_vel.torch, 100.0),
            "gr1_joint_pos": (self.gr1.data.joint_pos.torch, 100.0),
            "gr1_joint_vel": (self.gr1.data.joint_vel.torch, 100.0),
            "kuka_palm_pose": (torch.cat((kuka_palm_pose[:, :3] - origins, kuka_palm_pose[:, 3:]), dim=1), 100.0),
            "gr1_hand_pose": (torch.cat((gr1_hand_pose[:, :3] - origins, gr1_hand_pose[:, 3:]), dim=1), 100.0),
        }
        invalid = {
            name: ~(torch.isfinite(value) & (value.abs() < maximum_abs)).flatten(1).all(dim=1)
            for name, (value, maximum_abs) in fields.items()
        }
        bad = torch.stack(tuple(invalid.values())).any(dim=0)
        if bool(bad.any()):
            env_idx = int(bad.nonzero(as_tuple=False)[0, 0])
            fields_bad = [name for name, mask in invalid.items() if bool(mask[env_idx])]
            raise RuntimeError(
                f"Relay physics state became nonfinite or unphysical in env {env_idx}: {fields_bad}; "
                f"{self.diagnostic_state(env_idx)}"
            )

    def _arm_action(
        self,
        robot,
        joint_ids: torch.Tensor,
        hand_id: int,
        target_w: torch.Tensor,
        hand_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        relative_action: bool = False,
    ) -> torch.Tensor:
        """Apply one bounded DLS step using the public world-frame body Jacobian."""
        position = self._hand_point(robot, hand_id, hand_offset)
        error = (target_w - position).clamp(-0.50, 0.50)
        jacobian = robot.data.body_link_jacobian_w.torch[:, hand_id - 1, :, joint_ids]
        finite_jacobian = torch.isfinite(jacobian).flatten(1).all(dim=1)
        bounded_jacobian = jacobian.abs().flatten(1).amax(dim=1) <= 100.0
        invalid_jacobian = ~(finite_jacobian & bounded_jacobian)
        newly_invalid = invalid_jacobian & ~self.jacobian_fault_reported
        if bool(newly_invalid.any()):
            env_idx = int(newly_invalid.nonzero(as_tuple=False)[0, 0])
            print(
                f"[relay] invalid {robot.cfg.prim_path} hand Jacobian in env {env_idx}; "
                f"finite={bool(finite_jacobian[env_idx])}, "
                f"max_abs={float(jacobian[env_idx].abs().amax().item())}; "
                f"holding measured joints until invalid_state termination. {self.diagnostic_state(env_idx)}",
                flush=True,
            )
        self.jacobian_fault_reported |= invalid_jacobian
        jacobian = torch.where(invalid_jacobian[:, None, None], torch.zeros_like(jacobian), jacobian)
        linear = jacobian[:, :3, :]
        if any(hand_offset):
            local_offset = torch.as_tensor(hand_offset, device=self.device, dtype=position.dtype).expand_as(position)
            offset_w = quat_apply(robot.data.body_link_quat_w.torch[:, hand_id], local_offset)
            angular_columns = jacobian[:, 3:, :].transpose(1, 2)
            offset_columns = offset_w[:, None, :].expand_as(angular_columns)
            linear = linear + torch.cross(angular_columns, offset_columns, dim=2).transpose(1, 2)
        damping = 0.04 * torch.eye(3, device=self.device).expand(self.num_envs, 3, 3)
        task_metric = linear @ linear.transpose(1, 2) + damping
        try:
            delta = (linear.transpose(1, 2) @ torch.linalg.solve(task_metric, error.unsqueeze(-1))).squeeze(-1)
        except RuntimeError as exc:
            raise RuntimeError(f"Relay DLS solve failed for {robot.cfg.prim_path}; {self.diagnostic_state()}") from exc
        if not bool(torch.isfinite(delta).all()):
            raise RuntimeError(
                f"Relay DLS returned nonfinite joints for {robot.cfg.prim_path}; {self.diagnostic_state()}"
            )
        measured = robot.data.joint_pos.torch[:, joint_ids]
        default = robot.data.default_joint_pos.torch[:, joint_ids]
        limits = robot.data.soft_joint_pos_limits.torch[:, joint_ids]
        target = (measured + delta.clamp(-0.07, 0.07)).clamp(limits[..., 0] + 0.01, limits[..., 1] - 0.01)
        return target - measured if relative_action else target - default

    def _kuka_hand_action(self, open_mask: torch.Tensor) -> torch.Tensor:
        preload = torch.as_tensor(JUGGLE_SPHERE_PRELOAD_HAND_POSITION, device=self.device)
        opened = torch.as_tensor(JUGGLE_SPHERE_OPEN_HAND_POSITION, device=self.device)
        target = torch.where(open_mask[:, None], opened[None, :], preload[None, :])
        default = self.kuka.data.default_joint_pos.torch[:, self.kuka_hand_ids]
        return target - default

    def _hand_point(self, robot, body_id: int, offset: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> torch.Tensor:
        position = robot.data.body_link_pos_w.torch[:, body_id]
        if not any(offset):
            return position
        local_offset = torch.as_tensor(offset, device=self.device, dtype=position.dtype).expand_as(position)
        return position + quat_apply(robot.data.body_link_quat_w.torch[:, body_id], local_offset)

    def _gr1_grasp_metrics(self, ball_pos: torch.Tensor, ball_vel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return nearest named finger-link origin distance [m] and relative speed [m/s]."""
        finger_pos = self.gr1.data.body_link_pos_w.torch[:, self.gr1_grasp_ids]
        finger_vel = self.gr1.data.body_link_lin_vel_w.torch[:, self.gr1_grasp_ids]
        distances = torch.linalg.vector_norm(ball_pos[:, None, :] - finger_pos, dim=2)
        nearest = distances.argmin(dim=1)
        nearest_vel = finger_vel[torch.arange(self.num_envs, device=self.device), nearest]
        return distances.gather(1, nearest[:, None]).squeeze(1), torch.linalg.vector_norm(ball_vel - nearest_vel, dim=1)

    def _joint_ids(self, robot, names: tuple[str, ...]) -> torch.Tensor:
        ids, resolved = robot.find_joints(list(names), preserve_order=True, as_proxy=True)
        if tuple(resolved) != names:
            raise RuntimeError(f"Relay action joints resolved as {tuple(resolved)}; expected {names}.")
        return ids.torch

    def _body_id(self, robot, name: str) -> int:
        ids, resolved = robot.find_bodies(name)
        if len(ids) != 1 or resolved != [name]:
            raise RuntimeError(f"Relay hand link '{name}' was not found uniquely.")
        return ids[0]
