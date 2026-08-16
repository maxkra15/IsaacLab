# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared simulation-only helpers for Franka RJ45 reset artifacts.

This module is private to the two command-line tools next to it.  In
particular, it does not create placeholder state when the task physics is not
available: construction fails before an artifact or validation report can be
mistaken for simulation evidence.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import newton
import torch
import warp as wp
from isaaclab_newton.cloner import copy_newton_clone_source
from isaaclab_newton.physics import NewtonManager

from isaaclab import cloner
from isaaclab.utils import math as math_utils

from isaaclab_contrib.coupling import NewtonCouplerManager

from isaaclab_tasks.contrib.franka_rj45_insertion.physics.rj45_assembly import TASK_BODY_COUNT
from isaaclab_tasks.contrib.franka_rj45_insertion.rj45_env import (
    ARM_JOINTS,
    FINGER_JOINTS,
    FrankaRJ45InsertionEnv,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.rj45_env_cfg import RIGID_ENTRY, RJ45_ENTRY
from isaaclab_tasks.contrib.franka_rj45_insertion.task_success import (
    RJ45SuccessResult,
    rj45_insertion_success,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = _REPO_ROOT / "datasets/franka_rj45_insertion/reset_dataset.pt"
DEFAULT_VALIDATION_DIR = _REPO_ROOT / "logs/rsl_rl/franka_rj45_insertion/validation"
NOMINAL_GRASP_QUAT_XYZW = (0.5, 0.5, 0.5, -0.5)


def save_torch_atomic(payload: Mapping[str, Any], output: Path) -> None:
    """Atomically write a Torch artifact without leaving a valid-looking partial file."""
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def configured_arm_home(env: RJ45ResetToolEnv) -> torch.Tensor:
    """Return the configured Panda home; backend default buffers are zero at tool startup."""
    try:
        values = [float(env.cfg.scene.robot.init_state.joint_pos[name]) for name in ARM_JOINTS]
    except KeyError as exc:
        raise RuntimeError(f"Franka RJ45 config is missing the explicit home joint {exc.args[0]!r}.") from exc
    return torch.tensor(values, device=env.device, dtype=torch.float32)


class RJ45ResetToolEnv(FrankaRJ45InsertionEnv):
    """Headless task scene without the runtime reset artifact or RL managers."""

    def load_managers(self) -> None:
        """Bind only physics and robot handles required by the reset tools."""
        ensure_runtime = getattr(self, "_ensure_rj45_runtime", None)
        if callable(ensure_runtime):
            ensure_runtime()
        runtime = getattr(self, "_rj45_runtime", None)
        if runtime is None:
            raise RuntimeError(
                "Franka RJ45 physics did not expose a bound `_rj45_runtime` at PHYSICS_READY. "
                "Reset generation requires the real Newton assembly and will not synthesize state."
            )
        for name in ("task_body_ids", "default_body_q", "default_goal_target_w", "drive_enabled"):
            if not hasattr(runtime, name):
                raise RuntimeError(f"Franka RJ45 runtime is missing required physics field {name!r}.")
        if tuple(runtime.task_body_ids.shape) != (self.num_envs, TASK_BODY_COUNT):
            raise RuntimeError(
                f"Franka RJ45 runtime task-body map must have shape ({self.num_envs}, {TASK_BODY_COUNT}), "
                f"got {runtime.task_body_ids.shape}."
            )

        bind_physics_state = getattr(self, "_bind_physics_state", None)
        if not callable(bind_physics_state):
            raise RuntimeError("Franka RJ45 environment does not expose offline physics-state binding.")
        bind_physics_state()

        # Base initialization only needs these attributes to exist when a
        # visualizer is active.  Tool stepping deliberately bypasses managers.
        self.command_manager = None
        self.recorder_manager = None
        self.action_manager = None
        self.observation_manager = None
        self.termination_manager = None
        self.reward_manager = None
        self.curriculum_manager = None

    def setup_manager_visualizers(self) -> None:
        self.manager_visualizers = {}

    @property
    def rj45_runtime(self):
        runtime = getattr(self, "_rj45_runtime", None)
        if runtime is None:
            raise RuntimeError("Franka RJ45 runtime is unavailable.")
        return runtime

    @property
    def advance_dt(self) -> float:
        return float(self.step_dt if self._physics_handles_decimation else self.physics_dt)

    def advance(
        self,
        duration_s: float,
        update: Callable[[int, int, float], None] | None = None,
    ) -> int:
        """Advance actual coupled physics for at least ``duration_s``."""
        if not math.isfinite(duration_s) or duration_s < 0.0:
            raise ValueError("Simulation duration must be finite and non-negative.")
        step_count = int(math.ceil(duration_s / self.advance_dt))
        for step in range(step_count):
            if update is not None:
                update(step, step_count, (step + 1) / max(step_count, 1))
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.advance_dt)
        return step_count

    def set_drive(self, enabled: bool, targets_w: torch.Tensor | None = None) -> None:
        """Set the graph-safe assembly drive outside graph replay."""
        runtime = self.rj45_runtime
        mask = wp.ones(self.num_envs, dtype=wp.bool, device=self.device)
        if targets_w is not None:
            targets_w = torch.as_tensor(targets_w, device=self.device, dtype=torch.float32)
            if tuple(targets_w.shape) != (self.num_envs, 3):
                raise ValueError(f"Drive targets must have shape ({self.num_envs}, 3).")
            runtime.write_drive_targets(wp.from_torch(targets_w.contiguous(), dtype=wp.vec3), mask)
        runtime.set_drive_enabled(bool(enabled), mask)

    def restore_default_task(self) -> None:
        """Restore all assembly bodies and clear VBD/contact history."""
        mask = wp.ones(self.num_envs, dtype=wp.bool, device=self.device)
        runtime = self.rj45_runtime
        runtime.reset_to_default(_newton_states(), mask)
        runtime.set_drive_enabled(False, mask)
        runtime.restore_goal_drive_targets(mask)
        NewtonManager.invalidate_body_state(env_mask=mask)
        self.flush_reset_history()

    def read_task_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Read complete task state in environment-local coordinates."""
        state = NewtonManager.get_state_0()
        if state is None or state.body_q is None or state.body_qd is None:
            raise RuntimeError("Newton body state is unavailable.")
        body_q = wp.to_torch(state.body_q)[self._task_body_ids].clone()
        body_q[..., :3] -= self.env_origins[:, None, :]
        body_qd = wp.to_torch(state.body_qd)[self._task_body_ids].clone()
        return body_q, body_qd

    def write_task_state(self, body_q_e: torch.Tensor, body_qd: torch.Tensor) -> None:
        """Write complete task state into both Newton state buffers."""
        body_q_e = torch.as_tensor(body_q_e, device=self.device, dtype=torch.float32)
        body_qd = torch.as_tensor(body_qd, device=self.device, dtype=torch.float32)
        expected_q = (self.num_envs, TASK_BODY_COUNT, 7)
        expected_qd = (self.num_envs, TASK_BODY_COUNT, 6)
        if tuple(body_q_e.shape) != expected_q or tuple(body_qd.shape) != expected_qd:
            raise ValueError(
                f"Task state must have shapes {expected_q}/{expected_qd}, got "
                f"{tuple(body_q_e.shape)}/{tuple(body_qd.shape)}."
            )
        body_q_w = body_q_e.clone()
        body_q_w[..., :3] += self.env_origins[:, None, :]
        runtime = self.rj45_runtime
        runtime.write_state(
            _newton_states(),
            wp.from_torch(body_q_w.contiguous(), dtype=wp.transform),
            wp.from_torch(body_qd.contiguous(), dtype=wp.spatial_vector),
            wp.ones(self.num_envs, dtype=wp.bool, device=self.device),
        )
        NewtonManager.invalidate_body_state()

    def read_robot_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self._robot.data.joint_pos.torch
        qd = self._robot.data.joint_vel.torch
        return (
            q[:, self._arm_joint_ids].clone(),
            qd[:, self._arm_joint_ids].clone(),
            q[:, self._finger_joint_ids].clone(),
            qd[:, self._finger_joint_ids].clone(),
        )

    def write_robot_state(
        self,
        arm_q: torch.Tensor,
        finger_q: torch.Tensor,
        *,
        arm_target: torch.Tensor | None = None,
        arm_qd: torch.Tensor | None = None,
        finger_qd: torch.Tensor | None = None,
        finger_target: torch.Tensor | None = None,
    ) -> None:
        """Write robot state and PD targets for every tool world."""
        arm_q = torch.as_tensor(arm_q, device=self.device, dtype=torch.float32)
        finger_q = torch.as_tensor(finger_q, device=self.device, dtype=torch.float32)
        if tuple(arm_q.shape) != (self.num_envs, 7) or tuple(finger_q.shape) != (self.num_envs, 2):
            raise ValueError("Robot state has an unexpected shape.")
        arm_target = (
            arm_q if arm_target is None else torch.as_tensor(arm_target, device=self.device, dtype=torch.float32)
        )
        arm_qd = (
            torch.zeros_like(arm_q)
            if arm_qd is None
            else torch.as_tensor(arm_qd, device=self.device, dtype=torch.float32)
        )
        finger_qd = (
            torch.zeros_like(finger_q)
            if finger_qd is None
            else torch.as_tensor(finger_qd, device=self.device, dtype=torch.float32)
        )
        finger_target = (
            finger_q
            if finger_target is None
            else torch.as_tensor(finger_target, device=self.device, dtype=torch.float32)
        )
        if (
            tuple(arm_target.shape) != (self.num_envs, 7)
            or tuple(arm_qd.shape) != (self.num_envs, 7)
            or tuple(finger_qd.shape) != (self.num_envs, 2)
            or tuple(finger_target.shape) != (self.num_envs, 2)
        ):
            raise ValueError("Robot velocity/target state has an unexpected shape.")
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._robot.write_joint_state_to_sim_index(
            position=arm_q, velocity=arm_qd, joint_ids=self._arm_joint_ids, env_ids=env_ids
        )
        self._robot.set_joint_position_target_index(target=arm_target, joint_ids=self._arm_joint_ids, env_ids=env_ids)
        self._robot.write_joint_state_to_sim_index(
            position=finger_q, velocity=finger_qd, joint_ids=self._finger_joint_ids, env_ids=env_ids
        )
        self._robot.set_joint_position_target_index(
            target=finger_target, joint_ids=self._finger_joint_ids, env_ids=env_ids
        )

    def set_robot_targets(self, arm_q: torch.Tensor, finger_target: torch.Tensor) -> None:
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._robot.set_joint_position_target_index(target=arm_q, joint_ids=self._arm_joint_ids, env_ids=env_ids)
        self._robot.set_joint_position_target_index(
            target=finger_target, joint_ids=self._finger_joint_ids, env_ids=env_ids
        )

    def tcp_pose_e(self) -> torch.Tensor:
        hand = self._robot.data.body_link_pose_w.torch[:, self._tcp_body_idx]
        pos, quat = math_utils.combine_frame_transforms(
            hand[:, :3], hand[:, 3:7], self._tcp_offset_pos, self._tcp_offset_quat
        )
        return torch.cat((pos - self.env_origins, quat), dim=-1)

    def plug_grasp_position_e(self) -> torch.Tensor:
        plug = self.read_task_state()[0][:, 0]
        return plug[:, :3] + math_utils.quat_apply(plug[:, 3:7], self._plug_grasp_offset)

    def flush_reset_history(self) -> None:
        """Consume reset masks and remove stale outer/proxy contact records."""
        self.scene.write_data_to_sim()
        self.sim.forward()
        outer_contacts = NewtonManager.get_contacts()
        proxy_contacts, _, _ = NewtonCouplerManager.get_proxy_contact_data(RIGID_ENTRY, RJ45_ENTRY)
        for contacts in (outer_contacts, proxy_contacts):
            if contacts is not None and contacts.rigid_contact_count is not None:
                contacts.rigid_contact_count.zero_()
        self.scene.update(dt=0.0)


def _newton_states() -> tuple[newton.State, ...]:
    state_0 = NewtonManager.get_state_0()
    state_1 = NewtonManager.get_state_1()
    return (state_0,) if state_1 is None else (state_0, state_1)


@dataclass(frozen=True)
class IKResult:
    arm_q: torch.Tensor
    tcp_position: torch.Tensor
    tcp_quaternion: torch.Tensor
    valid: torch.Tensor
    position_residual: torch.Tensor
    rotation_residual: torch.Tensor


class FrankaResetIK:
    """Batched Newton IK constrained to the task's Panda branch and limits."""

    def __init__(
        self,
        env: RJ45ResetToolEnv,
        *,
        seed: int,
        seeds: int = 64,
        iterations: int = 160,
        noise_std: float = 0.5,
        sampler: str = "gauss",
    ) -> None:
        from isaaclab_newton.ik import (
            NewtonIKJointLimitObjectiveCfg,
            NewtonIKPoseObjectiveCfg,
            NewtonIKSolver,
            NewtonIKSolverCfg,
        )

        plan = env.sim.get_clone_plan()
        resolved = cloner.query.path_to_source(plan, env._robot.cfg.prim_path) if plan is not None else None
        if resolved is None:
            raise RuntimeError("Could not resolve the Franka clone-plan source for reset IK.")
        source_builder = copy_newton_clone_source(resolved[0])
        prototype_origin = -env.env_origins[0]
        prototype = newton.ModelBuilder(up_axis=source_builder.up_axis)
        prototype.add_builder(
            source_builder,
            xform=wp.transform(wp.vec3(*prototype_origin.tolist()), wp.quat_identity()),
        )
        self.model = prototype.finalize(device=str(env.device))
        body_names = [str(label).rsplit("/", 1)[-1] for label in self.model.body_label]
        hand_matches = [index for index, name in enumerate(body_names) if name == env.cfg.tcp_body_name]
        if len(hand_matches) != 1:
            raise RuntimeError(f"Expected one IK body named {env.cfg.tcp_body_name!r}, found {hand_matches}.")
        self.hand_id = hand_matches[0]

        joint_names = [str(label).rsplit("/", 1)[-1] for label in self.model.joint_label]
        joint_q_start = wp.to_torch(self.model.joint_q_start).to(device=env.device, dtype=torch.long)

        def coordinate_id(name: str) -> int:
            matches = [index for index, joint_name in enumerate(joint_names) if joint_name == name]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one IK joint named {name!r}, found {matches}.")
            return int(joint_q_start[matches[0]].item())

        self.arm_coordinate_ids = torch.tensor(
            [coordinate_id(name) for name in ARM_JOINTS], device=env.device, dtype=torch.long
        )
        self.finger_coordinate_ids = torch.tensor(
            [coordinate_id(name) for name in FINGER_JOINTS], device=env.device, dtype=torch.long
        )
        self.arm_limits = env._robot.data.soft_joint_pos_limits.torch[0, env._arm_joint_ids].clone()
        self.home_arm_q = configured_arm_home(env)
        self.capacity = env.num_envs
        self.seed_count = int(seeds)
        if sampler == "none" and self.seed_count != 1:
            raise ValueError("Franka reset IK requires seeds=1 when sampler='none'.")
        self.tcp_offset_position = torch.as_tensor(env.cfg.tcp_offset_pos, device=env.device)
        self.tcp_offset_quaternion = torch.as_tensor(env.cfg.tcp_offset_rot, device=env.device)
        self.joint_seed = wp.to_torch(self.model.joint_q).to(env.device).repeat(self.capacity, 1)
        self.joint_seed[:, self.arm_coordinate_ids] = self.home_arm_q

        objective_name = "franka_rj45_reset_tcp"
        objectives = [
            NewtonIKPoseObjectiveCfg(
                body_name=env.cfg.tcp_body_name,
                name=objective_name,
                body_offset_pos=env.cfg.tcp_offset_pos,
                body_offset_rot=env.cfg.tcp_offset_rot,
                position_weight=100.0,
                rotation_weight=8.0,
            ),
            NewtonIKJointLimitObjectiveCfg(weight=1.0),
        ]
        self.solver = NewtonIKSolver(
            NewtonIKSolverCfg(
                optimizer="lm",
                jacobian_mode="analytic",
                sampler=sampler,
                n_seeds=self.seed_count,
                noise_std=float(noise_std),
                iterations=int(iterations),
                lambda_initial=0.1,
                rng_seed=int(seed),
            ),
            model=self.model,
            num_envs=self.capacity,
            device=str(env.device),
            objectives=objectives,
            link_resolver=lambda _name: self.hand_id,
        )
        self.pose_objective = self.solver.objectives_by_name[objective_name]

    def solve(
        self,
        tcp_position: torch.Tensor,
        tcp_quaternion: torch.Tensor,
        finger_position: torch.Tensor,
        *,
        arm_seed: torch.Tensor | None = None,
    ) -> IKResult:
        tcp_position = torch.as_tensor(tcp_position, device=self.home_arm_q.device, dtype=torch.float32)
        tcp_quaternion = torch.as_tensor(tcp_quaternion, device=self.home_arm_q.device, dtype=torch.float32)
        finger_position = torch.as_tensor(finger_position, device=self.home_arm_q.device, dtype=torch.float32)
        if tuple(tcp_position.shape) != (self.capacity, 3):
            raise ValueError(f"IK targets must contain exactly {self.capacity} rows.")
        self.pose_objective.position_objective.set_target_positions(
            wp.from_torch(tcp_position.contiguous(), dtype=wp.vec3)
        )
        self.pose_objective.rotation_objective.set_target_rotations(
            wp.from_torch(tcp_quaternion.contiguous(), dtype=wp.vec4)
        )
        seed = self.joint_seed.clone()
        if arm_seed is not None:
            arm_seed = torch.as_tensor(arm_seed, device=self.home_arm_q.device, dtype=torch.float32)
            if tuple(arm_seed.shape) != (self.capacity, 7):
                raise ValueError(f"arm_seed must have shape ({self.capacity}, 7).")
            seed[:, self.arm_coordinate_ids] = arm_seed
        seed[:, self.finger_coordinate_ids] = finger_position
        self.solver.solve(wp.from_torch(seed.contiguous(), dtype=wp.float32))

        joint_q = wp.to_torch(self.solver.joint_q).reshape(self.capacity, self.seed_count, -1)
        costs = wp.to_torch(self.solver.costs).reshape(self.capacity, self.seed_count)
        residuals = wp.to_torch(self.solver.solver.residuals).reshape(self.capacity, self.seed_count, -1)
        position_residual = torch.linalg.vector_norm(residuals[:, :, :3] / 100.0, dim=-1)
        rotation_residual = torch.linalg.vector_norm(residuals[:, :, 3:6] / 8.0, dim=-1)
        arm_q = joint_q[:, :, self.arm_coordinate_ids]
        margin = torch.minimum(arm_q - self.arm_limits[:, 0], self.arm_limits[:, 1] - arm_q).amin(-1)
        seed_valid = (
            torch.isfinite(joint_q).all(-1)
            & torch.isfinite(costs)
            & torch.isfinite(position_residual)
            & torch.isfinite(rotation_residual)
            & (costs <= 1.0e-3)
            & (margin >= 0.01)
            & (position_residual <= 0.002)
            & (rotation_residual <= math.radians(2.0))
        )
        score = costs + 1.0e-4 * torch.square(arm_q - self.home_arm_q).sum(-1)
        score = score.masked_fill(~seed_valid, torch.inf)
        valid = seed_valid.any(-1)
        fallback = torch.nan_to_num(costs, nan=torch.inf, posinf=torch.inf, neginf=torch.inf).argmin(-1)
        best = torch.where(valid, score.argmin(-1), fallback)
        rows = torch.arange(self.capacity, device=joint_q.device)
        selected_joint = joint_q[rows, best].clone()
        selected_joint[:, self.finger_coordinate_ids] = finger_position
        selected_arm = selected_joint[:, self.arm_coordinate_ids]
        body_q = wp.to_torch(self.solver.solver.body_q).reshape(self.capacity, self.seed_count, -1, 7)
        hand_pose = body_q[rows, best, self.hand_id]
        actual_position, actual_quaternion = math_utils.combine_frame_transforms(
            hand_pose[:, :3],
            hand_pose[:, 3:7],
            self.tcp_offset_position.expand(self.capacity, -1),
            self.tcp_offset_quaternion.expand(self.capacity, -1),
        )
        return IKResult(
            arm_q=selected_arm,
            tcp_position=actual_position,
            tcp_quaternion=actual_quaternion,
            valid=valid,
            position_residual=position_residual[rows, best],
            rotation_residual=rotation_residual[rows, best],
        )


def identity_orientations(count: int, *, device: str | torch.device) -> torch.Tensor:
    result = torch.zeros((count, 4), device=device)
    result[:, 3] = 1.0
    return result


def randomized_orientations(
    base_quaternion: torch.Tensor,
    *,
    max_angle: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply bounded deterministic orientation jitter in xyzw convention."""
    count = len(base_quaternion)
    axis = torch.randn((count, 3), device=base_quaternion.device, generator=generator)
    axis /= torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1.0e-9)
    angle = (2.0 * torch.rand(count, device=base_quaternion.device, generator=generator) - 1.0) * max_angle
    delta = math_utils.quat_from_angle_axis(angle, axis)
    return math_utils.quat_unique(math_utils.quat_mul(delta, base_quaternion))


def scalar_goal_error(task_q: torch.Tensor, goal_q: torch.Tensor) -> torch.Tensor:
    """Match the environment's fixed-goal scalar error without manager state."""
    plug_error = task_q[:, 0, :3] - goal_q[None, 0, :3]
    axial = plug_error[:, 1].abs()
    radial = torch.linalg.vector_norm(plug_error[:, (0, 2)], dim=-1)
    latch_inverse = math_utils.quat_conjugate(goal_q[1, 3:7].expand(len(task_q), -1))
    latch_error = math_utils.quat_unique(math_utils.quat_mul(latch_inverse, task_q[:, 1, 3:7]))
    latch_angle = torch.linalg.vector_norm(math_utils.axis_angle_from_quat(latch_error), dim=-1)
    return axial + 2.0 * radial + 0.002 * latch_angle


def plug_relative_latch_angle(task_q: torch.Tensor) -> torch.Tensor:
    """Return the unsigned plug-relative latch angle [rad]."""
    plug_quat = task_q[:, 0, 3:7]
    latch_quat = task_q[:, 1, 3:7]
    relative = math_utils.quat_unique(math_utils.quat_mul(math_utils.quat_conjugate(plug_quat), latch_quat))
    return torch.linalg.vector_norm(math_utils.axis_angle_from_quat(relative), dim=-1)


def task_state_is_finite_and_normalized(task_q: torch.Tensor, task_qd: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(task_q).all(dim=(1, 2)) & torch.isfinite(task_qd).all(dim=(1, 2))
    quat_norm = torch.linalg.vector_norm(task_q[..., 3:7], dim=-1)
    return finite & (torch.abs(quat_norm - 1.0) <= 1.0e-3).all(dim=-1)


def joint_limit_mask(env: RJ45ResetToolEnv, arm_q: torch.Tensor, margin: float = 0.005) -> torch.Tensor:
    limits = env._robot.data.soft_joint_pos_limits.torch[0, env._arm_joint_ids]
    return ((arm_q >= limits[:, 0] + margin) & (arm_q <= limits[:, 1] - margin)).all(dim=-1)


@dataclass(frozen=True)
class GraspMetrics:
    valid: torch.Tensor
    tcp_distance: torch.Tensor
    bilateral_deflection: torch.Tensor


def grasp_metrics(
    env: RJ45ResetToolEnv,
    finger_target: torch.Tensor,
    *,
    max_distance: float | None = None,
    min_deflection: float = 2.5e-4,
) -> GraspMetrics:
    _, _, finger_q, _ = env.read_robot_state()
    tcp_distance = torch.linalg.vector_norm(env.tcp_pose_e()[:, :3] - env.plug_grasp_position_e(), dim=-1)
    deflection = finger_q - finger_target
    bilateral = (deflection >= min_deflection).all(dim=-1)
    closed = (finger_target <= 0.006).all(dim=-1) & (finger_q <= 0.012).all(dim=-1)
    distance_limit = float(env.cfg.max_tcp_grasp_distance if max_distance is None else max_distance)
    return GraspMetrics(
        valid=(tcp_distance <= distance_limit) & bilateral & closed,
        tcp_distance=tcp_distance,
        bilateral_deflection=deflection.amin(dim=-1),
    )


@dataclass(frozen=True)
class CollisionMetrics:
    valid: torch.Tensor
    invalid_contact_count: torch.Tensor
    grasp_contact_count: torch.Tensor
    left_grasp_contact_count: torch.Tensor
    right_grasp_contact_count: torch.Tensor
    contact_overflow: bool
    invalid_contact_pairs: tuple[str, ...]


def _collision_buffer_metrics(
    env: RJ45ResetToolEnv,
    contacts,
    model,
    state,
    penetration_tolerance: float,
) -> CollisionMetrics:
    """Inspect one contact buffer against the exact model/state layout that owns its indices."""
    valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    invalid_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    left_grasp_count = torch.zeros_like(invalid_count)
    right_grasp_count = torch.zeros_like(invalid_count)
    invalid_pairs: list[str] = []
    if contacts is None or contacts.rigid_contact_count is None:
        return CollisionMetrics(
            valid,
            invalid_count,
            invalid_count.clone(),
            left_grasp_count,
            right_grasp_count,
            False,
            (),
        )
    reported = int(wp.to_torch(contacts.rigid_contact_count)[0].item())
    capacity = int(contacts.rigid_contact_max)
    overflow = reported > capacity
    count = min(reported, capacity)
    if count == 0:
        if overflow:
            valid[:] = False
        return CollisionMetrics(
            valid,
            invalid_count,
            invalid_count.clone(),
            left_grasp_count,
            right_grasp_count,
            overflow,
            (),
        )

    shape0 = wp.to_torch(contacts.rigid_contact_shape0)[:count].long()
    shape1 = wp.to_torch(contacts.rigid_contact_shape1)[:count].long()
    body0 = wp.to_torch(model.shape_body)[shape0].long()
    body1 = wp.to_torch(model.shape_body)[shape1].long()
    shape_world = wp.to_torch(model.shape_world)
    world = torch.where(shape_world[shape0] >= 0, shape_world[shape0], shape_world[shape1]).long()
    q = wp.to_torch(state.body_q)
    point0 = wp.to_torch(contacts.rigid_contact_point0)[:count]
    point1 = wp.to_torch(contacts.rigid_contact_point1)[:count]
    world_point0 = point0.clone()
    world_point1 = point1.clone()
    dynamic0 = body0 >= 0
    dynamic1 = body1 >= 0
    if bool(dynamic0.any()):
        world_point0[dynamic0] = q[body0[dynamic0], :3] + math_utils.quat_apply(
            q[body0[dynamic0], 3:7], point0[dynamic0]
        )
    if bool(dynamic1.any()):
        world_point1[dynamic1] = q[body1[dynamic1], :3] + math_utils.quat_apply(
            q[body1[dynamic1], 3:7], point1[dynamic1]
        )
    normal = wp.to_torch(contacts.rigid_contact_normal)[:count]
    margin0 = wp.to_torch(contacts.rigid_contact_margin0)[:count]
    margin1 = wp.to_torch(contacts.rigid_contact_margin1)[:count]
    separation = (normal * (world_point1 - world_point0)).sum(-1) - margin0 - margin1

    body_labels = [str(label) for label in model.body_label]
    shape_labels = [str(label) for label in model.shape_label]
    for contact_id in range(count):
        world_id = int(world[contact_id])
        if not 0 <= world_id < env.num_envs:
            continue
        pair = []
        for body_id, shape_id in (
            (int(body0[contact_id]), int(shape0[contact_id])),
            (int(body1[contact_id]), int(shape1[contact_id])),
        ):
            body_label = body_labels[body_id] if body_id >= 0 else ""
            pair.append((body_label, shape_labels[shape_id]))
        robot_index = next(
            (
                index
                for index, (body_label, shape_label) in enumerate(pair)
                if "/Robot/" in body_label or "/Robot/" in shape_label
            ),
            None,
        )
        if robot_index is None:
            continue
        other_index = 1 - robot_index
        robot_label = " ".join(pair[robot_index])
        other_shape_label = pair[other_index][1]
        is_grasp_proxy = other_shape_label.endswith("/Plug/GraspProxy")
        is_left_finger = "panda_leftfinger" in robot_label
        is_right_finger = "panda_rightfinger" in robot_label
        if is_grasp_proxy and is_left_finger:
            left_grasp_count[world_id] += 1
        elif is_grasp_proxy and is_right_finger:
            right_grasp_count[world_id] += 1
        elif float(separation[contact_id]) < -penetration_tolerance:
            invalid_count[world_id] += 1
            valid[world_id] = False
            if len(invalid_pairs) < 64:
                first = pair[0][0] or pair[0][1]
                second = pair[1][0] or pair[1][1]
                invalid_pairs.append(
                    f"world={world_id} {first} <-> {second} separation={float(separation[contact_id]):.6g}"
                )
    if overflow:
        valid[:] = False
    grasp_count = left_grasp_count + right_grasp_count
    return CollisionMetrics(
        valid,
        invalid_count,
        grasp_count,
        left_grasp_count,
        right_grasp_count,
        overflow,
        tuple(invalid_pairs),
    )


def collision_metrics(
    env: RJ45ResetToolEnv,
    penetration_tolerance: float = 5.0e-4,
    *,
    require_bilateral_grasp: bool = True,
) -> CollisionMetrics:
    """Inspect outer and proxy-local contacts, optionally requiring bilateral proxy contact."""
    outer = _collision_buffer_metrics(
        env,
        NewtonManager.get_contacts(),
        NewtonManager.get_model(),
        NewtonManager.get_state_0(),
        penetration_tolerance,
    )
    proxy_contacts, destination_view, destination_state = NewtonCouplerManager.get_proxy_contact_data(
        RIGID_ENTRY,
        RJ45_ENTRY,
    )
    proxy = _collision_buffer_metrics(
        env,
        proxy_contacts,
        destination_view,
        destination_state,
        penetration_tolerance,
    )
    left_grasp_count = outer.left_grasp_contact_count + proxy.left_grasp_contact_count
    right_grasp_count = outer.right_grasp_contact_count + proxy.right_grasp_contact_count
    bilateral_grasp = (left_grasp_count > 0) & (right_grasp_count > 0)
    grasp_valid = bilateral_grasp if require_bilateral_grasp else torch.ones_like(bilateral_grasp)
    return CollisionMetrics(
        valid=outer.valid & proxy.valid & grasp_valid,
        invalid_contact_count=outer.invalid_contact_count + proxy.invalid_contact_count,
        grasp_contact_count=left_grasp_count + right_grasp_count,
        left_grasp_contact_count=left_grasp_count,
        right_grasp_contact_count=right_grasp_count,
        contact_overflow=outer.contact_overflow or proxy.contact_overflow,
        invalid_contact_pairs=outer.invalid_contact_pairs + proxy.invalid_contact_pairs,
    )


def interpolate_arm_motion(
    env: RJ45ResetToolEnv,
    start_q: torch.Tensor,
    end_q: torch.Tensor,
    finger_target: torch.Tensor,
    duration_s: float,
) -> None:
    """Execute a smooth cubic arm-target interpolation in real physics."""

    def update(_step: int, _steps: int, progress: float) -> None:
        blend = progress * progress * (3.0 - 2.0 * progress)
        env.set_robot_targets(torch.lerp(start_q, end_q, blend), finger_target)

    env.advance(duration_s, update)


def advance_reset_bias_hold(
    env: RJ45ResetToolEnv,
    duration_s: float,
    arm_target_bias: torch.Tensor,
    finger_target: torch.Tensor,
) -> int:
    """Replay zero arm action with one measured-state-relative target per simulation step."""
    arm_target_bias = torch.as_tensor(arm_target_bias, device=env.device, dtype=torch.float32)
    finger_target = torch.as_tensor(finger_target, device=env.device, dtype=torch.float32)
    if tuple(arm_target_bias.shape) != (env.num_envs, 7):
        raise ValueError(f"arm_target_bias must have shape ({env.num_envs}, 7).")
    if tuple(finger_target.shape) != (env.num_envs, 2):
        raise ValueError(f"finger_target must have shape ({env.num_envs}, 2).")

    def update(_step: int, _steps: int, _progress: float) -> None:
        current_arm_q, _, _, _ = env.read_robot_state()
        env.set_robot_targets(current_arm_q + arm_target_bias, finger_target)

    return env.advance(duration_s, update)


def exact_success_from_state(
    env: RJ45ResetToolEnv,
    task_q: torch.Tensor,
    task_qd: torch.Tensor,
    goal_task_q: torch.Tensor,
) -> RJ45SuccessResult:
    """Evaluate the runtime success predicate with the live task thresholds."""
    return rj45_insertion_success(
        task_q,
        task_qd,
        goal_task_q,
        axial_tolerance=env.cfg.success_axial_tolerance,
        axial_overtravel_tolerance=env.cfg.success_axial_overtravel_tolerance,
        radial_tolerance=env.cfg.success_radial_tolerance,
        plug_angle_tolerance=env.cfg.success_plug_angle_tolerance,
        latch_angle_tolerance=env.cfg.success_latch_angle_tolerance,
        maximum_plug_spatial_speed=env.cfg.success_max_plug_speed,
    )


def advance_exact_success_dwell(
    env: RJ45ResetToolEnv,
    goal_task_q: torch.Tensor,
    arm_target_bias: torch.Tensor,
    finger_target: torch.Tensor,
    *,
    duration_s: float | None = None,
    require_all_samples: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Hold zero action and sample exact insertion success after every step.

    The arm command follows the task's reset-aware controller convention:
    each tool step applies the stored measured-state target bias once, then
    holds that target through the physics substeps. The task construction
    drive must remain disabled for the stored sample and every replay sample.
    """
    arm_target_bias = torch.as_tensor(arm_target_bias, device=env.device, dtype=torch.float32)
    finger_target = torch.as_tensor(finger_target, device=env.device, dtype=torch.float32)
    if tuple(arm_target_bias.shape) != (env.num_envs, 7):
        raise ValueError(f"arm_target_bias must have shape ({env.num_envs}, 7).")
    if tuple(finger_target.shape) != (env.num_envs, 2):
        raise ValueError(f"finger_target must have shape ({env.num_envs}, 2).")
    if bool(wp.to_torch(env.rj45_runtime.drive_enabled).any()):
        raise RuntimeError("Exact RJ45 success evidence requires the task construction drive to be disabled.")

    required_steps = max(1, math.ceil(float(env.cfg.success_dwell_time_s) / env.advance_dt))
    if duration_s is None:
        sample_steps = required_steps
    else:
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive when provided.")
        sample_steps = math.ceil(duration_s / env.advance_dt)
    if sample_steps < required_steps:
        raise ValueError(f"Exact success replay needs at least {required_steps} post-step samples, got {sample_steps}.")

    initial_q, initial_qd = env.read_task_state()
    initial = exact_success_from_state(env, initial_q, initial_qd, goal_task_q)
    all_samples_success = initial.mask.clone()
    all_post_step_success = torch.ones_like(initial.mask)
    consecutive_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    maximum_consecutive_steps = torch.zeros_like(consecutive_steps)
    maximum_signed_axial_error = initial.signed_axial_error.clone()
    minimum_signed_axial_error = initial.signed_axial_error.clone()
    maximum_axial_error = initial.axial_error.clone()
    maximum_radial_error = initial.radial_error.clone()
    maximum_plug_angle_error = initial.plug_angle_error.clone()
    maximum_latch_angle_error = initial.latch_angle_error.clone()
    maximum_plug_spatial_speed = initial.plug_spatial_speed.clone()
    initial_arm_q, _, _, _ = env.read_robot_state()
    last_arm_target = initial_arm_q + arm_target_bias
    final = initial

    for _step in range(sample_steps):
        current_arm_q, _, _, _ = env.read_robot_state()
        last_arm_target = current_arm_q + arm_target_bias
        env.set_robot_targets(last_arm_target, finger_target)
        env.advance(env.advance_dt)
        if bool(wp.to_torch(env.rj45_runtime.drive_enabled).any()):
            raise RuntimeError("The task construction drive became enabled during exact success evidence.")
        task_q, task_qd = env.read_task_state()
        final = exact_success_from_state(env, task_q, task_qd, goal_task_q)
        all_samples_success &= final.mask
        all_post_step_success &= final.mask
        consecutive_steps = torch.where(final.mask, consecutive_steps + 1, torch.zeros_like(consecutive_steps))
        maximum_consecutive_steps = torch.maximum(maximum_consecutive_steps, consecutive_steps)
        maximum_signed_axial_error = torch.maximum(maximum_signed_axial_error, final.signed_axial_error)
        minimum_signed_axial_error = torch.minimum(minimum_signed_axial_error, final.signed_axial_error)
        maximum_axial_error = torch.maximum(maximum_axial_error, final.axial_error)
        maximum_radial_error = torch.maximum(maximum_radial_error, final.radial_error)
        maximum_plug_angle_error = torch.maximum(maximum_plug_angle_error, final.plug_angle_error)
        maximum_latch_angle_error = torch.maximum(maximum_latch_angle_error, final.latch_angle_error)
        maximum_plug_spatial_speed = torch.maximum(maximum_plug_spatial_speed, final.plug_spatial_speed)

    dwell_satisfied = consecutive_steps >= required_steps
    passed = initial.mask & dwell_satisfied
    if require_all_samples:
        passed &= all_samples_success
    return passed, {
        "stored_capture_success": initial.mask,
        "all_samples_success": all_samples_success,
        "all_post_step_success": all_post_step_success,
        "dwell_satisfied": dwell_satisfied,
        "required_dwell_steps": required_steps,
        "sample_steps": sample_steps,
        "final_consecutive_steps": consecutive_steps,
        "maximum_consecutive_steps": maximum_consecutive_steps,
        "initial_signed_axial_error": initial.signed_axial_error,
        "initial_axial_error": initial.axial_error,
        "initial_radial_error": initial.radial_error,
        "initial_plug_angle_error": initial.plug_angle_error,
        "initial_latch_angle_error": initial.latch_angle_error,
        "initial_plug_spatial_speed": initial.plug_spatial_speed,
        "final_signed_axial_error": final.signed_axial_error,
        "final_axial_error": final.axial_error,
        "final_radial_error": final.radial_error,
        "final_plug_angle_error": final.plug_angle_error,
        "final_latch_angle_error": final.latch_angle_error,
        "final_plug_spatial_speed": final.plug_spatial_speed,
        "maximum_signed_axial_error": maximum_signed_axial_error,
        "minimum_signed_axial_error": minimum_signed_axial_error,
        "maximum_axial_error": maximum_axial_error,
        "maximum_radial_error": maximum_radial_error,
        "maximum_plug_angle_error": maximum_plug_angle_error,
        "maximum_latch_angle_error": maximum_latch_angle_error,
        "maximum_plug_spatial_speed": maximum_plug_spatial_speed,
        "last_arm_target": last_arm_target,
    }


def scripted_recovery(
    env: RJ45ResetToolEnv,
    ik: FrankaResetIK,
    goal_task_q: torch.Tensor,
    orientation: torch.Tensor,
    finger_target: torch.Tensor,
    *,
    arm_target_start: torch.Tensor | None = None,
    goal_arm_target: torch.Tensor | None = None,
    motion_s: float = 2.0,
    settle_s: float = 0.5,
    compensation_max_iterations: int = 5,
    compensation_gain: float = 1.0,
    compensation_max_step_m: float = 0.006,
    compensation_motion_s: float = 0.35,
    compensation_hold_s: float = 0.25,
    compensation_tolerance_m: float = 0.0015,
    maximum_ik_joint_step_rad: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Attempt insertion using only the closed-grasp Franka, never the task drive.

    ``arm_target_start - measured_arm_position`` is retained along the
    trajectory so the replay matches the reset-aware zero-action controller.
    A caller should supply a deterministic (zero-noise, single-seed) IK
    instance for replay-invariant evidence. IK deltas are applied around the
    stored canonical arm target, preserving its calibrated equilibrium. The
    task drive remains disabled throughout.
    """
    if compensation_max_iterations < 0:
        raise ValueError("compensation_max_iterations must be non-negative.")
    for name, value in (
        ("compensation_gain", compensation_gain),
        ("compensation_max_step_m", compensation_max_step_m),
        ("compensation_motion_s", compensation_motion_s),
        ("compensation_hold_s", compensation_hold_s),
        ("compensation_tolerance_m", compensation_tolerance_m),
        ("maximum_ik_joint_step_rad", maximum_ik_joint_step_rad),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    env.set_drive(False)
    arm_start, _, _, _ = env.read_robot_state()
    if arm_target_start is None:
        arm_target_start = arm_start
    else:
        arm_target_start = torch.as_tensor(arm_target_start, device=env.device, dtype=torch.float32)
        if tuple(arm_target_start.shape) != (env.num_envs, 7):
            raise ValueError(f"arm_target_start must have shape ({env.num_envs}, 7).")
    reset_target_bias = arm_target_start - arm_start
    goal_plug = goal_task_q[0]
    grasp_offset = torch.as_tensor(env.cfg.plug_grasp_offset, device=env.device).expand(env.num_envs, -1)
    goal_position = goal_plug[:3].expand(env.num_envs, -1) + math_utils.quat_apply(
        goal_plug[3:7].expand(env.num_envs, -1), grasp_offset
    )
    current_task_q, _ = env.read_task_state()
    axial_remaining = (goal_plug[1] - current_task_q[:, 0, 1]).clamp_min(0.0)
    overtravel_distance = (0.25 * axial_remaining).clamp(max=0.004)
    overtravel = goal_position.clone()
    overtravel[:, 1] += overtravel_distance
    overtravel_ik = ik.solve(overtravel, orientation, finger_target, arm_seed=arm_target_start)
    goal_ik = ik.solve(goal_position, orientation, finger_target, arm_seed=overtravel_ik.arm_q)
    uses_canonical_goal_target = goal_arm_target is not None
    if goal_arm_target is None:
        goal_target = goal_ik.arm_q + reset_target_bias
    else:
        goal_arm_target = torch.as_tensor(goal_arm_target, device=env.device, dtype=torch.float32)
        if tuple(goal_arm_target.shape) == (7,):
            goal_arm_target = goal_arm_target.expand(env.num_envs, -1)
        if tuple(goal_arm_target.shape) != (env.num_envs, 7):
            raise ValueError(f"goal_arm_target must have shape (7,) or ({env.num_envs}, 7).")
        goal_target = goal_arm_target.clone()
    overtravel_target = goal_target + overtravel_ik.arm_q - goal_ik.arm_q
    initial_ik_joint_step = torch.abs(overtravel_ik.arm_q - goal_ik.arm_q).amax(dim=-1)
    initial_continuation_valid = initial_ik_joint_step <= maximum_ik_joint_step_rad
    target_valid = joint_limit_mask(env, overtravel_target) & joint_limit_mask(env, goal_target)
    ik_valid = goal_ik.valid & overtravel_ik.valid & target_valid & initial_continuation_valid
    interpolate_arm_motion(env, arm_target_start, overtravel_target, finger_target, 0.75 * motion_s)
    interpolate_arm_motion(env, overtravel_target, goal_target, finger_target, 0.25 * motion_s)
    env.set_robot_targets(goal_target, finger_target)
    env.advance(settle_s)

    # Contact compliance and the closed-grasp equilibrium leave a few
    # millimetres of configuration-dependent residual after a pure IK move.
    # Correct that residual using only measured plug motion and real Franka
    # commands.  Updating the IK objective (rather than teleporting either
    # body) also makes this a useful recoverability oracle for reset rows.
    commanded_position = goal_position.clone()
    current_target = goal_target
    # Keep actuator equilibrium targets and the matching kinematic IK
    # continuation separate.  The former contains a configuration-dependent
    # reset bias and is not a valid seed for the latter.
    current_command_ik_q = goal_ik.arm_q.clone()
    goal_error_history: list[torch.Tensor] = []
    plug_translation_error_history: list[torch.Tensor] = []
    correction_norm_history: list[torch.Tensor] = []
    ik_joint_step_history: list[torch.Tensor] = []
    compensation_iterations = 0
    for iteration in range(compensation_max_iterations + 1):
        current_task_q, _ = env.read_task_state()
        current_error = scalar_goal_error(current_task_q, goal_task_q)
        plug_translation_error = goal_plug[:3].expand(env.num_envs, -1) - current_task_q[:, 0, :3]
        goal_error_history.append(current_error.clone())
        plug_translation_error_history.append(plug_translation_error.clone())
        if bool((current_error <= compensation_tolerance_m).all()) or iteration == compensation_max_iterations:
            break

        active = current_error > compensation_tolerance_m
        correction = compensation_gain * plug_translation_error
        correction_norm = torch.linalg.vector_norm(correction, dim=-1, keepdim=True)
        correction *= torch.clamp(compensation_max_step_m / correction_norm.clamp_min(1.0e-9), max=1.0)
        correction *= active[:, None]
        correction_norm_history.append(torch.linalg.vector_norm(correction, dim=-1))
        next_commanded_position = commanded_position + correction
        compensated_ik = ik.solve(
            next_commanded_position,
            orientation,
            finger_target,
            arm_seed=current_command_ik_q,
        )
        ik_joint_step = compensated_ik.arm_q - current_command_ik_q
        ik_joint_step_norm = torch.abs(ik_joint_step).amax(dim=-1)
        ik_joint_step_history.append(ik_joint_step_norm)
        continuation_valid = ik_joint_step_norm <= maximum_ik_joint_step_rad
        proposed_target = torch.where(active[:, None], current_target + ik_joint_step, current_target)
        proposed_target_valid = joint_limit_mask(env, proposed_target)
        solution_valid = compensated_ik.valid & continuation_valid & proposed_target_valid
        ik_valid &= ~active | solution_valid
        target_valid &= ~active | proposed_target_valid
        command_update = active & solution_valid
        next_target = torch.where(command_update[:, None], proposed_target, current_target)
        next_commanded_position = torch.where(command_update[:, None], next_commanded_position, commanded_position)
        interpolate_arm_motion(env, current_target, next_target, finger_target, compensation_motion_s)
        env.set_robot_targets(next_target, finger_target)
        env.advance(compensation_hold_s)
        current_target = next_target
        current_command_ik_q = torch.where(command_update[:, None], compensated_ik.arm_q, current_command_ik_q)
        commanded_position = next_commanded_position
        compensation_iterations = iteration + 1

    # Return from the force-producing insertion push to the fixed seated
    # Franka equilibrium. This proves the latch maintains insertion without
    # relying on a permanently over-travelled arm command.
    if compensation_iterations:
        interpolate_arm_motion(env, current_target, goal_target, finger_target, compensation_motion_s)
        current_target = goal_target
        env.set_robot_targets(current_target, finger_target)
        env.advance(settle_s)
    dwell_arm_q, _, _, _ = env.read_robot_state()
    dwell_arm_target_bias = current_target - dwell_arm_q
    exact_success, exact_metrics = advance_exact_success_dwell(
        env,
        goal_task_q,
        dwell_arm_target_bias,
        finger_target,
        require_all_samples=True,
    )
    task_q, task_qd = env.read_task_state()
    final_arm_q, _, _, _ = env.read_robot_state()
    error = scalar_goal_error(task_q, goal_task_q)
    # Newton spatial velocities store world linear xyz first, angular xyz last.
    plug_speed = torch.linalg.vector_norm(task_qd[:, 0, :3], dim=-1)
    grasp = grasp_metrics(env, finger_target)
    collision = collision_metrics(env)
    success = (
        ik_valid
        & (error <= 0.002)
        & (plug_speed <= 0.03)
        & exact_success
        & grasp.valid
        & collision.valid
        & task_state_is_finite_and_normalized(task_q, task_qd)
    )
    return success, {
        "goal_error": error,
        "plug_speed": plug_speed,
        "tcp_grasp_distance": grasp.tcp_distance,
        "ik_valid": ik_valid,
        "target_valid": target_valid,
        "arm_target_bias_norm": torch.linalg.vector_norm(reset_target_bias, dim=-1),
        "arm_target_tracking_error": torch.linalg.vector_norm(exact_metrics["last_arm_target"] - final_arm_q, dim=-1),
        "overtravel_distance": overtravel_distance,
        "used_canonical_goal_arm_target": torch.full(
            (env.num_envs,), uses_canonical_goal_target, device=env.device, dtype=torch.bool
        ),
        "compensation_iterations": torch.full(
            (env.num_envs,), compensation_iterations, device=env.device, dtype=torch.int64
        ),
        "compensation_command_offset": commanded_position - goal_position,
        "goal_error_history": torch.stack(goal_error_history, dim=1),
        "plug_translation_error_history": torch.stack(plug_translation_error_history, dim=1),
        "correction_norm_history": (
            torch.stack(correction_norm_history, dim=1)
            if correction_norm_history
            else torch.empty((env.num_envs, 0), device=env.device)
        ),
        "initial_ik_joint_step_max_rad": initial_ik_joint_step,
        "ik_joint_step_history": (
            torch.stack(ik_joint_step_history, dim=1)
            if ik_joint_step_history
            else torch.empty((env.num_envs, 0), device=env.device)
        ),
        "maximum_ik_joint_step_rad": torch.full(
            (env.num_envs,), maximum_ik_joint_step_rad, device=env.device, dtype=torch.float32
        ),
        "tcp_goal_position_error": torch.linalg.vector_norm(env.tcp_pose_e()[:, :3] - goal_position, dim=-1),
        "invalid_contact_count": collision.invalid_contact_count,
        "invalid_contact_pairs": collision.invalid_contact_pairs,
        "left_grasp_contact_count": collision.left_grasp_contact_count,
        "right_grasp_contact_count": collision.right_grasp_contact_count,
        **{f"exact_success_{name}": value for name, value in exact_metrics.items() if name != "last_arm_target"},
    }


def repeated_to_env_count(tensor: torch.Tensor, count: int) -> torch.Tensor:
    """Repeat a row selection to exactly fill a fixed-width simulation scene."""
    if tensor.shape[0] == count:
        return tensor
    if tensor.shape[0] < 1 or tensor.shape[0] > count:
        raise ValueError("Rows must be non-empty and no wider than the simulation scene.")
    repetitions = math.ceil(count / tensor.shape[0])
    return tensor.repeat((repetitions,) + (1,) * (tensor.ndim - 1))[:count]


def package_versions() -> dict[str, str]:
    """Best-effort version identifiers for validation provenance."""
    result = {
        "newton": str(getattr(newton, "__version__", "unknown")),
        "warp": str(getattr(wp, "__version__", "unknown")),
    }
    try:
        import isaaclab

        result["isaaclab"] = str(getattr(isaaclab, "__version__", "unknown"))
    except Exception:
        result["isaaclab"] = "unknown"
    return result


__all__ = [
    "DEFAULT_DATASET_PATH",
    "DEFAULT_VALIDATION_DIR",
    "NOMINAL_GRASP_QUAT_XYZW",
    "CollisionMetrics",
    "FrankaResetIK",
    "GraspMetrics",
    "IKResult",
    "RJ45ResetToolEnv",
    "advance_exact_success_dwell",
    "advance_reset_bias_hold",
    "collision_metrics",
    "configured_arm_home",
    "exact_success_from_state",
    "grasp_metrics",
    "identity_orientations",
    "interpolate_arm_motion",
    "joint_limit_mask",
    "package_versions",
    "plug_relative_latch_angle",
    "randomized_orientations",
    "repeated_to_env_count",
    "save_torch_atomic",
    "scalar_goal_error",
    "scripted_recovery",
    "task_state_is_finite_and_normalized",
]
