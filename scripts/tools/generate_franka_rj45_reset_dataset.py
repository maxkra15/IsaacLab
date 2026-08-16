# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate physically settled reset states for Franka RJ45 insertion.

The fixed goal is derived first from Newton's authored unplugged state: the
task-only drive advances exactly 35 mm in +Y over seven simulated seconds,
holds briefly, is removed, and the connector settles passively for ten more
seconds.  Training rows are then collected from real coupled rollouts with a
closed Franka grasp.  No state is synthesized from a plug pose alone; every
row stores all 37 connector/cable body poses and velocities.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import warp as wp
from _franka_rj45_reset_tools import (
    DEFAULT_DATASET_PATH,
    NOMINAL_GRASP_QUAT_XYZW,
    FrankaResetIK,
    RJ45ResetToolEnv,
    advance_exact_success_dwell,
    advance_reset_bias_hold,
    collision_metrics,
    configured_arm_home,
    grasp_metrics,
    interpolate_arm_motion,
    joint_limit_mask,
    package_versions,
    plug_relative_latch_angle,
    randomized_orientations,
    save_torch_atomic,
    scalar_goal_error,
    scripted_recovery,
    task_state_is_finite_and_normalized,
)

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import math as math_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.franka_rj45_insertion.reset_dataset_io import (
    FRANKA_RJ45_RESET_DATASET_FORMAT,
    FRANKA_RJ45_RESET_DATASET_SCHEMA_VERSION,
    RESET_DATASET_STATE_NAMES,
    reset_dataset_content_digest,
    reset_dataset_validate_runtime,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.rj45_env_cfg import (
    FrankaRJ45InsertionEnvCfg,
    reset_dataset_task_contract,
)

_PHASE_NAMES = (
    "near_goal_1_to_3_mm",
    "near_goal_3_to_7_mm",
    "approach_7_to_12_mm",
    "approach_12_to_18_mm",
    "full_approach_18_to_25_mm",
)
_PHASE_DISTANCE_RANGES = (
    (0.0010, 0.0030),
    (0.0030, 0.0070),
    (0.0070, 0.0120),
    (0.0120, 0.0180),
    (0.0180, 0.0250),
)


@dataclass(frozen=True)
class GeneratorCfg:
    """Bounded, deterministic generation settings."""

    rows_per_phase: int = 64
    batch_size: int = 32
    seed: int = 2026
    max_batches_per_phase: int = 64
    goal_drive_distance_m: float = 0.035
    goal_drive_ramp_s: float = 7.0
    goal_drive_hold_s: float = 1.0
    goal_passive_settle_s: float = 10.0
    goal_stability_window_s: float = 2.0
    construction_robot_park_s: float = 0.25
    goal_grasp_approach_s: float = 2.0
    goal_grasp_hold_s: float = 2.0
    nominal_passive_grasp_hold_s: float = 1.0
    candidate_drive_s: float = 2.5
    candidate_grasp_approach_s: float = 2.0
    candidate_grasp_settle_s: float = 1.0
    candidate_compensation_max_iterations: int = 6
    candidate_compensation_gain: float = 1.0
    candidate_compensation_max_step_m: float = 0.008
    candidate_compensation_motion_s: float = 0.35
    candidate_compensation_hold_s: float = 0.25
    candidate_compensation_tolerance_m: float = 0.0005
    maximum_ik_joint_step_rad: float = 0.5
    recovery_motion_s: float = 2.0
    recovery_settle_s: float = 0.5
    recovery_compensation_max_iterations: int = 5
    recovery_compensation_gain: float = 1.0
    recovery_compensation_max_step_m: float = 0.006
    recovery_compensation_motion_s: float = 0.35
    recovery_compensation_hold_s: float = 0.25
    recovery_compensation_tolerance_m: float = 0.0015
    finger_open_position: float = 0.018
    finger_target: float = 0.004
    grasp_close_s: float = 1.0
    grasp_hold_s: float = 0.5
    tcp_position_jitter_m: float = 4.0e-4
    tcp_rotation_jitter_rad: float = math.radians(1.0)
    connector_lateral_jitter_m: float = 6.0e-4
    cable_wiggle_m: float = 5.0e-4
    compensation_max_iterations: int = 10
    compensation_gain: float = 1.0
    compensation_max_step_m: float = 0.03
    compensation_motion_s: float = 0.75
    compensation_hold_s: float = 0.5
    compensation_tolerance_m: float = 0.002
    cold_start_max_cycles: int = 6
    cold_start_goal_settle_s: float = 10.0
    cold_start_candidate_settle_s: float = 0.5
    cold_start_goal_max_task_drift_m: float = 0.005
    cold_start_goal_max_plug_drift_m: float = 0.0005
    cold_start_goal_max_connector_drift_m: float = 0.001
    cold_start_candidate_max_task_drift_m: float = 0.005
    cold_start_candidate_max_plug_drift_m: float = 0.003
    cold_start_candidate_max_connector_drift_m: float = 0.003
    cold_start_goal_max_cable_linear_speed_m_s: float = 0.01
    cold_start_candidate_max_cable_linear_speed_m_s: float = 0.02
    canonical_peak_latch_angle_min_rad: float = 0.1
    canonical_final_latch_angle_max_rad: float = 0.05

    def __post_init__(self) -> None:
        for name in ("rows_per_phase", "batch_size", "max_batches_per_phase"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.goal_drive_distance_m != 0.035:
            raise ValueError("The canonical Newton goal drive must remain exactly +35 mm.")
        if self.goal_drive_ramp_s < 7.0:
            raise ValueError("The canonical goal ramp must last at least seven simulated seconds.")
        if self.goal_passive_settle_s < 10.0:
            raise ValueError("The canonical goal must settle drive-free for at least ten simulated seconds.")
        if not 0.0 < self.goal_stability_window_s < self.goal_passive_settle_s:
            raise ValueError("goal_stability_window_s must lie inside the passive-settle interval.")
        if self.compensation_max_iterations <= 0:
            raise ValueError("compensation_max_iterations must be positive.")
        if self.candidate_compensation_max_iterations <= 0:
            raise ValueError("candidate_compensation_max_iterations must be positive.")
        if self.cold_start_max_cycles <= 0:
            raise ValueError("cold_start_max_cycles must be positive.")
        if self.construction_robot_park_s < 0.25:
            raise ValueError("The construction robot park must hold the authored plug for at least 0.25 s.")


class ResetDatasetGenerator:
    """Generate a fixed goal and five recoverable reset phases."""

    def __init__(self, env: RJ45ResetToolEnv, cfg: GeneratorCfg) -> None:
        if env.num_envs != cfg.batch_size:
            raise ValueError(f"Generator expected {cfg.batch_size} simulation worlds, got {env.num_envs}.")
        self.env = env
        self.cfg = cfg
        self.device = torch.device(env.device)
        self.random = torch.Generator(device=self.device).manual_seed(cfg.seed)
        # A single sampler-free solver avoids lane/call-sequence dependence
        # and is seeded locally from each preceding arm target. Keeping one
        # solver instance also avoids duplicate Newton IK model ownership.
        self.ik = FrankaResetIK(
            env,
            seed=cfg.seed,
            seeds=1,
            noise_std=0.0,
            sampler="none",
        )
        self.recovery_ik = self.ik
        self.open_finger_q = torch.full((env.num_envs, 2), cfg.finger_open_position, device=self.device)
        self.finger_target = torch.full((env.num_envs, 2), cfg.finger_target, device=self.device)
        self.home_arm_q = configured_arm_home(env).repeat(env.num_envs, 1)
        self.base_orientation = torch.tensor(NOMINAL_GRASP_QUAT_XYZW, device=self.device, dtype=torch.float32).repeat(
            env.num_envs, 1
        )
        self.attempt_counts = torch.zeros(len(_PHASE_NAMES), dtype=torch.int64)
        self.rejection_counts: dict[int, dict[str, int]] = {
            phase: defaultdict(int) for phase in range(len(_PHASE_NAMES))
        }
        self.cold_start_accepted_cycles: dict[int, list[int]] = {phase: [] for phase in range(len(_PHASE_NAMES))}
        self.cold_start_accepted_worst_body_indices: dict[int, list[int]] = {
            phase: [] for phase in range(len(_PHASE_NAMES))
        }

    def _drive_enabled(self) -> torch.Tensor:
        return wp.to_torch(self.env.rj45_runtime.drive_enabled).to(dtype=torch.bool)

    def _assert_drive_disabled(self, context: str) -> None:
        enabled = self._drive_enabled()
        if bool(enabled.any()):
            raise RuntimeError(
                f"The task construction drive must be disabled before {context}; "
                f"enabled worlds={torch.where(enabled)[0].tolist()}."
            )

    def _park_robot_and_restore_task(self) -> dict[str, Any]:
        """Park Franka while the drive holds the exact authored unplugged pose.

        The task is restored a second time after the physical robot park so
        the subsequent +35 mm construction ramp starts from the serialized
        Newton source state, rather than from a spring tracking residual.
        """
        self.env.restore_default_task()
        official_start_w = wp.to_torch(self.env.rj45_runtime.default_body_q)[:, 0, :3].clone()
        self.env.write_robot_state(
            self.home_arm_q,
            self.open_finger_q,
            finger_target=self.open_finger_q,
        )
        self.env.set_drive(True, official_start_w)
        self.env.flush_reset_history()
        park_steps = self.env.advance(
            self.cfg.construction_robot_park_s,
            lambda _step, _steps, _progress: self.env.set_drive(True, official_start_w),
        )
        parked_q, _ = self.env.read_task_state()
        parked_plug_w = parked_q[:, 0, :3] + self.env.env_origins
        park_end_error = torch.linalg.vector_norm(parked_plug_w - official_start_w, dim=-1)

        # Construction-only hard restore: keep the now-parked Franka, restore
        # all 37 task bodies exactly, and re-enable the authored-start hold.
        self.env.restore_default_task()
        self.env.set_drive(True, official_start_w)
        self.env.flush_reset_history()
        restored_q, _ = self.env.read_task_state()
        restored_plug_w = restored_q[:, 0, :3] + self.env.env_origins
        final_error = torch.linalg.vector_norm(restored_plug_w - official_start_w, dim=-1)
        if not bool(torch.allclose(restored_plug_w, official_start_w, atol=1.0e-6, rtol=0.0)):
            raise RuntimeError(
                f"Construction park failed to restore the exact authored RJ45 start: error={final_error.tolist()}."
            )
        if not bool(self._drive_enabled().all()):
            raise RuntimeError("The authored-start construction hold was not enabled in every world.")
        return {
            "construction_only_task_drive": True,
            "robot_park_duration_s": self.cfg.construction_robot_park_s,
            "robot_park_steps": park_steps,
            "drive_target": "exact_authored_unplugged_plug_position",
            "maximum_park_end_tracking_error_m": float(park_end_error.max()),
            "maximum_postpark_exact_restore_error_m": float(final_error.max()),
            "exact_task_restore_after_robot_park": True,
        }

    def _close_gripper(
        self,
        arm_target: torch.Tensor,
        finger_target: torch.Tensor | None = None,
    ) -> None:
        """Close from a non-overlapping approach through real actuator dynamics."""
        finger_target = self.finger_target if finger_target is None else finger_target

        def close(_step: int, _steps: int, progress: float) -> None:
            blend = progress * progress * (3.0 - 2.0 * progress)
            target = torch.lerp(self.open_finger_q, finger_target, blend)
            self.env.set_robot_targets(arm_target, target)

        self.env.advance(self.cfg.grasp_close_s, close)
        self.env.set_robot_targets(arm_target, finger_target)
        self.env.advance(self.cfg.grasp_hold_s)

    def _approach_tcp_with_compensation(
        self,
        desired_position: torch.Tensor,
        orientation: torch.Tensor,
        nominal_ik=None,
        *,
        approach_s: float | None = None,
    ) -> dict[str, Any]:
        """Reach a TCP target using bounded steady-state residual compensation.

        Each update measures the simulated Cartesian residual, adds that
        residual to the previous IK objective (with a 30 mm bound), resolves
        IK, and smoothly updates the joint target. Measured joint position and
        the converged actuator target are intentionally kept separate.
        """
        nominal_ik = nominal_ik or self.ik.solve(desired_position, orientation, self.open_finger_q)
        arm_start, _, _, _ = self.env.read_robot_state()
        interpolate_arm_motion(
            self.env,
            arm_start,
            nominal_ik.arm_q,
            self.open_finger_q,
            self.cfg.candidate_grasp_approach_s if approach_s is None else approach_s,
        )
        self.env.set_robot_targets(nominal_ik.arm_q, self.open_finger_q)
        self.env.advance(self.cfg.candidate_grasp_settle_s)
        commanded_position = desired_position.clone()
        arm_target = nominal_ik.arm_q
        ik_valid = nominal_ik.valid.clone()
        error_history: list[torch.Tensor] = []
        iteration_count = 0
        for iteration in range(self.cfg.compensation_max_iterations + 1):
            actual_position = self.env.tcp_pose_e()[:, :3].clone()
            error = desired_position - actual_position
            error_norm = torch.linalg.vector_norm(error, dim=-1)
            error_history.append(error_norm)
            if bool((error_norm <= self.cfg.compensation_tolerance_m).all()):
                break
            if iteration == self.cfg.compensation_max_iterations:
                break
            active = error_norm > self.cfg.compensation_tolerance_m
            correction_norm = torch.linalg.vector_norm(error, dim=-1, keepdim=True)
            correction = error * torch.clamp(
                self.cfg.compensation_max_step_m / correction_norm.clamp_min(1.0e-9),
                max=1.0,
            )
            next_commanded_position = commanded_position + self.cfg.compensation_gain * correction
            next_commanded_position = torch.where(active[:, None], next_commanded_position, commanded_position)
            compensated_ik = self.ik.solve(
                next_commanded_position,
                orientation,
                self.open_finger_q,
                arm_seed=arm_target,
            )
            ik_valid &= ~active | compensated_ik.valid
            next_arm_target = torch.where(active[:, None], compensated_ik.arm_q, arm_target)
            current_arm_q, _, _, _ = self.env.read_robot_state()
            interpolate_arm_motion(
                self.env,
                current_arm_q,
                next_arm_target,
                self.open_finger_q,
                self.cfg.compensation_motion_s,
            )
            self.env.set_robot_targets(next_arm_target, self.open_finger_q)
            self.env.advance(self.cfg.compensation_hold_s)
            commanded_position = next_commanded_position
            arm_target = next_arm_target
            iteration_count = iteration + 1

        final_arm_q, _, _, _ = self.env.read_robot_state()
        final_error = error_history[-1]
        return {
            "valid": ik_valid & (final_error <= self.cfg.compensation_tolerance_m),
            "arm_target": arm_target,
            "cartesian_target": commanded_position,
            "cartesian_bias": commanded_position - desired_position,
            "final_tcp_error": final_error,
            "arm_tracking_error": torch.linalg.vector_norm(arm_target - final_arm_q, dim=-1),
            "iterations": iteration_count,
            "error_history": torch.stack(error_history, dim=1),
        }

    def _stabilize_closed_candidate(
        self,
        desired_plug_position: torch.Tensor,
        orientation: torch.Tensor,
        finger_target: torch.Tensor,
        *,
        initial_arm_target: torch.Tensor,
        initial_command_ik_q: torch.Tensor,
        initial_commanded_tcp: torch.Tensor,
    ) -> dict[str, Any]:
        """Find the drive-free Franka target that holds a requested plug pose.

        The construction drive first places the plug geometrically.  Once it
        is released, this loop measures the physical plug residual and moves
        only the closed-grasp Franka.  Kinematic IK commands and biased
        actuator equilibrium targets are tracked separately so each update is
        an incremental continuation rather than a fresh IK branch.
        """
        self._assert_drive_disabled("stabilizing a drive-free candidate")
        current_target = initial_arm_target.clone()
        current_command_ik_q = initial_command_ik_q.clone()
        commanded_tcp = initial_commanded_tcp.clone()
        valid = joint_limit_mask(self.env, current_target)
        error_history: list[torch.Tensor] = []
        ik_joint_step_history: list[torch.Tensor] = []
        iterations = 0
        for iteration in range(self.cfg.candidate_compensation_max_iterations + 1):
            task_q, _ = self.env.read_task_state()
            translation_error = desired_plug_position - task_q[:, 0, :3]
            error_norm = torch.linalg.vector_norm(translation_error, dim=-1)
            error_history.append(error_norm.clone())
            if bool((error_norm <= self.cfg.candidate_compensation_tolerance_m).all()):
                break
            if iteration == self.cfg.candidate_compensation_max_iterations:
                break

            active = error_norm > self.cfg.candidate_compensation_tolerance_m
            correction = self.cfg.candidate_compensation_gain * translation_error
            correction_norm = torch.linalg.vector_norm(correction, dim=-1, keepdim=True)
            correction *= torch.clamp(
                self.cfg.candidate_compensation_max_step_m / correction_norm.clamp_min(1.0e-9),
                max=1.0,
            )
            correction *= active[:, None]
            proposed_tcp = commanded_tcp + correction
            next_ik = self.ik.solve(
                proposed_tcp,
                orientation,
                finger_target,
                arm_seed=current_command_ik_q,
            )
            ik_joint_step = next_ik.arm_q - current_command_ik_q
            ik_joint_step_norm = torch.abs(ik_joint_step).amax(dim=-1)
            ik_joint_step_history.append(ik_joint_step_norm)
            continuation_valid = ik_joint_step_norm <= self.cfg.maximum_ik_joint_step_rad
            proposed_target = torch.where(active[:, None], current_target + ik_joint_step, current_target)
            target_valid = joint_limit_mask(self.env, proposed_target)
            solution_valid = next_ik.valid & continuation_valid & target_valid
            valid &= ~active | solution_valid
            command_update = active & solution_valid
            next_target = torch.where(command_update[:, None], proposed_target, current_target)
            proposed_tcp = torch.where(command_update[:, None], proposed_tcp, commanded_tcp)
            interpolate_arm_motion(
                self.env,
                current_target,
                next_target,
                finger_target,
                self.cfg.candidate_compensation_motion_s,
            )
            self.env.set_robot_targets(next_target, finger_target)
            self.env.advance(self.cfg.candidate_compensation_hold_s)
            self._assert_drive_disabled("compensating a drive-free candidate")
            current_target = next_target
            current_command_ik_q = torch.where(command_update[:, None], next_ik.arm_q, current_command_ik_q)
            commanded_tcp = proposed_tcp
            iterations = iteration + 1

        final_task_q, _ = self.env.read_task_state()
        final_error = torch.linalg.vector_norm(desired_plug_position - final_task_q[:, 0, :3], dim=-1)
        return {
            "valid": valid & (final_error <= self.cfg.candidate_compensation_tolerance_m),
            "arm_target": current_target,
            "command_ik_q": current_command_ik_q,
            "commanded_tcp": commanded_tcp,
            "final_plug_error": final_error,
            "iterations": iterations,
            "error_history": torch.stack(error_history, dim=1),
            "ik_joint_step_history": (
                torch.stack(ik_joint_step_history, dim=1)
                if ik_joint_step_history
                else torch.empty((self.env.num_envs, 0), device=self.device)
            ),
        }

    def _capture_full_state(
        self,
        arm_target: torch.Tensor,
        finger_target: torch.Tensor,
        *,
        context: str,
    ) -> dict[str, torch.Tensor]:
        """Capture a complete drive-free robot/task state on the live device."""
        self._assert_drive_disabled(context)
        task_q, task_qd = self.env.read_task_state()
        arm_q, arm_qd, finger_q, finger_qd = self.env.read_robot_state()
        return {
            "arm_joint_position": arm_q.clone(),
            "arm_joint_target": torch.as_tensor(arm_target, device=self.device).clone(),
            "arm_joint_velocity": arm_qd.clone(),
            "finger_joint_position": finger_q.clone(),
            "finger_joint_velocity": finger_qd.clone(),
            "finger_joint_target": torch.as_tensor(finger_target, device=self.device).clone(),
            "task_body_pose": task_q.clone(),
            "task_body_velocity": task_qd.clone(),
        }

    def _restore_full_state(self, state: dict[str, torch.Tensor], *, context: str) -> None:
        """Restore both state buffers, actuator targets, and cold solver history."""
        self.env.write_task_state(state["task_body_pose"], state["task_body_velocity"])
        self.env.write_robot_state(
            state["arm_joint_position"],
            state["finger_joint_position"],
            arm_target=state["arm_joint_target"],
            arm_qd=state["arm_joint_velocity"],
            finger_qd=state["finger_joint_velocity"],
            finger_target=state["finger_joint_target"],
        )
        self.env.set_drive(False)
        self.env.flush_reset_history()
        self._assert_drive_disabled(context)

    def _cold_start_canonicalize(
        self,
        initial_state: dict[str, torch.Tensor],
        *,
        context: str,
        settle_s: float,
        maximum_plug_drift_m: float,
        maximum_connector_drift_m: float,
        maximum_task_body_drift_m: float,
        maximum_cable_linear_speed_m_s: float,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, list[dict[str, Any]]]:
        """Iterate a snapshot to a history-independent reset fixed point.

        A cycle restores the candidate into both Newton buffers, invalidates
        VBD/contact history, settles with the task drive off, and recaptures.
        The returned canonical state is the *pre-settle* state from the first
        cycle satisfying every drift/speed gate, so the exact stored state has
        itself passed a fresh cold-start replay.
        """
        candidate = {name: value.clone() for name, value in initial_state.items()}
        history: list[dict[str, Any]] = []
        stable = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool)
        post_state = candidate
        tested_candidate = candidate
        for cycle in range(1, self.cfg.cold_start_max_cycles + 1):
            tested_candidate = candidate
            self._restore_full_state(candidate, context=f"{context} cold-start replay cycle {cycle}")
            start_q, start_qd = self.env.read_task_state()
            arm_bias = candidate["arm_joint_target"] - candidate["arm_joint_position"]
            steps = advance_reset_bias_hold(
                self.env,
                settle_s,
                arm_bias,
                candidate["finger_joint_target"],
            )
            self._assert_drive_disabled(f"{context} cold-start replay cycle {cycle} evidence")
            final_q, final_qd = self.env.read_task_state()
            final_arm_q, final_arm_qd, final_finger_q, final_finger_qd = self.env.read_robot_state()
            body_drift = torch.linalg.vector_norm(final_q[..., :3] - start_q[..., :3], dim=-1)
            task_drift, worst_body = body_drift.max(dim=-1)
            plug_drift = body_drift[:, 0]
            connector_drift = body_drift[:, :2].amax(dim=-1)
            start_cable_speed, start_fastest_segment = torch.linalg.vector_norm(start_qd[:, 2:, :3], dim=-1).max(dim=-1)
            final_cable_speed, final_fastest_segment = torch.linalg.vector_norm(final_qd[:, 2:, :3], dim=-1).max(dim=-1)
            post_state = {
                "arm_joint_position": final_arm_q.clone(),
                "arm_joint_target": (final_arm_q + arm_bias).clone(),
                "arm_joint_velocity": final_arm_qd.clone(),
                "finger_joint_position": final_finger_q.clone(),
                "finger_joint_velocity": final_finger_qd.clone(),
                "finger_joint_target": candidate["finger_joint_target"].clone(),
                "task_body_pose": final_q.clone(),
                "task_body_velocity": final_qd.clone(),
            }
            finite = task_state_is_finite_and_normalized(final_q, final_qd)
            finite &= torch.stack(
                [torch.isfinite(value).reshape(self.env.num_envs, -1).all(dim=-1) for value in post_state.values()]
            ).all(dim=0)
            stable = (
                finite
                & joint_limit_mask(self.env, final_arm_q)
                & joint_limit_mask(self.env, post_state["arm_joint_target"])
                & (plug_drift <= maximum_plug_drift_m)
                & (connector_drift <= maximum_connector_drift_m)
                & (task_drift <= maximum_task_body_drift_m)
                & (start_cable_speed <= maximum_cable_linear_speed_m_s)
                & (final_cable_speed <= maximum_cable_linear_speed_m_s)
            )
            cycle_evidence = {
                "cycle": cycle,
                "duration_s": settle_s,
                "steps": steps,
                "stable_world_count": int(stable.sum()),
                "maximum_plug_drift_m": float(plug_drift.max()),
                "maximum_connector_drift_m": float(connector_drift.max()),
                "maximum_task_body_drift_m": float(task_drift.max()),
                "maximum_task_body_drift_by_world_m": task_drift.detach().cpu().tolist(),
                "worst_task_body_index_by_world": worst_body.detach().cpu().tolist(),
                "maximum_start_cable_linear_speed_m_s": float(start_cable_speed.max()),
                "maximum_start_cable_linear_speed_by_world_m_s": start_cable_speed.detach().cpu().tolist(),
                "start_fastest_cable_segment_by_world": start_fastest_segment.detach().cpu().tolist(),
                "maximum_final_cable_linear_speed_m_s": float(final_cable_speed.max()),
                "maximum_final_cable_linear_speed_by_world_m_s": final_cable_speed.detach().cpu().tolist(),
                "final_fastest_cable_segment_by_world": final_fastest_segment.detach().cpu().tolist(),
            }
            history.append(cycle_evidence)
            print(f"[RJ45 COLD START] {context}: {cycle_evidence}", flush=True)
            if bool(stable.all()):
                return tested_candidate, post_state, stable, history
            candidate = post_state
        return tested_candidate, post_state, stable, history

    @torch.inference_mode()
    def nominal_grasp_smoke(self) -> dict[str, Any]:
        """Prove the nominal open-approach/physical-close sequence is stable."""
        park_evidence = self._park_robot_and_restore_task()
        default_q, _ = self.env.read_task_state()
        grasp_offset = torch.as_tensor(self.env.cfg.plug_grasp_offset, device=self.device).expand(self.env.num_envs, -1)
        target_position = default_q[:, 0, :3] + math_utils.quat_apply(default_q[:, 0, 3:7], grasp_offset)
        solution = self.ik.solve(target_position, self.base_orientation, self.open_finger_q)
        if not bool(solution.valid.all()):
            raise RuntimeError(
                "Nominal Franka grasp smoke failed IK; the fixed grasp pose is unavailable on one or more worlds."
            )
        settlement = self._approach_tcp_with_compensation(
            target_position,
            self.base_orientation,
            solution,
        )
        # Reuse this physically converged, sampler-free solution as the local
        # seed for every later seated/start grasp solve.
        self.nominal_grasp_arm_seed = settlement["arm_target"].clone()
        before_q, _ = self.env.read_task_state()
        before_tcp = self.env.tcp_pose_e().clone()
        before_grasp_position = self.env.plug_grasp_position_e().clone()
        before_arm_q, _, before_finger_q, _ = self.env.read_robot_state()
        preclose_collision = collision_metrics(self.env, require_bilateral_grasp=False)
        preclose_live_grasp_distance = torch.linalg.vector_norm(before_tcp[:, :3] - before_grasp_position, dim=-1)
        preclose_valid = (
            settlement["valid"]
            & preclose_collision.valid
            & (preclose_live_grasp_distance <= self.cfg.compensation_tolerance_m)
        )
        if not bool(preclose_valid.all()):
            raise RuntimeError(
                "Nominal Franka pre-close convergence failed: "
                f"tcp_error={settlement['final_tcp_error'].tolist()}, "
                f"live_tcp_grasp_distance={preclose_live_grasp_distance.tolist()}, "
                f"arm_error={settlement['arm_tracking_error'].tolist()}, "
                f"invalid_contacts={preclose_collision.invalid_contact_count.tolist()}, "
                f"pairs={preclose_collision.invalid_contact_pairs}."
            )
        self._close_gripper(settlement["arm_target"])
        release_grasp = grasp_metrics(self.env, self.finger_target)
        release_collision = collision_metrics(self.env)
        if not bool((release_grasp.valid & release_collision.valid).all()):
            raise RuntimeError(
                "Nominal Franka close did not establish a bilateral physical proxy grasp before drive release: "
                f"grasp={release_grasp.valid.tolist()}, "
                f"left_contacts={release_collision.left_grasp_contact_count.tolist()}, "
                f"right_contacts={release_collision.right_grasp_contact_count.tolist()}, "
                f"invalid_contacts={release_collision.invalid_contact_count.tolist()}, "
                f"pairs={release_collision.invalid_contact_pairs}."
            )
        release_q, _ = self.env.read_task_state()
        self.env.set_drive(False)
        self._assert_drive_disabled("the nominal passive-grasp smoke")
        passive_hold_steps = self.env.advance(self.cfg.nominal_passive_grasp_hold_s)
        self._assert_drive_disabled("capturing nominal passive-grasp evidence")
        after_q, after_qd = self.env.read_task_state()
        after_tcp = self.env.tcp_pose_e().clone()
        after_grasp_position = self.env.plug_grasp_position_e().clone()
        after_arm_q, _, after_finger_q, _ = self.env.read_robot_state()
        grasp = grasp_metrics(self.env, self.finger_target)
        collision = collision_metrics(self.env)
        plug_displacement = torch.linalg.vector_norm(after_q[:, 0, :3] - before_q[:, 0, :3], dim=-1)
        plug_speed = torch.linalg.vector_norm(after_qd[:, 0, :3], dim=-1)
        passive_release_drift = torch.linalg.vector_norm(after_q[:, 0, :3] - release_q[:, 0, :3], dim=-1)
        valid = (
            solution.valid
            & preclose_valid
            & release_grasp.valid
            & release_collision.valid
            & grasp.valid
            & collision.valid
            & task_state_is_finite_and_normalized(after_q, after_qd)
            & (plug_displacement <= 0.002)
            & (plug_speed <= 0.08)
        )
        evidence = {
            "passed": bool(valid.all()),
            "orientation_xyzw": NOMINAL_GRASP_QUAT_XYZW,
            "open_finger_position_m": self.cfg.finger_open_position,
            "closed_finger_target_m": self.cfg.finger_target,
            "close_duration_s": self.cfg.grasp_close_s,
            "hold_duration_s": self.cfg.grasp_hold_s,
            "construction_park": park_evidence,
            "construction_drive_enabled_during_open_approach_and_close": True,
            "passive_drive_free_hold_s": self.cfg.nominal_passive_grasp_hold_s,
            "passive_drive_free_hold_steps": passive_hold_steps,
            "drive_was_disabled_for_passive_evidence": True,
            "maximum_plug_displacement_m": float(plug_displacement.max()),
            "maximum_postrelease_plug_drift_m": float(passive_release_drift.max()),
            "maximum_plug_speed": float(plug_speed.max()),
            "maximum_tcp_grasp_distance_m": float(grasp.tcp_distance.max()),
            "maximum_preclose_live_tcp_grasp_distance_m": float(preclose_live_grasp_distance.max()),
            "maximum_ik_position_residual_m": float(solution.position_residual.max()),
            "maximum_ik_fk_target_error_m": float(
                torch.linalg.vector_norm(solution.tcp_position - target_position, dim=-1).max()
            ),
            "maximum_preclose_tcp_target_error_m": float(
                torch.linalg.vector_norm(before_tcp[:, :3] - target_position, dim=-1).max()
            ),
            "maximum_postclose_tcp_target_error_m": float(
                torch.linalg.vector_norm(after_tcp[:, :3] - target_position, dim=-1).max()
            ),
            "maximum_preclose_arm_tracking_error": float(
                torch.linalg.vector_norm(before_arm_q - solution.arm_q, dim=-1).max()
            ),
            "maximum_postclose_arm_tracking_error": float(
                torch.linalg.vector_norm(after_arm_q - settlement["arm_target"], dim=-1).max()
            ),
            "target_update_algorithm": "bounded_iterative_cartesian_residual_compensation",
            "compensation_iterations": int(settlement["iterations"]),
            "maximum_compensated_tcp_error_m": float(settlement["final_tcp_error"].max()),
            "maximum_compensated_arm_tracking_error": float(settlement["arm_tracking_error"].max()),
            "maximum_cartesian_bias_m": float(torch.linalg.vector_norm(settlement["cartesian_bias"], dim=-1).max()),
            "maximum_error_by_iteration_m": settlement["error_history"].max(dim=0).values.cpu().tolist(),
            "preclose_finger_position_range_m": (
                float(before_finger_q.min()),
                float(before_finger_q.max()),
            ),
            "postclose_finger_position_range_m": (
                float(after_finger_q.min()),
                float(after_finger_q.max()),
            ),
            "minimum_bilateral_deflection_m": float(grasp.bilateral_deflection.min()),
            "release_left_grasp_contact_count": int(release_collision.left_grasp_contact_count.sum()),
            "release_right_grasp_contact_count": int(release_collision.right_grasp_contact_count.sum()),
            "invalid_contact_count": int(collision.invalid_contact_count.sum()),
            "allowed_grasp_contact_count": int(collision.grasp_contact_count.sum()),
            "left_grasp_contact_count": int(collision.left_grasp_contact_count.sum()),
            "right_grasp_contact_count": int(collision.right_grasp_contact_count.sum()),
        }
        if not evidence["passed"]:
            diagnostics = {
                "preclose_live_tcp_grasp_distance_m": preclose_live_grasp_distance.tolist(),
                "plug_displacement_xyz_m": (after_q[:, 0, :3] - before_q[:, 0, :3]).tolist(),
                "preclose_plug_position_m": before_q[:, 0, :3].tolist(),
                "postclose_plug_position_m": after_q[:, 0, :3].tolist(),
                "preclose_grasp_position_m": before_grasp_position.tolist(),
                "postclose_grasp_position_m": after_grasp_position.tolist(),
                "preclose_tcp_position_m": before_tcp[:, :3].tolist(),
                "postclose_tcp_position_m": after_tcp[:, :3].tolist(),
                "collision_pairs": collision.invalid_contact_pairs,
            }
            raise RuntimeError(f"Nominal Franka physical-grasp smoke failed: {evidence}; {diagnostics}")
        print(f"[RJ45 GRASP] Nominal physical close passed: {evidence}", flush=True)
        return evidence

    @torch.inference_mode()
    def derive_goal(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Derive and prove the one fixed, passive canonical goal."""
        construction_park_evidence = self._park_robot_and_restore_task()
        initial_q, _ = self.env.read_task_state()
        initial_plug_w = initial_q[:, 0, :3] + self.env.env_origins
        official_start_w = wp.to_torch(self.env.rj45_runtime.default_body_q)[:, 0, :3].clone()
        runtime_nominal_goal_w = wp.to_torch(self.env.rj45_runtime.default_goal_target_w).clone()
        expected_delta = torch.zeros_like(official_start_w)
        expected_delta[:, 1] = self.cfg.goal_drive_distance_m
        oracle_drive_target_w = official_start_w + expected_delta
        if not bool(torch.allclose(initial_plug_w, official_start_w, atol=1.0e-6, rtol=0.0)):
            raise RuntimeError(
                "Restoring the canonical goal did not reproduce Newton's official plug start before stepping: "
                f"error={torch.linalg.vector_norm(initial_plug_w - official_start_w, dim=-1).tolist()}."
            )
        self.env.set_drive(True, initial_plug_w)
        peak_latch_angle = plug_relative_latch_angle(initial_q)

        def sample_latch_angle() -> None:
            task_q, _ = self.env.read_task_state()
            peak_latch_angle.copy_(torch.maximum(peak_latch_angle, plug_relative_latch_angle(task_q)))

        def ramp_drive(_step: int, _steps: int, progress: float) -> None:
            # Quintic smoothstep avoids an impulsive spring target while
            # preserving exactly 35 mm total commanded travel.
            blend = progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))
            self.env.set_drive(True, torch.lerp(initial_plug_w, oracle_drive_target_w, blend))
            sample_latch_angle()

        ramp_steps = self.env.advance(self.cfg.goal_drive_ramp_s, ramp_drive)
        self.env.set_drive(True, oracle_drive_target_w)
        hold_steps = self.env.advance(
            self.cfg.goal_drive_hold_s,
            lambda _step, _steps, _progress: sample_latch_angle(),
        )
        release_q, release_qd = self.env.read_task_state()
        sample_latch_angle()
        self.env.set_drive(False)
        self._assert_drive_disabled("the canonical passive settle")

        early_settle = self.cfg.goal_passive_settle_s - self.cfg.goal_stability_window_s
        early_steps = self.env.advance(early_settle)
        stability_start_q, _ = self.env.read_task_state()
        stability_steps = self.env.advance(self.cfg.goal_stability_window_s)
        self._assert_drive_disabled("capturing the passive canonical task goal")
        passive_goal_q, passive_goal_qd = self.env.read_task_state()

        runtime_nominal_goal_e = runtime_nominal_goal_w - self.env.env_origins
        target_error = torch.linalg.vector_norm(passive_goal_q[:, 0, :3] - runtime_nominal_goal_e, dim=-1)
        insertion_travel = passive_goal_q[:, 0, 1] - initial_q[:, 0, 1]
        last_window_plug_drift = torch.linalg.vector_norm(
            passive_goal_q[:, 0, :3] - stability_start_q[:, 0, :3], dim=-1
        )
        last_window_task_drift = torch.linalg.vector_norm(
            passive_goal_q[..., :3] - stability_start_q[..., :3], dim=-1
        ).amax(dim=-1)
        last_window_connector_drift = torch.linalg.vector_norm(
            passive_goal_q[:, :2, :3] - stability_start_q[:, :2, :3], dim=-1
        ).amax(dim=-1)
        passive_plug_speed = torch.linalg.vector_norm(passive_goal_qd[:, 0, :3], dim=-1)
        passive_latch_angle = plug_relative_latch_angle(passive_goal_q)
        valid = task_state_is_finite_and_normalized(passive_goal_q, passive_goal_qd)
        valid &= peak_latch_angle >= self.cfg.canonical_peak_latch_angle_min_rad
        valid &= passive_latch_angle <= self.cfg.canonical_final_latch_angle_max_rad
        valid &= insertion_travel >= 0.020
        valid &= target_error <= 0.0015
        valid &= last_window_plug_drift <= 5.0e-4
        valid &= last_window_connector_drift <= 0.001
        valid &= passive_plug_speed <= 0.01
        if not bool(valid.all()):
            evidence = {
                "valid": valid.detach().cpu().tolist(),
                "insertion_travel_m": insertion_travel.detach().cpu().tolist(),
                "runtime_nominal_target_error_m": target_error.detach().cpu().tolist(),
                "last_window_plug_drift_m": last_window_plug_drift.detach().cpu().tolist(),
                "last_window_task_drift_m": last_window_task_drift.detach().cpu().tolist(),
                "last_window_connector_drift_m": last_window_connector_drift.detach().cpu().tolist(),
                "final_plug_speed": passive_plug_speed.detach().cpu().tolist(),
                "peak_plug_relative_latch_angle_rad": peak_latch_angle.detach().cpu().tolist(),
                "final_plug_relative_latch_angle_rad": passive_latch_angle.detach().cpu().tolist(),
            }
            raise RuntimeError(f"Canonical RJ45 goal failed passive validation: {evidence}")

        # Approach the seated connector with open fingers, close through real
        # actuator dynamics, and keep the task drive off. The full fixed goal
        # is captured only after this robot-coupled hold.
        grasp_offset = torch.as_tensor(self.env.cfg.plug_grasp_offset, device=self.device).expand(self.env.num_envs, -1)
        goal_tcp_position = passive_goal_q[:, 0, :3] + math_utils.quat_apply(passive_goal_q[:, 0, 3:7], grasp_offset)
        goal_ik = self.ik.solve(
            goal_tcp_position,
            self.base_orientation,
            self.open_finger_q,
            arm_seed=self.nominal_grasp_arm_seed,
        )
        if not bool(goal_ik.valid.all()):
            raise RuntimeError("Canonical seated Franka grasp has no valid fixed-orientation IK solution.")
        goal_settlement = self._approach_tcp_with_compensation(
            goal_tcp_position,
            self.base_orientation,
            goal_ik,
            approach_s=self.cfg.goal_grasp_approach_s,
        )
        preclose_collision = collision_metrics(self.env, require_bilateral_grasp=False)
        preclose_live_grasp_distance = torch.linalg.vector_norm(
            self.env.tcp_pose_e()[:, :3] - self.env.plug_grasp_position_e(), dim=-1
        )
        preclose_valid = (
            goal_settlement["valid"]
            & preclose_collision.valid
            & (preclose_live_grasp_distance <= self.cfg.compensation_tolerance_m)
        )
        if not bool(preclose_valid.all()):
            raise RuntimeError(
                "Canonical seated Franka pre-close convergence failed: "
                f"tcp_error={goal_settlement['final_tcp_error'].tolist()}, "
                f"live_tcp_grasp_distance={preclose_live_grasp_distance.tolist()}, "
                f"arm_error={goal_settlement['arm_tracking_error'].tolist()}, "
                f"invalid_contacts={preclose_collision.invalid_contact_count.tolist()}, "
                f"pairs={preclose_collision.invalid_contact_pairs}."
            )
        self._close_gripper(goal_settlement["arm_target"])
        extra_goal_hold = max(0.0, self.cfg.goal_grasp_hold_s - self.cfg.grasp_hold_s)
        self.env.advance(extra_goal_hold)
        self._assert_drive_disabled("capturing the full canonical Franka goal")
        goal_q, goal_qd = self.env.read_task_state()
        arm_q, arm_qd, finger_q, finger_qd = self.env.read_robot_state()
        goal_grasp = grasp_metrics(self.env, self.finger_target)
        goal_collision = collision_metrics(self.env)
        grasp_plug_drift = torch.linalg.vector_norm(goal_q[:, 0, :3] - passive_goal_q[:, 0, :3], dim=-1)
        grasp_task_drift = torch.linalg.vector_norm(goal_q[..., :3] - passive_goal_q[..., :3], dim=-1).amax(dim=-1)
        grasp_connector_drift = torch.linalg.vector_norm(goal_q[:, :2, :3] - passive_goal_q[:, :2, :3], dim=-1).amax(
            dim=-1
        )
        final_target_error = torch.linalg.vector_norm(goal_q[:, 0, :3] - runtime_nominal_goal_e, dim=-1)
        final_plug_speed = torch.linalg.vector_norm(goal_qd[:, 0, :3], dim=-1)
        warm_goal_latch_angle = plug_relative_latch_angle(goal_q)
        full_goal_valid = (
            goal_settlement["valid"]
            & goal_grasp.valid
            & goal_collision.valid
            & task_state_is_finite_and_normalized(goal_q, goal_qd)
            & joint_limit_mask(self.env, arm_q)
            & joint_limit_mask(self.env, goal_settlement["arm_target"])
            & (grasp_plug_drift <= 0.001)
            & (grasp_connector_drift <= 0.0015)
            & (final_target_error <= 0.0015)
            & (final_plug_speed <= 0.02)
            & (warm_goal_latch_angle <= self.cfg.canonical_final_latch_angle_max_rad)
        )
        if not bool(full_goal_valid.all()):
            raise RuntimeError(
                "Canonical full Franka goal failed closed-grasp validation: "
                f"valid={full_goal_valid.tolist()}, plug_drift={grasp_plug_drift.tolist()}, "
                f"connector_drift={grasp_connector_drift.tolist()}, target_error={final_target_error.tolist()}, "
                f"latch_angle={warm_goal_latch_angle.tolist()}, grasp={goal_grasp.valid.tolist()}, invalid_contacts="
                f"{goal_collision.invalid_contact_count.tolist()}."
            )

        warm_goal_state = self._capture_full_state(
            goal_settlement["arm_target"],
            self.finger_target,
            context="capturing the warm canonical goal for cold-start canonicalization",
        )
        canonical_goal_state, canonical_goal_post, cold_goal_stable, cold_goal_history = self._cold_start_canonicalize(
            warm_goal_state,
            context="canonical goal",
            settle_s=self.cfg.cold_start_goal_settle_s,
            maximum_plug_drift_m=self.cfg.cold_start_goal_max_plug_drift_m,
            maximum_connector_drift_m=self.cfg.cold_start_goal_max_connector_drift_m,
            maximum_task_body_drift_m=self.cfg.cold_start_goal_max_task_drift_m,
            maximum_cable_linear_speed_m_s=self.cfg.cold_start_goal_max_cable_linear_speed_m_s,
        )
        # Prove the exact stored capture under a fresh history reset and the
        # same measured-state-relative zero-action hold used during training.
        # The first replicated world is the one fixed runtime goal.
        self._restore_full_state(
            canonical_goal_state,
            context="canonical goal exact-success dwell replay",
        )
        canonical_arm_bias = canonical_goal_state["arm_joint_target"] - canonical_goal_state["arm_joint_position"]
        exact_goal_success, exact_goal_metrics = advance_exact_success_dwell(
            self.env,
            canonical_goal_state["task_body_pose"][0],
            canonical_arm_bias,
            canonical_goal_state["finger_joint_target"],
            require_all_samples=True,
        )
        exact_goal_post_q, _ = self.env.read_task_state()
        cold_goal_grasp = grasp_metrics(self.env, canonical_goal_post["finger_joint_target"])
        cold_goal_collision = collision_metrics(self.env)
        canonical_latch_angle = plug_relative_latch_angle(canonical_goal_state["task_body_pose"])
        cold_fixed_point_latch_angle = plug_relative_latch_angle(canonical_goal_post["task_body_pose"])
        cold_replay_latch_angle = plug_relative_latch_angle(exact_goal_post_q)
        cold_goal_valid = (
            cold_goal_stable
            & exact_goal_success
            & cold_goal_grasp.valid
            & cold_goal_collision.valid
            & (canonical_latch_angle <= self.cfg.canonical_final_latch_angle_max_rad)
            & (cold_fixed_point_latch_angle <= self.cfg.canonical_final_latch_angle_max_rad)
            & (cold_replay_latch_angle <= self.cfg.canonical_final_latch_angle_max_rad)
        )
        if not bool(cold_goal_valid.all()):
            raise RuntimeError(
                "Canonical full Franka goal did not converge to a history-independent cold-start fixed point: "
                f"stable={cold_goal_stable.tolist()}, grasp={cold_goal_grasp.valid.tolist()}, "
                f"exact_success={exact_goal_success.tolist()}, "
                f"exact_metrics={exact_goal_metrics}, "
                f"capture_latch_angle={canonical_latch_angle.tolist()}, "
                f"fixed_point_latch_angle={cold_fixed_point_latch_angle.tolist()}, "
                f"replay_latch_angle={cold_replay_latch_angle.tolist()}, "
                f"invalid_contacts={cold_goal_collision.invalid_contact_count.tolist()}, "
                f"history={cold_goal_history}."
            )

        warm_to_canonical_body_drift = torch.linalg.vector_norm(
            canonical_goal_state["task_body_pose"][..., :3] - warm_goal_state["task_body_pose"][..., :3], dim=-1
        )
        canonical_goal_q = canonical_goal_state["task_body_pose"]
        canonical_goal_qd = canonical_goal_state["task_body_velocity"]
        canonical_target_error = torch.linalg.vector_norm(canonical_goal_q[:, 0, :3] - runtime_nominal_goal_e, dim=-1)
        canonical_cable_speed, canonical_fastest_cable = torch.linalg.vector_norm(
            canonical_goal_qd[:, 2:, :3], dim=-1
        ).max(dim=-1)

        # Environment 0 is retained as the one fixed goal; cross-world spread
        # proves that replicated derivation stayed deterministic.
        goal_position_spread = torch.linalg.vector_norm(
            canonical_goal_q[:, 0, :3] - canonical_goal_q[:1, 0, :3], dim=-1
        ).amax()
        evidence: dict[str, Any] = {
            "construction_park": construction_park_evidence,
            "construction_drive_scope": "authored-start robot park and task-state construction only",
            "drive_distance_m": self.cfg.goal_drive_distance_m,
            "drive_direction": "+Y",
            "source_start_to_runtime_nominal_target_xyz_m": (runtime_nominal_goal_w[0] - official_start_w[0])
            .detach()
            .cpu()
            .tolist(),
            "source_start_to_oracle_drive_target_xyz_m": (oracle_drive_target_w[0] - official_start_w[0])
            .detach()
            .cpu()
            .tolist(),
            "official_start_restored_within_m": float(
                torch.linalg.vector_norm(initial_plug_w - official_start_w, dim=-1).max()
            ),
            "drive_ramp_s": self.cfg.goal_drive_ramp_s,
            "drive_hold_s": self.cfg.goal_drive_hold_s,
            "passive_settle_s": self.cfg.goal_passive_settle_s,
            "stability_window_s": self.cfg.goal_stability_window_s,
            "ramp_steps": ramp_steps,
            "drive_hold_steps": hold_steps,
            "passive_early_steps": early_steps,
            "passive_stability_steps": stability_steps,
            "insertion_travel_m": float(insertion_travel[0]),
            "runtime_nominal_target_error_m": float(target_error[0]),
            "drive_release_to_goal_plug_drift_m": float(
                torch.linalg.vector_norm(goal_q[0, 0, :3] - release_q[0, 0, :3])
            ),
            "last_window_plug_drift_m": float(last_window_plug_drift[0]),
            "last_window_max_task_body_drift_m": float(last_window_task_drift[0]),
            "last_window_max_plug_latch_drift_m": float(last_window_connector_drift[0]),
            "passive_final_plug_speed": float(passive_plug_speed[0]),
            "peak_plug_relative_latch_angle_rad": float(peak_latch_angle[0]),
            "passive_final_plug_relative_latch_angle_rad": float(passive_latch_angle[0]),
            "franka_grasp_orientation_xyzw": NOMINAL_GRASP_QUAT_XYZW,
            "franka_open_approach_s": self.cfg.goal_grasp_approach_s,
            "franka_closed_hold_s": self.cfg.goal_grasp_hold_s,
            "franka_closed_goal_plug_drift_m": float(grasp_plug_drift[0]),
            "franka_closed_goal_max_task_body_drift_m": float(grasp_task_drift[0]),
            "franka_closed_goal_max_plug_latch_drift_m": float(grasp_connector_drift[0]),
            "franka_closed_goal_target_error_m": float(final_target_error[0]),
            "franka_closed_goal_plug_speed": float(final_plug_speed[0]),
            "franka_closed_goal_plug_relative_latch_angle_rad": float(warm_goal_latch_angle[0]),
            "franka_closed_goal_tcp_distance_m": float(goal_grasp.tcp_distance[0]),
            "franka_closed_goal_minimum_bilateral_deflection_m": float(goal_grasp.bilateral_deflection[0]),
            "franka_closed_goal_invalid_contacts": int(goal_collision.invalid_contact_count[0]),
            "franka_closed_goal_left_proxy_contacts": int(goal_collision.left_grasp_contact_count[0]),
            "franka_closed_goal_right_proxy_contacts": int(goal_collision.right_grasp_contact_count[0]),
            "franka_target_update_algorithm": "bounded_iterative_cartesian_residual_compensation",
            "franka_compensation_iterations": int(goal_settlement["iterations"]),
            "franka_preclose_tcp_error_m": float(goal_settlement["final_tcp_error"][0]),
            "franka_preclose_live_tcp_grasp_distance_m": float(preclose_live_grasp_distance[0]),
            "franka_preclose_arm_target_tracking_error": float(goal_settlement["arm_tracking_error"][0]),
            "franka_cartesian_target_bias_m": float(torch.linalg.vector_norm(goal_settlement["cartesian_bias"][0])),
            "franka_compensation_error_history_m": goal_settlement["error_history"][0].cpu().tolist(),
            "cold_start_fixed_point_converged": True,
            "cold_start_fixed_point_cycles": len(cold_goal_history),
            "cold_start_fixed_point_history": cold_goal_history,
            "cold_start_maximum_warm_to_canonical_body_translation_m": float(warm_to_canonical_body_drift.max()),
            "cold_start_worst_warm_to_canonical_body_index": int(warm_to_canonical_body_drift[0].argmax()),
            "canonical_capture_maximum_cable_linear_speed_m_s": float(canonical_cable_speed.max()),
            "canonical_capture_fastest_cable_segment_by_world": canonical_fastest_cable.cpu().tolist(),
            "canonical_capture_runtime_nominal_target_error_m": float(canonical_target_error[0]),
            "canonical_capture_plug_relative_latch_angle_rad": float(canonical_latch_angle[0]),
            "cold_fixed_point_plug_relative_latch_angle_rad": float(cold_fixed_point_latch_angle[0]),
            "cold_replay_plug_relative_latch_angle_rad": float(cold_replay_latch_angle[0]),
            "canonical_capture_exact_runtime_success": bool(exact_goal_metrics["stored_capture_success"].all()),
            "canonical_exact_success_dwell_passed": bool(exact_goal_success.all()),
            "canonical_exact_success_all_post_step_samples": bool(exact_goal_metrics["all_post_step_success"].all()),
            "canonical_exact_success_required_dwell_steps": int(exact_goal_metrics["required_dwell_steps"]),
            "canonical_exact_success_sample_steps": int(exact_goal_metrics["sample_steps"]),
            "canonical_exact_success_maximum_signed_axial_error_m": float(
                exact_goal_metrics["maximum_signed_axial_error"].max()
            ),
            "canonical_exact_success_minimum_signed_axial_error_m": float(
                exact_goal_metrics["minimum_signed_axial_error"].min()
            ),
            "canonical_exact_success_maximum_axial_error_m": float(exact_goal_metrics["maximum_axial_error"].max()),
            "canonical_exact_success_maximum_radial_error_m": float(exact_goal_metrics["maximum_radial_error"].max()),
            "canonical_exact_success_maximum_plug_angle_error_rad": float(
                exact_goal_metrics["maximum_plug_angle_error"].max()
            ),
            "canonical_exact_success_maximum_latch_angle_error_rad": float(
                exact_goal_metrics["maximum_latch_angle_error"].max()
            ),
            "canonical_exact_success_maximum_plug_spatial_speed": float(
                exact_goal_metrics["maximum_plug_spatial_speed"].max()
            ),
            "replicated_goal_position_spread_m": float(goal_position_spread),
            "drive_was_disabled_for_evidence": True,
            "release_max_task_linear_speed_m_s": float(torch.linalg.vector_norm(release_qd[..., :3], dim=-1).amax()),
        }
        print(
            "[RJ45 GOAL] 35 mm drive complete; "
            f"runtime nominal target error={evidence['runtime_nominal_target_error_m']:.6f} m, "
            f"last-2s plug drift={evidence['last_window_plug_drift_m']:.6f} m",
            flush=True,
        )
        return {
            name: value[0].detach().cpu().to(torch.float32).contiguous() for name, value in canonical_goal_state.items()
        }, evidence

    def _sample_candidate_targets(
        self,
        phase: int,
        goal_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lower, upper = _PHASE_DISTANCE_RANGES[phase]
        distance = lower + (upper - lower) * torch.rand(self.env.num_envs, device=self.device, generator=self.random)
        candidate_plug = goal_q[0, :3].repeat(self.env.num_envs, 1)
        candidate_plug[:, 1] -= distance
        lateral_scale = self.cfg.connector_lateral_jitter_m * (0.5 + 0.125 * phase)
        lateral = 2.0 * torch.rand((self.env.num_envs, 2), device=self.device, generator=self.random) - 1.0
        lateral *= lateral_scale
        candidate_plug[:, 0] += lateral[:, 0]
        candidate_plug[:, 2] += lateral[:, 1]
        return candidate_plug, distance, lateral

    def _tcp_targets(
        self,
        plug_position: torch.Tensor,
        plug_quaternion: torch.Tensor,
        orientation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grasp_offset = torch.as_tensor(self.env.cfg.plug_grasp_offset, device=self.device).expand(self.env.num_envs, -1)
        position = plug_position + math_utils.quat_apply(plug_quaternion, grasp_offset)
        position += (
            2.0 * torch.rand(position.shape, device=self.device, generator=self.random) - 1.0
        ) * self.cfg.tcp_position_jitter_m
        return position, orientation

    def _record_rejections(self, phase: int, checks: dict[str, torch.Tensor], valid: torch.Tensor) -> None:
        for name, check in checks.items():
            self.rejection_counts[phase][name] += int((~check).sum().item())
        self.rejection_counts[phase]["accepted"] += int(valid.sum().item())

    @torch.inference_mode()
    def _generate_phase(self, phase: int, goal_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        required = self.cfg.rows_per_phase
        accepted: list[dict[str, torch.Tensor]] = []
        accepted_count = 0
        goal_q = goal_state["task_body_pose"].to(self.device)
        for _batch in range(self.cfg.max_batches_per_phase):
            if accepted_count >= required:
                break
            self._park_robot_and_restore_task()
            default_q, _ = self.env.read_task_state()
            orientation = randomized_orientations(
                self.base_orientation,
                max_angle=self.cfg.tcp_rotation_jitter_rad,
                generator=self.random,
            )
            candidate_plug, requested_distance, lateral = self._sample_candidate_targets(phase, goal_q)
            start_tcp, _ = self._tcp_targets(default_q[:, 0, :3], default_q[:, 0, 3:7], orientation)
            candidate_tcp, _ = self._tcp_targets(
                candidate_plug,
                goal_q[0, 3:7].repeat(self.env.num_envs, 1),
                orientation,
            )
            start_ik = self.ik.solve(
                start_tcp,
                orientation,
                self.open_finger_q,
                arm_seed=self.nominal_grasp_arm_seed,
            )

            start_settlement = self._approach_tcp_with_compensation(
                start_tcp,
                orientation,
                start_ik,
            )
            preclose_collision = collision_metrics(self.env, require_bilateral_grasp=False)
            preclose_live_grasp_distance = torch.linalg.vector_norm(
                self.env.tcp_pose_e()[:, :3] - self.env.plug_grasp_position_e(), dim=-1
            )
            preclose_valid = (
                start_ik.valid
                & start_settlement["valid"]
                & preclose_collision.valid
                & joint_limit_mask(self.env, start_settlement["arm_target"])
                & (preclose_live_grasp_distance <= self.cfg.compensation_tolerance_m)
            )
            closed_finger_target = torch.where(
                preclose_valid[:, None],
                self.finger_target,
                self.open_finger_q,
            )
            self._close_gripper(start_settlement["arm_target"], closed_finger_target)
            pre_drive_grasp = grasp_metrics(self.env, closed_finger_target)
            pre_drive_collision = collision_metrics(self.env)
            compensated_candidate_tcp = candidate_tcp + start_settlement["cartesian_bias"]
            candidate_ik = self.ik.solve(
                compensated_candidate_tcp,
                orientation,
                self.open_finger_q,
                arm_seed=start_settlement["arm_target"],
            )
            task_start_q, _ = self.env.read_task_state()
            start_plug_w = task_start_q[:, 0, :3] + self.env.env_origins
            candidate_plug_w = candidate_plug + self.env.env_origins
            wiggle_direction = torch.zeros_like(candidate_plug_w)
            wiggle_direction[:, 0] = lateral[:, 1]
            wiggle_direction[:, 2] = -lateral[:, 0]
            norm = torch.linalg.vector_norm(wiggle_direction, dim=-1, keepdim=True)
            fallback = torch.zeros_like(wiggle_direction)
            fallback[:, 0] = 1.0
            wiggle_direction = torch.where(norm > 1.0e-8, wiggle_direction / norm.clamp_min(1.0e-8), fallback)

            def drive_candidate(_step: int, _steps: int, progress: float) -> None:
                blend = progress * progress * (3.0 - 2.0 * progress)
                target = torch.lerp(start_plug_w, candidate_plug_w, blend)
                target += wiggle_direction * (self.cfg.cable_wiggle_m * math.sin(math.pi * progress))
                self.env.set_drive(True, target)
                self.env.set_robot_targets(
                    torch.lerp(start_settlement["arm_target"], candidate_ik.arm_q, blend),
                    closed_finger_target,
                )

            self.env.set_drive(True, start_plug_w)
            self.env.advance(self.cfg.candidate_drive_s, drive_candidate)
            self.env.set_drive(False)
            self._assert_drive_disabled("settling a candidate reset state")
            self.env.set_robot_targets(candidate_ik.arm_q, closed_finger_target)
            self.env.advance(self.cfg.candidate_grasp_settle_s)
            self._assert_drive_disabled("capturing a candidate reset state")

            candidate_settlement = self._stabilize_closed_candidate(
                candidate_plug,
                orientation,
                closed_finger_target,
                initial_arm_target=candidate_ik.arm_q,
                initial_command_ik_q=candidate_ik.arm_q,
                initial_commanded_tcp=compensated_candidate_tcp,
            )

            warm_candidate_state = self._capture_full_state(
                candidate_settlement["arm_target"],
                closed_finger_target,
                context="capturing a warm candidate for cold-start canonicalization",
            )
            canonical_state, canonical_post, cold_start_stable, cold_start_history = self._cold_start_canonicalize(
                warm_candidate_state,
                context=f"{_PHASE_NAMES[phase]} candidate batch {_batch + 1}",
                settle_s=self.cfg.cold_start_candidate_settle_s,
                maximum_plug_drift_m=self.cfg.cold_start_candidate_max_plug_drift_m,
                maximum_connector_drift_m=self.cfg.cold_start_candidate_max_connector_drift_m,
                maximum_task_body_drift_m=self.cfg.cold_start_candidate_max_task_drift_m,
                maximum_cable_linear_speed_m_s=self.cfg.cold_start_candidate_max_cable_linear_speed_m_s,
            )
            task_q = canonical_state["task_body_pose"]
            task_qd = canonical_state["task_body_velocity"]
            arm_q = canonical_state["arm_joint_position"]
            initial_error = scalar_goal_error(task_q, goal_q)
            axial_error = torch.abs(task_q[:, 0, 1] - goal_q[0, 1])
            grasp = grasp_metrics(self.env, closed_finger_target)
            collision = collision_metrics(self.env)
            capture_cable_speed = torch.linalg.vector_norm(task_qd[:, 2:, :3], dim=-1).amax(dim=-1)
            lower, upper = _PHASE_DISTANCE_RANGES[phase]
            checks = {
                "start_ik": start_ik.valid,
                "start_compensation": start_settlement["valid"],
                "preclose_tcp_to_live_grasp": (preclose_live_grasp_distance <= self.cfg.compensation_tolerance_m),
                "preclose_clearance": preclose_collision.valid,
                "candidate_ik": candidate_ik.valid,
                "candidate_equilibrium": candidate_settlement["valid"],
                "arm_target_limits": joint_limit_mask(self.env, canonical_state["arm_joint_target"]),
                "nominal_closed_grasp": pre_drive_grasp.valid,
                "pre_drive_collisions": pre_drive_collision.valid,
                "cold_start_fixed_point": cold_start_stable,
                "finite_task": task_state_is_finite_and_normalized(task_q, task_qd),
                "joint_limits": joint_limit_mask(self.env, arm_q),
                "closed_grasp": grasp.valid,
                "initial_collisions": collision.valid,
                "phase_distance": (axial_error >= lower - 5.0e-4) & (axial_error <= upper + 5.0e-4),
                "workspace": (
                    (task_q[:, 0, :3] >= self.env._task_workspace_lower)
                    & (task_q[:, 0, :3] <= self.env._task_workspace_upper)
                ).all(dim=-1),
                "settled_plug_linear_speed": (torch.linalg.vector_norm(task_qd[:, 0, :3], dim=-1) <= 0.08),
                "settled_cable_linear_speed": (
                    capture_cable_speed <= self.cfg.cold_start_candidate_max_cable_linear_speed_m_s
                ),
            }
            pre_recovery_valid = torch.stack(tuple(checks.values())).all(dim=0)

            # Preserve actual snapshots before the scripted robot-only
            # recoverability rollout mutates simulation state.
            snapshot = {
                **{name: value.clone() for name, value in canonical_state.items()},
                "phase": torch.full((self.env.num_envs,), phase, device=self.device, dtype=torch.int64),
                "difficulty": (initial_error / 0.025).clamp(0.0, 1.0),
                "initial_goal_error": initial_error,
                "progress_threshold": (0.25 * initial_error).clamp(2.0e-4, 0.005),
            }
            recovery_success, recovery_metrics = scripted_recovery(
                self.env,
                self.recovery_ik,
                goal_q,
                self.base_orientation,
                canonical_post["finger_joint_target"],
                arm_target_start=canonical_post["arm_joint_target"],
                goal_arm_target=goal_state["arm_joint_target"].to(self.device),
                motion_s=self.cfg.recovery_motion_s,
                settle_s=self.cfg.recovery_settle_s,
                compensation_max_iterations=self.cfg.recovery_compensation_max_iterations,
                compensation_gain=self.cfg.recovery_compensation_gain,
                compensation_max_step_m=self.cfg.recovery_compensation_max_step_m,
                compensation_motion_s=self.cfg.recovery_compensation_motion_s,
                compensation_hold_s=self.cfg.recovery_compensation_hold_s,
                compensation_tolerance_m=self.cfg.recovery_compensation_tolerance_m,
                maximum_ik_joint_step_rad=self.cfg.maximum_ik_joint_step_rad,
            )
            print(
                f"[RJ45 RECOVERY] {_PHASE_NAMES[phase]} batch {_batch + 1}: "
                f"goal_error={recovery_metrics['goal_error'].tolist()}, "
                f"overtravel={recovery_metrics['overtravel_distance'].tolist()}, "
                f"tcp_distance={recovery_metrics['tcp_grasp_distance'].tolist()}, "
                f"tcp_goal_error={recovery_metrics['tcp_goal_position_error'].tolist()}, "
                f"compensation_iterations={recovery_metrics['compensation_iterations'].tolist()}, "
                f"goal_error_history={recovery_metrics['goal_error_history'].tolist()}, "
                f"exact_capture={recovery_metrics['exact_success_stored_capture_success'].tolist()}, "
                f"exact_dwell={recovery_metrics['exact_success_dwell_satisfied'].tolist()}, "
                f"exact_axial={recovery_metrics['exact_success_final_signed_axial_error'].tolist()}, "
                f"exact_radial={recovery_metrics['exact_success_final_radial_error'].tolist()}, "
                f"exact_spatial_speed={recovery_metrics['exact_success_final_plug_spatial_speed'].tolist()}, "
                f"ik_valid={recovery_metrics['ik_valid'].tolist()}, "
                f"invalid_contacts={recovery_metrics['invalid_contact_count'].tolist()}, "
                f"invalid_pairs={recovery_metrics['invalid_contact_pairs']}",
                flush=True,
            )
            checks["scripted_recovery"] = recovery_success
            valid = pre_recovery_valid & recovery_success
            batch_valid_count = int(valid.sum().item())
            self._record_rejections(phase, checks, valid)
            self.attempt_counts[phase] += self.env.num_envs
            remaining = required - accepted_count
            chosen = torch.where(valid)[0][:remaining]
            if chosen.numel():
                accepted.append({name: value[chosen].detach().clone() for name, value in snapshot.items()})
                self.cold_start_accepted_cycles[phase].extend([len(cold_start_history)] * int(chosen.numel()))
                worst_body_by_world = cold_start_history[-1]["worst_task_body_index_by_world"]
                self.cold_start_accepted_worst_body_indices[phase].extend(
                    [int(worst_body_by_world[int(index)]) for index in chosen]
                )
                accepted_count += int(chosen.numel())
            requested_mm = tuple(
                round(1000.0 * float(value), 3)
                for value in (requested_distance.min(), requested_distance.median(), requested_distance.max())
            )
            settled_axial_mm = tuple(
                round(1000.0 * float(value), 3)
                for value in (axial_error.min(), axial_error.median(), axial_error.max())
            )
            candidate_equilibrium_mm = tuple(
                round(1000.0 * float(value), 3)
                for value in (
                    candidate_settlement["final_plug_error"].min(),
                    candidate_settlement["final_plug_error"].median(),
                    candidate_settlement["final_plug_error"].max(),
                )
            )
            batch_checks = ", ".join(f"{name}: {int(check.sum().item())}" for name, check in checks.items())
            print(
                f"[RJ45 RESET] {_PHASE_NAMES[phase]} accepted {accepted_count}/{required} "
                f"after {int(self.attempt_counts[phase])} simulated candidates "
                f"(batch_valid={batch_valid_count}/{self.env.num_envs}, "
                f"requested_mm={requested_mm}, settled_axial_mm={settled_axial_mm}, "
                f"candidate_equilibrium_mm={candidate_equilibrium_mm}, "
                f"batch_checks={{{batch_checks}}})",
                flush=True,
            )

        if accepted_count != required:
            raise RuntimeError(
                f"Generation exhausted {_PHASE_NAMES[phase]} after {int(self.attempt_counts[phase])} "
                f"candidates, accepting {accepted_count}/{required}; "
                f"rejections={dict(self.rejection_counts[phase])}."
            )
        return {name: torch.cat([part[name] for part in accepted], dim=0) for name in RESET_DATASET_STATE_NAMES}

    @torch.inference_mode()
    def generate(self) -> dict[str, Any]:
        grasp_evidence = self.nominal_grasp_smoke()
        goal_state, goal_evidence = self.derive_goal()
        phase_parts = [self._generate_phase(phase, goal_state) for phase in range(len(_PHASE_NAMES))]
        states = {name: torch.cat([part[name] for part in phase_parts], dim=0) for name in RESET_DATASET_STATE_NAMES}
        permutation = torch.randperm(len(states["phase"]), device=self.device, generator=self.random)
        cpu_states = {name: states[name][permutation].detach().cpu().contiguous() for name in RESET_DATASET_STATE_NAMES}
        contract = reset_dataset_task_contract(self.env.cfg)
        payload: dict[str, Any] = {
            "format": FRANKA_RJ45_RESET_DATASET_FORMAT,
            "schema_version": FRANKA_RJ45_RESET_DATASET_SCHEMA_VERSION,
            "metadata": {
                "task_contract": contract,
                "generator_cfg": asdict(self.cfg),
                "seed": self.cfg.seed,
                "phase_names": _PHASE_NAMES,
                "phase_distance_ranges_m": _PHASE_DISTANCE_RANGES,
                "phase_counts": torch.full((len(_PHASE_NAMES),), self.cfg.rows_per_phase, dtype=torch.int64),
                "attempt_counts": self.attempt_counts.clone(),
                "rejection_counts": {
                    _PHASE_NAMES[phase]: {
                        name: int(count) for name, count in sorted(self.rejection_counts[phase].items())
                    }
                    for phase in range(len(_PHASE_NAMES))
                },
                "goal_validation": goal_evidence,
                "nominal_grasp_validation": grasp_evidence,
                "construction_drive_policy": {
                    "construction_only": True,
                    "authored_start_hold_during_robot_park_s": self.cfg.construction_robot_park_s,
                    "authored_start_hold_during_open_approach_and_close": True,
                    "candidate_equilibrium_method": ("drive-release-closed-grasp-incremental-ik-residual-compensation"),
                    "candidate_equilibrium_drive_disabled": True,
                    "candidate_equilibrium_tolerance_m": self.cfg.candidate_compensation_tolerance_m,
                    "maximum_ik_joint_continuation_step_rad": self.cfg.maximum_ik_joint_step_rad,
                    "disabled_before_every_captured_goal_or_reset_snapshot": True,
                },
                "cold_start_canonicalization": {
                    "method": "restore-both-buffers-invalidate-history-settle-recapture-to-fixed-point",
                    "maximum_cycles": self.cfg.cold_start_max_cycles,
                    "goal_all_body_translation_gate_m": self.cfg.cold_start_goal_max_task_drift_m,
                    "candidate_all_body_translation_gate_m": self.cfg.cold_start_candidate_max_task_drift_m,
                    "goal_cable_linear_speed_gate_m_s": self.cfg.cold_start_goal_max_cable_linear_speed_m_s,
                    "candidate_cable_linear_speed_gate_m_s": (self.cfg.cold_start_candidate_max_cable_linear_speed_m_s),
                    "accepted_cycles_by_phase": {
                        _PHASE_NAMES[phase]: tuple(self.cold_start_accepted_cycles[phase])
                        for phase in range(len(_PHASE_NAMES))
                    },
                    "accepted_worst_body_indices_by_phase": {
                        _PHASE_NAMES[phase]: tuple(self.cold_start_accepted_worst_body_indices[phase])
                        for phase in range(len(_PHASE_NAMES))
                    },
                },
                "goal_is_fixed": True,
                "initial_states_are_randomized": True,
                "physics_versions": package_versions(),
            },
            "states": cpu_states,
            "goal_state": goal_state,
        }
        payload["content_sha256"] = reset_dataset_content_digest(payload)
        reset_dataset_validate_runtime(payload, expected_task_contract=contract)
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rows-per-phase", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches-per-phase", type=int, default=64)
    parser.add_argument("--validate", action="store_true", help="Replay the saved artifact before returning success.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Generate one row per phase as a deterministic smoke test.",
    )
    add_launcher_args(parser)
    args = parser.parse_args()

    rows_per_phase = 1 if args.quick else args.rows_per_phase
    batch_size = 5 if args.quick else args.batch_size
    max_batches = min(6, args.max_batches_per_phase) if args.quick else args.max_batches_per_phase
    cfg = GeneratorCfg(
        rows_per_phase=rows_per_phase,
        batch_size=batch_size,
        seed=args.seed,
        max_batches_per_phase=max_batches,
    )
    env_cfg = FrankaRJ45InsertionEnvCfg()
    env_cfg.scene.num_envs = cfg.batch_size
    env_cfg.sim.device = args.device
    env_cfg.seed = cfg.seed
    env_cfg.validate_config()

    payload: dict[str, Any] | None = None
    with launch_simulation(env_cfg, args):
        try:
            env = RJ45ResetToolEnv(env_cfg)
        except Exception as exc:
            raise RuntimeError(
                "Franka RJ45 reset generation could not construct the real coupled environment. "
                "Verify the task asset, PHYSICS_READY runtime binding, Newton callbacks, and GPU configuration."
            ) from exc
        try:
            payload = ResetDatasetGenerator(env, cfg).generate()
            save_torch_atomic(payload, args.output)
            if args.validate:
                from validate_franka_rj45_resets import ValidationCfg, validate_payload, write_validation_report

                report = validate_payload(
                    env,
                    payload,
                    ValidationCfg(seed=args.seed, quick=args.quick),
                )
                report_path = write_validation_report(report)
                if not report["passed"]:
                    raise RuntimeError(f"Generated artifact failed replay validation; see {report_path}.")
                print(f"[INFO] Validation report: {report_path}")
        finally:
            env.close()

    assert payload is not None
    output = args.output.expanduser().resolve()
    print(f"[INFO] Wrote {len(payload['states']['phase'])} reset rows to {output}.")
    print(f"[INFO] Content SHA-256: {payload['content_sha256']}")
    print(f"[INFO] Fixed goal passive evidence: {payload['metadata']['goal_validation']}")


if __name__ == "__main__":
    main()
