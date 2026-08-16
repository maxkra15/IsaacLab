# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay and validate a Franka RJ45 reset artifact in coupled simulation.

By default every row is restored, its solver/contact history is cleared, and
it is physically settled before a closed-grasp Franka recovery rollout.  The
fixed goal is tested separately for ten drive-free simulated seconds.  A JSON
report is always written for completed validation runs; unavailable task or
physics APIs fail before producing evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import warp as wp
from _franka_rj45_reset_tools import (
    DEFAULT_DATASET_PATH,
    DEFAULT_VALIDATION_DIR,
    NOMINAL_GRASP_QUAT_XYZW,
    FrankaResetIK,
    RJ45ResetToolEnv,
    advance_exact_success_dwell,
    advance_reset_bias_hold,
    collision_metrics,
    grasp_metrics,
    joint_limit_mask,
    package_versions,
    plug_relative_latch_angle,
    scalar_goal_error,
    scripted_recovery,
    task_state_is_finite_and_normalized,
)

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_contrib.coupling import NewtonCouplerManager

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.franka_rj45_insertion.reset_dataset_io import (
    FRANKA_RJ45_RESET_VALIDATION_FORMAT,
    FRANKA_RJ45_RESET_VALIDATION_SCHEMA_VERSION,
    reset_dataset_validate_runtime,
    reset_validation_report_validate_runtime,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.rj45_env_cfg import (
    RIGID_ENTRY,
    RJ45_ENTRY,
    FrankaRJ45InsertionEnvCfg,
    reset_dataset_task_contract,
)


@dataclass(frozen=True)
class ValidationCfg:
    """Deterministic replay thresholds and durations."""

    seed: int = 2026
    quick: bool = False
    sample_count: int | None = None
    goal_passive_s: float = 10.0
    row_settle_s: float = 0.5
    recovery_motion_s: float = 2.0
    recovery_settle_s: float = 0.5
    recovery_compensation_max_iterations: int = 5
    recovery_compensation_gain: float = 1.0
    recovery_compensation_max_step_m: float = 0.006
    recovery_compensation_motion_s: float = 0.35
    recovery_compensation_hold_s: float = 0.25
    recovery_compensation_tolerance_m: float = 0.0015
    maximum_goal_plug_drift_m: float = 5.0e-4
    maximum_goal_connector_drift_m: float = 1.0e-3
    maximum_goal_task_body_drift_m: float = 5.0e-3
    maximum_row_plug_drift_m: float = 3.0e-3
    maximum_row_task_body_drift_m: float = 5.0e-3
    maximum_settled_plug_speed: float = 0.08
    maximum_goal_cable_linear_speed_m_s: float = 0.01
    maximum_row_cable_linear_speed_m_s: float = 0.02
    maximum_goal_plug_relative_latch_angle_rad: float = 0.05

    def __post_init__(self) -> None:
        if self.sample_count is not None and self.sample_count <= 0:
            raise ValueError("sample_count must be positive when provided.")
        if self.goal_passive_s < 10.0:
            raise ValueError("Goal replay must remain passive for at least ten simulated seconds.")
        for name in ("row_settle_s", "recovery_motion_s", "recovery_settle_s"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive.")


def _selected_rows(states: dict[str, torch.Tensor], cfg: ValidationCfg) -> torch.Tensor:
    row_count = len(states["phase"])
    if cfg.quick:
        chosen = []
        for phase in range(5):
            rows = torch.where(states["phase"] == phase)[0]
            if rows.numel():
                chosen.append(rows[0])
        if not chosen:
            raise RuntimeError("Quick validation could not select any phase rows.")
        return torch.stack(chosen).long()
    if cfg.sample_count is None or cfg.sample_count >= row_count:
        return torch.arange(row_count, dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    return torch.randperm(row_count, generator=generator)[: cfg.sample_count]


def _contact_count() -> int:
    from isaaclab_newton.physics import NewtonManager

    proxy_contacts, _, _ = NewtonCouplerManager.get_proxy_contact_data(RIGID_ENTRY, RJ45_ENTRY)
    count = 0
    for contacts in (NewtonManager.get_contacts(), proxy_contacts):
        if contacts is not None and contacts.rigid_contact_count is not None:
            count += int(wp.to_torch(contacts.rigid_contact_count)[0].item())
    return count


def _goal_replay(
    env: RJ45ResetToolEnv,
    goal: dict[str, torch.Tensor],
    cfg: ValidationCfg,
) -> dict[str, Any]:
    goal_q = goal["task_body_pose"]
    goal_qd = goal["task_body_velocity"]
    repeated_q = goal_q.unsqueeze(0).repeat(env.num_envs, 1, 1)
    repeated_qd = goal_qd.unsqueeze(0).repeat(env.num_envs, 1, 1)
    arm_q = goal["arm_joint_position"].unsqueeze(0).repeat(env.num_envs, 1)
    arm_target = goal["arm_joint_target"].unsqueeze(0).repeat(env.num_envs, 1)
    arm_qd = goal["arm_joint_velocity"].unsqueeze(0).repeat(env.num_envs, 1)
    finger_q = goal["finger_joint_position"].unsqueeze(0).repeat(env.num_envs, 1)
    finger_qd = goal["finger_joint_velocity"].unsqueeze(0).repeat(env.num_envs, 1)
    finger_target = goal["finger_joint_target"].unsqueeze(0).repeat(env.num_envs, 1)
    env.write_task_state(repeated_q, repeated_qd)
    env.write_robot_state(
        arm_q,
        finger_q,
        arm_target=arm_target,
        arm_qd=arm_qd,
        finger_qd=finger_qd,
        finger_target=finger_target,
    )
    env.set_drive(False)
    env.flush_reset_history()
    contact_count_after_reset = _contact_count()
    start_q, start_qd = env.read_task_state()
    arm_target_bias = arm_target - arm_q
    exact_success, exact_metrics = advance_exact_success_dwell(
        env,
        goal_q,
        arm_target_bias,
        finger_target,
        duration_s=cfg.goal_passive_s,
        require_all_samples=True,
    )
    steps = int(exact_metrics["sample_steps"])
    final_q, final_qd = env.read_task_state()
    plug_drift = torch.linalg.vector_norm(final_q[:, 0, :3] - start_q[:, 0, :3], dim=-1)
    body_drift = torch.linalg.vector_norm(final_q[..., :3] - start_q[..., :3], dim=-1)
    task_drift, worst_body = body_drift.max(dim=-1)
    connector_drift = torch.linalg.vector_norm(final_q[:, :2, :3] - start_q[:, :2, :3], dim=-1).amax(dim=-1)
    replay_error = scalar_goal_error(final_q, goal_q)
    plug_speed = torch.linalg.vector_norm(final_qd[:, 0, :3], dim=-1)
    start_cable_speed, start_fastest_cable = torch.linalg.vector_norm(start_qd[:, 2:, :3], dim=-1).max(dim=-1)
    final_cable_speed, final_fastest_cable = torch.linalg.vector_norm(final_qd[:, 2:, :3], dim=-1).max(dim=-1)
    start_latch_angle = plug_relative_latch_angle(start_q)
    final_latch_angle = plug_relative_latch_angle(final_q)
    runtime_nominal_target_w = wp.to_torch(env.rj45_runtime.default_goal_target_w)
    runtime_nominal_target_e = runtime_nominal_target_w - env.env_origins
    runtime_nominal_target_error = torch.linalg.vector_norm(final_q[:, 0, :3] - runtime_nominal_target_e, dim=-1)
    finite = task_state_is_finite_and_normalized(final_q, final_qd)
    final_arm_q, _, _, _ = env.read_robot_state()
    grasp = grasp_metrics(env, finger_target)
    collision = collision_metrics(env)
    joints_valid = joint_limit_mask(env, final_arm_q)
    final_arm_target = final_arm_q + arm_target_bias
    targets_valid = joint_limit_mask(env, arm_target) & joint_limit_mask(env, final_arm_target)
    drive_disabled = not bool(wp.to_torch(env.rj45_runtime.drive_enabled).any())
    passed = (
        finite
        & grasp.valid
        & collision.valid
        & joints_valid
        & targets_valid
        & exact_success
        & (plug_drift <= cfg.maximum_goal_plug_drift_m)
        & (connector_drift <= cfg.maximum_goal_connector_drift_m)
        & (task_drift <= cfg.maximum_goal_task_body_drift_m)
        & (replay_error <= 0.001)
        & (runtime_nominal_target_error <= 0.0015)
        & (plug_speed <= 0.01)
        & (start_cable_speed <= cfg.maximum_goal_cable_linear_speed_m_s)
        & (final_cable_speed <= cfg.maximum_goal_cable_linear_speed_m_s)
        & (start_latch_angle <= cfg.maximum_goal_plug_relative_latch_angle_rad)
        & (final_latch_angle <= cfg.maximum_goal_plug_relative_latch_angle_rad)
    )
    return {
        "passed": bool(passed.all()) and drive_disabled and contact_count_after_reset == 0,
        "duration_s": cfg.goal_passive_s,
        "simulation_steps": steps,
        "simulation_time_s": steps * env.advance_dt,
        "drive_disabled": drive_disabled,
        "contact_count_after_history_reset": contact_count_after_reset,
        "stored_capture_exact_runtime_success": bool(exact_metrics["stored_capture_success"].all()),
        "all_post_step_exact_runtime_success": bool(exact_metrics["all_post_step_success"].all()),
        "exact_runtime_success_dwell_satisfied": bool(exact_metrics["dwell_satisfied"].all()),
        "exact_runtime_success_required_dwell_steps": int(exact_metrics["required_dwell_steps"]),
        "exact_runtime_success_final_consecutive_steps_by_world": exact_metrics["final_consecutive_steps"]
        .cpu()
        .tolist(),
        "exact_runtime_success_maximum_signed_axial_error_m": float(exact_metrics["maximum_signed_axial_error"].max()),
        "exact_runtime_success_minimum_signed_axial_error_m": float(exact_metrics["minimum_signed_axial_error"].min()),
        "exact_runtime_success_maximum_axial_error_m": float(exact_metrics["maximum_axial_error"].max()),
        "exact_runtime_success_maximum_radial_error_m": float(exact_metrics["maximum_radial_error"].max()),
        "exact_runtime_success_maximum_plug_angle_error_rad": float(exact_metrics["maximum_plug_angle_error"].max()),
        "exact_runtime_success_maximum_latch_angle_error_rad": float(exact_metrics["maximum_latch_angle_error"].max()),
        "exact_runtime_success_maximum_plug_spatial_speed": float(exact_metrics["maximum_plug_spatial_speed"].max()),
        "maximum_plug_drift_m": float(plug_drift.max()),
        "maximum_plug_latch_drift_m": float(connector_drift.max()),
        "maximum_task_body_drift_m": float(task_drift.max()),
        "worst_task_body_index_by_world": worst_body.cpu().tolist(),
        "maximum_replay_goal_error": float(replay_error.max()),
        "maximum_runtime_nominal_target_error_m": float(runtime_nominal_target_error.max()),
        "maximum_plug_speed": float(plug_speed.max()),
        "maximum_start_cable_linear_speed_m_s": float(start_cable_speed.max()),
        "start_fastest_cable_segment_by_world": start_fastest_cable.cpu().tolist(),
        "maximum_final_cable_linear_speed_m_s": float(final_cable_speed.max()),
        "final_fastest_cable_segment_by_world": final_fastest_cable.cpu().tolist(),
        "maximum_capture_plug_relative_latch_angle_rad": float(start_latch_angle.max()),
        "maximum_final_plug_relative_latch_angle_rad": float(final_latch_angle.max()),
        "finite_and_unit_quaternions": bool(finite.all()),
        "closed_bilateral_grasp": bool(grasp.valid.all()),
        "maximum_tcp_grasp_distance_m": float(grasp.tcp_distance.max()),
        "minimum_bilateral_finger_deflection_m": float(grasp.bilateral_deflection.min()),
        "joint_limits": bool(joints_valid.all()),
        "arm_target_limits": bool(targets_valid.all()),
        "maximum_arm_target_bias": float(torch.linalg.vector_norm(arm_target_bias, dim=-1).max()),
        "collision_constraints": bool(collision.valid.all()),
        "invalid_contact_count": int(collision.invalid_contact_count.sum()),
        "allowed_grasp_contact_count": int(collision.grasp_contact_count.sum()),
        "left_grasp_contact_count": int(collision.left_grasp_contact_count.sum()),
        "right_grasp_contact_count": int(collision.right_grasp_contact_count.sum()),
    }


def _row_dict(
    *,
    row_id: int,
    phase: int,
    checks: dict[str, bool],
    initial_error_expected: float,
    initial_error_replayed: float,
    plug_drift: float,
    task_drift: float,
    worst_task_body_index: int,
    plug_speed: float,
    capture_cable_speed: float,
    capture_fastest_cable_segment: int,
    settled_cable_speed: float,
    settled_fastest_cable_segment: int,
    tcp_distance: float,
    bilateral_deflection: float,
    invalid_contacts: int,
    grasp_contacts: int,
    left_grasp_contacts: int,
    right_grasp_contacts: int,
    arm_target_bias: float,
    recovery_error: float,
    recovery_speed: float,
    recovery_exact_success: dict[str, Any],
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "phase": phase,
        "passed": all(checks.values()),
        "checks": checks,
        "initial_goal_error_expected": initial_error_expected,
        "initial_goal_error_replayed": initial_error_replayed,
        "settle_plug_drift_m": plug_drift,
        "settle_max_task_body_drift_m": task_drift,
        "settle_worst_task_body_index": worst_task_body_index,
        "settled_plug_speed": plug_speed,
        "capture_max_cable_linear_speed_m_s": capture_cable_speed,
        "capture_fastest_cable_segment": capture_fastest_cable_segment,
        "settled_max_cable_linear_speed_m_s": settled_cable_speed,
        "settled_fastest_cable_segment": settled_fastest_cable_segment,
        "tcp_grasp_distance_m": tcp_distance,
        "minimum_bilateral_finger_deflection_m": bilateral_deflection,
        "invalid_initial_contact_count": invalid_contacts,
        "allowed_grasp_contact_count": grasp_contacts,
        "left_grasp_contact_count": left_grasp_contacts,
        "right_grasp_contact_count": right_grasp_contacts,
        "arm_target_bias": arm_target_bias,
        "recovery_goal_error": recovery_error,
        "recovery_plug_speed": recovery_speed,
        "recovery_exact_runtime_success": recovery_exact_success,
    }


@torch.inference_mode()
def validate_payload(
    env: RJ45ResetToolEnv,
    payload: dict[str, Any],
    cfg: ValidationCfg,
) -> dict[str, Any]:
    """Validate one already-loaded artifact in a live tool environment."""
    metadata, states_raw, goal_raw = reset_dataset_validate_runtime(
        payload,
        expected_task_contract=reset_dataset_task_contract(env.cfg),
    )
    states = {name: tensor.detach().cpu() for name, tensor in states_raw.items()}
    goal = {name: tensor.to(device=env.device) for name, tensor in goal_raw.items()}
    goal_q = goal["task_body_pose"]
    selected = _selected_rows(states, cfg)
    goal_result = _goal_replay(env, goal, cfg)

    ik = FrankaResetIK(env, seed=cfg.seed + 1, seeds=1, noise_std=0.0, sampler="none")
    rows: list[dict[str, Any]] = []
    for begin in range(0, len(selected), env.num_envs):
        selected_batch = selected[begin : begin + env.num_envs]
        real_count = len(selected_batch)
        simulation_rows = selected_batch
        if real_count < env.num_envs:
            repetitions = math.ceil(env.num_envs / real_count)
            simulation_rows = selected_batch.repeat(repetitions)[: env.num_envs]

        task_q = states["task_body_pose"][simulation_rows].to(env.device)
        task_qd = states["task_body_velocity"][simulation_rows].to(env.device)
        arm_q = states["arm_joint_position"][simulation_rows].to(env.device)
        arm_target = states["arm_joint_target"][simulation_rows].to(env.device)
        arm_qd = states["arm_joint_velocity"][simulation_rows].to(env.device)
        finger_q = states["finger_joint_position"][simulation_rows].to(env.device)
        finger_qd = states["finger_joint_velocity"][simulation_rows].to(env.device)
        finger_target = states["finger_joint_target"][simulation_rows].to(env.device)
        env.write_task_state(task_q, task_qd)
        env.write_robot_state(
            arm_q,
            finger_q,
            arm_target=arm_target,
            arm_qd=arm_qd,
            finger_qd=finger_qd,
            finger_target=finger_target,
        )
        env.set_drive(False)
        env.flush_reset_history()
        reset_contact_count = _contact_count()
        replay_start_q, replay_start_qd = env.read_task_state()
        replay_initial_error = scalar_goal_error(replay_start_q, goal_q)
        immediate_finite = task_state_is_finite_and_normalized(replay_start_q, replay_start_qd)
        immediate_joint_limits = joint_limit_mask(env, arm_q)
        immediate_target_limits = joint_limit_mask(env, arm_target)
        arm_target_bias = arm_target - arm_q

        # One real physics step materializes the reset contact set; the rest
        # of the settle interval tests whether the state stays learnable.
        advance_reset_bias_hold(env, env.advance_dt, arm_target_bias, finger_target)
        initial_collision = collision_metrics(env)
        remaining_settle = max(0.0, cfg.row_settle_s - env.advance_dt)
        advance_reset_bias_hold(env, remaining_settle, arm_target_bias, finger_target)
        settled_q, settled_qd = env.read_task_state()
        settled_arm_q, _, _, _ = env.read_robot_state()
        plug_drift = torch.linalg.vector_norm(settled_q[:, 0, :3] - replay_start_q[:, 0, :3], dim=-1)
        body_drift = torch.linalg.vector_norm(settled_q[..., :3] - replay_start_q[..., :3], dim=-1)
        task_drift, worst_body = body_drift.max(dim=-1)
        plug_speed = torch.linalg.vector_norm(settled_qd[:, 0, :3], dim=-1)
        capture_cable_speed, capture_fastest_cable = torch.linalg.vector_norm(replay_start_qd[:, 2:, :3], dim=-1).max(
            dim=-1
        )
        settled_cable_speed, settled_fastest_cable = torch.linalg.vector_norm(settled_qd[:, 2:, :3], dim=-1).max(dim=-1)
        settled_finite = task_state_is_finite_and_normalized(settled_q, settled_qd)
        settled_joint_limits = joint_limit_mask(env, settled_arm_q)
        settled_target = settled_arm_q + arm_target_bias
        settled_target_limits = joint_limit_mask(env, settled_target)
        grasp = grasp_metrics(env, finger_target)
        settled_collision = collision_metrics(env)
        orientation = torch.tensor(NOMINAL_GRASP_QUAT_XYZW, device=env.device).repeat(env.num_envs, 1)
        recovery_success, recovery_metrics = scripted_recovery(
            env,
            ik,
            goal_q,
            orientation,
            finger_target,
            arm_target_start=settled_target,
            goal_arm_target=goal["arm_joint_target"].to(env.device),
            motion_s=cfg.recovery_motion_s,
            settle_s=cfg.recovery_settle_s,
            compensation_max_iterations=cfg.recovery_compensation_max_iterations,
            compensation_gain=cfg.recovery_compensation_gain,
            compensation_max_step_m=cfg.recovery_compensation_max_step_m,
            compensation_motion_s=cfg.recovery_compensation_motion_s,
            compensation_hold_s=cfg.recovery_compensation_hold_s,
            compensation_tolerance_m=cfg.recovery_compensation_tolerance_m,
        )

        for local in range(real_count):
            row_id = int(selected_batch[local])
            error_expected = float(states["initial_goal_error"][row_id])
            error_replayed = float(replay_initial_error[local])
            checks = {
                "solver_and_contact_history_reset": reset_contact_count == 0,
                "finite_and_unit_quaternions": bool(immediate_finite[local] & settled_finite[local]),
                "joint_limits": bool(immediate_joint_limits[local] & settled_joint_limits[local]),
                "arm_target_limits": bool(immediate_target_limits[local] & settled_target_limits[local]),
                "closed_bilateral_grasp": bool(grasp.valid[local]),
                "initial_collision_constraints": bool(initial_collision.valid[local]),
                "settled_collision_constraints": bool(settled_collision.valid[local]),
                "settle_stability": bool(
                    (plug_drift[local] <= cfg.maximum_row_plug_drift_m)
                    & (task_drift[local] <= cfg.maximum_row_task_body_drift_m)
                    & (plug_speed[local] <= cfg.maximum_settled_plug_speed)
                    & (capture_cable_speed[local] <= cfg.maximum_row_cable_linear_speed_m_s)
                    & (settled_cable_speed[local] <= cfg.maximum_row_cable_linear_speed_m_s)
                ),
                "artifact_error_matches_replay": abs(error_expected - error_replayed) <= 5.0e-4,
                "recovery_exact_capture_success": bool(recovery_metrics["exact_success_stored_capture_success"][local]),
                "recovery_exact_all_post_step_samples": bool(
                    recovery_metrics["exact_success_all_post_step_success"][local]
                ),
                "recovery_exact_success_dwell": bool(recovery_metrics["exact_success_dwell_satisfied"][local]),
                "scripted_robot_recovery": bool(recovery_success[local]),
            }
            recovery_exact = {
                "stored_capture_success": bool(recovery_metrics["exact_success_stored_capture_success"][local]),
                "all_post_step_samples_success": bool(recovery_metrics["exact_success_all_post_step_success"][local]),
                "dwell_satisfied": bool(recovery_metrics["exact_success_dwell_satisfied"][local]),
                "required_dwell_steps": int(recovery_metrics["exact_success_required_dwell_steps"]),
                "sample_steps": int(recovery_metrics["exact_success_sample_steps"]),
                "final_consecutive_steps": int(recovery_metrics["exact_success_final_consecutive_steps"][local]),
                "maximum_consecutive_steps": int(recovery_metrics["exact_success_maximum_consecutive_steps"][local]),
                "final_signed_axial_error_m": float(recovery_metrics["exact_success_final_signed_axial_error"][local]),
                "final_axial_error_m": float(recovery_metrics["exact_success_final_axial_error"][local]),
                "final_radial_error_m": float(recovery_metrics["exact_success_final_radial_error"][local]),
                "final_plug_angle_error_rad": float(recovery_metrics["exact_success_final_plug_angle_error"][local]),
                "final_latch_angle_error_rad": float(recovery_metrics["exact_success_final_latch_angle_error"][local]),
                "final_plug_spatial_speed": float(recovery_metrics["exact_success_final_plug_spatial_speed"][local]),
                "maximum_signed_axial_error_m": float(
                    recovery_metrics["exact_success_maximum_signed_axial_error"][local]
                ),
                "minimum_signed_axial_error_m": float(
                    recovery_metrics["exact_success_minimum_signed_axial_error"][local]
                ),
                "maximum_axial_error_m": float(recovery_metrics["exact_success_maximum_axial_error"][local]),
                "maximum_radial_error_m": float(recovery_metrics["exact_success_maximum_radial_error"][local]),
                "maximum_plug_angle_error_rad": float(
                    recovery_metrics["exact_success_maximum_plug_angle_error"][local]
                ),
                "maximum_latch_angle_error_rad": float(
                    recovery_metrics["exact_success_maximum_latch_angle_error"][local]
                ),
                "maximum_plug_spatial_speed": float(
                    recovery_metrics["exact_success_maximum_plug_spatial_speed"][local]
                ),
            }
            rows.append(
                _row_dict(
                    row_id=row_id,
                    phase=int(states["phase"][row_id]),
                    checks=checks,
                    initial_error_expected=error_expected,
                    initial_error_replayed=error_replayed,
                    plug_drift=float(plug_drift[local]),
                    task_drift=float(task_drift[local]),
                    worst_task_body_index=int(worst_body[local]),
                    plug_speed=float(plug_speed[local]),
                    capture_cable_speed=float(capture_cable_speed[local]),
                    capture_fastest_cable_segment=int(capture_fastest_cable[local]),
                    settled_cable_speed=float(settled_cable_speed[local]),
                    settled_fastest_cable_segment=int(settled_fastest_cable[local]),
                    tcp_distance=float(grasp.tcp_distance[local]),
                    bilateral_deflection=float(grasp.bilateral_deflection[local]),
                    invalid_contacts=int(initial_collision.invalid_contact_count[local]),
                    grasp_contacts=int(initial_collision.grasp_contact_count[local]),
                    left_grasp_contacts=int(initial_collision.left_grasp_contact_count[local]),
                    right_grasp_contacts=int(initial_collision.right_grasp_contact_count[local]),
                    arm_target_bias=float(torch.linalg.vector_norm(arm_target[local] - arm_q[local])),
                    recovery_error=float(recovery_metrics["goal_error"][local]),
                    recovery_speed=float(recovery_metrics["plug_speed"][local]),
                    recovery_exact_success=recovery_exact,
                )
            )

    failed_rows = [row["row_id"] for row in rows if not row["passed"]]
    all_rows_selected = len(selected) == len(states["phase"])
    report: dict[str, Any] = {
        "format": FRANKA_RJ45_RESET_VALIDATION_FORMAT,
        "schema_version": FRANKA_RJ45_RESET_VALIDATION_SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "artifact_content_sha256": payload["content_sha256"],
        "task_contract": dict(metadata["task_contract"]),
        "validation_cfg": asdict(cfg),
        "physics_versions": package_versions(),
        "quick": cfg.quick,
        "full_dataset_replay": all_rows_selected,
        "evidence_complete": all_rows_selected and not cfg.quick,
        "selected_row_count": len(selected),
        "dataset_row_count": len(states["phase"]),
        "selected_row_ids": selected.tolist(),
        "goal_replay": goal_result,
        "rows": rows,
        "failed_row_ids": failed_rows,
        "passed": bool(goal_result["passed"]) and not failed_rows,
    }
    return report


def write_validation_report(report: dict[str, Any], output_dir: Path = DEFAULT_VALIDATION_DIR) -> Path:
    """Atomically write a timestamped JSON report below the project log root."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    digest = str(report.get("artifact_content_sha256", "unknown"))[:12]
    suffix = "quick" if report.get("quick") else "full"
    output = output_dir / f"reset_validation_{stamp}_{digest}_{suffix}.json"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output_dir)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def write_stable_validation_report(
    report: dict[str, Any],
    payload: dict[str, Any],
    env_cfg: FrankaRJ45InsertionEnvCfg,
    output: Path,
) -> Path:
    """Atomically publish the canonical gate only after complete evidence validates."""
    row_count = len(payload["states"]["phase"])
    reset_validation_report_validate_runtime(
        report,
        expected_content_sha256=payload["content_sha256"],
        expected_row_count=row_count,
        expected_task_contract=reset_dataset_task_contract(env_cfg),
    )
    output = output.expanduser()
    if not output.is_absolute():
        output = (Path(__file__).resolve().parents[2] / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument(
        "--stable-output",
        type=Path,
        help="Canonical full-pass report (defaults to the task config path).",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--quick", action="store_true", help="Replay one deterministic row from each phase.")
    add_launcher_args(parser)
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Franka RJ45 reset artifact does not exist: {input_path}")
    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    _, states, _ = reset_dataset_validate_runtime(payload)
    validation_cfg = ValidationCfg(seed=args.seed, quick=args.quick, sample_count=args.sample_count)
    selected_count = len(_selected_rows(dict(states), validation_cfg))
    batch_size = min(max(1, args.batch_size), selected_count)

    env_cfg = FrankaRJ45InsertionEnvCfg()
    env_cfg.scene.num_envs = batch_size
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    env_cfg.validate_config()
    report: dict[str, Any] | None = None
    with launch_simulation(env_cfg, args):
        try:
            env = RJ45ResetToolEnv(env_cfg)
        except Exception as exc:
            raise RuntimeError(
                "Franka RJ45 reset validation could not construct the real coupled environment. "
                "No validation evidence was emitted. Verify the task asset, PHYSICS_READY runtime binding, "
                "Newton callbacks, and GPU configuration."
            ) from exc
        try:
            report = validate_payload(env, payload, validation_cfg)
        finally:
            env.close()

    assert report is not None
    report_path = write_validation_report(report, args.output_dir)
    print(f"[INFO] Validation report: {report_path}")
    print(
        f"[INFO] Passed={report['passed']} goal={report['goal_replay']['passed']} "
        f"rows={report['selected_row_count'] - len(report['failed_row_ids'])}/{report['selected_row_count']}"
    )
    if report["passed"] and report["evidence_complete"] and report["full_dataset_replay"] and not report["quick"]:
        stable_output = args.stable_output or Path(env_cfg.reset_validation_report_path)
        stable_path = write_stable_validation_report(report, payload, env_cfg, stable_output)
        print(f"[INFO] Canonical validation gate: {stable_path}")
    if not report["passed"]:
        raise RuntimeError(
            f"Franka RJ45 reset validation failed for rows {report['failed_row_ids']}; see {report_path}."
        )


if __name__ == "__main__":
    main()
