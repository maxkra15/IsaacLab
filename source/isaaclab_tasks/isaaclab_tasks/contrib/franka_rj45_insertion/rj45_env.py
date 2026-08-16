# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based reset-driven Franka RJ45 insertion environment."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab_newton.cloner import newton_builder_world_hook
from isaaclab_newton.physics import NewtonManager

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.physics import PhysicsEvent
from isaaclab.utils import math as math_utils

from .reset_dataset_io import (
    RESET_DATASET_STATE_NAMES,
    reset_dataset_validate_runtime,
    reset_validation_report_validate_runtime,
)
from .rj45_env_cfg import configure_rj45_capacities, reset_dataset_task_contract
from .task_success import rj45_insertion_success

if TYPE_CHECKING:
    from .rj45_env_cfg import FrankaRJ45InsertionEnvCfg

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[5]
_GENERATOR_COMMAND = (
    "uv run python scripts/tools/generate_franka_rj45_reset_dataset.py --headless --device cuda:0 && "
    "uv run python scripts/tools/validate_franka_rj45_resets.py --headless --device cuda:0"
)
ARM_JOINTS = [f"panda_joint{index}" for index in range(1, 8)]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
TERMINAL_OUTCOME_NAMES = ("success", "lost_grasp", "nonfinite", "task_out_of_bounds", "time_out")


def _resolve_reset_dataset_path(configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Franka RJ45 reset dataset not found: {path}. Generate and validate it from the repository root with: "
            f"{_GENERATOR_COMMAND}"
        )
    return path


def _resolve_reset_validation_report_path(configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Franka RJ45 full reset-validation report not found: {path}. Generate it from the repository root "
            f"with: {_GENERATOR_COMMAND}"
        )
    return path


class FrankaRJ45InsertionEnv(ManagerBasedRLEnv):
    """Franka inserting the plug of Newton's compliant, mechanically latched RJ45 cable."""

    cfg: FrankaRJ45InsertionEnvCfg

    def __init__(self, cfg: FrankaRJ45InsertionEnvCfg, render_mode: str | None = None, **kwargs):
        configure_rj45_capacities(cfg)
        # Import lazily so config/schema tests do not compile Warp kernels or load the USD asset.
        from .physics.rj45_assembly import Rj45NewtonAssemblyBuilder

        self._rj45_builder = Rj45NewtonAssemblyBuilder()
        self._rj45_runtime = None
        self._rj45_task_translation = tuple(cfg.task_translation)
        self._rj45_task_rotation = tuple(cfg.task_rotation_xyzw)
        self._rj45_physics_ready_handle = NewtonManager.register_callback(
            self._bind_rj45_physics_ready,
            PhysicsEvent.PHYSICS_READY,
            order=-10,
            name="bind_rj45_runtime",
        )
        self._rj45_state_force_callback = self._prepare_rj45_substep
        self._rj45_post_step_callback = self._align_rj45_after_step
        NewtonManager.register_state_force_callback(self._rj45_state_force_callback)
        NewtonManager.register_post_step_callback(self._rj45_post_step_callback)
        try:
            with newton_builder_world_hook(self._add_rj45_world_to_builder):
                super().__init__(cfg, render_mode, **kwargs)
        except Exception:
            self._clear_rj45_callbacks()
            raise

    def _clear_rj45_callbacks(self) -> None:
        """Release task callbacks after normal close or partial initialization."""
        handle = getattr(self, "_rj45_physics_ready_handle", None)
        if handle is not None:
            handle.deregister()
            self._rj45_physics_ready_handle = None
        state_force_callback = getattr(self, "_rj45_state_force_callback", None)
        if state_force_callback is not None:
            NewtonManager.unregister_state_force_callback(state_force_callback)
            self._rj45_state_force_callback = None
        post_step_callback = getattr(self, "_rj45_post_step_callback", None)
        if post_step_callback is not None:
            NewtonManager.unregister_post_step_callback(post_step_callback)
            self._rj45_post_step_callback = None

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._clear_rj45_callbacks()

    def _add_rj45_world_to_builder(self, builder, env_id: int, position, quaternion) -> None:
        """Place the source asset at the fixed task transform within each environment."""
        if tuple(self._rj45_task_rotation) != (0.0, 0.0, 0.0, 1.0):
            raise RuntimeError("RJ45 task rotation must be identity.")
        task_position = [float(position[index]) + self._rj45_task_translation[index] for index in range(3)]
        self._rj45_builder.world_hook(builder, env_id, task_position, quaternion)

    def _ensure_rj45_runtime(self):
        """Return runtime arrays preallocated at ``PHYSICS_READY``."""
        if self._rj45_runtime is None:
            raise RuntimeError("RJ45 runtime was not bound before Newton solver capture.")
        return self._rj45_runtime

    def _bind_rj45_physics_ready(self, _payload=None) -> None:
        """Allocate task runtime arrays after model finalization and before capture."""
        self._rj45_runtime = self._rj45_builder.bind(NewtonManager.get_model())

    def _prepare_rj45_substep(self, state) -> None:
        """Faithfully align the previous solve, apply forces, and sync four anchors."""
        runtime = self._ensure_rj45_runtime()
        runtime.align_after_step(state)
        runtime.prepare_step(state)

    def _align_rj45_after_step(self) -> None:
        """Align the final substep so observations and rendering see correct capsules."""
        self._ensure_rj45_runtime().align_after_step(NewtonManager.get_state_0())

    def load_managers(self) -> None:
        """Bind raw Newton task bodies and load reset rows before curricula construct."""
        self._setup_after_physics()
        super().load_managers()

    def _setup_after_physics(self) -> None:
        self._bind_physics_state()
        self._load_reset_dataset(self.cfg.reset_dataset_path)
        self.reset_dataset_row_id = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.episode_succeeded = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # Same-step autoreset clears stateful termination terms before callers
        # can inspect them. Preserve the just-finished row and term buffers at
        # the reset boundary for deterministic evaluation/diagnostics.
        self.last_terminal_row_id = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.last_terminal_outcomes = {
            name: torch.zeros(self.num_envs, device=self.device, dtype=torch.bool) for name in TERMINAL_OUTCOME_NAMES
        }
        self._success_dwell_steps = max(
            1, math.ceil(float(self.cfg.success_dwell_time_s) / max(float(self.step_dt), 1.0e-6))
        )
        self._success_dwell_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _bind_physics_state(self) -> None:
        """Bind robot and task body indices without requiring a reset artifact.

        Offline generation subclasses call this method while intentionally
        skipping the RL managers and dataset loader.
        """
        self._robot = self.scene["robot"]
        arm_ids, _ = self._robot.find_joints(ARM_JOINTS, preserve_order=True, as_proxy=True)
        finger_ids, _ = self._robot.find_joints(FINGER_JOINTS, preserve_order=True, as_proxy=True)
        self._arm_joint_ids = arm_ids.torch
        self._finger_joint_ids = finger_ids.torch
        tcp_ids, _ = self._robot.find_bodies(self.cfg.tcp_body_name)
        if len(tcp_ids) != 1:
            raise RuntimeError(f"Expected one TCP body {self.cfg.tcp_body_name!r}, found {len(tcp_ids)}.")
        self._tcp_body_idx = tcp_ids[0]
        self.env_origins = self.scene.env_origins.to(device=self.device, dtype=torch.float32)
        self._tcp_offset_pos = torch.as_tensor(self.cfg.tcp_offset_pos, device=self.device).repeat(self.num_envs, 1)
        self._tcp_offset_quat = torch.as_tensor(self.cfg.tcp_offset_rot, device=self.device).repeat(self.num_envs, 1)
        self._plug_grasp_offset = torch.as_tensor(self.cfg.plug_grasp_offset, device=self.device).repeat(
            self.num_envs, 1
        )
        self._task_workspace_lower = torch.as_tensor(
            self.cfg.task_workspace_lower, device=self.device, dtype=torch.float32
        )
        self._task_workspace_upper = torch.as_tensor(
            self.cfg.task_workspace_upper, device=self.device, dtype=torch.float32
        )
        self._task_body_workspace_lower = torch.as_tensor(
            self.cfg.task_body_workspace_lower, device=self.device, dtype=torch.float32
        )
        self._task_body_workspace_upper = torch.as_tensor(
            self.cfg.task_body_workspace_upper, device=self.device, dtype=torch.float32
        )
        self._cable_observation_body_indices = torch.tensor(
            (2, 6, 11, 16, 21, 28, 36), device=self.device, dtype=torch.long
        )
        runtime = self._ensure_rj45_runtime()
        self._task_body_ids = wp.to_torch(runtime.task_body_ids).to(device=self.device, dtype=torch.long)
        if tuple(self._task_body_ids.shape) != (self.num_envs, 37):
            raise RuntimeError(f"RJ45 assembly body map must have shape ({self.num_envs}, 37).")
        self._task_reset_pose_staging = torch.zeros((self.num_envs, 37, 7), device=self.device, dtype=torch.float32)
        self._task_reset_velocity_staging = torch.zeros((self.num_envs, 37, 6), device=self.device, dtype=torch.float32)
        self._task_env_mask_staging = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _load_reset_dataset(self, configured_path: str) -> None:
        path = _resolve_reset_dataset_path(configured_path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        task_contract = reset_dataset_task_contract(self.cfg)
        _, states, goal = reset_dataset_validate_runtime(
            payload,
            expected_content_sha256=self.cfg.reset_dataset_content_sha256,
            expected_task_contract=task_contract,
        )
        report_path = _resolve_reset_validation_report_path(self.cfg.reset_validation_report_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        reset_validation_report_validate_runtime(
            report,
            expected_content_sha256=payload["content_sha256"],
            expected_row_count=len(states["phase"]),
            expected_task_contract=task_contract,
        )
        self._reset_dataset_states = {
            name: states[name].to(device=self.device, non_blocking=True) for name in RESET_DATASET_STATE_NAMES
        }
        self.goal_task_body_pose = goal["task_body_pose"].to(device=self.device, non_blocking=True)
        self.goal_task_body_velocity = goal["task_body_velocity"].to(device=self.device, non_blocking=True)
        logger.info(
            "Loaded %d physically validated RJ45 resets from %s (evidence: %s).",
            len(states["phase"]),
            path,
            report_path,
        )

    def _reset_idx(self, env_ids) -> None:
        """Capture terminal causes before manager resets mutate their state."""
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if ids.numel() and hasattr(self, "last_terminal_outcomes"):
            active_terms = set(self.termination_manager.active_terms)
            missing = set(TERMINAL_OUTCOME_NAMES) - active_terms
            if not missing:
                self.last_terminal_row_id[ids] = self.reset_dataset_row_id[ids]
                for name in TERMINAL_OUTCOME_NAMES:
                    self.last_terminal_outcomes[name][ids] = self.termination_manager.get_term(name)[ids]
        super()._reset_idx(env_ids)

    def _newton_state_tensors(self, *, secondary: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        state = NewtonManager.get_state_1() if secondary else NewtonManager.get_state_0()
        if state is None or state.body_q is None or state.body_qd is None:
            raise RuntimeError("Newton body state is unavailable.")
        return wp.to_torch(state.body_q), wp.to_torch(state.body_qd)

    def task_body_pose_w(self) -> torch.Tensor:
        body_q, _ = self._newton_state_tensors()
        return body_q[self._task_body_ids]

    def task_body_pose_e(self) -> torch.Tensor:
        pose = self.task_body_pose_w().clone()
        pose[..., :3] -= self.env_origins[:, None, :]
        return pose

    def task_body_velocity(self) -> torch.Tensor:
        _, body_qd = self._newton_state_tensors()
        return body_qd[self._task_body_ids]

    def snapshot_task_state_e(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Clone all persisted task bodies in the reset artifact's frame."""
        return self.task_body_pose_e().clone(), self.task_body_velocity().clone()

    def plug_pose_e(self) -> torch.Tensor:
        return self.task_body_pose_e()[:, 0]

    def latch_pose_e(self) -> torch.Tensor:
        return self.task_body_pose_e()[:, 1]

    def plug_goal_translation_error(self) -> torch.Tensor:
        """Return signed plug translation error in the fixed socket frame."""
        return self.plug_pose_e()[:, :3] - self.goal_task_body_pose[0, :3]

    def tcp_pose_e(self) -> torch.Tensor:
        hand = self._robot.data.body_link_pose_w.torch[:, self._tcp_body_idx]
        pos, quat = math_utils.combine_frame_transforms(
            hand[:, :3], hand[:, 3:7], self._tcp_offset_pos, self._tcp_offset_quat
        )
        return torch.cat((pos - self.env_origins, quat), dim=-1)

    def plug_grasp_position_e(self) -> torch.Tensor:
        plug = self.plug_pose_e()
        return plug[:, :3] + math_utils.quat_apply(plug[:, 3:7], self._plug_grasp_offset)

    def goal_error_components(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        position_error = self.plug_goal_translation_error()
        axial = position_error[:, 1].abs()
        radial_xz = position_error[:, (0, 2)]
        latch = self.latch_pose_e()
        goal_latch = self.goal_task_body_pose[1]
        goal_inverse = math_utils.quat_conjugate(goal_latch[3:7].repeat(self.num_envs, 1))
        latch_error = math_utils.quat_unique(math_utils.quat_mul(goal_inverse, latch[:, 3:7]))
        latch_angle = torch.linalg.vector_norm(math_utils.axis_angle_from_quat(latch_error), dim=-1)
        return axial, radial_xz, latch_angle

    def plug_orientation_error(self) -> torch.Tensor:
        """Return the shortest plug-to-fixed-goal rotation angle [rad]."""
        plug = self.plug_pose_e()
        goal_plug = self.goal_task_body_pose[0]
        goal_inverse = math_utils.quat_conjugate(goal_plug[3:7].repeat(self.num_envs, 1))
        error = math_utils.quat_unique(math_utils.quat_mul(goal_inverse, plug[:, 3:7]))
        return torch.linalg.vector_norm(math_utils.axis_angle_from_quat(error), dim=-1)

    def scalar_goal_error(self) -> torch.Tensor:
        axial, radial_xz, latch_angle = self.goal_error_components()
        return axial + 2.0 * torch.linalg.vector_norm(radial_xz, dim=-1) + 0.002 * latch_angle

    def insertion_success_mask(self) -> torch.Tensor:
        result = rj45_insertion_success(
            self.task_body_pose_e(),
            self.task_body_velocity(),
            self.goal_task_body_pose,
            axial_tolerance=self.cfg.success_axial_tolerance,
            axial_overtravel_tolerance=self.cfg.success_axial_overtravel_tolerance,
            radial_tolerance=self.cfg.success_radial_tolerance,
            plug_angle_tolerance=self.cfg.success_plug_angle_tolerance,
            latch_angle_tolerance=self.cfg.success_latch_angle_tolerance,
            maximum_plug_spatial_speed=self.cfg.success_max_plug_speed,
        )
        return result.mask

    def set_insertion_drive(
        self,
        enabled: bool,
        target_position_e: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Control the generator-only plug drive without changing RL state."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        self._task_env_mask_staging.zero_()
        self._task_env_mask_staging[env_ids] = True
        mask = wp.from_torch(self._task_env_mask_staging, dtype=wp.bool)
        runtime = self._ensure_rj45_runtime()
        if target_position_e is not None:
            target_position_e = torch.as_tensor(target_position_e, device=self.device, dtype=torch.float32)
            if target_position_e.shape == (3,):
                target_position_e = target_position_e.repeat(self.num_envs, 1)
            if target_position_e.shape != (self.num_envs, 3):
                raise ValueError(f"Drive target must have shape (3,) or ({self.num_envs}, 3).")
            target_w = target_position_e + self.env_origins
            runtime.write_drive_targets(wp.from_torch(target_w, dtype=wp.vec3), mask)
        runtime.set_drive_enabled(bool(enabled), mask)

    def _write_task_state(self, env_ids: torch.Tensor, rows: torch.Tensor) -> None:
        self.write_task_state_e(
            self._reset_dataset_states["task_body_pose"][rows],
            self._reset_dataset_states["task_body_velocity"][rows],
            env_ids,
        )

    def write_task_state_e(
        self,
        local_pose: torch.Tensor,
        velocity: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Restore complete RJ45 body state in selected environment frames."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        expected_pose_shape = (env_ids.numel(), 37, 7)
        expected_velocity_shape = (env_ids.numel(), 37, 6)
        if tuple(local_pose.shape) != expected_pose_shape or tuple(velocity.shape) != expected_velocity_shape:
            raise ValueError(
                f"RJ45 state must have shapes {expected_pose_shape}/{expected_velocity_shape}, got "
                f"{tuple(local_pose.shape)}/{tuple(velocity.shape)}."
            )
        pose_w = local_pose.clone()
        pose_w[..., :3] += self.env_origins[env_ids, None, :]
        self._task_reset_pose_staging[env_ids] = pose_w
        self._task_reset_velocity_staging[env_ids] = velocity
        self._task_env_mask_staging.zero_()
        self._task_env_mask_staging[env_ids] = True
        runtime = self._ensure_rj45_runtime()
        state_1 = NewtonManager.get_state_1()
        if state_1 is None:
            raise RuntimeError("RJ45 VBD coupling requires a secondary Newton state buffer.")
        runtime.write_state(
            (NewtonManager.get_state_0(), state_1),
            wp.from_torch(self._task_reset_pose_staging, dtype=wp.transform),
            wp.from_torch(self._task_reset_velocity_staging, dtype=wp.spatial_vector),
            wp.from_torch(self._task_env_mask_staging, dtype=wp.bool),
        )
        NewtonManager.invalidate_body_state(env_ids=wp.from_torch(env_ids.to(dtype=torch.int32), dtype=wp.int32))

    def _write_robot_state(self, env_ids: torch.Tensor, rows: torch.Tensor) -> None:
        states = self._reset_dataset_states
        arm_q = states["arm_joint_position"][rows]
        arm_qd = states["arm_joint_velocity"][rows]
        arm_target = states["arm_joint_target"][rows]
        finger_q = states["finger_joint_position"][rows]
        finger_qd = states["finger_joint_velocity"][rows]
        finger_target = states["finger_joint_target"][rows]
        self._robot.write_joint_state_to_sim_index(
            position=arm_q, velocity=arm_qd, joint_ids=self._arm_joint_ids, env_ids=env_ids
        )
        self._robot.set_joint_position_target_index(target=arm_target, joint_ids=self._arm_joint_ids, env_ids=env_ids)
        self.action_manager.get_term("arm_action").set_reset_target(arm_target, arm_q, env_ids=env_ids)
        self._robot.write_joint_state_to_sim_index(
            position=finger_q,
            velocity=finger_qd,
            joint_ids=self._finger_joint_ids,
            env_ids=env_ids,
        )
        self._robot.set_joint_position_target_index(
            target=finger_target, joint_ids=self._finger_joint_ids, env_ids=env_ids
        )
        self.action_manager.get_term("gripper_action").set_reset_position(finger_target[:, :1], env_ids=env_ids)
        # Materialize FK before VBD's robot proxies consume the reset hand/finger transforms.
        _ = self._robot.data.body_link_pose_w

    def reset_rj45_scene(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if env_ids.numel() == 0:
            return
        rows = self.reset_dataset_row_id[env_ids]
        if bool(torch.any((rows < 0) | (rows >= len(self._reset_dataset_states["phase"])))):
            raise RuntimeError("Curriculum assigned an invalid RJ45 reset row.")
        self.set_insertion_drive(False, env_ids=env_ids)
        self._write_task_state(env_ids, rows)
        self._write_robot_state(env_ids, rows)
        self.episode_succeeded[env_ids] = False
        self._success_dwell_count[env_ids] = 0
