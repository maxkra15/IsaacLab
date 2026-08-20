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
import stat
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

from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_env import FrankaRJ45PickInsertEnv
from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_reset_dataset_io import (
    PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY,
)
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


def _torch_artifact_values_equal(expected: Any, observed: Any) -> bool:
    """Return whether a CPU Torch reload exactly preserves a restricted payload."""
    if isinstance(expected, torch.Tensor):
        return (
            isinstance(observed, torch.Tensor)
            and expected.dtype == observed.dtype
            and tuple(expected.shape) == tuple(observed.shape)
            and torch.equal(expected.detach().cpu(), observed.detach().cpu())
        )
    if isinstance(expected, Mapping):
        return (
            isinstance(observed, Mapping)
            and tuple(expected) == tuple(observed)
            and all(_torch_artifact_values_equal(expected[name], observed[name]) for name in expected)
        )
    if isinstance(expected, tuple | list):
        return (
            type(expected) is type(observed)
            and len(expected) == len(observed)
            and all(
                _torch_artifact_values_equal(expected_item, observed_item)
                for expected_item, observed_item in zip(expected, observed, strict=True)
            )
        )
    return type(expected) is type(observed) and expected == observed


def _unlink_same_file(path: Path, identity: tuple[int, int]) -> None:
    """Unlink ``path`` only while it still names the temporary file we created."""
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return
    if (path_stat.st_dev, path_stat.st_ino) == identity:
        path.unlink()


def save_torch_atomic(payload: Mapping[str, Any], output: Path) -> None:
    """Durably replace a Torch artifact after a strict safe reload.

    The prior destination remains untouched until the same-directory temporary
    file has been flushed, synchronized, safely loaded on CPU, and compared
    exactly with the requested restricted payload.  If the parent-directory
    synchronization fails after replacement, the validated new inode may
    remain installed and callers must retry idempotently.
    """
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary_name)
    temporary_stat = os.fstat(descriptor)
    temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
    try:
        expected = dict(payload)
        with os.fdopen(descriptor, "w+b") as stream:
            descriptor = -1
            torch.save(expected, stream)
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            observed = torch.load(stream, map_location="cpu", weights_only=True)
        if not _torch_artifact_values_equal(expected, observed):
            raise RuntimeError("Torch artifact changed during its strict temporary-file reload.")
        reloaded_stat = temporary.lstat()
        if (
            not stat.S_ISREG(reloaded_stat.st_mode)
            or (reloaded_stat.st_dev, reloaded_stat.st_ino) != temporary_identity
            or reloaded_stat.st_nlink != 1
        ):
            raise RuntimeError("Torch artifact temporary-file identity changed before atomic publication.")
        os.replace(temporary, output)
        directory_descriptor = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        _unlink_same_file(temporary, temporary_identity)


def configured_arm_home(env: _ResetToolEnv) -> torch.Tensor:
    """Return the configured Panda home; backend default buffers are zero at tool startup."""
    try:
        values = [float(env.cfg.scene.robot.init_state.joint_pos[name]) for name in ARM_JOINTS]
    except KeyError as exc:
        raise RuntimeError(f"Franka RJ45 config is missing the explicit home joint {exc.args[0]!r}.") from exc
    return torch.tensor(values, device=env.device, dtype=torch.float32)


class _RJ45ResetToolMixin:
    """Offline mechanics shared by the insertion and pick-insert tool environments."""

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
        for name in (
            "layout",
            "task_body_ids",
            "default_body_q",
            "default_goal_target_w",
            "drive_enabled",
            "orientation_hold_enabled",
            "orientation_target_w",
        ):
            if not hasattr(runtime, name):
                raise RuntimeError(f"Franka RJ45 runtime is missing required physics field {name!r}.")
        task_body_count = int(runtime.layout.body_count)
        if tuple(runtime.task_body_ids.shape) != (self.num_envs, task_body_count):
            raise RuntimeError(
                f"Franka RJ45 runtime task-body map must have shape ({self.num_envs}, {task_body_count}), "
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
        *,
        post_step: Callable[[int, int, float], None] | None = None,
    ) -> int:
        """Advance actual coupled physics for at least ``duration_s``.

        ``update`` runs immediately before each real physics step and
        ``post_step`` runs after both the simulator step and scene update.  The
        latter is the evidence boundary for contact and velocity samples that
        must describe the state produced by that step.
        """
        if not math.isfinite(duration_s) or duration_s < 0.0:
            raise ValueError("Simulation duration must be finite and non-negative.")
        step_count = int(math.ceil(duration_s / self.advance_dt))
        for step in range(step_count):
            if update is not None:
                update(step, step_count, (step + 1) / max(step_count, 1))
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.advance_dt)
            finalize_pose_history = getattr(self, "finalize_pending_task_pose_history_restores", None)
            if callable(finalize_pose_history):
                finalize_pose_history(require_complete=True)
            if post_step is not None:
                post_step(step, step_count, (step + 1) / max(step_count, 1))
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

    def set_orientation_hold(self, enabled: bool, targets_w: torch.Tensor | None = None) -> None:
        """Set the graph-safe, construction-only plug orientation hold."""
        runtime = self.rj45_runtime
        mask = wp.ones(self.num_envs, dtype=wp.bool, device=self.device)
        if targets_w is not None:
            targets_w = torch.as_tensor(targets_w, device=self.device, dtype=torch.float32)
            if tuple(targets_w.shape) != (self.num_envs, 4):
                raise ValueError(f"Orientation targets must have shape ({self.num_envs}, 4).")
            runtime.write_orientation_hold_targets(
                wp.from_torch(targets_w.contiguous(), dtype=wp.quat),
                mask,
            )
        runtime.set_orientation_hold_enabled(bool(enabled), mask)

    def restore_default_task(self) -> None:
        """Restore all assembly bodies and clear VBD/contact history."""
        mask = wp.ones(self.num_envs, dtype=wp.bool, device=self.device)
        runtime = self.rj45_runtime
        runtime.reset_to_default(_newton_states(), mask)
        runtime.set_drive_enabled(False, mask)
        runtime.restore_goal_drive_targets(mask)
        runtime.restore_orientation_hold_targets(mask)
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
        task_body_count = int(self.rj45_runtime.layout.body_count)
        expected_q = (self.num_envs, task_body_count, 7)
        expected_qd = (self.num_envs, task_body_count, 6)
        if tuple(body_q.shape) != expected_q or tuple(body_qd.shape) != expected_qd:
            raise RuntimeError(
                f"Newton task state must have shapes {expected_q}/{expected_qd}, got "
                f"{tuple(body_q.shape)}/{tuple(body_qd.shape)}."
            )
        return body_q, body_qd

    def write_task_state(self, body_q_e: torch.Tensor, body_qd: torch.Tensor) -> None:
        """Write complete task state into both Newton state buffers."""
        body_q_e = torch.as_tensor(body_q_e, device=self.device, dtype=torch.float32)
        body_qd = torch.as_tensor(body_qd, device=self.device, dtype=torch.float32)
        task_body_count = int(self.rj45_runtime.layout.body_count)
        expected_q = (self.num_envs, task_body_count, 7)
        expected_qd = (self.num_envs, task_body_count, 6)
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
        plug_body_index = int(self.rj45_runtime.layout.plug_body_index)
        plug = self.read_task_state()[0][:, plug_body_index]
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


class RJ45ResetToolEnv(_RJ45ResetToolMixin, FrankaRJ45InsertionEnv):
    """Headless legacy insertion scene without reset artifacts or RL managers."""


class RJ45PickInsertResetToolEnv(_RJ45ResetToolMixin, FrankaRJ45PickInsertEnv):
    """Headless pick-insert scene without reset artifacts or RL managers."""

    def __init__(self, cfg, *args, grasp_proxy_friction: float | None = None, **kwargs) -> None:
        """Construct the tool with the immutable production grasp-proxy friction."""
        from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_env_cfg import (
            PICK_INSERT_GRASP_PROXY_FRICTION,
        )

        # ManagerBasedEnv.__del__ may run even when this preflight rejects the
        # configuration before the base constructor creates its close guard.
        self._is_closed = True
        configured = getattr(cfg, "grasp_proxy_friction", None)
        if (
            isinstance(configured, bool)
            or not isinstance(configured, int | float)
            or not math.isfinite(configured)
            or float(configured) != PICK_INSERT_GRASP_PROXY_FRICTION
        ):
            raise ValueError(
                "The pick-insert reset tool requires the exact production grasp-proxy friction "
                f"{PICK_INSERT_GRASP_PROXY_FRICTION}."
            )
        if grasp_proxy_friction is not None and (
            isinstance(grasp_proxy_friction, bool)
            or not isinstance(grasp_proxy_friction, int | float)
            or not math.isfinite(grasp_proxy_friction)
            or float(grasp_proxy_friction) != float(configured)
        ):
            raise ValueError("A reset-tool grasp-proxy friction override may only assert the production value.")
        self._reset_tool_grasp_proxy_friction = float(configured)
        super().__init__(cfg, *args, **kwargs)

    def _create_rj45_builder(self, cfg):
        """Forward the production pick-only proxy material into the offline builder."""
        from isaaclab_tasks.contrib.franka_rj45_insertion.physics import Rj45NewtonAssemblyBuilder
        from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_env_cfg import pick_insert_topology_cfg

        return Rj45NewtonAssemblyBuilder(
            topology_cfg=pick_insert_topology_cfg(cfg),
            task_translation=cfg.task_translation,
            task_rotation_xyzw=cfg.task_rotation_xyzw,
            grasp_proxy_friction=self._reset_tool_grasp_proxy_friction,
        )

    @property
    def grasp_proxy_friction(self) -> float:
        """Exact friction assigned only to the finger-only grasp proxy."""
        return float(self._rj45_builder.grasp_proxy_friction)


_ResetToolEnv = RJ45ResetToolEnv | RJ45PickInsertResetToolEnv


def pick_insert_tool_physical_contract(
    env: RJ45PickInsertResetToolEnv,
    *,
    finger_closed_target: float,
) -> dict[str, float]:
    """Validate and return the live production pick-insert tool contract."""
    from isaaclab_tasks.contrib.franka_rj45_insertion.physics import GRASP_FRICTION
    from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_env_cfg import (
        PICK_INSERT_CLOSED_FINGER_POSITION,
        PICK_INSERT_EFFECTIVE_GRASP_FRICTION,
        PICK_INSERT_GRASP_PROXY_FRICTION,
        PICK_INSERT_SUCCESS_MAX_PLUG_SPEED,
    )

    gripper_cfg = env.cfg.actions.gripper_action
    observed = {
        "finger_closed_target_m": float(finger_closed_target),
        "live_finger_close_position_m": float(gripper_cfg.close_position),
        "configured_grasp_proxy_raw_friction": float(env.cfg.grasp_proxy_friction),
        "live_grasp_proxy_raw_friction": float(env.grasp_proxy_friction),
        "effective_finger_proxy_friction": math.sqrt(GRASP_FRICTION * float(env.grasp_proxy_friction)),
        "success_max_plug_speed": float(env.cfg.success_max_plug_speed),
    }
    expected = {
        "finger_closed_target_m": PICK_INSERT_CLOSED_FINGER_POSITION,
        "live_finger_close_position_m": PICK_INSERT_CLOSED_FINGER_POSITION,
        "configured_grasp_proxy_raw_friction": PICK_INSERT_GRASP_PROXY_FRICTION,
        "live_grasp_proxy_raw_friction": PICK_INSERT_GRASP_PROXY_FRICTION,
        "effective_finger_proxy_friction": PICK_INSERT_EFFECTIVE_GRASP_FRICTION,
        "success_max_plug_speed": PICK_INSERT_SUCCESS_MAX_PLUG_SPEED,
    }
    mismatched = {name: value for name, value in observed.items() if value != expected[name]}
    if mismatched:
        raise ValueError(
            "The reset tool and live pick-insert environment must share the immutable production physical "
            f"contract: {mismatched}."
        )
    return observed


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


def _ik_seed_scores(
    costs: torch.Tensor,
    arm_q: torch.Tensor,
    selection_reference: torch.Tensor,
) -> torch.Tensor:
    """Score IK lanes against the caller's explicit continuation seed."""
    return costs + 1.0e-4 * torch.square(arm_q - selection_reference[:, None, :]).sum(-1)


class FrankaResetIK:
    """Batched Newton IK constrained to the task's Panda branch and limits."""

    def __init__(
        self,
        env: _ResetToolEnv,
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
        selection_reference = self.home_arm_q.expand(self.capacity, -1) if arm_seed is None else arm_seed
        score = _ik_seed_scores(costs, arm_q, selection_reference)
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


def batched_quat_slerp(q1: torch.Tensor, q2: torch.Tensor, tau: float) -> torch.Tensor:
    """Interpolate matching batches of XYZW quaternions along their shortest arcs."""
    q1 = torch.as_tensor(q1)
    q2 = torch.as_tensor(q2, device=q1.device, dtype=q1.dtype)
    if q1.shape != q2.shape or q1.ndim < 1 or q1.shape[-1] != 4:
        raise ValueError(f"Quaternion batches must have matching (..., 4) shapes, got {q1.shape} and {q2.shape}.")
    tau = float(tau)
    if not math.isfinite(tau) or not 0.0 <= tau <= 1.0:
        raise ValueError("Quaternion interpolation coefficient must be finite and lie in [0, 1].")
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    q2 = torch.nn.functional.normalize(q2, dim=-1)
    dot = (q1 * q2).sum(dim=-1, keepdim=True)
    q2 = torch.where(dot < 0.0, -q2, q2)
    dot = dot.abs().clamp(max=1.0)
    angle = torch.acos(dot)
    sin_angle = torch.sin(angle)
    denominator = sin_angle.clamp_min(torch.finfo(q1.dtype).eps)
    spherical = torch.sin((1.0 - tau) * angle) / denominator * q1 + torch.sin(tau * angle) / denominator * q2
    linear = (1.0 - tau) * q1 + tau * q2
    result = torch.where(dot > 1.0 - 4.0 * torch.finfo(q1.dtype).eps, linear, spherical)
    return torch.nn.functional.normalize(result, dim=-1)


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


def _batched_goal_task_pose(goal_q: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Return fixed or batched goal poses with the reference batch/device/dtype."""
    goal_q = torch.as_tensor(goal_q, device=reference.device, dtype=reference.dtype)
    batch_size = reference.shape[0]
    if goal_q.ndim == 2 and goal_q.shape[-1] == 7:
        return goal_q.unsqueeze(0).expand(batch_size, -1, -1)
    if goal_q.ndim == 3 and goal_q.shape[-1] == 7 and goal_q.shape[0] in (1, batch_size):
        return goal_q.expand(batch_size, -1, -1)
    raise ValueError(
        "Goal task poses must have shape (body_count, 7), (1, body_count, 7), or "
        f"({batch_size}, body_count, 7); got {tuple(goal_q.shape)}."
    )


def _validate_body_indices(body_count: int, plug_body_index: int, latch_body_index: int) -> None:
    for name, body_index in (("plug_body_index", plug_body_index), ("latch_body_index", latch_body_index)):
        if isinstance(body_index, bool) or not isinstance(body_index, int) or not 0 <= body_index < body_count:
            raise ValueError(f"{name} must index one of {body_count} task bodies, got {body_index!r}.")
    if plug_body_index == latch_body_index:
        raise ValueError("plug_body_index and latch_body_index must be distinct.")


def _resolve_layout_body_indices(
    env: _ResetToolEnv,
    plug_body_index: int | None,
    latch_body_index: int | None,
) -> tuple[int, int]:
    layout = env.rj45_runtime.layout
    resolved_plug = int(layout.plug_body_index) if plug_body_index is None else plug_body_index
    resolved_latch = int(layout.latch_body_index) if latch_body_index is None else latch_body_index
    _validate_body_indices(int(layout.body_count), resolved_plug, resolved_latch)
    return resolved_plug, resolved_latch


def scalar_goal_error(
    task_q: torch.Tensor,
    goal_q: torch.Tensor,
    *,
    plug_body_index: int = 0,
    latch_body_index: int = 1,
) -> torch.Tensor:
    """Return insertion error in each fixed or batched goal plug frame."""
    if task_q.ndim != 3 or task_q.shape[-1] != 7:
        raise ValueError(f"Task poses must have shape (batch_size, body_count, 7), got {tuple(task_q.shape)}.")
    _validate_body_indices(task_q.shape[1], plug_body_index, latch_body_index)
    goal = _batched_goal_task_pose(goal_q, task_q)
    _validate_body_indices(goal.shape[1], plug_body_index, latch_body_index)
    goal_plug = goal[:, plug_body_index]
    plug_error_w = task_q[:, plug_body_index, :3] - goal_plug[:, :3]
    plug_error = math_utils.quat_apply_inverse(goal_plug[:, 3:7], plug_error_w)
    axial = plug_error[:, 1].abs()
    radial = torch.linalg.vector_norm(plug_error[:, (0, 2)], dim=-1)
    goal_latch = goal[:, latch_body_index]
    latch_inverse = math_utils.quat_conjugate(goal_latch[:, 3:7])
    latch_error = math_utils.quat_unique(math_utils.quat_mul(latch_inverse, task_q[:, latch_body_index, 3:7]))
    latch_angle = torch.linalg.vector_norm(math_utils.axis_angle_from_quat(latch_error), dim=-1)
    return axial + 2.0 * radial + 0.002 * latch_angle


def plug_relative_latch_angle(
    task_q: torch.Tensor,
    *,
    plug_body_index: int = 0,
    latch_body_index: int = 1,
) -> torch.Tensor:
    """Return the unsigned plug-relative latch angle [rad]."""
    if task_q.ndim != 3 or task_q.shape[-1] != 7:
        raise ValueError(f"Task poses must have shape (batch_size, body_count, 7), got {tuple(task_q.shape)}.")
    _validate_body_indices(task_q.shape[1], plug_body_index, latch_body_index)
    plug_quat = task_q[:, plug_body_index, 3:7]
    latch_quat = task_q[:, latch_body_index, 3:7]
    relative = math_utils.quat_unique(math_utils.quat_mul(math_utils.quat_conjugate(plug_quat), latch_quat))
    return torch.linalg.vector_norm(math_utils.axis_angle_from_quat(relative), dim=-1)


def task_state_is_finite_and_normalized(task_q: torch.Tensor, task_qd: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(task_q).all(dim=(1, 2)) & torch.isfinite(task_qd).all(dim=(1, 2))
    quat_norm = torch.linalg.vector_norm(task_q[..., 3:7], dim=-1)
    return finite & (torch.abs(quat_norm - 1.0) <= 1.0e-3).all(dim=-1)


def joint_limit_mask(env: _ResetToolEnv, arm_q: torch.Tensor, margin: float | None = None) -> torch.Tensor:
    """Return worlds whose arm positions remain inside the configured soft-limit margin.

    Args:
        env: Reset-tool environment that owns the live action configuration.
        arm_q: Arm joint positions [rad], shape ``(..., joint_count)``.
        margin: Explicit soft-limit margin [rad]. Defaults to the live arm action margin.

    Returns:
        Per-world mask indicating whether all arm joints are within the reduced limits.
    """
    if margin is None:
        margin = float(env.cfg.actions.arm_action.joint_limit_margin)
    limits = env._robot.data.soft_joint_pos_limits.torch[0, env._arm_joint_ids]
    return ((arm_q >= limits[:, 0] + margin) & (arm_q <= limits[:, 1] - margin)).all(dim=-1)


def runtime_reset_biased_arm_target(
    env: _ResetToolEnv,
    current_arm_q: torch.Tensor,
    arm_target_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the live relative action's zero-action target and clamp.

    Returns the clamped target and the maximum per-world clamp magnitude.  The
    margin comes from the active action configuration so offline reset replay
    cannot silently diverge from training semantics.
    """
    current_arm_q = torch.as_tensor(current_arm_q, device=env.device, dtype=torch.float32)
    arm_target_bias = torch.as_tensor(arm_target_bias, device=env.device, dtype=torch.float32)
    expected_shape = (env.num_envs, len(env._arm_joint_ids))
    if tuple(current_arm_q.shape) != expected_shape or tuple(arm_target_bias.shape) != expected_shape:
        raise ValueError(
            "Current arm position and reset bias must both have shape "
            f"{expected_shape}, got {tuple(current_arm_q.shape)}/{tuple(arm_target_bias.shape)}."
        )
    margin = float(env.cfg.actions.arm_action.joint_limit_margin)
    limits = env._robot.data.soft_joint_pos_limits.torch[:, env._arm_joint_ids]
    unclamped = current_arm_q + arm_target_bias
    target = torch.clamp(
        unclamped,
        min=limits[..., 0] + margin,
        max=limits[..., 1] - margin,
    )
    return target, torch.abs(target - unclamped).amax(dim=-1)


def runtime_persistent_arm_target(
    env: _ResetToolEnv,
    arm_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the pick controller's constant absolute target and soft-limit clamp delta."""
    arm_target = torch.as_tensor(arm_target, device=env.device, dtype=torch.float32)
    expected_shape = (env.num_envs, len(env._arm_joint_ids))
    if tuple(arm_target.shape) != expected_shape:
        raise ValueError(f"arm_target must have shape {expected_shape}, got {tuple(arm_target.shape)}.")
    margin = float(env.cfg.actions.arm_action.joint_limit_margin)
    limits = env._robot.data.soft_joint_pos_limits.torch[:, env._arm_joint_ids]
    target = torch.clamp(
        arm_target,
        min=limits[..., 0] + margin,
        max=limits[..., 1] - margin,
    )
    return target, torch.abs(target - arm_target).amax(dim=-1)


@dataclass(frozen=True)
class GraspMetrics:
    valid: torch.Tensor
    tcp_distance: torch.Tensor
    bilateral_deflection: torch.Tensor


def grasp_metrics(
    env: _ResetToolEnv,
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


@dataclass(frozen=True)
class _CollisionLabelLayout:
    """Device label masks with one safe sentinel shape appended."""

    shape_count: int
    shape_body: torch.Tensor
    shape_world: torch.Tensor
    robot_shape: torch.Tensor
    left_finger_shape: torch.Tensor
    right_finger_shape: torch.Tensor
    grasp_proxy_shape: torch.Tensor
    display_labels: tuple[str, ...]


@dataclass(frozen=True)
class _CollisionBufferBinding:
    """Static interpretation of one model-owned contact buffer."""

    contacts: Any
    model: Any
    capacity: int
    contact_slots: torch.Tensor
    labels: _CollisionLabelLayout


@dataclass(frozen=True)
class _CollisionBufferReduction:
    """Device-resident collision reduction before optional diagnostics."""

    valid: torch.Tensor
    invalid_contact_count: torch.Tensor
    left_grasp_contact_count: torch.Tensor
    right_grasp_contact_count: torch.Tensor
    contact_overflow: torch.Tensor
    invalid_contact_mask: torch.Tensor
    contact_world: torch.Tensor
    contact_shape0: torch.Tensor
    contact_shape1: torch.Tensor
    separation: torch.Tensor
    negative_contact_count: torch.Tensor
    invalid_active_shape: torch.Tensor
    invalid_active_body: torch.Tensor


def _wait_for_warp_contact_data(tensor: torch.Tensor) -> None:
    """Order Torch reads after Warp without synchronizing the whole CUDA device."""
    if tensor.device.type != "cuda":
        return
    producer = wp.get_stream(str(tensor.device))
    torch_stream = torch.cuda.current_stream(tensor.device)
    if int(producer.cuda_stream) == int(torch_stream.cuda_stream):
        return
    wp.stream_from_torch(torch_stream).wait_stream(producer)


def _collision_label_layout(
    shape_body: torch.Tensor,
    shape_world: torch.Tensor,
    body_labels: list[str] | tuple[str, ...],
    shape_labels: list[str] | tuple[str, ...],
) -> _CollisionLabelLayout:
    """Bind the scalar collision label rules once for one model layout."""
    if shape_body.ndim != 1 or shape_world.ndim != 1 or shape_body.shape != shape_world.shape:
        raise RuntimeError("Collision model shape-body/world arrays must be matching vectors.")
    shape_labels = tuple(str(label) for label in shape_labels)
    body_labels = tuple(str(label) for label in body_labels)
    shape_count = shape_body.numel()
    if len(shape_labels) < shape_count:
        raise RuntimeError(
            f"Collision model exposes {shape_count} shape indices but only {len(shape_labels)} shape labels."
        )
    shape_labels = shape_labels[:shape_count]

    body_ids = shape_body.detach().cpu().tolist()
    if any(body_id >= len(body_labels) for body_id in body_ids):
        raise RuntimeError("Collision model shape-body indices exceed its body-label layout.")
    combined_labels: list[str] = []
    display_labels: list[str] = []
    robot_shape: list[bool] = []
    for body_id, shape_label in zip(body_ids, shape_labels, strict=True):
        body_label = body_labels[body_id] if body_id >= 0 else ""
        combined_labels.append(f"{body_label} {shape_label}")
        display_labels.append(body_label or shape_label)
        robot_shape.append("/Robot/" in body_label or "/Robot/" in shape_label)

    device = shape_body.device

    def padded_mask(values: list[bool]) -> torch.Tensor:
        return torch.tensor((*values, False), device=device, dtype=torch.bool)

    return _CollisionLabelLayout(
        shape_count=shape_count,
        shape_body=torch.cat((shape_body, shape_body.new_tensor((-1,)))),
        shape_world=torch.cat((shape_world.long(), shape_world.new_tensor((-1,), dtype=torch.long))),
        robot_shape=padded_mask(robot_shape),
        left_finger_shape=padded_mask(["panda_leftfinger" in label for label in combined_labels]),
        right_finger_shape=padded_mask(["panda_rightfinger" in label for label in combined_labels]),
        grasp_proxy_shape=padded_mask([label.endswith("/Plug/GraspProxy") for label in shape_labels]),
        display_labels=tuple(display_labels),
    )


def _collision_buffer_binding(
    env: _ResetToolEnv,
    contacts,
    model,
    contact_count: torch.Tensor,
) -> _CollisionBufferBinding:
    """Return an environment-local binding without relying on recyclable object ids."""
    capacity = int(contacts.rigid_contact_max)
    cache = getattr(env, "_reset_tool_collision_buffer_bindings", None)
    if cache is None:
        cache = []
        setattr(env, "_reset_tool_collision_buffer_bindings", cache)
    for binding in cache:
        if binding.contacts is contacts and binding.model is model and binding.capacity == capacity:
            return binding

    labels = _collision_label_layout(
        wp.to_torch(model.shape_body),
        wp.to_torch(model.shape_world),
        [str(label) for label in model.body_label],
        [str(label) for label in model.shape_label],
    )
    binding = _CollisionBufferBinding(
        contacts=contacts,
        model=model,
        capacity=capacity,
        contact_slots=torch.arange(capacity, device=contact_count.device, dtype=torch.long),
        labels=labels,
    )
    cache.append(binding)
    return binding


def _reduce_collision_buffer(
    *,
    contact_count: torch.Tensor,
    contact_slots: torch.Tensor,
    contact_shape0: torch.Tensor,
    contact_shape1: torch.Tensor,
    contact_point0: torch.Tensor,
    contact_point1: torch.Tensor,
    contact_normal: torch.Tensor,
    contact_margin0: torch.Tensor,
    contact_margin1: torch.Tensor,
    labels: _CollisionLabelLayout,
    body_q: torch.Tensor,
    num_envs: int,
    penetration_tolerance: float,
) -> _CollisionBufferReduction:
    """Classify a fixed-capacity contact buffer entirely on its resident device."""
    capacity = contact_slots.numel()
    if num_envs < 1:
        raise ValueError("Collision reduction requires at least one environment.")
    if contact_count.numel() < 1:
        raise RuntimeError("Collision contact-count buffer is empty.")
    for name, value in (
        ("shape0", contact_shape0),
        ("shape1", contact_shape1),
        ("point0", contact_point0),
        ("point1", contact_point1),
        ("normal", contact_normal),
        ("margin0", contact_margin0),
        ("margin1", contact_margin1),
    ):
        if value.shape[0] < capacity:
            raise RuntimeError(f"Collision {name} buffer is shorter than its declared capacity {capacity}.")
    if body_q.ndim != 2 or body_q.shape[1] != 7:
        raise RuntimeError(f"Collision body transforms must have shape (N, 7), got {tuple(body_q.shape)}.")

    reported = contact_count.reshape(-1)[0]
    active_count = reported.clamp(0, capacity)
    active = contact_slots < active_count
    shape0 = contact_shape0[:capacity]
    shape1 = contact_shape1[:capacity]
    valid_shape0 = (shape0 >= 0) & (shape0 < labels.shape_count)
    valid_shape1 = (shape1 >= 0) & (shape1 < labels.shape_count)
    valid_shape = valid_shape0 & valid_shape1
    safe_shape0 = torch.where(valid_shape0, shape0, torch.full_like(shape0, labels.shape_count))
    safe_shape1 = torch.where(valid_shape1, shape1, torch.full_like(shape1, labels.shape_count))

    body0 = labels.shape_body[safe_shape0]
    body1 = labels.shape_body[safe_shape1]
    body_count = body_q.shape[0]
    valid_body0 = body0 < body_count
    valid_body1 = body1 < body_count
    invalid_active_body = (active & valid_shape & (~valid_body0 | ~valid_body1)).any()

    point0 = contact_point0[:capacity]
    point1 = contact_point1[:capacity]
    if body_count:
        dynamic0 = (body0 >= 0) & valid_body0
        dynamic1 = (body1 >= 0) & valid_body1
        safe_body0 = body0.clamp(0, body_count - 1)
        safe_body1 = body1.clamp(0, body_count - 1)
        pose0 = body_q[safe_body0]
        pose1 = body_q[safe_body1]
        transformed0 = pose0[:, :3] + math_utils.quat_apply(pose0[:, 3:7], point0)
        transformed1 = pose1[:, :3] + math_utils.quat_apply(pose1[:, 3:7], point1)
        world_point0 = torch.where(dynamic0[:, None], transformed0, point0)
        world_point1 = torch.where(dynamic1[:, None], transformed1, point1)
    else:
        world_point0 = point0
        world_point1 = point1

    separation = (
        (contact_normal[:capacity] * (world_point1 - world_point0)).sum(-1)
        - contact_margin0[:capacity]
        - contact_margin1[:capacity]
    )
    world0 = labels.shape_world[safe_shape0]
    world1 = labels.shape_world[safe_shape1]
    world = torch.where(world0 >= 0, world0, world1)
    valid_world = (world >= 0) & (world < num_envs)
    safe_world = world.clamp(0, num_envs - 1)
    eligible = active & valid_shape & valid_body0 & valid_body1 & valid_world

    robot0 = labels.robot_shape[safe_shape0]
    robot1 = labels.robot_shape[safe_shape1]
    has_robot = robot0 | robot1
    robot_shape = torch.where(robot0, safe_shape0, safe_shape1)
    other_shape = torch.where(robot0, safe_shape1, safe_shape0)
    proxy_finger = eligible & has_robot & labels.grasp_proxy_shape[other_shape]
    left_contact = proxy_finger & labels.left_finger_shape[robot_shape]
    right_contact = proxy_finger & ~left_contact & labels.right_finger_shape[robot_shape]
    invalid_contact = eligible & has_robot & ~left_contact & ~right_contact & (separation < -penetration_tolerance)

    invalid_count = torch.zeros(num_envs, dtype=torch.long, device=contact_slots.device)
    left_count = torch.zeros_like(invalid_count)
    right_count = torch.zeros_like(invalid_count)
    invalid_count.scatter_add_(0, safe_world, invalid_contact.long())
    left_count.scatter_add_(0, safe_world, left_contact.long())
    right_count.scatter_add_(0, safe_world, right_contact.long())
    overflow = reported > capacity
    return _CollisionBufferReduction(
        valid=(invalid_count == 0) & ~overflow,
        invalid_contact_count=invalid_count,
        left_grasp_contact_count=left_count,
        right_grasp_contact_count=right_count,
        contact_overflow=overflow,
        invalid_contact_mask=invalid_contact,
        contact_world=world,
        contact_shape0=shape0,
        contact_shape1=shape1,
        separation=separation,
        negative_contact_count=reported < 0,
        invalid_active_shape=(active & ~valid_shape).any(),
        invalid_active_body=invalid_active_body,
    )


def _invalid_contact_pairs(
    reduction: _CollisionBufferReduction,
    labels: _CollisionLabelLayout,
) -> tuple[str, ...]:
    """Materialize at most 64 failing contacts in original buffer order."""
    contact_ids = torch.nonzero(reduction.invalid_contact_mask, as_tuple=False).flatten()[:64]
    if contact_ids.numel() == 0:
        return ()
    rows = (
        torch.stack(
            (
                reduction.contact_world[contact_ids].double(),
                reduction.contact_shape0[contact_ids].double(),
                reduction.contact_shape1[contact_ids].double(),
                reduction.separation[contact_ids].double(),
            ),
            dim=-1,
        )
        .detach()
        .cpu()
    )
    return tuple(
        f"world={int(world)} {labels.display_labels[int(shape0)]} <-> "
        f"{labels.display_labels[int(shape1)]} separation={float(separation):.6g}"
        for world, shape0, shape1, separation in rows.tolist()
    )


def _collision_buffer_metrics(
    env: _ResetToolEnv,
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
    contact_count = wp.to_torch(contacts.rigid_contact_count)
    _wait_for_warp_contact_data(contact_count)
    binding = _collision_buffer_binding(env, contacts, model, contact_count)
    reduction = _reduce_collision_buffer(
        contact_count=contact_count,
        contact_slots=binding.contact_slots,
        contact_shape0=wp.to_torch(contacts.rigid_contact_shape0),
        contact_shape1=wp.to_torch(contacts.rigid_contact_shape1),
        contact_point0=wp.to_torch(contacts.rigid_contact_point0),
        contact_point1=wp.to_torch(contacts.rigid_contact_point1),
        contact_normal=wp.to_torch(contacts.rigid_contact_normal),
        contact_margin0=wp.to_torch(contacts.rigid_contact_margin0),
        contact_margin1=wp.to_torch(contacts.rigid_contact_margin1),
        labels=binding.labels,
        body_q=wp.to_torch(state.body_q),
        num_envs=env.num_envs,
        penetration_tolerance=penetration_tolerance,
    )
    negative_count, invalid_shape, invalid_body, overflow, has_invalid = (
        bool(value)
        for value in torch.stack(
            (
                reduction.negative_contact_count,
                reduction.invalid_active_shape,
                reduction.invalid_active_body,
                reduction.contact_overflow,
                reduction.invalid_contact_mask.any(),
            )
        )
        .detach()
        .cpu()
        .tolist()
    )
    if negative_count:
        raise RuntimeError("Collision buffer reported a negative contact count.")
    if invalid_shape:
        raise RuntimeError("Collision buffer contains an active shape index outside its owning model layout.")
    if invalid_body:
        raise RuntimeError("Collision model contains an active body index outside its owning state layout.")
    invalid_pairs = _invalid_contact_pairs(reduction, binding.labels) if has_invalid else ()
    grasp_count = reduction.left_grasp_contact_count + reduction.right_grasp_contact_count
    return CollisionMetrics(
        reduction.valid,
        reduction.invalid_contact_count,
        grasp_count,
        reduction.left_grasp_contact_count,
        reduction.right_grasp_contact_count,
        overflow,
        invalid_pairs,
    )


def collision_metrics(
    env: _ResetToolEnv,
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


def _physical_validity_sample(
    env: _ResetToolEnv,
    finger_target: torch.Tensor,
) -> tuple[CollisionMetrics, GraspMetrics, torch.Tensor]:
    """Sample collisions and require real bilateral proxy contact independently.

    ``CollisionMetrics.valid`` is intentionally collision-only here.  Keeping
    the proxy-contact requirement separate prevents a transient missing finger
    contact from being reported as an unexplained collision failure.
    """
    collision = collision_metrics(env, require_bilateral_grasp=False)
    grasp = grasp_metrics(env, finger_target)
    bilateral_proxy_contact = (collision.left_grasp_contact_count > 0) & (collision.right_grasp_contact_count > 0)
    return collision, grasp, bilateral_proxy_contact


def _active_waypoint_count(
    distances: torch.Tensor,
    active_mask: torch.Tensor,
    maximum_step: float,
) -> tuple[int, torch.Tensor]:
    """Return an active-only waypoint count and active lanes with invalid distances."""
    distances = torch.as_tensor(distances)
    active_mask = torch.as_tensor(active_mask, device=distances.device, dtype=torch.bool)
    if distances.ndim != 1 or active_mask.shape != distances.shape:
        raise ValueError(
            f"distances and active_mask must have matching one-dimensional shapes, got "
            f"{tuple(distances.shape)} and {tuple(active_mask.shape)}."
        )
    maximum_step = float(maximum_step)
    if not math.isfinite(maximum_step) or maximum_step <= 0.0:
        raise ValueError("maximum_step must be finite and positive.")
    distance_valid = torch.isfinite(distances) & (distances >= 0.0)
    invalid_active = active_mask & ~distance_valid
    scheduled = active_mask & distance_valid
    if not bool(scheduled.any()):
        return 0, invalid_active
    maximum_distance = float(distances[scheduled].amax())
    return max(1, int(math.ceil(maximum_distance / maximum_step))), invalid_active


class _PerLaneTargetHold:
    """Keep failed simulation lanes at a fixed, finite robot command.

    The context intercepts robot target requests and wraps ``env.advance`` so
    the merged command is re-applied before every real physics step.  Active
    lanes receive the latest request.  Initially inactive lanes and lanes
    deactivated later receive their latched arm/finger targets instead.

    Deactivation is monotone.  A newly failed lane latches its measured arm
    position when the complete row is finite, otherwise it retains the last
    finite arm target sent to the simulator.  Its current finger target is
    always retained, which lets callers keep an open approach open and a
    failed grasp closed without embedding stage policy in this helper.

    Global failures such as contact-buffer overflow deliberately remain the
    caller's responsibility; this helper only isolates attributable lanes.
    """

    def __init__(
        self,
        env: _ResetToolEnv,
        active_mask: torch.Tensor,
        arm_target: torch.Tensor,
        finger_target: torch.Tensor,
    ) -> None:
        self._env = env
        self._num_envs = int(env.num_envs)
        self._device = torch.device(env.device)
        self._active = torch.as_tensor(active_mask, device=self._device, dtype=torch.bool).clone()
        self._requested_arm = torch.as_tensor(arm_target, device=self._device, dtype=torch.float32).clone()
        self._requested_finger = torch.as_tensor(finger_target, device=self._device, dtype=torch.float32).clone()
        if tuple(self._active.shape) != (self._num_envs,):
            raise ValueError(f"active_mask must have shape ({self._num_envs},).")
        if tuple(self._requested_arm.shape) != (self._num_envs, 7):
            raise ValueError(f"arm_target must have shape ({self._num_envs}, 7).")
        if tuple(self._requested_finger.shape) != (self._num_envs, 2):
            raise ValueError(f"finger_target must have shape ({self._num_envs}, 2).")
        if not bool(torch.isfinite(self._requested_arm).all()):
            raise ValueError("Initial arm targets must be finite.")
        if not bool(torch.isfinite(self._requested_finger).all()):
            raise ValueError("Initial finger targets must be finite.")

        measured_arm, _, _, _ = env.read_robot_state()
        measured_arm = torch.as_tensor(measured_arm, device=self._device, dtype=torch.float32)
        if tuple(measured_arm.shape) != (self._num_envs, 7):
            raise ValueError(f"Measured arm positions must have shape ({self._num_envs}, 7).")
        measured_finite = torch.isfinite(measured_arm).all(dim=-1)
        self._frozen_arm = torch.where(measured_finite[:, None], measured_arm, self._requested_arm).clone()
        self._frozen_finger = self._requested_finger.clone()
        self._last_sent_arm = torch.where(self._active[:, None], self._requested_arm, self._frozen_arm).clone()
        self._last_sent_finger = self._requested_finger.clone()
        self._initial_active = self._active.clone()
        self._reason_masks: dict[str, torch.Tensor] = {}
        self._entered = False
        self._send_count = 0

        self._original_set_robot_targets = env.set_robot_targets
        self._original_advance = env.advance
        env_dict = getattr(env, "__dict__", {})
        self._had_instance_set_robot_targets = "set_robot_targets" in env_dict
        self._instance_set_robot_targets = env_dict.get("set_robot_targets")
        self._had_instance_advance = "advance" in env_dict
        self._instance_advance = env_dict.get("advance")

    @property
    def active_mask(self) -> torch.Tensor:
        """Return a copy of the currently active lanes."""
        return self._active.clone()

    @property
    def failed_mask(self) -> torch.Tensor:
        """Return initially active lanes that have since been deactivated."""
        return self._initial_active & ~self._active

    @property
    def reason_masks(self) -> dict[str, torch.Tensor]:
        """Return copies of the first-failure masks recorded by reason."""
        return {reason: mask.clone() for reason, mask in self._reason_masks.items()}

    @property
    def last_sent_arm_target(self) -> torch.Tensor:
        """Return the latest finite arm target sent to every lane."""
        return self._last_sent_arm.clone()

    @property
    def last_sent_finger_target(self) -> torch.Tensor:
        """Return the latest finite finger target sent to every lane."""
        return self._last_sent_finger.clone()

    def _merged_targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        arm_target = torch.where(self._active[:, None], self._requested_arm, self._frozen_arm)
        finger_target = torch.where(self._active[:, None], self._requested_finger, self._frozen_finger)
        return arm_target, finger_target

    def _send(self) -> None:
        arm_target, finger_target = self._merged_targets()
        if not bool(torch.isfinite(arm_target).all()) or not bool(torch.isfinite(finger_target).all()):
            raise RuntimeError("Per-lane held robot targets must remain finite.")
        self._original_set_robot_targets(arm_target, finger_target)
        self._last_sent_arm.copy_(arm_target)
        self._last_sent_finger.copy_(finger_target)
        self._send_count += 1

    def set_robot_targets(self, arm_target: torch.Tensor, finger_target: torch.Tensor) -> None:
        """Request targets for active lanes while preserving all frozen lanes."""
        arm_target = torch.as_tensor(arm_target, device=self._device, dtype=torch.float32)
        finger_target = torch.as_tensor(finger_target, device=self._device, dtype=torch.float32)
        if tuple(arm_target.shape) != (self._num_envs, 7):
            raise ValueError(f"arm_target must have shape ({self._num_envs}, 7).")
        if tuple(finger_target.shape) != (self._num_envs, 2):
            raise ValueError(f"finger_target must have shape ({self._num_envs}, 2).")
        active_arm_finite = torch.isfinite(arm_target).all(dim=-1) | ~self._active
        active_finger_finite = torch.isfinite(finger_target).all(dim=-1) | ~self._active
        if not bool(active_arm_finite.all()):
            raise ValueError("Requested arm targets must be finite in every active lane.")
        if not bool(active_finger_finite.all()):
            raise ValueError("Requested finger targets must be finite in every active lane.")
        self._requested_arm.copy_(torch.where(self._active[:, None], arm_target, self._requested_arm))
        self._requested_finger.copy_(torch.where(self._active[:, None], finger_target, self._requested_finger))
        self._send()

    def deactivate(self, mask: torch.Tensor, *, reason: str) -> torch.Tensor:
        """Freeze newly failed lanes and return the lanes changed by this call."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A non-empty deactivation reason is required.")
        mask = torch.as_tensor(mask, device=self._device, dtype=torch.bool)
        if tuple(mask.shape) != (self._num_envs,):
            raise ValueError(f"deactivation mask must have shape ({self._num_envs},).")
        newly_failed = self._active & mask
        if not bool(newly_failed.any()):
            return newly_failed

        measured_arm, _, _, _ = self._env.read_robot_state()
        measured_arm = torch.as_tensor(measured_arm, device=self._device, dtype=torch.float32)
        if tuple(measured_arm.shape) != (self._num_envs, 7):
            raise ValueError(f"Measured arm positions must have shape ({self._num_envs}, 7).")
        measured_finite = torch.isfinite(measured_arm).all(dim=-1)
        held_arm = torch.where(measured_finite[:, None], measured_arm, self._last_sent_arm)
        self._frozen_arm.copy_(torch.where(newly_failed[:, None], held_arm, self._frozen_arm))
        self._frozen_finger.copy_(torch.where(newly_failed[:, None], self._last_sent_finger, self._frozen_finger))
        self._active.logical_and_(~newly_failed)
        reason_mask = self._reason_masks.setdefault(reason, torch.zeros_like(self._active))
        reason_mask.logical_or_(newly_failed)
        self._send()
        return newly_failed.clone()

    def _advance_with_hold(
        self,
        duration_s: float,
        update: Callable[[int, int, float], None] | None = None,
        *,
        post_step: Callable[[int, int, float], None] | None = None,
    ) -> int:
        def update_and_hold(step: int, steps: int, progress: float) -> None:
            send_count_before_update = self._send_count
            if update is not None:
                update(step, steps, progress)
            if self._send_count == send_count_before_update:
                self._send()

        return self._original_advance(duration_s, update_and_hold, post_step=post_step)

    def __enter__(self) -> _PerLaneTargetHold:
        if self._entered:
            raise RuntimeError("A per-lane target hold context cannot be entered twice.")
        if isinstance(getattr(self._env.set_robot_targets, "__self__", None), _PerLaneTargetHold):
            raise RuntimeError("Per-lane target hold contexts cannot be nested on one environment.")
        self._send()
        self._entered = True
        try:
            self._env.set_robot_targets = self.set_robot_targets
            self._env.advance = self._advance_with_hold
        except Exception:
            self._restore_environment_methods()
            self._entered = False
            raise
        return self

    def _restore_environment_methods(self) -> None:
        if self._had_instance_set_robot_targets:
            self._env.set_robot_targets = self._instance_set_robot_targets
        else:
            del self._env.set_robot_targets
        if self._had_instance_advance:
            self._env.advance = self._instance_advance
        else:
            del self._env.advance

    def __exit__(self, *_exception: object) -> None:
        self._restore_environment_methods()
        self._entered = False


def interpolate_arm_motion(
    env: _ResetToolEnv,
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


@dataclass(frozen=True)
class _ResetReplayState:
    """One exact task/robot state sampled at a reset replay boundary."""

    task_q: torch.Tensor
    task_qd: torch.Tensor
    arm_q: torch.Tensor
    arm_qd: torch.Tensor
    finger_q: torch.Tensor
    finger_qd: torch.Tensor


def _reset_replay_layout_signature(env: _ResetToolEnv) -> tuple[int, int | None, int, int, int]:
    """Validate and return the task indices needed by row replay evidence."""
    layout = env.rj45_runtime.layout
    body_count = int(layout.body_count)
    plug_index = int(layout.plug_body_index)
    socket_index = None if layout.socket_body_index is None else int(layout.socket_body_index)
    cable_slice = layout.cable_body_slice
    if body_count < 1 or not 0 <= plug_index < body_count:
        raise RuntimeError("Reset replay evidence received an invalid task body layout.")
    if socket_index is not None and not 0 <= socket_index < body_count:
        raise RuntimeError("Reset replay evidence received an invalid socket body index.")
    if not isinstance(cable_slice, slice) or cable_slice.step not in (None, 1):
        raise RuntimeError("Reset replay evidence requires one contiguous cable body slice.")
    cable_start, cable_stop, _ = cable_slice.indices(body_count)
    if cable_stop <= cable_start:
        raise RuntimeError("Reset replay evidence requires at least one cable body.")
    return body_count, socket_index, plug_index, cable_start, cable_stop


def _read_reset_replay_state(env: _ResetToolEnv, *, body_count: int) -> _ResetReplayState:
    """Read and shape-check one complete reset replay state."""
    task_q, task_qd = env.read_task_state()
    arm_q, arm_qd, finger_q, finger_qd = env.read_robot_state()
    arm_joint_count = len(env._arm_joint_ids)
    expected_shapes = {
        "task_q": (env.num_envs, body_count, 7),
        "task_qd": (env.num_envs, body_count, 6),
        "arm_q": (env.num_envs, arm_joint_count),
        "arm_qd": (env.num_envs, arm_joint_count),
        "finger_q": (env.num_envs, 2),
        "finger_qd": (env.num_envs, 2),
    }
    values = {
        "task_q": task_q,
        "task_qd": task_qd,
        "arm_q": arm_q,
        "arm_qd": arm_qd,
        "finger_q": finger_q,
        "finger_qd": finger_qd,
    }
    for name, expected_shape in expected_shapes.items():
        if not isinstance(values[name], torch.Tensor) or tuple(values[name].shape) != expected_shape:
            raise RuntimeError(
                f"Reset replay {name} must have shape {expected_shape}, got "
                f"{getattr(values[name], 'shape', type(values[name]).__name__)}."
            )
    return _ResetReplayState(**{name: value.clone() for name, value in values.items()})


def _per_world_finite(*values: torch.Tensor) -> torch.Tensor:
    """Return whether every value is finite in each world."""
    result = torch.ones(values[0].shape[0], dtype=torch.bool, device=values[0].device)
    for value in values:
        result &= torch.isfinite(value).reshape(value.shape[0], -1).all(dim=-1)
    return result


def _finite_maximum(values: torch.Tensor, *, dimensions: tuple[int, ...]) -> torch.Tensor:
    """Return a per-world maximum, promoting non-finite samples to infinity."""
    finite_values = torch.nan_to_num(values, nan=torch.inf, posinf=torch.inf, neginf=torch.inf)
    return finite_values.amax(dim=dimensions)


def _reset_replay_speed_metrics(
    state: _ResetReplayState,
    *,
    cable_slice: slice,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-world cable, arm, and finger speed maxima."""
    cable_speed = _finite_maximum(
        torch.linalg.vector_norm(state.task_qd[:, cable_slice, :3], dim=-1),
        dimensions=(1,),
    )
    arm_speed = _finite_maximum(torch.abs(state.arm_qd), dimensions=(1,))
    finger_speed = _finite_maximum(torch.abs(state.finger_qd), dimensions=(1,))
    return cable_speed, arm_speed, finger_speed


def _runtime_drives_disabled(env: _ResetToolEnv) -> torch.Tensor:
    """Return one per-world mask for both construction drives being disabled."""
    runtime = env.rj45_runtime

    def as_bool_tensor(value: Any, name: str) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else wp.to_torch(value)
        tensor = torch.as_tensor(tensor, device=env.device, dtype=torch.bool)
        if tuple(tensor.shape) != (env.num_envs,):
            raise RuntimeError(f"Runtime {name} must have shape ({env.num_envs},), got {tuple(tensor.shape)}.")
        return tensor

    translation = as_bool_tensor(runtime.drive_enabled, "drive_enabled")
    orientation = as_bool_tensor(runtime.orientation_hold_enabled, "orientation_hold_enabled")
    return ~translation & ~orientation


def _arm_target_tracking_limits(env: _ResetToolEnv, joint_count: int, dtype: torch.dtype) -> torch.Tensor:
    """Return live tracking limits, or infinity for the legacy relative controller."""
    configured = getattr(env.cfg.actions.arm_action, "tracking_error_limits", None)
    if configured is None:
        return torch.full((joint_count,), torch.inf, device=env.device, dtype=dtype)
    limits = torch.as_tensor(configured, device=env.device, dtype=dtype)
    if tuple(limits.shape) != (joint_count,) or not bool(torch.isfinite(limits).all()) or not bool((limits > 0).all()):
        raise ValueError(f"tracking_error_limits must contain {joint_count} finite positive values.")
    return limits


def _reset_replay_state_matches(left: _ResetReplayState, right: _ResetReplayState) -> bool:
    """Return whether two sampled states are exactly continuous."""
    return all(
        torch.equal(getattr(left, name), getattr(right, name)) for name in _ResetReplayState.__dataclass_fields__
    )


def _initialize_reset_replay_evidence(
    env: _ResetToolEnv,
    evidence: dict[str, Any],
    state: _ResetReplayState,
    *,
    layout_signature: tuple[int, int | None, int, int, int],
    arm_target_parameter: torch.Tensor,
    arm_target_semantics: str,
    finger_target: torch.Tensor,
    starts_grasped: bool,
) -> None:
    """Initialize one uninterrupted row replay evidence accumulator."""
    if evidence:
        raise ValueError("A new reset replay evidence dictionary must be empty.")
    _, socket_index, plug_index, cable_start, cable_stop = layout_signature
    cable_slice = slice(cable_start, cable_stop)
    cable_speed, arm_speed, finger_speed = _reset_replay_speed_metrics(state, cable_slice=cable_slice)
    stored_finite = _per_world_finite(
        state.task_q,
        state.task_qd,
        state.arm_q,
        state.arm_qd,
        state.finger_q,
        state.finger_qd,
        arm_target_parameter,
        finger_target,
    )
    true_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    zero_float = torch.zeros(env.num_envs, dtype=state.task_q.dtype, device=env.device)
    zero_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    evidence.update(
        {
            "starts_grasped": starts_grasped,
            "arm_target_semantics": arm_target_semantics,
            "stored_state_finite": stored_finite,
            "stored_task_state_finite_and_normalized": task_state_is_finite_and_normalized(
                state.task_q,
                state.task_qd,
            ),
            "stored_drive_disabled": _runtime_drives_disabled(env),
            "stored_maximum_cable_speed_m_s": cable_speed,
            "stored_maximum_arm_joint_speed_rad_s": arm_speed,
            "stored_maximum_finger_joint_speed_m_s": finger_speed,
            "stored_maximum_arm_target_tracking_error_rad": zero_float.clone(),
            "stored_arm_target_tracking_error_by_joint_rad": torch.zeros_like(state.arm_q),
            "stored_arm_target_tracking_bounded": true_mask.clone(),
            "all_post_step_state_finite": true_mask.clone(),
            "all_post_step_task_state_finite_and_normalized": true_mask.clone(),
            "all_post_step_collision_free": true_mask.clone(),
            "all_post_step_drive_disabled": true_mask.clone(),
            "all_post_step_bilateral_grasp": true_mask.clone(),
            "all_post_step_proxy_bilateral_contact": true_mask.clone(),
            "all_post_step_zero_proxy_contacts": true_mask.clone(),
            "all_post_step_expected_contact_state": true_mask.clone(),
            "all_post_step_arm_target_tracking_bounded": true_mask.clone(),
            "maximum_body_excursion_m": zero_float.clone(),
            "maximum_plug_excursion_m": zero_float.clone(),
            "maximum_socket_excursion_m": zero_float.clone(),
            "maximum_post_step_cable_speed_m_s": zero_float.clone(),
            "maximum_post_step_arm_joint_speed_rad_s": zero_float.clone(),
            "maximum_post_step_finger_joint_speed_m_s": zero_float.clone(),
            "maximum_post_step_arm_target_tracking_error_rad": zero_float.clone(),
            "maximum_post_step_arm_target_tracking_error_by_joint_rad": torch.zeros_like(state.arm_q),
            "maximum_arm_target_drift_rad": zero_float.clone(),
            "final_cable_speed_m_s": cable_speed.clone(),
            "final_arm_joint_speed_rad_s": arm_speed.clone(),
            "final_finger_joint_speed_m_s": finger_speed.clone(),
            "minimum_left_proxy_contact_count": zero_count.clone(),
            "minimum_right_proxy_contact_count": zero_count.clone(),
            "maximum_left_proxy_contact_count": zero_count.clone(),
            "maximum_right_proxy_contact_count": zero_count.clone(),
            "maximum_invalid_contact_count": zero_count.clone(),
            "any_contact_overflow": False,
            "invalid_contact_pairs": (),
            "post_step_samples": 0,
            "_continuity": {
                "env_id": id(env),
                "layout_signature": layout_signature,
                "arm_target_parameter": arm_target_parameter.clone(),
                "arm_target_semantics": arm_target_semantics,
                "finger_target": finger_target.clone(),
                "baseline_task_q": state.task_q.clone(),
                "last_state": state,
                "socket_index": socket_index,
                "plug_index": plug_index,
            },
        }
    )


def _continue_reset_replay_evidence(
    env: _ResetToolEnv,
    evidence: dict[str, Any],
    state: _ResetReplayState,
    *,
    layout_signature: tuple[int, int | None, int, int, int],
    arm_target_parameter: torch.Tensor,
    arm_target_semantics: str,
    finger_target: torch.Tensor,
    starts_grasped: bool,
) -> None:
    """Reject evidence reuse unless the replay is demonstrably continuous."""
    continuity = evidence.get("_continuity")
    if not isinstance(continuity, dict):
        raise ValueError("Existing reset replay evidence is missing private continuity state.")
    compatible = (
        continuity.get("env_id") == id(env)
        and continuity.get("layout_signature") == layout_signature
        and evidence.get("starts_grasped") is starts_grasped
        and evidence.get("arm_target_semantics") == arm_target_semantics
        and continuity.get("arm_target_semantics") == arm_target_semantics
        and isinstance(continuity.get("arm_target_parameter"), torch.Tensor)
        and torch.equal(continuity["arm_target_parameter"], arm_target_parameter)
        and isinstance(continuity.get("finger_target"), torch.Tensor)
        and torch.equal(continuity["finger_target"], finger_target)
        and isinstance(continuity.get("last_state"), _ResetReplayState)
        and _reset_replay_state_matches(continuity["last_state"], state)
    )
    if not compatible:
        raise ValueError(
            "Reset replay evidence can accumulate only across uninterrupted calls with the same environment, "
            "layout, semantics, targets, and exact final state."
        )


def _sample_reset_replay_post_step(
    env: _ResetToolEnv,
    evidence: dict[str, Any],
    state: _ResetReplayState,
    *,
    arm_target: torch.Tensor,
    finger_target: torch.Tensor,
) -> None:
    """Accumulate one mechanical sample after a real reset replay step."""
    continuity = evidence["_continuity"]
    baseline_task_q = continuity["baseline_task_q"]
    _, _, _, cable_start, cable_stop = continuity["layout_signature"]
    cable_slice = slice(cable_start, cable_stop)
    cable_speed, arm_speed, finger_speed = _reset_replay_speed_metrics(state, cable_slice=cable_slice)
    collision = collision_metrics(env, require_bilateral_grasp=False)
    if collision.contact_overflow:
        raise RuntimeError("Global contact-buffer overflow during reset replay.")
    drives_disabled = _runtime_drives_disabled(env)
    if not bool(drives_disabled.all()):
        raise RuntimeError("A construction drive became enabled during reset replay.")
    grasp = grasp_metrics(env, finger_target)
    left_contacts = collision.left_grasp_contact_count
    right_contacts = collision.right_grasp_contact_count
    expected_shape = (env.num_envs,)
    for name, value in (
        ("collision.valid", collision.valid),
        ("collision.invalid_contact_count", collision.invalid_contact_count),
        ("collision.left_grasp_contact_count", left_contacts),
        ("collision.right_grasp_contact_count", right_contacts),
        ("grasp.valid", grasp.valid),
    ):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
            raise RuntimeError(f"Reset replay {name} must have shape {expected_shape}.")

    proxy_bilateral = (left_contacts > 0) & (right_contacts > 0)
    zero_proxy_contacts = (left_contacts == 0) & (right_contacts == 0)
    expected_contact_state = grasp.valid & proxy_bilateral if evidence["starts_grasped"] else zero_proxy_contacts
    state_finite = _per_world_finite(
        state.task_q,
        state.task_qd,
        state.arm_q,
        state.arm_qd,
        state.finger_q,
        state.finger_qd,
        arm_target,
    )
    arm_target_tracking_error = _finite_maximum(torch.abs(arm_target - state.arm_q), dimensions=(1,))
    tracking_limits = _arm_target_tracking_limits(env, state.arm_q.shape[1], state.arm_q.dtype)
    arm_target_tracking_bounded = (torch.abs(arm_target - state.arm_q) <= tracking_limits).all(dim=-1)
    body_excursion = _finite_maximum(
        torch.linalg.vector_norm(state.task_q[..., :3] - baseline_task_q[..., :3], dim=-1),
        dimensions=(1,),
    )
    plug_index = continuity["plug_index"]
    plug_excursion = torch.nan_to_num(
        torch.linalg.vector_norm(
            state.task_q[:, plug_index, :3] - baseline_task_q[:, plug_index, :3],
            dim=-1,
        ),
        nan=torch.inf,
        posinf=torch.inf,
        neginf=torch.inf,
    )
    socket_index = continuity["socket_index"]
    if socket_index is None:
        socket_excursion = torch.zeros_like(plug_excursion)
    else:
        socket_excursion = torch.nan_to_num(
            torch.linalg.vector_norm(
                state.task_q[:, socket_index, :3] - baseline_task_q[:, socket_index, :3],
                dim=-1,
            ),
            nan=torch.inf,
            posinf=torch.inf,
            neginf=torch.inf,
        )

    evidence["all_post_step_state_finite"] &= state_finite
    evidence["all_post_step_task_state_finite_and_normalized"] &= task_state_is_finite_and_normalized(
        state.task_q,
        state.task_qd,
    )
    evidence["all_post_step_collision_free"] &= collision.valid
    evidence["all_post_step_drive_disabled"] &= drives_disabled
    evidence["all_post_step_bilateral_grasp"] &= grasp.valid
    evidence["all_post_step_proxy_bilateral_contact"] &= proxy_bilateral
    evidence["all_post_step_zero_proxy_contacts"] &= zero_proxy_contacts
    evidence["all_post_step_expected_contact_state"] &= expected_contact_state
    evidence["all_post_step_arm_target_tracking_bounded"] &= arm_target_tracking_bounded
    evidence["maximum_body_excursion_m"] = torch.maximum(evidence["maximum_body_excursion_m"], body_excursion)
    evidence["maximum_plug_excursion_m"] = torch.maximum(evidence["maximum_plug_excursion_m"], plug_excursion)
    evidence["maximum_socket_excursion_m"] = torch.maximum(evidence["maximum_socket_excursion_m"], socket_excursion)
    evidence["maximum_post_step_cable_speed_m_s"] = torch.maximum(
        evidence["maximum_post_step_cable_speed_m_s"],
        cable_speed,
    )
    evidence["maximum_post_step_arm_joint_speed_rad_s"] = torch.maximum(
        evidence["maximum_post_step_arm_joint_speed_rad_s"],
        arm_speed,
    )
    evidence["maximum_post_step_finger_joint_speed_m_s"] = torch.maximum(
        evidence["maximum_post_step_finger_joint_speed_m_s"],
        finger_speed,
    )
    evidence["maximum_post_step_arm_target_tracking_error_rad"] = torch.maximum(
        evidence["maximum_post_step_arm_target_tracking_error_rad"],
        arm_target_tracking_error,
    )
    evidence["maximum_post_step_arm_target_tracking_error_by_joint_rad"] = torch.maximum(
        evidence["maximum_post_step_arm_target_tracking_error_by_joint_rad"],
        torch.abs(arm_target - state.arm_q),
    )
    evidence["final_cable_speed_m_s"] = cable_speed
    evidence["final_arm_joint_speed_rad_s"] = arm_speed
    evidence["final_finger_joint_speed_m_s"] = finger_speed
    if evidence["post_step_samples"] == 0:
        evidence["minimum_left_proxy_contact_count"] = left_contacts.clone()
        evidence["minimum_right_proxy_contact_count"] = right_contacts.clone()
    else:
        evidence["minimum_left_proxy_contact_count"] = torch.minimum(
            evidence["minimum_left_proxy_contact_count"],
            left_contacts,
        )
        evidence["minimum_right_proxy_contact_count"] = torch.minimum(
            evidence["minimum_right_proxy_contact_count"],
            right_contacts,
        )
    evidence["maximum_left_proxy_contact_count"] = torch.maximum(
        evidence["maximum_left_proxy_contact_count"],
        left_contacts,
    )
    evidence["maximum_right_proxy_contact_count"] = torch.maximum(
        evidence["maximum_right_proxy_contact_count"],
        right_contacts,
    )
    evidence["maximum_invalid_contact_count"] = torch.maximum(
        evidence["maximum_invalid_contact_count"],
        collision.invalid_contact_count,
    )
    evidence["any_contact_overflow"] |= collision.contact_overflow
    invalid_pairs = list(evidence["invalid_contact_pairs"])
    for pair in collision.invalid_contact_pairs:
        if pair not in invalid_pairs and len(invalid_pairs) < 64:
            invalid_pairs.append(pair)
    evidence["invalid_contact_pairs"] = tuple(invalid_pairs)
    evidence["post_step_samples"] += 1
    continuity["last_state"] = state


def _advance_reset_target_hold(
    env: _ResetToolEnv,
    duration_s: float,
    arm_target_parameter: torch.Tensor,
    finger_target: torch.Tensor,
    *,
    arm_target_semantics: str,
    target_resolver: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    clamp_evidence: dict[str, Any] | None = None,
    replay_evidence: dict[str, Any] | None = None,
    starts_grasped: bool | None = None,
) -> int:
    """Replay one reset arm-target convention with shared mechanical evidence."""
    arm_target_parameter = torch.as_tensor(arm_target_parameter, device=env.device, dtype=torch.float32)
    finger_target = torch.as_tensor(finger_target, device=env.device, dtype=torch.float32)
    arm_joint_count = len(env._arm_joint_ids)
    if tuple(arm_target_parameter.shape) != (env.num_envs, arm_joint_count):
        raise ValueError(f"arm target parameter must have shape ({env.num_envs}, {arm_joint_count}).")
    if tuple(finger_target.shape) != (env.num_envs, 2):
        raise ValueError(f"finger_target must have shape ({env.num_envs}, 2).")
    if replay_evidence is None:
        if starts_grasped is not None:
            raise ValueError("starts_grasped is only valid when replay_evidence is provided.")
        replay_state = None
    else:
        if not isinstance(starts_grasped, bool):
            raise TypeError("starts_grasped must be a plain Boolean when replay_evidence is provided.")
        layout_signature = _reset_replay_layout_signature(env)
        replay_state = _read_reset_replay_state(env, body_count=layout_signature[0])
        if "_continuity" in replay_evidence:
            _continue_reset_replay_evidence(
                env,
                replay_evidence,
                replay_state,
                layout_signature=layout_signature,
                arm_target_parameter=arm_target_parameter,
                arm_target_semantics=arm_target_semantics,
                finger_target=finger_target,
                starts_grasped=starts_grasped,
            )
        else:
            _initialize_reset_replay_evidence(
                env,
                replay_evidence,
                replay_state,
                layout_signature=layout_signature,
                arm_target_parameter=arm_target_parameter,
                arm_target_semantics=arm_target_semantics,
                finger_target=finger_target,
                starts_grasped=starts_grasped,
            )

    def record_clamp(clamp_delta: torch.Tensor) -> None:
        if clamp_evidence is None:
            return
        previous_maximum = torch.as_tensor(
            clamp_evidence.get("maximum_arm_target_clamp_delta", torch.zeros_like(clamp_delta)),
            device=env.device,
            dtype=clamp_delta.dtype,
        )
        previous_any = torch.as_tensor(
            clamp_evidence.get("any_arm_target_clamped", torch.zeros_like(clamp_delta, dtype=torch.bool)),
            device=env.device,
            dtype=torch.bool,
        )
        expected_shape = (env.num_envs,)
        if tuple(previous_maximum.shape) != expected_shape or tuple(previous_any.shape) != expected_shape:
            raise ValueError(f"Existing clamp evidence must contain per-world tensors with shape {expected_shape}.")
        clamp_evidence["maximum_arm_target_clamp_delta"] = torch.maximum(previous_maximum, clamp_delta)
        clamp_evidence["any_arm_target_clamped"] = previous_any | (clamp_delta > 1.0e-7)

    if replay_state is None:
        initial_arm_q, _, _, _ = env.read_robot_state()
    else:
        initial_arm_q = replay_state.arm_q
    initial_target, initial_clamp_delta = target_resolver(initial_arm_q, arm_target_parameter)
    record_clamp(initial_clamp_delta)
    initial_target_reference = initial_target.clone()
    maximum_target_drift = torch.zeros(env.num_envs, device=env.device, dtype=initial_target.dtype)
    if replay_evidence is not None:
        tracking_limits = _arm_target_tracking_limits(env, replay_state.arm_q.shape[1], replay_state.arm_q.dtype)
        replay_evidence["stored_maximum_arm_target_tracking_error_rad"] = _finite_maximum(
            torch.abs(initial_target - replay_state.arm_q),
            dimensions=(1,),
        )
        replay_evidence["stored_arm_target_tracking_error_by_joint_rad"] = torch.abs(
            initial_target - replay_state.arm_q
        )
        replay_evidence["stored_arm_target_tracking_bounded"] = (
            torch.abs(initial_target - replay_state.arm_q) <= tracking_limits
        ).all(dim=-1)
    last_target = initial_target

    def update(_step: int, _steps: int, _progress: float) -> None:
        nonlocal last_target
        current_arm_q, _, _, _ = env.read_robot_state()
        target, clamp_delta = target_resolver(current_arm_q, arm_target_parameter)
        record_clamp(clamp_delta)
        maximum_target_drift.copy_(
            torch.maximum(maximum_target_drift, torch.abs(target - initial_target_reference).amax(dim=-1))
        )
        last_target = target
        env.set_robot_targets(target, finger_target)

    if replay_evidence is None:
        steps = env.advance(duration_s, update)
    else:

        def post_step(_step: int, _steps: int, _progress: float) -> None:
            layout_signature = replay_evidence["_continuity"]["layout_signature"]
            state = _read_reset_replay_state(env, body_count=layout_signature[0])
            _sample_reset_replay_post_step(
                env,
                replay_evidence,
                state,
                arm_target=last_target,
                finger_target=finger_target,
            )

        steps = env.advance(duration_s, update, post_step=post_step)
    if clamp_evidence is not None:
        previous_drift = torch.as_tensor(
            clamp_evidence.get("maximum_arm_target_drift", torch.zeros_like(maximum_target_drift)),
            device=env.device,
            dtype=maximum_target_drift.dtype,
        )
        clamp_evidence["maximum_arm_target_drift"] = torch.maximum(previous_drift, maximum_target_drift)
    if replay_evidence is not None:
        replay_evidence["maximum_arm_target_drift_rad"] = torch.maximum(
            replay_evidence["maximum_arm_target_drift_rad"],
            maximum_target_drift,
        )
    return steps


def advance_reset_bias_hold(
    env: _ResetToolEnv,
    duration_s: float,
    arm_target_bias: torch.Tensor,
    finger_target: torch.Tensor,
    *,
    clamp_evidence: dict[str, Any] | None = None,
    replay_evidence: dict[str, Any] | None = None,
    starts_grasped: bool | None = None,
) -> int:
    """Replay the legacy measured-state-relative reset target convention.

    When ``clamp_evidence`` is provided, its per-world maximum clamp delta and
    clamp flag are accumulated across the initial state, every commanded step,
    and any prior calls that used the same dictionary. When ``replay_evidence``
    is provided, the helper records stored-state mechanics and every real
    post-step state/contact sample. Reusing that dictionary is accepted only
    for an exactly continuous hold on the same environment.

    Args:
        env: Reset-tool environment to advance.
        duration_s: Simulation duration [s].
        arm_target_bias: Measured-state-relative arm target bias [rad], shape
            ``(num_envs, arm_joint_count)``.
        finger_target: Finger joint targets [m], shape ``(num_envs, 2)``.
        clamp_evidence: Optional mutable evidence dictionary. The helper updates
            ``maximum_arm_target_clamp_delta`` [rad] and ``any_arm_target_clamped``.
        replay_evidence: Optional mutable mechanical evidence dictionary. It
            must be empty on first use; keys prefixed with ``_`` hold private
            continuity state and must not be serialized.
        starts_grasped: Required with ``replay_evidence``. Grasped rows require
            bilateral grasp and proxy contacts after every step; open rows
            require zero left/right proxy contacts after every step.

    Returns:
        Number of simulation steps executed.
    """
    return _advance_reset_target_hold(
        env,
        duration_s,
        arm_target_bias,
        finger_target,
        arm_target_semantics="measured-state-relative-reset-bias",
        target_resolver=lambda current_q, bias: runtime_reset_biased_arm_target(env, current_q, bias),
        clamp_evidence=clamp_evidence,
        replay_evidence=replay_evidence,
        starts_grasped=starts_grasped,
    )


def advance_reset_absolute_target_hold(
    env: _ResetToolEnv,
    duration_s: float,
    arm_target: torch.Tensor,
    finger_target: torch.Tensor,
    *,
    clamp_evidence: dict[str, Any] | None = None,
    replay_evidence: dict[str, Any] | None = None,
    starts_grasped: bool | None = None,
) -> int:
    """Replay the pick task's constant stored absolute target under zero action."""
    return _advance_reset_target_hold(
        env,
        duration_s,
        arm_target,
        finger_target,
        arm_target_semantics="persistent-absolute",
        target_resolver=lambda _current_q, target: runtime_persistent_arm_target(env, target),
        clamp_evidence=clamp_evidence,
        replay_evidence=replay_evidence,
        starts_grasped=starts_grasped,
    )


def exact_success_from_state(
    env: _ResetToolEnv,
    task_q: torch.Tensor,
    task_qd: torch.Tensor,
    goal_task_q: torch.Tensor,
    *,
    plug_body_index: int | None = None,
    latch_body_index: int | None = None,
) -> RJ45SuccessResult:
    """Evaluate the runtime success predicate with the live task thresholds."""
    plug_body_index, latch_body_index = _resolve_layout_body_indices(env, plug_body_index, latch_body_index)
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
        plug_body_index=plug_body_index,
        latch_body_index=latch_body_index,
    )


def _validated_lane_goal_gate_violations(
    value: object,
    *,
    num_envs: int,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    """Validate and copy one callback's reason-to-violation mapping."""
    if not isinstance(value, Mapping):
        raise TypeError("The per-step lane goal gate must return a mapping of reasons to Boolean tensors.")
    validated: dict[str, torch.Tensor] = {}
    for reason, mask in value.items():
        if not isinstance(reason, str) or not reason.strip() or reason != reason.strip():
            raise ValueError("Per-step lane goal-gate reasons must be non-empty, trimmed strings.")
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"Per-step lane goal-gate mask {reason!r} must be a torch.Tensor.")
        if mask.dtype != torch.bool:
            raise TypeError(f"Per-step lane goal-gate mask {reason!r} must have Boolean dtype.")
        if tuple(mask.shape) != (num_envs,):
            raise ValueError(
                f"Per-step lane goal-gate mask {reason!r} must have shape ({num_envs},), got {tuple(mask.shape)}."
            )
        validated[reason] = mask.to(device=device).clone()
    return validated


def _cloned_success_result(result: RJ45SuccessResult) -> RJ45SuccessResult:
    """Return callback-safe copies of one exact-success sample."""
    return RJ45SuccessResult(
        mask=result.mask.clone(),
        signed_axial_error=result.signed_axial_error.clone(),
        axial_error=result.axial_error.clone(),
        radial_error=result.radial_error.clone(),
        plug_angle_error=result.plug_angle_error.clone(),
        latch_angle_error=result.latch_angle_error.clone(),
        plug_spatial_speed=result.plug_spatial_speed.clone(),
    )


def _cloned_collision_metrics(result: CollisionMetrics | None) -> CollisionMetrics | None:
    """Return callback-safe copies of one collision sample when available."""
    if result is None:
        return None
    return CollisionMetrics(
        valid=result.valid.clone(),
        invalid_contact_count=result.invalid_contact_count.clone(),
        grasp_contact_count=result.grasp_contact_count.clone(),
        left_grasp_contact_count=result.left_grasp_contact_count.clone(),
        right_grasp_contact_count=result.right_grasp_contact_count.clone(),
        contact_overflow=result.contact_overflow,
        invalid_contact_pairs=result.invalid_contact_pairs,
    )


def _cloned_grasp_metrics(result: GraspMetrics | None) -> GraspMetrics | None:
    """Return callback-safe copies of one grasp sample when available."""
    if result is None:
        return None
    return GraspMetrics(
        valid=result.valid.clone(),
        tcp_distance=result.tcp_distance.clone(),
        bilateral_deflection=result.bilateral_deflection.clone(),
    )


def advance_exact_success_dwell(
    env: _ResetToolEnv,
    goal_task_q: torch.Tensor,
    arm_target_bias: torch.Tensor,
    finger_target: torch.Tensor,
    *,
    duration_s: float | None = None,
    require_all_samples: bool = False,
    sample_physical_validity: bool = False,
    arm_target_is_absolute: bool = False,
    plug_body_index: int | None = None,
    latch_body_index: int | None = None,
    lane_hold: _PerLaneTargetHold | None = None,
    per_step_lane_goal_gate: Callable[[Mapping[str, Any]], Mapping[str, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Hold zero action and sample exact insertion success after every step.

    By default the arm command follows the legacy measured-state reset-bias
    convention. ``arm_target_is_absolute=True`` instead holds the pick task's
    stored persistent target exactly. The task construction drive must remain
    disabled for the stored sample and every replay sample.

    When ``per_step_lane_goal_gate`` is supplied, it receives a copied
    post-step snapshot and returns ``{reason: violation_mask}``. Every mask is
    validated before any lane is mutated. Violating active lanes are frozen by
    ``lane_hold`` before the next physics step and remain failed even if later
    samples recover. Callback errors and malformed results are batch-fatal.
    """
    arm_target_bias = torch.as_tensor(arm_target_bias, device=env.device, dtype=torch.float32)
    finger_target = torch.as_tensor(finger_target, device=env.device, dtype=torch.float32)
    if tuple(arm_target_bias.shape) != (env.num_envs, 7):
        raise ValueError(f"arm_target_bias must have shape ({env.num_envs}, 7).")
    if tuple(finger_target.shape) != (env.num_envs, 2):
        raise ValueError(f"finger_target must have shape ({env.num_envs}, 2).")
    if per_step_lane_goal_gate is not None and lane_hold is None:
        raise ValueError("per_step_lane_goal_gate requires lane_hold so failed lanes can be frozen immediately.")

    def resolve_arm_target(current_arm_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if arm_target_is_absolute:
            return runtime_persistent_arm_target(env, arm_target_bias)
        return runtime_reset_biased_arm_target(env, current_arm_q, arm_target_bias)

    runtime = env.rj45_runtime
    if bool(wp.to_torch(runtime.drive_enabled).any()) or bool(wp.to_torch(runtime.orientation_hold_enabled).any()):
        raise RuntimeError("Exact RJ45 success evidence requires both task construction drives to be disabled.")
    plug_body_index, latch_body_index = _resolve_layout_body_indices(env, plug_body_index, latch_body_index)

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
    initial = exact_success_from_state(
        env,
        initial_q,
        initial_qd,
        goal_task_q,
        plug_body_index=plug_body_index,
        latch_body_index=latch_body_index,
    )
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
    maximum_plug_relative_latch_angle = plug_relative_latch_angle(
        initial_q,
        plug_body_index=plug_body_index,
        latch_body_index=latch_body_index,
    )
    maximum_plug_spatial_speed = initial.plug_spatial_speed.clone()
    initial_arm_q, _, _, _ = env.read_robot_state()
    last_arm_target, initial_clamp_delta = resolve_arm_target(initial_arm_q)
    initial_arm_target = last_arm_target.clone()
    maximum_arm_target_clamp_delta = initial_clamp_delta.clone()
    any_arm_target_clamped = initial_clamp_delta > 1.0e-7
    maximum_arm_target_drift = torch.zeros_like(initial_clamp_delta)
    tracking_limits = _arm_target_tracking_limits(env, initial_arm_q.shape[1], initial_arm_q.dtype)
    maximum_arm_target_tracking_error_by_joint = torch.abs(last_arm_target - initial_arm_q)
    all_samples_arm_target_tracking_bounded = (maximum_arm_target_tracking_error_by_joint <= tracking_limits).all(
        dim=-1
    )
    if lane_hold is not None:
        lane_hold.deactivate(~initial.mask, reason="recovery-dwell-initial-success")
        lane_hold.deactivate(
            ~all_samples_arm_target_tracking_bounded,
            reason="recovery-dwell-initial-tracking",
        )
    final = initial
    all_samples_collision_free = torch.ones_like(initial.mask)
    all_samples_bilateral_grasp = torch.ones_like(initial.mask)
    all_samples_finite = torch.ones_like(initial.mask)
    maximum_body_excursion = torch.zeros(env.num_envs, device=env.device)
    maximum_arm_joint_speed = torch.zeros_like(maximum_body_excursion)
    maximum_finger_joint_speed = torch.zeros_like(maximum_body_excursion)
    maximum_cable_linear_speed = torch.zeros_like(maximum_body_excursion)
    maximum_invalid_contact_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    minimum_left_grasp_contact_count = torch.zeros_like(maximum_invalid_contact_count)
    minimum_right_grasp_contact_count = torch.zeros_like(maximum_invalid_contact_count)
    all_samples_proxy_bilateral_contact = torch.ones_like(initial.mask)
    first_collision_failure_step = torch.full_like(maximum_invalid_contact_count, -1)
    first_bilateral_grasp_failure_step = torch.full_like(maximum_invalid_contact_count, -1)
    any_contact_overflow = False
    sampled_invalid_contact_pairs: list[str] = []
    goal_task_q_batched = _batched_goal_task_pose(goal_task_q, initial_q)
    lane_goal_gate_passed = torch.ones_like(initial.mask)
    lane_goal_gate_violation_masks: dict[str, torch.Tensor] = {}
    lane_goal_gate_first_failure_steps: dict[str, torch.Tensor] = {}

    if sample_physical_validity:
        # Reset replay deliberately clears both Newton contact buffers.  The
        # initial state is valid for finite/kinematic checks, but contacts do
        # not exist again until the first real solver step below.
        initial_arm_q, initial_arm_qd, initial_finger_q, initial_finger_qd = env.read_robot_state()
        del initial_arm_q, initial_finger_q
        all_samples_finite &= task_state_is_finite_and_normalized(initial_q, initial_qd)
        all_samples_finite &= torch.isfinite(initial_arm_qd).all(-1) & torch.isfinite(initial_finger_qd).all(-1)
        maximum_arm_joint_speed = torch.maximum(maximum_arm_joint_speed, torch.abs(initial_arm_qd).amax(-1))
        maximum_finger_joint_speed = torch.maximum(
            maximum_finger_joint_speed,
            torch.abs(initial_finger_qd).amax(-1),
        )
        maximum_body_excursion = torch.maximum(
            maximum_body_excursion,
            torch.linalg.vector_norm(initial_q[..., :3] - goal_task_q_batched[..., :3], dim=-1).amax(-1),
        )
        cable_slice = runtime.layout.cable_body_slice
        maximum_cable_linear_speed = torch.maximum(
            maximum_cable_linear_speed,
            torch.linalg.vector_norm(initial_qd[:, cable_slice, :3], dim=-1).amax(-1),
        )

    for _step in range(sample_steps):
        current_arm_q, _, _, _ = env.read_robot_state()
        last_arm_target, clamp_delta = resolve_arm_target(current_arm_q)
        maximum_arm_target_clamp_delta = torch.maximum(maximum_arm_target_clamp_delta, clamp_delta)
        any_arm_target_clamped |= clamp_delta > 1.0e-7
        maximum_arm_target_drift = torch.maximum(
            maximum_arm_target_drift,
            torch.abs(last_arm_target - initial_arm_target).amax(dim=-1),
        )
        env.set_robot_targets(last_arm_target, finger_target)
        applied_arm_target = lane_hold.last_sent_arm_target if lane_hold is not None else last_arm_target
        applied_finger_target = lane_hold.last_sent_finger_target if lane_hold is not None else finger_target
        env.advance(env.advance_dt)
        if bool(wp.to_torch(runtime.drive_enabled).any()) or bool(wp.to_torch(runtime.orientation_hold_enabled).any()):
            raise RuntimeError("A task construction drive became enabled during exact success evidence.")
        task_q, task_qd = env.read_task_state()
        post_arm_q, post_arm_qd, post_finger_q, post_finger_qd = env.read_robot_state()
        tracking_error = torch.abs(applied_arm_target - post_arm_q)
        maximum_arm_target_tracking_error_by_joint = torch.maximum(
            maximum_arm_target_tracking_error_by_joint,
            tracking_error,
        )
        all_samples_arm_target_tracking_bounded &= (tracking_error <= tracking_limits).all(dim=-1)
        if lane_hold is not None:
            lane_hold.deactivate(
                ~(tracking_error <= tracking_limits).all(dim=-1),
                reason="recovery-dwell-tracking",
            )
        final = exact_success_from_state(
            env,
            task_q,
            task_qd,
            goal_task_q,
            plug_body_index=plug_body_index,
            latch_body_index=latch_body_index,
        )
        all_samples_success &= final.mask
        all_post_step_success &= final.mask
        if lane_hold is not None:
            lane_hold.deactivate(~final.mask, reason="recovery-dwell-exact-success")
        consecutive_steps = torch.where(final.mask, consecutive_steps + 1, torch.zeros_like(consecutive_steps))
        maximum_consecutive_steps = torch.maximum(maximum_consecutive_steps, consecutive_steps)
        maximum_signed_axial_error = torch.maximum(maximum_signed_axial_error, final.signed_axial_error)
        minimum_signed_axial_error = torch.minimum(minimum_signed_axial_error, final.signed_axial_error)
        maximum_axial_error = torch.maximum(maximum_axial_error, final.axial_error)
        maximum_radial_error = torch.maximum(maximum_radial_error, final.radial_error)
        maximum_plug_angle_error = torch.maximum(maximum_plug_angle_error, final.plug_angle_error)
        maximum_latch_angle_error = torch.maximum(maximum_latch_angle_error, final.latch_angle_error)
        sample_plug_relative_latch_angle = plug_relative_latch_angle(
            task_q,
            plug_body_index=plug_body_index,
            latch_body_index=latch_body_index,
        )
        maximum_plug_relative_latch_angle = torch.maximum(
            maximum_plug_relative_latch_angle,
            sample_plug_relative_latch_angle,
        )
        maximum_plug_spatial_speed = torch.maximum(maximum_plug_spatial_speed, final.plug_spatial_speed)
        sample_finite = task_state_is_finite_and_normalized(task_q, task_qd)
        sample_finite &= torch.isfinite(post_arm_q).all(-1) & torch.isfinite(post_arm_qd).all(-1)
        sample_finite &= torch.isfinite(post_finger_q).all(-1) & torch.isfinite(post_finger_qd).all(-1)
        sample_body_excursion = torch.nan_to_num(
            torch.linalg.vector_norm(task_q[..., :3] - initial_q[..., :3], dim=-1).amax(-1),
            nan=torch.inf,
            posinf=torch.inf,
            neginf=torch.inf,
        )
        sample_plug_excursion = torch.nan_to_num(
            torch.linalg.vector_norm(
                task_q[:, plug_body_index, :3] - initial_q[:, plug_body_index, :3],
                dim=-1,
            ),
            nan=torch.inf,
            posinf=torch.inf,
            neginf=torch.inf,
        )
        socket_body_index = getattr(runtime.layout, "socket_body_index", None)
        if socket_body_index is None:
            sample_socket_excursion = torch.zeros_like(sample_plug_excursion)
        else:
            sample_socket_excursion = torch.nan_to_num(
                torch.linalg.vector_norm(
                    task_q[:, int(socket_body_index), :3] - initial_q[:, int(socket_body_index), :3],
                    dim=-1,
                ),
                nan=torch.inf,
                posinf=torch.inf,
                neginf=torch.inf,
            )
        cable_slice = runtime.layout.cable_body_slice
        sample_cable_linear_speed = torch.nan_to_num(
            torch.linalg.vector_norm(task_qd[:, cable_slice, :3], dim=-1).amax(-1),
            nan=torch.inf,
            posinf=torch.inf,
            neginf=torch.inf,
        )
        sample_arm_joint_speed = torch.nan_to_num(
            torch.abs(post_arm_qd).amax(-1),
            nan=torch.inf,
            posinf=torch.inf,
            neginf=torch.inf,
        )
        sample_finger_joint_speed = torch.nan_to_num(
            torch.abs(post_finger_qd).amax(-1),
            nan=torch.inf,
            posinf=torch.inf,
            neginf=torch.inf,
        )
        collision: CollisionMetrics | None = None
        grasp: GraspMetrics | None = None
        proxy_bilateral: torch.Tensor | None = None
        if sample_physical_validity:
            collision, grasp, proxy_bilateral = _physical_validity_sample(env, finger_target)
            if collision.contact_overflow:
                raise RuntimeError("Global contact-buffer overflow during exact-success recovery dwell.")
            all_samples_collision_free &= collision.valid
            all_samples_proxy_bilateral_contact &= proxy_bilateral
            bilateral_grasp = grasp.valid & proxy_bilateral
            all_samples_bilateral_grasp &= bilateral_grasp
            all_samples_finite &= sample_finite
            if lane_hold is not None:
                lane_hold.deactivate(~collision.valid, reason="recovery-dwell-collision")
                lane_hold.deactivate(~bilateral_grasp, reason="recovery-dwell-lost-bilateral-grasp")
                lane_hold.deactivate(~sample_finite, reason="recovery-dwell-non-finite")
            if _step == 0:
                minimum_left_grasp_contact_count = collision.left_grasp_contact_count.clone()
                minimum_right_grasp_contact_count = collision.right_grasp_contact_count.clone()
            else:
                minimum_left_grasp_contact_count = torch.minimum(
                    minimum_left_grasp_contact_count,
                    collision.left_grasp_contact_count,
                )
                minimum_right_grasp_contact_count = torch.minimum(
                    minimum_right_grasp_contact_count,
                    collision.right_grasp_contact_count,
                )
            collision_failure = ~collision.valid & (first_collision_failure_step < 0)
            first_collision_failure_step = torch.where(
                collision_failure,
                torch.full_like(first_collision_failure_step, _step + 1),
                first_collision_failure_step,
            )
            grasp_failure = ~bilateral_grasp & (first_bilateral_grasp_failure_step < 0)
            first_bilateral_grasp_failure_step = torch.where(
                grasp_failure,
                torch.full_like(first_bilateral_grasp_failure_step, _step + 1),
                first_bilateral_grasp_failure_step,
            )
            any_contact_overflow |= collision.contact_overflow
            maximum_invalid_contact_count = torch.maximum(
                maximum_invalid_contact_count,
                collision.invalid_contact_count,
            )
            for pair in collision.invalid_contact_pairs:
                if pair not in sampled_invalid_contact_pairs and len(sampled_invalid_contact_pairs) < 64:
                    sampled_invalid_contact_pairs.append(pair)
            maximum_arm_joint_speed = torch.maximum(maximum_arm_joint_speed, sample_arm_joint_speed)
            maximum_finger_joint_speed = torch.maximum(
                maximum_finger_joint_speed,
                sample_finger_joint_speed,
            )
            maximum_body_excursion = torch.maximum(
                maximum_body_excursion,
                torch.linalg.vector_norm(task_q[..., :3] - goal_task_q_batched[..., :3], dim=-1).amax(-1),
            )
            maximum_cable_linear_speed = torch.maximum(
                maximum_cable_linear_speed,
                sample_cable_linear_speed,
            )

        if per_step_lane_goal_gate is not None:
            snapshot: dict[str, Any] = {
                "step": _step + 1,
                "sample_steps": sample_steps,
                "task_q": task_q.clone(),
                "task_qd": task_qd.clone(),
                "initial_task_q": initial_q.clone(),
                "goal_task_q": goal_task_q_batched.clone(),
                "arm_q": post_arm_q.clone(),
                "arm_qd": post_arm_qd.clone(),
                "finger_q": post_finger_q.clone(),
                "finger_qd": post_finger_qd.clone(),
                "arm_target": applied_arm_target.clone(),
                "finger_target": applied_finger_target.clone(),
                "arm_target_clamp_delta": clamp_delta.clone(),
                "arm_target_drift": torch.abs(applied_arm_target - initial_arm_target).amax(dim=-1),
                "arm_target_tracking_error_by_joint": tracking_error.clone(),
                "arm_target_tracking_bounded": (tracking_error <= tracking_limits).all(dim=-1),
                "state_finite": sample_finite.clone(),
                "body_excursion": sample_body_excursion.clone(),
                "plug_excursion": sample_plug_excursion.clone(),
                "socket_excursion": sample_socket_excursion.clone(),
                "cable_linear_speed": sample_cable_linear_speed.clone(),
                "arm_joint_speed": sample_arm_joint_speed.clone(),
                "finger_joint_speed": sample_finger_joint_speed.clone(),
                "plug_relative_latch_angle": sample_plug_relative_latch_angle.clone(),
                "exact_success": _cloned_success_result(final),
                "collision": _cloned_collision_metrics(collision),
                "grasp": _cloned_grasp_metrics(grasp),
                "proxy_bilateral_contact": None if proxy_bilateral is None else proxy_bilateral.clone(),
                "active_mask": lane_hold.active_mask,
            }
            violations = _validated_lane_goal_gate_violations(
                per_step_lane_goal_gate(snapshot),
                num_envs=env.num_envs,
                device=env.device,
            )
            active_before_goal_gate = lane_hold.active_mask
            for reason, violation_mask in violations.items():
                effective_violation = active_before_goal_gate & violation_mask
                accumulated = lane_goal_gate_violation_masks.setdefault(
                    reason,
                    torch.zeros_like(initial.mask),
                )
                accumulated.logical_or_(effective_violation)
                first_steps = lane_goal_gate_first_failure_steps.setdefault(
                    reason,
                    torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long),
                )
                new_first_failure = effective_violation & (first_steps < 0)
                first_steps.copy_(
                    torch.where(
                        new_first_failure,
                        torch.full_like(first_steps, _step + 1),
                        first_steps,
                    )
                )
                lane_goal_gate_passed &= ~effective_violation
            for reason, violation_mask in violations.items():
                lane_hold.deactivate(active_before_goal_gate & violation_mask, reason=f"recovery-dwell-goal:{reason}")

    dwell_satisfied = consecutive_steps >= required_steps
    passed = initial.mask & dwell_satisfied
    if require_all_samples:
        passed &= all_samples_success
    passed &= lane_goal_gate_passed
    if lane_hold is not None:
        passed &= lane_hold.active_mask
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
        "maximum_plug_relative_latch_angle": maximum_plug_relative_latch_angle,
        "maximum_plug_spatial_speed": maximum_plug_spatial_speed,
        "sampled_physical_validity": sample_physical_validity,
        "physical_validity_post_step_samples": sample_steps if sample_physical_validity else 0,
        "all_samples_collision_free": all_samples_collision_free,
        "all_samples_bilateral_grasp": all_samples_bilateral_grasp,
        "all_samples_proxy_bilateral_contact": all_samples_proxy_bilateral_contact,
        "all_samples_finite": all_samples_finite,
        "maximum_body_excursion": maximum_body_excursion,
        "maximum_arm_joint_speed": maximum_arm_joint_speed,
        "maximum_finger_joint_speed": maximum_finger_joint_speed,
        "maximum_cable_linear_speed": maximum_cable_linear_speed,
        "maximum_invalid_contact_count": maximum_invalid_contact_count,
        "minimum_left_grasp_contact_count": minimum_left_grasp_contact_count,
        "minimum_right_grasp_contact_count": minimum_right_grasp_contact_count,
        "first_collision_failure_step": first_collision_failure_step,
        "first_bilateral_grasp_failure_step": first_bilateral_grasp_failure_step,
        "any_contact_overflow": any_contact_overflow,
        "sampled_invalid_contact_pairs": tuple(sampled_invalid_contact_pairs),
        "maximum_arm_target_clamp_delta": maximum_arm_target_clamp_delta,
        "any_arm_target_clamped": any_arm_target_clamped,
        "arm_target_semantics": (
            "persistent-absolute" if arm_target_is_absolute else "measured-state-relative-reset-bias"
        ),
        "maximum_arm_target_drift": maximum_arm_target_drift,
        "maximum_arm_target_tracking_error_by_joint": maximum_arm_target_tracking_error_by_joint,
        "all_samples_arm_target_tracking_bounded": all_samples_arm_target_tracking_bounded,
        "last_arm_target": last_arm_target,
        "lane_goal_gate_passed": lane_goal_gate_passed,
        "lane_goal_gate_violation_masks": {
            reason: mask.clone() for reason, mask in lane_goal_gate_violation_masks.items()
        },
        "lane_goal_gate_first_failure_steps": {
            reason: steps.clone() for reason, steps in lane_goal_gate_first_failure_steps.items()
        },
        "lane_hold_active_mask": (lane_hold.active_mask if lane_hold is not None else torch.ones_like(initial.mask)),
        "lane_hold_failed_mask": (lane_hold.failed_mask if lane_hold is not None else torch.zeros_like(initial.mask)),
        "lane_hold_reason_masks": lane_hold.reason_masks if lane_hold is not None else {},
    }


def _recovery_cartesian_c2_progress(progress: float) -> float:
    """Map unit time to the immutable recovery path's C2 endpoint schedule."""
    if not math.isfinite(progress):
        raise ValueError("Recovery Cartesian schedule progress must be finite.")
    progress = min(max(float(progress), 0.0), 1.0)
    ramp = float(PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["execution"]["c2_ramp_fraction"])
    peak_rate = 1.0 / (1.0 - ramp)

    def ramp_integral(value: float) -> float:
        normalized = value / ramp
        return peak_rate * ramp * (normalized**3 - 0.5 * normalized**4)

    if progress < ramp:
        return ramp_integral(progress)
    if progress > 1.0 - ramp:
        return 1.0 - ramp_integral(1.0 - progress)
    return peak_rate * (0.5 * ramp + progress - ramp)


def _interpolate_recovery_cartesian_knots(knot_targets: torch.Tensor, path_progress: float) -> torch.Tensor:
    """Interpolate a monotone unit path through precomputed recovery knots."""
    if knot_targets.ndim != 3 or knot_targets.shape[0] < 2:
        raise ValueError("Recovery Cartesian knots must have shape (K + 1, N, J) with K >= 1.")
    if not math.isfinite(path_progress):
        raise ValueError("Recovery Cartesian path progress must be finite.")
    path_progress = min(max(float(path_progress), 0.0), 1.0)
    knot_count = knot_targets.shape[0] - 1
    coordinate = path_progress * knot_count
    lower = min(int(math.floor(coordinate)), knot_count - 1)
    return torch.lerp(knot_targets[lower], knot_targets[lower + 1], coordinate - lower)


def _pick_insert_recovery_plug_route(
    live_plug: torch.Tensor,
    goal_plug: torch.Tensor,
    *,
    phase: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Return the goal-stopped task-frame route and insertion-corridor preflight mask."""
    expected_shape = tuple(live_plug.shape)
    if live_plug.ndim != 2 or live_plug.shape[-1] != 7:
        raise ValueError(f"Live plug poses must have shape (N, 7), got {expected_shape}.")
    if tuple(goal_plug.shape) != expected_shape:
        raise ValueError(f"goal_plug must have shape {expected_shape}, got {tuple(goal_plug.shape)}.")
    route_mode = PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["phase_route_modes"].get(str(phase))
    if route_mode is None:
        raise ValueError(f"pick_insert_phase must be one of 0 through 5, got {phase!r}.")
    finite = torch.isfinite(live_plug).all(dim=-1) & torch.isfinite(goal_plug).all(dim=-1)
    if route_mode == "insertion-corridor":
        corridor = PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["insertion_corridor"]
        error_local = math_utils.quat_apply_inverse(
            goal_plug[:, 3:7],
            live_plug[:, :3] - goal_plug[:, :3],
        )
        axial_before_goal = -error_local[:, 1]
        radial_error = torch.linalg.vector_norm(error_local[:, (0, 2)], dim=-1)
        orientation_error = math_utils.quat_error_magnitude(live_plug[:, 3:7], goal_plug[:, 3:7])
        preflight = (
            finite
            & (axial_before_goal <= float(corridor["maximum_axial_distance_before_goal_m"]))
            & (axial_before_goal >= -float(corridor["maximum_axial_overtravel_m"]))
            & (radial_error <= float(corridor["maximum_radial_error_m"]))
            & (orientation_error <= float(corridor["maximum_orientation_error_rad"]))
        )
        return (goal_plug.clone(),), preflight
    if route_mode != "clearance-via-preinsert":
        raise RuntimeError(f"Unsupported immutable pick-insert recovery route mode {route_mode!r}.")

    clearance = PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["clearance_route"]
    preinsert = goal_plug.clone()
    preinsert_local = torch.zeros_like(preinsert[:, :3])
    preinsert_local[:, 1] = -float(clearance["preinsert_axial_offset_m"])
    preinsert[:, :3] += math_utils.quat_apply(goal_plug[:, 3:7], preinsert_local)
    transport_height = torch.maximum(
        live_plug[:, 2],
        preinsert[:, 2] + float(clearance["clearance_above_preinsert_m"]),
    )
    vertical_lift = live_plug.clone()
    vertical_lift[:, 2] = transport_height
    high_midpoint = preinsert.clone()
    high_midpoint[:, :2] = 0.5 * (live_plug[:, :2] + preinsert[:, :2])
    high_midpoint[:, 2] = transport_height
    overhead_preinsert = preinsert.clone()
    overhead_preinsert[:, 2] = transport_height
    return (
        vertical_lift,
        high_midpoint,
        overhead_preinsert,
        preinsert,
        goal_plug.clone(),
    ), finite


def _plug_route_to_tcp_targets(
    plug_targets: tuple[torch.Tensor, ...],
    *,
    grasp_offset: torch.Tensor,
    grasp_orientation: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Convert task-frame plug route knots into nominal task-frame TCP targets."""
    return tuple(
        torch.cat(
            (
                plug[:, :3] + math_utils.quat_apply(plug[:, 3:7], grasp_offset),
                math_utils.quat_unique(math_utils.quat_mul(plug[:, 3:7], grasp_orientation)),
            ),
            dim=-1,
        )
        for plug in plug_targets
    )


@dataclass(frozen=True)
class _RecoveryCartesianPlan:
    """One precomputed, lane-safe Cartesian recovery route in joint-target space."""

    segment_knots: tuple[torch.Tensor, ...]
    segment_waypoint_counts: tuple[int, ...]
    pre_densification_waypoint_count: torch.Tensor
    post_densification_waypoint_count: torch.Tensor
    terminal_raw: torch.Tensor
    terminal_target: torch.Tensor
    start_preload_bias: torch.Tensor
    goal_preload_bias: torch.Tensor
    preload_bias_difference: torch.Tensor
    maximum_raw_ik_joint_step: torch.Tensor
    maximum_commanded_joint_step_before_densification: torch.Tensor
    maximum_commanded_joint_step_after_densification: torch.Tensor
    command_densification_required_subknot_count: torch.Tensor
    command_densification_executed_subknot_count: torch.Tensor
    start_target_anchor_error: torch.Tensor
    canonical_endpoint_anchor_error: torch.Tensor
    maximum_segment_boundary_command_jump: torch.Tensor
    ik_valid: torch.Tensor
    target_valid: torch.Tensor

    @property
    def waypoint_count(self) -> int:
        return sum(self.segment_waypoint_counts)


@dataclass(frozen=True)
class _RecoveryCommandSchedule:
    """Bias-anchored and deterministically densified command knots."""

    segment_knots: tuple[torch.Tensor, ...]
    segment_waypoint_counts: tuple[int, ...]
    pre_densification_waypoint_count: torch.Tensor
    post_densification_waypoint_count: torch.Tensor
    start_preload_bias: torch.Tensor
    goal_preload_bias: torch.Tensor
    preload_bias_difference: torch.Tensor
    maximum_step_before_densification: torch.Tensor
    maximum_step_after_densification: torch.Tensor
    required_subknot_count: torch.Tensor
    executed_subknot_count: torch.Tensor
    start_target_anchor_error: torch.Tensor
    canonical_endpoint_anchor_error: torch.Tensor
    maximum_segment_boundary_jump: torch.Tensor


def _recovery_bias_blended_command_schedule(
    raw_segments: tuple[torch.Tensor, ...],
    current_target: torch.Tensor,
    endpoint_target: torch.Tensor | None,
    active_mask: torch.Tensor,
    *,
    maximum_command_step_rad: float,
    maximum_waypoint_count: int | None = None,
) -> _RecoveryCommandSchedule:
    """Anchor raw IK knots at both endpoints and densify the unchanged joint curve."""
    if not raw_segments or any(segment.ndim != 3 or segment.shape[0] < 2 for segment in raw_segments):
        raise ValueError("Recovery raw segments must contain (K + 1, N, J) knots with K >= 1.")
    reference_shape = tuple(raw_segments[0].shape[1:])
    if any(tuple(segment.shape[1:]) != reference_shape for segment in raw_segments):
        raise ValueError("Recovery raw segments must share one (N, J) lane/joint shape.")
    if tuple(current_target.shape) != reference_shape:
        raise ValueError(f"current_target must have shape {reference_shape}, got {tuple(current_target.shape)}.")
    if endpoint_target is not None and tuple(endpoint_target.shape) != reference_shape:
        raise ValueError(f"endpoint_target must have shape {reference_shape}, got {tuple(endpoint_target.shape)}.")
    active_mask = torch.as_tensor(active_mask, device=current_target.device, dtype=torch.bool)
    if tuple(active_mask.shape) != (reference_shape[0],):
        raise ValueError(f"active_mask must have shape ({reference_shape[0]},).")
    maximum_command_step_rad = float(maximum_command_step_rad)
    if not math.isfinite(maximum_command_step_rad) or maximum_command_step_rad <= 0.0:
        raise ValueError("maximum_command_step_rad must be finite and positive.")
    if maximum_waypoint_count is not None and maximum_waypoint_count <= 0:
        raise ValueError("maximum_waypoint_count must be positive when provided.")

    raw_start = raw_segments[0][0]
    raw_goal = raw_segments[-1][-1]
    start_bias = current_target - raw_start
    goal_bias = start_bias if endpoint_target is None else endpoint_target - raw_goal
    bias_difference = goal_bias - start_bias
    unique_interval_count = sum(segment.shape[0] - 1 for segment in raw_segments)
    command_segments: list[torch.Tensor] = []
    interval_offset = 0
    previous_terminal: torch.Tensor | None = None
    for raw_segment in raw_segments:
        interval_count = raw_segment.shape[0] - 1
        alpha = (
            torch.arange(
                interval_offset,
                interval_offset + interval_count + 1,
                device=current_target.device,
                dtype=current_target.dtype,
            )
            / unique_interval_count
        )
        command_segment = raw_segment + start_bias + alpha[:, None, None] * bias_difference
        command_segment = torch.where(active_mask[None, :, None], command_segment, current_target)
        if previous_terminal is None:
            command_segment[0] = current_target
        else:
            command_segment[0] = previous_terminal
        previous_terminal = command_segment[-1].clone()
        command_segments.append(command_segment)
        interval_offset += interval_count
    if endpoint_target is not None:
        command_segments[-1][-1] = torch.where(active_mask[:, None], endpoint_target, current_target)

    zero = torch.zeros(reference_shape[0], device=current_target.device, dtype=current_target.dtype)
    maximum_before = zero.clone()
    maximum_boundary_jump = zero.clone()
    previous = current_target
    for segment in command_segments:
        maximum_before = torch.maximum(
            maximum_before,
            torch.abs(segment[1:] - segment[:-1]).amax(dim=(0, 2)),
        )
        seam = torch.abs(segment[0] - previous).amax(dim=-1)
        maximum_before = torch.maximum(maximum_before, seam)
        maximum_boundary_jump = torch.maximum(maximum_boundary_jump, seam)
        previous = segment[-1]

    required_subknots = torch.zeros(reference_shape[0], device=current_target.device, dtype=torch.long)
    segment_division_counts: list[list[int]] = []
    for segment in command_segments:
        division_counts: list[int] = []
        for interval_start, interval_end in zip(segment[:-1], segment[1:], strict=True):
            interval_delta = torch.abs(interval_end - interval_start).amax(dim=-1)
            lane_division_ratio = interval_delta / maximum_command_step_rad
            if maximum_waypoint_count is not None:
                lane_division_ratio = torch.nan_to_num(
                    lane_division_ratio,
                    nan=maximum_waypoint_count + 1,
                    posinf=maximum_waypoint_count + 1,
                    neginf=maximum_waypoint_count + 1,
                ).clamp_max(maximum_waypoint_count + 1)
            lane_divisions = torch.ceil(lane_division_ratio).to(torch.long).clamp_min(1)
            required_subknots += torch.where(active_mask, lane_divisions - 1, torch.zeros_like(lane_divisions))
            active_delta = torch.where(active_mask, interval_delta, torch.zeros_like(interval_delta))
            if maximum_waypoint_count is not None:
                maximum_division_ratio = torch.nan_to_num(
                    active_delta.amax() / maximum_command_step_rad,
                    nan=maximum_waypoint_count + 1,
                    posinf=maximum_waypoint_count + 1,
                    neginf=maximum_waypoint_count + 1,
                ).clamp_max(maximum_waypoint_count + 1)
                division_count = max(1, int(math.ceil(float(maximum_division_ratio))))
            else:
                division_count = max(
                    1,
                    int(math.ceil(float(active_delta.amax()) / maximum_command_step_rad)),
                )
            division_counts.append(division_count)
        segment_division_counts.append(division_counts)

    planned_dense_waypoint_count = sum(sum(counts) for counts in segment_division_counts)
    cap_exceeded = maximum_waypoint_count is not None and planned_dense_waypoint_count > maximum_waypoint_count
    densified_segments: list[torch.Tensor] = []
    if not cap_exceeded:
        corrected_dense_waypoint_count = 0
        for segment, division_counts in zip(command_segments, segment_division_counts, strict=True):
            dense = [segment[0]]
            for interval_start, interval_end, division_count in zip(
                segment[:-1],
                segment[1:],
                division_counts,
                strict=True,
            ):
                while True:
                    interval_knots = [
                        torch.lerp(interval_start, interval_end, subdivision / division_count)
                        for subdivision in range(1, division_count + 1)
                    ]
                    observed_step = torch.stack(
                        [
                            torch.abs(right - left).amax(dim=-1)
                            for left, right in zip((interval_start, *interval_knots[:-1]), interval_knots, strict=True)
                        ]
                    ).amax(dim=0)
                    if bool((~active_mask | (observed_step <= maximum_command_step_rad)).all()):
                        break
                    division_count += 1
                dense.extend(interval_knots)
                corrected_dense_waypoint_count += division_count
            densified_segments.append(torch.stack(dense))
        planned_dense_waypoint_count = corrected_dense_waypoint_count
        cap_exceeded = maximum_waypoint_count is not None and planned_dense_waypoint_count > maximum_waypoint_count
    if cap_exceeded:
        # The caller deactivates these lanes before execution.  Keep a bounded
        # finite fallback schedule while reporting the capped route count.
        densified_segments = command_segments

    executed_subknots_per_route = 0 if cap_exceeded else max(planned_dense_waypoint_count - unique_interval_count, 0)

    maximum_after = zero.clone()
    previous = current_target
    for segment in densified_segments:
        maximum_after = torch.maximum(
            maximum_after,
            torch.abs(segment[1:] - segment[:-1]).amax(dim=(0, 2)),
        )
        maximum_after = torch.maximum(maximum_after, torch.abs(segment[0] - previous).amax(dim=-1))
        previous = segment[-1]
    endpoint_error = zero.clone()
    if endpoint_target is not None:
        endpoint_error = torch.where(
            active_mask,
            torch.abs(densified_segments[-1][-1] - endpoint_target).amax(dim=-1),
            zero,
        )
    return _RecoveryCommandSchedule(
        segment_knots=tuple(densified_segments),
        segment_waypoint_counts=tuple(segment.shape[0] - 1 for segment in densified_segments),
        pre_densification_waypoint_count=torch.where(
            active_mask,
            torch.full_like(required_subknots, unique_interval_count),
            torch.zeros_like(required_subknots),
        ),
        post_densification_waypoint_count=torch.where(
            active_mask,
            torch.full_like(required_subknots, planned_dense_waypoint_count),
            torch.zeros_like(required_subknots),
        ),
        start_preload_bias=torch.where(active_mask[:, None], start_bias, torch.zeros_like(start_bias)),
        goal_preload_bias=torch.where(active_mask[:, None], goal_bias, torch.zeros_like(goal_bias)),
        preload_bias_difference=torch.where(
            active_mask[:, None],
            bias_difference,
            torch.zeros_like(bias_difference),
        ),
        maximum_step_before_densification=maximum_before,
        maximum_step_after_densification=maximum_after,
        required_subknot_count=required_subknots,
        executed_subknot_count=torch.where(
            active_mask,
            torch.full_like(required_subknots, executed_subknots_per_route),
            torch.zeros_like(required_subknots),
        ),
        start_target_anchor_error=torch.abs(densified_segments[0][0] - current_target).amax(dim=-1),
        canonical_endpoint_anchor_error=endpoint_error,
        maximum_segment_boundary_jump=maximum_boundary_jump,
    )


def _plan_recovery_cartesian_route(
    env: _ResetToolEnv,
    ik: FrankaResetIK,
    tcp_targets: tuple[torch.Tensor, ...],
    finger_target: torch.Tensor,
    *,
    current_tcp: torch.Tensor,
    current_raw: torch.Tensor,
    current_target: torch.Tensor,
    lane_hold: _PerLaneTargetHold,
    reason_prefix: str,
    endpoint_arm_target: torch.Tensor | None,
    active_mask: torch.Tensor | None = None,
) -> _RecoveryCartesianPlan:
    """Precompute one bounded sequential-IK route and anchor its actuator targets."""
    if not tcp_targets:
        raise ValueError("A recovery Cartesian route requires at least one target pose.")
    planning = PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["planning"]
    maximum_translation_step = float(planning["maximum_translation_step_m"])
    maximum_rotation_step = float(planning["maximum_rotation_step_rad"])
    maximum_raw_step = float(planning["maximum_raw_ik_joint_step_rad"])
    maximum_commanded_step = float(planning["maximum_commanded_joint_step_rad"])
    densification_step_limit = float(planning["densification_step_limit_rad"])
    maximum_waypoints = int(planning["maximum_waypoints"])
    margin = float(planning["joint_limit_margin_rad"])
    expected_planning_contract = {
        "maximum_waypoints_scope": "post-densification-executed-global-unique-knots",
        "endpoint_command_bias_policy": "linear-start-to-goal-over-global-unique-route-knots",
        "exact_start_target": True,
        "exact_canonical_endpoint": True,
        "command_interval_densification": "deterministic-collinear-joint-subknots",
        "compensation_bias_policy": "constant-start-bias",
    }
    for name, expected in expected_planning_contract.items():
        if planning.get(name) != expected:
            raise RuntimeError(
                f"Unsupported Cartesian recovery planning contract {name}={planning.get(name)!r}; "
                f"expected {expected!r}."
            )
    if densification_step_limit != maximum_commanded_step:
        raise RuntimeError("Cartesian recovery densification and commanded-joint-step limits must be identical.")
    true_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    if active_mask is None:
        planning_active = lane_hold.active_mask
    else:
        active_mask = torch.as_tensor(active_mask, device=env.device, dtype=torch.bool)
        if tuple(active_mask.shape) != (env.num_envs,):
            raise ValueError(f"active_mask must have shape ({env.num_envs},).")
        planning_active = lane_hold.active_mask & active_mask
    ik_valid = true_mask.clone()
    target_valid = true_mask.clone()
    maximum_observed_raw_step = torch.zeros(env.num_envs, device=env.device)

    fallback_tcp = torch.zeros_like(current_tcp)
    fallback_tcp[:, 6] = 1.0
    current_tcp_finite = torch.isfinite(current_tcp).all(dim=-1)
    current_raw_finite = torch.isfinite(current_raw).all(dim=-1)
    active_before_preflight = planning_active & lane_hold.active_mask
    lane_hold.deactivate(
        active_before_preflight & ~(current_tcp_finite & current_raw_finite),
        reason=f"{reason_prefix}-preflight",
    )
    safe_current_tcp = torch.where(current_tcp_finite[:, None], current_tcp, fallback_tcp)
    safe_current_raw = torch.where(current_raw_finite[:, None], current_raw, lane_hold.last_sent_arm_target)
    safe_tcp_targets: list[torch.Tensor] = []
    prior_safe_tcp = safe_current_tcp
    for tcp_target in tcp_targets:
        target_finite = torch.isfinite(tcp_target).all(dim=-1)
        lane_hold.deactivate(
            planning_active & lane_hold.active_mask & ~target_finite,
            reason=f"{reason_prefix}-preflight",
        )
        safe_target = torch.where(target_finite[:, None], tcp_target, prior_safe_tcp)
        safe_tcp_targets.append(safe_target)
        prior_safe_tcp = safe_target
    safe_tcp_targets_tuple = tuple(safe_tcp_targets)
    segment_starts = (safe_current_tcp, *safe_tcp_targets_tuple[:-1])
    lane_segment_counts: list[torch.Tensor] = []
    lane_total = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    for segment_start, segment_target in zip(segment_starts, safe_tcp_targets_tuple, strict=True):
        translation = torch.linalg.vector_norm(segment_target[:, :3] - segment_start[:, :3], dim=-1)
        rotation = math_utils.quat_error_magnitude(segment_target[:, 3:7], segment_start[:, 3:7])
        finite_distance = torch.isfinite(translation) & torch.isfinite(rotation)
        lane_steps = torch.maximum(
            torch.ceil(torch.nan_to_num(translation, nan=torch.inf) / maximum_translation_step).to(torch.long),
            torch.ceil(torch.nan_to_num(rotation, nan=torch.inf) / maximum_rotation_step).to(torch.long),
        ).clamp_min(1)
        lane_steps = torch.where(finite_distance, lane_steps, torch.full_like(lane_steps, maximum_waypoints + 1))
        lane_segment_counts.append(lane_steps)
        lane_total += lane_steps
    lane_hold.deactivate(
        planning_active & (lane_total > maximum_waypoints),
        reason=f"{reason_prefix}-waypoint-cap",
    )
    segment_counts = tuple(
        max(
            1,
            int(
                torch.where(
                    planning_active & lane_hold.active_mask,
                    count,
                    torch.zeros_like(count),
                )
                .amax()
                .item()
            ),
        )
        for count in lane_segment_counts
    )
    if sum(segment_counts) > maximum_waypoints:
        lane_hold.deactivate(
            planning_active & lane_hold.active_mask,
            reason=f"{reason_prefix}-waypoint-cap",
        )

    active_before_start_solve = planning_active & lane_hold.active_mask
    start_solution = ik.solve(
        safe_current_tcp[:, :3],
        safe_current_tcp[:, 3:7],
        finger_target,
        arm_seed=safe_current_raw,
    )
    start_raw_step = torch.nan_to_num(
        torch.abs(start_solution.arm_q - safe_current_raw).amax(dim=-1),
        nan=torch.inf,
        posinf=torch.inf,
        neginf=torch.inf,
    )
    start_solution_finite = torch.isfinite(start_solution.arm_q).all(dim=-1)
    start_valid = start_solution.valid & start_solution_finite & (start_raw_step <= maximum_raw_step)
    maximum_observed_raw_step.copy_(torch.where(active_before_start_solve, start_raw_step, maximum_observed_raw_step))
    ik_valid &= ~active_before_start_solve | start_valid
    lane_hold.deactivate(
        active_before_start_solve & ~start_valid,
        reason=f"{reason_prefix}-preflight",
    )
    planned_raw = torch.where(
        (active_before_start_solve & start_valid)[:, None],
        start_solution.arm_q,
        safe_current_raw,
    )
    raw_segments: list[torch.Tensor] = []
    active_at_route_start = planning_active & lane_hold.active_mask
    for segment_start, segment_target, waypoint_count in zip(
        segment_starts,
        safe_tcp_targets_tuple,
        segment_counts,
        strict=True,
    ):
        segment_raw = [planned_raw.clone()]
        for waypoint in range(1, waypoint_count + 1):
            progress = waypoint / waypoint_count
            waypoint_position = torch.lerp(segment_start[:, :3], segment_target[:, :3], progress)
            waypoint_orientation = batched_quat_slerp(
                segment_start[:, 3:7],
                segment_target[:, 3:7],
                progress,
            )
            solution = ik.solve(
                waypoint_position,
                waypoint_orientation,
                finger_target,
                arm_seed=planned_raw,
            )
            active_before = planning_active & lane_hold.active_mask
            raw_step = torch.nan_to_num(
                torch.abs(solution.arm_q - planned_raw).amax(dim=-1),
                nan=torch.inf,
                posinf=torch.inf,
                neginf=torch.inf,
            )
            maximum_observed_raw_step.copy_(
                torch.where(
                    active_before,
                    torch.maximum(maximum_observed_raw_step, raw_step),
                    maximum_observed_raw_step,
                )
            )
            waypoint_valid = (
                solution.valid & torch.isfinite(solution.arm_q).all(dim=-1) & (raw_step <= maximum_raw_step)
            )
            ik_valid &= ~active_before | waypoint_valid
            lane_hold.deactivate(
                active_before & ~waypoint_valid,
                reason=f"{reason_prefix}-ik-continuity",
            )
            command_mask = active_before & waypoint_valid
            planned_raw = torch.where(command_mask[:, None], solution.arm_q, planned_raw)
            segment_raw.append(planned_raw.clone())
        raw_segments.append(torch.stack(segment_raw))

    terminal_raw = planned_raw.clone()
    safe_current_target = torch.where(
        torch.isfinite(current_target).all(dim=-1, keepdim=True),
        current_target,
        lane_hold.last_sent_arm_target,
    )
    if endpoint_arm_target is None:
        safe_endpoint_target = None
    else:
        endpoint_finite = torch.isfinite(endpoint_arm_target).all(dim=-1)
        lane_hold.deactivate(
            planning_active & lane_hold.active_mask & ~endpoint_finite,
            reason=f"{reason_prefix}-joint-limits",
        )
        safe_endpoint_target = torch.where(
            endpoint_finite[:, None],
            endpoint_arm_target,
            safe_current_target,
        )

    schedule_active = planning_active & lane_hold.active_mask
    schedule = _recovery_bias_blended_command_schedule(
        tuple(raw_segments),
        safe_current_target,
        safe_endpoint_target,
        schedule_active,
        maximum_command_step_rad=densification_step_limit,
        maximum_waypoint_count=maximum_waypoints,
    )
    terminal_target = schedule.segment_knots[-1][-1].clone()
    dense_cap_valid = schedule.post_densification_waypoint_count <= maximum_waypoints
    target_valid &= ~schedule_active | dense_cap_valid
    lane_hold.deactivate(
        schedule_active & ~dense_cap_valid,
        reason=f"{reason_prefix}-command-waypoint-cap",
    )
    stacked_targets = torch.cat(schedule.segment_knots, dim=0)
    start_anchor_valid = schedule.start_target_anchor_error == 0.0
    endpoint_anchor_valid = (
        torch.ones_like(start_anchor_valid)
        if safe_endpoint_target is None
        else schedule.canonical_endpoint_anchor_error == 0.0
    )
    command_continuity_valid = (
        (schedule.maximum_step_after_densification <= maximum_commanded_step)
        & start_anchor_valid
        & endpoint_anchor_valid
    )
    limits_valid = joint_limit_mask(env, stacked_targets, margin=margin).all(dim=0)
    target_valid &= ~planning_active | (command_continuity_valid & limits_valid)
    lane_hold.deactivate(
        active_at_route_start & ~command_continuity_valid,
        reason=f"{reason_prefix}-command-continuity",
    )
    lane_hold.deactivate(
        active_at_route_start & ~limits_valid,
        reason=f"{reason_prefix}-joint-limits",
    )
    return _RecoveryCartesianPlan(
        segment_knots=schedule.segment_knots,
        segment_waypoint_counts=schedule.segment_waypoint_counts,
        pre_densification_waypoint_count=schedule.pre_densification_waypoint_count,
        post_densification_waypoint_count=schedule.post_densification_waypoint_count,
        terminal_raw=terminal_raw,
        terminal_target=terminal_target,
        start_preload_bias=schedule.start_preload_bias,
        goal_preload_bias=schedule.goal_preload_bias,
        preload_bias_difference=schedule.preload_bias_difference,
        maximum_raw_ik_joint_step=maximum_observed_raw_step,
        maximum_commanded_joint_step_before_densification=schedule.maximum_step_before_densification,
        maximum_commanded_joint_step_after_densification=schedule.maximum_step_after_densification,
        command_densification_required_subknot_count=schedule.required_subknot_count,
        command_densification_executed_subknot_count=schedule.executed_subknot_count,
        start_target_anchor_error=schedule.start_target_anchor_error,
        canonical_endpoint_anchor_error=schedule.canonical_endpoint_anchor_error,
        maximum_segment_boundary_command_jump=schedule.maximum_segment_boundary_jump,
        ik_valid=ik_valid,
        target_valid=target_valid,
    )


def _initialize_recovery_cartesian_motion_evidence(env: _ResetToolEnv) -> dict[str, Any]:
    """Create the per-step physical evidence accumulator for Cartesian recovery."""
    true_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    zero = torch.zeros(env.num_envs, device=env.device)
    minimum_contacts = torch.full(
        (env.num_envs,),
        torch.iinfo(torch.int64).max,
        device=env.device,
        dtype=torch.long,
    )
    evidence: dict[str, Any] = {
        "samples": 0,
        "_current_segment": -1,
        "_current_knot": -1,
        "_elapsed_time_s": 0.0,
        "_next_segment_index": 0,
        "all_samples_finite": true_mask.clone(),
        "all_samples_collision_free": true_mask.clone(),
        "all_samples_bilateral_proxy_contact": true_mask.clone(),
        "all_samples_grasp_valid": true_mask.clone(),
        "all_samples_drives_disabled": true_mask.clone(),
        "all_samples_transport_speeds_bounded": true_mask.clone(),
        "all_samples_cable_speed_within_reset_limit": true_mask.clone(),
        "maximum_cable_linear_speed": zero.clone(),
        "maximum_plug_linear_speed": zero.clone(),
        "maximum_plug_angular_speed": zero.clone(),
        "maximum_arm_joint_speed": zero.clone(),
        "maximum_finger_joint_speed": zero.clone(),
        "maximum_invalid_contact_count": torch.zeros_like(minimum_contacts),
        "minimum_left_grasp_contact_count": minimum_contacts.clone(),
        "minimum_right_grasp_contact_count": minimum_contacts.clone(),
        "sampled_invalid_contact_pairs": [],
    }
    for component in (
        "plug_linear_speed",
        "plug_angular_speed",
        "arm_joint_speed",
        "finger_joint_speed",
    ):
        evidence[f"all_samples_{component}_bounded"] = true_mask.clone()
        evidence[f"first_{component}_failure_mask"] = torch.zeros_like(true_mask)
        evidence[f"first_{component}_failure_step"] = torch.full_like(minimum_contacts, -1)
        evidence[f"first_{component}_failure_segment"] = torch.full_like(minimum_contacts, -1)
        evidence[f"first_{component}_failure_knot"] = torch.full_like(minimum_contacts, -1)
        evidence[f"first_{component}_failure_time_s"] = torch.full_like(zero, -1.0)
    return evidence


def _sample_recovery_cartesian_motion(
    env: _ResetToolEnv,
    finger_target: torch.Tensor,
    lane_hold: _PerLaneTargetHold,
    evidence: dict[str, Any],
) -> None:
    """Apply every immutable post-step recovery gate and freeze failed lanes."""
    active_at_sample_start = lane_hold.active_mask
    task_q, task_qd = env.read_task_state()
    arm_q, arm_qd, finger_q, finger_qd = env.read_robot_state()
    collision, grasp, bilateral = _physical_validity_sample(env, finger_target)
    if collision.contact_overflow:
        raise RuntimeError("Global contact-buffer overflow during Cartesian scripted recovery.")
    drives_disabled = _runtime_drives_disabled(env)
    if not bool(drives_disabled.all()):
        raise RuntimeError("A construction drive became enabled during Cartesian scripted recovery.")
    layout = env.rj45_runtime.layout
    plug_index = int(layout.plug_body_index)
    cable_slice = layout.cable_body_slice
    cable_speed = _finite_maximum(
        torch.linalg.vector_norm(task_qd[:, cable_slice, :3], dim=-1),
        dimensions=(1,),
    )
    plug_linear_speed = torch.nan_to_num(
        torch.linalg.vector_norm(task_qd[:, plug_index, :3], dim=-1),
        nan=torch.inf,
        posinf=torch.inf,
        neginf=torch.inf,
    )
    plug_angular_speed = torch.nan_to_num(
        torch.linalg.vector_norm(task_qd[:, plug_index, 3:6], dim=-1),
        nan=torch.inf,
        posinf=torch.inf,
        neginf=torch.inf,
    )
    arm_speed = _finite_maximum(torch.abs(arm_qd), dimensions=(1,))
    finger_speed = _finite_maximum(torch.abs(finger_qd), dimensions=(1,))
    finite = task_state_is_finite_and_normalized(task_q, task_qd) & _per_world_finite(
        arm_q, arm_qd, finger_q, finger_qd
    )
    gates = PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["per_step_rejection_gates"]
    component_speed_valid = {
        "plug_linear_speed": plug_linear_speed <= float(gates["plug_linear_speed_m_s"]),
        "plug_angular_speed": plug_angular_speed <= float(gates["plug_angular_speed_rad_s"]),
        "arm_joint_speed": arm_speed <= float(gates["arm_joint_speed_rad_s"]),
        "finger_joint_speed": finger_speed <= float(gates["finger_joint_speed_m_s"]),
    }
    transport_speed_valid = torch.stack(tuple(component_speed_valid.values())).all(dim=0)
    cable_limit = float(PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["transient_cable_speed"]["reset_limit_m_s"])
    cable_within_limit = cable_speed <= cable_limit
    evidence["all_samples_finite"] &= finite
    evidence["all_samples_collision_free"] &= collision.valid
    evidence["all_samples_bilateral_proxy_contact"] &= bilateral
    evidence["all_samples_grasp_valid"] &= grasp.valid
    evidence["all_samples_drives_disabled"] &= drives_disabled
    evidence["all_samples_transport_speeds_bounded"] &= transport_speed_valid
    evidence["all_samples_cable_speed_within_reset_limit"] &= cable_within_limit
    for name, value in (
        ("maximum_cable_linear_speed", cable_speed),
        ("maximum_plug_linear_speed", plug_linear_speed),
        ("maximum_plug_angular_speed", plug_angular_speed),
        ("maximum_arm_joint_speed", arm_speed),
        ("maximum_finger_joint_speed", finger_speed),
    ):
        evidence[name] = torch.maximum(evidence[name], value)
    evidence["maximum_invalid_contact_count"] = torch.maximum(
        evidence["maximum_invalid_contact_count"], collision.invalid_contact_count
    )
    evidence["minimum_left_grasp_contact_count"] = torch.minimum(
        evidence["minimum_left_grasp_contact_count"], collision.left_grasp_contact_count
    )
    evidence["minimum_right_grasp_contact_count"] = torch.minimum(
        evidence["minimum_right_grasp_contact_count"], collision.right_grasp_contact_count
    )
    evidence["samples"] += 1
    evidence["_elapsed_time_s"] += float(env.advance_dt)
    for component, valid in component_speed_valid.items():
        evidence[f"all_samples_{component}_bounded"] &= valid
        first_mask = evidence[f"first_{component}_failure_mask"]
        new_failure = active_at_sample_start & ~valid & ~first_mask
        first_mask.logical_or_(new_failure)
        evidence[f"first_{component}_failure_step"].copy_(
            torch.where(
                new_failure,
                torch.full_like(evidence[f"first_{component}_failure_step"], evidence["samples"]),
                evidence[f"first_{component}_failure_step"],
            )
        )
        evidence[f"first_{component}_failure_segment"].copy_(
            torch.where(
                new_failure,
                torch.full_like(
                    evidence[f"first_{component}_failure_segment"],
                    int(evidence["_current_segment"]),
                ),
                evidence[f"first_{component}_failure_segment"],
            )
        )
        evidence[f"first_{component}_failure_knot"].copy_(
            torch.where(
                new_failure,
                torch.full_like(
                    evidence[f"first_{component}_failure_knot"],
                    int(evidence["_current_knot"]),
                ),
                evidence[f"first_{component}_failure_knot"],
            )
        )
        evidence[f"first_{component}_failure_time_s"].copy_(
            torch.where(
                new_failure,
                torch.full_like(
                    evidence[f"first_{component}_failure_time_s"],
                    float(evidence["_elapsed_time_s"]),
                ),
                evidence[f"first_{component}_failure_time_s"],
            )
        )
    for pair in collision.invalid_contact_pairs:
        if (
            pair not in evidence["sampled_invalid_contact_pairs"]
            and len(evidence["sampled_invalid_contact_pairs"]) < 64
        ):
            evidence["sampled_invalid_contact_pairs"].append(pair)
    lane_hold.deactivate(active_at_sample_start & ~finite, reason="recovery-cartesian-motion-non-finite")
    lane_hold.deactivate(active_at_sample_start & ~collision.valid, reason="recovery-cartesian-motion-collision")
    lane_hold.deactivate(
        active_at_sample_start & ~bilateral,
        reason="recovery-cartesian-motion-lost-bilateral-contact",
    )
    lane_hold.deactivate(active_at_sample_start & ~grasp.valid, reason="recovery-cartesian-motion-grasp-geometry")
    for component, valid in component_speed_valid.items():
        lane_hold.deactivate(
            active_at_sample_start & ~valid,
            reason=f"recovery-cartesian-motion-{component.replace('_', '-')}",
        )


def _execute_recovery_cartesian_plan(
    env: _ResetToolEnv,
    plan: _RecoveryCartesianPlan,
    finger_target: torch.Tensor,
    lane_hold: _PerLaneTargetHold,
    evidence: dict[str, Any],
    *,
    requested_duration_s: float,
) -> torch.Tensor:
    """Execute one precomputed route with the immutable C2 schedule and no knot dwells."""
    if not math.isfinite(requested_duration_s) or requested_duration_s <= 0.0:
        raise ValueError("Recovery Cartesian motion duration must be finite and positive.")
    execution = PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["execution"]
    duration_per_knot = max(
        float(execution["minimum_duration_per_knot_s"]),
        requested_duration_s / max(plan.waypoint_count, 1),
    )
    segment_settle_s = float(execution["segment_end_settle_s"])
    segment_base = int(evidence.get("_next_segment_index", 0))
    executed_segment_count = 0
    for local_segment_index, (segment_knots, waypoint_count) in enumerate(
        zip(
            plan.segment_knots,
            plan.segment_waypoint_counts,
            strict=True,
        )
    ):
        if not bool(lane_hold.active_mask.any()):
            break
        evidence["_current_segment"] = segment_base + local_segment_index

        def update(_step: int, _steps: int, progress: float) -> None:
            path_progress = _recovery_cartesian_c2_progress(progress)
            evidence["_current_knot"] = min(
                int(math.floor(path_progress * waypoint_count)),
                waypoint_count,
            )
            env.set_robot_targets(
                _interpolate_recovery_cartesian_knots(segment_knots, path_progress),
                finger_target,
            )

        env.advance(
            waypoint_count * duration_per_knot,
            update,
            post_step=lambda _step, _steps, _progress: _sample_recovery_cartesian_motion(
                env, finger_target, lane_hold, evidence
            ),
        )
        evidence["_current_knot"] = waypoint_count
        env.set_robot_targets(segment_knots[-1], finger_target)
        env.advance(
            segment_settle_s,
            post_step=lambda _step, _steps, _progress: _sample_recovery_cartesian_motion(
                env, finger_target, lane_hold, evidence
            ),
        )
        executed_segment_count += 1
    evidence["_next_segment_index"] = segment_base + executed_segment_count
    evidence["_current_segment"] = -1
    evidence["_current_knot"] = -1
    return lane_hold.last_sent_arm_target


def _scripted_recovery_cartesian_c2(  # noqa: C901
    env: _ResetToolEnv,
    ik: FrankaResetIK,
    goal_task_q: torch.Tensor,
    orientation: torch.Tensor | None,
    finger_target: torch.Tensor,
    *,
    arm_target_start: torch.Tensor | None,
    goal_arm_target: torch.Tensor | None,
    motion_s: float,
    settle_s: float,
    compensation_max_iterations: int,
    compensation_gain: float,
    compensation_max_step_m: float,
    compensation_motion_s: float,
    compensation_hold_s: float,
    compensation_tolerance_m: float,
    plug_body_index: int | None,
    latch_body_index: int | None,
    arm_target_is_absolute: bool,
    lane_hold: _PerLaneTargetHold,
    pick_insert_phase: int | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run the immutable pick-only incremental Cartesian scripted recovery."""
    if isinstance(pick_insert_phase, bool) or not isinstance(pick_insert_phase, int):
        raise ValueError("pick_insert_phase must be an integer from 0 through 5 for Cartesian recovery.")
    if str(pick_insert_phase) not in PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["phase_route_modes"]:
        raise ValueError(f"pick_insert_phase must be one of 0 through 5, got {pick_insert_phase!r}.")
    expected_initial_route_endpoint_policy = "canonical-goal-c2-stop-before-bounded-compensation"
    if (
        PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY.get("initial_route_endpoint_policy")
        != expected_initial_route_endpoint_policy
    ):
        raise RuntimeError(
            "Unsupported Cartesian recovery initial-route endpoint policy; expected "
            f"{expected_initial_route_endpoint_policy!r}."
        )
    expected_clearance_waypoints = (
        "vertical-lift",
        "high-midpoint",
        "overhead-preinsert",
        "preinsert",
        "canonical-goal",
    )
    clearance_waypoints = tuple(
        PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["clearance_route"].get("plug_pose_waypoints", ())
    )
    if clearance_waypoints != expected_clearance_waypoints:
        raise RuntimeError(
            "Unsupported Cartesian recovery clearance route; expected the canonical goal to be its first "
            "seated endpoint."
        )
    compensation_contract = PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["compensation"]
    if (
        compensation_contract.get("proactive_overtravel") is not False
        or compensation_contract.get("trigger") != "settled-goal-error-above-tolerance"
    ):
        raise RuntimeError(
            "Cartesian recovery compensation must be reactive to settled goal error with proactive overtravel disabled."
        )
    if goal_arm_target is None or not arm_target_is_absolute:
        raise ValueError(
            "Pick-insert Cartesian recovery requires a canonical goal_arm_target with absolute-target semantics."
        )
    if compensation_max_iterations < 0:
        raise ValueError("compensation_max_iterations must be non-negative.")
    for name, value in (
        ("motion_s", motion_s),
        ("settle_s", settle_s),
        ("compensation_gain", compensation_gain),
        ("compensation_max_step_m", compensation_max_step_m),
        ("compensation_motion_s", compensation_motion_s),
        ("compensation_hold_s", compensation_hold_s),
        ("compensation_tolerance_m", compensation_tolerance_m),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    env.set_drive(False)
    if not bool(_runtime_drives_disabled(env).all()):
        raise RuntimeError("Pick-insert Cartesian recovery requires both construction drives to be disabled.")
    arm_start, _, _, _ = env.read_robot_state()
    if arm_target_start is None:
        arm_target_start = arm_start
    else:
        arm_target_start = torch.as_tensor(arm_target_start, device=env.device, dtype=torch.float32)
        if tuple(arm_target_start.shape) != (env.num_envs, 7):
            raise ValueError(f"arm_target_start must have shape ({env.num_envs}, 7).")
    goal_arm_target = torch.as_tensor(goal_arm_target, device=env.device, dtype=torch.float32)
    if tuple(goal_arm_target.shape) == (7,):
        goal_arm_target = goal_arm_target.expand(env.num_envs, -1)
    if tuple(goal_arm_target.shape) != (env.num_envs, 7):
        raise ValueError(f"goal_arm_target must have shape (7,) or ({env.num_envs}, 7).")
    finger_target = torch.as_tensor(finger_target, device=env.device, dtype=torch.float32)
    if tuple(finger_target.shape) != (env.num_envs, 2):
        raise ValueError(f"finger_target must have shape ({env.num_envs}, 2).")
    reset_target_bias = arm_target_start - arm_start
    plug_body_index, latch_body_index = _resolve_layout_body_indices(env, plug_body_index, latch_body_index)
    current_task_q, _ = env.read_task_state()
    goal_task_q_batched = _batched_goal_task_pose(goal_task_q, current_task_q)
    live_plug = current_task_q[:, plug_body_index].clone()
    goal_plug = goal_task_q_batched[:, plug_body_index].clone()
    grasp_offset = torch.as_tensor(
        env.cfg.plug_grasp_offset,
        device=env.device,
        dtype=torch.float32,
    ).expand(env.num_envs, -1)
    configured_grasp_orientation = getattr(env.cfg, "plug_grasp_orientation_xyzw", None)
    if configured_grasp_orientation is None:
        if orientation is None:
            raise ValueError("orientation is required when the task does not configure plug_grasp_orientation_xyzw.")
        desired_orientation = torch.as_tensor(orientation, device=env.device, dtype=torch.float32)
        if tuple(desired_orientation.shape) == (4,):
            desired_orientation = desired_orientation.expand(env.num_envs, -1)
        if tuple(desired_orientation.shape) != (env.num_envs, 4):
            raise ValueError(f"orientation must have shape (4,) or ({env.num_envs}, 4).")
        grasp_orientation = math_utils.quat_unique(
            math_utils.quat_mul(math_utils.quat_conjugate(goal_plug[:, 3:7]), desired_orientation)
        )
    else:
        grasp_orientation = torch.as_tensor(
            configured_grasp_orientation,
            device=env.device,
            dtype=torch.float32,
        ).expand(env.num_envs, -1)
        desired_orientation = math_utils.quat_unique(math_utils.quat_mul(goal_plug[:, 3:7], grasp_orientation))
    goal_position = goal_plug[:, :3] + math_utils.quat_apply(goal_plug[:, 3:7], grasp_offset)
    overtravel_distance = torch.zeros(env.num_envs, device=env.device, dtype=goal_position.dtype)
    plug_route, route_preflight_valid = _pick_insert_recovery_plug_route(
        live_plug,
        goal_plug,
        phase=pick_insert_phase,
    )
    lane_hold.deactivate(~route_preflight_valid, reason="recovery-cartesian-route-geometry-preflight")
    tcp_route = _plug_route_to_tcp_targets(
        plug_route,
        grasp_offset=grasp_offset,
        grasp_orientation=grasp_orientation,
    )
    current_tcp = env.tcp_pose_e().clone()
    initial_plan = _plan_recovery_cartesian_route(
        env,
        ik,
        tcp_route,
        finger_target,
        current_tcp=current_tcp,
        current_raw=arm_start,
        current_target=lane_hold.last_sent_arm_target,
        lane_hold=lane_hold,
        reason_prefix="recovery-cartesian-route",
        endpoint_arm_target=goal_arm_target,
    )
    ik_valid = route_preflight_valid & initial_plan.ik_valid
    target_valid = initial_plan.target_valid
    maximum_raw_ik_joint_step = initial_plan.maximum_raw_ik_joint_step.clone()
    maximum_commanded_joint_step_before_densification = (
        initial_plan.maximum_commanded_joint_step_before_densification.clone()
    )
    maximum_commanded_joint_step_after_densification = (
        initial_plan.maximum_commanded_joint_step_after_densification.clone()
    )
    command_densification_required_subknot_count = initial_plan.command_densification_required_subknot_count.clone()
    command_densification_executed_subknot_count = initial_plan.command_densification_executed_subknot_count.clone()
    start_target_anchor_error = initial_plan.start_target_anchor_error.clone()
    canonical_endpoint_anchor_error = initial_plan.canonical_endpoint_anchor_error.clone()
    maximum_segment_boundary_command_jump = initial_plan.maximum_segment_boundary_command_jump.clone()
    maximum_preload_bias_difference = initial_plan.preload_bias_difference.abs().amax(dim=-1)
    total_cartesian_waypoints_before_densification = initial_plan.pre_densification_waypoint_count.clone()
    total_cartesian_waypoints_after_densification = initial_plan.post_densification_waypoint_count.clone()
    motion_evidence = _initialize_recovery_cartesian_motion_evidence(env)
    current_target = _execute_recovery_cartesian_plan(
        env,
        initial_plan,
        finger_target,
        lane_hold,
        motion_evidence,
        requested_duration_s=motion_s,
    )
    current_raw = initial_plan.terminal_raw
    env.set_robot_targets(initial_plan.terminal_target, finger_target)
    env.advance(
        settle_s,
        post_step=lambda _step, _steps, _progress: _sample_recovery_cartesian_motion(
            env, finger_target, lane_hold, motion_evidence
        ),
    )
    current_target = lane_hold.last_sent_arm_target

    commanded_position = goal_position.clone()
    goal_error_history: list[torch.Tensor] = []
    plug_translation_error_history: list[torch.Tensor] = []
    correction_norm_history: list[torch.Tensor] = []
    ik_joint_step_history: list[torch.Tensor] = []
    compensation_iterations = 0
    for iteration in range(compensation_max_iterations + 1):
        current_task_q, _ = env.read_task_state()
        current_error = scalar_goal_error(
            current_task_q,
            goal_task_q,
            plug_body_index=plug_body_index,
            latch_body_index=latch_body_index,
        )
        plug_translation_error = goal_plug[:, :3] - current_task_q[:, plug_body_index, :3]
        goal_error_history.append(current_error.clone())
        plug_translation_error_history.append(plug_translation_error.clone())
        active = lane_hold.active_mask & (current_error > compensation_tolerance_m)
        if not bool(active.any()) or iteration == compensation_max_iterations:
            break
        correction = compensation_gain * plug_translation_error
        correction_norm = torch.linalg.vector_norm(correction, dim=-1, keepdim=True)
        correction *= torch.clamp(compensation_max_step_m / correction_norm.clamp_min(1.0e-9), max=1.0)
        correction *= active[:, None]
        correction_norm_history.append(torch.linalg.vector_norm(correction, dim=-1))
        next_commanded_position = commanded_position + correction
        compensation_tcp = torch.cat((next_commanded_position, desired_orientation), dim=-1)
        compensation_plan = _plan_recovery_cartesian_route(
            env,
            ik,
            (compensation_tcp,),
            finger_target,
            current_tcp=env.tcp_pose_e().clone(),
            current_raw=current_raw,
            current_target=current_target,
            lane_hold=lane_hold,
            reason_prefix="recovery-cartesian-compensation",
            endpoint_arm_target=None,
            active_mask=active,
        )
        ik_valid &= ~active | compensation_plan.ik_valid
        target_valid &= ~active | compensation_plan.target_valid
        maximum_raw_ik_joint_step = torch.maximum(
            maximum_raw_ik_joint_step,
            compensation_plan.maximum_raw_ik_joint_step,
        )
        maximum_commanded_joint_step_before_densification = torch.maximum(
            maximum_commanded_joint_step_before_densification,
            compensation_plan.maximum_commanded_joint_step_before_densification,
        )
        maximum_commanded_joint_step_after_densification = torch.maximum(
            maximum_commanded_joint_step_after_densification,
            compensation_plan.maximum_commanded_joint_step_after_densification,
        )
        command_densification_required_subknot_count += compensation_plan.command_densification_required_subknot_count
        command_densification_executed_subknot_count += compensation_plan.command_densification_executed_subknot_count
        start_target_anchor_error = torch.maximum(
            start_target_anchor_error,
            compensation_plan.start_target_anchor_error,
        )
        canonical_endpoint_anchor_error = torch.maximum(
            canonical_endpoint_anchor_error,
            compensation_plan.canonical_endpoint_anchor_error,
        )
        maximum_segment_boundary_command_jump = torch.maximum(
            maximum_segment_boundary_command_jump,
            compensation_plan.maximum_segment_boundary_command_jump,
        )
        maximum_preload_bias_difference = torch.maximum(
            maximum_preload_bias_difference,
            compensation_plan.preload_bias_difference.abs().amax(dim=-1),
        )
        ik_joint_step_history.append(compensation_plan.maximum_raw_ik_joint_step)
        total_cartesian_waypoints_before_densification += compensation_plan.pre_densification_waypoint_count
        total_cartesian_waypoints_after_densification += compensation_plan.post_densification_waypoint_count
        current_target = _execute_recovery_cartesian_plan(
            env,
            compensation_plan,
            finger_target,
            lane_hold,
            motion_evidence,
            requested_duration_s=compensation_motion_s,
        )
        current_raw = compensation_plan.terminal_raw
        env.set_robot_targets(compensation_plan.terminal_target, finger_target)
        env.advance(
            compensation_hold_s,
            post_step=lambda _step, _steps, _progress: _sample_recovery_cartesian_motion(
                env, finger_target, lane_hold, motion_evidence
            ),
        )
        command_update = active & lane_hold.active_mask
        commanded_position = torch.where(
            command_update[:, None],
            next_commanded_position,
            commanded_position,
        )
        current_target = lane_hold.last_sent_arm_target
        compensation_iterations = iteration + 1

    if compensation_iterations:
        goal_tcp = torch.cat((goal_position, desired_orientation), dim=-1)
        return_plan = _plan_recovery_cartesian_route(
            env,
            ik,
            (goal_tcp,),
            finger_target,
            current_tcp=env.tcp_pose_e().clone(),
            current_raw=current_raw,
            current_target=current_target,
            lane_hold=lane_hold,
            reason_prefix="recovery-cartesian-return",
            endpoint_arm_target=goal_arm_target,
        )
        ik_valid &= return_plan.ik_valid
        target_valid &= return_plan.target_valid
        maximum_raw_ik_joint_step = torch.maximum(
            maximum_raw_ik_joint_step,
            return_plan.maximum_raw_ik_joint_step,
        )
        maximum_commanded_joint_step_before_densification = torch.maximum(
            maximum_commanded_joint_step_before_densification,
            return_plan.maximum_commanded_joint_step_before_densification,
        )
        maximum_commanded_joint_step_after_densification = torch.maximum(
            maximum_commanded_joint_step_after_densification,
            return_plan.maximum_commanded_joint_step_after_densification,
        )
        command_densification_required_subknot_count += return_plan.command_densification_required_subknot_count
        command_densification_executed_subknot_count += return_plan.command_densification_executed_subknot_count
        start_target_anchor_error = torch.maximum(start_target_anchor_error, return_plan.start_target_anchor_error)
        canonical_endpoint_anchor_error = torch.maximum(
            canonical_endpoint_anchor_error,
            return_plan.canonical_endpoint_anchor_error,
        )
        maximum_segment_boundary_command_jump = torch.maximum(
            maximum_segment_boundary_command_jump,
            return_plan.maximum_segment_boundary_command_jump,
        )
        maximum_preload_bias_difference = torch.maximum(
            maximum_preload_bias_difference,
            return_plan.preload_bias_difference.abs().amax(dim=-1),
        )
        total_cartesian_waypoints_before_densification += return_plan.pre_densification_waypoint_count
        total_cartesian_waypoints_after_densification += return_plan.post_densification_waypoint_count
        current_target = _execute_recovery_cartesian_plan(
            env,
            return_plan,
            finger_target,
            lane_hold,
            motion_evidence,
            requested_duration_s=compensation_motion_s,
        )
        current_raw = return_plan.terminal_raw
        del current_raw
        env.set_robot_targets(return_plan.terminal_target, finger_target)
        env.advance(
            settle_s,
            post_step=lambda _step, _steps, _progress: _sample_recovery_cartesian_motion(
                env, finger_target, lane_hold, motion_evidence
            ),
        )
        current_target = lane_hold.last_sent_arm_target

    speed_gates = PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["per_step_rejection_gates"]

    def dwell_speed_gate(snapshot: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
        task_qd = snapshot["task_qd"]
        plug_linear = torch.linalg.vector_norm(task_qd[:, plug_body_index, :3], dim=-1)
        plug_angular = torch.linalg.vector_norm(task_qd[:, plug_body_index, 3:6], dim=-1)
        return {
            "plug-linear-speed": plug_linear > float(speed_gates["plug_linear_speed_m_s"]),
            "plug-angular-speed": plug_angular > float(speed_gates["plug_angular_speed_rad_s"]),
            "arm-joint-speed": snapshot["arm_joint_speed"] > float(speed_gates["arm_joint_speed_rad_s"]),
            "finger-joint-speed": snapshot["finger_joint_speed"] > float(speed_gates["finger_joint_speed_m_s"]),
        }

    dwell_arm_q, _, _, _ = env.read_robot_state()
    dwell_arm_command = current_target if arm_target_is_absolute else current_target - dwell_arm_q
    exact_success, exact_metrics = advance_exact_success_dwell(
        env,
        goal_task_q,
        dwell_arm_command,
        finger_target,
        require_all_samples=True,
        sample_physical_validity=True,
        arm_target_is_absolute=arm_target_is_absolute,
        plug_body_index=plug_body_index,
        latch_body_index=latch_body_index,
        lane_hold=lane_hold,
        per_step_lane_goal_gate=dwell_speed_gate,
    )
    for component in (
        "plug_linear_speed",
        "plug_angular_speed",
        "arm_joint_speed",
        "finger_joint_speed",
    ):
        dwell_reason = component.replace("_", "-")
        dwell_failure = exact_metrics["lane_goal_gate_violation_masks"][dwell_reason]
        first_mask = motion_evidence[f"first_{component}_failure_mask"]
        new_failure = dwell_failure & ~first_mask
        first_mask.logical_or_(dwell_failure)
        dwell_step = exact_metrics["lane_goal_gate_first_failure_steps"][dwell_reason]
        motion_evidence[f"first_{component}_failure_step"].copy_(
            torch.where(
                new_failure,
                dwell_step + int(motion_evidence["samples"]),
                motion_evidence[f"first_{component}_failure_step"],
            )
        )
        motion_evidence[f"first_{component}_failure_time_s"].copy_(
            torch.where(
                new_failure,
                float(motion_evidence["_elapsed_time_s"]) + dwell_step * float(env.advance_dt),
                motion_evidence[f"first_{component}_failure_time_s"],
            )
        )
    task_q, task_qd = env.read_task_state()
    final_arm_q, _, _, _ = env.read_robot_state()
    error = scalar_goal_error(
        task_q,
        goal_task_q,
        plug_body_index=plug_body_index,
        latch_body_index=latch_body_index,
    )
    plug_speed = torch.linalg.vector_norm(task_qd[:, plug_body_index, :3], dim=-1)
    grasp = grasp_metrics(env, finger_target)
    collision = collision_metrics(env)
    if collision.contact_overflow:
        raise RuntimeError("Global contact-buffer overflow at the scripted-recovery result boundary.")
    success = (
        ik_valid
        & target_valid
        & (error <= 0.002)
        & (plug_speed <= 0.03)
        & exact_success
        & exact_metrics["all_samples_collision_free"]
        & exact_metrics["all_samples_bilateral_grasp"]
        & exact_metrics["all_samples_finite"]
        & grasp.valid
        & collision.valid
        & task_state_is_finite_and_normalized(task_q, task_qd)
        & lane_hold.active_mask
    )
    success &= exact_metrics["all_samples_arm_target_tracking_bounded"]
    success &= exact_metrics["maximum_arm_target_drift"] <= 1.0e-7
    lane_hold.deactivate(~ik_valid, reason="recovery-final-ik")
    lane_hold.deactivate(~grasp.valid, reason="recovery-final-grasp")
    lane_hold.deactivate(~collision.valid, reason="recovery-final-collision")
    lane_hold.deactivate(~success, reason="recovery-final-validation")
    success &= lane_hold.active_mask
    if motion_evidence["samples"] == 0:
        motion_evidence["minimum_left_grasp_contact_count"].zero_()
        motion_evidence["minimum_right_grasp_contact_count"].zero_()
    maximum_allowed_raw_step = float(
        PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["planning"]["maximum_raw_ik_joint_step_rad"]
    )
    return success, {
        "motion_policy": PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["motion_policy"],
        "pick_insert_phase": torch.full((env.num_envs,), pick_insert_phase, device=env.device, dtype=torch.long),
        "goal_error": error,
        "plug_speed": plug_speed,
        "tcp_grasp_distance": grasp.tcp_distance,
        "ik_valid": ik_valid,
        "target_valid": target_valid,
        "arm_target_bias_norm": torch.linalg.vector_norm(reset_target_bias, dim=-1),
        "arm_target_tracking_error": torch.linalg.vector_norm(exact_metrics["last_arm_target"] - final_arm_q, dim=-1),
        "overtravel_distance": overtravel_distance,
        "used_canonical_goal_arm_target": torch.ones(env.num_envs, device=env.device, dtype=torch.bool),
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
        "initial_ik_joint_step_max_rad": initial_plan.maximum_raw_ik_joint_step,
        "ik_joint_step_history": (
            torch.stack(ik_joint_step_history, dim=1)
            if ik_joint_step_history
            else torch.empty((env.num_envs, 0), device=env.device)
        ),
        "maximum_ik_joint_step_rad": torch.full(
            (env.num_envs,), maximum_allowed_raw_step, device=env.device, dtype=torch.float32
        ),
        "maximum_observed_raw_ik_joint_step_rad": maximum_raw_ik_joint_step,
        "start_preload_bias_by_joint_rad": initial_plan.start_preload_bias,
        "goal_preload_bias_by_joint_rad": initial_plan.goal_preload_bias,
        "preload_bias_difference_by_joint_rad": initial_plan.preload_bias_difference,
        "maximum_preload_bias_difference_rad": maximum_preload_bias_difference,
        "maximum_commanded_joint_step_before_densification_rad": (maximum_commanded_joint_step_before_densification),
        "maximum_commanded_joint_step_after_densification_rad": (maximum_commanded_joint_step_after_densification),
        "maximum_observed_commanded_joint_step_rad": maximum_commanded_joint_step_after_densification,
        "command_densification_required_subknot_count": command_densification_required_subknot_count,
        "command_densification_executed_subknot_count": command_densification_executed_subknot_count,
        "start_target_anchor_error_rad": start_target_anchor_error,
        "canonical_endpoint_anchor_error_rad": canonical_endpoint_anchor_error,
        "maximum_segment_boundary_command_jump_rad": maximum_segment_boundary_command_jump,
        "cartesian_route_waypoint_count_before_densification": (total_cartesian_waypoints_before_densification),
        "cartesian_route_waypoint_count_after_densification": total_cartesian_waypoints_after_densification,
        "cartesian_route_waypoint_count": total_cartesian_waypoints_after_densification,
        "cartesian_motion_sample_count": torch.full(
            (env.num_envs,), motion_evidence["samples"], device=env.device, dtype=torch.long
        ),
        **{
            f"cartesian_motion_{name}": value
            for name, value in motion_evidence.items()
            if name not in {"samples", "sampled_invalid_contact_pairs"} and not name.startswith("_")
        },
        "cartesian_motion_sampled_invalid_contact_pairs": tuple(motion_evidence["sampled_invalid_contact_pairs"]),
        "tcp_goal_position_error": torch.linalg.vector_norm(env.tcp_pose_e()[:, :3] - goal_position, dim=-1),
        "invalid_contact_count": collision.invalid_contact_count,
        "invalid_contact_pairs": collision.invalid_contact_pairs,
        "left_grasp_contact_count": collision.left_grasp_contact_count,
        "right_grasp_contact_count": collision.right_grasp_contact_count,
        "lane_failure_masks": lane_hold.reason_masks,
        **{f"exact_success_{name}": value for name, value in exact_metrics.items() if name != "last_arm_target"},
    }


def _scripted_recovery_legacy(
    env: _ResetToolEnv,
    ik: FrankaResetIK,
    goal_task_q: torch.Tensor,
    orientation: torch.Tensor | None,
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
    plug_body_index: int | None = None,
    latch_body_index: int | None = None,
    arm_target_is_absolute: bool = False,
    lane_hold: _PerLaneTargetHold | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Attempt insertion using only the closed-grasp Franka, never the task drive.

    ``arm_target_start - measured_arm_position`` is retained along the
    trajectory as an IK/actuator preload offset. The final dwell uses the
    task's live reset convention: fixed absolute targets for pick-insert and
    measured-state-relative bias for the legacy insertion task.
    A caller should supply a deterministic (zero-noise, single-seed) IK
    instance for replay-invariant evidence. IK deltas are applied around the
    stored canonical arm target, preserving its calibrated equilibrium. The
    task drive remains disabled throughout.  When ``lane_hold`` is supplied,
    every per-lane IK or physical failure is frozen before later recovery
    stages while global simulator/control-plane failures still propagate.
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
    plug_body_index, latch_body_index = _resolve_layout_body_indices(env, plug_body_index, latch_body_index)
    current_task_q, _ = env.read_task_state()
    goal_task_q_batched = _batched_goal_task_pose(goal_task_q, current_task_q)
    goal_plug = goal_task_q_batched[:, plug_body_index]
    grasp_offset = torch.as_tensor(env.cfg.plug_grasp_offset, device=env.device).expand(env.num_envs, -1)
    goal_position = goal_plug[:, :3] + math_utils.quat_apply(goal_plug[:, 3:7], grasp_offset)
    configured_grasp_orientation = getattr(env.cfg, "plug_grasp_orientation_xyzw", None)
    if configured_grasp_orientation is None:
        if orientation is None:
            raise ValueError("orientation is required when the task does not configure plug_grasp_orientation_xyzw.")
        desired_orientation = torch.as_tensor(orientation, device=env.device, dtype=torch.float32)
        if tuple(desired_orientation.shape) == (4,):
            desired_orientation = desired_orientation.expand(env.num_envs, -1)
        if tuple(desired_orientation.shape) != (env.num_envs, 4):
            raise ValueError(f"orientation must have shape (4,) or ({env.num_envs}, 4).")
    else:
        plug_grasp_orientation = torch.as_tensor(
            configured_grasp_orientation,
            device=env.device,
            dtype=torch.float32,
        ).expand(env.num_envs, -1)
        desired_orientation = math_utils.quat_unique(math_utils.quat_mul(goal_plug[:, 3:7], plug_grasp_orientation))
    plug_error_local = math_utils.quat_apply_inverse(
        goal_plug[:, 3:7],
        current_task_q[:, plug_body_index, :3] - goal_plug[:, :3],
    )
    axial_remaining = (-plug_error_local[:, 1]).clamp_min(0.0)
    overtravel_distance = (0.25 * axial_remaining).clamp(max=0.004)
    overtravel_local = torch.zeros_like(goal_position)
    overtravel_local[:, 1] = overtravel_distance
    overtravel = goal_position + math_utils.quat_apply(goal_plug[:, 3:7], overtravel_local)
    overtravel_ik = ik.solve(overtravel, desired_orientation, finger_target, arm_seed=arm_target_start)
    goal_ik = ik.solve(goal_position, desired_orientation, finger_target, arm_seed=overtravel_ik.arm_q)
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
    if lane_hold is not None:
        lane_hold.deactivate(~ik_valid, reason="recovery-initial-ik")
        recovery_active = lane_hold.active_mask
        trajectory_start = lane_hold.last_sent_arm_target
    else:
        recovery_active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        trajectory_start = arm_target_start
    initial_command_mask = recovery_active & ik_valid
    safe_overtravel_target = torch.where(
        initial_command_mask[:, None],
        overtravel_target,
        trajectory_start,
    )
    safe_goal_target = torch.where(initial_command_mask[:, None], goal_target, safe_overtravel_target)
    interpolate_arm_motion(env, trajectory_start, safe_overtravel_target, finger_target, 0.75 * motion_s)
    interpolate_arm_motion(env, safe_overtravel_target, safe_goal_target, finger_target, 0.25 * motion_s)
    env.set_robot_targets(safe_goal_target, finger_target)
    env.advance(settle_s)

    # Contact compliance and the closed-grasp equilibrium leave a few
    # millimetres of configuration-dependent residual after a pure IK move.
    # Correct that residual using only measured plug motion and real Franka
    # commands.  Updating the IK objective (rather than teleporting either
    # body) also makes this a useful recoverability oracle for reset rows.
    commanded_position = goal_position.clone()
    current_target = lane_hold.last_sent_arm_target if lane_hold is not None else safe_goal_target
    # Keep actuator equilibrium targets and the matching kinematic IK
    # continuation separate.  The former contains a configuration-dependent
    # reset bias and is not a valid seed for the latter.
    current_command_ik_q = torch.where(initial_command_mask[:, None], goal_ik.arm_q, arm_start)
    goal_error_history: list[torch.Tensor] = []
    plug_translation_error_history: list[torch.Tensor] = []
    correction_norm_history: list[torch.Tensor] = []
    ik_joint_step_history: list[torch.Tensor] = []
    compensation_iterations = 0
    for iteration in range(compensation_max_iterations + 1):
        current_task_q, _ = env.read_task_state()
        current_error = scalar_goal_error(
            current_task_q,
            goal_task_q,
            plug_body_index=plug_body_index,
            latch_body_index=latch_body_index,
        )
        plug_translation_error = goal_plug[:, :3] - current_task_q[:, plug_body_index, :3]
        goal_error_history.append(current_error.clone())
        plug_translation_error_history.append(plug_translation_error.clone())
        recovery_active = (
            lane_hold.active_mask
            if lane_hold is not None
            else torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        )
        if (
            not bool((recovery_active & (current_error > compensation_tolerance_m)).any())
            or iteration == compensation_max_iterations
        ):
            break

        active = recovery_active & (current_error > compensation_tolerance_m)
        correction = compensation_gain * plug_translation_error
        correction_norm = torch.linalg.vector_norm(correction, dim=-1, keepdim=True)
        correction *= torch.clamp(compensation_max_step_m / correction_norm.clamp_min(1.0e-9), max=1.0)
        correction *= active[:, None]
        correction_norm_history.append(torch.linalg.vector_norm(correction, dim=-1))
        next_commanded_position = commanded_position + correction
        compensated_ik = ik.solve(
            next_commanded_position,
            desired_orientation,
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
        if lane_hold is not None:
            lane_hold.deactivate(active & ~solution_valid, reason="recovery-compensation-ik")
        command_update = active & solution_valid
        next_target = torch.where(command_update[:, None], proposed_target, current_target)
        next_commanded_position = torch.where(command_update[:, None], next_commanded_position, commanded_position)
        interpolate_arm_motion(env, current_target, next_target, finger_target, compensation_motion_s)
        env.set_robot_targets(next_target, finger_target)
        env.advance(compensation_hold_s)
        current_target = lane_hold.last_sent_arm_target if lane_hold is not None else next_target
        current_command_ik_q = torch.where(command_update[:, None], compensated_ik.arm_q, current_command_ik_q)
        commanded_position = next_commanded_position
        compensation_iterations = iteration + 1

    # Return from the force-producing insertion push to the fixed seated
    # Franka equilibrium. This proves the latch maintains insertion without
    # relying on a permanently over-travelled arm command.
    if compensation_iterations:
        interpolate_arm_motion(env, current_target, safe_goal_target, finger_target, compensation_motion_s)
        env.set_robot_targets(safe_goal_target, finger_target)
        env.advance(settle_s)
        current_target = lane_hold.last_sent_arm_target if lane_hold is not None else safe_goal_target
    dwell_arm_q, _, _, _ = env.read_robot_state()
    dwell_arm_command = current_target if arm_target_is_absolute else current_target - dwell_arm_q
    exact_success, exact_metrics = advance_exact_success_dwell(
        env,
        goal_task_q,
        dwell_arm_command,
        finger_target,
        require_all_samples=True,
        sample_physical_validity=True,
        arm_target_is_absolute=arm_target_is_absolute,
        plug_body_index=plug_body_index,
        latch_body_index=latch_body_index,
        lane_hold=lane_hold,
    )
    task_q, task_qd = env.read_task_state()
    final_arm_q, _, _, _ = env.read_robot_state()
    error = scalar_goal_error(
        task_q,
        goal_task_q,
        plug_body_index=plug_body_index,
        latch_body_index=latch_body_index,
    )
    # Newton spatial velocities store world linear xyz first, angular xyz last.
    plug_speed = torch.linalg.vector_norm(task_qd[:, plug_body_index, :3], dim=-1)
    grasp = grasp_metrics(env, finger_target)
    collision = collision_metrics(env)
    if collision.contact_overflow:
        raise RuntimeError("Global contact-buffer overflow at the scripted-recovery result boundary.")
    success = (
        ik_valid
        & (error <= 0.002)
        & (plug_speed <= 0.03)
        & exact_success
        & exact_metrics["all_samples_collision_free"]
        & exact_metrics["all_samples_bilateral_grasp"]
        & exact_metrics["all_samples_finite"]
        & grasp.valid
        & collision.valid
        & task_state_is_finite_and_normalized(task_q, task_qd)
    )
    if arm_target_is_absolute:
        success &= exact_metrics["all_samples_arm_target_tracking_bounded"]
        success &= exact_metrics["maximum_arm_target_drift"] <= 1.0e-7
    if lane_hold is not None:
        lane_hold.deactivate(~ik_valid, reason="recovery-final-ik")
        lane_hold.deactivate(~grasp.valid, reason="recovery-final-grasp")
        lane_hold.deactivate(~collision.valid, reason="recovery-final-collision")
        lane_hold.deactivate(~success, reason="recovery-final-validation")
        success &= lane_hold.active_mask
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


def scripted_recovery(
    env: _ResetToolEnv,
    ik: FrankaResetIK,
    goal_task_q: torch.Tensor,
    orientation: torch.Tensor | None,
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
    plug_body_index: int | None = None,
    latch_body_index: int | None = None,
    arm_target_is_absolute: bool = False,
    lane_hold: _PerLaneTargetHold | None = None,
    motion_policy: str = "legacy-direct-joint",
    pick_insert_phase: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Dispatch scripted recovery while preserving the legacy default exactly."""
    legacy_policy = str(PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["legacy_default_motion_policy"])
    cartesian_policy = str(PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["motion_policy"])
    if motion_policy == legacy_policy:
        if pick_insert_phase is not None:
            raise ValueError("pick_insert_phase is only valid with the pick-insert Cartesian recovery policy.")
        return _scripted_recovery_legacy(
            env,
            ik,
            goal_task_q,
            orientation,
            finger_target,
            arm_target_start=arm_target_start,
            goal_arm_target=goal_arm_target,
            motion_s=motion_s,
            settle_s=settle_s,
            compensation_max_iterations=compensation_max_iterations,
            compensation_gain=compensation_gain,
            compensation_max_step_m=compensation_max_step_m,
            compensation_motion_s=compensation_motion_s,
            compensation_hold_s=compensation_hold_s,
            compensation_tolerance_m=compensation_tolerance_m,
            maximum_ik_joint_step_rad=maximum_ik_joint_step_rad,
            plug_body_index=plug_body_index,
            latch_body_index=latch_body_index,
            arm_target_is_absolute=arm_target_is_absolute,
            lane_hold=lane_hold,
        )
    if motion_policy != cartesian_policy:
        raise ValueError(f"motion_policy must be {legacy_policy!r} or {cartesian_policy!r}, got {motion_policy!r}.")

    def run(active_hold: _PerLaneTargetHold) -> tuple[torch.Tensor, dict[str, Any]]:
        return _scripted_recovery_cartesian_c2(
            env,
            ik,
            goal_task_q,
            orientation,
            finger_target,
            arm_target_start=arm_target_start,
            goal_arm_target=goal_arm_target,
            motion_s=motion_s,
            settle_s=settle_s,
            compensation_max_iterations=compensation_max_iterations,
            compensation_gain=compensation_gain,
            compensation_max_step_m=compensation_max_step_m,
            compensation_motion_s=compensation_motion_s,
            compensation_hold_s=compensation_hold_s,
            compensation_tolerance_m=compensation_tolerance_m,
            plug_body_index=plug_body_index,
            latch_body_index=latch_body_index,
            arm_target_is_absolute=arm_target_is_absolute,
            lane_hold=active_hold,
            pick_insert_phase=pick_insert_phase,
        )

    if lane_hold is not None:
        return run(lane_hold)
    measured_arm, _, _, _ = env.read_robot_state()
    initial_target = (
        measured_arm
        if arm_target_start is None
        else torch.as_tensor(
            arm_target_start,
            device=env.device,
            dtype=torch.float32,
        )
    )
    active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    with _PerLaneTargetHold(env, active, initial_target, finger_target) as owned_hold:
        success, metrics = run(owned_hold)
        metrics["lane_failure_masks"] = owned_hold.reason_masks
        return success, metrics


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
    "RJ45PickInsertResetToolEnv",
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
    "pick_insert_tool_physical_contract",
    "plug_relative_latch_angle",
    "randomized_orientations",
    "repeated_to_env_count",
    "runtime_reset_biased_arm_target",
    "save_torch_atomic",
    "scalar_goal_error",
    "scripted_recovery",
    "task_state_is_finite_and_normalized",
]
