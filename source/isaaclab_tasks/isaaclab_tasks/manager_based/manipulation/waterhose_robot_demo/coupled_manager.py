# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coupled Newton manager and solver configs for the waterhose robot demo."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import ClassVar

import numpy as np
import warp as wp
import newton
from newton import GeoType

from isaaclab.physics import PhysicsManager
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import (
    AdmmContactPairCfg,
    AdmmCouplingCfg,
    CoupledProxyCfg,
    CoupledSolverCfg,
    CoupledSolverEntryCfg,
    MJWarpSolverCfg,
    NewtonManager,
    NewtonSolverCfg,
    OneWayCouplingCfg,
    ProxyCouplingCfg,
)
from isaaclab_newton.physics.coupled_manager import NewtonCoupledManager

from .coupled_builder import (
    GRIPPER_DRIVER_DOFS,
    GRIPPER_FINGER_DOFS,
    WaterhoseCoupledBuildInfo,
    build_waterhose_coupled_builder,
    build_waterhose_robot_model,
)


_DEFAULT_ASSET_ROOT = str(
    Path(__file__).resolve().parents[5] / "isaaclab_assets" / "data" / "WaterhoseDemo"
)


class NewtonWaterhoseCoupledManager(NewtonCoupledManager):
    """Newton coupled-manager path for the waterhose robot demo."""

    _build_info: ClassVar[WaterhoseCoupledBuildInfo | None] = None
    _plug_body_id: ClassVar[int | None] = None
    _tip_body_id: ClassVar[int | None] = None
    _right_ee_body_id: ClassVar[int | None] = None
    _cable0_last_body_id: ClassVar[int | None] = None
    _plug_body_ids: ClassVar[list[int | None]] = []
    _tip_body_ids: ClassVar[list[int | None]] = []
    _right_ee_body_ids: ClassVar[list[int | None]] = []
    _cable0_last_body_ids: ClassVar[list[int | None]] = []
    _controller_ready: ClassVar[bool] = False
    _teleop_enabled: ClassVar[bool] = False
    _script_phase: ClassVar[int] = 0
    _script_phase_elapsed: ClassVar[float] = 0.0
    _script_last_time: ClassVar[float | None] = None
    _phase_start_pose: ClassVar[np.ndarray | None] = None
    _last_target_pose: ClassVar[np.ndarray | None] = None
    _manual_target_pose: ClassVar[np.ndarray | None] = None
    _ik_manager: ClassVar[object | None] = None
    _ik_solver: ClassVar[object | None] = None
    _ik_robot_model: ClassVar[object | None] = None
    _ik_joint_q: ClassVar[wp.array | None] = None
    _ik_target_pos: ClassVar[wp.array | None] = None
    _ik_target_rot: ClassVar[wp.array | None] = None
    _ik_pos_objs: ClassVar[list[object]] = []
    _ik_rot_objs: ClassVar[list[object]] = []
    _ik_fixed_tfs: ClassVar[list[wp.transform]] = []
    _gripper_open_value: ClassVar[float] = 0.0
    _gripper_closed_value: ClassVar[float] = 2.0 * 0.0036
    _plug_grasp_offset: ClassVar[np.ndarray] = np.array([0.0, -0.001, 0.01], dtype=np.float64)
    _approach_offset: ClassVar[np.ndarray] = np.array([0.0, 0.08, 0.0], dtype=np.float64)
    _engage_offset: ClassVar[np.ndarray] = np.array([0.01, 0.0, 0.0], dtype=np.float64)
    _retract_vector: ClassVar[np.ndarray] = np.array([0.0, 0.05, 0.0], dtype=np.float64)
    _grasp_orientation_offset: ClassVar[np.ndarray] = np.array([0.5, 0.5, -0.5, 0.5], dtype=np.float64)
    _SCRIPT_PHASES: ClassVar[tuple[str, ...]] = ("APPROACH", "ENGAGE", "GRASP", "HOLD_GRASP", "RETRACT", "DONE")
    _SCRIPT_DURATIONS: ClassVar[tuple[float, ...]] = (3.0, 1.5, 0.5, 0.5, 1.5, 999.0)

    @classmethod
    def instantiate_builder_from_stage(cls) -> None:
        cfg = cls._waterhose_cfg()
        num_envs = max(1, int(getattr(cfg, "num_envs", 1)))
        if num_envs != 1:
            raise RuntimeError(
                "The coupled-manager waterhose task is single-env: Newton's coupled ModelView does not yet "
                "support multi-world views for the MuJoCo (MJWarp) entry. Use Isaac-Waterhose-Robot-Demo-v0 "
                "for multi-env runs today."
            )
        builder, build_info = build_waterhose_coupled_builder(
            getattr(cfg, "asset_root", _DEFAULT_ASSET_ROOT),
            include_proxy_bodies=bool(getattr(cfg, "include_proxy_bodies", False)),
            num_envs=num_envs,
            env_spacing=float(getattr(cfg, "env_spacing", 2.5)),
        )
        NewtonManager.set_builder(builder)
        NewtonManager._num_envs = num_envs
        cls._build_info = build_info

    @classmethod
    def _build_solver(cls, model, solver_cfg: CoupledSolverCfg) -> None:
        cls._build_scene_mesh_sdfs(model)
        super()._build_solver(model, solver_cfg)
        cls._configure_vbd_solver()
        cls._restore_vbd_initial_body_poses()
        cls._resolve_tracked_bodies(model)
        cls._initialize_robot_control_targets()

    @classmethod
    def _solver_specific_clear(cls) -> None:
        cls._build_info = None
        cls._plug_body_id = None
        cls._tip_body_id = None
        cls._right_ee_body_id = None
        cls._cable0_last_body_id = None
        cls._plug_body_ids = []
        cls._tip_body_ids = []
        cls._right_ee_body_ids = []
        cls._cable0_last_body_ids = []
        cls._controller_ready = False
        cls._teleop_enabled = False
        cls._script_phase = 0
        cls._script_phase_elapsed = 0.0
        cls._script_last_time = None
        cls._phase_start_pose = None
        cls._last_target_pose = None
        cls._manual_target_pose = None
        cls._ik_manager = None
        cls._ik_solver = None
        cls._ik_robot_model = None
        cls._ik_joint_q = None
        cls._ik_target_pos = None
        cls._ik_target_rot = None
        cls._ik_pos_objs = []
        cls._ik_rot_objs = []
        cls._ik_fixed_tfs = []

    @classmethod
    def _build_scene_mesh_sdfs(cls, model) -> None:
        build_info = cls._build_info
        if build_info is None:
            return
        shape_type = model.shape_type.numpy()
        built_meshes: set[int] = set()
        for shape_id in build_info.scene_shape_ids:
            if shape_id < 0 or shape_id >= int(model.shape_count):
                continue
            if int(shape_type[shape_id]) != int(GeoType.MESH):
                continue
            mesh = model.shape_source[shape_id]
            if mesh is None or id(mesh) in built_meshes:
                continue
            try:
                mesh.build_sdf(max_resolution=64)
            except RuntimeError as exc:
                if "already has an SDF" not in str(exc):
                    raise
            built_meshes.add(id(mesh))

    @classmethod
    def _configure_vbd_solver(cls) -> None:
        try:
            vbd_solver = cls.get_entry_solver("vbd")
        except Exception:
            return
        if not hasattr(vbd_solver, "set_joint_constraint_mode"):
            return
        for joint_id in range(int(getattr(vbd_solver.model, "joint_count", 0))):
            vbd_solver.set_joint_constraint_mode(joint_id, hard=False)

    @classmethod
    def _restore_vbd_initial_body_poses(cls) -> None:
        """Restore VBD-owned cable poses after NewtonManager's initial global FK.

        The manager performs one generic FK pass before the coupled solver can
        install its FK articulation filter. That FK pass is valid for the MuJoCo
        robot, but it recomputes VBD cable bodies from zero joint coordinates
        and destroys the authored cable curve. VBD owns these body poses
        directly, so restore the task-local builder poses before the first
        simulation step and keep the VBD solver's previous-pose buffer aligned.
        """

        build_info = cls._build_info
        if build_info is None or build_info.vbd_initial_body_q is None:
            return

        if build_info.vbd_initial_body_q_by_env is not None:
            vbd_body_q_by_env = np.asarray(build_info.vbd_initial_body_q_by_env, dtype=np.float32)
        else:
            vbd_body_q = np.asarray(build_info.vbd_initial_body_q, dtype=np.float32)
            vbd_body_q_by_env = vbd_body_q.reshape(1, int(vbd_body_q.shape[0]), 7)

        num_envs = int(build_info.num_envs)
        env_body_count = int(build_info.env_body_count or (build_info.robot_body_count + build_info.vbd_body_count))
        vbd_body_count = int(build_info.vbd_body_count)

        for array in (
            getattr(NewtonManager._model, "body_q", None),
            getattr(NewtonManager._state_0, "body_q", None),
            getattr(NewtonManager._state_1, "body_q", None),
        ):
            if array is None:
                continue
            body_q_np = array.numpy()
            for env_id in range(num_envs):
                start = env_id * env_body_count + int(build_info.robot_body_count)
                end = start + vbd_body_count
                if end > int(body_q_np.shape[0]):
                    raise RuntimeError(
                        f"Cannot restore VBD body poses for env {env_id}: requested body range [{start}, {end}) "
                        f"but model has {body_q_np.shape[0]} bodies."
                    )
                body_q_np[start:end] = vbd_body_q_by_env[env_id]
            wp.copy(array, wp.array(body_q_np, dtype=wp.transform, device=array.device))

        try:
            vbd_solver = cls.get_entry_solver("vbd")
        except Exception:
            return
        body_q_prev = getattr(vbd_solver, "body_q_prev", None)
        if body_q_prev is not None:
            vbd_body_q_flat = vbd_body_q_by_env.reshape(num_envs * vbd_body_count, 7)
            if int(body_q_prev.shape[0]) == int(vbd_body_q_flat.shape[0]):
                wp.copy(body_q_prev, wp.array(vbd_body_q_flat, dtype=wp.transform, device=body_q_prev.device))

    @classmethod
    def _resolve_tracked_bodies(cls, model) -> None:
        labels = [str(label) for label in getattr(model, "body_label", [])]
        build_info = cls._build_info
        num_envs = max(1, int(getattr(build_info, "num_envs", 1)))
        cls._right_ee_body_ids = [
            _find_body_for_env(labels, "right_gripper_end_effector", env_id) for env_id in range(num_envs)
        ]
        cls._plug_body_ids = [_find_body_for_env(labels, "authored_head_0", env_id) for env_id in range(num_envs)]
        cls._tip_body_ids = [
            _find_body_for_env(labels, "water_hose_cable_0_edge_body_0", env_id) for env_id in range(num_envs)
        ]
        cls._cable0_last_body_ids = [
            _find_body_for_env(labels, "water_hose_cable_0_edge_body_42", env_id) for env_id in range(num_envs)
        ]
        cls._right_ee_body_id = cls._right_ee_body_ids[0]
        cls._plug_body_id = cls._plug_body_ids[0]
        cls._tip_body_id = cls._tip_body_ids[0]
        cls._cable0_last_body_id = cls._cable0_last_body_ids[0]

    @classmethod
    def _initialize_robot_control_targets(cls) -> None:
        build_info = cls._build_info
        state = NewtonManager._state_0
        control = NewtonManager._control
        if build_info is None or state is None or control is None:
            return
        target = getattr(control, "joint_target_pos", None)
        if target is None:
            return
        robot_q_count = int(build_info.robot_joint_q_count)
        env_q_count = int(build_info.env_joint_q_count or robot_q_count)
        num_envs = int(build_info.num_envs)
        if robot_q_count <= 0 or env_q_count <= 0:
            return
        target_np = target.numpy()
        joint_q_np = state.joint_q.numpy()
        for env_id in range(num_envs):
            start = env_id * env_q_count
            end = start + robot_q_count
            if end <= int(target_np.shape[0]) and end <= int(joint_q_np.shape[0]):
                target_np[start:end] = joint_q_np[start:end]
        wp.copy(target, wp.array(target_np, dtype=wp.float32, device=target.device))

    @classmethod
    def get_sim_time(cls) -> float:
        return float(PhysicsManager._sim_time)

    @classmethod
    def current_phase(cls) -> int:
        return int(cls._script_phase)

    @classmethod
    def get_right_ee_pose(cls) -> np.ndarray:
        return cls._body_pose(cls._right_ee_body_id, "right_gripper_end_effector")

    @classmethod
    def get_right_ee_poses(cls) -> np.ndarray:
        return cls._body_poses(cls._right_ee_body_ids, "right_gripper_end_effector")

    @classmethod
    def get_plug_pose(cls) -> np.ndarray:
        return cls._body_pose(cls._plug_body_id, "plug head")

    @classmethod
    def get_plug_poses(cls) -> np.ndarray:
        return cls._body_poses(cls._plug_body_ids, "plug head")

    @classmethod
    def get_tip_pose(cls) -> np.ndarray:
        return cls._body_pose(cls._tip_body_id, "cable tip")

    @classmethod
    def get_tip_poses(cls) -> np.ndarray:
        return cls._body_poses(cls._tip_body_ids, "cable tip")

    @classmethod
    def is_finite(cls) -> bool:
        state = NewtonManager._state_0
        if state is None:
            return False
        return bool(np.isfinite(state.body_q.numpy()).all() and np.isfinite(state.body_qd.numpy()).all())

    @classmethod
    def is_done(cls, max_demo_steps: int = 0) -> bool:
        del max_demo_steps
        return False

    @classmethod
    def is_success(cls) -> bool:
        return False

    @classmethod
    def set_teleop_enabled(cls, enabled: bool) -> None:
        cls._teleop_enabled = bool(enabled)
        if cls._teleop_enabled:
            cls._setup_controller()
            cls._manual_target_pose = cls.get_right_ee_pose()

    @classmethod
    def teleop_enabled(cls) -> bool:
        return bool(cls._teleop_enabled)

    @classmethod
    def apply_scripted_control(cls) -> None:
        cls._setup_controller()
        cls._advance_script_phase_clock()
        target_pose, gripper_value = cls._scripted_target()
        cls._solve_ik_to_target(target_pose, gripper_value)

    @classmethod
    def apply_teleop_command(cls, command) -> None:
        cls._setup_controller()
        command_np = command.detach().cpu().numpy().astype(np.float64, copy=False)
        if cls._manual_target_pose is None:
            cls._manual_target_pose = cls.get_right_ee_pose()

        pose = cls._manual_target_pose.copy()
        pose[:3] += command_np[:3]
        pose[3:] = _normalize_quat(_quat_multiply(_quat_from_rotvec(command_np[3:6]), pose[3:]))
        gripper = cls._gripper_open_value
        if command_np.shape[0] > 6:
            gripper = float(
                np.clip(cls._gripper_open_value - command_np[6] * cls._gripper_open_value, 0.0, cls._gripper_open_value)
            )
        cls._manual_target_pose = pose
        cls._solve_ik_to_target(pose, gripper)

    @classmethod
    def _setup_controller(cls) -> None:
        if cls._controller_ready:
            return

        from isaaclab_newton.ik.newton_ik_manager import NewtonIKManager, NewtonIKPoseObjective  # noqa: PLC0415
        from isaaclab_newton.ik.newton_ik_manager_cfg import NewtonIKManagerCfg  # noqa: PLC0415

        build_info = cls._build_info
        state = NewtonManager._state_0
        if build_info is None or state is None:
            raise RuntimeError("Cannot initialize waterhose controller before the Newton state is ready.")

        cfg = cls._waterhose_cfg()
        device = PhysicsManager._device
        cls._ik_robot_model = build_waterhose_robot_model(getattr(cfg, "asset_root", _DEFAULT_ASSET_ROOT), device)
        labels = [str(label) for label in cls._ik_robot_model.body_label]
        objective_specs = [
            ("right_gripper_end_effector", 1.0),
            ("left_gripper_end_effector", 1.0),
            ("torso_hip_yaw", 50.0),
        ]
        body_ids = [_find_body(labels, token) for token, _weight in objective_specs]
        if any(body_id is None for body_id in body_ids):
            raise RuntimeError(f"Could not resolve ADMM IK bodies from labels: {objective_specs}")

        pose_objectives = [
            NewtonIKPoseObjective(
                name=f"ee_{index}",
                link_index=int(body_id),
                position_weight=weight,
                rotation_weight=weight,
            )
            for index, (body_id, (_token, weight)) in enumerate(zip(body_ids, objective_specs))
        ]
        cls._ik_manager = NewtonIKManager(
            NewtonIKManagerCfg(
                command_type="pose",
                use_relative_mode=False,
                iterations=24,
                lambda_initial=0.1,
                jacobian_mode="analytic",
                joint_limit_weight=10.0,
            ),
            model=cls._ik_robot_model,
            num_envs=1,
            device=str(device),
            pose_objectives=pose_objectives,
        )
        cls._ik_solver = cls._ik_manager.solver
        cls._ik_pos_objs = [cls._ik_manager.position_objectives[f"ee_{index}"] for index in range(len(body_ids))]
        cls._ik_rot_objs = [cls._ik_manager.rotation_objectives[f"ee_{index}"] for index in range(len(body_ids))]

        body_q = state.body_q.numpy()
        cls._ik_fixed_tfs = [wp.transform(*body_q[int(body_ids[index])]) for index in (1, 2)]
        robot_q_count = int(cls._ik_robot_model.joint_coord_count)
        cls._ik_joint_q = wp.array(
            state.joint_q.numpy()[:robot_q_count].reshape(1, -1),
            dtype=wp.float32,
            device=device,
        )
        cls._ik_target_pos = wp.zeros(1, dtype=wp.vec3, device=device)
        cls._ik_target_rot = wp.zeros(1, dtype=wp.vec4, device=device)

        joint_limit_upper = cls._ik_robot_model.joint_limit_upper.numpy()
        cls._gripper_open_value = float(joint_limit_upper[GRIPPER_DRIVER_DOFS[0]]) * 0.5
        cls._phase_start_pose = cls.get_right_ee_pose()
        cls._last_target_pose = cls._phase_start_pose.copy()
        cls._manual_target_pose = cls._phase_start_pose.copy()
        cls._compute_grasp_offset()
        cls._controller_ready = True

    @classmethod
    def _advance_script_phase_clock(cls) -> None:
        sim_time = cls.get_sim_time()
        dt = 1.0 / 100.0 if cls._script_last_time is None else max(0.0, sim_time - cls._script_last_time)
        cls._script_last_time = sim_time
        cls._script_phase_elapsed += dt

        duration = cls._SCRIPT_DURATIONS[min(cls._script_phase, len(cls._SCRIPT_DURATIONS) - 1)]
        if cls._script_phase_elapsed < duration or cls._script_phase >= len(cls._SCRIPT_PHASES) - 1:
            return
        cls._script_phase += 1
        cls._script_phase_elapsed = 0.0
        cls._phase_start_pose = cls.get_right_ee_pose()

    @classmethod
    def _scripted_target(cls) -> tuple[np.ndarray, float]:
        phase_name = cls._SCRIPT_PHASES[min(cls._script_phase, len(cls._SCRIPT_PHASES) - 1)]
        phase_duration = cls._SCRIPT_DURATIONS[min(cls._script_phase, len(cls._SCRIPT_DURATIONS) - 1)]
        t = min(1.0, cls._script_phase_elapsed / max(phase_duration, 1.0e-6))
        t = t * t * (3.0 - 2.0 * t)

        plug_pose = cls.get_plug_pose()
        plug_pos = plug_pose[:3]
        plug_quat = plug_pose[3:]
        start_pose = cls._phase_start_pose if cls._phase_start_pose is not None else cls.get_right_ee_pose()

        target = start_pose.copy()
        gripper = cls._gripper_open_value
        if phase_name == "APPROACH":
            target[:3] = plug_pos + _quat_rotate(plug_quat, cls._plug_grasp_offset + cls._approach_offset)
            target[3:] = _normalize_quat(_quat_multiply(plug_quat, cls._grasp_orientation_offset))
        elif phase_name == "ENGAGE":
            target[:3] = plug_pos + _quat_rotate(plug_quat, cls._plug_grasp_offset) + cls._engage_offset
            target[3:] = _normalize_quat(_quat_multiply(plug_quat, cls._grasp_orientation_offset))
        elif phase_name == "GRASP":
            gripper = cls._gripper_open_value + (cls._gripper_closed_value - cls._gripper_open_value) * t
        elif phase_name == "HOLD_GRASP":
            gripper = cls._gripper_closed_value
        elif phase_name == "RETRACT":
            target[:3] = start_pose[:3] + _quat_rotate(plug_quat, cls._retract_vector)
            gripper = cls._gripper_closed_value

        pose = _interpolate_pose(start_pose, target, t)
        cls._last_target_pose = pose.copy()
        return pose, float(gripper)

    @classmethod
    def _solve_ik_to_target(cls, target_pose: np.ndarray, gripper_value: float) -> None:
        if cls._ik_solver is None or cls._ik_joint_q is None or cls._ik_target_pos is None or cls._ik_target_rot is None:
            return

        device = PhysicsManager._device
        pos = target_pose[:3].astype(np.float32)
        quat = _normalize_quat(target_pose[3:]).astype(np.float32)
        wp.copy(
            cls._ik_target_pos,
            wp.array([wp.vec3(float(pos[0]), float(pos[1]), float(pos[2]))], dtype=wp.vec3, device=device),
        )
        wp.copy(
            cls._ik_target_rot,
            wp.array(
                [wp.vec4(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))],
                dtype=wp.vec4,
                device=device,
            ),
        )
        cls._ik_pos_objs[0].set_target_positions(cls._ik_target_pos)
        cls._ik_rot_objs[0].set_target_rotations(cls._ik_target_rot)

        for objective_index, tf in enumerate(cls._ik_fixed_tfs, start=1):
            cls._ik_pos_objs[objective_index].set_target_position(0, wp.transform_get_translation(tf))
            q = wp.transform_get_rotation(tf)
            cls._ik_rot_objs[objective_index].set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

        cls._ik_solver.step(cls._ik_joint_q, cls._ik_joint_q, iterations=24)

        build_info = cls._build_info
        control = NewtonManager._control
        if build_info is None or control is None or getattr(control, "joint_target_pos", None) is None:
            return
        target = control.joint_target_pos
        target_np = target.numpy()
        robot_q_count = int(build_info.robot_joint_q_count)
        env_q_count = int(build_info.env_joint_q_count or robot_q_count)
        num_envs = int(build_info.num_envs)
        ik_joint_q = cls._ik_joint_q.numpy().reshape(-1)[:robot_q_count]
        for env_id in range(num_envs):
            start = env_id * env_q_count
            end = start + robot_q_count
            if end > int(target_np.shape[0]):
                continue
            target_np[start:end] = ik_joint_q
            cls._write_gripper_targets(target_np, gripper_value, offset=start)
        wp.copy(target, wp.array(target_np, dtype=wp.float32, device=target.device))

    @classmethod
    def _write_gripper_targets(cls, target_np: np.ndarray, gripper_value: float, *, offset: int = 0) -> None:
        right_value = float(gripper_value)
        left_value = cls._gripper_open_value
        target_np[offset + GRIPPER_DRIVER_DOFS[0]] = right_value
        target_np[offset + GRIPPER_DRIVER_DOFS[1]] = left_value
        target_np[offset + GRIPPER_FINGER_DOFS[0]] = -right_value
        target_np[offset + GRIPPER_FINGER_DOFS[1]] = right_value
        target_np[offset + GRIPPER_FINGER_DOFS[2]] = -left_value
        target_np[offset + GRIPPER_FINGER_DOFS[3]] = left_value

    @classmethod
    def _compute_grasp_offset(cls) -> None:
        if cls._plug_body_id is None or cls._cable0_last_body_id is None:
            cls._plug_grasp_offset = np.array([0.0, -0.001, 0.01], dtype=np.float64)
            return
        body_q = NewtonManager._state_0.body_q.numpy()
        head_pose = body_q[int(cls._plug_body_id)]
        capsule_pose = body_q[int(cls._cable0_last_body_id)]
        toward_cable_world = capsule_pose[:3] - head_pose[:3]
        norm = np.linalg.norm(toward_cable_world)
        if norm <= 1.0e-8:
            toward_cable_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            toward_cable_world = toward_cable_world / norm
            toward_cable_local = _quat_rotate(_quat_inverse(head_pose[3:]), toward_cable_world)
        cls._plug_grasp_offset = toward_cable_local * 0.01 + np.array([0.0, -0.001, 0.0], dtype=np.float64)

    @classmethod
    def _body_pose(cls, body_id: int | None, name: str) -> np.ndarray:
        if body_id is None:
            raise RuntimeError(f"Waterhose ADMM body '{name}' is not available.")
        state = NewtonManager._state_0
        if state is None:
            raise RuntimeError("Waterhose ADMM state is not initialized.")
        return state.body_q.numpy()[int(body_id)].copy()

    @classmethod
    def _body_poses(cls, body_ids: list[int], name: str) -> np.ndarray:
        if not body_ids:
            num_envs = max(1, int(getattr(cls._build_info, "num_envs", 1)))
            return np.zeros((num_envs, 7), dtype=np.float32)
        state = NewtonManager._state_0
        if state is None:
            raise RuntimeError("Waterhose ADMM state is not initialized.")
        body_q = state.body_q.numpy()
        poses = np.zeros((len(body_ids), 7), dtype=np.float32)
        for index, body_id in enumerate(body_ids):
            if body_id is None:
                raise RuntimeError(f"Waterhose ADMM body '{name}' is not available for env {index}.")
            poses[index] = body_q[int(body_id)]
        return poses

    @staticmethod
    def _waterhose_cfg():
        cfg = PhysicsManager._cfg
        return getattr(cfg, "solver_cfg", cfg)


@configclass
class WaterhoseVBDSolverCfg(NewtonSolverCfg):
    """Configuration for the VBD sub-solver used by the ADMM waterhose task."""

    solver_type: str = "vbd"
    requires_graph_coloring: ClassVar[bool] = True

    iterations: int = 40
    friction_epsilon: float = 0.1
    rigid_avbd_beta: float = 1.0e3
    rigid_contact_k_start: float = 1.0e3
    rigid_body_contact_buffer_size: int = 1024
    rigid_body_particle_contact_buffer_size: int = 1
    rigid_contact_hard: bool = True
    rigid_joint_linear_ke: float = 1.0e6
    rigid_joint_angular_ke: float = 1.0e6


@configclass
class WaterhoseAdmmSolverCfg(CoupledSolverCfg):
    """ADMM coupled solver config for the waterhose robot demo."""

    class_type: type[NewtonManager] | str = NewtonWaterhoseCoupledManager
    solver_type: str = "waterhose_admm"
    coupling_type: str = "admm"
    requires_graph_coloring: ClassVar[bool] = True

    asset_root: str = _DEFAULT_ASSET_ROOT
    entries: list[CoupledSolverEntryCfg] = field(default_factory=list)
    admm_coupling: AdmmCouplingCfg = AdmmCouplingCfg(
        iterations=5,
        rho=15.0,
        gamma=0.1,
        baumgarte=0.005,
        contact_pairs=[AdmmContactPairCfg(source="mujoco", destination="vbd", contact_distance=0.003)],
    )
    use_collision_pipeline: bool | None = False

    def __post_init__(self) -> None:
        if self.entries:
            return
        self.entries = [
            CoupledSolverEntryCfg(
                name="mujoco",
                solver_cfg=MJWarpSolverCfg(
                    solver="newton",
                    integrator="implicitfast",
                    cone="elliptic",
                    iterations=20,
                    ls_iterations=20,
                    ls_parallel=True,
                    use_mujoco_contacts=False,
                    impratio=1000.0,
                ),
                body_label_patterns=[r"mujoco/.*"],
                include_child_joints=True,
                include_body_shapes=True,
            ),
            CoupledSolverEntryCfg(
                name="vbd",
                solver_cfg=WaterhoseVBDSolverCfg(),
                solver_class="newton.solvers:SolverVBD",
                body_label_patterns=[r"vbd/.*"],
                include_child_joints=True,
                include_body_shapes=True,
            ),
        ]


def waterhose_vbd_proxy_collision_pipeline(model_view):
    """Create the VBD collision pipeline used by one-way gripper proxies."""

    return newton.CollisionPipeline(model_view, broad_phase="explicit", rigid_contact_max=30000)


def waterhose_mujoco_single_model_view(model_view) -> None:
    """Run MJWarp as one combined model when the parent Newton model has worlds.

    The coupled ModelView disables non-owned VBD bodies but keeps parent-model
    indexing intact for state reconciliation. MJWarp's separate-world converter
    expects a compact homogeneous robot-only model, so the one-way source entry
    uses a single MJWarp model over the view instead.
    """

    model_view.world_count = 1


@configclass
class WaterhoseOneWaySolverCfg(CoupledSolverCfg):
    """One-way coupled solver config for the waterhose robot demo.

    The MuJoCo robot entry is authoritative. Its gripper finger states are
    copied into duplicated VBD proxy bodies; the VBD cable/plug entry collides
    against those proxies, and proxy feedback is discarded by
    ``coupling_type="one_way"``.
    """

    class_type: type[NewtonManager] | str = NewtonWaterhoseCoupledManager
    solver_type: str = "waterhose_one_way"
    coupling_type: str = "one_way"
    requires_graph_coloring: ClassVar[bool] = True

    asset_root: str = _DEFAULT_ASSET_ROOT
    include_proxy_bodies: bool = True
    num_envs: int = 1
    env_spacing: float = 2.5
    entries: list[CoupledSolverEntryCfg] = field(default_factory=list)
    one_way_coupling: OneWayCouplingCfg = OneWayCouplingCfg(
        proxies=[
            CoupledProxyCfg(
                source="mujoco",
                destination="vbd",
                body_name_patterns=[rf"{name}" for name in (
                    "right_gripper_leftfinger",
                    "right_gripper_rightfinger",
                    "left_gripper_leftfinger",
                    "left_gripper_rightfinger",
                )],
                proxy_body_name_patterns=[rf"proxy_{name}" for name in (
                    "right_gripper_leftfinger",
                    "right_gripper_rightfinger",
                    "left_gripper_leftfinger",
                    "left_gripper_rightfinger",
                )],
                mode="lagged",
                mass_scale=1.0,
                collision_pipeline_factory=waterhose_vbd_proxy_collision_pipeline,
                collide_interval=5,
            )
        ]
    )
    use_collision_pipeline: bool | None = False

    def __post_init__(self) -> None:
        if self.entries:
            return
        self.entries = [
            CoupledSolverEntryCfg(
                name="mujoco",
                solver_cfg=MJWarpSolverCfg(
                    solver="newton",
                    integrator="implicitfast",
                    cone="elliptic",
                    iterations=20,
                    ls_iterations=10,
                    ls_parallel=True,
                    use_mujoco_contacts=False,
                    impratio=1000.0,
                ),
                body_label_patterns=[r".*mujoco/.*"],
                include_child_joints=True,
                include_body_shapes=True,
                configure_view=waterhose_mujoco_single_model_view,
                solver_kwargs={"separate_worlds": False},
            ),
            CoupledSolverEntryCfg(
                name="vbd",
                solver_cfg=WaterhoseVBDSolverCfg(iterations=15, rigid_contact_hard=False),
                solver_class="newton.solvers:SolverVBD",
                body_label_patterns=[r".*vbd/.*"],
                include_child_joints=True,
                include_body_shapes=True,
            ),
        ]


@configclass
class WaterhoseTwoWaySolverCfg(WaterhoseOneWaySolverCfg):
    """Experimental two-way proxy coupling for the waterhose robot demo.

    Uses the same embedded gripper proxies as the one-way config, but harvested
    proxy contact wrenches are fed back to the MuJoCo robot
    (``coupling_type="proxy"``). Newton applies the full proxy wrench including
    tangential friction, so the robot reacts more strongly than the one-way
    default. Treat as experimental.
    """

    solver_type: str = "waterhose_two_way"
    coupling_type: str = "proxy"
    proxy_coupling: ProxyCouplingCfg = ProxyCouplingCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.proxy_coupling.proxies:
            self.proxy_coupling.proxies = list(self.one_way_coupling.proxies)
            self.proxy_coupling.iterations = 1


def _find_body(labels: list[str], suffix_or_token: str) -> int | None:
    for body_id, label in enumerate(labels):
        if label.endswith(suffix_or_token) or suffix_or_token in label:
            return body_id
    return None


def _find_body_for_env(labels: list[str], suffix_or_token: str, env_id: int) -> int | None:
    env_prefix = f"env_{env_id}/"
    for body_id, label in enumerate(labels):
        if not label.startswith(env_prefix):
            continue
        if label.endswith(suffix_or_token) or suffix_or_token in label:
            return body_id
    return _find_body(labels, suffix_or_token) if env_id == 0 else None


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 1.0e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = left
    x2, y2, z2, w2 = right
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _quat_inverse(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quat(quat)
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float64)


def _quat_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    quat = _normalize_quat(quat)
    xyz = quat[:3]
    vec = np.asarray(vec, dtype=np.float64)
    t = 2.0 * np.cross(xyz, vec)
    return vec + quat[3] * t + np.cross(xyz, t)


def _quat_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(rotvec))
    if angle <= 1.0e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = rotvec / angle
    half = 0.5 * angle
    return np.array([axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half), np.cos(half)])


def _interpolate_pose(start: np.ndarray, target: np.ndarray, t: float) -> np.ndarray:
    t = float(np.clip(t, 0.0, 1.0))
    pose = np.empty(7, dtype=np.float64)
    pose[:3] = (1.0 - t) * start[:3] + t * target[:3]
    q0 = _normalize_quat(start[3:])
    q1 = _normalize_quat(target[3:])
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        pose[3:] = _normalize_quat(q0 + t * (q1 - q0))
        return pose
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    a = np.sin((1.0 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta
    pose[3:] = _normalize_quat(a * q0 + b * q1)
    return pose
