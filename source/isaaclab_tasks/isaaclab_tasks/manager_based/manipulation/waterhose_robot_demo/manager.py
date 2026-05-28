# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IsaacLab physics manager for the waterhose robot demo."""

from __future__ import annotations

import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaaclab.physics import PhysicsEvent, PhysicsManager, SceneDataBackend, SceneDataFormat
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import NewtonManager, NewtonSolverCfg
from isaaclab_newton.physics.newton_manager import NewtonSceneDataBackend

from .kit_display import assign_display_prim_paths, author_display_usd


_DEFAULT_ASSET_ROOT = str(
    Path(__file__).resolve().parents[5] / "isaaclab_assets" / "data" / "WaterhoseDemo"
)


class _WaterhoseSceneDataBackend(NewtonSceneDataBackend):
    """Scene-data adapter exposing the manager's current display model."""

    def __init__(self):
        self._scene_data = SceneDataFormat.Transform()

    @property
    def transforms(self) -> SceneDataFormat.Transform:
        state = NewtonWaterhoseManager.get_display_state()
        self._scene_data.transforms = None if state is None else state.body_q
        return self._scene_data

    @property
    def transform_count(self) -> int:
        model = NewtonWaterhoseManager.get_visualization_model()
        return 0 if model is None else int(model.body_count)

    @property
    def transform_paths(self) -> list[str]:
        model = NewtonWaterhoseManager.get_visualization_model()
        if model is None or model.body_label is None:
            return []
        return list(model.body_label)


class NewtonWaterhoseManager(NewtonManager):
    """Owns the local Newton waterhose runtime through IsaacLab's physics lifecycle."""

    _runtime: Any | None = None
    _viewer: Any | None = None
    _scene_data_backend: _WaterhoseSceneDataBackend | None = None
    _visualization_model: Any | None = None
    _display_state_0: Any | None = None
    _display_state_1: Any | None = None
    _display_control: Any | None = None
    _combined_runtime: Any | None = None
    _combined_model: Any | None = None
    _combined_state_0: Any | None = None
    _combined_state_1: Any | None = None
    _combined_control: Any | None = None
    _combined_robot_body_count: int = 0
    _combined_robot_shape_count: int = 0
    _kit_skipped_shape_ids: set[int] = set()
    _kit_display_model_id: int | None = None
    _kit_display_ready: bool = False
    _teleop_enabled: bool = False
    _is_playing: bool = True

    @classmethod
    def initialize(cls, sim_context) -> None:
        super().initialize(sim_context)
        cls._scene_data_backend = _WaterhoseSceneDataBackend()
        NewtonManager._scene_data_backend = cls._scene_data_backend
        cls._runtime = None
        cls._viewer = None
        cls._visualization_model = None
        cls._display_state_0 = None
        cls._display_state_1 = None
        cls._display_control = None
        cls._clear_combined_display()
        cls._clear_kit_display_state()
        cls._teleop_enabled = False
        cls._is_playing = True

    @classmethod
    def reset(cls, soft: bool = False) -> None:
        if soft and cls._runtime is not None:
            cls._publish_display_state()
            return
        cls._rebuild_runtime()

    @classmethod
    def forward(cls) -> None:
        import warp as wp  # noqa: PLC0415

        with wp.ScopedDevice(PhysicsManager._device):
            cls._publish_display_state()
            NewtonManager.sync_transforms_to_usd()

    @classmethod
    def get_scene_data_backend(cls) -> SceneDataBackend:
        if cls._scene_data_backend is None:
            cls._scene_data_backend = _WaterhoseSceneDataBackend()
        return cls._scene_data_backend

    @classmethod
    def step(cls) -> None:
        runtime = cls._runtime
        if runtime is None or not cls._is_playing:
            return
        import warp as wp  # noqa: PLC0415

        with wp.ScopedDevice(PhysicsManager._device):
            runtime.step()
            cls._publish_display_state()
        PhysicsManager._sim_time = float(getattr(runtime, "sim_time", PhysicsManager._sim_time))

    @classmethod
    def pre_render(cls) -> None:
        import warp as wp  # noqa: PLC0415

        with wp.ScopedDevice(PhysicsManager._device):
            cls._publish_display_state()
            NewtonManager.pre_render()

    @classmethod
    def close(cls) -> None:
        cls._close_runtime()
        cls._clear_newton_visualizer_state()
        cls._visualization_model = None
        cls._display_state_0 = None
        cls._display_state_1 = None
        cls._display_control = None
        cls._clear_combined_display()
        cls._clear_kit_display_state()
        cls._scene_data_backend = None
        super().close()

    @classmethod
    def _build_solver(cls, model, solver_cfg) -> None:
        """Satisfy the NewtonManager subclass contract.

        The waterhose demo owns a split MuJoCo/VBD runtime instead of a single
        Newton solver attached to one canonical model, so the normal
        NewtonManager.initialize_solver() path is intentionally unused.
        """
        raise RuntimeError("NewtonWaterhoseManager builds its split runtime in reset(), not initialize_solver().")

    @classmethod
    def play(cls) -> None:
        cls._is_playing = True

    @classmethod
    def pause(cls) -> None:
        cls._is_playing = False

    @classmethod
    def stop(cls) -> None:
        cls._is_playing = False

    @classmethod
    def wait_for_playing(cls) -> None:
        return

    @classmethod
    def set_decimation(cls, decimation: int) -> None:
        NewtonManager._decimation = max(1, int(decimation))

    @classmethod
    def handles_decimation(cls) -> bool:
        return False

    @classmethod
    def get_runtime(cls):
        return cls._runtime

    @classmethod
    def get_visualization_model(cls):
        return cls._visualization_model

    @classmethod
    def get_display_state(cls):
        return cls._display_state_0

    @classmethod
    def get_sim_time(cls) -> float:
        runtime = cls._runtime
        return float(getattr(runtime, "sim_time", PhysicsManager._sim_time))

    @classmethod
    def set_teleop_enabled(cls, enabled: bool) -> None:
        cls._teleop_enabled = bool(enabled)
        runtime = cls._runtime
        if cls._teleop_enabled and runtime is not None and getattr(runtime, "auto_mode", True):
            import warp as wp  # noqa: PLC0415

            with wp.ScopedDevice(PhysicsManager._device):
                runtime.auto_mode = False
                runtime._stop_auto_mode()

    @classmethod
    def teleop_enabled(cls) -> bool:
        return bool(cls._teleop_enabled)

    @classmethod
    def apply_teleop_command(cls, command: torch.Tensor) -> None:
        """Apply a 7D relative end-effector command to the manual IK target."""
        runtime = cls._runtime
        if runtime is None:
            return
        if getattr(runtime, "auto_mode", True):
            runtime.auto_mode = False
            runtime._stop_auto_mode()

        import warp as wp  # noqa: PLC0415

        with wp.ScopedDevice(PhysicsManager._device):
            cmd = command.detach().to("cpu", dtype=torch.float32).numpy().reshape(-1)
            if cmd.shape[0] < 6:
                return

            tf = runtime.ee_tfs[0]
            pos = wp.transform_get_translation(tf)
            quat = wp.transform_get_rotation(tf)
            dp = cmd[:3]
            pos = pos + wp.vec3(float(dp[0]), float(dp[1]), float(dp[2]))

            axis_angle_eef = cmd[3:6]
            angle = float(np.linalg.norm(axis_angle_eef))
            if angle > 1.0e-8:
                axis = axis_angle_eef / angle
                dq = wp.quat_from_axis_angle(wp.vec3(float(axis[0]), float(axis[1]), float(axis[2])), angle)
                quat = wp.normalize(quat * dq)
            runtime.ee_tfs[0] = wp.transform(pos, quat)

            if cmd.shape[0] >= 7:
                gripper_value = float(runtime.sm_gripper_open_value if cmd[6] > 0.0 else runtime.sm_gripper_closed_value)
                gripper_np = runtime.gripper_targets.numpy()
                gripper_np[0] = gripper_value
                wp.copy(runtime.gripper_targets, wp.array(gripper_np, dtype=wp.float32, device=runtime.gripper_targets.device))
                runtime.gripper_targets_list[0] = gripper_value
                runtime._sync_gripper_followers()

    @classmethod
    def current_phase(cls) -> int:
        runtime = cls._runtime
        if runtime is None or not hasattr(runtime, "sm_task_idx") or not hasattr(runtime, "sm_task_schedule"):
            return 0
        task_idx = int(runtime.sm_task_idx.numpy()[0])
        schedule = runtime.sm_task_schedule.numpy()
        return int(schedule[min(task_idx, len(schedule) - 1)])

    @classmethod
    def get_plug_pose(cls) -> np.ndarray:
        runtime = cls._require_runtime()
        body_id = int(getattr(runtime, "cable_head_body_idx", 0))
        return runtime.vbd_state_0.body_q.numpy()[body_id].copy()

    @classmethod
    def get_tip_pose(cls) -> np.ndarray:
        runtime = cls._require_runtime()
        body_id = int(getattr(runtime, "tip_capsule_body_idx", 0))
        return runtime.vbd_state_0.body_q.numpy()[body_id].copy()

    @classmethod
    def get_right_ee_pose(cls) -> np.ndarray:
        runtime = cls._require_runtime()
        labels = getattr(runtime.mujoco_model, "body_label", [])
        suffix = "/right_gripper_end_effector"
        for body_id, label in enumerate(labels):
            if label == "right_gripper_end_effector" or str(label).endswith(suffix):
                return runtime.state_0.body_q.numpy()[body_id].copy()
        raise RuntimeError("Body 'right_gripper_end_effector' not found in waterhose robot model.")

    @classmethod
    def is_finite(cls) -> bool:
        runtime = cls._runtime
        if runtime is None:
            return False
        return bool(
            np.isfinite(runtime.state_0.body_q.numpy()).all()
            and np.isfinite(runtime.state_0.body_qd.numpy()).all()
            and np.isfinite(runtime.vbd_state_0.body_q.numpy()).all()
            and np.isfinite(runtime.vbd_state_0.body_qd.numpy()).all()
        )

    @classmethod
    def is_success(cls) -> bool:
        """Return whether the plug has reached the task success condition."""
        runtime = cls._runtime
        if runtime is None or not cls.is_finite():
            return False
        if bool(getattr(runtime, "auto_mode", True)):
            return cls.current_phase() == 18
        try:
            tip_pos = runtime.vbd_state_0.body_q.numpy()[int(runtime.tip_capsule_body_idx), :3]
            socket_pos = np.asarray(runtime._socket_pos_np, dtype=np.float64)
            insertion_dir = np.asarray(runtime._insertion_dir_np, dtype=np.float64)
            insertion_dir /= max(float(np.linalg.norm(insertion_dir)), 1.0e-12)
            delta = np.asarray(tip_pos, dtype=np.float64) - socket_pos
            axial_depth = float(np.dot(delta, insertion_dir))
            lateral = float(np.linalg.norm(delta - axial_depth * insertion_dir))
            target_depth = float(getattr(runtime, "_insert_snap_depth", runtime.insert_final_depth))
            return axial_depth >= target_depth and lateral <= 0.025
        except (AttributeError, IndexError, TypeError, ValueError):
            return False

    @classmethod
    def get_recording_state(cls, device: str | torch.device | None = None) -> dict[str, torch.Tensor]:
        """Return a batched tensor snapshot of the local Newton runtime."""
        runtime = cls._runtime
        if runtime is None:
            return {}
        device = PhysicsManager._device if device is None else device

        def tensor(value, dtype=torch.float32) -> torch.Tensor:
            array = np.array(value, copy=True)
            return torch.as_tensor(array, dtype=dtype, device=device).unsqueeze(0)

        task_idx = 0
        task_elapsed = 0.0
        if hasattr(runtime, "sm_task_idx"):
            task_idx = int(runtime.sm_task_idx.numpy()[0])
        if hasattr(runtime, "sm_task_time_elapsed"):
            task_elapsed = float(runtime.sm_task_time_elapsed.numpy()[0])

        return {
            "robot_body_q": tensor(runtime.state_0.body_q.numpy()),
            "robot_body_qd": tensor(runtime.state_0.body_qd.numpy()),
            "vbd_body_q": tensor(runtime.vbd_state_0.body_q.numpy()),
            "vbd_body_qd": tensor(runtime.vbd_state_0.body_qd.numpy()),
            "joint_target_pos": tensor(runtime.control.joint_target_pos.numpy()),
            "gripper_targets": tensor(runtime.gripper_targets.numpy()),
            "right_ee_pose": tensor(cls.get_right_ee_pose()),
            "plug_pose": tensor(cls.get_plug_pose()),
            "tip_pose": tensor(cls.get_tip_pose()),
            "phase": torch.tensor([[float(cls.current_phase())]], dtype=torch.float32, device=device),
            "task_index": torch.tensor([[task_idx]], dtype=torch.int64, device=device),
            "task_elapsed": torch.tensor([[task_elapsed]], dtype=torch.float32, device=device),
            "frame_count": torch.tensor([[int(getattr(runtime, "frame_count", 0))]], dtype=torch.int64, device=device),
            "sim_time": torch.tensor([[float(getattr(runtime, "sim_time", 0.0))]], dtype=torch.float32, device=device),
            "auto_mode": torch.tensor([[bool(getattr(runtime, "auto_mode", True))]], dtype=torch.bool, device=device),
        }

    @classmethod
    def is_done(cls, max_demo_steps: int = 0) -> bool:
        runtime = cls._runtime
        if runtime is None:
            return True
        if max_demo_steps > 0 and int(getattr(runtime, "frame_count", 0)) >= int(max_demo_steps):
            return True
        return cls.current_phase() == 18

    @classmethod
    def reset_runtime(cls) -> None:
        runtime = cls._runtime
        if runtime is not None and int(getattr(runtime, "frame_count", 0)) == 0:
            cls._publish_display_state()
            PhysicsManager._sim_time = cls.get_sim_time()
            return
        cls._rebuild_runtime()

    @classmethod
    def _rebuild_runtime(cls) -> None:
        cls._close_runtime()
        cls._ensure_newton_on_path()
        runtime_args = cls._make_runtime_args()
        runtime_args.use_procedural_static_scene = cls._kit_visualizer_requested()

        cls.dispatch_event(PhysicsEvent.MODEL_INIT)
        preloaded_vbd_scene = None
        if not runtime_args.use_procedural_static_scene:
            preloaded_vbd_scene = cls._preload_vbd_static_scene(runtime_args)
        from .builder import create_simulation  # noqa: PLC0415

        cls._viewer = cls._make_viewer()
        if runtime_args.use_procedural_static_scene:
            cls._runtime = create_simulation(cls._viewer, runtime_args, preloaded_vbd_scene=preloaded_vbd_scene)
        else:
            import warp as wp  # noqa: PLC0415

            with wp.ScopedDevice(PhysicsManager._device):
                cls._runtime = create_simulation(cls._viewer, runtime_args, preloaded_vbd_scene=preloaded_vbd_scene)
        cls._publish_display_state()
        PhysicsManager._sim_time = cls.get_sim_time()
        cls.dispatch_event(PhysicsEvent.PHYSICS_READY)

    @classmethod
    def _close_runtime(cls) -> None:
        viewer = cls._viewer
        if viewer is not None and hasattr(viewer, "close"):
            viewer.close()
        cls._runtime = None
        cls._viewer = None

    @classmethod
    def _ensure_newton_on_path(cls) -> None:
        cfg = cls._waterhose_cfg()
        root = Path(getattr(cfg, "newton_root", "/home/maximiliank/Work/newton")).expanduser().resolve()
        if (root / "newton").is_dir():
            root_s = str(root)
            if root_s not in sys.path:
                sys.path.insert(0, root_s)
        os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

    @classmethod
    def _make_viewer(cls):
        import newton.viewer  # noqa: PLC0415

        cfg = cls._waterhose_cfg()
        return newton.viewer.ViewerNull(num_frames=int(getattr(cfg, "num_frames", 100000)))

    @classmethod
    def _make_runtime_args(cls) -> Namespace:
        cfg = cls._waterhose_cfg()
        return Namespace(
            device=PhysicsManager._device,
            viewer="null",
            rerun_address=None,
            output_path=str(getattr(cfg, "output_path", "waterhose_robot_demo_output.usd")),
            num_frames=int(getattr(cfg, "num_frames", 100000)),
            headless=True,
            test=False,
            quiet=bool(getattr(cfg, "quiet", True)),
            benchmark=False,
            warp_config=[],
            realtime=False,
            primary_view="mujoco",
            print_cable_poses=False,
            cable_pose_settle_seconds=None,
            print_robot_poses=False,
            broad_phase=str(getattr(cfg, "broad_phase", "explicit")),
            asset_root=str(Path(getattr(cfg, "asset_root", _DEFAULT_ASSET_ROOT)).expanduser()),
            use_procedural_static_scene=False,
        )

    @classmethod
    def _waterhose_cfg(cls):
        cfg = PhysicsManager._cfg
        return getattr(cfg, "solver_cfg", cfg)

    @classmethod
    def _preload_vbd_static_scene(cls, args):
        import newton  # noqa: PLC0415
        import warp as wp  # noqa: PLC0415

        asset_root = Path(getattr(args, "asset_root", _DEFAULT_ASSET_ROOT)).expanduser().resolve()
        scene_usd_path = asset_root / "Waterhose" / "Cable008" / "Cable008_Body.usda"

        builder = newton.ModelBuilder()
        builder.rigid_contact_margin = 0.0
        builder.rigid_gap = 0.001
        builder.default_shape_cfg.density = 1000.0
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 0.0
        builder.default_shape_cfg.mu = 0.2

        scene_body_ids: list[int] = []
        scene_shape_ids: list[int] = []
        if os.path.isfile(scene_usd_path):
            cls._debug_startup("vbd:preload_scene_usd")
            scene_result = builder.add_usd(
                str(scene_usd_path),
                xform=cls._compute_fridge_xform(wp),
                root_path="/root",
                load_sites=False,
                load_visual_shapes=True,
                hide_collision_shapes=False,
                parse_mujoco_options=False,
                only_load_enabled_joints=True,
                only_load_enabled_rigid_bodies=False,
            )
            scene_body_ids = sorted({int(v) for v in scene_result["path_body_map"].values()})
            for body_id in scene_body_ids:
                builder.body_mass[body_id] = 0.0
                builder.body_inv_mass[body_id] = 0.0
                builder.body_inertia[body_id] = wp.mat33()
                builder.body_inv_inertia[body_id] = wp.mat33()

            scene_shape_ids = sorted(int(v) for v in scene_result["path_shape_map"].values())
            for i in range(len(scene_shape_ids)):
                for j in range(i + 1, len(scene_shape_ids)):
                    builder.add_shape_collision_filter_pair(scene_shape_ids[i], scene_shape_ids[j])
            cls._debug_startup("vbd:preload_scene_usd_done")

        return builder, scene_body_ids, scene_shape_ids

    @staticmethod
    def _compute_fridge_xform(wp):
        table_half_z = 0.5 * (0.6 - 0.215)
        table_z = table_half_z
        table_top_z = table_z + table_half_z
        fridge_z_offset = 0.902 + table_top_z
        fridge_y_offset = (0.293 - 0.395) / 2
        quat = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), wp.pi / 2)
        return wp.transform(wp.vec3(0.95, fridge_y_offset, fridge_z_offset), quat)

    @classmethod
    def _publish_display_state(cls) -> None:
        runtime = cls._runtime
        if runtime is None:
            cls._visualization_model = None
            cls._display_state_0 = None
            cls._display_state_1 = None
            cls._display_control = None
            return

        model, state_0, state_1, control = cls._combined_display(runtime)

        cls._visualization_model = model
        cls._display_state_0 = state_0
        cls._display_state_1 = state_1
        cls._display_control = control
        cls._publish_to_newton_visualizer(model, state_0, state_1, control)

    @classmethod
    def _combined_display(cls, runtime):
        if cls._combined_runtime is not runtime or cls._combined_model is None:
            cls._build_combined_display(runtime)
        cls._update_combined_display_state(runtime)
        return cls._combined_model, cls._combined_state_0, cls._combined_state_1, cls._combined_control

    @classmethod
    def _build_combined_display(cls, runtime) -> None:
        cls._clear_combined_display()
        import newton  # noqa: PLC0415

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_builder(runtime._mujoco_display_builder, label_prefix="mujoco")
        builder.add_builder(runtime._vbd_display_builder, label_prefix="vbd")
        cls._prepare_display_builder(builder)
        assign_display_prim_paths(builder)
        model = builder.finalize(device=PhysicsManager._device)
        model.num_envs = 1

        cls._combined_runtime = runtime
        cls._combined_model = model
        cls._combined_state_0 = model.state()
        cls._combined_state_1 = model.state()
        cls._combined_control = model.control()
        cls._combined_robot_body_count = int(runtime.mujoco_model.body_count)
        cls._combined_robot_shape_count = int(runtime.mujoco_model.shape_count)
        cls._kit_skipped_shape_ids = {
            cls._combined_robot_shape_count + int(shape_id) for shape_id in getattr(runtime, "_scene_shape_ids", [])
        }

    @staticmethod
    def _prepare_display_builder(builder) -> None:
        """Give the combined display model independent mesh ownership."""
        import newton  # noqa: PLC0415

        cloned_sources: dict[int, Any] = {}
        prepared_sources = []
        for source in builder.shape_source:
            if isinstance(source, newton.Mesh):
                source_key = id(source)
                if source_key not in cloned_sources:
                    texture = source.texture
                    if isinstance(texture, np.ndarray):
                        texture = texture.copy()
                    mesh = newton.Mesh(
                        vertices=source.vertices.copy(),
                        indices=source.indices.copy(),
                        normals=source.normals.copy() if source.normals is not None else None,
                        uvs=source.uvs.copy() if source.uvs is not None else None,
                        compute_inertia=False,
                        is_solid=bool(source.is_solid),
                        maxhullvert=source.maxhullvert,
                        color=source.color,
                        roughness=source.roughness,
                        metallic=source.metallic,
                        texture=texture,
                        sdf=None,
                    )
                    mesh.mass = source.mass
                    mesh.com = source.com
                    mesh.inertia = source.inertia
                    mesh.has_inertia = source.has_inertia
                    cloned_sources[source_key] = mesh
                prepared_sources.append(cloned_sources[source_key])
            else:
                prepared_sources.append(source)
        builder.shape_source = prepared_sources

    @classmethod
    def _update_combined_display_state(cls, runtime) -> None:
        if cls._combined_state_0 is None:
            return
        import warp as wp  # noqa: PLC0415

        robot_q = runtime.state_0.body_q.numpy()
        vbd_q = runtime.vbd_state_0.body_q.numpy()
        body_q = np.empty((robot_q.shape[0] + vbd_q.shape[0], 7), dtype=robot_q.dtype)
        body_q[: robot_q.shape[0]] = robot_q
        body_q[robot_q.shape[0] :] = vbd_q
        wp.copy(
            cls._combined_state_0.body_q,
            wp.array(body_q, dtype=wp.transform, device=cls._combined_state_0.body_q.device),
        )

    @classmethod
    def _clear_combined_display(cls) -> None:
        cls._combined_runtime = None
        cls._combined_model = None
        cls._combined_state_0 = None
        cls._combined_state_1 = None
        cls._combined_control = None
        cls._combined_robot_body_count = 0
        cls._combined_robot_shape_count = 0
        cls._kit_skipped_shape_ids = set()
        cls._kit_display_model_id = None
        cls._kit_display_ready = False

    @classmethod
    def _publish_to_newton_visualizer(cls, model, state_0, state_1, control) -> None:
        try:
            from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415
        except Exception:
            return
        NewtonManager._model = model
        NewtonManager._state_0 = state_0
        NewtonManager._state_1 = state_1
        NewtonManager._control = control
        NewtonManager._num_envs = 1
        NewtonManager._scene_data_backend = cls.get_scene_data_backend()
        if cls._kit_visualizer_requested():
            cls._ensure_kit_display(model)
        NewtonManager._mark_transforms_dirty()

    @classmethod
    def _kit_visualizer_requested(cls) -> bool:
        sim = PhysicsManager._sim
        if sim is None:
            return False
        try:
            return "kit" in set(sim.resolve_visualizer_types())
        except Exception:
            return False

    @classmethod
    def _ensure_kit_display(cls, model) -> None:
        if cls._kit_display_model_id != id(model) or not cls._kit_display_ready:
            start = time.perf_counter()
            cls._debug_kit(f"ensure_kit_display:author_start skipped_shapes={len(cls._kit_skipped_shape_ids)}")
            author_display_usd(
                model,
                skipped_shape_ids=cls._kit_skipped_shape_ids,
            )
            cls._debug_kit(f"ensure_kit_display:author_done dt={time.perf_counter() - start:.3f}s")
            cls._kit_display_model_id = id(model)
            cls._kit_display_ready = True
            start = time.perf_counter()
            cls._debug_kit("ensure_kit_display:fabric_start")
            cls._initialize_kit_fabric_sync(model)
            cls._debug_kit(f"ensure_kit_display:fabric_done dt={time.perf_counter() - start:.3f}s")
        elif NewtonManager._usdrt_stage is None:
            cls._initialize_kit_fabric_sync(model)

    @classmethod
    def _initialize_kit_fabric_sync(cls, model) -> None:
        import usdrt  # noqa: PLC0415
        from isaaclab.sim.utils.stage import get_current_stage  # noqa: PLC0415
        from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415

        usdrt_stage = get_current_stage(fabric=True)
        if usdrt_stage is None:
            raise RuntimeError("Cannot initialize waterhose Kit display: no Fabric stage is available.")

        NewtonManager._usdrt_stage = usdrt_stage
        body_paths = list(getattr(model, "body_label", []) or [])
        missing_paths = []
        for body_id, prim_path in enumerate(body_paths):
            prim = usdrt_stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                missing_paths.append(prim_path)
                continue
            prim.CreateAttribute(NewtonManager._newton_index_attr, usdrt.Sdf.ValueTypeNames.UInt, True)
            prim.GetAttribute(NewtonManager._newton_index_attr).Set(body_id)
            prim.AddAppliedSchema("PhysicsRigidBodyAPI")
            xformable = usdrt.Rt.Xformable(prim)
            if not xformable.HasWorldXform():
                xformable.SetWorldXformFromUsd()
        if missing_paths:
            sample = ", ".join(missing_paths[:5])
            raise RuntimeError(f"Waterhose Kit display prims were not found in Fabric stage: {sample}")

        NewtonManager._usd_xform_ops.clear()
        NewtonManager._mark_transforms_dirty()
        NewtonManager.sync_transforms_to_usd()

    @classmethod
    def _clear_kit_display_state(cls) -> None:
        cls._kit_display_model_id = None
        cls._kit_display_ready = False
        try:
            from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415
        except Exception:
            return
        NewtonManager._usdrt_stage = None
        NewtonManager._transforms_dirty = False
        NewtonManager._usd_xform_ops.clear()

    @classmethod
    def _clear_newton_visualizer_state(cls) -> None:
        try:
            from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415
        except Exception:
            return
        NewtonManager._model = None
        NewtonManager._state_0 = None
        NewtonManager._state_1 = None
        NewtonManager._control = None
        NewtonManager._scene_data_backend = None
        cls._clear_kit_display_state()

    @classmethod
    def _require_runtime(cls):
        if cls._runtime is None:
            raise RuntimeError("Waterhose physics runtime is not initialized.")
        return cls._runtime

    @staticmethod
    def _debug_kit(message: str) -> None:
        if os.getenv("WATERHOSE_DEBUG_KIT_DISPLAY", "").lower() in {"1", "true", "yes", "on"}:
            print(f"[waterhose-kit] {message}", flush=True)

    @staticmethod
    def _debug_startup(message: str) -> None:
        if os.getenv("WATERHOSE_DEBUG_STARTUP", "").lower() in {"1", "true", "yes", "on"}:
            print(f"[waterhose-startup] {message}", flush=True)


@configclass
class WaterhoseNewtonSolverCfg(NewtonSolverCfg):
    """Configuration for the waterhose-specific Newton manager."""

    class_type: type[NewtonManager] | str = NewtonWaterhoseManager
    solver_type: str = "waterhose_robot_demo"

    newton_root: str = "/home/maximiliank/Work/newton"
    asset_root: str = _DEFAULT_ASSET_ROOT
    num_frames: int = 100000
    quiet: bool = True
    broad_phase: str = "explicit"
    output_path: str = "waterhose_robot_demo_output.usd"
    max_demo_steps: int = 0
