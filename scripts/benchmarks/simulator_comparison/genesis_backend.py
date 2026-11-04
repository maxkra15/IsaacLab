# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Genesis adapter for the simulator comparison's shared task contract."""

from __future__ import annotations

from typing import Any

import genesis as gs
import quadrants as qd
import torch


class Adapter:
    """Own a batched Genesis scene with explicitly configured physics and RGB cameras."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.device = torch.device(config["device"])
        if self.device.type != "cuda":
            raise ValueError("The comparison requires a CUDA device for both backends.")
        torch.cuda.set_device(self.device)
        gs.init(backend=gs.cuda, seed=config["seed"], precision="32", logging_level="warning", performance_mode=True)
        self._is_cartpole = config["task"] == "cartpole"
        if config["task"] not in ("cartpole", "anymal"):
            raise ValueError(f"Unsupported task: {config['task']}")
        self._joint_names = list(config["joint_names"])
        has_camera = config["vision"] == "rgb"
        camera_follows_root = has_camera and config.get("camera_motion", "fixed") == "root_translation"
        renderer = gs.renderers.BatchRenderer(use_rasterizer=True) if has_camera else gs.renderers.Rasterizer()
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=config["dt"], substeps=1, gravity=(0.0, 0.0, -9.81)),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                iterations=config["solver_iterations"],
                enable_collision=True,
                enable_self_collision=False,
                enable_joint_limit=True,
                use_hibernation=False,
            ),
            renderer=renderer,
            show_viewer=False,
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        )
        floor = self.scene.add_entity(gs.morphs.Plane())
        root_height = 0.0 if self._is_cartpole else config.get("initial_height", 0.55)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=config["asset_path"],
                pos=(0.0, 0.0, root_height),
                fixed=self._is_cartpole,
                merge_fixed_links=True,
                recompute_inertia=False,
                default_armature=0.0,
                decimate=False,
            )
        )
        self._camera = None
        if has_camera:
            self._camera = self.scene.add_camera(
                res=(config["width"], config["height"]),
                pos=tuple(config.get("camera_pos", (2.0, -2.0, 1.5))),
                lookat=tuple(config.get("camera_lookat", (0.0, 0.0, 0.5))),
                fov=config.get("camera_fov", 60.0),
                near=0.01,
                far=20.0,
                GUI=False,
            )
            self.scene.add_light(
                pos=(0.0, 0.0, 5.0),
                dir=(0.0, 0.0, -1.0),
                color=(1.0, 1.0, 1.0),
                intensity=1.0,
                directional=True,
            )
        self.scene.build(n_envs=config["num_envs"])
        if camera_follows_root:
            self._camera.follow_entity(self.robot, fix_orientation=True)
        self._joint_indices = [self.robot.get_joint(name).dofs_idx_local[0] for name in self._joint_names]
        self._action_indices = self._joint_indices[:1] if self._is_cartpole else self._joint_indices
        zeros = torch.zeros(len(self._joint_indices), device=self.device)
        self.robot.set_dofs_armature(zeros, self._joint_indices)
        self.robot.set_dofs_damping(zeros, self._joint_indices)
        self.robot.set_dofs_stiffness(zeros, self._joint_indices)
        self.robot.set_dofs_frictionloss(zeros, self._joint_indices)
        floor.set_friction(config.get("friction", 0.8))
        self.robot.set_friction(config.get("friction", 0.8))
        if not self._is_cartpole:
            self.robot.set_dofs_kp(torch.full((12,), config.get("kp", 80.0), device=self.device), self._joint_indices)
            self.robot.set_dofs_kv(torch.full((12,), config.get("kd", 2.0), device=self.device), self._joint_indices)
            effort = torch.full((12,), config.get("effort_limit", 80.0), device=self.device)
            self.robot.set_dofs_force_range(-effort, effort, self._joint_indices)
        self._initial_joint_pos = torch.tensor(
            config["initial_joint_pos"],
            device=self.device,
            dtype=torch.float32,
        ).expand(config["num_envs"], -1)
        self.robot.set_dofs_position(self._initial_joint_pos, self._joint_indices, zero_velocity=True)
        self._initial_qpos = self.robot.get_qpos().clone()
        self._zero_force = torch.zeros((config["num_envs"], 1), device=self.device)
        self.metadata = {
            "physics": "Genesis rigid solver",
            "constraint_solver": "Newton",
            "integrator": "approximate_implicitfast",
            "solver_iterations": config["solver_iterations"],
            "substeps": 1,
            "precision": "float32",
            "performance_mode": True,
            "self_collision": False,
            "joint_limits": True,
            "fixed_links_merged": True,
            "inertia_recomputed": False,
            "collision_meshes": "convex hulls without mesh decimation; engine-native primitive cooking",
            "armature": 0.0,
            "joint_damping": 0.0,
            "joint_friction": 0.0,
            "friction": config.get("friction", 0.8),
            "friction_combine": "maximum",
            "renderer": "gs-madrona rasterizer" if has_camera else "disabled",
            "camera_fov_axis": "vertical",
            "camera_near_m": 0.01,
            "camera_far_m": 20.0,
            "camera_motion": ("root_translation" if camera_follows_root else "fixed") if has_camera else None,
            "camera_orientation": "world_fixed" if has_camera else None,
            "camera_pose_update": (
                "Native follow_entity copies CUDA root positions to CPU for NumPy pose updates"
                if camera_follows_root
                else None
            ),
            "rendering_worlds": "isolated per-environment worlds" if has_camera else None,
            "controller": "force" if self._is_cartpole else "native position PD",
            "joint_names": self._joint_names,
            "link_names": [link.name for link in self.robot.links],
            "link_masses_kg": self.robot.get_links_mass().tolist(),
            "moving_mass_kg": float(self.robot.get_mass()),
            "num_links": self.robot.n_links,
            "num_dofs": self.robot.n_dofs,
        }
        self.reset()

    @property
    def joint_names(self) -> list[str]:
        """Names in the common task's observation and action order."""
        return self._joint_names

    def reset(self, env_ids: torch.Tensor | None = None, joint_pos: torch.Tensor | None = None) -> None:
        """Restore selected environments to their initial pose and zero velocity.

        Args:
            env_ids: Environments to restore; ``None`` selects all environments.
            joint_pos: Optional replacement joint coordinates [m or rad, depending on joint type],
                shape [selected environments, joints], in :attr:`joint_names` order.
        """
        qpos = self._initial_qpos if env_ids is None else self._initial_qpos[env_ids]
        self.robot.set_qpos(qpos, envs_idx=env_ids, zero_velocity=True)
        if joint_pos is not None:
            self.robot.set_dofs_position(joint_pos, self._joint_indices, env_ids, zero_velocity=True)
        if self._is_cartpole:
            forces = self._zero_force if env_ids is None else self._zero_force[env_ids]
            self.robot.control_dofs_force(forces, self._action_indices, env_ids)
        else:
            targets = self._initial_joint_pos if env_ids is None else self._initial_joint_pos[env_ids]
            self.robot.control_dofs_position(targets if joint_pos is None else joint_pos, self._action_indices, env_ids)

    def state(self) -> dict[str, torch.Tensor]:
        """Return ordered joints and world-frame root state as CUDA tensors.

        Positions use [m or rad, depending on joint type], velocities use [m/s or rad/s].
        Root quantities use [m], [m/s], [rad/s], and quaternion order (w, x, y, z).
        """
        return {
            "joint_pos": self.robot.get_dofs_position(self._joint_indices),
            "joint_vel": self.robot.get_dofs_velocity(self._joint_indices),
            "root_pos": self.robot.get_pos(),
            "root_quat": self.robot.get_quat(),
            "root_lin_vel": self.robot.get_vel(),
            "root_ang_vel": self.robot.get_ang(),
        }

    def apply_action(self, actions: torch.Tensor) -> None:
        """Set cart force [N] or ANYmal joint position targets [rad] for the next physics steps."""
        if self._is_cartpole:
            self.robot.control_dofs_force(actions, self._action_indices)
        else:
            self.robot.control_dofs_position(actions, self._action_indices)

    def physics_step(self) -> None:
        """Advance exactly one configured physics timestep [s] without rendering."""
        self.scene.step(update_visualizer=False)

    def physics_only_step(self) -> None:
        """Advance one timestep [s]; Genesis updates its state during the same solver call."""
        self.scene.step(update_visualizer=False)

    def refresh_state(self) -> None:
        """Complete the shared split-step contract; Genesis state is already current."""

    def render(self) -> torch.Tensor:
        """Render every environment as CUDA uint8 RGB, shape [environments, height, width, 3]."""
        if self._camera is None:
            raise RuntimeError("RGB observations were not enabled for this run.")
        rgb, _, _, _ = self._camera.render(rgb=True, force_render=True)
        expected = (self.config["num_envs"], self.config["height"], self.config["width"], 3)
        if not isinstance(rgb, torch.Tensor) or rgb.shape != expected or rgb.dtype != torch.uint8 or not rgb.is_cuda:
            raise RuntimeError(f"Genesis did not return CUDA uint8 RGB with shape {expected}.")
        return rgb

    def synchronize(self) -> None:
        """Wait for Genesis and Torch GPU work before taking a wall-clock timestamp."""
        qd.sync()
        torch.cuda.synchronize(self.device)

    def close(self) -> None:
        """Release scene resources and the Genesis runtime."""
        gs.destroy()
