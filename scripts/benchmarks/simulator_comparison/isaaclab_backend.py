# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab Newton/MuJoCo Warp adapter for the simulator comparison.

This module is imported only inside a benchmark subprocess, so runtime imports
and kitless scene construction are included in the startup measurement.
"""

from __future__ import annotations

import math
from importlib import import_module
from pathlib import Path
from typing import Any

import torch


class Adapter:
    """Own a kitless MJWarp scene with a shared URDF and optional GPU RGB cameras."""

    def __init__(self, config: dict[str, Any]):
        """Build the scene using the resolved benchmark configuration."""
        # Register importer schemas before USD caches definitions for the first scene.
        import_module("isaacsim.asset.importer.urdf")

        from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
        from isaaclab_newton.renderers import NewtonWarpRendererCfg
        from isaaclab_newton.sim.schemas import NewtonArticulationCfg

        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.app import launch_simulation
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sensors import CameraCfg
        from isaaclab.utils import instantiate

        self.config = config
        self.device = config["device"]
        self.dt = float(config["dt"])
        self.num_envs = int(config["num_envs"])
        self.joint_names = list(config["joint_names"])
        self._cartpole = config["task"] == "cartpole"
        self._camera = None
        self._camera_motion = config.get("camera_motion", "fixed")
        if self._camera_motion not in ("fixed", "root_translation"):
            raise ValueError(f"Unsupported camera motion: {self._camera_motion}")
        settings = config.get("solver_settings", {}).get("isaaclab", {})
        solver_cfg = MJWarpSolverCfg(
            solver="newton",
            integrator=str(settings.get("integrator", "implicitfast")),
            iterations=int(settings.get("iterations", 100)),
            ls_iterations=int(settings.get("ls_iterations", 50)),
            tolerance=float(settings.get("tolerance", 1e-6)),
            cone=str(settings.get("cone", "pyramidal")),
            impratio=float(settings.get("impratio", 1.0)),
            use_mujoco_contacts=bool(settings.get("use_mujoco_contacts", True)),
            njmax=int(settings.get("njmax", 300)),
            nconmax=settings.get("nconmax"),
            use_mujoco_cpu=False,
        )
        material = sim_utils.RigidBodyMaterialBaseCfg(
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.0,
        )
        sim_cfg = sim_utils.SimulationCfg(
            device=self.device,
            dt=self.dt,
            gravity=(0.0, 0.0, -9.81),
            physics=NewtonCfg(
                solver_cfg=solver_cfg,
                num_substeps=1,
                debug_mode=False,
                use_cuda_graph=True,
            ),
            physics_material=material,
            visualizer_cfgs=[],
            use_fabric=True,
            use_newton_actuators=True,
        )
        scene_cfg = InteractiveSceneCfg(
            num_envs=self.num_envs,
            env_spacing=float(config.get("env_spacing", 5.0)),
            lazy_sensor_update=True,
        )
        ground_size = (
            max(100.0, math.ceil(math.sqrt(self.num_envs)) * scene_cfg.env_spacing + 100.0)
            if config["vision"] == "rgb"
            else 100.0
        )
        scene_cfg.ground = AssetBaseCfg(
            prim_path="/World/Ground",
            spawn=sim_utils.GroundPlaneCfg(size=(ground_size, ground_size), physics_material=material),
        )
        initial_positions = dict(zip(self.joint_names, config["initial_joint_pos"], strict=True))
        asset_path = Path(config["asset_path"]).resolve()
        usd_dir = Path(config.get("asset_cache_dir", asset_path.parent / "isaaclab_mjwarp_usd"))
        scene_cfg.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            joint_ordering=self.joint_names,
            spawn=sim_utils.UrdfFileCfg(
                asset_path=str(asset_path),
                usd_dir=str(usd_dir),
                fix_base=self._cartpole,
                merge_fixed_joints=True,
                self_collision=False,
                collision_type="Convex Hull",
                run_asset_transformer=True,
                run_multi_physics_conversion=True,
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(target_type="none"),
                rigid_props=[sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True)],
                articulation_props=[NewtonArticulationCfg(self_collision_enabled=False)],
                physics_material=material,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, float(config["initial_height"])),
                joint_pos=initial_positions,
            ),
            actuators={
                "joints": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    joint_effort_limit=1000.0 if self._cartpole else float(config.get("effort_limit", 80.0)),
                    stiffness=0.0 if self._cartpole else 80.0,
                    damping=0.0 if self._cartpole else 2.0,
                    armature=0.0,
                    friction=0.0,
                    dynamic_friction=0.0,
                    viscous_friction=0.0,
                )
            },
        )
        if config["vision"] == "rgb":
            # Match camera geometry and output dimensions. Renderer quality is
            # recorded separately because Newton Warp and Madrona have different lighting.
            vertical_fov = float(config.get("camera_fov", 60.0))
            aperture = 2.0 * 24.0 * math.tan(math.radians(vertical_fov) / 2.0)
            scene_cfg.camera = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Camera",
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    horizontal_aperture=aperture * int(config["width"]) / int(config["height"]),
                    vertical_aperture=aperture,
                    clipping_range=(0.01, 20.0),
                ),
                width=int(config["width"]),
                height=int(config["height"]),
                data_types=["rgb"],
                renderer_cfg=NewtonWarpRendererCfg(
                    use_cuda_graph=True,
                    enable_textures=True,
                    enable_shadows=False,
                    enable_ambient_lighting=True,
                    create_default_light=True,
                ),
            )
        self._runtime = launch_simulation(
            {"sim": sim_cfg, "scene": scene_cfg},
            {"headless": True, "device": self.device},
        )
        self._runtime.__enter__()
        self._sim = sim_utils.SimulationContext(sim_cfg)
        self._scene = instantiate(scene_cfg)
        self._robot = self._scene["robot"]
        self._sim.reset()
        self._origins = self._scene.env_origins
        self._all_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._initial_joint_pos = torch.tensor(
            config["initial_joint_pos"], device=self.device, dtype=torch.float32
        ).repeat(self.num_envs, 1)
        self._efforts = torch.zeros_like(self._initial_joint_pos)
        if config["vision"] == "rgb":
            self._camera = self._scene["camera"]
            eye = torch.tensor(config.get("camera_pos", (2.0, -2.0, 1.5)), device=self.device, dtype=torch.float32)
            target = torch.tensor(config.get("camera_lookat", (0.0, 0.0, 0.5)), device=self.device, dtype=torch.float32)
            initial_root = torch.tensor(
                (0.0, 0.0, float(config["initial_height"])), device=self.device, dtype=torch.float32
            )
            self._camera_eye_offset = eye - initial_root
            self._camera_target_offset = target - initial_root
            self._camera.set_world_poses_from_view(self._origins + eye, self._origins + target)
        link_masses = self._robot.data.body_mass.torch[0].tolist()
        link_names = list(self._robot.body_names)
        moving_mass = sum(
            mass for name, mass in zip(link_names, link_masses, strict=True) if not self._cartpole or name != "base"
        )
        backend = self._sim.backend_manager.backend
        solver = backend.solver
        options = solver.mj_model.opt
        self.metadata = {
            "physics_backend": "Isaac Lab Newton / MuJoCo Warp",
            "solver": solver_cfg.solver,
            "iterations": solver_cfg.iterations,
            "ls_iterations": solver_cfg.ls_iterations,
            "tolerance": solver_cfg.tolerance,
            "integrator": solver_cfg.integrator,
            "cone": solver_cfg.cone,
            "impratio": solver_cfg.impratio,
            "use_mujoco_contacts": solver_cfg.use_mujoco_contacts,
            "njmax": solver_cfg.njmax,
            "nconmax": solver_cfg.nconmax,
            "actual_solver": {
                "class": f"{type(solver).__module__}.{type(solver).__qualname__}",
                "iterations": int(options.iterations),
                "ls_iterations": int(options.ls_iterations),
                "tolerance": float(options.tolerance),
                "solver_enum": int(options.solver),
                "integrator_enum": int(options.integrator),
                "cone_enum": int(options.cone),
                "impratio": float(options.impratio),
                "njmax": int(solver.mjw_data.njmax),
                "contact_buffer_capacity_all_worlds": int(solver.get_max_contact_count()),
            },
            "solver_profile": config.get("solver_profile", "shared"),
            "substeps": 1,
            "cuda_graph": True,
            "cuda_graph_available": backend.capture is not None,
            "cuda_graph_capture": "After the first eager physics step; commands update graph-bound arrays in place",
            "kit_runtime": False,
            "self_collision": False,
            "friction": 0.8,
            "friction_combine": "MuJoCo material mixing; equal 0.8 coefficients on floor and robot",
            "restitution": 0.0,
            "joint_order": self.joint_names,
            "physics_replication": "Newton native world replication",
            "link_names": link_names,
            "link_masses_kg": link_masses,
            "moving_mass_kg": moving_mass,
            "renderer": "Newton Warp batched Camera" if self._camera is not None else None,
            "single_gpu": True,
            "render_quality": "Native textured ray tracing; ambient/default light; no shadows"
            if self._camera is not None
            else None,
            "camera_cuda_graph": True if self._camera is not None else None,
            "camera_motion": self._camera_motion if self._camera is not None else None,
            "camera_pose_update": "Public batched set_world_poses_from_view with GPU torch/Warp tensors",
            "camera_pose_adapter_host_transfers": "No explicit GPU-to-CPU pose downloads",
            "camera_clipping_m": [0.01, 20.0] if self._camera is not None else None,
            "camera_near_clip_enforced": False if self._camera is not None else None,
            "env_spacing_m": float(config.get("env_spacing", 5.0)),
            "visual_ground_size_m": [ground_size, ground_size],
            "pd": None if self._cartpole else {"kp": 80.0, "kd": 2.0, "effort_limit_nm": 80.0},
            "pd_integration": "MuJoCo implicitfast native joint drives",
            "model_conversion": "Common URDF; fixed links merged; convex hull collisions; multi-physics USD conversion",
            "cache_dir": str(usd_dir),
            "physics_only_scope": "SimulationContext.step(render=False); no command flush or asset update",
        }
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None, joint_pos: torch.Tensor | None = None) -> None:
        """Reset selected roots and joints; joint coordinates are [m or rad]."""
        ids = self._all_ids if env_ids is None else env_ids
        if ids.numel() == 0:
            return
        root_pose = self._robot.data.default_root_pose.torch[ids].clone()
        root_pose[:, :3] += self._origins[ids]
        self._robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=ids)
        self._robot.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((ids.numel(), 6), device=self.device), env_ids=ids
        )
        positions = self._initial_joint_pos[ids] if joint_pos is None else joint_pos
        self._robot.write_joint_position_to_sim_index(position=positions, env_ids=ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(positions), env_ids=ids)
        self._robot.reset(env_ids=ids)
        if self._cartpole:
            self._efforts[ids] = 0.0
            self._robot.actuators.target_command.set_effort_index(value=self._efforts[ids], env_ids=ids)
        else:
            self._robot.actuators.target_command.set_position_index(value=positions, env_ids=ids)
        self._robot.write_data_to_sim()

    def state(self) -> dict[str, torch.Tensor]:
        """Return canonical joint and root state in SI units and wxyz quaternions."""
        data = self._robot.data
        quat_xyzw = data.root_quat_w.torch
        return {
            "joint_pos": data.joint_pos.torch,
            "joint_vel": data.joint_vel.torch,
            "root_pos": data.root_pos_w.torch - self._origins,
            "root_quat": quat_xyzw[:, [3, 0, 1, 2]],
            "root_lin_vel": data.root_link_lin_vel_w.torch,
            "root_ang_vel": data.root_link_ang_vel_w.torch,
        }

    def apply_action(self, actions: torch.Tensor) -> None:
        """Set cart force [N] or ANYmal joint position targets [rad]."""
        if self._cartpole:
            self._efforts.zero_()
            self._efforts[:, 0] = actions[:, 0]
            self._robot.actuators.target_command.set_effort_index(value=self._efforts)
        else:
            self._robot.actuators.target_command.set_position_index(value=actions)
        self._robot.write_data_to_sim()

    def physics_step(self) -> None:
        """Advance one physics timestep [s], including actuator flush and state invalidation."""
        self._robot.write_data_to_sim()
        self._sim.step(render=False)
        self._robot.update(self.dt)

    def physics_only_step(self) -> None:
        """Advance one physics timestep [s] without command or state processing."""
        self._sim.step(render=False)

    def refresh_state(self) -> None:
        """Invalidate cached articulation state after a pure simulation window."""
        self._robot.update(self.dt)

    def render(self) -> torch.Tensor:
        """Return current batched RGB images as a GPU tensor [N, H, W, 3]."""
        if self._camera is None:
            raise RuntimeError("RGB cameras were not enabled for this run.")
        if self._camera_motion == "root_translation":
            root_pos = self._robot.data.root_pos_w.torch
            self._camera.set_world_poses_from_view(
                root_pos + self._camera_eye_offset, root_pos + self._camera_target_offset
            )
        self._sim.render()
        self._camera.update(self.dt * int(self.config["decimation"]), force_recompute=True)
        return self._camera.data.output["rgb"].torch[..., :3]

    def synchronize(self) -> None:
        """Wait for all CUDA work on the configured device."""
        torch.cuda.synchronize(self.device)

    def diagnostics(self) -> dict[str, Any]:
        """Read final solver diagnostics outside benchmark timing windows.

        Overflow flags describe the last solver state, rather than cumulative
        overflow history. Downloading them may synchronize CUDA.
        """
        from mujoco_warp import OverflowType

        backend = self._sim.backend_manager.backend
        solver = backend.solver
        flags = solver.mjw_data.overflow.numpy().tolist()
        convergence_mask = int(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS)
        capacity_mask = int(OverflowType.ALL) & ~convergence_mask
        unknown_flags = [flag & ~int(OverflowType.ALL) for flag in flags]
        return {
            "graph_active": backend.graph is not None,
            "overflow_flags_at_end": flags,
            "fatal_capacity_flags_at_end": [
                (flag & capacity_mask) | unknown for flag, unknown in zip(flags, unknown_flags, strict=True)
            ],
            "convergence_limit_flags_at_end": [flag & convergence_mask for flag in flags],
            "unknown_overflow_flags_at_end": unknown_flags,
            "overflow_flags_decoded_at_end": [
                [entry.name for entry in OverflowType if flag & int(entry)] for flag in flags
            ],
            "overflow_flag_values": {flag.name: int(flag) for flag in OverflowType},
            "overflow_scope": "Last solver state only; not a cumulative history",
            "constraint_capacity_per_world": int(solver.mjw_data.njmax),
            "contact_capacity_all_worlds": int(solver.get_max_contact_count()),
        }

    def close(self) -> None:
        """Release the scene, Newton context, and renderer."""
        self._runtime.__exit__(None, None, None)
