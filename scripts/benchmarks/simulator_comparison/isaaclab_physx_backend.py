# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab / Isaac Sim PhysX adapter for the simulator comparison.

This module is imported only inside a benchmark subprocess, so runtime imports
and Kit startup are included in the subprocess's startup measurement.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch


class Adapter:
    """Own a headless PhysX scene with a shared URDF and optional batched RGB cameras."""

    def __init__(self, config: dict[str, Any]):
        """Build the scene using the resolved benchmark configuration."""
        from isaaclab_physx.physics import PhysxCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg
        from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg
        from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

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
        solver_settings = config.get("solver_settings", {})
        settings = solver_settings.get("isaaclab_physx", solver_settings.get("isaaclab", {}))
        iterations = int(settings.get("position_iterations", config.get("solver_iterations", 50)))
        velocity_iterations = int(settings.get("velocity_iterations", 1))
        material = PhysxRigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.0,
            friction_combine_mode="max",
            restitution_combine_mode="max",
        )
        sim_cfg = sim_utils.SimulationCfg(
            device=self.device,
            dt=self.dt,
            gravity=(0.0, 0.0, -9.81),
            physics=PhysxCfg(
                solver_type=1,
                min_position_iteration_count=iterations,
                max_position_iteration_count=iterations,
                min_velocity_iteration_count=velocity_iterations,
                max_velocity_iteration_count=velocity_iterations,
            ),
            physics_material=material,
            visualizer_cfgs=[],
            use_fabric=True,
            use_newton_actuators=False,
        )
        scene_cfg = InteractiveSceneCfg(
            num_envs=self.num_envs,
            env_spacing=float(config.get("env_spacing", 5.0)),
            lazy_sensor_update=True,
        )
        scene_cfg.ground = AssetBaseCfg(
            prim_path="/World/Ground",
            spawn=sim_utils.GroundPlaneCfg(physics_material=material),
        )
        scene_cfg.light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=1000.0),
        )
        initial_positions = dict(zip(self.joint_names, config["initial_joint_pos"], strict=True))
        asset_path = Path(config["asset_path"]).resolve()
        usd_dir = Path(config.get("asset_cache_dir", asset_path.parent / "isaaclab_usd"))
        scene_cfg.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            # PhysX's native replicator does not translate a world-fixed joint's
            # anchor. Parsing the USD clones preserves each fixed base's frame.
            cloning_contexts=() if self._cartpole else None,
            joint_ordering=self.joint_names,
            spawn=sim_utils.UrdfFileCfg(
                asset_path=str(asset_path),
                usd_dir=str(usd_dir),
                fix_base=self._cartpole,
                merge_fixed_joints=True,
                self_collision=False,
                collision_type="Convex Hull",
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(target_type="none"),
                rigid_props=[
                    sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
                    PhysxRigidBodyCfg(enable_gyroscopic_forces=True, max_depenetration_velocity=10.0),
                ],
                articulation_props=[
                    PhysxArticulationCfg(
                        enabled_self_collisions=False,
                        solver_position_iteration_count=iterations,
                        solver_velocity_iteration_count=velocity_iterations,
                        sleep_threshold=0.0,
                        stabilization_threshold=0.0,
                    )
                ],
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
            # recorded separately because RTX and Madrona have different lighting.
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
                renderer_cfg=IsaacRtxRendererCfg(
                    enable_scene_partitioning=True,
                    global_settings=IsaacRtxRendererGlobalSettingsCfg(
                        enable_reflections=False,
                        enable_global_illumination=False,
                        enable_translucency=False,
                        antialiasing_mode="Off",
                        samples_per_pixel=1,
                        enable_shadows=False,
                    ),
                ),
            )
        self._runtime = launch_simulation(
            {"sim": sim_cfg, "scene": scene_cfg},
            {"headless": True, "device": self.device, "enable_cameras": config["vision"] == "rgb", "multi_gpu": False},
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
        self.metadata = {
            "physics_backend": "Isaac Sim PhysX GPU",
            "solver": "TGS",
            "position_iterations": iterations,
            "velocity_iterations": velocity_iterations,
            "solver_profile": config.get("solver_profile", "fixed_cap"),
            "substeps": 1,
            "self_collision": False,
            "friction": 0.8,
            "friction_combine": "max",
            "restitution": 0.0,
            "joint_order": self.joint_names,
            "physics_replication": "USD clones" if self._cartpole else "native PhysX replication",
            "link_names": link_names,
            "link_masses_kg": link_masses,
            "moving_mass_kg": moving_mass,
            "renderer": "Isaac RTX batched Camera" if self._camera is not None else None,
            "single_gpu": True,
            "scene_partitioning": True if self._camera is not None else None,
            "render_quality": "1 spp; no AA, shadows, reflections or GI" if self._camera is not None else None,
            "camera_motion": self._camera_motion if self._camera is not None else None,
            "camera_pose_update": "Public batched set_world_poses_from_view with GPU torch/Warp tensors",
            "camera_pose_adapter_host_transfers": "No explicit GPU-to-CPU pose downloads",
            "camera_clipping_m": [0.01, 20.0] if self._camera is not None else None,
            "env_spacing_m": float(config.get("env_spacing", 5.0)),
            "pd": None if self._cartpole else {"kp": 80.0, "kd": 2.0, "effort_limit_nm": 80.0},
            "pd_integration": "PhysX implicit drives",
            "model_conversion": "Common URDF; fixed links merged; mesh collisions cooked as convex hulls",
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

    def close(self) -> None:
        """Release the scene, PhysX context, renderer, and Kit runtime."""
        self._runtime.__exit__(None, None, None)
