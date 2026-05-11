# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
import newton
from newton import JointTargetMode
from newton.solvers import SolverImplicitMPM, SolverMuJoCo

from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab_newton.physics import (
    NewtonManager,
)

from .ur10_particle_scoop_env_cfg import UR10ParticleScoopEnvCfg


class PolicyObservationSpheres:
    """Render the policy's compact particle observations as Newton sphere markers."""

    def __init__(self):
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/PolicyObservations",
            markers={
                "height_cell": sim_utils.SphereCfg(
                    radius=0.012,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 1.0)),
                ),
                "centroid": sim_utils.SphereCfg(
                    radius=0.035,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0)),
                ),
                "paddle": sim_utils.SphereCfg(
                    radius=0.025,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 1.0)),
                ),
                "bin": sim_utils.SphereCfg(
                    radius=0.025,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.25)),
                ),
            },
        )
        self._markers = VisualizationMarkers(marker_cfg)
        self._markers.set_visibility(False)
        self._enabled = False

    def set_enabled(self, enabled: bool) -> None:
        if enabled == self._enabled:
            return
        self._enabled = enabled
        self._markers.set_visibility(enabled)

    def update(self, env: "UR10ParticleScoopEnv", heightmap: torch.Tensor, paddle_pos: torch.Tensor, centroid: torch.Tensor):
        self.set_enabled(_show_policy_observation_spheres(env))
        if not self._enabled:
            return

        env_id = 0
        origin = env.scene.env_origins[env_id]
        map_size = env.cfg.heightmap_size
        cell_yx = torch.nonzero(heightmap[env_id] > 1.0e-3, as_tuple=False)

        if cell_yx.numel() > 0:
            cell_y = cell_yx[:, 0].float()
            cell_x = cell_yx[:, 1].float()
            cell_height = heightmap[env_id, cell_yx[:, 0], cell_yx[:, 1]]
            cell_pos = torch.stack(
                (
                    env._heightmap_x_min + (cell_x + 0.5) * env._heightmap_x_range / map_size,
                    env._heightmap_y_min + (cell_y + 0.5) * env._heightmap_y_range / map_size,
                    env.cfg.table_top_z + cell_height * env.cfg.heightmap_z_range + 0.02,
                ),
                dim=-1,
            )
            translations = cell_pos + origin
            marker_indices = torch.zeros(translations.shape[0], dtype=torch.int32, device=env.device)
        else:
            translations = torch.empty((0, 3), device=env.device)
            marker_indices = torch.empty((0,), dtype=torch.int32, device=env.device)

        feature_pos = torch.stack(
            (
                centroid[env_id] + origin,
                paddle_pos[env_id] + origin,
                env._bin_target + origin,
            ),
            dim=0,
        )
        translations = torch.cat((translations, feature_pos), dim=0)
        feature_indices = torch.tensor((1, 2, 3), dtype=torch.int32, device=env.device)
        marker_indices = torch.cat((marker_indices, feature_indices), dim=0)
        self._markers.visualize(translations=translations, marker_indices=marker_indices)


def _show_policy_observation_spheres(env: "UR10ParticleScoopEnv") -> bool:
    for visualizer in env.sim.visualizers:
        viewer = getattr(visualizer, "_viewer", None)
        if viewer is not None and getattr(viewer, "show_policy_observations", False):
            return True
    return False


class UR10ParticleScoopEnv(DirectRLEnv):
    """Pure Newton UR10 + MPM particle scooping task."""

    cfg: UR10ParticleScoopEnvCfg

    def __init__(self, cfg: UR10ParticleScoopEnvCfg, render_mode: str | None = None, **kwargs):
        self._joint_q_ids_list: list[list[int]] = []
        self._joint_qd_ids_list: list[list[int]] = []
        self._particle_ids_list: list[list[int]] = []
        self._ee_body_ids_list: list[int] = []
        super().__init__(cfg, render_mode, **kwargs)

        self._joint_q_ids = torch.tensor(self._joint_q_ids_list, device=self.device, dtype=torch.long)
        self._joint_qd_ids = torch.tensor(self._joint_qd_ids_list, device=self.device, dtype=torch.long)
        self._particle_ids = torch.tensor(self._particle_ids_list, device=self.device, dtype=torch.long)
        self._ee_body_ids = torch.tensor(self._ee_body_ids_list, device=self.device, dtype=torch.long)
        self._particle_count = int(self._particle_ids.shape[1])
        self._mpm_solver = self._create_mpm_solver()
        self._mpm_graph = self._capture_mpm_graph()

        state = NewtonManager.get_state_0()
        self._default_joint_q = wp.to_torch(state.joint_q)[self._joint_q_ids].clone()
        self._default_joint_qd = wp.to_torch(state.joint_qd)[self._joint_qd_ids].clone()
        self._default_particle_q = wp.to_torch(state.particle_q)[self._particle_ids].clone()
        self._joint_targets = self._default_joint_q.clone()
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._bin_center = torch.tensor(self.cfg.bin_center, device=self.device)
        self._bin_half_extents = torch.tensor(self.cfg.bin_inner_half_extents, device=self.device)
        self._bin_lower = self._bin_center - self._bin_half_extents
        self._bin_upper = self._bin_center + self._bin_half_extents
        self._bin_lower[2] = self.cfg.table_top_z - 0.5 * self.cfg.voxel_size
        self._bin_upper[2] = self.cfg.table_top_z + self.cfg.bin_wall_height + 0.5 * self.cfg.voxel_size
        self._bin_target = self._bin_center.clone()
        self._bin_target[2] = self.cfg.table_top_z + 0.5 * self.cfg.bin_wall_height
        self._workspace_lower = torch.tensor(
            (
                self.cfg.table_center[0] - 0.5 * self.cfg.table_size[0] - 0.10,
                self.cfg.table_center[1] - 0.5 * self.cfg.table_size[1] - 0.15,
                self.cfg.table_top_z - 0.10,
            ),
            device=self.device,
        )
        self._workspace_upper = torch.tensor(
            (
                self._bin_upper[0].item() + 0.15,
                self.cfg.table_center[1] + 0.5 * self.cfg.table_size[1] + 0.15,
                self.cfg.table_top_z + self.cfg.heightmap_z_range,
            ),
            device=self.device,
        )
        self._pile_center = 0.5 * (
            torch.tensor(self.cfg.pile_lo, device=self.device) + torch.tensor(self.cfg.pile_hi, device=self.device)
        )
        self._progress_start_x = float(self.cfg.pile_lo[0])
        self._progress_target_x = float(self.cfg.bin_center[0] - self.cfg.bin_inner_half_extents[0])
        self._heightmap_x_min = 0.05
        self._heightmap_x_range = max(1.10, float(self._bin_upper[0].item()) - self._heightmap_x_min + 0.05)
        self._heightmap_y_min = -0.45
        self._heightmap_y_range = 0.90
        self._heightmap_env_offsets = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            * self.cfg.heightmap_size
            * self.cfg.heightmap_size
        ).unsqueeze(1)
        self._previous_bin_count = torch.zeros(self.num_envs, device=self.device)
        self._previous_particle_progress = torch.zeros(self.num_envs, device=self.device)
        self._previous_bin_proximity = torch.zeros(self.num_envs, device=self.device)
        self._previous_paddle_pos = self._paddle_pos_e().clone()
        self._episode_sums = {
            "particle_count": torch.zeros(self.num_envs, device=self.device),
            "delta_count": torch.zeros(self.num_envs, device=self.device),
            "particle_progress": torch.zeros(self.num_envs, device=self.device),
            "bin_proximity": torch.zeros(self.num_envs, device=self.device),
            "delta_bin_proximity": torch.zeros(self.num_envs, device=self.device),
            "spill_penalty": torch.zeros(self.num_envs, device=self.device),
            "paddle_proximity": torch.zeros(self.num_envs, device=self.device),
            "paddle_speed_penalty": torch.zeros(self.num_envs, device=self.device),
            "action_penalty": torch.zeros(self.num_envs, device=self.device),
        }
        self._policy_observation_spheres = PolicyObservationSpheres()
        self._configure_newton_viewer()

    def step(self, action: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        action = action.to(self.device)
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        self._pre_physics_step(action)
        is_rendering = self.sim.is_rendering

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self._step_mpm()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render(skip_app_pumping=not self.render_enabled)
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            if self.render_enabled and is_rendering and self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

        if self.cfg.events and "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self._get_observations()
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _create_mpm_solver(self) -> SolverImplicitMPM:
        model = NewtonManager.get_model()
        state = NewtonManager.get_state_0()
        mpm_cfg = SolverImplicitMPM.Config()
        mpm_cfg.voxel_size = self.cfg.voxel_size
        mpm_cfg.grid_type = "fixed"
        mpm_cfg.grid_padding = self.cfg.mpm_grid_padding
        mpm_cfg.max_active_cell_count = self.cfg.mpm_max_active_cell_count
        mpm_cfg.strain_basis = "P0"
        mpm_cfg.transfer_scheme = "pic"
        mpm_cfg.max_iterations = self.cfg.mpm_iterations
        mpm_cfg.critical_fraction = 0.0
        mpm_cfg.air_drag = 1.0
        mpm_cfg.collider_velocity_mode = "backward"
        mpm_cfg.solver = "gauss-seidel"
        mpm_solver = SolverImplicitMPM(model, mpm_cfg)
        mpm_solver.setup_collider(body_mass=wp.zeros_like(model.body_mass), body_q=state.body_q)
        return mpm_solver

    def _capture_mpm_graph(self):
        if not wp.get_device().is_cuda or self._mpm_solver.grid_type != "fixed":
            return None
        with wp.ScopedCapture() as capture:
            self._simulate_mpm_step()
        return capture.graph

    def _step_mpm(self) -> None:
        if self._mpm_graph is not None:
            wp.capture_launch(self._mpm_graph)
        else:
            self._simulate_mpm_step()

    def _simulate_mpm_step(self) -> None:
        state = NewtonManager.get_state_0()
        self._mpm_solver.step(state, state, control=None, contacts=None, dt=self.physics_dt)

    def _setup_scene(self) -> None:
        builder = self._build_newton_model()
        NewtonManager._num_envs = self.scene.num_envs
        NewtonManager.set_builder(builder)

    def _build_newton_model(self) -> newton.ModelBuilder:
        proto, meta = self._build_world_proto()
        builder = NewtonManager.create_builder()
        SolverMuJoCo.register_custom_attributes(builder)
        SolverImplicitMPM.register_custom_attributes(builder)

        for env_id in range(self.scene.num_envs):
            origin = self.scene.env_origins[env_id].detach().cpu().tolist()
            builder.begin_world(label=f"env_{env_id}")
            body_offset = builder.body_count
            particle_offset = builder.particle_count
            joint_q_offset = builder.joint_coord_count
            joint_qd_offset = builder.joint_dof_count

            builder.add_builder(proto, xform=wp.transform(wp.vec3(*origin), wp.quat_identity()))

            particle_range = list(range(particle_offset, particle_offset + meta["particle_count"]))
            self._particle_ids_list.append(particle_range)
            self._joint_q_ids_list.append([joint_q_offset + idx for idx in meta["arm_q_ids"]])
            self._joint_qd_ids_list.append([joint_qd_offset + idx for idx in meta["arm_qd_ids"]])
            self._ee_body_ids_list.append(body_offset + int(meta["ee_body"]))
            builder.end_world()

        return builder

    def _build_world_proto(self) -> tuple[newton.ModelBuilder, dict[str, object]]:
        proto = NewtonManager.create_builder()
        SolverMuJoCo.register_custom_attributes(proto)
        SolverImplicitMPM.register_custom_attributes(proto)
        proto.default_shape_cfg.mu = 0.75
        proto.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.1,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )

        proto.add_urdf(
            self.cfg.ur10_urdf_path,
            xform=wp.transform(wp.vec3(*self.cfg.robot_base_pos), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
            ignore_inertial_definitions=False,
        )
        self._configure_ur10_joints(proto)
        self._disable_robot_particle_collisions(proto)

        ee_body = self._find_body(proto, self.cfg.ee_body_name)
        paddle_shapes = self._add_paddle_pad(proto, ee_body)
        workspace_shapes = self._add_workspace_colliders(proto)
        particle_start = proto.particle_count
        self._add_mpm_pile(proto)
        particle_end = proto.particle_count

        arm_q_ids, arm_qd_ids = self._resolve_arm_joint_ids(proto)
        if len(arm_q_ids) != self.cfg.action_space:
            raise RuntimeError(f"Expected {self.cfg.action_space} UR10 arm DOFs, found {len(arm_q_ids)}.")

        return proto, {
            "body_count": proto.body_count,
            "joint_count": proto.joint_count,
            "shape_count": proto.shape_count,
            "particle_count": particle_end - particle_start,
            "particle_collider_shapes": [*paddle_shapes, *workspace_shapes],
            "arm_q_ids": arm_q_ids,
            "arm_qd_ids": arm_qd_ids,
            "ee_body": ee_body,
            "paddle_body": ee_body,
        }

    def _disable_robot_particle_collisions(self, builder: newton.ModelBuilder) -> None:
        """Keep MPM particle coupling focused on the pan, table, and bin."""
        particle_collision = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape_id in range(builder.shape_count):
            builder.shape_flags[shape_id] &= ~particle_collision

    def _add_paddle_pad(self, builder: newton.ModelBuilder, body_id: int) -> list[int]:
        """Add one flat pad collider, like a tennis paddle face."""
        sx, sy, sz = self.cfg.paddle_size
        cfg = newton.ModelBuilder.ShapeConfig(
            mu=1.0,
            density=500.0,
            margin=self.cfg.paddle_collision_margin,
            gap=0.01,
        )

        return [
            builder.add_shape_box(
                body_id,
                xform=wp.transform(wp.vec3(*self.cfg.paddle_ee_offset), wp.quat_identity()),
                hx=0.5 * sx,
                hy=0.5 * sy,
                hz=0.5 * sz,
                cfg=cfg,
                color=(0.1, 0.25, 0.85),
            )
        ]

    def _configure_ur10_joints(self, builder: newton.ModelBuilder) -> None:
        initial_q = {
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        }
        arm_q_ids, arm_qd_ids = self._resolve_arm_joint_ids(builder)
        for q_id, qd_id, joint_name in zip(arm_q_ids, arm_qd_ids, self.cfg.arm_joint_names):
            builder.joint_q[q_id] = initial_q[joint_name]
            builder.joint_target_pos[qd_id] = initial_q[joint_name]
        for dof_id in range(builder.joint_dof_count):
            builder.joint_target_ke[dof_id] = 800.0
            builder.joint_target_kd[dof_id] = 60.0
            builder.joint_effort_limit[dof_id] = 100.0
            builder.joint_armature[dof_id] = 0.15
            builder.joint_target_mode[dof_id] = int(JointTargetMode.POSITION)

    def _resolve_arm_joint_ids(self, builder: newton.ModelBuilder) -> tuple[list[int], list[int]]:
        q_ids = []
        qd_ids = []
        for joint_name in self.cfg.arm_joint_names:
            matches = [i for i, label in enumerate(builder.joint_label) if label.endswith(joint_name)]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one joint matching {joint_name!r}, found {matches}.")
            joint_id = matches[0]
            q_ids.append(builder.joint_q_start[joint_id])
            qd_ids.append(builder.joint_qd_start[joint_id])
        return q_ids, qd_ids

    def _add_workspace_colliders(self, builder: newton.ModelBuilder) -> list[int]:
        workspace_shapes: list[int] = []
        tx, ty, tz = self.cfg.table_center
        sx, sy, sz = self.cfg.table_size
        table_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, density=0.0, margin=0.01, gap=0.01)
        workspace_shapes.append(
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(tx, ty, tz), wp.quat_identity()),
                hx=0.5 * sx,
                hy=0.5 * sy,
                hz=0.5 * sz,
                cfg=table_cfg,
                color=(0.45, 0.34, 0.24),
            )
        )

        wall_thickness = self.cfg.bin_wall_thickness
        bin_height = self.cfg.bin_wall_height
        bin_x, bin_y, _ = self.cfg.bin_center
        bin_half_x, bin_half_y, _ = self.cfg.bin_inner_half_extents
        wall_z = self.cfg.table_top_z + 0.5 * bin_height
        bottom_z = self.cfg.table_top_z - 0.5 * wall_thickness
        wall_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, density=0.0, margin=0.01, gap=0.01)
        workspace_shapes.append(
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(bin_x, bin_y, bottom_z), wp.quat_identity()),
                hx=bin_half_x + 0.5 * wall_thickness,
                hy=bin_half_y + 0.5 * wall_thickness,
                hz=0.5 * wall_thickness,
                cfg=wall_cfg,
                color=(0.08, 0.14, 0.26),
            )
        )
        walls = [
            ((2.0 * bin_half_x + wall_thickness, wall_thickness, bin_height), (bin_x, bin_y + bin_half_y, wall_z)),
            ((2.0 * bin_half_x + wall_thickness, wall_thickness, bin_height), (bin_x, bin_y - bin_half_y, wall_z)),
            ((wall_thickness, 2.0 * bin_half_y + wall_thickness, bin_height), (bin_x + bin_half_x, bin_y, wall_z)),
        ]
        for size, pos in walls:
            workspace_shapes.append(
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(wp.vec3(*pos), wp.quat_identity()),
                    hx=0.5 * size[0],
                    hy=0.5 * size[1],
                    hz=0.5 * size[2],
                    cfg=wall_cfg,
                    color=(0.1, 0.18, 0.32),
                )
            )
        return workspace_shapes

    def _add_mpm_pile(self, builder: newton.ModelBuilder) -> None:
        lo = np.array(self.cfg.pile_lo, dtype=np.float64)
        hi = np.array(self.cfg.pile_hi, dtype=np.float64)
        res = np.maximum(np.ceil(self.cfg.particles_per_cell * (hi - lo) / self.cfg.voxel_size), 1).astype(int)
        cell_size = (hi - lo) / res
        radius = float(np.max(cell_size) * 0.5)
        mass = float(np.prod(cell_size) * self.cfg.sand_density)
        builder.add_particle_grid(
            pos=wp.vec3(lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=int(res[0]) + 1,
            dim_y=int(res[1]) + 1,
            dim_z=int(res[2]) + 1,
            cell_x=float(cell_size[0]),
            cell_y=float(cell_size[1]),
            cell_z=float(cell_size[2]),
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
            custom_attributes={
                "mpm:friction": self.cfg.sand_friction,
                "mpm:damping": self.cfg.sand_damping,
                "mpm:young_modulus": self.cfg.sand_young_modulus,
                "mpm:yield_pressure": self.cfg.sand_yield_pressure,
                "mpm:tensile_yield_ratio": self.cfg.sand_tensile_yield_ratio,
            },
        )

    @staticmethod
    def _find_body(builder: newton.ModelBuilder, body_name: str) -> int:
        matches = [i for i, label in enumerate(builder.body_label) if label.endswith(body_name)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one body matching {body_name!r}, found {matches}.")
        return matches[0]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._apply_viewer_forces()
        self._actions = actions.clamp(-1.0, 1.0)
        self._joint_targets = self._joint_targets + self._actions * self.cfg.action_scale * self.step_dt
        self._joint_targets = torch.clamp(self._joint_targets, -2.0 * torch.pi, 2.0 * torch.pi)

    def _apply_action(self) -> None:
        control = NewtonManager.get_control()
        wp.to_torch(control.joint_target_pos)[self._joint_qd_ids] = self._joint_targets

    def _get_observations(self) -> dict:
        particle_pos = self._particle_pos_e()
        heightmap_grid = self._particle_heightmap(particle_pos)
        heightmap = heightmap_grid.reshape(self.num_envs, -1)
        state = NewtonManager.get_state_0()
        joint_q = wp.to_torch(state.joint_q)[self._joint_q_ids]
        joint_qd = wp.to_torch(state.joint_qd)[self._joint_qd_ids]
        bin_fraction = self._count_particles_in_bin(particle_pos)[:, None] / float(self._particle_count)
        paddle_pos = self._paddle_pos_e()
        particle_centroid = self._robust_particle_centroid(particle_pos)
        obs = torch.cat(
            (
                heightmap,
                joint_q / torch.pi,
                0.1 * joint_qd,
                paddle_pos,
                particle_centroid,
                self._bin_center.unsqueeze(0) - paddle_pos,
                bin_fraction,
            ),
            dim=-1,
        )
        self._policy_observation_spheres.update(self, heightmap_grid, paddle_pos, particle_centroid)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        particle_pos = self._particle_pos_e()
        count = self._count_particles_in_bin(particle_pos)
        delta_count = count - self._previous_bin_count
        self._previous_bin_count = count
        progress = self._particle_progress_toward_bin(particle_pos)
        delta_progress = progress - self._previous_particle_progress
        self._previous_particle_progress = progress
        bin_proximity = self._particle_bin_proximity(particle_pos)
        delta_bin_proximity = bin_proximity - self._previous_bin_proximity
        self._previous_bin_proximity = bin_proximity
        spill_fraction = self._particles_spilled(particle_pos).float().mean(dim=1)

        paddle_pos = self._paddle_pos_e()
        paddle_distance = torch.linalg.norm(paddle_pos - self._pile_center, dim=-1)
        paddle_proximity = torch.exp(-4.0 * paddle_distance)
        paddle_speed = torch.linalg.norm((paddle_pos - self._previous_paddle_pos) / self.step_dt, dim=-1)
        self._previous_paddle_pos = paddle_pos
        action_penalty = torch.sum(torch.square(self._actions), dim=-1)

        rewards = {
            "particle_count": self.cfg.reward_count_scale * count / float(self._particle_count),
            "delta_count": self.cfg.reward_delta_count_scale * torch.clamp(delta_count, min=0.0) / float(
                self._particle_count
            ),
            "particle_progress": self.cfg.reward_particle_progress_scale * torch.clamp(delta_progress, min=0.0),
            "bin_proximity": self.cfg.reward_bin_proximity_scale * bin_proximity,
            "delta_bin_proximity": self.cfg.reward_delta_bin_proximity_scale
            * torch.clamp(delta_bin_proximity, min=0.0),
            "spill_penalty": -self.cfg.reward_spill_penalty_scale * spill_fraction,
            "paddle_proximity": self.cfg.reward_paddle_proximity_scale * paddle_proximity,
            "paddle_speed_penalty": -self.cfg.reward_paddle_speed_penalty_scale * torch.square(paddle_speed),
            "action_penalty": -self.cfg.action_penalty_scale * action_penalty,
        }
        reward = torch.stack(list(rewards.values()), dim=0).sum(dim=0)
        for name, value in rewards.items():
            self._episode_sums[name] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = env_ids.long()

        if hasattr(self, "_episode_sums"):
            extras = {}
            for key, value in self._episode_sums.items():
                extras[f"Episode_Reward/{key}"] = value[env_ids].mean().item() / self.max_episode_length_s
                value[env_ids] = 0.0
            extras["Metrics/particles_in_bin"] = self._previous_bin_count[env_ids].mean().item()
            extras["Metrics/bin_proximity"] = self._previous_bin_proximity[env_ids].mean().item()
            self.extras["log"] = extras

        super()._reset_idx(env_ids)
        state_0 = NewtonManager.get_state_0()
        state_1 = NewtonManager.get_state_1()
        control = NewtonManager.get_control()
        joint_q = wp.to_torch(state_0.joint_q)
        joint_qd = wp.to_torch(state_0.joint_qd)
        particle_q = wp.to_torch(state_0.particle_q)
        particle_qd = wp.to_torch(state_0.particle_qd)

        joint_q[self._joint_q_ids[env_ids]] = self._default_joint_q[env_ids]
        joint_qd[self._joint_qd_ids[env_ids]] = self._default_joint_qd[env_ids]
        particle_q[self._particle_ids[env_ids]] = self._default_particle_q[env_ids]
        particle_qd[self._particle_ids[env_ids]] = 0.0
        wp.to_torch(state_1.joint_q)[self._joint_q_ids[env_ids]] = self._default_joint_q[env_ids]
        wp.to_torch(state_1.joint_qd)[self._joint_qd_ids[env_ids]] = self._default_joint_qd[env_ids]
        wp.to_torch(state_1.particle_q)[self._particle_ids[env_ids]] = self._default_particle_q[env_ids]
        wp.to_torch(state_1.particle_qd)[self._particle_ids[env_ids]] = 0.0

        self._joint_targets[env_ids] = self._default_joint_q[env_ids]
        wp.to_torch(control.joint_target_pos)[self._joint_qd_ids[env_ids]] = self._joint_targets[env_ids]
        if hasattr(self, "_mpm_solver"):
            self._mpm_solver._last_step_data.save_collider_current_position(state_0.body_q)
        particle_pos = self._particle_pos_e()
        self._previous_bin_count[env_ids] = self._count_particles_in_bin(particle_pos)[env_ids]
        self._previous_particle_progress[env_ids] = self._particle_progress_toward_bin(particle_pos)[env_ids]
        self._previous_bin_proximity[env_ids] = self._particle_bin_proximity(particle_pos)[env_ids]
        self._previous_paddle_pos[env_ids] = self._paddle_pos_e()[env_ids]

    def _particle_heightmap(self, particle_pos: torch.Tensor) -> torch.Tensor:
        map_size = self.cfg.heightmap_size
        rel_x = (particle_pos[..., 0] - self._heightmap_x_min) / self._heightmap_x_range
        rel_y = (particle_pos[..., 1] - self._heightmap_y_min) / self._heightmap_y_range
        px = torch.clamp((rel_x * map_size).long(), 0, map_size - 1)
        py = torch.clamp((rel_y * map_size).long(), 0, map_size - 1)
        particle_height = torch.clamp(
            (particle_pos[..., 2] - self.cfg.table_top_z) / self.cfg.heightmap_z_range, 0.0, 1.0
        )
        valid = (rel_x >= 0.0) & (rel_x < 1.0) & (rel_y >= 0.0) & (rel_y < 1.0)
        flat_indices = self._heightmap_env_offsets + py * map_size + px
        flat_values = torch.where(valid, particle_height, torch.zeros_like(particle_height))
        height = torch.zeros(self.num_envs * map_size * map_size, device=self.device)
        height.scatter_reduce_(0, flat_indices.reshape(-1), flat_values.reshape(-1), reduce="amax", include_self=True)
        return height.reshape(self.num_envs, map_size, map_size)

    def _particle_pos_e(self) -> torch.Tensor:
        particle_pos_w = wp.to_torch(NewtonManager.get_state_0().particle_q)[self._particle_ids]
        return particle_pos_w - self.scene.env_origins[:, None, :]

    def _paddle_pos_e(self) -> torch.Tensor:
        body_q = wp.to_torch(NewtonManager.get_state_0().body_q)
        return body_q[self._ee_body_ids, :3] - self.scene.env_origins

    def _robust_particle_centroid(self, particle_pos: torch.Tensor) -> torch.Tensor:
        in_workspace = self._particles_in_workspace(particle_pos)
        weights = in_workspace.float()
        weighted_sum = torch.sum(particle_pos * weights.unsqueeze(-1), dim=1)
        valid_count = weights.sum(dim=1, keepdim=True)
        denom = torch.clamp(valid_count, min=1.0)
        fallback = self._default_particle_q.mean(dim=1) - self.scene.env_origins
        return torch.where(valid_count > 0.0, weighted_sum / denom, fallback)

    def _count_particles_in_bin(self, particle_pos: torch.Tensor) -> torch.Tensor:
        return self._particles_in_bin(particle_pos).sum(dim=1, dtype=torch.float32)

    def _particles_in_bin(self, particle_pos: torch.Tensor) -> torch.Tensor:
        above_lower = particle_pos > self._bin_lower
        below_upper = particle_pos < self._bin_upper
        return torch.all(above_lower & below_upper, dim=-1)

    def _particles_in_workspace(self, particle_pos: torch.Tensor) -> torch.Tensor:
        above_lower = particle_pos > self._workspace_lower
        below_upper = particle_pos < self._workspace_upper
        return torch.all(above_lower & below_upper, dim=-1)

    def _particles_spilled(self, particle_pos: torch.Tensor) -> torch.Tensor:
        return ~self._particles_in_workspace(particle_pos)

    def _particle_bin_proximity(self, particle_pos: torch.Tensor) -> torch.Tensor:
        in_workspace = self._particles_in_workspace(particle_pos).float()
        xy_scale = torch.clamp(self._bin_half_extents[:2], min=1.0e-6)
        z_scale = max(float(self.cfg.bin_wall_height), 1.0e-6)
        xy_error = (particle_pos[..., :2] - self._bin_target[:2]) / xy_scale
        z_error = (particle_pos[..., 2] - self._bin_target[2]) / z_scale
        distance = torch.sqrt(torch.sum(torch.square(xy_error), dim=-1) + 0.25 * torch.square(z_error))
        score = torch.exp(-torch.clamp(distance, max=4.0))
        return (score * in_workspace).sum(dim=1) / torch.clamp(in_workspace.sum(dim=1), min=1.0)

    def _particle_progress_toward_bin(self, particle_pos: torch.Tensor) -> torch.Tensor:
        particle_x = particle_pos[..., 0]
        x_progress = (particle_x - self._progress_start_x) / max(self._progress_target_x - self._progress_start_x, 1.0e-6)
        x_progress = torch.clamp(x_progress, 0.0, 1.0)
        y_error = (particle_pos[..., 1] - self._bin_center[1]) / max(float(self.cfg.bin_inner_half_extents[1]), 1.0e-6)
        y_alignment = torch.exp(-torch.square(y_error))
        z_valid = (particle_pos[..., 2] > self._bin_lower[2]) & (particle_pos[..., 2] < self._bin_upper[2])
        return (x_progress * y_alignment * z_valid.float()).mean(dim=1)

    def _configure_newton_viewer(self) -> None:
        for visualizer in self.sim.visualizers:
            viewer = getattr(visualizer, "_viewer", None)
            if viewer is None:
                continue
            if hasattr(viewer, "show_particles"):
                viewer.show_particles = True
            if hasattr(viewer, "show_contacts"):
                viewer.show_contacts = True

    def _apply_viewer_forces(self) -> None:
        state = NewtonManager.get_state_0()
        for visualizer in self.sim.visualizers:
            viewer = getattr(visualizer, "_viewer", None)
            if viewer is not None and hasattr(viewer, "apply_forces"):
                viewer.apply_forces(state)
