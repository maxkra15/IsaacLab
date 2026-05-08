# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import warp as wp

import newton
from newton import JointTargetMode
from newton.solvers import SolverImplicitMPM, SolverMuJoCo

from isaaclab.envs import DirectRLEnv
from isaaclab_newton.physics import (
    CoupledProxyCfg,
    CoupledSolverEntryCfg,
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonManager,
    ProxyCouplingCfg,
)

from .ur10_particle_scoop_env_cfg import UR10ParticleScoopEnvCfg

RIGID_ENTRY = "ur10_rigid"
MPM_ENTRY = "particles_mpm"


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

        state = NewtonManager.get_state_0()
        self._default_joint_q = wp.to_torch(state.joint_q)[self._joint_q_ids].clone()
        self._default_joint_qd = wp.to_torch(state.joint_qd)[self._joint_qd_ids].clone()
        self._default_particle_q = wp.to_torch(state.particle_q)[self._particle_ids].clone()
        self._joint_targets = self._default_joint_q.clone()
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_bin_count = torch.zeros(self.num_envs, device=self.device)
        self._previous_particle_progress = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums = {
            "particle_count": torch.zeros(self.num_envs, device=self.device),
            "delta_count": torch.zeros(self.num_envs, device=self.device),
            "particle_progress": torch.zeros(self.num_envs, device=self.device),
            "paddle_proximity": torch.zeros(self.num_envs, device=self.device),
            "action_penalty": torch.zeros(self.num_envs, device=self.device),
        }
        self._configure_newton_viewer()

    def _setup_scene(self) -> None:
        builder, solver_cfg = self._build_newton_model()
        self.cfg.sim.physics.solver_cfg = solver_cfg
        NewtonManager._num_envs = self.scene.num_envs
        NewtonManager.set_builder(builder)

    def _build_newton_model(self) -> tuple[newton.ModelBuilder, object]:
        proto, meta = self._build_world_proto()
        builder = NewtonManager.create_builder()
        SolverMuJoCo.register_custom_attributes(builder)
        SolverImplicitMPM.register_custom_attributes(builder)

        rigid_bodies: list[int] = []
        rigid_joints: list[int] = []
        visible_shapes: list[int] = []
        particles: list[int] = []

        for env_id in range(self.scene.num_envs):
            origin = self.scene.env_origins[env_id].detach().cpu().tolist()
            builder.begin_world(label=f"env_{env_id}")
            body_offset = builder.body_count
            joint_offset = builder.joint_count
            shape_offset = builder.shape_count
            particle_offset = builder.particle_count
            joint_q_offset = builder.joint_coord_count
            joint_qd_offset = builder.joint_dof_count

            builder.add_builder(proto, xform=wp.transform(wp.vec3(*origin), wp.quat_identity()))

            rigid_bodies.extend(range(body_offset, body_offset + meta["body_count"]))
            rigid_joints.extend(range(joint_offset, joint_offset + meta["joint_count"]))
            visible_shapes.extend(range(shape_offset, shape_offset + meta["shape_count"]))
            particle_range = list(range(particle_offset, particle_offset + meta["particle_count"]))
            particles.extend(particle_range)
            self._particle_ids_list.append(particle_range)
            self._joint_q_ids_list.append([joint_q_offset + idx for idx in meta["arm_q_ids"]])
            self._joint_qd_ids_list.append([joint_qd_offset + idx for idx in meta["arm_qd_ids"]])
            self._ee_body_ids_list.append(body_offset + int(meta["ee_body"]))
            builder.end_world()

        solver_cfg = self.cfg.sim.physics.solver_cfg
        solver_cfg.entries = [
            CoupledSolverEntryCfg(
                name=RIGID_ENTRY,
                solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False, njmax=160, iterations=80),
                bodies=rigid_bodies,
                joints=rigid_joints,
                shapes=visible_shapes,
                substeps=2,
            ),
            CoupledSolverEntryCfg(
                name=MPM_ENTRY,
                solver_cfg=MPMSolverCfg(
                    voxel_size=self.cfg.voxel_size,
                    grid_type="fixed",
                    grid_padding=self.cfg.mpm_grid_padding,
                    max_active_cell_count=self.cfg.mpm_max_active_cell_count,
                    strain_basis="P0",
                    transfer_scheme="apic",
                    max_iterations=self.cfg.mpm_iterations,
                    critical_fraction=0.0,
                    collider_velocity_mode="forward",
                ),
                particles=particles,
                shapes=visible_shapes,
            ),
        ]
        solver_cfg.use_collision_pipeline = True
        solver_cfg.proxy_coupling = ProxyCouplingCfg(
            proxies=[
                CoupledProxyCfg(
                    source=RIGID_ENTRY,
                    destination=MPM_ENTRY,
                    bodies=rigid_bodies,
                    mass_scale=1.0,
                    mode="lagged",
                )
            ],
            iterations=1,
        )
        return builder, solver_cfg

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

        ee_body = self._find_body(proto, self.cfg.ee_body_name)
        sx, sy, sz = self.cfg.paddle_size
        proto.add_shape_box(
            ee_body,
            xform=wp.transform(wp.vec3(*self.cfg.paddle_ee_offset), wp.quat_identity()),
            hx=0.5 * sx,
            hy=0.5 * sy,
            hz=0.5 * sz,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.9, density=0.0),
            color=(0.1, 0.25, 0.85),
        )
        self._add_workspace_colliders(proto)
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
            "arm_q_ids": arm_q_ids,
            "arm_qd_ids": arm_qd_ids,
            "ee_body": ee_body,
        }

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

    def _add_workspace_colliders(self, builder: newton.ModelBuilder) -> None:
        tx, ty, tz = self.cfg.table_center
        sx, sy, sz = self.cfg.table_size
        table_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, density=0.0)
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(tx, ty, tz), wp.quat_identity()),
            hx=0.5 * sx,
            hy=0.5 * sy,
            hz=0.5 * sz,
            cfg=table_cfg,
            color=(0.45, 0.34, 0.24),
        )

        wall_thickness = 0.035
        bin_height = 0.22
        bin_x, bin_y, _ = self.cfg.bin_center
        bin_half_x, bin_half_y, _ = self.cfg.bin_inner_half_extents
        wall_z = self.cfg.table_top_z + 0.5 * bin_height
        bottom_z = self.cfg.table_top_z - 0.5 * wall_thickness
        wall_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, density=0.0)
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(bin_x, bin_y, bottom_z), wp.quat_identity()),
            hx=bin_half_x + 0.5 * wall_thickness,
            hy=bin_half_y + 0.5 * wall_thickness,
            hz=0.5 * wall_thickness,
            cfg=wall_cfg,
            color=(0.08, 0.14, 0.26),
        )
        walls = [
            ((2.0 * bin_half_x + wall_thickness, wall_thickness, bin_height), (bin_x, bin_y + bin_half_y, wall_z)),
            ((2.0 * bin_half_x + wall_thickness, wall_thickness, bin_height), (bin_x, bin_y - bin_half_y, wall_z)),
            ((wall_thickness, 2.0 * bin_half_y + wall_thickness, bin_height), (bin_x + bin_half_x, bin_y, wall_z)),
        ]
        for size, pos in walls:
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(*pos), wp.quat_identity()),
                hx=0.5 * size[0],
                hy=0.5 * size[1],
                hz=0.5 * size[2],
                cfg=wall_cfg,
                color=(0.1, 0.18, 0.32),
            )

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
        heightmap = self._particle_heightmap().reshape(self.num_envs, -1)
        state = NewtonManager.get_state_0()
        joint_q = wp.to_torch(state.joint_q)[self._joint_q_ids]
        joint_qd = wp.to_torch(state.joint_qd)[self._joint_qd_ids]
        bin_fraction = self._count_particles_in_bin()[:, None] / float(self._particle_count)
        paddle_pos = self._paddle_pos_e()
        particle_centroid = self._particle_pos_e().mean(dim=1)
        bin_center = torch.tensor(self.cfg.bin_center, device=self.device).unsqueeze(0).expand(self.num_envs, -1)
        obs = torch.cat(
            (
                heightmap,
                joint_q / torch.pi,
                0.1 * joint_qd,
                paddle_pos,
                particle_centroid,
                bin_center - paddle_pos,
                bin_fraction,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        count = self._count_particles_in_bin()
        delta_count = count - self._previous_bin_count
        self._previous_bin_count = count
        progress = self._particle_progress_toward_bin()
        delta_progress = progress - self._previous_particle_progress
        self._previous_particle_progress = progress

        pile_center = torch.tensor(self.cfg.pile_lo, device=self.device)
        pile_center = 0.5 * (pile_center + torch.tensor(self.cfg.pile_hi, device=self.device))
        paddle_distance = torch.linalg.norm(self._paddle_pos_e() - pile_center, dim=-1)
        paddle_proximity = torch.exp(-4.0 * paddle_distance)
        action_penalty = torch.sum(torch.square(self._actions), dim=-1)

        rewards = {
            "particle_count": self.cfg.reward_count_scale * count / float(self._particle_count),
            "delta_count": self.cfg.reward_delta_count_scale * torch.clamp(delta_count, min=0.0) / float(
                self._particle_count
            ),
            "particle_progress": self.cfg.reward_particle_progress_scale * torch.clamp(delta_progress, min=0.0),
            "paddle_proximity": self.cfg.reward_paddle_proximity_scale * paddle_proximity,
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
        self._previous_bin_count[env_ids] = self._count_particles_in_bin()[env_ids]
        self._previous_particle_progress[env_ids] = self._particle_progress_toward_bin()[env_ids]

    def _particle_heightmap(self) -> torch.Tensor:
        particle_pos = self._particle_pos_e()
        map_size = self.cfg.heightmap_size
        x_min, x_max = 0.05, 1.15
        y_min, y_max = -0.45, 0.45
        height = torch.zeros(self.num_envs, map_size, map_size, device=self.device)
        rel_x = (particle_pos[..., 0] - x_min) / (x_max - x_min)
        rel_y = (particle_pos[..., 1] - y_min) / (y_max - y_min)
        px = torch.clamp((rel_x * map_size).long(), 0, map_size - 1)
        py = torch.clamp((rel_y * map_size).long(), 0, map_size - 1)
        particle_height = torch.clamp(
            (particle_pos[..., 2] - self.cfg.table_top_z) / self.cfg.heightmap_z_range, 0.0, 1.0
        )
        env_range = torch.arange(self.num_envs, device=self.device)
        for particle_id in range(self._particle_count):
            valid = (rel_x[:, particle_id] >= 0.0) & (rel_x[:, particle_id] < 1.0)
            valid = valid & (rel_y[:, particle_id] >= 0.0) & (rel_y[:, particle_id] < 1.0)
            env_ids = env_range[valid]
            if env_ids.numel() == 0:
                continue
            height[env_ids, py[valid, particle_id], px[valid, particle_id]] = torch.maximum(
                height[env_ids, py[valid, particle_id], px[valid, particle_id]],
                particle_height[valid, particle_id],
            )
        return height

    def _particle_pos_e(self) -> torch.Tensor:
        particle_pos_w = wp.to_torch(NewtonManager.get_state_0().particle_q)[self._particle_ids]
        return particle_pos_w - self.scene.env_origins[:, None, :]

    def _paddle_pos_e(self) -> torch.Tensor:
        body_q = wp.to_torch(NewtonManager.get_state_0().body_q)
        return body_q[self._ee_body_ids, :3] - self.scene.env_origins

    def _count_particles_in_bin(self) -> torch.Tensor:
        particle_pos = self._particle_pos_e()
        cx, cy, cz = self.cfg.bin_center
        hx, hy, hz = self.cfg.bin_inner_half_extents
        in_x = (particle_pos[..., 0] > cx - hx) & (particle_pos[..., 0] < cx + hx)
        in_y = (particle_pos[..., 1] > cy - hy) & (particle_pos[..., 1] < cy + hy)
        in_z = (particle_pos[..., 2] > self.cfg.table_top_z - 0.03) & (particle_pos[..., 2] < cz + hz)
        return (in_x & in_y & in_z).float().sum(dim=1)

    def _particle_progress_toward_bin(self) -> torch.Tensor:
        particle_x = self._particle_pos_e()[..., 0]
        start_x = float(self.cfg.pile_lo[0])
        target_x = float(self.cfg.bin_center[0] - self.cfg.bin_inner_half_extents[0])
        normalized = (particle_x - start_x) / max(target_x - start_x, 1.0e-6)
        return torch.clamp(normalized, 0.0, 1.0).mean(dim=1)

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
