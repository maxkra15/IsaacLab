# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect or evaluate a six-action held-pour task using the actual Rizon demo.

Reuse the finalized native scene with --episodes. Episode zero retains the
original cloud; later episodes perturb particles within the validated cavity.
Vary --fill-seed for other base clouds and --seed for action variants; variants
of the same physical reset retain one split group::

    uv run python -m examples.mpm_world_model.collect_teapot --output /tmp/pour-0 \
        --seed 0 --particle-samples 1024 --headless -- --voxel_size 0.004

The prefix uses the original contact-acquired grasp. After two seconds the actor
controls bounded desired-pot residuals through the original differential IK.
--policy-checkpoint accepts a TorchScript actor [B,121] -> [B,6]. This is a new
prototype task, with its own observation, reward and success definition.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.mpm_world_model.franka_pour_state import CONTEXT_NAMES, configure_source_paths, source_digest
from examples.mpm_world_model.recording import EpisodeWriter, atomic_json
from examples.mpm_world_model.teapot_state import (
    ACTION_NAMES,
    SUMMARY_NAMES,
    TeapotResidualController,
    context_tensor,
    particle_summary,
    point_in_body_frame,
    pour_reward,
    rigid_state,
    spatial_sample,
)
from examples.mpm_world_model.teapot_video import NativeTeapotVideo, configure_capture_environment


def _warp_array(value):
    """Accept the demo revisions that expose native arrays or ProxyArray views."""
    return value.warp if hasattr(value, "warp") else value


def _reset_split(reset_group: str) -> str:
    """Predict the recorder's immutable SHA-256 reset-group split before stepping."""
    bucket = int(hashlib.sha256(reset_group.encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 8 else ("validation" if bucket == 8 else "test")


@wp.kernel
def _accumulate_rigid_capacity(
    nefc: wp.array(dtype=wp.int32),
    ncon: wp.array(dtype=wp.int32),
    overflow: wp.array(dtype=wp.int32),
    njmax: int,
    nconmax: int,
    capacity_mask: int,
    statistics: wp.array(dtype=wp.int32),
):
    """Retain transient saturation/overflow between slower policy observations."""
    constraints, contacts = nefc[0], ncon[0]
    statistics[0] = wp.max(statistics[0], constraints)
    statistics[1] = wp.max(statistics[1], contacts)
    capacity_bits = overflow[0] & capacity_mask
    # Conservatively reject saturation as well as a count strictly beyond cap;
    # this leaves headroom for native kernels that reserve multi-row records.
    if constraints >= njmax:
        capacity_bits = capacity_bits | (1 << 28)
    if contacts >= nconmax:
        capacity_bits = capacity_bits | (1 << 29)
    statistics[2] = statistics[2] | capacity_bits
    statistics[3] = statistics[3] | overflow[0]
    statistics[4] = statistics[4] + 1
    statistics[5] = constraints
    statistics[6] = contacts


class RigidCapacityMonitor:
    """Audit native rigid capacities at800Hz without a host read per microstep."""

    def __init__(self, solver, device):
        import mujoco_warp

        data = solver.mjw_data
        self.statistics = wp.zeros(7, dtype=wp.int32, device=device)
        self.zero = wp.zeros(1, dtype=wp.int32, device=device)
        self.device = device
        self.nefc = getattr(data, "nefc", self.zero)
        self.ncon = getattr(data, "nacon", getattr(data, "ncon", self.zero))
        self.overflow = getattr(data, "overflow", self.zero)
        self.njmax = int(getattr(data, "njmax", 0))
        self.nconmax = int(getattr(data, "naconmax", getattr(data, "nconmax", 0)))
        self.available = (
            data is not None
            and isinstance(self.nefc, wp.array)
            and isinstance(self.ncon, wp.array)
            and self.nefc is not self.zero
            and self.ncon is not self.zero
            and self.njmax > 0
            and self.nconmax > 0
            and int(getattr(data, "nworld", 1)) == 1
        )
        enum = getattr(mujoco_warp, "OverflowType", None)
        self.bit_names = {} if enum is None else {member.name: int(member) for member in enum}
        # Iteration-limit flags are convergence diagnostics, not buffer drops.
        self.capacity_mask = -1 if enum is None else 0
        for name, value in self.bit_names.items():
            if name not in ("NONE", "ALL", "ITERATIONS", "LS_ITERATIONS"):
                self.capacity_mask |= value
        self.diagnostic_names = [
            "max_nefc",
            "max_ncon",
            "capacity_overflow_or_saturation_bits",
            "all_native_overflow_bits",
            "native_steps_observed",
            "last_nefc",
            "last_ncon",
        ]

    def clear(self) -> None:
        """Separate episodes even on native versions with accumulated overflow flags."""
        self.statistics.zero_()
        self.overflow.zero_()

    def sample(self) -> None:
        if self.available:
            wp.launch(
                _accumulate_rigid_capacity,
                dim=1,
                inputs=[
                    self.nefc,
                    self.ncon,
                    self.overflow,
                    self.njmax,
                    self.nconmax,
                    self.capacity_mask,
                    self.statistics,
                ],
                device=self.device,
            )

    def snapshot(self) -> tuple[torch.Tensor, torch.Tensor]:
        values = wp.to_torch(self.statistics)
        return values, (values[2] == 0) & self.available

    def metadata(self) -> dict:
        return {
            "available": self.available,
            "njmax": self.njmax,
            "nconmax": self.nconmax,
            "native_overflow_flags_available": self.overflow is not self.zero,
            "native_overflow_bit_names": self.bit_names,
            "capacity_mask": self.capacity_mask,
            "collector_saturation_bits": {"nefc_at_or_above_njmax": 1 << 28, "ncon_at_or_above_nconmax": 1 << 29},
            "diagnostic_names": self.diagnostic_names,
            "sample_hz": 800,
            "semantics": "all_native_steps_including_grasp_prefix;iteration_limit_bits_reported_separately",
        }

    def outcome(self) -> dict:
        values, valid = self.snapshot()
        values = values.cpu().tolist()
        return {
            "rigid_capacity_valid": bool(valid),
            "rigid_capacity_observable_available": self.available,
            "max_rigid_nefc": values[0],
            "max_rigid_ncon": values[1],
            "rigid_capacity_overflow_or_saturation_bits": values[2],
            "rigid_all_native_overflow_bits": values[3],
            "rigid_native_steps_observed": values[4],
            "rigid_njmax": self.njmax,
            "rigid_nconmax": self.nconmax,
        }


class TeapotState:
    """Read the exact demo's state; classify mass against its actual hollow meshes."""

    def __init__(self, scene, motion, residual, demo, samples: int, seed: int, target_fraction: float, material: bool):
        import warp as wp
        from isaaclab_newton.physics import NewtonManager

        self.wp = wp
        self.scene, self.motion, self.residual, self.demo = scene, motion, residual, demo
        self.fluid, self.robot, self.container = scene["fluid"], scene["robot"], scene["container"]
        self.device = self.robot.device
        self.origin = torch.zeros(3, device=self.device)
        self.target_fraction, self.include_material = target_fraction, material
        self.grasp_reference = None
        self.max_grasp_drift = 0.0
        self.max_position_tracking_error = 0.0
        self.max_rotation_tracking_error = 0.0
        self.grasp_drift_threshold = 0.1
        self.grasp_diagnostic_names = [
            "palm_in_pot_x_m",
            "palm_in_pot_y_m",
            "palm_in_pot_z_m",
            "palm_relative_drift_x_m",
            "palm_relative_drift_y_m",
            "palm_relative_drift_z_m",
            "palm_relative_translation_drift_m",
            "pot_position_tracking_error_m",
            "pot_rotation_tracking_error_rad",
            "held_reference_valid",
            "retained_grasp_proxy",
        ]
        self.entry = NewtonManager._solver._entries["fluid"]
        self.solver = self.entry.solver
        self.capacity = RigidCapacityMonitor(NewtonManager._solver.solver("robot"), self.device)
        model = NewtonManager.get_model()
        self.mass = wp.to_torch(model.particle_mass).reshape(-1)
        self.radius = wp.to_torch(model.particle_radius).reshape(-1)
        position = self.fluid.data.particle_pos_w.torch.reshape(-1, 3)
        if len(position) != len(self.mass):
            raise ValueError("The teapot adapter requires the fluid to own all particles.")
        self.indices, self.weights = spatial_sample(position, self.mass, samples, seed)
        local_to_global = wp.to_torch(self.entry.particle_local_to_global).long()
        if not torch.equal(torch.sort(local_to_global).values, torch.arange(len(self.mass), device=self.device)):
            raise ValueError("The fluid entry particle mapping is not a complete permutation.")
        self.global_to_local = torch.argsort(local_to_global)
        self.meshes, self.mesh_bounds, self.geometry = [], [], []
        for asset, scale in ((demo.args_cli.container_usd, 1.0), (demo.args_cli.bowl_usd, demo.BOWL_SCALE)):
            path = Path(demo.retrieve_file_path(asset))
            vertices, faces = demo.load_asset_mesh(str(path))
            vertices = (vertices * scale).astype(np.float32)
            self.meshes.append(
                wp.Mesh(
                    points=wp.array(vertices, dtype=wp.vec3, device=self.device),
                    indices=wp.array(faces.reshape(-1), dtype=wp.int32, device=self.device),
                    support_winding_number=True,
                )
            )
            self.mesh_bounds.append((vertices.min(0), vertices.max(0)))
            self.geometry.append(
                {
                    "asset_uri": asset,
                    "asset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "vertices_faces_sha256": hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest(),
                    "local_aabb_lo": vertices.min(0).tolist(),
                    "local_aabb_hi": vertices.max(0).tolist(),
                }
            )
        self.flags = [wp.empty(len(self.mass), dtype=wp.float32, device=self.device) for _ in range(2)]
        self.bowl_pose = torch.tensor([*demo.BOWL_BASE_POS, 0, 0, 0, 1], device=self.device)
        self.zero_twist = torch.zeros(6, device=self.device)
        self.observation_names = (
            [f"joint_position_{name}" for name in self.robot.joint_names]
            + [f"joint_velocity_{name}" for name in self.robot.joint_names]
            + [
                f"{body}_{field}"
                for body in ("source", "target")
                for field in ("x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz")
            ]
            + list(SUMMARY_NAMES)
            + [f"residual_{name}" for name in ACTION_NAMES]
            + [f"previous_action_{name}" for name in ACTION_NAMES]
            + ["time_since_grasp_s", "remaining_fraction", "desired_target_fraction"]
        )
        if len(self.observation_names) != 121:
            raise ValueError(f"Unexpected Rizon observation width: {len(self.observation_names)}")

    def boundary(self) -> torch.Tensor:
        """Measured boundary roots after one native step, in the shared WXYZ format."""
        pose = self.container.data.root_link_pose_w.torch[0]
        twist = self.container.data.root_link_vel_w.torch[0]
        return torch.stack(
            (rigid_state(pose, twist, self.origin), rigid_state(self.bowl_pose, self.zero_twist, self.origin))
        )

    def _cavity(
        self,
        positions: torch.Tensor,
        pose: torch.Tensor,
        index: int,
        surface_margin: float = 0.0,
        water_level: float = 100.0,
    ) -> torch.Tensor:
        from isaaclab.utils.warp.particle_sampler import mark_cavity_points_kernel

        safe_positions = torch.where(torch.isfinite(positions).all(-1, keepdim=True), positions, 100.0)
        local = point_in_body_frame(safe_positions, pose).contiguous()
        self.wp.launch(
            mark_cavity_points_kernel,
            dim=len(positions),
            inputs=[
                self.wp.from_torch(local, dtype=self.wp.vec3),
                self.meshes[index].id,
                100.0,
                surface_margin,
                100.0,
                5,
                water_level,
                self.flags[index],
            ],
            device=self.device,
        )
        return self.wp.to_torch(self.flags[index]) > 0.5

    def begin_held_episode(self) -> None:
        """Acquire the diagnostic reference from the actual post-grasp physical state."""
        pot = self.container.data.root_link_pose_w.torch[0]
        palm = self.robot.data.body_link_pose_w.torch[0, self.motion._palm, :3]
        self.grasp_reference = point_in_body_frame(palm, pot).detach().clone()
        self.max_grasp_drift = 0.0
        self.max_position_tracking_error = 0.0
        self.max_rotation_tracking_error = 0.0

    def _grasp_diagnostics(self, sim_time: float) -> torch.Tensor:
        pot = self.container.data.root_link_pose_w.torch[0]
        palm = self.robot.data.body_link_pose_w.torch[0, self.motion._palm, :3]
        relative = point_in_body_frame(palm, pot)
        valid = self.grasp_reference is not None
        drift = relative - self.grasp_reference if valid else torch.zeros_like(relative)
        drift_norm = torch.linalg.vector_norm(drift)
        position, quaternion, _ = self.motion._pose_at_time(sim_time)
        desired_position = torch.tensor(position, device=self.device, dtype=pot.dtype)
        position_error = torch.linalg.vector_norm(pot[:3] - desired_position)
        actual_rotation = pot[3:].double()
        desired_rotation = torch.tensor(quaternion, device=self.device, dtype=torch.float64)
        denominator = actual_rotation.norm() * desired_rotation.norm()
        cosine_half = torch.dot(actual_rotation, desired_rotation).abs() / denominator.clamp_min(1.0e-20)
        rotation_error = (2.0 * torch.acos(cosine_half.clamp(0.0, 1.0))).float()
        if valid:
            self.max_grasp_drift = max(self.max_grasp_drift, float(drift_norm))
            self.max_position_tracking_error = max(self.max_position_tracking_error, float(position_error))
            self.max_rotation_tracking_error = max(self.max_rotation_tracking_error, float(rotation_error))
        retained = valid and self.max_grasp_drift <= self.grasp_drift_threshold
        return torch.cat(
            (
                relative,
                drift,
                torch.stack((drift_norm, position_error, rotation_error)),
                torch.tensor([valid, retained], device=self.device, dtype=pot.dtype),
            )
        )

    def _controller(self, sim_time: float) -> torch.Tensor:
        motion = self.motion
        anchor = np.zeros(6) if motion._thumb_anchor is None else np.concatenate(motion._thumb_anchor)
        hidden = np.concatenate(
            (
                motion._arm_position,
                motion._arm_velocity,
                motion._thumb_position,
                motion._thumb_velocity,
                anchor,
                self.residual.previous,
                self.residual.current,
                self.residual.action,
                [
                    self.residual.start_time,
                    sim_time,
                    -1 if motion._last_feedback_step is None else motion._last_feedback_step,
                    motion._next_feedback_step,
                    float(motion._gain_settled),
                    float(motion._thumb_anchor is not None),
                ],
            )
        )
        return torch.cat(
            (
                torch.as_tensor(hidden, dtype=torch.float32, device=self.device),
                self.wp.to_torch(_warp_array(self.robot.actuators.target_command.position)).reshape(-1),
                self.wp.to_torch(_warp_array(self.robot.actuators.target_command.velocity)).reshape(-1),
            )
        )

    def capture(self, sim_time: float) -> dict[str, torch.Tensor]:
        """Capture a pre-action or pre-reset endpoint, with all-cloud finite checks."""
        positions = self.fluid.data.particle_pos_w.torch.reshape(-1, 3)
        velocities = self.fluid.data.particle_vel_w.torch.reshape(-1, 3)
        source_pose = self.container.data.root_link_pose_w.torch[0]
        source = self._cavity(positions, source_pose, 0)
        target = self._cavity(positions, self.bowl_pose, 1)
        spilled = (positions[:, 2] <= self.demo.TABLE_TOP_Z + 2 * self.radius) | (positions.abs().amax(-1) > 3.0)
        summary, finite = particle_summary(positions - self.origin, velocities, self.mass, source, target, spilled)
        boundary = self.boundary()
        controller = self._controller(sim_time)
        clock = torch.tensor(
            [
                sim_time - 2.0,
                max(0.0, (18.0 - sim_time) / 16.0),
                self.target_fraction,
            ],
            device=self.device,
        )
        observation = torch.cat(
            (
                self.robot.data.joint_pos.torch[0],
                self.robot.data.joint_vel.torch[0],
                boundary.reshape(-1),
                summary,
                torch.as_tensor(self.residual.state(), device=self.device, dtype=torch.float32),
                clock,
            )
        )
        if observation.shape != (121,):
            raise ValueError(f"Teapot actor expects121 values, got {observation.shape}.")
        result = {
            "observation": observation,
            "critic_observation": observation.clone(),
            "particle_state": torch.cat((positions[self.indices] - self.origin, velocities[self.indices]), -1),
            "rigid_state": boundary,
            "controller_state": controller,
            "particle_summary": summary,
            "grasp_diagnostics": self._grasp_diagnostics(sim_time),
        }
        result["rigid_capacity_diagnostics"], result["rigid_capacity_valid"] = self.capacity.snapshot()
        finite &= torch.isfinite(boundary).all() & torch.isfinite(controller).all() & torch.isfinite(observation).all()
        finite &= torch.isfinite(result["grasp_diagnostics"]).all()
        if self.include_material:
            rows, aggregates, means = [], [], []
            valid = self.solver.fidelity == "mpm"
            for name, width in (
                ("particle_elastic_strain", 9),
                ("particle_Jp", 1),
                ("particle_qd_grad", 9),
                ("particle_stress", 9),
            ):
                value = getattr(self.entry.state_0.mpm, name, None)
                if value is None:
                    valid = False
                    rows.append(torch.zeros((len(self.indices), width), device=self.device))
                    aggregates.append(torch.zeros(2 * width, device=self.device))
                    means.append(torch.zeros(width, device=self.device))
                else:
                    full = self.wp.to_torch(value).reshape(-1, width)[self.global_to_local]
                    finite &= torch.isfinite(full).all()
                    rows.append(full[self.indices])
                    # Full-cloud material moments are authoritative; sampled
                    # internal tensors are optional decoder/diagnostic targets.
                    safe = torch.where(torch.isfinite(full), full, 0.0)
                    weight = self.mass / self.mass.sum()
                    mean = (safe * weight[:, None]).sum(0)
                    variance = ((safe - mean).square() * weight[:, None]).sum(0)
                    aggregates.append(torch.cat((mean, variance.sqrt())))
                    means.append(mean)
            result["material_state"] = torch.cat(rows, -1)
            result["material_moments"] = torch.cat(aggregates)
            result["material_summary"] = torch.cat((*means, torch.tensor([float(valid)], device=self.device)))
            result["material_state_valid"] = torch.isfinite(result["material_state"]).all(-1) & valid
            if self.solver.fidelity == "mpm":
                finite &= result["material_state_valid"].all()
        result["finite_mask"] = finite
        return {key: value.detach().clone() for key, value in result.items()}

    def metadata(self, args, initial: dict) -> dict:
        """Describe material units, measured geometry and reset provenance explicitly."""
        demo = self.demo
        values = dict.fromkeys(CONTEXT_NAMES, 0.0)
        values.update(
            {
                "policy_dt_s": 0.01,
                "native_dt_s": 1 / 800 / demo.FLUID_SUBSTEPS,
                "voxel_size_m": demo.VOXEL_SIZE,
                "particle_mass_kg": float(self.mass[0]),
                "particle_radius_m": float(self.radius[0]),
                "relaxation": demo.args_cli.particle_relaxation,
                "full_mpm": float(demo.args_cli.fidelity == "mpm"),
                "source_fill_level": demo.FILL_LEVEL,
                "target_fraction": args.target_fraction,
                "material_density": 1000.0,
                "material_young_modulus": 1.0e15,
                "material_poisson_ratio": 0.3,
                "material_viscosity": 1.0e-3,
                "material_damping": 1.0e-3,
                "material_yield_pressure": 1.0e15,
                "material_tensile_yield_ratio": 1.0,
                "proxy_mass_scale": 1.0,
                "material_law_liquid": 1.0,
                "coupling_one_way_prescribed_boundary": 1.0,
            }
        )
        for body, bounds in zip(("source", "target"), self.mesh_bounds):
            for suffix, point in zip(("lo", "hi"), bounds):
                for axis, number in zip("xyz", point):
                    values[f"{body}_inner_{suffix}_{axis}"] = float(number)
        reset_digest = hashlib.sha256()
        # Hash the full initial cloud, rather than an action-dependent later sample.
        reset_digest.update(self.fluid.data.particle_pos_w.torch.detach().cpu().numpy().tobytes())
        reset_digest.update(json.dumps(self.geometry, sort_keys=True).encode())
        return {
            "task_id": "rizon_sharpa_teapot_held_pour",
            "reset_group": reset_digest.hexdigest(),
            "seed": args.seed,
            "fill_seed": args.fill_seed,
            "dt": 0.01,
            "context": context_tensor(values, self.device).cpu(),
            "context_names": list(CONTEXT_NAMES),
            "particle_source_indices": self.indices.cpu(),
            "particle_mass": self.mass[self.indices].cpu(),
            "particle_weights": self.weights.cpu(),
            "total_particle_count": len(self.mass),
            "total_mass_kg": float(self.mass.sum()),
            "observation_names": self.observation_names,
            "action_names": list(ACTION_NAMES),
            "particle_summary_names": list(SUMMARY_NAMES),
            "rigid_names": ["source", "target"],
            "frame": "env_local_z_up",
            "quaternion_order": "WXYZ",
            "geometry": self.geometry,
            "geometry_bounds_semantics": "mesh_aabbs_not_interior_boxes",
            "region_measurement": "all_particles_double_wall_winding_and_5_of_6_axis_ray_cavity_heuristic",
            "coupling": "one_way_measured_rigid_boundary_to_fluid_no_fluid_feedback",
            "boundary_microtrajectory_semantics": "measured_after_each_800Hz_rigid_step_before_next_step",
            "residual_translation_limit_m": args.translation_limit,
            "residual_rotation_limit_rad": args.rotation_limit,
            "reward_definition": "100*delta_target_mass_fraction-200*delta_spill_mass_fraction-.001*sum(action^2)",
            "success_definition": (
                "target_fraction>=requested_and_spill_fraction<=.1_and_finite"
                "_and_retained_grasp_proxy_and_rigid_capacity_valid"
            ),
            "grasp_diagnostic_names": self.grasp_diagnostic_names,
            "grasp_drift_threshold_m_provisional": self.grasp_drift_threshold,
            "grasp_diagnostic_semantics": (
                "palm_translation_in_pot_frame_relative_to_measured_2s_held_state_not_contact_wrench_proof"
            ),
            "grasp_diagnostic_rate_hz": 100.0,
            "scripted_grasp_prefix_s": 2.0,
            "controller_feedback_hz": 100.0,
            "native_arguments": json.loads(json.dumps(vars(demo.args_cli), default=str)),
            "resolved_fluid_solver_cfg": demo._world_model_resolved_solver_cfg,
            "resolved_rigid_solver_cfg": demo._world_model_resolved_rigid_solver_cfg,
            "rigid_capacity_monitor": self.capacity.metadata(),
        }


class TeapotEpisodeReset:
    """Restore this exact native demo in place, preserving captured array addresses.

    Public asset writes invalidate Isaac caches, while SolverCoupled.reset clears
    solver-owned history (including MuJoCo warm starts and MPM constitutive/grid
    history). A soft SimulationContext reset alone does neither operation.
    """

    def __init__(self, adapter: TeapotState):
        from isaaclab_newton.physics import NewtonManager

        self.adapter = adapter
        self.manager = NewtonManager
        self.parent_initial = NewtonManager.get_model().state()
        self.parent_initial.assign(NewtonManager.get_state_0())
        self.scene_initial = self._clone_scene_state(adapter.scene.get_state())
        self.particle_initial = torch.cat(
            (
                adapter.fluid.data.particle_pos_w.torch,
                adapter.fluid.data.particle_vel_w.torch,
            ),
            -1,
        ).clone()
        self.stiffness = adapter.robot.data.joint_stiffness.torch.clone()
        self.damping = adapter.robot.data.joint_damping.torch.clone()
        motion = adapter.motion
        self.motion_numpy = {
            name: getattr(motion, name).copy()
            for name in (
                "_arm_position",
                "_arm_velocity",
                "_thumb_position",
                "_thumb_velocity",
            )
        }
        self.motion_gpu = {
            name: adapter.wp.clone(getattr(motion, name))
            for name in (
                "_arm_position_gpu",
                "_arm_velocity_gpu",
                "_thumb_position_gpu",
            )
        }
        self.first_feedback = motion._next_feedback_step
        self.target_position = adapter.wp.clone(_warp_array(adapter.robot.actuators.target_command.position))
        self.target_velocity = adapter.wp.clone(_warp_array(adapter.robot.actuators.target_command.velocity))

    @staticmethod
    def _clone_scene_state(value):
        if isinstance(value, dict):
            return {key: TeapotEpisodeReset._clone_scene_state(item) for key, item in value.items()}
        return value.clone()

    def restore(self, sim, jitter_seed: int, jitter_fraction: float) -> dict:
        """Restore q/v, internal history, actuator gains and the acquired-grasp controller."""
        import newton

        adapter, manager = self.adapter, self.manager
        state = manager.get_state_0()
        state.assign(self.parent_initial)
        other = manager.backend.state_1
        if other is not None and other is not state:
            other.assign(self.parent_initial)
        manager.backend.control.clear(manager.get_model())
        adapter.scene.reset()
        adapter.scene.reset_to(self.scene_initial)
        adapter.robot.write_joint_stiffness_to_sim_index(stiffness=self.stiffness)
        adapter.robot.write_joint_damping_to_sim_index(damping=self.damping)
        motion = adapter.motion
        for name, value in self.motion_numpy.items():
            getattr(motion, name)[:] = value
        for name, value in self.motion_gpu.items():
            adapter.wp.copy(getattr(motion, name), value)
        motion._thumb_anchor = None
        motion._last_feedback_step = None
        motion._next_feedback_step = self.first_feedback
        motion._gain_settled = False
        adapter.residual.reset()
        adapter.capacity.clear()
        adapter.grasp_reference = None
        adapter.max_grasp_drift = 0.0
        adapter.max_position_tracking_error = 0.0
        adapter.max_rotation_tracking_error = 0.0
        adapter.wp.copy(_warp_array(adapter.robot.actuators.target_command.position), self.target_position)
        adapter.wp.copy(_warp_array(adapter.robot.actuators.target_command.velocity), self.target_velocity)
        particles = self.particle_initial.clone()
        accepted = torch.ones(len(adapter.mass), dtype=torch.bool, device=adapter.device)
        pose = adapter.container.data.root_link_pose_w.torch[0]
        lower, upper = adapter.mesh_bounds[0]
        water = float(lower[2] + adapter.demo.FILL_LEVEL * (upper[2] - lower[2]))
        # The original sampler and world-to-local transform both use float32;
        # allow one micrometer in this validation, far below the voxel size.
        clearance = max(0.0, adapter.demo.PARTICLE_SURFACE_CLEARANCE - 1.0e-6)
        if jitter_fraction > 0:
            rng = np.random.default_rng(jitter_seed)
            offsets = rng.uniform(-1, 1, (len(adapter.mass), 3)).astype(np.float32)
            offsets = torch.as_tensor(offsets, device=adapter.device) * jitter_fraction * adapter.demo.FILL_SPACING
            candidate = particles[0, :, :3] + offsets
            accepted = adapter._cavity(candidate, pose, 0, clearance, water + 1.0e-6)
            particles[0, :, :3] = torch.where(accepted[:, None], candidate, particles[0, :, :3])
        inside = adapter._cavity(particles[0, :, :3], pose, 0, clearance, water + 1.0e-6)
        if not bool(inside.all()):
            raise RuntimeError(f"Reset violates native source cavity/clearance for {int((~inside).sum())} particles.")
        adapter.fluid.write_nodal_state_to_sim_index(particles)
        adapter.scene.write_data_to_sim()
        sim.forward()
        # flags=PARTICLE preserves restored robot/free-body joint coordinates;
        # MuJoCo still clears its persistent buffers regardless of these flags.
        manager._solver.reset(manager.get_state_0(), flags=newton.StateFlags.PARTICLE)
        state = manager.get_state_0()
        state.clear_forces()
        if other is not None and other is not state:
            other.assign(state)
        adapter.scene.update(sim.get_physics_dt())
        # Refuse to collect if the actual solver-owned constitutive arrays did
        # not reset. Both entry buffers must be clean before graph replay.
        for entry_state in (adapter.entry.state_0, adapter.entry.state_1):
            mpm = entry_state.mpm
            identity = torch.eye(3, device=adapter.device)
            for name in ("particle_elastic_strain", "particle_transform"):
                if not torch.equal(adapter.wp.to_torch(getattr(mpm, name)), identity.expand(len(adapter.mass), 3, 3)):
                    raise RuntimeError(f"Constitutive reset failed: {name}")
            for name in ("particle_qd_grad", "particle_stress"):
                if bool(adapter.wp.to_torch(getattr(mpm, name)).count_nonzero()):
                    raise RuntimeError(f"Constitutive reset failed: {name}")
            if not torch.equal(
                adapter.wp.to_torch(mpm.particle_Jp), adapter.wp.to_torch(adapter.solver.model.mpm.particle_Jp)
            ):
                raise RuntimeError("Constitutive reset failed: particle_Jp")
        actual = torch.cat((adapter.fluid.data.particle_pos_w.torch, adapter.fluid.data.particle_vel_w.torch), -1)
        if not torch.equal(actual, particles):
            raise RuntimeError("Particle reset q/v readback differs from the authored state.")
        displacement = torch.linalg.vector_norm(particles[0, :, :3] - self.particle_initial[0, :, :3], dim=-1)
        return {
            "jitter_seed": jitter_seed,
            "jitter_fraction_of_spacing": jitter_fraction,
            "jitter_accepted_fraction": float(accepted.float().mean()),
            "max_particle_displacement_m": float(displacement.amax()),
            "all_particles_inside_native_cavity": True,
            "constitutive_history_reset": True,
            "mujoCo_warmstarts_reset_via_solver_api": True,
        }


def argument_parser() -> argparse.ArgumentParser:
    """Parse collector flags before loading any simulator; native flags follow --."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--newton-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fill-seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--split-plan",
        nargs="+",
        choices=("train", "validation", "test"),
        help="Select native initial resets by recorder hash; plan length sets episode count. At most100 attempts each.",
    )
    parser.add_argument(
        "--reset-jitter", type=float, default=0.05, help="Post-first-reset jitter as fraction of fill spacing."
    )
    parser.add_argument("--particle-samples", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1600, help="100 Hz post-grasp action steps; at most1600.")
    parser.add_argument("--target-fraction", type=float, default=0.5)
    parser.add_argument("--translation-limit", type=float, default=0.02)
    parser.add_argument("--rotation-limit", type=float, default=0.2)
    parser.add_argument("--action-amplitude", type=float, default=0.5)
    parser.add_argument("--collider-basis", help="Opt-in collision quadrature, e.g. S2, pic256, pic356, pic512.")
    parser.add_argument("--strain-basis", help="Opt-in MPM strain basis; preserves native default when omitted.")
    parser.add_argument("--velocity-basis", help="Opt-in MPM velocity basis; preserves native default when omitted.")
    parser.add_argument("--mpm-iterations", type=int)
    parser.add_argument("--mpm-solver", help="Opt-in rheology solver token, e.g. auto or jacobi.")
    parser.add_argument("--rigid-njmax", type=int, help="Opt-in rigid constraint capacity per world.")
    parser.add_argument("--rigid-nconmax", type=int, help="Opt-in rigid contact capacity per world.")
    parser.add_argument("--transfer-scheme", choices=("pic", "apic"))
    parser.add_argument("--policy-checkpoint", type=Path, help="TorchScript actor121->6 for real-physics evaluation.")
    parser.add_argument(
        "--video-path", type=Path, help="Capture native particle evolution to this MP4; suffix per episode."
    )
    parser.add_argument("--video-fps", type=int, default=25, help="Capture rate; must divide800Hz.")
    parser.add_argument("--without-material", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("native_args", nargs=argparse.REMAINDER)
    return parser


def _configure_native_solvers(demo, args) -> None:
    """Apply opt-in numerical/capacity changes and retain both resolved configs."""
    native_create_sim_cfg = demo.create_sim_cfg

    def configured_sim_cfg():
        cfg = native_create_sim_cfg()
        fluid_entries = [entry for entry in cfg.physics.solver_cfg.entries if entry.name == "fluid"]
        rigid_entries = [entry for entry in cfg.physics.solver_cfg.entries if entry.name == "robot"]
        if len(fluid_entries) != 1 or len(rigid_entries) != 1:
            raise RuntimeError("Expected exactly one fluid and one robot solver entry.")
        fluid_cfg, rigid_cfg = fluid_entries[0].solver_cfg, rigid_entries[0].solver_cfg
        if args.rigid_njmax is not None:
            rigid_cfg.njmax = args.rigid_njmax
        if args.rigid_nconmax is not None:
            rigid_cfg.nconmax = args.rigid_nconmax
        for option, field in (
            ("collider_basis", "collider_basis"),
            ("strain_basis", "strain_basis"),
            ("velocity_basis", "velocity_basis"),
            ("mpm_iterations", "max_iterations"),
            ("mpm_solver", "solver"),
            ("transfer_scheme", "transfer_scheme"),
        ):
            value = getattr(args, option)
            if value is not None:
                setattr(fluid_cfg, field, value)
        demo._world_model_resolved_solver_cfg = json.loads(json.dumps(fluid_cfg.to_dict(), default=str))
        demo._world_model_resolved_rigid_solver_cfg = json.loads(json.dumps(rigid_cfg.to_dict(), default=str))
        return cfg

    demo.create_sim_cfg = configured_sim_cfg


def run(args) -> dict:
    """Reuse the native scene and startup, replacing only its outer data loop."""
    if args.split_plan:
        if args.episodes not in (1, len(args.split_plan)) or args.reset_jitter <= 0:
            raise ValueError("A split plan requires positive reset jitter and a matching episode count.")
        args.episodes = len(args.split_plan)
    configure_source_paths(args.repo, args.newton_root)
    configure_capture_environment(args.video_path is not None)
    import newton

    from isaaclab.paths import ISAACLAB_ROOT

    if Path(ISAACLAB_ROOT).resolve() != args.repo.resolve():
        raise RuntimeError(f"Wrong Isaac Lab import: {ISAACLAB_ROOT}")
    if args.newton_root and not Path(newton.__file__).resolve().is_relative_to(args.newton_root.resolve()):
        raise RuntimeError(f"Wrong Newton import: {newton.__file__}")
    sys.path.insert(0, str(args.repo / "examples"))
    native = args.native_args[1:] if args.native_args[:1] == ["--"] else args.native_args
    if any(flag in native for flag in ("--robot", "--max_steps", "--video")):
        raise ValueError("The collector owns robot/max_steps; request recording with --video-path.")
    argv = [
        str(args.repo / "examples/demos/teapot_fill.py"),
        "--robot",
        "rizon_sharpa",
        "--max_steps",
        str(1600 + args.steps * 8),
        "--fluid_render_mode",
        "particles",
        "--device",
        args.device,
        *native,
    ]
    if args.video_path is not None:
        argv.extend(["--video", "--video_fps", str(args.video_fps), "--visualizer", "newton_gl"])
    original_argv = sys.argv
    sys.argv = argv
    try:
        specification = importlib.util.spec_from_file_location("mpm_world_model_teapot_demo", argv[0])
        demo = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = demo
        specification.loader.exec_module(demo)
    finally:
        sys.argv = original_argv
    if args.headless and args.video_path is None:
        demo.args_cli.visualizer = []
    demo.FILL_SEED = args.fill_seed
    _configure_native_solvers(demo, args)
    actor = (
        None
        if args.policy_checkpoint is None
        else torch.jit.load(str(args.policy_checkpoint), map_location=args.device).eval()
    )
    metrics = {}

    def collect_loop(sim, scene, surface_renderer, robot_motion):
        if robot_motion is None:
            raise RuntimeError("Held-pour collection requires the physical Rizon grasp controller.")
        residual = TeapotResidualController(
            robot_motion, translation_limit=args.translation_limit, rotation_limit=args.rotation_limit
        )
        adapter = TeapotState(
            scene,
            robot_motion,
            residual,
            demo,
            args.particle_samples,
            args.fill_seed,
            args.target_fraction,
            not args.without_material,
        )
        dt = sim.get_physics_dt()
        if not np.isclose(dt, 1 / 800):
            raise ValueError("The calibrated physical grasp requires800Hz.")
        video = None

        def native_step(step):
            robot_motion.update(step * dt)
            scene.write_data_to_sim()
            sim.step(render=False)
            adapter.capacity.sample()
            scene.update(dt)
            if video is not None:
                video.step()

        reset = TeapotEpisodeReset(adapter) if args.episodes > 1 or args.split_plan else None
        writer = EpisodeWriter(
            args.output,
            {
                "isaac_root": str(args.repo.resolve()),
                "isaac_sources_sha256": source_digest(args.repo / "source"),
                "newton_file": newton.__file__,
                "newton_sources_sha256": source_digest(Path(newton.__file__).resolve().parent),
                "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "adapter_sha256": hashlib.sha256((Path(__file__).parent / "teapot_state.py").read_bytes()).hexdigest(),
                "video_helper_sha256": hashlib.sha256(
                    (Path(__file__).parent / "teapot_video.py").read_bytes()
                ).hexdigest(),
                "native_demo_file": demo.__file__,
                "native_demo_sha256": hashlib.sha256(Path(demo.__file__).read_bytes()).hexdigest(),
                "native_controller_file": sys.modules["rizon_sharpa_teapot"].__file__,
                "native_controller_sha256": hashlib.sha256(
                    Path(sys.modules["rizon_sharpa_teapot"].__file__).read_bytes()
                ).hexdigest(),
                "resolved_fluid_solver_cfg": demo._world_model_resolved_solver_cfg,
                "resolved_rigid_solver_cfg": demo._world_model_resolved_rigid_solver_cfg,
                "requested_split_plan": args.split_plan,
                "held_out_scope": "same_geometry_material_and_topology_with_small_initial_particle_jitter",
                "policy_checkpoint": None if args.policy_checkpoint is None else str(args.policy_checkpoint),
                "policy_sha256": None
                if args.policy_checkpoint is None
                else hashlib.sha256(args.policy_checkpoint.read_bytes()).hexdigest(),
            },
        )
        outcomes = []
        selected_groups = set()
        complete = False
        try:
            for episode_index in range(args.episodes):
                requested_split = None if args.split_plan is None else args.split_plan[episode_index]
                attempts = 100 if requested_split is not None else 1
                for attempt in range(attempts):
                    validation = {"kind": "original_native_startup"}
                    if episode_index or attempt:
                        jitter_seed = args.fill_seed + episode_index
                        if requested_split is not None:
                            jitter_seed = args.fill_seed + 100 * episode_index + attempt
                        validation = reset.restore(sim, jitter_seed, args.reset_jitter)
                    # Select a hash group using only its physical initial state,
                    # before grasp, action sampling, rewards or any task outcome.
                    initial_reset = adapter.capture(0.0)
                    if not bool(initial_reset["finite_mask"]):
                        raise RuntimeError("The native reset state is not finite.")
                    episode_metadata = adapter.metadata(args, initial_reset)
                    group = episode_metadata["reset_group"]
                    if requested_split is None or (
                        _reset_split(group) == requested_split and group not in selected_groups
                    ):
                        selected_groups.add(group)
                        break
                else:
                    raise RuntimeError(f"No unused {requested_split} reset group found in100 safe jitter candidates.")
                validation.update({"candidate_attempts": attempt + 1, "requested_split": requested_split})
                episode_metadata.update(
                    {
                        "seed": args.seed + episode_index,
                        "reset_index": episode_index,
                        "reset_validation": validation,
                    }
                )
                rng = np.random.default_rng(args.seed + episode_index)
                phases = rng.uniform(-np.pi, np.pi, 6)
                frequencies = rng.uniform(0.12, 0.4, 6)
                adapter.capacity.clear()
                if args.video_path is not None:
                    video_path = args.video_path
                    if args.episodes > 1:
                        video_path = video_path.with_name(f"{video_path.stem}-episode-{episode_index:04d}.mp4")
                    video = NativeTeapotVideo(
                        video_path,
                        args.video_fps,
                        1600 + args.steps * 8,
                        sim,
                        scene,
                        surface_renderer,
                        {
                            "task_id": episode_metadata["task_id"],
                            "reset_group": group,
                            "action_seed": args.seed + episode_index,
                            "fill_seed": args.fill_seed,
                            "reset_validation": validation,
                            "particle_count": len(adapter.mass),
                            "fidelity": demo.args_cli.fidelity,
                            "fluid_render_mode": demo.args_cli.fluid_render_mode,
                            "control": "scripted_grasp_then_learned_residual_around_scripted_pour"
                            if actor is not None
                            else (
                                "scripted_grasp_then_nominal_scripted_pour_zero_residual"
                                if args.action_amplitude == 0
                                else "scripted_grasp_then_scripted_pour_with_residual_actions"
                            ),
                            "action_amplitude": None if actor is not None else args.action_amplitude,
                            "scripted_grasp_duration_s": 2.0,
                            "provenance": writer.manifest["metadata"],
                            "graphics_environment": {
                                name: os.environ.get(name)
                                for name in ("CUDA_VISIBLE_DEVICES", "EGL_DEVICE_ID", "DISPLAY", "PYOPENGL_PLATFORM")
                            },
                        },
                    )
                for step in range(1600):
                    native_step(step)
                adapter.begin_held_episode()
                state = adapter.capture(2.0)
                writer.start_episode(0, state, episode_metadata)
                for index in range(args.steps):
                    sim_time = 2 + index * 0.01
                    if actor is None:
                        action = torch.as_tensor(
                            args.action_amplitude * np.sin(2 * np.pi * frequencies * (sim_time - 2) + phases),
                            device=args.device,
                            dtype=torch.float32,
                        )
                    else:
                        with torch.inference_mode():
                            action = actor(state["observation"][None])[0]
                        if action.shape != (6,) or not torch.isfinite(action).all():
                            raise ValueError("The actor must return six finite residual actions.")
                        action = action.clamp(-1, 1)
                    residual.apply_action(action.detach().cpu().numpy(), sim_time)
                    boundary = []
                    for microstep in range(8):
                        native_step(1600 + index * 8 + microstep)
                        boundary.append(adapter.boundary().detach().clone())
                    next_state = adapter.capture(sim_time + 0.01)
                    reward = pour_reward(state["particle_summary"], next_state["particle_summary"], action)
                    finite = bool(next_state["finite_mask"])
                    terminated = not finite
                    truncated = index == 1599
                    writer.append_transition(
                        0,
                        action,
                        reward,
                        terminated,
                        truncated,
                        next_state,
                        extra={
                            "boundary_microtrajectory": torch.stack(boundary),
                            "success": finite
                            and bool(next_state["particle_summary"][8] >= args.target_fraction)
                            and bool(next_state["particle_summary"][9] <= 0.1)
                            and bool(next_state["grasp_diagnostics"][-1])
                            and bool(next_state["rigid_capacity_valid"]),
                        },
                    )
                    state = next_state
                    if terminated or truncated or index == args.steps - 1:
                        writer.finish_episode(0, recording_cut=not (terminated or truncated))
                        break
                summary = state["particle_summary"].cpu()
                outcome = {
                    "episode_index": episode_index,
                    "reset_group": episode_metadata["reset_group"],
                    "transitions": writer.manifest["episodes"][-1]["transitions"],
                    "split": writer.manifest["episodes"][-1]["split"],
                    "finite": bool(state["finite_mask"]),
                    "source_fraction": float(summary[7]),
                    "target_fraction": float(summary[8]),
                    "spilled_fraction": float(summary[9]),
                    "in_flight_fraction": float(summary[10]),
                    "invalid_fraction": float(summary[11]),
                    "success": bool(state["finite_mask"])
                    and float(summary[8]) >= args.target_fraction
                    and float(summary[9]) <= 0.1
                    and bool(state["grasp_diagnostics"][-1])
                    and bool(state["rigid_capacity_valid"]),
                    "physics_valid": bool(state["finite_mask"]) and bool(state["rigid_capacity_valid"]),
                    "retained_grasp_proxy": bool(state["grasp_diagnostics"][-1]),
                    "max_palm_relative_translation_drift_m": adapter.max_grasp_drift,
                    "max_pot_position_tracking_error_m": adapter.max_position_tracking_error,
                    "max_pot_rotation_tracking_error_rad": adapter.max_rotation_tracking_error,
                    "grasp_drift_threshold_m_provisional": adapter.grasp_drift_threshold,
                    **adapter.capacity.outcome(),
                }
                if requested_split is not None and outcome["split"] != requested_split:
                    raise RuntimeError("Recorder split disagrees with the pre-physics reset selector.")
                outcomes.append(outcome)
                if video is not None:
                    video.close(outcome)
                    outcome["video_path"] = str(video.path.resolve())
                    video = None
                atomic_json(args.output / "episode-outcomes.json", {"episodes": outcomes})
                print(f"[TEAPOT EPISODE] {json.dumps(outcome)}", flush=True)
            complete = True
        finally:
            try:
                if video is not None:
                    video.close()
            finally:
                writer.close(complete=complete)
        metrics.update(
            {
                **outcomes[-1],
                "task_id": episode_metadata["task_id"],
                "episode_count": len(outcomes),
                "episodes": outcomes,
                "transitions": sum(row["transitions"] for row in outcomes),
                "success_rate": sum(row["success"] for row in outcomes) / len(outcomes),
                "held_pour_only": True,
                "policy_used": actor is not None,
                "physics_mode": demo.args_cli.fidelity,
            }
        )
        atomic_json(args.output / "metrics.json", metrics)

    demo.run_simulator = collect_loop
    demo.main()
    return metrics


def main() -> None:
    parser = argument_parser()
    args = parser.parse_args()
    if not 1 <= args.steps <= 1600 or min(args.particle_samples, args.episodes) < 1:
        parser.error("steps must lie in[1,1600]; particle-samples and episodes must be positive.")
    if not 0 <= args.reset_jitter <= 0.2:
        parser.error("reset-jitter must lie in[0,.2] of the native fill spacing.")
    if not 0 < args.target_fraction <= 1 or not 0 <= args.action_amplitude <= 1:
        parser.error("Target fraction must be in(0,1] and action amplitude in[0,1].")
    if min(args.translation_limit, args.rotation_limit) <= 0:
        parser.error("Residual limits must be positive.")
    if args.mpm_iterations is not None and args.mpm_iterations < 1:
        parser.error("mpm-iterations must be positive.")
    if any(value is not None and value < 1 for value in (args.rigid_njmax, args.rigid_nconmax)):
        parser.error("Rigid capacities must be positive.")
    if args.policy_checkpoint is not None and not args.policy_checkpoint.is_file():
        parser.error(f"Missing policy: {args.policy_checkpoint}")
    if args.video_path is not None and (
        args.video_path.suffix.lower() != ".mp4" or args.video_fps < 1 or 800 % args.video_fps
    ):
        parser.error("Video output must be.mp4 and video-fps must divide800Hz.")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
