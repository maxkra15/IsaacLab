# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical-state adapter for the unchanged Franka Pour 84/86/8 interface."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

CONTEXT_NAMES = (
    "policy_dt_s",
    "native_dt_s",
    "voxel_size_m",
    "particle_mass_kg",
    "particle_radius_m",
    "pair_friction",
    "pair_cohesion",
    "relaxation",
    "compliance_m_per_N",
    "full_mpm",
    "source_fill_level",
    "target_fraction",
    "material_density",
    "material_young_modulus",
    "material_poisson_ratio",
    "material_viscosity",
    "material_friction",
    "material_damping",
    "material_yield_pressure",
    "material_tensile_yield_ratio",
    "material_yield_stress",
    "material_hardening",
    "material_dilatancy",
    "material_law_granular",
    "coupling_staggered_proxy",
    "proxy_mass_scale",
    "source_inner_lo_x",
    "source_inner_lo_y",
    "source_inner_lo_z",
    "source_inner_hi_x",
    "source_inner_hi_y",
    "source_inner_hi_z",
    "target_inner_lo_x",
    "target_inner_lo_y",
    "target_inner_lo_z",
    "target_inner_hi_x",
    "target_inner_hi_y",
    "target_inner_hi_z",
    "material_law_liquid",
    "coupling_one_way_prescribed_boundary",
)


def configure_source_paths(repo: Path, newton_root: Path | None = None) -> None:
    """Override read-only dependency interpreters' editable source paths."""
    roots = [
        repo / "source" / name
        for name in (
            "isaaclab",
            "isaaclab_contrib",
            "isaaclab_newton",
            "isaaclab_tasks",
            "isaaclab_rl",
            "isaaclab_assets",
            "isaaclab_visualizers",
            "isaaclab_physx",
        )
    ]
    if newton_root is not None:
        roots.append(newton_root)
    for path in reversed(roots):
        sys.path.insert(0, str(path.resolve()))


def source_digest(root: Path) -> str:
    """Fingerprint the actual Python sources used by a collection run."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def policy_observations(groups: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate the exact trained actor and asymmetric critic groups."""
    actor = torch.cat((groups["policy"], groups["media"]), dim=-1)
    critic = torch.cat((groups["policy"], groups["media"], groups["privileged"]), dim=-1)
    if actor.shape[-1] != 84 or critic.shape[-1] != 86:
        raise ValueError(f"Unexpected policy interface: actor={actor.shape}, critic={critic.shape}")
    return actor, critic


class FrankaPourState:
    """Capture particle history from the entry that actually owns its state.

    Coordinates are environment-local; rigid quaternions in dataset shards are
    WXYZ. Native Isaac/Newton poses are XYZW and converted explicitly here.
    """

    def __init__(self, env, particle_samples: int = 468, include_material: bool = True):
        import warp as wp
        from isaaclab_newton.physics import NewtonManager

        self.env = env
        self.wp = wp
        self.entry = NewtonManager._solver._entries["media"]
        self.solver = self.entry.solver
        self.include_material = include_material
        self.count = env._num_particles
        sample_count = min(particle_samples, self.count)
        if sample_count < 1:
            raise ValueError("particle_samples must be positive")
        # Fixed indices preserve particle identity through each trajectory.
        self.indices = torch.linspace(0, self.count - 1, sample_count, device=env.device).round().long()
        local_to_global = wp.to_torch(self.entry.particle_local_to_global).long()
        if not torch.equal(torch.sort(local_to_global).values, torch.arange(len(local_to_global), device=env.device)):
            raise ValueError("Franka adapter requires the media entry to own every particle.")
        self.global_to_local = torch.argsort(local_to_global)
        model = NewtonManager.get_model()
        self.mass = wp.to_torch(model.particle_mass).reshape(env.num_envs, self.count)
        self.radius = wp.to_torch(model.particle_radius).reshape(env.num_envs, self.count)

    def _material(self) -> tuple[torch.Tensor, torch.Tensor]:
        rows = []
        mpm = self.entry.state_0.mpm
        valid = self.solver.fidelity == "mpm"
        for name, width in (
            ("particle_elastic_strain", 9),
            ("particle_Jp", 1),
            ("particle_qd_grad", 9),
            ("particle_stress", 9),
        ):
            value = getattr(mpm, name, None)
            if value is None:
                valid = False
                flat = torch.zeros((self.env.num_envs * self.count, width), device=self.env.device)
            else:
                flat = self.wp.to_torch(value).reshape(-1, width)[self.global_to_local]
            rows.append(flat.reshape(self.env.num_envs, self.count, width)[:, self.indices])
        material = torch.cat(rows, dim=-1)
        mask = torch.full(material.shape[:2], valid, dtype=torch.bool, device=self.env.device)
        mask &= torch.isfinite(material).all(dim=-1)
        return material, mask

    def capture(self, groups: dict | None = None) -> dict[str, torch.Tensor]:
        """Read one continuous physical state without sanitizing failed particles."""
        env = self.env
        if groups is None:
            groups = env.observation_manager.compute()
        observation, critic = policy_observations(groups)
        position = env._media.data.particle_pos_w.torch - env.env_origins[:, None]
        velocity = env._media.data.particle_vel_w.torch
        particle = torch.cat((position, velocity), dim=-1)
        poses, twists = [], []
        for asset in (env._source_cup, env._target_cup):
            poses.append(asset.data.root_link_pose_w.torch)
            twists.append(asset.data.root_link_vel_w.torch)
        poses.append(env._robot.data.body_link_pose_w.torch[:, env._tcp_body_idx])
        twists.append(env._robot.data.body_link_vel_w.torch[:, env._tcp_body_idx])
        rigid = []
        for pose, twist in zip(poses, twists):
            rigid.append(torch.cat((pose[:, :3] - env.env_origins, pose[:, 6:7], pose[:, 3:6], twist), dim=-1))
        arm = env.action_manager.get_term("arm_action")
        gripper = env.action_manager.get_term("gripper_action")
        controller = torch.cat(
            (
                env._robot.data.joint_pos.torch,
                env._robot.data.joint_vel.torch,
                arm._previous_delta,
                gripper._processed_actions,
                env.action_manager.action,
                env._success_dwell_count[:, None],
                env._lost_grasp_dwell_count[:, None],
                env._lifted_grasp_seen[:, None],
                env.episode_length_buf[:, None],
            ),
            dim=-1,
        ).float()
        source, target, spill = env._particle_region_masks()
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        summary = torch.cat(
            (
                position.mean(dim=1),
                velocity.mean(dim=1),
                (0.5 * self.mass * speed.square()).sum(dim=1, keepdim=True),
                torch.stack((source.float().mean(1), target.float().mean(1), spill.float().mean(1)), dim=-1),
                speed.amax(dim=1, keepdim=True),
                position[..., 2].amin(1, keepdim=True),
                position[..., 2].amax(1, keepdim=True),
            ),
            dim=-1,
        )
        result = {
            "observation": observation,
            "critic_observation": critic,
            "particle_state": particle[:, self.indices],
            "rigid_state": torch.stack(rigid, dim=1),
            "controller_state": controller,
            "particle_summary": summary,
        }
        finite = torch.isfinite(particle).all(dim=(1, 2))
        finite &= torch.isfinite(result["rigid_state"]).all(dim=(1, 2))
        finite &= torch.isfinite(controller).all(dim=1)
        finite &= torch.isfinite(observation).all(dim=1) & torch.isfinite(critic).all(dim=1)
        if self.include_material:
            result["material_state"], result["material_state_valid"] = self._material()
            if self.solver.fidelity == "mpm":
                finite &= result["material_state_valid"].all(dim=1)
        result["finite_mask"] = finite
        return {key: value.detach().clone() for key, value in result.items()}

    def episode_metadata(self, world_id: int, seed: int) -> dict:
        """Return reusable material/geometry context and an immutable split group."""
        env = self.env
        from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import _resolve_pour_solver_tree

        tree = _resolve_pour_solver_tree(env.cfg)
        cfg = tree.media_solver
        helper = self.solver._particle_solver
        names = [
            "policy_dt_s",
            "native_dt_s",
            "voxel_size_m",
            "particle_mass_kg",
            "particle_radius_m",
            "pair_friction",
            "pair_cohesion",
            "relaxation",
            "compliance_m_per_N",
            "full_mpm",
            "source_fill_level",
            "target_fraction",
        ]
        values = [
            env.step_dt,
            env.cfg.sim.dt / env.cfg.sim.physics.num_substeps / tree.media_entry.substeps,
            cfg.voxel_size,
            float(self.mass[world_id, 0]),
            float(self.radius[world_id, 0]),
            float(self.solver.model.particle_mu),
            float(self.solver.model.particle_cohesion),
            cfg.particle_relaxation,
            float(getattr(helper, "compliance", 0.0)),
            float(self.solver.fidelity == "mpm"),
            env.cfg.source_fill_level,
            float(env.pour_target_frac[world_id]),
        ]
        material = env.cfg.scene.media.spawn.material
        for name in (
            "density",
            "young_modulus",
            "poisson_ratio",
            "viscosity",
            "friction",
            "damping",
            "yield_pressure",
            "tensile_yield_ratio",
            "yield_stress",
            "hardening",
            "dilatancy",
        ):
            names.append("material_" + name)
            values.append(float(getattr(material, name)))
        names.extend(("material_law_granular", "coupling_staggered_proxy", "proxy_mass_scale"))
        values.extend((1.0, 1.0, float(tree.proxy.mass_scale)))
        for name, bounds in (
            ("source_inner", (env._source_inner_lo, env._source_inner_hi)),
            ("target_inner", (env._target_inner_lo, env._target_inner_hi)),
        ):
            for side, vector in zip(("lo", "hi"), bounds):
                names.extend(f"{name}_{side}_{axis}" for axis in "xyz")
                values.extend(float(item) for item in vector)
        names.extend(("material_law_liquid", "coupling_one_way_prescribed_boundary"))
        values.extend((0.0, 0.0))
        assert tuple(names) == CONTEXT_NAMES
        row = int(env.reset_dataset_row_id[world_id])
        return {
            "task_id": "franka_pour",
            "reset_group": f"franka_pour:row:{row}",
            "reset_row": row,
            "seed": seed,
            "dt": float(env.step_dt),
            "max_episode_length": int(env.max_episode_length),
            "context": torch.tensor(values, dtype=torch.float32),
            "context_names": names,
            "particle_indices": self.indices.cpu(),
            "particle_mass": self.mass[world_id, self.indices].cpu().clone(),
            "particle_radius": self.radius[world_id, self.indices].cpu().clone(),
            "particle_weights": self.mass[world_id, self.indices].cpu().clone() * (self.count / len(self.indices)),
            "particle_weight_units": "kg",
            "particle_count": self.count,
            "rigid_names": ["source_cup", "target_cup", "hand_link"],
            "quaternion_order": "wxyz",
            "frame": "env_local_z_up",
            "coordinate_frame": "environment_local",
            "material_law": "granular_drucker_prager",
            "coupling_mode": "staggered_two_way_proxy",
            "material_state_names": ["F:9", "Jp:1", "velocity_gradient:9", "stress:9"],
        }
