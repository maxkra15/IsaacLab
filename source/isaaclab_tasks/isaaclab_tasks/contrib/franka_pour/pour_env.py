# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based RL environment for a Franka pouring MPM media between two cups.

The visible dynamic source cup and kinematic receiving cup are scene-owned rigid objects. Their
USD bowl meshes are visual-only, while the source also owns an invisible rigid grasp proxy for
Newton-generated finger contacts. A narrow per-world Newton hook attaches cached hollow
particle-only colliders to both scene bodies and adds only two hidden solver objects: a rigid-only
receiving-cup copy and a particle-only spill floor.

A Newton :class:`CoupledSolverCfg` advances the robot and source cup in the ``arm`` MJWarp entry and
the particles, receiving cup, and spill floor in the implicit ``media`` entry. Proxy coupling makes
the source cup's particle collider available to MPM without assigning one body to two entries. Arm
control is relative DiffIK with a binary gripper action; all observable and reset state flows through
the scene assets' public APIs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import newton
import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import math as math_utils

from .cube_bowl_mesh import cube_bowl_inner_bounds, make_cube_bowl_mesh
from .cup_media import cup_cavity_lattice
from .mdp.terminations import _state_finite
from .reset_utils import boolean_selection_mask

if TYPE_CHECKING:
    from isaaclab_newton.assets import MPMObject

    from .pour_env_cfg import FrankaPourEnvCfg

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]


class FrankaPourEnv(ManagerBasedRLEnv):
    """Franka grasping a dynamic cup of MPM media (Newton proxy-coupled MPM), pouring by tilting."""

    cfg: FrankaPourEnvCfg

    def __init__(self, cfg: FrankaPourEnvCfg, render_mode: str | None = None, **kwargs):
        resolved_cfg = cfg.finalize()
        self._prepare_newton_extras(resolved_cfg)
        self._install_newton_builder_hook()
        try:
            super().__init__(resolved_cfg, render_mode, **kwargs)
        finally:
            self._remove_newton_builder_hook()

    def load_managers(self) -> None:
        self._setup_after_physics()
        super().load_managers()

    # ------------------------------------------------------------------ build
    def _prepare_newton_extras(self, cfg: FrankaPourEnvCfg) -> None:
        """Bake task-local Newton collision geometry from the resolved scene config.

        Runs before ``super().__init__`` so the per-world builder hook has the
        geometry and contact values available while the scene is imported.
        """
        # Watertight cube-cup collision meshes. The source's outer extents exactly match the solid
        # grasp box, so its visible walls and the rigid finger contacts no longer disagree.
        self._cup_vertices, self._cup_indices = make_cube_bowl_mesh(
            inner_width=float(cfg.source_cup_inner_width),
            inner_depth=float(cfg.source_cup_inner_depth),
            wall_thickness=float(cfg.source_cup_wall_thickness),
            cavity_depth=float(cfg.source_cup_cavity_depth),
            bottom_thickness=float(cfg.source_cup_bottom_thickness),
        )
        self._target_vertices, self._target_indices = make_cube_bowl_mesh(
            inner_width=float(cfg.target_cup_inner_width),
            inner_depth=float(cfg.target_cup_inner_depth),
            wall_thickness=float(cfg.target_cup_wall_thickness),
            cavity_depth=float(cfg.target_cup_cavity_depth),
            bottom_thickness=float(cfg.target_cup_bottom_thickness),
        )
        self._source_collider_mesh = newton.Mesh(
            self._cup_vertices,
            self._cup_indices,
            compute_inertia=False,
            is_solid=False,
        )
        self._target_collider_mesh = newton.Mesh(
            self._target_vertices,
            self._target_indices,
            compute_inertia=False,
            is_solid=False,
        )
        self._source_inner_lo, self._source_inner_hi = cube_bowl_inner_bounds(
            cfg.source_cup_inner_width,
            cfg.source_cup_inner_depth,
            cfg.source_cup_cavity_depth,
            cfg.source_cup_bottom_thickness,
        )
        self._target_inner_lo, self._target_inner_hi = cube_bowl_inner_bounds(
            cfg.target_cup_inner_width,
            cfg.target_cup_inner_depth,
            cfg.target_cup_cavity_depth,
            cfg.target_cup_bottom_thickness,
        )

        # Cup-local media lattice (the env transforms it by the live cup pose every reset).
        self._media_local_points, _ = cup_cavity_lattice(cfg)

        # Cup reset pose (env frame): resting on the table, opening up (cup-local +z is world +z).
        self._cup_reset_pos = np.asarray(cfg.cup_reset_pos, dtype=np.float64)
        self._cup_reset_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._target_reset_pos = np.asarray(cfg.target_cup_reset_pos, dtype=np.float64)
        self._target_reset_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._grasp_contact_ke = float(cfg.grasp_contact_ke)
        self._grasp_contact_kd = float(cfg.grasp_contact_kd)
        self._grasp_contact_kf = float(cfg.grasp_contact_kf)
        self._grasp_contact_mu = float(cfg.cup_grasp_box_friction)

        self._source_cup_friction = float(cfg.source_cup_friction)
        self._target_cup_friction = float(cfg.target_cup_friction)
        self._collider_margin = float(cfg.collider_margin)
        self._builder_hook = None

    # --------------------------------------------------------------- hook
    def _install_newton_builder_hook(self) -> None:
        self._builder_hook = self._add_pour_world_to_builder
        NewtonManager.register_builder_world_hook(self._builder_hook)

    def _remove_newton_builder_hook(self) -> None:
        hook = getattr(self, "_builder_hook", None)
        if hook is None:
            return
        NewtonManager.unregister_builder_world_hook(hook)
        self._builder_hook = None

    def _add_pour_world_to_builder(self, builder, env_id: int, position, quaternion) -> None:
        """Add only solver-specific collision representations to one imported scene world."""
        env_root = f"/World/envs/env_{env_id}"
        body_ids = self._current_world_range(builder, "body", env_id)
        shape_ids = self._current_world_range(builder, "shape", env_id)
        self._disable_robot_particle_collision(builder, body_ids, shape_ids)
        self._configure_finger_contact_material(builder, body_ids, shape_ids)

        source_body = self._find_world_body(builder, body_ids, env_id, "SourceCup")
        target_body = self._find_world_body(builder, body_ids, env_id, "TargetCup")
        self._add_kinematic_rigid_object_articulation(builder, target_body)
        grasp_proxy = self._find_world_shape(
            builder,
            shape_ids,
            env_id,
            "/SourceCup/geometry/grasp_proxy",
            body_id=source_body,
        )
        self._configure_grasp_proxy(builder, grasp_proxy)

        self._add_particle_collider(
            builder,
            body_id=source_body,
            mesh=self._source_collider_mesh,
            friction=self._source_cup_friction,
            label=f"{env_root}/SourceCup/ParticleCollider",
        )
        self._add_particle_collider(
            builder,
            body_id=target_body,
            mesh=self._target_collider_mesh,
            friction=self._target_cup_friction,
            label=f"{env_root}/TargetCup/ParticleCollider",
        )

        world_xform = wp.transform(
            wp.vec3(*[float(value) for value in position]),
            wp.quat(*[float(value) for value in quaternion]),
        )
        target_local_xform = wp.transform(
            wp.vec3(*self._target_reset_pos.tolist()),
            wp.quat(*self._target_reset_quat.tolist()),
        )
        target_rigid = builder.add_body(
            xform=wp.transform_multiply(world_xform, target_local_xform),
            mass=0.0,
            is_kinematic=True,
            lock_inertia=True,
            label=f"{env_root}/TargetCupRigid",
        )
        self._add_rigid_collider(
            builder,
            body_id=target_rigid,
            mesh=self._target_collider_mesh,
            friction=self._target_cup_friction,
            label=f"{env_root}/TargetCupRigid/Collision",
        )

        spill_floor = builder.add_body(
            xform=world_xform,
            mass=0.0,
            inertia=wp.mat33(),
            is_kinematic=True,
            lock_inertia=True,
            label=f"{env_root}/SpillFloor",
        )
        spill_shape = builder.add_shape_plane(
            body=spill_floor,
            xform=wp.transform_identity(),
            width=0.0,
            length=0.0,
            cfg=newton.ModelBuilder.ShapeConfig(
                mu=0.8,
                margin=self._collider_margin,
                has_shape_collision=False,
                has_particle_collision=True,
            ),
            color=(0.3, 0.3, 0.3),
            label=f"{env_root}/SpillFloor/Collision",
        )
        self._set_shape_roles(builder, spill_shape, rigid=False, particles=True, visible=False)

    @staticmethod
    def _current_world_range(builder, prefix: str, env_id: int) -> range:
        """Return the contiguous tail added for the currently open Newton world."""
        worlds = getattr(builder, f"{prefix}_world", None)
        if worlds is None:
            raise RuntimeError(f"Newton builder does not expose {prefix}_world assignments.")
        stop = len(worlds)
        start = stop
        while start > 0 and int(worlds[start - 1]) == env_id:
            start -= 1
        if start == stop:
            raise RuntimeError(f"Newton builder contains no {prefix} entries for open world {env_id}.")
        return range(start, stop)

    def _find_world_body(self, builder, body_ids: range, env_id: int, body_name: str) -> int:
        """Resolve exactly one imported body by Newton world and exact final path component."""
        matches = [body_id for body_id in body_ids if str(builder.body_label[body_id]).rsplit("/", 1)[-1] == body_name]
        if len(matches) != 1:
            labels = [str(builder.body_label[index]) for index in matches]
            raise RuntimeError(
                f"Expected exactly one {body_name!r} body in Newton world {env_id}, "
                f"found ids={matches}, labels={labels}."
            )
        return matches[0]

    @staticmethod
    def _add_kinematic_rigid_object_articulation(builder, body_id: int) -> None:
        """Expose an imported kinematic body through Newton's articulation-based rigid view."""
        body_label = str(builder.body_label[body_id])
        child_joints = [joint_id for _, joint_id in builder.joint_parents.get(body_id, ())]
        if not child_joints:
            joint_id = builder.add_joint_free(child=body_id, label=f"{body_label}/FreeJoint")
            builder.add_articulation([joint_id], label=body_label)
        elif len(child_joints) == 1:
            joint_id = child_joints[0]
            articulation_id = int(builder.joint_articulation[joint_id])
            if articulation_id < 0 or str(builder.articulation_label[articulation_id]) != body_label:
                raise RuntimeError(
                    f"Kinematic rigid body {body_label!r} has an unexpected joint/articulation association."
                )
        else:
            raise RuntimeError(
                f"Kinematic rigid body {body_label!r} must have at most one root joint, found {child_joints}."
            )

        builder.body_flags[body_id] = int(newton.BodyFlags.KINEMATIC)
        builder.body_mass[body_id] = 0.0
        builder.body_inv_mass[body_id] = 0.0
        builder.body_inertia[body_id] = wp.mat33()
        builder.body_inv_inertia[body_id] = wp.mat33()

    def _find_world_shape(self, builder, shape_ids: range, env_id: int, label_suffix: str, *, body_id: int) -> int:
        """Resolve exactly one imported shape by owning body and exact scene-relative path."""
        matches = [
            shape_id
            for shape_id in shape_ids
            if int(builder.shape_body[shape_id]) == body_id
            and str(builder.shape_label[shape_id]).endswith(label_suffix)
        ]
        if len(matches) != 1:
            labels = [str(builder.shape_label[index]) for index in matches]
            raise RuntimeError(
                f"Expected exactly one shape ending in {label_suffix!r} on body {body_id} "
                f"in Newton world {env_id}, found ids={matches}, labels={labels}."
            )
        return matches[0]

    def _configure_grasp_proxy(self, builder, shape_id: int) -> None:
        """Keep the imported grasp proxy rigid-only, invisible, and contact-tuned."""
        self._set_shape_roles(builder, shape_id, rigid=True, particles=False, visible=False)
        builder.shape_margin[shape_id] = self._collider_margin
        builder.shape_material_ke[shape_id] = self._grasp_contact_ke
        builder.shape_material_kd[shape_id] = self._grasp_contact_kd
        builder.shape_material_kf[shape_id] = self._grasp_contact_kf
        builder.shape_material_mu[shape_id] = self._grasp_contact_mu

    def _add_particle_collider(
        self,
        builder,
        *,
        body_id: int,
        mesh: newton.Mesh,
        friction: float,
        label: str,
    ) -> int:
        """Attach an invisible hollow particle-only collider to a scene-owned body."""
        shape_id = builder.add_shape_mesh(
            body_id,
            xform=wp.transform_identity(),
            mesh=mesh,
            cfg=newton.ModelBuilder.ShapeConfig(
                mu=friction,
                density=0.0,
                margin=self._collider_margin,
                has_shape_collision=False,
                has_particle_collision=True,
                is_visible=False,
            ),
            label=label,
        )
        self._set_shape_roles(builder, shape_id, rigid=False, particles=True, visible=False)
        builder.shape_margin[shape_id] = self._collider_margin
        builder.shape_material_mu[shape_id] = friction
        return shape_id

    def _add_rigid_collider(
        self,
        builder,
        *,
        body_id: int,
        mesh: newton.Mesh,
        friction: float,
        label: str,
    ) -> int:
        """Attach an invisible hollow rigid-only collider to a solver-owned body."""
        shape_id = builder.add_shape_mesh(
            body_id,
            xform=wp.transform_identity(),
            mesh=mesh,
            cfg=newton.ModelBuilder.ShapeConfig(
                mu=friction,
                density=0.0,
                ke=self._grasp_contact_ke,
                kd=self._grasp_contact_kd,
                kf=self._grasp_contact_kf,
                margin=self._collider_margin,
                has_shape_collision=True,
                has_particle_collision=False,
                is_visible=False,
            ),
            label=label,
        )
        self._set_shape_roles(builder, shape_id, rigid=True, particles=False, visible=False)
        builder.shape_margin[shape_id] = self._collider_margin
        builder.shape_material_mu[shape_id] = friction
        return shape_id

    @staticmethod
    def _set_shape_roles(builder, shape_id: int, *, rigid: bool, particles: bool, visible: bool) -> None:
        flags = int(builder.shape_flags[shape_id])
        assignments = (
            (newton.ShapeFlags.COLLIDE_SHAPES, rigid),
            (newton.ShapeFlags.COLLIDE_PARTICLES, particles),
            (newton.ShapeFlags.VISIBLE, visible),
        )
        for flag, enabled in assignments:
            if enabled:
                flags |= int(flag)
            else:
                flags &= ~int(flag)
        builder.shape_flags[shape_id] = flags

    def _disable_robot_particle_collision(self, builder, body_ids: range, shape_ids: range) -> None:
        """The robot shapes must not collide with MPM particles (only the cup cavity mesh does)."""
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape_id in shape_ids:
            body_id = int(builder.shape_body[shape_id])
            if body_id not in body_ids:
                continue
            body_label = str(builder.body_label[body_id])
            if "/Robot/" in body_label or body_label.endswith("/Robot"):
                builder.shape_flags[shape_id] &= ~collide_particles

    def _configure_finger_contact_material(self, builder, body_ids: range, shape_ids: range) -> None:
        """Use a rigid contact material on the two finger collision shapes.

        Applying the same material to the fingers and cup keeps the intended pair response
        independent of import-side material defaults.
        """
        finger_suffixes = ("/panda_leftfinger", "/panda_rightfinger")
        for shape_id in shape_ids:
            body_id = int(builder.shape_body[shape_id])
            if body_id not in body_ids:
                continue
            if str(builder.body_label[body_id]).endswith(finger_suffixes):
                builder.shape_material_ke[shape_id] = self._grasp_contact_ke
                builder.shape_material_kd[shape_id] = self._grasp_contact_kd
                builder.shape_material_kf[shape_id] = self._grasp_contact_kf
                builder.shape_material_mu[shape_id] = self._grasp_contact_mu

    # ----------------------------------------------------------- post-physics
    def _setup_after_physics(self) -> None:
        dev = self.device
        self._robot = self.scene["robot"]
        self._source_cup = self.scene["source_cup"]
        self._target_cup = self.scene["target_cup"]
        self._media: MPMObject = self.scene["media"]

        self._arm_joint_ids, _ = self._robot.find_joints(ARM_JOINTS, preserve_order=True)
        self._finger_joint_ids, _ = self._robot.find_joints(FINGER_JOINTS, preserve_order=True)
        tcp_body_ids, _ = self._robot.find_bodies(self.cfg.actions.arm_action.body_name)
        if len(tcp_body_ids) != 1:
            raise RuntimeError(
                f"Expected one TCP parent body named {self.cfg.actions.arm_action.body_name!r}, "
                f"found {len(tcp_body_ids)}."
            )
        self._tcp_body_idx = tcp_body_ids[0]
        tcp_offset = self.cfg.actions.arm_action.body_offset
        if tcp_offset is None:
            self._tcp_offset_pos = None
            self._tcp_offset_quat = None
        else:
            self._tcp_offset_pos = torch.tensor(tcp_offset.pos, device=dev).repeat(self.num_envs, 1)
            self._tcp_offset_quat = torch.tensor(tcp_offset.rot, device=dev).repeat(self.num_envs, 1)

        self.env_origins = self.scene.env_origins.to(device=dev, dtype=torch.float32)
        self._num_particles = int(self._media.particles_per_object)
        self._media_local_points_t = torch.as_tensor(self._media_local_points, device=dev, dtype=torch.float32)
        self._default_arm_q = self._robot.data.default_joint_pos.torch[:, self._arm_joint_ids].clone()
        self._source_inner_lo_t = torch.as_tensor(self._source_inner_lo, device=dev)
        self._source_inner_hi_t = torch.as_tensor(self._source_inner_hi, device=dev)
        self._target_inner_lo_t = torch.as_tensor(self._target_inner_lo, device=dev)
        self._target_inner_hi_t = torch.as_tensor(self._target_inner_hi, device=dev)
        self._containment_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._containment_cache_step = -1

    # ----------------------------------------------------------- poses / obs
    def _pose_w_to_e(self, pose_w: torch.Tensor) -> torch.Tensor:
        """Convert a public world-frame pose view to a finite environment-frame pose."""
        pos = torch.nan_to_num(pose_w[:, :3], nan=0.0, posinf=0.0, neginf=0.0) - self.env_origins
        quat = pose_w[:, 3:7]
        norm = torch.linalg.norm(torch.nan_to_num(quat), dim=-1, keepdim=True)
        ident = torch.zeros_like(quat)
        ident[:, 3] = 1.0
        quat = torch.where(norm > 1e-6, quat / torch.clamp(norm, min=1e-6), ident)
        return torch.cat((pos, quat), dim=-1)

    def ee_pose_e(self) -> torch.Tensor:
        """End-effector (panda_hand) pose in the env frame: ``(num_envs, 7)`` pos + xyzw quat."""
        return self._pose_w_to_e(self._robot.data.body_link_pose_w.torch[:, self._tcp_body_idx])

    def cup_pose_e(self) -> torch.Tensor:
        """Cup body pose in the env frame: ``(num_envs, 7)`` pos + xyzw quat."""
        return self._pose_w_to_e(self._source_cup.data.root_link_pose_w.torch)

    def target_pose_e(self) -> torch.Tensor:
        """Receiving-cup pose in the env frame: ``(num_envs, 7)`` pos + xyzw quat."""
        return self._pose_w_to_e(self._target_cup.data.root_link_pose_w.torch)

    def tcp_pose_e(self) -> torch.Tensor:
        """DiffIK-controlled tool-centre pose in the robot-root/env frame."""
        body_pose_w = self._robot.data.body_link_pose_w.torch[:, self._tcp_body_idx]
        root_pose_w = self._robot.data.root_link_pose_w.torch
        pos, quat = math_utils.subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], body_pose_w[:, :3], body_pose_w[:, 3:7]
        )
        if self._tcp_offset_pos is not None:
            pos, quat = math_utils.combine_frame_transforms(pos, quat, self._tcp_offset_pos, self._tcp_offset_quat)
        return torch.cat((torch.nan_to_num(pos), torch.nan_to_num(quat)), dim=-1)

    def tcp_pos_e(self) -> torch.Tensor:
        return self.tcp_pose_e()[:, :3]

    def cup_grasp_point_e(self) -> torch.Tensor:
        """World-facing grasp point at the middle of the source cup walls, in env coordinates."""
        pose = self.cup_pose_e()
        offset = torch.zeros((self.num_envs, 3), device=self.device)
        offset[:, 2] = float(self.cfg.cup_grasp_height)
        return pose[:, :3] + math_utils.quat_apply(pose[:, 3:7], offset)

    def gripper_width(self) -> torch.Tensor:
        """Distance represented by the two symmetric Panda finger joint positions [m]."""
        return self._robot.data.joint_pos.torch[:, self._finger_joint_ids].sum(dim=-1)

    @property
    def gripper_open_width(self) -> float:
        return 2.0 * float(self.cfg.gripper_open_pos)

    @property
    def gripper_grasp_width(self) -> float:
        return 2.0 * float(self.cfg.cup_grasp_box_half[1])

    @property
    def cup_reset_height(self) -> float:
        return float(self.cfg.cup_reset_pos[2])

    @property
    def num_particles(self) -> int:
        return self._num_particles

    @property
    def pour_target_frac(self) -> float:
        return float(self.cfg.pour_target_frac)

    def particle_pos_e(self) -> torch.Tensor:
        """Per-env MPM particle positions in env coordinates, shape ``(N, P, 3)``."""
        return self._media.data.particle_pos_w.torch - self.env_origins[:, None, :]

    def _points_inside_cup(
        self, points_e: torch.Tensor, pose_e: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor
    ) -> torch.Tensor:
        rel = points_e - pose_e[:, None, :3]
        quat = pose_e[:, None, 3:7].expand(-1, points_e.shape[1], -1)
        local = math_utils.quat_apply_inverse(quat, rel)
        margin = float(self.cfg.particle_count_margin)
        return ((local >= lo - margin) & (local <= hi + margin)).all(dim=-1)

    def _containment_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Boolean ``(source, target)`` particle masks, cached within one manager step."""
        step = int(getattr(self, "common_step_counter", -1))
        if self._containment_cache is not None and self._containment_cache_step == step:
            return self._containment_cache
        points = self.particle_pos_e()
        source = self._points_inside_cup(points, self.cup_pose_e(), self._source_inner_lo_t, self._source_inner_hi_t)
        target = self._points_inside_cup(points, self.target_pose_e(), self._target_inner_lo_t, self._target_inner_hi_t)
        self._containment_cache = (source, target)
        self._containment_cache_step = step
        return source, target

    def count_in_source(self) -> torch.Tensor:
        return self._containment_masks()[0].sum(dim=1).float()

    def count_in_target(self) -> torch.Tensor:
        return self._containment_masks()[1].sum(dim=1).float()

    def state_finite(self) -> torch.Tensor:
        """Per-env instability guard over robot, source cup, and MPM media state."""
        return _state_finite(
            self._robot.data.joint_pos.torch,
            self._source_cup.data.root_link_pose_w.torch,
            self._media.data.particle_pos_w.torch,
        )

    # ----------------------------------------------------------- reset
    def reset_pour_scene(self, env_ids: torch.Tensor) -> None:
        """Reset the arm, source cup, and particles through their public asset APIs."""
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()

        arm_q = self._default_arm_q[env_ids].clone()
        zero_arm_velocity = torch.zeros_like(arm_q)
        self._robot.write_joint_position_to_sim_index(
            position=arm_q,
            joint_ids=self._arm_joint_ids,
            env_ids=env_ids,
        )
        self._robot.write_joint_velocity_to_sim_index(
            velocity=zero_arm_velocity,
            joint_ids=self._arm_joint_ids,
            env_ids=env_ids,
        )
        self._robot.set_joint_position_target_index(
            target=arm_q,
            joint_ids=self._arm_joint_ids,
            env_ids=env_ids,
        )

        finger_open = torch.full((n, len(FINGER_JOINTS)), float(self.cfg.gripper_open_pos), device=self.device)
        self._robot.write_joint_position_to_sim_index(
            position=finger_open,
            joint_ids=self._finger_joint_ids,
            env_ids=env_ids,
        )
        self._robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(finger_open),
            joint_ids=self._finger_joint_ids,
            env_ids=env_ids,
        )
        self._robot.set_joint_position_target_index(
            target=finger_open,
            joint_ids=self._finger_joint_ids,
            env_ids=env_ids,
        )

        cup_pos = torch.as_tensor(self._cup_reset_pos, device=self.device, dtype=torch.float32)
        cup_quat = torch.as_tensor(self._cup_reset_quat, device=self.device, dtype=torch.float32)
        cup_world = cup_pos.unsqueeze(0) + self.env_origins[env_ids]
        cup_pose = torch.cat((cup_world, cup_quat.expand(n, -1)), dim=-1)
        self._source_cup.write_root_pose_to_sim_index(root_pose=cup_pose, env_ids=env_ids)
        self._source_cup.write_root_velocity_to_sim_index(
            root_velocity=cup_pose.new_zeros((n, 6)),
            env_ids=env_ids,
        )

        # Public root/joint writers invalidate FK. Reading a public body-pose view consumes
        # the accumulated articulation mask, making all dirtied body poses authoritative
        # before solver caches and source-cup proxy transforms are refreshed. Priming the
        # robot view also prevents its next observation from issuing a redundant FK launch.
        _ = self._robot.data.body_link_pose_w

        new_p = self._sample_cup_media(cup_world, cup_quat.expand(n, -1))
        self._media.write_particle_pos_to_sim_index(new_p, env_ids=env_ids)
        self._media.write_particle_velocity_to_sim_index(torch.zeros_like(new_p), env_ids=env_ids)
        NewtonManager.reset_solver_state(
            world_mask=wp.from_torch(boolean_selection_mask(self.num_envs, env_ids), dtype=wp.bool),
            flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE,
        )
        self._containment_cache = None
        self._containment_cache_step = -1

    def _sample_cup_media(self, cup_pos: torch.Tensor, cup_quat: torch.Tensor) -> torch.Tensor:
        """Transform the local media lattice into selected cup poses on the simulation device."""
        particle_count = self._media_local_points_t.shape[0]
        local_points = self._media_local_points_t.unsqueeze(0).expand(cup_pos.shape[0], -1, -1)
        quaternions = cup_quat.unsqueeze(1).expand(-1, particle_count, -1)
        return math_utils.quat_apply(quaternions, local_points) + cup_pos.unsqueeze(1)
