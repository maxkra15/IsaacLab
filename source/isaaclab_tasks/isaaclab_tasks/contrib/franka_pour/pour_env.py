# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based RL environment for a Franka pouring MPM media between two cups.

The visible dynamic source cup and kinematic receiving cup are scene-owned rigid objects. Their
USD bowl meshes are visual-only, while the source also owns an invisible rigid grasp proxy for
Newton-generated finger contacts. A narrow per-world Newton hook attaches cached hollow
particle-only colliders to both scene bodies and adds only one hidden solver object: a particle-only
spill floor.

A Newton :class:`~isaaclab_contrib.coupling.CoupledProxySolverCfg` advances the robot and both cups
in the ``arm`` MJWarp entry and the particles and spill floor in the implicit ``media`` entry. Proxy
coupling makes both cups' particle colliders available to MPM without assigning one body to two
entries. The policy commands arm joint positions and a continuous symmetric finger target; all
observable and reset state flows through the scene assets' public APIs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import newton
import numpy as np
import torch
import warp as wp
from isaaclab_newton.ik.newton_ik_objectives_cfg import NewtonIKJointLimitObjectiveCfg, NewtonIKPoseObjectiveCfg
from isaaclab_newton.ik.newton_ik_solver import NewtonIKSolver
from isaaclab_newton.ik.newton_ik_solver_cfg import NewtonIKSolverCfg
from isaaclab_newton.physics import NewtonManager

import isaaclab.sim as sim_utils
from isaaclab.cloner import resolve_clone_plan_source
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import math as math_utils

from .cube_bowl_mesh import cube_bowl_inner_bounds, make_cube_bowl_mesh
from .cup_media import cup_cavity_lattice
from .mdp.terminations import (
    _delivered_particle_mask,
    _particles_in_workspace,
    _rigid_state_in_bounds,
    _spilled_particle_mask,
    _state_finite,
)
from .reset_utils import (
    balanced_cyclic_permutations,
    boolean_selection_mask,
    randomization_extent_index_pools,
    sample_index_pools,
    target_xy_behind_source,
)

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
        self._grasp_contact_ke = float(cfg.grasp_contact_ke)
        self._grasp_contact_kd = float(cfg.grasp_contact_kd)
        self._grasp_contact_kf = float(cfg.grasp_contact_kf)
        self._grasp_contact_mu = float(cfg.cup_grasp_box_friction)

        self._source_cup_friction = float(cfg.source_cup_friction)
        self._target_cup_friction = float(cfg.target_cup_friction)
        self._collider_margin = float(cfg.collider_margin)
        self._particle_max_velocity = float(cfg.particle_max_velocity)
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
        builder.particle_max_velocity = self._particle_max_velocity
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

        self._add_rigid_collider(
            builder,
            body_id=target_body,
            mesh=self._target_collider_mesh,
            friction=self._target_cup_friction,
            label=f"{env_root}/TargetCup/Collision",
        )

        world_xform = wp.transform(
            wp.vec3(*[float(value) for value in position]),
            wp.quat(*[float(value) for value in quaternion]),
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
        self._joint_pos_limits_t = self._robot.data.joint_pos_limits.torch.clone()
        tcp_body_ids, _ = self._robot.find_bodies(self.cfg.tcp_body_name)
        if len(tcp_body_ids) != 1:
            raise RuntimeError(
                f"Expected one TCP parent body named {self.cfg.tcp_body_name!r}, found {len(tcp_body_ids)}."
            )
        self._tcp_body_idx = tcp_body_ids[0]
        self._tcp_offset_pos = torch.tensor(self.cfg.tcp_offset_pos, device=dev).repeat(self.num_envs, 1)
        self._tcp_offset_quat = torch.tensor(self.cfg.tcp_offset_rot, device=dev).repeat(self.num_envs, 1)

        self.env_origins = self.scene.env_origins.to(device=dev, dtype=torch.float32)
        self._num_particles = int(self._media.particles_per_object)
        self._media_local_points_t = torch.as_tensor(self._media_local_points, device=dev, dtype=torch.float32)
        self._particle_workspace_lower_t = torch.as_tensor(
            self.cfg.particle_workspace_lower_bound, device=dev, dtype=torch.float32
        )
        self._particle_workspace_upper_t = torch.as_tensor(
            self.cfg.particle_workspace_upper_bound, device=dev, dtype=torch.float32
        )
        self._curriculum_arm_q_t = torch.as_tensor(
            (
                self.cfg.curriculum_pour_arm_q,
                self.cfg.curriculum_carry_arm_q,
                self.cfg.arm_home,
                self.cfg.arm_home,
                self.cfg.arm_home,
            ),
            device=dev,
            dtype=torch.float32,
        )
        self._curriculum_cup_quat_t = torch.zeros((len(self.cfg.curriculum_stage_names), 4), device=dev)
        self._curriculum_cup_quat_t[:, 3] = 1.0
        contact_position = float(self.cfg.cup_grasp_box_half[1])
        self._curriculum_finger_pos_t = torch.tensor(
            (
                contact_position,
                contact_position,
                float(self.cfg.gripper_open_pos),
                float(self.cfg.gripper_open_pos),
                float(self.cfg.gripper_open_pos),
            ),
            device=dev,
        )
        arm_action_cfg = self.cfg.actions.arm_action
        if hasattr(arm_action_cfg, "waypoint_phases"):
            waypoint_phases = arm_action_cfg.waypoint_phases
            curriculum_reference_phases = (
                waypoint_phases[arm_action_cfg.align_waypoint],
                waypoint_phases[arm_action_cfg.lift_waypoint],
                waypoint_phases[arm_action_cfg.grasp_waypoint],
                waypoint_phases[0],
                waypoint_phases[0],
            )
        else:
            # The operator-only controller does not consume trajectory phases.
            curriculum_reference_phases = (0.0,) * len(self.cfg.curriculum_stage_names)
        self._curriculum_reference_phase_t = torch.tensor(curriculum_reference_phases, device=dev)
        self._nominal_reference_waypoints_t = torch.as_tensor(
            (
                self.cfg.arm_home,
                self.cfg.arm_home,
                self.cfg.arm_home,
                self.cfg.curriculum_carry_arm_q,
                self.cfg.curriculum_pour_arm_q,
                self.cfg.curriculum_pour_target_arm_q,
            ),
            device=dev,
            dtype=torch.float32,
        )
        self._grasp_stage_index = self.cfg.curriculum_stage_names.index("grasp")
        self._full_stage_index = self.cfg.curriculum_stage_names.index("full")
        self._randomized_stage_index = self.cfg.curriculum_stage_names.index("randomized")
        self._build_randomized_reset_bank()
        start_stage = int(self.cfg.curriculum_start_stage)
        start_randomization_level = int(self.cfg.curriculum_randomization_start_level)
        self.curriculum_stage = torch.full((self.num_envs,), start_stage, device=dev, dtype=torch.long)
        self.curriculum_randomization_level = torch.full(
            (self.num_envs,),
            start_randomization_level,
            device=dev,
            dtype=torch.long,
        )
        self.pour_target_frac = torch.full(
            (self.num_envs,),
            float(self.cfg.curriculum_target_frac[start_stage]),
            device=dev,
        )
        self.episode_succeeded = torch.zeros(self.num_envs, device=dev, dtype=torch.bool)
        self.ep_max_target_frac = torch.zeros(self.num_envs, device=dev)
        self._success_dwell_count = torch.zeros(self.num_envs, device=dev, dtype=torch.long)
        self._lost_grasp_dwell_count = torch.zeros(self.num_envs, device=dev, dtype=torch.long)
        self._target_entry_seen = torch.zeros(
            (self.num_envs, self.num_particles),
            device=dev,
            dtype=torch.bool,
        )
        self._held_delivered = torch.zeros_like(self._target_entry_seen)
        self._held_delivery_tracker_step = -1
        self._source_inner_lo_t = torch.as_tensor(self._source_inner_lo, device=dev)
        self._source_inner_hi_t = torch.as_tensor(self._source_inner_hi, device=dev)
        self._target_inner_lo_t = torch.as_tensor(self._target_inner_lo, device=dev)
        self._target_inner_hi_t = torch.as_tensor(self._target_inner_hi, device=dev)
        self._particle_region_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._particle_region_cache_step = -1

    def _build_randomized_reset_bank(self) -> None:
        """Build a small Newton-IK bank for collision-safe randomized pre-grasp resets."""
        plan = sim_utils.SimulationContext.instance().get_clone_plan()
        resolved = resolve_clone_plan_source(self._robot.cfg.prim_path, plan) if plan is not None else None
        if resolved is None:
            raise RuntimeError(f"Could not resolve clone-plan source for {self._robot.cfg.prim_path!r}.")
        source_path = resolved[0]
        model = NewtonManager.get_clone_prototype_model(source_path)

        hand_matches = [
            body_id
            for body_id, label in enumerate(model.body_label)
            if str(label).rsplit("/", 1)[-1] == self.cfg.tcp_body_name
        ]
        if len(hand_matches) != 1:
            raise RuntimeError(
                f"Expected one {self.cfg.tcp_body_name!r} body in the IK prototype, found {hand_matches}."
            )
        hand_id = hand_matches[0]
        joint_labels = [str(label).rsplit("/", 1)[-1] for label in model.joint_label]
        joint_q_start = wp.to_torch(model.joint_q_start).to(device=self.device, dtype=torch.long)

        def coordinate_id(joint_name: str) -> int:
            matches = [joint_id for joint_id, label in enumerate(joint_labels) if label == joint_name]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one {joint_name!r} joint in the IK prototype, found {matches}.")
            return int(joint_q_start[matches[0]].item())

        arm_coordinate_ids = torch.tensor(
            [coordinate_id(joint_name) for joint_name in ARM_JOINTS],
            device=self.device,
            dtype=torch.long,
        )
        finger_coordinate_ids = torch.tensor(
            [coordinate_id(joint_name) for joint_name in FINGER_JOINTS],
            device=self.device,
            dtype=torch.long,
        )

        def tcp_pose_for_arm_q(arm_q_values: tuple[float, ...]) -> torch.Tensor:
            """Evaluate one prototype arm configuration and return its TCP world pose."""
            joint_q = wp.to_torch(model.joint_q).to(device=self.device, dtype=torch.float32).clone()
            joint_q[arm_coordinate_ids] = torch.as_tensor(arm_q_values, device=self.device)
            joint_q[finger_coordinate_ids] = float(self.cfg.gripper_preload_pos)
            state = model.state()
            newton.eval_fk(
                model,
                wp.from_torch(joint_q.contiguous(), dtype=wp.float32),
                model.joint_qd,
                state,
            )
            hand_pose = wp.to_torch(state.body_q)[hand_id : hand_id + 1]
            tcp_pos, tcp_quat = math_utils.combine_frame_transforms(
                hand_pose[:, :3],
                hand_pose[:, 3:7],
                self._tcp_offset_pos[0:1],
                self._tcp_offset_quat[0:1],
            )
            return torch.cat((tcp_pos, tcp_quat), dim=-1)[0].clone()

        nominal_home_tcp_pose = tcp_pose_for_arm_q(self.cfg.arm_home)
        # The source cup is authored upright in the robot-base frame. Store the corresponding
        # nominal home TCP orientation as the desired cup-local grasp orientation so policy
        # observations remain correct when the source cup is yaw-randomized.
        self._desired_grasp_tcp_quat_c = math_utils.quat_unique(nominal_home_tcp_pose[3:7]).repeat(self.num_envs, 1)
        nominal_carry_tcp_pose = tcp_pose_for_arm_q(self.cfg.curriculum_carry_arm_q)
        nominal_pour_tcp_pose = tcp_pose_for_arm_q(self.cfg.curriculum_pour_arm_q)
        nominal_tilt_tcp_pose = tcp_pose_for_arm_q(self.cfg.curriculum_pour_target_arm_q)

        grid_size = int(self.cfg.curriculum_randomized_reset_ik_grid_size)
        source_range = torch.as_tensor(
            self.cfg.curriculum_randomized_source_position_range,
            device=self.device,
            dtype=torch.float32,
        )
        x_offsets = torch.linspace(-source_range[0], source_range[0], grid_size, device=self.device)
        y_offsets = torch.linspace(-source_range[1], source_range[1], grid_size, device=self.device)
        offset_x, offset_y = torch.meshgrid(x_offsets, y_offsets, indexing="ij")
        offsets = torch.stack((offset_x.flatten(), offset_y.flatten()), dim=-1)
        nominal_source = torch.as_tensor(self.cfg.cup_reset_pos, device=self.device)
        source_positions = nominal_source.repeat(offsets.shape[0], 1)
        source_positions[:, :2] += offsets

        samples_per_source = int(self.cfg.curriculum_randomized_reset_ik_samples_per_source)
        pair_count = samples_per_source // 2
        pair_index = torch.arange(pair_count, device=self.device, dtype=torch.float32) + 0.5
        pair_directions = (
            2.0
            * torch.stack(
                (
                    torch.frac(pair_index * 0.754877666),
                    torch.frac(pair_index * 0.569840296),
                    torch.frac(pair_index * 0.438447187),
                ),
                dim=-1,
            )
            - 1.0
        )
        paired_jitter = pair_directions * torch.as_tensor(
            self.cfg.curriculum_randomized_reset_tcp_jitter,
            device=self.device,
        )
        jitter_parts = [paired_jitter, -paired_jitter]
        if samples_per_source % 2:
            jitter_parts.insert(0, torch.zeros((1, 3), device=self.device))
        tcp_jitter_samples = torch.cat(jitter_parts, dim=0)

        # Pair upright source yaw with the symmetric TCP samples. Keeping both signs in each source
        # cell avoids a directional reset bias while retaining the existing compact bank size.
        yaw_range = float(self.cfg.curriculum_randomized_source_yaw_range)
        pair_yaws = (
            (torch.arange(pair_count, device=self.device, dtype=torch.float32) + 1.0) / max(pair_count, 1) * yaw_range
        )
        yaw_parts = [pair_yaws, -pair_yaws]
        if samples_per_source % 2:
            yaw_parts.insert(0, torch.zeros(1, device=self.device))
        source_yaw_samples = torch.cat(yaw_parts, dim=0)

        source_positions = source_positions.repeat_interleave(samples_per_source, dim=0)
        tcp_jitter = tcp_jitter_samples.repeat(offsets.shape[0], 1)
        # Rotate the yaw-to-jitter pairing in each source cell. Every cell retains identical yaw
        # and jitter marginals, while each fixed jitter sees every yaw globally within one count.
        source_yaws = balanced_cyclic_permutations(source_yaw_samples, offsets.shape[0]).reshape(-1)
        randomized_bank_size = source_positions.shape[0]

        # Stages two and three require the same five varied reach positions but an upright source.
        # Append dedicated solve-only rows so the randomized nominal-source rows keep all yaw values;
        # these rows are sliced out before constructing the stage-four bank and extent pools.
        source_positions = torch.cat((source_positions, nominal_source.repeat(samples_per_source, 1)), dim=0)
        tcp_jitter = torch.cat((tcp_jitter, tcp_jitter_samples), dim=0)
        source_yaws = torch.cat((source_yaws, torch.zeros(samples_per_source, device=self.device)), dim=0)
        bank_size = source_positions.shape[0]
        source_quaternions = torch.zeros((bank_size, 4), device=self.device)
        source_quaternions[:, 2] = torch.sin(0.5 * source_yaws)
        source_quaternions[:, 3] = torch.cos(0.5 * source_yaws)

        tcp_positions = source_positions.clone()
        tcp_positions[:, 2] += float(self.cfg.cup_grasp_height)
        tcp_positions += torch.as_tensor(
            self.cfg.curriculum_randomized_reset_tcp_standoff,
            device=self.device,
        )
        tcp_positions += tcp_jitter

        home_hand_pose = self._robot.data.body_link_pose_w.torch[0:1, self._tcp_body_idx]
        _, home_tcp_quat_w = math_utils.combine_frame_transforms(
            home_hand_pose[:, :3],
            home_hand_pose[:, 3:7],
            self._tcp_offset_pos[0:1],
            self._tcp_offset_quat[0:1],
        )
        target_positions_w = tcp_positions + self.env_origins[0]
        target_rotations_w = math_utils.quat_mul(
            source_quaternions,
            home_tcp_quat_w.expand(bank_size, -1),
        ).contiguous()

        target_name = "reset_tcp"
        solver = NewtonIKSolver(
            NewtonIKSolverCfg(
                optimizer="lm",
                jacobian_mode="analytic",
                sampler="none",
                n_seeds=1,
                iterations=int(self.cfg.curriculum_randomized_reset_ik_iterations),
                lambda_initial=0.1,
            ),
            model=model,
            num_envs=bank_size,
            device=str(model.device),
            objectives=[
                NewtonIKPoseObjectiveCfg(
                    body_name=self.cfg.tcp_body_name,
                    name=target_name,
                    body_offset_pos=self.cfg.tcp_offset_pos,
                    body_offset_rot=self.cfg.tcp_offset_rot,
                    # Grasp capture tolerates only millimetres of lateral error. Weight position
                    # strongly enough that the accepted IK cost cannot hide a centimetre-scale
                    # residual behind otherwise accurate orientation and joint-limit terms.
                    position_weight=100.0,
                    rotation_weight=5.0,
                ),
                # A hard post-solve margin check below enforces limits. Keeping this objective
                # secondary prevents it from trading centimetres of grasp error for extra margin.
                NewtonIKJointLimitObjectiveCfg(weight=1.0),
            ],
            link_resolver=lambda body_name: hand_id,
        )
        pose_objective = solver.objectives_by_name[target_name]
        pose_objective.position_objective.set_target_positions(
            wp.from_torch(target_positions_w.contiguous(), dtype=wp.vec3)
        )
        pose_objective.rotation_objective.set_target_rotations(
            wp.from_torch(target_rotations_w.contiguous(), dtype=wp.vec4)
        )

        seed = wp.to_torch(model.joint_q).to(device=self.device, dtype=torch.float32).repeat(bank_size, 1)
        seed[:, arm_coordinate_ids] = torch.as_tensor(self.cfg.arm_home, device=self.device)
        seed[:, finger_coordinate_ids] = float(self.cfg.gripper_open_pos)
        solved = wp.to_torch(solver.solve(wp.from_torch(seed.contiguous(), dtype=wp.float32))).clone()
        arm_q = solved[:, arm_coordinate_ids]
        costs = wp.to_torch(solver.costs).reshape(bank_size).clone()
        arm_limits = self._joint_pos_limits_t[0, self._arm_joint_ids]
        margin = torch.minimum(arm_q - arm_limits[:, 0], arm_limits[:, 1] - arm_q).amin(dim=-1)
        valid = (
            torch.isfinite(arm_q).all(dim=-1)
            & torch.isfinite(costs)
            & (costs <= float(self.cfg.curriculum_randomized_reset_ik_max_cost))
            & (margin >= float(self.cfg.curriculum_randomized_reset_ik_joint_margin))
        )
        if not bool(torch.all(valid)):
            first_invalid = int(torch.nonzero(~valid, as_tuple=False)[0])
            raise RuntimeError(
                "Newton IK must prevalidate every randomized Franka reset so source locations stay "
                f"uniformly represented; {int((~valid).sum())}/{bank_size} poses failed. "
                f"First invalid source={source_positions[first_invalid].tolist()}, "
                f"TCP={tcp_positions[first_invalid].tolist()}, cost={float(costs[first_invalid]):.6g}, "
                f"joint margin={float(margin[first_invalid]):.6g}."
            )

        # Route every randomized reset through a centered overhead pre-grasp before descending.
        # Direct joint interpolation from a laterally jittered reset pose to the cup can sweep a
        # finger through the light cup even while fully open; this intermediate pose keeps both
        # segments clear and makes the final descent nearly vertical.
        pregrasp_tcp_positions = source_positions.clone()
        pregrasp_tcp_positions[:, 2] += float(self.cfg.cup_grasp_height)
        pregrasp_tcp_positions += torch.as_tensor(
            self.cfg.curriculum_randomized_reset_tcp_standoff,
            device=self.device,
        )
        pregrasp_target_positions_w = pregrasp_tcp_positions + self.env_origins[0]
        pose_objective.position_objective.set_target_positions(
            wp.from_torch(pregrasp_target_positions_w.contiguous(), dtype=wp.vec3)
        )
        pregrasp_solved = wp.to_torch(solver.solve(wp.from_torch(solved.contiguous(), dtype=wp.float32))).clone()
        pregrasp_arm_q = pregrasp_solved[:, arm_coordinate_ids]
        pregrasp_costs = wp.to_torch(solver.costs).reshape(bank_size).clone()
        pregrasp_margin = torch.minimum(
            pregrasp_arm_q - arm_limits[:, 0],
            arm_limits[:, 1] - pregrasp_arm_q,
        ).amin(dim=-1)
        pregrasp_valid = (
            torch.isfinite(pregrasp_arm_q).all(dim=-1)
            & torch.isfinite(pregrasp_costs)
            & (pregrasp_costs <= float(self.cfg.curriculum_randomized_reset_ik_max_cost))
            & (pregrasp_margin >= float(self.cfg.curriculum_randomized_reset_ik_joint_margin))
        )
        if not bool(torch.all(pregrasp_valid)):
            first_invalid = int(torch.nonzero(~pregrasp_valid, as_tuple=False)[0])
            raise RuntimeError(
                "Newton IK must prevalidate every randomized Franka centered pre-grasp waypoint; "
                f"{int((~pregrasp_valid).sum())}/{bank_size} poses failed. "
                f"First invalid source={source_positions[first_invalid].tolist()}, "
                f"TCP={pregrasp_tcp_positions[first_invalid].tolist()}, "
                f"cost={float(pregrasp_costs[first_invalid]):.6g}, "
                f"joint margin={float(pregrasp_margin[first_invalid]):.6g}."
            )

        # Solve the paired grasp waypoint from the centered pre-grasp. Keeping every pose in the
        # same prevalidated bank avoids reset-time IK and preserves stationary action semantics.
        grasp_tcp_positions = source_positions.clone()
        grasp_tcp_positions[:, 2] += float(self.cfg.cup_grasp_height) - float(
            self.cfg.curriculum_grasp_descent_overshoot
        )
        grasp_target_positions_w = grasp_tcp_positions + self.env_origins[0]
        pose_objective.position_objective.set_target_positions(
            wp.from_torch(grasp_target_positions_w.contiguous(), dtype=wp.vec3)
        )
        grasp_solved = wp.to_torch(solver.solve(wp.from_torch(pregrasp_solved.contiguous(), dtype=wp.float32))).clone()
        grasp_arm_q = grasp_solved[:, arm_coordinate_ids]
        grasp_costs = wp.to_torch(solver.costs).reshape(bank_size).clone()
        grasp_margin = torch.minimum(
            grasp_arm_q - arm_limits[:, 0],
            arm_limits[:, 1] - grasp_arm_q,
        ).amin(dim=-1)
        grasp_valid = (
            torch.isfinite(grasp_arm_q).all(dim=-1)
            & torch.isfinite(grasp_costs)
            & (grasp_costs <= float(self.cfg.curriculum_randomized_reset_ik_max_cost))
            & (grasp_margin >= float(self.cfg.curriculum_randomized_reset_ik_joint_margin))
        )
        if not bool(torch.all(grasp_valid)):
            first_invalid = int(torch.nonzero(~grasp_valid, as_tuple=False)[0])
            raise RuntimeError(
                "Newton IK must prevalidate every randomized Franka grasp waypoint; "
                f"{int((~grasp_valid).sum())}/{bank_size} poses failed. "
                f"First invalid source={source_positions[first_invalid].tolist()}, "
                f"TCP={grasp_tcp_positions[first_invalid].tolist()}, "
                f"cost={float(grasp_costs[first_invalid]):.6g}, "
                f"joint margin={float(grasp_margin[first_invalid]):.6g}."
            )

        pair_index = torch.arange(randomized_bank_size, device=self.device, dtype=torch.float32) + 0.5
        unit_samples = torch.stack(
            (torch.frac(pair_index * 0.754877666), torch.frac(pair_index * 0.569840296)),
            dim=-1,
        )
        source_outer_half_x = self.cfg.source_cup_inner_width / 2.0 + self.cfg.source_cup_wall_thickness
        source_outer_half_y = self.cfg.source_cup_inner_depth / 2.0 + self.cfg.source_cup_wall_thickness
        target_outer_half_y = self.cfg.target_cup_inner_depth / 2.0 + self.cfg.target_cup_wall_thickness
        minimum_y_separation = (
            source_outer_half_x * torch.abs(torch.sin(source_yaws[:randomized_bank_size]))
            + source_outer_half_y * torch.abs(torch.cos(source_yaws[:randomized_bank_size]))
            + target_outer_half_y
            + float(self.cfg.curriculum_randomized_cup_clearance)
        )
        target_xy = target_xy_behind_source(
            source_positions[:randomized_bank_size, :2],
            target_center=self.cfg.curriculum_randomized_target_center_xy,
            target_half_range=self.cfg.curriculum_randomized_target_position_range,
            minimum_y_separation=minimum_y_separation,
            unit_samples=unit_samples,
        )
        target_positions = torch.as_tensor(self.cfg.target_cup_reset_pos, device=self.device).repeat(
            source_positions.shape[0], 1
        )
        target_positions[:randomized_bank_size, :2] = target_xy

        def solve_reference_waypoint(
            name: str,
            target_pose: torch.Tensor,
            initial_guess: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            pose_objective.position_objective.set_target_positions(
                wp.from_torch(target_pose[:, :3].contiguous(), dtype=wp.vec3)
            )
            pose_objective.rotation_objective.set_target_rotations(
                wp.from_torch(target_pose[:, 3:7].contiguous(), dtype=wp.vec4)
            )
            solved_full = wp.to_torch(solver.solve(wp.from_torch(initial_guess.contiguous(), dtype=wp.float32))).clone()
            solved_arm = solved_full[:, arm_coordinate_ids]
            solved_costs = wp.to_torch(solver.costs).reshape(bank_size).clone()
            solved_margin = torch.minimum(
                solved_arm - arm_limits[:, 0],
                arm_limits[:, 1] - solved_arm,
            ).amin(dim=-1)
            solved_valid = (
                torch.isfinite(solved_arm).all(dim=-1)
                & torch.isfinite(solved_costs)
                & (solved_costs <= float(self.cfg.curriculum_randomized_reset_ik_max_cost))
                & (solved_margin >= float(self.cfg.curriculum_randomized_reset_ik_joint_margin))
            )
            if not bool(torch.all(solved_valid)):
                first_invalid = int(torch.nonzero(~solved_valid, as_tuple=False)[0])
                raise RuntimeError(
                    f"Newton IK must prevalidate every randomized Franka {name} waypoint; "
                    f"{int((~solved_valid).sum())}/{bank_size} poses failed. "
                    f"First invalid target={target_pose[first_invalid].tolist()}, "
                    f"cost={float(solved_costs[first_invalid]):.6g}, "
                    f"joint margin={float(solved_margin[first_invalid]):.6g}."
                )
            return solved_full, solved_arm, solved_costs, solved_margin

        nominal_target = torch.as_tensor(self.cfg.target_cup_reset_pos, device=self.device)
        source_delta = source_positions - nominal_source
        target_delta = target_positions - nominal_target

        carry_target_pose = nominal_carry_tcp_pose.repeat(bank_size, 1)
        carry_position_range = torch.as_tensor(
            (*self.cfg.curriculum_randomized_carry_position_range, 0.0),
            device=self.device,
        )
        carry_target_pose[:, :3] += torch.clamp(source_delta, min=-carry_position_range, max=carry_position_range)

        def nominal_seed(arm_q_values: tuple[float, ...]) -> torch.Tensor:
            result = seed.clone()
            result[:, arm_coordinate_ids] = torch.as_tensor(arm_q_values, device=self.device)
            result[:, finger_coordinate_ids] = float(self.cfg.gripper_preload_pos)
            return result

        _, carry_arm_q, carry_costs, carry_margin = solve_reference_waypoint(
            "carry",
            carry_target_pose,
            grasp_solved,
        )
        pour_target_pose = nominal_pour_tcp_pose.repeat(bank_size, 1)
        pour_target_pose[:, :3] += target_delta
        pour_target_pose[:, 2] += float(self.cfg.curriculum_randomized_pour_clearance)
        _, pour_arm_q, pour_costs, pour_margin = solve_reference_waypoint(
            "pour",
            pour_target_pose,
            nominal_seed(self.cfg.curriculum_pour_arm_q),
        )
        tilt_target_pose = nominal_tilt_tcp_pose.repeat(bank_size, 1)
        tilt_target_pose[:, :3] += target_delta
        tilt_target_pose[:, 2] += float(self.cfg.curriculum_randomized_pour_clearance)
        _, tilt_arm_q, tilt_costs, tilt_margin = solve_reference_waypoint(
            "tilt",
            tilt_target_pose,
            nominal_seed(self.cfg.curriculum_pour_target_arm_q),
        )

        randomized_rows = slice(0, randomized_bank_size)
        reach_rows = slice(randomized_bank_size, bank_size)
        self._randomized_source_pos_bank_t = source_positions[randomized_rows]
        self._randomized_source_yaw_bank_t = source_yaws[randomized_rows]
        self._randomized_source_quat_bank_t = source_quaternions[randomized_rows]
        self._randomized_target_pos_bank_t = target_positions[randomized_rows]
        self._randomized_tcp_pos_bank_t = tcp_positions[randomized_rows]
        self._randomized_tcp_quat_bank_t = target_rotations_w[randomized_rows]
        self._randomized_arm_q_bank_t = arm_q[randomized_rows]
        self._randomized_pregrasp_arm_q_bank_t = pregrasp_arm_q[randomized_rows]
        self._randomized_grasp_arm_q_bank_t = grasp_arm_q[randomized_rows]
        self._randomized_carry_arm_q_bank_t = carry_arm_q[randomized_rows]
        self._randomized_pour_arm_q_bank_t = pour_arm_q[randomized_rows]
        self._randomized_tilt_arm_q_bank_t = tilt_arm_q[randomized_rows]
        self._randomized_reset_ik_cost_t = costs[randomized_rows]
        self._randomized_reset_ik_margin_t = margin[randomized_rows]
        self._randomized_pregrasp_ik_cost_t = pregrasp_costs[randomized_rows]
        self._randomized_pregrasp_ik_margin_t = pregrasp_margin[randomized_rows]
        self._randomized_grasp_ik_cost_t = grasp_costs[randomized_rows]
        self._randomized_grasp_ik_margin_t = grasp_margin[randomized_rows]
        self._randomized_carry_ik_cost_t = carry_costs[randomized_rows]
        self._randomized_carry_ik_margin_t = carry_margin[randomized_rows]
        self._randomized_pour_ik_cost_t = pour_costs[randomized_rows]
        self._randomized_pour_ik_margin_t = pour_margin[randomized_rows]
        self._randomized_tilt_ik_cost_t = tilt_costs[randomized_rows]
        self._randomized_tilt_ik_margin_t = tilt_margin[randomized_rows]
        self._randomized_extent_index_pools = randomization_extent_index_pools(
            source_positions[randomized_rows],
            source_yaws[randomized_rows],
            target_positions[randomized_rows],
            tcp_jitter[randomized_rows],
            source_center=self.cfg.cup_reset_pos[:2],
            source_half_range=self.cfg.curriculum_randomized_source_position_range,
            source_yaw_half_range=self.cfg.curriculum_randomized_source_yaw_range,
            target_center=self.cfg.curriculum_randomized_target_center_xy,
            target_half_range=self.cfg.curriculum_randomized_target_position_range,
            tcp_jitter_half_range=self.cfg.curriculum_randomized_reset_tcp_jitter,
            extent_levels=self.cfg.curriculum_randomization_extent_levels,
        )
        expected_bank_indices = torch.arange(randomized_bank_size, device=self.device)
        if not torch.equal(self._randomized_extent_index_pools[-1], expected_bank_indices):
            raise RuntimeError("The final randomization extent must contain every prevalidated IK bank row.")
        if source_positions[reach_rows].shape[0] != samples_per_source:
            raise RuntimeError(
                f"Expected {samples_per_source} dedicated nominal-source reach poses, "
                f"found {source_positions[reach_rows].shape[0]}."
            )
        self._reach_tcp_pos_bank_t = tcp_positions[reach_rows]
        self._reach_source_yaw_bank_t = source_yaws[reach_rows]
        self._reach_arm_q_bank_t = arm_q[reach_rows]
        self._reach_pregrasp_arm_q_bank_t = pregrasp_arm_q[reach_rows]
        self._reach_grasp_arm_q_bank_t = grasp_arm_q[reach_rows]
        self._reach_reset_ik_cost_t = costs[reach_rows]
        self._reach_reset_ik_margin_t = margin[reach_rows]
        # Newton IK and the zero-copy Warp/Torch views above run asynchronously. The temporary
        # solver and prototype are released when this method returns, so complete every gather
        # before their backing allocations can be reclaimed. This is a one-time startup barrier.
        wp.synchronize_device(model.device)

    # ----------------------------------------------------------- poses / obs
    def _pose_w_to_e(self, pose_w: torch.Tensor) -> torch.Tensor:
        """Convert a public world-frame pose view to a finite environment-frame pose."""
        pos = torch.nan_to_num(pose_w[:, :3], nan=0.0, posinf=0.0, neginf=0.0) - self.env_origins
        raw_quat = pose_w[:, 3:7]
        quat = torch.nan_to_num(raw_quat, nan=0.0, posinf=0.0, neginf=0.0)
        norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
        ident = torch.zeros_like(raw_quat)
        ident[:, 3] = 1.0
        valid = torch.isfinite(raw_quat).all(dim=-1, keepdim=True) & (norm > 1.0e-6)
        quat = torch.where(valid, quat / torch.clamp(norm, min=1.0e-6), ident)
        return torch.cat((pos, quat), dim=-1)

    def ee_pose_e(self) -> torch.Tensor:
        """End-effector (panda_hand) pose in the env frame: ``(num_envs, 7)`` pos + xyzw quat."""
        return self._pose_w_to_e(self._robot.data.body_link_pose_w.torch[:, self._tcp_body_idx])

    def cup_pose_e(self) -> torch.Tensor:
        """Cup body pose in the env frame: ``(num_envs, 7)`` pos + xyzw quat."""
        return self._pose_w_to_e(self._source_cup.data.root_link_pose_w.torch)

    def cup_velocity_w(self) -> torch.Tensor:
        """Source-cup linear and angular velocity in the world frame [m/s, rad/s]."""
        return self._source_cup.data.root_link_vel_w.torch

    def target_pose_e(self) -> torch.Tensor:
        """Receiving-cup pose in the env frame: ``(num_envs, 7)`` pos + xyzw quat."""
        return self._pose_w_to_e(self._target_cup.data.root_link_pose_w.torch)

    def tcp_pose_e(self) -> torch.Tensor:
        """Tool-centre pose in the robot-root/environment frame."""
        body_pose_w = self._robot.data.body_link_pose_w.torch[:, self._tcp_body_idx]
        root_pose_w = self._robot.data.root_link_pose_w.torch
        pos, quat = math_utils.subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], body_pose_w[:, :3], body_pose_w[:, 3:7]
        )
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
        finger_pos = self._robot.data.joint_pos.torch[:, self._finger_joint_ids]
        width = finger_pos.sum(dim=-1)
        valid = torch.isfinite(finger_pos).all(dim=-1) & torch.isfinite(width)
        return torch.where(valid, width, torch.full_like(width, float(self.gripper_open_width)))

    def finger_joint_pos(self) -> torch.Tensor:
        """Individual policy-controlled finger joint positions [m]."""
        return self._robot.data.joint_pos.torch[:, self._finger_joint_ids]

    def finger_joint_vel(self) -> torch.Tensor:
        """Individual policy-controlled finger joint velocities [m/s]."""
        return self._robot.data.joint_vel.torch[:, self._finger_joint_ids]

    def desired_grasp_tcp_quat_c(self) -> torch.Tensor:
        """Desired TCP orientation in the source-cup frame as canonical XYZW quaternions."""
        return self._desired_grasp_tcp_quat_c

    def arm_joint_pos(self) -> torch.Tensor:
        """Current policy-controlled arm joint positions [rad]."""
        return self._robot.data.joint_pos.torch[:, self._arm_joint_ids]

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

    def set_curriculum_stage(
        self,
        env_ids: list[int] | torch.Tensor | slice,
        stage: int,
    ) -> None:
        """Assign a curriculum stage and success threshold to selected environments."""
        if stage < 0 or stage >= len(self.cfg.curriculum_stage_names):
            raise ValueError(f"Curriculum stage {stage} is out of range.")
        self.curriculum_stage[env_ids] = stage
        self.pour_target_frac[env_ids] = float(self.cfg.curriculum_target_frac[stage])

    def set_curriculum_randomization_level(
        self,
        env_ids: list[int] | torch.Tensor | slice,
        level: int,
    ) -> None:
        """Assign one prevalidated source-randomization extent to selected environments."""
        if level < 0 or level >= len(self._randomized_extent_index_pools):
            raise ValueError(f"Curriculum randomization level {level} is out of range.")
        self.curriculum_randomization_level[env_ids] = level

    def curriculum_randomization_bank_size(self, level: int) -> int:
        """Return the number of prevalidated reset rows eligible at a randomization level."""
        if level < 0 or level >= len(self._randomized_extent_index_pools):
            raise ValueError(f"Curriculum randomization level {level} is out of range.")
        return int(self._randomized_extent_index_pools[level].numel())

    def particle_pos_e(self) -> torch.Tensor:
        """Per-env MPM particle positions in env coordinates, shape ``(N, P, 3)``."""
        return self._media.data.particle_pos_w.torch - self.env_origins[:, None, :]

    def particle_vel_e(self) -> torch.Tensor:
        """Per-env MPM particle velocities in environment axes, shape ``(N, P, 3)``."""
        # Environments differ only by translation, so world and environment velocity axes coincide.
        return self._media.data.particle_vel_w.torch

    def _points_inside_cup(
        self, points_e: torch.Tensor, pose_e: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor
    ) -> torch.Tensor:
        rel = points_e - pose_e[:, None, :3]
        quat = pose_e[:, None, 3:7].expand(-1, points_e.shape[1], -1)
        local = math_utils.quat_apply_inverse(quat, rel)
        margin = float(self.cfg.particle_count_margin)
        return ((local >= lo - margin) & (local <= hi + margin)).all(dim=-1)

    def _particle_region_masks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Boolean ``(source, target, spilled)`` masks, cached within one manager step."""
        step = int(getattr(self, "common_step_counter", -1))
        if self._particle_region_cache is not None and self._particle_region_cache_step == step:
            return self._particle_region_cache
        points = self.particle_pos_e()
        source = self._points_inside_cup(points, self.cup_pose_e(), self._source_inner_lo_t, self._source_inner_hi_t)
        target_region = self._points_inside_cup(
            points,
            self.target_pose_e(),
            self._target_inner_lo_t,
            self._target_inner_hi_t,
        )
        # Geometric overlap is not delivery: nesting the source cup inside the receiver must not
        # score particles that remain physically contained by the source cup.
        target = _delivered_particle_mask(source, target_region)
        spill_height = float(self.cfg.spill_table_height) + float(self.cfg.particle_count_margin)
        spilled = _spilled_particle_mask(points, source, target, max_height=spill_height)
        self._particle_region_cache = (source, target, spilled)
        self._particle_region_cache_step = step
        return source, target, spilled

    def particles_in_target_mask(self) -> torch.Tensor:
        """Particles inside the target cup and no longer inside the source cup."""
        return self._particle_region_masks()[1]

    def particle_region_masks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return cached source, source-exclusive target, and irreversible-spill masks."""
        return self._particle_region_masks()

    def update_held_delivery_tracker(self, held_pour: torch.Tensor) -> None:
        """Record particles whose target-entry edge occurs during a held pour.

        The tracker is idempotent within one manager step because success termination and reward
        evaluation consume the same state. An unheld entry does not permanently disqualify a
        particle: after it leaves the target, a later valid held re-entry can still qualify.

        Args:
            held_pour: Per-environment mask for a preloaded, lifted source grasp.
        """
        if held_pour.shape != (self.num_envs,):
            raise ValueError(f"held_pour must have shape ({self.num_envs},), got {tuple(held_pour.shape)}.")
        step = int(self.common_step_counter)
        if self._held_delivery_tracker_step == step:
            return
        in_target = self.particles_in_target_mask()
        target_entry = in_target & ~self._target_entry_seen
        self._held_delivered |= target_entry & held_pour.unsqueeze(-1)
        self._target_entry_seen.copy_(in_target)
        self._held_delivery_tracker_step = step

    def held_delivered_mask(self) -> torch.Tensor:
        """Particles with at least one target-entry edge during a held pour."""
        return self._held_delivered

    def current_held_delivered_mask(self) -> torch.Tensor:
        """Validly delivered particles that remain inside the receiving cup."""
        return self.particles_in_target_mask() & self._held_delivered

    def particles_spilled_mask(self) -> torch.Tensor:
        """Per-particle irreversible-spill membership used by one-time penalties."""
        return self._particle_region_masks()[2]

    def count_in_source(self) -> torch.Tensor:
        return self._particle_region_masks()[0].sum(dim=1).float()

    def count_in_target(self) -> torch.Tensor:
        return self._particle_region_masks()[1].sum(dim=1).float()

    def count_spilled(self) -> torch.Tensor:
        return self._particle_region_masks()[2].sum(dim=1).float()

    def spilled_fraction(self) -> torch.Tensor:
        return self.count_spilled() / max(self.num_particles, 1)

    def state_finite(self) -> torch.Tensor:
        """Per-env instability guard over robot, source cup, and MPM media state."""
        cup_velocity = self._source_cup.data.root_link_vel_w.torch
        return _state_finite(
            self._robot.data.joint_pos.torch,
            self._robot.data.joint_vel.torch,
            self._robot.data.body_link_pose_w.torch[:, self._tcp_body_idx],
            self._source_cup.data.root_link_pose_w.torch,
            cup_velocity[:, :3],
            cup_velocity[:, 3:],
            self._media.data.particle_pos_w.torch,
        )

    def rigid_state_in_bounds(self) -> torch.Tensor:
        """Return whether finite rigid state remains within task-safe observation bounds."""
        cup_velocity = self._source_cup.data.root_link_vel_w.torch
        return _rigid_state_in_bounds(
            self._robot.data.joint_pos.torch,
            self._robot.data.joint_vel.torch,
            self._joint_pos_limits_t,
            self._robot.data.body_link_pose_w.torch[:, self._tcp_body_idx],
            self._source_cup.data.root_link_pose_w.torch,
            cup_velocity[:, :3],
            cup_velocity[:, 3:],
            self.env_origins,
            self._particle_workspace_lower_t,
            self._particle_workspace_upper_t,
            joint_position_margin=self.cfg.state_bound_joint_position_margin,
            max_joint_velocity=self.cfg.state_bound_max_joint_velocity,
            max_cup_linear_velocity=self.cfg.state_bound_max_cup_linear_velocity,
            max_cup_angular_velocity=self.cfg.state_bound_max_cup_angular_velocity,
        )

    def particles_in_workspace(self) -> torch.Tensor:
        """Return a per-environment mask for media inside the configured local workspace."""
        return _particles_in_workspace(
            self.particle_pos_e(),
            self._particle_workspace_lower_t,
            self._particle_workspace_upper_t,
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
        world_mask = boolean_selection_mask(self.num_envs, env_ids)

        stage = self.curriculum_stage[env_ids]
        arm_q = self._curriculum_arm_q_t[stage].clone()
        reference_waypoints = self._nominal_reference_waypoints_t.unsqueeze(0).repeat(n, 1, 1)
        cup_pos_e = torch.as_tensor(self.cfg.cup_reset_pos, device=self.device).repeat(n, 1)
        target_pos_e = torch.as_tensor(self.cfg.target_cup_reset_pos, device=self.device).repeat(n, 1)
        source_quat = self._curriculum_cup_quat_t[stage].clone()
        target_quat = self._curriculum_cup_quat_t[stage].clone()
        grasp_rows = torch.nonzero(stage == self._grasp_stage_index, as_tuple=False).flatten()
        if grasp_rows.numel() > 0:
            bank_indices = torch.randint(
                self._reach_grasp_arm_q_bank_t.shape[0],
                (grasp_rows.numel(),),
                device=self.device,
            )
            arm_q[grasp_rows] = self._reach_grasp_arm_q_bank_t[bank_indices]
            reference_waypoints[grasp_rows, 2] = self._reach_grasp_arm_q_bank_t[bank_indices]
        full_rows = torch.nonzero(stage == self._full_stage_index, as_tuple=False).flatten()
        if full_rows.numel() > 0:
            bank_indices = torch.randint(
                self._reach_arm_q_bank_t.shape[0],
                (full_rows.numel(),),
                device=self.device,
            )
            arm_q[full_rows] = self._reach_arm_q_bank_t[bank_indices]
            reference_waypoints[full_rows, 0] = self._reach_arm_q_bank_t[bank_indices]
            reference_waypoints[full_rows, 1] = self._reach_pregrasp_arm_q_bank_t[bank_indices]
            reference_waypoints[full_rows, 2] = self._reach_grasp_arm_q_bank_t[bank_indices]
        randomized_rows = torch.nonzero(stage == self._randomized_stage_index, as_tuple=False).flatten()
        if randomized_rows.numel() > 0:
            randomization_levels = self.curriculum_randomization_level[env_ids[randomized_rows]]
            bank_indices = sample_index_pools(self._randomized_extent_index_pools, randomization_levels)
            arm_q[randomized_rows] = self._randomized_arm_q_bank_t[bank_indices]
            reference_waypoints[randomized_rows, 0] = self._randomized_arm_q_bank_t[bank_indices]
            reference_waypoints[randomized_rows, 1] = self._randomized_pregrasp_arm_q_bank_t[bank_indices]
            reference_waypoints[randomized_rows, 2] = self._randomized_grasp_arm_q_bank_t[bank_indices]
            reference_waypoints[randomized_rows, 3] = self._randomized_carry_arm_q_bank_t[bank_indices]
            reference_waypoints[randomized_rows, 4] = self._randomized_pour_arm_q_bank_t[bank_indices]
            reference_waypoints[randomized_rows, 5] = self._randomized_tilt_arm_q_bank_t[bank_indices]
            cup_pos_e[randomized_rows] = self._randomized_source_pos_bank_t[bank_indices]
            source_quat[randomized_rows] = self._randomized_source_quat_bank_t[bank_indices]
            target_pos_e[randomized_rows] = self._randomized_target_pos_bank_t[bank_indices]
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
        arm_action = self.action_manager.get_term("arm_action")
        if hasattr(arm_action, "set_reference"):
            arm_action.set_reference(
                reference_waypoints,
                self._curriculum_reference_phase_t[stage],
                arm_q,
                env_ids=env_ids,
            )
        else:
            arm_action.set_action_offset(arm_q, env_ids=env_ids)

        finger_position = self._curriculum_finger_pos_t[stage].unsqueeze(-1).expand(-1, len(FINGER_JOINTS))
        self._robot.write_joint_position_to_sim_index(
            position=finger_position,
            joint_ids=self._finger_joint_ids,
            env_ids=env_ids,
        )
        self._robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(finger_position),
            joint_ids=self._finger_joint_ids,
            env_ids=env_ids,
        )
        self._robot.set_joint_position_target_index(
            target=finger_position,
            joint_ids=self._finger_joint_ids,
            env_ids=env_ids,
        )
        gripper_target = torch.where(
            (stage <= 1).unsqueeze(-1),
            torch.full((n, 1), float(self.cfg.gripper_preload_pos), device=self.device),
            torch.full((n, 1), float(self.cfg.gripper_open_pos), device=self.device),
        )
        self.action_manager.get_term("gripper_action").set_reset_position(
            gripper_target,
            env_ids=env_ids,
        )

        # Public root/joint writers invalidate FK. Reading a public body-pose view consumes
        # the accumulated articulation mask, making all dirtied body poses authoritative
        # before solver caches and source-cup proxy transforms are refreshed. Priming the
        # robot view also prevents its next observation from issuing a redundant FK launch.
        _ = self._robot.data.body_link_pose_w

        grasp_offset = torch.zeros((n, 3), device=self.device)
        grasp_offset[:, 2] = float(self.cfg.cup_grasp_height)
        tcp_cup_pos_e = self.tcp_pos_e()[env_ids] - math_utils.quat_apply(source_quat, grasp_offset)
        # Lifted stages follow their solved TCP. Full and randomized stages use authored or
        # IK-paired table positions.
        lifted_stage = stage < 2
        cup_pos_e = torch.where(lifted_stage.unsqueeze(-1), tcp_cup_pos_e, cup_pos_e)
        cup_world = cup_pos_e + self.env_origins[env_ids]
        cup_pose = torch.cat((cup_world, source_quat), dim=-1)
        self._source_cup.write_root_pose_to_sim_index(root_pose=cup_pose, env_ids=env_ids)
        self._source_cup.write_root_velocity_to_sim_index(
            root_velocity=cup_pose.new_zeros((n, 6)),
            env_ids=env_ids,
        )
        target_world = target_pos_e + self.env_origins[env_ids]
        target_pose = torch.cat((target_world, target_quat), dim=-1)
        self._target_cup.write_root_pose_to_sim_index(root_pose=target_pose, env_ids=env_ids)
        self._target_cup.write_root_velocity_to_sim_index(
            root_velocity=target_pose.new_zeros((n, 6)),
            env_ids=env_ids,
        )

        new_p = self._sample_cup_media(cup_world, source_quat)
        self._media.write_particle_pos_to_sim_index(new_p, env_ids=env_ids)
        self._media.write_particle_velocity_to_sim_index(torch.zeros_like(new_p), env_ids=env_ids)
        NewtonManager.reset_solver_state(
            world_mask=wp.from_torch(world_mask, dtype=wp.bool),
            flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE,
        )
        self._particle_region_cache = None
        self._particle_region_cache_step = -1
        self.episode_succeeded[env_ids] = False
        self.ep_max_target_frac[env_ids] = 0.0
        self._success_dwell_count[env_ids] = 0
        self._lost_grasp_dwell_count[env_ids] = 0
        self._target_entry_seen[env_ids] = False
        self._held_delivered[env_ids] = False
        # Selective resets can occur after this step's termination/reward pass. Invalidating the
        # scalar cache is cheap and keeps direct reset/replay workflows correct as well.
        self._held_delivery_tracker_step = -1

    def _sample_cup_media(self, cup_pos: torch.Tensor, cup_quat: torch.Tensor) -> torch.Tensor:
        """Transform the local media lattice into selected cup poses on the simulation device."""
        particle_count = self._media_local_points_t.shape[0]
        local_points = self._media_local_points_t.unsqueeze(0).expand(cup_pos.shape[0], -1, -1)
        quaternions = cup_quat.unsqueeze(1).expand(-1, particle_count, -1)
        return math_utils.quat_apply(quaternions, local_points) + cup_pos.unsqueeze(1)
