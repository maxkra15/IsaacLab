# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based RL env: Franka grasping a dynamic cup of MPM media and pouring it.

A single rigid **dynamic** cup body rests on the table; the Franka grasps it with its fingers
through Newton-generated friction contacts resolved by MJWarp and pours the granular MPM media by
tilting. A Newton :class:`CoupledSolverCfg` with **proxy coupling** advances the robot + cup (MJWarp
``arm`` entry) and the media (implicit ``media`` entry) together in ``sim.step()``.

The cup carries two co-located shapes on one body (see :meth:`FrankaPourEnv._add_cup_body`):

* a SOLID grasp box (``COLLIDE_SHAPES``, owned by the ``arm`` entry) the fingers actually grip --
  thin cup walls slip the jaws, so the box gives them something to clamp; and
* a hollow cavity mesh (``COLLIDE_PARTICLES`` only, from
  :func:`.cube_bowl_mesh.make_cube_bowl_mesh`) that retains the media.

The cup body lives only in the ``arm`` entry; a proxy mapping (source ``arm``, destination
``media``) exposes its ``COLLIDE_PARTICLES`` mesh to the MPM solver as an auto-pose-synced collider
(a body cannot be owned by two coupled entries). The cup pose comes from the solver -- no weld
kernel drives it. Arm control is relative DiffIK plus a binary gripper open/close action.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import newton
import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import math as math_utils

from .cube_bowl_mesh import cube_bowl_inner_bounds, make_cube_bowl_mesh
from .cup_media import build_media_object_cfg, cup_cavity_lattice
from .mdp.terminations import _state_finite
from .pour_env_cfg import _mpm_solver_cfg, _resolve_mpm_cell_cap
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
        self._prepare_newton_extras(cfg)
        self._install_newton_builder_hook()
        try:
            super().__init__(cfg, render_mode, **kwargs)
        finally:
            self._remove_newton_builder_hook()

    def load_managers(self) -> None:
        self._setup_after_physics()
        NewtonManager.set_decimation(self.cfg.decimation)
        super().load_managers()

    # ------------------------------------------------------------------ build
    def _prepare_newton_extras(self, cfg: FrankaPourEnvCfg) -> None:
        """Bake the cup geometry and build the declarative cup-media scene asset.

        Runs before ``super().__init__`` so media spawn points see the final
        layout and material values and the per-world builder hook has the cup
        geometry available.
        """
        _mpm_solver_cfg(cfg).max_active_cell_count = _resolve_mpm_cell_cap(cfg)

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

        # Declared media spawn snapshot at the cup reset pose (re-sampled exactly every reset).
        if cfg.scene.media is None:
            cfg.scene.media = build_media_object_cfg(cfg, self._cup_reset_pos, self._cup_reset_quat)

        self._custom_proto, self._custom_meta = self._build_custom_proto(cfg)
        self._hand_ids_l: list[int] = []
        self._cup_body_ids_l: list[int] = []
        self._target_body_ids_l: list[int] = []
        self._cup_joint_ids_l: list[int] = []
        self._cup_joint_q_l: list[int] = []
        self._cup_joint_qd_l: list[int] = []
        self._arm_joint_ids_l: list[list[int]] = []
        self._arm_q_l: list[list[int]] = []
        self._arm_qd_l: list[list[int]] = []
        self._builder_hook = None

    # --------------------------------------------------------------- proto
    def _build_custom_proto(self, cfg: FrankaPourEnvCfg):
        """Per-env prototype: the single dynamic cup body (+ free joint) and a ground plane.

        The cup is ONE rigid dynamic body with a free joint to the world, carrying two co-located
        shapes: a solid grasp box (``COLLIDE_SHAPES``) and the hollow cavity mesh
        (``COLLIDE_PARTICLES``). The media particles are a scene-level :class:`MPMObject`
        (``cfg.scene.media``), not in this prototype.
        """
        proto = NewtonManager.create_builder()
        proto.default_shape_cfg.mu = 0.8
        cup_body, cup_joint = self._add_cup_body(proto, cfg)
        target_body, _ = self._add_target_cup_bodies(proto, cfg)
        # A body-attached, per-world particle plane keeps spilled media at table height. Selecting
        # this body explicitly avoids exposing every unrelated global static shape to the MPM view.
        spill_floor = proto.add_body(
            xform=wp.transform_identity(),
            mass=0.0,
            inertia=wp.mat33(),
            is_kinematic=True,
            lock_inertia=True,
            label="/World/SpillFloor",
        )
        spill_floor_shape = proto.add_shape_plane(
            body=spill_floor,
            xform=wp.transform_identity(),
            width=0.0,
            length=0.0,
            cfg=newton.ModelBuilder.ShapeConfig(
                mu=0.8,
                margin=cfg.collider_margin,
                has_shape_collision=False,
                has_particle_collision=True,
            ),
            color=(0.3, 0.3, 0.3),
            label="/World/SpillFloor/Collision",
        )
        proto.shape_flags[spill_floor_shape] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
        proto.shape_flags[spill_floor_shape] |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
        proto.shape_flags[spill_floor_shape] &= ~int(newton.ShapeFlags.VISIBLE)
        return proto, {
            "cup_body": cup_body,
            "cup_joint": cup_joint,
            "target_body": target_body,
        }

    def _add_cup_body(self, proto, cfg: FrankaPourEnvCfg) -> tuple[int, int]:
        """Add the single rigid dynamic cup body ``/World/Cup`` (grasp box + cavity mesh) + free joint."""
        box_half = np.asarray(cfg.cup_grasp_box_half, dtype=np.float64)
        # Solid-box inertia for the cup mass (the box is the cup's mass proxy).
        ixx = cfg.cup_mass / 3.0 * (box_half[1] ** 2 + box_half[2] ** 2)
        iyy = cfg.cup_mass / 3.0 * (box_half[0] ** 2 + box_half[2] ** 2)
        izz = cfg.cup_mass / 3.0 * (box_half[0] ** 2 + box_half[1] ** 2)
        inertia = wp.mat33(float(ixx), 0.0, 0.0, 0.0, float(iyy), 0.0, 0.0, 0.0, float(izz))

        cup_pos = self._cup_reset_pos
        cup_quat = self._cup_reset_quat
        cup_body = proto.add_link(
            xform=wp.transform(wp.vec3(*cup_pos.tolist()), wp.quat(*cup_quat.tolist())),
            mass=float(cfg.cup_mass),
            inertia=inertia,
            lock_inertia=True,
            label="/World/Cup",
        )
        # Free joint so MJWarp treats the cup as a movable dynamic body (grasped/pushed by contacts);
        # the joint coords mirror body_q (xyzw quat) and are how a reset rewrites the cup pose.
        cup_joint = proto.add_joint_free(child=cup_body, label="/World/Cup/FreeJoint")
        proto.add_articulation([cup_joint], label="/World/Cup")

        # Grasp box: SOLID, COLLIDE_SHAPES only (arm-entry contacts with the fingers; never MPM). Its
        # base sits at the cup base (z=0) so it spans the cup body the fingers close around.
        box_center_z = float(box_half[2])
        box_cfg = newton.ModelBuilder.ShapeConfig(
            mu=float(cfg.cup_grasp_box_friction),
            density=0.0,
            ke=float(cfg.grasp_contact_ke),
            kd=float(cfg.grasp_contact_kd),
            kf=float(cfg.grasp_contact_kf),
            margin=cfg.collider_margin,
        )
        box_sid = proto.add_shape_box(
            cup_body,
            xform=wp.transform(wp.vec3(0.0, 0.0, box_center_z), wp.quat_identity()),
            hx=float(box_half[0]),
            hy=float(box_half[1]),
            hz=float(box_half[2]),
            cfg=box_cfg,
            color=(0.85, 0.55, 0.20),
        )
        proto.shape_flags[box_sid] |= int(newton.ShapeFlags.COLLIDE_SHAPES)
        proto.shape_flags[box_sid] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
        proto.shape_flags[box_sid] &= ~int(newton.ShapeFlags.VISIBLE)
        proto.shape_margin[box_sid] = cfg.collider_margin

        # Cavity mesh: hollow, COLLIDE_PARTICLES only (visible). The proxy bridges this to MPM.
        mesh = newton.Mesh(self._cup_vertices, self._cup_indices, compute_inertia=False, is_solid=False)
        mesh_cfg = newton.ModelBuilder.ShapeConfig(
            mu=float(cfg.source_cup_friction), density=0.0, margin=cfg.collider_margin, has_particle_collision=True
        )
        mesh_sid = proto.add_shape_mesh(
            cup_body,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mesh=mesh,
            cfg=mesh_cfg,
            color=(0.95, 0.82, 0.16),
            label="/World/Cup/Collision",
        )
        proto.shape_flags[mesh_sid] |= int(newton.ShapeFlags.COLLIDE_PARTICLES) | int(newton.ShapeFlags.VISIBLE)
        proto.shape_flags[mesh_sid] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
        proto.shape_margin[mesh_sid] = cfg.collider_margin
        proto.shape_material_mu[mesh_sid] = float(cfg.source_cup_friction)
        if proto.shape_source[mesh_sid] is not None:
            proto.shape_source[mesh_sid].indices = proto.shape_source[mesh_sid].indices.reshape(-1)
        return cup_body, cup_joint

    def _add_target_cup_bodies(self, proto, cfg: FrankaPourEnvCfg) -> tuple[int, int]:
        """Add disjoint MPM-only and rigid-only copies of the fixed receiving cup.

        Coupled entries cannot own one body twice. The visible ``TargetCup`` is therefore selected
        only by the media entry and retains particles, while an invisible ``TargetCupRigid`` copy is
        selected only by the arm entry and prevents robot/source-cup tunnelling through the receiver.
        """

        pos = self._target_reset_pos
        quat = self._target_reset_quat

        def add_body(label: str, *, particle_collision: bool, visible: bool) -> int:
            body = proto.add_body(
                xform=wp.transform(wp.vec3(*pos.tolist()), wp.quat(*quat.tolist())),
                mass=0.0,
                is_kinematic=True,
                lock_inertia=True,
                label=f"/World/{label}",
            )
            mesh = newton.Mesh(self._target_vertices, self._target_indices, compute_inertia=False, is_solid=False)
            shape_cfg = newton.ModelBuilder.ShapeConfig(
                mu=float(cfg.target_cup_friction),
                density=0.0,
                ke=float(cfg.grasp_contact_ke),
                kd=float(cfg.grasp_contact_kd),
                margin=cfg.collider_margin,
                has_particle_collision=particle_collision,
            )
            sid = proto.add_shape_mesh(
                body,
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                mesh=mesh,
                cfg=shape_cfg,
                color=(0.20, 0.55, 0.90),
                label=f"/World/{label}/Collision",
            )
            if particle_collision:
                proto.shape_flags[sid] |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
                proto.shape_flags[sid] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
            else:
                proto.shape_flags[sid] |= int(newton.ShapeFlags.COLLIDE_SHAPES)
                proto.shape_flags[sid] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
            if visible:
                proto.shape_flags[sid] |= int(newton.ShapeFlags.VISIBLE)
            else:
                proto.shape_flags[sid] &= ~int(newton.ShapeFlags.VISIBLE)
            proto.shape_margin[sid] = cfg.collider_margin
            proto.shape_material_mu[sid] = float(cfg.target_cup_friction)
            if proto.shape_source[sid] is not None:
                proto.shape_source[sid].indices = proto.shape_source[sid].indices.reshape(-1)
            proto.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
            proto.body_mass[body] = 0.0
            proto.body_inv_mass[body] = 0.0
            proto.body_inertia[body] = wp.mat33()
            proto.body_inv_inertia[body] = wp.mat33()
            return body

        target = add_body("TargetCup", particle_collision=True, visible=True)
        target_rigid = add_body("TargetCupRigid", particle_collision=False, visible=False)
        return target, target_rigid

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
        env_root = f"/World/envs/env_{env_id}"
        self._disable_robot_particle_collision(builder, env_id)
        self._enable_robot_gravcomp(builder, env_id)
        self._configure_finger_contact_material(builder, env_id)

        hand = self._find_world_body(builder, env_id, "panda_hand")
        arm_q, arm_qd, arm_joints = self._find_world_joint_coords(builder, env_id, ARM_JOINTS)

        b_off = builder.body_count
        j_off = builder.joint_count
        label_starts = self._builder_label_starts(builder)
        builder.add_builder(
            self._custom_proto,
            xform=wp.transform(wp.vec3(*[float(v) for v in position]), wp.quat(*[float(v) for v in quaternion])),
        )
        self._rewrite_builder_labels(builder, label_starts, env_root)

        cup_joint = j_off + self._custom_meta["cup_joint"]
        self._hand_ids_l.append(hand)
        self._cup_body_ids_l.append(b_off + self._custom_meta["cup_body"])
        self._target_body_ids_l.append(b_off + self._custom_meta["target_body"])
        self._cup_joint_ids_l.append(cup_joint)
        self._cup_joint_q_l.append(int(builder.joint_q_start[cup_joint]))
        self._cup_joint_qd_l.append(int(builder.joint_qd_start[cup_joint]))
        self._arm_joint_ids_l.append(arm_joints)
        self._arm_q_l.append(arm_q)
        self._arm_qd_l.append(arm_qd)

    @staticmethod
    def _label_world(builder, prefix: str, index: int) -> int:
        worlds = getattr(builder, f"{prefix}_world", None)
        if worlds is None:
            return -1
        return int(worlds[index])

    def _find_world_body(self, builder, env_id: int, body_name: str) -> int:
        matches = [
            body_id
            for body_id, label in enumerate(builder.body_label)
            if self._label_world(builder, "body", body_id) == env_id and str(label).rsplit("/", 1)[-1] == body_name
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {body_name!r} body in Newton world {env_id}, found {matches}.")
        return matches[0]

    def _find_world_joint_coords(
        self, builder, env_id: int, joint_names: Sequence[str]
    ) -> tuple[list[int], list[int], list[int]]:
        joint_labels = [str(label) for label in builder.joint_label]
        joint_q, joint_qd, joint_ids = [], [], []
        for joint_name in joint_names:
            matches = [
                joint_id
                for joint_id, label in enumerate(joint_labels)
                if self._label_world(builder, "joint", joint_id) == env_id and label.rsplit("/", 1)[-1] == joint_name
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one {joint_name!r} joint in Newton world {env_id}, found {matches}.")
            joint_id = matches[0]
            joint_ids.append(joint_id)
            joint_q.append(int(builder.joint_q_start[joint_id]))
            joint_qd.append(int(builder.joint_qd_start[joint_id]))
        return joint_q, joint_qd, joint_ids

    def _enable_robot_gravcomp(self, builder, env_id: int) -> None:
        """MuJoCo gravity compensation on robot links. The MJWarp solver does not honor the PhysX
        ``disable_gravity`` flag, and relative DiffIK re-anchors to the current pose every step, so
        without this the arm continuously sags under gravity."""
        try:
            attr = builder.custom_attributes["mujoco:gravcomp"]
        except (KeyError, TypeError, AttributeError):
            return
        if attr.values is None:
            attr.values = {}
        for body_id in range(builder.body_count):
            if self._label_world(builder, "body", body_id) != env_id:
                continue
            if "/panda" in str(builder.body_label[body_id]):
                attr.values[int(body_id)] = 1.0

    def _disable_robot_particle_collision(self, builder, env_id: int) -> None:
        """The robot shapes must not collide with MPM particles (only the cup cavity mesh does)."""
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape_id in range(builder.shape_count):
            body_id = int(builder.shape_body[shape_id])
            if body_id < 0 or self._label_world(builder, "body", body_id) != env_id:
                continue
            body_label = str(builder.body_label[body_id])
            if "/Robot/" in body_label or body_label.endswith("/Robot"):
                builder.shape_flags[shape_id] &= ~collide_particles

    def _configure_finger_contact_material(self, builder, env_id: int) -> None:
        """Use a rigid contact material on the two finger collision shapes.

        Applying the same material to the fingers and cup keeps the intended pair response
        independent of import-side material defaults.
        """
        finger_suffixes = ("/panda_leftfinger", "/panda_rightfinger")
        for shape_id in range(builder.shape_count):
            body_id = int(builder.shape_body[shape_id])
            if body_id < 0 or self._label_world(builder, "body", body_id) != env_id:
                continue
            if str(builder.body_label[body_id]).endswith(finger_suffixes):
                builder.shape_material_ke[shape_id] = self._grasp_contact_ke
                builder.shape_material_kd[shape_id] = self._grasp_contact_kd
                builder.shape_material_kf[shape_id] = self._grasp_contact_kf
                builder.shape_material_mu[shape_id] = self._grasp_contact_mu

    @staticmethod
    def _builder_label_starts(builder) -> dict[str, int]:
        return {
            attr: len(getattr(builder, attr))
            for attr in ("body_label", "articulation_label", "joint_label", "shape_label")
        }

    @staticmethod
    def _env_label(label: str, env_root: str) -> str:
        if not isinstance(label, str):
            return label
        if label.startswith("/panda"):
            return f"{env_root}{label}"
        if label.startswith("/World/"):
            return f"{env_root}/{label[len('/World/') :]}"
        return label

    @classmethod
    def _rewrite_builder_labels(cls, builder, starts: dict[str, int], env_root: str) -> None:
        for attr, start in starts.items():
            labels = getattr(builder, attr)
            for idx in range(start, len(labels)):
                labels[idx] = cls._env_label(labels[idx], env_root)

    # ----------------------------------------------------------- post-physics
    def _setup_after_physics(self) -> None:
        dev = self.device
        self._robot = self.scene["robot"]
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
        self._hand_ids = torch.tensor(self._hand_ids_l, device=dev, dtype=torch.long)
        self._cup_body_ids = torch.tensor(self._cup_body_ids_l, device=dev, dtype=torch.long)
        self._target_body_ids = torch.tensor(self._target_body_ids_l, device=dev, dtype=torch.long)
        # Free-joint coordinate starts: q is [px, py, pz, qx, qy, qz, qw] (xyzw), qd is the spatial vel.
        self._cup_joint_q = torch.tensor(self._cup_joint_q_l, device=dev, dtype=torch.long)
        self._cup_joint_qd = torch.tensor(self._cup_joint_qd_l, device=dev, dtype=torch.long)

        self._arm_q_ids = torch.tensor(self._arm_q_l, device=dev, dtype=torch.long)
        self._arm_qd_ids = torch.tensor(self._arm_qd_l, device=dev, dtype=torch.long)
        model = NewtonManager.get_model()
        joint_articulation = wp.to_torch(model.joint_articulation).long()
        arm_joint_ids = torch.tensor(self._arm_joint_ids_l, device=dev, dtype=torch.long)
        cup_joint_ids = torch.tensor(self._cup_joint_ids_l, device=dev, dtype=torch.long).unsqueeze(1)
        arm_articulation_ids = joint_articulation[arm_joint_ids]
        self._cup_articulation_ids = joint_articulation[cup_joint_ids]
        self._reset_articulation_ids = torch.cat((arm_articulation_ids, self._cup_articulation_ids), dim=1)
        if bool((self._reset_articulation_ids < 0).any()):
            raise RuntimeError("Franka Pour reset joints must all belong to Newton articulations.")

        # Media is a scene-level MPMObject asset: per-env particle views + the reset-default snapshot.
        self._media: MPMObject = self.scene["media"]
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
    def _body_pose_e(self, body_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Env-frame ``(pos, xyzw quat)`` of the given Newton bodies (sanitized to finite)."""
        bq = wp.to_torch(NewtonManager.get_state_0().body_q)[body_ids]
        pos = torch.nan_to_num(bq[:, :3], nan=0.0, posinf=0.0, neginf=0.0) - self.env_origins
        quat = bq[:, 3:7]
        norm = torch.linalg.norm(torch.nan_to_num(quat), dim=-1, keepdim=True)
        ident = torch.zeros_like(quat)
        ident[:, 3] = 1.0
        quat = torch.where(norm > 1e-6, quat / torch.clamp(norm, min=1e-6), ident)
        return pos, quat

    def ee_pose_e(self) -> torch.Tensor:
        """End-effector (panda_hand) pose in the env frame: ``(num_envs, 7)`` pos + xyzw quat."""
        pos, quat = self._body_pose_e(self._hand_ids)
        return torch.cat((pos, quat), dim=-1)

    def cup_pose_e(self) -> torch.Tensor:
        """Cup body pose in the env frame: ``(num_envs, 7)`` pos + xyzw quat."""
        pos, quat = self._body_pose_e(self._cup_body_ids)
        return torch.cat((pos, quat), dim=-1)

    def target_pose_e(self) -> torch.Tensor:
        """Receiving-cup pose in the env frame: ``(num_envs, 7)`` pos + xyzw quat."""
        pos, quat = self._body_pose_e(self._target_body_ids)
        return torch.cat((pos, quat), dim=-1)

    def tcp_pose_e(self) -> torch.Tensor:
        """DiffIK-controlled tool-centre pose in the robot-root/env frame."""
        body_pose_w = self._robot.data.body_link_pose_w.torch[:, self._tcp_body_idx]
        root_pose_w = self._robot.data.root_pose_w.torch
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
        cup_body_q = wp.to_torch(NewtonManager.get_state_0().body_q)[self._cup_body_ids]
        return _state_finite(
            self._robot.data.joint_pos.torch,
            cup_body_q,
            self._media.data.particle_pos_w.torch,
        )

    # ----------------------------------------------------------- reset
    def reset_pour_scene(self, env_ids: torch.Tensor) -> None:
        """Reset the arm to home, open the gripper, place the cup on the table, and refill it."""
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)
        env_ids = env_ids.long()
        if env_ids.numel() == 0:
            return
        model = NewtonManager.get_model()
        s0, s1 = NewtonManager.get_state_0(), NewtonManager.get_state_1()
        n = env_ids.numel()

        # Arm -> home, fingers -> open, in both the robot view and the Newton states.
        arm_q = self._default_arm_q[env_ids].clone()
        zero_vel = torch.zeros_like(arm_q)
        self._robot.write_joint_position_to_sim_index(position=arm_q, joint_ids=self._arm_joint_ids, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=zero_vel, joint_ids=self._arm_joint_ids, env_ids=env_ids)
        self._robot.set_joint_position_target_index(target=arm_q, joint_ids=self._arm_joint_ids, env_ids=env_ids)
        finger_open = torch.full((n, len(FINGER_JOINTS)), float(self.cfg.gripper_open_pos), device=self.device)
        self._robot.write_joint_position_to_sim_index(
            position=finger_open, joint_ids=self._finger_joint_ids, env_ids=env_ids
        )
        self._robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(finger_open), joint_ids=self._finger_joint_ids, env_ids=env_ids
        )
        self._robot.set_joint_position_target_index(
            target=finger_open, joint_ids=self._finger_joint_ids, env_ids=env_ids
        )

        # Cup -> resting pose on the table (free body): write the free-joint coords (pos + xyzw quat)
        # and zero the cup joint velocity. The arm/finger joint_q were written via the robot view; we
        # also re-assert the arm joint_q + zero the arm/cup joint velocities directly in the Newton
        # states, then eval_fk so body_q reflects the reset pose.
        cup_pos = torch.as_tensor(self._cup_reset_pos, device=self.device, dtype=torch.float32)
        cup_quat = torch.as_tensor(self._cup_reset_quat, device=self.device, dtype=torch.float32)
        cup_world = cup_pos.unsqueeze(0) + self.env_origins[env_ids]
        cup_q_base = self._cup_joint_q[env_ids]
        cup_qd_base = self._cup_joint_qd[env_ids]
        articulation_mask = wp.from_torch(
            boolean_selection_mask(model.articulation_count, self._reset_articulation_ids[env_ids]),
            dtype=wp.bool,
        )
        states = (s0,) if s0 is s1 else (s0, s1)
        for state in states:
            jq = wp.to_torch(state.joint_q)
            jqd = wp.to_torch(state.joint_qd)
            for k in range(3):
                jq[cup_q_base + k] = cup_world[:, k]
            for k in range(4):
                jq[cup_q_base + 3 + k] = cup_quat[k]
            for k in range(6):
                jqd[cup_qd_base + k] = 0.0
            jq[self._arm_q_ids[env_ids]] = arm_q
            # q and qd layouts diverge after every free cup joint (7 q coordinates vs 6 qd
            # coordinates). Reusing q indices for qd happened to fit four worlds, then indexed two
            # elements past the qd buffer at eight worlds.
            jqd[self._arm_qd_ids[env_ids]] = 0.0
            newton.eval_fk(model, state.joint_q, state.joint_qd, state, articulation_mask)

        # Re-fill the cup cavity from the selected cups' live reset poses.
        new_p = self._sample_cup_media(cup_world, cup_quat.expand(n, -1))
        self._media.write_particle_pos_to_sim_index(new_p, env_ids=env_ids)
        self._media.write_particle_velocity_to_sim_index(torch.zeros_like(new_p), env_ids=env_ids)
        NewtonManager.reset_solver_state(
            world_mask=wp.from_torch(boolean_selection_mask(self.num_envs, env_ids), dtype=wp.bool),
        )
        self._containment_cache = None
        self._containment_cache_step = -1

    def _sample_cup_media(self, cup_pos: torch.Tensor, cup_quat: torch.Tensor) -> torch.Tensor:
        """Transform the local media lattice into selected cup poses on the simulation device."""
        particle_count = self._media_local_points_t.shape[0]
        local_points = self._media_local_points_t.unsqueeze(0).expand(cup_pos.shape[0], -1, -1)
        quaternions = cup_quat.unsqueeze(1).expand(-1, particle_count, -1)
        return math_utils.quat_apply(quaternions, local_points) + cup_pos.unsqueeze(1)
