# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based RL env: Franka scoop-bowl transferring MPM media between two containers.

An open bowl mesh, matching the particle-pour demo bowl and scaled down, is
synced to the Franka hand as a fixed, turnable end-effector. The source and
target bowls are visual USD assets backed by simple Newton collision proxies so
MPM does not rasterize dense visual meshes every step.
The arm is controlled in 4 DoF via Newton differential IK: the bowl-centre
position (x, y, z) plus a pitch that tilts the bowl to scoop and to pour. A Newton
``CoupledSolverCfg`` (MuJoCo arm + implicit MPM with static/kinematic colliders) advances
arm and media together in ``sim.step()``. The goal is to transfer media from a
source container into an empty target container; observations are a top-down
heightfield + proprioception for the actor, with privileged particle counts given
to the critic only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import newton
import numpy as np
import torch
import warp as wp
from isaaclab_newton.ik.newton_ik_solver import NewtonIKPoseObjective, NewtonIKSolver
from isaaclab_newton.ik.newton_ik_solver_cfg import NewtonIKSolverCfg
from isaaclab_newton.physics import NewtonManager
from newton.solvers import SolverImplicitMPM

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import math as math_utils
from isaaclab.utils.assets import check_file_path, retrieve_file_path

if TYPE_CHECKING:
    from .scoop_env_cfg import FrankaScoopEnvCfg

ARM_JOINTS = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
RIGID_ENTRY = "arm"
MPM_ENTRY = "media"
# Same procedural open bowl shape as scripts/demos/newton_cup_pour_mpm.py, scaled by
# cfg.ee_bowl_scale for the robot end-effector.
POUR_BOWL_INNER_BOTTOM_RADIUS = 0.045
POUR_BOWL_INNER_TOP_RADIUS = 0.19
POUR_BOWL_WALL_THICKNESS = 0.025
POUR_BOWL_HEIGHT = 0.13
POUR_BOWL_BOTTOM_THICKNESS = 0.025


def _q_rot_arr(q, v: np.ndarray) -> np.ndarray:
    """Rotate an (N, 3) array of vectors by xyzw quaternion ``q`` (vectorized)."""
    xyz = np.asarray(q[:3], dtype=np.float64)
    t = 2.0 * np.cross(np.broadcast_to(xyz, v.shape), v)
    return v + float(q[3]) * t + np.cross(np.broadcast_to(xyz, v.shape), t)


def _q_inv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _q_rot(q, v):
    xyz = np.asarray(q[:3], dtype=np.float64)
    t = 2.0 * np.cross(xyz, v)
    return np.asarray(v, dtype=np.float64) + float(q[3]) * t + np.cross(xyz, t)


def _q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def _compose_pos_quat(parent_pos, parent_quat, child_pos, child_quat) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(parent_pos, dtype=np.float64) + _q_rot(parent_quat, child_pos)
    quat = _q_mul(parent_quat, child_quat)
    return pos, quat / (np.linalg.norm(quat) + 1.0e-12)


def _create_open_bowl_mesh(
    *,
    inner_bottom_radius: float,
    inner_top_radius: float,
    wall_thickness: float,
    height: float,
    bottom_thickness: float,
    num_segments: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the simple flared open bowl mesh used by the particle-pour demo."""
    theta = np.linspace(0.0, 2.0 * math.pi, num_segments, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    outer_bottom_radius = inner_bottom_radius + wall_thickness
    outer_top_radius = inner_top_radius + wall_thickness

    def ring(radius: float, z: float) -> np.ndarray:
        return np.column_stack([radius * cos_t, radius * sin_t, np.full(num_segments, z)])

    inner_bottom = ring(inner_bottom_radius, bottom_thickness)
    inner_top = ring(inner_top_radius, height)
    outer_top = ring(outer_top_radius, height)
    outer_bottom = ring(outer_bottom_radius, 0.0)
    inner_center = np.array([[0.0, 0.0, bottom_thickness]], dtype=np.float32)
    outer_center = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    vertices = np.vstack([inner_bottom, inner_top, outer_top, outer_bottom, inner_center, outer_center]).astype(
        np.float32
    )
    inner_center_id = 4 * num_segments
    outer_center_id = inner_center_id + 1

    indices: list[int] = []
    for i in range(num_segments):
        j = (i + 1) % num_segments
        ib_i, ib_j = i, j
        it_i, it_j = i + num_segments, j + num_segments
        ot_i, ot_j = i + 2 * num_segments, j + 2 * num_segments
        ob_i, ob_j = i + 3 * num_segments, j + 3 * num_segments

        indices.extend([ib_i, it_i, ib_j])
        indices.extend([ib_j, it_i, it_j])
        indices.extend([ob_i, ob_j, ot_i])
        indices.extend([ot_i, ob_j, ot_j])
        indices.extend([it_i, ot_i, it_j])
        indices.extend([it_j, ot_i, ot_j])
        indices.extend([inner_center_id, ib_i, ib_j])
        indices.extend([outer_center_id, ob_j, ob_i])

    return vertices, np.asarray(indices, dtype=np.int32)


def _create_open_bucket_mesh(
    *,
    inner_radius: float,
    wall_thickness: float,
    height: float,
    bottom_thickness: float,
    num_segments: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a watertight-bottom, open-top cylindrical bucket mesh."""
    return _create_open_bowl_mesh(
        inner_bottom_radius=inner_radius,
        inner_top_radius=inner_radius,
        wall_thickness=wall_thickness,
        height=height,
        bottom_thickness=bottom_thickness,
        num_segments=num_segments,
    )


def _qrot_t(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    xyz, w = q[..., :3], q[..., 3:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to an xyzw quaternion."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def _author_xform(prim: Usd.Prim, pos, quat_xyzw=(0.0, 0.0, 0.0, 1.0), scale=None) -> None:
    """Author a compact TRS stack on a visual USD prim."""
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    x, y, z, w = [float(v) for v in quat_xyzw]
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    if scale is not None:
        if np.isscalar(scale):
            scale = (float(scale), float(scale), float(scale))
        xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2]))
        )


def _author_color(prim: Usd.Prim, color) -> None:
    if prim.IsA(UsdGeom.Gprim):
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])


def _qmul_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )


@wp.kernel
def _sync_scoop_bowl_body_kernel(
    source_body_q: wp.array(dtype=wp.transform),
    target_body_q: wp.array(dtype=wp.transform),
    target_body_qd: wp.array(dtype=wp.spatial_vector),
    hand_ids: wp.array(dtype=wp.int32),
    scoop_body_ids: wp.array(dtype=wp.int32),
    weld_pos: wp.vec3,
    weld_rot: wp.quat,
) -> None:
    env_id = wp.tid()
    hand_id = hand_ids[env_id]
    scoop_id = scoop_body_ids[env_id]

    hand_pose = source_body_q[hand_id]
    hand_pos = wp.transform_get_translation(hand_pose)
    hand_rot = wp.transform_get_rotation(hand_pose)
    scoop_pos = hand_pos + wp.quat_rotate(hand_rot, weld_pos)
    scoop_rot = hand_rot * weld_rot

    target_body_q[scoop_id] = wp.transform(scoop_pos, scoop_rot)
    target_body_qd[scoop_id] = wp.spatial_vector(
        wp.vec3(0.0, 0.0, 0.0),
        wp.vec3(0.0, 0.0, 0.0),
    )


@wp.kernel
def _pin_fixed_joints_kernel(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joint_q_ids: wp.array2d(dtype=wp.int32),
    joint_qd_ids: wp.array2d(dtype=wp.int32),
    joint_pos: float,
):
    env_id, joint_id = wp.tid()
    joint_q[joint_q_ids[env_id, joint_id]] = joint_pos
    joint_qd[joint_qd_ids[env_id, joint_id]] = 0.0


class FrankaScoopEnv(ManagerBasedRLEnv):
    """Franka scoop two-container MPM transfer environment (Newton coupled MPM)."""

    cfg: FrankaScoopEnvCfg

    def __init__(self, cfg: FrankaScoopEnvCfg, render_mode: str | None = None, **kwargs):
        self._origins_np = self._make_origin_grid(cfg.scene.num_envs, cfg.scene.env_spacing)
        self._table_height = 2.0 * float(cfg.table_half[2])
        self._prepare_newton_extras(cfg)
        self._install_newton_builder_hook()
        try:
            super().__init__(cfg, render_mode, **kwargs)
        finally:
            self._remove_newton_builder_hook()

    def load_managers(self) -> None:
        self._setup_after_physics()
        self._setup_scoop_bowl_body_sync()
        super().load_managers()
        # Author task USD visuals (cup/box/table/media points) only when Kit renders them, so headless
        # training neither pays for the authoring nor pulls the remote gripped-cup USD. _setup_kit_visual_sync
        # then registers the per-render cup-body + media-points sync (a no-op off the Kit visualizer).
        if "kit" in set(self.sim.resolve_visualizer_types()):
            self.spawn_kit_visuals()
        self._setup_kit_visual_sync()

    def step(self, action: torch.Tensor):
        result = super().step(action)
        self._pin_gripper_open_states(update_fk=True)
        self._sync_scoop_bowl_body()
        return result

    # ------------------------------------------------------------------ build
    @staticmethod
    def _make_origin_grid(num_envs: int, spacing: float) -> np.ndarray:
        cols = max(1, int(math.ceil(math.sqrt(num_envs))))
        rows = max(1, int(math.ceil(num_envs / cols)))
        o = np.zeros((num_envs, 3), dtype=np.float64)
        for i in range(num_envs):
            o[i, 0] = (i % cols) * spacing - 0.5 * (cols - 1) * spacing
            o[i, 1] = (i // cols) * spacing - 0.5 * (rows - 1) * spacing
        return o

    @staticmethod
    def _container_geometry_kind(cfg: FrankaScoopEnvCfg) -> str:
        kind = str(getattr(cfg, "container_geometry", "") or "").strip().lower()
        if not kind:
            kind = "pour_bowl" if getattr(cfg, "use_pour_bowl_mesh", False) else "box"
        aliases = {
            "cylinder": "bucket",
            "cylindrical": "bucket",
            "buckets": "bucket",
            "bowl": "pour_bowl",
            "pour-bowl": "pour_bowl",
            "pour_bowls": "pour_bowl",
        }
        kind = aliases.get(kind, kind)
        if kind not in {"bucket", "pour_bowl", "box"}:
            raise ValueError(f"Unsupported container_geometry={kind!r}; expected 'bucket', 'pour_bowl', or 'box'.")
        return kind

    @staticmethod
    def _configure_solver_container_patterns(cfg: FrankaScoopEnvCfg, kind: str) -> None:
        """Keep strict Newton body selectors aligned with the active container geometry."""
        rigid_patterns = {
            "bucket": [r".*/SourceBucketRigid$", r".*/TargetBucketRigid$"],
            "pour_bowl": [r".*/SourceBowlRigid$", r".*/TargetBowlRigid$"],
            # The pile box uses two co-located kinematic bodies (the solver forbids one body in both entries):
            # the hidden rigid proxy "{Source,Target}BoxRigid" blocks the cup here, while "{Source,Target}Box"
            # retains the pile in the MPM entry below.
            "box": [r".*/SourceBoxRigid$", r".*/TargetBoxRigid$"],
        }[kind]
        media_patterns = {
            "bucket": [r".*/ScoopBowl$", r".*/SourceBucket$", r".*/TargetBucket$"],
            "pour_bowl": [r".*/ScoopBowl$", r".*/SourceBowl$", r".*/TargetBowl$"],
            # Pile-retaining boxes are kinematic bodies (not static), so they are selected by pattern and
            # included as MPM colliders without include_static_shapes (which would pull in the giant ground).
            "box": [r".*/ScoopBowl$", r".*/SourceBox$", r".*/TargetBox$"],
        }[kind]
        for entry in cfg.sim.physics.solver_cfg.entries:
            if entry.name == RIGID_ENTRY:
                entry.body_label_patterns = list(rigid_patterns)
            elif entry.name == MPM_ENTRY:
                entry.body_label_patterns = list(media_patterns)

    def _prepare_newton_extras(self, cfg: FrankaScoopEnvCfg) -> None:
        self._container_geometry = self._container_geometry_kind(cfg)
        self._configure_solver_container_patterns(cfg, self._container_geometry)
        self._init_scoop_bowl_geometry(cfg)
        if cfg.gripped_cup_usd_path:
            # Physical side-grasp convention: cup +Z maps to panda_hand +X (upright at the
            # configured hand_home_quat), the cup centerline is between the fixed-open fingers,
            # and the near wall clears the front of the Panda hand collision by gripped_cup_base_clearance.
            self._weld_rot = np.array([0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)], dtype=np.float64)
            outer_r = self._ee_bowl_inner_top_radius + self._ee_bowl_wall_thickness
            self._bowl_center_hand = np.array(
                [0.0, 0.0, float(cfg.gripped_cup_hand_front_z) + outer_r + float(cfg.gripped_cup_base_clearance)],
                dtype=np.float64,
            )
            bc = _q_rot(self._weld_rot, self._ee_bowl_center_local)
            self._weld_pos = self._bowl_center_hand - bc
        else:
            # Procedural-bowl fallback: keep the old world-level bowl behavior.
            home_q = np.array(cfg.hand_home_quat, dtype=np.float64)
            home_q = home_q / (np.linalg.norm(home_q) + 1.0e-12)
            self._weld_rot = _q_inv(home_q)
            bc = _q_rot(self._weld_rot, self._ee_bowl_center_local)  # cavity centre in hand frame
            # Place the bowl centre on the Panda gripper midpoint, i.e. the centerline between the open fingers.
            self._weld_pos = np.array([-bc[0], -bc[1], cfg.bowl_reach - bc[2]])
            self._bowl_center_hand = self._weld_pos + bc  # = (0, 0, bowl_reach) in the hand frame

        self._custom_proto, self._custom_meta = self._build_custom_proto(cfg)
        self._hand_ids_l, self._scoop_body_ids_l = [], []
        self._arm_q_l, self._arm_qd_l, self._finger_q_l, self._finger_qd_l, self._psrc_l = [], [], [], [], []
        self._authored_visual_envs: set[int] = set()
        # (usd points attribute, global particle_q start, count) per env, rewritten each render by
        # _update_media_particles_visual.
        self._media_particle_prims: list[tuple[Usd.Attribute, int, int]] = []
        self._media_authored_envs: set[int] = set()
        # (translate op, orient op) per env on the cup VISUAL prim, rewritten each render from the live cup
        # body pose by _update_cup_visual_xform (the cup body's own Fabric worldMatrix is not synced -- see
        # _sync_kit_visuals).
        self._cup_visual_ops: list[tuple[UsdGeom.XformOp, UsdGeom.XformOp]] = []
        # Optional Kit obs-debug prims (cfg.debug_vis_obs): heightfield grid + centroid/target markers, both
        # rewritten each render by _update_obs_debug_visual.
        self._obs_hf_prims: list[tuple[Usd.Attribute, Usd.Attribute, np.ndarray, float]] = []
        self._obs_marker_prims: list[tuple[Usd.Attribute, np.ndarray]] = []
        self._obs_vis_envs: set[int] = set()
        self._resolved_gripped_cup_source_path = None
        self._resolved_gripped_cup_usd_path = None
        self._builder_hook = None

    def _install_newton_builder_hook(self) -> None:
        hooks = getattr(NewtonManager, "_per_world_builder_hooks", None)
        if hooks is None:
            hooks = []
            NewtonManager._per_world_builder_hooks = hooks
        self._builder_hook = self._add_scoop_world_to_builder
        hooks.append(self._builder_hook)

    def _remove_newton_builder_hook(self) -> None:
        hook = getattr(self, "_builder_hook", None)
        if hook is None:
            return
        hooks = getattr(NewtonManager, "_per_world_builder_hooks", None)
        if hooks is not None and hook in hooks:
            hooks.remove(hook)
        self._builder_hook = None

    def _add_scoop_world_to_builder(self, builder, env_id: int, position, quaternion) -> None:
        env_root = f"/World/envs/env_{env_id}"
        self._origins_np[env_id] = np.asarray(position, dtype=np.float64)
        self._author_custom_stage_prims(env_id)
        self._disable_robot_particle_collision(builder, env_id)
        self._hide_robot_visual_only_shapes(builder, env_id)

        hand = self._find_world_body(builder, env_id, "panda_hand")
        arm_q, arm_qd = self._find_world_arm_joint_coords(builder, env_id)
        finger_q, finger_qd = self._find_world_joint_coords(builder, env_id, FINGER_JOINTS)
        self._add_scoop_bowl_rigid_proxy(builder, self.cfg, hand, env_root)

        b_off = builder.body_count
        p_off = builder.particle_count
        label_starts = self._builder_label_starts(builder)
        builder.add_builder(
            self._custom_proto,
            xform=wp.transform(wp.vec3(*[float(v) for v in position]), wp.quat(*[float(v) for v in quaternion])),
        )
        self._rewrite_builder_labels(builder, label_starts, env_root)
        self._filter_fixed_base_table_contacts(builder, env_root)

        self._hand_ids_l.append(hand)
        self._scoop_body_ids_l.append(b_off + self._custom_meta["scoop_body"])
        self._arm_q_l.append(arm_q)
        self._arm_qd_l.append(arm_qd)
        self._finger_q_l.append(finger_q)
        self._finger_qd_l.append(finger_qd)
        self._psrc_l.append(list(range(p_off, p_off + self._custom_meta["npart"])))

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

    def _find_world_joint_coords(self, builder, env_id: int, joint_names: Sequence[str]) -> tuple[list[int], list[int]]:
        joint_labels = [str(label) for label in builder.joint_label]
        joint_q, joint_qd = [], []
        for joint_name in joint_names:
            matches = [
                joint_id
                for joint_id, label in enumerate(joint_labels)
                if self._label_world(builder, "joint", joint_id) == env_id and label.rsplit("/", 1)[-1] == joint_name
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one {joint_name!r} joint in Newton world {env_id}, found {matches}.")
            joint_id = matches[0]
            joint_q.append(int(builder.joint_q_start[joint_id]))
            joint_qd.append(int(builder.joint_qd_start[joint_id]))
        return joint_q, joint_qd

    def _find_world_arm_joint_coords(self, builder, env_id: int) -> tuple[list[int], list[int]]:
        return self._find_world_joint_coords(builder, env_id, ARM_JOINTS)

    def _disable_robot_particle_collision(self, builder, env_id: int) -> None:
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape_id in range(builder.shape_count):
            body_id = int(builder.shape_body[shape_id])
            if body_id < 0 or self._label_world(builder, "body", body_id) != env_id:
                continue
            body_label = str(builder.body_label[body_id])
            if "/Robot/" in body_label or body_label.endswith("/Robot"):
                builder.shape_flags[shape_id] &= ~collide_particles

    def _hide_robot_visual_only_shapes(self, builder, env_id: int) -> None:
        """Hide imported Panda visual meshes from Newton viewers while keeping collision shapes."""
        if not bool(getattr(self.cfg, "hide_robot_visual_shapes_in_newton", True)):
            return
        visible = int(newton.ShapeFlags.VISIBLE)
        collide_mask = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape_id in range(builder.shape_count):
            body_id = int(builder.shape_body[shape_id])
            if body_id < 0 or self._label_world(builder, "body", body_id) != env_id:
                continue
            body_label = str(builder.body_label[body_id])
            shape_label = str(builder.shape_label[shape_id])
            is_robot_shape = body_label.startswith(f"/World/envs/env_{env_id}/Robot/") or shape_label.startswith(
                f"/World/envs/env_{env_id}/Robot/"
            )
            if not is_robot_shape:
                continue
            if int(builder.shape_flags[shape_id]) & collide_mask:
                continue
            builder.shape_flags[shape_id] &= ~visible

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
        """Rewrite asset-local labels to the live IsaacLab env prim paths used by Kit sync."""
        for attr, start in starts.items():
            labels = getattr(builder, attr)
            for idx in range(start, len(labels)):
                labels[idx] = cls._env_label(labels[idx], env_root)

    @staticmethod
    def _filter_fixed_base_table_contacts(builder, env_root: str) -> None:
        """Avoid fixed Panda base/table contacts while preserving arm/cup/table contacts."""
        table_shape_ids = [sid for sid, label in enumerate(builder.shape_label) if str(label) == f"{env_root}/Table"]
        if not table_shape_ids:
            return
        base_shape_ids = []
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        for sid in range(builder.shape_count):
            if not (int(builder.shape_flags[sid]) & collide_shapes):
                continue
            body = int(builder.shape_body[sid])
            if body < 0:
                continue
            body_name = str(builder.body_label[body]).rsplit("/", 1)[-1]
            if body_name == "panda_link0":
                base_shape_ids.append(sid)
        existing_pairs = set(tuple(pair) for pair in getattr(builder, "shape_collision_filter_pairs", []))
        for table_sid in table_shape_ids:
            for base_sid in base_shape_ids:
                pair = (min(int(table_sid), int(base_sid)), max(int(table_sid), int(base_sid)))
                if pair in existing_pairs:
                    continue
                builder.add_shape_collision_filter_pair(*pair)
                existing_pairs.add(pair)

    def spawn_kit_visuals(self) -> None:
        """Author task USD visuals for custom Newton bodies, plus the live MPM media points prim.

        The static visuals (cup/box/table) are also authored during builder construction; the media points
        need particle bookkeeping (:attr:`_particle_ids`) and so are authored here, once it exists (Kit path).
        """
        for env_id in range(self.cfg.scene.num_envs):
            self._author_custom_stage_prims(env_id, self.sim.stage)
            self._spawn_media_particles_visual(self.sim.stage, f"/World/envs/env_{env_id}", env_id)
            self._spawn_obs_debug_visual(self.sim.stage, f"/World/envs/env_{env_id}", env_id)
        self._update_media_particles_visual()  # seed points before the first render
        self._update_obs_debug_visual()

    def _spawn_media_particles_visual(self, stage: Usd.Stage, root: str, env_id: int) -> None:
        """Author a UsdGeom.Points prim Kit renders as this env's live MPM media.

        Kit renders the prim's USD ``points``; :meth:`_update_media_particles_visual` rewrites them from the
        live world-space ``particle_q`` every render (mirroring the IsaacLab-mpm pour demo's KitParticlePoints).
        A direct USD write is used because Kit does not re-render a bare Points prim through the Fabric
        ``sync_particles_to_usd`` path used for deformable meshes.
        """
        if env_id in self._media_authored_envs:
            return
        particle_ids = getattr(self, "_particle_ids", None)
        if particle_ids is None or env_id >= int(particle_ids.shape[0]):
            return
        ids = particle_ids[env_id]  # contiguous global particle_q indices for this env
        count = int(ids.shape[0])
        if count == 0:
            return
        offset = int(ids[0].item())
        points = UsdGeom.Points.Define(stage, f"{root}/MediaParticles")
        width = float(self.cfg.voxel_size) / max(float(self.cfg.particles_per_cell), 1.0)
        points.CreateWidthsAttr(Vt.FloatArray([width] * count))
        _author_color(points.GetPrim(), (0.85, 0.72, 0.45))
        # (points attribute, global particle_q start, count) -- updated each render by _update_media_particles_visual.
        self._media_particle_prims.append((points.GetPointsAttr(), offset, count))
        self._media_authored_envs.add(env_id)

    def _update_media_particles_visual(self) -> None:
        """Rewrite each media points prim's USD ``points`` from the live world-space ``particle_q``.

        Direct per-render USD write (the IsaacLab-mpm ``KitParticlePoints`` pattern); no Fabric tagging,
        since Kit does not re-render a bare Points prim through ``sync_particles_to_usd``.
        """
        if not self._media_particle_prims:
            return
        state = NewtonManager.get_state_0()
        if state is None or getattr(state, "particle_q", None) is None:
            return
        pq = wp.to_torch(state.particle_q)
        with Sdf.ChangeBlock():
            for points_attr, offset, count in self._media_particle_prims:
                pts = pq[offset : offset + count].detach().cpu().numpy().astype(np.float32, copy=False)
                points_attr.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(pts)))

    def _spawn_obs_debug_visual(self, stage: Usd.Stage, root: str, env_id: int) -> None:
        """Author the obs-debug points (cfg.debug_vis_obs): the heightfield grid + centroid/target markers.

        Both are :class:`UsdGeom.Points` updated each render by :meth:`_update_obs_debug_visual` (the same
        direct-USD-write path the media points use), so they visualize exactly what the policy observes.
        """
        if not getattr(self.cfg, "debug_vis_obs", False) or env_id in self._obs_vis_envs:
            return
        hsz = self._hf_n
        lo = np.asarray(self.cfg.heightfield_lo, dtype=np.float64)
        hi = np.asarray(self.cfg.heightfield_hi, dtype=np.float64)
        origin = self._origins_np[env_id]
        idx = np.arange(hsz * hsz)
        # flattened index k = py*hsz + px (matches heightfield()); cell centres in world xy
        gx = lo[0] + (idx % hsz + 0.5) / hsz * (hi[0] - lo[0]) + origin[0]
        gy = lo[1] + (idx // hsz + 0.5) / hsz * (hi[1] - lo[1]) + origin[1]
        grid_xy = np.column_stack([gx, gy]).astype(np.float32)
        cell = float(hi[0] - lo[0]) / hsz

        hf = UsdGeom.Points.Define(stage, f"{root}/ObsHeightfield")
        hf.CreateWidthsAttr(Vt.FloatArray([0.55 * cell] * (hsz * hsz)))
        hf_col = hf.CreateDisplayColorPrimvar("vertex")
        hf_col.Set(Vt.Vec3fArray([Gf.Vec3f(0.5, 0.5, 0.5)] * (hsz * hsz)))

        mk = UsdGeom.Points.Define(stage, f"{root}/ObsMarkers")
        mk.CreateWidthsAttr(Vt.FloatArray([0.035, 0.035, 0.035, 0.04]))  # src, held, all, bowl-target
        mk_col = mk.CreateDisplayColorPrimvar("vertex")
        mk_col.Set(
            Vt.Vec3fArray(
                [Gf.Vec3f(1.0, 0.2, 0.2), Gf.Vec3f(0.2, 1.0, 0.2), Gf.Vec3f(0.2, 0.8, 1.0), Gf.Vec3f(1.0, 1.0, 0.2)]
            )
        )
        self._obs_hf_prims.append((hf.GetPointsAttr(), hf_col.GetAttr(), grid_xy, float(origin[2])))
        self._obs_marker_prims.append((mk.GetPointsAttr(), origin.astype(np.float32)))
        self._obs_vis_envs.add(env_id)

    def _update_obs_debug_visual(self) -> None:
        """Rewrite the obs-debug points from the live observations: heightfield surface + media centroids."""
        if not self._obs_hf_prims:
            return
        hf = self.heightfield().detach().cpu().numpy()  # (n, H*W) in [0,1]
        lo_z = float(self.cfg.heightfield_lo[2])
        rng_z = float(self.cfg.heightfield_hi[2]) - lo_z
        src = np.nan_to_num(self.source_media_centroid_e().detach().cpu().numpy())
        held = np.nan_to_num(self.bowl_media_centroid_e().detach().cpu().numpy())
        allc = np.nan_to_num(self.all_media_centroid_e().detach().cpu().numpy())
        tgt = np.nan_to_num(self._target_bowl_e.detach().cpu().numpy())
        with Sdf.ChangeBlock():
            for env_id, (pts_attr, col_attr, grid_xy, oz) in enumerate(self._obs_hf_prims):
                h = hf[env_id]
                pos = np.column_stack([grid_xy[:, 0], grid_xy[:, 1], lo_z + h * rng_z + oz]).astype(np.float32)
                pts_attr.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(pos)))
                # colour by normalized height: low -> blue, high -> red
                col = np.column_stack([h, np.full_like(h, 0.25), 1.0 - h]).astype(np.float32)
                col_attr.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(col)))
                mk_attr, origin = self._obs_marker_prims[env_id]
                mk = np.stack([src[env_id], held[env_id], allc[env_id], tgt[env_id]]).astype(np.float32) + origin
                mk_attr.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(mk)))

    def _author_custom_stage_prims(self, env_id: int, stage: Usd.Stage | None = None) -> None:
        if env_id in self._authored_visual_envs:
            return
        stage = get_current_stage() if stage is None else stage
        root = f"/World/envs/env_{env_id}"
        cfg = self.cfg
        self._spawn_scoop_bowl_visual(stage, root, cfg, env_id)
        self._spawn_table_visual(stage, root, cfg)
        if self._container_geometry == "bucket":
            self._spawn_bucket_visual(stage, root, cfg, cfg.source_center, "Source")
            self._spawn_bucket_visual(stage, root, cfg, cfg.target_center, "Target")
        elif self._container_geometry == "pour_bowl":
            self._spawn_bowl_visual(stage, root, cfg, cfg.source_center, "Source")
            self._spawn_bowl_visual(stage, root, cfg, cfg.target_center, "Target")
        else:
            self._spawn_box_container_visual(stage, root, cfg, cfg.source_center, "Source")
            self._spawn_box_container_visual(stage, root, cfg, cfg.target_center, "Target")
        # NB: media points are authored separately (see spawn_kit_visuals); this method runs during builder
        # construction before particle bookkeeping (_particle_ids) exists.
        self._authored_visual_envs.add(env_id)

    def _spawn_scoop_bowl_visual(self, stage: Usd.Stage, root: str, cfg: FrankaScoopEnvCfg, env_id: int) -> None:
        """Spawn the gripped-cup visual and drive it with an authored world xform updated each render.

        The visual is authored at ``ScoopBowlVisual`` -- deliberately NOT the cup body's label path
        (``ScoopBowl``) -- so :meth:`NewtonManager.start_simulation` does not tag it as a rigid body.
        ``sync_transforms_to_usd`` does not update the cup's (kinematic MPM-collider) Fabric worldMatrix, so a
        body-matched prim would freeze at its start pose; a frozen Fabric worldMatrix would also override any
        authored xform via cubric. Instead :meth:`_update_cup_visual_xform` rewrites the translate/orient ops
        here from the live cup body pose every render, the same per-frame USD-write pattern the media points use.
        """
        prim = stage.DefinePrim(f"{root}/ScoopBowlVisual", "Xform")
        # The decorative coffee-cup USD only matches the flared "mug" shape; the hemisphere ladle renders its
        # own collider mesh so the visual exactly matches the physics.
        if cfg.gripped_cup_usd_path and str(cfg.ee_cup_shape).lower() == "mug":
            self._spawn_gripped_cup_reference(stage, f"{root}/ScoopBowlVisual", cfg)
        else:
            self._spawn_procedural_scoop_bowl_mesh(stage, f"{root}/ScoopBowlVisual")

        # Seed the world xform at the home cup pose; _update_cup_visual_xform refreshes it each render.
        hand_pos = self._origins_np[env_id] + np.asarray(cfg.hand_home_pos, dtype=np.float64)
        scoop_pos, scoop_quat = self._scoop_bowl_pose_from_hand_pose(hand_pos, np.asarray(cfg.hand_home_quat))
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        orient_op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(Gf.Vec3d(*(float(x) for x in scoop_pos)))
        orient_op.Set(Gf.Quatd(float(scoop_quat[3]), Gf.Vec3d(*(float(scoop_quat[i]) for i in range(3)))))
        self._cup_visual_ops.append((translate_op, orient_op))

    def _spawn_procedural_scoop_bowl_mesh(self, stage: Usd.Stage, cup_root: str) -> None:
        mesh = UsdGeom.Mesh.Define(stage, f"{cup_root}/Mesh")
        mesh.CreatePointsAttr([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in self._ee_bowl_vertices])
        mesh.CreateFaceVertexCountsAttr([3] * (len(self._ee_bowl_indices) // 3))
        mesh.CreateFaceVertexIndicesAttr([int(i) for i in self._ee_bowl_indices])
        _author_color(mesh.GetPrim(), (0.95, 0.82, 0.16))

    def _spawn_gripped_cup_reference(self, stage: Usd.Stage, cup_root: str, cfg: FrankaScoopEnvCfg) -> None:
        usd_path = self._resolve_gripped_cup_usd_path(cfg.gripped_cup_usd_path)
        if usd_path is None:
            raise FileNotFoundError(f"Unable to resolve gripped cup USD asset: {cfg.gripped_cup_usd_path}")

        cup_prim = stage.DefinePrim(f"{cup_root}/CupVisual", "Xform")
        refs = cup_prim.GetReferences()
        refs.ClearReferences()
        if cfg.gripped_cup_usd_prim_path:
            refs.AddReference(usd_path, cfg.gripped_cup_usd_prim_path)
        else:
            refs.AddReference(usd_path)

        fit_offset = np.zeros(3, dtype=np.float64)
        fit_scale = 1.0
        fit_ok = True
        if cfg.gripped_cup_auto_fit_visual:
            fit_offset, fit_scale, fit_ok = self._compute_gripped_cup_visual_fit(cup_prim)
        if not fit_ok:
            raise RuntimeError(
                "Unable to fit gripped cup visual. Check gripped_cup_usd_path and "
                f"gripped_cup_usd_prim_path={cfg.gripped_cup_usd_prim_path!r}."
            )

        user_offset = np.asarray(cfg.gripped_cup_visual_offset, dtype=np.float64)
        user_quat = np.asarray(cfg.gripped_cup_visual_quat, dtype=np.float64)
        user_quat = user_quat / (np.linalg.norm(user_quat) + 1.0e-12)
        visual_scale = fit_scale * float(cfg.gripped_cup_visual_scale)
        _author_xform(cup_prim, fit_offset + user_offset, user_quat, visual_scale)
        self._mark_visual_cup_noncolliding(cup_prim)

    def _resolve_gripped_cup_usd_path(self, usd_path: str) -> str | None:
        """Resolve remote gripped-cup USDs to a local dependency-complete cache before referencing."""
        if self._resolved_gripped_cup_source_path == usd_path and self._resolved_gripped_cup_usd_path is not None:
            return self._resolved_gripped_cup_usd_path
        try:
            file_status = check_file_path(usd_path)
            if file_status == 0:
                return None
            resolved_path = retrieve_file_path(usd_path, force_download=False) if file_status == 2 else usd_path
        except Exception:
            return None
        self._resolved_gripped_cup_source_path = usd_path
        self._resolved_gripped_cup_usd_path = resolved_path
        return resolved_path

    def _compute_gripped_cup_visual_fit(self, cup_prim: Usd.Prim) -> tuple[np.ndarray, float, bool]:
        """Fit referenced cup visuals to the explicit open Newton cup collision proxy."""
        try:
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                useExtentsHint=True,
            )
            bounds = bbox_cache.ComputeLocalBound(cup_prim).GetRange()
            if bounds.IsEmpty():
                return np.zeros(3, dtype=np.float64), 1.0, False
            bmin = np.array(bounds.GetMin(), dtype=np.float64)
            bmax = np.array(bounds.GetMax(), dtype=np.float64)
        except Exception:
            return np.zeros(3, dtype=np.float64), 1.0, False

        size = bmax - bmin
        visual_diameter = float(max(size[0], size[1]))
        target_diameter = 2.0 * (self._ee_bowl_inner_top_radius + self._ee_bowl_wall_thickness)
        fit_scale = target_diameter / visual_diameter if visual_diameter > 1.0e-6 else 1.0
        offset = np.array(
            [
                -0.5 * (bmin[0] + bmax[0]) * fit_scale,
                -0.5 * (bmin[1] + bmax[1]) * fit_scale,
                -bmin[2] * fit_scale,
            ],
            dtype=np.float64,
        )
        return offset, fit_scale, True

    @staticmethod
    def _mark_visual_cup_noncolliding(cup_prim: Usd.Prim) -> None:
        """Keep the referenced SimReady cup visual-only; Newton owns collision explicitly."""
        for prim in Usd.PrimRange(cup_prim):
            if prim.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.render)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            for attr_name in ("physics:collisionEnabled", "physics:rigidBodyEnabled"):
                attr = prim.GetAttribute(attr_name)
                if attr.IsValid():
                    attr.Set(False)

    def _scoop_bowl_pose_from_hand_pose(self, hand_pos, hand_quat) -> tuple[np.ndarray, np.ndarray]:
        quat = np.asarray(hand_quat, dtype=np.float64)
        quat = quat / (np.linalg.norm(quat) + 1.0e-12)
        pos = np.asarray(hand_pos, dtype=np.float64) + _q_rot(quat, self._weld_pos)
        scoop_quat = _q_mul(quat, self._weld_rot)
        scoop_quat = scoop_quat / (np.linalg.norm(scoop_quat) + 1.0e-12)
        return pos, scoop_quat

    def _spawn_table_visual(self, stage: Usd.Stage, root: str, cfg: FrankaScoopEnvCfg) -> None:
        hx, hy, hz = cfg.table_half
        cx, cy = cfg.table_center_xy
        cube = UsdGeom.Cube.Define(stage, f"{root}/Table")
        cube.CreateSizeAttr(1.0)
        _author_xform(cube.GetPrim(), (cx, cy, -hz), scale=(2.0 * hx, 2.0 * hy, 2.0 * hz))
        _author_color(cube.GetPrim(), (0.45, 0.38, 0.30))

    def _spawn_bucket_visual(self, stage: Usd.Stage, root: str, cfg: FrankaScoopEnvCfg, center, label: str) -> None:
        cx, cy, _ = center
        base_z = self._bucket_base_z(cfg, center)
        bucket_prim = stage.DefinePrim(f"{root}/{label}Bucket", "Xform")
        _author_xform(bucket_prim, (cx, cy, base_z))
        stage.DefinePrim(f"{root}/{label}BucketRigid", "Xform")
        mesh = UsdGeom.Mesh.Define(stage, f"{root}/{label}Bucket/Mesh")
        mesh.CreatePointsAttr([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in self._bucket_vertices])
        mesh.CreateFaceVertexCountsAttr([3] * (len(self._bucket_indices) // 3))
        mesh.CreateFaceVertexIndicesAttr([int(i) for i in self._bucket_indices])
        col = (0.50, 0.42, 0.28) if label == "Source" else (0.25, 0.46, 0.55)
        _author_color(mesh.GetPrim(), col)

    def _spawn_bowl_visual(self, stage: Usd.Stage, root: str, cfg: FrankaScoopEnvCfg, center, label: str) -> None:
        cx, cy, cz = center
        base_z = cz - cfg.container_inner_half[2]
        scale = self._container_bowl_scale(cfg)
        vertices = (self._container_bowl_vertices * scale).astype(np.float32)
        bowl_prim = stage.DefinePrim(f"{root}/{label}Bowl", "Xform")
        _author_xform(bowl_prim, (cx, cy, base_z))
        stage.DefinePrim(f"{root}/{label}BowlRigid", "Xform")
        mesh = UsdGeom.Mesh.Define(stage, f"{root}/{label}Bowl/Mesh")
        mesh.CreatePointsAttr([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in vertices])
        mesh.CreateFaceVertexCountsAttr([3] * (len(self._container_bowl_indices) // 3))
        mesh.CreateFaceVertexIndicesAttr([int(i) for i in self._container_bowl_indices])
        col = (0.55, 0.43, 0.30) if label == "Source" else (0.30, 0.45, 0.55)
        _author_color(mesh.GetPrim(), col)

    def _spawn_box_container_visual(
        self, stage: Usd.Stage, root: str, cfg: FrankaScoopEnvCfg, center, label: str
    ) -> None:
        cx, cy, cz = center
        ihx, ihy, ihz = cfg.container_inner_half
        w = cfg.container_wall
        col = (0.5, 0.4, 0.3) if label == "Source" else (0.3, 0.45, 0.55)
        # Match the kinematic pile-box collider (_add_container): floor at the table top + SHALLOW walls of
        # half-height pile_box_wall_half (NOT ihz). Previously the visual used ihz, so the Kit walls rendered
        # ~4x taller than the Newton collider.
        base_z = cz - ihz
        wh = float(getattr(cfg, "pile_box_wall_half", ihz))
        pieces = [(0.0, 0.0, base_z - 0.5 * w, ihx + w, ihy + w, 0.5 * w)]
        pieces.extend(
            [
                (ihx + 0.5 * w, 0.0, base_z + wh, 0.5 * w, ihy + w, wh),
                (-(ihx + 0.5 * w), 0.0, base_z + wh, 0.5 * w, ihy + w, wh),
                (0.0, ihy + 0.5 * w, base_z + wh, ihx + w, 0.5 * w, wh),
                (0.0, -(ihy + 0.5 * w), base_z + wh, ihx + w, 0.5 * w, wh),
            ]
        )
        for idx, (dx, dy, pz, hx, hy, hz) in enumerate(pieces):
            cube = UsdGeom.Cube.Define(stage, f"{root}/{label}_{idx}")
            cube.CreateSizeAttr(1.0)
            _author_xform(cube.GetPrim(), (cx + dx, cy + dy, pz), scale=(2.0 * hx, 2.0 * hy, 2.0 * hz))
            _author_color(cube.GetPrim(), col)

    def _init_scoop_bowl_geometry(self, cfg: FrankaScoopEnvCfg) -> None:
        from .cup_mesh import make_cup_collision_mesh, make_hemisphere_scoop_mesh

        shape = str(getattr(cfg, "ee_cup_shape", "hemisphere")).lower()
        if shape == "hemisphere":
            # Simple, grid-robust thick hemispherical ladle: cavity depth == radius, shell thickness == wall.
            # inner_bottom_radius is 0 (the cavity floor is a point); the 5 geom attrs below still parameterize
            # the rigid table proxy (a cylinder approx) and the cone counting region.
            r_in = float(cfg.ee_ladle_radius)
            wall = float(cfg.ee_ladle_wall_thickness)
            self._ee_bowl_inner_bottom_radius = 0.0
            self._ee_bowl_inner_top_radius = r_in
            self._ee_bowl_wall_thickness = wall
            self._ee_bowl_bottom_thickness = wall
            self._ee_bowl_height = r_in + wall
            self._ee_bowl_vertices, self._ee_bowl_indices = make_hemisphere_scoop_mesh(
                inner_radius=r_in, wall_thickness=wall, num_segments=32, num_rings=10
            )
        else:
            if cfg.gripped_cup_usd_path:
                self._ee_bowl_inner_bottom_radius = float(cfg.ee_cup_inner_bottom_radius)
                self._ee_bowl_inner_top_radius = float(cfg.ee_cup_inner_top_radius)
                self._ee_bowl_wall_thickness = float(cfg.ee_cup_wall_thickness)
                self._ee_bowl_height = float(cfg.ee_cup_height)
                self._ee_bowl_bottom_thickness = float(cfg.ee_cup_bottom_thickness)
            else:
                scale = float(cfg.ee_bowl_scale)
                self._ee_bowl_inner_bottom_radius = POUR_BOWL_INNER_BOTTOM_RADIUS * scale
                self._ee_bowl_inner_top_radius = POUR_BOWL_INNER_TOP_RADIUS * scale
                self._ee_bowl_wall_thickness = POUR_BOWL_WALL_THICKNESS * scale
                self._ee_bowl_height = POUR_BOWL_HEIGHT * scale
                self._ee_bowl_bottom_thickness = POUR_BOWL_BOTTOM_THICKNESS * scale
            # Own watertight, thick-walled, low-poly mug asset (vs the thin pour-demo bowl that let media tunnel).
            self._ee_bowl_vertices, self._ee_bowl_indices = make_cup_collision_mesh(
                inner_bottom_radius=self._ee_bowl_inner_bottom_radius,
                inner_top_radius=self._ee_bowl_inner_top_radius,
                wall_thickness=self._ee_bowl_wall_thickness,
                cavity_depth=self._ee_bowl_height - self._ee_bowl_bottom_thickness,
                bottom_thickness=self._ee_bowl_bottom_thickness,
                num_segments=32,
            )
        self._ee_bowl_center_local = np.array(
            [0.0, 0.0, 0.5 * (self._ee_bowl_bottom_thickness + self._ee_bowl_height)], dtype=np.float64
        )
        self._container_bowl_vertices, self._container_bowl_indices = _create_open_bowl_mesh(
            inner_bottom_radius=POUR_BOWL_INNER_BOTTOM_RADIUS,
            inner_top_radius=POUR_BOWL_INNER_TOP_RADIUS,
            wall_thickness=POUR_BOWL_WALL_THICKNESS,
            height=POUR_BOWL_HEIGHT,
            bottom_thickness=POUR_BOWL_BOTTOM_THICKNESS,
        )
        self._bucket_vertices, self._bucket_indices = _create_open_bucket_mesh(
            inner_radius=float(cfg.bucket_inner_radius),
            wall_thickness=float(cfg.bucket_wall_thickness),
            height=float(cfg.bucket_height),
            bottom_thickness=float(cfg.bucket_bottom_thickness),
            num_segments=max(8, int(cfg.bucket_mesh_segments)),
        )
        self._ee_bowl_geom = {
            "inner_bottom_radius": self._ee_bowl_inner_bottom_radius,
            "inner_top_radius": self._ee_bowl_inner_top_radius,
            "wall_thickness": self._ee_bowl_wall_thickness,
            "height": self._ee_bowl_height,
            "bottom_thickness": self._ee_bowl_bottom_thickness,
        }

    @staticmethod
    def _container_bowl_scale(cfg: FrankaScoopEnvCfg) -> float:
        outer_diameter = 2.0 * (POUR_BOWL_INNER_TOP_RADIUS + POUR_BOWL_WALL_THICKNESS)
        return float(cfg.bowl_target_diameter) / outer_diameter

    @staticmethod
    def _bucket_base_z(cfg: FrankaScoopEnvCfg, center) -> float:
        return float(center[2]) - 0.5 * float(cfg.bucket_height)

    def _build_custom_proto(self, cfg: FrankaScoopEnvCfg):
        proto = NewtonManager.create_builder()
        SolverImplicitMPM.register_custom_attributes(proto)
        proto.default_shape_cfg.mu = 0.8
        scoop_body = self._add_scoop_bowl_mpm_body(proto, cfg)
        self._add_table(proto, cfg)
        if self._container_geometry == "bucket":
            self._add_bucket(proto, cfg, cfg.source_center, "Source")
            self._add_bucket(proto, cfg, cfg.target_center, "Target")
        elif self._container_geometry == "pour_bowl":
            self._add_bowl(proto, cfg, cfg.source_center, "Source")
            self._add_bowl(proto, cfg, cfg.target_center, "Target")
        else:
            self._add_container(proto, cfg, cfg.source_center, "Source")
            self._add_container(proto, cfg, cfg.target_center, "Target")
        # Floor plane matches the visible ground below the table; the table box itself is the MPM tabletop.
        proto.add_ground_plane(
            height=-self._table_height,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.8, margin=cfg.collider_margin, has_particle_collision=True),
            color=(0.3, 0.3, 0.3),
        )
        self._assert_mpm_collision_scope(proto, scoop_body)
        p0 = proto.particle_count
        self._add_media(proto, cfg)
        npart = proto.particle_count - p0

        return proto, {"scoop_body": scoop_body, "npart": npart}

    def _add_scoop_bowl_rigid_proxy(self, builder, cfg, hand: int, env_root: str) -> None:
        """Add open rigid scoop geometry to the scene-imported Franka hand."""
        scoop_shape_ids = self._add_open_bowl_rigid_proxy(
            builder,
            cfg,
            hand,
            float(cfg.ee_bowl_scale),
            (0.95, 0.82, 0.16),
            "Scoop",
            mu=cfg.ee_bowl_friction,
            base_pos=self._weld_pos,
            base_quat=self._weld_rot,
            label_root=f"{env_root}/ScoopBowlRigid",
            geometry=self._ee_bowl_geom,
        )
        self._filter_scoop_robot_shape_contacts(builder, env_root, scoop_shape_ids)

    @staticmethod
    def _filter_scoop_robot_shape_contacts(builder, env_root: str, scoop_shape_ids: Sequence[int]) -> None:
        """The gripped cup is welded to the hand; it should not solve self-contacts against the robot."""
        scoop_set = set(int(sid) for sid in scoop_shape_ids)
        existing_pairs = set(tuple(pair) for pair in getattr(builder, "shape_collision_filter_pairs", []))
        robot_shape_ids = []
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        for sid in range(builder.shape_count):
            if sid in scoop_set:
                continue
            if not (int(builder.shape_flags[sid]) & collide_shapes):
                continue
            body = int(builder.shape_body[sid])
            body_label = str(builder.body_label[body]) if body >= 0 else ""
            shape_label = str(builder.shape_label[sid])
            if body_label.startswith(f"{env_root}/Robot/") or shape_label.startswith(f"{env_root}/Robot/"):
                robot_shape_ids.append(sid)
        for scoop_sid in scoop_shape_ids:
            for robot_sid in robot_shape_ids:
                pair = (min(int(scoop_sid), int(robot_sid)), max(int(scoop_sid), int(robot_sid)))
                if pair in existing_pairs:
                    continue
                builder.add_shape_collision_filter_pair(*pair)
                existing_pairs.add(pair)

    def _add_scoop_bowl_mpm_body(self, proto, cfg) -> int:
        """Add the kinematic MPM collider that follows the Franka hand."""
        mesh = newton.Mesh(self._ee_bowl_vertices, self._ee_bowl_indices, compute_inertia=False, is_solid=False)
        cfg_m = newton.ModelBuilder.ShapeConfig(
            mu=cfg.ee_bowl_friction, density=0.0, margin=cfg.collider_margin, has_particle_collision=True
        )
        hand_pos = np.asarray(cfg.hand_home_pos, dtype=np.float64)
        hand_quat = np.asarray(cfg.hand_home_quat, dtype=np.float64)
        scoop_pos, scoop_quat = self._scoop_bowl_pose_from_hand_pose(hand_pos, hand_quat)
        scoop_body = proto.add_body(
            xform=wp.transform(wp.vec3(*scoop_pos.tolist()), wp.quat(*scoop_quat.tolist())),
            mass=0.0,
            is_kinematic=True,
            lock_inertia=True,
            label="/World/ScoopBowl",
        )
        sid = proto.add_shape_mesh(
            scoop_body,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mesh=mesh,
            cfg=cfg_m,
            color=(0.95, 0.82, 0.16),
            label="/World/ScoopBowl/Collision",
        )
        proto.shape_flags[sid] |= int(newton.ShapeFlags.COLLIDE_PARTICLES) | int(newton.ShapeFlags.VISIBLE)
        proto.shape_flags[sid] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
        proto.shape_margin[sid] = cfg.collider_margin
        proto.shape_material_mu[sid] = cfg.ee_bowl_friction
        if proto.shape_source[sid] is not None:
            proto.shape_source[sid].indices = proto.shape_source[sid].indices.reshape(-1)
        proto.body_flags[scoop_body] = int(newton.BodyFlags.KINEMATIC)
        proto.body_mass[scoop_body] = 0.0
        proto.body_inv_mass[scoop_body] = 0.0
        proto.body_inertia[scoop_body] = wp.mat33()
        proto.body_inv_inertia[scoop_body] = wp.mat33()
        return scoop_body

    def _assert_mpm_collision_scope(self, proto, scoop_body: int) -> None:
        """Fail if any non-bowl robot geometry is registered as an MPM collider."""
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        allowed_body_labels = {
            "/World/ScoopBowl",
            "/World/SourceBowl",
            "/World/TargetBowl",
            "/World/SourceBucket",
            "/World/TargetBucket",
            "/World/SourceBox",
            "/World/TargetBox",
        }
        bad = []
        for sid in range(proto.shape_count):
            if not (int(proto.shape_flags[sid]) & collide_particles):
                continue
            body = int(proto.shape_body[sid])
            label = str(proto.shape_label[sid])
            if body < 0:
                continue
            body_label = str(proto.body_label[body])
            if body == scoop_body and "/World/ScoopBowl/Collision" in label:
                continue
            if body_label in allowed_body_labels:
                continue
            bad.append((sid, label, body, body_label))
        if bad:
            raise RuntimeError(f"Only source/target/scoop bowl shapes may collide with MPM particles; found {bad}")

    def _add_container(self, proto, cfg, center, label) -> None:
        # Shallow pile-retaining box on the table. The coupled solver forbids a body from being owned by both the
        # MPM (media) and RIGID (arm) entries, so -- exactly like the cup -- we spawn two co-located kinematic
        # bodies sharing the same Floor + 4-wall geometry:
        #   "/World/{label}Box"      MPM collider (COLLIDE_PARTICLES, visible): retains the pile. Selected by the
        #                            MPM entry; include_static_shapes stays False (the giant static ground plane
        #                            as an MPM collider overflows the collider rasterization at high env counts).
        #   "/World/{label}BoxRigid" rigid collider (COLLIDE_SHAPES, hidden): blocks the rigid cup proxy via
        #                            MuJoCo. Selected by the RIGID entry; hidden so it does not double-render over
        #                            the MPM box.
        # The pile mounds above the shallow walls.
        cx, cy, cz = center
        ihx, ihy, ihz = cfg.container_inner_half
        w = cfg.container_wall
        col = (0.5, 0.4, 0.3) if label == "Source" else (0.3, 0.45, 0.55)
        base_z = cz - ihz  # box floor top == table top (env z=0)
        wh = float(getattr(cfg, "pile_box_wall_half", ihz))
        # (local xform, half-extents, sub-label) for the floor + 4 walls, shared by both bodies.
        box_specs = [(wp.vec3(0.0, 0.0, base_z - 0.5 * w), ihx + w, ihy + w, 0.5 * w, "Floor")]
        wall_specs = (
            (ihx + 0.5 * w, 0, 0.5 * w, ihy + w),
            (-(ihx + 0.5 * w), 0, 0.5 * w, ihy + w),
            (0, ihy + 0.5 * w, ihx + w, 0.5 * w),
            (0, -(ihy + 0.5 * w), ihx + w, 0.5 * w),
        )
        for k, (dx, dy, hx, hy) in enumerate(wall_specs):
            box_specs.append((wp.vec3(dx, dy, base_z + wh), hx, hy, wh, f"Wall_{k}"))

        def _spawn_box_body(suffix: str, particle_collision: bool) -> None:
            c = newton.ModelBuilder.ShapeConfig(
                mu=0.8, density=0.0, margin=cfg.collider_margin, has_particle_collision=particle_collision
            )
            body = proto.add_body(
                xform=wp.transform(wp.vec3(cx, cy, 0.0), wp.quat_identity()),
                mass=0.0,
                is_kinematic=True,
                lock_inertia=True,
                label=f"/World/{label}{suffix}",
            )
            for xf, hx, hy, hz, name in box_specs:
                sid = proto.add_shape_box(
                    body,
                    xform=wp.transform(xf, wp.quat_identity()),
                    hx=hx,
                    hy=hy,
                    hz=hz,
                    cfg=c,
                    color=col,
                    label=f"/World/{label}{suffix}/{name}",
                )
                if particle_collision:
                    proto.shape_flags[sid] |= int(newton.ShapeFlags.COLLIDE_PARTICLES) | int(newton.ShapeFlags.VISIBLE)
                    proto.shape_flags[sid] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
                else:
                    proto.shape_flags[sid] |= int(newton.ShapeFlags.COLLIDE_SHAPES)
                    proto.shape_flags[sid] &= ~(
                        int(newton.ShapeFlags.COLLIDE_PARTICLES) | int(newton.ShapeFlags.VISIBLE)
                    )
                proto.shape_margin[sid] = cfg.collider_margin
            proto.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
            proto.body_mass[body] = 0.0
            proto.body_inv_mass[body] = 0.0
            proto.body_inertia[body] = wp.mat33()
            proto.body_inv_inertia[body] = wp.mat33()

        _spawn_box_body("Box", particle_collision=True)  # MPM collider: retains the pile.
        _spawn_box_body("BoxRigid", particle_collision=False)  # rigid collider: blocks the rigid cup.

    def _add_table(self, proto, cfg) -> None:
        """Bolt-on table box with base on the floor and top at z=0.

        The Franka base is fixed at z=0 (bolted on the table top). The table collides with both
        the robot and MPM media so arm motions cannot pass through the workspace surface and spilled
        media rests on the finite tabletop instead of an infinite plane.
        """
        hx, hy, hz = cfg.table_half
        cx, cy = cfg.table_center_xy
        c = newton.ModelBuilder.ShapeConfig(mu=0.8, density=0.0, margin=cfg.collider_margin)
        sid = proto.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(cx, cy, -hz), wp.quat_identity()),
            hx=hx,
            hy=hy,
            hz=hz,
            cfg=c,
            color=(0.45, 0.38, 0.30),
            label="/World/Table",
        )
        proto.shape_flags[sid] |= (
            int(newton.ShapeFlags.COLLIDE_SHAPES)
            | int(newton.ShapeFlags.COLLIDE_PARTICLES)
            | int(newton.ShapeFlags.VISIBLE)
        )

    def _add_pedestal(self, proto, cfg, xy, top_z, col=(0.32, 0.30, 0.28)) -> None:
        """Small stand from the table top (z=0) up to ``top_z`` under a bowl."""
        if top_z <= 0.02:
            return
        px, py = cfg.pedestal_half
        c = newton.ModelBuilder.ShapeConfig(
            mu=0.8, density=0.0, margin=cfg.collider_margin, has_particle_collision=True
        )
        proto.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(xy[0], xy[1], 0.5 * top_z), wp.quat_identity()),
            hx=px,
            hy=py,
            hz=0.5 * top_z,
            cfg=c,
            color=col,
        )

    def _add_open_bowl_rigid_proxy(
        self,
        proto,
        cfg,
        body: int,
        scale: float,
        col,
        label: str,
        *,
        mu: float = 0.8,
        base_pos=(0.0, 0.0, 0.0),
        base_quat=(0.0, 0.0, 0.0, 1.0),
        label_root: str | None = None,
        geometry: dict[str, float] | None = None,
    ) -> list[int]:
        """Open rigid bowl proxy made from primitives so MuJoCo does not convex-cap the opening."""
        shape_ids: list[int] = []
        c = newton.ModelBuilder.ShapeConfig(mu=mu, density=0.0, margin=cfg.collider_margin)
        label_root = f"/World/{label}BowlRigid" if label_root is None else label_root.rstrip("/")
        if geometry is None:
            inner_bottom = POUR_BOWL_INNER_BOTTOM_RADIUS * scale
            inner_top = POUR_BOWL_INNER_TOP_RADIUS * scale
            wall = POUR_BOWL_WALL_THICKNESS * scale
            height = POUR_BOWL_HEIGHT * scale
            bottom = POUR_BOWL_BOTTOM_THICKNESS * scale
        else:
            inner_bottom = float(geometry["inner_bottom_radius"])
            inner_top = float(geometry["inner_top_radius"])
            wall = float(geometry["wall_thickness"])
            height = float(geometry["height"])
            bottom = float(geometry["bottom_thickness"])
        base_pos = np.asarray(base_pos, dtype=np.float64)
        base_quat = np.asarray(base_quat, dtype=np.float64)
        base_quat = base_quat / (np.linalg.norm(base_quat) + 1.0e-12)

        def local_xform(pos, quat=(0.0, 0.0, 0.0, 1.0)):
            p, q = _compose_pos_quat(base_pos, base_quat, np.asarray(pos, dtype=np.float64), quat)
            return wp.transform(wp.vec3(*p.tolist()), wp.quat(*q.tolist()))

        bottom_sid = proto.add_shape_cylinder(
            body,
            xform=local_xform((0.0, 0.0, 0.5 * bottom)),
            radius=inner_bottom + wall,
            half_height=0.5 * bottom,
            cfg=c,
            color=col,
            label=f"{label_root}/Bottom",
        )
        shape_ids.append(bottom_sid)
        proto.shape_flags[bottom_sid] |= int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.VISIBLE)
        proto.shape_flags[bottom_sid] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)

        segments = max(8, int(cfg.rigid_bowl_wall_segments))
        center_radius = inner_top + 0.5 * wall
        radial_half = 0.5 * wall
        tangent_half = center_radius * math.tan(math.pi / segments) * 1.08
        wall_center_z = 0.5 * (bottom + height)
        wall_half_z = 0.5 * (height - bottom)
        for i in range(segments):
            theta = 2.0 * math.pi * float(i) / float(segments)
            ct, st = math.cos(theta), math.sin(theta)
            q = np.array((0.0, 0.0, math.sin(0.5 * theta), math.cos(0.5 * theta)), dtype=np.float64)
            sid = proto.add_shape_box(
                body,
                xform=local_xform((center_radius * ct, center_radius * st, wall_center_z), q),
                hx=radial_half,
                hy=tangent_half,
                hz=wall_half_z,
                cfg=c,
                color=col,
                label=f"{label_root}/Wall_{i:02d}",
            )
            shape_ids.append(sid)
            proto.shape_flags[sid] |= int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.VISIBLE)
            proto.shape_flags[sid] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
        return shape_ids

    def _add_open_bucket_rigid_proxy(
        self,
        proto,
        cfg,
        body: int,
        col,
        label: str,
        *,
        mu: float = 0.8,
        base_pos=(0.0, 0.0, 0.0),
        base_quat=(0.0, 0.0, 0.0, 1.0),
        label_root: str | None = None,
    ) -> list[int]:
        """Open cylindrical bucket proxy made from rigid primitives for MuJoCo contacts."""
        shape_ids: list[int] = []
        c = newton.ModelBuilder.ShapeConfig(mu=mu, density=0.0, margin=cfg.collider_margin)
        label_root = f"/World/{label}BucketRigid" if label_root is None else label_root.rstrip("/")
        inner = float(cfg.bucket_inner_radius)
        wall = float(cfg.bucket_wall_thickness)
        height = float(cfg.bucket_height)
        bottom = float(cfg.bucket_bottom_thickness)
        base_pos = np.asarray(base_pos, dtype=np.float64)
        base_quat = np.asarray(base_quat, dtype=np.float64)
        base_quat = base_quat / (np.linalg.norm(base_quat) + 1.0e-12)

        def local_xform(pos, quat=(0.0, 0.0, 0.0, 1.0)):
            p, q = _compose_pos_quat(base_pos, base_quat, np.asarray(pos, dtype=np.float64), quat)
            return wp.transform(wp.vec3(*p.tolist()), wp.quat(*q.tolist()))

        bottom_sid = proto.add_shape_cylinder(
            body,
            xform=local_xform((0.0, 0.0, 0.5 * bottom)),
            radius=inner + wall,
            half_height=0.5 * bottom,
            cfg=c,
            color=col,
            label=f"{label_root}/Bottom",
        )
        shape_ids.append(bottom_sid)
        proto.shape_flags[bottom_sid] |= int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.VISIBLE)
        proto.shape_flags[bottom_sid] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)

        segments = max(8, int(cfg.bucket_rigid_wall_segments))
        center_radius = inner + 0.5 * wall
        radial_half = 0.5 * wall
        tangent_half = center_radius * math.tan(math.pi / segments) * 1.08
        wall_center_z = 0.5 * (bottom + height)
        wall_half_z = 0.5 * (height - bottom)
        for i in range(segments):
            theta = 2.0 * math.pi * float(i) / float(segments)
            ct, st = math.cos(theta), math.sin(theta)
            q = np.array((0.0, 0.0, math.sin(0.5 * theta), math.cos(0.5 * theta)), dtype=np.float64)
            sid = proto.add_shape_box(
                body,
                xform=local_xform((center_radius * ct, center_radius * st, wall_center_z), q),
                hx=radial_half,
                hy=tangent_half,
                hz=wall_half_z,
                cfg=c,
                color=col,
                label=f"{label_root}/Wall_{i:02d}",
            )
            shape_ids.append(sid)
            proto.shape_flags[sid] |= int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.VISIBLE)
            proto.shape_flags[sid] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
        return shape_ids

    def _add_bucket(self, proto, cfg, center, label) -> None:
        """Add a stationary cylindrical bucket as both a rigid and MPM collider."""
        cx, cy, _ = center
        base_z = self._bucket_base_z(cfg, center)
        col = (0.50, 0.42, 0.28) if label == "Source" else (0.25, 0.46, 0.55)
        rigid_body = proto.add_body(
            xform=wp.transform(wp.vec3(cx, cy, base_z), wp.quat_identity()),
            mass=0.0,
            is_kinematic=True,
            lock_inertia=True,
            label=f"/World/{label}BucketRigid",
        )
        self._add_open_bucket_rigid_proxy(proto, cfg, rigid_body, col, label)
        proto.body_flags[rigid_body] = int(newton.BodyFlags.KINEMATIC)
        proto.body_mass[rigid_body] = 0.0
        proto.body_inv_mass[rigid_body] = 0.0
        proto.body_inertia[rigid_body] = wp.mat33()
        proto.body_inv_inertia[rigid_body] = wp.mat33()

        mesh = newton.Mesh(self._bucket_vertices, self._bucket_indices, compute_inertia=False, is_solid=False)
        cfg_m = newton.ModelBuilder.ShapeConfig(
            mu=0.8, density=0.0, margin=cfg.collider_margin, has_particle_collision=True
        )
        body = proto.add_body(
            xform=wp.transform(wp.vec3(cx, cy, base_z), wp.quat_identity()),
            mass=0.0,
            is_kinematic=True,
            lock_inertia=True,
            label=f"/World/{label}Bucket",
        )
        sid = proto.add_shape_mesh(
            body,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mesh=mesh,
            cfg=cfg_m,
            color=col,
            label=f"/World/{label}Bucket/Collision",
        )
        proto.shape_flags[sid] |= int(newton.ShapeFlags.COLLIDE_PARTICLES) | int(newton.ShapeFlags.VISIBLE)
        proto.shape_flags[sid] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
        proto.shape_margin[sid] = cfg.collider_margin
        if proto.shape_source[sid] is not None:
            proto.shape_source[sid].indices = proto.shape_source[sid].indices.reshape(-1)
        proto.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
        proto.body_mass[body] = 0.0
        proto.body_inv_mass[body] = 0.0
        proto.body_inertia[body] = wp.mat33()
        proto.body_inv_inertia[body] = wp.mat33()
        if label == "Source":
            self._media_interior = {
                "kind": "bucket",
                "cx": cx,
                "cy": cy,
                "floor_z": base_z + float(cfg.bucket_bottom_thickness),
                "rim_z": base_z + float(cfg.bucket_height),
                "inner_radius": float(cfg.bucket_inner_radius),
            }

    def _add_bowl(self, proto, cfg, center, label) -> None:
        """Add a stationary procedural pour-demo bowl as an MPM collider."""
        cx, cy, cz = center
        base_z = cz - cfg.container_inner_half[2]
        scale = self._container_bowl_scale(cfg)
        col = (0.55, 0.43, 0.30) if label == "Source" else (0.30, 0.45, 0.55)
        mesh = newton.Mesh(
            self._container_bowl_vertices, self._container_bowl_indices, compute_inertia=False, is_solid=False
        )

        rigid_body = proto.add_body(
            xform=wp.transform(wp.vec3(cx, cy, base_z), wp.quat_identity()),
            mass=0.0,
            is_kinematic=True,
            lock_inertia=True,
            label=f"/World/{label}BowlRigid",
        )
        self._add_open_bowl_rigid_proxy(proto, cfg, rigid_body, scale, col, label)
        proto.body_flags[rigid_body] = int(newton.BodyFlags.KINEMATIC)
        proto.body_mass[rigid_body] = 0.0
        proto.body_inv_mass[rigid_body] = 0.0
        proto.body_inertia[rigid_body] = wp.mat33()
        proto.body_inv_inertia[rigid_body] = wp.mat33()

        cfg_m = newton.ModelBuilder.ShapeConfig(
            mu=0.8, density=0.0, margin=cfg.collider_margin, has_particle_collision=True
        )
        body = proto.add_body(
            xform=wp.transform(wp.vec3(cx, cy, base_z), wp.quat_identity()),
            mass=0.0,
            is_kinematic=True,
            lock_inertia=True,
            label=f"/World/{label}Bowl",
        )
        sid = proto.add_shape_mesh(
            body,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mesh=mesh,
            scale=(scale, scale, scale),
            cfg=cfg_m,
            color=col,
            label=f"/World/{label}Bowl/Collision",
        )
        proto.shape_flags[sid] |= int(newton.ShapeFlags.COLLIDE_PARTICLES) | int(newton.ShapeFlags.VISIBLE)
        proto.shape_flags[sid] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
        proto.shape_margin[sid] = cfg.collider_margin
        proto.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
        proto.body_mass[body] = 0.0
        proto.body_inv_mass[body] = 0.0
        proto.body_inertia[body] = wp.mat33()
        proto.body_inv_inertia[body] = wp.mat33()
        if label == "Source":
            inner_bottom = POUR_BOWL_INNER_BOTTOM_RADIUS * scale
            inner_top = POUR_BOWL_INNER_TOP_RADIUS * scale
            floor_z = base_z + POUR_BOWL_BOTTOM_THICKNESS * scale
            rim_z = base_z + POUR_BOWL_HEIGHT * scale
            self._media_interior = {
                "kind": "pour_bowl",
                "cx": cx,
                "cy": cy,
                "floor_z": floor_z,
                "rim_z": rim_z,
                "inner_bottom_r": inner_bottom,
                "inner_top_r": inner_top,
            }

    def _add_filtered_particles(self, proto, cfg, points: np.ndarray, cell: np.ndarray) -> None:
        radius = float(np.max(cell) * 0.45)
        mass = float(np.prod(cell) * cfg.sand_density)
        proto.add_particles(
            pos=points.astype(np.float32, copy=False).tolist(),
            vel=np.zeros_like(points, dtype=np.float32).tolist(),
            mass=[mass] * int(points.shape[0]),
            radius=[radius] * int(points.shape[0]),
            custom_attributes={
                "mpm:friction": cfg.sand_friction,
                "mpm:damping": cfg.sand_damping,
                "mpm:young_modulus": cfg.sand_young_modulus,
                "mpm:yield_pressure": cfg.sand_yield_pressure,
                "mpm:tensile_yield_ratio": cfg.sand_tensile_yield_ratio,
            },
        )

    def _add_pile_media(self, proto, cfg) -> None:
        """Spawn the source media as a natural granular pile (cone at the angle of repose) on the table."""
        import math as _math

        from .pile_sampling import sample_conical_pile

        cx, cy, cz = cfg.source_center
        ihz = float(cfg.container_inner_half[2])
        floor_z = cz - ihz  # box floor == table top (env z=0)
        spacing = float(cfg.voxel_size) / max(float(cfg.particles_per_cell), 1.0)
        angle = _math.atan(max(float(cfg.sand_friction), 0.05))  # angle of repose ~ atan(friction)
        height = float(cfg.pile_height)
        base_radius = height / max(_math.tan(angle), 1.0e-3)
        # Keep the pile base inside the retaining box footprint.
        base_radius = min(base_radius, float(cfg.container_inner_half[0]) - 2.0 * spacing)
        cone_volume = (_math.pi / 3.0) * base_radius * base_radius * height
        count = max(int(cone_volume / (spacing**3)), 64)
        points = sample_conical_pile(
            count,
            (float(cx), float(cy), float(floor_z) + 0.5 * spacing),
            height=height,
            base_radius=base_radius,
            jitter=float(cfg.pile_jitter),
            seed=7,
            device="cpu",
        )
        cell = np.full(3, spacing, dtype=np.float32)
        self._add_filtered_particles(proto, cfg, points, cell)

    def _add_media(self, proto, cfg) -> None:
        if self._container_geometry == "box":
            self._add_pile_media(proto, cfg)
            return
        interior = getattr(self, "_media_interior", None)
        if interior is not None and interior.get("kind") == "bucket":
            clearance = max(float(cfg.voxel_size), 3.0 * float(cfg.collider_margin))
            bottom_clearance = clearance
            top_clearance = clearance
            interior_height = max(float(interior["rim_z"]) - float(interior["floor_z"]), 1.0e-6)
            min_depth = 0.25 * float(cfg.voxel_size)
            fill_top = float(interior["floor_z"]) + interior_height * float(cfg.media_fill_frac)
            fill_top = min(fill_top, float(interior["rim_z"]) - top_clearance)
            radius = max(float(interior["inner_radius"]) - clearance, 0.25 * float(cfg.voxel_size))
            floor = float(interior["floor_z"]) + bottom_clearance
            depth = max(min_depth, fill_top - floor)
            lo = np.array([interior["cx"] - radius, interior["cy"] - radius, floor], dtype=np.float32)
            hi = np.array([interior["cx"] + radius, interior["cy"] + radius, floor + depth], dtype=np.float32)
            center_xy = np.array([interior["cx"], interior["cy"]], dtype=np.float32)
            radius_mode = "constant"
        elif self._container_geometry == "pour_bowl" and interior is not None:
            # Seed like the pour demo: at least one voxel / several margins away from the collider,
            # small jitter, then filter to the bowl's cylinder instead of relying on large grid jitter.
            clearance = max(float(cfg.voxel_size), 3.0 * float(cfg.collider_margin))
            bowl_height = max(float(interior["rim_z"]) - float(interior["floor_z"]), 1.0e-6)
            depth = max(0.0, bowl_height * float(cfg.media_fill_frac))
            depth = min(depth, max(bowl_height - 2.0 * clearance, 0.25 * float(cfg.voxel_size)))
            top_t = np.clip(depth / bowl_height, 0.0, 1.0)
            top_radius = float(interior["inner_bottom_r"]) + top_t * (
                float(interior["inner_top_r"]) - float(interior["inner_bottom_r"])
            )
            radius = max(top_radius - clearance, 0.25 * float(cfg.voxel_size))
            floor = float(interior["floor_z"]) + clearance
            lo = np.array([interior["cx"] - radius, interior["cy"] - radius, floor], dtype=np.float32)
            hi = np.array([interior["cx"] + radius, interior["cy"] + radius, floor + depth], dtype=np.float32)
            center_xy = np.array([interior["cx"], interior["cy"]], dtype=np.float32)
            radius_mode = "frustum"
        else:
            cx, cy, cz = cfg.source_center
            ihx, ihy, ihz = cfg.container_inner_half
            clearance = max(float(cfg.voxel_size), 3.0 * float(cfg.collider_margin), 0.015)
            lo = np.array([cx - ihx + clearance, cy - ihy + clearance, cz - ihz + clearance], dtype=np.float32)
            hi = np.array(
                [cx + ihx - clearance, cy + ihy - clearance, cz - ihz + clearance + 2 * ihz * cfg.media_fill_frac],
                dtype=np.float32,
            )
            center_xy = None
            radius_mode = "box"
        # Particle spacing follows the grid voxel: ``particles_per_cell`` samples per voxel per
        # axis (Newton MPM best practice, matching example_mujoco_mpm_coupled_solver). The bed is
        # thus always resolved consistently with the solver grid, and particle size tracks
        # ``voxel_size`` (coarser voxels -> proportionally larger particles).
        res = np.maximum(np.ceil(cfg.particles_per_cell * (hi - lo) / cfg.voxel_size), 1).astype(int)
        cell = (hi - lo) / res
        px = np.arange(int(res[0]) + 1, dtype=np.float32) * cell[0]
        py = np.arange(int(res[1]) + 1, dtype=np.float32) * cell[1]
        pz = np.arange(int(res[2]) + 1, dtype=np.float32) * cell[2]
        points = np.stack(np.meshgrid(px, py, pz, indexing="ij")).reshape(3, -1).T
        rng = np.random.default_rng(7)
        points += (rng.random(points.shape, dtype=np.float32) - 0.5) * (0.10 * float(np.max(cell)))
        points += lo
        if center_xy is not None and radius_mode == "frustum":
            z_t = np.clip(
                (points[:, 2] - float(interior["floor_z"]))
                / max(float(interior["rim_z"]) - float(interior["floor_z"]), 1.0e-6),
                0.0,
                1.0,
            )
            local_radius = (
                float(interior["inner_bottom_r"])
                + z_t * (float(interior["inner_top_r"]) - float(interior["inner_bottom_r"]))
                - clearance
            )
            local_radius = np.maximum(local_radius, 0.25 * float(cfg.voxel_size))
            normalized_xy = (points[:, :2] - center_xy) / local_radius[:, None]
            points = points[np.sum(normalized_xy * normalized_xy, axis=1) < 1.0]
        elif center_xy is not None and radius_mode == "constant":
            normalized_xy = (points[:, :2] - center_xy) / max(radius, 1.0e-6)
            points = points[np.sum(normalized_xy * normalized_xy, axis=1) < 1.0]
        if points.shape[0] == 0:
            raise RuntimeError("Particle initialization produced no media particles; reduce voxel size or clearance.")
        self._add_filtered_particles(proto, cfg, points, cell)

    # ------------------------------------------------------- post-physics setup
    def _setup_after_physics(self) -> None:
        cfg = self.cfg
        dev = self.device
        model_dev = NewtonManager.get_model().device
        self._robot = self.scene["robot"]
        if not self._robot.is_fixed_base or self._robot.num_base_dofs != 0:
            raise RuntimeError(
                "FrankaScoopEnv expects the standard fixed-base IsaacLab Panda. "
                f"Got is_fixed_base={self._robot.is_fixed_base}, num_base_dofs={self._robot.num_base_dofs}."
            )
        self._arm_joint_ids, self._arm_joint_names = self._robot.find_joints(ARM_JOINTS, preserve_order=True)
        if self._arm_joint_names != ARM_JOINTS:
            raise RuntimeError(f"Unexpected Franka arm joint order: {self._arm_joint_names}; expected {ARM_JOINTS}.")
        self._finger_joint_ids, self._finger_joint_names = self._robot.find_joints(FINGER_JOINTS, preserve_order=True)
        if self._finger_joint_names != FINGER_JOINTS:
            raise RuntimeError(
                f"Unexpected Franka finger joint order: {self._finger_joint_names}; expected {FINGER_JOINTS}."
            )
        hand_body_ids, hand_body_names = self._robot.find_bodies("panda_hand", preserve_order=True)
        if len(hand_body_ids) != 1:
            raise RuntimeError(f"Expected one panda_hand body, found {hand_body_names}.")
        self._hand_body_idx = int(hand_body_ids[0])
        self._hand_jacobi_body_idx = self._hand_body_idx - 1 if self._robot.is_fixed_base else self._hand_body_idx
        self._jacobi_arm_joint_ids = [int(joint_id) + self._robot.num_base_dofs for joint_id in self._arm_joint_ids]
        self.env_origins = self.scene.env_origins.to(device=dev, dtype=torch.float32)
        self._hand_ids = torch.tensor(self._hand_ids_l, device=dev, dtype=torch.long)
        self._scoop_body_ids = torch.tensor(self._scoop_body_ids_l, device=dev, dtype=torch.long)
        self._hand_ids_wp = wp.array(self._hand_ids_l, dtype=wp.int32, device=model_dev)
        self._scoop_body_ids_wp = wp.array(self._scoop_body_ids_l, dtype=wp.int32, device=model_dev)
        self._media_scoop_body_ids_wp = self._resolve_media_scoop_body_ids_wp(model_dev)
        self._weld_pos_wp = wp.vec3(*self._weld_pos.tolist())
        self._weld_rot_wp = wp.quat(*self._weld_rot.tolist())
        self._arm_q_ids = torch.tensor(self._arm_q_l, device=dev, dtype=torch.long)
        self._arm_qd_ids = torch.tensor(self._arm_qd_l, device=dev, dtype=torch.long)
        self._finger_q_ids = torch.tensor(self._finger_q_l, device=dev, dtype=torch.long)
        self._finger_qd_ids = torch.tensor(self._finger_qd_l, device=dev, dtype=torch.long)
        self._finger_q_ids_wp = wp.array(self._finger_q_l, dtype=wp.int32, device=model_dev)
        self._finger_qd_ids_wp = wp.array(self._finger_qd_l, dtype=wp.int32, device=model_dev)
        self._particle_ids = torch.tensor(self._psrc_l, device=dev, dtype=torch.long)
        self._num_particles = int(self._particle_ids.shape[1])

        st = NewtonManager.get_state_0()
        self._default_arm_q = self._robot.data.default_joint_pos.torch[:, self._arm_joint_ids].clone()
        self._fixed_finger_q = torch.full(
            (self.num_envs, len(FINGER_JOINTS)), float(cfg.gripper_open_pos), device=dev, dtype=torch.float32
        )
        self._default_particle_q = wp.to_torch(st.particle_q)[self._particle_ids].clone()

        self._bowl_center_hand_t = torch.tensor(self._bowl_center_hand, device=dev, dtype=torch.float32)
        self._weld_rot_t = torch.tensor(self._weld_rot, device=dev, dtype=torch.float32)
        self._home_quat_t = torch.tensor(cfg.hand_home_quat, device=dev, dtype=torch.float32)
        self._bowl_inner_bottom_r = float(self._ee_bowl_inner_bottom_radius)
        self._bowl_inner_top_r = float(self._ee_bowl_inner_top_radius)
        self._bowl_floor = float(self._ee_bowl_center_local[2] - self._ee_bowl_bottom_thickness)
        self._bowl_lip = float(self._ee_bowl_height - self._ee_bowl_center_local[2])

        self._src_center = torch.tensor(cfg.source_center, device=dev, dtype=torch.float32)
        self._tgt_center = torch.tensor(cfg.target_center, device=dev, dtype=torch.float32)
        self._cont_ih = torch.tensor(cfg.container_inner_half, device=dev, dtype=torch.float32)
        if self._container_geometry == "bucket":
            self._container_inner_bottom_r = float(cfg.bucket_inner_radius)
            self._container_inner_top_r = float(cfg.bucket_inner_radius)
            self._container_bottom_thickness = float(cfg.bucket_bottom_thickness)
            self._container_height = float(cfg.bucket_height)
        else:
            container_scale = self._container_bowl_scale(cfg)
            self._container_inner_bottom_r = float(POUR_BOWL_INNER_BOTTOM_RADIUS * container_scale)
            self._container_inner_top_r = float(POUR_BOWL_INNER_TOP_RADIUS * container_scale)
            self._container_bottom_thickness = float(POUR_BOWL_BOTTOM_THICKNESS * container_scale)
            self._container_height = float(POUR_BOWL_HEIGHT * container_scale)
        self._ws_lo = torch.tensor(cfg.workspace_lo, device=dev, dtype=torch.float32)
        self._ws_hi = torch.tensor(cfg.workspace_hi, device=dev, dtype=torch.float32)

        self._home_bowl_e = torch.tensor(
            np.array(cfg.hand_home_pos) + np.array(cfg.bowl_home_offset), device=dev, dtype=torch.float32
        )
        self._target_bowl_e = self._home_bowl_e.unsqueeze(0).repeat(self.num_envs, 1)
        self._pitch = torch.zeros(self.num_envs, device=dev)

        # heightfield grid bounds (env frame) covering both containers
        self._hf_lo = torch.tensor(cfg.heightfield_lo, device=dev, dtype=torch.float32)
        self._hf_hi = torch.tensor(cfg.heightfield_hi, device=dev, dtype=torch.float32)
        self._hf_n = int(cfg.heightfield_size)
        self._hf_env_off = (torch.arange(self.num_envs, device=dev) * self._hf_n * self._hf_n).unsqueeze(1)

        self._create_ik_solver()
        self._create_diffik_controller()

        # Seed every reset from the hand-tuned ``arm_home`` joint config (a known, unfolded kinematic
        # branch). The DLS ``_solve_ready_config`` can settle into a folded branch, which then seeds the
        # per-reset IK into divergence, so we keep ``arm_home`` (set in _setup_after_physics) as the seed.
        self._reset_arm_q = self._default_arm_q.clone()
        self._pitch[:] = float(cfg.home_pitch)

        self.curriculum_stage = torch.zeros(self.num_envs, device=dev, dtype=torch.long)
        self.episode_succeeded = torch.zeros(self.num_envs, device=dev, dtype=torch.bool)
        self.ep_max_in_target = torch.zeros(self.num_envs, device=dev)
        self.ep_max_in_bowl = torch.zeros(self.num_envs, device=dev)
        # Source-media count captured at reset; the ``removed_from_source`` reward measures how much media
        # has since been scooped out (a fill proxy that does not depend on the moving-cup bowl counter).
        self._init_source_count = torch.full((self.num_envs,), float(self._num_particles), device=dev)
        self.scoop_target_count = float(cfg.curriculum_target_count[0])
        self._region_mask_cache = None
        self._region_mask_cache_step = -1

    def _setup_scoop_bowl_body_sync(self) -> None:
        NewtonManager.register_post_actuator_callback(self._pin_gripper_open_state)
        NewtonManager.register_post_actuator_callback(self._sync_scoop_bowl_body)
        self._pin_gripper_open_states(update_fk=False)
        self._sync_scoop_bowl_body()
        self._sync_scoop_bowl_body(NewtonManager.get_state_1())
        # Newton captures the graph before task managers are loaded. Recapture after
        # task-local callbacks are registered so the gripped MPM collider follows the hand.
        NewtonManager.set_decimation(self.cfg.decimation)

    def _resolve_media_scoop_body_ids_wp(self, device: str) -> wp.array:
        entries = getattr(NewtonManager._solver, "_entries", {})
        media_entry = entries.get(MPM_ENTRY) if isinstance(entries, dict) else None
        if media_entry is None:
            raise RuntimeError(f"Expected coupled solver entry {MPM_ENTRY!r} for scoop-bowl MPM collider sync.")
        body_global_to_local = getattr(media_entry, "body_global_to_local", None)
        if body_global_to_local is None:
            raise RuntimeError(f"Coupled solver entry {MPM_ENTRY!r} does not expose body_global_to_local.")
        local_map = body_global_to_local.numpy()
        local_ids = [int(local_map[int(body_id)]) for body_id in self._scoop_body_ids_l]
        if any(local_id < 0 for local_id in local_ids):
            raise RuntimeError(f"Scoop bowl bodies are not present in coupled solver entry {MPM_ENTRY!r}: {local_ids}")
        self._media_entry_state = media_entry.state_0
        return wp.array(local_ids, dtype=wp.int32, device=device)

    def _pin_gripper_open_state(self, state=None) -> None:
        if state is None:
            state = NewtonManager.get_state_0()
        wp.launch(
            _pin_fixed_joints_kernel,
            dim=(self.num_envs, len(FINGER_JOINTS)),
            inputs=[
                state.joint_q,
                state.joint_qd,
                self._finger_q_ids_wp,
                self._finger_qd_ids_wp,
                float(self.cfg.gripper_open_pos),
            ],
            device=NewtonManager.get_model().device,
        )

    def _pin_gripper_open_states(self, *, update_fk: bool) -> None:
        s0, s1 = NewtonManager.get_state_0(), NewtonManager.get_state_1()
        self._pin_gripper_open_state(s0)
        self._pin_gripper_open_state(s1)
        if update_fk:
            model = NewtonManager.get_model()
            newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0, None)
            newton.eval_fk(model, s1.joint_q, s1.joint_qd, s1, None)

    def _sync_scoop_bowl_body(self, state=None) -> None:
        if state is None:
            state = NewtonManager.get_state_0()
        if state is None or state.body_q is None or state.body_qd is None:
            return
        wp.launch(
            _sync_scoop_bowl_body_kernel,
            dim=self.num_envs,
            inputs=[
                state.body_q,
                state.body_q,
                state.body_qd,
                self._hand_ids_wp,
                self._scoop_body_ids_wp,
                self._weld_pos_wp,
                self._weld_rot_wp,
            ],
            device=state.body_q.device,
        )
        parent_state = NewtonManager.get_state_0()
        media_state = getattr(self, "_media_entry_state", None)
        if state is parent_state and media_state is not None:
            wp.launch(
                _sync_scoop_bowl_body_kernel,
                dim=self.num_envs,
                inputs=[
                    parent_state.body_q,
                    media_state.body_q,
                    media_state.body_qd,
                    self._hand_ids_wp,
                    self._media_scoop_body_ids_wp,
                    self._weld_pos_wp,
                    self._weld_rot_wp,
                ],
                device=parent_state.body_q.device,
            )

    def _setup_kit_visual_sync(self) -> None:
        """Register the per-render Kit callback that drives the gripped cup body and the live MPM media points.

        Only meaningful on the Kit visualizer; a no-op otherwise (the Newton viewer renders Newton state
        directly, without a USD stage to keep in sync).
        """
        if "kit" not in set(self.sim.resolve_visualizer_types()):
            return
        self._kit_visual_callback_name = f"franka_scoop_visuals_{id(self)}"
        NewtonManager.register_pre_render_callback(self._kit_visual_callback_name, self._sync_kit_visuals)
        self._sync_kit_visuals()

    def _sync_kit_visuals(self) -> None:
        """Pre-render callback: keep the cup body state live, then refresh the cup visual + media points.

        ``sync_transforms_to_usd`` does not update the cup's (kinematic MPM-collider) Fabric worldMatrix, so
        the cup visual and the media points are both written directly to USD here every render -- the cup via
        its authored world xform (:meth:`_update_cup_visual_xform`), the media via its ``points`` attribute.
        """
        # Rendering after the solver but before ManagerBasedRLEnv.step() returns can otherwise catch
        # fixed-open fingers after contact impulses and before the post-step correction.
        self._pin_gripper_open_states(update_fk=True)
        self._sync_scoop_bowl_body()
        self._update_cup_visual_xform()
        self._update_media_particles_visual()
        self._update_obs_debug_visual()

    def _update_cup_visual_xform(self) -> None:
        """Rewrite each cup visual prim's world xform from the live cup body pose ``state_0.body_q``."""
        if not self._cup_visual_ops:
            return
        bq = wp.to_torch(NewtonManager.get_state_0().body_q).detach().cpu().numpy()
        with Sdf.ChangeBlock():
            for env_id, (translate_op, orient_op) in enumerate(self._cup_visual_ops):
                pose = bq[int(self._scoop_body_ids_l[env_id])]  # [px,py,pz, qx,qy,qz,qw] (newton transform)
                translate_op.Set(Gf.Vec3d(float(pose[0]), float(pose[1]), float(pose[2])))
                orient_op.Set(Gf.Quatd(float(pose[6]), Gf.Vec3d(float(pose[3]), float(pose[4]), float(pose[5]))))

    def close(self):
        callback_name = getattr(self, "_kit_visual_callback_name", None)
        if callback_name:
            NewtonManager.deregister_pre_render_callback(callback_name)
            self._kit_visual_callback_name = None
        super().close()

    # -------------------------------------------------------------------- IK
    def _create_ik_solver(self) -> None:
        prototype_info = NewtonManager.get_prototype_model(self._robot.cfg.prim_path)
        self._ik_model = prototype_info.model
        if self._ik_model is None:
            raise RuntimeError(f"Newton prototype model for '{self._robot.cfg.prim_path}' was not finalized.")
        ee_matches = [i for i, label in enumerate(self._ik_model.body_label) if str(label).endswith("panda_hand")]
        if len(ee_matches) != 1:
            raise RuntimeError(f"Expected one panda_hand body in the Newton IK prototype, found {ee_matches}.")
        ee = ee_matches[0]
        arm_q = []
        finger_q = []
        joint_labels = [str(label) for label in self._ik_model.joint_label]
        joint_q_start = wp.to_torch(self._ik_model.joint_q_start).detach().cpu()

        def _coord_ids(joint_name: str) -> int:
            matches = [i for i, label in enumerate(joint_labels) if label.endswith(joint_name)]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one {joint_name!r} joint in the Newton IK prototype, found {matches}.")
            return int(joint_q_start[matches[0]].item())

        for joint_name in ARM_JOINTS:
            arm_q.append(_coord_ids(joint_name))
        for joint_name in FINGER_JOINTS:
            finger_q.append(_coord_ids(joint_name))

        dev = self._ik_model.device
        self._ik_default = wp.to_torch(self._ik_model.joint_q).clone()
        self._ik_arm = torch.tensor(arm_q, device=self._ik_default.device, dtype=torch.long)
        self._ik_fingers = torch.tensor(finger_q, device=self._ik_default.device, dtype=torch.long)
        self._ik_default[self._ik_fingers] = float(self.cfg.gripper_open_pos)
        self._ik_target_name = "scoop_bowl"
        self._ik_solver = NewtonIKSolver(
            NewtonIKSolverCfg(
                command_type="pose",
                use_relative_mode=False,
                optimizer="lm",
                jacobian_mode="analytic",
                # Single warm-started seed (the current arm config, passed each solve): runtime tracking must be
                # smooth, so we LM-converge from the previous pose rather than re-sampling seeds. Multi-seed
                # roberts (for escaping folded branches) would let a different branch win each control step ->
                # joint jumps. The active reset seeds from the captured ``arm_home`` directly (not this IK), so
                # the branch-finding seeds are not needed here.
                sampler="none",
                n_seeds=1,
                iterations=self.cfg.ik_iterations,
                step_size=self.cfg.ik_step_size,
                lambda_initial=self.cfg.ik_lambda_initial,
                position_weight=self.cfg.ik_position_weight,
                rotation_weight=self.cfg.ik_rotation_weight,
                joint_limit_weight=self.cfg.ik_joint_limit_weight,
                use_persistent_seed=False,
            ),
            model=self._ik_model,
            num_envs=self.num_envs,
            device=dev,
            pose_objectives=[
                NewtonIKPoseObjective(
                    name=self._ik_target_name,
                    link_index=ee,
                    link_offset_pos=tuple(float(v) for v in self._bowl_center_hand.tolist()),
                )
            ],
        )
        joint_pos_limits = self._robot.data.joint_pos_limits.torch[:, self._arm_joint_ids].clone()
        lo = joint_pos_limits[..., 0]
        hi = joint_pos_limits[..., 1]
        bad = ~(torch.isfinite(lo) & torch.isfinite(hi)) | ((hi - lo) <= 1e-3) | ((hi - lo) > 100.0)
        self._arm_lo = torch.where(bad, torch.full_like(lo, -2 * math.pi), lo)
        self._arm_hi = torch.where(bad, torch.full_like(hi, 2 * math.pi), hi)

    def _create_diffik_controller(self) -> None:
        self._diffik_controller = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                ik_params={"lambda_val": float(self.cfg.diffik_lambda)},
            ),
            num_envs=self.num_envs,
            device=self.device,
        )

    def _solve_ik_full(
        self,
        target_bowl_e: torch.Tensor,
        target_quat: torch.Tensor,
        iterations: int,
        arm_seed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prev_iterations = self._ik_solver.cfg.iterations
        prev_step_size = self._ik_solver.cfg.step_size
        self._ik_solver.cfg.iterations = int(iterations)
        self._ik_solver.cfg.step_size = float(self.cfg.ik_step_size)
        try:
            seed = self._ik_default.unsqueeze(0).expand(self.num_envs, -1).clone()
            seed[:, self._ik_arm] = (self.arm_joint_q() if arm_seed is None else arm_seed).to(seed.device)
            seed[:, self._ik_fingers] = float(self.cfg.gripper_open_pos)
            self._ik_solver.set_target_pose(
                self._ik_target_name,
                target_bowl_e.to(seed.device, dtype=torch.float32),
                target_quat.to(seed.device, dtype=torch.float32),
            )
            result = self._ik_solver.solve(seed)
            result[:, self._ik_fingers] = float(self.cfg.gripper_open_pos)
            return result
        finally:
            self._ik_solver.cfg.iterations = prev_iterations
            self._ik_solver.cfg.step_size = prev_step_size

    def _solve_ready_config(self) -> torch.Tensor:
        """IK-solve a clear 'ready' arm config: bowl at the home hover, opening up (``home_pitch``)."""
        half = 0.5 * float(self.cfg.home_pitch)
        pq = torch.tensor([0.0, math.sin(half), 0.0, math.cos(half)], device=self.device)
        q = _qmul_t(pq.unsqueeze(0).expand(self.num_envs, -1), self._home_quat_t.unsqueeze(0).expand(self.num_envs, -1))
        q = q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-8)
        seed = self._ik_default.unsqueeze(0).expand(self.num_envs, -1).clone()
        seed[:, self._ik_arm] = self._default_arm_q.to(seed.device)
        seed[:, self._ik_fingers] = float(self.cfg.gripper_open_pos)
        self._ik_solver.set_target_pose(
            self._ik_target_name,
            self._home_bowl_e.unsqueeze(0).expand(self.num_envs, -1).to(seed.device, dtype=torch.float32),
            q.to(seed.device, dtype=torch.float32),
        )
        prev_iterations = self._ik_solver.cfg.iterations
        prev_step_size = self._ik_solver.cfg.step_size
        self._ik_solver.cfg.iterations = int(self.cfg.reset_ik_iterations)
        self._ik_solver.cfg.step_size = float(self.cfg.ik_step_size)
        try:
            return self._ik_solver.solve(seed)[:, self._ik_arm].clone().to(self.device)
        finally:
            self._ik_solver.cfg.iterations = prev_iterations
            self._ik_solver.cfg.step_size = prev_step_size

    def _pitch_to_hand_quat(self, pitch: torch.Tensor) -> torch.Tensor:
        """Target HAND world orientation = pitch (about world Y) applied to the home hand orientation.

        Tracking the hand near its home orientation keeps the arm reachable; the gripped cup frame is fixed
        to the hand, so pitch=0 keeps the cup opening up and pitch tilts it to scoop/pour.
        """
        half = 0.5 * pitch
        pq = torch.stack([torch.zeros_like(half), torch.sin(half), torch.zeros_like(half), torch.cos(half)], dim=-1)
        return _qmul_t(pq, self._home_quat_t.unsqueeze(0).expand(self.num_envs, -1))

    def _pitch_target_quat(self) -> torch.Tensor:
        return self._pitch_to_hand_quat(self._pitch)

    def _solve_target_config(
        self,
        target_bowl_e: torch.Tensor,
        pitch: torch.Tensor,
        iterations: int,
        arm_seed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._pitch_to_hand_quat(pitch)
        q = q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-8)
        solved = (
            self._solve_ik_full(target_bowl_e, q, iterations, arm_seed=arm_seed)[:, self._ik_arm]
            .clone()
            .to(self.device)
        )
        return torch.clamp(solved, self._arm_lo, self._arm_hi)

    def solve_arm_ik(self) -> torch.Tensor:
        if str(self.cfg.ik_backend).lower() == "diffik":
            return self._solve_arm_diffik()
        if str(self.cfg.ik_backend).lower() != "newton":
            raise RuntimeError(f"Unsupported ik_backend={self.cfg.ik_backend!r}; expected 'diffik' or 'newton'.")
        cur = self.arm_joint_q()
        q = self._pitch_target_quat()
        q = q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-8)
        solved_full = self._solve_ik_full(self._target_bowl_e, q, self.cfg.ik_iterations)
        solved = torch.nan_to_num(solved_full[:, self._ik_arm].to(cur.device), nan=0.0, posinf=0.0, neginf=0.0)
        solved = torch.clamp(solved, self._arm_lo, self._arm_hi)
        if self.cfg.max_ik_delta > 0.0:
            d = torch.clamp(solved - cur, -self.cfg.max_ik_delta, self.cfg.max_ik_delta)
            return torch.clamp(cur + d, self._arm_lo, self._arm_hi)
        return solved

    def _diffik_frame_pose_root(self) -> tuple[torch.Tensor, torch.Tensor]:
        hand_pos_w = self._robot.data.body_pos_w.torch[:, self._hand_body_idx]
        hand_quat_w = self._robot.data.body_quat_w.torch[:, self._hand_body_idx]
        hand_pos_b, hand_quat_b = math_utils.subtract_frame_transforms(
            self._robot.data.root_pos_w.torch,
            self._robot.data.root_quat_w.torch,
            hand_pos_w,
            hand_quat_w,
        )
        cup_pos_b, _ = math_utils.combine_frame_transforms(
            hand_pos_b,
            hand_quat_b,
            self._bowl_center_hand_t.unsqueeze(0).expand(self.num_envs, -1),
        )
        return cup_pos_b, hand_quat_b

    def _diffik_target_pose_root(
        self, target_bowl_e: torch.Tensor, target_hand_quat_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_pos_w = target_bowl_e + self.env_origins
        target_quat_w = target_hand_quat_w / torch.clamp(
            torch.linalg.norm(target_hand_quat_w, dim=-1, keepdim=True), min=1e-8
        )
        return math_utils.subtract_frame_transforms(
            self._robot.data.root_pos_w.torch,
            self._robot.data.root_quat_w.torch,
            target_pos_w,
            target_quat_w,
        )

    def _diffik_frame_jacobian_root(self, hand_quat_b: torch.Tensor) -> torch.Tensor:
        jacobian = self._robot.data.body_link_jacobian_w.torch[
            :, self._hand_jacobi_body_idx, :, self._jacobi_arm_joint_ids
        ].clone()
        root_rot = math_utils.matrix_from_quat(math_utils.quat_inv(self._robot.data.root_quat_w.torch))
        jacobian[:, :3, :] = torch.bmm(root_rot, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(root_rot, jacobian[:, 3:, :])
        offset_root = math_utils.quat_apply(
            hand_quat_b,
            self._bowl_center_hand_t.unsqueeze(0).expand(self.num_envs, -1),
        )
        jacobian[:, :3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(offset_root), jacobian[:, 3:, :])
        return jacobian

    def _solve_arm_diffik(self) -> torch.Tensor:
        cur = self.arm_joint_q()
        hand_quat_w = self._pitch_target_quat()
        target_pos_b, target_quat_b = self._diffik_target_pose_root(self._target_bowl_e, hand_quat_w)
        command = torch.cat((target_pos_b, target_quat_b), dim=-1)
        ee_pos_b, ee_quat_b = self._diffik_frame_pose_root()
        self._diffik_controller.set_command(command)
        jacobian = self._diffik_frame_jacobian_root(ee_quat_b)
        solved = self._diffik_controller.compute(ee_pos_b, ee_quat_b, jacobian, cur)
        solved = torch.nan_to_num(solved, nan=0.0, posinf=0.0, neginf=0.0)
        solved = torch.clamp(solved, self._arm_lo, self._arm_hi)
        max_delta = float(self.cfg.diffik_max_delta)
        if max_delta > 0.0:
            d = torch.clamp(solved - cur, -max_delta, max_delta)
            solved = torch.clamp(cur + d, self._arm_lo, self._arm_hi)
        return solved

    # ----------------------------------------------------------- state queries
    def _bowl_pose_w(self):
        bq = wp.to_torch(NewtonManager.get_state_0().body_q)[self._hand_ids]
        pos = torch.nan_to_num(bq[:, :3], nan=0.0, posinf=0.0, neginf=0.0)
        quat = bq[:, 3:7]
        # sanitize the hand quaternion -> identity if non-finite/degenerate (keeps rewards/obs finite)
        finite = torch.isfinite(quat).all(dim=-1, keepdim=True)
        norm = torch.linalg.norm(torch.nan_to_num(quat), dim=-1, keepdim=True)
        ident = torch.zeros_like(quat)
        ident[:, 3] = 1.0
        quat = torch.where(finite & (norm > 1e-6), quat / torch.clamp(norm, min=1e-6), ident)
        off = self._bowl_center_hand_t.unsqueeze(0).expand_as(pos)
        bowl_pos = pos + _qrot_t(quat, off)
        bowl_quat = _qmul_t(quat, self._weld_rot_t.unsqueeze(0).expand(self.num_envs, -1))
        return bowl_pos, bowl_quat

    def bowl_pos_e(self) -> torch.Tensor:
        return self._bowl_pose_w()[0] - self.env_origins

    def arm_joint_q(self) -> torch.Tensor:
        return self._robot.data.joint_pos.torch[:, self._arm_joint_ids]

    def arm_joint_qd(self) -> torch.Tensor:
        return self._robot.data.joint_vel.torch[:, self._arm_joint_ids]

    def hold_gripper_open_targets(self, env_ids: torch.Tensor | None = None) -> None:
        """Keep the Panda fingers fixed open; the gripper is not part of the action space."""
        if env_ids is None:
            self._robot.set_joint_position_target_index(
                target=self._fixed_finger_q,
                joint_ids=self._finger_joint_ids,
            )
            return
        self._robot.set_joint_position_target_index(
            target=self._fixed_finger_q[env_ids],
            joint_ids=self._finger_joint_ids,
            env_ids=env_ids,
        )

    def _reset_gripper_open(self, env_ids: torch.Tensor, state_0, state_1) -> None:
        finger_q = self._fixed_finger_q[env_ids]
        finger_qd = torch.zeros_like(finger_q)
        self._robot.write_joint_position_to_sim_index(
            position=finger_q,
            joint_ids=self._finger_joint_ids,
            env_ids=env_ids,
        )
        self._robot.write_joint_velocity_to_sim_index(
            velocity=finger_qd,
            joint_ids=self._finger_joint_ids,
            env_ids=env_ids,
        )
        self.hold_gripper_open_targets(env_ids)
        wp.to_torch(state_0.joint_q)[self._finger_q_ids[env_ids]] = finger_q
        wp.to_torch(state_0.joint_qd)[self._finger_qd_ids[env_ids]] = 0.0
        wp.to_torch(state_1.joint_q)[self._finger_q_ids[env_ids]] = finger_q
        wp.to_torch(state_1.joint_qd)[self._finger_qd_ids[env_ids]] = 0.0

    def particle_pos_e(self) -> torch.Tensor:
        pq = wp.to_torch(NewtonManager.get_state_0().particle_q)[self._particle_ids]
        # sanitize: push any non-finite particle far away so it is excluded from all regions
        pq = torch.nan_to_num(pq, nan=-100.0, posinf=-100.0, neginf=-100.0)
        return pq - self.env_origins[:, None, :]

    def count_in_bowl(self) -> torch.Tensor:
        """Particles inside the (possibly tilted) bowl cup region, per env."""
        return self._region_inside_mask()[..., 0].sum(dim=1).float()

    def _ensure_region_counter(self) -> None:
        """Lazily build the :class:`ParticleMeshCounter` for the bowl + source/target regions.

        The bowl cavity and the (flared) containers are represented as closed region meshes so the
        inside-test is an exact mesh-containment query (Warp winding number) instead of the previous
        analytic frustum approximation. The bowl region tracks the EE pose each step; the container
        regions are static in the env frame.
        """
        if getattr(self, "_region_counter", None) is not None:
            return
        from isaaclab_newton.utils.particle_mesh import (
            ParticleMeshCounter,
            make_box_region_mesh,
            make_frustum_region_mesh,
        )

        dev = self.device
        # bowl cavity: capped frustum in bowl-local frame (+Z = cup opening, origin at cavity center)
        bowl_mesh = make_frustum_region_mesh(
            self._bowl_inner_bottom_r, self._bowl_inner_top_r, -self._bowl_floor, self._bowl_lip, num_segments=48
        )
        src_pos, tgt_pos = self._src_center.clone(), self._tgt_center.clone()
        if self._container_geometry in {"bucket", "pour_bowl"}:
            # Capped bucket/frustum in container-local frame (origin at the container base).
            container_mesh = make_frustum_region_mesh(
                self._container_inner_bottom_r,
                self._container_inner_top_r,
                self._container_bottom_thickness,
                self._container_height,
                num_segments=48,
            )
            if self._container_geometry == "bucket":
                src_pos[2] = self._bucket_base_z(self.cfg, self.cfg.source_center)
                tgt_pos[2] = self._bucket_base_z(self.cfg, self.cfg.target_center)
            else:
                src_pos[2] = self._src_center[2] - self._cont_ih[2]
                tgt_pos[2] = self._tgt_center[2] - self._cont_ih[2]
        else:
            container_mesh = make_box_region_mesh(self._cont_ih.tolist())
        self._region_counter = ParticleMeshCounter(
            [bowl_mesh, container_mesh, container_mesh], num_envs=self.num_envs, device=dev
        )
        # region transforms (env frame): row 0 = bowl (refreshed per step), rows 1/2 = static src/tgt
        self._region_pos_buf = torch.zeros((3, self.num_envs, 3), device=dev)
        self._region_quat_buf = torch.zeros((3, self.num_envs, 4), device=dev)
        self._region_quat_buf[..., 3] = 1.0
        self._region_pos_buf[1, :, :] = src_pos
        self._region_pos_buf[2, :, :] = tgt_pos

    def _region_inside_mask(self) -> torch.Tensor:
        """Per-particle containment in [bowl, source, target], shape ``(num_envs, num_particles, 3)`` bool."""
        step = int(getattr(self, "common_step_counter", -1))
        if self._region_mask_cache is not None and self._region_mask_cache_step == step:
            return self._region_mask_cache
        self._ensure_region_counter()
        pe = self.particle_pos_e()
        _, bquat = self._bowl_pose_w()
        self._region_pos_buf[0] = self.bowl_pos_e()
        self._region_quat_buf[0] = bquat
        _, mask = self._region_counter.count(pe, self._region_pos_buf, self._region_quat_buf, return_mask=True)
        self._region_mask_cache = mask
        self._region_mask_cache_step = step
        return mask

    def _masked_centroid_e(self, mask: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        pe = self.particle_pos_e()
        weights = mask.float()
        cnt = weights.sum(dim=1, keepdim=True)
        cen = (pe * weights.unsqueeze(-1)).sum(dim=1) / torch.clamp(cnt, min=1.0)
        fallback_e = fallback.unsqueeze(0).expand_as(cen) if fallback.dim() == 1 else fallback
        return torch.where(cnt > 0, cen, fallback_e)

    def count_in_source(self) -> torch.Tensor:
        return self._region_inside_mask()[..., 1].sum(dim=1).float()

    def count_in_target(self) -> torch.Tensor:
        return self._region_inside_mask()[..., 2].sum(dim=1).float()

    def source_media_centroid_e(self) -> torch.Tensor:
        return self._masked_centroid_e(self._region_inside_mask()[..., 1], self._src_center)

    def bowl_media_centroid_e(self) -> torch.Tensor:
        return self._masked_centroid_e(self._region_inside_mask()[..., 0], self.bowl_pos_e())

    def all_media_centroid_e(self) -> torch.Tensor:
        pe = self.particle_pos_e()
        return pe.mean(dim=1)

    def heightfield(self) -> torch.Tensor:
        """Top-down max-height grid over both containers (env frame), shape (n, H*W) in [0,1]."""
        pe = self.particle_pos_e()
        n, hsz = self.num_envs, self._hf_n
        rx = (pe[..., 0] - self._hf_lo[0]) / (self._hf_hi[0] - self._hf_lo[0])
        ry = (pe[..., 1] - self._hf_lo[1]) / (self._hf_hi[1] - self._hf_lo[1])
        valid = torch.isfinite(pe).all(dim=-1) & (rx >= 0) & (rx < 1) & (ry >= 0) & (ry < 1)
        px = torch.clamp((rx * hsz).long(), 0, hsz - 1)
        py = torch.clamp((ry * hsz).long(), 0, hsz - 1)
        h = torch.clamp((pe[..., 2] - self._hf_lo[2]) / (self._hf_hi[2] - self._hf_lo[2]), 0.0, 1.0)
        flat = (self._hf_env_off + py * hsz + px).reshape(-1)
        vals = torch.where(valid, h, torch.zeros_like(h)).reshape(-1)
        out = torch.zeros(n * hsz * hsz, device=self.device)
        out.scatter_reduce_(0, flat, vals, reduce="amax", include_self=True)
        return out.reshape(n, hsz * hsz)

    def state_finite(self) -> torch.Tensor:
        st = NewtonManager.get_state_0()
        pq = wp.to_torch(st.particle_q)[self._particle_ids]
        bq = wp.to_torch(st.body_q)[self._hand_ids]
        jq = self.arm_joint_q()
        return torch.isfinite(pq).all(dim=(1, 2)) & torch.isfinite(jq).all(dim=1) & torch.isfinite(bq).all(dim=1)

    @staticmethod
    def _quat_conj(q: torch.Tensor) -> torch.Tensor:
        return torch.cat((-q[..., :3], q[..., 3:4]), dim=-1)

    # ----------------------------------------------------------------- resets
    def _curriculum_start_targets(
        self, env_ids: torch.Tensor, particle_q_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reset_start = str(getattr(self.cfg, "reset_start", "home")).strip().lower()
        if reset_start == "home":
            target = self._home_bowl_e.unsqueeze(0).expand(env_ids.numel(), -1).clone()
            pitches = torch.full((env_ids.numel(),), float(self.cfg.home_pitch), device=self.device)
            return target, pitches
        if reset_start not in {"source_curriculum", "curriculum", "source"}:
            raise ValueError(
                f"Unsupported reset_start={self.cfg.reset_start!r}; expected 'home' or 'source_curriculum'."
            )
        stage = self.curriculum_stage[env_ids].clamp(max=len(self.cfg.curriculum_start_bowl_offset) - 1)
        offsets = torch.tensor(self.cfg.curriculum_start_bowl_offset, device=self.device, dtype=torch.float32)[stage]
        pitches = torch.tensor(self.cfg.curriculum_start_pitch, device=self.device, dtype=torch.float32)[stage]
        particle_centroid_e = particle_q_w.mean(dim=1) - self.env_origins[env_ids]
        target = torch.clamp(particle_centroid_e + offsets, self._ws_lo, self._ws_hi)
        return target, pitches

    def reset_scoop_scene(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        s0, s1 = NewtonManager.get_state_0(), NewtonManager.get_state_1()
        pq0 = wp.to_torch(s0.particle_q)
        new_p = self._sample_media_reset(env_ids)
        pq0[self._particle_ids[env_ids]] = new_p
        wp.to_torch(s0.particle_qd)[self._particle_ids[env_ids]] = 0.0
        wp.to_torch(s1.particle_q)[self._particle_ids[env_ids]] = new_p
        wp.to_torch(s1.particle_qd)[self._particle_ids[env_ids]] = 0.0
        self._reset_mpm_particle_state(env_ids, (s0, s1))
        self._region_mask_cache = None
        self._region_mask_cache_step = -1
        # Staged scoop->dump curriculum: the (global) stage picks the reset pose + how much media is pre-loaded
        # into the cup. "pile" = the scoop start (arm_home, cup tilted at the pile); "target"/"home_up" = cup
        # opening-up over the target / source (IK converges for these upright poses) so the policy starts
        # holding a cupful and only has to dump.
        stage = int(self.curriculum_stage[env_ids][0].clamp(max=len(self.cfg.curriculum_reset_pose) - 1))
        pose_kind = str(self.cfg.curriculum_reset_pose[stage])
        n_cup = int(self.cfg.curriculum_cup_fill_count[stage])
        reset_q, pitch_val = self._reset_arm_config(pose_kind)
        reset_q = torch.where(torch.isfinite(reset_q), reset_q, self._default_arm_q)
        self._pitch[env_ids] = pitch_val
        zero_vel = torch.zeros_like(reset_q[env_ids])
        self._robot.write_joint_position_to_sim_index(
            position=reset_q[env_ids],
            joint_ids=self._arm_joint_ids,
            env_ids=env_ids,
        )
        self._robot.write_joint_velocity_to_sim_index(
            velocity=zero_vel,
            joint_ids=self._arm_joint_ids,
            env_ids=env_ids,
        )
        self._robot.set_joint_position_target_index(
            target=reset_q[env_ids],
            joint_ids=self._arm_joint_ids,
            env_ids=env_ids,
        )
        self._reset_gripper_open(env_ids, s0, s1)
        wp.to_torch(s0.joint_q)[self._arm_q_ids[env_ids]] = reset_q[env_ids]
        wp.to_torch(s0.joint_qd)[self._arm_qd_ids[env_ids]] = 0.0
        wp.to_torch(s1.joint_q)[self._arm_q_ids[env_ids]] = reset_q[env_ids]
        wp.to_torch(s1.joint_qd)[self._arm_qd_ids[env_ids]] = 0.0
        newton.eval_fk(NewtonManager.get_model(), s0.joint_q, s0.joint_qd, s0, None)
        newton.eval_fk(NewtonManager.get_model(), s1.joint_q, s1.joint_qd, s1, None)
        self._sync_scoop_bowl_body(s0)
        self._sync_scoop_bowl_body(s1)
        if n_cup > 0:  # pre-load a cupful into the (now opening-up) cavity for the early dump stages
            self._load_cup_media(env_ids, n_cup, s0, s1)
        # Hold the achieved reset pose. This avoids DiffIK chasing residual Newton-IK/collision error on
        # the first zero-action teleop frame.
        self._target_bowl_e[env_ids] = self.bowl_pos_e()[env_ids].detach()
        self.episode_succeeded[env_ids] = False
        self.ep_max_in_target[env_ids] = 0.0
        self.ep_max_in_bowl[env_ids] = 0.0
        self._reset_arm_q[env_ids] = reset_q[env_ids]
        # Capture how much media sits in the source right after re-piling (baseline for ``removed_from_source``).
        self._region_mask_cache = None
        self._init_source_count[env_ids] = self.count_in_source()[env_ids].detach()

    def _reset_mpm_particle_state(self, env_ids: torch.Tensor, states) -> None:
        """Reset the reset-envs' per-particle implicit-MPM state to its rest values.

        For each reset env's particles, restore the solver's registered defaults: elastic deformation gradient
        and particle frame -> identity (a deformation gradient of I means undeformed -- NOT literally zero),
        plastic ``Jp`` -> 1, APIC velocity gradient and stress -> 0. This guarantees a deterministic, stress-free
        fresh pile. For the current granular rheology it is effectively a no-op (plastic flow keeps the elastic
        strain at identity each step), but it is correct hygiene and matters if the material is made elastic.
        Gated by ``cfg.reset_mpm_particle_state``.
        """
        if not getattr(self.cfg, "reset_mpm_particle_state", True):
            return
        ids = self._particle_ids[env_ids]
        eye = torch.eye(3, device=self.device)
        rest_by_field = (
            ("particle_elastic_strain", eye),
            ("particle_transform", eye),
            ("particle_qd_grad", 0.0),
            ("particle_stress", 0.0),
            ("particle_Jp", 1.0),
        )
        for state in states:
            mpm = getattr(state, "mpm", None)
            if mpm is None:
                continue
            for name, rest in rest_by_field:
                arr = getattr(mpm, name, None)
                if arr is None:
                    continue
                wp.to_torch(arr)[ids] = rest

    def _reset_arm_config(self, pose_kind: str) -> tuple[torch.Tensor, float]:
        """Arm joint targets (shape ``[num_envs, n_arm]``) + cup pitch state for a curriculum reset pose.

        ``"pile"`` is the fixed scoop start (``arm_home``, cup tilted at the pile). ``"target"``/``"home_up"``
        solve the cup OPENING-UP over the target / source via IK -- the single-seed NewtonIK converges for
        these upright targets (verified), unlike the folded scoop poses -- so the cup can hold a pre-loaded
        cupful. Returns the scoop tilt for ``"pile"`` and ``0`` (opening up) otherwise.
        """
        if pose_kind == "pile":
            return self._default_arm_q, float(self.cfg.curriculum_start_pitch[0])
        # "home_up" hovers at a central spot clear of the pile (no reset pop) + reachable; "target" is over the
        # +y box (railed -- avoid).
        xy = self.cfg.loaded_hover_xy if pose_kind == "home_up" else self.cfg.target_center
        target = torch.tensor([float(xy[0]), float(xy[1]), float(self.cfg.dump_hover_z)], device=self.device)
        target = target.unsqueeze(0).expand(self.num_envs, -1)
        pitch = torch.zeros(self.num_envs, device=self.device)
        q = self._solve_target_config(target, pitch, int(self.cfg.reset_ik_iterations), arm_seed=self._default_arm_q)
        return q, 0.0

    def _load_cup_media(self, env_ids: torch.Tensor, n_cup: int, s0, s1) -> None:
        """Pre-load the first ``n_cup`` of each reset env's particles into the (opening-up) cup cavity.

        Spread across the cavity (not a tight blob) so the pre-loaded media is not overlapping -> stable.
        """
        bowl_w, _ = self._bowl_pose_w()  # world cavity centre at the just-set reset pose
        centers = bowl_w[env_ids].unsqueeze(1)  # (n, 1, 3)
        n = env_ids.numel()
        r_in = float(self._ee_bowl_inner_top_radius)
        off = torch.empty(n, n_cup, 3, device=self.device)
        off[..., :2] = (torch.rand(n, n_cup, 2, device=self.device) - 0.5) * 2.0 * (0.7 * r_in)
        off[..., 2] = (torch.rand(n, n_cup, device=self.device) - 0.7) * (0.6 * r_in)  # toward the cavity floor
        cup_pos = (centers + off).to(torch.float32)
        ids = self._particle_ids[env_ids][:, :n_cup]  # (n, n_cup) global particle indices
        for st in (s0, s1):
            wp.to_torch(st.particle_q)[ids] = cup_pos
            wp.to_torch(st.particle_qd)[ids] = 0.0

    def _sample_media_reset(self, env_ids: torch.Tensor) -> torch.Tensor:
        n = env_ids.numel()
        base = self._default_particle_q[env_ids].clone()  # world-frame pile in the source container
        stage = self.curriculum_stage[env_ids].clamp(max=len(self.cfg.curriculum_pile_xy_jitter) - 1)
        jit = torch.tensor(self.cfg.curriculum_pile_xy_jitter, device=self.device)[stage]
        off = torch.zeros(n, 3, device=self.device)
        off[:, :2] = (torch.rand(n, 2, device=self.device) - 0.5) * 2.0 * jit.unsqueeze(-1)
        return base + off.unsqueeze(1)
