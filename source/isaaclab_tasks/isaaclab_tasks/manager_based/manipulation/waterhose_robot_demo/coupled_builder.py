# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local Newton model construction for the coupled waterhose task."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton import GeoType

from isaaclab_newton.physics import NewtonManager

from .cable_curve_import import CableCurveImportResult, add_cable_from_usd_curve


VBD_KE = 1.0e3
VBD_KD = 0.0
GRIPPER_FINGER_BODY_NAMES = (
    "right_gripper_leftfinger",
    "right_gripper_rightfinger",
    "left_gripper_leftfinger",
    "left_gripper_rightfinger",
)

GRIPPER_DRIVER_DOFS = (13, 23)
GRIPPER_FINGER_DOFS = (14, 15, 24, 25)
LEROBOT_INITIAL_STATE_22 = (
    0.0,
    0.872664213180542,
    -1.5707811117172241,
    0.6981245279312134,
    3.796982127823867e-06,
    0.0,
    0.3021828234195709,
    -0.013802030123770237,
    -0.09509921818971634,
    -2.2242417335510254,
    -0.7117632627487183,
    0.14113007485866547,
    0.5137608647346497,
    -0.4555884897708893,
    0.2500312626361847,
    -0.665743887424469,
    -1.3314952850341797,
    -0.19328542053699493,
    -0.5307496786117554,
    0.6565361022949219,
    0.0913801970053464174,
    0.09098683297634125,
)


@dataclass
class WaterhoseCoupledBuildInfo:
    """Bookkeeping from task-local model construction."""

    num_envs: int = 1
    env_body_count: int = 0
    env_shape_count: int = 0
    env_joint_q_count: int = 0
    env_origins: np.ndarray | None = None
    robot_body_count: int = 0
    robot_shape_count: int = 0
    robot_joint_q_count: int = 0
    vbd_body_count: int = 0
    vbd_shape_count: int = 0
    vbd_initial_body_q: np.ndarray | None = None
    vbd_initial_body_q_by_env: np.ndarray | None = None
    cable_body_labels: list[str] = field(default_factory=list)
    plug_body_labels: list[str] = field(default_factory=list)
    scene_shape_ids: list[int] = field(default_factory=list)
    robot_proxy_body_labels: list[str] = field(default_factory=list)
    vbd_proxy_body_labels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WaterhoseAssetPaths:
    """Resolved asset paths used by the waterhose Newton builder."""

    root: Path
    robot_urdf: Path
    scene_usd: Path
    cable_usd: Path

    @classmethod
    def from_root(cls, asset_root: str | Path) -> "WaterhoseAssetPaths":
        root = Path(asset_root).expanduser().resolve()
        paths = cls(
            root=root,
            robot_urdf=root / "RBY1DF" / "urdf" / "robot_edited.urdf",
            scene_usd=root / "Waterhose" / "Cable008" / "Cable008_Body.usda",
            cable_usd=root / "Waterhose" / "Cable008" / "curve" / "cable_SRA_curve03.usda",
        )
        paths.validate()
        return paths

    def validate(self) -> None:
        for path in (self.robot_urdf, self.scene_usd, self.cable_usd):
            if not path.is_file():
                raise FileNotFoundError(f"Waterhose asset not found: {path}")


def build_waterhose_coupled_builder(
    asset_root: str | Path,
    *,
    include_proxy_bodies: bool = False,
    num_envs: int = 1,
    env_spacing: float = 2.5,
) -> tuple[newton.ModelBuilder, WaterhoseCoupledBuildInfo]:
    """Build a single Newton model containing the robot and VBD waterhose scene.

    The model is owned and stepped by the standard
    :class:`isaaclab_newton.physics.NewtonCoupledManager` path.
    """

    paths = WaterhoseAssetPaths.from_root(asset_root)
    shape_cfg = _create_collision_shape_config()
    fridge_xform = _compute_fridge_xform()

    robot_builder = _build_robot(paths.robot_urdf, shape_cfg)
    vbd_builder, cable_results, scene_shape_ids = _build_vbd_side(paths, fridge_xform)
    robot_proxy_body_labels: list[str] = []
    vbd_proxy_body_labels: list[str] = []
    if include_proxy_bodies:
        robot_proxy_body_labels, vbd_proxy_body_labels = _create_vbd_proxy_bodies(
            robot_builder,
            vbd_builder,
            scene_shape_ids=scene_shape_ids,
        )

    builder = NewtonManager.create_builder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.default_shape_cfg = shape_cfg
    builder.add_builder(robot_builder, label_prefix="mujoco")
    robot_body_count = int(robot_builder.body_count)
    robot_shape_count = int(robot_builder.shape_count)
    builder.add_builder(vbd_builder, label_prefix="vbd")
    _filter_robot_vbd_cross_contacts(
        builder=builder,
        robot_builder=robot_builder,
        vbd_builder=vbd_builder,
        robot_shape_count=robot_shape_count,
        cable_results=cable_results,
    )
    _sanitize_builder_labels(builder)
    builder.color()

    info = WaterhoseCoupledBuildInfo(
        num_envs=1,
        env_body_count=int(builder.body_count),
        env_shape_count=int(builder.shape_count),
        env_joint_q_count=int(builder.joint_coord_count),
        env_origins=np.zeros((1, 3), dtype=np.float32),
        robot_body_count=robot_body_count,
        robot_shape_count=robot_shape_count,
        robot_joint_q_count=int(robot_builder.joint_coord_count),
        vbd_body_count=int(vbd_builder.body_count),
        vbd_shape_count=int(vbd_builder.shape_count),
        vbd_initial_body_q=_builder_body_q_array(vbd_builder),
        vbd_initial_body_q_by_env=_builder_body_q_array(vbd_builder).reshape(1, int(vbd_builder.body_count), 7),
        cable_body_labels=[
            _sanitize_label(f"vbd/{_body_label(vbd_builder, body_id)}")
            for result in cable_results
            for body_id in result.cable_body_ids
        ],
        plug_body_labels=[
            _sanitize_label(f"vbd/{_body_label(vbd_builder, body_id)}")
            for result in cable_results
            for body_id in result.head_body_ids
        ],
        scene_shape_ids=[robot_shape_count + int(shape_id) for shape_id in scene_shape_ids],
        robot_proxy_body_labels=[_sanitize_label(f"mujoco/{label}") for label in robot_proxy_body_labels],
        vbd_proxy_body_labels=[_sanitize_label(f"vbd/{label}") for label in vbd_proxy_body_labels],
    )
    num_envs = max(1, int(num_envs))
    if num_envs > 1:
        builder, info = _replicate_waterhose_builder(builder, info, num_envs=num_envs, env_spacing=env_spacing)
    return builder, info


def _replicate_waterhose_builder(
    prototype: newton.ModelBuilder,
    info: WaterhoseCoupledBuildInfo,
    *,
    num_envs: int,
    env_spacing: float,
) -> tuple[newton.ModelBuilder, WaterhoseCoupledBuildInfo]:
    """Replicate the task prototype into one Newton multi-world builder."""

    origins = _grid_origins(num_envs, env_spacing)
    replicated = NewtonManager.create_builder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(replicated)
    replicated.default_shape_cfg = prototype.default_shape_cfg

    for env_id, origin in enumerate(origins):
        replicated.begin_world(label=f"env_{env_id}")
        replicated.add_builder(
            prototype,
            xform=wp.transform(
                wp.vec3(float(origin[0]), float(origin[1]), float(origin[2])),
                wp.quat_identity(),
            ),
            label_prefix=f"env_{env_id}",
        )
        replicated.end_world()

    replicated.color()
    vbd_initial_body_q = np.asarray(info.vbd_initial_body_q, dtype=np.float32)
    vbd_initial_body_q_by_env = np.repeat(vbd_initial_body_q.reshape(1, *vbd_initial_body_q.shape), num_envs, axis=0)
    vbd_initial_body_q_by_env[:, :, :3] += origins.reshape(num_envs, 1, 3)

    env_body_count = int(info.env_body_count or prototype.body_count)
    env_shape_count = int(info.env_shape_count or prototype.shape_count)
    return replicated, WaterhoseCoupledBuildInfo(
        num_envs=num_envs,
        env_body_count=env_body_count,
        env_shape_count=env_shape_count,
        env_joint_q_count=int(info.env_joint_q_count or prototype.joint_coord_count),
        env_origins=origins,
        robot_body_count=int(info.robot_body_count),
        robot_shape_count=int(info.robot_shape_count),
        robot_joint_q_count=int(info.robot_joint_q_count),
        vbd_body_count=int(info.vbd_body_count),
        vbd_shape_count=int(info.vbd_shape_count),
        vbd_initial_body_q=vbd_initial_body_q,
        vbd_initial_body_q_by_env=vbd_initial_body_q_by_env,
        cable_body_labels=_prefixed_env_labels(info.cable_body_labels, num_envs),
        plug_body_labels=_prefixed_env_labels(info.plug_body_labels, num_envs),
        scene_shape_ids=[
            env_id * env_shape_count + int(shape_id)
            for env_id in range(num_envs)
            for shape_id in info.scene_shape_ids
        ],
        robot_proxy_body_labels=_prefixed_env_labels(info.robot_proxy_body_labels, num_envs),
        vbd_proxy_body_labels=_prefixed_env_labels(info.vbd_proxy_body_labels, num_envs),
    )


def _grid_origins(num_envs: int, env_spacing: float) -> np.ndarray:
    """Return IsaacLab-style grid origins without requiring a live scene."""

    try:
        from isaaclab.cloner.cloner_utils import grid_transforms  # noqa: PLC0415

        origins, _ = grid_transforms(int(num_envs), float(env_spacing), device="cpu")
        return origins.cpu().numpy().astype(np.float32, copy=False)
    except Exception:
        cols = int(np.ceil(np.sqrt(num_envs)))
        origins = np.zeros((num_envs, 3), dtype=np.float32)
        for env_id in range(num_envs):
            origins[env_id, 0] = (env_id % cols) * float(env_spacing)
            origins[env_id, 1] = (env_id // cols) * float(env_spacing)
        return origins


def _prefixed_env_labels(labels: list[str], num_envs: int) -> list[str]:
    return [f"env_{env_id}/{label}" for env_id in range(num_envs) for label in labels]


def build_waterhose_robot_model(asset_root: str | Path, device: str | wp.context.Device):
    """Build the robot-only model used by Newton IK.

    The coupled simulation model prefixes labels and mixes robot/VBD bodies.
    Keeping IK on a robot-only model mirrors the reference demo and avoids
    controller assumptions leaking into the coupled solver model.
    """

    paths = WaterhoseAssetPaths.from_root(asset_root)
    robot_builder = _build_robot(paths.robot_urdf, _create_collision_shape_config())
    return robot_builder.finalize(device=wp.get_device(str(device)))


def _create_collision_shape_config() -> newton.ModelBuilder.ShapeConfig:
    shape_cfg = newton.ModelBuilder.ShapeConfig(
        margin=0.0,
        gap=0.002,
        ke=5.0e4,
        kd=5.0e2,
        mu=2.0,
        mu_torsional=0.01,
        mu_rolling=0.0,
    )
    shape_cfg.is_hydroelastic = False
    return shape_cfg


def _compute_fridge_xform() -> wp.transform:
    table_half_z = 0.5 * (0.6 - 0.215)
    table_top_z = 2.0 * table_half_z
    fridge_z_offset = 0.902 + table_top_z
    fridge_y_offset = (0.293 - 0.395) / 2.0
    fridge_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi / 2.0)
    return wp.transform(wp.vec3(0.95, fridge_y_offset, fridge_z_offset), fridge_rot)


def _build_robot(robot_urdf_path: Path, shape_cfg: newton.ModelBuilder.ShapeConfig) -> newton.ModelBuilder:
    robot = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
    robot.default_shape_cfg = shape_cfg

    robot.add_urdf(
        str(robot_urdf_path),
        floating=False,
        enable_self_collisions=False,
        parse_visuals_as_colliders=False,
        ignore_inertial_definitions=True,
    )

    for body_id, label in enumerate(robot.body_label):
        if str(label).endswith("gripper_dummy") and robot.body_mass[body_id] == 0.0:
            robot.body_mass[body_id] = 1.0e-6
            robot.body_inv_mass[body_id] = 1.0e6
            robot.body_inertia[body_id] = wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-10)
            robot.body_inv_inertia[body_id] = wp.inverse(robot.body_inertia[body_id])

    for dof in range(robot.joint_dof_count):
        if dof in GRIPPER_DRIVER_DOFS or dof in GRIPPER_FINGER_DOFS:
            continue
        robot.joint_target_ke[dof] = 120000.0
        robot.joint_target_kd[dof] = 12000.0
        robot.joint_effort_limit[dof] = 10000.0
        robot.joint_armature[dof] = 0.2

    for dof in GRIPPER_DRIVER_DOFS:
        robot.joint_target_ke[dof] = 10000.0
        robot.joint_target_kd[dof] = 1000.0
        robot.joint_effort_limit[dof] = 100000.0
        robot.joint_armature[dof] = 0.5

    for dof in GRIPPER_FINGER_DOFS:
        robot.joint_target_ke[dof] = 500000.0
        robot.joint_target_kd[dof] = 10000.0
        robot.joint_effort_limit[dof] = 500000.0
        robot.joint_armature[dof] = 0.5

    robot.joint_q = _initial_robot_joint_positions()
    _configure_mujoco_gravity_compensation(robot)
    return robot


def _build_vbd_side(
    paths: WaterhoseAssetPaths, fridge_xform: wp.transform
) -> tuple[newton.ModelBuilder, list[CableCurveImportResult], list[int]]:
    builder = _create_vbd_builder()
    scene_shape_ids = _load_static_scene(builder, paths.scene_usd, fridge_xform)

    cable_results = _add_waterhose_cables(paths.cable_usd, builder)
    _filter_cable_self_collisions(builder, cable_results)
    _zero_fixed_cable_bodies(builder, cable_results)

    cable_joint_ids = [
        joint_id
        for result in cable_results
        for joint_id in [*result.cable_joint_ids, *result.head_fixed_joint_ids]
    ]
    if cable_joint_ids:
        builder.add_articulation(cable_joint_ids, label="water_hose_cable_articulation")

    _transform_builder_bodies(
        builder,
        [body_id for result in cable_results for body_id in [*result.cable_body_ids, *result.head_body_ids]],
        fridge_xform,
    )
    _scale_plug_mesh_shapes(builder, cable_results, xy_scale=0.95, mu=10.0)
    return builder, cable_results, scene_shape_ids


def _create_vbd_builder() -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    builder.rigid_contact_margin = 0.0
    builder.rigid_gap = 0.001
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.ke = VBD_KE
    builder.default_shape_cfg.kd = VBD_KD
    builder.default_shape_cfg.mu = 0.2
    return builder


def _load_static_scene(builder: newton.ModelBuilder, scene_usd_path: Path, fridge_xform: wp.transform) -> list[int]:
    scene_result = builder.add_usd(
        str(scene_usd_path),
        xform=fridge_xform,
        root_path="/root",
        load_sites=False,
        load_visual_shapes=True,
        hide_collision_shapes=False,
        parse_mujoco_options=False,
        only_load_enabled_joints=True,
        only_load_enabled_rigid_bodies=False,
    )

    scene_body_ids = sorted({int(body_id) for body_id in scene_result["path_body_map"].values()})
    for body_id in scene_body_ids:
        builder.body_mass[body_id] = 0.0
        builder.body_inv_mass[body_id] = 0.0
        builder.body_inertia[body_id] = wp.mat33()
        builder.body_inv_inertia[body_id] = wp.mat33()

    scene_shape_ids = sorted(int(shape_id) for shape_id in scene_result["path_shape_map"].values())
    for left_index, left_shape in enumerate(scene_shape_ids):
        for right_shape in scene_shape_ids[left_index + 1 :]:
            builder.add_shape_collision_filter_pair(left_shape, right_shape)

    return scene_shape_ids


def _add_waterhose_cables(cable_usd_path: Path, builder: newton.ModelBuilder) -> list[CableCurveImportResult]:
    light_head_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, ke=VBD_KE, kd=VBD_KD, mu=10.0)
    cable_shape_cfg = builder.default_shape_cfg.copy()
    cable_shape_cfg.ke = VBD_KE
    cable_shape_cfg.kd = VBD_KD
    cable_shape_cfg.mu = 0.2

    results: list[CableCurveImportResult] = []
    for index, curve_prim_path in enumerate(("/World/cable001/curve_0", "/World/cable002/curve_0")):
        results.append(
            add_cable_from_usd_curve(
                builder=builder,
                source_usd_path=str(cable_usd_path),
                curve_prim_path=curve_prim_path,
                cable_label=f"water_hose_cable_{index}",
                cable_cfg=cable_shape_cfg,
                stretch_stiffness=1.0e6,
                stretch_damping=1.0e-2,
                bend_stiffness=2.0e1,
                bend_damping=1.0,
                wrap_in_articulation=False,
                head_shape_mode="mesh",
                head_cfg=light_head_cfg,
                head_mass=0.0,
            )
        )

    for result in results:
        if not result.head_body_ids or len(result.cable_body_ids) < 2:
            continue
        neighbor_idx = 1 if result is results[0] else -2
        neighbor_body = result.cable_body_ids[neighbor_idx]
        for head_body in result.head_body_ids:
            for head_shape in builder.body_shapes.get(head_body, []):
                for neighbor_shape in builder.body_shapes.get(neighbor_body, []):
                    builder.add_shape_collision_filter_pair(head_shape, neighbor_shape)
    return results


def _create_vbd_proxy_bodies(
    robot_builder: newton.ModelBuilder,
    vbd_builder: newton.ModelBuilder,
    *,
    scene_shape_ids: list[int],
) -> tuple[list[str], list[str]]:
    """Duplicate gripper finger bodies into the VBD scene as one-way proxies."""

    proxy_shape_cfg = vbd_builder.default_shape_cfg.copy()
    proxy_shape_cfg.ke = VBD_KE
    proxy_shape_cfg.kd = VBD_KD
    proxy_shape_cfg.mu = 1.0e6
    proxy_shape_cfg.margin = 0.001

    robot_labels: list[str] = []
    proxy_labels: list[str] = []
    proxy_shapes_by_name: dict[str, list[int]] = {}
    for body_id, label in enumerate(robot_builder.body_label):
        body_label = str(label)
        body_name = body_label.rsplit("/", 1)[-1]
        if body_name not in GRIPPER_FINGER_BODY_NAMES:
            continue

        mass = float(robot_builder.body_mass[body_id])
        if mass <= 0.0:
            continue
        proxy_label = f"proxy_{body_name}"
        proxy_body_id = vbd_builder.add_body(
            xform=robot_builder.body_q[body_id],
            mass=mass,
            inertia=robot_builder.body_inertia[body_id],
            lock_inertia=True,
            label=proxy_label,
        )
        proxy_shape_ids = _copy_body_shapes(
            src_builder=robot_builder,
            dst_builder=vbd_builder,
            src_body_id=body_id,
            dst_body_id=proxy_body_id,
            cfg=proxy_shape_cfg,
        )
        if not proxy_shape_ids:
            proxy_shape_ids.append(
                int(
                    vbd_builder.add_shape_box(
                        body=proxy_body_id,
                        hx=0.02,
                        hy=0.01,
                        hz=0.04,
                        cfg=proxy_shape_cfg,
                    )
                )
            )

        robot_labels.append(body_label)
        proxy_labels.append(proxy_label)
        proxy_shapes_by_name[body_name] = proxy_shape_ids

    if not robot_labels:
        raise RuntimeError("No RBY1 gripper finger bodies found for one-way VBD proxy coupling.")

    for gripper_prefix in ("right_gripper", "left_gripper"):
        left = proxy_shapes_by_name.get(f"{gripper_prefix}_leftfinger", [])
        right = proxy_shapes_by_name.get(f"{gripper_prefix}_rightfinger", [])
        for left_shape in left:
            for right_shape in right:
                vbd_builder.add_shape_collision_filter_pair(int(left_shape), int(right_shape))

    for proxy_shape_ids in proxy_shapes_by_name.values():
        for proxy_shape_id in proxy_shape_ids:
            for scene_shape_id in scene_shape_ids:
                vbd_builder.add_shape_collision_filter_pair(int(proxy_shape_id), int(scene_shape_id))

    return robot_labels, proxy_labels


def _copy_body_shapes(
    *,
    src_builder: newton.ModelBuilder,
    dst_builder: newton.ModelBuilder,
    src_body_id: int,
    dst_body_id: int,
    cfg: newton.ModelBuilder.ShapeConfig,
) -> list[int]:
    """Copy all shapes attached to one builder body to another builder body."""

    shape_ids: list[int] = []
    for shape_id in src_builder.body_shapes.get(src_body_id, []):
        shape_type = int(src_builder.shape_type[shape_id])
        scale = src_builder.shape_scale[shape_id]
        xform = src_builder.shape_transform[shape_id]
        pos = wp.transform_get_translation(xform)
        rot = wp.transform_get_rotation(xform)

        if shape_type == int(GeoType.SPHERE):
            copied_id = dst_builder.add_shape_sphere(
                body=dst_body_id,
                radius=float(scale[0]),
                pos=pos,
                rot=rot,
                cfg=cfg,
            )
        elif shape_type == int(GeoType.BOX):
            copied_id = dst_builder.add_shape_box(
                body=dst_body_id,
                hx=float(scale[0]),
                hy=float(scale[1]),
                hz=float(scale[2]),
                pos=pos,
                rot=rot,
                cfg=cfg,
            )
        elif shape_type == int(GeoType.CAPSULE):
            copied_id = dst_builder.add_shape_capsule(
                body=dst_body_id,
                radius=float(scale[0]),
                half_height=float(scale[1]),
                pos=pos,
                rot=rot,
                cfg=cfg,
            )
        elif shape_type == int(GeoType.MESH):
            copied_id = dst_builder.add_shape_mesh(
                body=dst_body_id,
                mesh=src_builder.shape_source[shape_id],
                xform=xform,
                cfg=cfg,
            )
        else:
            continue
        shape_ids.append(int(copied_id))
    return shape_ids


def _configure_mujoco_gravity_compensation(robot: newton.ModelBuilder) -> None:
    gravcomp_body = robot.custom_attributes["mujoco:gravcomp"]
    if gravcomp_body.values is None:
        gravcomp_body.values = {}
    for body_idx in range(1, robot.body_count):
        gravcomp_body.values[body_idx] = 1.0

    gravcomp_joint = robot.custom_attributes["mujoco:jnt_actgravcomp"]
    if gravcomp_joint.values is None:
        gravcomp_joint.values = {}
    for dof_idx in range(robot.joint_dof_count):
        if dof_idx not in GRIPPER_DRIVER_DOFS and dof_idx not in GRIPPER_FINGER_DOFS:
            gravcomp_joint.values[dof_idx] = True


def _initial_robot_joint_positions() -> list[float]:
    lr = LEROBOT_INITIAL_STATE_22
    q = [0.0] * 28
    q[0:6] = lr[0:6]
    q[6:13] = lr[6:13]
    q[12] += np.pi / 2.0
    q[13] = lr[20]
    q[14] = -lr[20] / 2.0
    q[15] = lr[20] / 2.0
    q[16:23] = lr[13:20]
    q[22] -= np.pi / 2.0
    q[23] = lr[21]
    q[24] = -lr[21] / 2.0
    q[25] = lr[21] / 2.0
    return q


def _filter_cable_self_collisions(builder: newton.ModelBuilder, cable_results: list[CableCurveImportResult]) -> None:
    for result in cable_results:
        shape_ids: list[int] = []
        for body_id in [*result.cable_body_ids, *result.head_body_ids]:
            shape_ids.extend(builder.body_shapes.get(body_id, []))
        for left_index, left_shape in enumerate(shape_ids):
            for right_shape in shape_ids[left_index + 1 :]:
                builder.add_shape_collision_filter_pair(left_shape, right_shape)


def _zero_fixed_cable_bodies(builder: newton.ModelBuilder, cable_results: list[CableCurveImportResult]) -> None:
    seen: set[int] = set()
    for body_id in [body_id for result in cable_results for body_id in result.fixed_body_ids]:
        if body_id in seen:
            continue
        seen.add(body_id)
        builder.body_mass[body_id] = 0.0
        builder.body_inv_mass[body_id] = 0.0
        builder.body_inertia[body_id] = wp.mat33()
        builder.body_inv_inertia[body_id] = wp.mat33()


def _transform_builder_bodies(builder: newton.ModelBuilder, body_ids: list[int], xform: wp.transform) -> None:
    rot = wp.transform_get_rotation(xform)
    pos = wp.transform_get_translation(xform)
    for body_id in body_ids:
        old = builder.body_q[body_id]
        old_pos = wp.transform_get_translation(old)
        old_rot = wp.transform_get_rotation(old)
        builder.body_q[body_id] = wp.transform(wp.quat_rotate(rot, old_pos) + pos, wp.normalize(rot * old_rot))


def _builder_body_q_array(builder: newton.ModelBuilder) -> np.ndarray:
    body_q = np.zeros((int(builder.body_count), 7), dtype=np.float32)
    for body_id, transform in enumerate(builder.body_q):
        pos = wp.transform_get_translation(transform)
        rot = wp.transform_get_rotation(transform)
        body_q[body_id] = (
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(rot[0]),
            float(rot[1]),
            float(rot[2]),
            float(rot[3]),
        )
    return body_q


def _scale_plug_mesh_shapes(
    builder: newton.ModelBuilder, cable_results: list[CableCurveImportResult], *, xy_scale: float, mu: float
) -> None:
    head_bodies = {body_id for result in cable_results for body_id in result.head_body_ids}
    for shape_id, body_id in enumerate(builder.shape_body):
        if int(body_id) not in head_bodies:
            continue
        if int(builder.shape_type[shape_id]) != int(GeoType.MESH):
            continue
        scale = builder.shape_scale[shape_id]
        builder.shape_scale[shape_id] = wp.vec3(float(scale[0]) * xy_scale, float(scale[1]) * xy_scale, float(scale[2]))
        builder.shape_material_mu[shape_id] = float(mu)


def _filter_robot_vbd_cross_contacts(
    *,
    builder: newton.ModelBuilder,
    robot_builder: newton.ModelBuilder,
    vbd_builder: newton.ModelBuilder,
    robot_shape_count: int,
    cable_results: list[CableCurveImportResult],
) -> None:
    """Restrict robot/VBD cross contacts to gripper fingers against cable shapes.

    Only the gripper-finger shapes should interact with the cable/plug world.
    Filtering every other robot-vs-VBD pair (e.g. robot-vs-fridge) keeps the
    cross-solver contact set small and well behaved.
    """

    robot_shapes = set(range(int(robot_builder.shape_count)))
    vbd_shapes = {robot_shape_count + shape_id for shape_id in range(int(vbd_builder.shape_count))}
    coupling_robot_shapes = _robot_finger_shape_ids(robot_builder)
    coupling_vbd_shapes = {
        robot_shape_count + shape_id
        for result in cable_results
        for body_id in [*result.cable_body_ids, *result.head_body_ids]
        for shape_id in vbd_builder.body_shapes.get(body_id, [])
    }

    for robot_shape in robot_shapes:
        for vbd_shape in vbd_shapes:
            if robot_shape in coupling_robot_shapes and vbd_shape in coupling_vbd_shapes:
                continue
            builder.add_shape_collision_filter_pair(robot_shape, vbd_shape)


def _robot_finger_shape_ids(robot_builder: newton.ModelBuilder) -> set[int]:
    finger_tokens = (
        "right_gripper_leftfinger",
        "right_gripper_rightfinger",
        "left_gripper_leftfinger",
        "left_gripper_rightfinger",
    )
    shape_ids: set[int] = set()
    for body_id, label in enumerate(robot_builder.body_label):
        label_text = str(label)
        if not any(token in label_text for token in finger_tokens):
            continue
        shape_ids.update(int(shape_id) for shape_id in robot_builder.body_shapes.get(body_id, []))
    if not shape_ids:
        raise RuntimeError("No RBY1 gripper finger shapes found for coupled cross contacts.")
    return shape_ids


def _body_label(builder: newton.ModelBuilder, body_id: int) -> str:
    if body_id < len(builder.body_label):
        return str(builder.body_label[body_id])
    return f"body_{body_id}"


def _sanitize_builder_labels(builder: newton.ModelBuilder) -> None:
    """Keep generated labels valid for Newton/Kit path handling.

    The task-local USD import labels can contain repeated slashes or authored
    prim paths with colons. Those labels are fine as plain strings but are not
    valid Sdf paths once Newton visualization mirrors them into USD.
    """

    for attr_name in ("body_label", "shape_label", "joint_label", "articulation_label"):
        labels = getattr(builder, attr_name, None)
        if labels is None:
            continue
        for index, label in enumerate(labels):
            labels[index] = _sanitize_label(str(label))


def _sanitize_label(label: str) -> str:
    cleaned = label.replace(":", "_")
    while "//" in cleaned:
        cleaned = cleaned.replace("//", "/")
    return cleaned.strip("/")
