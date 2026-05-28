# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local Newton model construction for the ADMM waterhose task."""

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
class WaterhoseAdmmBuildInfo:
    """Bookkeeping from task-local model construction."""

    robot_body_count: int = 0
    robot_shape_count: int = 0
    robot_joint_q_count: int = 0
    vbd_body_count: int = 0
    vbd_shape_count: int = 0
    vbd_initial_body_q: np.ndarray | None = None
    cable_body_labels: list[str] = field(default_factory=list)
    plug_body_labels: list[str] = field(default_factory=list)
    scene_shape_ids: list[int] = field(default_factory=list)


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


def build_waterhose_admm_builder(asset_root: str | Path) -> tuple[newton.ModelBuilder, WaterhoseAdmmBuildInfo]:
    """Build a single Newton model containing the robot and VBD waterhose scene.

    The model is owned and stepped by the standard
    :class:`isaaclab_newton.physics.NewtonCoupledManager` path.
    """

    paths = WaterhoseAssetPaths.from_root(asset_root)
    shape_cfg = _create_collision_shape_config()
    fridge_xform = _compute_fridge_xform()

    robot_builder = _build_robot(paths.robot_urdf, shape_cfg)
    vbd_builder, cable_results, scene_shape_ids = _build_vbd_side(paths, fridge_xform)

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

    info = WaterhoseAdmmBuildInfo(
        robot_body_count=robot_body_count,
        robot_shape_count=robot_shape_count,
        robot_joint_q_count=int(robot_builder.joint_coord_count),
        vbd_body_count=int(vbd_builder.body_count),
        vbd_shape_count=int(vbd_builder.shape_count),
        vbd_initial_body_q=_builder_body_q_array(vbd_builder),
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
    )
    return builder, info


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
    """Match the legacy one-way contact surface for ADMM.

    The original successful demo exposes only duplicated gripper-finger proxy
    shapes to the VBD cable world. A naive ADMM contact pair between the whole
    MuJoCo robot and the whole VBD scene creates robot-vs-fridge and
    robot-vs-cable contacts for every robot body, which is both much slower and
    behaviorally different. Keep ADMM cross contacts to finger shapes against
    cable/plug shapes.
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
        raise RuntimeError("No RBY1 gripper finger shapes found for ADMM coupling.")
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
