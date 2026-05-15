# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

_ISAACLAB_ROOT = Path(__file__).resolve().parents[6]
_DEFAULT_RBY1_URDF = _ISAACLAB_ROOT / "source/isaaclab_assets/data/Robots/RBY1DF/urdf/robot_edited.urdf"
_DEFAULT_CABLE_ROBOT_ROOT = Path("/home/maximiliank/Downloads/cable_robot.tar/cable_robot")
_DEFAULT_CABLE_ROBOT_ASSETS = _DEFAULT_CABLE_ROBOT_ROOT / "assets"


def make_waterhose_args(**overrides: Any) -> SimpleNamespace:
    """Create the namespace expected by the shared Newton waterhose builder."""
    defaults = dict(
        fps=300.0,
        max_steps=1400,
        num_envs=1,
        env_spacing=2.5,
        asset_root=_DEFAULT_CABLE_ROBOT_ASSETS,
        robot_urdf=None,
        scene_usd=None,
        cable_usd=None,
        cable_prims="/World/cable001/curve_0,/World/cable002/curve_0",
        cable_prim=None,
        hose_radius=0.003,
        gripper_drive_scale=0.5,
        grasp_friction=1.0e6,
        grasp_margin=0.001,
        grasp_contact_ke=2.0e5,
        sim_substeps=10,
        rigid_substeps=1,
        proxy_iterations=1,
        vbd_iterations=15,
        disable_cuda_graph=False,
        log_interval=0,
        print_cable_poses=False,
        cable_pose_settle_seconds=None,
        print_robot_poses=False,
        headless=False,
        livestream=-1,
        enable_cameras=False,
        xr=False,
        device="cuda:0",
        visualizer=["newton"],
        cpu=False,
        verbose=False,
        info=False,
        experience="",
        rendering_mode=None,
        kit_args="",
        anim_recording_enabled=False,
        anim_recording_start_time=0.0,
        anim_recording_stop_time=10.0,
        max_visible_envs=None,
    )
    defaults.update(overrides)
    for key in ("asset_root", "robot_urdf", "scene_usd", "cable_usd"):
        if defaults[key] is not None:
            defaults[key] = Path(defaults[key]).expanduser()
    return SimpleNamespace(**defaults)


args_cli = make_waterhose_args()


def configure_waterhose_args(**overrides: Any) -> SimpleNamespace:
    """Update the process-local waterhose builder arguments."""
    global args_cli
    args_cli = make_waterhose_args(**overrides)
    return args_cli


def build_waterhose_scene(**overrides: Any):
    """Build the Newton waterhose model and coupled solver configuration."""
    configure_waterhose_args(**overrides)
    import_newton_dependencies()
    scene_builder = WaterhoseSceneBuilder()
    builder, solver_cfg = scene_builder.build()
    return scene_builder, builder, solver_cfg


np = wp = newton = ik = sim_utils = None
SolverVBD = None
JointTargetMode = None
add_cable_from_usd_curve = None
CoupledProxyCfg = CoupledSolverCfg = CoupledSolverEntryCfg = None
MJWarpSolverCfg = NewtonCfg = NewtonCoupledManager = NewtonManager = NewtonSolverCfg = ProxyCouplingCfg = None

ROBOT_ENTRY = "mjc"
HOSE_ENTRY = "vbd"
ROBOT_PRIM_PATH = "/World/RBY1DF"

RIGHT_EE = "right_gripper_end_effector"
LEFT_EE = "left_gripper_end_effector"
TORSO = "torso_hip_yaw"
_ACTIVE_SCENE_BUILDER = None
PROXY_BODY_NAMES = {
    "right_gripper_base",
    "right_gripper_camera_bracket",
    "right_gripper_leftfinger",
    "right_gripper_rightfinger",
    "left_gripper_base",
    "left_gripper_camera_bracket",
    "left_gripper_leftfinger",
    "left_gripper_rightfinger",
}

LEROBOT_INITIAL_STATE_22 = [
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
    0.09138019700534642,
    0.09098683297634125,
]


def import_newton_dependencies() -> None:
    """Import Newton modules before Isaac Lab sim touches USD physics bindings."""
    global np, wp, newton, ik, SolverVBD, JointTargetMode

    if newton is not None:
        return

    import newton as newton_module
    import numpy as np_module
    import warp as wp_module
    from newton import JointTargetMode as JointTargetModeClass
    from newton.solvers import SolverVBD as SolverVBDClass

    np = np_module
    wp = wp_module
    newton = newton_module
    SolverVBD = SolverVBDClass
    JointTargetMode = JointTargetModeClass


def _load_cable_curve_importer():
    """Load the bundle's USD cable importer without requiring it to be installed as a package."""
    helper_path = args_cli.asset_root.expanduser().resolve().parent / "usd_cable_curve_import.py"
    if not helper_path.is_file():
        raise FileNotFoundError(f"Cable USD importer not found: {helper_path}")
    spec = importlib.util.spec_from_file_location("cable_robot_usd_cable_curve_import", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load cable USD importer from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.add_cable_from_usd_curve


def import_isaaclab_runtime_dependencies() -> None:
    """Import Isaac Lab simulation/config modules after Newton USD import."""
    global ik, sim_utils
    global CoupledProxyCfg, CoupledSolverCfg, CoupledSolverEntryCfg
    global MJWarpSolverCfg, NewtonCfg, NewtonCoupledManager, NewtonManager, NewtonSolverCfg, ProxyCouplingCfg

    if sim_utils is not None:
        return

    import newton.examples as newton_examples_module
    import newton.ik as ik_module
    from isaaclab_newton.physics import (
        CoupledProxyCfg as CoupledProxyCfgClass,
    )
    from isaaclab_newton.physics import (
        CoupledSolverCfg as CoupledSolverCfgClass,
    )
    from isaaclab_newton.physics import (
        CoupledSolverEntryCfg as CoupledSolverEntryCfgClass,
    )
    from isaaclab_newton.physics import (
        MJWarpSolverCfg as MJWarpSolverCfgClass,
    )
    from isaaclab_newton.physics import (
        NewtonCfg as NewtonCfgClass,
    )
    from isaaclab_newton.physics import (
        NewtonCoupledManager as NewtonCoupledManagerClass,
    )
    from isaaclab_newton.physics import (
        NewtonManager as NewtonManagerClass,
    )
    from isaaclab_newton.physics import (
        ProxyCouplingCfg as ProxyCouplingCfgClass,
    )
    from isaaclab_newton.physics.newton_manager_cfg import NewtonSolverCfg as NewtonSolverCfgClass

    import isaaclab.sim as sim_utils_module

    newton.examples = newton_examples_module
    ik = ik_module
    sim_utils = sim_utils_module
    CoupledProxyCfg = CoupledProxyCfgClass
    CoupledSolverCfg = CoupledSolverCfgClass
    CoupledSolverEntryCfg = CoupledSolverEntryCfgClass
    MJWarpSolverCfg = MJWarpSolverCfgClass
    NewtonCfg = NewtonCfgClass
    NewtonCoupledManager = NewtonCoupledManagerClass
    NewtonManager = NewtonManagerClass
    NewtonSolverCfg = NewtonSolverCfgClass
    ProxyCouplingCfg = ProxyCouplingCfgClass


def _lerobot_22_to_urdf_28(lr: list[float]) -> list[float]:
    """Convert the compact LeRobot RBY1 arm state used by the waterhose script to URDF DOFs."""
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


def _find_label_index(labels: list[str], short_name: str) -> int:
    """Find a label by exact name or URDF namespace suffix."""
    suffix = "/" + short_name
    for index, label in enumerate(labels):
        if label == short_name or label.endswith(suffix):
            return index
    raise ValueError(f"Label {short_name!r} not found.")


def _maybe_find_label_index(labels: list[str], short_name: str) -> int | None:
    try:
        return _find_label_index(labels, short_name)
    except ValueError:
        return None


def _quat_to_vec4(quat) -> object:
    return wp.vec4(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def _np_quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _np_quat_inverse(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _np_quat_rotate(q, v):
    xyz = np.asarray(q[:3], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(xyz, v)
    return v + q[3] * t + np.cross(xyz, t)


def _np_quat_from_axis_angle(axis, angle: float):
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm <= 1.0e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = axis / norm
    half = 0.5 * angle
    return np.array([axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half), np.cos(half)])


def _np_relative_transform(parent_pos, parent_quat, child_pos, child_quat) -> tuple[np.ndarray, np.ndarray]:
    parent_inv = _np_quat_inverse(parent_quat)
    rel_pos = _np_quat_rotate(
        parent_inv, np.asarray(child_pos, dtype=np.float64) - np.asarray(parent_pos, dtype=np.float64)
    )
    rel_quat = _np_quat_multiply(parent_inv, child_quat)
    return rel_pos, rel_quat / max(np.linalg.norm(rel_quat), 1.0e-12)


def _np_transform_point_quat(parent_pos, parent_quat, rel_pos, rel_quat) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(parent_pos, dtype=np.float64) + _np_quat_rotate(parent_quat, rel_pos)
    quat = _np_quat_multiply(parent_quat, rel_quat)
    return pos, quat / max(np.linalg.norm(quat), 1.0e-12)


def _np_quat_slerp(q0, q1, t):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / max(np.linalg.norm(result), 1.0e-12)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    a = np.sin((1.0 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta
    return a * q0 + b * q1


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _create_launcher_sim_cfg():
    """Create the minimal config used by ``launch_simulation`` to decide whether Kit is needed."""
    device = str(args_cli.device)
    if not device.startswith("cuda"):
        raise RuntimeError("The waterhose proxy-coupled demo requires a CUDA device.")
    dummy_newton_cfg = type("NewtonCfg", (), {"class_type": object})()
    return SimpleNamespace(
        dt=1.0 / args_cli.fps,
        device=device,
        gravity=(0.0, 0.0, -9.81),
        physics=dummy_newton_cfg,
    )


class WaterhoseSceneBuilder:
    """Build a shared model and Isaac Lab coupled-solver cfg for the demo."""

    def __init__(self):
        self.num_envs = int(args_cli.num_envs)
        self.env_spacing = float(args_cli.env_spacing)
        self.env_origins = [self._env_origin(env_id) for env_id in range(self.num_envs)]
        self.asset_root = args_cli.asset_root.expanduser().resolve()
        self.robot_urdf = self._resolve_robot_urdf()
        self.scene_usd = self._resolve_asset(args_cli.scene_usd, "Cable008/Cable008_Body.usda")
        self.cable_usd = self._resolve_asset(args_cli.cable_usd, "Cable008/curve/cable_SRA_curve03.usda")

        self.robot_shape_cfg = newton.ModelBuilder.ShapeConfig(margin=0.0, gap=0.005, ke=5.0e4, kd=5.0e2, mu=2.0)
        self.robot_shape_cfg.is_hydroelastic = False
        self.hose_shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=1000.0,
            margin=0.0,
            gap=0.001,
            ke=1.0e3,
            kd=0.0,
            mu=0.2,
        )
        self.static_shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            margin=1.0e-4,
            gap=0.001,
            ke=1.0e5,
            kd=1.0e-1,
            mu=1.0e1,
        )
        self.cable_stretch_stiffness = 1.0e6
        self.cable_stretch_damping = 1.0e-5
        self.cable_bend_stiffness = 2.0e1
        self.cable_bend_damping = 1.0

        self.table_pos = wp.vec3(0.95, -0.051, 0.1925)
        self.table_half_size = (0.55, 0.35, 0.1925)
        self.asset_xform = self._compute_asset_xform()
        self.socket_xform = self._compute_socket_xform(self.asset_xform)
        self.socket_pos = wp.transform_get_translation(self.socket_xform)
        self.socket_rot = wp.transform_get_rotation(self.socket_xform)
        self.hose_plug_tip_pos = self.socket_pos

        self.single_robot_model = None
        self.gripper_dofs: list[int] = []
        self.right_gripper_dofs: list[int] = []
        self.left_gripper_dofs: list[int] = []
        self.proxy_body_ids: list[int] = []
        self.proxy_shape_ids: list[int] = []
        self.tip_body_id = 0
        self.plug_body_id = 0
        self.grasp_body_id = 0
        self.scene_body_ids: list[int] = []
        self.scene_shape_ids: list[int] = []
        self.cable_body_q_targets: dict[int, tuple[float, float, float, float, float, float, float]] = {}
        self.primary_cable_body_ids: list[int] = []

        self._mujoco_body_count = 0
        self._mujoco_shape_count = 0
        self._mujoco_joint_count = 0
        self._mujoco_joint_coord_count = 0
        self._mujoco_joint_dof_count = 0
        self._mujoco_articulation_count = 0
        self._mujoco_body_ids: list[int] = []
        self._mujoco_shape_ids: list[int] = []
        self._mujoco_joint_ids: list[int] = []
        self._vbd_body_ids: list[int] = []
        self._vbd_shape_ids: list[int] = []
        self._vbd_joint_ids: list[int] = []
        self.cable_body_ids: list[int] = []
        self.cable_head_body_ids: list[int] = []
        self.robot_joint_coord_ids_by_env: list[list[int]] = []

    def _env_origin(self, env_id: int):
        cols = int(np.ceil(np.sqrt(self.num_envs)))
        row = env_id // cols
        col = env_id % cols
        return wp.vec3(float(col) * self.env_spacing, float(row) * self.env_spacing, 0.0)

    def _resolve_asset(self, explicit_path: Path | None, relative_path: str) -> Path:
        path = explicit_path.expanduser().resolve() if explicit_path is not None else self.asset_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Required cable robot asset not found: {path}")
        return path

    def _resolve_robot_urdf(self) -> Path:
        if args_cli.robot_urdf is not None:
            path = args_cli.robot_urdf.expanduser().resolve()
        else:
            bundled = self.asset_root / "rby1df/urdf/robot_edited.urdf"
            path = bundled if bundled.is_file() else _DEFAULT_RBY1_URDF
        if not path.is_file():
            raise FileNotFoundError(f"RBY1DF URDF not found: {path}")
        return path

    @staticmethod
    def _compute_asset_xform():
        """Match the bundle scripts' Cable008 placement relative to the robot."""
        table_half_z = 0.5 * (0.6 - 0.215)
        table_top_z = table_half_z + table_half_z
        z_offset = 0.902 + table_top_z
        y_offset = (0.293 - 0.395) / 2.0
        quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi / 2.0)
        return wp.transform(wp.vec3(0.95, y_offset, z_offset), quat)

    @staticmethod
    def _compute_socket_xform(asset_xform):
        socket_offset = wp.vec3(-0.259404, 0.362961, -0.262711)
        pos = wp.transform_point(asset_xform, socket_offset)
        rot = wp.transform_get_rotation(asset_xform) * wp.quat_from_axis_angle(
            wp.vec3(1.0, 0.0, 0.0), 20.0 * wp.pi / 180.0
        )
        return wp.transform(pos, rot)

    def build(self):
        """Return ``(builder, solver_cfg)`` for Isaac Lab's Newton manager."""
        static_scene_builder = self._build_static_scene_template()
        robot = self._build_robot()
        proto_builder = self._build_single_env_proto(robot, static_scene_builder)
        proto_meta = self._capture_proto_metadata(proto_builder, robot)
        builder = self._build_replicated_builder(proto_builder, proto_meta, robot, static_scene_builder)
        # Finalize the standalone IK model after USD scene parsing.  The USD
        # physics importer for Cable008 is fragile if called after CUDA/Warp
        # solver modules have already initialized during robot.finalize().
        self.single_robot_model = robot.finalize(device=str(args_cli.device))
        builder.color()
        import_isaaclab_runtime_dependencies()
        NewtonManager._num_envs = self.num_envs
        return builder, self._make_solver_cfg()

    def _build_single_env_proto(self, robot, static_scene_builder):
        builder = newton.ModelBuilder()
        SolverVBD.register_custom_attributes(builder)
        builder.default_shape_cfg = self.robot_shape_cfg
        builder.bound_mass = robot.bound_mass
        builder.bound_inertia = robot.bound_inertia
        builder.add_builder(robot)

        self._mujoco_body_count = builder.body_count
        self._mujoco_shape_count = builder.shape_count
        self._mujoco_joint_count = builder.joint_count
        self._mujoco_joint_coord_count = builder.joint_coord_count
        self._mujoco_joint_dof_count = builder.joint_dof_count
        self._mujoco_articulation_count = builder.articulation_count
        self._mujoco_body_ids = list(range(self._mujoco_body_count))
        self._mujoco_shape_ids = list(range(self._mujoco_shape_count))
        self._mujoco_joint_ids = list(range(self._mujoco_joint_count))

        self._add_waterhose_world(builder, static_scene_builder)
        self._select_proxy_bodies(builder)
        return builder

    def _capture_proto_metadata(self, proto_builder, robot) -> dict[str, object]:
        return {
            "mujoco_body_ids": list(self._mujoco_body_ids),
            "mujoco_shape_ids": list(self._mujoco_shape_ids),
            "mujoco_joint_ids": list(self._mujoco_joint_ids),
            "mujoco_joint_coord_count": self._mujoco_joint_coord_count,
            "mujoco_joint_dof_count": self._mujoco_joint_dof_count,
            "mujoco_articulation_count": self._mujoco_articulation_count,
            "vbd_body_ids": list(self._vbd_body_ids),
            "vbd_shape_ids": list(self._vbd_shape_ids),
            "vbd_joint_ids": list(self._vbd_joint_ids),
            "proxy_body_ids": list(self.proxy_body_ids),
            "proxy_shape_ids": list(self.proxy_shape_ids),
            "scene_body_ids": list(self.scene_body_ids),
            "scene_shape_ids": list(self.scene_shape_ids),
            "cable_body_ids": list(self.cable_body_ids),
            "cable_head_body_ids": list(self.cable_head_body_ids),
            "primary_cable_body_ids": list(self.primary_cable_body_ids),
            "tip_body_id": self.tip_body_id,
            "plug_body_id": self.plug_body_id,
            "grasp_body_id": self.grasp_body_id,
            "cable_body_q_targets": dict(self.cable_body_q_targets),
            "robot_joint_coord_count": robot.joint_coord_count,
        }

    def _build_replicated_builder(self, proto_builder, proto_meta: dict[str, object], robot, static_scene_builder):
        self._reset_replicated_metadata()
        if self.num_envs == 1:
            self._append_env_metadata(proto_meta, (0, 0, 0, 0, 0), self.env_origins[0], is_primary=True)
            return proto_builder

        builder = newton.ModelBuilder()
        SolverVBD.register_custom_attributes(builder)
        builder.default_shape_cfg = self.robot_shape_cfg
        builder.bound_mass = proto_builder.bound_mass
        builder.bound_inertia = proto_builder.bound_inertia
        for env_id, origin in enumerate(self.env_origins):
            body_offset = builder.body_count
            shape_offset = builder.shape_count
            joint_offset = builder.joint_count
            joint_coord_offset = builder.joint_coord_count
            builder.add_builder(robot, xform=wp.transform(origin, wp.quat_identity()))
            self._mujoco_body_ids.extend(range(body_offset, body_offset + robot.body_count))
            self._mujoco_shape_ids.extend(range(shape_offset, shape_offset + robot.shape_count))
            self._mujoco_joint_ids.extend(range(joint_offset, joint_offset + robot.joint_count))
            self._mujoco_body_count += robot.body_count
            self._mujoco_shape_count += robot.shape_count
            self._mujoco_joint_count += robot.joint_count
            self._mujoco_joint_coord_count += robot.joint_coord_count
            self._mujoco_joint_dof_count += robot.joint_dof_count
            self._mujoco_articulation_count += robot.articulation_count
            self.robot_joint_coord_ids_by_env.append(
                [joint_coord_offset + joint_id for joint_id in range(robot.joint_coord_count)]
            )

        self._select_proxy_bodies(builder)

        vbd_body_ids: list[int] = []
        vbd_shape_ids: list[int] = []
        vbd_joint_ids: list[int] = []
        primary_env_metadata = None
        for env_id, origin in enumerate(self.env_origins):
            self._add_waterhose_world(builder, static_scene_builder, origin=origin)
            vbd_body_ids.extend(self._vbd_body_ids)
            vbd_shape_ids.extend(self._vbd_shape_ids)
            vbd_joint_ids.extend(self._vbd_joint_ids)
            if env_id == 0:
                primary_env_metadata = {
                    "scene_body_ids": list(self.scene_body_ids),
                    "scene_shape_ids": list(self.scene_shape_ids),
                    "cable_body_ids": list(self.cable_body_ids),
                    "cable_head_body_ids": list(self.cable_head_body_ids),
                    "primary_cable_body_ids": list(self.primary_cable_body_ids),
                    "tip_body_id": self.tip_body_id,
                    "plug_body_id": self.plug_body_id,
                    "grasp_body_id": self.grasp_body_id,
                }

        self._vbd_body_ids = vbd_body_ids
        self._vbd_shape_ids = vbd_shape_ids
        self._vbd_joint_ids = vbd_joint_ids
        if primary_env_metadata is not None:
            self.scene_body_ids = primary_env_metadata["scene_body_ids"]
            self.scene_shape_ids = primary_env_metadata["scene_shape_ids"]
            self.cable_body_ids = primary_env_metadata["cable_body_ids"]
            self.cable_head_body_ids = primary_env_metadata["cable_head_body_ids"]
            self.primary_cable_body_ids = primary_env_metadata["primary_cable_body_ids"]
            self.tip_body_id = primary_env_metadata["tip_body_id"]
            self.plug_body_id = primary_env_metadata["plug_body_id"]
            self.grasp_body_id = primary_env_metadata["grasp_body_id"]
        return builder

    def _reset_replicated_metadata(self) -> None:
        self._mujoco_body_count = 0
        self._mujoco_shape_count = 0
        self._mujoco_joint_count = 0
        self._mujoco_joint_coord_count = 0
        self._mujoco_joint_dof_count = 0
        self._mujoco_articulation_count = 0
        self._mujoco_body_ids = []
        self._mujoco_shape_ids = []
        self._mujoco_joint_ids = []
        self._vbd_body_ids = []
        self._vbd_shape_ids = []
        self._vbd_joint_ids = []
        self.proxy_body_ids = []
        self.proxy_shape_ids = []
        self.cable_body_q_targets = {}
        self.robot_joint_coord_ids_by_env = []

    def _append_env_metadata(
        self,
        proto_meta: dict[str, object],
        offsets: tuple[int, int, int, int, int],
        origin,
        *,
        is_primary: bool,
    ) -> None:
        body_offset, shape_offset, joint_offset, joint_coord_offset, _joint_dof_offset = offsets

        def offset_ids(ids: list[int], offset: int) -> list[int]:
            return [offset + int(idx) for idx in ids]

        self._mujoco_body_ids.extend(offset_ids(proto_meta["mujoco_body_ids"], body_offset))
        self._mujoco_shape_ids.extend(offset_ids(proto_meta["mujoco_shape_ids"], shape_offset))
        self._mujoco_joint_ids.extend(offset_ids(proto_meta["mujoco_joint_ids"], joint_offset))
        self._vbd_body_ids.extend(offset_ids(proto_meta["vbd_body_ids"], body_offset))
        self._vbd_shape_ids.extend(offset_ids(proto_meta["vbd_shape_ids"], shape_offset))
        self._vbd_joint_ids.extend(offset_ids(proto_meta["vbd_joint_ids"], joint_offset))
        self.proxy_body_ids.extend(offset_ids(proto_meta["proxy_body_ids"], body_offset))
        self.proxy_shape_ids.extend(offset_ids(proto_meta["proxy_shape_ids"], shape_offset))

        self._mujoco_body_count += len(proto_meta["mujoco_body_ids"])
        self._mujoco_shape_count += len(proto_meta["mujoco_shape_ids"])
        self._mujoco_joint_count += len(proto_meta["mujoco_joint_ids"])
        self._mujoco_joint_coord_count += int(proto_meta["mujoco_joint_coord_count"])
        self._mujoco_joint_dof_count += int(proto_meta["mujoco_joint_dof_count"])
        self._mujoco_articulation_count += int(proto_meta["mujoco_articulation_count"])

        robot_joint_coord_count = int(proto_meta["robot_joint_coord_count"])
        self.robot_joint_coord_ids_by_env.append(
            [joint_coord_offset + joint_id for joint_id in range(robot_joint_coord_count)]
        )

        origin_np = np.array([float(origin[0]), float(origin[1]), float(origin[2]), 0.0, 0.0, 0.0, 0.0])
        for proto_body_id, target_q in proto_meta["cable_body_q_targets"].items():
            target_np = np.asarray(target_q, dtype=np.float64) + origin_np
            self.cable_body_q_targets[body_offset + int(proto_body_id)] = tuple(float(v) for v in target_np)

        if not is_primary:
            return
        self.scene_body_ids = offset_ids(proto_meta["scene_body_ids"], body_offset)
        self.scene_shape_ids = offset_ids(proto_meta["scene_shape_ids"], shape_offset)
        self.cable_body_ids = offset_ids(proto_meta["cable_body_ids"], body_offset)
        self.cable_head_body_ids = offset_ids(proto_meta["cable_head_body_ids"], body_offset)
        self.primary_cable_body_ids = offset_ids(proto_meta["primary_cable_body_ids"], body_offset)
        self.tip_body_id = body_offset + int(proto_meta["tip_body_id"])
        self.plug_body_id = body_offset + int(proto_meta["plug_body_id"])
        self.grasp_body_id = body_offset + int(proto_meta["grasp_body_id"])

    def _build_static_scene_template(self):
        """Import the authored static scene through Newton's USD importer before robot setup."""
        builder = newton.ModelBuilder()
        SolverVBD.register_custom_attributes(builder)
        builder.default_shape_cfg = self.static_shape_cfg
        self._add_static_table_and_socket(builder)
        return builder

    def _build_robot(self):
        robot = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
        robot.default_shape_cfg = self.robot_shape_cfg
        robot.bound_mass = 1.0e-4
        robot.bound_inertia = 1.0e-6
        robot.add_urdf(
            str(self.robot_urdf),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
            ignore_inertial_definitions=True,
        )

        self._patch_zero_mass_children(robot)
        self._disable_gripper_mimic_constraints(robot)
        self._discover_gripper_dofs(robot)
        self._configure_robot_dofs(robot)
        self._seed_initial_joint_state(robot)
        self._enable_gravity_compensation(robot)
        return robot

    def _patch_zero_mass_children(self, robot) -> None:
        """Patch invalid zero-mass driven links at runtime without editing the URDF asset."""
        min_inertia = wp.mat33(
            robot.bound_inertia,
            0.0,
            0.0,
            0.0,
            robot.bound_inertia,
            0.0,
            0.0,
            0.0,
            robot.bound_inertia,
        )
        for child in {int(v) for v in robot.joint_child if int(v) >= 0}:
            if robot.body_mass[child] <= 0.0:
                robot.body_mass[child] = robot.bound_mass
                robot.body_inv_mass[child] = 1.0 / robot.bound_mass
                robot.body_inertia[child] = min_inertia
                robot.body_inv_inertia[child] = wp.inverse(min_inertia)

    @staticmethod
    def _disable_gripper_mimic_constraints(robot) -> None:
        """Match the Newton cable example's pre-mimic gripper behavior."""
        mimic_enabled = getattr(robot, "constraint_mimic_enabled", [])
        mimic_count = len(mimic_enabled)
        if mimic_count > 0:
            robot.constraint_mimic_enabled[-mimic_count:] = [False] * mimic_count

    def _discover_gripper_dofs(self, robot) -> None:
        right_names = ["right_gripper_left_finger_joint", "right_gripper_right_finger_joint"]
        left_names = ["left_gripper_left_finger_joint", "left_gripper_right_finger_joint"]
        self.right_gripper_dofs = self._joint_dofs_by_name(robot, right_names)
        self.left_gripper_dofs = self._joint_dofs_by_name(robot, left_names)
        self.gripper_dofs = [*self.right_gripper_dofs, *self.left_gripper_dofs]

    @staticmethod
    def _joint_dofs_by_name(robot, names: list[str]) -> list[int]:
        dofs = []
        for name in names:
            joint_id = _maybe_find_label_index(robot.joint_label, name)
            if joint_id is not None:
                dofs.append(int(robot.joint_qd_start[joint_id]))
        return dofs

    def _configure_robot_dofs(self, robot) -> None:
        gripper_set = set(self.gripper_dofs)
        gripper_drive_scale = float(args_cli.gripper_drive_scale)
        for dof in range(robot.joint_dof_count):
            if dof in gripper_set:
                robot.joint_target_ke[dof] = 10000.0 * gripper_drive_scale
                robot.joint_target_kd[dof] = 1000.0 * gripper_drive_scale
                robot.joint_effort_limit[dof] = 100000.0 * gripper_drive_scale
                robot.joint_armature[dof] = 0.5
            else:
                robot.joint_target_ke[dof] = 45000.0
                robot.joint_target_kd[dof] = 4500.0
                robot.joint_effort_limit[dof] = 1000.0
                robot.joint_armature[dof] = 0.2
            robot.joint_target_mode[dof] = int(JointTargetMode.POSITION)

    def _seed_initial_joint_state(self, robot) -> None:
        q = _lerobot_22_to_urdf_28(LEROBOT_INITIAL_STATE_22)
        if len(q) != robot.joint_coord_count:
            q = (q + [0.0] * robot.joint_coord_count)[: robot.joint_coord_count]
        robot.joint_q = q
        robot.joint_target_pos = list(q)

    @staticmethod
    def _enable_gravity_compensation(robot) -> None:
        gravcomp_body = robot.custom_attributes.get("mujoco:gravcomp")
        if gravcomp_body is not None:
            gravcomp_body.values = gravcomp_body.values or {}
            for body_id in range(1, robot.body_count):
                gravcomp_body.values[body_id] = 1.0

        gravcomp_joint = robot.custom_attributes.get("mujoco:jnt_actgravcomp")
        if gravcomp_joint is not None:
            gravcomp_joint.values = gravcomp_joint.values or {}
            for dof_id in range(robot.joint_dof_count):
                gravcomp_joint.values[dof_id] = True

    def _add_waterhose_world(self, builder, static_scene_builder, origin=None) -> None:
        builder.default_shape_cfg = self.hose_shape_cfg
        builder.rigid_contact_margin = 0.0
        builder.rigid_gap = 0.001
        vbd_body_start = builder.body_count
        vbd_shape_start = builder.shape_count
        vbd_joint_start = builder.joint_count

        scene_body_start = builder.body_count
        scene_shape_start = builder.shape_count
        if origin is None:
            builder.add_builder(static_scene_builder)
        else:
            builder.add_builder(static_scene_builder, xform=wp.transform(origin, wp.quat_identity()))
        self.scene_body_ids = list(range(scene_body_start, builder.body_count))
        self.scene_shape_ids = list(range(scene_shape_start, builder.shape_count))
        for shape_index, shape_a in enumerate(self.scene_shape_ids):
            for shape_b in self.scene_shape_ids[shape_index + 1 :]:
                builder.add_shape_collision_filter_pair(shape_a, shape_b)

        self._add_hoses_from_usd(builder, origin=origin)

        self._vbd_body_ids = list(range(vbd_body_start, builder.body_count))
        self._vbd_shape_ids = list(range(vbd_shape_start, builder.shape_count))
        self._vbd_joint_ids = list(range(vbd_joint_start, builder.joint_count))

    def _add_hoses_from_usd(self, builder, origin=None) -> None:
        global add_cable_from_usd_curve

        if add_cable_from_usd_curve is None:
            add_cable_from_usd_curve = _load_cable_curve_importer()

        cfg = self.hose_shape_cfg.copy()
        head_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, ke=1.0e3, kd=0.0, mu=0.2)
        cable_joint_ids: list[int] = []
        fixed_body_ids: list[int] = []
        self.cable_body_ids = []
        self.cable_head_body_ids = []

        for cable_index, curve_prim_path in enumerate(self._cable_prim_paths()):
            result = add_cable_from_usd_curve(
                builder=builder,
                source_usd_path=str(self.cable_usd),
                curve_prim_path=curve_prim_path,
                cable_label=f"water_hose_cable_{cable_index}",
                cable_cfg=cfg,
                stretch_stiffness=self.cable_stretch_stiffness,
                stretch_damping=self.cable_stretch_damping,
                bend_stiffness=self.cable_bend_stiffness,
                bend_damping=self.cable_bend_damping,
                wrap_in_articulation=False,
                head_shape_mode="mesh",
                head_cfg=head_cfg,
                head_mass=0.0,
            )
            self.cable_body_ids.extend(result.cable_body_ids)
            self.cable_head_body_ids.extend(result.head_body_ids)
            cable_joint_ids.extend([*result.cable_joint_ids, *result.head_fixed_joint_ids])
            fixed_body_ids.extend(int(v) for v in result.fixed_body_ids)
            self._sanitize_imported_labels(builder, result, cable_index)
            self._filter_cable_self_collisions(builder, [*result.cable_body_ids, *result.head_body_ids])
            self._filter_head_parent_neighbor_collisions(builder, result, cable_index)
            self._cache_asset_transformed_body_targets(builder, [*result.cable_body_ids, *result.head_body_ids], origin)

            if cable_index == 0:
                self.primary_cable_body_ids = list(result.cable_body_ids)
                self.tip_body_id = result.cable_body_ids[0]
                self.plug_body_id = result.head_body_ids[0] if result.head_body_ids else result.cable_body_ids[0]
                self.grasp_body_id = self.plug_body_id

        if not self.cable_body_ids:
            raise RuntimeError("At least one cable prim must be configured.")
        if cable_joint_ids:
            builder.add_articulation(cable_joint_ids, label="waterhose_articulation")
        for body_id in sorted(set(fixed_body_ids)):
            builder.body_mass[body_id] = 0.0
            builder.body_inv_mass[body_id] = 0.0
            builder.body_inertia[body_id] = wp.mat33()
            builder.body_inv_inertia[body_id] = wp.mat33()

    @staticmethod
    def _cable_prim_paths() -> list[str]:
        if args_cli.cable_prim:
            return [args_cli.cable_prim]
        return [path.strip() for path in args_cli.cable_prims.split(",") if path.strip()]

    @staticmethod
    def _sanitize_imported_labels(builder, result, cable_index: int) -> None:
        """Replace importer labels containing USD-path punctuation with valid visualization paths."""
        body_ids = [*result.cable_body_ids, *result.head_body_ids]
        joint_ids = [*result.cable_joint_ids, *result.head_fixed_joint_ids]
        shape_ids: list[int] = []
        for body_id in body_ids:
            shape_ids.extend(builder.body_shapes.get(body_id, []))
        for index, body_id in enumerate(body_ids):
            builder.body_label[body_id] = f"/World/Waterhose/Cable_{cable_index}/Body_{index:03d}"
        for index, shape_id in enumerate(shape_ids):
            builder.shape_label[shape_id] = f"/World/Waterhose/Cable_{cable_index}/Shape_{index:03d}"
        for index, joint_id in enumerate(joint_ids):
            builder.joint_label[joint_id] = f"/World/Waterhose/Cable_{cable_index}/Joint_{index:03d}"

    @staticmethod
    def _filter_cable_self_collisions(builder, body_ids: list[int]) -> None:
        shape_ids: list[int] = []
        for body_id in body_ids:
            shape_ids.extend(builder.body_shapes.get(body_id, []))
        for i, shape_a in enumerate(shape_ids):
            for shape_b in shape_ids[i + 1 :]:
                builder.add_shape_collision_filter_pair(shape_a, shape_b)

    @staticmethod
    def _filter_head_parent_neighbor_collisions(builder, result, cable_index: int) -> None:
        if not result.head_body_ids or len(result.cable_body_ids) < 2:
            return
        neighbor_index = 1 if cable_index == 0 else -2
        neighbor_body = result.cable_body_ids[neighbor_index]
        for head_body in result.head_body_ids:
            for head_shape in builder.body_shapes.get(head_body, []):
                for neighbor_shape in builder.body_shapes.get(neighbor_body, []):
                    builder.add_shape_collision_filter_pair(head_shape, neighbor_shape)

    def apply_runtime_cable_asset_xform(self) -> None:
        """Apply the authored cable scene transform to model/state buffers after Isaac Lab reset."""
        if not self.cable_body_q_targets:
            return

        def apply_to_array(body_q_arr) -> None:
            body_q_np = body_q_arr.numpy()
            for body_id, target_q in self.cable_body_q_targets.items():
                body_q_np[body_id] = target_q
            wp.copy(body_q_arr, wp.array(body_q_np, dtype=wp.transform, device=body_q_arr.device))

        apply_to_array(NewtonManager.get_model().body_q)
        apply_to_array(NewtonManager.get_state_0().body_q)
        apply_to_array(NewtonManager.get_state_1().body_q)

        vbd_solver = NewtonCoupledManager.get_entry_solver(HOSE_ENTRY)
        body_q_prev = getattr(vbd_solver, "body_q_prev", None)
        if body_q_prev is not None:
            apply_to_array(body_q_prev)

        if args_cli.print_cable_poses:
            self._print_runtime_cable_pose_summary("manager_runtime_after_asset_xform")

    def configure_runtime_vbd_solver(self) -> None:
        """Match the waterhose reference: VBD joints run in soft penalty mode."""
        vbd_solver = NewtonCoupledManager.get_entry_solver(HOSE_ENTRY)
        set_mode = getattr(vbd_solver, "set_joint_constraint_mode", None)
        if set_mode is None:
            return
        for joint_id in self._vbd_joint_ids:
            set_mode(joint_id, hard=False)

    def _cache_asset_transformed_body_targets(self, builder, body_ids: list[int], origin=None) -> None:
        rot = wp.transform_get_rotation(self.asset_xform)
        pos = wp.transform_get_translation(self.asset_xform)
        if origin is not None:
            pos = pos + origin
        for body_id in body_ids:
            body_q = builder.body_q[body_id]
            old_pos = wp.transform_get_translation(body_q)
            old_rot = wp.transform_get_rotation(body_q)
            new_pos = wp.quat_rotate(rot, old_pos) + pos
            new_rot = rot * old_rot
            self.cable_body_q_targets[body_id] = (
                float(new_pos[0]),
                float(new_pos[1]),
                float(new_pos[2]),
                float(new_rot[0]),
                float(new_rot[1]),
                float(new_rot[2]),
                float(new_rot[3]),
            )

    def _print_runtime_cable_pose_summary(self, tag: str) -> None:
        body_q = NewtonManager.get_state_0().body_q.numpy()

        def fmt_transform(body_id: int) -> str:
            q = body_q[body_id]
            return (
                f"id={body_id} pos=({float(q[0]): .6f}, {float(q[1]): .6f}, {float(q[2]): .6f}) "
                f"quat_xyzw=({float(q[3]): .6f}, {float(q[4]): .6f}, {float(q[5]): .6f}, {float(q[6]): .6f})"
            )

        cable_heads = set(self.cable_head_body_ids)
        cable_groups: list[list[int]] = []
        current_group: list[int] = []
        for body_id in self.cable_body_ids:
            if body_id in cable_heads:
                continue
            if current_group and body_id != current_group[-1] + 1:
                cable_groups.append(current_group)
                current_group = []
            current_group.append(body_id)
        if current_group:
            cable_groups.append(current_group)

        print(f"[CABLE_POSE:{tag}] cable_count={len(cable_groups)}", flush=True)
        for cable_index, body_ids in enumerate(cable_groups):
            print(f"[CABLE_POSE:{tag}] cable={cable_index} first {fmt_transform(body_ids[0])}", flush=True)
            print(f"[CABLE_POSE:{tag}] cable={cable_index} last  {fmt_transform(body_ids[-1])}", flush=True)
            head_index = 0
            for head_body in self.cable_head_body_ids:
                if body_ids[0] <= head_body <= body_ids[-1] + 1:
                    print(
                        f"[CABLE_POSE:{tag}] cable={cable_index} head{head_index} {fmt_transform(head_body)}",
                        flush=True,
                    )
                    head_index += 1

    def _add_static_table_and_socket(self, builder) -> None:
        scene_result = builder.add_usd(
            str(self.scene_usd),
            xform=self.asset_xform,
            root_path="/root",
            load_sites=False,
            load_visual_shapes=True,
            hide_collision_shapes=False,
            parse_mujoco_options=False,
            only_load_enabled_joints=True,
            only_load_enabled_rigid_bodies=False,
        )
        self.scene_body_ids = sorted({int(v) for v in scene_result["path_body_map"].values()})
        for body_id in self.scene_body_ids:
            builder.body_mass[body_id] = 0.0
            builder.body_inv_mass[body_id] = 0.0
            builder.body_inertia[body_id] = wp.mat33()
            builder.body_inv_inertia[body_id] = wp.mat33()
        self.scene_shape_ids = sorted(int(v) for v in scene_result["path_shape_map"].values())

    def _select_proxy_bodies(self, builder) -> None:
        self.proxy_body_ids = []
        self.proxy_shape_ids = []
        for body_id in self._mujoco_body_ids:
            label = builder.body_label[body_id] if body_id < len(builder.body_label) else ""
            short_label = label.rsplit("/", 1)[-1]
            if short_label not in PROXY_BODY_NAMES:
                continue
            shape_ids = [
                int(shape_id)
                for shape_id in self._mujoco_shape_ids
                for shape_body in [builder.shape_body[shape_id]]
                if int(shape_body) == body_id
                and (int(builder.shape_flags[shape_id]) & int(newton.ShapeFlags.COLLIDE_SHAPES))
            ]
            if shape_ids:
                self.proxy_body_ids.append(body_id)
                self.proxy_shape_ids.extend(shape_ids)
        if not self.proxy_body_ids:
            raise RuntimeError("No RBY1 gripper proxy bodies found for waterhose coupling.")

    def _make_solver_cfg(self):
        global _ACTIVE_SCENE_BUILDER
        _ACTIVE_SCENE_BUILDER = self

        def create_proxy_collision_pipeline(model_view):
            return newton.examples.create_collision_pipeline(
                model_view,
                args_cli,
                contact_matching="sticky",
                contact_matching_pos_threshold=0.005,
                contact_matching_normal_dot_threshold=0.95,
            )

        return CoupledSolverCfg(
            entries=[
                CoupledSolverEntryCfg(
                    name=ROBOT_ENTRY,
                    solver_cfg=MJWarpSolverCfg(
                        solver="newton",
                        integrator="implicitfast",
                        cone="elliptic",
                        njmax=100000,
                        nconmax=100000,
                        iterations=20,
                        ls_iterations=10,
                        ls_parallel=True,
                        impratio=1000.0,
                        use_mujoco_contacts=True,
                    ),
                    bodies=self._mujoco_body_ids,
                    joints=self._mujoco_joint_ids,
                    shapes=self._mujoco_shape_ids,
                    configure_view=configure_mujoco_view,
                    substeps=args_cli.rigid_substeps,
                ),
                CoupledSolverEntryCfg(
                    name=HOSE_ENTRY,
                    solver_cfg=NewtonSolverCfg(),
                    solver_class=SolverVBD,
                    solver_kwargs={
                        "iterations": args_cli.vbd_iterations,
                        "friction_epsilon": 0.1,
                        "rigid_contact_hard": False,
                        "rigid_body_contact_buffer_size": 1024,
                        "rigid_body_particle_contact_buffer_size": 1,
                        "rigid_joint_linear_ke": 1.0e6,
                        "rigid_joint_angular_ke": 1.0e6,
                    },
                    bodies=self._vbd_body_ids,
                    joints=self._vbd_joint_ids,
                    shapes=self._vbd_shape_ids,
                    configure_view=configure_vbd_view,
                ),
            ],
            use_collision_pipeline=False,
            proxy_coupling=ProxyCouplingCfg(
                proxies=[
                    CoupledProxyCfg(
                        source=ROBOT_ENTRY,
                        destination=HOSE_ENTRY,
                        bodies=self.proxy_body_ids,
                        mass_scale=1.0,
                        mode="lagged",
                        collision_pipeline_factory=create_proxy_collision_pipeline,
                        collide_interval=5,
                    )
                ],
                iterations=args_cli.proxy_iterations,
            ),
        )


def _active_scene_builder() -> WaterhoseSceneBuilder:
    if _ACTIVE_SCENE_BUILDER is None:
        raise RuntimeError("Waterhose scene builder is not active.")
    return _ACTIVE_SCENE_BUILDER


def configure_mujoco_view(view) -> None:
    """Restrict the MuJoCo view to the robot prefix in the shared model."""
    scene_builder = _active_scene_builder()
    view.body_count = scene_builder._mujoco_body_count
    view.shape_count = scene_builder._mujoco_shape_count
    view.joint_count = scene_builder._mujoco_joint_count
    view.joint_coord_count = scene_builder._mujoco_joint_coord_count
    view.joint_dof_count = scene_builder._mujoco_joint_dof_count
    view.articulation_count = scene_builder._mujoco_articulation_count


def configure_vbd_view(view) -> None:
    """Match the reference script's gripper-finger proxy contact material."""
    scene_builder = _active_scene_builder()
    if not scene_builder.proxy_shape_ids:
        return
    proxy_shape_ids = np.asarray(scene_builder.proxy_shape_ids, dtype=np.int32)
    model = view.parent
    for attr, value in (
        ("shape_material_ke", float(args_cli.grasp_contact_ke)),
        ("shape_material_kd", 1.0e-1),
        ("shape_material_mu", float(args_cli.grasp_friction)),
        ("shape_margin", float(args_cli.grasp_margin)),
    ):
        data = getattr(model, attr, None)
        if data is None:
            continue
        data_np = data.numpy().copy()
        data_np[proxy_shape_ids] = value
        setattr(view, attr, wp.array(data_np, dtype=wp.float32, device=model.device))


class WaterhoseIKController:
    """Scripted Newton IK controller layered on Isaac Lab's Newton state/control buffers."""

    _PHASES = [
        ("approach_hose", 2.0),
        ("engage_hose", 0.9),
        ("grasp_hose", 0.4),
        ("hold_grasp", 0.25),
        ("retract", 0.9),
        ("settle", 0.1),
        ("approach_socket", 3.0),
        ("align_axes", 1.5),
        ("verify_align", 2.0),
        ("insert_hose", 3.0),
        ("release_hose", 1.0),
        ("withdraw", 2.0),
        ("wait_after_withdraw", 1.0),
        ("reapproach", 2.0),
        ("reengage", 0.4),
        ("regrasp", 3.0),
        ("pull", 4.0),
        ("final_release", 2.0),
        ("done", 999.0),
    ]

    def __init__(self, scene_builder: WaterhoseSceneBuilder):
        self.scene_builder = scene_builder
        self.model = NewtonManager.get_model()
        self.state = NewtonManager.get_state_0()
        self.control = NewtonManager.get_control()
        self.single_robot_model = scene_builder.single_robot_model
        self.ik_iters = 24
        self.start_time = 0.0
        self.last_task = ""
        self.phase_index = 0
        self.phase_start_time = 0.0
        self.last_target_pos = None
        self.last_target_quat = None
        self.live_hose_target_pos = None
        self.live_hose_target_quat = None
        self.live_hose_target_alpha = 0.35
        self.max_target_step = 0.018
        self.max_joint_step = 0.02
        self.max_gripper_joint_step = 0.20
        self.gripper_centering_k = 0.4
        self.gripper_axis_centering_k = 0.8
        self.gripper_centering_max_step = 0.003
        self.phase_start_ee_pos = np.zeros(3, dtype=np.float64)
        self.phase_start_ee_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.phase_start_plug_pos = np.zeros(3, dtype=np.float64)
        self.phase_start_plug_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.phase_target_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.verify_align_retries = 0
        self.max_verify_align_retries = 1000000
        self.align_lateral_threshold = 0.0005
        self.align_axis_cosine_threshold = -0.997
        self.align_lateral_gain = 1.0
        self.insert_start_depth = 0.005
        self.insert_final_depth = 0.035
        self.pull_distance = 0.06
        self.transfer_retract_standoff = 0.045
        self.transfer_retract_lift = 0.035
        self.socket_approach_standoff = 0.055
        self.socket_approach_lift = 0.035
        self.hose_approach_offset_local = np.array([0.0, 0.08, 0.0], dtype=np.float64)
        self.hose_engage_offset_local = np.array([0.01, 0.0, 0.0], dtype=np.float64)
        self.socket_pos_np = np.array([float(scene_builder.socket_pos[i]) for i in range(3)], dtype=np.float64)
        self.socket_rot_np = np.array([float(scene_builder.socket_rot[i]) for i in range(4)], dtype=np.float64)
        self.insertion_dir_np = _np_quat_rotate(self.socket_rot_np, np.array([0.0, 0.0, 1.0]))
        self.insertion_dir_np /= max(np.linalg.norm(self.insertion_dir_np), 1.0e-12)
        self.desired_tip_quat_np = _np_quat_multiply(
            self.socket_rot_np,
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )
        self.clearance_dir_np = -self.insertion_dir_np.copy()
        self.clearance_dir_np[2] = 0.0
        clearance_norm = np.linalg.norm(self.clearance_dir_np)
        if clearance_norm < 1.0e-8:
            self.clearance_dir_np = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        else:
            self.clearance_dir_np /= clearance_norm
        self.grasp_orientation_offset = _np_quat_multiply(
            _np_quat_from_axis_angle([1.0, 0.0, 0.0], np.pi / 2.0),
            _np_quat_from_axis_angle([0.0, 0.0, 1.0], -np.pi / 2.0),
        )
        self.grasp_shift = 0.004
        self.grasp_local_offset = np.array([0.0, -args_cli.hose_radius + 0.002, 0.0], dtype=np.float64)
        self.right_open_targets, self.right_closed_targets = self._gripper_target_pairs(
            scene_builder.right_gripper_dofs
        )
        self.left_open_targets, _ = self._gripper_target_pairs(scene_builder.left_gripper_dofs)

        self.ee_indices = {
            "right": _find_label_index(self.model.body_label, RIGHT_EE),
            "left": _find_label_index(self.model.body_label, LEFT_EE),
            "torso": _find_label_index(self.model.body_label, TORSO),
        }
        self.right_finger_body_ids = [
            body_id
            for body_id in (
                _maybe_find_label_index(self.model.body_label, "right_gripper_leftfinger"),
                _maybe_find_label_index(self.model.body_label, "right_gripper_rightfinger"),
            )
            if body_id is not None
        ]
        self._setup_ik()
        self._seed_control_targets()
        self._enter_phase(0, 0.0)

    def _gripper_target_pairs(self, dofs: list[int]) -> tuple[list[float], list[float]]:
        if len(dofs) < 2:
            return [0.0] * len(dofs), [0.0] * len(dofs)
        lower = self.model.joint_limit_lower.numpy()[dofs]
        upper = self.model.joint_limit_upper.numpy()[dofs]
        # RBY1 finger coordinates use opposite signs for left/right fingers.
        open_targets = [-0.04, 0.04]
        eps = 1.0e-4
        closed_targets = [float(upper[0] - eps), float(lower[1] + eps)]
        return open_targets, closed_targets

    def _setup_ik(self) -> None:
        body_q_np = self.state.body_q.numpy()
        weights = {"right": 1.0, "left": 1.0, "torso": 50.0}
        self.pos_objs = []
        self.rot_objs = []
        self.nominal_tfs = {}
        for name in ("right", "left", "torso"):
            body_id = self.ee_indices[name]
            tf = wp.transform(*body_q_np[body_id])
            self.nominal_tfs[name] = tf
            self.pos_objs.append(
                ik.IKObjectivePosition(
                    link_index=body_id,
                    link_offset=wp.vec3(0.0, 0.0, 0.0),
                    target_positions=wp.array([wp.transform_get_translation(tf)], dtype=wp.vec3),
                    weight=weights[name],
                )
            )
            quat = wp.transform_get_rotation(tf)
            self.rot_objs.append(
                ik.IKObjectiveRotation(
                    link_index=body_id,
                    link_offset_rotation=wp.quat_identity(),
                    target_rotations=wp.array([_quat_to_vec4(quat)], dtype=wp.vec4),
                    weight=weights[name],
                )
            )

        joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.single_robot_model.joint_limit_lower,
            joint_limit_upper=self.single_robot_model.joint_limit_upper,
            weight=10.0,
        )
        initial_joint_q = self.single_robot_model.joint_q.numpy().astype(np.float32, copy=False)
        self.ik_joint_q = wp.array(
            initial_joint_q,
            shape=(1, self.single_robot_model.joint_coord_count),
            dtype=wp.float32,
            device=self.model.device,
        )
        self.ik_solver = ik.IKSolver(
            model=self.single_robot_model,
            n_problems=1,
            objectives=[*self.pos_objs, *self.rot_objs, joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def _seed_control_targets(self) -> None:
        initial = self.single_robot_model.joint_q.numpy().astype(np.float32, copy=False)
        target_np = self.control.joint_target_pos.numpy().astype(np.float32, copy=True)
        for joint_coord_ids in self.scene_builder.robot_joint_coord_ids_by_env:
            target_np[joint_coord_ids] = initial
        wp.copy(self.control.joint_target_pos, wp.array(target_np, dtype=wp.float32, device=self.model.device))
        self.last_control_q = initial.astype(np.float32, copy=True)

    def update(self, sim_time: float) -> str:
        self.state = NewtonManager.get_state_0()
        while self._should_advance_phase(sim_time):
            self._enter_phase(min(self.phase_index + 1, len(self._PHASES) - 1), sim_time)
        target_pos, target_quat, gripper_value, task = self._target_for_phase(sim_time)
        target_pos, target_quat = self._filter_ik_target(target_pos, target_quat)
        self._set_ik_target(0, target_pos, target_quat)
        self._hold_nominal_target(1, "left")
        self._hold_nominal_target(2, "torso")
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        self._write_joint_targets(gripper_value)
        self.last_task = task
        return task

    def _enter_phase(self, phase_index: int, sim_time: float) -> None:
        self.phase_index = phase_index
        self.phase_start_time = sim_time
        self.phase_start_ee_pos, self.phase_start_ee_quat = self._ee_pose_np()
        self.phase_start_plug_pos, self.phase_start_plug_quat = self._plug_pose_np()
        phase = self._PHASES[self.phase_index][0]
        if phase in ("approach_hose", "reapproach"):
            self._reset_live_hose_target()
            self._update_grasp_local_offset()
        if phase in ("engage_hose", "reengage"):
            self._update_grasp_local_offset()
        if phase in ("align_axes", "verify_align", "insert_hose", "release_hose") or phase in ("approach_socket",):
            self.phase_target_quat = _np_quat_multiply(self.socket_rot_np, self.grasp_orientation_offset)
        else:
            # Hose grasp phases override this with a filtered live cable orientation.
            self.phase_target_quat = self.phase_start_ee_quat.copy()

    def _should_advance_phase(self, sim_time: float) -> bool:
        phase, duration = self._PHASES[self.phase_index]
        if phase == "done":
            return False
        elapsed = sim_time - self.phase_start_time
        if phase == "verify_align" and elapsed > 0.5:
            if self._verify_alignment_ok():
                self.verify_align_retries = 0
                return True
            if elapsed >= duration:
                self.verify_align_retries += 1
                self._enter_phase(self._phase_index("align_axes"), sim_time)
                return False
        return elapsed >= duration

    def _target_for_phase(self, sim_time: float):
        phase, duration = self._PHASES[self.phase_index]
        elapsed = sim_time - self.phase_start_time
        alpha = _smoothstep(elapsed / max(duration, 1.0e-6))

        if phase == "approach_hose":
            hose_pos, hose_quat = self._hose_target_pose("approach", live=True)
            target_pos = self._lerp(
                self.phase_start_ee_pos,
                hose_pos,
                alpha,
            )
            target_quat = _np_quat_slerp(self.phase_start_ee_quat, hose_quat, alpha)
            grip = 0.0
        elif phase == "engage_hose":
            hose_pos, hose_quat = self._hose_target_pose("engage", live=True)
            target_pos = self._lerp(
                self.phase_start_ee_pos,
                hose_pos,
                alpha,
            )
            target_pos = self._apply_gripper_centering(target_pos)
            target_quat = hose_quat
            grip = 0.0
        elif phase == "grasp_hose":
            target_pos, target_quat = self._hose_target_pose("engage", live=True)
            target_pos = self._apply_gripper_centering(target_pos)
            grip = _smoothstep(alpha / 0.35)
        elif phase == "hold_grasp":
            target_pos, target_quat = self._hose_target_pose("engage", live=True)
            target_pos = self._apply_gripper_centering(target_pos)
            grip = 1.0
        elif phase == "retract":
            target_pos = self.phase_start_ee_pos + alpha * self._transfer_retract_offset()
            target_quat = self.phase_start_ee_quat
            grip = 1.0
        elif phase == "settle":
            target_pos = self.phase_start_ee_pos
            target_quat = self.phase_start_ee_quat
            grip = 1.0
        elif phase == "approach_socket":
            target_pos = self._lerp(self.phase_start_ee_pos, self._socket_approach_pos(), alpha)
            target_quat = _np_quat_slerp(self.phase_start_ee_quat, self.phase_target_quat, alpha)
            grip = 1.0
        elif phase == "align_axes":
            ee_pos, ee_quat = self._ee_target_for_desired_tip(self._socket_start_pos(), self.desired_tip_quat_np)
            target_pos = self._lerp(self.phase_start_ee_pos, ee_pos, alpha)
            target_quat = _np_quat_slerp(self.phase_start_ee_quat, ee_quat, alpha)
            grip = 1.0
        elif phase == "verify_align":
            target_pos, target_quat = self._verify_alignment_target()
            grip = 1.0
        elif phase == "insert_hose":
            target_pos, target_quat = self._insert_target(alpha)
            grip = 1.0
        elif phase == "release_hose":
            target_pos = self.phase_start_ee_pos
            target_quat = self.phase_start_ee_quat
            grip = 1.0 - alpha
        elif phase == "withdraw":
            target_pos = self._lerp(
                self.phase_start_ee_pos,
                self.phase_start_ee_pos + 0.055 * self.clearance_dir_np + np.array([0.0, 0.0, 0.035]),
                alpha,
            )
            target_quat = self.phase_start_ee_quat
            grip = 0.0
        elif phase == "wait_after_withdraw":
            target_pos = self.phase_start_ee_pos
            target_quat = self.phase_start_ee_quat
            grip = 0.0
        elif phase == "reapproach":
            hose_pos, hose_quat = self._hose_target_pose("approach", live=True)
            target_pos = self._lerp(self.phase_start_ee_pos, hose_pos, alpha)
            target_quat = _np_quat_slerp(self.phase_start_ee_quat, hose_quat, alpha)
            grip = 0.0
        elif phase == "reengage":
            hose_pos, hose_quat = self._hose_target_pose("engage", live=True)
            target_pos = self._lerp(self.phase_start_ee_pos, hose_pos, alpha)
            target_pos = self._apply_gripper_centering(target_pos)
            target_quat = hose_quat
            grip = 0.0
        elif phase == "regrasp":
            target_pos, target_quat = self._hose_target_pose("engage", live=True)
            target_pos = self._apply_gripper_centering(target_pos)
            grip = _smoothstep(alpha / 0.35)
        elif phase == "pull":
            pull_vec = -self.pull_distance * self.insertion_dir_np + np.array([-0.03, 0.0, 0.0])
            target_pos = self.phase_start_ee_pos + alpha * pull_vec
            target_quat = self.phase_start_ee_quat
            grip = 1.0
        elif phase == "final_release":
            target_pos = self.phase_start_ee_pos + alpha * np.array([-0.08, 0.0, 0.04])
            target_quat = self.phase_start_ee_quat
            grip = 1.0 - alpha
        else:
            target_pos = self.phase_start_ee_pos
            target_quat = self.phase_start_ee_quat
            grip = 0.0

        return self._vec3(target_pos), self._quat(target_quat), grip, phase

    def _phase_index(self, name: str) -> int:
        for index, (phase, _duration) in enumerate(self._PHASES):
            if phase == name:
                return index
        raise ValueError(name)

    def _filter_ik_target(self, target_pos, target_quat):
        target_pos_np = np.array([float(target_pos[i]) for i in range(3)], dtype=np.float64)
        target_quat_np = np.array([float(target_quat[i]) for i in range(4)], dtype=np.float64)
        if self.last_target_pos is None:
            self.last_target_pos = target_pos_np
            self.last_target_quat = target_quat_np
        else:
            delta = target_pos_np - self.last_target_pos
            dist = np.linalg.norm(delta)
            if dist > self.max_target_step:
                target_pos_np = self.last_target_pos + delta * (self.max_target_step / dist)
            target_quat_np = _np_quat_slerp(self.last_target_quat, target_quat_np, 0.25)
            self.last_target_pos = target_pos_np
            self.last_target_quat = target_quat_np
        return self._vec3(target_pos_np), self._quat(target_quat_np)

    def _plug_pose_np(self) -> tuple[np.ndarray, np.ndarray]:
        return self._body_pose_np(self.scene_builder.plug_body_id)

    def _tip_pose_np(self) -> tuple[np.ndarray, np.ndarray]:
        return self._body_pose_np(self.scene_builder.tip_body_id)

    def _ee_pose_np(self) -> tuple[np.ndarray, np.ndarray]:
        return self._body_pose_np(self.ee_indices["right"])

    def _body_pose_np(self, body_id: int) -> tuple[np.ndarray, np.ndarray]:
        body_q = self.state.body_q.numpy()[body_id]
        return body_q[:3].astype(np.float64), body_q[3:].astype(np.float64)

    def _reset_live_hose_target(self) -> None:
        plug_pos, plug_quat = self._plug_pose_np()
        self.live_hose_target_pos = plug_pos.copy()
        self.live_hose_target_quat = plug_quat.copy()

    def _tracked_plug_pose(self, live: bool) -> tuple[np.ndarray, np.ndarray]:
        if not live:
            return self.phase_start_plug_pos, self.phase_start_plug_quat
        plug_pos, plug_quat = self._plug_pose_np()
        if self.live_hose_target_pos is None or self.live_hose_target_quat is None:
            self.live_hose_target_pos = plug_pos.copy()
            self.live_hose_target_quat = plug_quat.copy()
        else:
            alpha = self.live_hose_target_alpha
            self.live_hose_target_pos = (1.0 - alpha) * self.live_hose_target_pos + alpha * plug_pos
            self.live_hose_target_quat = _np_quat_slerp(self.live_hose_target_quat, plug_quat, alpha)
        return self.live_hose_target_pos.copy(), self.live_hose_target_quat.copy()

    def _grasp_quat(self, plug_quat: np.ndarray) -> np.ndarray:
        return _np_quat_multiply(plug_quat, self.grasp_orientation_offset)

    def _update_grasp_local_offset(self) -> None:
        body_ids = self.scene_builder.primary_cable_body_ids
        if not body_ids:
            return
        plug_pos, plug_quat = self._plug_pose_np()
        body_q = self.state.body_q.numpy()
        nearest_pos = None
        nearest_dist = float("inf")
        for body_id in body_ids:
            cable_pos = body_q[body_id, :3].astype(np.float64)
            dist = float(np.linalg.norm(cable_pos - plug_pos))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_pos = cable_pos
        if nearest_pos is None:
            return
        toward_cable_world = nearest_pos - plug_pos
        norm = np.linalg.norm(toward_cable_world)
        if norm <= 1.0e-8:
            return
        toward_cable_world /= norm
        toward_cable_local = _np_quat_rotate(_np_quat_inverse(plug_quat), toward_cable_world)
        self.grasp_local_offset = np.array(
            [
                toward_cable_local[0] * self.grasp_shift,
                -args_cli.hose_radius + 0.002 + toward_cable_local[1] * self.grasp_shift,
                toward_cable_local[2] * self.grasp_shift,
            ],
            dtype=np.float64,
        )

    def _hose_target_pose(self, mode: str, live: bool) -> tuple[np.ndarray, np.ndarray]:
        if live:
            self._update_grasp_local_offset()
        plug_pos, plug_quat = self._tracked_plug_pose(live)
        grasp_world = _np_quat_rotate(plug_quat, self._grasp_local_offset())
        if mode == "approach":
            extra_world = _np_quat_rotate(plug_quat, self.hose_approach_offset_local)
        elif mode == "engage":
            extra_world = _np_quat_rotate(plug_quat, self.hose_engage_offset_local)
        else:
            raise ValueError(mode)
        return plug_pos + grasp_world + extra_world, self._grasp_quat(plug_quat)

    def _apply_gripper_centering(self, target_pos: np.ndarray) -> np.ndarray:
        if len(self.right_finger_body_ids) != 2:
            return target_pos
        plug_pos, plug_quat = self._plug_pose_np()
        finger_0_pos, _ = self._body_pose_np(self.right_finger_body_ids[0])
        finger_1_pos, _ = self._body_pose_np(self.right_finger_body_ids[1])

        closing_axis = finger_1_pos - finger_0_pos
        closing_axis_norm = np.linalg.norm(closing_axis)
        if closing_axis_norm < 1.0e-8:
            return target_pos
        closing_axis /= closing_axis_norm

        finger_mid = 0.5 * (finger_0_pos + finger_1_pos)
        hose_axis = _np_quat_rotate(plug_quat, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        hose_axis /= max(np.linalg.norm(hose_axis), 1.0e-12)

        delta = -self.gripper_centering_k * np.dot(plug_pos - finger_mid, closing_axis) * closing_axis
        mid_to_hose = finger_mid - plug_pos
        radial_error = mid_to_hose - np.dot(mid_to_hose, hose_axis) * hose_axis
        delta -= self.gripper_axis_centering_k * radial_error
        delta -= np.dot(delta, hose_axis) * hose_axis

        delta_norm = np.linalg.norm(delta)
        if delta_norm > self.gripper_centering_max_step:
            delta *= self.gripper_centering_max_step / delta_norm
        return target_pos + delta

    def _grasp_local_offset(self) -> np.ndarray:
        return self.grasp_local_offset.copy()

    def _transfer_retract_offset(self) -> np.ndarray:
        return self.transfer_retract_standoff * self.clearance_dir_np + np.array(
            [0.0, 0.0, self.transfer_retract_lift], dtype=np.float64
        )

    def _socket_approach_pos(self) -> np.ndarray:
        return (
            self.socket_pos_np
            + self.socket_approach_standoff * self.clearance_dir_np
            + np.array([0.0, 0.0, self.socket_approach_lift], dtype=np.float64)
        )

    def _socket_start_pos(self) -> np.ndarray:
        return self.socket_pos_np + self.insert_start_depth * self.insertion_dir_np

    def _tip_axis_np(self) -> np.ndarray:
        _, tip_quat = self._tip_pose_np()
        tip_axis = _np_quat_rotate(tip_quat, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        return tip_axis / max(np.linalg.norm(tip_axis), 1.0e-12)

    def _alignment_errors(self) -> tuple[np.ndarray, float, float]:
        tip_pos, _ = self._tip_pose_np()
        delta = tip_pos - self._socket_start_pos()
        lateral = delta - np.dot(delta, self.insertion_dir_np) * self.insertion_dir_np
        axis_cosine = float(np.dot(self._tip_axis_np(), self.insertion_dir_np))
        return lateral, float(np.linalg.norm(lateral)), axis_cosine

    def _ee_target_for_desired_tip(
        self,
        desired_tip_pos: np.ndarray,
        desired_tip_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        tip_pos, tip_quat = self._tip_pose_np()
        ee_pos, ee_quat = self._ee_pose_np()
        tip_to_ee_pos, tip_to_ee_quat = _np_relative_transform(tip_pos, tip_quat, ee_pos, ee_quat)
        return _np_transform_point_quat(desired_tip_pos, desired_tip_quat, tip_to_ee_pos, tip_to_ee_quat)

    def _verify_alignment_target(self) -> tuple[np.ndarray, np.ndarray]:
        return self._ee_target_for_desired_tip(self._socket_start_pos(), self.desired_tip_quat_np)

    def _verify_alignment_ok(self) -> bool:
        _, lateral_error, axis_cosine = self._alignment_errors()
        return lateral_error < self.align_lateral_threshold and axis_cosine < self.align_axis_cosine_threshold

    def _insert_target(self, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        depth = self.insert_start_depth + alpha * (self.insert_final_depth - self.insert_start_depth)
        desired_tip_pos = self.socket_pos_np + depth * self.insertion_dir_np
        return self._ee_target_for_desired_tip(desired_tip_pos, self.desired_tip_quat_np)

    @staticmethod
    def _lerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
        return (1.0 - alpha) * a + alpha * b

    @staticmethod
    def _vec3(v: np.ndarray):
        return wp.vec3(float(v[0]), float(v[1]), float(v[2]))

    @staticmethod
    def _quat(q: np.ndarray):
        q = np.asarray(q, dtype=np.float64)
        q = q / max(np.linalg.norm(q), 1.0e-12)
        return wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    def _set_ik_target(self, objective_idx: int, pos, quat) -> None:
        self.pos_objs[objective_idx].set_target_position(0, pos)
        self.rot_objs[objective_idx].set_target_rotation(0, _quat_to_vec4(quat))

    def _hold_nominal_target(self, objective_idx: int, name: str) -> None:
        tf = self.nominal_tfs[name]
        self._set_ik_target(objective_idx, wp.transform_get_translation(tf), wp.transform_get_rotation(tf))

    def _write_joint_targets(self, gripper_alpha: float) -> None:
        q = self.ik_joint_q.numpy().reshape(-1).astype(np.float32, copy=True)
        self._set_gripper_targets(q, gripper_alpha)
        max_step = np.full_like(q, self.max_joint_step)
        gripper_dofs = self.scene_builder.gripper_dofs
        if gripper_dofs:
            max_step[gripper_dofs] = self.max_gripper_joint_step
        q = self.last_control_q + np.clip(q - self.last_control_q, -max_step, max_step)
        self.last_control_q = q.astype(np.float32, copy=True)
        limited_ik_q = wp.array(
            q,
            shape=(1, self.single_robot_model.joint_coord_count),
            dtype=wp.float32,
            device=self.model.device,
        )
        wp.copy(self.ik_joint_q, limited_ik_q)
        target_np = self.control.joint_target_pos.numpy().astype(np.float32, copy=True)
        for joint_coord_ids in self.scene_builder.robot_joint_coord_ids_by_env:
            target_np[joint_coord_ids] = q
        wp.copy(self.control.joint_target_pos, wp.array(target_np, dtype=wp.float32, device=self.model.device))

    def _set_gripper_targets(self, q: np.ndarray, right_alpha: float) -> None:
        right_alpha = max(0.0, min(1.0, right_alpha))
        for idx, dof in enumerate(self.scene_builder.right_gripper_dofs[:2]):
            q[dof] = (1.0 - right_alpha) * self.right_open_targets[idx] + right_alpha * self.right_closed_targets[idx]
        for idx, dof in enumerate(self.scene_builder.left_gripper_dofs[:2]):
            q[dof] = self.left_open_targets[idx]


def setup_kit_scene(sim, scene_builder: WaterhoseSceneBuilder) -> None:
    """Create simple Kit-side reference geometry when Kit visualization is active."""
    if "kit" not in sim.resolve_visualizer_types():
        return
    for env_id, origin in enumerate(scene_builder.env_origins):
        if scene_builder.num_envs == 1:
            scene_prim_path = "/World/Cable008Scene"
        else:
            scene_prim_path = f"/World/Env_{env_id}/Cable008Scene"
        if sim_utils.get_current_stage().GetPrimAtPath(scene_prim_path).IsValid():
            continue
        scene_cfg = sim_utils.UsdFileCfg(usd_path=str(scene_builder.scene_usd))
        scene_pos = wp.transform_get_translation(scene_builder.asset_xform)
        scene_rot = wp.transform_get_rotation(scene_builder.asset_xform)
        scene_cfg.func(
            scene_prim_path,
            scene_cfg,
            translation=(
                float(scene_pos[0] + origin[0]),
                float(scene_pos[1] + origin[1]),
                float(scene_pos[2] + origin[2]),
            ),
            orientation=(float(scene_rot[3]), float(scene_rot[0]), float(scene_rot[1]), float(scene_rot[2])),
        )
    if not sim_utils.get_current_stage().GetPrimAtPath("/World/DomeLight").IsValid():
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/DomeLight", light_cfg)


def configure_newton_viewer(sim) -> None:
    """Enable useful interaction and contact display in the Newton viewer."""
    for visualizer in sim.visualizers:
        viewer = getattr(visualizer, "_viewer", None)
        if viewer is None:
            continue
        if hasattr(viewer, "show_contacts"):
            viewer.show_contacts = True
        if hasattr(viewer, "picking_enabled"):
            viewer.picking_enabled = True
        if hasattr(viewer, "set_camera"):
            viewer.set_camera(wp.vec3(-1.2, -2.8, 1.6), pitch=-18.0, yaw=-300.0)


def apply_viewer_forces(sim) -> None:
    for visualizer in sim.visualizers:
        viewer = getattr(visualizer, "_viewer", None)
        if viewer is not None and hasattr(viewer, "apply_forces"):
            viewer.apply_forces(NewtonManager.get_state_0())


def keep_running(sim, step_count: int) -> bool:
    if args_cli.max_steps >= 0 and step_count >= args_cli.max_steps:
        return False
    if not sim.visualizers:
        return True
    return any(not viz.is_closed and viz.is_running() for viz in sim.visualizers)


def log_progress(step_count: int, task: str, scene_builder: WaterhoseSceneBuilder) -> None:
    if args_cli.log_interval <= 0 or step_count % args_cli.log_interval != 0:
        return
    state = NewtonManager.get_state_0()
    body_q = state.body_q.numpy()
    hose_pos = body_q[scene_builder.grasp_body_id, 0:3]
    tip_q = body_q[scene_builder.tip_body_id]
    tip_pos = tip_q[0:3]
    tip_axis = _np_quat_rotate(tip_q[3:], np.array([0.0, 0.0, 1.0], dtype=np.float64))
    socket_pos = np.array([float(scene_builder.socket_pos[i]) for i in range(3)], dtype=np.float64)
    socket_rot = np.array([float(scene_builder.socket_rot[i]) for i in range(4)], dtype=np.float64)
    insertion_dir = _np_quat_rotate(socket_rot, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    insertion_dir /= max(np.linalg.norm(insertion_dir), 1.0e-12)
    tip_delta = tip_pos - (socket_pos + 0.005 * insertion_dir)
    tip_lateral = tip_delta - np.dot(tip_delta, insertion_dir) * insertion_dir
    tip_lateral_mm = 1000.0 * float(np.linalg.norm(tip_lateral))
    tip_axis_dot = float(np.dot(tip_axis / max(np.linalg.norm(tip_axis), 1.0e-12), insertion_dir))
    right_ee_id = _find_label_index(NewtonManager.get_model().body_label, RIGHT_EE)
    right_ee_pos = body_q[right_ee_id, 0:3]
    ee_to_hose = float(np.linalg.norm(right_ee_pos - hose_pos))
    wrenches = NewtonCoupledManager.get_proxy_body_wrenches(ROBOT_ENTRY, HOSE_ENTRY)
    wrench_norm = 0.0
    if wrenches is not None:
        wrench_np = wrenches.numpy()
        proxy_body_ids = [body_id for body_id in scene_builder.proxy_body_ids if body_id < wrench_np.shape[0]]
        if proxy_body_ids:
            wrench_norm = float(np.linalg.norm(wrench_np[proxy_body_ids, 0:3]))
    print(
        "[INFO]: "
        f"step={step_count:05d} task={task:<15} "
        f"hose_grasp=({hose_pos[0]:.3f}, {hose_pos[1]:.3f}, {hose_pos[2]:.3f}) "
        f"ee_dist={ee_to_hose:.3f}m "
        f"tip_lat={tip_lateral_mm:.2f}mm tip_dot={tip_axis_dot:.3f} "
        f"|F_proxy|={wrench_norm:.2f}N",
        flush=True,
    )


def print_robot_pose_summary(tag: str) -> None:
    model = NewtonManager.get_model()
    body_q = NewtonManager.get_state_0().body_q.numpy()
    body_names = [
        "torso_hip_yaw",
        "right_gripper_base",
        "right_gripper_leftfinger",
        "right_gripper_rightfinger",
        RIGHT_EE,
        "left_gripper_base",
        "left_gripper_leftfinger",
        "left_gripper_rightfinger",
        LEFT_EE,
    ]
    print(f"[ROBOT_POSE:{tag}] body_count={model.body_count}", flush=True)
    for body_name in body_names:
        body_id = _find_label_index(model.body_label, body_name)
        q = body_q[body_id]
        print(
            f"[ROBOT_POSE:{tag}] {body_name} id={body_id} "
            f"pos=({float(q[0]): .6f}, {float(q[1]): .6f}, {float(q[2]): .6f}) "
            f"quat_xyzw=({float(q[3]): .6f}, {float(q[4]): .6f}, {float(q[5]): .6f}, {float(q[6]): .6f})",
            flush=True,
        )


def run_simulator(sim, controller: WaterhoseIKController, scene_builder: WaterhoseSceneBuilder) -> None:
    step_count = 0
    settle_printed = False
    settle_steps = None
    if args_cli.print_cable_poses and args_cli.cable_pose_settle_seconds is not None:
        settle_steps = max(0, int(round(args_cli.cable_pose_settle_seconds * args_cli.fps)))
    while keep_running(sim, step_count):
        apply_viewer_forces(sim)
        task = controller.update(step_count / args_cli.fps)
        sim.step(render=False)
        if settle_steps is not None and not settle_printed and step_count + 1 >= settle_steps:
            scene_builder._print_runtime_cable_pose_summary(f"manager_after_{args_cli.cable_pose_settle_seconds:g}s")
            settle_printed = True
        log_progress(step_count, task, scene_builder)
        if sim.is_rendering:
            sim.render()
        step_count += 1


def initialize_waterhose_runtime(sim, scene_builder: WaterhoseSceneBuilder, builder) -> None:
    """Install the Newton builder and run waterhose-specific post-reset hooks."""
    NewtonManager.set_builder(builder)
    sim.reset()
    scene_builder.configure_runtime_vbd_solver()
    scene_builder.apply_runtime_cable_asset_xform()
    if getattr(args_cli, "print_robot_poses", False):
        print_robot_pose_summary("manager_after_reset")
    if sim.is_rendering:
        NewtonManager.sync_transforms_to_usd()
    configure_newton_viewer(sim)


def body_pose_np(body_id: int):
    """Return a Newton body pose as numpy ``(position, quaternion_xyzw)``."""
    state = NewtonManager.get_state_0()
    q = state.body_q.numpy()[body_id]
    return q[:3].copy(), q[3:].copy()
