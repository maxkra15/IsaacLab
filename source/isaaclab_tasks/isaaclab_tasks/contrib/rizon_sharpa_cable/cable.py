# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime cable object and scoped Newton builder extension."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import newton
import newton.usd
import numpy as np
import torch
import warp as wp
from isaaclab_newton.assets import CableObject

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from .cable_cfg import RizonSharpaCableObjectCfg

_RJ45_ASSET_SHA256 = "50c95bcfb63544777f9148d548aac6f16b62f65cacbaaa9316453d579de4b4fa"
_RJ45_ASSET_PATH = Path(newton.__file__).resolve().parent / "examples" / "assets" / "rj45_plug.usd"
_CONNECTOR_SDF_GAP_M = 0.002
_CONNECTOR_SDF_NARROW_BAND_M = (-2.0 * _CONNECTOR_SDF_GAP_M, 2.0 * _CONNECTOR_SDF_GAP_M)
_CONNECTOR_SDF_MAX_RESOLUTION = 128


@dataclass(frozen=True)
class ConnectorRenderPart:
    """One connector mesh expressed in the native head-body frame."""

    name: str
    points: tuple[tuple[float, float, float], ...]
    face_vertex_counts: tuple[int, ...]
    face_vertex_indices: tuple[int, ...]


@dataclass(frozen=True)
class _ConnectorGeometry:
    render_parts: tuple[ConnectorRenderPart, ConnectorRenderPart]
    meshes: tuple[newton.Mesh, newton.Mesh]
    socket_render_part: ConnectorRenderPart
    socket_mesh: newton.Mesh
    collision_center: tuple[float, float, float]
    collision_half_extents: tuple[float, float, float]


def _rotate_source_to_hanging(point: tuple[float, float, float]) -> tuple[float, float, float]:
    """Map source -Y cable travel to world-local +Z cable travel."""
    x, y, z = point
    return (x, z, -y)


@lru_cache(maxsize=4)
def _connector_geometry(connector_rigid_span_m: float) -> _ConnectorGeometry:
    """Load the canonical plug once and express it in the hanging head frame."""
    from pxr import Gf, Usd, UsdGeom  # noqa: PLC0415

    with _RJ45_ASSET_PATH.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != _RJ45_ASSET_SHA256:
        raise RuntimeError(f"RJ45 asset SHA-256 mismatch: expected {_RJ45_ASSET_SHA256}, got {digest}.")

    stage = Usd.Stage.Open(str(_RJ45_ASSET_PATH))
    if stage is None:
        raise RuntimeError(f"Could not open the canonical RJ45 asset: {_RJ45_ASSET_PATH}")
    curve_prim = stage.GetPrimAtPath("/World/CableCurve")
    if not curve_prim.IsA(UsdGeom.BasisCurves):
        raise RuntimeError("Canonical RJ45 asset has no /World/CableCurve.")
    curve_xform = UsdGeom.Xformable(curve_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    source_start = curve_xform.Transform(UsdGeom.BasisCurves(curve_prim).GetPointsAttr().Get()[0])
    plug_prim = stage.GetPrimAtPath("/World/Plug")
    if not plug_prim.IsA(UsdGeom.Mesh):
        raise RuntimeError("Canonical RJ45 asset has no mesh /World/Plug.")
    # Preserve the exact source cable-to-plug datum used by the proven Franka
    # RJ45 assembly.  The first centerline point intentionally sits inside the
    # rear housing: the short embedded span is the connector's strain relief.
    # Moving the datum to the rear-most mesh face put the whole connector mass
    # below segment zero and introduced a large artificial bending lever arm.
    source_attachment = tuple(float(value) for value in source_start)
    head_center = (0.0, 0.0, 0.5 * float(connector_rigid_span_m))

    parts: list[ConnectorRenderPart] = []
    meshes: list[newton.Mesh] = []
    plug_points: tuple[tuple[float, float, float], ...] | None = None
    for name in ("Plug", "Latch"):
        prim = stage.GetPrimAtPath(f"/World/{name}")
        if not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"Canonical RJ45 asset has no mesh /World/{name}.")
        usd_mesh = UsdGeom.Mesh(prim)
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        points = []
        for source_point in usd_mesh.GetPointsAttr().Get():
            point_w = transform.Transform(source_point)
            relative = (
                float(point_w[0] - source_attachment[0]),
                float(point_w[1] - source_attachment[1]),
                float(point_w[2] - source_attachment[2]),
            )
            hanging = _rotate_source_to_hanging(relative)
            points.append(tuple(value - center for value, center in zip(hanging, head_center, strict=True)))
        points_tuple = tuple(points)
        counts = tuple(int(value) for value in usd_mesh.GetFaceVertexCountsAttr().Get())
        indices = tuple(int(value) for value in usd_mesh.GetFaceVertexIndicesAttr().Get())
        parts.append(ConnectorRenderPart(name.lower(), points_tuple, counts, indices))

        # Keep the original polygon topology above for USD rendering, but use
        # Newton's deterministic fan triangulation for collision/SDF meshes.
        # Passing the polygon corner list directly to ``newton.Mesh`` happens
        # to render, but SDF construction requires a true triangle index list.
        triangulated = newton.usd.get_mesh(prim, compute_inertia=False)
        collision_points = []
        for source_point in np.asarray(triangulated.vertices):
            point_w = transform.Transform(Gf.Vec3d(*(float(value) for value in source_point)))
            relative = (
                float(point_w[0] - source_attachment[0]),
                float(point_w[1] - source_attachment[1]),
                float(point_w[2] - source_attachment[2]),
            )
            hanging = _rotate_source_to_hanging(relative)
            collision_points.append(tuple(value - center for value, center in zip(hanging, head_center, strict=True)))
        meshes.append(
            newton.Mesh(
                np.asarray(collision_points, dtype=np.float32),
                np.asarray(triangulated.indices, dtype=np.int32),
            )
        )
        if name == "Plug":
            plug_points = points_tuple

    socket_prim = stage.GetPrimAtPath("/World/Socket")
    if not socket_prim.IsA(UsdGeom.Mesh):
        raise RuntimeError("Canonical RJ45 asset has no mesh /World/Socket.")
    socket_usd = UsdGeom.Mesh(socket_prim)
    socket_points = tuple(tuple(float(value) for value in point) for point in socket_usd.GetPointsAttr().Get())
    socket_counts = tuple(int(value) for value in socket_usd.GetFaceVertexCountsAttr().Get())
    socket_indices = tuple(int(value) for value in socket_usd.GetFaceVertexIndicesAttr().Get())
    socket_part = ConnectorRenderPart("socket", socket_points, socket_counts, socket_indices)
    triangulated_socket = newton.usd.get_mesh(socket_prim, compute_inertia=False)
    socket_mesh = newton.Mesh(
        np.asarray(triangulated_socket.vertices, dtype=np.float32),
        np.asarray(triangulated_socket.indices, dtype=np.int32),
    )

    assert plug_points is not None
    plug_array = np.asarray(plug_points, dtype=np.float64)
    lower = plug_array.min(axis=0)
    upper = plug_array.max(axis=0)
    center = 0.5 * (lower + upper)
    half_extents = 0.5 * (upper - lower)
    return _ConnectorGeometry(
        render_parts=(parts[0], parts[1]),
        meshes=(meshes[0], meshes[1]),
        socket_render_part=socket_part,
        socket_mesh=socket_mesh,
        collision_center=tuple(float(value) for value in center),
        collision_half_extents=tuple(float(value) for value in half_extents),
    )


def connector_render_parts(connector_rigid_span_m: float) -> tuple[ConnectorRenderPart, ConnectorRenderPart]:
    """Return immutable plug and latch meshes in the cable-head frame."""
    return _connector_geometry(float(connector_rigid_span_m)).render_parts


def socket_render_part() -> ConnectorRenderPart:
    """Return the immutable exact socket mesh in its body-local frame."""
    return _connector_geometry(0.01).socket_render_part


class RizonSharpaCableObject(CableObject):
    """Native cable with a convenient connector-pose observation."""

    cfg: RizonSharpaCableObjectCfg

    def get_connector_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the center pose of the cheap connector collider in world coordinates."""
        head_pose = self.data.segment_pose_w.torch[:, 0]
        center = _connector_geometry(float(self.cfg.connector_rigid_span_m)).collision_center
        local_pos = torch.tensor(center, device=head_pose.device, dtype=head_pose.dtype).expand(head_pose.shape[0], -1)
        local_quat = torch.zeros(head_pose.shape[0], 4, device=head_pose.device, dtype=head_pose.dtype)
        local_quat[:, 3] = 1.0
        return math_utils.combine_frame_transforms(
            head_pose[:, :3],
            math_utils.normalize(head_pose[:, 3:7]),
            local_pos,
            local_quat,
        )


class RizonSharpaCableBuilderExtension:
    """Add one rigid compound plug and one fixed tail to a native cable world."""

    def __init__(self, cfg: RizonSharpaCableObjectCfg):
        self.cfg = cfg

    def _expanded_path(self, world_idx: int) -> str:
        return self.cfg.prim_path.replace("env_.*", f"env_{world_idx}").replace("env_[^/]+", f"env_{world_idx}")

    def _curve_group_index(self, builder: newton.ModelBuilder, world_idx: int) -> int:
        expression = self.cfg.prim_path + self.cfg.curve_prim_suffix
        matches = [
            index
            for index, (label, world) in enumerate(zip(builder._cable_label, builder._cable_world, strict=True))
            if int(world) == world_idx and re.fullmatch(expression, str(label))
        ]
        if len(matches) != 1:
            available = tuple(
                (str(label), int(world))
                for label, world in zip(builder._cable_label, builder._cable_world, strict=True)
            )
            raise RuntimeError(
                f"Expected one hanging cable in world {world_idx}, found {len(matches)}; "
                f"pattern: {expression!r}; available native cable groups: {available}."
            )
        return matches[0]

    @staticmethod
    def _build_sdf(mesh: newton.Mesh) -> None:
        """Build the exact narrow-band connector SDF on first use."""
        if mesh.sdf is None:
            mesh.build_sdf(
                max_resolution=_CONNECTOR_SDF_MAX_RESOLUTION,
                narrow_band_range=_CONNECTOR_SDF_NARROW_BAND_M,
                margin=_CONNECTOR_SDF_GAP_M,
            )

    def _add_connector(
        self,
        builder: newton.ModelBuilder,
        world_idx: int,
        body_start: int,
        body_end: int,
    ) -> tuple[int, int]:
        from newton import ShapeFlags

        expanded = self._expanded_path(world_idx)
        head_path = f"{expanded}/connector_head"
        # The connector and the long first cable span are shapes on this same
        # rigid body. There is no connector joint: the first deformable cable
        # joint begins behind the plug's strain relief.
        head_body = body_start
        builder.body_label[head_body] = head_path

        # Kit renders the authored child meshes under this body path. Hide their static import in
        # NewtonGL, then add the same meshes as dynamic, render-only head shapes.
        for shape_id, (label, body, world) in enumerate(
            zip(builder.shape_label, builder.shape_body, builder.shape_world, strict=True)
        ):
            if int(world) == world_idx and int(body) == -1 and str(label).startswith(head_path):
                builder.shape_flags[shape_id] = int(builder.shape_flags[shape_id]) & ~int(ShapeFlags.VISIBLE)

        collision_group = -(world_idx + 1)
        for segment_index, body_id in enumerate(range(body_start, body_end)):
            builder.body_label[body_id] = head_path if segment_index == 0 else f"{expanded}/segment_{segment_index:02d}"
            for shape_id in builder.body_shapes[body_id]:
                builder.shape_collision_group[shape_id] = collision_group
                builder.shape_label[shape_id] = f"{expanded}/geometry/mesh_edge_capsule_{segment_index}"

        geometry = _connector_geometry(float(self.cfg.connector_rigid_span_m))
        colors = ((0.025, 0.19, 0.44), (0.10, 0.34, 0.62))
        for part, mesh, color in zip(geometry.render_parts, geometry.meshes, colors, strict=True):
            builder.add_shape_mesh(
                body=head_body,
                mesh=mesh,
                cfg=newton.ModelBuilder.ShapeConfig(
                    density=0.0,
                    collision_group=collision_group,
                    has_shape_collision=False,
                    has_particle_collision=False,
                ),
                color=color,
                label=f"{head_path}/{part.name}_visual",
            )

        hx, hy, hz = geometry.collision_half_extents
        if min(hx, hy, hz) <= 0.0:
            raise RuntimeError("The RJ45 connector collision bounds are degenerate.")
        self._build_sdf(geometry.meshes[0])
        plug_sdf_shape = builder.add_shape_mesh(
            body=head_body,
            mesh=geometry.meshes[0],
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                ke=1.0e5,
                kd=0.0,
                mu=0.0,
                margin=_CONNECTOR_SDF_GAP_M,
                gap=_CONNECTOR_SDF_GAP_M,
                collision_group=collision_group,
                is_visible=False,
                has_particle_collision=False,
            ),
            label=f"{expanded}/connector_sdf",
        )
        grip_shape = builder.add_shape_box(
            body=head_body,
            xform=wp.transform(geometry.collision_center, wp.quat_identity()),
            hx=hx,
            hy=hy,
            hz=hz,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=float(self.cfg.connector_density_kg_m3),
                ke=1.0e4,
                kd=0.1,
                mu=float(self.cfg.connector_friction),
                margin=float(self.cfg.connector_contact_margin_m),
                gap=float(self.cfg.connector_contact_gap_m),
                collision_group=collision_group,
                is_visible=False,
                has_particle_collision=False,
            ),
            label=f"{expanded}/connector_grip",
        )
        return plug_sdf_shape, grip_shape

    def _add_insertion_target(
        self,
        builder: newton.ModelBuilder,
        world_idx: int,
        env_position: list[float],
        env_rotation: list[float] | tuple[float, float, float, float],
    ) -> int:
        """Add one static exact socket SDF in front of the first GB300."""
        spawn_cfg = self.cfg.spawn
        geometry = _connector_geometry(float(self.cfg.connector_rigid_span_m))
        self._build_sdf(geometry.socket_mesh)
        env_tf = wp.transform(env_position, env_rotation)
        target_tf = wp.transform(
            spawn_cfg.insertion_target_position_e,
            spawn_cfg.insertion_target_rotation_xyzw,
        )
        socket_tf = wp.transform_multiply(env_tf, target_tf)
        env_root = self._expanded_path(world_idx).rsplit("/Cable", maxsplit=1)[0]
        socket_body = builder.add_link(xform=socket_tf, label=f"{env_root}/InsertionTarget")
        socket_shape = builder.add_shape_mesh(
            body=socket_body,
            mesh=geometry.socket_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                ke=1.0e5,
                kd=0.0,
                mu=0.0,
                margin=_CONNECTOR_SDF_GAP_M,
                gap=_CONNECTOR_SDF_GAP_M,
                has_particle_collision=False,
            ),
            color=(0.03, 0.62, 0.90),
            label=f"{env_root}/InsertionTarget/SocketSdf",
        )
        builder.body_mass[socket_body] = 0.0
        builder.body_inv_mass[socket_body] = 0.0
        builder.body_inertia[socket_body] = wp.mat33(0.0)
        builder.body_inv_inertia[socket_body] = wp.mat33(0.0)
        return socket_shape

    def _set_joint_damping(self, builder: newton.ModelBuilder, joint_start: int, joint_end: int) -> None:
        for joint_id in range(joint_start, joint_end):
            linear_dofs, angular_dofs = builder.joint_dof_dim[joint_id]
            if int(linear_dofs) != 2 or int(angular_dofs) != 2:
                raise RuntimeError(f"Cable joint {builder.joint_label[joint_id]!r} is not a 2+2 cable joint.")
            dof_start = int(builder.joint_qd_start[joint_id])
            builder.joint_target_kd[dof_start : dof_start + 4] = [
                float(self.cfg.stretch_shear_damping),
                float(self.cfg.stretch_shear_damping),
                float(self.cfg.bend_twist_damping),
                float(self.cfg.bend_twist_damping),
            ]

    def _add_tail_attachment(
        self,
        builder: newton.ModelBuilder,
        world_idx: int,
        tail_body: int,
        env_position: list[float],
        env_rotation: list[float] | tuple[float, float, float, float],
    ) -> None:
        anchor = sim_utils.find_first_matching_prim(self.cfg.tail_anchor_prim_path)
        if anchor is None:
            raise RuntimeError(f"Could not resolve cable tail anchor {self.cfg.tail_anchor_prim_path!r}.")
        match = re.match(r"(?P<env>.*/env_\d+)", anchor.GetPath().pathString)
        env_prim = anchor.GetStage().GetPrimAtPath(match.group("env")) if match else None
        reference = env_prim if env_prim is not None and env_prim.IsValid() else None
        anchor_xform = wp.transform(*sim_utils.resolve_prim_pose(anchor, ref_prim=reference))
        if reference is not None:
            anchor_xform = wp.transform_multiply(wp.transform(env_position, env_rotation), anchor_xform)
        child_xform = wp.transform_multiply(wp.transform_inverse(builder.body_q[tail_body]), anchor_xform)
        joint = builder.add_joint_fixed(
            parent=-1,
            child=tail_body,
            parent_xform=anchor_xform,
            child_xform=child_xform,
            label=f"{self._expanded_path(world_idx)}/tail_attachment",
            collision_filter_parent=True,
            enabled=True,
            custom_attributes={"vbd:joint_is_hard": 0},
        )
        builder.add_articulation([joint], label=f"{self._expanded_path(world_idx)}/tail_attachment_articulation")

    def add_to_builder(
        self,
        builder: newton.ModelBuilder,
        world_idx: int,
        env_position: list[float],
        env_rotation: list[float] | tuple[float, float, float, float],
    ) -> None:
        """Add connector geometry, damping, and the upper attachment to one world."""
        if not builder.has_custom_attribute("vbd:joint_is_hard"):
            from newton.solvers import SolverVBD

            SolverVBD.register_custom_attributes(builder)
        group = self._curve_group_index(builder, world_idx)
        body_start = int(builder._cable_body_start[group])
        body_end = int(builder._cable_body_end[group])
        joint_start = int(builder._cable_joint_start[group])
        joint_end = int(builder._cable_joint_end[group])
        if body_end <= body_start:
            raise RuntimeError(f"Hanging cable in world {world_idx} has no segment bodies.")
        self._set_joint_damping(builder, joint_start, joint_end)
        _, grip_shape = self._add_connector(builder, world_idx, body_start, body_end)
        socket_shape = self._add_insertion_target(builder, world_idx, env_position, env_rotation)
        # The cheap box exists only for robust hand grasping. Let insertion use
        # the exact plug/socket SDF pair rather than blocking on that box.
        builder.add_shape_collision_filter_pair(grip_shape, socket_shape)
        self._add_tail_attachment(builder, world_idx, body_end - 1, env_position, env_rotation)


__all__ = [
    "ConnectorRenderPart",
    "RizonSharpaCableBuilderExtension",
    "RizonSharpaCableObject",
    "connector_render_parts",
    "socket_render_part",
]
