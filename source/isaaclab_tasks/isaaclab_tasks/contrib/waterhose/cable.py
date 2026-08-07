# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native Newton cable support for the Waterhose connector and tail attachment."""

from __future__ import annotations

import math
import re
from dataclasses import MISSING

import newton
import torch
import warp as wp
from isaaclab_newton.assets import CableObject

import isaaclab.sim as sim_utils
from isaaclab.assets import CableObjectCfg
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.utils import clone
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import combine_frame_transforms, normalize


def _find_cable_curve(root_prim):
    """Return the one linear curve below a Waterhose cable root."""
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    curves = [prim for prim in Usd.PrimRange(root_prim) if prim.IsA(UsdGeom.BasisCurves)]
    if len(curves) != 1:
        raise RuntimeError(
            f"Waterhose cable asset at {root_prim.GetPath()} must contain exactly one BasisCurves prim; "
            f"found {len(curves)}."
        )
    curve = UsdGeom.BasisCurves(curves[0])
    counts = curve.GetCurveVertexCountsAttr().Get()
    points = curve.GetPointsAttr().Get()
    if len(counts) != 1 or int(counts[0]) < 3 or len(points) != int(counts[0]):
        raise RuntimeError(f"Waterhose cable curve at {curves[0].GetPath()} is not one valid open polyline.")
    return curve


def _resample_flexible_tail_by_arc_length(curve) -> None:
    """Redistribute the flexible tail while preserving the connector-bearing head segment."""
    from pxr import Vt  # noqa: PLC0415

    points = curve.GetPointsAttr().Get()
    # Segment 0 spans the rigid cable portion inside the plug. Splitting that deliberately long
    # segment would leave the plug rigidly attached to body 0 while visually overlapping later,
    # independently moving cable bodies. Preserve it and regularize only the flexible tail.
    tail_points = points[1:]
    segment_lengths = [
        (tail_points[index + 1] - tail_points[index]).GetLength() for index in range(len(tail_points) - 1)
    ]
    total_length = sum(segment_lengths)
    if total_length <= 0.0:
        raise RuntimeError(f"Waterhose cable tail at {curve.GetPath()} has zero length.")

    spacing = total_length / (len(tail_points) - 1)
    samples = [points[0], tail_points[0]]
    segment_index = 0
    segment_start_length = 0.0
    for sample_index in range(1, len(tail_points) - 1):
        target_length = sample_index * spacing
        while segment_start_length + segment_lengths[segment_index] < target_length:
            segment_start_length += segment_lengths[segment_index]
            segment_index += 1
        alpha = (target_length - segment_start_length) / segment_lengths[segment_index]
        samples.append(
            tail_points[segment_index] + alpha * (tail_points[segment_index + 1] - tail_points[segment_index])
        )
    samples.append(tail_points[-1])
    curve.GetPointsAttr().Set(Vt.Vec3fArray(samples))


@clone
def spawn_waterhose_cable_from_usd(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Load the legacy visual USD and author the standard cable schema in the stage layer.

    The distributed asset remains untouched.  Upstream Newton imports the resulting
    ``PhysicsCurvesDeformableSimAPI`` directly and Isaac Lab's native Fabric bridge synchronizes
    every cloned curve independently.
    """
    from pxr import Gf  # noqa: PLC0415

    root_prim = spawn_from_usd.__wrapped__(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    stage = root_prim.GetStage()
    curve = _find_cable_curve(root_prim)
    _resample_flexible_tail_by_arc_length(curve)
    curve_path = str(curve.GetPath())
    sim_utils.define_deformable_curve_properties(curve_path, stage=stage)

    # FileCfg binds the material while loading because this curve already has CollisionAPI. Bind
    # once more after adding the deformable schema so the standard cable relationship is explicit.
    if cfg.physics_material is None:
        raise ValueError("Waterhose cable USD spawning requires a CableMaterialCfg.")
    material_path = cfg.physics_material_path
    if not material_path.startswith("/"):
        material_path = f"{prim_path}/{material_path}"
    sim_utils.bind_physics_material(curve_path, material_path, stage=stage)

    # Newton's native cable frames are at segment centers. The visual plug parent in the legacy
    # USD used the old segment-start convention. Move that parent to the first midpoint and apply
    # the opposite local offset to its child, preserving the plug's world pose exactly.
    points = curve.GetPointsAttr().Get()
    p0, p1 = points[0], points[1]
    midpoint = 0.5 * (p0 + p1)
    half_length = 0.5 * (p1 - p0).GetLength()
    plug_parent = stage.GetPrimAtPath(f"{prim_path}/cable_edge_body_0")
    plug_visual = stage.GetPrimAtPath(f"{prim_path}/cable_edge_body_0/connector")
    if not plug_parent.IsValid() or not plug_visual.IsValid():
        raise RuntimeError(f"Waterhose cable connector visuals are missing below {prim_path!r}.")
    plug_parent.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*midpoint))
    connector_translation = plug_visual.GetAttribute("xformOp:translate").Get()
    plug_visual.GetAttribute("xformOp:translate").Set(
        Gf.Vec3d(connector_translation[0], connector_translation[1], connector_translation[2] - half_length)
    )

    # The legacy constant normal was not used by the original cable builder. Let Newton construct
    # its stable roll-free frames, which reproduce the authored segment orientation for this curve.
    curve.GetNormalsAttr().Block()
    return root_prim


@configclass
class WaterhoseCableObjectCfg(CableObjectCfg):
    """Native cable plus one compound connector and one soft fixed tail constraint."""

    class_type: type | str = "{DIR}.cable:WaterhoseCableObject"

    connector_usd_path: str = MISSING
    """USD containing exactly one rigid connector mesh."""

    connector_mass: float = 0.001
    """Connector mass [kg]."""

    connector_local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Connector origin in the legacy cable-head start frame [m]."""

    connector_local_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Connector orientation in the cable-head frame as ``(x, y, z, w)``."""

    connector_shape_label: str = "waterhose_connector"
    """Suffix used to identify the connector collision shape."""

    connector_collision_primitives: tuple[tuple[str, float, float, float], ...] = (
        ("body", 0.007338, 0.0075880055, 0.0004380645),
        ("shoulder", 0.006453, 0.0009638235, -0.0081137645),
        ("nose", 0.005568, 0.0025143230, -0.0115919110),
    )
    """Connector cylinders as ``(name, radius, half-height, center-z)`` in the visual plug frame."""

    connector_ke: float = 1.0e4
    """Connector contact stiffness [N/m]."""

    connector_kd: float = 1.0e-1
    """Connector contact damping [N.s/m]."""

    connector_mu: float = 0.5
    """Connector friction coefficient."""

    connector_margin: float = 0.0
    """Connector contact margin [m]."""

    connector_gap: float = 0.01
    """Connector contact gap [m]."""

    tail_anchor_prim_path: str = MISSING
    """Prim path of the static scene marker that anchors the hose tail."""

    curve_prim_suffix: str = "/curve_0"
    """Path of the standard deformable curve relative to the cable root."""

    stretch_shear_damping: float = 1.0e-2
    """Per-joint axial/shear damping retained from the validated cable [N.s/m]."""

    bend_twist_damping: float = 2.0e-2
    """Per-joint bend/twist damping retained from the validated cable [N.m.s/rad]."""


class WaterhoseCableObject(CableObject):
    """Native Newton cable with a query for the task's compound connector pose."""

    cfg: WaterhoseCableObjectCfg

    def __init__(self, cfg: WaterhoseCableObjectCfg):
        super().__init__(cfg)
        self._builder_extension: WaterhoseCableBuilderExtension | None = None
        self._connector_local_pose: torch.Tensor | None = None

    def bind_builder_extension(self, extension: WaterhoseCableBuilderExtension) -> None:
        """Bind build-time geometry metadata after scoped scene construction."""
        self._builder_extension = extension
        self._connector_local_pose = None

    @property
    def connector_head_body_ids_warp(self) -> wp.array:
        """Live Newton body indices for the connector-bearing cable segments.

        This task-specific view is used by the CUDA-graph scripted controller, which reads the
        connector pose directly from Newton's live state without refreshing the manager data
        buffers or copying body indices through the CPU.
        """
        return self.data._sim_bind_root_body_ids

    @property
    def connector_local_pos_from_head_com(self) -> tuple[float, float, float]:
        """Connector translation from the native cable head's center-of-mass frame."""
        if self._builder_extension is None or self._builder_extension.head_segment_half_length is None:
            raise RuntimeError("Waterhose cable builder extension was not bound during scene construction.")
        local_pos = list(self.cfg.connector_local_pos)
        local_pos[2] -= self._builder_extension.head_segment_half_length
        return tuple(local_pos)

    def get_connector_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return connector positions and orientations in the world frame."""
        if self._builder_extension is None or self._builder_extension.head_segment_half_length is None:
            raise RuntimeError("Waterhose cable builder extension was not bound during scene construction.")

        head_pose = self.data.segment_pose_w.torch[:, 0]
        if (
            self._connector_local_pose is None
            or self._connector_local_pose.device != head_pose.device
            or self._connector_local_pose.dtype != head_pose.dtype
        ):
            self._connector_local_pose = torch.tensor(
                (*self.connector_local_pos_from_head_com, *self.cfg.connector_local_quat),
                device=head_pose.device,
                dtype=head_pose.dtype,
            ).unsqueeze(0)
        local_pose = self._connector_local_pose.expand(head_pose.shape[0], -1)
        return combine_frame_transforms(
            head_pose[:, :3],
            normalize(head_pose[:, 3:7]),
            local_pose[:, :3],
            normalize(local_pose[:, 3:7]),
        )


class WaterhoseCableBuilderExtension:
    """Scoped per-world additions layered on Isaac Lab's native cable importer."""

    def __init__(self, cfg: WaterhoseCableObjectCfg):
        self.cfg = cfg
        self.head_segment_half_length: float | None = None
        self._connector_geometry = None

    def _expanded_asset_path(self, world_idx: int) -> str:
        return self.cfg.prim_path.replace("env_.*", f"env_{world_idx}")

    def _curve_group_index(self, builder: newton.ModelBuilder, world_idx: int) -> int:
        # The scoped world hook receives only the complete builder. Newton's
        # pinned cable importer does not yet expose public group-range metadata,
        # so identify this task's native cable through its importer-owned tables.
        # Keep this narrow adapter here until Newton exposes an extension API.
        expression = re.escape(self.cfg.prim_path + self.cfg.curve_prim_suffix).replace(r"\.\*", r"[^/]*")
        matches = [
            index
            for index, (label, world) in enumerate(zip(builder._cable_label, builder._cable_world, strict=True))
            if int(world) == world_idx and re.fullmatch(expression, str(label))
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one native Waterhose cable group in world {world_idx}, found {len(matches)}.")
        return matches[0]

    def _resolve_head_half_length(self) -> float:
        if self.head_segment_half_length is not None:
            return self.head_segment_half_length
        source_path = self.cfg.prim_path.replace("env_.*", "env_0")
        root_prim = sim_utils.get_current_stage().GetPrimAtPath(source_path)
        if not root_prim.IsValid():
            raise RuntimeError(f"Could not resolve Waterhose cable source prim {source_path!r}.")
        curve = _find_cable_curve(root_prim)
        points = curve.GetPointsAttr().Get()
        self.head_segment_half_length = 0.5 * float((points[1] - points[0]).GetLength())
        return self.head_segment_half_length

    def _load_connector_geometry(self):
        if self._connector_geometry is not None:
            return self._connector_geometry

        plug_builder = newton.ModelBuilder()
        result = plug_builder.add_usd(
            self.cfg.connector_usd_path,
            floating=False,
            load_visual_shapes=True,
            hide_collision_shapes=False,
            parse_mujoco_options=False,
        )
        shape_indices = list(dict.fromkeys(int(shape) for shape in result["path_shape_map"].values()))
        if plug_builder.body_count != 1 or len(shape_indices) != 1:
            raise RuntimeError(
                "Waterhose connector asset must contain exactly one rigid body and one mesh shape; "
                f"found {plug_builder.body_count} bodies and {len(shape_indices)} shapes."
            )

        shape = shape_indices[0]
        mesh = plug_builder.shape_source[shape]
        if not isinstance(mesh, newton.Mesh):
            raise TypeError("Waterhose connector shape must be a triangle mesh.")

        unit_mass, _, _ = newton.geometry.compute_inertia_shape(
            plug_builder.shape_type[shape],
            plug_builder.shape_scale[shape],
            mesh,
            1.0,
            plug_builder.shape_is_solid[shape],
            plug_builder.shape_margin[shape],
        )
        if unit_mass <= 0.0:
            raise RuntimeError("Waterhose connector mesh has zero volume; cannot infer its density.")

        source_body = int(plug_builder.shape_body[shape])
        source_xform = plug_builder.shape_transform[shape]
        if source_body >= 0:
            source_xform = wp.transform_multiply(plug_builder.body_q[source_body], source_xform)
        connector_xform = wp.transform_multiply(
            wp.transform(self.cfg.connector_local_pos, self.cfg.connector_local_quat),
            source_xform,
        )
        self._connector_geometry = (
            mesh,
            plug_builder.shape_scale[shape],
            connector_xform,
            float(self.cfg.connector_mass) / float(unit_mass),
            bool(plug_builder.shape_is_solid[shape]),
            plug_builder.shape_color[shape],
        )
        return self._connector_geometry

    def _add_connector(
        self,
        builder: newton.ModelBuilder,
        world_idx: int,
        body_start: int,
        body_end: int,
    ) -> None:
        from newton import ShapeFlags

        expanded_path = self._expanded_asset_path(world_idx)
        half_length = self._resolve_head_half_length()
        head_body = body_start
        builder.body_label[head_body] = f"{expanded_path}/cable_edge_body_0"

        # Kit renders the authored plug below the body label above. Hide the duplicate static visual
        # from Newton's viewer; its physical compound connector is added immediately afterwards.
        visual_prefix = f"{self.cfg.prim_path.replace('env_.*', 'env_0')}/cable_edge_body_0/connector"
        for shape_id, (label, body, world) in enumerate(
            zip(builder.shape_label, builder.shape_body, builder.shape_world, strict=True)
        ):
            if int(world) == world_idx and int(body) == -1 and str(label).startswith(visual_prefix):
                builder.shape_flags[shape_id] = int(builder.shape_flags[shape_id]) & ~int(ShapeFlags.VISIBLE)

        # Preserve the validated no-self-collision behavior of the previous cable while still
        # allowing the complete cable/plug compound to contact the grippers, housing, and socket.
        cable_collision_group = -(world_idx + 1)
        for body_id in range(body_start, body_end):
            for shape_id in builder.body_shapes[body_id]:
                builder.shape_collision_group[shape_id] = cable_collision_group

        mesh, scale, connector_xform, density, is_solid, color = self._load_connector_geometry()
        connector_xform = wp.transform_multiply(
            wp.transform((0.0, 0.0, -half_length), wp.quat_identity()),
            connector_xform,
        )
        primitive_volume = sum(
            2.0 * math.pi * radius**2 * half_height
            for _, radius, half_height, _ in self.cfg.connector_collision_primitives
        )
        if primitive_volume <= 0.0 or self.cfg.connector_mass <= 0.0:
            raise ValueError("Waterhose connector primitive dimensions and mass must be positive.")
        collider_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=float(self.cfg.connector_ke),
            kd=float(self.cfg.connector_kd),
            mu=float(self.cfg.connector_mu),
            margin=float(self.cfg.connector_margin),
            gap=float(self.cfg.connector_gap),
            collision_group=cable_collision_group,
            is_visible=False,
        )
        mass_before = float(builder.body_mass[head_body])
        # Keep the authored mesh as a render-only shape and use it to preserve the validated mass,
        # center of mass, and inertia. Contact generation sees only the three cheap primitives below.
        builder.add_shape_mesh(
            body=head_body,
            xform=connector_xform,
            mesh=mesh,
            scale=scale,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=density,
                collision_group=0,
                has_shape_collision=False,
                has_particle_collision=False,
                is_solid=is_solid,
            ),
            color=color,
            label=f"{expanded_path}/waterhose_plug_visual",
        )
        for name, radius, half_height, center_z in self.cfg.connector_collision_primitives:
            primitive_xform = wp.transform_multiply(
                connector_xform,
                wp.transform((0.0, 0.0, center_z), wp.quat_identity()),
            )
            builder.add_shape_cylinder(
                body=head_body,
                xform=primitive_xform,
                radius=radius,
                half_height=half_height,
                cfg=collider_cfg,
                color=color,
                label=f"{expanded_path}/{self.cfg.connector_shape_label}_{name}",
            )
        expected_mass = mass_before + float(self.cfg.connector_mass)
        if not math.isclose(float(builder.body_mass[head_body]), expected_mass, rel_tol=1.0e-5, abs_tol=1.0e-9):
            raise RuntimeError("Newton did not accumulate the requested connector mass onto the cable head.")

    def _set_cable_damping(self, builder: newton.ModelBuilder, joint_start: int, joint_end: int) -> None:
        for joint_id in range(joint_start, joint_end):
            linear_dofs, angular_dofs = builder.joint_dof_dim[joint_id]
            if int(linear_dofs) != 2 or int(angular_dofs) != 2:
                raise RuntimeError(
                    f"Native Waterhose cable joint {builder.joint_label[joint_id]!r} does not have 2+2 cable DOFs."
                )
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
        anchor_prim = sim_utils.find_first_matching_prim(self.cfg.tail_anchor_prim_path)
        if anchor_prim is None:
            raise RuntimeError(f"Could not resolve Waterhose tail anchor: {self.cfg.tail_anchor_prim_path!r}.")
        match = re.match(r"(?P<env>.*/env_\d+)", anchor_prim.GetPath().pathString)
        env_prim = anchor_prim.GetStage().GetPrimAtPath(match.group("env")) if match else None
        reference = env_prim if env_prim is not None and env_prim.IsValid() else None
        anchor_xform = wp.transform(*sim_utils.resolve_prim_pose(anchor_prim, ref_prim=reference))
        if reference is not None:
            anchor_xform = wp.transform_multiply(wp.transform(env_position, env_rotation), anchor_xform)

        child_xform = wp.transform_multiply(wp.transform_inverse(builder.body_q[tail_body]), anchor_xform)
        expanded_path = self._expanded_asset_path(world_idx)
        joint = builder.add_joint_fixed(
            parent=-1,
            child=tail_body,
            parent_xform=anchor_xform,
            child_xform=child_xform,
            label=f"{expanded_path}/tail_attachment_w{world_idx}",
            collision_filter_parent=True,
            enabled=True,
            custom_attributes={"vbd:joint_is_hard": 0},
        )
        builder.add_articulation([joint], label=f"{expanded_path}/tail_attachment_articulation")

    def add_to_builder(
        self,
        builder: newton.ModelBuilder,
        world_idx: int,
        env_position: list[float],
        env_rotation: list[float] | tuple[float, float, float, float],
    ) -> None:
        """Add the connector, damping, and tail constraint to one replicated cable world."""
        group = self._curve_group_index(builder, world_idx)
        body_start = int(builder._cable_body_start[group])
        body_end = int(builder._cable_body_end[group])
        joint_start = int(builder._cable_joint_start[group])
        joint_end = int(builder._cable_joint_end[group])
        if body_end <= body_start:
            raise RuntimeError(f"Native Waterhose cable in world {world_idx} contains no segment bodies.")

        self._set_cable_damping(builder, joint_start, joint_end)
        self._add_connector(builder, world_idx, body_start, body_end)
        self._add_tail_attachment(builder, world_idx, body_end - 1, env_position, env_rotation)
