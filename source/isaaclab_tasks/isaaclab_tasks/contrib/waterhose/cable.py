# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waterhose cable asset with a rigid connector lumped into its head body."""

from __future__ import annotations

import math
import re
from dataclasses import MISSING

import newton
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import combine_frame_transforms, normalize

from isaaclab_contrib.cable.cable_object import CableObject
from isaaclab_contrib.cable.cable_object_cfg import CableObjectCfg


@configclass
class WaterhoseCableObjectCfg(CableObjectCfg):
    """Cable configuration with one mesh rigidly lumped into a segment body."""

    class_type: type | str = "{DIR}.cable:WaterhoseCableObject"

    connector_usd_path: str = MISSING
    """USD containing one rigid body and one connector mesh."""

    connector_mass: float = 0.001
    """Connector mass [kg]."""

    connector_local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Connector origin in the cable-head frame [m]."""

    connector_local_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Connector orientation in the cable-head frame as ``(x, y, z, w)``."""

    connector_shape_label: str = "waterhose_connector"
    """Suffix used to identify the connector collision shape."""

    connector_ke: float = 1.0e4
    """Connector contact stiffness [N/m]."""

    connector_kd: float = 1.0e-1
    """Connector contact damping [N·s/m]."""

    connector_mu: float = 0.5
    """Connector friction coefficient."""

    connector_margin: float = 0.0
    """Connector contact margin [m]."""

    connector_gap: float = 0.01
    """Connector contact gap [m]."""

    tail_anchor_prim_path: str = MISSING
    """Prim path of the static scene marker that anchors the hose tail."""


class WaterhoseCableObject(CableObject):
    """Cable whose connector mesh contributes directly to the head mass and inertia."""

    cfg: WaterhoseCableObjectCfg

    def __init__(self, cfg: WaterhoseCableObjectCfg):
        super().__init__(cfg)
        self._connector_shape_indices: list[int] = []
        self._connector_head_body_ids: torch.Tensor | None = None
        self._connector_local_pose: torch.Tensor | None = None
        self._connector_geometry = None
        self._cable_registry_index = NewtonManager._cable_registry.index(self._registry_entry)

        # Keep waterhose-only construction outside the generic PR 5641 cable layer.
        NewtonManager._per_world_builder_hooks.append(self._add_connector_to_builder)
        NewtonManager._per_world_builder_hooks.append(self._add_tail_attachment_to_builder)

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

    def _add_connector_to_builder(
        self,
        builder: newton.ModelBuilder,
        world_idx: int,
        _env_position: list[float],
        _env_rotation: list[float] | tuple[float, float, float, float],
    ) -> None:
        """Add the connector mesh to this world's cable-head body."""

        if world_idx == 0:
            self._connector_shape_indices.clear()
            self._connector_head_body_ids = None
            self._connector_local_pose = None

        if world_idx >= len(self._registry_entry.body_offsets) or not self._registry_entry.edges:
            raise RuntimeError(f"Waterhose cable has no segment bodies in world {world_idx}.")
        head_body = int(self._registry_entry.body_offsets[world_idx])
        mesh, scale, connector_xform, density, is_solid, color = self._load_connector_geometry()

        mass_before = float(builder.body_mass[head_body])
        expanded_path = self._registry_entry.prim_path.replace("env_.*", f"env_{world_idx}")
        # Bind the hook-built head body to the authored render-only connector parent. Newton's
        # Fabric sync then moves that USD Xform with the real compound body instead of leaving a
        # second plug visual frozen at its authored pose.
        builder.body_label[head_body] = f"{expanded_path}/cable_edge_body_0"
        shape = builder.add_shape_mesh(
            body=head_body,
            xform=connector_xform,
            mesh=mesh,
            scale=scale,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=density,
                ke=float(self.cfg.connector_ke),
                kd=float(self.cfg.connector_kd),
                mu=float(self.cfg.connector_mu),
                margin=float(self.cfg.connector_margin),
                gap=float(self.cfg.connector_gap),
                collision_group=-(1 + self._cable_registry_index),
                is_solid=is_solid,
            ),
            color=color,
            label=f"{expanded_path}/{self.cfg.connector_shape_label}",
        )
        expected_mass = mass_before + float(self.cfg.connector_mass)
        if not math.isclose(float(builder.body_mass[head_body]), expected_mass, rel_tol=1.0e-5, abs_tol=1.0e-9):
            raise RuntimeError("Newton did not accumulate the requested connector mass onto the cable head.")
        self._connector_shape_indices.append(int(shape))

    def _add_tail_attachment_to_builder(
        self,
        builder: newton.ModelBuilder,
        world_idx: int,
        env_position: list[float],
        env_rotation: list[float] | tuple[float, float, float, float],
    ) -> None:
        """Attach the final rod segment to the authored static hose anchor."""
        if world_idx >= len(self._registry_entry.body_offsets) or not self._registry_entry.edges:
            raise RuntimeError(f"Waterhose cable has no tail segment in world {world_idx}.")

        anchor_prim = sim_utils.find_first_matching_prim(self.cfg.tail_anchor_prim_path)
        if anchor_prim is None:
            raise RuntimeError(f"Could not resolve waterhose tail anchor: {self.cfg.tail_anchor_prim_path!r}.")
        match = re.match(r"(?P<env>.*/env_\d+)", anchor_prim.GetPath().pathString)
        env_prim = anchor_prim.GetStage().GetPrimAtPath(match.group("env")) if match else None
        reference = env_prim if env_prim is not None and env_prim.IsValid() else None
        anchor_xform = wp.transform(*sim_utils.resolve_prim_pose(anchor_prim, ref_prim=reference))
        if reference is not None:
            anchor_xform = wp.transform_multiply(wp.transform(env_position, env_rotation), anchor_xform)

        tail_body = int(self._registry_entry.body_offsets[world_idx]) + len(self._registry_entry.edges) - 1
        expanded_path = self._registry_entry.prim_path.replace("env_.*", f"env_{world_idx}")
        joint = builder.add_joint_fixed(
            parent=-1,
            child=tail_body,
            parent_xform=anchor_xform,
            child_xform=wp.transform_identity(),
            label=f"{expanded_path}/tail_attachment_w{world_idx}",
            collision_filter_parent=True,
            enabled=True,
        )
        # The fixed joint is a loop constraint on the existing rod articulation. Register it
        # separately so the VBD entry includes it without changing the rod's parent tree.
        builder.add_articulation([joint], label=f"{expanded_path}/tail_attachment_articulation")

    def get_connector_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return connector positions and orientations in the world frame."""

        if len(self._connector_shape_indices) != len(self._registry_entry.body_offsets):
            raise RuntimeError("Waterhose connector shapes are unavailable or do not match the number of cable worlds.")

        state = NewtonManager.get_state_0()
        body_pose_all = wp.to_torch(state.body_q)
        if self._connector_head_body_ids is None or self._connector_head_body_ids.device != body_pose_all.device:
            self._connector_head_body_ids = torch.tensor(
                [int(body_offset) for body_offset in self._registry_entry.body_offsets],
                dtype=torch.long,
                device=body_pose_all.device,
            )
            self._connector_local_pose = None

        body_pose = body_pose_all[self._connector_head_body_ids]
        if (
            self._connector_local_pose is None
            or self._connector_local_pose.device != body_pose.device
            or self._connector_local_pose.dtype != body_pose.dtype
        ):
            self._connector_local_pose = torch.tensor(
                (*self.cfg.connector_local_pos, *self.cfg.connector_local_quat),
                device=body_pose.device,
                dtype=body_pose.dtype,
            ).unsqueeze(0)
        local_pose = self._connector_local_pose.expand(body_pose.shape[0], -1)
        connector_pos_w, connector_quat_w = combine_frame_transforms(
            body_pose[:, :3],
            normalize(body_pose[:, 3:7]),
            local_pose[:, :3],
            normalize(local_pose[:, 3:7]),
        )
        return connector_pos_w, connector_quat_w

    @property
    def connector_head_body_indices(self) -> tuple[int, ...]:
        """Newton body indices containing each environment's compound connector head."""

        return tuple(int(body_offset) for body_offset in self._registry_entry.body_offsets)
