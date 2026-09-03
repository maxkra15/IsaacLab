# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration and stage authoring for one hanging RJ45 cable."""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import CableObjectCfg
from isaaclab.sim.spawners.shapes.shapes import spawn_cable
from isaaclab.sim.utils import clone, create_prim
from isaaclab.utils.configclass import configclass

CABLE_LENGTH_M = 0.50
CABLE_SEGMENT_COUNT = 50
CABLE_CONNECTOR_RIGID_SPAN_M = 0.040
CABLE_FLEX_SEGMENT_LENGTH_M = (CABLE_LENGTH_M - CABLE_CONNECTOR_RIGID_SPAN_M) / (CABLE_SEGMENT_COUNT - 1)
CABLE_DIAMETER_M = 0.0065
CABLE_INITIAL_LATERAL_OFFSET_M = 0.10
CABLE_INITIAL_VERTICAL_SPAN_M = math.sqrt(CABLE_LENGTH_M**2 - CABLE_INITIAL_LATERAL_OFFSET_M**2)


def hanging_cable_positions() -> tuple[tuple[float, float, float], ...]:
    """Return an exact-length cable with a rigid connector strain-relief span."""
    arc_lengths = (0.0,) + tuple(
        CABLE_CONNECTOR_RIGID_SPAN_M + CABLE_FLEX_SEGMENT_LENGTH_M * (index - 1)
        for index in range(1, CABLE_SEGMENT_COUNT + 1)
    )
    return tuple(
        (
            CABLE_INITIAL_LATERAL_OFFSET_M * arc_length / CABLE_LENGTH_M,
            0.0,
            CABLE_INITIAL_VERTICAL_SPAN_M * arc_length / CABLE_LENGTH_M,
        )
        for arc_length in arc_lengths
    )


@clone
def spawn_hanging_rj45_cable(
    prim_path: str,
    cfg: HangingRj45CableSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Author the native cable curve and a head transform for the RJ45 visual."""
    root = spawn_cable.__wrapped__(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )

    # Import runtime geometry only after stage creation. This keeps task-config imports free of
    # Newton and USD bindings before Kit owns them.
    from pxr import Gf, UsdGeom, Vt  # noqa: PLC0415

    from .cable import connector_render_parts, socket_render_part  # noqa: PLC0415

    stage = root.GetStage()
    head_path = f"{prim_path}/connector_head"
    head = UsdGeom.Xform.Define(stage, head_path)
    first, second = cfg.positions[:2]
    midpoint = tuple(0.5 * (start + end) for start, end in zip(first, second, strict=True))
    head.AddTranslateOp().Set(Gf.Vec3d(*midpoint))
    colors = ((0.025, 0.19, 0.44), (0.10, 0.34, 0.62))
    for part, color in zip(connector_render_parts(CABLE_CONNECTOR_RIGID_SPAN_M), colors, strict=True):
        mesh = UsdGeom.Mesh.Define(stage, f"{head_path}/{part.name}")
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*point) for point in part.points]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray(part.face_vertex_counts))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(part.face_vertex_indices))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
        mesh.CreateDoubleSidedAttr(False)
    create_prim(f"{head_path}/geometry", prim_type="Scope", stage=stage)

    # Keep the exact insertion target under the cloned cable root for Kit.
    # Its configured pose is environment-local, while this root is translated
    # to the cable's free-end position.
    if translation is None:
        translation = (0.0, 0.0, 0.0)
    target_path = f"{prim_path}/InsertionTargetVisual"
    target = UsdGeom.Xform.Define(stage, target_path)
    target.AddTranslateOp().Set(
        Gf.Vec3d(
            *(
                target_value - root_value
                for target_value, root_value in zip(cfg.insertion_target_position_e, translation)
            )
        )
    )
    x, y, z, w = cfg.insertion_target_rotation_xyzw
    target.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    socket = socket_render_part()
    socket_mesh = UsdGeom.Mesh.Define(stage, f"{target_path}/socket")
    socket_mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*point) for point in socket.points]))
    socket_mesh.CreateFaceVertexCountsAttr(Vt.IntArray(socket.face_vertex_counts))
    socket_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(socket.face_vertex_indices))
    socket_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    socket_mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.03, 0.62, 0.90)]))
    socket_mesh.CreateDoubleSidedAttr(False)
    return root


@configclass
class HangingRj45CableSpawnerCfg(sim_utils.CableCfg):
    """Procedural cable curve with a dynamic RJ45 visual at its head."""

    func = spawn_hanging_rj45_cable

    insertion_target_position_e: tuple[float, float, float] = MISSING
    """Environment-local position of the floating socket body origin [m]."""

    insertion_target_rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Environment-local orientation of the floating socket body."""


@configclass
class RizonSharpaCableObjectCfg(CableObjectCfg):
    """Native Newton cable plus one rigid compound RJ45 head and one fixed tail."""

    class_type: type | str = "isaaclab_tasks.contrib.rizon_sharpa_cable.cable:RizonSharpaCableObject"

    tail_anchor_prim_path: str = MISSING
    """Prim path of the fixed upper cable attachment."""

    curve_prim_suffix: str = "/geometry/mesh"
    """Native deformable-curve path relative to the cable root."""

    connector_rigid_span_m: float = CABLE_CONNECTOR_RIGID_SPAN_M
    """Length of the connector-owning first cable span [m]."""

    connector_density_kg_m3: float = 2_000.0
    """Density of the cheap connector collision box [kg/m^3]."""

    connector_friction: float = 4.0
    """Coulomb friction coefficient of the connector grasp proxy."""

    connector_contact_margin_m: float = 0.0005
    """Physical connector contact margin [m]."""

    connector_contact_gap_m: float = 0.004
    """Connector broad-phase contact gap [m]."""

    stretch_shear_damping: float = 1.0e-2
    """Per-joint axial/shear damping [N.s/m]."""

    bend_twist_damping: float = 2.0e-2
    """Per-joint bend/twist damping [N.m.s/rad]."""


__all__ = [
    "CABLE_DIAMETER_M",
    "CABLE_CONNECTOR_RIGID_SPAN_M",
    "CABLE_FLEX_SEGMENT_LENGTH_M",
    "CABLE_INITIAL_LATERAL_OFFSET_M",
    "CABLE_INITIAL_VERTICAL_SPAN_M",
    "CABLE_LENGTH_M",
    "CABLE_SEGMENT_COUNT",
    "HangingRj45CableSpawnerCfg",
    "RizonSharpaCableObjectCfg",
    "hanging_cable_positions",
    "spawn_hanging_rj45_cable",
]
