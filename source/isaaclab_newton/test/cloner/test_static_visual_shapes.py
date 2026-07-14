# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for importing static render-only meshes into Newton clone builders."""

import newton
from isaaclab_newton.cloner.newton_clone_utils import add_static_visual_shapes_from_stage

from pxr import Gf, Usd, UsdGeom, UsdPhysics


def _define_triangle(stage: Usd.Stage, path: str) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(1.0, 0.0, 0.0),
            Gf.Vec3f(0.0, 1.0, 0.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    return mesh


def test_static_render_meshes_are_imported_as_collision_disabled_world_shapes():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Root")
    _define_triangle(stage, "/Root/Visible")

    collision_mesh = _define_triangle(stage, "/Root/Collision")
    UsdPhysics.CollisionAPI.Apply(collision_mesh.GetPrim())

    guide_mesh = _define_triangle(stage, "/Root/Guide")
    guide_mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)

    invisible_mesh = _define_triangle(stage, "/Root/Invisible")
    invisible_mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)

    _define_triangle(stage, "/Root/Duplicate")

    builder = newton.ModelBuilder()
    builder.add_shape_box(
        body=-1,
        cfg=newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
            has_particle_collision=False,
        ),
        label="/Root/Duplicate",
    )

    added = add_static_visual_shapes_from_stage(builder, stage, "/Root")

    assert added == 1
    assert builder.shape_label.count("/Root/Duplicate") == 1
    visible_index = builder.shape_label.index("/Root/Visible")
    assert builder.shape_body[visible_index] == -1
    assert builder.shape_flags[visible_index] == int(newton.ShapeFlags.VISIBLE)
    assert not any(label in builder.shape_label for label in ("/Root/Collision", "/Root/Guide", "/Root/Invisible"))
