# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Material-space artwork for the deforming textile atelier curtain."""

from __future__ import annotations

from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from isaaclab.sim.spawners.materials.visual_materials import spawn_preview_surface
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg

_TEXTURE_PATH = Path(__file__).with_name("kinetic_tapestry.png")


def spawn_kinetic_tapestry_material(
    prim_path: str,
    cfg: PreviewSurfaceCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Bind a tapestry texture to the cloth's own subdivided visual mesh.

    This callback runs while the rectangle prototype is being spawned. Isaac Lab
    then clones the textured mesh and material together for every environment.

    Args:
        prim_path: Absolute path of the visual material.
        cfg: Preview-surface configuration.
        translation: Unused material translation [m].
        orientation: Unused material rotation (xyzw quaternion).

    Returns:
        The authored preview-surface shader prim.
    """
    shader_prim = spawn_preview_surface(prim_path, cfg, translation, orientation)
    stage = shader_prim.GetStage()
    material_path = Sdf.Path(prim_path)
    mesh_path = material_path.GetParentPath().AppendChild("mesh")
    mesh = UsdGeom.Mesh.Get(stage, mesh_path)
    if not mesh:
        raise ValueError(f"Tapestry material requires a sibling mesh at '{mesh_path}'.")

    points = mesh.GetPointsAttr().Get()
    x_min = min(point[0] for point in points)
    x_max = max(point[0] for point in points)
    y_min = min(point[1] for point in points)
    y_max = max(point[1] for point in points)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Tapestry mesh must span both local X and Y axes: '{mesh.GetPath()}'.")
    texcoords = [
        Gf.Vec2f((point[0] - x_min) / (x_max - x_min), (point[1] - y_min) / (y_max - y_min)) for point in points
    ]
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
        texcoords
    )

    material = UsdShade.Material.Get(stage, material_path)
    st_name = material.CreateInput("frame:stPrimvarName", Sdf.ValueTypeNames.Token)
    st_name.Set("st")
    st_reader = UsdShade.Shader.Define(stage, material_path.AppendChild("StReader"))
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).ConnectToSource(st_name)
    st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    texture = UsdShade.Shader.Define(stage, material_path.AppendChild("Texture"))
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(_TEXTURE_PATH)))
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    UsdShade.Shader(shader_prim).CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    return shader_prim
