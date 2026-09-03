# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render-only GB300 showroom for the standalone Sharpa cable task."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

GB300_SIMREADY_REPOSITORY = "nvidia/simready-dsx"
GB300_SIMREADY_REVISION = "5938869019f0d2afb6b9b808ed1ab1bc6e0e0961"
GB300_SIMREADY_RELATIVE_PATH = "GB300/simready_usd/payloads/external.usd"
GB300_SIMREADY_EXTERNAL_USD_SHA256 = "5e0b7b3b58d005b24909b8d2e735c49997f8dbea72352b51911326343ef1e7bb"
GB300_SIMREADY_EXTERNAL_USD_SIZE = 473_434_496
GB300_SIMREADY_LICENSE = "CC-BY-4.0"
GB300_SIMREADY_ASSET_ROOT_ENV = "ISAACLAB_SIMREADY_DSX_GB300_ROOT"
GB300_SIMREADY_DOWNLOAD_URL = (
    "https://huggingface.co/datasets/nvidia/simready-dsx/resolve/"
    f"{GB300_SIMREADY_REVISION}/{GB300_SIMREADY_RELATIVE_PATH}?download=true"
)

GB300_SHOWROOM_RACK_COUNT = 8
GB300_SHOWROOM_RACK_SPACING_X_M = 0.6569000133217894
GB300_SHOWROOM_RACK_ROTATION_XYZW = (0.0, 0.0, -0.7071067811865476, 0.7071067811865476)
GB300_SHOWROOM_RACK_FIRST_TRANSLATION_E = (0.5800, 0.6833, 0.0325162718685001)
GB300_SHOWROOM_RACK_TRANSLATIONS_E = tuple(
    (
        GB300_SHOWROOM_RACK_FIRST_TRANSLATION_E[0] + index * GB300_SHOWROOM_RACK_SPACING_X_M,
        GB300_SHOWROOM_RACK_FIRST_TRANSLATION_E[1],
        GB300_SHOWROOM_RACK_FIRST_TRANSLATION_E[2],
    )
    for index in range(GB300_SHOWROOM_RACK_COUNT)
)
GB300_SHOWROOM_FLOOR_COLOR = (0.965, 0.975, 0.985)
GB300_SHOWROOM_FLOOR_ROUGHNESS = 0.10
GB300_SHOWROOM_BACKWALL_CENTER_E = (2.85, 1.76, 1.60)
GB300_SHOWROOM_BACKWALL_SIZE_M = (6.40, 0.08, 3.20)


def default_gb300_external_usd_path() -> Path:
    """Return the content-addressed default path of the pinned GB300 payload."""
    return (
        Path.home()
        / ".cache"
        / "isaaclab"
        / "simready-dsx"
        / "sha256"
        / GB300_SIMREADY_EXTERNAL_USD_SHA256
        / "external.usd"
    )


@lru_cache(maxsize=4)
def verify_gb300_external_usd(path: str | Path) -> Path:
    """Verify and return the exact local SimReady GB300 payload."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Pinned GB300 external USD is missing: {resolved}")
    size = resolved.stat().st_size
    if size != GB300_SIMREADY_EXTERNAL_USD_SIZE:
        raise RuntimeError(
            f"GB300 external USD size mismatch: expected {GB300_SIMREADY_EXTERNAL_USD_SIZE}, got {size}."
        )
    with resolved.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != GB300_SIMREADY_EXTERNAL_USD_SHA256:
        raise RuntimeError(
            f"GB300 external USD SHA-256 mismatch: expected {GB300_SIMREADY_EXTERNAL_USD_SHA256}, got {digest}."
        )
    return resolved


def configured_gb300_external_usd() -> Path:
    """Resolve the required pinned payload from its override or default cache."""
    configured_root = os.environ.get(GB300_SIMREADY_ASSET_ROOT_ENV)
    if configured_root:
        root = Path(configured_root).expanduser()
        candidate = root if root.suffix.lower() in (".usd", ".usda", ".usdc") else root / "external.usd"
    else:
        candidate = default_gb300_external_usd_path()
    if not candidate.is_file():
        raise FileNotFoundError(
            "The Rizon Sharpa showroom requires the pinned SimReady GB300 payload at "
            f"{candidate}. Download {GB300_SIMREADY_DOWNLOAD_URL}, or set {GB300_SIMREADY_ASSET_ROOT_ENV}."
        )
    return verify_gb300_external_usd(candidate)


def _preview_surface_material(
    stage,
    path: str,
    *,
    color: tuple[float, float, float],
    roughness: float,
    metallic: float = 0.0,
):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind_material(prim, material) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )


def _author_backwall(stage, root_path: str, material) -> None:
    from pxr import Gf, UsdGeom

    wall = UsdGeom.Cube.Define(stage, f"{root_path}/Studio/BackWall")
    wall.CreateSizeAttr(1.0)
    wall.AddTranslateOp().Set(Gf.Vec3d(*GB300_SHOWROOM_BACKWALL_CENTER_E))
    wall.AddScaleOp().Set(Gf.Vec3d(*GB300_SHOWROOM_BACKWALL_SIZE_M))
    _bind_material(wall.GetPrim(), material)


def _author_lights(stage, root_path: str) -> None:
    from pxr import Gf, UsdLux

    key = UsdLux.RectLight.Define(stage, f"{root_path}/Studio/Lights/Key")
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.99, 0.97))
    key.CreateIntensityAttr(4200.0)
    key.CreateWidthAttr(7.0)
    key.CreateHeightAttr(2.4)
    key.AddTranslateOp().Set(Gf.Vec3d(2.85, -1.75, 2.55))
    key.AddRotateXOp().Set(72.0)

    fill = UsdLux.SphereLight.Define(stage, f"{root_path}/Studio/Lights/Fill")
    fill.CreateColorAttr(Gf.Vec3f(0.88, 0.94, 1.0))
    fill.CreateIntensityAttr(3000.0)
    fill.CreateRadiusAttr(0.75)
    fill.AddTranslateOp().Set(Gf.Vec3d(-0.35, -1.10, 1.00))

    rim = UsdLux.SphereLight.Define(stage, f"{root_path}/Studio/Lights/Rim")
    rim.CreateColorAttr(Gf.Vec3f(0.92, 0.96, 1.0))
    rim.CreateIntensityAttr(3800.0)
    rim.CreateRadiusAttr(1.0)
    rim.AddTranslateOp().Set(Gf.Vec3d(5.70, 0.80, 2.00))


def _author_racks(stage, root_path: str, payload: Path) -> None:
    from pxr import Gf, Sdf, UsdGeom

    racks_root = f"{root_path}/GB300Row"
    UsdGeom.Scope.Define(stage, racks_root)
    x, y, z, w = GB300_SHOWROOM_RACK_ROTATION_XYZW
    for index, translation in enumerate(GB300_SHOWROOM_RACK_TRANSLATIONS_E):
        rack_path = f"{racks_root}/Rack{index:02d}"
        rack = UsdGeom.Xform.Define(stage, rack_path)
        matrix = Gf.Matrix4d(1.0)
        matrix.SetRotate(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        matrix.SetTranslateOnly(Gf.Vec3d(*translation))
        rack.MakeMatrixXform().Set(matrix)

        # Keep the SimReady root transform on a payload child. It contains the
        # compensating +90-degree turn and authored vertical lift.
        asset = UsdGeom.Xform.Define(stage, f"{rack_path}/Asset").GetPrim()
        asset.GetPayloads().AddPayload(Sdf.Payload(str(payload), "/external"))
        # Absolute material bindings in this asset cannot cross an instance
        # prototype boundary, so composition-correct copies remain non-instanceable.
        asset.SetInstanceable(False)


def author_rizon_sharpa_showroom(stage, root_path: str = "/World/RizonSharpaShowroom") -> None:
    """Author eight render-only GB300s and a neutral white Kit studio."""
    from pxr import UsdGeom

    if stage.GetPrimAtPath(root_path).IsValid():
        return
    payload = configured_gb300_external_usd()
    UsdGeom.Xform.Define(stage, root_path)
    UsdGeom.Scope.Define(stage, f"{root_path}/Studio")
    UsdGeom.Scope.Define(stage, f"{root_path}/Studio/Looks")

    floor_material = _preview_surface_material(
        stage,
        f"{root_path}/Studio/Looks/GlossyWhiteFloor",
        color=GB300_SHOWROOM_FLOOR_COLOR,
        roughness=GB300_SHOWROOM_FLOOR_ROUGHNESS,
        metallic=0.04,
    )
    for path in ("/World/GroundPlane", "/World/GroundPlane/Geometry"):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            _bind_material(prim, floor_material)

    wall_material = _preview_surface_material(
        stage,
        f"{root_path}/Studio/Looks/WhiteBackwall",
        color=(0.985, 0.985, 0.99),
        roughness=0.22,
    )
    _author_backwall(stage, root_path, wall_material)
    _author_lights(stage, root_path)
    _author_racks(stage, root_path, payload)


def rizon_sharpa_showroom_contract() -> dict[str, object]:
    """Return the presentation-only showroom contract without opening USD."""
    return {
        "rack_count": GB300_SHOWROOM_RACK_COUNT,
        "rack_spacing_x_m": GB300_SHOWROOM_RACK_SPACING_X_M,
        "rack_translations_e_m": GB300_SHOWROOM_RACK_TRANSLATIONS_E,
        "rack_rotation_xyzw": GB300_SHOWROOM_RACK_ROTATION_XYZW,
        "floor": {
            "height_m": 0.0,
            "color": GB300_SHOWROOM_FLOOR_COLOR,
            "roughness": GB300_SHOWROOM_FLOOR_ROUGHNESS,
        },
        "backwall": {
            "center_e_m": GB300_SHOWROOM_BACKWALL_CENTER_E,
            "size_m": GB300_SHOWROOM_BACKWALL_SIZE_M,
        },
        "pedestal": "physical-single-cuboid-authored-by-scene",
        "physics_effect": "none-render-only-racks-wall-and-lights",
        "asset": {
            "repository": GB300_SIMREADY_REPOSITORY,
            "revision": GB300_SIMREADY_REVISION,
            "relative_path": GB300_SIMREADY_RELATIVE_PATH,
            "file_sha256": GB300_SIMREADY_EXTERNAL_USD_SHA256,
            "file_size_bytes": GB300_SIMREADY_EXTERNAL_USD_SIZE,
            "license": GB300_SIMREADY_LICENSE,
        },
    }


__all__ = [
    "GB300_SHOWROOM_BACKWALL_CENTER_E",
    "GB300_SHOWROOM_BACKWALL_SIZE_M",
    "GB300_SHOWROOM_FLOOR_COLOR",
    "GB300_SHOWROOM_FLOOR_ROUGHNESS",
    "GB300_SHOWROOM_RACK_COUNT",
    "GB300_SHOWROOM_RACK_ROTATION_XYZW",
    "GB300_SHOWROOM_RACK_SPACING_X_M",
    "GB300_SHOWROOM_RACK_TRANSLATIONS_E",
    "author_rizon_sharpa_showroom",
    "configured_gb300_external_usd",
    "default_gb300_external_usd_path",
    "rizon_sharpa_showroom_contract",
    "verify_gb300_external_usd",
]
