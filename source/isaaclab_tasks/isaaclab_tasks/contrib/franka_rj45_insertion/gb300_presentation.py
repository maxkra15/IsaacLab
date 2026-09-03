# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit SimReady and NewtonGL fallback presentation for the GB300 task."""

from __future__ import annotations

import torch

from .gb300_workcell import GB300_STUDIO_FLOOR_HEIGHT_M

GB300_STUDIO_FLOOR_COLOR = (0.965, 0.975, 0.985)
GB300_STUDIO_FLOOR_ROUGHNESS = 0.10
GB300_STUDIO_BACKWALL_CENTER_E = (2.85, 1.76, GB300_STUDIO_FLOOR_HEIGHT_M + 1.60)
GB300_STUDIO_BACKWALL_SIZE_M = (6.40, 0.08, 3.20)
GB300_STUDIO_BACKWALL_COLOR = (0.985, 0.985, 0.99)
GB300_STUDIO_KEY_LIGHT_POSITION_E = (2.85, -1.75, 2.30)
GB300_STUDIO_FILL_LIGHT_POSITION_E = (-0.35, -1.10, 0.65)
GB300_STUDIO_RIM_LIGHT_POSITION_E = (5.70, 0.80, 1.80)

GB300_FRANKA_PEDESTAL_CENTER_E = (0.0, 0.0, 0.5 * GB300_STUDIO_FLOOR_HEIGHT_M)
GB300_FRANKA_PEDESTAL_SIZE_M = (0.38, 0.38, -GB300_STUDIO_FLOOR_HEIGHT_M)


def _preview_surface_material(
    stage,
    path: str,
    *,
    color: tuple[float, float, float],
    roughness: float,
    metallic: float = 0.0,
    opacity: float = 1.0,
):
    """Author one portable white-studio material."""
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind_material(prim, material) -> None:
    """Bind one material strongly enough to override referenced descendants."""
    from pxr import UsdShade

    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)


def _author_box(
    stage,
    path: str,
    *,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    material,
    collision: bool,
    visible: bool = True,
):
    """Author one axis-aligned visual/collision cuboid."""
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center))
    cube.AddScaleOp().Set(Gf.Vec3d(*size))
    if material is not None:
        _bind_material(cube.GetPrim(), material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if not visible:
        UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
    return cube.GetPrim()


def _author_studio(stage, prim_path: str) -> None:
    """Apply a glossy floor and author the white studio plus Franka support."""
    from pxr import Gf, UsdGeom, UsdLux

    studio_path = f"{prim_path}/Studio"
    UsdGeom.Xform.Define(stage, studio_path)
    floor_material = _preview_surface_material(
        stage,
        f"{studio_path}/Looks/GlossyWhiteFloor",
        color=GB300_STUDIO_FLOOR_COLOR,
        roughness=GB300_STUDIO_FLOOR_ROUGHNESS,
        metallic=0.04,
    )
    ground = stage.GetPrimAtPath("/World/GroundPlane")
    if ground.IsValid():
        _bind_material(ground, floor_material)
    ground_geometry = stage.GetPrimAtPath("/World/GroundPlane/Geometry")
    if ground_geometry.IsValid():
        _bind_material(ground_geometry, floor_material)

    wall_material = _preview_surface_material(
        stage,
        f"{studio_path}/Looks/WhiteBackwall",
        color=GB300_STUDIO_BACKWALL_COLOR,
        roughness=0.22,
    )
    _author_box(
        stage,
        f"{studio_path}/BackWall",
        center=GB300_STUDIO_BACKWALL_CENTER_E,
        size=GB300_STUDIO_BACKWALL_SIZE_M,
        material=wall_material,
        collision=False,
    )

    pedestal_material = _preview_surface_material(
        stage,
        f"{studio_path}/Looks/RobotPedestal",
        color=(0.82, 0.84, 0.87),
        roughness=0.20,
        metallic=0.52,
    )
    _author_box(
        stage,
        f"{studio_path}/FrankaPedestal",
        center=GB300_FRANKA_PEDESTAL_CENTER_E,
        size=GB300_FRANKA_PEDESTAL_SIZE_M,
        material=pedestal_material,
        collision=False,
    )

    key = UsdLux.RectLight.Define(stage, f"{studio_path}/Lights/Key")
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.99, 0.97))
    key.CreateIntensityAttr(4700.0)
    key.CreateWidthAttr(7.0)
    key.CreateHeightAttr(2.4)
    key.AddTranslateOp().Set(Gf.Vec3d(*GB300_STUDIO_KEY_LIGHT_POSITION_E))
    key.AddRotateXOp().Set(72.0)

    fill = UsdLux.SphereLight.Define(stage, f"{studio_path}/Lights/Fill")
    fill.CreateColorAttr(Gf.Vec3f(0.88, 0.94, 1.0))
    fill.CreateIntensityAttr(3600.0)
    fill.CreateRadiusAttr(0.75)
    fill.AddTranslateOp().Set(Gf.Vec3d(*GB300_STUDIO_FILL_LIGHT_POSITION_E))

    rim = UsdLux.SphereLight.Define(stage, f"{studio_path}/Lights/Rim")
    rim.CreateColorAttr(Gf.Vec3f(0.92, 0.96, 1.0))
    rim.CreateIntensityAttr(4600.0)
    rim.CreateRadiusAttr(1.0)
    rim.AddTranslateOp().Set(Gf.Vec3d(*GB300_STUDIO_RIM_LIGHT_POSITION_E))


def _author_cabinet_row(stage, prim_path: str) -> None:
    """Payload the gapless eight-cabinet SimReady row."""
    from pxr import Gf, Sdf, UsdGeom

    from .gb300_asset import configured_gb300_external_usd
    from .gb300_workcell import GB300_PRESENTATION_RACK_ROTATIONS_XYZW, GB300_PRESENTATION_RACK_TRANSLATIONS_E

    payload = configured_gb300_external_usd(required=True)
    assert payload is not None
    UsdGeom.Scope.Define(stage, f"{prim_path}/GB200Scenery")
    for index, (translation, rotation) in enumerate(
        zip(GB300_PRESENTATION_RACK_TRANSLATIONS_E, GB300_PRESENTATION_RACK_ROTATIONS_XYZW, strict=True)
    ):
        rack_path = f"{prim_path}/Exterior" if index == 0 else f"{prim_path}/GB200Scenery/Rack{index:02d}"
        rack = UsdGeom.Xform.Define(stage, rack_path)
        x, y, z, w = rotation
        matrix = Gf.Matrix4d(1.0)
        matrix.SetRotate(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        matrix.SetTranslateOnly(Gf.Vec3d(*translation))
        rack.MakeMatrixXform().Set(matrix)
        # Payload on a child prim so the source root's authored +90-degree
        # rotation and +0.108 m lift compose beneath our world placement.
        # Payloading onto ``rack`` itself would let the stronger placement
        # xformOpOrder suppress the source transform and turn/bury the CAD.
        asset = UsdGeom.Xform.Define(stage, f"{rack_path}/Asset")
        asset_prim = asset.GetPrim()
        asset_prim.GetPayloads().AddPayload(Sdf.Payload(str(payload), "/external"))
        # The SimReady source uses absolute material-binding targets below
        # ``/external/Looks``. Making this payload root instanceable causes Kit
        # to reject those bindings as targets outside the instance prototype,
        # flattening the SN2201 faces into an unreadable black panel. USD still
        # shares the immutable source layer and texture caches across these
        # presentation-only payloads, so preserve composition correctness here.
        asset_prim.SetInstanceable(False)


def _author_first_rack_collision_shell(stage, prim_path: str) -> None:
    """Author invisible static cuboids for the first cabinet collision shell."""
    from pxr import UsdGeom

    from .gb300_workcell import GB300_WORKCELL_CFG

    collision_root = f"{prim_path}/FirstRackCollision"
    UsdGeom.Scope.Define(stage, collision_root)
    for box in GB300_WORKCELL_CFG.boxes:
        if not box.collidable:
            continue
        _author_box(
            stage,
            f"{collision_root}/{box.name.replace('/', '_')}",
            center=box.center_m,
            size=box.size_m,
            material=None,
            collision=True,
            visible=False,
        )


def author_gb300_kit_presentation(stage, prim_path: str) -> None:
    """Payload the legacy Franka-task presentation without changing its physics."""
    from pxr import UsdGeom

    UsdGeom.Xform.Define(stage, prim_path)
    _author_cabinet_row(stage, prim_path)

    _author_studio(stage, prim_path)


def gb300_presentation_contract() -> dict[str, object]:
    """Return render-only scenery values without loading USD or affecting reset contracts."""
    from .gb300_workcell import (
        GB300_PRESENTATION_RACK_COUNT,
        GB300_PRESENTATION_RACK_ROTATIONS_XYZW,
        GB300_PRESENTATION_RACK_SPACING_X_M,
        GB300_PRESENTATION_RACK_TRANSLATIONS_E,
    )

    return {
        "rack_count": GB300_PRESENTATION_RACK_COUNT,
        "rack_spacing_x_m": GB300_PRESENTATION_RACK_SPACING_X_M,
        "rack_translations_e_m": GB300_PRESENTATION_RACK_TRANSLATIONS_E,
        "rack_rotations_xyzw": GB300_PRESENTATION_RACK_ROTATIONS_XYZW,
        "asset_composition": "placement-parent-plus-payload-child-preserves-simready-root-xform",
        "table": "absent",
        "franka_pedestal": {
            "center_e_m": GB300_FRANKA_PEDESTAL_CENTER_E,
            "size_m": GB300_FRANKA_PEDESTAL_SIZE_M,
            "role": "render-only-base-support",
        },
        "cables": {
            "visible_total": 1,
            "physical_task_cables": 1,
            "render_only_drops": 0,
            "free_visible_plug_ends": 1,
        },
        "floor": {
            "color": GB300_STUDIO_FLOOR_COLOR,
            "roughness": GB300_STUDIO_FLOOR_ROUGHNESS,
            "material": "UsdPreviewSurface",
        },
        "backwall": {
            "center_e_m": GB300_STUDIO_BACKWALL_CENTER_E,
            "size_m": GB300_STUDIO_BACKWALL_SIZE_M,
            "color": GB300_STUDIO_BACKWALL_COLOR,
        },
        "lighting": {
            "key_position_e_m": GB300_STUDIO_KEY_LIGHT_POSITION_E,
            "fill_position_e_m": GB300_STUDIO_FILL_LIGHT_POSITION_E,
            "rim_position_e_m": GB300_STUDIO_RIM_LIGHT_POSITION_E,
            "style": "neutral-key-cool-fill-and-rim",
        },
        "physics_effect": "none-render-only-racks-and-franka-pedestal",
    }


def _marker_cfg():
    import isaaclab.sim as sim_utils
    from isaaclab.markers.visualization_markers_cfg import VisualizationMarkersCfg

    from .gb300_workcell import GB300_WORKCELL_CFG

    colors = tuple(dict.fromkeys(box.color for box in GB300_WORKCELL_CFG.boxes))
    return VisualizationMarkersCfg(
        prim_path="/Visuals/RJ45GB300/Workcell",
        markers={
            f"material_{index}": sim_utils.CuboidCfg(
                size=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color,
                    metallic=0.45 if max(color) > 0.3 else 0.12,
                    roughness=0.30,
                ),
            )
            for index, color in enumerate(colors)
        },
    )


def gb300_marker_state(
    env_origins: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a lightweight rack-shell marker state without synthetic ports."""
    from .gb300_workcell import GB300_WORKCELL_CFG

    origins = torch.as_tensor(env_origins)
    if origins.ndim != 2 or origins.shape[1] != 3 or not origins.dtype.is_floating_point:
        raise ValueError(f"env_origins must be floating point with shape (N, 3), got {origins.shape}.")
    boxes = GB300_WORKCELL_CFG.boxes
    colors = tuple(dict.fromkeys(box.color for box in boxes))
    color_to_index = {color: index for index, color in enumerate(colors)}
    device, dtype = origins.device, origins.dtype
    centers = torch.tensor([box.center_m for box in boxes], device=device, dtype=dtype)
    scales = torch.tensor([box.size_m for box in boxes], device=device, dtype=dtype)
    marker_indices = torch.tensor([color_to_index[box.color] for box in boxes], device=device, dtype=torch.int32)
    count, num_envs = len(boxes), len(origins)
    translations = origins[:, None] + centers[None]
    orientations = torch.zeros((num_envs, count, 4), device=device, dtype=dtype)
    orientations[..., 3] = 1.0
    return (
        translations.flatten(0, 1),
        orientations.flatten(0, 1),
        scales[None].expand(num_envs, -1, -1).flatten(0, 1),
        marker_indices[None].expand(num_envs, -1).flatten(),
        torch.arange(num_envs, device=device, dtype=torch.int32).repeat_interleave(count),
    )


class NewtonGlGb300WorkcellPresentation:
    """Render a cheap GB300 silhouette without adding Newton shapes."""

    def __init__(self, sim, env_origins: torch.Tensor):
        from isaaclab.markers import VisualizationMarkers

        self._sim = sim
        self._env_origins = env_origins
        self._marker = VisualizationMarkers(_marker_cfg())
        self._callback_id = f"rj45_gb300_workcell:{id(self)}"
        self._closed = False
        self._sim.vis_marker_registry.add_callback(self._callback_id, self._update)
        self._update()

    def _update(self, _event=None) -> None:
        if self._closed:
            return
        translations, orientations, scales, marker_indices, environment_ids = gb300_marker_state(self._env_origins)
        self._marker.visualize(
            translations=translations,
            orientations=orientations,
            scales=scales,
            marker_indices=marker_indices,
            environment_ids=environment_ids,
        )

    def close(self) -> None:
        """Remove the callback and marker group; safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        self._sim.vis_marker_registry.remove_callback(self._callback_id)
        marker = self._marker
        self._marker = None
        marker.set_visibility(False)


__all__ = [
    "GB300_FRANKA_PEDESTAL_CENTER_E",
    "GB300_FRANKA_PEDESTAL_SIZE_M",
    "NewtonGlGb300WorkcellPresentation",
    "author_gb300_kit_presentation",
    "gb300_marker_state",
    "gb300_presentation_contract",
]
