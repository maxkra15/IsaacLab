# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render-only GB300 row, populated switch ports and deterministic activity lights."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import ManagerTermBase
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from isaaclab.utils.version import has_kit

from . import mdp
from .assets import configured_gb300_external_usd
from .workcell import (
    FLOOR_OFFSET_M,
    GB300_PRESENTATION_RACK_ROTATIONS_XYZW,
    GB300_PRESENTATION_RACK_TRANSLATIONS_E,
    GB300_SIMREADY_COMPOSED_BOUNDS_MAX_M,
    GB300_SIMREADY_COMPOSED_BOUNDS_MIN_M,
    SOCKET_POSITION_E,
    UPPER_SOCKET_POSITION_E,
    head_goal_for_socket,
)

if TYPE_CHECKING:
    from pxr import Usd, UsdShade

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg

# Measured centers of the unused, right-hand eight-port bank in the pinned CAD.
# The working pair lies at X=0.377/0.523 m; these leads do not cross its approach.
POPULATED_PORT_X_M = (0.64100519, 0.68294519, 0.71089519)
POPULATED_PORT_LED_X_M = (0.6461701880, 0.6881101915, 0.7160701905)
PATCH_COLORS = ((0.018, 0.26, 0.85), (0.95, 0.65, 0.015), (0.72, 0.78, 0.83))
RACK_HEIGHT_M = GB300_SIMREADY_COMPOSED_BOUNDS_MAX_M[2] - GB300_SIMREADY_COMPOSED_BOUNDS_MIN_M[2]
CEILING_HEIGHT_M = RACK_HEIGHT_M + 0.55


@clone
def spawn_installation_presentation(
    prim_path: str,
    cfg: InstallationPresentationCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn shared, non-colliding scenery through the normal scene lifecycle."""
    from pxr import Gf, UsdGeom, UsdLux

    stage = sim_utils.get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0, 0, 0))))
    x, y, z, w = orientation or (0, 0, 0, 1)
    root.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    asset_path = configured_gb300_external_usd(required=True)
    defer_racks = has_kit()
    for index, (position, rotation) in enumerate(
        zip(GB300_PRESENTATION_RACK_TRANSLATIONS_E, GB300_PRESENTATION_RACK_ROTATIONS_XYZW, strict=True)
    ):
        rack = UsdGeom.Xform.Define(stage, f"{prim_path}/Racks/Rack{index:02d}")
        rack.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], position[2] + FLOOR_OFFSET_M))
        x, y, z, w = rotation
        rack.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
        asset = stage.DefinePrim(f"{rack.GetPath()}/Asset")
        if defer_racks:
            # Kit loads the CAD after Newton builds the physical model, avoiding
            # redundant import of millions of render-only triangles per rack.
            stage.Unload(asset.GetPath())
            asset.GetPayloads().AddPayload(str(asset_path))
        else:
            asset.GetReferences().AddReference(str(asset_path))
        asset.SetInstanceable(True)

    _author_patch_leads(stage, prim_path)
    _author_switchboard_lights(stage, prim_path, cfg.target_socket_position_e)
    if not cfg.include_architecture:
        return root.GetPrim()
    metal = _material(stage, f"{prim_path}/Looks/Structure", (0.12, 0.15, 0.19), metallic=0.65)
    ceiling = _material(stage, f"{prim_path}/Looks/Ceiling", (0.45, 0.49, 0.53), roughness=0.85)
    wall = _material(stage, f"{prim_path}/Looks/Wall", (0.43, 0.47, 0.51), roughness=0.85)
    luminous = _material(stage, f"{prim_path}/Looks/Luminaire", (0.75, 0.85, 0.94), emission=(3.0, 3.4, 3.8))
    # Leave realistic service clearance, filling the former multi-meter void
    # with overhead routing instead of stretching the rack asset vertically.
    _box(stage, f"{prim_path}/Architecture/Ceiling", (2.6, 0.25, CEILING_HEIGHT_M), (7.0, 5.6, 0.10), ceiling)
    # Keep the front camera/work aisle open and >1 m of rear service clearance.
    _box(
        stage,
        f"{prim_path}/Architecture/BackWall",
        (2.6, 3.0, CEILING_HEIGHT_M / 2),
        (7.0, 0.10, CEILING_HEIGHT_M),
        wall,
    )
    _box(
        stage,
        f"{prim_path}/Architecture/EndWall",
        (6.05, 0.25, CEILING_HEIGHT_M / 2),
        (0.10, 5.6, CEILING_HEIGHT_M),
        wall,
    )
    for index, y in enumerate((-0.85, 1.15)):
        _box(stage, f"{prim_path}/Overhead/Busway{index}", (2.6, y, RACK_HEIGHT_M + 0.23), (6.6, 0.12, 0.12), metal)
    for index, x in enumerate((0.35, 1.9, 3.45, 5.0)):
        _box(stage, f"{prim_path}/Overhead/Hanger{index}", (x, 0.15, RACK_HEIGHT_M + 0.40), (0.035, 2.2, 0.025), metal)
        _box(
            stage,
            f"{prim_path}/Lighting/Housing{index}",
            (x, -0.45, CEILING_HEIGHT_M - 0.11),
            (0.75, 0.14, 0.045),
            metal,
        )
        _box(
            stage,
            f"{prim_path}/Lighting/Diffuser{index}",
            (x, -0.45, CEILING_HEIGHT_M - 0.14),
            (0.70, 0.11, 0.012),
            luminous,
        )
        light = UsdLux.RectLight.Define(stage, f"{prim_path}/Lighting/Light{index}")
        light.AddTranslateOp().Set(Gf.Vec3d(x, -0.45, CEILING_HEIGHT_M - 0.15))
        light.CreateWidthAttr(0.70)
        light.CreateHeightAttr(0.11)
        light.CreateIntensityAttr(1700.0)
        light.CreateColorAttr(Gf.Vec3f(0.86, 0.93, 1.0))
    return root.GetPrim()


class SwitchboardActivity(ManagerTermBase):
    """Activate the target indicator on capture and retain its link until withdrawal."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._connected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._env_ids = torch.arange(env.num_envs, device=env.device)

    @property
    def connected(self) -> torch.Tensor:
        """Measured target link state for each environment, shape [N]."""
        return self._connected

    def __call__(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        # Depressing the clip does not disconnect a plug that is still seated.
        self._connected[ids] = mdp.head_seated(env)[ids] & (self._connected[ids] | mdp.latch_captured(env)[ids])
        update_switchboard_lights(env, env_ids, self._connected)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = self._env_ids if env_ids is None else self._env_ids[env_ids]
        self._connected[ids] = False
        update_switchboard_lights(self._env, ids, self._connected)


def update_switchboard_lights(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor | None, connected: torch.Tensor | None = None
) -> None:
    """Animate status colors at simulation time without changing observations or physics."""
    from isaaclab.assets import VisualMaterial

    materials = [env.scene["activity_a"], env.scene["activity_b"]]
    count = env.num_envs if env_ids is None else len(env_ids)
    tick = int(env.common_step_counter * env.step_dt * 10.0)
    # Offset, non-random activity patterns are reproducible across recordings.
    levels = (1.0 if tick % 7 < 4 else 0.06, 1.0 if (tick + 3) % 11 < 7 else 0.06)
    colors = torch.tensor([(0.06 * level, 0.82 * level, 0.035 * level) for level in levels], device=env.device)
    colors = colors[:, None, :].expand(-1, count, -1)
    if connected is not None:
        materials.append(env.scene["activity_target"])
        linked = connected if env_ids is None else connected[env_ids]
        colors = torch.cat((colors, colors[:1] * linked[None, :, None]), dim=0)
    VisualMaterial.write_channels(materials, {"color": colors, "emissive_color": colors * 3.0}, env_ids)


@configclass
class InstallationPresentationCfg(sim_utils.SpawnerCfg):
    """Static actual-CAD scenery with no bodies, joints or collision schemas."""

    func = spawn_installation_presentation
    include_architecture: bool = True
    """Author the shared ceiling and lights; disable when composing an external facility."""
    target_socket_position_e: tuple[float, float, float] = SOCKET_POSITION_E
    """Target socket origin in environment coordinates [m], used to register its indicator."""


def _material(
    stage: Usd.Stage,
    path: str,
    color: tuple[float, float, float],
    *,
    metallic: float = 0.0,
    roughness: float = 0.38,
    emission: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> UsdShade.Material:
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emission))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _box(stage, path, center, size, material) -> None:
    from pxr import Gf, UsdGeom, UsdShade

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    box.AddTranslateOp().Set(Gf.Vec3d(*center))
    box.AddScaleOp().Set(Gf.Vec3d(*size))
    UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(material)


def _author_patch_leads(stage: Usd.Stage, root: str) -> None:
    from pxr import Gf, UsdGeom, UsdShade, Vt

    from .assets import connector_render_parts
    from .cable import CONNECTOR_COLORS, CONNECTOR_SPAN_M, _gb300_connector_normals, _gb300_connector_points

    parts = connector_render_parts(CONNECTOR_SPAN_M)
    connector_materials = {
        name: _material(stage, f"{root}/Looks/Connector{name.capitalize()}", color, roughness=0.5)
        for name, color in CONNECTOR_COLORS.items()
    }
    plug_points = _gb300_connector_points(next(part.points for part in parts if part.name == "plug"))
    rear = plug_points[np.abs(plug_points[:, 2] - plug_points[:, 2].max()) < 1.0e-7]
    rear_center = (rear.min(axis=0) + rear.max(axis=0)) / 2
    # The connector's visible strain relief ends before the terminal rod does.
    # Map its actual rear-face center through the same +90 degree X rotation.
    exit_offset = np.array((rear_center[0], -rear_center[2], rear_center[1]))
    for index, (x, color) in enumerate(zip(POPULATED_PORT_X_M, PATCH_COLORS, strict=True)):
        material = _material(stage, f"{root}/Looks/Patch{index}", color)
        ends = [
            np.array(head_goal_for_socket((x, socket[1], socket[2])))
            for socket in (SOCKET_POSITION_E, UPPER_SOCKET_POSITION_E)
        ]
        for endpoint, center in enumerate(ends):
            connector = UsdGeom.Xform.Define(stage, f"{root}/PopulatedPorts/Patch{index}/Plug{endpoint}")
            connector.AddTranslateOp().Set(Gf.Vec3d(*center))
            connector.AddRotateXOp().Set(90.0)
            for part in parts:
                mesh = UsdGeom.Mesh.Define(stage, f"{connector.GetPath()}/{part.name}")
                mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(_gb300_connector_points(part.points)))
                mesh.CreateFaceVertexCountsAttr(part.face_vertex_counts)
                mesh.CreateFaceVertexIndicesAttr(part.face_vertex_indices)
                mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                if part.normals:
                    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(_gb300_connector_normals(part)))
                    mesh.SetNormalsInterpolation(part.normals_interpolation)
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(connector_materials[part.name])
        start, end = (center + exit_offset for center in ends)
        t = np.linspace(0, 1, 41)[:, None]
        reach = np.array((0.0, -0.095 - index * 0.01, 0.0))
        curve = (1 - t) ** 3 * start + 3 * (1 - t) ** 2 * t * (start + reach)
        curve += 3 * (1 - t) * t**2 * (end + reach) + t**3 * end
        curve[:, 2] -= 0.025 * (16 * t[:, 0] ** 2 * (1 - t[:, 0]) ** 2)
        points, faces = _tube_mesh(curve, radius=0.00325, end_tangents=((0.0, -1.0, 0.0), (0.0, 1.0, 0.0)))
        # The source CAD's 6.5 mm jacket continues through the slotted boot.
        # Extend internally to its solid shoulder without changing the exterior curve.
        for exit_point in (start, end):
            inner = exit_point + (0.0, rear_center[2] + 0.016, 0.0)
            core_points, core_faces = _tube_mesh(
                np.array((inner, exit_point)),
                radius=0.00325,
                end_tangents=((0.0, -1.0, 0.0), (0.0, -1.0, 0.0)),
            )
            faces = np.vstack((faces, core_faces + len(points)))
            points = np.vstack((points, core_points))
        mesh = UsdGeom.Mesh.Define(stage, f"{root}/PopulatedPorts/Patch{index}/Jacket")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr([4] * len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def _tube_mesh(
    centers: np.ndarray, radius: float, *, end_tangents: tuple[tuple[float, float, float], tuple[float, float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    tangent = np.gradient(centers, axis=0)
    tangent[[0, -1]] = end_tangents
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.cross(tangent, (1.0, 0.0, 0.0))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    binormal = np.cross(tangent, normal)
    angle = np.arange(10) * (2 * math.pi / 10)
    ring = np.cos(angle)[None, :, None] * normal[:, None, :] + np.sin(angle)[None, :, None] * binormal[:, None, :]
    points = (centers[:, None, :] + radius * ring).reshape(-1, 3)
    faces = [
        (row * 10 + i, row * 10 + (i + 1) % 10, (row + 1) * 10 + (i + 1) % 10, (row + 1) * 10 + i)
        for row in range(len(centers) - 1)
        for i in range(10)
    ]
    return points, np.asarray(faces)


def _author_switchboard_lights(
    stage: Usd.Stage, root: str, target_socket_position_e: tuple[float, float, float]
) -> None:
    from pxr import Gf, UsdGeom, UsdShade

    def indicator(path, center, material):
        # Actual SN2201 light-pipe face, offset outward 20 micrometers to avoid z-fighting.
        x, y, z = center
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(
            [
                Gf.Vec3f(x + dx, y - 0.00002, z + dz)
                for dx, dz in ((-0.00155, -0.00095), (0.00155, -0.00095), (0.00155, 0.00095), (-0.00155, 0.00095))
            ]
        )
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    def led_center(socket):
        return (socket[0] + 0.0049949875, socket[1] - 0.0127475909, socket[2] + 0.003740)

    materials = [
        _material(stage, f"{root}/Looks/Activity{name}", (0.06, 0.82, 0.035), emission=(0.18, 2.46, 0.105))
        for name in ("A", "B")
    ]
    for level, socket in enumerate((SOCKET_POSITION_E, UPPER_SOCKET_POSITION_E)):
        for column, x in enumerate(POPULATED_PORT_LED_X_M):
            indicator(
                f"{root}/Status/Level{level}/Port{column}",
                (x, socket[1] - 0.0127475909, socket[2] + 0.003740),
                materials[(level + column) % 2],
            )
    indicator(f"{root}/Status/Source", led_center(UPPER_SOCKET_POSITION_E), materials[1])
    target = _material(stage, f"{root}/Looks/ActivityTarget", (0.0, 0.0, 0.0))
    indicator(f"{root}/Status/Target", led_center(target_socket_position_e), target)
