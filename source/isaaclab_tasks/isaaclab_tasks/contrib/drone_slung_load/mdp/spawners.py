# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD spawners for slung-load attachments, colors, and local drone visuals."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path

from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils.configclass import configclass

from ..system import ROTOR_ARM_LENGTH, ROTOR_HEIGHT


@clone
def spawn_physics_attachment(
    prim_path: str,
    cfg: PhysicsAttachmentCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Author a proposal ``PhysicsAttachment`` under an imageable xform.

    The returned prim is an ``Xform`` so Isaac Lab's clone decorator can set
    visibility. The attachment itself lives at ``{prim_path}/attachment`` and is
    kept **outside** the cable articulation (a loop/attachment joint). Stiffness
    is left unauthored so Newton imports a hard ball joint.

    ``src0`` / ``src1`` are relationship targets relative to the spawned xform
    (for example ``../Cable/geometry/mesh``). They are rewritten relative to the
    child attachment prim.

    Args:
        prim_path: Absolute prim path of the attachment xform.
        cfg: Attachment configuration.
        translation: Unused; accepted for spawner compatibility.
        orientation: Unused; accepted for spawner compatibility.
        **kwargs: Additional arguments consumed by :func:`clone`.

    Returns:
        The created xform prim.
    """
    del translation, orientation, kwargs
    from pxr import Sdf, UsdGeom

    stage = get_current_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    attach_prim = stage.DefinePrim(f"{prim_path}/attachment", "PhysicsAttachment")
    attach_prim.CreateRelationship("physics:src0").SetTargets([_attachment_relative_target(cfg.src0)])
    if cfg.src1:
        attach_prim.CreateRelationship("physics:src1").SetTargets([_attachment_relative_target(cfg.src1)])
    attach_prim.CreateAttribute("physics:type0", Sdf.ValueTypeNames.Token).Set(cfg.type0)
    attach_prim.CreateAttribute("physics:type1", Sdf.ValueTypeNames.Token).Set(cfg.type1)
    attach_prim.CreateAttribute("physics:indices0", Sdf.ValueTypeNames.IntArray).Set(list(cfg.indices0))
    if cfg.indices1 is not None:
        attach_prim.CreateAttribute("physics:indices1", Sdf.ValueTypeNames.IntArray).Set(list(cfg.indices1))
    if cfg.coords0 is not None:
        attach_prim.CreateAttribute("physics:coords0", Sdf.ValueTypeNames.Vector3fArray).Set(
            [tuple(coord) for coord in cfg.coords0]
        )
    if cfg.coords1 is not None:
        attach_prim.CreateAttribute("physics:coords1", Sdf.ValueTypeNames.Vector3fArray).Set(
            [tuple(coord) for coord in cfg.coords1]
        )
    attach_prim.CreateAttribute("physics:attachmentEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    return xform.GetPrim()


def _set_mesh_display_color(prim_path: str, color: tuple[float, float, float] | None) -> None:
    """Author ``displayColor`` on ``{prim_path}/geometry/mesh`` for Newton GL."""
    if color is None:
        return
    from pxr import Gf, Sdf, UsdGeom

    mesh = get_current_stage().GetPrimAtPath(f"{prim_path}/geometry/mesh")
    if not mesh:
        return
    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant, 1
    )
    primvar.Set([Gf.Vec3f(*color)])


def _spawn_cuboid_with_color(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn a cuboid and author Newton-visible ``displayColor`` on the mesh."""
    from isaaclab.sim.spawners.shapes.shapes import spawn_cuboid

    prim = spawn_cuboid(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    _set_mesh_display_color(str(prim.GetPath()), getattr(cfg, "display_color", None))
    return prim


def _set_primitive_display_color(primitive, color: tuple[float, float, float]) -> None:
    """Author a constant color on a visual-only USD primitive."""
    from pxr import Gf

    primitive.CreateDisplayColorAttr([Gf.Vec3f(*color)])


_CRAZYFLIE_SOURCE_ROTOR_SITE = 0.0296
"""Approximate horizontal rotor coordinate in Newton's Crazyflie example asset [m]."""

_CRAZYFLIE_SOURCE_ROTOR_HEIGHT = 0.02682415
"""Common rotor-plane coordinate along the source asset's Y-up axis [m]."""

_CRAZYFLIE_BODY_MESH_COLORS = {
    "body/arms": "arm",
    "body/battery": (0.12, 0.13, 0.15),
    "body/battery_holder": (0.04, 0.05, 0.06),
    "body/board": "body",
    "body/motor_mounts": "arm",
    "body/motors": (0.10, 0.11, 0.12),
    "body/pin_racks": (0.16, 0.17, 0.18),
    "body/pins": (0.55, 0.57, 0.60),
}
"""Per-component Newton-GL colors for the unmaterialed example mesh."""

# After rotating source +Y to target +Z, these paths are in physical rotor order:
# (+x, -y), (-x, -y), (-x, +y), (+x, +y).
_CRAZYFLIE_ROTOR_PATHS = (
    "propeller_cw_front/propeller",
    "propeller_ccw_front/propeller",
    "propeller_cw_back/propeller",
    "propeller_ccw_back/propeller",
)


def _resolve_crazyflie_asset() -> str | None:
    """Resolve the optional Crazyflie mesh shipped by the installed Newton package."""
    try:
        from newton import examples as newton_examples
    except ImportError:
        return None

    try:
        asset_path = Path(newton_examples.get_asset("crazyflie.usd"))
    except (OSError, TypeError):
        return None
    return str(asset_path) if asset_path.is_file() else None


def _subtree_is_visual_only(root_prim) -> bool:
    """Return whether a referenced subtree contains meshes but no rigid physics schemas."""
    from pxr import Usd, UsdGeom, UsdPhysics

    found_mesh = False
    for prim in Usd.PrimRange(root_prim):
        found_mesh |= prim.IsA(UsdGeom.Mesh)
        if (
            prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or prim.HasAPI(UsdPhysics.CollisionAPI)
            or prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            or prim.HasAPI(UsdPhysics.MassAPI)
            or prim.IsA(UsdPhysics.Joint)
        ):
            return False
    return found_mesh


def _spawn_crazyflie_visuals(visuals_path: str, cfg: DroneCuboidCfg, asset_path: str) -> bool:
    """Reference and color Newton's visual-only Crazyflie example asset."""
    from pxr import Gf, Usd, UsdGeom

    stage = get_current_stage()
    model_path = f"{visuals_path}/crazyflie"
    model = UsdGeom.Xform.Define(stage, model_path)
    scale = cfg.arm_length / _CRAZYFLIE_SOURCE_ROTOR_SITE
    model_xform = UsdGeom.Xformable(model)
    # The source is Y-up. Align its rotor plane to the configured physical
    # rotor height; the body may sit below that plane as on the source vehicle.
    model_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, cfg.rotor_z - _CRAZYFLIE_SOURCE_ROTOR_HEIGHT * scale))
    model_xform.AddRotateXOp().Set(90.0)
    model_xform.AddScaleOp().Set(Gf.Vec3f(scale))
    if not model.GetPrim().GetReferences().AddReference(asset_path):
        stage.RemovePrim(model_path)
        return False
    if not _subtree_is_visual_only(model.GetPrim()):
        stage.RemovePrim(model_path)
        return False

    # Give every current or future mesh a Newton-GL-safe base color before the
    # component-specific overrides below.
    for prim in Usd.PrimRange(model.GetPrim()):
        mesh = UsdGeom.Mesh(prim)
        if mesh:
            _set_primitive_display_color(mesh, cfg.arm_color)
    for relative_path, color_spec in _CRAZYFLIE_BODY_MESH_COLORS.items():
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"{model_path}/{relative_path}"))
        if mesh:
            color = cfg.display_color if color_spec == "body" else cfg.arm_color if color_spec == "arm" else color_spec
            _set_primitive_display_color(mesh, color)
    for relative_path, color in zip(_CRAZYFLIE_ROTOR_PATHS, cfg.rotor_colors, strict=True):
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"{model_path}/{relative_path}"))
        if mesh:
            _set_primitive_display_color(mesh, color)
    return True


def _spawn_procedural_drone_visuals(visuals_path: str, cfg: DroneCuboidCfg) -> None:
    """Author the visual-only fallback body, crossed arms, and four rotor discs."""
    from pxr import Gf, UsdGeom

    stage = get_current_stage()
    body = UsdGeom.Cube.Define(stage, f"{visuals_path}/body")
    body.CreateSizeAttr(1.0)
    UsdGeom.Xformable(body).AddScaleOp().Set(Gf.Vec3f(*cfg.size))
    _set_primitive_display_color(body, cfg.display_color)

    arm_length = 2.0 * math.sqrt(2.0) * cfg.arm_length
    for index, angle in enumerate((-45.0, 45.0)):
        arm = UsdGeom.Cube.Define(stage, f"{visuals_path}/arm_{index}")
        arm.CreateSizeAttr(1.0)
        arm_xform = UsdGeom.Xformable(arm)
        arm_xform.AddRotateZOp().Set(angle)
        arm_xform.AddScaleOp().Set(Gf.Vec3f(arm_length, cfg.arm_width, cfg.arm_height))
        _set_primitive_display_color(arm, cfg.arm_color)

    rotor_positions = (
        (cfg.arm_length, -cfg.arm_length, cfg.rotor_z),
        (-cfg.arm_length, -cfg.arm_length, cfg.rotor_z),
        (-cfg.arm_length, cfg.arm_length, cfg.rotor_z),
        (cfg.arm_length, cfg.arm_length, cfg.rotor_z),
    )
    for index, (position, color) in enumerate(zip(rotor_positions, cfg.rotor_colors, strict=True)):
        rotor = UsdGeom.Cylinder.Define(stage, f"{visuals_path}/rotor_{index}")
        rotor.CreateRadiusAttr(cfg.rotor_radius)
        rotor.CreateHeightAttr(cfg.rotor_height)
        rotor.CreateAxisAttr(UsdGeom.Tokens.z)
        UsdGeom.Xformable(rotor).AddTranslateOp().Set(Gf.Vec3d(*position))
        _set_primitive_display_color(rotor, color)


def _author_explicit_inertia(root, cfg: DroneCuboidCfg) -> None:
    """Validate and author optional principal moments on the authoritative body."""
    if cfg.diagonal_inertia is None:
        if cfg.principal_axes is not None:
            raise ValueError("DroneCuboidCfg principal_axes requires diagonal_inertia.")
        return

    inertia = tuple(float(value) for value in cfg.diagonal_inertia)
    if len(inertia) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in inertia):
        raise ValueError("DroneCuboidCfg diagonal_inertia must contain three finite positive moments.")
    for index, moment in enumerate(inertia):
        if moment > sum(inertia) - moment + 1.0e-12:
            raise ValueError(
                f"DroneCuboidCfg diagonal_inertia[{index}] violates the principal-moment triangle inequality."
            )

    axes = (1.0, 0.0, 0.0, 0.0) if cfg.principal_axes is None else tuple(float(v) for v in cfg.principal_axes)
    if len(axes) != 4 or any(not math.isfinite(value) for value in axes):
        raise ValueError("DroneCuboidCfg principal_axes must be a finite (w, x, y, z) quaternion.")
    norm = math.sqrt(sum(value * value for value in axes))
    if not math.isclose(norm, 1.0, rel_tol=1.0e-5, abs_tol=1.0e-6):
        raise ValueError("DroneCuboidCfg principal_axes must be a unit (w, x, y, z) quaternion.")

    from pxr import Gf, UsdPhysics

    mass_api = UsdPhysics.MassAPI(root)
    if not mass_api:
        mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*inertia))
    mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(*axes))


def spawn_drone_cuboid(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn one authoritative rigid cuboid with an optional detailed visual-only drone mesh.

    The root and central cuboid remain the only rigid body and collider. When the
    installed Newton package supplies its ``crazyflie.usd`` example asset, that
    unmaterialed, visual-only mesh is referenced, reoriented from Y-up to Z-up,
    scaled to the configured rotor sites, and colored for Newton GL. This task does
    not vendor the binary or assert provenance beyond its presence in the installed
    Newton distribution. A procedural visual-only drone is authored if the optional
    asset is absent or gains any rigid-physics schemas.
    """
    from pxr import UsdGeom

    root = _spawn_cuboid_with_color(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    stage = get_current_stage()
    root_path = str(root.GetPath())
    visuals_path = f"{root_path}/visuals"
    UsdGeom.Xform.Define(stage, visuals_path)
    asset_path = _resolve_crazyflie_asset()
    if asset_path is None or not _spawn_crazyflie_visuals(visuals_path, cfg, asset_path):
        _spawn_procedural_drone_visuals(visuals_path, cfg)

    # Keep the conservative collider active but out of both Kit and Newton-GL
    # render geometry; the detailed or procedural children provide visuals.
    collision_mesh = UsdGeom.Imageable(stage.GetPrimAtPath(f"{root_path}/geometry/mesh"))
    if collision_mesh:
        collision_mesh.MakeInvisible()
    _author_explicit_inertia(root, cfg)
    return root


def spawn_sphere_with_color(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn a sphere and author Newton-visible ``displayColor`` on the mesh."""
    from isaaclab.sim.spawners.shapes.shapes import spawn_sphere

    prim = spawn_sphere(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    _set_mesh_display_color(str(prim.GetPath()), getattr(cfg, "display_color", None))
    return prim


def spawn_cable_with_color(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn a cable and author Newton-visible ``displayColor`` on the curve."""
    from isaaclab.sim.spawners.shapes.shapes import spawn_cable

    prim = spawn_cable(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    _set_mesh_display_color(str(prim.GetPath()), getattr(cfg, "display_color", None))
    return prim


def _attachment_relative_target(path: str) -> str:
    """Rewrite an xform-relative target so it is valid from the child attachment prim."""
    if path.startswith("/"):
        return path
    return f"../{path}"


@configclass
class DroneCuboidCfg(CuboidCfg):
    """One rigid cuboid/collider plus a solver-clean visual-only quadrotor."""

    func: Callable = spawn_drone_cuboid

    display_color: tuple[float, float, float] = (0.10, 0.65, 0.95)
    """Newton-visible central-body color."""

    arm_length: float = ROTOR_ARM_LENGTH
    """Rotor x/y coordinate from the body center [m]."""

    arm_width: float = 0.008
    """Visual arm width [m]."""

    arm_height: float = 0.006
    """Visual arm height [m]."""

    arm_color: tuple[float, float, float] = (0.05, 0.07, 0.09)
    """Visual arm color."""

    rotor_radius: float = 0.028
    """Visual rotor-disc radius [m]."""

    rotor_height: float = 0.004
    """Visual rotor-disc thickness [m]."""

    rotor_z: float = ROTOR_HEIGHT
    """Visual rotor-disc center height in the drone frame [m]."""

    rotor_colors: tuple[tuple[float, float, float], ...] = (
        (0.95, 0.25, 0.10),
        (0.18, 0.22, 0.26),
        (0.18, 0.22, 0.26),
        (0.95, 0.25, 0.10),
    )
    """Per-rotor marker colors in allocation order."""

    diagonal_inertia: tuple[float, float, float] | None = None
    """Optional principal moments ``(Ixx, Iyy, Izz)`` [kg m^2].

    Values must be finite, positive, and satisfy the rigid-body triangle
    inequalities. When set, they override collider-inferred inertia through
    ``UsdPhysics.MassAPI`` and are consumed by both Isaac Sim and Newton.
    """

    principal_axes: tuple[float, float, float, float] | None = None
    """Optional unit principal-axes quaternion in Isaac Lab ``(w, x, y, z)`` order.

    Requires :attr:`diagonal_inertia`; an omitted value means identity axes.
    """


@configclass
class PhysicsAttachmentCfg(SpawnerCfg):
    """Spawn a Newton-importable cable-to-rigid ``PhysicsAttachment``."""

    func: Callable = spawn_physics_attachment

    src0: str = ""
    """Relationship target for the cable curve, relative to the spawned xform."""

    src1: str = ""
    """Relationship target for the rigid xform, relative to the spawned xform."""

    type0: str = "point"
    """Attachment type on the cable. Newton ball joints use ``point``."""

    type1: str = "xform"
    """Attachment type on the rigid body. ``xform`` lowers to a position-only ball joint."""

    indices0: Sequence[int] = (0,)
    """Cable control-point indices. ``0`` is the first point, ``-1`` is not accepted; use the last index."""

    indices1: Sequence[int] | None = None
    """Must stay unset for ``type1='xform'``."""

    coords0: Sequence[tuple[float, float, float]] | None = None
    """Optional cable-local coordinates. Unused for point attachments."""

    coords1: Sequence[tuple[float, float, float]] | None = None
    """Anchor coordinates in the rigid body's local frame [m]."""
