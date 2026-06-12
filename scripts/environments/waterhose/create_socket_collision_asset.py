# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extract the Waterhose fridge socket colliders into one standalone USD mesh.

Run this with Isaac Lab Python so Pixar USD bindings are available:

    ./isaaclab.sh -p scripts/environments/waterhose/create_socket_collision_asset.py

The generated ``socket_collision.usda`` bakes the selected Cable008 collider
fragments into one mesh under ``/SocketCollision/socket_collision_mesh``. That
makes the socket a single asset that can be referenced, disabled, selected, or
given SDF collision properties independently from the rest of the fridge.

With ``--embed-in-fridge``, the same combined mesh is also written directly into
``fridge.usda`` at ``/root/Cable008/SocketCollision/Cable008_SocketCollision``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

REPO_ROOT = Path(__file__).resolve().parents[3]
FRIDGE_DIR = REPO_ROOT / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "contrib" / "waterhose" / "assets" / "fridge"
DEFAULT_FRIDGE_USD = FRIDGE_DIR / "fridge.usda"
DEFAULT_OUTPUT_USD = FRIDGE_DIR / "socket_collision.usda"
DEFAULT_INSPECTION_USD = FRIDGE_DIR / "fridge_socket_single_inspection.usda"
DEFAULT_REPORT_CSV = FRIDGE_DIR / "socket_collision_manifest.csv"
DEFAULT_EMBED_PARENT = "/root/Cable008"
DEFAULT_EMBED_XFORM = "/root/Cable008/SocketCollision"
DEFAULT_EMBED_MESH = "/root/Cable008/SocketCollision/Cable008_SocketCollision"
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
HOLE_AXIS = (0.0, -math.sin(math.radians(20.0)), math.cos(math.radians(20.0)))

# The light-blue SocketNearby meshes from fridge_socket_inspection.usda.
DEFAULT_SOCKET_COLLIDERS = [
    "Cable008_Collider202",
    "Cable008_Collider195",
    "Cable008_Collider44",
    "Cable008_Collider158",
    "Cable008_Collider167",
    "Cable008_Collider173",
    "Cable008_Collider12",
    "Cable008_Collider102",
    "Cable008_Collider213",
    "Cable008_Collider4",
    "Cable008_Collider27",
    "Cable008_Collider227",
    "Cable008_Collider30",
    "Cable008_Collider34",
    "Cable008_Collider10",
    "Cable008_Collider177",
    "Cable008_Collider171",
    "Cable008_Collider64",
    "Cable008_Collider187",
    "Cable008_Collider136",
    "Cable008_Collider204",
    "Cable008_Collider22",
    "Cable008_Collider232",
    "Cable008_Collider225",
    "Cable008_Collider57",
    "Cable008_Collider54",
    "Cable008_Collider19",
]


def _relative_asset_path(asset_path: Path, layer_path: Path) -> str:
    return Path(os.path.relpath(asset_path.resolve(), layer_path.resolve().parent)).as_posix()


def _vec3f(value) -> Gf.Vec3f:
    return Gf.Vec3f(float(value[0]), float(value[1]), float(value[2]))


def _extent(points: list[Gf.Vec3f]) -> list[Gf.Vec3f]:
    mins = [min(point[i] for point in points) for i in range(3)]
    maxs = [max(point[i] for point in points) for i in range(3)]
    return [Gf.Vec3f(*mins), Gf.Vec3f(*maxs)]


def _set_translate(xform: UsdGeom.Xform, translate: tuple[float, float, float]) -> None:
    prim = xform.GetPrim()
    xformable = UsdGeom.Xformable(xform.GetPrim())
    xformable.ClearXformOpOrder()
    prim.RemoveProperty("xformOp:translate")
    if any(abs(value) > 0.0 for value in translate):
        xformable.AddTranslateOp().Set(Gf.Vec3d(*translate))


def apply_front_inset(points: list[Gf.Vec3f], *, axis: str, side: str, inset_m: float) -> list[Gf.Vec3f]:
    """Clamp one outer side of the socket mesh inward by a small distance."""
    if inset_m <= 0.0:
        return points

    axis_index = AXIS_INDEX[axis]
    values = [float(point[axis_index]) for point in points]
    if side == "positive":
        limit = max(values) - inset_m
        return [
            Gf.Vec3f(*(limit if i == axis_index and float(point[i]) > limit else float(point[i]) for i in range(3)))
            for point in points
        ]

    limit = min(values) + inset_m
    return [
        Gf.Vec3f(*(limit if i == axis_index and float(point[i]) < limit else float(point[i]) for i in range(3)))
        for point in points
    ]


def _get_source_mesh(stage: Usd.Stage, name: str) -> UsdGeom.Mesh:
    path = f"/root/Cable008/Collisions/{name}"
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise ValueError(f"Missing source collider prim: {path}")
    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        raise ValueError(f"Source collider prim is not a UsdGeom.Mesh: {path}")
    return mesh


def extract_mesh_data(
    fridge_usd: Path,
    collider_names: list[str],
    *,
    target_frame_path: str | None = None,
) -> tuple[list[Gf.Vec3f], list[int], list[int], list[dict[str, str]]]:
    source_stage = Usd.Stage.Open(str(fridge_usd))
    if source_stage is None:
        raise RuntimeError(f"Could not open fridge USD: {fridge_usd}")

    xform_cache = UsdGeom.XformCache()
    target_frame_inv = None
    if target_frame_path is not None:
        target_frame_prim = source_stage.GetPrimAtPath(target_frame_path)
        if not target_frame_prim:
            raise ValueError(f"Missing target frame prim: {target_frame_path}")
        target_frame_inv = xform_cache.GetLocalToWorldTransform(target_frame_prim).GetInverse()

    points: list[Gf.Vec3f] = []
    face_counts: list[int] = []
    face_indices: list[int] = []
    manifest: list[dict[str, str]] = []

    for name in collider_names:
        mesh = _get_source_mesh(source_stage, name)
        source_points = mesh.GetPointsAttr().Get() or []
        source_counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
        source_indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
        local_to_stage = xform_cache.GetLocalToWorldTransform(mesh.GetPrim())

        offset = len(points)
        for point in source_points:
            transformed = local_to_stage.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
            if target_frame_inv is not None:
                transformed = target_frame_inv.Transform(transformed)
            points.append(_vec3f(transformed))
        face_counts.extend(int(count) for count in source_counts)
        face_indices.extend(int(index) + offset for index in source_indices)

        approximation_attr = mesh.GetPrim().GetAttribute("physics:approximation")
        blender_name_attr = mesh.GetPrim().GetAttribute("userProperties:blender:object_name")
        manifest.append(
            {
                "name": name,
                "path": str(mesh.GetPath()),
                "point_count": str(len(source_points)),
                "face_count": str(len(source_counts)),
                "physics_approximation": str(approximation_attr.Get() if approximation_attr else ""),
                "blender_object_name": str(blender_name_attr.Get() if blender_name_attr else ""),
            }
        )

    return points, face_counts, face_indices, manifest


def write_socket_asset(
    output_usd: Path,
    collider_names: list[str],
    points: list[Gf.Vec3f],
    face_counts: list[int],
    face_indices: list[int],
    *,
    front_inset_m: float = 0.0,
    front_axis: str = "y",
    front_side: str = "positive",
    offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    output_usd.parent.mkdir(parents=True, exist_ok=True)
    if output_usd.exists():
        output_usd.unlink()

    stage = Usd.Stage.CreateNew(str(output_usd))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")

    root = UsdGeom.Xform.Define(stage, "/SocketCollision")
    stage.SetDefaultPrim(root.GetPrim())
    _set_translate(root, offset_m)
    root.GetPrim().CreateAttribute("socket:sourceColliderNames", Sdf.ValueTypeNames.StringArray).Set(collider_names)
    root.GetPrim().CreateAttribute("socket:frontInsetMeters", Sdf.ValueTypeNames.Double).Set(front_inset_m)
    root.GetPrim().CreateAttribute("socket:frontInsetAxis", Sdf.ValueTypeNames.String).Set(front_axis)
    root.GetPrim().CreateAttribute("socket:frontInsetSide", Sdf.ValueTypeNames.String).Set(front_side)
    root.GetPrim().CreateAttribute("socket:offsetMeters", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(*offset_m))
    root.GetPrim().CreateAttribute("socket:holeAxis", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(*HOLE_AXIS))

    mesh = UsdGeom.Mesh.Define(stage, "/SocketCollision/socket_collision_mesh")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr().Set("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr(_extent(points))

    primvars = UsdGeom.PrimvarsAPI(mesh)
    primvars.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant).Set(
        [Gf.Vec3f(0.1, 0.75, 1.0)]
    )

    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision.CreateApproximationAttr().Set("none")

    stage.GetRootLayer().Save()


def write_embedded_socket_to_fridge(
    fridge_usd: Path,
    collider_names: list[str],
    points: list[Gf.Vec3f],
    face_counts: list[int],
    face_indices: list[int],
    *,
    front_inset_m: float = 0.0,
    front_axis: str = "y",
    front_side: str = "positive",
    offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    source_collider_state: str = "preserve",
    xform_path: str = DEFAULT_EMBED_XFORM,
    mesh_path: str = DEFAULT_EMBED_MESH,
) -> None:
    stage = Usd.Stage.Open(str(fridge_usd))
    if stage is None:
        raise RuntimeError(f"Could not open fridge USD: {fridge_usd}")

    xform = UsdGeom.Xform.Define(stage, xform_path)
    _set_translate(xform, offset_m)
    xform.GetPrim().CreateAttribute("socket:sourceColliderNames", Sdf.ValueTypeNames.StringArray).Set(collider_names)
    xform.GetPrim().CreateAttribute("socket:description", Sdf.ValueTypeNames.String).Set(
        "Combined selectable socket collision mesh generated from Cable008 collider fragments."
    )
    xform.GetPrim().CreateAttribute("socket:frontInsetMeters", Sdf.ValueTypeNames.Double).Set(front_inset_m)
    xform.GetPrim().CreateAttribute("socket:frontInsetAxis", Sdf.ValueTypeNames.String).Set(front_axis)
    xform.GetPrim().CreateAttribute("socket:frontInsetSide", Sdf.ValueTypeNames.String).Set(front_side)
    xform.GetPrim().CreateAttribute("socket:offsetMeters", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(*offset_m))
    xform.GetPrim().CreateAttribute("socket:holeAxis", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(*HOLE_AXIS))
    xform.GetPrim().CreateAttribute("socket:sourceColliderState", Sdf.ValueTypeNames.String).Set(source_collider_state)
    xform.GetPrim().CreateAttribute("visibility", Sdf.ValueTypeNames.Token).Set("inherited")
    xform.GetPrim().CreateAttribute("purpose", Sdf.ValueTypeNames.Token).Set("default")

    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr().Set("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr(_extent(points))
    mesh.GetPrim().CreateAttribute("visibility", Sdf.ValueTypeNames.Token).Set("inherited")
    mesh.GetPrim().CreateAttribute("purpose", Sdf.ValueTypeNames.Token).Set("default")

    primvars = UsdGeom.PrimvarsAPI(mesh)
    primvars.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant).Set(
        [Gf.Vec3f(0.1, 0.75, 1.0)]
    )

    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision.CreateApproximationAttr().Set("none")

    if source_collider_state != "preserve":
        active = source_collider_state == "active"
        for name in collider_names:
            source_prim = stage.GetPrimAtPath(f"/root/Cable008/Collisions/{name}")
            if not source_prim:
                raise ValueError(f"Missing source collider prim while setting active state: {name}")
            source_prim.SetActive(active)

    stage.GetRootLayer().Save()


def write_manifest(report_csv: Path, manifest: list[dict[str, str]]) -> None:
    with report_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["name", "path", "point_count", "face_count", "physics_approximation", "blender_object_name"],
        )
        writer.writeheader()
        writer.writerows(manifest)


def write_single_inspection_layer(
    inspection_usd: Path,
    fridge_usd: Path,
    socket_usd: Path,
    collider_names: list[str],
) -> None:
    fridge_ref = _relative_asset_path(fridge_usd, inspection_usd)
    socket_ref = _relative_asset_path(socket_usd, inspection_usd)
    inactive_overrides = "\n".join(f'                over "{name}" (active = false) {{}}' for name in collider_names)

    content = f"""#usda 1.0
(
    defaultPrim = "Inspection"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Inspection"
{{
    def Xform "Fridge" (
        references = @{fridge_ref}@</root>
    )
    {{
        over "Cable008"
        {{
            over "Collisions"
            {{
{inactive_overrides}
            }}
        }}
    }}

    def Xform "SocketCollision" (
        references = @{socket_ref}@</SocketCollision>
    )
    {{
    }}
}}
"""
    inspection_usd.write_text(content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fridge-usd", type=Path, default=DEFAULT_FRIDGE_USD)
    parser.add_argument("--output-usd", type=Path, default=DEFAULT_OUTPUT_USD)
    parser.add_argument("--inspection-usd", type=Path, default=DEFAULT_INSPECTION_USD)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument(
        "--embed-in-fridge",
        action="store_true",
        help=f"Also write the combined mesh into fridge.usda at {DEFAULT_EMBED_MESH}.",
    )
    parser.add_argument(
        "--front-inset-mm",
        type=float,
        default=0.0,
        help="Clamp the socket mouth/front side inward by this many millimeters.",
    )
    parser.add_argument(
        "--front-axis",
        choices=sorted(AXIS_INDEX),
        default="y",
        help="Axis used for the socket mouth/front inset. The fridge socket front is currently positive Y.",
    )
    parser.add_argument(
        "--front-side",
        choices=["positive", "negative"],
        default="positive",
        help="Which side of the front axis to clamp inward.",
    )
    parser.add_argument(
        "--offset-mm",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="Rigid local translation applied to the combined socket Xform, in millimeters.",
    )
    parser.add_argument(
        "--offset-along-hole-mm",
        type=float,
        default=0.0,
        help=(
            "Additional rigid translation along the socket insertion axis in millimeters. "
            f"The hole axis is {HOLE_AXIS}."
        ),
    )
    parser.add_argument(
        "--source-collider-state",
        choices=["preserve", "active", "inactive"],
        default="preserve",
        help=(
            "Whether to preserve, enable, or disable the original collider fragments "
            "after embedding the combined socket mesh."
        ),
    )
    parser.add_argument(
        "--collider",
        dest="colliders",
        action="append",
        help="Collider prim name to include. Repeat to override the default light-blue socket set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collider_names = args.colliders or DEFAULT_SOCKET_COLLIDERS
    front_inset_m = args.front_inset_mm * 0.001
    base_offset_m = tuple(value * 0.001 for value in args.offset_mm)
    hole_offset_m = args.offset_along_hole_mm * 0.001
    offset_m = tuple(base_offset_m[i] + hole_offset_m * HOLE_AXIS[i] for i in range(3))
    points, face_counts, face_indices, manifest = extract_mesh_data(args.fridge_usd, collider_names)
    points = apply_front_inset(points, axis=args.front_axis, side=args.front_side, inset_m=front_inset_m)
    write_socket_asset(
        args.output_usd,
        collider_names,
        points,
        face_counts,
        face_indices,
        front_inset_m=front_inset_m,
        front_axis=args.front_axis,
        front_side=args.front_side,
        offset_m=offset_m,
    )
    write_manifest(args.report_csv, manifest)
    write_single_inspection_layer(args.inspection_usd, args.fridge_usd, args.output_usd, collider_names)
    if args.embed_in_fridge:
        embedded_points, embedded_counts, embedded_indices, _ = extract_mesh_data(
            args.fridge_usd, collider_names, target_frame_path=DEFAULT_EMBED_PARENT
        )
        embedded_points = apply_front_inset(
            embedded_points, axis=args.front_axis, side=args.front_side, inset_m=front_inset_m
        )
        write_embedded_socket_to_fridge(
            args.fridge_usd,
            collider_names,
            embedded_points,
            embedded_counts,
            embedded_indices,
            front_inset_m=front_inset_m,
            front_axis=args.front_axis,
            front_side=args.front_side,
            offset_m=offset_m,
            source_collider_state=args.source_collider_state,
        )

    print(f"Wrote socket asset: {args.output_usd}")
    print(f"Wrote inspection layer: {args.inspection_usd}")
    print(f"Wrote manifest: {args.report_csv}")
    if args.embed_in_fridge:
        print(f"Embedded socket mesh into fridge: {args.fridge_usd}")
        print(f"Embedded prim: {DEFAULT_EMBED_MESH}")
        if args.source_collider_state != "preserve":
            print(f"Set source collider fragments to: {args.source_collider_state}")
        if front_inset_m > 0.0:
            print(f"Applied front inset: {args.front_inset_mm:g} mm along {args.front_side} {args.front_axis.upper()}")
        if any(abs(value) > 0.0 for value in offset_m):
            offset_mm = tuple(value * 1000.0 for value in offset_m)
            print(f"Applied total rigid offset: ({offset_mm[0]:g}, {offset_mm[1]:g}, {offset_mm[2]:g}) mm")
        if args.offset_along_hole_mm:
            print(f"Applied hole-axis offset: {args.offset_along_hole_mm:g} mm along {HOLE_AXIS}")
    print(f"Combined {len(collider_names)} source colliders into one mesh.")
    print("Open with:")
    print(f"  ./isaaclab.sh -s {args.inspection_usd}")


if __name__ == "__main__":
    main()
