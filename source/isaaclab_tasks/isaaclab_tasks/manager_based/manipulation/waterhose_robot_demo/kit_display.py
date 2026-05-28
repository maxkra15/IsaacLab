# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit display authoring for the waterhose robot demo."""

from __future__ import annotations

import os
import re
import time
from typing import Any

import numpy as np


DISPLAY_ROOT_PATH = "/World/WaterhoseDemo/Dynamic"
STATIC_SCENE_ROOT_PATH = "/World/WaterhoseDemo/StaticScenes"
LEGACY_STATIC_SCENE_PATH = "/World/WaterhoseDemo/Static"


def assign_display_prim_paths(builder: Any, root_path: str = DISPLAY_ROOT_PATH) -> None:
    """Assign USD prim paths to Newton display body and shape labels."""

    used_paths: set[str] = set()
    body_paths = []
    for body_id, label in enumerate(builder.body_label):
        body_paths.append(_unique_path(f"{root_path}/Bodies/body_{body_id:03d}_{_label_token(label, 'body')}", used_paths))

    world_shape_root = f"{root_path}/WorldShapes"
    shape_paths = []
    for shape_id, label in enumerate(builder.shape_label):
        body_id = int(builder.shape_body[shape_id])
        parent_path = body_paths[body_id] if 0 <= body_id < len(body_paths) else world_shape_root
        shape_paths.append(_unique_path(f"{parent_path}/shape_{shape_id:03d}_{_label_token(label, 'shape')}", used_paths))

    builder.body_label[:] = body_paths
    builder.shape_label[:] = shape_paths


def author_display_usd(
    model: Any,
    root_path: str = DISPLAY_ROOT_PATH,
    skipped_shape_ids: set[int] | None = None,
) -> None:
    """Author the combined Newton display model into the current IsaacLab USD stage."""

    import newton  # noqa: PLC0415
    from isaaclab.sim import SimulationContext  # noqa: PLC0415
    from isaaclab.sim.utils.stage import get_current_stage  # noqa: PLC0415
    from pxr import Gf, UsdGeom, Vt  # noqa: PLC0415

    sim = SimulationContext.instance()
    stage = getattr(sim, "stage", None) if sim is not None else get_current_stage()
    if stage is None:
        raise RuntimeError("Cannot author waterhose Kit display: no USD stage is available.")

    start = time.perf_counter()
    _debug(
        "author_display_usd:start "
        f"bodies={getattr(model, 'body_count', 0)} shapes={getattr(model, 'shape_count', 0)} "
        f"skipped_shapes={len(skipped_shape_ids or set())}"
    )
    if stage.GetPrimAtPath(root_path).IsValid():
        stage.RemovePrim(root_path)
    _define_xform_path(stage, root_path, UsdGeom)
    UsdGeom.Xform.Define(stage, f"{root_path}/Bodies")
    UsdGeom.Xform.Define(stage, f"{root_path}/WorldShapes")

    body_paths = list(getattr(model, "body_label", []) or [])
    for body_path in body_paths:
        if isinstance(body_path, str) and body_path.startswith("/"):
            UsdGeom.Xform.Define(stage, body_path)

    _debug(f"author_display_usd:bodies dt={time.perf_counter() - start:.3f}s")
    _author_shapes(stage, model, newton, Gf, UsdGeom, Vt, skipped_shape_ids or set())
    _debug(f"author_display_usd:done dt={time.perf_counter() - start:.3f}s")


def author_static_scene_references(
    scene_usd_path: str,
    env_origins: Any,
    fridge_xform: Any,
    root_path: str = STATIC_SCENE_ROOT_PATH,
) -> None:
    """Author one referenced static fridge scene per environment for Kit visualization."""

    from isaaclab.sim import SimulationContext  # noqa: PLC0415
    from isaaclab.sim.utils.stage import get_current_stage  # noqa: PLC0415
    from pxr import Gf, UsdGeom  # noqa: PLC0415

    sim = SimulationContext.instance()
    stage = getattr(sim, "stage", None) if sim is not None else get_current_stage()
    if stage is None:
        raise RuntimeError("Cannot author waterhose static scene: no USD stage is available.")

    origins = np.asarray(env_origins, dtype=np.float64).reshape(-1, 3)
    if origins.size == 0:
        return

    start = time.perf_counter()
    if stage.GetPrimAtPath(LEGACY_STATIC_SCENE_PATH).IsValid():
        stage.RemovePrim(LEGACY_STATIC_SCENE_PATH)
    if stage.GetPrimAtPath(root_path).IsValid():
        stage.RemovePrim(root_path)

    _define_xform_path(stage, root_path, UsdGeom)
    base_transform = np.asarray(fridge_xform, dtype=np.float64).reshape(7)
    for env_id, origin in enumerate(origins):
        transform = base_transform.copy()
        transform[:3] += origin
        prim_path = f"{root_path}/env_{env_id}"
        prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
        prim.GetReferences().AddReference(scene_usd_path)
        prim.SetInstanceable(True)
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "waterhose_static").Set(
            _matrix_from_transform(transform, Gf)
        )

    _debug(
        "author_static_scene_references:done "
        f"envs={len(origins)} path={scene_usd_path} dt={time.perf_counter() - start:.3f}s"
    )


def _author_shapes(
    stage: Any,
    model: Any,
    newton: Any,
    Gf: Any,
    UsdGeom: Any,
    Vt: Any,
    skipped_shape_ids: set[int],
) -> None:
    shape_flags = _array_np(getattr(model, "shape_flags", None))
    shape_types = _array_np(getattr(model, "shape_type", None))
    shape_scales = _array_np(getattr(model, "shape_scale", None))
    shape_transforms = _array_np(getattr(model, "shape_transform", None))
    shape_labels = list(getattr(model, "shape_label", []) or [])
    shape_sources = list(getattr(model, "shape_source", []) or [])
    shape_colors = _array_np(getattr(model, "shape_color", None))

    if shape_types is None or shape_scales is None or shape_transforms is None:
        return

    visible_flag = int(newton.ShapeFlags.VISIBLE)
    authored = 0
    skipped = 0
    vertices_total = 0
    triangles_total = 0
    start = time.perf_counter()
    for shape_id, shape_type in enumerate(shape_types.tolist()):
        if shape_id in skipped_shape_ids:
            skipped += 1
            continue
        if shape_flags is not None and not (int(shape_flags[shape_id]) & visible_flag):
            continue
        if shape_id >= len(shape_labels):
            continue
        shape_path = shape_labels[shape_id]
        if not isinstance(shape_path, str) or not shape_path.startswith("/"):
            continue

        mesh_data = _mesh_for_shape(
            newton=newton,
            shape_type=int(shape_type),
            scale=np.asarray(shape_scales[shape_id], dtype=np.float32),
            source=shape_sources[shape_id] if shape_id < len(shape_sources) else None,
        )
        if mesh_data is None:
            continue

        vertices, indices, normals = mesh_data
        if vertices.size == 0 or indices.size == 0:
            continue

        vertices_total += int(vertices.shape[0])
        triangles_total += int(indices.size // 3)
        authored += 1
        if authored == 1 or authored % 25 == 0:
            _debug(
                "author_display_usd:shape "
                f"count={authored} shape_id={shape_id} verts={vertices_total} tris={triangles_total} "
                f"dt={time.perf_counter() - start:.3f}s"
            )

        mesh = UsdGeom.Mesh.Define(stage, shape_path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32, copy=False)))
        mesh.CreateFaceVertexCountsAttr([3] * int(indices.size // 3))
        mesh.CreateFaceVertexIndicesAttr(indices.astype(np.int32, copy=False).reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        if normals is not None and normals.shape == vertices.shape:
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32, copy=False)))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        color = np.asarray(shape_colors[shape_id] if shape_colors is not None else (0.55, 0.55, 0.55), dtype=float)
        color = np.clip(color[:3], 0.0, 1.0)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])

        xformable = UsdGeom.Xformable(mesh.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "waterhose_shape").Set(
            _matrix_from_transform(shape_transforms[shape_id], Gf)
        )
    _debug(
        "author_display_usd:shapes_done "
        f"authored={authored} skipped={skipped} verts={vertices_total} tris={triangles_total} "
        f"dt={time.perf_counter() - start:.3f}s"
    )


def _mesh_for_shape(newton: Any, shape_type: int, scale: np.ndarray, source: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    geo_type = newton.GeoType(shape_type)
    mesh = None
    if geo_type in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH):
        mesh = source if isinstance(source, newton.Mesh) else None
    elif geo_type == newton.GeoType.SPHERE:
        mesh = newton.Mesh.create_sphere(float(scale[0]), num_latitudes=16, num_longitudes=24, compute_inertia=False)
        scale = np.ones(3, dtype=np.float32)
    elif geo_type == newton.GeoType.ELLIPSOID:
        mesh = newton.Mesh.create_ellipsoid(float(scale[0]), float(scale[1]), float(scale[2]), compute_inertia=False)
        scale = np.ones(3, dtype=np.float32)
    elif geo_type == newton.GeoType.CAPSULE:
        mesh = newton.Mesh.create_capsule(
            float(scale[0]),
            float(scale[1]),
            up_axis=newton.Axis.Z,
            segments=24,
            compute_inertia=False,
        )
        scale = np.ones(3, dtype=np.float32)
    elif geo_type == newton.GeoType.CYLINDER:
        mesh = newton.Mesh.create_cylinder(
            float(scale[0]),
            float(scale[1]),
            up_axis=newton.Axis.Z,
            segments=24,
            compute_inertia=False,
        )
        scale = np.ones(3, dtype=np.float32)
    elif geo_type == newton.GeoType.CONE:
        mesh = newton.Mesh.create_cone(
            float(scale[0]),
            float(scale[1]),
            up_axis=newton.Axis.Z,
            segments=24,
            compute_inertia=False,
        )
        scale = np.ones(3, dtype=np.float32)
    elif geo_type == newton.GeoType.BOX:
        mesh = newton.Mesh.create_box(float(scale[0]), float(scale[1]), float(scale[2]), compute_inertia=False)
        scale = np.ones(3, dtype=np.float32)

    if mesh is None:
        return None

    vertices = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3).copy()
    indices = np.asarray(mesh.indices, dtype=np.int32).reshape(-1).copy()
    normals = None if mesh.normals is None else np.asarray(mesh.normals, dtype=np.float32).reshape(-1, 3).copy()
    if not np.allclose(scale, np.ones(3, dtype=np.float32)):
        vertices *= scale.reshape(1, 3)
        normals = None
    return vertices, indices, normals


def _matrix_from_transform(transform: Any, Gf: Any) -> Any:
    values = np.asarray(transform, dtype=np.float64).reshape(7)
    pos = values[:3]
    quat = values[3:7]
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        quat = quat / norm

    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotateOnly(Gf.Quatd(float(quat[3]), Gf.Vec3d(float(quat[0]), float(quat[1]), float(quat[2]))))
    matrix.SetTranslateOnly(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    return matrix


def _array_np(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def _define_xform_path(stage: Any, path: str, UsdGeom: Any) -> None:
    current = ""
    for token in path.split("/"):
        if not token:
            continue
        current = f"{current}/{token}"
        UsdGeom.Xform.Define(stage, current)


def _debug(message: str) -> None:
    if os.getenv("WATERHOSE_DEBUG_KIT_DISPLAY", "").lower() in {"1", "true", "yes", "on"}:
        print(f"[waterhose-kit] {message}", flush=True)


def _label_token(label: Any, fallback: str) -> str:
    text = str(label or fallback).strip().split("/")[-1]
    token = re.sub(r"[^A-Za-z0-9_]", "_", text)
    token = re.sub(r"_+", "_", token).strip("_")
    if not token:
        token = fallback
    if token[0].isdigit():
        token = f"{fallback}_{token}"
    return token[:64]


def _unique_path(path: str, used_paths: set[str]) -> str:
    if path not in used_paths:
        used_paths.add(path)
        return path
    index = 1
    while f"{path}_{index}" in used_paths:
        index += 1
    unique = f"{path}_{index}"
    used_paths.add(unique)
    return unique
