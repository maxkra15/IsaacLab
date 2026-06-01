# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD visualization support for the waterhose robot demo.

The physics model is generated in Newton, while the IsaacLab scene config
authors the visible robot, fridge, and cable USDs under normal env prim paths.
This module only provides the bridge: it relabels Newton bodies to matching USD
prims and updates the rendered BasisCurves from simulated cable segments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .coupled_builder import WaterhoseAssetPaths, WaterhoseCoupledBuildInfo


_CABLE_CURVE_PRIMS = ("cable001/curve_0", "cable002/curve_0")
_PLUG_VISUAL_PRIMS = ("plug_mesh/plug_mesh", "plug_mesh02/plug_mesh")


def prepare_usd_scene_for_newton_sync(
    *,
    sim: Any,
    builder: Any,
    build_info: WaterhoseCoupledBuildInfo,
    asset_root: str | Path,
) -> None:
    """Spawn USD visuals and relabel Newton bodies to Kit prim paths."""

    import isaaclab.sim as sim_utils  # noqa: PLC0415

    stage = sim_utils.get_current_stage()
    if stage is None:
        return

    # Kit uses the authored USD assets below. The Newton color back-fill is meant
    # for generated debug geometry and can scan many irrelevant stage prims here.
    os.environ.setdefault("ISAACLAB_REPLACE_NEWTON_SHAPE_COLORS", "0")

    paths = WaterhoseAssetPaths.from_root(asset_root)
    for env_id in range(max(1, int(build_info.num_envs))):
        _ensure_visual_assets(stage, env_id, paths)

    _disable_physics(stage, "/World/envs")
    _relabel_robot_bodies(stage, builder, build_info)
    _relabel_scene_bodies(stage, builder, build_info)
    _relabel_plug_bodies(stage, builder, build_info)
    _define_missing_body_prims(stage, builder, build_info)
    _install_cable_curve_sync(build_info)
    _configure_camera(sim)


def sync_kit_cable_curves_from_newton(build_info: WaterhoseCoupledBuildInfo) -> None:
    """Mirror simulated Newton cable centerlines into Kit BasisCurves."""

    try:
        import isaaclab.sim as sim_utils  # noqa: PLC0415
        from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415
    except Exception:
        return

    stage = sim_utils.get_current_stage()
    if stage is None or NewtonManager.get_state_0() is None:
        return

    try:
        body_q = NewtonManager.get_state_0().body_q.numpy()
    except Exception:
        return

    curves = build_info.cable_body_ids_by_curve
    lengths = build_info.cable_segment_lengths_by_curve
    if not curves:
        return
    curves_per_env = max(1, len(curves) // max(1, int(build_info.num_envs)))
    for curve_index, body_ids in enumerate(curves):
        if not body_ids:
            continue
        env_id = min(curve_index // curves_per_env, max(1, int(build_info.num_envs)) - 1)
        local_curve_id = curve_index % len(_CABLE_CURVE_PRIMS)
        curve_path = f"{_cable_visual_root(env_id)}/{_CABLE_CURVE_PRIMS[local_curve_id]}"
        prim = stage.GetPrimAtPath(curve_path)
        if not prim.IsValid():
            continue
        segment_lengths = lengths[curve_index] if curve_index < len(lengths) else []
        points = _cable_points_from_body_q(body_q, body_ids, segment_lengths)
        if points is None:
            continue
        _set_basis_curve_points(prim, points, radius=0.003)


def _install_cable_curve_sync(build_info: WaterhoseCoupledBuildInfo) -> None:
    try:
        from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415
    except Exception:
        return
    register = getattr(NewtonManager, "register_pre_render_callback", None)
    if register is None:
        return

    callback_name = f"waterhose_demo_kit_curves_{id(build_info)}"

    def _sync() -> None:
        sync_kit_cable_curves_from_newton(build_info)

    register(callback_name, _sync)


def _spawn_usd_once(
    stage: Any,
    prim_path: str,
    usd_path: str,
    *,
    translation: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
    variants: dict[str, str] | None = None,
) -> None:
    if stage.GetPrimAtPath(prim_path).IsValid():
        return
    import isaaclab.sim as sim_utils  # noqa: PLC0415

    parent = prim_path.rsplit("/", 1)[0]
    if parent:
        _define_xform_path(stage, parent)
    cfg = sim_utils.UsdFileCfg(usd_path=usd_path, variants=variants)
    cfg.func(prim_path, cfg, translation=translation, orientation=orientation)
    _debug(f"spawned {prim_path} <- {usd_path}")


def _ensure_visual_assets(stage: Any, env_id: int, paths: WaterhoseAssetPaths) -> None:
    """Fallback for direct manager tests that bypass the scene config."""

    env_root = _env_root(env_id)
    _define_xform_path(stage, env_root)
    _spawn_usd_once(
        stage,
        f"{env_root}/Robot",
        str(paths.robot_usd),
        translation=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
    )
    fridge_pos, fridge_quat = _fridge_transform()
    _spawn_usd_once(
        stage,
        f"{env_root}/Fridge",
        str(paths.fridge_usd),
        translation=fridge_pos,
        orientation=fridge_quat,
    )
    _spawn_usd_once(
        stage,
        _cable_visual_root(env_id),
        str(paths.cable_usd),
        translation=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
    )


def _relabel_robot_bodies(stage: Any, builder: Any, build_info: WaterhoseCoupledBuildInfo) -> None:
    robot_ids = list(build_info.robot_body_ids)
    if not robot_ids:
        return
    bodies_per_env = max(1, int(build_info.robot_body_count))
    for index, body_id in enumerate(robot_ids):
        env_id = min(index // bodies_per_env, max(1, int(build_info.num_envs)) - 1)
        body_name = str(builder.body_label[body_id]).rsplit("/", 1)[-1]
        builder.body_label[body_id] = _resolve_body_prim_path(stage, f"{_env_root(env_id)}/Robot", body_name)


def _relabel_scene_bodies(stage: Any, builder: Any, build_info: WaterhoseCoupledBuildInfo) -> None:
    # Static scene visuals are already spawned as the fridge USD. Relabel the
    # few scene bodies that have matching prim names so Fabric sync can keep them
    # on the authored visual asset if their transforms are ever reset.
    vbd_ids = list(build_info.vbd_body_ids)
    if not vbd_ids:
        return
    vbd_per_env = max(1, int(build_info.vbd_body_count))
    for index, body_id in enumerate(vbd_ids):
        label = str(builder.body_label[body_id])
        if "Cable008" not in label:
            continue
        env_id = min(index // vbd_per_env, max(1, int(build_info.num_envs)) - 1)
        body_name = label.rsplit("/", 1)[-1]
        builder.body_label[body_id] = _resolve_body_prim_path(stage, f"{_env_root(env_id)}/Fridge", body_name)


def _relabel_plug_bodies(stage: Any, builder: Any, build_info: WaterhoseCoupledBuildInfo) -> None:
    head_curves = build_info.cable_head_body_ids_by_curve
    if not head_curves:
        return
    curves_per_env = max(1, len(head_curves) // max(1, int(build_info.num_envs)))
    for curve_index, head_ids in enumerate(head_curves):
        if not head_ids:
            continue
        env_id = min(curve_index // curves_per_env, max(1, int(build_info.num_envs)) - 1)
        visual = _PLUG_VISUAL_PRIMS[curve_index % len(_PLUG_VISUAL_PRIMS)]
        visual_path = f"{_cable_visual_root(env_id)}/{visual}"
        if not stage.GetPrimAtPath(visual_path).IsValid():
            continue
        for body_id in head_ids:
            builder.body_label[int(body_id)] = visual_path


def _resolve_body_prim_path(stage: Any, root_path: str, body_name: str) -> str:
    from pxr import Usd  # noqa: PLC0415

    direct = f"{root_path}/{body_name}"
    if stage.GetPrimAtPath(direct).IsValid():
        return direct
    root = stage.GetPrimAtPath(root_path)
    if root.IsValid():
        for prim in Usd.PrimRange(root):
            if prim.GetName() == body_name:
                return str(prim.GetPath())
    _define_xform_path(stage, direct)
    return direct


def _define_missing_body_prims(stage: Any, builder: Any, build_info: WaterhoseCoupledBuildInfo) -> None:
    """Create empty Xforms for Newton bodies without authored render geometry."""

    env_body_count = max(1, int(build_info.env_body_count or len(builder.body_label)))
    num_envs = max(1, int(build_info.num_envs))
    for env_id in range(num_envs):
        fallback_root = f"{_env_root(env_id)}/NewtonBodies"
        _define_xform_path(stage, fallback_root)
        _set_visibility(stage, fallback_root, visible=False)

    for body_id, label in enumerate(list(builder.body_label)):
        body_label = str(label)
        if body_label.startswith("/") and stage.GetPrimAtPath(body_label).IsValid():
            continue

        env_id = min(body_id // env_body_count, num_envs - 1)
        body_name = body_label.rsplit("/", 1)[-1] if body_label else f"body_{body_id}"
        prim_name = _safe_prim_name(body_name)
        prim_path = f"{_env_root(env_id)}/NewtonBodies/{prim_name}"
        if stage.GetPrimAtPath(prim_path).IsValid():
            prim_path = f"{_env_root(env_id)}/NewtonBodies/{prim_name}_{body_id:04d}"
        builder.body_label[body_id] = prim_path
        _define_xform_path(stage, prim_path)


def _disable_physics(stage: Any, root_path: str) -> None:
    from pxr import Sdf, Usd  # noqa: PLC0415

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return
    bool_type = Sdf.ValueTypeNames.Bool
    for prim in Usd.PrimRange(root):
        schemas = set(prim.GetAppliedSchemas())
        if "PhysicsRigidBodyAPI" in schemas:
            prim.CreateAttribute("physics:rigidBodyEnabled", bool_type, False).Set(False)
            prim.CreateAttribute("physics:kinematicEnabled", bool_type, False).Set(True)
        if "PhysicsCollisionAPI" in schemas:
            prim.CreateAttribute("physics:collisionEnabled", bool_type, False).Set(False)
        if "PhysicsArticulationRootAPI" in schemas or "PhysxArticulationAPI" in schemas:
            prim.CreateAttribute("physxArticulation:articulationEnabled", bool_type, False).Set(False)


def _cable_points_from_body_q(
    body_q: np.ndarray, body_ids: list[int], segment_lengths: list[float]
) -> np.ndarray | None:
    body_indices = np.asarray(body_ids, dtype=np.int64)
    if body_indices.size == 0 or np.any(body_indices < 0) or np.any(body_indices >= body_q.shape[0]):
        return None
    poses = np.asarray(body_q[body_indices], dtype=np.float64)
    lengths = np.asarray(segment_lengths, dtype=np.float64)
    if lengths.shape[0] != poses.shape[0]:
        if lengths.shape[0] > poses.shape[0]:
            lengths = lengths[: poses.shape[0]]
        elif lengths.shape[0] > 0:
            lengths = np.pad(lengths, (0, poses.shape[0] - lengths.shape[0]), mode="edge")
        else:
            lengths = np.full(poses.shape[0], 0.01, dtype=np.float64)

    points = np.empty((poses.shape[0] + 1, 3), dtype=np.float32)
    points[:-1] = poses[:, :3].astype(np.float32)
    points[-1] = (
        poses[-1, :3]
        + _quat_rotate_xyzw(poses[-1, 3:7], np.array([0.0, 0.0, float(lengths[-1])], dtype=np.float64))
    ).astype(np.float32)
    if not np.all(np.isfinite(points)) or float(np.max(np.abs(points))) > 50.0:
        return None
    return points


def _set_basis_curve_points(prim: Any, points: np.ndarray, radius: float) -> None:
    from pxr import Gf, UsdGeom, Vt  # noqa: PLC0415

    curve = UsdGeom.BasisCurves(prim)
    if not curve:
        return
    points = np.ascontiguousarray(points, dtype=np.float32)
    curve_points = Vt.Vec3fArray([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])
    curve.GetPointsAttr().Set(curve_points)
    curve.GetCurveVertexCountsAttr().Set(Vt.IntArray([int(points.shape[0])]))
    curve.GetWidthsAttr().Set(Vt.FloatArray([float(2.0 * radius)] * int(points.shape[0])))


def _define_xform_path(stage: Any, prim_path: str) -> None:
    from pxr import UsdGeom  # noqa: PLC0415

    current = ""
    for token in prim_path.strip("/").split("/"):
        if not token:
            continue
        current = f"{current}/{token}"
        if not stage.GetPrimAtPath(current).IsValid():
            UsdGeom.Xform.Define(stage, current)


def _set_visibility(stage: Any, prim_path: str, *, visible: bool) -> None:
    from pxr import UsdGeom  # noqa: PLC0415

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


def _safe_prim_name(name: str) -> str:
    safe = "".join(char if char.isalnum() or char == "_" else "_" for char in name)
    return safe or "body"


def _env_root(env_id: int) -> str:
    return f"/World/envs/env_{int(env_id)}"


def _cable_visual_root(env_id: int) -> str:
    return f"{_env_root(env_id)}/WaterhoseCableCurves"


def _fridge_transform() -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    table_half_z = 0.5 * (0.6 - 0.215)
    table_top_z = 2.0 * table_half_z
    pos = (
        0.95,
        (0.293 - 0.395) / 2.0,
        0.902 + table_top_z,
    )
    s = float(np.sin(np.pi / 4.0))
    c = float(np.cos(np.pi / 4.0))
    return pos, (0.0, 0.0, s, c)


def _quat_rotate_xyzw(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return v
    q = q / norm
    q_xyz = q[:3]
    q_w = float(q[3])
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def _configure_camera(sim: Any) -> None:
    if "kit" not in sim.resolve_visualizer_types():
        return
    for visualizer in getattr(sim, "visualizers", ()):
        viewport_api = getattr(visualizer, "_viewport_api", None)
        if viewport_api is not None:
            try:
                viewport_api.set_active_camera("/OmniverseKit_Persp")
            except Exception:
                pass


def _debug(message: str) -> None:
    if os.getenv("WATERHOSE_DEBUG_KIT_DISPLAY", "").lower() in {"1", "true", "yes", "on"}:
        print(f"[waterhose-kit] {message}", flush=True)
