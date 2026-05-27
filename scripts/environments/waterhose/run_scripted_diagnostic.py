# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the RBY1 waterhose scripted motion inside the Isaac Lab environment.

This diagnostic helps separate cable/contact instability from manual teleop input.
Use ``--mode action`` to exercise the environment action path, or
``--mode direct`` to drive Newton control targets with the scripted controller.
"""

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher

from isaaclab_tasks.manager_based.manipulation.waterhose.launch import prepare_waterhose_launch


parser = argparse.ArgumentParser(description="Scripted diagnostic runner for the RBY1 waterhose task.")
parser.add_argument("--task", type=str, default="Isaac-Waterhose-RBY1DF-IK-Rel-Play-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--max_steps", type=int, default=2000, help="Maximum number of simulation steps.")
parser.add_argument("--log_interval", type=int, default=15, help="Print diagnostics every N steps.")
parser.add_argument(
    "--mode",
    choices=("action", "direct"),
    default="action",
    help=(
        "'action' routes the scripted target through the task action term. "
        "'direct' writes Newton control targets directly."
    ),
)
parser.add_argument("--stop_after_phase", type=str, default=None, help="Stop after this scripted phase is reached.")
parser.add_argument("--stop_on_done", action="store_true", help="Stop when the env reports done.")
parser.add_argument("--disable_cuda_graph", action="store_true", help="Disable Newton CUDA graph capture.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

waterhose_launch = prepare_waterhose_launch(args_cli, parser=parser)

app_launcher = None
simulation_app = None
if not waterhose_launch.uses_kitless_waterhose:
    app_launcher = AppLauncher(vars(args_cli))
    simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.manager_based.manipulation.waterhose import waterhose_core as core
from isaaclab_tasks.utils import launch_simulation, parse_env_cfg


def _simulation_is_running(env) -> bool:
    if simulation_app is not None:
        return simulation_app.is_running()
    if env.sim.visualizers:
        return any(visualizer.is_running() and not visualizer.is_closed for visualizer in env.sim.visualizers)
    return True


def _close_simulation_app() -> None:
    if simulation_app is not None:
        simulation_app.close()


def _diagnostic_snapshot(env, step: int, phase: str) -> str:
    scene_builder = env.waterhose_scene_builder
    model = core.NewtonManager.get_model()
    state = core.NewtonManager.get_state_0()
    body_q = state.body_q.numpy()
    body_qd = state.body_qd.numpy()

    cable_body_ids = [*scene_builder.cable_body_ids, *scene_builder.cable_head_body_ids]
    cable_finite = bool(np.isfinite(body_q[cable_body_ids]).all()) if cable_body_ids else True
    if cable_body_ids:
        cable_speed = np.linalg.norm(body_qd[cable_body_ids, :3], axis=1)
        max_cable_speed = float(np.max(cable_speed))
    else:
        max_cable_speed = 0.0

    tip_pos = body_q[scene_builder.tip_body_id, :3]
    plug_pos = body_q[scene_builder.plug_body_id, :3]
    right_ee_id = _find_body_id(model.body_label, core.RIGHT_EE)
    ee_pos = body_q[right_ee_id, :3]

    ee_to_plug = float(np.linalg.norm(ee_pos - plug_pos))
    return (
        f"step={step:05d} phase={phase:<18} "
        f"ee_to_plug={ee_to_plug:.4f}m "
        f"{_scripted_target_tracking_summary(env, ee_pos)} "
        f"tip=({tip_pos[0]:+.3f},{tip_pos[1]:+.3f},{tip_pos[2]:+.3f}) "
        f"plug=({plug_pos[0]:+.3f},{plug_pos[1]:+.3f},{plug_pos[2]:+.3f}) "
        f"max_cable_speed={max_cable_speed:.3f}m/s "
        f"finite={cable_finite} "
        f"{_runtime_contact_summary(scene_builder, model)} "
        f"{_joint_target_delta_summary(model, state, core.NewtonManager.get_control(), scene_builder)}"
    )


def _scripted_target_tracking_summary(env, ee_pos: np.ndarray) -> str:
    controller = getattr(env, "_scripted_controller", None)
    if controller is None:
        return "target_err=<unavailable>"
    filtered_target = getattr(controller, "last_filtered_target_pos", None)
    raw_target = getattr(controller, "last_raw_target_pos", None)
    if filtered_target is None:
        return "target_err=<pending>"
    ee_error = float(np.linalg.norm(np.asarray(filtered_target, dtype=np.float64) - ee_pos))
    if raw_target is None:
        return f"target_err={ee_error:.4f}m"
    filter_error = float(
        np.linalg.norm(np.asarray(raw_target, dtype=np.float64) - np.asarray(filtered_target, dtype=np.float64))
    )
    return f"target_err={ee_error:.4f}m filter_lag={filter_error:.4f}m"


def _find_body_id(labels: list[str], short_name: str) -> int:
    suffix = "/" + short_name
    for index, label in enumerate(labels):
        if label == short_name or label.endswith(suffix):
            return index
    raise RuntimeError(f"Could not find body named {short_name!r}.")


def _joint_label_for_dof(model, dof_id: int) -> str:
    starts = _array_np(getattr(model, "joint_dof_start", None))
    labels = getattr(model, "joint_label", None)
    if starts is None or labels is None:
        return f"dof{dof_id}"
    owner = None
    for joint_id, start in enumerate(starts):
        start_i = int(start)
        if start_i <= dof_id:
            owner = joint_id
        else:
            break
    if owner is None or owner >= len(labels):
        return f"dof{dof_id}"
    return str(labels[owner]).rsplit("/", 1)[-1]


def _joint_target_delta_summary(model, state, control, scene_builder, top_k: int = 4) -> str:
    joint_envs = getattr(scene_builder, "robot_joint_coord_ids_by_env", [])
    if not joint_envs:
        return "joint_delta=<unavailable>"
    ids = np.asarray(joint_envs[0], dtype=np.int64)
    joint_q = _array_np(getattr(state, "joint_q", None))
    joint_target = _array_np(getattr(control, "joint_target_pos", None))
    if joint_q is None or joint_target is None or joint_q.shape[0] <= int(ids.max()):
        return "joint_delta=<unavailable>"
    delta = joint_target[ids] - joint_q[ids]
    gripper = set(getattr(scene_builder, "gripper_dofs", []))
    ranked = [
        (int(global_id), float(delta[local_id]))
        for local_id, global_id in enumerate(ids)
        if int(global_id) not in gripper
    ]
    ranked.sort(key=lambda item: abs(item[1]), reverse=True)
    top = ranked[:top_k]
    if not top:
        return "joint_delta=none"
    return "joint_delta=" + ",".join(f"{_joint_label_for_dof(model, dof)}:{value:+.3f}" for dof, value in top)


def _runtime_contact_summary(scene_builder, model) -> str:
    """Summarise the most recent VBD-side rigid contact buffer by group.

    Under ADMM coupling there is no per-pair proxy collision pipeline to
    introspect; we read the live VBD entry's rigid contact buffer via the
    coupled solver's entry-local pipeline instead. Returns a compact
    breakdown by body-group ownership (cable, head, finger, scene).
    """
    solver = getattr(core.NewtonManager, "_solver", None)
    entries = getattr(solver, "_entries", None) if solver is not None else None
    contacts = None
    if entries is not None and core.HOSE_ENTRY in entries:
        contacts = getattr(entries[core.HOSE_ENTRY], "contacts", None)
    if contacts is None:
        contacts = getattr(core.NewtonManager, "_contacts", None)
    if contacts is None:
        return "contacts=<none>"
    count_array = _array_np(getattr(contacts, "rigid_contact_count", None))
    if count_array is None:
        return "contacts=<unavailable>"
    count = int(count_array[0])
    if count <= 0:
        return "contacts=0"

    shape0 = _array_np(getattr(contacts, "rigid_contact_shape0", None))
    shape1 = _array_np(getattr(contacts, "rigid_contact_shape1", None))
    if shape0 is None or shape1 is None:
        return f"contacts={count}"
    cable_shapes = set(_shape_ids_for_bodies(model, scene_builder.cable_body_ids))
    head_shapes = set(_shape_ids_for_bodies(model, scene_builder.cable_head_body_ids))
    scene_shapes = set(scene_builder.scene_shape_ids)
    finger_shapes = set(_shape_ids_for_bodies(model, scene_builder.gripper_finger_body_ids))
    categories = {
        "cable_scene": 0,
        "head_scene": 0,
        "cable_finger": 0,
        "head_finger": 0,
        "finger_scene": 0,
    }
    scene_hits: dict[int, int] = {}
    for a_raw, b_raw in zip(shape0[:count], shape1[:count], strict=False):
        a = int(a_raw)
        b = int(b_raw)
        pair = {a, b}
        if pair & cable_shapes and pair & scene_shapes:
            categories["cable_scene"] += 1
        if pair & head_shapes and pair & scene_shapes:
            categories["head_scene"] += 1
            for shape_id in pair & scene_shapes:
                scene_hits[shape_id] = scene_hits.get(shape_id, 0) + 1
        if pair & cable_shapes and pair & finger_shapes:
            categories["cable_finger"] += 1
        if pair & head_shapes and pair & finger_shapes:
            categories["head_finger"] += 1
        if pair & finger_shapes and pair & scene_shapes:
            categories["finger_scene"] += 1
    top_scene = sorted(scene_hits.items(), key=lambda item: item[1], reverse=True)[:2]
    scene_labels = getattr(model, "shape_label", [])
    top_scene_str = ",".join(
        f"{str(scene_labels[shape_id]).rsplit('/', 1)[-1]}:{hit_count}" for shape_id, hit_count in top_scene
    )
    return "contacts=" + " ".join(
        [f"total={count}", *[f"{k}={v}" for k, v in categories.items()], f"head_scene_top={top_scene_str or 'none'}"]
    )


def _array_np(value):
    if value is None:
        return None
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _stats(values, precision: int = 4) -> str:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return "n=0"
    return (
        f"n={array.size} min={array.min():.{precision}g} "
        f"mean={array.mean():.{precision}g} max={array.max():.{precision}g}"
    )


def _shape_material_summary(model, shape_ids: list[int], label: str) -> str:
    if not shape_ids:
        return f"{label}: shapes=0"
    ids = np.asarray(shape_ids, dtype=np.int64)
    parts = [f"{label}: shapes={ids.size}"]
    for attr in ("shape_material_ke", "shape_material_kd", "shape_material_mu", "shape_margin", "shape_gap", "shape_contact_gap"):
        values = _array_np(getattr(model, attr, None))
        if values is not None and values.shape[0] > int(ids.max()):
            parts.append(f"{attr}({_stats(values[ids], precision=3)})")
    return " ".join(parts)


def _shape_geometry_summary(model, shape_ids: list[int], label: str) -> str:
    if not shape_ids:
        return f"{label}: shapes=0"
    ids = np.asarray(shape_ids, dtype=np.int64)
    shape_type = _array_np(getattr(model, "shape_type", None))
    shape_scale = _array_np(getattr(model, "shape_scale", None))
    shape_body = _array_np(getattr(model, "shape_body", None))
    parts = [f"{label}: shapes={ids.size}"]
    if shape_type is not None and shape_type.shape[0] > int(ids.max()):
        unique, counts = np.unique(shape_type[ids], return_counts=True)
        parts.append("types=" + ",".join(f"{int(t)}:{int(c)}" for t, c in zip(unique, counts, strict=True)))
    if shape_scale is not None and shape_scale.shape[0] > int(ids.max()):
        for axis, name in enumerate(("x", "y", "z")):
            parts.append(f"scale_{name}({_stats(shape_scale[ids, axis], precision=4)})")
    if shape_body is not None and shape_body.shape[0] > int(ids.max()):
        body_ids = sorted({int(v) for v in shape_body[ids] if int(v) >= 0})
        parts.append(f"attached_bodies={len(body_ids)}")
    return " ".join(parts)


def _shape_ids_for_bodies(model, body_ids: list[int]) -> list[int]:
    if not body_ids:
        return []
    shape_body = _array_np(getattr(model, "shape_body", None))
    shape_flags = _array_np(getattr(model, "shape_flags", None))
    if shape_body is None:
        return []
    body_set = {int(body_id) for body_id in body_ids}
    shape_ids: list[int] = []
    for shape_id, body_id in enumerate(shape_body):
        if int(body_id) not in body_set:
            continue
        if shape_flags is not None and not int(shape_flags[shape_id]):
            continue
        shape_ids.append(int(shape_id))
    return shape_ids


def _vbd_view_finger_shape_material_summary(scene_builder, model, label: str) -> str:
    view = core.NewtonCoupledManager.get_entry_view(core.HOSE_ENTRY)
    finger_shape_ids = _shape_ids_for_bodies(model, scene_builder.gripper_finger_body_ids)
    return _shape_material_summary(view, finger_shape_ids, label)


def _body_mass_summary(model, body_ids: list[int], label: str) -> str:
    if not body_ids:
        return f"{label}: bodies=0"
    masses = _array_np(getattr(model, "body_mass", None))
    if masses is None:
        return f"{label}: bodies={len(body_ids)} body_mass=<unavailable>"
    ids = np.asarray(body_ids, dtype=np.int64)
    return f"{label}: bodies={ids.size} body_mass({_stats(masses[ids], precision=4)})"


def _body_inertia_summary(model, body_ids: list[int], label: str) -> str:
    if not body_ids:
        return f"{label}: bodies=0"
    inertias = _array_np(getattr(model, "body_inertia", None))
    if inertias is None:
        return f"{label}: bodies={len(body_ids)} body_inertia=<unavailable>"
    ids = np.asarray(body_ids, dtype=np.int64)
    selected = inertias[ids]
    if selected.ndim == 3:
        diag = np.stack([selected[:, 0, 0], selected[:, 1, 1], selected[:, 2, 2]], axis=-1)
    else:
        diag = selected.reshape((selected.shape[0], -1))[:, :3]
    return (
        f"{label}: bodies={ids.size} "
        f"Ixx({_stats(diag[:, 0], precision=4)}) "
        f"Iyy({_stats(diag[:, 1], precision=4)}) "
        f"Izz({_stats(diag[:, 2], precision=4)})"
    )


def _contact_pair_summary(model, scene_builder) -> str:
    pairs = _array_np(getattr(model, "shape_contact_pairs", None))
    if pairs is None:
        return "contact_pairs=<unavailable>"
    cable_shapes = set(_shape_ids_for_bodies(model, scene_builder.cable_body_ids))
    head_shapes = set(_shape_ids_for_bodies(model, scene_builder.cable_head_body_ids))
    scene_shapes = set(scene_builder.scene_shape_ids)
    finger_shapes = set(_shape_ids_for_bodies(model, scene_builder.gripper_finger_body_ids))

    counts = {
        "total": int(pairs.shape[0]),
        "cable_scene": 0,
        "head_scene": 0,
        "cable_finger": 0,
        "head_finger": 0,
    }
    for raw_a, raw_b in pairs:
        a = int(raw_a)
        b = int(raw_b)
        pair = {a, b}
        if pair & cable_shapes and pair & scene_shapes:
            counts["cable_scene"] += 1
        if pair & head_shapes and pair & scene_shapes:
            counts["head_scene"] += 1
        if pair & cable_shapes and pair & finger_shapes:
            counts["cable_finger"] += 1
        if pair & head_shapes and pair & finger_shapes:
            counts["head_finger"] += 1
    return "contact_pairs=" + " ".join(f"{key}={value}" for key, value in counts.items())


def _mesh_sdf_summary(model, shape_ids: list[int], label: str) -> str:
    if not shape_ids:
        return f"{label}: shapes=0"
    shape_source = getattr(model, "shape_source", None)
    shape_type = _array_np(getattr(model, "shape_type", None))
    if shape_source is None or shape_type is None:
        return f"{label}: sdf=<unavailable>"
    mesh_count = 0
    sdf_count = 0
    for shape_id in shape_ids:
        if int(shape_type[shape_id]) not in (8, 10):
            continue
        mesh_count += 1
        mesh = shape_source[shape_id]
        if mesh is not None and getattr(mesh, "sdf", None) is not None:
            sdf_count += 1
    return f"{label}: mesh_shapes={mesh_count} sdf_ready={sdf_count}"


def _print_startup_report(env) -> None:
    scene_builder = env.waterhose_scene_builder
    model = core.NewtonManager.get_model()
    cfg = scene_builder.cfg
    solver = getattr(core.NewtonManager, "_solver", None)
    solver_type = type(solver).__name__ if solver is not None else "<none>"
    physics_cfg = getattr(env.cfg, "sim", None).physics if getattr(env.cfg, "sim", None) is not None else None

    print("[INFO] Waterhose startup diagnostic report", flush=True)
    print(
        "[INFO] sim: "
        f"dt={float(env.step_dt):.6f}s num_envs={env.num_envs} "
        f"use_cuda_graph={getattr(physics_cfg, 'use_cuda_graph', '<unknown>')} "
        f"num_substeps={getattr(physics_cfg, 'num_substeps', '<unknown>')} solver={solver_type}",
        flush=True,
    )
    print(
        "[INFO] cfg: "
        f"cable_num_segments={getattr(cfg, 'cable_num_segments', '<missing>')} "
        f"vbd_cable_density={getattr(cfg, 'vbd_cable_density', '<missing>')} "
        f"vbd_cable_mu={getattr(cfg, 'vbd_cable_mu', '<missing>')} "
        f"vbd_default_ke={getattr(cfg, 'vbd_default_contact_ke', '<missing>')} "
        f"vbd_default_kd={getattr(cfg, 'vbd_default_contact_kd', '<missing>')} "
        f"grasp_mu={getattr(cfg, 'grasp_friction', '<missing>')} "
        f"grasp_ke={getattr(cfg, 'grasp_contact_ke', '<missing>')}",
        flush=True,
    )
    print(
        "[INFO] solver cfg: "
        f"mujoco_iterations={getattr(cfg, 'mujoco_iterations', '<missing>')} "
        f"mujoco_ls_iterations={getattr(cfg, 'mujoco_ls_iterations', '<missing>')} "
        f"mujoco_use_mujoco_contacts={getattr(cfg, 'mujoco_use_mujoco_contacts', '<missing>')} "
        f"vbd_iterations={getattr(cfg, 'vbd_iterations', '<missing>')} "
        f"vbd_rigid_avbd_beta={getattr(cfg, 'vbd_rigid_avbd_beta', '<missing>')} "
        f"vbd_rigid_contact_history={getattr(cfg, 'vbd_rigid_contact_history', '<missing>')} "
        f"vbd_rigid_contact_buffer_size={getattr(cfg, 'vbd_rigid_contact_buffer_size', '<missing>')}",
        flush=True,
    )
    print(
        "[INFO] admm coupling: "
        f"iterations={getattr(cfg, 'admm_iterations', '<missing>')} "
        f"rho={getattr(cfg, 'admm_rho', '<missing>')} "
        f"gamma={getattr(cfg, 'admm_gamma', '<missing>')} "
        f"baumgarte={getattr(cfg, 'admm_baumgarte', '<missing>')} "
        f"contact_distance={getattr(cfg, 'admm_contact_distance', '<missing>')} "
        f"detection_margin={getattr(cfg, 'admm_detection_margin', '<missing>')}",
        flush=True,
    )
    print(
        "[INFO] head mesh: "
        f"vbd_head_mesh_ke={getattr(cfg, 'vbd_head_mesh_ke', '<missing>')} "
        f"vbd_head_mesh_kd={getattr(cfg, 'vbd_head_mesh_kd', '<missing>')} "
        f"vbd_head_mesh_mu={getattr(cfg, 'vbd_head_mesh_mu', '<missing>')} "
        f"vbd_head_mesh_margin={getattr(cfg, 'vbd_head_mesh_margin', '<missing>')} "
        f"vbd_head_mesh_xy_scale={getattr(cfg, 'vbd_head_mesh_xy_scale', '<missing>')}",
        flush=True,
    )

    for curve_index, body_ids in enumerate(scene_builder.cable_body_ids_by_curve):
        lengths = scene_builder.cable_segment_lengths_by_curve[curve_index]
        print(
            "[INFO] cable curve "
            f"{curve_index}: bodies={len(body_ids)} segments={len(lengths)} "
            f"segment_length({_stats(lengths, precision=4)})",
            flush=True,
        )
    print("[INFO] " + _body_mass_summary(model, scene_builder.cable_body_ids, "cable"), flush=True)
    print("[INFO] " + _body_inertia_summary(model, scene_builder.cable_body_ids, "cable"), flush=True)
    print("[INFO] " + _body_mass_summary(model, scene_builder.cable_head_body_ids, "cable_heads"), flush=True)
    print("[INFO] " + _body_inertia_summary(model, scene_builder.cable_head_body_ids, "cable_heads"), flush=True)
    print("[INFO] " + _body_mass_summary(model, scene_builder.gripper_finger_body_ids, "gripper_fingers"), flush=True)
    print("[INFO] " + _body_inertia_summary(model, scene_builder.gripper_finger_body_ids, "gripper_fingers"), flush=True)
    vbd_view = core.NewtonCoupledManager.get_entry_view(core.HOSE_ENTRY)
    print(
        "[INFO] gripper finger labels: "
        + ", ".join(model.body_label[i] for i in scene_builder.gripper_finger_body_ids),
        flush=True,
    )
    cable_shape_ids = _shape_ids_for_bodies(model, scene_builder.cable_body_ids)
    cable_head_shape_ids = _shape_ids_for_bodies(model, scene_builder.cable_head_body_ids)
    finger_shape_ids = _shape_ids_for_bodies(model, scene_builder.gripper_finger_body_ids)
    print("[INFO] " + _shape_geometry_summary(model, cable_shape_ids, "cable_shapes_geometry"), flush=True)
    print("[INFO] " + _shape_material_summary(model, cable_shape_ids, "cable_shapes"), flush=True)
    print("[INFO] " + _shape_geometry_summary(model, cable_head_shape_ids, "cable_head_shapes_geometry"), flush=True)
    print("[INFO] " + _shape_material_summary(model, cable_head_shape_ids, "cable_head_shapes"), flush=True)
    print("[INFO] " + _shape_geometry_summary(model, finger_shape_ids, "finger_shapes_geometry"), flush=True)
    print("[INFO] " + _shape_material_summary(model, finger_shape_ids, "finger_shapes"), flush=True)
    print("[INFO] " + _shape_geometry_summary(model, scene_builder._vbd_shape_ids, "vbd_shapes_geometry"), flush=True)
    print("[INFO] parent " + _contact_pair_summary(model, scene_builder), flush=True)
    print("[INFO] vbd_view " + _contact_pair_summary(vbd_view, scene_builder), flush=True)
    print("[INFO] " + _mesh_sdf_summary(vbd_view, scene_builder.scene_shape_ids, "scene_mesh_sdf"), flush=True)
    print("[INFO] " + _mesh_sdf_summary(vbd_view, cable_head_shape_ids, "cable_head_mesh_sdf"), flush=True)
    print("[INFO] " + _mesh_sdf_summary(vbd_view, finger_shape_ids, "finger_mesh_sdf"), flush=True)
    print("[INFO] " + _vbd_view_finger_shape_material_summary(scene_builder, model, "finger_shapes_vbd_view"), flush=True)
    print("[INFO] " + _shape_material_summary(model, scene_builder.scene_shape_ids, "scene_shapes"), flush=True)
    print("[INFO] " + _shape_material_summary(model, scene_builder._vbd_shape_ids, "vbd_shapes"), flush=True)


def _should_stop_after_phase(phase: str) -> bool:
    return bool(args_cli.stop_after_phase and phase == args_cli.stop_after_phase)


def _profile_summary(env, step_count: int, wall_time: float) -> str:
    sim_time = step_count * float(env.step_dt)
    rtf = sim_time / max(wall_time, 1.0e-12)
    steps_per_second = step_count / max(wall_time, 1.0e-12)
    return f"[PROFILE] steps={step_count} sim_time={sim_time:.3f}s wall_time={wall_time:.3f}s rtf={rtf:.3f} steps_per_s={steps_per_second:.1f}"


def _run_action_mode(env) -> None:
    env.reset()
    step = 0
    print("[INFO] Running scripted waterhose diagnostic in action mode.")
    start_time = time.perf_counter()
    while step < args_cli.max_steps and _simulation_is_running(env):
        with torch.inference_mode():
            actions = env.scripted_action()
            _, _, terminated, truncated, _ = env.step(actions)
        phase = getattr(env, "waterhose_last_scripted_phase", "unknown")
        if args_cli.log_interval > 0 and step % args_cli.log_interval == 0:
            print(_diagnostic_snapshot(env, step, phase), flush=True)
        if args_cli.stop_on_done and (bool(terminated.any().item()) or bool(truncated.any().item())):
            print("[INFO] Env reported done; stopping.")
            break
        if _should_stop_after_phase(phase):
            print(f"[INFO] Reached stop_after_phase={phase!r}; stopping.", flush=True)
            break
        step += 1
    print(_profile_summary(env, step, time.perf_counter() - start_time), flush=True)


def _run_direct_mode(env) -> None:
    env.reset()
    controller = core.WaterhoseIKController(env.waterhose_scene_builder)
    step = 0
    print("[INFO] Running scripted waterhose diagnostic in direct Newton-control mode.")
    start_time = time.perf_counter()
    while step < args_cli.max_steps and _simulation_is_running(env):
        sim_time = step * float(env.step_dt)
        core.apply_viewer_forces(env.sim)
        phase = controller.update(sim_time)
        env.sim.step(render=False)
        if env.sim.is_rendering:
            core.sync_kit_cable_curves_from_newton(env.waterhose_scene_builder)
            env.sim.render()
        if args_cli.log_interval > 0 and step % args_cli.log_interval == 0:
            print(_diagnostic_snapshot(env, step, phase), flush=True)
        if _should_stop_after_phase(phase):
            print(f"[INFO] Reached stop_after_phase={phase!r}; stopping.", flush=True)
            break
        step += 1
    print(_profile_summary(env, step, time.perf_counter() - start_time), flush=True)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(f"Expected ManagerBasedRLEnvCfg, got {type(env_cfg).__name__}.")
    env_cfg.terminations.time_out = None
    if hasattr(env_cfg, "disable_cuda_graph"):
        env_cfg.disable_cuda_graph = bool(args_cli.disable_cuda_graph)
        env_cfg.sync_waterhose_sim_cfg()

    launch_context = None
    if simulation_app is None:
        launch_context = launch_simulation(env_cfg, args_cli)
        launch_context.__enter__()

    env = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        _print_startup_report(env)
        if args_cli.mode == "action":
            _run_action_mode(env)
        else:
            _run_direct_mode(env)
    finally:
        if env is not None:
            env.close()
        if launch_context is not None:
            launch_context.__exit__(None, None, None)
        _close_simulation_app()


if __name__ == "__main__":
    main()
