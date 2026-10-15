# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Capture pretrained Franka beam and cloth policies with native MJWarp/VBD physics."""

from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import newton
import newton._src.solvers.coupled.solver_coupled as coupled
import numpy as np
import torch
import trimesh
import warp as wp
from bundle import write_binary_files, write_policy
from franka_control import manual_control
from isaaclab_newton.physics import NewtonManager
from manipulation import _actor_action, _append_mesh, _joint_coordinates, _pack_visuals
from newton_web import Parameter, export_graph
from rsl_rl.modules import EmpiricalNormalization
from scenes import package_bundle, validate_toolchain

from isaaclab.app import launch_simulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim import get_current_stage

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

WIDTHS = (93, 256, 128, 64, 8)
CHECKPOINT_SHA256 = {
    "soft": "6e80e7beae1e0dead3218c4fd52fc2ec319dff9b0fb763a5d836d69d30389f6e",
    "cloth": "7e4491b0d4b7d37b4df50291bce8b3c5d091edb1a394b07f657f356299a95405",
}

# Match the task's separate float32 Torch operations when processing actions.
wp.set_module_options({"fuse_fp": False})


def record_bundle_hashes(bundle: Path, web: Path) -> None:
    """Bind native trajectory evidence to the exact compiled artifact, including its mesh."""
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(web.iterdir()) if p.is_file()}
    (bundle / "validation.json").write_text(json.dumps({"files": hashes}, indent=2) + "\n")


@wp.kernel
def _gather_1d(source: wp.array(dtype=Any), mapping: wp.array(dtype=int), destination: wp.array(dtype=Any)):
    i = wp.tid()
    destination[i] = source[mapping[i]]


@wp.kernel
def _gather_2d(source: wp.array2d(dtype=Any), mapping: wp.array(dtype=int), destination: wp.array2d(dtype=Any)):
    i, j = wp.tid()
    destination[i, j] = source[mapping[i], j]


@contextmanager
def portable_coupler_controls():
    """Capture Newton's indexed control copies as equivalent explicit gather kernels.

    Warp 1.17 CPU APIC cannot record ``wp.copy(dst, src[indices])``. Keep this
    adapter build-local; it neither changes Newton's controls nor approximates physics.
    """
    original = coupled._copy_mapped_control_attribute

    def copy(source_control, destination_control, name, mapping):
        source = coupled._nested_attribute_value(source_control, name)
        destination = coupled._nested_attribute_value(destination_control, name)
        if not isinstance(source, wp.array) or not isinstance(destination, wp.array) or not destination.shape[0]:
            original(source_control, destination_control, name, mapping)
            return
        if source.ndim not in (1, 2):
            raise ValueError(f"Unreviewed coupled control shape: {name}, {source.shape}")
        wp.launch(
            _gather_1d if source.ndim == 1 else _gather_2d,
            destination.shape,
            [source, mapping, destination],
            device=destination.device,
        )

    coupled._copy_mapped_control_attribute = copy
    try:
        yield
    finally:
        coupled._copy_mapped_control_attribute = original


@wp.kernel
def _robot_targets(
    q: wp.array2d(dtype=float),
    coordinates: wp.array(dtype=int),
    grip: wp.array(dtype=float),
    targets: wp.array2d(dtype=float),
):
    i = wp.tid()
    if i < 7:
        targets[0, i] = q[0, coordinates[i]]
    else:
        targets[0, i] = grip[0]


@wp.kernel
def _volume_material(scale: wp.array(dtype=float), initial: wp.array2d(dtype=float), material: wp.array2d(dtype=float)):
    i, j = wp.tid()
    material[i, j] = initial[i, j] * wp.where(j < 2, scale[0], 1.0)


@wp.kernel
def _cloth_material(scale: wp.array(dtype=float), initial: wp.array2d(dtype=float), material: wp.array2d(dtype=float)):
    i, j = wp.tid()
    material[i, j] = initial[i, j] * wp.where(j == 0, scale[0], 1.0)


@wp.kernel
def _drag_object(
    index: wp.array(dtype=int),
    target: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    force: wp.array(dtype=wp.vec3),
):
    i = index[0]
    if i >= 0:
        f = 20.0 * (target[0] - q[i]) - 0.15 * qd[i]
        force[i] += f * wp.min(1.0, 2.0 / wp.max(wp.length(f), 1.0e-6))


def pack_workbench(stage, model, output: Path, cloth: bool, extra_roots=()) -> dict:
    """Use the native robot/support visuals and a light visual bench over the native support collider."""
    roots = ["/World/envs/env_0/Robot"]
    if cloth:
        roots += ["/World/envs/env_0/SupportNegY", "/World/envs/env_0/SupportPosY"]
    visuals = _pack_visuals(stage, roots + list(extra_roots), model.body_label, output)
    data = bytearray(output.read_bytes())
    pieces = [((1.3, 0.9, 0.045), (0.5, 0, -0.0225), (0.24, 0.31, 0.34))]
    pieces += [((0.055, 0.055, 1.0), (x, y, -0.55), (0.065, 0.085, 0.10)) for x in (-0.08, 1.08) for y in (-0.38, 0.38)]
    for index, (size, center, color) in enumerate(pieces):
        mesh = trimesh.creation.box(extents=size)
        vertices = (mesh.vertices + center).astype("<f4")
        _append_mesh(data, visuals["meshes"], vertices, mesh.faces, -1, f"workbench_{index}", color)
    return batch_visuals(visuals, bytes(data), output)


def batch_visuals(visuals: dict, raw: bytes, output: Path) -> dict:
    # All geometry here has a body-local transform and one flat material.
    # Batch by body/material at build time to reduce browser draw calls.
    groups = {}
    for mesh in visuals["meshes"]:
        key = (mesh["body"], tuple(mesh["color"]))
        groups.setdefault(key, []).append(mesh)
    data, merged = bytearray(), []
    for index, ((body, color), parts) in enumerate(groups.items()):
        vertices, faces, count = [], [], 0
        for part in parts:
            vertices.append(
                np.frombuffer(raw, dtype="<f4", count=part["vertexCount"] * 3, offset=part["vertexOffset"]).reshape(
                    -1, 3
                )
            )
            faces.append(
                np.frombuffer(raw, dtype="<u2", count=part["indexCount"], offset=part["indexOffset"])
                .reshape(-1, 3)
                .astype(np.int32)
                + count
            )
            count += part["vertexCount"]
        vertices = np.concatenate(vertices)
        faces = np.concatenate(faces)
        if body < 0:
            # Bench boxes have planar faces, rather than smoothed corner normals.
            vertices = vertices[faces.reshape(-1)]
            faces = np.arange(len(vertices)).reshape(-1, 3)
        _append_mesh(data, merged, vertices, faces, body, f"body_{body}_{index}", color)
    output.write_bytes(data)
    return {"file": output.name, "byteLength": len(data), "meshes": merged}


@wp.kernel
def _policy_targets(
    action: wp.array(dtype=float),
    q: wp.array(dtype=float),
    indices: wp.array(dtype=int),
    limits: wp.array(dtype=wp.vec2),
    release: wp.array(dtype=int),
    targets: wp.array2d(dtype=float),
):
    i = wp.tid()
    if i < 7:
        targets[0, i] = q[indices[i]] + 0.03 * action[i]
    else:
        opening = wp.clamp(action[7], -1.0, 1.0)
        if release[0] != 0:
            opening = 1.0
        lower = limits[0][0]
        upper = limits[0][1]
        targets[0, 7] = opening * (upper - lower) * 0.5 + (lower + upper) * 0.5


@wp.kernel
def _set_height(height: wp.array(dtype=float), command: wp.array(dtype=float)):
    command[2] = height[0]


@wp.kernel
def _policy_observation(
    q: wp.array(dtype=float),
    qd: wp.array(dtype=float),
    bodies: wp.array(dtype=wp.transform),
    particles: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    dofs: wp.array(dtype=int),
    defaults: wp.array(dtype=float),
    nodes: wp.array(dtype=int),
    root: int,
    command: wp.array(dtype=float),
    action: wp.array(dtype=float),
    mean: wp.array(dtype=float),
    denominator: wp.array(dtype=float),
    raw: wp.array(dtype=float),
    observation: wp.array(dtype=float),
):
    i = wp.tid()
    value = float(0.0)
    if i < 9:
        value = q[indices[i]] - defaults[i]
    elif i < 18:
        value = qd[dofs[i - 9]]
    elif i < 78:
        point = wp.transform_point(wp.transform_inverse(bodies[root]), particles[nodes[(i - 18) // 3]])
        value = point[(i - 18) % 3]
    elif i < 85:
        value = command[i - 78]
    else:
        value = action[i - 85]
    raw[i] = value
    observation[i] = (value - mean[i]) / denominator[i]


def export(kind: str, output: Path, checkpoint: Path) -> None:
    versions = validate_toolchain()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256[kind]:
        raise ValueError("Checkpoint changed; review its task and policy contract before rebuilding")
    torch.set_num_threads(1)
    actor = torch.load(checkpoint, map_location="cpu", weights_only=True)["actor_state_dict"]
    mean_values = actor["obs_normalizer._mean"].numpy()[0]
    denominator_values = actor["obs_normalizer._std"].numpy()[0] + EmpiricalNormalization(WIDTHS[0]).eps
    cloth = kind == "cloth"
    task = f"Isaac-Lift-{'Cloth' if cloth else 'Soft'}-Franka"
    cfg = parse_env_cfg(task, device="cpu", num_envs=1)
    cfg.play_mode()
    cfg.seed = 42
    cfg.scene.deformable.spawn.edge_refinement = 3 if cloth else 8
    cfg.sim.physics.use_cuda_graph = False
    cfg.sim.physics.solver_cfg.entries[0].solver_cfg.use_mujoco_contacts = True
    for entry in cfg.sim.physics.solver_cfg.entries:
        if hasattr(entry.solver_cfg, "particle_enable_tile_solve"):
            entry.solver_cfg.particle_enable_tile_solve = False
    command_cfg = cfg.commands.deformable_pose
    command_cfg.debug_vis = False
    command_cfg.resampling_time_range = (1000.0, 1000.0)
    command_cfg.ranges.pos_x = (0.55, 0.55)
    command_cfg.ranges.pos_y = (0.0, 0.0)
    command_cfg.ranges.pos_z = (0.32, 0.32)
    if cfg.actions.arm_action.scale != 0.03 or cfg.decimation != 4 or cfg.sim.physics.num_substeps != 2:
        raise ValueError("Reviewed relative-action or control timing contract changed")
    with launch_simulation(cfg, {"visualizer": ["none"], "device": "cpu"}):
        env = ManagerBasedRLEnv(cfg)
        try:
            env.reset(seed=42)
            env.step(torch.zeros((1, 8)))  # Materialize lazy solver resources before recording.
            obs, _ = env.reset(seed=42)
            model, state = NewtonManager.get_model(), NewtonManager.get_state_0()
            robot = env.scene["robot"]
            coordinates, dof_ids = _joint_coordinates(model, robot.joint_names)
            if robot.joint_names != [f"panda_joint{i}" for i in range(1, 8)] + [
                "panda_finger_joint1",
                "panda_finger_joint2",
            ] or obs["policy"].shape != (1, WIDTHS[0]):
                raise ValueError("Reviewed observation, action or joint order changed")
            root = next(i for i, name in enumerate(model.body_label) if name.endswith("/panda_link0"))
            hand = next(i for i, name in enumerate(model.body_label) if name.endswith("/panda_hand"))
            indices = wp.array(coordinates, dtype=int, device="cpu")
            dofs = wp.array(dof_ids, dtype=int, device="cpu")
            defaults = wp.array(robot.data.default_joint_pos.torch.numpy()[0], dtype=float, device="cpu")
            point_term = env.observation_manager.cfg.policy.deformable_sampled_points.func
            nodes = wp.array(point_term.node_ids.numpy()[0].astype(np.int32), dtype=int, device="cpu")
            command = wp.array(env.command_manager.get_command("deformable_pose").numpy()[0], dtype=float, device="cpu")
            height = wp.array([command.numpy()[2]], dtype=float, device="cpu")
            action = wp.zeros(8, dtype=float, device="cpu")
            release = wp.zeros(1, dtype=int, device="cpu")
            limits = wp.array(robot.data.soft_joint_pos_limits.torch.numpy()[0, 7:8], dtype=wp.vec2, device="cpu")
            raw = wp.zeros(WIDTHS[0], dtype=float, device="cpu")
            mean = wp.array(mean_values, dtype=float, device="cpu")
            denominator = wp.array(denominator_values, dtype=float, device="cpu")
            observation = wp.zeros(WIDTHS[0], dtype=float, device="cpu")
            observation_args = [
                state.joint_q,
                state.joint_qd,
                state.body_q,
                state.particle_q,
                indices,
                dofs,
                defaults,
                nodes,
                root,
                command,
                action,
                mean,
                denominator,
                raw,
                observation,
            ]
            wp.launch(_policy_observation, WIDTHS[0], observation_args, device="cpu")
            np.testing.assert_allclose(raw.numpy(), obs["policy"].numpy()[0], atol=2e-5, rtol=2e-5)
            drag_index = wp.array([-1], dtype=int, device="cpu")
            drag_target = wp.zeros(1, dtype=wp.vec3, device="cpu")
            NewtonManager.register_state_force_callback(
                lambda s: wp.launch(
                    _drag_object, 1, [drag_index, drag_target, s.particle_q, s.particle_qd, s.particle_f], device="cpu"
                )
            )
            scale = wp.ones(1, dtype=float, device="cpu")
            material = model.edge_bending_properties if cloth else model.tet_materials
            initial_material = wp.clone(material)
            soft_model = NewtonManager._solver.solver("soft").model
            soft_material = soft_model.edge_bending_properties if cloth else soft_model.tet_materials
            if soft_material.shape != material.shape:
                raise ValueError("Unexpected coupled material mapping")
            apply_material = _cloth_material if cloth else _volume_material
            targets = robot.actuators.target_command.position.warp
            with portable_coupler_controls(), wp.ScopedCapture(device="cpu", apic=True) as capture:
                wp.launch(apply_material, material.shape, [scale, initial_material, material], device="cpu")
                if soft_material.ptr != material.ptr:
                    wp.launch(apply_material, material.shape, [scale, initial_material, soft_material], device="cpu")
                wp.launch(_set_height, 1, [height, command], device="cpu")
                for _ in range(1 if NewtonManager.handles_decimation() else cfg.decimation):
                    # Relative targets are recomputed at every native action-application
                    # boundary, including each eager-actuator physics call.
                    wp.launch(
                        _policy_targets, 8, [action, state.joint_q, indices, limits, release, targets], device="cpu"
                    )
                    env.scene.write_data_to_sim()
                    NewtonManager.step()
                wp.launch(_policy_observation, WIDTHS[0], observation_args, device="cpu")
            regions = {}
            for key, region in capture.graph._apic_capture._regions.items():
                if region[0] not in regions or region[2] > regions[region[0]][1][2]:
                    regions[region[0]] = (key, region)
            capture.graph._apic_capture._regions = dict(regions.values())
            env.scene.update(dt=env.step_dt)
            obs = env.observation_manager.compute()
            wp.launch(_policy_observation, WIDTHS[0], observation_args, device="cpu")
            np.testing.assert_allclose(raw.numpy(), obs["policy"].numpy()[0], atol=2e-5, rtol=2e-5)
            bindings = {
                n: getattr(state, n) for n in ("body_q", "body_qd", "joint_q", "joint_qd", "particle_q", "particle_qd")
            }
            display = SimpleNamespace(
                body_count=model.body_count,
                particle_count=model.particle_count,
                tri_count=model.tri_count,
                tri_indices=model.tri_indices,
                tri_color=model.tri_color,
                shape_type=wp.array([int(newton.GeoType.PLANE)], dtype=int, device="cpu"),
                shape_body=wp.array([-1], dtype=int, device="cpu"),
                shape_scale=wp.array([wp.vec3(8, 8, 0)], dtype=wp.vec3, device="cpu"),
                shape_transform=wp.array(
                    [wp.transform(wp.vec3(0, 0, -1.05), wp.quat_identity())], dtype=wp.transform, device="cpu"
                ),
            )
            export_graph(
                capture.graph,
                model=display,
                output=output,
                timestep=env.step_dt,
                inputs={
                    **bindings,
                    "action": action,
                    "observation": observation,
                    "command": command,
                    "height": height,
                    "material": scale,
                    "release": release,
                    "drag_index": drag_index,
                    "drag_target": drag_target,
                },
                outputs={**bindings, "observation": observation},
                persistent=("height", "material"),
                parameters=(
                    Parameter("height", 0, "Target height [m]", 0.25, 0.5, 0.005),
                    Parameter("material", 0, "Bending scale" if cloth else "Elastic stiffness scale", 0.25, 4.0, 0.01),
                ),
            )
            visuals = pack_workbench(get_current_stage(), model, output / "visuals.bin", cloth)
            path = output / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["isaacLabDemo"] = {
                "kind": "franka_" + kind,
                "title": "Franka cloth policy" if cloth else "Franka soft-body policy",
                "viewer": "robot",
                "controls": "parameters",
                "policy": write_policy(checkpoint, output, WIDTHS),
                "decimation": 1,
                "cycleSteps": 150,
                "visuals": visuals,
                "rootBody": root,
                "cameraTarget": [0.35, 0.25, 0],
                "cameraRadius": 1.85,
                "cameraAzimuth": -2.1,
                "groundHeight": -1.05,
                "scene": "workbench",
                "showTarget": True,
                "toneMappingExposure": 1.15,
                "logParameters": ["material"],
                "maxStepsPerFrame": 2,
                "renderFPS": 30,
                "softDrag": True,
                "boundaryNote": (
                    "Pretrained RSL-RL policy · MJWarp + VBD contacts · Material changes test generalization"
                ),
                "source": {
                    "task": task,
                    "versions": versions,
                    "checkpointSha256": digest,
                    "physicsTimestep": cfg.sim.dt,
                    "physicsDecimation": cfg.decimation,
                    "controller": (
                        "Published Newton VBD RSL-RL checkpoint; native observations/actions and contact grasp"
                    ),
                },
            }
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            (output / "manual-control.json").write_text(
                json.dumps(manual_control(model, robot, root, hand, kind)) + "\n"
            )
            # Compare from the exact serialized state, including both coupled
            # solvers' internal allocations and their lagged feedback buffers.
            snapshots = [
                (region[1], ctypes.string_at(region[1], region[2]))
                for region in capture.graph._apic_capture._regions.values()
            ]
            check_targets = wp.clone(targets)
            native_actions, native_body, native_particles = [], [], []
            # Prove preprocessing against the actual task managers through grasp and hold.
            for frame in range(149):
                raw_values = obs["policy"].numpy()[0]
                decision = _actor_action(actor, WIDTHS, (raw_values - mean_values) / denominator_values)
                action.assign(decision.numpy())
                wp.launch(
                    _policy_targets, 8, [action, state.joint_q, indices, limits, release, check_targets], device="cpu"
                )
                env.action_manager.process_action(decision.reshape(1, -1))
                env.action_manager.apply_action()
                env.scene.write_data_to_sim()
                np.testing.assert_array_equal(check_targets.numpy()[0, :8], targets.numpy()[0, :8])
                obs, _, terminated, truncated, _ = env.step(decision.reshape(1, -1))
                if terminated.any() or truncated.any():
                    raise RuntimeError("Native policy terminated before its reviewed horizon")
                wp.launch(_policy_observation, WIDTHS[0], observation_args, device="cpu")
                np.testing.assert_allclose(raw.numpy(), obs["policy"].numpy()[0], atol=2e-4, rtol=2e-4)
                native_actions.append(decision.numpy().copy())
                native_body.append(state.body_q.numpy().copy())
                native_particles.append(state.particle_q.numpy().copy())
            native_height = float(native_particles[-1][:, 2].mean())
            if native_height < 0.18:
                raise RuntimeError(f"Native pretrained policy did not hold the object: {native_height:.3f} m")
            print("NATIVE_POLICY", task, "held height", native_height, flush=True)
            for pointer, data in snapshots:
                ctypes.memmove(pointer, data, len(data))
            rows, particles, observations = [], [], []
            for decision in native_actions:
                action.assign(decision)
                wp.capture_launch(capture.graph)
                if not all(np.isfinite(getattr(state, name).numpy()).all() for name in bindings):
                    raise RuntimeError("Non-finite captured policy trajectory")
                rows.append(state.body_q.numpy().copy())
                particles.append(state.particle_q.numpy().copy())
                observations.append(observation.numpy().copy())
            body_delta = np.max(np.abs(np.asarray(rows) - np.asarray(native_body)))
            particle_delta = np.max(np.abs(np.asarray(particles) - np.asarray(native_particles)))
            height_after = float(particles[-1][:, 2].mean())
            print(
                "CAPTURED_POLICY",
                task,
                "body delta",
                body_delta,
                "particle delta",
                particle_delta,
                "held height",
                height_after,
                flush=True,
            )
            if body_delta > 1e-6 or particle_delta > 1e-6 or height_after < 0.18:
                raise RuntimeError("Captured policy step disagrees with the native task")
            np.savez(
                output / "reference.npz",
                actions=native_actions,
                body_q=rows,
                particle_q=particles,
                observation=observations,
                native_body_q=native_body,
                native_particle_q=native_particles,
                joint_indices=coordinates,
                dof_indices=dof_ids,
                default_joint_positions=defaults.numpy(),
                sampled_nodes=nodes.numpy(),
                normalization_mean=mean_values,
                normalization_denominator=denominator_values,
                root_body=root,
            )
        finally:
            env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("soft", "cloth"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--emxx", default="em++")
    a = p.parse_args()
    bundle = a.output / ("franka_" + a.kind)
    web = a.output / ("franka_" + a.kind + "-web")
    export(a.kind, bundle, a.checkpoint)
    manifest = package_bundle(bundle, web, a.emxx)
    files = write_binary_files(gzip.compress((bundle / "visuals.bin").read_bytes(), mtime=0), web, "visuals.bin.gz")
    manifest["isaacLabDemo"]["visuals"].update(file="visuals.bin.gz", files=files)
    files = write_binary_files((bundle / "policy.bin").read_bytes(), web, "policy.bin")
    manifest["isaacLabDemo"]["policy"].update(file="policy.bin", files=files)
    (web / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    record_bundle_hashes(bundle, web)


if __name__ == "__main__":
    main()
