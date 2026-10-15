# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A compact interactive rollout of the contrib Franka three-cube stacking scene."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import newton
import newton.ik as ik
import numpy as np
import warp as wp
from bundle import write_binary_files
from deformable import _robot_targets, pack_workbench, record_bundle_hashes
from franka_control import manual_control
from isaaclab_newton.physics import NewtonManager
from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg
from manipulation import _joint_coordinates
from newton_web import export_graph
from scenes import package_bundle, validate_toolchain

from isaaclab.app import launch_simulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim import CuboidCfg, MassCfg, RigidBodyMaterialCfg, UsdPhysicsCollisionCfg, get_current_stage

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK = "IsaacContrib-Stack-Cube-Franka"


@wp.kernel
def _stack_path(
    clock: wp.array(dtype=float),
    placement: wp.array(dtype=float),
    release: wp.array(dtype=int),
    manual: wp.array(dtype=int),
    goal: wp.array(dtype=wp.vec3),
    grip: wp.array(dtype=float),
    dt: float,
):
    if manual[0] != 0:
        return
    t = clock[0]
    desired = wp.vec3(0.55, 0.05, 0.17)
    opening = float(0.04)
    if t >= 2.0:
        desired[2] = wp.lerp(0.17, 0.023, wp.clamp((t - 2.0) / 2.0, 0.0, 1.0))
    if t >= 4.0:
        opening = wp.lerp(0.04, 0.0, wp.clamp((t - 4.0) / 1.0, 0.0, 1.0))
    if t >= 5.0:
        desired[2] = wp.lerp(0.023, 0.18, wp.clamp((t - 5.0) / 1.5, 0.0, 1.0))
    if t >= 6.5:
        desired = wp.lerp(
            wp.vec3(0.55, 0.05, 0.18), wp.vec3(placement[0], placement[1], 0.18), wp.clamp((t - 6.5) / 1.5, 0.0, 1.0)
        )
    if t >= 8.0:
        desired[2] = wp.lerp(0.18, 0.073, wp.clamp((t - 8.0) / 1.5, 0.0, 1.0))
    if t >= 9.5:
        opening = wp.lerp(0.0, 0.04, wp.clamp((t - 9.5) / 0.7, 0.0, 1.0))
    if t >= 10.5:
        desired[2] = wp.lerp(0.073, 0.18, wp.clamp((t - 10.5) / 1.0, 0.0, 1.0))
    if release[0] != 0:
        opening = 0.04
    displacement = desired - goal[0]
    goal[0] += displacement * wp.min(1.0, 0.2 * dt / wp.max(wp.length(displacement), 1.0e-6))
    grip[0] = opening
    clock[0] = wp.min(t + dt, 12.0)


@wp.kernel
def _manual_targets(manual: wp.array(dtype=int), desired: wp.array(dtype=float), targets: wp.array2d(dtype=float)):
    i = wp.tid()
    if manual[0] != 0:
        targets[0, i] = desired[i]


def scene_cfg():
    """Use aligned procedural cuboids instead of imported fitted block colliders."""
    cfg = parse_env_cfg(TASK, device="cpu", num_envs=1, overrides=["physics=newton_mjwarp"])
    cfg.play_mode()
    cfg.seed = 42
    cfg.scene.replicate_physics = True
    cfg.scene.robot.spawn.variants["Physics"] = "mujoco"
    # The browser controller reads Newton transforms directly; omit the unused
    # frame sensor and its observation terms from the packaged scene.
    cfg.scene.ee_frame = None
    cfg.observations.policy.object = None
    if hasattr(cfg.observations, "subtask_terms"):
        cfg.observations.subtask_terms = None
    cfg.observations.policy.eef_pos = None
    cfg.observations.policy.eef_quat = None
    cfg.sim.physics.use_cuda_graph = False
    cfg.sim.physics.collision_cfg = None
    cfg.sim.physics.solver_cfg.use_mujoco_contacts = True
    cfg.sim.physics.solver_cfg.ls_iterations = 50
    cfg.sim.physics.solver_cfg.nconmax = 64
    cfg.sim.physics.solver_cfg.njmax = 256
    for name, term in vars(cfg.events).items():
        if getattr(term, "mode", None) == "reset":
            setattr(cfg.events, name, None)
    cfg.scene.robot.spawn.rigid_props = {"/.*": [cfg.scene.robot.spawn.rigid_props, MujocoRigidBodyCfg(gravcomp=1.0)]}
    material = RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
    cfg.scene.table.spawn = CuboidCfg(
        size=(1.3, 0.9, 0.045),
        collision_props=UsdPhysicsCollisionCfg(),
        physics_material=material,
    )
    cfg.scene.table.init_state.pos = (0.5, 0.0, -0.0225)
    cfg.scene.table.init_state.rot = (0, 0, 0, 1)
    for name in ("cube_1", "cube_2", "cube_3"):
        cube = getattr(cfg.scene, name)
        cube.spawn = CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=cube.spawn.rigid_props,
            collision_props=UsdPhysicsCollisionCfg(),
            mass_props=MassCfg(mass=0.02),
            physics_material=material,
        )
        cube.init_state.pos = (*cube.init_state.pos[:2], 0.0253)
    return cfg


def export(output: Path) -> None:
    versions = validate_toolchain()
    cfg = scene_cfg()
    with launch_simulation(cfg, {"visualizer": ["none"], "device": "cpu"}):
        env = ManagerBasedRLEnv(cfg)
        try:
            env.reset(seed=42)
            model, state = NewtonManager.get_model(), NewtonManager.get_state_0()
            robot = env.scene["robot"]
            robot.write_joint_state_to_sim_index(
                position=robot.data.default_joint_pos.torch, velocity=robot.data.default_joint_vel.torch
            )
            robot.actuators.target_command.position.warp.assign(robot.data.default_joint_pos.torch.numpy())
            for _ in range(5):
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
            coordinates, dofs = _joint_coordinates(model, robot.joint_names)
            hand = next(i for i, name in enumerate(model.body_label) if name.endswith("/panda_hand"))
            root = next(i for i, name in enumerate(model.body_label) if name.endswith("/panda_link0"))
            red = next(i for i, name in enumerate(model.body_label) if "/Cube_2" in name)
            blue = next(i for i, name in enumerate(model.body_label) if "/Cube_1" in name)
            tcp = wp.transform_point(wp.transform(*state.body_q.numpy()[hand]), wp.vec3(0, 0, 0.1034))
            goal = wp.array([tcp], dtype=wp.vec3, device="cpu")
            q_ik = wp.clone(state.joint_q).reshape((1, model.joint_coord_count))
            mask = np.zeros(model.joint_dof_count, dtype=bool)
            mask[dofs[:7]] = True
            coordinate = wp.array(coordinates, dtype=int, device="cpu")
            inverse = ik.IKSolver(
                model,
                1,
                [
                    ik.IKObjectivePosition(hand, wp.vec3(0, 0, 0.1034), goal),
                    ik.IKObjectiveRotation(
                        hand, wp.quat_identity(), wp.array([wp.vec4(1, 0, 0, 0)], dtype=wp.vec4, device="cpu")
                    ),
                    ik.IKObjectiveJointLimit(model.joint_limit_lower, model.joint_limit_upper, weight=5.0),
                ],
                jacobian_mode="analytic",
                joint_dof_mask=wp.array(mask, dtype=bool, device="cpu"),
            )
            clock = wp.zeros(1, device="cpu")
            placement = wp.array([0.4, 0.0], dtype=float, device="cpu")
            release = wp.zeros(1, dtype=int, device="cpu")
            manual = wp.zeros(1, dtype=int, device="cpu")
            manual_target = wp.array(robot.data.default_joint_pos.torch.numpy()[0], dtype=float, device="cpu")
            grip = wp.array([0.04], dtype=float, device="cpu")
            with wp.ScopedCapture(device="cpu", apic=True) as capture:
                wp.launch(_stack_path, 1, [clock, placement, release, manual, goal, grip, cfg.sim.dt], device="cpu")
                inverse.step(q_ik, q_ik, iterations=3)
                wp.launch(
                    _robot_targets,
                    len(coordinates),
                    [q_ik, coordinate, grip, robot.actuators.target_command.position.warp],
                    device="cpu",
                )
                wp.launch(
                    _manual_targets,
                    9,
                    [manual, manual_target, robot.actuators.target_command.position.warp],
                    device="cpu",
                )
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
            regions = {}
            for key, region in capture.graph._apic_capture._regions.items():
                if region[0] not in regions or region[2] > regions[region[0]][1][2]:
                    regions[region[0]] = (key, region)
            capture.graph._apic_capture._regions = dict(regions.values())
            bindings = {n: getattr(state, n) for n in ("body_q", "body_qd", "joint_q", "joint_qd")}
            display = SimpleNamespace(
                body_count=model.body_count,
                particle_count=0,
                tri_count=0,
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
                inputs={
                    **bindings,
                    "placement": placement,
                    "release": release,
                    "clock": clock,
                    "manual": manual,
                    "manual_target": manual_target,
                },
                outputs=bindings,
                output=output,
                timestep=cfg.sim.dt,
                persistent=("placement",),
            )
            visuals = pack_workbench(
                get_current_stage(),
                model,
                output / "visuals.bin",
                False,
                extra_roots=[f"/World/envs/env_0/Cube_{i}" for i in (1, 2, 3)],
            )
            green = next(i for i, name in enumerate(model.body_label) if "/Cube_3" in name)
            palette = {red: (0.65, 0.07, 0.03), blue: (0.03, 0.20, 0.58), green: (0.08, 0.38, 0.13)}
            for mesh in visuals["meshes"]:
                if mesh["body"] in palette:
                    mesh["color"] = palette[mesh["body"]]
                    mesh["flatShading"] = True
            path = output / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["isaacLabDemo"] = {
                "kind": "franka_stack",
                "title": "Franka cube stacking",
                "controls": "parameters",
                "visuals": visuals,
                "rootBody": root,
                "redBody": red,
                "blueBody": blue,
                "handBody": hand,
                "scene": "workbench",
                "cycleSteps": round(13 / cfg.sim.dt),
                "cameraTarget": [0.4, 0.15, 0],
                "cameraRadius": 1.8,
                "cameraAzimuth": -2.1,
                "groundHeight": -1.05,
                "toneMappingExposure": 1.15,
                "maxStepsPerFrame": 12,
                "renderFPS": 30,
                "boundaryNote": "Contact grasp · Scripted Newton IK, no trained policy",
                "source": {
                    "task": TASK,
                    "versions": versions,
                    "controller": "Newton analytic IK and scripted pick/place, no trained policy",
                    "geometry": "50 mm, 20 g procedural cuboids; visual and collision geometry aligned",
                },
            }
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            (output / "manual-control.json").write_text(
                json.dumps(manual_control(model, robot, root, hand, "stack")) + "\n"
            )
            reference, frames = [], []
            lift = 0.0
            fps = round(1 / cfg.sim.dt)
            for frame in range(13 * fps):
                wp.capture_launch(capture.graph)
                q = state.body_q.numpy()
                if frame < 60 or (frame + 1) % (fps // 2) == 0:
                    frames.append(frame + 1)
                    reference.append(q.copy())
                if not all(np.isfinite(getattr(state, name).numpy()).all() for name in bindings):
                    raise RuntimeError("Non-finite stacking rollout")
                lift = max(lift, float(q[red, 2]))
                if frame % fps == fps - 1:
                    print("TIME", (frame + 1) * cfg.sim.dt, "RED", q[red, :3], "BLUE", q[blue, :3], flush=True)
            delta = q[red, :3] - q[blue, :3]
            if (
                lift < 0.10
                or np.linalg.norm(delta[:2]) > 0.01
                or abs(delta[2] - 0.05) > 0.001
                or abs(q[blue, 2] - 0.025) > 0.0005
            ):
                raise RuntimeError(f"Failed stack: lift={lift}, delta={delta}")
            np.savez(output / "reference.npz", frames=frames, body_q=reference)
            print("VERIFIED", TASK, "lift", lift, "stack", delta, flush=True)
        finally:
            env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--emxx", default="em++")
    a = p.parse_args()
    bundle = a.output / "franka_stack"
    web = a.output / "franka_stack-web"
    export(bundle)
    manifest = package_bundle(bundle, web, a.emxx)
    files = write_binary_files(gzip.compress((bundle / "visuals.bin").read_bytes(), mtime=0), web, "visuals.bin.gz")
    manifest["isaacLabDemo"]["visuals"].update(file="visuals.bin.gz", files=files)
    (web / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    record_bundle_hashes(bundle, web)


if __name__ == "__main__":
    main()
