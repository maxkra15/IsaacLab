# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import time

import torch

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation


parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--warmup", type=int, default=30)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--disable-pour-bowl-mesh", action="store_true")
parser.add_argument("--container-geometry", choices=("bucket", "pour_bowl", "box"), default=None)
parser.add_argument("--reset-start", choices=("home", "source_curriculum"), default=None)
parser.add_argument("--decimation", type=int, default=None)
parser.add_argument("--dt", type=float, default=None)
parser.add_argument("--voxel-size", type=float, default=None)
parser.add_argument("--mpm-iterations", type=int, default=None)
parser.add_argument("--ik-iterations", type=int, default=None)
parser.add_argument("--ik-backend", choices=("diffik", "newton"), default=None)
parser.add_argument("--diffik-lambda", type=float, default=None)
parser.add_argument("--diffik-max-delta", type=float, default=None)
parser.add_argument("--enable-mpm-timers", action="store_true")
add_launcher_args(parser)
args_cli = parser.parse_args()


def main() -> None:
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg

    cfg = FrankaScoopEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = str(args_cli.device)
    if args_cli.disable_pour_bowl_mesh:
        cfg.container_geometry = "box"
        cfg.use_pour_bowl_mesh = False
    if args_cli.container_geometry is not None:
        cfg.container_geometry = args_cli.container_geometry
    if args_cli.reset_start is not None:
        cfg.reset_start = args_cli.reset_start
    if args_cli.decimation is not None:
        cfg.decimation = args_cli.decimation
        cfg.sim.render_interval = cfg.decimation
    if args_cli.dt is not None:
        cfg.sim.dt = args_cli.dt
    if args_cli.voxel_size is not None:
        cfg.voxel_size = args_cli.voxel_size
    if args_cli.mpm_iterations is not None:
        cfg.mpm_iterations = args_cli.mpm_iterations
    if args_cli.ik_iterations is not None:
        cfg.ik_iterations = args_cli.ik_iterations
    if args_cli.ik_backend is not None:
        cfg.ik_backend = args_cli.ik_backend
    if args_cli.diffik_lambda is not None:
        cfg.diffik_lambda = args_cli.diffik_lambda
    if args_cli.diffik_max_delta is not None:
        cfg.diffik_max_delta = args_cli.diffik_max_delta
    if args_cli.enable_mpm_timers:
        for entry in cfg.sim.physics.solver_cfg.entries:
            if entry.name == "media":
                entry.solver_cfg.solver_kwargs = {"enable_timers": True}

    with launch_simulation(cfg, args_cli):
        from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv
        from newton import ShapeFlags

        env = FrankaScoopEnv(cfg)
        try:
            from isaaclab_newton.physics import NewtonManager

            model = NewtonManager.get_model()
            solver = NewtonManager._solver
            mpm = solver.solver("media") if hasattr(solver, "solver") else solver.solvers["media"]

            collide_particles = int(ShapeFlags.COLLIDE_PARTICLES)
            robot_particle_shapes = []
            particle_shapes = []
            for sid, label in enumerate(model.shape_label):
                flags = int(model.shape_flags.numpy()[sid])
                if flags & collide_particles:
                    particle_shapes.append((sid, str(label), int(model.shape_body.numpy()[sid])))
                    if "/panda/" in str(label) and "ScoopBowl" not in str(label):
                        robot_particle_shapes.append((sid, str(label)))

            print(
                f"[profile] envs={env.num_envs} decimation={env.cfg.decimation} dt={env.cfg.sim.dt} "
                f"voxel={env.cfg.voxel_size} particles={model.particle_count} "
                f"shapes={model.shape_count} particle_shapes={len(particle_shapes)} "
                f"robot_particle_shapes={robot_particle_shapes}",
                flush=True,
            )
            print(f"[profile] particle_shape_labels={particle_shapes}", flush=True)

            collider_meshes = getattr(getattr(mpm, "model", None), "_collider_meshes", None)
            if collider_meshes is None:
                collider_meshes = getattr(mpm, "_collider_meshes", None)
            if collider_meshes is not None:
                stats = []
                for cid, mesh in enumerate(collider_meshes):
                    points = int(mesh.points.shape[0])
                    tris = int(mesh.indices.shape[0] // 3)
                    stats.append((cid, points, tris))
                print(f"[profile] mpm_collider_meshes={stats}", flush=True)

            obs, _ = env.reset()
            action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            for _ in range(args_cli.warmup):
                env.step(action)
            if env.device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args_cli.steps):
                env.step(action)
            if env.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            print(
                f"[profile] step_ms={elapsed * 1000.0 / args_cli.steps:.3f} "
                f"physics_ticks_per_env_step={env.cfg.decimation}",
                flush=True,
            )
            print(f"[profile] finite={torch.isfinite(obs['policy']).all().item()}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
