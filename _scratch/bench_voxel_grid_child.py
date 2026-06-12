# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Single benchmark run of the scoop env for one (voxel_size, grid_type) config.

Launched as a subprocess by ``bench_voxel_grid.py``; writes a JSON result to ``--out``.
Zero-action workload after warmup, mirroring the profile probe. NOTE: the env cfg's
``__post_init__`` bakes voxel/grid/cell settings into the MPM solver entry at
construction, so this script patches BOTH the top-level cfg fields (consumed live by
media spawning / counting) and the baked ``MPMSolverCfg`` entry.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Scoop voxel/grid benchmark child.")
parser.add_argument("--voxel", type=float, required=True)
parser.add_argument("--grid-type", choices=("fixed", "sparse", "dense"), required=True)
parser.add_argument("--num-envs", type=int, default=48)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--max-active-cells", type=int, required=True)
parser.add_argument("--grid-padding", type=int, default=-1, help="-1 keeps the task cfg default (8).")
parser.add_argument("--cuda-graph", type=int, default=1)
parser.add_argument("--out", type=str, required=True)
add_launcher_args(parser)
args_cli = parser.parse_args()


def main() -> None:
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg

    cfg = FrankaScoopEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = str(args_cli.device)
    # Top-level fields: read live by media spawning, counting regions, and the env.
    cfg.voxel_size = float(args_cli.voxel)
    cfg.grid_type = str(args_cli.grid_type)
    cfg.mpm_max_active_cells = int(args_cli.max_active_cells)
    cfg.use_cuda_graph = bool(args_cli.cuda_graph)
    # Benchmark the plain pile task: pin the curriculum to its last stage (empty cup at the
    # pile, fixed arm config) so reset does no cup pre-load and no multi-seed IK solve.
    cfg.curriculum_start_stage = len(cfg.curriculum_reset_pose) - 1
    cfg.curriculum_freeze = True
    if args_cli.grid_padding >= 0:
        cfg.mpm_grid_padding = int(args_cli.grid_padding)
    # Baked solver entry: __post_init__ already copied the ORIGINAL values in.
    for entry in cfg.sim.physics.solver_cfg.entries:
        if entry.name == "media":
            entry.solver_cfg.voxel_size = float(args_cli.voxel)
            entry.solver_cfg.grid_type = str(args_cli.grid_type)
            entry.solver_cfg.max_active_cell_count = int(args_cli.max_active_cells)
            if args_cli.grid_padding >= 0:
                entry.solver_cfg.grid_padding = int(args_cli.grid_padding)
    cfg.sim.physics.use_cuda_graph = bool(args_cli.cuda_graph)

    result: dict = {
        "voxel": float(args_cli.voxel),
        "grid_type": str(args_cli.grid_type),
        "num_envs": int(args_cli.num_envs),
        "steps": int(args_cli.steps),
        "max_active_cells": int(args_cli.max_active_cells),
        "cuda_graph_requested": bool(args_cli.cuda_graph),
        "status": "failed",
    }

    t_start = time.perf_counter()
    with launch_simulation(cfg, args_cli):
        from isaaclab_newton.physics import NewtonManager
        from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv

        env = FrankaScoopEnv(cfg)
        try:
            obs, _ = env.reset()
            result["particles"] = int(NewtonManager.get_model().particle_count)
            result["particles_per_env"] = result["particles"] // env.num_envs

            action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            for _ in range(args_cli.warmup):
                env.step(action)
            torch.cuda.synchronize()
            result["startup_s"] = time.perf_counter() - t_start
            result["cuda_graph_active"] = NewtonManager._graph is not None

            free_b, total_b = torch.cuda.mem_get_info()
            result["cuda_used_after_warmup_gib"] = (total_b - free_b) / 2**30
            result["cuda_total_gib"] = total_b / 2**30

            t0 = time.perf_counter()
            for _ in range(args_cli.steps):
                env.step(action)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            free_b, _ = torch.cuda.mem_get_info()
            result["cuda_used_after_run_gib"] = (total_b - free_b) / 2**30
            result["step_ms"] = elapsed * 1000.0 / args_cli.steps
            result["env_steps_per_s"] = env.num_envs * args_cli.steps / elapsed
            result["finite"] = bool(env.state_finite().all())
            result["obs_finite"] = bool(torch.isfinite(obs["policy"]).all())
            result["status"] = "ok"
        finally:
            env.close()

    with open(args_cli.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[bench-child] {result}", flush=True)


if __name__ == "__main__":
    main()
