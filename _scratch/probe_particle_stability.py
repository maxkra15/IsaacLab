# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Particle stability probe: detect reset/init explosions per voxel and curriculum stage.

Measures, right after reset and over the following steps: max particle speed, particles escaped
from a generous env-frame bounding box, and the source-pile count. Compares curriculum stage 0
(pre-loaded cup above the target) against the pile stage (no pre-load) to isolate the cause.
"""

from __future__ import annotations

import argparse

import torch

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--voxel", type=float, default=0.015)
parser.add_argument("--stage", type=int, default=0)
parser.add_argument("--pile-height", type=float, default=None)
parser.add_argument("--dump-hover", type=float, default=None)
parser.add_argument("--grid-padding", type=int, default=None)
parser.add_argument("--grid-type", choices=("fixed", "sparse"), default=None)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=60)
add_launcher_args(parser)
args_cli = parser.parse_args()


def main() -> None:
    from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg

    cfg = FrankaScoopEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = str(args_cli.device)
    cfg.voxel_size = float(args_cli.voxel)
    for entry in cfg.sim.physics.solver_cfg.entries:
        if entry.name == "media":
            entry.solver_cfg.voxel_size = float(args_cli.voxel)
    if args_cli.pile_height is not None:
        cfg.pile_height = float(args_cli.pile_height)
    if args_cli.dump_hover is not None:
        cfg.dump_hover_z = float(args_cli.dump_hover)
    if args_cli.grid_padding is not None:
        cfg.mpm_grid_padding = int(args_cli.grid_padding)
        for entry in cfg.sim.physics.solver_cfg.entries:
            if entry.name == "media":
                entry.solver_cfg.grid_padding = int(args_cli.grid_padding)
    if args_cli.grid_type is not None:
        cfg.grid_type = args_cli.grid_type
        cfg.use_cuda_graph = args_cli.grid_type == "fixed"
        cfg.sim.physics.use_cuda_graph = cfg.use_cuda_graph
        for entry in cfg.sim.physics.solver_cfg.entries:
            if entry.name == "media":
                entry.solver_cfg.grid_type = args_cli.grid_type
                if args_cli.grid_type == "sparse":
                    entry.solver_cfg.grid_padding = 0
                    entry.solver_cfg.max_active_cell_count = -1
    cfg.curriculum_start_stage = int(args_cli.stage)
    cfg.curriculum_freeze = True

    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg)
        try:
            env.reset()
            media = env.scene["media"]
            zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)

            def stats():
                pe = env.particle_pos_e()  # env frame, nan-sanitized to -100
                vel = media.data.particle_vel_w.torch
                speed = torch.linalg.norm(vel, dim=-1)
                escaped = (
                    (pe[..., 0].abs() > 0.9)
                    | (pe[..., 1].abs() > 0.9)
                    | (pe[..., 2] < -0.10)
                    | (pe[..., 2] > 0.70)
                )
                return float(speed.max()), int(escaped.sum()), float(env.count_in_source().mean())

            v0, e0, s0 = stats()
            pe0 = env.particle_pos_e().clone()
            vmax, emax = v0, e0
            for i in range(args_cli.steps):
                env.step(zero)
                v, e, _ = stats()
                vmax, emax = max(vmax, v), max(emax, e)
            v_end, e_end, s_end = stats()
            # localize the escapees: index range (cup pre-load = first indices per env) and origin
            pe_end = env.particle_pos_e()
            esc = (
                (pe_end[..., 0].abs() > 0.9)
                | (pe_end[..., 1].abs() > 0.9)
                | (pe_end[..., 2] < -0.10)
                | (pe_end[..., 2] > 0.70)
            )
            n_pre = int(cfg.curriculum_cup_fill_count[min(args_cli.stage, len(cfg.curriculum_cup_fill_count) - 1)])
            esc_pre = int(esc[:, :max(n_pre, 1)].sum()) if n_pre > 0 else 0
            if esc.any():
                pe0_esc = pe0[esc]
                cen = pe0_esc.mean(dim=0)
                print(
                    f"[stab-loc] escapees={int(esc.sum())} of which pre-load-indexed={esc_pre}"
                    f" | origin centroid (env frame)=({cen[0]:.3f},{cen[1]:.3f},{cen[2]:.3f})"
                    f" | origin z range=({pe0_esc[:,2].min():.3f},{pe0_esc[:,2].max():.3f})"
                    f" | origin y range=({pe0_esc[:,1].min():.3f},{pe0_esc[:,1].max():.3f})",
                    flush=True,
                )
            print(
                f"[stab] voxel={args_cli.voxel*1000:g}mm stage={args_cli.stage}"
                f" pile_h={cfg.pile_height} envs={env.num_envs}"
                f" | @reset: vmax={v0:.2f} m/s escaped={e0} in_source={s0:.0f}"
                f" | over {args_cli.steps} steps: vmax={vmax:.2f} m/s escaped_max={emax}"
                f" | @end: escaped={e_end} in_source={s_end:.0f}"
                f" finite={bool(env.state_finite().all())}",
                flush=True,
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
