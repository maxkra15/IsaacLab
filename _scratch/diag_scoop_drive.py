# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Closed-loop scoop probe: actively DRIVE the cup onto the source pile (P-controller in xy),
dip, then lift -- and log count_in_bowl / count_in_source. Unlike the earlier probe (which only
moved vertically and stayed at the reset pose), this confirms whether the cup reaches the pile and
whether the bowl counter registers a real scoop."""
from __future__ import annotations

import argparse
import torch

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=2)
add_launcher_args(parser)
args_cli = parser.parse_args()


def main() -> None:
    from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg

    cfg = FrankaScoopEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = str(args_cli.device)
    # Place the SOURCE pile within the cup's comfortable opening-up reach (cup rests ~x=0.16; this is a
    # short hop). Validates reach + fill + the bowl counter + retention in one shot.
    cfg.source_center = (0.30, -0.12, 0.08)
    cfg.workspace_lo = (0.08, -0.34, 0.02)
    cfg.workspace_hi = (0.55, 0.34, 0.30)

    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg)
        try:
            env.reset()
            n, dev = env.num_envs, env.action_manager.total_action_dim,
            dev = env.device
            act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=dev)
            env.step(act)

            def log(tag):
                pe = env.particle_pos_e(); bp = env.bowl_pos_e()
                cen = env.source_media_centroid_e()
                d = torch.linalg.norm(pe - bp[:, None, :], dim=-1).amin(dim=1)
                print(f"[{tag:10s}] in_bowl={[int(v) for v in env.count_in_bowl().tolist()]} "
                      f"in_src={[int(v) for v in env.count_in_source().tolist()]} "
                      f"bowl_e0={[round(v,3) for v in bp[0].tolist()]} "
                      f"src_cen0={[round(v,3) for v in cen[0].tolist()]} "
                      f"nearest_d={[round(v,3) for v in d.tolist()]}", flush=True)

            log("reset")
            # Drive the cup toward the source centroid (xy) with a P-controller, descending toward the pile.
            for i in range(120):
                bp = env.bowl_pos_e()
                cen = env.source_media_centroid_e()
                err = cen - bp
                a = torch.zeros_like(act)
                a[:, 0] = torch.clamp(err[:, 0] * 8.0, -1, 1)
                a[:, 1] = torch.clamp(err[:, 1] * 8.0, -1, 1)
                # descend once roughly above the pile (xy error small), else hold height while approaching
                close = torch.linalg.norm(err[:, :2], dim=-1) < 0.05
                a[:, 2] = torch.where(close, torch.full_like(a[:, 2], -1.0), torch.clamp(err[:, 2] * 4.0, -0.3, 0.3))
                env.step(a)
                if i % 20 == 19:
                    log(f"drive{i+1}")
            log("dipped")
            # Lift straight up.
            a = torch.zeros_like(act); a[:, 2] = 1.0
            for i in range(60):
                env.step(a)
                if i % 20 == 19:
                    log(f"lift{i+1}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
