# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Scripted-dip diagnostic: does the cup interact with the SUPPORTED source pile, and does
``count_in_bowl`` register it? Drives a straight dip into the source, a hold, then a lift, logging
counts + geometry each phase. Distinguishes a counter/pose bug from a no-scoop / tunneling issue."""
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
    # Easy-start: target the cup at the source-pile centroid + a small z so it hovers right above the pile.
    cfg.reset_start = "source_curriculum"
    cfg.curriculum_start_bowl_offset = ((0.0, 0.0, 0.05),) * 5
    # Relocate the source pile into the cup's actual reachable zone (cup opening-up only reaches x~0.16-0.27).
    cfg.source_center = (0.24, -0.10, 0.08)
    cfg.target_center = (0.24, 0.10, 0.08)
    cfg.workspace_lo = (0.08, -0.34, 0.02)
    cfg.workspace_hi = (0.45, 0.34, 0.30)

    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg)
        try:
            env.reset()
            n, dev = env.num_envs, env.device
            zero = torch.zeros(n, env.action_manager.total_action_dim, device=dev)
            env.step(zero)

            def log(tag):
                pe = env.particle_pos_e()                       # (n, P, 3)
                bp = env.bowl_pos_e()                           # (n, 3)
                cen = pe.mean(dim=1)                             # source pile centroid (env frame)
                d = torch.linalg.norm(pe - bp[:, None, :], dim=-1).amin(dim=1)  # nearest particle to bowl center
                b0 = [round(v, 3) for v in bp[0].tolist()]
                c0 = [round(v, 3) for v in cen[0].tolist()]
                print(f"[{tag:9s}] in_bowl={env.count_in_bowl().tolist()} "
                      f"in_src={[int(v) for v in env.count_in_source().tolist()]} "
                      f"in_tgt={env.count_in_target().tolist()} "
                      f"bowl_e0={b0} pile_cen0={c0} nearest_d={[round(v,3) for v in d.tolist()]}", flush=True)

            log("reset")
            # Phase 1: descend straight down into the source pile (dz = -1) for 40 steps.
            a = zero.clone(); a[:, 2] = -1.0
            for i in range(40):
                env.step(a)
                if i % 10 == 9:
                    log(f"dip{i+1}")
            # Phase 2: hold for 20 steps.
            for i in range(20):
                env.step(zero)
            log("hold")
            # Phase 3: lift straight up (dz = +1) for 50 steps.
            a = zero.clone(); a[:, 2] = 1.0
            for i in range(50):
                env.step(a)
                if i % 10 == 9:
                    log(f"lift{i+1}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
