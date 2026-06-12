# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Validate the curriculum/reward/init-cup-up redesign WITHOUT touching the committed solver settings.
Overrides only the solver (sparse + no cuda graph) inside this probe so it runs on the current cfg
(which OOMs with fixed+graph+2^21 cells). Checks: (1) cup resets opening-up right above the source pile
with the cavity floor ABOVE the media (not inside); (2) the sparse reward terms + curriculum load and
compute finite values; (3) media can be scooped out (removed_from_source moves)."""
from __future__ import annotations

import argparse
import torch

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=4)
add_launcher_args(parser)
args_cli = parser.parse_args()


def main() -> None:
    from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg

    cfg = FrankaScoopEnvCfg()
    # Probe-only solver override to dodge the fixed+graph+2^21 OOM (user owns the committed solver cfg).
    cfg.grid_type = "sparse"
    cfg.use_cuda_graph = False
    cfg.__post_init__()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = str(args_cli.device)

    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg)
        try:
            env.reset()
            bp = env.bowl_pos_e()
            cen = env.source_media_centroid_e()
            pe = env.particle_pos_e()
            media_top = pe[..., 2].amax(dim=1)
            cavity_floor_z = bp[:, 2] - float(env._bowl_floor)        # bottom of the cup cavity (env z)
            print(f"[reset] cup_e0={[round(v,3) for v in bp[0].tolist()]} "
                  f"src_cen0={[round(v,3) for v in cen[0].tolist()]} "
                  f"cavity_floor_z={[round(v,3) for v in cavity_floor_z.tolist()]} "
                  f"media_top_z={[round(v,3) for v in media_top.tolist()]} "
                  f"clearance(floor-mediatop)={[round(float(cavity_floor_z[i]-media_top[i]),3) for i in range(env.num_envs)]}",
                  flush=True)
            print(f"[reset] in_src={[int(v) for v in env.count_in_source().tolist()]} "
                  f"in_bowl={[int(v) for v in env.count_in_bowl().tolist()]} "
                  f"pitch={[round(float(v),2) for v in env._pitch.tolist()]} "
                  f"reward_terms={list(env.reward_manager.active_terms)}", flush=True)

            # Drive a scripted dip toward the source centroid + pitch down to dig, then check reward signals.
            act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            for i in range(80):
                bp = env.bowl_pos_e(); cen = env.source_media_centroid_e(); err = cen - bp
                a = torch.zeros_like(act)
                a[:, 0] = torch.clamp(err[:, 0] * 8.0, -1, 1)
                a[:, 1] = torch.clamp(err[:, 1] * 8.0, -1, 1)
                a[:, 2] = -0.6           # ease down into the pile
                a[:, 3] = 0.4            # pitch to dig
                obs, rew, term, trunc, info = env.step(a)
                if i % 20 == 19:
                    removed = (env._init_source_count - env.count_in_source()).clamp(min=0)
                    print(f"[dig{i+1:02d}] rew={[round(float(v),3) for v in rew.tolist()]} "
                          f"in_bowl={[int(v) for v in env.count_in_bowl().tolist()]} "
                          f"removed={[int(v) for v in removed.tolist()]} "
                          f"finite={bool(env.state_finite().all())}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
