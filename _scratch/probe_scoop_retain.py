"""Headless: hemisphere ladle scoop -- build, dig into the pile, lift, and track retention (count_in_bowl).

    ./scoop_run.sh -p _scratch/probe_scoop_retain.py --device cuda:0 --headless
"""

import argparse

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()


def main() -> None:
    import torch

    from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg

    cfg = FrankaScoopEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = str(args_cli.device)

    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg)
        print(f"[SCOOP] shape={cfg.ee_cup_shape} ladle_r={cfg.ee_ladle_radius} wall={cfg.ee_ladle_wall_thickness} "
              f"voxel={cfg.voxel_size} -> wall/voxel={cfg.ee_ladle_wall_thickness/cfg.voxel_size:.2f}", flush=True)
        env.reset()

        def report(tag):
            ib = int(env.count_in_bowl()[0])
            src = int(env.count_in_source()[0])
            bp = [round(float(x), 3) for x in env.bowl_pos_e()[0].tolist()]
            print(f"[SCOOP] {tag:14s} in_bowl={ib:5d} in_source={src:5d} bowl_e={bp} pitch={float(env._pitch[0]):+.2f}",
                  flush=True)

        report("after reset")
        # Drive +x/-y toward the pile at source_center (0.32,-0.23), descend in while leveling the opening UP
        # (pitch->0 via -dpitch), then lift straight up holding the opening up -> media should be retained.
        phases = [
            ("approach", torch.tensor([[1.0, -0.4, -0.3, -0.3]]), 30),  # into pile, ends pitch ~+1.2
            ("dig",      torch.tensor([[0.4, -0.2, -0.8, -0.4]]), 25),  # ends ~level (pitch ~+0.2), captured
            ("lift",     torch.tensor([[-0.2, 0.0, 0.8, 0.0]]), 45),    # straight-ish up, NO tilt (opening stays up)
            ("hold",     torch.tensor([[0.0, 0.0, 0.0, 0.0]]), 30),
        ]
        for tag, act, n in phases:
            act = act.to(env.device)
            for _ in range(n):
                obs, *_ = env.step(act)
            report(tag)
        fin = bool(torch.isfinite(obs["policy"]).all())
        print(f"[SCOOP] obs_finite={fin}", flush=True)
        env.close()


main()
