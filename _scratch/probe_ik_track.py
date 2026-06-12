"""Headless: drive a smooth Cartesian action and measure NewtonIK tracking quality.

Reports per-step EE tracking error, max joint step, and joint "jerk" (2nd difference) -- a smooth IK has
small, consistent steps; branch-hopping shows up as jerk spikes.

    ./scoop_run.sh -p _scratch/probe_ik_track.py --device cuda:0 --headless
    ./scoop_run.sh -p _scratch/probe_ik_track.py --device cuda:0 --headless --backend diffik   # compare
"""

import argparse

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=("newton", "diffik"), default=None)
add_launcher_args(parser)
args_cli = parser.parse_args()


def main() -> None:
    import torch

    from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg

    cfg = FrankaScoopEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = str(args_cli.device)
    if args_cli.backend:
        cfg.ik_backend = args_cli.backend

    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg)
        env.reset()
        print(f"[IK] backend={cfg.ik_backend}", flush=True)

        # Gentle command (0.2 deflection -> target ~0.06 m/s) so the arm can keep up; if NewtonIK tracks,
        # error starts near 0 and stays a small steady lag (not a 30cm frame bug).
        mag = 0.2
        segments = [
            ("move +x", torch.tensor([[mag, 0.0, 0.0, 0.0]])),
            ("move +z", torch.tensor([[0.0, 0.0, mag, 0.0]])),
            ("move -y", torch.tensor([[0.0, -mag, 0.0, 0.0]])),
            ("tilt +",  torch.tensor([[0.0, 0.0, 0.0, mag]])),
        ]
        prev_q = env.arm_joint_q()[0].clone()
        prev_dq = torch.zeros_like(prev_q)
        for name, act in segments:
            act = act.to(env.device)
            trace, max_step, max_jerk = [], 0.0, 0.0
            for i in range(50):
                env.step(act)
                q = env.arm_joint_q()[0]
                dq = q - prev_q
                max_jerk = max(max_jerk, (dq - prev_dq).abs().max().item())
                max_step = max(max_step, float(dq.abs().max()))
                prev_q, prev_dq = q.clone(), dq.clone()
                err = float((env.bowl_pos_e()[0] - env._target_bowl_e[0]).norm())
                if i in (0, 9, 49):
                    trace.append((i + 1, err))
            tr = " ".join(f"s{n}={e*1000:.0f}mm" for n, e in trace)
            print(f"[IK] {name:8s}: err[{tr}] | max_step={max_step:.4f}rad max_jerk={max_jerk:.4f}rad "
                  f"finite={bool(torch.isfinite(q).all())}", flush=True)
        env.close()


main()
