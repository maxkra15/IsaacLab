# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Capture a reliable near-pile reset arm config: drive the cup to a pose just above the pile with the
stable runtime DiffIK (a P-controller on the action), then print the converged 7 arm joint angles. These
are hardcoded as the deterministic reset config so the cup SPAWNS right by the pile (no flaky reset IK)."""
from __future__ import annotations

import argparse
import torch

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()


def main() -> None:
    from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg

    cfg = FrankaScoopEnvCfg()
    cfg.grid_type = "sparse"          # probe-only solver override to run on the committed (OOMing) cfg
    cfg.use_cuda_graph = False
    cfg.__post_init__()
    cfg.scene.num_envs = 1
    cfg.sim.device = str(args_cli.device)

    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg)
        try:
            env.reset()
            dev = env.device
            act = torch.zeros(1, env.action_manager.total_action_dim, device=dev)
            env.step(act)
            # Target: cup just above the pile's near (robot-side) edge, gentle tilt toward the pile.
            # User's teleop-found scoop-start pose (env frame): cup above the pile, tilted to dig.
            target = torch.tensor([0.177, -0.183, 0.173], device=dev)
            target_pitch = 2.06
            best_d = 1e9
            best_q = env.arm_joint_q()[0].clone()
            best_cup = env.bowl_pos_e()[0].clone()
            for i in range(240):
                bp = env.bowl_pos_e()[0]
                err = target - bp
                a = torch.zeros_like(act)
                a[0, 0] = torch.clamp(err[0] * 3.0, -0.7, 0.7)
                a[0, 1] = torch.clamp(err[1] * 3.0, -0.7, 0.7)
                a[0, 2] = torch.clamp(err[2] * 3.0, -0.7, 0.7)
                a[0, 3] = torch.clamp(torch.tensor(target_pitch - float(env._pitch[0])) * 3.0, -1.0, 1.0)
                env.step(a)
                d = float(torch.linalg.norm(env.bowl_pos_e()[0] - target))
                if d < best_d:
                    best_d = d
                    best_q = env.arm_joint_q()[0].clone()
                    best_cup = env.bowl_pos_e()[0].clone()
                if i % 40 == 39:
                    print(f"[step{i+1}] cup={[round(float(v),3) for v in env.bowl_pos_e()[0].tolist()]} "
                          f"d={d:.3f} best_d={best_d:.3f}", flush=True)
            print(f"[CAPTURED] best_cup={[round(float(v),3) for v in best_cup.tolist()]} best_d={best_d:.3f}", flush=True)
            print(f"[CAPTURED] arm_home = ({', '.join(f'{float(v):.4f}' for v in best_q.tolist())})", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
