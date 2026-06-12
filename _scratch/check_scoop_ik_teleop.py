# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse

import torch

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation


parser = argparse.ArgumentParser(description="Headless stress check for Franka scoop teleop IK actions.")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps-per-command", type=int, default=24)
parser.add_argument("--ik-rotation-weight", type=float, default=None)
parser.add_argument("--ik-iterations", type=int, default=None)
parser.add_argument("--ik-step-size", type=float, default=None)
parser.add_argument("--max-ik-delta", type=float, default=None)
add_launcher_args(parser)
args_cli = parser.parse_args()


def _fmt(v: torch.Tensor) -> str:
    return "[" + ", ".join(f"{float(x):+.3f}" for x in v.detach().cpu().flatten()) + "]"


def main() -> None:
    from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg

    cfg = FrankaScoopEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = str(args_cli.device)
    if args_cli.ik_rotation_weight is not None:
        cfg.ik_rotation_weight = args_cli.ik_rotation_weight
    if args_cli.ik_iterations is not None:
        cfg.ik_iterations = args_cli.ik_iterations
    if args_cli.ik_step_size is not None:
        cfg.ik_step_size = args_cli.ik_step_size
    if args_cli.max_ik_delta is not None:
        cfg.max_ik_delta = args_cli.max_ik_delta

    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg)
        try:
            env.reset()
            action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            commands = [
                ("zero", (0.0, 0.0, 0.0, 0.0)),
                ("+x", (0.6, 0.0, 0.0, 0.0)),
                ("-x", (-0.6, 0.0, 0.0, 0.0)),
                ("+y", (0.0, 0.6, 0.0, 0.0)),
                ("-y", (0.0, -0.6, 0.0, 0.0)),
                ("+z", (0.0, 0.0, 0.6, 0.0)),
                ("-z", (0.0, 0.0, -0.6, 0.0)),
                ("+pitch", (0.0, 0.0, 0.0, 0.8)),
                ("-pitch", (0.0, 0.0, 0.0, -0.8)),
                ("mixed", (0.45, -0.35, 0.25, 0.6)),
            ]
            prev_q = env.arm_joint_q().clone()
            worst_err = 0.0
            worst_dq = 0.0
            for name, cmd in commands:
                action[:, :] = torch.tensor(cmd, device=env.device)
                for _ in range(args_cli.steps_per_command):
                    obs, _, _, _, _ = env.step(action)
                    q = env.arm_joint_q()
                    bowl_e = env.bowl_pos_e()
                    target = env._target_bowl_e
                    err = torch.linalg.norm(bowl_e - target, dim=-1)
                    dq = torch.max(torch.abs(q - prev_q))
                    worst_err = max(worst_err, float(err.max()))
                    worst_dq = max(worst_dq, float(dq))
                    finite = bool(torch.isfinite(obs["policy"]).all() and env.state_finite().all())
                    if not finite:
                        print(
                            f"[ik] nonfinite command={name} target={_fmt(target[0])} "
                            f"bowl={_fmt(bowl_e[0])} pitch={float(env._pitch[0]):+.3f}",
                            flush=True,
                        )
                        raise SystemExit(1)
                    prev_q = q.clone()
                print(
                    f"[ik] {name:>7} target={_fmt(target[0])} bowl={_fmt(bowl_e[0])} "
                    f"err={float(err[0]):.4f} pitch={float(env._pitch[0]):+.3f} "
                    f"last_dq={float(dq):.4f}",
                    flush=True,
                )
            print(f"[ik] finite=True worst_err={worst_err:.4f} worst_step_joint_delta={worst_dq:.4f}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
