# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse

import torch

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation


parser = argparse.ArgumentParser(description="Minimal Franka scoop multi-env init check.")
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=0)
add_launcher_args(parser)
args_cli = parser.parse_args()


def main() -> None:
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg

    cfg = FrankaScoopEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = str(args_cli.device)

    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg)
        try:
            obs, _ = env.reset()
            action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            for _ in range(args_cli.steps):
                obs, _, _, _, _ = env.step(action)
            solver = NewtonManager._solver
            mpm = solver.solver("media") if hasattr(solver, "solver") else solver.solvers["media"]
            mpm_model = getattr(mpm, "model", mpm)
            print(
                "[train-init] "
                f"envs={env.num_envs} particles={NewtonManager.get_model().particle_count} "
                f"policy_shape={tuple(obs['policy'].shape)} finite={bool(torch.isfinite(obs['policy']).all())} "
                f"mpm_grid_type={getattr(mpm_model, 'grid_type', getattr(mpm, 'grid_type', 'unknown'))} "
                f"use_cuda_graph={env.cfg.sim.physics.use_cuda_graph}",
                flush=True,
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
