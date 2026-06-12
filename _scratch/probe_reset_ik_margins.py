# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose the loaded-pose reset IK: per-joint limit margins + achieved cup pose error.

For each loaded curriculum pose ("target_up", "home_up") solve the multi-seed reset IK,
then additionally sweep a free yaw about world Z (an opening-up cup is yaw-invariant) and
report, per yaw candidate: per-joint limit margins, position error of the cup centre, and
cup-opening up-alignment after FK. This tells us whether a non-railed branch exists at all
and which yaw to use.
"""

from __future__ import annotations

import argparse
import math

import torch

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Reset IK margin probe.")
parser.add_argument("--num-envs", type=int, default=4)
add_launcher_args(parser)
args_cli = parser.parse_args()


def main() -> None:
    from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv, _qmul_t
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg

    cfg = FrankaScoopEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = str(args_cli.device)

    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg)
        try:
            env.reset()
            arm_names = [n.replace("panda_joint", "j") for n in env._arm_joint_names]

            def yaw_quat(yaw: float) -> torch.Tensor:
                half = 0.5 * yaw
                return torch.tensor([0.0, 0.0, math.sin(half), math.cos(half)], device=env.device)

            for pose_kind, xy in (("target_up", cfg.target_center[:2]), ("home_up", cfg.loaded_hover_xy)):
                target = torch.tensor(
                    [float(xy[0]), float(xy[1]), float(cfg.dump_hover_z)], device=env.device
                ).unsqueeze(0).expand(env.num_envs, -1)
                print(f"=== pose {pose_kind} target={target[0].tolist()} ===", flush=True)
                for yaw_deg in (0, 45, 90, 135, 180, -45, -90, -135):
                    yaw = math.radians(yaw_deg)
                    # hand orientation: yaw (world Z) applied on top of the opening-up home orientation
                    q_up = env._pitch_to_hand_quat(torch.zeros(env.num_envs, device=env.device))
                    qz = yaw_quat(yaw).unsqueeze(0).expand(env.num_envs, -1)
                    q_t = _qmul_t(qz, q_up)
                    q_t = q_t / torch.clamp(torch.linalg.norm(q_t, dim=-1, keepdim=True), min=1e-8)
                    sol = env._solve_ik_full(
                        target,
                        q_t,
                        int(cfg.reset_ik_iterations),
                        arm_seed=env._default_arm_q,
                        solver=env._reset_ik_solver,
                    )[:, env._ik_arm].to(env.device)
                    margins = torch.minimum(sol - env._arm_lo, env._arm_hi - sol)  # (n, 7), pre-clamp
                    sol_c = torch.clamp(sol, env._arm_lo, env._arm_hi)
                    # apply to sim + FK to measure the achieved cup pose
                    import newton
                    import warp as wp

                    from isaaclab_newton.physics import NewtonManager

                    s0 = NewtonManager.get_state_0()
                    wp.to_torch(s0.joint_q)[env._arm_q_ids] = sol_c
                    newton.eval_fk(NewtonManager.get_model(), s0.joint_q, s0.joint_qd, s0, None)
                    env._sync_scoop_bowl_body(s0)
                    bowl_pos = env.bowl_pos_e()
                    _, bowl_quat = env._bowl_pose_w()
                    # cup opening axis: cup local +Z rotated to world; want it pointing up
                    z = torch.zeros(env.num_envs, 3, device=env.device)
                    z[:, 2] = 1.0
                    xyz, w = bowl_quat[:, :3], bowl_quat[:, 3:4]
                    t2 = 2.0 * torch.cross(xyz, z, dim=-1)
                    up = z + w * t2 + torch.cross(xyz, t2, dim=-1)
                    pos_err = torch.linalg.norm(bowl_pos - target, dim=-1)
                    worst_joint = margins.argmin(dim=-1)
                    print(
                        f"  yaw={yaw_deg:+4d}: min_margin={margins.min().item():+.3f} rad "
                        f"(joint {arm_names[int(worst_joint[0])]}) "
                        f"pos_err={pos_err.max().item():.4f} m up_z={up[:, 2].min().item():+.3f}",
                        flush=True,
                    )
        finally:
            env.close()


if __name__ == "__main__":
    main()
