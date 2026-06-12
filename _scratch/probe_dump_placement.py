"""Feasibility: can we reset the cup OPENING-UP (to hold a pre-loaded cupful) via IK from arm_home?

Tests _solve_ready_config (cup at home hover, opening up) + _solve_target_config at a few poses, and reports
position error + how 'up' the cup opening points (cup +Z . world +Z; 1.0 = opening straight up).

    ./scoop_run.sh -p _scratch/probe_dump_placement.py --device cuda:0 --headless
"""
import argparse

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()


def main() -> None:
    import torch
    import warp as wp
    import newton

    from isaaclab_newton.physics import NewtonManager
    from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg

    cfg = FrankaScoopEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = str(args_cli.device)
    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg)
        env.reset()

        def apply_and_report(tag, arm_q, target_e):
            s0 = NewtonManager.get_state_0()
            wp.to_torch(s0.joint_q)[env._arm_q_ids[[0]]] = arm_q.to(env.device)
            newton.eval_fk(NewtonManager.get_model(), s0.joint_q, s0.joint_qd, s0, None)
            env._sync_scoop_bowl_body(s0)
            bp = env.bowl_pos_e()[0]
            # cup opening = cup +Z axis in world; opening-up = dot with world +Z
            _, bq = env._bowl_pose_w()
            bq = bq[0]
            x, y, z, w = bq[0], bq[1], bq[2], bq[3]
            up_z = 1.0 - 2.0 * (x * x + y * y)  # R[2,2] = cup local +Z mapped to world Z
            err = float((bp - target_e.to(bp.device)).norm())
            print(f"[PLACE] {tag}: bowl_e={[round(float(v),3) for v in bp]} target={[round(float(v),3) for v in target_e]} "
                  f"pos_err={err*1000:.0f}mm opening_up={float(up_z):+.2f} (1=up)", flush=True)

        home = env._home_bowl_e.to(env.device)
        print(f"[PLACE] _home_bowl_e={[round(float(v),3) for v in home]} arm_home pitch≈2.06 (tilted)", flush=True)
        # 1) ready config (cup opening up at home hover)
        apply_and_report("ready_config (home, up)", env._solve_ready_config()[0], home)
        # 2) explicit solve: opening-up at the home position
        tgt = home.clone()
        q = env._solve_target_config(tgt.unsqueeze(0), torch.zeros(1, device=env.device),
                                     int(cfg.reset_ik_iterations), arm_seed=env._default_arm_q)[0]
        apply_and_report("solve(home, pitch=0)", q, tgt)
        # 3) explicit solve: opening-up over the TARGET box
        tgt2 = torch.tensor([cfg.target_center[0], cfg.target_center[1], 0.25], device=env.device)
        q2 = env._solve_target_config(tgt2.unsqueeze(0), torch.zeros(1, device=env.device),
                                      int(cfg.reset_ik_iterations), arm_seed=env._default_arm_q)[0]
        apply_and_report("solve(over target, pitch=0)", q2, tgt2)
        env.close()


main()
