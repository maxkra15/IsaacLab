"""Scan candidate container positions for cup reachability (opening-up): pos error + joint-limit railing.
A position is "good" if the IK reaches it (low pos_err) without any joint pinned near its limit."""
import argparse

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()


def main():
    import torch

    from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg

    cfg = FrankaScoopEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = str(args_cli.device)
    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg)
        env.reset()
        import warp as wp
        import newton
        from isaaclab_newton.physics import NewtonManager

        lo, hi = env._arm_lo[0], env._arm_hi[0]
        # current target at several pitches (0=opening up, larger=tilted to pour) + a few positions
        cands = [(0.32, 0.22, 0.0), (0.32, 0.22, 1.0), (0.32, 0.22, 1.8), (0.32, 0.22, 2.4),
                 (0.30, 0.12, 1.8), (0.32, 0.0, 1.8), (0.40, -0.18, 0.0)]
        for x, y, pitch in cands:
            tgt = torch.tensor([[x, y, cfg.dump_hover_z]], device=env.device)
            q = env._solve_target_config(tgt, torch.full((1,), pitch, device=env.device),
                                         int(cfg.reset_ik_iterations), arm_seed=env._default_arm_q)
            s0 = NewtonManager.get_state_0()
            wp.to_torch(s0.joint_q)[env._arm_q_ids[[0]]] = q[[0]].to(env.device)
            newton.eval_fk(NewtonManager.get_model(), s0.joint_q, s0.joint_qd, s0, None)
            env._sync_scoop_bowl_body(s0)
            bp = env.bowl_pos_e()[0]
            err = float((bp[:2] - torch.tensor([x, y], device=bp.device)).norm())
            qa = q[0]
            # margin to nearest joint limit (rad); small => railed
            margin = torch.minimum(qa - lo, hi - qa).min().item()
            railed = "RAILED" if margin < 0.05 else "ok"
            print(f"[REACH] ({x:.2f},{y:+.2f}) pitch={pitch:.1f}: pos_err={err*1000:5.0f}mm "
                  f"min_joint_margin={margin:+.2f}rad  {railed}", flush=True)
        env.close()


main()
