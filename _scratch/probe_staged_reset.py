"""Verify the staged scoop->dump curriculum reset: loaded dump stage (cup over target, pre-loaded, stable,
tilt delivers) + scoop stage (empty at pile)."""
import argparse
from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); add_launcher_args(parser); args_cli = parser.parse_args()

def main():
    import torch, warp as wp
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg
    cfg = FrankaScoopEnvCfg(); cfg.scene.num_envs = 1; cfg.sim.device = str(args_cli.device)
    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg); env.reset()
        dev = env.device; eid = torch.tensor([0], device=dev)
        ids = env._particle_ids[0]
        def maxspeed():
            return float(wp.to_torch(NewtonManager.get_state_0().particle_qd)[ids].norm(dim=-1).max())
        for stage in (0, 3):
            env.curriculum_stage[:] = stage
            env.reset_scoop_scene(eid)
            bp = [round(float(v), 3) for v in env.bowl_pos_e()[0]]
            print(f"[STAGE {stage}] pose={cfg.curriculum_reset_pose[stage]} cup_fill={cfg.curriculum_cup_fill_count[stage]} "
                  f"bowl_e={bp} in_bowl={int(env.count_in_bowl()[0])} pitch={float(env._pitch[0]):+.2f}", flush=True)
            peak = 0.0
            for _ in range(12):  # stability: does the pre-loaded cup pop?
                env.step(torch.zeros((1, env.action_manager.total_action_dim), device=dev)); peak = max(peak, maxspeed())
            print(f"[STAGE {stage}] after 12 still steps: in_bowl={int(env.count_in_bowl()[0])} peak_speed={peak:.2f} m/s", flush=True)
            if stage == 0:  # tilt to dump -> does the cup tilt, media leave, reach the target?
                cup_ids = env._particle_ids[0][:80]
                z0 = float(wp.to_torch(NewtonManager.get_state_0().particle_q)[cup_ids, 2].mean())
                for sign in (1.0, -1.0):
                    for _ in range(45):
                        env.step(torch.tensor([[0.0, 0.0, 0.0, sign]], device=dev))
                    _, bq = env._bowl_pose_w(); up = float(1.0 - 2.0 * (bq[0, 0] ** 2 + bq[0, 1] ** 2))
                    zc = float(wp.to_torch(NewtonManager.get_state_0().particle_q)[cup_ids, 2].mean())
                    print(f"[STAGE 0] tilt {sign:+.0f}: pitch={float(env._pitch[0]):+.2f} opening_up={up:+.2f} "
                          f"cupful_z {z0:.2f}->{zc:.2f} in_bowl={int(env.count_in_bowl()[0])} "
                          f"in_target={int(env.count_in_target()[0])}", flush=True)
        env.close()
main()
