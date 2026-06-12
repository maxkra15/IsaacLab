"""Headless: is the cup body's transform live in state_0.body_q (what sync_transforms_to_usd renders),
or only in the media sub-state? Drives the arm and compares.

    ./scoop_run.sh -p _scratch/probe_cup_state.py --device cuda:0 --headless
"""

import argparse

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()


def main() -> None:
    import torch
    import warp as wp

    from isaaclab_newton.physics import NewtonManager
    from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg

    cfg = FrankaScoopEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = str(args_cli.device)
    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg)
        env.reset()
        scoop_id = int(env._scoop_body_ids_l[0])
        hand_id = int(env._hand_ids_l[0])
        s0 = NewtonManager.get_state_0()
        nbody = int(s0.body_q.shape[0])
        print(f"[PROBE] scoop_body_id(global)={scoop_id} hand_id={hand_id} state_0.body_q len={nbody}", flush=True)
        act = torch.tensor([[0.0, -0.7, -0.4, 0.2]], device=env.device)  # drive the cup -y/down/tilt
        for i in range(60):
            env.step(act)
            if i % 20 == 19:
                bq = wp.to_torch(NewtonManager.get_state_0().body_q)
                cup_s0 = [round(float(x), 3) for x in bq[scoop_id, :3].tolist()] if scoop_id < bq.shape[0] else "OOR"
                hand_s0 = [round(float(x), 3) for x in bq[hand_id, :3].tolist()]
                bowl_e = [round(float(x), 3) for x in env.bowl_pos_e()[0].tolist()]
                print(f"[PROBE] step {i+1}: state_0 cup_pos={cup_s0} hand_pos={hand_s0} | env.bowl_e={bowl_e}", flush=True)
        env.close()


main()
