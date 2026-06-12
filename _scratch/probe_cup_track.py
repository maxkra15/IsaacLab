"""Headless: does the cup VISUAL prim (ScoopBowlVisual) track the live cup body pose via the authored
xform updated by _update_cup_visual_xform? Drives the arm and compares the prim world translate to body_q.

    ./scoop_run.sh -p _scratch/probe_cup_track.py --device cuda:0 --headless
"""

import argparse

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()


def main() -> None:
    import torch
    import warp as wp
    from pxr import Usd, UsdGeom

    from isaaclab_newton.physics import NewtonManager
    from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg

    cfg = FrankaScoopEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = str(args_cli.device)
    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg)
        env.reset()
        env.spawn_kit_visuals()  # authors ScoopBowlVisual + seeds the xform ops
        stage = env.sim.stage
        scoop_id = int(env._scoop_body_ids_l[0])

        def cup_prim_world():
            p = stage.GetPrimAtPath("/World/envs/env_0/ScoopBowlVisual")
            m = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            return [round(float(x), 4) for x in m.ExtractTranslation()]

        act = torch.tensor([[0.0, -0.7, -0.4, 0.2]], device=env.device)
        for i in range(60):
            env.step(act)
            env._update_cup_visual_xform()  # fires from _sync_kit_visuals under Kit; call manually headless
            if i % 20 == 19:
                bq = wp.to_torch(NewtonManager.get_state_0().body_q)
                body_t = [round(float(x), 4) for x in bq[scoop_id, :3].tolist()]
                print(f"[TRACK] step {i+1}: ScoopBowlVisual world={cup_prim_world()} | cup body_q={body_t}",
                      flush=True)
        env.close()


main()
