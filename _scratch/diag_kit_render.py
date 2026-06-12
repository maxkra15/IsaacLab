"""Drive the cup into the pile under Kit and check whether the cup-visual transform and the
MediaParticles Fabric points actually update (isolates sync-writes-correctly vs Kit-doesn't-render).

    SCOOP_KIT_DBG=1 DISPLAY=:1 ./scoop_run.sh -p _scratch/diag_kit_render.py --device cuda:0 --visualizer kit
"""

import argparse

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()


def _cup_usd_pos(stage):
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath("/World/envs/env_0/ScoopBowl")
    if not prim.IsValid():
        return None
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation()
    return [round(float(t[0]), 3), round(float(t[1]), 3), round(float(t[2]), 3)]


def _usd_points_sample(stage):
    """Read the live MediaParticles USD points (what Kit renders now) for env_0."""
    try:
        from pxr import UsdGeom

        prim = stage.GetPrimAtPath("/World/envs/env_0/MediaParticles")
        if not prim.IsValid():
            return "usd prim invalid"
        pts = UsdGeom.Points(prim).GetPointsAttr().Get()
        if pts is None:
            return "points=None"
        n = len(pts)
        p0 = [round(float(c), 3) for c in pts[0]]
        pm = [round(float(c), 3) for c in pts[n // 2]]
        return f"n={n} p0={p0} pmid={pm}"
    except Exception as exc:  # noqa: BLE001
        return f"read-failed: {exc}"


def main() -> None:
    import torch

    from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg

    cfg = FrankaScoopEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = str(args_cli.device)
    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg)
        env.reset()
        stage = env.sim.stage
        # drive the cup -y (into the pile) + down + tilt to pour/scoop
        act = torch.tensor([[0.0, -0.7, -0.5, 0.3]], device=env.device)
        print(f"[REND] INITIAL cup_usd={_cup_usd_pos(stage)} bowl_e={[round(float(x),3) for x in env.bowl_pos_e()[0]]} "
              f"usd_pts=({_usd_points_sample(stage)})", flush=True)
        for i in range(90):
            obs, *_ = env.step(act)
            if i % 30 == 29:
                pq = env._default_particle_q if False else None  # noqa
                live = None
                try:
                    from isaaclab_newton.physics import NewtonManager
                    pq_live = NewtonManager.get_state_0().particle_q
                    import warp as wp
                    live = [round(float(c), 3) for c in wp.to_torch(pq_live)[env._particle_ids[0, 0]].tolist()]
                except Exception as exc:  # noqa: BLE001
                    live = f"err {exc}"
                print(f"[REND] step {i+1}: cup_usd={_cup_usd_pos(stage)} "
                      f"bowl_e={[round(float(x),3) for x in env.bowl_pos_e()[0]]} "
                      f"in_bowl={int(env.count_in_bowl()[0])} src={int(env.count_in_source()[0])} "
                      f"particle_q[0]={live} usd_pts=({_usd_points_sample(stage)})", flush=True)
        env.close()


main()
