"""Headless: dump robot-base / table / source+target container world bounds to size the table shift.

    ./scoop_run.sh -p _scratch/probe_layout.py --device cuda:0 --headless
"""

import argparse

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()


def main() -> None:
    import numpy as np
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
        env.spawn_kit_visuals()
        stage = env.sim.stage
        bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], True)

        def world_bounds(paths):
            lo = np.array([1e9, 1e9, 1e9]); hi = -lo
            for p in paths:
                pr = stage.GetPrimAtPath(p)
                if not pr.IsValid():
                    continue
                r = bbc.ComputeWorldBound(pr).ComputeAlignedRange()
                if r.IsEmpty():
                    continue
                lo = np.minimum(lo, np.array(r.GetMin())); hi = np.maximum(hi, np.array(r.GetMax()))
            return lo, hi

        # robot base
        bq = wp.to_torch(NewtonManager.get_state_0().body_q).detach().cpu().numpy()
        labels = list(NewtonManager.get_model().body_label)
        base_i = next((i for i, l in enumerate(labels) if str(l).endswith("panda_link0")), None)
        if base_i is not None:
            print(f"[LAYOUT] panda_link0 world pos = {[round(float(x),3) for x in bq[base_i,:3]]}", flush=True)

        # robot collision extent (all Robot prims)
        robot_paths = [str(p.GetPath()) for p in stage.Traverse()
                       if "/env_0/Robot/" in str(p.GetPath()) and p.IsA(UsdGeom.Gprim)]
        rlo, rhi = world_bounds(robot_paths)
        print(f"[LAYOUT] robot visual x-range = [{rlo[0]:.3f}, {rhi[0]:.3f}]  y=[{rlo[1]:.3f},{rhi[1]:.3f}]", flush=True)

        tlo, thi = world_bounds(["/World/envs/env_0/Table"])
        print(f"[LAYOUT] TABLE  x=[{tlo[0]:.3f},{thi[0]:.3f}] y=[{tlo[1]:.3f},{thi[1]:.3f}] z=[{tlo[2]:.3f},{thi[2]:.3f}]", flush=True)

        for label in ("Source", "Target"):
            paths = [f"/World/envs/env_0/{label}_{k}" for k in range(5)]
            lo, hi = world_bounds(paths)
            print(f"[LAYOUT] {label:6s} x=[{lo[0]:.3f},{hi[0]:.3f}] y=[{lo[1]:.3f},{hi[1]:.3f}] z=[{lo[2]:.3f},{hi[2]:.3f}]", flush=True)
        print(f"[LAYOUT] cfg: table_half={cfg.table_half} table_center_xy={cfg.table_center_xy} "
              f"source={cfg.source_center} target={cfg.target_center} inner_half={cfg.container_inner_half}", flush=True)
        env.close()


main()
