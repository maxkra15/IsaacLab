"""Diagnose why the decorative gripped-cup USD hovers: dump the auto-fit numbers and compare the
decorative cup's post-fit local bbox against the collider mesh's known local frame.

    ./scoop_run.sh -p _scratch/probe_cup_fit.py --device cuda:0 --headless
"""

import argparse

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()


def main() -> None:
    import numpy as np
    from pxr import Usd, UsdGeom

    from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg

    cfg = FrankaScoopEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = str(args_cli.device)
    # keep the DEFAULT gripped_cup_usd_path (the decorative coffee cup) -- that's what we're diagnosing

    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg)
        env.reset()
        stage = env.sim.stage

        total_h = env._ee_bowl_bottom_thickness + (env._ee_bowl_height - env._ee_bowl_bottom_thickness)
        outer_top_r = env._ee_bowl_inner_top_radius + env._ee_bowl_wall_thickness
        print(f"[FIT] collider mesh local frame: z in [0, {total_h:.4f}], XY radius outer_top={outer_top_r:.4f}, "
              f"target_diameter={2*outer_top_r:.4f}", flush=True)

        env.spawn_kit_visuals()
        cup_root = "/World/envs/env_0/ScoopBowl"

        # ScoopBowl world transform (driven by native sync = collider body origin)
        sb = stage.GetPrimAtPath(cup_root)
        m_sb = UsdGeom.Xformable(sb).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        print(f"[FIT] ScoopBowl world translate = {[round(float(x),4) for x in m_sb.ExtractTranslation()]}", flush=True)

        # The referenced decorative cup
        cv = stage.GetPrimAtPath(f"{cup_root}/CupVisual")
        if not cv.IsValid():
            print("[FIT] CupVisual prim INVALID (no decorative cup authored)", flush=True)
            env.close()
            return

        # raw referenced bbox (before the CupVisual local xform) -- what _compute_gripped_cup_visual_fit saw
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], True)
        # local bound = bound in CupVisual's own space (includes its authored fit xform)
        local_b = bbox_cache.ComputeLocalBound(cv).GetRange()
        lo, hi = np.array(local_b.GetMin()), np.array(local_b.GetMax())
        print(f"[FIT] CupVisual local bound (after fit xform): min={[round(float(x),4) for x in lo]} "
              f"max={[round(float(x),4) for x in hi]} size={[round(float(x),4) for x in (hi-lo)]}", flush=True)

        # world bound of the decorative cup vs the collider body origin
        world_b = bbox_cache.ComputeWorldBound(cv).ComputeAlignedRange()
        wlo, whi = np.array(world_b.GetMin()), np.array(world_b.GetMax())
        print(f"[FIT] CupVisual WORLD bound: min={[round(float(x),4) for x in wlo]} "
              f"max={[round(float(x),4) for x in whi]}", flush=True)
        sbt = np.array([float(x) for x in m_sb.ExtractTranslation()])
        print(f"[FIT] decorative-cup center - ScoopBowl origin = "
              f"{[round(float(x),4) for x in (0.5*(wlo+whi) - sbt)]} "
              f"(expect XY~0, Z~{0.5*total_h:.3f} if aligned)", flush=True)

        # the local xform authored on CupVisual (fit_offset+user_offset, scale)
        xf = UsdGeom.Xformable(cv)
        for op in xf.GetOrderedXformOps():
            print(f"[FIT]   CupVisual xformOp {op.GetOpName()} = {op.Get()}", flush=True)
        env.close()


main()
