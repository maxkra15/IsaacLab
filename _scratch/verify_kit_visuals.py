"""Headless verification of the Kit visual authoring (cup/box/table + MediaParticles).

Forces spawn_kit_visuals() (bypassing the kit-visualizer guard) and inspects the authored USD prims.
spawn_kit_visuals() seeds the points via _update_media_particles_visual (a direct USD write), so the
MediaParticles points must already hold live particle_q positions here -- no Fabric stage needed.

    ./scoop_run.sh -p _scratch/verify_kit_visuals.py --device cuda:0 --headless
"""

import argparse

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()


def main() -> None:
    import torch
    from pxr import UsdGeom

    from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg

    cfg = FrankaScoopEnvCfg()
    cfg.scene.num_envs = 2
    cfg.sim.device = str(args_cli.device)
    cfg.gripped_cup_usd_path = ""  # procedural cup -> no remote download; isolates the authoring path

    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg)
        env.reset()
        act = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        for _ in range(20):
            obs, *_ = env.step(act)
        print(f"[VERIFY] stepped 20x; obs finite={bool(torch.isfinite(obs['policy']).all())} "
              f"src0={int(env.count_in_source()[0])}", flush=True)

        pid = getattr(env, "_particle_ids", None)
        print(f"[VERIFY] pre-spawn: visualizers={sorted(env.sim.resolve_visualizer_types())} "
              f"authored_envs={sorted(env._authored_visual_envs)} "
              f"particle_ids_shape={None if pid is None else tuple(pid.shape)} "
              f"media_prims={len(env._media_particle_prims)}", flush=True)
        # Headless resolves no kit visualizer, so load_managers skips this; call it directly to exercise the
        # real Kit path (static visuals already authored early -> no-op; media authored now with _particle_ids).
        env.spawn_kit_visuals()
        stage = env.sim.stage
        for env_id in range(cfg.scene.num_envs):
            root = f"/World/envs/env_{env_id}"
            for sub in ("ScoopBowl", "Source_0", "Source_3", "Target_0", "Table", "MediaParticles"):
                p = stage.GetPrimAtPath(f"{root}/{sub}")
                print(f"[VERIFY] {root}/{sub}: valid={p.IsValid()} type={p.GetTypeName() if p.IsValid() else '-'}", flush=True)
            mp = stage.GetPrimAtPath(f"{root}/MediaParticles")
            if mp.IsValid():
                pts = UsdGeom.Points(mp).GetPointsAttr().Get()
                w = UsdGeom.Points(mp).GetWidthsAttr().Get()
                c = UsdGeom.Points(mp).GetDisplayColorAttr().Get()
                p0 = [round(float(x), 3) for x in pts[0]] if pts else None
                seeded = bool(pts is not None and any(abs(x) > 1e-9 for x in pts[0]))
                print(f"[VERIFY]   MediaParticles points={len(pts) if pts is not None else 0} "
                      f"widths={len(w) if w is not None else 0} width0={float(w[0]) if w else -1:.4f} "
                      f"color={list(c[0]) if c else None} p0={p0} seeded={seeded}", flush=True)
        prim_summary = [(off, cnt) for _attr, off, cnt in env._media_particle_prims]
        print(f"[VERIFY] _media_particle_prims (offset, count) per env: {prim_summary}", flush=True)

        # Confirm the per-render updater MOVES the points (not just seeds them): drive the cup into the pile,
        # re-run the update Kit calls each frame, and check the authored USD points changed.
        mp0 = stage.GetPrimAtPath("/World/envs/env_0/MediaParticles")
        before = list(UsdGeom.Points(mp0).GetPointsAttr().Get())
        drive = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        drive[:, :2] = torch.tensor([0.0, -0.7], device=env.device)  # push -y into the media
        for _ in range(40):
            env.step(drive)
        env._update_media_particles_visual()
        after = list(UsdGeom.Points(mp0).GetPointsAttr().Get())
        moved = sum(1 for a, b in zip(before, after) if (a - b).GetLength() > 1e-4)
        print(f"[VERIFY] per-render update moved {moved}/{len(after)} points; "
              f"p0 {[round(float(x),3) for x in before[0]]} -> {[round(float(x),3) for x in after[0]]}", flush=True)
        env.close()


main()
