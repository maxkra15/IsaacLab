"""Headless: verify the delivery-success termination is wired + the obs-debug viz authors/updates."""
import argparse
from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); add_launcher_args(parser); args_cli = parser.parse_args()

def main():
    import torch
    from pxr import Usd, UsdGeom
    from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg
    cfg = FrankaScoopEnvCfg(); cfg.scene.num_envs = 1; cfg.sim.device = str(args_cli.device)
    cfg.debug_vis_obs = True
    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg); env.reset()
        # termination wiring
        terms = list(env.termination_manager.active_terms)
        print(f"[OBS] termination terms = {terms}", flush=True)
        print(f"[OBS] target_success_count = {env.cfg.target_success_count}", flush=True)
        import isaaclab_tasks.contrib.franka_scoop.mdp as mdp
        d = mdp.delivered_success(env)
        print(f"[OBS] delivered_success shape={tuple(d.shape)} any={bool(d.any())} (expect all False, target empty)", flush=True)
        # obs viz
        env.spawn_kit_visuals()
        for _ in range(15):
            env.step(torch.zeros((1, env.action_manager.total_action_dim), device=env.device))
        env._update_obs_debug_visual()
        st = env.sim.stage
        hf = UsdGeom.Points(st.GetPrimAtPath("/World/envs/env_0/ObsHeightfield"))
        pts = hf.GetPointsAttr().Get(); col = hf.GetDisplayColorAttr().Get()
        zs = [float(p[2]) for p in pts]
        print(f"[OBS] ObsHeightfield pts={len(pts)} colors={len(col)} z=[{min(zs):.3f},{max(zs):.3f}]", flush=True)
        mk = UsdGeom.Points(st.GetPrimAtPath("/World/envs/env_0/ObsMarkers")).GetPointsAttr().Get()
        print(f"[OBS] ObsMarkers pts={len(mk)} (src/held/all/target) sample0={[round(float(c),3) for c in mk[0]]}", flush=True)
        env.close()
main()
