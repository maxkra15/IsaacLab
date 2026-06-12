"""A/B test the MPM-state reset: settle the pile (build residual elastic strain), re-reset, and measure
the residual strain + the post-reset particle 'pop' (max particle speed) with vs without the fix.

    ./scoop_run.sh -p _scratch/probe_reset_pop.py --device cuda:0 --headless                 # fix ON (default)
    ./scoop_run.sh -p _scratch/probe_reset_pop.py --device cuda:0 --headless --no-reset-mpm  # fix OFF
"""
import argparse

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--no-reset-mpm", action="store_true")
parser.add_argument("--mug", action="store_true", help="use the old flared mug cup instead of the hemisphere")
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
    cfg.reset_mpm_particle_state = not args_cli.no_reset_mpm
    if args_cli.mug:
        cfg.ee_cup_shape = "mug"
    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg)
        env.reset()
        ids = env._particle_ids[0]
        zero = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)

        def strain_dev():
            F = wp.to_torch(NewtonManager.get_state_0().mpm.particle_elastic_strain)[ids]
            return float((F - torch.eye(3, device=F.device)).norm(dim=(1, 2)).mean())

        def max_speed():
            qd = wp.to_torch(NewtonManager.get_state_0().particle_qd)[ids]
            return float(qd.norm(dim=-1).max())

        # Baseline: settling of the FRESH spawned cone right after the very first reset (no prior dynamics).
        peak0 = 0.0
        for i in range(45):
            env.step(zero)
            peak0 = max(peak0, max_speed())
            if (i + 1) in (12, 30, 45):
                print(f"[POP] FRESH-SPAWN step {i+1:2d}: max_speed={max_speed():.3f} m/s (peak so far {peak0:.3f})", flush=True)
        env.reset()

        # Drive the cup INTO the pile (collider embedded in media)...
        dig = torch.tensor([[1.0, -0.4, -0.6, -0.3]], device=env.device)
        for _ in range(40):
            env.step(dig)
        # ...then LIFT it clear and let the media settle with NO active contact before resetting.
        clear = torch.tensor([[-1.0, 0.3, 1.0, 0.0]], device=env.device)
        for _ in range(50):
            env.step(clear)
        print(f"[POP] after lift+settle: max_speed={max_speed():.3f} m/s (should be calm before reset)", flush=True)
        print(f"[POP] pre-reset bowl_e={[round(float(x),3) for x in env.bowl_pos_e()[0].tolist()]} "
              f"(home ~0.17); reset will teleport it back", flush=True)
        print(f"[POP] reset_mpm={cfg.reset_mpm_particle_state} | BEFORE re-reset: ||F-I||={strain_dev():.4f}", flush=True)
        env.reset_scoop_scene(torch.tensor([0], device=env.device))
        print(f"[POP]   immediately AFTER reset: ||F-I||={strain_dev():.4f} (fix -> ~0)", flush=True)
        # introspect the MPM sub-solver's collider buffers
        try:
            mpm = NewtonManager._solver.solver("media")
            ms = env._media_entry_state
            print(f"[POP]   mpm solver={type(mpm).__name__} has _last_step_data={hasattr(mpm,'_last_step_data')}", flush=True)
            lsd = mpm._last_step_data
            bqp = getattr(lsd, "body_q_prev", None)
            cbq = getattr(mpm._mpm_model, "collider_body_q", None)
            import warp as wp2
            def t(a):
                return None if a is None else wp2.to_torch(a)
            bqp_t, cbq_t, ms_t = t(bqp), t(cbq), t(ms.body_q)
            print(f"[POP]   shapes: body_q_prev={None if bqp_t is None else tuple(bqp_t.shape)} "
                  f"collider_body_q={None if cbq_t is None else tuple(cbq_t.shape)} "
                  f"media.body_q={tuple(ms_t.shape)}", flush=True)
            if bqp_t is not None and cbq_t is not None and bqp_t.shape == cbq_t.shape:
                print(f"[POP]   ||collider_body_q - body_q_prev|| (pos) = "
                      f"{float((cbq_t[:, :3]-bqp_t[:, :3]).norm()):.4f}  (0 => no teleport seen by solver)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[POP]   introspect failed: {exc}", flush=True)
        src0 = int(env.count_in_source()[0])
        peak = 0.0
        for i in range(45):  # would-be "pre-settle": does the kick decay + does media stay in the box?
            env.step(zero)
            s = max_speed()
            peak = max(peak, s)
            if (i + 1) in (1, 3, 6, 12, 20, 30, 45):
                print(f"[POP]   step {i+1:2d}: max_speed={s:.3f} m/s  in_source={int(env.count_in_source()[0])}", flush=True)
        print(f"[POP] PEAK={peak:.3f} m/s | source {src0} -> {int(env.count_in_source()[0])} (media kept?)", flush=True)
        env.close()


main()
