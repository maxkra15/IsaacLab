from __future__ import annotations

import argparse

import torch

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--container-geometry", choices=("bucket", "pour_bowl", "box"), default=None)
parser.add_argument("--steps", type=int, default=0)
add_launcher_args(parser)
args_cli = parser.parse_args()


def main() -> None:
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg

    cfg = FrankaScoopEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = str(args_cli.device)
    if args_cli.container_geometry is not None:
        cfg.container_geometry = args_cli.container_geometry

    with launch_simulation(cfg, args_cli):
        from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv
        from isaaclab_newton.physics import NewtonManager
        from newton import ShapeFlags

        env = FrankaScoopEnv(cfg)
        try:
            model = NewtonManager.get_model()
            state = NewtonManager.get_state_0()
            env.reset()
            shape_body = model.shape_body.numpy()
            shape_flags = model.shape_flags.numpy()
            collide_shapes = int(ShapeFlags.COLLIDE_SHAPES)
            collide_particles = int(ShapeFlags.COLLIDE_PARTICLES)
            visible = int(ShapeFlags.VISIBLE)

            print("[inspect] geometry", flush=True)
            print(f"  bucket_inner_radius={cfg.bucket_inner_radius:.4f}", flush=True)
            print(f"  bucket_wall_thickness={cfg.bucket_wall_thickness:.4f}", flush=True)
            print(f"  bucket_height={cfg.bucket_height:.4f}", flush=True)
            print(f"  bucket_bottom_thickness={cfg.bucket_bottom_thickness:.4f}", flush=True)
            cup_outer_top = cfg.ee_cup_inner_top_radius + cfg.ee_cup_wall_thickness
            cup_outer_bottom = cfg.ee_cup_inner_bottom_radius + cfg.ee_cup_wall_thickness
            print(f"  cup_outer_diameter_top={2.0 * cup_outer_top:.4f}", flush=True)
            print(f"  cup_outer_diameter_bottom={2.0 * cup_outer_bottom:.4f}", flush=True)
            print(f"  radial_clearance_to_cup_top={cfg.bucket_inner_radius - cup_outer_top:.4f}", flush=True)
            print(f"  source_center={cfg.source_center} target_center={cfg.target_center}", flush=True)

            print("[inspect] selected body poses", flush=True)
            body_q = state.body_q.numpy()
            for bid, label in enumerate(model.body_label):
                label = str(label)
                if any(
                    key in label
                    for key in (
                        "panda_link0",
                        "panda_hand",
                        "panda_leftfinger",
                        "panda_rightfinger",
                        "ScoopBowl",
                        "SourceBucket",
                        "TargetBucket",
                    )
                ):
                    q = body_q[bid]
                    print(
                        f"  body {bid:03d} {label} pos=({q[0]:+.4f},{q[1]:+.4f},{q[2]:+.4f}) "
                        f"quat=({q[3]:+.4f},{q[4]:+.4f},{q[5]:+.4f},{q[6]:+.4f})",
                        flush=True,
                    )

            print("[inspect] selected shapes", flush=True)
            for sid, label in enumerate(model.shape_label):
                label = str(label)
                body = int(shape_body[sid])
                body_label = str(model.body_label[body]) if body >= 0 else "<static>"
                if any(
                    key in label or key in body_label
                    for key in (
                        "panda_link0",
                        "panda_hand",
                        "panda_leftfinger",
                        "panda_rightfinger",
                        "ScoopBowl",
                        "SourceBucket",
                        "TargetBucket",
                        "Table",
                        "ground",
                    )
                ):
                    flags = int(shape_flags[sid])
                    print(
                        f"  shape {sid:03d} body={body:03d} flags="
                        f"{'S' if flags & collide_shapes else '-'}{'P' if flags & collide_particles else '-'}"
                        f"{'V' if flags & visible else '-'} "
                        f"label={label} body_label={body_label}",
                        flush=True,
                    )

            pairs = [tuple(pair) for pair in getattr(model, "shape_collision_filter_pairs", [])]
            table_ids = {sid for sid, label in enumerate(model.shape_label) if str(label).endswith("/Table")}
            base_ids = {
                sid
                for sid, label in enumerate(model.shape_label)
                if "panda_link0" in str(label)
                or (
                    int(shape_body[sid]) >= 0
                    and str(model.body_label[int(shape_body[sid])]).endswith("/panda_link0")
                )
            }
            filtered_base_table = [
                pair
                for pair in pairs
                if (pair[0] in table_ids and pair[1] in base_ids)
                or (pair[1] in table_ids and pair[0] in base_ids)
            ]
            print(f"[inspect] filter_pairs={len(pairs)} base_table_filtered={filtered_base_table}", flush=True)
            if args_cli.steps > 0:
                base_id = next(i for i, label in enumerate(model.body_label) if str(label).endswith("/panda_link0"))
                hand_id = next(i for i, label in enumerate(model.body_label) if str(label).endswith("/panda_hand"))
                start_base = state.body_q.numpy()[base_id].copy()
                start_hand = state.body_q.numpy()[hand_id].copy()
                action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
                for _ in range(args_cli.steps):
                    env.step(action)
                end_base = state.body_q.numpy()[base_id].copy()
                end_hand = state.body_q.numpy()[hand_id].copy()
                print(
                    "[inspect] zero_step_delta "
                    f"base_pos={float(((end_base[:3] - start_base[:3]) ** 2).sum() ** 0.5):.8f} "
                    f"base_quat={float(((end_base[3:] - start_base[3:]) ** 2).sum() ** 0.5):.8f} "
                    f"hand_pos={float(((end_hand[:3] - start_hand[:3]) ** 2).sum() ** 0.5):.8f} "
                    f"hand_quat={float(((end_hand[3:] - start_hand[3:]) ** 2).sum() ** 0.5):.8f}",
                    flush=True,
                )
            print(f"[inspect] finite={torch.isfinite(env.obs_buf['policy']).all().item() if hasattr(env, 'obs_buf') else True}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
