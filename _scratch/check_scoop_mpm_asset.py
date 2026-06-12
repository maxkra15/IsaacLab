# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke-check the MPMObject-based scoop env + the dump-first curriculum stages.

For each curriculum stage: force the stage, reset, verify particle/media wiring
(MPMObject views vs Newton state), step a few frames with zero actions, and report
cup retention of the pre-load, joint-limit margins, target counts, and finiteness.
"""

from __future__ import annotations

import argparse

import torch

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Franka scoop MPMObject + curriculum-stage smoke check.")
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=40)
add_launcher_args(parser)
args_cli = parser.parse_args()


def main() -> None:
    import warp as wp

    from isaaclab_newton.physics import NewtonManager
    from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg

    cfg = FrankaScoopEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = str(args_cli.device)

    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg)
        try:
            obs, _ = env.reset()
            media = env.scene["media"]
            n_model = NewtonManager.get_model().particle_count
            n_asset = media.num_instances * media.particles_per_object
            view = media.data.particle_pos_w.torch
            raw = wp.to_torch(NewtonManager.get_state_0().particle_q)[env._particle_ids]
            view_matches_state = bool(torch.allclose(view, raw))
            print(
                f"[asset] model_particles={n_model} asset_particles={n_asset} "
                f"view_shape={tuple(view.shape)} view_matches_state={view_matches_state}",
                flush=True,
            )

            zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            tilt = zero.clone()
            tilt[:, 3] = 1.0  # pitch rate: tip the cup over to pour
            all_ids = torch.arange(env.num_envs, device=env.device)
            mgr = env.curriculum_manager
            stage_term = dict(zip(mgr._term_names, mgr._term_cfgs))["stage"].func
            for stage in range(len(cfg.curriculum_reset_pose)):
                stage_term.stage = stage
                stage_term._apply(env)
                env._reset_idx(all_ids)  # full reset path: events + action-manager term resets
                loaded = cfg.curriculum_cup_fill_count[stage] > 0
                n_cup0 = env.count_in_bowl()
                bowl0 = env.bowl_pos_e().clone()
                margins = (
                    torch.minimum(env.arm_joint_q() - env._arm_lo, env._arm_hi - env.arm_joint_q()).amin().item()
                )
                settle_resets = 0
                settle = 10 if loaded else args_cli.steps
                for _ in range(settle):
                    obs, _, terminated, truncated, _ = env.step(zero)
                    settle_resets += int((terminated | truncated).sum())
                n_cup_settled = env.count_in_bowl()
                drift = torch.linalg.norm(env.bowl_pos_e() - bowl0, dim=-1).max().item()
                if loaded:  # scripted dump: tilt the cup and watch the media reach the target
                    for _ in range(30):
                        obs, _, terminated, _, _ = env.step(tilt)
                succeeded = env.episode_succeeded.tolist()
                print(
                    f"[stage {stage} pose={cfg.curriculum_reset_pose[stage]!r} "
                    f"fill={cfg.curriculum_cup_fill_count[stage]}] "
                    f"in_cup@reset={n_cup0.tolist()} in_cup@settled={n_cup_settled.tolist()} "
                    f"settle_resets={settle_resets} bowl_drift={drift:.4f} m "
                    f"in_target@end={env.count_in_target().tolist()} "
                    f"succeeded={succeeded if loaded else 'n/a'} "
                    f"min_joint_margin={margins:.3f} rad "
                    f"finite={bool(env.state_finite().all())} obs_finite={bool(torch.isfinite(obs['policy']).all())}",
                    flush=True,
                )
        finally:
            env.close()


if __name__ == "__main__":
    main()
