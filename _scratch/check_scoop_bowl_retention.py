# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import warp as wp

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation


parser = argparse.ArgumentParser(description="Check that particles initialized inside the Franka scoop bowl stay inside.")
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--media-fill-frac", type=float, default=0.04)
parser.add_argument("--min-retained-frac", type=float, default=0.85)
parser.add_argument("--voxel-size", type=float, default=None)
add_launcher_args(parser)
args_cli = parser.parse_args()


def _qrot_t(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    xyz, w = q[..., :3], q[..., 3:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _sample_local_bowl_points(env, count: int) -> np.ndarray:
    rng = np.random.default_rng(11)
    voxel = float(env.cfg.voxel_size)
    margin = float(env.cfg.collider_margin)
    wall_clearance = max(0.25 * voxel, 1.5 * margin)
    bottom_clearance = max(0.5 * voxel, 1.5 * margin)
    top_clearance = max(0.5 * voxel, 1.5 * margin)

    z_min = -float(env._bowl_floor) + bottom_clearance
    z_max = float(env._bowl_lip) - top_clearance
    if z_max <= z_min:
        raise RuntimeError(
            f"Scoop bowl interior is under-resolved for retention test: z_min={z_min:.5f}, z_max={z_max:.5f}. "
            "Reduce --voxel-size or increase the EE bowl scale."
        )

    z0 = -float(env._bowl_floor)
    z1 = float(env._bowl_lip)
    points = np.empty((count, 3), dtype=np.float32)
    written = 0
    while written < count:
        batch = max(256, count - written)
        z = rng.uniform(z_min, z_max, size=batch).astype(np.float32)
        t = np.clip((z - z0) / max(z1 - z0, 1.0e-6), 0.0, 1.0)
        radius = env._bowl_inner_bottom_r + t * (env._bowl_inner_top_r - env._bowl_inner_bottom_r)
        radius = np.maximum(radius - wall_clearance, 0.2 * voxel)
        theta = rng.uniform(0.0, 2.0 * math.pi, size=batch).astype(np.float32)
        rho = np.sqrt(rng.uniform(0.0, 1.0, size=batch).astype(np.float32)) * radius
        candidate = np.column_stack((rho * np.cos(theta), rho * np.sin(theta), z)).astype(np.float32)
        take = min(candidate.shape[0], count - written)
        points[written:written + take] = candidate[:take]
        written += take
    return points


def _seed_particles_in_scoop_bowl(env) -> None:
    from isaaclab_newton.physics import NewtonManager

    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    local = torch.tensor(_sample_local_bowl_points(env, env._num_particles), device=env.device)
    bowl_pos, bowl_quat = env._bowl_pose_w()
    world = bowl_pos[:, None, :] + _qrot_t(
        bowl_quat[:, None, :].expand(-1, env._num_particles, -1),
        local[None, :, :].expand(env.num_envs, -1, -1),
    )

    state_0 = NewtonManager.get_state_0()
    state_1 = NewtonManager.get_state_1()
    wp.to_torch(state_0.particle_q)[env._particle_ids[env_ids]] = world
    wp.to_torch(state_0.particle_qd)[env._particle_ids[env_ids]] = 0.0
    wp.to_torch(state_1.particle_q)[env._particle_ids[env_ids]] = world
    wp.to_torch(state_1.particle_qd)[env._particle_ids[env_ids]] = 0.0
    NewtonManager._mark_transforms_dirty()


def main() -> None:
    from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg

    cfg = FrankaScoopEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.media_fill_frac = args_cli.media_fill_frac
    cfg.sim.device = str(args_cli.device)
    if args_cli.voxel_size is not None:
        cfg.voxel_size = args_cli.voxel_size
    # Probe-only solver override to dodge the committed fixed+graph+2^21 OOM (user owns the real solver cfg).
    cfg.grid_type = "sparse"
    cfg.use_cuda_graph = False
    cfg.__post_init__()

    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg)
        try:
            env.reset()
            action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            # Reset writes joint state directly; take one zero-action step so Newton recomputes FK before
            # we sample the moving scoop-bowl pose used for particle placement.
            env.step(action)
            _seed_particles_in_scoop_bowl(env)
            initial = env.count_in_bowl().clone()
            for _ in range(args_cli.steps):
                env.step(action)
            final = env.count_in_bowl().clone()
            retained = final / torch.clamp(initial, min=1.0)
            from isaaclab_newton.physics import NewtonManager

            pe = wp.to_torch(NewtonManager.get_state_0().particle_q)[env._particle_ids] - env.env_origins[:, None, :]
            bowl_pos, bowl_quat = env._bowl_pose_w()
            local = _qrot_t(
                env._quat_conj(bowl_quat)[:, None, :].expand(-1, env._num_particles, -1),
                pe - (bowl_pos - env.env_origins)[:, None, :],
            )
            local_min = local.amin(dim=1)
            local_max = local.amax(dim=1)
            radial_max = torch.linalg.norm(local[..., :2], dim=-1).amax(dim=1)
            finite = env.state_finite().all().item()
            print(
                "[retention] "
                f"particles={env._num_particles} initial={initial.tolist()} final={final.tolist()} "
                f"retained={retained.tolist()} finite={finite} "
                f"local_min={local_min.tolist()} local_max={local_max.tolist()} radial_max={radial_max.tolist()}",
                flush=True,
            )
            if not finite or bool((retained < args_cli.min_retained_frac).any()):
                raise SystemExit(1)
        finally:
            env.close()


if __name__ == "__main__":
    main()
