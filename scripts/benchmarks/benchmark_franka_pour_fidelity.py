# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare pouring material modes with identical policy and task interfaces.

Run one case per process so CUDA graphs, solver allocations, and PPO storage are
released before increasing the world count. Rendering is optional and should be
measured separately from training throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import newton
import torch
import warp as wp
from isaaclab_newton.physics import NewtonMPMManager
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.paths import ISAACLAB_ROOT
from isaaclab.utils import to_dict, validate

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.contrib.franka_pour.config.franka.agents.rsl_rl_ppo_cfg import FrankaPourResetDatasetPPORunnerCfg
from isaaclab_tasks.contrib.franka_pour.mdp.observations import particle_fractions_obs
from isaaclab_tasks.contrib.franka_pour.media import media_particle_count
from isaaclab_tasks.contrib.franka_pour.pour_env import FrankaPourEnv
from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import (
    FrankaPourResetDatasetEnvCfg,
    PourMediaFidelityStageCfg,
    _resolve_pour_solver_tree,
)


def main() -> None:
    """Run a bounded checkpoint rollout, PPO continuation, or comparison clip."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mpm", "s2", "s2_cached", "particles2", "particles4"), default="mpm")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--warmup_steps", type=int, default=16)
    parser.add_argument("--train_iterations", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reset_dataset", type=Path, help="Optional local copy of the canonical reset artifact.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu_threads", type=int, default=1)
    parser.add_argument(
        "--mpm_iterations", type=int, default=5, help="Matched S2-mode iteration budget; default MPM stays 24."
    )
    parser.add_argument("--voxel_size", type=float)
    parser.add_argument("--collider_basis", help="Explicit contact-basis override; use S2 for voxel curricula.")
    parser.add_argument("--warmstart_contact_impulses", action="store_true")
    parser.add_argument("--particle_radius", type=float)
    parser.add_argument(
        "--particle_friction", type=float, help="Approximate pair friction; collider friction stays fixed."
    )
    parser.add_argument("--switch_success_rate", type=float, help="Optional success-gated transition to full MPM.")
    parser.add_argument("--switch_voxel_size", type=float)
    parser.add_argument("--switch_particle_radius", type=float)
    parser.add_argument("--curriculum_min_episodes", type=int, default=128)
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--freeze_reset_sampler", action="store_true", help="Use frozen reset replay in every compared mode."
    )
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    args = parser.parse_args()
    if (
        args.num_envs <= 0
        or args.steps <= 0
        or args.warmup_steps < 0
        or args.train_iterations < 0
        or args.cpu_threads <= 0
    ):
        parser.error("World and step budgets must be positive; warmup and training budgets must be nonnegative.")
    if args.video and args.train_iterations:
        parser.error("Record video and benchmark training in separate runs.")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = True

    cfg = FrankaPourResetDatasetEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    if args.reset_dataset is not None:
        cfg.reset_dataset_path = str(args.reset_dataset.resolve())
    cfg.curriculum_freeze = args.freeze_reset_sampler
    solver_tree = _resolve_pour_solver_tree(cfg)
    media_solver = solver_tree.media_solver
    if args.mode.startswith("particles"):
        media_solver.fidelity = "particles"
        media_solver.particle_iterations = int(args.mode.removeprefix("particles"))
    elif args.mode.startswith("s2"):
        media_solver.collider_basis = "S2"
        media_solver.solver = "gauss-seidel"
        media_solver.max_iterations = args.mpm_iterations
        media_solver.warmstart_contact_impulses = args.mode == "s2_cached"
    if args.voxel_size is not None:
        # The spawn lattice remains fixed, so solver grid changes never alter observations or mass.
        media_solver.voxel_size = args.voxel_size
    if args.collider_basis is not None:
        media_solver.collider_basis = args.collider_basis
    if args.warmstart_contact_impulses:
        media_solver.warmstart_contact_impulses = True
    if args.particle_friction is not None:
        media_solver.particle_friction = args.particle_friction
    if args.switch_success_rate is not None:
        cfg.media_fidelity_stages = (
            PourMediaFidelityStageCfg(
                success_rate=args.switch_success_rate,
                fidelity="mpm",
                voxel_size=args.switch_voxel_size,
                particle_radius=args.switch_particle_radius,
            ),
        )
        cfg.media_fidelity_min_episodes = args.curriculum_min_episodes

    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = args.output.parent / args.output.stem
    cfg.log_dir = str(log_dir)
    if args.video:
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg  # noqa: PLC0415

        cfg.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(0.9, 0.65, 0.5),
                lookat=(0.5, 0.0, 0.1),
                show_particles=True,
                particle_color=(0.85, 0.72, 0.45),
            )
        ]
        cfg.video_recorders = [
            VideoRecorderCfg(
                source="viz:newton_gl",
                output_dir=str(log_dir / "videos"),
                output_filename_prefix=args.mode,
                video_length=args.steps + args.warmup_steps,
            )
        ]
    validate(cfg)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    report = {
        "mode": args.mode,
        "hostname": socket.gethostname(),
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "num_envs": args.num_envs,
        "particles_per_env": media_particle_count(cfg.scene.media),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint is not None else None,
        "seed": args.seed,
        "cpu_threads": torch.get_num_threads(),
        "torch_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "env_spacing_m": cfg.scene.env_spacing,
        "physics_step_s": cfg.sim.dt,
        "policy_step_s": cfg.sim.dt * cfg.decimation,
        "physics_substeps": cfg.sim.physics.num_substeps,
        "media_entry_substeps": solver_tree.media_entry.substeps,
        "rigid_entry_substeps": solver_tree.arm_entry.substeps,
        "proxy_mass_scale": solver_tree.proxy.mass_scale,
        "video_enabled": args.video,
        "isaaclab_commit": revision.stdout.strip() or None,
        "workflow_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "isaaclab_python_sha256": _python_source_digest(Path(ISAACLAB_ROOT) / "source"),
        "newton_python_sha256": _python_source_digest(Path(newton.__file__).resolve().parent),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
        "warp_version": wp.__version__,
        "torch_version": str(torch.__version__),
        "rsl_rl_version": version("rsl-rl-lib"),
        "mujoco_warp_version": version("mujoco-warp"),
        "initial_media_solver": to_dict(media_solver),
    }
    setup_started = time.perf_counter()
    with launch_simulation(cfg, args):
        env = FrankaPourEnv(cfg)
        try:
            model = NewtonMPMManager.get_model()
            shape_types = model.shape_type.numpy()
            shape_flags = model.shape_flags.numpy()
            rigid_shapes = (shape_flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
            particle_shapes = (shape_flags & int(newton.ShapeFlags.COLLIDE_PARTICLES)) != 0
            report["collision_geometry"] = {
                "rigid_convex_hulls": int(((shape_types == int(newton.GeoType.CONVEX_MESH)) & rigid_shapes).sum()),
                "rigid_triangle_meshes": int(((shape_types == int(newton.GeoType.MESH)) & rigid_shapes).sum()),
                "particle_triangle_meshes": int(((shape_types == int(newton.GeoType.MESH)) & particle_shapes).sum()),
            }
            print(f"Collision geometry: {report['collision_geometry']}", flush=True)
            report["initial_particle_count"] = int(model.particle_count)
            report["initial_total_particle_mass_kg"] = wp.to_torch(model.particle_mass).sum().item()
            if args.particle_radius is not None:
                NewtonMPMManager.set_particle_radius(args.particle_radius, entry_name="media")
            report["initial_active_fidelity"] = NewtonMPMManager.get_fidelity(entry_name="media")
            print("Preparing the RSL-RL environment wrapper.", flush=True)
            wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
            agent_cfg = FrankaPourResetDatasetPPORunnerCfg()
            agent_cfg.device = args.device
            agent_cfg.seed = args.seed
            print("Constructing the PPO policy and value networks.", flush=True)
            runner = OnPolicyRunner(
                wrapped,
                to_dict(agent_cfg),
                log_dir=str(log_dir) if args.train_iterations else None,
                device=args.device,
            )
            if args.checkpoint is not None:
                print(f"Loading the saved policy and optimizer from {args.checkpoint}.", flush=True)
                runner.load(str(args.checkpoint), map_location=args.device)
            report["loaded_learning_iteration"] = runner.current_learning_iteration
            policy = runner.get_inference_policy(device=args.device)
            obs = wrapped.get_observations()
            report["observation_dimensions"] = {key: int(value.shape[-1]) for key, value in obs.items()}
            report["actor_observation_dimension"] = sum(
                int(obs[key].shape[-1]) for key in agent_cfg.obs_groups["actor"]
            )
            report["critic_observation_dimension"] = sum(
                int(obs[key].shape[-1]) for key in agent_cfg.obs_groups["critic"]
            )
            report["action_dimension"] = int(wrapped.num_actions)
            report["setup_wall_seconds"] = time.perf_counter() - setup_started
            with torch.inference_mode():
                print(f"Warming up {args.warmup_steps} policy steps.", flush=True)
                warmup_started = time.perf_counter()
                for _ in range(args.warmup_steps):
                    obs, _, done, _ = wrapped.step(policy(obs))
                    policy.reset(done)
                wp.synchronize_device(args.device)
                report["warmup_wall_seconds"] = time.perf_counter() - warmup_started
                initial_particles = env.scene["media"].data.particle_pos_w.torch.clone()
                initial_joints = env.scene["robot"].data.joint_pos.torch.clone()
                physics_particles = wp.to_torch(NewtonMPMManager.get_state_0().particle_q).reshape(args.num_envs, -1, 3)
                centroids = physics_particles[:4].mean(dim=1)
                report["physics_particle_centroids_first_worlds_m"] = centroids.tolist()
                report["env_origins_first_worlds_m"] = env.scene.env_origins[:4].tolist()
                report["first_worlds_min_centroid_distance_m"] = (
                    torch.pdist(centroids).min().item() if centroids.shape[0] > 1 else None
                )
                outcomes = torch.zeros(4, dtype=torch.int64, device=args.device)
                wp.synchronize_device(args.device)
                print(f"Timing {args.steps} policy steps.", flush=True)
                started = time.perf_counter()
                for _ in range(args.steps):
                    obs, _, done, _ = wrapped.step(policy(obs))
                    policy.reset(done)
                    outcomes[0] += done.sum()
                    outcomes[1] += env.termination_manager.get_term("success").sum()
                    outcomes[2] += env.termination_manager.get_term("spill").sum()
                    outcomes[3] += env.termination_manager.get_term("particle_out_of_bounds").sum()
                wp.synchronize_device(args.device)
                elapsed = time.perf_counter() - started
            completed, successes, spills, out_of_bounds = outcomes.tolist()
            report["rollout"] = {
                "policy_steps": args.steps,
                "wall_seconds": elapsed,
                "policy_batches_per_second": args.steps / elapsed,
                "environment_steps_per_second": args.steps * args.num_envs / elapsed,
                "completed_episodes": completed,
                "successful_episodes": successes,
                "spill_terminations": spills,
                "particle_out_of_bounds_terminations": out_of_bounds,
                "success_rate": successes / completed if completed else None,
            }
            if args.train_iterations:
                training_start_step = env.common_step_counter
                training_start_fidelity = NewtonMPMManager.get_fidelity(entry_name="media")
                wp.synchronize_device(args.device)
                started = time.perf_counter()
                runner.learn(num_learning_iterations=args.train_iterations, init_at_random_ep_len=False)
                wp.synchronize_device(args.device)
                elapsed = time.perf_counter() - started
                transitions = args.train_iterations * agent_cfg.num_steps_per_env * args.num_envs
                report["training"] = {
                    "iterations": args.train_iterations,
                    "start_policy_step": training_start_step,
                    "end_policy_step": env.common_step_counter,
                    "start_fidelity": training_start_fidelity,
                    "end_fidelity": NewtonMPMManager.get_fidelity(entry_name="media"),
                    "environment_steps": transitions,
                    "wall_seconds": elapsed,
                    "environment_steps_per_second": transitions / elapsed,
                    "includes_optimizer_logging_and_checkpoint_save": True,
                }
            particles = env.scene["media"].data.particle_pos_w.torch
            joints = env.scene["robot"].data.joint_pos.torch
            report["particles_finite"] = bool(torch.isfinite(particles).all().item())
            report["robot_finite"] = bool(torch.isfinite(joints).all().item())
            report["particle_max_displacement_m"] = float((particles - initial_particles).abs().max().item())
            report["joint_max_displacement_rad"] = float((joints - initial_joints).abs().max().item())
            source_fraction, target_fraction, spill_fraction = particle_fractions_obs(env).mean(dim=0).tolist()
            report["final_current_episode_mean_fractions"] = {
                "source": source_fraction,
                "target": target_fraction,
                "spill": spill_fraction,
                "retained_in_cups": source_fraction + target_fraction,
            }
            report["final_active_fidelity"] = NewtonMPMManager.get_fidelity(entry_name="media")
            curriculum_metrics = dict(env.curriculum_manager.get_active_iterable_terms(0)).get("media_fidelity", [])
            report["media_fidelity_curriculum"] = dict(
                zip(("stage", "success_rate", "last_transition_policy_step"), curriculum_metrics)
            )
            report["final_particle_count"] = int(model.particle_count)
            report["final_total_particle_mass_kg"] = wp.to_torch(model.particle_mass).sum().item()
            radii = wp.to_torch(NewtonMPMManager.get_model().particle_radius)
            report["final_particle_radius_m"] = {"min": radii.min().item(), "max": radii.max().item()}
            report["final_media_solver"] = to_dict(media_solver)
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info(args.device)
                report["gpu"] = {
                    "name": torch.cuda.get_device_name(args.device),
                    "total_memory_bytes": total_bytes,
                    "all_processes_used_memory_bytes": total_bytes - free_bytes,
                    "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                }
        finally:
            env.close()
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))
    if not report["particles_finite"] or not report["robot_finite"]:
        raise RuntimeError("The comparison produced nonfinite particle or robot state.")
    if args.steps >= 16 and (
        report["particle_max_displacement_m"] < 1.0e-8 or report["joint_max_displacement_rad"] < 1.0e-8
    ):
        raise RuntimeError("Particles or the Franka did not move during the comparison.")


def _python_source_digest(root: Path) -> str:
    """Fingerprint Python sources for archived deployments without Git metadata."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    main()
