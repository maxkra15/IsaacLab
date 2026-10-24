# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect continuous physical episodes for a coupled material world model.

Example (run from the repository root)::

    uv run python -m examples.mpm_world_model.collect_franka_pour \
        --checkpoint /path/model_7450.pt --reset-dataset /path/reset_dataset.pt \
        --output /path/dataset --num-envs 32 --steps 360 --mode mpm --headless
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.mpm_world_model.franka_pour_state import FrankaPourState, configure_source_paths, source_digest
from examples.mpm_world_model.recording import EpisodeMetrics, EpisodeWriter, atomic_json


def argument_parser(evaluation: bool = False) -> argparse.ArgumentParser:
    """Create a parser without importing or initializing a simulator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--newton-root", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reset-dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument(
        "--warmup-steps", type=int, default=0, help="Recorded warmup excluded from aggregate eval counters."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("mpm", "pbd", "xpbd"), default="mpm")
    parser.add_argument("--particle-iterations", type=int, default=1)
    parser.add_argument("--global-substeps", type=int, default=2)
    parser.add_argument("--media-substeps", type=int, default=2)
    parser.add_argument("--collision-decimation", type=int, default=0)
    parser.add_argument("--compliance", type=float, default=0.0029478469130082285)
    parser.add_argument("--contact-capacity", type=int, default=128)
    parser.add_argument("--particle-samples", type=int, default=468)
    parser.add_argument("--perturb-std", type=float, default=0.0 if evaluation else 0.15)
    parser.add_argument("--perturb-fraction", type=float, default=0.0 if evaluation else 0.5)
    parser.add_argument("--max-bytes", type=int, default=3_000_000_000)
    parser.add_argument("--without-material", action="store_true")
    parser.add_argument("--metrics-only", action="store_true", default=evaluation)
    parser.add_argument(
        "--fine-tune-iterations", type=int, default=0, help="Actual real-MPM PPO iterations; requires --train-only."
    )
    parser.add_argument("--fine-tune-rollout-steps", type=int, default=96)
    parser.add_argument("--fine-tune-epochs", type=int, help="Defaults to the task PPO configuration (5).")
    parser.add_argument(
        "--fine-tune-learning-rate", type=float, help="Override optimizer LR after loading; default inherits it."
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Save an MPM-finetuned RSL checkpoint without collecting/evaluating transitions.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    return parser


def validate_arguments(parser: argparse.ArgumentParser, args) -> None:
    """Reject invalid budgets before simulator initialization."""
    if (
        min(
            args.num_envs,
            args.steps,
            args.particle_iterations,
            args.global_substeps,
            args.media_substeps,
            args.particle_samples,
        )
        < 1
    ):
        parser.error("World, step, iteration, substep, and particle budgets must be positive.")
    if min(args.warmup_steps, args.collision_decimation) < 0:
        parser.error("Warmup and collision-decimation budgets must be nonnegative.")
    if not 0 <= args.perturb_fraction <= 1 or args.perturb_std < 0 or args.compliance < 0:
        parser.error("Perturbation fractions/std and compliance must be nonnegative and bounded.")
    for path in (args.checkpoint, args.reset_dataset):
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    if args.max_bytes < 1:
        parser.error("max-bytes must be positive")
    if args.fine_tune_iterations < 0:
        parser.error("fine-tune-iterations must be nonnegative")
    if args.fine_tune_rollout_steps < 1 or (args.fine_tune_epochs is not None and args.fine_tune_epochs < 1):
        parser.error("Fine-tune rollout steps and epochs must be positive.")
    if args.fine_tune_learning_rate is not None and (
        not math.isfinite(args.fine_tune_learning_rate) or args.fine_tune_learning_rate <= 0
    ):
        parser.error("fine-tune-learning-rate must be finite and positive")
    if bool(args.fine_tune_iterations) != args.train_only:
        parser.error(
            "Use --train-only together with positive --fine-tune-iterations; evaluate the saved checkpoint separately."
        )
    if args.train_only and args.mode != "mpm":
        parser.error("Fine-tuning in this entry point requires real --mode mpm.")


def reset_exploration_state(done, noise, selected, generator, perturb_fraction):
    """Reset action noise only in worlds ended by an RSL environment step."""
    done = done.bool()
    if done.any():
        selected[done] = torch.rand((int(done.sum()), 1), generator=generator, device=done.device) < perturb_fraction
        noise[done] = 0.0
    return done


def override_loaded_learning_rate(algorithm, learning_rate: float | None) -> list[float]:
    """Preserve the loaded optimizer rate unless an explicit override is given."""
    if learning_rate is not None:
        algorithm.learning_rate = learning_rate
        for group in algorithm.optimizer.param_groups:
            group["lr"] = learning_rate
    return [float(group["lr"]) for group in algorithm.optimizer.param_groups]


def fine_tune_runner(runner, args, runner_cfg: dict) -> dict:
    """Run and time actual MPM PPO after restoring/overriding its optimizer."""
    start_iteration = int(runner.current_learning_iteration)
    actual_rates = override_loaded_learning_rate(runner.alg, args.fine_tune_learning_rate)
    runner.cfg["algorithm"]["learning_rate"] = float(runner.alg.learning_rate)
    runner_cfg["algorithm"]["learning_rate"] = float(runner.alg.learning_rate)
    atomic_json(args.output / "runner_cfg.json", runner_cfg)
    phases = []
    original_log = runner.logger.log

    def observed_log(*positional, **keywords):
        phases.append(
            {
                "iteration": int(keywords["it"]),
                "collection_seconds": float(keywords["collect_time"]),
                "learning_seconds": float(keywords["learn_time"]),
                "loss": {key: float(value) for key, value in keywords["loss_dict"].items()},
            }
        )
        return original_log(*positional, **keywords)

    runner.logger.log = observed_log
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    learn_started = time.perf_counter()
    try:
        runner.learn(num_learning_iterations=args.fine_tune_iterations, init_at_random_ep_len=False)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
    finally:
        runner.logger.log = original_log
    learn_seconds = time.perf_counter() - learn_started
    checkpoint = args.output / "model_mpm_finetuned.pt"
    runner.save(str(checkpoint))
    transitions = args.num_envs * int(runner_cfg["num_steps_per_env"]) * args.fine_tune_iterations
    return {
        "requested_ppo_iterations": args.fine_tune_iterations,
        "loaded_iteration": start_iteration,
        "saved_iteration": int(runner.current_learning_iteration),
        "rollout_steps_per_env_per_iteration": int(runner_cfg["num_steps_per_env"]),
        "environment_transitions": transitions,
        "epochs_per_iteration": int(runner.alg.num_learning_epochs),
        "minibatches_per_epoch": int(runner.alg.num_mini_batches),
        "actual_optimizer_learning_rates": actual_rates,
        "learn_wall_seconds": learn_seconds,
        "environment_steps_per_second": transitions / learn_seconds,
        "rsl_phase_timings": phases,
        "rsl_collection_seconds": sum(item["collection_seconds"] for item in phases),
        "rsl_learning_seconds": sum(item["learning_seconds"] for item in phases),
        "timing_scope": (
            "Synchronized runner.learn wall includes rollout, PPO, logging and native automatic saves; "
            "excludes environment/network construction and explicit final export. First-step capture/compilation, "
            "if any, remains included. RSL phase times are its own CPU wall diagnostics."
        ),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "evaluation_performed": False,
    }


def run(args) -> dict:
    """Run native physics with pre-reset recording and the existing RSL actor."""
    configure_source_paths(args.repo, args.newton_root)
    import newton
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.app import launch_simulation
    from isaaclab.managers import RecorderTerm, RecorderTermCfg
    from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg
    from isaaclab.paths import ISAACLAB_ROOT
    from isaaclab.utils import to_dict, validate

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_tasks.contrib.franka_pour.config.franka.agents.rsl_rl_ppo_cfg import (
        FrankaPourResetDatasetPPORunnerCfg,
    )
    from isaaclab_tasks.contrib.franka_pour.pour_env import FrankaPourEnv
    from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import FrankaPourResetDatasetEnvCfg, _resolve_pour_solver_tree

    assert Path(ISAACLAB_ROOT).resolve() == args.repo.resolve(), ISAACLAB_ROOT
    if args.newton_root:
        assert Path(newton.__file__).resolve().parents[1] == args.newton_root.resolve()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = FrankaPourResetDatasetEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.reset_dataset_path = str(args.reset_dataset.resolve())
    cfg.curriculum_freeze = True
    cfg.sim.physics.num_substeps = args.global_substeps
    cfg.sim.physics.collision_decimation = args.collision_decimation
    cfg.sim.visualizer_cfgs = []
    tree = _resolve_pour_solver_tree(cfg)
    tree.media_entry.substeps = args.media_substeps
    tree.media_solver.fidelity = "mpm" if args.mode == "mpm" else "particles"
    tree.media_solver.particle_iterations = args.particle_iterations
    cfg.log_dir = str(args.output / "runtime")
    agent_cfg = FrankaPourResetDatasetPPORunnerCfg()
    agent_cfg.device, agent_cfg.seed = args.device, args.seed
    if args.train_only:
        agent_cfg.num_steps_per_env = args.fine_tune_rollout_steps
        if args.fine_tune_epochs is not None:
            agent_cfg.algorithm.num_learning_epochs = args.fine_tune_epochs
        if args.fine_tune_learning_rate is not None:
            agent_cfg.algorithm.learning_rate = args.fine_tune_learning_rate
    atomic_json(args.output / "runner_cfg.json", to_dict(agent_cfg))
    provenance = {
        "task_id": "franka_pour",
        "seed": args.seed,
        "mode": args.mode,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "reset_dataset_sha256": hashlib.sha256(args.reset_dataset.read_bytes()).hexdigest(),
        "isaaclab_python_sha256": source_digest(args.repo / "source"),
        "newton_python_sha256": source_digest(Path(newton.__file__).resolve().parent),
        "arguments": vars(args),
        "observation_dim": 84,
        "critic_observation_dim": 86,
        "action_dim": 8,
        "coupling": {
            "media_substeps": args.media_substeps,
            "rigid_substeps": tree.arm_entry.substeps,
            "global_substeps": args.global_substeps,
            "collision_decimation": args.collision_decimation,
        },
    }
    atomic_json(args.output / "provenance.json", provenance)
    writer = None if args.metrics_only or args.train_only else EpisodeWriter(args.output / "episodes", provenance)
    counters = {
        "completed": 0,
        "success": 0,
        "nonfinite_states": 0,
        "first_success": 0,
        "first_failure": 0,
        "steps": 0,
        "terminations": {},
    }
    state = {"adapter": None, "teacher_action": None, "perturbation": None, "complete": False, "is_warmup": False}
    first_done = torch.zeros(args.num_envs, dtype=torch.bool)
    episode_metrics = EpisodeMetrics(args.num_envs, cfg.sim.dt * cfg.decimation)
    term_order = []

    class PhysicalEpisodeRecorder(RecorderTerm):
        def _adapter(self):
            if state["adapter"] is None:
                state["adapter"] = FrankaPourState(self._env, args.particle_samples, not args.without_material)
            return state["adapter"]

        def record_post_reset(self, env_ids):
            ids = range(self._env.num_envs) if env_ids is None else env_ids
            if isinstance(ids, slice):
                ids = range(self._env.num_envs)[ids]
            ids = torch.as_tensor(list(ids), dtype=torch.long, device=self._env.device)
            episode_metrics.reset(ids.cpu(), self._env.reset_dataset_row_id[ids].cpu())
            if writer is None:
                return None, None
            adapter = self._adapter()
            captured = adapter.capture()
            for index in ids:
                index = int(index)
                writer.start_episode(
                    index,
                    {key: value[index] for key, value in captured.items()},
                    adapter.episode_metadata(index, args.seed),
                )
            return None, None

        def record_post_step(self):
            env = self._env
            adapter = self._adapter()
            captured = adapter.capture(env.obs_buf)
            terminated, truncated = env.reset_terminated, env.reset_time_outs
            done = (terminated | truncated).cpu()
            success = env.termination_manager.get_term("success").cpu()
            counters["steps"] += 1
            if not state["is_warmup"]:
                counters["completed"] += int(done.sum())
                counters["success"] += int((done & success).sum())
            counters["nonfinite_states"] += int((~captured["finite_mask"]).sum())
            newly_done = done & ~first_done
            counters["first_success"] += int((newly_done & success).sum())
            counters["first_failure"] += int((newly_done & ~success).sum())
            first_done.logical_or_(done)
            if not term_order:
                term_order.extend(env.termination_manager.active_terms)
            terms = torch.stack([env.termination_manager.get_term(name) for name in term_order], dim=-1)
            episode_metrics.update(
                env.reward_buf,
                env.episode_length_buf,
                env.reset_dataset_row_id,
                terminated,
                truncated,
                success,
                terms,
                term_order,
                warmup=state["is_warmup"],
            )
            if not state["is_warmup"]:
                for column, name in enumerate(term_order):
                    counters["terminations"][name] = counters["terminations"].get(name, 0) + int(terms[:, column].sum())
            if writer is not None:
                action = env.action_manager.action
                for world in range(env.num_envs):
                    extra = {
                        "termination_terms": terms[world],
                        "teacher_action": state["teacher_action"][world],
                        "perturbation": state["perturbation"][world],
                        "reward_terms": env.reward_manager._step_reward[world] * env.step_dt,
                        "warmup": torch.tensor(state["is_warmup"]),
                    }
                    writer.append_transition(
                        world,
                        action[world],
                        env.reward_buf[world],
                        terminated[world],
                        truncated[world],
                        {key: value[world] for key, value in captured.items()},
                        extra,
                    )
                    if done[world]:
                        writer.finish_episode(world)
            return None, None

    recorder_cfg = RecorderManagerBaseCfg(dataset_export_mode=0)
    recorder_cfg.physical_episode = RecorderTermCfg(class_type=PhysicalEpisodeRecorder)
    # PPO owns rollout counters while training. A fresh evaluator keeps training
    # transitions out of reported policy metrics.
    cfg.recorders = {} if args.train_only else recorder_cfg
    original_init = None
    if args.mode == "xpbd":
        from newton._src.solvers.implicit_mpm.particle_solver_xpbd import _XPBDParticleSolver

        original_init = newton.solvers.SolverImplicitMPM.__init__

        def initialize(solver, *positional, **keywords):
            original_init(solver, *positional, **keywords)
            old = solver._particle_solver
            solver._particle_solver = _XPBDParticleSolver(
                solver.model,
                iterations=args.particle_iterations,
                compliance=args.compliance,
                contact_capacity=args.contact_capacity,
                relaxation=old.relaxation,
                friction=old.friction,
                cohesion=old.cohesion,
            )

        newton.solvers.SolverImplicitMPM.__init__ = initialize
    validate(cfg)
    try:
        with launch_simulation(cfg, args):
            env = FrankaPourEnv(cfg)
            try:
                wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
                runner = OnPolicyRunner(
                    wrapped,
                    to_dict(agent_cfg),
                    log_dir=str(args.output / "ppo") if args.train_only else None,
                    device=args.device,
                )
                runner.load(str(args.checkpoint), map_location=args.device)
                counters["loaded_learning_iteration"] = int(runner.current_learning_iteration)
                counters["loaded_optimizer_learning_rates"] = [
                    float(group["lr"]) for group in runner.alg.optimizer.param_groups
                ]
                print(
                    json.dumps(
                        {
                            "loaded_learning_iteration": counters["loaded_learning_iteration"],
                            "actor_observation_dim": 84,
                            "critic_observation_dim": 86,
                            "action_dim": 8,
                            "mode": args.mode,
                            "train_only": args.train_only,
                        }
                    ),
                    flush=True,
                )
                if args.train_only:
                    counters["fine_tune"] = fine_tune_runner(runner, args, to_dict(agent_cfg))
                    state["complete"] = True
                    return counters
                policy = runner.get_inference_policy(device=args.device)
                obs = wrapped.get_observations()
                generator = torch.Generator(device=args.device).manual_seed(args.seed + 10000)
                noise = torch.zeros((args.num_envs, 8), device=args.device)
                selected = (
                    torch.rand((args.num_envs, 1), generator=generator, device=args.device) < args.perturb_fraction
                )
                with torch.inference_mode():
                    for step in range(args.steps + args.warmup_steps):
                        state["is_warmup"] = step < args.warmup_steps
                        teacher = policy(obs)
                        noise = 0.9 * noise + 0.4358899 * torch.randn(
                            noise.shape, generator=generator, device=args.device
                        )
                        perturbation = noise * args.perturb_std * selected
                        action = (teacher + perturbation).clamp(-1.0, 1.0)
                        state["teacher_action"], state["perturbation"] = teacher, action - teacher
                        obs, _, done, _ = wrapped.step(action)
                        done = reset_exploration_state(done, noise, selected, generator, args.perturb_fraction)
                        policy.reset(done)
                        helper = state["adapter"].solver._particle_solver
                        if args.mode == "xpbd":
                            helper.check_contact_capacity()
                        if writer and step % 32 == 0:
                            written = sum(path.stat().st_size for path in (args.output / "episodes").glob("*.pt"))
                            if written > args.max_bytes:
                                raise RuntimeError("Collection storage budget exceeded.")
                        if (step + 1) % 30 == 0:
                            print(
                                json.dumps(
                                    {
                                        "policy_step": step + 1,
                                        "completed": counters["completed"],
                                        "success": counters["success"],
                                        "nonfinite": counters["nonfinite_states"],
                                    }
                                ),
                                flush=True,
                            )
                state["complete"] = True
            finally:
                env.close()
    finally:
        if original_init is not None:
            newton.solvers.SolverImplicitMPM.__init__ = original_init
        if writer is not None:
            writer.close(complete=state["complete"])
        if not args.train_only:
            counters.update(episode_metrics.summary())
        counters.update(
            {
                "complete": state["complete"],
                "first_unfinished": int((~first_done).sum()),
                "first_cohort_size": args.num_envs,
                "first_cohort_full_horizon": args.steps + args.warmup_steps >= 360
                and state["complete"]
                and not args.train_only,
                "aggregate_evaluation_policy_steps": args.steps,
                "warmup_policy_steps": args.warmup_steps,
                "phase": "real_mpm_fine_tuning" if args.train_only else "policy_rollout",
                "success_fraction_completed": counters["success"] / max(counters["completed"], 1),
                "termination_order": term_order,
                "provenance": "provenance.json",
            }
        )
        atomic_json(args.output / "evaluation.json", counters)
    return counters


def main() -> None:
    parser = argument_parser()
    args = parser.parse_args()
    validate_arguments(parser, args)
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
