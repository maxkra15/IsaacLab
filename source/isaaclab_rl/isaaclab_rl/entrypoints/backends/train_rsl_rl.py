# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL training logic for the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import logging
import os
import platform
import time

from packaging import version

from isaaclab.app import add_launcher_args, report_activity

from isaaclab_rl.entrypoints._torchrun import resolve_log_dir, should_write_run_metadata
from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import (
    CHECKPOINT_SELECTORS,
    add_common_train_args,
    apply_env_overrides,
    apply_video_recording,
    configure_io_descriptors,
    create_isaaclab_env,
    dump_train_configs,
    enable_cameras_for_video,
    pre_launch_video_config,
    resolve_checkpoint_selector,
    scoped_torch_backend_flags,
    set_hydra_args,
    show_run_summary,
    startup_screen,
    validate_distributed_device,
    wrap_training_capture,
    write_run_manifest,
)

import isaaclab_tasks  # noqa: F401

logger = logging.getLogger(__name__)

RSL_RL_VERSION = "5.0.1"
_MISSING_ENVIRONMENT_STEP = object()

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _bind_environment_step_checkpoint(runner: object, env: object) -> bool:
    """Bind an optional algorithm checkpoint hook to the live environment step."""
    algorithm = getattr(runner, "alg", None)
    bind_provider = getattr(algorithm, "bind_environment_step_provider", None)
    if bind_provider is None:
        return False
    if not callable(bind_provider):
        raise TypeError("bind_environment_step_provider must be callable.")
    base_env = env.unwrapped
    bind_provider(lambda: base_env.common_step_counter)
    return True


def _validate_environment_step(value: object, name: str) -> int:
    """Validate an environment control-step value used during resume."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _synchronize_environment_step(step: int, *, distributed: bool, device: str) -> int:
    """Broadcast rank zero's restored step and assert every worker loaded it."""
    if not distributed:
        return step

    import torch

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("Distributed environment-step restore requires an initialized process group.")
    local_step = torch.tensor(step, dtype=torch.int64, device=device)
    synchronized_step = local_step.clone()
    torch.distributed.broadcast(synchronized_step, src=0)
    mismatch_count = torch.ne(local_step, synchronized_step).to(dtype=torch.int64)
    torch.distributed.all_reduce(mismatch_count, op=torch.distributed.ReduceOp.SUM)
    if mismatch_count.item() != 0:
        raise RuntimeError(
            "Distributed workers restored different environment control steps: "
            f"local={local_step.item()}, rank_zero={synchronized_step.item()}, "
            f"mismatched_workers={mismatch_count.item()}."
        )
    return int(synchronized_step.item())


def _restore_environment_step_checkpoint(
    runner: object,
    env: object,
    *,
    num_steps_per_env: int,
    distributed: bool,
) -> int | None:
    """Restore an optional exact environment step before the runner reads observations."""
    algorithm = getattr(runner, "alg", None)
    restored_step = getattr(
        algorithm,
        "restored_environment_common_step_counter",
        _MISSING_ENVIRONMENT_STEP,
    )
    if restored_step is _MISSING_ENVIRONMENT_STEP:
        return None

    if restored_step is None:
        completed_updates = _validate_environment_step(
            getattr(algorithm, "completed_updates", None),
            "completed_updates",
        )
        rollout_steps = _validate_environment_step(num_steps_per_env, "num_steps_per_env")
        if rollout_steps == 0:
            raise ValueError("num_steps_per_env must be positive.")
        restored_step = completed_updates * rollout_steps
        logger.warning(
            "Checkpoint predates exact environment-step persistence; inferred common_step_counter=%d "
            "from %d completed updates and %d rollout steps.",
            restored_step,
            completed_updates,
            rollout_steps,
        )
    else:
        restored_step = _validate_environment_step(
            restored_step,
            "restored_environment_common_step_counter",
        )

    base_env = env.unwrapped
    restored_step = _synchronize_environment_step(
        restored_step,
        distributed=distributed,
        device=base_env.device,
    )
    base_env.common_step_counter = restored_step

    compute_curriculum_step = getattr(getattr(base_env, "curriculum_manager", None), "compute_step", None)
    if callable(compute_curriculum_step):
        compute_curriculum_step()

    observation_manager = getattr(base_env, "observation_manager", None)
    compute_observations = getattr(observation_manager, "compute", None)
    if callable(compute_observations):
        base_env.obs_buf = compute_observations()
    return restored_step


def _check_rsl_rl_version() -> str:
    """Check that the installed RSL-RL version is supported."""
    installed_version = metadata.version("rsl-rl-lib")
    if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
        if platform.system() == "Windows":
            cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
        else:
            cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
        print(
            f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
            f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
            f"\n\n\t{' '.join(cmd)}\n"
        )
        raise SystemExit(1)
    return installed_version


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse RSL-RL training arguments."""
    from isaaclab.utils.string import list_intersection, string_to_callable

    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
    add_common_train_args(
        parser,
        agent_default="rsl_rl_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument(
        "--external_callback",
        default=None,
        help="Fully qualified path to an externally defined callback.",
    )
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    args_cli, remaining_args = setup_preset_cli(parser, argv, agent_library="rsl_rl")
    enable_cameras_for_video(args_cli)

    remaining_args_env_registration = None
    if args_cli.external_callback:
        external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
        remaining_args_env_registration = external_callback_function()

    set_hydra_args(list_intersection(remaining_args, remaining_args_env_registration))
    return args_cli


def run(argv: list[str]) -> None:
    """Train an RSL-RL agent while restoring the caller's Torch backend settings."""
    args_cli = _parse_args(argv)
    with scoped_torch_backend_flags(
        cuda_matmul_allow_tf32=True,
        cudnn_allow_tf32=True,
        cudnn_deterministic=False,
        cudnn_benchmark=False,
    ):
        _run(args_cli)


def _run(args_cli: argparse.Namespace) -> None:
    """Execute RSL-RL training with parsed arguments."""
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    from isaaclab.app import launch_simulation
    from isaaclab.envs import DirectMARLEnvCfg
    from isaaclab.utils.seed import configure_seed

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config

    installed_version = _check_rsl_rl_version()

    with startup_screen(args_cli, num_stages=3) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
        pre_launch_video_config(env_cfg, args_cli=args_cli)
        show_run_summary(screen, args_cli, env_cfg, library="rsl_rl", action="train")
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli):
            agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
            apply_env_overrides(args_cli, env_cfg)
            agent_cfg.max_iterations = (
                args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
            )

            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

            env_cfg.seed = agent_cfg.seed
            validate_distributed_device(args_cli)

            if args_cli.distributed:
                global_rank = int(os.getenv("RANK", "0"))
                agent_cfg.device = env_cfg.sim.device

                seed = agent_cfg.seed + global_rank
                env_cfg.seed = seed
                agent_cfg.seed = seed

            log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
            print(f"[INFO] Logging experiment in directory: {log_root_path}")
            log_dir = resolve_log_dir(
                log_root_path,
                agent_cfg.run_name,
                distributed=args_cli.distributed,
            )
            print(f"Exact experiment name requested from command line: {os.path.basename(log_dir)}")
            if should_write_run_metadata(args_cli.distributed):
                write_run_manifest(
                    log_dir,
                    library="rsl_rl",
                    task=args_cli.task,
                    metadata={"agent": args_cli.agent},
                )

            configure_io_descriptors(env_cfg, args_cli, logger)
            env_cfg.log_dir = log_dir
            apply_video_recording(env_cfg, log_dir, args_cli)

            screen.stage("Creating environment")
            env = create_isaaclab_env(
                args_cli.task,
                env_cfg,
                args_cli,
                convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
            )

            if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
                if args_cli.checkpoint in CHECKPOINT_SELECTORS:
                    resume_path = resolve_checkpoint_selector(
                        log_root_path,
                        args_cli.checkpoint,
                        library="rsl_rl",
                        task=args_cli.task,
                        checkpoint_pattern=r"model_.*\.pt",
                        metadata={"agent": args_cli.agent},
                    )
                else:
                    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

            env = wrap_training_capture(env, log_dir, args_cli)

            screen.stage("Preparing agent")
            start_time = time.time()
            report_activity("Wrapping environment")
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            report_activity(None)

            report_activity("Building policy")
            if agent_cfg.class_name == "OnPolicyRunner":
                runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
            elif agent_cfg.class_name == "DistillationRunner":
                runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
            else:
                raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
            report_activity(None)
            _bind_environment_step_checkpoint(runner, env)

            # configure_seed must run after runner construction so torch determinism does not disturb its initialization
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)

            runner.add_git_repo_to_log(__file__)
            if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
                print(f"[INFO]: Loading model checkpoint from: {resume_path}")
                runner.load(resume_path)
                _restore_environment_step_checkpoint(
                    runner,
                    env,
                    num_steps_per_env=agent_cfg.num_steps_per_env,
                    distributed=args_cli.distributed,
                )

            if should_write_run_metadata(args_cli.distributed):
                dump_train_configs(log_dir, env_cfg, agent_cfg)

            screen.close()
            try:
                runner.learn(
                    num_learning_iterations=agent_cfg.max_iterations,
                    init_at_random_ep_len=agent_cfg.init_at_random_ep_len,
                )
                print(f"Training time: {round(time.time() - start_time, 2)} seconds")
                env.close()
            except KeyboardInterrupt:
                pass
