# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RoboLearn training backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import time
from datetime import datetime

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.seed import configure_seed

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

from ..common import (
    add_common_train_args,
    apply_env_overrides,
    apply_video_recording,
    close_env,
    create_isaaclab_env,
    dump_train_configs,
    enable_cameras_for_video,
    pre_launch_video_config,
    resolve_seed,
    set_hydra_args,
    show_run_summary,
    startup_screen,
    write_run_manifest,
)
from .cli_args_robolearn import add_robolearn_args, resolve_checkpoint, restore_agent_config, select_agent

logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse RoboLearn training arguments."""
    parser = argparse.ArgumentParser(description="Train a FlashSAC or Warp-NN PPO policy with RoboLearn.")
    add_common_train_args(
        parser,
        agent_default=None,
        agent_help="Name of the RoboLearn agent configuration entry point.",
        include_distributed=False,
    )
    add_robolearn_args(parser)
    parser.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    if args_cli.frontend != "torch":
        parser.error("RoboLearn requires --frontend torch to preserve pre-reset terminal observations.")
    if args_cli.capture_env_sensors > 0:
        parser.error(
            "RoboLearn does not currently support --capture_env_sensors; visualizer --video remains available."
        )
    select_agent(args_cli)
    enable_cameras_for_video(args_cli)
    set_hydra_args(hydra_args)
    return args_cli


def run(argv: list[str]) -> None:
    """Train a RoboLearn policy against the task's registered environment."""
    args_cli = _parse_args(argv)
    with startup_screen(args_cli, num_stages=2) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
        if isinstance(env_cfg, DirectMARLEnvCfg):
            raise ValueError("RoboLearn currently requires a single-agent environment.")
        log_root_path = os.path.abspath(os.path.join("logs", "robolearn", agent_cfg.experiment_name))
        checkpoint_path = None
        if args_cli.checkpoint is not None:
            checkpoint_path = resolve_checkpoint(args_cli, agent_cfg, log_root_path)
            restore_agent_config(agent_cfg, args_cli, checkpoint_path)
            log_root_path = os.path.abspath(os.path.join("logs", "robolearn", agent_cfg.experiment_name))
        if args_cli.algorithm is not None and args_cli.algorithm != agent_cfg.algorithm:
            raise ValueError("The requested algorithm does not match the selected agent configuration.")
        pre_launch_video_config(env_cfg, args_cli)
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
            # Keep optional learning dependencies out of task registration and unrelated CLI paths.
            from ...robolearn import RoboLearnRunner

            apply_env_overrides(args_cli, env_cfg)
            seed = resolve_seed(args_cli.seed)
            if seed is not None:
                agent_cfg.seed = seed
            if args_cli.max_iterations is not None:
                agent_cfg.max_iterations = args_cli.max_iterations
            if args_cli.run_name is not None:
                agent_cfg.run_name = args_cli.run_name
            agent_cfg.device = env_cfg.sim.device
            env_cfg.seed = agent_cfg.seed
            env_cfg.compute_final_obs = True
            show_run_summary(screen, args_cli, env_cfg, library="robolearn", action="train")

            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_name += f"_{agent_cfg.algorithm}"
            if agent_cfg.run_name:
                run_name += f"_{agent_cfg.run_name}"
            log_dir = os.path.join(log_root_path, run_name)
            logger.info(f"Logging experiment in directory: {log_dir}")
            write_run_manifest(
                log_dir,
                library="robolearn",
                task=args_cli.task,
                metadata={"agent": args_cli.agent, "algorithm": agent_cfg.algorithm},
            )
            dump_train_configs(log_dir, env_cfg, agent_cfg)
            env_cfg.log_dir = log_dir
            apply_video_recording(env_cfg, log_dir, args_cli)

            screen.stage("Creating environment")
            env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
            cleanup.callback(lambda: close_env(env))
            runner_type = RoboLearnRunner
            if agent_cfg.capture_rollout:
                if args_cli.task != "Isaac-Velocity-Flat-G1" or args_cli.video:
                    raise ValueError("capture_rollout currently supports headless Isaac-Velocity-Flat-G1 training.")
                from ...robolearn.captured_runner import CapturedG1Runner

                runner_type = CapturedG1Runner
            runner = runner_type(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
            if checkpoint_path is not None:
                runner.load(checkpoint_path)
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)

            screen.close()
            start_time = time.perf_counter()
            try:
                runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
            except KeyboardInterrupt:
                logger.info("RoboLearn training interrupted.")
            finally:
                logger.info(f"Training time: {time.perf_counter() - start_time:.2f} seconds")
