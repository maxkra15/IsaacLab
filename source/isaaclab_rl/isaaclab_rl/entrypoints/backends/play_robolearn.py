# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RoboLearn playback backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.seed import configure_seed

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

from ..common import (
    add_common_play_args,
    apply_env_overrides,
    apply_video_recording,
    close_env,
    create_isaaclab_env,
    enable_cameras_for_video,
    pre_launch_video_config,
    resolve_seed,
    run_playback,
    set_hydra_args,
    show_run_summary,
    startup_screen,
)
from .cli_args_robolearn import add_robolearn_args, resolve_checkpoint, restore_agent_config, select_agent

logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse RoboLearn playback arguments."""
    parser = argparse.ArgumentParser(description="Play a Torch or Warp-NN checkpoint with RoboLearn.")
    add_common_play_args(
        parser, agent_default=None, agent_help="Name of the RoboLearn agent configuration entry point."
    )
    add_robolearn_args(parser)
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    if args_cli.frontend != "torch":
        parser.error("RoboLearn requires --frontend torch to preserve pre-reset terminal observations.")
    select_agent(args_cli)
    enable_cameras_for_video(args_cli)
    set_hydra_args(hydra_args)
    return args_cli


def run(argv: list[str]) -> None:
    """Play a deterministic RoboLearn policy using the shared visualizer lifecycle."""
    args_cli = _parse_args(argv)
    with startup_screen(args_cli, num_stages=3) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
        if isinstance(env_cfg, DirectMARLEnvCfg):
            raise ValueError("RoboLearn currently requires a single-agent environment.")
        log_root_path = os.path.abspath(os.path.join("logs", "robolearn", agent_cfg.experiment_name))
        checkpoint_path = resolve_checkpoint(args_cli, agent_cfg, log_root_path)
        restore_agent_config(agent_cfg, args_cli, checkpoint_path)
        pre_launch_video_config(env_cfg, args_cli)
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
            from ...robolearn import RoboLearnRunner

            apply_env_overrides(args_cli, env_cfg)
            seed = resolve_seed(args_cli.seed)
            if seed is not None:
                agent_cfg.seed = seed
            agent_cfg.device = env_cfg.sim.device
            env_cfg.seed = agent_cfg.seed
            env_cfg.compute_final_obs = True
            show_run_summary(screen, args_cli, env_cfg, library="robolearn", action="play")
            log_dir = os.path.dirname(checkpoint_path)
            env_cfg.log_dir = log_dir
            apply_video_recording(env_cfg, log_dir, args_cli, subdir="play", checkpoint_path=checkpoint_path)

            screen.stage("Creating environment")
            env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
            cleanup.callback(lambda: close_env(env))
            screen.stage("Loading policy")
            logger.info(f"Loading model checkpoint from: {checkpoint_path}")
            runner = RoboLearnRunner(env, agent_cfg.to_dict(), device=agent_cfg.device)
            runner.load(checkpoint_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)
            obs, _ = env.reset()

            def step() -> None:
                nonlocal obs
                obs, _, _, _, _ = env.step(policy(obs))

            screen.close()
            run_playback(step, dt=env.unwrapped.step_dt, args_cli=args_cli, env_cfg=env_cfg)
