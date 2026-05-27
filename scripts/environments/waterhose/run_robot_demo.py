# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the literal Newton waterhose robot success demo as an IsaacLab task.

This script creates the manager-style task `Isaac-Waterhose-Robot-Demo-Play-v0`,
whose environment wraps `newton/examples/cable_robot/example_waterhose_scene2_insert_extract_success.py`.
The Newton success demo still owns model construction, two-way coupling, proxy
gripper bodies, collision pipelines, and the scripted state machine.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher
from isaaclab_tasks.manager_based.manipulation.waterhose.launch import (
    add_waterhose_teleop_args,
    create_waterhose_spacemouse_device,
    prepare_waterhose_launch,
)


parser = argparse.ArgumentParser(description="Run the Newton waterhose robot success demo through IsaacLab.")
parser.add_argument("--task", type=str, default="Isaac-Waterhose-Robot-Demo-Play-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Only 1 is supported by the reference demo wrapper.")
parser.add_argument("--max_steps", type=int, default=2000, help="Maximum manager steps to run.")
parser.add_argument("--newton_viewer", choices=("gl", "null", "usd"), default="gl", help="Viewer for the Newton demo.")
parser.add_argument("--primary_view", choices=("mujoco", "vbd"), default="mujoco", help="Reference demo primary view.")
parser.add_argument("--reference_newton_root", type=str, default="/home/maximiliank/Work/newton", help="Newton repo root.")
parser.add_argument("--reference_headless", action="store_true", help="Run Newton GL viewer headless.")
parser.add_argument("--teleop", action="store_true", help="Drive the reference demo manually with a SpaceMouse.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default=None,
    help="Teleop device. For kitless waterhose robot demo teleop, use spacemouse.",
)
parser.add_argument("--sensitivity", type=float, default=1.0, help="Teleop sensitivity scale.")
parser.add_argument("--debug_teleop", action="store_true", help="Print periodic teleop commands.")
add_waterhose_teleop_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Ensure the local Newton checkout (which contains `newton.examples`) wins
# before prepare_waterhose_launch() imports Newton dependencies.
reference_root = Path(args_cli.reference_newton_root).expanduser().resolve()
if str(reference_root) not in sys.path:
    sys.path.insert(0, str(reference_root))

waterhose_launch = prepare_waterhose_launch(
    args_cli,
    task_name=args_cli.task,
    parser=parser,
    default_standalone_spacemouse=bool(args_cli.teleop),
    require_standalone_spacemouse=bool(args_cli.teleop),
    standalone_spacemouse_error=(
        "Waterhose robot demo teleoperation in the Newton viewer requires --teleop_device spacemouse."
    ),
)

app_launcher = None
simulation_app = None
if not waterhose_launch.uses_kitless_waterhose:
    app_launcher = AppLauncher(vars(args_cli))
    simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils import launch_simulation, parse_env_cfg


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise TypeError(f"Expected ManagerBasedRLEnvCfg, got {type(env_cfg).__name__}.")

    env_cfg.scene.num_envs = 1
    env_cfg.reference_newton_root = args_cli.reference_newton_root
    env_cfg.reference_viewer = args_cli.newton_viewer
    env_cfg.reference_primary_view = args_cli.primary_view
    env_cfg.reference_headless = bool(args_cli.reference_headless)
    env_cfg.max_demo_steps = int(args_cli.max_steps)

    launch_context = None
    if simulation_app is None:
        launch_context = launch_simulation(env_cfg, args_cli)
        launch_context.__enter__()

    env = None
    step = 0
    start = time.perf_counter()
    teleop_interface = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        if args_cli.teleop:
            teleop_interface = create_waterhose_spacemouse_device(
                args_cli,
                args_cli.sensitivity,
                simple_by_default=True,
            )
            teleop_interface.reset()
            print(f"[INFO] Teleop device: {teleop_interface}", flush=True)

        def running() -> bool:
            if simulation_app is not None:
                return simulation_app.is_running()
            viewer = getattr(env, "reference_viewer", None)
            if viewer is not None and hasattr(viewer, "is_running"):
                return viewer.is_running()
            return True

        while running() and step < args_cli.max_steps:
            if teleop_interface is not None:
                command = teleop_interface.advance()
                env.apply_teleop_command(command)
                if args_cli.debug_teleop and step % 15 == 0:
                    command_cpu = command.detach().cpu()
                    print(
                        "teleop command:",
                        " ".join(f"{float(value):+.3f}" for value in command_cpu.tolist()),
                        flush=True,
                    )
            obs, rew, terminated, truncated, extras = env.step(actions)
            del obs, rew, extras
            if bool(torch.any(terminated | truncated).item()):
                break
            step += 1
    finally:
        elapsed = time.perf_counter() - start
        if env is not None:
            sim_time = step * float(env.step_dt)
            print(
                f"[PROFILE] steps={step} sim_time={sim_time:.3f}s wall_time={elapsed:.3f}s "
                f"rtf={sim_time / max(elapsed, 1e-12):.3f} steps_per_s={step / max(elapsed, 1e-12):.1f}",
                flush=True,
            )
            env.close()
        if launch_context is not None:
            launch_context.__exit__(None, None, None)


if __name__ == "__main__":
    main()
    if simulation_app is not None:
        simulation_app.close()

