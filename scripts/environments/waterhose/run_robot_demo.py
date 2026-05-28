# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the waterhose robot demo IsaacLab task."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_newton.physics import NewtonCfg
from isaaclab_tasks.manager_based.manipulation.waterhose_robot_demo.manager import (
    NewtonWaterhoseManager,
    WaterhoseNewtonSolverCfg,
)
from isaaclab_tasks.manager_based.manipulation.waterhose_robot_demo.teleop import (
    add_waterhose_spacemouse_args,
    create_waterhose_spacemouse_device,
)
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, parse_env_cfg


DEFAULT_TASK = "Isaac-Waterhose-Robot-Demo-Play-v0"
DEFAULT_ASSET_ROOT = str(
    Path(__file__).resolve().parents[3] / "source" / "isaaclab_assets" / "data" / "WaterhoseDemo"
)
SUPPORTED_VISUALIZERS = {"kit", "newton", "none"}


class ExplicitVisualizerAction(argparse.Action):
    """Track explicit use of the local --vis alias."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f"{self.dest}_explicit", True)


def parse_visualizer_csv(value: str) -> list[str]:
    """Parse IsaacLab-style comma-delimited visualizer selections."""
    values = [token.strip().lower() for token in value.split(",")]
    if not values or any(not token for token in values):
        raise argparse.ArgumentTypeError("Use a comma-separated visualizer list, for example --vis kit,newton.")
    if any(" " in token for token in values):
        raise argparse.ArgumentTypeError("Visualizer names must be comma-separated without spaces.")
    invalid = [token for token in values if token not in SUPPORTED_VISUALIZERS]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported visualizer value(s): {', '.join(invalid)}.")
    return values


def visualizer_types(args_cli: argparse.Namespace) -> set[str]:
    """Return normalized visualizer names from parsed launcher args."""
    visualizer = getattr(args_cli, "visualizer", None) or []
    if isinstance(visualizer, str):
        visualizer = parse_visualizer_csv(visualizer)
    return {str(item).strip().lower() for item in visualizer if str(item).strip()}


def validate_visualizers(args_cli: argparse.Namespace, parser: argparse.ArgumentParser) -> set[str]:
    """Resolve display selection using IsaacLab visualizer names."""
    if getattr(args_cli, "visualizer", None) is None:
        args_cli.visualizer = ["none"] if bool(getattr(args_cli, "headless", False)) else ["newton"]

    selected = visualizer_types(args_cli)
    if "none" in selected and len(selected) > 1:
        parser.error("--vis none cannot be combined with other visualizers.")

    unsupported = selected - SUPPORTED_VISUALIZERS
    if unsupported:
        parser.error(
            "This demo supports --vis newton, --vis kit, --vis kit,newton, or --vis none. "
            f"Unsupported value(s): {', '.join(sorted(unsupported))}."
        )
    if bool(getattr(args_cli, "headless", False)) and "newton" in selected:
        parser.error("The Newton visualizer needs a display. Use --vis none for headless runs.")

    if bool(getattr(args_cli, "xr", False)):
        if "none" in selected:
            parser.error("--xr requires Kit; use --vis kit.")
        if "kit" not in selected:
            args_cli.visualizer = ["kit", *[item for item in args_cli.visualizer if item != "kit"]]
            selected = visualizer_types(args_cli)

    args_cli.visualizer = [] if "none" in selected else list(selected)
    setattr(args_cli, "visualizer_disable_all", "none" in selected)
    setattr(args_cli, "visualizer_explicit", True)
    return selected


parser = argparse.ArgumentParser(description="Run the waterhose robot demo through IsaacLab.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Task name.")
parser.add_argument("--mode", choices=("scripted", "teleop"), default="scripted", help="Control mode.")
parser.add_argument("--teleop", action="store_true", help="Alias for --mode teleop.")
parser.add_argument("--num_envs", type=int, default=1, help="Only 1 is supported.")
parser.add_argument("--max_steps", type=int, default=2000, help="Maximum manager steps to run.")
parser.add_argument("--max_demo_steps", type=int, default=0, help="Optional env-level termination bound; 0 disables it.")
parser.add_argument("--newton_root", type=str, default="/home/maximiliank/Work/newton", help="Newton repo root.")
parser.add_argument("--asset_root", type=str, default=DEFAULT_ASSET_ROOT, help="WaterhoseDemo asset root.")
parser.add_argument("--teleop_device", type=str, default=None, help="Teleop device. Use spacemouse.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Teleop sensitivity scale.")
parser.add_argument("--debug_teleop", action="store_true", help="Print periodic teleop commands.")
add_waterhose_spacemouse_args(parser)
add_launcher_args(parser)
parser.add_argument(
    "--vis",
    dest="visualizer",
    type=parse_visualizer_csv,
    action=ExplicitVisualizerAction,
    help="Alias for --visualizer. Supported here: newton, kit, kit,newton, none.",
)
args_cli = parser.parse_args()

if args_cli.teleop:
    args_cli.mode = "teleop"
if args_cli.num_envs != 1:
    parser.error("The waterhose robot demo currently supports exactly one environment.")
if args_cli.mode == "teleop" and args_cli.teleop_device not in (None, "spacemouse"):
    parser.error("This demo currently supports --teleop_device spacemouse.")
if args_cli.mode == "teleop" and args_cli.teleop_device is None:
    args_cli.teleop_device = "spacemouse"

selected_visualizers = validate_visualizers(args_cli, parser)


def _startup_report() -> None:
    sim = NewtonWaterhoseManager.get_runtime()
    if sim is None:
        print("[DEMO] startup: runtime is not initialized", flush=True)
        return
    visualization_model = NewtonWaterhoseManager.get_visualization_model()
    display_bodies = 0 if visualization_model is None else int(visualization_model.body_count)
    display_shapes = 0 if visualization_model is None else int(visualization_model.shape_count)
    print(
        "[DEMO] startup: "
        f"robot_bodies={sim.mujoco_model.body_count} robot_shapes={sim.mujoco_model.shape_count} "
        f"vbd_bodies={sim.vbd_model.body_count} vbd_shapes={sim.vbd_model.shape_count} "
        f"display_bodies={display_bodies} display_shapes={display_shapes} "
        f"proxies={len(getattr(sim, 'proxy_body_ids', []))} "
        f"proxy_shapes={len(getattr(sim, '_proxy_shape_ids', []))}",
        flush=True,
    )


def _configure_env_cfg(env_cfg: ManagerBasedRLEnvCfg) -> None:
    env_cfg.scene.num_envs = 1
    env_cfg.max_demo_steps = int(args_cli.max_demo_steps)
    physics_cfg = env_cfg.sim.physics
    if not isinstance(physics_cfg, NewtonCfg) or not isinstance(physics_cfg.solver_cfg, WaterhoseNewtonSolverCfg):
        raise TypeError(
            "Expected NewtonCfg(solver_cfg=WaterhoseNewtonSolverCfg), "
            f"got {type(physics_cfg).__name__}."
        )
    solver_cfg = physics_cfg.solver_cfg
    solver_cfg.newton_root = args_cli.newton_root
    solver_cfg.asset_root = args_cli.asset_root
    solver_cfg.max_demo_steps = int(args_cli.max_demo_steps)
    if "newton" in selected_visualizers:
        from isaaclab_visualizers.newton import NewtonVisualizerCfg  # noqa: PLC0415

        newton_viz_cfg = NewtonVisualizerCfg(
            eye=(-2.55, -7.1, 2.3),
            lookat=(0.55, -0.42, 0.9),
            show_static=True,
        )
        existing_cfgs = env_cfg.sim.visualizer_cfgs or []
        if not isinstance(existing_cfgs, list):
            existing_cfgs = [existing_cfgs]
        env_cfg.sim.visualizer_cfgs = [cfg for cfg in existing_cfgs if cfg.visualizer_type != "newton"]
        env_cfg.sim.visualizer_cfgs.append(newton_viz_cfg)


def _simulation_is_running(env) -> bool:
    if env.sim.visualizers:
        return any(visualizer.is_running() and not visualizer.is_closed for visualizer in env.sim.visualizers)
    return True


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise TypeError(f"Expected ManagerBasedRLEnvCfg, got {type(env_cfg).__name__}.")
    _configure_env_cfg(env_cfg)

    env = None
    step = 0
    start = time.perf_counter()
    teleop_interface = None
    with launch_simulation(env_cfg, args_cli):
        try:
            env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
            env.reset()
            _startup_report()
            actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
            if args_cli.mode == "teleop":
                teleop_interface = create_waterhose_spacemouse_device(args_cli, args_cli.sensitivity)
                teleop_interface.reset()
                NewtonWaterhoseManager.set_teleop_enabled(True)
                print(f"[INFO] Teleop device: {teleop_interface}", flush=True)

            while _simulation_is_running(env) and step < args_cli.max_steps:
                if teleop_interface is not None:
                    command = teleop_interface.advance()
                    actions.zero_()
                    command = command.to(device=env.device, dtype=actions.dtype).reshape(-1)
                    width = min(actions.shape[1], command.numel())
                    actions[:, :width] = command[:width].reshape(1, width).repeat(env.num_envs, 1)
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


if __name__ == "__main__":
    main()
