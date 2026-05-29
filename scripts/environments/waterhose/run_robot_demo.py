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

from isaaclab.app import AppLauncher


DEFAULT_TASK = "Isaac-Waterhose-Robot-Demo-v0"
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


def _debug_runner(message: str) -> None:
    if os.getenv("WATERHOSE_DEBUG_RUNNER", "").lower() in {"1", "true", "yes", "on"}:
        print(f"[waterhose-runner] {message}", flush=True)


def validate_visualizers(args_cli: argparse.Namespace, parser: argparse.ArgumentParser) -> set[str]:
    """Resolve display selection using IsaacLab visualizer names."""
    if getattr(args_cli, "visualizer", None) is None:
        args_cli.visualizer = ["none"] if bool(getattr(args_cli, "headless", False)) else ["kit"]
        setattr(args_cli, "visualizer_explicit", True)

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
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--max_steps", type=int, default=2000, help="Maximum manager steps to run.")
parser.add_argument("--max_demo_steps", type=int, default=0, help="Optional env-level termination bound; 0 disables it.")
parser.add_argument(
    "--newton_root",
    type=str,
    default=os.getenv("NEWTON_ROOT", ""),
    help="Optional Newton repo root to add to PYTHONPATH when Newton is not installed in the environment.",
)
parser.add_argument("--asset_root", type=str, default=DEFAULT_ASSET_ROOT, help="WaterhoseDemo asset root.")
parser.add_argument("--teleop_device", type=str, default=None, help="Teleop device. Use spacemouse.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Teleop sensitivity scale.")
parser.add_argument("--profile", action="store_true", help="Print rollout timing after the run.")
parser.add_argument("--debug_teleop", action="store_true", help="Print periodic teleop commands.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--spacemouse_pos_sensitivity", type=float, default=None)
parser.add_argument("--spacemouse_rot_sensitivity", type=float, default=None)
parser.add_argument("--spacemouse_simple_x_sign", type=float, choices=(-1.0, 1.0), default=-1.0)
parser.add_argument("--spacemouse_simple_y_sign", type=float, choices=(-1.0, 1.0), default=-1.0)
parser.add_argument("--spacemouse_simple_z_sign", type=float, choices=(-1.0, 1.0), default=1.0)
parser.add_argument("--spacemouse_simple_yaw_sign", type=float, choices=(-1.0, 1.0), default=-1.0)
parser.add_argument("--spacemouse_simple_deadzone", type=float, default=1.0e-3)
parser.add_argument(
    "--spacemouse_simple_yaw_translation_lock",
    action=argparse.BooleanOptionalAction,
    default=False,
)
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
if args_cli.num_envs < 1:
    parser.error("--num_envs must be at least 1.")
if args_cli.num_envs != 1 and "admm" in args_cli.task.lower():
    parser.error("The experimental ADMM coupled task is single-env only.")
if args_cli.mode == "teleop" and args_cli.teleop_device not in (None, "spacemouse"):
    parser.error("This demo currently supports --teleop_device spacemouse.")
if args_cli.mode == "teleop" and args_cli.teleop_device is None:
    args_cli.teleop_device = "spacemouse"

selected_visualizers = validate_visualizers(args_cli, parser)


def _configure_env_cfg(env_cfg) -> None:
    from isaaclab_newton.physics import NewtonCfg  # noqa: PLC0415
    from isaaclab_tasks.manager_based.manipulation.waterhose_robot_demo.coupled_manager import (  # noqa: PLC0415
        WaterhoseAdmmSolverCfg,
        WaterhoseOneWaySolverCfg,
    )
    from isaaclab_tasks.manager_based.manipulation.waterhose_robot_demo.manager import (  # noqa: PLC0415
        WaterhoseNewtonSolverCfg,
    )

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    if hasattr(env_cfg, "max_demo_steps"):
        env_cfg.max_demo_steps = int(args_cli.max_demo_steps)
    physics_cfg = env_cfg.sim.physics
    if not isinstance(physics_cfg, NewtonCfg) or not isinstance(
        physics_cfg.solver_cfg, (WaterhoseNewtonSolverCfg, WaterhoseAdmmSolverCfg, WaterhoseOneWaySolverCfg)
    ):
        raise TypeError(
            "Expected NewtonCfg(solver_cfg=WaterhoseNewtonSolverCfg|WaterhoseAdmmSolverCfg|WaterhoseOneWaySolverCfg), "
            f"got {type(physics_cfg).__name__}."
        )
    solver_cfg = physics_cfg.solver_cfg
    solver_cfg.num_envs = int(env_cfg.scene.num_envs)
    solver_cfg.env_spacing = float(env_cfg.scene.env_spacing)
    if hasattr(solver_cfg, "newton_root"):
        solver_cfg.newton_root = args_cli.newton_root
    solver_cfg.asset_root = args_cli.asset_root
    if hasattr(solver_cfg, "max_demo_steps"):
        solver_cfg.max_demo_steps = int(args_cli.max_demo_steps)


def _configure_visualizers(env_cfg) -> None:
    visualizer_cfgs = []
    if "kit" in selected_visualizers:
        from isaaclab_visualizers.kit import KitVisualizerCfg  # noqa: PLC0415

        visualizer_cfgs.append(
            KitVisualizerCfg(
                eye=(-2.55, -7.1, 2.3),
                lookat=(0.55, -0.42, 0.9),
            )
        )
    if "newton" in selected_visualizers:
        from isaaclab_visualizers.newton import NewtonVisualizerCfg  # noqa: PLC0415

        visualizer_cfgs.append(
            NewtonVisualizerCfg(
                eye=(-2.55, -7.1, 2.3),
                lookat=(0.55, -0.42, 0.9),
                show_static=True,
            )
        )
    env_cfg.sim.visualizer_cfgs = visualizer_cfgs or None


def _parse_configured_env_cfg():
    import isaaclab_tasks  # noqa: F401, PLC0415
    from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: PLC0415
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: PLC0415

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=int(args_cli.num_envs))

    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise TypeError(f"Expected ManagerBasedRLEnvCfg, got {type(env_cfg).__name__}.")
    _configure_env_cfg(env_cfg)
    return env_cfg


def _simulation_is_running(env) -> bool:
    if env.sim.visualizers:
        return any(visualizer.is_running() and not visualizer.is_closed for visualizer in env.sim.visualizers)
    return True


def _set_task_teleop_enabled(env_cfg, enabled: bool) -> None:
    """Enable teleop on the selected waterhose Newton manager."""

    solver_cfg = env_cfg.sim.physics.solver_cfg
    from isaaclab_tasks.manager_based.manipulation.waterhose_robot_demo.coupled_manager import (  # noqa: PLC0415
        NewtonWaterhoseCoupledManager,
        WaterhoseAdmmSolverCfg,
        WaterhoseOneWaySolverCfg,
    )
    from isaaclab_tasks.manager_based.manipulation.waterhose_robot_demo.manager import (  # noqa: PLC0415
        WaterhoseNewtonSolverCfg,
        NewtonWaterhoseManager,
    )

    if isinstance(solver_cfg, (WaterhoseAdmmSolverCfg, WaterhoseOneWaySolverCfg)):
        NewtonWaterhoseCoupledManager.set_teleop_enabled(enabled)
    elif isinstance(solver_cfg, WaterhoseNewtonSolverCfg):
        NewtonWaterhoseManager.set_teleop_enabled(enabled)


def _run_env(env_cfg) -> None:
    import gymnasium as gym  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from isaaclab_tasks.manager_based.manipulation.waterhose_robot_demo.teleop import (  # noqa: PLC0415
        create_waterhose_spacemouse_device,
    )

    env = None
    step = 0
    start = time.perf_counter()
    rollout_start = start
    teleop_interface = None
    try:
        _debug_runner("gym_make:start")
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        _debug_runner("gym_make:done")
        _debug_runner("reset:start")
        env.reset()
        _debug_runner("reset:done")
        actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        if args_cli.mode == "teleop":
            teleop_interface = create_waterhose_spacemouse_device(args_cli, args_cli.sensitivity)
            teleop_interface.reset()
            _set_task_teleop_enabled(env_cfg, True)
            print(f"[INFO] Teleop device: {teleop_interface}", flush=True)

        rollout_start = time.perf_counter()
        _debug_runner(f"loop:start running={_simulation_is_running(env)} max_steps={args_cli.max_steps}")
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
        _debug_runner(f"loop:done steps={step} running={_simulation_is_running(env)}")
    except BaseException as exc:
        _debug_runner(f"exception type={type(exc).__name__} value={exc!r}")
        raise
    finally:
        elapsed = time.perf_counter() - start
        _debug_runner(f"finally env={env is not None} profile={bool(args_cli.profile)} steps={step}")
        if env is not None and args_cli.profile:
            setup_time = max(0.0, rollout_start - start)
            rollout_time = max(time.perf_counter() - rollout_start, 1e-12)
            sim_time = step * float(env.step_dt)
            from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415

            graph_state = "captured" if getattr(NewtonManager, "_graph", None) is not None else "eager"
            print(
                f"[PROFILE] steps={step} sim_time={sim_time:.3f}s setup_time={setup_time:.3f}s "
                f"rollout_time={rollout_time:.3f}s wall_time={elapsed:.3f}s "
                f"rtf={sim_time / rollout_time:.3f} steps_per_s={step / rollout_time:.1f} "
                f"cuda_graph={graph_state}",
                flush=True,
            )
        if env is not None:
            env.close()


def main() -> None:
    if "kit" in selected_visualizers:
        app_launcher = AppLauncher(args_cli)
        try:
            env_cfg = _parse_configured_env_cfg()
            _configure_visualizers(env_cfg)
            if hasattr(app_launcher, "device"):
                env_cfg.sim.device = app_launcher.device
            _run_env(env_cfg)
        finally:
            app_launcher.app.close()
        return

    env_cfg = _parse_configured_env_cfg()
    _configure_visualizers(env_cfg)
    from isaaclab.app import launch_simulation  # noqa: PLC0415

    with launch_simulation(env_cfg, args_cli):
        _run_env(env_cfg)


if __name__ == "__main__":
    main()
