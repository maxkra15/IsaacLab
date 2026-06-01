# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the waterhose robot demo IsaacLab task."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")
# The demo renders authored USD assets from the IsaacLab scene config. The
# deprecated Newton color back-fill is only useful for generated debug geometry
# and adds noisy warnings during startup.
os.environ.setdefault("ISAACLAB_REPLACE_NEWTON_SHAPE_COLORS", "0")

from isaaclab.app import AppLauncher


DEFAULT_TASK = "Isaac-Waterhose-Robot-Demo-v0"
DEFAULT_ASSET_ROOT = str(
    Path(__file__).resolve().parents[3] / "source" / "isaaclab_assets" / "data" / "WaterhoseDemo"
)
SUPPORTED_VISUALIZERS = {"kit", "newton", "none"}
SCENE_CONFIG_COUPLED_TASKS = {
    "Isaac-Waterhose-v0",
    "Isaac-Waterhose-Proxy-v0",
    "Isaac-Waterhose-RBY1-IK-Abs-v0",
    "Isaac-Waterhose-Robot-Demo-Coupled-v0",
    "Isaac-Waterhose-Robot-Demo-Proxy-Coupled-v0",
}


class ExplicitVisualizerAction(argparse.Action):
    """Track explicit use of the local --vis alias."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f"{self.dest}_explicit", True)


def parse_visualizer_csv(value: str) -> list[str]:
    """Parse IsaacLab-style comma-delimited visualizer selections."""
    values = [token.strip().lower() for token in value.split(",")]
    if not values or any(not token for token in values):
        raise argparse.ArgumentTypeError("Use a visualizer name such as --vis kit, --vis newton, or --vis none.")
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
        print(f"[waterhose-runner] {message}", file=sys.__stderr__, flush=True)


def _install_faulthandler() -> None:
    if os.getenv("WATERHOSE_FAULTHANDLER", "").lower() not in {"1", "true", "yes", "on"}:
        return
    import faulthandler

    faulthandler.enable(file=sys.__stderr__, all_threads=True)
    faulthandler.register(signal.SIGUSR1, file=sys.__stderr__, all_threads=True)


def _prefer_cuda_for_waterhose_xr(args_cli: argparse.Namespace) -> None:
    """Keep the Newton waterhose runtime on CUDA for XR unless the user chose another device."""

    if not bool(getattr(args_cli, "xr", False)):
        return
    if bool(getattr(args_cli, "device_explicit", False)):
        return
    args_cli.device = "cuda:0"
    args_cli.device_explicit = True


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
            "This demo supports --vis newton, --vis kit, or --vis none. "
            f"Unsupported value(s): {', '.join(sorted(unsupported))}."
        )
    if {"kit", "newton"}.issubset(selected):
        parser.error(
            "The waterhose demo runner supports one visible backend per process. "
            "Use --vis kit for the Isaac Lab/Kit view, --vis newton for the standalone Newton view, "
            "or --vis none for headless/profile runs."
        )
    if bool(getattr(args_cli, "headless", False)) and selected != {"none"}:
        parser.error(
            "Visible visualizers cannot be combined with --headless. "
            "Remove --headless for --vis kit/newton, or use --vis none for headless runs."
        )

    if bool(getattr(args_cli, "xr", False)):
        if "none" in selected:
            parser.error("--xr requires Kit; use --vis kit.")
        if "kit" not in selected:
            args_cli.visualizer = ["kit", *[item for item in args_cli.visualizer if item != "kit"]]
            selected = visualizer_types(args_cli)

    if "none" in selected:
        args_cli.headless = True
        args_cli.visualizer = []
    else:
        args_cli.visualizer = list(selected)
    setattr(args_cli, "visualizer_disable_all", "none" in selected)
    setattr(args_cli, "visualizer_explicit", True)
    return selected


def _local_isaacsim_kit_args() -> str:
    """Return Kit extension-folder args for a source-built Isaac Sim checkout."""
    isaacsim_root = Path(__file__).resolve().parents[3] / "_isaac_sim"
    if not isaacsim_root.exists():
        return ""

    kit_args = []
    for extension_dir in ("exts", "extsDeprecated", "extscache"):
        extension_path = isaacsim_root / extension_dir
        if extension_path.is_dir():
            kit_args.append(f"--ext-folder={extension_path.resolve()}")
    return " ".join(kit_args)


def _ensure_local_isaacsim_kit_args(args_cli: argparse.Namespace, selected_visualizers: set[str]) -> None:
    """Make direct runner launches see local Isaac Sim extensions."""
    if "kit" not in selected_visualizers or getattr(args_cli, "kit_args", ""):
        return

    kit_args = _local_isaacsim_kit_args()
    if kit_args:
        args_cli.kit_args = kit_args
        _debug_runner(f"kit_args={kit_args}")


def _prewarm_static_scene_cache(asset_root: str) -> None:
    """Generate the Newton static-scene collision cache before Kit starts."""

    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "from isaaclab_tasks.manager_based.manipulation.waterhose_robot_demo.coupled_builder "
            "import ensure_static_scene_cache; "
            "ensure_static_scene_cache(sys.argv[1])"
        ),
        asset_root,
    ]
    result = subprocess.run(command, env=os.environ.copy(), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to prepare the waterhose static-scene collision cache required for Kit visualization."
        )


def _uses_scene_config_coupled_task() -> bool:
    """Whether the selected task needs SimulationApp before config import."""

    return args_cli.task.split(":")[-1] in SCENE_CONFIG_COUPLED_TASKS


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
parser.add_argument(
    "--asset_root",
    type=str,
    default=os.getenv("WATERHOSE_ASSETS_DIR", ""),
    help="Optional WaterhoseDemo asset root. The packaged task assets are used when omitted.",
)
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
    help="Alias for --visualizer. Supported here: newton, kit, none.",
)
args_cli = parser.parse_args()
_install_faulthandler()

if args_cli.teleop:
    args_cli.mode = "teleop"
if args_cli.num_envs < 1:
    parser.error("--num_envs must be at least 1.")
if args_cli.mode == "teleop" and args_cli.teleop_device not in (None, "keyboard", "spacemouse"):
    parser.error("This demo currently supports --teleop_device keyboard or spacemouse.")
if args_cli.mode == "teleop" and args_cli.teleop_device is None:
    args_cli.teleop_device = "spacemouse"

selected_visualizers = validate_visualizers(args_cli, parser)
if args_cli.mode == "teleop" and args_cli.teleop_device == "keyboard" and "kit" not in selected_visualizers:
    parser.error("Keyboard teleop requires a Kit window. Use --vis kit.")
_prefer_cuda_for_waterhose_xr(args_cli)
if args_cli.asset_root:
    os.environ["WATERHOSE_ASSETS_DIR"] = args_cli.asset_root
_ensure_local_isaacsim_kit_args(args_cli, selected_visualizers)


def _configure_env_cfg(env_cfg) -> None:
    from isaaclab_newton.physics import NewtonCfg  # noqa: PLC0415

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    if hasattr(env_cfg, "max_demo_steps"):
        env_cfg.max_demo_steps = int(args_cli.max_demo_steps)
    physics_cfg = env_cfg.sim.physics
    if not isinstance(physics_cfg, NewtonCfg):
        raise TypeError(
            "Expected NewtonCfg(solver_cfg=WaterhoseOneWaySolverCfg|WaterhoseCoupledSolverCfg|"
            "WaterhoseProxyCoupledSolverCfg), "
            f"got {type(physics_cfg).__name__}."
        )

    solver_cfg = physics_cfg.solver_cfg
    try:
        from isaaclab_tasks.manager_based.manipulation.waterhose_robot_demo.coupled_manager import (  # noqa: PLC0415
            WaterhoseCoupledSolverCfg,
            WaterhoseOneWaySolverCfg,
            WaterhoseProxyCoupledSolverCfg,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The waterhose robot demo requires the Newton coupled-manager API in isaaclab_newton."
        ) from exc

    solver_types = (WaterhoseCoupledSolverCfg, WaterhoseOneWaySolverCfg, WaterhoseProxyCoupledSolverCfg)
    if not isinstance(solver_cfg, solver_types):
        return
    if isinstance(solver_cfg, WaterhoseOneWaySolverCfg) and int(env_cfg.scene.num_envs) != 1:
        raise ValueError(
            "Isaac-Waterhose-Robot-Demo-v0 is the stable task-local one-way demo and currently supports "
            "one environment when using upstream Newton PR 2848. Use "
            "Isaac-Waterhose-Robot-Demo-Proxy-Coupled-v0 or Isaac-Waterhose-Robot-Demo-Coupled-v0 for "
            "standard scene-config multi-env coupling."
        )

    solver_cfg.num_envs = int(env_cfg.scene.num_envs)
    solver_cfg.env_spacing = float(env_cfg.scene.env_spacing)
    if hasattr(solver_cfg, "newton_root"):
        solver_cfg.newton_root = args_cli.newton_root
    solver_cfg.asset_root = args_cli.asset_root or getattr(solver_cfg, "asset_root", DEFAULT_ASSET_ROOT)
    _configure_scene_visual_assets(env_cfg.scene, solver_cfg.asset_root)
    if hasattr(solver_cfg, "max_demo_steps"):
        solver_cfg.max_demo_steps = int(args_cli.max_demo_steps)


def _configure_scene_visual_assets(scene_cfg, asset_root: str) -> None:
    """Keep IsaacLab scene visuals in sync with the Newton builder asset root."""

    root = Path(asset_root).expanduser().resolve()
    usd_paths = {
        "robot_visual": root / "rby1df" / "rby1df.usda",
        "fridge_visual": root / "fridge" / "fridge.usda",
        "cable_visual": root / "Waterhose" / "Cable008" / "curve" / "cable_SRA_curve03.usda",
    }
    for attr_name, usd_path in usd_paths.items():
        asset_cfg = getattr(scene_cfg, attr_name, None)
        spawn_cfg = getattr(asset_cfg, "spawn", None)
        if spawn_cfg is not None and hasattr(spawn_cfg, "usd_path"):
            spawn_cfg.usd_path = str(usd_path)


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

    try:
        from isaaclab_tasks.manager_based.manipulation.waterhose_robot_demo.coupled_manager import (  # noqa: PLC0415
            NewtonWaterhoseCoupledManager,
            WaterhoseCoupledSolverCfg,
            WaterhoseOneWaySolverCfg,
            WaterhoseProxyCoupledSolverCfg,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Teleop for the waterhose robot demo requires the Newton coupled-manager API."
        ) from exc

    if isinstance(solver_cfg, (WaterhoseCoupledSolverCfg, WaterhoseOneWaySolverCfg, WaterhoseProxyCoupledSolverCfg)):
        NewtonWaterhoseCoupledManager.set_teleop_enabled(enabled)


def _run_env(env_cfg) -> None:
    import gymnasium as gym  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from isaaclab_tasks.manager_based.manipulation.waterhose_robot_demo.teleop import (  # noqa: PLC0415
        create_waterhose_keyboard_device,
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
            if args_cli.teleop_device == "keyboard":
                teleop_interface = create_waterhose_keyboard_device(args_cli, args_cli.sensitivity)
            else:
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
                file=sys.__stderr__,
                flush=True,
            )
        if env is not None:
            env.close()


def main() -> None:
    from isaaclab_tasks.utils import launch_simulation  # noqa: PLC0415

    if "kit" in selected_visualizers or _uses_scene_config_coupled_task():
        # The scene-config coupled tasks import Newton/USD builders while resolving
        # the task. SimulationApp must own the USD Python bindings first; otherwise
        # Kit extensions can see preloaded pxr modules from the venv and fail during
        # startup. The stable task-local one-way demo still needs its static-scene
        # cache prewarmed for Kit display.
        if "kit" in selected_visualizers and not _uses_scene_config_coupled_task():
            _prewarm_static_scene_cache(args_cli.asset_root or DEFAULT_ASSET_ROOT)
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
