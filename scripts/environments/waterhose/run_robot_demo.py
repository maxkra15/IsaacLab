# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the waterhose robot demo IsaacLab task."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")
# The demo renders authored USD assets from the IsaacLab scene config. The
# deprecated Newton color back-fill is only useful for generated debug geometry
# and adds noisy warnings during startup.
os.environ.setdefault("ISAACLAB_REPLACE_NEWTON_SHAPE_COLORS", "0")

from isaaclab.app import AppLauncher

DEFAULT_TASK = "Isaac-Waterhose-Coupled-v0"
DEFAULT_MAX_STEPS = 4500
SCENE_CONFIG_COUPLED_TASKS = {
    "Isaac-Waterhose-Coupled-v0",
    "Isaac-Waterhose-Admm-v0",
}
SCENE_CONFIG_SCRIPTED_TASKS = {
    "Isaac-Waterhose-Coupled-v0",
    "Isaac-Waterhose-Admm-v0",
}


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
    if os.getenv("WATERHOSE_USE_LOCAL_ISAACSIM_EXTS", "").lower() not in {"1", "true", "yes", "on"}:
        return

    kit_args = _local_isaacsim_kit_args()
    if kit_args:
        args_cli.kit_args = kit_args
        _debug_runner(f"kit_args={kit_args}")


def _ensure_display_for_visible_visualizer(selected_visualizers: set[str]) -> None:
    """Use a local X display for visible visualizers when the shell did not export DISPLAY."""

    if not ({"kit", "newton"} & selected_visualizers):
        return
    if os.environ.get("DISPLAY"):
        return
    x11_root = Path("/tmp/.X11-unix")
    for display in ("1", "0"):
        if (x11_root / f"X{display}").exists():
            os.environ["DISPLAY"] = f":{display}"
            _debug_runner(f"DISPLAY was unset; using DISPLAY=:{display}")
            return


def _task_id() -> str:
    return args_cli.task.split(":")[-1]


def _uses_scene_config_coupled_task() -> bool:
    """Whether the selected task needs SimulationApp before config import."""

    return _task_id() in SCENE_CONFIG_COUPLED_TASKS


def _requested_visualizer_types(args_cli: argparse.Namespace) -> set[str]:
    """Return visualizer names parsed by IsaacLab's AppLauncher CLI support."""

    visualizer = getattr(args_cli, "visualizer", None)
    if not visualizer:
        return set()
    if isinstance(visualizer, str):
        visualizer = [token.strip() for token in visualizer.split(",")]
    return {str(item).strip().lower() for item in visualizer if str(item).strip()}


def _startup_visualizer_types(args_cli: argparse.Namespace) -> set[str]:
    """Visualizers that may need a display before AppLauncher normalizes settings."""

    if bool(getattr(args_cli, "headless", False)) and bool(getattr(args_cli, "headless_explicit", False)):
        return set()
    if bool(getattr(args_cli, "visualizer_explicit", False)):
        return _requested_visualizer_types(args_cli)
    if _uses_scene_config_coupled_task():
        return {"kit", "newton"}
    return _requested_visualizer_types(args_cli)


def _set_scene_config_visualizer_intent(args_cli: argparse.Namespace) -> None:
    """Mirror launch_simulation's config-derived visualizer intent for early AppLauncher startup."""

    if _uses_scene_config_coupled_task():
        args_cli.visualizer_intent = {"has_any_visualizers": True, "has_kit_visualizer": True}


parser = argparse.ArgumentParser(description="Run the scripted waterhose robot demo through IsaacLab.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum manager steps to run.")
parser.add_argument(
    "--settle_time", type=float, default=2.0, help="Initial settle time for scene-config scripted IK demos."
)
parser.add_argument(
    "--debug_script", action="store_true", help="Print phase transitions for scene-config scripted IK demos."
)
parser.add_argument(
    "--asset_root",
    type=str,
    default=os.getenv("WATERHOSE_ASSETS_DIR", ""),
    help="Optional WaterhoseDemo asset root. The packaged task assets are used when omitted.",
)
parser.add_argument("--profile", action="store_true", help="Print rollout timing after the run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
_install_faulthandler()

if args_cli.num_envs < 1:
    parser.error("--num_envs must be at least 1.")

startup_visualizers = _startup_visualizer_types(args_cli)
_prefer_cuda_for_waterhose_xr(args_cli)
if args_cli.asset_root:
    os.environ["WATERHOSE_ASSETS_DIR"] = args_cli.asset_root
if "kit" in startup_visualizers and os.getenv("WATERHOSE_KIT_MULTI_GPU", "").lower() not in {"1", "true", "yes", "on"}:
    args_cli.multi_gpu = False
_ensure_display_for_visible_visualizer(startup_visualizers)
_ensure_local_isaacsim_kit_args(args_cli, startup_visualizers)


def _configure_env_cfg(env_cfg) -> None:
    from isaaclab_newton.physics import NewtonCfg  # noqa: PLC0415

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    if _task_id() in SCENE_CONFIG_SCRIPTED_TASKS:
        # The scripted demo plays the full grasp -> insert -> release -> pull-out arc
        # and stops itself once the state machine reaches DONE. Env-level resets must
        # not fire mid-demo: the success termination triggers right at seating and the
        # auto-reset teleports the robot home (looks like the arm "flips out"), and the
        # 30 s time-out lands just before PULL_OUT finishes.
        terminations = getattr(env_cfg, "terminations", None)
        if terminations is not None and hasattr(terminations, "success"):
            terminations.success = None
        if terminations is not None and hasattr(terminations, "time_out"):
            terminations.time_out = None
    physics_cfg = env_cfg.sim.physics
    if not isinstance(physics_cfg, NewtonCfg):
        raise TypeError(f"Expected a Newton-backed task config, got {type(physics_cfg).__name__}.")


def _parse_configured_env_cfg():
    from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: PLC0415

    import isaaclab_tasks  # noqa: F401, PLC0415
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


def _run_env(env_cfg) -> None:
    import gymnasium as gym  # noqa: PLC0415
    import torch  # noqa: PLC0415

    env = None
    step = 0
    start = time.perf_counter()
    rollout_start = start
    scripted_state = None
    done_linger_steps = None
    try:
        _debug_runner("gym_make:start")
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        _debug_runner("gym_make:done")
        _debug_runner("reset:start")
        env.reset()
        _debug_runner("reset:done")
        actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        task_name = args_cli.task.split(":")[-1]
        if task_name in SCENE_CONFIG_SCRIPTED_TASKS:
            from isaaclab_tasks.contrib.waterhose.scripted_state_machine import (  # noqa: PLC0415
                create_scripted_policy,
            )

            scripted_state = create_scripted_policy(
                env,
                settle_time=args_cli.settle_time,
                debug=args_cli.debug_script,
            )

        rollout_start = time.perf_counter()
        _debug_runner(f"loop:start running={_simulation_is_running(env)} max_steps={args_cli.max_steps}")
        while _simulation_is_running(env) and step < args_cli.max_steps:
            if scripted_state is not None:
                actions = scripted_state.compute(env)
            else:
                actions.zero_()
            obs, rew, terminated, truncated, extras = env.step(actions)
            del obs, rew, extras
            # The termination check forces a per-step CPU<->GPU sync; skip it while
            # profiling so the rollout reflects raw stepping throughput (the run is
            # already bounded by --max_steps).
            if not args_cli.profile and bool(torch.any(terminated | truncated).item()):
                break
            # Scripted demo: once every env reaches DONE, hold the final pose briefly,
            # then stop the simulation and close.
            if scripted_state is not None and not args_cli.profile and done_linger_steps is None:
                if bool((scripted_state.phase == scripted_state.DONE).all().item()):
                    done_linger_steps = max(1, int(round(1.0 / env.step_dt)))
                    _debug_runner(f"scripted demo DONE at step={step}; closing after {done_linger_steps} linger steps")
            if done_linger_steps is not None:
                done_linger_steps -= 1
                if done_linger_steps <= 0:
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
    if _uses_scene_config_coupled_task():
        # The scene-config coupled tasks import Newton/USD builders while resolving
        # the task. SimulationApp must own the USD Python bindings first; otherwise
        # Kit extensions can see preloaded pxr modules from the venv and fail at startup.
        _set_scene_config_visualizer_intent(args_cli)
        app_launcher = AppLauncher(args_cli)
        try:
            env_cfg = _parse_configured_env_cfg()
            if hasattr(app_launcher, "device"):
                env_cfg.sim.device = app_launcher.device
            _run_env(env_cfg)
        finally:
            app_launcher.app.close()
        return

    env_cfg = _parse_configured_env_cfg()
    from isaaclab.app import launch_simulation  # noqa: PLC0415

    with launch_simulation(env_cfg, args_cli):
        _run_env(env_cfg)


if __name__ == "__main__":
    main()
