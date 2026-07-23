# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the waterhose robot demo IsaacLab task."""

from __future__ import annotations

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

DEFAULT_TASK = "Isaac-Waterhose-Coupled-v0"
DEFAULT_MAX_STEPS = 4500
GRAPH_PHASE_POLL_INTERVAL = 30


class _WallClockRateLimiter:
    """Pace a rollout to its configured simulation step without accumulating drift."""

    def __init__(self, step_dt: float):
        if step_dt <= 0.0:
            raise ValueError(f"Expected a positive simulation step, got {step_dt}.")
        self._step_dt = step_dt
        self._next_step_time = time.perf_counter()

    def sleep(self) -> None:
        """Wait until one simulation step has elapsed in wall-clock time."""
        self._next_step_time += self._step_dt
        now = time.perf_counter()
        remaining = self._next_step_time - now
        if remaining > 0.0:
            time.sleep(remaining)
            return

        # Do not try to replay missed wall-clock deadlines after a debugger pause,
        # window move, or temporarily slow render. Resume pacing from the present.
        if remaining < -4.0 * self._step_dt:
            self._next_step_time = now


def _task_id() -> str:
    return args_cli.task.split(":")[-1]


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
    return {"kit", "newton"}


def _set_scene_config_visualizer_intent(args_cli: argparse.Namespace) -> None:
    """Mirror launch_simulation's config-derived visualizer intent for early AppLauncher startup."""

    args_cli.visualizer_intent = {"has_any_visualizers": True, "has_kit_visualizer": True}


def _coupled_task_needs_kit(args_cli: argparse.Namespace) -> bool:
    """Whether the scripted coupled task must boot Omniverse Kit.

    The task runs on ``NewtonCfg`` with a coupler solver, so it can run Kit-free under a
    kitless visualizer. Kit is required when:

    * the Kit visualizer is explicitly requested (``--visualizer kit``), or
    * livestreaming is enabled (it needs a Kit viewport to produce video), or
    * no visualizer is requested on the command line. The task config installs both a Kit and a
      Newton visualizer, so with no CLI override the Newton runtime resolves Kit rendering as active
      and sets up Fabric/usdrt sync (Kit-only). Booting Kit keeps that default working; pass an
      explicit kitless visualizer (``--visualizer newton``/``rerun``/``viser``/``none``) to skip Kit.

    With an explicit kitless visualizer, the run goes through :func:`~isaaclab.app.launch_simulation`,
    which skips Kit entirely for this Newton-backed, camera-free task.
    """

    livestream = getattr(args_cli, "livestream", -1)
    livestream_mode = (
        int(livestream) if livestream is not None and int(livestream) >= 0 else int(os.environ.get("LIVESTREAM", "0"))
    )
    if livestream_mode > 0:
        return True

    if bool(getattr(args_cli, "headless", False)) and bool(getattr(args_cli, "headless_explicit", False)):
        return False
    if bool(getattr(args_cli, "visualizer_explicit", False)) and getattr(args_cli, "visualizer", None) is None:
        return False
    if bool(getattr(args_cli, "visualizer_explicit", False)) and bool(
        getattr(args_cli, "visualizer_disable_all", False)
    ):
        return False

    requested = _requested_visualizer_types(args_cli)
    if not requested:
        return True
    return "kit" in requested


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
parser.add_argument("--profile", action="store_true", help="Print rollout timing after the run.")
parser.add_argument(
    "--realtime",
    action=argparse.BooleanOptionalAction,
    default=None,
    help=(
        "Pace the rollout to simulation time. This is enabled by default for visible Kit/Newton runs;"
        " use --no-realtime for an unpaced benchmark."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs < 1:
    parser.error("--num_envs must be at least 1.")
if _task_id() != DEFAULT_TASK:
    parser.error(
        f"this runner supports only {DEFAULT_TASK!r}; use teleop_se3_agent.py for the human-driven teleop task"
    )

startup_visualizers = _startup_visualizer_types(args_cli)
if args_cli.realtime is None:
    args_cli.realtime = bool({"kit", "newton"} & startup_visualizers)


def _configure_env_cfg(env_cfg) -> None:
    from isaaclab_newton.physics import NewtonCfg  # noqa: PLC0415

    env_cfg.scene.num_envs = int(args_cli.num_envs)
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
    scripted_completed = False
    control_graph_captured = False
    done_linger_steps = None
    rate_limiter = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()
        if args_cli.realtime:
            rate_limiter = _WallClockRateLimiter(float(env.step_dt))
            print(
                f"[INFO]: Playing the visible waterhose demo in real time at"
                f" {1.0 / float(env.step_dt):.1f} Hz (--no-realtime disables pacing).",
                flush=True,
            )
        from isaaclab_tasks.contrib.waterhose.scripted_state_machine import (  # noqa: PLC0415
            connector_retained_mask,
            create_scripted_policy,
        )

        scripted_state = create_scripted_policy(
            env,
            settle_time=args_cli.settle_time,
            debug=args_cli.debug_script,
        )
        control_graph_captured = bool(getattr(scripted_state, "is_control_graph_captured", False))

        if args_cli.profile and "cuda" in str(env.device):
            import warp as wp  # noqa: PLC0415

            wp.synchronize_device(str(env.device))

        rollout_start = time.perf_counter()
        while _simulation_is_running(env) and step < args_cli.max_steps:
            if control_graph_captured:
                scripted_state.step_environment(env)
                terminated = truncated = None
            else:
                actions = scripted_state.compute(env)
                obs, rew, terminated, truncated, extras = env.step(actions)
                del obs, rew, extras
            # The termination check forces a per-step CPU<->GPU sync; skip it while
            # profiling so the rollout reflects raw stepping throughput (the run is
            # already bounded by --max_steps).
            if not control_graph_captured and not args_cli.profile and bool(torch.any(terminated | truncated).item()):
                break
            # Scripted demo: once every env reaches DONE, hold the final pose briefly,
            # then stop the simulation and close.
            if not args_cli.profile and done_linger_steps is None:
                if control_graph_captured:
                    if step % GRAPH_PHASE_POLL_INTERVAL == 0:
                        phases = scripted_state.report_phase(step)
                        is_done = all(phase == scripted_state.DONE for phase in phases)
                    else:
                        is_done = False
                else:
                    is_done = bool((scripted_state.phase == scripted_state.DONE).all().item())
                if is_done:
                    if not bool(connector_retained_mask(env).all().item()):
                        raise RuntimeError("Scripted waterhose reached DONE without a retained connector.")
                    done_linger_steps = max(1, int(round(1.0 / env.step_dt)))
            if done_linger_steps is not None:
                if not bool(connector_retained_mask(env).all().item()):
                    raise RuntimeError("Waterhose connector was lost during the final retention check.")
                done_linger_steps -= 1
                if done_linger_steps <= 0:
                    scripted_completed = True
                    break
            step += 1
            if rate_limiter is not None:
                rate_limiter.sleep()
        if not args_cli.profile and step >= args_cli.max_steps and not scripted_completed:
            raise RuntimeError(
                f"Scripted waterhose did not complete a retained insertion within {args_cli.max_steps} steps."
            )
    finally:
        if env is not None and args_cli.profile and "cuda" in str(env.device):
            import warp as wp  # noqa: PLC0415

            wp.synchronize_device(str(env.device))
        elapsed = time.perf_counter() - start
        if env is not None and args_cli.profile:
            setup_time = max(0.0, rollout_start - start)
            rollout_time = max(time.perf_counter() - rollout_start, 1e-12)
            sim_time = step * float(env.step_dt)
            from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415

            graph_state = "captured" if getattr(NewtonManager, "_graph", None) is not None else "eager"
            control_graph_state = "captured" if control_graph_captured else "eager"
            phase_state = ""
            if control_graph_captured:
                phase = scripted_state.read_phases()[0]
                phase_state = f" phase={scripted_state.PHASE_NAMES[phase]}"
            print(
                f"[PROFILE] steps={step} sim_time={sim_time:.3f}s setup_time={setup_time:.3f}s "
                f"rollout_time={rollout_time:.3f}s wall_time={elapsed:.3f}s "
                f"rtf={sim_time / rollout_time:.3f} steps_per_s={step / rollout_time:.1f} "
                f"control_graph={control_graph_state} physics_graph={graph_state}{phase_state}",
                file=sys.__stderr__,
                flush=True,
            )
        if env is not None:
            env.close()


def main() -> None:
    if _coupled_task_needs_kit(args_cli):
        # Kit path: the scene-config coupled task imports Newton/USD builders while resolving
        # the task. When Kit is needed, SimulationApp must own the USD Python bindings first;
        # otherwise Kit extensions can see preloaded pxr modules from the venv and fail at startup.
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

    # Kitless path (e.g. ``--visualizer newton`` or ``none``). ``launch_simulation`` walks the config
    # and skips Kit entirely for this Newton-backed, camera-free task. The pxr-binding ordering
    # constraint above does not apply when Kit never boots.
    env_cfg = _parse_configured_env_cfg()
    from isaaclab.app import launch_simulation  # noqa: PLC0415

    with launch_simulation(env_cfg, args_cli):
        _run_env(env_cfg)


if __name__ == "__main__":
    main()
