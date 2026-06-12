# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SpaceMouse teleoperation for the Franka scoop MPM env.

Maps the 6-DoF SpaceMouse to the env's 5-DoF action: lateral/vertical motion drives
the scoop bowl target in env/world Cartesian (x, y, z); one twist axis tilts the bowl
(negative = opening up to hold, positive = pour) and the yaw twist aims the pour
direction about vertical. The right button resets the episode. Uses the TELEOP cfg
preset (lift-style feel: held-target Newton full-pose IK, no smoothing, no RL time-out)
unless --rl-cfg is given.

    # interactive (needs a display + the SpaceMouse connected):
    ./scoop_run.sh -p _scratch/teleop_scoop.py --device cuda:0 --visualizer newton
    ./scoop_run.sh -p _scratch/teleop_scoop.py --device cuda:0 --visualizer kit

    # headless logic smoke-test (no device, zero command):
    ./scoop_run.sh -p _scratch/teleop_scoop.py --device cuda:0 --headless --mock --steps 60
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--pos-gain", type=float, default=1.0, help="SpaceMouse translation -> bowl x/y/z action gain")
parser.add_argument("--rot-gain", type=float, default=0.30, help="SpaceMouse twist -> bowl pitch action gain")
parser.add_argument("--yaw-gain", type=float, default=0.30, help="SpaceMouse yaw twist -> bowl yaw action gain")
parser.add_argument("--yaw-axis", type=int, default=5, help="which SpaceMouse rot axis aims the pour (3=roll,4=pitch,5=yaw)")
parser.add_argument("--invert-yaw", action="store_true", help="invert selected SpaceMouse yaw axis")
parser.add_argument("--rl-cfg", action="store_true", help="use the RL training cfg instead of the TELEOP preset")
parser.add_argument("--spacemouse-pos-sensitivity", type=float, default=0.4, help="SpaceMouse translation sensitivity")
parser.add_argument("--spacemouse-rot-sensitivity", type=float, default=0.8, help="SpaceMouse rotation sensitivity")
parser.add_argument("--deadzone", type=float, default=0.05, help="ignore SpaceMouse axes with magnitude below this")
parser.add_argument("--pitch-axis", type=int, default=4, help="which SpaceMouse rot axis drives pitch (3=roll,4=pitch,5=yaw)")
parser.add_argument("--ik-backend", choices=("diffik", "newton"), default=None, help="runtime IK backend override")
parser.add_argument("--reset-start", choices=("home", "source_curriculum"), default=None, help="reset pose policy override")
parser.add_argument("--container-geometry", choices=("bucket", "pour_bowl", "box"), default=None, help="container geometry override")
parser.add_argument("--invert-x", action="store_true", help="invert SpaceMouse x translation")
parser.add_argument("--invert-y", action="store_true", help="invert SpaceMouse y translation")
parser.add_argument("--invert-z", action="store_true", help="invert SpaceMouse z translation")
parser.add_argument("--invert-pitch", action="store_true", help="invert selected SpaceMouse pitch axis")
parser.add_argument("--debug-cmd", action="store_true", help="print raw SpaceMouse command/action mapping")
parser.add_argument("--steps", type=int, default=-1, help="stop after N steps; <0 runs until the viewer closes")
parser.add_argument("--mock", action="store_true", help="no device; zero command (for a headless smoke test)")
parser.add_argument("--debug-vis", action="store_true",
                    help="Kit: draw the live policy observations (heightfield grid + media centroids/target)")
add_launcher_args(parser)
args_cli = parser.parse_args()


def _requested_visualizers() -> set[str]:
    visualizer = getattr(args_cli, "visualizer", None)
    if not visualizer:
        return set()
    if isinstance(visualizer, str):
        return {token.strip().lower() for token in visualizer.split(",") if token.strip()}
    return {str(token).strip().lower() for token in visualizer if str(token).strip()}


def _ensure_display_for_kit() -> None:
    if os.environ.get("DISPLAY"):
        return
    x11_root = Path("/tmp/.X11-unix")
    for display in ("1", "0"):
        if (x11_root / f"X{display}").exists():
            os.environ["DISPLAY"] = f":{display}"
            return


def _make_cfg(device: str):
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg, FrankaScoopEnvCfg_TELEOP

    cfg = FrankaScoopEnvCfg() if args_cli.rl_cfg else FrankaScoopEnvCfg_TELEOP()
    cfg.scene.num_envs = 1
    cfg.sim.device = str(device)
    if args_cli.ik_backend is not None:
        cfg.ik_backend = args_cli.ik_backend
    if args_cli.reset_start is not None:
        cfg.reset_start = args_cli.reset_start
    if args_cli.container_geometry is not None:
        cfg.container_geometry = args_cli.container_geometry
    cfg.debug_vis_obs = bool(args_cli.debug_vis)
    return cfg


def _apply_deadzone(cmd, deadzone: float):
    if deadzone <= 0.0:
        return cmd
    return torch.where(torch.abs(cmd) < deadzone, torch.zeros_like(cmd), cmd)


def _run_env(cfg) -> None:
    import torch
    from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv

    env = FrankaScoopEnv(cfg)
    dev = env.device

    device = None
    if not args_cli.mock:
        from isaaclab.devices.spacemouse.se3_spacemouse import Se3SpaceMouse
        from isaaclab.devices.spacemouse.se3_spacemouse_cfg import Se3SpaceMouseCfg

        device = Se3SpaceMouse(Se3SpaceMouseCfg(
            pos_sensitivity=args_cli.spacemouse_pos_sensitivity,
            rot_sensitivity=args_cli.spacemouse_rot_sensitivity,
            gripper_term=False,
            sim_device=str(dev),
        ))
        print(device, flush=True)
        device.add_callback("R", lambda: setattr(main, "_reset_req", True))
        device.reset()
    main._reset_req = False

    obs, _ = env.reset()
    act = torch.zeros(1, env.action_manager.total_action_dim, device=dev)
    pax = max(3, min(5, args_cli.pitch_axis))
    yax = max(3, min(5, args_cli.yaw_axis))
    signs = torch.tensor([
        -1.0 if args_cli.invert_x else 1.0,
        -1.0 if args_cli.invert_y else 1.0,
        -1.0 if args_cli.invert_z else 1.0,
        -1.0 if args_cli.invert_pitch else 1.0,
        -1.0 if args_cli.invert_yaw else 1.0,
    ], device=dev)
    print("[TELEOP] ready: translation is env/world x/y/z; twist axes tilt the bowl (pitch) and "
          "aim the pour (yaw). Right button=reset.", flush=True)

    def viewer_running() -> bool:
        vis = getattr(env.sim, "visualizers", None)
        if not vis:
            return True  # headless / no viewer -> rely on --steps
        return any((not getattr(v, "is_closed", False)) and v.is_running() for v in vis)

    count = 0
    try:
        while viewer_running() and (args_cli.steps < 0 or count < args_cli.steps):
            if main._reset_req:
                env.reset()
                if device is not None:
                    device.reset()
                main._reset_req = False
            if device is not None:
                raw_cmd = device.advance().to(dev)  # [dx, dy, dz, rx, ry, rz]
                cmd = _apply_deadzone(raw_cmd, args_cli.deadzone)
                act[0, 0] = torch.clamp(cmd[0] * signs[0] * args_cli.pos_gain, -1.0, 1.0)
                act[0, 1] = torch.clamp(cmd[1] * signs[1] * args_cli.pos_gain, -1.0, 1.0)
                act[0, 2] = torch.clamp(cmd[2] * signs[2] * args_cli.pos_gain, -1.0, 1.0)
                act[0, 3] = torch.clamp(cmd[pax] * signs[3] * args_cli.rot_gain, -1.0, 1.0)
                act[0, 4] = torch.clamp(cmd[yax] * signs[4] * args_cli.yaw_gain, -1.0, 1.0)
                if args_cli.debug_cmd and count % 15 == 0:
                    print(f"[TELEOP] raw={[round(float(x), 3) for x in raw_cmd]} "
                          f"cmd={[round(float(x), 3) for x in cmd]} "
                          f"act={[round(float(x), 3) for x in act[0]]}", flush=True)
            obs, rew, term, trunc, info = env.step(act)
            if count % 30 == 0:
                print(f"[TELEOP] bowl_e={[round(float(x), 3) for x in env.bowl_pos_e()[0]]} "
                      f"pitch={float(env._pitch[0]):+.2f} yaw={float(env._yaw[0]):+.2f} "
                      f"in_bowl={int(env.count_in_bowl()[0])} "
                      f"src={int(env.count_in_source()[0])} tgt={int(env.count_in_target()[0])} "
                      f"fin={bool(torch.isfinite(obs['policy']).all())}", flush=True)
            count += 1
    finally:
        env.close()


def main() -> None:
    # launch_simulation configures the Newton coupled solver AND the requested visualizer (kit/newton/...).
    # A bare AppLauncher path skips that wiring and the scoop env tears itself down before the render loop.
    if "kit" in _requested_visualizers():
        _ensure_display_for_kit()
    cfg = _make_cfg(str(args_cli.device))
    with launch_simulation(cfg, args_cli):
        _run_env(cfg)


if __name__ == "__main__":
    main()
