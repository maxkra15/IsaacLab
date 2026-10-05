# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Insert, release, regrasp and extract an Ethernet cable in one physical episode.

The manager-based task owns physics and reset. This example returns ordinary
TCP and finger targets only; it does not reset at the handoff or attach the plug.
Hydroelastic clip contact currently requires the Newton corrections documented
in the task README. Select a corrected checkout explicitly with PYTHONPATH;
the example never patches or replaces the installed dependency.

.. code-block:: bash

    uv run isaaclab example gb300-sharpa-insertion
    uv run --extra isaacsim --extra video isaaclab example gb300-sharpa-insertion --viz kit --video
    uv run isaaclab example gb300-sharpa-insertion --viz none --no_presentation

"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def main() -> int:
    """Run the bounded physical cycle; return success only after extraction."""
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--max_steps",
        "--num_steps",
        dest="num_steps",
        type=int,
        default=3600,
        help="Maximum steps in one uninterrupted episode.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no_presentation", action="store_true", help="Skip render-only assets, retaining collisions.")
    parser.add_argument("--video", action="store_true", help="Record with the standard task video recorder.")
    parser.add_argument("--output_dir", type=Path, default=Path("logs/demos/gb300_sharpa_insertion"))
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["newton_gl"])
    args = parser.parse_args()
    if args.num_steps <= 0:
        parser.error("--num_steps must be positive")
    from isaaclab_newton.physics import NewtonCfg

    with launch_simulation(NewtonCfg(), args):
        result = _run(args)
        if result != 0:
            raise SystemExit(result)
    return result


def _run(args: argparse.Namespace) -> int:
    import gymnasium as gym
    import numpy as np
    import torch
    from cable_installation.config import RELEASE_HAND_RAD, RemovalPostures
    from cable_installation.contacts import GraspContactReader
    from cable_installation.cycle import CableCycleStateMachine
    from cable_installation.insertion import CableInstallStateMachine

    from isaaclab.app import launch_simulation
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab.utils.math import subtract_frame_transforms

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.contrib.gb300_sharpa_insertion import mdp
    from isaaclab_tasks.contrib.gb300_sharpa_insertion.assets import RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES
    from isaaclab_tasks.contrib.gb300_sharpa_insertion.robot_cfg import CLOSE_HAND_RAD
    from isaaclab_tasks.contrib.gb300_sharpa_insertion.workcell import head_goal_for_socket
    from isaaclab_tasks.utils import parse_env_cfg

    task = "IsaacContrib-Rizon-Sharpa-GB300-Cable-Install"
    registered_cfg = parse_env_cfg(task, overrides=["presets=scripted"])
    # Composition-affecting options must be applied before scene construction.
    cfg = type(registered_cfg)(presentation_enabled=not args.no_presentation)
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.episode_length_s = args.num_steps * cfg.sim.dt * cfg.decimation + 1.0
    if args.video:
        cfg.video_recorders = [
            VideoRecorderCfg(
                output_dir=str(args.output_dir),
                output_filename_prefix=f"cycle_{datetime.now():%Y%m%d_%H%M%S_%f}",
                video_length=args.num_steps,
                frame_stride=2,
                fps=30,
            )
        ]
    with launch_simulation(cfg, args):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            env.reset(seed=args.seed)
            if env.action_manager.total_action_dim != 29:
                raise RuntimeError("The scripted preset must provide absolute TCP and 22 finger targets.")
            goal_position = np.asarray(head_goal_for_socket(cfg.target_socket_position_e))
            goal_position += env.scene.env_origins[0].cpu().numpy()
            insertion = CableInstallStateMachine(
                mdp.tcp_pose(env)[0].cpu().numpy(),
                np.r_[goal_position, 2**-0.5, 0.0, 0.0, 2**-0.5],
                env.step_dt,
            )
            machine = CableCycleStateMachine(insertion, RemovalPostures(), CLOSE_HAND_RAD, RELEASE_HAND_RAD)
            reader = GraspContactReader()
            robot = env.scene["robot"]
            hand_ids, _ = robot.find_joints(RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES, preserve_order=True)
            initial_hand = robot.data.joint_pos.torch[:, hand_ids].clone()
            closed_hand = initial_hand.new_tensor(CLOSE_HAND_RAD)[None]
            step = reset_count = 0
            reason = "Simulation closed or step budget exhausted before the complete cycle."
            prior_phase = machine.phase
            with torch.inference_mode():
                while env.sim.is_running() and step < args.num_steps:
                    if not env.sim.is_playing():
                        env.sim.render()
                        continue
                    command, finger_targets = machine.update(
                        mdp.tcp_pose(env)[0].cpu().numpy(),
                        mdp.head_pose(env)[0].cpu().numpy(),
                        reader.read(),
                        latch_engaged=bool(mdp.latch_captured(env)[0]),
                        latch_angle_rad=float(mdp.latch_state(env)[0, 0]),
                        capture_margins_m=mdp.latch_capture_margins(env)[0].cpu().numpy(),
                        keeper_release_margin_m=float(mdp.latch_release_margin(env)[0]),
                        keeper_exit_margin_m=float(mdp.latch_keeper_exit_margin(env)[0]),
                    )
                    if machine.phase != prior_phase:
                        print(f"[GB300 cycle] step={step} phase={machine.phase}", flush=True)
                        prior_phase = machine.phase
                    if machine.phase in ("DONE", "FAILED"):
                        reason = machine.failure
                        break
                    target = torch.as_tensor(command, dtype=torch.float32, device=env.device)[None]
                    root = robot.data.root_pose_w.torch
                    position, rotation = subtract_frame_transforms(
                        root[:, :3], root[:, 3:], target[:, :3], target[:, 3:]
                    )
                    fingers = torch.as_tensor(finger_targets, dtype=torch.float32, device=env.device)[None]
                    if step * env.step_dt < 0.2:
                        fraction = min((step + 1) * env.step_dt / 0.2, 1.0)
                        fingers = torch.lerp(initial_hand, closed_hand, fraction)
                    observations, rewards, terminated, truncated, _ = env.step(
                        torch.cat((position, rotation, fingers), dim=-1)
                    )
                    step += 1
                    if not torch.isfinite(observations["policy"]).all() or not torch.isfinite(rewards).all():
                        raise RuntimeError("Cable cycle encountered a non-finite simulation state.")
                    if bool((terminated | truncated).any()):
                        reset_count += int((terminated | truncated).sum())
                        reason = "Environment reset before the complete cycle."
                        break
            success = machine.phase == "DONE" and insertion.phase == "DONE" and reset_count == 0
            print(
                json.dumps(
                    {
                        "success": success,
                        "phase": machine.phase,
                        "reason": reason,
                        "insertion_completed": insertion.phase == "DONE",
                        "environment_resets": reset_count,
                        "steps": step,
                        "measured_withdrawal_m": machine.measured_withdrawal_m,
                        "maximum_penetration_m": machine.max_penetration_m,
                    },
                    indent=2,
                ),
                flush=True,
            )
            return 0 if success else 1
        finally:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
