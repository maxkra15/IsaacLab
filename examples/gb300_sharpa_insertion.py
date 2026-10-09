# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Insert, release, regrasp and extract an Ethernet cable in one physical episode.

The manager-based task owns physics and reset. This example returns ordinary
TCP and finger targets only; it does not reset at the handoff or attach the plug to the hand.
Measured pose and latch feedback remain active; CPU contact diagnostics are opt-in.
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
import logging
import math
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


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
    parser.add_argument("--settle_duration_s", type=float, default=0.0, help="Initial physical grasp hold [s].")
    parser.add_argument("--no_presentation", action="store_true", help="Skip render-only assets, retaining collisions.")
    # Retain the old flag; the scripted preset already selects this boundary.
    parser.add_argument("--one_way_cable", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--video", action="store_true", help="Record with the standard task video recorder.")
    parser.add_argument("--validate_contacts", action="store_true", help="Enable CPU contact diagnostics and guards.")
    parser.add_argument(
        "--failure_tail_s", type=float, default=0.0, help="Hold robot targets and keep simulating after a fault [s]."
    )
    parser.add_argument("--output_dir", type=Path, default=Path("logs/demos/gb300_sharpa_insertion"))
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["newton_gl"])
    args = parser.parse_args()
    if args.num_steps <= 0:
        parser.error("--num_steps must be positive")
    if not math.isfinite(args.settle_duration_s) or args.settle_duration_s < 0.0:
        parser.error("--settle_duration_s must be finite and nonnegative")
    if not math.isfinite(args.failure_tail_s) or args.failure_tail_s < 0.0:
        parser.error("--failure_tail_s must be finite and nonnegative")
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
    from cable_installation.config import RemovalPostures
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
    cfg = type(registered_cfg)(presentation_enabled=not args.no_presentation, one_way_cable=args.one_way_cable)
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.episode_length_s = args.num_steps * cfg.sim.dt * cfg.decimation + 1.0
    if args.video:
        cfg.sim.default_visualizer_cfg.eye = (0.31, -0.035, 2.18)
        cfg.sim.default_visualizer_cfg.lookat = (0.52, 0.25, 2.08)
        cfg.sim.default_visualizer_cfg.focal_length = 18.0
        cfg.sim.default_visualizer_cfg.streaming_view = False
        cfg.video_recorders = [
            VideoRecorderCfg(
                output_dir=str(args.output_dir),
                output_filename_prefix=f"cycle_{datetime.now():%Y%m%d_%H%M%S_%f}",
                video_length=args.num_steps,
            )
        ]
    with launch_simulation(cfg, args):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            observations, _ = env.reset(seed=args.seed)
            if env.action_manager.total_action_dim != 29:
                raise RuntimeError("The scripted preset must provide absolute TCP and 22 finger targets.")

            def controller_state(status: torch.Tensor) -> np.ndarray:
                # One small readback: TCP, head, hinge, capture/keeper geometry, finite/reset status.
                captured, capture_margins, release_margin, exit_margin = mdp.latch_metrics(env)
                return (
                    torch.cat(
                        (
                            mdp.tcp_pose(env)[0],
                            mdp.head_pose(env)[0],
                            mdp.latch_state(env)[0, :1],
                            captured[0, None],
                            capture_margins[0],
                            release_margin[0, None],
                            exit_margin[0, None],
                            status,
                        )
                    )
                    .cpu()
                    .numpy()
                )

            state = controller_state(
                torch.stack(
                    (
                        torch.isfinite(observations["policy"]).all(),
                        torch.zeros((), dtype=torch.bool, device=env.device),
                    )
                )
            )
            if not state[-2]:
                raise RuntimeError("Cable cycle encountered a non-finite reset state.")
            goal_position = np.asarray(head_goal_for_socket(cfg.target_socket_position_e))
            goal_position += env.scene.env_origins[0].cpu().numpy()
            insertion = CableInstallStateMachine(
                state[:7],
                np.r_[goal_position, 2**-0.5, 0.0, 0.0, 2**-0.5],
                env.step_dt,
                retreat_offset_w=(0.0, 0.0, 0.0),
            )
            postures = RemovalPostures()
            machine = CableCycleStateMachine(
                insertion,
                postures,
                CLOSE_HAND_RAD,
                settle_duration_s=args.settle_duration_s,
            )
            reader = None
            if args.validate_contacts:
                from cable_installation.contacts import GraspContactReader

                reader = GraspContactReader()
            robot = env.scene["robot"]
            hand_ids, _ = robot.find_joints(RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES, preserve_order=True)
            initial_hand = robot.data.joint_pos.torch[:, hand_ids].clone()
            closed_hand = initial_hand.new_tensor(CLOSE_HAND_RAD)[None]
            step = reset_count = 0
            failure_step: int | None = None
            failure_tail_steps = math.ceil(min(args.failure_tail_s / env.step_dt, args.num_steps))
            reason = "Simulation closed or step budget exhausted before the complete cycle."
            prior_phase = machine.phase
            with torch.inference_mode():
                while env.sim.is_running() and step < args.num_steps:
                    if not env.sim.is_playing():
                        env.sim.render()
                        continue
                    if failure_step is None:
                        command, finger_targets = machine.update(
                            state[:7],
                            state[7:14],
                            None if reader is None else reader.read(),
                            latch_engaged=bool(state[15]),
                            latch_angle_rad=float(state[14]),
                            capture_margins_m=state[16:19],
                            keeper_release_margin_m=float(state[19]),
                            keeper_exit_margin_m=float(state[20]),
                        )
                    if machine.phase != prior_phase:
                        logger.info("[GB300 cycle] step=%d phase=%s", step, machine.phase)
                        prior_phase = machine.phase
                    if machine.phase == "DONE":
                        reason = machine.failure
                        break
                    if machine.phase == "FAILED":
                        if failure_step is None:
                            failure_step = step
                            reason = machine.failure
                            logger.warning(
                                "[GB300 cycle] fault=%s; holding targets for %g s", reason, args.failure_tail_s
                            )
                        if step - failure_step >= failure_tail_steps:
                            break
                    targets = torch.as_tensor(
                        np.r_[command, finger_targets], dtype=torch.float32, device=env.device
                    ).unsqueeze(0)
                    root = robot.data.root_pose_w.torch
                    position, rotation = subtract_frame_transforms(
                        root[:, :3], root[:, 3:], targets[:, :3], targets[:, 3:7]
                    )
                    fingers = targets[:, 7:]
                    if failure_step is None and step * env.step_dt < 0.2:
                        fraction = min((step + 1) * env.step_dt / 0.2, 1.0)
                        fingers = torch.lerp(initial_hand, closed_hand, fraction)
                    observations, rewards, terminated, truncated, _ = env.step(
                        torch.cat((position, rotation, fingers), dim=-1)
                    )
                    step += 1
                    state = controller_state(
                        torch.stack(
                            (
                                torch.isfinite(observations["policy"]).all() & torch.isfinite(rewards).all(),
                                (terminated | truncated).sum(),
                            )
                        )
                    )
                    if not state[-2]:
                        raise RuntimeError("Cable cycle encountered a non-finite simulation state.")
                    if state[-1]:
                        reset_count += int(state[-1])
                        if failure_step is None:
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
                        "failure_step": failure_step,
                        "post_failure_simulation_s": 0.0
                        if failure_step is None
                        else (step - failure_step) * env.step_dt,
                        "measured_withdrawal_m": 0.0 if machine.removal is None else float(machine.goal[1] - state[8]),
                        "contact_validation": reader is not None,
                        "maximum_penetration_m": None if reader is None else machine.max_penetration_m,
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
