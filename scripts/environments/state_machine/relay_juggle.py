# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a KUKA-Allegro toss with a physical GR1T2 open-hand deflection.

.. code-block:: bash

    uv run --extra video python scripts/environments/state_machine/relay_juggle.py --video --max_steps 720

The default ``--video`` source records Newton GL under ``videos/relay_juggle``.
For Kit, use ``--video_source kit --viz kit`` with the ``video,isaacsim`` extras.
Either visualizer may bootstrap Kit and require prior acceptance of NVIDIA's
Omniverse EULA; this script does not accept it automatically.

The scripted demo validates the outbound release and a post-release GR1T2
touch. It does not demonstrate a GR1T2 catch or return throw.
"""

from __future__ import annotations

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Measured KUKA toss and GR1T2 open-hand deflection demo.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of relay stages to simulate.")
parser.add_argument("--max_steps", type=int, default=720, help="Maximum environment steps before exiting.")
parser.add_argument("--video", action="store_true", help="Record one Newton GL or Kit viewport clip.")
parser.add_argument("--video_source", choices=("newton", "kit"), default="newton", help="Video viewport source.")
parser.add_argument("--video_dir", type=str, default="videos/relay_juggle", help="Video output directory.")
parser.add_argument("--video_length", type=int, default=600, help="Maximum recorded environment steps.")
parser.add_argument("--diagnostics", action="store_true", help="Print periodic ball/hand kinematics.")
parser.add_argument("--diagnostic_interval", type=int, default=30, help="Steps between diagnostic state lines.")
parser.add_argument("--seed", type=int, default=42, help="Deterministic simulation seed.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])


def main() -> None:
    """Launch, drive, and summarize a finite physical outbound deflection."""
    args = parser.parse_args()
    if args.num_envs < 1 or args.max_steps < 1 or args.video_length < 1 or args.diagnostic_interval < 1:
        parser.error("num_envs, max_steps, video_length, and diagnostic_interval must be positive.")
    if args.video:
        if args.video_source == "kit":
            if args.visualizer not in (None, ["newton_gl"], ["kit"]):
                parser.error("--video_source kit requires --viz kit.")
            args.visualizer = ["kit"]
            args.enable_cameras = True
        else:
            if args.visualizer not in (None, ["newton_gl"]):
                parser.error("--video_source newton requires --viz newton_gl.")
            args.visualizer = ["newton_gl"]

    import gymnasium as gym
    import torch

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.contrib.relay_juggle.relay_env_cfg import TASK_ID, RelayJuggleEnvCfg

    cfg = RelayJuggleEnvCfg()
    cfg.seed = args.seed
    cfg.record_relay_invalid_details = True
    cfg.scene.num_envs = args.num_envs
    if args.device is not None:
        cfg.sim.device = args.device
    if args.video:
        if args.video_source == "kit":
            from isaaclab_visualizers.kit import KitVisualizerCfg

            cfg.sim.visualizer_cfgs = [KitVisualizerCfg(eye=(1.35, -2.15, 1.65), lookat=(-0.28, 0.0, 0.95))]
        else:
            from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

            cfg.sim.visualizer_cfgs = [
                NewtonGLVisualizerCfg(
                    eye=(1.35, -2.15, 1.65),
                    lookat=(-0.28, 0.0, 0.95),
                    window_width=1280,
                    window_height=720,
                )
            ]
        cfg.video_recorders = [
            VideoRecorderCfg(
                source=f"visualizer:{args.video_source}",
                output_dir=args.video_dir,
                output_filename_prefix=f"kuka_gr1t2_relay_{args.video_source}",
                video_length=min(args.video_length, args.max_steps),
                fps=30,
                step_offset=5 if args.video_source == "kit" else 0,
            )
        ]

    with launch_simulation(cfg, args):
        from isaaclab_tasks.contrib.relay_juggle.relay_state_machine import RelayStateMachine

        env = gym.make(TASK_ID, cfg=cfg)
        try:
            env.reset(seed=args.seed)
            machine = RelayStateMachine(env.unwrapped)
            if args.diagnostics:
                print(f"[relay] initial {machine.diagnostic_state()}")
                thumb_id = int(machine.gr1_hand_ids[-1])
                print(
                    "[relay] GR1 right thumb distal joint velocity limits "
                    f"solver={machine.gr1.data.joint_vel_limits.torch[0, thumb_id].item():.3f} rad/s, "
                    f"soft={machine.gr1.data.soft_joint_vel_limits.torch[0, thumb_id].item():.3f} rad/s"
                )
            reset_count = 0
            for step in range(args.max_steps):
                with torch.inference_mode():
                    actions = machine.act()
                    _, _, terminated, truncated, _ = env.step(actions)
                    done = terminated | truncated
                    if bool(done.any()):
                        reset_count += int(done.sum().item())
                        reasons = {
                            term: int(env.unwrapped.termination_manager.get_term(term)[done].sum().item())
                            for term in env.unwrapped.termination_manager.active_terms
                        }
                        print(f"[relay] step={step + 1} reset reasons={reasons}")
                        if reasons.get("invalid_state", 0):
                            details = getattr(env.unwrapped, "_relay_invalid_state_details", {})
                            print(f"[relay] invalid-state fields={details}")
                        machine.reset(done.nonzero(as_tuple=False).flatten())
                if args.diagnostics and ((step + 1) % args.diagnostic_interval == 0 or bool(done.any())):
                    print(f"[relay] step={step + 1} {machine.diagnostic_state()}")
                if (step + 1) % 120 == 0 or step + 1 == args.max_steps:
                    print(f"[relay] step={step + 1} resets={reset_count} metrics={machine.metrics()}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
