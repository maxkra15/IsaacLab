# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture a saved policy's real-MPM reset-dataset replay with Newton GL.

This entry point reuses the evaluator's configuration, policy loading and native
simulation loop. Video is illustrative; use the separate 64-world evaluations
for task metrics. Invoke only after the selected GPU has been released by its
owner, with CUDA_VISIBLE_DEVICES set to that GPU's UUID.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.mpm_world_model.collect_franka_pour import argument_parser, run, validate_arguments
from examples.mpm_world_model.franka_pour_state import configure_source_paths
from examples.mpm_world_model.recording import atomic_json


class CaptureTimeline:
    """Relate actual captured frames to native steps and auto-reset boundaries."""

    def __init__(self, warmup_steps: int):
        self.warmup_steps = warmup_steps
        self.policy_step = 0
        self.episode_id = 0
        self.frames = []
        self.resets = []

    def reset(self, old_row: int, new_row: int, reasons: list[str], episode_length: int) -> None:
        previous_episode = self.episode_id
        if self.policy_step:
            self.episode_id += 1
        self.resets.append(
            {
                "policy_step": self.policy_step,
                "old_episode_id": previous_episode,
                "new_episode_id": self.episode_id,
                "old_reset_row": old_row,
                "new_reset_row": new_row,
                "termination_reasons": reasons,
                "completed_episode_length_policy_steps": episode_length,
                "initialization_reset": self.policy_step == 0,
            }
        )

    def observe_reset(self, env, env_ids, native_reset):
        """Record a native tensor/slice selection without changing its reset call."""
        selected = range(env.num_envs)[env_ids] if isinstance(env_ids, slice) else env_ids
        active = len(selected) > 0
        if active:
            old_row = int(env.reset_dataset_row_id[0])
            length = int(env.episode_length_buf[0])
            reasons = (
                [
                    name
                    for name in env.termination_manager.active_terms
                    if bool(env.termination_manager.get_term(name)[0])
                ]
                if self.policy_step
                else []
            )
        result = native_reset(env, env_ids)
        if active:
            self.reset(old_row, int(env.reset_dataset_row_id[0]), reasons, length)
        return result

    def frame(self, row: int, episode_step: int, dt: float, shape: list[int]) -> None:
        self.frames.append(
            {
                "frame_index": len(self.frames),
                "policy_step": self.policy_step,
                "simulated_time_s": self.policy_step * dt,
                "warmup": self.policy_step <= self.warmup_steps,
                "episode_id": self.episode_id,
                "reset_row": row,
                "episode_step": episode_step,
                "resets_this_step": [
                    index for index, event in enumerate(self.resets) if event["policy_step"] == self.policy_step
                ],
                "shape": shape,
                "displayed_state": "post_step_after_auto_reset",
            }
        )


def capture_parser():
    """Expose the evaluator's native inputs and a small set of capture options."""
    parser = argument_parser(evaluation=True)
    parser.description = __doc__
    parser.set_defaults(num_envs=1, warmup_steps=16, headless=True)
    parser.add_argument("--label", default="Saved policy", help="Human-readable policy label saved in frame metadata.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--eye", type=float, nargs=3, default=(0.9, 0.65, 0.5))
    parser.add_argument("--lookat", type=float, nargs=3, default=(0.5, 0.0, 0.1))
    parser.add_argument(
        "--egl-device-index",
        type=int,
        default=0,
        help="Host EGL device ordinal; independent of the UUID-masked logical CUDA device.",
    )
    return parser


def validate_capture(parser, args) -> None:
    """Reject changes to the deterministic evaluation and playback contract."""
    validate_arguments(parser, args)
    if args.num_envs != 1 or args.mode != "mpm":
        parser.error("Capture requires one real-MPM world; use evaluate_franka_pour for batched metrics.")
    if args.perturb_fraction or args.perturb_std or args.train_only or args.fine_tune_iterations:
        parser.error("Capture replays the saved actor without perturbation or training.")
    if min(args.width, args.height) < 1 or args.width % 2 or args.height % 2 or args.egl_device_index < 0:
        parser.error("Use positive even video dimensions and a nonnegative EGL device index.")
    if not all(math.isfinite(value) for value in (*args.eye, *args.lookat)) or args.eye == args.lookat:
        parser.error("Camera coordinates must be finite and eye/lookat must differ.")
    if args.device != "cuda:0":
        parser.error("Use logical --device cuda:0 with exactly one UUID-masked physical GPU.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new output directory; existing capture artifacts are never overwritten.")


def capture(args) -> dict:
    """Add scoped rendering observers to the unchanged native evaluation loop."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.startswith("GPU-") or "," in visible:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly one GPU UUID before launching capture.")
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)
    nvidia_egl = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    if nvidia_egl.is_file():
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(nvidia_egl)
    # Pyglet's EGL ordinal is a host ordinal, not the masked CUDA ordinal.
    import pyglet

    pyglet.options["headless"] = True
    pyglet.options["headless_device"] = args.egl_device_index
    configure_source_paths(args.repo, args.newton_root)
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    import isaaclab.app as application
    from isaaclab.envs.utils.video_recorder import VideoRecorder
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab.utils import to_dict
    from isaaclab.visualizers.visualizer_cfg import WindowCfg

    from isaaclab_tasks.contrib.franka_pour.pour_env import FrankaPourEnv
    from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import _resolve_pour_solver_tree

    timeline = CaptureTimeline(args.warmup_steps)
    observer = {"env": None, "complete": False}
    metadata = {
        "label": args.label,
        "role": "illustrative_native_mpm_video_not_batched_evaluation",
        "description": "Saved policy; reset-dataset replay; real MPM; physical particles; 1x playback",
        "camera": {"eye": args.eye, "lookat": args.lookat, "width": args.width, "height": args.height},
        "cuda_visible_devices": visible,
        "egl_device_index": args.egl_device_index,
        "playback_rate": 1.0,
        "fps": 30,
        "expected_frame_count": args.warmup_steps + args.steps,
        "expected_duration_s": (args.warmup_steps + args.steps) / 30,
        "warmup_steps_included_in_video": args.warmup_steps,
        "physics_metrics": "evaluation.json",
        "provenance": "provenance.json",
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    patches = []

    def patch(owner, name, replacement):
        patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    original_launch = application.launch_simulation

    @contextmanager
    def launch_with_video(cfg, launcher_args):
        cfg.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=tuple(args.eye),
                lookat=tuple(args.lookat),
                headless=True,
                window=WindowCfg(size=(args.width, args.height)),
                show_particles=True,
                particle_color=(0.85, 0.72, 0.45),
                enable_picking=False,
            )
        ]
        cfg.video_recorders = [
            VideoRecorderCfg(
                source="viz:newton_gl",
                output_dir=str(args.output / "videos"),
                output_filename_prefix="native_mpm",
                video_length=args.steps + args.warmup_steps,
                fps=30,
                frame_stride=1,
            )
        ]
        tree = _resolve_pour_solver_tree(cfg)
        metadata["media_solver"] = to_dict(tree.media_solver)
        metadata["coupling"] = {
            "global_substeps": cfg.sim.physics.num_substeps,
            "media_substeps": tree.media_entry.substeps,
            "rigid_substeps": tree.arm_entry.substeps,
            "collision_decimation": cfg.sim.physics.collision_decimation,
        }
        with original_launch(cfg, launcher_args) as runtime:
            yield runtime

    patch(application, "launch_simulation", launch_with_video)
    original_init = FrankaPourEnv.__init__

    def initialize(env, *positional, **keywords):
        original_init(env, *positional, **keywords)
        observer["env"] = env
        env.sim.initialize_visualizers()
        env.sim.set_camera_view(tuple(args.eye), tuple(args.lookat))
        metadata["active_visualizers"] = [viz.cfg.visualizer_type for viz in env.sim.visualizers]
        metadata["policy_step_s"] = float(env.step_dt)
        if not math.isclose(env.step_dt, 1 / 30, rel_tol=0, abs_tol=1e-10):
            raise RuntimeError("Capture assumes the unchanged 30Hz Franka policy interface.")

    patch(FrankaPourEnv, "__init__", initialize)
    original_step = FrankaPourEnv.step

    def observed_step(env, *positional, **keywords):
        timeline.policy_step += 1
        return original_step(env, *positional, **keywords)

    patch(FrankaPourEnv, "step", observed_step)
    original_reset = FrankaPourEnv._reset_idx

    def observed_reset(env, env_ids):
        if observer["env"] is env:
            return timeline.observe_reset(env, env_ids, original_reset)
        return original_reset(env, env_ids)

    patch(FrankaPourEnv, "_reset_idx", observed_reset)
    original_recorder_init = VideoRecorder.__init__

    def initialize_recorder(recorder, cfg, env):
        original_recorder_init(recorder, cfg, env)
        original_capture = recorder._capture

        def observed_capture():
            frame = original_capture()
            if frame is not None:
                timeline.frame(
                    int(env.reset_dataset_row_id[0]),
                    int(env.episode_length_buf[0]),
                    float(env.step_dt),
                    list(frame.shape),
                )
            return frame

        recorder._capture = observed_capture

    patch(VideoRecorder, "__init__", initialize_recorder)
    try:
        result = run(args)
        videos = sorted((args.output / "videos").glob("*.mp4"))
        if len(videos) != 1 or not videos[0].stat().st_size:
            raise RuntimeError("Native VideoRecorder did not write exactly one nonempty MP4; inspect run.log.")
        if len(timeline.frames) != args.warmup_steps + args.steps:
            raise RuntimeError("Captured frame count does not match the physical policy-step clock.")
        if any(frame["shape"][:2] != [args.height, args.width] for frame in timeline.frames):
            raise RuntimeError("Captured frames do not match the requested camera dimensions.")
        metadata["videos"] = [
            {
                "path": str(path.relative_to(args.output)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in videos
        ]
        observer["complete"] = True
        return result
    finally:
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)
        metadata.update(complete=observer["complete"], frames=timeline.frames, resets=timeline.resets)
        atomic_json(args.output / "capture.json", metadata)


def main() -> None:
    parser = capture_parser()
    args = parser.parse_args()
    validate_capture(parser, args)
    print(json.dumps(capture(args), indent=2))


if __name__ == "__main__":
    main()
