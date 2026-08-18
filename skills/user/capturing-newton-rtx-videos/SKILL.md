---
name: isaaclab-capturing-newton-rtx-videos
description: Captures polished, reproducible MP4 demonstrations from Isaac Lab RL tasks with the Newton RTX (OVRTX) path tracer. Use when recording a task or policy rollout with the shared --video flags, choosing Newton RTX framing or lighting, or diagnosing empty, unavailable, or low-quality Newton RTX captures.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Capture Newton RTX Videos

## When To Use

Use this skill for a presentation-quality Newton render, not for fast interactive debugging. Use Newton GL for iteration, then switch to Newton RTX for the final bounded capture.

## Workflow

1. Start from a physically stable task and reduce the recording to the one or few environments that tell the story. Use a seeded policy rollout for reproducible motion.
2. Use the RL entrypoint's existing `--video`, `--video_length`, and `--video_interval` flags. Do not add duplicate video arguments or a bespoke export loop to the task. The entrypoint populates `env_cfg.video_recorders`, and Isaac Lab's environment recorder writes H.264 MP4 through MoviePy.
3. Select the visualizer explicitly with `--viz newton_rtx` (equivalently `--visualizer newton_rtx`). Without it, `--video` falls back to a headless Kit visualizer rather than Newton RTX. Install MoviePy and its FFmpeg runtime with the `video` extra.
4. Select Newton physics separately. For the manager-based Cartpole task, use `physics=newton_mjwarp`; the visualizer and physics backend are independent choices.
5. Compose in the task configuration: set an intentional `sim.default_visualizer_cfg` camera, keep the subject in frame for the whole action, and use readable material colors. When a task needs RTX-only settings such as 1920x1080 output or studio lighting, use a `NewtonRTXVisualizerCfg` as `sim.default_visualizer_cfg`; the CLI still decides whether to activate it. Do not put camera or resolution settings on `VideoRecorderCfg`.
6. Train a policy without expensive presentation rendering, then use the same task's play command for the final clip. This manager-based Cartpole example records one environment for 240 environment steps:

   ```bash
   uv run --extra video isaaclab play --rl_library rsl_rl \
       --task Isaac-Cartpole --checkpoint latest --num_envs 1 \
       --video --video_length 240 --video_interval 0 \
       --viz newton_rtx --max_visible_envs 1 \
       physics=newton_mjwarp \
       'env.sim.default_visualizer_cfg.eye=[4.0,-5.0,3.5]' \
       'env.sim.default_visualizer_cfg.lookat=[0.0,0.0,2.5]' \
       env.sim.default_visualizer_cfg.focal_length=18.0
   ```

   `--checkpoint latest` requires an existing Cartpole RSL-RL run. If none exists, train it first as shown in [Examples](examples.md). The clip is written under the selected run's `videos/play` directory.
   The camera targets the Cartpole asset around its spawned height instead of looking at the ground origin.
7. Inspect the MP4 at native resolution. Verify the camera, lighting, motion, frame rate, and clip duration before increasing resolution or duration; Newton RTX capture performs a GPU-to-CPU readback for every sampled frame.

## Validation

1. Run a short capture first and confirm that the output is a playable MP4 in `<log_dir>/videos/train` or `<log_dir>/videos/play`, with the expected frame count and frame rate.
2. Confirm that the startup summary reports `source: visualizer:newton_rtx`, the OVRTX runtime initializes, and the recorder reports no frame-capture error.
3. Confirm reproducibility by repeating the same seeded bounded rollout and checking for the same framing, action timing, and absence of user interaction.
4. Confirm that the final camera keeps the subject and its key contact or motion event visible without clipping.
5. For skill changes, run:

   ```bash
   uv run --no-project python tools/skills/cli.py check
   ```

## Maintenance

Keep this skill synchronized with the RL entrypoint video flags, `VideoRecorderCfg`, the environment recorder, the Newton RTX visualizer, and the video-recording guide. Do not promise controls that `NewtonRTXVisualizerCfg` does not expose; path-tracer quality settings currently use ViewerRTX defaults.

## References

- [Reference](reference.md)
- [Examples](examples.md)
- [Evaluations](evaluations.md)
- [Newton RTX visualizer config](../../../source/isaaclab_visualizers/isaaclab_visualizers/newton/newton_visualizer_cfg.py)
- [Newton RTX visualizer implementation](../../../source/isaaclab_visualizers/isaaclab_visualizers/newton/newton_visualizer.py)
- [Video recording guide](../../../docs/source/how-to/record_video.rst)
- [Visualization guide](../../../docs/source/overview/core-concepts/visualization.rst)
