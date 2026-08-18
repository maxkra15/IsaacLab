# Newton RTX Capture Reference

## Capture contract

- Select `newton_rtx`, not the deprecated `newton` alias (which selects Newton GL).
- `NewtonRTXVisualizerCfg` uses the OVRTX path tracer and captures its native-resolution LDR RGB framebuffer through `render_rgb_array()`.
- Configure the framebuffer with `window_width` and `window_height`; configure its lighting with `rtx_environment`: `default`, `studio`, or `none`.
- `render_rgb_array()` returns a `uint8` array shaped `(H, W, 3)` or `None` if the viewer is unavailable. Treat `None` as a capture error rather than writing a partial clip.
- Each captured RTX frame reads pixels back from the GPU. Keep early review clips short and sample only at the output frame rate.

## RL recording path

For RL tasks, use the `isaaclab train` or `isaaclab play` entrypoint. Their shared `--video` flags configure Isaac Lab's environment recorder; do not add a task-local parser or manual frame loop.

`--video` enables recording. `--video_length` and `--video_interval` override the corresponding fields on every configured recorder. With no task-side recorder, `--viz newton_rtx` causes the entrypoint to create `VideoRecorderCfg(source="visualizer:newton_rtx")` and choose the standard log directory automatically. The environment captures after its steps and encodes H.264 MP4 with MoviePy.

MoviePy and its FFmpeg runtime are supplied by the project `video` extra; invoke the recording command with `uv run --extra video ...`.

## Task configuration boundary

- Put camera composition and visualizer-specific resolution or lighting on `sim.default_visualizer_cfg` or a backend-specific config in `sim.visualizer_cfgs`.
- Put persistent capture cadence, FPS, output naming, and source selection in `env_cfg.video_recorders` only when CLI defaults are insufficient.
- Keep `--video`, `--video_length`, and `--video_interval` in the shared RL entrypoint. A task should not define duplicates.
- Select physics independently from rendering. `physics=newton_mjwarp` chooses Newton physics; `--viz newton_rtx` chooses the OVRTX visualizer.

## Presentation checklist

- Begin with a stable reset and deterministic control or scripted motion.
- Frame the object of interest at a three-quarter angle, preserve foreground/background separation, and leave room for its full motion.
- Use `studio` when material shape and highlights need to read clearly; use `default` when a broader scene benefits from dome lighting.
- Use physically meaningful materials and simple, contrasting colors. Avoid changing physics solely to improve appearance.
- Keep capture-only work isolated from training defaults: high-resolution RTX readback is intentionally expensive.

## Current limitations

Newton RTX is experimental. Visualization markers, live plots, and the tiled streaming-camera panel are not displayable in the RTX viewer. Its path tracer keeps rendering at full cost while paused. Use Newton GL, Rerun, or Viser for interactive instrumentation, then return to RTX for the final render.
