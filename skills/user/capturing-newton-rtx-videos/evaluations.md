# Newton RTX Video Capture Evaluations

## Scenario 1: Record Manager-Based Cartpole

Query: "Make a Newton RTX video of the manager-based Cartpole policy."

Expected behavior:

- Uses the existing `isaaclab play` entrypoint with task `Isaac-Cartpole`.
- Passes `--video`, a bounded `--video_length`, and `--viz newton_rtx`.
- Selects Newton physics independently with `physics=newton_mjwarp`.
- Uses one visible environment and explains that the MP4 lands in the run's `videos/play` directory.

Known failure modes:

- Adds a separate Cartpole export script or a duplicate task-local `--video` flag.
- Omits `--viz newton_rtx`, causing `--video` to auto-create the default Kit capture backend.
- Omits the `video` extra and fails to import MoviePy.

## Scenario 2: Configure A New RL Task

Query: "Make my manager-based RL task produce a polished Newton RTX clip when I pass --video."

Expected behavior:

- Reuses the shared train/play `--video`, `--video_length`, and `--video_interval` arguments.
- Configures the camera and any RTX-only resolution or lighting choices on the visualizer config.
- Adds `VideoRecorderCfg(source="visualizer:newton_rtx")` only if persistent task-side recorder defaults are required.
- Keeps normal training free of RTX capture unless the flags request it.

Known failure modes:

- Adds another parser or manually calls MoviePy from the task.
- Places camera or resolution fields on `VideoRecorderCfg`.

## Scenario 3: Improve A Flat-Looking Rollout

Query: "The Newton RTX rollout works but looks flat. How should I make it presentation-ready?"

Expected behavior:

- Recommends an intentional camera, legible materials, a small number of environments, and `rtx_environment="studio"` when appropriate.
- Preserves the physical scenario and deterministic control path.
- Notes that resolution and per-frame readback increase cost, so a short review render should come first.

Known failure modes:

- Claims that unsupported RTX streaming panels or debug overlays are presentation features.
- Promises exposed RTX denoiser or sampling settings that the current config does not provide.
