# Examples

## Manager-based Cartpole policy video

Train the `Isaac-Cartpole` manager-based task with Newton physics. Training does not need the presentation renderer:

```bash
uv run isaaclab train --rl_library rsl_rl \
    --task Isaac-Cartpole \
    physics=newton_mjwarp
```

Then capture the trained policy through the same task entrypoint:

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

The shared play entrypoint maps the flags as follows:

- `--video` enables the task's environment recorder.
- `--video_length 240` captures 240 environment steps.
- `--video_interval 0` requests one clip at the start of play.
- `--viz newton_rtx` creates the OVRTX visualizer and makes the automatic recorder source `visualizer:newton_rtx`.
- `--num_envs 1 --max_visible_envs 1` produces a clean single-environment composition.
- `physics=newton_mjwarp` selects Newton physics; it does not select the visualizer.
- The three `env.sim.default_visualizer_cfg` overrides frame the Cartpole asset at its spawned height without changing the task source.

The MP4 is written below the resolved Cartpole log run in `videos/play`. MoviePy encoding is internal; do not add a second encoder script.

## Optional task-side recorder defaults

Normally the CLI flags are sufficient. Add a recorder to an environment config only when the task needs persistent settings that the shared flags do not expose, such as a descriptive filename, fixed FPS, or frame stride:

```python
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

env_cfg.video_recorders = [
    VideoRecorderCfg(
        source="visualizer:newton_rtx",
        video_length=240,
        video_interval=0,
        fps=30,
        output_dir="videos/policy-rollout",
        output_filename_prefix="cartpole",
    )
]
```

`--video_length` and `--video_interval` override those fields when supplied on the command line. The remaining task defaults are preserved.

## Studio-lit renderer configuration

```python
from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg

env_cfg.sim.default_visualizer_cfg = NewtonRTXVisualizerCfg(
    eye=(4.0, -5.0, 3.5),
    lookat=(0.0, 0.0, 2.5),
    focal_length=18.0,
    window_width=1920,
    window_height=1080,
    rtx_environment="studio",
    headless=True,
)
```

Using `default_visualizer_cfg` makes these settings hints for whichever visualizer the command selects; it does not force presentation rendering during ordinary training. Do not create a task-local `--video` argument: `isaaclab train` and `isaaclab play` already provide it.
