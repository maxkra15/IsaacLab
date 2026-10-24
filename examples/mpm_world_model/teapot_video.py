# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in native teapot video using the existing Newton GL framebuffer recorder."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from .recording import atomic_json


def configure_capture_environment(enabled: bool) -> None:
    """Select EGL before Newton imports windows, including on hosts with DISPLAY."""
    if not enabled:
        return
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyglet

    # Pyglet's headless option selects EGL; a hidden X window alone does not.
    # CUDA-to-EGL device-index mappings are intentionally not inferred.
    pyglet.options["headless"] = True


class NativeTeapotVideo:
    """Capture actual native physics frames, including the scripted grasp prefix.

    The recorder sees only the current rendered solver state. A learned actor
    supplies residual controls; this class never predicts particle positions.
    """

    def __init__(self, path, fps, native_steps, sim, scene, surface_renderer, provenance):
        from isaaclab.envs.utils.video_recorder import VideoRecorder
        from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

        self.path = Path(path)
        self.sidecar = self.path.with_suffix(".mp4.json")
        if self.path.exists() or self.sidecar.exists():
            raise FileExistsError(f"Refusing to overwrite native capture: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(prefix=f".{self.path.stem}-", dir=self.path.parent)
        self._staged = Path(self._temporary.name) / "native_0000.mp4"
        self._recorder = VideoRecorder(
            VideoRecorderCfg(
                source="viz:newton_gl",
                output_dir=self._temporary.name,
                output_filename_prefix="native",
                video_length=native_steps,
                frame_stride=800 // fps,
                fps=fps,
            ),
            SimpleNamespace(sim=sim, scene=scene, step_dt=sim.get_physics_dt()),
        )
        self.sim, self.surface_renderer = sim, surface_renderer
        self.stride, self.steps, self.fps = 800 // fps, 0, fps
        self.requested_native_steps = native_steps
        self.provenance = provenance
        self.closed = False

    def step(self) -> None:
        """Run once after each real800Hz physics endpoint, before its data capture."""
        self.steps += 1
        if self.steps % self.stride == 0:
            if self.surface_renderer is not None:
                self.surface_renderer.update()
            # Headless Newton GL renders on demand; is_rendering can be False
            # even though a framebuffer consumer is explicitly configured.
            self.sim.render()
        self._recorder.step()

    def close(self, outcome: dict | None = None) -> None:
        """Publish only a nonempty encoded file, retaining outcomes without success filtering."""
        if self.closed:
            return
        self.closed = True
        self._recorder.close()
        metadata = {
            **self.provenance,
            "native_steps_captured": self.steps,
            "requested_native_steps": self.requested_native_steps,
            "fps": self.fps,
            "expected_frame_count": self.steps // self.stride,
            "physical_duration_s": self.steps / 800.0,
            "capture_source": "current_native_solver_state_rendered_by_NewtonGL",
            "outcome": outcome,
            "outcome_recorded": outcome is not None,
            "full_requested_capture": self.steps == self.requested_native_steps,
            "full_task_horizon": self.steps == 14400,
        }
        try:
            # VideoRecorder logs encoding errors internally. Turn a missing
            # artifact into an explicit failure rather than claiming a video.
            if not self._staged.is_file() or self._staged.stat().st_size == 0:
                metadata["capture_error"] = "Native recorder produced no nonempty MP4."
                atomic_json(self.sidecar, metadata)
                raise RuntimeError(f"Native video encoding failed: {self.path}")
            self._staged.replace(self.path)
            metadata["video_path"] = str(self.path.resolve())
            metadata["video_bytes"] = self.path.stat().st_size
            atomic_json(self.sidecar, metadata)
        finally:
            self._temporary.cleanup()
