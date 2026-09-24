# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visual set dressing for the coupled MJWarp–MPM Franka pouring task."""

from __future__ import annotations

from pathlib import Path

from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import FrankaPourResetDatasetEnvCfg, PourSceneCfg

_BACKDROP_USD = Path(__file__).parent / "assets" / "foundry_backdrop.usda"


@configclass
class KineticFoundrySceneCfg(PourSceneCfg):
    """Pouring scene with a visual-only foundry backdrop behind the SeattleLab table."""

    backdrop = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/FoundryBackdrop",
        spawn=UsdFileCfg(usd_path=str(_BACKDROP_USD)),
    )


@configclass
class KineticFoundryEnvCfg(FrankaPourResetDatasetEnvCfg):
    """Manager-based task with trainable PPO actions and coupled granular media."""

    scene: KineticFoundrySceneCfg = KineticFoundrySceneCfg(num_envs=2, env_spacing=2.5, replicate_physics=True)

    def __post_init__(self) -> None:
        """Keep the parent task physics and show its particles in the Newton viewer."""
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg  # noqa: PLC0415

        super().__post_init__()
        self.sim.default_visualizer_cfg = NewtonGLVisualizerCfg(
            eye=(0.95, -0.72, 0.55),
            lookat=(0.48, -0.04, 0.18),
            focal_length=22.0,
            window_width=960,
            window_height=540,
            show_particles=True,
            particle_color=(0.97, 0.61, 0.22),
        )
