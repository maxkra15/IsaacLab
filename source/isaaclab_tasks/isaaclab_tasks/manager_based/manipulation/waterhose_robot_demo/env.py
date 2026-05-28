# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment wrapper for the waterhose robot demo task."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv

from .manager import WaterhoseNewtonSolverCfg


def sync_waterhose_solver_cfg(cfg) -> None:
    """Keep the local Newton solver config aligned with IsaacLab scene overrides."""

    solver_cfg = getattr(getattr(getattr(cfg, "sim", None), "physics", None), "solver_cfg", None)
    if not isinstance(solver_cfg, WaterhoseNewtonSolverCfg):
        return

    scene_cfg = getattr(cfg, "scene", None)
    if scene_cfg is not None:
        solver_cfg.num_envs = int(getattr(scene_cfg, "num_envs", solver_cfg.num_envs))
        solver_cfg.env_spacing = float(getattr(scene_cfg, "env_spacing", solver_cfg.env_spacing))

    if hasattr(cfg, "max_demo_steps"):
        solver_cfg.max_demo_steps = int(getattr(cfg, "max_demo_steps"))


class WaterhoseRobotDemoEnv(ManagerBasedRLEnv):
    """Manager-based IsaacLab environment for the stable waterhose robot demo."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        sync_waterhose_solver_cfg(cfg)
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
