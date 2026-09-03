# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone environment wrapper for scoped cable construction."""

from __future__ import annotations

from isaaclab_newton.cloner import newton_builder_world_hook

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.version import has_kit


class RizonSharpaCableEnv(ManagerBasedRLEnv):
    """Teleoperate one MJWarp Sharpa hand against one VBD hanging cable."""

    def _init_sim(self) -> None:
        # Import Newton runtime code only after the launcher has created Kit (when requested).
        from .cable import RizonSharpaCableBuilderExtension

        extension = RizonSharpaCableBuilderExtension(self.cfg.scene.cable)
        self._cable_builder_extension = extension
        with newton_builder_world_hook(extension.add_to_builder):
            super()._init_sim()

        # Add the large SimReady payloads only after Newton has finalized the
        # coupled model and the isolated IK prototype. The showroom is purely
        # presentational and must never enter either physics topology.
        if has_kit():
            from .showroom import author_rizon_sharpa_showroom

            with sim_utils.use_stage(self.sim.stage):
                author_rizon_sharpa_showroom(self.sim.stage)


__all__ = ["RizonSharpaCableEnv"]
