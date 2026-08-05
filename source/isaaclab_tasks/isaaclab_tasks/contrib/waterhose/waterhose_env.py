# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment wrappers that scope Waterhose's native cable builder extension."""

from __future__ import annotations

from isaaclab_newton.cloner import newton_builder_world_hook

from isaaclab.envs import ManagerBasedRLEnv

from .cable import WaterhoseCableBuilderExtension, WaterhoseCableObject


class WaterhoseCableEnvMixin:
    """Install task-specific cable additions only while this scene is replicated."""

    def _init_sim(self) -> None:
        extension = WaterhoseCableBuilderExtension(self.cfg.scene.cable1)
        self._waterhose_cable_builder_extension = extension
        with newton_builder_world_hook(extension.add_to_builder):
            super()._init_sim()

        cable = self.scene["cable1"]
        if not isinstance(cable, WaterhoseCableObject):
            raise TypeError(f"Expected WaterhoseCableObject, got {type(cable).__name__}.")
        cable.bind_builder_extension(extension)


class WaterhoseRLEnv(WaterhoseCableEnvMixin, ManagerBasedRLEnv):
    """Manager-based Waterhose environment with scoped native-cable construction."""
