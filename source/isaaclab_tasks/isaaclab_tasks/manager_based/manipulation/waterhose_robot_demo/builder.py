# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local construction entry point for the Newton waterhose simulation."""

from __future__ import annotations


def create_simulation(viewer, args, preloaded_vbd_scene=None):
    """Create the local Newton simulation."""
    from .simulation import WaterhoseRobotDemoSimulation  # noqa: PLC0415

    return WaterhoseRobotDemoSimulation(viewer, args, preloaded_vbd_scene=preloaded_vbd_scene)
