# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka two-ended-cable task registered to native ports on a SimReady GB300 rack."""

from __future__ import annotations

from .dual_rack_env import FrankaRJ45DualRackInsertEnv


class FrankaRJ45Gb300InsertEnv(FrankaRJ45DualRackInsertEnv):
    """Insert the free RJ45 end into one selected native GB300 SN2201 jack."""

    def _workcell_cfg(self):
        from .gb300_workcell import GB300_WORKCELL_CFG

        return GB300_WORKCELL_CFG

    def _create_newton_gl_workcell_presentations(self) -> tuple[object, ...]:
        from .gb300_presentation import NewtonGlGb300WorkcellPresentation

        return (NewtonGlGb300WorkcellPresentation(self.sim, self.env_origins),)

    def _reset_dataset_task_contract(self) -> dict[str, object]:
        from .gb300_env_cfg import gb300_reset_dataset_task_contract

        return gb300_reset_dataset_task_contract(self.cfg)


__all__ = ["FrankaRJ45Gb300InsertEnv"]
