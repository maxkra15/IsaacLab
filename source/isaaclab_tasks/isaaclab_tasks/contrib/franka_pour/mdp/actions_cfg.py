# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Franka Pour action terms."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .actions import (
        CurriculumGripperPositionAction,
        EMARelativeJointPositionAction,
    )


@configclass
class EMARelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for :class:`EMARelativeJointPositionAction`."""

    alpha: float = 1.0
    """Weight of the newest relative joint delta; one preserves the unfiltered action."""

    class_type: type[EMARelativeJointPositionAction] | str = "{DIR}.actions:EMARelativeJointPositionAction"


@configclass
class CurriculumGripperPositionActionCfg(ActionTermCfg):
    """Configuration for :class:`CurriculumGripperPositionAction`."""

    joint_names: list[str] = MISSING
    scale: float = 0.04
    """Per-finger residual delta per policy-action unit [m]; unused in binary mode."""
    alpha: float = 0.2
    """Interpolation weight applied to the selected finger target."""
    binary_threshold: float | None = None
    """Optional threshold selecting filtered close/maximum targets; values below it close."""
    close_position: float = 0.0
    neutral_position: float = 0.025
    """Largest per-finger command accepted from the action [m]."""
    default_position: float | None = None
    """Per-finger residual-mode zero command and initial target [m]. ``None`` uses ``close_position``."""
    contact_min_deflection: float = 0.001
    """Minimum settled position-drive deflection required on each finger [m]."""
    class_type: type[CurriculumGripperPositionAction] | str = "{DIR}.actions:CurriculumGripperPositionAction"
