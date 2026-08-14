# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Juggle-specific reset-preload action specialization."""

from __future__ import annotations

import torch

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.stack.mdp.actions import ResetPreservingRelativeJointPositionAction
from isaaclab_tasks.contrib.stack.mdp.actions_cfg import ResetPreservingRelativeJointPositionActionCfg

from .reset import JugglePhase
from .runtime import get_juggle_runtime_state


class JuggleResetPreservingRelativeJointPositionAction(ResetPreservingRelativeJointPositionAction):
    """Anchor the sphere cradle until the policy deliberately opens the hand."""

    cfg: JuggleResetPreservingRelativeJointPositionActionCfg

    def _reset_grasp_pair_ids(self) -> torch.Tensor:
        """Return the one sphere-cradle calibration ID for every environment."""
        state = get_juggle_runtime_state(self._env)
        return torch.zeros_like(state.current_phases)

    def _reset_preload_active_mask(self) -> torch.Tensor:
        """Protect reset-authored states in which the sphere starts cradled."""
        phase = get_juggle_runtime_state(self._env).current_phases
        return (phase == int(JugglePhase.HELD_PRETHROW)) | (phase == int(JugglePhase.STABLE_CATCH))


@configclass
class JuggleResetPreservingRelativeJointPositionActionCfg(ResetPreservingRelativeJointPositionActionCfg):
    """Configuration for the sphere-cradle preload handoff."""

    class_type: type[JuggleResetPreservingRelativeJointPositionAction] | str = (
        "{DIR}.actions:JuggleResetPreservingRelativeJointPositionAction"
    )
