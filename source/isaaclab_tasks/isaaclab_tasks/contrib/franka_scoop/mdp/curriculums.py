# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Staged dump-first curriculum for the scoop transfer task.

Starts trivially easy — the cup resets PRE-LOADED with media, opening-up, directly
above the target box (``curriculum_reset_pose``/``curriculum_cup_fill_count``), so a
single tilt delivers and the policy experiences success within a few steps — and
advances stage only after the rolling success-rate EMA clears a threshold. Later
stages move the start to a loaded hover (carry + dump) and finally to the empty cup
at the pile (full scoop->carry->dump), while the delivered-particle requirement
(``curriculum_target_count`` -> ``env.scoop_target_count``, consumed by the
``delivered`` termination and success bonus) and the pile randomization ramp up.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
    from ..scoop_env import FrankaScoopEnv


class ScoopCurriculum(ManagerTermBase):
    """Advance a global difficulty stage based on a success-rate EMA."""

    def __init__(self, cfg: CurriculumTermCfg, env: FrankaScoopEnv):
        super().__init__(cfg, env)
        self.max_stage = len(env.cfg.curriculum_target_count) - 1
        self.stage = int(min(max(int(env.cfg.curriculum_start_stage), 0), self.max_stage))
        self.success_ema = 0.0
        self.resets_in_stage = 0
        self._apply(env)

    def _apply(self, env: FrankaScoopEnv) -> None:
        if hasattr(env, "curriculum_stage"):
            env.curriculum_stage[:] = self.stage
        env.scoop_target_count = float(env.cfg.curriculum_target_count[self.stage])

    def __call__(self, env: FrankaScoopEnv, env_ids: Sequence[int]):
        cfg = env.cfg
        succeeded = env.episode_succeeded[env_ids]
        n = int(succeeded.numel())
        if n > 0:
            rate = float(succeeded.float().mean().item())
            a = float(cfg.curriculum_success_ema_alpha)
            self.success_ema = (1.0 - a) * self.success_ema + a * rate
            self.resets_in_stage += n
            if (
                not cfg.curriculum_freeze
                and self.stage < self.max_stage
                and self.resets_in_stage >= cfg.curriculum_min_resets_per_stage
                and self.success_ema >= cfg.curriculum_success_threshold
            ):
                self.stage += 1
                self.success_ema = 0.0
                self.resets_in_stage = 0
        self._apply(env)
        out = {
            "stage": float(self.stage),
            "success_ema": float(self.success_ema),
            "target_count": float(env.scoop_target_count),
        }
        if n > 0:
            out["max_in_target"] = float(env.ep_max_in_target[env_ids].mean().item())
            out["max_in_bowl"] = float(env.ep_max_in_bowl[env_ids].mean().item())
        return out
