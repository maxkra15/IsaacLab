# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton differential-IK action term for the scoop bowl: position (x,y,z) + pitch.

The Panda fingers are fixed open by the environment and are not action controlled.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from ..scoop_env import FrankaScoopEnv


class ScoopAction(ActionTerm):
    """4-DoF action: scoop-bowl position delta (x,y,z) + bowl pitch delta (tilt to scoop/pour)."""

    cfg: ScoopActionCfg
    _env: FrankaScoopEnv

    def __init__(self, cfg: ScoopActionCfg, env: FrankaScoopEnv):
        super().__init__(cfg, env)
        self._dim = 4
        self._raw = torch.zeros(env.num_envs, self._dim, device=env.device)
        self._proc = torch.zeros_like(self._raw)
        self._joint_targets = env._default_arm_q.clone()

    @property
    def action_dim(self) -> int:
        return self._dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._proc

    def process_actions(self, actions: torch.Tensor) -> None:
        env = self._env
        a = torch.nan_to_num(actions, nan=0.0).clamp(-1.0, 1.0)
        self._raw[:] = a
        self._proc[:] = torch.lerp(self._proc, a, env.cfg.action_smoothing)
        moving = torch.any(torch.abs(self._proc) > 1.0e-6, dim=1)
        if not bool(moving.any()):
            env._target_bowl_e[:] = torch.clamp(env.bowl_pos_e(), env._ws_lo, env._ws_hi)
            return
        if not bool(moving.all()):
            env._target_bowl_e[~moving] = torch.clamp(env.bowl_pos_e()[~moving], env._ws_lo, env._ws_hi)
        dt = env.step_dt
        env._target_bowl_e[moving] = torch.clamp(
            env._target_bowl_e[moving] + self._proc[moving, :3] * env.cfg.cartesian_action_scale * dt,
            env._ws_lo,
            env._ws_hi,
        )
        env._pitch[moving] = torch.clamp(
            env._pitch[moving] + self._proc[moving, 3] * env.cfg.pitch_action_scale * dt,
            env.cfg.min_pitch,
            env.cfg.max_pitch,
        )
        joint_targets = env.solve_arm_ik()
        if str(env.cfg.ik_backend).lower() == "diffik":
            max_delta = float(env.cfg.diffik_max_delta)
        else:
            max_delta = float(env.cfg.max_ik_delta)
        if max_delta > 0.0:
            delta = torch.clamp(joint_targets - self._joint_targets, -max_delta, max_delta)
            joint_targets = torch.clamp(self._joint_targets + delta, env._arm_lo, env._arm_hi)
        self._joint_targets[moving] = joint_targets[moving]

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target_index(
            target=self._joint_targets,
            joint_ids=self._env._arm_joint_ids,
        )
        self._env.hold_gripper_open_targets()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw[env_ids] = 0.0
        self._proc[env_ids] = 0.0
        self._joint_targets[env_ids] = self._env._reset_arm_q[env_ids]


@configclass
class ScoopActionCfg(ActionTermCfg):
    class_type: type = ScoopAction
    asset_name: str = "robot"  # unused; the term reads Newton state directly
