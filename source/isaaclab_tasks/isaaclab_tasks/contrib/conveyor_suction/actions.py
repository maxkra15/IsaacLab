# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A downward suction tool controlled by Cartesian residuals and a binary vacuum."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.optimize import least_squares

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..conveyor_franka.mdp.reset_events import franka_tool_position

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.envs import ManagerBasedEnv

TOOL_OFFSET = (0.0, 0.0, 0.105)
DOWN_QUATERNION = (0.9238795325, 0.3826834324, 0.0, 0.0)


class CartesianSuctionArm(DifferentialInverseKinematicsAction):
    """Reuse the standard six-dimensional IK solver with a fixed downward orientation."""

    def __init__(self, cfg: CartesianSuctionArmCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self.invalid_actions = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.previous_actions = torch.zeros_like(self._raw_actions)
        self._needs_reference = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._down = torch.tensor(DOWN_QUATERNION, device=self.device).expand(self.num_envs, -1)
        self._lower = torch.tensor((-0.45, -0.95, 0.045), device=self.device)
        self._upper = torch.tensor((1.0, 0.95, 0.75), device=self.device)
        self._shoulder = torch.tensor((0.0, 0.0, 0.333), device=self.device)

    @property
    def action_dim(self) -> int:
        return 3

    def process_actions(self, actions: torch.Tensor) -> None:
        self.invalid_actions[:] = ~torch.isfinite(actions).all(dim=-1)
        self.previous_actions.copy_(self._raw_actions)
        self._raw_actions[:] = torch.nan_to_num(actions).clamp(-1.0, 1.0)
        self._processed_actions[:] = self._raw_actions * self._scale
        position, rotation = self._compute_frame_pose()
        reference = torch.where(self._needs_reference[:, None], position, self._ik_controller.ee_pos_des)
        desired = (reference + self._processed_actions).clamp(min=self._lower, max=self._upper)
        displacement = desired - self._shoulder
        radius = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True).clamp_min(1.0e-6)
        desired = self._shoulder + displacement * (0.78 / radius).clamp(max=1.0)
        self._ik_controller.set_command(torch.cat((desired, self._down), dim=-1), position, rotation)
        self._needs_reference[:] = False

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        self.invalid_actions[slice(None) if env_ids is None else env_ids] = False
        self.previous_actions[slice(None) if env_ids is None else env_ids] = 0.0
        self._needs_reference[slice(None) if env_ids is None else env_ids] = True


@configclass
class CartesianSuctionArmCfg(DifferentialInverseKinematicsActionCfg):
    """Three Cartesian residuals, with orientation constrained by the tool."""

    class_type: type = CartesianSuctionArm


class VacuumAction(ActionTerm):
    """Signed binary command: negative seals, nonnegative releases."""

    def __init__(self, cfg: VacuumActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self.previous_actions = torch.zeros_like(self._raw_actions)
        self.invalid_actions = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self.invalid_actions[:] = ~torch.isfinite(actions).all(dim=-1)
        self.previous_actions.copy_(self._raw_actions)
        self._raw_actions[:] = torch.nan_to_num(actions).clamp(-1.0, 1.0)
        self._processed_actions[:] = torch.where(self._raw_actions < 0.0, -1.0, 1.0)

    def apply_actions(self) -> None:
        self._env.gripper.set_command(self._processed_actions[:, 0] < 0.0, self._env.active_ids)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0.0
        self._processed_actions[ids] = 1.0
        self.invalid_actions[ids] = False
        self.previous_actions[ids] = 0.0


@configclass
class VacuumActionCfg(ActionTermCfg):
    """One binary action for a Newton suction tool."""

    class_type: type = VacuumAction
    asset_name: str = "robot"


def solve_tool_ik(positions: torch.Tensor) -> torch.Tensor:
    """Generate collision-screenable downward Panda reset poses at startup, on the CPU."""
    targets = positions.detach().cpu().numpy()
    solutions = []
    for target in targets:
        angle = math.atan2(float(target[1]), float(target[0]))

        def joints(values: np.ndarray) -> torch.Tensor:
            shoulder, elbow = values
            return torch.tensor((angle, shoulder, 0.0, elbow, 0.0, shoulder - elbow, angle), dtype=torch.float64)

        def error(values: np.ndarray) -> np.ndarray:
            current = franka_tool_position(joints(values).unsqueeze(0), tool_offset_z=0.212)[0]
            return current.numpy() - target

        fit = least_squares(error, (0.35, -2.0), bounds=((-1.7, -3.0), (1.7, -0.15)), max_nfev=80)
        if np.linalg.norm(fit.fun) > 0.003:
            raise ValueError(f"Unreachable suction reset position {target.tolist()}: {fit.fun.tolist()}")
        solutions.append(joints(fit.x).to(dtype=positions.dtype))
    return torch.stack(solutions).to(device=positions.device)


def spawn_suction_franka(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Keep the existing arm asset and author a compact, hand-mounted suction tool."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation)
    sim_utils.make_uninstanceable(str(prim.GetPath()))
    hand = next(p for p in Usd.PrimRange(prim) if p.GetName() == "panda_hand")
    for child in Usd.PrimRange(prim):
        if "finger" in child.GetPath().pathString:
            if child.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(child).MakeInvisible()
            if child.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Set(False)
    tool = str(hand.GetPath()) + "/SuctionTool"
    for name, position, size, color in (
        ("Mount", (0.0, 0.0, 0.065), (0.035, 0.035, 0.055), (0.18, 0.20, 0.23)),
        ("Plate", (0.0, 0.0, 0.096), (0.028, 0.028, 0.005), (0.62, 0.67, 0.72)),
    ):
        geometry = sim_utils.CuboidCfg(size=size, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))
        geometry.func(f"{tool}/{name}", geometry, translation=position)
    for index, (x, y) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        cup = UsdGeom.Cylinder.Define(prim.GetStage(), f"{tool}/Cup{index}")
        cup.CreateRadiusAttr(0.004)
        cup.CreateHeightAttr(0.006)
        cup.AddTranslateOp().Set(Gf.Vec3d(x * 0.007, y * 0.007, 0.102))
        cup.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.55, 0.65)])
        UsdPhysics.CollisionAPI.Apply(cup.GetPrim()).CreateCollisionEnabledAttr(True)
    return prim
