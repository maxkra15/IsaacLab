# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka task for inserting the free end of an already-connected RJ45 cable."""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.utils import math as math_utils

from .pick_insert_env import FrankaRJ45PickInsertEnv


class FrankaRJ45DualRackInsertEnv(FrankaRJ45PickInsertEnv):
    """Route one cable from a fixed lower-rack plug into an upper-rack socket."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        self._dual_rack_network_switch_presentations = ()
        super().__init__(cfg, render_mode, **kwargs)

        if self._should_create_newton_gl_workcell_presentation():
            presentations = []
            try:
                presentations.extend(self._create_newton_gl_workcell_presentations())
            except Exception:
                for presentation in presentations:
                    presentation.close()
                self.close()
                raise
            self._dual_rack_network_switch_presentations = tuple(presentations)

    def _should_create_newton_gl_workcell_presentation(self) -> bool:
        """Return whether the active viewer consumes marker-only decor."""
        from .network_switch_presentation import should_enable_newton_gl_marker_presentation

        return should_enable_newton_gl_marker_presentation(getattr(self, "sim", None))

    def _create_newton_gl_workcell_presentations(self) -> tuple[object, ...]:
        """Create the T-slot frame and both AS4610 marker fallbacks."""
        from .dual_rack_workcell import DUAL_RACK_ANCHORED_ACCENT_COLOR, DUAL_RACK_TARGET_ACCENT_COLOR
        from .dual_rack_workcell_presentation import NewtonGlDualRackWorkcellPresentation
        from .network_switch_presentation import NewtonGlNetworkSwitchPresentation

        return (
            NewtonGlDualRackWorkcellPresentation(self.sim, self.env_origins),
            NewtonGlNetworkSwitchPresentation(
                self.sim,
                self.socket_pose_e,
                prim_path="/Visuals/RJ45DualRack/TargetSwitch",
                active_port_color=DUAL_RACK_TARGET_ACCENT_COLOR,
            ),
            NewtonGlNetworkSwitchPresentation(
                self.sim,
                self.anchored_socket_pose_e,
                prim_path="/Visuals/RJ45DualRack/AnchoredSwitch",
                active_port_color=DUAL_RACK_ANCHORED_ACCENT_COLOR,
            ),
        )

    def close(self) -> None:
        """Release both NewtonGL switch groups before simulation teardown."""
        presentations = getattr(self, "_dual_rack_network_switch_presentations", ())
        try:
            for presentation in presentations:
                presentation.close()
        finally:
            self._dual_rack_network_switch_presentations = ()
            super().close()

    def _create_rj45_builder(self, cfg):
        """Build the dynamic target end plus the static workcell/source end."""
        from .physics import Rj45NewtonAssemblyBuilder
        from .pick_insert_env_cfg import pick_insert_topology_cfg

        return Rj45NewtonAssemblyBuilder(
            topology_cfg=pick_insert_topology_cfg(cfg),
            task_translation=cfg.task_translation,
            task_rotation_xyzw=cfg.task_rotation_xyzw,
            grasp_proxy_friction=cfg.grasp_proxy_friction,
            workcell_cfg=self._workcell_cfg(),
        )

    def _workcell_cfg(self):
        """Return the immutable dual-AS4610 physical workcell."""
        from .dual_rack_workcell import DUAL_RACK_WORKCELL_CFG

        return DUAL_RACK_WORKCELL_CFG

    def _bind_physics_state(self) -> None:
        """Bind the persisted free end and the non-persisted anchored geometry."""
        super()._bind_physics_state()
        runtime = self._ensure_rj45_runtime()
        if runtime.anchored_connector_q_w is None or runtime.anchored_cable_endpoint_w is None:
            raise RuntimeError("Dual-rack runtime is missing its static anchored connector geometry.")
        anchored = wp.to_torch(runtime.anchored_connector_q_w).to(device=self.device, dtype=torch.float32).clone()
        if tuple(anchored.shape) != (self.num_envs, 3, 7):
            raise RuntimeError(f"Anchored connector pose map must have shape ({self.num_envs}, 3, 7).")
        anchored[..., :3] -= self.env_origins[:, None, :]
        self._anchored_connector_pose_e = anchored
        target = wp.to_torch(runtime.anchored_cable_endpoint_w).to(device=self.device, dtype=torch.float32).clone()
        target -= self.env_origins
        self._anchored_cable_target_e = target
        self._cable_segment_lengths = wp.to_torch(runtime.cable_segment_lengths).to(
            device=self.device,
            dtype=torch.float32,
        )

    def _reset_dataset_task_contract(self) -> dict[str, object]:
        """Return the contract that binds the second connector and rack frame."""
        from .dual_rack_env_cfg import dual_rack_reset_dataset_task_contract

        return dual_rack_reset_dataset_task_contract(self.cfg)

    def anchored_socket_pose_e(self) -> torch.Tensor:
        """Static lower-rack socket pose in the environment frame."""
        return self._anchored_connector_pose_e[:, 0]

    def anchored_plug_pose_e(self) -> torch.Tensor:
        """Static seated cable-plug pose in the environment frame."""
        return self._anchored_connector_pose_e[:, 1]

    def anchored_latch_pose_e(self) -> torch.Tensor:
        """Static seated latch pose in the environment frame."""
        return self._anchored_connector_pose_e[:, 2]

    def anchored_cable_target_position_e(self) -> torch.Tensor:
        """Exact cable exit of the seated static plug [m]."""
        return self._anchored_cable_target_e

    def anchored_cable_endpoint_position_e(self) -> torch.Tensor:
        """Current positive-local-Z endpoint of the pinned final cable capsule."""
        cable_stop = int(self._task_layout.cable_body_slice.stop)
        tail = self.task_body_pose_e()[:, cable_stop - 1]
        half_length = 0.5 * self._cable_segment_lengths[-1]
        local_endpoint = torch.zeros((self.num_envs, 3), device=self.device, dtype=tail.dtype)
        local_endpoint[:, 2] = half_length
        return tail[:, :3] + math_utils.quat_apply(tail[:, 3:7], local_endpoint)


__all__ = ["FrankaRJ45DualRackInsertEnv"]
