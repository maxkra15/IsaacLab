# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NewtonGL-only T-slot presentation for the dual-rack workcell."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.visualization_markers_cfg import VisualizationMarkersCfg

from .dual_rack_workcell import DUAL_RACK_FRAME_COLOR, DUAL_RACK_WORKCELL_CFG


def _t_slot_marker_cfg() -> VisualizationMarkersCfg:
    """Return one unit-cuboid prototype shared by all extrusion pieces."""
    return VisualizationMarkersCfg(
        prim_path="/Visuals/RJ45DualRack/TSlotFrame",
        markers={
            "anodized_aluminum": sim_utils.CuboidCfg(
                size=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=DUAL_RACK_FRAME_COLOR,
                    metallic=0.72,
                    roughness=0.24,
                ),
            )
        },
    )


def dual_rack_t_slot_marker_state(
    env_origins: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return dense marker state for every presentation-only extrusion piece."""
    origins = torch.as_tensor(env_origins)
    if origins.ndim != 2 or origins.shape[1] != 3 or not origins.dtype.is_floating_point:
        raise ValueError(f"env_origins must be floating point with shape (N, 3), got {origins.shape}.")
    boxes = tuple(box for box in DUAL_RACK_WORKCELL_CFG.boxes if box.visible)
    if not boxes:
        raise RuntimeError("Dual-rack T-slot presentation contains no visible boxes.")
    device, dtype = origins.device, origins.dtype
    centers = torch.tensor([box.center_m for box in boxes], device=device, dtype=dtype)
    scales = torch.tensor([box.size_m for box in boxes], device=device, dtype=dtype)
    count = len(boxes)
    num_envs = len(origins)
    translations = origins[:, None] + centers[None]
    orientations = torch.zeros((num_envs, count, 4), device=device, dtype=dtype)
    orientations[..., 3] = 1.0
    return (
        translations.flatten(0, 1),
        orientations.flatten(0, 1),
        scales[None].expand(num_envs, -1, -1).flatten(0, 1),
        torch.zeros(num_envs * count, device=device, dtype=torch.int32),
        torch.arange(num_envs, device=device, dtype=torch.int32).repeat_interleave(count),
    )


class NewtonGlDualRackWorkcellPresentation:
    """Render true-profile T-slot pieces without adding Newton collision shapes."""

    def __init__(self, sim, env_origins: torch.Tensor):
        self._sim = sim
        self._env_origins = env_origins
        self._marker = VisualizationMarkers(_t_slot_marker_cfg())
        self._callback_id = f"rj45_dual_rack_t_slot:{id(self)}"
        self._closed = False
        self._sim.vis_marker_registry.add_callback(self._callback_id, self._update)
        self._update()

    def _update(self, _event=None) -> None:
        """Refresh all environment-local extrusion pieces before rendering."""
        if self._closed:
            return
        translations, orientations, scales, marker_indices, environment_ids = dual_rack_t_slot_marker_state(
            self._env_origins
        )
        self._marker.visualize(
            translations=translations,
            orientations=orientations,
            scales=scales,
            marker_indices=marker_indices,
            environment_ids=environment_ids,
        )

    def close(self) -> None:
        """Remove the callback and marker group; safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        self._sim.vis_marker_registry.remove_callback(self._callback_id)
        marker = self._marker
        self._marker = None
        marker.set_visibility(False)


__all__ = ["NewtonGlDualRackWorkcellPresentation", "dual_rack_t_slot_marker_state"]
