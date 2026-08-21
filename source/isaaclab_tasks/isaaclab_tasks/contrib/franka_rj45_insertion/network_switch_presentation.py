# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Viewer-only network-switch presentation for the RJ45 pick-and-insert task."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.visualization_markers_cfg import VisualizationMarkersCfg
from isaaclab.utils import math as math_utils

_CHASSIS_MARKER = 0
_PORT_MARKER = 1
_ACTIVE_PORT_MARKER = 2

# Bounds of NVIDIA's AS4610 after the transform in
# ``rj45_assembly._presentation_switch_transform``. The selected port is the
# local origin, so these boxes share the exact socket-relative frame used by Kit.
_SWITCH_MIN = (-0.21826283, -0.01146542, -0.02488871)
_SWITCH_MAX = (0.26810951, 0.35749699, 0.01873206)
_PANEL_THICKNESS = 0.004


@dataclass(frozen=True)
class _PresentationBox:
    """One box in the socket-local presentation model."""

    marker_index: int
    center: tuple[float, float, float]
    size: tuple[float, float, float]


def _network_switch_boxes() -> tuple[_PresentationBox, ...]:
    """Build a lightweight AS4610-like shell, front ports, and active-port frame."""
    lo_x, lo_y, lo_z = _SWITCH_MIN
    hi_x, hi_y, hi_z = _SWITCH_MAX
    width, depth, height = hi_x - lo_x, hi_y - lo_y, hi_z - lo_z
    center_x, center_y, center_z = (lo_x + hi_x) * 0.5, (lo_y + hi_y) * 0.5, (lo_z + hi_z) * 0.5
    thickness = _PANEL_THICKNESS

    # A shell, rather than one solid cuboid, keeps the active socket and insertion
    # corridor visually open while retaining the source asset's outer dimensions.
    boxes = [
        _PresentationBox(_CHASSIS_MARKER, (lo_x + 0.5 * thickness, center_y, center_z), (thickness, depth, height)),
        _PresentationBox(_CHASSIS_MARKER, (hi_x - 0.5 * thickness, center_y, center_z), (thickness, depth, height)),
        _PresentationBox(_CHASSIS_MARKER, (center_x, center_y, lo_z + 0.5 * thickness), (width, depth, thickness)),
        _PresentationBox(_CHASSIS_MARKER, (center_x, center_y, hi_z - 0.5 * thickness), (width, depth, thickness)),
        _PresentationBox(
            _CHASSIS_MARKER,
            (center_x, hi_y - 0.5 * thickness, center_z),
            (width - 2.0 * thickness, thickness, height - 2.0 * thickness),
        ),
        _PresentationBox(
            _CHASSIS_MARKER,
            (center_x, lo_y + 0.5 * thickness, lo_z + 0.003),
            (width - 2.0 * thickness, thickness, 0.006),
        ),
        _PresentationBox(
            _CHASSIS_MARKER,
            (center_x, lo_y + 0.5 * thickness, hi_z - 0.003),
            (width - 2.0 * thickness, thickness, 0.006),
        ),
    ]

    # Forty-seven dark neighbors plus the live socket make the characteristic
    # two-row, 48-port front. Upper-row column 11 is the active opening at (0, 0, 0).
    port_pitch = 0.016
    port_size = (0.0135, 0.002, 0.0105)
    for row, port_z in enumerate((0.0, -0.014)):
        for column in range(24):
            if row == 0 and column == 11:
                continue
            boxes.append(
                _PresentationBox(
                    _PORT_MARKER,
                    ((column - 11) * port_pitch, -0.0125, port_z),
                    port_size,
                )
            )

    # Match the Kit presentation's cyan frame without filling the socket mouth.
    for center, size in (
        ((-0.0101, -0.0133, 0.0), (0.0015, 0.0012, 0.0175)),
        ((0.0101, -0.0133, 0.0), (0.0015, 0.0012, 0.0175)),
        ((0.0, -0.0133, -0.0081), (0.0217, 0.0012, 0.0015)),
        ((0.0, -0.0133, 0.0081), (0.0217, 0.0012, 0.0015)),
    ):
        boxes.append(_PresentationBox(_ACTIVE_PORT_MARKER, center, size))
    return tuple(boxes)


_NETWORK_SWITCH_BOXES = _network_switch_boxes()


def should_enable_newton_gl_network_switch(sim, *, reset_source: str) -> bool:
    """Return whether a standalone Newton GL play session needs the marker proxy."""
    if sim is None or reset_source != "procedural":
        return False
    needs_kit_backend = (
        sim.has_gui
        or bool(sim.get_setting("/isaaclab/render/rtx_sensors"))
        or bool(sim.get_setting("/isaaclab/xr/enabled"))
        or sim.has_offscreen_render
        or any(
            visualizer.supports_markers() and visualizer.pumps_app_update() and visualizer.cfg.enable_markers
            for visualizer in sim.visualizers
        )
    )
    has_newton_gl_backend = any(
        getattr(visualizer.cfg, "visualizer_type", None) == "newton_gl"
        and visualizer.supports_markers()
        and not visualizer.pumps_app_update()
        and visualizer.cfg.enable_markers
        for visualizer in sim.visualizers
    )
    return has_newton_gl_backend and not needs_kit_backend


def _network_switch_marker_cfg() -> VisualizationMarkersCfg:
    """Return shared marker prototypes for the Newton GL presentation."""

    def _box(color: tuple[float, float, float]) -> sim_utils.CuboidCfg:
        return sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )

    return VisualizationMarkersCfg(
        prim_path="/Visuals/RJ45PickInsert/NetworkSwitch",
        markers={
            "chassis": _box((0.22, 0.25, 0.29)),
            "ports": _box((0.018, 0.026, 0.04)),
            "active_port": _box((0.04, 0.65, 0.95)),
        },
    )


def _network_switch_marker_state(
    socket_pose_e: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform socket-local presentation boxes into each environment frame."""
    if socket_pose_e.ndim != 2 or socket_pose_e.shape[1] != 7:
        raise ValueError(f"Expected socket poses with shape (N, 7), got {tuple(socket_pose_e.shape)}.")
    num_envs = socket_pose_e.shape[0]
    device = socket_pose_e.device
    dtype = socket_pose_e.dtype
    local_centers = torch.tensor([box.center for box in _NETWORK_SWITCH_BOXES], device=device, dtype=dtype)
    scales = torch.tensor([box.size for box in _NETWORK_SWITCH_BOXES], device=device, dtype=dtype)
    marker_indices = torch.tensor(
        [box.marker_index for box in _NETWORK_SWITCH_BOXES],
        device=device,
        dtype=torch.int32,
    )
    box_count = local_centers.shape[0]

    socket_position = socket_pose_e[:, None, :3]
    socket_orientation = socket_pose_e[:, None, 3:7].expand(-1, box_count, -1)
    local_centers = local_centers[None].expand(num_envs, -1, -1)
    translations = socket_position + math_utils.quat_apply(
        socket_orientation.reshape(-1, 4), local_centers.reshape(-1, 3)
    ).reshape(num_envs, box_count, 3)
    return (
        translations.flatten(0, 1),
        socket_orientation.flatten(0, 1),
        scales[None].expand(num_envs, -1, -1).flatten(0, 1),
        marker_indices[None].expand(num_envs, -1).flatten(),
        torch.arange(num_envs, device=device, dtype=torch.int32).repeat_interleave(box_count),
    )


class NewtonGlNetworkSwitchPresentation:
    """Render a socket-following network switch through Isaac Lab's marker API."""

    def __init__(self, sim, socket_pose_provider: Callable[[], torch.Tensor]):
        self._sim = sim
        self._socket_pose_provider = socket_pose_provider
        self._marker = VisualizationMarkers(_network_switch_marker_cfg())
        self._callback_id = f"rj45_pick_insert_network_switch:{id(self)}"
        self._closed = False
        self._sim.vis_marker_registry.add_callback(self._callback_id, self._update)
        self._update()

    def _update(self, _event=None) -> None:
        """Update marker instances immediately before the active viewer renders."""
        if self._closed:
            return
        translations, orientations, scales, marker_indices, environment_ids = _network_switch_marker_state(
            self._socket_pose_provider()
        )
        self._marker.visualize(
            translations=translations,
            orientations=orientations,
            scales=scales,
            marker_indices=marker_indices,
            environment_ids=environment_ids,
        )

    def close(self) -> None:
        """Remove the render callback and marker group; safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        self._sim.vis_marker_registry.remove_callback(self._callback_id)
        marker = self._marker
        self._marker = None
        marker.set_visibility(False)
