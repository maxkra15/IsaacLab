# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.contrib.franka_rj45_insertion import network_switch_presentation as presentation


def _contains(box: presentation._PresentationBox, point: tuple[float, float, float]) -> bool:
    return all(abs(value - center) <= 0.5 * size for value, center, size in zip(point, box.center, box.size))


def test_network_switch_boxes_match_source_bounds_and_leave_active_socket_open():
    boxes = presentation._NETWORK_SWITCH_BOXES
    assert sum(box.marker_index == presentation._CHASSIS_MARKER for box in boxes) == 7
    assert sum(box.marker_index == presentation._PORT_MARKER for box in boxes) == 47
    assert sum(box.marker_index == presentation._ACTIVE_PORT_MARKER for box in boxes) == 4
    assert not any(_contains(box, (0.0, -0.0125, 0.0)) for box in boxes)

    chassis = [box for box in boxes if box.marker_index == presentation._CHASSIS_MARKER]
    actual_min = tuple(min(box.center[axis] - 0.5 * box.size[axis] for box in chassis) for axis in range(3))
    actual_max = tuple(max(box.center[axis] + 0.5 * box.size[axis] for box in chassis) for axis in range(3))
    assert actual_min == pytest.approx(presentation._SWITCH_MIN)
    assert actual_max == pytest.approx(presentation._SWITCH_MAX)

    for marker_cfg in presentation._network_switch_marker_cfg().markers.values():
        assert marker_cfg.rigid_props is None
        assert marker_cfg.mass_props is None
        assert marker_cfg.collision_props is None
        assert marker_cfg.physics_material is None


def test_network_switch_marker_state_is_dense_environment_major_and_follows_socket_pose():
    half_sqrt_two = 2.0**-0.5
    socket_pose = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.4, -0.3, 0.2, 0.0, 0.0, half_sqrt_two, half_sqrt_two],
        ],
        dtype=torch.float32,
    )

    translations, orientations, scales, marker_indices, environment_ids = presentation._network_switch_marker_state(
        socket_pose
    )

    box_count = len(presentation._NETWORK_SWITCH_BOXES)
    assert translations.shape == (2 * box_count, 3)
    assert orientations.shape == (2 * box_count, 4)
    assert scales.shape == (2 * box_count, 3)
    assert torch.equal(environment_ids, torch.arange(2, dtype=torch.int32).repeat_interleave(box_count))
    assert torch.equal(marker_indices[:box_count], marker_indices[box_count:])
    assert torch.allclose(orientations[:box_count], socket_pose[0, 3:7].expand(box_count, -1))
    assert torch.allclose(orientations[box_count:], socket_pose[1, 3:7].expand(box_count, -1))

    local = torch.tensor(presentation._NETWORK_SWITCH_BOXES[0].center)
    assert torch.allclose(translations[0], local)
    expected_rotated = torch.tensor([-local[1], local[0], local[2]]) + socket_pose[1, :3]
    assert torch.allclose(translations[box_count], expected_rotated, atol=1.0e-6)


def test_network_switch_presentation_callback_and_close_are_idempotent(monkeypatch):
    class FakeRegistry:
        def __init__(self):
            self.callbacks = {}
            self.removed = []

        def add_callback(self, name, callback):
            self.callbacks[name] = callback

        def remove_callback(self, name):
            self.removed.append(name)
            self.callbacks.pop(name, None)

    class FakeMarker:
        instances = []

        def __init__(self, cfg):
            self.cfg = cfg
            self.visualizations = []
            self.visibility = []
            type(self).instances.append(self)

        def visualize(self, **kwargs):
            self.visualizations.append(kwargs)

        def set_visibility(self, visible):
            self.visibility.append(visible)

    monkeypatch.setattr(presentation, "VisualizationMarkers", FakeMarker)
    registry = FakeRegistry()
    sim = SimpleNamespace(vis_marker_registry=registry)
    pose = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]])

    switch = presentation.NewtonGlNetworkSwitchPresentation(sim, lambda: pose)
    marker = FakeMarker.instances[-1]
    callback_id = switch._callback_id
    assert callback_id in registry.callbacks
    assert len(marker.visualizations) == 1
    assert marker.visualizations[0]["translations"].shape[0] == len(presentation._NETWORK_SWITCH_BOXES)

    registry.callbacks[callback_id]()
    assert len(marker.visualizations) == 2
    switch.close()
    switch.close()
    assert registry.removed == [callback_id]
    assert marker.visibility == [False]


def test_network_switch_marker_proxy_is_only_enabled_for_standalone_gl_play():
    class FakeVisualizer:
        def __init__(self, visualizer_type: str, *, pumps_kit: bool = False, enable_markers: bool = True):
            self.cfg = SimpleNamespace(visualizer_type=visualizer_type, enable_markers=enable_markers)
            self._pumps_kit = pumps_kit

        def supports_markers(self):
            return True

        def pumps_app_update(self):
            return self._pumps_kit

    def _sim(*visualizers, gui=False, offscreen=False, rtx=False, xr=False):
        settings = {
            "/isaaclab/render/rtx_sensors": rtx,
            "/isaaclab/xr/enabled": xr,
        }
        return SimpleNamespace(
            has_gui=gui,
            has_offscreen_render=offscreen,
            visualizers=visualizers,
            get_setting=settings.__getitem__,
        )

    newton_gl = FakeVisualizer("newton_gl")
    assert presentation.should_enable_newton_gl_network_switch(_sim(newton_gl), reset_source="procedural")
    assert not presentation.should_enable_newton_gl_network_switch(_sim(newton_gl), reset_source="dataset")
    assert not presentation.should_enable_newton_gl_network_switch(
        _sim(FakeVisualizer("newton_gl", enable_markers=False)), reset_source="procedural"
    )
    assert not presentation.should_enable_newton_gl_network_switch(
        _sim(newton_gl, FakeVisualizer("kit", pumps_kit=True)), reset_source="procedural"
    )
    assert not presentation.should_enable_newton_gl_network_switch(_sim(newton_gl, rtx=True), reset_source="procedural")
