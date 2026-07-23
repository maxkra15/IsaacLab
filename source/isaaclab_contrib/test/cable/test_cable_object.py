# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for Newton cable rod authoring."""

from __future__ import annotations

import numpy as np
import warp as wp
from newton import JointType, ModelBuilder, eval_fk

from isaaclab_contrib.cable import cable_object
from isaaclab_contrib.cable.cable_object import CableRegistryEntry, add_cable_entry_to_builder


def _curved_cable_entry() -> CableRegistryEntry:
    return CableRegistryEntry(
        prim_path="/World/envs/env_.*/Cable",
        node_positions=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.1, 0.0, 0.0), wp.vec3(0.1, 0.1, 0.0)],
        edges=[(0, 1), (1, 2)],
        radius=0.005,
        density=100.0,
    )


def test_add_cable_entry_builds_only_vbd_rod_bodies():
    """Cable authoring must not add particles or a synthetic free-root joint."""

    builder = ModelBuilder()
    entry = _curved_cable_entry()

    add_cable_entry_to_builder(
        builder,
        entry,
        env_idx=0,
        env_position=[0.0, 0.0, 0.0],
        env_rotation=[0.0, 0.0, 0.0, 1.0],
    )

    assert builder.particle_count == 0
    assert builder.body_count == len(entry.edges)
    assert builder.joint_count == len(entry.edges) - 1
    assert set(int(joint_type) for joint_type in builder.joint_type) == {int(JointType.CABLE)}
    assert entry.body_offsets == [0]


def test_cable_capsules_inherit_builder_shape_defaults():
    """Task contact tuning must reach hook-authored cable capsules."""

    builder = ModelBuilder()
    builder.default_shape_cfg.ke = 1234.0
    builder.default_shape_cfg.kd = 0.25
    builder.default_shape_cfg.mu = 0.75
    builder.default_shape_cfg.margin = 0.002
    builder.default_shape_cfg.gap = 0.007

    add_cable_entry_to_builder(
        builder,
        _curved_cable_entry(),
        env_idx=0,
        env_position=[0.0, 0.0, 0.0],
        env_rotation=[0.0, 0.0, 0.0, 1.0],
    )

    np.testing.assert_allclose(builder.shape_material_ke, [1234.0, 1234.0])
    np.testing.assert_allclose(builder.shape_material_kd, [0.25, 0.25])
    np.testing.assert_allclose(builder.shape_material_mu, [0.75, 0.75])
    np.testing.assert_allclose(builder.shape_margin, [0.002, 0.002])
    np.testing.assert_allclose(builder.shape_gap, [0.007, 0.007])


def test_newton_eval_fk_preserves_curved_cable_body_poses():
    """Latest Newton leaves VBD-owned CABLE transforms unchanged during generic FK."""

    builder = ModelBuilder()
    entry = _curved_cable_entry()
    builder.add_rod_graph(
        node_positions=entry.node_positions,
        edges=entry.edges,
        radius=entry.radius,
        body_frame_origin="start",
        wrap_in_articulation=True,
    )
    model = builder.finalize(device="cpu")
    state = model.state()
    body_q_before = state.body_q.numpy().copy()

    eval_fk(model, state.joint_q, state.joint_qd, state)

    np.testing.assert_array_equal(state.body_q.numpy(), body_q_before)


def test_install_cable_builder_hooks_is_idempotent(monkeypatch):
    """Repeated manager setup must not duplicate cable callbacks."""

    manager = cable_object.SimulationManager
    monkeypatch.setattr(manager, "_cable_registry", ["stale"], raising=False)
    monkeypatch.setattr(manager, "_per_world_builder_hooks", [], raising=False)
    monkeypatch.setattr(manager, "_post_replicate_hooks", [], raising=False)
    monkeypatch.setattr(manager, "_pre_render_callbacks", {}, raising=False)

    cable_object.install_cable_builder_hooks()
    cable_object.install_cable_builder_hooks()

    assert manager._cable_registry == []
    assert manager._per_world_builder_hooks == [cable_object.add_registered_cables_to_builder]
    assert manager._post_replicate_hooks == [cable_object.color_registered_cables]
    assert manager._pre_render_callbacks == {"cable_curve_sync": cable_object.sync_registered_cable_curves_to_usd}
