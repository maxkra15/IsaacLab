# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the waterhose physics configuration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from isaaclab_newton.physics import NewtonManager
from newton import ShapeFlags

from isaaclab_tasks.contrib.waterhose import waterhose_env_cfg


def _solver_tuning(monkeypatch, *, substeps: str | None = None, vbd_iters: str | None = None) -> tuple[int, int]:
    if substeps is None:
        monkeypatch.delenv("WATERHOSE_SUBSTEPS", raising=False)
    else:
        monkeypatch.setenv("WATERHOSE_SUBSTEPS", substeps)
    if vbd_iters is None:
        monkeypatch.delenv("WATERHOSE_VBD_ITERS", raising=False)
    else:
        monkeypatch.setenv("WATERHOSE_VBD_ITERS", vbd_iters)

    cfg = waterhose_env_cfg.WaterhoseEnvCfg()
    entries = {entry.name: entry for entry in cfg.sim.physics.solver_cfg.entries}
    return cfg.sim.physics.num_substeps, entries["vbd"].solver_cfg.iterations


def test_waterhose_uses_validated_solver_tuning_defaults(monkeypatch):
    assert _solver_tuning(monkeypatch) == (8, 16)


def test_waterhose_solver_tuning_accepts_environment_overrides(monkeypatch):
    assert _solver_tuning(monkeypatch, substeps="6", vbd_iters="12") == (6, 12)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("WATERHOSE_SUBSTEPS", "not-an-int"),
        ("WATERHOSE_SUBSTEPS", "0"),
        ("WATERHOSE_VBD_ITERS", "-1"),
    ],
)
def test_waterhose_solver_tuning_rejects_invalid_values(monkeypatch, name, value):
    monkeypatch.delenv("WATERHOSE_SUBSTEPS", raising=False)
    monkeypatch.delenv("WATERHOSE_VBD_ITERS", raising=False)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=rf"{name} must be a positive integer"):
        waterhose_env_cfg.WaterhoseEnvCfg()


def test_robot_collision_filter_preserves_only_right_fingers(monkeypatch):
    collide_mask = int(ShapeFlags.COLLIDE_SHAPES) | int(ShapeFlags.COLLIDE_PARTICLES)
    visible = int(ShapeFlags.VISIBLE)
    builder = SimpleNamespace(
        body_label=[
            "/World/envs/env_0/Robot/right_arm_link_1",
            "/World/envs/env_0/Robot/right_gripper_leftfinger",
            "/World/envs/env_0/Cable1/cable_edge_body_0",
        ],
        shape_body=[0, 1, 2, -1],
        shape_flags=[collide_mask | visible] * 4,
    )
    monkeypatch.setattr(NewtonManager, "_builder", builder, raising=False)

    waterhose_env_cfg._restrict_rby1df_collision_to_right_gripper()

    assert builder.shape_flags[0] & collide_mask == 0
    assert builder.shape_flags[0] & visible
    assert builder.shape_flags[1] & collide_mask == collide_mask
    assert builder.shape_flags[2] & collide_mask == collide_mask
    assert builder.shape_flags[3] & collide_mask == collide_mask


def test_robot_collision_filter_is_registered(monkeypatch):
    registrations = []

    def record(callback, event, **kwargs):
        registrations.append((callback, event, kwargs))

    monkeypatch.setattr(NewtonManager, "register_callback", record)
    waterhose_env_cfg._register_rby1df_collision_restriction()

    matches = [item for item in registrations if item[2]["name"] == "waterhose_restrict_rby1df_collision"]
    assert len(matches) == 1
    assert matches[0][0] is waterhose_env_cfg._restrict_rby1df_collision_to_right_gripper


def test_waterhose_reset_restores_the_full_scene_and_joint_targets():
    events = waterhose_env_cfg.EventCfg()

    assert events.reset_scene.func is waterhose_env_cfg.mdp.reset_scene_to_default
    assert events.reset_scene.params == {"reset_joint_targets": True}
    assert not hasattr(events, "reset_robot_joints")
