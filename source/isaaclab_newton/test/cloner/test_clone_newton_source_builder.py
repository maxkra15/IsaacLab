# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for cloning retained Newton replication sources."""

import newton
import pytest
from isaaclab_newton.cloner import newton_builder_clone_source
from isaaclab_newton.physics import NewtonManager


def test_newton_builder_clone_source_owns_mutable_containers(monkeypatch):
    """Structural mutations on a clone do not change the retained source."""
    prototype = newton.ModelBuilder()
    body_id = prototype.add_body(label="source")
    mesh = newton.Mesh(
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        indices=[0, 1, 2],
        compute_inertia=False,
    )
    prototype.add_shape_mesh(
        body_id,
        mesh=mesh,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
        label="triangle",
    )
    monkeypatch.setattr(NewtonManager, "_cl_protos", {"/World/envs/env_0": prototype})

    clone = newton_builder_clone_source("/World/envs/env_0")

    clone.body_label[body_id] = "clone"
    clone.body_shapes[body_id].append(99)
    clone.shape_source[0] = None

    assert prototype.body_label == ["source"]
    assert prototype.body_shapes[body_id] == [0]
    assert prototype.shape_source[0] is mesh


def test_newton_builder_clone_source_detaches_geometry_before_finalize(monkeypatch):
    """Finalizing a clone does not rematerialize retained-source geometry."""
    prototype = newton.ModelBuilder()
    body_id = prototype.add_body(label="source")
    mesh = newton.Mesh(
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        indices=[0, 1, 2],
        compute_inertia=False,
    )
    prototype.add_shape_mesh(
        body_id,
        mesh=mesh,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
    )
    monkeypatch.setattr(NewtonManager, "_cl_protos", {"/World/envs/env_0": prototype})

    clone = newton_builder_clone_source("/World/envs/env_0")

    assert clone.shape_source[0].mesh is None
    assert prototype.shape_source[0].mesh is None

    clone.finalize(device="cpu")

    assert clone.shape_source[0].mesh is not None
    assert prototype.shape_source[0].mesh is None


def test_newton_builder_clone_source_rejects_missing_source(monkeypatch):
    """Unknown paths report both the missing path and retained alternatives."""
    prototype = newton.ModelBuilder()
    monkeypatch.setattr(NewtonManager, "_cl_protos", {"/World/envs/env_0": prototype})

    with pytest.raises(RuntimeError, match="/World/missing"):
        newton_builder_clone_source("/World/missing")
