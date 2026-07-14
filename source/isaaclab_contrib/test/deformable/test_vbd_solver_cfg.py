# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Pure-Python tests for VBD solver configuration forwarding."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from isaaclab_contrib.coupling import NewtonCouplerManager
from isaaclab_contrib.deformable import vbd_manager
from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg


class _FakeArray:
    """Minimal array exposing Newton's host-copy interface."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.int32)

    def numpy(self):
        return self._values.copy()


def test_vbd_configured_rigid_options_match_newton_constructor():
    """Every forwarded rigid option remains an explicit Newton constructor keyword."""
    constructor_options = set(inspect.signature(vbd_manager.SolverVBD.__init__).parameters)
    configured_options = {
        "friction_epsilon",
        "rigid_contact_k_start",
        "rigid_contact_hard",
        "rigid_contact_history",
        "rigid_body_contact_buffer_size",
        "rigid_body_particle_contact_buffer_size",
        "rigid_avbd_beta",
        "rigid_avbd_gamma",
        "rigid_joint_linear_ke",
        "rigid_joint_angular_ke",
        "rigid_joint_linear_k_start",
        "rigid_joint_angular_k_start",
        "rigid_joint_linear_kd",
        "rigid_joint_angular_kd",
    }

    assert configured_options <= constructor_options
    assert "rigid_joint_hard" not in constructor_options
    assert callable(vbd_manager.SolverVBD.set_joint_constraint_mode)


@pytest.mark.parametrize("rigid_joint_hard", [True, False])
def test_vbd_factory_forwards_rigid_options_and_applies_joint_mode(monkeypatch, rigid_joint_hard):
    """VBD construction forwards Newton options and applies the Isaac Lab all-joints mode."""

    class _FakeSolverVBD:
        def __init__(
            self,
            model,
            *,
            iterations,
            friction_epsilon,
            rigid_contact_k_start,
            rigid_contact_hard,
            rigid_contact_history,
            rigid_body_contact_buffer_size,
            rigid_body_particle_contact_buffer_size,
            rigid_avbd_beta,
            rigid_avbd_gamma,
            rigid_joint_linear_ke,
            rigid_joint_angular_ke,
            rigid_joint_linear_k_start,
            rigid_joint_angular_k_start,
            rigid_joint_linear_kd,
            rigid_joint_angular_kd,
        ):
            self.model = model
            self.options = {
                "iterations": iterations,
                "friction_epsilon": friction_epsilon,
                "rigid_contact_k_start": rigid_contact_k_start,
                "rigid_contact_hard": rigid_contact_hard,
                "rigid_contact_history": rigid_contact_history,
                "rigid_body_contact_buffer_size": rigid_body_contact_buffer_size,
                "rigid_body_particle_contact_buffer_size": rigid_body_particle_contact_buffer_size,
                "rigid_avbd_beta": rigid_avbd_beta,
                "rigid_avbd_gamma": rigid_avbd_gamma,
                "rigid_joint_linear_ke": rigid_joint_linear_ke,
                "rigid_joint_angular_ke": rigid_joint_angular_ke,
                "rigid_joint_linear_k_start": rigid_joint_linear_k_start,
                "rigid_joint_angular_k_start": rigid_joint_angular_k_start,
                "rigid_joint_linear_kd": rigid_joint_linear_kd,
                "rigid_joint_angular_kd": rigid_joint_angular_kd,
            }
            self.joint_mode_calls = []

        def set_joint_constraint_mode(self, joint_index: int, *, hard: bool) -> None:
            self.joint_mode_calls.append((joint_index, hard))

    monkeypatch.setattr(vbd_manager, "SolverVBD", _FakeSolverVBD)
    model = SimpleNamespace(joint_count=3)
    cfg = VBDSolverCfg(
        iterations=7,
        friction_epsilon=0.125,
        rigid_contact_k_start=123.0,
        rigid_contact_hard=False,
        rigid_contact_history=True,
        rigid_body_contact_buffer_size=4096,
        rigid_body_particle_contact_buffer_size=2048,
        rigid_avbd_beta=456.0,
        rigid_avbd_gamma=0.875,
        rigid_joint_linear_ke=1.0e7,
        rigid_joint_angular_ke=2.0e7,
        rigid_joint_linear_k_start=300.0,
        rigid_joint_angular_k_start=40.0,
        rigid_joint_linear_kd=5.0,
        rigid_joint_angular_kd=6.0,
        rigid_joint_hard=rigid_joint_hard,
    )

    solver = vbd_manager.NewtonVBDManager._create_solver(model, cfg)

    assert solver.options == {
        "iterations": 7,
        "friction_epsilon": 0.125,
        "rigid_contact_k_start": 123.0,
        "rigid_contact_hard": False,
        "rigid_contact_history": True,
        "rigid_body_contact_buffer_size": 4096,
        "rigid_body_particle_contact_buffer_size": 2048,
        "rigid_avbd_beta": 456.0,
        "rigid_avbd_gamma": 0.875,
        "rigid_joint_linear_ke": 1.0e7,
        "rigid_joint_angular_ke": 2.0e7,
        "rigid_joint_linear_k_start": 300.0,
        "rigid_joint_angular_k_start": 40.0,
        "rigid_joint_linear_kd": 5.0,
        "rigid_joint_angular_kd": 6.0,
    }
    expected_calls = [] if rigid_joint_hard else [(0, False), (1, False), (2, False)]
    assert solver.joint_mode_calls == expected_calls


def test_coupled_vbd_articulations_are_excluded_from_generic_fk(monkeypatch):
    """The real coupler manager passes resolved VBD joint ownership to the FK filter."""

    class _FakeVBD:
        pass

    class _FakeCoupledSolver:
        def entry_names(self):
            return ("rigid", "soft")

        def solver(self, name):
            return _FakeVBD() if name == "soft" else object()

    recorded: list[list[bool] | None] = []
    monkeypatch.setattr(vbd_manager, "SolverVBD", _FakeVBD)
    monkeypatch.setattr(
        vbd_manager.NewtonManager,
        "_model",
        SimpleNamespace(
            articulation_count=2,
            joint_articulation=_FakeArray([0, 1]),
        ),
    )
    monkeypatch.setattr(vbd_manager.NewtonManager, "_solver", _FakeCoupledSolver())
    # Ownership is adapter lifecycle state, not an ad-hoc attribute on Newton's
    # SolverCoupled instance. This mirrors the concrete NewtonCouplerManager path.
    monkeypatch.setattr(
        NewtonCouplerManager,
        "_resolved_entries_by_name",
        {
            "rigid": SimpleNamespace(joints=[0]),
            "soft": SimpleNamespace(joints=[1]),
        },
        raising=False,
    )
    monkeypatch.setattr(
        vbd_manager.NewtonManager,
        "_set_fk_articulation_filter",
        classmethod(lambda cls, mask: recorded.append(None if mask is None else list(mask))),
    )

    NewtonCouplerManager._configure_fk_articulation_filter()

    assert recorded == [[True, False]]
