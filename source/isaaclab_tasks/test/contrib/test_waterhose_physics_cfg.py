# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the waterhose physics configuration."""

from __future__ import annotations

import pytest

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
