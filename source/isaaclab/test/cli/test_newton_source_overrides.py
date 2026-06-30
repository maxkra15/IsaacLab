# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for explicit local Newton and Warp source overrides."""

from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def test_launcher_does_not_implicitly_select_adjacent_physics_checkouts():
    """Local physics sources must be selected explicitly and remain location-independent."""
    launcher = (_REPOSITORY_ROOT / "isaaclab.sh").read_text(encoding="utf-8")

    assert "$ISAACLAB_PATH/../newton-coupled" not in launcher
    assert "NEWTON_SOURCE_DIR" in launcher
    assert "WARP_SOURCE_DIR" in launcher


def test_local_physics_setup_document_is_location_independent():
    """The source-validation recipe must not embed a developer's workstation path."""
    setup = (_REPOSITORY_ROOT / "docs/newton_local_setup.md").read_text(encoding="utf-8")

    assert "/home/" not in setup
    assert "file://" in setup
    assert "/path/to/newton" in setup
    assert "/path/to/warp" in setup


def test_source_validator_is_location_independent_and_checks_native_library():
    """The executable provenance check must cover source revisions and Warp's loaded binary."""
    validator = (_REPOSITORY_ROOT / "tools/validate_newton_sources.py").read_text(encoding="utf-8")

    assert "/home/" not in validator
    assert "rev-parse" in validator
    assert "--porcelain" in validator
    assert "--untracked-files=all" in validator
    assert "newton.__file__" in validator
    assert "wp.__file__" in validator
    assert "runtime.core._name" in validator
