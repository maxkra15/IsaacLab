# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for scoped Newton per-world builder hooks."""

import pytest
from isaaclab_newton.cloner import newton_builder_world_hook
from isaaclab_newton.physics import NewtonManager


def test_newton_builder_world_hook_registration_is_idempotent(monkeypatch):
    """Existing and nested registrations are not duplicated or removed."""

    def hook(*_args):
        pass

    hooks = []
    monkeypatch.setattr(NewtonManager, "_per_world_builder_hooks", hooks)

    with newton_builder_world_hook(hook):
        assert hooks == [hook]
        with newton_builder_world_hook(hook):
            assert hooks == [hook]
        assert hooks == [hook]
    assert hooks == []

    hooks.append(hook)
    with newton_builder_world_hook(hook):
        assert hooks == [hook]
    assert hooks == [hook]


def test_newton_builder_world_hook_preserves_other_hooks(monkeypatch):
    """Cleanup preserves hooks registered before and during the context."""

    def existing(*_args):
        pass

    def temporary(*_args):
        pass

    def added_later(*_args):
        pass

    hooks = [existing]
    monkeypatch.setattr(NewtonManager, "_per_world_builder_hooks", hooks)

    with pytest.raises(RuntimeError, match="stop"):
        with newton_builder_world_hook(temporary):
            hooks.append(added_later)
            assert hooks == [existing, temporary, added_later]
            raise RuntimeError("stop")

    assert hooks == [existing, added_later]
