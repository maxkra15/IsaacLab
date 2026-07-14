# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for CloudXR dependency compatibility helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from isaaclab_teleop import preload_cloudxr_websockets
from isaaclab_teleop.cloudxr import prefer_cuda_for_xr

_WEBSOCKETS_MODULE_NAMES = (
    "websockets.asyncio.client",
    "websockets.asyncio.server",
    "websockets.client",
    "websockets.server",
)


def test_xr_cuda_preference_updates_only_implicit_device_selection() -> None:
    implicit = SimpleNamespace(xr=True, device="cpu", device_explicit=False)
    explicit = SimpleNamespace(xr=True, device="cpu", device_explicit=True)
    desktop = SimpleNamespace(xr=False, device="cpu", device_explicit=False)

    assert prefer_cuda_for_xr(implicit) is True
    assert implicit.device == "cuda:0"
    assert implicit.device_explicit is True
    assert prefer_cuda_for_xr(explicit) is False
    assert explicit.device == "cpu"
    assert prefer_cuda_for_xr(desktop) is False
    assert desktop.device == "cpu"


def test_preload_cloudxr_websockets_is_consistent_and_idempotent() -> None:
    """The preload keeps all CloudXR WebSockets modules in one package."""
    preload_cloudxr_websockets()

    package_root = Path(sys.modules["websockets"].__file__).resolve().parent
    module_ids = {name: id(sys.modules[name]) for name in _WEBSOCKETS_MODULE_NAMES}
    for name in _WEBSOCKETS_MODULE_NAMES:
        assert Path(sys.modules[name].__file__).resolve().is_relative_to(package_root)

    preload_cloudxr_websockets()

    assert {name: id(sys.modules[name]) for name in _WEBSOCKETS_MODULE_NAMES} == module_ids


def test_preload_cloudxr_websockets_rejects_old_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """The preload rejects WebSockets releases without the asyncio API."""
    import websockets

    monkeypatch.setattr(websockets, "__version__", "12.0")

    with pytest.raises(RuntimeError, match="requires websockets >= 14"):
        preload_cloudxr_websockets()


def test_preload_cloudxr_websockets_rejects_mixed_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The preload detects modules cached from Kit's bundled package."""
    preload_cloudxr_websockets()
    monkeypatch.setattr(sys.modules["websockets.client"], "__file__", str(tmp_path / "websockets" / "client.py"))

    with pytest.raises(RuntimeError, match="mixed websockets installation"):
        preload_cloudxr_websockets()
