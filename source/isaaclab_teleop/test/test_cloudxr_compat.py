# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for CloudXR dependency compatibility helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from isaaclab_teleop import preload_cloudxr_websockets
from isaaclab_teleop.cloudxr import align_cloudxr_gpu_for_xr

_WEBSOCKETS_MODULE_NAMES = (
    "websockets.asyncio.client",
    "websockets.asyncio.server",
    "websockets.client",
    "websockets.server",
)


def test_xr_cloudxr_gpu_uses_default_vulkan_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """The separate CloudXR Vulkan process uses Kit's default physical GPU."""
    monkeypatch.delenv("NV_GPU_INDEX", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert align_cloudxr_gpu_for_xr(SimpleNamespace(xr=True, device="cuda:0")) is True
    assert os.environ["NV_GPU_INDEX"] == "0"
    assert align_cloudxr_gpu_for_xr(SimpleNamespace(xr=True, device="cuda:0")) is False

    monkeypatch.delenv("NV_GPU_INDEX")
    assert align_cloudxr_gpu_for_xr(SimpleNamespace(xr=True, device="cuda")) is True
    assert os.environ["NV_GPU_INDEX"] == "0"

    monkeypatch.delenv("NV_GPU_INDEX")
    assert align_cloudxr_gpu_for_xr(SimpleNamespace(xr=False, device="cuda:1")) is False
    assert "NV_GPU_INDEX" not in os.environ
    assert align_cloudxr_gpu_for_xr(SimpleNamespace(xr=True, device="cpu")) is False
    assert "NV_GPU_INDEX" not in os.environ


def test_xr_cloudxr_gpu_does_not_confuse_cuda_with_vulkan(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unmasked secondary simulation GPU does not imply a secondary Vulkan renderer."""
    monkeypatch.delenv("NV_GPU_INDEX", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert align_cloudxr_gpu_for_xr(SimpleNamespace(xr=True, device="cuda:1")) is False
    assert "NV_GPU_INDEX" not in os.environ


@pytest.mark.parametrize(
    ("visible_devices", "device", "expected_index"),
    [
        ("1", "cuda:0", "1"),
        ("2,0", "cuda:1", "0"),
    ],
)
def test_xr_cloudxr_gpu_resolves_cuda_visible_devices(
    monkeypatch: pytest.MonkeyPatch, visible_devices: str, device: str, expected_index: str
) -> None:
    monkeypatch.delenv("NV_GPU_INDEX", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)

    assert align_cloudxr_gpu_for_xr(SimpleNamespace(xr=True, device=device)) is True
    assert os.environ["NV_GPU_INDEX"] == expected_index


def test_xr_cloudxr_gpu_preserves_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0")
    monkeypatch.setenv("NV_GPU_INDEX", "3")

    assert align_cloudxr_gpu_for_xr(SimpleNamespace(xr=True, device="cuda:1")) is False
    assert os.environ["NV_GPU_INDEX"] == "3"


def test_xr_cloudxr_gpu_ignores_uuid_visibility(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NV_GPU_INDEX", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-1234")

    assert align_cloudxr_gpu_for_xr(SimpleNamespace(xr=True, device="cuda:0")) is False
    assert "NV_GPU_INDEX" not in os.environ


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
