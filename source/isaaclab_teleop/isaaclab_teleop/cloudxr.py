# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for starting CloudXR alongside Isaac Sim."""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from pathlib import Path
from types import ModuleType

_WEBSOCKETS_MODULE_NAMES = (
    "websockets.asyncio.client",
    "websockets.asyncio.server",
    "websockets.client",
    "websockets.server",
)


def align_cloudxr_gpu_for_xr(args_cli: argparse.Namespace) -> bool:
    """Select a Kit-compatible physical GPU for CloudXR.

    CloudXR creates the OpenXR Vulkan compositor in a separate process. On a
    multi-GPU host, Kit's renderer can remain on its primary Vulkan GPU while
    Newton simulation runs on another CUDA device. Consequently, an unmasked
    ``--device cuda:N`` is not sufficient evidence that CloudXR should use
    physical GPU ``N``.

    The CloudXR runtime documents ``NV_GPU_INDEX`` as its physical-device
    selector. This helper selects physical GPU 0 for the default unmasked
    ``cuda:0`` launch and resolves numeric ``CUDA_VISIBLE_DEVICES`` remapping.
    It deliberately leaves unmasked nonzero CUDA devices alone so CloudXR
    stays with Kit's primary Vulkan GPU. An existing ``NV_GPU_INDEX`` is
    treated as an explicit user override.

    Returns:
        Whether ``NV_GPU_INDEX`` changed.
    """
    if not bool(getattr(args_cli, "xr", False)) or "NV_GPU_INDEX" in os.environ:
        return False

    device = str(getattr(args_cli, "device", ""))
    match = re.fullmatch(r"cuda(?::(\d+))?", device)
    if match is None:
        return False

    logical_index = int(match.group(1) or 0)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        if logical_index != 0:
            return False
        gpu_index = "0"
    else:
        device_tokens = [token.strip() for token in visible_devices.split(",") if token.strip()]
        if logical_index >= len(device_tokens):
            return False
        gpu_index = device_tokens[logical_index]
        if not gpu_index.isdecimal():
            # CloudXR expects a physical ordinal. UUID and MIG visibility tokens
            # cannot be translated reliably without querying the driver.
            return False
    if int(gpu_index) < 0:
        return False
    os.environ["NV_GPU_INDEX"] = gpu_index
    return True


def _module_path(module: ModuleType) -> Path:
    """Return the resolved source path for a Python module."""
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise RuntimeError(f"Cannot determine the source of {module.__name__!r}.")
    return Path(module_file).resolve()


def _verify_module_roots(package_root: Path, modules: tuple[ModuleType, ...]) -> None:
    """Verify that modules originate from one ``websockets`` package."""
    for module in modules:
        module_path = _module_path(module)
        if not module_path.is_relative_to(package_root):
            raise RuntimeError(
                "CloudXR cannot use a mixed websockets installation: "
                f"{module.__name__!r} was loaded from {module_path}, outside {package_root}."
            )


def preload_cloudxr_websockets() -> None:
    """Preload a consistent WebSockets implementation before Isaac Sim starts.

    Isaac Sim may add its bundled Python packages to :data:`sys.path` while
    Kit starts. Preloading the client and server modules keeps CloudXR from
    combining those bundled modules with the newer WebSockets installation
    required by Isaac Teleop. Calling this function more than once is safe.

    Raises:
        RuntimeError: If WebSockets 14 or newer is unavailable, or modules
            from different WebSockets installations are already loaded.
    """
    try:
        websockets = importlib.import_module("websockets")
    except ImportError as exc:
        raise RuntimeError("CloudXR requires websockets >= 14.") from exc

    version = str(getattr(websockets, "__version__", ""))
    try:
        major_version = int(version.partition(".")[0])
    except ValueError as exc:
        raise RuntimeError(f"Cannot determine the installed websockets version from {version!r}.") from exc
    if major_version < 14:
        raise RuntimeError(f"CloudXR requires websockets >= 14; found {version!r} at {_module_path(websockets)}.")

    package_root = _module_path(websockets).parent
    cached_modules = tuple(module for name in _WEBSOCKETS_MODULE_NAMES if (module := sys.modules.get(name)) is not None)
    _verify_module_roots(package_root, cached_modules)

    try:
        modules = tuple(importlib.import_module(name) for name in _WEBSOCKETS_MODULE_NAMES)
    except ImportError as exc:
        raise RuntimeError(f"CloudXR requires a complete websockets >= 14 installation from {package_root}.") from exc
    _verify_module_roots(package_root, modules)
