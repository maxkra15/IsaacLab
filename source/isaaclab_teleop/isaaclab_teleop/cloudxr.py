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


def prefer_cuda_for_xr(args_cli: argparse.Namespace) -> bool:
    """Select ``cuda:0`` for an XR workload unless the user chose a device.

    AppLauncher intentionally defaults implicit XR launches to CPU. Newton-backed
    XR tasks that require GPU execution can call this helper before constructing
    AppLauncher while still respecting an explicit ``--device`` selection.

    Returns:
        Whether the device selection was changed.
    """
    if not bool(getattr(args_cli, "xr", False)) or bool(getattr(args_cli, "device_explicit", False)):
        return False
    args_cli.device = "cuda:0"
    args_cli.device_explicit = True
    return True


def prefer_single_gpu_for_xr(args_cli: argparse.Namespace, *, allow_multi_gpu: bool = False) -> bool:
    """Disable single-process multi-GPU rendering for XR unless opted in.

    Multi-GPU Kit rendering can make XR shutdown wait on render semaphores when
    simulation and display rendering use different GPUs. XR entry points can
    call this helper before constructing :class:`~isaaclab.app.AppLauncher`.

    Args:
        args_cli: Parsed application launcher arguments.
        allow_multi_gpu: Whether to preserve the launcher's multi-GPU setting.

    Returns:
        Whether the multi-GPU selection was changed.
    """
    if not bool(getattr(args_cli, "xr", False)) or allow_multi_gpu:
        return False
    if getattr(args_cli, "multi_gpu", None) is False:
        return False
    args_cli.multi_gpu = False
    return True


def align_cloudxr_gpu_for_xr(args_cli: argparse.Namespace) -> bool:
    """Make CloudXR use the same GPU as an XR application's CUDA device.

    CloudXR creates the OpenXR Vulkan compositor in a separate process. On a
    multi-GPU host it otherwise defaults to physical GPU 0, even when Kit was
    launched with ``--device cuda:N``. External-memory swapchain images cannot
    be shared across those physical devices and Kit commonly reports the failed
    allocation as ``VK_ERROR_OUT_OF_DEVICE_MEMORY``.

    The CloudXR runtime documents ``NV_GPU_INDEX`` as its physical-device
    selector. This helper derives it from ``args_cli.device`` for XR launches.

    Returns:
        Whether ``NV_GPU_INDEX`` changed.
    """
    if not bool(getattr(args_cli, "xr", False)):
        return False

    device = str(getattr(args_cli, "device", ""))
    match = re.fullmatch(r"cuda(?::(\d+))?", device)
    if match is None:
        return False

    gpu_index = match.group(1) or "0"
    if os.environ.get("NV_GPU_INDEX") == gpu_index:
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
