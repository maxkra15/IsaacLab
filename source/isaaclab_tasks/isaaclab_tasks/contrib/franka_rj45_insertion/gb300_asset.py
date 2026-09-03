# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pinned local resolver for the render-only NVIDIA SimReady GB300 asset."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

GB300_SIMREADY_REPOSITORY = "nvidia/simready-dsx"
GB300_SIMREADY_REVISION = "5938869019f0d2afb6b9b808ed1ab1bc6e0e0961"
GB300_SIMREADY_RELATIVE_PATH = "GB300/simready_usd/payloads/external.usd"
GB300_SIMREADY_EXTERNAL_USD_SHA256 = "5e0b7b3b58d005b24909b8d2e735c49997f8dbea72352b51911326343ef1e7bb"
GB300_SIMREADY_EXTERNAL_USD_SIZE = 473_434_496
GB300_SIMREADY_LICENSE = "CC-BY-4.0"
GB300_SIMREADY_ASSET_ROOT_ENV = "ISAACLAB_SIMREADY_DSX_GB300_ROOT"
GB300_SIMREADY_DOWNLOAD_URL = (
    "https://huggingface.co/datasets/nvidia/simready-dsx/resolve/"
    f"{GB300_SIMREADY_REVISION}/{GB300_SIMREADY_RELATIVE_PATH}?download=true"
)


def default_gb300_external_usd_path() -> Path:
    """Return the content-addressed default cache path for the external rack USD."""
    return (
        Path.home()
        / ".cache"
        / "isaaclab"
        / "simready-dsx"
        / "sha256"
        / GB300_SIMREADY_EXTERNAL_USD_SHA256
        / "external.usd"
    )


@lru_cache(maxsize=4)
def verify_gb300_external_usd(path: str | Path) -> Path:
    """Verify and return one exact local GB300 external payload."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Pinned GB300 external USD is missing: {resolved}")
    size = resolved.stat().st_size
    if size != GB300_SIMREADY_EXTERNAL_USD_SIZE:
        raise RuntimeError(
            f"GB300 external USD size mismatch: expected {GB300_SIMREADY_EXTERNAL_USD_SIZE}, got {size}."
        )
    with resolved.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != GB300_SIMREADY_EXTERNAL_USD_SHA256:
        raise RuntimeError(
            "GB300 external USD SHA-256 mismatch: "
            f"expected {GB300_SIMREADY_EXTERNAL_USD_SHA256}, got {digest}."
        )
    return resolved


def configured_gb300_external_usd(*, required: bool) -> Path | None:
    """Resolve the configured payload, optionally failing when it is absent."""
    configured_root = os.environ.get(GB300_SIMREADY_ASSET_ROOT_ENV)
    if configured_root:
        root = Path(configured_root).expanduser()
        candidate = root if root.suffix.lower() in (".usd", ".usda", ".usdc") else root / "external.usd"
    else:
        candidate = default_gb300_external_usd_path()
    if not candidate.is_file():
        if required:
            raise FileNotFoundError(
                "The GB300 Kit presentation needs the pinned SimReady payload. Download "
                f"{GB300_SIMREADY_DOWNLOAD_URL} to {candidate}, or set {GB300_SIMREADY_ASSET_ROOT_ENV}."
            )
        return None
    return verify_gb300_external_usd(candidate)


def gb300_asset_contract() -> dict[str, object]:
    """Return path-independent provenance for the optional presentation asset."""
    return {
        "repository": GB300_SIMREADY_REPOSITORY,
        "revision": GB300_SIMREADY_REVISION,
        "relative_path": GB300_SIMREADY_RELATIVE_PATH,
        "file_sha256": GB300_SIMREADY_EXTERNAL_USD_SHA256,
        "file_size_bytes": GB300_SIMREADY_EXTERNAL_USD_SIZE,
        "license": GB300_SIMREADY_LICENSE,
        "role": "render-only-no-physics-or-reset-state-dependency",
    }


__all__ = [
    "GB300_SIMREADY_ASSET_ROOT_ENV",
    "GB300_SIMREADY_DOWNLOAD_URL",
    "GB300_SIMREADY_EXTERNAL_USD_SHA256",
    "GB300_SIMREADY_EXTERNAL_USD_SIZE",
    "GB300_SIMREADY_LICENSE",
    "GB300_SIMREADY_RELATIVE_PATH",
    "GB300_SIMREADY_REPOSITORY",
    "GB300_SIMREADY_REVISION",
    "configured_gb300_external_usd",
    "default_gb300_external_usd_path",
    "gb300_asset_contract",
    "verify_gb300_external_usd",
]
