# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pinned Rizon4s Sharpa asset and articulation contract."""

from __future__ import annotations

import hashlib
import math
import os
from functools import lru_cache
from pathlib import Path

RIZON_SHARPA_REPOSITORY = "https://gitlab-master.nvidia.com/dex/fabrics-sim.git"
RIZON_SHARPA_REVISION = "d0dbd1ddaefc4996db546949a7dfb37e39afcbeb"
RIZON_SHARPA_RELATIVE_PATH = (
    "src/fabrics_sim/models/robots/urdf/rizon4s_sharpa/"
    "rizon4s_sharpa_no_spheres/rizon4s_sharpa_no_spheres_generated.usd"
)
RIZON_SHARPA_USD_NAME = "rizon4s_sharpa_no_spheres_generated.usd"
RIZON_SHARPA_USD_SHA256 = "24f59deb88db896563aa74a4001e8522b75af6060493ab7bb652f21d81efcff8"
RIZON_SHARPA_BUNDLE_SHA256 = "ae5d22792b44fb6d29a7691d4276bc061a5529132f01e7a0eb5795a482595d63"
RIZON_SHARPA_BUNDLE_FILE_COUNT = 15
RIZON_SHARPA_BUNDLE_SIZE_BYTES = 379_258_308
RIZON_SHARPA_ASSET_ROOT_ENV = "ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT"
RIZON_SHARPA_LICENSE = "NVIDIA-Proprietary-upstream-license"

RIZON_SHARPA_BASE_LINK_NAME = "base_link"
RIZON_SHARPA_NATIVE_PALM_BODY_NAME = "right_hand_C_MC"
# Fabrics-Sim authors this fixed child from measured knuckle geometry. Its
# canonical axes are X toward the knuckles and Z out of the palm. The native
# Sharpa palm swaps X/Z and must never be used directly as a pose objective.
RIZON_SHARPA_END_EFFECTOR_BODY_NAME = "r_palm_ctrl"
RIZON_SHARPA_ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES = (
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
)

RIZON_SHARPA_ARM_HOME_RAD = (0.0, -0.698, 0.0, 1.571, 0.0, 0.698, 0.0)
RIZON_SHARPA_RIGHT_HAND_OPEN_RAD = (0.0,) * len(RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES)
RIZON_SHARPA_RIGHT_HAND_LIMITS_RAD = (
    (-0.1745, 1.9199),
    (-0.3491, 0.3491),
    (-0.5236, 1.3963),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (-0.17453293, 1.5708),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (0.0, 1.3963),
    (-0.17453293, 1.5708),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (0.0, 1.3963),
    (-0.17453293, 1.5708),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (0.0, 1.3963),
    (0.0, 0.2618),
    (-0.17453293, 1.5708),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (0.0, 1.3963),
)
RIZON_SHARPA_RIGHT_HAND_CLOSE_RAD = tuple(
    math.radians(value)
    for value in (
        58.0,
        12.0,
        44.0,
        0.0,
        52.0,
        58.0,
        0.0,
        68.0,
        48.0,
        62.0,
        0.0,
        72.0,
        50.0,
        64.0,
        0.0,
        74.0,
        52.0,
        8.0,
        66.0,
        0.0,
        76.0,
        54.0,
    )
)
RIZON_SHARPA_LOCAL_BOUNDS_MIN_M = (-0.1054999977, -0.2252846243, 0.0)
RIZON_SHARPA_LOCAL_BOUNDS_MAX_M = (0.3730500270, 0.0824961811, 1.3059999992)

_REQUIRED_PATHS = (
    RIZON_SHARPA_USD_NAME,
    "textures/t_Rizon4s_EmissiveMask.1001.png",
    "textures/t_Rizon4s_EmissiveMask.1002.png",
    "textures/t_Rizon4s_alb.1001.png",
    "textures/t_Rizon4s_alb.1002.png",
    "textures/t_Rizon4s_alb.1003.png",
    "textures/t_Rizon4s_nor.1001.png",
    "textures/t_Rizon4s_nor.1002.png",
    "textures/t_Rizon4s_nor.1003.png",
    "textures/t_Rizon4s_orm.1001.png",
    "textures/t_Rizon4s_orm.1002.png",
    "textures/t_Rizon4s_orm.1003.png",
    "textures/t_Sharpa_alb.png",
    "textures/t_Sharpa_nor.png",
    "textures/t_Sharpa_orm.png",
)


def default_rizon_sharpa_bundle_root() -> Path:
    """Return the content-addressed default asset bundle root."""
    return (
        Path.home() / ".cache" / "isaaclab" / "fabrics-sim" / "rizon4s-sharpa" / "sha256" / RIZON_SHARPA_BUNDLE_SHA256
    )


def _bundle_digest(root: Path) -> tuple[str, int]:
    manifest = hashlib.sha256()
    total_size = 0
    for relative_path in _REQUIRED_PATHS:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Pinned Rizon4s Sharpa dependency is missing: {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest.update(f"{digest}  {relative_path}\n".encode())
        total_size += path.stat().st_size
    return manifest.hexdigest(), total_size


@lru_cache(maxsize=4)
def verify_rizon_sharpa_bundle(root: str | Path) -> Path:
    """Verify the exact upstream USD bundle and return its USD path."""
    resolved = Path(root).expanduser().resolve()
    if resolved.is_file():
        resolved = resolved.parent
    digest, total_size = _bundle_digest(resolved)
    if digest != RIZON_SHARPA_BUNDLE_SHA256:
        raise RuntimeError(
            f"Rizon4s Sharpa bundle SHA-256 mismatch: expected {RIZON_SHARPA_BUNDLE_SHA256}, got {digest}."
        )
    if total_size != RIZON_SHARPA_BUNDLE_SIZE_BYTES:
        raise RuntimeError(
            f"Rizon4s Sharpa bundle size mismatch: expected {RIZON_SHARPA_BUNDLE_SIZE_BYTES}, got {total_size}."
        )
    return resolved / RIZON_SHARPA_USD_NAME


def configured_rizon_sharpa_usd(*, required: bool) -> Path | None:
    """Resolve the configured bundle, optionally failing if it is absent."""
    configured_root = os.environ.get(RIZON_SHARPA_ASSET_ROOT_ENV)
    root = Path(configured_root).expanduser() if configured_root else default_rizon_sharpa_bundle_root()
    candidate = root.parent if root.suffix.lower() in (".usd", ".usda", ".usdc") else root
    if not (candidate / RIZON_SHARPA_USD_NAME).is_file():
        if required:
            raise FileNotFoundError(
                "The Rizon Sharpa task needs the pinned Fabrics-Sim bundle. "
                f"Check out {RIZON_SHARPA_REPOSITORY} at {RIZON_SHARPA_REVISION}, copy "
                f"{RIZON_SHARPA_RELATIVE_PATH} and its sibling textures to {candidate}, "
                f"or set {RIZON_SHARPA_ASSET_ROOT_ENV}."
            )
        return None
    return verify_rizon_sharpa_bundle(candidate)


def rizon_sharpa_asset_contract() -> dict[str, object]:
    """Return path-independent asset provenance and articulation semantics."""
    return {
        "repository": RIZON_SHARPA_REPOSITORY,
        "revision": RIZON_SHARPA_REVISION,
        "relative_path": RIZON_SHARPA_RELATIVE_PATH,
        "usd_sha256": RIZON_SHARPA_USD_SHA256,
        "bundle_sha256": RIZON_SHARPA_BUNDLE_SHA256,
        "bundle_file_count": RIZON_SHARPA_BUNDLE_FILE_COUNT,
        "bundle_size_bytes": RIZON_SHARPA_BUNDLE_SIZE_BYTES,
        "license": RIZON_SHARPA_LICENSE,
        "role": "fixed-base-articulation",
        "articulation": {
            "base_link": RIZON_SHARPA_BASE_LINK_NAME,
            "end_effector_body": RIZON_SHARPA_END_EFFECTOR_BODY_NAME,
            "native_palm_body": RIZON_SHARPA_NATIVE_PALM_BODY_NAME,
            "end_effector_frame": "canonical-x-knuckles-z-out-of-palm",
            "arm_joint_names": RIZON_SHARPA_ARM_JOINT_NAMES,
            "right_hand_joint_names": RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES,
            "arm_dof": len(RIZON_SHARPA_ARM_JOINT_NAMES),
            "right_hand_dof": len(RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES),
            "local_bounds_min_m": RIZON_SHARPA_LOCAL_BOUNDS_MIN_M,
            "local_bounds_max_m": RIZON_SHARPA_LOCAL_BOUNDS_MAX_M,
        },
    }


__all__ = [
    "RIZON_SHARPA_ARM_HOME_RAD",
    "RIZON_SHARPA_ARM_JOINT_NAMES",
    "RIZON_SHARPA_ASSET_ROOT_ENV",
    "RIZON_SHARPA_BASE_LINK_NAME",
    "RIZON_SHARPA_BUNDLE_FILE_COUNT",
    "RIZON_SHARPA_BUNDLE_SHA256",
    "RIZON_SHARPA_BUNDLE_SIZE_BYTES",
    "RIZON_SHARPA_END_EFFECTOR_BODY_NAME",
    "RIZON_SHARPA_LICENSE",
    "RIZON_SHARPA_LOCAL_BOUNDS_MAX_M",
    "RIZON_SHARPA_LOCAL_BOUNDS_MIN_M",
    "RIZON_SHARPA_NATIVE_PALM_BODY_NAME",
    "RIZON_SHARPA_RELATIVE_PATH",
    "RIZON_SHARPA_REPOSITORY",
    "RIZON_SHARPA_REVISION",
    "RIZON_SHARPA_RIGHT_HAND_CLOSE_RAD",
    "RIZON_SHARPA_RIGHT_HAND_LIMITS_RAD",
    "RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES",
    "RIZON_SHARPA_RIGHT_HAND_OPEN_RAD",
    "RIZON_SHARPA_USD_NAME",
    "RIZON_SHARPA_USD_SHA256",
    "configured_rizon_sharpa_usd",
    "default_rizon_sharpa_bundle_root",
    "rizon_sharpa_asset_contract",
    "verify_rizon_sharpa_bundle",
]
