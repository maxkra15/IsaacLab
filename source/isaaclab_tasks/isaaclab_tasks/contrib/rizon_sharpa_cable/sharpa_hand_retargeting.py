# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pinned NVIDIA DexPilot assets for right-hand Sharpa retargeting."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .robot_asset import RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES

ISAAC_TELEOP_REPOSITORY = "https://github.com/NVIDIA/IsaacTeleop"
ISAAC_TELEOP_REVISION = "c5fe6624cc4dff456485d2e786922c8e41100f83"
SHARPA_URDF_REPOSITORY = "https://github.com/sharpa-robotics/sharpa-urdf-usd-xml"
SHARPA_URDF_REVISION = "3e953f588ba9954cebaa720aaa4cee06a43a068e"

SHARPA_DEXPILOT_CONFIG_NAME = "sharpa_wave_right_dexpilot.yml"
SHARPA_DEXPILOT_CONFIG_SHA256 = "d3e9e017084f45c1d2f80718d2fa976a08f47bd5e2da879b8aecf849a6cf25e7"
SHARPA_DEXPILOT_CONFIG_SIZE_BYTES = 985
SHARPA_DEXPILOT_URDF_NAME = "right_sharpa_wave.urdf"
SHARPA_DEXPILOT_URDF_SHA256 = "7a9ab7f824482d23765b2da40b7e96fc605e7e70eda4615a5ca51fea88afb845"
SHARPA_DEXPILOT_URDF_SIZE_BYTES = 34_545

# Official Isaac Teleop Sharpa/GR1 mapping from OpenXR hand-tracking axes into
# the robot hand's kinematic base. DexHandRetargeter consumes this row-major.
SHARPA_HANDTRACKING_TO_BASELINK = (0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0)

# OpenXR's wrist frame uses +Y through the back of the hand and +Z away from
# the fingertips. Fabrics-Sim's measured ``right_hand_C_MC -> r_palm_ctrl``
# fixed joint uses RPY (1.54660563, -1.36673426, 1.54598752) radians. Composing
# that joint with NVIDIA's official OpenXR-to-Sharpa matrix above gives this
# intrinsic XYZ offset for Se3AbsRetargeter. Keeping the exact geometry-derived
# value avoids the former approximately 90-degree palm error.
SHARPA_OPENXR_TO_CANONICAL_PALM_RPY_DEG = (
    90.28118057,
    2.77868426,
    -101.70219973,
)


def _verified_asset(name: str, expected_sha256: str, expected_size: int) -> Path:
    """Return a packaged retargeting asset after exact content verification."""
    path = Path(__file__).with_name("assets") / name
    if not path.is_file():
        raise FileNotFoundError(f"Packaged Sharpa DexPilot asset is missing: {path}")
    size = path.stat().st_size
    if size != expected_size:
        raise RuntimeError(f"Sharpa DexPilot asset {name} has size {size}; expected {expected_size}.")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(f"Sharpa DexPilot asset {name} has SHA-256 {digest}; expected {expected_sha256}.")
    return path


def sharpa_dexpilot_config_path() -> Path:
    """Return the exact official NVIDIA Sharpa DexPilot configuration."""
    return _verified_asset(
        SHARPA_DEXPILOT_CONFIG_NAME,
        SHARPA_DEXPILOT_CONFIG_SHA256,
        SHARPA_DEXPILOT_CONFIG_SIZE_BYTES,
    )


def sharpa_dexpilot_urdf_path() -> Path:
    """Return the exact official standalone right-hand Sharpa URDF."""
    return _verified_asset(
        SHARPA_DEXPILOT_URDF_NAME,
        SHARPA_DEXPILOT_URDF_SHA256,
        SHARPA_DEXPILOT_URDF_SIZE_BYTES,
    )


def sharpa_hand_retargeting_contract() -> dict[str, object]:
    """Return the reproducible official Sharpa DexPilot contract."""
    return {
        "method": "DexPilot",
        "implementation_repository": ISAAC_TELEOP_REPOSITORY,
        "implementation_revision": ISAAC_TELEOP_REVISION,
        "urdf_repository": SHARPA_URDF_REPOSITORY,
        "urdf_revision": SHARPA_URDF_REVISION,
        "config_sha256": SHARPA_DEXPILOT_CONFIG_SHA256,
        "config_size_bytes": SHARPA_DEXPILOT_CONFIG_SIZE_BYTES,
        "urdf_sha256": SHARPA_DEXPILOT_URDF_SHA256,
        "urdf_size_bytes": SHARPA_DEXPILOT_URDF_SIZE_BYTES,
        "handtracking_to_baselink": SHARPA_HANDTRACKING_TO_BASELINK,
        "openxr_to_canonical_palm_rpy_deg": SHARPA_OPENXR_TO_CANONICAL_PALM_RPY_DEG,
        "wrist_link": "right_hand_C_MC",
        "finger_joint_names": RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES,
        "low_pass_alpha": 0.2,
        "scaling_factor": 1.2,
    }


__all__ = [
    "ISAAC_TELEOP_REPOSITORY",
    "ISAAC_TELEOP_REVISION",
    "SHARPA_DEXPILOT_CONFIG_SHA256",
    "SHARPA_DEXPILOT_URDF_SHA256",
    "SHARPA_HANDTRACKING_TO_BASELINK",
    "SHARPA_OPENXR_TO_CANONICAL_PALM_RPY_DEG",
    "SHARPA_URDF_REPOSITORY",
    "SHARPA_URDF_REVISION",
    "sharpa_dexpilot_config_path",
    "sharpa_dexpilot_urdf_path",
    "sharpa_hand_retargeting_contract",
]
