# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Franka cabinet asset-path compatibility."""

import re

import pytest

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.cabinet.config.franka.joint_pos_env_cfg import FrankaCabinetSceneCfg
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_ENV_ROOT = "/World/envs/env_0"
_MENAGERIE_ARM_PATH = (
    f"{_ENV_ROOT}/Robot/Geometry/panda_link0/panda_link1/panda_link2/panda_link3/"
    "panda_link4/panda_link5/panda_link6/panda_link7"
)


def _matches_environment_path(pattern: str, path: str) -> bool:
    return re.fullmatch(pattern.replace("{ENV_REGEX_NS}", _ENV_ROOT), path) is not None


def test_franka_cabinet_frame_paths_match_legacy_and_menagerie_assets() -> None:
    """Frame selectors must resolve the same links from either maintained Franka hierarchy."""
    cfg = FrankaCabinetSceneCfg(num_envs=1, env_spacing=2.0)
    targets = {target.name: target.prim_path for target in cfg.ee_frame.target_frames}

    expected_paths = {
        cfg.ee_frame.prim_path: (
            f"{_ENV_ROOT}/Robot/panda_link0",
            f"{_ENV_ROOT}/Robot/Geometry/panda_link0",
        ),
        targets["ee_tcp"]: (
            f"{_ENV_ROOT}/Robot/panda_hand",
            f"{_MENAGERIE_ARM_PATH}/panda_hand",
        ),
        targets["tool_leftfinger"]: (
            f"{_ENV_ROOT}/Robot/panda_leftfinger",
            f"{_MENAGERIE_ARM_PATH}/panda_hand/panda_leftfinger",
        ),
        targets["tool_rightfinger"]: (
            f"{_ENV_ROOT}/Robot/panda_rightfinger",
            f"{_MENAGERIE_ARM_PATH}/panda_hand/panda_rightfinger",
        ),
    }

    for pattern, paths in expected_paths.items():
        assert all(_matches_environment_path(pattern, path) for path in paths)


@pytest.mark.parametrize("task", ["Isaac-Open-Drawer-Franka", "Isaac-Open-Drawer-Franka-Direct"])
@pytest.mark.parametrize(
    ("physics_preset", "expected_variants"),
    [
        (
            "isaacsim_physx",
            {"Physics": "physx", "Colliders": "physx_convex_hulls_compact"},
        ),
        (
            "newton_mjwarp",
            {"Physics": "mujoco", "Colliders": "physx_minimal_compact"},
        ),
    ],
)
def test_franka_cabinet_uses_backend_variants_from_shared_asset(
    task: str, physics_preset: str, expected_variants: dict[str, str]
) -> None:
    """Both cabinet workflows must use one Menagerie root with backend-specific payloads."""
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg = resolve_presets(cfg, selected=(physics_preset,))

    assert cfg.scene.robot.spawn.usd_path.endswith("/FrankaEmika/franka_panda.usda")
    assert cfg.scene.robot.spawn.variants == expected_variants
