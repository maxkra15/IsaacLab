# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import pytest


def test_waterhose_physics_assets_are_task_local():
    from isaaclab_tasks.contrib.waterhose import waterhose_env_cfg

    expected_dir = Path(waterhose_env_cfg.__file__).resolve().parent / "assets"

    assert Path(waterhose_env_cfg._FRIDGE_USD) == expected_dir / "fridge" / "fridge_waterhose.usda"
    assert Path(waterhose_env_cfg._RBY1_USD) == expected_dir / "rby1df" / "rby1df_waterhose.usda"
    assert Path(waterhose_env_cfg._PLUG_USD) == expected_dir / "fridge" / "cable" / "plug.usda"
    assert Path(waterhose_env_cfg._CABLE1_USD) == expected_dir / "fridge" / "cable" / "cable001.usda"


def test_required_waterhose_assets_exist():
    from isaaclab_tasks.contrib.waterhose import waterhose_env_cfg

    expected_dir = Path(waterhose_env_cfg.__file__).resolve().parent / "assets"
    required_assets = (
        Path(waterhose_env_cfg._FRIDGE_USD),
        Path(waterhose_env_cfg._RBY1_USD),
        Path(waterhose_env_cfg._PLUG_USD),
        Path(waterhose_env_cfg._CABLE1_USD),
        Path(waterhose_env_cfg._SKY_HDR),
        Path(waterhose_env_cfg._GROUND_USD),
        Path(waterhose_env_cfg._FRIDGE_COLLISION_PROXY_USD),
        expected_dir / "fridge" / "cable" / "plug_visual.usda",
    )
    if not expected_dir.is_dir():
        pytest.skip("External waterhose asset bundle is not installed")
    missing_assets = [path for path in required_assets if not path.is_file()]
    assert not missing_assets, f"Missing required waterhose assets: {missing_assets}"
