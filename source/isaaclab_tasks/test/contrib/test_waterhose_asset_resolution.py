# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path


def test_waterhose_physics_assets_are_task_local():
    from isaaclab_tasks.contrib.waterhose import waterhose_env_cfg

    expected_dir = Path(waterhose_env_cfg.__file__).resolve().parent / "assets"

    assert Path(waterhose_env_cfg._FRIDGE_USD) == expected_dir / "fridge" / "fridge_waterhose.usda"
    assert Path(waterhose_env_cfg._RBY1_USD) == expected_dir / "rby1df" / "rby1df_waterhose.usda"
    assert Path(waterhose_env_cfg._PLUG_USD) == expected_dir / "fridge" / "cable" / "plug.usda"
    assert Path(waterhose_env_cfg._CABLE1_USD) == expected_dir / "fridge" / "cable" / "cable001.usda"
    assert (expected_dir / "fridge" / "cable" / "plug_visual.usda").is_file()


def test_waterhose_assets_are_self_contained():
    from isaaclab_tasks.contrib.waterhose import waterhose_env_cfg

    required_assets = (
        waterhose_env_cfg._FRIDGE_USD,
        waterhose_env_cfg._RBY1_USD,
        waterhose_env_cfg._PLUG_USD,
        waterhose_env_cfg._CABLE1_USD,
        waterhose_env_cfg._SKY_HDR,
        waterhose_env_cfg._GROUND_USD,
        waterhose_env_cfg._FRIDGE_COLLISION_PROXY_USD,
    )
    assert all(Path(path).is_file() for path in required_assets)
