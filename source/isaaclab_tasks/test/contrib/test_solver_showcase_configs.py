# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only registration and authored-asset checks for the solver showcases."""

from pathlib import Path

import pytest

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config


@pytest.mark.parametrize(
    ("task_name", "scene_asset_name"),
    (
        ("IsaacContrib-Kinetic-Foundry", "backdrop"),
        ("IsaacContrib-Textile-Atelier-Kuka-GR1T2", "frame"),
        ("IsaacContrib-RelayJuggle-KukaAllegro-GR1T2", "gallery"),
    ),
)
def test_solver_showcase_config_references_authored_usda(task_name: str, scene_asset_name: str) -> None:
    """Each registered PPO task resolves a scene with its authored USDA present."""
    env_cfg, agent_cfg = resolve_task_config(task_name, "rsl_rl_cfg_entry_point", overrides=[])

    assert agent_cfg is not None
    usd_path = Path(getattr(env_cfg.scene, scene_asset_name).spawn.usd_path)
    assert usd_path.suffix == ".usda"
    assert usd_path.is_file(), f"{task_name} references a missing authored asset: {usd_path}"
