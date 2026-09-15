# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared Franka Menagerie asset selections across core tasks."""

import pytest

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


@pytest.mark.parametrize("task", ["Isaac-Lift-Franka", "Isaac-Reorient-Franka"])
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
def test_rigid_franka_tasks_use_backend_variants_from_shared_asset(
    task: str, physics_preset: str, expected_variants: dict[str, str]
) -> None:
    """Rigid manipulation tasks use compact full-arm collisions only on PhysX."""
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg = resolve_presets(cfg, selected=(physics_preset,))

    assert cfg.scene.robot.spawn.usd_path.endswith("/FrankaEmika/franka_panda.usda")
    assert cfg.scene.robot.spawn.variants == expected_variants
    assert all(actuator.viscous_friction == 0.0 for actuator in cfg.scene.robot.actuators.values())


@pytest.mark.parametrize(
    ("task", "selected_presets"),
    [
        ("Isaac-Lift-Soft-Franka", ("isaacsim_physx",)),
        ("Isaac-Lift-Soft-Franka", ("newton_mjwarp_vbd_proxy",)),
        ("Isaac-Lift-Cloth-Franka", ("isaacsim_physx",)),
        ("Isaac-Lift-Cloth-Franka", ("newton_mjwarp_vbd_proxy",)),
        ("Isaac-Lift-Cable-Franka", ("newton_mjwarp_vbd_proxy",)),
        ("Isaac-Lift-Soft-Franka-Camera", ("isaacsim_physx", "isaacsim_rtx")),
        ("Isaac-Lift-Soft-Franka-Camera", ("newton_mjwarp_vbd_proxy", "newton")),
        ("Isaac-Lift-Cloth-Franka-Camera", ("isaacsim_physx", "isaacsim_rtx")),
        ("Isaac-Lift-Cloth-Franka-Camera", ("newton_mjwarp_vbd_proxy", "newton")),
        ("Isaac-Lift-Cable-Franka-Camera", ("newton_mjwarp_vbd_proxy", "newton")),
    ],
)
def test_deformable_franka_tasks_use_compact_gripper_colliders(task: str, selected_presets: tuple[str, ...]) -> None:
    """State and camera tasks avoid unnecessary full-arm contact geometry on either backend."""
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg = resolve_presets(cfg, selected=selected_presets)

    physics_preset = selected_presets[0]
    expected_physics = "physx" if physics_preset == "isaacsim_physx" else "mujoco"
    assert cfg.scene.robot.spawn.usd_path.endswith("/FrankaEmika/franka_panda.usda")
    assert cfg.scene.robot.spawn.variants == {
        "Physics": expected_physics,
        "Colliders": "physx_minimal_compact",
    }
    assert all(actuator.viscous_friction == 0.0 for actuator in cfg.scene.robot.actuators.values())
