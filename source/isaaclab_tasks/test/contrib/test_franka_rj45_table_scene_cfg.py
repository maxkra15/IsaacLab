# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration contracts for the Franka RJ45 Seattle table helper."""

from pathlib import Path

import pytest
from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.sim.schemas import CollisionBaseCfg, RigidBodyBaseCfg, UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg

from isaaclab_tasks.contrib.franka_rj45_insertion.asset_provenance import (
    FRANKA_RJ45_ASSET_CLOSURE_TREE_SHA256,
    FRANKA_RJ45_SEATTLE_TABLE_LOGICAL_URI,
    FrankaRJ45AssetClosure,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.table_scene_cfg import (
    configure_seattle_table_external_asset,
    make_seattle_table_scene_assets,
)


def test_seattle_table_visual_is_fixed_and_collision_free() -> None:
    """Pin the stack task's table pose and recursively disable authored colliders."""
    visual, _ = make_seattle_table_scene_assets()

    assert visual.prim_path == "{ENV_REGEX_NS}/Table"
    assert tuple(visual.init_state.pos) == (0.5, 0.0, 0.0)
    assert tuple(visual.init_state.rot) == (0.0, 0.0, 0.707, 0.707)
    assert isinstance(visual.spawn, sim_utils.UsdFileCfg)
    assert str(visual.spawn.usd_path) == FRANKA_RJ45_SEATTLE_TABLE_LOGICAL_URI
    assert visual.spawn.make_uninstanceable is True
    assert visual.spawn.visual_material is None
    assert visual.spawn.physics_material is None
    assert len(visual.spawn.rigid_props) == 1
    assert isinstance(visual.spawn.rigid_props[0], UsdPhysicsRigidBodyCfg)
    assert visual.spawn.rigid_props[0].kinematic_enabled is True
    assert len(visual.spawn.collision_props) == 1
    assert isinstance(visual.spawn.collision_props[0], UsdPhysicsCollisionCfg)
    assert visual.spawn.collision_props[0].collision_enabled is False


def test_seattle_table_contact_surface_matches_stack_physics() -> None:
    """Pin the aligned hidden slab and every Newton contact-material value."""
    _, contact_surface = make_seattle_table_scene_assets()

    assert contact_surface.prim_path == "{ENV_REGEX_NS}/TableContactSurface"
    assert tuple(contact_surface.init_state.pos) == (0.3439, 0.0, -0.02)
    assert isinstance(contact_surface.spawn, sim_utils.CuboidCfg)
    assert tuple(contact_surface.spawn.size) == (1.28, 0.91, 0.04)
    assert contact_surface.spawn.visible is False
    assert isinstance(contact_surface.spawn.rigid_props, RigidBodyBaseCfg)
    assert contact_surface.spawn.rigid_props.kinematic_enabled is True
    assert isinstance(contact_surface.spawn.collision_props, CollisionBaseCfg)
    assert contact_surface.spawn.collision_props.contact_offset == 0.0
    assert contact_surface.spawn.collision_props.rest_offset == 0.0

    material = contact_surface.spawn.physics_material
    assert isinstance(material, NewtonMaterialPropertiesCfg)
    assert material.static_friction == 1.0
    assert material.dynamic_friction == 0.8
    assert material.restitution == 0.0
    assert material.torsional_friction == 0.002
    assert material.rolling_friction == 0.0001
    assert material.contact_stiffness == 1.0e4
    assert material.contact_damping == 200.0


def test_seattle_table_factory_honors_prim_paths_and_returns_fresh_cfgs() -> None:
    """Allow variant-local paths without sharing mutable configuration objects."""
    visual, contact_surface = make_seattle_table_scene_assets("/World/VisualTable", "/World/TableCollision")
    next_visual, next_contact_surface = make_seattle_table_scene_assets()

    assert visual.prim_path == "/World/VisualTable"
    assert contact_surface.prim_path == "/World/TableCollision"
    assert visual is not next_visual
    assert contact_surface is not next_contact_surface
    assert visual.spawn is not next_visual.spawn
    assert contact_surface.spawn is not next_contact_surface.spawn


def test_seattle_table_binds_only_a_verified_closure_entrypoint(tmp_path: Path) -> None:
    visual, _ = make_seattle_table_scene_assets()
    root = tmp_path / "closure"
    closure = FrankaRJ45AssetClosure(
        root=root,
        franka_usd_path=root / "franka.usda",
        seattle_table_usd_path=root / "table.usd",
        tree_sha256=FRANKA_RJ45_ASSET_CLOSURE_TREE_SHA256,
    )

    configure_seattle_table_external_asset(visual, closure)

    assert str(visual.spawn.usd_path) == str(closure.seattle_table_usd_path)
    configure_seattle_table_external_asset(visual)
    assert str(visual.spawn.usd_path) == FRANKA_RJ45_SEATTLE_TABLE_LOGICAL_URI

    wrong_closure = FrankaRJ45AssetClosure(
        root=root,
        franka_usd_path=root / "franka.usda",
        seattle_table_usd_path=root / "table.usd",
        tree_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="wrong external-asset closure"):
        configure_seattle_table_external_asset(visual, wrong_closure)
