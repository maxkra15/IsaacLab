# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Seattle table visual and aligned native contact surface for Franka RJ45 scenes."""

from __future__ import annotations

from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.schemas import CollisionBaseCfg, RigidBodyBaseCfg, UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg

from .asset_provenance import (
    FRANKA_RJ45_ASSET_CLOSURE_TREE_SHA256,
    FRANKA_RJ45_SEATTLE_TABLE_LOGICAL_URI,
    FrankaRJ45AssetClosure,
)

SEATTLE_TABLE_USD_PATH = FRANKA_RJ45_SEATTLE_TABLE_LOGICAL_URI
"""Path-independent Seattle Lab table identity used by diagnostic configurations."""


def make_seattle_table_scene_assets(
    visual_prim_path: str = "{ENV_REGEX_NS}/Table",
    contact_prim_path: str = "{ENV_REGEX_NS}/TableContactSurface",
) -> tuple[AssetBaseCfg, AssetBaseCfg]:
    """Create the visual-only Seattle table and its aligned native contact slab.

    The imported table retains its authored visual materials, while all authored
    colliders are disabled recursively. The separate native cuboid places its
    top face at the visible tabletop height ``z=0`` and replaces the imported
    table's contact geometry.

    Args:
        visual_prim_path: Prim path for the imported Seattle table.
        contact_prim_path: Prim path for the invisible contact surface.

    Returns:
        The visual table and contact-surface configurations, in that order.
    """
    visual = AssetBaseCfg(
        prim_path=visual_prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.0),
            rot=(0.0, 0.0, 0.707, 0.707),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=SEATTLE_TABLE_USD_PATH,
            make_uninstanceable=True,
            rigid_props=[UsdPhysicsRigidBodyCfg(kinematic_enabled=True)],
            collision_props=[UsdPhysicsCollisionCfg(collision_enabled=False)],
        ),
    )
    contact_surface = AssetBaseCfg(
        prim_path=contact_prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3439, 0.0, -0.02)),
        spawn=sim_utils.CuboidCfg(
            size=(1.28, 0.91, 0.04),
            visible=False,
            rigid_props=RigidBodyBaseCfg(kinematic_enabled=True),
            collision_props=CollisionBaseCfg(contact_offset=0.0, rest_offset=0.0),
            physics_material=NewtonMaterialPropertiesCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
                torsional_friction=0.002,
                rolling_friction=0.0001,
                contact_stiffness=1.0e4,
                contact_damping=200.0,
            ),
        ),
    )
    return visual, contact_surface


def configure_seattle_table_external_asset(
    table_cfg: AssetBaseCfg,
    verified_closure: FrankaRJ45AssetClosure | None = None,
) -> None:
    """Use the logical table identity for diagnostics or a verified local entrypoint for production."""
    if not isinstance(table_cfg.spawn, sim_utils.UsdFileCfg):
        raise TypeError("The Seattle table external asset requires UsdFileCfg.")
    if verified_closure is None:
        table_cfg.spawn.usd_path = FRANKA_RJ45_SEATTLE_TABLE_LOGICAL_URI
        return
    if verified_closure.tree_sha256 != FRANKA_RJ45_ASSET_CLOSURE_TREE_SHA256:
        raise ValueError("Cannot bind a Seattle table asset from the wrong external-asset closure.")
    table_cfg.spawn.usd_path = str(verified_closure.seattle_table_usd_path)


__all__ = [
    "SEATTLE_TABLE_USD_PATH",
    "configure_seattle_table_external_asset",
    "make_seattle_table_scene_assets",
]
