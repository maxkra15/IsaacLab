# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD authoring tests for slung-load PhysicsAttachment spawners."""

import pytest

from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils

from isaaclab_tasks.contrib.drone_slung_load.mdp import spawners
from isaaclab_tasks.contrib.drone_slung_load.mdp.spawners import DroneCuboidCfg, PhysicsAttachmentCfg
from isaaclab_tasks.contrib.drone_slung_load.system import DRONE_DIAGONAL_INERTIA, ROTOR_ARM_LENGTH, ROTOR_HEIGHT

pytestmark = pytest.mark.unit


@pytest.fixture
def stage():
    return sim_utils.create_new_stage()


def test_spawn_physics_attachment_authors_hard_ball_joint(stage):
    UsdGeom.Xform.Define(stage, "/World/Robot")
    UsdGeom.Xform.Define(stage, "/World/Cable/geometry/mesh")
    cfg = PhysicsAttachmentCfg(
        src0="../Cable/geometry/mesh",
        src1="../Robot",
        indices0=(0,),
        coords1=((0.0, 0.0, -0.02),),
    )

    root_prim = cfg.func("/World/DroneCableAttach", cfg)
    attach_prim = stage.GetPrimAtPath("/World/DroneCableAttach/attachment")

    assert root_prim.GetTypeName() == "Xform"
    assert attach_prim.GetTypeName() == "PhysicsAttachment"
    assert [str(path) for path in attach_prim.GetRelationship("physics:src0").GetTargets()] == [
        "/World/Cable/geometry/mesh"
    ]
    assert [str(path) for path in attach_prim.GetRelationship("physics:src1").GetTargets()] == ["/World/Robot"]
    assert attach_prim.GetAttribute("physics:type0").Get() == "point"
    assert attach_prim.GetAttribute("physics:type1").Get() == "xform"
    assert list(attach_prim.GetAttribute("physics:indices0").Get()) == [0]
    assert tuple(attach_prim.GetAttribute("physics:coords1").Get()[0]) == pytest.approx((0.0, 0.0, -0.02))
    # Unauthored stiffness is required for Newton to import a hard ball joint.
    assert not attach_prim.HasAttribute("physics:stiffness")
    assert attach_prim.GetAttribute("physics:attachmentEnabled").Get() is True


def test_spawn_drone_fallback_keeps_one_body_collider_and_ordered_rotors(stage, monkeypatch):
    monkeypatch.setattr(spawners, "_resolve_crazyflie_asset", lambda: None)
    cfg = DroneCuboidCfg(
        size=(0.08, 0.08, 0.025),
        rigid_props=sim_utils.RigidBodyBaseCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.305),
        collision_props=sim_utils.CollisionBaseCfg(),
        diagonal_inertia=DRONE_DIAGONAL_INERTIA,
        principal_axes=(1.0, 0.0, 0.0, 0.0),
    )

    root = cfg.func("/World/Robot", cfg)

    subtree = list(UsdGeom.Imageable(root).GetPrim().GetStage().Traverse())
    robot_subtree = [prim for prim in subtree if str(prim.GetPath()).startswith("/World/Robot")]
    rigid_prims = [prim for prim in robot_subtree if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    collision_prims = [prim for prim in robot_subtree if prim.HasAPI(UsdPhysics.CollisionAPI)]
    body_visual = stage.GetPrimAtPath("/World/Robot/visuals/body")
    rotor_prims = [stage.GetPrimAtPath(f"/World/Robot/visuals/rotor_{index}") for index in range(4)]
    mass_api = UsdPhysics.MassAPI(root)

    assert [str(prim.GetPath()) for prim in rigid_prims] == ["/World/Robot"]
    assert [str(prim.GetPath()) for prim in collision_prims] == ["/World/Robot/geometry/mesh"]
    assert not root.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert all("Physx" not in schema for prim in robot_subtree for schema in prim.GetAppliedSchemas())
    assert UsdGeom.Imageable(collision_prims[0]).ComputeVisibility() == UsdGeom.Tokens.invisible
    assert body_visual.IsValid() and body_visual.GetTypeName() == "Cube"
    assert not body_visual.HasAPI(UsdPhysics.CollisionAPI)
    assert all(prim.IsValid() and prim.GetTypeName() == "Cylinder" for prim in rotor_prims)
    assert all(not prim.HasAPI(UsdPhysics.CollisionAPI) for prim in rotor_prims)
    positions = [tuple(UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Get()) for prim in rotor_prims]
    assert positions == pytest.approx(
        [
            (ROTOR_ARM_LENGTH, -ROTOR_ARM_LENGTH, ROTOR_HEIGHT),
            (-ROTOR_ARM_LENGTH, -ROTOR_ARM_LENGTH, ROTOR_HEIGHT),
            (-ROTOR_ARM_LENGTH, ROTOR_ARM_LENGTH, ROTOR_HEIGHT),
            (ROTOR_ARM_LENGTH, ROTOR_ARM_LENGTH, ROTOR_HEIGHT),
        ]
    )
    assert tuple(mass_api.GetDiagonalInertiaAttr().Get()) == pytest.approx(DRONE_DIAGONAL_INERTIA)
    principal_axes = mass_api.GetPrincipalAxesAttr().Get()
    assert principal_axes.GetReal() == pytest.approx(1.0)
    assert tuple(principal_axes.GetImaginary()) == pytest.approx((0.0, 0.0, 0.0))


def test_spawn_drone_references_colored_visual_only_crazyflie_at_rotor_sites(stage):
    asset_path = spawners._resolve_crazyflie_asset()
    if asset_path is None:
        pytest.skip("Installed Newton package does not provide the optional Crazyflie example asset.")
    cfg = DroneCuboidCfg(
        size=(0.08, 0.08, 0.025),
        rigid_props=sim_utils.RigidBodyBaseCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.305),
        collision_props=sim_utils.CollisionBaseCfg(),
    )

    root = cfg.func("/World/Robot", cfg)

    robot_subtree = [prim for prim in stage.Traverse() if str(prim.GetPath()).startswith("/World/Robot")]
    visual_subtree = [prim for prim in robot_subtree if str(prim.GetPath()).startswith("/World/Robot/visuals")]
    rigid_prims = [prim for prim in robot_subtree if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    collision_prims = [prim for prim in robot_subtree if prim.HasAPI(UsdPhysics.CollisionAPI)]
    model = stage.GetPrimAtPath("/World/Robot/visuals/crazyflie")
    meshes = [prim for prim in Usd.PrimRange(model) if prim.IsA(UsdGeom.Mesh)]

    assert [str(prim.GetPath()) for prim in rigid_prims] == ["/World/Robot"]
    assert [str(prim.GetPath()) for prim in collision_prims] == ["/World/Robot/geometry/mesh"]
    assert not root.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert model.IsValid() and len(meshes) == 12
    assert all(
        not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        and not prim.HasAPI(UsdPhysics.CollisionAPI)
        and not prim.HasAPI(UsdPhysics.MassAPI)
        and not prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        and not prim.IsA(UsdPhysics.Joint)
        for prim in visual_subtree
    )
    assert all(UsdGeom.Mesh(prim).GetDisplayColorAttr().Get() for prim in meshes)

    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(model)
    midpoint = bbox.ComputeAlignedRange().GetMidpoint()
    assert tuple(midpoint)[:2] == pytest.approx((0.0, 0.0), abs=5.0e-4)
    assert abs(midpoint[2]) < 0.02

    rotor_paths = [f"/World/Robot/visuals/crazyflie/{path}" for path in spawners._CRAZYFLIE_ROTOR_PATHS]
    xform_cache = UsdGeom.XformCache()
    rotor_positions = [
        tuple(xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)).ExtractTranslation())
        for path in rotor_paths
    ]
    expected_positions = (
        (ROTOR_ARM_LENGTH, -ROTOR_ARM_LENGTH, ROTOR_HEIGHT),
        (-ROTOR_ARM_LENGTH, -ROTOR_ARM_LENGTH, ROTOR_HEIGHT),
        (-ROTOR_ARM_LENGTH, ROTOR_ARM_LENGTH, ROTOR_HEIGHT),
        (ROTOR_ARM_LENGTH, ROTOR_ARM_LENGTH, ROTOR_HEIGHT),
    )
    for position, expected in zip(rotor_positions, expected_positions, strict=True):
        assert position == pytest.approx(expected, abs=5.0e-4)
    rotor_colors = [
        tuple(UsdGeom.Mesh(stage.GetPrimAtPath(path)).GetDisplayColorAttr().Get()[0]) for path in rotor_paths
    ]
    for color, expected in zip(rotor_colors, cfg.rotor_colors, strict=True):
        assert color == pytest.approx(expected)


@pytest.mark.parametrize(
    ("diagonal_inertia", "principal_axes", "error_match"),
    [
        ((8.0e-4, -8.0e-4, 1.4e-3), None, "finite positive"),
        ((8.0e-4, 8.0e-4, 2.0e-3), None, "triangle inequality"),
        ((8.0e-4, 8.0e-4, 1.4e-3), (2.0, 0.0, 0.0, 0.0), "unit"),
        (None, (1.0, 0.0, 0.0, 0.0), "requires diagonal_inertia"),
    ],
)
def test_spawn_drone_rejects_invalid_explicit_inertia(
    stage, monkeypatch, diagonal_inertia, principal_axes, error_match
):
    monkeypatch.setattr(spawners, "_resolve_crazyflie_asset", lambda: None)
    cfg = DroneCuboidCfg(
        size=(0.08, 0.08, 0.025),
        rigid_props=sim_utils.RigidBodyBaseCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.305),
        collision_props=sim_utils.CollisionBaseCfg(),
        diagonal_inertia=diagonal_inertia,
        principal_axes=principal_axes,
    )

    with pytest.raises(ValueError, match=error_match):
        cfg.func("/World/Robot", cfg)
