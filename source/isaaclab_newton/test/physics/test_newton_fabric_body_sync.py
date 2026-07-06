# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton USD/Fabric body binding and transform synchronization."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from isaaclab.app import AppLauncher

# Launch Isaac Sim before importing Newton modules so USD schema bindings are initialized.
simulation_app = AppLauncher(headless=True).app

from isaaclab_newton.physics import NewtonManager

from pxr import Usd, UsdGeom


class _FakeAttribute:
    def __init__(self, value_type, custom):
        self.value_type = value_type
        self.custom = custom
        self.value = None

    def Set(self, value):
        self.value = value


class _FakeMatrixAttribute:
    def __init__(self, valid):
        self.valid = valid

    def IsValid(self):
        return self.valid


class _FakePrim:
    def __init__(self, valid=True, hierarchy_attrs_valid=True):
        self.valid = valid
        self.attributes = {}
        self.applied_schemas = []
        self.created_world_matrix_attrs = 0
        self.created_local_matrix_attrs = 0
        self.hierarchy_world_attr = _FakeMatrixAttribute(hierarchy_attrs_valid)
        self.hierarchy_local_attr = _FakeMatrixAttribute(hierarchy_attrs_valid)
        self.set_world_xform_from_usd = 0

    def IsValid(self):
        return self.valid

    def CreateAttribute(self, name, value_type, custom=False):
        self.attributes[name] = _FakeAttribute(value_type, custom)
        return self.attributes[name]

    def GetAttribute(self, name):
        return self.attributes[name]

    def AddAppliedSchema(self, schema):
        self.applied_schemas.append(schema)


class _FakeStage:
    def __init__(self, prims=None):
        self.prims = prims or {}
        self.defined_prims = []

    def GetPrimAtPath(self, path):
        return self.prims.get(path, _FakePrim(valid=False))

    def DefinePrim(self, path, prim_type):
        prim = _FakePrim()
        self.prims[path] = prim
        self.defined_prims.append((path, prim_type))
        return prim


class _FakeXformable:
    def __init__(self, prim):
        self.prim = prim

    def SetWorldXformFromUsd(self):
        self.prim.set_world_xform_from_usd += 1

    def GetFabricHierarchyWorldMatrixAttr(self):
        return self.prim.hierarchy_world_attr

    def CreateFabricHierarchyWorldMatrixAttr(self):
        self.prim.created_world_matrix_attrs += 1
        self.prim.hierarchy_world_attr.valid = True

    def GetFabricHierarchyLocalMatrixAttr(self):
        return self.prim.hierarchy_local_attr

    def CreateFabricHierarchyLocalMatrixAttr(self):
        self.prim.created_local_matrix_attrs += 1
        self.prim.hierarchy_local_attr.valid = True


class _FakeFabricHierarchy:
    def __init__(self):
        self.update_world_xforms_count = 0
        self.reset_xform_stacks = {}

    def update_world_xforms(self):
        self.update_world_xforms_count += 1

    def set_reset_xform_stack(self, path, enabled):
        self.reset_xform_stacks[path] = enabled


class _FakeRt:
    Xformable = _FakeXformable


class _FakeValueTypeNames:
    UInt = "UInt"


class _FakeSdf:
    ValueTypeNames = _FakeValueTypeNames
    Path = staticmethod(str)


class _FakeUsdrt:
    Rt = _FakeRt
    Sdf = _FakeSdf


def test_initialize_fabric_body_prims_uses_existing_fabric_prim():
    prim = _FakePrim()
    stage = _FakeStage({"/World/envs/env_0/Robot/base": prim})
    fabric_hierarchy = _FakeFabricHierarchy()

    NewtonManager._initialize_fabric_body_prims(
        stage, fabric_hierarchy, _FakeUsdrt, [("/World/envs/env_0/Robot/base", 3)]
    )

    assert stage.defined_prims == []
    assert prim.set_world_xform_from_usd == 1
    assert prim.created_world_matrix_attrs == 0
    assert prim.created_local_matrix_attrs == 0
    assert prim.GetAttribute("newton:index").value_type == "UInt"
    assert prim.GetAttribute("newton:index").custom is True
    assert prim.GetAttribute("newton:index").value == 3
    assert prim.applied_schemas == []
    assert fabric_hierarchy.reset_xform_stacks == {"/World/envs/env_0/Robot/base": True}
    assert fabric_hierarchy.update_world_xforms_count == 1


def test_initialize_fabric_body_prims_materializes_missing_hierarchy_attrs():
    prim = _FakePrim(hierarchy_attrs_valid=False)
    stage = _FakeStage({"/World/envs/env_0/Robot/base": prim})
    fabric_hierarchy = _FakeFabricHierarchy()

    NewtonManager._initialize_fabric_body_prims(
        stage, fabric_hierarchy, _FakeUsdrt, [("/World/envs/env_0/Robot/base", 3)]
    )

    assert prim.created_world_matrix_attrs == 1
    assert prim.created_local_matrix_attrs == 1
    assert prim.set_world_xform_from_usd == 1
    assert fabric_hierarchy.update_world_xforms_count == 1


def test_initialize_fabric_xform_hierarchy_includes_instance_proxy_meshes():
    usd_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(usd_stage, "/Prototype")
    UsdGeom.Mesh.Define(usd_stage, "/Prototype/visual")
    UsdGeom.Xform.Define(usd_stage, "/World")
    instance = UsdGeom.Xform.Define(usd_stage, "/World/RobotVisual").GetPrim()
    instance.GetReferences().AddInternalReference("/Prototype")
    instance.SetInstanceable(True)
    assert usd_stage.GetPrimAtPath("/World/RobotVisual/visual").IsInstanceProxy()

    paths = ("/World", "/World/RobotVisual", "/World/RobotVisual/visual")
    fabric_prims = {path: _FakePrim(hierarchy_attrs_valid=False) for path in paths}
    fabric_hierarchy = _FakeFabricHierarchy()

    NewtonManager._initialize_fabric_xform_hierarchy(
        usd_stage,
        _FakeStage(fabric_prims),
        fabric_hierarchy,
        _FakeUsdrt,
    )

    for path in paths:
        prim = fabric_prims[path]
        assert prim.created_world_matrix_attrs == 1, path
        assert prim.created_local_matrix_attrs == 1, path
        assert prim.set_world_xform_from_usd == 1, path
    assert fabric_hierarchy.update_world_xforms_count == 1


@pytest.mark.parametrize("prim_path", ["/World/envs/env_1/SpillFloor", "solver_only_body"])
def test_initialize_fabric_body_prims_skips_solver_only_bodies(prim_path):
    stage = _FakeStage()
    fabric_hierarchy = _FakeFabricHierarchy()

    NewtonManager._initialize_fabric_body_prims(stage, fabric_hierarchy, _FakeUsdrt, [(prim_path, 7)])

    assert stage.defined_prims == []
    assert stage.prims == {}
    assert fabric_hierarchy.update_world_xforms_count == 1


def test_sync_transforms_propagates_cpu_fallback_after_fabric_writes(monkeypatch):
    """Older Kit versions fall back to CPU propagation after local-matrix writes."""
    from isaaclab.physics import PhysicsManager

    events = []

    class _Selection:
        def GetCount(self):
            return 1

        def PrepareForReuse(self):
            events.append("prepare")

    class _FabricStage:
        def __init__(self):
            self.selection = _Selection()

        def GetFabricId(self):
            return SimpleNamespace(id=1)

        def GetStageIdAsStageId(self):
            return 2

        def SelectPrims(self, **kwargs):
            events.append("select")
            return self.selection

    class _HierarchyApi:
        def get_fabric_hierarchy(self, fabric_id, stage_id):
            return SimpleNamespace(update_world_xforms=lambda: events.append("update"))

    fake_usdrt = SimpleNamespace(
        Sdf=SimpleNamespace(ValueTypeNames=SimpleNamespace(Matrix4d="Matrix4d", UInt="UInt")),
        Usd=SimpleNamespace(Access=SimpleNamespace(ReadWrite="ReadWrite", Read="Read")),
        hierarchy=SimpleNamespace(IFabricHierarchy=_HierarchyApi),
    )
    stage = _FabricStage()
    fabric_arrays = {
        "omni:fabric:localMatrix": SimpleNamespace(shape=(1,)),
        "newton:index": SimpleNamespace(shape=(1,)),
    }

    monkeypatch.setitem(sys.modules, "usdrt", fake_usdrt)
    monkeypatch.setattr(PhysicsManager, "_device", "cpu", raising=False)
    monkeypatch.setattr(NewtonManager, "_usdrt_stage", stage, raising=False)
    monkeypatch.setattr(NewtonManager, "_model", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", SimpleNamespace(body_q=object()), raising=False)
    monkeypatch.setattr(NewtonManager, "_transforms_dirty", True, raising=False)
    monkeypatch.setattr(NewtonManager, "_newton_index_attr", "newton:index", raising=False)
    monkeypatch.setattr(
        "isaaclab_newton.physics.newton_manager.wp.fabricarray",
        lambda selection, attrib: events.append(f"array:{attrib}") or fabric_arrays[attrib],
    )
    monkeypatch.setattr(
        "isaaclab_newton.physics.newton_manager.wp.launch",
        lambda *args, **kwargs: events.append("launch"),
    )
    monkeypatch.setattr(
        "isaaclab_newton.physics.newton_manager.wp.synchronize_device",
        lambda *args, **kwargs: events.append("synchronize"),
    )

    NewtonManager.sync_transforms_to_usd()

    assert events == [
        "select",
        "prepare",
        "array:omni:fabric:localMatrix",
        "array:newton:index",
        "launch",
        "synchronize",
        "update",
    ]


def test_sync_transforms_uses_public_gpu_hierarchy_after_local_writes(monkeypatch):
    """Current Kit propagates Newton local matrices through the public GPU API."""
    from isaaclab.physics import PhysicsManager

    events = []

    class _Selection:
        def GetCount(self):
            return 1

        def PrepareForReuse(self):
            events.append("prepare")

    class _FabricStage:
        def GetFabricId(self):
            return SimpleNamespace(id=17)

        def GetStageIdAsStageId(self):
            return 2

        def SelectPrims(self, **kwargs):
            events.append("select")
            return _Selection()

    class _Hierarchy:
        def update_world_xforms_gpu(self, no_structural_changes_hint):
            events.append(f"gpu_update:{no_structural_changes_hint}")
            return True

        def update_world_xforms(self):
            events.append("cpu_update")

    class _HierarchyApi:
        def get_fabric_hierarchy(self, fabric_id, stage_id):
            return _Hierarchy()

    fake_usdrt = SimpleNamespace(
        Sdf=SimpleNamespace(ValueTypeNames=SimpleNamespace(Matrix4d="Matrix4d", UInt="UInt")),
        Usd=SimpleNamespace(Access=SimpleNamespace(ReadWrite="ReadWrite", Read="Read")),
        hierarchy=SimpleNamespace(IFabricHierarchy=_HierarchyApi),
    )
    fabric_arrays = {
        "omni:fabric:localMatrix": SimpleNamespace(shape=(1,)),
        "newton:index": SimpleNamespace(shape=(1,)),
    }

    monkeypatch.setitem(sys.modules, "usdrt", fake_usdrt)
    monkeypatch.setattr(PhysicsManager, "_device", "cpu", raising=False)
    monkeypatch.setattr(NewtonManager, "_usdrt_stage", _FabricStage(), raising=False)
    monkeypatch.setattr(NewtonManager, "_model", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", SimpleNamespace(body_q=object()), raising=False)
    monkeypatch.setattr(NewtonManager, "_newton_index_attr", "newton:index", raising=False)
    monkeypatch.setattr(
        "isaaclab_newton.physics.newton_manager.wp.fabricarray",
        lambda selection, attrib: events.append(f"array:{attrib}") or fabric_arrays[attrib],
    )
    monkeypatch.setattr(
        "isaaclab_newton.physics.newton_manager.wp.launch",
        lambda *args, **kwargs: events.append("launch"),
    )
    monkeypatch.setattr(
        "isaaclab_newton.physics.newton_manager.wp.synchronize_device",
        lambda *args, **kwargs: events.append("synchronize"),
    )

    for _ in range(2):
        monkeypatch.setattr(NewtonManager, "_transforms_dirty", True, raising=False)
        NewtonManager.sync_transforms_to_usd()

    assert events.count("gpu_update:False") == 2
    assert "cpu_update" not in events
    for update_index in (index for index, event in enumerate(events) if event == "gpu_update:False"):
        assert events[update_index - 1] == "synchronize"
