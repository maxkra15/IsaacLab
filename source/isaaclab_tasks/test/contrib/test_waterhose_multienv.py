# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-environment regressions for the native Waterhose cable.

These tests intentionally use the task's coupled Newton configuration and external
asset bundle. They exercise the native ``CableObject`` world ownership, connector
compound, masked reset, and per-curve Fabric synchronization contracts.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from isaaclab.app import AppLauncher

# Launch Isaac Sim before importing Newton, USD, or task modules.
try:
    simulation_app = AppLauncher(headless=True, enable_cameras=True).app
except ImportError:
    pytest.skip("The full Isaac Sim runtime is required", allow_module_level=True)

import torch
import warp as wp
from isaaclab_newton.cloner import newton_builder_world_hook
from isaaclab_newton.physics import NewtonManager

from pxr import Usd, UsdGeom
from usdrt import Gf, Rt

import isaaclab.sim as sim_utils
from isaaclab.assets import BaseCableObject, CableObjectCfg
from isaaclab.envs.mdp.events import reset_scene_to_default
from isaaclab.scene import InteractiveScene
from isaaclab.sim import build_simulation_context
from isaaclab.sim.utils.queries import has_deformable_curve_api
from isaaclab.utils import math as math_utils

from isaaclab_tasks.contrib.waterhose import waterhose_env_cfg
from isaaclab_tasks.contrib.waterhose.cable import WaterhoseCableBuilderExtension, WaterhoseCableObject

pytestmark = pytest.mark.isaacsim_ci


def _require_native_waterhose_cable() -> None:
    """Require the native cable configuration and external Waterhose assets."""

    cable_cfg = waterhose_env_cfg.WaterhoseSceneCfg().cable1
    if not isinstance(cable_cfg, CableObjectCfg):
        pytest.skip("Waterhose has not yet migrated to Isaac Lab's native CableObject API")

    required_assets = (
        waterhose_env_cfg._FRIDGE_USD,
        waterhose_env_cfg._RBY1_USD,
        waterhose_env_cfg._CABLE1_USD,
        waterhose_env_cfg._PLUG_USD,
        waterhose_env_cfg._FRIDGE_ROBOT_COLLISION_PROXY_USD,
        waterhose_env_cfg._FRIDGE_CABLE_COLLISION_PROXY_USD,
    )
    missing_assets = [path for path in required_assets if not Path(path).is_file()]
    if missing_assets:
        pytest.skip(f"External Waterhose asset bundle is not installed: {missing_assets}")


def _cuda_device() -> str:
    """Prefer the user-requested second GPU while remaining portable to one-GPU CI."""

    device_count = wp.get_cuda_device_count()
    if device_count == 0:
        pytest.skip("A CUDA device is required for the coupled Waterhose simulation")
    return "cuda:1" if device_count > 1 else "cuda:0"


@contextmanager
def _waterhose_scene(num_envs: int):
    """Build the production coupled scene without opening a visualizer window."""

    _require_native_waterhose_cable()
    cfg = waterhose_env_cfg.WaterhoseEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.sim.device = _cuda_device()
    cfg.sim.visualizer_cfgs = []

    # Graph capture is orthogonal to environment ownership and makes this focused
    # regression needlessly expensive.  All production solver/contact parameters
    # and the real CouplerProxy configuration remain unchanged.
    cfg.sim.physics.use_cuda_graph = False
    extension = WaterhoseCableBuilderExtension(cfg.scene.cable1)

    with newton_builder_world_hook(extension.add_to_builder):
        with build_simulation_context(sim_cfg=cfg.sim) as sim:
            sim._app_control_on_stop_handle = None
            scene = InteractiveScene(cfg.scene)
            sim.register_interactive_scene(scene)
            try:
                sim.reset()
                cable = scene["cable1"]
                assert isinstance(cable, WaterhoseCableObject)
                cable.bind_builder_extension(extension)
                scene.reset()
                scene.update(0.0)
                yield sim, scene
            finally:
                sim.register_interactive_scene(None)


def _segment_body_ids(cable: BaseCableObject) -> torch.Tensor:
    """Return segment body IDs through the native cable articulation view."""

    model = NewtonManager.get_model()
    root_ids = wp.to_torch(cable.root_view.get_attribute("joint_parent", model)[:, 0, 0]).long().unsqueeze(1)
    link_ids = wp.to_torch(cable.root_view.get_attribute("joint_child", model)[:, 0]).long()
    return torch.cat((root_ids, link_ids), dim=1)


def _connector_shape_ids(model, num_envs: int) -> torch.Tensor:
    """Resolve one task-authored connector collision shape per concrete environment."""

    connector_token = waterhose_env_cfg._CONNECTOR_SHAPE_TOKEN
    shapes_by_env: dict[int, list[int]] = {env_id: [] for env_id in range(num_envs)}
    for shape_id, label in enumerate(model.shape_label):
        if not label or connector_token not in label:
            continue
        match = re.search(r"/World/envs/env_(\d+)/Cable1(?:[/_]|$)", label)
        if match is None:
            continue
        env_id = int(match.group(1))
        if env_id in shapes_by_env:
            shapes_by_env[env_id].append(shape_id)

    invalid = {env_id: shape_ids for env_id, shape_ids in shapes_by_env.items() if len(shape_ids) != 1}
    assert not invalid, f"Expected one connector shape per Waterhose environment, found {invalid}"
    return torch.tensor(
        [shapes_by_env[env_id][0] for env_id in range(num_envs)],
        dtype=torch.long,
        device=wp.to_torch(model.shape_body).device,
    )


def _curve_path(env_id: int) -> str:
    """Resolve the one schema-authored cable curve beneath a concrete environment."""

    stage = sim_utils.get_current_stage()
    root = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Cable1")
    assert root.IsValid(), f"Waterhose cable prim is missing for environment {env_id}"
    curves = [
        prim.GetPath().pathString
        for prim in Usd.PrimRange(root)
        if prim.IsA(UsdGeom.BasisCurves) and has_deformable_curve_api(prim)
    ]
    assert len(curves) == 1, f"Expected one native cable curve in env {env_id}, found {curves}"
    return curves[0]


def _fabric_curve_points_world(curve_path: str) -> torch.Tensor:
    """Read the curve points consumed by Kit/RTX and transform them to world space."""

    stage = sim_utils.get_current_stage(fabric=True)
    assert stage is not None, "The rendering-side Fabric stage is unavailable"
    prim = stage.GetPrimAtPath(curve_path)
    assert prim.IsValid(), f"Fabric cable curve is missing: {curve_path}"

    points_attr = prim.GetAttribute("points")
    points_attr.SyncDataToCpu()
    points = points_attr.Get()
    world_matrix = Rt.Xformable(prim).GetFabricHierarchyWorldMatrixAttr().Get()
    assert world_matrix is not None, f"Fabric curve has no world matrix: {curve_path}"
    return torch.tensor(
        [[float(component) for component in world_matrix.Transform(Gf.Vec3d(*map(float, point)))] for point in points],
        dtype=torch.float32,
    )


@pytest.mark.parametrize("num_envs", [1, 2, 4])
def test_waterhose_cable_instances_are_isolated_by_newton_world(num_envs):
    """Every cloned cable must bind to and advance in its matching Newton world."""

    with _waterhose_scene(num_envs) as (sim, scene):
        cable = scene["cable1"]
        assert isinstance(cable, BaseCableObject)
        assert scene.cable_objects["cable1"] is cable
        assert cable.num_instances == num_envs
        assert cable.data.segment_pose_w.torch.shape == (num_envs, cable.num_segments, 7)

        model = NewtonManager.get_model()
        assert model.world_count == num_envs
        body_ids = _segment_body_ids(cable)
        body_world = wp.to_torch(model.body_world)[body_ids]
        expected_world = (
            torch.arange(num_envs, device=body_world.device, dtype=body_world.dtype).unsqueeze(1).expand_as(body_world)
        )
        torch.testing.assert_close(body_world, expected_world, rtol=0.0, atol=0.0)

        # Clones must begin with identical cable-local state.  This catches an env-0
        # cable populated from the last environment even before rendering is involved.
        initial_pose = cable.data.segment_pose_w.torch.clone()
        relative_position = initial_pose[..., :3] - scene.env_origins[:, None, :]
        torch.testing.assert_close(
            relative_position,
            relative_position[0:1].expand_as(relative_position),
            rtol=0.0,
            atol=1.0e-5,
        )

        # Give every world the same nonzero cable velocity.  Each instance must be
        # advanced by the coupled VBD entry, and cloned trajectories must agree in
        # their environment-local frame.
        velocity = torch.zeros_like(cable.data.segment_velocity_w.torch)
        velocity[..., 2] = -0.05
        cable.write_segment_velocity_to_sim_index(segment_velocity=velocity)
        before_step = cable.data.segment_pose_w.torch.clone()
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim.cfg.dt)
        after_step = cable.data.segment_pose_w.torch.clone()

        assert torch.isfinite(after_step).all()
        displacement = torch.linalg.vector_norm(after_step[..., :3] - before_step[..., :3], dim=-1).amax(dim=1)
        assert torch.all(displacement > 1.0e-8), f"At least one cable world did not advance: {displacement}"
        relative_after = after_step[..., :3] - scene.env_origins[:, None, :]
        torch.testing.assert_close(
            relative_after,
            relative_after[0:1].expand_as(relative_after),
            rtol=1.0e-4,
            atol=2.0e-4,
        )


def test_waterhose_connector_is_compound_with_the_matching_cable_head():
    """Each labeled connector shape must belong to its own native cable-head body."""

    with _waterhose_scene(4) as (_, scene):
        cable = scene["cable1"]
        model = NewtonManager.get_model()
        segment_body_ids = _segment_body_ids(cable)
        head_body_ids = segment_body_ids[:, 0]
        connector_shape_ids = _connector_shape_ids(model, num_envs=4)
        connector_body_ids = wp.to_torch(model.shape_body)[connector_shape_ids]
        torch.testing.assert_close(connector_body_ids, head_body_ids, rtol=0.0, atol=0.0)
        assert torch.unique(connector_body_ids).numel() == 4

        body_world = wp.to_torch(model.body_world)[connector_body_ids]
        expected_world = torch.arange(4, device=body_world.device, dtype=body_world.dtype)
        torch.testing.assert_close(body_world, expected_world, rtol=0.0, atol=0.0)

        # Connector collision geometry is a shape on the cable head.  Its local
        # transform must be cloned identically, and the resulting world pose must
        # differ only by the InteractiveScene environment origin.
        connector_local_pose = wp.to_torch(model.shape_transform)[connector_shape_ids]
        torch.testing.assert_close(
            connector_local_pose,
            connector_local_pose[0:1].expand_as(connector_local_pose),
            rtol=0.0,
            atol=1.0e-6,
        )
        head_pose = wp.to_torch(NewtonManager.get_state_0().body_q)[connector_body_ids]
        connector_position, connector_orientation = math_utils.combine_frame_transforms(
            head_pose[:, :3],
            head_pose[:, 3:7],
            connector_local_pose[:, :3],
            connector_local_pose[:, 3:7],
        )
        connector_position_local = connector_position - scene.env_origins
        torch.testing.assert_close(
            connector_position_local,
            connector_position_local[0:1].expand_as(connector_position_local),
            rtol=0.0,
            atol=1.0e-5,
        )
        orientation_alignment = torch.abs(torch.sum(connector_orientation * connector_orientation[0:1], dim=-1))
        torch.testing.assert_close(
            orientation_alignment,
            torch.ones_like(orientation_alignment),
            rtol=0.0,
            atol=1.0e-5,
        )


def test_waterhose_native_cable_masked_reset_does_not_touch_other_worlds():
    """A selected reset must restore only those cable instances."""

    with _waterhose_scene(4) as (_, scene):
        cable = scene["cable1"]
        default_pose = cable.data.default_segment_pose_w.torch.clone()
        default_velocity = cable.data.default_segment_velocity_w.torch.clone()

        translation = torch.tensor(
            [[0.02, 0.00, 0.01], [0.04, 0.01, 0.02], [0.06, 0.02, 0.03], [0.08, 0.03, 0.04]],
            dtype=default_pose.dtype,
            device=default_pose.device,
        )
        disturbed_pose = default_pose.clone()
        disturbed_pose[..., :3] += translation[:, None, :]
        disturbed_velocity = default_velocity.clone()
        disturbed_velocity[..., :3] = translation[:, None, :] * 10.0
        cable.write_segment_pose_to_sim_index(segment_pose=disturbed_pose)
        cable.write_segment_velocity_to_sim_index(segment_velocity=disturbed_velocity)

        reset_env_ids = torch.tensor([0, 2], dtype=torch.long, device=scene.device)
        reset_scene_to_default(SimpleNamespace(scene=scene), reset_env_ids)

        selected = [0, 2]
        untouched = [1, 3]
        torch.testing.assert_close(cable.data.segment_pose_w.torch[selected], default_pose[selected])
        torch.testing.assert_close(cable.data.segment_velocity_w.torch[selected], default_velocity[selected])
        torch.testing.assert_close(cable.data.segment_pose_w.torch[untouched], disturbed_pose[untouched])
        torch.testing.assert_close(cable.data.segment_velocity_w.torch[untouched], disturbed_velocity[untouched])


def test_waterhose_fabric_sync_updates_each_concrete_curve_independently():
    """Distinct cable teleports must reach the matching env's Kit/RTX curve."""

    with _waterhose_scene(4) as (sim, scene):
        cable = scene["cable1"]
        curve_paths = [_curve_path(env_id) for env_id in range(4)]

        sim.render()
        wp.synchronize_device(sim.device)
        points_before = [_fabric_curve_points_world(path) for path in curve_paths]

        translations = torch.tensor(
            [[0.011, 0.000, 0.007], [0.023, 0.004, 0.013], [0.037, 0.009, 0.019], [0.053, 0.015, 0.029]],
            dtype=cable.data.segment_pose_w.torch.dtype,
            device=cable.data.segment_pose_w.torch.device,
        )
        translated_pose = cable.data.segment_pose_w.torch.clone()
        translated_pose[..., :3] += translations[:, None, :]
        cable.write_segment_pose_to_sim_index(segment_pose=translated_pose)
        cable.write_segment_velocity_to_sim_index(
            segment_velocity=torch.zeros_like(cable.data.segment_velocity_w.torch)
        )

        sim.render()
        wp.synchronize_device(sim.device)
        points_after = [_fabric_curve_points_world(path) for path in curve_paths]

        for env_id, (before, after) in enumerate(zip(points_before, points_after, strict=True)):
            expected_delta = translations[env_id].cpu().expand_as(after)
            torch.testing.assert_close(after - before, expected_delta, rtol=0.0, atol=2.0e-4)
