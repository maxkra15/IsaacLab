# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualization regressions for the multi-world Franka Pour task."""

from __future__ import annotations

import os

import pytest

_RUNTIME_AVAILABLE = bool(os.environ.get("EXP_PATH"))
_RUNTIME_UNAVAILABLE_REASON = "Isaac Sim runtime is unavailable because EXP_PATH is not set."
_TEST_DEVICE = os.environ.get("ISAACLAB_TEST_DEVICE", "cuda:0")

if _RUNTIME_AVAILABLE:
    from isaaclab.app import AppLauncher

    # Launch Kit before importing simulation-dependent modules.
    app_launcher = AppLauncher(headless=True, device=_TEST_DEVICE)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import newton
    import numpy as np
    import torch
    import warp as wp
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_physx.sim.views import FabricFrameView
    from isaaclab_visualizers.kit import KitVisualizer, KitVisualizerCfg
    from isaaclab_visualizers.newton import NewtonVisualizer, NewtonVisualizerCfg

    from pxr import UsdGeom

    import isaaclab.sim as sim_utils

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import MPM_ENTRY, RIGID_ENTRY
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import FrankaPourEnvCfg

pytestmark = [pytest.mark.isaacsim_ci, pytest.mark.newton_ci]

_TASK_ID = "Isaac-Pour-Franka-v0"
_SCENE_PARTITION_ENV_VAR = "ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION"


def test_franka_pour_viewer_frames_first_environment():
    """The task viewer should interpret its camera pose relative to environment zero."""
    cfg = FrankaPourEnvCfg()

    assert cfg.viewer.origin_type == "env"
    assert cfg.viewer.env_index == 0


def _make_visualization_cfg():
    cfg = parse_env_cfg(_TASK_ID, device=_TEST_DEVICE, num_envs=2)
    cfg.seed = 37
    cfg.scene.env_spacing = 2.5
    cfg.decimation = 1
    cfg.num_substeps = 1
    cfg.sim.render_interval = 1
    # Keep the eager double-buffered manager on an even substep count, matching
    # the production configuration's stable public state bindings.
    cfg.sim.physics.num_substeps = 2
    cfg.sim.physics.use_cuda_graph = False

    entries = {entry.name: entry for entry in cfg.sim.physics.solver_cfg.entries}
    entries[RIGID_ENTRY].substeps = 1
    entries[MPM_ENTRY].solver_cfg.max_iterations = 2
    cfg.sim.visualizer_cfgs = [
        KitVisualizerCfg(headless=True, randomly_sample_visible_envs=False),
        NewtonVisualizerCfg(
            headless=True,
            show_particles=True,
            enable_shadows=False,
            enable_sky=False,
            randomly_sample_visible_envs=False,
        ),
    ]
    return cfg


def _assert_unpartitioned(prim, attribute_name: str) -> None:
    attribute = prim.GetAttribute(attribute_name)
    assert not attribute.IsValid() or not attribute.HasAuthoredValueOpinion(), (
        f"Unexpected authored {attribute_name!r} on {prim.GetPath()}."
    )


def _shape_matches(model, label_fragment: str) -> list[tuple[int, int, bool]]:
    visible_flag = int(newton.ShapeFlags.VISIBLE)
    flags = model.shape_flags.numpy()
    worlds = model.shape_world.numpy()
    matches = [
        (shape_id, int(worlds[shape_id]), bool(int(flags[shape_id]) & visible_flag))
        for shape_id, label in enumerate(model.shape_label)
        if label_fragment in str(label)
    ]
    assert matches, f"No Newton shapes matched {label_fragment!r}."
    return matches


def _assert_shape_distribution(
    model, label_fragment: str, *, visible: bool, expected_per_world: dict[int, int]
) -> None:
    matches = _shape_matches(model, label_fragment)
    actual_per_world = {
        world: sum(match_world == world for _, match_world, _ in matches) for world in expected_per_world
    }
    assert actual_per_world == expected_per_world, (label_fragment, matches)
    assert all(match_visible is visible for _, _, match_visible in matches), (label_fragment, matches)


def _assert_fabric_poses_match(view: FabricFrameView, expected_paths: list[str], expected_pose_w: torch.Tensor) -> None:
    assert view.prim_paths == expected_paths
    positions, orientations = view.get_world_poses()
    torch.testing.assert_close(positions.torch, expected_pose_w[:, :3], rtol=0.0, atol=1.0e-5)
    quaternion_alignment = torch.abs(torch.sum(orientations.torch * expected_pose_w[:, 3:7], dim=-1))
    torch.testing.assert_close(
        quaternion_alignment,
        torch.ones_like(quaternion_alignment),
        rtol=0.0,
        atol=1.0e-5,
    )


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason=_RUNTIME_UNAVAILABLE_REASON)
def test_franka_pour_kit_and_newton_visualize_both_worlds(monkeypatch: pytest.MonkeyPatch):
    """Both renderers should consume the two isolated worlds without adding another world offset."""
    monkeypatch.delenv(_SCENE_PARTITION_ENV_VAR, raising=False)
    sim_utils.create_new_stage()
    env = None
    try:
        env = gym.make(_TASK_ID, cfg=_make_visualization_cfg())
        task = env.unwrapped
        task.sim._app_control_on_stop_handle = None
        env.reset()

        stage = sim_utils.get_current_stage()
        model = NewtonManager.get_model()
        wp.synchronize_device(model.device)
        assert str(model.device) == _TEST_DEVICE

        expected_origins = np.array([[1.25, 0.0, 0.0], [-1.25, 0.0, 0.0]], dtype=np.float32)
        origins = task.scene.env_origins.detach().cpu().numpy()
        np.testing.assert_allclose(origins, expected_origins, rtol=0.0, atol=1.0e-6)

        local_positions = {
            "Robot": tuple(task.cfg.scene.robot.init_state.pos),
            "SourceCup": tuple(task.cfg.cup_reset_pos),
            "TargetCup": tuple(task.cfg.target_cup_reset_pos),
        }
        for env_id, origin in enumerate(origins):
            env_root = stage.GetPrimAtPath(f"/World/envs/env_{env_id}")
            assert env_root.IsValid()
            assert UsdGeom.Imageable(env_root).ComputeVisibility() == UsdGeom.Tokens.inherited
            _assert_unpartitioned(env_root, "primvars:omni:scenePartition")

            for asset_name, expected_local_position in local_positions.items():
                asset_path = f"/World/envs/env_{env_id}/{asset_name}"
                asset_prim = stage.GetPrimAtPath(asset_path)
                assert asset_prim.IsValid(), asset_path
                local_position, _ = sim_utils.resolve_prim_pose(asset_prim, ref_prim=env_root)
                world_position, _ = sim_utils.resolve_prim_pose(asset_prim)
                np.testing.assert_allclose(local_position, expected_local_position, rtol=0.0, atol=1.0e-6)
                np.testing.assert_allclose(
                    world_position,
                    origin + np.asarray(expected_local_position),
                    rtol=0.0,
                    atol=1.0e-6,
                )

            for cup_name in ("SourceCup", "TargetCup"):
                mesh_path = f"/World/envs/env_{env_id}/{cup_name}/geometry/mesh"
                mesh = UsdGeom.Mesh.Get(stage, mesh_path)
                assert mesh.GetPrim().IsValid(), mesh_path
                assert UsdGeom.Imageable(mesh).ComputeVisibility() == UsdGeom.Tokens.inherited
            assert not stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Cup").IsValid()

        source_positions = task.scene["source_cup"].data.root_link_pose_w.torch[:, :3].detach().cpu().numpy()
        target_positions = task.scene["target_cup"].data.root_link_pose_w.torch[:, :3].detach().cpu().numpy()
        np.testing.assert_allclose(
            source_positions,
            origins + np.asarray(task.cfg.cup_reset_pos),
            rtol=0.0,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            target_positions,
            origins + np.asarray(task.cfg.target_cup_reset_pos),
            rtol=0.0,
            atol=1.0e-6,
        )

        body_labels = [str(label) for label in model.body_label]
        for env_id in (0, 1):
            for body_name in ("SourceCup", "TargetCup", "TargetCupRigid", "SpillFloor"):
                expected_label = f"/World/envs/env_{env_id}/{body_name}"
                assert body_labels.count(expected_label) == 1, expected_label
            assert f"/World/envs/env_{env_id}/Cup" not in body_labels

        assert {world for _, world, _ in _shape_matches(model, "/Robot/")} == {0, 1}
        _assert_shape_distribution(model, "/SourceCup/geometry/mesh", visible=True, expected_per_world={0: 1, 1: 1})
        _assert_shape_distribution(model, "/TargetCup/geometry/mesh", visible=True, expected_per_world={0: 1, 1: 1})
        _assert_shape_distribution(
            model, "/SourceCup/geometry/grasp_proxy", visible=False, expected_per_world={0: 1, 1: 1}
        )
        _assert_shape_distribution(model, "/ParticleCollider", visible=False, expected_per_world={0: 2, 1: 2})
        _assert_shape_distribution(model, "/TargetCupRigid/Collision", visible=False, expected_per_world={0: 1, 1: 1})
        _assert_shape_distribution(model, "/SpillFloor/Collision", visible=False, expected_per_world={0: 1, 1: 1})
        assert set(model.particle_world.numpy().tolist()) == {0, 1}

        point_paths = {
            prim.GetPath().pathString
            for prim in stage.Traverse()
            if prim.IsA(UsdGeom.Points) and prim.GetPath().pathString.startswith("/World/Visuals/MPMParticles/")
        }
        assert len(point_paths) == 2
        assert {path.rsplit("/", 1)[-1] for path in point_paths} == {"env_0", "env_1"}
        point_path_by_env = {path.rsplit("/", 1)[-1]: path for path in point_paths}

        cameras = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Camera)]
        assert cameras
        for camera in cameras:
            _assert_unpartitioned(camera, "omni:scenePartition")

        kit_visualizers = [visualizer for visualizer in task.sim.visualizers if isinstance(visualizer, KitVisualizer)]
        newton_visualizers = [
            visualizer for visualizer in task.sim.visualizers if isinstance(visualizer, NewtonVisualizer)
        ]
        assert len(kit_visualizers) == 1
        assert len(newton_visualizers) == 1
        assert kit_visualizers[0].cfg.max_visible_envs is None
        assert kit_visualizers[0].get_visualized_env_ids() is None

        newton_visualizer = newton_visualizers[0]
        assert newton_visualizer.cfg.max_visible_envs is None
        assert newton_visualizer.get_visualized_env_ids() is None
        # NewtonVisualizer has no public accessor for the native viewer.
        viewer = newton_visualizer._viewer
        assert viewer is not None
        assert model.world_count == 2
        assert viewer._visible_worlds is None
        assert viewer._visible_worlds_mask is None
        np.testing.assert_array_equal(viewer.world_offsets.numpy(), np.zeros((2, 3), dtype=np.float32))
        assert viewer.show_particles is True

        robot_paths = [f"/World/envs/env_{env_id}/Robot/panda_link0" for env_id in range(2)]
        source_paths = [f"/World/envs/env_{env_id}/SourceCup" for env_id in range(2)]
        robot_fabric_view = FabricFrameView("/World/envs/env_.*/Robot/panda_link0", device=_TEST_DEVICE)
        source_fabric_view = FabricFrameView("/World/envs/env_.*/SourceCup", device=_TEST_DEVICE)
        # Prime each view's one-time USD-to-Fabric binding before physics changes the matrices.
        robot_fabric_view.get_world_poses()
        source_fabric_view.get_world_poses()

        actions = torch.zeros((task.num_envs, task.action_manager.total_action_dim), device=task.device)
        env.step(actions)
        wp.synchronize_device(model.device)

        link0_ids, _ = task.scene["robot"].find_bodies("panda_link0", preserve_order=True)
        _assert_fabric_poses_match(
            robot_fabric_view,
            robot_paths,
            task.scene["robot"].data.body_link_pose_w.torch[:, link0_ids[0]],
        )
        _assert_fabric_poses_match(
            source_fabric_view,
            source_paths,
            task.scene["source_cup"].data.root_link_pose_w.torch,
        )

        particle_offsets = task.scene[MPM_ENTRY].particle_offsets.numpy().astype(np.int64, copy=False)
        particles_per_world = int(task.scene[MPM_ENTRY].particles_per_object)
        particle_q = NewtonManager.get_state_0().particle_q.numpy()
        for env_id, offset in enumerate(particle_offsets):
            points = UsdGeom.Points.Get(stage, point_path_by_env[f"env_{env_id}"]).GetPointsAttr().Get()
            actual = np.asarray(points, dtype=np.float32)
            expected = particle_q[int(offset) : int(offset) + particles_per_world]
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-6)
    finally:
        if env is not None:
            env.close()
