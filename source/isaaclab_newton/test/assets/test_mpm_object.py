# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import warp as wp

newton = pytest.importorskip("newton")

from isaaclab_newton.assets.mpm_object import MPMObject, MPMObjectCfg
from isaaclab_newton.assets.mpm_object.mpm_object import MPMObjectRegistryEntry, add_mpm_entry_to_builder
from isaaclab_newton.physics import MPMSolverCfg, NewtonCfg, NewtonMPMManager
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg, MPMPointsCfg

from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.configclass import configclass


def test_mpm_particle_material_emits_custom_attributes():
    """MPM materials are value cfgs forwarded as Newton custom attributes, not USD material spawners."""
    from isaaclab_newton.sim.spawners.mpm.mpm import _material_custom_attributes

    attrs = _material_custom_attributes(MPMParticleMaterialCfg(viscosity=0.1))

    assert attrs["mpm:friction"] == pytest.approx(0.68)
    assert attrs["mpm:viscosity"] == pytest.approx(0.1)
    assert "density" not in attrs


def test_mpm_object_cfg_resolves_asset_class():
    cfg = MPMObjectCfg(
        prim_path="/World/envs/env_.*/Sand",
        spawn=MPMGridCfg(lower=(0.0, 0.0, 0.0), upper=(0.1, 0.1, 0.1), voxel_size=0.1),
    )

    assert cfg.class_type.__name__ == MPMObject.__name__


def test_mpm_object_empty_active_mask_write_is_noop():
    """An empty instance selection must not resolve or rebuild Newton material arrays."""
    empty_ids = wp.empty(0, dtype=wp.int32, device="cpu")

    class _FakeMedia:
        _particles_per_object = 4
        _particle_activation_target = None

        @staticmethod
        def _resolve_env_ids(env_ids):
            del env_ids
            return empty_ids

        @staticmethod
        def _as_warp(*args, **kwargs):
            del args, kwargs
            pytest.fail("empty activity writes must return before data conversion")

    media = _FakeMedia()
    MPMObject.write_particle_active_mask_to_sim_index(
        media,
        torch.empty((0, 4), dtype=torch.bool),
        env_ids=[],
    )

    assert media._particle_activation_target is None


def test_mpm_grid_emission_records_constant_offsets_per_env():
    builder = newton.ModelBuilder()
    NewtonMPMManager._register_builder_attributes(builder)

    cfg = MPMObjectCfg(
        prim_path="/World/envs/env_.*/Sand",
        spawn=MPMGridCfg(
            lower=(0.0, 0.0, 0.0),
            upper=(0.1, 0.1, 0.1),
            voxel_size=0.1,
            particles_per_cell=1.0,
            jitter=0.0,
            particle_placement="cell_center",
        ),
    )
    entry = MPMObjectRegistryEntry(cfg)

    add_mpm_entry_to_builder(builder, entry, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    add_mpm_entry_to_builder(builder, entry, 1, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])

    assert entry.particles_per_object == 1
    assert entry.particle_offsets == [0, 1]
    assert builder.particle_count == 2


def test_mpm_points_emission_records_constant_offsets_per_env():
    builder = newton.ModelBuilder()
    NewtonMPMManager._register_builder_attributes(builder)

    cfg = MPMObjectCfg(
        prim_path="/World/envs/env_.*/Fluid",
        spawn=MPMPointsCfg(
            positions=((0.0, 0.0, 0.0), (0.0, 0.0, 0.1), (0.0, 0.1, 0.0)),
            velocities=((0.0, 0.0, 0.0), (0.0, 0.0, 0.1), (0.0, 0.1, 0.0)),
            mass=0.01,
            radius=0.02,
            material=MPMParticleMaterialCfg(viscosity=0.1, friction=0.0),
        ),
    )
    entry = MPMObjectRegistryEntry(cfg)

    add_mpm_entry_to_builder(builder, entry, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    add_mpm_entry_to_builder(builder, entry, 1, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0])

    assert entry.particles_per_object == 3
    assert entry.particle_offsets == [0, 3]
    assert builder.particle_count == 6


def test_mpm_object_initializes_from_interactive_scene():
    @configclass
    class MPMSceneCfg(InteractiveSceneCfg):
        media = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sand",
            spawn=MPMGridCfg(
                lower=(0.0, 0.0, 0.0),
                upper=(0.1, 0.1, 0.1),
                voxel_size=0.1,
                particle_placement="cell_center",
            ),
        )

    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(solver_cfg=MPMSolverCfg(max_iterations=2, voxel_size=0.05), use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(MPMSceneCfg(num_envs=2, env_spacing=1.0))
        sim.reset()

        media = scene["media"]
        assert media.num_instances == 2
        assert media.particles_per_object == 1
        assert media.data.particle_pos_w.torch.shape == (2, 1, 3)
        assert not NewtonMPMManager._particle_visual_prims

        default_state = media.data.default_particle_state_w.torch.clone()
        shifted_state = default_state[0:1].clone()
        shifted_state[..., 2] += 0.05

        media.write_particle_state_to_sim_index(
            shifted_state,
            env_ids=torch.tensor([0], device=sim.device, dtype=torch.int32),
        )
        torch.testing.assert_close(media.data.particle_state_w.torch[0:1], shifted_state)

        media.reset(env_ids=[0])
        torch.testing.assert_close(media.data.particle_state_w.torch[0], default_state[0])


def test_mpm_object_runtime_active_mask_refreshes_eager_sparse_solver():
    """Deactivate and reactivate fixed particle slots through Newton's public eager refresh."""

    @configclass
    class MPMSceneCfg(InteractiveSceneCfg):
        media = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sand",
            spawn=MPMGridCfg(
                lower=(0.0, 0.0, 0.2),
                upper=(0.2, 0.1, 0.3),
                voxel_size=0.1,
                particles_per_cell=1.0,
                particle_placement="cell_center",
            ),
        )

    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MPMSolverCfg(
                max_iterations=2,
                voxel_size=0.1,
                grid_type="sparse",
                separate_worlds=True,
                max_active_cell_count=128,
                max_leaf_node_count=128,
                max_lower_node_count=64,
                max_upper_node_count=32,
            ),
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(MPMSceneCfg(num_envs=2, env_spacing=1.0))
        sim.reset()
        media = scene["media"]
        assert media.particles_per_object == 4

        first_mask = torch.tensor(
            [[True, False, True, False], [False, True, False, True]],
            dtype=torch.bool,
            device=sim.device,
        )
        media.write_particle_active_mask_to_sim_index(first_mask)
        sim.step(render=False)

        model = NewtonMPMManager.get_model()
        target = media._particle_activation_target
        assert target is not None
        public_flag_pointers = tuple(array.ptr for array in target.model_particle_flags)
        assert any(target.solver.model.particle_flags is array for array in target.model_particle_flags)
        active_flag = int(newton.ParticleFlags.ACTIVE)
        model_active = (wp.to_torch(model.particle_flags).reshape(2, 4) & active_flag) != 0
        torch.testing.assert_close(model_active, first_mask)
        velocity = media.data.particle_vel_w.torch.clone()
        assert torch.all(velocity[first_mask][:, 2].abs() > 0.0)
        torch.testing.assert_close(velocity[~first_mask], torch.zeros_like(velocity[~first_mask]))

        media.reset()
        second_mask = ~first_mask
        media.write_particle_active_mask_to_sim_index(second_mask)
        world_mask = wp.array([True, True, False], dtype=wp.bool, device=sim.device)
        NewtonMPMManager.reset_solver_state(world_mask=world_mask, flags=newton.StateFlags.PARTICLE)
        sim.step(render=False)

        assert tuple(array.ptr for array in target.model_particle_flags) == public_flag_pointers
        assert target.solver.model.particle_flags.ptr in public_flag_pointers
        model_active = (wp.to_torch(model.particle_flags).reshape(2, 4) & active_flag) != 0
        torch.testing.assert_close(model_active, second_mask)
        velocity = media.data.particle_vel_w.torch
        assert torch.all(velocity[second_mask][:, 2].abs() > 0.0)
        torch.testing.assert_close(velocity[~second_mask], torch.zeros_like(velocity[~second_mask]))
        assert model._isaaclab_particle_flags_dynamic is True


def test_mpm_solver_refreshes_kinematic_rigid_body_transforms():
    import isaaclab.sim as sim_utils  # noqa: PLC0415

    @configclass
    class MPMSceneCfg(InteractiveSceneCfg):
        collider = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/KinematicBox",
            spawn=sim_utils.CuboidCfg(
                size=(0.1, 0.1, 0.1),
                rigid_props=sim_utils.NewtonRigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.NewtonCollisionPropertiesCfg(collision_enabled=True),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
        )
        media = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sand",
            spawn=MPMGridCfg(lower=(-0.05, -0.05, 0.3), upper=(0.05, 0.05, 0.4), voxel_size=0.05),
        )

    sim_cfg = SimulationCfg(
        dt=1.0 / 60.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(solver_cfg=MPMSolverCfg(max_iterations=2, voxel_size=0.05), use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(MPMSceneCfg(num_envs=1, env_spacing=0.0))
        sim.reset()

        collider = scene["collider"]
        angle = 0.5
        root_pose = torch.tensor(
            [[0.1, 0.0, 0.25, 0.0, math.sin(0.5 * angle), 0.0, math.cos(0.5 * angle)]],
            dtype=torch.float32,
            device=collider.device,
        )
        collider.write_root_link_pose_to_sim_index(root_pose=root_pose)
        sim.step(render=False)

        body_labels = list(NewtonMPMManager.get_model().body_label)
        body_idx = body_labels.index("/World/envs/env_0/KinematicBox")
        body_q = NewtonMPMManager.get_state_0().body_q.numpy()[body_idx]

        np.testing.assert_allclose(body_q, root_pose.detach().cpu().numpy()[0], rtol=1.0e-5, atol=1.0e-6)


def test_mpm_object_creates_usd_points_for_kit_visualizer(monkeypatch):
    @configclass
    class MPMSceneCfg(InteractiveSceneCfg):
        media = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sand",
            spawn=MPMGridCfg(
                lower=(0.0, 0.0, 0.0),
                upper=(0.1, 0.1, 0.1),
                voxel_size=0.1,
                visual_color=(0.1, 0.2, 0.3),
            ),
        )

    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(solver_cfg=MPMSolverCfg(max_iterations=2, voxel_size=0.05), use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        monkeypatch.setattr(sim, "resolve_visualizer_types", lambda: ["kit"])
        scene = InteractiveScene(MPMSceneCfg(num_envs=2, env_spacing=1.0))

        from pxr import UsdGeom  # noqa: PLC0415

        sim.reset()

        media = scene["media"]
        records = NewtonMPMManager._particle_visual_prims
        assert len(records) == media.num_instances

        for env_idx, (prim_path, record) in enumerate(sorted(records.items())):
            assert record.offset == media._recorded_particle_offsets[env_idx]
            assert record.count == media.particles_per_object
            assert record.sync_frequency == 1

            points_prim = media.stage.GetPrimAtPath(prim_path)
            assert points_prim.IsValid()
            points = UsdGeom.Points(points_prim)
            assert len(points.GetPointsAttr().Get()) == media.particles_per_object
            assert len(points.GetWidthsAttr().Get()) == media.particles_per_object
            assert tuple(points.GetDisplayColorAttr().Get()[0]) == pytest.approx((0.1, 0.2, 0.3))


def test_mpm_kit_points_follow_particle_state(monkeypatch):
    @configclass
    class MPMSceneCfg(InteractiveSceneCfg):
        media = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sand",
            spawn=MPMGridCfg(
                lower=(0.0, 0.0, 0.1),
                upper=(0.1, 0.1, 0.2),
                voxel_size=0.05,
                visual_color=(0.1, 0.2, 0.3),
            ),
        )

    sim_cfg = SimulationCfg(
        dt=1.0 / 60.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(solver_cfg=MPMSolverCfg(max_iterations=2, voxel_size=0.05), use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        monkeypatch.setattr(sim, "resolve_visualizer_types", lambda: ["kit"])
        scene = InteractiveScene(MPMSceneCfg(num_envs=1, env_spacing=0.0))
        sim.reset()

        from pxr import UsdGeom  # noqa: PLC0415

        media = scene["media"]
        prim_path = next(iter(NewtonMPMManager._particle_visual_prims))
        points = UsdGeom.Points(media.stage.GetPrimAtPath(prim_path))
        points_before = np.asarray(points.GetPointsAttr().Get(), dtype=np.float32)

        for _ in range(3):
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
            sim.render()

        points_after = np.asarray(points.GetPointsAttr().Get(), dtype=np.float32)
        particle_pos = media.data.particle_pos_w.torch.detach().cpu().numpy()[0]

        assert np.max(np.abs(points_after - points_before)) > 0.0
        np.testing.assert_allclose(points_after, particle_pos, rtol=1.0e-5, atol=1.0e-6)
