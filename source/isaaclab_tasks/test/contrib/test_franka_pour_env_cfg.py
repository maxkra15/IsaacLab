# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Construction tests for the Franka pour env config (no simulator)."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.schemas import MujocoJointCfg

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sim.schemas import MassCfg, UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg

from isaaclab_contrib.coupling import CoupledProxySolverCfg, NewtonCoupledSolverManager

import isaaclab_tasks.contrib.franka_pour as franka_pour
import isaaclab_tasks.contrib.franka_pour.config.franka  # noqa: F401
import isaaclab_tasks.contrib.franka_pour.mdp as mdp
import isaaclab_tasks.contrib.franka_pour.pour_env_cfg as pour_env_cfg
from isaaclab_tasks.contrib.franka_pour.config.franka.agents.rsl_rl_ppo_cfg import FrankaPourPPORunnerCfg
from isaaclab_tasks.contrib.franka_pour.cube_bowl_spawner_cfg import CubeBowlSpawnerCfg
from isaaclab_tasks.contrib.franka_pour.cup_media import cup_cavity_lattice
from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import (
    FrankaPourEnvCfg,
    FrankaPourEnvCfg_PLAY,
    FrankaPourEnvCfg_TELEOP,
    _mpm_solver_cfg,
    _resolve_mpm_cell_cap,
    _resolve_mpm_upper_node_cap,
)
from isaaclab_tasks.contrib.franka_pour.reset_utils import (
    balanced_cyclic_permutations,
    boolean_selection_mask,
    randomization_extent_index_pools,
    sample_index_pools,
    target_xy_behind_source,
)


def test_boolean_selection_mask_preserves_device_shape_and_dtype():
    selected = torch.tensor([[1, 3], [3, 4]], dtype=torch.long)

    mask = boolean_selection_mask(6, selected)

    assert mask.device == selected.device
    assert mask.dtype == torch.bool
    assert mask.shape == (6,)
    assert mask.tolist() == [False, True, False, True, True, False]


def test_balanced_cyclic_permutations_preserve_marginals_and_balance_pairings():
    values = torch.tensor([0.0, 0.5, 1.0, -0.5, -1.0])

    permutations = balanced_cyclic_permutations(values, group_count=49)

    assert permutations.shape == (49, 5)
    expected = torch.sort(values).values.expand(49, -1)
    torch.testing.assert_close(torch.sort(permutations, dim=-1).values, expected)
    for column in range(permutations.shape[1]):
        _, counts = torch.unique(permutations[:, column], return_counts=True)
        assert int(counts.amax() - counts.amin()) <= 1
    assert torch.unique(permutations, dim=0).shape[0] == values.numel()


def test_randomization_extent_pools_combine_all_axes_and_are_nested():
    source_positions = torch.tensor([[0.50, 0.00], [0.525, 0.00], [0.55, 0.00], [0.575, 0.00], [0.60, 0.00]])
    target_positions = torch.tensor([[0.50, -0.20], [0.51, -0.20], [0.54, -0.20], [0.52, -0.20], [0.50, -0.15]])
    source_yaws = torch.tensor([0.00, 0.10, 0.05, 0.15, 0.02])
    tcp_jitter = torch.tensor(
        [[0.00, 0.00, 0.00], [0.012, 0.00, 0.00], [0.006, 0.00, 0.00], [0.018, 0.00, 0.00], [0.004, 0.00, 0.00]]
    )

    pools = randomization_extent_index_pools(
        source_positions,
        source_yaws,
        target_positions,
        tcp_jitter,
        source_center=(0.50, 0.00),
        source_half_range=(0.10, 0.10),
        source_yaw_half_range=0.20,
        target_center=(0.50, -0.20),
        target_half_range=(0.05, 0.05),
        tcp_jitter_half_range=(0.02, 0.02, 0.02),
        extent_levels=(0.5, 0.8, 1.0),
    )

    assert [pool.tolist() for pool in pools] == [[0], [0, 1, 2], [0, 1, 2, 3, 4]]
    assert bool(torch.all(torch.isin(pools[0], pools[1])))
    assert bool(torch.all(torch.isin(pools[1], pools[2])))
    torch.testing.assert_close(pools[2], torch.arange(5))


def test_randomization_extent_pools_handle_zero_range_axes_without_nan():
    source_positions = torch.tensor([[0.5, -0.1], [0.5, 0.0], [0.5, 0.1], [0.5, 0.0]])
    target_positions = torch.tensor([[0.4, -0.2], [0.4, -0.1], [0.4, 0.0], [0.4, -0.1]])
    source_yaws = torch.zeros(4)
    tcp_jitter = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.01, 0.0]])

    pools = randomization_extent_index_pools(
        source_positions,
        source_yaws,
        target_positions,
        tcp_jitter,
        source_center=(0.5, 0.0),
        source_half_range=(0.0, 0.1),
        source_yaw_half_range=0.0,
        target_center=(0.4, -0.1),
        target_half_range=(0.0, 0.1),
        tcp_jitter_half_range=(0.0, 0.0, 0.0),
        extent_levels=(0.5, 1.0),
    )

    assert [pool.tolist() for pool in pools] == [[1], [0, 1, 2]]


def test_randomization_extent_pools_include_source_yaw():
    pools = randomization_extent_index_pools(
        torch.zeros((3, 2)),
        torch.tensor([0.0, 0.05, -0.10]),
        torch.zeros((3, 2)),
        torch.zeros((3, 3)),
        source_center=(0.0, 0.0),
        source_half_range=(0.0, 0.0),
        source_yaw_half_range=0.10,
        target_center=(0.0, 0.0),
        target_half_range=(0.0, 0.0),
        tcp_jitter_half_range=(0.0, 0.0, 0.0),
        extent_levels=(0.5, 1.0),
    )

    assert [pool.tolist() for pool in pools] == [[0, 1], [0, 1, 2]]


def test_randomization_extent_pools_require_aligned_bank_rows():
    with pytest.raises(ValueError, match="same row count"):
        randomization_extent_index_pools(
            torch.zeros((2, 3)),
            torch.zeros(2),
            torch.zeros((1, 3)),
            torch.zeros((2, 3)),
            source_center=(0.0, 0.0),
            source_half_range=(0.1, 0.1),
            source_yaw_half_range=0.1,
            target_center=(0.0, 0.0),
            target_half_range=(0.1, 0.1),
            tcp_jitter_half_range=(0.1, 0.1, 0.1),
            extent_levels=(1.0,),
        )


def test_index_pool_sampling_maps_local_slots_back_to_global_bank_rows(monkeypatch):
    pools = (torch.tensor([4, 8]), torch.tensor([1, 5, 9]), torch.tensor([0, 3, 6, 9]))
    pool_ids = torch.tensor([0, 2, 1, 0, 2])

    def sample_last_slot(high, size, *, device):
        return torch.full(size, high - 1, device=device, dtype=torch.long)

    monkeypatch.setattr(torch, "randint", sample_last_slot)
    sampled = sample_index_pools(pools, pool_ids)

    assert sampled.tolist() == [8, 9, 9, 8, 9]


def test_target_randomization_is_bounded_and_separated_behind_source():
    source_xy = torch.tensor([[0.40, -0.10], [0.60, 0.10]])
    unit_samples = torch.tensor([[0.0, 1.0], [1.0, 0.5]])

    minimum_y_separation = torch.tensor([0.129, 0.150])
    target_xy = target_xy_behind_source(
        source_xy,
        target_center=(0.50, -0.18),
        target_half_range=(0.05, 0.05),
        minimum_y_separation=minimum_y_separation,
        unit_samples=unit_samples,
    )

    assert bool(torch.all(target_xy[:, 0] >= 0.45))
    assert bool(torch.all(target_xy[:, 0] <= 0.55))
    assert bool(torch.all(target_xy[:, 1] >= -0.23))
    assert bool(torch.all(target_xy[:, 1] <= -0.13))
    assert bool(torch.all(source_xy[:, 1] - target_xy[:, 1] >= minimum_y_separation - 1.0e-6))


def test_target_randomization_rejects_a_source_position_without_feasible_separation():
    source_xy = torch.tensor([[0.50, -0.11]])

    with pytest.raises(ValueError, match="No target y-position"):
        target_xy_behind_source(
            source_xy,
            target_center=(0.50, -0.18),
            target_half_range=(0.05, 0.05),
            minimum_y_separation=0.129,
            unit_samples=torch.tensor([[0.5, 0.5]]),
        )


def test_finalize_builds_scene_assets_without_mutating_the_caller():
    original = FrankaPourEnvCfg()
    original.scene.num_envs = 8

    resolved = original.finalize()

    assert original.scene.source_cup is None
    assert original.scene.target_cup is None
    assert original.scene.media is None
    assert isinstance(resolved.scene.source_cup, RigidObjectCfg)
    assert isinstance(resolved.scene.target_cup, RigidObjectCfg)
    assert isinstance(resolved.scene.media, MPMObjectCfg)
    assert resolved is not original
    assert resolved.scene is not original.scene
    assert resolved.sim.physics.scene_cfg is resolved.scene
    assert _media_capacity(resolved) == 8 * _aligned_particle_capacity(resolved)


def test_finalize_colocates_only_large_isolated_physics_batches():
    cfg = FrankaPourEnvCfg()
    original_spacing = cfg.scene.env_spacing
    cfg.scene.num_envs = 3068

    resolved = cfg.finalize()

    assert cfg.scene.env_spacing == pytest.approx(original_spacing)
    assert resolved.scene.env_spacing == pytest.approx(0.0)
    cfg.scene.num_envs = cfg.colocate_physics_min_envs - 1
    assert cfg.finalize().scene.env_spacing == pytest.approx(original_spacing)
    cfg.scene.num_envs = cfg.colocate_physics_min_envs
    assert cfg.finalize().scene.env_spacing == pytest.approx(0.0)
    assert FrankaPourEnvCfg_PLAY().finalize().scene.env_spacing == pytest.approx(original_spacing)


def test_finalize_preserves_spacing_when_colocation_is_disabled_or_worlds_are_shared():
    cfg = FrankaPourEnvCfg()
    cfg.scene.num_envs = 3068
    original_spacing = cfg.scene.env_spacing
    cfg.colocate_physics_min_envs = None
    assert cfg.finalize().scene.env_spacing == pytest.approx(original_spacing)

    cfg.colocate_physics_min_envs = 1024
    _mpm_solver_cfg(cfg).separate_worlds = False
    assert cfg.finalize().scene.env_spacing == pytest.approx(original_spacing)


@pytest.mark.parametrize("invalid_value", [0, -1])
def test_finalize_rejects_invalid_physics_colocation_threshold(invalid_value):
    cfg = FrankaPourEnvCfg()
    cfg.colocate_physics_min_envs = invalid_value

    with pytest.raises(ValueError, match="colocate_physics_min_envs"):
        cfg.finalize()


@pytest.mark.parametrize("invalid_value", [True, 1024.0])
def test_finalize_rejects_noninteger_physics_colocation_threshold(invalid_value):
    cfg = FrankaPourEnvCfg()
    cfg.colocate_physics_min_envs = invalid_value

    with pytest.raises(TypeError, match="colocate_physics_min_envs"):
        cfg.finalize()


def test_finalize_propagates_final_cup_overrides_to_fresh_assets():
    original = FrankaPourEnvCfg()
    original.source_cup_inner_width = 0.041
    original.target_cup_inner_depth = 0.153
    original.cup_grasp_box_half = (0.021, 0.022, 0.043)
    original.gripper_preload_pos = 0.017
    original.cup_grasp_box_friction = 2.4
    original.target_cup_friction = 1.3
    original.cup_mass = 0.071
    original.cup_reset_pos = (0.43, 0.02, 0.01)
    original.target_cup_reset_pos = (0.47, -0.21, 0.02)
    original.arm_home = (0.1, 0.2, 0.3, -2.4, 0.5, 3.2, 0.7)
    original.arm_stiffness = 550.0
    original.success_dwell_time_s = 0.10

    first = original.finalize()
    first_source = first.scene.source_cup
    first_target = first.scene.target_cup
    assert isinstance(first_source.spawn, CubeBowlSpawnerCfg)
    assert isinstance(first_target.spawn, CubeBowlSpawnerCfg)
    assert first_source.spawn.inner_width == pytest.approx(0.041)
    assert first_target.spawn.inner_depth == pytest.approx(0.153)
    assert first_source.spawn.grasp_proxy_half_extents == pytest.approx((0.021, 0.022, 0.043))
    assert first_source.spawn.physics_material.static_friction == pytest.approx(2.4)
    assert first_source.spawn.physics_material.dynamic_friction == pytest.approx(2.4)
    assert first_target.spawn.physics_material.static_friction == pytest.approx(1.3)
    assert first_target.spawn.physics_material.dynamic_friction == pytest.approx(1.3)
    assert first_source.spawn.mass_props.mass == pytest.approx(0.071)
    assert first_source.init_state.pos == pytest.approx((0.43, 0.02, 0.01))
    assert first_source.init_state.rot == (0.0, 0.0, 0.0, 1.0)
    assert first_target.init_state.pos == pytest.approx((0.47, -0.21, 0.02))
    assert first_target.init_state.rot == (0.0, 0.0, 0.0, 1.0)
    assert [first.scene.robot.init_state.joint_pos[f"panda_joint{i}"] for i in range(1, 8)] == pytest.approx(
        original.arm_home
    )
    assert first.scene.robot.actuators["panda_shoulder"].stiffness == pytest.approx(550.0)
    assert first.terminations.success.params["dwell_time_s"] == pytest.approx(0.10)
    assert first.actions.gripper_action.close_position == pytest.approx(0.016)
    assert first.rewards.task_progress.params["grasp_preload_position"] == pytest.approx(0.017)
    assert first.rewards.task_progress.params["max_gripper_command"] == pytest.approx(0.017)
    assert first.rewards.delivered.params["max_gripper_command"] == pytest.approx(0.017)
    assert first.terminations.success.params["max_gripper_command"] == pytest.approx(0.017)

    first.source_cup_inner_width = 0.049
    second = first.finalize()

    assert second.scene.source_cup is not first.scene.source_cup
    assert second.scene.target_cup is not first.scene.target_cup
    assert second.scene.media is not first.scene.media
    assert first.scene.source_cup.spawn.inner_width == pytest.approx(0.041)
    assert second.scene.source_cup.spawn.inner_width == pytest.approx(0.049)


def test_resolved_cups_have_backend_neutral_rigid_properties():
    resolved = FrankaPourEnvCfg().finalize()
    source = resolved.scene.source_cup
    target = resolved.scene.target_cup

    assert source.prim_path == "{ENV_REGEX_NS}/SourceCup"
    assert isinstance(source.spawn.rigid_props, UsdPhysicsRigidBodyCfg)
    assert source.spawn.rigid_props.rigid_body_enabled is True
    assert source.spawn.rigid_props.kinematic_enabled is False
    assert isinstance(source.spawn.collision_props, UsdPhysicsCollisionCfg)
    assert source.spawn.collision_props.collision_enabled is True
    assert isinstance(source.spawn.mass_props, MassCfg)
    assert isinstance(source.spawn.physics_material, RigidBodyMaterialBaseCfg)
    assert source.spawn.grasp_proxy_half_extents == resolved.cup_grasp_box_half

    assert target.prim_path == "{ENV_REGEX_NS}/TargetCup"
    assert isinstance(target.spawn.rigid_props, UsdPhysicsRigidBodyCfg)
    assert target.spawn.rigid_props.rigid_body_enabled is True
    assert target.spawn.rigid_props.kinematic_enabled is True
    assert target.spawn.grasp_proxy_half_extents is None
    assert isinstance(target.spawn.physics_material, RigidBodyMaterialBaseCfg)


def test_robot_authors_mujoco_gravity_compensation_with_joint_fragment():
    fragments = FrankaPourEnvCfg().scene.robot.spawn.joint_drive_props

    assert isinstance(fragments, list)
    assert any(isinstance(fragment, MujocoJointCfg) and fragment.actuatorgravcomp is True for fragment in fragments)


def test_public_fixed_size_tuple_annotations_are_specific():
    expected = {
        "arm_home": "tuple[float, float, float, float, float, float, float]",
        "curriculum_pour_arm_q": "tuple[float, float, float, float, float, float, float]",
        "curriculum_carry_arm_q": "tuple[float, float, float, float, float, float, float]",
        "tcp_offset_pos": "tuple[float, float, float]",
        "tcp_offset_rot": "tuple[float, float, float, float]",
        "cup_grasp_box_half": "tuple[float, float, float]",
        "cup_reset_pos": "tuple[float, float, float]",
        "target_cup_reset_pos": "tuple[float, float, float]",
        "particle_workspace_lower_bound": "tuple[float, float, float]",
        "particle_workspace_upper_bound": "tuple[float, float, float]",
    }

    assert {name: FrankaPourEnvCfg.__annotations__[name] for name in expected} == expected


def test_cfg_routes_each_body_to_exactly_one_solver():
    cfg = FrankaPourEnvCfg().finalize()
    solver = cfg.sim.physics.solver_cfg
    entries = {entry.name: entry for entry in solver.entries}
    assert set(entries) == {"arm", "media"}
    assert isinstance(solver, CoupledProxySolverCfg)

    arm = entries["arm"]
    media = entries["media"]
    assert arm.solver_cfg.integrator == "implicitfast"
    assert arm.solver_cfg.use_mujoco_contacts is False
    assert solver.use_collision_pipeline is True
    assert cfg.sim.physics.collision_cfg.soft_contact_max == 0
    assert arm.include_static_shapes is True
    assert arm.bodies == [SceneEntityCfg("robot"), SceneEntityCfg("source_cup"), SceneEntityCfg("target_cup")]
    assert arm.substeps == cfg.num_substeps
    assert media.all_particles is True
    assert media.bodies == [r".*/SpillFloor$"]
    assert media.include_static_shapes is False
    assert media.in_place is True
    assert media.solver_cfg.separate_worlds is True
    assert media.solver_cfg.grid_type == "sparse"
    assert media.solver_cfg.grid_padding == 0
    assert media.solver_cfg.max_active_cell_count > 0
    assert media.solver_cfg.solver == "jacobi"
    assert media.solver_cfg.max_iterations == 24
    assert cfg.sim.physics.use_cuda_graph is True

    proxies = solver.proxies
    assert len(proxies) == 1
    assert proxies[0].source == "arm" and proxies[0].destination == "media"
    assert proxies[0].bodies == [SceneEntityCfg("source_cup"), SceneEntityCfg("target_cup")]
    assert solver.iterations == cfg.proxy_iterations
    assert not hasattr(pour_env_cfg, "CUP_LABEL_PATTERN")
    assert all(
        "Cup$" not in selector for entry in solver.entries for selector in entry.bodies if isinstance(selector, str)
    )


def test_finalize_propagates_post_init_solver_overrides():
    cfg = FrankaPourEnvCfg()
    cfg.mpm_iterations = 48
    cfg.voxel_size = 0.02
    cfg.num_substeps = 3
    cfg.use_cuda_graph = False
    cfg.proxy_iterations = 4
    cfg.proxy_mass_scale = 0.25

    # Match Hydra's override timing: the nested tree still contains values authored during
    # ``__post_init__`` until finalization resolves the public top-level controls.
    assert _media_entry(cfg).solver_cfg.max_iterations == 24
    assert _media_entry(cfg).solver_cfg.voxel_size == pytest.approx(0.01)

    resolved = cfg.finalize()
    coupled_cfg = resolved.sim.physics.solver_cfg
    entries = {entry.name: entry for entry in coupled_cfg.entries}
    proxy = coupled_cfg.proxies[0]

    assert entries["media"].solver_cfg.max_iterations == 48
    assert entries["media"].solver_cfg.voxel_size == pytest.approx(0.02)
    assert entries["arm"].substeps == 3
    assert resolved.sim.physics.num_substeps == 3
    assert resolved.sim.physics.use_cuda_graph is False
    assert isinstance(coupled_cfg, CoupledProxySolverCfg)
    assert coupled_cfg.iterations == 4
    assert proxy.mass_scale == pytest.approx(0.25)


def test_visible_source_geometry_matches_grasp_proxy_and_fits_gripper():
    cfg = FrankaPourEnvCfg()
    outer_width = cfg.source_cup_inner_width + 2.0 * cfg.source_cup_wall_thickness
    outer_depth = cfg.source_cup_inner_depth + 2.0 * cfg.source_cup_wall_thickness
    outer_height = cfg.source_cup_cavity_depth + cfg.source_cup_bottom_thickness
    assert (outer_width, outer_depth, outer_height) == pytest.approx((0.056, 0.056, 0.036))
    assert cfg.cup_grasp_box_half == pytest.approx((outer_width / 2.0, outer_depth / 2.0, outer_height / 2.0))
    assert outer_depth < 2.0 * cfg.gripper_open_pos
    assert cfg.grasp_contact_ke >= 1.0e5
    assert cfg.grasp_contact_kd >= 5.0e2
    assert cfg.cup_grasp_box_friction >= 2.0
    # A source deeper than its opening can retain a settled granular bed even at the Franka's
    # stable wrist limit. Keep it shallow enough to pour before the physical grasp destabilizes.
    assert cfg.source_cup_cavity_depth <= cfg.source_cup_inner_width
    assert cfg.source_cup_bottom_thickness < cfg.cup_grasp_height < outer_height


def test_scene_cups_use_narrow_solver_only_builder_hook():
    source_path = Path(franka_pour.__file__).with_name("pour_env.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert {"_build_custom_proto", "_add_cup_body", "_add_target_cup_bodies"}.isdisjoint(function_names)

    hook = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_add_pour_world_to_builder"
    )
    hook_calls = {
        node.func.attr for node in ast.walk(hook) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert {"_find_world_body", "_add_particle_collider", "_add_rigid_collider"} <= hook_calls

    target_bridge = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_add_kinematic_rigid_object_articulation"
    )
    bridge_calls = {
        node.func.attr
        for node in ast.walk(target_bridge)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert {"add_joint_free", "add_articulation"} <= bridge_calls


def test_builder_hook_limits_lookups_to_current_world_tail():
    from isaaclab_tasks.contrib.franka_pour.pour_env import FrankaPourEnv

    builder = SimpleNamespace(
        body_world=[None, None, 0, 0, 1, 1, 1],
        shape_world=[None, 0, 0, 1, 1],
    )

    assert FrankaPourEnv._current_world_range(builder, "body", 1) == range(4, 7)
    assert FrankaPourEnv._current_world_range(builder, "shape", 1) == range(3, 5)
    with pytest.raises(RuntimeError, match="no body entries for open world 2"):
        FrankaPourEnv._current_world_range(builder, "body", 2)


def test_task_has_receiving_cup_safe_actions_and_success_threshold():
    cfg = FrankaPourEnvCfg()
    assert cfg.target_cup_reset_pos != cfg.cup_reset_pos
    assert isinstance(cfg.actions.arm_action, mdp.TrajectoryJointPositionActionCfg)
    assert cfg.actions.arm_action.joint_names == [f"panda_joint{i}" for i in range(1, 8)]
    assert cfg.actions.arm_action.preserve_order is True
    assert cfg.actions.arm_action.residual_scale == pytest.approx((0.03,) * 7)
    assert cfg.actions.arm_action.waypoint_phases == pytest.approx((0.0, 0.12, 0.24, 0.40, 0.62, 1.0))
    assert cfg.actions.arm_action.phase_rate == pytest.approx(0.40)
    assert cfg.actions.arm_action.approach_phase_rate == pytest.approx(0.40)
    assert cfg.actions.arm_action.transport_phase_rate == pytest.approx(0.40)
    assert cfg.actions.arm_action.grasp_dwell_steps == 5
    assert cfg.actions.arm_action.grasp_max_linear_velocity == pytest.approx(0.10)
    assert cfg.actions.arm_action.grasp_max_angular_velocity == pytest.approx(1.0)
    assert cfg.actions.arm_action.alpha == pytest.approx(0.10)
    for joint_name, home in zip(cfg.actions.arm_action.joint_names, cfg.arm_home, strict=True):
        lower, upper = cfg.actions.arm_action.clip[joint_name]
        assert lower <= home <= upper
    # Keep a meaningful particle margin below the demonstrated trajectory's lower tail while the
    # one-time delivery reward still drives transfer beyond the success threshold.
    assert cfg.curriculum_target_frac[-1] == pytest.approx(0.30)
    assert cfg.episode_length_s == pytest.approx(5.0)


def test_training_curriculum_closes_by_default_and_penalizes_irrecoverable_spill():
    cfg = FrankaPourEnvCfg()
    assert cfg.actions.arm_action.alpha == pytest.approx(0.10)
    assert cfg.actions.arm_action.grasp_gate_stage == 1
    assert type(cfg.actions.gripper_action).__name__ == "CurriculumGripperPositionActionCfg"
    assert cfg.actions.gripper_action.scale == pytest.approx(0.001)
    assert cfg.actions.gripper_action.alpha == pytest.approx(0.2)
    assert not hasattr(cfg.actions.gripper_action, "hold_offset_through_stage")
    assert cfg.actions.gripper_action.close_position == pytest.approx(0.024)
    assert cfg.actions.gripper_action.open_position == pytest.approx(cfg.gripper_open_pos)
    reward_names = {name for name, term_cfg in vars(cfg.rewards).items() if isinstance(term_cfg, RewardTermCfg)}
    assert reward_names == {
        "task_progress",
        "delivered",
        "success",
        "spill",
        "failure",
        "action_rate",
        "action_magnitude",
    }
    assert cfg.rewards.task_progress.func is mdp.PourTaskProgress
    assert cfg.rewards.task_progress.weight == pytest.approx(5.0)
    assert cfg.rewards.task_progress.params["grasp_reach_std"] == pytest.approx(0.015)
    assert cfg.rewards.task_progress.params["grasp_preload_position"] == pytest.approx(0.025)
    assert cfg.rewards.task_progress.params["source_offset_xy"] == pytest.approx(cfg.pour_source_offset_xy)
    assert cfg.rewards.task_progress.params["target_tilt"] == pytest.approx(math.radians(150.0))
    minimum_drain_tilt = math.atan2(cfg.source_cup_cavity_depth, 0.5 * cfg.source_cup_inner_depth)
    assert cfg.rewards.task_progress.params["target_tilt"] > minimum_drain_tilt
    assert cfg.rewards.task_progress.params["target_tilt"] < math.pi
    assert cfg.rewards.task_progress.params["pour_direction_xy"] == pytest.approx((0.0, -1.0))
    assert cfg.rewards.task_progress.params["source_mouth_height"] == pytest.approx(
        cfg.source_cup_bottom_thickness + cfg.source_cup_cavity_depth
    )
    assert cfg.rewards.task_progress.params["alignment_radius"] == pytest.approx(0.15)
    assert cfg.rewards.task_progress.params["active_through_stage"] == 1
    assert cfg.rewards.task_progress.params["min_lift_height"] == pytest.approx(0.05)
    assert cfg.rewards.task_progress.params["max_tcp_distance"] == pytest.approx(0.018)
    assert cfg.rewards.task_progress.params["max_gripper_width_error"] == pytest.approx(0.006)
    assert cfg.rewards.task_progress.params["max_gripper_command"] == pytest.approx(cfg.gripper_preload_pos)
    assert not hasattr(cfg.rewards, "reach")
    assert not hasattr(cfg.rewards, "grasp")
    assert not hasattr(cfg.rewards, "lift")
    assert not hasattr(cfg.rewards, "align")
    assert not hasattr(cfg.rewards, "tilt")
    assert cfg.rewards.delivered.func is mdp.HeldDeliveryProgress
    assert cfg.rewards.delivered.params["min_lift_height"] == pytest.approx(0.05)
    assert cfg.rewards.delivered.params["max_tcp_distance"] == pytest.approx(0.018)
    assert cfg.rewards.delivered.params["max_gripper_width_error"] == pytest.approx(0.006)
    assert cfg.rewards.delivered.params["max_gripper_command"] == pytest.approx(cfg.gripper_preload_pos)
    assert cfg.rewards.spill.func is mdp.NewlySpilledParticles
    assert cfg.rewards.task_progress.weight < cfg.rewards.success.weight
    assert cfg.rewards.spill.weight == pytest.approx(-30.0)
    assert cfg.rewards.failure.func is mdp.terminal_failure
    assert cfg.rewards.failure.weight == pytest.approx(-35.0)
    assert cfg.terminations.spill.func is mdp.excessive_spill
    assert cfg.terminations.extreme_rigid_state.func is mdp.extreme_rigid_state
    assert cfg.terminations.lost_grasp.func is mdp.lost_lifted_grasp
    assert cfg.terminations.lost_grasp.params["dwell_time_s"] == pytest.approx(0.05)
    assert cfg.terminations.success.func is mdp.stable_pour_success
    assert cfg.terminations.success.params["dwell_time_s"] == pytest.approx(0.15)
    assert cfg.terminations.success.params["min_lift_height"] == pytest.approx(0.05)
    assert cfg.terminations.success.params["max_tcp_distance"] == pytest.approx(0.018)
    assert cfg.terminations.success.params["max_gripper_width_error"] == pytest.approx(0.006)
    assert cfg.terminations.success.params["max_gripper_command"] == pytest.approx(cfg.gripper_preload_pos)
    termination_names = [
        name for name, term_cfg in vars(cfg.terminations).items() if isinstance(term_cfg, TerminationTermCfg)
    ]
    assert termination_names[-2:] == ["success", "time_out"]
    assert cfg.terminations.time_out.func is mdp.unsuccessful_time_out
    assert cfg.max_spill_fraction == pytest.approx(0.10)
    assert cfg.spill_table_height == pytest.approx(0.0)
    assert cfg.state_bound_joint_position_margin == pytest.approx(0.05)
    assert cfg.state_bound_max_joint_velocity == pytest.approx(20.0)
    assert cfg.state_bound_max_cup_linear_velocity == pytest.approx(10.0)
    assert cfg.state_bound_max_cup_angular_velocity == pytest.approx(50.0)
    assert abs(cfg.rewards.action_rate.weight) <= 0.002
    assert cfg.rewards.action_magnitude.func is mdp.action_l2
    assert cfg.rewards.action_magnitude.weight == pytest.approx(-0.05)

    agent = FrankaPourPPORunnerCfg()
    assert agent.class_name == "OnPolicyRunner"
    assert agent.save_interval == 50
    assert agent.actor.distribution_cfg.class_name == "GaussianDistribution"
    assert agent.actor.distribution_cfg.init_std == pytest.approx(0.1)
    assert agent.actor.distribution_cfg.std_type == "log"
    assert agent.actor.obs_normalization is False
    assert agent.critic.obs_normalization is False
    assert agent.clip_actions == pytest.approx(1.0)
    assert agent.algorithm.entropy_coef == pytest.approx(1.0e-3)
    assert agent.algorithm.num_learning_epochs == 5
    assert agent.algorithm.clip_param == pytest.approx(0.2)
    assert agent.algorithm.learning_rate == pytest.approx(1.0e-4)
    assert agent.algorithm.max_grad_norm == pytest.approx(1.0)
    assert agent.algorithm.schedule == "fixed"
    assert cfg.rewards.task_progress.params["discount_factor"] == pytest.approx(agent.algorithm.gamma)
    assert agent.obs_groups == {"actor": ["policy"], "critic": ["policy", "privileged"]}
    assert agent.logger == "wandb"
    assert agent.wandb_project == "franka-pour-mpm"


def test_backward_curriculum_config_is_complete_and_play_uses_randomized_task():
    cfg = FrankaPourEnvCfg()
    stage_count = len(cfg.curriculum_stage_names)

    assert cfg.is_finite_horizon is True
    assert isinstance(cfg.curriculum.stage, CurriculumTermCfg)
    assert cfg.curriculum.stage.func is mdp.PourCurriculum
    assert cfg.curriculum_stage_names == ("pour", "carry", "grasp", "full", "randomized")
    assert stage_count == 5
    assert len(cfg.curriculum_target_frac) == stage_count
    assert cfg.curriculum_target_frac == pytest.approx((0.1, 0.2, 0.3, 0.3, 0.3))
    assert list(cfg.curriculum_target_frac) == sorted(cfg.curriculum_target_frac)
    assert cfg.curriculum_start_stage == 0
    assert cfg.curriculum_freeze is False
    assert cfg.curriculum_min_resets_per_stage == 4096
    assert cfg.curriculum_previous_stage_replay_fraction == pytest.approx(0.1)
    assert cfg.curriculum_randomization_extent_levels == pytest.approx((2.0 / 3.0, 5.0 / 6.0, 1.0))
    assert cfg.curriculum_randomization_start_level == 0
    # Do not expose a generic stage identifier. The actor does receive behaviorally relevant
    # finite-state variables and the active delivery goal so the finite-horizon MDP is Markov.
    assert not hasattr(cfg.observations.policy, "curriculum_context")
    assert cfg.observations.policy.arm_reference_phase.func is mdp.arm_reference_phase_obs
    assert cfg.observations.policy.arm_reference_error.func is mdp.arm_reference_error_obs
    assert cfg.observations.policy.trajectory_status.func is mdp.trajectory_status_obs
    assert cfg.observations.policy.time_remaining.func is mdp.time_remaining_obs
    assert cfg.observations.policy.pour_target_fraction.func is mdp.pour_target_fraction_obs
    assert not hasattr(cfg.observations.policy, "success_dwell")
    assert not hasattr(cfg.observations.policy, "lost_grasp_dwell")
    assert not hasattr(cfg.observations.policy, "target_pose")
    assert not hasattr(cfg.observations.policy, "particle_fractions")
    assert not hasattr(cfg.observations.policy, "particle_transfer")
    assert cfg.observations.policy.arm_q.scale == pytest.approx(0.3)
    assert cfg.observations.policy.arm_qd.scale == pytest.approx(0.05)
    assert cfg.observations.policy.arm_reference_error.scale == pytest.approx(0.3)
    assert cfg.observations.policy.tcp_to_grasp_position_c.func is mdp.tcp_to_grasp_position_c_obs
    assert cfg.observations.policy.tcp_to_grasp_position_c.scale == pytest.approx(10.0)
    assert cfg.observations.policy.grasp_to_tcp_quat.func is mdp.grasp_to_tcp_quat_obs
    assert cfg.observations.policy.target_position_c.func is mdp.target_position_c_obs
    assert cfg.observations.policy.target_position_c.scale == pytest.approx(5.0)
    assert cfg.observations.policy.finger_position.func is mdp.finger_position_obs
    assert cfg.observations.policy.finger_position.scale == pytest.approx(25.0)
    assert cfg.observations.policy.finger_velocity.func is mdp.finger_velocity_obs
    assert cfg.observations.policy.finger_velocity.scale == pytest.approx(5.0)
    assert cfg.observations.policy.last_action.scale == pytest.approx(0.2)
    assert cfg.observations.policy.gripper_contact.func is mdp.gripper_contact_obs
    assert cfg.observations.policy.gripper_contact.scale == pytest.approx(250.0)
    assert cfg.observations.privileged.success_dwell.func is mdp.success_dwell_obs
    assert cfg.observations.privileged.lost_grasp_dwell.func is mdp.lost_grasp_dwell_obs
    assert cfg.observations.privileged.cup_velocity.func is mdp.cup_velocity_obs
    assert cfg.observations.privileged.particle_fractions.func is mdp.particle_fractions_obs
    assert cfg.observations.privileged.particle_transfer.func is mdp.particle_transfer_obs
    assert cfg.observations.privileged.held_delivery_history.func is mdp.held_delivery_history_obs
    assert cfg.actions.gripper_action.contact_min_deflection == pytest.approx(0.001)
    assert cfg.actions.gripper_action.contact_max_velocity == pytest.approx(0.05)

    agent = FrankaPourPPORunnerCfg()
    episode_steps = round(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
    documented_training_env_count = 512
    available_resets = documented_training_env_count * agent.max_iterations * agent.num_steps_per_env // episode_steps
    required_resets = cfg.curriculum_min_resets_per_stage * (
        stage_count - 1 + len(cfg.curriculum_randomization_extent_levels)
    )
    assert required_resets <= available_resets

    assert cfg.curriculum_randomized_source_position_range == pytest.approx((0.12, 0.10))
    assert cfg.curriculum_randomized_carry_position_range == pytest.approx((0.10, 0.10))
    assert cfg.curriculum_randomized_source_yaw_range == pytest.approx(math.radians(15.0))
    assert cfg.curriculum_randomized_target_center_xy == pytest.approx((0.50, -0.21))
    assert cfg.curriculum_randomized_target_position_range == pytest.approx((0.05, 0.05))
    assert cfg.curriculum_randomized_cup_clearance == pytest.approx(0.04)
    assert cfg.curriculum_randomized_reset_tcp_standoff == pytest.approx((0.0, 0.0, 0.12))
    assert cfg.curriculum_randomized_reset_tcp_jitter == pytest.approx((0.04, 0.04, 0.02))
    assert cfg.curriculum_randomized_reset_tcp_min_grasp_distance == pytest.approx(0.09)
    assert cfg.curriculum_randomized_pour_clearance == pytest.approx(0.01)
    assert cfg.curriculum_randomized_reset_ik_grid_size == 7
    assert cfg.curriculum_randomized_reset_ik_samples_per_source == 5

    arm_configs = (
        cfg.curriculum_pour_arm_q,
        cfg.curriculum_carry_arm_q,
        cfg.arm_home,
        cfg.arm_home,
        cfg.arm_home,
    )
    for arm_q in arm_configs:
        assert len(arm_q) == 7
        for joint_name, position in zip(cfg.actions.arm_action.joint_names, arm_q, strict=True):
            lower, upper = cfg.actions.arm_action.clip[joint_name]
            assert lower <= position <= upper

    for preset in (FrankaPourEnvCfg_PLAY(), FrankaPourEnvCfg_TELEOP()):
        assert preset.curriculum_start_stage == stage_count - 1
        assert preset.curriculum_randomization_start_level == len(preset.curriculum_randomization_extent_levels) - 1
        assert preset.curriculum_freeze is True


def test_finalize_resynchronizes_overridden_gripper_action_bound():
    cfg = FrankaPourEnvCfg()
    cfg.gripper_open_pos = 0.035
    cfg.success_max_tcp_distance = 0.02

    resolved = cfg.finalize()

    assert resolved.actions.gripper_action.open_position == pytest.approx(0.035)
    assert resolved.actions.arm_action.grasp_max_tcp_distance == pytest.approx(0.02)
    assert resolved.scene.robot.init_state.joint_pos["panda_finger_joint.*"] == pytest.approx(0.035)


def test_backward_curriculum_rejects_invalid_stage_and_arm_overrides():
    invalid_stage = FrankaPourEnvCfg()
    invalid_stage.curriculum_start_stage = 5
    with pytest.raises(ValueError, match="curriculum_start_stage"):
        invalid_stage.finalize()

    invalid_arm = FrankaPourEnvCfg()
    invalid_arm.curriculum_pour_arm_q = (10.0,) * 7
    with pytest.raises(ValueError, match="outside"):
        invalid_arm.finalize()

    invalid_replay = FrankaPourEnvCfg()
    invalid_replay.curriculum_previous_stage_replay_fraction = 1.0
    with pytest.raises(ValueError, match="curriculum_previous_stage_replay_fraction"):
        invalid_replay.finalize()

    invalid_randomization_start = FrankaPourEnvCfg()
    invalid_randomization_start.curriculum_randomization_start_level = 3
    with pytest.raises(ValueError, match="curriculum_randomization_start_level"):
        invalid_randomization_start.finalize()


@pytest.mark.parametrize(
    "field,value",
    [
        ("waypoint_count", 7),
        ("grasp_waypoint", 1),
        ("waypoint_phases", (0.0, 0.12, 0.24, 0.40, 0.40, 1.0)),
    ],
)
def test_backward_curriculum_rejects_arm_trajectory_layouts_it_cannot_author(field, value):
    cfg = FrankaPourEnvCfg()
    setattr(cfg.actions.arm_action, field, value)

    with pytest.raises(ValueError, match="waypoint|phases"):
        cfg.finalize()


@pytest.mark.parametrize(
    "levels",
    [
        (),
        (0.0, 1.0),
        (0.5, float("inf"), 1.0),
        (0.5, 0.5, 1.0),
        (0.75, 0.5, 1.0),
        (0.5, 0.9),
        (0.5, 1.1),
    ],
)
def test_randomized_curriculum_rejects_invalid_randomization_extent_levels(levels):
    cfg = FrankaPourEnvCfg()
    cfg.curriculum_randomization_extent_levels = levels

    with pytest.raises(ValueError, match="curriculum_randomization_extent_levels"):
        cfg.finalize()


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("target_tilt", 0.0),
        ("target_tilt", math.pi),
        ("pour_direction_xy", (0.0, 0.0)),
        ("pour_direction_xy", (0.0,)),
        ("alignment_radius", 0.0),
        ("active_through_stage", 5),
        ("discount_factor", 0.0),
    ],
)
def test_tilt_curriculum_rejects_invalid_configuration(parameter, value):
    cfg = FrankaPourEnvCfg()
    cfg.rewards.task_progress.params[parameter] = value

    with pytest.raises(ValueError, match=parameter):
        cfg.finalize()


@pytest.mark.parametrize(
    "field,value",
    [
        ("curriculum_randomized_source_position_range", (-0.1, 0.1)),
        ("curriculum_randomized_carry_position_range", (0.13, 0.10)),
        ("curriculum_randomized_source_yaw_range", math.pi / 2.0),
        ("curriculum_randomized_target_position_range", (0.05, float("inf"))),
        ("curriculum_randomized_cup_clearance", -0.01),
        ("curriculum_randomized_reset_tcp_standoff", (0.0, 0.05)),
        ("curriculum_randomized_reset_tcp_jitter", (0.01, -0.01, 0.01)),
        ("curriculum_randomized_reset_tcp_min_grasp_distance", 0.0),
        ("curriculum_grasp_descent_overshoot", 0.0),
        ("curriculum_randomized_reset_ik_grid_size", 1),
        ("curriculum_randomized_reset_ik_samples_per_source", 1),
        ("curriculum_randomized_reset_ik_samples_per_source", 4),
        ("curriculum_randomized_reset_ik_iterations", 0),
    ],
)
def test_randomized_curriculum_rejects_invalid_configuration(field, value):
    cfg = FrankaPourEnvCfg()
    setattr(cfg, field, value)

    with pytest.raises(ValueError, match=field):
        cfg.finalize()


def test_randomized_curriculum_rejects_ranges_without_collision_free_cup_placement():
    cfg = FrankaPourEnvCfg()
    cfg.curriculum_randomized_source_position_range = (0.10, 0.20)

    with pytest.raises(ValueError, match="no collision-free target y-position"):
        cfg.finalize()


def test_randomized_curriculum_rejects_tcp_jitter_that_can_cross_below_the_grasp():
    cfg = FrankaPourEnvCfg()
    cfg.curriculum_randomized_reset_tcp_standoff = (0.0, 0.0, 0.01)
    cfg.curriculum_randomized_reset_tcp_jitter = (0.01, 0.01, 0.02)

    with pytest.raises(ValueError, match="above the source-cup grasp point"):
        cfg.finalize()


def test_randomized_curriculum_rejects_tcp_box_below_minimum_grasp_distance():
    cfg = FrankaPourEnvCfg()
    cfg.curriculum_randomized_reset_tcp_min_grasp_distance = 0.11

    with pytest.raises(ValueError, match="cannot guarantee curriculum_randomized_reset_tcp_min_grasp_distance"):
        cfg.finalize()


@pytest.mark.parametrize(
    "field,value",
    [
        ("success_dwell_time_s", 0.0),
        ("success_min_lift_height", 0.0),
        ("success_max_tcp_distance", float("inf")),
        ("success_max_gripper_width_error", 0.0),
        ("state_bound_joint_position_margin", -0.1),
        ("state_bound_max_joint_velocity", 0.0),
        ("state_bound_max_cup_linear_velocity", float("inf")),
        ("state_bound_max_cup_angular_velocity", 0.0),
        ("curriculum_randomized_pour_clearance", -0.001),
    ],
)
def test_success_and_state_bounds_reject_invalid_configuration(field, value):
    cfg = FrankaPourEnvCfg()
    setattr(cfg, field, value)

    with pytest.raises(ValueError, match=field):
        cfg.finalize()


def test_finger_actuator_is_kept_and_teleop_disables_timeout():
    cfg = FrankaPourEnvCfg()
    hand = cfg.scene.robot.actuators["panda_hand"]
    assert hand.stiffness == pytest.approx(1500.0)
    assert hand.damping == pytest.approx(75.0)
    assert hand.armature == pytest.approx(0.0)
    assert FrankaPourEnvCfg_PLAY().scene.num_envs == 4
    assert FrankaPourEnvCfg_TELEOP().terminations.time_out is None


def _media_capacity(cfg):
    return _media_entry(cfg).solver_cfg.max_active_cell_count


def _aligned_particle_capacity(cfg):
    particle_count = cup_cavity_lattice(cfg)[0].shape[0]
    alignment = cfg.mpm_cell_capacity_alignment
    return ((particle_count + alignment - 1) // alignment) * alignment


def _media_entry(cfg):
    return next(entry for entry in cfg.sim.physics.solver_cfg.entries if entry.name == "media")


@pytest.mark.parametrize("num_envs", [1, 4, 8, 64, 200])
def test_sparse_capacity_is_particle_bounded_and_scales_exactly_per_world(num_envs):
    cfg = FrankaPourEnvCfg()
    cfg.scene.num_envs = num_envs
    configured_capacity = _media_capacity(cfg)

    resolved_capacity = _resolve_mpm_cell_cap(cfg)

    assert resolved_capacity == _aligned_particle_capacity(cfg) * num_envs
    particle_count = cup_cavity_lattice(cfg)[0].shape[0]
    assert particle_count * num_envs <= resolved_capacity
    assert resolved_capacity < (particle_count + cfg.mpm_cell_capacity_alignment) * num_envs
    assert _media_capacity(cfg) == configured_capacity


def test_sparse_capacity_tracks_media_particle_count_and_honors_exact_override():
    play_cfg = FrankaPourEnvCfg_PLAY()
    assert _resolve_mpm_cell_cap(play_cfg) == _aligned_particle_capacity(play_cfg) * play_cfg.scene.num_envs

    denser_cfg = FrankaPourEnvCfg()
    denser_cfg.particles_per_cell = 8.0
    assert _resolve_mpm_cell_cap(denser_cfg) == (_aligned_particle_capacity(denser_cfg) * denser_cfg.scene.num_envs)
    assert _resolve_mpm_cell_cap(denser_cfg) > _resolve_mpm_cell_cap(FrankaPourEnvCfg())

    play_cfg.mpm_cell_cap_override = 23456
    assert _resolve_mpm_cell_cap(play_cfg) == 23456


def test_sparse_capacity_rejects_nonpositive_alignment():
    cfg = FrankaPourEnvCfg()
    cfg.mpm_cell_capacity_alignment = 0

    with pytest.raises(ValueError, match="alignment"):
        _resolve_mpm_cell_cap(cfg)


def test_mpm_uses_pic27_colliders_and_bounded_200_env_capacity():
    cfg = FrankaPourEnvCfg()
    cfg.scene.num_envs = 200

    capacity = _resolve_mpm_cell_cap(cfg)
    upper_capacity = _resolve_mpm_upper_node_cap(cfg)
    solver_cfg = _media_entry(cfg.finalize()).solver_cfg

    # PIC27 bounds collider nodes by particle samples rather than the sparse-grid cell reserve,
    # while Q1 keeps the captured velocity solve compact.
    assert solver_cfg.velocity_basis == "Q1"
    assert solver_cfg.collider_basis == "pic27"
    assert solver_cfg.max_upper_node_count == upper_capacity
    assert capacity == _aligned_particle_capacity(cfg) * cfg.scene.num_envs
    assert upper_capacity == 64


@pytest.mark.parametrize("num_envs, expected", [(1, 32), (64, 32), (100, 32), (200, 64), (400, 128)])
def test_sparse_upper_capacity_scales_with_bounded_packed_worlds(num_envs, expected):
    cfg = FrankaPourEnvCfg()
    cfg.scene.num_envs = num_envs

    assert _resolve_mpm_upper_node_cap(cfg) == expected
    assert _media_entry(cfg.finalize()).solver_cfg.max_upper_node_count == expected

    cfg.mpm_upper_node_cap_override = 77
    assert _resolve_mpm_upper_node_cap(cfg) == 77


def test_coarse_voxel_resolution_finalizes_without_manual_hierarchy_overrides():
    cfg = FrankaPourEnvCfg(voxel_size=0.03)
    cfg.scene.num_envs = 1

    assert cfg.voxel_size == pytest.approx(0.03)
    particles, _ = cup_cavity_lattice(cfg)
    assert particles.shape[0] > 0

    solver_cfg = _media_entry(cfg.finalize()).solver_cfg

    assert solver_cfg.voxel_size == pytest.approx(0.03)
    assert solver_cfg.max_active_cell_count == _aligned_particle_capacity(cfg)
    assert solver_cfg.max_leaf_node_count == -1
    assert solver_cfg.max_lower_node_count == -1
    assert solver_cfg.max_upper_node_count == 32


def test_particle_workspace_bounds_contain_every_curriculum_reset():
    cfg = FrankaPourEnvCfg()
    assert cfg.terminations.particle_out_of_bounds.func is mdp.particle_out_of_bounds
    assert cfg.particle_max_velocity == pytest.approx(10.0)
    lower = torch.tensor(cfg.particle_workspace_lower_bound)
    upper = torch.tensor(cfg.particle_workspace_upper_bound)
    local_particles = torch.from_numpy(cup_cavity_lattice(cfg)[0])

    source_range = torch.tensor((*cfg.curriculum_randomized_source_position_range, 0.0))
    source_center = torch.tensor(cfg.cup_reset_pos)
    for signed_range in (-source_range, source_range):
        particles = local_particles + source_center + signed_range
        assert bool(torch.all(particles >= lower))
        assert bool(torch.all(particles <= upper))

    invalid = FrankaPourEnvCfg()
    invalid.particle_workspace_lower_bound = (1.0, 0.0, 0.0)
    invalid.particle_workspace_upper_bound = (0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="particle_workspace"):
        invalid.finalize()

    randomized_outside = FrankaPourEnvCfg()
    randomized_outside.particle_workspace_upper_bound = (0.55, 1.0, 1.5)
    with pytest.raises(ValueError, match="randomized source media"):
        randomized_outside.finalize()

    invalid_velocity = FrankaPourEnvCfg()
    invalid_velocity.particle_max_velocity = 0.0
    with pytest.raises(ValueError, match="particle_max_velocity"):
        invalid_velocity.finalize()

    invalid_spill_fraction = FrankaPourEnvCfg()
    invalid_spill_fraction.max_spill_fraction = 1.0
    with pytest.raises(ValueError, match="max_spill_fraction"):
        invalid_spill_fraction.finalize()

    invalid_spill_height = FrankaPourEnvCfg()
    invalid_spill_height.spill_table_height = float("inf")
    with pytest.raises(ValueError, match="spill_table_height"):
        invalid_spill_height.finalize()

    invalid_count_margin = FrankaPourEnvCfg()
    invalid_count_margin.particle_count_margin = -0.001
    with pytest.raises(ValueError, match="particle_count_margin"):
        invalid_count_margin.finalize()


def test_capacity_resolver_reads_fixed_grid_type_from_media_entry_without_mutation():
    cfg = FrankaPourEnvCfg()
    media = _media_entry(cfg)
    media.solver_cfg.grid_type = "fixed"
    arm = next(entry for entry in cfg.sim.physics.solver_cfg.entries if entry.name == "arm")
    arm.solver_cfg.max_active_cell_count = 777

    resolved_capacity = _resolve_mpm_cell_cap(cfg)

    assert resolved_capacity == 120000
    assert _media_capacity(cfg) == 120000
    assert arm.solver_cfg.max_active_cell_count == 777


def test_media_selector_includes_spill_floor_without_unrelated_shapes():
    cfg = FrankaPourEnvCfg().finalize()
    media = _media_entry(cfg)
    model = SimpleNamespace(
        body_label=[
            "/World/envs/env_0/TargetCup",
            "/World/envs/env_0/SpillFloor",
            "/World/envs/env_0/Robot/panda_hand",
        ],
        shape_count=4,
        shape_body=torch.tensor([0, 1, 2, -1], dtype=torch.int32),
        particle_count=3,
    )

    resolved = NewtonCoupledSolverManager._resolve_entry_cfg(model, media, cfg.scene)

    assert resolved.bodies == [1]
    assert resolved.shapes == [1]
    assert resolved.particles == [0, 1, 2]


@pytest.mark.parametrize(
    "task_id, cfg_name",
    [
        ("Isaac-Pour-Franka-v0", "FrankaPourEnvCfg"),
        ("Isaac-Pour-Franka-Play-v0", "FrankaPourEnvCfg_PLAY"),
        ("Isaac-Pour-Franka-Teleop-v0", "FrankaPourEnvCfg_TELEOP"),
    ],
)
def test_gym_registration_exactly_matches_task_entry_points(task_id, cfg_name):
    spec = gym.spec(task_id)
    assert spec.entry_point == "isaaclab_tasks.contrib.franka_pour.pour_env:FrankaPourEnv"
    assert spec.disable_env_checker is True
    expected_kwargs = {
        "env_cfg_entry_point": f"isaaclab_tasks.contrib.franka_pour.pour_env_cfg:{cfg_name}",
    }
    if task_id != "Isaac-Pour-Franka-Teleop-v0":
        expected_kwargs["rsl_rl_cfg_entry_point"] = (
            "isaaclab_tasks.contrib.franka_pour.config.franka.agents.rsl_rl_ppo_cfg:FrankaPourPPORunnerCfg"
        )
    assert spec.kwargs == expected_kwargs


def test_task_source_does_not_traverse_private_solver_state():
    source_path = Path(franka_pour.__file__).with_name("pour_env.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    private_manager_attrs = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr.startswith("_")
        and isinstance(node.value, ast.Name)
        and node.value.id in {"NewtonManager", "NewtonCoupledSolverManager"}
    }
    assert private_manager_attrs == set()


def test_tcp_pose_uses_public_robot_pose_data_and_configured_offset():
    source_path = Path(franka_pour.__file__).with_name("pour_env.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    tcp_pose = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "tcp_pose_e")

    attributes = {node.attr for node in ast.walk(tcp_pose) if isinstance(node, ast.Attribute)}
    assert "get_term" not in attributes
    assert "_compute_frame_pose" not in attributes
    assert {"body_link_pose_w", "root_link_pose_w"} <= attributes
    assert {"subtract_frame_transforms", "combine_frame_transforms"} <= attributes

    setup = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_setup_after_physics"
    )
    setup_source = ast.unparse(setup)
    assert "cfg.tcp_body_name" in setup_source
    assert "cfg.tcp_offset_pos" in setup_source
    assert "cfg.tcp_offset_rot" in setup_source
    assert "cfg.actions.arm_action.body_name" not in setup_source
    assert "cfg.actions.arm_action.body_offset" not in setup_source


def test_media_refill_is_batched_on_device_without_host_readback():
    source_path = Path(franka_pour.__file__).with_name("pour_env.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    sample_media = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_sample_cup_media"
    )

    attributes = {node.attr for node in ast.walk(sample_media) if isinstance(node, ast.Attribute)}
    assert {"cpu", "numpy", "tolist"}.isdisjoint(attributes)
    assert "quat_apply" in attributes


def test_task_reset_uses_public_asset_writers_and_body_pose_refresh():
    source_path = Path(franka_pour.__file__).with_name("pour_env.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

    function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "_reset_mpm_particle_state" not in function_names

    reset_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "reset_solver_state"
    ]
    assert len(reset_calls) == 1
    assert all(keyword.arg != "state" for keyword in reset_calls[0].keywords)

    reset = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "reset_pour_scene"
    )
    call_names = {
        node.func.attr
        for node in ast.walk(reset)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert {
        "write_joint_position_to_sim_index",
        "write_joint_velocity_to_sim_index",
        "write_root_pose_to_sim_index",
        "write_root_velocity_to_sim_index",
        "write_particle_pos_to_sim_index",
        "write_particle_velocity_to_sim_index",
        "reset_solver_state",
    } <= call_names
    assert {"eval_fk", "get_state_0", "get_state_1"}.isdisjoint(call_names)

    reset_attributes = {node.attr for node in ast.walk(reset) if isinstance(node, ast.Attribute)}
    assert "body_link_pose_w" in reset_attributes

    manager_forward_calls = [
        node
        for node in ast.walk(reset)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"forward", "forward_pending"}
    ]
    assert manager_forward_calls == []
