# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Construction tests for the Franka pour env config (no simulator)."""

import ast
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import NewtonCoupledManager
from isaaclab_newton.sim.schemas import MujocoJointCfg

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas import MassCfg, UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg

import isaaclab_tasks.contrib.franka_pour as franka_pour
import isaaclab_tasks.contrib.franka_pour.config.franka  # noqa: F401
import isaaclab_tasks.contrib.franka_pour.pour_env_cfg as pour_env_cfg
from isaaclab_tasks.contrib.franka_pour.config.franka.agents.rsl_rl_ppo_cfg import FrankaPourPPORunnerCfg
from isaaclab_tasks.contrib.franka_pour.cube_bowl_spawner_cfg import CubeBowlSpawnerCfg
from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import (
    TARGET_CUP_RIGID_LABEL_PATTERN,
    FrankaPourEnvCfg,
    FrankaPourEnvCfg_PLAY,
    FrankaPourEnvCfg_TELEOP,
    _resolve_mpm_cell_cap,
)
from isaaclab_tasks.contrib.franka_pour.reset_utils import boolean_selection_mask


def test_boolean_selection_mask_preserves_device_shape_and_dtype():
    selected = torch.tensor([[1, 3], [3, 4]], dtype=torch.long)

    mask = boolean_selection_mask(6, selected)

    assert mask.device == selected.device
    assert mask.dtype == torch.bool
    assert mask.shape == (6,)
    assert mask.tolist() == [False, True, False, True, True, False]


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
    assert resolved.sim.physics.solver_cfg.scene_cfg is resolved.scene
    assert _media_capacity(resolved) == 8 * max(resolved.mpm_min_cells_per_env, resolved.mpm_cells_per_env)


def test_finalize_propagates_final_cup_overrides_to_fresh_assets():
    original = FrankaPourEnvCfg()
    original.source_cup_inner_width = 0.041
    original.target_cup_inner_depth = 0.153
    original.cup_grasp_box_half = (0.021, 0.022, 0.043)
    original.cup_grasp_box_friction = 2.4
    original.target_cup_friction = 1.3
    original.cup_mass = 0.071
    original.cup_reset_pos = (0.43, 0.02, 0.01)
    original.target_cup_reset_pos = (0.47, -0.21, 0.02)

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
        "cup_grasp_box_half": "tuple[float, float, float]",
        "cup_reset_pos": "tuple[float, float, float]",
        "target_cup_reset_pos": "tuple[float, float, float]",
    }

    assert {name: FrankaPourEnvCfg.__annotations__[name] for name in expected} == expected


def test_cfg_routes_each_body_to_exactly_one_solver():
    cfg = FrankaPourEnvCfg().finalize()
    solver = cfg.sim.physics.solver_cfg
    entries = {entry.name: entry for entry in solver.entries}
    assert set(entries) == {"arm", "media"}
    assert solver.coupling_type == "proxy"

    arm = entries["arm"]
    media = entries["media"]
    assert arm.solver_cfg.integrator == "implicitfast"
    assert arm.solver_cfg.use_mujoco_contacts is False
    assert solver.use_collision_pipeline is True
    assert cfg.sim.physics.collision_cfg.soft_contact_max == 0
    assert arm.preserve_shape_ids is True
    assert arm.include_static_shapes is True
    assert arm.body_entities == [SceneEntityCfg("robot"), SceneEntityCfg("source_cup")]
    assert arm.body_label_patterns == [TARGET_CUP_RIGID_LABEL_PATTERN]
    assert media.all_particles is True
    assert media.body_entities == [SceneEntityCfg("target_cup")]
    assert media.body_label_patterns == [r".*/SpillFloor$"]
    assert media.include_static_shapes is False
    assert media.in_place is True
    assert media.solver_cfg.separate_worlds is True
    assert media.solver_cfg.grid_type == "sparse"
    assert media.solver_cfg.grid_padding == 0
    assert media.solver_cfg.max_active_cell_count > 0
    assert media.solver_cfg.solver == "jacobi"
    assert cfg.sim.physics.use_cuda_graph is True

    proxies = solver.proxy_coupling.proxies
    assert len(proxies) == 1
    assert proxies[0].source == "arm" and proxies[0].destination == "media"
    assert proxies[0].body_entities == [SceneEntityCfg("source_cup")]
    assert proxies[0].body_label_patterns == []
    assert not hasattr(pour_env_cfg, "CUP_LABEL_PATTERN")
    assert all("Cup$" not in pattern for entry in solver.entries for pattern in entry.body_label_patterns)


def test_visible_source_geometry_matches_grasp_proxy_and_fits_gripper():
    cfg = FrankaPourEnvCfg()
    outer_width = cfg.source_cup_inner_width + 2.0 * cfg.source_cup_wall_thickness
    outer_depth = cfg.source_cup_inner_depth + 2.0 * cfg.source_cup_wall_thickness
    outer_height = cfg.source_cup_cavity_depth + cfg.source_cup_bottom_thickness
    assert cfg.cup_grasp_box_half[:2] == pytest.approx((outer_width / 2.0, outer_depth / 2.0))
    assert cfg.cup_grasp_box_half[2] >= outer_height / 2.0
    assert outer_depth < 2.0 * cfg.gripper_open_pos
    assert cfg.grasp_contact_ke >= 1.0e5
    assert cfg.grasp_contact_kd >= 5.0e2
    assert cfg.cup_grasp_box_friction >= 2.0
    # A source deeper than its opening can retain a settled granular bed even at the Franka's
    # stable wrist limit. Keep it shallow enough to pour before the physical grasp destabilizes.
    assert cfg.source_cup_cavity_depth <= cfg.source_cup_inner_width
    assert cfg.source_cup_bottom_thickness < cfg.cup_grasp_height < outer_height


def test_dynamic_cup_free_joint_belongs_to_a_newton_articulation():
    source_path = Path(franka_pour.__file__).with_name("pour_env.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    add_cup_body = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_add_cup_body"
    )
    builder_calls = {
        node.func.attr
        for node in ast.walk(add_cup_body)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr.startswith("add_")
    }

    assert "add_link" in builder_calls
    assert "add_body" not in builder_calls
    assert "add_joint_free" in builder_calls
    assert "add_articulation" in builder_calls


def test_task_has_receiving_cup_safe_actions_and_success_threshold():
    cfg = FrankaPourEnvCfg()
    assert cfg.target_cup_reset_pos != cfg.cup_reset_pos
    assert cfg.actions.arm_action.scale <= 0.1
    # The scripted physical-grasp reference transfers ~41%; make the first-stage success bonus
    # reachable while the dense delivered-fraction reward can still drive higher transfer.
    assert abs(cfg.pour_target_frac - 0.4) < 1e-9
    assert cfg.episode_length_s >= 10.0


def test_training_curriculum_closes_by_default_and_prioritizes_lift_over_spill_avoidance():
    cfg = FrankaPourEnvCfg()
    assert type(cfg.actions.gripper_action).__name__ == "AbsBinaryJointPositionActionCfg"
    assert cfg.actions.gripper_action.positive_threshold is True
    assert cfg.actions.gripper_action.threshold >= 0.2
    assert cfg.rewards.lift.weight >= 8.0
    assert cfg.rewards.lift_command.weight >= cfg.rewards.lift.weight
    assert cfg.rewards.align_command.weight >= cfg.rewards.grasp.weight
    assert cfg.rewards.tilt_command.weight >= cfg.rewards.grasp.weight
    assert abs(cfg.rewards.spill.weight) <= 0.5
    assert abs(cfg.rewards.action_l2.weight) <= 0.002

    agent = FrankaPourPPORunnerCfg()
    assert agent.actor.distribution_cfg.init_std <= 0.1


def test_finger_actuator_is_kept_and_teleop_disables_timeout():
    cfg = FrankaPourEnvCfg()
    hand = cfg.scene.robot.actuators["panda_hand"]
    assert hand.armature == 0.5
    assert hand.damping * (1.0 / 120.0) / hand.armature < 2.0
    assert FrankaPourEnvCfg_PLAY().scene.num_envs == 4
    assert FrankaPourEnvCfg_TELEOP().terminations.time_out is None


def _media_capacity(cfg):
    return _media_entry(cfg).solver_cfg.max_active_cell_count


def _media_entry(cfg):
    return next(entry for entry in cfg.sim.physics.solver_cfg.entries if entry.name == "media")


@pytest.mark.parametrize("num_envs, expected", [(1, 16000), (4, 64000), (8, 128000), (64, 1024000)])
def test_sparse_capacity_scales_exactly_per_world(num_envs, expected):
    cfg = FrankaPourEnvCfg()
    cfg.scene.num_envs = num_envs
    configured_capacity = _media_capacity(cfg)

    resolved_capacity = _resolve_mpm_cell_cap(cfg)

    assert resolved_capacity == expected
    assert _media_capacity(cfg) == configured_capacity


def test_sparse_capacity_honors_play_floor_and_exact_override():
    play_cfg = FrankaPourEnvCfg_PLAY()
    assert _resolve_mpm_cell_cap(play_cfg) == 96000

    play_cfg.mpm_cell_cap_override = 23456
    assert _resolve_mpm_cell_cap(play_cfg) == 23456


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

    resolved = NewtonCoupledManager._resolve_entry_cfg(model, media, cfg.scene)

    assert resolved.bodies == [0, 1]
    assert resolved.shapes == [0, 1]
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
    assert spec.kwargs == {
        "env_cfg_entry_point": f"isaaclab_tasks.contrib.franka_pour.pour_env_cfg:{cfg_name}",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.contrib.franka_pour.config.franka.agents.rsl_rl_ppo_cfg:FrankaPourPPORunnerCfg"
        ),
    }


def test_task_source_does_not_traverse_private_solver_state():
    source_path = Path(franka_pour.__file__).with_name("pour_env.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    private_manager_attrs = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr.startswith("_")
        and isinstance(node.value, ast.Name)
        and node.value.id in {"NewtonManager", "NewtonCoupledManager"}
    }
    assert private_manager_attrs == set()


def test_tcp_pose_uses_public_robot_pose_data_and_configured_offset():
    source_path = Path(franka_pour.__file__).with_name("pour_env.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    tcp_pose = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "tcp_pose_e")

    attributes = {node.attr for node in ast.walk(tcp_pose) if isinstance(node, ast.Attribute)}
    assert "get_term" not in attributes
    assert "_compute_frame_pose" not in attributes
    assert {"body_link_pose_w", "root_pose_w"} <= attributes
    assert {"subtract_frame_transforms", "combine_frame_transforms"} <= attributes


def test_media_refill_is_batched_on_device_without_host_readback():
    source_path = Path(franka_pour.__file__).with_name("pour_env.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    sample_media = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_sample_cup_media"
    )

    attributes = {node.attr for node in ast.walk(sample_media) if isinstance(node, ast.Attribute)}
    assert {"cpu", "numpy", "tolist"}.isdisjoint(attributes)
    assert "quat_apply" in attributes


def test_task_reset_delegates_history_and_masks_forward_kinematics():
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

    fk_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "eval_fk"
    ]
    assert fk_calls
    assert all(
        len(call.args) >= 5 and not (isinstance(call.args[4], ast.Constant) and call.args[4].value is None)
        for call in fk_calls
    )
