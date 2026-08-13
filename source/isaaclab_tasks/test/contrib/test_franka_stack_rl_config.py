# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import sys
import textwrap
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg
from tensordict import TensorDict

from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.schemas import CollisionBaseCfg, RigidBodyBaseCfg, UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.visualizers import VisualizerCfg

from isaaclab_rl.entrypoints.common import resolve_play_task_name

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.stack import mdp, spawners
from isaaclab_tasks.contrib.stack.mdp import (
    ResetBufferedGripperActionCfg,
    WorkspaceBoundedRelativeJointPositionActionCfg,
    reset_events,
)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_MENAGERIE_CFG


def test_franka_stack_rl_config_uses_physical_reset_table_and_compensated_arm():
    cfg = parse_env_cfg("IsaacContrib-Stack-Cube-Franka-RL", device="cuda:0", num_envs=8)

    assert cfg.scene.num_envs == 8
    assert cfg.scene.replicate_physics
    assert isinstance(cfg.sim.physics, NewtonCfg)
    assert isinstance(cfg.sim.physics.solver_cfg, MJWarpSolverCfg)
    assert cfg.sim.physics.solver_cfg.njmax == 256
    assert cfg.sim.physics.solver_cfg.nconmax == 100
    # Match the proven Lift integration cadence: two 5 ms solver steps for
    # each 10 ms physics tick, and two physics ticks per policy action.
    assert cfg.sim.physics.num_substeps == 2
    assert cfg.decimation == 2
    assert cfg.sim.render_interval == 2
    # External contacts must be refreshed for each 5 ms solver step. Reusing
    # the 10 ms contact set lets a released three-cube tower drift apart.
    assert cfg.sim.physics.collision_decimation == 1
    assert not cfg.sim.physics.solver_cfg.use_mujoco_contacts
    assert isinstance(cfg.sim.physics.collision_cfg, NewtonCollisionPipelineCfg)
    assert cfg.sim.physics.solver_cfg.iterations == 100
    assert cfg.sim.physics.solver_cfg.ls_iterations == 50
    assert cfg.sim.physics.solver_cfg.impratio == 10.0
    assert cfg.sim.physics.solver_cfg.integrator == "implicitfast"
    assert cfg.sim.physics.solver_cfg.cone == "elliptic"
    assert cfg.sim.physics.use_cuda_graph
    assert cfg.sim.physics.default_shape_cfg.margin == 0.0
    assert cfg.sim.physics.default_shape_cfg.gap == 0.0
    assert cfg.sim.gravity == (0.0, 0.0, -9.81)
    assert isinstance(cfg.sim.default_visualizer_cfg, VisualizerCfg)
    cube_colors = [
        cfg.scene.cube_1.spawn.display_color,
        cfg.scene.cube_2.spawn.display_color,
        cfg.scene.cube_3.spawn.display_color,
    ]
    assert len(set(cube_colors)) == 3
    for cube_id, cube_cfg in enumerate((cfg.scene.cube_1, cfg.scene.cube_2, cfg.scene.cube_3), start=1):
        assert isinstance(cube_cfg.spawn.rigid_props, RigidBodyBaseCfg)
        assert isinstance(cube_cfg.spawn.collision_props, CollisionBaseCfg)
        assert cube_cfg.spawn.collision_props.contact_offset == 0.0
        assert cube_cfg.spawn.collision_props.rest_offset == 0.0
        assert isinstance(cube_cfg.spawn.physics_material, NewtonMaterialPropertiesCfg)
        assert not cube_cfg.spawn.rigid_props.disable_gravity
        assert cube_cfg.spawn.physics_material.static_friction == 1.0
        assert cube_cfg.spawn.physics_material.dynamic_friction == 0.8
        # Keep the cube contacts stiff enough that a settled three-cube tower
        # does not visibly shorten under gravity.
        assert cube_cfg.spawn.physics_material.contact_stiffness == 1.0e4
        assert cube_cfg.spawn.physics_material.contact_damping == 200.0
        assert cube_cfg.spawn.visual_material is None
        assert cube_cfg.spawn.semantic_tags == [("class", f"cube_{cube_id}")]
    assert not hasattr(cfg.rewards, "progress")
    assert not hasattr(cfg.rewards, "reaching_cube")
    assert not hasattr(cfg.rewards, "lifting_cube")
    assert not hasattr(cfg.rewards, "cube_goal_tracking")
    assert cfg.rewards.success.func.__name__ == "stack_success_pulse"
    assert cfg.rewards.success.params == {"context_term_name": "progress_context"}
    assert cfg.rewards.success.weight == 2.0
    assert cfg.rewards.failure.func.__name__ == "irrecoverable_stack_failure"
    assert cfg.rewards.failure.params == {"success_termination_name": "success"}
    assert cfg.rewards.failure.weight == -2.0e-4
    assert cfg.events.reset_all.func.__name__ == "reset_scene_to_default"
    assert cfg.events.reset_from_state_buffer.func.__name__ == "StackResetStateTable"
    assert cfg.events.reset_from_state_buffer.params["fixed_row_id"] is None
    assert cfg.events.reset_from_state_buffer.params["fixed_recipe"] is None
    assert cfg.events.reset_from_state_buffer.params["table_evaluation_env_fraction"] == 0.0
    assert cfg.events.reset_from_state_buffer.params["fixed_role_permutation"] is None
    assert cfg.events.reset_from_state_buffer.params["table_rows_per_layout"] == 64
    assert cfg.events.reset_from_state_buffer.params["table_target_potential"] is None
    assert cfg.events.reset_from_state_buffer.params["arm_joint_noise_range"] == 0.0
    assert cfg.events.reset_from_state_buffer.params["table_arm_joint_noise_range"] == 0.080
    assert cfg.events.reset_from_state_buffer.params["table_cube_planar_translation_range"] == 0.015
    assert cfg.events.reset_from_state_buffer.params["table_cube_rotation_range"] == 0.45
    assert cfg.events.reset_from_state_buffer.params["force_full_goal"]
    assert cfg.events.reset_from_state_buffer.params["continuation_probability"] == 1.0
    assert cfg.events.reset_from_state_buffer.params["fixed_continue_to_final"]
    assert not hasattr(cfg.events, "variable_gravity")
    assert isinstance(cfg.actions.arm_action, WorkspaceBoundedRelativeJointPositionActionCfg)
    assert cfg.actions.arm_action.gravity_compensation
    assert cfg.sim.render_interval == cfg.decimation
    assert cfg.actions.arm_action.scale == 0.05
    assert cfg.actions.arm_action.max_delta == 0.05
    assert cfg.actions.arm_action.workspace_lower == (-0.303, -0.200, -0.328, -2.759, -0.124, 2.393, 0.271)
    assert cfg.actions.arm_action.workspace_upper == (0.450, 0.603, 0.147, -2.000, 0.345, 3.112, 1.200)
    assert cfg.scene.robot.spawn.usd_path == FRANKA_PANDA_MENAGERIE_CFG.spawn.usd_path
    assert not cfg.scene.robot.spawn.rigid_props.disable_gravity
    assert set(cfg.scene.robot.actuators) == {"panda_arm", "panda_hand"}
    arm = cfg.scene.robot.actuators["panda_arm"]
    assert arm.joint_names_expr == ["panda_joint[1-7]"]
    assert arm.effort_limit_sim == {"panda_joint[1-4]": 87.0, "panda_joint[5-7]": 12.0}
    assert arm.velocity_limit is None
    assert arm.velocity_limit_sim == {"panda_joint[1-4]": 2.175, "panda_joint[5-7]": 2.61}
    assert arm.stiffness == {
        "panda_joint[1-4]": 600.0,
        "panda_joint5": 250.0,
        "panda_joint6": 150.0,
        "panda_joint7": 50.0,
    }
    assert arm.damping == {
        "panda_joint[1-4]": 50.0,
        "panda_joint5": 30.0,
        "panda_joint6": 25.0,
        "panda_joint7": 15.0,
    }
    assert arm.armature == {
        "panda_joint[1-2]": 0.6057,
        "panda_joint[3-4]": 0.4625,
        "panda_joint[5-7]": 0.2055,
    }
    hand = cfg.scene.robot.actuators["panda_hand"]
    assert hand.joint_names_expr == ["panda_finger_joint[1-2]"]
    assert hand.effort_limit_sim == 70.0
    assert hand.velocity_limit is None
    assert hand.velocity_limit_sim == 2.0
    assert hand.stiffness == 350.0
    assert hand.damping == 175.0
    assert hand.armature == 0.1
    assert cfg.actions.gripper_action.close_command_expr == {"panda_finger_.*": 0.0}
    assert isinstance(cfg.actions.gripper_action, ResetBufferedGripperActionCfg)
    assert cfg.actions.gripper_action.force_close_steps == 5
    assert cfg.events.reset_from_state_buffer.params["closed_finger_position"] == 0.020
    assert cfg.events.reset_from_state_buffer.params["placed_finger_position"] == 0.021
    assert cfg.rewards.action_l2.func.__name__ == "action_term_l2"
    assert cfg.rewards.action_l2.params == {"action_name": "arm_action"}
    assert cfg.rewards.action_l2.weight == -1.0e-4
    assert cfg.rewards.action_rate.weight == -1.0e-4
    assert cfg.rewards.joint_vel.func.__name__ == "finite_joint_velocity_l2"
    assert cfg.rewards.joint_vel.weight == -1.0e-4
    assert cfg.rewards.joint_vel.params["asset_cfg"].joint_names == ["panda_joint.*"]
    assert cfg.rewards.joint_vel.params["maximum_velocity"] == 3.0
    assert cfg.curriculum.reset_sampling.func.__name__ == "StackResetTableCurriculum"
    assert cfg.curriculum.reset_sampling.params["success_context_name"] == "learning_progress_context"
    assert cfg.curriculum.reset_sampling.params["final_success_context_name"] == "progress_context"
    success_monitor = cfg.curriculum.reset_sampling.params["success_monitor"]
    assert isinstance(success_monitor, mdp.SuccessMonitorCfg)
    assert success_monitor.monitored_history_len == 50
    assert success_monitor.target_success_rate == 0.5
    assert success_monitor.kappa == 1.0
    assert success_monitor.temperature == 1.0
    assert cfg.curriculum.reset_sampling.params["table_sampling_probability"] == 0.35
    assert cfg.curriculum.reset_sampling.params["global_sampling"] is False
    assert cfg.terminations.progress_context.func.__name__ == "StableOrderInvariantStackGoal"
    assert cfg.terminations.progress_context.params == {
        "minimum_episode_steps": 3,
        "hold_steps": 5,
        "maximum_cube_velocity": 0.10,
        "minimum_finger_release_position": 0.023,
    }
    assert cfg.terminations.learning_progress_context.func.__name__ == "StackResetLearningProgress"
    assert cfg.terminations.learning_progress_context.params == {"minimum_episode_steps": 3}
    assert cfg.terminations.success.func.__name__ == "success_after_minimum_horizon"
    assert cfg.terminations.success.params == {
        "context_term_name": "progress_context",
        "minimum_episode_length_s": 0.1,
    }
    assert cfg.terminations.time_out.func.__name__ == "time_out"
    assert cfg.terminations.time_out.params == {}
    assert cfg.terminations.time_out.time_out
    assert cfg.terminations.nonfinite_robot_state.func.__name__ == "nonfinite_robot_state"
    assert cfg.terminations.nonfinite_cube_state.func.__name__ == "nonfinite_cube_state"
    assert cfg.terminations.cube_workspace_invalid.func.__name__ == "cube_out_of_workspace"
    assert cfg.episode_length_s == 20.0
    assert cfg.observations.policy.concatenate_terms
    assert cfg.observations.policy.object.func.__name__ == "role_conditioned_stack_obs"
    assert cfg.observations.policy.eef_quat is None
    assert cfg.observations.policy.eef_axes.func.__name__ == "franka_ee_axes"
    assert cfg.observations.policy.eef_velocity.func.__name__ == "franka_ee_velocity"
    assert cfg.observations.policy.stack_state is None
    assert cfg.observations.policy.joint_pos.func.__name__ == "joint_pos_rel"
    assert cfg.observations.policy.joint_pos.params["asset_cfg"].joint_names == ["panda_joint.*"]
    assert cfg.observations.policy.joint_vel.func.__name__ == "joint_vel_rel"
    assert cfg.observations.policy.joint_vel.params["asset_cfg"].joint_names == ["panda_joint.*"]
    assert cfg.observations.rgb_camera is None
    assert cfg.observations.subtask_terms is None
    assert cfg.scene.ee_frame is None
    assert cfg.scene.robot.spawn.usd_path == FRANKA_PANDA_MENAGERIE_CFG.spawn.usd_path
    assert isinstance(cfg.scene.table.spawn, UsdFileCfg)
    assert cfg.scene.table.spawn.usd_path.endswith("/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    assert len(cfg.scene.table.spawn.rigid_props) == 1
    assert isinstance(cfg.scene.table.spawn.rigid_props[0], UsdPhysicsRigidBodyCfg)
    assert cfg.scene.table.spawn.rigid_props[0].kinematic_enabled
    assert cfg.scene.table.init_state.pos == [0.5, 0, 0]
    assert cfg.scene.table.init_state.rot == [0, 0, 0.707, 0.707]
    table_contact_surface = cfg.scene.table_contact_surface
    assert isinstance(table_contact_surface, AssetBaseCfg)
    assert table_contact_surface.prim_path == "{ENV_REGEX_NS}/TableContactSurface"
    assert table_contact_surface.init_state.pos == (0.3439, 0.0, -0.02)
    assert isinstance(table_contact_surface.spawn, sim_utils.CuboidCfg)
    assert table_contact_surface.spawn.size == (1.28, 0.91, 0.04)
    assert table_contact_surface.spawn.visible is False
    assert isinstance(table_contact_surface.spawn.rigid_props, RigidBodyBaseCfg)
    assert table_contact_surface.spawn.rigid_props.kinematic_enabled
    assert isinstance(table_contact_surface.spawn.collision_props, CollisionBaseCfg)
    assert table_contact_surface.spawn.collision_props.contact_offset == 0.0
    assert table_contact_surface.spawn.collision_props.rest_offset == 0.0
    assert isinstance(table_contact_surface.spawn.physics_material, NewtonMaterialPropertiesCfg)
    assert table_contact_surface.spawn.physics_material.static_friction == 1.0
    assert table_contact_surface.spawn.physics_material.dynamic_friction == 0.8
    assert table_contact_surface.spawn.physics_material.contact_stiffness == 1.0e4
    assert table_contact_surface.spawn.physics_material.contact_damping == 200.0
    assert cfg.sim.default_visualizer_cfg.eye == (1.4, 1.4, 0.9)
    assert cfg.sim.default_visualizer_cfg.lookat == (0.5, 0.0, 0.1)
    assert cfg.scene.cube_1.spawn.size == (0.04, 0.04, 0.04)
    assert cfg.scene.cube_2.spawn.size == (0.04, 0.04, 0.04)
    assert cfg.scene.cube_3.spawn.size == (0.04, 0.04, 0.04)
    assert cfg.scene.cube_1.spawn.physics_material.static_friction == 1.0
    assert cfg.scene.cube_1.spawn.physics_material.dynamic_friction == 0.8


def test_franka_stack_rl_config_rejects_inconsistent_runtime_contracts():
    cfg = load_cfg_from_registry("IsaacContrib-Stack-Cube-Franka-RL", "env_cfg_entry_point")
    cfg.actions.arm_action.scale = cfg.actions.arm_action.max_delta + 0.01
    with pytest.raises(ValueError, match="hidden saturation"):
        cfg.validate()

    cfg = load_cfg_from_registry("IsaacContrib-Stack-Cube-Franka-RL", "env_cfg_entry_point")
    cfg.sim.render_interval = cfg.decimation + 1
    with pytest.raises(ValueError, match="render_interval must equal decimation"):
        cfg.validate()

    cfg = load_cfg_from_registry("IsaacContrib-Stack-Cube-Franka-RL", "env_cfg_entry_point")
    cfg.sim.physics = SimpleNamespace()
    with pytest.raises(TypeError, match="require the Newton physics backend"):
        cfg.validate()

    cfg = load_cfg_from_registry("IsaacContrib-Stack-Cube-Franka-RL", "env_cfg_entry_point")
    cfg.scene.cube_1.spawn.semantic_tags = None
    with pytest.raises(ValueError, match="semantic_tags"):
        cfg.validate()


def test_franka_stack_camera_config_rejects_stale_reset_frames():
    cfg = load_cfg_from_registry("IsaacContrib-Stack-Cube-Franka-RL-Camera", "env_cfg_entry_point")
    cfg.num_rerenders_on_reset = 0

    with pytest.raises(ValueError, match="num_rerenders_on_reset"):
        cfg.validate()


def test_franka_stack_rl_config_resolution_is_omni_and_pxr_free():
    """Keep runtime action and USD imports deferred until after Kit launches."""
    program = textwrap.dedent(
        """
        import sys

        class _Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("omni", "pxr"):
                    raise ImportError(f"BLOCKED eager import of {name!r} before launch_simulation")
                return None

        sys.meta_path.insert(0, _Blocker())

        import isaaclab_tasks
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        cfg = parse_env_cfg("IsaacContrib-Stack-Cube-Franka-RL", device="cuda:0", num_envs=1)
        cfg.play_mode()
        """
    )
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"RL stack config eagerly imported omni/pxr:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_colored_cube_spawner_colors_standard_collision_geometry():
    """Use one standard cuboid for physics and rendering without duplicate geometry."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Cube")
    collider = UsdGeom.Cube.Define(stage, "/World/Cube/geometry/mesh")
    collider.CreateSizeAttr(0.04)
    UsdGeom.Xformable(collider).AddScaleOp().Set(Gf.Vec3d(1.0, 1.25, 1.5))
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    cfg = spawners.ColoredCuboidCfg(
        size=(0.04, 0.05, 0.06),
        display_color=(0.05, 0.15, 0.80),
    )

    spawners._author_display_color("/World/Cube", cfg, stage)

    collider_prim = stage.GetPrimAtPath("/World/Cube/geometry/mesh")
    assert collider_prim.IsValid()
    assert collider_prim.IsA(UsdGeom.Cube)
    assert collider_prim.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdGeom.Imageable(collider_prim).ComputePurpose() == UsdGeom.Tokens.default_
    assert tuple(UsdGeom.Gprim(collider_prim).GetDisplayColorPrimvar().Get()[0]) == pytest.approx(cfg.display_color)
    assert not stage.GetPrimAtPath("/World/Cube/geometry/visual").IsValid()


def test_franka_stack_rl_registration_exposes_rsl_rl_ppo_config():
    spec = gym.spec("IsaacContrib-Stack-Cube-Franka-RL")
    agent_cfg = load_cfg_from_registry("IsaacContrib-Stack-Cube-Franka-RL", "rsl_rl_cfg_entry_point")

    assert "IsaacContrib-Stack-Cube-Franka-Newton-DiffIK-Rel-v0" not in gym.registry
    assert "IsaacContrib-Stack-Cube-Franka-Newton-IK-Rel-v0" not in gym.registry
    assert spec.kwargs["env_cfg_entry_point"].endswith(":FrankaCubeStackRLEnvCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":FrankaStackPPORunnerCfg")
    assert agent_cfg.experiment_name == "franka_stack"
    assert agent_cfg.obs_groups == {"actor": ["policy"], "critic": ["policy"]}
    assert agent_cfg.actor.hidden_dims == [512, 256, 128]
    assert agent_cfg.actor.activation == "elu"
    assert agent_cfg.actor.obs_normalization
    assert agent_cfg.actor.distribution_cfg.init_std == 0.45
    assert agent_cfg.actor.distribution_cfg.std_range == (0.15, 0.65)
    assert agent_cfg.actor.distribution_cfg.class_name.endswith(":StackGaussianDistribution")
    assert not hasattr(agent_cfg.actor.distribution_cfg, "gripper_std")
    assert agent_cfg.clip_actions == 1.0
    assert agent_cfg.save_interval == 25
    assert agent_cfg.num_steps_per_env == 32
    assert agent_cfg.max_iterations == 7000
    assert agent_cfg.algorithm.num_mini_batches == 16
    assert agent_cfg.algorithm.entropy_coef == 0.001
    assert agent_cfg.algorithm.class_name.endswith(":StackPPO")
    assert agent_cfg.algorithm.learning_rate == 1.0e-4
    assert agent_cfg.algorithm.schedule == "fixed"
    assert agent_cfg.algorithm.gamma == 0.999
    assert agent_cfg.algorithm.symmetry_cfg is None
    assert agent_cfg.critic.hidden_dims == [512, 256, 128]
    assert agent_cfg.critic.activation == "elu"
    assert agent_cfg.critic.obs_normalization


def test_franka_stack_camera_actor_is_deployable_and_critic_is_asymmetric():
    """Keep simulator cube state out of every actor observation group."""
    task_name = "IsaacContrib-Stack-Cube-Franka-RL-Camera"
    spec = gym.spec(task_name)
    cfg = parse_env_cfg(task_name, device="cuda:0", num_envs=8)
    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")

    assert spec.kwargs["env_cfg_entry_point"].endswith(":FrankaCubeStackCameraRLEnvCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":FrankaStackCameraPPORunnerCfg")
    assert cfg.scene.num_envs == 8
    assert cfg.scene.base_camera.data_types == ["rgb"]
    assert cfg.scene.base_camera.width == 128
    assert cfg.scene.base_camera.height == 128
    assert cfg.scene.base_camera.offset.convention == "opengl"
    assert cfg.scene.base_camera.renderer_cfg.renderer_type == "newton_warp"
    assert cfg.num_rerenders_on_reset == 1

    assert set(vars(cfg.observations.policy)) >= {
        "actions",
        "joint_pos",
        "joint_vel",
        "gripper_pos",
        "eef_velocity",
        "eef_axes",
        "eef_position",
    }
    assert not hasattr(cfg.observations.policy, "object")
    assert not hasattr(cfg.observations.policy, "cube_positions")
    assert not hasattr(cfg.observations.policy, "cube_orientations")
    assert cfg.observations.privileged.object.func is mdp.role_conditioned_stack_obs
    assert cfg.observations.base_image.rgb.func is mdp.TemporalNormalizedRgbImage
    assert cfg.observations.base_image.rgb.params["history_length"] == 2
    assert cfg.observations.policy.enable_corruption
    assert not cfg.observations.base_image.enable_corruption

    assert agent_cfg.experiment_name == "franka_stack_camera"
    assert agent_cfg.obs_groups == {
        "actor": ["policy", "base_image"],
        "critic": ["policy", "privileged"],
    }
    assert "privileged" not in agent_cfg.obs_groups["actor"]
    assert agent_cfg.actor.class_name.endswith(":SpatialSoftmaxCNNModel")
    assert agent_cfg.actor.cnn_cfg.output_channels == [16, 32, 32]
    assert agent_cfg.actor.distribution_cfg.class_name.endswith(":StackGaussianDistribution")
    assert agent_cfg.algorithm.schedule == "fixed"
    assert agent_cfg.algorithm.learning_rate == 7.0e-5
    assert agent_cfg.algorithm.num_mini_batches == 8
    assert agent_cfg.algorithm.entropy_coef == 0.005
    assert agent_cfg.max_iterations == 15000

    # The camera variant changes only the policy interface and rendering.
    state_cfg = parse_env_cfg("IsaacContrib-Stack-Cube-Franka-RL", device="cuda:0", num_envs=8)
    assert cfg.actions == state_cfg.actions
    assert cfg.rewards == state_cfg.rewards
    assert cfg.curriculum == state_cfg.curriculum
    assert cfg.terminations == state_cfg.terminations
    assert cfg.events.reset_from_state_buffer.func is state_cfg.events.reset_from_state_buffer.func
    for name, value in state_cfg.events.reset_from_state_buffer.params.items():
        if name == "fixed_role_permutation":
            assert cfg.events.reset_from_state_buffer.params[name] == 0
        else:
            assert cfg.events.reset_from_state_buffer.params[name] == value


def test_franka_stack_camera_distillation_routes_privilege_only_to_teacher():
    """Match the trained state actor exactly while isolating student inputs."""
    task_name = "IsaacContrib-Stack-Cube-Franka-RL-Camera-Distillation"
    spec = gym.spec(task_name)
    cfg = parse_env_cfg(task_name, device="cuda:0", num_envs=8)
    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    state_cfg = parse_env_cfg("IsaacContrib-Stack-Cube-Franka-RL", device="cuda:0", num_envs=8)

    assert spec.kwargs["env_cfg_entry_point"].endswith(":FrankaCubeStackCameraDistillationEnvCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":FrankaStackCameraDistillationRunnerCfg")
    assert agent_cfg.class_name == "DistillationRunner"
    assert agent_cfg.obs_groups == {
        "student": ["policy", "base_image"],
        "teacher": ["teacher"],
    }
    assert "teacher" not in agent_cfg.obs_groups["student"]
    assert agent_cfg.student.class_name.endswith(":StackVisualDistillationModel")
    assert agent_cfg.student.visual_target_indices == (*range(22, 31), *range(40, 67), 85)
    assert agent_cfg.student.teacher_object_start == 22
    assert agent_cfg.student.obs_normalization
    assert agent_cfg.student.distribution_cfg.class_name.endswith(":StackDistillationDistribution")
    assert agent_cfg.student.distribution_cfg.init_std == 0.05
    assert agent_cfg.student.distribution_cfg.std_range == (0.02, 0.12)
    assert agent_cfg.teacher.class_name == "MLPModel"
    assert agent_cfg.teacher.hidden_dims == [512, 256, 128]
    assert agent_cfg.teacher.obs_normalization
    assert agent_cfg.teacher.distribution_cfg.class_name.endswith(":StackGaussianDistribution")
    assert agent_cfg.algorithm.loss_type == "huber"
    assert agent_cfg.algorithm.class_name.endswith(":StackDistillation")
    assert agent_cfg.algorithm.learning_rate == 1.0e-4
    assert agent_cfg.algorithm.schedule == "fixed"
    assert agent_cfg.algorithm.desired_kl == 0.01
    assert agent_cfg.algorithm.adaptive_learning_rate_min == 1.0e-5
    assert agent_cfg.algorithm.adaptive_learning_rate_max == 3.0e-4
    assert agent_cfg.algorithm.adaptive_learning_rate_factor == 1.5
    assert agent_cfg.algorithm.kl_measurement_samples == 2048
    assert agent_cfg.algorithm.gradient_length == 1
    assert agent_cfg.algorithm.num_learning_epochs == 5
    assert agent_cfg.algorithm.teacher_pretrain_updates == 40
    assert agent_cfg.algorithm.dagger_gate_recipe_ids == (0,)
    assert agent_cfg.algorithm.dagger_gate_success_rate == 0.95
    assert agent_cfg.algorithm.dagger_gate_min_attempts == 32
    assert not agent_cfg.algorithm.dagger_success_gate
    assert agent_cfg.algorithm.student_control_fraction_start == 0.25
    assert agent_cfg.algorithm.student_control_fraction_end == 0.25
    assert agent_cfg.algorithm.stepwise_student_control
    assert agent_cfg.algorithm.student_control_anneal_updates == 900
    assert agent_cfg.algorithm.student_control_feedback_gain == 0.5
    assert agent_cfg.algorithm.evaluation_success_ema_alpha == 0.25
    assert agent_cfg.algorithm.recipe_count == 9
    assert agent_cfg.algorithm.recipe_names[0] == "final_release"
    assert agent_cfg.algorithm.recipe_names[-1] == "table"
    assert agent_cfg.algorithm.recipe_balance
    assert agent_cfg.algorithm.table_recipe_weight == 3.0
    assert agent_cfg.algorithm.student_state_loss_weight == 3.0
    assert agent_cfg.algorithm.controller_warmup_updates == 40
    assert agent_cfg.algorithm.distillation_context_obs_group == "distillation_context"
    assert agent_cfg.algorithm.arm_loss_weight == 1.0
    assert agent_cfg.algorithm.gripper_loss_weight == 2.0
    assert agent_cfg.algorithm.auxiliary_loss_weight == 0.5
    assert agent_cfg.algorithm.action_clip == 1.0
    assert agent_cfg.algorithm.evaluation_envs_per_recipe == 4
    assert agent_cfg.algorithm.success_reward_threshold == 1.0
    assert cfg.events.reset_from_state_buffer.params["fixed_role_permutation"] == 0
    assert cfg.events.reset_from_state_buffer.params["evaluation_recipe_ids"] == tuple(range(9))
    assert cfg.events.reset_from_state_buffer.params["evaluation_envs_per_recipe"] == 4
    assert cfg.curriculum.reset_sampling.params["evaluation_env_count"] == 36
    assert not agent_cfg.init_at_random_ep_len
    assert cfg.observations.distillation_context.recipe.func is mdp.stack_reset_recipe_one_hot
    assert cfg.events.camera_calibration.params["eye_position_noise"] == (0.0, 0.0, 0.0)
    assert cfg.events.camera_calibration.params["lookat_position_noise"] == (0.0, 0.0, 0.0)
    assert not cfg.observations.policy.enable_corruption
    assert cfg.observations.policy.joint_pos.noise is None
    assert cfg.observations.policy.joint_vel.noise is None
    assert cfg.observations.policy.gripper_pos.noise is None
    assert not hasattr(cfg.observations, "teacher_swapped")
    assert "distillation_context" not in agent_cfg.obs_groups["student"]

    state_terms = [
        name for name, term in vars(state_cfg.observations.policy).items() if term is not None and hasattr(term, "func")
    ]
    teacher_terms = [
        name for name, term in vars(cfg.observations.teacher).items() if term is not None and hasattr(term, "func")
    ]
    assert (
        teacher_terms
        == state_terms
        == [
            "actions",
            "joint_pos",
            "joint_vel",
            "object",
            "gripper_pos",
            "eef_velocity",
            "eef_axes",
        ]
    )
    for name in teacher_terms:
        teacher_term = getattr(cfg.observations.teacher, name)
        state_term = getattr(state_cfg.observations.policy, name)
        assert teacher_term.func is state_term.func
        assert teacher_term.params == state_term.params
        assert teacher_term.noise is None


def test_franka_stack_distilled_student_is_strictly_ppo_compatible():
    """Fine-tuning must load the complete camera student and start at iteration zero."""
    from isaaclab_tasks.contrib.stack.config.franka.agents.rsl_rl_distillation_cfg import (
        FrankaStackCameraDistillationRunnerCfg,
        StackVisualDistillationModel,
    )
    from isaaclab_tasks.contrib.stack.config.franka.agents.rsl_rl_ppo_cfg import (
        FrankaStackCameraFineTunePPORunnerCfg,
        StackPPO,
    )

    observations = TensorDict(
        {
            "policy": torch.zeros(2, 39),
            "base_image": torch.zeros(2, 6, 128, 128),
        },
        batch_size=[2],
    )
    distillation_cfg = FrankaStackCameraDistillationRunnerCfg()
    fine_tune_cfg = FrankaStackCameraFineTunePPORunnerCfg()
    student = StackVisualDistillationModel(
        observations,
        distillation_cfg.obs_groups,
        "student",
        8,
        hidden_dims=distillation_cfg.student.hidden_dims,
        activation=distillation_cfg.student.activation,
        obs_normalization=distillation_cfg.student.obs_normalization,
        distribution_cfg=distillation_cfg.student.distribution_cfg.to_dict(),
        cnn_cfg=distillation_cfg.student.cnn_cfg.to_dict(),
        init_temperature=distillation_cfg.student.init_temperature,
        auxiliary_hidden_dims=distillation_cfg.student.auxiliary_hidden_dims,
        visual_target_indices=distillation_cfg.student.visual_target_indices,
        teacher_object_start=distillation_cfg.student.teacher_object_start,
    )
    from isaaclab_tasks.core.lift.config.kuka_allegro.agents.models import SpatialSoftmaxCNNModel

    actor = SpatialSoftmaxCNNModel(
        observations,
        fine_tune_cfg.obs_groups,
        "actor",
        8,
        hidden_dims=fine_tune_cfg.actor.hidden_dims,
        activation=fine_tune_cfg.actor.activation,
        obs_normalization=fine_tune_cfg.actor.obs_normalization,
        distribution_cfg=fine_tune_cfg.actor.distribution_cfg.to_dict(),
        cnn_cfg=fine_tune_cfg.actor.cnn_cfg.to_dict(),
        init_temperature=fine_tune_cfg.actor.init_temperature,
    )
    algorithm = StackPPO.__new__(StackPPO)
    algorithm._raw_actor = actor

    assert algorithm.load({"student_state_dict": student.state_dict()}, load_cfg=None, strict=True) is False
    assert set(actor.state_dict()) == {name for name in student.state_dict() if not name.startswith("auxiliary_head.")}
    assert all(parameter.requires_grad for parameter in student.parameters())


def test_stack_distillation_gripper_output_has_a_behavior_cloning_gradient():
    from isaaclab_tasks.contrib.stack.config.franka.agents.rsl_rl_distillation_cfg import (
        StackDistillationDistribution,
    )

    distribution = StackDistillationDistribution(output_dim=8, init_std=0.45)
    raw_output = torch.zeros(4, 8, requires_grad=True)
    cloning_output = distribution.deterministic_output(raw_output)

    cloning_output[:, -1].sum().backward()

    assert torch.all(raw_output.grad[:, -1] > 0.0)
    assert torch.allclose(cloning_output[:, -1], torch.zeros(4))


def test_stack_distillation_gates_dagger_and_controls_state_occupancy(monkeypatch):
    """Student collection should start only after successful held-out cloning."""
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage

    from isaaclab_tasks.contrib.stack.config.franka.agents.rsl_rl_distillation_cfg import (
        StackDistillation,
        StackDistillationDistributionCfg,
        StackTeacherControllerAdapterModel,
    )
    from isaaclab_tasks.contrib.stack.config.franka.agents.rsl_rl_ppo_cfg import StackGaussianDistributionCfg

    recipe_ids = torch.tensor([*range(9), 0, 1, 8])
    observations = TensorDict(
        {
            "student_obs": torch.zeros(12, 3),
            "base_image": torch.rand(12, 6, 32, 32) - 0.5,
            "teacher_obs": torch.zeros(12, 5),
            "distillation_context": torch.nn.functional.one_hot(recipe_ids, num_classes=9).float(),
        },
        batch_size=[12],
    )
    obs_groups = {"student": ["student_obs", "base_image"], "teacher": ["teacher_obs"]}
    student = StackTeacherControllerAdapterModel(
        observations,
        obs_groups,
        "student",
        8,
        hidden_dims=[16],
        distribution_cfg=StackDistillationDistributionCfg(init_std=0.05, std_range=(0.02, 0.12)).to_dict(),
        cnn_cfg={
            "output_channels": [4, 4],
            "kernel_size": [5, 3],
            "stride": [2, 2],
            "activation": "elu",
        },
        teacher_observation_dim=5,
        controller_hidden_dims=[16],
    )
    teacher = MLPModel(
        observations,
        obs_groups,
        "teacher",
        8,
        hidden_dims=[16],
        distribution_cfg=StackGaussianDistributionCfg(init_std=0.45).to_dict(),
    )
    storage = RolloutStorage("distillation", 12, 2, observations, [8], "cpu")
    algorithm = StackDistillation(
        student,
        teacher,
        storage,
        num_learning_epochs=1,
        gradient_length=1,
        learning_rate=1.0e-3,
        loss_type="huber",
        device="cpu",
        teacher_pretrain_updates=2,
        dagger_gate_recipe_ids=(0,),
        dagger_gate_success_rate=0.9,
        dagger_gate_min_attempts=1,
        dagger_success_gate=True,
        student_control_fraction_start=0.10,
        student_control_fraction_end=0.20,
        student_control_anneal_updates=3,
        student_control_feedback_gain=1.0,
        evaluation_success_ema_alpha=1.0,
        evaluation_envs_per_recipe=1,
        recipe_count=9,
        distillation_context_obs_group="distillation_context",
    )

    weights_recipe_ids = torch.tensor([0, 0, 0, 0, 1, 1, 8])
    weights = algorithm._balanced_recipe_weights(weights_recipe_ids)
    assert weights.mean() == pytest.approx(1.0)
    recipe_weight_totals = torch.stack([weights[weights_recipe_ids == recipe].sum() for recipe in (0, 1, 8)])
    assert recipe_weight_totals[0] == pytest.approx(recipe_weight_totals[1])
    assert recipe_weight_totals[2] == pytest.approx(3.0 * recipe_weight_totals[0])

    teacher_actions = torch.tensor(
        [
            [2.0, -2.0, 0.5, 0.0, 0.2, -0.2, 0.4, 1.0],
            [-2.0, 2.0, -0.5, 0.0, -0.2, 0.2, -0.4, -1.0],
        ]
    )
    correct_output = torch.cat((teacher_actions[:, :-1].clamp(-1.0, 1.0), torch.tensor([[10.0], [-10.0]])), dim=-1)
    wrong_gripper_output = correct_output.clone()
    wrong_gripper_output[:, -1] *= -1.0
    correct_terms = algorithm._behavior_terms(correct_output, teacher_actions)
    wrong_terms = algorithm._behavior_terms(wrong_gripper_output, teacher_actions)
    assert correct_terms[0] == pytest.approx(0.0)
    assert correct_terms[1] < 1.0e-4
    assert correct_terms[2] == pytest.approx(0.0)
    assert correct_terms[3] == pytest.approx(1.0)
    assert correct_terms[4] == pytest.approx(0.5)
    assert wrong_terms[0] == pytest.approx(0.0)
    assert wrong_terms[1] > 9.0
    assert wrong_terms[3] == pytest.approx(0.0)

    for copied, source in zip(student.mlp.controller.parameters(), teacher.mlp.parameters(), strict=True):
        torch.testing.assert_close(copied, source)
        assert not copied.requires_grad

    algorithm.num_updates = 1
    algorithm._evaluation_cumulative_attempts[0] = 1
    algorithm._evaluation_success_ema[0] = 0.0
    algorithm._evaluation_ema_initialized[0] = True
    algorithm._maybe_unlock_dagger()
    assert not algorithm._dagger_unlocked
    assert algorithm._student_control_probability() == pytest.approx(0.0)

    algorithm.num_updates = 2
    algorithm._maybe_unlock_dagger()
    assert not algorithm._dagger_unlocked
    assert algorithm._student_control_probability() == pytest.approx(0.10)

    algorithm._evaluation_success_ema[0] = 1.0
    algorithm._maybe_unlock_dagger()
    assert algorithm._dagger_unlocked
    assert algorithm._student_control_probability() == pytest.approx(0.10)
    assert algorithm._target_student_control_fraction() == pytest.approx(0.10)
    algorithm._update_student_episode_probability(observed_student_fraction=0.50)
    assert algorithm._student_control_probability() == pytest.approx(0.10)
    algorithm._update_student_episode_probability(observed_student_fraction=0.01)
    assert algorithm._student_control_probability() == pytest.approx(0.10)

    algorithm._dagger_unlocked = False
    algorithm._student_episode_probability = 0.0
    algorithm.num_updates = 0
    algorithm._evaluation_cumulative_attempts.zero_()
    algorithm._evaluation_cumulative_successes.zero_()
    deterministic_student_actions = algorithm.student(observations).detach().clamp(-1.0, 1.0)
    deterministic_teacher_actions = algorithm.teacher(observations).detach().clamp(-1.0, 1.0)
    actions = algorithm.act(observations)
    assert torch.isfinite(algorithm.student.output_std).all()
    torch.testing.assert_close(actions[:9], deterministic_student_actions[:9])
    assert not torch.any(algorithm._teacher_control_mask[:9])
    assert torch.all(algorithm._teacher_control_mask[9:])
    torch.testing.assert_close(actions[9:], deterministic_teacher_actions[9:])
    success_pulse = torch.zeros(12)
    success_pulse[0] = 2.0
    algorithm.process_env_step(
        observations,
        rewards=success_pulse,
        dones=torch.zeros(12, dtype=torch.bool),
        extras={},
    )
    algorithm.act(observations)
    algorithm.process_env_step(
        observations,
        rewards=torch.zeros(12),
        dones=torch.ones(12, dtype=torch.bool),
        extras={},
    )
    metrics = algorithm.update()
    assert metrics["recipe_final_release_eval_attempts"] == 1.0
    assert metrics["recipe_final_release_eval_success_rate"] == 1.0
    assert metrics["recipe_final_release_eval_cumulative_success_rate"] == 1.0
    assert metrics["teacher_control_fraction"] == 1.0
    assert metrics["kl"] >= 0.0
    assert algorithm.learning_rate == pytest.approx(1.0e-3)
    assert metrics["dagger_active"] == 0.0
    assert metrics["dagger_progress"] == 0.0
    # The success gate controls the occupancy ramp, not whether DAgger can
    # collect the small amount of off-trajectory data needed to reach success.
    assert metrics["student_control_probability"] == pytest.approx(0.0)
    assert metrics["student_control_fraction_target"] == pytest.approx(0.0)

    saved = algorithm.save()
    assert saved["stack_distillation_state"]["num_updates"] == 1
    assert not saved["stack_distillation_state"]["dagger_unlocked"]
    assert saved["stack_distillation_state"]["evaluation_cumulative_successes"][0] == 1.0
    assert saved["stack_distillation_state"]["learning_rate"] == pytest.approx(1.0e-3)

    algorithm.schedule = "adaptive"
    algorithm.learning_rate = 1.0e-4
    algorithm.adaptive_learning_rate_min = 1.0e-5
    algorithm.adaptive_learning_rate_max = 3.0e-4
    algorithm.adaptive_learning_rate_factor = 1.5
    for param_group in algorithm.optimizer.param_groups:
        param_group["lr"] = algorithm.learning_rate

    algorithm.is_multi_gpu = True
    algorithm.gpu_world_size = 1
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op: None)
    monkeypatch.setattr(torch.distributed, "broadcast", lambda tensor, src: None)
    with torch.inference_mode():
        inference_kl = torch.tensor(0.03)
    algorithm._adapt_learning_rate(inference_kl)
    assert algorithm.learning_rate == pytest.approx(1.0e-4 / 1.5)
    assert algorithm.optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-4 / 1.5)
    algorithm._adapt_learning_rate(torch.tensor(0.001))
    assert algorithm.learning_rate == pytest.approx(1.0e-4)


def test_stack_distillation_auxiliary_loss_updates_the_visual_state_head():
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage

    from isaaclab_tasks.contrib.stack.config.franka.agents.rsl_rl_distillation_cfg import (
        StackDistillation,
        StackDistillationDistributionCfg,
        StackTeacherControllerAdapterModel,
    )
    from isaaclab_tasks.contrib.stack.config.franka.agents.rsl_rl_ppo_cfg import StackGaussianDistributionCfg

    observations = TensorDict(
        {
            "student_obs": torch.zeros(12, 3),
            "base_image": torch.rand(12, 6, 32, 32) - 0.5,
            "teacher_obs": torch.rand(12, 5) * 2.0 - 1.0,
            "distillation_context": torch.nn.functional.one_hot(
                torch.tensor([*range(9), 0, 4, 8]), num_classes=9
            ).float(),
        },
        batch_size=[12],
    )
    obs_groups = {"student": ["student_obs", "base_image"], "teacher": ["teacher_obs"]}
    student = StackTeacherControllerAdapterModel(
        observations,
        obs_groups,
        "student",
        8,
        hidden_dims=[16],
        distribution_cfg=StackDistillationDistributionCfg(init_std=0.05, std_range=(0.02, 0.12)).to_dict(),
        cnn_cfg={
            "output_channels": [4, 4],
            "kernel_size": [5, 3],
            "stride": [2, 2],
            "activation": "elu",
        },
        teacher_observation_dim=5,
        controller_hidden_dims=[16],
    )
    teacher = MLPModel(
        observations,
        obs_groups,
        "teacher",
        8,
        hidden_dims=[16],
        distribution_cfg=StackGaussianDistributionCfg(init_std=0.45).to_dict(),
    )
    storage = RolloutStorage("distillation", 12, 1, observations, [8], "cpu")
    algorithm = StackDistillation(
        student,
        teacher,
        storage,
        num_learning_epochs=1,
        gradient_length=1,
        learning_rate=1.0e-3,
        loss_type="huber",
        device="cpu",
        auxiliary_loss_weight=0.5,
        evaluation_envs_per_recipe=1,
        teacher_pretrain_updates=10,
    )
    initial_parameters = tuple(parameter.detach().clone() for parameter in student.mlp.state_adapter.parameters())

    algorithm.act(observations)
    algorithm.process_env_step(
        observations,
        rewards=torch.zeros(12),
        dones=torch.zeros(12, dtype=torch.bool),
        extras={},
    )
    metrics = algorithm.update()

    assert metrics["behavior_auxiliary"] > 0.0
    assert metrics["auxiliary_mae"] > 0.0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial_parameters, student.mlp.state_adapter.parameters(), strict=True)
    )


def test_franka_stack_camera_finetune_uses_the_deployable_environment():
    task_name = "IsaacContrib-Stack-Cube-Franka-RL-Camera-Finetune"
    spec = gym.spec(task_name)
    cfg = parse_env_cfg(task_name, device="cuda:0", num_envs=8)
    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")

    assert spec.kwargs["env_cfg_entry_point"].endswith(":FrankaCubeStackCameraRLEnvCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":FrankaStackCameraFineTunePPORunnerCfg")
    assert not hasattr(cfg.observations, "teacher")
    assert agent_cfg.class_name == "OnPolicyRunner"
    assert agent_cfg.run_name == "finetune"
    assert agent_cfg.obs_groups["actor"] == ["policy", "base_image"]
    assert "privileged" not in agent_cfg.obs_groups["actor"]


def test_franka_stack_camera_rgb_normalization_is_stationary_and_channel_first():
    """The image value must not depend on other environments in the batch."""
    image = torch.tensor(
        [
            [[[0, 127, 255], [255, 127, 0]]],
            [[[255, 255, 255], [0, 0, 0]]],
        ],
        dtype=torch.uint8,
    )
    sensor = SimpleNamespace(data=SimpleNamespace(output={"rgb": image}))
    env = SimpleNamespace(
        observation_manager=object(),
        scene=SimpleNamespace(sensors={"base_camera": sensor}),
    )

    normalized = mdp.normalized_rgb_image(env)

    assert normalized.shape == (2, 3, 1, 2)
    assert normalized[0, 0, 0, 0] == pytest.approx(-0.5)
    assert normalized[0, 2, 0, 0] == pytest.approx(0.5)
    assert normalized[1, 0, 0, 0] == pytest.approx(0.5)
    assert normalized[1, 0, 0, 1] == pytest.approx(-0.5)


def test_franka_stack_camera_temporal_rgb_repeats_after_reset_and_keeps_motion():
    """A deployable two-frame observation exposes motion without crossing episodes."""
    first = torch.tensor(
        [
            [[[0, 0, 0], [255, 255, 255]]],
            [[[64, 64, 64], [128, 128, 128]]],
        ],
        dtype=torch.uint8,
    )
    second = 255 - first
    sensor = SimpleNamespace(data=SimpleNamespace(output={"rgb": first}))
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        observation_manager=object(),
        scene=SimpleNamespace(sensors={"base_camera": sensor}),
    )
    cfg = SimpleNamespace(params={"sensor_cfg": SimpleNamespace(name="base_camera"), "history_length": 2})
    temporal = mdp.TemporalNormalizedRgbImage(cfg, env)

    initial = temporal(env, cfg.params["sensor_cfg"], history_length=2)
    sensor.data.output["rgb"] = second
    moved = temporal(env, cfg.params["sensor_cfg"], history_length=2)

    assert initial.shape == (2, 6, 1, 2)
    torch.testing.assert_close(initial[:, :3], initial[:, 3:])
    torch.testing.assert_close(moved[:, :3], initial[:, 3:])
    assert not torch.equal(moved[:, :3], moved[:, 3:])

    temporal.reset(torch.tensor([0]))
    reset = temporal(env, cfg.params["sensor_cfg"], history_length=2)
    torch.testing.assert_close(reset[0, :3], reset[0, 3:])


def test_franka_stack_camera_calibration_uses_environment_origins():
    """Zero-noise calibration should place each camera in its own clone."""

    class _Camera:
        def set_world_poses_from_view(self, eyes, targets, env_ids):
            self.eyes = eyes
            self.targets = targets
            self.env_ids = env_ids

    camera = _Camera()
    origins = torch.tensor(((0.0, 0.0, 0.0), (2.5, 0.0, 0.0), (5.0, 0.0, 0.0)))
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        scene=SimpleNamespace(env_origins=origins, sensors={"base_camera": camera}),
    )

    mdp.randomize_camera_calibration(
        env,
        torch.tensor((0, 2)),
        eye=(0.9, -0.55, 0.48),
        lookat=(0.48, 0.0, 0.08),
        eye_position_noise=(0.0, 0.0, 0.0),
        lookat_position_noise=(0.0, 0.0, 0.0),
    )

    torch.testing.assert_close(camera.eyes, origins[[0, 2]] + torch.tensor((0.9, -0.55, 0.48)))
    torch.testing.assert_close(camera.targets, origins[[0, 2]] + torch.tensor((0.48, 0.0, 0.08)))
    assert torch.equal(camera.env_ids, torch.tensor((0, 2)))


def test_legacy_play_task_name_redirects_to_training_task():
    assert "IsaacContrib-Stack-Cube-Franka-RL-Play" not in gym.registry
    with pytest.warns(FutureWarning, match="was removed"):
        resolved = resolve_play_task_name("IsaacContrib-Stack-Cube-Franka-RL-Play")
    assert resolved == "IsaacContrib-Stack-Cube-Franka-RL"


def test_stack_gaussian_distribution_combines_bounded_arm_and_binary_gripper():
    from isaaclab_tasks.contrib.stack.config.franka.agents.rsl_rl_ppo_cfg import (
        StackGaussianDistribution,
        StackGaussianDistributionCfg,
    )

    distribution_cfg = StackGaussianDistributionCfg(init_std=0.45)

    assert distribution_cfg.class_name.endswith(":StackGaussianDistribution")
    assert distribution_cfg.init_std == 0.45
    assert distribution_cfg.std_range == (0.15, 0.65)
    assert not hasattr(distribution_cfg, "gripper_std")

    distribution = StackGaussianDistribution(
        output_dim=8,
        init_std=distribution_cfg.init_std,
        std_range=distribution_cfg.std_range,
    )
    raw_output = torch.tensor(
        [
            [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, 14.0],
            [-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, -9.0],
        ],
        requires_grad=True,
    )
    distribution.update(raw_output)

    assert torch.allclose(distribution.mean[:, :-1], raw_output[:, :-1])
    assert torch.all(distribution.mean[:, -1].abs() < 1.0)
    assert torch.equal(torch.sign(distribution.mean[:, -1]), torch.sign(raw_output[:, -1]))
    expected_arm_std = torch.full((2, 7), 0.45)
    assert torch.allclose(distribution.std[:, :-1], expected_arm_std)
    expected_gripper_std = 2.0 * torch.sqrt(torch.sigmoid(raw_output[:, -1]) * (1.0 - torch.sigmoid(raw_output[:, -1])))
    assert torch.allclose(distribution.std[:, -1], expected_gripper_std)
    deterministic = distribution.deterministic_output(raw_output)
    assert torch.allclose(deterministic[:, :-1], raw_output[:, :-1])
    assert torch.equal(deterministic[:, -1], torch.tensor([1.0, -1.0]))
    assert torch.all(torch.isin(distribution.sample()[:, -1], torch.tensor([-1.0, 1.0])))

    # The logistic parameterization must retain a useful entropy gradient near
    # the lower bound. A forward clamp would return exactly zero here.
    with torch.no_grad():
        distribution.std_param[:-1].fill_(-4.0)
    distribution.update(raw_output)
    distribution.entropy.mean().backward()
    assert torch.all(distribution.std_param.grad[:-1].abs() > 0.0)
    assert torch.all(raw_output.grad[:, -1].abs() > 0.0)

    # Saturated but finite gripper logits must not turn the PPO KL into
    # infinity. The generic Bernoulli KL rounds these probabilities to 0/1.
    old_params = (
        torch.zeros((2, 7)),
        torch.ones((2, 7)),
        torch.tensor([[1000.0], [-1000.0]]),
    )
    new_params = (
        torch.zeros((2, 7)),
        torch.ones((2, 7)),
        torch.tensor([[-1000.0], [1000.0]]),
    )
    extreme_kl = distribution.kl_divergence(old_params, new_params)
    assert torch.all(torch.isfinite(extreme_kl))
    assert torch.allclose(extreme_kl, torch.full((2,), 1000.0))


def test_play_config_starts_scattered_and_keeps_full_stack_horizon():
    cfg = parse_env_cfg("IsaacContrib-Stack-Cube-Franka-RL", device="cuda:0")
    cfg.play_mode()

    assert cfg.scene.num_envs == 50
    assert cfg.episode_length_s == 30.0
    assert cfg.events.reset_from_state_buffer.params["fixed_row_id"] is None
    assert cfg.events.reset_from_state_buffer.params["fixed_recipe"] == int(reset_events.StackResetRecipe.TABLE)
    assert cfg.events.reset_from_state_buffer.params["force_full_goal"]
    assert cfg.curriculum is None
    assert cfg.terminations.progress_context.func.__name__ == "StableOrderInvariantStackGoal"
    assert cfg.terminations.success.func.__name__ == "success_after_minimum_horizon"
