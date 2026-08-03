# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import sys
import textwrap

import gymnasium as gym
import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab.sim.schemas import CollisionBaseCfg, RigidBodyBaseCfg, UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg

from isaaclab_rl.entrypoints.common import resolve_play_task_name

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.stack import spawners
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
    assert cfg.viewer.origin_type == "env"
    assert cfg.viewer.env_index == 0
    cube_colors = [
        cfg.scene.cube_1.spawn.display_color,
        cfg.scene.cube_2.spawn.display_color,
        cfg.scene.cube_3.spawn.display_color,
    ]
    assert len(set(cube_colors)) == 3
    for cube_cfg in (cfg.scene.cube_1, cfg.scene.cube_2, cfg.scene.cube_3):
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
    assert not hasattr(cfg.rewards, "progress")
    assert not hasattr(cfg.rewards, "reaching_cube")
    assert not hasattr(cfg.rewards, "lifting_cube")
    assert not hasattr(cfg.rewards, "cube_goal_tracking")
    assert cfg.rewards.success.func.__name__ == "stack_success_pulse"
    assert cfg.rewards.success.params == {"context_term_name": "progress_context"}
    assert cfg.rewards.success.weight == 100.0
    assert cfg.rewards.failure.func.__name__ == "irrecoverable_stack_failure"
    assert cfg.rewards.failure.params == {"success_termination_name": "success"}
    assert cfg.rewards.failure.weight == -0.01
    assert cfg.events.reset_all.func.__name__ == "reset_scene_to_default"
    assert cfg.events.reset_from_state_buffer.func.__name__ == "StackResetStateTable"
    assert cfg.events.reset_from_state_buffer.params["fixed_row_id"] is None
    assert cfg.events.reset_from_state_buffer.params["fixed_recipe"] is None
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
    assert cfg.actions.arm_action.scale == 0.25
    assert cfg.actions.arm_action.max_delta == 0.15
    assert cfg.actions.arm_action.workspace_lower == (-0.303, -0.200, -0.328, -2.759, -0.124, 2.393, 0.271)
    assert cfg.actions.arm_action.workspace_upper == (0.450, 0.603, 0.147, -2.000, 0.345, 3.112, 1.200)
    assert cfg.scene.robot.spawn.usd_path == FRANKA_PANDA_MENAGERIE_CFG.spawn.usd_path
    assert not cfg.scene.robot.spawn.rigid_props.disable_gravity
    assert set(cfg.scene.robot.actuators) == {"panda_arm", "panda_hand"}
    arm = cfg.scene.robot.actuators["panda_arm"]
    assert arm.joint_names_expr == ["panda_joint[1-7]"]
    assert arm.effort_limit_sim == {"panda_joint[1-4]": 87.0, "panda_joint[5-7]": 12.0}
    assert arm.velocity_limit is None
    assert arm.velocity_limit_sim == {"panda_joint[1-4]": 20.0, "panda_joint[5-7]": 25.0}
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
    assert cfg.curriculum.reset_sampling.params["monitored_history_len"] == 50
    assert cfg.curriculum.reset_sampling.params["target_success_rate"] == 0.5
    assert cfg.curriculum.reset_sampling.params["kappa"] == 1.0
    assert cfg.curriculum.reset_sampling.params["epsilon"] == 4.83e-4
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
    assert cfg.viewer.eye == (1.4, 1.4, 0.9)
    assert cfg.viewer.lookat == (0.5, 0.0, 0.1)
    assert cfg.scene.cube_1.spawn.size == (0.04, 0.04, 0.04)
    assert cfg.scene.cube_2.spawn.size == (0.04, 0.04, 0.04)
    assert cfg.scene.cube_3.spawn.size == (0.04, 0.04, 0.04)
    assert cfg.scene.cube_1.spawn.physics_material.static_friction == 1.0
    assert cfg.scene.cube_1.spawn.physics_material.dynamic_friction == 0.8


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
