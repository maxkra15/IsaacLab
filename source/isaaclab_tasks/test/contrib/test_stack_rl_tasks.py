# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public-contract tests for the reset-oriented Franka and KUKA stack tasks."""

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.managers import CurriculumTermCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.stack import mdp
from isaaclab_tasks.contrib.stack.config.franka.agents.rsl_rl_distillation_cfg import (
    StackDistillationAlgorithmCfg,
    StackVisualDistillationModelCfg,
)
from isaaclab_tasks.contrib.stack.config.franka.agents.rsl_rl_ppo_cfg import (
    StackGaussianDistribution,
)
from isaaclab_tasks.contrib.stack.config.kuka_allegro.agents.rsl_rl_ppo_cfg import (
    KukaAllegroGaussianDistribution,
)
from isaaclab_tasks.contrib.stack.mdp.kuka_allegro_reset import (
    KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES,
    KUKA_ALLEGRO_LARGE_CUBE_EDGE_LENGTH,
)
from isaaclab_tasks.contrib.stack.mdp.runtime_state import (
    create_stack_reset_runtime_state,
    get_stack_reset_runtime_state,
)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
from isaaclab_tasks.utils.reset_sampling import ResetStateCatalog

FRANKA_STATE_TASK = "IsaacContrib-Stack-Cube-Franka-RL"
FRANKA_CAMERA_TASK = "IsaacContrib-Stack-Cube-Franka-RL-Camera"
FRANKA_DISTILLATION_TASK = "IsaacContrib-Stack-Cube-Franka-RL-Camera-Distillation"
KUKA_STATE_TASK = "IsaacContrib-Stack-Cube-KukaAllegro-RL"


@pytest.mark.parametrize(
    ("task_name", "env_cfg_name", "runner_cfg_name"),
    (
        (FRANKA_STATE_TASK, "FrankaCubeStackRLEnvCfg", "FrankaStackPPORunnerCfg"),
        (FRANKA_CAMERA_TASK, "FrankaCubeStackCameraRLEnvCfg", "FrankaStackCameraPPORunnerCfg"),
        (
            FRANKA_DISTILLATION_TASK,
            "FrankaCubeStackCameraDistillationEnvCfg",
            "FrankaStackCameraDistillationRunnerCfg",
        ),
        (KUKA_STATE_TASK, "KukaAllegroCubeStackRLEnvCfg", "KukaAllegroStackPPORunnerCfg"),
    ),
)
def test_supported_tasks_are_registered(task_name: str, env_cfg_name: str, runner_cfg_name: str):
    """Each supported task resolves one environment and one RSL-RL configuration."""
    spec = gym.spec(task_name)

    assert spec.kwargs["env_cfg_entry_point"].endswith(f":{env_cfg_name}")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(f":{runner_cfg_name}")
    assert load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point") is not None


def test_obsolete_camera_fine_tune_task_is_not_registered():
    with pytest.raises(gym.error.Error):
        gym.spec("IsaacContrib-Stack-Cube-Franka-RL-Camera-Finetune")


@pytest.fixture(scope="module")
def stack_cfgs():
    cfgs = {
        task_name: parse_env_cfg(task_name, device="cuda:0", num_envs=8)
        for task_name in (FRANKA_STATE_TASK, FRANKA_CAMERA_TASK, FRANKA_DISTILLATION_TASK, KUKA_STATE_TASK)
    }
    for cfg in cfgs.values():
        cfg.validate_config()
    return cfgs


def test_franka_state_task_exposes_the_training_contract(stack_cfgs):
    cfg = stack_cfgs[FRANKA_STATE_TASK]

    assert cfg.scene.num_envs == 8
    assert cfg.scene.replicate_physics
    assert isinstance(cfg.sim.physics, NewtonCfg)
    assert isinstance(cfg.sim.physics.solver_cfg, MJWarpSolverCfg)
    assert cfg.sim.physics.num_substeps == 2
    assert cfg.sim.physics.collision_decimation == 1
    assert cfg.decimation == cfg.sim.render_interval == 2
    assert cfg.actions.arm_action.gravity_compensation
    assert cfg.actions.arm_action.scale == cfg.actions.arm_action.max_delta == 0.05
    assert isinstance(cfg.actions.gripper_action, mdp.ResetBufferedGripperActionCfg)
    assert cfg.events.reset_from_state_buffer.func is mdp.StackResetStateTable
    assert cfg.curriculum.reset_sampling.func is mdp.StackResetTableCurriculum
    assert isinstance(
        cfg.curriculum.reset_sampling.params["outcome_monitor"],
        mdp.RollingOutcomeMonitorCfg,
    )
    sampler_cfg = cfg.curriculum.reset_sampling.params["adaptive_sampler"]
    assert isinstance(sampler_cfg, mdp.AdaptiveResetSamplerCfg)
    assert cfg.curriculum.reset_sampling.params["table_sampling_probability"] == 0.35
    assert sampler_cfg.coverage_fraction == pytest.approx(3.0 / 13.0)
    assert cfg.terminations.progress_context.func is mdp.StableOrderInvariantStackGoal
    assert cfg.rewards.success.func is mdp.stack_success_pulse
    assert cfg.observations.policy.object.func is mdp.role_conditioned_stack_obs
    assert cfg.observations.policy.cube_positions is None
    assert cfg.observations.policy.cube_orientations is None
    assert cfg.scene.cube_1.spawn.size == (0.04, 0.04, 0.04)
    assert cfg.scene.cube_1.spawn.physics_material.contact_stiffness == 1.0e4
    assert cfg.scene.table_contact_surface.spawn.physics_material.contact_stiffness == 1.0e4


def test_camera_actor_has_only_deployable_observations(stack_cfgs):
    cfg = stack_cfgs[FRANKA_CAMERA_TASK]
    runner = load_cfg_from_registry(FRANKA_CAMERA_TASK, "rsl_rl_cfg_entry_point")

    assert cfg.scene.base_camera.height == cfg.scene.base_camera.width == 128
    assert cfg.scene.base_camera.data_types == ["rgb"]
    assert cfg.num_rerenders_on_reset == 1
    assert cfg.events.reset_from_state_buffer.params["fixed_role_permutation"] == 0
    assert not hasattr(cfg.observations.policy, "object")
    assert hasattr(cfg.observations, "base_image")
    assert not hasattr(cfg.observations, "teacher")
    assert runner.obs_groups["actor"] == ["policy", "base_image"]
    assert runner.obs_groups["critic"] == ["policy", "privileged"]
    assert "privileged" not in runner.obs_groups["actor"]


def test_distillation_task_adds_privileged_labels_without_changing_the_student(stack_cfgs):
    cfg = stack_cfgs[FRANKA_DISTILLATION_TASK]
    runner = load_cfg_from_registry(FRANKA_DISTILLATION_TASK, "rsl_rl_cfg_entry_point")

    assert hasattr(cfg.observations, "teacher")
    assert hasattr(cfg.observations, "distillation_context")
    assert cfg.observations.distillation_context.recipe.func is mdp.stack_reset_recipe_one_hot
    assert cfg.events.reset_from_state_buffer.params["evaluation_recipe_ids"] == tuple(range(9))
    assert cfg.events.reset_from_state_buffer.params["evaluation_envs_per_recipe"] == 4
    assert cfg.curriculum.reset_sampling.params["evaluation_env_count"] == 36
    assert runner.obs_groups == {"student": ["policy", "base_image"], "teacher": ["teacher"]}
    assert isinstance(runner.student, StackVisualDistillationModelCfg)
    assert isinstance(runner.algorithm, StackDistillationAlgorithmCfg)
    assert not hasattr(runner.algorithm, "stepwise_student_control")
    assert not hasattr(runner.algorithm, "controller_warmup_updates")


def test_kuka_task_has_one_complete_23_dof_state_policy(stack_cfgs):
    cfg = stack_cfgs[KUKA_STATE_TASK]
    runner = load_cfg_from_registry(KUKA_STATE_TASK, "rsl_rl_cfg_entry_point")

    assert cfg.events.reset_from_state_buffer.func is mdp.KukaAllegroResetStateTable
    assert mdp.KukaAllegroResetStateTable.__module__.endswith(".kuka_reset_events")
    assert cfg.actions.arm_action.gravity_compensation
    assert cfg.actions.arm_action.scale == cfg.actions.arm_action.max_delta == 0.12
    assert isinstance(cfg.actions.gripper_action, mdp.ResetPreservingRelativeJointPositionActionCfg)
    assert tuple(cfg.actions.gripper_action.joint_names) == KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES
    assert len(cfg.actions.gripper_action.joint_names) == 16
    assert cfg.actions.gripper_action.scale == cfg.actions.gripper_action.max_delta == 0.10
    assert cfg.terminations.progress_context.func is mdp.StableFullHandOrderInvariantStackGoal
    assert cfg.curriculum.reset_sampling.params["global_sampling"] is True
    assert cfg.scene.cube_1.spawn.size == (KUKA_ALLEGRO_LARGE_CUBE_EDGE_LENGTH,) * 3
    assert cfg.observations.policy.cube_x_axes.func is mdp.role_conditioned_cube_x_axes
    assert len(cfg.observations.policy.hand_joint_pos.params["asset_cfg"].joint_names) == 16
    assert len(cfg.observations.policy.hand_joint_vel.params["asset_cfg"].joint_names) == 16
    assert len(cfg.observations.policy.hand_tip_positions.params["body_cfg"].body_names) == 4
    assert not hasattr(cfg.observations.policy, "grasp_pair")
    assert runner.actor.distribution_cfg.arm_action_dim == 7


@pytest.mark.parametrize(
    "task_name",
    (FRANKA_STATE_TASK, FRANKA_CAMERA_TASK, FRANKA_DISTILLATION_TASK, KUKA_STATE_TASK),
)
def test_play_mode_uses_randomized_table_starts(task_name: str):
    cfg = parse_env_cfg(task_name, device="cuda:0", num_envs=4)

    cfg.play_mode()

    assert cfg.events.reset_from_state_buffer.params["fixed_recipe"] == int(mdp.StackResetRecipe.TABLE)
    assert cfg.curriculum is None
    assert cfg.scene.num_envs == 4


def _make_stack_curriculum(num_envs: int = 1000):
    """Create a simulation-free curriculum with the production Franka row geometry."""
    recipe_names = tuple(recipe.name.lower() for recipe in mdp.StackResetRecipe)
    rows_per_layout = (18, 33, 65, 33, 33, 33, 65, 33, 64)
    recipe_ids = []
    layout_ids = []
    for layout_id in range(18):
        for recipe_id, row_count in enumerate(rows_per_layout):
            recipe_ids.extend((recipe_id,) * row_count)
            layout_ids.extend((layout_id,) * row_count)
    recipe_ids = torch.tensor(recipe_ids, dtype=torch.long)
    layout_ids = torch.tensor(layout_ids, dtype=torch.long)
    catalog = ResetStateCatalog(
        row_count=recipe_ids.numel(),
        metadata={"recipe_ids": recipe_ids, "layout_ids": layout_ids},
    )
    reset_term = SimpleNamespace(
        catalog=catalog,
        row_count=catalog.row_count,
        recipe_ids=recipe_ids,
        layout_ids=layout_ids,
        layout_count=18,
        recipe_names=recipe_names,
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        event_manager=SimpleNamespace(
            get_term_cfg=lambda _name: SimpleNamespace(func=reset_term),
        ),
    )
    create_stack_reset_runtime_state(env)
    params = {
        "outcome_monitor": mdp.RollingOutcomeMonitorCfg(history_length=50, prior_strength=2.0),
        "adaptive_sampler": mdp.AdaptiveResetSamplerCfg(
            target_success_rate=0.5,
            kappa=1.0,
            temperature=1.0,
            coverage_fraction=3.0 / 13.0,
            epsilon=1.0e-4,
        ),
        "success_context_name": "learning_progress_context",
        "final_success_context_name": "progress_context",
        "table_sampling_probability": 0.35,
        "global_sampling": False,
        "evaluation_env_count": 0,
    }
    cfg = CurriculumTermCfg(func=mdp.StackResetTableCurriculum, params=params)
    return env, params, mdp.StackResetTableCurriculum(cfg, env)


def test_stack_curriculum_uses_explicit_table_coverage_frontier_mixture():
    env, params, curriculum = _make_stack_curriculum()

    metrics = curriculum(env, torch.arange(env.num_envs), **params)
    state = get_stack_reset_runtime_state(env)
    reset_term = env.event_manager.get_term_cfg("reset_from_state_buffer").func
    recipe_ids = reset_term.catalog.metadata["recipe_ids"]
    selected_recipes = recipe_ids[state.row_ids]

    assert torch.count_nonzero(selected_recipes == int(mdp.StackResetRecipe.TABLE)) == 350
    assert metrics["sampler_table_assignments"] == 350
    assert metrics["sampler_coverage_assignments"] == 150
    assert metrics["sampler_adaptive_assignments"] == 500
    assert metrics["table_probability"] == pytest.approx(0.35)

    # At the smoothed unseen prior, every semantic intermediate recipe has
    # equal adaptive-frontier mass despite having 18, 33, or 65 rows/layout.
    frontier_masses = torch.stack(
        [
            metrics[f"recipe_{recipe.name.lower()}_frontier_probability"]
            for recipe in mdp.StackResetRecipe
            if recipe != mdp.StackResetRecipe.TABLE
        ]
    )
    torch.testing.assert_close(frontier_masses, torch.full((8,), 1.0 / 8.0))


def test_stack_curriculum_state_roundtrip_preserves_next_draw():
    first_env, params, first = _make_stack_curriculum(num_envs=128)
    first(first_env, torch.arange(first_env.num_envs), **params)
    saved = first.get_state()

    second_env, second_params, second = _make_stack_curriculum(num_envs=128)
    second.set_state(saved)

    restored = second.get_state()
    assert set(restored) == set(saved)
    for name in saved:
        torch.testing.assert_close(restored[name], saved[name])
    first(first_env, torch.arange(first_env.num_envs), **params)
    second(second_env, torch.arange(second_env.num_envs), **second_params)
    torch.testing.assert_close(
        get_stack_reset_runtime_state(first_env).row_ids,
        get_stack_reset_runtime_state(second_env).row_ids,
    )


def test_mixed_franka_distribution_matches_the_physical_action_space():
    distribution = StackGaussianDistribution(output_dim=8, init_std=0.45)
    output = torch.zeros((256, 8))
    distribution.update(output)

    samples = distribution.sample()

    assert samples.shape == (256, 8)
    assert set(samples[:, -1].unique().tolist()) <= {-1.0, 1.0}
    assert torch.equal(distribution.deterministic_output(output)[:, -1], torch.ones(256))
    assert distribution.log_prob(samples).shape == (256,)
    assert torch.isfinite(distribution.entropy).all()


def test_kuka_distribution_covers_all_arm_and_hand_actions():
    distribution = KukaAllegroGaussianDistribution(output_dim=23)
    output = torch.zeros((32, 23))
    distribution.update(output)

    assert distribution.sample().shape == (32, 23)
    assert torch.allclose(distribution.std[0, :7], torch.full((7,), 0.35), atol=1.0e-6)
    assert torch.allclose(distribution.std[0, 7:], torch.full((16,), 0.15), atol=1.0e-6)


def test_reset_runtime_state_has_one_typed_owner():
    env = SimpleNamespace(num_envs=3, device="cpu")

    state = create_stack_reset_runtime_state(env)

    assert get_stack_reset_runtime_state(env) is state
    assert set(vars(env)) == {"num_envs", "device", "stack_reset_state"}
    assert state.row_ids.shape == (3,)
    assert state.role_to_cube.shape == (3, 3)
    with pytest.raises(AttributeError):
        get_stack_reset_runtime_state(SimpleNamespace())


def _cube(positions: torch.Tensor):
    return SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=positions),
            root_vel_w=SimpleNamespace(torch=torch.zeros((positions.shape[0], 6))),
        )
    )


def test_stack_progress_is_independent_of_cube_identity_and_order():
    positions = (
        torch.tensor(((0.45, 0.00, 0.02), (0.45, 0.00, 0.10))),
        torch.tensor(((0.45, 0.00, 0.06), (0.45, 0.00, 0.02))),
        torch.tensor(((0.60, 0.10, 0.02), (0.45, 0.00, 0.06))),
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene={f"cube_{index + 1}": _cube(value) for index, value in enumerate(positions)},
    )

    progress = mdp.order_invariant_stack_progress(env)

    torch.testing.assert_close(progress, torch.tensor((1.0, 2.0)))
