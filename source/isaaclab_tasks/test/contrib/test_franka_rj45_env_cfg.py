# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration contracts for reset-driven Franka RJ45 insertion."""

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from isaaclab_newton.physics import NewtonManager, VBDSolverCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.franka_rj45_insertion import rj45_env
from isaaclab_tasks.contrib.franka_rj45_insertion.config.franka.agents.rsl_rl_ppo_cfg import (
    RJ45GaussianBernoulliDistribution,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.mdp.observations import sampled_cable_positions_obs
from isaaclab_tasks.contrib.franka_rj45_insertion.mdp.terminations import task_out_of_bounds
from isaaclab_tasks.contrib.franka_rj45_insertion.reset_dataset_io import reset_dataset_digest
from isaaclab_tasks.contrib.franka_rj45_insertion.rj45_env_cfg import (
    FrankaRJ45InsertionEnvCfg,
    configure_rj45_capacities,
    reset_dataset_task_contract,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.task_success import (
    RJ45_SUCCESS_PREDICATE_VERSION,
    rj45_insertion_success,
)


def test_task_is_registered_with_training_configuration():
    spec = gym.spec("IsaacContrib-Franka-RJ45-Insertion")

    assert spec.entry_point.endswith(":FrankaRJ45InsertionEnv")
    assert "rsl_rl_cfg_entry_point" in spec.kwargs


def test_newton_fidelity_and_fixed_goal_contract_are_explicit():
    cfg = FrankaRJ45InsertionEnvCfg()
    cfg.validate()
    coupled = cfg.sim.physics.solver_cfg
    vbd = coupled.entries[1].solver_cfg

    assert isinstance(vbd, VBDSolverCfg)
    assert vbd.iterations == 12
    assert vbd.rigid_compliant_alm is True
    assert vbd.rigid_body_contact_buffer_size == 256
    assert cfg.sim.physics.num_substeps == 3
    assert cfg.sim.physics.collision_decimation == 1
    assert cfg.sim.dt / cfg.sim.physics.num_substeps == pytest.approx(1.0 / 360.0)
    assert coupled.proxies[0].mode == "staggered"
    contract = reset_dataset_task_contract(cfg)
    assert contract["contract_version"] == 3
    assert contract["task_body_count"] == 37
    assert contract["validation_geometry"]["success_predicate_version"] == RJ45_SUCCESS_PREDICATE_VERSION
    assert contract["robot"]["reset_control_convention"]["gravity_compensation"] is False
    assert contract["actions"]["arm"]["gravity_compensation"] is False
    assert cfg.scene.robot.actuators["panda_arm"].stiffness["panda_joint[1-4]"] == 600.0
    assert cfg.scene.robot.actuators["panda_hand"].stiffness == 350.0
    assert cfg.rewards.failure.params["include_time_out"] is True


def test_contact_capacity_tracks_late_world_count_override():
    cfg = FrankaRJ45InsertionEnvCfg()
    cfg.scene.num_envs = 19

    configure_rj45_capacities(cfg)

    assert cfg.sim.physics.collision_cfg.rigid_contact_max == 19 * 1024


def test_reset_dataset_path_error_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setattr(rj45_env, "_REPO_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError) as exc_info:
        rj45_env._resolve_reset_dataset_path("datasets/franka_rj45_insertion/reset_dataset.pt")

    message = str(exc_info.value)
    assert "generate_franka_rj45_reset_dataset.py" in message
    assert "validate_franka_rj45_resets.py" in message


def test_reset_validation_report_path_error_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setattr(rj45_env, "_REPO_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="full reset-validation report"):
        rj45_env._resolve_reset_validation_report_path(
            "logs/rsl_rl/franka_rj45_insertion/validation/reset_validation.json"
        )


def test_rj45_callback_cleanup_handles_partial_initialization(monkeypatch):
    class _Handle:
        deregistered = False

        def deregister(self):
            self.deregistered = True

    env = object.__new__(rj45_env.FrankaRJ45InsertionEnv)
    env._is_closed = True
    handle = _Handle()
    env._rj45_physics_ready_handle = handle
    env._rj45_state_force_callback = env._prepare_rj45_substep
    env._rj45_post_step_callback = env._align_rj45_after_step
    monkeypatch.setattr(NewtonManager, "_state_force_callbacks", [env._rj45_state_force_callback])
    monkeypatch.setattr(NewtonManager, "_post_step_callbacks", [env._rj45_post_step_callback])

    env._clear_rj45_callbacks()
    env._clear_rj45_callbacks()

    assert handle.deregistered
    assert NewtonManager._state_force_callbacks == []
    assert NewtonManager._post_step_callbacks == []


def test_reset_boundary_preserves_terminal_row_and_outcomes(monkeypatch):
    env = object.__new__(rj45_env.FrankaRJ45InsertionEnv)
    env._is_closed = True
    env.sim = SimpleNamespace(device="cpu")
    env.reset_dataset_row_id = torch.tensor([10, 11, 12, 13])
    env.last_terminal_row_id = torch.full((4,), -1, dtype=torch.long)
    env.last_terminal_outcomes = {name: torch.zeros(4, dtype=torch.bool) for name in rj45_env.TERMINAL_OUTCOME_NAMES}
    term_values = {
        name: torch.tensor([index == offset for index in range(4)])
        for offset, name in enumerate(rj45_env.TERMINAL_OUTCOME_NAMES[:4])
    }
    term_values["time_out"] = torch.tensor([False, True, False, True])
    env.termination_manager = SimpleNamespace(
        active_terms=list(rj45_env.TERMINAL_OUTCOME_NAMES),
        get_term=term_values.__getitem__,
    )
    delegated = []
    monkeypatch.setattr(
        rj45_env.ManagerBasedRLEnv,
        "_reset_idx",
        lambda _env, env_ids: delegated.append(torch.as_tensor(env_ids).clone()),
    )

    env._reset_idx(torch.tensor([1, 3]))

    assert env.last_terminal_row_id.tolist() == [-1, 11, -1, 13]
    for name in rj45_env.TERMINAL_OUTCOME_NAMES:
        assert env.last_terminal_outcomes[name][[1, 3]].tolist() == term_values[name][[1, 3]].tolist()
    assert len(delegated) == 1
    assert delegated[0].tolist() == [1, 3]


def test_invalid_goal_rotation_override_is_rejected():
    cfg = FrankaRJ45InsertionEnvCfg()
    cfg.task_rotation_xyzw = (0.0, 0.0, 1.0, 0.0)

    with pytest.raises(ValueError, match="identity task rotation"):
        cfg.validate()


def test_policy_distribution_samples_the_physical_binary_gripper():
    distribution = RJ45GaussianBernoulliDistribution(output_dim=8)
    output = torch.zeros((256, 8))
    distribution.update(output)

    samples = distribution.sample()
    old_params = tuple(parameter.clone() for parameter in distribution.params)
    distribution.update(output + 0.1)
    kl = distribution.kl_divergence(old_params, distribution.params)

    assert set(samples[:, -1].tolist()) <= {-1.0, 1.0}
    assert torch.isfinite(distribution.log_prob(samples)).all()
    assert torch.isfinite(distribution.entropy).all()
    assert torch.isfinite(kl).all()


def test_reset_contract_resolves_coupled_entries_by_name():
    canonical = FrankaRJ45InsertionEnvCfg()
    reordered = FrankaRJ45InsertionEnvCfg()
    reordered.sim.physics.solver_cfg.entries.reverse()

    reordered.validate()

    assert reset_dataset_task_contract(reordered) == reset_dataset_task_contract(canonical)


def test_reset_contract_changes_with_every_reset_relevant_subsystem():
    baseline = reset_dataset_digest(reset_dataset_task_contract(FrankaRJ45InsertionEnvCfg()))
    mutated = []

    cfg = FrankaRJ45InsertionEnvCfg()
    cfg.actions.gripper_action.close_position += 0.001
    mutated.append(cfg)
    cfg = FrankaRJ45InsertionEnvCfg()
    cfg.actions.gripper_action.contact_min_deflection += 0.0001
    mutated.append(cfg)
    cfg = FrankaRJ45InsertionEnvCfg()
    cfg.sim.physics.solver_cfg.proxies[0].mode = "lagged"
    mutated.append(cfg)
    cfg = FrankaRJ45InsertionEnvCfg()
    cfg.sim.physics.solver_cfg.proxies[0].mass_scale = 2.0
    mutated.append(cfg)
    cfg = FrankaRJ45InsertionEnvCfg()
    cfg.sim.physics.solver_cfg.entries[0].solver_cfg.ls_iterations += 1
    mutated.append(cfg)
    cfg = FrankaRJ45InsertionEnvCfg()
    cfg.sim.physics.default_shape_cfg.gap += 0.001
    mutated.append(cfg)
    cfg = FrankaRJ45InsertionEnvCfg()
    cfg.scene.robot.actuators["panda_hand"].stiffness += 1.0
    mutated.append(cfg)

    assert all(reset_dataset_digest(reset_dataset_task_contract(cfg)) != baseline for cfg in mutated)


def test_reset_contract_preserves_callable_identity():
    def spawn_variant_a(*_args, **_kwargs):
        return None

    def spawn_variant_b(*_args, **_kwargs):
        return None

    cfg_a = FrankaRJ45InsertionEnvCfg()
    cfg_a.scene.table.spawn.func = spawn_variant_a
    cfg_b = FrankaRJ45InsertionEnvCfg()
    cfg_b.scene.table.spawn.func = spawn_variant_b

    assert reset_dataset_digest(reset_dataset_task_contract(cfg_a)) != reset_dataset_digest(
        reset_dataset_task_contract(cfg_b)
    )


def test_reset_contract_ignores_scene_resolved_spawn_paths():
    cfg = FrankaRJ45InsertionEnvCfg()
    baseline = reset_dataset_task_contract(cfg)

    cfg.scene.robot.spawn.spawn_path = "/World/envs/env_0/Robot"
    cfg.scene.table.spawn.spawn_path = "/World/envs/env_0/Table"
    cfg.scene.ground.spawn.spawn_path = "/World/GroundPlane"

    assert reset_dataset_task_contract(cfg) == baseline


def test_real_reset_contract_round_trips_through_weights_only_load(tmp_path):
    contract = reset_dataset_task_contract(FrankaRJ45InsertionEnvCfg())
    artifact = tmp_path / "task_contract.pt"

    torch.save(contract, artifact)
    loaded = torch.load(artifact, map_location="cpu", weights_only=True)

    assert reset_dataset_digest(loaded) == reset_dataset_digest(contract)


def test_success_requires_plug_orientation_and_rejects_axial_overtravel():
    count = 7
    goal = torch.zeros((37, 7))
    goal[:, 6] = 1.0
    pose = goal.unsqueeze(0).repeat(count, 1, 1)
    velocity = torch.zeros((count, 37, 6))
    pose[1, 0, 1] = 5.0e-4
    pose[2, 0, 0] = 1.0e-3
    pose[3, 0, 5:7] = torch.tensor([torch.sin(torch.tensor(0.05)), torch.cos(torch.tensor(0.05))])
    pose[4, 1, 5:7] = torch.tensor([torch.sin(torch.tensor(0.05)), torch.cos(torch.tensor(0.05))])
    velocity[5, 0, 3] = 0.02
    pose[6, 0, 1] = -5.0e-4
    cfg = SimpleNamespace(
        success_axial_tolerance=8.0e-4,
        success_axial_overtravel_tolerance=2.0e-4,
        success_radial_tolerance=7.5e-4,
        success_plug_angle_tolerance=0.05,
        success_latch_angle_tolerance=0.05,
        success_max_plug_speed=0.01,
    )
    fake = SimpleNamespace(
        cfg=cfg,
        goal_task_body_pose=goal,
        task_body_pose_e=lambda: pose,
        task_body_velocity=lambda: velocity,
    )

    delegated = rj45_env.FrankaRJ45InsertionEnv.insertion_success_mask(fake)
    result = rj45_insertion_success(
        pose,
        velocity,
        goal,
        axial_tolerance=cfg.success_axial_tolerance,
        axial_overtravel_tolerance=cfg.success_axial_overtravel_tolerance,
        radial_tolerance=cfg.success_radial_tolerance,
        plug_angle_tolerance=cfg.success_plug_angle_tolerance,
        latch_angle_tolerance=cfg.success_latch_angle_tolerance,
        maximum_plug_spatial_speed=cfg.success_max_plug_speed,
    )

    assert delegated.tolist() == [True, False, False, False, False, False, True]
    assert torch.equal(delegated, result.mask)
    assert result.signed_axial_error.tolist() == pytest.approx([0.0, 5.0e-4, 0.0, 0.0, 0.0, 0.0, -5.0e-4])


def test_plug_orientation_error_uses_fixed_goal_quaternion():
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
    quarter_turn_z = torch.tensor([0.0, 0.0, 2.0**-0.5, 2.0**-0.5])
    plug_pose = torch.zeros((2, 7))
    plug_pose[:, 3:7] = torch.stack((identity, quarter_turn_z))
    goal = torch.zeros((37, 7))
    goal[0, 3:7] = identity
    fake = SimpleNamespace(num_envs=2, goal_task_body_pose=goal, plug_pose_e=lambda: plug_pose)

    error = rj45_env.FrankaRJ45InsertionEnv.plug_orientation_error(fake)

    assert error.tolist() == pytest.approx([0.0, torch.pi / 2], abs=1.0e-5)


def test_task_bounds_cover_every_cable_body_and_spatial_velocity():
    pose = torch.zeros((3, 37, 7))
    velocity = torch.zeros((3, 37, 6))
    pose[1, 18, 0] = 2.0
    velocity[2, 22, 0] = 21.0
    fake = SimpleNamespace(
        cfg=SimpleNamespace(
            max_plug_spatial_speed=20.0,
            max_task_body_angular_speed=50.0,
            max_task_body_linear_speed=20.0,
        ),
        _task_workspace_lower=torch.tensor([-1.0, -1.0, -1.0]),
        _task_workspace_upper=torch.tensor([1.0, 1.0, 1.0]),
        _task_body_workspace_lower=torch.tensor([-1.0, -1.0, -1.0]),
        _task_body_workspace_upper=torch.tensor([1.0, 1.0, 1.0]),
        task_body_pose_e=lambda: pose,
        task_body_velocity=lambda: velocity,
    )

    assert task_out_of_bounds(fake).tolist() == [False, True, True]


def test_cable_goal_offset_observation_is_bounded():
    indices = torch.tensor((2, 6, 11, 16, 21, 28, 36))
    current = torch.zeros((1, 37, 7))
    current[:, indices, :3] = 100.0
    fake = SimpleNamespace(
        num_envs=1,
        cfg=SimpleNamespace(max_cable_goal_offset=0.25),
        _cable_observation_body_indices=indices,
        goal_task_body_pose=torch.zeros((37, 7)),
        task_body_pose_e=lambda: current,
    )

    observation = sampled_cable_positions_obs(fake)

    assert observation.shape == (1, 21)
    assert torch.all(observation == 0.25)
