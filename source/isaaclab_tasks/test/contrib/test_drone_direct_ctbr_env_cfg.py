# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused contracts for the rigid-drone Direct-CTBR task variation."""

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg

import isaaclab_tasks.contrib.drone_slung_load.mdp as mdp
from isaaclab_tasks.contrib.drone_slung_load.drone_direct_ctbr_env_cfg import (
    DroneWaypointDirectCTBREnvCfg,
)
from isaaclab_tasks.contrib.drone_slung_load.drone_slung_load_env_cfg import (
    DroneSlungLoadWaypointDirectCTBREnvCfg,
)
from isaaclab_tasks.contrib.drone_slung_load.system import nominal_drone_hover_action, nominal_hover_action

pytestmark = pytest.mark.unit


def test_rigid_drone_task_registration_and_runner_contract():
    import isaaclab_tasks.contrib.drone_slung_load.config.newton_drone  # noqa: F401
    from isaaclab_tasks.contrib.drone_slung_load import (
        DRONE_DIRECT_CTBR_HARD_ROUTES_EXPERIMENT_NAME,
        DRONE_SLUNG_LOAD_WANDB_PROJECT,
    )
    from isaaclab_tasks.contrib.drone_slung_load.config.newton_drone.agents.rsl_rl_ppo_cfg import (
        DroneSlungLoadWaypointDirectCTBRPPORunnerCfg,
        DroneWaypointDirectCTBRPPORunnerCfg,
    )

    spec = gym.spec("IsaacContrib-Drone-Waypoint-FLARE-DirectCTBR")
    assert spec.kwargs["env_cfg_entry_point"].endswith(":DroneWaypointDirectCTBREnvCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":DroneWaypointDirectCTBRPPORunnerCfg")
    assert not spec.id.endswith("-v0")

    runner = DroneWaypointDirectCTBRPPORunnerCfg()
    slung_runner = DroneSlungLoadWaypointDirectCTBRPPORunnerCfg()
    assert runner.experiment_name == DRONE_DIRECT_CTBR_HARD_ROUTES_EXPERIMENT_NAME
    assert runner.experiment_name == "drone_waypoint_flare_direct_ctbr_hard_routes_v4"
    assert runner.wandb_project == DRONE_SLUNG_LOAD_WANDB_PROJECT
    assert runner.obs_groups == {"actor": ["policy"], "critic": ["policy"]}
    assert runner.actor.distribution_cfg.initial_mean == pytest.approx((nominal_drone_hover_action(), 0.0, 0.0, 0.0))
    assert nominal_drone_hover_action() == pytest.approx(2.0 / 3.5 - 1.0)
    assert nominal_drone_hover_action() < nominal_hover_action()
    assert runner.num_steps_per_env == slung_runner.num_steps_per_env == 500
    assert runner.max_iterations == slung_runner.max_iterations == 680
    assert runner.save_interval == slung_runner.save_interval == 5
    assert runner.algorithm.num_learning_epochs == slung_runner.algorithm.num_learning_epochs == 2
    assert runner.algorithm.num_mini_batches == slung_runner.algorithm.num_mini_batches == 20
    assert runner.algorithm.gamma == slung_runner.algorithm.gamma == pytest.approx(0.999)
    assert runner.algorithm.lam == slung_runner.algorithm.lam == pytest.approx(0.999)
    assert runner.algorithm.learning_rate_decay_start_update == 639
    assert runner.algorithm.learning_rate_decay_updates == 40
    assert runner.algorithm.entropy_decay_start_update == 639
    assert runner.algorithm.entropy_decay_updates == 40
    assert runner.to_dict()["obs_groups"] == {"actor": ["policy"], "critic": ["policy"]}


def test_rigid_drone_removes_suspended_assets_and_vbd_only():
    drone = DroneWaypointDirectCTBREnvCfg()
    slung = DroneSlungLoadWaypointDirectCTBREnvCfg()

    assert drone.scene.robot.to_dict() == slung.scene.robot.to_dict()
    assert drone.scene.ground.to_dict() == slung.scene.ground.to_dict()
    assert drone.scene.payload is None
    assert drone.scene.cable is None
    assert drone.scene.drone_cable_attach is None
    assert drone.scene.cable_payload_attach is None
    assert isinstance(drone.sim.physics.solver_cfg, MJWarpSolverCfg)
    assert drone.sim.physics.num_substeps == 1
    assert drone.decimation == 1
    assert drone.sim.dt == pytest.approx(0.01)
    assert drone.actions.thrust.dt == pytest.approx(0.01)
    assert drone.actions.thrust.to_dict() == slung.actions.thrust.to_dict()


def test_rigid_drone_preserves_direct_route_geometry_with_a_rigid_speed_envelope():
    drone = DroneWaypointDirectCTBREnvCfg()
    slung = DroneSlungLoadWaypointDirectCTBREnvCfg()
    drone_route = drone.commands.route.to_dict()
    slung_route = slung.commands.route.to_dict()

    assert drone_route.pop("record_slung_load_metrics") is False
    assert slung_route.pop("record_slung_load_metrics") is True
    assert drone_route.pop("maximum_lateral_acceleration") == pytest.approx(6.0)
    assert slung_route.pop("maximum_lateral_acceleration") == pytest.approx(3.0)
    assert drone_route.pop("maximum_braking_acceleration") == pytest.approx(6.0)
    assert slung_route.pop("maximum_braking_acceleration") == pytest.approx(4.0)
    assert drone_route == slung_route

    drone.evaluation_mode()
    slung.evaluation_mode()
    drone_route = drone.commands.route.to_dict()
    slung_route = slung.commands.route.to_dict()
    drone_route.pop("record_slung_load_metrics")
    slung_route.pop("record_slung_load_metrics")
    drone_route.pop("maximum_lateral_acceleration")
    slung_route.pop("maximum_lateral_acceleration")
    drone_route.pop("maximum_braking_acceleration")
    slung_route.pop("maximum_braking_acceleration")
    assert drone_route == slung_route
    assert drone.commands.route.route_family == slung.commands.route.route_family == "bounded_hard_mix"
    assert drone.episode_length_s == pytest.approx(15.0)
    assert slung.episode_length_s == pytest.approx(20.0)


def test_rigid_and_slung_direct_tasks_share_body_conditioned_policy_terms():
    drone = DroneWaypointDirectCTBREnvCfg()
    slung = DroneSlungLoadWaypointDirectCTBREnvCfg()

    assert drone.observations.policy.drone_velocity.to_dict() == slung.observations.policy.drone_velocity.to_dict()
    assert drone.observations.policy.drone_velocity.func is mdp.body_lin_vel_normalized
    assert drone.observations.policy.drone_velocity.params["speed_scale"] == pytest.approx(4.5)
    assert drone.observations.policy.path_tracking.to_dict() == slung.observations.policy.path_tracking.to_dict()
    assert drone.observations.policy.path_tracking.params["cross_track_scale"] == pytest.approx(1.0)

    group_settings = {
        "concatenate_terms",
        "concatenate_dim",
        "enable_corruption",
        "history_length",
        "flatten_history_dim",
    }
    drone_term_order = tuple(
        name
        for name, value in vars(drone.observations.policy).items()
        if name not in group_settings and value is not None
    )
    slung_term_order = tuple(
        name
        for name, value in vars(slung.observations.policy).items()
        if name not in group_settings and value is not None
    )
    assert drone_term_order == (
        "drone_velocity",
        "body_rotation",
        "body_angular_velocity",
        "waypoint_offsets",
        "path_tracking",
        "path_speed",
        "previous_action",
    )
    assert slung_term_order == (
        "drone_velocity",
        "body_rotation",
        "body_angular_velocity",
        "swing_angles",
        "swing_angular_velocity",
        "waypoint_offsets",
        "path_tracking",
        "path_speed",
        "previous_action",
    )
    assert sum((3, 9, 3, 6, 15, 3, 4)) == 43
    assert sum((3, 9, 3, 2, 2, 6, 15, 3, 4)) == 47


def test_rigid_drone_mdp_has_no_load_only_terms_or_control_priors():
    cfg = DroneWaypointDirectCTBREnvCfg()

    active = lambda group: {name for name, value in group.__dict__.items() if value is not None}  # noqa: E731
    assert active(cfg.rewards) == {
        "episode_metrics",
        "crash",
        "path_progress",
        "path_velocity",
        "completion",
        "path_precision",
        "body_rate",
        "body_tilt",
        "action_acceleration",
    }
    assert active(cfg.terminations) == {
        "time_out",
        "drone_crash",
        "illegal_drone",
        "illegal_action",
        "drone_out_of_workspace",
        "path_corridor",
        "route_completed",
    }
    assert active(cfg.events) == {"reset_base"}

    assert cfg.observations.policy.swing_angles is None
    assert cfg.observations.policy.swing_angular_velocity is None
    assert cfg.observations.privileged is None
    assert cfg.events.reset_slung_load is None
    assert cfg.rewards.swing_safety is None
    assert cfg.rewards.swing_magnitude is None
    assert cfg.rewards.transverse_speed is None
    slung = DroneSlungLoadWaypointDirectCTBREnvCfg()
    assert cfg.rewards.waypoint_advance is slung.rewards.waypoint_advance is None
    drone_progress = cfg.rewards.path_progress.to_dict()
    slung_progress = slung.rewards.path_progress.to_dict()
    assert drone_progress["params"].pop("maximum_lateral_acceleration") == pytest.approx(6.0)
    assert slung_progress["params"].pop("maximum_lateral_acceleration") == pytest.approx(3.0)
    assert drone_progress == slung_progress
    assert cfg.rewards.path_velocity.to_dict() == slung.rewards.path_velocity.to_dict()
    assert cfg.terminations.payload_crash is None
    assert cfg.terminations.illegal_payload is None
    assert cfg.terminations.illegal_cable is None
    assert cfg.terminations.cable_integrity is None
    assert cfg.terminations.payload_out_of_workspace is None
    assert cfg.rewards.crash.params["unsafe_term_names"] == (
        "drone_crash",
        "illegal_drone",
        "illegal_action",
        "drone_out_of_workspace",
        "path_corridor",
    )

    action = cfg.actions.thrust
    assert action.residual_body_rate_limits is None
    assert action.attitude_hold_gain == pytest.approx(0.0)
    assert action.horizontal_velocity_damping_gain == pytest.approx(0.0)
    assert action.vertical_velocity_damping_gain == pytest.approx(0.0)
    assert action.path_velocity_command_name is None
    assert action.path_velocity_cross_track_gain == pytest.approx(0.0)
    assert action.path_velocity_curvature_feedforward_gain == pytest.approx(0.0)
    assert action.suspended_mass == pytest.approx(0.0)
    assert not action.tilt_compensation


def test_rigid_drone_play_and_evaluation_modes_are_load_agnostic():
    cfg = DroneWaypointDirectCTBREnvCfg()
    cfg.evaluation_mode()
    assert cfg.events.reset_slung_load is None

    cfg = DroneWaypointDirectCTBREnvCfg()
    cfg.play_mode()
    assert cfg.scene.num_envs == 1
    assert cfg.events.reset_slung_load is None
    assert isinstance(cfg.sim.physics.solver_cfg, MJWarpSolverCfg)
    assert cfg.sim.physics.num_substeps == 1
    assert cfg.commands.route.debug_vis


def test_rigid_command_metrics_are_capability_marked_and_do_not_read_load_assets():
    term = object.__new__(mdp.WaypointSequenceCommand)
    term.cfg = SimpleNamespace(record_slung_load_metrics=False)
    term._env = SimpleNamespace(num_envs=2, device="cpu")
    term.metrics = {}
    term._initialize_episode_metrics()
    torch.testing.assert_close(term.metrics["slung_load_metrics_available"], torch.zeros(2))

    pose = torch.zeros(2, 7)
    pose[:, 6] = 1.0
    velocity = torch.zeros(2, 6)
    velocity[:, 0] = torch.tensor((2.0, 4.0))
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_link_pose_w=SimpleNamespace(torch=pose),
            body_com_vel_w=SimpleNamespace(torch=velocity),
        )
    )
    # Deliberately expose only the robot: touching payload/cable would fail.
    term._env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene={"robot": robot},
        episode_length_buf=torch.tensor((10, 10)),
        step_dt=0.01,
    )
    term._record_episode_metrics(
        torch.tensor((0.2, 0.4)),
        waypoint_fraction=torch.tensor((0.25, 0.50)),
        waypoint_completed=torch.tensor((False, False)),
    )

    metrics = term._episode_metrics.metrics
    torch.testing.assert_close(metrics["drone_speed_mean"], torch.tensor((2.0, 4.0)))
    for name in (
        "swing_angle_mean",
        "transverse_speed_rms",
        "payload_speed_mean",
        "cable_relative_separation_mean",
        "cable_joint_error_mean",
    ):
        torch.testing.assert_close(metrics[name], torch.zeros(2))


def test_slung_command_metric_capability_default_is_preserved():
    cfg = DroneSlungLoadWaypointDirectCTBREnvCfg()
    assert cfg.commands.route.record_slung_load_metrics is True
    assert cfg.scene.payload is not None
    assert cfg.scene.cable is not None
    assert cfg.events.reset_slung_load is not None
