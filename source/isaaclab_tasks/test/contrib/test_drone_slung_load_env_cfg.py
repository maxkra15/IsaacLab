# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration contracts for the FLARE Newton slung-load task."""

import math
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from isaaclab_contrib.deformable import VBDSolverCfg

import isaaclab_tasks.contrib.drone_slung_load.mdp as mdp
from isaaclab_tasks.contrib.drone_slung_load.drone_slung_load_env_cfg import (
    GROUND_HEIGHT,
    HOVER_HEIGHT,
    MAX_CABLE_JOINT_ERROR,
    MAX_CABLE_RELATIVE_SEPARATION,
    DroneSlungLoadWaypointDirectCTBREnvCfg,
    DroneSlungLoadWaypointEnhancedEnvCfg,
    DroneSlungLoadWaypointEnvCfg,
)
from isaaclab_tasks.contrib.drone_slung_load.system import (
    CABLE_BEND_DAMPING,
    CABLE_BEND_MODULUS,
    CABLE_DENSITY,
    CABLE_MASS,
    CABLE_NOMINAL_LENGTH,
    CABLE_NUM_POINTS,
    CABLE_STRETCH_MODULUS,
    CABLE_THICKNESS,
    CABLE_TWIST_MODULUS,
    DRONE_DIAGONAL_INERTIA,
    DRONE_MASS,
    PAYLOAD_MASS,
    PAYLOAD_RADIUS,
    ROTOR_ARM_LENGTH,
    ROTOR_HEIGHT,
    ROTOR_MOMENT_COEFFICIENT,
    ROTOR_THRUST_COEFFICIENT,
    ROTOR_YAW_COEFFICIENT,
)

pytestmark = pytest.mark.unit


def _mock_env(positions: torch.Tensor):
    pose = torch.zeros(positions.shape[0], 7)
    pose[:, :3] = positions
    pose[:, 6] = 1.0
    velocity = torch.zeros(positions.shape[0], 6)

    class _Scene(dict):
        def __init__(self):
            super().__init__(
                robot=SimpleNamespace(
                    data=SimpleNamespace(
                        body_link_pose_w=SimpleNamespace(torch=pose),
                        body_com_vel_w=SimpleNamespace(torch=velocity),
                    )
                )
            )
            self.env_origins = torch.zeros(positions.shape[0], 3)

    return SimpleNamespace(num_envs=positions.shape[0], scene=_Scene())


def test_scene_matches_reported_flare_physical_system():
    cfg = DroneSlungLoadWaypointEnvCfg()

    assert pytest.approx(0.305) == DRONE_MASS
    assert pytest.approx(0.070) == PAYLOAD_MASS
    assert pytest.approx(0.50) == CABLE_NOMINAL_LENGTH
    assert CABLE_NUM_POINTS == 9
    assert cfg.scene.robot.spawn.mass_props.mass == DRONE_MASS
    assert cfg.scene.robot.spawn.diagonal_inertia == DRONE_DIAGONAL_INERTIA
    assert pytest.approx((5.6e-4, 5.6e-4, 8.6e-4)) == DRONE_DIAGONAL_INERTIA
    assert cfg.scene.payload.spawn.mass_props.mass == PAYLOAD_MASS
    assert cfg.scene.robot.init_state.pos == (0.0, 0.0, HOVER_HEIGHT)
    assert cfg.scene.payload.init_state.pos == (0.0, 0.0, HOVER_HEIGHT - CABLE_NOMINAL_LENGTH)
    assert cfg.scene.ground.init_state.pos == (0.0, 0.0, GROUND_HEIGHT)
    assert pytest.approx(1.5) == HOVER_HEIGHT - GROUND_HEIGHT
    assert cfg.scene.env_spacing == 0.0
    assert cfg.scene.cable.init_state.pos[2] == HOVER_HEIGHT
    assert cfg.terminations.drone_crash.params["minimum_height"] - GROUND_HEIGHT == pytest.approx(0.18)
    assert cfg.terminations.payload_crash.params["minimum_height"] - GROUND_HEIGHT == pytest.approx(
        PAYLOAD_RADIUS + 0.005
    )
    assert cfg.terminations.drone_out_of_workspace.params["z_max"] - GROUND_HEIGHT == pytest.approx(4.0)
    assert cfg.terminations.payload_out_of_workspace.params["z_max"] - GROUND_HEIGHT == pytest.approx(4.0)
    assert cfg.scene.drone_cable_attach.spawn.coords1 == ((0.0, 0.0, 0.0),)
    assert cfg.scene.cable_payload_attach.spawn.coords1 == ((0.0, 0.0, 0.0),)


def test_cable_is_thin_light_flexible_and_nearly_inextensible():
    cfg = DroneSlungLoadWaypointEnvCfg()
    material = cfg.scene.cable.spawn.physics_material

    assert pytest.approx(0.002) == CABLE_THICKNESS
    assert pytest.approx(1150.0) == CABLE_DENSITY
    assert pytest.approx(CABLE_DENSITY * math.pi * (0.5 * CABLE_THICKNESS) ** 2 * CABLE_NOMINAL_LENGTH) == CABLE_MASS
    assert CABLE_MASS < 0.002
    assert material.stretch_stiffness == CABLE_STRETCH_MODULUS == 5.0e8
    assert material.shear_stiffness is None
    assert material.bend_stiffness == CABLE_BEND_MODULUS == 0.0
    assert material.twist_stiffness == CABLE_TWIST_MODULUS == 0.0
    assert cfg.scene.cable.spawn.collision_props[0].collision_enabled is False


def test_solver_and_action_run_at_paper_control_cadence():
    cfg = DroneSlungLoadWaypointEnvCfg()

    assert cfg.decimation == 1
    assert cfg.sim.dt == pytest.approx(0.01)
    assert cfg.actions.thrust.dt == pytest.approx(0.01)
    assert cfg.sim.physics.num_substeps == 8
    assert isinstance(cfg.sim.physics.solver_cfg, VBDSolverCfg)
    assert cfg.sim.physics.solver_cfg.iterations == 32


def test_action_targets_only_drone_and_matches_visual_rotor_geometry():
    cfg = DroneSlungLoadWaypointEnvCfg()
    action = cfg.actions.thrust

    assert action.asset_name == "robot"
    assert not hasattr(action, "payload_name")
    assert not hasattr(action, "cable_name")
    assert action.arm_length == cfg.scene.robot.spawn.arm_length == ROTOR_ARM_LENGTH
    assert action.arm_length == pytest.approx(0.08)
    assert action.rotor_z == cfg.scene.robot.spawn.rotor_z == ROTOR_HEIGHT
    assert action.max_thrust_to_weight == pytest.approx(3.5)
    assert action.max_body_rates == (15.0, 15.0, 5.0)
    assert action.residual_body_rate_limits is None
    assert action.vertical_velocity_damping_gain == pytest.approx(0.0)
    assert action.suspended_mass == pytest.approx(0.0)
    assert not action.tilt_compensation
    assert action.rate_gains == pytest.approx((0.016, 0.016, 0.028))
    assert action.rate_integral_gains == pytest.approx((0.0, 0.0, 0.0))
    assert action.rate_derivative_gains == pytest.approx((0.0, 0.0, 0.0))
    assert action.rate_integral_error_limits == pytest.approx((0.0, 0.0, 0.0))
    assert action.rate_derivative_cutoff_hz == pytest.approx(30.0)
    assert action.allocation_mode == "collective_priority"
    assert action.maximum_tilt_compensation_angle == pytest.approx(0.5)
    assert action.rotor_thrust_limits == pytest.approx((0.0, 3.5 * DRONE_MASS * 9.81 / 4.0))
    assert action.yaw_coeff == pytest.approx(ROTOR_YAW_COEFFICIENT)
    assert action.yaw_coeff == pytest.approx(ROTOR_MOMENT_COEFFICIENT / ROTOR_THRUST_COEFFICIENT)


def test_actor_observation_is_exact_flare_waypoint_state_plus_previous_action():
    cfg = DroneSlungLoadWaypointEnvCfg()
    policy = cfg.observations.policy

    assert policy.drone_velocity.func is mdp.world_lin_vel_normalized
    assert policy.body_rotation.func is mdp.body_rotation_matrix
    assert policy.swing_angles.func is mdp.swing_angles_normalized
    assert policy.waypoint_offsets.func is mdp.waypoint_offsets_normalized
    assert policy.previous_action.func is mdp.previous_action
    assert not policy.enable_corruption
    assert not hasattr(policy, "body_angular_velocity")
    assert cfg.observations.privileged.cable_separation.func is mdp.cable_relative_separation
    assert cfg.observations.privileged.upper_cable_tangent.func is mdp.upper_cable_tangent_b


def test_rewards_match_flare_equations_and_table_weights():
    cfg = DroneSlungLoadWaypointEnvCfg()

    assert cfg.rewards.progress.func is mdp.waypoint_progress
    assert cfg.rewards.progress.weight == pytest.approx(10.0)
    assert cfg.rewards.action_smoothness.func is mdp.action_delta_l2
    assert cfg.rewards.action_smoothness.weight == pytest.approx(-1.0e-4)
    assert cfg.rewards.swing_safety.func is mdp.swing_safety_impulse
    assert cfg.rewards.swing_safety.weight == pytest.approx(-3.0)
    assert cfg.rewards.crash.func is mdp.crash_impulse
    assert cfg.rewards.crash.weight == pytest.approx(-10.0)


def test_only_reported_initial_swing_is_randomized():
    cfg = DroneSlungLoadWaypointEnvCfg()

    assert not hasattr(cfg.events, "randomize_payload_mass")
    assert cfg.events.reset_slung_load.params["cable_length"] == CABLE_NOMINAL_LENGTH
    assert "payload_radius" not in cfg.events.reset_slung_load.params
    assert cfg.events.reset_slung_load.params["max_initial_swing"] == pytest.approx(0.10)
    pose_range = cfg.events.reset_base.params["pose_range"]
    assert pose_range == {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (HOVER_HEIGHT, HOVER_HEIGHT),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    assert cfg.curriculum is None


def test_maximal_coordinate_guards_detect_nan_and_workspace_exit():
    env = _mock_env(torch.tensor([[0.0, 0.0, 1.5], [7.0, 0.0, 1.5], [float("nan"), 0.0, 1.5]]))

    assert mdp.out_of_workspace(env, x_bound=6.0, y_bound=3.0, z_max=4.0).tolist() == [False, True, False]
    assert mdp.illegal_link_state(env).tolist() == [False, False, True]
    assert mdp.link_height_below_minimum(env, minimum_height=0.18).tolist() == [False, False, False]


@pytest.mark.parametrize(
    ("tracking_error", "expected"),
    [
        ((3.0, -3.0, 2.0), False),
        ((3.001, 0.0, 0.0), True),
        ((0.0, -3.001, 0.0), True),
        ((0.0, 0.0, 2.001), True),
    ],
)
def test_active_waypoint_workspace_guard_is_translation_invariant(tracking_error, expected):
    targets = torch.tensor([[4.0, -4.0, 0.3], [-9.0, 11.0, -2.0]])
    positions = targets + torch.tensor(tracking_error)
    env = _mock_env(positions)
    env.command_manager = SimpleNamespace(get_command=lambda name: torch.cat((targets, targets), dim=-1))

    result = mdp.active_waypoint_error_out_of_bounds(
        env,
        command_name="route",
        x_bound=3.0,
        y_bound=3.0,
        z_bound=2.0,
    )

    assert result.tolist() == [expected, expected]


def test_payload_ground_guard_includes_contact_margin_and_equality():
    cfg = DroneSlungLoadWaypointEnvCfg()
    threshold = cfg.terminations.payload_crash.params["minimum_height"]
    env = _mock_env(torch.tensor([[0.0, 0.0, threshold], [0.0, 0.0, threshold + 1.0e-3]]))

    assert threshold - GROUND_HEIGHT > PAYLOAD_RADIUS
    assert mdp.link_height_below_minimum(env, minimum_height=threshold).tolist() == [True, False]


def test_cable_and_action_safety_terms_are_fail_closed():
    cfg = DroneSlungLoadWaypointEnvCfg()
    segment_pose = torch.zeros(3, 4, 7)
    segment_pose[..., 6] = 1.0
    segment_pose[..., 2] = torch.tensor([-0.0625, -0.1875, -0.3125, -0.4375])
    segment_velocity = torch.zeros(3, 4, 6)
    segment_pose[1, 2, 0] = float("nan")
    segment_velocity[2, 1, 0] = float("inf")
    cable = SimpleNamespace(
        data=SimpleNamespace(
            segment_pose_w=SimpleNamespace(torch=segment_pose),
            segment_velocity_w=SimpleNamespace(torch=segment_velocity),
        )
    )
    env = SimpleNamespace(
        scene={"cable": cable},
        action_manager=SimpleNamespace(
            action=torch.tensor([[0.0, 0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        ),
    )

    assert mdp.illegal_cable_state(env).tolist() == [False, True, True]
    assert mdp.illegal_action(env).tolist() == [False, True, False]
    assert cfg.terminations.cable_integrity.params["max_relative_separation"] == MAX_CABLE_RELATIVE_SEPARATION
    assert cfg.terminations.cable_integrity.params["max_joint_error"] == MAX_CABLE_JOINT_ERROR
    assert cfg.terminations.illegal_cable.func is mdp.illegal_cable_state
    assert cfg.terminations.illegal_action.func is mdp.illegal_action


def test_waypoint_task_registration_and_runner_are_clean_defaults():
    import isaaclab_tasks.contrib.drone_slung_load.config.newton_drone  # noqa: F401
    from isaaclab_tasks.contrib.drone_slung_load.config.newton_drone.agents.rsl_rl_ppo_cfg import (
        DroneSlungLoadWaypointPPORunnerCfg,
    )

    spec = gym.spec("IsaacContrib-DroneSlungLoad-Waypoint-FLARE")
    assert spec.kwargs["env_cfg_entry_point"].endswith(":DroneSlungLoadWaypointEnvCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":DroneSlungLoadWaypointPPORunnerCfg")

    runner = DroneSlungLoadWaypointPPORunnerCfg()
    assert runner.logger == "wandb"
    assert runner.obs_groups == {"actor": ["policy"], "critic": ["policy", "privileged"]}
    expected_hover = 2.0 * ((DRONE_MASS + PAYLOAD_MASS + CABLE_MASS) / DRONE_MASS) / 3.5 - 1.0
    assert runner.actor.distribution_cfg.initial_mean[0] == pytest.approx(expected_hover)


def test_enhanced_task_uses_all_heading_routes_and_continuous_stability_objective():
    import isaaclab_tasks.contrib.drone_slung_load.config.newton_drone  # noqa: F401

    cfg = DroneSlungLoadWaypointEnhancedEnvCfg()
    route = cfg.commands.route

    assert route.randomize_waypoints
    assert not route.regenerate_on_completion
    assert route.random_waypoint_count == 24
    assert route.route_family == "bounded_template_mix"
    assert route.figure_eight_probability == pytest.approx(0.5)
    assert route.samples_per_lap == 24
    assert route.aspect_ratio_range == pytest.approx((0.94, 1.0))
    assert route.vertical_amplitude_range == pytest.approx((0.0, 0.15))
    assert route.random_waypoint_ranges.pos_x == (-4.0, 4.0)
    assert route.random_waypoint_ranges.pos_y == (-4.0, 4.0)
    assert route.random_waypoint_ranges.pos_z == (-0.4, 0.4)
    assert route.minimum_waypoint_separation == pytest.approx(0.75)
    assert route.maximum_waypoint_separation == pytest.approx(1.5)
    assert route.nominal_heading_change == pytest.approx(math.radians(40.0))
    assert route.maximum_heading_change == pytest.approx(math.radians(60.0))
    assert route.maximum_vertical_step == pytest.approx(0.15)
    assert route.random_sampling_attempts == 8
    assert route.route_sampling_attempts == 4
    assert route.acceptance_radius == pytest.approx(0.50)
    assert route.spline_enabled
    assert route.spline_tangent_scale == pytest.approx(1.0)
    assert route.spline_projection_samples == 12
    assert route.spline_progressive_advancement
    assert route.spline_plane_crossing_lateral_tolerance == pytest.approx(0.30)
    assert route.spline_max_waypoint_advances_per_step == 4
    assert route.target_cruise_speed == pytest.approx(1.25)
    assert route.maximum_lateral_acceleration == pytest.approx(3.0)
    assert route.maximum_braking_acceleration == pytest.approx(4.0)
    assert route.speed_lookahead_distances == pytest.approx((0.0, 0.75, 1.50))
    assert cfg.episode_length_s == pytest.approx(15.0)
    for term, asset_name in (
        (cfg.terminations.drone_out_of_workspace, "robot"),
        (cfg.terminations.payload_out_of_workspace, "payload"),
    ):
        assert term.func is mdp.active_waypoint_error_out_of_bounds
        assert term.params["command_name"] == "route"
        assert term.params["x_bound"] == pytest.approx(3.0)
        assert term.params["y_bound"] == pytest.approx(3.0)
        assert term.params["z_bound"] == pytest.approx(2.0)
        assert term.params["asset_cfg"].name == asset_name
    assert cfg.observations.policy.swing_angular_velocity.func is mdp.swing_angular_velocity_normalized
    assert cfg.observations.policy.path_tracking.func is mdp.path_tracking_features_b
    assert cfg.observations.policy.path_tracking.params["lookahead_distances"] == (0.75, 1.50)
    assert cfg.observations.policy.path_tracking.params["cross_track_scale"] == pytest.approx(0.20)
    assert cfg.observations.policy.path_speed.func is mdp.path_speed_features
    assert cfg.observations.policy.path_speed.params["speed_scale"] == pytest.approx(3.5)
    assert cfg.rewards.progress is None
    assert cfg.rewards.path_progress.func is mdp.path_arc_length_progress
    assert cfg.rewards.path_progress.weight == pytest.approx(3.0)
    assert cfg.rewards.path_progress.params["maximum_rate"] == pytest.approx(1.25)
    assert cfg.rewards.path_progress.params["maximum_lateral_acceleration"] == pytest.approx(3.0)
    assert cfg.rewards.path_speed.func is mdp.path_tangent_speed_tracking_l2
    assert cfg.rewards.path_speed.weight == pytest.approx(-1.0)
    assert cfg.rewards.path_speed.params["overspeed_weight"] == pytest.approx(2.0)
    assert cfg.rewards.completion.func is mdp.RouteCompletionImpulse
    assert cfg.rewards.completion.weight == pytest.approx(50.0)
    assert cfg.rewards.completion.params["early_completion_scale"] == pytest.approx(1.0)
    assert cfg.rewards.crash.func is mdp.unsafe_termination_impulse
    assert cfg.rewards.crash.weight == pytest.approx(-100.0)
    assert "route_completed" not in cfg.rewards.crash.params["unsafe_term_names"]
    assert cfg.rewards.path_precision.func is mdp.path_tracking_precision_log1p
    assert cfg.rewards.swing_magnitude.func is mdp.total_swing_angle_l2
    assert cfg.rewards.transverse_speed.func is mdp.payload_transverse_speed_l2
    assert cfg.rewards.body_rate.func is mdp.body_angular_velocity_l2
    assert cfg.rewards.body_tilt.func is mdp.body_tilt_exp
    assert cfg.rewards.action_acceleration.func is mdp.NormalizedActionAccelerationL2
    assert cfg.rewards.path_precision.weight == pytest.approx(-2.0)
    assert cfg.rewards.path_precision.params["cross_track_scale"] == pytest.approx(0.50)
    assert cfg.rewards.path_precision.params["transverse_velocity_scale"] == pytest.approx(1.00)
    assert cfg.rewards.path_precision.params["transverse_speed_weight"] == pytest.approx(0.25)
    assert cfg.rewards.swing_magnitude.weight == pytest.approx(-0.25)
    assert cfg.rewards.transverse_speed.weight == pytest.approx(-0.05)
    assert cfg.rewards.body_rate.weight == pytest.approx(-0.005)
    assert cfg.rewards.body_tilt.weight == pytest.approx(-0.1)
    assert cfg.rewards.action_acceleration.weight == pytest.approx(-0.005)
    assert cfg.actions.thrust.attitude_hold_gain == pytest.approx(12.0)
    assert cfg.actions.thrust.residual_body_rate_limits == pytest.approx((10.0, 10.0, 2.5))
    assert cfg.actions.thrust.max_body_rates == pytest.approx((15.0, 15.0, 5.0))
    assert cfg.actions.thrust.horizontal_velocity_damping_gain == pytest.approx(4.0)
    assert cfg.actions.thrust.vertical_velocity_damping_gain == pytest.approx(2.0)
    assert cfg.actions.thrust.path_velocity_command_name == "route"
    assert cfg.actions.thrust.path_velocity_cross_track_gain == pytest.approx(1.5)
    assert cfg.actions.thrust.path_velocity_maximum_cross_track_speed == pytest.approx(0.75)
    assert cfg.actions.thrust.path_velocity_curvature_feedforward_gain == pytest.approx(1.0)
    assert cfg.actions.thrust.suspended_mass == pytest.approx(PAYLOAD_MASS + CABLE_MASS)
    assert cfg.actions.thrust.tilt_compensation
    assert cfg.actions.thrust.maximum_velocity_hold_tilt == pytest.approx(0.42)
    assert cfg.actions.thrust.maximum_tilt_compensation_angle == pytest.approx(0.5)
    assert cfg.events.reset_base.func is mdp.reset_drone_state_on_annulus
    assert cfg.events.reset_base.params["radius_range"] == pytest.approx((4.3, 4.8))
    assert cfg.events.reset_base.params["height"] == pytest.approx(HOVER_HEIGHT)
    assert cfg.events.reset_base.params["roll_range"] == pytest.approx((-0.05, 0.05))
    assert cfg.events.reset_base.params["pitch_range"] == pytest.approx((-0.05, 0.05))
    assert cfg.events.reset_base.params["yaw"] == pytest.approx(0.0)
    assert cfg.terminations.path_corridor.func is mdp.path_corridor_violation
    assert cfg.terminations.path_corridor.params["maximum_distance"] == pytest.approx(1.50)
    assert cfg.terminations.route_completed.func is mdp.route_completed
    assert cfg.scene.cable.spawn.physics_material.bend_damping == pytest.approx(CABLE_BEND_DAMPING)
    assert cfg.curriculum.precision_speed.func is mdp.PrecisionSpeedCurriculumV13
    assert cfg.curriculum.precision_speed.update_mode == "step"
    baseline = DroneSlungLoadWaypointEnvCfg()
    assert baseline.scene.cable.spawn.physics_material.bend_damping is None
    assert baseline.actions.thrust.attitude_hold_gain == pytest.approx(0.0)
    assert baseline.actions.thrust.horizontal_velocity_damping_gain == pytest.approx(0.0)
    assert baseline.actions.thrust.vertical_velocity_damping_gain == pytest.approx(0.0)
    assert baseline.actions.thrust.path_velocity_command_name is None
    assert baseline.actions.thrust.suspended_mass == pytest.approx(0.0)
    assert not baseline.actions.thrust.tilt_compensation
    assert baseline.actions.thrust.maximum_tilt_compensation_angle == pytest.approx(0.5)
    assert baseline.terminations.drone_out_of_workspace.params["x_bound"] == pytest.approx(6.0)
    assert baseline.terminations.drone_out_of_workspace.params["y_bound"] == pytest.approx(3.0)
    assert baseline.terminations.drone_out_of_workspace.func is mdp.out_of_workspace
    assert baseline.terminations.payload_out_of_workspace.func is mdp.out_of_workspace
    assert baseline.rewards.crash.weight == pytest.approx(-10.0)

    spec = gym.spec("IsaacContrib-DroneSlungLoad-Waypoint-FLARE-Enhanced")
    assert spec.kwargs["env_cfg_entry_point"].endswith(":DroneSlungLoadWaypointEnhancedEnvCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":DroneSlungLoadWaypointEnhancedPPORunnerCfg")

    cfg.evaluation_mode()
    assert cfg.commands.route.randomize_waypoints
    assert not cfg.commands.route.regenerate_on_completion
    assert cfg.commands.route.random_waypoint_count == 24
    assert cfg.commands.route.acceptance_radius == pytest.approx(0.15)
    assert cfg.rewards.path_progress.params["maximum_rate"] == pytest.approx(3.50)
    assert cfg.rewards.path_progress.params["maximum_lateral_acceleration"] == pytest.approx(3.0)
    assert cfg.rewards.path_precision.weight == pytest.approx(-4.0)
    assert cfg.rewards.path_precision.params["cross_track_scale"] == pytest.approx(0.20)
    assert cfg.rewards.path_precision.params["transverse_velocity_scale"] == pytest.approx(0.40)
    assert cfg.terminations.path_corridor.params["maximum_distance"] == pytest.approx(0.75)
    assert cfg.curriculum is None
    assert cfg.episode_length_s == pytest.approx(15.0)


def test_direct_ctbr_task_uses_route_first_objective_and_removes_every_actuation_prior():
    import isaaclab_tasks.contrib.drone_slung_load.config.newton_drone  # noqa: F401

    direct = DroneSlungLoadWaypointDirectCTBREnvCfg()
    enhanced = DroneSlungLoadWaypointEnhancedEnvCfg()
    action = direct.actions.thrust

    for name in (
        "random_waypoint_count",
        "samples_per_lap",
        "spline_tangent_scale",
        "spline_projection_samples",
    ):
        assert getattr(direct.commands.route, name) == getattr(enhanced.commands.route, name)
    assert direct.commands.route.randomize_waypoints
    assert not direct.commands.route.regenerate_on_completion
    assert direct.commands.route.random_waypoint_count == 24
    assert direct.commands.route.route_family == "random_walk"
    assert direct.commands.route.random_waypoint_ranges.pos_x == pytest.approx((-12.5, 12.5))
    assert direct.commands.route.random_waypoint_ranges.pos_y == pytest.approx((-12.5, 12.5))
    assert direct.commands.route.random_waypoint_ranges.pos_z == pytest.approx((0.0, 0.0))
    assert direct.commands.route.minimum_waypoint_separation == pytest.approx(0.49)
    assert direct.commands.route.maximum_waypoint_separation == pytest.approx(0.51)
    assert direct.commands.route.nominal_heading_change == pytest.approx(0.0)
    assert direct.commands.route.maximum_heading_change == pytest.approx(math.radians(5.0))
    assert direct.commands.route.maximum_vertical_step == pytest.approx(0.0)
    assert direct.commands.route.aspect_ratio_range == pytest.approx((0.94, 1.0))
    assert direct.commands.route.vertical_amplitude_range == pytest.approx((0.0, 0.0))
    assert direct.commands.route.figure_eight_probability == pytest.approx(0.0)
    assert direct.commands.route.acceptance_radius == pytest.approx(1.0)
    assert direct.commands.route.target_cruise_speed == pytest.approx(1.00)
    assert direct.events.reset_base.func is enhanced.events.reset_base.func is mdp.reset_drone_state_on_annulus
    assert direct.events.reset_base.params["radius_range"] == enhanced.events.reset_base.params["radius_range"]
    assert direct.events.reset_base.params["roll_range"] == pytest.approx((-0.005, 0.005))
    assert direct.events.reset_base.params["pitch_range"] == pytest.approx((-0.005, 0.005))
    assert direct.events.reset_slung_load.params["max_initial_swing"] == pytest.approx(0.02)

    assert action.max_body_rates == pytest.approx((15.0, 15.0, 5.0))
    assert action.residual_body_rate_limits is None
    assert action.attitude_hold_gain == pytest.approx(0.0)
    assert action.horizontal_velocity_damping_gain == pytest.approx(0.0)
    assert action.vertical_velocity_damping_gain == pytest.approx(0.0)
    assert action.path_velocity_command_name is None
    assert action.path_velocity_cross_track_gain == pytest.approx(0.0)
    assert action.path_velocity_curvature_feedforward_gain == pytest.approx(0.0)
    assert action.suspended_mass == pytest.approx(0.0)
    assert not action.tilt_compensation
    assert action.rate_gains == pytest.approx((0.016, 0.016, 0.028))
    assert action.rate_integral_gains == pytest.approx((0.040, 0.040, 0.070))
    assert action.rate_derivative_gains == pytest.approx((8.0e-5, 8.0e-5, 1.4e-4))
    assert action.rate_integral_error_limits == pytest.approx((0.50, 0.50, 0.11))
    assert action.rate_derivative_cutoff_hz == pytest.approx(20.0)
    assert action.allocation_mode == "rate_priority"

    assert enhanced.observations.policy.drone_velocity.func is mdp.world_lin_vel_normalized
    assert enhanced.observations.policy.path_tracking.params["cross_track_scale"] == pytest.approx(0.20)
    assert direct.observations.policy.drone_velocity.func is mdp.body_lin_vel_normalized
    assert direct.observations.policy.drone_velocity.params["speed_scale"] == pytest.approx(4.5)
    assert direct.observations.policy.body_angular_velocity.func is mdp.body_ang_vel_normalized
    assert direct.observations.privileged.body_angular_velocity is None
    assert direct.observations.policy.path_tracking.func is mdp.path_tracking_features_b
    assert direct.observations.policy.path_tracking.params["cross_track_scale"] == pytest.approx(1.00)
    assert direct.observations.policy.path_speed.params["speed_scale"] == pytest.approx(4.5)
    assert direct.commands.route.maximum_lateral_acceleration == pytest.approx(3.0)
    assert direct.commands.route.maximum_braking_acceleration == pytest.approx(4.0)
    assert direct.commands.route.speed_lookahead_distances == pytest.approx((0.0, 0.75, 1.50, 2.25))
    assert direct.rewards.path_progress.weight == pytest.approx(1.0)
    assert direct.rewards.path_progress.params["maximum_rate"] == pytest.approx(1.00)
    assert direct.rewards.path_progress.params["maximum_lateral_acceleration"] == pytest.approx(3.0)
    assert direct.rewards.path_progress.params["positive_progress_gate_distance"] == pytest.approx(0.75)
    assert "positive_progress_gate_distance" not in enhanced.rewards.path_progress.params
    assert direct.rewards.waypoint_advance is None
    assert direct.rewards.action_smoothness is None
    assert direct.rewards.swing_safety is None
    assert direct.rewards.path_speed is None
    assert direct.rewards.path_velocity.func is mdp.path_velocity_tracking_l2
    assert direct.rewards.path_velocity.weight == pytest.approx(-0.20)
    assert direct.rewards.path_velocity.params["command_name"] == "route"
    assert direct.rewards.path_velocity.params["cross_track_gain"] == pytest.approx(1.5)
    assert direct.rewards.path_velocity.params["maximum_cross_track_speed"] == pytest.approx(0.75)
    assert direct.rewards.path_velocity.params["velocity_error_scale"] == pytest.approx(1.0)
    assert direct.rewards.path_velocity.params["asset_cfg"].name == "robot"
    assert direct.rewards.completion.weight == pytest.approx(25.0)
    assert direct.rewards.completion.params["early_completion_scale"] == pytest.approx(0.0)
    assert direct.rewards.path_precision.func is mdp.path_tracking_precision_exp
    assert direct.rewards.path_precision.weight == pytest.approx(-0.10)
    assert direct.rewards.path_precision.params["cross_track_scale"] == pytest.approx(1.00)
    assert direct.rewards.path_precision.params["transverse_velocity_scale"] == pytest.approx(1.50)
    assert direct.rewards.swing_magnitude.weight == pytest.approx(-0.05)
    assert direct.rewards.transverse_speed.weight == pytest.approx(-0.01)
    assert direct.rewards.body_rate.weight == pytest.approx(-0.001)
    assert direct.rewards.body_tilt.weight == pytest.approx(-0.02)
    assert direct.rewards.action_acceleration.weight == pytest.approx(-0.001)
    assert direct.terminations.path_corridor.params["maximum_distance"] == pytest.approx(2.5)
    assert direct.curriculum.precision_speed.func is mdp.DirectCTBRRouteCurriculum
    assert direct.curriculum.precision_speed.params == {}
    assert direct.episode_length_s == pytest.approx(20.0)

    spec = gym.spec("IsaacContrib-DroneSlungLoad-Waypoint-FLARE-DirectCTBR")
    assert spec.kwargs["env_cfg_entry_point"].endswith(":DroneSlungLoadWaypointDirectCTBREnvCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":DroneSlungLoadWaypointDirectCTBRPPORunnerCfg")

    direct.evaluation_mode()
    assert direct.curriculum is None
    assert direct.commands.route.randomize_waypoints
    assert not direct.commands.route.regenerate_on_completion
    assert direct.commands.route.random_waypoint_count == 24
    assert direct.commands.route.route_family == "bounded_hard_mix"
    assert direct.commands.route.samples_per_lap == 24
    assert direct.commands.route.aspect_ratio_range == pytest.approx((0.94, 1.0))
    assert direct.commands.route.figure_eight_probability == pytest.approx(0.50)
    assert direct.commands.route.vertical_amplitude_range == pytest.approx((0.0, 0.15))
    assert direct.commands.route.random_waypoint_ranges.pos_x == pytest.approx((-4.80, 4.80))
    assert direct.commands.route.random_waypoint_ranges.pos_y == pytest.approx((-4.80, 4.80))
    assert direct.commands.route.random_waypoint_ranges.pos_z == pytest.approx((-0.40, 0.40))
    assert direct.commands.route.minimum_waypoint_separation == pytest.approx(0.90)
    assert direct.commands.route.maximum_waypoint_separation == pytest.approx(1.35)
    assert direct.commands.route.nominal_heading_change == pytest.approx(math.radians(100.0))
    assert direct.commands.route.maximum_heading_change == pytest.approx(math.radians(110.0))
    assert direct.commands.route.maximum_vertical_step == pytest.approx(0.15)
    assert direct.commands.route.random_sampling_attempts == 32
    assert direct.commands.route.route_sampling_attempts == 16
    assert direct.commands.route.random_heading_change_interval == 3
    assert direct.commands.route.acceptance_radius == pytest.approx(0.50)
    assert direct.commands.route.target_cruise_speed == pytest.approx(3.50)
    assert direct.rewards.path_progress.params["maximum_rate"] == pytest.approx(3.50)
    assert direct.rewards.path_progress.params["positive_progress_gate_distance"] == pytest.approx(0.75)
    assert direct.terminations.path_corridor.params["maximum_distance"] == pytest.approx(1.50)
    assert direct.episode_length_s == pytest.approx(20.0)


def test_play_mode_is_deterministic_and_preserves_physics():
    from isaaclab_tasks.contrib.drone_slung_load.drone_slung_load_env_cfg import (
        FIGURE_EIGHT_EVAL_WAYPOINT_OFFSETS,
    )

    cfg = DroneSlungLoadWaypointEnvCfg()
    assert not cfg.commands.route.debug_vis
    cfg.play_mode()

    assert cfg.scene.num_envs == 1
    assert cfg.events.reset_slung_load.params["max_initial_swing"] == 0.0
    assert cfg.sim.physics.num_substeps == 8
    assert cfg.sim.physics.solver_cfg.iterations == 32
    assert cfg.commands.route.debug_vis
    assert cfg.commands.route.waypoint_offsets == FIGURE_EIGHT_EVAL_WAYPOINT_OFFSETS


def test_figure_eight_evaluation_route_is_three_closed_laps():
    from isaaclab_tasks.contrib.drone_slung_load.drone_slung_load_env_cfg import (
        FIGURE_EIGHT_EVAL_WAYPOINT_OFFSETS,
    )

    waypoints = torch.tensor(((0.0, 0.0, 0.0), *FIGURE_EIGHT_EVAL_WAYPOINT_OFFSETS))
    assert waypoints.shape == (49, 3)
    for lap_end in (16, 32, 48):
        torch.testing.assert_close(waypoints[lap_end], waypoints[0], atol=1.0e-6, rtol=0.0)
    # Each lap crosses at its quarter and three-quarter points and reaches the
    # opposite tip halfway through.
    torch.testing.assert_close(waypoints[4], torch.tensor([2.0, 0.0, 0.0]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(waypoints[8], torch.tensor([4.0, 0.0, 0.0]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(waypoints[12], torch.tensor([2.0, 0.0, 0.0]), atol=1.0e-6, rtol=0.0)
    assert waypoints[:, 1].min() == pytest.approx(-1.0)
    assert waypoints[:, 1].max() == pytest.approx(1.0)
