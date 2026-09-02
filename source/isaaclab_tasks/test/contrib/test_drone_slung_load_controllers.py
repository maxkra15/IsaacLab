# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the FLARE thrust/body-rate controller."""

import math

import pytest
import torch

from isaaclab_tasks.contrib.drone_slung_load.mdp.controllers import (
    FlareController,
    allocate_rotor_thrusts,
    apply_motor_lag,
    body_rate_torque,
    quadrotor_allocation_matrix,
    quadrotor_rotor_geometry,
    reconstruct_wrench,
    scale_flare_actions,
)
from isaaclab_tasks.contrib.drone_slung_load.system import ROTOR_ARM_LENGTH, ROTOR_YAW_COEFFICIENT

pytestmark = pytest.mark.unit


def test_flare_action_scaling_clamps_extrema_and_maps_zero():
    actions = torch.tensor(
        [
            [-2.0, -2.0, -2.0, -2.0],
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 2.0],
        ]
    )
    command = scale_flare_actions(actions, drone_mass=torch.ones(3), gravity=10.0)

    torch.testing.assert_close(
        command,
        torch.tensor(
            [
                [0.0, -15.0, -15.0, -5.0],
                [17.5, 0.0, 0.0, 0.0],
                [35.0, 15.0, 15.0, 5.0],
            ]
        ),
    )


def test_flare_collective_thrust_uses_drone_mass():
    actions = torch.tensor([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
    command = scale_flare_actions(actions, drone_mass=torch.tensor([0.75, 2.0]), gravity=9.81)

    torch.testing.assert_close(command[:, 0], torch.tensor([25.75125, 0.0]))


def test_allocation_matrix_matches_forces_at_rotor_sites_and_yaw_reaction_torques():
    arm_length = 0.13
    yaw_coeff = 0.07
    rotor_thrusts = torch.tensor([2.0, 3.0, 4.0, 5.0])
    rotor_positions_b, yaw_directions = quadrotor_rotor_geometry(arm_length=arm_length, rotor_z=0.03)
    rotor_forces_b = torch.zeros(4, 3)
    rotor_forces_b[:, 2] = rotor_thrusts
    reaction_torques_b = torch.zeros(4, 3)
    reaction_torques_b[:, 2] = yaw_directions * yaw_coeff * rotor_thrusts

    net_force_b = rotor_forces_b.sum(dim=0)
    net_torque_b = (torch.linalg.cross(rotor_positions_b, rotor_forces_b) + reaction_torques_b).sum(dim=0)
    site_wrench = torch.cat((net_force_b[2:3], net_torque_b))
    matrix_wrench = reconstruct_wrench(
        rotor_thrusts.unsqueeze(0),
        quadrotor_allocation_matrix(arm_length=arm_length, yaw_coeff=yaw_coeff),
    ).squeeze(0)

    torch.testing.assert_close(site_wrench, matrix_wrench)


def test_default_rotor_geometry_and_allocator_match_later_flare_vehicle():
    positions, yaw_directions = quadrotor_rotor_geometry()
    allocation = quadrotor_allocation_matrix()

    torch.testing.assert_close(
        positions[:, :2],
        torch.tensor(
            [
                [ROTOR_ARM_LENGTH, -ROTOR_ARM_LENGTH],
                [-ROTOR_ARM_LENGTH, -ROTOR_ARM_LENGTH],
                [-ROTOR_ARM_LENGTH, ROTOR_ARM_LENGTH],
                [ROTOR_ARM_LENGTH, ROTOR_ARM_LENGTH],
            ]
        ),
    )
    torch.testing.assert_close(allocation[1], positions[:, 1])
    torch.testing.assert_close(allocation[2], -positions[:, 0])
    torch.testing.assert_close(allocation[3], yaw_directions * ROTOR_YAW_COEFFICIENT)


def test_body_rate_torque_uses_rate_error_and_axis_limits():
    torque, error = body_rate_torque(
        rate_command=torch.tensor([[2.0, -3.0, 0.5]]),
        body_rate=torch.tensor([[0.5, 1.0, -0.5]]),
        gains=(2.0, 1.0, 0.25),
        torque_limits=(1.0, 2.0, 0.2),
    )

    torch.testing.assert_close(error, torch.tensor([[1.5, -4.0, 1.0]]))
    torch.testing.assert_close(torque, torch.tensor([[1.0, -2.0, 0.2]]))


def test_allocation_reconstructs_unclipped_feasible_wrench():
    allocation = quadrotor_allocation_matrix(arm_length=0.13, yaw_coeff=0.07)
    rotor_thrusts = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
    wrench = reconstruct_wrench(rotor_thrusts, allocation)

    allocated = allocate_rotor_thrusts(wrench, allocation, thrust_limits=(0.0, 10.0))

    torch.testing.assert_close(allocated, rotor_thrusts, atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(reconstruct_wrench(allocated, allocation), wrench, atol=1.0e-5, rtol=1.0e-5)
    assert torch.isfinite(allocated).all()


def test_rotor_allocation_clamps_infeasible_negative_collective_to_zero():
    allocation = quadrotor_allocation_matrix()
    impossible_wrench = torch.tensor([[-10.0, 100.0, -100.0, 50.0]])

    allocated = allocate_rotor_thrusts(impossible_wrench, allocation, thrust_limits=(0.0, 6.0))

    assert torch.all(allocated >= 0.0)
    assert torch.all(allocated <= 6.0)
    assert torch.isfinite(allocated).all()
    torch.testing.assert_close(allocated, torch.zeros_like(allocated))


def test_rotor_allocation_preserves_collective_and_moment_direction_when_desaturated():
    allocation = quadrotor_allocation_matrix()
    desired = torch.tensor([[12.0, 1.0, -0.8, 0.4]])

    allocated = allocate_rotor_thrusts(desired, allocation, thrust_limits=(0.0, 6.0))
    realized = reconstruct_wrench(allocated, allocation)

    torch.testing.assert_close(realized[:, 0], desired[:, 0])
    assert torch.all((allocated >= 0.0) & (allocated <= 6.0))
    moment_scale = realized[:, 1:] / desired[:, 1:]
    torch.testing.assert_close(moment_scale, moment_scale[:, :1].expand_as(moment_scale), atol=1.0e-5, rtol=1.0e-5)
    assert 0.0 < moment_scale[0, 0].item() < 1.0


def test_rate_priority_allocation_shifts_collective_before_uniformly_scaling_moments():
    allocation = quadrotor_allocation_matrix(arm_length=0.13, yaw_coeff=0.07)
    differential = torch.tensor(
        [
            [-0.8, 0.4, -0.2, 0.6],
            [-1.6, 0.8, -0.4, 1.2],
        ]
    )
    unconstrained = 0.2 + differential
    desired = reconstruct_wrench(unconstrained, allocation)

    allocated = allocate_rotor_thrusts(
        desired,
        allocation,
        thrust_limits=(0.0, 2.0),
        allocation_mode="rate_priority",
    )
    realized = reconstruct_wrench(allocated, allocation)

    assert torch.all((allocated >= 0.0) & (allocated <= 2.0))
    assert torch.all(realized[:, 0] > desired[:, 0])
    torch.testing.assert_close(realized[0, 1:], desired[0, 1:], atol=1.0e-6, rtol=1.0e-6)
    moment_scale = realized[1, 1:] / desired[1, 1:]
    torch.testing.assert_close(moment_scale, moment_scale[:1].expand_as(moment_scale), atol=1.0e-6, rtol=1.0e-6)
    assert 0.0 < moment_scale[0].item() < 1.0


def test_flare_action_scaling_sanitizes_nonfinite_policy_output():
    command = scale_flare_actions(
        torch.tensor([[float("nan"), float("inf"), -float("inf"), 0.0]]),
        drone_mass=torch.ones(1),
        gravity=10.0,
    )

    torch.testing.assert_close(command, torch.tensor([[17.5, 15.0, -15.0, 0.0]]))


def test_motor_lag_rises_slower_than_it_falls():
    rising = apply_motor_lag(
        target=torch.ones(1, 4),
        current=torch.zeros(1, 4),
        dt=0.01,
        tau_up=0.065,
        tau_down=0.005,
    )
    falling = apply_motor_lag(
        target=torch.zeros(1, 4),
        current=torch.ones(1, 4),
        dt=0.01,
        tau_up=0.065,
        tau_down=0.005,
    )

    expected_rise = 1.0 - math.exp(-0.01 / 0.065)
    expected_fall = math.exp(-0.01 / 0.005)
    torch.testing.assert_close(rising, torch.full((1, 4), expected_rise))
    torch.testing.assert_close(falling, torch.full((1, 4), expected_fall))
    assert rising[0, 0] < 1.0 - falling[0, 0]


def test_flare_controller_reset_clears_selected_motor_state():
    controller = FlareController(num_envs=2, device="cpu", rotor_thrust_limits=(0.0, 20.0))
    command = torch.tensor([[10.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]])
    controller.compute(command, body_rate=torch.zeros(2, 3))
    before_reset = controller.motor_thrusts.clone()

    controller.reset(torch.tensor([1]))

    torch.testing.assert_close(controller.motor_thrusts[0], before_reset[0])
    torch.testing.assert_close(controller.motor_thrusts[1], torch.zeros(4))


def test_flare_controller_pid_filters_measured_rate_derivative_without_command_kick():
    dt = 0.1
    cutoff_hz = 1.0
    derivative_gain = 0.01
    controller = FlareController(
        num_envs=1,
        device="cpu",
        dt=dt,
        rate_gains=0.0,
        rate_derivative_gains=(derivative_gain, 0.0, 0.0),
        rate_derivative_cutoff_hz=cutoff_hz,
        torque_limits=10.0,
        rotor_thrust_limits=(0.0, 20.0),
    )

    first = controller.compute(torch.tensor([[4.0, 0.0, 0.0, 0.0]]), torch.tensor([[1.0, 0.0, 0.0]]))
    second = controller.compute(torch.tensor([[4.0, 6.0, 0.0, 0.0]]), torch.tensor([[2.0, 0.0, 0.0]]))
    third = controller.compute(torch.tensor([[4.0, -6.0, 0.0, 0.0]]), torch.tensor([[2.0, 0.0, 0.0]]))

    decay = math.exp(-2.0 * math.pi * cutoff_hz * dt)
    expected_second_derivative = (1.0 - decay) / dt
    torch.testing.assert_close(first.torque_command, torch.zeros(1, 3))
    torch.testing.assert_close(
        second.torque_command[:, 0],
        torch.tensor([-derivative_gain * expected_second_derivative]),
    )
    torch.testing.assert_close(
        third.torque_command[:, 0],
        torch.tensor([-derivative_gain * decay * expected_second_derivative]),
    )


def test_flare_controller_pid_bounds_integral_and_blocks_windup_at_torque_limit():
    controller = FlareController(
        num_envs=1,
        device="cpu",
        dt=0.1,
        rate_gains=(2.0, 0.0, 0.0),
        rate_integral_gains=(1.0, 1.0, 0.0),
        rate_integral_error_limits=(0.5, 0.25, 0.0),
        torque_limits=(1.0, 10.0, 1.0),
        rotor_thrust_limits=(0.0, 20.0),
    )
    command = torch.tensor([[4.0, 1.0, 1.0, 0.0]])

    for _ in range(5):
        output = controller.compute(command, torch.zeros(1, 3))

    torch.testing.assert_close(controller.rate_error_integral, torch.tensor([[0.0, 0.25, 0.0]]))
    torch.testing.assert_close(output.torque_command, torch.tensor([[1.0, 0.25, 0.0]]))


def test_flare_controller_subset_reset_clears_pid_state_and_reinitializes_derivative():
    controller = FlareController(
        num_envs=3,
        device="cpu",
        dt=0.1,
        rate_gains=0.0,
        rate_integral_gains=0.1,
        rate_derivative_gains=0.01,
        rate_integral_error_limits=1.0,
        rate_derivative_cutoff_hz=1.0,
        torque_limits=10.0,
        rotor_thrust_limits=(0.0, 20.0),
    )
    command = torch.tensor([[4.0, 1.0, 1.0, 1.0]]).expand(3, -1)
    controller.compute(command, torch.ones(3, 3))
    controller.compute(command, 2.0 * torch.ones(3, 3))
    middle_integral = controller.rate_error_integral[1].clone()
    middle_derivative = controller.filtered_body_rate_derivative[1].clone()
    middle_motor_thrusts = controller.motor_thrusts[1].clone()

    controller.reset((0, 2))

    torch.testing.assert_close(controller.rate_error_integral[[0, 2]], torch.zeros(2, 3))
    torch.testing.assert_close(controller.filtered_body_rate_derivative[[0, 2]], torch.zeros(2, 3))
    torch.testing.assert_close(controller.motor_thrusts[[0, 2]], torch.zeros(2, 4))
    torch.testing.assert_close(controller.rate_error_integral[1], middle_integral)
    torch.testing.assert_close(controller.filtered_body_rate_derivative[1], middle_derivative)
    torch.testing.assert_close(controller.motor_thrusts[1], middle_motor_thrusts)

    controller.compute(command, 3.0 * torch.ones(3, 3))
    torch.testing.assert_close(controller.filtered_body_rate_derivative[[0, 2]], torch.zeros(2, 3))


def test_zero_pid_gains_and_default_allocation_preserve_proportional_controller_bitwise():
    default = FlareController(num_envs=4, device="cpu", rotor_thrust_limits=(0.0, 20.0))
    explicit_legacy = FlareController(
        num_envs=4,
        device="cpu",
        rate_integral_gains=0.0,
        rate_derivative_gains=0.0,
        rate_integral_error_limits=100.0,
        rate_derivative_cutoff_hz=2.0,
        allocation_mode="collective_priority",
        rotor_thrust_limits=(0.0, 20.0),
    )
    command = torch.tensor(
        [
            [4.0, 1.0, -2.0, 0.5],
            [9.0, -5.0, 4.0, -1.0],
            [2.0, 20.0, -20.0, 10.0],
            [-1.0, 0.0, 0.0, 0.0],
        ]
    )
    body_rate = torch.tensor(
        [
            [0.5, -1.0, 0.0],
            [-2.0, 3.0, -0.5],
            [1.0, -1.0, 2.0],
            [0.0, 0.0, 0.0],
        ]
    )

    for _ in range(3):
        default_output = default.compute(command, body_rate)
        explicit_output = explicit_legacy.compute(command, body_rate)

        assert torch.equal(default_output.rate_error, explicit_output.rate_error)
        assert torch.equal(default_output.torque_command, explicit_output.torque_command)
        assert torch.equal(default_output.target_rotor_thrusts, explicit_output.target_rotor_thrusts)
        assert torch.equal(default_output.rotor_thrusts, explicit_output.rotor_thrusts)
        assert torch.equal(default_output.wrench, explicit_output.wrench)


def test_flare_controller_cached_path_matches_validated_public_reference():
    generator = torch.Generator().manual_seed(8)
    num_envs = 256
    controller = FlareController(
        num_envs=num_envs,
        device="cpu",
        dt=0.008,
        rate_gains=(0.017, 0.015, 0.029),
        torque_limits=(0.18, 0.21, 0.075),
        arm_length=0.06,
        yaw_coeff=0.012,
        rotor_thrust_limits=(0.0, 2.8),
        tau_up=0.047,
        tau_down=0.019,
    )
    reference_motor_thrusts = torch.empty(num_envs, 4).uniform_(0.0, 2.8, generator=generator)
    controller.motor_thrusts.copy_(reference_motor_thrusts)

    for _ in range(5):
        command = torch.empty(num_envs, 4).uniform_(-1.0, 1.0, generator=generator)
        command[:, 0] = torch.empty(num_envs).uniform_(-5.0, 18.0, generator=generator)
        body_rate = torch.empty(num_envs, 3).uniform_(-20.0, 20.0, generator=generator)

        reference_torque, reference_error = body_rate_torque(
            command[:, 1:4],
            body_rate,
            gains=controller.rate_gains,
            torque_limits=controller.torque_limits,
        )
        reference_desired_wrench = torch.cat((command[:, :1], reference_torque), dim=-1)
        reference_targets = allocate_rotor_thrusts(
            reference_desired_wrench,
            controller.allocation_matrix,
            thrust_limits=controller.rotor_thrust_limits,
        )
        reference_motor_thrusts = apply_motor_lag(
            reference_targets,
            reference_motor_thrusts,
            dt=controller.dt,
            tau_up=controller.tau_up,
            tau_down=controller.tau_down,
        ).clamp(*controller.rotor_thrust_limits)
        reference_wrench = reconstruct_wrench(reference_motor_thrusts, controller.allocation_matrix)

        output = controller.compute(command, body_rate)

        torch.testing.assert_close(output.rate_error, reference_error)
        torch.testing.assert_close(output.torque_command, reference_torque)
        torch.testing.assert_close(output.target_rotor_thrusts, reference_targets, atol=1.0e-6, rtol=1.0e-5)
        torch.testing.assert_close(output.rotor_thrusts, reference_motor_thrusts, atol=1.0e-6, rtol=1.0e-5)
        torch.testing.assert_close(output.wrench, reference_wrench, atol=1.0e-6, rtol=1.0e-5)
