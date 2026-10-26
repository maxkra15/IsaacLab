# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical command bounds and full-task reward/proxy contracts, without assets."""

import math

import pytest
import torch

from examples.mpm_world_model.teapot_full_task import (
    INITIAL_ROOT_Z,
    PALM_IN_POT_POSITION,
    FullJointController,
    full_phase_metrics,
    full_task_reward,
    full_task_success,
)


def controller(dt=0.01):
    return FullJointController(
        torch.linspace(-0.2, 0.4, 29), torch.full((29,), -1.0), torch.full((29,), 2.0), torch.ones(29), dt=dt
    )


def observation(height=0.0):
    state = torch.zeros(182)
    state[60] = INITIAL_ROOT_Z + height
    state[[61, 74, 169]] = 1.0  # All three poses use WXYZ.
    state[166:169] = state[58:61] + torch.tensor(PALM_IN_POT_POSITION)
    state[91] = 1.0
    state[165] = 24.0
    state[180] = 1.0
    state[181] = 0.5
    return state


def test_joint_commands_hold_reset_and_respect_physical_rates():
    control = controller()
    for step in range(20):
        old_gain = control.stiffness.clone()
        control.apply_action(torch.zeros(30), step * 0.01)
        torch.testing.assert_close(control.current, control.initial, atol=0, rtol=0)
        assert abs(float(control.stiffness - old_gain)) <= 5.0
    assert control.stiffness == 62.0
    control.apply_action(torch.ones(30), 0.2)
    torch.testing.assert_close(control.current - control.initial, torch.full((29,), 0.01))
    assert control.state().shape == (60,)
    control.reset()
    torch.testing.assert_close(control.current, control.initial, atol=0, rtol=0)
    assert control.stiffness == 24.0


def test_all_thirty_channels_reach_and_encode_asymmetric_limits():
    control = controller(dt=1.0)
    for target, gain in ((control.upper, 100.0), (control.lower, 24.0)):
        action = control.encode_command(target, gain)
        for _ in range(4):
            control.apply_action(action, 0.0)
        torch.testing.assert_close(control.current, target)
        assert control.stiffness == gain
    with pytest.raises(ValueError, match="finite"):
        control.apply_action(torch.full((30,), float("nan")), 0.0)
    with pytest.raises(ValueError, match="positive"):
        controller(dt=0.0)


def test_grasp_reference_uses_actual_pot_rotation_in_wxyz_order():
    state = observation(height=0.07)
    state[61:65] = torch.tensor([math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
    x, y, z = PALM_IN_POT_POSITION
    state[166:169] = state[58:61] + torch.tensor([-y, x, z])
    metrics = full_phase_metrics(state)
    assert metrics["palm_grasp_distance_m"] < 1e-6
    assert metrics["lifted_grasp_proxy"]
    state[166] += 0.2
    assert not full_phase_metrics(state)["lifted_grasp_proxy"]


def test_delivery_needs_lift_and_gravity_spill_never_earns_delivery_credit():
    before = observation()
    spilled = before.clone()
    spilled[91], spilled[93] = 0.5, 0.5
    assert full_task_reward(before, spilled, torch.zeros(30)) == -100.0
    delivered_on_table = before.clone()
    delivered_on_table[91], delivered_on_table[92] = 0.4, 0.6
    assert full_task_reward(before, delivered_on_table, torch.zeros(30)) == 0.0
    assert not full_task_success(delivered_on_table)
    lifted_before = observation(height=0.07)
    lifted_after = lifted_before.clone()
    lifted_after[91], lifted_after[92] = 0.4, 0.6
    torch.testing.assert_close(full_task_reward(lifted_before, lifted_after, torch.zeros(30)), torch.tensor(60.0))
    assert full_task_success(lifted_after)
    approach_start = before.clone()
    approach_start[166] += 0.2
    assert full_task_reward(approach_start, before, torch.zeros(30)) > 0
    assert not full_task_success(before)


def test_batched_reward_rejects_nonfinite_state_without_nan_loss():
    before = torch.stack((observation(height=0.07), observation(height=0.07)))
    after = before.clone()
    after[:, 91], after[:, 92] = 0.4, 0.6
    after[1, 60] = float("inf")
    reward = full_task_reward(before, after, torch.zeros((2, 30)))
    torch.testing.assert_close(reward, torch.tensor([60.0, -10.0]))
    assert full_task_success(after).tolist() == [True, False]
    action = torch.zeros(30)
    action[0] = float("nan")
    assert full_task_reward(before[0], after[0], action) == -10.0
    invented_mass = after[0].clone()
    invented_mass[95] = -0.2
    assert not full_task_success(invented_mass)
    assert full_task_reward(before[0], invented_mass, torch.zeros(30)) == -10.0
