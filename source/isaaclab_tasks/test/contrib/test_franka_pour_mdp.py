# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Franka pour reward terms (no simulator)."""

import torch

from isaaclab_tasks.contrib.franka_pour.mdp import rewards, terminations


class FakeActionManager:
    def __init__(self, num_envs: int):
        self.action = torch.zeros((num_envs, 7))


class FakeEnv:
    """Minimal vectorized stand-in exposing the interface consumed by pure reward terms."""

    def __init__(self):
        self.num_envs = 4
        self.num_particles = 1000
        self.pour_target_frac = 0.9
        self.gripper_open_width = 0.08
        self.gripper_grasp_width = 0.055
        self.cup_reset_height = 0.0
        self.action_manager = FakeActionManager(self.num_envs)

        self._tcp = torch.tensor([[0.50, 0.00, 0.032], [0.50, 0.00, 0.032], [0.50, 0.00, 0.032], [0.20, 0.00, 0.032]])
        self._grasp = torch.tensor([[0.50, 0.00, 0.032]]).repeat(self.num_envs, 1)
        self._cup = torch.tensor(
            [
                [0.50, 0.00, 0.00, 0.0, 0.0, 0.0, 1.0],
                [0.50, 0.00, 0.12, 0.0, 0.0, 0.0, 1.0],
                [0.50, -0.17, 0.12, 0.7071068, 0.0, 0.0, 0.7071068],
                [0.20, 0.00, 0.00, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        self._target = torch.tensor([[0.50, -0.18, 0.00, 0.0, 0.0, 0.0, 1.0]]).repeat(self.num_envs, 1)
        self._width = torch.tensor([0.08, 0.055, 0.055, 0.055])
        self._src = torch.tensor([1000.0, 750.0, 400.0, 0.0])
        self._tgt = torch.tensor([0.0, 250.0, 500.0, 950.0])

    def tcp_pos_e(self):
        return self._tcp

    def cup_grasp_point_e(self):
        return self._grasp

    def cup_pose_e(self):
        return self._cup

    def target_pose_e(self):
        return self._target

    def gripper_width(self):
        return self._width

    def count_in_target(self):
        return self._tgt

    def count_in_source(self):
        return self._src


def test_reach_reward_prefers_tcp_at_grasp_point():
    reward = rewards.reach_cup(FakeEnv(), std=0.10)
    assert reward[0] > 0.99
    assert reward[0] > reward[3]


def test_grasp_reward_requires_nearby_tcp_and_closed_fingers():
    reward = rewards.grasp_cup(FakeEnv(), reach_std=0.10)
    assert reward[1] > reward[0]  # near + closed beats near + open
    assert reward[1] > reward[3]  # near + closed beats far + closed


def test_lift_and_alignment_are_stage_gated():
    env = FakeEnv()
    lift = rewards.lift_cup(env, target_height=0.12, reach_std=0.10)
    align = rewards.align_cup_over_target(env, lift_height=0.06, std=0.12)
    assert lift[1] > lift[0]
    assert align[2] > align[1]  # same lift, but only env 2 is over target
    assert align[0] == 0.0  # alignment is not rewarded while the cup remains on the table


def test_tilt_reward_requires_lift_and_alignment():
    reward = rewards.tilt_over_target(FakeEnv(), lift_height=0.06, align_std=0.12)
    assert reward[2] > 0.8  # lifted, aligned, and rotated 90 degrees
    assert reward[1] < 0.1  # lifted but far from target and upright
    assert reward[0] == 0.0  # still on table


def test_particle_fractions_spill_and_success():
    env = FakeEnv()
    assert torch.allclose(rewards.particles_in_target(env), torch.tensor([0.0, 0.25, 0.5, 0.95]))
    assert torch.allclose(rewards.particles_in_source(env), torch.tensor([1.0, 0.75, 0.4, 0.0]))
    assert torch.allclose(rewards.spilled_particles(env), torch.tensor([0.0, 0.0, 0.1, 0.05]), atol=1e-6)
    assert rewards.pour_success_bonus(env).tolist() == [0.0, 0.0, 0.0, 1.0]


def test_stage_command_progress_rewards_only_actions_that_advance_the_task():
    env = FakeEnv()
    env._width[:] = 0.055

    # Two identical grasped, table-height states: only positive world-z motion advances lift.
    env._tcp[0] = env._grasp[0]
    env._tcp[1] = env._grasp[1]
    env._cup[0] = env._cup[0]
    env._cup[1] = env._cup[0]
    env.action_manager.action[0, 2] = 0.5
    env.action_manager.action[1, 2] = -0.5
    lift_progress = rewards.lift_command_progress(env)
    assert lift_progress[0] > 0.0
    assert lift_progress[1] == 0.0

    # Once lifted, motion toward target (-y) is useful and motion away is not.
    env._cup[0, :3] = torch.tensor([0.50, 0.00, 0.12])
    env._cup[1, :3] = torch.tensor([0.50, 0.00, 0.12])
    env.action_manager.action[0, :2] = torch.tensor([0.0, -0.5])
    env.action_manager.action[1, :2] = torch.tensor([0.0, 0.5])
    align_progress = rewards.align_command_progress(env)
    assert align_progress[0] > 0.0
    assert align_progress[1] == 0.0

    # At the receiver, positive x-axis wrist rotation advances the demonstrated pour.
    env._cup[2] = torch.tensor([0.50, -0.18, 0.12, 0.0, 0.0, 0.0, 1.0])
    env.action_manager.action[2, 3] = 0.5
    assert rewards.tilt_command_progress(env)[2] > 0.0


def test_state_finite_rejects_raw_nonfinite_cup_positions_and_quaternions():
    robot_joint_pos = torch.zeros((3, 7))
    cup_body_q = torch.tensor(
        [
            [0.5, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
            [float("nan"), 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
            [0.5, 0.0, 0.1, 0.0, float("inf"), 0.0, 1.0],
        ]
    )
    particle_pos = torch.zeros((3, 16, 3))

    finite = terminations._state_finite(robot_joint_pos, cup_body_q, particle_pos)

    assert finite.tolist() == [True, False, False]
