# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared task boundaries: terminal RGB ownership and deterministic reaching goals."""

import importlib.util
import math
import unittest
from pathlib import Path

import torch

SPEC = importlib.util.spec_from_file_location("simulator_tasks", Path(__file__).parents[1] / "tasks.py")
tasks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tasks)
PPO_SPEC = importlib.util.spec_from_file_location("simulator_ppo", Path(__file__).parents[1] / "ppo.py")
ppo = importlib.util.module_from_spec(PPO_SPEC)
PPO_SPEC.loader.exec_module(ppo)


def franka_config() -> dict:
    """Provide the pinned Panda chain for CPU tests independent of simulator importers."""
    origins = [
        ([0.0, 0.0, 0.333], 0.0),
        ([0.0, 0.0, 0.0], -math.pi / 2),
        ([0.0, -0.316, 0.0], math.pi / 2),
        ([0.0825, 0.0, 0.0], math.pi / 2),
        ([-0.0825, 0.384, 0.0], -math.pi / 2),
        ([0.0, 0.0, 0.0], math.pi / 2),
        ([0.088, 0.0, 0.0], math.pi / 2),
    ]
    return {
        "task": "franka",
        "device": "cpu",
        "num_envs": 3,
        "seed": 17,
        "dt": 0.005,
        "decimation": 4,
        "vision": "none",
        "initial_joint_pos": [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, math.pi / 4],
        "joint_lower_limits": [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671],
        "joint_upper_limits": [2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671],
        "end_effector_offset": [0.0, 0.0, 0.212],
        "kinematic_chain": [
            {"origin_xyz": xyz, "origin_rpy": [roll, 0.0, 0.0], "axis": [0.0, 0.0, 1.0]} for xyz, roll in origins
        ],
    }


class PositionAdapter:
    """Store task inputs and expose controlled state without integrating dynamics."""

    joint_names = [f"panda_joint{index}" for index in range(1, 8)]

    def __init__(self):
        self.joints = torch.zeros((3, 7))
        self.ee = torch.tensor([[0.3, 0.0, 0.5]]).expand(3, -1).clone()
        self.action = None

    def reset(self, env_ids, joint_pos):
        self.joints[env_ids] = joint_pos

    def state(self):
        return {
            "joint_pos": self.joints,
            "joint_vel": torch.zeros_like(self.joints),
            "root_pos": torch.zeros((3, 3)),
            "ee_pos": self.ee,
        }

    def apply_action(self, action):
        self.action = action.clone()

    def physics_step(self):
        pass


class TestFrankaReaching(unittest.TestCase):
    """Protect target coordinates, partial resets and the seven-action reaching contract."""

    def test_fk_matches_straight_arm_geometry_and_world_base_rotation(self):
        task = tasks.Task(PositionAdapter(), franka_config())
        angles = torch.zeros((2, 7))
        angles[1, 0] = math.pi / 2
        # At zero joint angles the opposed 82.5 mm offsets cancel, and
        # the downward grasp offset subtracts from 333+316+384 mm height.
        expected = torch.tensor([[0.088, 0.0, 0.821], [0.0, 0.088, 0.821]])
        torch.testing.assert_close(task.forward_kinematics(angles), expected, atol=1e-6, rtol=0)

    def test_seeded_goals_partial_reset_action_bounds_and_distance_reward(self):
        adapter = PositionAdapter()
        task = tasks.Task(adapter, franka_config())
        replay = tasks.Task(PositionAdapter(), franka_config())
        task.reset()
        replay.reset()
        torch.testing.assert_close(task.target_pos, replay.target_pos, rtol=0, atol=0)
        self.assertGreater((task.target_pos[1:] - task.target_pos[0]).abs().max().item(), 0.01)
        untouched_target = task.target_pos[[0, 2]].clone()
        untouched_joints = adapter.joints[[0, 2]].clone()
        task.reset(torch.tensor([1]))
        torch.testing.assert_close(task.target_pos[[0, 2]], untouched_target, rtol=0, atol=0)
        torch.testing.assert_close(adapter.joints[[0, 2]], untouched_joints, rtol=0, atol=0)
        # Different physical EE distances must change both the reported metric
        # and reward, while preserving a valid seven-joint action command.
        adapter.ee.copy_(task.target_pos)
        adapter.ee[:, 0] += torch.tensor([0.0, 0.04, 0.2])
        _, reward, done, info = task.step(torch.full((3, 7), 10.0))
        torch.testing.assert_close(info["reaching_error_m"], torch.tensor([0.0, 0.04, 0.2]), atol=1e-6, rtol=0)
        self.assertTrue(bool((reward[:-1] > reward[1:]).all()))
        self.assertFalse(bool(done.any()))
        self.assertTrue(bool((adapter.action >= task.joint_lower).all()))
        self.assertTrue(bool((adapter.action <= task.joint_upper).all()))
        torch.testing.assert_close(adapter.action, (task.nominal + 0.75).expand(3, -1))
        self.assertEqual(task.error_key, "reaching_error_m")
        self.assertEqual(task.error_threshold, 0.05)

    def test_evaluation_reports_reaching_error_and_restores_training_random_stream(self):
        adapter = PositionAdapter()
        task = tasks.Task(adapter, franka_config())
        old_random_state = task.generator.get_state().clone()
        reference = tasks.Task(PositionAdapter(), franka_config())
        reference.generator.manual_seed(1007)
        reference.reset()
        expected_error = (reference.target_pos - adapter.ee).norm(dim=-1).mean().item()

        class ZeroPolicy:
            def __call__(self, obs, images):
                return torch.distributions.Normal(torch.zeros((len(obs), 7)), torch.ones((len(obs), 7))), None

        result = ppo.evaluate(task, ZeroPolicy(), steps=2, seed=1007)
        self.assertAlmostEqual(result["reaching_error_m"], expected_error, places=6)
        self.assertEqual(result["threshold_reached"], expected_error <= 0.05)
        self.assertEqual(result["survival_fraction"], 1.0)
        self.assertNotIn("pole_angle_abs_rad", result)
        self.assertTrue(torch.equal(task.generator.get_state(), old_random_state))


class TestTerminalImages(unittest.TestCase):
    """Protect the RGB timeout handoff consumed by PPO bootstrapping."""

    def test_timeout_images_survive_reused_camera_buffer_and_partial_reset(self):
        class BufferAdapter:
            """Expose stored coordinates and a camera buffer overwritten on each render."""

            joint_names = ["slider", "pole"]

            def __init__(self):
                self.joints = torch.zeros((3, 2))
                self.velocities = torch.zeros_like(self.joints)
                self.pixels = torch.zeros((3, 1, 1, 3), dtype=torch.uint8)
                self.buffer = torch.zeros_like(self.pixels)

            def reset(self, env_ids, joint_pos):
                self.joints[env_ids] = joint_pos
                self.pixels[env_ids] = 0

            def state(self):
                return {"joint_pos": self.joints, "joint_vel": self.velocities, "root_pos": torch.zeros((3, 3))}

            def render(self):
                self.buffer.copy_(self.pixels)
                return self.buffer

            def apply_action(self, action):
                pass

            def physics_step(self):
                pass

        adapter = BufferAdapter()
        task = tasks.Task(
            adapter,
            {
                "device": "cpu",
                "num_envs": 3,
                "task": "cartpole",
                "seed": 7,
                "dt": 0.5,
                "decimation": 5,
                "vision": "rgb",
            },
        )
        action = torch.zeros((3, 1))
        task.reset()
        task.step(action)
        # Stagger episode ages through the public reset interface, so only
        # environments 0 and 2 reach the next time limit.
        task.reset(torch.tensor([1]))
        adapter.joints[:, 0] = 0.75
        adapter.joints[:, 1] = 0.0
        adapter.pixels.copy_(torch.tensor([[[[11, 12, 13]]], [[[21, 22, 23]]], [[[31, 32, 33]]]], dtype=torch.uint8))

        observation, _, done, info = task.step(action)

        self.assertTrue(torch.equal(done, torch.tensor([True, False, True])))
        self.assertTrue(torch.equal(info["timeout"], torch.tensor([True, False, True])))
        expected = torch.tensor([[[[11, 12, 13]]], [[[31, 32, 33]]]], dtype=torch.uint8)
        self.assertTrue(torch.equal(info["terminal_images"], expected))
        self.assertTrue(torch.equal(info["terminal_observation"][[0, 2], 0], torch.tensor([0.75, 0.75])))
        self.assertTrue(bool((observation[[0, 2], 0].abs() <= 0.1).all()))
        self.assertEqual(observation[1, 0].item(), 0.75)
        self.assertTrue(bool((task.images[[0, 2]] == 0).all()))
        adapter.buffer.fill_(99)
        self.assertTrue(torch.equal(info["terminal_images"], expected))


if __name__ == "__main__":
    unittest.main()
