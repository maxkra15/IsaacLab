# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ownership of terminal RGB observations across automatic episode resets."""

import importlib.util
import unittest
from pathlib import Path

import torch

SPEC = importlib.util.spec_from_file_location("simulator_tasks", Path(__file__).parents[1] / "tasks.py")
tasks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tasks)


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
