# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frame-clock and auto-reset metadata contracts for illustrative native replay."""

from types import SimpleNamespace

import torch

from examples.mpm_world_model.capture_franka_pour import CaptureTimeline


def test_capture_timeline_keeps_reset_boundary_and_physical_clock():
    """A frame after autoreset names the new episode and retains the terminal event."""
    timeline = CaptureTimeline(warmup_steps=16)
    env = SimpleNamespace(
        num_envs=1,
        reset_dataset_row_id=torch.tensor([7]),
        episode_length_buf=torch.tensor([0]),
        termination_manager=SimpleNamespace(active_terms=["success"], get_term=lambda name: torch.tensor([True])),
    )
    received = []

    def native_reset(env, selection):
        received.append(selection)
        env.reset_dataset_row_id[selection] += 10
        env.episode_length_buf[selection] = 0
        return "reset_result"

    all_worlds = slice(None)
    assert timeline.observe_reset(env, all_worlds, native_reset) == "reset_result"
    assert received[-1] is all_worlds
    assert timeline.resets[-1]["initialization_reset"]
    timeline.policy_step = 16
    timeline.frame(17, 16, 1 / 30, [720, 1280, 3])
    timeline.policy_step = 17
    env.episode_length_buf[0] = 17
    terminated_worlds = torch.tensor([0])
    timeline.observe_reset(env, terminated_worlds, native_reset)
    assert received[-1] is terminated_worlds
    timeline.frame(27, 0, 1 / 30, [720, 1280, 3])
    timeline.observe_reset(env, torch.empty(0, dtype=torch.long), native_reset)
    timeline.observe_reset(env, slice(0, 0), native_reset)
    assert len(timeline.resets) == 2
    timeline.policy_step = 376
    timeline.frame(27, 359, 1 / 30, [720, 1280, 3])
    first, reset, last = timeline.frames
    assert first["warmup"] and not reset["warmup"]
    assert reset["episode_id"] == 1 and reset["reset_row"] == 27 and reset["episode_step"] == 0
    event = timeline.resets[reset["resets_this_step"][0]]
    assert (event["old_reset_row"], event["new_reset_row"]) == (17, 27)
    assert event["termination_reasons"] == ["success"]
    assert event["completed_episode_length_policy_steps"] == 17
    assert last["simulated_time_s"] == 376 / 30
    assert last["frame_index"] == 2
