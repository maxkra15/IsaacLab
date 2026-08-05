# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim-free tests for replaying legacy Waterhose initial state."""

from types import SimpleNamespace

import torch

from isaaclab.envs import ManagerBasedRLMimicEnv

from isaaclab_tasks.contrib.waterhose.waterhose_mimic_env import (
    WaterhoseMimicEnv,
    normalize_waterhose_mimic_initial_state,
)


def _legacy_initial_state() -> dict:
    return {
        "articulation": {
            "robot": {"joint_position": torch.tensor([[1.0]])},
            "cable1": {
                "root_pose": torch.zeros(1, 7),
                "joint_position": torch.zeros(1, 4),
            },
        },
        "rigid_object": {"prop": {"root_pose": torch.ones(1, 7)}},
    }


def _native_cable_defaults() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    poses = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                [1.5, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            ],
            [
                [11.0, 5.0, 7.0, 0.0, 0.0, 0.0, 1.0],
                [11.5, 5.0, 7.0, 0.0, 0.0, 0.0, 1.0],
            ],
        ]
    )
    velocities = torch.arange(2 * 2 * 6, dtype=torch.float32).reshape(2, 2, 6)
    env_origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 4.0, 6.0]])
    return poses, velocities, env_origins


def test_legacy_initial_state_uses_selected_native_cable_default_without_mutation():
    """Old articulation cable state becomes the selected env's relative native default."""
    state = _legacy_initial_state()
    poses, velocities, env_origins = _native_cable_defaults()

    normalized, used_default = normalize_waterhose_mimic_initial_state(
        state,
        default_segment_pose_w=poses,
        default_segment_velocity_w=velocities,
        env_origins=env_origins,
        env_ids=torch.tensor([1]),
        is_relative=True,
    )

    assert used_default
    assert "cable1" in state["articulation"]
    assert "cable_object" not in state
    assert normalized["articulation"]["robot"] is state["articulation"]["robot"]
    assert "cable1" not in normalized["articulation"]
    expected_pose = poses[1:2].clone()
    expected_pose[..., :3] -= env_origins[1:2, None, :]
    torch.testing.assert_close(normalized["cable_object"]["cable1"]["segment_pose"], expected_pose)
    torch.testing.assert_close(normalized["cable_object"]["cable1"]["segment_velocity"], velocities[1:2])


def test_transitional_initial_state_preserves_native_cable_state():
    """If both formats are present, native segment state wins over the legacy entry."""
    state = _legacy_initial_state()
    native_cable_state = {
        "segment_pose": torch.full((1, 2, 7), 42.0),
        "segment_velocity": torch.full((1, 2, 6), -3.0),
    }
    state["cable_object"] = {"cable1": native_cable_state}
    poses, velocities, env_origins = _native_cable_defaults()

    normalized, used_default = normalize_waterhose_mimic_initial_state(
        state,
        default_segment_pose_w=poses,
        default_segment_velocity_w=velocities,
        env_origins=env_origins,
    )

    assert not used_default
    assert normalized["cable_object"]["cable1"] is native_cable_state
    assert "cable1" not in normalized["articulation"]


def test_current_native_initial_state_is_returned_unchanged():
    """Current recordings bypass the compatibility path entirely."""
    poses, velocities, env_origins = _native_cable_defaults()
    state = {
        "articulation": {"robot": {"joint_position": torch.tensor([[1.0]])}},
        "cable_object": {
            "cable1": {
                "segment_pose": poses[0:1].clone(),
                "segment_velocity": velocities[0:1].clone(),
            }
        },
    }

    normalized, used_default = normalize_waterhose_mimic_initial_state(
        state,
        default_segment_pose_w=poses,
        default_segment_velocity_w=velocities,
        env_origins=env_origins,
    )

    assert normalized is state
    assert not used_default


def test_waterhose_reset_forwards_normalized_legacy_state(monkeypatch):
    """The task-level reset shim reaches the generic reset with native cable state."""
    state = _legacy_initial_state()
    poses, velocities, env_origins = _native_cable_defaults()
    forwarded = {}

    def fake_reset_to(self, state, env_ids, seed=None, is_relative=False):
        forwarded.update(state=state, env_ids=env_ids, seed=seed, is_relative=is_relative)
        return "reset-result"

    monkeypatch.setattr(ManagerBasedRLMimicEnv, "reset_to", fake_reset_to)
    env = WaterhoseMimicEnv.__new__(WaterhoseMimicEnv)
    env._is_closed = True
    cable = SimpleNamespace(
        data=SimpleNamespace(
            default_segment_pose_w=SimpleNamespace(torch=poses),
            default_segment_velocity_w=SimpleNamespace(torch=velocities),
        )
    )

    class FakeScene:
        def __init__(self):
            self.env_origins = env_origins

        def __getitem__(self, name):
            assert name == "cable1"
            return cable

    env.scene = FakeScene()

    result = env.reset_to(state, [0], seed=7, is_relative=True)

    assert result == "reset-result"
    assert forwarded["seed"] == 7
    assert forwarded["is_relative"] is True
    assert "cable1" not in forwarded["state"]["articulation"]
    assert "cable1" in forwarded["state"]["cable_object"]
