# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral observation, reward and release contracts for cable insertion."""

import math
from types import SimpleNamespace

import torch

from isaaclab_tasks.contrib.gb300_sharpa_insertion import mdp
from isaaclab_tasks.contrib.gb300_sharpa_insertion.workcell import SOCKET_POSITION_E, head_goal_for_socket
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg


def _seated_env():
    origins = torch.tensor(((0.0, 0.0, 0.0), (8.0, -3.0, 0.0)))
    rotation = torch.tensor((2**-0.5, 0.0, 0.0, 2**-0.5))
    head = torch.cat((origins + origins.new_tensor(head_goal_for_socket(SOCKET_POSITION_E)), rotation.repeat(2, 1)), 1)
    # Equivalent quaternion signs and distinct base frames must not change success.
    head[1, 3:] *= -1
    root = torch.tensor(((0.1, -0.5, 1.0, 0.0, 0.0, 0.0, 1.0), (8.2, -3.4, 1.2, 0.0, 0.0, 2**-0.5, 2**-0.5)))
    tcp = head[:, None].clone()
    tcp[:, 0, 1] -= 0.25

    def data_tensor(value):
        return SimpleNamespace(torch=value)

    return SimpleNamespace(
        cfg=SimpleNamespace(hydroelastic_contacts=False, target_socket_position_e=SOCKET_POSITION_E),
        num_envs=2,
        tcp_body_id=0,
        scene=_Scene(
            {
                "env_origins": origins,
                "robot": SimpleNamespace(
                    data=SimpleNamespace(root_pose_w=data_tensor(root), body_pose_w=data_tensor(tcp))
                ),
                "cable": SimpleNamespace(data=SimpleNamespace(segment_pose_w=data_tensor(head[:, None]))),
                "latch": SimpleNamespace(
                    data=SimpleNamespace(
                        root_pose_w=data_tensor(head.clone()), root_com_ang_vel_w=data_tensor(torch.zeros(2, 3))
                    )
                ),
            }
        ),
    )


def test_seating_reward_is_frame_invariant_and_measures_the_connector():
    """Equal seating is rewarded in independent frames; moving the plug lowers reward."""
    env = _seated_env()
    torch.testing.assert_close(mdp.head_goal_position(env), torch.ones(2))
    torch.testing.assert_close(mdp.head_goal_orientation(env), torch.ones(2))
    env.scene["cable"].data.segment_pose_w.torch[:, 0, 0] += 0.04
    torch.testing.assert_close(mdp.head_goal_position(env), torch.full((2,), 1.0 - math.tanh(0.04 / 0.05)))
    env.scene["cable"].data.segment_pose_w.torch[0, 0, 0] = float("nan")
    assert mdp.head_goal_position(env)[0] == 0.0


def test_success_requires_captured_seating_and_measured_hand_clearance():
    """Release is a geometric outcome, not a prescribed hand-joint keyframe."""
    env = _seated_env()
    assert mdp.insertion_success(env).all()
    env.scene["robot"].data.body_pose_w.torch[:, 0, 1] += 0.2
    assert not mdp.insertion_success(env).any()
    assert mdp.insertion_success(env, require_release=False).all()
    env.scene["robot"].data.body_pose_w.torch[:, 0, 1] -= 0.2
    env.scene["latch"].data.root_pose_w.torch[:, 2] -= 0.02
    assert not mdp.insertion_success(env).any()


class _Scene(dict):
    @property
    def env_origins(self):
        return self["env_origins"]


def test_registered_presets_pair_normalized_and_absolute_action_contracts():
    """PPO defaults are bounded; scripted poses are not clipped as normalized actions."""
    task = "IsaacContrib-Rizon-Sharpa-GB300-Cable-Install"
    normalized = parse_env_cfg(task)
    absolute = parse_env_cfg(task, overrides=["presets=scripted"])
    runner = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    normalized.validate()
    absolute.validate()
    assert normalized.actions.right_palm.objectives[0].use_relative_mode
    assert normalized.actions.right_palm.clip == {".*": (-1.0, 1.0)}
    assert normalized.actions.right_hand.use_default_offset
    assert len(normalized.actions.right_hand.joint_names) == 22
    assert runner.default.clip_actions == 1.0
    assert not absolute.actions.right_palm.objectives[0].use_relative_mode
    assert absolute.actions.right_palm.clip is None
    assert not absolute.actions.right_hand.use_default_offset
    assert runner.scripted.clip_actions is None
