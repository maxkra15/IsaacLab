# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral observation, reward and release contracts for cable insertion."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab_tasks.contrib.gb300_sharpa_insertion import mdp
from isaaclab_tasks.contrib.gb300_sharpa_insertion.workcell import SOCKET_POSITION_E, head_goal_for_socket


def _seated_env(hydroelastic_contacts=False):
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

    connector_poses = data_tensor(torch.stack((head, head), dim=1))
    return SimpleNamespace(
        cfg=SimpleNamespace(hydroelastic_contacts=hydroelastic_contacts, target_socket_position_e=SOCKET_POSITION_E),
        num_envs=2,
        tcp_body_id=0,
        head_body_id=1,
        latch_body_id=0,
        scene=_Scene(
            {
                "env_origins": origins,
                "robot": SimpleNamespace(
                    data=SimpleNamespace(root_pose_w=data_tensor(root), body_pose_w=data_tensor(tcp))
                ),
                "cable": SimpleNamespace(
                    data=SimpleNamespace(
                        segment_pose_w=data_tensor(head[:, None]),
                        segment_velocity_w=data_tensor(torch.zeros(2, 1, 6)),
                    )
                ),
                "connector": SimpleNamespace(
                    data=SimpleNamespace(
                        root_pose_w=data_tensor(head.clone()),
                        root_com_ang_vel_w=data_tensor(torch.zeros(2, 3)),
                        body_link_pose_w=connector_poses,
                        body_pose_w=connector_poses,
                        body_com_ang_vel_w=data_tensor(torch.zeros(2, 2, 3)),
                    )
                ),
                "latch": SimpleNamespace(
                    data=SimpleNamespace(
                        root_pose_w=data_tensor(head.clone()),
                        root_com_ang_vel_w=data_tensor(torch.zeros(2, 3)),
                        body_link_pose_w=data_tensor(head[:, None].clone()),
                        body_com_ang_vel_w=data_tensor(torch.zeros(2, 1, 3)),
                    )
                ),
            }
        ),
    )


def test_success_requires_captured_seating_and_measured_hand_clearance():
    """Release is a geometric outcome, not a prescribed hand-joint keyframe."""
    env = _seated_env()
    assert mdp.insertion_success(env).all()
    captured, capture_margins, release_margin, exit_margin = mdp.latch_metrics(env)
    assert captured.all()
    # Independently registered CAD extents at the seated 90-degree connector pose.
    torch.testing.assert_close(
        capture_margins, torch.tensor(((0.000208728, 0.001015795, 0.0008375),) * 2), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(release_margin, torch.full((2,), 0.0008375), atol=1e-6, rtol=0)
    torch.testing.assert_close(exit_margin, torch.full((2,), 0.006058737), atol=1e-6, rtol=0)
    env.scene["robot"].data.body_pose_w.torch[:, 0, 1] += 0.2
    assert not mdp.insertion_success(env).any()
    assert mdp.insertion_success(env, require_release=False).all()
    env.scene["robot"].data.body_pose_w.torch[:, 0, 1] -= 0.2
    # A 30-degree socket-Y tilt clears one barb, but the other still catches.
    sine, cosine = math.sin(math.pi / 12), math.cos(math.pi / 12)
    env.scene["latch"].data.body_link_pose_w.torch[:, 0, 3:] = torch.tensor((cosine, sine, -sine, cosine)) * 2**-0.5
    captured, capture_margins, release_margin, _ = mdp.latch_metrics(env)
    assert not captured.any()
    torch.testing.assert_close(capture_margins[:, 2], torch.full((2,), -0.000895853246), atol=1e-6, rtol=0)
    torch.testing.assert_close(release_margin, torch.full((2,), 0.001446690754), atol=1e-6, rtol=0)
    env.scene["latch"].data.body_link_pose_w.torch[:, 0, 2] -= 0.02
    assert not mdp.insertion_success(env).any()
    captured, _, release_margin, _ = mdp.latch_metrics(env)
    assert not captured.any()
    assert (release_margin < 0).all()


@pytest.mark.parametrize("hydroelastic_contacts", (True, False))
def test_connector_observations_follow_physical_bodies_with_stale_root_coordinates(hydroelastic_contacts):
    """Maximal-coordinate motion must change rewards and relative clip state."""
    env = _seated_env(hydroelastic_contacts)
    torch.testing.assert_close(mdp.head_goal_position(env), torch.ones(2))
    torch.testing.assert_close(mdp.head_goal_orientation(env), torch.ones(2))
    torch.testing.assert_close(mdp.latch_state(env), torch.zeros(2, 2))
    assert mdp.insertion_success(env).all()
    if hydroelastic_contacts:
        connector = env.scene["connector"].data
        head = connector.body_link_pose_w.torch[:, env.head_body_id]
        head_velocity = connector.body_com_ang_vel_w.torch[:, env.head_body_id]
        latch = connector.body_link_pose_w.torch[:, env.latch_body_id]
        latch_velocity = connector.body_com_ang_vel_w.torch[:, env.latch_body_id]
    else:
        cable = env.scene["cable"].data
        head = cable.segment_pose_w.torch[:, 0]
        head_velocity = cable.segment_velocity_w.torch[:, 0, 3:]
        latch = env.scene["latch"].data.body_link_pose_w.torch[:, 0]
        latch_velocity = env.scene["latch"].data.body_com_ang_vel_w.torch[:, 0]

    # Leave generalized root poses/velocities at reset while the physical bodies move.
    head[:, 0] += 0.04
    head[:, 3:] = 0.5  # A 90-degree world-Z turn after the initial 90-degree X turn.
    latch[:, :3] = head[:, :3]
    sine, cosine = math.sin(0.06), math.cos(0.06)
    latch[:, 3:] = head.new_tensor((cosine + sine, cosine + sine, cosine - sine, cosine - sine)) / 2
    head_velocity[:] = head.new_tensor((0.2, 0.3, 0.4))
    latch_velocity[:] = head.new_tensor((0.2, -0.4, 0.4))

    torch.testing.assert_close(mdp.head_goal_position(env), torch.full((2,), 1.0 - math.tanh(0.04 / 0.05)))
    # The clip is compressed 0.12 rad and opens at 0.7 rad/s about world -Y.
    torch.testing.assert_close(mdp.latch_state(env), torch.tensor(((-0.12, 0.7), (-0.12, 0.7))))
    assert not mdp.insertion_success(env, require_release=False).any()
    head[0, 0] = float("nan")
    assert mdp.head_goal_position(env)[0] == 0.0


class _Scene(dict):
    @property
    def env_origins(self):
        return self["env_origins"]


@pytest.mark.parametrize("measure_hand", (False, True))
def test_cycle_requires_measured_hand_opening_before_regrasp(measure_hand):
    """Opening targets alone must not establish release beside a seated connector."""
    from isaaclab_tasks.contrib.gb300_sharpa_insertion.robot_cfg import CLOSE_HAND_RAD

    from examples.gb300_sharpa_insertion import CableCycleStateMachine

    goal = np.array((0.5, 0.3, 2.1, 0.0, 0.0, 0.0, 1.0))
    machine = CableCycleStateMachine(goal, goal, 1 / 60, CLOSE_HAND_RAD)
    machine.phase = "INSERTION/RELEASE"
    for _ in range(1000):
        machine.update(
            goal,
            goal,
            latch_engaged=True,
            latch_angle_rad=0.0,
            capture_margins_m=np.array((0.0002, 0.001, 0.0008)),
            keeper_release_margin_m=0.0008,
            keeper_exit_margin_m=0.006,
            hand_joint_pos=np.asarray(CLOSE_HAND_RAD) if measure_hand else None,
        )
        if machine.phase == "FAILED":
            break
    assert not machine.insertion_completed, "Targets without measured release must not start extraction."
    assert machine.phase == "FAILED"
    assert machine.failure == "RELEASE timed out"
