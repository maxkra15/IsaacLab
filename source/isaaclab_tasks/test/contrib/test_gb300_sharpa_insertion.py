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
from scipy.spatial.transform import Rotation

from isaaclab.assets import VisualMaterial
from isaaclab.managers import EventManager

from isaaclab_tasks.contrib.gb300_sharpa_insertion import mdp
from isaaclab_tasks.contrib.gb300_sharpa_insertion.workcell import SOCKET_POSITION_E, head_goal_for_socket
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg


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
    assert not normalized.sim.physics.capture_implicit_actuators
    assert absolute.sim.physics.capture_implicit_actuators
    for digit in ("right_ring_", "right_pinky_"):
        assert digit in normalized.grasp_body_patterns
        assert digit not in absolute.grasp_body_patterns
    for name in ("right_thumb", "right_fingers"):
        # Coarse scripted stepping must not retune the normalized training preset.
        assert normalized.scene.robot.actuators[name].joint_effort_limit == 3.3
        assert normalized.scene.robot.actuators[name].armature is None
        assert absolute.scene.robot.actuators[name].joint_effort_limit is None
        assert absolute.scene.robot.actuators[name].armature > 0.0


def test_switchboard_activity_runs_through_the_event_manager(monkeypatch):
    """Port activity follows measured capture through clip release, withdrawal and subset resets."""
    writes = []

    def record_channels(materials, channels, env_ids):
        writes.append((env_ids.clone(), channels["color"].clone(), channels["emissive_color"].clone()))

    monkeypatch.setattr(VisualMaterial, "write_channels", record_channels)
    env = _seated_env(True)
    env.device = "cpu"
    env.sim = SimpleNamespace(is_playing=lambda: True)
    env.common_step_counter = 6
    env.step_dt = 1.0 / 60.0
    env.scene.update(activity_a=object(), activity_b=object(), activity_target=object())
    latch = env.scene["connector"].data.body_link_pose_w.torch[:, env.latch_body_id]
    resting = latch.clone()
    # Independent CAD hinge and rotation: depress the clip without translating the head.
    hinge = np.array((0.0, 0.004677868448, -0.030711488256))
    head_rotation = Rotation.from_euler("x", np.pi / 2)
    compression = Rotation.from_euler("x", 0.225)
    pressed = resting.clone()
    pressed[:, :3] += torch.tensor(head_rotation.apply(hinge - compression.apply(hinge)), dtype=pressed.dtype)
    pressed[:, 3:] = torch.tensor((head_rotation * compression).as_quat(), dtype=pressed.dtype)
    latch[:] = pressed
    assert not mdp.latch_captured(env).any()
    cfg = parse_env_cfg("IsaacContrib-Rizon-Sharpa-GB300-Cable-Install", overrides=["presets=scripted"])
    manager = EventManager({"activity": cfg.events.activity}, env)
    manager.apply("interval", dt=0.1)
    assert writes[-1][1].shape == writes[-1][2].shape == (3, 2, 3)
    assert not writes[-1][2][2].any(), "A seated but uncaptured plug must not activate the port."

    latch[0] = resting[0]
    manager.apply("interval", dt=0.1)
    assert writes[-1][2][2, 0].any() and not writes[-1][2][2, 1].any()
    env.common_step_counter = 24
    manager.apply("interval", dt=0.1)
    assert not torch.equal(writes[-1][2][:2], writes[-2][2][:2]), "Installed-port activity still blinks."
    assert not torch.equal(writes[-1][2][2, 0], writes[-2][2][2, 0]), "The latched target must blink too."

    latch[0] = pressed[0]
    manager.apply("interval", dt=0.1)
    assert writes[-1][2][2, 0].any(), "Pressing the clip alone must not turn off a seated link."
    env.scene["connector"].data.body_link_pose_w.torch[0, :, 1] -= 0.01
    manager.apply("interval", dt=0.1)
    assert not writes[-1][2][2, 0].any(), "Withdrawing the plug must extinguish the target light."

    latch[1] = resting[1]
    manager.apply("interval", dt=0.1)
    assert writes[-1][2][2, 1].any()
    manager.apply("interval", dt=0.05)
    manager.reset(torch.tensor([0]))
    manager.apply("interval", dt=0.05)
    assert writes[-1][0].tolist() == [1]
    assert writes[-1][2][2, 0].any(), "Resetting another environment must preserve this port's link."
    latch[1] = pressed[1]
    manager.reset(torch.tensor([1]))
    manager.apply("interval", dt=0.1)
    assert not writes[-1][2][2].any(), "Reset must require a fresh latch capture."


@pytest.mark.parametrize("validate_contacts", (True, False))
def test_insertion_opens_smoothly_through_retreat_after_measured_release(validate_contacts):
    """Measured seating gates release; optional contact validation preserves smooth opening."""
    from examples.cable_installation.contacts import GraspContacts
    from examples.cable_installation.insertion import CableInstallStateMachine

    goal = np.array((0.5, 0.3, 2.1, 0.0, 0.0, 0.0, 1.0))
    machine = CableInstallStateMachine(goal, goal, 1 / 60)
    grasp = GraspContacts(True, True, 0.0003, False, "")
    released = GraspContacts(False, False, 0.0, False, "")
    tcp = goal.copy()
    head = goal.copy()
    opening = []
    retreat_started = False
    uncaptured_steps = 0
    for _ in range(1200):
        contacts = released if machine.phase in ("RELEASE", "RETREAT") else grasp
        latch_engaged = True
        if machine.phase == "INSERT" and uncaptured_steps < 60:
            latch_engaged = False
            uncaptured_steps += 1
        previous = tcp.copy()
        tcp, width = machine.update(tcp, head, contacts if validate_contacts else None, latch_engaged=latch_engaged)
        if not latch_engaged:
            assert machine.phase == "INSERT", "An unmeasured grasp must not bypass actual clip capture."
        if machine.phase in ("RELEASE", "RETREAT"):
            opening.append(width)
        else:
            head[:3] += tcp[:3] - previous[:3]
        if machine.phase == "RETREAT":
            retreat_started = True
        if machine.phase in ("DONE", "FAILED"):
            break
    assert retreat_started and machine.phase == "DONE", machine.failure
    assert uncaptured_steps == 60
    increments = np.diff((0.0, *opening))
    assert np.all(increments >= -1e-12)
    assert increments.max() <= 0.01 * machine.dt + 1e-12
    assert opening[-1] == pytest.approx(0.04)


def test_cycle_regrasps_without_retreat_or_finger_jump():
    """Release holds still; a nearby coordinated open-hand correction reaches the clip."""
    from isaaclab_tasks.contrib.gb300_sharpa_insertion.robot_cfg import CLOSE_HAND_RAD

    from examples.cable_installation.config import RemovalPostures
    from examples.cable_installation.cycle import CableCycleStateMachine
    from examples.cable_installation.insertion import CableInstallStateMachine

    postures = RemovalPostures()
    goal = np.array((0.5, 0.3, 2.1, 2**-0.5, 0.0, 0.0, 2**-0.5))
    tcp = np.r_[goal[:3] + postures.removal_tcp_offset_w, postures.removal_tcp_orientation_w]
    tcp[:3] += (0.003, -0.008, -0.012)
    # A settled physical grasp can require a different wrist heading at seating.
    tcp[3:] = (Rotation.from_rotvec((0.14, 0.0, 0.0)) * Rotation.from_quat(tcp[3:])).as_quat()
    head = goal.copy()
    insertion = CableInstallStateMachine(tcp, goal, 1 / 60, retreat_offset_w=(0.0, 0.0, 0.0))
    released = postures.removal_hand_open
    machine = CableCycleStateMachine(insertion, postures, CLOSE_HAND_RAD)
    previous_fingers = np.asarray(CLOSE_HAND_RAD)
    seated_tcp = None
    maximum_finger_increment = 0.0
    for _ in range(1800):
        previous_tcp = tcp.copy()
        prior_phase = insertion.phase
        tcp, fingers = machine.update(
            tcp,
            head,
            latch_engaged=True,
            latch_angle_rad=0.0,
            capture_margins_m=np.array((0.0002, 0.001, 0.0008)),
            keeper_release_margin_m=0.0008,
            keeper_exit_margin_m=0.006,
        )
        if prior_phase in ("RAISE", "ORIENT", "ALIGN", "INSERT"):
            head[:3] += tcp[:3] - previous_tcp[:3]
        if prior_phase == "INSERT" and insertion.phase == "RELEASE":
            seated_tcp = tcp.copy()
            # Model a captured plug settling in the jack after the hand opens.
            head[:3] += (0.0003, -0.0002, -0.001)
            head[3:] = (Rotation.from_rotvec((0.04, 0.0, 0.0)) * Rotation.from_quat(goal[3:])).as_quat()
        if seated_tcp is not None:
            assert np.linalg.norm(tcp[:3] - seated_tcp[:3]) <= 0.018
            assert seated_tcp[1] - 1e-9 <= tcp[1] <= seated_tcp[1] + 0.011
            assert (Rotation.from_quat(seated_tcp[3:]).inv() * Rotation.from_quat(tcp[3:])).magnitude() <= 0.15
            if machine.removal is None:
                np.testing.assert_allclose(tcp, seated_tcp, atol=1e-9)
            maximum_finger_increment = max(maximum_finger_increment, np.abs(fingers - previous_fingers).max())
        previous_fingers = fingers
        if machine.phase in ("REMOVAL/SETTLE", "REMOVAL/ORIENT", "REMOVAL/APPROACH", "REMOVAL/SUPPORT"):
            np.testing.assert_allclose(fingers, released, atol=1e-9)
        if machine.phase in ("REMOVAL/SUPPORT", "FAILED"):
            break
    assert machine.phase == "REMOVAL/SUPPORT", machine.failure
    assert insertion.phase == "DONE", "Measured hands-off retention must precede regrasp."
    x, y, z = postures.removal_tcp_offset_w
    sine, cosine = math.sin(0.04), math.cos(0.04)
    expected_offset = np.array((x, cosine * y - sine * z, sine * y + cosine * z))
    np.testing.assert_allclose(tcp[:3], head[:3] + expected_offset, atol=0.002)
    expected_rotation = Rotation.from_rotvec((0.04, 0.0, 0.0)) * Rotation.from_quat(postures.removal_tcp_orientation_w)
    assert (expected_rotation.inv() * Rotation.from_quat(tcp[3:])).magnitude() < 0.03
    posture_span = np.abs(np.asarray(CLOSE_HAND_RAD) - released).max()
    assert maximum_finger_increment <= (0.01 / 0.04) * posture_span * insertion.dt + 1e-9


@pytest.mark.parametrize("recover", (True, False))
def test_removal_pauses_at_measured_pose_and_requires_stable_clip_recovery(recover):
    """Housing purchase precedes clip depression; measured recovery still gates withdrawal."""
    from examples.cable_installation.config import RemovalPostures
    from examples.cable_installation.contacts import GraspContacts
    from examples.cable_installation.extraction import CableRemovalStateMachine, removal_hand_targets

    head = np.array((0.5, 0.3, 2.1, 0.0, 0.0, 0.0, 1.0))
    machine = CableRemovalStateMachine(
        head, head, head, 1 / 60, settle_duration_s=0.0, thumb_clearance_reserve_m=0.0001
    )
    contacts = GraspContacts(
        True,
        True,
        0.0003,
        False,
        "",
        (
            (
                "/World/envs/env_0/Robot/right_index_elastomer/hydroelastic_grasp | /World/envs/env_0/Cable/head/grip",
                0.0003,
            ),
        ),
    )
    tcp = head.copy()
    postures = RemovalPostures()

    def update(clearance_m, angle_rad=-0.2):
        return machine.update(
            tcp,
            head,
            contacts if recover else None,
            latch_angle_rad=angle_rad,
            capture_margins_m=np.array((0.0002, 0.001, clearance_m)),
            thumb_latch_contact=True,
            keeper_exit_margin_m=0.0002,
        )

    for _ in range(400):
        previous_phase = machine.phase
        tcp, blends = update(-0.0002 if machine.phase in ("PRESS", "PINCH") else 0.0008)
        if previous_phase == "SUPPORT" and machine.phase == "PRESS":
            assert blends[1] == 0.0, "The clip must remain unpressed while the housing pinch closes."
            fingers = removal_hand_targets(postures, blends, machine.grip_blend)
            np.testing.assert_allclose(fingers[5:13], postures.removal_hand_support[5:13])
        if machine.phase == "WITHDRAW":
            fingers = removal_hand_targets(postures, blends, machine.grip_blend)
            np.testing.assert_allclose(fingers[5:13], (*postures.removal_index_grasp, *postures.removal_middle_grasp))
            break
    assert machine.phase == "WITHDRAW"
    for _ in range(15):
        previous = tcp.copy()
        tcp, _ = update(-0.0002)
        head[:3] += tcp[:3] - previous[:3]
    # Measured tracking lag must cancel a still-pulling action target on pause.
    tcp[1] += 0.001
    head[1] += 0.001
    for _ in range(45):
        command, _ = update(0.0001, -0.1)
        assert machine.phase == "WITHDRAW"
        np.testing.assert_allclose(command, tcp, atol=1e-12)
    if not recover:
        for _ in range(100):
            command, _ = update(0.0001, -0.1)
            np.testing.assert_allclose(command, tcp, atol=1e-12)
            if machine.phase == "FAILED":
                break
        assert machine.phase == "FAILED"
        assert "Clip rebounded" in machine.failure
        return
    previous_thumb = machine.hand_blends[1]
    command, blends = update(-0.00014, -0.23)
    assert blends[1] < previous_thumb, "Hinge overtravel must back off even while keeper clearance is insufficient."
    np.testing.assert_allclose(command, tcp, atol=1e-12)
    for _ in range(8):
        command, _ = update(-0.0002)
        np.testing.assert_allclose(command, tcp, atol=1e-12)
    for _ in range(30):
        previous = tcp.copy()
        tcp, _ = update(-0.0002)
        head[:3] += tcp[:3] - previous[:3]
        assert machine.phase == "WITHDRAW"
    assert tcp[1] < command[1]
    assert machine.failure is None
