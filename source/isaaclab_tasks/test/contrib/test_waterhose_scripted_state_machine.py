# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the waterhose scripted end-effector trajectory."""

from types import SimpleNamespace

import torch
import warp as wp

from isaaclab.utils.math import normalize, quat_apply

from isaaclab_tasks.contrib.waterhose import scripted_state_machine
from isaaclab_tasks.contrib.waterhose.scripted_state_machine import WaterhoseDemoState, connector_retained_mask


class _FakeRobot:
    def __init__(self):
        identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.data = SimpleNamespace(
            root_link_pose_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, *identity.tolist()]])),
            body_pos_w=SimpleNamespace(torch=torch.tensor([[[0.4, 0.2, 0.8], [0.2, 0.1, 0.8], [0.0, 0.0, 0.5]]])),
            body_quat_w=SimpleNamespace(torch=identity.repeat(1, 3, 1)),
        )
        self._body_ids = {"right_gripper_base": 0, "left_gripper_base": 1, "torso_hip_yaw": 2}

    def find_bodies(self, name):
        return [self._body_ids[name]], [name]


class _FakeCable:
    def __init__(self):
        self.position = torch.tensor([[-0.25, 0.25, 0.25]])
        self.orientation = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    def get_connector_pose_w(self):
        return self.position, self.orientation


class _FakeScene(dict):
    def __init__(self, robot, cable):
        super().__init__(robot=robot, cable1=cable)
        self.env_origins = torch.zeros((1, 3))


def test_scripted_policy_warms_deferred_rtx_graph_before_control_capture(monkeypatch):
    """Kit's deferred physics capture must not force the whole rollout onto the eager manager path."""
    from isaaclab_newton.physics import NewtonManager

    graph = object()
    calls = []

    class FakeSim:
        def step(self, *, render):
            calls.append(("step", render))
            NewtonManager._graph = graph
            NewtonManager._graph_capture_pending = False

    env = SimpleNamespace(
        device="cuda:1",
        num_envs=1,
        step_dt=0.01,
        sim=FakeSim(),
        reset=lambda: calls.append(("reset", None)),
    )
    graph_policy = object()
    monkeypatch.setattr(NewtonManager, "_graph", None)
    monkeypatch.setattr(NewtonManager, "_graph_capture_pending", True)
    monkeypatch.setattr(
        scripted_state_machine,
        "WaterhoseGraphDemoState",
        lambda selected_env, *, settle_time: graph_policy,
    )

    policy = scripted_state_machine.create_scripted_policy(env, settle_time=2.0)

    assert policy is graph_policy
    assert calls == [("step", False), ("reset", None)]


def test_carry_target_recenters_connector_slip_within_phase():
    """CARRY must keep the measured connector tip centred while the grasp settles."""

    cable = _FakeCable()
    env = SimpleNamespace(
        scene=_FakeScene(_FakeRobot(), cable),
        action_manager=SimpleNamespace(total_action_dim=22),
    )
    state = WaterhoseDemoState(num_envs=1, step_dt=0.01, device="cpu", settle_time=0.01, debug=False)
    state.phase[:] = state.CARRY

    # Evaluate the completed waypoint before and after a lateral grasp displacement.
    state.compute(env)
    state.elapsed[:] = state.durations[state.CARRY]
    target_before_slip = state.compute(env)[:, :7].clone()

    # A contact deflection changes the measured connector pose while the robot pose is unchanged.
    # The command must respond so the physical connector, rather than an entry snapshot, remains
    # centred on the socket. This live translational correction is required for insertion.
    cable.position[:, 0] += 0.10
    state.elapsed[:] = state.durations[state.CARRY]
    target_after_slip = state.compute(env)[:, :7]

    assert torch.linalg.norm(target_after_slip[:, :3] - target_before_slip[:, :3]) > 0.05


def test_carry_does_not_advance_on_hard_timeout():
    """CARRY must wait for convergence instead of timing a missed grasp into ALIGN."""

    cable = _FakeCable()
    env = SimpleNamespace(
        scene=_FakeScene(_FakeRobot(), cable),
        action_manager=SimpleNamespace(total_action_dim=22),
    )
    state = WaterhoseDemoState(num_envs=1, step_dt=0.01, device="cpu", settle_time=0.01, debug=False)
    state.phase[:] = state.CARRY
    state.elapsed[:] = 2.0 * state.durations[state.CARRY]
    state.pos_tolerance.zero_()
    state.rot_tolerance = 0.0

    state.compute(env)

    assert state.phase.item() == state.CARRY


def test_lost_grasp_retries_approach_before_align():
    """A connector that stops following the gripper must restart the pick."""

    cable = _FakeCable()
    env = SimpleNamespace(
        scene=_FakeScene(_FakeRobot(), cable),
        action_manager=SimpleNamespace(total_action_dim=22),
    )
    state = WaterhoseDemoState(num_envs=1, step_dt=0.01, device="cpu", settle_time=0.01, debug=False)

    # RETRACT entry captures the measured tip-in-EE offset for the closed grasp.
    state.phase[:] = state.RETRACT
    state.compute(env)

    # Emulate a missed/slipped grasp: the gripper remains in place while the connector separates.
    cable.position[:, 0] += 0.10
    state.phase[:] = state.CARRY
    state.elapsed.zero_()
    state.compute(env)

    assert state.phase.item() == state.APPROACH
    assert state.elapsed.item() == 0.0
    assert not state._grasp_reference_valid.item()


def test_graph_state_machine_retries_a_lost_grasp_on_cpu():
    """The captured Warp controller must apply the same measured-grasp recovery."""

    device = "cpu"
    identity = wp.transform()
    robot_body_q = wp.array([[identity]], dtype=wp.transform, device=device)
    robot_root_q = wp.array([identity], dtype=wp.transform, device=device)
    cable_body_q = wp.array(
        [wp.transform(wp.vec3(0.10, 0.0, 0.0), wp.quat_identity())], dtype=wp.transform, device=device
    )
    cable_head_bodies = wp.array([0], dtype=wp.int32, device=device)
    env_origins = wp.zeros(1, dtype=wp.vec3, device=device)
    phase = wp.array([WaterhoseDemoState.CARRY], dtype=wp.int32, device=device)
    elapsed = wp.zeros(1, dtype=wp.float32, device=device)
    durations = wp.array(WaterhoseDemoState.DURATIONS, dtype=wp.float32, device=device)
    phase_ee = wp.array([identity], dtype=wp.transform, device=device)
    phase_connector = wp.array([identity], dtype=wp.transform, device=device)
    frozen_tip_offset = wp.zeros(1, dtype=wp.vec3, device=device)
    frozen_insert_rotation = wp.array([wp.quat_identity()], dtype=wp.quat, device=device)
    grasp_reference_valid = wp.array([True], dtype=wp.bool, device=device)
    right_target_position = wp.zeros(1, dtype=wp.vec3, device=device)
    right_target_rotation = wp.zeros(1, dtype=wp.vec4, device=device)
    gripper_blend = wp.zeros(1, dtype=wp.float32, device=device)
    diagnostics = wp.zeros((1, 6), dtype=wp.float32, device=device)

    wp.launch(
        scripted_state_machine._update_state_machine_wp,
        dim=1,
        inputs=[
            robot_body_q,
            robot_root_q,
            cable_body_q,
            cable_head_bodies,
            env_origins,
            phase,
            elapsed,
            durations,
            phase_ee,
            phase_connector,
            frozen_tip_offset,
            frozen_insert_rotation,
            grasp_reference_valid,
            0,
            identity,
            identity,
            wp.vec3(0.0, 0.0, 0.0),
            wp.quat_identity(),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.quat_identity(),
            wp.vec3(0.0, 0.0, 0.0),
            0.01,
            right_target_position,
            right_target_rotation,
            gripper_blend,
            diagnostics,
        ],
        device=device,
    )

    assert phase.numpy()[0] == WaterhoseDemoState.APPROACH
    assert elapsed.numpy()[0] == 0.0
    assert not grasp_reference_valid.numpy()[0]


def test_approach_target_is_fixed_to_its_phase_entry_connector_pose():
    """APPROACH must not chase cable motion and shift the connector within the finger pads."""

    cable = _FakeCable()
    env = SimpleNamespace(
        scene=_FakeScene(_FakeRobot(), cable),
        action_manager=SimpleNamespace(total_action_dim=22),
    )
    state = WaterhoseDemoState(num_envs=1, step_dt=0.01, device="cpu", settle_time=0.01, debug=False)
    state.phase[:] = state.APPROACH

    state.compute(env)
    state.elapsed[:] = state.durations[state.APPROACH]
    target_before_motion = state.compute(env)[:, :3].clone()

    cable.position[:, 0] += 0.10
    state.elapsed[:] = state.durations[state.APPROACH]
    target_after_motion = state.compute(env)[:, :3]

    torch.testing.assert_close(target_after_motion, target_before_motion)


def test_insert_retries_alignment_on_hard_timeout_when_connector_is_not_retained():
    """INSERT must re-align instead of reporting progress after a failed push."""

    cable = _FakeCable()
    env = SimpleNamespace(
        scene=_FakeScene(_FakeRobot(), cable),
        action_manager=SimpleNamespace(total_action_dim=22),
    )
    state = WaterhoseDemoState(num_envs=1, step_dt=0.01, device="cpu", settle_time=0.01, debug=False)
    cable.position[:] = state.socket_pos_w + 1.0
    state.phase[:] = state.INSERT
    state.elapsed[:] = 2.0 * state.durations[state.INSERT]

    # The fake connector is far from the socket. Reaching the generic 2x phase timeout must return
    # to ALIGN, never advance to HOLD_INSERTED or wait forever on the failed axial target.
    state.compute(env)

    assert state.phase.item() == state.ALIGN
    assert state.elapsed.item() == 0.0


def test_align_replans_from_measured_pose_on_hard_timeout():
    """A stale ALIGN correction must be refreshed without advancing to INSERT."""

    cable = _FakeCable()
    env = SimpleNamespace(
        scene=_FakeScene(_FakeRobot(), cable),
        action_manager=SimpleNamespace(total_action_dim=22),
    )
    state = WaterhoseDemoState(num_envs=1, step_dt=0.01, device="cpu", settle_time=0.01, debug=False)
    cable.position[:] = state.socket_pos_w + 1.0
    state.phase[:] = state.ALIGN
    state.elapsed[:] = 2.0 * state.durations[state.ALIGN]

    state.compute(env)

    assert state.phase.item() == state.ALIGN
    assert state.elapsed.item() == 0.0


def test_insert_does_not_advance_on_ee_convergence_when_connector_is_not_retained():
    """INSERT completion must reflect the connector pose, not only the commanded EE pose."""

    cable = _FakeCable()
    env = SimpleNamespace(
        scene=_FakeScene(_FakeRobot(), cable),
        action_manager=SimpleNamespace(total_action_dim=22),
    )
    state = WaterhoseDemoState(num_envs=1, step_dt=0.01, device="cpu", settle_time=0.01, debug=False)
    cable.position[:] = state.socket_pos_w + 1.0
    state.phase[:] = state.INSERT
    state.elapsed[:] = state.durations[state.INSERT]
    state.pos_tolerance.fill_(torch.inf)
    state.rot_tolerance = torch.inf

    # Force the generic end-effector convergence predicate true while leaving the connector far
    # from the socket. INSERT must still wait for measured retention.
    state.compute(env)

    assert state.phase.item() == state.INSERT


def test_connector_retention_uses_measured_seated_pose():
    """The terminal predicate accepts the validated seat and rejects lateral loss."""

    cable = _FakeCable()
    env = SimpleNamespace(
        scene=_FakeScene(_FakeRobot(), cable),
        action_manager=SimpleNamespace(total_action_dim=22),
    )
    state = WaterhoseDemoState(num_envs=1, step_dt=0.01, device="cpu", settle_time=0.01, debug=False)
    cable.orientation[:] = state.socket_quat_w
    insertion_axis = normalize(quat_apply(state.socket_quat_w, state.connector_axis_local))
    tip_offset = quat_apply(cable.orientation, state.connector_tip_local_pos)
    cable.position[:] = state.socket_pos_w + state.seated_tip_depth * insertion_axis - tip_offset

    assert connector_retained_mask(env).item()

    cable.position[:, 0] += 2.0e-3
    assert not connector_retained_mask(env).item()
