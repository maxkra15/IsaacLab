# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure compatibility tests for the Franka RJ45 reset-tool helpers."""

import math
import os
import stat
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from isaaclab.utils import math as math_utils

from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_env import FrankaRJ45PickInsertEnv
from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_env_cfg import (
    PICK_INSERT_CLOSED_FINGER_POSITION,
    PICK_INSERT_EFFECTIVE_GRASP_FRICTION,
    PICK_INSERT_GRASP_PROXY_FRICTION,
    PICK_INSERT_SUCCESS_MAX_PLUG_SPEED,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.rj45_env import FrankaRJ45InsertionEnv

from scripts.tools import _franka_rj45_reset_tools as reset_tools
from scripts.tools._franka_rj45_reset_tools import (
    RJ45PickInsertResetToolEnv,
    RJ45ResetToolEnv,
    exact_success_from_state,
    plug_relative_latch_angle,
    scalar_goal_error,
)


def _identity_pose(batch_size: int, body_count: int) -> torch.Tensor:
    pose = torch.zeros((batch_size, body_count, 7), dtype=torch.float32)
    pose[..., 6] = 1.0
    return pose


def test_torch_artifact_atomic_save_strictly_reloads_and_syncs_file_and_directory(tmp_path, monkeypatch) -> None:
    output = tmp_path / "artifact.pt"
    payload = {
        "tensor": torch.tensor(((1.0, 2.0), (3.0, 4.0)), dtype=torch.float32),
        "metadata": {"sequence": (True, 7, "exact")},
    }
    synchronized_modes: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(descriptor: int) -> None:
        synchronized_modes.append(os.fstat(descriptor).st_mode)
        real_fsync(descriptor)

    monkeypatch.setattr(reset_tools.os, "fsync", recording_fsync)

    reset_tools.save_torch_atomic(payload, output)

    loaded = torch.load(output, map_location="cpu", weights_only=True)
    assert reset_tools._torch_artifact_values_equal(payload, loaded)
    assert any(stat.S_ISREG(mode) for mode in synchronized_modes)
    assert any(stat.S_ISDIR(mode) for mode in synchronized_modes)
    assert not list(tmp_path.glob(".artifact.pt.*.tmp"))


def test_torch_artifact_atomic_save_preserves_prior_destination_on_reload_mismatch(tmp_path, monkeypatch) -> None:
    output = tmp_path / "artifact.pt"
    prior = b"prior-artifact-bytes"
    output.write_bytes(prior)
    monkeypatch.setattr(reset_tools, "_torch_artifact_values_equal", lambda _expected, _observed: False)

    with pytest.raises(RuntimeError, match="changed during its strict temporary-file reload"):
        reset_tools.save_torch_atomic({"tensor": torch.ones(2)}, output)

    assert output.read_bytes() == prior
    assert not list(tmp_path.glob(".artifact.pt.*.tmp"))


def test_torch_artifact_atomic_save_directory_sync_failure_leaves_validated_replacement(
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "artifact.pt"
    torch.save({"tensor": torch.zeros(2)}, output)
    replacement = {"tensor": torch.ones(2)}
    real_fsync = os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(reset_tools.os, "fsync", fail_directory_fsync)

    with pytest.raises(OSError, match="injected directory fsync failure"):
        reset_tools.save_torch_atomic(replacement, output)

    loaded = torch.load(output, map_location="cpu", weights_only=True)
    assert reset_tools._torch_artifact_values_equal(replacement, loaded)
    assert not list(tmp_path.glob(".artifact.pt.*.tmp"))


def _scalar_collision_reference(case: dict[str, Any], penetration_tolerance: float) -> dict[str, Any]:
    """Frozen CPU reference for the former per-contact collision loop."""
    values = {name: value.detach().cpu() if isinstance(value, torch.Tensor) else value for name, value in case.items()}
    num_envs = int(values["num_envs"])
    capacity = int(values["contact_shape0"].shape[0])
    reported = int(values["contact_count"][0])
    count = min(reported, capacity)
    overflow = reported > capacity
    valid = torch.ones(num_envs, dtype=torch.bool)
    invalid_count = torch.zeros(num_envs, dtype=torch.long)
    left_count = torch.zeros_like(invalid_count)
    right_count = torch.zeros_like(invalid_count)
    invalid_mask = torch.zeros(capacity, dtype=torch.bool)
    invalid_pairs: list[str] = []
    body_labels = [str(label) for label in values["body_labels"]]
    shape_labels = [str(label) for label in values["shape_labels"]]
    shape_body = values["shape_body"]
    shape_world = values["shape_world"]
    body_q = values["body_q"]
    for contact_id in range(count):
        shape0 = int(values["contact_shape0"][contact_id])
        shape1 = int(values["contact_shape1"][contact_id])
        body0 = int(shape_body[shape0])
        body1 = int(shape_body[shape1])
        world0 = int(shape_world[shape0])
        world1 = int(shape_world[shape1])
        world = world0 if world0 >= 0 else world1
        if not 0 <= world < num_envs:
            continue

        points = []
        pair = []
        for body_id, shape_id, point in (
            (body0, shape0, values["contact_point0"][contact_id]),
            (body1, shape1, values["contact_point1"][contact_id]),
        ):
            body_label = body_labels[body_id] if body_id >= 0 else ""
            pair.append((body_label, shape_labels[shape_id]))
            if body_id >= 0:
                pose = body_q[body_id]
                point = pose[:3] + math_utils.quat_apply(pose[None, 3:7], point[None])[0]
            points.append(point)
        separation = (
            torch.dot(values["contact_normal"][contact_id], points[1] - points[0])
            - values["contact_margin0"][contact_id]
            - values["contact_margin1"][contact_id]
        )
        robot_index = next(
            (
                index
                for index, (body_label, shape_label) in enumerate(pair)
                if "/Robot/" in body_label or "/Robot/" in shape_label
            ),
            None,
        )
        if robot_index is None:
            continue
        robot_label = " ".join(pair[robot_index])
        other_shape_label = pair[1 - robot_index][1]
        is_grasp_proxy = other_shape_label.endswith("/Plug/GraspProxy")
        if is_grasp_proxy and "panda_leftfinger" in robot_label:
            left_count[world] += 1
        elif is_grasp_proxy and "panda_rightfinger" in robot_label:
            right_count[world] += 1
        elif float(separation) < -penetration_tolerance:
            invalid_count[world] += 1
            invalid_mask[contact_id] = True
            valid[world] = False
            if len(invalid_pairs) < 64:
                first = pair[0][0] or pair[0][1]
                second = pair[1][0] or pair[1][1]
                invalid_pairs.append(f"world={world} {first} <-> {second} separation={float(separation):.6g}")
    if overflow:
        valid[:] = False
    return {
        "valid": valid,
        "invalid_contact_count": invalid_count,
        "left_grasp_contact_count": left_count,
        "right_grasp_contact_count": right_count,
        "contact_overflow": overflow,
        "invalid_contact_mask": invalid_mask,
        "invalid_contact_pairs": tuple(invalid_pairs),
    }


def _vector_collision_result(
    case: dict[str, Any],
    penetration_tolerance: float,
    device: torch.device,
) -> tuple[reset_tools._CollisionBufferReduction, reset_tools._CollisionLabelLayout]:
    tensors = {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in case.items()}
    labels = reset_tools._collision_label_layout(
        tensors["shape_body"],
        tensors["shape_world"],
        tensors["body_labels"],
        tensors["shape_labels"],
    )
    capacity = tensors["contact_shape0"].shape[0]
    reduction = reset_tools._reduce_collision_buffer(
        contact_count=tensors["contact_count"],
        contact_slots=torch.arange(capacity, device=device, dtype=torch.long),
        contact_shape0=tensors["contact_shape0"],
        contact_shape1=tensors["contact_shape1"],
        contact_point0=tensors["contact_point0"],
        contact_point1=tensors["contact_point1"],
        contact_normal=tensors["contact_normal"],
        contact_margin0=tensors["contact_margin0"],
        contact_margin1=tensors["contact_margin1"],
        labels=labels,
        body_q=tensors["body_q"],
        num_envs=tensors["num_envs"],
        penetration_tolerance=penetration_tolerance,
    )
    return reduction, labels


def _curated_collision_case() -> dict[str, Any]:
    separations = torch.tensor((-0.01, -0.01, -0.001, -0.01, -0.0005, -0.002, -0.003, -0.01, -0.004, float("nan"), 0.0))
    point0 = torch.zeros((len(separations), 3))
    point1 = torch.zeros_like(point0)
    point1[:, 0] = separations
    normal = torch.zeros_like(point0)
    normal[:, 0] = 1.0
    body_q = _identity_pose(1, 4)[0]
    return {
        "num_envs": 2,
        "contact_count": torch.tensor((10,), dtype=torch.int32),
        "contact_shape0": torch.tensor((0, 2, 4, 5, 4, 4, 3, 8, 0, 4, 99), dtype=torch.int32),
        "contact_shape1": torch.tensor((2, 1, 5, 3, 5, 6, 4, 5, 1, 5, -3), dtype=torch.int32),
        "contact_point0": point0,
        "contact_point1": point1,
        "contact_normal": normal,
        "contact_margin0": torch.zeros(len(separations)),
        "contact_margin1": torch.zeros(len(separations)),
        "shape_body": torch.tensor((0, 1, -1, -1, 2, 3, -1, -1, -1), dtype=torch.int32),
        "shape_world": torch.tensor((0, 0, 0, 0, 1, 1, 1, 1, 2), dtype=torch.int32),
        "body_q": body_q,
        "body_labels": (
            "/World/envs/env_0/Robot/panda_leftfinger",
            "/World/envs/env_0/Robot/panda_rightfinger",
            "/World/envs/env_1/Robot/panda_hand",
            "/World/envs/env_1/Object/task",
        ),
        "shape_labels": (
            "/World/envs/env_0/Robot/left/Collision",
            "/World/envs/env_0/Robot/right/Collision",
            "/World/envs/env_0/Plug/GraspProxy",
            "/World/envs/env_0/Table/Collision",
            "/World/envs/env_1/Robot/hand/Collision",
            "/World/envs/env_1/Object/task/Collision",
            "/World/envs/env_1/Plug/GraspProxyExtra",
            "/World/envs/env_1/Plug/GraspProxy",
            "/World/envs/env_2/Robot/shape_only",
        ),
    }


def _assert_collision_reduction_matches_reference(
    reduction: reset_tools._CollisionBufferReduction,
    labels: reset_tools._CollisionLabelLayout,
    reference: dict[str, Any],
) -> None:
    assert torch.equal(reduction.valid.cpu(), reference["valid"])
    assert torch.equal(reduction.invalid_contact_count.cpu(), reference["invalid_contact_count"])
    assert torch.equal(reduction.left_grasp_contact_count.cpu(), reference["left_grasp_contact_count"])
    assert torch.equal(reduction.right_grasp_contact_count.cpu(), reference["right_grasp_contact_count"])
    assert bool(reduction.contact_overflow.cpu()) is reference["contact_overflow"]
    assert torch.equal(reduction.invalid_contact_mask.cpu(), reference["invalid_contact_mask"])
    assert not bool(reduction.negative_contact_count.cpu())
    assert not bool(reduction.invalid_active_shape.cpu())
    assert not bool(reduction.invalid_active_body.cpu())
    assert reset_tools._invalid_contact_pairs(reduction, labels) == reference["invalid_contact_pairs"]


def test_vector_collision_reduction_preserves_scalar_contact_semantics() -> None:
    tolerance = 2**-11
    case = _curated_collision_case()
    case["contact_point1"][4, 0] = -tolerance
    reference = _scalar_collision_reference(case, tolerance)

    reduction, labels = _vector_collision_result(case, tolerance, torch.device("cpu"))

    assert reference["invalid_contact_count"].tolist() == [2, 2]
    assert reference["left_grasp_contact_count"].tolist() == [1, 0]
    assert reference["right_grasp_contact_count"].tolist() == [1, 0]
    assert reference["invalid_contact_mask"].nonzero().flatten().tolist() == [2, 5, 6, 8]
    _assert_collision_reduction_matches_reference(reduction, labels, reference)


def test_vector_collision_reduction_processes_capacity_then_invalidates_all_on_overflow() -> None:
    tolerance = 2**-11
    case = _curated_collision_case()
    case["contact_point1"][4, 0] = -tolerance
    case["contact_shape0"][10] = 5
    case["contact_shape1"][10] = 3
    case["contact_count"][0] = case["contact_shape0"].shape[0] + 3
    reference = _scalar_collision_reference(case, tolerance)

    reduction, labels = _vector_collision_result(case, tolerance, torch.device("cpu"))

    assert reference["contact_overflow"] is True
    assert not bool(reference["valid"].any())
    _assert_collision_reduction_matches_reference(reduction, labels, reference)


def test_collision_buffer_adapter_caches_layout_and_preserves_diagnostic_order(monkeypatch) -> None:
    capacity = 70
    separations = -torch.arange(1, capacity + 1, dtype=torch.float32) / 1000.0
    point0 = torch.zeros((capacity, 3))
    point1 = torch.zeros_like(point0)
    point1[:, 0] = separations
    normal = torch.zeros_like(point0)
    normal[:, 0] = 1.0
    case = {
        "num_envs": 1,
        "contact_count": torch.tensor((capacity,), dtype=torch.int32),
        "contact_shape0": torch.zeros(capacity, dtype=torch.int32),
        "contact_shape1": torch.ones(capacity, dtype=torch.int32),
        "contact_point0": point0,
        "contact_point1": point1,
        "contact_normal": normal,
        "contact_margin0": torch.zeros(capacity),
        "contact_margin1": torch.zeros(capacity),
        "shape_body": torch.tensor((0, -1), dtype=torch.int32),
        "shape_world": torch.tensor((0, 0), dtype=torch.int32),
        "body_q": _identity_pose(1, 1)[0],
        "body_labels": ("/World/envs/env_0/Robot/panda_hand",),
        "shape_labels": ("/Robot/hand/Collision", "/Table/Collision", "/Unused/ExtraLabel"),
    }
    contacts = SimpleNamespace(
        rigid_contact_max=capacity,
        rigid_contact_count=case["contact_count"],
        rigid_contact_shape0=case["contact_shape0"],
        rigid_contact_shape1=case["contact_shape1"],
        rigid_contact_point0=case["contact_point0"],
        rigid_contact_point1=case["contact_point1"],
        rigid_contact_normal=case["contact_normal"],
        rigid_contact_margin0=case["contact_margin0"],
        rigid_contact_margin1=case["contact_margin1"],
    )
    model = SimpleNamespace(
        shape_body=case["shape_body"],
        shape_world=case["shape_world"],
        body_label=case["body_labels"],
        shape_label=case["shape_labels"],
    )
    state = SimpleNamespace(body_q=case["body_q"])
    env = SimpleNamespace(num_envs=1, device="cpu")
    monkeypatch.setattr(reset_tools.wp, "to_torch", lambda value: value)
    reference = _scalar_collision_reference(case, 5.0e-4)

    first = reset_tools._collision_buffer_metrics(env, contacts, model, state, 5.0e-4)
    second = reset_tools._collision_buffer_metrics(env, contacts, model, state, 5.0e-4)
    contacts.rigid_contact_count[0] = capacity + 1
    overflow = reset_tools._collision_buffer_metrics(env, contacts, model, state, 5.0e-4)

    assert first.invalid_contact_count.tolist() == [70]
    assert first.invalid_contact_pairs == reference["invalid_contact_pairs"]
    assert len(first.invalid_contact_pairs) == 64
    assert first.invalid_contact_pairs[0].endswith("separation=-0.001")
    assert first.invalid_contact_pairs[-1].endswith("separation=-0.064")
    assert second.invalid_contact_pairs == first.invalid_contact_pairs
    assert overflow.contact_overflow is True
    assert overflow.invalid_contact_count.tolist() == [70]
    assert not bool(overflow.valid.any())
    assert len(env._reset_tool_collision_buffer_bindings) == 1


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_vector_collision_reduction_matches_randomized_scalar_reference(device_name: str) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_name)
    shape_body = torch.tensor((0, 1, -1, 2, 3, -1, -1, -1), dtype=torch.int32)
    shape_world = torch.tensor((0, 0, 0, 1, 1, -1, 2, 3), dtype=torch.int32)
    body_labels = (
        "/World/envs/env_0/Robot/panda_leftfinger",
        "/World/envs/env_0/Robot/panda_rightfinger",
        "/World/envs/env_1/Robot/panda_hand",
        "/World/envs/env_1/Object/task",
    )
    shape_labels = (
        "/World/envs/env_0/Robot/left/Collision",
        "/World/envs/env_0/Robot/right/Collision",
        "/World/envs/env_0/Plug/GraspProxy",
        "/World/envs/env_1/Robot/hand/Collision",
        "/World/envs/env_1/Object/task/Collision",
        "/World/static/Table/Collision",
        "/World/envs/env_2/Robot/shape_only",
        "/World/envs/env_3/Plug/GraspProxy",
    )
    quaternions = torch.tensor(
        (
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 2**-0.5, 2**-0.5),
            (0.0, 0.0, -(2**-0.5), 2**-0.5),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    tolerance = 0.025
    capacity = 32
    for seed in range(12):
        random = torch.Generator().manual_seed(seed)
        body_q = torch.zeros((4, 7))
        body_q[:, :3] = torch.randint(-2, 3, (4, 3), generator=random).float() * 0.1
        body_q[:, 3:7] = quaternions
        points = torch.randint(-3, 4, (2, capacity, 3), generator=random).float() * 0.05
        normal_axis = torch.randint(0, 3, (capacity,), generator=random)
        normal_sign = torch.randint(0, 2, (capacity,), generator=random).float() * 2.0 - 1.0
        normal = torch.zeros((capacity, 3))
        normal.scatter_(1, normal_axis[:, None], normal_sign[:, None])
        case = {
            "num_envs": 3,
            "contact_count": torch.randint(0, capacity + 3, (1,), generator=random, dtype=torch.int32),
            "contact_shape0": torch.randint(0, len(shape_labels), (capacity,), generator=random, dtype=torch.int32),
            "contact_shape1": torch.randint(0, len(shape_labels), (capacity,), generator=random, dtype=torch.int32),
            "contact_point0": points[0],
            "contact_point1": points[1],
            "contact_normal": normal,
            "contact_margin0": torch.zeros(capacity),
            "contact_margin1": torch.zeros(capacity),
            "shape_body": shape_body,
            "shape_world": shape_world,
            "body_q": body_q,
            "body_labels": body_labels,
            "shape_labels": shape_labels,
        }
        reference = _scalar_collision_reference(case, tolerance)

        reduction, labels = _vector_collision_result(case, tolerance, device)

        assert torch.equal(reduction.valid.cpu(), reference["valid"])
        assert torch.equal(reduction.invalid_contact_count.cpu(), reference["invalid_contact_count"])
        assert torch.equal(reduction.left_grasp_contact_count.cpu(), reference["left_grasp_contact_count"])
        assert torch.equal(reduction.right_grasp_contact_count.cpu(), reference["right_grasp_contact_count"])
        assert bool(reduction.contact_overflow.cpu()) is reference["contact_overflow"]
        assert torch.equal(reduction.invalid_contact_mask.cpu(), reference["invalid_contact_mask"])
        observed_pairs = reset_tools._invalid_contact_pairs(reduction, labels)
        assert observed_pairs == reference["invalid_contact_pairs"]


def test_collide_only_metrics_refreshes_outer_and_proxy_without_stepping(monkeypatch) -> None:
    """Static reset admission must reuse the production contact classifier."""
    events: list[tuple[Any, ...]] = []
    state = object()
    contacts = object()
    expected = object()

    class Pipeline:
        def collide(self, received_state, received_contacts):
            events.append(("outer", received_state, received_contacts))

    monkeypatch.setattr(reset_tools.NewtonManager, "forward", lambda: events.append(("forward",)))
    monkeypatch.setattr(reset_tools.NewtonManager, "get_state_0", lambda: state)
    monkeypatch.setattr(reset_tools.NewtonManager, "get_contacts", lambda: contacts)
    monkeypatch.setattr(reset_tools.NewtonManager, "_collision_pipeline", Pipeline())
    monkeypatch.setattr(
        reset_tools.NewtonCouplerManager,
        "refresh_proxy_collision_contacts",
        lambda source, destination: events.append(("proxy", source, destination)),
    )

    def classify(env, penetration_tolerance, *, require_bilateral_grasp):
        events.append(("classify", env, penetration_tolerance, require_bilateral_grasp))
        return expected

    monkeypatch.setattr(reset_tools, "collision_metrics", classify)
    env = object()

    result = reset_tools.collide_only_metrics(
        env,
        penetration_tolerance=1.0e-4,
        require_bilateral_grasp=False,
    )

    assert result is expected
    assert events == [
        ("forward",),
        ("outer", state, contacts),
        ("proxy", reset_tools.RIGID_ENTRY, reset_tools.RJ45_ENTRY),
        ("classify", env, 1.0e-4, False),
    ]


@pytest.mark.parametrize("missing", ("state", "contacts", "pipeline"))
def test_collide_only_metrics_requires_complete_collision_runtime(monkeypatch, missing: str) -> None:
    monkeypatch.setattr(reset_tools.NewtonManager, "forward", lambda: None)
    monkeypatch.setattr(reset_tools.NewtonManager, "get_state_0", lambda: None if missing == "state" else object())
    monkeypatch.setattr(reset_tools.NewtonManager, "get_contacts", lambda: None if missing == "contacts" else object())
    monkeypatch.setattr(
        reset_tools.NewtonManager,
        "_collision_pipeline",
        None if missing == "pipeline" else SimpleNamespace(collide=lambda *_args: None),
    )

    with pytest.raises(RuntimeError, match="initialized outer collision pipeline"):
        reset_tools.collide_only_metrics(object())


def test_warp_to_torch_contact_ordering_uses_stream_event_without_device_sync(monkeypatch) -> None:
    producer = SimpleNamespace(cuda_stream=11)
    torch_stream = SimpleNamespace(cuda_stream=22)
    waits: list[Any] = []
    consumer = SimpleNamespace(wait_stream=lambda other: waits.append(other))
    tensor = SimpleNamespace(device=torch.device("cuda:0"))
    monkeypatch.setattr(reset_tools.wp, "get_stream", lambda _device: producer)
    monkeypatch.setattr(reset_tools.torch.cuda, "current_stream", lambda _device: torch_stream)
    monkeypatch.setattr(
        reset_tools.wp,
        "stream_from_torch",
        lambda stream: consumer if stream is torch_stream else None,
    )

    reset_tools._wait_for_warp_contact_data(tensor)

    assert waits == [producer]
    torch_stream.cuda_stream = producer.cuda_stream
    reset_tools._wait_for_warp_contact_data(tensor)
    assert waits == [producer]


def test_batched_quat_slerp_uses_shortest_normalized_arc_without_mutating_inputs() -> None:
    start = torch.tensor(((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)))
    end = torch.tensor(((0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, -1.0)))
    original_end = end.clone()

    midpoint = reset_tools.batched_quat_slerp(start, end, 0.5)

    expected = torch.tensor(((0.0, 0.0, 2**-0.5, 2**-0.5), (0.0, 0.0, 0.0, 1.0)))
    assert torch.allclose(midpoint, expected, atol=1.0e-6, rtol=0.0)
    assert torch.allclose(torch.linalg.vector_norm(midpoint, dim=-1), torch.ones(2), atol=1.0e-6, rtol=0.0)
    assert torch.equal(end, original_end)


@pytest.mark.parametrize("tau", (-0.01, 1.01, float("nan")))
def test_batched_quat_slerp_rejects_invalid_interpolation_coefficient(tau: float) -> None:
    with pytest.raises(ValueError, match="lie in \\[0, 1\\]"):
        reset_tools.batched_quat_slerp(torch.tensor((0.0, 0.0, 0.0, 1.0)), torch.ones(4), tau)


class _LaneHoldFakeEnv:
    device = "cpu"
    num_envs = 3

    def __init__(self) -> None:
        self.arm_q = torch.arange(21, dtype=torch.float32).reshape(3, 7)
        self.current_arm_target = torch.zeros((3, 7))
        self.current_finger_target = torch.zeros((3, 2))
        self.target_writes: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.physics_commands: list[tuple[torch.Tensor, torch.Tensor]] = []

    def read_robot_state(self):
        return (
            self.arm_q.clone(),
            torch.zeros_like(self.arm_q),
            torch.zeros((3, 2)),
            torch.zeros((3, 2)),
        )

    def set_robot_targets(self, arm_target: torch.Tensor, finger_target: torch.Tensor) -> None:
        self.current_arm_target = arm_target.clone()
        self.current_finger_target = finger_target.clone()
        self.target_writes.append((arm_target.clone(), finger_target.clone()))

    def advance(self, duration_s: float, update=None, *, post_step=None) -> int:
        steps = int(duration_s)
        for step in range(steps):
            if update is not None:
                update(step, steps, (step + 1) / steps)
            self.physics_commands.append((self.current_arm_target.clone(), self.current_finger_target.clone()))
            if post_step is not None:
                post_step(step, steps, (step + 1) / steps)
        return steps


def test_per_lane_target_hold_never_moves_initially_inactive_lanes_and_reapplies_each_step() -> None:
    env = _LaneHoldFakeEnv()
    initial_arm_target = torch.full((3, 7), 100.0)
    initial_finger_target = torch.full((3, 2), 0.04)
    measured_inactive_arm = env.arm_q[1].clone()

    with reset_tools._PerLaneTargetHold(
        env,
        torch.tensor((True, False, True)),
        initial_arm_target,
        initial_finger_target,
    ) as hold:
        assert env.advance(3) == 3
        env.set_robot_targets(torch.full((3, 7), 200.0), torch.full((3, 2), 0.02))
        env.advance(2)

        assert torch.equal(hold.active_mask, torch.tensor((True, False, True)))
        assert not bool(hold.failed_mask.any())

    assert len(env.physics_commands) == 5
    assert len(env.target_writes) >= 6  # Enter, each no-update step, explicit request, and each later step.
    for arm_target, finger_target in env.physics_commands:
        assert torch.equal(arm_target[1], measured_inactive_arm)
        assert torch.equal(finger_target[1], initial_finger_target[1])
    assert torch.equal(env.physics_commands[-1][0][0], torch.full((7,), 200.0))
    assert torch.equal(env.physics_commands[-1][0][2], torch.full((7,), 200.0))


def test_per_lane_target_hold_deactivation_is_monotone_and_latches_finite_stage_commands() -> None:
    env = _LaneHoldFakeEnv()
    initial_arm_target = torch.zeros((3, 7))
    initial_finger_target = torch.full((3, 2), 0.04)
    measured_lane_one = torch.full((7,), 7.5)
    lane_two_last_finite_target = torch.full((7,), 21.0)

    with reset_tools._PerLaneTargetHold(
        env,
        torch.ones(3, dtype=torch.bool),
        initial_arm_target,
        initial_finger_target,
    ) as hold:

        def update(step: int, _steps: int, _progress: float) -> None:
            arm_target = torch.full((3, 7), float(10 + step))
            arm_target[2] = float(20 + step)
            finger_target = torch.full((3, 2), 0.004 if step == 0 else 0.04)
            env.set_robot_targets(arm_target, finger_target)

        def post_step(step: int, _steps: int, _progress: float) -> None:
            if step == 0:
                env.arm_q[1] = measured_lane_one
                changed = hold.deactivate(torch.tensor((False, True, False)), reason="lost-contact")
                assert torch.equal(changed, torch.tensor((False, True, False)))
            elif step == 1:
                env.arm_q[2] = torch.nan
                changed = hold.deactivate(torch.tensor((False, False, True)), reason="speed")
                assert torch.equal(changed, torch.tensor((False, False, True)))
            elif step == 2:
                changed = hold.deactivate(torch.tensor((False, True, True)), reason="later")
                assert not bool(changed.any())

        env.advance(4, update, post_step=post_step)
        reason_masks = hold.reason_masks

        assert torch.equal(hold.active_mask, torch.tensor((True, False, False)))
        assert torch.equal(hold.failed_mask, torch.tensor((False, True, True)))
        assert torch.equal(reason_masks["lost-contact"], torch.tensor((False, True, False)))
        assert torch.equal(reason_masks["speed"], torch.tensor((False, False, True)))
        assert "later" not in reason_masks
        assert torch.equal(hold.last_sent_arm_target[1], measured_lane_one)
        assert torch.equal(hold.last_sent_arm_target[2], lane_two_last_finite_target)
        assert torch.equal(hold.last_sent_finger_target[1], torch.full((2,), 0.004))
        assert torch.equal(hold.last_sent_finger_target[2], torch.full((2,), 0.04))

    for arm_target, finger_target in env.physics_commands[1:]:
        assert torch.equal(arm_target[1], measured_lane_one)
        assert torch.equal(finger_target[1], torch.full((2,), 0.004))
    for arm_target, finger_target in env.physics_commands[2:]:
        assert torch.equal(arm_target[2], lane_two_last_finite_target)
        assert torch.equal(finger_target[2], torch.full((2,), 0.04))


def test_per_lane_target_hold_propagates_caller_owned_batch_failure_and_restores_environment() -> None:
    env = _LaneHoldFakeEnv()
    original_set_robot_targets = env.set_robot_targets
    original_advance = env.advance

    def batch_fatal(_step: int, _steps: int, _progress: float) -> None:
        raise RuntimeError("global contact overflow")

    with pytest.raises(RuntimeError, match="global contact overflow"):
        with reset_tools._PerLaneTargetHold(
            env,
            torch.ones(3, dtype=torch.bool),
            torch.zeros((3, 7)),
            torch.full((3, 2), 0.04),
        ):
            env.advance(1, post_step=batch_fatal)

    assert env.set_robot_targets == original_set_robot_targets
    assert env.advance == original_advance


def test_per_lane_target_hold_propagates_last_arm_and_finger_targets_between_stages() -> None:
    env = _LaneHoldFakeEnv()
    initial_arm = torch.zeros((3, 7))
    open_finger = torch.full((3, 2), 0.04)

    with reset_tools._PerLaneTargetHold(env, torch.ones(3, dtype=torch.bool), initial_arm, open_finger) as first:
        closed_arm = torch.full((3, 7), 3.0)
        closed_finger = torch.full((3, 2), 0.004)
        env.set_robot_targets(closed_arm, closed_finger)
        env.arm_q[1] = torch.full((7,), 1.5)
        first.deactivate(torch.tensor((False, True, False)), reason="stage-one-failure")
        next_active = first.active_mask
        next_arm = first.last_sent_arm_target
        next_finger = first.last_sent_finger_target

    with reset_tools._PerLaneTargetHold(env, next_active, next_arm, next_finger) as second:
        env.set_robot_targets(torch.full((3, 7), 9.0), torch.full((3, 2), 0.02))
        env.advance(2)

        assert torch.equal(second.active_mask, torch.tensor((True, False, True)))
        assert torch.equal(second.last_sent_arm_target[1], next_arm[1])
        assert torch.equal(second.last_sent_finger_target[1], next_finger[1])
        assert torch.equal(second.last_sent_arm_target[0], torch.full((7,), 9.0))
        assert torch.equal(second.last_sent_finger_target[0], torch.full((2,), 0.02))


def _exact_dwell_state(num_envs: int, body_count: int) -> dict[str, torch.Tensor]:
    return {
        "task_q": _identity_pose(num_envs, body_count),
        "task_qd": torch.zeros((num_envs, body_count, 6)),
        "arm_q": torch.zeros((num_envs, 7)),
        "arm_qd": torch.zeros((num_envs, 7)),
        "finger_q": torch.full((num_envs, 2), 0.005),
        "finger_qd": torch.zeros((num_envs, 2)),
    }


class _ExactDwellLaneFakeEnv:
    device = "cpu"
    advance_dt = 1.0
    _arm_joint_ids = torch.arange(7)

    def __init__(self, states: list[dict[str, torch.Tensor]]) -> None:
        self.states = states
        self.index = 0
        self.num_envs = len(states[0]["task_q"])
        self.layout = SimpleNamespace(
            body_count=states[0]["task_q"].shape[1],
            socket_body_index=0,
            plug_body_index=1,
            latch_body_index=2,
            cable_body_slice=slice(3, states[0]["task_q"].shape[1]),
        )
        self.rj45_runtime = SimpleNamespace(
            layout=self.layout,
            drive_enabled=torch.zeros(self.num_envs, dtype=torch.bool),
            orientation_hold_enabled=torch.zeros(self.num_envs, dtype=torch.bool),
        )
        self.cfg = SimpleNamespace(
            success_dwell_time_s=2.0,
            success_axial_tolerance=1.0e-3,
            success_axial_overtravel_tolerance=1.0e-3,
            success_radial_tolerance=1.0e-3,
            success_plug_angle_tolerance=0.1,
            success_latch_angle_tolerance=0.1,
            success_max_plug_speed=0.1,
            actions=SimpleNamespace(
                arm_action=SimpleNamespace(
                    joint_limit_margin=0.02,
                    tracking_error_limits=None,
                )
            ),
        )
        limits = torch.tensor([[[-1.0, 1.0]] * 7] * self.num_envs)
        self._robot = SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=SimpleNamespace(torch=limits)))
        self.current_arm_target = torch.zeros((self.num_envs, 7))
        self.current_finger_target = torch.zeros((self.num_envs, 2))
        self.physics_commands: list[tuple[torch.Tensor, torch.Tensor]] = []

    def read_task_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.states[self.index]
        return state["task_q"].clone(), state["task_qd"].clone()

    def read_robot_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.states[self.index]
        return tuple(state[name].clone() for name in ("arm_q", "arm_qd", "finger_q", "finger_qd"))

    def set_robot_targets(self, arm_target: torch.Tensor, finger_target: torch.Tensor) -> None:
        self.current_arm_target = arm_target.clone()
        self.current_finger_target = finger_target.clone()

    def advance(self, duration_s: float, update=None, *, post_step=None) -> int:
        step_count = int(duration_s / self.advance_dt)
        for step in range(step_count):
            if update is not None:
                update(step, step_count, (step + 1) / step_count)
            self.physics_commands.append((self.current_arm_target.clone(), self.current_finger_target.clone()))
            self.index += 1
            if post_step is not None:
                post_step(step, step_count, (step + 1) / step_count)
        return step_count


def test_exact_dwell_goal_gate_freezes_failed_lane_while_peer_completes(monkeypatch) -> None:
    states = [_exact_dwell_state(3, 4) for _ in range(3)]
    states[1]["arm_q"][0] = 0.25
    states[1]["arm_qd"][0] = 0.2
    states[1]["task_q"][0, 0, 0] = 0.02
    env = _ExactDwellLaneFakeEnv(states)
    monkeypatch.setattr(reset_tools.wp, "to_torch", lambda value: value)
    arm_target = torch.full((3, 7), 0.5)
    finger_target = torch.tensor(((0.004, 0.004), (0.04, 0.04), (0.004, 0.004)))
    snapshots: list[dict[str, Any]] = []

    def goal_gate(snapshot: dict[str, Any]) -> dict[str, torch.Tensor]:
        snapshots.append(snapshot)
        return {
            "arm-speed": snapshot["arm_joint_speed"] > 0.1,
            "socket-excursion": snapshot["socket_excursion"] > 0.01,
        }

    with reset_tools._PerLaneTargetHold(
        env,
        torch.tensor((True, False, True)),
        arm_target,
        finger_target,
    ) as lane_hold:
        passed, evidence = reset_tools.advance_exact_success_dwell(
            env,
            states[0]["task_q"],
            arm_target,
            finger_target,
            duration_s=2.0,
            require_all_samples=True,
            arm_target_is_absolute=True,
            lane_hold=lane_hold,
            per_step_lane_goal_gate=goal_gate,
        )

        assert passed.tolist() == [False, False, True]
        assert lane_hold.active_mask.tolist() == [False, False, True]
        assert evidence["lane_goal_gate_passed"].tolist() == [False, True, True]
        assert evidence["lane_goal_gate_violation_masks"]["arm-speed"].tolist() == [True, False, False]
        assert evidence["lane_goal_gate_violation_masks"]["socket-excursion"].tolist() == [True, False, False]
        assert evidence["lane_goal_gate_first_failure_steps"]["arm-speed"].tolist() == [1, -1, -1]
        assert evidence["lane_goal_gate_first_failure_steps"]["socket-excursion"].tolist() == [1, -1, -1]
        assert evidence["lane_hold_reason_masks"]["recovery-dwell-goal:arm-speed"].tolist() == [True, False, False]

    assert [snapshot["step"] for snapshot in snapshots] == [1, 2]
    assert snapshots[1]["active_mask"].tolist() == [False, False, True]
    second_arm_target, second_finger_target = env.physics_commands[1]
    assert torch.equal(second_arm_target[0], torch.full((7,), 0.25))
    assert torch.equal(second_arm_target[1], torch.zeros(7))
    assert torch.equal(second_arm_target[2], torch.full((7,), 0.5))
    assert torch.equal(second_finger_target[0], torch.full((2,), 0.004))
    assert torch.equal(second_finger_target[1], torch.full((2,), 0.04))
    assert torch.equal(second_finger_target[2], torch.full((2,), 0.004))


def test_exact_dwell_hard_failure_freezes_lane_before_peer_finishes(monkeypatch) -> None:
    states = [_exact_dwell_state(2, 4) for _ in range(3)]
    states[1]["arm_q"][0] = 0.3
    states[1]["task_q"][0, 1, 0] = 0.01
    env = _ExactDwellLaneFakeEnv(states)
    monkeypatch.setattr(reset_tools.wp, "to_torch", lambda value: value)
    arm_target = torch.full((2, 7), 0.5)
    finger_target = torch.full((2, 2), 0.004)

    with reset_tools._PerLaneTargetHold(
        env,
        torch.ones(2, dtype=torch.bool),
        arm_target,
        finger_target,
    ) as lane_hold:
        passed, evidence = reset_tools.advance_exact_success_dwell(
            env,
            states[0]["task_q"],
            arm_target,
            finger_target,
            duration_s=2.0,
            require_all_samples=True,
            arm_target_is_absolute=True,
            lane_hold=lane_hold,
        )

        assert passed.tolist() == [False, True]
        assert evidence["all_samples_success"].tolist() == [False, True]
        assert evidence["lane_hold_reason_masks"]["recovery-dwell-exact-success"].tolist() == [True, False]

    assert torch.equal(env.physics_commands[1][0][0], torch.full((7,), 0.3))
    assert torch.equal(env.physics_commands[1][0][1], torch.full((7,), 0.5))


def test_exact_dwell_goal_gate_validates_every_mask_before_deactivation(monkeypatch) -> None:
    states = [_exact_dwell_state(2, 4) for _ in range(3)]
    env = _ExactDwellLaneFakeEnv(states)
    monkeypatch.setattr(reset_tools.wp, "to_torch", lambda value: value)
    arm_target = torch.full((2, 7), 0.5)
    finger_target = torch.full((2, 2), 0.004)

    with reset_tools._PerLaneTargetHold(
        env,
        torch.ones(2, dtype=torch.bool),
        arm_target,
        finger_target,
    ) as lane_hold:
        with pytest.raises(TypeError, match="must have Boolean dtype"):
            reset_tools.advance_exact_success_dwell(
                env,
                states[0]["task_q"],
                arm_target,
                finger_target,
                duration_s=2.0,
                arm_target_is_absolute=True,
                lane_hold=lane_hold,
                per_step_lane_goal_gate=lambda _snapshot: {
                    "would-fail": torch.tensor((True, False)),
                    "malformed": torch.zeros(2),
                },
            )

        assert lane_hold.active_mask.tolist() == [True, True]
        assert "recovery-dwell-goal:would-fail" not in lane_hold.reason_masks


def test_exact_dwell_contact_overflow_remains_batch_fatal(monkeypatch) -> None:
    states = [_exact_dwell_state(2, 4) for _ in range(3)]
    env = _ExactDwellLaneFakeEnv(states)
    monkeypatch.setattr(reset_tools.wp, "to_torch", lambda value: value)
    monkeypatch.setattr(
        reset_tools,
        "_physical_validity_sample",
        lambda *_args, **_kwargs: (
            reset_tools.CollisionMetrics(
                valid=torch.ones(2, dtype=torch.bool),
                invalid_contact_count=torch.zeros(2, dtype=torch.long),
                grasp_contact_count=torch.full((2,), 2, dtype=torch.long),
                left_grasp_contact_count=torch.ones(2, dtype=torch.long),
                right_grasp_contact_count=torch.ones(2, dtype=torch.long),
                contact_overflow=True,
                invalid_contact_pairs=(),
            ),
            reset_tools.GraspMetrics(
                valid=torch.ones(2, dtype=torch.bool),
                tcp_distance=torch.zeros(2),
                bilateral_deflection=torch.ones(2),
            ),
            torch.ones(2, dtype=torch.bool),
        ),
    )
    arm_target = torch.full((2, 7), 0.5)
    finger_target = torch.full((2, 2), 0.004)

    with pytest.raises(RuntimeError, match="Global contact-buffer overflow"):
        with reset_tools._PerLaneTargetHold(
            env,
            torch.ones(2, dtype=torch.bool),
            arm_target,
            finger_target,
        ) as lane_hold:
            reset_tools.advance_exact_success_dwell(
                env,
                states[0]["task_q"],
                arm_target,
                finger_target,
                duration_s=2.0,
                sample_physical_validity=True,
                arm_target_is_absolute=True,
                lane_hold=lane_hold,
                per_step_lane_goal_gate=lambda _snapshot: {},
            )


def test_active_waypoint_count_ignores_inactive_nonfinite_distances_and_reports_active_invalid_lanes() -> None:
    count, invalid = reset_tools._active_waypoint_count(
        torch.tensor((float("nan"), 0.021, float("inf"), -1.0)),
        torch.tensor((False, True, True, False)),
        0.01,
    )

    assert count == 3
    assert torch.equal(invalid, torch.tensor((False, False, True, False)))


def test_active_waypoint_count_short_circuits_an_empty_active_batch() -> None:
    count, invalid = reset_tools._active_waypoint_count(
        torch.tensor((float("nan"), float("inf"))),
        torch.zeros(2, dtype=torch.bool),
        0.01,
    )

    assert count == 0
    assert not bool(invalid.any())


def test_runtime_reset_biased_arm_target_matches_live_soft_limit_clamp() -> None:
    """Replay zero actions with the same per-world soft-limit margin as training."""
    limits = torch.tensor(
        [
            [[-1.0, 1.0], [-0.5, 0.5]],
            [[-0.8, 0.8], [-0.4, 0.4]],
        ],
        dtype=torch.float32,
    )
    env = SimpleNamespace(
        device="cpu",
        num_envs=2,
        _arm_joint_ids=torch.tensor((0, 1)),
        cfg=SimpleNamespace(actions=SimpleNamespace(arm_action=SimpleNamespace(joint_limit_margin=0.02))),
        _robot=SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=SimpleNamespace(torch=limits))),
    )
    current = torch.tensor(((0.10, -0.20), (0.79, -0.39)))
    bias = torch.tensor(((0.05, 0.10), (0.10, -0.10)))

    target, clamp_delta = reset_tools.runtime_reset_biased_arm_target(env, current, bias)

    assert torch.allclose(target[0], torch.tensor((0.15, -0.10)))
    assert torch.allclose(target[1], torch.tensor((0.78, -0.38)))
    assert torch.allclose(clamp_delta, torch.tensor((0.0, 0.11)), atol=1.0e-7, rtol=0.0)


def test_runtime_reset_biased_arm_target_rejects_wrong_shape() -> None:
    env = SimpleNamespace(device="cpu", num_envs=2, _arm_joint_ids=(0, 1))

    with pytest.raises(ValueError, match="must both have shape"):
        reset_tools.runtime_reset_biased_arm_target(env, torch.zeros((1, 2)), torch.zeros((2, 2)))


def test_persistent_absolute_target_is_constant_while_measured_position_changes() -> None:
    limits = torch.tensor([[[-1.0, 1.0]] * 7])

    class FakeEnv:
        device = "cpu"
        num_envs = 1
        _arm_joint_ids = torch.arange(7)
        cfg = SimpleNamespace(actions=SimpleNamespace(arm_action=SimpleNamespace(joint_limit_margin=0.02)))
        _robot = SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=SimpleNamespace(torch=limits)))

        def __init__(self) -> None:
            self.states = iter((torch.zeros((1, 7)), torch.full((1, 7), 0.1), torch.full((1, 7), -0.1)))
            self.targets: list[torch.Tensor] = []

        def read_robot_state(self):
            arm_q = next(self.states)
            return arm_q, torch.zeros_like(arm_q), torch.zeros((1, 2)), torch.zeros((1, 2))

        def set_robot_targets(self, arm_target: torch.Tensor, _finger_target: torch.Tensor) -> None:
            self.targets.append(arm_target.clone())

        def advance(self, duration_s: float, update) -> int:
            assert duration_s == 2.0
            for step in range(2):
                update(step, 2, (step + 1) / 2)
            return 2

    env = FakeEnv()
    target = torch.full((1, 7), 0.2)
    evidence: dict[str, Any] = {}

    steps = reset_tools.advance_reset_absolute_target_hold(
        env,
        2.0,
        target,
        torch.zeros((1, 2)),
        clamp_evidence=evidence,
    )

    assert steps == 2
    assert len(env.targets) == 2
    assert all(torch.equal(command, target) for command in env.targets)
    assert torch.equal(evidence["maximum_arm_target_drift"], torch.zeros(1))
    assert evidence["any_arm_target_clamped"].tolist() == [False]


def test_joint_limit_mask_uses_live_margin_by_default_and_allows_override() -> None:
    limits = torch.tensor([[[-1.0, 1.0], [-0.5, 0.5]]])
    env = SimpleNamespace(
        _arm_joint_ids=torch.tensor((0, 1)),
        cfg=SimpleNamespace(actions=SimpleNamespace(arm_action=SimpleNamespace(joint_limit_margin=0.02))),
        _robot=SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=SimpleNamespace(torch=limits))),
    )
    arm_q = torch.tensor(((0.985, 0.0), (0.0, -0.485)))

    assert reset_tools.joint_limit_mask(env, arm_q).tolist() == [False, False]
    assert reset_tools.joint_limit_mask(env, arm_q, margin=0.005).tolist() == [True, True]


def test_advance_reset_bias_hold_accumulates_clamp_evidence_across_calls() -> None:
    class FakeEnv:
        device = "cpu"
        num_envs = 2
        _arm_joint_ids = torch.arange(7)
        cfg = SimpleNamespace(actions=SimpleNamespace(arm_action=SimpleNamespace(joint_limit_margin=0.02)))
        _robot = SimpleNamespace(
            data=SimpleNamespace(
                soft_joint_pos_limits=SimpleNamespace(torch=torch.tensor([[[-1.0, 1.0]] * 7, [[-1.0, 1.0]] * 7]))
            )
        )

        def __init__(self, arm_states: list[torch.Tensor]) -> None:
            self.arm_states = iter(arm_states)
            self.targets: list[torch.Tensor] = []

        def read_robot_state(self):
            arm_q = next(self.arm_states)
            return arm_q, torch.zeros_like(arm_q), torch.zeros((2, 2)), torch.zeros((2, 2))

        def set_robot_targets(self, arm_target: torch.Tensor, _finger_target: torch.Tensor) -> None:
            self.targets.append(arm_target.clone())

        def advance(self, duration_s: float, update) -> int:
            step_count = int(duration_s)
            for step in range(step_count):
                update(step, step_count, (step + 1) / step_count)
            return step_count

    bias = torch.tensor(((0.1,) * 7, (-0.1,) * 7))
    finger_target = torch.zeros((2, 2))
    evidence: dict[str, Any] = {}
    first_env = FakeEnv(
        [
            torch.tensor(((0.95,) * 7, (0.0,) * 7)),
            torch.tensor(((0.0,) * 7, (-0.95,) * 7)),
        ]
    )

    steps = reset_tools.advance_reset_bias_hold(
        first_env,
        1.0,
        bias,
        finger_target,
        clamp_evidence=evidence,
    )

    assert steps == 1
    assert len(first_env.targets) == 1
    assert torch.allclose(first_env.targets[0][:, 0], torch.tensor((0.1, -0.98)))
    assert torch.allclose(
        evidence["maximum_arm_target_clamp_delta"],
        torch.tensor((0.07, 0.07)),
        atol=1.0e-7,
        rtol=0.0,
    )
    assert evidence["any_arm_target_clamped"].tolist() == [True, True]

    second_env = FakeEnv(
        [
            torch.tensor(((0.0,) * 7, (-0.99,) * 7)),
            torch.tensor(((0.0,) * 7, (0.0,) * 7)),
        ]
    )
    reset_tools.advance_reset_bias_hold(
        second_env,
        1.0,
        bias,
        finger_target,
        clamp_evidence=evidence,
    )

    assert torch.allclose(
        evidence["maximum_arm_target_clamp_delta"],
        torch.tensor((0.07, 0.11)),
        atol=1.0e-7,
        rtol=0.0,
    )
    assert evidence["any_arm_target_clamped"].tolist() == [True, True]


def test_advance_reset_bias_hold_collects_initial_evidence_at_zero_duration() -> None:
    limits = torch.tensor([[[-1.0, 1.0]] * 7])

    class FakeEnv:
        device = "cpu"
        num_envs = 1
        _arm_joint_ids = torch.arange(7)
        cfg = SimpleNamespace(actions=SimpleNamespace(arm_action=SimpleNamespace(joint_limit_margin=0.02)))
        _robot = SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=SimpleNamespace(torch=limits)))

        def read_robot_state(self):
            arm_q = torch.tensor(((0.95,) * 7,))
            return arm_q, torch.zeros_like(arm_q), torch.zeros((1, 2)), torch.zeros((1, 2))

        def set_robot_targets(self, _arm_target: torch.Tensor, _finger_target: torch.Tensor) -> None:
            raise AssertionError("A zero-duration hold must not command a target.")

        def advance(self, duration_s: float, _update) -> int:
            assert duration_s == 0.0
            return 0

    evidence: dict[str, Any] = {}

    steps = reset_tools.advance_reset_bias_hold(
        FakeEnv(),
        0.0,
        torch.full((1, 7), 0.1),
        torch.zeros((1, 2)),
        clamp_evidence=evidence,
    )

    assert steps == 0
    assert torch.allclose(
        evidence["maximum_arm_target_clamp_delta"],
        torch.tensor((0.07,)),
        atol=1.0e-7,
        rtol=0.0,
    )
    assert evidence["any_arm_target_clamped"].tolist() == [True]


def test_reset_tool_advance_calls_post_step_after_scene_update() -> None:
    events: list[str] = []

    class FakeScene:
        def write_data_to_sim(self) -> None:
            events.append("write")

        def update(self, *, dt: float) -> None:
            assert dt == 0.1
            events.append("scene_update")

    class FakeSimulation:
        def step(self, *, render: bool) -> None:
            assert render is False
            events.append("sim_step")

    fake_env = SimpleNamespace(advance_dt=0.1, scene=FakeScene(), sim=FakeSimulation())

    steps = reset_tools._RJ45ResetToolMixin.advance(
        fake_env,
        0.2,
        lambda *_args: events.append("update"),
        post_step=lambda *_args: events.append("post_step"),
    )

    assert steps == 2
    assert events == [
        "update",
        "write",
        "sim_step",
        "scene_update",
        "post_step",
        "update",
        "write",
        "sim_step",
        "scene_update",
        "post_step",
    ]


def _mechanical_replay_state(
    *,
    num_envs: int,
    body_count: int,
    cable_slice: slice,
    arm_speed: tuple[float, ...],
    finger_speed: tuple[float, ...],
    cable_speed: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    task_q = _identity_pose(num_envs, body_count)
    task_qd = torch.zeros((num_envs, body_count, 6))
    arm_q = torch.zeros((num_envs, 7))
    arm_qd = torch.tensor(arm_speed)[:, None].expand(-1, 7).clone()
    finger_q = torch.full((num_envs, 2), 0.005)
    finger_qd = torch.tensor(finger_speed)[:, None].expand(-1, 2).clone()
    task_qd[:, cable_slice, 0] = torch.tensor(cable_speed)[:, None]
    return {
        "task_q": task_q,
        "task_qd": task_qd,
        "arm_q": arm_q,
        "arm_qd": arm_qd,
        "finger_q": finger_q,
        "finger_qd": finger_qd,
    }


class _MechanicalReplayEnv:
    device = "cpu"
    advance_dt = 1.0
    _arm_joint_ids = torch.arange(7)

    def __init__(
        self,
        states: list[dict[str, torch.Tensor]],
        *,
        layout: SimpleNamespace,
        drive_states: list[tuple[tuple[bool, ...], tuple[bool, ...]]] | None = None,
    ) -> None:
        self.states = states
        self.index = 0
        self.num_envs = len(states[0]["task_q"])
        self.targets: list[torch.Tensor] = []
        self.drive_states = drive_states or [((False,) * self.num_envs, (False,) * self.num_envs) for _ in states]
        limits = torch.tensor([[[-1.0, 1.0]] * 7] * self.num_envs)
        self._robot = SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=SimpleNamespace(torch=limits)))
        self.cfg = SimpleNamespace(
            actions=SimpleNamespace(arm_action=SimpleNamespace(joint_limit_margin=0.02)),
            max_tcp_grasp_distance=0.01,
        )
        self.rj45_runtime = SimpleNamespace(
            layout=layout,
            drive_enabled=torch.tensor(self.drive_states[0][0]),
            orientation_hold_enabled=torch.tensor(self.drive_states[0][1]),
        )

    def read_task_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.states[self.index]
        return state["task_q"].clone(), state["task_qd"].clone()

    def read_robot_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.states[self.index]
        return tuple(state[name].clone() for name in ("arm_q", "arm_qd", "finger_q", "finger_qd"))

    def set_robot_targets(self, arm_target: torch.Tensor, _finger_target: torch.Tensor) -> None:
        self.targets.append(arm_target.clone())

    def advance(self, duration_s: float, update, *, post_step=None) -> int:
        step_count = int(duration_s / self.advance_dt)
        for step in range(step_count):
            update(step, step_count, (step + 1) / step_count)
            self.index += 1
            translation, orientation = self.drive_states[self.index]
            self.rj45_runtime.drive_enabled = torch.tensor(translation)
            self.rj45_runtime.orientation_hold_enabled = torch.tensor(orientation)
            if post_step is not None:
                post_step(step, step_count, (step + 1) / step_count)
        return step_count


def test_reset_bias_hold_samples_mechanics_after_every_step_and_accumulates_continuously(monkeypatch) -> None:
    layout = SimpleNamespace(
        body_count=7,
        socket_body_index=1,
        plug_body_index=3,
        cable_body_slice=slice(4, 7),
    )
    states = [
        _mechanical_replay_state(
            num_envs=2,
            body_count=layout.body_count,
            cable_slice=layout.cable_body_slice,
            arm_speed=(0.1, 0.2),
            finger_speed=(0.01, 0.02),
            cable_speed=(0.03, 0.04),
        ),
        _mechanical_replay_state(
            num_envs=2,
            body_count=layout.body_count,
            cable_slice=layout.cable_body_slice,
            arm_speed=(0.3, 0.4),
            finger_speed=(0.05, 0.06),
            cable_speed=(0.07, 0.08),
        ),
        _mechanical_replay_state(
            num_envs=2,
            body_count=layout.body_count,
            cable_slice=layout.cable_body_slice,
            arm_speed=(0.5, 0.6),
            finger_speed=(0.09, 0.10),
            cable_speed=(0.11, 0.12),
        ),
        _mechanical_replay_state(
            num_envs=2,
            body_count=layout.body_count,
            cable_slice=layout.cable_body_slice,
            arm_speed=(0.2, 0.1),
            finger_speed=(0.03, 0.02),
            cable_speed=(0.04, 0.03),
        ),
    ]
    states[1]["task_q"][:, 1, 0] = torch.tensor((0.01, 0.02))
    states[1]["task_q"][:, 3, 0] = torch.tensor((0.02, 0.01))
    states[2]["task_q"][:, 0, 0] = torch.tensor((0.04, 0.03))
    states[2]["task_q"][:, 1, 0] = torch.tensor((0.03, 0.04))
    states[2]["task_q"][:, 3, 0] = torch.tensor((0.05, 0.02))
    states[3]["task_q"][:, 0, 0] = torch.tensor((0.02, 0.01))
    drive_states = [
        ((False, False), (False, False)),
        ((False, False), (False, False)),
        ((False, False), (False, False)),
        ((False, False), (False, False)),
    ]
    env = _MechanicalReplayEnv(states, layout=layout, drive_states=drive_states)
    collisions = [
        reset_tools.CollisionMetrics(
            valid=torch.tensor((True, True)),
            invalid_contact_count=torch.tensor((0, 0)),
            grasp_contact_count=torch.tensor((2, 2)),
            left_grasp_contact_count=torch.tensor((1, 1)),
            right_grasp_contact_count=torch.tensor((1, 1)),
            contact_overflow=False,
            invalid_contact_pairs=(),
        ),
        reset_tools.CollisionMetrics(
            valid=torch.tensor((True, False)),
            invalid_contact_count=torch.tensor((0, 2)),
            grasp_contact_count=torch.tensor((2, 1)),
            left_grasp_contact_count=torch.tensor((1, 1)),
            right_grasp_contact_count=torch.tensor((1, 0)),
            contact_overflow=False,
            invalid_contact_pairs=("world=1 robot <-> table",),
        ),
        reset_tools.CollisionMetrics(
            valid=torch.tensor((True, True)),
            invalid_contact_count=torch.tensor((0, 0)),
            grasp_contact_count=torch.tensor((3, 2)),
            left_grasp_contact_count=torch.tensor((2, 1)),
            right_grasp_contact_count=torch.tensor((1, 1)),
            contact_overflow=False,
            invalid_contact_pairs=(),
        ),
    ]
    grasps = [torch.tensor((True, True)), torch.tensor((True, False)), torch.tensor((True, True))]
    requested_bilateral: list[bool] = []
    requested_retaining: list[bool] = []

    def fake_collision_metrics(fake_env, *, require_bilateral_grasp):
        requested_bilateral.append(require_bilateral_grasp)
        return collisions[fake_env.index - 1]

    def fake_grasp_metrics(fake_env, _finger_target, *, retaining_grasp):
        requested_retaining.append(retaining_grasp)
        return reset_tools.GraspMetrics(
            valid=grasps[fake_env.index - 1],
            tcp_distance=torch.zeros(fake_env.num_envs),
            bilateral_deflection=torch.ones(fake_env.num_envs),
        )

    monkeypatch.setattr(reset_tools, "collision_metrics", fake_collision_metrics)
    monkeypatch.setattr(reset_tools, "grasp_metrics", fake_grasp_metrics)
    evidence: dict[str, Any] = {}
    bias = torch.zeros((2, 7))
    finger_target = torch.full((2, 2), 0.004)

    first_steps = reset_tools.advance_reset_bias_hold(
        env,
        1.0,
        bias,
        finger_target,
        replay_evidence=evidence,
        starts_grasped=True,
    )
    second_steps = reset_tools.advance_reset_bias_hold(
        env,
        2.0,
        bias,
        finger_target,
        replay_evidence=evidence,
        starts_grasped=True,
    )

    assert (first_steps, second_steps) == (1, 2)
    assert requested_bilateral == [False, False, False]
    assert requested_retaining == [True, True, True]
    assert evidence["post_step_samples"] == 3
    assert torch.allclose(evidence["stored_maximum_arm_joint_speed_rad_s"], torch.tensor((0.1, 0.2)))
    assert torch.allclose(evidence["stored_maximum_finger_joint_speed_m_s"], torch.tensor((0.01, 0.02)))
    assert torch.allclose(evidence["stored_maximum_cable_speed_m_s"], torch.tensor((0.03, 0.04)))
    assert torch.allclose(evidence["maximum_post_step_arm_joint_speed_rad_s"], torch.tensor((0.5, 0.6)))
    assert torch.allclose(evidence["maximum_post_step_finger_joint_speed_m_s"], torch.tensor((0.09, 0.10)))
    assert torch.allclose(evidence["maximum_post_step_cable_speed_m_s"], torch.tensor((0.11, 0.12)))
    assert torch.allclose(evidence["final_arm_joint_speed_rad_s"], torch.tensor((0.2, 0.1)))
    assert torch.allclose(evidence["maximum_body_excursion_m"], torch.tensor((0.05, 0.04)))
    assert torch.allclose(evidence["maximum_plug_excursion_m"], torch.tensor((0.05, 0.02)))
    assert torch.allclose(evidence["maximum_socket_excursion_m"], torch.tensor((0.03, 0.04)))
    assert evidence["all_post_step_collision_free"].tolist() == [True, False]
    assert evidence["all_post_step_bilateral_grasp"].tolist() == [True, False]
    assert evidence["all_post_step_proxy_bilateral_contact"].tolist() == [True, False]
    assert evidence["all_post_step_expected_contact_state"].tolist() == [True, False]
    assert evidence["all_post_step_drive_disabled"].tolist() == [True, True]
    assert evidence["minimum_left_proxy_contact_count"].tolist() == [1, 1]
    assert evidence["minimum_right_proxy_contact_count"].tolist() == [1, 0]
    assert evidence["maximum_left_proxy_contact_count"].tolist() == [2, 1]
    assert evidence["maximum_invalid_contact_count"].tolist() == [0, 2]
    assert evidence["any_contact_overflow"] is False
    assert evidence["invalid_contact_pairs"] == ("world=1 robot <-> table",)


@pytest.mark.parametrize("global_failure", ("contact-overflow", "construction-drive"))
def test_reset_bias_hold_propagates_global_runtime_failures(monkeypatch, global_failure: str) -> None:
    layout = SimpleNamespace(body_count=5, socket_body_index=0, plug_body_index=1, cable_body_slice=slice(2, 5))
    states = [
        _mechanical_replay_state(
            num_envs=2,
            body_count=layout.body_count,
            cable_slice=layout.cable_body_slice,
            arm_speed=(0.0, 0.0),
            finger_speed=(0.0, 0.0),
            cable_speed=(0.0, 0.0),
        )
        for _ in range(2)
    ]
    drive_states = None
    if global_failure == "construction-drive":
        drive_states = [
            ((False, False), (False, False)),
            ((True, False), (False, False)),
        ]
    env = _MechanicalReplayEnv(states, layout=layout, drive_states=drive_states)
    collision = reset_tools.CollisionMetrics(
        valid=torch.ones(2, dtype=torch.bool),
        invalid_contact_count=torch.zeros(2, dtype=torch.long),
        grasp_contact_count=torch.full((2,), 2, dtype=torch.long),
        left_grasp_contact_count=torch.ones(2, dtype=torch.long),
        right_grasp_contact_count=torch.ones(2, dtype=torch.long),
        contact_overflow=global_failure == "contact-overflow",
        invalid_contact_pairs=(),
    )
    monkeypatch.setattr(reset_tools, "collision_metrics", lambda *_args, **_kwargs: collision)
    monkeypatch.setattr(
        reset_tools,
        "grasp_metrics",
        lambda *_args, **_kwargs: reset_tools.GraspMetrics(
            valid=torch.ones(2, dtype=torch.bool),
            tcp_distance=torch.zeros(2),
            bilateral_deflection=torch.ones(2),
        ),
    )

    expected = "contact-buffer overflow" if global_failure == "contact-overflow" else "construction drive"
    with pytest.raises(RuntimeError, match=expected):
        reset_tools.advance_reset_bias_hold(
            env,
            1.0,
            torch.zeros((2, 7)),
            torch.full((2, 2), 0.004),
            replay_evidence={},
            starts_grasped=True,
        )


def test_reset_bias_hold_requires_zero_proxy_contacts_for_open_rows(monkeypatch) -> None:
    layout = SimpleNamespace(body_count=5, socket_body_index=0, plug_body_index=1, cable_body_slice=slice(2, 5))
    states = [
        _mechanical_replay_state(
            num_envs=2,
            body_count=layout.body_count,
            cable_slice=layout.cable_body_slice,
            arm_speed=(0.0, 0.0),
            finger_speed=(0.0, 0.0),
            cable_speed=(0.0, 0.0),
        )
        for _ in range(3)
    ]
    env = _MechanicalReplayEnv(states, layout=layout)
    contact_counts = [torch.tensor((0, 0)), torch.tensor((0, 1))]

    def fake_collision_metrics(fake_env, *, require_bilateral_grasp):
        assert require_bilateral_grasp is False
        right = contact_counts[fake_env.index - 1]
        zeros = torch.zeros(fake_env.num_envs, dtype=torch.long)
        return reset_tools.CollisionMetrics(
            valid=torch.ones(fake_env.num_envs, dtype=torch.bool),
            invalid_contact_count=zeros,
            grasp_contact_count=right,
            left_grasp_contact_count=zeros,
            right_grasp_contact_count=right,
            contact_overflow=False,
            invalid_contact_pairs=(),
        )

    monkeypatch.setattr(reset_tools, "collision_metrics", fake_collision_metrics)
    monkeypatch.setattr(
        reset_tools,
        "grasp_metrics",
        lambda fake_env, _finger_target, **_kwargs: reset_tools.GraspMetrics(
            valid=torch.zeros(fake_env.num_envs, dtype=torch.bool),
            tcp_distance=torch.ones(fake_env.num_envs),
            bilateral_deflection=torch.zeros(fake_env.num_envs),
        ),
    )
    evidence: dict[str, Any] = {}

    reset_tools.advance_reset_bias_hold(
        env,
        2.0,
        torch.zeros((2, 7)),
        torch.full((2, 2), 0.018),
        replay_evidence=evidence,
        starts_grasped=False,
    )

    assert evidence["all_post_step_zero_proxy_contacts"].tolist() == [True, False]
    assert evidence["all_post_step_expected_contact_state"].tolist() == [True, False]


def test_reset_bias_hold_rejects_discontinuous_evidence_reuse(monkeypatch) -> None:
    layout = SimpleNamespace(body_count=5, socket_body_index=None, plug_body_index=0, cable_body_slice=slice(2, 5))
    states = [
        _mechanical_replay_state(
            num_envs=1,
            body_count=layout.body_count,
            cable_slice=layout.cable_body_slice,
            arm_speed=(0.0,),
            finger_speed=(0.0,),
            cable_speed=(0.0,),
        )
        for _ in range(2)
    ]
    env = _MechanicalReplayEnv(states, layout=layout)
    monkeypatch.setattr(
        reset_tools,
        "collision_metrics",
        lambda fake_env, **_kwargs: reset_tools.CollisionMetrics(
            valid=torch.ones(fake_env.num_envs, dtype=torch.bool),
            invalid_contact_count=torch.zeros(fake_env.num_envs, dtype=torch.long),
            grasp_contact_count=torch.zeros(fake_env.num_envs, dtype=torch.long),
            left_grasp_contact_count=torch.zeros(fake_env.num_envs, dtype=torch.long),
            right_grasp_contact_count=torch.zeros(fake_env.num_envs, dtype=torch.long),
            contact_overflow=False,
            invalid_contact_pairs=(),
        ),
    )
    monkeypatch.setattr(
        reset_tools,
        "grasp_metrics",
        lambda fake_env, _finger_target, **_kwargs: reset_tools.GraspMetrics(
            valid=torch.zeros(fake_env.num_envs, dtype=torch.bool),
            tcp_distance=torch.ones(fake_env.num_envs),
            bilateral_deflection=torch.zeros(fake_env.num_envs),
        ),
    )
    evidence: dict[str, Any] = {}
    bias = torch.zeros((1, 7))
    finger_target = torch.full((1, 2), 0.018)
    reset_tools.advance_reset_bias_hold(
        env,
        1.0,
        bias,
        finger_target,
        replay_evidence=evidence,
        starts_grasped=False,
    )
    env.states[env.index]["task_q"][0, 0, 0] = 0.001

    with pytest.raises(ValueError, match="uninterrupted calls"):
        reset_tools.advance_reset_bias_hold(
            env,
            0.0,
            bias,
            finger_target,
            replay_evidence=evidence,
            starts_grasped=False,
        )


def test_reset_bias_hold_records_nonfinite_stored_robot_speed_as_failed_evidence() -> None:
    layout = SimpleNamespace(body_count=5, socket_body_index=None, plug_body_index=0, cable_body_slice=slice(2, 5))
    state = _mechanical_replay_state(
        num_envs=1,
        body_count=layout.body_count,
        cable_slice=layout.cable_body_slice,
        arm_speed=(float("inf"),),
        finger_speed=(0.0,),
        cable_speed=(0.0,),
    )
    env = _MechanicalReplayEnv([state], layout=layout)
    evidence: dict[str, Any] = {}

    steps = reset_tools.advance_reset_bias_hold(
        env,
        0.0,
        torch.zeros((1, 7)),
        torch.full((1, 2), 0.018),
        replay_evidence=evidence,
        starts_grasped=False,
    )

    assert steps == 0
    assert evidence["stored_state_finite"].tolist() == [False]
    assert torch.isinf(evidence["stored_maximum_arm_joint_speed_rad_s"]).all()


def test_ik_seed_scores_follow_explicit_continuation_reference() -> None:
    """Prefer the local chained branch when equally valid multi-seed lanes differ."""
    costs = torch.zeros((1, 2))
    arm_q = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])

    near_first = reset_tools._ik_seed_scores(costs, arm_q, torch.tensor([[0.1, 0.1]]))
    near_second = reset_tools._ik_seed_scores(costs, arm_q, torch.tensor([[0.9, 0.9]]))

    assert near_first.argmin(dim=-1).tolist() == [0]
    assert near_second.argmin(dim=-1).tolist() == [1]


def test_grasp_metrics_use_live_acquisition_and_retention_axis_tolerances() -> None:
    """Preserve the live axis-only predicate as a legacy tool fallback."""

    class FakeEnv:
        num_envs = 2
        device = "cpu"
        cfg = SimpleNamespace(
            max_tcp_grasp_distance=0.02,
            grasp_acquisition_axis_tolerance_rad=math.radians(15.0),
            grasp_retention_axis_tolerance_rad=math.radians(25.0),
        )

        def __init__(self) -> None:
            self.calls: list[tuple[float, torch.Tensor]] = []

        def read_robot_state(self):
            return torch.zeros((2, 7)), torch.zeros((2, 7)), torch.full((2, 2), 0.005), torch.zeros((2, 2))

        def tcp_pose_e(self):
            return _identity_pose(2, 1)[:, 0]

        def plug_grasp_position_e(self):
            return torch.zeros((2, 3))

        def grasp_axis_alignment_mask(self, tolerance, *, tcp_orientation_xyzw):
            self.calls.append((float(tolerance), tcp_orientation_xyzw.clone()))
            return torch.full((2,), float(tolerance) >= math.radians(20.0), dtype=torch.bool)

    env = FakeEnv()
    finger_target = torch.full((2, 2), 0.004)

    acquiring = reset_tools.grasp_metrics(env, finger_target)
    retaining = reset_tools.grasp_metrics(env, finger_target, retaining_grasp=True)

    assert acquiring.valid.tolist() == [False, False]
    assert retaining.valid.tolist() == [True, True]
    assert [call[0] for call in env.calls] == pytest.approx((math.radians(15.0), math.radians(25.0)))
    assert all(torch.equal(call[1], _identity_pose(2, 1)[:, 0, 3:7]) for call in env.calls)


def test_grasp_metrics_prefer_live_coupled_contact_alignment_predicate() -> None:
    """Certify pick-insert grasps through its authoritative combined predicate."""

    class FakeEnv:
        num_envs = 2
        device = "cpu"
        cfg = SimpleNamespace(
            max_tcp_grasp_distance=0.02,
            grasp_acquisition_axis_tolerance_rad=math.radians(15.0),
            grasp_retention_axis_tolerance_rad=math.radians(25.0),
        )

        def __init__(self) -> None:
            self.calls: list[float] = []

        def read_robot_state(self):
            return torch.zeros((2, 7)), torch.zeros((2, 7)), torch.full((2, 2), 0.005), torch.zeros((2, 2))

        def tcp_pose_e(self):
            return _identity_pose(2, 1)[:, 0]

        def plug_grasp_position_e(self):
            return torch.zeros((2, 3))

        def grasp_contact_alignment_mask(self, tolerance, *, tcp_orientation_xyzw):
            assert torch.equal(tcp_orientation_xyzw, _identity_pose(2, 1)[:, 0, 3:7])
            self.calls.append(float(tolerance))
            return torch.tensor([False, True])

        def grasp_axis_alignment_mask(self, _tolerance, *, tcp_orientation_xyzw):
            raise AssertionError(f"Legacy predicate unexpectedly called with {tcp_orientation_xyzw!r}.")

    env = FakeEnv()
    metrics = reset_tools.grasp_metrics(env, torch.full((2, 2), 0.004))

    assert metrics.valid.tolist() == [False, True]
    assert env.calls == pytest.approx([math.radians(15.0)])


def test_physical_validity_separates_collision_and_live_proxy_contact(monkeypatch) -> None:
    """Keep collision validity separate and delegate intended contact semantics to runtime."""
    collision = reset_tools.CollisionMetrics(
        valid=torch.tensor((True, False)),
        invalid_contact_count=torch.tensor((0, 1)),
        grasp_contact_count=torch.tensor((2, 1)),
        left_grasp_contact_count=torch.tensor((1, 1)),
        right_grasp_contact_count=torch.tensor((1, 0)),
        contact_overflow=False,
        invalid_contact_pairs=("world=1 robot <-> table",),
    )
    grasp = reset_tools.GraspMetrics(
        valid=torch.tensor((True, True)),
        tcp_distance=torch.zeros(2),
        bilateral_deflection=torch.ones(2),
    )
    requested_bilateral: list[bool] = []

    def fake_collision_metrics(_env, *, require_bilateral_grasp):
        requested_bilateral.append(require_bilateral_grasp)
        return collision

    monkeypatch.setattr(reset_tools, "collision_metrics", fake_collision_metrics)
    requested_retaining: list[bool] = []

    def fake_grasp_metrics(_env, _finger_target, *, retaining_grasp):
        requested_retaining.append(retaining_grasp)
        return grasp

    monkeypatch.setattr(reset_tools, "grasp_metrics", fake_grasp_metrics)
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        bilateral_grasp_proxy_contact_mask=lambda: torch.tensor((False, True)),
    )

    sampled_collision, sampled_grasp, bilateral_proxy = reset_tools._physical_validity_sample(
        env,
        torch.zeros((2, 2)),
    )

    assert requested_bilateral == [False]
    assert requested_retaining == [True]
    assert sampled_collision is collision
    assert sampled_grasp is grasp
    assert sampled_collision.valid.tolist() == [True, False]
    assert bilateral_proxy.tolist() == [False, True]


def test_reset_tool_environments_use_sibling_mixin_inheritance() -> None:
    """Keep each offline tool paired with its real environment without a diamond MRO."""
    assert issubclass(RJ45ResetToolEnv, FrankaRJ45InsertionEnv)
    assert issubclass(RJ45PickInsertResetToolEnv, FrankaRJ45PickInsertEnv)
    assert not issubclass(RJ45PickInsertResetToolEnv, RJ45ResetToolEnv)
    assert RJ45ResetToolEnv.__mro__[1] is RJ45PickInsertResetToolEnv.__mro__[1]


def test_pick_reset_tool_scopes_grasp_proxy_friction_to_its_builder() -> None:
    """Thread the production pick-only value into the filtered grasp proxy."""
    tool = object.__new__(RJ45PickInsertResetToolEnv)
    tool._is_closed = True
    tool._reset_tool_grasp_proxy_friction = PICK_INSERT_GRASP_PROXY_FRICTION
    cfg = SimpleNamespace(
        resettable_socket=True,
        free_plug_rotation=True,
        extra_cable_segments=10,
        include_task_support_plane=False,
        task_translation=(0.58, 0.15, 0.0),
        task_rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    builder = tool._create_rj45_builder(cfg)
    tool._rj45_builder = builder

    assert builder.grasp_proxy_friction == PICK_INSERT_GRASP_PROXY_FRICTION
    assert tool.grasp_proxy_friction == PICK_INSERT_GRASP_PROXY_FRICTION


def test_pick_reset_tool_constructor_accepts_only_the_configured_production_proxy_friction(monkeypatch) -> None:
    """Keep the former diagnostic constructor argument as an exact assertion only."""
    initialized: list[object] = []

    def fake_init(self, cfg, *_args, **_kwargs):
        self.cfg = cfg
        self._is_closed = True
        initialized.append(self)

    monkeypatch.setattr(FrankaRJ45PickInsertEnv, "__init__", fake_init)
    cfg = SimpleNamespace(grasp_proxy_friction=PICK_INSERT_GRASP_PROXY_FRICTION)

    default_tool = RJ45PickInsertResetToolEnv(cfg)
    asserted_tool = RJ45PickInsertResetToolEnv(
        cfg,
        grasp_proxy_friction=PICK_INSERT_GRASP_PROXY_FRICTION,
    )

    assert default_tool._reset_tool_grasp_proxy_friction == PICK_INSERT_GRASP_PROXY_FRICTION
    assert asserted_tool._reset_tool_grasp_proxy_friction == PICK_INSERT_GRASP_PROXY_FRICTION
    assert initialized == [default_tool, asserted_tool]
    with pytest.raises(ValueError, match="may only assert the production value"):
        RJ45PickInsertResetToolEnv(cfg, grasp_proxy_friction=2.0)
    with pytest.raises(ValueError, match="exact production grasp-proxy friction"):
        RJ45PickInsertResetToolEnv(SimpleNamespace(grasp_proxy_friction=2.0))


def test_pick_reset_tool_physical_contract_reports_live_production_values() -> None:
    """Derive every promoted reset-tool value from the live config and builder."""
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            actions=SimpleNamespace(gripper_action=SimpleNamespace(close_position=PICK_INSERT_CLOSED_FINGER_POSITION)),
            grasp_proxy_friction=PICK_INSERT_GRASP_PROXY_FRICTION,
            success_max_plug_speed=PICK_INSERT_SUCCESS_MAX_PLUG_SPEED,
        ),
        grasp_proxy_friction=PICK_INSERT_GRASP_PROXY_FRICTION,
    )

    contract = reset_tools.pick_insert_tool_physical_contract(
        env,
        finger_closed_target=PICK_INSERT_CLOSED_FINGER_POSITION,
    )

    assert contract == {
        "finger_closed_target_m": PICK_INSERT_CLOSED_FINGER_POSITION,
        "live_finger_close_position_m": PICK_INSERT_CLOSED_FINGER_POSITION,
        "configured_grasp_proxy_raw_friction": PICK_INSERT_GRASP_PROXY_FRICTION,
        "live_grasp_proxy_raw_friction": PICK_INSERT_GRASP_PROXY_FRICTION,
        "effective_finger_proxy_friction": PICK_INSERT_EFFECTIVE_GRASP_FRICTION,
        "success_max_plug_speed": PICK_INSERT_SUCCESS_MAX_PLUG_SPEED,
    }


@pytest.mark.parametrize(
    ("path", "invalid"),
    (
        ("finger_closed_target", 0.001),
        ("live_finger_close_position", 0.001),
        ("configured_grasp_proxy_friction", 2.0),
        ("live_grasp_proxy_friction", 2.0),
        ("success_max_plug_speed", PICK_INSERT_SUCCESS_MAX_PLUG_SPEED / 2.0),
    ),
)
def test_pick_reset_tool_physical_contract_rejects_divergent_values(path: str, invalid: float) -> None:
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            actions=SimpleNamespace(gripper_action=SimpleNamespace(close_position=PICK_INSERT_CLOSED_FINGER_POSITION)),
            grasp_proxy_friction=PICK_INSERT_GRASP_PROXY_FRICTION,
            success_max_plug_speed=PICK_INSERT_SUCCESS_MAX_PLUG_SPEED,
        ),
        grasp_proxy_friction=PICK_INSERT_GRASP_PROXY_FRICTION,
    )
    finger_closed_target = PICK_INSERT_CLOSED_FINGER_POSITION
    if path == "finger_closed_target":
        finger_closed_target = invalid
    elif path == "live_finger_close_position":
        env.cfg.actions.gripper_action.close_position = invalid
    elif path == "configured_grasp_proxy_friction":
        env.cfg.grasp_proxy_friction = invalid
    elif path == "live_grasp_proxy_friction":
        env.grasp_proxy_friction = invalid
    else:
        env.cfg.success_max_plug_speed = invalid

    with pytest.raises(ValueError, match="immutable production physical contract"):
        reset_tools.pick_insert_tool_physical_contract(env, finger_closed_target=finger_closed_target)


def test_legacy_identity_fixed_goal_error_is_bitwise_unchanged() -> None:
    """Preserve the original fixed-goal arithmetic for the legacy identity task."""
    goal = _identity_pose(1, 3)[0]
    task = goal.unsqueeze(0).repeat(2, 1, 1)
    task[:, 0, :3] = torch.tensor(((0.01, 0.02, 0.03), (-0.04, -0.05, 0.06)))
    plug_error = task[:, 0, :3] - goal[None, 0, :3]
    expected = plug_error[:, 1].abs() + 2.0 * torch.linalg.vector_norm(plug_error[:, (0, 2)], dim=-1)

    actual = scalar_goal_error(task, goal)

    assert torch.equal(actual, expected)


def test_batched_rotated_goals_and_layout_indices_use_goal_local_axes() -> None:
    """Resolve plug/latch after a socket prefix and measure translation in each goal frame."""
    goal = _identity_pose(2, 4)
    half_yaw = torch.tensor(torch.pi / 4)
    goal[1, 1, 3:7] = torch.tensor((0.0, 0.0, torch.sin(half_yaw), torch.cos(half_yaw)))
    goal[:, 2, 3:7] = goal[:, 1, 3:7]
    task = goal.clone()
    local_error = torch.tensor(((0.01, 0.02, 0.03), (-0.02, -0.04, 0.01)))
    task[:, 1, :3] += math_utils.quat_apply(goal[:, 1, 3:7], local_error)
    expected = local_error[:, 1].abs() + 2.0 * torch.linalg.vector_norm(local_error[:, (0, 2)], dim=-1)

    actual = scalar_goal_error(task, goal, plug_body_index=1, latch_body_index=2)
    latch_angle = plug_relative_latch_angle(task, plug_body_index=1, latch_body_index=2)

    assert torch.allclose(actual, expected, atol=1.0e-7, rtol=0.0)
    assert torch.equal(latch_angle, torch.zeros_like(latch_angle))


def test_exact_success_defaults_to_runtime_layout_for_batched_goals() -> None:
    """Forward topology indices and per-world goal frames to the shared success predicate."""
    layout = SimpleNamespace(body_count=4, socket_body_index=0, plug_body_index=1, latch_body_index=2)
    cfg = SimpleNamespace(
        success_axial_tolerance=8.0e-4,
        success_axial_overtravel_tolerance=2.0e-4,
        success_radial_tolerance=7.5e-4,
        success_plug_angle_tolerance=0.05,
        success_latch_angle_tolerance=0.05,
        success_max_plug_speed=0.01,
    )
    env = SimpleNamespace(cfg=cfg, rj45_runtime=SimpleNamespace(layout=layout))
    goal = _identity_pose(2, layout.body_count)
    half_yaw = torch.tensor(torch.pi / 4)
    goal[1, 1, 3:7] = torch.tensor((0.0, 0.0, torch.sin(half_yaw), torch.cos(half_yaw)))
    goal[:, 2, 3:7] = goal[:, 1, 3:7]
    task = goal.clone()
    local_error = torch.tensor(((0.0, 5.0e-4, 0.0), (0.0, -5.0e-4, 0.0)))
    task[:, 1, :3] += math_utils.quat_apply(goal[:, 1, 3:7], local_error)
    velocity = torch.zeros((2, layout.body_count, 6))

    result = exact_success_from_state(env, task, velocity, goal)

    assert result.mask.tolist() == [False, True]
    assert torch.allclose(result.signed_axial_error, local_error[:, 1], atol=1.0e-7, rtol=0.0)


def test_scripted_recovery_composes_goal_orientation_and_local_overtravel(monkeypatch) -> None:
    """Build each IK target from the per-world goal frame rather than world Y."""
    layout = SimpleNamespace(body_count=4, socket_body_index=0, plug_body_index=1, latch_body_index=2)
    grasp_orientation = (0.5, 0.5, 0.5, -0.5)
    cfg = SimpleNamespace(
        plug_grasp_offset=(0.0, -0.025, 0.0),
        plug_grasp_orientation_xyzw=grasp_orientation,
    )
    goal = _identity_pose(2, layout.body_count)
    goal[:, 1, :3] = torch.tensor(((0.5, 0.1, 0.2), (0.6, -0.1, 0.25)))
    half_yaw = torch.tensor(torch.pi / 4)
    goal[1, 1, 3:7] = torch.tensor((0.0, 0.0, torch.sin(half_yaw), torch.cos(half_yaw)))
    goal[:, 2, 3:7] = goal[:, 1, 3:7]
    task = goal.clone()
    behind_local = torch.tensor(((0.0, -0.008, 0.0), (0.0, -0.008, 0.0)))
    task[:, 1, :3] += math_utils.quat_apply(goal[:, 1, 3:7], behind_local)

    class FakeEnv:
        num_envs = 2
        device = "cpu"
        rj45_runtime = SimpleNamespace(layout=layout)

        def __init__(self) -> None:
            self.cfg = cfg

        def set_drive(self, _enabled: bool) -> None:
            pass

        def read_robot_state(self):
            return torch.zeros((2, 7)), torch.zeros((2, 7)), torch.zeros((2, 2)), torch.zeros((2, 2))

        def read_task_state(self):
            return task.clone(), torch.zeros((2, layout.body_count, 6))

        def set_robot_targets(self, _arm_q: torch.Tensor, _finger_target: torch.Tensor) -> None:
            pass

        def advance(self, _duration_s: float) -> int:
            return 1

        def tcp_pose_e(self) -> torch.Tensor:
            return torch.zeros((2, 7))

    class FakeIK:
        def __init__(self) -> None:
            self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

        def solve(self, position, orientation, _finger_position, *, arm_seed=None):
            del arm_seed
            self.calls.append((position.clone(), orientation.clone()))
            zeros = torch.zeros((2, 7))
            return reset_tools.IKResult(
                arm_q=zeros,
                tcp_position=position,
                tcp_quaternion=orientation,
                valid=torch.ones(2, dtype=torch.bool),
                position_residual=torch.zeros(2),
                rotation_residual=torch.zeros(2),
            )

    monkeypatch.setattr(reset_tools, "joint_limit_mask", lambda _env, arm_q: torch.ones(len(arm_q), dtype=torch.bool))
    monkeypatch.setattr(reset_tools, "interpolate_arm_motion", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        reset_tools,
        "advance_exact_success_dwell",
        lambda *_args, **_kwargs: (
            torch.ones(2, dtype=torch.bool),
            {
                "last_arm_target": torch.zeros((2, 7)),
                "all_samples_collision_free": torch.ones(2, dtype=torch.bool),
                "all_samples_bilateral_grasp": torch.ones(2, dtype=torch.bool),
                "all_samples_finite": torch.ones(2, dtype=torch.bool),
                "maximum_body_excursion": torch.zeros(2),
                "maximum_cable_linear_speed": torch.zeros(2),
            },
        ),
    )
    monkeypatch.setattr(
        reset_tools,
        "grasp_metrics",
        lambda *_args, **_kwargs: reset_tools.GraspMetrics(
            valid=torch.ones(2, dtype=torch.bool),
            tcp_distance=torch.zeros(2),
            bilateral_deflection=torch.ones(2),
        ),
    )
    monkeypatch.setattr(
        reset_tools,
        "collision_metrics",
        lambda *_args, **_kwargs: reset_tools.CollisionMetrics(
            valid=torch.ones(2, dtype=torch.bool),
            invalid_contact_count=torch.zeros(2, dtype=torch.long),
            grasp_contact_count=torch.ones(2, dtype=torch.long),
            left_grasp_contact_count=torch.ones(2, dtype=torch.long),
            right_grasp_contact_count=torch.ones(2, dtype=torch.long),
            contact_overflow=False,
            invalid_contact_pairs=(),
        ),
    )
    monkeypatch.setattr(
        reset_tools,
        "task_state_is_finite_and_normalized",
        lambda task_q, _task_qd: torch.ones(len(task_q), dtype=torch.bool),
    )
    ik = FakeIK()

    reset_tools.scripted_recovery(
        FakeEnv(),
        ik,
        goal,
        torch.tensor((0.0, 0.0, 0.0, 1.0)).expand(2, -1),
        torch.zeros((2, 2)),
        compensation_max_iterations=0,
    )

    goal_plug = goal[:, 1]
    grasp_offset = torch.tensor(cfg.plug_grasp_offset).expand(2, -1)
    expected_goal_position = goal_plug[:, :3] + math_utils.quat_apply(goal_plug[:, 3:7], grasp_offset)
    overtravel_local = torch.tensor(((0.0, 0.002, 0.0), (0.0, 0.002, 0.0)))
    expected_overtravel = expected_goal_position + math_utils.quat_apply(goal_plug[:, 3:7], overtravel_local)
    expected_orientation = math_utils.quat_unique(
        math_utils.quat_mul(goal_plug[:, 3:7], torch.tensor(grasp_orientation).expand(2, -1))
    )

    assert len(ik.calls) == 2
    assert torch.allclose(ik.calls[0][0], expected_overtravel, atol=1.0e-7, rtol=0.0)
    assert torch.allclose(ik.calls[1][0], expected_goal_position, atol=1.0e-7, rtol=0.0)
    assert torch.allclose(ik.calls[0][1], expected_orientation, atol=1.0e-7, rtol=0.0)
    assert torch.allclose(ik.calls[1][1], expected_orientation, atol=1.0e-7, rtol=0.0)


@pytest.mark.parametrize("phase", (2, 3, 4, 5))
def test_pick_insert_recovery_clearance_route_stops_at_goal_without_overtravel(phase: int) -> None:
    live = _identity_pose(1, 1)[:, 0]
    goal = live.clone()
    live[:, :3] = torch.tensor(((0.4, -0.1, 0.01),))
    goal[:, :3] = torch.tensor(((0.6, 0.1, 0.0),))

    route, preflight = reset_tools._pick_insert_recovery_plug_route(
        live,
        goal,
        phase=phase,
    )

    assert preflight.tolist() == [True]
    assert len(route) == 5
    expected_positions = torch.tensor(
        (
            (0.4, -0.1, 0.10),
            (0.5, -0.04, 0.10),
            (0.6, 0.02, 0.10),
            (0.6, 0.02, 0.0),
            (0.6, 0.1, 0.0),
        )
    )
    assert torch.allclose(torch.stack([pose[0, :3] for pose in route]), expected_positions, atol=1.0e-7, rtol=0.0)
    assert torch.equal(route[-1], goal)


@pytest.mark.parametrize("phase", (0, 1))
def test_pick_insert_recovery_insertion_corridor_stops_at_goal_and_preserves_preflight(phase: int) -> None:
    goal = _identity_pose(3, 1)[:, 0]
    live = goal.clone()
    live[:, 1] = torch.tensor((-0.099, -0.101, -0.05))
    live[:, 0] = torch.tensor((0.0, 0.0, 0.0061))

    route, preflight = reset_tools._pick_insert_recovery_plug_route(
        live,
        goal,
        phase=phase,
    )

    assert len(route) == 1
    assert torch.equal(route[0], goal)
    assert preflight.tolist() == [True, False, False]


def test_pick_insert_cartesian_recovery_reports_zero_overtravel_and_anchors_goal_endpoint(monkeypatch) -> None:
    class GoalStopEnv:
        num_envs = 1
        device = "cpu"
        advance_dt = 0.1

        def __init__(self) -> None:
            self.cfg = SimpleNamespace(
                plug_grasp_offset=(0.0, 0.0, 0.0),
                plug_grasp_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
            self.rj45_runtime = SimpleNamespace(
                layout=SimpleNamespace(
                    body_count=3,
                    plug_body_index=0,
                    latch_body_index=1,
                    cable_body_slice=slice(2, 3),
                )
            )
            self.task_q = _identity_pose(1, 3)
            self.task_q[:, 0, 1] = -0.03
            self.task_qd = torch.zeros((1, 3, 6))
            self.arm_q = torch.zeros((1, 7))
            self.last_arm_target = self.arm_q.clone()

        def set_drive(self, _enabled: bool) -> None:
            pass

        def read_task_state(self):
            return self.task_q.clone(), self.task_qd.clone()

        def read_robot_state(self):
            return self.arm_q.clone(), torch.zeros_like(self.arm_q), torch.zeros((1, 2)), torch.zeros((1, 2))

        def tcp_pose_e(self) -> torch.Tensor:
            return self.task_q[:, 0].clone()

        def set_robot_targets(self, arm_target: torch.Tensor, _finger_target: torch.Tensor) -> None:
            self.last_arm_target = arm_target.clone()

        def advance(self, duration_s: float, update=None, *, post_step=None) -> int:
            steps = max(1, int(torch.ceil(torch.tensor(duration_s / self.advance_dt)).item()))
            for step in range(steps):
                progress = (step + 1) / steps
                if update is not None:
                    update(step, steps, progress)
                if post_step is not None:
                    post_step(step, steps, progress)
            return steps

    env = GoalStopEnv()
    goal_q = _identity_pose(1, 3)
    goal_arm_target = torch.full((1, 7), 0.1)
    finger_target = torch.zeros((1, 2))
    planner_calls: list[dict[str, Any]] = []

    def plan_route(
        _env,
        _ik,
        tcp_targets,
        _finger_target,
        *,
        current_tcp,
        current_raw,
        current_target,
        lane_hold,
        reason_prefix,
        endpoint_arm_target,
        active_mask=None,
    ):
        del current_tcp, active_mask
        planner_calls.append(
            {
                "tcp_targets": tuple(target.clone() for target in tcp_targets),
                "endpoint_arm_target": endpoint_arm_target.clone(),
                "reason_prefix": reason_prefix,
            }
        )
        terminal_target = endpoint_arm_target.clone()
        knots = torch.stack((current_target, terminal_target))
        zero = torch.zeros(env.num_envs)
        zero_long = torch.zeros(env.num_envs, dtype=torch.long)
        zero_joint = torch.zeros_like(current_target)
        return reset_tools._RecoveryCartesianPlan(
            segment_knots=(knots,),
            segment_waypoint_counts=(1,),
            pre_densification_waypoint_count=torch.ones_like(zero_long),
            post_densification_waypoint_count=torch.ones_like(zero_long),
            terminal_raw=current_raw.clone(),
            terminal_target=terminal_target,
            start_preload_bias=zero_joint,
            goal_preload_bias=terminal_target - current_raw,
            preload_bias_difference=terminal_target - current_raw,
            maximum_raw_ik_joint_step=zero,
            maximum_commanded_joint_step_before_densification=zero,
            maximum_commanded_joint_step_after_densification=zero,
            command_densification_required_subknot_count=zero_long,
            command_densification_executed_subknot_count=zero_long,
            start_target_anchor_error=zero,
            canonical_endpoint_anchor_error=zero,
            maximum_segment_boundary_command_jump=zero,
            ik_valid=lane_hold.active_mask,
            target_valid=lane_hold.active_mask,
        )

    def execute_plan(_env, plan, sent_finger_target, lane_hold, _evidence, *, requested_duration_s):
        del requested_duration_s
        env.set_robot_targets(plan.terminal_target, sent_finger_target)
        return lane_hold.last_sent_arm_target

    false_mask = torch.zeros(1, dtype=torch.bool)
    dwell_metrics = {
        "last_arm_target": goal_arm_target.clone(),
        "all_samples_collision_free": torch.ones(1, dtype=torch.bool),
        "all_samples_bilateral_grasp": torch.ones(1, dtype=torch.bool),
        "all_samples_finite": torch.ones(1, dtype=torch.bool),
        "all_samples_arm_target_tracking_bounded": torch.ones(1, dtype=torch.bool),
        "maximum_arm_target_drift": torch.zeros(1),
        "dwell_satisfied": torch.ones(1, dtype=torch.bool),
        "lane_goal_gate_violation_masks": {
            reason: false_mask.clone()
            for reason in ("plug-linear-speed", "plug-angular-speed", "arm-joint-speed", "finger-joint-speed")
        },
        "lane_goal_gate_first_failure_steps": {
            reason: torch.full((1,), -1, dtype=torch.long)
            for reason in ("plug-linear-speed", "plug-angular-speed", "arm-joint-speed", "finger-joint-speed")
        },
    }
    collision = reset_tools.CollisionMetrics(
        valid=torch.ones(1, dtype=torch.bool),
        invalid_contact_count=torch.zeros(1, dtype=torch.long),
        grasp_contact_count=torch.ones(1, dtype=torch.long),
        left_grasp_contact_count=torch.ones(1, dtype=torch.long),
        right_grasp_contact_count=torch.ones(1, dtype=torch.long),
        contact_overflow=False,
        invalid_contact_pairs=(),
    )
    grasp = reset_tools.GraspMetrics(
        valid=torch.ones(1, dtype=torch.bool),
        tcp_distance=torch.zeros(1),
        bilateral_deflection=torch.ones(1),
    )
    monkeypatch.setattr(reset_tools, "_runtime_drives_disabled", lambda _env: torch.ones(1, dtype=torch.bool))
    monkeypatch.setattr(reset_tools, "_plan_recovery_cartesian_route", plan_route)
    monkeypatch.setattr(reset_tools, "_execute_recovery_cartesian_plan", execute_plan)
    monkeypatch.setattr(reset_tools, "_sample_recovery_cartesian_motion", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(reset_tools, "scalar_goal_error", lambda *_args, **_kwargs: torch.zeros(1))
    monkeypatch.setattr(
        reset_tools,
        "advance_exact_success_dwell",
        lambda *_args, **_kwargs: (torch.ones(1, dtype=torch.bool), dwell_metrics),
    )
    monkeypatch.setattr(reset_tools, "grasp_metrics", lambda *_args, **_kwargs: grasp)
    monkeypatch.setattr(reset_tools, "collision_metrics", lambda *_args, **_kwargs: collision)
    monkeypatch.setattr(
        reset_tools,
        "task_state_is_finite_and_normalized",
        lambda *_args, **_kwargs: torch.ones(1, dtype=torch.bool),
    )

    with reset_tools._PerLaneTargetHold(
        env,
        torch.ones(1, dtype=torch.bool),
        env.last_arm_target,
        finger_target,
    ) as hold:
        success, metrics = reset_tools._scripted_recovery_cartesian_c2(
            env,
            SimpleNamespace(),
            goal_q,
            None,
            finger_target,
            arm_target_start=env.last_arm_target,
            goal_arm_target=goal_arm_target,
            motion_s=2.0,
            settle_s=0.5,
            compensation_max_iterations=0,
            compensation_gain=1.0,
            compensation_max_step_m=0.006,
            compensation_motion_s=0.35,
            compensation_hold_s=0.25,
            compensation_tolerance_m=0.0015,
            plug_body_index=0,
            latch_body_index=1,
            arm_target_is_absolute=True,
            lane_hold=hold,
            pick_insert_phase=1,
        )

    assert success.tolist() == [True]
    assert len(planner_calls) == 1
    assert planner_calls[0]["reason_prefix"] == "recovery-cartesian-route"
    assert len(planner_calls[0]["tcp_targets"]) == 1
    assert torch.equal(planner_calls[0]["tcp_targets"][0], goal_q[:, 0])
    assert torch.equal(planner_calls[0]["endpoint_arm_target"], goal_arm_target)
    assert torch.equal(metrics["overtravel_distance"], torch.zeros(1))
    assert torch.equal(env.last_arm_target, goal_arm_target)
    assert metrics["used_canonical_goal_arm_target"].tolist() == [True]


def test_recovery_cartesian_c2_schedule_is_symmetric_monotone_and_flat_at_endpoints() -> None:
    progress = [index / 100 for index in range(101)]
    path = [reset_tools._recovery_cartesian_c2_progress(value) for value in progress]

    assert path[0] == 0.0
    assert path[-1] == 1.0
    assert all(left <= right for left, right in zip(path[:-1], path[1:], strict=True))
    assert all(abs(left + right - 1.0) <= 1.0e-12 for left, right in zip(path, reversed(path), strict=True))
    epsilon = 1.0e-6
    assert reset_tools._recovery_cartesian_c2_progress(epsilon) / epsilon < 1.0e-6
    assert (1.0 - reset_tools._recovery_cartesian_c2_progress(1.0 - epsilon)) / epsilon < 1.0e-6


def test_recovery_command_schedule_blends_bias_over_global_unique_knots() -> None:
    raw_first = torch.zeros((3, 1, 2), dtype=torch.float64)
    raw_first[:, 0, 0] = torch.tensor((0.0, 0.01, 0.02), dtype=torch.float64)
    raw_second = torch.zeros((3, 1, 2), dtype=torch.float64)
    raw_second[:, 0, 0] = torch.tensor((0.02, 0.03, 0.04), dtype=torch.float64)
    current_target = torch.tensor(((0.03, -0.02),), dtype=torch.float64)
    endpoint_target = torch.tensor(((0.11, 0.02),), dtype=torch.float64)

    schedule = reset_tools._recovery_bias_blended_command_schedule(
        (raw_first, raw_second),
        current_target,
        endpoint_target,
        torch.tensor((True,)),
        maximum_command_step_rad=0.1,
    )

    assert torch.equal(schedule.segment_knots[0][0], current_target)
    assert torch.equal(schedule.segment_knots[-1][-1], endpoint_target)
    assert torch.equal(schedule.segment_knots[0][-1], schedule.segment_knots[1][0])
    expected_start_bias = current_target - raw_first[0]
    expected_goal_bias = endpoint_target - raw_second[-1]
    expected_seam = raw_first[-1] + torch.lerp(expected_start_bias, expected_goal_bias, 0.5)
    assert torch.allclose(schedule.segment_knots[0][-1], expected_seam, atol=1.0e-12, rtol=0.0)
    assert torch.equal(schedule.start_preload_bias, expected_start_bias)
    assert torch.equal(schedule.goal_preload_bias, expected_goal_bias)
    assert schedule.pre_densification_waypoint_count.tolist() == [4]
    assert schedule.post_densification_waypoint_count.tolist() == [4]
    assert schedule.maximum_segment_boundary_jump.tolist() == [0.0]


def test_recovery_command_schedule_densifies_over_limit_but_not_inclusive_boundary() -> None:
    raw = torch.zeros((2, 1, 1), dtype=torch.float64)
    current_target = torch.zeros((1, 1), dtype=torch.float64)

    dense = reset_tools._recovery_bias_blended_command_schedule(
        (raw,),
        current_target,
        torch.tensor(((0.05,),), dtype=torch.float64),
        torch.tensor((True,)),
        maximum_command_step_rad=0.02,
    )
    boundary = reset_tools._recovery_bias_blended_command_schedule(
        (raw,),
        current_target,
        torch.tensor(((0.02,),), dtype=torch.float64),
        torch.tensor((True,)),
        maximum_command_step_rad=0.02,
    )

    assert dense.segment_waypoint_counts == (3,)
    assert dense.required_subknot_count.tolist() == [2]
    assert dense.executed_subknot_count.tolist() == [2]
    assert dense.post_densification_waypoint_count.tolist() == [3]
    assert dense.maximum_step_before_densification.tolist() == [0.05]
    assert dense.maximum_step_after_densification.item() <= 0.02
    assert torch.equal(dense.segment_knots[-1][-1], torch.tensor(((0.05,),), dtype=torch.float64))
    assert boundary.segment_waypoint_counts == (1,)
    assert boundary.required_subknot_count.tolist() == [0]
    assert boundary.executed_subknot_count.tolist() == [0]
    assert boundary.maximum_step_after_densification.tolist() == [0.02]


def test_recovery_compensation_schedule_retains_constant_start_bias() -> None:
    raw = torch.tensor((((0.0,),), ((0.01,),), ((0.02,),)), dtype=torch.float64)
    current_target = torch.tensor(((0.03,),), dtype=torch.float64)

    schedule = reset_tools._recovery_bias_blended_command_schedule(
        (raw,),
        current_target,
        None,
        torch.tensor((True,)),
        maximum_command_step_rad=0.02,
    )

    expected_bias = torch.tensor(((0.03,),), dtype=torch.float64)
    assert torch.equal(schedule.start_preload_bias, expected_bias)
    assert torch.equal(schedule.goal_preload_bias, expected_bias)
    assert torch.equal(schedule.preload_bias_difference, torch.zeros_like(expected_bias))
    assert torch.equal(
        schedule.segment_knots[0][:, 0, 0],
        torch.tensor((0.03, 0.04, 0.05), dtype=torch.float64),
    )


class _CartesianPlannerFakeEnv:
    num_envs = 3
    device = "cpu"
    advance_dt = 0.2
    _arm_joint_ids = tuple(range(7))

    def __init__(self) -> None:
        limits = torch.tensor(((-2.0, 2.0),) * 7).unsqueeze(0)
        self._robot = SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=SimpleNamespace(torch=limits)))
        self.measured_arm = torch.zeros((self.num_envs, 7))
        self.sent_targets: list[torch.Tensor] = []
        self.last_target = torch.zeros_like(self.measured_arm)
        self.applied_steps: list[torch.Tensor] = []

    def read_robot_state(self):
        return (
            self.measured_arm.clone(),
            torch.zeros_like(self.measured_arm),
            torch.zeros((self.num_envs, 2)),
            torch.zeros((self.num_envs, 2)),
        )

    def set_robot_targets(self, arm_target: torch.Tensor, _finger_target: torch.Tensor) -> None:
        self.last_target = arm_target.clone()
        self.sent_targets.append(arm_target.clone())

    def advance(self, duration_s: float, update=None, *, post_step=None) -> int:
        steps = max(1, int(torch.ceil(torch.tensor(duration_s / self.advance_dt)).item()))
        for step in range(steps):
            progress = (step + 1) / steps
            if update is not None:
                update(step, steps, progress)
            self.applied_steps.append(self.last_target.clone())
            if post_step is not None:
                post_step(step, steps, progress)
        return steps


class _FiniteRecordingIK:
    def __init__(self) -> None:
        self.inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def solve(self, position, orientation, _finger_target, *, arm_seed=None):
        assert torch.isfinite(position).all()
        assert torch.isfinite(orientation).all()
        assert torch.isfinite(arm_seed).all()
        self.inputs.append((position.clone(), orientation.clone(), arm_seed.clone()))
        arm_q = arm_seed.clone()
        arm_q[:, 0] = 0.1 * position[:, 0]
        return reset_tools.IKResult(
            arm_q=arm_q,
            tcp_position=position.clone(),
            tcp_quaternion=orientation.clone(),
            valid=torch.ones(len(position), dtype=torch.bool),
            position_residual=torch.zeros(len(position)),
            rotation_residual=torch.zeros(len(position)),
        )


def test_cartesian_planner_finite_substitutes_inactive_lanes_and_holds_nonparticipants() -> None:
    env = _CartesianPlannerFakeEnv()
    finger_target = torch.zeros((env.num_envs, 2))
    current_target = torch.zeros((env.num_envs, 7))
    current_raw = current_target.clone()
    current_raw[1] = torch.nan
    current_tcp = _identity_pose(env.num_envs, 1)[:, 0]
    current_tcp[1] = torch.nan
    target_tcp = current_tcp.clone()
    target_tcp[0, 0] = 0.004
    target_tcp[1] = torch.nan
    target_tcp[2, 0] = 0.25
    ik = _FiniteRecordingIK()
    hold = reset_tools._PerLaneTargetHold(
        env,
        torch.tensor((True, False, True)),
        current_target,
        finger_target,
    )

    plan = reset_tools._plan_recovery_cartesian_route(
        env,
        ik,
        (target_tcp,),
        finger_target,
        current_tcp=current_tcp,
        current_raw=current_raw,
        current_target=current_target,
        lane_hold=hold,
        reason_prefix="test-cartesian",
        endpoint_arm_target=current_target,
        active_mask=torch.tensor((True, False, False)),
    )

    assert ik.inputs
    assert all(torch.isfinite(value).all() for call in ik.inputs for value in call)
    assert plan.segment_waypoint_counts == (2,)
    assert torch.equal(plan.segment_knots[0][0], current_target)
    assert torch.equal(plan.segment_knots[-1][-1], current_target)
    assert plan.start_target_anchor_error.tolist() == [0.0, 0.0, 0.0]
    assert plan.canonical_endpoint_anchor_error.tolist() == [0.0, 0.0, 0.0]
    assert torch.equal(plan.segment_knots[0][:, 1], torch.zeros_like(plan.segment_knots[0][:, 1]))
    assert torch.equal(plan.segment_knots[0][:, 2], torch.zeros_like(plan.segment_knots[0][:, 2]))
    stacked = torch.zeros((4, env.num_envs, 7))
    assert reset_tools.joint_limit_mask(env, stacked, margin=0.02).shape == (4, env.num_envs)
    assert reset_tools.joint_limit_mask(env, stacked, margin=0.02).all(dim=0).shape == (env.num_envs,)


def test_cartesian_planner_caps_post_densification_route_before_execution(monkeypatch) -> None:
    env = _CartesianPlannerFakeEnv()
    finger_target = torch.zeros((env.num_envs, 2))
    current_target = torch.zeros((env.num_envs, 7))
    current_tcp = _identity_pose(env.num_envs, 1)[:, 0]
    target_tcp = current_tcp.clone()
    target_tcp[0, 0] = 0.002
    endpoint_target = current_target.clone()
    endpoint_target[0, 0] = 0.05
    hold = reset_tools._PerLaneTargetHold(
        env,
        torch.tensor((True, False, False)),
        current_target,
        finger_target,
    )
    monkeypatch.setitem(
        reset_tools.PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["planning"],
        "maximum_waypoints",
        2,
    )

    plan = reset_tools._plan_recovery_cartesian_route(
        env,
        _FiniteRecordingIK(),
        (target_tcp,),
        finger_target,
        current_tcp=current_tcp,
        current_raw=current_target,
        current_target=current_target,
        lane_hold=hold,
        reason_prefix="test-cartesian",
        endpoint_arm_target=endpoint_target,
        active_mask=torch.tensor((True, False, False)),
    )

    assert plan.pre_densification_waypoint_count.tolist() == [1, 0, 0]
    assert plan.post_densification_waypoint_count.tolist() == [3, 0, 0]
    assert plan.segment_waypoint_counts == (1,)
    assert plan.command_densification_required_subknot_count.tolist() == [2, 0, 0]
    assert plan.command_densification_executed_subknot_count.tolist() == [0, 0, 0]
    assert plan.target_valid.tolist() == [False, True, True]
    assert not bool(hold.active_mask.any())
    assert hold.reason_masks["test-cartesian-command-waypoint-cap"].tolist() == [True, False, False]
    assert env.applied_steps == []


def test_cartesian_executor_freezes_failed_lane_on_the_next_physics_step(monkeypatch) -> None:
    env = _CartesianPlannerFakeEnv()
    env.measured_arm[1] = 0.123
    finger_target = torch.zeros((env.num_envs, 2))
    initial_target = torch.zeros((env.num_envs, 7))
    knots = torch.zeros((3, env.num_envs, 7))
    knots[1, :, 0] = torch.tensor((0.01, 0.02, 0.03))
    knots[2, :, 0] = torch.tensor((0.02, 0.04, 0.06))
    plan = reset_tools._RecoveryCartesianPlan(
        segment_knots=(knots,),
        segment_waypoint_counts=(2,),
        pre_densification_waypoint_count=torch.full((env.num_envs,), 2, dtype=torch.long),
        post_densification_waypoint_count=torch.full((env.num_envs,), 2, dtype=torch.long),
        terminal_raw=torch.zeros_like(initial_target),
        terminal_target=knots[-1],
        start_preload_bias=torch.zeros_like(initial_target),
        goal_preload_bias=torch.zeros_like(initial_target),
        preload_bias_difference=torch.zeros_like(initial_target),
        maximum_raw_ik_joint_step=torch.zeros(env.num_envs),
        maximum_commanded_joint_step_before_densification=torch.zeros(env.num_envs),
        maximum_commanded_joint_step_after_densification=torch.zeros(env.num_envs),
        command_densification_required_subknot_count=torch.zeros(env.num_envs, dtype=torch.long),
        command_densification_executed_subknot_count=torch.zeros(env.num_envs, dtype=torch.long),
        start_target_anchor_error=torch.zeros(env.num_envs),
        canonical_endpoint_anchor_error=torch.zeros(env.num_envs),
        maximum_segment_boundary_command_jump=torch.zeros(env.num_envs),
        ik_valid=torch.ones(env.num_envs, dtype=torch.bool),
        target_valid=torch.ones(env.num_envs, dtype=torch.bool),
    )
    sample_count = 0

    def fail_lane_after_first_step(_env, _finger_target, hold, _evidence) -> None:
        nonlocal sample_count
        sample_count += 1
        if sample_count == 1:
            hold.deactivate(torch.tensor((False, True, False)), reason="injected-physical-failure")

    monkeypatch.setattr(reset_tools, "_sample_recovery_cartesian_motion", fail_lane_after_first_step)
    with reset_tools._PerLaneTargetHold(
        env,
        torch.ones(env.num_envs, dtype=torch.bool),
        initial_target,
        finger_target,
    ) as hold:
        reset_tools._execute_recovery_cartesian_plan(
            env,
            plan,
            finger_target,
            hold,
            {},
            requested_duration_s=0.4,
        )

    assert len(env.applied_steps) >= 2
    assert not torch.equal(env.applied_steps[0][1], env.measured_arm[1])
    assert torch.equal(env.applied_steps[1][1], env.measured_arm[1])
    assert all(torch.equal(target[1], env.measured_arm[1]) for target in env.applied_steps[1:])


def test_cartesian_motion_evidence_attributes_each_speed_component(monkeypatch) -> None:
    class MotionEvidenceEnv(_CartesianPlannerFakeEnv):
        num_envs = 4
        advance_dt = 0.1

        def __init__(self) -> None:
            super().__init__()
            self.rj45_runtime = SimpleNamespace(layout=SimpleNamespace(plug_body_index=0, cable_body_slice=slice(1, 2)))
            self.task_q = _identity_pose(self.num_envs, 2)
            self.task_qd = torch.zeros((self.num_envs, 2, 6))
            self.arm_qd = torch.zeros((self.num_envs, 7))
            self.finger_qd = torch.zeros((self.num_envs, 2))

        def read_task_state(self):
            return self.task_q.clone(), self.task_qd.clone()

        def read_robot_state(self):
            return (
                self.measured_arm.clone(),
                self.arm_qd.clone(),
                torch.zeros((self.num_envs, 2)),
                self.finger_qd.clone(),
            )

    env = MotionEvidenceEnv()
    speed_gates = reset_tools.PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["per_step_rejection_gates"]
    env.task_qd[0, 0, 0] = float(speed_gates["plug_linear_speed_m_s"]) + 0.001
    env.task_qd[1, 0, 3] = float(speed_gates["plug_angular_speed_rad_s"]) + 0.001
    env.arm_qd[2, 0] = float(speed_gates["arm_joint_speed_rad_s"]) + 0.001
    env.finger_qd[3, 0] = float(speed_gates["finger_joint_speed_m_s"]) + 0.001
    valid = torch.ones(env.num_envs, dtype=torch.bool)
    collision = reset_tools.CollisionMetrics(
        valid=valid,
        invalid_contact_count=torch.zeros(env.num_envs, dtype=torch.long),
        grasp_contact_count=torch.ones(env.num_envs, dtype=torch.long),
        left_grasp_contact_count=torch.ones(env.num_envs, dtype=torch.long),
        right_grasp_contact_count=torch.ones(env.num_envs, dtype=torch.long),
        contact_overflow=False,
        invalid_contact_pairs=(),
    )
    grasp = reset_tools.GraspMetrics(
        valid=valid,
        tcp_distance=torch.zeros(env.num_envs),
        bilateral_deflection=torch.ones(env.num_envs),
    )
    monkeypatch.setattr(reset_tools, "_physical_validity_sample", lambda *_args: (collision, grasp, valid))
    monkeypatch.setattr(reset_tools, "_runtime_drives_disabled", lambda _env: valid)
    monkeypatch.setattr(
        reset_tools,
        "task_state_is_finite_and_normalized",
        lambda *_args: valid,
    )
    finger_target = torch.zeros((env.num_envs, 2))
    evidence = reset_tools._initialize_recovery_cartesian_motion_evidence(env)
    evidence["_current_segment"] = 7
    evidence["_current_knot"] = 3

    with reset_tools._PerLaneTargetHold(
        env,
        valid,
        torch.zeros((env.num_envs, 7)),
        finger_target,
    ) as hold:
        reset_tools._sample_recovery_cartesian_motion(env, finger_target, hold, evidence)

        assert not bool(hold.active_mask.any())
        expected_reasons = (
            "recovery-cartesian-motion-plug-linear-speed",
            "recovery-cartesian-motion-plug-angular-speed",
            "recovery-cartesian-motion-arm-joint-speed",
            "recovery-cartesian-motion-finger-joint-speed",
        )
        assert tuple(hold.reason_masks) == expected_reasons
        for lane, component in enumerate(
            ("plug_linear_speed", "plug_angular_speed", "arm_joint_speed", "finger_joint_speed")
        ):
            expected_mask = torch.zeros(env.num_envs, dtype=torch.bool)
            expected_mask[lane] = True
            assert torch.equal(evidence[f"first_{component}_failure_mask"], expected_mask)
            assert evidence[f"first_{component}_failure_step"][lane].item() == 1
            assert evidence[f"first_{component}_failure_segment"][lane].item() == 7
            assert evidence[f"first_{component}_failure_knot"][lane].item() == 3
            assert evidence[f"first_{component}_failure_time_s"][lane].item() == pytest.approx(0.1)
