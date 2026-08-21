# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical seed and rigid transforms for continuous RJ45 play resets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import resources
from typing import Any

import torch

from isaaclab.utils import math as math_utils

from .reset_dataset_io import reset_dataset_content_digest

PLAY_RESET_SEED_FORMAT = "isaaclab-franka-rj45-pick-insert-play-reset-seed"
PLAY_RESET_SEED_SCHEMA_VERSION = 1
PLAY_RESET_SEED_CONTENT_SHA256 = "856418384a5bde453d73da12a0cd4f16bbe1839222811dcff49148a8a353f35d"

_PLAY_RESET_SEED_RESOURCE = "data/play_loose_cable_seed.pt"
_PLAY_RESET_STATE_NAMES = (
    "task_body_pose",
    "task_body_previous_pose",
    "task_body_coupling_previous_pose",
    "task_body_velocity",
    "goal_task_body_pose",
)


def _validate_sha256(value: Any, *, name: str) -> str:
    """Return one valid lowercase hexadecimal SHA-256 digest."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _validate_play_reset_seed(
    payload: Any,
    *,
    expected_task_body_order: Sequence[str],
    expected_physics_contract: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Validate an untrusted safely loaded play-reset seed."""
    if not isinstance(payload, Mapping):
        raise TypeError("RJ45 play-reset seed must be a mapping.")
    if set(payload) != {"format", "schema_version", "metadata", "state", "content_sha256"}:
        raise ValueError("RJ45 play-reset seed has unexpected or missing top-level fields.")
    if payload.get("format") != PLAY_RESET_SEED_FORMAT:
        raise ValueError(f"Expected RJ45 play-reset seed format {PLAY_RESET_SEED_FORMAT!r}.")
    if payload.get("schema_version") != PLAY_RESET_SEED_SCHEMA_VERSION:
        raise ValueError(f"Expected RJ45 play-reset seed schema version {PLAY_RESET_SEED_SCHEMA_VERSION}.")
    content_sha256 = _validate_sha256(payload.get("content_sha256"), name="play-reset seed content_sha256")
    if content_sha256 != PLAY_RESET_SEED_CONTENT_SHA256:
        raise ValueError("RJ45 play-reset seed content digest does not match the packaged task asset.")
    if content_sha256 != reset_dataset_content_digest(payload):
        raise ValueError("RJ45 play-reset seed content digest does not match its payload.")

    metadata = payload.get("metadata")
    expected_metadata_names = {
        "source_dataset_content_sha256",
        "source_row_id",
        "source_phase",
        "task_body_order",
        "pose_frame",
        "velocity_frame",
        "physics_contract",
    }
    if not isinstance(metadata, Mapping) or set(metadata) != expected_metadata_names:
        raise ValueError("RJ45 play-reset seed has unexpected or missing metadata fields.")
    _validate_sha256(metadata.get("source_dataset_content_sha256"), name="source dataset content_sha256")
    if type(metadata.get("source_row_id")) is not int or metadata["source_row_id"] < 0:
        raise ValueError("RJ45 play-reset seed source_row_id must be a non-negative plain integer.")
    if metadata.get("source_phase") != 5:
        raise ValueError("RJ45 play-reset seed must come from the full-pick phase.")
    task_body_order = metadata.get("task_body_order")
    if tuple(task_body_order) != tuple(expected_task_body_order):
        raise ValueError("RJ45 play-reset seed task-body order does not match the live task.")
    if metadata.get("pose_frame") != "environment-local-xyzw":
        raise ValueError("RJ45 play-reset seed uses an unsupported pose frame.")
    if metadata.get("velocity_frame") != "world-linear-angular":
        raise ValueError("RJ45 play-reset seed uses an unsupported velocity frame.")
    if metadata.get("physics_contract") != expected_physics_contract:
        raise ValueError("RJ45 play-reset seed physics contract does not match the live task.")

    state = payload.get("state")
    if not isinstance(state, Mapping) or set(state) != set(_PLAY_RESET_STATE_NAMES):
        raise ValueError("RJ45 play-reset seed has unexpected or missing state tensors.")
    body_count = len(expected_task_body_order)
    specs = {
        "task_body_pose": (body_count, 7),
        "task_body_previous_pose": (body_count, 7),
        "task_body_coupling_previous_pose": (body_count, 7),
        "task_body_velocity": (body_count, 6),
        "goal_task_body_pose": (body_count, 7),
    }
    result: dict[str, torch.Tensor] = {}
    for name, shape in specs.items():
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"RJ45 play-reset seed state.{name} must be a torch.Tensor.")
        if (
            value.device.type != "cpu"
            or value.dtype != torch.float32
            or value.layout != torch.strided
            or value.is_quantized
            or value.requires_grad
            or not value.is_contiguous()
            or tuple(value.shape) != shape
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(
                f"RJ45 play-reset seed state.{name} is not a finite contiguous CPU float32 {shape} tensor."
            )
        result[name] = value.detach().clone()
    for name in (
        "task_body_pose",
        "task_body_previous_pose",
        "task_body_coupling_previous_pose",
        "goal_task_body_pose",
    ):
        norms = torch.linalg.vector_norm(result[name][..., 3:7], dim=-1)
        if not bool(torch.all(torch.abs(norms - 1.0) <= 1.0e-5)):
            raise ValueError(f"RJ45 play-reset seed state.{name} contains a non-unit quaternion.")
    return result


def load_play_reset_seed(
    *,
    expected_task_body_order: Sequence[str],
    expected_physics_contract: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Load the packaged physical play-reset seed using PyTorch's restricted loader."""
    resource = resources.files(__package__).joinpath(_PLAY_RESET_SEED_RESOURCE)
    with resource.open("rb") as seed_file:
        payload = torch.load(seed_file, map_location="cpu", weights_only=True)
    return _validate_play_reset_seed(
        payload,
        expected_task_body_order=expected_task_body_order,
        expected_physics_contract=expected_physics_contract,
    )


def rigidly_transform_task_pose(
    task_pose: torch.Tensor,
    source_frame: torch.Tensor,
    destination_frame: torch.Tensor,
) -> torch.Tensor:
    """Apply one rigid transform to batched task-body poses in XYZW convention."""
    if task_pose.ndim != 3 or task_pose.shape[-1] != 7:
        raise ValueError("Task pose must have shape (N, B, 7).")
    if source_frame.shape != (task_pose.shape[0], 7) or destination_frame.shape != source_frame.shape:
        raise ValueError("Source and destination frames must have shape (N, 7).")
    delta_q = math_utils.quat_mul(destination_frame[:, 3:7], math_utils.quat_conjugate(source_frame[:, 3:7]))
    rotation = delta_q[:, None, :].expand(-1, task_pose.shape[1], -1)
    relative_position = task_pose[..., :3] - source_frame[:, None, :3]
    position = destination_frame[:, None, :3] + math_utils.quat_apply(rotation, relative_position)
    orientation = math_utils.quat_mul(rotation, task_pose[..., 3:7])
    orientation /= torch.linalg.vector_norm(orientation, dim=-1, keepdim=True).clamp_min(1.0e-9)
    return torch.cat((position, orientation), dim=-1)


def rigidly_transform_task_state(
    task_pose: torch.Tensor,
    task_velocity: torch.Tensor,
    source_frame: torch.Tensor,
    destination_frame: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one rigid transform to task poses and world-frame spatial velocities."""
    if task_velocity.shape != (*task_pose.shape[:-1], 6):
        raise ValueError("Task velocity must have shape (N, B, 6) matching the task pose.")
    transformed_pose = rigidly_transform_task_pose(task_pose, source_frame, destination_frame)
    delta_q = math_utils.quat_mul(destination_frame[:, 3:7], math_utils.quat_conjugate(source_frame[:, 3:7]))
    rotation = delta_q[:, None, :].expand(-1, task_velocity.shape[1], -1)
    linear = math_utils.quat_apply(rotation, task_velocity[..., :3])
    angular = math_utils.quat_apply(rotation, task_velocity[..., 3:6])
    return transformed_pose, torch.cat((linear, angular), dim=-1)


def transform_play_reset_seed(
    seed: Mapping[str, torch.Tensor],
    *,
    socket_body_index: int,
    plug_body_index: int,
    socket_pose: torch.Tensor,
    pickup_pose: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Place one physical seed at batched socket and pickup frames."""
    if socket_pose.ndim != 2 or socket_pose.shape[-1] != 7 or pickup_pose.shape != socket_pose.shape:
        raise ValueError("Socket and pickup poses must have matching shape (N, 7).")
    count = socket_pose.shape[0]
    state = {name: seed[name].unsqueeze(0).repeat(count, 1, 1) for name in _PLAY_RESET_STATE_NAMES}
    body_count = state["task_body_pose"].shape[1]
    if not 0 <= socket_body_index < body_count or not 0 <= plug_body_index < body_count:
        raise IndexError("Socket and plug body indices must select bodies in the packaged seed.")

    source_socket = state["task_body_pose"][:, socket_body_index].clone()
    state["task_body_pose"], state["task_body_velocity"] = rigidly_transform_task_state(
        state["task_body_pose"],
        state["task_body_velocity"],
        source_socket,
        socket_pose,
    )
    state["task_body_previous_pose"] = rigidly_transform_task_pose(
        state["task_body_previous_pose"], source_socket, socket_pose
    )
    state["task_body_coupling_previous_pose"] = rigidly_transform_task_pose(
        state["task_body_coupling_previous_pose"], source_socket, socket_pose
    )
    state["goal_task_body_pose"] = rigidly_transform_task_pose(
        state["goal_task_body_pose"],
        state["goal_task_body_pose"][:, socket_body_index],
        socket_pose,
    )

    socket_state = {
        name: value[:, socket_body_index].clone() for name, value in state.items() if name != "goal_task_body_pose"
    }
    source_plug = state["task_body_pose"][:, plug_body_index].clone()
    state["task_body_pose"], state["task_body_velocity"] = rigidly_transform_task_state(
        state["task_body_pose"],
        state["task_body_velocity"],
        source_plug,
        pickup_pose,
    )
    state["task_body_previous_pose"] = rigidly_transform_task_pose(
        state["task_body_previous_pose"], source_plug, pickup_pose
    )
    state["task_body_coupling_previous_pose"] = rigidly_transform_task_pose(
        state["task_body_coupling_previous_pose"], source_plug, pickup_pose
    )
    for name, value in socket_state.items():
        state[name][:, socket_body_index] = value
    return state


__all__ = [
    "PLAY_RESET_SEED_CONTENT_SHA256",
    "PLAY_RESET_SEED_FORMAT",
    "PLAY_RESET_SEED_SCHEMA_VERSION",
    "load_play_reset_seed",
    "rigidly_transform_task_pose",
    "rigidly_transform_task_state",
    "transform_play_reset_seed",
]
