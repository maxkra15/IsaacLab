# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate slung-load RSL-RL checkpoints on seeded randomized routes.

Aggregation and checkpoint selection are deliberately importable without
launching Isaac Sim. Runtime imports stay inside :func:`main`. Every checkpoint
sees the same initial route in each evaluation environment; only those initial
episodes are scored so policy-dependent autoresets cannot change the route set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import fmean
from typing import Any

_COMMON_UNSAFE_RESULT_KEYS = (
    "drone_crashes",
    "illegal_state_terminations",
    "workspace_exits",
)

_SLUNG_LOAD_UNSAFE_RESULT_KEYS = (
    "payload_crashes",
    "cable_integrity_failures",
)

_ROUTE_FAMILY_NAMES = {0: "ellipse", 1: "figure_eight", 2: "random_corner"}
_ROUTE_FAMILY_CAPABILITIES = {
    "bounded_template_mix": (0, 1),
    "bounded_hard_mix": (1, 2),
}
_LEGACY_ROUTE_FAMILY_IDS = _ROUTE_FAMILY_CAPABILITIES["bounded_template_mix"]

_COMMON_REQUIRED_EPISODE_KEYS = (
    "position_rmse",
    "position_error_max",
    "cross_track_error_mean",
    "cross_track_error_rms",
    "cross_track_error_max",
    "drone_speed_mean",
    "drone_speed_max",
    "route_family_id",
    "route_waypoints_passed",
    "route_traversal_fraction",
    "route_arc_length_traversed",
    "route_arc_length_traversal_rate",
    "waypoint_completion_fraction",
    "waypoint_completed",
    "waypoint_completion_time",
    "waypoint_count",
    "waypoint_arrivals",
    "waypoint_precision_hits",
    "waypoint_precision_hit_fraction",
    "waypoint_precision_misses",
    "waypoint_precision_miss_distance_mean",
    "waypoint_precision_miss_distance_max",
    "waypoint_arrival_time_mean",
    "waypoint_arrival_time_min",
    "waypoint_arrival_time_max",
    "waypoint_throughput",
    "route_completions",
    "target_distance_completed",
    "episode_duration",
    "drone_crash",
    "illegal_state",
    "workspace_exit",
    "path_corridor_exit",
    "success_termination",
)

_SLUNG_LOAD_REQUIRED_EPISODE_KEYS = (
    "swing_angle_mean",
    "swing_angle_rms",
    "swing_angle_max",
    "transverse_speed_rms",
    "payload_speed_mean",
    "payload_speed_max",
    "cable_relative_separation_mean",
    "cable_relative_separation_max",
    "cable_joint_error_mean",
    "cable_joint_error_max",
    "payload_crash",
    "cable_integrity_failure",
)

# Backward-compatible public test helper: legacy callers model the strict slung-load schema.
_REQUIRED_EPISODE_KEYS = _COMMON_REQUIRED_EPISODE_KEYS + _SLUNG_LOAD_REQUIRED_EPISODE_KEYS

_SLUNG_LOAD_PROFILE = "slung_load"
_DRONE_ONLY_PROFILE = "drone_only"


def _resolve_task_profile(episodes: list[dict[str, float]]) -> str:
    """Resolve one fail-closed task profile from the command capability metric."""
    availability: set[int] = set()
    for episode_index, episode in enumerate(episodes):
        raw_value = episode.get("slung_load_metrics_available", 1.0)
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Episode {episode_index} has an invalid slung_load_metrics_available capability."
            ) from error
        if not math.isfinite(value) or value not in (0.0, 1.0):
            raise ValueError(f"Episode {episode_index} has an invalid slung_load_metrics_available capability.")
        availability.add(int(value))
    if len(availability) != 1:
        raise ValueError("Evaluation episodes disagree about slung-load metric availability.")
    return _SLUNG_LOAD_PROFILE if availability == {1} else _DRONE_ONLY_PROFILE


def _required_episode_keys(task_profile: str) -> tuple[str, ...]:
    """Return the required metric schema for one resolved task profile."""
    if task_profile == _SLUNG_LOAD_PROFILE:
        return _REQUIRED_EPISODE_KEYS
    return _COMMON_REQUIRED_EPISODE_KEYS


def _binary_event_keys(task_profile: str) -> tuple[str, ...]:
    """Return termination metrics that must be binary for one task profile."""
    keys = ("drone_crash", "illegal_state", "workspace_exit", "path_corridor_exit", "success_termination")
    if task_profile == _SLUNG_LOAD_PROFILE:
        keys += ("payload_crash", "cable_integrity_failure")
    return keys


def _mean(episodes: list[dict[str, float]], name: str) -> float:
    return float(fmean(float(episode[name]) for episode in episodes))


def _maximum(episodes: list[dict[str, float]], name: str) -> float:
    return float(max(float(episode[name]) for episode in episodes))


def _sum(episodes: list[dict[str, float]], name: str) -> float:
    return float(sum(float(episode[name]) for episode in episodes))


def _event_count(episodes: list[dict[str, float]], name: str) -> int:
    return int(round(_sum(episodes, name)))


def _aggregate_metric_groups(
    episodes: list[dict[str, float]], waypoint_count: int, *, task_profile: str
) -> dict[str, Any]:
    """Aggregate one non-empty overall or route-family episode subset."""
    completed_times = [
        float(episode["waypoint_completion_time"]) for episode in episodes if float(episode["waypoint_completed"]) > 0.0
    ]
    episode_time_total = _sum(episodes, "episode_duration")
    active_times = [
        float(episode["waypoint_completion_time"])
        if float(episode["waypoint_completed"]) > 0.0
        else float(episode["episode_duration"])
        for episode in episodes
    ]
    active_time_total = float(sum(active_times))
    arc_length_total = _sum(episodes, "route_arc_length_traversed")
    precision_hits_total = _event_count(episodes, "waypoint_precision_hits")
    precision_misses_total = _event_count(episodes, "waypoint_precision_misses")
    passed_total = _event_count(episodes, "route_waypoints_passed")
    hit_episodes = [episode for episode in episodes if float(episode["waypoint_precision_hits"]) > 0.0]
    target_distance_total = _sum(episodes, "target_distance_completed")
    mean_interarrival_time = (
        sum(
            float(episode["waypoint_arrival_time_mean"]) * float(episode["waypoint_precision_hits"])
            for episode in hit_episodes
        )
        / precision_hits_total
        if precision_hits_total > 0
        else 0.0
    )
    mean_precision_miss_distance = (
        sum(
            float(episode["waypoint_precision_miss_distance_mean"]) * float(episode["waypoint_precision_misses"])
            for episode in episodes
        )
        / precision_misses_total
        if precision_misses_total > 0
        else 0.0
    )
    groups: dict[str, Any] = {
        "tracking": {
            "position_rmse": _mean(episodes, "position_rmse"),
            "max_position_error": _maximum(episodes, "position_error_max"),
            "cross_track_error_mean": _mean(episodes, "cross_track_error_mean"),
            "cross_track_error_rms": _mean(episodes, "cross_track_error_rms"),
            "cross_track_error_max": _maximum(episodes, "cross_track_error_max"),
        },
        "speed": {
            "drone_mean": _mean(episodes, "drone_speed_mean"),
            "drone_max": _maximum(episodes, "drone_speed_max"),
        },
        "physics_safety": {
            "drone_crashes": _event_count(episodes, "drone_crash"),
            "illegal_state_terminations": _event_count(episodes, "illegal_state"),
            "workspace_exits": _event_count(episodes, "workspace_exit"),
        },
        "route": {
            "targets_per_route": waypoint_count,
            "waypoints_passed_total": passed_total,
            "waypoints_passed_mean": _mean(episodes, "route_waypoints_passed"),
            "traversal_fraction": _mean(episodes, "route_traversal_fraction"),
            "arc_length_traversed_total": arc_length_total,
            "arc_length_traversed_mean": _mean(episodes, "route_arc_length_traversed"),
            "episode_arc_length_rate": arc_length_total / episode_time_total,
            "active_arc_length_rate": arc_length_total / active_time_total,
            "mean_reported_arc_length_rate": _mean(episodes, "route_arc_length_traversal_rate"),
            "route_completions_total": _event_count(episodes, "route_completions"),
            "completion_rate": _mean(episodes, "waypoint_completed"),
            "completion_time": float(fmean(completed_times)) if completed_times else 0.0,
            "episode_time_total": episode_time_total,
            "active_time_total": active_time_total,
        },
        "precision": {
            "hits_total": precision_hits_total,
            "hits_mean": _mean(episodes, "waypoint_precision_hits"),
            "hit_fraction_of_passed": precision_hits_total / passed_total if passed_total > 0 else 0.0,
            "mean_episode_hit_fraction": _mean(episodes, "waypoint_precision_hit_fraction"),
            "misses_total": precision_misses_total,
            "misses_mean": _mean(episodes, "waypoint_precision_misses"),
            "miss_distance_mean": mean_precision_miss_distance,
            "miss_distance_max": _maximum(episodes, "waypoint_precision_miss_distance_max"),
            "episode_hit_rate_hz": precision_hits_total / episode_time_total,
            "active_hit_rate_hz": precision_hits_total / active_time_total,
            "mean_interarrival_time": mean_interarrival_time,
            "min_interarrival_time": (
                min(float(episode["waypoint_arrival_time_min"]) for episode in hit_episodes) if hit_episodes else 0.0
            ),
            "max_interarrival_time": _maximum(episodes, "waypoint_arrival_time_max"),
            "target_distance_total": target_distance_total,
            "episode_target_distance_rate": target_distance_total / episode_time_total,
            "active_target_distance_rate": target_distance_total / active_time_total,
        },
        "termination": {
            "success_terminations": _event_count(episodes, "success_termination"),
            "path_corridor_exits": _event_count(episodes, "path_corridor_exit"),
            "path_corridor_exit_rate": _mean(episodes, "path_corridor_exit"),
        },
    }
    if task_profile == _SLUNG_LOAD_PROFILE:
        groups["swing"] = {
            "mean_angle": _mean(episodes, "swing_angle_mean"),
            "rms_angle": _mean(episodes, "swing_angle_rms"),
            "max_angle": _maximum(episodes, "swing_angle_max"),
            "transverse_speed_rms": _mean(episodes, "transverse_speed_rms"),
        }
        groups["speed"].update(
            payload_mean=_mean(episodes, "payload_speed_mean"),
            payload_max=_maximum(episodes, "payload_speed_max"),
        )
        groups["physics_safety"].update(
            payload_crashes=_event_count(episodes, "payload_crash"),
            cable_integrity_failures=_event_count(episodes, "cable_integrity_failure"),
            cable_relative_separation_mean=_mean(episodes, "cable_relative_separation_mean"),
            cable_relative_separation_max=_maximum(episodes, "cable_relative_separation_max"),
            cable_joint_error_mean=_mean(episodes, "cable_joint_error_mean"),
            cable_joint_error_max=_maximum(episodes, "cable_joint_error_max"),
        )
    return groups


def checkpoint_iteration(checkpoint_path: str) -> int | None:
    """Extract the last integer from a checkpoint filename."""
    matches = re.findall(r"\d+", Path(checkpoint_path).stem)
    return int(matches[-1]) if matches else None


def _validate_route_family_ids(route_family_ids: tuple[int, ...]) -> tuple[int, int]:
    """Return one canonical two-family evaluation capability or fail closed."""
    try:
        family_ids = tuple(route_family_ids)
    except TypeError as error:
        raise ValueError("route_family_ids must select one supported two-family capability.") from error
    if any(not isinstance(family_id, int) or isinstance(family_id, bool) for family_id in family_ids):
        raise ValueError("route_family_ids must contain integer route-family IDs.")
    supported_capabilities = tuple(_ROUTE_FAMILY_CAPABILITIES.values())
    if family_ids not in supported_capabilities:
        raise ValueError(
            f"route_family_ids must be exactly one supported capability: {supported_capabilities}; got {family_ids}."
        )
    return family_ids


def _route_family_ids_for_cfg(route_family: object) -> tuple[int, int]:
    """Resolve the command route-family mode to its scored evaluator capability."""
    try:
        return _ROUTE_FAMILY_CAPABILITIES[route_family]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Evaluation requires route_family='bounded_template_mix' or 'bounded_hard_mix'; got {route_family!r}."
        ) from error


def _command_route_family_ids(
    command_term: Any,
    episode_count: int,
    route_family_ids: tuple[int, ...] = _LEGACY_ROUTE_FAMILY_IDS,
):
    """Read and validate the command's authoritative integer family-ID tensor."""
    import torch

    expected_ids = _validate_route_family_ids(route_family_ids)
    if not isinstance(episode_count, int) or isinstance(episode_count, bool) or episode_count <= 0:
        raise ValueError("episode_count must be a positive integer.")
    if not hasattr(command_term, "route_family_id"):
        raise ValueError("Route command must expose the integer route_family_id tensor.")
    family_ids = command_term.route_family_id
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if not isinstance(family_ids, torch.Tensor) or family_ids.ndim != 1 or family_ids.dtype not in integer_dtypes:
        raise ValueError("Route command route_family_id must be a rank-one integer tensor.")
    if family_ids.shape[0] < episode_count:
        raise ValueError("Route command route_family_id is shorter than the scored episode count.")
    scored_family_ids = family_ids[:episode_count]
    observed_ids = {int(family_id) for family_id in scored_family_ids.detach().to(device="cpu").tolist()}
    unknown_ids = observed_ids.difference(_ROUTE_FAMILY_NAMES)
    if unknown_ids:
        raise ValueError(f"Route command contains unknown route family IDs: {sorted(unknown_ids)}.")
    unexpected_ids = observed_ids.difference(expected_ids)
    if unexpected_ids:
        raise ValueError(
            "Route command family IDs do not match the configured evaluation capability: "
            f"expected {expected_ids}, observed {sorted(observed_ids)}."
        )
    missing_ids = set(expected_ids).difference(observed_ids)
    if missing_ids:
        missing_names = ", ".join(
            _ROUTE_FAMILY_NAMES[family_id] for family_id in expected_ids if family_id in missing_ids
        )
        raise ValueError(f"Seeded mixed route suite is missing route families: {missing_names}.")
    return scored_family_ids


def _route_family_names_from_counts(family_counts: object, episode_count: object) -> tuple[str, str] | None:
    """Validate one aggregate suite capability and return its canonical family names."""
    if not isinstance(family_counts, dict):
        return None
    for family_ids in _ROUTE_FAMILY_CAPABILITIES.values():
        family_names = tuple(_ROUTE_FAMILY_NAMES[family_id] for family_id in family_ids)
        if set(family_counts) != set(family_names):
            continue
        counts = tuple(family_counts[name] for name in family_names)
        if (
            all(isinstance(count, int) and not isinstance(count, bool) and count > 0 for count in counts)
            and isinstance(episode_count, int)
            and not isinstance(episode_count, bool)
            and sum(counts) == episode_count
        ):
            return family_names
    return None


def _validate_episode_route_family_id(value: float, episode_index: int, expected_family_ids: tuple[int, int]) -> None:
    """Validate one numeric episode family against the configured capability."""
    if not value.is_integer() or int(value) not in _ROUTE_FAMILY_NAMES:
        raise ValueError(f"Episode {episode_index} has an invalid route family ID.")
    if int(value) not in expected_family_ids:
        raise ValueError(f"Episode {episode_index} does not match the configured route-family capability.")


def aggregate_checkpoint_results(
    episodes: list[dict[str, float]],
    *,
    task_id: str,
    checkpoint_path: str,
    seed: int,
    route_suite_sha256: str,
    route_family_ids: tuple[int, ...] = _LEGACY_ROUTE_FAMILY_IDS,
) -> dict[str, Any]:
    """Aggregate completed-episode summaries into JSON-native values."""
    expected_family_ids = _validate_route_family_ids(route_family_ids)
    if re.fullmatch(r"[0-9a-f]{64}", route_suite_sha256) is None:
        raise ValueError("route_suite_sha256 must be a lowercase SHA-256 digest.")
    if not episodes:
        raise ValueError("At least one completed episode is required.")
    task_profile = _resolve_task_profile(episodes)
    required_episode_keys = _required_episode_keys(task_profile)
    for episode_index, episode in enumerate(episodes):
        missing = [name for name in required_episode_keys if name not in episode]
        if missing:
            raise ValueError(f"Episode {episode_index} is missing required metrics: {', '.join(missing)}.")
        values = {name: float(episode[name]) for name in required_episode_keys}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"Episode {episode_index} contains a non-finite required metric.")
        if any(value < 0.0 for value in values.values()):
            raise ValueError(f"Episode {episode_index} contains a negative magnitude or count metric.")
        event_keys = _binary_event_keys(task_profile)
        if any(values[name] not in (0.0, 1.0) for name in event_keys):
            raise ValueError(f"Episode {episode_index} has a non-binary termination event metric.")
        _validate_episode_route_family_id(values["route_family_id"], episode_index, expected_family_ids)
        waypoint_count = values["waypoint_count"]
        passed = values["route_waypoints_passed"]
        traversal_fraction = values["route_traversal_fraction"]
        legacy_completion_fraction = values["waypoint_completion_fraction"]
        completed = values["waypoint_completed"]
        completion_time = values["waypoint_completion_time"]
        duration = values["episode_duration"]
        route_completions = values["route_completions"]
        precision_hits = values["waypoint_precision_hits"]
        precision_misses = values["waypoint_precision_misses"]
        arrivals = values["waypoint_arrivals"]
        precision_hit_fraction = values["waypoint_precision_hit_fraction"]
        target_distance = values["target_distance_completed"]
        arrival_time_mean = values["waypoint_arrival_time_mean"]
        arrival_time_min = values["waypoint_arrival_time_min"]
        arrival_time_max = values["waypoint_arrival_time_max"]
        if waypoint_count <= 0.0 or not waypoint_count.is_integer():
            raise ValueError(f"Episode {episode_index} has an invalid waypoint count.")
        count_values = {
            "route waypoints passed": passed,
            "precision hits": precision_hits,
            "precision misses": precision_misses,
        }
        if any(value > waypoint_count or not value.is_integer() for value in count_values.values()):
            raise ValueError(f"Episode {episode_index} has an invalid traversal or precision count.")
        if completed not in (0.0, 1.0) or not 0.0 <= traversal_fraction <= 1.0:
            raise ValueError(f"Episode {episode_index} has invalid route traversal/completion metrics.")
        if not math.isclose(traversal_fraction, passed / waypoint_count, abs_tol=1.0e-6):
            raise ValueError(f"Episode {episode_index} has inconsistent traversal fraction and passed count.")
        if not math.isclose(legacy_completion_fraction, traversal_fraction, abs_tol=1.0e-6):
            raise ValueError(f"Episode {episode_index} has inconsistent legacy and route traversal fractions.")
        if bool(completed) != (passed == waypoint_count):
            raise ValueError(f"Episode {episode_index} has inconsistent route completion and traversal count.")
        if precision_hits + precision_misses != passed:
            raise ValueError(f"Episode {episode_index} has precision hit/miss counts inconsistent with traversal.")
        if arrivals != precision_hits:
            raise ValueError(f"Episode {episode_index} has waypoint arrivals inconsistent with precision hits.")
        expected_hit_fraction = precision_hits / max(passed, 1.0)
        if not math.isclose(precision_hit_fraction, expected_hit_fraction, abs_tol=1.0e-6):
            raise ValueError(f"Episode {episode_index} has an inconsistent precision-hit fraction.")
        if duration <= 0.0 or completion_time < 0.0 or completion_time > duration:
            raise ValueError(f"Episode {episode_index} has invalid timing metrics.")
        if bool(completed) != (completion_time > 0.0):
            raise ValueError(f"Episode {episode_index} has inconsistent completion status and completion time.")
        if route_completions != completed:
            raise ValueError(f"Episode {episode_index} has inconsistent route completion count.")
        if values["success_termination"] != completed:
            raise ValueError(f"Episode {episode_index} has inconsistent success termination and route completion.")
        if completed > 0.0 and values["route_arc_length_traversed"] <= 0.0:
            raise ValueError(f"Episode {episode_index} completed without positive traversed arc length.")
        if precision_hits == 0.0:
            if any(value != 0.0 for value in (arrival_time_mean, arrival_time_min, arrival_time_max, target_distance)):
                raise ValueError(f"Episode {episode_index} has precision-hit statistics without a hit.")
        elif not (
            0.0 < arrival_time_min <= arrival_time_mean <= arrival_time_max <= duration and target_distance > 0.0
        ):
            raise ValueError(f"Episode {episode_index} has inconsistent precision-hit timing or distance metrics.")
        miss_distance_mean = values["waypoint_precision_miss_distance_mean"]
        miss_distance_max = values["waypoint_precision_miss_distance_max"]
        if precision_misses == 0.0 and (miss_distance_mean != 0.0 or miss_distance_max != 0.0):
            raise ValueError(f"Episode {episode_index} has miss distances without a precision miss.")
        if precision_misses > 0.0 and not 0.0 < miss_distance_mean <= miss_distance_max:
            raise ValueError(f"Episode {episode_index} has inconsistent precision-miss distances.")
    waypoint_counts = {int(episode["waypoint_count"]) for episode in episodes}
    if len(waypoint_counts) != 1 or next(iter(waypoint_counts)) <= 0:
        raise ValueError("Every randomized evaluation episode must use the same positive waypoint count.")
    waypoint_count = next(iter(waypoint_counts))
    family_episodes = {
        name: [episode for episode in episodes if int(episode["route_family_id"]) == family_id]
        for family_id in expected_family_ids
        for name in (_ROUTE_FAMILY_NAMES[family_id],)
    }
    missing_families = [name for name, subset in family_episodes.items() if not subset]
    if missing_families:
        raise ValueError(f"Seeded mixed route suite is missing route families: {', '.join(missing_families)}.")
    family_counts = {name: len(subset) for name, subset in family_episodes.items()}
    groups = _aggregate_metric_groups(episodes, waypoint_count, task_profile=task_profile)
    result: dict[str, Any] = {
        "seed": int(seed),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_iteration": checkpoint_iteration(checkpoint_path),
        "task_id": task_id,
        "episode_count": len(episodes),
        "task_profile": task_profile,
        "evaluation_suite": {
            "kind": "seeded_randomized_finite_routes",
            "route_suite_sha256": route_suite_sha256,
            "route_family_counts": family_counts,
            "task_profile": task_profile,
        },
        **groups,
        "route_families": {
            name: {
                "episode_count": len(subset),
                **_aggregate_metric_groups(subset, waypoint_count, task_profile=task_profile),
            }
            for name, subset in family_episodes.items()
        },
    }
    return result


def select_checkpoint(
    results: list[dict[str, Any]],
    *,
    min_completion_rate: float = 0.5,
    min_family_completion_rate: float | None = None,
    min_traversal_fraction: float = 0.0,
    min_family_traversal_fraction: float | None = None,
    min_active_arc_rate: float = 0.0,
    min_family_active_arc_rate: float | None = None,
    min_precision_hit_fraction: float = 0.80,
    max_corridor_exit_rate: float = 0.0,
    max_mean_swing: float = math.radians(15.0),
    max_rms_swing: float = math.radians(20.0),
    max_peak_swing: float = 1.0,
    max_transverse_speed_rms: float = 1.0,
    max_cross_track_rms: float = 0.60,
    max_cross_track_error: float = 1.75,
    max_cable_relative_separation: float = 0.02,
    max_cable_joint_error: float = 0.005,
) -> dict[str, Any]:
    """Select the best checkpoint that passes aggregate and per-family gates."""
    limits = (
        max_mean_swing,
        max_rms_swing,
        max_peak_swing,
        max_transverse_speed_rms,
        max_cross_track_rms,
        max_cross_track_error,
        max_cable_relative_separation,
        max_cable_joint_error,
    )
    if not math.isfinite(min_completion_rate) or not 0.0 <= min_completion_rate <= 1.0:
        raise ValueError("min_completion_rate must be finite and within [0, 1].")
    if min_family_completion_rate is None:
        min_family_completion_rate = min_completion_rate
    if not math.isfinite(min_family_completion_rate) or not 0.0 <= min_family_completion_rate <= 1.0:
        raise ValueError("min_family_completion_rate must be finite and within [0, 1].")
    if not math.isfinite(min_traversal_fraction) or not 0.0 <= min_traversal_fraction <= 1.0:
        raise ValueError("min_traversal_fraction must be finite and within [0, 1].")
    if min_family_traversal_fraction is None:
        min_family_traversal_fraction = min_traversal_fraction
    if not math.isfinite(min_family_traversal_fraction) or not 0.0 <= min_family_traversal_fraction <= 1.0:
        raise ValueError("min_family_traversal_fraction must be finite and within [0, 1].")
    if not math.isfinite(min_active_arc_rate) or min_active_arc_rate < 0.0:
        raise ValueError("min_active_arc_rate must be finite and nonnegative.")
    if min_family_active_arc_rate is None:
        min_family_active_arc_rate = min_active_arc_rate
    if not math.isfinite(min_family_active_arc_rate) or min_family_active_arc_rate < 0.0:
        raise ValueError("min_family_active_arc_rate must be finite and nonnegative.")
    if not math.isfinite(max_corridor_exit_rate) or not 0.0 <= max_corridor_exit_rate <= 1.0:
        raise ValueError("max_corridor_exit_rate must be finite and within [0, 1].")
    if not math.isfinite(min_precision_hit_fraction) or not 0.0 <= min_precision_hit_fraction <= 1.0:
        raise ValueError("min_precision_hit_fraction must be finite and within [0, 1].")
    if any(not math.isfinite(value) or value < 0.0 for value in limits):
        raise ValueError("Checkpoint safety limits must be finite and nonnegative.")

    def comparison_identity(
        result: dict[str, Any],
    ) -> tuple[str, int, int, str, str, tuple[tuple[str, int], ...], str] | None:
        suite = result.get("evaluation_suite")
        if not isinstance(suite, dict):
            return None
        task_id = result.get("task_id")
        seed = result.get("seed")
        episode_count = result.get("episode_count")
        kind = suite.get("kind")
        digest = suite.get("route_suite_sha256")
        family_counts = suite.get("route_family_counts")
        task_profile = result.get("task_profile", suite.get("task_profile", _SLUNG_LOAD_PROFILE))
        if (
            not isinstance(task_id, str)
            or not task_id
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or not isinstance(episode_count, int)
            or isinstance(episode_count, bool)
            or episode_count <= 0
            or kind != "seeded_randomized_finite_routes"
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or task_profile not in (_SLUNG_LOAD_PROFILE, _DRONE_ONLY_PROFILE)
        ):
            return None
        family_names = _route_family_names_from_counts(family_counts, episode_count)
        if family_names is None:
            return None
        family_count_identity = tuple((name, family_counts[name]) for name in family_names)
        return task_id, seed, episode_count, kind, digest, family_count_identity, task_profile

    identities = [comparison_identity(result) for result in results]
    if not identities or any(identity is None for identity in identities) or len(set(identities)) != 1:
        raise ValueError(
            "Checkpoints must share one valid task, seed, mixed-family episode count, and randomized route suite "
            "digest."
        )
    task_profile = identities[0][-1]
    family_names = tuple(name for name, _ in identities[0][-2])

    def groups_pass(
        container: dict[str, Any],
        *,
        completion_threshold: float,
        traversal_threshold: float,
        active_arc_rate_threshold: float,
    ) -> bool:
        group_names = ["physics_safety", "tracking", "speed", "route", "precision", "termination"]
        if task_profile == _SLUNG_LOAD_PROFILE:
            group_names.append("swing")
        groups = [container.get(name) for name in group_names]
        if not all(isinstance(group, dict) for group in groups):
            return False
        safety = container["physics_safety"]
        swing = container.get("swing")
        tracking = container["tracking"]
        route = container["route"]
        precision = container["precision"]
        termination = container["termination"]
        try:
            all_metrics = tuple(float(value) for group in groups for value in group.values())
            completion_rate = float(route["completion_rate"])
            traversal_fraction = float(route["traversal_fraction"])
            active_arc_rate = float(route["active_arc_length_rate"])
            completion_time = float(route["completion_time"])
            route_completions = float(route["route_completions_total"])
            success_terminations = float(termination["success_terminations"])
            precision_hit_fraction = float(precision["hit_fraction_of_passed"])
            unsafe_names = _COMMON_UNSAFE_RESULT_KEYS
            if task_profile == _SLUNG_LOAD_PROFILE:
                unsafe_names += _SLUNG_LOAD_UNSAFE_RESULT_KEYS
            slung_load_limits_pass = task_profile == _DRONE_ONLY_PROFILE or (
                isinstance(swing, dict)
                and float(swing["mean_angle"]) <= max_mean_swing
                and float(swing["rms_angle"]) <= max_rms_swing
                and float(swing["max_angle"]) <= max_peak_swing
                and float(swing["transverse_speed_rms"]) <= max_transverse_speed_rms
                and float(safety["cable_relative_separation_max"]) <= max_cable_relative_separation
                and float(safety["cable_joint_error_max"]) <= max_cable_joint_error
            )
            return (
                all(math.isfinite(value) for value in all_metrics)
                and all(value >= 0.0 for value in all_metrics)
                and all(float(safety[name]) == 0.0 for name in unsafe_names)
                and float(termination["path_corridor_exit_rate"]) <= max_corridor_exit_rate
                and completion_rate >= completion_threshold
                and traversal_fraction >= traversal_threshold
                and active_arc_rate >= active_arc_rate_threshold
                and 0.0 <= completion_rate <= 1.0
                and 0.0 <= traversal_fraction <= 1.0
                and 0.0 <= precision_hit_fraction <= 1.0
                and precision_hit_fraction >= min_precision_hit_fraction
                and (completion_rate <= 0.0 or completion_time > 0.0)
                and route_completions == success_terminations
                and slung_load_limits_pass
                and float(tracking["cross_track_error_rms"]) <= max_cross_track_rms
                and float(tracking["cross_track_error_max"]) <= max_cross_track_error
                and float(route["episode_time_total"]) > 0.0
                and float(route["active_time_total"]) > 0.0
                and (float(precision["hits_total"]) <= 0.0 or float(precision["mean_interarrival_time"]) > 0.0)
            )
        except (KeyError, TypeError, ValueError):
            return False

    def is_safe(result: dict[str, Any]) -> bool:
        families = result.get("route_families")
        if not isinstance(families, dict) or set(families) != set(family_names):
            return False
        if not groups_pass(
            result,
            completion_threshold=min_completion_rate,
            traversal_threshold=min_traversal_fraction,
            active_arc_rate_threshold=min_active_arc_rate,
        ):
            return False
        for family_name, family in families.items():
            if not isinstance(family, dict):
                return False
            expected_count = result["evaluation_suite"]["route_family_counts"][family_name]
            if family.get("episode_count") != expected_count or not groups_pass(
                family,
                completion_threshold=min_family_completion_rate,
                traversal_threshold=min_family_traversal_fraction,
                active_arc_rate_threshold=min_family_active_arc_rate,
            ):
                return False
        return True

    eligible = [result for result in results if is_safe(result)]
    if not eligible:
        raise ValueError(
            "No checkpoint satisfies the aggregate and per-family traversal, completion, route-speed, "
            "path-corridor, tracking, applicable physical-safety, and zero-unsafe-event constraints."
        )
    return min(
        eligible,
        key=lambda result: (
            -min(float(result["route_families"][family]["route"]["completion_rate"]) for family in family_names),
            -float(result["route"]["completion_rate"]),
            -min(float(result["route_families"][family]["route"]["traversal_fraction"]) for family in family_names),
            -float(result["route"]["active_arc_length_rate"]),
            -float(result["route"]["traversal_fraction"]),
            (
                float(result["route"]["completion_time"])
                if float(result["route"]["completion_rate"]) > 0.0
                else math.inf
            ),
            -float(result["precision"]["hit_fraction_of_passed"]),
            float(result["tracking"]["cross_track_error_rms"]),
            float(result.get("swing", {}).get("mean_angle", 0.0)),
            float(result["tracking"]["position_rmse"]),
        ),
    )


def _episode_from_env(env, env_id: int, *, episode_duration: float) -> dict[str, float]:
    command_term = env.unwrapped.command_manager.get_term("route")
    episode = {name: float(values[env_id].item()) for name, values in command_term.last_episode_metrics.items()}
    waypoint_count = int(command_term.waypoints_e.shape[1])
    episode["waypoint_count"] = float(waypoint_count)
    episode["episode_duration"] = float(episode_duration)
    task_profile = _resolve_task_profile([episode])
    termination_manager = env.unwrapped.termination_manager

    def fired(*names: str) -> int:
        return int(any(bool(termination_manager.get_term(name)[env_id].item()) for name in names))

    def fired_if_present(name: str) -> int:
        """Read an optional safety term without weakening required-term checks."""
        if name not in termination_manager.active_terms:
            return 0
        return fired(name)

    common_terms = {
        "drone_crash": fired("drone_crash"),
        "illegal_state": fired("illegal_drone", "illegal_action"),
        "workspace_exit": fired("drone_out_of_workspace"),
        "path_corridor_exit": fired_if_present("path_corridor"),
        "success_termination": fired_if_present("route_completed"),
    }
    if task_profile == _SLUNG_LOAD_PROFILE:
        common_terms.update(
            payload_crash=fired("payload_crash"),
            illegal_state=fired("illegal_drone", "illegal_payload", "illegal_cable", "illegal_action"),
            workspace_exit=fired("drone_out_of_workspace", "payload_out_of_workspace"),
            cable_integrity_failure=fired("cable_integrity"),
        )
    episode.update(common_terms)
    return episode


def prepare_evaluation_env_cfg(env_cfg: Any, num_envs: int) -> Any:
    """Select finite randomized routes and evaluation vectorization.

    The task's training route distribution and episode horizon are preserved.
    Streaming is disabled, when supported, so a route is a finite benchmark and
    the same seeded route can be compared across checkpoints.
    """
    env_cfg.scene.num_envs = num_envs
    env_cfg.evaluation_mode()
    env_cfg.commands.route.randomize_waypoints = True
    if hasattr(env_cfg.commands.route, "regenerate_on_completion"):
        env_cfg.commands.route.regenerate_on_completion = False
    env_cfg.commands.route.debug_vis = False
    return env_cfg


def _route_suite_sha256(
    command_term: Any,
    episode_count: int,
    route_family_ids: tuple[int, ...] = _LEGACY_ROUTE_FAMILY_IDS,
) -> str:
    """Return a stable digest of reset-relative routes and ordered family IDs."""
    offsets = command_term.waypoints_e[:episode_count] - command_term.route_anchor_e[:episode_count].unsqueeze(1)
    route_array = offsets.detach().to(device="cpu").contiguous().numpy()
    family_array = (
        _command_route_family_ids(command_term, episode_count, route_family_ids)
        .detach()
        .to(device="cpu")
        .contiguous()
        .numpy()
    )
    digest = hashlib.sha256()
    for array in (route_array, family_array):
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _evaluate_runtime(args: argparse.Namespace, checkpoint: str) -> dict[str, Any]:
    import importlib.metadata as metadata

    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab.utils.seed import configure_seed

    from isaaclab_rl.entrypoints.common import create_isaaclab_env
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

    resolved_checkpoint = retrieve_file_path(checkpoint)
    configure_seed(args.seed, torch_deterministic=True)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    prepare_evaluation_env_cfg(env_cfg, args.num_envs)
    route_family_ids = _route_family_ids_for_cfg(env_cfg.commands.route.route_family)
    env_cfg.seed = args.seed
    if args.visualization == "none":
        env_cfg.sim.visualizer_cfgs = []
    else:
        env_cfg.sim.visualizer_cfgs = [env_cfg.sim.default_visualizer_cfg]
        env_cfg.commands.route.debug_vis = True

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    runtime_args = argparse.Namespace(frontend="torch")
    env = create_isaaclab_env(args.task, env_cfg, runtime_args, convert_marl_to_single_agent=False)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resolved_checkpoint)
    policy = runner.get_inference_policy(device=wrapped.unwrapped.device)

    episodes_by_env: dict[int, dict[str, float]] = {}
    scored_env_ids = set(range(args.episodes))
    elapsed_steps = 0
    obs = wrapped.get_observations()
    command_term = wrapped.unwrapped.command_manager.get_term("route")
    try:
        route_suite_sha256 = _route_suite_sha256(command_term, args.episodes, route_family_ids)
    except (TypeError, ValueError):
        wrapped.close()
        raise
    try:
        while len(episodes_by_env) < args.episodes:
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = wrapped.step(actions)
            elapsed_steps += 1
            # Same-step autoreset has already snapshotted command metrics. Read them and
            # the terminal-step termination buffers before resetting recurrent policy state.
            # Only each scored environment's initial episode is retained. This keeps the
            # randomized route suite identical even when policies terminate at different times.
            completed_env_ids = dones.nonzero(as_tuple=False).flatten().tolist()
            for env_id in completed_env_ids:
                if env_id in scored_env_ids and env_id not in episodes_by_env:
                    episodes_by_env[env_id] = _episode_from_env(
                        wrapped,
                        env_id,
                        episode_duration=elapsed_steps * wrapped.unwrapped.step_dt,
                    )
            policy.reset(dones)
    finally:
        wrapped.close()
    episodes = [episodes_by_env[env_id] for env_id in range(args.episodes)]
    return aggregate_checkpoint_results(
        episodes,
        task_id=args.task,
        checkpoint_path=resolved_checkpoint,
        seed=args.seed,
        route_suite_sha256=route_suite_sha256,
        route_family_ids=route_family_ids,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Registered Isaac Lab task ID.")
    parser.add_argument("--checkpoint", nargs="+", required=True, help="One or more RSL-RL checkpoints.")
    parser.add_argument("--seed", type=int, default=42, help="Fixed evaluation seed.")
    parser.add_argument(
        "--num_envs",
        "--num-envs",
        dest="num_envs",
        type=int,
        default=32,
        help="Vectorized environments; must be at least --episodes for a matched seeded route suite.",
    )
    parser.add_argument(
        "--num_episodes",
        "--episodes",
        dest="episodes",
        type=int,
        required=True,
        help="Number of seeded randomized initial episodes per checkpoint.",
    )
    parser.add_argument("--output", "--output_path", dest="output", type=Path, required=True, help="Output JSON path.")
    parser.add_argument(
        "--visualization",
        "--viz",
        dest="visualization",
        choices=("none", "newton_gl"),
        default="none",
        help="Optional visualization mode; evaluation is headless by default.",
    )
    parser.add_argument("--device", default="cuda:0", help="Simulation and policy device.")
    parser.add_argument(
        "--min-completion-rate",
        type=float,
        default=0.5,
        help="Minimum fraction of randomized routes completed.",
    )
    parser.add_argument(
        "--min-family-completion-rate",
        type=float,
        default=None,
        help="Minimum completion rate for each route family; defaults to --min-completion-rate.",
    )
    parser.add_argument(
        "--min-traversal-fraction",
        type=float,
        default=0.0,
        help="Minimum mean route traversal fraction.",
    )
    parser.add_argument(
        "--min-family-traversal-fraction",
        type=float,
        default=None,
        help="Minimum traversal fraction for each route family; defaults to --min-traversal-fraction.",
    )
    parser.add_argument(
        "--min-active-arc-rate",
        type=float,
        default=0.0,
        help="Minimum active indexed-route arc-length rate [m/s].",
    )
    parser.add_argument(
        "--min-family-active-arc-rate",
        type=float,
        default=None,
        help="Minimum active arc-length rate [m/s] for each family; defaults to --min-active-arc-rate.",
    )
    parser.add_argument(
        "--min-precision-hit-fraction",
        type=float,
        default=0.80,
        help="Minimum strict waypoint-hit fraction among traversed waypoints, overall and per family.",
    )
    parser.add_argument(
        "--max-corridor-exit-rate",
        type=float,
        default=0.0,
        help="Maximum path-corridor exit rate overall and for each route family.",
    )
    parser.add_argument("--max-mean-swing", type=float, default=math.radians(15.0), help="Maximum mean swing [rad].")
    parser.add_argument("--max-rms-swing", type=float, default=math.radians(20.0), help="Maximum RMS swing [rad].")
    parser.add_argument("--max-peak-swing", type=float, default=1.0, help="Maximum peak swing [rad].")
    parser.add_argument(
        "--max-transverse-speed-rms",
        type=float,
        default=1.0,
        help="Maximum payload transverse-speed RMS [m/s].",
    )
    parser.add_argument(
        "--max-cross-track-rms", type=float, default=0.60, help="Maximum indexed-route cross-track RMS [m]."
    )
    parser.add_argument(
        "--max-cross-track-error", type=float, default=1.75, help="Maximum indexed-route cross-track error [m]."
    )
    parser.add_argument(
        "--max-cable-relative-separation",
        type=float,
        default=0.02,
        help="Maximum summed joint-gap fraction of nominal cable length.",
    )
    parser.add_argument(
        "--max-cable-joint-error", type=float, default=0.005, help="Maximum cable joint/attachment gap [m]."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate checkpoints in the selected kitless Isaac Lab backend."""
    args = _build_parser().parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive.")
    if args.episodes > args.num_envs:
        raise ValueError(
            "--num-envs must be at least --episodes so every checkpoint sees the same initial randomized routes."
        )
    safety_limits = (
        args.max_mean_swing,
        args.max_rms_swing,
        args.max_peak_swing,
        args.max_transverse_speed_rms,
        args.max_cross_track_rms,
        args.max_cross_track_error,
        args.max_cable_relative_separation,
        args.max_cable_joint_error,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in safety_limits):
        raise ValueError("Evaluation safety limits must be finite and nonnegative.")
    if not math.isfinite(args.min_completion_rate) or not 0.0 <= args.min_completion_rate <= 1.0:
        raise ValueError("--min-completion-rate must be finite and within [0, 1].")
    if args.min_family_completion_rate is not None and (
        not math.isfinite(args.min_family_completion_rate) or not 0.0 <= args.min_family_completion_rate <= 1.0
    ):
        raise ValueError("--min-family-completion-rate must be finite and within [0, 1].")
    if not math.isfinite(args.min_traversal_fraction) or not 0.0 <= args.min_traversal_fraction <= 1.0:
        raise ValueError("--min-traversal-fraction must be finite and within [0, 1].")
    if args.min_family_traversal_fraction is not None and (
        not math.isfinite(args.min_family_traversal_fraction) or not 0.0 <= args.min_family_traversal_fraction <= 1.0
    ):
        raise ValueError("--min-family-traversal-fraction must be finite and within [0, 1].")
    if not math.isfinite(args.min_active_arc_rate) or args.min_active_arc_rate < 0.0:
        raise ValueError("--min-active-arc-rate must be finite and nonnegative.")
    if args.min_family_active_arc_rate is not None and (
        not math.isfinite(args.min_family_active_arc_rate) or args.min_family_active_arc_rate < 0.0
    ):
        raise ValueError("--min-family-active-arc-rate must be finite and nonnegative.")
    if not math.isfinite(args.min_precision_hit_fraction) or not 0.0 <= args.min_precision_hit_fraction <= 1.0:
        raise ValueError("--min-precision-hit-fraction must be finite and within [0, 1].")
    if not math.isfinite(args.max_corridor_exit_rate) or not 0.0 <= args.max_corridor_exit_rate <= 1.0:
        raise ValueError("--max-corridor-exit-rate must be finite and within [0, 1].")
    import isaaclab_tasks  # noqa: F401

    results = [_evaluate_runtime(args, checkpoint) for checkpoint in args.checkpoint]
    route_suite_identities = {
        (
            result["evaluation_suite"]["route_suite_sha256"],
            tuple(sorted(result["evaluation_suite"]["route_family_counts"].items())),
            result.get("task_profile", result["evaluation_suite"].get("task_profile", _SLUNG_LOAD_PROFILE)),
        )
        for result in results
    }
    if len(route_suite_identities) != 1:
        raise RuntimeError("Checkpoints were evaluated on different randomized route suites.")
    route_suite_sha256, route_family_counts_items, task_profile = next(iter(route_suite_identities))
    selection_error: ValueError | None = None
    try:
        selected = select_checkpoint(
            results,
            min_completion_rate=args.min_completion_rate,
            min_family_completion_rate=args.min_family_completion_rate,
            min_traversal_fraction=args.min_traversal_fraction,
            min_family_traversal_fraction=args.min_family_traversal_fraction,
            min_active_arc_rate=args.min_active_arc_rate,
            min_family_active_arc_rate=args.min_family_active_arc_rate,
            min_precision_hit_fraction=args.min_precision_hit_fraction,
            max_corridor_exit_rate=args.max_corridor_exit_rate,
            max_mean_swing=args.max_mean_swing,
            max_rms_swing=args.max_rms_swing,
            max_peak_swing=args.max_peak_swing,
            max_transverse_speed_rms=args.max_transverse_speed_rms,
            max_cross_track_rms=args.max_cross_track_rms,
            max_cross_track_error=args.max_cross_track_error,
            max_cable_relative_separation=args.max_cable_relative_separation,
            max_cable_joint_error=args.max_cable_joint_error,
        )
    except ValueError as error:
        selection_error = error
        selected = None
    payload = {
        "evaluation_suite": {
            "kind": "seeded_randomized_finite_routes",
            "seed": args.seed,
            "episode_count": args.episodes,
            "route_suite_sha256": route_suite_sha256,
            "route_family_counts": dict(route_family_counts_items),
            "task_profile": task_profile,
        },
        "results": results,
        "selected_checkpoint": None if selected is None else selected["checkpoint_path"],
    }
    if selection_error is not None:
        payload["selection_error"] = str(selection_error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if selection_error is not None:
        message = f"{selection_error} Raw evaluation results were written to {args.output}."
        raise ValueError(message) from selection_error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
