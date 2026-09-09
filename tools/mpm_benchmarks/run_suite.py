# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run isolated Newton MPM benchmark points and aggregate presentation CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shlex
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tomllib

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
POINT_SCRIPT = SCRIPT_DIR / "benchmark_mpm_granular.py"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "results"
SUMMARY_METRICS = (
    "step_time_mean_ms",
    "step_time_p95_ms",
    "wall_seconds_per_simulated_second",
    "outer_steps_per_s",
    "aggregate_env_steps_per_s",
    "million_particle_updates_per_s",
    "million_particle_substep_updates_per_s",
    "single_world_realtime_factor",
    "aggregate_world_realtime_factor",
    "gpu_memory_peak_mib",
    "gpu_memory_peak_delta_mib",
    "gpu_utilization_mean_percent",
    "gpu_power_mean_w",
    "root_displacement_mean_m",
    "final_aabb_x_mean_m",
    "final_aabb_y_mean_m",
    "final_aabb_z_mean_m",
    "final_particle_rms_radius_mean_m",
)
SPECIMEN_EXTENT = (0.68, 0.50, 0.72)


def create_parser() -> argparse.ArgumentParser:
    """Create the suite-runner argument parser."""
    parser = argparse.ArgumentParser(description="Run a reproducible Newton MPM presentation benchmark suite.")
    parser.add_argument(
        "--manifest", type=Path, default=SCRIPT_DIR / "presentation_suite.toml", help="Experiment manifest."
    )
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Parent result directory.")
    parser.add_argument("--suite_id", default=None, help="Optional immutable suite directory name override.")
    parser.add_argument("--repetitions", type=int, default=None, help="Override manifest repetitions.")
    parser.add_argument("--only_sweep", action="append", default=[], help="Run only named sweeps; repeatable.")
    parser.add_argument("--num_shards", type=int, default=1, help="Split globally ordered runs across this many jobs.")
    parser.add_argument("--shard_index", type=int, default=0, help="Zero-based shard selected by this job.")
    parser.add_argument("--dry_run", action="store_true", help="Print the resolved run order without launching.")
    parser.add_argument("--fail_fast", action="store_true", help="Stop after the first failed point.")
    parser.add_argument(
        "--allow_busy_gpu",
        action="store_true",
        help="Run despite the manifest's initial GPU-memory/utilization guard (results remain contention-sensitive).",
    )
    return parser


def _next_power_of_two(value: float, minimum: int = 1) -> int:
    """Return the smallest power of two no smaller than ``value`` and ``minimum``."""
    required = max(int(math.ceil(value)), minimum)
    return 1 << (required - 1).bit_length()


def _expected_particle_count(voxel_size: float, particles_per_cell: float) -> int:
    """Mirror the point harness's exact cell-centered cuboid count."""
    spacing = voxel_size / particles_per_cell
    axis_counts = [max(math.ceil(extent / spacing - 0.5 - 1.0e-12), 1) for extent in SPECIMEN_EXTENT]
    return math.prod(axis_counts)


def _initial_grid_cell_count(voxel_size: float) -> int:
    """Return the initially occupied physical-voxel count per environment."""
    return math.prod(max(math.ceil(extent / voxel_size), 1) for extent in SPECIMEN_EXTENT)


def _grid_capacities(case: dict[str, Any]) -> dict[str, int]:
    """Mirror the point harness's complete sparse-grid capacity hierarchy."""
    num_envs = int(case["num_envs"])
    initial_cells = num_envs * _initial_grid_cell_count(float(case["voxel_size"]))
    active = _next_power_of_two(initial_cells * float(case["capacity_factor"]), minimum=1 << 12)
    leaf = _next_power_of_two(max(active / 4, num_envs * 256), minimum=1 << 8)
    lower = _next_power_of_two(max(active / 32, num_envs * 64), minimum=1 << 5)
    upper = _next_power_of_two(max(active / 128, num_envs * 32), minimum=1 << 3)
    return {
        "max_active_cell_count": active,
        "max_leaf_node_count": min(leaf, active),
        "max_lower_node_count": min(lower, leaf),
        "max_upper_node_count": min(upper, lower),
    }


def _sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: object) -> str:
    """Return a filesystem-safe stable token."""
    text = str(value).strip().lower()
    return "".join(character if character.isalnum() else "_" for character in text).strip("_")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    """Write homogeneous rows to CSV, including a header for an empty explicit schema."""
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    if not fieldnames:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Any) -> None:
    """Atomically write one JSON artifact."""
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _load_manifest(path: Path) -> dict[str, Any]:
    """Load and minimally validate a suite manifest."""
    with path.open("rb") as stream:
        manifest = tomllib.load(stream)
    if "defaults" not in manifest or not manifest.get("sweeps"):
        raise ValueError("Manifest must define [defaults] and at least one [[sweeps]] table.")
    return manifest


def _preflight_reason(case: dict[str, Any]) -> str:
    """Return an explicit suite-side capacity guard reason, or an empty string."""
    explicit_reason = str(case.get("preflight_skip_reason", "")).strip()
    if explicit_reason:
        return explicit_reason
    total_particles = int(case["estimated_total_particles"])
    max_particles = int(case.get("preflight_max_total_particles", 0))
    if max_particles and total_particles > max_particles:
        return f"estimated total particles {total_particles:,} exceed guard {max_particles:,}"
    capacity_labels = {
        "max_active_cell_count": "active-cell",
        "max_leaf_node_count": "leaf-node",
        "max_lower_node_count": "lower-node",
        "max_upper_node_count": "upper-node",
    }
    for capacity_name, label in capacity_labels.items():
        capacity = int(case[f"estimated_{capacity_name}"])
        guard = int(case.get(f"preflight_{capacity_name}", 0))
        if guard and capacity > guard:
            return f"configured {label} capacity {capacity:,} exceeds guard {guard:,}"
    return ""


def _resolve_cases(
    manifest: dict[str, Any], repetitions_override: int | None, selected: set[str]
) -> list[dict[str, Any]]:
    """Expand sweep cases and repetitions into independent child-process runs."""
    defaults = dict(manifest["defaults"])
    repetitions = repetitions_override if repetitions_override is not None else int(defaults.pop("repetitions", 3))
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    cases: list[dict[str, Any]] = []
    for sweep in manifest["sweeps"]:
        name = str(sweep["name"])
        if selected and name not in selected:
            continue
        for case_index, case in enumerate(sweep.get("cases", [])):
            resolved = {**defaults, **case}
            resolved.update(
                {
                    "sweep": name,
                    "x_parameter": sweep["x_parameter"],
                    "x_unit": sweep.get("x_unit", ""),
                    "case_id": case.get("id", f"{_slug(name)}_{case_index:02d}"),
                    "case_label": case.get("label", str(case.get("x_value", case_index))),
                }
            )
            if "x_value" not in resolved:
                raise ValueError(f"Sweep {name!r} case {case_index} is missing x_value.")
            particles_per_env = _expected_particle_count(
                float(resolved["voxel_size"]), float(resolved["particles_per_cell"])
            )
            resolved["estimated_particles_per_env"] = particles_per_env
            resolved["estimated_total_particles"] = int(resolved["num_envs"]) * particles_per_env
            for capacity_name, capacity in _grid_capacities(resolved).items():
                resolved[f"estimated_{capacity_name}"] = capacity
            resolved["preflight_reason"] = _preflight_reason(resolved)
            for replicate in range(repetitions):
                cases.append({**resolved, "replicate": replicate})
    if not cases:
        raise ValueError("The selected manifest contains no runnable cases.")
    return cases


def _randomized_block_order(cases: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Randomize cases within each repetition to balance run-order and thermal drift."""
    by_replicate: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_replicate[int(case["replicate"])].append(case)
    ordered: list[dict[str, Any]] = []
    for replicate in sorted(by_replicate):
        block = by_replicate[replicate]
        random.Random(seed + replicate).shuffle(block)
        ordered.extend(block)
    return ordered


def _format_cli_value(value: Any) -> str:
    """Format a manifest scalar for a child CLI."""
    if isinstance(value, bool):
        raise TypeError("Boolean manifest values require explicit flag handling.")
    return str(value)


def _point_command(case: dict[str, Any], suite_id: str, output_dir: Path, run_order: int) -> list[str]:
    """Build one benchmark-point command from a resolved manifest case."""
    command = [
        sys.executable,
        str(POINT_SCRIPT),
        "--output_dir",
        str(output_dir),
        "--suite_id",
        suite_id,
        "--sweep",
        str(case["sweep"]),
        "--case_id",
        str(case["case_id"]),
        "--case_label",
        str(case["case_label"]),
        "--x_parameter",
        str(case["x_parameter"]),
        "--x_value",
        _format_cli_value(case["x_value"]),
        "--x_unit",
        str(case.get("x_unit", "")),
        "--replicate",
        str(case["replicate"]),
        "--run_order",
        str(run_order),
    ]
    option_map = {
        "num_envs": "--num_envs",
        "voxel_size": "--voxel_size",
        "particles_per_cell": "--particles_per_cell",
        "dt": "--dt",
        "substeps": "--substeps",
        "warmup_steps": "--warmup_steps",
        "warmup_min_wall_s": "--warmup_min_wall_s",
        "measurement_duration": "--measurement_duration",
        "timing_batch_steps": "--timing_batch_steps",
        "gpu_monitor_interval_ms": "--gpu_monitor_interval_ms",
        "max_iterations": "--max_iterations",
        "tolerance": "--tolerance",
        "capacity_factor": "--capacity_factor",
        "collider_margin": "--collider_margin",
        "device": "--device",
    }
    for key, option in option_map.items():
        if key in case:
            command.extend((option, _format_cli_value(case[key])))
    if case.get("disable_cuda_graph", False):
        command.append("--disable_cuda_graph")
    if not case.get("project_outside_colliders", True):
        command.append("--no_project_outside_colliders")
    return command


def _t_critical_95(sample_count: int) -> float:
    """Return a two-sided 95% Student-t critical value."""
    table = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
        16: 2.131,
        17: 2.120,
        18: 2.110,
        19: 2.101,
        20: 2.093,
        21: 2.086,
        22: 2.080,
        23: 2.074,
        24: 2.069,
        25: 2.064,
        26: 2.060,
        27: 2.056,
        28: 2.052,
        29: 2.048,
        30: 2.045,
    }
    return table.get(sample_count, 1.960 if sample_count > 30 else math.nan)


def _case_key(row: dict[str, Any]) -> tuple[str, str]:
    """Return the stable sweep/case grouping key."""
    return str(row["sweep"]), str(row["case_id"])


def _planned_summary_base(case: dict[str, Any]) -> dict[str, Any]:
    """Create presentation columns even when every repetition was skipped or failed."""
    return {
        "sweep": case["sweep"],
        "case_id": case["case_id"],
        "case_label": case["case_label"],
        "x_parameter": case["x_parameter"],
        "x_value": case["x_value"],
        "x_unit": case.get("x_unit", ""),
        "num_envs": case["num_envs"],
        "voxel_size_m": case["voxel_size"],
        "particles_per_cell_axis": case["particles_per_cell"],
        "particles_per_voxel_3d": float(case["particles_per_cell"]) ** 3,
        "particles_per_env": case["estimated_particles_per_env"],
        "total_particles": case["estimated_total_particles"],
        "dt_s": case["dt"],
        "substeps": case["substeps"],
        "measured_simulated_s": case["measurement_duration"],
        "max_iterations": case["max_iterations"],
        "tolerance": case["tolerance"],
        "cuda_graph_requested": not case.get("disable_cuda_graph", False),
        "grid_type": "rebuildable_sparse",
        "max_active_cell_count": case["estimated_max_active_cell_count"],
        "max_leaf_node_count": case["estimated_max_leaf_node_count"],
        "max_lower_node_count": case["estimated_max_lower_node_count"],
        "max_upper_node_count": case["estimated_max_upper_node_count"],
        "preflight_reason": case.get("preflight_reason", ""),
    }


def _aggregate(
    rows: list[dict[str, Any]], planned: list[dict[str, Any]], outcomes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Outer-join planned cases with outcomes and aggregate successful fresh-process runs."""
    successful_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    planned_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    outcome_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        successful_groups[_case_key(row)].append(row)
    for case in planned:
        planned_groups[_case_key(case)].append(case)
    for outcome in outcomes:
        outcome_groups[_case_key(outcome)].append(outcome)

    summary_rows: list[dict[str, Any]] = []
    stable_success_fields = (
        "num_envs",
        "voxel_size_m",
        "particles_per_cell_axis",
        "particles_per_env",
        "total_particles",
        "dt_s",
        "substeps",
        "max_iterations",
        "tolerance",
        "max_active_cell_count",
        "max_leaf_node_count",
        "max_lower_node_count",
        "max_upper_node_count",
    )
    copied_success_fields = (
        "cuda_graph_active",
        "gpu_name",
        "git_commit",
        "git_dirty",
    )
    for key, planned_group in planned_groups.items():
        case = planned_group[0]
        group = successful_groups.get(key, [])
        group_outcomes = outcome_groups.get(key, [])
        summary = _planned_summary_base(case)
        if group:
            first = group[0]
            for field in stable_success_fields:
                values = {str(row.get(field)) for row in group}
                if len(values) != 1:
                    raise RuntimeError(f"Case {key} changed {field} across repetitions: {sorted(values)}")
                summary[field] = first[field]
            for field in copied_success_fields:
                summary[field] = first.get(field, "")

        status_counts: dict[str, int] = defaultdict(int)
        for outcome in group_outcomes:
            status_counts[str(outcome["status"])] += 1
        expected = len(planned_group)
        completed = len(group)
        failed = status_counts["failed"]
        skipped = status_counts["skipped_preflight"]
        pending = max(expected - completed - failed - skipped, 0)
        if completed == expected:
            status = "completed"
        elif completed:
            status = "partial"
        elif skipped == expected:
            status = "skipped_preflight"
        elif failed:
            status = "failed"
        else:
            status = "pending"
        summary.update(
            {
                "status": status,
                "expected_repetitions": expected,
                "completed_repetitions": completed,
                "failed_repetitions": failed,
                "skipped_repetitions": skipped,
                "pending_repetitions": pending,
            }
        )
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in group if row.get(metric) not in (None, "")]
            if not values:
                continue
            mean = statistics.fmean(values)
            standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
            t_value = _t_critical_95(len(values))
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = standard_deviation
            summary[f"{metric}_cv_percent"] = standard_deviation / abs(mean) * 100.0 if mean else 0.0
            summary[f"{metric}_median"] = statistics.median(values)
            summary[f"{metric}_min"] = min(values)
            summary[f"{metric}_max"] = max(values)
            summary[f"{metric}_ci95_half_width"] = (
                t_value * standard_deviation / math.sqrt(len(values)) if len(values) > 1 else ""
            )
        summary_rows.append(summary)
    summary_rows.sort(key=lambda row: (str(row["sweep"]), float(row["x_value"])))

    scaling_rows = [
        row
        for row in summary_rows
        if row["sweep"] == "environment_scaling"
        and row["status"] == "completed"
        and "aggregate_env_steps_per_s_mean" in row
    ]
    baseline = next((row for row in scaling_rows if int(row["num_envs"]) == 1), None)
    if baseline is not None:
        baseline_throughput = float(baseline["aggregate_env_steps_per_s_mean"])
        for row in scaling_rows:
            throughput = float(row["aggregate_env_steps_per_s_mean"])
            environment_ratio = int(row["num_envs"])
            row["throughput_speedup_vs_env_1"] = throughput / baseline_throughput
            row["parallel_scaling_efficiency"] = throughput / (baseline_throughput * environment_ratio)
            row["latency_inflation_vs_env_1"] = float(row["step_time_mean_ms_mean"]) / float(
                baseline["step_time_mean_ms_mean"]
            )

    timestep_rows = [
        row
        for row in summary_rows
        if row["sweep"] == "time_step_fixed_scene"
        and row["status"] == "completed"
        and "final_aabb_x_mean_m_mean" in row
    ]
    if timestep_rows:
        reference = min(timestep_rows, key=lambda row: float(row["dt_s"]))
        signature_fields = (
            "root_displacement_mean_m_mean",
            "final_aabb_x_mean_m_mean",
            "final_aabb_y_mean_m_mean",
            "final_aabb_z_mean_m_mean",
            "final_particle_rms_radius_mean_m_mean",
        )
        reference_values = [float(reference[field]) for field in signature_fields]
        reference_norm = math.sqrt(sum(value * value for value in reference_values))
        for row in timestep_rows:
            values = [float(row[field]) for field in signature_fields]
            difference_norm = math.sqrt(
                sum(
                    (value - reference_value) ** 2
                    for value, reference_value in zip(values, reference_values, strict=True)
                )
            )
            row["final_shape_signature_relative_error_vs_finest_percent"] = (
                100.0 * difference_norm / reference_norm if reference_norm else 0.0
            )
            row["final_shape_reference_dt_s"] = reference["dt_s"]
    return summary_rows


def _load_result_for_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    """Load and validate the successful result belonging to one outcome."""
    result_path = Path(str(outcome["result_path"]))
    with result_path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"{result_path} does not contain a JSON object")
    for section in ("identity", "configuration", "flat"):
        if not isinstance(result.get(section), dict):
            raise ValueError(f"{result_path} is missing object section {section!r}")
    if not isinstance(result.get("schema_version"), str) or not result["schema_version"]:
        raise ValueError(f"{result_path} has no schema version")

    identity = result["identity"]
    expected_identity = {
        "suite_id": outcome["suite_id"],
        "sweep": outcome["sweep"],
        "case_id": outcome["case_id"],
        "replicate": outcome["replicate"],
        "run_order": outcome["run_order"],
        "status": "completed",
    }
    for field, expected in expected_identity.items():
        if identity.get(field) != expected:
            raise ValueError(f"{result_path} identity {field!r} is {identity.get(field)!r}; expected {expected!r}")

    configuration = result["configuration"]
    expected_configuration = {
        "num_envs": outcome["num_envs"],
        "voxel_size_m": outcome["voxel_size_m"],
        "particles_per_cell_axis": outcome["particles_per_cell_axis"],
        "particles_per_env": outcome["estimated_particles_per_env"],
        "total_particles": outcome["estimated_total_particles"],
        "max_active_cell_count": outcome["estimated_max_active_cell_count"],
        "max_leaf_node_count": outcome["estimated_max_leaf_node_count"],
        "max_lower_node_count": outcome["estimated_max_lower_node_count"],
        "max_upper_node_count": outcome["estimated_max_upper_node_count"],
    }
    for field, expected in expected_configuration.items():
        if configuration.get(field) != expected:
            raise ValueError(
                f"{result_path} configuration {field!r} is {configuration.get(field)!r}; expected {expected!r}"
            )

    flat = result["flat"]
    if flat.get("schema_version") != result["schema_version"]:
        raise ValueError(f"{result_path} flat row has a mismatched schema version")
    for field, expected in expected_identity.items():
        if flat.get(field) != expected:
            raise ValueError(f"{result_path} flat identity {field!r} is {flat.get(field)!r}; expected {expected!r}")
    for field, expected in expected_configuration.items():
        if flat.get(field) != expected:
            raise ValueError(
                f"{result_path} flat configuration {field!r} is {flat.get(field)!r}; expected {expected!r}"
            )
    return flat


def _collect_results(suite_dir: Path) -> list[dict[str, Any]]:
    """Read validated child results referenced by completed outcomes only."""
    outcomes = _read_json(suite_dir / "outcomes.json", [])
    successful: list[dict[str, Any]] = []
    seen_runs: set[tuple[str, str, int, int]] = set()
    for outcome in outcomes:
        if outcome.get("status") != "completed":
            continue
        run_key = (
            str(outcome["sweep"]),
            str(outcome["case_id"]),
            int(outcome["replicate"]),
            int(outcome["run_order"]),
        )
        if run_key in seen_runs:
            raise ValueError(f"Duplicate completed outcome identity: {run_key}")
        seen_runs.add(run_key)
        successful.append(_load_result_for_outcome(outcome))
    successful.sort(key=lambda row: int(row["run_order"]))
    return successful


def _read_json(path: Path, default: Any) -> Any:
    """Read a JSON artifact when it exists."""
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _classify_failure(exit_code: int, error: str, result_error: str) -> str:
    """Classify a failed child outcome without claiming that every signal kill was an OOM."""
    normalized = error.lower()
    if "capacity was exceeded" in normalized:
        return "sparse_grid_capacity_exceeded"
    if "out of memory" in normalized or "failed to allocate" in normalized:
        return "out_of_memory"
    if "failed to create volume" in normalized:
        return "likely_out_of_memory"
    if exit_code in (-9, 137):
        return "sigkill_or_likely_out_of_memory"
    if exit_code < 0:
        return "signal"
    if exit_code != 0:
        return "nonzero_exit"
    if result_error:
        return "invalid_or_missing_result"
    return "unknown"


def _write_aggregates(suite_dir: Path) -> None:
    """Regenerate long-form and presentation-ready suite CSVs, retaining N/A cases."""
    successful = _collect_results(suite_dir)
    planned = _read_json(suite_dir / "planned_runs.json", [])
    outcomes = _read_json(suite_dir / "outcomes.json", [])
    _write_csv(suite_dir / "runs.csv", successful, fieldnames=list(successful[0]) if successful else ["status"])
    _write_csv(suite_dir / "outcomes.csv", outcomes, fieldnames=None if outcomes else ["status"])
    summary = _aggregate(successful, planned, outcomes)
    _write_csv(suite_dir / "summary.csv", summary)
    for sweep in sorted({str(row["sweep"]) for row in summary}):
        presentation_rows = [row for row in summary if row["sweep"] == sweep and row["status"] == "completed"]
        _write_csv(suite_dir / f"presentation_{sweep}.csv", presentation_rows)
    failed = [row for row in outcomes if row["status"] == "failed"]
    skipped = [row for row in outcomes if row["status"] == "skipped_preflight"]
    if failed:
        _write_csv(suite_dir / "failures.csv", failed)
    if skipped:
        _write_csv(suite_dir / "skipped.csv", skipped)


def _gpu_snapshot() -> dict[str, Any]:
    """Return a lightweight pre-suite GPU snapshot."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    rows = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 8:
            continue
        rows.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "uuid": fields[2],
                "memory_total_mib": float(fields[3]),
                "memory_used_mib": float(fields[4]),
                "memory_free_mib": float(fields[5]),
                "utilization_percent": float(fields[6]),
                "temperature_c": float(fields[7]),
            }
        )
    return {"gpus": rows, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}


def _enforce_idle_gpu(manifest: dict[str, Any], snapshot: dict[str, Any], allow_busy: bool) -> None:
    """Reject benchmark collection under obvious GPU contention unless explicitly overridden."""
    if allow_busy or not snapshot.get("gpus"):
        return
    suite_cfg = manifest.get("suite", {})
    gpu = snapshot["gpus"][0]
    memory_limit = float(suite_cfg.get("max_initial_gpu_memory_used_mib", 4096.0))
    utilization_limit = float(suite_cfg.get("max_initial_gpu_utilization_percent", 25.0))
    violations = []
    if gpu["memory_used_mib"] > memory_limit:
        violations.append(f"memory used {gpu['memory_used_mib']:.0f} MiB > {memory_limit:.0f} MiB")
    if gpu["utilization_percent"] > utilization_limit:
        violations.append(f"utilization {gpu['utilization_percent']:.0f}% > {utilization_limit:.0f}%")
    if violations:
        raise RuntimeError(
            "GPU is not quiescent enough for presentation measurements ("
            + "; ".join(violations)
            + "). Stop unrelated GPU work or use --allow_busy_gpu only for a non-reportable smoke test."
        )


def _snapshot_sources(suite_dir: Path, manifest_path: Path, initial_gpu: dict[str, Any]) -> None:
    """Copy executable benchmark sources and record immutable provenance hashes."""
    source_dir = suite_dir / "benchmark_sources"
    source_dir.mkdir()
    source_paths = [
        POINT_SCRIPT,
        Path(__file__).resolve(),
        manifest_path,
        SCRIPT_DIR / "PLAN.md",
        SCRIPT_DIR / "README.md",
    ]
    copied = []
    for source in source_paths:
        if not source.exists():
            continue
        destination = source_dir / source.name
        shutil.copy2(source, destination)
        copied.append(
            {
                "source": str(source),
                "snapshot": str(destination.relative_to(suite_dir)),
                "sha256": _sha256(destination),
            }
        )
    patch_path = source_dir / "tracked_repository_changes.patch"
    with patch_path.open("w", encoding="utf-8") as stream:
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "diff", "--binary"],
            stdout=stream,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
    provenance = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "working_directory": os.getcwd(),
        "sources": copied,
        "tracked_repository_patch_sha256": _sha256(patch_path),
        "initial_gpu": initial_gpu,
    }
    _write_json(suite_dir / "suite_provenance.json", provenance)


def main() -> None:
    """Resolve, randomize, run, and aggregate a suite manifest."""
    args = create_parser().parse_args()
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    cases = _resolve_cases(manifest, args.repetitions, set(args.only_sweep))
    suite_seed = int(manifest.get("suite", {}).get("randomization_seed", 20260909))
    cases = _randomized_block_order(cases, suite_seed)
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard_index < num_shards.")
    globally_ordered = [{**case, "global_run_order": order} for order, case in enumerate(cases, start=1)]
    cases = [
        case for case in globally_ordered if (int(case["global_run_order"]) - 1) % args.num_shards == args.shard_index
    ]
    suite_id = args.suite_id or f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_mpm_presentation"
    suite_dir = args.output_root.resolve() / suite_id
    if suite_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing suite: {suite_dir}")

    print(
        f"[SUITE] {suite_id}: shard {args.shard_index + 1}/{args.num_shards}, {len(cases)} independent runs",
        flush=True,
    )
    if args.dry_run:
        for case in cases:
            run_order = int(case["global_run_order"])
            print(
                f"{run_order:03d} {case['sweep']}/{case['case_id']} replicate={case['replicate']} "
                f"envs={case.get('num_envs')} voxel={case.get('voxel_size')} ppc={case.get('particles_per_cell')} "
                f"dt={case.get('dt')} particles={case['estimated_total_particles']:,} "
                f"grid(active/leaf/lower/upper)={case['estimated_max_active_cell_count']:,}/"
                f"{case['estimated_max_leaf_node_count']:,}/"
                f"{case['estimated_max_lower_node_count']:,}/"
                f"{case['estimated_max_upper_node_count']:,} "
                f"status={'SKIP: ' + case['preflight_reason'] if case['preflight_reason'] else 'run'}",
                flush=True,
            )
        return

    initial_gpu = _gpu_snapshot()
    _enforce_idle_gpu(manifest, initial_gpu, args.allow_busy_gpu)
    suite_dir.mkdir(parents=True)
    (suite_dir / "runs").mkdir()
    shutil.copy2(manifest_path, suite_dir / "manifest.toml")
    planned = [{**case, "run_order": int(case["global_run_order"])} for case in cases]
    _write_json(suite_dir / "planned_runs.json", planned)
    _write_csv(suite_dir / "planned_runs.csv", planned)
    _snapshot_sources(suite_dir, manifest_path, initial_gpu)
    outcomes: list[dict[str, Any]] = []
    for shard_run_index, case in enumerate(cases, start=1):
        run_order = int(case["global_run_order"])
        run_dir = suite_dir / "runs" / str(case["sweep"]) / str(case["case_id"]) / f"replicate_{case['replicate']:02d}"
        log_path = run_dir.parent / f"replicate_{case['replicate']:02d}.log"
        command = _point_command(case, suite_id, run_dir, run_order)
        outcome = {
            "suite_id": suite_id,
            "run_order": run_order,
            "sweep": case["sweep"],
            "case_id": case["case_id"],
            "case_label": case["case_label"],
            "replicate": case["replicate"],
            "num_envs": case["num_envs"],
            "voxel_size_m": case["voxel_size"],
            "particles_per_cell_axis": case["particles_per_cell"],
            "estimated_particles_per_env": case["estimated_particles_per_env"],
            "estimated_total_particles": case["estimated_total_particles"],
            "estimated_max_active_cell_count": case["estimated_max_active_cell_count"],
            "estimated_max_leaf_node_count": case["estimated_max_leaf_node_count"],
            "estimated_max_lower_node_count": case["estimated_max_lower_node_count"],
            "estimated_max_upper_node_count": case["estimated_max_upper_node_count"],
            "command": shlex.join(command),
            "log_path": str(log_path),
            "result_path": str(run_dir / "result.json"),
        }
        if case["preflight_reason"]:
            outcome.update(
                {
                    "status": "skipped_preflight",
                    "reason": case["preflight_reason"],
                    "started_utc": "",
                    "finished_utc": datetime.now(UTC).isoformat(),
                    "exit_code": "",
                }
            )
            outcomes.append(outcome)
            _write_json(suite_dir / "outcomes.json", outcomes)
            _write_aggregates(suite_dir)
            print(
                f"[{shard_run_index}/{len(cases)}] SKIPPED "
                f"{case['sweep']}/{case['case_id']}: {case['preflight_reason']}",
                flush=True,
            )
            continue
        print(
            f"[{shard_run_index}/{len(cases)}] {case['sweep']}/{case['case_id']} r{case['replicate']} ...",
            flush=True,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
        started = datetime.now(UTC).isoformat()
        with log_path.open("w", encoding="utf-8") as log_stream:
            log_stream.write(f"COMMAND={shlex.join(command)}\n")
            log_stream.flush()
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        failure_path = run_dir / "failure.json"
        failure = _read_json(failure_path, {}) if failure_path.exists() else {}
        error_type = str(failure.get("error_type", ""))
        error = str(failure.get("error", ""))
        result_error = ""
        if completed.returncode == 0:
            try:
                _load_result_for_outcome(outcome)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as validation_error:
                result_error = f"{type(validation_error).__name__}: {validation_error}"
        status = "completed" if completed.returncode == 0 and not result_error else "failed"
        if completed.returncode != 0:
            reason = f"child exited with code {completed.returncode}"
        else:
            reason = result_error
        outcome.update(
            {
                "status": status,
                "reason": reason,
                "started_utc": started,
                "finished_utc": datetime.now(UTC).isoformat(),
                "exit_code": completed.returncode,
            }
        )
        if status == "failed":
            outcome["failure_kind"] = _classify_failure(completed.returncode, error, result_error)
            outcome["error_type"] = error_type or ("InvalidResult" if result_error else "")
            outcome["error"] = error or result_error
        outcomes.append(outcome)
        _write_json(suite_dir / "outcomes.json", outcomes)
        _write_aggregates(suite_dir)
        outcome_text = "complete" if status == "completed" else f"FAILED ({completed.returncode})"
        print(f"[{shard_run_index}/{len(cases)}] {outcome_text}", flush=True)
        if status == "failed" and args.fail_fast:
            break

    _write_aggregates(suite_dir)
    successful = _collect_results(suite_dir)
    failed = sum(row["status"] == "failed" for row in outcomes)
    skipped = sum(row["status"] == "skipped_preflight" for row in outcomes)
    print(
        f"[DONE] successful={len(successful)} failed={failed} skipped={skipped} results={suite_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
