# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run one reproducible, headless Newton MPM granular benchmark point.

This is a presentation-benchmark harness, kept under the ignored ``outputs``
tree so it cannot accidentally become part of the three-PR integration branch.
Each invocation creates one immutable result directory containing raw timing
samples, one flat CSV row, and a self-describing JSON result.

The timed boundary is one standard Isaac Lab tick:
``sim.step(render=False)`` followed by ``scene.update(dt)``. Rebuildable-sparse
MPM checks grid status on the host after every stock simulation step, so the
reported step distribution is completed-work latency rather than enqueue time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import socket
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from isaaclab.app import launch_simulation

SCHEMA_VERSION = "1.0"
BENCHMARK_NAME = "newton_mpm_granular_cylinder_impact"
SPECIMEN_LO = (-0.34, -0.25, -0.36)
SPECIMEN_HI = (0.34, 0.25, 0.36)
BLOCK_EXTENT = tuple(upper - lower for lower, upper in zip(SPECIMEN_LO, SPECIMEN_HI, strict=True))
SPECIMEN_INITIAL_Z = 1.55
GRAVITY = (0.0, 0.0, -9.81)
JITTER_SEED = 42
JITTER_FRACTION = 0.30
ENV_SPACING = 2.0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def create_parser() -> argparse.ArgumentParser:
    """Create the single-point benchmark argument parser."""
    parser = argparse.ArgumentParser(description="Benchmark a replicated Newton MPM granular-impact workload.")
    parser.add_argument("--output_dir", type=Path, required=True, help="New immutable directory for this run.")
    parser.add_argument("--suite_id", required=True, help="Parent benchmark-suite identifier.")
    parser.add_argument("--sweep", required=True, help="One-factor study name.")
    parser.add_argument("--case_id", required=True, help="Stable case identifier within the study.")
    parser.add_argument("--case_label", required=True, help="Presentation-facing case label.")
    parser.add_argument("--x_parameter", required=True, help="Presentation x-axis field.")
    parser.add_argument("--x_value", type=float, required=True, help="Numeric presentation x-axis value.")
    parser.add_argument("--x_unit", default="", help="Presentation x-axis unit.")
    parser.add_argument("--replicate", type=int, required=True, help="Zero-based independent repetition index.")
    parser.add_argument("--run_order", type=_positive_int, required=True, help="One-based execution order.")
    parser.add_argument("--num_envs", type=_positive_int, default=1, help="Independent parallel MPM worlds.")
    parser.add_argument("--voxel_size", type=_positive_float, default=0.1, help="MPM voxel edge length [m].")
    parser.add_argument(
        "--particles_per_cell",
        type=_positive_float,
        default=2.0,
        help="Particle sampling multiplier per voxel axis; 2 means eight particles per 3D voxel.",
    )
    parser.add_argument("--dt", type=_positive_float, default=1.0 / 120.0, help="Outer simulation timestep [s].")
    parser.add_argument("--substeps", type=_positive_int, default=1, help="Newton solver substeps per outer step.")
    parser.add_argument(
        "--warmup_steps",
        type=_positive_int,
        default=50,
        help="Excluded warm-up steps, including deferred CUDA-graph capture; state is reset afterward.",
    )
    parser.add_argument(
        "--warmup_min_wall_s",
        type=_nonnegative_float,
        default=1.0,
        help="Minimum excluded warm-up wall time [s], in addition to --warmup_steps.",
    )
    parser.add_argument(
        "--measurement_duration", type=_positive_float, default=6.0, help="Measured simulated duration [s]."
    )
    parser.add_argument(
        "--timing_batch_steps",
        type=_positive_int,
        default=20,
        help="Steps between GPU synchronization boundaries used for timing samples.",
    )
    parser.add_argument(
        "--gpu_monitor_interval_ms",
        type=_positive_int,
        default=500,
        help="Sampling interval for contextual nvidia-smi telemetry [ms].",
    )
    parser.add_argument("--max_iterations", type=_positive_int, default=250, help="Rheology solver iteration cap.")
    parser.add_argument("--tolerance", type=_positive_float, default=1.0e-4, help="Rheology solver tolerance.")
    parser.add_argument(
        "--capacity_factor",
        type=_positive_float,
        default=8.0,
        help="Active-cell capacity relative to initially occupied voxels, before power-of-two rounding.",
    )
    parser.add_argument(
        "--collider_margin",
        type=_positive_float,
        default=0.05,
        help="Fixed physical contact margin for the floor and cylinder [m].",
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device used by Newton.")
    parser.add_argument(
        "--disable_cuda_graph", action="store_true", help="Disable Newton CUDA graph replay for an explicit ablation."
    )
    parser.add_argument(
        "--no_project_outside_colliders",
        action="store_true",
        help="Disable the granular demo's post-substep particle/collider projection pass.",
    )
    parser.add_argument(
        "--sample_grid_topology",
        action="store_true",
        help="Sample active sparse-grid topology after every step for untimed capacity calibration.",
    )
    return parser


def _next_power_of_two(value: float, minimum: int = 1) -> int:
    """Return the smallest power of two no smaller than ``value`` and ``minimum``."""
    required = max(int(math.ceil(value)), minimum)
    return 1 << (required - 1).bit_length()


def expected_particle_count(voxel_size: float, particles_per_cell: float) -> int:
    """Return the exact paper-inspired cuboid particle count per environment."""
    spacing = voxel_size / particles_per_cell
    axis_counts = [max(math.ceil(extent / spacing - 0.5 - 1.0e-12), 1) for extent in BLOCK_EXTENT]
    return math.prod(axis_counts)


def initial_grid_cell_count(voxel_size: float) -> int:
    """Return the number of physical MPM voxels initially covered by the block."""
    resolution = [max(math.ceil(extent / voxel_size), 1) for extent in BLOCK_EXTENT]
    return math.prod(resolution)


def derive_grid_capacities(num_envs: int, voxel_size: float, capacity_factor: float) -> dict[str, int]:
    """Derive proportional sparse-grid capacities for isolated worlds.

    The active-cell budget scales with initial physical grid occupancy, not
    particle count. This keeps the fixed-voxel particle-sampling study from
    accidentally changing the grid workload.
    """
    initial_cells = num_envs * initial_grid_cell_count(voxel_size)
    active = _next_power_of_two(initial_cells * capacity_factor, minimum=1 << 12)
    leaf = _next_power_of_two(max(active / 4, num_envs * 256), minimum=1 << 8)
    # Isolated worlds occupy disjoint hierarchy branches even when the total
    # number of active cells is modest.  Retain per-world branch headroom as
    # well as the aggregate sparse-grid ratio.
    lower = _next_power_of_two(max(active / 32, num_envs * 64), minimum=1 << 5)
    upper = _next_power_of_two(max(active / 128, num_envs * 32), minimum=1 << 3)
    return {
        "max_active_cell_count": active,
        "max_leaf_node_count": min(leaf, active),
        "max_lower_node_count": min(lower, leaf),
        "max_upper_node_count": min(upper, lower),
    }


def _duration_to_steps(duration: float, dt: float, *, allow_zero: bool = False) -> int:
    """Convert a simulated duration to the nearest whole outer step."""
    steps = int(round(duration / dt))
    return max(steps, 0 if allow_zero else 1)


def _grid_topology_stats(solver) -> tuple[int, int, int, int]:
    """Return active cell/leaf/lower/upper counts from the solver's Warp volume."""
    stats = solver._scratchpad.grid.cell_grid.get_active_stats()
    return stats.voxel_count, stats.leaf_node_count, stats.lower_node_count, stats.upper_node_count


def _create_specimen_points(voxel_size: float, particles_per_cell: float):
    """Create the same deterministic cuboid lattice used by the tuning demo."""
    import numpy as np

    spacing = voxel_size / particles_per_cell
    x_axis = np.arange(SPECIMEN_LO[0] + 0.5 * spacing, SPECIMEN_HI[0], spacing)
    y_axis = np.arange(SPECIMEN_LO[1] + 0.5 * spacing, SPECIMEN_HI[1], spacing)
    z_axis = np.arange(SPECIMEN_LO[2] + 0.5 * spacing, SPECIMEN_HI[2], spacing)
    points = np.stack(np.meshgrid(x_axis, y_axis, z_axis, indexing="ij"), axis=-1).reshape(-1, 3)
    jitter_half_width = JITTER_FRACTION * spacing
    points += np.random.default_rng(JITTER_SEED).uniform(-jitter_half_width, jitter_half_width, points.shape)
    return points.astype(np.float32)


def _percentile(values: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty series."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = percentile / 100.0 * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(values: list[float]) -> dict[str, float | int]:
    """Return presentation-ready descriptive statistics for a non-empty series."""
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "std": standard_deviation,
        "cv_percent": standard_deviation / abs(mean) * 100.0 if mean else 0.0,
        "p50": _percentile(values, 50.0),
        "p95": _percentile(values, 95.0),
        "p99": _percentile(values, 99.0),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def _command_output(command: list[str]) -> str | None:
    """Return stripped command output, or ``None`` when unavailable."""
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    """Capture the repository revision and dirty state."""
    commit = _command_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    branch = _command_output(["git", "-C", str(repo_root), "branch", "--show-current"])
    status = _command_output(["git", "-C", str(repo_root), "status", "--porcelain"])
    return {"commit": commit, "branch": branch, "dirty": bool(status), "status_porcelain": status or ""}


def _package_version(distribution: str) -> str | None:
    """Return an installed distribution version when available."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


_GPU_FIELDS = (
    "timestamp",
    "index",
    "name",
    "uuid",
    "driver_version",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "power.draw",
    "temperature.gpu",
    "clocks.sm",
)


def _parse_optional_float(value: str) -> float | None:
    """Parse one numeric ``nvidia-smi`` value."""
    try:
        return float(value.strip())
    except ValueError:
        return None


def _parse_gpu_row(line: str) -> dict[str, Any] | None:
    """Parse one no-header, no-units ``nvidia-smi`` query row."""
    values = [value.strip() for value in line.split(",")]
    if len(values) != len(_GPU_FIELDS):
        return None
    row: dict[str, Any] = dict(zip(_GPU_FIELDS, values, strict=True))
    for field in (
        "index",
        "memory.total",
        "memory.used",
        "utilization.gpu",
        "power.draw",
        "temperature.gpu",
        "clocks.sm",
    ):
        row[field] = _parse_optional_float(row[field])
    return row


def _gpu_index(device: str) -> int:
    """Extract a CUDA index from an Isaac Lab device string."""
    if not device.startswith("cuda"):
        raise ValueError(f"Newton MPM benchmark requires a CUDA device, received {device!r}.")
    _, separator, suffix = device.partition(":")
    return int(suffix) if separator else 0


def _nvidia_smi_selector(device: str) -> str:
    """Map a process-local CUDA device to its physical index or UUID."""
    logical_index = _gpu_index(device)
    visible_devices = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value]
    if visible_devices:
        if logical_index >= len(visible_devices):
            raise ValueError(f"CUDA device {device!r} is outside CUDA_VISIBLE_DEVICES={','.join(visible_devices)!r}.")
        selector = visible_devices[logical_index]
    else:
        selector = str(logical_index)
    inventory = _command_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
    rows = (
        []
        if not inventory
        else [tuple(value.strip() for value in line.split(",", 1)) for line in inventory.splitlines()]
    )
    if len(rows) == 1:
        return rows[0][1]
    if selector.isdigit() and any(index == selector for index, _ in rows):
        return selector
    if logical_index < len(rows):
        return rows[logical_index][1]
    return selector


def query_gpu(device: str) -> dict[str, Any]:
    """Capture one GPU state snapshot through ``nvidia-smi``."""
    output = _command_output(
        [
            "nvidia-smi",
            "-i",
            _nvidia_smi_selector(device),
            f"--query-gpu={','.join(_GPU_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return {}
    return _parse_gpu_row(output.splitlines()[0]) or {}


class GpuMonitor:
    """Collect low-frequency GPU telemetry during the measured window."""

    def __init__(self, device: str, interval_ms: int = 200) -> None:
        self._device = device
        self._interval_ms = interval_ms
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        """Start one persistent ``nvidia-smi`` query process."""
        command = [
            "nvidia-smi",
            "-i",
            _nvidia_smi_selector(self._device),
            f"--query-gpu={','.join(_GPU_FIELDS)}",
            "--format=csv,noheader,nounits",
            "-lms",
            str(self._interval_ms),
        ]
        try:
            self._process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except OSError:
            self._process = None

    def stop(self) -> list[dict[str, Any]]:
        """Stop telemetry collection and return parsed samples."""
        if self._process is None:
            return []
        self._process.terminate()
        try:
            stdout, _ = self._process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            self._process.kill()
            stdout, _ = self._process.communicate(timeout=3)
        self._process = None
        return [row for line in stdout.splitlines() if (row := _parse_gpu_row(line)) is not None]


def _telemetry_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize numeric GPU telemetry fields."""
    summary: dict[str, Any] = {"sample_count": len(samples)}
    for field, output_name in (
        ("memory.used", "memory_used_mib"),
        ("utilization.gpu", "utilization_percent"),
        ("power.draw", "power_draw_w"),
        ("temperature.gpu", "temperature_c"),
        ("clocks.sm", "sm_clock_mhz"),
    ):
        values = [float(row[field]) for row in samples if row.get(field) is not None]
        if values:
            summary[output_name] = _summary(values)
    return summary


def _create_sim_cfg(args: argparse.Namespace, capacities: dict[str, int]):
    """Create the headless Newton MPM simulation configuration."""
    from isaaclab_newton.physics import MPMSolverCfg, NewtonCfg

    import isaaclab.sim as sim_utils

    return sim_utils.SimulationCfg(
        dt=args.dt,
        device=args.device,
        gravity=GRAVITY,
        visualizer_cfgs=[],
        physics=NewtonCfg(
            solver_cfg=MPMSolverCfg(
                voxel_size=args.voxel_size,
                grid_type="sparse",
                grid_padding=0,
                max_active_cell_count=capacities["max_active_cell_count"],
                max_leaf_node_count=capacities["max_leaf_node_count"],
                max_lower_node_count=capacities["max_lower_node_count"],
                max_upper_node_count=capacities["max_upper_node_count"],
                separate_worlds=True,
                max_iterations=args.max_iterations,
                tolerance=args.tolerance,
                solver="auto",
                warmstart_mode="auto",
                transfer_scheme="apic",
                integration_scheme="pic",
                strain_basis="P0",
                velocity_basis="Q1",
                collider_basis="S2",
                air_drag=1.0e-3,
                project_outside_colliders=not args.no_project_outside_colliders,
            ),
            num_substeps=args.substeps,
            use_cuda_graph=not args.disable_cuda_graph,
        ),
    )


def _create_scene_cfg(args: argparse.Namespace, particle_points):
    """Create isolated replicas of the granular impact scene."""
    from isaaclab_newton.assets import MPMObjectCfg
    from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils.configclass import configclass

    particle_spacing = args.voxel_size / args.particles_per_cell
    collision_props = sim_utils.NewtonCollisionPropertiesCfg(
        collision_enabled=True,
        contact_margin=args.collider_margin,
    )
    collider_material = sim_utils.NewtonMaterialPropertiesCfg(static_friction=0.55, dynamic_friction=0.55)

    @configclass
    class GranularBenchmarkSceneCfg(InteractiveSceneCfg):
        """Replicated, renderer-free granular impact workload."""

        floor = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Floor",
            spawn=sim_utils.CuboidCfg(
                size=(2.0, 2.0, 0.1),
                collision_props=collision_props,
                physics_material=collider_material,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
        )
        cylinder = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Cylinder",
            spawn=sim_utils.CylinderCfg(
                radius=0.20,
                height=1.15,
                axis="Y",
                collision_props=collision_props,
                physics_material=collider_material,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.12, 0.42)),
        )
        media = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/GranularMedia",
            spawn=MPMPointsCfg(
                positions=particle_points.tolist(),
                mass=particle_spacing**3 * 1000.0,
                radius=0.5 * particle_spacing,
                visible=False,
                material=MPMParticleMaterialCfg(
                    density=1000.0,
                    young_modulus=1.0e15,
                    poisson_ratio=0.3,
                    friction=0.68,
                    yield_pressure=1.0e12,
                ),
            ),
            init_state=MPMObjectCfg.InitialStateCfg(pos=(0.0, 0.0, SPECIMEN_INITIAL_Z)),
        )

    return GranularBenchmarkSceneCfg(num_envs=args.num_envs, env_spacing=ENV_SPACING)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically write a JSON document."""
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a non-empty homogeneous row collection as CSV."""
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    """Create the one-row CSV representation of a successful result."""
    identity = result["identity"]
    config = result["configuration"]
    metrics = result["metrics"]
    validation = result["validation"]
    gpu_static = result["hardware"].get("gpu_before", {})
    telemetry = result["hardware"].get("gpu_telemetry", {})
    flat = {
        "schema_version": result["schema_version"],
        "suite_id": identity["suite_id"],
        "run_id": identity["run_id"],
        "sweep": identity["sweep"],
        "case_id": identity["case_id"],
        "case_label": identity["case_label"],
        "x_parameter": identity["x_parameter"],
        "x_value": identity["x_value"],
        "x_unit": identity["x_unit"],
        "replicate": identity["replicate"],
        "run_order": identity["run_order"],
        "timestamp_utc": identity["timestamp_utc"],
        "status": identity["status"],
        "num_envs": config["num_envs"],
        "voxel_size_m": config["voxel_size_m"],
        "particles_per_cell_axis": config["particles_per_cell_axis"],
        "particles_per_voxel_3d": config["particles_per_voxel_3d"],
        "particle_spacing_m": config["particle_spacing_m"],
        "particles_per_env": config["particles_per_env"],
        "total_particles": config["total_particles"],
        "particle_mass_kg": config["particle_mass_kg"],
        "total_material_mass_per_env_kg": config["total_material_mass_per_env_kg"],
        "initial_particle_positions_sha256": config["initial_particle_positions_sha256"],
        "dt_s": config["dt_s"],
        "substeps": config["substeps"],
        "solver_substep_dt_s": config["solver_substep_dt_s"],
        "warmup_steps": config["warmup_steps"],
        "warmup_simulated_s": config["warmup_simulated_s"],
        "measured_steps": config["measured_steps"],
        "measured_simulated_s": config["measured_simulated_s"],
        "timing_batch_steps": config["timing_batch_steps"],
        "max_iterations": config["max_iterations"],
        "tolerance": config["tolerance"],
        "cuda_graph_requested": config["cuda_graph_requested"],
        "cuda_graph_active": config["cuda_graph_active"],
        "grid_type": config["grid_type"],
        "separate_worlds": config["separate_worlds"],
        "project_outside_colliders": config["project_outside_colliders"],
        "max_active_cell_count": config["max_active_cell_count"],
        "max_leaf_node_count": config["max_leaf_node_count"],
        "max_lower_node_count": config["max_lower_node_count"],
        "max_upper_node_count": config["max_upper_node_count"],
        "setup_wall_s": metrics["setup_wall_s"],
        "first_step_wall_s": metrics["first_step_wall_s"],
        "warmup_wall_s": metrics["warmup_wall_s"],
        "measured_wall_s": metrics["measured_wall_s"],
        "wall_seconds_per_simulated_second": metrics["wall_seconds_per_simulated_second"],
        "step_time_mean_ms": metrics["overall_step_time_ms"],
        "step_sample_mean_ms": metrics["step_time_ms"]["mean"],
        "step_time_std_ms": metrics["step_time_ms"]["std"],
        "step_time_p50_ms": metrics["step_time_ms"]["p50"],
        "step_time_p95_ms": metrics["step_time_ms"]["p95"],
        "step_time_p99_ms": metrics["step_time_ms"]["p99"],
        "step_time_cv_percent": metrics["step_time_ms"]["cv_percent"],
        "realtime_budget_ms": metrics["realtime_budget_ms"],
        "p95_within_realtime_budget": metrics["p95_within_realtime_budget"],
        "presentation_budget_ms": metrics["presentation_budget_ms"],
        "p95_within_4ms_budget": metrics["p95_within_4ms_budget"],
        "outer_steps_per_s": metrics["outer_steps_per_s"],
        "aggregate_env_steps_per_s": metrics["aggregate_env_steps_per_s"],
        "particle_updates_per_s": metrics["particle_updates_per_s"],
        "million_particle_updates_per_s": metrics["million_particle_updates_per_s"],
        "particle_substep_updates_per_s": metrics["particle_substep_updates_per_s"],
        "million_particle_substep_updates_per_s": metrics["million_particle_substep_updates_per_s"],
        "single_world_realtime_factor": metrics["single_world_realtime_factor"],
        "aggregate_world_realtime_factor": metrics["aggregate_world_realtime_factor"],
        "impact_phase_step_time_mean_ms": metrics["phase_step_time_ms"].get("impact_0_to_1s", {}).get("mean", ""),
        "flow_phase_step_time_mean_ms": metrics["phase_step_time_ms"].get("flow_1_to_3s", {}).get("mean", ""),
        "settling_phase_step_time_mean_ms": metrics["phase_step_time_ms"].get("settling_after_3s", {}).get("mean", ""),
        "gpu_name": gpu_static.get("name", ""),
        "gpu_uuid": gpu_static.get("uuid", ""),
        "driver_version": gpu_static.get("driver_version", ""),
        "gpu_total_memory_mib": gpu_static.get("memory.total", ""),
        "gpu_memory_peak_mib": telemetry.get("memory_used_mib", {}).get("max", ""),
        "gpu_memory_peak_delta_mib": metrics["gpu_memory_peak_delta_mib"],
        "gpu_utilization_mean_percent": telemetry.get("utilization_percent", {}).get("mean", ""),
        "gpu_power_mean_w": telemetry.get("power_draw_w", {}).get("mean", ""),
        "gpu_temperature_max_c": telemetry.get("temperature_c", {}).get("max", ""),
        "finite_state": validation["finite_state"],
        "evolved": validation["evolved"],
        "reset_position_max_error_m": validation["reset_position_max_error_m"],
        "reset_velocity_max_error_m_s": validation["reset_velocity_max_error_m_s"],
        "root_displacement_mean_m": validation["root_displacement_mean_m"],
        "final_aabb_x_mean_m": validation["final_aabb_extent_mean_m"][0],
        "final_aabb_y_mean_m": validation["final_aabb_extent_mean_m"][1],
        "final_aabb_z_mean_m": validation["final_aabb_extent_mean_m"][2],
        "final_particle_rms_radius_mean_m": validation["final_particle_rms_radius_mean_m"],
        "final_height_mean_m": validation["final_height_mean_m"],
        "final_height_min_m": validation["final_height_min_m"],
        "final_height_max_m": validation["final_height_max_m"],
        "final_speed_max_m_s": validation["final_speed_max_m_s"],
        "final_active_cell_count_private": (
            validation["final_active_cell_count_private"]
            if validation["final_active_cell_count_private"] is not None
            else ""
        ),
        "final_active_cell_capacity_utilization_percent": (
            validation["final_active_cell_capacity_utilization_percent"]
            if validation["final_active_cell_capacity_utilization_percent"] is not None
            else ""
        ),
        "final_active_cell_count_unavailable_reason": validation["final_active_cell_count_unavailable_reason"] or "",
        "git_commit": result["software"]["git"]["commit"] or "",
        "git_branch": result["software"]["git"]["branch"] or "",
        "git_dirty": result["software"]["git"]["dirty"],
        "newton_version": result["software"]["packages"].get("newton", "") or "",
        "warp_version": result["software"]["packages"].get("warp-lang", "") or "",
        "isaaclab_version": result["software"]["packages"].get("isaaclab", "") or "",
    }
    return flat


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one benchmark point and return its complete result document."""
    import torch
    import warp as wp
    from isaaclab_newton.physics import NewtonManager, NewtonMPMManager

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene

    repo_root = Path(__file__).resolve().parents[2]
    particle_points = _create_specimen_points(args.voxel_size, args.particles_per_cell)
    expected_per_env = expected_particle_count(args.voxel_size, args.particles_per_cell)
    if len(particle_points) != expected_per_env:
        raise RuntimeError(
            f"Analytic particle count {expected_per_env} disagrees with generated count {len(particle_points)}."
        )
    initial_particle_positions_sha256 = hashlib.sha256(particle_points.tobytes(order="C")).hexdigest()
    particle_spacing = args.voxel_size / args.particles_per_cell
    particle_mass = particle_spacing**3 * 1000.0
    capacities = derive_grid_capacities(args.num_envs, args.voxel_size, args.capacity_factor)
    measured_steps = _duration_to_steps(args.measurement_duration, args.dt)
    run_id = f"{args.case_id}_r{args.replicate:02d}_o{args.run_order:03d}"
    timestamp_utc = datetime.now(UTC).isoformat()
    gpu_before = query_gpu(args.device)
    process_start = time.perf_counter()

    sim_cfg = _create_sim_cfg(args, capacities)
    with launch_simulation(sim_cfg, args):
        setup_start = time.perf_counter()
        sim = sim_utils.SimulationContext(sim_cfg)
        scene = InteractiveScene(_create_scene_cfg(args, particle_points))
        sim.reset()
        scene.reset()
        media = scene["media"]
        actual_per_env = int(media.particles_per_object)
        actual_total = int(media.num_instances * actual_per_env)
        if media.num_instances != args.num_envs:
            raise RuntimeError(f"Expected {args.num_envs} MPM instances, received {media.num_instances}.")
        if actual_per_env != expected_per_env:
            raise RuntimeError(f"Expected {expected_per_env} particles/environment, received {actual_per_env}.")
        wp.synchronize_device(sim.device)
        setup_wall_s = time.perf_counter() - setup_start

        # Record deferred graph capture separately; it is never included in throughput.
        warmup_start = time.perf_counter()
        first_step_start = time.perf_counter()
        sim.step(render=False)
        scene.update(args.dt)
        wp.synchronize_device(sim.device)
        first_step_wall_s = time.perf_counter() - first_step_start
        warmup_steps_completed = 1
        while warmup_steps_completed < args.warmup_steps or time.perf_counter() - warmup_start < args.warmup_min_wall_s:
            sim.step(render=False)
            scene.update(args.dt)
            warmup_steps_completed += 1
        wp.synchronize_device(sim.device)
        warmup_wall_s = time.perf_counter() - warmup_start
        cuda_graph_active = bool(NewtonManager._graph is not None)
        if not args.disable_cuda_graph and not cuda_graph_active:
            raise RuntimeError("CUDA graph capture was requested but did not become active during warm-up.")

        # Warm-up must not remove the impact from the measured trajectory.
        scene.reset()
        NewtonMPMManager.reset_solver_state()
        wp.synchronize_device(sim.device)
        reset_state = media.data.particle_state_w.torch
        default_state = media.data.default_particle_state_w.torch
        reset_position_max_error_m = float((reset_state[..., :3] - default_state[..., :3]).abs().max().item())
        reset_velocity_max_error_m_s = float((reset_state[..., 3:] - default_state[..., 3:]).abs().max().item())
        if reset_position_max_error_m > 1.0e-6 or reset_velocity_max_error_m_s > 1.0e-6:
            raise RuntimeError(
                "Warm-up reset did not restore the initial particle state exactly enough: "
                f"position error={reset_position_max_error_m:g} m, "
                f"velocity error={reset_velocity_max_error_m_s:g} m/s."
            )
        initial_root = media.data.root_pos_w.torch.clone()
        wp.synchronize_device(sim.device)
        del reset_state, default_state
        torch.cuda.empty_cache()

        implicit_solvers = NewtonMPMManager._implicit_mpm_solvers()
        if len(implicit_solvers) != 1:
            raise RuntimeError(f"Expected one implicit MPM solver, received {len(implicit_solvers)}.")
        solver = implicit_solvers[0]

        raw_batch_samples: list[tuple[int, int, float]] = []
        raw_step_samples: list[tuple[int, float]] = []
        raw_grid_topology_samples: list[tuple[int, float, int, int, int, int]] = []
        topology_sampling_overhead_ns = 0
        if args.sample_grid_topology:
            topology_start_ns = time.perf_counter_ns()
            topology = _grid_topology_stats(solver)
            topology_sampling_overhead_ns += time.perf_counter_ns() - topology_start_ns
            raw_grid_topology_samples.append((-1, 0.0, *topology))
        completed_steps = 0
        monitor = GpuMonitor(args.device, interval_ms=args.gpu_monitor_interval_ms)
        monitor.start()
        measurement_start_ns = time.perf_counter_ns()
        try:
            while completed_steps < measured_steps:
                batch_steps = min(args.timing_batch_steps, measured_steps - completed_steps)
                start_step = completed_steps
                batch_start_ns = time.perf_counter_ns()
                batch_topology_sampling_overhead_ns = 0
                for _ in range(batch_steps):
                    step_index = completed_steps
                    step_start_ns = time.perf_counter_ns()
                    sim.step(render=False)
                    scene.update(args.dt)
                    if not cuda_graph_active:
                        wp.synchronize_device(sim.device)
                    step_finished_ns = time.perf_counter_ns()
                    completed_steps += 1
                    raw_step_samples.append((step_index, (step_finished_ns - step_start_ns) / 1.0e6))
                    if args.sample_grid_topology:
                        topology_start_ns = time.perf_counter_ns()
                        topology = _grid_topology_stats(solver)
                        topology_elapsed_ns = time.perf_counter_ns() - topology_start_ns
                        topology_sampling_overhead_ns += topology_elapsed_ns
                        batch_topology_sampling_overhead_ns += topology_elapsed_ns
                        raw_grid_topology_samples.append((step_index, (step_index + 1) * args.dt, *topology))
                wp.synchronize_device(sim.device)
                batch_finished_ns = time.perf_counter_ns()
                synchronized_wall_s = (batch_finished_ns - batch_start_ns - batch_topology_sampling_overhead_ns) / 1.0e9
                raw_batch_samples.append((start_step, completed_steps, synchronized_wall_s))
        finally:
            measurement_finished_ns = time.perf_counter_ns()
            telemetry_samples = monitor.stop()
        measured_wall_s = (measurement_finished_ns - measurement_start_ns - topology_sampling_overhead_ns) / 1.0e9
        step_samples = [
            {
                "step_index": step_index,
                "step_start_simulation_time_s": step_index * args.dt,
                "step_end_simulation_time_s": (step_index + 1) * args.dt,
                "phase": (
                    "impact_0_to_1s"
                    if step_index * args.dt < 1.0
                    else "flow_1_to_3s"
                    if step_index * args.dt < 3.0
                    else "settling_after_3s"
                ),
                "step_time_ms": step_time_ms,
            }
            for step_index, step_time_ms in raw_step_samples
        ]
        samples = [
            {
                "sample_index": sample_index,
                "start_step": start_step,
                "end_step": end_step,
                "start_simulation_time_s": start_step * args.dt,
                "end_simulation_time_s": end_step * args.dt,
                "num_steps": end_step - start_step,
                "synchronized_wall_s": synchronized_wall_s,
                "batch_average_step_time_ms": synchronized_wall_s / (end_step - start_step) * 1000.0,
                "outer_steps_per_s": (end_step - start_step) / synchronized_wall_s,
                "aggregate_env_steps_per_s": args.num_envs * (end_step - start_step) / synchronized_wall_s,
                "million_particle_updates_per_s": (
                    actual_total * (end_step - start_step) / synchronized_wall_s / 1.0e6
                ),
            }
            for sample_index, (start_step, end_step, synchronized_wall_s) in enumerate(raw_batch_samples)
        ]
        grid_topology_samples = [
            {
                "step_index": step_index,
                "simulation_time_s": simulation_time_s,
                "active_cell_count": active_cell_count,
                "leaf_node_count": leaf_node_count,
                "lower_node_count": lower_node_count,
                "upper_node_count": upper_node_count,
            }
            for (
                step_index,
                simulation_time_s,
                active_cell_count,
                leaf_node_count,
                lower_node_count,
                upper_node_count,
            ) in raw_grid_topology_samples
        ]

        final_pos = media.data.particle_pos_w.torch
        final_vel = media.data.particle_vel_w.torch
        final_root = media.data.root_pos_w.torch
        finite_state = bool(torch.isfinite(final_pos).all().item() and torch.isfinite(final_vel).all().item())
        if not finite_state:
            raise RuntimeError("The final MPM particle state contains non-finite values.")
        root_displacement = torch.linalg.vector_norm(final_root - initial_root, dim=-1)
        final_aabb_extent = final_pos.amax(dim=1) - final_pos.amin(dim=1)
        final_particle_rms_radius = torch.sqrt(
            torch.mean(torch.sum((final_pos - final_root.unsqueeze(1)) ** 2, dim=-1), dim=-1)
        )
        final_active_cell_count: int | None = None
        active_cell_unavailable_reason: str | None = None
        try:
            partition = solver._scratchpad.velocity_test.space_partition.geo_partition
            partition_cells = partition._cells.numpy()
            final_active_cell_count = int((partition_cells >= 0).sum())
        except Exception as error:
            active_cell_unavailable_reason = f"{type(error).__name__}: {error}"

        step_times_ms = [float(sample["step_time_ms"]) for sample in step_samples]
        step_time_summary = _summary(step_times_ms)
        overall_step_time_ms = measured_wall_s / measured_steps * 1000.0
        phase_step_times = {
            phase: _summary([float(sample["step_time_ms"]) for sample in step_samples if sample["phase"] == phase])
            for phase in ("impact_0_to_1s", "flow_1_to_3s", "settling_after_3s")
            if any(sample["phase"] == phase for sample in step_samples)
        }
        outer_steps_per_s = measured_steps / measured_wall_s
        aggregate_env_steps_per_s = args.num_envs * outer_steps_per_s
        particle_updates_per_s = actual_total * outer_steps_per_s
        realtime_budget_ms = args.dt * 1000.0
        presentation_budget_ms = 4.0
        telemetry_summary = _telemetry_summary(telemetry_samples)
        gpu_memory_peak_mib = telemetry_summary.get("memory_used_mib", {}).get("max")
        gpu_memory_before_mib = gpu_before.get("memory.used")
        gpu_memory_peak_delta_mib = (
            float(gpu_memory_peak_mib) - float(gpu_memory_before_mib)
            if gpu_memory_peak_mib is not None and gpu_memory_before_mib is not None
            else ""
        )
        root_displacement_mean_m = float(root_displacement.mean().item())
        evolved = root_displacement_mean_m > 1.0e-3
        if not evolved:
            raise RuntimeError("The granular specimen did not evolve measurably during the timed trajectory.")
        metrics = {
            "setup_wall_s": setup_wall_s,
            "first_step_wall_s": first_step_wall_s,
            "warmup_wall_s": warmup_wall_s,
            "measured_wall_s": measured_wall_s,
            "wall_seconds_per_simulated_second": measured_wall_s / (measured_steps * args.dt),
            "overall_step_time_ms": overall_step_time_ms,
            "step_time_ms": step_time_summary,
            "step_time_sample_definition": (
                "Completed-work Isaac Lab tick latency. CUDA-graph runs synchronize through the stock sparse-grid "
                "status check; eager runs add an explicit benchmark synchronization."
            ),
            "phase_step_time_ms": phase_step_times,
            "realtime_budget_ms": realtime_budget_ms,
            "p95_within_realtime_budget": step_time_summary["p95"] <= realtime_budget_ms,
            "presentation_budget_ms": presentation_budget_ms,
            "p95_within_4ms_budget": step_time_summary["p95"] <= presentation_budget_ms,
            "outer_steps_per_s": outer_steps_per_s,
            "aggregate_env_steps_per_s": aggregate_env_steps_per_s,
            "particle_updates_per_s": particle_updates_per_s,
            "million_particle_updates_per_s": particle_updates_per_s / 1.0e6,
            "particle_substep_updates_per_s": particle_updates_per_s * args.substeps,
            "million_particle_substep_updates_per_s": particle_updates_per_s * args.substeps / 1.0e6,
            "single_world_realtime_factor": args.dt * outer_steps_per_s,
            "aggregate_world_realtime_factor": args.num_envs * args.dt * outer_steps_per_s,
            "gpu_memory_peak_delta_mib": gpu_memory_peak_delta_mib,
            "topology_sampling_overhead_s": topology_sampling_overhead_ns / 1.0e9,
        }
        topology_peaks = (
            {
                field: max(int(sample[field]) for sample in grid_topology_samples)
                for field in ("active_cell_count", "leaf_node_count", "lower_node_count", "upper_node_count")
            }
            if grid_topology_samples
            else {}
        )
        final_active_cell_capacity_utilization = (
            100.0 * final_active_cell_count / capacities["max_active_cell_count"]
            if final_active_cell_count is not None
            else None
        )
        validation = {
            "finite_state": finite_state,
            "evolved": evolved,
            "reset_position_max_error_m": reset_position_max_error_m,
            "reset_velocity_max_error_m_s": reset_velocity_max_error_m_s,
            "root_displacement_mean_m": root_displacement_mean_m,
            "final_aabb_extent_mean_m": [float(value) for value in final_aabb_extent.mean(dim=0).tolist()],
            "final_particle_rms_radius_mean_m": float(final_particle_rms_radius.mean().item()),
            "final_height_mean_m": float(final_pos[..., 2].mean().item()),
            "final_height_min_m": float(final_pos[..., 2].min().item()),
            "final_height_max_m": float(final_pos[..., 2].max().item()),
            "final_speed_max_m_s": float(torch.linalg.vector_norm(final_vel, dim=-1).max().item()),
            "final_active_cell_count_private": final_active_cell_count,
            "final_active_cell_capacity_utilization_percent": final_active_cell_capacity_utilization,
            "final_active_cell_count_unavailable_reason": active_cell_unavailable_reason,
            "final_active_cell_count_note": (
                "Version-fragile final snapshot from ExplicitGeometryPartition._cells; not a peak active-cell count."
            ),
            "actual_solver_iterations": None,
            "solver_residual": None,
            "convergence_unavailable_reason": (
                "Newton 1.6 does not retain per-step iteration/residual diagnostics after the temporary solve data "
                "is released; configured max_iterations is a cap, not an observed iteration count."
            ),
            "grid_topology_peaks": topology_peaks,
            "grid_topology_sampling_note": (
                "Capacity-calibration diagnostic sampled after each completed step; sampling overhead was removed "
                "from aggregate timing but these runs should not be used as presentation throughput results."
                if args.sample_grid_topology
                else "Disabled for reportable throughput collection."
            ),
        }
        del initial_root, final_root, final_pos, final_vel, root_displacement, final_aabb_extent
        del final_particle_rms_radius
        torch.cuda.empty_cache()

    result = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_name": BENCHMARK_NAME,
        "identity": {
            "suite_id": args.suite_id,
            "run_id": run_id,
            "sweep": args.sweep,
            "case_id": args.case_id,
            "case_label": args.case_label,
            "x_parameter": args.x_parameter,
            "x_value": args.x_value,
            "x_unit": args.x_unit,
            "replicate": args.replicate,
            "run_order": args.run_order,
            "timestamp_utc": timestamp_utc,
            "status": "completed",
        },
        "configuration": {
            "num_envs": args.num_envs,
            "voxel_size_m": args.voxel_size,
            "particles_per_cell_axis": args.particles_per_cell,
            "particles_per_voxel_3d": args.particles_per_cell**3,
            "particle_spacing_m": args.voxel_size / args.particles_per_cell,
            "particle_mass_kg": particle_mass,
            "total_material_mass_per_env_kg": actual_per_env * particle_mass,
            "initial_particle_positions_sha256": initial_particle_positions_sha256,
            "jitter_half_width_m": JITTER_FRACTION * args.voxel_size / args.particles_per_cell,
            "jitter_fraction_of_spacing": JITTER_FRACTION,
            "jitter_seed": JITTER_SEED,
            "block_extent_m": BLOCK_EXTENT,
            "block_volume_m3": math.prod(BLOCK_EXTENT),
            "particles_per_env": actual_per_env,
            "total_particles": actual_total,
            "dt_s": args.dt,
            "substeps": args.substeps,
            "solver_substep_dt_s": args.dt / args.substeps,
            "warmup_minimum_steps": args.warmup_steps,
            "warmup_minimum_wall_s": args.warmup_min_wall_s,
            "warmup_steps": warmup_steps_completed,
            "warmup_simulated_s": warmup_steps_completed * args.dt,
            "reset_after_warmup": True,
            "simulation_clock_reset_after_warmup": False,
            "measured_steps": measured_steps,
            "measured_simulated_s": measured_steps * args.dt,
            "timing_batch_steps": args.timing_batch_steps,
            "gpu_monitor_interval_ms": args.gpu_monitor_interval_ms,
            "timed_boundary": "sim.step(render=False) + scene.update(dt)",
            "timing_mode": "per_step_completed_work_with_batch_boundary_sync",
            "grid_topology_sampling_enabled": args.sample_grid_topology,
            "per_step_sparse_grid_status_sync": cuda_graph_active,
            "explicit_per_step_sync": not cuda_graph_active,
            "max_iterations": args.max_iterations,
            "tolerance": args.tolerance,
            "cuda_graph_requested": not args.disable_cuda_graph,
            "cuda_graph_active": cuda_graph_active,
            "grid_type": "rebuildable_sparse",
            "grid_padding": 0,
            "separate_worlds": True,
            "capacity_factor": args.capacity_factor,
            **capacities,
            "solver_requested": "auto",
            "solver_resolved": list(getattr(solver, "solver", ())),
            "warmstart_mode_requested": "auto",
            "warmstart_mode_resolved": getattr(solver, "_stress_warmstart", None),
            "transfer_scheme": "apic",
            "integration_scheme": "pic",
            "strain_basis": "P0",
            "velocity_basis": "Q1",
            "collider_basis": "S2",
            "project_outside_colliders": not args.no_project_outside_colliders,
            "collider_margin_m": args.collider_margin,
            "material": {
                "density_kg_m3": 1000.0,
                "young_modulus_pa": 1.0e15,
                "poisson_ratio": 0.3,
                "friction": 0.68,
                "yield_pressure_pa": 1.0e12,
            },
            "device": args.device,
        },
        "metrics": metrics,
        "validation": validation,
        "samples": samples,
        "step_samples": step_samples,
        "grid_topology_samples": grid_topology_samples,
        "hardware": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "gpu_before": gpu_before,
            "gpu_after": query_gpu(args.device),
            "gpu_telemetry": telemetry_summary,
            "gpu_telemetry_samples": telemetry_samples,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "software": {
            "python": platform.python_version(),
            "packages": {
                distribution: _package_version(distribution)
                for distribution in ("isaaclab", "isaaclab-newton", "newton", "warp-lang", "torch", "numpy")
            },
            "git": _git_provenance(repo_root),
        },
        "provenance": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "working_directory": os.getcwd(),
            "process_wall_s": time.perf_counter() - process_start,
        },
    }
    result["flat"] = _flatten_result(result)
    return result


def main() -> None:
    """Parse arguments, execute one run, and write immutable artifacts."""
    args = create_parser().parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    try:
        result = run(args)
    except Exception as error:
        _write_json(
            args.output_dir / "failure.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "command": shlex.join([sys.executable, *sys.argv]),
                "timestamp_utc": datetime.now(UTC).isoformat(),
            },
        )
        raise
    _write_json(args.output_dir / "result.json", result)
    _write_csv(args.output_dir / "run.csv", [result["flat"]])
    _write_csv(args.output_dir / "samples.csv", result["samples"])
    _write_csv(args.output_dir / "step_samples.csv", result["step_samples"])
    _write_csv(args.output_dir / "gpu_samples.csv", result["hardware"]["gpu_telemetry_samples"])
    _write_csv(args.output_dir / "grid_topology_samples.csv", result["grid_topology_samples"])
    print(
        "[RESULT] "
        f"{result['identity']['case_id']} replicate={result['identity']['replicate']} "
        f"particles={result['configuration']['total_particles']:,} "
        f"step={result['metrics']['step_time_ms']['mean']:.3f} ms "
        f"aggregate={result['metrics']['aggregate_world_realtime_factor']:.2f}x realtime",
        flush=True,
    )
    print(f"MPM_BENCHMARK_RESULT={args.output_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
