# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a standalone HTML report from simulator comparison result files."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import shutil
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

MATCH_FIELDS = (
    "task",
    "vision",
    "mode",
    "num_envs",
    "dt",
    "decimation",
    "width",
    "height",
    "steps",
    "warmup_steps",
    "solver_iterations",
    "cpu_threads",
    "iterations",
    "rollout_steps",
    "evaluation_interval",
    "evaluation_steps",
    "initial_height",
    "initial_joint_pos",
    "joint_names",
    "friction",
    "kp",
    "kd",
    "effort_limit",
    "env_spacing",
    "camera_pos",
    "camera_lookat",
    "camera_fov",
)
COLORS = ("#245cab", "#b04b17", "#347a4a", "#8050a4", "#a43662", "#217e83")
BACKEND_LABELS = {
    "genesis": "Genesis",
    "isaaclab": "Isaac Lab · MJWarp",
    "isaaclab_physx": "Isaac Lab · PhysX (reference)",
    "mjlab": "mjlab",
}
TASK_ERRORS = {
    "anymal": ("tracking_error_m_s", "Speed tracking error [m/s]", "m/s"),
    "cartpole": ("pole_angle_abs_rad", "Pole angle error [rad]", "rad"),
    "franka": ("reaching_error_m", "End-effector reaching error [m]", "m"),
}
METRICS = {
    "startup_s": ("Worker startup", "s", False),
    "first_step_s": ("First step", "s", False),
    "env_steps_per_s": ("Environment throughput", "env transitions/s", True),
    "physics_steps_per_s": ("Physics throughput", "env physics steps/s", True),
    "render_frames_per_s": ("Image throughput", "frames/s", True),
    "training_s": ("Training time", "s", False),
    "process_wall_s": ("Complete worker process", "s", False),
}


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _format(value: float) -> str:
    return f"{value:,.3g}"


def _value(record: dict, path: str) -> object:
    value = record
    for field in path.split("."):
        value = value.get(field) if isinstance(value, dict) else None
    return value


def _statistics(runs: list[dict], path: str) -> str:
    values = [_number(_value(run, path)) for run in runs]
    values = [value for value in values if value is not None]
    if not values:
        return "—"
    return (
        f'<span title="min–max: {_format(min(values))}–{_format(max(values))}; n={len(values)}">'
        f"{_format(statistics.median(values))}</span>"
    )


def _table(headers: tuple[str, ...], rows: list[list[str]]) -> str:
    return (
        '<div class="table"><table><thead><tr>'
        + "".join(f"<th>{_escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
        + "</tbody></table></div>"
    )


def _summary_tables(groups: list[tuple[dict, list[dict]]]) -> str:
    timing, learning = [], []
    for index, (config, runs) in enumerate(groups):
        backends = sorted({str(run.get("config", {}).get("backend", "unknown")) for run in runs})
        label = " · ".join(str(config.get(field, "?")) for field in ("task", "vision", "num_envs"))
        label = f'<a href="#comparison-{index}">{_escape(label)}</a>'
        for backend in backends:
            attempts = [run for run in runs if str(run.get("config", {}).get("backend", "unknown")) == backend]
            successful = [run for run in attempts if run.get("status") == "ok"]
            count = f"{len(successful)}/{len(attempts)}"
            if config.get("mode") != "learning":
                timing.append(
                    [
                        label,
                        _escape(config.get("mode")),
                        _escape(BACKEND_LABELS.get(backend, backend)),
                        count,
                        _statistics(successful, "metrics.startup_s"),
                        _statistics(successful, "metrics.physics_steps_per_s"),
                        _statistics(successful, "metrics.env_steps_per_s"),
                    ]
                )
                continue
            seeds = ", ".join(str(run.get("config", {}).get("seed", "?")) for run in successful) or "—"
            error_key, _, unit = TASK_ERRORS.get(config.get("task"), ("task_error", "Task error", ""))
            threshold_runs = [run for run in successful if isinstance(_value(run, "metrics.time_to_threshold"), dict)]
            reported = [run for run in successful if "time_to_threshold" in run.get("metrics", {})]
            reached = len(threshold_runs)
            threshold = _statistics(threshold_runs, "metrics.time_to_threshold.elapsed_s")
            if reported:
                threshold += f" <small>{reached}/{len(reported)} reached; {len(reported) - reached} censored</small>"
            learning.append(
                [
                    label,
                    _escape(BACKEND_LABELS.get(backend, backend)),
                    count,
                    _escape(seeds),
                    _statistics(successful, "metrics.evaluation.survival_fraction"),
                    _statistics(successful, f"metrics.evaluation.{error_key}") + f" {unit}",
                    _statistics(successful, "metrics.training_s"),
                    threshold,
                ]
            )
    parts = [
        '<section><h2>Results at a glance</h2><p class="muted">Medians; hover over a value for its '
        "observed minimum–maximum and sample count. Successful/attempted counts retain failed runs. "
        "Each linked workload has matching configuration, source inputs and GPU identity.</p>"
    ]
    if timing:
        parts.append(
            _table(
                (
                    "Task · vision · envs",
                    "Mode",
                    "Backend",
                    "Runs",
                    "Startup [s]",
                    "Physics [env steps/s]",
                    "Rollout [transitions/s]",
                ),
                timing,
            )
        )
    if learning:
        parts.append(
            '<h3>Held-out learning outcomes</h3><p class="muted">Survival means no task failure during the '
            "evaluation horizon: no balance termination for ANYmal/cartpole, and valid joint state for Franka. "
            "Task error is speed tracking for ANYmal, pole angle for cartpole, "
            "and end-effector distance for Franka. "
            "The first observed target uses the training clock, including evaluation and excluding startup. "
            "Target-time medians are conditional on reaching the target; censored seeds are excluded from that median. "
            "Checks occur every configured evaluation interval (50 iterations in the main matrix). "
            "An unreached target is censored at the recorded budget.</p>"
        )
        parts.append(
            _table(
                (
                    "Task · vision · envs",
                    "Backend",
                    "Runs",
                    "Seeds",
                    "Survival [fraction]",
                    "Task error",
                    "Training [s]",
                    "First target, reached seeds [training s]",
                ),
                learning,
            )
        )
    parts.append("</section>")
    return "".join(parts)


def group_results(runs: list[dict]) -> list[tuple[dict, list[dict]]]:
    """Group runs with matching workload configuration, source inputs and GPU identity."""
    groups = defaultdict(list)
    for run in runs:
        config = run.get("config", {})
        contract = {field: config.get(field) for field in MATCH_FIELDS}
        contract.update(
            {
                field: value
                for field, value in config.items()
                if field not in {"backend", "seed", "repeat", "adapter_module", "asset_path", "provenance"}
            }
        )
        provenance = run.get("provenance", {})
        assets = provenance.get("asset_manifest", {})
        if config.get("task") == "franka":
            assets = assets.get("franka", {})
        contract["_provenance"] = {
            "source_revisions": provenance.get("source_revisions"),
            "gpu": provenance.get("hardware", {}).get("gpu"),
            "cuda_device_uuid": provenance.get("runtime", {}).get("cuda_device_uuid"),
            "platform": provenance.get("hardware", {}).get("platform"),
            "source_urdf_sha256": assets.get("source_urdf_sha256"),
            "meshes_sha256": assets.get("meshes_sha256"),
            "harness_sha256": provenance.get("harness_sha256", provenance.get("harness_git_diff_sha256")),
        }
        key = json.dumps(contract, sort_keys=True)
        groups[key].append(run)
    return [(json.loads(key), values) for key, values in sorted(groups.items())]


def _bar_chart(series: list[tuple[str, list[float]]], unit: str, label: str) -> str:
    series = [(name, values) for name, values in series if values]
    if not series:
        return ""
    maximum = max(max(values) for _, values in series)
    if maximum <= 0:
        return ""
    height = 44 + 42 * len(series)
    parts = [
        f'<svg viewBox="0 0 760 {height}" role="img" aria-label="{_escape(label)}: median and min–max">',
        '<line x1="160" x2="160" y1="15" y2="' + str(height - 30) + '" stroke="#cbd3dc"/>',
    ]
    for index, (name, values) in enumerate(series):
        y = 22 + index * 42
        median = statistics.median(values)
        width = 410 * median / maximum
        lower, upper = (160 + 410 * value / maximum for value in (min(values), max(values)))
        parts.extend(
            [
                f'<text x="148" y="{y + 13}" text-anchor="end">{_escape(BACKEND_LABELS.get(name, name))}</text>',
                f'<rect x="160" y="{y}" width="{width:.2f}" height="20" fill="{COLORS[index % len(COLORS)]}"/>',
                f'<path d="M {lower:.2f} {y + 10} H {upper:.2f} M {lower:.2f} {y + 5} V {y + 15}'
                f' M {upper:.2f} {y + 5} V {y + 15}" stroke="#192333" stroke-width="2"/>',
                f'<text x="588" y="{y + 13}">{_format(median)} (n={len(values)})</text>',
            ]
        )
    parts.append(
        f'<text x="160" y="{height - 5}">0</text><text x="570" y="{height - 5}"'
        f' text-anchor="end">{_format(maximum)} {_escape(unit)}</text></svg>'
    )
    return "".join(parts)


def _learning_chart(runs: list[dict], x_key: str, y_key: str, x_label: str, y_label: str) -> str:
    traces = []
    for run in runs:
        if run.get("status") != "ok":
            continue
        points = []
        for sample in run.get("metrics", {}).get("learning", []):
            x = _number(sample.get(x_key))
            y = _number(_value(sample, y_key))
            if x is not None and y is not None:
                points.append((x, y))
        if points:
            config = run.get("config", {})
            traces.append((str(config.get("backend", "unknown")), config.get("seed", "?"), sorted(points)))
    if not traces:
        return ""
    names = sorted({name for name, _, _ in traces})
    colors = {name: COLORS[index % len(COLORS)] for index, name in enumerate(names)}
    max_x = max(point[0] for _, _, points in traces for point in points) or 1
    min_y = min(point[1] for _, _, points in traces for point in points)
    max_y = max(point[1] for _, _, points in traces for point in points)
    if y_key == "evaluation.survival_fraction":
        min_y, max_y = 0.0, 1.0
    elif y_key.startswith("evaluation."):
        min_y = min(0.0, min_y)
    span_y = (max_y - min_y) or 1
    parts = [
        f'<svg viewBox="0 0 760 340" role="img" aria-label="{_escape(y_label)} by {_escape(x_label)}">',
        '<path d="M 80 25 V 270 H 720" fill="none" stroke="#94a3b8"/>',
    ]
    for index in range(5):
        y_value = min_y + span_y * index / 4
        y = 270 - 240 * index / 4
        parts.append(f'<line x1="80" x2="720" y1="{y}" y2="{y}" stroke="#e4e9ee"/>')
        parts.append(f'<text x="70" y="{y + 5}" text-anchor="end">{_format(y_value)}</text>')
    for name, seed, points in traces:
        coordinates = " ".join(f"{80 + 640 * x / max_x:.2f},{270 - 240 * (y - min_y) / span_y:.2f}" for x, y in points)
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{colors[name]}"'
            f' stroke-width="2" opacity=".7"><title>{_escape(BACKEND_LABELS.get(name, name))} '
            f"seed {_escape(seed)}</title></polyline>"
        )
        if len(points) == 1:
            x, y = points[0]
            parts.append(
                f'<circle cx="{80 + 640 * x / max_x:.2f}" cy="{270 - 240 * (y - min_y) / span_y:.2f}"'
                f' r="3" fill="{colors[name]}"/>'
            )
    parts.extend(
        [
            f'<text x="80" y="290">0</text><text x="720" y="290" text-anchor="end">{_format(max_x)}</text>',
            f'<text x="400" y="318" text-anchor="middle">{_escape(x_label)}</text>',
            f'<text x="80" y="17">{_escape(y_label)}</text></svg>',
            '<p class="legend">'
            + " · ".join(
                f'<span style="color:{colors[name]}">{_escape(BACKEND_LABELS.get(name, name))}</span>' for name in names
            )
            + " · one trace per seed</p>",
        ]
    )
    return "".join(parts)


def _comparison(config: dict, runs: list[dict], index: int) -> str:
    backends = sorted({str(run.get("config", {}).get("backend", "unknown")) for run in runs})
    successful = [run for run in runs if run.get("status") == "ok"]
    title = f"{config.get('task', 'unknown')} · {config.get('mode', 'unknown')} · {config.get('vision', 'unknown')}"
    parts = [
        f'<section id="comparison-{index}"><h2>{_escape(title)}</h2>',
        '<p class="muted">'
        + " · ".join(
            f"{_escape(key)}={_escape(config.get(key))}"
            for key in ("num_envs", "dt", "decimation", "steps", "solver_profile", "solver_iterations")
            if config.get(key) is not None
        )
        + "</p>",
        "<details><summary>Matched workload configuration</summary><pre>"
        + _escape(json.dumps(config, indent=2, sort_keys=True))
        + "</pre></details>",
    ]
    if not successful:
        parts.append('<p class="pending">No successful measurements for this configuration.</p>')
    else:
        if config.get("mode") == "learning":
            parts.append("<details><summary>Timing distributions</summary>")
        rows = []
        for metric, (label, unit, higher) in METRICS.items():
            series = []
            for backend in backends:
                values = [
                    _number(run.get(metric) if metric == "process_wall_s" else run.get("metrics", {}).get(metric))
                    for run in successful
                    if str(run.get("config", {}).get("backend", "unknown")) == backend
                ]
                series.append((backend, [value for value in values if value is not None]))
            if not any(values for _, values in series):
                continue
            direction = "higher is faster" if higher else "lower is faster"
            parts.append(f"<figure><h3>{_escape(label)} <small>{_escape(unit)} · {direction}</small></h3>")
            parts.append(_bar_chart(series, unit, label))
            parts.append(
                "<figcaption>Median; whiskers show minimum–maximum across successful runs. "
                "Failed runs are listed below.</figcaption></figure>"
            )
            for backend, values in series:
                if values:
                    rows.append(
                        f"<tr><td>{_escape(label)}</td><td>{_escape(BACKEND_LABELS.get(backend, backend))}</td>"
                        f"<td>{_format(statistics.median(values))}</td>"
                        f"<td>{_format(min(values))}–{_format(max(values))}</td><td>{len(values)}</td></tr>"
                    )
        if rows:
            parts.append(
                '<details><summary>Exact summaries</summary><div class="table"><table><thead><tr>'
                "<th>Metric</th><th>Backend</th>"
                "<th>Median</th><th>Min–max</th><th>n</th></tr></thead><tbody>"
                + "".join(rows)
                + "</tbody></table></div></details>"
            )
        if config.get("mode") == "learning":
            parts.append("</details>")
            error_key, error_label, _ = TASK_ERRORS.get(config.get("task"), ("task_error", "Task error", ""))
            for y_key, y_label in (("survival_fraction", "Held-out survival [fraction]"), (error_key, error_label)):
                for x_key, x_label in (
                    ("elapsed_s", "Training wall time [s]"),
                    ("env_steps", "Environment transitions"),
                ):
                    curve = _learning_chart(runs, x_key, f"evaluation.{y_key}", x_label, y_label)
                    if curve:
                        parts.append(
                            f"<figure><h3>{_escape(y_label)} against {_escape(x_label.lower())}</h3>{curve}"
                            "<figcaption>Held-out reset states and deterministic policy actions; one trace per seed. "
                            "Error averages include steps after automatic resets.</figcaption></figure>"
                        )
            evaluations = []
            for run in successful:
                if not run.get("metrics", {}).get("evaluation"):
                    continue
                threshold = _value(run, "metrics.time_to_threshold")
                threshold_time = _statistics([run], "metrics.time_to_threshold.elapsed_s")
                if threshold is None and "time_to_threshold" in run.get("metrics", {}):
                    threshold_time = "censored"
                evaluations.append(
                    [
                        _escape(BACKEND_LABELS.get(run["config"]["backend"], run["config"]["backend"])),
                        _escape(run.get("config", {}).get("seed")),
                        _statistics([run], "metrics.evaluation.survival_fraction"),
                        _statistics([run], f"metrics.evaluation.{error_key}"),
                        _escape(_value(run, "metrics.evaluation.threshold_reached")),
                        threshold_time,
                    ]
                )
            if evaluations:
                parts.append(
                    "<details><summary>Final held-out evaluations per seed</summary>"
                    + _table(
                        (
                            "Backend",
                            "Seed",
                            "Survival [fraction]",
                            error_label,
                            "Target reached",
                            "First target [training s]",
                        ),
                        evaluations,
                    )
                    + "</details>"
                )
    parts.append(
        '<div class="table"><table><thead><tr><th>Backend</th><th>Seed</th><th>Status</th>'
        "<th>Raw measurement / reason</th></tr></thead><tbody>"
    )
    for run in runs:
        cfg = run.get("config", {})
        backend = str(cfg.get("backend", "unknown"))
        label = BACKEND_LABELS.get(backend, backend)
        status = run.get("status", "unknown")
        parts.append(
            f"<tr><td>{_escape(label)}</td><td>{_escape(cfg.get('seed', '?'))}</td>"
            f'<td>{_escape(status)}</td><td><a href="{_escape(run["download"])}">JSON</a>'
            + (f" · {_escape(run.get('error', ''))}" if status != "ok" else "")
            + "</td></tr>"
        )
        if run.get("camera_download"):
            parts.append(
                f'<tr><td colspan="4"><a href="{_escape(run["camera_download"])}">'
                f'<img src="{_escape(run["camera_download"])}" width="{_escape(cfg.get("width", 64))}"'
                ' loading="lazy" style="image-rendering:pixelated"'
                f' alt="{_escape(label)} camera validation frame">'
                "</a> First usable RGB frame; native renderer quality differs.</td></tr>"
            )
    parts.append("</tbody></table></div></section>")
    return "".join(parts)


def build_report(input_dir: Path, output_dir: Path) -> Path:
    """Write an HTML report and copies of raw measurements to an output directory."""
    sources = sorted(input_dir.rglob("*.result.json"))
    if not sources:
        raise ValueError(f"No *.result.json measurements found in {input_dir}")
    raw_dir = output_dir / "data" / "runs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for source in sources:
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        relative = f"data/runs/{digest[:16]}-{source.name.removesuffix('.result.json')}.json"
        shutil.copyfile(source, output_dir / relative)
        try:
            run = json.loads(raw)
            if not isinstance(run, dict) or not isinstance(run.get("config"), dict):
                raise ValueError("Measurement requires a JSON object with a config object")
            if run.get("schema_version") != 1:
                raise ValueError("Unsupported measurement schema_version")
        except (ValueError, UnicodeDecodeError) as error:
            run = {"status": "failed", "config": {"backend": "invalid record"}, "error": str(error)}
        run["download"] = relative
        run["sha256"] = digest
        camera_sample = run.get("camera_sample")
        if isinstance(camera_sample, str):
            camera = source.parent / camera_sample
            if (
                camera.suffix.lower() == ".png"
                and camera.resolve().is_relative_to(source.parent.resolve())
                and camera.is_file()
            ):
                camera_relative = f"data/runs/{digest[:16]}-camera.png"
                shutil.copyfile(camera, output_dir / camera_relative)
                run["camera_download"] = camera_relative
        runs.append(run)
    groups = group_results(runs)
    counts = Counter(run.get("status", "unknown") for run in runs)
    success_count = counts.get("ok", 0)
    status = f"{success_count} successful runs · {len(runs) - success_count} failed or unsupported runs"
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status_counts": dict(counts),
        "comparison_fields": MATCH_FIELDS,
        "runs": runs,
    }
    (output_dir / "data" / "measurements.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    provenance = {}
    for run in runs:
        value = run.get("provenance", {})
        if value:
            environment = {
                key: value.get(key)
                for key in (
                    "source_revisions",
                    "package_versions",
                    "runtime",
                    "cache_policy",
                    "harness_sha256",
                )
            }
            environment["hardware"] = {key: value.get("hardware", {}).get(key) for key in ("platform", "cpu", "gpu")}
            provenance[json.dumps(environment, sort_keys=True)] = environment
    methods = """
<section id="methods"><details><summary>Measurement contract</summary>
<p>Physics times synchronized stepping with prescribed controls; rollout includes actions, observations,
rewards and resets. One environment transition contains the configured number of physics steps.
Worker startup covers runtime imports through the first usable observation; Python/site initialization is outside
this clock. The first step and warmed throughput are separate. Complete process wall time is also recorded.
A fresh process can reuse disk caches.</p>
<p>The shared benchmark baseline uses Genesis's Newton solver with a 25-iteration maximum and native tolerance.
Isaac Lab runs without Kit and uses Newton/MuJoCo Warp with Newton solving, implicitfast integration,
100 solver / 50 line-search iterations, tolerance 1e-6, pyramidal friction, MuJoCo contacts,
one substep and CUDA graphs.
These settings do not establish equal solver work or numerical accuracy. Stock ANYmal training presets use
different substep/contact/friction settings; this shared task is a separate benchmark baseline.</p>
<p>ANYmal physics runs can fall or settle under prescribed controls;
throughput alone does not establish walking quality.
The backend metadata records controls and assets. Genesis uses Madrona rasterization and Isaac Lab uses
Newton Warp ray tracing. RGB augments proprioception in the common CNN policy; these tasks do not establish
vision-only navigation. Renderer quality differs, and image throughput does not establish quality equivalence.
MJWarp graph activity and fatal capacity flags are checked at the final solver state.
Convergence-limit flags are retained separately and do not reject a run. Final flags are not a cumulative history.</p>
<p>Comparisons share task, vision mode, mode, batch size, timestep, decimation, image dimensions and measured
step count. Maximum capacity is separate from speed at a matched batch size. Successful-run summaries retain
their sample count and observed range; they are not confidence intervals. Hardware and shared training dependencies
must match. Engine-specific simulation and rendering packages differ and are pinned below.</p>
<p>Learning curves use each seed’s held-out survival and task error against wall time and transitions.
The target requires survival ≥0.95 and error ≤0.25 m/s for ANYmal, ≤0.1 rad for cartpole, or ≤0.05 m for Franka.
Evaluation uses automatic
resets after failure: survival means no failure at any point in the evaluation horizon; for Franka this means
finite joint state within joint limits plus a 0.1 rad margin, rather than no fall. Its evaluation lasts 5 s.
Reported reward and error
also include steps after resets. Training time includes periodic evaluation but excludes model/optimizer creation,
simulator startup and checkpoint saving. Complete process time includes those costs and teardown.
Asset equivalence and useful final behavior are necessary before claiming a learning advantage.</p>
</details></section>
<section><h2>Why repeat this comparison?</h2>
<p><a href="https://arxiv.org/html/2607.13059v1">GPUSimBench</a> reported approximately 6× Franka throughput
for Genesis using different maximum batch sizes (131,072 versus 32,768), with rendering and learning excluded.
Its public <a href="https://github.com/zhanghuzhenyu/gpu_sim_bench/tree/7d39254e6642bef924166ac74e7c6940c713a471">implementation</a>
also compares Genesis scene stepping with an Isaac Lab task step. These results motivate matched workloads;
they do not establish a general training speedup. That study used an older Isaac Lab/PhysX stack.
This MJWarp comparison changes the backend and versions, so it is not a direct RTX 5070 reproduction.</p></section>
"""
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light"><title>Genesis and Isaac Lab MJWarp · Simulator comparison</title>
<style>
:root{--paper:#fafaf8;--ink:#192333;--muted:#536174;--line:#d9dfe5;--blue:#245cab}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 system-ui,sans-serif}
main{max-width:1100px;margin:auto;padding:32px clamp(18px,4vw,50px) 70px}a{color:var(--blue)}
h1,h2{font-family:Georgia,serif;line-height:1.15;font-weight:400}h1{font-size:clamp(2.3rem,6vw,3.8rem);margin:18px 0}
h2{font-size:1.9rem}h3{font-size:1.05rem;margin:0 0 12px}
small,.muted,figcaption,.legend{color:var(--muted);font-size:.88rem}
section{border-top:1px solid var(--line);margin-top:34px;padding-top:15px}
figure{margin:24px 0;padding:22px;background:white;border:1px solid var(--line);border-radius:6px}
svg{display:block;width:100%;height:auto;min-width:540px}svg text{font:13px system-ui,sans-serif;fill:var(--ink)}
figure,.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}
th{white-space:nowrap}td{overflow-wrap:anywhere}section>p+.table td,section>h3+p+.table td{white-space:nowrap}
pre{font-size:.78rem;white-space:pre-wrap;overflow-wrap:anywhere;
background:white;padding:16px;border:1px solid var(--line)}
summary{cursor:pointer;font-weight:600}details{margin:16px 0}
.pending{padding:16px;background:#fff5df;border:1px solid #e4c98d}
nav{display:flex;flex-wrap:wrap;gap:10px 20px;margin:22px 0}footer{margin-top:40px;color:var(--muted);font-size:.85rem}
</style></head><body><main><a href="../../">Research reports · Maximilian Krause</a>
<p class="muted">SIMULATION · VISION · LEARNING</p><h1>Genesis and Isaac Lab MJWarp</h1>
<p>Genesis versus Isaac Lab with Newton/MuJoCo Warp: ANYmal-C walking, cartpole balance and Franka reaching,
with proprioception or proprioception plus RGB.</p>
"""
    document += f'<p class="muted">Generated {_escape(summary["generated_at"])} · measurement schema 1</p>'
    busy = any(
        run.get("provenance", {}).get(
            "other_active_compute_detected",
            bool(str(run.get("provenance", {}).get("hardware", {}).get("gpu_processes_before", "")).strip()),
        )
        for run in runs
    )
    if busy:
        document += (
            '<p class="pending"><strong>Shared GPU · provisional measurements.</strong> '
            "Another GPU compute process was recorded before the suite. "
            "These runs do not establish a simulator winner. "
            "Repeat on an idle GPU before using timing differences for a performance claim.</p>"
        )
    document += f'<p class="{"muted" if success_count else "pending"}">{_escape(status)}. '
    document += "Measurements below are local to the recorded hardware and source revisions.</p>"
    document += '<nav><a href="#methods">Methods</a><a href="#provenance">Versions and hardware</a>'
    document += '<a href="data/measurements.json">All measurements (JSON)</a></nav>'
    if not success_count:
        document += (
            '<p class="pending">The report contains no successful timing or learning measurements. '
            "No simulator ranking can be inferred.</p>"
        )
    document += _summary_tables(groups)
    for index, (config, values) in enumerate(groups):
        label = " · ".join(str(config.get(key, "?")) for key in ("task", "mode", "vision", "num_envs"))
        document += f"<details><summary>Plots and evidence: {_escape(label)} environments</summary>"
        document += _comparison(config, values, index) + "</details>"
    document += methods
    document += (
        '<section id="provenance"><h2>Versions and hardware</h2><p>Exact provenance per distinct environment; '
        "each raw JSON also includes backend metadata and configuration.</p>"
    )
    revisions = sorted(
        {
            (str(name), str(revision))
            for value in provenance.values()
            for name, revision in value.get("source_revisions", {}).items()
        }
    )
    if revisions:
        document += '<div class="table"><table><thead><tr><th>Source</th><th>Exact revision</th></tr></thead><tbody>'
        document += "".join(
            f"<tr><td>{_escape(name)}</td><td><code>{_escape(revision)}</code></td></tr>"
            for name, revision in revisions
        )
        document += "</tbody></table></div>"
    for index, value in enumerate(provenance.values()):
        document += (
            f"<details><summary>Environment {index + 1}</summary>"
            f"<pre>{_escape(json.dumps(value, indent=2, sort_keys=True))}</pre></details>"
        )
    document += (
        "</section><footer>Generated from recorded JSON measurements. "
        "Raw-file SHA-256 hashes are included in the aggregate download.</footer></main></body></html>\n"
    )
    path = output_dir / "index.html"
    path.write_text(document, encoding="utf-8")
    return path


def main() -> None:
    """Parse report paths and generate the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    print(build_report(args.input_dir, args.output_dir))


if __name__ == "__main__":
    main()
