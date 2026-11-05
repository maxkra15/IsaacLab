# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export standalone scientific figures and derived data from recorded benchmark results."""

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from report import BACKEND_LABELS, group_results

COLORS = {"genesis": "#245cab", "isaaclab": "#b04b17", "mjlab": "#347a4a", "isaaclab_physx": "#8050a4"}
STYLES = ("-", "--", "-.", ":")


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _identity(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]


def _save(fig, output: Path, name: str) -> None:
    for suffix in ("svg", "png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _runtime(records: list[dict], output: Path) -> None:
    workloads = defaultdict(list)
    for record in records:
        config = record["config"]
        if config["mode"] in ("physics", "rollout"):
            workloads[(config["mode"], config["vision"], config.get("width"), config.get("height"))].append(record)
    for (mode, vision, width, height), matched in workloads.items():
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), squeeze=False, layout="constrained")
        throughput = "physics_steps_per_s" if mode == "physics" else "env_steps_per_s"
        unit = "Environment physics steps/s" if mode == "physics" else "Environment transitions/s"
        for row, task in enumerate(("anymal", "cartpole")):
            series = defaultdict(list)
            for record in matched:
                config = record["config"]
                if config["task"] == task:
                    condition = {key: value for key, value in config.items() if key != "num_envs"}
                    series[(record["backend"], _identity(condition))].append(record)
            counts = Counter(backend for backend, _ in series)
            for index, ((backend, condition), points) in enumerate(sorted(series.items())):
                label = BACKEND_LABELS.get(backend, backend) + (
                    f" · condition {condition}" if counts[backend] > 1 else ""
                )
                points.sort(key=lambda point: point["config"]["num_envs"])
                for column, metric in enumerate((throughput, "startup_s")):
                    values = [point for point in points if metric in point["summary"]]
                    x = [point["config"]["num_envs"] for point in values]
                    medians = [point["summary"][metric]["median"] for point in values]
                    errors = [
                        [point["summary"][metric]["median"] - point["summary"][metric]["min"] for point in values],
                        [point["summary"][metric]["max"] - point["summary"][metric]["median"] for point in values],
                    ]
                    axes[row, column].errorbar(
                        x,
                        medians,
                        yerr=errors,
                        color=COLORS.get(backend, "#8050a4"),
                        linestyle=STYLES[index % len(STYLES)],
                        marker="o",
                        capsize=3,
                        label=label,
                    )
            for column, title in enumerate((unit, "Startup to usable scene [s]")):
                ax = axes[row, column]
                ax.set(title=f"{task.upper()} · {title}", xlabel="Parallel environments", ylabel=title)
                if series:
                    ax.set_xscale("log", base=2)
                    ax.set_xticks(
                        sorted({point["config"]["num_envs"] for points in series.values() for point in points})
                    )
                    ax.xaxis.set_major_formatter(ScalarFormatter())
                    ax.legend(fontsize=7)
                else:
                    ax.text(0.5, 0.5, "No successful measurements", ha="center", transform=ax.transAxes)
        image_label = f"RGB {width}×{height}" if vision == "rgb" else "no vision"
        baseline = (
            "Shared benchmark baseline; "
            if all(record["config"].get("solver_profile") == "shared" for record in matched)
            else ""
        )
        fig.suptitle(f"{mode.capitalize()} · {image_label}\n{baseline}medians and observed min–max")
        _save(fig, output, f"runtime-{mode}-{vision}-{width}x{height}")


def _learning(records: list[dict], output: Path) -> None:
    grouped = defaultdict(list)
    for record in records:
        if record["config"]["mode"] == "learning":
            grouped[_identity(record["config"])].append(record)
    for identity, matched in grouped.items():
        config = matched[0]["config"]
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), squeeze=False, layout="constrained")
        error = "tracking_error_m_s" if config["task"] == "anymal" else "pole_angle_abs_rad"
        error_label = "Speed tracking error [m/s]" if config["task"] == "anymal" else "Pole angle error [rad]"
        for record in matched:
            for index, trace in enumerate(record["learning"]):
                label = f"{BACKEND_LABELS.get(record['backend'], record['backend'])} · seed {trace['seed']}"
                for row, (metric, ylabel) in enumerate(
                    (("survival_fraction", "Held-out survival [fraction]"), (error, error_label))
                ):
                    for column, (axis, xlabel) in enumerate(
                        (("elapsed_s", "Training wall time [s]"), ("env_steps", "Environment transitions"))
                    ):
                        points = [
                            (sample.get(axis), sample.get("evaluation", {}).get(metric)) for sample in trace["samples"]
                        ]
                        points = [(x, y) for x, y in points if _number(x) is not None and _number(y) is not None]
                        if points:
                            x, y = zip(*points)
                            axes[row, column].plot(
                                x,
                                y,
                                color=COLORS.get(record["backend"], "#8050a4"),
                                linestyle=STYLES[index % len(STYLES)],
                                alpha=0.8,
                                label=label,
                            )
                        axes[row, column].set(xlabel=xlabel, ylabel=ylabel)
        for row in range(2):
            for ax in axes[row]:
                ax.set_ylim(bottom=0, top=1.02 if row == 0 else None)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, fontsize=7)
        fig.suptitle(
            f"{config['task'].upper()} · {config['vision']} · {config['num_envs']} environments · "
            f"{config.get('iterations', '?')} training iterations\n"
            f"{config.get('solver_profile', 'Unrecorded profile')} benchmark baseline; "
            "held-out evaluation; one curve per seed"
        )
        _save(fig, output, f"learning-{config['task']}-{config['vision']}-{config['num_envs']}-{identity}")


def export_figures(input_dir: Path, output_dir: Path) -> Path:
    """Export matched successful measurements without combining distinct training budgets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(input_dir.rglob("*.result.json"))
    if not paths:
        raise ValueError(f"No *.result.json measurements found in {input_dir}")
    runs = [json.loads(path.read_text()) for path in paths]
    records = []
    for config, grouped in group_results(runs):
        backends = sorted({run["config"]["backend"] for run in grouped})
        for backend in backends:
            successful = [run for run in grouped if run["config"]["backend"] == backend and run.get("status") == "ok"]
            if not successful:
                continue
            summary = {}
            for metric in ("startup_s", "physics_steps_per_s", "env_steps_per_s"):
                values = [_number(run.get("metrics", {}).get(metric)) for run in successful]
                values = [value for value in values if value is not None]
                if values:
                    summary[metric] = {
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                        "n": len(values),
                    }
            records.append(
                {
                    "config": config,
                    "backend": backend,
                    "summary": summary,
                    "learning": [
                        {"seed": run["config"]["seed"], "samples": run.get("metrics", {}).get("learning", [])}
                        for run in successful
                    ]
                    if config["mode"] == "learning"
                    else [],
                }
            )
    plt.rcParams.update(
        {"font.size": 9, "axes.grid": True, "grid.alpha": 0.2, "svg.fonttype": "none", "pdf.fonttype": 42}
    )
    _runtime(records, output_dir)
    _learning(records, output_dir)
    path = output_dir / "figuredata.json"
    data = {
        "schema_version": 1,
        "status_counts": dict(Counter(run.get("status", "unknown") for run in runs)),
        "records": records,
    }
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    print(export_figures(args.input_dir, args.output_dir))
