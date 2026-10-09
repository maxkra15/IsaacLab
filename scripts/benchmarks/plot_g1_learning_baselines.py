# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export standalone scientific figures from a verified G1 primary summary.

Matplotlib is a report-tool dependency only. No simulator or learner is loaded::

    uv run --no-sync python scripts/benchmarks/plot_g1_learning_baselines.py \
        --summary logs/g1-learning-baselines-20261009/operations/primary-summary.json \
        --output logs/g1-learning-baselines-20261009

The output contains ``figures/figures.json`` and searchable SVG, PDF and PNG
versions of each figure. Ranges describe three seeds, not confidence intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

ALGORITHMS = ("rsl_rl_ppo", "warp_ppo", "flashsac")
LABELS = {"rsl_rl_ppo": "RSL PPO", "warp_ppo": "Warp PPO", "flashsac": "FlashSAC"}
COLORS = {"rsl_rl_ppo": "#2563eb", "warp_ppo": "#7c3aed", "flashsac": "#d97706"}
SEED_COLORS = {"rsl_rl_ppo": "#1e40af", "warp_ppo": "#5b21b6", "flashsac": "#92400e"}
CHECKPOINTS = (250, 500, 1000, 1500, 2050)
PPO_SOURCE = "de5c84d81091e6d48e05c64b47366a4d95d1a75f"
FLASH_SOURCE = "734ad065e13c33d564ff86e1a22a138ce678c781"
ROBO_SOURCE = "dc772f9d65c21441c9bdb70ed905557074e00f7e"


def _load_summary(path: Path) -> tuple[dict, str]:
    """Validate the complete cohort, provenance and numerical inputs."""
    content = path.read_bytes()
    summary = json.loads(content)
    if (
        summary["protocol"] != "g1_initialization_matched_primary_summary_v1"
        or summary["primary_run_count"] != 9
        or len(summary["runs"]) != 9
    ):
        raise ValueError("Expected the verified nine-run initialization-matched primary summary.")

    def check_finite(value):
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError("The summary contains nonfinite numbers.")
        if isinstance(value, dict):
            for item in value.values():
                check_finite(item)
        elif isinstance(value, list):
            for item in value:
                check_finite(item)

    check_finite(summary)
    expected_pairs = {
        (iteration, scenario) for iteration in CHECKPOINTS for scenario in ("native_commands", "forward_0_5")
    }
    for algorithm in ALGORITHMS:
        runs = [run for run in summary["runs"] if run["algorithm"] == algorithm]
        if len(runs) != 3 or {run["seed"] for run in runs} != {0, 1, 2}:
            raise ValueError(f"Expected three distinct seeds for {algorithm}.")
        for run in runs:
            pins = run["source_pins"]
            source = FLASH_SOURCE if algorithm == "flashsac" else PPO_SOURCE
            if pins["isaaclab"] != source or pins["robolearn"] != ROBO_SOURCE:
                raise ValueError("The summary does not identify the reviewed source cohorts.")
            for key in ("mdp_sha256", "mdp_source_sha256"):
                if pins[key] != summary[key] or re.fullmatch(r"[0-9a-f]{64}", pins[key]) is None:
                    raise ValueError("The selected MDP hashes differ or are invalid.")
            if run["initial_episode_counter_randomization"]["value"] is not True:
                raise ValueError("Initial episode counters are not matched.")
            if not run["updates"]["weights_updated"] or not run["updates"]["weights_and_optimizer_finite"]:
                raise ValueError("The summary does not confirm finite updated policy weights.")
            rows = run["common_evaluations"]
            if len(rows) != 10 or {(row["iteration"], row["scenario"]) for row in rows} != expected_pairs:
                raise ValueError("Common evaluation checkpoints are incomplete.")
            for row in rows:
                if row["evaluation_envs"] != 64 or row["transitions"] != row["iteration"] * 1024 * 24:
                    raise ValueError("An evaluation budget differs from the common protocol.")
    return summary, hashlib.sha256(content).hexdigest()


def _style_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#e2e8f0", linewidth=0.6)
    axis.set_axisbelow(True)


def _range(values: list[list[float]]) -> tuple[list, list, list]:
    columns = list(zip(*values, strict=True))
    return (
        [statistics.mean(column) for column in columns],
        [min(column) for column in columns],
        [max(column) for column in columns],
    )


def _learning_curves(summary: dict, scenario: str, plot_module):
    """Show measured seed trajectories and descriptive mean/range bands."""
    figure, axes = plot_module.subplots(1, 2, figsize=(8.0, 3.7), layout="constrained")
    success = "walking_success_rate" if scenario == "forward_0_5" else "tracking_success_rate"
    for algorithm in ALGORITHMS:
        runs = sorted((run for run in summary["runs"] if run["algorithm"] == algorithm), key=lambda run: run["seed"])
        rows = [
            sorted(
                (row for row in run["common_evaluations"] if row["scenario"] == scenario),
                key=lambda row: row["iteration"],
            )
            for run in runs
        ]
        transitions = [row["transitions"] / 1e6 for row in rows[0]]
        for axis, field, scale in ((axes[0], success, 100), (axes[1], "linear_velocity_error_mean", 1)):
            values = [[row[field] * scale for row in seed] for seed in rows]
            mean, lower, upper = _range(values)
            for seed in values:
                axis.plot(
                    transitions, seed, color=SEED_COLORS[algorithm], alpha=0.5, linewidth=0.7, marker=".", markersize=3
                )
            axis.fill_between(transitions, lower, upper, color=COLORS[algorithm], alpha=0.13, linewidth=0)
            axis.plot(
                transitions,
                mean,
                color=COLORS[algorithm],
                linewidth=1.9,
                marker="o",
                markersize=4,
                label=LABELS[algorithm],
            )
    for axis in axes:
        _style_axis(axis)
        axis.set(xlabel="Transitions collected (millions)", xlim=(0, 52))
    axes[0].set(
        ylabel="Success (%)",
        ylim=(0, 105),
        title="a  Walking success" if scenario == "forward_0_5" else "a  Native tracking success",
    )
    axes[1].set(ylabel="Mean planar velocity error (m/s)", ylim=(0, None), title="b  Planar velocity tracking")
    if scenario == "forward_0_5":
        axes[0].axhline(80, color="#64748b", linestyle="--", linewidth=0.7)
        axes[1].axhline(0.2, color="#64748b", linestyle="--", linewidth=0.7)
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    figure.suptitle(
        "Forward command: 0.5 m/s" if scenario == "forward_0_5" else "Native command distribution", fontsize=11
    )
    figure.supxlabel("Thin lines: seeds. Bands: 3-seed min–max, not confidence intervals.", fontsize=8)
    return figure


def _runtime_costs(summary: dict, plot_module):
    """Separate measured loop components from the complete command pipeline."""
    from matplotlib.patches import Patch

    figure, axes = plot_module.subplots(1, 2, figsize=(8.0, 3.8), layout="constrained")
    for index, algorithm in enumerate(ALGORITHMS):
        runs = [run for run in summary["runs"] if run["algorithm"] == algorithm]
        for axis, fields, colors, total in (
            (axes[0], ("rollout_seconds", "update_seconds"), ("#d6dde8", COLORS[algorithm]), "training_seconds"),
            (
                axes[1],
                ("smoke_command_seconds", "train_command_seconds", "evaluation_command_seconds"),
                ("#ead8b7", "#93a4b8", "#e3e8ef"),
                "all_three_commands_seconds",
            ),
        ):
            left = 0.0
            for field, color in zip(fields, colors, strict=True):
                width = statistics.mean(run["timings"][field] for run in runs) / 60
                axis.barh(index, width, left=left, height=0.54, color=color)
                left += width
            totals = [run["timings"][total] / 60 for run in runs]
            mean = statistics.mean(totals)
            axis.errorbar(
                mean,
                index,
                xerr=[[mean - min(totals)], [max(totals) - mean]],
                fmt="none",
                ecolor="#334155",
                capsize=3,
                linewidth=0.8,
            )
            axis.annotate(
                f"{mean:.2f}", (max(totals), index), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8
            )
    for axis in axes:
        _style_axis(axis)
        axis.grid(False, axis="y")
        axis.set(
            yticks=range(3),
            yticklabels=[LABELS[algorithm] for algorithm in ALGORITHMS],
            xlabel="Minutes",
        )
        axis.invert_yaxis()
        axis.margins(x=0.15)
        axis.set_xlim(left=0)
        for label, algorithm in zip(axis.get_yticklabels(), ALGORITHMS, strict=True):
            label.set_color(COLORS[algorithm])
    axes[0].set_title("a  Synchronized training loop", loc="left")
    axes[1].set_title("b  Measured command pipeline", loc="left")
    axes[0].legend(
        handles=[
            Patch(color="#d6dde8", label="Rollout / collection"),
            Patch(color="#64748b", label="Learning updates"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    axes[1].legend(
        handles=[
            Patch(color=color, label=label)
            for color, label in (("#ead8b7", "Smoke"), ("#93a4b8", "Train"), ("#e3e8ef", "Offline eval."))
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    return figure


def _final_quality(summary: dict, plot_module):
    """Compare final checkpoints using individual seeds and mean/range markers."""
    from matplotlib.lines import Line2D

    figure, axes = plot_module.subplots(1, 2, figsize=(8.0, 3.7), layout="constrained")
    for index, algorithm in enumerate(ALGORITHMS):
        runs = sorted((run for run in summary["runs"] if run["algorithm"] == algorithm), key=lambda run: run["seed"])
        for scenario, offset, marker in (("native_commands", -0.14, "o"), ("forward_0_5", 0.14, "s")):
            for axis, field, scale in ((axes[0], "linear_velocity_error_mean", 1), (axes[1], "survival_rate", 100)):
                values = [run["final_quality"][scenario][field] * scale for run in runs]
                mean = statistics.mean(values)
                position = index + offset
                axis.scatter(
                    [position - 0.025, position, position + 0.025], values, s=12, color=COLORS[algorithm], alpha=0.35
                )
                axis.errorbar(
                    position,
                    mean,
                    yerr=[[mean - min(values)], [max(values) - mean]],
                    fmt=marker,
                    color=COLORS[algorithm],
                    markerfacecolor=COLORS[algorithm] if marker == "o" else "white",
                    markersize=6,
                    capsize=3,
                    linewidth=1.1,
                )
    for axis in axes:
        _style_axis(axis)
        axis.set(xticks=range(3), xticklabels=[LABELS[algorithm] for algorithm in ALGORITHMS], xlim=(-0.5, 2.5))
    axes[0].set(ylabel="Mean planar velocity error (m/s)", ylim=(0, None), title="a  Final velocity tracking")
    axes[1].set(ylabel="Survival (%)", ylim=(0, 105), title="b  Full-episode survival")
    figure.suptitle("Final checkpoints: 50.38 million transitions", fontsize=11)
    axes[0].legend(
        handles=[
            Line2D(
                [],
                [],
                marker=marker,
                color="#475569",
                linestyle="none",
                markerfacecolor="#475569" if marker == "o" else "white",
                label=label,
            )
            for marker, label in (("o", "Native commands"), ("s", "Forward 0.5 m/s"))
        ],
        frameon=False,
        fontsize=8,
    )
    figure.supxlabel("Large points: means. Whiskers: 3-seed min–max, not confidence intervals.", fontsize=8)
    return figure


def main() -> None:
    """Read the verified summary and export four publication-style figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary, digest = _load_summary(args.summary)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "axes.edgecolor": "#64748b",
            "axes.linewidth": 0.6,
        }
    )
    output = args.output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    specs = (
        (
            "sample-efficiency-forward",
            _learning_curves(summary, "forward_0_5", plt),
            "Forward 0.5 m/s evaluations. Dark thin lines show seeds; colored lines are means and bands are "
            "three-seed min–max ranges, not confidence intervals. Lines connect five measured checkpoints; "
            "intermediate values were not evaluated. Episode walking success requires survival, planar error "
            "below 0.25 m/s, yaw error below 0.4 rad/s and forward speed above 0.25 m/s. Dashed 80% success "
            "and 0.2 m/s mean-error thresholds are two aggregate gate conditions; survival of at least 90% "
            "is also required.",
        ),
        (
            "sample-efficiency-native",
            _learning_curves(summary, "native_commands", plt),
            "Native commands retain commanded standing and resampling. Tracking success requires survival, "
            "mean planar error below 0.5 m/s and mean yaw error below 0.8 rad/s; standing success is included. "
            "Thin lines show seeds, colored lines means, and bands three-seed min–max ranges, not confidence "
            "intervals. Lines connect measured checkpoints and do not establish exact crossing times.",
        ),
        (
            "runtime-costs",
            _runtime_costs(summary, plt),
            "Means and total-time min–max whiskers over three seeds. Left: synchronized training-loop phases. "
            "Flash interleaves learning and attributes updates using CUDA events; host launch overhead remains "
            "in the collection remainder. Algorithm-specific learning work differs. Right: measured smoke, "
            "training and offline-evaluation command costs. These are not fully cold timings; OSMO queue, "
            "checkout/bootstrap and artifact retrieval are excluded.",
        ),
        (
            "final-quality",
            _final_quality(summary, plt),
            "Final 50.38M-transition checkpoints, with 64 evaluation worlds per scenario and seed. Large "
            "points are means, faint points individual seeds, and whiskers three-seed min–max ranges, not "
            "confidence intervals. Filled circles show native commands including standing; open squares show "
            "forward 0.5 m/s. Survival measures completion of the native 20-second episode independently of "
            "velocity tracking.",
        ),
    )
    for identifier, figure, caption in specs:
        record = {"id": identifier, "caption": caption}
        for extension in ("svg", "pdf", "png"):
            filename = f"{identifier}.{extension}"
            figure.savefig(output / filename, bbox_inches="tight", facecolor="white")
            record[extension] = filename
        plt.close(figure)
        records.append(record)
    if hashlib.sha256(args.summary.read_bytes()).hexdigest() != digest:
        raise ValueError("The input summary changed during figure generation.")
    (output / "figures.json").write_text(json.dumps({"summary_sha256": digest, "figures": records}, indent=2) + "\n")
    print(f"Exported {len(records)} figures as SVG, PDF and PNG to {output}")


if __name__ == "__main__":
    main()
