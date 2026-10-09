# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the reviewed 9 October G1 experiment as a compact paper.

Generate the figures with ``plot_g1_learning_baselines.py`` first. The report
embeds SVG and PDF exports, includes per-seed results and source hashes, and
links to local raw evidence without copying large contact traces into HTML.
The curated narrative is bound to the reviewed summary hash. New experiments
must update both the narrative and this pin, or use the general dashboard.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import json
import os
import statistics
from pathlib import Path

ALGORITHMS = {"rsl_rl_ppo": "RSL PPO", "warp_ppo": "Warp PPO", "flashsac": "FlashSAC"}
COLORS = {"rsl_rl_ppo": "#2563eb", "warp_ppo": "#7c3aed", "flashsac": "#d97706"}
REVIEWED_SUMMARY_SHA256 = "e8921b605345c720e06c837fe08366f1e58c4d0442f0c13441b0ddd9f592958c"


def _data_url(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def _mean(runs: list[dict], section: str, metric: str) -> float:
    return statistics.mean(run[section][metric] for run in runs)


def _number(value: float, *, percent: bool = False) -> str:
    return f"{value * 100:.1f}%" if percent else f"{value:.3f}"


def _quality_rows(runs: list[dict], scenario: str) -> str:
    rows = []
    for algorithm, label in ALGORITHMS.items():
        group = [run for run in runs if run["algorithm"] == algorithm]
        values = [run["final_quality"][scenario] for run in group]
        error = [point["linear_velocity_error_mean"] for point in values]
        survival = statistics.mean(point["survival_rate"] for point in values)
        success_metric = "walking_success_rate" if scenario == "forward_0_5" else "tracking_success_rate"
        success = statistics.mean(point[success_metric] for point in values)
        rows.append(
            f'<tr><th><i style="background:{COLORS[algorithm]}"></i>{label}</th>'
            f"<td>{statistics.mean(error):.3f}<small>{min(error):.3f}–{max(error):.3f}</small></td>"
            f"<td>{_number(survival, percent=True)}</td><td>{_number(success, percent=True)}</td></tr>"
        )
    return "".join(rows)


def _runtime_rows(runs: list[dict]) -> str:
    rows = []
    for algorithm, label in ALGORITHMS.items():
        group = [run for run in runs if run["algorithm"] == algorithm]
        timings = ("training_seconds", "update_seconds", "smoke_command_seconds", "all_three_commands_seconds")
        cells = "".join(f"<td>{_mean(group, 'timings', metric) / 60:.2f}</td>" for metric in timings)
        rows.append(f"<tr><th>{label}</th>{cells}</tr>")
    return "".join(rows)


def _gate_rows(runs: list[dict]) -> str:
    rows = []
    for run in runs:
        gate = run["walking_gate_observations"]
        first = gate["first_observed_pass"]
        if first:
            lower, upper = first["observed_transition_interval"]
            interval = f"({lower / 1e6:.3f}, {upper / 1e6:.3f}]"
            iteration = str(first["observed_iteration"])
            elapsed = f"{first['process_seconds_at_observation'] / 60:.2f}"
        else:
            interval, iteration, elapsed = "No observed pass", "—", "—"
        regressions = ", ".join(str(point["failed_iteration"]) for point in gate["pass_to_fail_regressions"])
        rows.append(
            f"<tr><th>{ALGORITHMS[run['algorithm']]}</th><td>{run['seed']}</td>"
            f"<td>{iteration}</td><td>{interval}</td><td>{elapsed}</td><td>{regressions or 'None observed'}</td></tr>"
        )
    return "".join(rows)


def _evidence_rows(runs: list[dict], output: Path) -> str:
    rows = []
    for run in runs:
        path = Path(run["comparison_path"])
        relative = html.escape(os.path.relpath(path, output.parent), quote=True)
        rows.append(
            f"<tr><th>{ALGORITHMS[run['algorithm']]} · seed {run['seed']}</th>"
            f'<td><a href="{relative}">Raw measurements</a></td><td><code>{run["comparison_sha256"]}</code></td></tr>'
        )
    return "".join(rows)


def _csv_export(runs: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "algorithm",
            "seed",
            "scenario",
            "iteration",
            "transitions",
            "loop_seconds",
            "planar_error_mps",
            "survival_rate",
            "walking_success_rate",
            "tracking_success_rate",
            "walking_gate_passed",
        ]
    )
    for run in runs:
        for point in run["common_evaluations"]:
            writer.writerow(
                [
                    run["algorithm"],
                    run["seed"],
                    point["scenario"],
                    point["iteration"],
                    point["transitions"],
                    point["time_seconds"],
                    point["linear_velocity_error_mean"],
                    point["survival_rate"],
                    point.get("walking_success_rate", ""),
                    point["tracking_success_rate"],
                    point.get("walking_gate_passed", ""),
                ]
            )
    return "data:text/csv;base64," + base64.b64encode(buffer.getvalue().encode()).decode()


def render_paper(summary_path: Path, figures_dir: Path, output: Path) -> None:
    """Write a self-contained report after checking the figure and source pins."""
    summary_raw = summary_path.read_bytes()
    if hashlib.sha256(summary_raw).hexdigest() != REVIEWED_SUMMARY_SHA256:
        raise ValueError("The curated paper narrative requires the reviewed experiment summary.")
    summary = json.loads(summary_raw)
    runs = summary["runs"]
    expected = {(algorithm, seed) for algorithm in ALGORITHMS for seed in (0, 1, 2)}
    if len(runs) != 9 or {(run["algorithm"], run["seed"]) for run in runs} != expected:
        raise ValueError("The paper requires exactly three seeds for each primary learner.")
    for run in runs:
        if hashlib.sha256(Path(run["comparison_path"]).read_bytes()).hexdigest() != run["comparison_sha256"]:
            raise ValueError("A raw input changed after retrieval verification.")
        if json.loads(Path(run["retrieval_audit_path"]).read_text())["status"] != "PASS":
            raise ValueError("All primary runs require a passing retrieval audit.")
    inventory = json.loads((figures_dir / "figures.json").read_text())
    if inventory["summary_sha256"] != hashlib.sha256(summary_raw).hexdigest():
        raise ValueError("Figures must come from this exact summary.")
    figures = {}
    for figure in inventory["figures"]:
        figures[figure["id"]] = {
            "svg": _data_url(figures_dir / figure["svg"], "image/svg+xml"),
            "pdf": _data_url(figures_dir / figure["pdf"], "application/pdf"),
            "caption": figure["caption"],
        }
    forward, native = (_quality_rows(runs, scenario) for scenario in ("forward_0_5", "native_commands"))
    plot_data = json.dumps(figures).replace("<", "\\u003c")
    ppo_pin = next(run["source_pins"]["isaaclab"] for run in runs if run["algorithm"] == "rsl_rl_ppo")
    flash_pin = next(run["source_pins"]["isaaclab"] for run in runs if run["algorithm"] == "flashsac")
    robo_pin = runs[0]["source_pins"]["robolearn"]
    replacements = {
        "@FIGURE_DATA@": plot_data,
        "@FORWARD_ROWS@": forward,
        "@NATIVE_ROWS@": native,
        "@RUNTIME_ROWS@": _runtime_rows(runs),
        "@GATE_ROWS@": _gate_rows(runs),
        "@EVIDENCE_ROWS@": _evidence_rows(runs, output),
        "@CSV@": _csv_export(runs),
        "@PPO_PIN@": ppo_pin,
        "@FLASH_PIN@": flash_pin,
        "@ROBO_PIN@": robo_pin,
        "@MDP_HASH@": summary["mdp_sha256"],
        "@SUMMARY_HASH@": hashlib.sha256(summary_raw).hexdigest(),
    }
    document = Path(__file__).with_name("g1_learning_paper.html").read_text()
    for key, value in replacements.items():
        document = document.replace(key, value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_paper(args.summary, args.figures_dir, args.output)
    print(f"Report: {args.output.resolve()} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
