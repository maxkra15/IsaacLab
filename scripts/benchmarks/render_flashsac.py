# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render recorded Cartpole FlashSAC comparisons without changing raw evidence.

Example::

    uv run --no-sync python scripts/benchmarks/render_flashsac.py \
        --input logs/cartpole/comparison.json --output logs/cartpole/report.html

The HTML embeds all figures and raw JSON downloads. Standalone SVG/PDF figures,
byte-preserved input files, and a derived inventory are also written beside it.
Use ``--public`` for a compact derived dataset with original-input hashes, omitting
archive paths, full source manifests, checkpoints, and private context documents.
Choose a fresh public output directory; existing files are not removed.
Only recorded evaluations are summarized; incomplete cohorts remain explicit.
This renderer imports neither Isaac Lab nor a learner and does not use a GPU.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

RECIPES = {
    "torch-fp32-eager": ("Torch FP32 · eager", "#2563eb"),
    "torch-fp32-compiled": ("Torch FP32 · compiled", "#0891b2"),
    "torch-fp16-compiled": ("Torch FP16 AMP · compiled", "#d97706"),
    "warp-fp32-captured": ("Warp FP32 · captured", "#7c3aed"),
}
METRICS = (
    ("return_mean", "Evaluation return", "Return", False),
    ("upright_fraction", "Upright occupancy", "Fraction", True),
    ("pole_angle_rms", "Wrapped pole angle RMS", "Radians", False),
    ("cart_position_rms", "Cart position RMS", "Metres", False),
    ("survival_rate", "Cart-bound survival", "Fraction", True),
    ("action_saturation_fraction", "Action saturation", "Fraction", True),
)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _label(recipe: str) -> str:
    return RECIPES.get(recipe, (recipe.replace("_", " ").replace("-", " "), "#536174"))[0]


def _color(recipe: str) -> str:
    return RECIPES.get(recipe, (recipe, "#536174"))[1]


def _number(value: Any, *, fraction: bool = False) -> str:
    if not _finite(value):
        return "—"
    return f"{100 * value:.1f}%" if fraction else f"{value:,.3g}"


def _count(value: Any) -> str:
    return f"{value:,.0f}" if _finite(value) else "—"


def _range(values: list[float], *, fraction: bool = False) -> str:
    if not values:
        return "—"
    mean = _number(statistics.mean(values), fraction=fraction)
    low, high = (_number(value, fraction=fraction) for value in (min(values), max(values)))
    return f"{mean}<small>{low}–{high}; n={len(values)}</small>"


def _data_url(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def _pick(value: dict, keys: tuple[str, ...]) -> dict:
    return {key: value[key] for key in keys if key in value}


def _artifact_transfer_receipt(path: Path | None) -> dict | None:
    """Read only the explicit transfer schema; never publish arbitrary receipt text."""
    if path is None:
        return None
    raw = path.read_bytes()
    original = json.loads(raw)
    workflow = original.get("workflow_id")
    if not isinstance(workflow, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", workflow):
        raise ValueError("Artifact transfer workflow_id must be a workflow name without paths.")
    receipt = {"workflow_id": workflow, "original_receipt_sha256": hashlib.sha256(raw).hexdigest()}
    timestamps = []
    for field in ("started_utc", "finished_utc"):
        value = original.get(field)
        if not isinstance(value, str):
            raise ValueError(f"Artifact transfer {field} must be a UTC ISO 8601 timestamp.")
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if instant.utcoffset() is None or instant.utcoffset().total_seconds() != 0:
            raise ValueError(f"Artifact transfer {field} must use UTC.")
        timestamps.append(instant)
        receipt[field] = instant.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if timestamps[1] < timestamps[0]:
        raise ValueError("Artifact transfer finished before its recorded start.")
    for field in ("completed_runs_at_start", "exit_code", "staged_bytes"):
        if field == "staged_bytes" and field not in original:
            continue
        value = original.get(field)
        if not _finite(value) or value < 0 or int(value) != value:
            raise ValueError(f"Artifact transfer {field} must be a finite nonnegative integer.")
        receipt[field] = int(value)
    receipt["public_disclosure"] = (
        "Network and disk artifact-transfer traffic overlapped later training runs. "
        "The transfer ran no GPU profiling or computation; host or disk overhead may affect later timings. "
        "Recorded timing values are unchanged. Staged bytes describe retained files, not measured wire traffic."
    )
    return receipt


def _public_document(document: dict, original_sha256: str) -> dict:
    """Project the comparison-driver schema explicitly, preserving measured values."""
    original = document.get("metadata", {})
    metadata = _pick(original, ("task", "physics", "mdp", "smoke", "synthetic", "driver_sha256"))
    metadata["dataset_kind"] = "derived public measurements"
    metadata["original_input_sha256"] = original_sha256
    metadata["protocol"] = _pick(
        original.get("protocol", {}),
        (
            "num_envs",
            "horizon",
            "iterations",
            "eval_envs",
            "eval_seed",
            "checkpoints",
            "buffer_max_length",
            "buffer_min_length",
            "sample_batch_size",
            "learning_rate_decay_step",
            "updates_per_step",
            "n_step",
            "actor_update_period",
        ),
    )
    metadata["versions"] = _pick(
        original.get("versions", {}),
        ("torch", "warp-lang", "warp-nn", "robolearn-rl", "newton", "mujoco", "mujoco-warp", "python", "cuda"),
    )
    metadata["timing_definitions"] = _pick(
        original.get("timing_definitions", {}),
        (
            "training_seconds",
            "process_seconds",
            "learn_seconds",
            "loop_seconds",
            "learn_overhead_seconds",
            "checkpoint_seconds",
            "first_policy_call_seconds",
            "first_update_group_seconds",
        ),
    )
    repositories = {}
    source_files = {
        "isaaclab": (
            "scripts/benchmarks/compare_flashsac.py",
            "source/isaaclab_rl/isaaclab_rl/robolearn/runner.py",
            "source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_manager_env_cfg.py",
            "source/isaaclab/isaaclab/envs/mdp/rewards.py",
        ),
        "robolearn": (
            "src/robolearn/flashsac/config.py",
            "src/robolearn/flashsac/agent.py",
            "src/robolearn/warp/flashsac.py",
            "src/robolearn/warp/_flash_networks.py",
            "src/robolearn/warp/_flash_normalization.py",
            "src/robolearn/warp/_flash_replay.py",
            "src/robolearn/warp/_linear.py",
        ),
    }
    for name, fields in source_files.items():
        recorded = original.get("source", {}).get("repositories", {}).get(name, {})
        if recorded:
            repositories[name] = _pick(recorded, ("base_commit", "archive_sha256"))
            repositories[name]["file_sha256"] = _pick(recorded.get("file_sha256", {}), fields)
    metadata["source"] = {"repositories": repositories}
    algorithm_fields = (
        "device",
        "buffer_device",
        "seed",
        "normalize_reward",
        "normalized_G_max",
        "asymmetric_observation",
        "buffer_max_length",
        "buffer_min_length",
        "sample_batch_size",
        "learning_rate_init",
        "learning_rate_peak",
        "learning_rate_end",
        "learning_rate_warmup_rate",
        "learning_rate_warmup_step",
        "learning_rate_decay_rate",
        "learning_rate_decay_step",
        "actor_num_blocks",
        "actor_hidden_dim",
        "actor_bc_alpha",
        "actor_noise_zeta_mu",
        "actor_noise_zeta_max",
        "actor_update_period",
        "critic_num_blocks",
        "critic_hidden_dim",
        "critic_num_bins",
        "critic_min_v",
        "critic_max_v",
        "critic_target_update_tau",
        "temp_initial_value",
        "temp_target_sigma",
        "temp_target_entropy",
        "gamma",
        "n_step",
        "use_compile",
        "compile_mode",
        "use_amp",
        "load_optimizer",
        "load_reward_normalizer",
    )
    timing_fields = (
        "environment_startup_seconds",
        "constructor_seconds",
        "fixture_load_seconds",
        "learn_seconds",
        "prepare_seconds",
        "checkpoint_seconds",
        "first_policy_call_seconds",
        "first_update_group_seconds",
        "steady_from_iteration",
        "loop_seconds",
        "collection_seconds",
        "update_seconds",
        "steady_loop_seconds",
        "steady_collection_seconds",
        "steady_update_seconds",
        "steady_transitions",
        "steady_fps",
        "startup_seconds",
        "learn_overhead_seconds",
    )
    evaluation_fields = (
        "iteration",
        "transitions",
        "training_seconds",
        "return_mean",
        "return_std",
        "normalized_score",
        "episode_length_mean",
        "survival_rate",
        "time_limit_fraction",
        "cart_bound_fraction",
        "upright_fraction",
        "pole_angle_abs_mean",
        "pole_angle_rms",
        "cart_position_abs_mean",
        "cart_position_rms",
        "unwrapped_pole_angle_squared_mean",
        "action_rms",
        "action_saturation_fraction",
        "action_abs_max",
        "episode_count",
        "max_episode_steps",
        "step_dt",
    )
    runs = []
    for original_run in document["runs"]:
        run = _pick(
            original_run,
            (
                "recipe",
                "seed",
                "algorithm",
                "fixture_sha256",
                "transitions",
                "iterations",
                "process_seconds",
                "evaluation_process_seconds",
            ),
        )
        config = original_run.get("agent_config", {})
        run["agent_config"] = _pick(
            config,
            (
                "algorithm",
                "seed",
                "device",
                "num_steps_per_env",
                "max_iterations",
                "save_interval",
                "observation_group",
                "critic_group",
                "clip_actions",
                "init_at_random_ep_len",
                "updates_per_step",
                "flash_updates_during_rollout",
                "capture_updates",
                "capture_rollout",
            ),
        )
        run["agent_config"]["algorithm_cfg"] = _pick(config.get("algorithm_cfg", {}), algorithm_fields)
        run["timing"] = _pick(original_run.get("timing", {}), timing_fields)
        run["optimizer"] = {
            role: _pick(
                original_run.get("optimizer", {}).get(role, {}),
                ("attempted", "completed", "min", "max", "coverage", "amp_skipped"),
            )
            for role in ("actor", "critic", "temperature")
        }
        run["health"] = _pick(
            original_run.get("health", {}),
            (
                "finite_weights",
                "finite_optimizer",
                "actor_parameters_changed",
                "critic_parameters_changed",
                "nonfinite_actions",
            ),
        )
        run["hardware"] = _pick(original_run.get("hardware", {}), ("gpu", "total_memory_bytes", "compute_capability"))
        run["gpu_isolation"] = (
            "dedicated one-L40 allocation recorded before comparison"
            if str(original_run.get("gpu_isolation", "")).startswith("dedicated-one-L40")
            else "unknown"
        )
        evaluations = []
        for point in original_run.get("evaluations", []):
            selected = _pick(point, evaluation_fields)
            selected["initial_state_hashes"] = {
                field: _pick(point.get("initial_state_hashes", {}).get(field, {}), ("sha256", "shape", "dtype"))
                for field in ("joint_pos", "joint_vel", "root_pose", "root_velocity", "observations")
            }
            evaluations.append(selected)
        run["evaluations"] = evaluations
        primer = original_run.get("evaluation_initialization")
        if primer:
            run["evaluation_initialization"] = _pick(primer, ("seed", "unscored_native_resets"))
            for stage in ("before", "after"):
                key = f"soft_joint_velocity_limits_{stage}"
                run["evaluation_initialization"][key] = _pick(
                    primer.get(key, {}), ("sha256", "shape", "per_joint_min", "per_joint_max")
                )
        runs.append(run)
    return {"metadata": metadata, "runs": runs}


def _load_inputs(
    paths: list[Path], output: Path, contexts: list[Path], appendix: Path | None, public: bool = False
) -> dict:
    """Read immutable measurements and reject repeated learner seeds."""
    data_dir = output.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sources, runs, seen = [], [], set()
    protocols, tasks, backends = set(), set(), set()
    for index, path in enumerate(paths):
        raw = path.read_bytes()
        document = json.loads(raw)
        metadata = document.get("metadata", {})
        if not isinstance(document.get("runs"), list) or not document["runs"]:
            raise ValueError(f"{path} must contain a nonempty runs list.")
        protocols.add(json.dumps(metadata.get("protocol", {}), sort_keys=True))
        tasks.add(metadata.get("task"))
        backends.add((metadata.get("physics"), metadata.get("mdp")))
        original_sha256 = hashlib.sha256(raw).hexdigest()
        if public:
            document = _public_document(document, original_sha256)
            metadata = document["metadata"]
            raw = (json.dumps(document, indent=2, allow_nan=False) + "\n").encode()
        source_name = (
            ("measurements.json" if len(paths) == 1 else f"measurements-{index + 1:02d}.json")
            if public
            else f"comparison-{index + 1:02d}.json"
        )
        (data_dir / source_name).write_bytes(raw)
        sources.append(
            {
                "path": path.name if public else str(path.resolve()),
                "name": source_name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "raw": raw,
                "metadata": metadata,
                **({"original_input_sha256": original_sha256} if public else {}),
            }
        )
        for original in document["runs"]:
            key = (original["recipe"], original["seed"])
            if key in seen:
                raise ValueError(f"Repeated recipe/seed {key}; choose one measured cohort.")
            seen.add(key)
            run = dict(original)
            run["report_metadata"] = metadata
            run["report_source"] = source_name
            evaluations = sorted(run.get("evaluations", []), key=lambda point: point["iteration"])
            if len({point["iteration"] for point in evaluations}) != len(evaluations):
                raise ValueError(f"Repeated evaluation checkpoint in {key}.")
            run["evaluations"] = evaluations
            runs.append(run)
    # Different task protocols are separate experiments, not additional seeds.
    if len(protocols) != 1 or len(tasks) != 1 or len(backends) != 1:
        raise ValueError("Inputs must have the same task, backend, MDP, and protocol.")
    evidence = []
    for index, path in enumerate([*contexts, *([appendix] if appendix else [])]):
        raw = path.read_bytes()
        document = json.loads(raw)
        name = f"{'numerical' if path == appendix else 'context'}-{index + 1:02d}.json"
        (data_dir / name).write_bytes(raw)
        evidence.append(
            {
                "path": str(path.resolve()),
                "name": name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "raw": raw,
                "document": document,
                "appendix": path == appendix,
            }
        )
    order = {recipe: index for index, recipe in enumerate(RECIPES)}
    runs.sort(key=lambda run: (order.get(run["recipe"], len(order)), run["recipe"], run["seed"]))
    return {"runs": runs, "sources": sources, "contexts": evidence}


def _groups(runs: list[dict]) -> dict[str, list[dict]]:
    return {
        recipe: [run for run in runs if run["recipe"] == recipe]
        for recipe in dict.fromkeys(run["recipe"] for run in runs)
    }


def _measured_summary(runs: list[dict], final_iteration: int, metadata: dict) -> str:
    """Describe available measured means without defining a winner or quality gate."""
    if metadata.get("synthetic"):
        return ""
    grouped = _groups(runs)
    final = {
        recipe: [point for run in group for point in run["evaluations"] if point["iteration"] == final_iteration]
        for recipe, group in grouped.items()
    }
    available = {
        recipe: points
        for recipe, points in final.items()
        if len(points) == len(grouped[recipe])
        and all(_finite(point.get("return_mean")) and _finite(point.get("upright_fraction")) for point in points)
    }
    if not available:
        return ""
    expected = {(recipe, seed) for recipe in RECIPES for seed in (0, 1, 2)}
    complete = {(run["recipe"], run["seed"]) for run in runs} == expected and len(available) == len(grouped)
    scope = "In the complete recorded cohort" if complete else "In this partial recorded cohort"
    if metadata.get("smoke"):
        scope = "In this short integration smoke"
    descriptions = []
    for recipe, points in available.items():
        mean_return = statistics.mean(point["return_mean"] for point in points)
        upright_values = [point["upright_fraction"] for point in points]
        upright = statistics.mean(upright_values)
        survival = [point.get("survival_rate") for point in points]
        survival_description = (
            f", cart-bound survival {_number(statistics.mean(survival), fraction=True)}"
            if all(_finite(value) for value in survival)
            else ""
        )
        descriptions.append(
            f"{html.escape(_label(recipe))}: return {_number(mean_return)}, "
            f"upright {_number(upright, fraction=True)} "
            f"(observed seed range {_number(min(upright_values), fraction=True)}–"
            f"{_number(max(upright_values), fraction=True)}){survival_description} (n={len(points)})"
        )
    sentences = [f"{scope}, final iteration {final_iteration} means were " + "; ".join(descriptions) + "."]
    budgets = {run["transitions"] for run in runs}
    if len(budgets) == 1:
        sentences.append(
            f"All included runs collected {_count(next(iter(budgets)))} transitions; runtime is compared "
            "at that fixed sample budget."
        )
    warp = grouped.get("warp-fp32-captured", [])
    comparisons = []
    process_costs = []
    if warp:
        warp_fps = [run.get("timing", {}).get("steady_fps") for run in warp]
        warp_process = [run.get("process_seconds") for run in warp]
        for recipe in ("torch-fp32-compiled", "torch-fp16-compiled"):
            group = grouped.get(recipe, [])
            if not group:
                continue
            fps = [run.get("timing", {}).get("steady_fps") for run in group]
            if all(_finite(value) and value > 0 for value in [*warp_fps, *fps]):
                ratio = statistics.mean(warp_fps) / statistics.mean(fps)
                comparisons.append(f"{ratio:.3f}× {html.escape(_label(recipe))} (n={len(group)})")
            process = [run.get("process_seconds") for run in group]
            if all(_finite(value) and value > 0 for value in [*warp_process, *process]):
                cost = statistics.mean(process)
                ratio = cost / statistics.mean(warp_process)
                process_costs.append(
                    f"{html.escape(_label(recipe))} {_number(cost)} s "
                    f"(Torch/Warp wall ratio {ratio:.3f}; n={len(group)})"
                )
        if comparisons:
            sentences.append(f"Warp’s mean warmed throughput (n={len(warp)}) was " + " and ".join(comparisons) + ".")
        if process_costs:
            sentences.append(
                f"Mean full training-process wall was {_number(statistics.mean(warp_process))} s "
                f"for Warp (n={len(warp)}), versus " + "; ".join(process_costs) + "."
            )
    return '<p class="note">' + " ".join(sentences) + "</p>"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": "#a9b1bd",
            "text.color": "#192333",
            "axes.labelcolor": "#536174",
            "xtick.color": "#536174",
            "ytick.color": "#536174",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "robolearn-cartpole-flashsac",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _export_figure(figure: plt.Figure, directory: Path, name: str) -> dict:
    exports = {}
    for extension in ("svg", "pdf"):
        path = directory / f"{name}.{extension}"
        metadata = {"Date": None} if extension == "svg" else {"CreationDate": None, "ModDate": None}
        figure.savefig(path, bbox_inches="tight", metadata=metadata)
        exports[extension] = path.read_bytes()
    plt.close(figure)
    return exports


def _plot_learning(runs: list[dict], directory: Path, axis: str) -> dict:
    """Draw seed curves and checkpoint means; bands describe the observed range."""
    figure, axes = plt.subplots(2, 3, figsize=(11.6, 6.1))
    for subplot, (metric, title, unit, fraction) in zip(axes.flat, METRICS, strict=True):
        for recipe, group in _groups(runs).items():
            checkpoint_values: dict[int, list[tuple[float, float]]] = {}
            for run in group:
                points = [
                    point for point in run["evaluations"] if _finite(point.get(metric)) and _finite(point.get(axis))
                ]
                xs = [point[axis] for point in points]
                ys = [point[metric] for point in points]
                subplot.plot(xs, ys, color=_color(recipe), alpha=0.25, linewidth=0.9, marker=".", markersize=3)
                for point in points:
                    checkpoint_values.setdefault(point["iteration"], []).append((point[axis], point[metric]))
            # A mean/range requires every included seed at the same checkpoint.
            samples = [values for _, values in sorted(checkpoint_values.items()) if len(values) == len(group)]
            if samples:
                xs = [statistics.mean(pair[0] for pair in values) for values in samples]
                ys = [statistics.mean(pair[1] for pair in values) for values in samples]
                subplot.fill_between(
                    xs,
                    [min(pair[1] for pair in values) for values in samples],
                    [max(pair[1] for pair in values) for values in samples],
                    color=_color(recipe),
                    alpha=0.09,
                    linewidth=0,
                )
                subplot.plot(
                    xs,
                    ys,
                    color=_color(recipe),
                    linewidth=1.8,
                    marker="o",
                    markersize=3,
                    label=f"{_label(recipe)} (n={len(group)})",
                )
        subplot.set_title(title, loc="left")
        subplot.set_ylabel(unit)
        subplot.grid(axis="y", color="#e7ebef", linewidth=0.6)
        if axis == "transitions":
            subplot.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e6:g}"))
            subplot.set_xlabel("Collected transitions (millions)")
        else:
            subplot.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 60:g}"))
            subplot.set_xlabel("Recorded training time (minutes)")
        if fraction:
            subplot.set_ylim(-0.03, 1.03)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    figure.tight_layout(rect=(0, 0.065, 1, 1), h_pad=2, w_pad=2)
    return _export_figure(figure, directory, f"learning-{'transitions' if axis == 'transitions' else 'time'}")


def _plot_runtime(runs: list[dict], directory: Path) -> dict:
    """Keep overlapping phase measurements in separate panels rather than stack them."""
    panels = (
        ("Production process and loop", (("process_seconds", "Process"), ("loop_seconds", "Loop")), "Seconds"),
        (
            "Startup and first-use costs",
            (
                ("startup_seconds", "Startup"),
                ("prepare_seconds", "Prepare"),
                ("first_policy_call_seconds", "First policy"),
                ("first_update_group_seconds", "First update"),
            ),
            "Seconds",
        ),
        (
            "Warm loop phase totals",
            (("steady_collection_seconds", "Collection"), ("steady_update_seconds", "Update")),
            "Seconds",
        ),
        ("Steady training throughput", (("steady_fps", "Transitions / second"),), "Transitions / second"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.6, 7.1))
    grouped = _groups(runs)
    for subplot, (title, metrics, unit) in zip(axes.flat, panels, strict=True):
        width = 0.8 / len(grouped)
        for group_index, (recipe, group) in enumerate(grouped.items()):
            xs = np.arange(len(metrics)) - 0.4 + width * (group_index + 0.5)
            for x, (metric, _) in zip(xs, metrics, strict=True):
                values = [run.get(metric, run.get("timing", {}).get(metric)) for run in group]
                values = [value for value in values if _finite(value)]
                if not values:
                    continue
                mean = statistics.mean(values)
                subplot.bar(x, mean, width=width * 0.92, color=_color(recipe), alpha=0.85)
                subplot.errorbar(
                    x,
                    mean,
                    yerr=[[mean - min(values)], [max(values) - mean]],
                    color="#192333",
                    linewidth=0.7,
                    capsize=2,
                    fmt="none",
                )
                subplot.scatter([x] * len(values), values, color="#192333", s=7, alpha=0.7, zorder=3)
        subplot.set_title(title, loc="left")
        subplot.set_xticks(np.arange(len(metrics)), [label for _, label in metrics])
        subplot.set_ylabel(unit)
        subplot.grid(axis="y", color="#e7ebef", linewidth=0.6)
        subplot.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=_color(recipe)) for recipe in grouped]
    figure.legend(
        handles,
        [_label(recipe) for recipe in grouped],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.025),
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1), h_pad=2, w_pad=2)
    return _export_figure(figure, directory, "runtime")


def _figure(exports: dict, title: str, caption: str, name: str) -> str:
    image_url = _data_url(exports["svg"], "image/svg+xml")
    links = "".join(
        f'<a download="{name}.{extension}" href="{_data_url(raw, mime)}">{extension.upper()}</a>'
        for extension, raw, mime in (
            ("svg", exports["svg"], "image/svg+xml"),
            ("pdf", exports["pdf"], "application/pdf"),
        )
    )
    return (
        f'<figure><div class="figure-scroll"><img src="{image_url}" alt="{html.escape(title, quote=True)}"></div>'
        f'<figcaption>{caption}</figcaption><div class="exports">Download: {links}</div></figure>'
    )


def _quality_table(runs: list[dict], final_iteration: int) -> str:
    rows = []
    for recipe, group in _groups(runs).items():
        final = [point for run in group for point in run["evaluations"] if point["iteration"] == final_iteration]
        cells = []
        for metric, _, _, fraction in METRICS:
            values = [point[metric] for point in final if _finite(point.get(metric))]
            cells.append(f"<td>{_range(values, fraction=fraction)}</td>")
        rows.append(
            f"<tr><th>{html.escape(_label(recipe))}<small>{len(final)}/{len(group)} final evaluations</small>"
            f"</th>{''.join(cells)}</tr>"
        )
    headings = "".join(f"<th>{html.escape(title)}</th>" for _, title, _, _ in METRICS)
    return (
        f'<div class="table-scroll"><table><thead><tr><th>Recipe</th>{headings}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _runtime_table(runs: list[dict]) -> str:
    metrics = (
        "process_seconds",
        "loop_seconds",
        "startup_seconds",
        "collection_seconds",
        "update_seconds",
        "checkpoint_seconds",
        "steady_fps",
    )
    headings = ("Process s", "Loop s", "Startup s", "Collection s", "Update s", "Checkpoint s", "Steady FPS")
    rows = []
    for recipe, group in _groups(runs).items():
        cells = []
        for metric in metrics:
            values = [run.get(metric, run.get("timing", {}).get(metric)) for run in group]
            cells.append(f"<td>{_range([value for value in values if _finite(value)])}</td>")
        rows.append(f"<tr><th>{html.escape(_label(recipe))}</th>{''.join(cells)}</tr>")
    return (
        '<div class="table-scroll"><table><thead><tr><th>Recipe</th>'
        + "".join(f"<th>{heading}</th>" for heading in headings)
        + f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _optimizer_table(runs: list[dict]) -> str:
    rows = []
    for run in runs:
        cells = []
        for role in ("actor", "critic", "temperature"):
            counter = run.get("optimizer", {}).get(role, {})
            completed = _count(counter.get("completed"))
            low, high = counter.get("min"), counter.get("max")
            if _finite(low) and _finite(high):
                completed += f"<small>parameter steps {low:,}–{high:,}</small>"
            cells.append(
                f"<td>{completed}<small>attempted {_count(counter.get('attempted'))}; "
                f"skipped {_count(counter.get('amp_skipped'))}; "
                f"coverage {html.escape(str(counter.get('coverage', 'unrecorded')))}</small></td>"
            )
        health = run.get("health", {})
        proof = "; ".join(
            f"{key.replace('_', ' ')}: {health.get(key, 'unrecorded')}"
            for key in (
                "finite_weights",
                "finite_optimizer",
                "actor_parameters_changed",
                "critic_parameters_changed",
                "nonfinite_actions",
            )
        )
        rows.append(
            f"<tr><th>{html.escape(_label(run['recipe']))}<small>seed {run['seed']}</small></th>"
            f'{"".join(cells)}<td class="proof">{html.escape(proof)}</td></tr>'
        )
    return (
        '<div class="table-scroll"><table><thead><tr><th>Recipe / seed</th><th>Actor</th>'
        "<th>Critic</th><th>Temperature</th><th>Native-state health</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _evidence_table(sources: list[dict]) -> str:
    rows = []
    for source in sources:
        link = _data_url(source["raw"], "application/json")
        original = source.get("original_input_sha256")
        hashes = f"<code>{source['sha256']}</code>"
        if original:
            hashes = f"Published data: {hashes}<small>Original input: <code>{original}</code></small>"
        rows.append(
            f'<tr><th><a download="{source["name"]}" href="{link}">{source["name"]}</a></th>'
            f"<td>{hashes}<small>{html.escape(source['path'])}</small></td></tr>"
        )
    public = all("original_input_sha256" in source for source in sources)
    label = "Derived public download" if public else "Unchanged raw download"
    hash_label = "Published / original-input SHA256" if public else "SHA256 / original path"
    return (
        f'<div class="table-scroll"><table class="evidence"><thead><tr><th>{label}</th>'
        f"<th>{hash_label}</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _details(title: str, document: Any) -> str:
    text = html.escape(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False))
    return f"<details><summary>{html.escape(title)}</summary><pre><code>{text}</code></pre></details>"


CSS = """
:root{--ink:#192333;--muted:#536174;--line:#dce1e7;--wash:#f4f6f8}*{box-sizing:border-box}
body{margin:0;background:white;color:var(--ink);font:16px/1.65 system-ui,-apple-system,sans-serif}
main{max-width:1060px;margin:auto;padding:52px 40px}header{border-bottom:2px solid var(--ink);padding-bottom:25px}
.eyebrow{font-size:11px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
h1{font:500 clamp(34px,5vw,50px)/1.15 Georgia,serif;letter-spacing:-.03em;max-width:850px;margin:17px 0}
.subtitle{font-size:18px;color:var(--muted);max-width:780px}.byline{font-size:12px;color:var(--muted)}
nav{display:flex;gap:22px;flex-wrap:wrap;padding:16px 0;border-bottom:1px solid var(--line);font-size:13px}
a{color:#244a80;text-underline-offset:3px}nav a{text-decoration:none}
a:focus-visible,button:focus-visible{outline:3px solid #94b9ff;outline-offset:3px}
section{margin:38px 0;scroll-margin-top:20px}h2{font:500 27px/1.3 Georgia,serif}h3{font-size:16px;margin-top:24px}
p{max-width:890px}.abstract{font:18px/1.65 Georgia,serif}
.note{padding:14px 18px;background:var(--wash);border-left:3px solid #8793a3;font-size:13px}
figure{margin:22px 0 30px}.figure-scroll{overflow-x:auto}figure img{width:100%;height:auto;display:block}
figcaption{font-size:13px;color:var(--muted);margin-top:12px}
.exports{font-size:11px;display:flex;gap:13px;margin-top:9px}
.controls{display:flex;align-items:center;flex-wrap:wrap;gap:8px;font-size:12px}
.controls span{color:var(--muted);margin-right:4px}
button{font:600 12px system-ui,sans-serif;padding:8px 12px;background:white;color:var(--ink);
border:1px solid var(--line);border-radius:4px;cursor:pointer}
button[aria-pressed=true]{background:var(--ink);color:white}.table-scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:12px;line-height:1.5;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:11px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th:first-child{text-align:left;min-width:155px}
thead th{border-top:1px solid var(--ink);border-bottom:1px solid var(--ink);font-size:11px;color:var(--muted)}
small{display:block;font-size:10px;color:var(--muted);margin-top:5px}.proof{text-align:left;max-width:260px;font-size:11px}
details{border-top:1px solid var(--line);padding:12px 0;margin-top:12px}
summary{cursor:pointer;font-size:13px;font-weight:600}
pre{padding:16px;background:var(--wash);overflow:auto}
code{font:11px/1.55 ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere}
.evidence td{text-align:left}.evidence code{font-size:10px}li{margin:9px 0;font-size:13px}
footer{border-top:1px solid var(--line);padding-top:18px;color:var(--muted);font-size:11px}
[hidden]{display:none!important}@media(max-width:680px){main{padding:30px 18px}nav{gap:14px}
th,td{padding:9px 6px}h2{font-size:24px}figure img{min-width:660px}}
@media print{main{padding:0;max-width:none}body{font-size:11px}nav,.controls,.exports{display:none}
.chart-panel[hidden]{display:block!important}h1{font-size:32px}h2{font-size:22px}
figure{break-inside:avoid}section{margin:25px 0}figure img{min-width:0}}
"""


def render_report(
    inputs: list[Path],
    output: Path,
    contexts: list[Path],
    appendix: Path | None,
    public: bool = False,
    artifact_transfer: Path | None = None,
) -> None:
    """Write a self-contained HTML paper and scientific figure exports."""
    if public and (contexts or appendix):
        raise ValueError("--public omits context and numerical-appendix documents; render those in local mode.")
    transfer = _artifact_transfer_receipt(artifact_transfer)
    output.parent.mkdir(parents=True, exist_ok=True)
    measured = _load_inputs(inputs, output, contexts, appendix, public)
    runs, sources = measured["runs"], measured["sources"]
    metadata = sources[0]["metadata"]
    protocol = metadata.get("protocol", {})
    final_iteration = int(protocol.get("iterations", max(run["iterations"] for run in runs)))
    expected_checkpoints = protocol.get("checkpoints", [0, 50, 100, 150, final_iteration])
    figures_dir = output.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _style()
    figures = {
        "transitions": _plot_learning(runs, figures_dir, "transitions"),
        "time": _plot_learning(runs, figures_dir, "training_seconds"),
        "runtime": _plot_runtime(runs, figures_dir),
    }
    groups = _groups(runs)
    cohort = "; ".join(
        f"{_label(recipe)}: seeds {', '.join(str(run['seed']) for run in group)}" for recipe, group in groups.items()
    )
    completed = sum(
        all(
            any(point["iteration"] == checkpoint for point in run["evaluations"]) for checkpoint in expected_checkpoints
        )
        for run in runs
    )
    expected = {(recipe, seed) for recipe in RECIPES for seed in (0, 1, 2)}
    complete_cohort = {(run["recipe"], run["seed"]) for run in runs} == expected
    coverage = f"{completed}/{len(runs)} runs contain all requested evaluations. " + (
        "The four-recipe, three-seed cohort is complete."
        if complete_cohort
        else "This is a partial or differently sized cohort; no missing seeds are imputed."
    )
    diagnostic_note = ""
    if metadata.get("synthetic"):
        diagnostic_note = '<p class="note"><strong>SYNTHETIC SCHEMA FIXTURE — NOT TRAINING RESULTS.</strong> '
        diagnostic_note += "This file only checks the offline renderer and contains placeholder values.</p>"
    elif metadata.get("smoke"):
        diagnostic_note = '<p class="note"><strong>Short integration smoke.</strong> '
        diagnostic_note += "This reduced budget is a runtime check, not the production learning comparison.</p>"
    transition_values = sorted({run["transitions"] for run in runs})
    budget = ", ".join(f"{value:,}" for value in transition_values)
    inventory = {
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources": [_pick(source, ("path", "name", "sha256", "original_input_sha256")) for source in sources],
        "contexts": [{key: source[key] for key in ("path", "name", "sha256")} for source in measured["contexts"]],
        **({"artifact_transfer": transfer} if transfer else {}),
        "runs": [
            {
                "recipe": run["recipe"],
                "seed": run["seed"],
                "transitions": run["transitions"],
                "fixture_sha256": run.get("fixture_sha256"),
                **({"log_dir": run.get("log_dir")} if not public else {}),
                "evaluated_iterations": [point["iteration"] for point in run["evaluations"]],
            }
            for run in runs
        ],
    }
    inventory_path = output.parent / ("data/provenance.json" if public else "report-inventory.json")
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
    timing_definition = metadata.get("timing_definitions", {}).get(
        "training_seconds",
        "Cumulative synchronized collection-plus-update "
        "iteration wall time; excludes constructor, prepare, checkpoint I/O, logging, "
        "imports, and teardown. Startup is shown separately.",
    )
    appendix_section = ""
    if appendix:
        entry = next(source for source in measured["contexts"] if source["appendix"])
        appendix_section = (
            '<section id="numerical"><h2>Numerical appendix</h2><p class="note">'
            "This is an initialized fixed-batch diagnostic, not a training-quality result. FP32 reductions and "
            "rare critic activation or minimum-Q branch differences can change gradients and subsequent Adam steps. "
            "A local FP64 comparison does not establish universal accuracy or speed. Exact Torch trajectory matching "
            "is not a gate for the learning comparison.</p>"
            + _evidence_table([entry])
            + _details("Recorded numerical diagnostic", entry["document"])
            + "</section>"
        )
    source_details = "".join(
        _details(f"Measurement metadata · {source['name']}", source["metadata"]) for source in sources
    )
    run_details = [
        {
            "recipe": run["recipe"],
            "seed": run["seed"],
            "agent_config": run.get("agent_config"),
            "fixture_sha256": run.get("fixture_sha256"),
            "timing": run.get("timing"),
            "optimizer": run.get("optimizer"),
            "health": run.get("health"),
            "hardware": run.get("hardware"),
            "evaluation_process_seconds": run.get("evaluation_process_seconds"),
            "evaluation_initialization": run.get("evaluation_initialization"),
            **({"gpu_isolation": run.get("gpu_isolation")} if public else {"log_dir": run.get("log_dir")}),
        }
        for run in runs
    ]
    context_details = "".join(
        _details(f"Runtime context · {source['name']}", source["document"])
        for source in measured["contexts"]
        if not source["appendix"]
    )
    time_caption = (
        "Thin lines show individual seeds; thick lines show checkpoint means, with the observed min–max "
        "range. Each time-axis mean uses the mean recorded elapsed time at that checkpoint; it does not interpolate "
        f"an exact crossing time. {html.escape(str(timing_definition))} At checkpoint zero, training time is zero "
        "for the initial weights; setup, preparation, and the separate evaluation subprocess still cost wall time."
    )
    initialization_note = ""
    if all(run.get("evaluation_initialization", {}).get("unscored_native_resets") == 1 for run in runs):
        initialization_note = (
            "One unscored native reset initializes the actuator soft velocity-limit buffers before the scored "
            "seeded resets; per-run initialization receipts retain the before/after limits."
        )
    transfer_note = ""
    if transfer:
        transfer_note = (
            "<li><strong>Artifact-transfer overlap:</strong> transfer began at "
            f"{html.escape(transfer['started_utc'])} with {transfer['completed_runs_at_start']} runs completed, "
            f"and ended at {html.escape(transfer['finished_utc'])}. "
            f"{html.escape(transfer['public_disclosure'])}</li>"
        )
    evidence_description = (
        "Downloads contain compact derived public measurements. Recorded numerical measurements are preserved; "
        "archive paths, full source manifests, and execution context documents are omitted. Published-data hashes "
        "and original unchanged-input hashes are identified separately. Native checkpoints and full logs remain "
        "in the experiment archive."
        if public
        else "These downloads contain the original input bytes unchanged. SHA256 hashes identify each measured source; "
        "the report inventory and renderer hash describe this derived presentation. Native logs and checkpoints "
        "remain in the recorded artifact directories."
    )
    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Recorded Cartpole FlashSAC learning and runtime comparison in Torch and Warp.">
<title>Cartpole FlashSAC: learning and runtime</title><style>{CSS}</style></head><body><main>
<header><div class="eyebrow">RoboLearn · Isaac Lab · Recorded implementation comparison</div>
<h1>Cartpole FlashSAC: learning and runtime in Torch and Warp</h1>
<p class="subtitle">The authors’ architecture across eager Torch, compiled Torch, compiled mixed precision,
and captured Warp FP32, evaluated on the same Cartpole MDP.</p>
<div class="byline">{len(runs)} recorded runs · {budget} collected transitions per run ·
final requested iteration {final_iteration}</div></header>
<nav aria-label="Report sections"><a href="#overview">Overview</a><a href="#learning">Learning curves</a>
<a href="#quality">Final evaluations</a><a href="#runtime">Runtime</a><a href="#updates">Optimizer health</a>
<a href="#methods">Methods</a><a href="#evidence">Evidence</a></nav>
<section id="overview"><h2>Comparison scope</h2><p class="abstract">This experiment asks whether the faithful
Warp FlashSAC port learns a useful policy and how its production cost compares with three Torch recipes.
Quality comes from common evaluations, and speed comes from measured training phases. Numerical trajectories
may differ across implementations and precisions.</p>{diagnostic_note}
{_measured_summary(runs, final_iteration, metadata)}
<p class="note">{html.escape(coverage)}<br>{html.escape(cohort)}</p>
<p>No universal speed or learning advantage follows from this cohort. Three seeds support descriptive means
and observed ranges, not a statistical significance claim.</p></section>
<section id="learning"><h2>Learning curves</h2>
<div class="controls" aria-label="Learning-curve horizontal axis"><span>Horizontal axis</span>
<button type="button" data-axis="transitions" aria-pressed="true">Collected transitions</button>
<button type="button" data-axis="time" aria-pressed="false">Recorded training time</button></div>
<div class="chart-panel" id="curve-transitions">{
        _figure(
            figures["transitions"],
            "Six common evaluation metrics versus collected transitions",
            "Thin lines show individual learner seeds; thick lines and bands show means and observed min–max "
            "ranges at common recorded checkpoints. Evaluation return uses the native reward. Upright occupancy "
            "and wrapped angle describe balance; surviving within the cart bounds alone does not.",
            "learning-transitions",
        )
    }</div>
<div class="chart-panel" id="curve-time" hidden>{
        _figure(
            figures["time"],
            "Six common evaluation metrics versus recorded training time",
            time_caption,
            "learning-time",
        )
    }</div>
</section><section id="quality"><h2>Final common evaluations</h2>
<p>Only evaluations recorded at iteration {final_iteration} enter this table. Each cell is a mean and observed
seed range; unavailable measurements stay blank.</p>{_quality_table(runs, final_iteration)}
<p class="note">Survival and time-limit completion measure episode termination. Falling or rotating the pole
does not terminate this MDP. The native <code>joint_pos_target_l2</code> reward penalizes the squared pole angle
wrapped to [−π, π], and the angle RMS shown here uses the same wrapping. Policy observations retain raw angles;
<code>unwrapped_pole_angle_squared_mean</code> is a rotation diagnostic. Native return also includes velocity
and termination penalties.</p></section>
<section id="runtime"><h2>Runtime and startup costs</h2>
{
        _figure(
            figures["runtime"],
            "Process, startup, phase, and steady throughput measurements",
            "Bars show recipe means; dots are individual seeds and whiskers span the observed min–max range. "
            "Warm phase totals exclude the same first-use interval as steady throughput; cold totals remain "
            "in the table below. These panels overlap: prepare is part of startup, and first policy/update "
            "measurements are already inside the full loop. Do not add bars across panels. Steady throughput "
            "excludes the driver’s recorded first-use interval; it is not cold startup-inclusive throughput.",
            "runtime",
        )
    }
{_runtime_table(runs)}<p class="note">Production-process wall time, measured training loop time, and first-use
compile/capture costs answer different questions. The process does not include OSMO queueing, image preparation,
artifact retrieval, or unrelated jobs. Production process time is the training subprocess; the separate common
evaluation subprocess is retained in raw evidence. Loop totals and the learning-curve time axis use synchronized
collection-plus-update iteration wall time; collection/update phase measurements also use CUDA events.
Learn wall time includes prepare, logging,
and native checkpoint I/O; checkpoint totals also include the explicit final save. The exact measurement boundaries
are retained below. Shared GPU measurements
must not be described as isolated unless their runtime evidence explicitly establishes isolation.</p></section>
<section id="updates"><h2>Completed optimizer work and native-state health</h2>
<p>Attempted replay updates can differ from completed Adam steps when AMP skips a nonfinite gradient.
Parameter-step minima/maxima and coverage expose incomplete optimizer state. Health comes from native checkpoints,
not from training return alone.</p>{_optimizer_table(runs)}</section>
<section id="methods"><h2>Method and interpretation</h2><ul>
<li><strong>Common task:</strong> the existing Isaac Lab Cartpole MDP is shared by all recipes. Physics uses
the configured backend; observations, rewards, resets, and terminations remain Torch operations. Physics and
the Warp learner use separate graphs. This is not one fully Warp environment-and-learning graph.</li>
<li><strong>Budget:</strong> {html.escape(str(protocol.get("num_envs", "unrecorded")))} environments,
{html.escape(str(protocol.get("horizon", "unrecorded")))} vector steps per iteration, and
{final_iteration} requested iterations. Checkpoints: {html.escape(str(expected_checkpoints))}.
Warmup transitions count toward the production transition budget; evaluation transitions do not.</li>
<li><strong>Cartpole semantics:</strong> control interval 1/60 s (physics step 1/120 s, decimation 2),
300-step / 5-second time limit, and cart-bound termination at ±3 m. There is no pole-fall termination.
Native rewards integrate the control-step dt; the alive-only return upper bound is 5 over a 5-second episode.
Upright occupancy uses |wrapped angle| &lt; 0.25 rad. Evaluation measures each world’s first episode after a
common reset, and aggregates episode-normalized state metrics. Survival and time-limit completion identify
the same event here. Action saturation is the fraction of normalized actions with magnitude &gt; 0.95,
averaged over actions and first episodes; it describes control behavior without defining a quality gate.
{initialization_note}</li>
<li><strong>Policy comparison:</strong> the four recipes use the authors’ actor, twin distributional critic,
normalizations, bounded stochastic actions, and replay update schedule. Common initial parameter fixtures and
evaluation initial-state hashes are retained for inspection. Evaluations use deterministic policy actions,
{html.escape(str(protocol.get("eval_envs", "unrecorded")))} worlds, and reset seed
{html.escape(str(protocol.get("eval_seed", "unrecorded")))}. Different RNG and FP32/FP16 arithmetic can produce
different learned policies; matching every Torch update is not required. One common
{html.escape(str(protocol.get("eval_envs", "unrecorded")))}-world reset fixture provides paired evaluations,
and limits evidence for generalization to other resets or environment conditions.</li>
<li><strong>Scope:</strong> this controlled Cartpole implementation comparison is not a reproduction of the
FlashSAC paper’s robot results. Training-return windows may differ; common evaluations are the quality comparison.</li>
{transfer_note}
</ul>{_details("Complete recorded protocol", protocol)}{
        _details("Per-run recipe, timing, counters, and checkpoint health", run_details)
    }
</section>{appendix_section}<section id="evidence"><h2>{
        "Measurements and provenance" if public else "Raw evidence and provenance"
    }</h2>
<p>{evidence_description}</p>{_evidence_table(sources + measured["contexts"])}
{_details("Derived report inventory", inventory)}{source_details}{context_details}
<h3>Attribution and references</h3><ul>
<li><a href="https://arxiv.org/html/2604.04539v2">FlashSAC paper and author attribution</a> and
<a href="https://github.com/Holiday-Robot/FlashSAC">Holiday-Robot/FlashSAC author implementation</a>.
The upstream MIT notices and source revision are preserved by RoboLearn.</li>
<li><a href="https://nvidia.github.io/warp-nn/">NVIDIA WarpNN</a>,
<a href="https://github.com/isaac-sim/IsaacLab">Isaac Lab</a>, and
<a href="https://github.com/maxkra15/RoboLearn">RoboLearn experimental learner integration</a>.
Exact measured source and runtime revisions appear in the embedded metadata.</li></ul></section>
<footer>Offline report generated from recorded measurements. Figures are exported as SVG and PDF;
means and min–max ranges are descriptive. No external scripts, tracking, or network assets are required.</footer>
</main><script>
for (const button of document.querySelectorAll('[data-axis]')) {{
  button.addEventListener('click', () => {{
    for (const item of document.querySelectorAll('[data-axis]')) {{
      const selected = item === button;
      item.setAttribute('aria-pressed', String(selected));
      document.getElementById('curve-' + item.dataset.axis).hidden = !selected;
    }}
  }});
}}
</script></body></html>"""
    output.write_text(content, encoding="utf-8")
    print(f"Wrote {output} with {len(runs)} recorded runs; {coverage}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True, help="Measured comparison.json manifests.")
    parser.add_argument("--output", type=Path, required=True, help="Self-contained HTML destination.")
    parser.add_argument(
        "--context", nargs="*", type=Path, default=[], help="Optional runtime/provenance JSON evidence."
    )
    parser.add_argument("--numerical-appendix", type=Path, help="Optional recorded fixed-batch FP64 diagnostic JSON.")
    parser.add_argument(
        "--artifact-transfer",
        type=Path,
        help="Explicit artifact-prefetch receipt; only safe disclosure fields are used.",
    )
    parser.add_argument(
        "--public", action="store_true", help="Export derived measurements and hashes without archive paths."
    )
    args = parser.parse_args()
    render_report(args.input, args.output, args.context, args.numerical_appendix, args.public, args.artifact_transfer)


if __name__ == "__main__":
    main()
