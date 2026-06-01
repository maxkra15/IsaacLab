#!/usr/bin/env python3
"""Generate the waterhose teleop scaling Markdown and PDF report."""

from __future__ import annotations

import json
import math
import subprocess
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


REPORT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = REPORT_DIR / "results.jsonl"
PDF_PATH = REPORT_DIR / "waterhose_teleop_scaling_report.pdf"
MD_PATH = REPORT_DIR / "waterhose_teleop_scaling_report.md"


def load_results() -> list[dict]:
    latest: dict[int, dict] = {}
    for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        latest[int(item["num_envs"])] = item
    return [latest[n] for n in sorted(latest)]


def git_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPORT_DIR.parents[1], text=True).strip()
    except Exception:
        return "unknown"


def nvidia_smi() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def measured_rows(results: list[dict]) -> list[dict]:
    return [r for r in results if r.get("profile") and r.get("returncode") == 0 and not r.get("timed_out")]


def fit_setup_seconds_per_env(rows: list[dict]) -> float:
    # Use the larger measured points where fixed startup overhead is less dominant.
    tail = [r for r in rows if r["num_envs"] >= 8]
    if not tail:
        tail = rows
    xs = [r["num_envs"] for r in tail]
    ys = [r["profile"]["setup_time_s"] for r in tail]
    if len(xs) < 2:
        return ys[0] / xs[0]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return y_mean / x_mean
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom


def write_markdown(results: list[dict], commit: str, branch: str, gpu_info: str) -> None:
    rows = measured_rows(results)
    setup_slope = fit_setup_seconds_per_env(rows)
    table_lines = [
        "| num_envs | status | setup (s) | rollout (s) | live Hz | env steps/s | step time (ms) | GPU delta (GiB) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    all_envs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    by_env = {r["num_envs"]: r for r in results}
    for n in all_envs:
        r = by_env.get(n)
        if r is None:
            table_lines.append(f"| {n:,} | not run after 64-env timeout | - | - | - | - | - | - |")
            continue
        if r.get("profile"):
            p = r["profile"]
            gpu = r.get("gpu_memory_delta_mib")
            gpu_gib = gpu / 1024.0 if gpu is not None else math.nan
            table_lines.append(
                f"| {n:,} | completed | {p['setup_time_s']:.1f} | {p['rollout_time_s']:.2f} | "
                f"{p['control_hz']:.1f} | {p['env_steps_per_s']:.1f} | {p['step_time_ms']:.1f} | {gpu_gib:.2f} |"
            )
        elif r.get("timed_out"):
            gpu = r.get("gpu_memory_delta_mib")
            gpu_gib = gpu / 1024.0 if gpu is not None else math.nan
            table_lines.append(
                f"| {n:,} | timed out at {r['elapsed_s']:.0f}s | - | - | - | - | - | {gpu_gib:.2f} |"
            )
        else:
            table_lines.append(f"| {n:,} | failed | - | - | - | - | - | - |")

    best_hz = max(rows, key=lambda r: r["profile"]["control_hz"])
    best_env_sps = max(rows, key=lambda r: r["profile"]["env_steps_per_s"])

    md = f"""# Waterhose Teleop Scaling Report

NVIDIA | Isaac Lab internal technical report

Date: {date.today().isoformat()}

Branch: `{branch}`
Commit: `{commit}`
Task: `Isaac-Waterhose-Robot-Demo-v0`
Mode: built-in teleop path, idle SpaceMouse command broadcast to all environments
GPU: `{gpu_info.splitlines()[0] if gpu_info else 'unknown'}`

## Executive Summary

The current default waterhose task does not scale like a batched Isaac Lab environment. It creates independent split Newton runtimes per environment. In teleop mode, aggregate env-step throughput stays nearly flat at about 73 env steps/s from 2 to 32 envs, while live control rate falls almost exactly inversely with `num_envs`.

For live teleop, 1 env is the only comfortable operating point on this machine: {best_hz['profile']['control_hz']:.1f} Hz. Two envs is borderline at {next(r for r in rows if r['num_envs'] == 2)['profile']['control_hz']:.1f} Hz. By 8 envs, teleop drops to {next(r for r in rows if r['num_envs'] == 8)['profile']['control_hz']:.1f} Hz. 32 envs completes, but at only {next(r for r in rows if r['num_envs'] == 32)['profile']['control_hz']:.1f} Hz.

The 64-env run did not finish within the 300 s timeout. Larger requested points through 8192 envs were not run after that timeout because the measured setup slope is roughly {setup_slope:.2f} s per additional environment at the larger measured points, and peak GPU memory was already about {next(r for r in results if r['num_envs'] == 64)['gpu_memory_delta_mib'] / 1024.0:.1f} GiB before the 64-env run completed.

## Methodology

Each point launched:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \\
  --task Isaac-Waterhose-Robot-Demo-v0 \\
  --mode teleop --teleop_device spacemouse \\
  --vis none --num_envs N --max_steps 60 --profile --device cuda:0
```

The SpaceMouse device was initialized normally. The benchmark used idle input, so the same zero/idle teleop command was broadcast to all environments. This measures the teleop control path and environment stepping throughput, not human reaction quality or rendering latency.

Metrics:

- Live Hz: manager steps per wall-clock second, equivalent to teleop control update rate.
- Env steps/s: `live Hz * num_envs`, useful for aggregate simulation throughput.
- Setup time: time before rollout starts.
- GPU delta: peak `nvidia-smi` memory above the pre-run baseline.

## Scaling Table

{chr(10).join(table_lines)}

## Interpretation

The measured scaling is serial-runtime scaling: total work increases almost linearly with `num_envs`, but the implementation does not recover that cost through batching. This is why env steps/s stays flat instead of rising, and why live Hz degrades from {best_hz['profile']['control_hz']:.1f} Hz at 1 env to {next(r for r in rows if r['num_envs'] == 32)['profile']['control_hz']:.1f} Hz at 32 envs.

Recommended operating points:

| Use case | Recommended num_envs | Reason |
| --- | ---: | --- |
| Live SpaceMouse / client demo | 1 | Highest control rate and shortest startup |
| Side-by-side visual smoke | 2 | Still usable for comparison, but below 40 Hz |
| Offline robustness smoke | 4-8 | Aggregate throughput is flat, useful only to expose multi-env bugs |
| Data collection / training | Not this architecture | Needs a true batched Newton model or a different collection strategy |

## Caveat

This report profiles the stable default one-way task, not the experimental coupled-manager task. The result is dominated by the current N-independent-runtime architecture, not by SpaceMouse polling.
"""
    MD_PATH.write_text(md, encoding="utf-8")


def add_wrapped_text(fig, text: str, x: float, y: float, width_chars: int = 95, size: int = 10):
    import textwrap

    wrapped = "\n".join(textwrap.wrap(text, width=width_chars))
    fig.text(x, y, wrapped, ha="left", va="top", fontsize=size)


def write_pdf(results: list[dict], commit: str, branch: str, gpu_info: str) -> None:
    rows = measured_rows(results)
    setup_slope = fit_setup_seconds_per_env(rows)
    envs = [r["num_envs"] for r in rows]
    live_hz = [r["profile"]["control_hz"] for r in rows]
    env_sps = [r["profile"]["env_steps_per_s"] for r in rows]
    setup = [r["profile"]["setup_time_s"] for r in rows]
    step_ms = [r["profile"]["step_time_ms"] for r in rows]
    gpu_gib = [(r.get("gpu_memory_delta_mib") or 0) / 1024.0 for r in rows]
    timeout64 = next((r for r in results if r["num_envs"] == 64), None)

    with PdfPages(PDF_PATH) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.06, 0.92, "Waterhose Teleop Scaling", fontsize=24, weight="bold")
        fig.text(0.06, 0.875, "Isaac-Waterhose-Robot-Demo-v0 | one-way split Newton runtime", fontsize=13)
        fig.text(0.06, 0.835, f"Date: {date.today().isoformat()}   Branch: {branch}   Commit: {commit}", fontsize=10)
        fig.text(0.06, 0.805, f"GPU: {gpu_info.splitlines()[0] if gpu_info else 'unknown'}", fontsize=10)
        summary = (
            "Executive summary: the current default waterhose task does not scale as a batched Isaac Lab "
            "environment. Aggregate env-step throughput remains nearly flat at about 73 env steps/s from "
            "2 to 32 envs, while live teleop control rate falls inversely with num_envs. 64 envs did not "
            "complete within the 300 s timeout; higher requested points through 8192 were not run."
        )
        add_wrapped_text(fig, summary, 0.06, 0.73, width_chars=105, size=12)
        methodology = (
            "Methodology: each point launched run_robot_demo.py with --mode teleop, --teleop_device spacemouse, "
            "--vis none, --max_steps 60, and --profile. The SpaceMouse device was initialized normally; input "
            "was idle and broadcast to all environments. Live Hz is manager steps/s. Env steps/s is live Hz "
            "multiplied by num_envs."
        )
        add_wrapped_text(fig, methodology, 0.06, 0.61, width_chars=105, size=10)
        recommendations = (
            "Recommendation: use 1 env for live client teleop. 2 envs is borderline. 4-8 envs can be used for "
            "offline smoke checks, but they do not increase aggregate throughput. This architecture is not "
            "appropriate for high-throughput teleop data collection or training."
        )
        add_wrapped_text(fig, recommendations, 0.06, 0.50, width_chars=105, size=11)
        table_data = []
        for r in results:
            if r["num_envs"] > 64:
                continue
            if r.get("profile"):
                p = r["profile"]
                table_data.append([
                    f"{r['num_envs']:,}",
                    "completed",
                    f"{p['setup_time_s']:.1f}",
                    f"{p['control_hz']:.1f}",
                    f"{p['env_steps_per_s']:.1f}",
                    f"{(r.get('gpu_memory_delta_mib') or 0) / 1024.0:.2f}",
                ])
            else:
                table_data.append([
                    f"{r['num_envs']:,}",
                    f"timeout {r['elapsed_s']:.0f}s",
                    "-",
                    "-",
                    "-",
                    f"{(r.get('gpu_memory_delta_mib') or 0) / 1024.0:.2f}",
                ])
        ax = fig.add_axes([0.06, 0.08, 0.88, 0.32])
        ax.axis("off")
        table = ax.table(
            cellText=table_data,
            colLabels=["envs", "status", "setup s", "live Hz", "env steps/s", "GPU GiB"],
            loc="center",
            cellLoc="right",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.35)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        ax = axes[0, 0]
        ax.plot(envs, live_hz, marker="o")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title("Live Teleop Control Rate")
        ax.set_xlabel("Number of envs")
        ax.set_ylabel("Manager steps/s (Hz)")
        ax.grid(True, which="both", alpha=0.3)
        ax.annotate("1 env: best live teleop", xy=(envs[0], live_hz[0]), xytext=(1.5, live_hz[0] * 0.8), arrowprops={"arrowstyle": "->"})

        ax = axes[0, 1]
        ax.plot(envs, env_sps, marker="o", color="tab:green")
        ax.set_xscale("log", base=2)
        ax.set_title("Aggregate Env Throughput")
        ax.set_xlabel("Number of envs")
        ax.set_ylabel("Env steps/s")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(env_sps) * 1.25)
        ax.text(envs[1], max(env_sps) * 1.1, "Flat throughput: no batching gain", fontsize=9)

        ax = axes[1, 0]
        ax.plot(envs, setup, marker="o", color="tab:orange")
        ax.set_xscale("log", base=2)
        ax.set_title("Startup Cost")
        ax.set_xlabel("Number of envs")
        ax.set_ylabel("Setup time (s)")
        ax.grid(True, alpha=0.3)
        if timeout64 is not None:
            ax.scatter([64], [timeout64["elapsed_s"]], color="red", marker="x", s=80, label="64 timeout")
            ax.legend()
        ax.text(2, max(setup) * 0.82, f"large-N slope ~{setup_slope:.2f} s/env", fontsize=9)

        ax = axes[1, 1]
        ax.plot(envs, gpu_gib, marker="o", color="tab:red")
        ax.set_xscale("log", base=2)
        ax.set_title("Peak GPU Memory Delta")
        ax.set_xlabel("Number of envs")
        ax.set_ylabel("GiB")
        ax.grid(True, alpha=0.3)
        if timeout64 is not None:
            ax.scatter([64], [(timeout64.get("gpu_memory_delta_mib") or 0) / 1024.0], color="red", marker="x", s=80)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.plot(envs, step_ms, marker="o", label="measured step time")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title("Teleop Frame Time")
        ax.set_xlabel("Number of envs")
        ax.set_ylabel("ms per manager step")
        ax.grid(True, which="both", alpha=0.3)
        ax.axhline(16.7, color="tab:green", linestyle="--", label="60 Hz")
        ax.axhline(33.3, color="tab:orange", linestyle="--", label="30 Hz")
        ax.axhline(100.0, color="tab:red", linestyle="--", label="10 Hz")
        ax.legend()
        ax.text(
            1.1,
            max(step_ms) * 0.45,
            "The control frame time roughly doubles with each env doubling.\n"
            "That is the signature of independent serial runtimes, not batched simulation.",
            fontsize=10,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    results = load_results()
    commit = git_text(["git", "rev-parse", "--short", "HEAD"])
    branch = git_text(["git", "branch", "--show-current"])
    gpu_info = nvidia_smi()
    write_markdown(results, commit, branch, gpu_info)
    write_pdf(results, commit, branch, gpu_info)
    print(PDF_PATH)
    print(MD_PATH)


if __name__ == "__main__":
    main()
