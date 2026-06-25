# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env-count scaling sweep for the coupled waterhose task (proxy vs ADMM coupling).

Runs each (solver, num_envs) config in its own subprocess (``bench_child.py``), samples
nvidia-smi for peak GPU memory, and writes a stable, reusable dataform: a results JSON
(full records) + a flat CSV. No PDF is produced; the numbers feed the LaTeX report.
Newton/Warp allocate outside torch's caching allocator, so GPU memory is read from
nvidia-smi, not ``torch.cuda``.

    ./isaaclab.sh -p _scratch/reports/waterhose_scaling/bench_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

SOLVERS = [
    ("proxy", "Isaac-Waterhose-Coupled-v0"),
    ("admm", "Isaac-Waterhose-Admm-v0"),
]

parser = argparse.ArgumentParser(description="Waterhose env-count scaling sweep.")
parser.add_argument("--env-counts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
parser.add_argument("--solvers", type=str, nargs="+", default=[s[0] for s in SOLVERS])
parser.add_argument("--warmup", type=int, default=50)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--run-timeout", type=int, default=1200)
parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES value (physical GPU index).")
parser.add_argument("--out-dir", type=str, default=str(HERE))
args = parser.parse_args()

_TASK_BY_SOLVER = dict(SOLVERS)


def gpu_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "-i", args.gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30,
    )
    rows = [ln.strip() for ln in out.stdout.strip().splitlines() if ln.strip()]
    return int(rows[0]) if rows else -1


def wait_for_idle_gpu(timeout_s: int = 180, idle_mib: int = 1500) -> int:
    """Block until the GPU frees down to ``idle_mib`` (2 clean samples), so the next run's baseline and
    throughput are not contaminated by the previous run's residual memory / thermal state. Returns the
    idle baseline MiB (or the last reading on timeout)."""
    deadline, clean, used = time.time() + timeout_s, 0, gpu_used_mib()
    while time.time() < deadline:
        used = gpu_used_mib()
        if 0 <= used < idle_mib:
            clean += 1
            if clean >= 2:
                return used
        else:
            clean = 0
        time.sleep(2.0)
    return used


class PeakMemSampler:
    def __init__(self) -> None:
        self.peak_mib = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak_mib = max(self.peak_mib, gpu_used_mib())
            self._stop.wait(0.5)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=5)


CSV_COLUMNS = [
    "solver", "num_envs", "status", "step_ms", "env_steps_per_s", "steps_per_s_per_env",
    "startup_s", "peak_mem_mib", "baseline_mem_mib", "run_mem_gib", "cuda_graph_active",
    "timed_steps", "warmup_steps", "wall_s", "error",
]  # fmt: skip


def main() -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    configs = [(s, n) for s in args.solvers for n in sorted(args.env_counts)]
    child_env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu}

    for i, (solver, envs) in enumerate(configs):
        task = _TASK_BY_SOLVER[solver]
        tag = f"{solver}_n{envs}"
        json_path = out_dir / f"run_{tag}.json"
        log_path = out_dir / f"run_{tag}.log"
        json_path.unlink(missing_ok=True)

        print(f"[wh-scal] ({i + 1}/{len(configs)}) {tag}: task={task}", flush=True)
        baseline = wait_for_idle_gpu()

        cmd = [
            str(REPO / "isaaclab.sh"), "-p", str(HERE / "bench_child.py"),
            "--task", task, "--num_envs", str(envs),
            "--warmup", str(args.warmup), "--steps", str(args.steps),
            "--headless", "--visualizer", "none", "--out", str(json_path),
        ]  # fmt: skip
        t0 = time.perf_counter()
        status = "ok"
        with PeakMemSampler() as mem, open(log_path, "w") as log_f:
            try:
                proc = subprocess.run(
                    cmd, stdout=log_f, stderr=subprocess.STDOUT, timeout=args.run_timeout, env=child_env, cwd=str(REPO)
                )
                if proc.returncode != 0:
                    status = "crashed"
            except subprocess.TimeoutExpired:
                status = "timeout"
        wall = time.perf_counter() - t0

        rec: dict = {
            "solver": solver,
            "task": task,
            "num_envs": envs,
            "status": status,
            "wall_s": round(wall, 1),
            "peak_mem_mib": mem.peak_mib,
            "baseline_mem_mib": baseline,
        }
        if json_path.exists():
            rec.update(json.loads(json_path.read_text()))  # child's status/metrics win when present
        elif status == "ok":
            rec["status"] = "crashed"
        if rec["status"] not in ("ok", "nan"):
            lines = log_path.read_text(errors="replace").splitlines()
            sig = [ln for ln in lines if any(k in ln.lower() for k in ("out of memory", "cuda error", "error", "exceeded"))]
            rec.setdefault("error", (sig or lines)[-1][:300] if (sig or lines) else "no log output")
        results.append(rec)
        print(
            f"[wh-scal] -> {rec['status']} (wall {wall:.0f}s, step {rec.get('step_ms', float('nan')):.1f} ms, "
            f"{rec.get('env_steps_per_s', float('nan')):.0f} env-steps/s, peak {mem.peak_mib} MiB)",
            flush=True,
        )

    stamp = datetime.date.today().isoformat()
    json_out = out_dir / f"waterhose_env_scaling_{stamp}.json"
    json_out.write_text(json.dumps(results, indent=2))

    csv_out = out_dir / f"waterhose_env_scaling_{stamp}.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in results:
            w.writerow({
                "solver": r["solver"],
                "num_envs": r["num_envs"],
                "status": r["status"],
                "step_ms": round(r["step_ms"], 2) if "step_ms" in r else "",
                "env_steps_per_s": round(r["env_steps_per_s"], 1) if "env_steps_per_s" in r else "",
                "steps_per_s_per_env": round(r["steps_per_s_per_env"], 2) if "steps_per_s_per_env" in r else "",
                "startup_s": round(r["startup_s"], 1) if "startup_s" in r else "",
                "peak_mem_mib": r["peak_mem_mib"],
                "baseline_mem_mib": r["baseline_mem_mib"],
                "run_mem_gib": round((r["peak_mem_mib"] - r["baseline_mem_mib"]) / 1024, 2),
                "cuda_graph_active": r.get("cuda_graph_active", ""),
                "timed_steps": args.steps,
                "warmup_steps": args.warmup,
                "wall_s": r["wall_s"],
                "error": r.get("error", ""),
            })  # fmt: skip
    print(f"[wh-scal] results -> {json_out}\n[wh-scal] results -> {csv_out}", flush=True)


if __name__ == "__main__":
    main()
