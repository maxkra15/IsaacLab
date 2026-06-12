# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Env-count x voxel-size scaling sweep for the Franka scoop MPM env (fixed grid).

Runs each (voxel, num_envs) config in its own subprocess (bench_voxel_grid_child.py),
gates every run on an idle GPU, samples nvidia-smi for peak memory, and writes the
results to a stable, reusable dataform: results JSON (full records) + CSV (flat table).
No PDF is produced; the numbers feed the LaTeX report.

    ./env_isaaclab/bin/python _scratch/bench_env_scaling.py
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

parser = argparse.ArgumentParser(description="Scoop env-count x voxel scaling sweep.")
parser.add_argument("--voxels", type=float, nargs="+", default=[0.020, 0.015, 0.010])
parser.add_argument("--env-counts", type=int, nargs="+", default=[1, 8, 32, 128, 398])
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--run-timeout", type=int, default=2400)
parser.add_argument("--idle-timeout", type=int, default=14400)
parser.add_argument("--out-dir", type=str, default=str(HERE / "reports" / "env_scaling"))
args = parser.parse_args()

_IGNORED_GPU_PROCS = ("gnome-remote-desktop", "Xorg", "gnome-shell", "kwin")


def _q(query: str) -> list[str]:
    out = subprocess.run(
        ["nvidia-smi", query, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30
    )
    return [line.strip() for line in out.stdout.strip().splitlines() if line.strip()]


def gpu_used_mib() -> int:
    rows = _q("--query-gpu=memory.used")
    return int(rows[0]) if rows else -1


def gpu_busy_pids() -> list[str]:
    rows = _q("--query-compute-apps=pid,process_name")
    return [r for r in rows if not any(i in r for i in _IGNORED_GPU_PROCS)]


def wait_for_idle_gpu(timeout_s: int, busy_mib: int = 3000) -> int:
    """Block until no compute jobs (2 clean samples 5 s apart); returns the idle baseline MiB."""
    deadline, clean = time.time() + timeout_s, 0
    while time.time() < deadline:
        pids = gpu_busy_pids()
        used = gpu_used_mib()
        if not pids and used < busy_mib:
            clean += 1
            if clean >= 2:
                return used
            time.sleep(5)
            continue
        clean = 0
        print(f"[scal] GPU busy (used={used} MiB, jobs={pids}); waiting ...", flush=True)
        time.sleep(15)
    raise RuntimeError(f"GPU did not become idle within {timeout_s} s; aborting sweep.")


class PeakMemSampler:
    def __init__(self) -> None:
        self.peak_mib = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak_mib = max(self.peak_mib, gpu_used_mib())
            self._stop.wait(1.0)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=5)


def expected_particles(voxel: float, num_envs: int) -> int:
    """Mirror media_spawning._pile_points count math (default pile geometry)."""
    spacing = voxel / 2.0  # particles_per_cell = 2
    base_radius = min(0.150 / math.tan(math.atan(0.7)), 0.14 - 2.0 * spacing)
    cone_volume = (math.pi / 3.0) * base_radius**2 * 0.150
    return max(int(cone_volume / spacing**3), 64) * num_envs


CSV_COLUMNS = [
    "voxel_mm", "grid_type", "num_envs", "particles_total", "particles_per_env", "max_active_cells",
    "status", "step_ms", "env_steps_per_s", "steps_per_s_per_env", "startup_s",
    "peak_mem_mib", "baseline_mem_mib", "peak_mem_delta_gib", "cuda_graph_active",
    "timed_steps", "warmup_steps", "wall_s", "error",
]  # fmt: skip


def main() -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    # Cheap -> expensive: coarse voxels first, small env counts first.
    configs = [(v, n) for v in sorted(args.voxels, reverse=True) for n in sorted(args.env_counts)]
    for i, (voxel, envs) in enumerate(configs):
        total = expected_particles(voxel, envs)
        cells = max(400_000, int(0.35 * total))
        tag = f"v{voxel * 1000:g}mm_n{envs}"
        json_path = out_dir / f"run_{tag}.json"
        log_path = out_dir / f"run_{tag}.log"
        json_path.unlink(missing_ok=True)

        print(f"[scal] ({i + 1}/{len(configs)}) {tag}: ~{total:,} particles, cells={cells:,}", flush=True)
        baseline = wait_for_idle_gpu(args.idle_timeout)

        cmd = [
            str(REPO / "scoop_run.sh"), "-p", str(HERE / "bench_voxel_grid_child.py"), "--headless",
            "--voxel", f"{voxel:g}", "--grid-type", "fixed", "--num-envs", str(envs),
            "--steps", str(args.steps), "--warmup", str(args.warmup),
            "--max-active-cells", str(cells), "--cuda-graph", "1", "--out", str(json_path),
        ]  # fmt: skip
        t0 = time.perf_counter()
        status = "ok"
        with PeakMemSampler() as mem, open(log_path, "w") as log_f:
            try:
                proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, timeout=args.run_timeout)
                if proc.returncode != 0:
                    status = "crashed"
            except subprocess.TimeoutExpired:
                status = "timeout"
        wall = time.perf_counter() - t0

        rec: dict = {
            "voxel": voxel,
            "grid_type": "fixed",
            "num_envs": envs,
            "expected_particles": total,
            "max_active_cells": cells,
            "status": status,
            "wall_s": round(wall, 1),
            "peak_mem_mib": mem.peak_mib,
            "baseline_mem_mib": baseline,
        }
        if json_path.exists():
            rec.update(json.loads(json_path.read_text()))
        elif status == "ok":
            rec["status"] = "crashed"
        if rec["status"] != "ok":
            lines = log_path.read_text(errors="replace").splitlines()
            sig = [ln for ln in lines if any(k in ln.lower() for k in ("out of memory", "cuda error", "error", "exceeded"))]
            rec["error"] = (sig or lines)[-1][:300] if (sig or lines) else "no log output"
        results.append(rec)
        print(
            f"[scal] -> {rec['status']} (wall {wall:.0f}s, step {rec.get('step_ms', float('nan')):.1f} ms,"
            f" peak {mem.peak_mib} MiB)",
            flush=True,
        )

    stamp = datetime.date.today().isoformat()
    json_out = out_dir / f"scoop_env_scaling_{stamp}.json"
    json_out.write_text(json.dumps(results, indent=2))

    csv_out = out_dir / f"scoop_env_scaling_{stamp}.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in results:
            n = r["num_envs"]
            w.writerow({
                "voxel_mm": r["voxel"] * 1000.0,
                "grid_type": r["grid_type"],
                "num_envs": n,
                "particles_total": r.get("particles", r["expected_particles"]),
                "particles_per_env": r.get("particles_per_env", r["expected_particles"] // max(n, 1)),
                "max_active_cells": r["max_active_cells"],
                "status": r["status"],
                "step_ms": round(r["step_ms"], 2) if "step_ms" in r else "",
                "env_steps_per_s": round(r["env_steps_per_s"], 1) if "env_steps_per_s" in r else "",
                "steps_per_s_per_env": round(r["env_steps_per_s"] / n, 2) if "env_steps_per_s" in r else "",
                "startup_s": round(r["startup_s"], 1) if "startup_s" in r else "",
                "peak_mem_mib": r["peak_mem_mib"],
                "baseline_mem_mib": r["baseline_mem_mib"],
                "peak_mem_delta_gib": round((r["peak_mem_mib"] - r["baseline_mem_mib"]) / 1024, 2),
                "cuda_graph_active": r.get("cuda_graph_active", ""),
                "timed_steps": args.steps,
                "warmup_steps": args.warmup,
                "wall_s": r["wall_s"],
                "error": r.get("error", ""),
            })  # fmt: skip
    print(f"[scal] results -> {json_out}\n[scal] results -> {csv_out}", flush=True)


if __name__ == "__main__":
    main()
