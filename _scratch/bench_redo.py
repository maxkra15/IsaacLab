# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Redo/extension sweep for the scoop scaling report.

1. Repeat fixed @ 20 mm @ 128 envs (outlier check).
2. Fixed @ 20 mm env ladder 598 -> 800 -> 1024, stopping at the first OOM.
3. Sparse/dense with their NATURAL configuration (grid_padding=0, max_active_cell_count=-1;
   the task default padding=8 dilates the sparse active set massively, and the cell cap is
   ignored by sparse anyway): 15 mm at 1/8/32/128/398 envs, skipping larger counts for a grid
   after its first failure. Plus a padding A/B run (sparse 15 mm @ 48, padding 8 vs 0) to
   attribute the earlier OOMs.

Writes redo_results JSON+CSV in the same schema as bench_env_scaling.py.
"""

from __future__ import annotations

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
OUT = HERE / "reports" / "env_scaling"
OUT.mkdir(parents=True, exist_ok=True)

STEPS, WARMUP = 120, 20
_IGNORED = ("gnome-remote-desktop", "Xorg", "gnome-shell", "kwin")


def _q(query: str) -> list[str]:
    out = subprocess.run(
        ["nvidia-smi", query, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30
    )
    return [line.strip() for line in out.stdout.strip().splitlines() if line.strip()]


def gpu_used_mib() -> int:
    rows = _q("--query-gpu=memory.used")
    return int(rows[0]) if rows else -1


def wait_idle(timeout_s: int = 14400) -> int:
    deadline, clean = time.time() + timeout_s, 0
    while time.time() < deadline:
        pids = [r for r in _q("--query-compute-apps=pid,process_name") if not any(i in r for i in _IGNORED)]
        used = gpu_used_mib()
        if not pids and used < 3000:
            clean += 1
            if clean >= 2:
                return used
            time.sleep(5)
            continue
        clean = 0
        print(f"[redo] GPU busy ({pids}); waiting ...", flush=True)
        time.sleep(15)
    raise RuntimeError("GPU never idle")


class Sampler:
    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, gpu_used_mib())
            self._stop.wait(1.0)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=5)


def expected_particles(voxel: float, num_envs: int) -> int:
    spacing = voxel / 2.0
    base_radius = min(0.150 / math.tan(math.atan(0.7)), 0.14 - 2.0 * spacing)
    cone_volume = (math.pi / 3.0) * base_radius**2 * 0.150
    return max(int(cone_volume / spacing**3), 64) * num_envs


def run_one(voxel: float, grid: str, envs: int, cells: int, padding: int, graph: int, tag: str) -> dict:
    json_path = OUT / f"run_{tag}.json"
    log_path = OUT / f"run_{tag}.log"
    json_path.unlink(missing_ok=True)
    total = expected_particles(voxel, envs)
    print(f"[redo] {tag}: ~{total:,} particles, cells={cells}, padding={padding}", flush=True)
    baseline = wait_idle()
    cmd = [
        str(REPO / "scoop_run.sh"), "-p", str(HERE / "bench_voxel_grid_child.py"), "--headless",
        "--voxel", f"{voxel:g}", "--grid-type", grid, "--num-envs", str(envs),
        "--steps", str(STEPS), "--warmup", str(WARMUP),
        "--max-active-cells", str(cells), "--grid-padding", str(padding),
        "--cuda-graph", str(graph), "--out", str(json_path),
    ]  # fmt: skip
    t0 = time.perf_counter()
    status = "ok"
    with Sampler() as mem, open(log_path, "w") as log_f:
        try:
            proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, timeout=2400)
            if proc.returncode != 0:
                status = "crashed"
        except subprocess.TimeoutExpired:
            status = "timeout"
    rec: dict = {
        "voxel": voxel,
        "grid_type": grid,
        "num_envs": envs,
        "expected_particles": total,
        "max_active_cells": cells,
        "grid_padding": padding,
        "status": status,
        "wall_s": round(time.perf_counter() - t0, 1),
        "peak_mem_mib": mem.peak,
        "baseline_mem_mib": baseline,
    }
    if json_path.exists():
        rec.update(json.loads(json_path.read_text()))
    elif status == "ok":
        rec["status"] = "crashed"
    if rec["status"] != "ok":
        lines = log_path.read_text(errors="replace").splitlines()
        sig = [ln for ln in lines if any(k in ln.lower() for k in ("out of memory", "cuda error", "error"))]
        rec["error"] = (sig or lines)[-1][:300] if (sig or lines) else "no log output"
    print(
        f"[redo] -> {rec['status']} step={rec.get('step_ms', float('nan')):.1f} ms"
        f" steady={rec.get('cuda_used_after_run_gib', float('nan')):.1f} GiB peak={mem.peak} MiB",
        flush=True,
    )
    return rec


def main() -> None:
    results: list[dict] = []

    def auto_cells(voxel: float, envs: int) -> int:
        return max(400_000, int(0.35 * expected_particles(voxel, envs)))

    # 1. Repeat fixed 20mm @ 128.
    results.append(run_one(0.020, "fixed", 128, auto_cells(0.020, 128), 8, 1, "redo_v20mm_fixed_n128"))

    # 2. Fixed 20mm env ladder until OOM.
    for envs in (598, 800, 1024):
        rec = run_one(0.020, "fixed", envs, auto_cells(0.020, envs), 8, 1, f"redo_v20mm_fixed_n{envs}")
        results.append(rec)
        if rec["status"] != "ok":
            break

    # 3. Sparse/dense natural config (padding 0, cap -1) at 15mm, skip-after-failure.
    for grid in ("sparse", "dense"):
        for envs in (1, 8, 32, 128, 398):
            rec = run_one(0.015, grid, envs, -1, 0, 0, f"redo_v15mm_{grid}_n{envs}_p0")
            results.append(rec)
            if rec["status"] != "ok":
                break

    # 4. Padding A/B attribution: sparse 15mm @ 48 envs, padding 8 vs 0.
    results.append(run_one(0.015, "sparse", 48, -1, 8, 0, "redo_v15mm_sparse_n48_p8"))
    results.append(run_one(0.015, "sparse", 48, -1, 0, 0, "redo_v15mm_sparse_n48_p0"))

    stamp = datetime.date.today().isoformat()
    json_out = OUT / f"scoop_redo_{stamp}.json"
    json_out.write_text(json.dumps(results, indent=2))
    cols = [
        "voxel_mm", "grid_type", "num_envs", "grid_padding", "max_active_cells", "status", "step_ms",
        "env_steps_per_s", "steps_per_s_per_env", "startup_s", "steady_mem_gib", "peak_mem_mib",
        "cuda_graph_active", "particles_total", "wall_s", "error",
    ]  # fmt: skip
    with open(OUT / f"scoop_redo_{stamp}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            n = r["num_envs"]
            w.writerow({
                "voxel_mm": r["voxel"] * 1000.0,
                "grid_type": r["grid_type"],
                "num_envs": n,
                "grid_padding": r["grid_padding"],
                "max_active_cells": r["max_active_cells"],
                "status": r["status"],
                "step_ms": round(r["step_ms"], 2) if "step_ms" in r else "",
                "env_steps_per_s": round(r["env_steps_per_s"], 1) if "env_steps_per_s" in r else "",
                "steps_per_s_per_env": round(r["env_steps_per_s"] / n, 2) if "env_steps_per_s" in r else "",
                "startup_s": round(r["startup_s"], 1) if "startup_s" in r else "",
                "steady_mem_gib": round(r["cuda_used_after_run_gib"], 2) if "cuda_used_after_run_gib" in r else "",
                "peak_mem_mib": r["peak_mem_mib"],
                "cuda_graph_active": r.get("cuda_graph_active", ""),
                "particles_total": r.get("particles", r["expected_particles"]),
                "wall_s": r["wall_s"],
                "error": r.get("error", ""),
            })  # fmt: skip
    print(f"[redo] results -> {json_out}", flush=True)


if __name__ == "__main__":
    main()
