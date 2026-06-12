# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Aux benchmark runs: sparse/dense grid viability vs env count and max_active_cell_count.

Standalone companion to bench_voxel_grid.py (which parses argv at import, so the few
helpers are duplicated here). Writes one merged JSON consumed via --aux-results.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "reports" / "aux"
OUT.mkdir(parents=True, exist_ok=True)

VOXEL = 0.015
STEPS, WARMUP = 60, 10
_IGNORED = ("gnome-remote-desktop", "Xorg", "gnome-shell", "kwin")

# (grid_type, num_envs, max_active_cells) — cap experiments at 48 envs, env-count probe below.
CONFIGS = [
    ("sparse", 48, 1_088_740),
    ("sparse", 48, -1),
    ("dense", 48, -1),
    ("sparse", 24, 400_000),
    ("dense", 24, 400_000),
    ("sparse", 12, 400_000),
    ("dense", 12, 400_000),
]


def _q(query: str) -> list[str]:
    out = subprocess.run(
        ["nvidia-smi", query, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30
    )
    return [line.strip() for line in out.stdout.strip().splitlines() if line.strip()]


def used_mib() -> int:
    rows = _q("--query-gpu=memory.used")
    return int(rows[0]) if rows else -1


def wait_idle(timeout_s: int = 3600) -> None:
    deadline, clean = time.time() + timeout_s, 0
    while time.time() < deadline:
        pids = [r for r in _q("--query-compute-apps=pid,process_name") if not any(i in r for i in _IGNORED)]
        if not pids and used_mib() < 3000:
            clean += 1
            if clean >= 2:
                return
            time.sleep(5)
            continue
        clean = 0
        print(f"[aux] GPU busy: {pids}; waiting ...", flush=True)
        time.sleep(15)
    raise RuntimeError("GPU never idle")


class Sampler:
    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, used_mib())
            self._stop.wait(1.0)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=5)


def main() -> None:
    results = []
    for grid, envs, cells in CONFIGS:
        tag = f"aux_{grid}_n{envs}_c{cells}"
        json_path = OUT / f"{tag}.json"
        log_path = OUT / f"{tag}.log"
        json_path.unlink(missing_ok=True)
        print(f"[aux] {tag}", flush=True)
        wait_idle()
        cmd = [
            str(REPO / "scoop_run.sh"), "-p", str(HERE / "bench_voxel_grid_child.py"), "--headless",
            "--voxel", f"{VOXEL:g}", "--grid-type", grid, "--num-envs", str(envs),
            "--steps", str(STEPS), "--warmup", str(WARMUP),
            "--max-active-cells", str(cells), "--cuda-graph", "0", "--out", str(json_path),
        ]  # fmt: skip
        status = "ok"
        with Sampler() as mem, open(log_path, "w") as log_f:
            try:
                proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, timeout=1500)
                if proc.returncode != 0:
                    status = "crashed"
            except subprocess.TimeoutExpired:
                status = "timeout"
        rec = {
            "voxel": VOXEL,
            "grid_type": grid,
            "num_envs": envs,
            "max_active_cells": cells,
            "status": status,
            "peak_mem_mib": mem.peak,
        }
        if json_path.exists():
            rec.update(json.loads(json_path.read_text()))
        elif status == "ok":
            rec["status"] = "crashed"
        results.append(rec)
        print(f"[aux] -> {rec['status']} peak={mem.peak} MiB step_ms={rec.get('step_ms')}", flush=True)

    out = OUT / "aux_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[aux] results -> {out}", flush=True)


if __name__ == "__main__":
    main()
