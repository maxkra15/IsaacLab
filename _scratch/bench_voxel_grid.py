# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Voxel-size x grid-type benchmark sweep for the Franka scoop MPM env + PDF report.

Runs each (voxel_size, grid_type) config in its own subprocess (bench_voxel_grid_child.py)
so an OOM/crash cannot kill the sweep, gates every run on the GPU being idle, samples
nvidia-smi for peak memory during each run, and renders a multi-page PDF report.

Run with the plain venv python (no sim in this process):
    ./env_isaaclab/bin/python _scratch/bench_voxel_grid.py
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

parser = argparse.ArgumentParser(description="Scoop voxel/grid benchmark sweep + PDF report.")
parser.add_argument("--voxels", type=float, nargs="+", default=[0.0225, 0.015, 0.010, 0.007])
parser.add_argument("--grids", type=str, nargs="+", default=["fixed", "sparse", "dense"])
parser.add_argument("--num-envs", type=int, default=48)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--run-timeout", type=int, default=1800, help="Per-run timeout [s].")
parser.add_argument("--idle-timeout", type=int, default=1800, help="Max wait for an idle GPU [s].")
parser.add_argument("--out-dir", type=str, default=str(HERE / "reports"))
parser.add_argument("--report-only", type=str, default=None, help="Re-render the PDF from an existing results JSON.")
parser.add_argument(
    "--aux-results",
    type=str,
    nargs="*",
    default=[],
    help="Extra results JSONs (e.g. sparse/dense env-count probes) rendered on a dedicated page.",
)
args = parser.parse_args()

SUBSTEPS_PER_ENV_STEP = 4  # decimation 2 x num_substeps 2 (cfg defaults)


# --------------------------------------------------------------------------- gpu helpers
def _nvidia_query(query: str, *extra: str) -> list[str]:
    out = subprocess.run(
        ["nvidia-smi", query, "--format=csv,noheader,nounits", *extra],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return [line.strip() for line in out.stdout.strip().splitlines() if line.strip()]


# Desktop daemons that permanently hold a (small) CUDA context on this workstation;
# they do not count as "jobs" for the idle gate.
_IGNORED_GPU_PROCS = ("gnome-remote-desktop", "Xorg", "gnome-shell", "kwin")


def gpu_compute_pids() -> list[str]:
    rows = _nvidia_query("--query-compute-apps=pid,process_name")
    return [r for r in rows if not any(ign in r for ign in _IGNORED_GPU_PROCS)]


def gpu_used_mib() -> int:
    rows = _nvidia_query("--query-gpu=memory.used")
    return int(rows[0]) if rows else -1


def gpu_info() -> dict:
    rows = _nvidia_query("--query-gpu=name,memory.total,driver_version")
    name, total, driver = (rows[0].split(", ") + ["?", "?", "?"])[:3]
    return {"name": name, "total_mib": int(total), "driver": driver}


def wait_for_idle_gpu(timeout_s: int, busy_mib: int = 3000) -> None:
    """Block until the GPU has no compute jobs (2 consecutive clean samples 5 s apart)."""
    deadline = time.time() + timeout_s
    clean = 0
    while time.time() < deadline:
        pids = gpu_compute_pids()
        used = gpu_used_mib()
        if not pids and used < busy_mib:
            clean += 1
            if clean >= 2:
                return
            time.sleep(5)
            continue
        clean = 0
        print(f"[sweep] GPU busy (used={used} MiB, compute apps={pids}); waiting ...", flush=True)
        time.sleep(15)
    raise RuntimeError(f"GPU did not become idle within {timeout_s} s; aborting sweep.")


class PeakMemSampler:
    """Samples global nvidia-smi used memory every second while a run is active."""

    def __init__(self) -> None:
        self.peak_mib = 0
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            used = gpu_used_mib()
            if used > 0:
                self.samples.append(used)
                self.peak_mib = max(self.peak_mib, used)
            self._stop.wait(1.0)

    def __enter__(self) -> "PeakMemSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


# --------------------------------------------------------------------------- sweep
def expected_particles(voxel: float, num_envs: int) -> int:
    """Mirror media_spawning._pile_points count math (pile geometry at default cfg)."""
    spacing = voxel / 2.0  # particles_per_cell = 2
    angle = math.atan(0.7)  # media_material.friction
    height = 0.150  # pile_height
    base_radius = min(height / math.tan(angle), 0.14 - 2.0 * spacing)  # container_inner_half[0]
    cone_volume = (math.pi / 3.0) * base_radius**2 * height
    return max(int(cone_volume / spacing**3), 64) * num_envs


def run_sweep() -> list[dict]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    # Cheap -> expensive: big voxels first so failures cluster at the end.
    configs = [(v, g) for v in sorted(args.voxels, reverse=True) for g in args.grids]
    for i, (voxel, grid) in enumerate(configs):
        total_particles = expected_particles(voxel, args.num_envs)
        cells = max(400_000, int(0.35 * total_particles))
        cuda_graph = 1 if grid == "fixed" else 0  # only the fixed grid is graph-capturable
        tag = f"v{voxel:g}_{grid}"
        json_path = out_dir / f"run_{tag}.json"
        log_path = out_dir / f"run_{tag}.log"
        json_path.unlink(missing_ok=True)

        print(f"[sweep] ({i + 1}/{len(configs)}) {tag}: ~{total_particles} particles, cells={cells}", flush=True)
        wait_for_idle_gpu(args.idle_timeout)

        cmd = [
            str(REPO / "scoop_run.sh"),
            "-p",
            str(HERE / "bench_voxel_grid_child.py"),
            "--headless",
            "--voxel",
            f"{voxel:g}",
            "--grid-type",
            grid,
            "--num-envs",
            str(args.num_envs),
            "--steps",
            str(args.steps),
            "--warmup",
            str(args.warmup),
            "--max-active-cells",
            str(cells),
            "--cuda-graph",
            str(cuda_graph),
            "--out",
            str(json_path),
        ]
        t0 = time.perf_counter()
        status, error = "ok", ""
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
            "grid_type": grid,
            "num_envs": args.num_envs,
            "expected_particles": total_particles,
            "max_active_cells": cells,
            "cuda_graph_requested": bool(cuda_graph),
            "status": status,
            "wall_s": wall,
            "peak_mem_mib": mem.peak_mib,
        }
        if json_path.exists():
            rec.update(json.loads(json_path.read_text()))
        elif status == "ok":
            rec["status"] = "crashed"
        if rec["status"] != "ok":
            tail = log_path.read_text(errors="replace").splitlines()
            sig = [
                line
                for line in tail
                if any(k in line.lower() for k in ("out of memory", "cuda error", "error", "exceeded", "abort"))
            ]
            rec["error"] = (sig or tail)[-1][:300] if (sig or tail) else "no log output"
        results.append(rec)
        print(f"[sweep] -> {rec['status']} (wall {wall:.0f}s, peak {mem.peak_mib} MiB)", flush=True)

    stamp = datetime.date.today().isoformat()
    results_path = out_dir / f"scoop_voxel_grid_results_{stamp}.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"[sweep] results -> {results_path}", flush=True)
    return results


# --------------------------------------------------------------------------- report
GRID_COLORS = {"fixed": "#2a7fcb", "sparse": "#e08a2e", "dense": "#3a9a55"}


def render_report(results: list[dict], out_pdf: Path, aux: list[dict] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    git_rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    info = gpu_info()
    ok = [r for r in results if r["status"] == "ok"]
    voxels = sorted({r["voxel"] for r in results}, reverse=True)
    grids = [g for g in ("fixed", "sparse", "dense") if any(r["grid_type"] == g for r in results)]

    def by(grid: str, key: str) -> tuple[list[float], list[float]]:
        xs, ys = [], []
        for v in voxels:
            r = next((r for r in ok if r["grid_type"] == grid and r["voxel"] == v), None)
            if r is not None and key in r:
                xs.append(v * 1000.0)  # mm
                ys.append(r[key])
        return xs, ys

    with PdfPages(out_pdf) as pdf:
        # ---- page 1: title + summary table -------------------------------------------------
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.suptitle("Franka Scoop MPM Environment — Voxel Size × Grid Type Benchmark", fontsize=16, y=0.97)
        meta = (
            f"Date: {datetime.datetime.now():%Y-%m-%d %H:%M}   GPU: {info['name']} ({info['total_mib']} MiB,"
            f" driver {info['driver']})\n"
            f"Branch commit: {git_rev.stdout.strip()}   num_envs: {args.num_envs}   timed steps: {args.steps}"
            f" (after {args.warmup} warmup)   workload: zero actions\n"
            f"Per env step: decimation 2 × 2 substeps = {SUBSTEPS_PER_ENV_STEP} MPM solves, 24 iterations each."
            "   CUDA graph: requested for the fixed grid only (sparse/dense are not capturable)."
        )
        fig.text(0.06, 0.86, meta, fontsize=9, va="top")
        cols = ["voxel [mm]", "grid", "particles", "step [ms]", "env-steps/s", "peak mem [GiB]", "startup [s]", "graph", "status"]
        rows = []
        for r in sorted(results, key=lambda r: (-r["voxel"], grids.index(r["grid_type"]))):
            rows.append([
                f"{r['voxel'] * 1000:g}",
                r["grid_type"],
                f"{r.get('particles', r['expected_particles']):,}",
                f"{r['step_ms']:.1f}" if "step_ms" in r else "—",
                f"{r['env_steps_per_s']:.0f}" if "env_steps_per_s" in r else "—",
                f"{r['peak_mem_mib'] / 1024:.1f}",
                f"{r['startup_s']:.0f}" if "startup_s" in r else "—",
                {True: "on", False: "off"}.get(r.get("cuda_graph_active"), "—"),
                r["status"] + ("" if r["status"] == "ok" else " *"),
            ])
        ax = fig.add_axes([0.05, 0.08, 0.9, 0.66])
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.45)
        for j in range(len(cols)):
            table[0, j].set_facecolor("#d8e4f0")
        fails = [r for r in results if r["status"] != "ok"]
        if fails:
            txt = "\n".join(f"* {r['voxel']*1000:g} mm / {r['grid_type']}: {r['status']} — {r.get('error', '')[:160]}" for r in fails)
            fig.text(0.06, 0.05, txt, fontsize=7, va="bottom", color="#8a1f1f")
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 2: throughput -------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(11.7, 8.3))
        fig.suptitle("Throughput", fontsize=14)
        for g in grids:
            xs, ys = by(g, "env_steps_per_s")
            axes[0].plot(xs, ys, "o-", color=GRID_COLORS[g], label=g)
        axes[0].set_xlabel("voxel size [mm]")
        axes[0].set_ylabel("env-steps / s (all envs)")
        axes[0].set_title(f"Throughput vs voxel size ({args.num_envs} envs)")
        axes[0].invert_xaxis()
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        width = 0.25
        for gi, g in enumerate(grids):
            xs, ys = by(g, "step_ms")
            pos = [voxels.index(x / 1000.0) + (gi - 1) * width for x in xs]
            axes[1].bar(pos, ys, width=width, color=GRID_COLORS[g], label=g)
        axes[1].set_xticks(range(len(voxels)))
        axes[1].set_xticklabels([f"{v*1000:g}" for v in voxels])
        axes[1].set_xlabel("voxel size [mm]")
        axes[1].set_ylabel("env step time [ms]")
        axes[1].set_title("Step time (lower is better)")
        axes[1].grid(alpha=0.3, axis="y")
        axes[1].legend()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 3: memory ------------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(11.7, 8.3))
        fig.suptitle("GPU memory", fontsize=14)
        for g in grids:
            xs, ys = by(g, "peak_mem_mib")
            axes[0].plot(xs, [y / 1024 for y in ys], "o-", color=GRID_COLORS[g], label=f"{g} (peak, nvidia-smi)")
            xs2, ys2 = by(g, "cuda_used_after_run_gib")
            axes[0].plot(xs2, ys2, "x--", color=GRID_COLORS[g], alpha=0.6, label=f"{g} (steady, in-run)")
        axes[0].axhline(info["total_mib"] / 1024, color="red", ls=":", label="GPU capacity")
        axes[0].set_xlabel("voxel size [mm]")
        axes[0].set_ylabel("GPU memory [GiB]")
        axes[0].set_title("Memory vs voxel size")
        axes[0].invert_xaxis()
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        for g in grids:
            pts = [(r.get("particles", r["expected_particles"]), r["peak_mem_mib"] / 1024) for r in ok if r["grid_type"] == g]
            if pts:
                axes[1].plot(*zip(*sorted(pts)), "o-", color=GRID_COLORS[g], label=g)
        axes[1].set_xlabel("total particles")
        axes[1].set_ylabel("peak GPU memory [GiB]")
        axes[1].set_title("Memory vs particle count")
        axes[1].set_xscale("log")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 4: efficiency + startup ------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(11.7, 8.3))
        fig.suptitle("Efficiency and startup", fontsize=14)
        for g in grids:
            xs, ys = [], []
            for r in ok:
                if r["grid_type"] == g and "step_ms" in r:
                    n = r.get("particles", r["expected_particles"])
                    xs.append(r["voxel"] * 1000.0)
                    ys.append(r["step_ms"] * 1000.0 / (n * SUBSTEPS_PER_ENV_STEP))  # µs / particle-substep
            order = sorted(range(len(xs)), key=lambda k: -xs[k])
            axes[0].plot([xs[k] for k in order], [ys[k] for k in order], "o-", color=GRID_COLORS[g], label=g)
        axes[0].set_xlabel("voxel size [mm]")
        axes[0].set_ylabel("µs per particle per MPM substep")
        axes[0].set_title("Solver efficiency (lower is better)")
        axes[0].set_yscale("log")
        axes[0].invert_xaxis()
        axes[0].grid(alpha=0.3, which="both")
        axes[0].legend()
        width = 0.25
        for gi, g in enumerate(grids):
            xs, ys = by(g, "startup_s")
            pos = [voxels.index(x / 1000.0) + (gi - 1) * width for x in xs]
            axes[1].bar(pos, ys, width=width, color=GRID_COLORS[g], label=g)
        axes[1].set_xticks(range(len(voxels)))
        axes[1].set_xticklabels([f"{v*1000:g}" for v in voxels])
        axes[1].set_xlabel("voxel size [mm]")
        axes[1].set_ylabel("startup + warmup time [s]")
        axes[1].set_title("Startup (build + finalize + graph capture + warmup)")
        axes[1].grid(alpha=0.3, axis="y")
        axes[1].legend()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 5 (optional): sparse/dense viability vs env count -------------------------------
        if aux:
            fig, axes = plt.subplots(1, 2, figsize=(11.7, 8.3))
            fig.suptitle("Secondary probe: sparse/dense grid viability vs env count (voxel 15 mm)", fontsize=13)
            env_counts = sorted({r["num_envs"] for r in aux})
            fixed_ref = next(
                (r for r in results if r["status"] == "ok" and r["grid_type"] == "fixed" and r["voxel"] == 0.015),
                None,
            )
            for g in ("sparse", "dense"):
                xs = [r["num_envs"] for r in aux if r["grid_type"] == g and r["status"] == "ok"]
                ys = [r["step_ms"] for r in aux if r["grid_type"] == g and r["status"] == "ok"]
                if xs:
                    axes[0].plot(xs, ys, "o-", color=GRID_COLORS[g], label=g)
                fx = [r["num_envs"] for r in aux if r["grid_type"] == g and r["status"] != "ok"]
                for x in fx:
                    axes[0].axvline(x, color=GRID_COLORS[g], ls=":", alpha=0.4)
            if fixed_ref and "step_ms" in fixed_ref:
                axes[0].axhline(
                    fixed_ref["step_ms"],
                    color=GRID_COLORS["fixed"],
                    ls="--",
                    label=f"fixed @ {fixed_ref['num_envs']} envs ({fixed_ref['step_ms']:.0f} ms)",
                )
            axes[0].set_xlabel("num envs")
            axes[0].set_ylabel("env step time [ms]")
            axes[0].set_title("Step time where the run fits (dotted = OOM)")
            axes[0].grid(alpha=0.3)
            axes[0].legend(fontsize=8)
            for gi, g in enumerate(("sparse", "dense")):
                xs, ys = [], []
                for n in env_counts:
                    r = next((r for r in aux if r["grid_type"] == g and r["num_envs"] == n), None)
                    if r is not None:
                        xs.append(env_counts.index(n) + (gi - 0.5) * 0.35)
                        ys.append(r["peak_mem_mib"] / 1024)
                axes[1].bar(xs, ys, width=0.35, color=GRID_COLORS[g], label=g)
            axes[1].axhline(info["total_mib"] / 1024, color="red", ls=":", label="GPU capacity")
            axes[1].set_xticks(range(len(env_counts)))
            axes[1].set_xticklabels([str(n) for n in env_counts])
            axes[1].set_xlabel("num envs")
            axes[1].set_ylabel("peak GPU memory [GiB]")
            axes[1].set_title("Peak memory (incl. failed-at-init runs)")
            axes[1].grid(alpha=0.3, axis="y")
            axes[1].legend(fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)

        # ---- page 6: observations ----------------------------------------------------------------
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.suptitle("Observations", fontsize=14, y=0.96)
        lines: list[str] = []
        if ok:
            best = max(ok, key=lambda r: r.get("env_steps_per_s", 0))
            lines.append(
                f"• Best throughput: {best['grid_type']} grid @ {best['voxel']*1000:g} mm voxel — "
                f"{best['env_steps_per_s']:.0f} env-steps/s ({best['step_ms']:.1f} ms/step, "
                f"{best.get('particles', 0):,} particles)."
            )
            for v in voxels:
                fr = next((r for r in ok if r["grid_type"] == "fixed" and r["voxel"] == v), None)
                sr = next((r for r in ok if r["grid_type"] == "sparse" and r["voxel"] == v), None)
                if fr and sr and "step_ms" in fr and "step_ms" in sr:
                    lines.append(
                        f"• {v*1000:g} mm: fixed grid (+CUDA graph) is {sr['step_ms'] / fr['step_ms']:.2f}× faster than"
                        f" sparse per step; peak memory {fr['peak_mem_mib']/1024:.1f} vs {sr['peak_mem_mib']/1024:.1f} GiB."
                    )
            heaviest = max(ok, key=lambda r: r["peak_mem_mib"])
            lines.append(
                f"• Highest memory among successful runs: {heaviest['grid_type']} @ {heaviest['voxel']*1000:g} mm — "
                f"{heaviest['peak_mem_mib']/1024:.1f} GiB of {info['total_mib']/1024:.1f} GiB."
            )
        for r in results:
            if r["status"] != "ok":
                lines.append(
                    f"• FAILED: {r['grid_type']} @ {r['voxel']*1000:g} mm ({r['status']}). {r.get('error', '')[:200]}"
                )
        if aux:
            lines.append("")
            lines.append("Secondary probe (voxel 15 mm):")
            for g in ("sparse", "dense"):
                oks = sorted(r["num_envs"] for r in aux if r["grid_type"] == g and r["status"] == "ok")
                bads = sorted(r["num_envs"] for r in aux if r["grid_type"] == g and r["status"] != "ok")
                lines.append(f"• {g}: ok at {oks or 'no'} envs, failed at {sorted(set(bads))} envs.")
            cap_runs = [r for r in aux if r["num_envs"] == 48 and r["grid_type"] == "sparse"]
            if cap_runs and all(r["status"] != "ok" for r in cap_runs):
                lines.append(
                    "• max_active_cell_count is NOT the lever: sparse @ 48 envs fails with both a 1.09M cap and"
                    " unlimited (-1), so the 7 mm sparse success is a property of the grid layout, not the cap."
                )
            lines.append(
                "• Sparse/dense init memory scales with the multi-env spatial extent rather than particle count"
                " (failures all OOM during solver init at ~30 GiB regardless of voxel), which is why only the"
                " fixed grid is usable at training env counts on this GPU."
            )
        lines += [
            "",
            "Caveats:",
            "• Zero-action workload: the cup barely moves, so collider rasterization and grid re-allocation are",
            "  near best-case for sparse/dense; training workloads move media and may widen the gap.",
            "• max_active_cells was auto-scaled per voxel (0.35 × particles, ≥ 400k) and is part of the memory cost",
            "  for the fixed grid (preallocated FEM/BSR matrices).",
            "• Physics validity differs across voxel sizes: the ladle wall (24 mm) needs ≥ ~1.5 voxels to stay",
            "  watertight, so voxels ≳ 16 mm will leak media in real training; small voxels need a larger",
            "  fixed-grid padding for the carry phase. This sweep measures performance only.",
            "• CUDA graphs only apply to the fixed grid; sparse/dense run eager (their per-step allocation",
            "  patterns are not capturable).",
        ]
        fig.text(0.05, 0.9, "\n".join(lines), fontsize=9.5, va="top", family="DejaVu Sans")
        pdf.savefig(fig)
        plt.close(fig)

    print(f"[sweep] report -> {out_pdf}", flush=True)


def main() -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        results = json.loads(Path(args.report_only).read_text())
    else:
        results = run_sweep()
    aux: list[dict] = []
    for p in args.aux_results:
        aux.extend(json.loads(Path(p).read_text()))
    stamp = datetime.date.today().isoformat()
    render_report(results, out_dir / f"scoop_voxel_grid_report_{stamp}.pdf", aux=aux or None)


if __name__ == "__main__":
    main()
