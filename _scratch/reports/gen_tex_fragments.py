# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Print LaTeX table rows and pgfplots coordinate lists from an env-scaling results CSV.

Keeps the report reproducible: re-run the sweep, then regenerate the fragments and paste
(or diff) them into scoop_env_scaling_report.tex.

    python3 gen_tex_fragments.py env_scaling/scoop_env_scaling_<date>.csv
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict


def fmt(value: str, nd: int = 1) -> str:
    return f"{float(value):,.{nd}f}".replace(",", "{,}") if value else "---"


def main(path: str) -> None:
    rows = list(csv.DictReader(open(path)))
    by_voxel: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        by_voxel[float(r["voxel_mm"])].append(r)

    for voxel, group in sorted(by_voxel.items(), reverse=True):
        group.sort(key=lambda r: int(r["num_envs"]))
        print(f"% ---- table rows: voxel {voxel:g} mm ----")
        for r in group:
            if r["status"] == "ok":
                print(
                    f"{r['num_envs']:>4} & {fmt(r['step_ms'])} & {fmt(r['env_steps_per_s'], 0)} &"
                    f" {fmt(r['steps_per_s_per_env'], 2)} & {fmt(r['startup_s'], 0)} &"
                    f" {float(r['peak_mem_mib']) / 1024:.1f} \\\\"
                )
            else:
                print(
                    f"{r['num_envs']:>4} & \\multicolumn{{4}}{{c}}{{\\emph{{{r['status']}}}}} &"
                    f" {float(r['peak_mem_mib']) / 1024:.1f} \\\\"
                )
        for key, nd in (("env_steps_per_s", 0), ("step_ms", 1), ("startup_s", 1)):
            pts = " ".join(f"({r['num_envs']},{float(r[key]):.{nd}f})" for r in group if r["status"] == "ok" and r[key])
            print(f"% {key} coords v{voxel:g}mm: {pts}")
        pts = " ".join(f"({r['num_envs']},{float(r['peak_mem_mib']) / 1024:.2f})" for r in group)
        print(f"% peak_mem_gib coords v{voxel:g}mm: {pts}")
        print()


if __name__ == "__main__":
    main(sys.argv[1])
