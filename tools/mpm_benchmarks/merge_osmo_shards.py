# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Merge independent OSMO benchmark shards into one presentation result set."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_suite


def create_parser() -> argparse.ArgumentParser:
    """Create the shard-aggregation argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="One OSMO task output root.")
    parser.add_argument("--output", type=Path, required=True, help="New aggregate output directory.")
    parser.add_argument("--workflow_id", required=True, help="OSMO workflow identifier recorded in provenance.")
    parser.add_argument(
        "--expected_input_count", type=int, help="Fail unless exactly this many task output roots were supplied."
    )
    return parser


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _run_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return str(row["sweep"]), str(row["case_id"]), int(row["replicate"]), int(row["run_order"])


def main() -> None:
    """Copy raw shards, validate their union, and emit combined CSV/JSON artifacts."""
    args = create_parser().parse_args()
    if args.expected_input_count is not None and len(args.input) != args.expected_input_count:
        raise ValueError(f"Expected {args.expected_input_count} shard inputs, received {len(args.input)}.")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite aggregate output: {output}")
    output.mkdir(parents=True)
    raw_root = output / "raw_shards"
    raw_root.mkdir()

    planned_by_key: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    outcomes_by_key: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    successful_by_key: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    shard_inventory: list[dict[str, Any]] = []
    task_failure_paths: list[Path] = []

    for shard_index, input_root in enumerate(args.input):
        input_root = input_root.resolve()
        shard_copy = raw_root / f"shard_{shard_index:02d}"
        shutil.copytree(input_root, shard_copy)
        planned_paths = sorted(input_root.glob("**/planned_runs.json"))
        outcome_paths = sorted(input_root.glob("**/outcomes.json"))
        result_paths = sorted(input_root.glob("**/result.json"))
        shard_task_failures = sorted(input_root.glob("**/task_failure*.txt"))
        task_failure_paths.extend(shard_task_failures)
        shard_inventory.append(
            {
                "shard_index": shard_index,
                "input": str(input_root),
                "planned_files": len(planned_paths),
                "outcome_files": len(outcome_paths),
                "result_files": len(result_paths),
                "task_failure_files": len(shard_task_failures),
            }
        )
        for path in planned_paths:
            for row in _read_json(path):
                key = _run_key(row)
                if key in planned_by_key:
                    raise RuntimeError(f"Duplicate planned run {key} in {path}")
                planned_by_key[key] = row
        for path in outcome_paths:
            for row in _read_json(path):
                key = _run_key(row)
                if key in outcomes_by_key:
                    raise RuntimeError(f"Duplicate outcome {key} in {path}")
                outcomes_by_key[key] = row
        for path in result_paths:
            result = _read_json(path)
            flat = result["flat"]
            key = _run_key(flat)
            if key in successful_by_key:
                raise RuntimeError(f"Duplicate successful result {key} in {path}")
            successful_by_key[key] = flat

    if task_failure_paths:
        raise RuntimeError(f"OSMO shard process failures: {[str(path) for path in task_failure_paths]}")

    planned = [planned_by_key[key] for key in sorted(planned_by_key, key=lambda item: item[3])]
    outcomes = [outcomes_by_key[key] for key in sorted(outcomes_by_key, key=lambda item: item[3])]
    successful = [successful_by_key[key] for key in sorted(successful_by_key, key=lambda item: item[3])]
    completed_outcome_keys = {key for key, row in outcomes_by_key.items() if row.get("status") == "completed"}
    if set(successful_by_key) != completed_outcome_keys:
        missing = sorted(completed_outcome_keys - set(successful_by_key))
        unexpected = sorted(set(successful_by_key) - completed_outcome_keys)
        raise RuntimeError(f"Completed outcome/result mismatch: missing={missing}, unexpected={unexpected}")
    cohort_fields = (
        "gpu_name",
        "gpu_total_memory_mib",
        "driver_version",
        "git_commit",
        "newton_version",
        "warp_version",
        "isaaclab_version",
    )
    cohort_values = {
        field: sorted({str(row.get(field, "")) for row in successful if row.get(field) not in (None, "")})
        for field in cohort_fields
    }
    mixed_fields = {field: values for field, values in cohort_values.items() if len(values) > 1}
    if mixed_fields:
        raise RuntimeError(f"Refusing to merge a mixed hardware/software cohort: {mixed_fields}")
    gpu_names = cohort_values["gpu_name"]

    run_suite._write_json(output / "planned_runs.json", planned)
    run_suite._write_json(output / "outcomes.json", outcomes)
    run_suite._write_csv(output / "planned_runs.csv", planned)
    run_suite._write_csv(output / "outcomes.csv", outcomes)
    run_suite._write_csv(output / "runs.csv", successful, fieldnames=list(successful[0]) if successful else ["status"])
    summary = run_suite._aggregate(successful, planned, outcomes)
    run_suite._write_csv(output / "summary.csv", summary)
    for sweep in sorted({str(row["sweep"]) for row in summary}):
        presentation_rows = [row for row in summary if row["sweep"] == sweep and row["status"] == "completed"]
        run_suite._write_csv(output / f"presentation_{sweep}.csv", presentation_rows)

    failed = [row for row in outcomes if row.get("status") == "failed"]
    skipped = [row for row in outcomes if row.get("status") == "skipped_preflight"]
    if failed:
        run_suite._write_csv(output / "failures.csv", failed)
    if skipped:
        run_suite._write_csv(output / "skipped.csv", skipped)
    run_suite._write_json(
        output / "osmo_provenance.json",
        {
            "workflow_id": args.workflow_id,
            "merged_utc": datetime.now(UTC).isoformat(),
            "shards": shard_inventory,
            "planned_runs": len(planned),
            "outcomes": len(outcomes),
            "successful_runs": len(successful),
            "gpu_names": gpu_names,
            "cohort": {field: values[0] if values else "" for field, values in cohort_values.items()},
            "mixed_gpu_cohort": False,
            "mixed_hardware_or_software_cohort": False,
        },
    )
    print(
        f"[OSMO MERGE] planned={len(planned)} outcomes={len(outcomes)} successful={len(successful)} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
