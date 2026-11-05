# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gate matched source cohorts and assemble an offline Pages report for review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FIELDS = ("backend", "task", "mode", "vision", "num_envs", "seed")
PROFILE = {
    "genesis": {"max_iterations": 25, "tolerance": None},
    "isaaclab": {
        "solver": "newton",
        "integrator": "implicitfast",
        "iterations": 100,
        "ls_iterations": 50,
        "tolerance": 1e-6,
        "cone": "pyramidal",
        "impratio": 1.0,
        "use_mujoco_contacts": True,
        "njmax": 300,
        "nconmax": None,
        "num_substeps": 1,
        "use_cuda_graph": True,
    },
    "isaaclab_physx": {"position_iterations": 4, "velocity_iterations": 0},
}
RUNTIME = {
    "torch": "2.11.0+cu130",
    "torch_cuda": "13.0",
    "torch_matmul_precision": "highest",
    "cudnn_benchmark": False,
    "cudnn_deterministic": False,
}
UPSTREAM = {
    "isaaclab": "efa240ec39794f5d80430cbe38620626954c2c4d",
    "genesis": "969c3bd1449d0f0c79effd7b6a72614ed46ba6e7",
    "newton": "42f7def118c2c4299ce493457bafd4e8a22a696e",
}
HARNESS_PATH = Path("scripts/benchmarks/simulator_comparison")
EXCLUDED_RUNTIME = {"report", "export_figures", "validate_adapter", "assemble_report"}


def require(condition: bool, message: str) -> None:
    """Reject incomplete or inconsistent publication evidence."""
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    """Return the SHA-256 hash of a file without loading it all into memory."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, data: dict[str, Any] | list[Any]) -> None:
    """Write JSON evidence, rejecting non-finite numbers."""
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def runtime_fingerprint(files: dict[str, bytes]) -> str:
    """Hash committed runtime files using the suite's filename-and-bytes contract."""
    runtime = [
        (Path(name).name, content)
        for name, content in files.items()
        if Path(name).parent == HARNESS_PATH and Path(name).suffix == ".py" and Path(name).stem not in EXCLUDED_RUNTIME
    ]
    require(bool(runtime), "Source snapshot contains no runtime files")
    return hashlib.sha256(b"".join(name.encode() + content for name, content in sorted(runtime))).hexdigest()


def committed_sources(
    harness: Path, commit: str, source_archive: Path | None = None
) -> tuple[dict[str, bytes], str, str | None]:
    """Read exact committed source files, runtime fingerprint and upstream base."""
    require(re.fullmatch("[0-9a-f]{40}", commit), "source_commit must be a full 40-character commit SHA")
    if source_archive is not None:
        with zipfile.ZipFile(source_archive) as archive:
            files = {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}
        require("LICENSE" in files, "Source archive must include the BSD license")
        return files, runtime_fingerprint(files), None
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=harness, text=True).strip())
    relative = harness.relative_to(repo)
    raw = subprocess.check_output(["git", "archive", commit, str(relative), "LICENSE"], cwd=repo)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        files = {member.name: archive.extractfile(member).read() for member in archive if member.isfile()}
    fingerprint = runtime_fingerprint(files)
    upstream = subprocess.check_output(["git", "merge-base", commit, "origin/main"], cwd=repo, text=True).strip()
    return files, fingerprint, upstream


def expected_cases(tasks: tuple[str, ...] = ("anymal", "cartpole")) -> set[tuple[str, str, str, str, int, int]]:
    """Return the complete publication matrix of backend/task/workload/seed tuples."""
    workloads = [("physics", "none", n) for n in (256, 4096)]
    workloads += [("rollout", "none", n) for n in (256, 4096)] + [("rollout", "rgb", 64)]
    workloads += [("learning", vision, 512) for vision in ("none", "rgb")]
    return {
        (b, t, m, v, n, s) for b in ("genesis", "isaaclab") for t in tasks for m, v, n in workloads for s in (0, 1, 2)
    }


def zero_overflow(flags: object) -> bool:
    """Check that a recorded flag array is present and contains only zero flags."""
    return (
        isinstance(flags, list)
        and bool(flags)
        and all(zero_overflow(value) if isinstance(value, list) else value == 0 for value in flags)
    )


def check_mjwarp(
    config: dict[str, Any],
    metadata: dict[str, Any],
    diagnostics: dict[str, Any],
    kit_modules: list[str] | None,
    path: Path,
) -> None:
    """Validate main MJWarp identity, effective options and final-state diagnostics."""
    configured = {
        key: value for key, value in PROFILE["isaaclab"].items() if key not in {"num_substeps", "use_cuda_graph"}
    }
    require(all(metadata.get(key) == value for key, value in configured.items()), f"MJWarp settings differ: {path}")
    require(
        metadata.get("physics_backend") == "Isaac Lab Newton / MuJoCo Warp"
        and metadata.get("solver_profile") == "shared"
        and metadata.get("kit_runtime") is False
        and kit_modules == []
        and metadata.get("substeps") == 1
        and metadata.get("cuda_graph") is True,
        f"Wrong main backend or Kit/substep/graph state: {path}",
    )
    actual = metadata.get("actual_solver", {})
    # MuJoCo mjtype.h enums: Newton=2, implicitfast=3, pyramidal=0.
    effective = {
        "iterations": 100,
        "ls_iterations": 50,
        "tolerance": 1e-6,
        "impratio": 1.0,
        "solver_enum": 2,
        "integrator_enum": 3,
        "cone_enum": 0,
        "njmax": 300,
    }
    require(
        "mujoco" in actual.get("class", "").lower()
        and all(
            isinstance(actual.get(key), (int, float)) and math.isclose(actual[key], value, rel_tol=1e-7, abs_tol=1e-12)
            for key, value in effective.items()
        ),
        f"Effective MJWarp model options differ: {path}",
    )
    require(
        diagnostics.get("graph_active") is True and zero_overflow(diagnostics.get("fatal_capacity_flags_at_end")),
        f"Inactive MJWarp graph or nonzero/missing final-state fatal capacity flags: {path}",
    )
    require(configured["njmax"] == diagnostics.get("constraint_capacity_per_world"), f"MJWarp capacity differs: {path}")
    if config["vision"] == "rgb":
        require(metadata.get("renderer") == "Newton Warp batched Camera", f"Wrong main RGB renderer: {path}")


def validate_runs(
    loaded: list[tuple[Path, dict[str, Any]]], fingerprint: str, tasks: tuple[str, ...] = ("anymal", "cartpole")
) -> list[tuple[Path, dict[str, Any]]]:
    """Require the complete matched matrix with consistent code, hardware and runtime."""
    from report import TASK_ERRORS, group_results

    accepted = [
        (path, run)
        for path, run in loaded
        if run.get("status") == "ok" and run.get("provenance", {}).get("other_active_compute_detected") is not True
    ]
    expected = expected_cases(tasks)
    cases = Counter(tuple(run["config"].get(field) for field in FIELDS) for _, run in accepted)
    require(
        len(accepted) == len(expected) and set(cases) == expected and max(cases.values(), default=0) == 1,
        f"Expected {len(expected)} unique successful cases; got {len(accepted)}. Missing={expected - set(cases)}; "
        f"unexpected={set(cases) - expected}; duplicates={[case for case, n in cases.items() if n > 1]}",
    )
    devices = set()
    for path, run in accepted:
        config, provenance, metrics = run["config"], run["provenance"], run["metrics"]
        samples, runtime = provenance.get("gpu_activity_samples", []), provenance["runtime"]
        checks = {
            "schema": run.get("schema_version") == 1,
            "shared profile": config.get("solver_profile") == "shared"
            and config.get("solver_settings") == PROFILE
            and config.get("solver_iterations") is None,
            "timestep": config.get("dt") == 0.005 and config.get("decimation") == 4,
            "dimensions": (config.get("width"), config.get("height")) == (64, 64),
            "source code": provenance.get("harness_sha256") == fingerprint,
            "upstream pins": provenance.get("source_revisions") == UPSTREAM,
            "activity audit": provenance.get("other_active_compute_detected") is False
            and samples
            and all(sample.get("external_compute") == [] for sample in samples),
            "Torch/CUDA/CuDNN": all(runtime.get(key) == value for key, value in RUNTIME.items()),
        }
        for label, passed in checks.items():
            require(passed, f"Unexpected {label}: {path}")
        assets = provenance.get("asset_manifest", {})
        if config["task"] == "franka":
            assets = assets.get("franka", {})
        require(
            re.fullmatch(r"[0-9a-f]{64}", assets.get("source_urdf_sha256", ""))
            and isinstance(assets.get("meshes_sha256"), dict)
            and assets["meshes_sha256"],
            f"Missing physical asset hashes: {path}",
        )
        device = str(uuid.UUID(runtime["cuda_device_uuid"].removeprefix("GPU-")))
        hardware_uuids = re.findall(r"GPU-([0-9a-fA-F-]{36})", provenance["hardware"]["gpu"])
        require(
            device in {str(uuid.UUID(value)) for value in hardware_uuids}, f"GPU UUID not in hardware record: {path}"
        )
        devices.add(device)
        metadata = run.get("backend_metadata", {})
        if config["backend"] == "genesis":
            actual = (
                metadata.get("solver_profile"),
                metadata.get("constraint_solver"),
                metadata.get("solver_iterations"),
            )
            expected = ("shared", "Newton", 25)
            require(actual == expected, f"Genesis solver metadata differs: {path}")
        else:
            check_mjwarp(config, metadata, run.get("backend_diagnostics", {}), runtime.get("kit_modules_loaded"), path)
        keys = ["startup_s", "physics_steps_per_s" if config["mode"] == "physics" else "env_steps_per_s"]
        require(
            all(
                isinstance(metrics.get(key), (int, float)) and math.isfinite(metrics[key]) and metrics[key] > 0
                for key in keys
            ),
            f"Missing/nonpositive timing metrics: {path}",
        )
        if config["mode"] == "learning":
            require(
                (config.get("iterations"), config.get("rollout_steps"), config.get("evaluation_steps"))
                == (500, 24, 250),
                f"Unexpected learning budget: {path}",
            )
            require(config.get("evaluation_interval") == 50, f"Unexpected evaluation interval: {path}")
            curve = metrics.get("learning", [])
            evaluation = metrics.get("evaluation", {})
            error = evaluation.get(TASK_ERRORS[config["task"]][0])
            survival = evaluation.get("survival_fraction")
            require(
                curve
                and curve[0]["iteration"] == 0
                and curve[-1]["iteration"] == 500
                and curve[-1]["env_steps"] == 500 * 24 * 512
                and evaluation.get("steps") == 250
                and isinstance(error, (int, float))
                and math.isfinite(error)
                and error >= 0
                and isinstance(survival, (int, float))
                and math.isfinite(survival)
                and 0 <= survival <= 1,
                f"Incomplete learning/evaluation evidence: {path}",
            )
    require(len(devices) == 1, f"Different physical GPUs: {devices}")
    groups = group_results([run for _, run in accepted])
    expected_groups = 7 * len(tasks)
    require(
        len(groups) == expected_groups and all(len(values) == 6 for _, values in groups),
        f"Expected {expected_groups} matched six-run groups",
    )
    return accepted


def flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten scalar metrics into CSV columns without duplicating learning arrays."""
    result = {}
    for key, item in value.items():
        if isinstance(item, dict):
            result.update(flatten(item, prefix + key + "."))
        elif not isinstance(item, list):
            result[prefix + key] = item
    return result


def validate_evidence(
    directory: Path,
    fingerprint: str,
    validator_hash: str,
    assets: dict[str, Any],
    tasks: tuple[str, ...] = ("anymal", "cartpole"),
) -> list[tuple[Path, dict[str, Any]]]:
    """Require validators for each task, backend and vision using that cohort's exact source and inputs."""
    expected = {
        (backend, task, vision) for backend in ("genesis", "isaaclab") for task in tasks for vision in ("none", "rgb")
    }
    selected = {}
    for path in sorted(directory.rglob("*.json")):
        evidence = json.loads(path.read_text())
        if evidence.get("kind") != "adapter_validation" or evidence.get("status") != "ok":
            continue
        config = evidence.get("config", {})
        provenance = config.get("provenance", {})
        if (
            config.get("solver_profile") != "shared"
            or provenance.get("harness_sha256") != fingerprint
            or evidence.get("validator_sha256") != validator_hash
        ):
            continue
        key = tuple(config.get(field) for field in ("backend", "task", "vision"))
        require(key in expected and key not in selected, f"Unexpected or duplicate final validator: {path}")
        require(
            config.get("solver_settings") == PROFILE
            and config.get("dt") == 0.005
            and config.get("decimation") == 4
            and config.get("num_envs", 0) >= 2
            and provenance.get("source_revisions") == UPSTREAM,
            f"Validation settings differ: {path}",
        )
        recorded_assets = provenance.get("asset_manifest", {})
        expected_assets = assets
        if key[1] == "franka":
            recorded_assets = recorded_assets.get("franka", {})
            expected_assets = assets.get("franka", {})
        require(
            expected_assets
            and all(
                recorded_assets.get(field) == expected_assets.get(field)
                for field in ("source_urdf_sha256", "meshes_sha256")
            ),
            f"Validation physical assets differ: {path}",
        )
        required = {"finite_state", "joint_order", "subset_reset", "moving_mass"}
        required |= {
            "anymal": {"anymal_short_stability"},
            "cartpole": {"cartpole_independent_dynamics", "cartpole_force_direction"},
            "franka": {"franka_independent_end_effector_kinematics", "franka_short_stability"},
        }[key[1]]
        if key[2] == "rgb":
            required |= {"rgb_tensor_contract", "rgb_nonblank_every_env", "rgb_distinct_environments"}
        if key[0] == "isaaclab":
            required |= {
                "mjwarp_graph_after_warmup",
                "mjwarp_changed_control_after_capture",
                "mjwarp_commands_preserve_graph",
                "mjwarp_no_capacity_overflow_at_end",
                "mjwarp_kitless_runtime",
                "mjwarp_finite_after_controls",
            }
        checks = evidence.get("checks", {})
        require(
            required <= checks.keys() and all(check.get("passed") is True for check in checks.values()),
            f"Missing or failed independent validation checks: {path}",
        )
        metadata = evidence.get("backend_metadata", {})
        require(metadata.get("solver_profile") == "shared", f"Wrong validation backend profile: {path}")
        if key[0] == "isaaclab":
            kit_modules = checks.get("mjwarp_kitless_runtime", {}).get("loaded_kit_modules")
            check_mjwarp(config, metadata, evidence.get("backend_diagnostics", {}), kit_modules, path)
        else:
            require(
                metadata.get("constraint_solver") == "Newton" and metadata.get("solver_iterations") == 25,
                f"Wrong Genesis validation solver: {path}",
            )
        selected[key] = (path, evidence)
    require(set(selected) == expected, f"Expected {len(expected)} final validators; missing={expected - set(selected)}")
    rgb_directories = [path.parent for path, record in selected.values() if record["config"]["vision"] == "rgb"]
    require(
        len(set(rgb_directories)) == 2 * len(tasks), "RGB validators need separate directories to preserve their frames"
    )
    require(
        all(list(directory.glob("validation_camera_env*.png")) for directory in rgb_directories),
        "Missing independent RGB validation frames",
    )
    return list(selected.values())


def load_cohorts(args: argparse.Namespace, harness: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a local launch manifest or the same portable manifest from an exported report."""
    manifest_path = getattr(args, "cohort_manifest", None)
    manifest = json.loads(manifest_path.read_text()) if manifest_path else {}
    base = manifest_path.resolve().parent if manifest_path else Path.cwd()
    specs = manifest.get("cohorts")
    if specs is None:
        require(args.input_dir and args.validation_dir, "Provide cohort_manifest or input_dir and validation_dir")
        specs = [
            {
                "name": "baseline",
                "tasks": ["anymal", "cartpole"],
                "source_commit": args.source_commit,
                "input_dir": str(args.input_dir.resolve()),
                "validation_dir": str(args.validation_dir.resolve()),
            }
        ]
    else:
        require(
            {tuple(spec.get("tasks", [])) for spec in specs}
            in ({("anymal", "cartpole")}, {("anymal", "cartpole"), ("franka",)})
            and len(specs) == len({tuple(spec.get("tasks", [])) for spec in specs}),
            "Combined publication requires baseline ANYmal/cartpole and a separate Franka cohort",
        )
    cohorts, names = [], set()
    for spec in specs:
        name, tasks = spec["name"], tuple(spec["tasks"])
        require(
            re.fullmatch(r"[a-z0-9_-]+", name) and name not in names, "Cohort names must be unique path-safe labels"
        )
        names.add(name)
        archive = base / spec["source_archive"] if spec.get("source_archive") else None
        if archive:
            require(digest(archive) == spec.get("source_archive_sha256"), f"Source archive hash differs: {name}")
        sources, fingerprint, upstream = committed_sources(harness, spec["source_commit"], archive)
        require(
            spec.get("runtime_harness_sha256", fingerprint) == fingerprint,
            f"Declared cohort runtime differs from its source: {name}",
        )
        cameras = {}
        if "records" in spec:
            paths = [base / record["path"] for record in spec["records"]]
            cameras = {
                base / record["path"]: base / record["camera_sample"]
                for record in spec["records"]
                if record.get("camera_sample")
            }
            source = None
        else:
            source = base / spec["input_dir"]
            paths = sorted(source.rglob("*.result.json"))
        loaded = [(path, json.loads(path.read_text())) for path in paths]
        accepted = validate_runs(loaded, fingerprint, tasks)
        actual_device = accepted[0][1]["provenance"]["runtime"]["cuda_device_uuid"]
        require(
            str(uuid.UUID(spec.get("cuda_device_uuid", actual_device).removeprefix("GPU-")))
            == str(uuid.UUID(actual_device.removeprefix("GPU-"))),
            f"Declared cohort GPU differs from measurements: {name}",
        )
        validator = sources.get(str(HARNESS_PATH / "validate_adapter.py"))
        require(validator is not None, f"Source snapshot lacks validator: {name}")
        validator_hash = hashlib.sha256(validator).hexdigest()
        require(spec.get("validator_sha256", validator_hash) == validator_hash, f"Declared validator differs: {name}")
        validation = base / spec["validation_dir"]
        require(validation.is_dir(), f"Missing final validation directory: {name}")
        evidence = validate_evidence(
            validation, fingerprint, validator_hash, accepted[0][1]["provenance"]["asset_manifest"], tasks
        )
        cohorts.append(
            dict(
                name=name,
                tasks=tasks,
                source_commit=spec["source_commit"],
                sources=sources,
                fingerprint=fingerprint,
                validator_hash=validator_hash,
                upstream=upstream or spec.get("isaaclab_upstream_base"),
                source=source,
                validation=validation,
                loaded=loaded,
                accepted=accepted,
                evidence=evidence,
                cameras=cameras,
            )
        )
    hosts = set()
    for cohort in cohorts:
        for _, run in cohort["accepted"]:
            hardware = run["provenance"]["hardware"]
            cpu = "\n".join(line for line in hardware.get("cpu", "").splitlines() if "MHz" not in line)
            # lscpu scaling frequency changes between launches on the same host.
            inventory = tuple(sorted(re.findall(r"GPU-([0-9a-fA-F-]{36})", hardware["gpu"])))
            hosts.add((hardware.get("platform"), cpu, inventory))
    require(len(hosts) == 1, "Source cohorts must share the recorded CPU host")
    return cohorts, manifest.get("publisher", {})


def assemble_report(args: argparse.Namespace) -> None:
    """Build a reviewable report from complete final measurements and validation evidence."""
    harness, output = args.harness_dir.resolve(), args.output_dir.resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(harness))
    from export_figures import export_figures
    from report import BACKEND_LABELS, build_report

    cohorts, publisher = load_cohorts(args, harness)
    manifest_path = getattr(args, "cohort_manifest", None)
    publisher_archive = None
    if publisher.get("source_archive"):
        publisher_archive = manifest_path.resolve().parent / publisher["source_archive"]
        require(digest(publisher_archive) == publisher.get("source_archive_sha256"), "Publisher archive hash differs")
        require(publisher.get("source_commit") == args.source_commit, "Publisher revision differs from source_commit")
    sources, _, _ = committed_sources(harness, args.source_commit, publisher_archive)
    for name in ("report.py", "export_figures.py", "assemble_report.py", "validate_adapter.py"):
        committed = sources.get(str(HARNESS_PATH / name))
        require(
            committed is not None and (harness / name).read_bytes() == committed,
            f"Publication helper differs from source_commit: {name}",
        )
    require(not output.exists() or not any(output.iterdir()), "Output must be empty to avoid stale artifacts")
    cases = [(cohort, path, run) for cohort in cohorts for path, run in cohort["accepted"]]
    accepted = [(path, run) for _, path, run in cases]
    expected_groups = sum(7 * len(cohort["tasks"]) for cohort in cohorts)
    with tempfile.TemporaryDirectory(prefix="simulator-report-accepted-") as temporary:
        for index, (cohort, path, run) in enumerate(cases):
            case = Path(temporary) / str(index)
            case.mkdir()
            shutil.copyfile(path, case / "run.result.json")
            if run.get("camera_sample"):
                sample = Path(run["camera_sample"])
                require(not sample.is_absolute() and ".." not in sample.parts, f"Unsafe RGB path: {path}")
                image = cohort["cameras"].get(path, path.parent / sample)
                require(image.is_file() and image.suffix == ".png", f"Missing RGB: {path}")
                target = case / sample
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(image, target)
        build_report(Path(temporary), output)
        export_figures(Path(temporary), output / "figures")
    data = output / "data"
    published_runs = {run["sha256"]: run for run in json.loads((data / "measurements.json").read_text())["runs"]}
    rows = [
        {
            "cohort": cohort["name"],
            **{field: run["config"][field] for field in FIELDS},
            **flatten(run["metrics"], "metrics."),
            "process_wall_s": run.get("process_wall_s"),
        }
        for cohort in cohorts
        for _, run in cohort["accepted"]
    ]
    columns = ["cohort", *FIELDS] + sorted(set().union(*(row.keys() for row in rows)) - {"cohort", *FIELDS})
    with (data / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    excluded, portable_cohorts = [], []
    validation_gallery = "<details><summary>Independent validation frames and JSON</summary>"
    for cohort in cohorts:
        name, validation = cohort["name"], cohort["validation"]
        accepted_paths = {path for path, _ in cohort["accepted"]}
        excluded_records = []
        for path, run in cohort["loaded"]:
            if path in accepted_paths:
                continue
            relative = f"data/excluded_attempts/{name}-{digest(path)[:16]}.json"
            target = output / relative
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
            excluded_records.append({"path": Path(relative).relative_to("data").as_posix()})
            excluded.append(
                {
                    "file": relative,
                    "cohort": name,
                    "status": run.get("status"),
                    "error": run.get("error"),
                    "config": run.get("config"),
                }
            )
        validation_root = data / "validation" / name
        for path in validation.rglob("*"):
            if path.is_file() and not path.is_symlink() and path.suffix in {".json", ".png", ".log"}:
                target = validation_root / path.relative_to(validation)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        for path, record in cohort["evidence"]:
            config = record["config"]
            label = f"{BACKEND_LABELS[config['backend']]} · {config['task']} · {config['vision']}"
            relative = (validation_root / path.relative_to(validation)).relative_to(output).as_posix()
            validation_gallery += f'<p><a href="{html.escape(relative)}">{html.escape(label)}</a></p>'
            if config["vision"] == "rgb":
                for image in sorted(path.parent.glob("*.png")):
                    relative = (validation_root / image.relative_to(validation)).relative_to(output).as_posix()
                    validation_gallery += (
                        f'<a href="{html.escape(relative)}" title="{html.escape(image.stem)}">'
                        f'<img src="{html.escape(relative)}" width="64" style="image-rendering:pixelated" '
                        f'alt="{html.escape(label + " " + image.stem)}"></a> '
                    )
        archive_name = f"source-{name}.zip"
        with zipfile.ZipFile(output / archive_name, "w", zipfile.ZIP_DEFLATED) as archive:
            for path, content in sorted(cohort["sources"].items()):
                archive.writestr(path, content)
        records = list(excluded_records)
        for path, _ in cohort["accepted"]:
            published = published_runs[digest(path)]
            record = {"path": Path(published["download"]).relative_to("data").as_posix()}
            if published.get("camera_download"):
                record["camera_sample"] = Path(published["camera_download"]).relative_to("data").as_posix()
            records.append(record)
        runtime = cohort["accepted"][0][1]["provenance"]["runtime"]
        portable_cohorts.append(
            {
                "name": name,
                "tasks": list(cohort["tasks"]),
                "source_commit": cohort["source_commit"],
                "source_archive": "../" + archive_name,
                "source_archive_sha256": digest(output / archive_name),
                "runtime_harness_sha256": cohort["fingerprint"],
                "validator_sha256": cohort["validator_hash"],
                "isaaclab_upstream_base": cohort["upstream"],
                "cuda_device_uuid": runtime["cuda_device_uuid"],
                "validation_dir": "validation/" + name,
                "records": records,
                "accepted_validators": [path.relative_to(validation).as_posix() for path, _ in cohort["evidence"]],
            }
        )
    validation_gallery += "</details>"
    write_json(data / "excluded_attempts.json", excluded)
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(sources.items()):
            archive.writestr(name, content)
    portable_manifest = {
        "schema_version": 2,
        "publisher": {
            "source_commit": args.source_commit,
            "source_archive": "../source.zip",
            "source_archive_sha256": digest(output / "source.zip"),
        },
        "cohorts": portable_cohorts,
    }
    write_json(data / "cohorts.json", portable_manifest)
    if args.include_run_archive:
        with zipfile.ZipFile(output / "run_artifacts.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for cohort in cohorts:
                if cohort["source"] is None:
                    continue
                for path, _ in cohort["loaded"]:
                    for name in ("config.json", "stdout.log", "policy.pt", "camera.png"):
                        candidate = path.parent / name
                        if candidate.is_file() and not candidate.is_symlink():
                            archive.write(
                                candidate, str(Path(cohort["name"]) / path.parent.relative_to(cohort["source"]) / name)
                            )
    page, count = re.subn(
        r'<section id="comparison-(\d+)"><h2>(.*?)</h2>(.*?)</section>',
        r'<details id="comparison-\1"><summary>\2</summary>\3</details>',
        (output / "index.html").read_text(),
        flags=re.S,
    )
    require(count == expected_groups, f"Unexpected detailed report layout: {count} groups")
    page = page.replace(
        f"{len(accepted)} successful runs · 0 failed or unsupported runs.",
        f"{len(accepted)} accepted runs · {len(excluded)} excluded attempts retained separately.",
        1,
    )
    overview = "<section><h2>Measured performance</h2>"
    for pattern in ("runtime-physics-none-*.svg", "runtime-rollout-rgb-*.svg"):
        figure = next((output / "figures").glob(pattern))
        overview += (
            f'<figure><a href="figures/{figure.name}" title="Open full size SVG">'
            f'<img src="figures/{figure.name}" style="width:100%;height:auto" '
            'alt="Matched simulator runtime and startup"></a><figcaption>'
            f'<a href="figures/{figure.name}">Open full size</a> · Medians and observed ranges; '
            "matched batches within each task and source/GPU cohort.</figcaption></figure>"
        )
    page = page.replace(
        "<section><h2>Results at a glance</h2>", overview + "</section><section><h2>Results at a glance</h2>", 1
    )
    links = (
        f'<section><h2>Reproduce and download</h2><p><a href="https://github.com/maxkra15/IsaacLab/tree/'
        f'{args.source_commit}/scripts/benchmarks/simulator_comparison">Publisher source</a> · '
        '<a href="source.zip">Publisher ZIP + BSD license</a> · '
        '<a href="data/cohorts.json">Portable cohort manifest</a> · '
        '<a href="data/measurements.csv">Measurements CSV</a> · <a href="data/provenance.json">Provenance</a> · '
        '<a href="figures/figuredata.json">Figure data</a> · '
        f'<a href="data/excluded_attempts.json">Excluded attempts ({len(excluded)})</a></p>'
        '<details><summary>Measured source and GPU cohorts</summary><div class="table"><table><thead>'
        "<tr><th>Tasks</th><th>Source revision</th><th>Physical GPU UUID</th><th>Source</th></tr></thead><tbody>"
    )
    for cohort in portable_cohorts:
        revision, name = cohort["source_commit"], cohort["name"]
        links += (
            f"<tr><td>{html.escape(', '.join(cohort['tasks']))}</td><td>"
            f'<a href="https://github.com/maxkra15/IsaacLab/tree/{revision}/scripts/benchmarks/simulator_comparison">'
            f"{revision[:12]}</a></td><td>{html.escape(cohort['cuda_device_uuid'])}</td>"
            f'<td><a href="source-{name}.zip">ZIP + license</a></td></tr>'
        )
    links += (
        "</tbody></table></div><p>Each backend pair uses the same source and physical GPU. "
        "Cohorts share the recorded CPU host; comparisons apply within each task.</p></details>"
        "<p>Shared benchmark baseline: Genesis Newton, maximum 25 iterations with native tolerance; "
        "Isaac Lab Newton/MuJoCo Warp, implicitfast, 100 solver / 50 line-search iterations, tolerance 1e-6, "
        "pyramidal friction and one substep. Native renderers differ in image quality. Final-state fatal capacity "
        "flags do not cover earlier transient overflow; convergence-limit flags are retained separately. "
        'These settings do not establish equal physical accuracy.</p><div class="table"><table><tbody>'
    )
    for figure in sorted((output / "figures").glob("*.svg")):
        links += (
            f"<tr><td>{html.escape(figure.stem)}</td>"
            + "".join(
                f'<td><a href="figures/{figure.stem}.{suffix}">{suffix.upper()}</a></td>'
                for suffix in ("svg", "png", "pdf")
            )
            + "</tr>"
        )
    page = page.replace(
        '<section id="methods">',
        links
        + "</tbody></table></div>"
        + validation_gallery
        + '<p><a href="SHA256SUMS">SHA-256 checksums</a></p></section><section id="methods">',
        1,
    )
    (output / "index.html").write_text(page)
    provenance = {
        "schema_version": 2,
        "assembled_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "publisher": portable_manifest["publisher"],
        "cohorts": portable_cohorts,
        "accepted_runs": len(accepted),
        "matched_groups": expected_groups,
        "recorded_source_revisions": UPSTREAM,
        "excluded_attempts": len(excluded),
        "shared_host": accepted[0][1]["provenance"]["hardware"],
        "derivatives_sha256": {
            str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*")) if path.is_file()
        },
    }
    write_json(data / "provenance.json", provenance)
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{digest(path)}  {path.relative_to(output)}\n"
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "SHA256SUMS"
        )
    )
    print(output / "index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output_dir", "harness_dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--input_dir", type=Path)
    parser.add_argument("--source_commit", required=True, help="Committed publisher source revision")
    parser.add_argument("--validation_dir", type=Path)
    parser.add_argument("--cohort_manifest", type=Path)
    parser.add_argument("--include_run_archive", action="store_true")
    assemble_report(parser.parse_args())
