# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gate the final 84-run shared MJWarp matrix and assemble an offline Pages report for review."""

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


def committed_sources(harness: Path, commit: str) -> tuple[dict[str, bytes], str, str]:
    """Read exact committed source files, runtime fingerprint and upstream base."""
    require(re.fullmatch("[0-9a-f]{40}", commit), "source_commit must be a full 40-character commit SHA")
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=harness, text=True).strip())
    relative = harness.relative_to(repo)
    raw = subprocess.check_output(["git", "archive", commit, str(relative), "LICENSE"], cwd=repo)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        files = {member.name: archive.extractfile(member).read() for member in archive if member.isfile()}
    runtime = [
        (Path(name).name, content)
        for name, content in files.items()
        if Path(name).parent == relative
        and Path(name).suffix == ".py"
        and Path(name).stem not in {"report", "export_figures", "validate_adapter", "assemble_report"}
    ]
    fingerprint = hashlib.sha256(b"".join(name.encode() + content for name, content in sorted(runtime))).hexdigest()
    upstream = subprocess.check_output(["git", "merge-base", commit, "origin/main"], cwd=repo, text=True).strip()
    return files, fingerprint, upstream


def expected_cases() -> set[tuple[str, str, str, str, int, int]]:
    """Return the complete publication matrix of backend/task/workload/seed tuples."""
    workloads = [("physics", "none", n) for n in (256, 4096)]
    workloads += [("rollout", "none", n) for n in (256, 4096)] + [("rollout", "rgb", 64)]
    workloads += [("learning", vision, 512) for vision in ("none", "rgb")]
    return {
        (b, t, m, v, n, s)
        for b in ("genesis", "isaaclab")
        for t in ("anymal", "cartpole")
        for m, v, n in workloads
        for s in (0, 1, 2)
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


def validate_runs(loaded: list[tuple[Path, dict[str, Any]]], fingerprint: str) -> list[tuple[Path, dict[str, Any]]]:
    """Require the complete matched matrix with consistent code, hardware and runtime."""
    from report import group_results

    accepted = [(path, run) for path, run in loaded if run.get("status") == "ok"]
    cases = Counter(tuple(run["config"].get(field) for field in FIELDS) for _, run in accepted)
    require(
        len(accepted) == 84 and set(cases) == expected_cases() and max(cases.values(), default=0) == 1,
        f"Expected 84 unique successful cases; got {len(accepted)}. Missing={expected_cases() - set(cases)}; "
        f"unexpected={set(cases) - expected_cases()}; duplicates={[case for case, n in cases.items() if n > 1]}",
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
            require(
                curve
                and curve[0]["iteration"] == 0
                and curve[-1]["iteration"] == 500
                and curve[-1]["env_steps"] == 500 * 24 * 512
                and metrics.get("evaluation", {}).get("steps") == 250,
                f"Incomplete learning/evaluation evidence: {path}",
            )
    require(len(devices) == 1, f"Different physical GPUs: {devices}")
    groups = group_results([run for _, run in accepted])
    require(len(groups) == 14 and all(len(values) == 6 for _, values in groups), "Expected 14 matched six-run groups")
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
) -> list[tuple[Path, dict[str, Any]]]:
    """Require eight successful validators for the final runtime, validator source and physical inputs."""
    expected = {
        (backend, task, vision)
        for backend in ("genesis", "isaaclab")
        for task in ("anymal", "cartpole")
        for vision in ("none", "rgb")
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
        require(
            all(recorded_assets.get(field) == assets.get(field) for field in ("source_urdf_sha256", "meshes_sha256")),
            f"Validation physical assets differ: {path}",
        )
        required = {"finite_state", "joint_order", "subset_reset", "moving_mass"}
        required |= (
            {"anymal_short_stability"}
            if key[1] == "anymal"
            else {"cartpole_independent_dynamics", "cartpole_force_direction"}
        )
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
    require(set(selected) == expected, f"Expected eight final validators; missing={expected - set(selected)}")
    rgb_directories = [path.parent for path, record in selected.values() if record["config"]["vision"] == "rgb"]
    require(len(set(rgb_directories)) == 4, "RGB validators need separate directories to preserve their frames")
    require(
        all(list(directory.glob("validation_camera_env*.png")) for directory in rgb_directories),
        "Missing independent RGB validation frames",
    )
    return list(selected.values())


def assemble_report(args: argparse.Namespace) -> None:
    """Build a reviewable report from complete final measurements and validation evidence."""
    harness, source, output = args.harness_dir.resolve(), args.input_dir.resolve(), args.output_dir.resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(harness))
    from export_figures import export_figures
    from report import BACKEND_LABELS, build_report

    sources, fingerprint, upstream = committed_sources(harness, args.source_commit)
    for name in ("report.py", "export_figures.py", "assemble_report.py", "validate_adapter.py"):
        committed = [content for path, content in sources.items() if Path(path).name == name]
        require(
            len(committed) == 1 and (harness / name).read_bytes() == committed[0],
            f"Publication helper differs from source_commit: {name}",
        )
    loaded = [(path, json.loads(path.read_text())) for path in sorted(source.rglob("*.result.json"))]
    accepted = validate_runs(loaded, fingerprint)
    require(not output.exists() or not any(output.iterdir()), "Output must be empty to avoid stale artifacts")
    require(args.validation_dir and args.validation_dir.is_dir(), "Eight final validators require validation_dir")
    validator = next(content for name, content in sources.items() if Path(name).name == "validate_adapter.py")
    validator_hash = hashlib.sha256(validator).hexdigest()
    evidence = validate_evidence(
        args.validation_dir, fingerprint, validator_hash, accepted[0][1]["provenance"].get("asset_manifest", {})
    )
    with tempfile.TemporaryDirectory(prefix="simulator-report-accepted-") as temporary:
        for index, (path, run) in enumerate(accepted):
            case = Path(temporary) / str(index)
            case.mkdir()
            shutil.copyfile(path, case / "run.result.json")
            if run.get("camera_sample"):
                image = path.parent / run["camera_sample"]
                require(
                    image.resolve().is_relative_to(path.parent.resolve()) and image.is_file(),
                    f"Unsafe/missing RGB: {path}",
                )
                shutil.copyfile(image, case / run["camera_sample"])
        build_report(Path(temporary), output)
        export_figures(Path(temporary), output / "figures")
    data = output / "data"
    rows = [
        {
            **{field: run["config"][field] for field in FIELDS},
            **flatten(run["metrics"], "metrics."),
            "process_wall_s": run.get("process_wall_s"),
        }
        for _, run in accepted
    ]
    columns = list(FIELDS) + sorted(set().union(*(row.keys() for row in rows)) - set(FIELDS))
    with (data / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    excluded = []
    for path, run in loaded:
        if run.get("status") != "ok":
            relative = f"data/excluded_attempts/{digest(path)[:16]}.json"
            target = output / relative
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
            excluded.append(
                {"file": relative, "status": run.get("status"), "error": run.get("error"), "config": run.get("config")}
            )
    write_json(data / "excluded_attempts.json", excluded)
    if args.validation_dir:
        for path in args.validation_dir.rglob("*"):
            if path.is_file() and not path.is_symlink() and path.suffix in {".json", ".png", ".log"}:
                target = data / "validation" / path.relative_to(args.validation_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    validation_gallery = "<details><summary>Independent validation frames and JSON</summary>"
    for path, record in evidence:
        config = record["config"]
        label = f"{BACKEND_LABELS[config['backend']]} · {config['task']} · {config['vision']}"
        relative = (Path("data/validation") / path.relative_to(args.validation_dir)).as_posix()
        validation_gallery += f'<p><a href="{html.escape(relative)}">{html.escape(label)}</a></p>'
        if config["vision"] == "rgb":
            for image in sorted(path.parent.glob("*.png")):
                relative = (Path("data/validation") / image.relative_to(args.validation_dir)).as_posix()
                validation_gallery += (
                    f'<a href="{html.escape(relative)}" title="{html.escape(image.stem)}">'
                    f'<img src="{html.escape(relative)}" width="64" style="image-rendering:pixelated" '
                    f'alt="{html.escape(label + " " + image.stem)}"></a> '
                )
    validation_gallery += "</details>"
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(sources.items()):
            archive.writestr(name, content)
    if args.include_run_archive:
        with zipfile.ZipFile(output / "run_artifacts.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path, _ in loaded:
                for name in ("config.json", "stdout.log", "policy.pt", "camera.png"):
                    candidate = path.parent / name
                    if candidate.is_file() and not candidate.is_symlink():
                        archive.write(candidate, str(path.parent.relative_to(source) / name))
    page, count = re.subn(
        r'<section id="comparison-(\d+)"><h2>(.*?)</h2>(.*?)</section>',
        r'<details id="comparison-\1"><summary>\2</summary>\3</details>',
        (output / "index.html").read_text(),
        flags=re.S,
    )
    require(count == 14, f"Unexpected detailed report layout: {count} groups")
    page = page.replace(
        "84 successful runs · 0 failed or unsupported runs.",
        f"84 accepted runs · {len(excluded)} excluded attempts retained separately.",
        1,
    )
    overview = "<section><h2>Measured performance</h2>"
    for pattern in ("runtime-physics-none-*.svg", "runtime-rollout-rgb-*.svg"):
        figure = next((output / "figures").glob(pattern))
        overview += (
            f'<figure><img src="figures/{figure.name}" style="width:100%;height:auto" '
            'alt="Matched simulator runtime and startup"><figcaption>Medians and observed ranges; '
            "shared MJWarp benchmark baseline, identical batch sizes and common task code.</figcaption></figure>"
        )
    page = page.replace(
        "<section><h2>Results at a glance</h2>", overview + "</section><section><h2>Results at a glance</h2>", 1
    )
    links = (
        f'<section><h2>Reproduce and download</h2><p><a href="https://github.com/maxkra15/IsaacLab/tree/'
        f'{args.source_commit}/scripts/benchmarks/simulator_comparison">Exact source commit</a> · '
        '<a href="source.zip">Source ZIP + BSD license</a> · <a href="data/measurements.csv">Measurements CSV</a> · '
        '<a href="data/provenance.json">Provenance</a> · <a href="figures/figuredata.json">Figure data</a> · '
        f'<a href="data/excluded_attempts.json">Excluded attempts ({len(excluded)})</a></p>'
        "<p>Shared benchmark baseline: "
        "Genesis Newton, maximum 25 iterations with native tolerance; Isaac Lab Newton/MuJoCo Warp, implicitfast, "
        "100 solver / 50 line-search iterations, tolerance 1e-6, pyramidal friction and one substep. "
        "Newton Warp and Madrona renderers have different image quality. Final-state fatal capacity flags do not cover "
        "earlier transient overflow; convergence-limit flags are retained separately. "
        "These settings do not establish equal physical accuracy.</p>"
        '<div class="table"><table><tbody>'
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
        links + "</tbody></table></div>" + validation_gallery + '<p><a href="SHA256SUMS">'
        'SHA-256 checksums</a></p></section><section id="methods">',
        1,
    )
    (output / "index.html").write_text(page)
    provenance = {
        "schema_version": 1,
        "assembled_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "runtime_harness_sha256": fingerprint,
        "recorded_source_revisions": accepted[0][1]["provenance"]["source_revisions"],
        "isaaclab_upstream_base": {"sha": upstream, "method": "git merge-base source_commit origin/main"},
        "accepted_runs": 84,
        "validator_sha256": validator_hash,
        "accepted_validators": [str(path.relative_to(args.validation_dir)) for path, _ in evidence],
        "excluded_attempts": len(excluded),
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
    for name in ("input_dir", "output_dir", "harness_dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source_commit", required=True)
    parser.add_argument("--validation_dir", type=Path, required=True)
    parser.add_argument("--include_run_archive", action="store_true")
    assemble_report(parser.parse_args())
