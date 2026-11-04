# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a serial, resumable matrix across isolated simulator interpreters."""

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def active_compute(excluded_group: int | None = None) -> list[dict]:
    """Sample active CUDA processes, excluding this benchmark's process group."""
    output = command_output(["nvidia-smi", "pmon", "-c", "1", "-s", "u"])
    groups = command_output(["ps", "-eo", "pid=,pgid="])
    own_pids = {
        int(row.split()[0])
        for row in groups.splitlines()
        if len(row.split()) == 2 and int(row.split()[1]) == excluded_group
    }
    active = []
    for row in output.splitlines():
        fields = row.split()
        if len(fields) < 5 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        if "C" in fields[2] and fields[3].isdigit() and int(fields[3]) > 5 and int(fields[1]) not in own_pids:
            active.append({"gpu": int(fields[0]), "pid": int(fields[1]), "sm_percent": int(fields[3])})
    return active


def monitor_gpu(process_group: int, stop: threading.Event, samples: list[dict]) -> None:
    """Retain evidence of concurrent GPU compute during an isolated run."""
    started = time.perf_counter()
    while not stop.is_set():
        samples.append({"elapsed_s": time.perf_counter() - started, "external_compute": active_compute(process_group)})
        stop.wait(3.0)


def command_output(command: list[str], cwd: Path | None = None) -> str:
    """Collect provenance without hiding missing tools."""
    process = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return process.stdout.strip() if process.returncode == 0 else process.stderr.strip()


def run_suite(args) -> int:
    """Run the requested matrix, retaining inputs, logs, and unsuccessful cases."""
    from prepare import prepare

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "suite.request.json").write_text(
        json.dumps(
            {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "argv": sys.argv,
                "parameters": vars(args),
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    assets = output / "assets"
    manifest = prepare(args.genesis_path.resolve(), assets)
    source_revisions = {
        "isaaclab": command_output(["git", "rev-parse", "HEAD"], ROOT),
        "genesis": command_output(["git", "rev-parse", "HEAD"], args.genesis_path),
    }
    hardware = {
        "platform": platform.platform(),
        "cpu": command_output(["lscpu"]),
        "gpu": command_output(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,power.limit", "--format=csv,noheader"]
        ),
        "gpu_processes_before": command_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]
        ),
    }
    jobs = []
    for backend in args.backends:
        interpreters = {
            "genesis": args.genesis_python,
            "isaaclab": args.isaaclab_python,
            **dict(value.split("=", 1) for value in args.backend_python),
        }
        if backend not in interpreters:
            raise ValueError(f"Provide --backend_python {backend}=/path/to/python")
        interpreter = Path(interpreters[backend]).absolute()
        environment = os.environ.copy()
        environment.update(
            PYGLET_HEADLESS="1",
            PYOPENGL_PLATFORM="egl",
            OMP_NUM_THREADS=str(args.cpu_threads),
            OPENBLAS_NUM_THREADS=str(args.cpu_threads),
            MKL_NUM_THREADS=str(args.cpu_threads),
            ISAACSIM_ACCEPT_EULA="Y",
            OMNI_KIT_ACCEPT_EULA="YES",
        )
        # Explicit latest package paths take precedence over installed editable mappings.
        paths = [str(p) for p in (ROOT / "source").iterdir() if p.is_dir()]
        if backend == "genesis":
            paths.insert(0, str(args.genesis_path.resolve()))
        environment["PYTHONPATH"] = os.pathsep.join(paths + [environment.get("PYTHONPATH", "")])
        # Query the actual import search path, including explicitly shared runtime
        # directories. pip/uv freeze can omit packages borrowed through a .pth file.
        freeze_script = (
            "import importlib.metadata as m; versions={}; "
            "[(versions.setdefault(d.metadata['Name'].lower().replace('_','-'), d.version)) "
            "for d in m.distributions() if d.metadata['Name']]; "
            "print('\\n'.join(f'{name}=={version}' for name,version in sorted(versions.items())))"
        )
        freeze = command_output([str(interpreter), "-c", freeze_script])
        if not freeze:
            raise RuntimeError(f"No package provenance could be collected from {interpreter}")
        (output / f"{backend}.freeze.txt").write_text(freeze + "\n")
        provenance = {
            "source_revisions": source_revisions,
            "hardware": hardware,
            "asset_manifest": manifest,
            "package_versions": freeze,
            "harness_sha256": hashlib.sha256(
                b"".join(
                    p.name.encode() + p.read_bytes()
                    for p in sorted(HERE.glob("*.py"))
                    if p.stem not in {"report", "export_figures", "validate_adapter"}
                )
            ).hexdigest(),
            "cache_policy": "process cold; disk/kernel/shader caches reused; no OS cache flush",
            "cpu_threads": args.cpu_threads,
        }
        for task in args.tasks:
            for mode in args.modes:
                visions = ["none"] if mode == "physics" else args.visions
                for vision in visions:
                    counts = args.num_envs if vision == "none" else args.vision_num_envs
                    if mode == "learning":
                        counts = [args.learning_num_envs if vision == "none" else args.vision_learning_num_envs]
                    for count in counts:
                        for seed in args.seeds:
                            config = {
                                "backend": backend,
                                "adapter_module": f"{backend}_backend",
                                "task": task,
                                "mode": mode,
                                "vision": vision,
                                "num_envs": count,
                                "seed": seed,
                                "dt": args.dt,
                                "decimation": args.decimation,
                                "device": args.device,
                                "width": args.width,
                                "height": args.height,
                                "solver_iterations": args.solver_iterations,
                                "steps": args.steps,
                                "warmup_steps": args.warmup_steps,
                                "iterations": args.iterations,
                                "rollout_steps": args.rollout_steps,
                                "evaluation_interval": args.evaluation_interval,
                                "evaluation_steps": args.evaluation_steps,
                                "initial_height": 0.55 if task == "anymal" else 0.0,
                                "joint_names": [
                                    f"{leg}_{axis}"
                                    for leg in ("LF", "RF", "LH", "RH")
                                    for axis in ("HAA", "HFE", "KFE")
                                ]
                                if task == "anymal"
                                else ["slider", "pole"],
                                "initial_joint_pos": [0, 0.4, -0.8, 0, 0.4, -0.8, 0, -0.4, 0.8, 0, -0.4, 0.8]
                                if task == "anymal"
                                else [0, 0],
                                "asset_path": str(assets / f"{task}.urdf"),
                                "cpu_threads": args.cpu_threads,
                                "env_spacing": 100.0 if vision == "rgb" else 5.0,
                                "camera_motion": "root_translation" if task == "anymal" else "fixed",
                                "camera_pos": [2, -2, 1.5] if task == "anymal" else [4, -4, 2.5],
                                "camera_lookat": [0, 0, 0.5] if task == "anymal" else [0, 0, 0.6],
                                "camera_fov": 60,
                                "provenance": provenance,
                            }
                            jobs.append((config, interpreter, environment))
    # Interleave engines and batch sizes to reduce ordering/temperature drift.
    random.Random(args.schedule_seed).shuffle(jobs)
    jobs.sort(key=lambda job: args.modes.index(job[0]["mode"]))
    results = [execute_case(config, interpreter, environment, args) for config, interpreter, environment in jobs]
    failures = sum(result["status"] != "ok" for result in results)
    (output / "suite.summary.json").write_text(
        json.dumps(
            {
                "expected_runs": len(jobs),
                "successful_runs": len(jobs) - failures,
                "unsuccessful_runs": failures,
            },
            indent=2,
        )
        + "\n"
    )
    return failures


def execute_case(config: dict, interpreter: Path, environment: dict, args) -> dict:
    """Execute one isolated case and retain failures and native teardown errors."""
    identity = {key: value for key, value in config.items() if key != "provenance"}
    identity["source_revisions"] = config["provenance"]["source_revisions"]
    identity["harness_sha256"] = config["provenance"]["harness_sha256"]
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    backend, task, mode, vision, count, seed = (
        config[k] for k in ("backend", "task", "mode", "vision", "num_envs", "seed")
    )
    case = args.output_dir.resolve() / f"{backend}-{task}-{mode}-{vision}-{count}-s{seed}-{key}"
    case.mkdir(exist_ok=True)
    result_path = case / "run.result.json"
    if result_path.exists() and not args.rerun:
        print(f"Retained {case.name}", flush=True)
        return json.loads(result_path.read_text())
    deadline = time.monotonic() + args.idle_wait_s
    while active_compute():
        if time.monotonic() >= deadline:
            raise RuntimeError("GPU remains busy; completed cases are retained. Resume when it is idle.")
        print("Waiting for active GPU compute to finish...", flush=True)
        time.sleep(5.0)
    config_path = case / "config.json"
    config["provenance"]["hardware"]["gpu_processes_before"] = command_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]
    )
    config["provenance"]["hardware"]["gpu_utilization_before"] = command_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader"]
    )
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Running {case.name}", flush=True)
    started = time.perf_counter()
    with (case / "stdout.log").open("w") as log:
        process = subprocess.Popen(
            [
                str(interpreter),
                str(HERE / "worker.py"),
                "--config",
                str(config_path),
                "--output",
                str(result_path),
            ],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        samples = []
        stop_sampling = threading.Event()
        monitor = threading.Thread(target=monitor_gpu, args=(process.pid, stop_sampling, samples), daemon=True)
        monitor.start()
        try:
            returncode = process.wait(timeout=args.timeout_s)
            status = "failed"
        except subprocess.TimeoutExpired:
            import signal

            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            returncode, status = -9, "timeout"
        finally:
            finished = time.perf_counter()
            stop_sampling.set()
            monitor.join(timeout=5.0)
    elapsed = finished - started
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if returncode != 0 and result["status"] == "ok":
            result.update(status="failed", error=f"Simulator teardown failed (exit {returncode})")
    else:
        result = {
            "schema_version": 1,
            "status": status,
            "config": config,
            "provenance": config["provenance"],
            "metrics": {},
            "error": f"Worker produced no result (exit {returncode}); inspect stdout.log",
        }
    result.update(process_wall_s=elapsed, returncode=returncode)
    result["provenance"]["gpu_activity_samples"] = samples
    result["provenance"]["other_active_compute_detected"] = any(sample["external_compute"] for sample in samples)
    if result["provenance"]["other_active_compute_detected"] and result["status"] == "ok":
        result.update(status="contended", error="Another active GPU compute process overlapped this run; repeat it.")
    result["provenance"]["hardware"]["gpu_processes_after"] = command_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]
    )
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"  {result['status']} ({elapsed:.1f}s) {result.get('error', '')}", flush=True)
    return result


def parser() -> argparse.ArgumentParser:
    """Build the CLI; adding another simulator only requires a module and interpreter."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--genesis_path", type=Path, required=True)
    result.add_argument("--genesis_python", default=str(ROOT / ".benchmark_sources/genesis_env/bin/python"))
    result.add_argument("--isaaclab_python", default=str(ROOT / ".benchmark_sources/isaaclab_env/bin/python"))
    result.add_argument("--output_dir", type=Path, required=True)
    result.add_argument("--backends", nargs="+", default=["genesis", "isaaclab"])
    result.add_argument("--backend_python", action="append", default=[], metavar="NAME=PATH")
    result.add_argument("--tasks", nargs="+", choices=["cartpole", "anymal"], default=["cartpole", "anymal"])
    result.add_argument(
        "--modes", nargs="+", choices=["physics", "rollout", "learning"], default=["physics", "rollout", "learning"]
    )
    result.add_argument("--visions", nargs="+", choices=["none", "rgb"], default=["none", "rgb"])
    result.add_argument("--num_envs", nargs="+", type=int, default=[1, 256, 4096])
    result.add_argument("--vision_num_envs", nargs="+", type=int, default=[1, 64, 256])
    result.add_argument("--learning_num_envs", type=int, default=1024)
    result.add_argument("--vision_learning_num_envs", type=int, default=64)
    result.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    result.add_argument("--steps", type=int, default=500)
    result.add_argument("--warmup_steps", type=int, default=50)
    result.add_argument("--iterations", type=int, default=200)
    result.add_argument("--rollout_steps", type=int, default=24)
    result.add_argument("--dt", type=float, default=0.005)
    result.add_argument("--decimation", type=int, default=4)
    result.add_argument("--solver_iterations", type=int, default=50)
    result.add_argument("--evaluation_interval", type=int, default=20)
    result.add_argument("--evaluation_steps", type=int, default=250)
    result.add_argument("--width", type=int, default=64)
    result.add_argument("--height", type=int, default=64)
    result.add_argument("--cpu_threads", type=int, default=8)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--timeout_s", type=int, default=7200)
    result.add_argument("--idle_wait_s", type=int, default=3600)
    result.add_argument("--schedule_seed", type=int, default=1729)
    result.add_argument("--rerun", action="store_true")
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    numeric = (
        args.num_envs
        + args.vision_num_envs
        + [
            args.steps,
            args.iterations,
            args.cpu_threads,
            args.learning_num_envs,
            args.vision_learning_num_envs,
            args.evaluation_steps,
            args.evaluation_interval,
            args.rollout_steps,
            args.decimation,
            args.solver_iterations,
            args.dt,
        ]
    )
    if min(numeric) <= 0 or args.warmup_steps < 0:
        raise SystemExit("Counts must be positive and warmup_steps nonnegative")
    raise SystemExit(1 if run_suite(args) else 0)
