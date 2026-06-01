#!/usr/bin/env python3
"""Run short waterhose teleop scaling benchmarks and write JSONL results."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path


PROFILE_RE = re.compile(
    r"\[PROFILE\]\s+steps=(?P<steps>\d+)\s+sim_time=(?P<sim_time>[0-9.]+)s\s+"
    r"setup_time=(?P<setup_time>[0-9.]+)s\s+rollout_time=(?P<rollout_time>[0-9.]+)s\s+"
    r"wall_time=(?P<wall_time>[0-9.]+)s\s+rtf=(?P<rtf>[0-9.]+)\s+"
    r"steps_per_s=(?P<steps_per_s>[0-9.]+)\s+cuda_graph=(?P<cuda_graph>\w+)"
)


def query_gpu_memory_mib() -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    values = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(line.split(",")[0].strip()))
        except ValueError:
            pass
    return max(values) if values else None


class GpuMemorySampler:
    def __init__(self, interval_s: float = 0.25):
        self.interval_s = interval_s
        self.baseline_mib = query_gpu_memory_mib()
        self.peak_mib = self.baseline_mib
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "GpuMemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = query_gpu_memory_mib()
            if value is not None:
                self.peak_mib = value if self.peak_mib is None else max(self.peak_mib, value)
            self._stop.wait(self.interval_s)

    @property
    def delta_mib(self) -> int | None:
        if self.baseline_mib is None or self.peak_mib is None:
            return None
        return max(0, self.peak_mib - self.baseline_mib)


def run_one(repo_root: Path, num_envs: int, max_steps: int, timeout_s: float, device: str) -> dict:
    cmd = [
        str(repo_root / "isaaclab.sh"),
        "-p",
        "scripts/environments/waterhose/run_robot_demo.py",
        "--task",
        "Isaac-Waterhose-Robot-Demo-v0",
        "--mode",
        "teleop",
        "--teleop_device",
        "spacemouse",
        "--vis",
        "none",
        "--num_envs",
        str(num_envs),
        "--max_steps",
        str(max_steps),
        "--profile",
        "--device",
        device,
    ]
    env = os.environ.copy()
    env.setdefault("PXR_WORK_THREAD_LIMIT", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")

    started = time.perf_counter()
    with GpuMemorySampler() as memory:
        try:
            completed = subprocess.run(
                cmd,
                cwd=repo_root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
            output = completed.stdout
            returncode = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            output = stdout + stderr
            returncode = None
            timed_out = True

    elapsed = time.perf_counter() - started
    result = {
        "num_envs": num_envs,
        "max_steps": max_steps,
        "command": shlex.join(cmd),
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_s": elapsed,
        "gpu_memory_baseline_mib": memory.baseline_mib,
        "gpu_memory_peak_mib": memory.peak_mib,
        "gpu_memory_delta_mib": memory.delta_mib,
        "profile": None,
        "error_tail": "\n".join(output.splitlines()[-80:]),
    }

    match = PROFILE_RE.search(output)
    if match:
        profile = match.groupdict()
        result["profile"] = {
            "steps": int(profile["steps"]),
            "sim_time_s": float(profile["sim_time"]),
            "setup_time_s": float(profile["setup_time"]),
            "rollout_time_s": float(profile["rollout_time"]),
            "wall_time_s": float(profile["wall_time"]),
            "control_hz": float(profile["steps_per_s"]),
            "rtf": float(profile["rtf"]),
            "cuda_graph": profile["cuda_graph"],
        }
        result["profile"]["env_steps_per_s"] = result["profile"]["control_hz"] * num_envs
        result["profile"]["step_time_ms"] = 1000.0 / max(result["profile"]["control_hz"], 1.0e-12)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("results.jsonl"))
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--envs", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as f:
        for num_envs in args.envs:
            print(f"[bench] num_envs={num_envs}", flush=True)
            result = run_one(args.repo_root.resolve(), num_envs, args.max_steps, args.timeout_s, args.device)
            f.write(json.dumps(result, sort_keys=True) + "\n")
            f.flush()
            status = "ok" if result["profile"] and result["returncode"] == 0 else "failed"
            print(f"[bench] num_envs={num_envs} {status}", flush=True)
            if result["timed_out"]:
                print(f"[bench] timeout at num_envs={num_envs}; stopping sweep", flush=True)
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
