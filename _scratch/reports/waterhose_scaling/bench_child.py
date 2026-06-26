# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Single waterhose-env throughput measurement (one solver x one env count).

Boots the coupled task kitless (exactly like ``run_robot_demo.py --visualizer none``),
drives it with the scripted grasp-and-insert policy, runs a warm-up window (which pays the
one-time coupled-solver CUDA-graph capture), then times a measured window with explicit
``wp.synchronize()`` fences. Checks the Newton body state for NaN so a batched run that
silently diverges is reported as ``nan`` rather than a fast (wrong) number. Writes a JSON
record consumed by ``bench_sweep.py``.

    ./isaaclab.sh -p _scratch/reports/waterhose_scaling/bench_child.py \
        --task Isaac-Waterhose-Coupled-v0 --num_envs 8 --warmup 60 --steps 200 \
        --headless --visualizer none --out /tmp/run.json
"""

from __future__ import annotations

import argparse
import json
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Waterhose single-config throughput bench.")
parser.add_argument("--task", type=str, default="Isaac-Waterhose-Coupled-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--warmup", type=int, default=60)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--settle_time", type=float, default=0.3)
parser.add_argument("--no_graph", action="store_true", help="Disable the CUDA graph (eager) for comparison.")
parser.add_argument("--out", type=str, required=True, help="Path to write the JSON record.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Kitless: this Newton-backed, camera-free task runs without Omniverse Kit. Pass
# ``--headless --visualizer none`` on the command line so argparse types the value correctly.
args.headless = True


def _write(record: dict) -> None:
    with open(args.out, "w") as handle:
        json.dump(record, handle, indent=2)


record: dict = {
    "task": args.task,
    "num_envs": args.num_envs,
    "warmup_steps": args.warmup,
    "timed_steps": args.steps,
    "status": "crashed",
}

try:
    import torch  # noqa: PLC0415

    import isaaclab_tasks  # noqa: F401, PLC0415
    from isaaclab.app import launch_simulation  # noqa: PLC0415
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: PLC0415

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.scene.num_envs = args.num_envs
    if args.no_graph:
        env_cfg.sim.physics.use_cuda_graph = False
    # Scripted demo: drop the resetting terminations so the measured window never teleports
    # an env home mid-rollout (matches run_robot_demo.py for the scripted tasks).
    for name in ("success", "time_out"):
        if hasattr(env_cfg.terminations, name):
            setattr(env_cfg.terminations, name, None)

    t_setup = time.perf_counter()
    with launch_simulation(env_cfg, args):
        import gymnasium as gym  # noqa: PLC0415
        import warp as wp  # noqa: PLC0415
        from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415

        from isaaclab_tasks.contrib.waterhose.scripted_state_machine import create_scripted_policy  # noqa: PLC0415

        env = gym.make(args.task, cfg=env_cfg).unwrapped
        env.reset()
        sm = create_scripted_policy(env, settle_time=args.settle_time, debug=False)
        setup_s = time.perf_counter() - t_setup

        # Warm-up window: triggers the one-time graph capture and reaches the contact-rich
        # grasp/carry region so the measured window reflects steady manipulation cost.
        for _ in range(args.warmup):
            env.step(sm.compute(env))
        wp.synchronize()

        graph_active = getattr(NewtonManager, "_graph", None) is not None

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            env.step(sm.compute(env))
        wp.synchronize()
        measured_s = max(time.perf_counter() - t0, 1e-12)

        body_q = wp.to_torch(NewtonManager.get_state_0().body_q)
        has_nan = bool(torch.isnan(body_q).any().item()) or bool((~torch.isfinite(body_q)).any().item())
        torch_peak_gib = torch.cuda.max_memory_allocated() / (1024**3)

        steps_per_s = args.steps / measured_s
        record.update(
            {
                "status": "nan" if has_nan else "ok",
                "step_ms": 1000.0 * measured_s / args.steps,
                "steps_per_s_per_env": steps_per_s,
                "env_steps_per_s": steps_per_s * args.num_envs,
                "startup_s": setup_s,
                "measured_s": measured_s,
                "torch_peak_gib": torch_peak_gib,
                "cuda_graph_active": graph_active,
            }
        )
        env.close()
    _write(record)
    print(f"[bench_child] {args.task} n={args.num_envs}: {record['status']} "
          f"{record.get('step_ms', float('nan')):.1f} ms/step "
          f"{record.get('env_steps_per_s', float('nan')):.1f} env-steps/s", flush=True)
except BaseException as exc:  # noqa: BLE001
    record["error"] = f"{type(exc).__name__}: {exc}"[:400]
    _write(record)
    print(f"[bench_child] FAILED {args.task} n={args.num_envs}: {record['error']}", flush=True)
    raise
