# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Audit the current-main G1 cohort and derive a summary without editing raw inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ALGORITHMS = ("rsl_rl_ppo", "warp_ppo", "flashsac")
CHECKPOINTS = (0, 250, 500, 1000, 1500, 2050)
SCENARIOS = ("native_commands", "forward_0_5")


def summarize(paths: list[Path], output: Path, algorithms: list[str]) -> None:
    """Require common budgets, MDPs and evaluated checkpoints for each requested learner."""
    expected = {(algorithm, seed) for algorithm in algorithms for seed in (0, 1, 2)}
    runs, keys, mdps, contracts = [], set(), set(), set()
    for path in paths:
        raw = path.read_bytes()
        document = json.loads(raw)
        metadata = document["metadata"]
        if metadata["protocol"] != "g1_main_learning_baselines_20261020_v1":
            raise ValueError(f"Unexpected experiment protocol: {path}.")
        if len(document["runs"]) != 1:
            raise ValueError(f"Each input must contain exactly one run: {path}.")
        run = document["runs"][0]
        key = run["algorithm"], run["seed"]
        if key in keys or key not in expected:
            raise ValueError(f"Unexpected or duplicated learner/seed: {key}.")
        keys.add(key)
        if run["status"] != "completed" or run["return_code"] != 0:
            raise ValueError(f"Training did not complete: {path}.")
        budget = run["iterations"], run["num_envs"], run["horizon"], run["transitions"]
        if budget != (2050, 1024, 24, 50_380_800):
            raise ValueError(f"Unexpected actual collection budget: {budget}.")
        evidence = run["learning_evidence"]
        if not evidence["weights_updated"] or not evidence["weights_and_optimizer_finite"]:
            raise ValueError(f"Finite updated parameters were not verified: {path}.")
        if run["algorithm"] in {"rsl_rl_ppo", "warp_ppo"}:
            counts = run["actor_gradient_updates"], run["critic_gradient_updates"], evidence["optimizer_steps"]
            if counts != (41_000, 41_000, 41_000):
                raise ValueError(f"PPO update budget differs from stock five epochs/four minibatches: {counts}.")
        points = run["evaluations"]
        wanted = {(scenario, iteration) for scenario in SCENARIOS for iteration in CHECKPOINTS}
        if len(points) != len(wanted) or {(p["scenario"], p["iteration"]) for p in points} != wanted:
            raise ValueError(f"Missing or duplicated common evaluations: {path}.")
        for point in points:
            if point["evaluation_envs"] != 64 or point["evaluation_seed"] != 10_000 + run["seed"]:
                raise ValueError(f"Common evaluation worlds/seed mismatch: {path}.")
            if point["transitions"] != point["iteration"] * 1024 * 24:
                raise ValueError(f"Evaluation checkpoint budget mismatch: {path}.")
            for metric in ("return_mean", "survival_rate", "linear_velocity_error_mean", "upright_fraction"):
                if not math.isfinite(point[metric]):
                    raise ValueError(f"Nonfinite evaluation metric {metric}: {path}.")
        mdps.add(metadata["mdp_sha256"])
        contract = {
            name: run["runtime_contract"][name] for name in ("observation_dim", "critic_observation_dim", "action_dim")
        }
        contracts.add(json.dumps(contract, sort_keys=True))
        forward = sorted((p for p in points if p["scenario"] == "forward_0_5"), key=lambda p: p["iteration"])
        first_pass = next((p for p in forward if p["walking_gate_passed"]), None)
        runs.append(
            {
                "algorithm": run["algorithm"],
                "seed": run["seed"],
                "comparison_path": str(path.resolve()),
                "comparison_sha256": hashlib.sha256(raw).hexdigest(),
                "source_pins": {
                    "isaaclab": metadata["isaaclab_revision"],
                    "robolearn": metadata["robolearn_revision"],
                },
                "versions": metadata["versions"],
                "training_seconds": run["training_seconds"],
                "rollout_seconds": run["rollout_seconds"],
                "update_seconds": run["update_seconds"],
                "process_wall_seconds": run["process_wall_seconds"],
                "steady_fps": run["steady_fps"],
                "actor_gradient_updates": run["actor_gradient_updates"],
                "critic_gradient_updates": run["critic_gradient_updates"],
                "learning_evidence": evidence,
                "first_observed_forward_pass_iteration": first_pass["iteration"] if first_pass else None,
                "common_evaluations": points,
                "final_quality": {
                    scenario: next(p for p in points if p["scenario"] == scenario and p["iteration"] == 2050)
                    for scenario in SCENARIOS
                },
            }
        )
    if keys != expected or len(mdps) != 1 or len(contracts) != 1:
        raise ValueError("The cohort is incomplete or has differing training MDPs/runtime dimensions.")
    summary = {
        "status": "PASS",
        "protocol": "g1_main_learning_baselines_20261020_v1",
        "run_count": len(runs),
        "evaluation_count": len(runs) * 12,
        "transitions_per_run": 50_380_800,
        "mdp_sha256": next(iter(mdps)),
        "runtime_contract": json.loads(next(iter(contracts))),
        "notes": [
            "Initial policies add two common evaluations per run; the old cohort did not include them.",
            "First observed walking passes are sparse checkpoint observations, not exact convergence times.",
            "Current-main physics and dependencies changed; cross-date differences do not isolate a learner change.",
            "MDPs execute in Torch; physics and learner graphs remain separate.",
            "Three seeds describe variation and do not establish statistical significance.",
        ],
        "runs": sorted(runs, key=lambda run: (run["algorithm"], run["seed"])),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"PASS: {len(runs)} runs, {summary['evaluation_count']} common evaluations; {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    args = parser.parse_args()
    summarize(args.input, args.output, args.algorithms)


if __name__ == "__main__":
    main()
