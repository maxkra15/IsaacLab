# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Profile native RSL-RL PPO with synchronized iteration timings and JSON metrics.

Accepts the native RSL-RL training flags, for example::

    uv run python scripts/benchmarks/profile_rsl_ppo.py --task Isaac-Cartpole \
        physics=newton_mjwarp --num_envs 1024 --max_iterations 150

The native learner, environment wrapper, logger, and checkpointing are retained.
GPU fences delimit collection and learning; logging/checkpoint I/O is excluded
from those durations. Episode statistics retain RSL-RL's last-100-episode window.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.entrypoints.backends import train_rsl_rl


class TimedOnPolicyRunner(OnPolicyRunner):
    """Instrument the native runner's existing algorithm and logger boundaries."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._profile_device = torch.device(self.device)
        self._profile_rollout_start = 0.0
        self._profile_update_start = 0.0
        self._profile_steps = 0
        self._profile_rollout_seconds = 0.0
        self._profile_update_seconds = 0.0
        act, compute_returns, update, log = self.alg.act, self.alg.compute_returns, self.alg.update, self.logger.log

        def timed_act(*args, **kwargs):
            if self._profile_steps == 0:
                self._synchronize()
                self._profile_rollout_start = time.perf_counter()
            self._profile_steps += 1
            return act(*args, **kwargs)

        def timed_compute_returns(*args, **kwargs):
            if self._profile_steps != self.cfg["num_steps_per_env"]:
                raise RuntimeError("The profiled rollout does not match the configured horizon.")
            self._synchronize()
            self._profile_update_start = time.perf_counter()
            self._profile_rollout_seconds = self._profile_update_start - self._profile_rollout_start
            return compute_returns(*args, **kwargs)

        def timed_update(*args, **kwargs):
            result = update(*args, **kwargs)
            self._synchronize()
            self._profile_update_seconds = time.perf_counter() - self._profile_update_start
            return result

        def timed_log(**kwargs):
            kwargs["collect_time"] = self._profile_rollout_seconds
            kwargs["learn_time"] = self._profile_update_seconds
            log(**kwargs)
            self._write_metrics(kwargs)
            self._profile_steps = 0

        self.alg.act = timed_act
        self.alg.compute_returns = timed_compute_returns
        self.alg.update = timed_update
        self.logger.log = timed_log
        if self.logger.log_dir is not None:
            profile_dir = Path(self.logger.log_dir)
            profile_dir.mkdir(parents=True, exist_ok=True)
            (profile_dir / "profile.json").write_text(
                json.dumps(
                    {
                        "timing": "CUDA-synchronized rollout and update wall times",
                        "episode_statistic": "last_100_completed_episodes",
                        "actor_parameters": sum(parameter.numel() for parameter in self.alg.actor.parameters()),
                        "critic_parameters": sum(parameter.numel() for parameter in self.alg.critic.parameters()),
                        "num_envs": self.env.num_envs,
                        "horizon": self.cfg["num_steps_per_env"],
                    },
                    indent=2,
                )
                + "\n"
            )

    def _write_metrics(self, information: dict) -> None:
        if self.logger.log_dir is None:
            return
        iteration = information["it"] + 1
        steps = self.env.num_envs * self.cfg["num_steps_per_env"]
        updates = self.alg.num_learning_epochs * self.alg.num_mini_batches
        seconds = self._profile_rollout_seconds + self._profile_update_seconds
        metrics = {
            "timestamp_unix": time.time(),
            "iteration": iteration,
            "algorithm": "rsl_rl_ppo",
            "total_steps": steps * iteration,
            "gradient_updates": updates * iteration,
            "actor_gradient_updates": updates * iteration,
            "critic_gradient_updates": updates * iteration,
            "updates_this_iteration": updates,
            "rollout_seconds": self._profile_rollout_seconds,
            "update_seconds": self._profile_update_seconds,
            "iteration_seconds": seconds,
            "fps": steps / seconds,
            "warmup_seconds": 0.0,
            "completed_episodes": None,
            "episode_statistic": "last_100_completed_episodes",
            "mean_episode_return": statistics.mean(self.logger.rewbuffer) if self.logger.rewbuffer else None,
            "mean_episode_length": statistics.mean(self.logger.lenbuffer) if self.logger.lenbuffer else None,
            "losses": {key: float(value) for key, value in information["loss_dict"].items()},
            "torch_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(self._profile_device) if self._profile_device.type == "cuda" else 0
            ),
        }
        with (Path(self.logger.log_dir) / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")

    def _synchronize(self) -> None:
        if self._profile_device.type == "cuda":
            torch.cuda.synchronize(self._profile_device)


def main() -> None:
    """Run the native entrypoint with instrumentation scoped to this process."""
    original = train_rsl_rl.OnPolicyRunner
    train_rsl_rl.OnPolicyRunner = TimedOnPolicyRunner
    try:
        train_rsl_rl.run(sys.argv[1:])
    finally:
        train_rsl_rl.OnPolicyRunner = original


if __name__ == "__main__":
    main()
