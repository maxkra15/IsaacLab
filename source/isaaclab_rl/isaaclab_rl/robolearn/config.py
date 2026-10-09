# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-independent configuration for the optional RoboLearn runners."""

from typing import Any, Literal

from isaaclab.utils.configclass import configclass


@configclass
class RoboLearnRunnerCfg:
    """Configure FlashSAC or Warp-NN PPO without importing either learning dependency."""

    algorithm: Literal["flashsac", "warp_ppo"] = "flashsac"
    seed: int = 0
    device: str = "cuda:0"
    num_steps_per_env: int = 16
    """Environment steps collected from every environment per learning iteration."""
    max_iterations: int = 150
    save_interval: int = 50
    experiment_name: str = "experiment"
    run_name: str = ""
    observation_group: str = "policy"
    critic_group: str | None = None
    """Optional privileged observations for FlashSAC; Warp PPO uses the policy group."""
    clip_actions: float | None = 1.0
    """Action clipping limit, or ``None`` to preserve native Gaussian policy actions."""
    init_at_random_ep_len: bool = False
    """Randomize initial episode counters, matching the native RSL-RL option."""
    updates_per_step: int = 1
    """FlashSAC gradient updates per vector step once replay warmup is complete."""
    flash_updates_during_rollout: bool = False
    """Interleave FlashSAC replay updates with collection, as in the authors' runner."""
    capture_updates: bool = True
    """Capture the complete Warp PPO learning update in a reusable CUDA graph."""
    capture_rollout: bool = False
    """Capture G1 Newton physics, Warp MDP, rollout collection, and PPO in one graph.

    This experimental path currently supports only the unmodified flat G1 task,
    headless Newton MJWarp physics, and Warp PPO. Other tasks use the usual runner.
    """
    algorithm_cfg: dict[str, Any] = {}
    """Keyword arguments for RoboLearn's FlashSACConfig or PPOConfig."""
