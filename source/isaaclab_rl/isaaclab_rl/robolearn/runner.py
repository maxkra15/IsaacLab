# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab training and inference adapter for the optional RoboLearn library."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

logger = logging.getLogger(__name__)


class RoboLearnRunner:
    """Run FlashSAC or WarpNN PPO against the same Isaac Lab environment contract.

    Every iteration collects ``num_steps_per_env`` steps from every environment.
    FlashSAC then performs that many replay updates times ``updates_per_step``;
    WarpNN PPO performs one full-batch PPO update. The latter captures the learning
    update only, while Isaac Lab stepping and observation assembly remain eager.

    Checkpoints consist of a JSON file and an adjacent directory of algorithm
    state. FlashSAC checkpoints restore networks and optimizers; replay is rebuilt
    after resuming. Both algorithms accept a different environment count on play.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv | DirectRLEnv,
        cfg: dict,
        log_dir: str | None = None,
        device: str = "cuda:0",
    ):
        """Initialize the selected learner without importing inactive dependencies.

        Args:
            env: Vector environment created with ``compute_final_obs=True``.
            cfg: Serialized :class:`~isaaclab_rl.robolearn.RoboLearnRunnerCfg`.
            log_dir: Run directory, or ``None`` for inference without logging.
            device: Learner device, which must match the environment device.
        """
        from robolearn.isaaclab import IsaacLabEnv

        self.cfg = deepcopy(cfg)
        self.device = _resolve_device(device)
        env_device = _resolve_device(env.unwrapped.device)
        if self.device.type != env_device.type or self.device.index != env_device.index:
            raise ValueError("RoboLearn requires the learner and environment on the same device.")
        if cfg["clip_actions"] != 1.0:
            raise ValueError("RoboLearn uses normalized actions clipped to [-1, 1]; clip_actions must be 1.0.")
        if min(cfg["num_steps_per_env"], cfg["save_interval"], cfg["updates_per_step"]) < 1:
            raise ValueError("Rollout horizon, save interval, and updates per step must be positive.")
        self.env = IsaacLabEnv(
            env,
            observation_group=cfg["observation_group"],
            critic_group=cfg["critic_group"],
        )
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.current_learning_iteration = 0
        self.total_steps = 0
        self.gradient_updates = 0
        self.actor_gradient_updates = 0
        self.critic_gradient_updates = 0
        self._warmup_seconds = 0.0
        self._prepared = False
        self._episode_returns = torch.zeros(self.env.num_envs, device=self.device)
        self._episode_lengths = torch.zeros(self.env.num_envs, device=self.device)
        algorithm_cfg = deepcopy(cfg["algorithm_cfg"])
        algorithm_cfg["seed"] = cfg["seed"]
        torch.manual_seed(cfg["seed"])

        if cfg["algorithm"] == "flashsac":
            from robolearn.flashsac import FlashSAC, FlashSACConfig

            algorithm_cfg["device"] = str(self.device)
            flash_cfg = FlashSACConfig(**algorithm_cfg)
            self._actor_update_period = flash_cfg.actor_update_period
            self.agent = FlashSAC(
                self.env.observation_dim,
                self.env.action_dim,
                num_envs=self.env.num_envs,
                cfg=flash_cfg,
                critic_observation_dim=self.env.critic_observation_dim,
            )
        elif cfg["algorithm"] == "warp_ppo":
            import warp as wp
            from robolearn.warp import PPOConfig, WarpPPO

            if cfg["critic_group"] is not None:
                raise ValueError("WarpNN PPO currently requires a shared actor and critic observation group.")
            self.agent = WarpPPO(
                self.env.observation_dim,
                self.env.action_dim,
                num_envs=self.env.num_envs,
                horizon=cfg["num_steps_per_env"],
                config=PPOConfig(**algorithm_cfg),
                device=str(self.device),
            )
            # Warp 1.17 cannot capture an imported Torch default stream. Own a
            # Warp stream and use its Torch mirror for device-side dependencies.
            self._warp_stream = wp.Stream(self.agent.device)
            self._torch_warp_stream = wp.stream_to_torch(self._warp_stream)
            self._warp_stream.wait_stream(wp.get_stream(self.agent.device))
            self._warp_obs = wp.zeros(
                (self.env.num_envs, self.env.observation_dim), dtype=wp.float32, device=self.agent.device
            )
            self._warp_next_obs = wp.zeros_like(self._warp_obs)
            self._warp_terminated = torch.zeros(self.env.num_envs, dtype=torch.int32, device=self.device)
            self._warp_truncated = torch.zeros_like(self._warp_terminated)
        else:
            raise ValueError(f"Unknown RoboLearn algorithm: {cfg['algorithm']!r}.")

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Collect rollouts, train, and write synchronized per-iteration metrics.

        Args:
            num_learning_iterations: Number of additional iterations to run.
            init_at_random_ep_len: Randomize the initial episode counters, as in
                RSL-RL. Leave disabled for comparisons with identical resets.
        """
        if num_learning_iterations < 1:
            raise ValueError("The number of learning iterations must be positive.")
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        observations = self.env.reset(seed=self.cfg["seed"])
        if init_at_random_ep_len:
            self.env.env.episode_length_buf[:] = torch.randint_like(
                self.env.env.episode_length_buf, high=self.env.env.max_episode_length
            )
        self._episode_returns.zero_()
        self._episode_lengths.zero_()
        prepared_before = self._prepared
        self._prepare_learning()
        first_iteration = self.current_learning_iteration + 1
        end_iteration = self.current_learning_iteration + num_learning_iterations

        for iteration in range(first_iteration, end_iteration + 1):
            episode_totals = torch.zeros(3, device=self.device)
            self._synchronize()
            rollout_start = time.perf_counter()
            for step in range(self.cfg["num_steps_per_env"]):
                observations, transition = self._collect_step(observations, step)
                self._episode_returns.add_(transition["reward"])
                self._episode_lengths.add_(1)
                done = transition["terminated"] | transition["truncated"]
                episode_totals[0].add_(torch.where(done, self._episode_returns, 0).sum())
                episode_totals[1].add_(torch.where(done, self._episode_lengths, 0).sum())
                episode_totals[2].add_(done.sum())
                self._episode_returns.masked_fill_(done, 0)
                self._episode_lengths.masked_fill_(done, 0)
            self._synchronize()
            rollout_seconds = time.perf_counter() - rollout_start
            update_start = time.perf_counter()
            losses, updates, actor_updates = self._update()
            self._synchronize()
            update_seconds = time.perf_counter() - update_start
            steps = self.env.num_envs * self.cfg["num_steps_per_env"]
            self.total_steps += steps
            self.gradient_updates += updates
            self.actor_gradient_updates += actor_updates
            self.critic_gradient_updates += updates
            self.current_learning_iteration = iteration
            total_return, total_length, episodes = episode_totals.tolist()
            metrics = {
                "iteration": iteration,
                "algorithm": self.cfg["algorithm"],
                "total_steps": self.total_steps,
                "gradient_updates": self.gradient_updates,
                "actor_gradient_updates": self.actor_gradient_updates,
                "critic_gradient_updates": self.critic_gradient_updates,
                "updates_this_iteration": updates,
                "rollout_seconds": rollout_seconds,
                "update_seconds": update_seconds,
                "iteration_seconds": rollout_seconds + update_seconds,
                "fps": steps / (rollout_seconds + update_seconds),
                "warmup_seconds": self._warmup_seconds if iteration == first_iteration and not prepared_before else 0.0,
                "completed_episodes": int(episodes),
                "mean_episode_return": total_return / episodes if episodes else None,
                "mean_episode_length": total_length / episodes if episodes else None,
                "losses": losses,
            }
            if self.log_dir is not None:
                with (self.log_dir / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(metrics, allow_nan=False) + "\n")
                if iteration % self.cfg["save_interval"] == 0 or iteration == end_iteration:
                    self.save(str(self.log_dir / f"model_{iteration}.json"))
            logger.info(
                "RoboLearn %s iteration %d: %.0f steps/s, episode return %s",
                self.cfg["algorithm"],
                iteration,
                metrics["fps"],
                metrics["mean_episode_return"],
            )

    def save(self, path: str) -> None:
        """Save model and optimizer state beside a readable JSON checkpoint."""
        checkpoint = Path(path)
        if checkpoint.suffix != ".json":
            raise ValueError("RoboLearn checkpoint metadata must use the .json extension.")
        state_dir = checkpoint.with_suffix("")
        state_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg["algorithm"] == "flashsac":
            self.agent.save(str(state_dir))
        else:
            self.agent.save(state_dir / "policy.npz")
        metadata = {
            "format_version": 1,
            "config": self.cfg,
            "iteration": self.current_learning_iteration,
            "total_steps": self.total_steps,
            "gradient_updates": self.gradient_updates,
            "actor_gradient_updates": self.actor_gradient_updates,
            "critic_gradient_updates": self.critic_gradient_updates,
            "observation_dim": self.env.observation_dim,
            "critic_observation_dim": self.env.critic_observation_dim,
            "action_dim": self.env.action_dim,
            "replay_included": False,
        }
        temporary = checkpoint.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(checkpoint)

    def load(self, path: str) -> None:
        """Restore model and optimizer state after validating the observation contract."""
        checkpoint = Path(path)
        metadata = json.loads(checkpoint.read_text())
        if metadata["format_version"] != 1:
            raise ValueError("Unsupported RoboLearn checkpoint format version.")
        for key in ("algorithm", "observation_group", "critic_group", "clip_actions", "algorithm_cfg"):
            if json.dumps(self.cfg[key], sort_keys=True) != json.dumps(metadata["config"][key], sort_keys=True):
                raise ValueError(f"The checkpoint's {key} does not match the runner configuration.")
        dimensions = {
            "observation_dim": self.env.observation_dim,
            "critic_observation_dim": self.env.critic_observation_dim,
            "action_dim": self.env.action_dim,
        }
        if any(metadata[key] != value for key, value in dimensions.items()):
            raise ValueError("Checkpoint observation or action dimensions do not match the environment.")
        state_dir = checkpoint.with_suffix("")
        if self.cfg["algorithm"] == "flashsac":
            self.agent.load(str(state_dir))
        else:
            self.agent.load(state_dir / "policy.npz")
        self.current_learning_iteration = metadata["iteration"]
        self.total_steps = metadata["total_steps"]
        self.gradient_updates = metadata["gradient_updates"]
        self.actor_gradient_updates = metadata["actor_gradient_updates"]
        self.critic_gradient_updates = metadata["critic_gradient_updates"]

    def get_inference_policy(self, device: str | None = None) -> Callable:
        """Return a deterministic policy accepting an observation mapping or tensor.

        The returned action is clipped to [-1, 1] on the environment device. The
        policy does not change the learner device or its fixed environment count.
        """
        if device is not None and _resolve_device(device) != self.device:
            raise ValueError("Inference must use the runner's environment device.")

        def policy(observations):
            if isinstance(observations, Mapping):
                observations = self.env._observations(observations)
            with torch.no_grad():
                if self.cfg["algorithm"] == "flashsac":
                    return self.agent.act(observations, training=False).clamp(-1, 1)
                import warp as wp

                environment_stream = torch.cuda.current_stream(self.device)
                self._torch_warp_stream.wait_stream(environment_stream)
                observations.record_stream(self._torch_warp_stream)
                with torch.cuda.stream(self._torch_warp_stream), wp.ScopedStream(self._warp_stream, sync_enter=False):
                    wp.copy(self._warp_obs, wp.from_torch(observations, dtype=wp.float32))
                    actions = self.agent.act(self._warp_obs, deterministic=True)
                    result = wp.to_torch(actions).clamp(-1, 1)
                environment_stream.wait_stream(self._torch_warp_stream)
                result.record_stream(environment_stream)
                return result

        return policy

    def _prepare_learning(self) -> None:
        if self._prepared or self.cfg["algorithm"] != "warp_ppo":
            return
        import warp as wp

        self._synchronize()
        start = time.perf_counter()
        with wp.ScopedStream(self._warp_stream, sync_enter=False):
            # The first update compiles and captures against persistent arrays.
            # Restore all parameters, Adam state, and RNG before collecting data.
            state = self.agent.state_dict()
            self.agent.update(capture=self.cfg["capture_updates"])
            self.agent.load_state_dict(state)
        self._synchronize()
        self._warmup_seconds = time.perf_counter() - start
        self._prepared = True

    def _collect_step(self, observations: torch.Tensor, step: int) -> tuple[torch.Tensor, dict]:
        with torch.no_grad():
            if self.cfg["algorithm"] == "flashsac":
                actions = (
                    self.agent.act(observations)
                    if self.agent.ready
                    else torch.empty((self.env.num_envs, self.env.action_dim), device=self.device).uniform_(-1, 1)
                )
                observations, transition = self.env.step(actions)
                self.agent.process_transition(transition)
                return observations, transition
            import warp as wp

            environment_stream = torch.cuda.current_stream(self.device)
            self._torch_warp_stream.wait_stream(environment_stream)
            observations.record_stream(self._torch_warp_stream)
            with wp.ScopedStream(self._warp_stream, sync_enter=False):
                wp.copy(self._warp_obs, wp.from_torch(observations, dtype=wp.float32))
                actions = wp.to_torch(self.agent.act(self._warp_obs))
            environment_stream.wait_stream(self._torch_warp_stream)
            # Keep physics on its existing streams. The adapter clips out of
            # place, preserving Gaussian samples and their PPO log probabilities.
            observations, transition = self.env.step(actions)
            self._warp_terminated.copy_(transition["terminated"])
            self._warp_truncated.copy_(transition["truncated"])
            self._torch_warp_stream.wait_stream(environment_stream)
            # These transient Torch tensors may be released before the learner
            # stream consumes them; tell its allocator about that dependency.
            transition["next_observation"].record_stream(self._torch_warp_stream)
            transition["reward"].record_stream(self._torch_warp_stream)
            with wp.ScopedStream(self._warp_stream, sync_enter=False):
                wp.copy(self._warp_next_obs, wp.from_torch(transition["next_observation"], dtype=wp.float32))
                self.agent.store(
                    step,
                    self._warp_obs,
                    wp.from_torch(transition["reward"], dtype=wp.float32),
                    wp.from_torch(self._warp_terminated, dtype=wp.int32),
                    wp.from_torch(self._warp_truncated, dtype=wp.int32),
                    self._warp_next_obs,
                )
                return observations, transition

    def _update(self) -> tuple[dict[str, float], int, int]:
        if self.cfg["algorithm"] == "flashsac":
            if not self.agent.ready:
                return {}, 0, 0
            count = self.cfg["num_steps_per_env"] * self.cfg["updates_per_step"]
            totals: dict[str, float] = {}
            for _ in range(count):
                for key, value in self.agent.update().items():
                    totals[key] = totals.get(key, 0.0) + value
            actor_updates = sum(
                (self.gradient_updates + index) % self._actor_update_period == 0 for index in range(count)
            )
            return {key: value / count for key, value in totals.items()}, count, actor_updates
        import warp as wp

        with wp.ScopedStream(self._warp_stream, sync_enter=False):
            self.agent.update(capture=self.cfg["capture_updates"])
            policy_loss, value_loss, entropy = self.agent.metrics.numpy().tolist()
        return (
            {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
            },
            self.agent.config.epochs,
            self.agent.config.epochs,
        )

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


def _resolve_device(device: str | torch.device) -> torch.device:
    """Resolve an unindexed CUDA device to Torch's current device."""
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved
