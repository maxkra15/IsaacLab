# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-graph G1 rollout and PPO training using the existing Isaac Lab scene."""

from __future__ import annotations

import json
import logging
import time

import torch
import warp as wp

from .runner import RoboLearnRunner

logger = logging.getLogger(__name__)


@wp.kernel(enable_backward=False)
def _clip_actions(actions: wp.array2d(dtype=wp.float32), limit: float, clipped: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    clipped[i, j] = wp.clamp(actions[i, j], -limit, limit)


class CapturedG1Runner(RoboLearnRunner):
    """Capture a fixed rollout horizon and its complete PPO update together.

    Scene construction and startup randomization use the registered Isaac Lab
    task. The captured cycle operates on persistent Warp arrays. Logging and
    checkpoint serialization synchronize outside the CUDA graph.
    """

    def __init__(self, env, cfg: dict, log_dir: str | None = None, device: str = "cuda:0"):
        from .captured_g1 import CapturedG1Env

        if cfg["algorithm"] != "warp_ppo" or cfg["critic_group"] is not None:
            raise ValueError("Captured G1 training requires Warp PPO and shared actor/critic observations.")
        if cfg["observation_group"] != "policy":
            raise ValueError("Captured G1 training requires the policy observation group.")
        # Reuse the ordinary observation contract, learner construction,
        # inference adapter, and checkpoint format. Torch allocations here are
        # startup work; the recorded cycle below uses only Warp operations.
        super().__init__(env, cfg, log_dir=log_dir, device=device)
        self.adapter = CapturedG1Env(env.unwrapped)
        if self.adapter.observation_dim != self.env.observation_dim or self.adapter.action_dim != self.env.action_dim:
            raise ValueError("The captured MDP and registered environment observation/action contracts differ.")
        self._stream = self._warp_stream
        self._observations = self._warp_obs
        self._actions = wp.zeros((self.env.num_envs, self.env.action_dim), device=self.agent.device)
        self._events = [wp.Event(self.agent.device, enable_timing=True) for _ in range(3)]
        self._graph = None

    def _launch_iteration(self) -> None:
        self.adapter.episode_totals.zero_()
        self.adapter.mdp.command_metric_totals.zero_()
        wp.record_event(self._events[0])
        for step in range(self.cfg["num_steps_per_env"]):
            # Keep the sampled state until storage: the environment overwrites
            # its observation buffer while stepping and resetting finished worlds.
            wp.copy(self._observations, self.adapter.observations)
            actions = self.agent.act(self._observations)
            if self.cfg["clip_actions"] is not None:
                wp.launch(
                    _clip_actions,
                    dim=actions.shape,
                    inputs=[actions, self.cfg["clip_actions"], self._actions],
                    device=self.agent.device,
                )
                actions = self._actions
            self.adapter.step(actions)
            self.agent.store(
                step,
                self._observations,
                self.adapter.rewards,
                self.adapter.terminated,
                self.adapter.truncated,
                self.adapter.next_observations,
            )
        wp.record_event(self._events[1])
        self.agent.launch_update()
        wp.record_event(self._events[2])

    def _prepare_graph(self) -> None:
        if self._graph is not None:
            return
        wp.synchronize_device(self.agent.device)
        started = time.perf_counter()
        with wp.ScopedStream(self._stream, sync_enter=False):
            model_state = self.agent.state_dict()
            environment_state = self.adapter.state_dict()
            # Compile and allocate before capture, then restore every persistent
            # state buffer so graph preparation consumes no training experience.
            self._launch_iteration()
            wp.synchronize_stream(self._stream)
            self.agent.load_state_dict(model_state)
            self.adapter.load_state_dict(environment_state)
            with wp.ScopedCapture(device=self.agent.device) as captured:
                self._launch_iteration()
            self._graph = captured.graph
            self.agent.load_state_dict(model_state)
            self.adapter.load_state_dict(environment_state)
        wp.synchronize_stream(self._stream)
        self._warmup_seconds = time.perf_counter() - started

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Replay the complete training graph and log its synchronized timings."""
        if num_learning_iterations < 1:
            raise ValueError("Use a positive iteration count.")
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            if self.current_learning_iteration == 0 and not (self.log_dir / "initial.json").exists():
                self.save(str(self.log_dir / "initial.json"))
        with wp.ScopedStream(self._stream, sync_enter=False):
            self.adapter.reset(self.cfg["seed"])
            if init_at_random_ep_len:
                with torch.cuda.stream(self._torch_warp_stream):
                    initial_steps = torch.randint(
                        self.adapter.mdp.max_episode_length,
                        (self.env.num_envs,),
                        dtype=torch.int32,
                        device=self.device,
                    )
                    wp.copy(self.adapter.mdp.episode_steps, wp.from_torch(initial_steps, dtype=wp.int32))
        prepared_before = self._graph is not None
        self._prepare_graph()
        first = self.current_learning_iteration + 1
        end = self.current_learning_iteration + num_learning_iterations
        steps = self.env.num_envs * self.cfg["num_steps_per_env"]
        for iteration in range(first, end + 1):
            started = time.perf_counter()
            wp.capture_launch(self._graph, stream=self._stream)
            wp.synchronize_stream(self._stream)
            wall_seconds = time.perf_counter() - started
            rollout_seconds = wp.get_event_elapsed_time(self._events[0], self._events[1], synchronize=False) / 1000
            update_seconds = wp.get_event_elapsed_time(self._events[1], self._events[2], synchronize=False) / 1000
            self.current_learning_iteration = iteration
            self.total_steps += steps
            updates = self.agent.config.epochs * self.agent.config.num_mini_batches
            self.gradient_updates += updates
            self.actor_gradient_updates += updates
            self.critic_gradient_updates += updates
            episode_return, episode_length, episodes = self.adapter.episode_totals.numpy().tolist()
            losses = self.agent.metrics.numpy().tolist()
            metrics = {
                "timestamp_unix": time.time(),
                "iteration": iteration,
                "algorithm": "warp_ppo",
                "capture_scope": "physics_mdp_rollout_ppo",
                "total_steps": self.total_steps,
                "gradient_updates": self.gradient_updates,
                "actor_gradient_updates": self.actor_gradient_updates,
                "critic_gradient_updates": self.critic_gradient_updates,
                "updates_this_iteration": updates,
                "rollout_seconds": rollout_seconds,
                "update_seconds": update_seconds,
                "iteration_seconds": wall_seconds,
                "gpu_iteration_seconds": rollout_seconds + update_seconds,
                "fps": steps / wall_seconds,
                "warmup_seconds": self._warmup_seconds if iteration == first and not prepared_before else 0.0,
                "completed_episodes": int(episodes),
                "mean_episode_return": episode_return / episodes if episodes else None,
                "mean_episode_length": episode_length / episodes if episodes else None,
                "losses": dict(zip(("policy", "value", "entropy"), losses, strict=True)),
                "health": self.agent.health_metrics(),
            }
            if self.log_dir is not None:
                with (self.log_dir / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(metrics, allow_nan=False) + "\n")
                if iteration % self.cfg["save_interval"] == 0 or iteration == end:
                    self.save(str(self.log_dir / f"model_{iteration}.json"))
            logger.info("Captured G1 PPO iteration %d: %.0f steps/s", iteration, metrics["fps"])
