# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One PPO implementation and evaluation protocol for all simulator adapters."""

import time

import torch
from torch import nn
from torch.distributions import Normal


class Policy(nn.Module):
    """Actor/critic with optional RGB features and shared proprioception."""

    def __init__(self, obs_dim: int, action_dim: int, vision: bool):
        super().__init__()
        self.encoder = None
        if vision:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 5, 2),
                nn.ELU(),
                nn.Conv2d(16, 32, 3, 2),
                nn.ELU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(512, 64),
                nn.ELU(),
            )
        self.body = nn.Sequential(
            nn.Linear(obs_dim + (64 if vision else 0), 128), nn.ELU(), nn.Linear(128, 128), nn.ELU()
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

    def forward(self, obs: torch.Tensor, images: torch.Tensor | None = None) -> tuple:
        """Return action distribution and value; images are uint8 RGB [N,H,W,3]."""
        if self.encoder is not None:
            pixels = images[..., :3].permute(0, 3, 1, 2).float() / 255.0
            obs = torch.cat((obs, self.encoder(pixels)), -1)
        hidden = self.body(obs)
        return Normal(self.actor(hidden), self.log_std.exp()), self.critic(hidden).squeeze(-1)


@torch.no_grad()
def evaluate(task, policy: Policy, steps: int, seed: int) -> dict:
    """Measure held-out reset survival and task quality over a fixed horizon [s]."""
    old_generator = task.generator.get_state()
    task.generator.manual_seed(seed)
    obs = task.reset()
    failed = torch.zeros(task.n, dtype=torch.bool, device=task.device)
    returns = torch.zeros(task.n, device=task.device)
    errors = torch.zeros_like(returns)
    key = "tracking_error_m_s" if task.walk else "pole_angle_abs_rad"
    for _ in range(steps):
        distribution, _ = policy(obs, task.images)
        obs, reward, _, info = task.step(distribution.mean)
        returns += reward
        errors += info[key]
        failed |= info["fallen"].bool()
    result = {
        "survival_fraction": (~failed).float().mean().item(),
        "return_mean": (returns / steps).mean().item(),
        key: (errors / steps).mean().item(),
        "steps": steps,
        "seed": seed,
    }
    result["threshold_reached"] = result["survival_fraction"] >= 0.95 and (result[key] <= (0.25 if task.walk else 0.1))
    task.generator.set_state(old_generator)
    return result


def train(task, config: dict, output_dir) -> dict:
    """Train a fixed PPO budget and retain evaluation curves, checkpoints, and timings."""
    torch.manual_seed(config["seed"])
    task.generator.manual_seed(config["seed"])
    obs = task.reset()
    policy = Policy(obs.shape[-1], task.action_dim, config["vision"] == "rgb").to(task.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    horizon = config["rollout_steps"]
    n = task.n
    curve = []
    task.adapter.synchronize()
    start = time.perf_counter()
    initial = evaluate(task, policy, config["evaluation_steps"], config["seed"] + 100000)
    curve.append(
        {
            "iteration": 0,
            "elapsed_s": time.perf_counter() - start,
            "env_steps": 0,
            "reward_mean": None,
            "evaluation": initial,
        }
    )
    obs = task.reset()
    first_success = {"elapsed_s": curve[0]["elapsed_s"], "env_steps": 0} if initial["threshold_reached"] else None
    for iteration in range(1, config["iterations"] + 1):
        observations, actions, log_probs, rewards, values, dones, timeout_values, pixels = ([] for _ in range(8))
        with torch.no_grad():
            for _ in range(horizon):
                images = task.images
                distribution, value = policy(obs, images)
                action = distribution.sample()
                observations.append(obs)
                actions.append(action)
                log_probs.append(distribution.log_prob(action).sum(-1))
                values.append(value)
                if images is not None:
                    pixels.append(images[..., :3].clone().to(torch.uint8))
                obs, reward, done, info = task.step(action)
                rewards.append(reward)
                dones.append(done)
                terminal_value = torch.zeros_like(value)
                if bool(info["timeout"].any()):
                    mask = info["timeout"]
                    _, bootstrap = policy(info["terminal_observation"][mask], info.get("terminal_images"))
                    terminal_value[mask] = bootstrap
                timeout_values.append(terminal_value)
            _, next_value = policy(obs, task.images)
            advantages = torch.zeros((horizon, n), device=task.device)
            gae = torch.zeros(n, device=task.device)
            for step in reversed(range(horizon)):
                continuing = (~dones[step]).float()
                bootstrap = next_value * continuing + timeout_values[step]
                delta = rewards[step] + 0.99 * bootstrap - values[step]
                gae = delta + 0.99 * 0.95 * continuing * gae
                advantages[step] = gae
                next_value = values[step]
            targets = advantages + torch.stack(values)
            flat_obs = torch.stack(observations).flatten(0, 1)
            flat_action = torch.stack(actions).flatten(0, 1)
            flat_logp = torch.stack(log_probs).flatten()
            flat_adv = advantages.flatten()
            flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
            flat_target = targets.flatten()
            flat_pixels = torch.stack(pixels).flatten(0, 1) if pixels else None
        for _ in range(4):
            order = torch.randperm(n * horizon, device=task.device)
            for index in order.tensor_split(4):
                distribution, predicted = policy(
                    flat_obs[index], flat_pixels[index] if flat_pixels is not None else None
                )
                ratio = (distribution.log_prob(flat_action[index]).sum(-1) - flat_logp[index]).exp()
                surrogate = torch.minimum(ratio * flat_adv[index], ratio.clamp(0.8, 1.2) * flat_adv[index])
                loss = -surrogate.mean() + 0.5 * (predicted - flat_target[index]).square().mean()
                loss -= 0.005 * distribution.entropy().sum(-1).mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("PPO loss became non-finite")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
        if iteration % config["evaluation_interval"] == 0 or iteration == config["iterations"]:
            evaluation = evaluate(task, policy, config["evaluation_steps"], config["seed"] + 100000)
            task.adapter.synchronize()
            elapsed = time.perf_counter() - start
            record = {
                "iteration": iteration,
                "elapsed_s": elapsed,
                "env_steps": iteration * horizon * n,
                "reward_mean": torch.stack(rewards).mean().item(),
                "evaluation": evaluation,
            }
            curve.append(record)
            print(f"PPO {iteration}/{config['iterations']}: {evaluation}", flush=True)
            if first_success is None and evaluation["threshold_reached"]:
                first_success = {"elapsed_s": elapsed, "env_steps": record["env_steps"]}
            obs = task.reset()
    task.adapter.synchronize()
    elapsed = time.perf_counter() - start
    torch.save({"state_dict": policy.state_dict(), "config": config}, output_dir / "policy.pt")
    return {
        "learning": curve,
        "evaluation": curve[-1]["evaluation"],
        "training_s": elapsed,
        "time_to_threshold": first_success,
        "threshold_censored": first_success is None,
        "env_steps_per_s": config["iterations"] * horizon * n / elapsed,
        "algorithm": {
            "name": "PPO",
            "lr": 3e-4,
            "gamma": 0.99,
            "lambda": 0.95,
            "epochs": 4,
            "minibatches": 4,
            "clip": 0.2,
            "entropy_coef": 0.005,
            "observation": "proprioception + CNN RGB" if config["vision"] == "rgb" else "proprioception",
            "evaluation_included_in_wall_time": True,
        },
    }
