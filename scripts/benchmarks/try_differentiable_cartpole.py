# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Try physics-to-policy gradients on the native Isaac Lab Cartpole scene.

This standalone experiment needs the pinned MJWarp adjoint dependency documented
in ``differentiable_cartpole_physics.py`` and RoboLearn's experimental learner.
It records finite differences before training, full native episode evaluations,
and optional eager/captured update parity. It makes no matched PPO speed claim.
Reproduction and measured pilot results:
https://github.com/maxkra15/RoboLearn/blob/experiment/differentiable-cartpole/docs/differentiable.md
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from differentiable_cartpole_physics import NativeDifferentiableCartpole, install_native_compatibility
from robolearn.warp.experimental import PathwiseConfig, WarpPathwiseActorCritic

from isaaclab.app import launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


@wp.kernel(enable_backward=True)
def _mean_reward(rewards: wp.array[wp.float32], result: wp.array[wp.float32]):
    i = wp.tid()
    wp.atomic_add(result, 0, rewards[i] / float(rewards.shape[0]))


def _write(path: Path, result: dict) -> None:
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def provenance() -> dict:
    """Record the command, source content, dependencies, and selected hardware."""
    import robolearn.warp.experimental as learner

    sources = [Path(__file__), Path(__file__).with_name("differentiable_cartpole_physics.py"), Path(learner.__file__)]
    roots = {"isaaclab": sources[0].resolve().parents[2], "robolearn": sources[-1].resolve().parents[3]}
    revisions = {}
    for name, root in roots.items():
        revisions[name] = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    return {
        "task": "Isaac-Cartpole",
        "command": [sys.executable, *sys.argv],
        "hardware": torch.cuda.get_device_name(0),
        "source_revisions": revisions,
        "source_sha256": {str(source): hashlib.sha256(source.read_bytes()).hexdigest() for source in sources},
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "warp-lang", "warp-nn", "newton", "mujoco-warp", "robolearn-rl")
        },
    }


def _snapshot(adapter: NativeDifferentiableCartpole) -> dict:
    """Preserve all integration state and reset counters for fixed-start probes."""
    data = adapter.states[0]
    arrays = {name: wp.clone(value) for name, value in vars(data).items() if isinstance(value, wp.array)}
    return {"data": arrays, "rng": wp.clone(adapter.rng), "steps": wp.clone(adapter.episode_steps)}


def _restore(adapter: NativeDifferentiableCartpole, snapshot: dict) -> None:
    for name, value in snapshot["data"].items():
        wp.copy(getattr(adapter.states[0], name), value)
    wp.copy(adapter.rng, snapshot["rng"])
    wp.copy(adapter.episode_steps, snapshot["steps"])


def gradients(agent, adapter) -> dict:
    """Measure action and policy derivatives from identical native start states."""
    state = _snapshot(adapter)
    actions = wp.full((adapter.num_envs, 1), 0.05, dtype=wp.float32, device=adapter.device, requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float32, device=adapter.device, requires_grad=True)

    def physics_objective(value: float, backward: bool = False):
        _restore(adapter, state)
        adapter.begin_rollout()
        actions.fill_(value)
        loss.zero_()
        actions.grad.zero_()
        with wp.Tape() as tape:
            reward, _ = adapter.step(0, actions)
            wp.launch(_mean_reward, dim=adapter.num_envs, inputs=[reward], outputs=[loss], device=adapter.device)
        if backward:
            tape.backward(loss)
        result = float(loss.numpy()[0])
        gradient = float(actions.grad.numpy().sum())
        tape.zero()
        return result, gradient

    _, analytic = physics_objective(0.05, True)
    physics = []
    for epsilon in (0.001, 0.005, 0.01):
        plus, _ = physics_objective(0.05 + epsilon)
        minus, _ = physics_objective(0.05 - epsilon)
        finite_difference = (plus - minus) / (2 * epsilon)
        physics.append(
            {
                "epsilon": epsilon,
                "analytic": analytic,
                "finite_difference": finite_difference,
                "relative_error": abs(analytic - finite_difference) / max(abs(analytic), abs(finite_difference), 1e-8),
            }
        )

    _restore(adapter, state)
    tape = agent.forward_actor(adapter)
    tape.backward(agent.actor_loss)
    base_loss = float(agent.actor_loss.numpy()[0])
    parameter = agent.actor_parameters[-1]
    original = parameter.numpy().copy()
    derivative = parameter.grad.numpy().copy()
    all_gradients = [p.grad.numpy().copy() for p in agent.actor_parameters]
    norm = float(np.sqrt(sum(np.square(g, dtype=np.float64).sum() for g in all_gradients)))
    tape.zero()
    index = np.unravel_index(np.abs(derivative).argmax(), derivative.shape)
    policy = []
    for epsilon in (0.001, 0.005, 0.01):
        losses = []
        for sign in (1, -1):
            candidate = original.copy()
            candidate[index] += sign * epsilon
            parameter.assign(candidate)
            _restore(adapter, state)
            probe = agent.forward_actor(adapter)
            losses.append(float(agent.actor_loss.numpy()[0]))
            probe.zero()
        fd = (losses[0] - losses[1]) / (2 * epsilon)
        ad = float(derivative[index])
        policy.append(
            {
                "epsilon": epsilon,
                "parameter": "final_actor_bias",
                "analytic": ad,
                "finite_difference": fd,
                "relative_error": abs(ad - fd) / max(abs(ad), abs(fd), 1e-8),
            }
        )
    parameter.assign(original)
    _restore(adapter, state)
    if not np.isfinite(norm) or norm <= 1e-10:
        raise RuntimeError("The initial physics-to-policy gradient is absent or non-finite.")
    result = {
        "physics_action": physics,
        "actor_parameter": policy,
        "actor_gradient_norm": norm,
        "fixed_start_actor_loss": base_loss,
        "critic_initially_zero": True,
    }
    result["accepted"] = (
        min(row["relative_error"] for row in physics) < 0.05 and min(row["relative_error"] for row in policy) < 0.05
    )
    return result


def native_parity(env, adapter) -> dict:
    """Compare a complete control step with the ordinary Isaac Lab environment."""
    snapshot = _snapshot(adapter)
    actions = torch.full((env.num_envs, 1), 0.05, device=env.device)
    adapter.begin_rollout()
    reward, _ = adapter.step(0, wp.from_torch(actions, dtype=wp.float32))
    adapter_observation = adapter.observe(1).numpy().copy()
    adapter_reward = reward.numpy().copy()
    observations, native_reward, _, _, _ = env.step(actions)
    robot = env.scene["robot"]
    cart = robot.find_joints("slider_to_cart")[0][0]
    pole = robot.find_joints("cart_to_pole")[0][0]
    order = [cart, pole, robot.num_joints + cart, robot.num_joints + pole]
    result = {
        "observation_max_abs_error": float(
            np.max(np.abs(adapter_observation - observations["policy"][:, order].cpu().numpy()))
        ),
        "reward_max_abs_error": float(np.max(np.abs(adapter_reward - native_reward.cpu().numpy()))),
        "mjwarp_timestep": adapter.model.opt.timestep.numpy().tolist(),
    }
    _restore(adapter, snapshot)
    result["accepted"] = result["observation_max_abs_error"] < 1e-4 and result["reward_max_abs_error"] < 1e-6
    return result


def evaluate(env, agent, seed: int, *, zero: bool = False) -> dict:
    """Evaluate the first complete episode per world using the native manager MDP."""
    observations, _ = env.reset(seed=seed)
    robot = env.scene["robot"]
    cart = robot.find_joints("slider_to_cart")[0][0]
    pole = robot.find_joints("cart_to_pole")[0][0]
    order = [cart, pole, robot.num_joints + cart, robot.num_joints + pole]
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    returns = torch.zeros(env.num_envs, device=env.device)
    lengths = torch.zeros_like(returns)
    upright = torch.zeros_like(returns)
    squared_angles = torch.zeros_like(returns)
    saturation = torch.zeros_like(returns)
    survived = torch.zeros_like(active)
    for _ in range(env.max_episode_length):
        inputs = observations["policy"][:, order].contiguous()
        action = (
            torch.zeros((env.num_envs, 1), device=env.device)
            if zero
            else wp.to_torch(agent.act(wp.from_torch(inputs, dtype=wp.float32))).detach()
        )
        observations, reward, terminated, truncated, extras = env.step(action)
        terminal = extras.get("final_obs", observations)["policy"]
        state = torch.where((terminated | truncated)[:, None], terminal, observations["policy"])
        angle = torch.remainder(state[:, pole] + torch.pi, 2 * torch.pi) - torch.pi
        returns += reward * active
        lengths += active
        upright += (angle.abs() < 0.2) * active
        squared_angles += angle.square() * active
        saturation += (action[:, 0].abs() > 0.95) * active
        survived |= active & truncated & ~terminated
        active &= ~(terminated | truncated)
        if not active.any():
            break
    return {
        "seed": seed,
        "episodes": env.num_envs,
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std(unbiased=False)),
        "normalized_score": float(100 * returns.mean() / 5),
        "episode_length_mean": float(lengths.mean()),
        "survival_rate": float(survived.float().mean()),
        "upright_fraction": float((upright / lengths).mean()),
        "wrapped_angle_rms": float(torch.sqrt((squared_angles / lengths).mean())),
        "action_saturation_fraction": float((saturation / lengths).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--gradient_only", action="store_true")
    args = parser.parse_args()
    if min(args.iterations, args.num_envs, args.horizon) < 1:
        parser.error("Iterations, environment count, and horizon must be positive.")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(4)
    install_native_compatibility()
    cfg = parse_env_cfg("Isaac-Cartpole", device="cuda:0", num_envs=args.num_envs, overrides=["physics=newton_mjwarp"])
    cfg.compute_final_obs = True
    cfg.sim.physics.use_cuda_graph = False
    with launch_simulation(cfg):
        env = gym.make("Isaac-Cartpole", cfg=cfg).unwrapped
        try:
            env.reset(seed=args.seed)
            adapter = NativeDifferentiableCartpole(env, seed=args.seed)
            adapter.prepare(args.horizon)
            agent = WarpPathwiseActorCritic(4, 1, args.num_envs, args.horizon, config=PathwiseConfig(seed=args.seed))
            result = {
                "metadata": {**adapter.metadata(), **provenance()},
                "config": asdict(agent.config),
                "notes": [
                    "SHAC-inspired deterministic n-step prototype, not a SHAC reproduction.",
                    "Full native manager Cartpole evaluations; fallen poles do not terminate.",
                    "Local GPU shares an unrelated demo; timings are not isolated speed comparisons.",
                ],
                "evaluations": [],
                "iterations": 0,
                "transitions": 0,
            }
            result["native_parity"] = native_parity(env, adapter)
            print("NATIVE_PARITY", json.dumps(result["native_parity"]), flush=True)
            if not result["native_parity"]["accepted"]:
                _write(args.output / "results.json", result)
                raise RuntimeError("Adapter differs from the native control-step MDP.")
            result["gradients"] = gradients(agent, adapter)
            _write(args.output / "results.json", result)
            print("GRADIENT_DIAGNOSTICS", json.dumps(result["gradients"]), flush=True)
            if not result["gradients"]["accepted"]:
                raise RuntimeError("Finite-difference agreement is insufficient; stop before training.")
            if args.gradient_only:
                return
            initial = agent.state_dict()
            initial_physics = _snapshot(adapter)
            agent.save(args.output / "initial.npz")
            result["zero_action_evaluation"] = evaluate(env, agent, 10000, zero=True)
            result["evaluations"].append({"iteration": 0, **evaluate(env, agent, 10000)})
            # Warm the complete update, then restore parameters, optimizer and
            # physical state so setup work is excluded from the training budget.
            agent.launch_update(adapter)
            wp.synchronize_device(agent.device)
            eager = agent.state_dict()
            eager_qpos = adapter.states[0].qpos.numpy().copy()
            eager_qvel = adapter.states[0].qvel.numpy().copy()
            agent.load_state_dict(initial)
            _restore(adapter, initial_physics)
            graph = None
            if args.capture:
                with wp.ScopedCapture(device=agent.device) as capture:
                    agent.launch_update(adapter)
                graph = capture.graph
                wp.capture_launch(graph)
                wp.synchronize_device(agent.device)
                replay = agent.state_dict()
                errors = {k: float(np.max(np.abs(eager[k] - replay[k]))) for k in eager if "parameter" in k}
                optimizer_errors = {k: float(np.max(np.abs(eager[k] - replay[k]))) for k in eager if "adam" in k}
                qpos_error = float(np.max(np.abs(eager_qpos - adapter.states[0].qpos.numpy())))
                qvel_error = float(np.max(np.abs(eager_qvel - adapter.states[0].qvel.numpy())))
                result["capture_parity"] = {
                    "parameter_max_abs_errors": errors,
                    "qpos_max_abs_error": qpos_error,
                    "qvel_max_abs_error": qvel_error,
                    "optimizer_max_abs_errors": optimizer_errors,
                }
                checked_errors = [*errors.values(), *optimizer_errors.values(), qpos_error, qvel_error]
                if not np.isfinite(checked_errors).all() or max(checked_errors) > 2e-5:
                    _write(args.output / "results.json", result)
                    raise RuntimeError("Captured and eager updates disagree.")
                agent.load_state_dict(initial)
                _restore(adapter, initial_physics)
            result["capture"] = graph is not None
            train_started = time.perf_counter()
            checkpoints = {100, 250, 500, 1000, args.iterations}
            with (args.output / "metrics.jsonl").open("w") as metrics:
                for iteration in range(1, args.iterations + 1):
                    wp.synchronize_device(agent.device)
                    tick = time.perf_counter()
                    if graph is None:
                        agent.launch_update(adapter)
                    else:
                        wp.capture_launch(graph)
                    wp.synchronize_device(agent.device)
                    row = {
                        "iteration": iteration,
                        "transitions": iteration * args.num_envs * args.horizon,
                        "seconds": time.perf_counter() - tick,
                        "actor_loss": float(agent.actor_loss.numpy()[0]),
                        "critic_loss": float(agent.critic_loss.numpy()[0]),
                    }
                    if not np.isfinite([row["actor_loss"], row["critic_loss"]]).all():
                        raise RuntimeError(f"Non-finite learning loss at iteration {iteration}.")
                    metrics.write(json.dumps(row) + "\n")
                    metrics.flush()
                    if iteration % 25 == 0:
                        print("PATHWISE_PROGRESS", json.dumps(row), flush=True)
                    if iteration in checkpoints:
                        agent.save(args.output / f"model_{iteration}.npz")
                        result["evaluations"].append({"iteration": iteration, **evaluate(env, agent, 10000)})
                        result.update(iterations=iteration, transitions=row["transitions"])
                        _write(args.output / "results.json", result)
            result["training_and_evaluation_seconds"] = time.perf_counter() - train_started
            final = agent.state_dict()
            result["weight_proof"] = {
                "finite": all(np.isfinite(value).all() for value in final.values()),
                "actor_parameter_delta_l2": float(
                    np.sqrt(
                        sum(
                            np.square(final[k] - initial[k], dtype=np.float64).sum()
                            for k in final
                            if k.startswith("actor_parameter_")
                        )
                    )
                ),
                "critic_parameter_delta_l2": float(
                    np.sqrt(
                        sum(
                            np.square(final[k] - initial[k], dtype=np.float64).sum()
                            for k in final
                            if k.startswith("critic_parameter_")
                        )
                    )
                ),
                "actor_updates": int(final["actor_adam_timestep"][0]),
                "critic_updates": int(final["critic_adam_timestep"][0]),
            }
            result["evaluations"].append({"iteration": args.iterations, **evaluate(env, agent, 10001)})
            if not result["weight_proof"]["finite"]:
                _write(args.output / "results.json", result)
                raise RuntimeError("Final parameters or Adam state are non-finite.")
            result["process_seconds"] = time.perf_counter() - started
            _write(args.output / "results.json", result)
            print("PATHWISE_FINISHED", json.dumps(result["weight_proof"]), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
