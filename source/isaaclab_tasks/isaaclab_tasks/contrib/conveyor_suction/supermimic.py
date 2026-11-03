# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental observation-space dynamics with learned native sparse receipt events.

Only imagined rollouts use event probabilities. The real environment retains its
controlled-release qualification, physical bin receipts, failures and reset sampler.
The observed state is partial: other parcels and randomized physical parameters
are not fully represented. Model loss is not a task-success measurement.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from rsl_rl.algorithms.supermimic import SuperMimic
from rsl_rl.algorithms.supermimic_task import DifferentiableTask, State
from rsl_rl.models.supermimic_actor import SuperMimicActor
from rsl_rl.modules.normalization import EmpiricalNormalization
from rsl_rl.modules.world_model import WorldModel
from torch import nn

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import normalize

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def native_events(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Read physical successor events after native reward/termination evaluation."""
    command = env.command_manager.get_term("sort")
    return torch.stack((command.new_success, command.failed), dim=-1).float()


def accepted_actions(env: ManagerBasedRLEnv, previous: bool = False) -> torch.Tensor:
    """Read the finite commands accepted by both native action terms."""
    arm = env.action_manager.get_term("arm_action")
    vacuum = env.action_manager.get_term("gripper_action")
    return torch.cat(
        (arm.previous_actions, vacuum.previous_actions) if previous else (arm.raw_actions, vacuum.raw_actions), dim=-1
    )


@configclass
class VectorObservationsCfg(ObservationGroupCfg):
    """One flat, noise-free vector for the model-based learner."""

    value: ObservationTermCfg = None
    enable_corruption = False


class SuctionObservationState:
    """Integrate observed motion while retaining parcel dimensions analytically.

    Goals and action history are separate inputs. Quaternion normalization and
    a monotone lift flag prevent invalid imagined feature vectors.
    """

    increment_dim = 302

    def __init__(self, device: str) -> None:
        self.error_floor = torch.full((self.increment_dim,), 0.01, device=device)

    def unflatten(self, flat: torch.Tensor) -> State:
        return (*flat[..., :49].split((27, 3, 19), dim=-1), flat[..., 52:308])

    def features(self, state: State) -> torch.Tensor:
        return torch.cat(state, dim=-1)

    def increments(self, state: State, next_state: State) -> torch.Tensor:
        return torch.cat(tuple(next_state[index] - state[index] for index in (0, 2, 3)), dim=-1)

    def integrate(self, state: State, increment: torch.Tensor) -> State:
        robot_parcel, tool, cloud = increment.split((27, 19, 256), dim=-1)
        robot_parcel = state[0] + robot_parcel
        robot_parcel = torch.cat(
            (robot_parcel[..., :17], normalize(robot_parcel[..., 17:21]), robot_parcel[..., 21:]), dim=-1
        )
        tool = state[2] + tool
        tool = torch.cat(
            (
                tool[..., :3],
                normalize(tool[..., 3:7]),
                tool[..., 7:16],
                tool[..., 16:17].clamp(0.0, 1.0),
                torch.maximum(state[2][..., 17:18], tool[..., 17:18].clamp(0.0, 1.0)),
                tool[..., 18:19].clamp(0.0, 1.0),
            ),
            dim=-1,
        )
        cloud = (state[3] + cloud).unflatten(-1, (64, 4))
        validity = cloud[..., 3:].clamp(0.0, 1.0)
        cloud = torch.cat((cloud[..., :3] * validity, validity), dim=-1).flatten(-2)
        return robot_parcel, state[1], tool, cloud

    def error(self, prediction: State, target: State) -> torch.Tensor:
        return self.increments(target, prediction)


class SuctionTask(DifferentiableTask):
    """Keep sparse native reward parity through a learned two-event successor head."""

    num_labels = 2

    def __init__(self, env: ManagerBasedRLEnv, device: str = "cpu") -> None:
        owner = env.unwrapped
        self.representation = SuctionObservationState(device)
        assert owner.observation_manager.group_obs_dim["state"] == (312,)
        assert owner.observation_manager.active_terms["state"] == [
            "joints",
            "joint_velocities",
            "parcel",
            "tool",
            "suction",
            "goal",
            "collision_point_cloud",
            "actions",
        ]
        assert owner.action_manager.total_action_dim == 4
        assert not owner.cfg.commands.sort.continue_after_delivery
        names = ("delivery", "failure", "effort", "action_rate", "joint_velocity")
        if set(owner.reward_manager.active_terms) != set(names):
            raise ValueError("The sparse adapter requires the native conveyor reward terms.")
        self.weights = torch.tensor(
            [owner.reward_manager.get_term_cfg(name).weight * owner.step_dt for name in names], device=device
        )
        scale = owner.cfg.observations.state.joint_velocities.scale
        self.velocity_scale = 1.0 if scale is None else float(scale)
        self.cloud_noise_cfg = owner.cfg.observations.policy.collision_point_cloud.noise

    def actor_obs(
        self, state: State, goal: torch.Tensor, context: torch.Tensor, prev_actions: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat((*state[:3], goal, state[3], prev_actions[..., -1, :].clamp(-1.0, 1.0)), dim=-1)

    def actor_goal(self, state: State, goal: torch.Tensor) -> torch.Tensor:
        return goal

    def advance(
        self, next_state: State, goal: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return goal, context

    def dynamics_context(self, state: State, context: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
        return torch.cat((context, prev_actions.flatten(-2)), dim=-1)

    def critic_obs(
        self, state: State, goal: torch.Tensor, context: torch.Tensor, prev_actions: torch.Tensor
    ) -> torch.Tensor:
        return self.actor_obs(state, goal, context, prev_actions)

    def reward(
        self,
        state: State,
        actions: torch.Tensor,
        next_state: State,
        goal: torch.Tensor,
        prev_actions: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        accepted = actions.clamp(-1.0, 1.0)
        velocity = torch.nan_to_num(next_state[0][..., 7:14] / self.velocity_scale).clamp(-3.0, 3.0)
        terms = torch.stack(
            (
                labels[..., 0],
                labels[..., 1],
                accepted[..., :3].square().sum(-1),
                (accepted - prev_actions[..., -1, :].clamp(-1.0, 1.0)).square().sum(-1),
                velocity.square().sum(-1),
            ),
            dim=-1,
        )
        return (terms * self.weights).sum(-1)

    def termination_prob(
        self, next_state: State, goal: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        return labels.sum(-1).clamp(0.0, 1.0)

    def add_noise(self, observation: torch.Tensor) -> torch.Tensor:
        cloud = observation[..., 52:308]
        if self.cloud_noise_cfg is not None:
            points = cloud.unflatten(-1, (64, 4))
            cloud = torch.cat((points[..., :3], (points[..., 3:] >= 0.5).to(points.dtype)), dim=-1).flatten(-2)
            cloud = self.cloud_noise_cfg.func(cloud.reshape(-1, 256), self.cloud_noise_cfg).reshape_as(cloud)
        return torch.cat((observation[..., :52], cloud, observation[..., 308:]), dim=-1)


class _SuctionMean(nn.Module):
    """Bound means with a 0.1% vacuum choice floor under fixed 0.1 Gaussian noise."""

    def __init__(self) -> None:
        super().__init__()
        self.vacuum_bound = 0.30902323061678133

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        vacuum = self.vacuum_bound * (output[..., -1:] / self.vacuum_bound).tanh()
        return torch.cat((output[..., :-1].tanh(), vacuum), dim=-1)


class SuctionActor(SuperMimicActor):
    """Use upstream history/export with bounded means and 98% initial seal retention."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.obs_norm = EmpiricalNormalization(self.obs_dim, eps=0.0, until=0)
        self.goal_norm = EmpiricalNormalization(self.goal_dim, eps=0.0, until=0)
        output = _SuctionMean()
        nn.init.orthogonal_(self.decoder.output.weight, gain=0.01)
        nn.init.zeros_(self.decoder.output.bias)
        with torch.no_grad():
            self.decoder.output.bias[-1] = output.vacuum_bound * math.atanh(-0.2053748910631823 / output.vacuum_bound)
        self.decoder.append(output)


class SuctionWorldModel(WorldModel):
    """Keep geometrical inputs in fixed units; normalize only predicted increments."""

    def __init__(self, representation, feature_dim: int, num_actions: int, context_dim: int, **kwargs) -> None:
        super().__init__(representation, feature_dim, num_actions, context_dim, **kwargs)
        self.in_norm = EmpiricalNormalization(feature_dim + num_actions + context_dim, eps=0.0, until=0)


class EventMonitoredSuperMimic(SuperMimic):
    """Report real sparse-event coverage and teacher-forced head calibration.

    These bounded diagnostic banks never change replay sampling or learning losses.
    They restart on resume and are not native task-success evaluation cohorts.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._native_events = torch.zeros(2, device=self.device)
        self._native_transitions = 0
        self._event_banks = [torch.empty((0, 323), device=self.device) for _ in range(3)]

    @torch.no_grad()
    def process_env_step(self, obs, rewards, dones, extras) -> None:
        final = extras["final_obs"].to(self.device)
        labels = self._set(final, "labels")
        rows = torch.cat(
            (self._set(final, "state"), self._set(self._last_obs, "context"), self._last_previous_actions.flatten(-2)),
            -1,
        )
        masks = (labels[:, 0] > 0.5, labels[:, 1] > 0.5, labels.sum(-1) == 0)
        for index, mask in enumerate(masks):
            self._event_banks[index] = torch.cat((self._event_banks[index], rows[mask][-64:]), 0)[-64:].clone()
        self._native_events += labels.sum(0)
        self._native_transitions += labels.shape[0]
        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:
        metrics = super().update()
        metrics.update(
            {
                "native_receipts_observed": self._native_events[0].item(),
                "native_failures_observed": self._native_events[1].item(),
                "native_transitions_observed": float(self._native_transitions),
                "native_receipt_label_rate": (self._native_events[0] / max(1, self._native_transitions)).item(),
            }
        )
        if self.iteration > 0 and self.iteration % 25 == 0:
            with torch.no_grad():
                for name, rows in zip(("receipt", "failure", "negative"), self._event_banks):
                    if not rows.shape[0]:
                        continue
                    state = self.task.representation.unflatten(rows[:, :312])
                    context = self.task.dynamics_context(state, rows[:, 312:315], rows[:, 315:].unflatten(-1, (2, 4)))
                    probability = self.world_model.label_logits(state, context).sigmoid()
                    metrics[f"calibration_receipt_probability_on_{name}"] = probability[:, 0].mean().item()
                    metrics[f"calibration_failure_probability_on_{name}"] = probability[:, 1].mean().item()
                    metrics[f"calibration_{name}_samples"] = float(rows.shape[0])
        return metrics
