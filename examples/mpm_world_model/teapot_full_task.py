# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Full grasp--lift--pour controls and geometric task measurements.

This module has no scene, trajectory, IK, or teacher dependencies. All policy
commands act on the physical robot's joint targets. Grasp measurements are
explicitly geometric proxies, not measurements of contact force or force closure.
"""

from __future__ import annotations

import math

import torch

FULL_TASK_ID = "rizon_sharpa_teapot_full"
FULL_OBSERVATION_DIM = 182
FULL_ACTION_DIM = 30
FULL_POLICY_DT = 0.01
FULL_EPISODE_SECONDS = 18.0
FULL_ELAPSED_INDEX = 179
FULL_REMAINING_INDEX = 180
FULL_GOAL_INDEX = 181

# Native Rizon scene reset and calibrated palm position near a handle grasp.
# The latter is a geometric reward reference, never a commanded robot pose.
INITIAL_ROOT_Z = 0.90041
PALM_IN_POT_POSITION = (-0.206058637337, -0.066785491216, 0.060669620349)
NEAR_GRASP_DISTANCE_M = 0.08
MINIMUM_LIFT_M = 0.05
MAXIMUM_SPILL_FRACTION = 0.1
HAND_STIFFNESS_MIN = 24.0
HAND_STIFFNESS_MAX = 100.0
HAND_STIFFNESS_RATE = 500.0


class FullJointController:
    """Rate-limit 29 normalized joint offsets and one hand-stiffness command.

    Joint offsets are relative to the authored ungrasped reset. Their scales
    reach both hard limits, including asymmetric ranges. A zero joint action
    holds the initial targets; the independent zero stiffness action requests
    the midpoint gain. Reset itself restores the native initial gain of24.
    """

    def __init__(self, initial, lower, upper, speed_limits, dt: float = FULL_POLICY_DT):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("The control timestep must be finite and positive.")
        self.initial = torch.as_tensor(initial).detach().clone()
        if self.initial.shape != (29,) or not self.initial.is_floating_point():
            raise ValueError("Initial joint positions must be a floating tensor with29 entries.")
        self.lower, self.upper, self.speed_limits = [
            torch.as_tensor(value, device=self.initial.device, dtype=self.initial.dtype).detach().clone()
            for value in (lower, upper, speed_limits)
        ]
        if any(value.shape != (29,) for value in (self.lower, self.upper, self.speed_limits)):
            raise ValueError("Joint limits and rated speeds must have29 entries.")
        if not all(
            bool(torch.isfinite(value).all()) for value in (self.initial, self.lower, self.upper, self.speed_limits)
        ):
            raise ValueError("Initial positions, joint limits and rated speeds must be finite.")
        if bool((self.lower >= self.upper).any()) or bool((self.speed_limits <= 0).any()):
            raise ValueError("Joint ranges and rated speeds must be strictly positive.")
        if bool(((self.initial < self.lower) | (self.initial > self.upper)).any()):
            raise ValueError("The ungrasped reset must lie inside the hard joint limits.")
        self.dt = float(dt)
        self.scale = torch.maximum((self.upper - self.initial).abs(), (self.lower - self.initial).abs())
        self.reset()

    def reset(self) -> None:
        """Restore command state without moving any scene body or joint."""
        self.current = self.initial.clone()
        self.previous = self.initial.clone()
        self.action = self.initial.new_zeros(FULL_ACTION_DIM)
        self.stiffness = self.initial.new_tensor(HAND_STIFFNESS_MIN)

    def encode_command(self, target, gain) -> torch.Tensor:
        """Encode a physical teacher command without executing a teacher controller."""
        target = torch.as_tensor(target, device=self.initial.device, dtype=self.initial.dtype)
        gain = torch.as_tensor(gain, device=self.initial.device, dtype=self.initial.dtype)
        if target.shape != (29,) or gain.numel() != 1:
            raise ValueError("A command needs29 joint targets and one stiffness value.")
        if not bool(torch.isfinite(target).all() & torch.isfinite(gain).all()):
            raise ValueError("Encoded targets and stiffness must be finite.")
        target = torch.maximum(self.lower, torch.minimum(self.upper, target))
        normalized_gain = 2 * (gain.reshape(()) - HAND_STIFFNESS_MIN) / (HAND_STIFFNESS_MAX - HAND_STIFFNESS_MIN) - 1
        return torch.cat(((target - self.initial) / self.scale, normalized_gain.reshape(1))).clamp(-1, 1)

    def apply_action(self, action, sim_time: float) -> None:
        """Latch one policy command; time never selects a scripted phase or target."""
        action = torch.as_tensor(action, device=self.initial.device, dtype=self.initial.dtype)
        if action.shape != (FULL_ACTION_DIM,) or not bool(torch.isfinite(action).all()):
            raise ValueError("A full-control action must contain30 finite values.")
        if not math.isfinite(sim_time):
            raise ValueError("Simulation time must be finite.")
        self.action = action.clamp(-1, 1).clone()
        desired = self.initial + self.scale * self.action[:29]
        desired = torch.maximum(self.lower, torch.minimum(self.upper, desired))
        self.previous = self.current.clone()
        increment = (desired - self.previous).clamp(-self.speed_limits * self.dt, self.speed_limits * self.dt)
        self.current = torch.maximum(self.lower, torch.minimum(self.upper, self.previous + increment))
        desired_gain = HAND_STIFFNESS_MIN + 0.5 * (self.action[-1] + 1) * (HAND_STIFFNESS_MAX - HAND_STIFFNESS_MIN)
        self.stiffness = self.stiffness + (desired_gain - self.stiffness).clamp(
            -HAND_STIFFNESS_RATE * self.dt, HAND_STIFFNESS_RATE * self.dt
        )

    def state(self) -> torch.Tensor:
        """Return current targets29, clipped action30 and applied stiffness1."""
        return torch.cat((self.current, self.action, self.stiffness.reshape(1)))


def _observation(observation) -> tuple[torch.Tensor, torch.Tensor]:
    observation = torch.as_tensor(observation)
    if observation.ndim < 1 or observation.shape[-1] != FULL_OBSERVATION_DIM or not observation.is_floating_point():
        raise ValueError("A full-task observation must end in182 floating values.")
    finite = torch.isfinite(observation).all(-1)
    return torch.nan_to_num(observation, nan=0, posinf=0, neginf=0), finite


def full_phase_metrics(observation, *, initial_root_z: float = INITIAL_ROOT_Z) -> dict[str, torch.Tensor]:
    """Measure current reach, lift and delivery without a time-dependent phase.

    Observations contain q29,v29,source13,target13,particle summary22,
    controller60,palm13,elapsed/remaining/goal3. All poses use environment-local
    XYZ and WXYZ quaternion order. The grasp proxy compares the measured palm
    with the calibrated handle-relative palm reference in the measured pot frame.
    """
    obs, finite = _observation(observation)
    source_position, source_quaternion = obs[..., 58:61], obs[..., 61:65]
    palm_position = obs[..., 166:169]
    quaternion_norm = torch.linalg.vector_norm(source_quaternion, dim=-1, keepdim=True)
    quaternion = source_quaternion / quaternion_norm.clamp_min(1e-12)
    vector = palm_position - source_position
    cross = torch.linalg.cross(quaternion[..., 1:], vector)
    relative_palm = vector - 2 * quaternion[..., :1] * cross + 2 * torch.linalg.cross(quaternion[..., 1:], cross)
    reference = obs.new_tensor(PALM_IN_POT_POSITION)
    distance = torch.linalg.vector_norm(relative_palm - reference, dim=-1)
    height = (source_position[..., 2] - initial_root_z).clamp_min(0)
    pose_valid = quaternion_norm.squeeze(-1) > 1e-6
    for start in (74, 169):
        pose_valid &= torch.linalg.vector_norm(obs[..., start : start + 4], dim=-1) > 1e-6
    fractions = obs[..., 91:96]
    mass_valid = (
        (fractions >= -1e-6).all(-1) & (fractions <= 1 + 1e-6).all(-1) & ((fractions.sum(-1) - 1).abs() <= 1e-3)
    )
    valid = finite & pose_valid & mass_valid & (obs[..., 95] <= 1e-6)
    near = valid & (distance <= NEAR_GRASP_DISTANCE_M)
    return {
        "palm_grasp_distance_m": distance,
        "lift_height_m": height,
        "near_grasp_proxy": near,
        "lifted_grasp_proxy": near & (height >= MINIMUM_LIFT_M),
        "source_fraction": obs[..., 91],
        "target_fraction": obs[..., 92],
        "spill_fraction": obs[..., 93],
        "in_flight_fraction": obs[..., 94],
        "invalid_fraction": obs[..., 95],
        "finite": finite,
        "valid": valid,
    }


def full_task_success(observation) -> torch.Tensor:
    """Return a provisional instantaneous geometric grasp--lift--pour success.

    Source-contained particles must already be excluded from target fraction by
    the physical adapter. This helper cannot validate contact forces, force
    closure, collider penetration, or an earlier sustained-lift history.
    """
    obs, _ = _observation(observation)
    metrics = full_phase_metrics(observation)
    goal = obs[..., FULL_GOAL_INDEX]
    return (
        metrics["lifted_grasp_proxy"]
        & (goal > 0)
        & (goal <= 1)
        & (metrics["target_fraction"] >= goal)
        & (metrics["spill_fraction"] <= MAXIMUM_SPILL_FRACTION)
    )


def full_task_reward(previous, current, action) -> torch.Tensor:
    """Shape reach/lift progress and reward delivery only with a measured lift.

    Positive target-mass changes need a current or previous lifted-grasp proxy;
    target losses always retain their negative reward. Spill increments always
    subtract reward. Small potential differences help exploration without making
    approaching an unlifted pot equivalent to task success. No clocks, scripted
    targets or hidden controller states enter this calculation.
    """
    previous_obs, _ = _observation(previous)
    current_obs, _ = _observation(current)
    action = torch.as_tensor(action, device=current_obs.device, dtype=current_obs.dtype)
    if previous_obs.shape != current_obs.shape or action.shape != (*current_obs.shape[:-1], FULL_ACTION_DIM):
        raise ValueError("Reward requires matching previous/current observations and one30-value action per state.")
    action_finite = torch.isfinite(action).all(-1)
    action = torch.nan_to_num(action, nan=0, posinf=0, neginf=0)
    before, after = full_phase_metrics(previous), full_phase_metrics(current)
    reach_before = torch.exp(-before["palm_grasp_distance_m"] / 0.05)
    reach_after = torch.exp(-after["palm_grasp_distance_m"] / 0.05)
    lift_before = (before["lift_height_m"] / 0.08).clamp(0, 1) * before["near_grasp_proxy"]
    lift_after = (after["lift_height_m"] / 0.08).clamp(0, 1) * after["near_grasp_proxy"]
    delivery = after["target_fraction"] - before["target_fraction"]
    lift_gate = before["lifted_grasp_proxy"] | after["lifted_grasp_proxy"]
    qualified_delivery = torch.where(delivery > 0, delivery * lift_gate, delivery)
    action_change = action.clamp(-1, 1) - previous_obs[..., 135:165]
    reward = (
        2 * (reach_after - reach_before)
        + 10 * (lift_after - lift_before)
        + 100 * qualified_delivery
        - 200 * (after["spill_fraction"] - before["spill_fraction"])
        - 0.001 * action_change.square().sum(-1)
    )
    return torch.where(before["valid"] & after["valid"] & action_finite, reward, torch.full_like(reward, -10))
