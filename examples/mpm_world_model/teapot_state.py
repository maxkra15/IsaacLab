# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Held-teapot task measurements and bounded references for the physical IK controller.

This prototype has a separate observation/action interface from Franka Pour.
Fluid follows measured rigid boundaries in one direction; no learned fluid force
is fed back into the robot. All mass/reward measurements use the complete cloud.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .franka_pour_state import CONTEXT_NAMES

ACTION_NAMES = ("translation_x", "translation_y", "translation_z", "rotation_x", "rotation_y", "rotation_z")
SUMMARY_NAMES = (
    "com_x",
    "com_y",
    "com_z",
    "mean_vx",
    "mean_vy",
    "mean_vz",
    "kinetic_energy_j",
    "source_fraction",
    "target_fraction",
    "spilled_fraction",
    "in_flight_fraction",
    "invalid_fraction",
    "rms_speed_m_s",
    "max_speed_m_s",
    "min_z_m",
    "max_z_m",
    "cov_xx",
    "cov_yy",
    "cov_zz",
    "cov_xy",
    "cov_xz",
    "cov_yz",
)


def spatial_sample(
    position: torch.Tensor, mass: torch.Tensor, count: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose fixed spatial strata with their represented masses, without replacement.

    The Morton ordering is computed only once, before recording. A sampled
    particle keeps its identity throughout an episode. Represented masses are
    for reconstruction/learning; authoritative task metrics never use samples.
    """
    points = position.detach().cpu().numpy().astype(np.float64)
    masses = mass.detach().cpu().numpy().astype(np.float64)
    if count < 1 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Sampling needs a positive count and finite, nonempty positions.")
    if masses.shape != (len(points),) or not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("Every particle must have a finite positive mass.")
    scaled = np.clip((points - points.min(0)) / np.maximum(np.ptp(points, axis=0), 1.0e-12), 0, 1)
    coordinates = np.rint(scaled * 1023).astype(np.uint64)
    codes = np.zeros(len(points), dtype=np.uint64)
    for bit in range(10):
        for axis in range(3):
            codes |= ((coordinates[:, axis] >> bit) & 1) << (3 * bit + axis)
    order = np.argsort(codes, kind="stable")
    edges = np.linspace(0, len(order), min(count, len(order)) + 1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    indices = np.array([order[rng.integers(lo, hi)] for lo, hi in zip(edges[:-1], edges[1:])])
    weights = np.array([masses[order[lo:hi]].sum() for lo, hi in zip(edges[:-1], edges[1:])])
    return torch.as_tensor(indices, device=position.device), torch.as_tensor(weights, device=position.device)


def rigid_state(pose_xyzw: torch.Tensor, twist: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
    """Export root-link positions in environment axes and quaternions in WXYZ."""
    return torch.cat((pose_xyzw[..., :3] - origin, pose_xyzw[..., 6:7], pose_xyzw[..., 3:6], twist), dim=-1)


def point_in_body_frame(point_world: torch.Tensor, pose_xyzw: torch.Tensor) -> torch.Tensor:
    """Express a world-space point in a body's link frame (unit XYZW pose)."""
    displacement = point_world - pose_xyzw[..., :3]
    inverse_vector = -pose_xyzw[..., 3:6].expand_as(displacement)
    cross = 2.0 * torch.linalg.cross(inverse_vector, displacement)
    return displacement + pose_xyzw[..., 6:7] * cross + torch.linalg.cross(inverse_vector, cross)


def particle_summary(
    position: torch.Tensor,
    velocity: torch.Tensor,
    mass: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    spilled: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition all mass, retaining invalid and airborne particles explicitly."""
    valid = torch.isfinite(position).all(-1) & torch.isfinite(velocity).all(-1)
    target = target & valid
    source = source & valid & ~target
    spilled = spilled & valid & ~target & ~source
    airborne = valid & ~target & ~source & ~spilled
    total = mass.sum()
    fractions = torch.stack([(mass * mask).sum() / total for mask in (source, target, spilled, airborne, ~valid)])
    # Invalid particles remain visible in the last mass bin and finite_mask;
    # finite-only moments avoid poisoning every recorded feature with NaN.
    q = torch.where(valid[:, None], position, 0.0)
    v = torch.where(valid[:, None], velocity, 0.0)
    valid_mass = (mass * valid).sum().clamp_min(torch.finfo(mass.dtype).tiny)
    center = (q * mass[:, None]).sum(0) / valid_mass
    mean_velocity = (v * mass[:, None]).sum(0) / valid_mass
    delta = torch.where(valid[:, None], q - center, 0.0)
    covariance = (delta.T * mass) @ delta / valid_mass
    speed2 = v.square().sum(-1)
    z_min = torch.where(valid, position[:, 2], float("inf")).amin()
    z_max = torch.where(valid, position[:, 2], -float("inf")).amax()
    extrema = torch.stack((z_min, z_max))
    extrema = torch.where(torch.isfinite(extrema), extrema, 0.0)
    summary = torch.cat(
        (
            center,
            mean_velocity,
            (0.5 * mass * speed2).sum()[None],
            fractions,
            ((mass * speed2).sum() / valid_mass).sqrt()[None],
            speed2.amax().sqrt()[None],
            extrema,
            covariance[(0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)],
        )
    )
    return summary, valid.all()


def pour_reward(previous: torch.Tensor, current: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Reward newly received mass and penalize newly spilled mass and control effort."""
    return 100.0 * (current[8] - previous[8]) - 200.0 * (current[9] - previous[9]) - 0.001 * action.square().sum()


class TeapotResidualController:
    """Add smooth bounded desired-pot residuals without overriding physical poses.

    Actions select translation/rotation-vector offsets. A quintic interpolation
    has zero endpoint velocity, and its derivative is included in the reference
    twist passed to the existing measured-pose differential IK. The fingers and
    all robot contact dynamics remain the original demo's implementation.
    """

    def __init__(self, motion, dt: float = 0.01, translation_limit: float = 0.02, rotation_limit: float = 0.2):
        if dt <= 0 or translation_limit <= 0 or rotation_limit <= 0:
            raise ValueError("Control period and residual limits must be positive.")
        self.motion = motion
        self.dt = dt
        self.limit = np.array([translation_limit] * 3 + [rotation_limit] * 3)
        self.rate = np.array([0.05] * 3 + [0.3] * 3)
        self.previous = np.zeros(6)
        self.current = np.zeros(6)
        self.action = np.zeros(6)
        self.start_time = 0.0
        self.nominal = motion._pose_at_time
        motion._pose_at_time = self.reference

    def apply_action(self, action: np.ndarray, sim_time: float) -> None:
        """Latch one normalized action at a 100 Hz control boundary."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("A teapot action must have six finite values.")
        self.previous = self.current.copy()
        self.action = np.clip(action, -1.0, 1.0)
        # The quintic derivative peaks at 1.875. Bound the actual reference rate.
        step = self.rate * self.dt / 1.875
        self.current += np.clip(self.action * self.limit - self.current, -step, step)
        self.start_time = float(sim_time)

    def reference(self, sim_time: float):
        """Evaluate a desired pose and its consistent world-frame twist."""
        position, quaternion, twist = self.nominal(sim_time)
        u = float(np.clip((sim_time - self.start_time) / self.dt, 0, 1))
        blend = u**3 * (10 + u * (-15 + 6 * u))
        derivative = 30 * u**2 * (1 - u) ** 2 / self.dt
        residual = self.previous + blend * (self.current - self.previous)
        residual_velocity = derivative * (self.current - self.previous)
        rotation = Rotation.from_rotvec(residual[3:])
        vector = residual[3:]
        theta = np.linalg.norm(vector)
        cross = np.array([[0, -vector[2], vector[1]], [vector[2], 0, -vector[0]], [-vector[1], vector[0], 0]])
        if theta < 1.0e-5:
            jacobian = np.eye(3) + 0.5 * cross + cross @ cross / 6
        else:
            jacobian = (
                np.eye(3) + (1 - np.cos(theta)) / theta**2 * cross + (theta - np.sin(theta)) / theta**3 * cross @ cross
            )
        angular = jacobian @ residual_velocity[3:] + rotation.apply(np.asarray(twist)[3:])
        return (
            tuple(np.asarray(position) + residual[:3]),
            tuple((rotation * Rotation.from_quat(quaternion)).as_quat()),
            tuple(np.concatenate((np.asarray(twist)[:3] + residual_velocity[:3], angular))),
        )

    def state(self) -> np.ndarray:
        """Expose the applied endpoint and last action, which are needed for prediction."""
        return np.concatenate((self.current, self.action))

    def reset(self) -> None:
        """Start a new native episode without stacking reference wrappers."""
        self.previous.fill(0.0)
        self.current.fill(0.0)
        self.action.fill(0.0)
        self.start_time = 0.0


def context_tensor(values: dict[str, float], device: str) -> torch.Tensor:
    """Enforce the shared named material/geometry context instead of positional guessing."""
    if set(values) != set(CONTEXT_NAMES):
        raise ValueError(f"Context differs from the shared schema: {set(values) ^ set(CONTEXT_NAMES)}")
    return torch.tensor([values[name] for name in CONTEXT_NAMES], dtype=torch.float32, device=device)
