# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared task semantics; backend adapters only integrate dynamics and render."""

import math

import torch

ANYMAL_JOINTS = [f"{leg}_{axis}" for leg in ("LF", "RF", "LH", "RH") for axis in ("HAA", "HFE", "KFE")]
ANYMAL_POSE = [0.0, 0.4, -0.8, 0.0, 0.4, -0.8, 0.0, -0.4, 0.8, 0.0, -0.4, 0.8]
FRANKA_JOINTS = [f"panda_joint{index}" for index in range(1, 8)]


class Task:
    """Vectorized balance, forward walking or reaching with shared rewards and resets."""

    def __init__(self, adapter, config: dict):
        self.adapter = adapter
        self.config = config
        self.device = config["device"]
        self.n = config["num_envs"]
        self.walk = config["task"] == "anymal"
        self.reach = config["task"] == "franka"
        expected = FRANKA_JOINTS if self.reach else ANYMAL_JOINTS if self.walk else ["slider", "pole"]
        if list(adapter.joint_names) != expected:
            raise ValueError(f"Canonical joint order mismatch: {adapter.joint_names} != {expected}")
        self.action_dim = 7 if self.reach else 12 if self.walk else 1
        self.generator = torch.Generator(device=self.device).manual_seed(config["seed"])
        pose = config["initial_joint_pos"] if self.reach else ANYMAL_POSE if self.walk else [0.0, 0.0]
        self.nominal = torch.tensor(pose, device=self.device)
        self.error_key = (
            "reaching_error_m" if self.reach else "tracking_error_m_s" if self.walk else "pole_angle_abs_rad"
        )
        self.error_threshold = 0.05 if self.reach else 0.25 if self.walk else 0.1
        if self.reach:
            self.joint_lower = torch.tensor(config["joint_lower_limits"], device=self.device)
            self.joint_upper = torch.tensor(config["joint_upper_limits"], device=self.device)
            origins = []
            for joint in config["kinematic_chain"]:
                if joint["axis"] != [0.0, 0.0, 1.0]:
                    raise ValueError("The pinned Panda chain requires revolute z axes.")
                roll, pitch, yaw = joint["origin_rpy"]
                cr, sr, cp, sp, cy, sy = (
                    math.cos(roll),
                    math.sin(roll),
                    math.cos(pitch),
                    math.sin(pitch),
                    math.cos(yaw),
                    math.sin(yaw),
                )
                x, y, z = joint["origin_xyz"]
                origins.append(
                    [
                        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, x],
                        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, y],
                        [-sp, cp * sr, cp * cr, z],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            self._joint_origins = torch.tensor(origins, device=self.device)
            self._ee_offset = torch.tensor(config["end_effector_offset"], device=self.device)
            self.target_pos = torch.zeros((self.n, 3), device=self.device)
        self.last_action = torch.zeros((self.n, self.action_dim), device=self.device)
        self.age = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.horizon = round((10.0 if self.walk else 5.0) / (config["dt"] * config["decimation"]))
        self.images = None
        self.max_root_displacement = torch.zeros((), device=self.device)

    def forward_kinematics(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """Return the Panda grasp point [m] in the fixed-base world frame from URDF joint origins."""
        transform = torch.eye(4, device=self.device).expand(len(joint_pos), -1, -1).clone()
        rotation = torch.eye(4, device=self.device).expand_as(transform).clone()
        for index, origin in enumerate(self._joint_origins):
            cosine, sine = joint_pos[:, index].cos(), joint_pos[:, index].sin()
            rotation[:, 0, 0], rotation[:, 0, 1] = cosine, -sine
            rotation[:, 1, 0], rotation[:, 1, 1] = sine, cosine
            transform = transform @ origin @ rotation
        return transform[:, :3, 3] + transform[:, :3, :3] @ self._ee_offset

    def reset(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Reset joint positions [rad or m] using the task's seeded generator."""
        if env_ids is None:
            env_ids = torch.arange(self.n, device=self.device)
        size = (len(env_ids), len(self.nominal))
        noise = torch.rand(size, device=self.device, generator=self.generator) * 2 - 1
        joints = self.nominal + noise * (0.05 if self.walk else 0.1)
        if self.reach:
            joints = joints.clamp(self.joint_lower, self.joint_upper)
            # Sampling bounded joint goals makes target reachability explicit;
            # both engines receive the same target distribution and random stream.
            goal_noise = torch.rand(size, device=self.device, generator=self.generator) * 2 - 1
            goal = (self.nominal + goal_noise * 0.4).clamp(self.joint_lower, self.joint_upper)
            self.target_pos[env_ids] = self.forward_kinematics(goal)
        self.adapter.reset(env_ids, joint_pos=joints)
        self.age[env_ids] = 0
        self.last_action[env_ids] = 0
        if self.config["vision"] == "rgb":
            self.images = self.adapter.render()
        return self.observe(self.adapter.state())

    def observe(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return observations in shared coordinates; velocities use [m/s, rad/s]."""
        if self.reach:
            obs = torch.cat(
                (
                    state["joint_pos"] - self.nominal,
                    state["joint_vel"] * 0.05,
                    state["ee_pos"],
                    self.target_pos - state["ee_pos"],
                    self.last_action,
                ),
                -1,
            )
        elif not self.walk:
            q, v = state["joint_pos"], state["joint_vel"]
            obs = torch.stack((q[:, 0], v[:, 0] * 0.2, torch.sin(q[:, 1]), torch.cos(q[:, 1]), v[:, 1] * 0.2), -1)
        else:
            q = state["root_quat"]
            # Gravity projected into the base frame for wxyz quaternions.
            w, x, y, z = q.unbind(-1)
            gravity = torch.stack((2 * (w * y - x * z), -2 * (w * x + y * z), 2 * (x * x + y * y) - 1), -1)
            obs = torch.cat(
                (
                    gravity,
                    state["root_lin_vel"],
                    state["root_ang_vel"] * 0.25,
                    state["joint_pos"] - self.nominal,
                    state["joint_vel"] * 0.05,
                    self.last_action,
                ),
                -1,
            )
        return obs.clamp(-20, 20)

    def step(self, action: torch.Tensor, auto_reset: bool = True) -> tuple:
        """Advance one control interval [s] and compute task reward/termination."""
        action = action.clamp(-1, 1)
        if self.reach:
            target = (self.nominal + action * 0.75).clamp(self.joint_lower, self.joint_upper)
        else:
            target = self.nominal + action * 0.35 if self.walk else action * 20.0
        self.adapter.apply_action(target)
        for _ in range(self.config["decimation"]):
            self.adapter.physics_step()
        if self.config["vision"] == "rgb":
            self.images = self.adapter.render()
        state = self.adapter.state()
        displacement = state["root_pos"][:, :2].norm(dim=-1).max()
        torch.maximum(self.max_root_displacement, displacement, out=self.max_root_displacement)
        self.age += 1
        if self.reach:
            error = (self.target_pos - state["ee_pos"]).norm(dim=-1)
            reward = (
                1.0
                - torch.tanh(error * 5.0)
                + 0.25 * (error <= self.error_threshold).float()
                - 0.0001 * state["joint_vel"].square().sum(-1)
                - 0.01 * (action - self.last_action).square().sum(-1)
            )
            fallen = (
                (state["joint_pos"] < self.joint_lower - 0.1) | (state["joint_pos"] > self.joint_upper + 0.1)
            ).any(-1)
            info = {self.error_key: error, "fallen": fallen.float()}
        elif self.walk:
            quat = state["root_quat"]
            upright = 1 - 2 * (quat[:, 1].square() + quat[:, 2].square())
            error = state["root_lin_vel"][:, 0] - 1.0
            reward = (
                torch.exp(-error.square() / 0.25)
                + 0.25 * upright
                - 0.05 * state["root_lin_vel"][:, 1].square()
                - 0.001 * state["joint_vel"].square().sum(-1)
                - 0.01 * (action - self.last_action).square().sum(-1)
            )
            fallen = (state["root_pos"][:, 2] < 0.25) | (upright < 0.4)
            info = {"tracking_error_m_s": error.abs(), "fallen": fallen.float()}
        else:
            q, v = state["joint_pos"], state["joint_vel"]
            angle = torch.atan2(torch.sin(q[:, 1]), torch.cos(q[:, 1]))
            fallen = (angle.abs() > 0.5) | (q[:, 0].abs() > 2.4)
            reward = 1.0 - angle.square() - 0.05 * q[:, 0].square() - 0.001 * v.square().sum(-1)
            info = {"pole_angle_abs_rad": angle.abs(), "fallen": fallen.float()}
        finite = torch.isfinite(reward) & torch.isfinite(state["joint_pos"]).all(-1)
        if not bool(finite.all()):
            raise FloatingPointError("Non-finite physics state or reward")
        reward = reward - fallen.float() * 2
        timeout = self.age >= self.horizon
        done = fallen | timeout
        self.last_action.copy_(action)
        obs = self.observe(state)
        # Preserve the terminal observation for correct time-limit bootstrapping.
        info["terminal_observation"] = obs.clone()
        info["timeout"] = timeout & ~fallen
        if self.images is not None and bool(info["timeout"].any()):
            info["terminal_images"] = self.images[info["timeout"]]
        if auto_reset and bool(done.any()):
            self.reset(done.nonzero().flatten())
            obs = self.observe(self.adapter.state())
        return obs, reward, done, info
