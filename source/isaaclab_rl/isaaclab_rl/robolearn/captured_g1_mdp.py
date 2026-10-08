# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Device-side MDP for the native flat G1 task, with fixed buffers for CUDA capture.

The scene and its startup material/mass randomization remain owned by Isaac Lab.
This module replaces the step-time Torch managers with masked Warp launches, and
uses the existing experimental reward/event kernels directly. The physics adapter
must preserve native sampling separately for contacts and joint acceleration.
``_physics_handles_decimation`` controls the acceleration update cadence; with
lazy sensors, contacts are refreshed when the MDP reads them after the control
step. The experimental frontend's decimation path is not assumed here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from isaaclab_experimental.envs.mdp import events as warp_events
from isaaclab_experimental.envs.mdp import rewards as warp_rewards
from isaaclab_experimental.envs.mdp import terminations as warp_terminations
from isaaclab_newton.kernels.state_kernels import body_ang_vel_from_root, body_lin_vel_from_root
from isaaclab_tasks_experimental.core.velocity.mdp.rewards import (
    _feet_air_time_positive_biped_kernel,
    _feet_slide_kernel,
    _track_ang_vel_z_world_exp_kernel,
    _track_lin_vel_xy_yaw_frame_exp_kernel,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@wp.struct
class _CommandConfig:
    linear_x: wp.vec2f
    linear_y: wp.vec2f
    angular_z: wp.vec2f
    heading: wp.vec2f
    resampling_time: wp.vec2f
    push_time: wp.vec2f
    push_x: wp.vec2f
    push_y: wp.vec2f
    standing_fraction: float
    heading_fraction: float
    heading_stiffness: float
    heading_enabled: bool


@wp.struct
class _CommandBuffers:
    values: wp.array(dtype=wp.float32, ndim=2)
    heading: wp.array(dtype=wp.float32)
    standing: wp.array(dtype=wp.bool)
    heading_env: wp.array(dtype=wp.bool)
    time_left: wp.array(dtype=wp.float32)
    push_time_left: wp.array(dtype=wp.float32)
    error_xy: wp.array(dtype=wp.float32)
    error_yaw: wp.array(dtype=wp.float32)
    metric_steps: wp.array(dtype=wp.int32)


@wp.func
def _resample_command(i: int, state: wp.uint32, cfg: _CommandConfig, buffers: _CommandBuffers) -> wp.uint32:
    buffers.time_left[i] = wp.randf(state, cfg.resampling_time[0], cfg.resampling_time[1])
    buffers.values[i, 0] = wp.randf(state, cfg.linear_x[0], cfg.linear_x[1])
    buffers.values[i, 1] = wp.randf(state, cfg.linear_y[0], cfg.linear_y[1])
    buffers.values[i, 2] = wp.randf(state, cfg.angular_z[0], cfg.angular_z[1])
    if cfg.heading_enabled:
        buffers.heading[i] = wp.randf(state, cfg.heading[0], cfg.heading[1])
        buffers.heading_env[i] = wp.randf(state) <= cfg.heading_fraction
    buffers.standing[i] = wp.randf(state) <= cfg.standing_fraction
    return state


@wp.kernel
def _initialize(
    seed: int,
    rng: wp.array(dtype=wp.uint32),
    reset_mask: wp.array(dtype=wp.bool),
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
    steps: wp.array(dtype=wp.int32),
    returns: wp.array(dtype=wp.float32),
    commands: _CommandBuffers,
):
    i = wp.tid()
    rng[i] = wp.rand_init(seed, i)
    reset_mask[i] = True
    terminated[i] = 0
    truncated[i] = 0
    steps[i] = 0
    returns[i] = 0.0
    commands.error_xy[i] = 0.0
    commands.error_yaw[i] = 0.0
    commands.metric_steps[i] = 0


@wp.kernel
def _process_actions(
    incoming: wp.array(dtype=wp.float32, ndim=2),
    default_positions: wp.array(dtype=wp.float32, ndim=2),
    scale: float,
    action: wp.array(dtype=wp.float32, ndim=2),
    previous_action: wp.array(dtype=wp.float32, ndim=2),
    targets: wp.array(dtype=wp.float32, ndim=2),
):
    i, j = wp.tid()
    previous_action[i, j] = action[i, j]
    action[i, j] = incoming[i, j]
    targets[i, j] = default_positions[i, j] + scale * incoming[i, j]


@wp.kernel
def _advance_termination(
    contact_termination: wp.array(dtype=wp.bool),
    max_episode_length: int,
    steps: wp.array(dtype=wp.int32),
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
    reset_mask: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    steps[i] += 1
    timeout = steps[i] >= max_episode_length
    terminated[i] = wp.int32(contact_termination[i])
    truncated[i] = wp.int32(timeout)
    reset_mask[i] = contact_termination[i] or timeout


@wp.kernel
def _sum_rewards(
    terms: wp.array(dtype=wp.float32, ndim=2),
    weights: wp.array(dtype=wp.float32),
    reset_mask: wp.array(dtype=wp.bool),
    steps: wp.array(dtype=wp.int32),
    rewards: wp.array(dtype=wp.float32),
    returns: wp.array(dtype=wp.float32),
    episode_totals: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    reward = float(0.0)
    for k in range(weights.shape[0]):
        reward += terms[k, i] * weights[k]
    rewards[i] = reward
    returns[i] += reward
    if reset_mask[i]:
        wp.atomic_add(episode_totals, 0, returns[i])
        wp.atomic_add(episode_totals, 1, float(steps[i]))
        wp.atomic_add(episode_totals, 2, 1.0)


@wp.kernel
def _reset_metadata(
    reset_mask: wp.array(dtype=wp.bool),
    rng: wp.array(dtype=wp.uint32),
    cfg: _CommandConfig,
    commands: _CommandBuffers,
    default_positions: wp.array(dtype=wp.float32, ndim=2),
    steps: wp.array(dtype=wp.int32),
    returns: wp.array(dtype=wp.float32),
    action: wp.array(dtype=wp.float32, ndim=2),
    previous_action: wp.array(dtype=wp.float32, ndim=2),
    targets: wp.array(dtype=wp.float32, ndim=2),
    metric_totals: wp.array(dtype=wp.float32),
    xy_success_threshold: float,
    yaw_success_threshold: float,
):
    i = wp.tid()
    if not reset_mask[i]:
        return
    if steps[i] > 0:
        metric_steps = float(wp.max(commands.metric_steps[i], 1))
        mean_xy = commands.error_xy[i] / metric_steps
        mean_yaw = commands.error_yaw[i] / metric_steps
        wp.atomic_add(metric_totals, 0, mean_xy)
        wp.atomic_add(metric_totals, 1, mean_yaw)
        success = mean_xy < xy_success_threshold and mean_yaw < yaw_success_threshold
        wp.atomic_add(metric_totals, 2, wp.float32(success))
    commands.error_xy[i] = 0.0
    commands.error_yaw[i] = 0.0
    commands.metric_steps[i] = 0
    steps[i] = 0
    returns[i] = 0.0
    for j in range(action.shape[1]):
        action[i, j] = 0.0
        previous_action[i, j] = 0.0
        targets[i, j] = default_positions[i, j]
    state = rng[i]
    commands.push_time_left[i] = wp.randf(state, cfg.push_time[0], cfg.push_time[1])
    state = _resample_command(i, state, cfg, commands)
    rng[i] = state


@wp.kernel
def _update_commands_and_pushes(
    root_poses: wp.array(dtype=wp.transformf),
    root_velocities: wp.array(dtype=wp.spatial_vectorf),
    rng: wp.array(dtype=wp.uint32),
    cfg: _CommandConfig,
    commands: _CommandBuffers,
    dt: float,
    push_mask: wp.array(dtype=wp.bool),
    push_velocities: wp.array(dtype=wp.spatial_vectorf),
):
    i = wp.tid()
    pose = root_poses[i]
    velocity = root_velocities[i]
    linear = body_lin_vel_from_root(pose, velocity)
    angular = body_ang_vel_from_root(pose, velocity)
    commands.error_xy[i] += wp.length(wp.vec2f(commands.values[i, 0] - linear[0], commands.values[i, 1] - linear[1]))
    commands.error_yaw[i] += wp.abs(commands.values[i, 2] - angular[2])
    commands.metric_steps[i] += 1
    state = rng[i]
    commands.time_left[i] -= dt
    if commands.time_left[i] <= 0.0:
        state = _resample_command(i, state, cfg, commands)
    if cfg.heading_enabled and commands.heading_env[i]:
        forward = wp.quat_rotate(wp.transform_get_rotation(pose), wp.vec3f(1.0, 0.0, 0.0))
        heading_error = commands.heading[i] - wp.atan2(forward[1], forward[0])
        # Match wrap_to_pi, including the positive-pi boundary convention.
        wrapped = (heading_error + wp.pi) - wp.floor((heading_error + wp.pi) / (2.0 * wp.pi)) * (2.0 * wp.pi)
        error = wrapped - wp.pi
        if wrapped == 0.0 and heading_error > 0.0:
            error = wp.pi
        commands.values[i, 2] = wp.clamp(cfg.heading_stiffness * error, cfg.angular_z[0], cfg.angular_z[1])
    if commands.standing[i]:
        commands.values[i, 0] = 0.0
        commands.values[i, 1] = 0.0
        commands.values[i, 2] = 0.0
    commands.push_time_left[i] -= dt
    push = commands.push_time_left[i] < 1.0e-6
    push_mask[i] = push
    if push:
        commands.push_time_left[i] = wp.randf(state, cfg.push_time[0], cfg.push_time[1])
        # Native push_by_setting_velocity adds a perturbation to the current COM velocity.
        velocity[0] += wp.randf(state, cfg.push_x[0], cfg.push_x[1])
        velocity[1] += wp.randf(state, cfg.push_y[0], cfg.push_y[1])
        push_velocities[i] = velocity
    rng[i] = state


@wp.kernel
def _observations(
    root_poses: wp.array(dtype=wp.transformf),
    root_velocities: wp.array(dtype=wp.spatial_vectorf),
    gravity: wp.array(dtype=wp.vec3f),
    positions: wp.array(dtype=wp.float32, ndim=2),
    velocities: wp.array(dtype=wp.float32, ndim=2),
    default_positions: wp.array(dtype=wp.float32, ndim=2),
    default_velocities: wp.array(dtype=wp.float32, ndim=2),
    commands: wp.array(dtype=wp.float32, ndim=2),
    actions: wp.array(dtype=wp.float32, ndim=2),
    noise_low: wp.array(dtype=wp.float32),
    noise_high: wp.array(dtype=wp.float32),
    rng: wp.array(dtype=wp.uint32),
    reset_mask: wp.array(dtype=wp.bool),
    terminal_only: bool,
    out: wp.array(dtype=wp.float32, ndim=2),
):
    i = wp.tid()
    if terminal_only and not reset_mask[i]:
        return
    pose = root_poses[i]
    linear = body_lin_vel_from_root(pose, root_velocities[i])
    angular = body_ang_vel_from_root(pose, root_velocities[i])
    projected = wp.quat_rotate_inv(wp.transform_get_rotation(pose), wp.normalize(gravity[i]))
    state = rng[i]
    for k in range(3):
        out[i, k] = linear[k] + wp.randf(state, noise_low[k], noise_high[k])
        out[i, 3 + k] = angular[k] + wp.randf(state, noise_low[3 + k], noise_high[3 + k])
        out[i, 6 + k] = projected[k] + wp.randf(state, noise_low[6 + k], noise_high[6 + k])
        out[i, 9 + k] = commands[i, k]
    joints = positions.shape[1]
    for j in range(joints):
        out[i, 12 + j] = (
            positions[i, j] - default_positions[i, j] + wp.randf(state, noise_low[12 + j], noise_high[12 + j])
        )
        out[i, 12 + joints + j] = (
            velocities[i, j]
            - default_velocities[i, j]
            + wp.randf(state, noise_low[12 + joints + j], noise_high[12 + joints + j])
        )
        out[i, 12 + 2 * joints + j] = actions[i, j]
    rng[i] = state


@wp.kernel
def _select_next_observations(
    observations: wp.array(dtype=wp.float32, ndim=2),
    reset_mask: wp.array(dtype=wp.bool),
    next_observations: wp.array(dtype=wp.float32, ndim=2),
):
    i, j = wp.tid()
    if not reset_mask[i]:
        next_observations[i, j] = observations[i, j]


class G1CapturedMDP:
    """Own fixed device state for flat G1 MDP launches, independent of physics stepping.

    Call ``pre_step``, advance the native physics and contact/acceleration buffers,
    ``post_physics``, ``prepare_resets``, apply the supplied masked physics reset,
    ``post_reset``, apply supplied masked pushes, then ``compute_observations``.
    ``next_observations`` retains pre-reset terminal observations for PPO bootstrap.
    Initial reset uses ``post_reset(initial=True)`` to preserve the native command
    resampling semantics. All allocations and name resolution happen in construction.
    """

    reward_names = (
        "track_lin_vel_xy_exp",
        "track_ang_vel_z_exp",
        "lin_vel_z_l2",
        "ang_vel_xy_l2",
        "dof_torques_l2",
        "dof_acc_l2",
        "action_rate_l2",
        "feet_air_time",
        "feet_slide",
        "flat_orientation_l2",
        "dof_pos_limits",
        "termination_penalty",
        "joint_deviation_hip",
        "joint_deviation_arms",
        "joint_deviation_fingers",
        "joint_deviation_torso",
    )

    def __init__(self, env: ManagerBasedRLEnv, joint_acc: wp.array):
        self.device = env.device
        self.num_envs = env.num_envs
        robot = env.scene["robot"]
        data = robot.data
        self.action_dim = robot.num_joints
        self.observation_dim = 12 + 3 * self.action_dim
        self.step_dt = env.step_dt
        self.max_episode_length = env.max_episode_length
        self._validate(env)
        self._root_pose = data.root_link_pose_w.warp
        self._root_velocity = data.root_com_vel_w.warp
        self._gravity = data.GRAVITY_VEC_W.warp
        self._joint_pos = data.joint_pos.warp
        self._joint_vel = data.joint_vel.warp
        self._default_joint_pos = data.default_joint_pos.warp
        self._default_joint_vel = data.default_joint_vel.warp
        self._default_root_pose = data.default_root_pose.warp
        self._default_root_vel = data.default_root_vel.warp
        self._soft_joint_limits = data.soft_joint_pos_limits.warp
        self._soft_velocity_limits = data._soft_joint_vel_limits
        self._env_origins = wp.from_torch(env.scene.env_origins, dtype=wp.vec3f)
        self._joint_ids = wp.array(list(range(self.action_dim)), dtype=wp.int32, device=self.device)
        sensor = env.scene.sensors["contact_forces"]
        contact_data = sensor.data
        self._contact_history = contact_data.net_normal_forces_w_history.warp
        self._contact_times = contact_data.current_contact_time.warp
        self._air_times = contact_data.current_air_time.warp

        shape = (self.num_envs, self.action_dim)
        self.action = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self.previous_action = wp.zeros_like(self.action)
        self.joint_targets = wp.zeros_like(self.action)
        self.joint_positions = wp.zeros_like(self.action)
        self.joint_velocities = wp.zeros_like(self.action)
        self.root_poses = wp.zeros(self.num_envs, dtype=wp.transformf, device=self.device)
        self.root_velocities = wp.zeros(self.num_envs, dtype=wp.spatial_vectorf, device=self.device)
        self.push_velocities = wp.zeros_like(self.root_velocities)
        self.push_mask = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self.reset_mask = wp.zeros_like(self.push_mask)
        self.terminated = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        self.truncated = wp.zeros_like(self.terminated)
        self.episode_steps = wp.zeros_like(self.terminated)
        self.rewards = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.episode_returns = wp.zeros_like(self.rewards)
        self.episode_totals = wp.zeros(3, dtype=wp.float32, device=self.device)
        self.command_metric_totals = wp.zeros_like(self.episode_totals)
        self.rng = wp.zeros(self.num_envs, dtype=wp.uint32, device=self.device)
        self.observations = wp.zeros((self.num_envs, self.observation_dim), dtype=wp.float32, device=self.device)
        self.next_observations = wp.zeros_like(self.observations)
        self.commands = wp.zeros((self.num_envs, 3), dtype=wp.float32, device=self.device)
        self.heading_target = wp.zeros_like(self.rewards)
        self.standing_env = wp.zeros_like(self.reset_mask)
        self.heading_env = wp.zeros_like(self.reset_mask)
        self.command_time_left = wp.zeros_like(self.rewards)
        self.push_time_left = wp.zeros_like(self.rewards)
        self.command_error_xy = wp.zeros_like(self.rewards)
        self.command_error_yaw = wp.zeros_like(self.rewards)
        self.command_metric_steps = wp.zeros_like(self.episode_steps)
        self._contact_termination = wp.zeros_like(self.reset_mask)
        self.reward_terms = wp.zeros((len(self.reward_names), self.num_envs), dtype=wp.float32, device=self.device)
        self._term_outputs = [self.reward_terms[k] for k in range(len(self.reward_names))]
        self._weights = wp.array(
            [getattr(env.cfg.rewards, name).weight * self.step_dt for name in self.reward_names],
            dtype=wp.float32,
            device=self.device,
        )
        self._command_cfg, self._command_buffers = self._bind_commands(env)
        self._xy_success_threshold = env.cfg.commands.base_velocity.vel_xy_success_threshold
        self._yaw_success_threshold = env.cfg.commands.base_velocity.vel_yaw_success_threshold
        self._action_scale = float(env.cfg.actions.joint_pos.scale)
        self._noise_low, self._noise_high = self._bind_noise(env)
        self._reset_ranges = self._bind_reset_ranges(env)
        self._joint_reset_ranges = (
            *env.cfg.events.reset_robot_joints.params["position_range"],
            *env.cfg.events.reset_robot_joints.params["velocity_range"],
        )
        self._termination_ids = wp.array(sensor.find_bodies("torso_link")[0], dtype=wp.int32, device=self.device)
        self._termination_threshold = env.cfg.terminations.base_contact.params["threshold"]
        self._reward_launches = self._bind_rewards(env, robot, data, sensor, joint_acc)

    def reset(self, seed: int) -> None:
        """Initialize device RNG/counters and request a reset of every environment."""
        self.episode_totals.zero_()
        self.command_metric_totals.zero_()
        self.push_mask.zero_()
        wp.launch(
            _initialize,
            self.num_envs,
            inputs=[
                seed,
                self.rng,
                self.reset_mask,
                self.terminated,
                self.truncated,
                self.episode_steps,
                self.episode_returns,
                self._command_buffers,
            ],
            device=self.device,
        )

    def pre_step(self, actions: wp.array) -> None:
        """Remember the raw environment actions and compute native affine joint targets."""
        wp.launch(
            _process_actions,
            (self.num_envs, self.action_dim),
            inputs=[
                actions,
                self._default_joint_pos,
                self._action_scale,
                self.action,
                self.previous_action,
                self.joint_targets,
            ],
            device=self.device,
        )

    def post_physics(self) -> None:
        """Compute native terminations/rewards and retain terminal observations before reset."""
        wp.launch(
            warp_terminations._illegal_contact_kernel,
            self.num_envs,
            inputs=[
                self._contact_history,
                self._termination_ids,
                self._termination_threshold,
                self._contact_termination,
            ],
            device=self.device,
        )
        wp.launch(
            _advance_termination,
            self.num_envs,
            inputs=[
                self._contact_termination,
                self.max_episode_length,
                self.episode_steps,
                self.terminated,
                self.truncated,
                self.reset_mask,
            ],
            device=self.device,
        )
        for kernel, inputs in self._reward_launches:
            wp.launch(kernel, self.num_envs, inputs=inputs, device=self.device)
        wp.launch(
            _sum_rewards,
            self.num_envs,
            inputs=[
                self.reward_terms,
                self._weights,
                self.reset_mask,
                self.episode_steps,
                self.rewards,
                self.episode_returns,
                self.episode_totals,
            ],
            device=self.device,
        )
        self._observe(self.next_observations, terminal_only=True)

    def prepare_resets(self) -> None:
        """Sample native root and joint reset states for the reset mask."""
        wp.launch(
            warp_events._reset_root_state_uniform_kernel,
            self.num_envs,
            inputs=[
                self.reset_mask,
                self.rng,
                self._default_root_pose,
                self._default_root_vel,
                self._env_origins,
                self.root_poses,
                self.root_velocities,
                *self._reset_ranges,
            ],
            device=self.device,
        )
        wp.launch(
            warp_events._reset_joints_by_scale_kernel,
            self.num_envs,
            inputs=[
                self.reset_mask,
                self._joint_ids,
                self.rng,
                self._default_joint_pos,
                self._default_joint_vel,
                self.joint_positions,
                self.joint_velocities,
                self._soft_joint_limits,
                self._soft_velocity_limits,
                *self._joint_reset_ranges,
            ],
            device=self.device,
        )

    def post_reset(self, initial: bool = False) -> None:
        """Reset masked metadata, update commands, and prepare interval velocity pushes."""
        wp.launch(
            _reset_metadata,
            self.num_envs,
            inputs=[
                self.reset_mask,
                self.rng,
                self._command_cfg,
                self._command_buffers,
                self._default_joint_pos,
                self.episode_steps,
                self.episode_returns,
                self.action,
                self.previous_action,
                self.joint_targets,
                self.command_metric_totals,
                self._xy_success_threshold,
                self._yaw_success_threshold,
            ],
            device=self.device,
        )
        if initial:
            self.push_mask.zero_()
            return
        wp.launch(
            _update_commands_and_pushes,
            self.num_envs,
            inputs=[
                self._root_pose,
                self._root_velocity,
                self.rng,
                self._command_cfg,
                self._command_buffers,
                self.step_dt,
                self.push_mask,
                self.push_velocities,
            ],
            device=self.device,
        )

    def compute_observations(self) -> None:
        """Build the next policy observation after masked resets, commands, and pushes."""
        self._observe(self.observations, terminal_only=False)
        wp.launch(
            _select_next_observations,
            (self.num_envs, self.observation_dim),
            inputs=[self.observations, self.reset_mask, self.next_observations],
            device=self.device,
        )

    def _observe(self, out: wp.array, terminal_only: bool) -> None:
        wp.launch(
            _observations,
            self.num_envs,
            inputs=[
                self._root_pose,
                self._root_velocity,
                self._gravity,
                self._joint_pos,
                self._joint_vel,
                self._default_joint_pos,
                self._default_joint_vel,
                self.commands,
                self.action,
                self._noise_low,
                self._noise_high,
                self.rng,
                self.reset_mask,
                terminal_only,
                out,
            ],
            device=self.device,
        )

    def _validate(self, env: ManagerBasedRLEnv) -> None:
        expected_observations = [
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
        if self.action_dim != 37 or env.observation_manager.active_terms["policy"] != expected_observations:
            raise ValueError("Captured G1 requires the native 37-joint, 123-observation flat task.")
        if set(env.reward_manager.active_terms) != set(self.reward_names):
            raise ValueError("Captured G1 requires the native flat G1 reward terms.")
        if env.termination_manager.active_terms != ["time_out", "base_contact"]:
            raise ValueError("Captured G1 supports only native timeout and torso-contact terminations.")
        if env.command_manager.active_terms != ["base_velocity"]:
            raise ValueError("Captured G1 supports only the native base_velocity command.")
        action_cfg = env.cfg.actions.joint_pos
        if not isinstance(action_cfg.scale, (float, int)) or not action_cfg.use_default_offset or action_cfg.clip:
            raise ValueError("Captured G1 requires a scalar joint-position scale, default offset, and no term clip.")
        action_ids = env.scene["robot"].find_joints(action_cfg.joint_names)[0]
        if action_ids != list(range(self.action_dim)):
            raise ValueError("Captured G1 actions must select all joints in asset order.")
        events = env.event_manager.active_terms
        if events.get("reset") != ["base_external_force_torque", "reset_base", "reset_robot_joints"]:
            raise ValueError("Captured G1 requires the native three reset events.")
        if events.get("interval") != ["push_robot"]:
            raise ValueError("Captured G1 supports only the native push_robot interval event.")
        push_cfg = env.cfg.events.push_robot
        if push_cfg.is_global_time or not push_cfg.resample_interval_on_reset:
            raise ValueError("Captured G1 requires independent push timers resampled at episode reset.")
        if set(push_cfg.params["velocity_range"]) - {"x", "y"}:
            raise ValueError("Captured G1 supports planar interval velocity pushes.")
        force_cfg = env.cfg.events.base_external_force_torque
        if force_cfg.params["force_range"] != (0.0, 0.0) or force_cfg.params["torque_range"] != (0.0, 0.0):
            raise ValueError("Captured G1 requires zero reset external force and torque.")
        if env.curriculum_manager.active_terms or env.cfg.scene.terrain.terrain_type != "plane":
            raise ValueError("Captured G1 requires flat terrain without a curriculum.")

    def _bind_commands(self, env: ManagerBasedRLEnv) -> tuple[_CommandConfig, _CommandBuffers]:
        command = env.cfg.commands.base_velocity
        push = env.cfg.events.push_robot
        cfg = _CommandConfig()
        cfg.linear_x = wp.vec2f(*command.ranges.lin_vel_x)
        cfg.linear_y = wp.vec2f(*command.ranges.lin_vel_y)
        cfg.angular_z = wp.vec2f(*command.ranges.ang_vel_z)
        cfg.heading = wp.vec2f(*(command.ranges.heading or (0.0, 0.0)))
        cfg.resampling_time = wp.vec2f(*command.resampling_time_range)
        cfg.push_time = wp.vec2f(*push.interval_range_s)
        cfg.push_x = wp.vec2f(*push.params["velocity_range"].get("x", (0.0, 0.0)))
        cfg.push_y = wp.vec2f(*push.params["velocity_range"].get("y", (0.0, 0.0)))
        cfg.standing_fraction = command.rel_standing_envs
        cfg.heading_fraction = command.rel_heading_envs
        cfg.heading_stiffness = command.heading_control_stiffness
        cfg.heading_enabled = command.heading_command
        buffers = _CommandBuffers()
        buffers.values = self.commands
        buffers.heading = self.heading_target
        buffers.standing = self.standing_env
        buffers.heading_env = self.heading_env
        buffers.time_left = self.command_time_left
        buffers.push_time_left = self.push_time_left
        buffers.error_xy = self.command_error_xy
        buffers.error_yaw = self.command_error_yaw
        buffers.metric_steps = self.command_metric_steps
        return cfg, buffers

    def _bind_noise(self, env: ManagerBasedRLEnv) -> tuple[wp.array, wp.array]:
        low = [0.0] * self.observation_dim
        high = [0.0] * self.observation_dim
        policy = env.cfg.observations.policy
        if policy.enable_corruption:
            offsets = [
                ("base_lin_vel", 0, 3),
                ("base_ang_vel", 3, 3),
                ("projected_gravity", 6, 3),
                ("joint_pos", 12, self.action_dim),
                ("joint_vel", 12 + self.action_dim, self.action_dim),
            ]
            for name, offset, width in offsets:
                noise = getattr(policy, name).noise
                if noise is not None:
                    if noise.operation != "add" or not hasattr(noise, "n_min"):
                        raise ValueError("Captured G1 supports native additive uniform observation noise.")
                    low[offset : offset + width] = [float(noise.n_min)] * width
                    high[offset : offset + width] = [float(noise.n_max)] * width
        return wp.array(low, dtype=wp.float32, device=self.device), wp.array(high, dtype=wp.float32, device=self.device)

    def _bind_reset_ranges(self, env: ManagerBasedRLEnv) -> tuple[wp.vec3f, ...]:
        params = env.cfg.events.reset_base.params
        ranges = []
        for key in ("pose_range", "velocity_range"):
            values = [params[key].get(axis, (0.0, 0.0)) for axis in ("x", "y", "z", "roll", "pitch", "yaw")]
            for start in (0, 3):
                ranges.extend(wp.vec3f(*(values[start + j][bound] for j in range(3))) for bound in (0, 1))
        return tuple(ranges)

    def _bind_rewards(self, env, robot, data, sensor, joint_acc: wp.array) -> list[tuple]:
        outputs = dict(zip(self.reward_names, self._term_outputs))
        cfg = env.cfg.rewards

        def joint_mask(term_name):
            entity = getattr(cfg, term_name).params["asset_cfg"]
            ids = robot.find_joints(entity.joint_names or ".*")[0]
            return wp.array([j in ids for j in range(self.action_dim)], dtype=wp.bool, device=self.device)

        foot_entity = cfg.feet_slide.params["sensor_cfg"]
        asset_foot_entity = cfg.feet_slide.params["asset_cfg"]
        # Preserve each native selector's order, including its pairing in feet_slide.
        sensor_foot_ids = wp.array(sensor.find_bodies(foot_entity.body_names)[0], dtype=wp.int32, device=self.device)
        asset_foot_ids = wp.array(
            robot.find_bodies(asset_foot_entity.body_names)[0], dtype=wp.int32, device=self.device
        )
        # References in this fixed launch list keep masks and zero-copy views alive.
        launches = [
            (
                _track_lin_vel_xy_yaw_frame_exp_kernel,
                [
                    data.root_quat_w.warp,
                    data.root_lin_vel_w.warp,
                    self.commands,
                    1.0 / cfg.track_lin_vel_xy_exp.params["std"] ** 2,
                    outputs["track_lin_vel_xy_exp"],
                ],
            ),
            (
                _track_ang_vel_z_world_exp_kernel,
                [
                    data.root_ang_vel_w.warp,
                    self.commands,
                    1.0 / cfg.track_ang_vel_z_exp.params["std"] ** 2,
                    outputs["track_ang_vel_z_exp"],
                ],
            ),
            (warp_rewards._lin_vel_z_l2_kernel, [self._root_pose, self._root_velocity, outputs["lin_vel_z_l2"]]),
            (warp_rewards._ang_vel_xy_l2_kernel, [self._root_pose, self._root_velocity, outputs["ang_vel_xy_l2"]]),
            (
                warp_rewards._sum_sq_masked_kernel,
                [robot.actuators.applied_effort.warp, joint_mask("dof_torques_l2"), outputs["dof_torques_l2"]],
            ),
            (warp_rewards._sum_sq_masked_kernel, [joint_acc, joint_mask("dof_acc_l2"), outputs["dof_acc_l2"]]),
            (warp_rewards._sum_sq_diff_2d_kernel, [self.action, self.previous_action, outputs["action_rate_l2"]]),
            (
                _feet_air_time_positive_biped_kernel,
                [
                    self._air_times,
                    self._contact_times,
                    sensor_foot_ids,
                    self.commands,
                    cfg.feet_air_time.params["threshold"],
                    outputs["feet_air_time"],
                ],
            ),
            (
                _feet_slide_kernel,
                [
                    data.body_lin_vel_w.warp,
                    self._contact_history,
                    asset_foot_ids,
                    sensor_foot_ids,
                    self._contact_history.shape[1],
                    outputs["feet_slide"],
                ],
            ),
            (
                warp_rewards._flat_orientation_l2_kernel,
                [self._root_pose, self._gravity, outputs["flat_orientation_l2"]],
            ),
            (
                warp_rewards._joint_pos_limits_kernel,
                [self._joint_pos, self._soft_joint_limits, joint_mask("dof_pos_limits"), outputs["dof_pos_limits"]],
            ),
            (warp_rewards._is_terminated_kernel, [self._contact_termination, outputs["termination_penalty"]]),
        ]
        for name in ("joint_deviation_hip", "joint_deviation_arms", "joint_deviation_fingers", "joint_deviation_torso"):
            launches.append(
                (
                    warp_rewards._sum_abs_diff_masked_kernel,
                    [self._joint_pos, self._default_joint_pos, joint_mask(name), outputs[name]],
                )
            )
        return launches
