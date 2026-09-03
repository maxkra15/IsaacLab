# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observations, phase progress, rewards, and terminations for juggling."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as functional

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import math as math_utils

from .reset import (
    GRAVITY_Z,
    JUGGLE_SPHERE_CENTER_OFFSET,
    JUGGLE_SPHERE_OPEN_HAND_POSITION,
    JUGGLE_SPHERE_PRELOAD_HAND_POSITION,
    JuggleLocalGoal,
    JugglePhase,
    local_goal_for_phase,
)
from .runtime import get_juggle_runtime_state

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def tool_state(
    env: ManagerBasedRLEnv,
    tool_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["palm_link"]),
    tool_offset: tuple[float, float, float] = JUGGLE_SPHERE_CENTER_OFFSET,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return offset tool position, orientation, linear velocity, and angular velocity.

    Position and velocity use world axes. The quaternion is scalar-last ``(x, y, z, w)``.
    """
    robot: Articulation = env.scene[tool_body_cfg.name]
    body_ids = tool_body_cfg.body_ids
    body_position = robot.data.body_pos_w.torch[:, body_ids].reshape(env.num_envs, -1, 3)[:, 0]
    body_quaternion = robot.data.body_quat_w.torch[:, body_ids].reshape(env.num_envs, -1, 4)[:, 0]
    # ``body_pos_w`` is the link-frame origin. Pair it with link-frame twist;
    # the generic ``body_*_vel_w`` fields are COM-frame velocities and would
    # make the offset-point velocity internally inconsistent.
    body_linear_velocity = robot.data.body_link_lin_vel_w.torch[:, body_ids].reshape(env.num_envs, -1, 3)[:, 0]
    body_angular_velocity = robot.data.body_link_ang_vel_w.torch[:, body_ids].reshape(env.num_envs, -1, 3)[:, 0]
    offset = body_position.new_tensor(tool_offset).expand(env.num_envs, -1)
    offset_w = math_utils.quat_apply(body_quaternion, offset)
    position = body_position + offset_w
    linear_velocity = body_linear_velocity + torch.linalg.cross(body_angular_velocity, offset_w, dim=1)
    return position, body_quaternion, linear_velocity, body_angular_velocity


def ball_position_relative_to_tool(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    tool_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["palm_link"]),
    tool_offset: tuple[float, float, float] = JUGGLE_SPHERE_CENTER_OFFSET,
) -> torch.Tensor:
    """Return ball-center displacement from the pinch center [m]."""
    ball: RigidObject = env.scene[ball_cfg.name]
    position, _, _, _ = tool_state(env, tool_body_cfg, tool_offset)
    return ball.data.root_pos_w.torch - position


def ball_velocity_relative_to_tool(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    tool_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["palm_link"]),
    tool_offset: tuple[float, float, float] = JUGGLE_SPHERE_CENTER_OFFSET,
) -> torch.Tensor:
    """Return ball linear velocity relative to the pinch center [m/s]."""
    ball: RigidObject = env.scene[ball_cfg.name]
    _, _, linear_velocity, _ = tool_state(env, tool_body_cfg, tool_offset)
    return ball.data.root_lin_vel_w.torch - linear_velocity


def ball_height_and_velocity(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Return local ball height and world linear velocity [m, m/s]."""
    ball: RigidObject = env.scene[ball_cfg.name]
    height = (ball.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(1)
    return torch.cat((height, ball.data.root_lin_vel_w.torch), dim=1)


def ball_height_above_release_hand_and_velocity(
    env: ManagerBasedRLEnv,
    target_height_gain: float = 1.0,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Return normalized height above the release hand and world velocity.

    The first component is the ball-center height above the last supported hand
    height, divided by ``target_height_gain``. The remaining three components
    are the unscaled world-frame ball linear velocity [m/s], preserving the
    four-element width of :func:`ball_height_and_velocity`.

    Args:
        env: Manager-based juggling environment.
        target_height_gain: Positive target height used to normalize the height difference [m].
        ball_cfg: Ball scene entity.

    Returns:
        Normalized release-relative height and ball linear velocity, shape ``(N, 4)``.
    """
    if target_height_gain <= 0.0:
        raise ValueError("target_height_gain must be positive.")
    ball: RigidObject = env.scene[ball_cfg.name]
    state = get_juggle_runtime_state(env)
    local_height = ball.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    normalized_height = ((local_height - state.release_heights) / float(target_height_gain)).unsqueeze(1)
    return torch.cat((normalized_height, ball.data.root_lin_vel_w.torch), dim=1)


def fingertips_relative_to_ball(
    env: ManagerBasedRLEnv,
    fingertip_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Return four fingertip positions relative to the ball center [m]."""
    robot: Articulation = env.scene[fingertip_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    positions = robot.data.body_pos_w.torch[:, fingertip_cfg.body_ids]
    return (positions - ball.data.root_pos_w.torch.unsqueeze(1)).reshape(env.num_envs, -1)


def fingertip_velocities_relative_to_ball(
    env: ManagerBasedRLEnv,
    fingertip_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Return four fingertip linear velocities relative to the ball [m/s]."""
    robot: Articulation = env.scene[fingertip_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    velocities = robot.data.body_link_lin_vel_w.torch[:, fingertip_cfg.body_ids]
    return (velocities - ball.data.root_lin_vel_w.torch.unsqueeze(1)).reshape(env.num_envs, -1)


def palm_twist(
    env: ManagerBasedRLEnv,
    tool_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["palm_link"]),
    tool_offset: tuple[float, float, float] = JUGGLE_SPHERE_CENTER_OFFSET,
) -> torch.Tensor:
    """Return pinch-center linear and palm angular velocity [m/s, rad/s]."""
    _, _, linear_velocity, angular_velocity = tool_state(env, tool_body_cfg, tool_offset)
    return torch.cat((linear_velocity, angular_velocity), dim=1)


def ball_angular_velocity(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Return world-frame ball angular velocity [rad/s]."""
    ball: RigidObject = env.scene[ball_cfg.name]
    return ball.data.root_ang_vel_w.torch


def tool_axes(
    env: ManagerBasedRLEnv,
    tool_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["palm_link"]),
    tool_offset: tuple[float, float, float] = JUGGLE_SPHERE_CENTER_OFFSET,
) -> torch.Tensor:
    """Return the pinch frame's world X and Z axes as a continuous 6-D orientation."""
    _, quaternion, _, _ = tool_state(env, tool_body_cfg, tool_offset)
    x_axis = quaternion.new_tensor((1.0, 0.0, 0.0)).expand(env.num_envs, -1)
    z_axis = quaternion.new_tensor((0.0, 0.0, 1.0)).expand(env.num_envs, -1)
    return torch.cat((math_utils.quat_apply(quaternion, x_axis), math_utils.quat_apply(quaternion, z_axis)), dim=1)


def hand_closure(
    env: ManagerBasedRLEnv,
    hand_cfg: SceneEntityCfg,
    open_positions: tuple[float, ...] = JUGGLE_SPHERE_OPEN_HAND_POSITION,
    contact_positions: tuple[float, ...] = JUGGLE_SPHERE_PRELOAD_HAND_POSITION,
) -> torch.Tensor:
    """Project the 16-joint posture from calibrated open (zero) to contact (one)."""
    robot: Articulation = env.scene[hand_cfg.name]
    positions = robot.data.joint_pos.torch[:, hand_cfg.joint_ids]
    opened = positions.new_tensor(open_positions)
    direction = positions.new_tensor(contact_positions) - opened
    closure = torch.sum((positions - opened) * direction, dim=1) / torch.sum(torch.square(direction)).clamp_min(1.0e-12)
    return closure.clamp(0.0, 1.0).unsqueeze(1)


def phase_one_hot(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the current physical phase as an eight-element one-hot vector."""
    phase = get_juggle_runtime_state(env).current_phases
    return functional.one_hot(phase, num_classes=len(JugglePhase)).to(dtype=torch.float32)


def sphere_support_from_fingertips(
    fingertip_positions_relative_to_ball: torch.Tensor,
    near_distance: float = 0.075,
    opposed_pair_dot: float = -0.20,
    strong_opposition_dot: float = -0.70,
    supporting_height: float = 0.01,
) -> torch.Tensor:
    """Estimate multi-finger sphere support from public fingertip geometry.

    Args:
        fingertip_positions_relative_to_ball: Fingertip displacements from the ball center [m],
            shape ``(N, F, 3)``.
        near_distance: Maximum center distance for a nearby fingertip [m].
        opposed_pair_dot: Maximum radial-vector dot product for a supporting pair.
        strong_opposition_dot: Dot product at which opposition alone proves support.
        supporting_height: Maximum fingertip height above the ball center for below/equator support [m].

    Returns:
        Boolean support estimate, shape ``(N,)``.
    """
    if fingertip_positions_relative_to_ball.ndim != 3 or fingertip_positions_relative_to_ball.shape[-1] != 3:
        raise ValueError("Fingertip displacements must have shape (N, F, 3).")
    if fingertip_positions_relative_to_ball.shape[1] < 2:
        raise ValueError("Sphere support requires at least two fingertips.")
    distances = torch.linalg.vector_norm(fingertip_positions_relative_to_ball, dim=2)
    near = distances < near_distance
    radial = fingertip_positions_relative_to_ball / distances.clamp_min(1.0e-8).unsqueeze(-1)
    pair_dots = torch.einsum("nfi,nji->nfj", radial, radial)
    finger_count = fingertip_positions_relative_to_ball.shape[1]
    distinct_pairs = torch.triu(
        torch.ones((finger_count, finger_count), dtype=torch.bool, device=pair_dots.device),
        diagonal=1,
    )
    valid_pairs = near.unsqueeze(2) & near.unsqueeze(1) & distinct_pairs.unsqueeze(0)
    best_pair_dot = torch.where(valid_pairs, pair_dots, torch.ones_like(pair_dots)).amin(dim=(1, 2))
    supported_from_below = (near & (fingertip_positions_relative_to_ball[..., 2] < supporting_height)).any(dim=1)
    return (
        (near.sum(dim=1) >= 2)
        & (best_pair_dot < opposed_pair_dot)
        & (supported_from_below | (best_pair_dot < strong_opposition_dot))
    )


def first_ascent_apex_crossing(
    first_ascent_active: torch.Tensor,
    seen_initial_ascent: torch.Tensor,
    vertical_velocity: torch.Tensor,
) -> torch.Tensor:
    """Detect the one admissible upward-to-downward crossing after reset.

    The active latch prevents a later bounce from being mistaken for the first
    toss apex after the original crossing has already been consumed.
    """
    return first_ascent_active & seen_initial_ascent & (vertical_velocity <= 0.0)


class JuggleProgressContext(ManagerTermBase):
    """Track the next local transition and a complete release-to-catch cycle."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._no_termination = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._local_goals = torch.tensor(
            [int(local_goal_for_phase(phase)) for phase in JugglePhase],
            dtype=torch.long,
            device=env.device,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear only edge-triggered outputs; the reset event owns episode state."""
        state = get_juggle_runtime_state(self._env)
        if env_ids is None:
            env_ids = slice(None)
        state.new_local_success[env_ids] = False
        state.new_cycle_success[env_ids] = False
        state.new_height_success[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        tool_body_cfg: SceneEntityCfg,
        fingertip_cfg: SceneEntityCfg,
        ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
        tool_offset: tuple[float, float, float] = JUGGLE_SPHERE_CENTER_OFFSET,
        release_separation_distance: float = 0.03,
        release_clear_steps: int = 2,
        apex_height_gain: float = 0.06,
        catch_approach_distance: float = 0.12,
        catch_distance: float = 0.055,
        contact_maximum_relative_speed: float = 0.45,
        stable_maximum_relative_speed: float = 0.20,
        stable_catch_steps: int = 15,
        apex_maximum_horizontal_displacement: float | None = None,
        track_supported_release_reference: bool = False,
        rearm_after_stable_catch: bool = False,
    ) -> torch.Tensor:
        """Update phase progress and return an all-false context mask."""
        if release_clear_steps < 1 or stable_catch_steps < 1:
            raise ValueError("Release-clear and stable-catch dwell lengths must be positive.")
        if apex_maximum_horizontal_displacement is not None and (
            not math.isfinite(apex_maximum_horizontal_displacement) or apex_maximum_horizontal_displacement <= 0.0
        ):
            raise ValueError("The apex horizontal-displacement limit must be finite and positive.")
        if not isinstance(track_supported_release_reference, bool):
            raise TypeError("track_supported_release_reference must be a Boolean value.")
        if not isinstance(rearm_after_stable_catch, bool):
            raise TypeError("rearm_after_stable_catch must be a Boolean value.")
        state = get_juggle_runtime_state(env)
        state.new_local_success.zero_()
        state.new_cycle_success.zero_()
        state.new_height_success.zero_()
        ball: RigidObject = env.scene[ball_cfg.name]
        tool_position, _, tool_linear_velocity, _ = tool_state(env, tool_body_cfg, tool_offset)
        ball_position = ball.data.root_pos_w.torch
        ball_velocity = ball.data.root_lin_vel_w.torch
        distance = torch.linalg.vector_norm(ball_position - tool_position, dim=1)
        relative_speed = torch.linalg.vector_norm(ball_velocity - tool_linear_velocity, dim=1)
        robot: Articulation = env.scene[fingertip_cfg.name]
        fingertip_relative_positions = robot.data.body_pos_w.torch[:, fingertip_cfg.body_ids] - ball_position.unsqueeze(
            1
        )
        sphere_supported = sphere_support_from_fingertips(fingertip_relative_positions)
        local_height = ball_position[:, 2] - env.scene.env_origins[:, 2]
        vertical_velocity = ball_velocity[:, 2]
        phase = state.current_phases
        next_phase = phase.clone()

        prethrow = (phase == int(JugglePhase.HELD_PRETHROW)) | (phase == int(JugglePhase.STABLE_CATCH))
        if track_supported_release_reference:
            # Preserve the last physically supported hand pose. Once support
            # is lost this reference stays fixed, so moving the hand after
            # release cannot make the one-metre target easier. The short-toss
            # task deliberately retains its original authored reset reference
            # for checkpoint-compatible success semantics.
            tool_height = tool_position[:, 2] - env.scene.env_origins[:, 2]
            tool_position_xy = tool_position[:, :2] - env.scene.env_origins[:, :2]
            # The calibrated cradle can be supported by proximal/palm geometry
            # before the public fingertip proxy becomes true. Distance remains
            # a reliable ownership signal up to the clear-release boundary.
            supported_prethrow = prethrow & (sphere_supported | (distance < release_separation_distance))
            state.release_heights.copy_(torch.where(supported_prethrow, tool_height, state.release_heights))
            state.release_origins_xy.copy_(
                torch.where(supported_prethrow.unsqueeze(1), tool_position_xy, state.release_origins_xy)
            )
        state.seen_initial_ascent |= prethrow & (vertical_velocity > 0.25)
        state.first_ascent_active &= ~(prethrow & ~state.seen_initial_ascent & (vertical_velocity < -0.10))
        predicted_apex = local_height + torch.square(torch.clamp_min(vertical_velocity, 0.0)) / (2.0 * 9.81)
        ball_position_xy = ball_position[:, :2] - env.scene.env_origins[:, :2]
        fresh_release = (
            prethrow
            & state.first_ascent_active
            & state.seen_initial_ascent
            & ~sphere_supported
            & (distance >= release_separation_distance)
            & (predicted_apex >= state.release_heights + apex_height_gain)
            & (env.episode_length_buf > 0)
        )
        next_phase[fresh_release] = int(JugglePhase.RELEASE)
        state.seen_release[fresh_release] = True
        state.seen_apex[fresh_release] = False
        state.release_clear_steps[fresh_release] = 1

        release_or_ascent = (phase == int(JugglePhase.RELEASE)) | (phase == int(JugglePhase.ASCENDING))
        remains_clear = ~sphere_supported & (distance >= 0.8 * release_separation_distance)
        state.release_clear_steps.copy_(
            torch.where(
                release_or_ascent & remains_clear,
                state.release_clear_steps + 1,
                torch.where(release_or_ascent, torch.zeros_like(state.release_clear_steps), state.release_clear_steps),
            )
        )
        first_apex_crossing = first_ascent_apex_crossing(
            state.first_ascent_active,
            state.seen_initial_ascent,
            vertical_velocity,
        )
        state.first_ascent_active &= ~first_apex_crossing
        if apex_maximum_horizontal_displacement is None:
            within_apex_corridor = torch.ones_like(first_apex_crossing)
        else:
            horizontal_displacement = torch.linalg.vector_norm(ball_position_xy - state.release_origins_xy, dim=1)
            within_apex_corridor = horizontal_displacement <= apex_maximum_horizontal_displacement
        valid_apex_clearance = (
            remains_clear
            & (state.release_clear_steps >= release_clear_steps)
            & (local_height >= state.release_heights + apex_height_gain)
            & within_apex_corridor
        )
        invalid_release = release_or_ascent & first_apex_crossing & ~valid_apex_clearance
        state.seen_release[invalid_release] = False
        release_ready_for_ascent = (
            (phase == int(JugglePhase.RELEASE))
            & state.seen_release
            & (state.release_clear_steps >= release_clear_steps)
            & (env.episode_length_buf > 0)
        )
        next_phase[release_ready_for_ascent] = int(JugglePhase.ASCENDING)

        # A valid physical separation may be recognized only one policy step
        # before the first apex.  Accept that same-step RELEASE -> ASCENDING ->
        # APEX path instead of consuming the one-shot crossing while the phase
        # label still says RELEASE.
        apex_source = (phase == int(JugglePhase.ASCENDING)) | release_ready_for_ascent
        fresh_apex = (
            apex_source & state.seen_release & first_apex_crossing & valid_apex_clearance & (vertical_velocity > -0.30)
        )
        next_phase[fresh_apex] = int(JugglePhase.APEX)
        state.seen_apex[fresh_apex] = True
        state.new_height_success.copy_(fresh_apex & ~state.height_success)
        state.height_success |= state.new_height_success

        mask = (phase == int(JugglePhase.APEX)) & (vertical_velocity < -0.12)
        next_phase[mask] = int(JugglePhase.DESCENDING)

        fresh_approach = (
            (phase == int(JugglePhase.DESCENDING))
            & (vertical_velocity < 0.0)
            & (distance < catch_approach_distance)
            & (env.episode_length_buf > 0)
        )
        next_phase[fresh_approach] = int(JugglePhase.CATCH_APPROACH)

        fresh_contact = (
            (phase == int(JugglePhase.CATCH_APPROACH))
            & (distance < catch_distance)
            & sphere_supported
            & (relative_speed < contact_maximum_relative_speed)
        )
        next_phase[fresh_contact] = int(JugglePhase.CATCH_CONTACT)

        retained_in_cradle = (
            (distance < catch_distance) & sphere_supported & (relative_speed < stable_maximum_relative_speed)
        )
        contact_or_stable = (phase == int(JugglePhase.CATCH_CONTACT)) | (phase == int(JugglePhase.STABLE_CATCH))
        state.stable_catch_steps.copy_(
            torch.where(
                contact_or_stable & retained_in_cradle,
                state.stable_catch_steps + 1,
                torch.zeros_like(state.stable_catch_steps),
            )
        )
        became_stable = (phase == int(JugglePhase.CATCH_CONTACT)) & (state.stable_catch_steps >= stable_catch_steps)
        next_phase[became_stable] = int(JugglePhase.STABLE_CATCH)

        direct_release_to_apex = fresh_apex & (phase == int(JugglePhase.RELEASE))
        state.visited_phase_bits |= torch.bitwise_left_shift(
            direct_release_to_apex.to(dtype=state.visited_phase_bits.dtype),
            int(JugglePhase.ASCENDING),
        )
        state.current_phases.copy_(next_phase)
        state.visited_phase_bits |= torch.bitwise_left_shift(torch.ones_like(next_phase), next_phase)
        phase_goal_ids = self._local_goals[state.start_phases]
        goal_ids = torch.where(state.local_goal_ids >= 0, state.local_goal_ids, phase_goal_ids)
        apex_goal = goal_ids == int(JuggleLocalGoal.FLIGHT_APEX)
        approach_goal = goal_ids == int(JuggleLocalGoal.CATCH_APPROACH)
        contact_goal = goal_ids == int(JuggleLocalGoal.CATCH_CONTACT)
        catch_goal = goal_ids == int(JuggleLocalGoal.STABLE_CATCH)
        # Each local target is a fresh event after reset, never the authored state itself.
        local_event = (
            (apex_goal & fresh_apex)
            | (approach_goal & fresh_approach)
            | (contact_goal & fresh_contact)
            | (catch_goal & became_stable)
        )
        state.new_local_success.copy_(local_event & ~state.local_success)
        state.local_success |= state.new_local_success

        required_bits = sum(1 << int(phase_id) for phase_id in JugglePhase if phase_id != JugglePhase.HELD_PRETHROW)
        visited_cycle = (state.visited_phase_bits & required_bits) == required_bits
        cycle_eligible = state.canonical_start
        completed_cycle = cycle_eligible & state.seen_release & state.seen_apex & visited_cycle & became_stable
        # Non-rearmed callers latch their first completion. Juggle training
        # retains episode-level success while emitting a fresh reward pulse for
        # every later physical catch-and-relaunch cycle.
        state.new_cycle_success.copy_(
            completed_cycle if rearm_after_stable_catch else completed_cycle & ~state.cycle_success
        )
        state.cycle_success |= state.new_cycle_success
        # Extras may outlive the termination pass and the runtime tensors are
        # cleared in-place during autoreset. Clone terminal values so logging
        # cannot observe the reset mutation instead of the completed episode.
        env.extras["successes"] = state.cycle_success.clone()
        env.extras["static_held_successes"] = (state.cycle_success & state.static_held_start).clone()
        if rearm_after_stable_catch:
            # Endless play keeps the physical catch untouched and only starts
            # a fresh logical cycle. The phase label matches a pre-throw start,
            # but the caught posture remains physical and must be covered by
            # multi-cycle training rather than being teleported to a reset row.
            rearm = became_stable
            held_phase = int(JugglePhase.HELD_PRETHROW)
            state.current_phases[rearm] = held_phase
            state.visited_phase_bits[rearm] = 1 << held_phase
            state.stable_catch_steps[rearm] = 0
            state.release_clear_steps[rearm] = 0
            state.first_ascent_active[rearm] = True
            state.seen_initial_ascent[rearm] = False
            state.seen_release[rearm] = False
            state.seen_apex[rearm] = False
            state.height_success[rearm] = False
        return self._no_termination


def local_transition_pulse(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a unit-integral pulse on the first phase-local success."""
    return get_juggle_runtime_state(env).new_local_success.float() / max(float(env.step_dt), 1.0e-6)


def apex_height_pulse(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a unit-integral pulse for an apex caused by the trained reset goal."""
    state = get_juggle_runtime_state(env)
    apex_goal = state.local_goal_ids <= int(JuggleLocalGoal.FLIGHT_APEX)
    success = state.new_height_success & (state.canonical_start | apex_goal)
    return success.float() / max(float(env.step_dt), 1.0e-6)


def non_height_local_transition_pulse(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a unit-integral pulse for a fresh local success other than apex height."""
    state = get_juggle_runtime_state(env)
    success = state.new_local_success & ~state.new_height_success
    return success.float() / max(float(env.step_dt), 1.0e-6)


def juggle_physical_progress_potential(
    env: ManagerBasedRLEnv,
    tool_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["palm_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    tool_offset: tuple[float, float, float] = JUGGLE_SPHERE_CENTER_OFFSET,
    target_height_gain: float = 1.0,
    apex_maximum_horizontal_displacement: float = 0.15,
    catch_distance_scale: float = 0.12,
    catch_relative_speed_scale: float = 0.45,
    canonical_launch_fraction: float = 0.5,
) -> torch.Tensor:
    """Return bounded, label-free launch/catch progress from live physics.

    Launch progress predicts the first ballistic apex from the ball's current
    height and velocity, then discounts trajectories whose predicted apex
    leaves the configured horizontal corridor. Catch progress rewards reducing
    both ball/tool distance and relative speed. Non-canonical reset episodes
    use the complete ``[0, 1]`` range until their local goal succeeds, then
    continue with the same launch/catch staging as a full-cycle episode. The
    first half is reserved for launch and the second for catch, making the
    stage boundary continuous at a qualified apex.
    """
    positive_parameters = (
        ("target_height_gain", target_height_gain),
        ("apex_maximum_horizontal_displacement", apex_maximum_horizontal_displacement),
        ("catch_distance_scale", catch_distance_scale),
        ("catch_relative_speed_scale", catch_relative_speed_scale),
    )
    for name, value in positive_parameters:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if (
        isinstance(canonical_launch_fraction, bool)
        or not isinstance(canonical_launch_fraction, (int, float))
        or not math.isfinite(canonical_launch_fraction)
        or not 0.0 < canonical_launch_fraction < 1.0
    ):
        raise ValueError("canonical_launch_fraction must lie strictly inside (0, 1).")

    state = get_juggle_runtime_state(env)
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_position = ball.data.root_pos_w.torch
    ball_velocity = ball.data.root_lin_vel_w.torch
    local_position = ball_position - env.scene.env_origins
    upward_velocity = ball_velocity[:, 2].clamp_min(0.0)
    time_to_apex = upward_velocity / -float(GRAVITY_Z)
    predicted_height_gain = (
        local_position[:, 2] - state.release_heights + torch.square(upward_velocity) / (-2.0 * float(GRAVITY_Z))
    )
    height_quality = (predicted_height_gain / float(target_height_gain)).clamp(0.0, 1.0)
    predicted_apex_xy = local_position[:, :2] + ball_velocity[:, :2] * time_to_apex.unsqueeze(1)
    predicted_apex_displacement = torch.linalg.vector_norm(predicted_apex_xy - state.release_origins_xy, dim=1)
    corridor_quality = (1.0 - predicted_apex_displacement / float(apex_maximum_horizontal_displacement)).clamp(0.0, 1.0)
    launch_potential = height_quality * corridor_quality

    tool_position, _, tool_linear_velocity, _ = tool_state(env, tool_body_cfg, tool_offset)
    distance = torch.linalg.vector_norm(ball_position - tool_position, dim=1)
    relative_speed = torch.linalg.vector_norm(ball_velocity - tool_linear_velocity, dim=1)
    distance_ratio = distance / float(catch_distance_scale)
    speed_ratio = relative_speed / float(catch_relative_speed_scale)
    catch_potential = torch.reciprocal(1.0 + torch.square(distance_ratio)) * torch.reciprocal(
        1.0 + torch.square(speed_ratio)
    )

    legacy_goal_lookup = torch.tensor(
        [int(local_goal_for_phase(phase)) for phase in JugglePhase],
        dtype=torch.long,
        device=state.local_goal_ids.device,
    )
    goal_ids = torch.where(state.local_goal_ids >= 0, state.local_goal_ids, legacy_goal_lookup[state.start_phases])
    apex_goal = goal_ids == int(JuggleLocalGoal.FLIGHT_APEX)
    continuous_cycle = state.canonical_start | state.local_success
    continuous_catch_stage = continuous_cycle & state.height_success
    continuous_launch_stage = continuous_cycle & ~state.height_success
    potential = torch.where(apex_goal, launch_potential, catch_potential)
    potential = torch.where(
        continuous_launch_stage,
        float(canonical_launch_fraction) * launch_potential,
        potential,
    )
    potential = torch.where(
        continuous_catch_stage,
        float(canonical_launch_fraction) + (1.0 - float(canonical_launch_fraction)) * catch_potential,
        potential,
    )
    return torch.nan_to_num(potential, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


class JugglePhysicalProgressReward(ManagerTermBase):
    """Return reset-aware discounted differences of physical juggling progress."""

    _POTENTIAL_PARAMETER_NAMES = (
        "tool_body_cfg",
        "ball_cfg",
        "tool_offset",
        "target_height_gain",
        "apex_maximum_horizontal_displacement",
        "catch_distance_scale",
        "catch_relative_speed_scale",
        "canonical_launch_fraction",
    )

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._baseline_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._skip_next = torch.ones_like(self._baseline_valid)
        self._previous_phase = get_juggle_runtime_state(env).current_phases.clone()
        self._potential_parameters = {
            name: cfg.params[name] for name in self._POTENTIAL_PARAMETER_NAMES if name in cfg.params
        }

    def _potential(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        return juggle_physical_progress_potential(env, **self._potential_parameters)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Latch the post-event physical reset as a zero-credit baseline."""
        selected = slice(None) if env_ids is None else env_ids
        current = self._potential(self._env)
        self._previous[selected] = current[selected]
        self._baseline_valid[selected] = True
        self._previous_phase[selected] = get_juggle_runtime_state(self._env).current_phases[selected]
        # The first manager evaluation may include simulator settling from the
        # authored row. Re-baseline it instead of crediting that passive motion.
        self._skip_next[selected] = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        gamma: float = 1.0,
        drop_termination_name: str = "ball_out_of_workspace",
        tool_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["palm_link"]),
        ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
        tool_offset: tuple[float, float, float] = JUGGLE_SPHERE_CENTER_OFFSET,
        target_height_gain: float = 1.0,
        apex_maximum_horizontal_displacement: float = 0.15,
        catch_distance_scale: float = 0.12,
        catch_relative_speed_scale: float = 0.45,
        canonical_launch_fraction: float = 0.5,
    ) -> torch.Tensor:
        """Return ``(gamma * Phi(next) - Phi(previous)) / step_dt``.

        A dropped-ball terminal has zero potential on the same step, repaying
        progress already collected. RewardManager multiplies the returned rate
        by ``step_dt``, leaving an integrated shaping delta bounded by one.
        """
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(gamma):
            raise ValueError("gamma must be finite.")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must lie inside (0, 1].")
        if not isinstance(drop_termination_name, str) or not drop_termination_name:
            raise ValueError("drop_termination_name must not be empty.")
        step_dt = float(env.step_dt)
        if not math.isfinite(step_dt) or step_dt <= 0.0:
            raise ValueError("The environment step_dt must be finite and positive.")

        current = juggle_physical_progress_potential(
            env,
            tool_body_cfg=tool_body_cfg,
            ball_cfg=ball_cfg,
            tool_offset=tool_offset,
            target_height_gain=target_height_gain,
            apex_maximum_horizontal_displacement=apex_maximum_horizontal_displacement,
            catch_distance_scale=catch_distance_scale,
            catch_relative_speed_scale=catch_relative_speed_scale,
            canonical_launch_fraction=canonical_launch_fraction,
        )
        try:
            dropped = env.termination_manager.get_term(drop_termination_name).to(dtype=torch.bool)
        except KeyError as error:
            raise ValueError(f"Unknown termination term: {drop_termination_name!r}.") from error
        terminal_potential = torch.where(dropped, torch.zeros_like(current), current)
        missing_baseline = ~self._baseline_valid
        previous = torch.where(missing_baseline, current, self._previous)
        delta = float(gamma) * terminal_potential - previous
        state = get_juggle_runtime_state(env)
        held_phase = state.current_phases == int(JugglePhase.HELD_PRETHROW)
        rearmed_cycle = held_phase & (self._previous_phase != int(JugglePhase.HELD_PRETHROW))
        completed_local_stage = state.new_local_success & ~state.canonical_start
        zero_credit_rebaseline = (
            self._skip_next | missing_baseline | state.new_cycle_success | rearmed_cycle | completed_local_stage
        ) & ~dropped
        delta = torch.where(zero_credit_rebaseline, torch.zeros_like(delta), delta)
        self._previous.copy_(terminal_potential)
        self._previous_phase.copy_(state.current_phases)
        self._baseline_valid.fill_(True)
        self._skip_next.zero_()
        return torch.nan_to_num(delta / step_dt, nan=0.0, posinf=1.0 / step_dt, neginf=-1.0 / step_dt).clamp_(
            -1.0 / step_dt,
            1.0 / step_dt,
        )


def ball_out_of_workspace_pulse(
    env: ManagerBasedRLEnv,
    termination_term_name: str = "ball_out_of_workspace",
) -> torch.Tensor:
    """Return a unit-integral pulse for the configured ball-workspace termination."""
    if not termination_term_name:
        raise ValueError("termination_term_name must not be empty.")
    try:
        out_of_workspace = env.termination_manager.get_term(termination_term_name)
    except KeyError as error:
        raise ValueError(f"Unknown termination term: {termination_term_name!r}.") from error
    return out_of_workspace.float() / max(float(env.step_dt), 1.0e-6)


def full_cycle_pulse(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a unit-integral pulse on a complete toss-and-catch cycle."""
    return get_juggle_runtime_state(env).new_cycle_success.float() / max(float(env.step_dt), 1.0e-6)


def cycle_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate when one complete toss-and-catch cycle succeeds."""
    return get_juggle_runtime_state(env).cycle_success


def noncanonical_local_goal_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End a phase-local episode on its fresh physical success event."""
    state = get_juggle_runtime_state(env)
    return state.local_success & ~state.canonical_start


def ball_out_of_workspace(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    workspace_lower: tuple[float, float, float] = (0.20, -0.35, 0.08),
    workspace_upper: tuple[float, float, float] = (0.80, 0.35, 0.90),
) -> torch.Tensor:
    """Terminate a dropped or escaped ball."""
    ball: RigidObject = env.scene[ball_cfg.name]
    position = ball.data.root_pos_w.torch - env.scene.env_origins
    lower = position.new_tensor(workspace_lower)
    upper = position.new_tensor(workspace_upper)
    return ((position < lower) | (position > upper)).any(dim=1)


def nonfinite_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Terminate non-finite robot or ball states before they poison a rollout."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    robot_finite = torch.isfinite(robot.data.joint_pos.torch[:, robot_cfg.joint_ids]).all(dim=1)
    robot_finite &= torch.isfinite(robot.data.joint_vel.torch[:, robot_cfg.joint_ids]).all(dim=1)
    ball_finite = torch.isfinite(ball.data.root_pose_w.torch).all(dim=1)
    ball_finite &= torch.isfinite(ball.data.root_vel_w.torch).all(dim=1)
    return ~(robot_finite & ball_finite)
