# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analytic phase resets for one-ball KUKA-Allegro juggling."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import EventTermCfg, ManagerTermBase

from isaaclab_tasks.contrib.stack.mdp.kuka_allegro_reset import (
    KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES,
    kuka_allegro_tool_pose,
)
from isaaclab_tasks.utils.reset_sampling import ResetStateCatalog

from .runtime import create_juggle_runtime_state, initialize_juggle_episode_state

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


BALL_RADIUS = 0.04
"""Radius of the juggling ball [m]."""

BALL_MASS = 0.075
"""Mass of the juggling ball [kg]."""

GRAVITY_Z = -9.81
"""World-frame vertical gravitational acceleration [m/s^2]."""

JUGGLE_SPHERE_CENTER_OFFSET = (0.02790133, -0.03190392, 0.03965311)
"""Settled sphere-cradle center in the Allegro palm frame [m]."""

JUGGLE_SPHERE_PRELOAD_HAND_POSITION = (
    -0.0619058833,
    0.9506817460,
    0.3618645370,
    1.4499270916,
    -0.1500000060,
    1.2170184851,
    1.4057863951,
    0.7829402089,
    -0.1169813424,
    0.1852552444,
    1.3507008553,
    0.4363770485,
    1.2318177223,
    0.2235770822,
    0.3533166349,
    0.7815441489,
)
"""Palm/side cradle that retained the sphere for four Newton seconds [rad].

The posture is intentionally described as a cradle rather than a fingertip pinch: live Newton
validation showed that support comes from the palm and proximal/distal finger-link geometry.
"""

JUGGLE_SPHERE_FLIGHT_GATE_HAND_POSITION = (
    0.02265799,
    0.44551727,
    0.00733247,
    0.88615793,
    0.06259898,
    0.69524920,
    0.88273019,
    0.36905348,
    0.05468165,
    -0.10329334,
    0.79448146,
    0.07339861,
    0.71987367,
    -0.06808281,
    0.03197674,
    0.36520752,
)
"""Actuator-reachable, collision-clear vertical-flight gate [rad].

The posture is the measured state reached after 20 policy steps toward a fully extended hand,
not the unreachable command endpoint. Frozen-state Newton sweeps verified clean passage and an
action-discriminative close across tool heights, approach speeds, and lateral offsets.
"""

JUGGLE_SPHERE_OPEN_HAND_POSITION = JUGGLE_SPHERE_FLIGHT_GATE_HAND_POSITION
"""Release direction used by the reset-preload handoff [rad]."""

JUGGLE_SPHERE_CONTACT_HAND_POSITION = JUGGLE_SPHERE_FLIGHT_GATE_HAND_POSITION
"""Open gate used by catch-approach/contact rows [rad]."""

KUKA_ARM_JOINT_NAMES: tuple[str, ...] = tuple(f"iiwa7_joint_{joint_id}" for joint_id in range(1, 8))
"""Canonical seven-joint KUKA arm order."""

KUKA_ALLEGRO_JUGGLE_ARM_WORKSPACE_LOWER: tuple[float, ...] = (-2.7, -1.9, -2.7, 0.3, -2.7, -1.9, -3.05)
"""Collision-screened lower arm boundary [rad]."""

KUKA_ALLEGRO_JUGGLE_ARM_WORKSPACE_UPPER: tuple[float, ...] = (2.7, 1.9, 2.7, 2.0694, 2.7, 1.9, 3.05)
"""Collision-screened upper arm boundary [rad]."""


class JugglePhase(IntEnum):
    """Ordered physical phases in one complete vertical toss-and-catch cycle."""

    HELD_PRETHROW = 0
    RELEASE = 1
    ASCENDING = 2
    APEX = 3
    DESCENDING = 4
    CATCH_APPROACH = 5
    CATCH_CONTACT = 6
    STABLE_CATCH = 7


class JuggleLocalGoal(IntEnum):
    """Fresh event required to credit a phase-local episode."""

    FLIGHT_APEX = 0
    CATCH_APPROACH = 1
    CATCH_CONTACT = 2
    STABLE_CATCH = 3


_LOCAL_GOAL_BY_PHASE = (
    JuggleLocalGoal.FLIGHT_APEX,
    JuggleLocalGoal.FLIGHT_APEX,
    JuggleLocalGoal.FLIGHT_APEX,
    JuggleLocalGoal.CATCH_APPROACH,
    JuggleLocalGoal.CATCH_APPROACH,
    JuggleLocalGoal.CATCH_CONTACT,
    JuggleLocalGoal.STABLE_CATCH,
    JuggleLocalGoal.FLIGHT_APEX,
)


def local_goal_for_phase(phase: JugglePhase) -> JuggleLocalGoal:
    """Return the fresh physical event trained by a reset phase."""
    return _LOCAL_GOAL_BY_PHASE[int(phase)]


# Each pose places the sphere-cradle center at approximately (0.50, 0.00, z)
# with the palm opening aligned to world +Z.
_ARM_ANCHOR_HEIGHTS = (0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.52, 0.56)
_ARM_ANCHOR_POSITIONS = (
    (2.15492783, 1.25944785, -2.31044499, 1.44886897, -2.69980076, 1.62759907, -3.00147131),
    (2.11232341, 1.21832065, -2.22995099, 1.47334334, -2.69731542, 1.62863823, -2.92693183),
    (2.09426120, 1.15379100, -2.19415376, 1.48760476, -2.65542972, 1.64765074, -2.88137803),
    (2.07768221, 1.09248307, -2.15824116, 1.49153356, -2.61120805, 1.67076378, -2.83233563),
    (2.06184456, 1.03339312, -2.12273932, 1.48638531, -2.56518443, 1.69525967, -2.77988880),
    (2.04852612, 0.97780378, -2.08841950, 1.47013934, -2.51878143, 1.72291170, -2.72244994),
    (2.03702554, 0.92688426, -2.05517126, 1.44490997, -2.47107735, 1.75010598, -2.66275505),
    (2.02956190, 0.88013022, -2.02452271, 1.40824169, -2.42430160, 1.77893998, -2.59641719),
)


def ballistic_state(
    release_position: torch.Tensor,
    release_velocity: torch.Tensor,
    flight_time: torch.Tensor,
    gravity_z: float = GRAVITY_Z,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate free-flight positions and velocities analytically.

    Args:
        release_position: Release positions [m], shape ``(..., 3)``.
        release_velocity: Release linear velocities [m/s], shape ``(..., 3)``.
        flight_time: Time after release [s], broadcastable to ``release_position[..., 0]``.
        gravity_z: Vertical gravitational acceleration [m/s^2].

    Returns:
        Ball positions [m] and velocities [m/s], each with shape ``(..., 3)``.
    """
    if release_position.shape != release_velocity.shape or release_position.shape[-1] != 3:
        raise ValueError("Release position and velocity must have matching (..., 3) shapes.")
    time = torch.as_tensor(flight_time, dtype=release_position.dtype, device=release_position.device)
    while time.ndim < release_position.ndim:
        time = time.unsqueeze(-1)
    acceleration = torch.zeros_like(release_position)
    acceleration[..., 2] = gravity_z
    position = release_position + release_velocity * time + 0.5 * acceleration * torch.square(time)
    velocity = release_velocity + acceleration * time
    return position, velocity


def kuka_allegro_tool_point_velocity(
    arm_position: torch.Tensor,
    arm_velocity: torch.Tensor,
    tool_offset: tuple[float, float, float] = JUGGLE_SPHERE_CENTER_OFFSET,
    finite_difference_dt: float = 1.0e-4,
) -> torch.Tensor:
    """Compute the FK-consistent palm-fixed tool-point velocity [m/s].

    A centered directional derivative is used deliberately: the reset catalog
    is generated offline once, while this definition guarantees that the ball
    velocity and the exact kinematics used to place it cannot silently diverge.

    Args:
        arm_position: KUKA arm joint positions [rad], shape ``(..., 7)``.
        arm_velocity: KUKA arm joint velocities [rad/s], shape ``(..., 7)``.
        tool_offset: Tool point in the palm frame [m].
        finite_difference_dt: Centered differentiation interval [s].

    Returns:
        Tool-point linear velocity [m/s], shape ``(..., 3)``.
    """
    if arm_position.shape != arm_velocity.shape or arm_position.shape[-1] != 7:
        raise ValueError("Arm position and velocity must have matching (..., 7) shapes.")
    if finite_difference_dt <= 0.0:
        raise ValueError("finite_difference_dt must be positive.")
    half_step = 0.5 * finite_difference_dt
    position_after, _ = kuka_allegro_tool_pose(arm_position + half_step * arm_velocity, tool_offset)
    position_before, _ = kuka_allegro_tool_pose(arm_position - half_step * arm_velocity, tool_offset)
    return (position_after - position_before) / finite_difference_dt


def _radical_inverse(index: int, base: int) -> float:
    """Return one deterministic low-discrepancy coordinate in ``[0, 1)``."""
    value = 0.0
    inverse_base = 1.0 / base
    factor = inverse_base
    while index:
        index, digit = divmod(index, base)
        value += digit * factor
        factor *= inverse_base
    return value


class JuggleResetStateSource:
    """Immutable, balanced reset rows for one vertical toss/catch cycle.

    The source keeps physical rows separate from the eight monitored phase items. Free-flight rows are
    generated from one release state with the ballistic equations; no state is interpolated across release
    or contact discontinuities.
    """

    def __init__(
        self,
        rows_per_phase: int = 64,
        device: str | torch.device = "cpu",
    ) -> None:
        """Build a deterministic reset source.

        Args:
            rows_per_phase: Number of physical variations per phase.
            device: Torch device holding the reset tensors.
        """
        if isinstance(rows_per_phase, bool) or not isinstance(rows_per_phase, int) or rows_per_phase < 8:
            raise ValueError("rows_per_phase must be an integer of at least eight.")
        self.rows_per_phase = rows_per_phase
        self.device = torch.device(device)
        self.phase_count = len(JugglePhase)
        self.row_count = rows_per_phase * self.phase_count

        rows = [self._make_row(phase, variant) for phase in JugglePhase for variant in range(rows_per_phase)]
        self.arm_positions = torch.tensor([row[0] for row in rows], dtype=torch.float32, device=self.device)
        self.arm_velocities = torch.tensor([row[1] for row in rows], dtype=torch.float32, device=self.device)
        self.hand_positions = torch.tensor([row[2] for row in rows], dtype=torch.float32, device=self.device)
        self.hand_velocities = torch.zeros_like(self.hand_positions)
        self.ball_positions = torch.tensor([row[3] for row in rows], dtype=torch.float32, device=self.device)
        self.ball_quaternions = torch.zeros((self.row_count, 4), dtype=torch.float32, device=self.device)
        self.ball_quaternions[:, 3] = 1.0
        self.ball_velocities = torch.zeros((self.row_count, 6), dtype=torch.float32, device=self.device)
        self.ball_velocities[:, :3] = torch.tensor([row[4] for row in rows], dtype=torch.float32, device=self.device)
        self.phase_ids = torch.tensor([int(row[5]) for row in rows], dtype=torch.long, device=self.device)
        self.release_positions = torch.tensor([row[6] for row in rows], dtype=torch.float32, device=self.device)
        self.release_velocities = torch.tensor([row[7] for row in rows], dtype=torch.float32, device=self.device)
        self.flight_times = torch.tensor([row[8] for row in rows], dtype=torch.float32, device=self.device)
        self.ballistic_rows = torch.tensor([row[9] for row in rows], dtype=torch.bool, device=self.device)
        self.launch_reference_heights = torch.tensor([row[10] for row in rows], dtype=torch.float32, device=self.device)
        self.static_held_rows = torch.tensor([row[11] for row in rows], dtype=torch.bool, device=self.device)

        tensors = (
            self.arm_positions,
            self.arm_velocities,
            self.hand_positions,
            self.hand_velocities,
            self.ball_positions,
            self.ball_quaternions,
            self.ball_velocities,
            self.release_positions,
            self.release_velocities,
            self.flight_times,
            self.launch_reference_heights,
        )
        if not all(bool(torch.isfinite(values).all()) for values in tensors):
            raise RuntimeError("Juggle reset source contains a non-finite physical value.")
        self.catalog = ResetStateCatalog(
            row_count=self.row_count,
            metadata={
                "phase": self.phase_ids,
                "ballistic": self.ballistic_rows,
                "flight_time": self.flight_times,
                "static_held": self.static_held_rows,
            },
            row_to_item=self.phase_ids,
        )
        self.phase_rows = tuple(torch.where(self.phase_ids == int(phase))[0] for phase in JugglePhase)

    def _make_row(
        self,
        phase: JugglePhase,
        variant: int,
    ) -> tuple[
        tuple[float, ...],
        tuple[float, ...],
        tuple[float, ...],
        tuple[float, float, float],
        tuple[float, float, float],
        JugglePhase,
        tuple[float, float, float],
        tuple[float, float, float],
        float,
        bool,
        float,
        bool,
    ]:
        """Author one phase-local physical row."""
        sample_id = variant + 1
        height_coordinate = _radical_inverse(sample_id, 2)
        speed_coordinate = _radical_inverse(sample_id, 3)
        time_coordinate = _radical_inverse(sample_id, 5)
        catch_coordinate = _radical_inverse(sample_id, 7)
        lateral_coordinate = 2.0 * _radical_inverse(sample_id, 11) - 1.0

        release_height = 0.36 + 0.06 * height_coordinate
        release_speed = 1.10 + 0.30 * speed_coordinate
        lateral_angle = 2.0 * math.pi * _radical_inverse(sample_id, 13)
        lateral_speed = 0.19 + 0.15 * _radical_inverse(sample_id, 17)
        lateral_velocity_y = 0.12 * math.sin(lateral_angle)
        lateral_velocity_x = math.sqrt(max(0.0, lateral_speed**2 - lateral_velocity_y**2))
        if math.cos(lateral_angle) < 0.0:
            lateral_velocity_x = -lateral_velocity_x
        release_position = (
            0.50 + 0.004 * math.cos(lateral_angle),
            0.004 * math.sin(lateral_angle),
            release_height,
        )
        release_velocity = (
            lateral_velocity_x,
            lateral_velocity_y,
            release_speed,
        )
        flight_time = 0.0
        ballistic = False
        static_held = False
        launch_reference_height = release_height

        contact_hand = JUGGLE_SPHERE_CONTACT_HAND_POSITION
        preload_hand = JUGGLE_SPHERE_PRELOAD_HAND_POSITION
        flight_hand = JUGGLE_SPHERE_FLIGHT_GATE_HAND_POSITION
        catch_arm_height = 0.30 + 0.12 * catch_coordinate
        arm_velocity = (0.0,) * 7

        if phase is JugglePhase.HELD_PRETHROW:
            held_height = 0.30 + 0.12 * height_coordinate
            # Half of canonical starts are truly at rest; the other half are
            # moving bootstrap states that teach the opening/toss transition.
            static_held = variant < self.rows_per_phase // 2
            upward_speed = 0.0 if static_held else 0.80 + 0.45 * speed_coordinate
            arm_position, arm_velocity, ball_position, ball_velocity = _moving_arm_state(
                held_height,
                upward_speed,
            )
            hand_position = preload_hand
            launch_reference_height = ball_position[2]
        elif phase is JugglePhase.RELEASE:
            gate_gap = 0.07 + 0.03 * time_coordinate
            tool_height = min(_ARM_ANCHOR_HEIGHTS[-1], max(_ARM_ANCHOR_HEIGHTS[0], release_position[2] - gate_gap))
            arm_position = _interpolate_arm_anchor(tool_height)
            ball_position = release_position
            ball_velocity = release_velocity
            hand_position = flight_hand
        elif phase in (JugglePhase.ASCENDING, JugglePhase.APEX, JugglePhase.DESCENDING):
            apex_time = release_velocity[2] / -GRAVITY_Z
            if phase is JugglePhase.ASCENDING:
                flight_time = 0.02 + (0.68 * apex_time - 0.02) * time_coordinate
            elif phase is JugglePhase.APEX:
                flight_time = apex_time + 0.012 * (2.0 * time_coordinate - 1.0)
            else:
                flight_time = apex_time + 0.018 + 0.055 * time_coordinate
            position, velocity = ballistic_state(
                torch.tensor(release_position, dtype=torch.float64),
                torch.tensor(release_velocity, dtype=torch.float64),
                torch.tensor(flight_time, dtype=torch.float64),
            )
            ball_position = tuple(float(value) for value in position)
            ball_velocity = tuple(float(value) for value in velocity)
            # Correlate the interception miss with time-to-catch: early flight
            # may be farther from the cup, while descending rows stay outside
            # the 12 cm approach event at reset.
            if phase is JugglePhase.DESCENDING:
                gate_gap = 0.135 + 0.025 * catch_coordinate
            else:
                gate_gap = 0.10 + 0.025 * catch_coordinate
            tool_height = min(_ARM_ANCHOR_HEIGHTS[-1], max(_ARM_ANCHOR_HEIGHTS[0], ball_position[2] - gate_gap))
            arm_position = _interpolate_arm_anchor(tool_height)
            hand_position = flight_hand
            ballistic = True
        elif phase in (JugglePhase.CATCH_APPROACH, JugglePhase.CATCH_CONTACT):
            arm_position, arm_velocity, ball_position, ball_velocity = _moving_arm_state(
                catch_arm_height,
                0.0,
            )
            if phase is JugglePhase.CATCH_APPROACH:
                z_offset = 0.080 + 0.040 * height_coordinate
                # The reachable palm-up gate is robust to five millimetres of
                # lateral miss across the complete approach-speed/height grid.
                # Larger synthetic offsets fall outside that verified capture
                # basin and turn close-command rows into unavoidable misses.
                lateral_limit = 0.005
                x_offset = lateral_limit * lateral_coordinate
                y_offset = lateral_limit * (2.0 * _radical_inverse(sample_id, 13) - 1.0)
                horizontal_speed = -0.04 * lateral_coordinate
                downward_speed = -0.15 - 0.30 * speed_coordinate
            else:
                z_offset = 0.040 + 0.035 * height_coordinate
                lateral_limit = 0.004
                x_offset = lateral_limit * lateral_coordinate
                y_offset = lateral_limit * (2.0 * _radical_inverse(sample_id, 13) - 1.0)
                horizontal_speed = -0.03 * lateral_coordinate
                downward_speed = -0.10 - 0.25 * speed_coordinate
            ball_position = (
                ball_position[0] + x_offset,
                ball_position[1] + y_offset,
                ball_position[2] + z_offset,
            )
            ball_velocity = (horizontal_speed, 0.0, downward_speed)
            hand_position = contact_hand
        else:
            stable_height = 0.30 + 0.12 * height_coordinate
            arm_position, arm_velocity, ball_position, ball_velocity = _moving_arm_state(
                stable_height,
                0.0,
            )
            hand_position = preload_hand
            launch_reference_height = ball_position[2]

        return (
            tuple(arm_position),
            tuple(arm_velocity),
            tuple(hand_position),
            tuple(ball_position),
            tuple(ball_velocity),
            phase,
            release_position,
            release_velocity,
            flight_time,
            ballistic,
            launch_reference_height,
            static_held,
        )


class JuggleResetEvent(ManagerTermBase):
    """Apply rows selected by the juggling reset curriculum."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        rows_per_phase = int(cfg.params.get("rows_per_phase", 64))
        self.source = JuggleResetStateSource(rows_per_phase=rows_per_phase, device=env.device)
        self.catalog = self.source.catalog
        self.row_count = self.source.row_count
        self.phase_ids = self.source.phase_ids
        self._robot: Articulation = env.scene["robot"]
        self._ball: RigidObject = env.scene["ball"]
        self._arm_joint_ids = self._robot.find_joints(KUKA_ARM_JOINT_NAMES, preserve_order=True)[0]
        self._hand_joint_ids = self._robot.find_joints(KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES, preserve_order=True)[0]
        if len(self._arm_joint_ids) != 7 or len(self._hand_joint_ids) != 16:
            raise RuntimeError("Juggle reset requires the complete 7+16 KUKA-Allegro articulation.")
        self.runtime = create_juggle_runtime_state(env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        rows_per_phase: int = 64,
        fixed_phase: int | None = None,
        static_held_only: bool = False,
    ) -> None:
        """Write selected robot and ball states to simulation."""
        del rows_per_phase
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
        if ids.numel() == 0:
            return
        if fixed_phase is not None:
            if not 0 <= fixed_phase < len(JugglePhase):
                raise ValueError("fixed_phase is outside the juggling phase range.")
            phase_rows = self.source.phase_rows[fixed_phase]
            if static_held_only:
                if fixed_phase != int(JugglePhase.HELD_PRETHROW):
                    raise ValueError("static_held_only requires the held-prethrow fixed phase.")
                phase_rows = phase_rows[self.source.static_held_rows[phase_rows]]
            row_ids = phase_rows[torch.randint(phase_rows.numel(), (ids.numel(),), device=env.device)]
            self.runtime.row_ids[ids] = row_ids
        else:
            row_ids = self.runtime.row_ids[ids]
            invalid = (row_ids < 0) | (row_ids >= self.row_count)
            fallback_rows = torch.randint(self.row_count, (ids.numel(),), device=env.device)
            row_ids = torch.where(invalid, fallback_rows, row_ids)
            self.runtime.row_ids[ids] = row_ids

        joint_positions = self._robot.data.default_joint_pos.torch[ids].clone()
        joint_velocities = torch.zeros_like(joint_positions)
        joint_positions[:, self._arm_joint_ids] = self.source.arm_positions[row_ids]
        joint_positions[:, self._hand_joint_ids] = self.source.hand_positions[row_ids]
        joint_velocities[:, self._arm_joint_ids] = self.source.arm_velocities[row_ids]
        joint_velocities[:, self._hand_joint_ids] = self.source.hand_velocities[row_ids]
        self._robot.set_joint_position_target_index(target=joint_positions, env_ids=ids)
        # qdot belongs only to the instantaneous physical reset. A persistent
        # nonzero actuator velocity target would invisibly drive the toss after reset.
        self._robot.set_joint_velocity_target_index(target=torch.zeros_like(joint_velocities), env_ids=ids)
        self._robot.write_joint_position_to_sim_index(position=joint_positions, env_ids=ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=joint_velocities, env_ids=ids)

        root_pose = self._ball.data.default_root_pose.torch[ids].clone()
        root_pose[:, :3] = self.source.ball_positions[row_ids] + env.scene.env_origins[ids]
        root_pose[:, 3:7] = self.source.ball_quaternions[row_ids]
        root_velocity = self.source.ball_velocities[row_ids]
        self._ball.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=ids)
        self._ball.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=ids)

        phases = self.source.phase_ids[row_ids]
        initialize_juggle_episode_state(
            self.runtime,
            ids,
            phases,
            self.source.launch_reference_heights[row_ids],
            self.source.static_held_rows[row_ids],
        )


def _moving_arm_state(
    height: float,
    upward_speed: float,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Return an attached arm/ball state with FK-consistent velocity."""
    arm_position, unit_arm_velocity = _interpolate_arm_anchor_and_derivative(height)
    position_tensor = torch.tensor(arm_position, dtype=torch.float64)
    unit_velocity_tensor = torch.tensor(unit_arm_velocity, dtype=torch.float64)
    unit_tool_velocity = kuka_allegro_tool_point_velocity(position_tensor, unit_velocity_tensor)
    if abs(float(unit_tool_velocity[2])) < 0.5:
        raise RuntimeError("Calibrated upward arm velocity does not move the sphere center upward.")
    arm_velocity_tensor = unit_velocity_tensor * (upward_speed / float(unit_tool_velocity[2]))
    ball_position, _ = kuka_allegro_tool_pose(position_tensor, JUGGLE_SPHERE_CENTER_OFFSET)
    ball_velocity = kuka_allegro_tool_point_velocity(position_tensor, arm_velocity_tensor)
    return (
        tuple(float(value) for value in position_tensor),
        tuple(float(value) for value in arm_velocity_tensor),
        tuple(float(value) for value in ball_position),
        tuple(float(value) for value in ball_velocity),
    )


def _interpolate_arm_anchor(height: float) -> tuple[float, ...]:
    """Interpolate calibrated arm positions [rad]."""
    return _interpolate_arm_anchor_and_derivative(height)[0]


def _interpolate_arm_anchor_and_derivative(height: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Interpolate palm-up arm positions and their height derivative [rad, rad/m]."""
    if not _ARM_ANCHOR_HEIGHTS[0] <= height <= _ARM_ANCHOR_HEIGHTS[-1]:
        raise ValueError("Requested juggling tool height is outside the calibrated anchors.")
    upper = next(
        (index for index, anchor_height in enumerate(_ARM_ANCHOR_HEIGHTS) if anchor_height >= height),
        len(_ARM_ANCHOR_HEIGHTS) - 1,
    )
    lower = max(0, upper - 1)
    if lower == upper:
        upper = min(len(_ARM_ANCHOR_HEIGHTS) - 1, lower + 1)
    lower_height = _ARM_ANCHOR_HEIGHTS[lower]
    upper_height = _ARM_ANCHOR_HEIGHTS[upper]
    span = upper_height - lower_height
    alpha = (height - lower_height) / span
    lower_pose = _ARM_ANCHOR_POSITIONS[lower]
    upper_pose = _ARM_ANCHOR_POSITIONS[upper]
    pose = tuple(low + alpha * (high - low) for low, high in zip(lower_pose, upper_pose, strict=True))
    derivative = tuple((high - low) / span for low, high in zip(lower_pose, upper_pose, strict=True))
    return pose, derivative
