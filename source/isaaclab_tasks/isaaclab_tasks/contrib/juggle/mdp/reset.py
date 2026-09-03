# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analytic phase resets for one-ball KUKA-Allegro juggling."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
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

JUGGLE_RESET_PARAMETER_DIM = 8
"""Dimension of the normalized continuous reset-parameter vector."""

_JUGGLE_HELD_HEIGHT_RANGE = (0.28, 0.56)
"""Tool-height range for generalized meter pre-throw starts [m]."""

_JUGGLE_HELD_WORKSPACE_YAW_AMPLITUDE = 0.24
"""Base-yaw amplitude for generalized meter pre-throw starts [rad]."""

_JUGGLE_HELD_ARM_PERTURBATION = (0.12, 0.10, 0.10)
"""Joint-2/3/4 posture amplitudes around the palm-up arm manifold [rad]."""

_JUGGLE_HELD_BALL_RADIAL_OFFSET = 0.005
"""Maximum ball-center offset across the palm cradle [m]."""

_JUGGLE_HELD_BALL_VERTICAL_OFFSET = 0.002
"""Maximum ball-center offset normal to the palm cradle [m]."""


@dataclass(frozen=True)
class JuggleResetProfile:
    """Physical difficulty ranges used to author one reset catalog.

    A profile changes only reset-state generation.  The progress term remains
    the authority for deciding whether a realized trajectory actually met the
    configured task objective.

    Attributes:
        name: Stable profile name.
        profile_id: Stable non-negative checkpoint identifier.
        minimum_apex_height_gain: Required ballistic height gain [m].
        release_speed_range: Authored upward release-speed range [m/s].
        lateral_speed_range: Authored horizontal release-speed range [m/s].
        lateral_y_amplitude: Horizontal Y-velocity sampling amplitude [m/s].
        moving_held_speed_range: Moving-prethrow tool-speed range [m/s].
        descending_time_after_apex_range: Authored descent-time range after apex [s].
        catch_approach_tool_speed_range: Downward tool-speed range for approach rows [m/s].
        catch_approach_relative_speed_range: Ball/tool relative-speed range for approach rows [m/s].
        catch_contact_tool_speed_range: Downward tool-speed range for contact rows [m/s].
        catch_contact_relative_speed_range: Ball/tool relative-speed range for contact rows [m/s].
        catch_hand_open_fraction: Catch-hand interpolation fraction between preload and flight poses.
        workspace_yaw_amplitude: Maximum base-joint yaw variation around the palm-up manifold [rad].
        difficulty_band_count: Number of reset difficulty bands.
    """

    name: str
    profile_id: int
    minimum_apex_height_gain: float
    release_speed_range: tuple[float, float]
    lateral_speed_range: tuple[float, float]
    lateral_y_amplitude: float
    moving_held_speed_range: tuple[float, float]
    descending_time_after_apex_range: tuple[float, float]
    catch_approach_tool_speed_range: tuple[float, float]
    catch_approach_relative_speed_range: tuple[float, float]
    catch_contact_tool_speed_range: tuple[float, float]
    catch_contact_relative_speed_range: tuple[float, float]
    catch_hand_open_fraction: float
    workspace_yaw_amplitude: float = 0.0
    difficulty_band_count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("A juggling reset profile requires a name.")
        if isinstance(self.profile_id, bool) or not isinstance(self.profile_id, int) or self.profile_id < 0:
            raise ValueError("A juggling reset profile ID must be a non-negative integer.")
        if not math.isfinite(self.minimum_apex_height_gain) or self.minimum_apex_height_gain <= 0.0:
            raise ValueError("The minimum apex-height gain must be finite and positive.")
        for field_name, values in (
            ("release speed", self.release_speed_range),
            ("lateral speed", self.lateral_speed_range),
            ("moving-held speed", self.moving_held_speed_range),
            ("post-apex time", self.descending_time_after_apex_range),
            ("catch-approach relative speed", self.catch_approach_relative_speed_range),
            ("catch-contact relative speed", self.catch_contact_relative_speed_range),
        ):
            if len(values) != 2 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"The {field_name} range must contain two finite values.")
            if values[0] < 0.0 or values[0] >= values[1]:
                raise ValueError(f"The {field_name} range must be non-negative and increasing.")
        for field_name, values in (
            ("catch-approach tool speed", self.catch_approach_tool_speed_range),
            ("catch-contact tool speed", self.catch_contact_tool_speed_range),
        ):
            if len(values) != 2 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"The {field_name} range must contain two finite values.")
            if values[0] < 0.0 or values[0] > values[1]:
                raise ValueError(f"The {field_name} range must be non-negative and non-decreasing.")
        if (
            not math.isfinite(self.lateral_y_amplitude)
            or not 0.0 <= self.lateral_y_amplitude <= self.lateral_speed_range[0]
        ):
            raise ValueError("The lateral Y amplitude must fit inside the minimum lateral speed.")
        if not math.isfinite(self.catch_hand_open_fraction) or not 0.0 <= self.catch_hand_open_fraction <= 1.0:
            raise ValueError("The catch hand open fraction must lie in [0, 1].")
        if not math.isfinite(self.workspace_yaw_amplitude) or not 0.0 <= self.workspace_yaw_amplitude <= 0.20:
            raise ValueError("The workspace yaw amplitude must lie in [0, 0.20] rad.")
        if (
            isinstance(self.difficulty_band_count, bool)
            or not isinstance(self.difficulty_band_count, int)
            or self.difficulty_band_count < 1
        ):
            raise ValueError("difficulty_band_count must be a positive integer.")
        ballistic_gain = self.release_speed_range[0] ** 2 / (2.0 * -GRAVITY_Z)
        if ballistic_gain < self.minimum_apex_height_gain:
            raise ValueError("The minimum release speed cannot reach the profile's apex-height gain.")


LOW_TOSS_RESET_PROFILE = JuggleResetProfile(
    name="low_toss",
    profile_id=0,
    minimum_apex_height_gain=0.06,
    release_speed_range=(1.10, 1.40),
    lateral_speed_range=(0.19, 0.34),
    lateral_y_amplitude=0.12,
    moving_held_speed_range=(0.80, 1.25),
    descending_time_after_apex_range=(0.018, 0.073),
    catch_approach_tool_speed_range=(0.0, 0.0),
    catch_approach_relative_speed_range=(0.15, 0.45),
    catch_contact_tool_speed_range=(0.0, 0.0),
    catch_contact_relative_speed_range=(0.10, 0.35),
    catch_hand_open_fraction=1.0,
    workspace_yaw_amplitude=0.0,
    difficulty_band_count=1,
)
"""The validated short-toss catalog used by the original task."""

METER_TOSS_RESET_PROFILE = JuggleResetProfile(
    name="meter_toss",
    profile_id=1,
    minimum_apex_height_gain=1.00,
    release_speed_range=(4.50, 4.75),
    # Keep a one-metre return inside the arm's empirically reachable XY
    # corridor.  The short task's 0.19--0.34 m/s drift would accumulate to
    # 17--31 cm over the roughly 0.9 s flight.
    lateral_speed_range=(0.02, 0.08),
    lateral_y_amplitude=0.015,
    # These rows bootstrap the launch controller from reachable arm motion.
    # Faster 3.6--4.8 m/s resets outran the controller and produced large XY
    # misses; the bounded range below retained zero-action safety while making
    # a true one-metre apex reachable under feedback control.
    moving_held_speed_range=(0.0, 1.0),
    # Cover early through late descent.  The final capture mechanics are
    # authored separately so a reset never teleports through contact.
    descending_time_after_apex_range=(0.040, 0.350),
    # Keep the return within the arm follower's measured capture basin. The
    # ball still falls relative to the hand, so policy control is required.
    catch_approach_tool_speed_range=(0.0, 1.0),
    catch_approach_relative_speed_range=(0.05, 0.15),
    catch_contact_tool_speed_range=(0.0, 1.0),
    catch_contact_relative_speed_range=(0.05, 0.15),
    # Half-open remained passive-safe but was reachable by one close action.
    catch_hand_open_fraction=0.50,
    # Base yaw moves the same palm-up manifold through a broader lateral
    # workspace without inventing unrelated arm postures.
    workspace_yaw_amplitude=0.15,
    difficulty_band_count=4,
)
"""Randomized one-metre launch, ballistic flight, and catchable return curriculum."""

_RESET_PROFILES = {
    LOW_TOSS_RESET_PROFILE.name: LOW_TOSS_RESET_PROFILE,
    METER_TOSS_RESET_PROFILE.name: METER_TOSS_RESET_PROFILE,
}
if len({profile.profile_id for profile in _RESET_PROFILES.values()}) != len(_RESET_PROFILES):
    raise RuntimeError("Built-in juggling reset profiles must have unique stable IDs.")


def juggle_reset_profile(name: str) -> JuggleResetProfile:
    """Resolve a public reset profile by name."""
    try:
        return _RESET_PROFILES[name]
    except KeyError as error:
        raise ValueError(f"Unknown juggling reset profile: {name!r}.") from error


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

_METER_LOCAL_GOAL_BY_PHASE = (
    JuggleLocalGoal.FLIGHT_APEX,
    JuggleLocalGoal.STABLE_CATCH,
    JuggleLocalGoal.STABLE_CATCH,
    JuggleLocalGoal.STABLE_CATCH,
    JuggleLocalGoal.STABLE_CATCH,
    JuggleLocalGoal.STABLE_CATCH,
    JuggleLocalGoal.STABLE_CATCH,
    JuggleLocalGoal.FLIGHT_APEX,
)
"""Action-dependent phase goals for the one-metre reset distribution.

Free-flight resets inherit enough upward velocity to cross the apex without a
policy action, so their local target is the stable catch rather than that
passive ballistic event. Held rows still require a deliberate release.
"""


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


def _catalog_reset_parameters(variant: int) -> tuple[float, ...]:
    """Return the historical low-discrepancy coordinates for one row variant."""
    sample_id = variant + 1
    return (
        _radical_inverse(sample_id, 2),
        _radical_inverse(sample_id, 3),
        _radical_inverse(sample_id, 5),
        _radical_inverse(sample_id, 7),
        _radical_inverse(sample_id, 11),
        _radical_inverse(sample_id, 13),
        _radical_inverse(sample_id, 17),
        _radical_inverse(sample_id, 19),
    )


class JuggleResetStateSource:
    """Immutable, balanced reset rows for one vertical toss/catch cycle.

    The source keeps physical rows separate from monitored competence items. Free-flight rows are generated
    from one release state with the ballistic equations; no state is interpolated across release or contact
    discontinuities.
    """

    def __init__(
        self,
        rows_per_phase: int = 64,
        device: str | torch.device = "cpu",
        profile: JuggleResetProfile = LOW_TOSS_RESET_PROFILE,
        parameter_sampling: str = "catalog",
        continuous_seed: int = 0,
    ) -> None:
        """Build a deterministic reset source.

        Args:
            rows_per_phase: Number of physical variations per phase.
            device: Torch device holding the reset tensors.
            profile: Physical ranges used to author the reset rows.
            parameter_sampling: ``"catalog"`` for the historical low-discrepancy rows or
                ``"continuous"`` for a scrambled Sobol proposal bank.
            continuous_seed: Non-negative Sobol scramble seed used by continuous proposals.
        """
        if isinstance(rows_per_phase, bool) or not isinstance(rows_per_phase, int) or rows_per_phase < 8:
            raise ValueError("rows_per_phase must be an integer of at least eight.")
        self.rows_per_phase = rows_per_phase
        self.device = torch.device(device)
        if not isinstance(profile, JuggleResetProfile):
            raise TypeError("profile must be a JuggleResetProfile.")
        if parameter_sampling not in ("catalog", "continuous"):
            raise ValueError("parameter_sampling must be 'catalog' or 'continuous'.")
        if isinstance(continuous_seed, bool) or not isinstance(continuous_seed, int) or continuous_seed < 0:
            raise ValueError("continuous_seed must be a non-negative integer.")
        self.profile = profile
        self.parameter_sampling = parameter_sampling
        self.continuous_seed = continuous_seed
        self.is_meter_profile = profile.profile_id == METER_TOSS_RESET_PROFILE.profile_id
        self.phase_count = len(JugglePhase)
        self.row_count = rows_per_phase * self.phase_count

        parameter_rows: list[tuple[float, ...] | None]
        if parameter_sampling == "continuous":
            engine = torch.quasirandom.SobolEngine(
                dimension=JUGGLE_RESET_PARAMETER_DIM,
                scramble=True,
                seed=continuous_seed,
            )
            parameter_rows = [tuple(float(value) for value in row) for row in engine.draw(self.row_count)]
        else:
            parameter_rows = [None] * self.row_count
        rows = [
            self._make_row(phase, variant, parameter_rows[int(phase) * rows_per_phase + variant])
            for phase in JugglePhase
            for variant in range(rows_per_phase)
        ]
        self.reset_parameters = torch.tensor(
            [
                _catalog_reset_parameters(variant) if parameters is None else parameters
                for phase in JugglePhase
                for variant, parameters in enumerate(
                    parameter_rows[int(phase) * rows_per_phase : (int(phase) + 1) * rows_per_phase]
                )
            ],
            dtype=torch.float32,
            device=self.device,
        )
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
        self.release_origins_xy = self.release_positions[:, :2].clone()
        self.release_velocities = torch.tensor([row[7] for row in rows], dtype=torch.float32, device=self.device)
        self.flight_times = torch.tensor([row[8] for row in rows], dtype=torch.float32, device=self.device)
        self.ballistic_rows = torch.tensor([row[9] for row in rows], dtype=torch.bool, device=self.device)
        self.launch_reference_heights = torch.tensor([row[10] for row in rows], dtype=torch.float32, device=self.device)
        self.static_held_rows = torch.tensor([row[11] for row in rows], dtype=torch.bool, device=self.device)
        self.preload_assist_rows = torch.tensor([row[12] for row in rows], dtype=torch.bool, device=self.device)
        if parameter_sampling == "continuous":
            self.difficulty_band_ids = torch.clamp(
                torch.floor(self.reset_parameters[:, 1] * self.profile.difficulty_band_count).long(),
                max=self.profile.difficulty_band_count - 1,
            )
        else:
            variants = torch.arange(self.row_count, device=self.device) % self.rows_per_phase
            self.difficulty_band_ids = torch.div(
                variants * self.profile.difficulty_band_count,
                self.rows_per_phase,
                rounding_mode="floor",
            )
        held_rows = self.phase_ids == int(JugglePhase.HELD_PRETHROW)
        self.canonical_start_rows = held_rows.clone()
        self.item_ids = self.phase_ids.clone()
        self.item_names = tuple(phase.name.lower() for phase in JugglePhase)
        local_goal_by_phase = _LOCAL_GOAL_BY_PHASE
        if self.is_meter_profile:
            # Only rest starts measure the complete deployment behavior. Moving
            # held rows are a separate launch-bootstrap item with an actual
            # one-metre apex goal, never an action label.
            self.canonical_start_rows &= self.static_held_rows
            moving_held_rows = held_rows & ~self.static_held_rows
            self.item_ids[moving_held_rows] = len(JugglePhase)
            self.item_names = (*self.item_names, "moving_held_launch")
            local_goal_by_phase = _METER_LOCAL_GOAL_BY_PHASE
        goal_lookup = torch.tensor([int(goal) for goal in local_goal_by_phase], dtype=torch.long, device=self.device)
        self.local_goal_ids = goal_lookup[self.phase_ids]
        self.model_features = self._model_features()

        item_count = int(self.item_ids.max().item()) + 1
        self.item_rows = tuple(torch.where(self.item_ids == item_id)[0] for item_id in range(item_count))
        self.item_phase_ids = torch.empty(item_count, dtype=torch.long, device=self.device)
        self.canonical_item_mask = torch.zeros(item_count, dtype=torch.bool, device=self.device)
        self.adaptive_item_mask = torch.ones(item_count, dtype=torch.bool, device=self.device)
        for item_id, item_rows in enumerate(self.item_rows):
            item_phases = torch.unique(self.phase_ids[item_rows])
            if item_phases.numel() != 1:
                raise RuntimeError("Every juggling competence item must belong to exactly one physical phase.")
            self.item_phase_ids[item_id] = item_phases[0]
            if torch.unique(self.local_goal_ids[item_rows]).numel() != 1:
                raise RuntimeError("A juggling competence item cannot mix physical local goals.")
            canonical_values = self.canonical_start_rows[item_rows]
            if canonical_values.any() and not canonical_values.all():
                raise RuntimeError("A juggling competence item cannot mix canonical and local reset rows.")
            self.canonical_item_mask[item_id] = canonical_values.all()
        self.adaptive_item_mask[self.canonical_item_mask] = False
        if self.is_meter_profile:
            # A settled-catch row plus reset preload assistance would mostly
            # train the reset mechanism. Keep it available for fixed-phase
            # inspection but out of the adaptive PPO distribution.
            stable_item = int(self.item_ids[self.phase_ids == int(JugglePhase.STABLE_CATCH)][0])
            self.adaptive_item_mask[stable_item] = False
        self.canonical_row_ids = torch.where(self.canonical_start_rows)[0]
        if self.canonical_row_ids.numel() == 0:
            raise RuntimeError("A juggling reset source requires at least one canonical start row.")
        if not bool(self.adaptive_item_mask.any()):
            raise RuntimeError("A juggling reset source requires at least one adaptive local item.")

        tensors = (
            self.arm_positions,
            self.arm_velocities,
            self.hand_positions,
            self.hand_velocities,
            self.ball_positions,
            self.ball_quaternions,
            self.ball_velocities,
            self.release_positions,
            self.release_origins_xy,
            self.release_velocities,
            self.flight_times,
            self.launch_reference_heights,
            self.local_goal_ids,
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
                "preload_assist": self.preload_assist_rows,
                "difficulty_band": self.difficulty_band_ids,
                "canonical_start": self.canonical_start_rows,
                "local_goal": self.local_goal_ids,
                "reset_parameters": self.reset_parameters,
            },
            row_to_item=self.item_ids,
        )
        self.phase_rows = tuple(torch.where(self.phase_ids == int(phase))[0] for phase in JugglePhase)

    def _make_row(
        self,
        phase: JugglePhase,
        variant: int,
        parameters: tuple[float, ...] | None = None,
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
        bool,
    ]:
        """Author one phase-local physical row."""
        if parameters is None:
            parameters = _catalog_reset_parameters(variant)
            continuous_parameters = False
        else:
            if len(parameters) != JUGGLE_RESET_PARAMETER_DIM:
                raise ValueError(f"parameters must contain {JUGGLE_RESET_PARAMETER_DIM} values.")
            if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in parameters):
                raise ValueError("parameters must be finite and lie in [0, 1].")
            continuous_parameters = True
        (
            height_coordinate,
            speed_coordinate,
            time_coordinate,
            catch_coordinate,
            lateral_unit,
            angle_coordinate,
            lateral_speed_coordinate,
            workspace_unit,
        ) = parameters
        lateral_coordinate = 2.0 * lateral_unit - 1.0
        workspace_coordinate = 2.0 * workspace_unit - 1.0
        workspace_yaw = self.profile.workspace_yaw_amplitude * workspace_coordinate

        release_height = 0.36 + 0.06 * height_coordinate
        release_minimum, release_maximum = self.profile.release_speed_range
        release_speed = release_minimum + (release_maximum - release_minimum) * speed_coordinate
        lateral_angle = 2.0 * math.pi * angle_coordinate
        lateral_minimum, lateral_maximum = self.profile.lateral_speed_range
        lateral_speed = lateral_minimum + (lateral_maximum - lateral_minimum) * lateral_speed_coordinate
        lateral_velocity_y = self.profile.lateral_y_amplitude * math.sin(lateral_angle)
        lateral_velocity_x = math.sqrt(max(0.0, lateral_speed**2 - lateral_velocity_y**2))
        if math.cos(lateral_angle) < 0.0:
            lateral_velocity_x = -lateral_velocity_x
        if self.profile.workspace_yaw_amplitude > 0.0:
            release_arm_position = torch.tensor(
                _interpolate_arm_anchor_with_yaw(release_height, workspace_yaw), dtype=torch.float64
            )
            release_position_tensor, _ = kuka_allegro_tool_pose(
                release_arm_position,
                JUGGLE_SPHERE_CENTER_OFFSET,
            )
            release_position = tuple(float(value) for value in release_position_tensor)
        else:
            # Keep the original LOW catalog bit-for-bit unchanged.
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
        preload_assist = False
        launch_reference_height = release_position[2]

        preload_hand = JUGGLE_SPHERE_PRELOAD_HAND_POSITION
        flight_hand = JUGGLE_SPHERE_FLIGHT_GATE_HAND_POSITION
        contact_hand = tuple(
            preload + self.profile.catch_hand_open_fraction * (opened - preload)
            for preload, opened in zip(preload_hand, flight_hand, strict=True)
        )
        catch_arm_height = 0.30 + 0.12 * catch_coordinate
        arm_velocity = (0.0,) * 7

        if phase is JugglePhase.HELD_PRETHROW:
            held_height = 0.30 + 0.12 * height_coordinate
            held_workspace_yaw = workspace_yaw
            held_arm_perturbation = (0.0, 0.0, 0.0)
            held_ball_offset = JUGGLE_SPHERE_CENTER_OFFSET
            if continuous_parameters and self.is_meter_profile:
                minimum_height, maximum_height = _JUGGLE_HELD_HEIGHT_RANGE
                held_height = minimum_height + (maximum_height - minimum_height) * height_coordinate
                held_workspace_yaw = _JUGGLE_HELD_WORKSPACE_YAW_AMPLITUDE * workspace_coordinate
                held_arm_perturbation = tuple(
                    amplitude * coordinate
                    for amplitude, coordinate in zip(
                        _JUGGLE_HELD_ARM_PERTURBATION,
                        (
                            2.0 * catch_coordinate - 1.0,
                            lateral_coordinate,
                            2.0 * angle_coordinate - 1.0,
                        ),
                        strict=True,
                    )
                )
                radial_offset = _JUGGLE_HELD_BALL_RADIAL_OFFSET * math.sqrt(lateral_speed_coordinate)
                radial_angle = 2.0 * math.pi * angle_coordinate
                within_motion_class = (2.0 * time_coordinate) % 1.0
                held_ball_offset = (
                    JUGGLE_SPHERE_CENTER_OFFSET[0] + radial_offset * math.cos(radial_angle),
                    JUGGLE_SPHERE_CENTER_OFFSET[1] + radial_offset * math.sin(radial_angle),
                    JUGGLE_SPHERE_CENTER_OFFSET[2]
                    + _JUGGLE_HELD_BALL_VERTICAL_OFFSET * (2.0 * within_motion_class - 1.0),
                )
            # Rest starts train the complete deployment task; moving attached
            # starts make deliberate release discoverable from randomized
            # heights and speeds without prescribing a policy action.
            static_held = time_coordinate < 0.5 if continuous_parameters else variant < self.rows_per_phase // 2
            moving_minimum, moving_maximum = self.profile.moving_held_speed_range
            upward_speed = 0.0 if static_held else moving_minimum + (moving_maximum - moving_minimum) * speed_coordinate
            arm_position, arm_velocity, ball_position, ball_velocity = _moving_arm_state(
                held_height,
                upward_speed,
                held_workspace_yaw,
                joint_perturbation=held_arm_perturbation,
                tool_offset=held_ball_offset,
            )
            hand_position = preload_hand
            preload_assist = True
            launch_reference_height = ball_position[2]
        elif phase is JugglePhase.RELEASE:
            gate_gap = 0.07 + 0.03 * time_coordinate
            tool_height = min(_ARM_ANCHOR_HEIGHTS[-1], max(_ARM_ANCHOR_HEIGHTS[0], release_position[2] - gate_gap))
            arm_position = _interpolate_arm_anchor_with_yaw(tool_height, workspace_yaw)
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
                descent_minimum, descent_maximum = self.profile.descending_time_after_apex_range
                flight_time = apex_time + descent_minimum + (descent_maximum - descent_minimum) * time_coordinate
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
            arm_position = _interpolate_arm_anchor_with_yaw(tool_height, workspace_yaw)
            hand_position = flight_hand
            ballistic = True
        elif phase in (JugglePhase.CATCH_APPROACH, JugglePhase.CATCH_CONTACT):
            if phase is JugglePhase.CATCH_APPROACH:
                if self.is_meter_profile:
                    z_offset = 0.050 + 0.015 * height_coordinate
                else:
                    z_offset = 0.080 + 0.040 * height_coordinate
                # The reachable palm-up gate is robust to five millimetres of
                # lateral miss across the complete approach-speed/height grid.
                # Larger synthetic offsets fall outside that verified capture
                # basin and turn close-command rows into unavoidable misses.
                lateral_limit = 0.005
                x_offset = lateral_limit * lateral_coordinate
                y_offset = lateral_limit * (2.0 * angle_coordinate - 1.0)
                horizontal_speed = -0.04 * lateral_coordinate
                tool_minimum, tool_maximum = self.profile.catch_approach_tool_speed_range
                relative_minimum, relative_maximum = self.profile.catch_approach_relative_speed_range
            else:
                if self.is_meter_profile:
                    z_offset = 0.045 + 0.010 * height_coordinate
                else:
                    z_offset = 0.040 + 0.035 * height_coordinate
                lateral_limit = 0.004
                x_offset = lateral_limit * lateral_coordinate
                y_offset = lateral_limit * (2.0 * angle_coordinate - 1.0)
                horizontal_speed = -0.03 * lateral_coordinate
                tool_minimum, tool_maximum = self.profile.catch_contact_tool_speed_range
                relative_minimum, relative_maximum = self.profile.catch_contact_relative_speed_range
            tool_downward_speed = -(tool_minimum + (tool_maximum - tool_minimum) * speed_coordinate)
            relative_downward_speed = relative_minimum + (relative_maximum - relative_minimum) * speed_coordinate
            arm_position, arm_velocity, ball_position, ball_velocity = _moving_arm_state(
                catch_arm_height,
                tool_downward_speed,
                workspace_yaw,
            )
            ball_position = (
                ball_position[0] + x_offset,
                ball_position[1] + y_offset,
                ball_position[2] + z_offset,
            )
            ball_velocity = (
                ball_velocity[0] + horizontal_speed,
                ball_velocity[1],
                ball_velocity[2] - relative_downward_speed,
            )
            hand_position = contact_hand
        else:
            stable_height = 0.30 + 0.12 * height_coordinate
            arm_position, arm_velocity, ball_position, ball_velocity = _moving_arm_state(
                stable_height,
                0.0,
                workspace_yaw,
            )
            hand_position = preload_hand
            launch_reference_height = ball_position[2]
            preload_assist = True

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
            preload_assist,
        )

    def _model_features(self) -> torch.Tensor:
        """Return normalized parameters with phase-irrelevant coordinates masked out."""
        features = self.reset_parameters.clone()
        relevant_by_phase = torch.tensor(
            (
                (1, 1, 1, 1, 1, 1, 1, 1),
                (1, 1, 1, 0, 1, 1, 1, 1),
                (1, 1, 1, 1, 1, 1, 1, 1),
                (1, 1, 1, 1, 1, 1, 1, 1),
                (1, 1, 1, 1, 1, 1, 1, 1),
                (1, 1, 0, 1, 1, 1, 0, 1),
                (1, 1, 0, 1, 1, 1, 0, 1),
                (1, 0, 0, 0, 0, 0, 0, 1),
            ),
            dtype=features.dtype,
            device=self.device,
        )
        return features * relevant_by_phase[self.phase_ids]


class JuggleResetEvent(ManagerTermBase):
    """Apply rows selected by the juggling reset curriculum."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        rows_per_phase = int(cfg.params.get("rows_per_phase", 64))
        profile = juggle_reset_profile(str(cfg.params.get("profile", LOW_TOSS_RESET_PROFILE.name)))
        sampling_mode = str(cfg.params.get("sampling_mode", "semantic"))
        continuous_seed = int(cfg.params.get("continuous_seed", 0))
        self.source = JuggleResetStateSource(
            rows_per_phase=rows_per_phase,
            device=env.device,
            profile=profile,
            parameter_sampling="continuous" if sampling_mode == "continuous" else "catalog",
            continuous_seed=continuous_seed,
        )
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
        profile: str = LOW_TOSS_RESET_PROFILE.name,
        sampling_mode: str = "semantic",
        continuous_seed: int = 0,
    ) -> None:
        """Write selected robot and ball states to simulation."""
        del rows_per_phase, profile, sampling_mode, continuous_seed
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
            state=self.runtime,
            env_ids=ids,
            phases=phases,
            release_heights=self.source.launch_reference_heights[row_ids],
            static_held_start=self.source.static_held_rows[row_ids],
            preload_assist_start=self.source.preload_assist_rows[row_ids],
            item_ids=self.source.item_ids[row_ids],
            canonical_start=self.source.canonical_start_rows[row_ids],
            release_origins_xy=self.source.release_origins_xy[row_ids],
            local_goal_ids=self.source.local_goal_ids[row_ids],
        )


def _moving_arm_state(
    height: float,
    upward_speed: float,
    workspace_yaw: float = 0.0,
    *,
    joint_perturbation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tool_offset: tuple[float, float, float] = JUGGLE_SPHERE_CENTER_OFFSET,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Return an attached arm/ball state with FK-consistent velocity."""
    if len(joint_perturbation) != 3 or not all(math.isfinite(value) for value in joint_perturbation):
        raise ValueError("joint_perturbation must contain three finite values.")
    if len(tool_offset) != 3 or not all(math.isfinite(value) for value in tool_offset):
        raise ValueError("tool_offset must contain three finite values.")
    arm_position, unit_arm_velocity = _interpolate_arm_anchor_and_derivative(height)
    arm_position = (
        arm_position[0] + workspace_yaw,
        arm_position[1] + joint_perturbation[0],
        arm_position[2] + joint_perturbation[1],
        arm_position[3] + joint_perturbation[2],
        *arm_position[4:],
    )
    position_tensor = torch.tensor(arm_position, dtype=torch.float64)
    unit_velocity_tensor = torch.tensor(unit_arm_velocity, dtype=torch.float64)
    unit_tool_velocity = kuka_allegro_tool_point_velocity(position_tensor, unit_velocity_tensor, tool_offset)
    if abs(float(unit_tool_velocity[2])) < 0.5:
        raise RuntimeError("Calibrated upward arm velocity does not move the sphere center upward.")
    arm_velocity_tensor = unit_velocity_tensor * (upward_speed / float(unit_tool_velocity[2]))
    ball_position, _ = kuka_allegro_tool_pose(position_tensor, tool_offset)
    ball_velocity = kuka_allegro_tool_point_velocity(position_tensor, arm_velocity_tensor, tool_offset)
    return (
        tuple(float(value) for value in position_tensor),
        tuple(float(value) for value in arm_velocity_tensor),
        tuple(float(value) for value in ball_position),
        tuple(float(value) for value in ball_velocity),
    )


def _interpolate_arm_anchor(height: float) -> tuple[float, ...]:
    """Interpolate calibrated arm positions [rad]."""
    return _interpolate_arm_anchor_and_derivative(height)[0]


def _interpolate_arm_anchor_with_yaw(height: float, workspace_yaw: float) -> tuple[float, ...]:
    """Interpolate a palm-up arm position and rotate it through the workspace [rad]."""
    position = _interpolate_arm_anchor(height)
    return (position[0] + workspace_yaw, *position[1:])


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
