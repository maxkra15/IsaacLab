# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-relative waypoint commands for the FLARE slung-load task."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import euler_xyz_from_quat

from .bodies import link_lin_vel_w, link_pose_w
from .episode_metrics import EpisodeMetricAccumulator
from .observations import cable_joint_error, cable_relative_separation, payload_transverse_velocity_b, total_swing_angle

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_WAYPOINT_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/route_waypoints",
    markers={
        "active": sim_utils.SphereCfg(
            radius=0.12,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.72, 0.05), emissive_color=(0.25, 0.12, 0.0)
            ),
        ),
        "completed": sim_utils.SphereCfg(
            radius=0.09,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.85, 0.25)),
        ),
        "future": sim_utils.SphereCfg(
            radius=0.09,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.65, 1.0)),
        ),
    },
)

_ROUTE_SEGMENT_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/route_segments",
    markers={
        "completed": sim_utils.CylinderCfg(
            radius=0.015,
            height=1.0,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.85, 0.25)),
        ),
        "future": sim_utils.CylinderCfg(
            radius=0.015,
            height=1.0,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.65, 1.0)),
        ),
    },
)


def _route_segment_marker_transforms(
    waypoints_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return midpoint, orientation, and scale for Z-axis unit cylinders joining waypoints."""
    segment = waypoints_w[:, 1:] - waypoints_w[:, :-1]
    length = torch.linalg.norm(segment, dim=-1)
    direction = segment / length.clamp_min(torch.finfo(segment.dtype).eps).unsqueeze(-1)

    # Shortest-arc quaternion from local +Z to the segment direction. The only
    # singular case is exactly -Z, for which a 180-degree X rotation is used.
    quat = torch.stack(
        (-direction[..., 1], direction[..., 0], torch.zeros_like(length), 1.0 + direction[..., 2]), dim=-1
    )
    opposite = quat.square().sum(dim=-1) <= torch.finfo(segment.dtype).eps
    opposite_quat = torch.zeros_like(quat)
    opposite_quat[..., 0] = 1.0
    quat = torch.where(opposite.unsqueeze(-1), opposite_quat, quat)
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(torch.finfo(segment.dtype).eps)

    midpoint = 0.5 * (waypoints_w[:, 1:] + waypoints_w[:, :-1])
    scale = torch.ones_like(midpoint)
    scale[..., 2] = length
    return midpoint, quat, scale


def _sample_bounded_waypoint_offsets(
    num_routes: int,
    num_waypoints: int,
    lower: torch.Tensor,
    upper: torch.Tensor,
    minimum_separation: float,
    max_sampling_attempts: int,
    maximum_separation: float | None = None,
    maximum_heading_change: float | None = None,
    maximum_vertical_step: float | None = None,
    nominal_heading_change: float | None = None,
    route_sampling_attempts: int = 4,
    random_heading_change_interval: int = 1,
    independent_initial_heading_attempts: bool = False,
    select_smoothest_route_attempt: bool = True,
) -> torch.Tensor:
    """Sample reset-relative waypoint routes inside an axis-aligned box.

    Sampling is vectorized across environments and candidate attempts.
    Rejection sampling preserves a uniform conditional distribution whenever
    it succeeds. The deterministic geometric fallback makes the separation
    contract exact even for an exceptionally unlucky sequence of draws.

    Args:
        num_routes: Number of routes to sample.
        num_waypoints: Number of waypoints per route.
        lower: Lower XYZ bounds [m], shape ``(3,)`` or ``(num_routes, 3)``.
        upper: Upper XYZ bounds [m], shape ``(3,)`` or ``(num_routes, 3)``.
        minimum_separation: Minimum anchor-to-first and consecutive spacing [m].
        max_sampling_attempts: Number of uniform candidates tried per waypoint.
        maximum_separation: Optional maximum anchor-to-first and consecutive spacing [m].
        maximum_heading_change: Optional maximum planar heading change [rad] between
            consecutive segments. The first segment remains uniformly distributed
            over all headings.
        nominal_heading_change: Half-width [rad] of the uniform turn distribution
            away from boundaries. Defaults to ``maximum_heading_change``.
        maximum_vertical_step: Optional maximum absolute vertical change [m] per
            segment. This is intended for use with ``maximum_heading_change``.
        route_sampling_attempts: Complete route candidates sampled before a
            deterministic smooth fallback is used.
        random_heading_change_interval: Number of chords between unconstrained
            random heading-change draws. Intermediate chords hold heading
            except for the existing boundary-centering correction.
        independent_initial_heading_attempts: Whether complete-route retries
            sample independent initial headings. The default retains one
            heading per output route; anchor-aware mixtures enable this so a
            boundary-facing first draw cannot invalidate every retry.
        select_smoothest_route_attempt: Whether to choose the valid retry with
            the lowest mean turn. When ``False``, choose the first valid retry
            without biasing randomized corners toward straighter routes.

    Returns:
        Reset-relative waypoint offsets [m], shape ``(num_routes, num_waypoints, 3)``.
    """
    if (
        isinstance(random_heading_change_interval, bool)
        or not isinstance(random_heading_change_interval, int)
        or random_heading_change_interval < 1
    ):
        raise ValueError("random_heading_change_interval must be a positive integer.")
    if not isinstance(select_smoothest_route_attempt, bool):
        raise ValueError("select_smoothest_route_attempt must be a boolean.")
    requested_num_routes = num_routes
    route_candidate_count = route_sampling_attempts if maximum_heading_change is not None else 1
    num_routes *= route_candidate_count

    def expand_bounds(value: torch.Tensor, name: str) -> torch.Tensor:
        if value.shape == (3,):
            per_route = value.unsqueeze(0).expand(requested_num_routes, -1)
        elif value.shape == (requested_num_routes, 3):
            per_route = value
        else:
            raise ValueError(f"{name} must have shape (3,) or ({requested_num_routes}, 3).")
        return per_route.repeat(route_candidate_count, 1)

    lower = expand_bounds(lower, "lower")
    upper = expand_bounds(upper, "upper")
    route_initial_heading = None
    if maximum_heading_change is not None:
        heading_count = num_routes if independent_initial_heading_attempts else requested_num_routes
        initial_heading = 2.0 * torch.pi * torch.rand(heading_count, device=lower.device, dtype=lower.dtype) - torch.pi
        route_initial_heading = (
            initial_heading if independent_initial_heading_attempts else initial_heading.repeat(route_candidate_count)
        )
    offsets = torch.empty(num_routes, num_waypoints, 3, device=lower.device, dtype=lower.dtype)
    previous = torch.zeros(num_routes, 3, device=lower.device, dtype=lower.dtype)
    minimum_separation_sq = minimum_separation**2
    maximum_separation_sq = None if maximum_separation is None else maximum_separation**2
    span = upper - lower
    previous_heading = torch.zeros(num_routes, device=lower.device, dtype=lower.dtype)
    has_heading = torch.zeros(num_routes, device=lower.device, dtype=torch.bool)
    nominal_turn_limit = maximum_heading_change if nominal_heading_change is None else nominal_heading_change

    def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi

    def sample_candidates() -> torch.Tensor:
        if maximum_separation is None:
            sample = torch.rand(max_sampling_attempts, num_routes, 3, device=lower.device, dtype=lower.dtype)
            return lower + sample * span

        if maximum_heading_change is not None:
            # The first segment covers the full compass. Later segments sample
            # bounded changes about the accepted planar heading, producing
            # smooth routes without privileging any global travel direction.
            sample = torch.rand(max_sampling_attempts, num_routes, 3, device=lower.device, dtype=lower.dtype)
            radius = minimum_separation + sample[..., 0] * (maximum_separation - minimum_separation)
            assert route_initial_heading is not None
            initial_heading = route_initial_heading.unsqueeze(0).expand(max_sampling_attempts, -1)
            assert nominal_turn_limit is not None
            random_heading_change = nominal_turn_limit * (2.0 * sample[..., 1] - 1.0)
            if waypoint_index % random_heading_change_interval != 0:
                random_heading_change.zero_()
            box_center = 0.5 * (lower + upper)
            half_span = 0.5 * span
            normalized_position = torch.abs((previous - box_center) / half_span.clamp_min(torch.finfo(lower.dtype).eps))
            boundary_proximity = torch.amax(normalized_position[:, :2], dim=-1)
            steering_weight = ((boundary_proximity - 0.45) / 0.35).clamp(0.0, 1.0)
            heading_to_center = torch.atan2(box_center[:, 1] - previous[:, 1], box_center[:, 0] - previous[:, 0])
            center_heading_change = wrap_to_pi(heading_to_center - previous_heading).clamp(
                -nominal_turn_limit, nominal_turn_limit
            )
            heading_change = (1.0 - steering_weight).unsqueeze(0) * random_heading_change
            heading_change += steering_weight.unsqueeze(0) * center_heading_change.unsqueeze(0)
            heading = torch.where(
                has_heading.unsqueeze(0), previous_heading.unsqueeze(0) + heading_change, initial_heading
            )

            vertical_limit = radius
            if maximum_vertical_step is not None:
                vertical_limit = torch.minimum(vertical_limit, torch.full_like(radius, maximum_vertical_step))
            vertical_step = vertical_limit * (2.0 * sample[..., 2] - 1.0)
            planar_step = torch.sqrt((radius.square() - vertical_step.square()).clamp_min(0.0))
            displacement = torch.stack(
                (planar_step * torch.cos(heading), planar_step * torch.sin(heading), vertical_step), dim=-1
            )
            return previous.unsqueeze(0) + displacement

        # Uniform samples from a spherical shell make bounded-step routes an
        # isotropic random walk instead of a sequence of unrelated box points.
        sample = torch.rand(max_sampling_attempts, num_routes, 3, device=lower.device, dtype=lower.dtype)
        direction_z = 2.0 * sample[..., 0] - 1.0
        direction_xy = torch.sqrt((1.0 - direction_z.square()).clamp_min(0.0))
        azimuth = 2.0 * torch.pi * sample[..., 1]
        direction = torch.stack(
            (direction_xy * torch.cos(azimuth), direction_xy * torch.sin(azimuth), direction_z), dim=-1
        )
        radius = (minimum_separation**3 + sample[..., 2] * (maximum_separation**3 - minimum_separation**3)).pow(
            1.0 / 3.0
        )
        return previous.unsqueeze(0) + radius.unsqueeze(-1) * direction

    def is_valid(candidate: torch.Tensor) -> torch.Tensor:
        distance_sq = torch.sum(torch.square(candidate - previous.unsqueeze(0)), dim=-1)
        valid = torch.all((candidate >= lower) & (candidate <= upper), dim=-1)
        valid &= distance_sq >= minimum_separation_sq
        if maximum_separation_sq is not None:
            valid &= distance_sq <= maximum_separation_sq
        return valid

    def smooth_fallback() -> tuple[torch.Tensor, torch.Tensor]:
        """Try deterministic headings within the configured turn limit."""
        assert maximum_heading_change is not None
        assert maximum_separation is not None
        assert nominal_turn_limit is not None
        candidate_count = max(max_sampling_attempts, 17)
        turn = torch.linspace(
            -maximum_heading_change,
            maximum_heading_change,
            candidate_count,
            device=lower.device,
            dtype=lower.dtype,
        ).unsqueeze(1)
        heading_to_center = torch.atan2(
            0.5 * (lower[:, 1] + upper[:, 1]) - previous[:, 1],
            0.5 * (lower[:, 0] + upper[:, 0]) - previous[:, 0],
        )
        center_turn = wrap_to_pi(heading_to_center - previous_heading).clamp(
            -maximum_heading_change, maximum_heading_change
        )
        # Center-steering is considered first, followed by an evenly spaced
        # turn fan. This resolves nearly all boundary rejections without a CPU
        # branch or a reversal.
        turn = torch.cat((center_turn.unsqueeze(0), turn.expand(-1, num_routes)), dim=0)
        heading = previous_heading.unsqueeze(0) + turn
        radius = torch.full_like(heading, minimum_separation)
        vertical_step = torch.zeros_like(radius)
        planar_step = torch.sqrt((radius.square() - vertical_step.square()).clamp_min(0.0))
        candidate = previous.unsqueeze(0) + torch.stack(
            (planar_step * torch.cos(heading), planar_step * torch.sin(heading), vertical_step), dim=-1
        )
        valid = is_valid(candidate)
        has_valid = torch.any(valid, dim=0)
        first_valid = torch.argmax(valid.to(torch.int32), dim=0)
        route_ids = torch.arange(num_routes, device=lower.device)
        return candidate[first_valid, route_ids], has_valid

    def feasible_fallback() -> torch.Tensor:
        # Move from the closest box point toward the farthest corner until the
        # minimum radius is reached. Convexity keeps the result inside bounds;
        # validation guarantees the target radius does not exceed the maximum.
        closest = torch.maximum(torch.minimum(previous, upper), lower)
        farthest = torch.where(previous < 0.5 * (lower + upper), upper, lower)
        if maximum_heading_change is not None:
            # Correlated routes start inside the box. A step toward the farthest
            # corner is always feasible under the validated half-diagonal
            # constraint, and limiting its Z component preserves the vertical
            # step contract. This final fallback is only needed for geometries
            # whose boundary leaves no heading inside the configured turn fan.
            direction = farthest - previous
            if maximum_vertical_step is not None:
                direction[..., 2] = direction[..., 2].clamp(-maximum_vertical_step, maximum_vertical_step)
            distance = torch.linalg.vector_norm(direction, dim=-1)
            target = torch.full_like(distance, minimum_separation)
            normalized_direction = direction / distance.clamp_min(torch.finfo(lower.dtype).eps).unsqueeze(-1)
            return previous + target.unsqueeze(-1) * normalized_direction
        closest_distance = torch.linalg.vector_norm(closest - previous, dim=-1)
        target_distance = torch.maximum(closest_distance, torch.full_like(closest_distance, minimum_separation))
        box_direction = farthest - closest
        offset = closest - previous
        quadratic_a = torch.sum(box_direction.square(), dim=-1)
        quadratic_b = 2.0 * torch.sum(offset * box_direction, dim=-1)
        quadratic_c = torch.sum(offset.square(), dim=-1) - target_distance.square()
        discriminant = (quadratic_b.square() - 4.0 * quadratic_a * quadratic_c).clamp_min(0.0)
        fraction = (-quadratic_b + torch.sqrt(discriminant)) / (2.0 * quadratic_a).clamp_min(
            torch.finfo(lower.dtype).eps
        )
        fraction = torch.where(quadratic_a > 0.0, fraction.clamp(0.0, 1.0), torch.zeros_like(fraction))
        return (closest + fraction.unsqueeze(-1) * box_direction).clamp(lower, upper)

    for waypoint_index in range(num_waypoints):
        candidates = sample_candidates()
        valid = is_valid(candidates)
        has_valid = torch.any(valid, dim=0)
        first_valid = torch.argmax(valid.to(torch.int32), dim=0)
        route_ids = torch.arange(num_routes, device=lower.device)
        candidate = candidates[first_valid, route_ids]
        if maximum_heading_change is not None:
            smooth_candidate, has_smooth_candidate = smooth_fallback()
            candidate = torch.where(has_valid.unsqueeze(-1), candidate, smooth_candidate)
            has_valid |= has_smooth_candidate
        candidate = torch.where(has_valid.unsqueeze(-1), candidate, feasible_fallback())
        offsets[:, waypoint_index] = candidate
        displacement = candidate - previous
        planar_distance = torch.linalg.vector_norm(displacement[:, :2], dim=-1)
        candidate_heading = torch.atan2(displacement[:, 1], displacement[:, 0])
        previous_heading = torch.where(
            planar_distance > torch.finfo(lower.dtype).eps, candidate_heading, previous_heading
        )
        has_heading |= planar_distance > torch.finfo(lower.dtype).eps
        previous = candidate

    if maximum_heading_change is None:
        return offsets

    candidate_routes = offsets.reshape(route_candidate_count, requested_num_routes, num_waypoints, 3)
    anchor = torch.zeros(route_candidate_count, requested_num_routes, 1, 3, device=lower.device, dtype=lower.dtype)
    segment = torch.diff(torch.cat((anchor, candidate_routes), dim=2), dim=2)
    heading = torch.atan2(segment[..., 1], segment[..., 0])
    heading_change = torch.abs(wrap_to_pi(torch.diff(heading, dim=2)))
    # Reconstructing headings from accumulated float32 positions can amplify
    # roundoff near the configured limit. A scale-aware tolerance prevents one
    # numerically marginal turn in a long route from replacing the entire route
    # with the deterministic constant-curvature feasibility fallback.
    heading_tolerance = 128.0 * torch.finfo(lower.dtype).eps
    route_valid = torch.all(heading_change <= maximum_heading_change + heading_tolerance, dim=2)
    has_valid_route = torch.any(route_valid, dim=0)
    if select_smoothest_route_attempt:
        route_curvature = torch.mean(heading_change, dim=2)
        route_curvature = torch.where(route_valid, route_curvature, torch.full_like(route_curvature, torch.inf))
        selected_route_attempt = torch.argmin(route_curvature, dim=0)
    else:
        selected_route_attempt = torch.argmax(route_valid.to(torch.int32), dim=0)
    route_ids = torch.arange(requested_num_routes, device=lower.device)
    selected = candidate_routes[selected_route_attempt, route_ids]

    # A constant-curvature polygon is a deterministic feasibility fallback:
    # each chord has the minimum configured length and turns by exactly the
    # configured limit. Validation ensures its full circle fits the XY box.
    initial_heading = torch.atan2(candidate_routes[0, :, 0, 1], candidate_routes[0, :, 0, 0])
    chirality = torch.where(
        torch.rand(requested_num_routes, device=lower.device, dtype=lower.dtype) < 0.5,
        -torch.ones(requested_num_routes, device=lower.device, dtype=lower.dtype),
        torch.ones(requested_num_routes, device=lower.device, dtype=lower.dtype),
    )
    segment_id = torch.arange(num_waypoints, device=lower.device, dtype=lower.dtype).unsqueeze(0)
    fallback_heading = initial_heading.unsqueeze(1) + chirality.unsqueeze(1) * maximum_heading_change * segment_id
    fallback_segment = torch.stack(
        (
            minimum_separation * torch.cos(fallback_heading),
            minimum_separation * torch.sin(fallback_heading),
            torch.zeros_like(fallback_heading),
        ),
        dim=-1,
    )
    fallback_route = torch.cumsum(fallback_segment, dim=1)
    return torch.where(has_valid_route[:, None, None], selected, fallback_route)


def _sample_bounded_ellipse_waypoints(
    anchor_e: torch.Tensor,
    num_waypoints: int,
    samples_per_lap: int,
    aspect_ratio_range: tuple[float, float],
    vertical_amplitude_range: tuple[float, float],
) -> torch.Tensor:
    """Sample closed, origin-centered ellipse routes from perimeter anchors.

    The anchor's planar radius and azimuth define the ellipse major radius and
    orientation. Each environment independently samples its minor-to-major axis
    ratio, travel direction, and vertical sinusoid amplitude. Whole laps close
    exactly on the anchor, so a multi-lap route is continuous without a
    rejection sampler or a mid-episode coordinate drift.

    Args:
        anchor_e: Post-reset route anchors in environment coordinates [m], shape ``(N, 3)``.
        num_waypoints: Number of route samples. Must be a multiple of ``samples_per_lap``.
        samples_per_lap: Uniform samples in each closed lap.
        aspect_ratio_range: Inclusive minor-to-major axis ratio range.
        vertical_amplitude_range: Inclusive vertical sinusoid amplitude range [m].

    Returns:
        Absolute environment-frame waypoints [m], shape ``(N, num_waypoints, 3)``.
    """
    if anchor_e.ndim != 2 or anchor_e.shape[1] != 3:
        raise ValueError(f"anchor_e must have shape (N, 3), got {tuple(anchor_e.shape)}.")
    if num_waypoints <= 0 or samples_per_lap <= 0 or num_waypoints % samples_per_lap != 0:
        raise ValueError("num_waypoints must be a positive multiple of samples_per_lap.")

    planar_anchor = anchor_e[:, :2]
    major_radius = torch.linalg.vector_norm(planar_anchor, dim=-1)
    if not torch.isfinite(anchor_e).all() or torch.any(major_radius <= torch.finfo(anchor_e.dtype).eps):
        raise ValueError("Bounded ellipse anchors must be finite and have positive planar radius.")

    route_count = anchor_e.shape[0]
    aspect_low, aspect_high = aspect_ratio_range
    amplitude_low, amplitude_high = vertical_amplitude_range
    aspect_ratio = aspect_low + (aspect_high - aspect_low) * torch.rand(
        route_count, device=anchor_e.device, dtype=anchor_e.dtype
    )
    vertical_amplitude = amplitude_low + (amplitude_high - amplitude_low) * torch.rand(
        route_count, device=anchor_e.device, dtype=anchor_e.dtype
    )
    chirality = torch.where(
        torch.rand(route_count, device=anchor_e.device, dtype=anchor_e.dtype) < 0.5,
        -torch.ones(route_count, device=anchor_e.device, dtype=anchor_e.dtype),
        torch.ones(route_count, device=anchor_e.device, dtype=anchor_e.dtype),
    )

    major_axis = planar_anchor / major_radius.unsqueeze(-1)
    minor_axis = torch.stack((-major_axis[:, 1], major_axis[:, 0]), dim=-1)
    sample_index = torch.arange(1, num_waypoints + 1, device=anchor_e.device, dtype=anchor_e.dtype)
    phase = chirality.unsqueeze(1) * (math.tau / samples_per_lap) * sample_index.unsqueeze(0)
    ellipse_xy = major_radius[:, None, None] * torch.cos(phase).unsqueeze(-1) * major_axis[:, None, :]
    ellipse_xy += (
        major_radius[:, None, None]
        * aspect_ratio[:, None, None]
        * torch.sin(phase).unsqueeze(-1)
        * minor_axis[:, None, :]
    )
    height = anchor_e[:, 2:3] + vertical_amplitude[:, None] * torch.sin(2.0 * phase)
    waypoints_e = torch.cat((ellipse_xy, height.unsqueeze(-1)), dim=-1)

    # Avoid sub-ULP closure gaps from evaluating sin/cos at integer multiples
    # of 2*pi. Exact repeated knots also preserve the indexed lap transition.
    lap_closure = torch.remainder(torch.arange(1, num_waypoints + 1, device=anchor_e.device), samples_per_lap) == 0
    return torch.where(lap_closure[None, :, None], anchor_e[:, None, :], waypoints_e)


def _sample_bounded_figure_eight_waypoints(
    anchor_e: torch.Tensor,
    num_waypoints: int,
    samples_per_lap: int,
    vertical_amplitude_range: tuple[float, float],
) -> torch.Tensor:
    """Sample closed figure-eights made from two tangent circular lobes.

    The post-reset anchor is the outer tip of one lobe. One lap follows half
    of that lobe to the origin, circles the opposite lobe, and returns over the
    remaining half. The two origin visits have distinct progress indices, so
    projection and waypoint advancement cannot switch to the other branch.

    Args:
        anchor_e: Post-reset route anchors in environment coordinates [m], shape ``(N, 3)``.
        num_waypoints: Number of route samples. Must be a multiple of ``samples_per_lap``.
        samples_per_lap: Uniform samples in each closed lap. Must be divisible by four.
        vertical_amplitude_range: Inclusive vertical sinusoid amplitude range [m].

    Returns:
        Absolute environment-frame waypoints [m], shape ``(N, num_waypoints, 3)``.
    """
    if anchor_e.ndim != 2 or anchor_e.shape[1] != 3:
        raise ValueError(f"anchor_e must have shape (N, 3), got {tuple(anchor_e.shape)}.")
    if num_waypoints <= 0 or samples_per_lap <= 0 or samples_per_lap % 4 != 0 or num_waypoints % samples_per_lap != 0:
        raise ValueError("num_waypoints must be a multiple of a samples_per_lap value divisible by four.")

    planar_anchor = anchor_e[:, :2]
    outer_radius = torch.linalg.vector_norm(planar_anchor, dim=-1)
    if not torch.isfinite(anchor_e).all() or torch.any(outer_radius <= torch.finfo(anchor_e.dtype).eps):
        raise ValueError("Bounded figure-eight anchors must be finite and have positive planar radius.")

    route_count = anchor_e.shape[0]
    amplitude_low, amplitude_high = vertical_amplitude_range
    vertical_amplitude = amplitude_low + (amplitude_high - amplitude_low) * torch.rand(
        route_count, device=anchor_e.device, dtype=anchor_e.dtype
    )
    chirality = torch.where(
        torch.rand(route_count, device=anchor_e.device, dtype=anchor_e.dtype) < 0.5,
        -torch.ones(route_count, device=anchor_e.device, dtype=anchor_e.dtype),
        torch.ones(route_count, device=anchor_e.device, dtype=anchor_e.dtype),
    )

    sample_number = torch.arange(1, num_waypoints + 1, device=anchor_e.device)
    lap_sample = torch.remainder(sample_number - 1, samples_per_lap) + 1
    phase = lap_sample.to(anchor_e.dtype) / float(samples_per_lap)
    lobe_radius = 0.5 * outer_radius[:, None]

    first_angle = 4.0 * math.pi * phase
    middle_angle = -4.0 * math.pi * (phase - 0.25)
    final_angle = math.pi + 4.0 * math.pi * (phase - 0.75)
    local_x = torch.where(
        phase <= 0.25,
        lobe_radius * (1.0 + torch.cos(first_angle)),
        torch.where(
            phase <= 0.75,
            lobe_radius * (-1.0 + torch.cos(middle_angle)),
            lobe_radius * (1.0 + torch.cos(final_angle)),
        ),
    )
    local_y = torch.where(
        phase <= 0.25,
        lobe_radius * torch.sin(first_angle),
        torch.where(
            phase <= 0.75,
            lobe_radius * torch.sin(middle_angle),
            lobe_radius * torch.sin(final_angle),
        ),
    )
    local_y *= chirality[:, None]

    major_axis = planar_anchor / outer_radius.unsqueeze(-1)
    minor_axis = torch.stack((-major_axis[:, 1], major_axis[:, 0]), dim=-1)
    figure_eight_xy = local_x.unsqueeze(-1) * major_axis[:, None, :]
    figure_eight_xy += local_y.unsqueeze(-1) * minor_axis[:, None, :]
    height = anchor_e[:, 2:3] + vertical_amplitude[:, None] * torch.sin(math.tau * phase)[None, :]
    waypoints_e = torch.cat((figure_eight_xy, height.unsqueeze(-1)), dim=-1)

    quarter_crossing = torch.remainder(lap_sample, samples_per_lap) == samples_per_lap // 4
    three_quarter_crossing = torch.remainder(lap_sample, samples_per_lap) == 3 * samples_per_lap // 4
    crossing = quarter_crossing | three_quarter_crossing
    crossing_point = torch.zeros_like(waypoints_e)
    crossing_point[..., 2] = height
    waypoints_e = torch.where(crossing[None, :, None], crossing_point, waypoints_e)
    lap_closure = torch.remainder(sample_number, samples_per_lap) == 0
    return torch.where(lap_closure[None, :, None], anchor_e[:, None, :], waypoints_e)


def _sample_bounded_template_waypoints(
    anchor_e: torch.Tensor,
    num_waypoints: int,
    samples_per_lap: int,
    aspect_ratio_range: tuple[float, float],
    vertical_amplitude_range: tuple[float, float],
    figure_eight_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample an independent ellipse or figure-eight for every route.

    Returns:
        A tuple of waypoints [m], shape ``(N, W, 3)``, and a Boolean figure-eight
        mask, shape ``(N,)``.
    """
    ellipse = _sample_bounded_ellipse_waypoints(
        anchor_e,
        num_waypoints,
        samples_per_lap,
        aspect_ratio_range,
        vertical_amplitude_range,
    )
    figure_eight = _sample_bounded_figure_eight_waypoints(
        anchor_e,
        num_waypoints,
        samples_per_lap,
        vertical_amplitude_range,
    )
    is_figure_eight = (
        torch.rand(anchor_e.shape[0], device=anchor_e.device, dtype=anchor_e.dtype) < figure_eight_probability
    )
    return torch.where(is_figure_eight[:, None, None], figure_eight, ellipse), is_figure_eight


def _constant_curvature_fallback_mask(
    waypoint_offsets: torch.Tensor,
    minimum_separation: float,
    maximum_heading_change: float,
) -> torch.Tensor:
    """Identify the bounded-walk sampler's deterministic circular fallback.

    The fallback has constant minimum-length planar chords and a constant turn
    magnitude. Detecting it lets route mixtures replace the rare feasibility
    fallback with a genuinely different bounded family instead of exposing a
    circle-like route to the policy.

    Args:
        waypoint_offsets: Reset-relative waypoint offsets [m], shape ``(N, W, 3)``.
        minimum_separation: Configured fallback chord length [m].
        maximum_heading_change: Configured fallback turn magnitude [rad].

    Returns:
        Boolean fallback mask, shape ``(N,)``.
    """
    if waypoint_offsets.ndim != 3 or waypoint_offsets.shape[-1] != 3 or waypoint_offsets.shape[1] < 2:
        raise ValueError("waypoint_offsets must have shape (N, W, 3) with W >= 2.")
    anchor = torch.zeros(waypoint_offsets.shape[0], 1, 3, device=waypoint_offsets.device, dtype=waypoint_offsets.dtype)
    segment = torch.diff(torch.cat((anchor, waypoint_offsets), dim=1), dim=1)
    segment_length = torch.linalg.vector_norm(segment, dim=-1)
    heading = torch.atan2(segment[..., 1], segment[..., 0])
    heading_change = torch.remainder(torch.diff(heading, dim=1) + torch.pi, 2.0 * torch.pi) - torch.pi
    scale = max(1.0, abs(minimum_separation), abs(maximum_heading_change))
    tolerance = 512.0 * torch.finfo(waypoint_offsets.dtype).eps * scale
    constant_length = torch.all(torch.abs(segment_length - minimum_separation) <= tolerance, dim=1)
    constant_turn = torch.all(torch.abs(torch.abs(heading_change) - maximum_heading_change) <= tolerance, dim=1)
    planar = torch.all(torch.abs(segment[..., 2]) <= tolerance, dim=1)
    return constant_length & constant_turn & planar


def _hard_route_spline_sampling_margin(
    minimum_separation: float,
    maximum_separation: float,
    maximum_heading_change: float,
    maximum_vertical_step: float,
    spline_tangent_scale: float,
) -> tuple[float, float]:
    """Return planar and vertical knot margins that bound cubic overshoot [m].

    Cubic Hermite derivative bases have maximum magnitude ``4 / 27``. Both
    endpoint derivatives can point toward the same box face, so ``8 / 27``
    times their component bound is reserved inside that face. The vertical
    component bound additionally uses the configured step and planar-turn
    limits; this preserves useful height variation in a shallow route box.
    """
    deviation_factor = (8.0 / 27.0) * spline_tangent_scale
    planar_margin = deviation_factor * maximum_separation
    vertical_chord_ratio = min(maximum_vertical_step / minimum_separation, 1.0)
    planar_chord_ratio = math.sqrt(max(1.0 - vertical_chord_ratio**2, 0.0))
    half_turn_cosine = max(math.cos(0.5 * maximum_heading_change), torch.finfo(torch.float64).eps)
    vertical_tangent_ratio = vertical_chord_ratio / math.sqrt(
        (planar_chord_ratio * half_turn_cosine) ** 2 + vertical_chord_ratio**2
    )
    vertical_margin = deviation_factor * maximum_separation * vertical_tangent_ratio
    return planar_margin, vertical_margin


def _indexed_segment_projection(
    position_e: torch.Tensor,
    route_anchor_e: torch.Tensor,
    waypoints_e: torch.Tensor,
    active_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project positions onto their indexed active route segments.

    The active segment ends at ``waypoints_e[active_index]`` and begins at the
    route anchor for index zero or the preceding waypoint otherwise. Selecting
    the segment by progress index, rather than spatial proximity, prevents a
    self-intersection from switching the tracked branch.

    Args:
        position_e: Robot positions in environment coordinates [m], shape ``(N, 3)``.
        route_anchor_e: Route start positions [m], shape ``(N, 3)``.
        waypoints_e: Route waypoints [m], shape ``(N, W, 3)``.
        active_index: Active waypoint indices, shape ``(N,)``.

    Returns:
        A tuple containing projected points [m] and error vectors from the
        projection to the robot [m], each with shape ``(N, 3)``.
    """
    env_ids = torch.arange(position_e.shape[0], device=position_e.device)
    segment_end = waypoints_e[env_ids, active_index]
    previous_index = torch.clamp(active_index - 1, min=0)
    previous_waypoint = waypoints_e[env_ids, previous_index]
    segment_start = torch.where((active_index == 0).unsqueeze(-1), route_anchor_e, previous_waypoint)
    segment = segment_end - segment_start
    segment_length_sq = torch.sum(torch.square(segment), dim=-1)
    projection_fraction = torch.sum((position_e - segment_start) * segment, dim=-1)
    projection_fraction /= segment_length_sq.clamp_min(torch.finfo(position_e.dtype).eps)
    projection_fraction = projection_fraction.clamp(0.0, 1.0)
    projection = segment_start + projection_fraction.unsqueeze(-1) * segment
    return projection, position_e - projection


_GAUSS_LEGENDRE_NODES = (-0.906179845938664, -0.538469310105683, 0.0, 0.538469310105683, 0.906179845938664)
_GAUSS_LEGENDRE_WEIGHTS = (
    0.236926885056189,
    0.478628670499366,
    0.568888888888889,
    0.478628670499366,
    0.236926885056189,
)


def _normalize_path_vector(vector: torch.Tensor, fallback: torch.Tensor | None = None) -> torch.Tensor:
    """Return finite unit vectors, using ``fallback`` for degenerate inputs."""
    finite_vector = torch.nan_to_num(vector)
    norm = torch.linalg.vector_norm(finite_vector, dim=-1, keepdim=True)
    normalized = finite_vector / norm.clamp_min(torch.finfo(finite_vector.dtype).eps)
    if fallback is None:
        return torch.where(norm > torch.finfo(finite_vector.dtype).eps, normalized, torch.zeros_like(normalized))
    finite_fallback = torch.nan_to_num(fallback)
    fallback_norm = torch.linalg.vector_norm(finite_fallback, dim=-1, keepdim=True)
    fallback_unit = finite_fallback / fallback_norm.clamp_min(torch.finfo(finite_vector.dtype).eps)
    fallback_unit = torch.where(
        fallback_norm > torch.finfo(finite_vector.dtype).eps, fallback_unit, torch.zeros_like(fallback_unit)
    )
    return torch.where(norm > torch.finfo(finite_vector.dtype).eps, normalized, fallback_unit)


def _path_knot_tangents(route_points_e: torch.Tensor) -> torch.Tensor:
    """Construct shared geometric tangents for a waypoint-interpolating path.

    Interior tangents bisect the incoming and outgoing unit chord directions.
    Sharing one direction at each knot makes the cubic path geometrically
    tangent-continuous while remaining robust to unequal segment lengths.

    Args:
        route_points_e: Route anchor followed by waypoints [m], shape ``(..., P, 3)``.

    Returns:
        Unit knot tangent directions, shape ``(..., P, 3)``.
    """
    if route_points_e.ndim < 2 or route_points_e.shape[-1] != 3 or route_points_e.shape[-2] < 2:
        raise ValueError("route_points_e must have shape (..., P, 3) with P >= 2.")
    chord = torch.diff(route_points_e, dim=-2)
    chord_direction = _normalize_path_vector(chord)
    tangent = torch.zeros_like(route_points_e)
    tangent[..., 0, :] = chord_direction[..., 0, :]
    tangent[..., -1, :] = chord_direction[..., -1, :]
    if route_points_e.shape[-2] > 2:
        tangent[..., 1:-1, :] = _normalize_path_vector(
            chord_direction[..., :-1, :] + chord_direction[..., 1:, :],
            fallback=chord_direction[..., 1:, :],
        )
    return tangent


def _bounded_ellipse_path_tangents(
    anchor_e: torch.Tensor,
    waypoints_e: torch.Tensor,
    samples_per_lap: int,
) -> torch.Tensor:
    """Return analytic periodic tangents for sampled bounded ellipses.

    The sampler's aspect ratio, chirality, and vertical amplitude are recovered
    from its first non-degenerate sample. This keeps the waypoint sampler's
    tensor-only public return value while avoiding endpoint and lap-closure
    chord biases in the interpolating spline.
    """
    if anchor_e.ndim != 2 or anchor_e.shape[1] != 3:
        raise ValueError(f"anchor_e must have shape (N, 3), got {tuple(anchor_e.shape)}.")
    if waypoints_e.ndim != 3 or waypoints_e.shape[0] != anchor_e.shape[0] or waypoints_e.shape[2] != 3:
        raise ValueError("waypoints_e must have shape (N, W, 3) with the same route count as anchor_e.")
    if samples_per_lap < 4 or waypoints_e.shape[1] % samples_per_lap != 0:
        raise ValueError("samples_per_lap must be at least four and divide the waypoint count.")

    planar_anchor = anchor_e[:, :2]
    major_radius = torch.linalg.vector_norm(planar_anchor, dim=-1)
    major_axis = planar_anchor / major_radius.clamp_min(torch.finfo(anchor_e.dtype).eps).unsqueeze(-1)
    minor_axis = torch.stack((-major_axis[:, 1], major_axis[:, 0]), dim=-1)
    first_minor_coordinate = torch.sum(waypoints_e[:, 0, :2] * minor_axis, dim=-1)
    chirality = torch.where(first_minor_coordinate < 0.0, -torch.ones_like(major_radius), torch.ones_like(major_radius))

    phase_step = math.tau / samples_per_lap
    aspect_ratio = torch.abs(first_minor_coordinate)
    aspect_ratio /= (major_radius * math.sin(phase_step)).clamp_min(torch.finfo(anchor_e.dtype).eps)
    vertical_sample_scale = math.sin(2.0 * phase_step)
    if abs(vertical_sample_scale) <= 1.0e-6:
        # Four samples per lap alias the twice-per-lap vertical sinusoid to
        # zero, so its amplitude cannot be recovered from the waypoint tensor.
        signed_vertical_amplitude = torch.zeros_like(major_radius)
    else:
        signed_vertical_amplitude = (waypoints_e[:, 0, 2] - anchor_e[:, 2]) / vertical_sample_scale

    knot_index = torch.arange(waypoints_e.shape[1] + 1, device=anchor_e.device, dtype=anchor_e.dtype)
    phase = chirality.unsqueeze(1) * phase_step * knot_index.unsqueeze(0)
    planar_derivative = chirality[:, None, None] * (
        -major_radius[:, None, None] * torch.sin(phase).unsqueeze(-1) * major_axis[:, None, :]
        + major_radius[:, None, None]
        * aspect_ratio[:, None, None]
        * torch.cos(phase).unsqueeze(-1)
        * minor_axis[:, None, :]
    )
    vertical_derivative = 2.0 * signed_vertical_amplitude[:, None] * torch.cos(2.0 * phase)
    return _normalize_path_vector(torch.cat((planar_derivative, vertical_derivative.unsqueeze(-1)), dim=-1))


def _bounded_figure_eight_path_tangents(
    anchor_e: torch.Tensor,
    waypoints_e: torch.Tensor,
    samples_per_lap: int,
) -> torch.Tensor:
    """Return analytic periodic tangents for sampled tangent-lobe figure-eights."""
    if anchor_e.ndim != 2 or anchor_e.shape[1] != 3:
        raise ValueError(f"anchor_e must have shape (N, 3), got {tuple(anchor_e.shape)}.")
    if waypoints_e.ndim != 3 or waypoints_e.shape[0] != anchor_e.shape[0] or waypoints_e.shape[2] != 3:
        raise ValueError("waypoints_e must have shape (N, W, 3) with the same route count as anchor_e.")
    if samples_per_lap < 4 or samples_per_lap % 4 != 0 or waypoints_e.shape[1] % samples_per_lap != 0:
        raise ValueError("samples_per_lap must be divisible by four and divide the waypoint count.")

    planar_anchor = anchor_e[:, :2]
    outer_radius = torch.linalg.vector_norm(planar_anchor, dim=-1)
    major_axis = planar_anchor / outer_radius.clamp_min(torch.finfo(anchor_e.dtype).eps).unsqueeze(-1)
    minor_axis = torch.stack((-major_axis[:, 1], major_axis[:, 0]), dim=-1)
    first_minor_coordinate = torch.sum(waypoints_e[:, 0, :2] * minor_axis, dim=-1)
    chirality = torch.where(first_minor_coordinate < 0.0, -torch.ones_like(outer_radius), torch.ones_like(outer_radius))
    phase_step = math.tau / samples_per_lap
    vertical_amplitude = (waypoints_e[:, 0, 2] - anchor_e[:, 2]) / math.sin(phase_step)

    knot_index = torch.arange(waypoints_e.shape[1] + 1, device=anchor_e.device)
    phase = torch.remainder(knot_index, samples_per_lap).to(anchor_e.dtype) / float(samples_per_lap)
    lobe_radius = 0.5 * outer_radius[:, None]
    first_angle = 4.0 * math.pi * phase
    middle_angle = -4.0 * math.pi * (phase - 0.25)
    final_angle = math.pi + 4.0 * math.pi * (phase - 0.75)
    local_dx = torch.where(
        phase <= 0.25,
        -4.0 * math.pi * lobe_radius * torch.sin(first_angle),
        torch.where(
            phase <= 0.75,
            4.0 * math.pi * lobe_radius * torch.sin(middle_angle),
            -4.0 * math.pi * lobe_radius * torch.sin(final_angle),
        ),
    )
    local_dy = torch.where(
        phase <= 0.25,
        4.0 * math.pi * lobe_radius * torch.cos(first_angle),
        torch.where(
            phase <= 0.75,
            -4.0 * math.pi * lobe_radius * torch.cos(middle_angle),
            4.0 * math.pi * lobe_radius * torch.cos(final_angle),
        ),
    )
    local_dy *= chirality[:, None]
    planar_derivative = local_dx.unsqueeze(-1) * major_axis[:, None, :]
    planar_derivative += local_dy.unsqueeze(-1) * minor_axis[:, None, :]
    vertical_derivative = 2.0 * math.pi * vertical_amplitude[:, None] * torch.cos(math.tau * phase)[None, :]
    return _normalize_path_vector(torch.cat((planar_derivative, vertical_derivative.unsqueeze(-1)), dim=-1))


def _bounded_template_path_tangents(
    anchor_e: torch.Tensor,
    waypoints_e: torch.Tensor,
    samples_per_lap: int,
    is_figure_eight: torch.Tensor,
) -> torch.Tensor:
    """Select analytic periodic ellipse or figure-eight tangents per route."""
    if is_figure_eight.shape != (anchor_e.shape[0],):
        raise ValueError(f"is_figure_eight must have shape ({anchor_e.shape[0]},).")
    ellipse = _bounded_ellipse_path_tangents(anchor_e, waypoints_e, samples_per_lap)
    figure_eight = _bounded_figure_eight_path_tangents(anchor_e, waypoints_e, samples_per_lap)
    return torch.where(is_figure_eight[:, None, None], figure_eight, ellipse)


def _cubic_hermite_path(
    start: torch.Tensor,
    end: torch.Tensor,
    start_derivative: torch.Tensor,
    end_derivative: torch.Tensor,
    parameter: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate a cubic Hermite path and its first two parameter derivatives."""
    extra_dimensions = parameter.ndim - (start.ndim - 1)
    if extra_dimensions < 0:
        raise ValueError("parameter must contain every leading path-control dimension.")
    for _ in range(extra_dimensions):
        start = start.unsqueeze(-2)
        end = end.unsqueeze(-2)
        start_derivative = start_derivative.unsqueeze(-2)
        end_derivative = end_derivative.unsqueeze(-2)
    u = parameter.unsqueeze(-1)
    coefficient_a = 2.0 * start - 2.0 * end + start_derivative + end_derivative
    coefficient_b = -3.0 * start + 3.0 * end - 2.0 * start_derivative - end_derivative
    coefficient_c = start_derivative
    position = ((coefficient_a * u + coefficient_b) * u + coefficient_c) * u + start
    first_derivative = (3.0 * coefficient_a * u + 2.0 * coefficient_b) * u + coefficient_c
    second_derivative = 6.0 * coefficient_a * u + 2.0 * coefficient_b
    return position, first_derivative, second_derivative


def _cubic_path_arc_length(
    start: torch.Tensor,
    end: torch.Tensor,
    start_derivative: torch.Tensor,
    end_derivative: torch.Tensor,
    parameter: torch.Tensor,
) -> torch.Tensor:
    """Approximate cubic arc length from zero to ``parameter`` with five-point quadrature [m]."""
    finite_parameter = torch.nan_to_num(parameter).clamp(0.0, 1.0)
    nodes = torch.as_tensor(_GAUSS_LEGENDRE_NODES, device=start.device, dtype=start.dtype)
    weights = torch.as_tensor(_GAUSS_LEGENDRE_WEIGHTS, device=start.device, dtype=start.dtype)
    quadrature_parameter = 0.5 * finite_parameter.unsqueeze(-1) * (nodes + 1.0)
    _, derivative, _ = _cubic_hermite_path(start, end, start_derivative, end_derivative, quadrature_parameter)
    speed = torch.linalg.vector_norm(torch.nan_to_num(derivative), dim=-1)
    return torch.nan_to_num(0.5 * finite_parameter * torch.sum(speed * weights, dim=-1)).clamp_min_(0.0)


def _project_cubic_path_segment(
    position_e: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
    start_derivative: torch.Tensor,
    end_derivative: torch.Tensor,
    coarse_samples: int,
    newton_iterations: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project positions onto one indexed cubic segment without switching route branches.

    A uniform coarse search selects a basin, followed by bounded Newton refinement
    of the squared-distance stationary condition. All operations are batched and
    stay on device.

    Returns:
        Projection [m], projection-to-position error [m], unit tangent, curvature
        binormal [1/m], and segment parameter, with leading shape ``position_e.shape[:-1]``.
    """
    if coarse_samples < 2:
        raise ValueError("coarse_samples must be at least two.")
    sample_parameter = torch.linspace(0.0, 1.0, coarse_samples + 1, device=position_e.device, dtype=position_e.dtype)
    sample_parameter = sample_parameter.expand(*position_e.shape[:-1], -1)
    sample_position, _, _ = _cubic_hermite_path(start, end, start_derivative, end_derivative, sample_parameter)
    finite_position = torch.nan_to_num(position_e)
    distance_sq = torch.sum(torch.square(sample_position - finite_position.unsqueeze(-2)), dim=-1)
    closest_sample = torch.argmin(torch.nan_to_num(distance_sq, nan=torch.inf), dim=-1)
    parameter = closest_sample.to(position_e.dtype) / float(coarse_samples)

    for _ in range(newton_iterations):
        curve_position, derivative, second_derivative = _cubic_hermite_path(
            start, end, start_derivative, end_derivative, parameter
        )
        residual = curve_position - finite_position
        objective_derivative = torch.sum(residual * derivative, dim=-1)
        objective_second_derivative = torch.sum(derivative.square() + residual * second_derivative, dim=-1)
        valid_denominator = torch.abs(objective_second_derivative) > torch.finfo(position_e.dtype).eps
        update = torch.where(
            valid_denominator,
            objective_derivative / objective_second_derivative,
            torch.zeros_like(objective_derivative),
        )
        parameter = torch.nan_to_num(parameter - update).clamp_(0.0, 1.0)

    projection, derivative, second_derivative = _cubic_hermite_path(
        start, end, start_derivative, end_derivative, parameter
    )
    tangent = _normalize_path_vector(derivative)
    speed = torch.linalg.vector_norm(torch.nan_to_num(derivative), dim=-1)
    curvature = torch.cross(derivative, second_derivative, dim=-1)
    curvature /= speed.pow(3).clamp_min(torch.finfo(position_e.dtype).eps).unsqueeze(-1)
    finite = torch.isfinite(position_e).all(dim=-1)
    projection = torch.where(finite.unsqueeze(-1), torch.nan_to_num(projection), torch.zeros_like(projection))
    error = torch.where(finite.unsqueeze(-1), finite_position - projection, torch.zeros_like(projection))
    tangent = torch.where(finite.unsqueeze(-1), torch.nan_to_num(tangent), torch.zeros_like(tangent))
    curvature = torch.where(finite.unsqueeze(-1), torch.nan_to_num(curvature), torch.zeros_like(curvature))
    parameter = torch.where(finite, parameter, torch.zeros_like(parameter))
    return projection, error, tangent, curvature, parameter


class _EpisodeMetricsCommand(CommandTerm):
    """Episode instrumentation shared by waypoint-like command terms."""

    def _initialize_episode_metrics(self) -> None:
        self._episode_metrics = EpisodeMetricAccumulator(self.num_envs, self.device)
        self.metrics.update(self._episode_metrics.metrics)
        self.last_episode_metrics = {
            name: torch.zeros_like(value) for name, value in self._episode_metrics.metrics.items()
        }

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> dict[str, float]:
        """Snapshot completed-episode metrics, then reset the selected commands."""
        selected = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        for name, value in self.metrics.items():
            self.last_episode_metrics[name][selected] = value[selected]
        extras = super().reset(selected)
        self._episode_metrics.reset(selected)
        return extras

    def record_metrics_step(self) -> None:
        """Record one post-physics sample before any terminated environment resets."""
        self._record_current_metrics()

    def _update_metrics(self) -> None:
        # Metrics are sampled explicitly from the reward phase, before autoreset.
        return

    def _record_episode_metrics(
        self,
        position_error: torch.Tensor,
        *,
        waypoint_fraction: torch.Tensor,
        waypoint_completed: torch.Tensor,
    ) -> None:
        record_slung_load_metrics = getattr(self.cfg, "record_slung_load_metrics", True)
        if record_slung_load_metrics:
            swing_angle = total_swing_angle(self._env)
            transverse_speed = torch.linalg.norm(payload_transverse_velocity_b(self._env), dim=-1)
            payload_speed = torch.linalg.norm(link_lin_vel_w(self._env, SceneEntityCfg("payload")), dim=-1)
            relative_separation = cable_relative_separation(self._env)
            joint_error = cable_joint_error(self._env)
        else:
            # Keep one stable episode schema for slung-load and rigid-only tasks.
            # The separate capability metric prevents these placeholders from
            # being interpreted as measured zero swing or perfect cable health.
            zeros = torch.zeros_like(position_error)
            swing_angle = zeros
            transverse_speed = zeros
            payload_speed = zeros
            relative_separation = zeros
            joint_error = zeros
        self._episode_metrics.update(
            position_error=position_error,
            swing_angle=swing_angle,
            transverse_speed=transverse_speed,
            drone_speed=torch.linalg.norm(link_lin_vel_w(self._env), dim=-1),
            payload_speed=payload_speed,
            cable_relative_separation=relative_separation,
            cable_joint_error=joint_error,
            waypoint_fraction=waypoint_fraction,
            waypoint_completed=waypoint_completed,
            elapsed_time=self._env.episode_length_buf.float() * self._env.step_dt,
        )

    def _record_current_metrics(self) -> None:
        raise NotImplementedError


class WaypointSequenceCommand(_EpisodeMetricsCommand):
    """Track a reset-relative sequence and expose its current and next waypoint.

    Configured offsets are expressed in the drone's reset-yaw frame. At reset,
    they are yaw-rotated, translated from the reset position, and stored in the
    environment frame. :attr:`command` concatenates the active waypoint and its
    successor, each in the environment frame [m]. At the final waypoint, both
    halves of the command contain that final waypoint.
    """

    cfg: WaypointSequenceCommandCfg

    def __init__(self, cfg: WaypointSequenceCommandCfg, env: ManagerBasedRLEnv):
        self._validate_cfg(cfg)
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        num_waypoints = cfg.random_waypoint_count if cfg.randomize_waypoints else len(cfg.waypoint_offsets)
        self.route_anchor_e = torch.zeros(self.num_envs, 3, device=self.device)
        self.waypoints_e = torch.zeros(self.num_envs, num_waypoints, 3, device=self.device)
        self.route_is_figure_eight = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.route_family_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.current_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.previous_distance_sq = torch.zeros(self.num_envs, device=self.device)
        if cfg.spline_enabled:
            self._initialize_spline_path()
        self._initialize_episode_metrics()

    @staticmethod
    def _validate_cfg(cfg: WaypointSequenceCommandCfg) -> None:  # noqa: C901
        if not isinstance(getattr(cfg, "record_slung_load_metrics", True), bool):
            raise ValueError("record_slung_load_metrics must be a boolean.")
        if len(cfg.waypoint_offsets) < 2:
            raise ValueError("waypoint_offsets must contain at least two 3D offsets.")
        if any(len(offset) != 3 for offset in cfg.waypoint_offsets):
            raise ValueError("Each waypoint offset must contain exactly three values.")
        if not torch.isfinite(torch.as_tensor(cfg.waypoint_offsets, dtype=torch.float64)).all():
            raise ValueError("waypoint_offsets must contain only finite values.")
        if not torch.isfinite(torch.tensor(cfg.acceptance_radius)) or cfg.acceptance_radius <= 0.0:
            raise ValueError("acceptance_radius must be positive and finite.")
        spline_enabled = getattr(cfg, "spline_enabled", False)
        spline_projection_samples = getattr(cfg, "spline_projection_samples", 12)
        spline_tangent_scale = getattr(cfg, "spline_tangent_scale", 0.75)
        if not isinstance(spline_enabled, bool):
            raise ValueError("spline_enabled must be a boolean.")
        if not isinstance(spline_projection_samples, int) or spline_projection_samples < 2:
            raise ValueError("spline_projection_samples must be at least two.")
        if not math.isfinite(spline_tangent_scale) or not 0.0 < spline_tangent_scale <= 1.0:
            raise ValueError("spline_tangent_scale must be finite and lie in (0, 1].")
        progressive_advancement = getattr(cfg, "spline_progressive_advancement", False)
        if not isinstance(progressive_advancement, bool):
            raise ValueError("spline_progressive_advancement must be a boolean.")
        if progressive_advancement and not spline_enabled:
            raise ValueError("spline_progressive_advancement requires spline_enabled=True.")
        crossing_tolerance = getattr(cfg, "spline_plane_crossing_lateral_tolerance", 0.50)
        if not math.isfinite(crossing_tolerance) or crossing_tolerance <= 0.0:
            raise ValueError("spline_plane_crossing_lateral_tolerance must be positive and finite.")
        maximum_advances = getattr(cfg, "spline_max_waypoint_advances_per_step", 4)
        if isinstance(maximum_advances, bool) or not isinstance(maximum_advances, int) or maximum_advances < 1:
            raise ValueError("spline_max_waypoint_advances_per_step must be a positive integer.")
        target_cruise_speed = getattr(cfg, "target_cruise_speed", 0.0)
        maximum_lateral_acceleration = getattr(cfg, "maximum_lateral_acceleration", 0.0)
        maximum_braking_acceleration = getattr(cfg, "maximum_braking_acceleration", 0.0)
        speed_lookahead_distances = getattr(cfg, "speed_lookahead_distances", (0.0, 0.75, 1.50))
        speed_values = {
            "target_cruise_speed": target_cruise_speed,
            "maximum_lateral_acceleration": maximum_lateral_acceleration,
            "maximum_braking_acceleration": maximum_braking_acceleration,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0.0
            for value in speed_values.values()
        ):
            raise ValueError("Path speed-reference values must be finite nonnegative scalars.")
        if target_cruise_speed > 0.0 and (maximum_lateral_acceleration <= 0.0 or maximum_braking_acceleration <= 0.0):
            raise ValueError(
                "Positive target_cruise_speed requires positive maximum_lateral_acceleration and "
                "maximum_braking_acceleration."
            )
        if (
            not isinstance(speed_lookahead_distances, Sequence)
            or not speed_lookahead_distances
            or any(
                isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0.0
                for value in speed_lookahead_distances
            )
        ):
            raise ValueError("speed_lookahead_distances must contain finite nonnegative values.")
        if getattr(cfg, "regenerate_on_completion", False) and not getattr(cfg, "randomize_waypoints", False):
            raise ValueError("regenerate_on_completion requires randomize_waypoints=True.")
        route_family = getattr(cfg, "route_family", "random_walk")
        route_families = {"random_walk", "bounded_ellipse", "bounded_template_mix", "bounded_hard_mix"}
        if route_family not in route_families:
            raise ValueError(
                "route_family must be 'random_walk', 'bounded_ellipse', 'bounded_template_mix', or 'bounded_hard_mix'."
            )
        if route_family != "random_walk" and not getattr(cfg, "randomize_waypoints", False):
            raise ValueError(f"route_family='{route_family}' requires randomize_waypoints=True.")
        if getattr(cfg, "randomize_waypoints", False):
            if cfg.random_waypoint_count < 2:
                raise ValueError("random_waypoint_count must be at least two.")
            if route_family in {"bounded_ellipse", "bounded_template_mix", "bounded_hard_mix"}:
                WaypointSequenceCommand._validate_bounded_ellipse_cfg(cfg)
                if route_family in {"bounded_template_mix", "bounded_hard_mix"}:
                    if cfg.samples_per_lap % 4 != 0:
                        raise ValueError(
                            "samples_per_lap must be divisible by four for bounded_template_mix or bounded_hard_mix."
                        )
                    probability = torch.as_tensor(cfg.figure_eight_probability, dtype=torch.float64)
                    if (
                        isinstance(cfg.figure_eight_probability, bool)
                        or probability.ndim != 0
                        or not torch.isfinite(probability)
                        or probability < 0.0
                        or probability > 1.0
                    ):
                        raise ValueError("figure_eight_probability must be finite and lie in [0, 1].")
                if route_family != "bounded_hard_mix":
                    return
            if cfg.random_sampling_attempts < 1:
                raise ValueError("random_sampling_attempts must be positive.")
            if cfg.route_sampling_attempts < 1:
                raise ValueError("route_sampling_attempts must be positive.")
            if (
                isinstance(cfg.random_heading_change_interval, bool)
                or not isinstance(cfg.random_heading_change_interval, int)
                or cfg.random_heading_change_interval < 1
            ):
                raise ValueError("random_heading_change_interval must be a positive integer.")
            ranges = cfg.random_waypoint_ranges
            bounds = torch.tensor((ranges.pos_x, ranges.pos_y, ranges.pos_z), dtype=torch.float64)
            if bounds.shape != (3, 2) or not torch.isfinite(bounds).all():
                raise ValueError("Random waypoint XYZ ranges must contain two finite bounds per axis.")
            if torch.any(bounds[:, 0] > bounds[:, 1]):
                raise ValueError("Each random waypoint lower bound must not exceed its upper bound.")
            if not torch.isfinite(torch.tensor(cfg.minimum_waypoint_separation)):
                raise ValueError("minimum_waypoint_separation must be finite.")
            if cfg.minimum_waypoint_separation < 0.0:
                raise ValueError("minimum_waypoint_separation must be non-negative.")
            half_diagonal = 0.5 * torch.linalg.vector_norm(bounds[:, 1] - bounds[:, 0])
            if cfg.minimum_waypoint_separation > half_diagonal:
                raise ValueError(
                    "minimum_waypoint_separation cannot exceed half the random waypoint box diagonal; "
                    "otherwise consecutive sampling is not guaranteed to be feasible."
                )
            if cfg.maximum_waypoint_separation is not None:
                maximum_separation = torch.tensor(cfg.maximum_waypoint_separation, dtype=torch.float64)
                if not torch.isfinite(maximum_separation) or maximum_separation <= 0.0:
                    raise ValueError("maximum_waypoint_separation must be positive and finite when provided.")
                if cfg.maximum_waypoint_separation < cfg.minimum_waypoint_separation:
                    raise ValueError(
                        "maximum_waypoint_separation must be greater than or equal to minimum_waypoint_separation."
                    )
                closest_to_anchor = torch.maximum(torch.minimum(torch.zeros(3), bounds[:, 1]), bounds[:, 0])
                if torch.linalg.vector_norm(closest_to_anchor) > cfg.maximum_waypoint_separation:
                    raise ValueError(
                        "maximum_waypoint_separation is too small to reach the random waypoint box from the anchor."
                    )
            if route_family == "bounded_hard_mix":
                if cfg.maximum_heading_change is None or cfg.maximum_heading_change >= math.pi:
                    raise ValueError("bounded_hard_mix requires maximum_heading_change strictly between zero and pi.")
                if cfg.maximum_vertical_step is None:
                    raise ValueError("bounded_hard_mix requires maximum_vertical_step.")
            WaypointSequenceCommand._validate_smooth_route_cfg(cfg, bounds)
            if route_family == "bounded_hard_mix" and spline_enabled:
                assert cfg.maximum_waypoint_separation is not None
                assert cfg.maximum_heading_change is not None
                assert cfg.maximum_vertical_step is not None
                planar_margin, vertical_margin = _hard_route_spline_sampling_margin(
                    cfg.minimum_waypoint_separation,
                    cfg.maximum_waypoint_separation,
                    cfg.maximum_heading_change,
                    cfg.maximum_vertical_step,
                    cfg.spline_tangent_scale,
                )
                required_margin = torch.tensor((planar_margin, planar_margin, vertical_margin), dtype=torch.float64)
                if torch.any(2.0 * required_margin >= bounds[:, 1] - bounds[:, 0]):
                    raise ValueError("bounded_hard_mix spline margins must fit inside the waypoint bounds.")

    @staticmethod
    def _validate_bounded_ellipse_cfg(cfg: WaypointSequenceCommandCfg) -> None:
        """Validate the opt-in, origin-centered periodic route family."""
        if isinstance(cfg.samples_per_lap, bool) or not isinstance(cfg.samples_per_lap, int):
            raise ValueError("samples_per_lap must be an integer.")
        if cfg.samples_per_lap < 4:
            raise ValueError("samples_per_lap must be at least four.")
        if cfg.random_waypoint_count % cfg.samples_per_lap != 0:
            raise ValueError("random_waypoint_count must be a multiple of samples_per_lap.")

        aspect_ratio = torch.as_tensor(cfg.aspect_ratio_range, dtype=torch.float64)
        if (
            aspect_ratio.shape != (2,)
            or not torch.isfinite(aspect_ratio).all()
            or aspect_ratio[0] <= 0.0
            or aspect_ratio[0] > aspect_ratio[1]
            or aspect_ratio[1] > 1.0
        ):
            raise ValueError("aspect_ratio_range must be finite, ordered, and lie in (0, 1].")
        vertical_amplitude = torch.as_tensor(cfg.vertical_amplitude_range, dtype=torch.float64)
        if (
            vertical_amplitude.shape != (2,)
            or not torch.isfinite(vertical_amplitude).all()
            or vertical_amplitude[0] < 0.0
            or vertical_amplitude[0] > vertical_amplitude[1]
        ):
            raise ValueError("vertical_amplitude_range must contain ordered finite nonnegative values.")

    @staticmethod
    def _validate_smooth_route_cfg(cfg: WaypointSequenceCommandCfg, bounds: torch.Tensor) -> None:
        """Validate options specific to heading-correlated routes."""
        if cfg.maximum_heading_change is not None:
            heading_change = torch.tensor(cfg.maximum_heading_change, dtype=torch.float64)
            if not torch.isfinite(heading_change) or not 0.0 < cfg.maximum_heading_change <= torch.pi:
                raise ValueError("maximum_heading_change must be in the interval (0, pi].")
            if cfg.maximum_waypoint_separation is None:
                raise ValueError("maximum_heading_change requires maximum_waypoint_separation.")
            if cfg.minimum_waypoint_separation <= 0.0:
                raise ValueError("maximum_heading_change requires a positive minimum_waypoint_separation.")
            if torch.any(bounds[:, 0] > 0.0) or torch.any(bounds[:, 1] < 0.0):
                raise ValueError("Heading-correlated random waypoint ranges must contain the route anchor.")
            fallback_excursion = cfg.minimum_waypoint_separation / math.sin(0.5 * cfg.maximum_heading_change)
            if torch.any(-bounds[:2, 0] < fallback_excursion) or torch.any(bounds[:2, 1] < fallback_excursion):
                raise ValueError(
                    "Heading-correlated XY ranges are too small for the constant-curvature route fallback."
                )
            if cfg.nominal_heading_change is not None:
                nominal_heading_change = torch.tensor(cfg.nominal_heading_change, dtype=torch.float64)
                if not torch.isfinite(nominal_heading_change) or cfg.nominal_heading_change < 0.0:
                    raise ValueError("nominal_heading_change must be non-negative and finite when provided.")
                if cfg.nominal_heading_change > cfg.maximum_heading_change:
                    raise ValueError("nominal_heading_change cannot exceed maximum_heading_change.")
        if cfg.maximum_vertical_step is not None:
            vertical_step = torch.tensor(cfg.maximum_vertical_step, dtype=torch.float64)
            if not torch.isfinite(vertical_step) or cfg.maximum_vertical_step < 0.0:
                raise ValueError("maximum_vertical_step must be non-negative and finite when provided.")
            if cfg.maximum_heading_change is None:
                raise ValueError("maximum_vertical_step requires maximum_heading_change.")
            assert cfg.maximum_waypoint_separation is not None
            if cfg.maximum_vertical_step > cfg.maximum_waypoint_separation:
                raise ValueError("maximum_vertical_step cannot exceed maximum_waypoint_separation.")
            half_span = 0.5 * (bounds[:, 1] - bounds[:, 0])
            effective_half_diagonal = torch.linalg.vector_norm(
                torch.stack((half_span[0], half_span[1], torch.minimum(half_span[2], vertical_step)))
            )
            if cfg.minimum_waypoint_separation > effective_half_diagonal:
                raise ValueError(
                    "minimum_waypoint_separation exceeds the box half-diagonal allowed by maximum_vertical_step."
                )

    def _initialize_episode_metrics(self) -> None:
        super()._initialize_episode_metrics()
        if not hasattr(self, "route_is_figure_eight"):
            self.route_is_figure_eight = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if not hasattr(self, "route_family_id"):
            self.route_family_id = self.route_is_figure_eight.long().clone()
        self._cross_track_count = torch.zeros(self.num_envs, device=self.device)
        self._cross_track_sum = torch.zeros(self.num_envs, device=self.device)
        self._cross_track_sq_sum = torch.zeros(self.num_envs, device=self.device)
        self._last_arrival_time = torch.zeros(self.num_envs, device=self.device)
        self._arrival_interval_sum = torch.zeros(self.num_envs, device=self.device)
        self._arrival_interval_min = torch.full((self.num_envs,), torch.inf, device=self.device)
        self._precision_miss_count = torch.zeros(self.num_envs, device=self.device)
        self._precision_miss_distance_sum = torch.zeros(self.num_envs, device=self.device)
        self._route_arc_length_traversed = torch.zeros(self.num_envs, device=self.device)
        route_metrics = {
            "route_family_id": self.route_family_id.float().clone(),
            "slung_load_metrics_available": torch.full(
                (self.num_envs,),
                float(getattr(getattr(self, "cfg", None), "record_slung_load_metrics", True)),
                device=self.device,
            ),
            "cross_track_error_mean": torch.zeros(self.num_envs, device=self.device),
            "cross_track_error_rms": torch.zeros(self.num_envs, device=self.device),
            "cross_track_error_max": torch.zeros(self.num_envs, device=self.device),
            "route_waypoints_passed": torch.zeros(self.num_envs, device=self.device),
            "route_traversal_fraction": torch.zeros(self.num_envs, device=self.device),
            "route_arc_length_traversed": torch.zeros(self.num_envs, device=self.device),
            "route_arc_length_traversal_rate": torch.zeros(self.num_envs, device=self.device),
            "waypoint_arrivals": torch.zeros(self.num_envs, device=self.device),
            "waypoint_precision_hits": torch.zeros(self.num_envs, device=self.device),
            "waypoint_precision_hit_fraction": torch.zeros(self.num_envs, device=self.device),
            "waypoint_precision_misses": torch.zeros(self.num_envs, device=self.device),
            "waypoint_precision_miss_distance_mean": torch.zeros(self.num_envs, device=self.device),
            "waypoint_precision_miss_distance_max": torch.zeros(self.num_envs, device=self.device),
            "waypoint_arrival_time_mean": torch.zeros(self.num_envs, device=self.device),
            "waypoint_arrival_time_min": torch.zeros(self.num_envs, device=self.device),
            "waypoint_arrival_time_max": torch.zeros(self.num_envs, device=self.device),
            "waypoint_throughput": torch.zeros(self.num_envs, device=self.device),
            "route_completions": torch.zeros(self.num_envs, device=self.device),
            "target_distance_completed": torch.zeros(self.num_envs, device=self.device),
            "episode_duration": torch.zeros(self.num_envs, device=self.device),
        }
        self.metrics.update(route_metrics)
        self.last_episode_metrics.update({name: torch.zeros_like(value) for name, value in route_metrics.items()})

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> dict[str, float]:
        """Reset route generation and all trajectory metrics for selected environments."""
        selected = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        extras = super().reset(selected)
        self._cross_track_count[selected] = 0.0
        self._cross_track_sum[selected] = 0.0
        self._cross_track_sq_sum[selected] = 0.0
        self._last_arrival_time[selected] = 0.0
        self._arrival_interval_sum[selected] = 0.0
        self._arrival_interval_min[selected] = torch.inf
        self._precision_miss_count[selected] = 0.0
        self._precision_miss_distance_sum[selected] = 0.0
        self._route_arc_length_traversed[selected] = 0.0
        self.metrics["cross_track_error_mean"][selected] = 0.0
        self.metrics["cross_track_error_rms"][selected] = 0.0
        self.metrics["cross_track_error_max"][selected] = 0.0
        self.metrics["route_arc_length_traversed"][selected] = 0.0
        self.metrics["route_arc_length_traversal_rate"][selected] = 0.0
        return extras

    @property
    def command(self) -> torch.Tensor:
        """Active and following environment-frame waypoints [m], shape ``(N, 6)``."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        following_index = torch.clamp(self.current_index + 1, max=self.waypoints_e.shape[1] - 1)
        active = self.waypoints_e[env_ids, self.current_index]
        following = self.waypoints_e[env_ids, following_index]
        return torch.cat((active, following), dim=-1)

    @property
    def spline_enabled(self) -> bool:
        """Whether the optional tangent-continuous path representation is active."""
        cfg = getattr(self, "cfg", None)
        return bool(getattr(cfg, "spline_enabled", False)) and hasattr(self, "path_knot_tangent_e")

    def _initialize_spline_path(self) -> None:
        """Allocate compact path geometry and current-projection buffers."""
        waypoint_count = self.waypoints_e.shape[1]
        self.path_knot_tangent_e = torch.zeros(self.num_envs, waypoint_count + 1, 3, device=self.device)
        self.path_segment_length = torch.zeros(self.num_envs, waypoint_count, device=self.device)
        self.path_segment_start_length = torch.zeros_like(self.path_segment_length)
        self.path_total_length = torch.zeros(self.num_envs, device=self.device)
        self.previous_path_progress = torch.zeros(self.num_envs, device=self.device)
        self._path_projection_e = torch.zeros(self.num_envs, 3, device=self.device)
        self._path_cross_track_error_e = torch.zeros_like(self._path_projection_e)
        self._path_tangent_e = torch.zeros_like(self._path_projection_e)
        self._path_curvature_e = torch.zeros_like(self._path_projection_e)
        self._path_projection_parameter = torch.zeros(self.num_envs, device=self.device)
        self._path_projection_segment_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._path_progress = torch.zeros(self.num_envs, device=self.device)
        self._path_state_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._path_speed_reference_e = torch.zeros(self.num_envs, device=self.device)
        self._path_cache_step: int | None = None
        self._path_speed_reference_cache_key: tuple[object, ...] | None = None

    def _require_spline_path(self) -> None:
        if not self.spline_enabled:
            raise RuntimeError("Spline path features require WaypointSequenceCommandCfg(spline_enabled=True).")

    def _invalidate_path_cache(self) -> None:
        if self.spline_enabled:
            self._path_cache_step = None
            self._path_speed_reference_cache_key = None

    def _route_points_e(self) -> torch.Tensor:
        route_anchor_e = getattr(self, "route_anchor_e", torch.zeros_like(self.waypoints_e[:, 0]))
        return torch.cat((route_anchor_e.unsqueeze(1), self.waypoints_e), dim=1)

    def _knot_plane_crossing(
        self,
        position_e: torch.Tensor,
        waypoint_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return bounded forward-plane crossings and their lateral miss distances [m]."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        knot = self.waypoints_e[env_ids, waypoint_index]
        tangent_index = (waypoint_index + 1).unsqueeze(-1).expand(-1, 3)
        tangent = torch.gather(self.path_knot_tangent_e, 1, tangent_index.unsqueeze(1))[:, 0]
        tangent = _normalize_path_vector(tangent)
        displacement = torch.nan_to_num(position_e - knot)
        forward_distance = torch.sum(displacement * tangent, dim=-1)
        lateral_vector = displacement - forward_distance.unsqueeze(-1) * tangent
        lateral_distance = torch.linalg.vector_norm(lateral_vector, dim=-1)
        tolerance = getattr(self.cfg, "spline_plane_crossing_lateral_tolerance", 0.50)
        finite = torch.isfinite(position_e).all(dim=-1) & torch.isfinite(lateral_distance)
        crossed = finite & (forward_distance > 0.0) & (lateral_distance <= tolerance)
        return crossed, torch.nan_to_num(lateral_distance)

    def _route_progress_preview(
        self,
        robot_position_e: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Preview monotonic waypoint traversal at one fixed robot position.

        Returns the resulting index and completion mask, number of passed knots,
        strict-radius hit count and distance, plane-only miss count and distance,
        and maximum plane-only miss distance. The fixed iteration bound keeps the
        hot path device-side while allowing more than one knot to be crossed in
        a high-speed control step.
        """
        waypoint_index = self.current_index.clone()
        completed = self.completed.clone()
        passed_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        precision_hit_count = torch.zeros_like(passed_count)
        precision_hit_distance = torch.zeros(self.num_envs, device=self.device)
        precision_miss_count = torch.zeros_like(passed_count)
        precision_miss_distance = torch.zeros(self.num_envs, device=self.device)
        precision_miss_distance_max = torch.zeros(self.num_envs, device=self.device)
        route_points = self._route_points_e()
        env_ids = torch.arange(self.num_envs, device=self.device)
        final_index = self.waypoints_e.shape[1] - 1
        progressive = self.spline_enabled and getattr(self.cfg, "spline_progressive_advancement", False)
        iterations = getattr(self.cfg, "spline_max_waypoint_advances_per_step", 4) if progressive else 1

        for _ in range(iterations):
            active = self.waypoints_e[env_ids, waypoint_index]
            distance = torch.linalg.vector_norm(active - robot_position_e, dim=-1)
            precision_hit = (~completed) & (distance <= self.cfg.acceptance_radius)
            if progressive:
                plane_crossed, lateral_distance = self._knot_plane_crossing(robot_position_e, waypoint_index)
                plane_crossed &= ~completed
            else:
                plane_crossed = torch.zeros_like(precision_hit)
                lateral_distance = torch.zeros_like(distance)
            passed = precision_hit | plane_crossed
            plane_only = passed & ~precision_hit

            segment_start_index = waypoint_index.unsqueeze(-1).expand(-1, 3)
            segment_end_index = (waypoint_index + 1).unsqueeze(-1).expand(-1, 3)
            segment_start = torch.gather(route_points, 1, segment_start_index.unsqueeze(1))[:, 0]
            segment_end = torch.gather(route_points, 1, segment_end_index.unsqueeze(1))[:, 0]
            segment_distance = torch.linalg.vector_norm(segment_end - segment_start, dim=-1)

            passed_count += passed.long()
            precision_hit_count += precision_hit.long()
            precision_hit_distance += precision_hit.float() * segment_distance
            precision_miss_count += plane_only.long()
            precision_miss_distance += plane_only.float() * lateral_distance
            precision_miss_distance_max = torch.where(
                plane_only,
                torch.maximum(precision_miss_distance_max, lateral_distance),
                precision_miss_distance_max,
            )

            reached_final = passed & (waypoint_index == final_index)
            completed |= reached_final
            advance = passed & ~reached_final
            waypoint_index += advance.long()

        return (
            waypoint_index,
            completed,
            passed_count,
            precision_hit_count,
            precision_hit_distance,
            precision_miss_count,
            precision_miss_distance,
            precision_miss_distance_max,
        )

    def _path_segment_controls(
        self, segment_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather cubic controls for per-environment segment indices."""
        squeeze = segment_index.ndim == 1
        gather_index = segment_index.unsqueeze(1) if squeeze else segment_index
        if gather_index.ndim != 2 or gather_index.shape[0] != self.num_envs:
            raise ValueError(
                f"segment_index must have shape ({self.num_envs},) or ({self.num_envs}, P), "
                f"got {tuple(segment_index.shape)}."
            )
        route_points = self._route_points_e()
        vector_index = gather_index.unsqueeze(-1).expand(-1, -1, 3)
        start = torch.gather(route_points, 1, vector_index)
        end = torch.gather(route_points, 1, vector_index + 1)
        start_tangent = torch.gather(self.path_knot_tangent_e, 1, vector_index)
        end_tangent = torch.gather(self.path_knot_tangent_e, 1, vector_index + 1)
        chord_length = torch.linalg.vector_norm(end - start, dim=-1, keepdim=True)
        derivative_scale = chord_length * getattr(self.cfg, "spline_tangent_scale", 0.75)
        start_derivative = start_tangent * derivative_scale
        end_derivative = end_tangent * derivative_scale
        if squeeze:
            return start[:, 0], end[:, 0], start_derivative[:, 0], end_derivative[:, 0]
        return start, end, start_derivative, end_derivative

    def _rebuild_spline_path(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Rebuild compact cubic geometry for selected reset routes."""
        if not self.spline_enabled:
            return
        route_points = torch.cat((self.route_anchor_e[env_ids].unsqueeze(1), self.waypoints_e[env_ids]), dim=1)
        route_family = getattr(self.cfg, "route_family", "random_walk")
        if route_family == "bounded_ellipse":
            tangent = _bounded_ellipse_path_tangents(
                self.route_anchor_e[env_ids], self.waypoints_e[env_ids], self.cfg.samples_per_lap
            )
        elif route_family == "bounded_template_mix":
            tangent = _bounded_template_path_tangents(
                self.route_anchor_e[env_ids],
                self.waypoints_e[env_ids],
                self.cfg.samples_per_lap,
                self.route_is_figure_eight[env_ids],
            )
        elif route_family == "bounded_hard_mix":
            random_tangent = _path_knot_tangents(route_points)
            figure_eight_tangent = _bounded_figure_eight_path_tangents(
                self.route_anchor_e[env_ids], self.waypoints_e[env_ids], self.cfg.samples_per_lap
            )
            tangent = torch.where(
                self.route_is_figure_eight[env_ids][:, None, None], figure_eight_tangent, random_tangent
            )
        else:
            tangent = _path_knot_tangents(route_points)
        chord_length = torch.linalg.vector_norm(torch.diff(route_points, dim=1), dim=-1, keepdim=True)
        derivative_scale = chord_length * getattr(self.cfg, "spline_tangent_scale", 0.75)
        start_derivative = tangent[:, :-1] * derivative_scale
        end_derivative = tangent[:, 1:] * derivative_scale
        segment_length = _cubic_path_arc_length(
            route_points[:, :-1],
            route_points[:, 1:],
            start_derivative,
            end_derivative,
            torch.ones_like(chord_length[..., 0]),
        )
        self.path_knot_tangent_e[env_ids] = tangent
        self.path_segment_length[env_ids] = segment_length
        cumulative_length = torch.cumsum(segment_length, dim=1)
        self.path_segment_start_length[env_ids] = cumulative_length - segment_length
        self.path_total_length[env_ids] = cumulative_length[:, -1]
        self.previous_path_progress[env_ids] = 0.0
        self._invalidate_path_cache()

    def _update_cached_path_state(self) -> None:
        """Update a bounded current-or-next spline projection once per step."""
        self._require_spline_path()
        step = getattr(self._env, "common_step_counter", None)
        if step is not None and self._path_cache_step == step:
            return
        robot_position_e = link_pose_w(self.robot)[:, :3] - self._env.scene.env_origins
        env_ids = torch.arange(self.num_envs, device=self.device)
        progressive = getattr(self.cfg, "spline_progressive_advancement", False)
        final_index = self.waypoints_e.shape[1] - 1
        if progressive:
            following_index = torch.clamp(self.current_index + 1, max=final_index)
            candidate_index = torch.stack((self.current_index, following_index), dim=1)
            start, end, start_derivative, end_derivative = self._path_segment_controls(candidate_index)
            projection, error, tangent, curvature, parameter = _project_cubic_path_segment(
                robot_position_e.unsqueeze(1).expand(-1, 2, -1),
                start,
                end,
                start_derivative,
                end_derivative,
                coarse_samples=getattr(self.cfg, "spline_projection_samples", 12),
            )
            crossed, _ = self._knot_plane_crossing(robot_position_e, self.current_index)
            has_next = (~self.completed) & (self.current_index < final_index)
            distance_sq = torch.sum(error.square(), dim=-1)
            choose_next = has_next & crossed & (distance_sq[:, 1] <= distance_sq[:, 0])
            choice = choose_next.long()
            vector_choice = choice[:, None, None].expand(-1, 1, 3)
            scalar_choice = choice[:, None]
            projection = torch.gather(projection, 1, vector_choice)[:, 0]
            error = torch.gather(error, 1, vector_choice)[:, 0]
            tangent = torch.gather(tangent, 1, vector_choice)[:, 0]
            curvature = torch.gather(curvature, 1, vector_choice)[:, 0]
            parameter = torch.gather(parameter, 1, scalar_choice)[:, 0]
            selected_index = torch.gather(candidate_index, 1, scalar_choice)[:, 0]
            start = torch.gather(start, 1, vector_choice)[:, 0]
            end = torch.gather(end, 1, vector_choice)[:, 0]
            start_derivative = torch.gather(start_derivative, 1, vector_choice)[:, 0]
            end_derivative = torch.gather(end_derivative, 1, vector_choice)[:, 0]
        else:
            selected_index = self.current_index
            start, end, start_derivative, end_derivative = self._path_segment_controls(selected_index)
            projection, error, tangent, curvature, parameter = _project_cubic_path_segment(
                robot_position_e,
                start,
                end,
                start_derivative,
                end_derivative,
                coarse_samples=getattr(self.cfg, "spline_projection_samples", 12),
            )
        segment_start_length = self.path_segment_start_length[env_ids, selected_index]
        partial_length = _cubic_path_arc_length(start, end, start_derivative, end_derivative, parameter)
        valid = torch.isfinite(robot_position_e).all(dim=-1) & torch.isfinite(partial_length)
        self._path_projection_e.copy_(projection)
        self._path_cross_track_error_e.copy_(error)
        self._path_tangent_e.copy_(tangent)
        self._path_curvature_e.copy_(curvature)
        self._path_projection_parameter.copy_(parameter)
        self._path_projection_segment_index.copy_(selected_index)
        self._path_progress.copy_(
            torch.where(
                valid,
                segment_start_length + torch.nan_to_num(partial_length),
                torch.zeros_like(partial_length),
            )
        )
        self._path_state_valid.copy_(valid)
        self._path_cache_step = step

    @property
    def path_projection_e(self) -> torch.Tensor:
        """Closest point on the indexed active cubic segment [m], shape ``(N, 3)``."""
        self._update_cached_path_state()
        return self._path_projection_e

    @property
    def path_cross_track_error_e(self) -> torch.Tensor:
        """Projection-to-robot indexed spline error [m], shape ``(N, 3)``."""
        self._update_cached_path_state()
        return self._path_cross_track_error_e

    @property
    def path_cross_track_distance(self) -> torch.Tensor:
        """Indexed spline cross-track distance [m], shape ``(N,)``."""
        return torch.linalg.vector_norm(self.path_cross_track_error_e, dim=-1)

    @property
    def path_tangent_e(self) -> torch.Tensor:
        """Unit tangent at the indexed spline projection, shape ``(N, 3)``."""
        self._update_cached_path_state()
        return self._path_tangent_e

    @property
    def path_curvature_e(self) -> torch.Tensor:
        """Curvature binormal at the indexed spline projection [1/m], shape ``(N, 3)``."""
        self._update_cached_path_state()
        return self._path_curvature_e

    @property
    def path_progress(self) -> torch.Tensor:
        """Projected arc length from the route anchor [m], shape ``(N,)``."""
        self._update_cached_path_state()
        return self._path_progress

    @property
    def path_progress_fraction(self) -> torch.Tensor:
        """Projected route completion fraction in ``[0, 1]``, shape ``(N,)``."""
        progress = self.path_progress
        return (progress / self.path_total_length.clamp_min(torch.finfo(progress.dtype).eps)).clamp_(0.0, 1.0)

    @property
    def path_remaining_distance(self) -> torch.Tensor:
        """Remaining indexed spline arc length [m], shape ``(N,)``."""
        remaining = (self.path_total_length - self.path_progress).clamp_min_(0.0)
        return torch.where(self.completed, torch.zeros_like(remaining), remaining)

    def compute_path_speed_reference(
        self,
        cruise_speed: float,
        maximum_lateral_acceleration: float,
        maximum_braking_acceleration: float,
        curvature_lookahead_distances: Sequence[float] = (0.0, 0.75, 1.50),
    ) -> torch.Tensor:
        """Return a shared curvature- and stopping-limited path speed [m/s].

        Args:
            cruise_speed: Unconstrained target path speed [m/s].
            maximum_lateral_acceleration: Curvature acceleration limit [m/s^2].
            maximum_braking_acceleration: Along-path stopping deceleration [m/s^2].
            curvature_lookahead_distances: Forward spline samples [m] used for
                the curvature braking envelope.

        Returns:
            Per-environment nonnegative speed references [m/s], shape ``(N,)``.
        """
        scalar_values = {
            "cruise_speed": cruise_speed,
            "maximum_lateral_acceleration": maximum_lateral_acceleration,
            "maximum_braking_acceleration": maximum_braking_acceleration,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0.0
            for value in scalar_values.values()
        ):
            raise ValueError("Path speed-reference limits must be finite positive scalars.")
        distances = tuple(curvature_lookahead_distances)
        if not distances or any(
            isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0.0
            for value in distances
        ):
            raise ValueError("curvature_lookahead_distances must contain finite nonnegative values.")
        self._require_spline_path()
        progress = self.path_progress
        remaining = self.path_remaining_distance
        distance = torch.as_tensor(distances, device=self.device, dtype=progress.dtype)
        effective_distance = torch.minimum(distance.unsqueeze(0), remaining.unsqueeze(1))
        _, _, lookahead_curvature = self.sample_path_at_arc_length(progress.unsqueeze(1) + effective_distance)
        curvature_magnitude = torch.linalg.vector_norm(lookahead_curvature, dim=-1)
        epsilon = torch.finfo(curvature_magnitude.dtype).eps
        curve_speed = torch.sqrt(maximum_lateral_acceleration / curvature_magnitude.clamp_min(epsilon))

        # A future curve permits its own lateral-limit speed plus exactly the
        # kinetic energy that can be shed over the distance to that sample.
        # This is deliberately not a max-curvature cap: distant curvature must
        # reduce speed only when it enters the available braking envelope.
        lookahead_limit = torch.sqrt(
            curve_speed.square() + 2.0 * maximum_braking_acceleration * effective_distance
        ).amin(dim=1)
        current_curvature = torch.linalg.vector_norm(self.path_curvature_e, dim=-1)
        current_curve_limit = torch.sqrt(maximum_lateral_acceleration / current_curvature.clamp_min(epsilon))
        finish_limit = torch.sqrt(2.0 * maximum_braking_acceleration * remaining)
        cruise = torch.full_like(current_curve_limit, cruise_speed)
        return torch.minimum(
            torch.minimum(torch.minimum(cruise, current_curve_limit), lookahead_limit), finish_limit
        ).clamp_min_(0.0)

    @property
    def path_speed_reference(self) -> torch.Tensor:
        """Live config-driven path speed reference [m/s], shape ``(N,)``.

        A nonpositive configured cruise speed disables the reference and
        returns zeros, preserving existing tasks that do not consume it.
        """
        cruise_speed = getattr(self.cfg, "target_cruise_speed", 0.0)
        if cruise_speed <= 0.0:
            if hasattr(self, "_path_speed_reference_e"):
                self._path_speed_reference_e.zero_()
                self._path_speed_reference_cache_key = None
                return self._path_speed_reference_e
            return torch.zeros(self.num_envs, device=self.device, dtype=self.waypoints_e.dtype)
        maximum_lateral_acceleration = self.cfg.maximum_lateral_acceleration
        maximum_braking_acceleration = self.cfg.maximum_braking_acceleration
        lookahead_distances = tuple(self.cfg.speed_lookahead_distances)
        cache_key = (
            getattr(self._env, "common_step_counter", None),
            float(cruise_speed),
            float(maximum_lateral_acceleration),
            float(maximum_braking_acceleration),
            lookahead_distances,
        )
        if getattr(self, "_path_speed_reference_cache_key", None) == cache_key:
            return self._path_speed_reference_e
        reference = self.compute_path_speed_reference(
            cruise_speed=cruise_speed,
            maximum_lateral_acceleration=maximum_lateral_acceleration,
            maximum_braking_acceleration=maximum_braking_acceleration,
            curvature_lookahead_distances=lookahead_distances,
        )
        self._path_speed_reference_e.copy_(reference)
        self._path_speed_reference_cache_key = cache_key
        return self._path_speed_reference_e

    @property
    def path_state_valid(self) -> torch.Tensor:
        """Whether the current indexed spline projection used finite state."""
        self._update_cached_path_state()
        return self._path_state_valid

    def sample_path_at_arc_length(self, arc_length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample the spline at absolute per-route arc lengths.

        Args:
            arc_length: Arc lengths from the route anchor [m], shape ``(N,)`` or ``(N, P)``.

        Returns:
            Path positions [m], unit tangents, and curvature binormals [1/m], each
            matching ``arc_length.shape + (3,)``.
        """
        self._require_spline_path()
        squeeze = arc_length.ndim == 1
        target = arc_length.unsqueeze(1) if squeeze else arc_length
        if target.ndim != 2 or target.shape[0] != self.num_envs:
            raise ValueError(
                f"arc_length must have shape ({self.num_envs},) or ({self.num_envs}, P), got {tuple(arc_length.shape)}."
            )
        target = torch.nan_to_num(target).clamp_min_(0.0)
        target = torch.minimum(target, self.path_total_length.unsqueeze(1))
        segment_end_length = self.path_segment_start_length + self.path_segment_length
        segment_index = torch.searchsorted(segment_end_length.contiguous(), target.contiguous(), right=False)
        segment_index.clamp_(max=self.path_segment_length.shape[1] - 1)
        start, end, start_derivative, end_derivative = self._path_segment_controls(segment_index)
        segment_start_length = torch.gather(self.path_segment_start_length, 1, segment_index)
        segment_length = torch.gather(self.path_segment_length, 1, segment_index)
        desired_segment_length = (target - segment_start_length).clamp_min_(0.0)
        parameter = (desired_segment_length / segment_length.clamp_min(torch.finfo(target.dtype).eps)).clamp_(0.0, 1.0)
        for _ in range(3):
            current_length = _cubic_path_arc_length(start, end, start_derivative, end_derivative, parameter)
            _, derivative, _ = _cubic_hermite_path(start, end, start_derivative, end_derivative, parameter)
            speed = torch.linalg.vector_norm(torch.nan_to_num(derivative), dim=-1)
            parameter += (desired_segment_length - current_length) / speed.clamp_min(torch.finfo(target.dtype).eps)
            parameter.clamp_(0.0, 1.0)
        position, derivative, second_derivative = _cubic_hermite_path(
            start, end, start_derivative, end_derivative, parameter
        )
        tangent = _normalize_path_vector(derivative)
        speed = torch.linalg.vector_norm(torch.nan_to_num(derivative), dim=-1)
        curvature = torch.cross(derivative, second_derivative, dim=-1)
        curvature /= speed.pow(3).clamp_min(torch.finfo(target.dtype).eps).unsqueeze(-1)
        position = torch.nan_to_num(position)
        tangent = torch.nan_to_num(tangent)
        curvature = torch.nan_to_num(curvature)
        if squeeze:
            return position[:, 0], tangent[:, 0], curvature[:, 0]
        return position, tangent, curvature

    def path_preview_e(self, lookahead_distances: Sequence[float]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample positions, tangents, and curvature ahead of the current projection.

        Args:
            lookahead_distances: Nonnegative forward arc-length offsets [m].

        Returns:
            Environment-frame positions [m], unit tangents, and curvature
            binormals [1/m], each with shape ``(N, P, 3)``.
        """
        distances = tuple(lookahead_distances)
        if not distances or any(not math.isfinite(value) or value < 0.0 for value in distances):
            raise ValueError("lookahead_distances must contain finite nonnegative values.")
        distance = torch.as_tensor(distances, device=self.device, dtype=self.path_progress.dtype)
        return self.sample_path_at_arc_length(self.path_progress.unsqueeze(1) + distance.unsqueeze(0))

    @property
    def active_segment_projection_e(self) -> torch.Tensor:
        """Projection of each robot onto its indexed active route segment [m]."""
        if self.spline_enabled:
            return self.path_projection_e
        robot_pos_e = link_pose_w(self.robot)[:, :3] - self._env.scene.env_origins
        projection, _ = _indexed_segment_projection(
            robot_pos_e, self.route_anchor_e, self.waypoints_e, self.current_index
        )
        return projection

    @property
    def cross_track_error_e(self) -> torch.Tensor:
        """Error from the active-segment projection to each robot [m], shape ``(N, 3)``."""
        if self.spline_enabled:
            return self.path_cross_track_error_e
        robot_pos_e = link_pose_w(self.robot)[:, :3] - self._env.scene.env_origins
        _, error = _indexed_segment_projection(robot_pos_e, self.route_anchor_e, self.waypoints_e, self.current_index)
        return error

    @property
    def cross_track_distance(self) -> torch.Tensor:
        """Distance from each robot to its indexed active route segment [m], shape ``(N,)``."""
        if self.spline_enabled:
            return self.path_cross_track_distance
        return torch.linalg.vector_norm(self.cross_track_error_e, dim=-1)

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        # Allow callers to retune offsets between episodes while preserving tensor shape.
        self._validate_cfg(self.cfg)
        randomize_waypoints = getattr(self.cfg, "randomize_waypoints", False)
        expected_count = self.cfg.random_waypoint_count if randomize_waypoints else len(self.cfg.waypoint_offsets)
        if expected_count != self.waypoints_e.shape[1]:
            raise ValueError("The number of waypoints cannot change after command construction.")

        robot_pose = link_pose_w(self.robot)[env_ids]
        anchor_e = robot_pose[:, :3] - self._env.scene.env_origins[env_ids]
        if not hasattr(self, "route_anchor_e"):
            self.route_anchor_e = torch.zeros_like(self.waypoints_e[:, 0])
        self.route_anchor_e[env_ids] = anchor_e
        _, _, reset_yaw = euler_xyz_from_quat(robot_pose[:, 3:7])
        if not hasattr(self, "route_is_figure_eight"):
            self.route_is_figure_eight = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if not hasattr(self, "route_family_id"):
            self.route_family_id = self.route_is_figure_eight.long().clone()
        self.route_is_figure_eight[env_ids] = False
        self.route_family_id[env_ids] = 0
        if randomize_waypoints:
            route_family = getattr(self.cfg, "route_family", "random_walk")
            if route_family == "bounded_ellipse":
                waypoints_e = _sample_bounded_ellipse_waypoints(
                    anchor_e,
                    self.cfg.random_waypoint_count,
                    self.cfg.samples_per_lap,
                    self.cfg.aspect_ratio_range,
                    self.cfg.vertical_amplitude_range,
                )
                first_displacement = waypoints_e[:, 0] - anchor_e
            elif route_family == "bounded_template_mix":
                waypoints_e, is_figure_eight = _sample_bounded_template_waypoints(
                    anchor_e,
                    self.cfg.random_waypoint_count,
                    self.cfg.samples_per_lap,
                    self.cfg.aspect_ratio_range,
                    self.cfg.vertical_amplitude_range,
                    self.cfg.figure_eight_probability,
                )
                self.route_is_figure_eight[env_ids] = is_figure_eight
                self.route_family_id[env_ids] = is_figure_eight.long()
                first_displacement = waypoints_e[:, 0] - anchor_e
            elif route_family == "bounded_hard_mix":
                ranges = self.cfg.random_waypoint_ranges
                bounds = torch.tensor(
                    (ranges.pos_x, ranges.pos_y, ranges.pos_z), device=self.device, dtype=robot_pose.dtype
                )
                lower_e = bounds[:, 0].unsqueeze(0).expand(len(env_ids), -1).clone()
                upper_e = bounds[:, 1].unsqueeze(0).expand(len(env_ids), -1).clone()
                # XY limits are numerical environment-frame bounds like the
                # periodic templates. Z remains reset-relative so the same
                # task works at a different nominal flight altitude.
                lower_e[:, 2] += anchor_e[:, 2]
                upper_e[:, 2] += anchor_e[:, 2]
                if self.spline_enabled:
                    assert self.cfg.maximum_waypoint_separation is not None
                    assert self.cfg.maximum_heading_change is not None
                    assert self.cfg.maximum_vertical_step is not None
                    planar_margin, vertical_margin = _hard_route_spline_sampling_margin(
                        self.cfg.minimum_waypoint_separation,
                        self.cfg.maximum_waypoint_separation,
                        self.cfg.maximum_heading_change,
                        self.cfg.maximum_vertical_step,
                        self.cfg.spline_tangent_scale,
                    )
                    margin = torch.tensor(
                        (planar_margin, planar_margin, vertical_margin),
                        device=self.device,
                        dtype=robot_pose.dtype,
                    )
                    lower_e += margin
                    upper_e -= margin
                random_offsets = _sample_bounded_waypoint_offsets(
                    len(env_ids),
                    self.cfg.random_waypoint_count,
                    lower_e - anchor_e,
                    upper_e - anchor_e,
                    self.cfg.minimum_waypoint_separation,
                    self.cfg.random_sampling_attempts,
                    self.cfg.maximum_waypoint_separation,
                    self.cfg.maximum_heading_change,
                    self.cfg.maximum_vertical_step,
                    self.cfg.nominal_heading_change,
                    self.cfg.route_sampling_attempts,
                    self.cfg.random_heading_change_interval,
                    independent_initial_heading_attempts=True,
                    select_smoothest_route_attempt=False,
                )
                assert self.cfg.maximum_heading_change is not None
                circular_fallback = _constant_curvature_fallback_mask(
                    random_offsets, self.cfg.minimum_waypoint_separation, self.cfg.maximum_heading_change
                )
                random_waypoints_e = anchor_e.unsqueeze(1) + random_offsets
                figure_eight_waypoints_e = _sample_bounded_figure_eight_waypoints(
                    anchor_e,
                    self.cfg.random_waypoint_count,
                    self.cfg.samples_per_lap,
                    self.cfg.vertical_amplitude_range,
                )
                is_figure_eight = (
                    torch.rand(len(env_ids), device=self.device, dtype=robot_pose.dtype)
                    < self.cfg.figure_eight_probability
                )
                # The correlated walk's rare constant-curvature feasibility
                # fallback looks like the circle this family is meant to avoid.
                # Substitute the independently sampled eight instead.
                is_figure_eight |= circular_fallback
                waypoints_e = torch.where(is_figure_eight[:, None, None], figure_eight_waypoints_e, random_waypoints_e)
                self.route_is_figure_eight[env_ids] = is_figure_eight
                self.route_family_id[env_ids] = torch.where(
                    is_figure_eight,
                    torch.ones_like(self.route_family_id[env_ids]),
                    torch.full_like(self.route_family_id[env_ids], 2),
                )
                first_displacement = waypoints_e[:, 0] - anchor_e
            else:
                ranges = self.cfg.random_waypoint_ranges
                bounds = torch.tensor(
                    (ranges.pos_x, ranges.pos_y, ranges.pos_z), device=self.device, dtype=robot_pose.dtype
                )
                offsets = _sample_bounded_waypoint_offsets(
                    len(env_ids),
                    self.cfg.random_waypoint_count,
                    bounds[:, 0],
                    bounds[:, 1],
                    self.cfg.minimum_waypoint_separation,
                    self.cfg.random_sampling_attempts,
                    self.cfg.maximum_waypoint_separation,
                    self.cfg.maximum_heading_change,
                    self.cfg.maximum_vertical_step,
                    self.cfg.nominal_heading_change,
                    self.cfg.route_sampling_attempts,
                    self.cfg.random_heading_change_interval,
                )
                cos_yaw = torch.cos(reset_yaw).unsqueeze(1)
                sin_yaw = torch.sin(reset_yaw).unsqueeze(1)
                rotated = offsets.clone()
                rotated[..., 0] = cos_yaw * offsets[..., 0] - sin_yaw * offsets[..., 1]
                rotated[..., 1] = sin_yaw * offsets[..., 0] + cos_yaw * offsets[..., 1]
                waypoints_e = anchor_e.unsqueeze(1) + rotated
                first_displacement = rotated[:, 0]
        else:
            fixed_offsets = torch.as_tensor(self.cfg.waypoint_offsets, device=self.device, dtype=robot_pose.dtype)
            offsets = fixed_offsets.unsqueeze(0).expand(len(env_ids), -1, -1)
            cos_yaw = torch.cos(reset_yaw).unsqueeze(1)
            sin_yaw = torch.sin(reset_yaw).unsqueeze(1)
            rotated = offsets.clone()
            rotated[..., 0] = cos_yaw * offsets[..., 0] - sin_yaw * offsets[..., 1]
            rotated[..., 1] = sin_yaw * offsets[..., 0] + cos_yaw * offsets[..., 1]
            waypoints_e = anchor_e.unsqueeze(1) + rotated
            first_displacement = rotated[:, 0]
        self.waypoints_e[env_ids] = waypoints_e

        self.current_index[env_ids] = 0
        self.completed[env_ids] = False
        self.previous_distance_sq[env_ids] = torch.sum(torch.square(first_displacement), dim=-1)
        self._rebuild_spline_path(env_ids)
        if "route_family_id" in getattr(self, "metrics", {}):
            self.metrics["route_family_id"][env_ids] = self.route_family_id[env_ids].float()

    def _update_command(self) -> None:
        """Advance accepted or safely crossed knots and regenerate completed routes."""
        robot_pos_e = link_pose_w(self.robot)[:, :3] - self._env.scene.env_origins
        env_ids = torch.arange(self.num_envs, device=self.device)
        active = self.waypoints_e[env_ids, self.current_index]
        distance_sq = torch.sum(torch.square(active - robot_pos_e), dim=-1)
        previous_index = self.current_index.clone()
        previous_completed = self.completed.clone()
        next_index, next_completed, *_ = self._route_progress_preview(robot_pos_e)
        reached_final = next_completed & ~previous_completed
        regenerate = reached_final & getattr(self.cfg, "regenerate_on_completion", False)
        self.current_index.copy_(next_index)
        self.completed.copy_(next_completed & ~regenerate)
        advance = self.current_index != previous_index

        # RewardManager evaluates the old final waypoint before CommandManager.
        # Re-anchoring here therefore pays final-target progress exactly once,
        # then seeds the following step's potential at the new target distance.
        regenerate_env_ids = regenerate.nonzero(as_tuple=False).flatten()
        if len(regenerate_env_ids) > 0:
            self._resample_command(regenerate_env_ids)

        advanced_target = self.waypoints_e[env_ids, self.current_index]
        advanced_distance_sq = torch.sum(torch.square(advanced_target - robot_pos_e), dim=-1)
        target_changed = advance | regenerate
        self.previous_distance_sq[:] = torch.where(target_changed, advanced_distance_sq, distance_sq)
        if self.spline_enabled:
            # RewardManager evaluates path progress before CommandManager. Seed
            # the next potential after any segment switch so the discrete knot
            # index cannot create an artificial arc-length reward impulse.
            self._invalidate_path_cache()
            self.previous_path_progress.copy_(self.path_progress)

    def _record_current_metrics(self) -> None:
        robot_pos_e = link_pose_w(self.robot)[:, :3] - self._env.scene.env_origins
        env_ids = torch.arange(self.num_envs, device=self.device)
        active = self.waypoints_e[env_ids, self.current_index]
        distance = torch.linalg.norm(active - robot_pos_e, dim=-1)
        (
            preview_index,
            preview_completed,
            _,
            precision_hit_count,
            precision_hit_distance,
            precision_miss_count,
            precision_miss_distance,
            precision_miss_distance_max,
        ) = self._route_progress_preview(robot_pos_e)
        completed_count = preview_index + preview_completed.long()
        reached_final = preview_completed & ~self.completed
        elapsed_time = self._env.episode_length_buf.float() * self._env.step_dt
        route_arc_length = self._route_arc_length_preview(robot_pos_e, preview_index, preview_completed)
        self._route_arc_length_traversed.copy_(torch.maximum(self._route_arc_length_traversed, route_arc_length))
        self.metrics["route_arc_length_traversed"].copy_(self._route_arc_length_traversed)
        self.metrics["route_arc_length_traversal_rate"].copy_(
            self._route_arc_length_traversed / torch.nan_to_num(elapsed_time).clamp_min(self._env.step_dt)
        )
        self._record_arrival_metrics(
            precision_hit_count,
            reached_final,
            precision_hit_distance,
            precision_miss_count,
            precision_miss_distance,
            precision_miss_distance_max,
            completed_count,
            elapsed_time,
        )
        if not hasattr(self, "route_is_figure_eight"):
            self.route_is_figure_eight = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if not hasattr(self, "route_family_id"):
            self.route_family_id = self.route_is_figure_eight.long().clone()
        self.metrics["route_family_id"].copy_(self.route_family_id.float())
        self._update_cross_track_metrics(self.cross_track_distance)
        episode_completed = preview_completed
        if getattr(self.cfg, "regenerate_on_completion", False):
            episode_completed = torch.zeros_like(preview_completed)
        self._record_episode_metrics(
            distance,
            waypoint_fraction=completed_count.float() / self.waypoints_e.shape[1],
            waypoint_completed=episode_completed,
        )

    def _route_arc_length_preview(
        self,
        robot_position_e: torch.Tensor,
        preview_index: torch.Tensor,
        preview_completed: torch.Tensor,
    ) -> torch.Tensor:
        """Return projected monotonic traversal distance for the preview state [m]."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        if self.spline_enabled:
            projected = self.path_progress
            segment_start = self.path_segment_start_length[env_ids, preview_index]
            total_length = self.path_total_length
        else:
            route_points = self._route_points_e()
            segment = torch.diff(route_points, dim=1)
            segment_length = torch.linalg.vector_norm(segment, dim=-1)
            cumulative_length = torch.cumsum(segment_length, dim=1)
            segment_start_length = cumulative_length - segment_length
            segment_start = segment_start_length[env_ids, preview_index]
            start = route_points[env_ids, preview_index]
            direction = segment[env_ids, preview_index]
            parameter = torch.sum((robot_position_e - start) * direction, dim=-1)
            parameter /= torch.sum(direction.square(), dim=-1).clamp_min(torch.finfo(direction.dtype).eps)
            projected = segment_start + parameter.clamp(0.0, 1.0) * segment_length[env_ids, preview_index]
            total_length = cumulative_length[:, -1]

        # A multi-knot plane crossing provides an exact lower bound even when
        # the bounded projection window still contains only current + next.
        traversal = torch.maximum(torch.nan_to_num(projected), segment_start)
        traversal = torch.where(preview_completed, total_length, traversal)
        return torch.minimum(traversal.clamp_min_(0.0), total_length)

    def _record_arrival_metrics(
        self,
        precision_hit_count: torch.Tensor,
        reached_final: torch.Tensor,
        precision_hit_distance: torch.Tensor,
        precision_miss_count: torch.Tensor,
        precision_miss_distance: torch.Tensor,
        precision_miss_distance_max: torch.Tensor,
        completed_count: torch.Tensor,
        elapsed_time: torch.Tensor,
    ) -> None:
        """Accumulate strict precision hits separately from route traversal."""
        finite_elapsed = torch.nan_to_num(elapsed_time).reshape(-1).clamp_min(0.0)
        hit_count = precision_hit_count.float()
        hit = precision_hit_count > 0
        interval = (finite_elapsed - self._last_arrival_time).clamp_min(0.0)

        self.metrics["waypoint_arrivals"] += hit_count
        self.metrics["waypoint_precision_hits"] += hit_count
        self.metrics["route_completions"] += reached_final.float()
        self.metrics["target_distance_completed"] += torch.nan_to_num(precision_hit_distance).clamp_min(0.0)
        self._arrival_interval_sum += hit_count * interval
        self._arrival_interval_min[:] = torch.where(
            hit, torch.minimum(self._arrival_interval_min, interval), self._arrival_interval_min
        )
        self._last_arrival_time[:] = torch.where(hit, finite_elapsed, self._last_arrival_time)

        finite_miss_count = precision_miss_count.float()
        self._precision_miss_count += finite_miss_count
        self._precision_miss_distance_sum += torch.nan_to_num(precision_miss_distance).clamp_min(0.0)
        self.metrics["waypoint_precision_misses"].copy_(self._precision_miss_count)
        self.metrics["waypoint_precision_miss_distance_mean"].copy_(
            self._precision_miss_distance_sum / self._precision_miss_count.clamp_min(1.0)
        )
        self.metrics["waypoint_precision_miss_distance_max"].copy_(
            torch.maximum(
                self.metrics["waypoint_precision_miss_distance_max"],
                torch.nan_to_num(precision_miss_distance_max).clamp_min(0.0),
            )
        )

        finite_completed_count = torch.nan_to_num(completed_count).float().clamp(0.0, self.waypoints_e.shape[1])
        self.metrics["route_waypoints_passed"].copy_(finite_completed_count)
        self.metrics["route_traversal_fraction"].copy_(finite_completed_count / self.waypoints_e.shape[1])
        self.metrics["waypoint_precision_hit_fraction"].copy_(
            self.metrics["waypoint_precision_hits"] / finite_completed_count.clamp_min(1.0)
        )

        arrival_count = self.metrics["waypoint_arrivals"]
        has_arrival = arrival_count > 0.0
        divisor = arrival_count.clamp_min(1.0)
        self.metrics["waypoint_arrival_time_mean"][:] = self._arrival_interval_sum / divisor
        self.metrics["waypoint_arrival_time_min"][:] = torch.where(
            has_arrival, self._arrival_interval_min, torch.zeros_like(self._arrival_interval_min)
        )
        self.metrics["waypoint_arrival_time_max"][:] = torch.where(
            hit,
            torch.maximum(self.metrics["waypoint_arrival_time_max"], interval),
            self.metrics["waypoint_arrival_time_max"],
        )
        self.metrics["episode_duration"][:] = finite_elapsed
        self.metrics["waypoint_throughput"][:] = arrival_count / finite_elapsed.clamp_min(self._env.step_dt)

    def _update_cross_track_metrics(self, distance: torch.Tensor) -> None:
        """Accumulate active-segment cross-track distance summaries."""
        finite_distance = torch.nan_to_num(distance).reshape(-1)
        active = ~self._episode_metrics._completion_time_recorded
        active_float = active.float()
        self._cross_track_count += active_float
        self._cross_track_sum += active_float * finite_distance
        self._cross_track_sq_sum += active_float * finite_distance.square()
        count = self._cross_track_count.clamp_min(1.0)
        self.metrics["cross_track_error_mean"][:] = self._cross_track_sum / count
        self.metrics["cross_track_error_rms"][:] = torch.sqrt(self._cross_track_sq_sum / count)
        self.metrics["cross_track_error_max"][:] = torch.where(
            active,
            torch.maximum(self.metrics["cross_track_error_max"], finite_distance),
            self.metrics["cross_track_error_max"],
        )

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "waypoint_visualizer"):
                self.waypoint_visualizer = VisualizationMarkers(self.cfg.waypoint_visualizer_cfg)
                self.route_segment_visualizer = VisualizationMarkers(self.cfg.route_segment_visualizer_cfg)
            self.waypoint_visualizer.set_visibility(True)
            self.route_segment_visualizer.set_visibility(True)
        elif hasattr(self, "waypoint_visualizer"):
            self.waypoint_visualizer.set_visibility(False)
            self.route_segment_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        del event
        if not getattr(self.robot, "is_initialized", True):
            return

        waypoints_w = self.waypoints_e + self._env.scene.env_origins.unsqueeze(1)
        waypoint_count = waypoints_w.shape[1]
        waypoint_id = torch.arange(waypoint_count, device=self.device).unsqueeze(0)
        completed = waypoint_id < self.current_index.unsqueeze(1)
        completed |= self.completed.unsqueeze(1) & (waypoint_id == self.current_index.unsqueeze(1))
        active = (~self.completed).unsqueeze(1) & (waypoint_id == self.current_index.unsqueeze(1))

        # Prototype order is active, completed, future.
        waypoint_marker_id = torch.full_like(waypoint_id.expand(self.num_envs, -1), 2, dtype=torch.int32)
        waypoint_marker_id[completed] = 1
        waypoint_marker_id[active] = 0
        self.waypoint_visualizer.visualize(
            translations=waypoints_w.reshape(-1, 3), marker_indices=waypoint_marker_id.reshape(-1)
        )

        route_points_w = torch.cat(
            (self.route_anchor_e.unsqueeze(1) + self._env.scene.env_origins.unsqueeze(1), waypoints_w), dim=1
        )
        segment_route_index = torch.arange(waypoint_count, device=self.device)
        if self.spline_enabled:
            samples_per_segment = max(getattr(self.cfg, "spline_projection_samples", 12), 8)
            route_points_e = self._route_points_e()
            chord_length = torch.linalg.vector_norm(torch.diff(route_points_e, dim=1), dim=-1, keepdim=True)
            derivative_scale = chord_length * getattr(self.cfg, "spline_tangent_scale", 0.75)
            parameter = torch.linspace(
                0.0, 1.0, samples_per_segment + 1, device=self.device, dtype=route_points_e.dtype
            )
            parameter = parameter.expand(self.num_envs, waypoint_count, -1)
            sampled_points_e, _, _ = _cubic_hermite_path(
                route_points_e[:, :-1],
                route_points_e[:, 1:],
                self.path_knot_tangent_e[:, :-1] * derivative_scale,
                self.path_knot_tangent_e[:, 1:] * derivative_scale,
                parameter,
            )
            route_points_e = torch.cat(
                (sampled_points_e[:, :, :-1].reshape(self.num_envs, -1, 3), route_points_e[:, -1:]), dim=1
            )
            route_points_w = route_points_e + self._env.scene.env_origins.unsqueeze(1)
            segment_route_index = segment_route_index.repeat_interleave(samples_per_segment)
        segment_midpoint, segment_quat, segment_scale = _route_segment_marker_transforms(route_points_w)
        segment_id = segment_route_index.unsqueeze(0)
        # Prototype order is completed, future. A segment is complete after its
        # destination waypoint becomes active or completed.
        segment_completed = segment_id < self.current_index.unsqueeze(1)
        segment_completed |= self.completed.unsqueeze(1) & (segment_id == self.current_index.unsqueeze(1))
        segment_marker_id = (~segment_completed).to(torch.int32)
        self.route_segment_visualizer.visualize(
            translations=segment_midpoint.reshape(-1, 3),
            orientations=segment_quat.reshape(-1, 4),
            scales=segment_scale.reshape(-1, 3),
            marker_indices=segment_marker_id.reshape(-1),
        )


@configclass
class WaypointSequenceCommandCfg(CommandTermCfg):
    """Configuration for a reset-position/yaw-relative waypoint sequence."""

    class_type: type[WaypointSequenceCommand] = WaypointSequenceCommand

    record_slung_load_metrics: bool = True
    """Whether payload, swing, and cable episode metrics are physically available.

    Rigid-only task variants set this to ``False``. Their command term retains
    the common metric schema with zero placeholders and publishes
    ``slung_load_metrics_available=0`` so evaluators can mark those channels as
    unavailable instead of interpreting them as perfect slung-load behavior.
    """

    resampling_time_range: tuple[float, float] = (1.0e8, 1.0e8)
    """Keep one waypoint sequence for the whole episode."""

    waypoint_offsets: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (2.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
    )
    """Offsets from the reset drone pose [m], expressed in its reset-yaw frame."""

    randomize_waypoints: bool = False
    """Whether to sample a new bounded waypoint sequence at every episode reset."""

    regenerate_on_completion: bool = False
    """Whether a completed random sequence is immediately replaced without ending the episode."""

    random_waypoint_count: int = 4
    """Fixed number of waypoints in a sampled route."""

    route_family: str = "random_walk"
    """Random route family: walk, ellipse, periodic mix, or hard-route mix.

    The bounded ellipse is centered at the numerical environment origin and
    derives its major radius and orientation from the post-reset drone anchor.
    The bounded template mix samples either that ellipse or a tangent-lobe
    figure-eight independently for every environment. The bounded hard mix
    samples either a tangent-lobe figure-eight or a heading-correlated bounded
    random route, using the existing random-waypoint geometry limits.
    """

    samples_per_lap: int = 24
    """Number of uniformly spaced samples per bounded periodic lap."""

    aspect_ratio_range: tuple[float, float] = (0.94, 1.0)
    """Minor-to-major axis ratio range for bounded ellipses."""

    vertical_amplitude_range: tuple[float, float] = (0.0, 0.15)
    """Vertical sinusoid amplitude range for bounded periodic routes [m]."""

    figure_eight_probability: float = 0.5
    """Probability of sampling a figure-eight in either bounded route mix."""

    @configclass
    class Ranges:
        """Uniform reset-relative sampling ranges for random waypoint routes."""

        pos_x: tuple[float, float] = (-4.0, 4.0)
        """Waypoint X-coordinate range [m] in the reset-yaw frame."""

        pos_y: tuple[float, float] = (-4.0, 4.0)
        """Waypoint Y-coordinate range [m] in the reset-yaw frame."""

        pos_z: tuple[float, float] = (-0.75, 0.75)
        """Waypoint Z-coordinate range [m] relative to reset height."""

    random_waypoint_ranges: Ranges = Ranges()
    """Axis-aligned sampling bounds for random waypoint routes."""

    minimum_waypoint_separation: float = 1.0
    """Minimum anchor-to-first and consecutive waypoint distance [m]."""

    maximum_waypoint_separation: float | None = None
    """Optional maximum anchor-to-first and consecutive waypoint distance [m]."""

    maximum_heading_change: float | None = None
    """Optional maximum planar turn [rad] between consecutive route segments."""

    nominal_heading_change: float | None = None
    """Optional uniform turn half-width [rad] used away from route boundaries."""

    maximum_vertical_step: float | None = None
    """Optional maximum absolute waypoint-to-waypoint vertical change [m]."""

    random_sampling_attempts: int = 32
    """Uniform candidates tried per waypoint before using a feasible corner fallback."""

    route_sampling_attempts: int = 4
    """Complete smooth-route candidates tried before using the constant-curvature fallback."""

    random_heading_change_interval: int = 1
    """Chords between random turn draws; boundary-centering may still turn sooner."""

    acceptance_radius: float = 0.5
    """Distance [m] at which the active waypoint advances.

    This value is read on every command update and may be changed safely by a
    curriculum without rebuilding the command term.
    """

    spline_enabled: bool = False
    """Whether to track a waypoint-interpolating tangent-continuous cubic path.

    This is opt-in so existing waypoint and polyline tasks keep their exact
    command, reward, metric, and visualization behavior.
    """

    spline_tangent_scale: float = 0.75
    """Shared knot-derivative magnitude as a fraction of each segment chord.

    Values below one avoid overshoot on short randomized segments while the
    shared direction preserves geometric tangent continuity at every waypoint.
    This geometry setting is applied whenever a route is resampled.
    """

    spline_projection_samples: int = 12
    """Uniform projection-basin samples per active cubic segment.

    Three bounded Newton iterations refine the selected sample entirely on the
    simulation device.
    """

    spline_progressive_advancement: bool = False
    """Whether a bounded forward knot-plane crossing may advance the route.

    The default preserves sphere-only advancement. When enabled, spline
    projection may recover onto only the immediately following segment, so a
    missed knot cannot strand the index or jump across a self-intersection.
    """

    spline_plane_crossing_lateral_tolerance: float = 0.50
    """Maximum lateral distance from a crossed knot plane [m]."""

    spline_max_waypoint_advances_per_step: int = 4
    """Maximum monotonic knot advances evaluated in one control step."""

    target_cruise_speed: float = 0.0
    """Live shared path cruise speed [m/s]; zero disables the reference."""

    maximum_lateral_acceleration: float = 0.0
    """Lateral acceleration limit used by the shared speed reference [m/s^2]."""

    maximum_braking_acceleration: float = 0.0
    """Stopping deceleration used by the shared speed reference [m/s^2]."""

    speed_lookahead_distances: tuple[float, ...] = (0.0, 0.75, 1.50)
    """Forward curvature samples used by the shared speed reference [m]."""

    asset_name: str = "robot"
    """Scene entity used to anchor and evaluate the sequence."""

    waypoint_visualizer_cfg: VisualizationMarkersCfg = _WAYPOINT_MARKER_CFG
    """Markers for the active, completed, and future waypoints."""

    route_segment_visualizer_cfg: VisualizationMarkersCfg = _ROUTE_SEGMENT_MARKER_CFG
    """Unit-cylinder markers joining consecutive waypoints into a route polyline."""
