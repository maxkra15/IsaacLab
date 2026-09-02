# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused unit coverage for randomized and indexed slung-load routes."""

import math
from types import SimpleNamespace

import pytest
import torch

import isaaclab_tasks.contrib.drone_slung_load.mdp.commands as commands_module

pytestmark = pytest.mark.unit


def _spline_term(
    robot_position: torch.Tensor,
    waypoints_e: torch.Tensor,
    current_index: torch.Tensor | None = None,
) -> commands_module.WaypointSequenceCommand:
    """Construct a spline command term without launching a simulator."""
    num_envs, waypoint_count, _ = waypoints_e.shape
    quaternion = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(num_envs, -1)
    pose = torch.cat((robot_position, quaternion), dim=-1)
    term = object.__new__(commands_module.WaypointSequenceCommand)
    term.cfg = commands_module.WaypointSequenceCommandCfg(
        waypoint_offsets=tuple((float(index + 1), 0.0, 0.0) for index in range(waypoint_count)),
        spline_enabled=True,
    )
    term.robot = SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=pose)))
    term._env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        common_step_counter=0,
        scene=SimpleNamespace(env_origins=torch.zeros(num_envs, 3)),
    )
    term.route_anchor_e = torch.zeros(num_envs, 3)
    term.waypoints_e = waypoints_e.clone()
    term.current_index = torch.zeros(num_envs, dtype=torch.long) if current_index is None else current_index.clone()
    term.completed = torch.zeros(num_envs, dtype=torch.bool)
    term.previous_distance_sq = torch.zeros(num_envs)
    term._initialize_spline_path()
    term._rebuild_spline_path(torch.arange(num_envs))
    return term


def _sample_routes(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return commands_module._sample_bounded_waypoint_offsets(
        num_routes=512,
        num_waypoints=6,
        lower=torch.tensor([-4.0, -3.0, -0.5]),
        upper=torch.tensor([4.0, 3.0, 0.75]),
        minimum_separation=0.8,
        max_sampling_attempts=16,
        maximum_separation=2.25,
    )


def _hard_mix_term(
    route_count: int,
    *,
    figure_eight_probability: float = 0.5,
    spline_enabled: bool = True,
) -> commands_module.WaypointSequenceCommand:
    """Construct the final annulus-anchored W24 hard-route command without simulation."""
    angle = torch.arange(route_count, dtype=torch.float32) * (math.tau / route_count) - math.pi
    radius = torch.linspace(4.3, 4.8, route_count)
    anchor = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle), torch.zeros_like(radius)), dim=-1)
    quaternion = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(route_count, -1)
    pose = torch.cat((anchor, quaternion), dim=-1)
    cfg = commands_module.WaypointSequenceCommandCfg(
        randomize_waypoints=True,
        random_waypoint_count=24,
        route_family="bounded_hard_mix",
        samples_per_lap=24,
        vertical_amplitude_range=(0.0, 0.15),
        figure_eight_probability=figure_eight_probability,
        random_waypoint_ranges=commands_module.WaypointSequenceCommandCfg.Ranges(
            pos_x=(-4.8, 4.8), pos_y=(-4.8, 4.8), pos_z=(-0.4, 0.4)
        ),
        minimum_waypoint_separation=0.9,
        maximum_waypoint_separation=1.35,
        nominal_heading_change=math.radians(100.0),
        maximum_heading_change=math.radians(110.0),
        maximum_vertical_step=0.15,
        random_sampling_attempts=32,
        route_sampling_attempts=16,
        random_heading_change_interval=3,
        spline_enabled=spline_enabled,
        spline_tangent_scale=1.0,
    )
    commands_module.WaypointSequenceCommand._validate_cfg(cfg)
    term = object.__new__(commands_module.WaypointSequenceCommand)
    term.cfg = cfg
    term.robot = SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=pose)))
    term._env = SimpleNamespace(
        num_envs=route_count,
        device="cpu",
        common_step_counter=0,
        scene=SimpleNamespace(env_origins=torch.zeros(route_count, 3)),
    )
    term.route_anchor_e = torch.zeros(route_count, 3)
    term.waypoints_e = torch.zeros(route_count, 24, 3)
    term.route_is_figure_eight = torch.zeros(route_count, dtype=torch.bool)
    term.route_family_id = torch.zeros(route_count, dtype=torch.long)
    term.current_index = torch.zeros(route_count, dtype=torch.long)
    term.completed = torch.zeros(route_count, dtype=torch.bool)
    term.previous_distance_sq = torch.zeros(route_count)
    if spline_enabled:
        term._initialize_spline_path()
    term.metrics = {"route_family_id": torch.zeros(route_count)}
    return term


def test_random_routes_cover_all_headings_and_respect_geometry_contracts():
    routes = _sample_routes(seed=7)

    assert routes.shape == (512, 6, 3)
    lower = torch.tensor([-4.0, -3.0, -0.5])
    upper = torch.tensor([4.0, 3.0, 0.75])
    assert torch.all(routes >= lower)
    assert torch.all(routes <= upper)

    points = torch.cat((torch.zeros(routes.shape[0], 1, 3), routes), dim=1)
    segment = torch.diff(points, dim=1)
    separation = torch.linalg.vector_norm(segment, dim=-1)
    assert torch.all(separation >= 0.8 - 1.0e-6)
    assert torch.all(separation <= 2.25 + 1.0e-6)

    planar_segment = segment[..., :2].reshape(-1, 2)
    quadrants = (
        (planar_segment[:, 0] > 0.0) & (planar_segment[:, 1] > 0.0),
        (planar_segment[:, 0] < 0.0) & (planar_segment[:, 1] > 0.0),
        (planar_segment[:, 0] < 0.0) & (planar_segment[:, 1] < 0.0),
        (planar_segment[:, 0] > 0.0) & (planar_segment[:, 1] < 0.0),
    )
    assert all(torch.any(quadrant) for quadrant in quadrants)


def test_random_route_sampling_is_reproducible_under_torch_seed():
    first = _sample_routes(seed=1234)
    second = _sample_routes(seed=1234)
    different_seed = _sample_routes(seed=1235)

    torch.testing.assert_close(first, second)
    assert not torch.equal(first, different_seed)


def test_bounded_ellipse_routes_are_local_periodic_and_all_heading():
    angle = torch.tensor([0.0, 0.7, -2.1])
    radius = torch.tensor([4.3, 4.55, 4.8])
    anchor = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle), torch.zeros_like(radius)), dim=-1)

    torch.manual_seed(20260817)
    routes = commands_module._sample_bounded_ellipse_waypoints(
        anchor,
        num_waypoints=48,
        samples_per_lap=24,
        aspect_ratio_range=(0.94, 1.0),
        vertical_amplitude_range=(0.0, 0.15),
    )

    assert routes.shape == (3, 48, 3)
    assert torch.all(torch.linalg.vector_norm(routes[..., :2], dim=-1) <= radius.unsqueeze(1) + 1.0e-6)
    assert torch.all(torch.abs(routes[..., 2] - anchor[:, None, 2]) <= 0.15 + 1.0e-6)
    torch.testing.assert_close(routes[:, 23], anchor, rtol=0.0, atol=0.0)
    torch.testing.assert_close(routes[:, 47], anchor, rtol=0.0, atol=0.0)

    points = torch.cat((anchor.unsqueeze(1), routes[:, :24]), dim=1)
    segment = torch.diff(points, dim=1)
    heading = torch.remainder(torch.atan2(segment[..., 1], segment[..., 0]), math.tau)
    heading_octant = torch.floor(heading / (math.pi / 4.0)).long().clamp_max(7)
    for route_octants in heading_octant:
        assert torch.unique(route_octants).numel() == 8


def test_bounded_circle_spline_has_precision_speed_curvature_contract():
    anchor = torch.tensor([[4.3, 0.0, 0.0]])
    routes = commands_module._sample_bounded_ellipse_waypoints(
        anchor,
        num_waypoints=48,
        samples_per_lap=24,
        aspect_ratio_range=(1.0, 1.0),
        vertical_amplitude_range=(0.0, 0.0),
    )
    route_points = torch.cat((anchor.unsqueeze(1), routes), dim=1)
    tangent = commands_module._path_knot_tangents(route_points)
    chord_length = torch.linalg.vector_norm(torch.diff(route_points, dim=1), dim=-1, keepdim=True)
    parameter = torch.linspace(0.0, 1.0, 25).expand(1, 48, -1)
    _, derivative, second_derivative = commands_module._cubic_hermite_path(
        route_points[:, :-1],
        route_points[:, 1:],
        tangent[:, :-1] * chord_length,
        tangent[:, 1:] * chord_length,
        parameter,
    )
    curvature = torch.linalg.vector_norm(torch.cross(derivative, second_derivative, dim=-1), dim=-1)
    curvature /= torch.linalg.vector_norm(derivative, dim=-1).pow(3).clamp_min(torch.finfo(derivative.dtype).eps)

    assert 53.5 <= torch.sum(chord_length).item() <= 54.5
    assert torch.quantile(curvature, 0.99) <= 0.30


def test_bounded_figure_eight_has_exact_crossings_closure_and_all_headings():
    angle = torch.tensor([0.0, 0.7, -2.1])
    radius = torch.tensor([4.3, 4.55, 4.8])
    anchor = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle), torch.zeros_like(radius)), dim=-1)

    torch.manual_seed(20260818)
    routes = commands_module._sample_bounded_figure_eight_waypoints(
        anchor,
        num_waypoints=48,
        samples_per_lap=24,
        vertical_amplitude_range=(0.0, 0.15),
    )

    assert routes.shape == (3, 48, 3)
    assert torch.all(torch.linalg.vector_norm(routes[..., :2], dim=-1) <= radius.unsqueeze(1) + 1.0e-6)
    assert torch.all(torch.abs(routes[..., 2] - anchor[:, None, 2]) <= 0.15 + 1.0e-6)
    torch.testing.assert_close(routes[:, 23], anchor, rtol=0.0, atol=0.0)
    torch.testing.assert_close(routes[:, 47], anchor, rtol=0.0, atol=0.0)
    torch.testing.assert_close(routes[:, (5, 17), :2], torch.zeros(3, 2, 2), rtol=0.0, atol=0.0)

    points = torch.cat((anchor.unsqueeze(1), routes[:, :24]), dim=1)
    segment = torch.diff(points, dim=1)
    heading = torch.remainder(torch.atan2(segment[..., 1], segment[..., 0]), math.tau)
    heading_octant = torch.floor(heading / (math.pi / 4.0)).long().clamp_max(7)
    for route_octants in heading_octant:
        assert torch.unique(route_octants).numel() == 8


def test_bounded_template_mix_samples_both_periodic_families():
    route_count = 4096
    angle = torch.linspace(-math.pi, math.pi, route_count)
    radius = torch.linspace(4.3, 4.8, route_count)
    anchor = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle), torch.zeros_like(radius)), dim=-1)

    torch.manual_seed(42)
    routes, is_figure_eight = commands_module._sample_bounded_template_waypoints(
        anchor,
        num_waypoints=48,
        samples_per_lap=24,
        aspect_ratio_range=(0.94, 1.0),
        vertical_amplitude_range=(0.0, 0.15),
        figure_eight_probability=0.5,
    )

    assert 0.47 <= torch.mean(is_figure_eight.float()) <= 0.53
    torch.testing.assert_close(routes[:, 23], anchor, rtol=0.0, atol=0.0)
    torch.testing.assert_close(routes[:, 47], anchor, rtol=0.0, atol=0.0)
    assert torch.all(torch.linalg.vector_norm(routes[..., :2], dim=-1) <= radius.unsqueeze(1) + 1.0e-5)


def test_bounded_hard_mix_samples_eights_and_bounded_random_corners_with_finite_splines():
    route_count = 1024
    term = _hard_mix_term(route_count)

    torch.manual_seed(20260823)
    term._resample_command(torch.arange(route_count))

    assert set(torch.unique(term.route_family_id).tolist()) == {1, 2}
    figure_eight_fraction = torch.mean((term.route_family_id == 1).float())
    assert 0.45 <= figure_eight_fraction <= 0.55
    assert torch.equal(term.route_is_figure_eight, term.route_family_id == 1)
    torch.testing.assert_close(term.metrics["route_family_id"], term.route_family_id.float())

    random_route = term.route_family_id == 2
    random_waypoints = term.waypoints_e[random_route]
    random_anchor = term.route_anchor_e[random_route]
    random_offsets = random_waypoints - random_anchor.unsqueeze(1)
    random_points = torch.cat((random_anchor.unsqueeze(1), random_waypoints), dim=1)
    random_segment = torch.diff(random_points, dim=1)
    random_length = torch.linalg.vector_norm(random_segment, dim=-1)
    random_heading = torch.atan2(random_segment[..., 1], random_segment[..., 0])
    random_turn = torch.remainder(torch.diff(random_heading, dim=1) + math.pi, math.tau) - math.pi

    assert torch.all(torch.abs(random_waypoints[..., :2]) <= 4.8 + 1.0e-6)
    assert torch.all(torch.abs(random_offsets[..., 2]) <= 0.4 + 1.0e-6)
    assert torch.all(random_length >= 0.9 - 1.0e-6)
    assert torch.all(random_length <= 1.35 + 1.0e-6)
    assert torch.all(torch.abs(random_segment[..., 2]) <= 0.15 + 1.0e-6)
    assert torch.max(torch.abs(random_turn)) <= math.radians(110.0) + 2.0e-6
    assert not torch.any(
        commands_module._constant_curvature_fallback_mask(
            random_offsets, minimum_separation=0.9, maximum_heading_change=math.radians(110.0)
        )
    )
    assert torch.unique(random_waypoints.reshape(random_waypoints.shape[0], -1), dim=0).shape[0] == len(
        random_waypoints
    )

    figure_eight = term.route_family_id == 1
    torch.testing.assert_close(
        term.waypoints_e[figure_eight, 23], term.route_anchor_e[figure_eight], rtol=0.0, atol=0.0
    )

    route_points = torch.cat((term.route_anchor_e.unsqueeze(1), term.waypoints_e), dim=1)
    chord_length = torch.linalg.vector_norm(torch.diff(route_points, dim=1), dim=-1, keepdim=True)
    parameter = torch.linspace(0.0, 1.0, 33).expand(route_count, 24, -1)
    spline_position, derivative, second_derivative = commands_module._cubic_hermite_path(
        route_points[:, :-1],
        route_points[:, 1:],
        term.path_knot_tangent_e[:, :-1] * chord_length,
        term.path_knot_tangent_e[:, 1:] * chord_length,
        parameter,
    )
    speed = torch.linalg.vector_norm(derivative, dim=-1)
    curvature = torch.linalg.vector_norm(torch.cross(derivative, second_derivative, dim=-1), dim=-1)
    curvature /= speed.pow(3).clamp_min(torch.finfo(speed.dtype).eps)
    assert torch.all(torch.abs(spline_position[..., :2]) <= 4.8 + 1.0e-5)
    assert torch.all(torch.abs(spline_position[..., 2]) <= 0.4 + 1.0e-5)
    assert torch.isfinite(curvature).all()
    assert torch.min(speed) > 0.5

    midpoint, tangent, midpoint_curvature = term.sample_path_at_arc_length(0.5 * term.path_total_length)
    speed_reference = term.compute_path_speed_reference(3.5, 6.0, 6.0)
    assert torch.isfinite(midpoint).all()
    assert torch.isfinite(tangent).all()
    assert torch.isfinite(midpoint_curvature).all()
    assert torch.all((speed_reference > 0.0) & (speed_reference <= 3.5))


def test_bounded_hard_mix_replaces_constant_curvature_fallback_with_figure_eight(monkeypatch):
    route_count = 8
    term = _hard_mix_term(route_count, figure_eight_probability=0.0, spline_enabled=False)
    segment_id = torch.arange(24, dtype=torch.float32)
    heading = math.radians(110.0) * segment_id
    fallback = 0.9 * torch.stack((torch.cos(heading), torch.sin(heading), torch.zeros_like(heading)), dim=-1).cumsum(
        dim=0
    )
    monkeypatch.setattr(
        commands_module,
        "_sample_bounded_waypoint_offsets",
        lambda *args, **kwargs: fallback.unsqueeze(0).expand(route_count, -1, -1).clone(),
    )

    torch.manual_seed(20260824)
    term._resample_command(torch.arange(route_count))

    assert torch.all(term.route_family_id == 1)
    assert torch.all(term.route_is_figure_eight)
    torch.testing.assert_close(term.waypoints_e[:, 23], term.route_anchor_e, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("sampler", "tangent_function", "curvature_limit"),
    (
        (commands_module._sample_bounded_ellipse_waypoints, commands_module._bounded_ellipse_path_tangents, 0.27),
        (
            commands_module._sample_bounded_figure_eight_waypoints,
            commands_module._bounded_figure_eight_path_tangents,
            0.51,
        ),
    ),
)
def test_bounded_template_analytic_tangents_are_periodic_and_remove_knot_spikes(
    sampler, tangent_function, curvature_limit
):
    route_count = 256
    angle = torch.linspace(-math.pi, math.pi, route_count)
    radius = torch.linspace(4.3, 4.8, route_count)
    anchor = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle), torch.zeros_like(radius)), dim=-1)
    torch.manual_seed(20260818)
    if sampler is commands_module._sample_bounded_ellipse_waypoints:
        waypoints = sampler(anchor, 48, 24, (0.94, 1.0), (0.0, 0.15))
    else:
        waypoints = sampler(anchor, 48, 24, (0.0, 0.15))
    route_points = torch.cat((anchor.unsqueeze(1), waypoints), dim=1)
    tangent = tangent_function(anchor, waypoints, 24)

    torch.testing.assert_close(tangent[:, 0], tangent[:, 24], atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(tangent[:, 0], tangent[:, 48], atol=1.0e-6, rtol=0.0)
    chord_length = torch.linalg.vector_norm(torch.diff(route_points, dim=1), dim=-1, keepdim=True)
    parameter = torch.linspace(0.0, 1.0, 17).expand(route_count, 48, -1)
    _, derivative, second_derivative = commands_module._cubic_hermite_path(
        route_points[:, :-1],
        route_points[:, 1:],
        tangent[:, :-1] * chord_length,
        tangent[:, 1:] * chord_length,
        parameter,
    )
    curvature = torch.linalg.vector_norm(torch.cross(derivative, second_derivative, dim=-1), dim=-1)
    curvature /= torch.linalg.vector_norm(derivative, dim=-1).pow(3).clamp_min(torch.finfo(derivative.dtype).eps)

    assert torch.isfinite(curvature).all()
    assert torch.quantile(curvature, 0.99) <= curvature_limit


@pytest.mark.parametrize(
    ("name", "value", "message"),
    (
        ("random_waypoint_count", 47, "multiple of samples_per_lap"),
        ("samples_per_lap", 3, "at least four"),
        ("aspect_ratio_range", (0.0, 1.0), "aspect_ratio_range"),
        ("aspect_ratio_range", (0.95, 1.01), "aspect_ratio_range"),
        ("vertical_amplitude_range", (-0.01, 0.15), "vertical_amplitude_range"),
    ),
)
def test_bounded_ellipse_cfg_rejects_invalid_geometry(name, value, message):
    cfg = commands_module.WaypointSequenceCommandCfg(
        randomize_waypoints=True,
        route_family="bounded_ellipse",
        random_waypoint_count=48,
    )
    setattr(cfg, name, value)

    with pytest.raises(ValueError, match=message):
        commands_module.WaypointSequenceCommand._validate_cfg(cfg)


def test_bounded_template_mix_rejects_invalid_probability_and_lap_quarters():
    cfg = commands_module.WaypointSequenceCommandCfg(
        randomize_waypoints=True,
        route_family="bounded_template_mix",
        random_waypoint_count=24,
        samples_per_lap=12,
    )
    commands_module.WaypointSequenceCommand._validate_cfg(cfg)

    cfg.figure_eight_probability = 1.01
    with pytest.raises(ValueError, match="figure_eight_probability"):
        commands_module.WaypointSequenceCommand._validate_cfg(cfg)

    cfg.figure_eight_probability = 0.5
    cfg.samples_per_lap = 6
    with pytest.raises(ValueError, match="divisible by four"):
        commands_module.WaypointSequenceCommand._validate_cfg(cfg)


def test_smooth_random_routes_are_heading_balanced_and_avoid_reversals():
    torch.manual_seed(20260814)
    route_count = 4096
    routes = commands_module._sample_bounded_waypoint_offsets(
        num_routes=route_count,
        num_waypoints=12,
        lower=torch.tensor([-4.0, -4.0, -0.4]),
        upper=torch.tensor([4.0, 4.0, 0.4]),
        minimum_separation=0.75,
        max_sampling_attempts=16,
        maximum_separation=1.5,
        maximum_heading_change=math.radians(60.0),
        maximum_vertical_step=0.15,
        nominal_heading_change=math.radians(40.0),
        route_sampling_attempts=16,
    )

    points = torch.cat((torch.zeros(route_count, 1, 3), routes), dim=1)
    segment = torch.diff(points, dim=1)
    segment_length = torch.linalg.vector_norm(segment, dim=-1)
    heading = torch.atan2(segment[..., 1], segment[..., 0])
    heading_change = torch.remainder(torch.diff(heading, dim=1) + math.pi, 2.0 * math.pi) - math.pi
    absolute_heading_change_deg = torch.rad2deg(torch.abs(heading_change))

    assert torch.all(routes >= torch.tensor([-4.0, -4.0, -0.4]))
    assert torch.all(routes <= torch.tensor([4.0, 4.0, 0.4]))
    assert torch.all(segment_length >= 0.75 - 1.0e-6)
    assert torch.all(segment_length <= 1.5 + 1.0e-6)
    assert torch.all(torch.abs(segment[..., 2]) <= 0.15 + 1.0e-6)
    assert torch.max(absolute_heading_change_deg) <= 60.001
    assert torch.quantile(absolute_heading_change_deg, 0.95) <= 45.0
    assert 20.0 <= torch.mean(absolute_heading_change_deg) <= 32.0
    assert 12.0 <= torch.mean(torch.sum(segment_length, dim=1)) <= 15.0

    initial_heading = torch.remainder(heading[:, 0], 2.0 * math.pi)
    heading_octant = torch.floor(initial_heading / (math.pi / 4.0)).long().clamp_max(7)
    octant_count = torch.bincount(heading_octant, minlength=8)
    assert torch.all(octant_count >= 0.10 * route_count)
    assert torch.all(octant_count <= 0.15 * route_count)


def test_random_heading_interval_makes_long_edges_between_random_corners():
    torch.manual_seed(20260822)
    routes = commands_module._sample_bounded_waypoint_offsets(
        num_routes=512,
        num_waypoints=12,
        lower=torch.tensor([-50.0, -50.0, 0.0]),
        upper=torch.tensor([50.0, 50.0, 0.0]),
        minimum_separation=0.99,
        max_sampling_attempts=8,
        maximum_separation=1.01,
        maximum_heading_change=math.radians(110.0),
        maximum_vertical_step=0.0,
        nominal_heading_change=math.radians(80.0),
        route_sampling_attempts=1,
        random_heading_change_interval=3,
    )
    points = torch.cat((torch.zeros(512, 1, 3), routes), dim=1)
    segment = torch.diff(points, dim=1)
    heading = torch.atan2(segment[..., 1], segment[..., 0])
    heading_change = torch.remainder(torch.diff(heading, dim=1) + math.pi, math.tau) - math.pi
    random_corner = torch.arange(1, 12) % 3 == 0

    assert torch.max(torch.abs(heading_change[:, ~random_corner])) < 2.0e-5
    assert 35.0 <= torch.mean(torch.rad2deg(torch.abs(heading_change[:, random_corner]))) <= 45.0
    assert torch.max(torch.abs(heading_change)) <= math.radians(80.0) + 2.0e-6


def test_route_attempt_selection_can_preserve_unbiased_random_corners():
    route_count = 128
    angle = torch.arange(route_count) * (math.tau / route_count) - math.pi
    radius = torch.linspace(4.3, 4.8, route_count)
    anchor = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle), torch.zeros_like(radius)), dim=-1)
    planar_margin, vertical_margin = commands_module._hard_route_spline_sampling_margin(
        0.9, 1.35, math.radians(110.0), 0.15, 1.0
    )
    lower = torch.tensor([-4.8 + planar_margin, -4.8 + planar_margin, -0.4 + vertical_margin]) - anchor
    upper = torch.tensor([4.8 - planar_margin, 4.8 - planar_margin, 0.4 - vertical_margin]) - anchor

    def sample(select_smoothest: bool) -> torch.Tensor:
        torch.manual_seed(20260825)
        return commands_module._sample_bounded_waypoint_offsets(
            num_routes=route_count,
            num_waypoints=24,
            lower=lower,
            upper=upper,
            minimum_separation=0.9,
            max_sampling_attempts=32,
            maximum_separation=1.35,
            maximum_heading_change=math.radians(110.0),
            maximum_vertical_step=0.15,
            nominal_heading_change=math.radians(80.0),
            route_sampling_attempts=16,
            random_heading_change_interval=3,
            independent_initial_heading_attempts=True,
            select_smoothest_route_attempt=select_smoothest,
        )

    def scheduled_corner_mean(routes: torch.Tensor) -> torch.Tensor:
        points = torch.cat((torch.zeros(route_count, 1, 3), routes), dim=1)
        segment = torch.diff(points, dim=1)
        heading = torch.atan2(segment[..., 1], segment[..., 0])
        turn = torch.remainder(torch.diff(heading, dim=1) + math.pi, math.tau) - math.pi
        scheduled_corner = torch.arange(1, 24) % 3 == 0
        return torch.mean(torch.rad2deg(torch.abs(turn[:, scheduled_corner])))

    smoothest = sample(select_smoothest=True)
    first_valid = sample(select_smoothest=False)
    smoothest_mean = scheduled_corner_mean(smoothest)
    first_valid_mean = scheduled_corner_mean(first_valid)

    assert 38.0 <= first_valid_mean <= 48.0
    assert first_valid_mean >= smoothest_mean + 5.0
    assert not torch.equal(first_valid, smoothest)


def test_direct_straight_foundation_sampler_is_planar_and_avoids_curved_fallback():
    route_count = 512
    lower = torch.tensor([-12.5, -12.5, 0.0])
    upper = torch.tensor([12.5, 12.5, 0.0])

    torch.manual_seed(20260821)
    routes = commands_module._sample_bounded_waypoint_offsets(
        num_routes=route_count,
        num_waypoints=24,
        lower=lower,
        upper=upper,
        minimum_separation=0.49,
        max_sampling_attempts=8,
        maximum_separation=0.51,
        maximum_heading_change=math.radians(5.0),
        maximum_vertical_step=0.0,
        nominal_heading_change=0.0,
        route_sampling_attempts=4,
    )

    points = torch.cat((torch.zeros(route_count, 1, 3), routes), dim=1)
    segment = torch.diff(points, dim=1)
    segment_length = torch.linalg.vector_norm(segment, dim=-1)
    heading = torch.atan2(segment[..., 1], segment[..., 0])
    heading_change = torch.remainder(torch.diff(heading, dim=1) + math.pi, 2.0 * math.pi) - math.pi

    assert torch.all(routes >= lower)
    assert torch.all(routes <= upper)
    assert torch.all(segment_length >= 0.49 - 1.0e-6)
    assert torch.all(segment_length <= 0.51 + 1.0e-6)
    torch.testing.assert_close(routes[..., 2], torch.zeros_like(routes[..., 2]))
    # The sampler's curved feasibility fallback turns by up to five degrees;
    # this tight bound proves the configured route stayed on its sampled ray.
    assert torch.max(torch.abs(heading_change)) < 1.0e-5
    assert torch.all(torch.sum(segment_length, dim=1) >= 24 * 0.49 - 1.0e-5)
    assert torch.all(torch.sum(segment_length, dim=1) <= 24 * 0.51 + 1.0e-5)
    initial_heading = torch.remainder(heading[:, 0], 2.0 * math.pi)
    heading_octant = torch.floor(initial_heading / (math.pi / 4.0)).long().clamp_max(7)
    assert torch.unique(heading_octant).numel() == 8


def test_route_family_switch_uses_same_w24_buffer_on_next_resample():
    cfg = commands_module.WaypointSequenceCommandCfg(
        randomize_waypoints=True,
        regenerate_on_completion=False,
        random_waypoint_count=24,
        route_family="random_walk",
        samples_per_lap=24,
        aspect_ratio_range=(0.94, 1.0),
        vertical_amplitude_range=(0.0, 0.0),
        figure_eight_probability=0.0,
        random_waypoint_ranges=commands_module.WaypointSequenceCommandCfg.Ranges(
            pos_x=(-12.5, 12.5), pos_y=(-12.5, 12.5), pos_z=(0.0, 0.0)
        ),
        minimum_waypoint_separation=0.49,
        maximum_waypoint_separation=0.51,
        nominal_heading_change=0.0,
        maximum_heading_change=math.radians(5.0),
        maximum_vertical_step=0.0,
    )
    pose = torch.tensor([[4.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    term = object.__new__(commands_module.WaypointSequenceCommand)
    term.cfg = cfg
    term.robot = SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=pose)))
    term._env = SimpleNamespace(num_envs=1, device="cpu", scene=SimpleNamespace(env_origins=torch.zeros(1, 3)))
    term.route_anchor_e = torch.zeros(1, 3)
    term.waypoints_e = torch.zeros(1, 24, 3)
    term.current_index = torch.zeros(1, dtype=torch.long)
    term.completed = torch.zeros(1, dtype=torch.bool)
    term.previous_distance_sq = torch.zeros(1)

    torch.manual_seed(20260821)
    term._resample_command(torch.tensor([0]))
    straight_waypoints = term.waypoints_e.clone()
    straight_segment = torch.diff(torch.cat((pose[:, None, :3], straight_waypoints), dim=1), dim=1)
    straight_heading = torch.atan2(straight_segment[..., 1], straight_segment[..., 0])
    straight_turn = torch.remainder(torch.diff(straight_heading, dim=1) + math.pi, 2.0 * math.pi) - math.pi
    assert torch.max(torch.abs(straight_turn)) < 1.0e-5

    cfg.route_family = "bounded_template_mix"
    torch.testing.assert_close(term.waypoints_e, straight_waypoints)
    term._resample_command(torch.tensor([0]))

    assert term.waypoints_e.shape == straight_waypoints.shape == (1, 24, 3)
    assert not torch.equal(term.waypoints_e, straight_waypoints)
    torch.testing.assert_close(term.waypoints_e[:, 23], pose[:, :3], rtol=0.0, atol=0.0)
    assert not term.route_is_figure_eight.item()


def test_random_route_fallback_preserves_exact_minimum_and_maximum_spacing():
    spacing = math.sqrt(2.0)
    routes = commands_module._sample_bounded_waypoint_offsets(
        num_routes=8,
        num_waypoints=4,
        lower=torch.tensor([-1.0, -1.0, 0.0]),
        upper=torch.tensor([1.0, 1.0, 0.0]),
        minimum_separation=spacing,
        max_sampling_attempts=1,
        maximum_separation=spacing,
    )

    points = torch.cat((torch.zeros(routes.shape[0], 1, 3), routes), dim=1)
    separation = torch.linalg.vector_norm(torch.diff(points, dim=1), dim=-1)
    torch.testing.assert_close(separation, torch.full_like(separation, spacing), atol=1.0e-6, rtol=0.0)
    assert torch.all(routes >= torch.tensor([-1.0, -1.0, 0.0]))
    assert torch.all(routes <= torch.tensor([1.0, 1.0, 0.0]))


def test_random_route_cfg_rejects_a_separation_that_cannot_be_guaranteed():
    cfg = commands_module.WaypointSequenceCommandCfg(
        randomize_waypoints=True,
        random_waypoint_ranges=commands_module.WaypointSequenceCommandCfg.Ranges(
            pos_x=(-1.0, 1.0), pos_y=(-1.0, 1.0), pos_z=(0.0, 0.0)
        ),
        minimum_waypoint_separation=1.5,
    )

    with pytest.raises(ValueError, match="half the random waypoint box diagonal"):
        commands_module.WaypointSequenceCommand._validate_cfg(cfg)


def test_smooth_random_route_cfg_rejects_invalid_attempt_and_turn_limits():
    cfg = commands_module.WaypointSequenceCommandCfg(
        randomize_waypoints=True,
        minimum_waypoint_separation=0.75,
        maximum_waypoint_separation=1.5,
        maximum_heading_change=math.radians(60.0),
        nominal_heading_change=math.radians(40.0),
    )
    cfg.route_sampling_attempts = 0
    with pytest.raises(ValueError, match="route_sampling_attempts must be positive"):
        commands_module.WaypointSequenceCommand._validate_cfg(cfg)

    cfg.route_sampling_attempts = 4
    cfg.random_heading_change_interval = 0
    with pytest.raises(ValueError, match="random_heading_change_interval must be a positive integer"):
        commands_module.WaypointSequenceCommand._validate_cfg(cfg)

    cfg.random_heading_change_interval = 1
    cfg.nominal_heading_change = math.radians(61.0)
    with pytest.raises(ValueError, match="nominal_heading_change cannot exceed maximum_heading_change"):
        commands_module.WaypointSequenceCommand._validate_cfg(cfg)


def test_route_regeneration_requires_randomized_waypoints():
    cfg = commands_module.WaypointSequenceCommandCfg(regenerate_on_completion=True)

    with pytest.raises(ValueError, match="requires randomize_waypoints=True"):
        commands_module.WaypointSequenceCommand._validate_cfg(cfg)


def test_indexed_cross_track_projection_stays_on_active_self_intersection_branch():
    # Segment zero follows y=x and segment two follows y=-x. The robot lies on
    # segment zero, but progress index two must keep it assigned to segment two.
    position = torch.tensor([[0.5, 0.5, 0.0]])
    anchor = torch.tensor([[-2.0, -2.0, 0.0]])
    waypoints = torch.tensor([[[2.0, 2.0, 0.0], [-2.0, 2.0, 0.0], [2.0, -2.0, 0.0]]])

    projection, error = commands_module._indexed_segment_projection(
        position, anchor, waypoints, active_index=torch.tensor([2])
    )

    torch.testing.assert_close(projection, torch.zeros_like(projection))
    torch.testing.assert_close(error, position)
    torch.testing.assert_close(torch.linalg.vector_norm(error, dim=-1), torch.tensor([math.sqrt(0.5)]))


def test_cubic_route_interpolates_waypoint_with_shared_tangent_direction():
    route_points = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]]])
    tangent = commands_module._path_knot_tangents(route_points)
    chord_length = torch.linalg.vector_norm(torch.diff(route_points, dim=1), dim=-1, keepdim=True)
    derivative = tangent * torch.cat((chord_length[:, :1], chord_length), dim=1) * 0.75

    position, path_derivative, _ = commands_module._cubic_hermite_path(
        route_points[:, :-1],
        route_points[:, 1:],
        derivative[:, :-1],
        derivative[:, 1:],
        torch.tensor([[1.0, 0.0]]),
    )

    torch.testing.assert_close(position[:, 0], route_points[:, 1])
    torch.testing.assert_close(position[:, 1], route_points[:, 1])
    derivative_direction = torch.nn.functional.normalize(path_derivative, dim=-1)
    torch.testing.assert_close(derivative_direction[:, 0], derivative_direction[:, 1])
    torch.testing.assert_close(derivative_direction[:, 0], tangent[:, 1])


def test_spline_command_projects_on_indexed_curve_and_samples_arc_length():
    waypoints = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]]])
    term = _spline_term(torch.zeros(1, 3), waypoints, current_index=torch.tensor([1]))
    start, end, start_derivative, end_derivative = term._path_segment_controls(term.current_index)
    on_path, _, _ = commands_module._cubic_hermite_path(
        start, end, start_derivative, end_derivative, torch.tensor([0.35])
    )
    term.robot.data.body_link_pose_w.torch[:, :3] = on_path + torch.tensor([[0.0, 0.0, 0.10]])
    term._env.common_step_counter += 1

    torch.testing.assert_close(term.path_projection_e, on_path, atol=2.0e-5, rtol=0.0)
    torch.testing.assert_close(term.path_cross_track_error_e, torch.tensor([[0.0, 0.0, 0.10]]), atol=2.0e-5, rtol=0.0)
    torch.testing.assert_close(term.path_cross_track_distance, torch.tensor([0.10]), atol=2.0e-5, rtol=0.0)
    torch.testing.assert_close(torch.linalg.vector_norm(term.path_tangent_e, dim=-1), torch.ones(1))
    assert torch.isfinite(term.path_curvature_e).all()
    assert term.path_segment_start_length[0, 1] < term.path_progress[0] < term.path_segment_start_length[0, 2]

    segment_end_length = term.path_segment_start_length + term.path_segment_length
    sampled_position, sampled_tangent, sampled_curvature = term.sample_path_at_arc_length(segment_end_length)
    torch.testing.assert_close(sampled_position, waypoints, atol=2.0e-5, rtol=0.0)
    torch.testing.assert_close(torch.linalg.vector_norm(sampled_tangent, dim=-1), torch.ones(1, 3))
    assert torch.isfinite(sampled_curvature).all()
    assert all(value.shape == (1, 2, 3) for value in term.path_preview_e((0.75, 1.50)))
    assert term.command.shape == (1, 6)


def test_spline_acceptance_and_precision_parameters_remain_runtime_mutable():
    waypoints = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.5, 0.0]]])
    term = _spline_term(torch.tensor([[0.75, 0.0, 0.0]]), waypoints)
    term.cfg.acceptance_radius = 0.20

    term._update_command()
    torch.testing.assert_close(term.current_index, torch.zeros(1, dtype=torch.long))

    term.cfg.acceptance_radius = 0.30
    term._env.common_step_counter += 1
    term._update_command()

    torch.testing.assert_close(term.current_index, torch.ones(1, dtype=torch.long))
    torch.testing.assert_close(term.previous_path_progress, term.path_progress)
    assert torch.isfinite(term.previous_path_progress).all()


@pytest.mark.parametrize("lateral_miss", (0.16, 0.20))
def test_progressive_spline_advances_after_a_bounded_precision_miss(lateral_miss):
    waypoints = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    term = _spline_term(torch.tensor([[1.05, lateral_miss, 0.0]]), waypoints)
    term.cfg.acceptance_radius = 0.15
    term.cfg.spline_progressive_advancement = True
    term.cfg.spline_plane_crossing_lateral_tolerance = 0.25

    next_index, completed, passed, hits, _, misses, miss_distance, _ = term._route_progress_preview(
        term.robot.data.body_link_pose_w.torch[:, :3]
    )

    torch.testing.assert_close(next_index, torch.ones(1, dtype=torch.long))
    assert not completed.item()
    torch.testing.assert_close(passed, torch.ones(1, dtype=torch.long))
    torch.testing.assert_close(hits, torch.zeros(1, dtype=torch.long))
    torch.testing.assert_close(misses, torch.ones(1, dtype=torch.long))
    torch.testing.assert_close(miss_distance, torch.tensor([lateral_miss]))


@pytest.mark.parametrize("position", ((0.85, 0.0, 0.0), (1.05, 0.30, 0.0), (0.0, 0.20, 0.0)))
def test_progressive_spline_does_not_advance_backward_or_laterally(position):
    waypoints = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    term = _spline_term(torch.tensor([position]), waypoints)
    term.cfg.acceptance_radius = 0.10
    term.cfg.spline_progressive_advancement = True
    term.cfg.spline_plane_crossing_lateral_tolerance = 0.25

    next_index, completed, passed, *_ = term._route_progress_preview(term.robot.data.body_link_pose_w.torch[:, :3])

    torch.testing.assert_close(next_index, torch.zeros(1, dtype=torch.long))
    assert not completed.item()
    torch.testing.assert_close(passed, torch.zeros(1, dtype=torch.long))


def test_progressive_spline_can_cross_multiple_knots_in_one_step_without_reward_jump():
    waypoints = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]])
    term = _spline_term(torch.tensor([[3.10, 0.0, 0.0]]), waypoints)
    term.cfg.acceptance_radius = 0.05
    term.cfg.spline_progressive_advancement = True
    term.cfg.spline_plane_crossing_lateral_tolerance = 0.25
    term.cfg.spline_max_waypoint_advances_per_step = 4

    preview_index, completed, passed, *_ = term._route_progress_preview(term.robot.data.body_link_pose_w.torch[:, :3])
    torch.testing.assert_close(preview_index, torch.tensor([3]))
    torch.testing.assert_close(passed, torch.tensor([3]))
    assert not completed.item()

    term._update_command()
    torch.testing.assert_close(term.current_index, torch.tensor([3]))
    torch.testing.assert_close(term.previous_path_progress, term.path_progress)
    torch.testing.assert_close(term.path_progress - term.previous_path_progress, torch.zeros(1))


def test_progressive_figure_eight_projection_cannot_jump_to_the_other_crossing_branch():
    anchor = torch.tensor([[4.5, 0.0, 0.0]])
    waypoints = commands_module._sample_bounded_figure_eight_waypoints(
        anchor,
        num_waypoints=24,
        samples_per_lap=24,
        vertical_amplitude_range=(0.0, 0.0),
    )
    crossing = waypoints[:, 5].clone()
    term = _spline_term(crossing, waypoints, current_index=torch.tensor([5]))
    term.route_anchor_e.copy_(anchor)
    term.route_is_figure_eight = torch.ones(1, dtype=torch.bool)
    term.cfg.route_family = "bounded_template_mix"
    term.cfg.samples_per_lap = 24
    term.cfg.spline_progressive_advancement = True
    term.cfg.spline_plane_crossing_lateral_tolerance = 0.25
    term.cfg.acceptance_radius = 0.01
    term._rebuild_spline_path(torch.arange(1))

    _ = term.path_projection_e
    assert term._path_projection_segment_index.item() in (5, 6)
    preview_index, completed, passed, *_ = term._route_progress_preview(crossing)
    torch.testing.assert_close(preview_index, torch.tensor([6]))
    torch.testing.assert_close(passed, torch.ones(1, dtype=torch.long))
    assert not completed.item()


def test_path_speed_reference_uses_future_curve_braking_envelope_and_live_cache():
    waypoints = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]).expand(2, -1, -1)
    term = _spline_term(torch.zeros(2, 3), waypoints)
    term._path_progress.copy_(torch.tensor([10.0, 10.0]))
    term.path_total_length.copy_(torch.tensor([100.0, 10.5]))
    term._path_curvature_e.copy_(torch.tensor([[0.0, 0.0, 0.25], [0.0, 0.0, 0.25]]))
    term._update_cached_path_state = lambda: None
    sample_count = 0

    def sample_path(arc_length):
        nonlocal sample_count
        sample_count += 1
        curvature = torch.zeros(*arc_length.shape, 3)
        curvature[..., 2] = 1.0
        return torch.zeros_like(curvature), torch.zeros_like(curvature), curvature

    term.sample_path_at_arc_length = sample_path
    term.cfg.target_cruise_speed = 10.0
    term.cfg.maximum_lateral_acceleration = 4.0
    term.cfg.maximum_braking_acceleration = 1.0
    term.cfg.speed_lookahead_distances = (2.0,)

    first = term.path_speed_reference.clone()
    second = term.path_speed_reference.clone()
    # Future curvature permits sqrt(v_curve^2 + 2*a_brake*d) = sqrt(8),
    # while the short route is governed by its sqrt(2*a_brake*remaining) stop.
    torch.testing.assert_close(first, torch.tensor([math.sqrt(8.0), 1.0]))
    torch.testing.assert_close(second, first)
    assert sample_count == 1

    term.cfg.target_cruise_speed = 2.5
    torch.testing.assert_close(term.path_speed_reference, torch.tensor([2.5, 1.0]))
    assert sample_count == 2
    term._env.common_step_counter += 1
    _ = term.path_speed_reference
    assert sample_count == 3


def test_plane_completed_route_records_total_arc_length_and_completion_time(monkeypatch):
    waypoints = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    term = _spline_term(torch.tensor([[2.05, 0.18, 0.0]]), waypoints, current_index=torch.tensor([1]))
    term.cfg.acceptance_radius = 0.10
    term.cfg.spline_progressive_advancement = True
    term.cfg.spline_plane_crossing_lateral_tolerance = 0.25
    term._env.step_dt = 0.02
    term._env.episode_length_buf = torch.tensor([250])
    term.metrics = {}
    term._initialize_episode_metrics()
    zero = torch.zeros(1, 3)
    monkeypatch.setattr(commands_module, "payload_transverse_velocity_b", lambda _env: zero)
    monkeypatch.setattr(commands_module, "total_swing_angle", lambda _env: torch.zeros(1))
    monkeypatch.setattr(commands_module, "link_lin_vel_w", lambda _env, *_args, **_kwargs: zero)
    monkeypatch.setattr(commands_module, "cable_relative_separation", lambda _env: torch.zeros(1))
    monkeypatch.setattr(commands_module, "cable_joint_error", lambda _env: torch.zeros(1))

    term._record_current_metrics()

    torch.testing.assert_close(term.metrics["waypoint_precision_hits"], torch.zeros(1))
    torch.testing.assert_close(term.metrics["waypoint_precision_misses"], torch.ones(1))
    torch.testing.assert_close(term.metrics["route_arc_length_traversed"], term.path_total_length)
    torch.testing.assert_close(term.metrics["waypoint_completion_time"], torch.tensor([5.0]))
    term._update_command()
    assert term.completed.item()


@pytest.mark.parametrize(
    ("name", "value"),
    (("spline_projection_samples", 1), ("spline_tangent_scale", 0.0), ("spline_tangent_scale", 1.1)),
)
def test_spline_command_cfg_rejects_invalid_geometry(name, value):
    cfg = commands_module.WaypointSequenceCommandCfg(spline_enabled=True)
    setattr(cfg, name, value)

    with pytest.raises(ValueError):
        commands_module.WaypointSequenceCommand._validate_cfg(cfg)


def test_cross_track_properties_and_command_keep_per_environment_shapes():
    pose = torch.tensor(
        [
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    term = object.__new__(commands_module.WaypointSequenceCommand)
    term.robot = SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=pose)))
    term._env = SimpleNamespace(num_envs=2, device="cpu", scene=SimpleNamespace(env_origins=torch.zeros(2, 3)))
    term.route_anchor_e = torch.tensor([[-2.0, -2.0, 0.0], [0.0, 0.0, 0.0]])
    term.waypoints_e = torch.tensor(
        [
            [[2.0, 2.0, 0.0], [-2.0, 2.0, 0.0], [2.0, -2.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 3.0, 0.0]],
        ]
    )
    term.current_index = torch.tensor([2, 2])

    assert term.command.shape == (2, 6)
    assert term.active_segment_projection_e.shape == (2, 3)
    assert term.cross_track_error_e.shape == (2, 3)
    assert term.cross_track_distance.shape == (2,)
    torch.testing.assert_close(term.cross_track_distance, torch.tensor([math.sqrt(0.5), 0.0]))


def test_randomized_waypoint_command_keeps_fixed_count_and_six_value_interface():
    cfg = commands_module.WaypointSequenceCommandCfg(
        randomize_waypoints=True,
        random_waypoint_count=7,
        maximum_waypoint_separation=2.0,
    )
    pose = torch.tensor(
        [
            [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    term = object.__new__(commands_module.WaypointSequenceCommand)
    term.cfg = cfg
    term.robot = SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=pose)))
    term._env = SimpleNamespace(num_envs=2, device="cpu", scene=SimpleNamespace(env_origins=torch.zeros(2, 3)))
    term.route_anchor_e = torch.zeros(2, 3)
    term.waypoints_e = torch.zeros(2, 7, 3)
    term.current_index = torch.zeros(2, dtype=torch.long)
    term.completed = torch.zeros(2, dtype=torch.bool)
    term.previous_distance_sq = torch.zeros(2)

    torch.manual_seed(99)
    term._resample_command(torch.tensor([0, 1]))

    assert term.waypoints_e.shape == (2, 7, 3)
    assert term.command.shape == (2, 6)


def test_bounded_ellipse_command_derives_centered_route_from_post_reset_anchor():
    cfg = commands_module.WaypointSequenceCommandCfg(
        randomize_waypoints=True,
        route_family="bounded_ellipse",
        random_waypoint_count=48,
        samples_per_lap=24,
    )
    pose = torch.tensor(
        [
            [4.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, -4.8, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    term = object.__new__(commands_module.WaypointSequenceCommand)
    term.cfg = cfg
    term.robot = SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=pose)))
    term._env = SimpleNamespace(num_envs=2, device="cpu", scene=SimpleNamespace(env_origins=torch.zeros(2, 3)))
    term.route_anchor_e = torch.zeros(2, 3)
    term.waypoints_e = torch.zeros(2, 48, 3)
    term.current_index = torch.zeros(2, dtype=torch.long)
    term.completed = torch.zeros(2, dtype=torch.bool)
    term.previous_distance_sq = torch.zeros(2)

    torch.manual_seed(20260817)
    term._resample_command(torch.tensor([0, 1]))

    torch.testing.assert_close(term.route_anchor_e, pose[:, :3])
    torch.testing.assert_close(term.waypoints_e[:, 23], pose[:, :3], rtol=0.0, atol=0.0)
    torch.testing.assert_close(term.waypoints_e[:, 47], pose[:, :3], rtol=0.0, atol=0.0)
    assert torch.all(torch.abs(term.waypoints_e[..., :2]) <= 4.8 + 1.0e-6)
    assert torch.all(term.previous_distance_sq > 0.0)
    assert term.command.shape == (2, 6)


def test_waypoint_advance_updates_next_target_distance_without_torch_any(monkeypatch):
    pose = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    term = object.__new__(commands_module.WaypointSequenceCommand)
    term.cfg = SimpleNamespace(acceptance_radius=0.25)
    term.robot = SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=pose)))
    term._env = SimpleNamespace(num_envs=3, device="cpu", scene=SimpleNamespace(env_origins=torch.zeros(3, 3)))
    term.waypoints_e = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ]
    )
    term.current_index = torch.tensor([0, 0, 1])
    term.completed = torch.zeros(3, dtype=torch.bool)
    term.previous_distance_sq = torch.full((3,), -1.0)
    with monkeypatch.context() as sync_guard:
        sync_guard.setattr(commands_module.torch, "any", lambda *_args, **_kwargs: pytest.fail("torch.any host sync"))
        term._update_command()

    torch.testing.assert_close(term.current_index, torch.tensor([1, 0, 1]))
    torch.testing.assert_close(term.completed, torch.tensor([False, False, True]))
    torch.testing.assert_close(term.previous_distance_sq, torch.tensor([4.0, 2.0, 0.0]))


def test_completed_random_route_regenerates_without_reward_potential_jump():
    cfg = commands_module.WaypointSequenceCommandCfg(
        randomize_waypoints=True,
        regenerate_on_completion=True,
        random_waypoint_count=2,
        minimum_waypoint_separation=0.75,
        maximum_waypoint_separation=1.5,
        acceptance_radius=0.25,
    )
    pose = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    term = object.__new__(commands_module.WaypointSequenceCommand)
    term.cfg = cfg
    term.robot = SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=pose)))
    term._env = SimpleNamespace(num_envs=1, device="cpu", scene=SimpleNamespace(env_origins=torch.zeros(1, 3)))
    term.route_anchor_e = torch.zeros(1, 3)
    term.waypoints_e = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    term.current_index = torch.ones(1, dtype=torch.long)
    term.completed = torch.zeros(1, dtype=torch.bool)
    term.previous_distance_sq = torch.ones(1)

    torch.manual_seed(20260814)
    term._update_command()

    torch.testing.assert_close(term.route_anchor_e, pose[:, :3])
    torch.testing.assert_close(term.current_index, torch.zeros(1, dtype=torch.long))
    assert not term.completed.item()
    new_distance_sq = torch.sum(torch.square(term.command[:, :3] - pose[:, :3]), dim=-1)
    torch.testing.assert_close(term.previous_distance_sq, new_distance_sq)
    assert 0.75**2 - 1.0e-6 <= new_distance_sq.item() <= 1.5**2 + 1.0e-6


def test_streaming_route_metrics_count_arrivals_intervals_and_distance():
    pose = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    term = object.__new__(commands_module.WaypointSequenceCommand)
    term._env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.01,
        episode_length_buf=torch.tensor([100]),
        scene=SimpleNamespace(env_origins=torch.zeros(1, 3)),
    )
    term.robot = SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=pose)))
    term.cfg = SimpleNamespace(acceptance_radius=0.5, regenerate_on_completion=True)
    term.metrics = {}
    term._initialize_episode_metrics()
    term.route_anchor_e = torch.zeros(1, 3)
    term.waypoints_e = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    term.current_index = torch.zeros(1, dtype=torch.long)
    term.completed = torch.zeros(1, dtype=torch.bool)
    recorded_completion = []
    term._update_cross_track_metrics = lambda _distance: None
    term._record_episode_metrics = lambda _distance, *, waypoint_fraction, waypoint_completed: (
        recorded_completion.append(waypoint_completed.clone())
    )

    term._record_current_metrics()
    term.current_index[:] = 1
    pose[:, 0] = 2.0
    term._env.episode_length_buf[:] = 150
    term._record_current_metrics()

    torch.testing.assert_close(term.metrics["waypoint_arrivals"], torch.tensor([2.0]))
    torch.testing.assert_close(term.metrics["route_completions"], torch.tensor([1.0]))
    torch.testing.assert_close(term.metrics["target_distance_completed"], torch.tensor([2.0]))
    torch.testing.assert_close(term.metrics["waypoint_arrival_time_mean"], torch.tensor([0.75]))
    torch.testing.assert_close(term.metrics["waypoint_arrival_time_min"], torch.tensor([0.5]))
    torch.testing.assert_close(term.metrics["waypoint_arrival_time_max"], torch.tensor([1.0]))
    torch.testing.assert_close(term.metrics["waypoint_throughput"], torch.tensor([2.0 / 1.5]))
    torch.testing.assert_close(term.metrics["episode_duration"], torch.tensor([1.5]))
    assert not torch.cat(recorded_completion).any()


def test_cross_track_episode_metrics_report_mean_rms_and_max():
    term = object.__new__(commands_module.WaypointSequenceCommand)
    term._env = SimpleNamespace(num_envs=2, device="cpu")
    term.metrics = {}
    term._initialize_episode_metrics()

    term._update_cross_track_metrics(torch.tensor([1.0, 2.0]))
    term._update_cross_track_metrics(torch.tensor([3.0, 4.0]))

    torch.testing.assert_close(term.metrics["cross_track_error_mean"], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(term.metrics["cross_track_error_rms"], torch.sqrt(torch.tensor([5.0, 10.0])))
    torch.testing.assert_close(term.metrics["cross_track_error_max"], torch.tensor([3.0, 4.0]))


def test_cross_track_episode_metrics_reset_only_selected_environments(monkeypatch):
    term = object.__new__(commands_module.WaypointSequenceCommand)
    term._env = SimpleNamespace(num_envs=2, device="cpu")
    term.metrics = {}
    term._initialize_episode_metrics()
    term._update_cross_track_metrics(torch.tensor([1.0, 2.0]))
    monkeypatch.setattr(commands_module._EpisodeMetricsCommand, "reset", lambda self, env_ids=None: {})

    term.reset(torch.tensor([0]))

    for name in ("cross_track_error_mean", "cross_track_error_rms", "cross_track_error_max"):
        torch.testing.assert_close(term.metrics[name], torch.tensor([0.0, 2.0]))
    torch.testing.assert_close(term._cross_track_count, torch.tensor([0.0, 1.0]))
