# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small tensor utilities shared by Franka Pour reset orchestration."""

import math

import torch


def boolean_selection_mask(count: int, selected: torch.Tensor) -> torch.Tensor:
    """Return a fixed-size boolean mask selecting the supplied indices."""
    if count < 0:
        raise ValueError(f"Mask length must be non-negative, got {count}.")
    mask = torch.zeros(count, dtype=torch.bool, device=selected.device)
    mask[selected.reshape(-1).long()] = True
    return mask


def balanced_cyclic_permutations(values: torch.Tensor, group_count: int) -> torch.Tensor:
    """Return deterministic cyclic permutations with balanced column-wise value counts."""
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError(f"values must be a nonempty one-dimensional tensor, got shape {tuple(values.shape)}.")
    if group_count < 0:
        raise ValueError(f"group_count must be nonnegative, got {group_count}.")
    group_ids = torch.arange(group_count, device=values.device).unsqueeze(-1)
    value_ids = torch.arange(values.numel(), device=values.device).unsqueeze(0)
    return values[(group_ids + value_ids) % values.numel()]


def randomization_extent_index_pools(
    source_positions: torch.Tensor,
    source_yaws: torch.Tensor,
    target_positions: torch.Tensor,
    tcp_jitter: torch.Tensor,
    *,
    source_center: tuple[float, float] | torch.Tensor,
    source_half_range: tuple[float, float] | torch.Tensor,
    source_yaw_half_range: float,
    target_center: tuple[float, float] | torch.Tensor,
    target_half_range: tuple[float, float] | torch.Tensor,
    tcp_jitter_half_range: tuple[float, float, float] | torch.Tensor,
    extent_levels: tuple[float, ...],
    tolerance: float = 1.0e-6,
) -> tuple[torch.Tensor, ...]:
    """Return nested bank indices within combined normalized reset extents.

    Each extent is a Chebyshev radius over source XY position [m], source yaw [rad], target XY
    position [m], and TCP jitter [m], normalized by their configured half-ranges. A zero-range axis
    contributes zero difficulty at its center and excludes rows displaced from that center.
    """
    if source_positions.ndim != 2 or source_positions.shape[-1] < 2:
        raise ValueError(f"source_positions must have shape (N, D) with D >= 2, got {tuple(source_positions.shape)}.")
    if source_yaws.ndim != 1:
        raise ValueError(f"source_yaws must have shape (N,), got {tuple(source_yaws.shape)}.")
    if target_positions.ndim != 2 or target_positions.shape[-1] < 2:
        raise ValueError(f"target_positions must have shape (N, D) with D >= 2, got {tuple(target_positions.shape)}.")
    if tcp_jitter.ndim != 2 or tcp_jitter.shape[-1] != 3:
        raise ValueError(f"tcp_jitter must have shape (N, 3), got {tuple(tcp_jitter.shape)}.")
    row_count = source_positions.shape[0]
    if source_yaws.shape[0] != row_count or target_positions.shape[0] != row_count or tcp_jitter.shape[0] != row_count:
        raise ValueError(
            "source_positions, source_yaws, target_positions, and tcp_jitter must have the same row count."
        )
    if (
        source_yaws.device != source_positions.device
        or target_positions.device != source_positions.device
        or tcp_jitter.device != source_positions.device
    ):
        raise ValueError("source_positions, source_yaws, target_positions, and tcp_jitter must be on the same device.")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")
    if not extent_levels:
        raise ValueError("extent_levels must not be empty.")
    levels = tuple(float(extent) for extent in extent_levels)
    if any(not math.isfinite(extent) or extent < 0.0 for extent in levels):
        raise ValueError("extent_levels must contain finite nonnegative values.")
    if any(right <= left for left, right in zip(levels, levels[1:], strict=False)):
        raise ValueError("extent_levels must be strictly increasing.")

    def normalized_offsets(
        values: torch.Tensor,
        center_values: tuple[float, ...] | torch.Tensor,
        half_range_values: tuple[float, ...] | torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        center = torch.as_tensor(center_values, device=values.device, dtype=values.dtype)
        half_range = torch.as_tensor(half_range_values, device=values.device, dtype=values.dtype)
        if center.shape != (values.shape[1],) or half_range.shape != (values.shape[1],):
            raise ValueError(f"{name}_center and {name}_half_range must each contain {values.shape[1]} coordinates.")
        if bool(torch.any(~torch.isfinite(values))) or bool(torch.any(~torch.isfinite(center))):
            raise ValueError(f"{name} values and center must be finite.")
        if bool(torch.any(~torch.isfinite(half_range))) or bool(torch.any(half_range < 0.0)):
            raise ValueError(f"{name}_half_range must contain finite nonnegative values.")

        offsets = torch.abs(values - center)
        positive_range = half_range > 0.0
        result = torch.zeros_like(offsets)
        result[:, positive_range] = offsets[:, positive_range] / half_range[positive_range]
        if bool(torch.any(~positive_range)):
            result[:, ~positive_range] = torch.where(
                offsets[:, ~positive_range] <= tolerance,
                torch.zeros_like(offsets[:, ~positive_range]),
                torch.full_like(offsets[:, ~positive_range], torch.inf),
            )
        return result

    normalized_source = normalized_offsets(source_positions[:, :2], source_center, source_half_range, "source")
    normalized_source_yaw = normalized_offsets(
        source_yaws.unsqueeze(-1),
        torch.zeros(1, device=source_yaws.device, dtype=source_yaws.dtype),
        torch.as_tensor((source_yaw_half_range,), device=source_yaws.device, dtype=source_yaws.dtype),
        "source_yaw",
    )
    normalized_target = normalized_offsets(target_positions[:, :2], target_center, target_half_range, "target")
    normalized_tcp_jitter = normalized_offsets(
        tcp_jitter,
        torch.zeros(3, device=tcp_jitter.device, dtype=tcp_jitter.dtype),
        tcp_jitter_half_range,
        "tcp_jitter",
    )
    difficulty = torch.cat(
        (normalized_source, normalized_source_yaw, normalized_target, normalized_tcp_jitter), dim=-1
    ).amax(dim=-1)

    pools = tuple(torch.nonzero(difficulty <= extent + tolerance, as_tuple=False).flatten() for extent in levels)
    if any(pool.numel() == 0 for pool in pools):
        raise ValueError("Every randomization extent level must select at least one bank row.")
    return pools


def sample_index_pools(index_pools: tuple[torch.Tensor, ...], pool_ids: torch.Tensor) -> torch.Tensor:
    """Sample one global bank index per row from its selected device-resident pool."""
    if pool_ids.ndim != 1:
        raise ValueError(f"pool_ids must be one-dimensional, got shape {tuple(pool_ids.shape)}.")
    result = torch.empty_like(pool_ids, dtype=torch.long)
    for pool_id, index_pool in enumerate(index_pools):
        rows = torch.nonzero(pool_ids == pool_id, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        slots = torch.randint(index_pool.numel(), (rows.numel(),), device=pool_ids.device)
        result[rows] = index_pool[slots]
    return result


def target_xy_behind_source(
    source_xy: torch.Tensor,
    *,
    target_center: tuple[float, float] | torch.Tensor,
    target_half_range: tuple[float, float] | torch.Tensor,
    minimum_y_separation: float | torch.Tensor,
    unit_samples: torch.Tensor,
) -> torch.Tensor:
    """Map unit-square samples to target positions safely behind each source cup [m]."""
    if source_xy.ndim != 2 or source_xy.shape[-1] != 2:
        raise ValueError(f"source_xy must have shape (N, 2), got {tuple(source_xy.shape)}.")
    if unit_samples.shape != source_xy.shape:
        raise ValueError(
            f"unit_samples must match source_xy shape {tuple(source_xy.shape)}, got {tuple(unit_samples.shape)}."
        )
    separation = torch.as_tensor(minimum_y_separation, device=source_xy.device, dtype=source_xy.dtype)
    if separation.ndim == 0:
        separation = separation.expand(source_xy.shape[0])
    elif separation.shape != (source_xy.shape[0],):
        raise ValueError("minimum_y_separation must be a scalar or contain one value per source row.")
    if bool(torch.any(~torch.isfinite(separation))) or bool(torch.any(separation < 0.0)):
        raise ValueError("minimum_y_separation must be finite and nonnegative.")
    if bool(torch.any((unit_samples < 0.0) | (unit_samples > 1.0))):
        raise ValueError("unit_samples must lie in [0, 1].")

    center = torch.as_tensor(target_center, device=source_xy.device, dtype=source_xy.dtype)
    half_range = torch.as_tensor(target_half_range, device=source_xy.device, dtype=source_xy.dtype)
    if center.shape != (2,) or half_range.shape != (2,):
        raise ValueError("target_center and target_half_range must each contain two coordinates.")
    if bool(torch.any(~torch.isfinite(center))) or bool(torch.any(~torch.isfinite(half_range))):
        raise ValueError("Target randomization bounds must be finite.")
    if bool(torch.any(half_range < 0.0)):
        raise ValueError("target_half_range must be nonnegative.")

    lower = center - half_range
    upper = center + half_range
    allowed_y_upper = torch.minimum(
        torch.full_like(source_xy[:, 1], upper[1]),
        source_xy[:, 1] - separation,
    )
    if bool(torch.any(allowed_y_upper < lower[1])):
        raise ValueError("No target y-position satisfies the configured range and source-cup separation.")

    target_x = lower[0] + unit_samples[:, 0] * (upper[0] - lower[0])
    target_y = lower[1] + unit_samples[:, 1] * (allowed_y_upper - lower[1])
    return torch.stack((target_x, target_y), dim=-1)
