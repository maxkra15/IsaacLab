# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Description of reset rows and the competence items they train."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, slots=True)
class ResetStateCatalog:
    """Validated row-aligned metadata for a reset-state source.

    Physical rows and monitored competence items are distinct concepts. By default each physical
    row is its own competence item. ``row_to_item`` may instead group multiple continuously varied
    or otherwise equivalent rows under one outcome estimate.

    Args:
        row_count: Number of physical reset rows.
        metadata: Named tensors whose first dimension is ``row_count``.
        row_to_item: Optional contiguous, zero-based competence-item id for every physical row.
    """

    row_count: int
    metadata: Mapping[str, torch.Tensor] = field(default_factory=dict)
    row_to_item: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Validate catalog dimensions and competence ids."""
        if isinstance(self.row_count, bool) or not isinstance(self.row_count, int) or self.row_count < 1:
            raise ValueError("row_count must be a positive integer.")

        metadata = dict(self.metadata)
        for name, values in metadata.items():
            if not isinstance(name, str) or not name:
                raise ValueError("metadata names must be non-empty strings.")
            if not isinstance(values, torch.Tensor):
                raise TypeError(f"metadata[{name!r}] must be a torch.Tensor.")
            if values.ndim == 0 or values.shape[0] != self.row_count:
                raise ValueError(f"metadata[{name!r}] must have leading dimension {self.row_count}.")
        object.__setattr__(self, "metadata", metadata)

        if self.row_to_item is None:
            return
        if not isinstance(self.row_to_item, torch.Tensor):
            raise TypeError("row_to_item must be a torch.Tensor or None.")
        if self.row_to_item.ndim != 1 or self.row_to_item.shape[0] != self.row_count:
            raise ValueError(f"row_to_item must have shape ({self.row_count},).")
        if (
            self.row_to_item.dtype == torch.bool
            or self.row_to_item.is_floating_point()
            or self.row_to_item.is_complex()
        ):
            raise TypeError("row_to_item must have an integer dtype other than bool.")
        if bool((self.row_to_item < 0).any()):
            raise ValueError("row_to_item must contain non-negative ids.")
        item_ids = torch.unique(self.row_to_item, sorted=True)
        expected = torch.arange(item_ids.numel(), device=item_ids.device, dtype=item_ids.dtype)
        if not torch.equal(item_ids, expected):
            raise ValueError("row_to_item ids must be contiguous and start at zero.")

    @property
    def item_count(self) -> int:
        """Return the number of monitored competence items."""
        if self.row_to_item is None:
            return self.row_count
        return int(self.row_to_item.max().item()) + 1

    def item_ids_for_rows(self, row_ids: torch.Tensor) -> torch.Tensor:
        """Map physical row ids to competence-item ids.

        Args:
            row_ids: Physical reset row ids.

        Returns:
            Competence-item ids with the same shape and device as ``row_ids``.
        """
        if row_ids.dtype == torch.bool or row_ids.is_floating_point() or row_ids.is_complex():
            raise TypeError("row_ids must have an integer dtype other than bool.")
        if row_ids.numel() > 0:
            # Keep this invariant check asynchronous on CUDA.  Catalog lookup is part of the
            # reset hot path, where converting the reduction to a Python bool would serialize
            # every vectorized environment reset with the host.
            torch._assert_async(
                ((row_ids >= 0) & (row_ids < self.row_count)).all(),
                "row_ids contains an id outside the catalog.",
            )
        if self.row_to_item is None:
            return row_ids
        return self.row_to_item.to(device=row_ids.device)[row_ids].to(dtype=row_ids.dtype)
