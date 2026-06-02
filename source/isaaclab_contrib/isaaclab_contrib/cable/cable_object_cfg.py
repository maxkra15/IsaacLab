# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the cable asset class."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.actuators import ActuatorBaseCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.utils.configclass import configclass


@configclass
class CableAttachmentCfg:
    """Weld a cable endpoint to a body on another spawned asset via a Newton fixed joint.

    Note:
        Only solvers that honor cross-articulation fixed joints (VBD, XPBD) enforce
        this constraint; per-articulation solvers (e.g. MuJoCo, Featherstone) may
        silently drop it. The default VBD solver in this contrib honors it.
    """

    target_prim_path: str = MISSING
    """Prim path of the target rigid body.

    Matched exactly (not by pattern) against the builder's ``body_label`` filtered
    by ``body_world``. Two forms are tried: the path as written (handles
    USD-imported targets carrying the unexpanded ``env_.*`` template at hook
    time), and the same path with ``env_.*`` replaced by ``env_{world_idx}``
    (handles builder-hook targets that are pre-expanded per env).
    """

    cable_anchor: int = -1
    """Rod-segment body index to anchor, Python-style (``0`` = head, ``-1`` = tail).

    The cable has ``len(edges)`` segment bodies; out-of-range values raise at
    attachment-hook time.
    """

    cable_local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Joint anchor position [m] in the cable end-segment's local frame."""

    cable_local_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Joint anchor orientation ``(x, y, z, w)`` in the cable end-segment's local frame."""

    target_local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Joint anchor position [m] in the target body's local frame."""

    target_local_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Joint anchor orientation ``(x, y, z, w)`` in the target body's local frame."""


@configclass
class CableSdfCaptureCfg:
    """Static SDF contact geometry that mechanically captures a cable segment.

    The capture is added directly to the Newton builder as a static mesh shape.
    It does not create a body or a joint; coupled-solver entries should select
    it with ``shape_label_patterns``.
    """

    cable_anchor: int = -1
    """Rod-segment body index to surround, Python-style."""

    label_suffix: str = "sdf_capture"
    """Suffix appended under the cable prim path for the Newton shape label."""

    outer_radius: float = 0.012
    """Outer radius of the generated connector shell [m]."""

    bore_clearance: float = 0.0015
    """Clearance added to the cable radius for the main bore [m]."""

    retainer_hole_radius: float | None = None
    """Radius of the retaining lip hole [m]. ``None`` uses 75% of cable radius."""

    retainer_offset: float = 0.0045
    """Distance behind the segment start where the retaining lip begins [m]."""

    retainer_thickness: float = 0.0015
    """Axial thickness of the retaining lip [m]."""

    tail_clearance: float = 0.006
    """Extra closed-end clearance beyond the segment end [m]."""

    through_sleeve: bool = False
    """When true, build an open through-sleeve instead of a closed retaining cup."""

    radial_segments: int = 32
    """Number of radial tessellation segments in the generated mesh."""

    sdf_max_resolution: int = 64
    """Maximum SDF grid resolution for the generated connector mesh."""

    margin: float = 0.0
    """Newton contact margin for the connector shape [m]."""

    gap: float = 0.001
    """Newton contact gap for the connector shape [m]."""

    mu: float = 1.0
    """Newton friction coefficient for the connector shape."""


@configclass
class CableObjectCfg(ArticulationCfg):
    """Configuration for a cable / 1D-rod asset (Newton backend).

    Inherits all of :class:`ArticulationCfg` and overrides two defaults so the
    base :meth:`Articulation._initialize_impl` runs unchanged on cables. See
    :attr:`articulation_root_prim_path` and :attr:`actuators` for the rationale
    behind each override.
    """

    class_type: type | str = "{DIR}.cable_object:CableObject"

    articulation_root_prim_path: str | None = "/cable_articulation"
    """Sub-label produced by :meth:`newton.ModelBuilder.add_rod_graph` under the
    cable's source prim (``f"{label}_articulation"`` with ``label =
    "{prim_path}/cable"``). Overrides the base default (``None``, which would
    trigger a ``UsdPhysics.ArticulationRootAPI`` stage search) because Newton
    rod-graph cables don't author that schema. The base ``_initialize_impl``
    composes this with :attr:`prim_path` to build the
    :class:`newton.selection.ArticulationView` selector."""

    actuators: dict[str, ActuatorBaseCfg] = {}
    """Empty by design: cables have no user-defined actuators (joint stiffness
    is material-like, applied internally by the solver). Overrides the base
    ``MISSING`` default so the inherited ``_process_actuators_cfg`` iterates an
    empty dict instead of crashing on ``MISSING``; a harmless
    ``logger.warning("Not all actuators are configured!")`` is expected."""

    resample_segment_length: float | None = None
    """Optional arc-length resample target [m]. When set, the cable's authored control
    points are resampled to approximately uniform segments of this length (endpoints
    preserved) before the rod is built; ``None`` keeps the authored points as-is. Use it
    to remove non-uniform segments (e.g. a long lead-in) that destabilize the rod solve.
    Note: segment indices change, so prefer endpoint-relative ``cable_anchor`` values
    (``0`` / ``-1``) on attachments when this is enabled."""

    attachments: list[CableAttachmentCfg] = []

    sdf_captures: list[CableSdfCaptureCfg] = []
    """Static SDF contact captures associated with this cable."""
