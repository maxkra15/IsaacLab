# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable geometry for the two-switch RJ45 T-slot workcell.

The workcell deliberately separates presentation from contact geometry.  Kit
renders the detailed AS4610 assets, while Newton receives only the exact RJ45
connector meshes and the inexpensive cuboids declared here.  This keeps the
robot/cable contact model deterministic without turning the decorative rack
and extrusion details into hundreds of collision meshes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

DUAL_RACK_WORKCELL_CONTRACT_VERSION = 4

DUAL_RACK_TARGET_SOCKET_POSITION_E = (0.58, 0.07, 0.14)
"""Target socket-body origin in the environment frame [m]."""

DUAL_RACK_ANCHORED_SOCKET_POSITION_E = (0.58, 0.07, 0.075)
"""Already-connected socket-body origin in the environment frame [m]."""

# Exact socket translation in the verified Newton source asset.  The task
# translation is the assembly origin, not the socket-body origin.
_RJ45_SOCKET_POSITION_Z_M = 0.011653749272227287
DUAL_RACK_TARGET_TASK_TRANSLATION = (
    DUAL_RACK_TARGET_SOCKET_POSITION_E[0],
    DUAL_RACK_TARGET_SOCKET_POSITION_E[1],
    DUAL_RACK_TARGET_SOCKET_POSITION_E[2] - _RJ45_SOCKET_POSITION_Z_M,
)
DUAL_RACK_ANCHORED_TASK_TRANSLATION = (
    DUAL_RACK_ANCHORED_SOCKET_POSITION_E[0],
    DUAL_RACK_ANCHORED_SOCKET_POSITION_E[1],
    DUAL_RACK_ANCHORED_SOCKET_POSITION_E[2] - _RJ45_SOCKET_POSITION_Z_M,
)
DUAL_RACK_TASK_ROTATION_XYZW = (0.0, 0.0, 0.0, 1.0)

# AS4610 bounds after the presentation transform.  Both the detailed Kit asset
# and NewtonGL marker shell are aligned to the socket-local origin.
DUAL_RACK_SWITCH_MIN_SOCKET_LOCAL = (-0.21826283, -0.01146542, -0.02488871)
DUAL_RACK_SWITCH_MAX_SOCKET_LOCAL = (0.26810951, 0.35749699, 0.01873206)

DUAL_RACK_FRAME_COLOR = (0.56, 0.60, 0.66)
DUAL_RACK_GROOVE_COLOR = (0.035, 0.045, 0.060)
DUAL_RACK_TARGET_ACCENT_COLOR = (0.025, 0.42, 0.78)
DUAL_RACK_ANCHORED_ACCENT_COLOR = (0.04, 0.66, 0.24)
DUAL_RACK_CABLE_TABLE_CENTERLINE_HEIGHT_M = 0.00365
"""Nominal slack-cable centerline height above the table frame [m].

This is the cable radius plus 0.4 mm.  It starts the lowest slack span close
to its measured soft-contact equilibrium instead of dropping it from twice
the radius and repeatedly exciting a marginal table-contact rebound.
"""

DUAL_RACK_CABLE_CONTACT_DAMPING_N_S_M = 100.0
"""Normal damping for the two-ended cable's contacts [N·s/m]."""

_T_SLOT_PROFILE_SIZE_M = 0.025
_T_SLOT_TOP_CLEARANCE_M = 0.010
_T_SLOT_ROBOT_ACCESS_HALF_WIDTH_M = 0.105
_CABLE_ROUTE_RESTORE_ATOL_M = 1.0e-6


def _finite_tuple(values: tuple[float, ...], *, length: int, name: str) -> tuple[float, ...]:
    """Return one finite fixed-width tuple as plain floats."""
    if len(values) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(values)}.")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values, got {values}.")
    return result


@dataclass(frozen=True)
class DualRackWorkcellBoxCfg:
    """One environment-local presentation and/or collision cuboid.

    Attributes:
        name: Stable path suffix below ``Rj45Assembly/Workcell``.
        center_m: Cuboid center in the environment frame [m].
        size_m: Full xyz dimensions [m].
        color: Linear RGB presentation color.
        collidable: Whether Newton should create a static box shape.
        visible: Whether Kit/NewtonGL should show the box.
    """

    name: str
    center_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    color: tuple[float, float, float]
    collidable: bool = True
    visible: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name.startswith("/") or ".." in self.name:
            raise ValueError(f"Workcell box name must be a non-empty relative path, got {self.name!r}.")
        center = _finite_tuple(self.center_m, length=3, name=f"{self.name}.center_m")
        size = _finite_tuple(self.size_m, length=3, name=f"{self.name}.size_m")
        color = _finite_tuple(self.color, length=3, name=f"{self.name}.color")
        if any(value <= 0.0 for value in size):
            raise ValueError(f"Workcell box dimensions must be positive, got {size} for {self.name!r}.")
        if any(not 0.0 <= value <= 1.0 for value in color):
            raise ValueError(f"Workcell box RGB values must lie in [0, 1], got {color} for {self.name!r}.")
        if not isinstance(self.collidable, bool) or not isinstance(self.visible, bool):
            raise TypeError("Workcell box collidable/visible flags must be bools.")
        object.__setattr__(self, "center_m", center)
        object.__setattr__(self, "size_m", size)
        object.__setattr__(self, "color", color)


@dataclass(frozen=True)
class DualRackAnchoredConnectorCfg:
    """Static socket, seated plug, and latch at the cable's anchored end."""

    task_translation_m: tuple[float, float, float] = DUAL_RACK_ANCHORED_TASK_TRANSLATION
    task_rotation_xyzw: tuple[float, float, float, float] = DUAL_RACK_TASK_ROTATION_XYZW
    accent_color: tuple[float, float, float] = DUAL_RACK_ANCHORED_ACCENT_COLOR

    def __post_init__(self) -> None:
        translation = _finite_tuple(self.task_translation_m, length=3, name="anchored task translation")
        rotation = _finite_tuple(self.task_rotation_xyzw, length=4, name="anchored task rotation")
        color = _finite_tuple(self.accent_color, length=3, name="anchored accent color")
        norm = math.sqrt(sum(value * value for value in rotation))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError(f"Anchored connector quaternion must be normalized, got norm={norm}.")
        if any(not 0.0 <= value <= 1.0 for value in color):
            raise ValueError("Anchored connector accent color must lie in [0, 1].")
        object.__setattr__(self, "task_translation_m", translation)
        object.__setattr__(self, "task_rotation_xyzw", rotation)
        object.__setattr__(self, "accent_color", color)


@dataclass(frozen=True)
class DualRackWorkcellCfg:
    """Complete static two-rack workcell specification."""

    anchored_connector: DualRackAnchoredConnectorCfg
    boxes: tuple[DualRackWorkcellBoxCfg, ...]
    target_accent_color: tuple[float, float, float] = DUAL_RACK_TARGET_ACCENT_COLOR
    cable_contact_damping_n_s_m: float = DUAL_RACK_CABLE_CONTACT_DAMPING_N_S_M
    presentation_kind: str = "dual-as4610"
    contract_version: int = DUAL_RACK_WORKCELL_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.anchored_connector, DualRackAnchoredConnectorCfg):
            raise TypeError("anchored_connector must be DualRackAnchoredConnectorCfg.")
        if not self.boxes or any(not isinstance(box, DualRackWorkcellBoxCfg) for box in self.boxes):
            raise ValueError("Dual-rack workcell requires at least one valid box specification.")
        names = tuple(box.name for box in self.boxes)
        if len(names) != len(set(names)):
            raise ValueError("Dual-rack workcell box names must be unique.")
        color = _finite_tuple(self.target_accent_color, length=3, name="target accent color")
        if any(not 0.0 <= value <= 1.0 for value in color):
            raise ValueError("Target connector accent color must lie in [0, 1].")
        damping = self.cable_contact_damping_n_s_m
        if isinstance(damping, bool) or not isinstance(damping, int | float) or not math.isfinite(damping):
            raise TypeError("Cable contact damping must be one finite real scalar.")
        if damping < 0.0:
            raise ValueError("Cable contact damping must be non-negative.")
        if self.presentation_kind not in ("dual-as4610", "gb300"):
            raise ValueError("presentation_kind must be 'dual-as4610' or 'gb300'.")
        if type(self.contract_version) is not int or self.contract_version != DUAL_RACK_WORKCELL_CONTRACT_VERSION:
            raise ValueError(f"Dual-rack workcell contract version must be {DUAL_RACK_WORKCELL_CONTRACT_VERSION}.")
        object.__setattr__(self, "target_accent_color", color)
        object.__setattr__(self, "cable_contact_damping_n_s_m", float(damping))


def _rack_collision_boxes(
    rack_name: str,
    socket_position: tuple[float, float, float],
) -> list[DualRackWorkcellBoxCfg]:
    """Build a closed switch shell whose front slab leaves one connector opening."""
    lo = DUAL_RACK_SWITCH_MIN_SOCKET_LOCAL
    hi = DUAL_RACK_SWITCH_MAX_SOCKET_LOCAL
    sx, sy, sz = socket_position
    thickness = 0.004
    width, depth, height = (hi[index] - lo[index] for index in range(3))
    center = tuple((lo[index] + hi[index]) * 0.5 for index in range(3))
    rack_color = (0.19, 0.22, 0.26)

    def box(
        suffix: str,
        local_center: tuple[float, float, float],
        size: tuple[float, float, float],
    ) -> DualRackWorkcellBoxCfg:
        return DualRackWorkcellBoxCfg(
            name=f"Racks/{rack_name}/{suffix}",
            center_m=(sx + local_center[0], sy + local_center[1], sz + local_center[2]),
            size_m=size,
            color=rack_color,
            collidable=True,
            # The detailed AS4610/marker presentation owns visible rack pixels.
            visible=False,
        )

    boxes = [
        box("LeftSide", (lo[0] + 0.5 * thickness, center[1], center[2]), (thickness, depth, height)),
        box("RightSide", (hi[0] - 0.5 * thickness, center[1], center[2]), (thickness, depth, height)),
        box("Bottom", (center[0], center[1], lo[2] + 0.5 * thickness), (width, depth, thickness)),
        box("Top", (center[0], center[1], hi[2] - 0.5 * thickness), (width, depth, thickness)),
        box(
            "Rear",
            (center[0], hi[1] - 0.5 * thickness, center[2]),
            (width - 2.0 * thickness, thickness, height - 2.0 * thickness),
        ),
    ]

    # One four-piece front panel blocks the chassis while keeping a generous
    # 36 x 30 mm corridor around the real SDF insertion geometry.
    opening_half_x = 0.018
    opening_half_z = 0.015
    front_y = lo[1] + 0.5 * thickness
    left_width = -opening_half_x - lo[0]
    right_width = hi[0] - opening_half_x
    boxes.extend(
        (
            box(
                "FrontLeft",
                ((lo[0] - opening_half_x) * 0.5, front_y, center[2]),
                (left_width, thickness, height),
            ),
            box(
                "FrontRight",
                ((hi[0] + opening_half_x) * 0.5, front_y, center[2]),
                (right_width, thickness, height),
            ),
            box(
                "FrontBottom",
                (0.0, front_y, (lo[2] - opening_half_z) * 0.5),
                (2.0 * opening_half_x, thickness, -opening_half_z - lo[2]),
            ),
            box(
                "FrontTop",
                (0.0, front_y, (hi[2] + opening_half_z) * 0.5),
                (2.0 * opening_half_x, thickness, hi[2] - opening_half_z),
            ),
        )
    )
    return boxes


def _frame_collision_boxes() -> list[DualRackWorkcellBoxCfg]:
    """Build hidden 25 mm box contacts tightly enclosing both switch chassis."""
    aluminum = DUAL_RACK_FRAME_COLOR
    thickness = _T_SLOT_PROFILE_SIZE_M
    half_thickness = 0.5 * thickness
    rack_min_x = DUAL_RACK_TARGET_SOCKET_POSITION_E[0] + DUAL_RACK_SWITCH_MIN_SOCKET_LOCAL[0]
    rack_max_x = DUAL_RACK_TARGET_SOCKET_POSITION_E[0] + DUAL_RACK_SWITCH_MAX_SOCKET_LOCAL[0]
    rack_min_y = DUAL_RACK_TARGET_SOCKET_POSITION_E[1] + DUAL_RACK_SWITCH_MIN_SOCKET_LOCAL[1]
    rack_max_y = DUAL_RACK_TARGET_SOCKET_POSITION_E[1] + DUAL_RACK_SWITCH_MAX_SOCKET_LOCAL[1]
    upper_rack_top = (
        max(
            DUAL_RACK_TARGET_SOCKET_POSITION_E[2],
            DUAL_RACK_ANCHORED_SOCKET_POSITION_E[2],
        )
        + DUAL_RACK_SWITCH_MAX_SOCKET_LOCAL[2]
    )
    # Side uprights touch the chassis bounds.  Front/rear uprights share the
    # same centerlines as the rack-support crossmembers, so the complete frame
    # reads as one regular T-slot grid rather than two offset rectangles.
    x_left = rack_min_x - half_thickness
    x_right = rack_max_x + half_thickness
    y_front = rack_min_y + half_thickness
    y_rear = rack_max_y - half_thickness
    z_bottom = half_thickness
    z_top = upper_rack_top + _T_SLOT_TOP_CLEARANCE_M + half_thickness
    post_height = z_top + half_thickness
    boxes: list[DualRackWorkcellBoxCfg] = []

    for side, x in (("Left", x_left), ("Right", x_right)):
        for depth, y in (("Front", y_front), ("Rear", y_rear)):
            boxes.append(
                DualRackWorkcellBoxCfg(
                    name=f"Frame/Posts/{side}{depth}",
                    center_m=(x, y, 0.5 * post_height),
                    size_m=(thickness, thickness, post_height),
                    color=aluminum,
                    visible=False,
                )
            )

    span_x = x_right - x_left + thickness
    span_y = y_rear - y_front + thickness
    for level, z in (("Base", z_bottom), ("Top", z_top)):
        for depth, y in (("Front", y_front), ("Rear", y_rear)):
            # The cable-facing lower edge is a genuine opening, not a rail
            # displaced away from the rack.  The free cable can therefore lie
            # on the table without crossing hidden contact geometry.
            if level == "Base" and depth == "Front":
                continue
            if level == "Top" and depth == "Front":
                # Preserve the aligned front rail on both sides of the rack,
                # but leave an honest service opening above the active port.
                # The canonical top-down Franka grasp occupies this column;
                # a continuous rail would make the task physically impossible.
                outer_left = x_left - half_thickness
                outer_right = x_right + half_thickness
                access_left = DUAL_RACK_TARGET_SOCKET_POSITION_E[0] - _T_SLOT_ROBOT_ACCESS_HALF_WIDTH_M
                access_right = DUAL_RACK_TARGET_SOCKET_POSITION_E[0] + _T_SLOT_ROBOT_ACCESS_HALF_WIDTH_M
                for side_name, edge_left, edge_right in (
                    ("Left", outer_left, access_left),
                    ("Right", access_right, outer_right),
                ):
                    boxes.append(
                        DualRackWorkcellBoxCfg(
                            name=f"Frame/{level}/X{depth}{side_name}",
                            center_m=(0.5 * (edge_left + edge_right), y, z),
                            size_m=(edge_right - edge_left, thickness, thickness),
                            color=aluminum,
                            visible=False,
                        )
                    )
                continue
            boxes.append(
                DualRackWorkcellBoxCfg(
                    name=f"Frame/{level}/X{depth}",
                    center_m=((x_left + x_right) * 0.5, y, z),
                    size_m=(span_x, thickness, thickness),
                    color=aluminum,
                    visible=False,
                )
            )
        for side, x in (("Left", x_left), ("Right", x_right)):
            boxes.append(
                DualRackWorkcellBoxCfg(
                    name=f"Frame/{level}/Y{side}",
                    center_m=(x, (y_front + y_rear) * 0.5, z),
                    size_m=(thickness, span_y, thickness),
                    color=aluminum,
                    visible=False,
                )
            )
    return boxes


def _rack_support_collision_boxes() -> list[DualRackWorkcellBoxCfg]:
    """Return hidden T-slot-sized crossmembers directly supporting both racks."""
    result: list[DualRackWorkcellBoxCfg] = []
    thickness = _T_SLOT_PROFILE_SIZE_M
    half_thickness = 0.5 * thickness
    rack_min_x = DUAL_RACK_TARGET_SOCKET_POSITION_E[0] + DUAL_RACK_SWITCH_MIN_SOCKET_LOCAL[0]
    rack_max_x = DUAL_RACK_TARGET_SOCKET_POSITION_E[0] + DUAL_RACK_SWITCH_MAX_SOCKET_LOCAL[0]
    x_left = rack_min_x - half_thickness
    x_right = rack_max_x + half_thickness
    span_x = x_right - x_left + thickness
    rack_front = DUAL_RACK_TARGET_SOCKET_POSITION_E[1] + DUAL_RACK_SWITCH_MIN_SOCKET_LOCAL[1]
    rack_rear = DUAL_RACK_TARGET_SOCKET_POSITION_E[1] + DUAL_RACK_SWITCH_MAX_SOCKET_LOCAL[1]
    for rack_name, socket_z in (
        ("Anchored", DUAL_RACK_ANCHORED_SOCKET_POSITION_E[2]),
        ("Target", DUAL_RACK_TARGET_SOCKET_POSITION_E[2]),
    ):
        chassis_bottom = socket_z + DUAL_RACK_SWITCH_MIN_SOCKET_LOCAL[2]
        support_z = chassis_bottom - 0.5 * thickness
        for depth_name, y in (
            ("Front", rack_front + 0.5 * thickness),
            ("Rear", rack_rear - 0.5 * thickness),
        ):
            result.append(
                DualRackWorkcellBoxCfg(
                    name=f"Frame/RackSupports/{rack_name}{depth_name}",
                    center_m=((x_left + x_right) * 0.5, y, support_z),
                    size_m=(span_x, thickness, thickness),
                    color=DUAL_RACK_FRAME_COLOR,
                    visible=False,
                )
            )
    return result


def _t_slot_visual_boxes(structural_boxes: list[DualRackWorkcellBoxCfg]) -> list[DualRackWorkcellBoxCfg]:
    """Build an open 25 mm extrusion profile around each hidden contact box.

    Two recessed webs and four corner rails form a recognizable T-slot cross
    section.  They are presentation-only; Newton still collides one cuboid per
    structural member.
    """
    visuals: list[DualRackWorkcellBoxCfg] = []
    rail_width = 0.006
    web_thickness = 0.004
    slot_recess = 0.004
    for structural in structural_boxes:
        size = structural.size_m
        main_axis = max(range(3), key=size.__getitem__)
        cross_axes = tuple(axis for axis in range(3) if axis != main_axis)
        length = 0.985 * size[main_axis]

        # Recessed orthogonal webs provide a connected aluminum cross without
        # filling the four slot mouths at the outside faces.
        for web_axis, thin_axis in (cross_axes, cross_axes[::-1]):
            web_size = [rail_width, rail_width, rail_width]
            web_size[main_axis] = length
            web_size[web_axis] = size[web_axis] - 2.0 * slot_recess
            web_size[thin_axis] = web_thickness
            visuals.append(
                DualRackWorkcellBoxCfg(
                    name=f"TSlotVisual/{structural.name}/Web{web_axis}",
                    center_m=structural.center_m,
                    size_m=tuple(web_size),
                    color=DUAL_RACK_FRAME_COLOR,
                    collidable=False,
                )
            )

        # Four bright longitudinal corner rails form the lips of all four
        # T-slots.  The empty space between them is the visible dark channel.
        first_axis, second_axis = cross_axes
        first_offset = 0.5 * (size[first_axis] - rail_width)
        second_offset = 0.5 * (size[second_axis] - rail_width)
        for first_sign in (-1.0, 1.0):
            for second_sign in (-1.0, 1.0):
                rail_center = list(structural.center_m)
                rail_center[first_axis] += first_sign * first_offset
                rail_center[second_axis] += second_sign * second_offset
                rail_size = [rail_width, rail_width, rail_width]
                rail_size[main_axis] = length
                visuals.append(
                    DualRackWorkcellBoxCfg(
                        name=(
                            f"TSlotVisual/{structural.name}/Rail"
                            f"{'P' if first_sign > 0.0 else 'N'}{first_axis}"
                            f"{'P' if second_sign > 0.0 else 'N'}{second_axis}"
                        ),
                        center_m=tuple(rail_center),
                        size_m=tuple(rail_size),
                        color=DUAL_RACK_FRAME_COLOR,
                        collidable=False,
                    )
                )
    return visuals


def make_dual_rack_workcell_cfg() -> DualRackWorkcellCfg:
    """Return the single production two-rack workcell specification."""
    frame = _frame_collision_boxes()
    supports = _rack_support_collision_boxes()
    structure = [*frame, *supports]
    rack = [
        *_rack_collision_boxes("Target", DUAL_RACK_TARGET_SOCKET_POSITION_E),
        *_rack_collision_boxes("Anchored", DUAL_RACK_ANCHORED_SOCKET_POSITION_E),
    ]
    return DualRackWorkcellCfg(
        anchored_connector=DualRackAnchoredConnectorCfg(),
        boxes=tuple((*structure, *rack, *_t_slot_visual_boxes(structure))),
    )


DUAL_RACK_WORKCELL_CFG = make_dual_rack_workcell_cfg()


def route_dual_rack_cable_points_numpy(
    fixed_prefix_points: np.ndarray,
    anchored_endpoint: np.ndarray,
    segment_lengths: np.ndarray,
    *,
    fixed_suffix_points: np.ndarray | None = None,
    table_centerline_height_m: float = DUAL_RACK_CABLE_TABLE_CENTERLINE_HEIGHT_M,
    bisection_iterations: int = 48,
) -> np.ndarray:
    """Route one exact-length cable from its free-plug prefix to the anchored plug.

    The first points are preserved exactly because the runtime kinematically
    synchronizes the first four cable bodies to the movable plug.  The
    remaining length is distributed along one smooth circular slack arc.
    Bisection selects the circle radius and plane so every authored body span
    retains its exact rest length while all discrete endpoints clear the table.
    No random body-wise perturbation is introduced.

    Args:
        fixed_prefix_points: Consecutive cable endpoints with shape ``(K+1, 3)``.
        anchored_endpoint: Final cable endpoint with shape ``(3,)``.
        segment_lengths: Positive rest lengths for all ``S`` cable segments.
        table_centerline_height_m: Height used by the two slack controls [m].
        bisection_iterations: Fixed bounded root iterations.

    Returns:
        Cable endpoints with shape ``(S+1, 3)``.
    """
    prefix = np.asarray(fixed_prefix_points, dtype=np.float64)
    final_endpoint = np.asarray(anchored_endpoint, dtype=np.float64)
    lengths = np.asarray(segment_lengths, dtype=np.float64)
    if prefix.ndim != 2 or prefix.shape[1] != 3 or prefix.shape[0] < 2:
        raise ValueError(f"fixed_prefix_points must have shape (K+1, 3), got {prefix.shape}.")
    if final_endpoint.shape != (3,):
        raise ValueError(f"anchored_endpoint must have shape (3,), got {final_endpoint.shape}.")
    if lengths.ndim != 1 or lengths.size < prefix.shape[0] - 1 or np.any(lengths <= 0.0):
        raise ValueError("segment_lengths must contain enough finite positive lengths.")
    if not np.isfinite(prefix).all() or not np.isfinite(final_endpoint).all() or not np.isfinite(lengths).all():
        raise ValueError("Cable routing inputs must be finite.")
    if not math.isfinite(table_centerline_height_m):
        raise ValueError("table_centerline_height_m must be finite.")
    if type(bisection_iterations) is not int or bisection_iterations < 1:
        raise ValueError("bisection_iterations must be a positive plain integer.")

    fixed_segment_count = prefix.shape[0] - 1
    if fixed_suffix_points is None:
        suffix = final_endpoint[None]
    else:
        suffix = np.asarray(fixed_suffix_points, dtype=np.float64)
        if suffix.ndim != 2 or suffix.shape[1] != 3 or suffix.shape[0] < 2 or not np.isfinite(suffix).all():
            raise ValueError(f"fixed_suffix_points must be finite with shape (L+1, 3), got {suffix.shape}.")
        if not np.allclose(suffix[-1], final_endpoint, rtol=0.0, atol=1.0e-9):
            raise ValueError("The fixed cable suffix must end at anchored_endpoint.")
    fixed_suffix_segment_count = suffix.shape[0] - 1
    if fixed_segment_count + fixed_suffix_segment_count >= lengths.size:
        raise ValueError("Fixed cable prefix and suffix leave no routed middle segment.")
    if fixed_suffix_segment_count and not np.allclose(
        np.linalg.norm(np.diff(suffix, axis=0), axis=-1),
        lengths[-fixed_suffix_segment_count:],
        rtol=0.0,
        # The Newton builder owns float32 transforms.  A translated, rotated
        # anchored connector can therefore accumulate one ULP at metre-scale
        # coordinates before this NumPy routing boundary.
        atol=_CABLE_ROUTE_RESTORE_ATOL_M,
    ):
        raise ValueError("Fixed cable suffix does not preserve the authored trailing rest lengths.")
    remaining_lengths = lengths[fixed_segment_count : lengths.size - fixed_suffix_segment_count]
    remaining_length = float(remaining_lengths.sum())
    start = prefix[-1]
    route_endpoint = suffix[0]
    chord = route_endpoint - start
    direct_distance = float(np.linalg.norm(chord))
    if direct_distance > remaining_length + 1.0e-9:
        raise ValueError(
            f"Cable endpoint separation {direct_distance:.6f} m exceeds remaining rest length {remaining_length:.6f} m."
        )

    if direct_distance <= 1.0e-9:
        raise ValueError("Dual-rack cable routing requires distinct prefix and anchored endpoints.")

    # Place every remaining segment as a chord on one circle.  This is both
    # smoother and stronger than sampling a polyline by arclength: every
    # Euclidean body-to-body span remains exactly its authored rest length,
    # including spans that straddle what would otherwise be a polyline corner.
    maximum_half_chord = 0.5 * float(remaining_lengths.max())

    def total_angle(radius: float) -> float:
        return float((2.0 * np.arcsin(np.clip(remaining_lengths / (2.0 * radius), -1.0, 1.0))).sum())

    lower_radius = maximum_half_chord * (1.0 + 1.0e-12)
    upper_radius = max(remaining_length, lower_radius * 2.0)
    while total_angle(upper_radius) >= 2.0 * math.pi:
        upper_radius *= 2.0
    # The useful branch has total angle in (0, 2*pi): its endpoint distance
    # grows monotonically from zero to the straight-chain length.
    if total_angle(lower_radius) > 2.0 * math.pi:
        for _ in range(bisection_iterations):
            midpoint_radius = 0.5 * (lower_radius + upper_radius)
            if total_angle(midpoint_radius) > 2.0 * math.pi:
                lower_radius = midpoint_radius
            else:
                upper_radius = midpoint_radius
        lower_radius = upper_radius

    def endpoint_distance(radius: float) -> float:
        angle = total_angle(radius)
        return float(2.0 * radius * math.sin(0.5 * angle))

    upper_radius = max(upper_radius, remaining_length)
    while endpoint_distance(upper_radius) < direct_distance:
        upper_radius *= 2.0
    for _ in range(bisection_iterations):
        midpoint_radius = 0.5 * (lower_radius + upper_radius)
        if endpoint_distance(midpoint_radius) < direct_distance:
            lower_radius = midpoint_radius
        else:
            upper_radius = midpoint_radius
    radius = 0.5 * (lower_radius + upper_radius)
    segment_angles = 2.0 * np.arcsin(np.clip(remaining_lengths / (2.0 * radius), -1.0, 1.0))
    angle = float(segment_angles.sum())

    chord_axis = chord / direct_distance
    upward = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    upward -= float(np.dot(upward, chord_axis)) * chord_axis
    upward_norm = float(np.linalg.norm(upward))
    if upward_norm <= 1.0e-9:
        upward = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    else:
        upward /= upward_norm
    lateral = np.cross(chord_axis, upward)
    lateral /= np.linalg.norm(lateral)

    # Select the circle plane so every routed endpoint stays above the table.
    # Any perpendicular magnitude not needed vertically becomes a gentle
    # lateral loop.  Solving this from every discrete endpoint is stronger than
    # assuming the symmetric midpoint is the unique vertical extremum on a
    # major (>pi) slack arc.
    initial_angle = -0.5 * angle - 0.5 * math.pi
    cumulative_angles = np.cumsum(segment_angles)
    u_coefficients = radius * (np.cos(initial_angle + cumulative_angles) - math.cos(initial_angle))
    v_coefficients = radius * (np.sin(initial_angle + cumulative_angles) - math.sin(initial_angle))
    base_z = start[2] + u_coefficients * chord_axis[2]
    maximum_axis_z = abs(float(upward[2]))
    minimum_allowed_axis_z = -maximum_axis_z
    maximum_allowed_axis_z = maximum_axis_z
    for value_z, coefficient in zip(base_z, v_coefficients, strict=True):
        if coefficient > 1.0e-10:
            minimum_allowed_axis_z = max(
                minimum_allowed_axis_z,
                (table_centerline_height_m - value_z) / coefficient,
            )
        elif coefficient < -1.0e-10:
            maximum_allowed_axis_z = min(
                maximum_allowed_axis_z,
                (table_centerline_height_m - value_z) / coefficient,
            )
    if minimum_allowed_axis_z > maximum_allowed_axis_z + 1.0e-9:
        raise ValueError("No dual-rack cable circle plane can satisfy the table-clearance constraint.")
    candidate_axis_z = (minimum_allowed_axis_z, maximum_allowed_axis_z)
    axis_z = min(
        candidate_axis_z,
        key=lambda candidate: abs(float(np.min(base_z + v_coefficients * candidate)) - table_centerline_height_m),
    )
    axis_z = min(max(axis_z, -maximum_axis_z), maximum_axis_z)
    vertical_weight = axis_z / max(maximum_axis_z, 1.0e-9)
    bow_axis = vertical_weight * upward + math.sqrt(max(0.0, 1.0 - vertical_weight**2)) * lateral

    radial_start = radius * (math.cos(initial_angle) * chord_axis + math.sin(initial_angle) * bow_axis)
    center = start - radial_start
    routed = np.stack(
        [
            center
            + radius
            * (
                math.cos(initial_angle + cumulative_angle) * chord_axis
                + math.sin(initial_angle + cumulative_angle) * bow_axis
            )
            for cumulative_angle in cumulative_angles
        ]
    )
    routed[-1] = route_endpoint
    result = np.concatenate((prefix, routed, suffix[1:]), axis=0)
    actual_lengths = np.linalg.norm(np.diff(result, axis=0), axis=1)
    if not np.allclose(actual_lengths, lengths, rtol=0.0, atol=_CABLE_ROUTE_RESTORE_ATOL_M):
        raise RuntimeError(
            "Dual-rack cable routing did not preserve segment rest lengths: "
            f"maximum error={float(np.max(np.abs(actual_lengths - lengths))):.3e} m."
        )
    return result


def route_dual_rack_cable_points_torch(
    fixed_prefix_points: torch.Tensor,
    anchored_endpoint: torch.Tensor,
    segment_lengths: torch.Tensor,
    *,
    fixed_suffix_points: torch.Tensor | None = None,
    table_centerline_height_m: float = DUAL_RACK_CABLE_TABLE_CENTERLINE_HEIGHT_M,
    bisection_iterations: int = 48,
) -> torch.Tensor:
    """Batched Torch equivalent of :func:`route_dual_rack_cable_points_numpy`."""
    import torch

    prefix = torch.as_tensor(fixed_prefix_points)
    final_endpoint = torch.as_tensor(anchored_endpoint, device=prefix.device, dtype=prefix.dtype)
    lengths = torch.as_tensor(segment_lengths, device=prefix.device, dtype=prefix.dtype)
    if prefix.ndim != 3 or prefix.shape[-1] != 3 or prefix.shape[1] < 2:
        raise ValueError(f"fixed_prefix_points must have shape (N, K+1, 3), got {tuple(prefix.shape)}.")
    if tuple(final_endpoint.shape) != (prefix.shape[0], 3):
        raise ValueError(
            f"anchored_endpoint must have shape ({prefix.shape[0]}, 3), got {tuple(final_endpoint.shape)}."
        )
    if lengths.ndim != 1 or lengths.numel() < prefix.shape[1] - 1:
        raise ValueError("segment_lengths must be a one-dimensional full cable profile.")
    if type(bisection_iterations) is not int or bisection_iterations < 1:
        raise ValueError("bisection_iterations must be a positive plain integer.")

    fixed_segment_count = prefix.shape[1] - 1
    if fixed_suffix_points is None:
        suffix = final_endpoint[:, None]
    else:
        suffix = torch.as_tensor(fixed_suffix_points, device=prefix.device, dtype=prefix.dtype)
        if suffix.ndim != 3 or suffix.shape[0] != prefix.shape[0] or suffix.shape[2] != 3 or suffix.shape[1] < 2:
            raise ValueError(
                f"fixed_suffix_points must have shape ({prefix.shape[0]}, L+1, 3), got {tuple(suffix.shape)}."
            )
        if not bool(torch.allclose(suffix[:, -1], final_endpoint, rtol=0.0, atol=1.0e-6)):
            raise ValueError("The fixed cable suffix must end at anchored_endpoint.")
    fixed_suffix_segment_count = suffix.shape[1] - 1
    if fixed_segment_count + fixed_suffix_segment_count >= lengths.numel():
        raise ValueError("Fixed cable prefix and suffix leave no routed middle segment.")
    if fixed_suffix_segment_count and not bool(
        torch.allclose(
            torch.linalg.vector_norm(torch.diff(suffix, dim=1), dim=-1),
            lengths[-fixed_suffix_segment_count:][None].expand(prefix.shape[0], -1),
            rtol=0.0,
            atol=2.0e-6,
        )
    ):
        raise ValueError("Fixed cable suffix does not preserve the authored trailing rest lengths.")
    remaining_lengths = lengths[fixed_segment_count : lengths.numel() - fixed_suffix_segment_count]
    remaining_length = remaining_lengths.sum()
    start = prefix[:, -1]
    route_endpoint = suffix[:, 0]
    chord = route_endpoint - start
    direct_distance = torch.linalg.vector_norm(chord, dim=-1)
    if bool((direct_distance <= 1.0e-9).any()) or bool((direct_distance > remaining_length + 1.0e-6).any()):
        raise ValueError("A routed cable endpoint is degenerate or exceeds the remaining rest length.")

    maximum_half_chord = 0.5 * remaining_lengths.max()

    def total_angle(radius: torch.Tensor) -> torch.Tensor:
        return (2.0 * torch.asin((remaining_lengths[None] / (2.0 * radius[:, None])).clamp(-1.0, 1.0))).sum(dim=-1)

    count = prefix.shape[0]
    lower_radius = torch.full(
        (count,),
        float(maximum_half_chord) * (1.0 + 1.0e-6),
        device=prefix.device,
        dtype=prefix.dtype,
    )
    upper_radius = torch.full_like(lower_radius, max(float(remaining_length), 2.0 * float(maximum_half_chord)))
    for _ in range(bisection_iterations):
        midpoint = 0.5 * (lower_radius + upper_radius)
        angle_above_turn = total_angle(midpoint) > 2.0 * math.pi
        lower_radius = torch.where(angle_above_turn, midpoint, lower_radius)
        upper_radius = torch.where(angle_above_turn, upper_radius, midpoint)
    lower_radius = upper_radius

    def endpoint_distance(radius: torch.Tensor) -> torch.Tensor:
        angle = total_angle(radius)
        return 2.0 * radius * torch.sin(0.5 * angle)

    upper_radius = torch.maximum(upper_radius, torch.full_like(upper_radius, float(remaining_length)))
    # The straight-chain limit at this bound is already sufficient for the
    # configured workcell. Keep a bounded expansion for explicit custom calls.
    for _ in range(16):
        needs_more = endpoint_distance(upper_radius) < direct_distance
        upper_radius = torch.where(needs_more, 2.0 * upper_radius, upper_radius)
    if bool((endpoint_distance(upper_radius) < direct_distance).any()):
        raise RuntimeError("Could not bracket a batched dual-rack cable circle radius.")
    for _ in range(bisection_iterations):
        midpoint = 0.5 * (lower_radius + upper_radius)
        below_chord = endpoint_distance(midpoint) < direct_distance
        lower_radius = torch.where(below_chord, midpoint, lower_radius)
        upper_radius = torch.where(below_chord, upper_radius, midpoint)
    radius = 0.5 * (lower_radius + upper_radius)
    segment_angles = 2.0 * torch.asin((remaining_lengths[None] / (2.0 * radius[:, None])).clamp(-1.0, 1.0))
    angle = segment_angles.sum(dim=-1)
    chord_axis = chord / direct_distance[:, None]

    global_up = torch.tensor((0.0, 0.0, 1.0), device=prefix.device, dtype=prefix.dtype).expand_as(chord_axis)
    upward = global_up - (global_up * chord_axis).sum(dim=-1, keepdim=True) * chord_axis
    upward_norm = torch.linalg.vector_norm(upward, dim=-1, keepdim=True)
    fallback = torch.tensor((1.0, 0.0, 0.0), device=prefix.device, dtype=prefix.dtype).expand_as(upward)
    upward = torch.where(upward_norm > 1.0e-9, upward / upward_norm.clamp_min(1.0e-9), fallback)
    lateral = torch.linalg.cross(chord_axis, upward, dim=-1)
    lateral /= torch.linalg.vector_norm(lateral, dim=-1, keepdim=True).clamp_min(1.0e-9)

    initial_angle = -0.5 * angle - 0.5 * math.pi
    cumulative_angles = torch.cumsum(segment_angles, dim=-1)
    u_coefficients = radius[:, None] * (
        torch.cos(initial_angle[:, None] + cumulative_angles) - torch.cos(initial_angle)[:, None]
    )
    v_coefficients = radius[:, None] * (
        torch.sin(initial_angle[:, None] + cumulative_angles) - torch.sin(initial_angle)[:, None]
    )
    base_z = start[:, 2:3] + u_coefficients * chord_axis[:, 2:3]
    maximum_axis_z = upward[:, 2].abs()
    lower_allowed = -maximum_axis_z
    upper_allowed = maximum_axis_z
    table_height = torch.as_tensor(table_centerline_height_m, device=prefix.device, dtype=prefix.dtype)
    positive = v_coefficients > 1.0e-10
    negative = v_coefficients < -1.0e-10
    lower_candidates = torch.where(
        positive,
        (table_height - base_z) / v_coefficients.clamp_min(1.0e-10),
        torch.full_like(v_coefficients, -torch.inf),
    )
    upper_candidates = torch.where(
        negative,
        (table_height - base_z) / v_coefficients.clamp_max(-1.0e-10),
        torch.full_like(v_coefficients, torch.inf),
    )
    lower_allowed = torch.maximum(lower_allowed, lower_candidates.amax(dim=-1))
    upper_allowed = torch.minimum(upper_allowed, upper_candidates.amin(dim=-1))
    if bool((lower_allowed > upper_allowed + 1.0e-6).any()):
        raise ValueError("No batched dual-rack cable circle plane satisfies table clearance.")
    minimum_at_lower = (base_z + v_coefficients * lower_allowed[:, None]).amin(dim=-1)
    minimum_at_upper = (base_z + v_coefficients * upper_allowed[:, None]).amin(dim=-1)
    use_lower = (minimum_at_lower - table_height).abs() <= (minimum_at_upper - table_height).abs()
    axis_z = torch.where(use_lower, lower_allowed, upper_allowed).clamp(-maximum_axis_z, maximum_axis_z)
    vertical_weight = axis_z / maximum_axis_z.clamp_min(1.0e-9)
    bow_axis = (
        vertical_weight[:, None] * upward
        + torch.sqrt((1.0 - vertical_weight.square()).clamp_min(0.0))[:, None] * lateral
    )

    radial_start = radius[:, None] * (
        torch.cos(initial_angle)[:, None] * chord_axis + torch.sin(initial_angle)[:, None] * bow_axis
    )
    center = start - radial_start
    sample_angles = initial_angle[:, None] + cumulative_angles
    routed = center[:, None] + radius[:, None, None] * (
        torch.cos(sample_angles)[..., None] * chord_axis[:, None]
        + torch.sin(sample_angles)[..., None] * bow_axis[:, None]
    )
    routed[:, -1] = route_endpoint
    return torch.cat((prefix, routed, suffix[:, 1:]), dim=1)


def dual_rack_cable_body_poses_torch(
    cable_points: torch.Tensor,
    *,
    free_plug_orientation_xyzw: torch.Tensor,
    prefix_rotations_xyzw: torch.Tensor,
) -> torch.Tensor:
    """Convert routed endpoints to cable body center poses in xyzw order."""
    import torch

    points = torch.as_tensor(cable_points)
    if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] < 2:
        raise ValueError(f"cable_points must have shape (N, S+1, 3), got {tuple(points.shape)}.")
    segment = points[:, 1:] - points[:, :-1]
    direction = segment / torch.linalg.vector_norm(segment, dim=-1, keepdim=True).clamp_min(1.0e-9)
    dz = direction[..., 2]
    regular_w = torch.sqrt((1.0 + dz).clamp_min(0.0) * 0.5)
    denominator = (2.0 * regular_w).clamp_min(1.0e-8)
    regular_xyz = torch.stack(
        (-direction[..., 1] / denominator, direction[..., 0] / denominator, torch.zeros_like(dz)), dim=-1
    )
    regular = torch.cat((regular_xyz, regular_w[..., None]), dim=-1)
    opposite = torch.zeros_like(regular)
    opposite[..., 0] = 1.0
    orientation = torch.where((dz < -1.0 + 1.0e-6)[..., None], opposite, regular)
    orientation /= torch.linalg.vector_norm(orientation, dim=-1, keepdim=True).clamp_min(1.0e-9)

    prefix_rotations = torch.as_tensor(prefix_rotations_xyzw, device=points.device, dtype=points.dtype)
    free_orientation = torch.as_tensor(free_plug_orientation_xyzw, device=points.device, dtype=points.dtype)
    prefix_count = prefix_rotations.shape[0]
    if tuple(prefix_rotations.shape) != (prefix_count, 4) or tuple(free_orientation.shape) != (
        points.shape[0],
        4,
    ):
        raise ValueError("Cable prefix/free-plug quaternion shapes are invalid.")
    from isaaclab.utils import math as math_utils

    synchronized = math_utils.quat_mul(
        free_orientation[:, None].expand(-1, prefix_count, -1).reshape(-1, 4),
        prefix_rotations[None].expand(points.shape[0], -1, -1).reshape(-1, 4),
    ).reshape(points.shape[0], prefix_count, 4)
    orientation[:, :prefix_count] = synchronized
    center = 0.5 * (points[:, :-1] + points[:, 1:])
    return torch.cat((center, orientation), dim=-1)


def dual_rack_cable_workcell_intersections_numpy(
    cable_points: np.ndarray,
    *,
    cable_radius_m: float,
    cfg: DualRackWorkcellCfg = DUAL_RACK_WORKCELL_CFG,
) -> tuple[str, ...]:
    """Return cuboids intersected by any cable span using expanded-AABB slabs."""
    points = np.asarray(cable_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"cable_points must be finite with shape (S+1, 3), got {points.shape}.")
    if not math.isfinite(cable_radius_m) or cable_radius_m <= 0.0:
        raise ValueError("cable_radius_m must be finite and positive.")
    p0 = points[:-1]
    direction = points[1:] - p0
    result: list[str] = []
    for box in (item for item in cfg.boxes if item.collidable):
        center = np.asarray(box.center_m)
        half_size = 0.5 * np.asarray(box.size_m) + cable_radius_m
        lower, upper = center - half_size, center + half_size
        parallel = np.abs(direction) <= 1.0e-12
        parallel_outside = parallel & ((p0 < lower) | (p0 > upper))
        safe_direction = np.where(parallel, 1.0, direction)
        first = (lower - p0) / safe_direction
        second = (upper - p0) / safe_direction
        axis_enter = np.where(parallel, -np.inf, np.minimum(first, second))
        axis_exit = np.where(parallel, np.inf, np.maximum(first, second))
        enter = np.maximum(axis_enter.max(axis=-1), 0.0)
        exit = np.minimum(axis_exit.min(axis=-1), 1.0)
        if np.any((~parallel_outside.any(axis=-1)) & (enter <= exit)):
            result.append(box.name)
    return tuple(result)


def dual_rack_cable_workcell_intersection_mask_torch(
    cable_points: torch.Tensor,
    *,
    cable_radius_m: float,
    cfg: DualRackWorkcellCfg = DUAL_RACK_WORKCELL_CFG,
) -> torch.Tensor:
    """Return one conservative cable/cuboid intersection flag per environment."""
    import torch

    points = torch.as_tensor(cable_points)
    if points.ndim != 3 or points.shape[1] < 2 or points.shape[2] != 3 or not points.dtype.is_floating_point:
        raise ValueError(f"cable_points must be floating point with shape (N, S+1, 3), got {points.shape}.")
    if not math.isfinite(cable_radius_m) or cable_radius_m <= 0.0:
        raise ValueError("cable_radius_m must be finite and positive.")
    boxes = tuple(box for box in cfg.boxes if box.collidable)
    centers = torch.tensor([box.center_m for box in boxes], device=points.device, dtype=points.dtype)
    half_sizes = (
        0.5
        * torch.tensor(
            [box.size_m for box in boxes],
            device=points.device,
            dtype=points.dtype,
        )
        + cable_radius_m
    )
    lower = centers - half_sizes
    upper = centers + half_sizes
    p0 = points[:, :-1, None, :]
    direction = (points[:, 1:] - points[:, :-1])[:, :, None, :]
    parallel = direction.abs() <= 1.0e-9
    parallel_outside = parallel & ((p0 < lower[None, None]) | (p0 > upper[None, None]))
    safe_direction = torch.where(parallel, torch.ones_like(direction), direction)
    first = (lower[None, None] - p0) / safe_direction
    second = (upper[None, None] - p0) / safe_direction
    axis_enter = torch.where(parallel, torch.full_like(first, -torch.inf), torch.minimum(first, second))
    axis_exit = torch.where(parallel, torch.full_like(first, torch.inf), torch.maximum(first, second))
    enter = torch.maximum(axis_enter.amax(dim=-1), torch.zeros((), device=points.device, dtype=points.dtype))
    exit = torch.minimum(axis_exit.amin(dim=-1), torch.ones((), device=points.device, dtype=points.dtype))
    per_segment_box = (~parallel_outside.any(dim=-1)) & (enter <= exit)
    return per_segment_box.any(dim=(1, 2))


def dual_rack_workcell_contract(cfg: DualRackWorkcellCfg = DUAL_RACK_WORKCELL_CFG) -> dict[str, object]:
    """Return the path-independent physical/presentation workcell contract."""
    return {
        "contract_version": cfg.contract_version,
        "target_task_translation_m": DUAL_RACK_TARGET_TASK_TRANSLATION,
        "target_task_rotation_xyzw": DUAL_RACK_TASK_ROTATION_XYZW,
        "target_socket_position_e_m": DUAL_RACK_TARGET_SOCKET_POSITION_E,
        "anchored_connector": {
            "task_translation_m": cfg.anchored_connector.task_translation_m,
            "task_rotation_xyzw": cfg.anchored_connector.task_rotation_xyzw,
            "socket_position_e_m": DUAL_RACK_ANCHORED_SOCKET_POSITION_E,
            "state_semantics": "static-seated-socket-plug-latch-not-persisted-in-reset-rows",
            "cable_attachment": "four-pinned-trailing-segments-ending-at-anchored-plug-cable-exit",
        },
        "cable_contact_damping_n_s_m": cfg.cable_contact_damping_n_s_m,
        "presentation_kind": cfg.presentation_kind,
        "switch_socket_local_bounds_m": {
            "minimum": DUAL_RACK_SWITCH_MIN_SOCKET_LOCAL,
            "maximum": DUAL_RACK_SWITCH_MAX_SOCKET_LOCAL,
        },
        "collision_geometry": {
            "representation": "axis-aligned-static-cuboids",
            "insertion_openings": "36x30-mm-front-panel-corridor-centered-on-each-real-sdf-socket",
            "boxes": tuple(
                {
                    "name": box.name,
                    "center_m": box.center_m,
                    "size_m": box.size_m,
                }
                for box in cfg.boxes
                if box.collidable
            ),
        },
        "presentation": {
            "detailed_racks": "two-AS4610-visual-instances-aligned-to-real-sockets",
            "t_slot_representation": "recessed-cross-web-plus-four-corner-rails-over-hidden-box-contact",
            "t_slot_visual_box_count": sum(box.visible for box in cfg.boxes),
            "visual_only_profile_piece_count": sum(box.visible and not box.collidable for box in cfg.boxes),
        },
    }


__all__ = [
    "DUAL_RACK_ANCHORED_ACCENT_COLOR",
    "DUAL_RACK_ANCHORED_SOCKET_POSITION_E",
    "DUAL_RACK_ANCHORED_TASK_TRANSLATION",
    "DUAL_RACK_CABLE_TABLE_CENTERLINE_HEIGHT_M",
    "DUAL_RACK_TARGET_ACCENT_COLOR",
    "DUAL_RACK_TARGET_SOCKET_POSITION_E",
    "DUAL_RACK_TARGET_TASK_TRANSLATION",
    "DUAL_RACK_TASK_ROTATION_XYZW",
    "DUAL_RACK_WORKCELL_CFG",
    "DUAL_RACK_WORKCELL_CONTRACT_VERSION",
    "DualRackAnchoredConnectorCfg",
    "DualRackWorkcellBoxCfg",
    "DualRackWorkcellCfg",
    "dual_rack_cable_body_poses_torch",
    "dual_rack_cable_workcell_intersection_mask_torch",
    "dual_rack_cable_workcell_intersections_numpy",
    "dual_rack_workcell_contract",
    "make_dual_rack_workcell_cfg",
    "route_dual_rack_cable_points_numpy",
    "route_dual_rack_cable_points_torch",
]
