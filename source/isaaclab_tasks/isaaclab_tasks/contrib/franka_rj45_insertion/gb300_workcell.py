# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native CAD port registration and lightweight collision shell for a GB300 rack."""

from __future__ import annotations

from .dual_rack_workcell import (
    DualRackAnchoredConnectorCfg,
    DualRackWorkcellBoxCfg,
    DualRackWorkcellCfg,
)
from .gb300_asset import gb300_asset_contract

GB300_WORKCELL_CONTRACT_VERSION = 3
GB300_CABLE_CONTACT_DAMPING_N_S_M = 200.0

# The task stays in the standard Franka frame. The composed SimReady cabinet's
# front is local +X, so a -90-degree yaw maps that face to environment -Y and
# aligns its SN2201 jacks with the identity-oriented RJ45 insertion assembly.
# Keep the payload on a child prim: ``/external`` already authors a +90-degree
# root rotation and a +0.108 m lift, which are part of these composed values.
GB300_WORKCELL_FRAME_TRANSLATION_E = (0.0, 0.0, 0.0)
GB300_WORKCELL_FRAME_ROTATION_XYZW = (0.0, 0.0, 0.0, 1.0)
GB300_TASK_ROTATION_XYZW = GB300_WORKCELL_FRAME_ROTATION_XYZW
GB300_ROBOT_POSITION_E = GB300_WORKCELL_FRAME_TRANSLATION_E
GB300_ROBOT_ROTATION_XYZW = GB300_WORKCELL_FRAME_ROTATION_XYZW

GB300_PRESENTATION_TRANSLATION_E = (0.5800, 0.6833, -1.89475)
GB300_PRESENTATION_ROTATION_XYZW = (0.0, 0.0, -0.7071067811865476, 0.7071067811865476)
GB300_SIMREADY_ROOT_TRANSLATION_M = (0.0, -2.398081733190338e-17, 0.10800000000000003)
GB300_SIMREADY_ROOT_ROTATION_XYZ_DEG = (0.0, 0.0, 90.0)
GB300_SIMREADY_COMPOSED_BOUNDS_MIN_M = (-0.9940365173959482, -0.35440000280737877, -0.032516271868500105)
GB300_SIMREADY_COMPOSED_BOUNDS_MAX_M = (0.6283000076413155, 0.30250001051441067, 2.41371728817683)
GB300_STUDIO_FLOOR_HEIGHT_M = GB300_PRESENTATION_TRANSLATION_E[2] + GB300_SIMREADY_COMPOSED_BOUNDS_MIN_M[2]
"""Exact world height of the lowest composed SimReady cabinet point [m]."""
GB300_PRESENTATION_RACK_FRONT_Y_E = GB300_PRESENTATION_TRANSLATION_E[1] - GB300_SIMREADY_COMPOSED_BOUNDS_MAX_M[0]
"""Front-most world-space plane of the first rotated cabinet bounds [m]."""
GB300_PRESENTATION_RACK_COUNT = 8
GB300_PRESENTATION_RACK_SPACING_X_M = GB300_SIMREADY_COMPOSED_BOUNDS_MAX_M[1] - GB300_SIMREADY_COMPOSED_BOUNDS_MIN_M[1]
"""Exact rotated cabinet width, so adjacent SimReady bounds touch [m]."""
GB300_SCENERY_RACK_ORIGIN_E = (
    GB300_PRESENTATION_TRANSLATION_E[0] + GB300_PRESENTATION_RACK_SPACING_X_M,
    GB300_PRESENTATION_TRANSLATION_E[1],
    GB300_PRESENTATION_TRANSLATION_E[2],
)
GB300_SCENERY_RACK_ROTATION_XYZW = GB300_PRESENTATION_ROTATION_XYZW
GB300_PRESENTATION_RACK_TRANSLATIONS_E = tuple(
    (
        GB300_PRESENTATION_TRANSLATION_E[0] + index * GB300_PRESENTATION_RACK_SPACING_X_M,
        GB300_PRESENTATION_TRANSLATION_E[1],
        GB300_PRESENTATION_TRANSLATION_E[2],
    )
    for index in range(GB300_PRESENTATION_RACK_COUNT)
)
GB300_PRESENTATION_RACK_ROTATIONS_XYZW = (GB300_PRESENTATION_ROTATION_XYZW,) * GB300_PRESENTATION_RACK_COUNT
"""Eight consistently front-facing cabinets; only the first owns task physics."""

_RJ45_SOCKET_POSITION_Z_M = 0.011653749272227287
_RJ45_SOCKET_FRONT_TO_ORIGIN_M = 0.01275

# Registration extracted from the pinned SimReady SN2201 mesh. The authored
# switch has 48 1GBase-T jacks in two rows of 24. Four well-spaced columns from
# its robot-reachable center span provide eight targets; one separate jack at
# the left anchors the already-seated cable end. Coordinates are in the
# composed ``/external`` asset frame, before the presentation transform above.
GB300_SN2201_SOURCE_MESH = "/external/tn__0000_NV_MSN2201TOR_08132024_Ze0/tn__0000_NV_MSN2201TOR_08132024_Ze0_Merged"
GB300_SN2201_PORT_FRONT_ASSET_X_M = 0.3546743
GB300_SN2201_TARGET_COLUMN_ASSET_Y_M = (-0.1470448, -0.1050448, -0.0568248, -0.0148248)
GB300_SN2201_ANCHORED_COLUMN_ASSET_Y_M = -0.2030448
GB300_SN2201_PORT_ROW_ASSET_Z_M = (2.03475, 2.04887)


def _registered_socket_position(asset_y: float, asset_z: float) -> tuple[float, float, float]:
    """Map one native SN2201 jack center to an exact socket-body origin [m]."""
    socket_asset_x = GB300_SN2201_PORT_FRONT_ASSET_X_M - _RJ45_SOCKET_FRONT_TO_ORIGIN_M
    return (
        GB300_PRESENTATION_TRANSLATION_E[0] + asset_y,
        GB300_PRESENTATION_TRANSLATION_E[1] - socket_asset_x,
        GB300_PRESENTATION_TRANSLATION_E[2] + asset_z,
    )


GB300_TARGET_SOCKET_POSITIONS_E = tuple(
    _registered_socket_position(asset_y, asset_z)
    for asset_z in GB300_SN2201_PORT_ROW_ASSET_Z_M
    for asset_y in GB300_SN2201_TARGET_COLUMN_ASSET_Y_M
)
"""Eight socket-body origins registered behind native SimReady SN2201 jacks [m]."""

# Use an upper-row central jack for canonical certification. Persisted rows
# still select all eight target ports uniformly.
GB300_DEFAULT_TARGET_SOCKET_POSITION_E = GB300_TARGET_SOCKET_POSITIONS_E[6]
GB300_ANCHORED_SOCKET_POSITION_E = _registered_socket_position(
    GB300_SN2201_ANCHORED_COLUMN_ASSET_Y_M,
    GB300_SN2201_PORT_ROW_ASSET_Z_M[0],
)


def _task_translation(socket_position: tuple[float, float, float]) -> tuple[float, float, float]:
    return (socket_position[0], socket_position[1], socket_position[2] - _RJ45_SOCKET_POSITION_Z_M)


GB300_TARGET_TASK_TRANSLATIONS = tuple(_task_translation(position) for position in GB300_TARGET_SOCKET_POSITIONS_E)
GB300_DEFAULT_TARGET_TASK_TRANSLATION = _task_translation(GB300_DEFAULT_TARGET_SOCKET_POSITION_E)
GB300_ANCHORED_TASK_TRANSLATION = _task_translation(GB300_ANCHORED_SOCKET_POSITION_E)

_RACK_X_MIN = GB300_PRESENTATION_TRANSLATION_E[0] + GB300_SIMREADY_COMPOSED_BOUNDS_MIN_M[1]
_RACK_X_MAX = GB300_PRESENTATION_TRANSLATION_E[0] + GB300_SIMREADY_COMPOSED_BOUNDS_MAX_M[1]
_RACK_Y_REAR = GB300_PRESENTATION_TRANSLATION_E[1] - GB300_SIMREADY_COMPOSED_BOUNDS_MIN_M[0]
_RACK_Z_MIN = GB300_STUDIO_FLOOR_HEIGHT_M
_RACK_Z_MAX = GB300_PRESENTATION_TRANSLATION_E[2] + GB300_SIMREADY_COMPOSED_BOUNDS_MAX_M[2]
_SERVICE_X_MIN = 0.350
_SERVICE_X_MAX = 0.590
_SERVICE_Z_MIN = 0.110
_SERVICE_Z_MAX = 0.185


def _box(
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    collidable: bool,
    visible: bool,
) -> DualRackWorkcellBoxCfg:
    return DualRackWorkcellBoxCfg(
        name=name,
        center_m=center,
        size_m=size,
        color=color,
        collidable=collidable,
        visible=visible,
    )


def _rack_collision_boxes() -> list[DualRackWorkcellBoxCfg]:
    """Return the front-facing cabinet shell with an open robot service side."""
    color = (0.08, 0.095, 0.11)
    thickness = 0.008
    width = _RACK_X_MAX - _RACK_X_MIN
    # Keep the large cabinet shell behind the robot's calibrated Cartesian
    # approach envelope.  The exact sockets own all front-panel contact and a
    # thin backstop still prevents cable/hand travel into the service volume.
    collision_front = 0.45
    depth = _RACK_Y_REAR - collision_front
    height = _RACK_Z_MAX - _RACK_Z_MIN
    x_center = 0.5 * (_RACK_X_MIN + _RACK_X_MAX)
    y_center = 0.5 * (collision_front + _RACK_Y_REAR)
    z_center = 0.5 * (_RACK_Z_MIN + _RACK_Z_MAX)
    result = [
        _box(
            "GB300/Collision/Left",
            (_RACK_X_MIN + 0.5 * thickness, y_center, z_center),
            (thickness, depth, height),
            color,
            collidable=True,
            visible=False,
        ),
        _box(
            "GB300/Collision/Right",
            (_RACK_X_MAX - 0.5 * thickness, y_center, z_center),
            (thickness, depth, height),
            color,
            collidable=True,
            visible=False,
        ),
        _box(
            "GB300/Collision/Rear",
            (x_center, _RACK_Y_REAR - 0.5 * thickness, z_center),
            (width - 2.0 * thickness, thickness, height),
            color,
            collidable=True,
            visible=False,
        ),
        _box(
            "GB300/Collision/Bottom",
            (x_center, y_center, _RACK_Z_MIN + 0.5 * thickness),
            (width, depth, thickness),
            color,
            collidable=True,
            visible=False,
        ),
        _box(
            "GB300/Collision/Top",
            (x_center, y_center, _RACK_Z_MAX - 0.5 * thickness),
            (width, depth, thickness),
            color,
            collidable=True,
            visible=False,
        ),
    ]
    result.extend(
        (
            _box(
                "GB300/Collision/ServiceBayBackstop",
                (0.5 * (_SERVICE_X_MIN + _SERVICE_X_MAX), 0.430, 0.5 * (_SERVICE_Z_MIN + _SERVICE_Z_MAX)),
                (_SERVICE_X_MAX - _SERVICE_X_MIN, thickness, _SERVICE_Z_MAX - _SERVICE_Z_MIN),
                color,
                collidable=True,
                visible=False,
            ),
        )
    )
    return result


GB300_WORKCELL_CFG = DualRackWorkcellCfg(
    anchored_connector=DualRackAnchoredConnectorCfg(
        task_translation_m=GB300_ANCHORED_TASK_TRANSLATION,
        task_rotation_xyzw=GB300_TASK_ROTATION_XYZW,
        accent_color=(0.035, 0.72, 0.28),
    ),
    boxes=tuple(_rack_collision_boxes()),
    target_accent_color=(0.03, 0.62, 0.90),
    cable_contact_damping_n_s_m=GB300_CABLE_CONTACT_DAMPING_N_S_M,
    presentation_kind="gb300",
)


def gb300_workcell_contract() -> dict[str, object]:
    """Return the physical port-selection and presentation contract."""
    return {
        "contract_version": GB300_WORKCELL_CONTRACT_VERSION,
        "target_socket_candidates_e_m": GB300_TARGET_SOCKET_POSITIONS_E,
        "default_target_task_translation_m": GB300_DEFAULT_TARGET_TASK_TRANSLATION,
        "target_selection": "one-uniform-discrete-candidate-per-persisted-reset-row",
        "active_collision": "one-resettable-hidden-exact-rj45-sdf-at-selected-native-jack",
        "inactive_ports": "native-simready-sn2201-cad-only-no-sdf",
        "anchored_socket_position_e_m": GB300_ANCHORED_SOCKET_POSITION_E,
        "anchored_task_translation_m": GB300_ANCHORED_TASK_TRANSLATION,
        "cable_contact_damping_n_s_m": GB300_WORKCELL_CFG.cable_contact_damping_n_s_m,
        "collision_geometry": {
            "representation": "static-recessed-cuboid-cabinet-shell-plus-service-bay-backstop-open-front",
            "boxes": tuple(
                {"name": box.name, "center_m": box.center_m, "size_m": box.size_m}
                for box in GB300_WORKCELL_CFG.boxes
                if box.collidable
            ),
        },
        "presentation_transform": {
            "translation_e_m": GB300_PRESENTATION_TRANSLATION_E,
            "rotation_xyzw": GB300_PRESENTATION_ROTATION_XYZW,
            "composition": "placement-parent-plus-payload-child-preserves-simready-root-xform",
            "simready_root_translation_m": GB300_SIMREADY_ROOT_TRANSLATION_M,
            "simready_root_rotation_xyz_deg": GB300_SIMREADY_ROOT_ROTATION_XYZ_DEG,
            "simready_composed_bounds_min_m": GB300_SIMREADY_COMPOSED_BOUNDS_MIN_M,
            "simready_composed_bounds_max_m": GB300_SIMREADY_COMPOSED_BOUNDS_MAX_M,
            "studio_floor_height_m": GB300_STUDIO_FLOOR_HEIGHT_M,
        },
        "workcell_frame_transform": {
            "translation_e_m": GB300_WORKCELL_FRAME_TRANSLATION_E,
            "rotation_xyzw": GB300_WORKCELL_FRAME_ROTATION_XYZW,
            "scope": "identity-robot-task-and-collision-shell-frame",
        },
        "table": "absent-no-visual-or-contact-shape",
        "native_cad_port_registration": {
            "source_mesh": GB300_SN2201_SOURCE_MESH,
            "authored_port_count": 48,
            "authored_layout": "two-rows-of-24-1gbase-t-rj45-jacks",
            "target_column_asset_y_m": GB300_SN2201_TARGET_COLUMN_ASSET_Y_M,
            "anchored_column_asset_y_m": GB300_SN2201_ANCHORED_COLUMN_ASSET_Y_M,
            "port_row_asset_z_m": GB300_SN2201_PORT_ROW_ASSET_Z_M,
            "port_front_asset_x_m": GB300_SN2201_PORT_FRONT_ASSET_X_M,
            "socket_front_to_origin_m": _RJ45_SOCKET_FRONT_TO_ORIGIN_M,
            "socket_orientation_xyzw": GB300_TASK_ROTATION_XYZW,
            "registration": "hidden-sdf-socket-front-coincident-with-native-cad-jack-no-added-port-visual",
        },
        "simready_asset": gb300_asset_contract(),
    }


__all__ = [
    "GB300_ANCHORED_SOCKET_POSITION_E",
    "GB300_ANCHORED_TASK_TRANSLATION",
    "GB300_CABLE_CONTACT_DAMPING_N_S_M",
    "GB300_DEFAULT_TARGET_SOCKET_POSITION_E",
    "GB300_DEFAULT_TARGET_TASK_TRANSLATION",
    "GB300_PRESENTATION_ROTATION_XYZW",
    "GB300_PRESENTATION_RACK_FRONT_Y_E",
    "GB300_PRESENTATION_RACK_COUNT",
    "GB300_PRESENTATION_RACK_ROTATIONS_XYZW",
    "GB300_PRESENTATION_RACK_SPACING_X_M",
    "GB300_PRESENTATION_RACK_TRANSLATIONS_E",
    "GB300_PRESENTATION_TRANSLATION_E",
    "GB300_SIMREADY_COMPOSED_BOUNDS_MAX_M",
    "GB300_SIMREADY_COMPOSED_BOUNDS_MIN_M",
    "GB300_SIMREADY_ROOT_ROTATION_XYZ_DEG",
    "GB300_SIMREADY_ROOT_TRANSLATION_M",
    "GB300_SN2201_ANCHORED_COLUMN_ASSET_Y_M",
    "GB300_SN2201_PORT_FRONT_ASSET_X_M",
    "GB300_SN2201_PORT_ROW_ASSET_Z_M",
    "GB300_SN2201_SOURCE_MESH",
    "GB300_SN2201_TARGET_COLUMN_ASSET_Y_M",
    "GB300_STUDIO_FLOOR_HEIGHT_M",
    "GB300_SCENERY_RACK_ORIGIN_E",
    "GB300_SCENERY_RACK_ROTATION_XYZW",
    "GB300_TARGET_SOCKET_POSITIONS_E",
    "GB300_TARGET_TASK_TRANSLATIONS",
    "GB300_TASK_ROTATION_XYZW",
    "GB300_WORKCELL_CFG",
    "GB300_WORKCELL_CONTRACT_VERSION",
    "GB300_WORKCELL_FRAME_ROTATION_XYZW",
    "GB300_WORKCELL_FRAME_TRANSLATION_E",
    "GB300_ROBOT_POSITION_E",
    "GB300_ROBOT_ROTATION_XYZW",
    "gb300_workcell_contract",
]
