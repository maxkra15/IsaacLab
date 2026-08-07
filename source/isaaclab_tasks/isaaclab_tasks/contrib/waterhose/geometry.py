# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared geometric constants for the waterhose task."""

from __future__ import annotations


def quat_xyzw_from_wxyz(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Convert a USD-authored quaternion to Isaac Lab math convention."""

    w, x, y, z = quat_wxyz
    return (x, y, z, w)


# EE contact frame offset from right_gripper_base. The finger pad spans base-z [-0.0735, -0.1355]
# (base..tip); -0.125 grips in the tip third of the pad so the flat fingertip surface closes on the plug.
RIGHT_GRIPPER_EE_FRAME_POS = (0.0, 0.0, -0.125)
RIGHT_GRIPPER_EE_FRAME_QUAT_XYZW = (0.70710677, 0.70710677, 0.0, 0.0)

# Mirrored contact frame on the left gripper. The sign follows the authored
# ``left_gripper_end_effector`` site in the RBY1 asset.
LEFT_GRIPPER_EE_FRAME_POS = (0.0, 0.0, -0.125)
LEFT_GRIPPER_EE_FRAME_QUAT_XYZW = (-0.70710677, 0.70710677, 0.0, 0.0)

# Newton's USD cable importer places each segment body frame at its midpoint, with local +Z directed
# from the edge's start node toward its end node.
FRIDGE_POS = (0.0, 0.0, 0.5)
# The source cable pose puts the connector 4 mm through the closed housing front. The legacy open
# proxy silently tolerated that overlap; move the complete cable-and-anchor assembly into clear space
# so the watertight housing starts from a valid contact configuration without changing the cable shape.
CABLE_SCENE_OFFSET = (0.0, 0.004, 0.0)
CABLE_POS = tuple(p + offset for p, offset in zip(FRIDGE_POS, CABLE_SCENE_OFFSET))
CABLE1_TAIL_NODE_42 = (-0.18810473382472992, 0.3453156650066376, -0.25986239314079285)
CABLE1_ANCHOR_NODE = CABLE1_TAIL_NODE_42
ANCHOR_POS = tuple(p + n for p, n in zip(CABLE_POS, CABLE1_ANCHOR_NODE))

# Authored plug pose expressed in cable segment 0's original start-node frame. Segment 0 is the long,
# rigid connector-bearing span and is intentionally excluded from flexible-tail resampling. The plug
# mesh is added directly to this body, so Newton derives one compound mass, center of mass, and inertia.
CONNECTOR_LOCAL_POS = (-7.4394047e-05, 2.1046400e-04, 2.4835587e-02)
CONNECTOR_LOCAL_QUAT_XYZW = (8.3250636e-03, -9.9994665e-01, -5.7898150e-03, 1.9637942e-03)
CONNECTOR_MASS = 0.001

# Socket mouth pose, expressed in the environment frame. USD stores xformOp:orient as (w, x, y, z);
# Isaac Lab math helpers and action offsets use (x, y, z, w).
SOCKET_MOUTH_POS = (-0.259345, 0.344709, 0.28698)
SOCKET_ROT_QUAT_WXYZ = (0.984808, 0.173648, 0.0, 0.0)
SOCKET_ROT_QUAT_XYZW = quat_xyzw_from_wxyz(SOCKET_ROT_QUAT_WXYZ)
# Signed locations of the connector's physical +Z tip along the canonical socket axis. Align the
# compound connector with clear space in front of the bore; rotating it at the mouth lets the tip
# catch the socket edge and lever the cable out of the gripper. Once aligned, the controller freezes
# the corrected orientation and follows the axis straight to the seated pose.
SOCKET_ALIGN_TIP_DEPTH = -0.030
# The physical +Z connector face docks against the near face of the socket SDF (about -3.1 mm).
# Keep roughly 1 mm of solver/contact clearance instead of driving the 7.2 mm flange into the washer.
SOCKET_SEATED_TIP_DEPTH = -0.004
# A physically seated connector can settle behind the commanded face-contact depth under compliant
# gripper and socket contact. Treat insertion depth as a retained range instead of requiring proximity
# to the commanded target: the visible Newton trajectory settles around -14 mm while remaining
# coaxial and sub-millimetre centred. The upper bound still rejects excessive forward penetration.
SOCKET_RETAINED_MIN_TIP_DEPTH = -0.016
SOCKET_RETAINED_MAX_TIP_DEPTH = 0.001
SOCKET_RETAINED_RADIAL_TOLERANCE = 0.001
# About 2.6 degrees. The connector settles near 2 degrees under pure socket contact, so a tighter
# threshold flickers even while depth and sub-millimetre radial alignment remain physically seated.
SOCKET_RETAINED_AXIS_COS = 0.999
SOCKET_COLLISION_XFORM_SUFFIX = "/Cable008/SocketCollision"
SOCKET_COLLISION_MESH_SUFFIX = f"{SOCKET_COLLISION_XFORM_SUFFIX}/Cable008_SocketCollision"
SOCKET_COLLISION_MESH_PATTERN = rf".*/Fridge{SOCKET_COLLISION_MESH_SUFFIX}.*"

# Grasp point relative to the plug frame. The offset is biased toward the fridge/socket side of the
# plug flange so the full finger pad, not just its trailing edge, carries the plug.
CABLE_RADIUS = 0.003
PLUG_GRASP_OFFSET = (0.0, -CABLE_RADIUS + 0.002, 0.003)
# Positive-Z extent of the connector imported from Newton's canonical plug asset.
CONNECTOR_TIP_LEN = 0.00802607
# Centre of the connected +Z flange. (A second 3.1 mm ring in the source point array is orphaned and
# has no faces, so it must not be treated as a physical insertion nose.)
CONNECTOR_TIP_LOCAL_POS = (0.00002717, 0.00008527, CONNECTOR_TIP_LEN)
