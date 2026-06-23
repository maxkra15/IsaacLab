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

# add_rod_graph places each segment's body frame at the edge's start node u
# (edge (u, v), +Z from u->v), so cable_local_pos=(0, 0, 0) welds at u.
FRIDGE_POS = (0.0, 0.0, 0.5)
CABLE1_TAIL_NODE_42 = (-0.18810473382472992, 0.3453156650066376, -0.25986239314079285)
CABLE1_ANCHOR_NODE = CABLE1_TAIL_NODE_42
ANCHOR_POS = tuple(p + n for p, n in zip(FRIDGE_POS, CABLE1_ANCHOR_NODE))
CABLE_HEAD_TO_PLUG_ORIGIN_LOCAL_Z = 0.022

# Socket mouth pose, expressed in the environment frame. USD stores xformOp:orient as (w, x, y, z);
# Isaac Lab math helpers and action offsets use (x, y, z, w).
SOCKET_MOUTH_POS = (-0.259345, 0.344709, 0.28698)
SOCKET_ROT_QUAT_WXYZ = (0.984808, 0.173648, 0.0, 0.0)
SOCKET_ROT_QUAT_XYZW = quat_xyzw_from_wxyz(SOCKET_ROT_QUAT_WXYZ)
SOCKET_COLLISION_XFORM_SUFFIX = "/Cable008/SocketCollision"
SOCKET_COLLISION_MESH_SUFFIX = f"{SOCKET_COLLISION_XFORM_SUFFIX}/Cable008_SocketCollision"
SOCKET_COLLISION_MESH_PATTERN = rf".*/Fridge{SOCKET_COLLISION_MESH_SUFFIX}.*"

# The connector-housing body collision is authored two ways under Fridge/Cable008. The robot
# (MJWarp entry) collides with the per-fragment convex hulls under Collisions/Cable008_Collider* --
# cheap, accurate convex collision in MuJoCo-Warp. The deformable hose (VBD entry) instead collides
# with a single welded mesh under BodyCollision/Cable008_BodyCollision, so the per-substep
# particle-vs-shape soft-contact pass runs over one shape rather than the full hull set (the cost
# scales with the shape count). Both are world-static shapes (shape_body < 0); each entry selects
# its own representation by label so the robot does not also pick up the (concave) welded mesh.
FRIDGE_BODY_COLLISION_MESH_PATTERN = r".*/Fridge/Cable008/Collisions/Cable008_Collider\d+.*"
FRIDGE_BODY_WELDED_MESH_PATTERN = r".*/Fridge/Cable008/BodyCollision/Cable008_BodyCollision.*"

# Grasp point relative to the plug frame. The offset is biased toward the fridge/socket side of the
# plug flange so the full finger pad, not just its trailing edge, carries the plug.
CABLE_RADIUS = 0.003
PLUG_GRASP_OFFSET = (0.0, -CABLE_RADIUS + 0.002, 0.003)
CONNECTOR_TIP_LEN = 0.014106234
