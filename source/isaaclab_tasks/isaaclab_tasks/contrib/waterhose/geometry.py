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

# Connector-housing collision: a single welded mesh under BodyCollision/Cable008_BodyCollision
# (a world-static shape, shape_body < 0). The robot's right gripper collides it through the MJWarp
# entry, which -- with use_mujoco_contacts=False -- runs Newton's collision pipeline directly against
# this one concave mesh. No per-fragment convex-hull decomposition is needed (MuJoCo-Warp would have
# convexified a single concave mesh, but its compiled geom is inert when Newton supplies the contacts),
# so the collision-pair enumeration stays O(1 housing shape) instead of O(hundreds of hulls). The mesh
# excludes the socket bore, leaving it open for plug insertion.
FRIDGE_HOUSING_COLLISION_MESH_PATTERN = r".*/Fridge/Cable008/BodyCollision/Cable008_BodyCollision.*"

# Below-socket collision wall. A SOLID box occupying the fridge body directly below/behind the socket so
# the robot gripper cannot dip/tunnel into the concave connector housing while inserting the plug. It is
# a per-env kinematic body (a ``GeoType.BOX`` primitive -> analytic solid contact, so box-vs-mesh stays
# robust unlike the hollow housing mesh), cloned by the replicator, and rendered only under the Newton
# viewer's "Collisions" toggle (its VISIBLE flag is cleared at model-init -- see
# ``_hide_fridge_collider_visuals``).
#
# CRITICAL: keep EVERY dimension thick (>= ~0.1 m). MuJoCo-Warp does no continuous collision detection,
# so a thin slab (e.g. a 0.02 m wall) is stepped straight through by the stiff position-controlled
# gripper between the once-per-step collision checks -- the contact pair IS generated and resolves as a
# hard constraint, but a thin slab simply has no overlap at the sampled configuration. A thick solid
# block keeps the gripper overlapping for several steps, so the contact is reliably caught. (This is also
# why the 245 solid convex hulls held but the single concave housing *shell* mesh tunnels.)
#
# Placement (environment frame): the +y face sits at the socket plane (y~0.341) where the gripper
# approaches from the robot side (+y); the box extends in -y into the fridge body (unused space), so the
# thickness never intrudes on the grasp/insert corridor. The socket mouth is at z=0.28698; the box top
# (z~0.215) stays below it so it clears the seated plug. Tune the size/pose live in the viewer, but do
# not let any dimension go thin. The token must appear in the spawned collider's shape label so the hide
# hook matches it.
FRIDGE_FLOOR_SIZE = (0.5, 0.3, 0.5)
FRIDGE_FLOOR_POS = (-0.259345, 0.191, -0.035)
FRIDGE_FLOOR_COLLISION_TOKEN = "FridgeFloor"

# Grasp point relative to the plug frame. The offset is biased toward the fridge/socket side of the
# plug flange so the full finger pad, not just its trailing edge, carries the plug.
CABLE_RADIUS = 0.003
PLUG_GRASP_OFFSET = (0.0, -CABLE_RADIUS + 0.002, 0.003)
CONNECTOR_TIP_LEN = 0.014106234
