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
# Seated Plug1 origin, measured at HOLD_INSERTED (instrumented run, tip 4 mm into the bore).
# This is the snap-lock PIN POINT: the dormant fixed joint pulls the plug origin here, so it
# has ~zero linear violation at the moment it is activated.
SOCKET_SNAP_ANCHOR_POS = (-0.258298, 0.345281, 0.276657)
# The anchor BODY must not sit at the pin point: its (import-required) collider would block
# the arriving plug. Park it 50 mm behind the mouth along the bore axis (inside the fridge,
# which the VBD entry does not collide with) and map the pin point back via the attachment's
# target_local_pos — the anchor rot maps +Z onto the bore axis, so the local offset is +50 mm Z.
_SNAP_ANCHOR_SETBACK = 0.05
_BORE_AXIS = (0.0, -0.342020, 0.939693)  # SOCKET_ROT applied to +Z (20 deg about +X)
SOCKET_SNAP_ANCHOR_BODY_POS = tuple(p - _SNAP_ANCHOR_SETBACK * a for p, a in zip(SOCKET_SNAP_ANCHOR_POS, _BORE_AXIS))
SOCKET_SNAP_ANCHOR_LOCAL_OFFSET = (0.0, 0.0, _SNAP_ANCHOR_SETBACK)
# Orientation of the kinematic snap-lock anchor body (xyzw, fed to InitialStateCfg.rot).
# This is deliberately the raw SOCKET_ROT_QUAT_WXYZ tuple, NOT SOCKET_ROT_QUAT_XYZW. It looks
# like a wxyz/xyzw mistake, but it is the long-standing, most-tested value: the snap geometry
# above (measured pin point, setback, local offset) was calibrated with the anchor at the
# orientation this tuple produces when consumed as xyzw, and it has the most stable evidence.
# Swapping to the nominal xyzw socket orientation showed no stability benefit in testing, so
# leave it as-is. NOTE: the snap-lock *engagement* in HOLD_INSERTED is marginally unstable in
# its own right under either orientation (the soft latch occasionally drifts the inserted plug
# and the cable diverges ~mid-HOLD_INSERTED, run-dependent) -- that is a separate open tuning
# item, unmasked once the ALIGN coaxial-axis fix let the demo reach insertion reliably.
SOCKET_SNAP_ANCHOR_ROT = SOCKET_ROT_QUAT_WXYZ
SOCKET_COLLISION_XFORM_SUFFIX = "/Cable008/SocketCollision"
SOCKET_COLLISION_MESH_SUFFIX = f"{SOCKET_COLLISION_XFORM_SUFFIX}/Cable008_SocketCollision"
SOCKET_COLLISION_MESH_PATTERN = rf".*/Fridge{SOCKET_COLLISION_MESH_SUFFIX}.*"

# Grasp point relative to the plug frame. The offset is biased toward the fridge/socket side of the
# plug flange so the full finger pad, not just its trailing edge, carries the plug.
CABLE_RADIUS = 0.003
PLUG_GRASP_OFFSET = (0.0, -CABLE_RADIUS + 0.002, 0.003)
CONNECTOR_TIP_LEN = 0.014106234
