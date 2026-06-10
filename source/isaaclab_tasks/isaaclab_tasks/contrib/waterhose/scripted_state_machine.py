# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted IK state machine for the RBY1 waterhose pick-insert-extract demo.

Phases: REST -> APPROACH -> ENGAGE -> GRASP -> HOLD_GRASP -> RETRACT -> SETTLE ->
CARRY -> ALIGN -> INSERT -> HOLD_INSERTED -> RELEASE -> BACKOFF -> REAPPROACH ->
REGRASP -> PULL_OUT -> DONE.

Design (deliberately simple and robust):

* Each phase has a fixed *duration* and computes a single EE *target pose* from a
  snapshot taken on phase entry, plus a fixed geometric offset; the commanded pose
  is a smoothstep blend from the entry pose to the target.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import (
    combine_frame_transforms,
    normalize,
    quat_apply,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

# Grasp contact frame expressed in the right_gripper_base local frame: grip in the TIP third of
# the finger pad. Must match _RIGHT_GRIPPER_EE_FRAME_POS in waterhose_env_cfg.py.
_RIGHT_EE_FROM_BASE_POS = (0.0, 0.0, -0.125)
_RIGHT_EE_FROM_BASE_QUAT = (0.70710677, 0.70710677, 0.0, 0.0)

# Grasp point relative to the plug frame: side grasp biased slightly toward the fridge/socket side
# of the large plug flange. The graspable flange cylinder (dia ~14.6 mm) spans plug-frame z in
# [-7.15, +8.0] mm; +3 mm keeps the pad on the full flange but moves off the cable-side rim so the
# full finger surface, not just the trailing edge, carries the plug.
_CABLE_RADIUS = 0.003
_GRASP_SHIFT = 0.003
_PLUG_GRASP_OFFSET = (0.0, -_CABLE_RADIUS + 0.002, _GRASP_SHIFT)

# Gripper command convention used by the IK action term: +1 fully open, -1 fully closed.
_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSED = -1.0

_SOCKET_MOUTH_POS = (-0.259345, 0.344709, 0.28698)
# Matches waterhose_env_cfg._SOCKET_ROT: authored as USD-style (w, x, y, z).
_SOCKET_ROT_WXYZ = (0.984808, 0.173648, 0.0, 0.0)


def _smoothstep(alpha: torch.Tensor) -> torch.Tensor:
    """Classic 3a^2 - 2a^3 ease-in/ease-out on a clamped [0, 1] interpolant."""
    alpha = torch.clamp(alpha, 0.0, 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _blend_quat(start_quat: torch.Tensor, target_quat: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """Shortest-path normalized-lerp between two quaternions (per env)."""
    target_quat = torch.where(
        torch.sum(start_quat * target_quat, dim=-1, keepdim=True) < 0.0, -target_quat, target_quat
    )
    return normalize(start_quat * (1.0 - blend) + target_quat * blend)


def _xyzw_from_wxyz(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Convert an authored/USD quaternion to IsaacLab math helper convention."""

    w, x, y, z = quat_wxyz
    return (x, y, z, w)


def _quat_from_two_vectors(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Quaternion rotating normalized ``source`` vectors onto ``target`` vectors."""

    source = normalize(source)
    target = normalize(target)
    cross = torch.linalg.cross(source, target, dim=-1)
    cross_norm = torch.linalg.norm(cross, dim=-1, keepdim=True)
    dot = torch.sum(source * target, dim=-1, keepdim=True).clamp(-1.0, 1.0)

    x_axis = torch.zeros_like(source)
    x_axis[:, 0] = 1.0
    y_axis = torch.zeros_like(source)
    y_axis[:, 1] = 1.0
    fallback = torch.linalg.cross(source, x_axis, dim=-1)
    fallback_norm = torch.linalg.norm(fallback, dim=-1, keepdim=True)
    fallback_y = torch.linalg.cross(source, y_axis, dim=-1)
    fallback = torch.where(fallback_norm > 1.0e-6, fallback, fallback_y)
    fallback = normalize(fallback)

    axis = torch.where(cross_norm > 1.0e-6, cross / cross_norm.clamp_min(1.0e-6), fallback)
    angle = torch.atan2(cross_norm.squeeze(-1), dot.squeeze(-1))
    opposite = (cross_norm.squeeze(-1) <= 1.0e-6) & (dot.squeeze(-1) < 0.0)
    angle = torch.where(opposite, torch.full_like(angle, torch.pi), angle)
    return normalize(quat_from_angle_axis(angle, axis))


class WaterhoseDemoState:
    """Per-environment scripted pick-insert-extract state machine."""

    REST = 0
    APPROACH = 1
    ENGAGE = 2
    GRASP = 3
    HOLD_GRASP = 4
    RETRACT = 5
    SETTLE = 6
    CARRY = 7
    ALIGN = 8
    INSERT = 9
    HOLD_INSERTED = 10
    RELEASE = 11
    BACKOFF = 12
    REAPPROACH = 13
    REGRASP = 14
    PULL_OUT = 15
    DONE = 16

    PHASE_NAMES = (
        "REST",
        "APPROACH",
        "ENGAGE",
        "GRASP",
        "HOLD_GRASP",
        "RETRACT",
        "SETTLE",
        "CARRY",
        "ALIGN",
        "INSERT",
        "HOLD_INSERTED",
        "RELEASE",
        "BACKOFF",
        "REAPPROACH",
        "REGRASP",
        "PULL_OUT",
        "DONE",
    )
    # Minimum time spent in each phase [s]; a phase advances once this elapsed AND the EE
    # converged (or a 2x hard timeout). Insert/extract phases get generous time + tolerance.
    DURATIONS = (
        0.25,
        3.0,
        1.5,
        0.5,
        0.5,
        1.5,
        0.3,
        5.0,
        2.0,
        4.0,
        1.0,
        0.8,
        1.5,
        2.0,
        0.7,
        3.0,
        1.0e6,
    )

    def __init__(self, num_envs: int, step_dt: float, device: torch.device | str, settle_time: float, debug: bool):
        self.num_envs = int(num_envs)
        self.step_dt = float(step_dt)
        self.device = device
        self.debug = bool(debug)

        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.elapsed = torch.zeros(self.num_envs, device=device)
        self.last_reported_phase = torch.full((self.num_envs,), -1, dtype=torch.long, device=device)

        # Phase-entry snapshots (world frame) and the commanded pose (base frame).
        self.phase_start_pos_w = torch.zeros((self.num_envs, 3), device=device)
        self.phase_start_quat_w = torch.zeros((self.num_envs, 4), device=device)
        self.phase_start_quat_w[:, 3] = 1.0
        self.phase_plug_pos_w = torch.zeros((self.num_envs, 3), device=device)
        self.phase_plug_quat_w = torch.zeros((self.num_envs, 4), device=device)
        self.phase_plug_quat_w[:, 3] = 1.0
        self.command_pose = torch.zeros((self.num_envs, 7), device=device)
        self.command_pose[:, 6] = 1.0

        durations = list(self.DURATIONS)
        durations[self.REST] = max(float(settle_time), self.step_dt)
        self.durations = torch.tensor(durations, dtype=torch.float32, device=device)

        # Convergence tolerances (generous; combined with the min duration this gives smooth motion).
        self.pos_tolerance = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32, device=device)
        self.rot_tolerance = 15.0 * torch.pi / 180.0

        # Fixed geometric offsets.
        self.plug_grasp_offset = self._vec(_PLUG_GRASP_OFFSET)
        self.approach_offset = self._vec((0.0, 0.08, 0.0))
        self.engage_offset = self._vec((0.01, 0.0, 0.0))
        self.retract_vector = self._vec((0.0, 0.05, 0.0))
        self.connector_axis_local = self._vec((0.0, 0.0, 1.0))

        # Insertion geometry along the bore (= connector) axis, relative to the socket mouth.
        # The scripted target is expressed by the cable connector tip, not the EE frame: CARRY stops
        # with the tip outside the mouth, ALIGN dwells there, and INSERT seats the tip shallowly.
        # The socket bore axis points mostly upward, so this standoff also keeps the gripper lower
        # and away from the insertion mesh during the lift/align motion.
        self.preinsert_standoff = 0.018
        # SHALLOW seat: the authored socket mesh is only a thin shell around the mouth. Do not drive
        # the connector tip through the visible socket asset.
        self.insert_final_depth = 0.0
        self.extract_clearance = 0.05
        self.gripper_backoff_distance = 0.10
        # In the IsaacLab plug frame +Z is the connector axis; the connector tip is ~14 mm along it.
        self.connector_tip_len = 0.014106234

        # EE orientation that grasps the plug from the side: Rx(+90) * Rz(-90).
        z_axis = self._vec((0.0, 0.0, 1.0))
        x_axis = self._vec((1.0, 0.0, 0.0))
        q_rz = quat_from_angle_axis(torch.full((self.num_envs,), -torch.pi / 2.0, device=device), z_axis)
        q_rx = quat_from_angle_axis(torch.full((self.num_envs,), torch.pi / 2.0, device=device), x_axis)
        self.grasp_orientation_offset = normalize(quat_mul(q_rx, q_rz))
        self.connector_tip_pos_in_ee = quat_apply(
            quat_inv(self.grasp_orientation_offset),
            self.connector_axis_local * self.connector_tip_len - self.plug_grasp_offset,
        )

        # Socket mouth pose (env-local; env_origins added at runtime). MUST mirror the spawned
        # Embedded fridge socket collider (waterhose_env_cfg._SOCKET_MOUTH_POS / _SOCKET_ROT). The
        # socket is placed to match the grasped plug's natural post-settle connector presentation,
        # so the bore axis = the connector axis and insertion is a short straight push.
        self.socket_pos_w = self._vec(_SOCKET_MOUTH_POS)
        self.socket_quat_w = normalize(self._vec(_xyzw_from_wxyz(_SOCKET_ROT_WXYZ)))

        self.ee_offset_pos = self._vec(_RIGHT_EE_FROM_BASE_POS)
        self.ee_offset_quat = self._vec(_RIGHT_EE_FROM_BASE_QUAT)

        self._ee_body_id = None
        self._cable_grasp_body_id = 0
        self._cable_tip_body_id = 0

    def _vec(self, values) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.phase[env_ids] = self.REST
        self.elapsed[env_ids] = 0.0
        self.last_reported_phase[env_ids] = -1
        self.phase_start_quat_w[env_ids] = 0.0
        self.phase_start_quat_w[env_ids, 3] = 1.0
        self.phase_plug_pos_w[env_ids] = 0.0
        self.phase_plug_quat_w[env_ids] = 0.0
        self.phase_plug_quat_w[env_ids, 3] = 1.0

    def compute(self, env) -> torch.Tensor:
        robot = env.scene["robot"]
        try:
            plug = env.scene["plug1"]
        except KeyError:
            plug = None
        try:
            cable = env.scene["cable1"]
        except KeyError:
            cable = None

        if self._ee_body_id is None:
            self._ee_body_id = robot.find_bodies("right_gripper_base")[0][0]

        root_pose_w = robot.data.root_link_pose_w.torch
        root_pos_w = root_pose_w[:, :3]
        root_quat_w = root_pose_w[:, 3:]

        ee_base_pos_w = robot.data.body_pos_w.torch[:, self._ee_body_id]
        ee_base_quat_w = robot.data.body_quat_w.torch[:, self._ee_body_id]
        ee_pos_w, ee_quat_w = combine_frame_transforms(
            ee_base_pos_w, ee_base_quat_w, self.ee_offset_pos, self.ee_offset_quat
        )

        if plug is not None:
            plug_pose_w = plug.data.root_link_pose_w.torch
            plug_pos_w = plug_pose_w[:, :3]
            plug_quat_w = normalize(plug_pose_w[:, 3:])
        elif cable is not None:
            plug_pos_w = cable.data.body_pos_w.torch[:, self._cable_grasp_body_id]
            plug_quat_w = normalize(cable.data.body_quat_w.torch[:, self._cable_grasp_body_id])
        else:
            plug_pos_w = ee_pos_w
            plug_quat_w = ee_quat_w

        # Socket pose in world (offset by the per-env origin).
        socket_pos_w = self.socket_pos_w
        env_origins = getattr(env.scene, "env_origins", None)
        if env_origins is not None:
            socket_pos_w = socket_pos_w + env_origins.to(device=self.device, dtype=socket_pos_w.dtype)
        insertion_dir_w = normalize(quat_apply(self.socket_quat_w, self._vec((0.0, 0.0, 1.0))))

        # Phase-entry snapshots (branch-free masked writes; no host sync).
        first_step = self.elapsed == 0.0
        self.phase_start_pos_w[first_step] = ee_pos_w[first_step]
        self.phase_start_quat_w[first_step] = ee_quat_w[first_step]
        self.phase_plug_pos_w[first_step] = plug_pos_w[first_step]
        self.phase_plug_quat_w[first_step] = plug_quat_w[first_step]
        connector_dir = quat_apply(plug_quat_w, self.connector_axis_local)
        tip_pos_w = plug_pos_w + connector_dir * self.connector_tip_len
        measured_tip_pos_w = tip_pos_w
        cable_tip_axis_w = connector_dir
        if cable is not None:
            try:
                cable_tip_quat_w = normalize(cable.data.body_quat_w.torch[:, self._cable_tip_body_id])
                cable_tip_axis_w = normalize(quat_apply(cable_tip_quat_w, self.connector_axis_local))
            except (AttributeError, IndexError):
                pass

        start_pos_w = self.phase_start_pos_w
        start_quat_w = self.phase_start_quat_w
        phase_plug_pos_w = self.phase_plug_pos_w
        phase_plug_quat_w = self.phase_plug_quat_w

        # EE orientation/position that aligns the gripper with the phase-entry plug pose for the pick.
        grasp_quat_w = normalize(quat_mul(phase_plug_quat_w, self.grasp_orientation_offset))
        grasp_pos_w = phase_plug_pos_w + quat_apply(phase_plug_quat_w, self.plug_grasp_offset)

        phase = self.phase
        target_pos_w = start_pos_w.clone()
        target_quat_w = start_quat_w.clone()
        t_grip = torch.zeros(self.num_envs, device=self.device)

        def set_target(mask, pos_w, quat_w, grip):
            target_pos_w[mask] = pos_w[mask]
            target_quat_w[mask] = quat_w[mask]
            t_grip[mask] = grip

        # --- Pick ---
        approach = phase == self.APPROACH
        set_target(approach, grasp_pos_w + quat_apply(phase_plug_quat_w, self.approach_offset), grasp_quat_w, 0.0)

        engage = phase == self.ENGAGE
        set_target(engage, grasp_pos_w + self.engage_offset, grasp_quat_w, 0.0)

        # GRASP: hold pose, close the gripper over the phase duration.
        grasp = phase == self.GRASP
        grasp_blend = _smoothstep(self.elapsed / torch.clamp(self.durations[self.GRASP], min=1.0e-6))
        target_pos_w[grasp] = start_pos_w[grasp]
        target_quat_w[grasp] = start_quat_w[grasp]
        t_grip[grasp] = grasp_blend[grasp]

        hold = phase == self.HOLD_GRASP
        t_grip[hold] = 1.0

        retract = phase == self.RETRACT
        set_target(retract, start_pos_w + quat_apply(phase_plug_quat_w, self.retract_vector), start_quat_w, 1.0)

        settle = phase == self.SETTLE
        t_grip[settle] = 1.0

        # --- Insert / extract ---
        # Targets are computed from the connector tip pose.  The gripper frame is offset behind the
        # tip, so aiming the EE itself at the socket mouth overshoots and drives the plug into the
        # fridge.  CARRY moves to the standoff; ALIGN/INSERT then use the measured cable-tip capsule
        # axis so the hose, not just the plug rigid body, becomes coaxial with the socket bore.
        ins_dir = insertion_dir_w  # bore axis into the socket = R(socket_quat) @ +Z
        socket_grasp_quat = normalize(quat_mul(self.socket_quat_w, self.grasp_orientation_offset))
        # Cable segment local +Z points back along the hose, so -Z should point into the socket.
        coax_delta_quat = _quat_from_two_vectors(cable_tip_axis_w, -ins_dir)
        coaxial_grasp_quat = normalize(quat_mul(coax_delta_quat, ee_quat_w))

        def ee_pos_for_tip(target_tip_pos_w, target_ee_quat_w):
            return target_tip_pos_w - quat_apply(target_ee_quat_w, self.connector_tip_pos_in_ee)

        preinsert_tip_pos = socket_pos_w - self.preinsert_standoff * ins_dir
        inserted_tip_pos = socket_pos_w + self.insert_final_depth * ins_dir
        extracted_tip_pos = socket_pos_w - self.extract_clearance * ins_dir
        approach_pos = ee_pos_for_tip(preinsert_tip_pos, socket_grasp_quat)
        coax_approach_pos = ee_pos_for_tip(preinsert_tip_pos, coaxial_grasp_quat)
        coax_inserted_pos = ee_pos_for_tip(inserted_tip_pos, coaxial_grasp_quat)
        coax_extracted_pos = ee_pos_for_tip(extracted_tip_pos, coaxial_grasp_quat)

        # CARRY: move up to the pre-insert standoff and rotate into socket alignment during the move.
        carry = phase == self.CARRY
        set_target(carry, approach_pos, socket_grasp_quat, 1.0)

        # ALIGN: hold 1 cm outside the mouth while correcting the measured cable axis onto the bore.
        align = phase == self.ALIGN
        set_target(align, coax_approach_pos, coaxial_grasp_quat, 1.0)

        # INSERT: push the connector tip forward along the bore axis to the shallow seated depth.
        insert = phase == self.INSERT
        set_target(insert, coax_inserted_pos, coaxial_grasp_quat, 1.0)

        # HOLD_INSERTED: dwell at the seated pose before releasing the first grasp.
        hold_ins = phase == self.HOLD_INSERTED
        set_target(hold_ins, coax_inserted_pos, coaxial_grasp_quat, 1.0)

        # RELEASE: open the fingers while holding the inserted pose; do not pull the cable yet.
        release = phase == self.RELEASE
        release_blend = _smoothstep(self.elapsed / torch.clamp(self.durations[self.RELEASE], min=1.0e-6))
        target_pos_w[release] = start_pos_w[release]
        target_quat_w[release] = start_quat_w[release]
        t_grip[release] = 1.0 - release_blend[release]

        # BACKOFF: with the gripper open, move sideways away from the socket/cable.
        withdraw_dir_w = quat_apply(self.socket_quat_w, self._vec((0.0, 1.0, 0.0)))
        backoff_pos = coax_inserted_pos + self.gripper_backoff_distance * withdraw_dir_w
        backoff = phase == self.BACKOFF
        set_target(backoff, backoff_pos, start_quat_w, 0.0)

        # REAPPROACH: return open fingers to the inserted cable head.
        reapproach = phase == self.REAPPROACH
        set_target(reapproach, coax_inserted_pos, coaxial_grasp_quat, 0.0)

        # REGRASP: close on the inserted cable head before extraction.
        regrasp = phase == self.REGRASP
        regrasp_blend = _smoothstep(self.elapsed / torch.clamp(self.durations[self.REGRASP], min=1.0e-6))
        target_pos_w[regrasp] = start_pos_w[regrasp]
        target_quat_w[regrasp] = start_quat_w[regrasp]
        t_grip[regrasp] = regrasp_blend[regrasp]

        # PULL_OUT: after the second grasp, remove the cable by pulling straight out of the socket.
        pull_out = phase == self.PULL_OUT
        set_target(pull_out, coax_extracted_pos, coaxial_grasp_quat, 1.0)

        # DONE: keep holding the cable at the pulled-out pose.
        done = phase == self.DONE
        t_grip[done] = 1.0

        # Smoothstep blend from the entry pose to the target pose (world frame).
        blend = _smoothstep(self.elapsed / self.durations[self.phase]).unsqueeze(-1)
        cmd_pos_w = start_pos_w * (1.0 - blend) + target_pos_w * blend
        cmd_quat_w = _blend_quat(start_quat_w, target_quat_w, blend)

        # Debug-only plug diagnostics; targets remain phase-entry smoothstep poses.
        plug_cos_val = torch.sum(connector_dir * ins_dir, dim=-1)
        tip_cos_val = torch.sum(cable_tip_axis_w * ins_dir, dim=-1)

        cmd_pos_b, cmd_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, cmd_pos_w, cmd_quat_w)
        self.command_pose[:, :3] = cmd_pos_b
        self.command_pose[:, 3:] = cmd_quat_b

        gripper = (_GRIPPER_OPEN + (_GRIPPER_CLOSED - _GRIPPER_OPEN) * t_grip).unsqueeze(-1)
        actions = torch.cat((self.command_pose, gripper), dim=-1)

        # --- Advance: min duration met AND converged (or hard 2x timeout). ---
        position_error = torch.abs(target_pos_w - ee_pos_w)
        rotation_error = quat_error_magnitude(target_quat_w, ee_quat_w)
        converged = torch.all(position_error < self.pos_tolerance, dim=-1) & (rotation_error < self.rot_tolerance)

        axial_depth = torch.sum((measured_tip_pos_w - socket_pos_w) * ins_dir, dim=-1)

        if self.debug:
            changed = self.phase != self.last_reported_phase
            if bool(changed[0].item()):
                name = self.PHASE_NAMES[int(self.phase[0].item())]
                print(
                    f"[waterhose_ik] {name}: "
                    f"pos_err={position_error[0].detach().cpu().tolist()} "
                    f"rot_err={float(rotation_error[0].detach().cpu()):.4f} "
                    f"plug_cos={float(plug_cos_val[0].detach().cpu()):+.2f} "
                    f"tip_cos={float(tip_cos_val[0].detach().cpu()):+.2f} "
                    f"depth_mm={float(axial_depth[0].detach().cpu()) * 1000.0:.1f} "
                    f"grip={float(gripper[0, 0].detach().cpu()):.2f}",
                    flush=True,
                )
            self.last_reported_phase[changed] = self.phase[changed]

        self.elapsed += self.step_dt
        timed_out = self.elapsed >= self.durations[self.phase]
        hard_timeout = self.elapsed >= 2.0 * self.durations[self.phase]
        should_advance = timed_out & (converged | hard_timeout) & (self.phase < self.DONE)

        self.phase[should_advance] += 1
        self.elapsed[should_advance] = 0.0

        return actions


def create_scripted_policy(env, *, settle_time: float = 4.0, debug: bool = False) -> WaterhoseDemoState:
    """Create the task-local scripted policy used by the demo launcher."""

    return WaterhoseDemoState(
        env.num_envs,
        env.step_dt,
        env.device,
        settle_time,
        debug,
    )
