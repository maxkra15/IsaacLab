# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted IK state machine for the RBY1 waterhose grasp-and-insert demo.

Phases: ``REST -> APPROACH -> ENGAGE -> GRASP -> HOLD_GRASP -> RETRACT -> SETTLE ->
CARRY -> ALIGN -> INSERT -> HOLD_INSERTED -> RELEASE -> BACKOFF -> DONE``.

Each phase has a fixed minimum *duration* and a single end-effector *target pose* derived
from a snapshot taken on phase entry plus a fixed geometric offset. The commanded pose is a
smoothstep blend from the entry pose to the target, and a phase advances once its minimum
duration has elapsed and the end effector has converged (or a hard timeout fires).

The output is the action vector the task's IK action term consumes. For the registered multi-body
Newton-IK tasks it is ``[right_ee pose(7), left_hold pose(7), torso_hold pose(7), gripper(1)]`` --
root-frame positions with ``(x, y, z, w)`` quaternions, where the two hold blocks pin the torso and
the idle left gripper. For an end-effector-only action variant it collapses to
``[right_ee pose(7), gripper(1)]``. :meth:`WaterhoseDemoState.compute` selects the layout from the
action manager's total action dimension.
"""

from __future__ import annotations

import torch
import warp as wp

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

from .geometry import (
    CONNECTOR_TIP_LEN,
    PLUG_GRASP_OFFSET,
    RIGHT_GRIPPER_EE_FRAME_POS,
    RIGHT_GRIPPER_EE_FRAME_QUAT_XYZW,
    SOCKET_MOUTH_POS,
    SOCKET_ROT_QUAT_XYZW,
)

# Gripper command convention used by the IK action term: +1 fully open, -1 fully closed.
_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSED = -1.0


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
    """Per-environment scripted grasp-and-insert state machine."""

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
    DONE = 13

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
        "DONE",
    )
    # Minimum time spent in each phase [s]. A phase advances once this has elapsed AND the end
    # effector has converged (or a 2x hard timeout fires). Insert/extract phases get generous
    # time and tolerance; DONE is terminal.
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
        # Terminal readout of the env-0 phase, printed once on every change.
        self._last_printed_phase = -1
        self._step_count = 0

        # Phase-entry snapshots (world frame) and the commanded pose (base frame).
        self.phase_start_pos_w = torch.zeros((self.num_envs, 3), device=device)
        self.phase_start_quat_w = torch.zeros((self.num_envs, 4), device=device)
        self.phase_start_quat_w[:, 3] = 1.0
        self.phase_plug_pos_w = torch.zeros((self.num_envs, 3), device=device)
        self.phase_plug_quat_w = torch.zeros((self.num_envs, 4), device=device)
        self.phase_plug_quat_w[:, 3] = 1.0
        self.command_pose = torch.zeros((self.num_envs, 7), device=device)
        self.command_pose[:, 6] = 1.0
        # Multi-body Newton-IK hold targets: [left_gripper_base pose(7), torso_hip_yaw pose(7)],
        # root frame, quaternions in (x, y, z, w) per the Newton IK action convention. Captured once
        # from the settled pose and held for the whole demo so the torso (and idle left arm) stay put
        # while the right arm tracks the connector. Consumed only when the active action exposes the
        # hold objectives; EE-only action variants ignore them.
        self.hold_poses = torch.zeros((self.num_envs, 14), device=device)
        self.hold_poses[:, 6] = 1.0
        self.hold_poses[:, 13] = 1.0
        self._holds_captured = False
        self._left_hold_body_id = None
        self._torso_hold_body_id = None

        durations = list(self.DURATIONS)
        durations[self.REST] = max(float(settle_time), self.step_dt)
        self.durations = torch.tensor(durations, dtype=torch.float32, device=device)

        # Convergence tolerances (generous; combined with the min duration this gives smooth motion).
        self.pos_tolerance = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32, device=device)
        self.rot_tolerance = 15.0 * torch.pi / 180.0
        # ALIGN also waits until the connector is actually coaxial with the bore (cos of the angle
        # between the connector axis and the bore axis), not just until the EE reaches its commanded
        # pose -- the gripped plug has compliance, so the connector axis lags the EE orientation.
        # 0.9995 ~= 1.8 deg; the hard timeout (2x ALIGN duration) bounds the dwell.
        self.coax_cos_tolerance = 0.9995

        # Fixed geometric offsets.
        self.plug_grasp_offset = self._vec(PLUG_GRASP_OFFSET)
        self.approach_offset = self._vec((0.0, 0.08, 0.0))
        self.engage_offset = self._vec((0.01, 0.0, 0.0))
        self.retract_vector = self._vec((0.0, 0.05, 0.0))
        self.connector_axis_local = self._vec((0.0, 0.0, 1.0))

        # Insertion geometry along the bore (= connector) axis, relative to the socket mouth.
        # The scripted target is expressed at the cable connector tip, not the end-effector frame:
        # CARRY stops with the tip just outside the mouth, ALIGN dwells there, and INSERT seats the
        # tip shallowly. The bore axis points mostly upward, so the standoff also keeps the gripper
        # low and clear of the socket mesh during the lift/align motion.
        self.preinsert_standoff = 0.018
        # Commanded connector-tip depth past the socket mouth: the forward-facing tip seats this far
        # into the bore (~3 mm). Kept shallow so the connector seats near the mouth instead of being
        # driven against the back of the bore, which buckles the hose and destabilizes the coupled
        # solve -- especially now that the lower bore friction lets the plug slide in freely instead
        # of being passively limited. Lower for a shallower seat, raise for a deeper one.
        self.insert_final_depth = 0.002
        self.gripper_backoff_distance = 0.10
        self.connector_tip_len = CONNECTOR_TIP_LEN

        # End-effector orientation that grasps the plug from the side: Rx(+90) * Rz(-90).
        z_axis = self._vec((0.0, 0.0, 1.0))
        x_axis = self._vec((1.0, 0.0, 0.0))
        q_rz = quat_from_angle_axis(torch.full((self.num_envs,), -torch.pi / 2.0, device=device), z_axis)
        q_rx = quat_from_angle_axis(torch.full((self.num_envs,), torch.pi / 2.0, device=device), x_axis)
        self.grasp_orientation_offset = normalize(quat_mul(q_rx, q_rz))
        self.connector_tip_pos_in_ee = quat_apply(
            quat_inv(self.grasp_orientation_offset),
            self.connector_axis_local * self.connector_tip_len - self.plug_grasp_offset,
        )
        # Connector-tip offset in the end-effector frame. Initialized to the static estimate and
        # overwritten per env at INSERT entry with the live grasp offset, so the insertion push
        # targets the true connector tip on the bore centerline (see compute()).
        self._tip_offset_frozen = self.connector_tip_pos_in_ee.clone()

        self.socket_pos_w = self._vec(SOCKET_MOUTH_POS)
        self.socket_quat_w = normalize(self._vec(SOCKET_ROT_QUAT_XYZW))

        self.ee_offset_pos = self._vec(RIGHT_GRIPPER_EE_FRAME_POS)
        self.ee_offset_quat = self._vec(RIGHT_GRIPPER_EE_FRAME_QUAT_XYZW)

        self._ee_body_id = None

        # Newton body ids of the per-env plug bodies, resolved lazily on first compute. The plug
        # is welded to the coupled cable, so its ground-truth pose is read from the Newton state
        # rather than the asset view.
        self._plug_body_ids = None

    def _vec(self, values) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)

    def _bind_plug_bodies(self) -> None:
        """Resolve the per-env plug Newton body ids for ground-truth pose reads."""

        from isaaclab_newton.physics.newton_manager import NewtonManager

        body_labels = [str(label) for label in NewtonManager.get_model().body_label]
        plug_ids = []
        for env_index in range(self.num_envs):
            token = f"env_{env_index}/Plug1"
            matches = [i for i, label in enumerate(body_labels) if label.endswith(token)]
            if len(matches) != 1:
                raise RuntimeError(f"Expected exactly one '{token}' body, found {len(matches)}.")
            plug_ids.append(matches[0])
        self._plug_body_ids = torch.as_tensor(plug_ids, device=self.device, dtype=torch.long)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.phase[env_ids] = self.REST
        self.elapsed[env_ids] = 0.0
        self.last_reported_phase[env_ids] = -1
        self._tip_offset_frozen[env_ids] = self.connector_tip_pos_in_ee[env_ids]
        self.phase_start_quat_w[env_ids] = 0.0
        self.phase_start_quat_w[env_ids, 3] = 1.0
        self.phase_plug_pos_w[env_ids] = 0.0
        self.phase_plug_quat_w[env_ids] = 0.0
        self.phase_plug_quat_w[env_ids, 3] = 1.0

    def compute(self, env) -> torch.Tensor:
        from isaaclab_newton.physics.newton_manager import NewtonManager

        # Terminal readout of the current phase (env 0), printed once on every change.
        self._step_count += 1
        current_phase = int(self.phase[0].item())
        if current_phase != self._last_printed_phase:
            print(
                f"[waterhose SM] step {self._step_count}: phase = {self.PHASE_NAMES[current_phase]}",
                flush=True,
            )
            self._last_printed_phase = current_phase

        robot = env.scene["robot"]

        if self._ee_body_id is None:
            self._ee_body_id = robot.find_bodies("right_gripper_base")[0][0]
        if self._plug_body_ids is None:
            self._bind_plug_bodies()

        root_pose_w = robot.data.root_link_pose_w.torch
        root_pos_w = root_pose_w[:, :3]
        root_quat_w = root_pose_w[:, 3:]

        ee_base_pos_w = robot.data.body_pos_w.torch[:, self._ee_body_id]
        ee_base_quat_w = robot.data.body_quat_w.torch[:, self._ee_body_id]
        ee_pos_w, ee_quat_w = combine_frame_transforms(
            ee_base_pos_w, ee_base_quat_w, self.ee_offset_pos, self.ee_offset_quat
        )

        # Live connector pose from the coupled-solver state (ground truth). The plug is welded to the
        # deformable cable and its RigidObject asset view is stale/frozen for a coupled body, so the
        # whole pick (approach/engage/grasp) reads the Newton body state to track where the hose
        # actually hangs. This keeps the pick agnostic to the cable's resting pose: changing the cable
        # stiffness (which lets it hang lower / at a different angle) needs no state-machine retuning.
        plug_state_pose = wp.to_torch(NewtonManager.get_state_0().body_q)[self._plug_body_ids]
        plug_pos_w = plug_state_pose[:, :3]
        plug_quat_w = normalize(plug_state_pose[:, 3:7])

        socket_pos_w = self.socket_pos_w + env.scene.env_origins.to(device=self.device, dtype=self.socket_pos_w.dtype)
        insertion_dir_w = normalize(quat_apply(self.socket_quat_w, self._vec((0.0, 0.0, 1.0))))

        # Phase-entry snapshots (branch-free masked writes; no host sync).
        first_step = self.elapsed == 0.0
        self.phase_start_pos_w[first_step] = ee_pos_w[first_step]
        self.phase_start_quat_w[first_step] = ee_quat_w[first_step]
        self.phase_plug_pos_w[first_step] = plug_pos_w[first_step]
        self.phase_plug_quat_w[first_step] = plug_quat_w[first_step]
        connector_dir = quat_apply(plug_quat_w, self.connector_axis_local)

        # Connector tip from the same live plug state: the plug origin advanced by the tip length
        # along the plug +Z axis. The coaxial-alignment axis points back along the hose (the plug is
        # welded to the cable head rotated ~180 deg), so it is the plug -Z, which matches the
        # ``-ins_dir`` convention used below.
        cable_tip_quat_w = normalize(plug_state_pose[:, 3:7])
        cable_tip_axis_w = normalize(quat_apply(cable_tip_quat_w, self._vec((0.0, 0.0, -1.0))))
        cable_tip_pos_w = plug_state_pose[:, :3] + quat_apply(
            cable_tip_quat_w, self._vec((0.0, 0.0, CONNECTOR_TIP_LEN))
        )

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

        # --- Insert ---
        # Targets are computed from the connector tip pose. The gripper frame is offset behind the
        # tip, so aiming the EE itself at the socket mouth would overshoot and drive the plug into
        # the fridge. CARRY moves to the standoff; ALIGN/INSERT use the cable-tip axis so the hose,
        # not just the plug rigid body, becomes coaxial with the socket bore.
        ins_dir = insertion_dir_w  # bore axis into the socket = R(socket_quat) @ +Z
        socket_grasp_quat = normalize(quat_mul(self.socket_quat_w, self.grasp_orientation_offset))
        # Cable segment local +Z points back along the hose, so -Z points into the socket.
        coax_delta_quat = _quat_from_two_vectors(cable_tip_axis_w, -ins_dir)
        coaxial_grasp_quat = normalize(quat_mul(coax_delta_quat, ee_quat_w))

        # Connector-tip offset in the EE frame for tip targeting. The static estimate assumes an
        # idealized grasp; the plug settles a few mm off it laterally, which is fatal for a 6 mm
        # connector in a 6.3 mm bore (it would ride the wall instead of entering coaxially). The
        # live offset, derived from the current plug/EE poses, lands the real tip on the bore axis:
        #   * CARRY/ALIGN: live offset to center the tip at the mouth (no bore contact yet).
        #   * INSERT entry: freeze the centered offset and hold it through INSERT/HOLD_INSERTED so
        #     the push commands a straight insertion instead of chasing bore-contact deflection.
        #   * Otherwise (REST..SETTLE, RELEASE..DONE): the static estimate.
        grasped_tip_offset_ee = quat_apply(quat_inv(ee_quat_w), cable_tip_pos_w - ee_pos_w)
        insert_entry = (phase == self.INSERT) & first_step
        if bool(insert_entry.any()):
            self._tip_offset_frozen[insert_entry] = grasped_tip_offset_ee[insert_entry]
        live_mask = (phase == self.CARRY) | (phase == self.ALIGN)
        frozen_mask = (phase == self.INSERT) | (phase == self.HOLD_INSERTED)
        tip_offset = self.connector_tip_pos_in_ee.clone()
        tip_offset = torch.where(live_mask.unsqueeze(-1), grasped_tip_offset_ee, tip_offset)
        tip_offset = torch.where(frozen_mask.unsqueeze(-1), self._tip_offset_frozen, tip_offset)

        def ee_pos_for_tip(target_tip_pos_w, target_ee_quat_w):
            return target_tip_pos_w - quat_apply(target_ee_quat_w, tip_offset)

        preinsert_tip_pos = socket_pos_w - self.preinsert_standoff * ins_dir
        inserted_tip_pos = socket_pos_w + self.insert_final_depth * ins_dir
        approach_pos = ee_pos_for_tip(preinsert_tip_pos, socket_grasp_quat)
        coax_approach_pos = ee_pos_for_tip(preinsert_tip_pos, coaxial_grasp_quat)
        coax_inserted_pos = ee_pos_for_tip(inserted_tip_pos, coaxial_grasp_quat)

        # CARRY: move up to the pre-insert standoff and rotate into socket alignment during the move.
        carry = phase == self.CARRY
        set_target(carry, approach_pos, socket_grasp_quat, 1.0)

        # ALIGN: hold just outside the mouth while correcting the cable axis onto the bore.
        align = phase == self.ALIGN
        set_target(align, coax_approach_pos, coaxial_grasp_quat, 1.0)

        # INSERT: push the connector tip forward along the bore axis to the shallow seated depth.
        insert = phase == self.INSERT
        set_target(insert, coax_inserted_pos, coaxial_grasp_quat, 1.0)

        # HOLD_INSERTED: dwell at the seated pose before releasing the grasp.
        hold_ins = phase == self.HOLD_INSERTED
        set_target(hold_ins, coax_inserted_pos, coaxial_grasp_quat, 1.0)

        # RELEASE: open the fingers while holding the inserted pose; do not pull the cable.
        release = phase == self.RELEASE
        release_blend = _smoothstep(self.elapsed / torch.clamp(self.durations[self.RELEASE], min=1.0e-6))
        target_pos_w[release] = start_pos_w[release]
        target_quat_w[release] = start_quat_w[release]
        t_grip[release] = 1.0 - release_blend[release]

        # BACKOFF: with the gripper open, move sideways away from the socket and cable.
        withdraw_dir_w = quat_apply(self.socket_quat_w, self._vec((0.0, 1.0, 0.0)))
        backoff_pos = coax_inserted_pos + self.gripper_backoff_distance * withdraw_dir_w
        backoff = phase == self.BACKOFF
        set_target(backoff, backoff_pos, start_quat_w, 0.0)

        # DONE: the gripper has backed away with the plug left inserted; keep the fingers open.
        done = phase == self.DONE
        t_grip[done] = 0.0

        # Smoothstep blend from the entry pose to the target pose (world frame).
        blend = _smoothstep(self.elapsed / self.durations[self.phase]).unsqueeze(-1)
        cmd_pos_w = start_pos_w * (1.0 - blend) + target_pos_w * blend
        cmd_quat_w = _blend_quat(start_quat_w, target_quat_w, blend)

        cmd_pos_b, cmd_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, cmd_pos_w, cmd_quat_w)
        self.command_pose[:, :3] = cmd_pos_b
        # Isaac Lab math and the Newton IK action both use (x, y, z, w).
        self.command_pose[:, 3:] = cmd_quat_b

        if not self._holds_captured:
            # Capture the multi-body hold targets (root frame, xyzw) once from the settled pose.
            if self._left_hold_body_id is None:
                self._left_hold_body_id = robot.find_bodies("left_gripper_base")[0][0]
                self._torso_hold_body_id = robot.find_bodies("torso_hip_yaw")[0][0]
            for slot, body_id in ((0, self._left_hold_body_id), (7, self._torso_hold_body_id)):
                hold_pos_b, hold_quat_b = subtract_frame_transforms(
                    root_pos_w,
                    root_quat_w,
                    robot.data.body_pos_w.torch[:, body_id],
                    robot.data.body_quat_w.torch[:, body_id],
                )
                self.hold_poses[:, slot : slot + 3] = hold_pos_b
                self.hold_poses[:, slot + 3 : slot + 7] = hold_quat_b
            self._holds_captured = True

        gripper = (_GRIPPER_OPEN + (_GRIPPER_CLOSED - _GRIPPER_OPEN) * t_grip).unsqueeze(-1)
        # Match the active action layout: the multi-body Newton IK action consumes
        # [ee pose(7), left hold(7), torso hold(7)]; EE-only variants consume just the EE pose.
        total_dim = env.action_manager.total_action_dim
        if total_dim == self.command_pose.shape[-1] + self.hold_poses.shape[-1] + 1:
            actions = torch.cat((self.command_pose, self.hold_poses, gripper), dim=-1)
        else:
            actions = torch.cat((self.command_pose, gripper), dim=-1)

        # --- Advance: min duration met AND converged (or hard 2x timeout). ---
        position_error = torch.abs(target_pos_w - ee_pos_w)
        rotation_error = quat_error_magnitude(target_quat_w, ee_quat_w)
        converged = torch.all(position_error < self.pos_tolerance, dim=-1) & (rotation_error < self.rot_tolerance)
        # ALIGN additionally requires the live connector axis to be coaxial with the bore, so INSERT
        # only begins once the gripped plug (not just the EE) is actually lined up. Other phases are
        # unaffected. The hard timeout still bounds the dwell if it cannot fully converge.
        coax_cos = torch.sum(connector_dir * ins_dir, dim=-1)
        align_phase = self.phase == self.ALIGN
        converged = converged & (~align_phase | (coax_cos > self.coax_cos_tolerance))

        if self.debug:
            plug_cos_val = torch.sum(connector_dir * ins_dir, dim=-1)
            tip_cos_val = torch.sum(cable_tip_axis_w * ins_dir, dim=-1)
            target_tip_pos_w = target_pos_w + quat_apply(target_quat_w, self.connector_tip_pos_in_ee)
            target_tip_depth = torch.sum((target_tip_pos_w - socket_pos_w) * ins_dir, dim=-1)
            changed = self.phase != self.last_reported_phase
            if bool(changed[0].item()):
                name = self.PHASE_NAMES[int(self.phase[0].item())]
                print(
                    f"[waterhose_ik] {name}: "
                    f"pos_err={position_error[0].detach().cpu().tolist()} "
                    f"rot_err={float(rotation_error[0].detach().cpu()):.4f} "
                    f"plug_cos={float(plug_cos_val[0].detach().cpu()):+.2f} "
                    f"tip_cos={float(tip_cos_val[0].detach().cpu()):+.2f} "
                    f"target_depth_mm={float(target_tip_depth[0].detach().cpu()) * 1000.0:.1f} "
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
