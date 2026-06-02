# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted IK state machine for the RBY1 waterhose scene-config tasks.

This is a faithful port of the time-scheduled state machine used by the working
Newton ``cable_robot`` insert/extract success demo
(``newton/examples/cable_robot/example_waterhose_scene2_insert_extract_success.py``).

Design (deliberately simple and robust):

* Each phase has a fixed *duration* and computes a single EE *target pose* from a
  snapshot taken on phase entry (the EE pose and the plug pose at that instant)
  plus a fixed geometric offset.
* The commanded pose is a smoothstep blend from the entry pose to the target
  pose, with the rotation interpolated by shortest-path normalized lerp.
* The gripper is a simple open->closed lerp (no force feedback).
* A phase advances once it has run for at least its duration *and* the EE has
  converged to the target (with a hard 2x-duration timeout so a phase can never
  stall the demo).

There is intentionally no force-feedback grip, finger centering, axis-alignment
search, or insertion integral control here -- those made the previous version
brittle. The plug is a rigid body welded to the cable head, so a timed grasp
holds it just like the reference demo.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import (
    combine_frame_transforms,
    normalize,
    quat_apply,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_mul,
    subtract_frame_transforms,
)

# Grasp contact frame expressed in the right_gripper_base local frame.
_RIGHT_EE_FROM_BASE_POS = (0.0, 0.0, -0.1055)
_RIGHT_EE_FROM_BASE_QUAT = (0.70710677, 0.70710677, 0.0, 0.0)

# Grasp point relative to the plug frame: side grasp, shifted slightly toward the
# cable body so both fingers wrap the plug symmetrically.
_CABLE_RADIUS = 0.003
_GRASP_SHIFT = 0.01
_PLUG_GRASP_OFFSET = (0.0, -_CABLE_RADIUS + 0.002, _GRASP_SHIFT)

# Gripper joint command convention used by the IK action term: +1 fully open,
# -1 fully closed.
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


class WaterhoseDemoState:
    """Per-environment scripted pick-and-insert state machine."""

    REST = 0
    APPROACH = 1
    ENGAGE = 2
    GRASP = 3
    HOLD_GRASP = 4
    RETRACT = 5
    SETTLE = 6
    APPROACH_TARGET = 7
    INSERT = 8
    RELEASE = 9
    WITHDRAW = 10
    DONE = 11

    PHASE_NAMES = (
        "REST",
        "APPROACH",
        "ENGAGE",
        "GRASP",
        "HOLD_GRASP",
        "RETRACT",
        "SETTLE",
        "APPROACH_TARGET",
        "INSERT",
        "RELEASE",
        "WITHDRAW",
        "DONE",
    )
    # Minimum time spent in each phase (seconds); a phase advances once this has
    # elapsed and the EE has converged (or a 2x hard timeout is reached).
    DURATIONS = (0.25, 3.0, 1.5, 0.5, 0.5, 1.5, 0.3, 5.0, 5.0, 1.0, 2.0, 1.0e6)

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
        self.phase_start_quat_w[:, 0] = 1.0
        self.phase_plug_pos_w = torch.zeros((self.num_envs, 3), device=device)
        self.phase_plug_quat_w = torch.zeros((self.num_envs, 4), device=device)
        self.phase_plug_quat_w[:, 0] = 1.0
        self.command_pose = torch.zeros((self.num_envs, 7), device=device)
        self.command_pose[:, 3] = 1.0

        durations = list(self.DURATIONS)
        durations[self.REST] = max(float(settle_time), self.step_dt)
        self.durations = torch.tensor(durations, dtype=torch.float32, device=device)

        # Convergence tolerances (generous; combined with the min duration this
        # gives smooth motion without ever stalling).
        self.pos_tolerance = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32, device=device)
        self.rot_tolerance = 15.0 * torch.pi / 180.0

        # Fixed geometric offsets (all match the reference success demo).
        self.plug_grasp_offset = self._vec(_PLUG_GRASP_OFFSET)
        self.approach_offset = self._vec((0.0, 0.08, 0.0))
        self.engage_offset = self._vec((0.01, 0.0, 0.0))
        self.retract_vector = self._vec((0.0, 0.05, 0.0))
        self.withdraw_offset = self._vec((-0.10, 0.0, 0.0))

        # Insertion geometry: stand off this far from the socket on APPROACH_TARGET,
        # then push the EE this far along the socket axis during INSERT.
        self.insertion_start_depth = 0.005
        self.insert_travel = 0.03

        # EE orientation that grasps the plug from the side: Rx(+90) * Rz(-90).
        z_axis = self._vec((0.0, 0.0, 1.0))
        x_axis = self._vec((1.0, 0.0, 0.0))
        q_rz = quat_from_angle_axis(torch.full((self.num_envs,), -torch.pi / 2.0, device=device), z_axis)
        q_rx = quat_from_angle_axis(torch.full((self.num_envs,), torch.pi / 2.0, device=device), x_axis)
        self.grasp_orientation_offset = normalize(quat_mul(q_rx, q_rz))

        # Socket pose (env-local; env_origins added at runtime).
        self.socket_pos_w = self._vec((-0.259404, 0.362961, 0.5 - 0.262711))
        socket_angle = torch.full((self.num_envs,), 0.3490658503988659, device=device)
        self.socket_quat_w = quat_from_angle_axis(socket_angle, self._vec((1.0, 0.0, 0.0)))

        self.ee_offset_pos = self._vec(_RIGHT_EE_FROM_BASE_POS)
        self.ee_offset_quat = self._vec(_RIGHT_EE_FROM_BASE_QUAT)

        self._ee_body_id = None

    def _vec(self, values) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.phase[env_ids] = self.REST
        self.elapsed[env_ids] = 0.0
        self.last_reported_phase[env_ids] = -1
        self.phase_start_quat_w[env_ids] = 0.0
        self.phase_start_quat_w[env_ids, 0] = 1.0
        self.phase_plug_pos_w[env_ids] = 0.0
        self.phase_plug_quat_w[env_ids] = 0.0
        self.phase_plug_quat_w[env_ids, 0] = 1.0

    def compute(self, env) -> torch.Tensor:
        robot = env.scene["robot"]
        try:
            plug = env.scene["plug1"]
        except KeyError:
            plug = None

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

        start_pos_w = self.phase_start_pos_w
        start_quat_w = self.phase_start_quat_w
        snap_plug_pos_w = self.phase_plug_pos_w
        snap_plug_quat_w = self.phase_plug_quat_w

        # EE orientation that aligns the gripper with the (snapshotted) plug, and
        # with the socket for the insertion phases.
        grasp_quat_w = normalize(quat_mul(snap_plug_quat_w, self.grasp_orientation_offset))
        socket_grasp_quat_w = normalize(quat_mul(self.socket_quat_w, self.grasp_orientation_offset))
        grasp_pos_w = snap_plug_pos_w + quat_apply(snap_plug_quat_w, self.plug_grasp_offset)

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
        set_target(
            approach,
            grasp_pos_w + quat_apply(snap_plug_quat_w, self.approach_offset),
            grasp_quat_w,
            0.0,
        )

        engage = phase == self.ENGAGE
        set_target(engage, grasp_pos_w + self.engage_offset, grasp_quat_w, 0.0)

        # GRASP: hold pose, close the gripper over the phase duration.
        grasp = phase == self.GRASP
        grasp_blend = _smoothstep(self.elapsed / torch.clamp(self.durations[self.GRASP], min=1.0e-6))
        target_pos_w[grasp] = start_pos_w[grasp]
        target_quat_w[grasp] = start_quat_w[grasp]
        t_grip[grasp] = grasp_blend[grasp]

        hold = phase == self.HOLD_GRASP
        t_grip[hold] = 1.0  # pose already starts at the hold pose

        retract = phase == self.RETRACT
        set_target(
            retract,
            start_pos_w + quat_apply(snap_plug_quat_w, self.retract_vector),
            start_quat_w,
            1.0,
        )

        settle = phase == self.SETTLE
        t_grip[settle] = 1.0

        # --- Insert ---
        approach_target = phase == self.APPROACH_TARGET
        set_target(
            approach_target,
            socket_pos_w + self.insertion_start_depth * insertion_dir_w,
            socket_grasp_quat_w,
            1.0,
        )

        insert = phase == self.INSERT
        set_target(insert, start_pos_w + self.insert_travel * insertion_dir_w, start_quat_w, 1.0)

        # RELEASE: hold pose, open the gripper over the phase duration.
        release = phase == self.RELEASE
        release_blend = _smoothstep(self.elapsed / torch.clamp(self.durations[self.RELEASE], min=1.0e-6))
        t_grip[release] = 1.0 - release_blend[release]

        withdraw = phase == self.WITHDRAW
        set_target(withdraw, start_pos_w + self.withdraw_offset, start_quat_w, 0.0)

        # Smoothstep blend from the entry pose to the target pose (world frame).
        blend = _smoothstep(self.elapsed / self.durations[self.phase]).unsqueeze(-1)
        cmd_pos_w = start_pos_w * (1.0 - blend) + target_pos_w * blend
        cmd_quat_w = _blend_quat(start_quat_w, target_quat_w, blend)

        cmd_pos_b, cmd_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, cmd_pos_w, cmd_quat_w)
        self.command_pose[:, :3] = cmd_pos_b
        self.command_pose[:, 3:] = cmd_quat_b

        gripper = (_GRIPPER_OPEN + (_GRIPPER_CLOSED - _GRIPPER_OPEN) * t_grip).unsqueeze(-1)
        actions = torch.cat((self.command_pose, gripper), dim=-1)

        # --- Advance: min duration met AND converged (or hard 2x timeout). ---
        position_error = torch.abs(target_pos_w - ee_pos_w)
        rotation_error = quat_error_magnitude(target_quat_w, ee_quat_w)
        converged = torch.all(position_error < self.pos_tolerance, dim=-1) & (rotation_error < self.rot_tolerance)

        if self.debug:
            changed = self.phase != self.last_reported_phase
            if bool(changed[0].item()):
                name = self.PHASE_NAMES[int(self.phase[0].item())]
                print(
                    f"[waterhose_ik] {name}: "
                    f"pos_err={position_error[0].detach().cpu().tolist()} "
                    f"rot_err={float(rotation_error[0].detach().cpu()):.4f} "
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
