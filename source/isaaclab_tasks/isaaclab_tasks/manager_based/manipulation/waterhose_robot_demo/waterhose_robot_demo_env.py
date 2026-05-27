# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-style environment wrapping Newton's waterhose robot success demo."""

from __future__ import annotations

import importlib
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv

from .waterhose_robot_demo_env_cfg import WaterhoseRobotDemoEnvCfg


class WaterhoseRobotDemoEnv(ManagerBasedRLEnv):
    """Thin IsaacLab manager wrapper around Newton's reference waterhose success demo.

    This task intentionally keeps the reference demo's simulation architecture
    intact: MuJoCo/Newton robot model, separate VBD cable/fridge model, duplicated
    proxy gripper bodies, hand-written two-way coupling kernels, and the original
    scripted state machine. IsaacLab provides task registration, manager lifecycle,
    observations, and run-script integration.
    """

    cfg: WaterhoseRobotDemoEnvCfg

    def __init__(self, cfg: WaterhoseRobotDemoEnvCfg, render_mode: str | None = None, **kwargs):
        self.reference_demo = None
        self.reference_viewer = None
        self._reference_module = None
        self._initializing_manager = True
        self._build_reference_demo(cfg)
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self._initializing_manager = False
        self.obs_buf = self.observation_manager.compute(update_history=True)

    def _ensure_reference_on_path(self, cfg: WaterhoseRobotDemoEnvCfg) -> None:
        root = Path(cfg.reference_newton_root).expanduser().resolve()
        if not (root / "newton").is_dir():
            raise FileNotFoundError(f"reference_newton_root does not contain a newton package: {root}")
        root_s = str(root)
        if root_s not in sys.path:
            sys.path.insert(0, root_s)

    def _make_reference_args(self, cfg: WaterhoseRobotDemoEnvCfg) -> Namespace:
        return Namespace(
            device=cfg.reference_device,
            viewer=cfg.reference_viewer,
            rerun_address=None,
            output_path="waterhose_robot_demo_output.usd",
            num_frames=int(cfg.reference_num_frames),
            headless=bool(cfg.reference_headless),
            test=False,
            quiet=bool(cfg.reference_quiet),
            benchmark=False,
            warp_config=[],
            realtime=False,
            primary_view=str(cfg.reference_primary_view),
            no_twoway=False,
            print_cable_poses=False,
            cable_pose_settle_seconds=None,
            print_robot_poses=False,
            broad_phase="explicit",
        )

    def _make_reference_viewer(self, cfg: WaterhoseRobotDemoEnvCfg):
        self._ensure_reference_on_path(cfg)
        import newton.viewer  # noqa: PLC0415

        viewer_type = str(cfg.reference_viewer).lower()
        if viewer_type == "gl":
            return newton.viewer.ViewerGL(headless=bool(cfg.reference_headless))
        if viewer_type == "null":
            return newton.viewer.ViewerNull(num_frames=int(cfg.reference_num_frames))
        if viewer_type == "usd":
            return newton.viewer.ViewerUSD(
                output_path="waterhose_robot_demo_output.usd", num_frames=int(cfg.reference_num_frames)
            )
        raise ValueError(f"Unsupported reference_viewer={cfg.reference_viewer!r}; expected 'gl', 'null', or 'usd'.")

    def _build_reference_demo(self, cfg: WaterhoseRobotDemoEnvCfg) -> None:
        self._ensure_reference_on_path(cfg)
        module = importlib.import_module(cfg.reference_module)
        self._reference_module = module
        self.reference_viewer = self._make_reference_viewer(cfg)
        self.reference_demo = module.Example(self.reference_viewer, self._make_reference_args(cfg))

    def apply_teleop_command(self, command: torch.Tensor) -> None:
        """Apply a 7D SpaceMouse command to the reference demo's manual IK targets.

        The command layout matches IsaacLab SE(3) teleop devices:
        ``[dx, dy, dz, rx, ry, rz, gripper]``. Translation and rotation are
        already scaled by the device; the gripper term is ``+1`` for open and
        ``-1`` for closed.
        """
        ex = self.reference_demo
        if ex is None:
            return
        if getattr(ex, "auto_mode", True):
            ex.auto_mode = False
            ex._stop_auto_mode()

        import warp as wp  # noqa: PLC0415

        cmd = command.detach().to("cpu", dtype=torch.float32).numpy().reshape(-1)
        if cmd.shape[0] < 6:
            return

        tf = ex.ee_tfs[0]
        pos = wp.transform_get_translation(tf)
        quat = wp.transform_get_rotation(tf)
        dp = cmd[:3]
        pos = pos + wp.vec3(float(dp[0]), float(dp[1]), float(dp[2]))

        rotvec = cmd[3:6]
        angle = float(np.linalg.norm(rotvec))
        if angle > 1.0e-8:
            axis = rotvec / angle
            dq = wp.quat_from_axis_angle(wp.vec3(float(axis[0]), float(axis[1]), float(axis[2])), angle)
            quat = wp.normalize(dq * quat)
        ex.ee_tfs[0] = wp.transform(pos, quat)

        if cmd.shape[0] >= 7:
            gripper_value = float(ex.sm_gripper_open_value if cmd[6] > 0.0 else ex.sm_gripper_closed_value)
            gripper_np = ex.gripper_targets.numpy()
            gripper_np[0] = gripper_value
            wp.copy(ex.gripper_targets, wp.array(gripper_np, dtype=wp.float32))
            ex.gripper_targets_list[0] = gripper_value
            ex._sync_gripper_followers()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if getattr(self, "_initializing_manager", False):
            return
        # The reference demo does not expose partial reset; rebuild for full-env resets.
        if isinstance(env_ids, slice) or len(env_ids) >= self.num_envs:
            self._close_reference()
            self._build_reference_demo(self.cfg)

    def _close_reference(self) -> None:
        viewer = getattr(self, "reference_viewer", None)
        if viewer is not None and hasattr(viewer, "close"):
            viewer.close()
        self.reference_demo = None
        self.reference_viewer = None

    def step(self, action: torch.Tensor):
        """Step the reference Newton demo once and run manager bookkeeping."""
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()
        self.action_manager.apply_action()

        self.reference_demo.step()
        if self.sim.is_rendering or str(self.cfg.reference_viewer).lower() in {"gl", "usd"}:
            self.reference_demo.render()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        self.recorder_manager.record_post_step()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed(seed)
        env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.recorder_manager.record_pre_reset(env_ids)
        self._reset_idx(env_ids)
        self.recorder_manager.record_post_reset(env_ids)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.extras

    def close(self) -> None:
        self._close_reference()
        super().close()

