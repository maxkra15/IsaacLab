# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted MPM kinetic foundry: lift, align, pour, and return a grasped cup.

This is a scripted visual demonstration, not the PPO action contract. The registered
manager-based task retains its eight-dimensional arm-and-gripper policy action and adaptive reset curriculum.

.. code-block:: bash

    uv run --extra video python scripts/environments/state_machine/kinetic_foundry.py \
        --video --video_dir videos/kinetic_foundry --max_steps 1800 --seed 42 --viz newton_gl
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import VideoRecorderCfg
from isaaclab.envs import mdp as lab_mdp
from isaaclab.utils import math as math_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.franka_pour import mdp as pour_mdp
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

if TYPE_CHECKING:
    from isaaclab_tasks.contrib.franka_pour.pour_env import FrankaPourEnv
    from isaaclab_tasks.contrib.kinetic_foundry.foundry_env_cfg import KineticFoundryEnvCfg


_TASK_ID = "IsaacContrib-Kinetic-Foundry"
_TRANSFER_THRESHOLD = 0.70
_PHASE_TIMEOUT_S = (3.0, 8.0, 8.0, 24.0, 12.0, 8.0, 6.0, 3.0)


def _select_pre_pour_reset_row(env: FrankaPourEnv, env_ids: Sequence[int] | torch.Tensor | slice) -> dict[str, float]:
    """Select an upright grasp row with the receiver on the cup's pouring side.

    This replaces only the demo's reset curriculum. The trainable task keeps its adaptive
    replay sampler and original arm-and-gripper action space.
    """
    ids = (
        torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
        if isinstance(env_ids, slice)
        else torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()
    )
    if ids.numel() == 0:
        return {}
    states = env._reset_dataset_states
    source = states["source_root_pose"]
    target = states["target_root_pose"]
    delta = target[:, :3] - source[:, :3]
    quaternion = source[:, 3:7]
    up_alignment = 1.0 - 2.0 * (quaternion[:, 0].square() + quaternion[:, 1].square())
    # The validated dataset encodes category 1 as grasping and region 2 as transport.
    eligible = (
        (states["category"] == 1)
        & (states["reset_region"] == 2)
        & (source[:, 2] < 0.04)
        & (up_alignment > 0.995)
        & (delta[:, 1] < -0.14)
        & (delta[:, 1] > -0.30)
        & (delta[:, 0].abs() < 0.05)
    )
    candidate_rows = torch.nonzero(eligible, as_tuple=False).flatten()
    if candidate_rows.numel() == 0:
        raise RuntimeError("The validated reset dataset has no upright, source-full grasp row for the demo.")
    source_candidates = source[candidate_rows]
    target_candidates = target[candidate_rows]
    delta_candidates = delta[candidate_rows]
    score = (
        0.5 * (source_candidates[:, 0] - 0.5).abs()
        + source_candidates[:, 1].abs()
        + 2.0 * delta_candidates[:, 0].abs()
        + (target_candidates[:, 1] + 0.22).abs()
    )
    row = candidate_rows[torch.argmin(score)]
    env.reset_dataset_row_id[ids] = row
    env.pour_target_frac[ids] = float(env.cfg.pour_target_frac)
    return {"pre_pour_reset_row": float(row)}


def _record_particle_out_of_bounds(env: FrankaPourEnv) -> torch.Tensor:
    """Preserve the task's safety decision and record escaped-particle diagnostics."""
    failed = pour_mdp.particle_out_of_bounds(env)
    if bool(failed[0]):
        particles = env.particle_pos_e()[0]
        lower = env._particle_workspace_lower_t
        upper = env._particle_workspace_upper_t
        outside = ~((particles >= lower) & (particles <= upper)).all(dim=-1)
        minimum = tuple(float(value) for value in particles.amin(dim=0))
        maximum = tuple(float(value) for value in particles.amax(dim=0))
        env._foundry_oob_diagnostics = f"outside={int(outside.sum())}, min_xyz={minimum}, max_xyz={maximum}"
    return failed


class Phase(IntEnum):
    """Ordered physical stages of the pouring demonstration."""

    SETTLE = 0
    LIFT = 1
    ALIGN = 2
    POUR = 3
    RECOVER = 4
    RETURN = 5
    LOWER = 6
    RELEASE = 7
    DONE = 8
    FAILED = 9


class KineticFoundryStateMachine:
    """Move one pre-grasped cup using measured grasp, pose, and particle-transfer gates."""

    def __init__(self, env: FrankaPourEnv) -> None:
        """Capture the validated grasp transform and initial cup pose.

        Args:
            env: The initialized one-world Franka Pour environment.
        """
        if env.num_envs != 1:
            raise ValueError("The cinematic state machine requires exactly one environment.")
        self.step_dt = float(env.step_dt)
        self.phase = Phase.SETTLE
        self.step_count = 0
        self.phase_start_step = 0
        self.contact_loss_steps = 0
        self.failure_reason = ""
        self.home_pose = env.cup_pose_e().clone()
        self.commanded_cup_pose = self.home_pose.clone()
        tcp_pose = env.tcp_pose_e()
        self.cup_to_tcp_pos, self.cup_to_tcp_quat = math_utils.subtract_frame_transforms(
            self.home_pose[:, :3], self.home_pose[:, 3:7], tcp_pose[:, :3], tcp_pose[:, 3:7]
        )
        target_pose = env.target_pose_e()
        self.align_pose = self.home_pose.clone()
        self.align_pose[:, :3] = target_pose[:, :3] + self.home_pose.new_tensor((0.0, 0.10, 0.215))
        self.transport_pose = self.home_pose.clone()
        self.transport_pose[:, 2] = torch.maximum(self.home_pose[:, 2] + 0.14, target_pose[:, 2] + 0.20)
        self.tip_pose = self.align_pose.clone()
        angle = self.home_pose.new_full((1,), math.radians(140.0))
        self.tip_axis = self.home_pose.new_tensor(((1.0, 0.0, 0.0),))
        tip_rotation = math_utils.quat_from_angle_axis(angle, self.tip_axis)
        self.tip_pose[:, 3:7] = math_utils.quat_mul(tip_rotation, self.home_pose[:, 3:7])
        self.lower_pose = self.home_pose.clone()
        self.lower_pose[:, 2] += 0.012
        initial_fractions = pour_mdp.particle_fractions_obs(env)[0]
        self.initial_source_fraction = float(initial_fractions[0])
        self.initial_target_fraction = float(initial_fractions[1])
        if self.initial_source_fraction < 0.90 or self.initial_target_fraction > 0.05:
            raise RuntimeError(
                "The selected reset row is not a source-full pre-pour state: "
                f"source={self.initial_source_fraction:.3f}, target={self.initial_target_fraction:.3f}."
            )
        self.last_target_fraction = self.initial_target_fraction
        self.last_source_fraction = self.initial_source_fraction
        self.max_target_gain = 0.0
        self.max_spill_fraction = 0.0
        self.last_progress_gain = 0.0
        self.last_progress_step = 0
        self.pour_rocking = False
        self.completed_phases: list[str] = []

    def _transition(self, next_phase: Phase, target_gain: float, spill_fraction: float) -> None:
        previous = self.phase
        self.completed_phases.append(previous.name.lower())
        self.phase = next_phase
        self.phase_start_step = self.step_count
        if next_phase == Phase.POUR:
            self.last_progress_gain = target_gain
            self.last_progress_step = self.step_count
        if previous == Phase.POUR:
            self.pour_rocking = False
        print(
            f"step={self.step_count:04d} {previous.name.lower()} -> {next_phase.name.lower()} "
            f"target_gain={target_gain:.3f} spill={spill_fraction:.3f}",
            flush=True,
        )

    def _fail(self, reason: str) -> None:
        self.failure_reason = reason
        self.phase = Phase.FAILED
        print(f"step={self.step_count:04d} failed: {reason}", flush=True)

    def _goal_pose(self) -> torch.Tensor:
        if self.phase in (Phase.SETTLE, Phase.RELEASE, Phase.DONE, Phase.FAILED):
            return self.home_pose if self.phase == Phase.SETTLE else self.commanded_cup_pose
        if self.phase == Phase.LIFT:
            return self.transport_pose
        if self.phase in (Phase.ALIGN, Phase.RECOVER):
            return self.align_pose
        if self.phase == Phase.POUR:
            return self.tip_pose
        if self.phase == Phase.RETURN:
            return self.transport_pose
        return self.lower_pose

    def _update_pour_goal(self, target_gain: float, source_fraction: float) -> None:
        """Rock a stalled, still-filled cup by at most 12 degrees over the receiver."""
        if target_gain >= self.last_progress_gain + 0.01:
            self.last_progress_gain = target_gain
            self.last_progress_step = self.step_count
        phase_age_s = (self.step_count - self.phase_start_step) * self.step_dt
        stall_age_s = (self.step_count - self.last_progress_step) * self.step_dt
        rocking = phase_age_s > 3.0 and stall_age_s > 1.5 and source_fraction > 0.02
        if rocking and not self.pour_rocking:
            print(
                f"step={self.step_count:04d} pour flow stalled; rocking cup gently "
                f"(source={source_fraction:.3f}, target_gain={target_gain:.3f})",
                flush=True,
            )
        self.pour_rocking = rocking
        rocking_angle = 12.0 * math.sin(2.0 * math.pi * (stall_age_s - 1.5) / 3.0) if rocking else 0.0
        angle = self.home_pose.new_full((1,), math.radians(140.0 + rocking_angle))
        rotation = math_utils.quat_from_angle_axis(angle, self.tip_axis)
        self.tip_pose[:, 3:7] = math_utils.quat_mul(rotation, self.home_pose[:, 3:7])

    def _advance_command(self, goal_pose: torch.Tensor) -> None:
        """Rate-limit the IK target so cup inertia never receives a pose jump."""
        max_position_step = 0.12 * self.step_dt
        delta = goal_pose[:, :3] - self.commanded_cup_pose[:, :3]
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        self.commanded_cup_pose[:, :3] += delta * torch.clamp(max_position_step / distance.clamp_min(1.0e-6), max=1.0)

        # Slower back-rotation lets residual grains drain into the receiver instead of
        # accelerating them out of the bounded particle workspace.
        max_angle_step = (0.4 if self.phase == Phase.RECOVER else 1.0) * self.step_dt
        current = self.commanded_cup_pose[:, 3:7]
        destination = goal_pose[:, 3:7]
        destination = torch.where((current * destination).sum(dim=-1, keepdim=True) < 0, -destination, destination)
        angle = math_utils.quat_error_magnitude(current, destination).unsqueeze(-1)
        blend = torch.clamp(max_angle_step / angle.clamp_min(1.0e-6), max=1.0)
        self.commanded_cup_pose[:, 3:7] = torch.nn.functional.normalize(
            current * (1.0 - blend) + destination * blend, dim=-1
        )

    def action(self, env: FrankaPourEnv) -> torch.Tensor:
        """Evaluate phase gates and return an absolute TCP pose plus gripper command."""
        cup_pose = env.cup_pose_e()
        fractions = pour_mdp.particle_fractions_obs(env)[0]
        source_fraction, target_fraction, spill_fraction = (float(value) for value in fractions)
        target_gain = target_fraction - self.initial_target_fraction
        self.last_source_fraction = source_fraction
        self.last_target_fraction = target_fraction
        self.max_target_gain = max(self.max_target_gain, target_gain)
        self.max_spill_fraction = max(self.max_spill_fraction, spill_fraction)
        gripper = env.action_manager.get_term("gripper_action")
        contact = bool(gripper.bilateral_contact[0])
        cup_speed = float(torch.linalg.vector_norm(env.scene["source_cup"].data.root_link_vel_w.torch[0, :3]))
        position_error = float(torch.linalg.vector_norm(cup_pose[0, :3] - self._goal_pose()[0, :3]))
        upright_error = float(math_utils.quat_error_magnitude(cup_pose[:, 3:7], self.home_pose[:, 3:7])[0])
        age_s = (self.step_count - self.phase_start_step) * self.step_dt

        if self.phase == Phase.SETTLE and contact and cup_speed < 0.06 and age_s >= 0.5:
            self._transition(Phase.LIFT, target_gain, spill_fraction)
        elif self.phase == Phase.LIFT and contact and cup_pose[0, 2] >= self.home_pose[0, 2] + 0.095:
            self._transition(Phase.ALIGN, target_gain, spill_fraction)
        elif self.phase == Phase.ALIGN and contact and position_error < 0.035 and upright_error < 0.18:
            self._transition(Phase.POUR, target_gain, spill_fraction)
        elif self.phase == Phase.POUR and target_gain >= _TRANSFER_THRESHOLD and spill_fraction <= 0.10:
            self._transition(Phase.RECOVER, target_gain, spill_fraction)
        elif self.phase == Phase.RECOVER and contact and upright_error < 0.18 and position_error < 0.04:
            self._transition(Phase.RETURN, target_gain, spill_fraction)
        elif self.phase == Phase.RETURN and contact and position_error < 0.04:
            self._transition(Phase.LOWER, target_gain, spill_fraction)
        elif self.phase == Phase.LOWER and contact and position_error < 0.025 and upright_error < 0.18:
            self._transition(Phase.RELEASE, target_gain, spill_fraction)
        elif (
            self.phase == Phase.RELEASE
            and age_s > 0.5
            and float(env.gripper_width()[0]) > 0.055
            and target_gain >= _TRANSFER_THRESHOLD
            and spill_fraction <= 0.10
        ):
            self._transition(Phase.DONE, target_gain, spill_fraction)

        if self.phase in (Phase.LIFT, Phase.ALIGN, Phase.POUR, Phase.RECOVER, Phase.RETURN, Phase.LOWER):
            self.contact_loss_steps = 0 if contact else self.contact_loss_steps + 1
            if self.contact_loss_steps * self.step_dt > 0.5:
                self._fail("bilateral cup contact lost for more than 0.5 s")
        else:
            self.contact_loss_steps = 0

        if self.phase.value < Phase.DONE:
            phase_timeout_s = _PHASE_TIMEOUT_S[int(self.phase)]
            if (self.step_count - self.phase_start_step) * self.step_dt > phase_timeout_s:
                self._fail(f"{self.phase.name.lower()} gate did not pass within {phase_timeout_s:.1f} s")

        if self.phase == Phase.POUR:
            self._update_pour_goal(target_gain, source_fraction)
        self._advance_command(self._goal_pose())
        tcp_pos, tcp_quat = math_utils.combine_frame_transforms(
            self.commanded_cup_pose[:, :3],
            self.commanded_cup_pose[:, 3:7],
            self.cup_to_tcp_pos,
            self.cup_to_tcp_quat,
        )
        gripper_command = 1.0 if self.phase in (Phase.RELEASE, Phase.DONE) else -1.0
        action = torch.cat((tcp_pos, tcp_quat, cup_pose.new_full((1, 1), gripper_command)), dim=-1)
        self.step_count += 1
        return action


def _make_demo_cfg(
    max_steps: int, video: bool, video_dir: Path, video_source: str, seed: int = 42
) -> KineticFoundryEnvCfg:
    """Keep the trainable task intact and configure only this scripted playback."""
    env_cfg, _ = resolve_task_config(_TASK_ID, "")
    env_cfg.seed = seed
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum_freeze = True
    env_cfg.curriculum.reset_dataset.func = _select_pre_pour_reset_row
    env_cfg.episode_length_s = max_steps * env_cfg.sim.dt * env_cfg.decimation + 1.0
    env_cfg.terminations.success = None
    env_cfg.terminations.particle_out_of_bounds.func = _record_particle_out_of_bounds
    env_cfg.actions.arm_action = lab_mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.6},
        ),
        body_offset=lab_mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
    )
    if video:
        env_cfg.video_recorders = [
            VideoRecorderCfg(
                source=video_source,
                output_dir=str(video_dir),
                output_filename_prefix="kinetic_foundry",
                video_length=max_steps,
                frame_stride=2 if max_steps > 1 else 1,
            )
        ]
    return env_cfg


def main() -> None:
    """Launch the one-world foundry demo and report physical outcome metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_steps", type=int, default=1800, help="Maximum environment steps to simulate.")
    parser.add_argument("--seed", type=int, default=42, help="Best-effort playback seed (MPM is not deterministic).")
    parser.add_argument("--video", action="store_true", help="Record MP4 clips from the Newton visualizer.")
    parser.add_argument("--video_dir", type=Path, default=Path("videos/kinetic_foundry"), help="MP4 output directory.")
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["newton_gl"])
    args_cli, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra_args]
    if args_cli.max_steps <= 0:
        parser.error("--max_steps must be positive")
    requested_visualizers = args_cli.visualizer or []
    if args_cli.video and "newton_rtx" in requested_visualizers:
        video_source = "visualizer:newton_rtx"
    elif args_cli.video and "newton_gl" in requested_visualizers:
        video_source = "visualizer:newton"
    elif args_cli.video:
        parser.error("--video requires --viz newton_gl or --viz newton_rtx")
    else:
        video_source = "visualizer:newton"

    env_cfg = _make_demo_cfg(args_cli.max_steps, args_cli.video, args_cli.video_dir, video_source, args_cli.seed)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(_TASK_ID, cfg=env_cfg)
        try:
            with torch.inference_mode():
                env.reset()
                controller = KineticFoundryStateMachine(env.unwrapped)
                print(
                    f"Kinetic foundry: reset row {int(env.unwrapped.reset_dataset_row_id[0])}; "
                    f"source={controller.initial_source_fraction:.3f}, "
                    f"target={controller.initial_target_fraction:.3f}; "
                    f"one world, {args_cli.max_steps} step limit.",
                    flush=True,
                )
                for step in range(args_cli.max_steps):
                    action = controller.action(env.unwrapped)
                    if controller.phase in (Phase.DONE, Phase.FAILED):
                        break
                    _, _, terminated, truncated, _ = env.step(action)
                    if bool((terminated | truncated)[0]):
                        terms = env.unwrapped.termination_manager
                        activated = [name for name in terms.active_terms if bool(terms.get_term(name)[0])]
                        diagnosis = getattr(env.unwrapped, "_foundry_oob_diagnostics", "")
                        detail = f"; {diagnosis}" if diagnosis else ""
                        controller._fail(f"environment termination: {', '.join(activated) or 'unknown term'}{detail}")
                        break
                    if step % 90 == 0:
                        print(
                            f"step={step:04d} phase={controller.phase.name.lower()} "
                            f"target_gain_max={controller.max_target_gain:.3f} "
                            f"source={controller.last_source_fraction:.3f} "
                            f"spill_max={controller.max_spill_fraction:.3f}",
                            flush=True,
                        )
                if controller.phase not in (Phase.DONE, Phase.FAILED):
                    controller._fail(f"max_steps={args_cli.max_steps} reached before completion")
                successful = controller.phase == Phase.DONE
                print(
                    f"result={'success' if successful else 'incomplete'} "
                    f"phase={controller.phase.name.lower()} "
                    f"completed={','.join(controller.completed_phases)} "
                    f"target_fraction_initial={controller.initial_target_fraction:.3f} "
                    f"target_fraction_final={controller.last_target_fraction:.3f} "
                    f"target_gain_max={controller.max_target_gain:.3f} "
                    f"source_fraction_final={controller.last_source_fraction:.3f} "
                    f"spill_fraction_max={controller.max_spill_fraction:.3f} "
                    f"steps={controller.step_count} "
                    f"reason={controller.failure_reason or 'none'}",
                    flush=True,
                )
                if args_cli.video:
                    print(f"video_dir={args_cli.video_dir}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
