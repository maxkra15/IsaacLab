# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Two-arm state machine for a physically pinned VBD curtain.

.. code-block:: bash

    uv run python scripts/environments/state_machine/textile_atelier.py --max_steps 800
    uv run python scripts/environments/state_machine/textile_atelier.py --passive_baseline --max_steps 800
    uv run --extra video python scripts/environments/state_machine/textile_atelier.py --video --max_steps 800

Only arm IK targets are commanded. The state machine measures hand proximity and
the motion of fixed material patches; it never writes cloth positions after reset.
"""

from __future__ import annotations

import argparse
import sys
from enum import IntEnum

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.utils.math import quat_apply_inverse

import isaaclab_tasks.contrib.textile_atelier  # noqa: F401
from isaaclab_tasks.contrib.textile_atelier.mdp import curtain_deflection_patch_masks
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

TASK_NAME = "IsaacContrib-Textile-Atelier-Kuka-GR1T2"

parser = argparse.ArgumentParser(description="Run the two-arm VBD curtain state machine.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel atelier scenes.")
parser.add_argument("--max_steps", type=int, default=800, help="Maximum environment steps; the run always stops.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible scene and reset sampling.")
parser.add_argument("--passive_baseline", action="store_true", help="Measure curtain settling with zero joint actions.")
parser.add_argument("--video", action="store_true", help="Record a finite MP4 clip.")
parser.add_argument("--video_dir", type=str, default="videos/textile_atelier", help="MP4 output directory.")
parser.add_argument(
    "--video_source",
    type=str,
    default="visualizer:newton",
    choices=("visualizer:newton", "visualizer:kit"),
    help="Active visualizer to record; Kit needs camera rendering.",
)
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


class Phase(IntEnum):
    """Ordered phases of the measured two-hand curtain press."""

    SETTLE = 0
    APPROACH = 1
    KUKA_PRESS = 2
    GR_PRESS = 3
    HOLD = 4
    COMPLETE = 5


def _body_ids(robot, names: tuple[str, ...]) -> list[int]:
    """Collect collidable hand links for cloth-distance measurements."""
    selected: set[int] = set()
    for name in names:
        ids, _ = robot.find_bodies(name)
        selected.update(ids)
    if not selected:
        raise RuntimeError(f"No hand bodies matched {names!r}.")
    return sorted(selected)


class TextileAtelierStateMachine:
    """Press two material patches without directly moving the VBD sheet.

    A completion requires a sustained front-normal displacement beyond the
    settled shape while both collidable robot hands are near the curtain.
    Proximity is not represented as a force measurement.
    """

    def __init__(self, env) -> None:
        self.scene = env.scene
        self.num_envs = env.num_envs
        self.device = env.device
        self.kuka = self.scene["kuka"]
        self.humanoid = self.scene["humanoid"]
        self.cloth = self.scene["cloth"]
        self.kuka_ee_id = self.kuka.find_bodies("palm_link")[0][0]
        self.humanoid_ee_id = self.humanoid.find_bodies("right_hand_pitch_link")[0][0]
        self.kuka_hand_ids = _body_ids(self.kuka, ("palm_link", "(index|middle|thumb)_link_3"))
        self.humanoid_hand_ids = _body_ids(self.humanoid, ("right_hand_pitch_link", "R_(index|middle|thumb).*_link"))
        self.reference_nodes = self.cloth.data.default_nodal_state_w.torch[..., :3]
        self.left_patch, self.right_patch = curtain_deflection_patch_masks(self.reference_nodes)
        self.env_ids = torch.arange(self.num_envs, device=self.device)
        self.phase = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.phase_steps = torch.zeros_like(self.phase)
        self.hold_steps = torch.zeros_like(self.phase)
        self.completion_count = torch.zeros_like(self.phase)
        self.kuka_site = torch.zeros((self.num_envs, 3), device=self.device)
        self.humanoid_site = torch.zeros_like(self.kuka_site)
        self.press_start_kuka_pos = torch.zeros_like(self.kuka_site)
        self.press_start_humanoid_pos = torch.zeros_like(self.kuka_site)
        self.home_kuka_pos = torch.zeros_like(self.kuka_site)
        self.home_humanoid_pos = torch.zeros_like(self.kuka_site)
        self.baseline_left_y = torch.zeros(self.num_envs, device=self.device)
        self.baseline_right_y = torch.zeros_like(self.baseline_left_y)
        self.kuka_side = torch.ones_like(self.baseline_left_y)
        self.humanoid_side = torch.ones_like(self.baseline_left_y)
        self.peak_left_press = torch.zeros_like(self.baseline_left_y)
        self.peak_right_press = torch.zeros_like(self.baseline_left_y)
        self.ever_both_near = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Restart phases and capture live home hand poses after a scene reset."""
        if env_ids is None:
            env_ids = self.env_ids
        self.phase[env_ids] = Phase.SETTLE
        self.phase_steps[env_ids] = 0
        self.hold_steps[env_ids] = 0
        self.home_kuka_pos[env_ids] = self.kuka.data.body_pos_w.torch[env_ids, self.kuka_ee_id]
        self.home_humanoid_pos[env_ids] = self.humanoid.data.body_pos_w.torch[env_ids, self.humanoid_ee_id]
        self.baseline_left_y[env_ids] = 0.0
        self.baseline_right_y[env_ids] = 0.0

    def _set_phase(self, mask: torch.Tensor, phase: Phase) -> None:
        self.phase[mask] = phase
        self.phase_steps[mask] = 0

    @staticmethod
    def _patch_y(nodes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (nodes[..., 1] * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    def _site_near_hand(self, nodes: torch.Tensor, hand_pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        distance = torch.linalg.norm(nodes - hand_pos.unsqueeze(1), dim=-1)
        site_ids = distance.masked_fill(~mask, float("inf")).argmin(dim=1)
        return nodes[self.env_ids, site_ids]

    @staticmethod
    def _side_offset(side: torch.Tensor, distance: float) -> torch.Tensor:
        return torch.stack((torch.zeros_like(side), side * distance, torch.zeros_like(side)), dim=-1)

    @staticmethod
    def _position_in_root(robot, target_pos_w: torch.Tensor) -> torch.Tensor:
        return quat_apply_inverse(robot.data.root_quat_w.torch, target_pos_w - robot.data.root_pos_w.torch)

    def command(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Advance the measured FSM and return six task-space IK action values."""
        nodes = self.cloth.data.nodal_pos_w.torch
        kuka_pos = self.kuka.data.body_pos_w.torch[:, self.kuka_ee_id]
        humanoid_pos = self.humanoid.data.body_pos_w.torch[:, self.humanoid_ee_id]
        kuka_hand = self.kuka.data.body_pos_w.torch[:, self.kuka_hand_ids]
        humanoid_hand = self.humanoid.data.body_pos_w.torch[:, self.humanoid_hand_ids]
        kuka_patch_pairs = torch.cdist(kuka_hand, nodes).masked_fill(~self.left_patch.unsqueeze(1), float("inf"))
        humanoid_patch_pairs = torch.cdist(humanoid_hand, nodes).masked_fill(
            ~self.right_patch.unsqueeze(1), float("inf")
        )
        kuka_patch_distance = kuka_patch_pairs.amin(dim=(1, 2))
        humanoid_patch_distance = humanoid_patch_pairs.amin(dim=(1, 2))
        kuka_patch_nearest = kuka_patch_pairs.flatten(1).argmin(dim=1)
        humanoid_patch_nearest = humanoid_patch_pairs.flatten(1).argmin(dim=1)
        kuka_patch_node = nodes[self.env_ids, kuka_patch_nearest % nodes.shape[1]]
        humanoid_patch_node = nodes[self.env_ids, humanoid_patch_nearest % nodes.shape[1]]
        kuka_patch_link = kuka_hand[self.env_ids, kuka_patch_nearest // nodes.shape[1]]
        humanoid_patch_link = humanoid_hand[self.env_ids, humanoid_patch_nearest // nodes.shape[1]]
        self.ever_both_near |= (kuka_patch_distance < 0.08) & (humanoid_patch_distance < 0.08)

        left_y = self._patch_y(nodes, self.left_patch)
        right_y = self._patch_y(nodes, self.right_patch)
        left_press = -self.kuka_side * (left_y - self.baseline_left_y)
        right_press = -self.humanoid_side * (right_y - self.baseline_right_y)
        self.peak_left_press = torch.maximum(self.peak_left_press, left_press)
        self.peak_right_press = torch.maximum(self.peak_right_press, right_press)

        self.phase_steps += 1
        settled = (self.phase == Phase.SETTLE) & (self.phase_steps >= 40)
        self.kuka_site[settled] = self._site_near_hand(nodes, kuka_pos, self.left_patch)[settled]
        self.humanoid_site[settled] = self._site_near_hand(nodes, humanoid_pos, self.right_patch)[settled]
        self.kuka_side[settled] = torch.where(kuka_pos[settled, 1] >= self.kuka_site[settled, 1], 1.0, -1.0)
        self.humanoid_side[settled] = torch.where(humanoid_pos[settled, 1] >= self.humanoid_site[settled, 1], 1.0, -1.0)
        self.baseline_left_y[settled] = left_y[settled]
        self.baseline_right_y[settled] = right_y[settled]
        self._set_phase(settled, Phase.APPROACH)

        approaching = self.phase == Phase.APPROACH
        self.kuka_site[approaching] = self._site_near_hand(nodes, kuka_pos, self.left_patch)[approaching]
        self.humanoid_site[approaching] = self._site_near_hand(nodes, humanoid_pos, self.right_patch)[approaching]
        kuka_approach = self.kuka_site + self._side_offset(self.kuka_side, 0.04)
        humanoid_approach = self.humanoid_site + self._side_offset(self.humanoid_side, 0.10)
        kuka_goal = self.home_kuka_pos.clone()
        humanoid_goal = self.home_humanoid_pos.clone()

        kuka_goal[approaching] = kuka_approach[approaching]
        humanoid_goal[approaching] = humanoid_approach[approaching]
        reached_approach = (
            approaching & (self.phase_steps >= 15) & (kuka_patch_distance < 0.075) & (humanoid_patch_distance < 0.080)
        )
        self.press_start_kuka_pos[reached_approach] = kuka_pos[reached_approach]
        self.press_start_humanoid_pos[reached_approach] = humanoid_pos[reached_approach]
        kuka_link_side = kuka_patch_link[:, 1] - kuka_patch_node[:, 1]
        humanoid_link_side = humanoid_patch_link[:, 1] - humanoid_patch_node[:, 1]
        self.kuka_side[reached_approach] = torch.where(
            kuka_link_side[reached_approach].abs() >= 0.005,
            torch.sign(kuka_link_side[reached_approach]),
            self.kuka_side[reached_approach],
        )
        self.humanoid_side[reached_approach] = torch.where(
            humanoid_link_side[reached_approach].abs() >= 0.005,
            torch.sign(humanoid_link_side[reached_approach]),
            self.humanoid_side[reached_approach],
        )
        self._set_phase(reached_approach, Phase.KUKA_PRESS)
        kuka_press_goal = self.press_start_kuka_pos + self._side_offset(self.kuka_side, -0.035)
        humanoid_press_goal = self.press_start_humanoid_pos + self._side_offset(self.humanoid_side, -0.03)

        kuka_pressing = self.phase == Phase.KUKA_PRESS
        kuka_goal[kuka_pressing] = kuka_press_goal[kuka_pressing]
        humanoid_goal[kuka_pressing] = humanoid_approach[kuka_pressing]
        shaped_left = (
            kuka_pressing
            & (self.phase_steps >= 20)
            & (left_press >= 0.015)
            & ((left_press - right_press) >= 0.008)
            & (kuka_patch_distance < 0.07)
        )
        self._set_phase(shaped_left, Phase.GR_PRESS)

        gr_pressing = self.phase == Phase.GR_PRESS
        kuka_goal[gr_pressing] = kuka_press_goal[gr_pressing]
        humanoid_goal[gr_pressing] = humanoid_press_goal[gr_pressing]
        shaped_right = (
            gr_pressing & (self.phase_steps >= 20) & (right_press >= 0.015) & (humanoid_patch_distance < 0.07)
        )
        self._set_phase(shaped_right, Phase.HOLD)

        holding = self.phase == Phase.HOLD
        sustaining = holding | (self.phase == Phase.COMPLETE)
        kuka_goal[sustaining] = kuka_press_goal[sustaining]
        humanoid_goal[sustaining] = humanoid_press_goal[sustaining]
        qualifies = (
            holding
            & (left_press >= 0.012)
            & (right_press >= 0.012)
            & (kuka_patch_distance < 0.08)
            & (humanoid_patch_distance < 0.08)
        )
        self.hold_steps = torch.where(holding, torch.where(qualifies, self.hold_steps + 1, 0), self.hold_steps)
        completed = holding & (self.hold_steps >= 35)
        self.completion_count += completed.long()
        self._set_phase(completed, Phase.COMPLETE)

        actions = torch.cat(
            (
                self._position_in_root(self.kuka, kuka_goal),
                self._position_in_root(self.humanoid, humanoid_goal),
            ),
            dim=-1,
        )
        top_edge = self.reference_nodes[..., 2] >= self.reference_nodes[..., 2].amax(dim=1, keepdim=True) - 1.0e-4
        pin_targets = self.cloth.data.nodal_kinematic_target.torch[..., :3]
        top_error = torch.linalg.norm(nodes - pin_targets, dim=-1).masked_fill(~top_edge, 0.0).amax(dim=1)
        return actions, {
            "kuka_patch_distance": kuka_patch_distance,
            "humanoid_patch_distance": humanoid_patch_distance,
            "left_press": left_press,
            "right_press": right_press,
            "left_y": left_y,
            "right_y": right_y,
            "kuka_goal_error": torch.linalg.norm(kuka_pos - kuka_goal, dim=-1),
            "humanoid_goal_error": torch.linalg.norm(humanoid_pos - humanoid_goal, dim=-1),
            "top_pin_error": top_error,
            "cloth_min_z": nodes[..., 2].amin(dim=1),
        }

    def report(self, step: int, metrics: dict[str, torch.Tensor]) -> None:
        """Report observable motion without inferring unmeasured contact force."""
        counts = torch.bincount(self.phase, minlength=len(Phase)).cpu().tolist()
        phases = ", ".join(f"{phase.name.lower()}={counts[phase]}" for phase in Phase if counts[phase])
        print(
            f"[atelier step {step}] {phases}; "
            f"hand/patch distance={metrics['kuka_patch_distance'][0].item():.3f}/"
            f"{metrics['humanoid_patch_distance'][0].item():.3f} m; "
            f"mid-panel y={metrics['left_y'][0].item():+.3f}/{metrics['right_y'][0].item():+.3f} m; "
            f"left/right press since settle={metrics['left_press'][0].item():+.3f}/"
            f"{metrics['right_press'][0].item():+.3f} m; "
            f"goal error: KUKA={metrics['kuka_goal_error'][0].item():.3f} m, "
            f"GR1T2={metrics['humanoid_goal_error'][0].item():.3f} m; "
            f"top pin error={metrics['top_pin_error'][0].item():.4f} m, "
            f"hem min z={metrics['cloth_min_z'][0].item():.3f} m, "
            f"qualifying hold={self.hold_steps[0].item()} steps"
        )


def main() -> None:
    """Launch the finite, physically measured curtain demonstration."""
    if args_cli.num_envs < 1 or args_cli.max_steps < 1:
        parser.error("--num_envs and --max_steps must both be positive")
    env_cfg, _ = resolve_task_config(TASK_NAME, "")
    env_cfg.seed = args_cli.seed
    env_cfg.actions = type(env_cfg)().actions.joint if args_cli.passive_baseline else type(env_cfg)().actions.ik
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = max(
        env_cfg.episode_length_s, args_cli.max_steps * env_cfg.sim.dt * env_cfg.decimation + 1.0
    )
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    args_cli.visualizer_explicit = True
    if args_cli.video:
        if args_cli.video_source == "visualizer:kit":
            from isaaclab_visualizers.kit import KitVisualizerCfg

            args_cli.visualizer = ["kit"]
            env_cfg.sim.visualizer_cfgs = [
                KitVisualizerCfg(
                    eye=env_cfg.sim.default_visualizer_cfg.eye,
                    lookat=env_cfg.sim.default_visualizer_cfg.lookat,
                )
            ]
        else:
            args_cli.visualizer = ["newton_gl"]
            env_cfg.sim.visualizer_cfgs = [env_cfg.sim.default_visualizer_cfg]
        env_cfg.video_recorders = [
            VideoRecorderCfg(
                source=args_cli.video_source,
                output_dir=args_cli.video_dir,
                output_filename_prefix="textile_atelier",
                video_length=args_cli.max_steps,
                fps=30,
            )
        ]
        if args_cli.video_source == "visualizer:kit":
            args_cli.enable_cameras = True

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK_NAME, cfg=env_cfg)
        try:
            env.reset(seed=args_cli.seed)
            if args_cli.passive_baseline:
                zero_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                cloth = env.unwrapped.scene["cloth"]
                reference_nodes = cloth.data.default_nodal_state_w.torch[..., :3]
                top_edge = reference_nodes[..., 2] >= reference_nodes[..., 2].amax(dim=1, keepdim=True) - 1.0e-4
                pin_targets = cloth.data.nodal_kinematic_target.torch[..., :3]
                for step in range(args_cli.max_steps):
                    with torch.inference_mode():
                        env.step(zero_actions)
                        if (step + 1) % 100 == 0 or step == args_cli.max_steps - 1:
                            nodes = cloth.data.nodal_pos_w.torch
                            top_error = torch.linalg.norm(nodes - pin_targets, dim=-1)[top_edge].amax().item()
                            y = nodes[..., 1]
                            z = nodes[..., 2]
                            print(
                                f"[atelier passive step {step + 1}] cloth y min/max="
                                f"{y.amin().item():+.3f}/{y.amax().item():+.3f} m, "
                                f"z min/max={z.amin().item():.3f}/{z.amax().item():.3f} m, "
                                f"top pin error={top_error:.4f} m"
                            )
                return
            machine = TextileAtelierStateMachine(env.unwrapped)
            if env.action_space.shape[-1] != 6:
                raise RuntimeError(f"The IK preset must expose two 3D position actions; got {env.action_space.shape}.")
            for step in range(args_cli.max_steps):
                with torch.inference_mode():
                    actions, metrics = machine.command()
                    _, _, terminated, truncated, _ = env.step(actions)
                    if (step + 1) % 100 == 0 or step == args_cli.max_steps - 1:
                        machine.report(step + 1, metrics)
                    dones = terminated | truncated
                    if dones.any():
                        print(f"[atelier] {dones.sum().item()} environment(s) reset after a boundary/time-out.")
                        machine.reset(dones.nonzero(as_tuple=False).squeeze(-1))
            print(
                f"[atelier] measured completions: {machine.completion_count.sum().item()} total; "
                f"both hands near cloth: {machine.ever_both_near.sum().item()}/{env.unwrapped.num_envs} scenes; "
                f"peak left/right press: {machine.peak_left_press.max().item():.3f}/"
                f"{machine.peak_right_press.max().item():.3f} m."
            )
            if args_cli.video:
                print(f"[atelier] video output: {args_cli.video_dir}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
