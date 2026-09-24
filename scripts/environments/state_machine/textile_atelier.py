# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured two-arm state machine for the VBD textile atelier.

.. code-block:: bash

    uv run python scripts/environments/state_machine/textile_atelier.py --max_steps 800
    uv run python scripts/environments/state_machine/textile_atelier.py --passive_baseline --max_steps 800
    uv run --extra video python scripts/environments/state_machine/textile_atelier.py --video --max_steps 800

The sheet is never moved by the script: only robot arm IK targets are commanded.
KUKA presses a local cloth patch while GR1T2 presents its hand at the far edge.
"""

from __future__ import annotations

import argparse
import sys
from enum import IntEnum

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms

import isaaclab_tasks.contrib.textile_atelier  # noqa: F401
from isaaclab_tasks.contrib.textile_atelier.mdp import cloth_press_patch_masks
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

TASK_NAME = "IsaacContrib-Textile-Atelier-Kuka-GR1T2"

parser = argparse.ArgumentParser(description="Run the two-arm VBD textile atelier state machine.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel atelier scenes.")
parser.add_argument("--max_steps", type=int, default=800, help="Maximum environment steps; the run always stops.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible scene and reset sampling.")
parser.add_argument("--passive_baseline", action="store_true", help="Measure cloth settling with zero joint actions.")
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
    """Ordered, measured phases of the local cloth-press motion."""

    SETTLE = 0
    APPROACH = 1
    PRESS = 2
    DRAPE = 3
    HOLD = 4
    COMPLETE = 5


def _body_ids(robot, names: tuple[str, ...]) -> list[int]:
    """Collect the available robot bodies used for hand-to-cloth distance checks."""
    selected: set[int] = set()
    for name in names:
        ids, _ = robot.find_bodies(name)
        selected.update(ids)
    if not selected:
        raise RuntimeError(f"No hand bodies matched {names!r}.")
    return sorted(selected)


class TextileAtelierStateMachine:
    """Command two IK end-effectors from measured robot and cloth state.

    Proximity is measured from collidable links to cloth nodes [m], not force sensors.
    A completion requires a sustained local left-patch drop relative to the
    far-side patch while both hands remain near the sheet. There is no release.
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
        self.kuka_hand_ids = _body_ids(
            self.kuka,
            ("palm_link", "(index|middle|thumb)_link_3"),
        )
        self.humanoid_hand_ids = _body_ids(
            self.humanoid,
            ("right_hand_pitch_link", "R_(index|middle|thumb).*_link"),
        )
        self.phase = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.phase_steps = torch.zeros_like(self.phase)
        self.qualified_hold_steps = torch.zeros_like(self.phase)
        reference_nodes = self.cloth.data.default_nodal_state_w.torch[..., :3]
        self.left_patch, self.right_patch = cloth_press_patch_masks(reference_nodes)
        self.baseline_asymmetry = torch.zeros(self.num_envs, device=self.device)
        self.max_press_asymmetry = torch.zeros(self.num_envs, device=self.device)
        self.best_press_asymmetry = torch.zeros(self.num_envs, device=self.device)
        self.press_bias = torch.zeros(self.num_envs, device=self.device)
        self.press_start_kuka_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.press_start_humanoid_pos = torch.zeros_like(self.press_start_kuka_pos)
        self.both_near_sheet = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ever_near_sheet = torch.zeros_like(self.both_near_sheet)
        self.completion_count = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.home_kuka_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.home_humanoid_pos = torch.zeros_like(self.home_kuka_pos)
        self.home_kuka_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.kuka_site_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.humanoid_site_id = torch.zeros_like(self.kuka_site_id)
        self.env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the phase and capture live hand poses after an environment reset."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.phase[env_ids] = Phase.SETTLE
        self.phase_steps[env_ids] = 0
        self.qualified_hold_steps[env_ids] = 0
        self.home_kuka_pos[env_ids] = self.kuka.data.body_pos_w.torch[env_ids, self.kuka_ee_id]
        self.home_humanoid_pos[env_ids] = self.humanoid.data.body_pos_w.torch[env_ids, self.humanoid_ee_id]
        self.home_kuka_quat[env_ids] = self.kuka.data.body_quat_w.torch[env_ids, self.kuka_ee_id]
        self.baseline_asymmetry[env_ids] = 0.0
        self.max_press_asymmetry[env_ids] = 0.0
        self.press_bias[env_ids] = 0.0
        self.press_start_kuka_pos[env_ids] = 0.0
        self.press_start_humanoid_pos[env_ids] = 0.0
        self.both_near_sheet[env_ids] = False
        self.kuka_site_id[env_ids] = 0
        self.humanoid_site_id[env_ids] = 0

    def _set_phase(self, mask: torch.Tensor, next_phase: Phase) -> None:
        self.phase[mask] = next_phase
        self.phase_steps[mask] = 0

    @staticmethod
    def _patch_height(nodes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (nodes[..., 2] * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    @staticmethod
    def _step_toward(current: torch.Tensor, target: torch.Tensor, max_step: float = 0.012) -> torch.Tensor:
        delta = target - current
        scale = torch.clamp(max_step / torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1.0e-6), max=1.0)
        return current + scale * delta

    def _target_in_root(self, robot, target_pos_w: torch.Tensor, target_quat_w: torch.Tensor) -> torch.Tensor:
        pos_b, quat_b = subtract_frame_transforms(
            robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, target_pos_w, target_quat_w
        )
        return torch.cat((pos_b, quat_b), dim=-1)

    def _position_in_root(self, robot, target_pos_w: torch.Tensor) -> torch.Tensor:
        return quat_apply_inverse(robot.data.root_quat_w.torch, target_pos_w - robot.data.root_pos_w.torch)

    def command(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Advance the finite-state machine and return ten IK action values."""
        nodes = self.cloth.data.nodal_pos_w.torch
        x = nodes[..., 0]
        span = x.amax(dim=1) - x.amin(dim=1)

        kuka_pos = self.kuka.data.body_pos_w.torch[:, self.kuka_ee_id]
        humanoid_pos = self.humanoid.data.body_pos_w.torch[:, self.humanoid_ee_id]
        kuka_hand = self.kuka.data.body_pos_w.torch[:, self.kuka_hand_ids]
        humanoid_hand = self.humanoid.data.body_pos_w.torch[:, self.humanoid_hand_ids]
        kuka_distance = torch.cdist(kuka_hand, nodes).amin(dim=(1, 2))
        humanoid_distance = torch.cdist(humanoid_hand, nodes).amin(dim=(1, 2))
        self.both_near_sheet |= (kuka_distance < 0.08) & (humanoid_distance < 0.08)
        self.ever_near_sheet |= self.both_near_sheet

        self.phase_steps += 1
        settling = (self.phase == Phase.SETTLE) & (self.phase_steps >= 30)
        # Choose material points that each hand can reach from its measured settled pose.
        # Restrict the candidates to opposite halves of the sheet, then track those
        # same material points as the cloth deforms instead of chasing a moving centroid.
        mid_x = 0.5 * (x.amin(dim=1) + x.amax(dim=1))
        kuka_site_distance = torch.linalg.norm(nodes - kuka_pos.unsqueeze(1), dim=-1)
        humanoid_site_distance = torch.linalg.norm(nodes - humanoid_pos.unsqueeze(1), dim=-1)
        kuka_nearest = kuka_site_distance.masked_fill(x > mid_x.unsqueeze(1), float("inf")).argmin(dim=1)
        humanoid_nearest = humanoid_site_distance.masked_fill(x < mid_x.unsqueeze(1), float("inf")).argmin(dim=1)
        self.kuka_site_id[settling] = kuka_nearest[settling]
        self.humanoid_site_id[settling] = humanoid_nearest[settling]
        left_height = self._patch_height(nodes, self.left_patch)
        right_height = self._patch_height(nodes, self.right_patch)
        self.baseline_asymmetry[settling] = (right_height - left_height)[settling]
        self._set_phase(settling, Phase.APPROACH)
        kuka_site = nodes[self.env_ids, self.kuka_site_id]
        humanoid_site = nodes[self.env_ids, self.humanoid_site_id]
        press_asymmetry = right_height - left_height - self.baseline_asymmetry

        kuka_approach = kuka_site + torch.tensor([0.0, 0.0, 0.12], device=self.device)
        humanoid_approach = humanoid_site + torch.tensor([0.0, 0.0, 0.12], device=self.device)
        kuka_goal = self.home_kuka_pos.clone()
        humanoid_goal = self.home_humanoid_pos.clone()

        approaching = self.phase == Phase.APPROACH
        kuka_goal[approaching] = kuka_approach[approaching]
        humanoid_goal[approaching] = humanoid_approach[approaching]
        reached_approach = (
            approaching
            & (self.phase_steps >= 12)
            # KUKA's raised hover pose is not reachable with its held wrist
            # orientation, but its fingers can still rest beside the sheet.
            & (kuka_distance < 0.06)
            # GR1T2 reaches a measured hover pose; cloth proximity is checked
            # in PRESS, after the arm has descended from that hover.
            & (torch.linalg.norm(humanoid_pos - humanoid_approach, dim=-1) < 0.07)
        )
        self._set_phase(reached_approach, Phase.PRESS)

        pressing = self.phase == Phase.PRESS
        kuka_press_offset = torch.clamp(0.11 - self.phase_steps.float() * 0.001, min=-0.01)
        humanoid_press_offset = torch.clamp(0.17 - self.phase_steps.float() * 0.001, min=0.03)
        kuka_press_goal = kuka_site.clone()
        humanoid_press_goal = humanoid_site.clone()
        kuka_press_goal[:, 2] += kuka_press_offset
        humanoid_press_goal[:, 2] += humanoid_press_offset
        kuka_goal[pressing] = kuka_press_goal[pressing]
        humanoid_goal[pressing] = humanoid_press_goal[pressing]
        reached_sheet = pressing & (self.phase_steps >= 25) & (kuka_distance < 0.08) & (humanoid_distance < 0.08)
        self.press_start_kuka_pos[reached_sheet] = kuka_pos[reached_sheet]
        self.press_start_humanoid_pos[reached_sheet] = humanoid_pos[reached_sheet]
        site_separation = humanoid_site[:, 0] - kuka_site[:, 0]
        self.press_bias[reached_sheet] = (0.12 * site_separation[reached_sheet]).clamp(0.035, 0.06)
        self._set_phase(reached_sheet, Phase.DRAPE)

        draping = self.phase == Phase.DRAPE
        self.max_press_asymmetry = torch.maximum(
            self.max_press_asymmetry, torch.where(self.phase >= Phase.DRAPE, press_asymmetry, 0.0)
        )
        self.best_press_asymmetry = torch.maximum(self.best_press_asymmetry, self.max_press_asymmetry)
        # Mild opposing IK biases sustain the two presentation poses; the cloth
        # response, not commanded wrist travel, determines whether draping occurred.
        kuka_drape_goal = self.press_start_kuka_pos.clone()
        humanoid_drape_goal = self.press_start_humanoid_pos.clone()
        kuka_drape_goal[:, 0] += self.press_bias
        humanoid_drape_goal[:, 0] -= self.press_bias
        kuka_goal[draping] = kuka_drape_goal[draping]
        humanoid_goal[draping] = humanoid_drape_goal[draping]
        draped = (
            draping
            & (self.phase_steps >= 25)
            & (press_asymmetry >= 0.010)
            & (kuka_distance < 0.04)
            & (humanoid_distance < 0.08)
        )
        self._set_phase(draped, Phase.HOLD)
        self.qualified_hold_steps[draped] = 0

        holding = self.phase == Phase.HOLD
        complete = self.phase == Phase.COMPLETE
        sustaining = holding | complete
        kuka_goal[sustaining] = kuka_drape_goal[sustaining]
        humanoid_goal[sustaining] = humanoid_drape_goal[sustaining]
        qualifying_hold = holding & (press_asymmetry >= 0.010) & (kuka_distance < 0.04) & (humanoid_distance < 0.08)
        self.qualified_hold_steps = torch.where(
            holding, torch.where(qualifying_hold, self.qualified_hold_steps + 1, 0), self.qualified_hold_steps
        )
        sustained_press = holding & (self.qualified_hold_steps >= 45)
        self.completion_count += sustained_press.long()
        self._set_phase(sustained_press, Phase.COMPLETE)

        kuka_target = self._step_toward(kuka_pos, kuka_goal)
        humanoid_target = self._step_toward(humanoid_pos, humanoid_goal)
        actions = torch.cat(
            (
                self._target_in_root(self.kuka, kuka_target, self.home_kuka_quat),
                self._position_in_root(self.humanoid, humanoid_target),
            ),
            dim=-1,
        )
        return actions, {
            "kuka_distance": kuka_distance,
            "humanoid_distance": humanoid_distance,
            "press_asymmetry": press_asymmetry,
            "left_patch_height": left_height,
            "right_patch_height": right_height,
            "cloth_min_z": nodes[..., 2].amin(dim=1),
            "cloth_mean_z": nodes[..., 2].mean(dim=1),
            "cloth_max_z": nodes[..., 2].amax(dim=1),
            "cloth_x_span": span,
            "kuka_wrist_x": kuka_pos[:, 0],
            "humanoid_wrist_x": humanoid_pos[:, 0],
            "kuka_goal_x": kuka_goal[:, 0],
            "humanoid_goal_x": humanoid_goal[:, 0],
            "kuka_goal_error": torch.linalg.norm(kuka_pos - kuka_goal, dim=-1),
            "humanoid_goal_error": torch.linalg.norm(humanoid_pos - humanoid_goal, dim=-1),
        }

    def report(self, step: int, metrics: dict[str, torch.Tensor]) -> None:
        """Print physical progress metrics without inferring unmeasured contact forces."""
        counts = torch.bincount(self.phase, minlength=len(Phase)).cpu().tolist()
        phases = ", ".join(f"{phase.name.lower()}={counts[phase]}" for phase in Phase if counts[phase])
        print(
            f"[atelier step {step}] {phases}; "
            f"nearest hand-link/cloth: KUKA={metrics['kuka_distance'].min().item():.3f} m, "
            f"GR1T2={metrics['humanoid_distance'].min().item():.3f} m; "
            f"left/right patch z={metrics['left_patch_height'][0].item():.3f}/"
            f"{metrics['right_patch_height'][0].item():.3f} m; "
            f"press asymmetry since settle={metrics['press_asymmetry'][0].item():.3f} m "
            f"(peak={self.max_press_asymmetry.max().item():.3f} m, "
            f"qualifying hold={self.qualified_hold_steps[0].item()} steps); "
            f"wrist/goal x: KUKA={metrics['kuka_wrist_x'][0].item():.3f}/{metrics['kuka_goal_x'][0].item():.3f} m, "
            f"GR1T2={metrics['humanoid_wrist_x'][0].item():.3f}/{metrics['humanoid_goal_x'][0].item():.3f} m; "
            f"active goal errors: KUKA={metrics['kuka_goal_error'][0].item():.3f} m, "
            f"GR1T2={metrics['humanoid_goal_error'][0].item():.3f} m; "
            f"cloth z min/mean/max={metrics['cloth_min_z'][0].item():.3f}/"
            f"{metrics['cloth_mean_z'][0].item():.3f}/{metrics['cloth_max_z'][0].item():.3f} m, "
            f"x-span={metrics['cloth_x_span'][0].item():.3f} m"
        )


def main() -> None:
    """Launch the environment and run a finite, measurable two-arm demonstration."""
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
                    eye=(1.90, -2.30, 1.70),
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
                settled_min_z = None
                for step in range(args_cli.max_steps):
                    with torch.inference_mode():
                        env.step(zero_actions)
                        nodes = env.unwrapped.scene["cloth"].data.nodal_pos_w.torch
                        if step == 29:
                            settled_min_z = nodes[..., 2].amin().item()
                        if (step + 1) % 100 == 0 or step == args_cli.max_steps - 1:
                            z = nodes[..., 2]
                            span = nodes[..., 0].amax(dim=1) - nodes[..., 0].amin(dim=1)
                            drop = 0.0 if settled_min_z is None else settled_min_z - z.amin().item()
                            print(
                                f"[atelier passive step {step + 1}] cloth z min/mean/max="
                                f"{z.amin().item():.3f}/{z.mean().item():.3f}/{z.amax().item():.3f} m, "
                                f"x-span={span[0].item():.3f} m, post-settle min-z drop={drop:.3f} m"
                            )
                return
            machine = TextileAtelierStateMachine(env.unwrapped)
            if env.action_space.shape[-1] != 10:
                raise RuntimeError(
                    f"The IK preset must expose a 7D KUKA pose and 3D humanoid position; got {env.action_space.shape}."
                )
            for step in range(args_cli.max_steps):
                with torch.inference_mode():
                    actions, metrics = machine.command()
                    _, _, terminated, truncated, _ = env.step(actions)
                    if (step + 1) % 100 == 0 or step == args_cli.max_steps - 1:
                        machine.report(step + 1, metrics)
                    dones = terminated | truncated
                    if dones.any():
                        print(f"[atelier] {dones.sum().item()} environment(s) reset after a cloth boundary/time-out.")
                        machine.reset(dones.nonzero(as_tuple=False).squeeze(-1))
            print(
                f"[atelier] qualified completions: {machine.completion_count.sum().item()} total; "
                f"both hands near cloth: {machine.ever_near_sheet.sum().item()}/{env.unwrapped.num_envs} scenes; "
                f"peak localized press asymmetry: {machine.best_press_asymmetry.max().item():.3f} m."
            )
            if args_cli.video:
                print(f"[atelier] video output: {args_cli.video_dir}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
