# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted IK demo for the RBY1 waterhose task.

.. code-block:: bash

    WATERHOSE_ASSETS_DIR=/path/to/waterhose/assets \
      ./isaaclab.sh -p scripts/environments/state_machine/waterhose_rby1_ik.py \
      --task Isaac-Waterhose-RBY1-IK-Abs-v0 --num_envs 1
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run a scripted IK demo for the RBY1 waterhose task.")
parser.add_argument("--task", type=str, default="Isaac-Waterhose-RBY1-IK-Abs-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--max_steps", type=int, default=1800, help="Maximum environment steps. Use 0 to run until closed.")
parser.add_argument("--settle_time", type=float, default=2.0, help="Seconds to hold the initial pose before moving.")
parser.add_argument("--debug", action="store_true", help="Print phase transitions and target positions for env 0.")
parser.add_argument(
    "--asset_root",
    type=str,
    default=os.environ.get("WATERHOSE_ASSETS_DIR", ""),
    help="Optional waterhose asset root. Also accepted through WATERHOSE_ASSETS_DIR.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.asset_root:
    os.environ["WATERHOSE_ASSETS_DIR"] = args_cli.asset_root

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


class WaterhoseDemoState:
    """Small per-env scripted pick/insert state machine."""

    REST = 0
    APPROACH = 1
    ENGAGE = 2
    GRASP = 3
    LIFT = 4
    MOVE_TO_SOCKET = 5
    INSERT = 6
    RELEASE = 7
    WITHDRAW = 8
    DONE = 9

    PHASE_NAMES = ("REST", "APPROACH", "ENGAGE", "GRASP", "LIFT", "MOVE_TO_SOCKET", "INSERT", "RELEASE", "WITHDRAW", "DONE")
    DURATIONS = (0.25, 4.0, 2.0, 0.8, 2.0, 4.0, 2.0, 0.8, 2.0, 1.0e6)

    def __init__(self, num_envs: int, step_dt: float, device: torch.device | str, settle_time: float, debug: bool):
        self.num_envs = int(num_envs)
        self.step_dt = float(step_dt)
        self.device = device
        self.debug = bool(debug)
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.elapsed = torch.zeros(self.num_envs, device=device)
        self.last_reported_phase = torch.full((self.num_envs,), -1, dtype=torch.long, device=device)
        self.phase_start_pose = torch.zeros((self.num_envs, 7), device=device)
        self.command_pose = torch.zeros((self.num_envs, 7), device=device)
        self.command_pose[:, 6] = 1.0
        durations = list(self.DURATIONS)
        durations[self.REST] = max(float(settle_time), self.step_dt)
        self.durations = torch.tensor(durations, dtype=torch.float32, device=device)
        self.approach_offset_b = torch.tensor([0.0, 0.0, 0.10], dtype=torch.float32, device=device)
        self.engage_offset_b = torch.tensor([0.0, 0.0, 0.025], dtype=torch.float32, device=device)
        self.lift_offset_b = torch.tensor([0.0, 0.0, 0.16], dtype=torch.float32, device=device)
        self.socket_pre_insert_offset_b = torch.tensor([0.0, -0.10, 0.06], dtype=torch.float32, device=device)
        self.insert_offset_b = torch.tensor([0.0, 0.0, 0.02], dtype=torch.float32, device=device)
        self.withdraw_offset_b = torch.tensor([0.0, -0.12, 0.04], dtype=torch.float32, device=device)
        self.socket_pos_w = torch.tensor(
            [-0.259404, 0.362961, 0.5 - 0.262711],
            dtype=torch.float32,
            device=device,
        ).repeat(self.num_envs, 1)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.phase[env_ids] = self.REST
        self.elapsed[env_ids] = 0.0
        self.last_reported_phase[env_ids] = -1

    def compute(self, env) -> torch.Tensor:
        robot = env.scene["robot"]
        plug = env.scene["plug1"]
        ee_body_id = robot.find_bodies("right_gripper_dummy")[0][0]

        root_pos_w = robot.data.root_pos_w.torch
        root_quat_w = robot.data.root_quat_w.torch
        ee_pos_w = robot.data.body_pos_w.torch[:, ee_body_id]
        ee_quat_w = robot.data.body_quat_w.torch[:, ee_body_id]
        plug_pos_w = plug.data.root_pos_w.torch
        plug_quat_w = plug.data.root_quat_w.torch

        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        plug_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, plug_pos_w, plug_quat_w)
        socket_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, self.socket_pos_w, None)

        current_pose = torch.cat((ee_pos_b, ee_quat_b), dim=-1)
        first_step = self.elapsed == 0.0
        if torch.any(first_step):
            self.phase_start_pose[first_step] = current_pose[first_step]

        target_pose = self.phase_start_pose.clone()
        # The scripted path is position-first. Keep the currently observed gripper orientation as the command
        # orientation so IK does not fight a stale or misaligned grasp-frame quaternion while the cable settles.
        target_pose[:, 3:] = ee_quat_b
        gripper = torch.ones((self.num_envs, 1), device=self.device)

        approach = self.phase == self.APPROACH
        target_pose[approach, :3] = plug_pos_b[approach] + self.approach_offset_b

        engage = self.phase == self.ENGAGE
        target_pose[engage, :3] = plug_pos_b[engage] + self.engage_offset_b

        grasp = self.phase == self.GRASP
        target_pose[grasp, :3] = self.phase_start_pose[grasp, :3]
        gripper[grasp] = -1.0

        lift = self.phase == self.LIFT
        target_pose[lift, :3] = self.phase_start_pose[lift, :3] + self.lift_offset_b
        gripper[lift] = -1.0

        move_to_socket = self.phase == self.MOVE_TO_SOCKET
        target_pose[move_to_socket, :3] = socket_pos_b[move_to_socket] + self.socket_pre_insert_offset_b
        gripper[move_to_socket] = -1.0

        insert = self.phase == self.INSERT
        target_pose[insert, :3] = socket_pos_b[insert] + self.insert_offset_b
        gripper[insert] = -1.0

        release = self.phase == self.RELEASE
        target_pose[release, :3] = self.phase_start_pose[release, :3]

        withdraw = self.phase == self.WITHDRAW
        target_pose[withdraw, :3] = self.phase_start_pose[withdraw, :3] + self.withdraw_offset_b

        done = self.phase == self.DONE
        target_pose[done, :3] = self.phase_start_pose[done, :3]

        durations = self.durations[self.phase].unsqueeze(-1)
        blend = torch.clamp((self.elapsed / durations.squeeze(-1)).unsqueeze(-1), 0.0, 1.0)
        blend = blend * blend * (3.0 - 2.0 * blend)
        self.command_pose[:, :3] = self.phase_start_pose[:, :3] * (1.0 - blend) + target_pose[:, :3] * blend
        self.command_pose[:, 3:] = target_pose[:, 3:]

        actions = torch.cat((self.command_pose, gripper), dim=-1)

        if self.debug:
            changed = self.phase != self.last_reported_phase
            if bool(changed[0].item()):
                phase_name = self.PHASE_NAMES[int(self.phase[0].item())]
                print(
                    f"[waterhose_ik] {phase_name}: "
                    f"ee={ee_pos_b[0].detach().cpu().tolist()} "
                    f"plug={plug_pos_b[0].detach().cpu().tolist()} "
                    f"socket={socket_pos_b[0].detach().cpu().tolist()} "
                    f"target={target_pose[0, :3].detach().cpu().tolist()}",
                    flush=True,
                )
            self.last_reported_phase[changed] = self.phase[changed]

        self.elapsed += self.step_dt
        should_advance = self.elapsed >= self.durations[self.phase]
        should_advance &= self.phase < self.DONE
        if torch.any(should_advance):
            self.phase[should_advance] += 1
            self.elapsed[should_advance] = 0.0
            self.phase_start_pose[should_advance] = current_pose[should_advance]

        return actions


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    state_machine = WaterhoseDemoState(env.num_envs, env.step_dt, env.device, args_cli.settle_time, args_cli.debug)
    actions = state_machine.compute(env)

    step = 0
    while simulation_app.is_running() and (args_cli.max_steps <= 0 or step < args_cli.max_steps):
        _, _, terminated, truncated, _ = env.step(actions)
        dones = terminated | truncated
        if torch.any(dones):
            state_machine.reset(dones.nonzero(as_tuple=False).squeeze(-1))
        actions = state_machine.compute(env)
        step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
