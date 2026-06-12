# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal probe for the Newton IK action on the waterhose task.

Commands "hold the current end-effector pose" as an absolute pose action and logs,
every N steps, the action's internal world-frame targets, the live body pose, and
the per-frame error under both EE-offset conventions. Localizes convention or
solver issues without the scripted state machine in the loop.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Waterhose-Coupled-v0")
parser.add_argument("--steps", type=int, default=240)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.contrib.waterhose.geometry import (  # noqa: E402
    RIGHT_GRIPPER_EE_FRAME_POS,
    RIGHT_GRIPPER_EE_FRAME_QUAT_XYZW,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.terminations.success = None
    env_cfg.terminations.time_out = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    ee_body = robot.find_bodies("right_gripper_base")[0][0]
    device = env.device

    offset_pos = torch.tensor(RIGHT_GRIPPER_EE_FRAME_POS, device=device).unsqueeze(0)
    quat_const = RIGHT_GRIPPER_EE_FRAME_QUAT_XYZW
    # Both readings of the historical constant: as wxyz (SM convention) and as true xyzw.
    offset_quat_as_wxyz = torch.tensor(quat_const, device=device).unsqueeze(0)
    offset_quat_xyzw_to_wxyz = torch.tensor(
        [quat_const[3], quat_const[0], quat_const[1], quat_const[2]], device=device
    ).unsqueeze(0)

    def ee_pose(offset_quat_wxyz):
        body_pos = robot.data.body_pos_w.torch[:, ee_body]
        body_quat = robot.data.body_quat_w.torch[:, ee_body]
        return math_utils.combine_frame_transforms(body_pos, body_quat, offset_pos, offset_quat_wxyz)

    root_pos = robot.data.root_pos_w.torch
    root_quat = robot.data.root_quat_w.torch

    # Freeze the initial pose (SM-convention frame) as the fixed target: "stay here".
    hold_pos_w, hold_quat_w = ee_pose(offset_quat_as_wxyz)
    cmd_pos_b, cmd_quat_b = math_utils.subtract_frame_transforms(root_pos, root_quat, hold_pos_w, hold_quat_w)

    term = env.action_manager.get_term("arm_action")
    total_dim = env.action_manager.total_action_dim
    actions = torch.zeros((1, total_dim), device=device)
    actions[:, 0:3] = cmd_pos_b
    actions[:, 3:7] = cmd_quat_b  # (x, y, z, w) end to end
    if total_dim >= 22:
        # Hold objectives: current left gripper + torso poses.
        for slot, name in ((7, "left_gripper_base"), (14, "torso_hip_yaw")):
            body_id = robot.find_bodies(name)[0][0]
            pos_b, quat_b = math_utils.subtract_frame_transforms(
                root_pos, root_quat, robot.data.body_pos_w.torch[:, body_id], robot.data.body_quat_w.torch[:, body_id]
            )
            actions[:, slot : slot + 3] = pos_b
            actions[:, slot + 3 : slot + 7] = quat_b
    actions[:, -1] = 1.0  # gripper open

    print(f"[probe] action dim={total_dim}")
    print(f"[probe] hold target (world, SM frame): pos={hold_pos_w[0].tolist()} quat_wxyz={hold_quat_w[0].tolist()}")

    for step in range(args_cli.steps):
        env.step(actions)
        if step % 40 == 0 or step == args_cli.steps - 1:
            cur_pos_sm, cur_quat_sm = ee_pose(offset_quat_as_wxyz)
            cur_pos_tx, cur_quat_tx = ee_pose(offset_quat_xyzw_to_wxyz)
            err_sm_pos = (cur_pos_sm - hold_pos_w).norm().item()
            err_sm_rot = math_utils.quat_error_magnitude(cur_quat_sm, hold_quat_w).item()
            err_tx_pos = (cur_pos_tx - hold_pos_w).norm().item()
            err_tx_rot = math_utils.quat_error_magnitude(cur_quat_tx, hold_quat_w).item()
            driver = term._drivers[0]
            tgt_pos = driver.objective.position_objective.target_positions.numpy()[0]
            tgt_rot = driver.objective.rotation_objective.target_rotations.numpy()[0]
            print(
                f"[probe] step {step:4d}: err(SM frame) pos={err_sm_pos:.4f} rot={err_sm_rot:.4f} | "
                f"err(true-xyzw frame) pos={err_tx_pos:.4f} rot={err_tx_rot:.4f} | "
                f"objective target pos={tgt_pos.round(4).tolist()} rot_xyzw={tgt_rot.round(4).tolist()}",
                flush=True,
            )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
