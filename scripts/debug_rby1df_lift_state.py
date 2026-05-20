# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded state printer for the RBY1DF cube-lift task."""

import argparse
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.waterhose.launch import prepare_waterhose_launch
from isaaclab_tasks.utils import (
    add_launcher_args,
    fold_preset_tokens,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)


parser = argparse.ArgumentParser(description="Print RBY1DF lift task state while stepping bounded actions.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-RBY1DF-v0")
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--print_every", type=int, default=20)
parser.add_argument("--action_mode", choices=("zero", "random"), default="zero")
parser.add_argument("--random_scale", type=float, default=1.0)
parser.add_argument("--disable_table_collision", action="store_true", default=False)
parser.add_argument("--disable_object_collision", action="store_true", default=False)
parser.add_argument("--disable_table_object_collision", action="store_true", default=False)
parser.add_argument("--use_mujoco_contacts", action="store_true", default=False)
parser.add_argument("--contact_margin", type=float, default=None)
parser.add_argument("--contact_gap", type=float, default=None)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
prepare_waterhose_launch(args_cli)
sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)


def _ids(robot, expr):
    joint_ids, joint_names = robot.find_joints(expr, preserve_order=True)
    return joint_ids, joint_names


def _tensor(data):
    return data.torch if hasattr(data, "torch") else data


def _fmt(values):
    return "[" + ", ".join(f"{value:+.4f}" for value in values) + "]"


def _print_group(robot, group_name, expr):
    joint_ids, joint_names = _ids(robot, expr)
    joint_pos = _tensor(robot.data.joint_pos)[0, joint_ids].detach().cpu().tolist()
    joint_vel = _tensor(robot.data.joint_vel)[0, joint_ids].detach().cpu().tolist()
    joint_target = _tensor(robot.data.joint_pos_target)[0, joint_ids].detach().cpu().tolist()
    print(f"  {group_name} names: {joint_names}", flush=True)
    print(f"  {group_name} pos:    {_fmt(joint_pos)}", flush=True)
    print(f"  {group_name} target: {_fmt(joint_target)}", flush=True)
    print(f"  {group_name} vel:    {_fmt(joint_vel)}", flush=True)


def _print_state(env, step):
    robot = env.unwrapped.scene["robot"]
    obj = env.unwrapped.scene["object"]
    ee_frame = env.unwrapped.scene["ee_frame"]
    root_pose = _tensor(robot.data.root_link_pose_w)[0].detach().cpu().tolist()
    root_vel = _tensor(robot.data.root_link_vel_w)[0].detach().cpu().tolist()
    obj_pose = _tensor(obj.data.root_pose_w)[0].detach().cpu().tolist()
    ee_pos = _tensor(ee_frame.data.target_pos_w)[0, 0].detach().cpu().tolist()
    print(f"\n[STATE step={step}]", flush=True)
    print(f"  root pose [xyz xyzw]: {_fmt(root_pose)}", flush=True)
    print(f"  root vel  [lin ang]:  {_fmt(root_vel)}", flush=True)
    print(f"  object pose [xyz xyzw]: {_fmt(obj_pose)}", flush=True)
    print(f"  ee pos [xyz]: {_fmt(ee_pos)}", flush=True)
    print(f"  ee-object delta [xyz]: {_fmt([ee_pos[i] - obj_pose[i] for i in range(3)])}", flush=True)
    _print_group(robot, "torso", "torso_joint_[1-6]")
    _print_group(robot, "right_arm", "right_arm_joint_[1-7]")
    _print_group(robot, "left_arm", "left_arm_joint_[1-7]")
    _print_group(robot, "head", "head_joint_[1-2]")
    _print_group(robot, "right_gripper", "right_gripper_.*finger_joint.*")


def main():
    torch.manual_seed(42)
    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        if args_cli.disable_table_collision or args_cli.disable_table_object_collision:
            env_cfg.scene.table.spawn.collision_props.collision_enabled = False
        if args_cli.disable_object_collision or args_cli.disable_table_object_collision:
            env_cfg.scene.object.spawn.collision_props.collision_enabled = False
        if args_cli.use_mujoco_contacts:
            env_cfg.sim.physics.solver_cfg.use_mujoco_contacts = True
        if args_cli.contact_margin is not None:
            env_cfg.sim.physics.default_shape_cfg.margin = args_cli.contact_margin
            env_cfg.scene.table.spawn.collision_props.contact_margin = args_cli.contact_margin
            env_cfg.scene.object.spawn.collision_props.contact_margin = args_cli.contact_margin
        if args_cli.contact_gap is not None:
            env_cfg.sim.physics.default_shape_cfg.gap = args_cli.contact_gap
            env_cfg.scene.table.spawn.collision_props.contact_gap = args_cli.contact_gap
            env_cfg.scene.object.spawn.collision_props.contact_gap = args_cli.contact_gap
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

        env = gym.make(args_cli.task, cfg=env_cfg)
        robot = env.unwrapped.scene["robot"]
        print(f"[INFO] action_space={env.action_space}", flush=True)
        print(f"[INFO] observation_space={env.observation_space}", flush=True)
        print(f"[INFO] robot num_joints={robot.num_joints}", flush=True)
        print(f"[INFO] robot num_bodies={robot.num_bodies}", flush=True)
        print(f"[INFO] robot num_base_dofs={robot.num_base_dofs}", flush=True)
        print(f"[INFO] robot fixed_base={robot.is_fixed_base}", flush=True)
        print(f"[INFO] joint_names={robot.joint_names}", flush=True)

        env.reset()
        _print_state(env, 0)

        for step in range(1, args_cli.steps + 1):
            if args_cli.action_mode == "zero":
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            else:
                actions = args_cli.random_scale * (
                    2.0 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1.0
                )
            with torch.inference_mode():
                env.step(actions)
            if step % args_cli.print_every == 0:
                _print_state(env, step)

        env.close()


if __name__ == "__main__":
    main()
