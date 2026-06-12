# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse

import torch
import warp as wp

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation


parser = argparse.ArgumentParser(description="Compare scoop IK target, IK-predicted pose, and simulated pose.")
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--cmd", type=float, nargs=4, default=(0.0, 0.0, 0.8, 0.0))
parser.add_argument("--ik-rotation-weight", type=float, default=None)
parser.add_argument("--max-ik-delta", type=float, default=None)
parser.add_argument("--disable-bowl-rigid-collision", action="store_true")
add_launcher_args(parser)
args_cli = parser.parse_args()


def _qrot_t(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    xyz, w = q[..., :3], q[..., 3:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _fmt(v: torch.Tensor) -> str:
    return "[" + ", ".join(f"{float(x):+.4f}" for x in v.detach().cpu().flatten()) + "]"


def _ik_predicted_bowl(env) -> torch.Tensor:
    from newton._src.sim.ik.ik_common import eval_fk_batched

    torch_dev = env._ik_default.device
    wp_dev = env._ik_model.device
    joint_q = env._ik_default.unsqueeze(0).expand(env.num_envs, -1).clone()
    joint_q[:, env._ik_arm] = env.arm_joint_q().to(torch_dev)
    joint_q[:, env._ik_fingers] = float(env.cfg.gripper_open_pos)
    body_q = wp.zeros((env.num_envs, env._ik_model.body_count), dtype=wp.transform, device=wp_dev)
    body_qd = wp.zeros((env.num_envs, env._ik_model.body_count), dtype=wp.spatial_vector, device=wp_dev)
    joint_qd = wp.zeros((env.num_envs, env._ik_model.joint_dof_count), dtype=wp.float32, device=wp_dev)
    eval_fk_batched(env._ik_model, joint_q, joint_qd, body_q, body_qd)
    ee = [i for i, label in enumerate(env._ik_model.body_label) if str(label).endswith("panda_hand")][0]
    hand = wp.to_torch(body_q)[:, ee]
    quat = hand[:, 3:7]
    quat = quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=1.0e-8)
    off = env._bowl_center_hand_t.unsqueeze(0).expand(env.num_envs, -1)
    return hand[:, :3] + _qrot_t(quat, off)


def main() -> None:
    from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg
    from isaaclab_newton.physics import NewtonManager
    from newton import ShapeFlags

    cfg = FrankaScoopEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = str(args_cli.device)
    if args_cli.ik_rotation_weight is not None:
        cfg.ik_rotation_weight = args_cli.ik_rotation_weight
    if args_cli.max_ik_delta is not None:
        cfg.max_ik_delta = args_cli.max_ik_delta

    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg)
        try:
            env.reset()
            if args_cli.disable_bowl_rigid_collision:
                model = NewtonManager.get_model()
                flags = model.shape_flags.numpy()
                for sid, label in enumerate(model.shape_label):
                    if "BowlRigid" in str(label):
                        flags[sid] &= ~int(ShapeFlags.COLLIDE_SHAPES)
                model.shape_flags.assign(flags)
            action = torch.tensor(args_cli.cmd, device=env.device, dtype=torch.float32).view(1, 4)
            prev = env.arm_joint_q().clone()
            for step in range(args_cli.steps):
                obs, _, _, _, _ = env.step(action)
                pred = _ik_predicted_bowl(env)
                actual = env.bowl_pos_e()
                target = env._target_bowl_e
                q = env.arm_joint_q()
                ctrl = wp.to_torch(NewtonManager.get_control().joint_target_q)[env._arm_q_ids]
                if step % 5 == 0 or step == args_cli.steps - 1:
                    print(
                        f"[debug-ik] step={step:03d} target={_fmt(target[0])} "
                        f"ik_pred={_fmt(pred[0])} actual={_fmt(actual[0])} "
                        f"target_err={float(torch.linalg.norm(actual - target, dim=-1)[0]):.5f} "
                        f"ik_err={float(torch.linalg.norm(pred - target, dim=-1)[0]):.5f} "
                        f"ctrl_dq={float(torch.max(torch.abs(ctrl - q))):.5f} "
                        f"sim_dq={float(torch.max(torch.abs(q - prev))):.5f} "
                        f"finite={bool(torch.isfinite(obs['policy']).all())}",
                        flush=True,
                    )
                prev = q.clone()
        finally:
            env.close()


if __name__ == "__main__":
    main()
