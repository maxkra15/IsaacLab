# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime test: SWEEP a box through the gripper finger and watch for contacts.

Avoids guessing the finger mesh extent: moves the kinematic floor box vertically through the right
finger over many steps, so at some point it MUST partially overlap (entering/exiting) regardless of the
exact mesh size. Reads the live MJWarp-entry contacts each step and reports the max (floor, finger) count
seen. The box position in collision space is verified to track the commanded sweep.
"""

from __future__ import annotations

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True, "device": "cuda:0"})
simulation_app = app_launcher.app

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Isaac-Waterhose-Coupled-v0"


def main():
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    env_cfg.scene.fridge_floor.spawn.size = (0.25, 0.25, 0.12)  # thin-ish; faces cross the finger as it sweeps

    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()

    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics.coupled_manager import NewtonCoupledManager

    b = NewtonManager._builder
    shape_label = [str(x) for x in b.shape_label]
    body_label = [str(x) for x in b.body_label]
    floor_ids = {i for i, l in enumerate(shape_label) if "FridgeFloor" in l}
    finger_ids = {i for i, l in enumerate(shape_label) if ("EE_FINGER" in l or "finger" in l)}
    floor_body = next(i for i, l in enumerate(body_label) if "FridgeFloor" in l)

    robot = env.scene["robot"]
    floor = env.scene["fridge_floor"]
    fidx = [i for i, n in enumerate(robot.data.body_names) if n.endswith("right_gripper_leftfinger")][0]

    contacts = NewtonCoupledManager.get_entry_solver("mjc")._contacts

    actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    for _ in range(5):
        env.step(actions)

    finger_w = robot.data.body_pos_w[0, fidx].clone()
    print(f"finger world = {finger_w.tolist()}")

    max_ff = 0
    first = None
    pos_ok = True
    n = 50
    for step in range(n):
        z = finger_w[2].item() - 0.25 + step * (0.5 / n)  # sweep box center z from finger_z-0.25 to +0.25
        pose = torch.zeros((env.num_envs, 7), device=env.device)
        pose[0, 0] = finger_w[0]
        pose[0, 1] = finger_w[1]
        pose[0, 2] = z
        pose[0, 6] = 1.0
        floor.write_root_pose_to_sim(pose)
        env.step(actions)

        cz = float(NewtonManager._state_0.body_q.numpy()[floor_body][2])
        if abs(cz - z) > 0.03:
            pos_ok = False
        cnt = int(contacts.rigid_contact_count.numpy()[0])
        s0 = contacts.rigid_contact_shape0.numpy()[:cnt]
        s1 = contacts.rigid_contact_shape1.numpy()[:cnt]
        ff = sum(
            1
            for j in range(cnt)
            if (s0[j] in floor_ids and s1[j] in finger_ids) or (s1[j] in floor_ids and s0[j] in finger_ids)
        )
        if ff:
            if first is None:
                first = step
            print(f"  step {step}: box_z={z:.3f} finger_z={finger_w[2].item():.3f} total={cnt} floor<->finger={ff}")
        max_ff = max(max_ff, ff)

    print("\n================ FLOOR CONTACT SWEEP TEST ================")
    print(f"box tracked commanded sweep in collision space: {pos_ok}")
    print(f"floor<->finger contacts over sweep: max_per_step={max_ff} first_step={first}")
    if first is not None:
        print("RESULT: robot<->box contacts ARE generated at runtime (collision is LIVE).")
    elif not pos_ok:
        print("RESULT: INCONCLUSIVE -- box did not track the sweep in collision space.")
    else:
        print("RESULT: box swept fully through the finger but produced ZERO contacts -- collision NOT happening.")
    print("================ END ================\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
