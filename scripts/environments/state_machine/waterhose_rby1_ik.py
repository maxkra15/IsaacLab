# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted IK demo for the RBY1 waterhose task.

.. code-block:: bash

    WATERHOSE_ASSETS_DIR=/path/to/waterhose/assets \
      ./isaaclab.sh -p scripts/environments/state_machine/waterhose_rby1_ik.py \
      --task Isaac-Waterhose-Coupled-v0 --num_envs 1
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run a scripted IK demo for the RBY1 waterhose task.")
parser.add_argument("--task", type=str, default="Isaac-Waterhose-Coupled-v0", help="Task name.")
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
from isaaclab_tasks.contrib.waterhose.scripted_state_machine import WaterhoseDemoState  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


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
