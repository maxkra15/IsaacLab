# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

import argparse
import os
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
add_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# pass remaining args to Hydra
sys.argv = [sys.argv[0]] + hydra_args

# PLACEHOLDER: Extension template (do not remove this comment)


def _get_visualizer_types() -> set[str]:
    """Return visualizer types requested through the launcher CLI."""
    visualizers = getattr(args_cli, "visualizer", None)
    if not visualizers:
        return set()
    if isinstance(visualizers, str):
        visualizers = [token.strip() for token in visualizers.split(",")]
    return {str(visualizer).strip().lower() for visualizer in visualizers if str(visualizer).strip()}


def _prepare_waterhose_visualizer_imports() -> None:
    """Configure waterhose imports for the selected visualizer startup path."""
    visualizer_types = _get_visualizer_types()
    if visualizer_types & {"kit", "newton"}:
        os.environ.setdefault("DISPLAY", ":1")
    if args_cli.task is None or "waterhose" not in args_cli.task.lower() or "kit" not in visualizer_types:
        return
    os.environ["ISAACLAB_WATERHOSE_DEFER_NEWTON_IMPORT"] = "1"


def main():
    """Random actions agent with Isaac Lab environment."""

    torch.manual_seed(42)
    _prepare_waterhose_visualizer_imports()

    # parse configuration via Hydra (supports preset selection, e.g. env.sim.physics=newton_mjwarp)
    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        # override with CLI arguments
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg)

        # print info (this is vectorized environment)
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        # reset environment
        env.reset()
        # simulate environment
        sim = env.unwrapped.sim
        while True:
            if sim.visualizers:
                # visualizer mode: run until the visualizer window is closed
                if not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                    break
            # run everything in inference mode
            with torch.inference_mode():
                # sample actions from -1 to 1
                actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                # apply actions
                env.step(actions)

        # close the simulator
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
