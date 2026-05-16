# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a scripted successful demonstration for the RBY1 Newton waterhose task."""

from __future__ import annotations

import argparse
import os
import sys

import gymnasium as gym
import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

parser = argparse.ArgumentParser(description="Record scripted RBY1 waterhose demonstrations.")
parser.add_argument("--task", type=str, default="Isaac-Waterhose-RBY1DF-IK-Rel-v0", help="Waterhose task id.")
parser.add_argument("--dataset_file", type=str, default="./datasets/waterhose_scripted.hdf5")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=1800)
parser.add_argument("--num_success_steps", type=int, default=5)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args


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
    if "kit" in visualizer_types:
        os.environ["ISAACLAB_WATERHOSE_DEFER_NEWTON_IMPORT"] = "1"
        return

    from isaaclab_tasks.manager_based.manipulation.waterhose import waterhose_core as core

    core.import_newton_dependencies()


def main():
    _prepare_waterhose_visualizer_imports()

    env_cfg, _ = resolve_task_config(args_cli.task, "")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        output_dir = os.path.dirname(args_cli.dataset_file) or "."
        output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
        os.makedirs(output_dir, exist_ok=True)
        env_cfg.recorders = ActionStateRecorderManagerCfg()
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()
        for _ in range(args_cli.max_steps):
            with torch.inference_mode():
                _, _, terminated, _, _ = env.step(env.scripted_action())
            if bool(torch.any(terminated)):
                break
        env.close()
        print(f"[INFO]: Scripted waterhose demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    main()
