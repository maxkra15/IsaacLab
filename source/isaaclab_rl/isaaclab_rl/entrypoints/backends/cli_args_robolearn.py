# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Argument and checkpoint helpers for the optional RoboLearn backend."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.utils.assets import retrieve_file_path

from isaaclab_tasks.utils import get_checkpoint_path

from ..common import CHECKPOINT_SELECTORS, normalize_task_name, resolve_checkpoint_selector

CHECKPOINT_PATTERN = r"model_\d+\.json"


def add_robolearn_args(parser: argparse.ArgumentParser) -> None:
    """Add learning algorithm and checkpoint selectors."""
    parser.add_argument("--algorithm", choices=["flashsac", "warp_ppo"], default=None)
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint JSON path, run directory, latest, or best."
    )


def select_agent(args_cli: argparse.Namespace) -> None:
    """Select the algorithm's registered config unless an explicit agent was requested."""
    if args_cli.agent is None:
        args_cli.agent = (
            f"robolearn_{args_cli.algorithm}_cfg_entry_point"
            if args_cli.algorithm is not None
            else "robolearn_cfg_entry_point"
        )


def resolve_checkpoint(args_cli: argparse.Namespace, agent_cfg: object, log_root_path: str) -> str:
    """Resolve a compatible run manifest or explicit RoboLearn checkpoint."""
    selector = args_cli.checkpoint or "latest"
    if selector in CHECKPOINT_SELECTORS:
        return resolve_checkpoint_selector(
            log_root_path,
            selector,
            library="robolearn",
            task=normalize_task_name(args_cli.task),
            checkpoint_pattern=CHECKPOINT_PATTERN,
            metadata={"algorithm": agent_cfg.algorithm},
        )
    if os.path.isdir(selector):
        return get_checkpoint_path(os.path.dirname(selector), os.path.basename(selector), CHECKPOINT_PATTERN)
    return retrieve_file_path(selector)


def restore_agent_config(agent_cfg: object, args_cli: argparse.Namespace, checkpoint_path: str) -> None:
    """Restore the policy and action contract before constructing a runner."""
    checkpoint = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    saved = checkpoint["config"]
    if args_cli.algorithm is not None and args_cli.algorithm != saved["algorithm"]:
        raise ValueError("The requested algorithm does not match the checkpoint.")
    # Algorithm keyword dictionaries are separate schemas, so restore rather than merge them.
    agent_cfg.algorithm_cfg = saved["algorithm_cfg"]
    agent_cfg.from_dict(saved)
