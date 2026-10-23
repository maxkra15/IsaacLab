# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an unchanged RSL actor in real MPM, retaining first-cohort outcomes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.mpm_world_model.collect_franka_pour import argument_parser, run, validate_arguments


def main() -> None:
    parser = argument_parser(evaluation=True)
    args = parser.parse_args()
    validate_arguments(parser, args)
    if args.mode != "mpm":
        parser.error("Real-MPM evaluation requires --mode mpm.")
    if args.perturb_fraction or args.perturb_std:
        parser.error("Policy evaluation requires zero action perturbation.")
    if args.train_only or args.fine_tune_iterations:
        parser.error("Use collect_franka_pour --train-only for fine-tuning, then evaluate its saved checkpoint here.")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
