# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Materialize or verify the pinned Franka RJ45 external-asset closure.

Only an existing local source tree is accepted.  This tool never downloads an
asset or resolves a remote URL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab_tasks.contrib.franka_rj45_insertion.asset_provenance import (
    FRANKA_RJ45_ASSET_CLOSURE_FILE_COUNT,
    FRANKA_RJ45_ASSET_CLOSURE_TOTAL_SIZE,
    FRANKA_RJ45_ASSET_CLOSURE_TREE_SHA256,
    FrankaRJ45AssetClosure,
    materialize_franka_rj45_asset_closure,
    verify_franka_rj45_asset_closure,
)


def _summary(closure: FrankaRJ45AssetClosure) -> dict[str, object]:
    return {
        "root": str(closure.root),
        "tree_sha256": closure.tree_sha256,
        "file_count": FRANKA_RJ45_ASSET_CLOSURE_FILE_COUNT,
        "total_size": FRANKA_RJ45_ASSET_CLOSURE_TOTAL_SIZE,
        "franka_usd_path": str(closure.franka_usd_path),
        "seattle_table_usd_path": str(closure.seattle_table_usd_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser(
        "materialize",
        help="Copy the exact closure from an existing local source tree into a SHA-addressed cache.",
    )
    materialize.add_argument("--source-tree", type=Path, required=True)
    materialize.add_argument("--cache-root", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="Verify an existing exact closure without modifying it.")
    verify.add_argument("--root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "materialize":
        closure = materialize_franka_rj45_asset_closure(args.source_tree, args.cache_root)
    else:
        closure = verify_franka_rj45_asset_closure(args.root)
    if closure.tree_sha256 != FRANKA_RJ45_ASSET_CLOSURE_TREE_SHA256:
        raise RuntimeError("Verified closure returned the wrong pinned tree digest.")
    print(json.dumps(_summary(closure), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
