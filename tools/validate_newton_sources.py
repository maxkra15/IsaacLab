# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the exact local Newton and Warp revisions loaded by Isaac Lab."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import newton
import warp as wp


def _source_root(variable: str, package: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(f"{variable} must name the reviewed {package} checkout.")
    root = Path(value).expanduser().resolve()
    if not (root / package / "__init__.py").is_file():
        raise RuntimeError(f"{variable} does not contain {package}/__init__.py: {root}")
    return root


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _assert_clean_checkout(root: Path, package: str) -> None:
    result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        raise RuntimeError(f"The reviewed {package} checkout is dirty; exact revision validation is impossible.")


def _assert_equal(actual: object, expected: object, description: str) -> None:
    if actual != expected:
        raise RuntimeError(f"Unexpected {description}: got {actual!s}, expected {expected!s}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--newton_revision", required=True, help="Expected immutable Newton Git revision.")
    parser.add_argument("--warp_revision", required=True, help="Expected immutable Warp Git revision.")
    args = parser.parse_args()

    newton_root = _source_root("NEWTON_SOURCE_DIR", "newton")
    warp_root = _source_root("WARP_SOURCE_DIR", "warp")
    _assert_clean_checkout(newton_root, "Newton")
    _assert_clean_checkout(warp_root, "Warp")
    _assert_equal(_git_revision(newton_root), args.newton_revision, "Newton Git revision")
    _assert_equal(_git_revision(warp_root), args.warp_revision, "Warp Git revision")

    wp.init()
    newton_file = Path(newton.__file__).resolve()
    warp_file = Path(wp.__file__).resolve()
    warp_native = Path(wp._src.context.runtime.core._name).resolve()
    _assert_equal(newton_file, newton_root / "newton/__init__.py", "Newton import path")
    _assert_equal(warp_file, warp_root / "warp/__init__.py", "Warp import path")
    _assert_equal(warp_native.parent, warp_root / "warp/bin", "Warp native-library directory")

    print(f"Newton: {newton_file} ({newton.__version__}, {args.newton_revision})")
    print(f"Warp: {warp_file} ({wp.__version__}, {args.warp_revision})")
    print(f"Warp native: {warp_native}")


if __name__ == "__main__":
    main()
