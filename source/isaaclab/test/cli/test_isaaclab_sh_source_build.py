# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for launching a source-built Isaac Sim through ``isaaclab.sh``."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

_ISAACLAB_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class _LaunchResult:
    """Captured values from a fake source-build launch."""

    result: subprocess.CompletedProcess[str]
    repo: Path
    repo_python: Path
    stale_venv: Path
    purelib: Path
    existing_kit_path: Path
    record: dict[str, str]


def _write_executable(path: Path, contents: str) -> None:
    """Write an executable test helper."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _run_fake_source_build(tmp_path: Path) -> _LaunchResult:
    """Run the repository launcher against an isolated fake source build."""
    repo = tmp_path / "IsaacLab"
    repo.mkdir()
    shutil.copy2(_ISAACLAB_ROOT / "isaaclab.sh", repo / "isaaclab.sh")
    (repo / "source" / "isaaclab").mkdir(parents=True)

    stale_venv = tmp_path / "removed-env"
    repo_python = repo / "env_isaaclab" / "bin" / "python"
    purelib = repo / "env_isaaclab" / "lib" / "python3.12" / "site-packages"
    existing_kit_path = tmp_path / "existing-kit" / "pip_prebundle"
    existing_kit_path.mkdir(parents=True)
    record_path = tmp_path / "source-python-record.txt"
    direct_python_record = tmp_path / "direct-python-record.txt"

    _write_executable(
        repo_python,
        """#!/usr/bin/env bash
set -eu
if [[ "$*" == *"sysconfig"* ]]; then
    printf '%s\n' "$FAKE_PURELIB"
    exit 0
fi
printf 'invoked\n' > "$DIRECT_PYTHON_RECORD"
exit 73
""",
    )

    setup_python_env = repo / "_isaac_sim" / "setup_python_env.sh"
    setup_python_env.parent.mkdir(parents=True)
    setup_python_env.write_text(
        'export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}${SCRIPT_DIR}/kit/pip_prebundle"\n',
        encoding="utf-8",
    )

    source_python = repo / "_isaac_sim" / "python.sh"
    _write_executable(
        source_python,
        """#!/usr/bin/env bash
set -eu
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/setup_python_env.sh"
{
    printf 'launcher=%s\n' "$0"
    printf 'pythonexe=%s\n' "${PYTHONEXE:-}"
    printf 'pythonpath=%s\n' "${PYTHONPATH:-}"
    printf 'arguments=%s\n' "$*"
} > "$LAUNCH_RECORD"
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "DIRECT_PYTHON_RECORD": str(direct_python_record),
            "FAKE_PURELIB": str(purelib),
            "LAUNCH_RECORD": str(record_path),
            "PYTHONPATH": str(existing_kit_path),
            "VIRTUAL_ENV": str(stale_venv),
        }
    )
    for name in ("CONDA_PREFIX", "PYTHONEXE"):
        env.pop(name, None)

    result = subprocess.run(
        ["bash", str(repo / "isaaclab.sh"), "-p", "-c", "pass"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    record = {}
    if record_path.exists():
        record = dict(line.split("=", maxsplit=1) for line in record_path.read_text(encoding="utf-8").splitlines())

    return _LaunchResult(
        result=result,
        repo=repo,
        repo_python=repo_python,
        stale_venv=stale_venv,
        purelib=purelib,
        existing_kit_path=existing_kit_path,
        record=record,
    )


def test_stale_virtual_env_uses_repo_env_through_source_python(tmp_path: Path) -> None:
    """A stale active-env marker must fall back to the repo env while retaining source-build setup."""
    launch = _run_fake_source_build(tmp_path)

    assert launch.result.returncode == 0, launch.result.stderr
    assert Path(launch.record["launcher"]).resolve() == (launch.repo / "_isaac_sim" / "python.sh").resolve()
    assert Path(launch.record["pythonexe"]).resolve() == launch.repo_python.resolve()
    assert str(launch.stale_venv) not in launch.record["pythonexe"]


def test_source_python_prefers_checkout_and_env_packages_over_kit(tmp_path: Path) -> None:
    """Checkout sources and the selected env's packages must precede Kit's bundled packages."""
    launch = _run_fake_source_build(tmp_path)

    assert launch.result.returncode == 0, launch.result.stderr
    python_paths = launch.record["pythonpath"].split(os.pathsep)
    local_source = str(launch.repo / "source" / "isaaclab")
    purelib = str(launch.purelib)
    existing_kit_path = str(launch.existing_kit_path)

    assert python_paths.index(local_source) < python_paths.index(purelib)
    assert python_paths.index(purelib) < python_paths.index(existing_kit_path)
