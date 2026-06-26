# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_setup_script_creates_isaacsim_conda_env_shim_for_public_build(tmp_path):
    repo_root = Path(__file__).resolve().parents[4]
    setup_script = repo_root / "waterhose-setup.sh"
    install_root = tmp_path / "install"
    waterhose_repo = install_root / "IsaacLab-waterhose"
    isaacsim_dir = install_root / "IsaacSim"
    isaacsim_build_dir = isaacsim_dir / "_build" / "linux-x86_64" / "release"
    wheeled_robots_config = isaacsim_build_dir / "exts" / "isaacsim.robot.wheeled_robots" / "config"

    waterhose_repo.mkdir(parents=True)
    wheeled_robots_config.mkdir(parents=True)
    (isaacsim_build_dir / "python.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (isaacsim_build_dir / "setup_python_env.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (wheeled_robots_config / "extension.toml").write_text("[package]\n", encoding="utf-8")

    script = f"""
        set -euo pipefail
        WATERHOSE_SETUP_SKIP_MAIN=1 source {setup_script}
        link_isaacsim_build {waterhose_repo} {isaacsim_dir}
        test -L {waterhose_repo / "_isaac_sim"}
        test -f {isaacsim_build_dir / "setup_conda_env.sh"}
        grep -q 'setup_python_env.sh' {isaacsim_build_dir / "setup_conda_env.sh"}
        grep -q 'CARB_APP_PATH' {isaacsim_build_dir / "setup_conda_env.sh"}
    """
    result = subprocess.run(["bash", "-lc", script], env=os.environ.copy(), text=True, capture_output=True)

    assert result.returncode == 0, result.stdout + result.stderr
