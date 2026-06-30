#!/usr/bin/env bash

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Exit on error.
set -e

# Get repo directory.
export ISAACLAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Find python to run CLI.
if [ -n "$VIRTUAL_ENV" ]; then
    python_exe="$VIRTUAL_ENV/bin/python"
elif [ -n "$CONDA_PREFIX" ]; then
    python_exe="$CONDA_PREFIX/bin/python"
elif [ -f "$ISAACLAB_PATH/env_isaaclab/bin/python" ]; then
    python_exe="$ISAACLAB_PATH/env_isaaclab/bin/python"
elif [ -f "$ISAACLAB_PATH/_isaac_sim/python.sh" ]; then
    python_exe="$ISAACLAB_PATH/_isaac_sim/python.sh"
else
    # Fallback to system python
    python_exe="python3"
fi

# Add repo-local source packages to PYTHONPATH so this checkout does not
# accidentally import matching packages from a shared virtual environment.
source_roots="$ISAACLAB_PATH/source/isaaclab"
for source_dir in "$ISAACLAB_PATH"/source/*; do
    if [ "$source_dir" = "$ISAACLAB_PATH/source/isaaclab" ]; then
        continue
    fi
    if [ -d "$source_dir" ]; then
        source_roots="$source_roots:$source_dir"
    fi
done

# Local physics-source overrides are explicit so an adjacent checkout cannot
# silently replace the versions installed in the active environment.
newton_source_dir="${NEWTON_SOURCE_DIR:-}"
if [ -n "$newton_source_dir" ]; then
    if [ ! -f "$newton_source_dir/newton/__init__.py" ]; then
        echo "[ERROR] NEWTON_SOURCE_DIR does not contain newton/__init__.py: $newton_source_dir" >&2
        exit 1
    fi
    source_roots="$newton_source_dir:$source_roots"
fi

warp_source_dir="${WARP_SOURCE_DIR:-}"
if [ -n "$warp_source_dir" ]; then
    if [ ! -f "$warp_source_dir/warp/__init__.py" ]; then
        echo "[ERROR] WARP_SOURCE_DIR does not contain warp/__init__.py: $warp_source_dir" >&2
        exit 1
    fi
    source_roots="$warp_source_dir:$source_roots"
fi
export PYTHONPATH="$source_roots:${PYTHONPATH:-}"

# If a local Isaac Sim binary is present, source its env setup so that
# PYTHONPATH/PATH/EXP_PATH are correct without depending on a conda
# activate.d hook (those don't fire reliably under e.g. `conda run`).
if [ -d "$ISAACLAB_PATH/_isaac_sim" ]; then
    if [ -f "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh" ]; then
        # shellcheck disable=SC1091
        . "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh" >/dev/null 2>&1 || true
    else
        echo "[WARNING] _isaac_sim is present but _isaac_sim/setup_conda_env.sh is missing; Isaac Sim env vars not exported." >&2
        echo "[WARNING] Re-extract the Isaac Sim binary zip if you intend to use the bundled binary." >&2
    fi
fi

# Execute CLI.
exec "$python_exe" -c "from isaaclab.cli import cli; cli()" "$@"
