#!/usr/bin/env bash

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Exit on error.
set -e

# Get repo directory.
export ISAACLAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Ignore stale environment variables instead of failing on a Python executable
# that no longer exists.
if [ -n "${VIRTUAL_ENV:-}" ] && [ ! -x "$VIRTUAL_ENV/bin/python" ]; then
    echo "[WARNING] Ignoring stale VIRTUAL_ENV=$VIRTUAL_ENV (no executable Python found)." >&2
    unset VIRTUAL_ENV
fi
if [ -n "${CONDA_PREFIX:-}" ] && [ ! -x "$CONDA_PREFIX/bin/python" ]; then
    echo "[WARNING] Ignoring stale CONDA_PREFIX=$CONDA_PREFIX (no executable Python found)." >&2
    unset CONDA_PREFIX
fi

# Select one Python environment. Active environments take precedence, followed
# by the conventional repository-local environments.
python_env=""
python_env_kind=""
if [ -n "${VIRTUAL_ENV:-}" ]; then
    python_env="$VIRTUAL_ENV"
    python_env_kind="venv"
elif [ -n "${CONDA_PREFIX:-}" ]; then
    python_env="$CONDA_PREFIX"
    python_env_kind="conda"
elif [ -x "$ISAACLAB_PATH/env_isaaclab/bin/python" ]; then
    python_env="$ISAACLAB_PATH/env_isaaclab"
    python_env_kind="venv"
elif [ -x "$ISAACLAB_PATH/.venv/bin/python" ]; then
    python_env="$ISAACLAB_PATH/.venv"
    python_env_kind="venv"
fi

selected_python=""
if [ -n "$python_env" ]; then
    selected_python="$python_env/bin/python"
    export PATH="$python_env/bin:$PATH"
    if [ "$python_env_kind" = "venv" ]; then
        export VIRTUAL_ENV="$python_env"
    fi
fi

# Put this checkout's packages first, then the selected environment's
# site-packages. Isaac Sim's source-build launcher appends its bundled package
# paths later, preserving this ordering.
python_paths=""
for source_path in "$ISAACLAB_PATH"/source/*; do
    if [ -d "$source_path" ]; then
        python_paths="${python_paths:+$python_paths:}$source_path"
    fi
done
if [ -n "$selected_python" ]; then
    python_site_packages="$($selected_python -I -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    python_paths="${python_paths:+$python_paths:}$python_site_packages"
fi
if [ -n "${PYTHONPATH:-}" ]; then
    python_paths="${python_paths:+$python_paths:}$PYTHONPATH"
fi
export PYTHONPATH="$python_paths"

# Let Kit associate direct wrapper launches with the Isaac Sim desktop icon.
export RESOURCE_NAME="${RESOURCE_NAME:-IsaacSim}"

# If a local Isaac Sim binary is present, source its env setup so that
# PYTHONPATH/PATH/EXP_PATH are correct without depending on a conda
# activate.d hook (those don't fire reliably under e.g. `conda run`).
if [ -d "$ISAACLAB_PATH/_isaac_sim" ]; then
    if [ -f "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh" ]; then
        # shellcheck disable=SC1091
        . "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh" >/dev/null 2>&1 || true
    elif [ -x "$ISAACLAB_PATH/_isaac_sim/python.sh" ] && [ -f "$ISAACLAB_PATH/_isaac_sim/setup_python_env.sh" ]; then
        # A source-built Isaac Sim configures its complete runtime from
        # python.sh. PYTHONEXE lets it retain the selected Isaac Lab venv.
        if [ -n "$selected_python" ]; then
            export PYTHONEXE="$selected_python"
        else
            unset PYTHONEXE
        fi
        python_exe="$ISAACLAB_PATH/_isaac_sim/python.sh"
    else
        echo "[WARNING] _isaac_sim is present but has no recognized environment setup." >&2
    fi
fi

if [ -z "${python_exe:-}" ]; then
    if [ -n "$selected_python" ]; then
        python_exe="$selected_python"
    elif [ -x "$ISAACLAB_PATH/_isaac_sim/python.sh" ]; then
        python_exe="$ISAACLAB_PATH/_isaac_sim/python.sh"
    else
        # Fallback to system python.
        python_exe="python3"
    fi
fi

# Execute CLI.
exec "$python_exe" -c "from isaaclab.cli import cli; cli()" "$@"
