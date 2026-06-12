#!/usr/bin/env bash
# Dev helper (untracked): run IsaacLab-coupling code against THIS worktree's source.
# The shared venv (symlinked to IsaacLab-ik) editable-installs isaaclab_tasks/isaaclab_newton
# from the IK worktree, which lacks the coupled-solver/MPM API. This PYTHONPATH shim shadows
# those with the coupling worktree's source (setuptools PathFinder runs before the editable
# meta-path finder), non-destructively. Usage: ./scoop_run.sh -p scripts/demos/foo.py --headless
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source"
# Also put the venv's site-packages AHEAD of Isaac Sim's pip_prebundle (which
# setup_python_env.sh appends to PYTHONPATH). Otherwise early imports (torch ->
# typing_extensions, numpy, ...) cache the OLDER prebundle copies in sys.modules
# before isaaclab's _deprioritize_prebundle_paths() can demote the paths, e.g.
# pydantic_core needing typing_extensions.Sentinel (wandb) breaks at train time.
VENV_SITE="$("$HERE/env_isaaclab/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export PYTHONPATH="$SRC/isaaclab_newton:$SRC/isaaclab_tasks:$SRC/isaaclab_tasks_experimental:$SRC/isaaclab_assets:$SRC/isaaclab_rl:$SRC/isaaclab_experimental:$SRC/isaaclab:$VENV_SITE:${PYTHONPATH:-}"
exec "$HERE/isaaclab.sh" "$@"
