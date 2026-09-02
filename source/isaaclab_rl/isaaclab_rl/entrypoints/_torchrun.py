# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight torchrun identity helpers for reinforcement-learning entrypoints."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from datetime import datetime


def _distributed_run_directory_name(environ: Mapping[str, str]) -> str:
    """Return a path-safe name shared by every worker in one torchrun launch."""
    run_id = environ.get("TORCHELASTIC_RUN_ID", "").strip()
    if not run_id:
        raise RuntimeError(
            "Distributed RSL-RL training requires TORCHELASTIC_RUN_ID. "
            "Launch it with torchrun or the Isaac Lab multi-GPU launcher."
        )

    # Keep the rendezvous identity recognizable while bounding the path component. The digest
    # distinguishes identities that normalize to the same text (and makes truncation collision-safe).
    readable_id = re.sub(r"[^A-Za-z0-9._-]+", "-", run_id).strip("._-")[:48] or "run"
    identity_digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:12]
    return f"torchrun_{readable_id}_{identity_digest}"


def resolve_log_dir(
    log_root_path: str,
    run_name: str,
    *,
    distributed: bool,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve a run directory without rank-local clocks for distributed launches."""
    if distributed:
        directory_name = _distributed_run_directory_name(os.environ if environ is None else environ)
    else:
        directory_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if run_name:
        directory_name += f"_{run_name}"
    return os.path.join(log_root_path, directory_name)


def should_write_run_metadata(distributed: bool, environ: Mapping[str, str] | None = None) -> bool:
    """Return whether this process owns shared run metadata writes."""
    if not distributed:
        return True
    process_environ = os.environ if environ is None else environ
    return int(process_environ.get("RANK", "0")) == 0
