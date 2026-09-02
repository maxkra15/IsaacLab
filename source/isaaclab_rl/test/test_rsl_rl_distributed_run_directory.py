# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure tests for RSL-RL distributed run-directory ownership."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from isaaclab_rl.entrypoints import _torchrun


def test_distributed_ranks_resolve_the_same_run_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    """A torchrun rendezvous identity, unlike a rank-local clock, is common to every worker."""

    class _ForbiddenRankLocalClock:
        @staticmethod
        def now() -> datetime:
            raise AssertionError("distributed run naming must not read a rank-local clock")

    monkeypatch.setattr(_torchrun, "datetime", _ForbiddenRankLocalClock)
    rank_zero_env = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "TORCHELASTIC_RUN_ID": "workflow-42",
        "TORCHELASTIC_ERROR_FILE": "/tmp/attempt/0/error.json",
    }
    other_node_env = {
        "RANK": "8",
        "LOCAL_RANK": "0",
        "TORCHELASTIC_RUN_ID": "workflow-42",
        "TORCHELASTIC_ERROR_FILE": "/tmp/attempt/8/error.json",
    }

    rank_zero_dir = _torchrun.resolve_log_dir("/logs/experiment", "trial", distributed=True, environ=rank_zero_env)
    other_node_dir = _torchrun.resolve_log_dir("/logs/experiment", "trial", distributed=True, environ=other_node_env)

    assert rank_zero_dir == other_node_dir
    assert Path(rank_zero_dir).name.startswith("torchrun_workflow-42_")
    assert Path(rank_zero_dir).name.endswith("_trial")


def test_distributed_run_identity_is_a_safe_bounded_path_component() -> None:
    """An arbitrary rendezvous id cannot escape the experiment directory or exceed filename limits."""
    log_dir = _torchrun.resolve_log_dir(
        "/logs/experiment",
        "",
        distributed=True,
        environ={"TORCHELASTIC_RUN_ID": "../../job name/" + "x" * 300},
    )

    assert Path(log_dir).parent == Path("/logs/experiment")
    assert len(Path(log_dir).name) <= len("torchrun_") + 48 + 1 + 12


def test_distributed_run_requires_torchrun_identity() -> None:
    """A distributed launch fails clearly instead of silently returning rank-local paths."""
    with pytest.raises(RuntimeError, match="TORCHELASTIC_RUN_ID"):
        _torchrun.resolve_log_dir("/logs/experiment", "", distributed=True, environ={"RANK": "0"})


def test_direct_torchrun_static_default_remains_supported() -> None:
    """Direct torchrun commands using its ``none`` default still converge on one directory."""
    log_dir = _torchrun.resolve_log_dir(
        "/logs/experiment", "trial", distributed=True, environ={"TORCHELASTIC_RUN_ID": "none"}
    )

    assert Path(log_dir).name.startswith("torchrun_none_")
    assert Path(log_dir).name.endswith("_trial")


def test_non_distributed_run_preserves_timestamp_naming(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single-process runs keep their historical second-resolution timestamp and suffix."""

    class _FixedDateTime:
        @staticmethod
        def now() -> datetime:
            return datetime(2026, 8, 14, 12, 34, 56)

    monkeypatch.setattr(_torchrun, "datetime", _FixedDateTime)

    assert _torchrun.resolve_log_dir("/logs/experiment", "trial", distributed=False) == (
        "/logs/experiment/2026-08-14_12-34-56_trial"
    )


@pytest.mark.parametrize(
    ("distributed", "environ", "expected"),
    [
        (False, {"RANK": "9"}, True),
        (True, {"RANK": "0", "LOCAL_RANK": "0"}, True),
        (True, {"RANK": "1", "LOCAL_RANK": "1"}, False),
        (True, {"RANK": "8", "LOCAL_RANK": "0"}, False),
    ],
)
def test_only_global_rank_zero_owns_run_metadata(distributed: bool, environ: dict[str, str], expected: bool) -> None:
    """Local rank zero on a later node must not race global rank zero's metadata writes."""
    assert _torchrun.should_write_run_metadata(distributed, environ) is expected
