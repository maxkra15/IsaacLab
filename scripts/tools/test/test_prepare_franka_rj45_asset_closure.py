# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for the offline Franka RJ45 asset-closure CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from isaaclab_tasks.contrib.franka_rj45_insertion.asset_provenance import (
    FRANKA_RJ45_ASSET_CLOSURE_TREE_SHA256,
    FrankaRJ45AssetClosure,
)

from scripts.tools import prepare_franka_rj45_asset_closure as cli


@pytest.mark.parametrize("command", ("verify", "materialize"))
def test_cli_emits_the_verified_content_addressed_paths(
    command: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    closure_root = tmp_path / "cache/sha256" / FRANKA_RJ45_ASSET_CLOSURE_TREE_SHA256
    closure = FrankaRJ45AssetClosure(
        root=closure_root,
        franka_usd_path=closure_root / "franka.usda",
        seattle_table_usd_path=closure_root / "table.usd",
        tree_sha256=FRANKA_RJ45_ASSET_CLOSURE_TREE_SHA256,
    )
    observed: list[tuple[Path, ...]] = []
    if command == "verify":
        monkeypatch.setattr(
            cli, "verify_franka_rj45_asset_closure", lambda root: (observed.append((root,)), closure)[1]
        )
        argv = ["verify", "--root", str(closure_root)]
    else:
        monkeypatch.setattr(
            cli,
            "materialize_franka_rj45_asset_closure",
            lambda source, cache: (observed.append((source, cache)), closure)[1],
        )
        argv = ["materialize", "--source-tree", str(tmp_path / "source"), "--cache-root", str(tmp_path / "cache")]

    assert cli.main(argv) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["root"] == str(closure_root)
    assert output["tree_sha256"] == FRANKA_RJ45_ASSET_CLOSURE_TREE_SHA256
    assert output["file_count"] == 19
    assert output["total_size"] == 83_718_325
    assert observed


def test_cli_rejects_a_result_with_the_wrong_tree_digest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    closure = FrankaRJ45AssetClosure(
        root=tmp_path,
        franka_usd_path=tmp_path / "franka.usda",
        seattle_table_usd_path=tmp_path / "table.usd",
        tree_sha256="0" * 64,
    )
    monkeypatch.setattr(cli, "verify_franka_rj45_asset_closure", lambda _root: closure)

    with pytest.raises(RuntimeError, match="wrong pinned tree digest"):
        cli.main(["verify", "--root", str(tmp_path)])
