# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


def test_waterhose_asset_resolver_prefers_setup_extracted_bundle(monkeypatch, tmp_path):
    from isaaclab_tasks.contrib.waterhose import waterhose_env_cfg

    monkeypatch.delenv("WATERHOSE_ASSETS_DIR", raising=False)
    module_dir = tmp_path / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "contrib" / "waterhose"
    package_assets_dir = module_dir / "assets"
    setup_assets_dir = tmp_path / "source" / "isaaclab_assets" / "data" / "WaterhoseDemo"
    package_assets_dir.mkdir(parents=True)
    setup_assets_dir.mkdir(parents=True)

    assert waterhose_env_cfg._resolve_waterhose_assets_dir(str(module_dir)) == str(setup_assets_dir)


def test_waterhose_asset_resolver_keeps_env_override(monkeypatch, tmp_path):
    from isaaclab_tasks.contrib.waterhose import waterhose_env_cfg

    override_dir = tmp_path / "custom_assets"
    monkeypatch.setenv("WATERHOSE_ASSETS_DIR", str(override_dir))

    assert waterhose_env_cfg._resolve_waterhose_assets_dir(str(tmp_path)) == str(override_dir)
