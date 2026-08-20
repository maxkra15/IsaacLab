# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the pinned, fail-closed Franka RJ45 external-asset closure."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest

from isaaclab_tasks.contrib.franka_rj45_insertion import asset_provenance as provenance

_PINNED_FILES = (
    (
        "Isaac/IsaacLab/Robots/FrankaEmika/franka_panda.usda",
        1_732,
        "ab094f718668e6fa24cb2595c3bf9c07a74a7e5ec4bd29e1ed7a975b69d9b701",
    ),
    (
        "Isaac/IsaacLab/Robots/FrankaEmika/payloads/Physics/mujoco.usda",
        25_544,
        "82f363dd872bd8ca88847a76b1daf267fdace4095f6d505f3eb16acf49596f8d",
    ),
    (
        "Isaac/IsaacLab/Robots/FrankaEmika/payloads/Physics/physics.usda",
        29_884,
        "0dc38454f02ea14d9ddd2437995fdc7c4a65634443cacdcc2a04e3de25655e00",
    ),
    (
        "Isaac/IsaacLab/Robots/FrankaEmika/payloads/Physics/physx.usda",
        5_037,
        "d68c2d5ad9dad3e107ee41d477bb828c9410e473ccd20e0ed32df0b7925c32b3",
    ),
    (
        "Isaac/IsaacLab/Robots/FrankaEmika/payloads/base.usda",
        71_251,
        "7121cbfc8489dbac81ce6a781b16f9cc52390f9ff75ca3ef75022d775098bc6d",
    ),
    (
        "Isaac/IsaacLab/Robots/FrankaEmika/payloads/geometries.usd",
        7_097_437,
        "c64f015a69302f5b277040fc54fc7addedab49c47e67d1b79f374414dd4e991b",
    ),
    (
        "Isaac/IsaacLab/Robots/FrankaEmika/payloads/instances.usda",
        80_355,
        "04857fae746db145b454d3c3065b3bd8e04a9b87e35961c9ffc171ede3248cec",
    ),
    (
        "Isaac/IsaacLab/Robots/FrankaEmika/payloads/materials.usda",
        6_716,
        "3b87896947047adc23abafac72ccb113b26a32293645db8e755c51f2abe58b9b",
    ),
    (
        "Isaac/IsaacLab/Robots/FrankaEmika/payloads/robot.usda",
        5_328,
        "060bd8bc6f47b6d57ef918d05629af36763998a275c5102bd4aa4540e0fcf15b",
    ),
    (
        "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableBase_BaseColor.png",
        6_335_308,
        "2cf139f46874df8a133c26eb51405161824ddecb0ce6a094e6477a675656bfe4",
    ),
    (
        "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableBase_Metallic.png",
        119_609,
        "c56bdeeecef908a1a6869af71556ad9c43794d0d292e3b8463aa7b82f3367771",
    ),
    (
        "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableBase_Normal.png",
        34_371_270,
        "bfcbadb3f98c9fe116bc6eb2c040d09196c40ca16347367900f8fd2d3451e10d",
    ),
    (
        "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableBase_Roughness.png",
        7_077_839,
        "35426471bf910ae4ef32d044b360489de8477c97cf1a75ad25341ef2a25a644b",
    ),
    (
        "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableParts_BaseColor.png",
        905_610,
        "5d095ed2e4fe2fa18d35a2677f3c281796fba135544439db8ec32d851144fbc0",
    ),
    (
        "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableParts_Metallic.png",
        414_290,
        "fdcd7dcec52034b60662cccf58c02ae442c0a87d60b099fcf6cd9ccd6cf5535d",
    ),
    (
        "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableParts_Normal.png",
        13_632_148,
        "9413615d7cf9809f639ec9722022c164ad73a14add32a94293fc7c69d97b601c",
    ),
    (
        "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableParts_Roughness.png",
        2_995_685,
        "1cb2096ba97d3519f876b9d0dc84822bdd428cf93f0887a8f75e3bba208c4a35",
    ),
    (
        "Isaac/Props/Mounts/SeattleLabTable/table.usd",
        10_538_698,
        "8e6e2284e0eba868341b5afacf663d4f84005febef4f9c178d916e609d44c114",
    ),
    (
        "Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        4_584,
        "479ed8fd374d8eafdedf3ca3b9d0d5981ffc6d5c5016929347f748be93d5df17",
    ),
)

_FIXTURE_CONTENTS = {
    "assets/root.usda": b"fixture-root",
    "assets/textures/base-color.png": b"fixture-texture",
}


def _fixture_manifest() -> provenance.AssetClosureManifest:
    files = tuple(
        provenance.AssetClosureFile(path, len(contents), hashlib.sha256(contents).hexdigest())
        for path, contents in sorted(_FIXTURE_CONTENTS.items())
    )
    return provenance.AssetClosureManifest(
        format=provenance.FRANKA_RJ45_ASSET_CLOSURE_FORMAT,
        schema_version=provenance.FRANKA_RJ45_ASSET_CLOSURE_SCHEMA_VERSION,
        logical_uri="isaaclab-asset-closure://test/fixture-v1",
        tree_sha256=provenance.asset_closure_tree_digest(files),
        files=files,
    )


def _write_fixture_tree(root: Path) -> None:
    for relative_path, contents in _FIXTURE_CONTENTS.items():
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)


def test_pinned_manifest_is_the_exact_isaac_61_dependency_closure() -> None:
    observed = tuple(
        (entry.relative_path, entry.size, entry.sha256) for entry in provenance.FRANKA_RJ45_ASSET_CLOSURE_FILES
    )

    assert observed == _PINNED_FILES
    assert len(observed) == provenance.FRANKA_RJ45_ASSET_CLOSURE_FILE_COUNT == 19
    assert sum(entry[1] for entry in observed) == provenance.FRANKA_RJ45_ASSET_CLOSURE_TOTAL_SIZE == 83_718_325
    expected_tree_sha256 = "060d50ba665ae9850a8ef19c1b42a2a55579d4f8924cadfc37c1ec83ca726a76"
    assert expected_tree_sha256 == provenance.FRANKA_RJ45_ASSET_CLOSURE_TREE_SHA256
    assert provenance.asset_closure_tree_digest(provenance.FRANKA_RJ45_ASSET_CLOSURE_FILES) == expected_tree_sha256


def test_logical_contract_is_path_independent_and_binds_entrypoints() -> None:
    contract = provenance.franka_rj45_asset_contract()
    serialized = json.dumps(contract, sort_keys=True)

    assert contract["logical_uri"] == "isaaclab-asset-closure://franka-rj45/scene-v1"
    assert contract["tree_sha256"] == provenance.FRANKA_RJ45_ASSET_CLOSURE_TREE_SHA256
    assert contract["file_count"] == 19
    assert contract["total_size"] == 83_718_325
    assert contract["entrypoints"]["franka_robot"]["logical_uri"].startswith("isaac-asset://6.1/")
    assert contract["entrypoints"]["seattle_table"]["relative_path"].endswith("table_instanceable.usd")
    assert contract["runtime_dependencies"] == (
        {"logical_uri": "mdl://OmniPBR.mdl", "provider": "pinned-renderer-runtime-image"},
    )
    assert "/tmp/" not in serialized
    assert "/home/" not in serialized


def test_verify_asset_closure_accepts_only_the_exact_tree(tmp_path: Path) -> None:
    root = tmp_path / "closure"
    _write_fixture_tree(root)

    verified = provenance.verify_asset_closure(root, _fixture_manifest())

    assert verified.root == root
    assert verified.file_count == 2
    assert verified.total_size == sum(map(len, _FIXTURE_CONTENTS.values()))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "missing"),
        ("modified", "digest mismatch"),
        ("extra_file", "unexpected file"),
        ("extra_directory", "unexpected directory"),
    ),
)
def test_verify_asset_closure_rejects_missing_modified_and_extra_entries(
    tmp_path: Path, mutation: str, message: str
) -> None:
    root = tmp_path / "closure"
    _write_fixture_tree(root)
    if mutation == "missing":
        (root / "assets/root.usda").unlink()
    elif mutation == "modified":
        (root / "assets/root.usda").write_bytes(b"changed-root")
    elif mutation == "extra_file":
        (root / "assets/extra.txt").write_text("extra")
    else:
        (root / "assets/extra").mkdir()

    with pytest.raises(provenance.AssetClosureError, match=message):
        provenance.verify_asset_closure(root, _fixture_manifest())


def test_verify_asset_closure_rejects_expected_file_symlink(tmp_path: Path) -> None:
    root = tmp_path / "closure"
    _write_fixture_tree(root)
    external = tmp_path / "external.usda"
    external.write_bytes(_FIXTURE_CONTENTS["assets/root.usda"])
    expected = root / "assets/root.usda"
    expected.unlink()
    expected.symlink_to(external)

    with pytest.raises(provenance.AssetClosureError, match="symbolic link"):
        provenance.verify_asset_closure(root, _fixture_manifest())


def test_verify_asset_closure_rejects_root_and_unexpected_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "closure"
    _write_fixture_tree(root)
    root_link = tmp_path / "closure-link"
    root_link.symlink_to(root, target_is_directory=True)

    with pytest.raises(provenance.AssetClosureError, match="root must not be a symbolic link"):
        provenance.verify_asset_closure(root_link, _fixture_manifest())

    (root / "unexpected-link").symlink_to(root / "assets", target_is_directory=True)
    with pytest.raises(provenance.AssetClosureError, match="symbolic link"):
        provenance.verify_asset_closure(root, _fixture_manifest())


@pytest.mark.parametrize("path", ("../escape", "/absolute", "a\\b", "a//b", "a/./b", "a/../b"))
def test_manifest_rejects_noncanonical_or_traversing_paths(tmp_path: Path, path: str) -> None:
    entry = provenance.AssetClosureFile(path, 0, hashlib.sha256(b"").hexdigest())
    manifest = provenance.AssetClosureManifest(
        format=provenance.FRANKA_RJ45_ASSET_CLOSURE_FORMAT,
        schema_version=provenance.FRANKA_RJ45_ASSET_CLOSURE_SCHEMA_VERSION,
        logical_uri="isaaclab-asset-closure://test/invalid-v1",
        tree_sha256="0" * 64,
        files=(entry,),
    )

    with pytest.raises(provenance.AssetClosureError, match="canonical|relative"):
        provenance.verify_asset_closure(tmp_path, manifest)


def test_configured_root_is_optional_but_must_be_absolute(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_name = provenance.FRANKA_RJ45_ASSET_CLOSURE_ENV

    assert provenance.configured_franka_rj45_asset_closure(environ={}) is None
    with pytest.raises(provenance.AssetClosureError, match="must name"):
        provenance.configured_franka_rj45_asset_closure(required=True, environ={})
    with pytest.raises(provenance.AssetClosureError, match="absolute"):
        provenance.configured_franka_rj45_asset_closure(environ={env_name: "relative/closure"})

    sentinel = object()
    observed: list[Path] = []

    def _verify(root: Path) -> object:
        observed.append(root)
        return sentinel

    monkeypatch.setattr(provenance, "verify_franka_rj45_asset_closure", _verify)
    assert provenance.configured_franka_rj45_asset_closure(environ={env_name: str(tmp_path)}) is sentinel
    assert observed == [tmp_path]


def test_materialize_copies_only_manifest_files_into_the_sha_cache(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_fixture_tree(source)
    (source / "unrelated").mkdir()
    (source / "unrelated/asset.usd").write_text("not in the closure")
    cache = tmp_path / "cache"
    manifest = _fixture_manifest()

    verified = provenance.materialize_asset_closure_from_source_tree(source, cache, manifest)

    expected_root = cache / "sha256" / manifest.tree_sha256
    assert verified.root == expected_root
    assert not (expected_root / "unrelated").exists()
    assert stat.S_IMODE((expected_root / "assets/root.usda").stat().st_mode) == 0o444
    assert provenance.verify_asset_closure(expected_root, manifest) == verified
    assert sorted(path.name for path in (cache / "sha256").iterdir()) == [manifest.tree_sha256]


def test_materialize_reuses_verified_target_without_overwriting_it(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_fixture_tree(source)
    cache = tmp_path / "cache"
    manifest = _fixture_manifest()
    first = provenance.materialize_asset_closure_from_source_tree(source, cache, manifest)
    first_inode = (first.root / "assets/root.usda").stat().st_ino

    second = provenance.materialize_asset_closure_from_source_tree(source, cache, manifest)

    assert second == first
    assert (second.root / "assets/root.usda").stat().st_ino == first_inode


def test_materialize_rejects_modified_source_without_publishing_partial_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_fixture_tree(source)
    (source / "assets/root.usda").write_bytes(b"changed-root")
    cache = tmp_path / "cache"
    manifest = _fixture_manifest()

    with pytest.raises(provenance.AssetClosureError, match="digest mismatch"):
        provenance.materialize_asset_closure_from_source_tree(source, cache, manifest)

    assert list((cache / "sha256").iterdir()) == []


def test_materialize_rejects_invalid_existing_target_and_sha_directory_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_fixture_tree(source)
    manifest = _fixture_manifest()
    cache = tmp_path / "cache"
    invalid_target = cache / "sha256" / manifest.tree_sha256
    invalid_target.mkdir(parents=True)
    (invalid_target / "sentinel").write_text("do not overwrite")

    with pytest.raises(provenance.AssetClosureError, match="missing"):
        provenance.materialize_asset_closure_from_source_tree(source, cache, manifest)
    assert (invalid_target / "sentinel").read_text() == "do not overwrite"

    other_cache = tmp_path / "other-cache"
    other_cache.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (other_cache / "sha256").symlink_to(outside, target_is_directory=True)
    with pytest.raises(provenance.AssetClosureError, match="must not be a symbolic link"):
        provenance.materialize_asset_closure_from_source_tree(source, other_cache, manifest)
    assert list(outside.iterdir()) == []


def test_materialize_does_not_accept_a_url_as_a_source_tree(tmp_path: Path) -> None:
    with pytest.raises(provenance.AssetClosureError, match="does not exist"):
        provenance.materialize_asset_closure_from_source_tree(
            "https://example.invalid/assets",
            tmp_path / "cache",
            _fixture_manifest(),
        )
