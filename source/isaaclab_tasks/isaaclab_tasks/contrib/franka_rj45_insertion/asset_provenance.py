# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pinned external-asset closure for the Franka RJ45 pick-insert scene.

The NVIDIA-authored Franka and Seattle-table assets remain outside the source
repository.  This module records their exact dependency closure and verifies a
local, content-addressed copy before production artifact generation or replay.
"""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

FRANKA_RJ45_ASSET_CLOSURE_FORMAT = "isaaclab-franka-rj45-external-asset-closure"
"""Serialization-stable identifier for the external asset closure."""

FRANKA_RJ45_ASSET_CLOSURE_SCHEMA_VERSION = 1
"""Schema version for the external asset closure."""

FRANKA_RJ45_ASSET_CLOSURE_ENV = "ISAACLAB_FRANKA_RJ45_ASSET_CLOSURE_ROOT"
"""Environment variable naming a verified local closure root."""

FRANKA_RJ45_ASSET_CLOSURE_LOGICAL_URI = "isaaclab-asset-closure://franka-rj45/scene-v1"
"""Path-independent identity of the combined scene closure."""

FRANKA_RJ45_FRANKA_RELATIVE_PATH = "Isaac/IsaacLab/Robots/FrankaEmika/franka_panda.usda"
FRANKA_RJ45_FRANKA_LOGICAL_URI = f"isaac-asset://6.1/{FRANKA_RJ45_FRANKA_RELATIVE_PATH}"
FRANKA_RJ45_SEATTLE_TABLE_RELATIVE_PATH = "Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd"
FRANKA_RJ45_SEATTLE_TABLE_LOGICAL_URI = f"isaac-asset://6.1/{FRANKA_RJ45_SEATTLE_TABLE_RELATIVE_PATH}"

FRANKA_RJ45_ASSET_CLOSURE_TREE_SHA256 = "060d50ba665ae9850a8ef19c1b42a2a55579d4f8924cadfc37c1ec83ca726a76"
"""SHA-256 digest of the canonical path/size/content manifest."""

FRANKA_RJ45_ASSET_CLOSURE_FILE_COUNT = 19
FRANKA_RJ45_ASSET_CLOSURE_TOTAL_SIZE = 83_718_325
FRANKA_RJ45_ASSET_CLOSURE_TREE_DIGEST_ALGORITHM = "sha256-path-size-content-v1"

_TREE_DIGEST_DOMAIN = b"isaaclab-franka-rj45-asset-closure-tree-v1\0"
_COPY_BUFFER_SIZE = 1024 * 1024


class AssetClosureError(ValueError):
    """Raised when an external asset closure is absent, malformed, or modified."""


@dataclass(frozen=True, slots=True)
class AssetClosureFile:
    """One regular file in a pinned asset closure."""

    relative_path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class AssetClosureManifest:
    """Path-independent identity and exact file manifest for one asset closure."""

    format: str
    schema_version: int
    logical_uri: str
    tree_sha256: str
    files: tuple[AssetClosureFile, ...]


@dataclass(frozen=True, slots=True)
class VerifiedAssetClosure:
    """A local asset tree that passed exact manifest verification."""

    root: Path
    tree_sha256: str
    file_count: int
    total_size: int


@dataclass(frozen=True, slots=True)
class FrankaRJ45AssetClosure:
    """Verified task entrypoints inside the pinned Franka RJ45 closure."""

    root: Path
    franka_usd_path: Path
    seattle_table_usd_path: Path
    tree_sha256: str


FRANKA_RJ45_ASSET_CLOSURE_FILES = (
    AssetClosureFile(
        FRANKA_RJ45_FRANKA_RELATIVE_PATH,
        1_732,
        "ab094f718668e6fa24cb2595c3bf9c07a74a7e5ec4bd29e1ed7a975b69d9b701",
    ),
    AssetClosureFile(
        "Isaac/IsaacLab/Robots/FrankaEmika/payloads/Physics/mujoco.usda",
        25_544,
        "82f363dd872bd8ca88847a76b1daf267fdace4095f6d505f3eb16acf49596f8d",
    ),
    AssetClosureFile(
        "Isaac/IsaacLab/Robots/FrankaEmika/payloads/Physics/physics.usda",
        29_884,
        "0dc38454f02ea14d9ddd2437995fdc7c4a65634443cacdcc2a04e3de25655e00",
    ),
    AssetClosureFile(
        "Isaac/IsaacLab/Robots/FrankaEmika/payloads/Physics/physx.usda",
        5_037,
        "d68c2d5ad9dad3e107ee41d477bb828c9410e473ccd20e0ed32df0b7925c32b3",
    ),
    AssetClosureFile(
        "Isaac/IsaacLab/Robots/FrankaEmika/payloads/base.usda",
        71_251,
        "7121cbfc8489dbac81ce6a781b16f9cc52390f9ff75ca3ef75022d775098bc6d",
    ),
    AssetClosureFile(
        "Isaac/IsaacLab/Robots/FrankaEmika/payloads/geometries.usd",
        7_097_437,
        "c64f015a69302f5b277040fc54fc7addedab49c47e67d1b79f374414dd4e991b",
    ),
    AssetClosureFile(
        "Isaac/IsaacLab/Robots/FrankaEmika/payloads/instances.usda",
        80_355,
        "04857fae746db145b454d3c3065b3bd8e04a9b87e35961c9ffc171ede3248cec",
    ),
    AssetClosureFile(
        "Isaac/IsaacLab/Robots/FrankaEmika/payloads/materials.usda",
        6_716,
        "3b87896947047adc23abafac72ccb113b26a32293645db8e755c51f2abe58b9b",
    ),
    AssetClosureFile(
        "Isaac/IsaacLab/Robots/FrankaEmika/payloads/robot.usda",
        5_328,
        "060bd8bc6f47b6d57ef918d05629af36763998a275c5102bd4aa4540e0fcf15b",
    ),
    AssetClosureFile(
        "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableBase_BaseColor.png",
        6_335_308,
        "2cf139f46874df8a133c26eb51405161824ddecb0ce6a094e6477a675656bfe4",
    ),
    AssetClosureFile(
        "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableBase_Metallic.png",
        119_609,
        "c56bdeeecef908a1a6869af71556ad9c43794d0d292e3b8463aa7b82f3367771",
    ),
    AssetClosureFile(
        "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableBase_Normal.png",
        34_371_270,
        "bfcbadb3f98c9fe116bc6eb2c040d09196c40ca16347367900f8fd2d3451e10d",
    ),
    AssetClosureFile(
        "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableBase_Roughness.png",
        7_077_839,
        "35426471bf910ae4ef32d044b360489de8477c97cf1a75ad25341ef2a25a644b",
    ),
    AssetClosureFile(
        "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableParts_BaseColor.png",
        905_610,
        "5d095ed2e4fe2fa18d35a2677f3c281796fba135544439db8ec32d851144fbc0",
    ),
    AssetClosureFile(
        "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableParts_Metallic.png",
        414_290,
        "fdcd7dcec52034b60662cccf58c02ae442c0a87d60b099fcf6cd9ccd6cf5535d",
    ),
    AssetClosureFile(
        "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableParts_Normal.png",
        13_632_148,
        "9413615d7cf9809f639ec9722022c164ad73a14add32a94293fc7c69d97b601c",
    ),
    AssetClosureFile(
        "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableParts_Roughness.png",
        2_995_685,
        "1cb2096ba97d3519f876b9d0dc84822bdd428cf93f0887a8f75e3bba208c4a35",
    ),
    AssetClosureFile(
        "Isaac/Props/Mounts/SeattleLabTable/table.usd",
        10_538_698,
        "8e6e2284e0eba868341b5afacf663d4f84005febef4f9c178d916e609d44c114",
    ),
    AssetClosureFile(
        FRANKA_RJ45_SEATTLE_TABLE_RELATIVE_PATH,
        4_584,
        "479ed8fd374d8eafdedf3ca3b9d0d5981ffc6d5c5016929347f748be93d5df17",
    ),
)
"""Exact Isaac 6.1 Franka and Seattle-table dependency files."""

FRANKA_RJ45_ASSET_CLOSURE_MANIFEST = AssetClosureManifest(
    format=FRANKA_RJ45_ASSET_CLOSURE_FORMAT,
    schema_version=FRANKA_RJ45_ASSET_CLOSURE_SCHEMA_VERSION,
    logical_uri=FRANKA_RJ45_ASSET_CLOSURE_LOGICAL_URI,
    tree_sha256=FRANKA_RJ45_ASSET_CLOSURE_TREE_SHA256,
    files=FRANKA_RJ45_ASSET_CLOSURE_FILES,
)
"""Pinned manifest consumed by verification and offline materialization."""


def _validate_sha256(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AssetClosureError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


def _validate_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\0" in value or "\\" in value:
        raise AssetClosureError("Asset closure paths must be non-empty canonical POSIX paths.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts) or path.as_posix() != value:
        raise AssetClosureError(f"Asset closure path is not a canonical relative POSIX path: {value!r}.")
    return value


def _validated_manifest_files(manifest: AssetClosureManifest) -> tuple[AssetClosureFile, ...]:
    if not isinstance(manifest, AssetClosureManifest):
        raise AssetClosureError("Asset closure manifest has the wrong type.")
    if manifest.format != FRANKA_RJ45_ASSET_CLOSURE_FORMAT:
        raise AssetClosureError(f"Unsupported asset closure format: {manifest.format!r}.")
    if manifest.schema_version != FRANKA_RJ45_ASSET_CLOSURE_SCHEMA_VERSION:
        raise AssetClosureError(f"Unsupported asset closure schema version: {manifest.schema_version!r}.")
    parsed_uri = urlparse(manifest.logical_uri)
    if not parsed_uri.scheme or not parsed_uri.netloc:
        raise AssetClosureError("Asset closure logical_uri must be an absolute logical URI.")
    _validate_sha256(manifest.tree_sha256, field_name="Asset closure tree_sha256")
    if not manifest.files:
        raise AssetClosureError("Asset closure manifest must contain at least one file.")

    paths = []
    for entry in manifest.files:
        if not isinstance(entry, AssetClosureFile):
            raise AssetClosureError("Asset closure file entries have the wrong type.")
        paths.append(_validate_relative_path(entry.relative_path))
        if type(entry.size) is not int or entry.size < 0:
            raise AssetClosureError(f"Asset closure size must be a non-negative integer: {entry.relative_path!r}.")
        _validate_sha256(entry.sha256, field_name=f"Asset closure file digest for {entry.relative_path!r}")
    if paths != sorted(paths):
        raise AssetClosureError("Asset closure file entries must be sorted by relative path.")
    if len(paths) != len(set(paths)):
        raise AssetClosureError("Asset closure file entries must have unique relative paths.")
    return manifest.files


def asset_closure_tree_digest(files: Sequence[AssetClosureFile]) -> str:
    """Return the canonical tree digest for sorted path, size, and file-digest entries."""
    ordered = tuple(sorted(files, key=lambda entry: entry.relative_path))
    temporary = AssetClosureManifest(
        format=FRANKA_RJ45_ASSET_CLOSURE_FORMAT,
        schema_version=FRANKA_RJ45_ASSET_CLOSURE_SCHEMA_VERSION,
        logical_uri="isaaclab-asset-closure://digest/input",
        tree_sha256="0" * 64,
        files=ordered,
    )
    _validated_manifest_files(temporary)
    digest = hashlib.sha256(_TREE_DIGEST_DOMAIN)
    for entry in ordered:
        digest.update(entry.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry.size).encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(entry.sha256))
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_manifest(manifest: AssetClosureManifest) -> tuple[AssetClosureFile, ...]:
    files = _validated_manifest_files(manifest)
    observed_tree = asset_closure_tree_digest(files)
    if observed_tree != manifest.tree_sha256:
        raise AssetClosureError(
            f"Asset closure manifest tree digest mismatch: expected {manifest.tree_sha256}, got {observed_tree}."
        )
    return files


def _absolute_path(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _require_directory(path: Path, *, description: str) -> None:
    try:
        path_stat = path.lstat()
    except FileNotFoundError as exc:
        raise AssetClosureError(f"{description} does not exist: {path}.") from exc
    if stat.S_ISLNK(path_stat.st_mode):
        raise AssetClosureError(f"{description} must not be a symbolic link: {path}.")
    if not stat.S_ISDIR(path_stat.st_mode):
        raise AssetClosureError(f"{description} is not a directory: {path}.")


def _expected_directories(files: Sequence[AssetClosureFile]) -> set[str]:
    directories = set()
    for entry in files:
        parent = PurePosixPath(entry.relative_path).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _require_regular_manifest_file(root: Path, entry: AssetClosureFile) -> Path:
    candidate = root
    parts = PurePosixPath(entry.relative_path).parts
    for index, part in enumerate(parts):
        candidate = candidate / part
        try:
            candidate_stat = candidate.lstat()
        except FileNotFoundError as exc:
            raise AssetClosureError(f"Asset closure file is missing: {entry.relative_path}.") from exc
        if stat.S_ISLNK(candidate_stat.st_mode):
            raise AssetClosureError(f"Asset closure contains a symbolic link: {entry.relative_path}.")
        if index < len(parts) - 1 and not stat.S_ISDIR(candidate_stat.st_mode):
            raise AssetClosureError(f"Asset closure path component is not a directory: {entry.relative_path}.")
        if index == len(parts) - 1 and not stat.S_ISREG(candidate_stat.st_mode):
            raise AssetClosureError(f"Asset closure entry is not a regular file: {entry.relative_path}.")
    return candidate


def _sha256_regular_file(path: Path, entry: AssetClosureFile) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AssetClosureError(f"Unable to open asset closure file safely: {entry.relative_path}.") from exc
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise AssetClosureError(f"Asset closure entry is not a regular file: {entry.relative_path}.")
        if file_stat.st_size != entry.size:
            raise AssetClosureError(
                f"Asset closure file size mismatch for {entry.relative_path}: "
                f"expected {entry.size}, got {file_stat.st_size}."
            )
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(_COPY_BUFFER_SIZE), b""):
                digest.update(block)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _copy_verified_regular_file(source_root: Path, destination: Path, entry: AssetClosureFile) -> None:
    """Copy one expected regular file without following a last-component link."""
    source = _require_regular_manifest_file(source_root, entry)
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_descriptor = os.open(source, source_flags)
    except OSError as exc:
        raise AssetClosureError(f"Unable to open asset source file safely: {entry.relative_path}.") from exc
    try:
        source_stat = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise AssetClosureError(f"Asset source entry is not a regular file: {entry.relative_path}.")
        if source_stat.st_size != entry.size:
            raise AssetClosureError(
                f"Asset source file size mismatch for {entry.relative_path}: "
                f"expected {entry.size}, got {source_stat.st_size}."
            )
        try:
            destination_descriptor = os.open(destination, destination_flags, 0o600)
        except OSError as exc:
            raise AssetClosureError(f"Unable to create staged asset file: {entry.relative_path}.") from exc
        try:
            digest = hashlib.sha256()
            while block := os.read(source_descriptor, _COPY_BUFFER_SIZE):
                digest.update(block)
                remaining = memoryview(block)
                while remaining:
                    written = os.write(destination_descriptor, remaining)
                    if written <= 0:
                        raise AssetClosureError(f"Unable to finish staged asset file: {entry.relative_path}.")
                    remaining = remaining[written:]
            observed_digest = digest.hexdigest()
            if observed_digest != entry.sha256:
                raise AssetClosureError(
                    f"Asset source file digest mismatch for {entry.relative_path}: "
                    f"expected {entry.sha256}, got {observed_digest}."
                )
            os.fchmod(destination_descriptor, 0o444)
        finally:
            os.close(destination_descriptor)
    finally:
        os.close(source_descriptor)


def _verify_expected_files(root: Path, files: Sequence[AssetClosureFile]) -> None:
    for entry in files:
        candidate = _require_regular_manifest_file(root, entry)
        observed_digest = _sha256_regular_file(candidate, entry)
        if observed_digest != entry.sha256:
            raise AssetClosureError(
                f"Asset closure file digest mismatch for {entry.relative_path}: "
                f"expected {entry.sha256}, got {observed_digest}."
            )


def _reject_unexpected_entries(root: Path, files: Sequence[AssetClosureFile]) -> None:
    expected_files = {entry.relative_path for entry in files}
    expected_directories = _expected_directories(files)
    for current_root, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current = Path(current_root)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            candidate = current / name
            relative = candidate.relative_to(root).as_posix()
            candidate_stat = candidate.lstat()
            if stat.S_ISLNK(candidate_stat.st_mode):
                raise AssetClosureError(f"Asset closure contains a symbolic link: {relative}.")
            if not stat.S_ISDIR(candidate_stat.st_mode):
                raise AssetClosureError(f"Asset closure contains a non-directory path component: {relative}.")
            if relative not in expected_directories:
                raise AssetClosureError(f"Asset closure contains an unexpected directory: {relative}.")
        for name in file_names:
            candidate = current / name
            relative = candidate.relative_to(root).as_posix()
            candidate_stat = candidate.lstat()
            if stat.S_ISLNK(candidate_stat.st_mode):
                raise AssetClosureError(f"Asset closure contains a symbolic link: {relative}.")
            if not stat.S_ISREG(candidate_stat.st_mode):
                raise AssetClosureError(f"Asset closure contains a non-regular file: {relative}.")
            if relative not in expected_files:
                raise AssetClosureError(f"Asset closure contains an unexpected file: {relative}.")


def verify_asset_closure(root: str | os.PathLike[str], manifest: AssetClosureManifest) -> VerifiedAssetClosure:
    """Verify every byte and reject every unexpected entry in a local closure."""
    files = _validate_manifest(manifest)
    absolute_root = _absolute_path(root)
    _require_directory(absolute_root, description="Asset closure root")
    _verify_expected_files(absolute_root, files)
    _reject_unexpected_entries(absolute_root, files)
    return VerifiedAssetClosure(
        root=absolute_root,
        tree_sha256=manifest.tree_sha256,
        file_count=len(files),
        total_size=sum(entry.size for entry in files),
    )


def _franka_rj45_closure(verified: VerifiedAssetClosure) -> FrankaRJ45AssetClosure:
    return FrankaRJ45AssetClosure(
        root=verified.root,
        franka_usd_path=verified.root / FRANKA_RJ45_FRANKA_RELATIVE_PATH,
        seattle_table_usd_path=verified.root / FRANKA_RJ45_SEATTLE_TABLE_RELATIVE_PATH,
        tree_sha256=verified.tree_sha256,
    )


def verify_franka_rj45_asset_closure(root: str | os.PathLike[str]) -> FrankaRJ45AssetClosure:
    """Verify and resolve the pinned Franka and Seattle-table entrypoints."""
    return _franka_rj45_closure(verify_asset_closure(root, FRANKA_RJ45_ASSET_CLOSURE_MANIFEST))


def configured_franka_rj45_asset_closure(
    *,
    required: bool = False,
    environ: Mapping[str, str] | None = None,
) -> FrankaRJ45AssetClosure | None:
    """Resolve and verify the task-specific closure configured in the environment."""
    environment = os.environ if environ is None else environ
    configured = environment.get(FRANKA_RJ45_ASSET_CLOSURE_ENV, "")
    if not isinstance(configured, str):
        raise AssetClosureError(f"{FRANKA_RJ45_ASSET_CLOSURE_ENV} must be a filesystem path.")
    configured = configured.strip()
    if not configured:
        if required:
            raise AssetClosureError(f"{FRANKA_RJ45_ASSET_CLOSURE_ENV} must name the pinned local asset closure.")
        return None
    expanded = Path(configured).expanduser()
    if not expanded.is_absolute():
        raise AssetClosureError(f"{FRANKA_RJ45_ASSET_CLOSURE_ENV} must be an absolute filesystem path.")
    return verify_franka_rj45_asset_closure(expanded)


def materialize_asset_closure_from_source_tree(
    source_tree: str | os.PathLike[str],
    cache_root: str | os.PathLike[str],
    manifest: AssetClosureManifest,
) -> VerifiedAssetClosure:
    """Copy a verified local source tree atomically into a SHA-addressed cache.

    The source may contain unrelated assets; only manifest-listed regular files
    are copied.  This function never resolves URLs or performs network I/O.
    """
    files = _validate_manifest(manifest)
    absolute_source = _absolute_path(source_tree)
    _require_directory(absolute_source, description="Asset source tree")

    absolute_cache = _absolute_path(cache_root)
    with suppress(FileExistsError):
        absolute_cache.mkdir(parents=True, exist_ok=False)
    _require_directory(absolute_cache, description="Asset closure cache root")

    target_parent = absolute_cache / "sha256"
    with suppress(FileExistsError):
        target_parent.mkdir(exist_ok=False)
    _require_directory(target_parent, description="Asset closure SHA-256 cache directory")

    target = target_parent / manifest.tree_sha256
    if target.exists() or target.is_symlink():
        return verify_asset_closure(target, manifest)

    staging = Path(tempfile.mkdtemp(prefix=f".{manifest.tree_sha256}.", dir=target_parent))
    payload = staging / "payload"
    try:
        payload.mkdir()
        for entry in files:
            destination = payload / Path(*PurePosixPath(entry.relative_path).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _copy_verified_regular_file(absolute_source, destination, entry)
        verify_asset_closure(payload, manifest)
        try:
            payload.rename(target)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not (target.exists() or target.is_symlink()):
                raise
        return verify_asset_closure(target, manifest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def materialize_franka_rj45_asset_closure(
    source_tree: str | os.PathLike[str], cache_root: str | os.PathLike[str]
) -> FrankaRJ45AssetClosure:
    """Materialize the pinned Franka RJ45 closure from an existing local tree."""
    verified = materialize_asset_closure_from_source_tree(
        source_tree,
        cache_root,
        FRANKA_RJ45_ASSET_CLOSURE_MANIFEST,
    )
    return _franka_rj45_closure(verified)


def franka_rj45_asset_contract() -> dict[str, object]:
    """Return the path-independent external-asset contract for task artifacts."""
    _validate_manifest(FRANKA_RJ45_ASSET_CLOSURE_MANIFEST)
    return {
        "contract_version": FRANKA_RJ45_ASSET_CLOSURE_SCHEMA_VERSION,
        "logical_uri": FRANKA_RJ45_ASSET_CLOSURE_LOGICAL_URI,
        "tree_sha256": FRANKA_RJ45_ASSET_CLOSURE_TREE_SHA256,
        "tree_digest_algorithm": FRANKA_RJ45_ASSET_CLOSURE_TREE_DIGEST_ALGORITHM,
        "file_count": FRANKA_RJ45_ASSET_CLOSURE_FILE_COUNT,
        "total_size": FRANKA_RJ45_ASSET_CLOSURE_TOTAL_SIZE,
        "entrypoints": {
            "franka_robot": {
                "logical_uri": FRANKA_RJ45_FRANKA_LOGICAL_URI,
                "relative_path": FRANKA_RJ45_FRANKA_RELATIVE_PATH,
                "sha256": FRANKA_RJ45_ASSET_CLOSURE_FILES[0].sha256,
            },
            "seattle_table": {
                "logical_uri": FRANKA_RJ45_SEATTLE_TABLE_LOGICAL_URI,
                "relative_path": FRANKA_RJ45_SEATTLE_TABLE_RELATIVE_PATH,
                "sha256": FRANKA_RJ45_ASSET_CLOSURE_FILES[-1].sha256,
            },
        },
        "runtime_dependencies": (
            {
                "logical_uri": "mdl://OmniPBR.mdl",
                "provider": "pinned-renderer-runtime-image",
            },
        ),
    }


if len(FRANKA_RJ45_ASSET_CLOSURE_FILES) != FRANKA_RJ45_ASSET_CLOSURE_FILE_COUNT:
    raise RuntimeError("Pinned Franka RJ45 asset closure file count is inconsistent.")
if sum(entry.size for entry in FRANKA_RJ45_ASSET_CLOSURE_FILES) != FRANKA_RJ45_ASSET_CLOSURE_TOTAL_SIZE:
    raise RuntimeError("Pinned Franka RJ45 asset closure total size is inconsistent.")
if asset_closure_tree_digest(FRANKA_RJ45_ASSET_CLOSURE_FILES) != FRANKA_RJ45_ASSET_CLOSURE_TREE_SHA256:
    raise RuntimeError("Pinned Franka RJ45 asset closure tree digest is inconsistent.")


__all__ = [
    "AssetClosureError",
    "AssetClosureFile",
    "AssetClosureManifest",
    "FRANKA_RJ45_ASSET_CLOSURE_ENV",
    "FRANKA_RJ45_ASSET_CLOSURE_FILE_COUNT",
    "FRANKA_RJ45_ASSET_CLOSURE_FILES",
    "FRANKA_RJ45_ASSET_CLOSURE_FORMAT",
    "FRANKA_RJ45_ASSET_CLOSURE_LOGICAL_URI",
    "FRANKA_RJ45_ASSET_CLOSURE_MANIFEST",
    "FRANKA_RJ45_ASSET_CLOSURE_SCHEMA_VERSION",
    "FRANKA_RJ45_ASSET_CLOSURE_TOTAL_SIZE",
    "FRANKA_RJ45_ASSET_CLOSURE_TREE_DIGEST_ALGORITHM",
    "FRANKA_RJ45_ASSET_CLOSURE_TREE_SHA256",
    "FRANKA_RJ45_FRANKA_LOGICAL_URI",
    "FRANKA_RJ45_FRANKA_RELATIVE_PATH",
    "FRANKA_RJ45_SEATTLE_TABLE_LOGICAL_URI",
    "FRANKA_RJ45_SEATTLE_TABLE_RELATIVE_PATH",
    "FrankaRJ45AssetClosure",
    "VerifiedAssetClosure",
    "asset_closure_tree_digest",
    "configured_franka_rj45_asset_closure",
    "franka_rj45_asset_contract",
    "materialize_asset_closure_from_source_tree",
    "materialize_franka_rj45_asset_closure",
    "verify_asset_closure",
    "verify_franka_rj45_asset_closure",
]
