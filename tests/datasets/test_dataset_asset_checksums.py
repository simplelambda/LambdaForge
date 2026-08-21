"""Regression coverage for portable DatasetAsset checksum semantics."""

from __future__ import annotations

import hashlib
from pathlib import Path

from lambdaforge.data import DatasetAsset, DatasetIndex, DatasetMember
from lambdaforge.tasks import TaskArtifact


def _index(path: Path, asset: DatasetAsset) -> DatasetIndex:
    return DatasetIndex.write(
        path / "members.jsonl",
        (DatasetMember("protein-1", assets={"annotation": asset}),),
    )


def test_dataset_file_checksum_is_the_sha256_of_file_bytes(tmp_path: Path) -> None:
    asset_path = tmp_path / "annotation.npz"
    content = b"portable scientific bytes"
    asset_path.write_bytes(content)
    index = _index(
        tmp_path,
        DatasetAsset(
            "annotation.npz",
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
        ),
    )

    validation = index.validate(tmp_path, require_checksums=True)

    assert validation == {
        "valid": True,
        "member_count": 1,
        "errors": [],
        "warnings": [],
    }


def test_legacy_filename_prefixed_checksum_remains_readable(tmp_path: Path) -> None:
    asset_path = tmp_path / "annotation.npz"
    asset_path.write_bytes(b"legacy publication")
    digest, size = TaskArtifact.fingerprint_path(asset_path)
    index = _index(
        tmp_path,
        DatasetAsset("annotation.npz", sha256=digest, size_bytes=size),
    )

    validation = index.validate(tmp_path, require_checksums=True)

    assert validation["valid"] is True
    assert validation["warnings"] == [
        "protein-1.annotation: legacy file checksum accepted"
    ]


def test_dataset_file_checksum_still_rejects_changed_bytes(tmp_path: Path) -> None:
    asset_path = tmp_path / "annotation.npz"
    asset_path.write_bytes(b"actual")
    index = _index(
        tmp_path,
        DatasetAsset("annotation.npz", sha256=hashlib.sha256(b"other").hexdigest()),
    )

    validation = index.validate(tmp_path, require_checksums=True)

    assert validation["valid"] is False
    assert validation["errors"] == ["protein-1.annotation: asset integrity differs"]
