"""Portable content identity regressions for files and directory trees."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lambdaforge.work.managed import canonical_fingerprint, fingerprint


def _write_tree(root: Path, order: tuple[str, ...]) -> None:
    for relative in order:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"content:{relative}".encode())


def test_tree_identity_is_independent_of_creation_order_and_metadata(tmp_path: Path) -> None:
    names = ("a/value.bin", "a-plain.bin", "Z.txt", "á.txt")
    first, second = tmp_path / "first", tmp_path / "second"
    _write_tree(first, names)
    _write_tree(second, tuple(reversed(names)))
    os.chmod(second / "Z.txt", 0o600)
    os.utime(second / "Z.txt", (1, 1))

    assert canonical_fingerprint(first) == canonical_fingerprint(second)
    assert fingerprint(first) == fingerprint(second)


def test_tree_identity_normalizes_portable_unicode_paths(tmp_path: Path) -> None:
    composed, decomposed = tmp_path / "composed", tmp_path / "decomposed"
    (composed / "café").mkdir(parents=True)
    (decomposed / "cafe\N{COMBINING ACUTE ACCENT}").mkdir(parents=True)
    (composed / "café" / "record.txt").write_bytes(b"same")
    (decomposed / "cafe\N{COMBINING ACUTE ACCENT}" / "record.txt").write_bytes(b"same")

    assert canonical_fingerprint(composed) == canonical_fingerprint(decomposed)


def test_tree_identity_records_empty_directories_and_entry_types(tmp_path: Path) -> None:
    with_empty, without_empty = tmp_path / "with", tmp_path / "without"
    (with_empty / "empty").mkdir(parents=True)
    without_empty.mkdir()

    assert canonical_fingerprint(with_empty) != canonical_fingerprint(without_empty)


def test_tree_identity_rejects_canonical_unicode_collisions(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "café").write_bytes(b"one")
    (root / "cafe\N{COMBINING ACUTE ACCENT}").write_bytes(b"two")

    with pytest.raises(ValueError, match="collide after canonical Unicode"):
        canonical_fingerprint(root)
