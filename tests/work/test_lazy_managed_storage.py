"""Missing cache/checkpoint reads must not materialize empty scientific storage trees."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from lambdaforge.work.cache import WorkCache
from lambdaforge.work.checkpoints import CheckpointCollection
from lambdaforge.work.managed import ManagedFileStore


def test_unopened_and_missing_managed_storage_reads_create_nothing(tmp_path: Path) -> None:
    cache = WorkCache(tmp_path / "cache")
    checkpoints = CheckpointCollection(tmp_path / "checkpoints")
    store = ManagedFileStore(tmp_path / "managed", scope="checkpoint")
    assert cache.get("missing", "default") == "default"
    assert not checkpoints.exists("missing.json")
    assert checkpoints.restore_reference("missing", sha256="a" * 64, size_bytes=1) is None
    assert store.restore("missing") is None
    for service in (cache, checkpoints):
        with pytest.raises(FileNotFoundError):
            service.file("missing")
    with pytest.raises(FileNotFoundError):
        checkpoints.load_json("missing.json")
    assert list(tmp_path.iterdir()) == []


def test_first_write_creates_only_needed_metadata_and_still_serializes_builds(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    cache = WorkCache(root)
    built = []

    def build(path: Path) -> None:
        built.append(path)
        path.write_bytes(b"exact managed bytes")

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: cache.file("nested/file", build=build), range(4)))
    assert len(built) == 1
    assert len({result.sha256 for result in results}) == 1
    assert all(result.read_bytes() == b"exact managed bytes" for result in results)
    checkpoints = CheckpointCollection(tmp_path / "checkpoints")
    checkpoints.save_json("state.json", {"epoch": 1})
    assert checkpoints.load_json("state.json") == {"epoch": 1}
    assert not (checkpoints._root / ".managed").exists()
    assert not (root / "unrelated").exists()


def test_existing_managed_reads_do_not_create_writer_locks(tmp_path: Path) -> None:
    root = tmp_path / "managed"
    initial = ManagedFileStore(root, scope="checkpoint")
    stored = initial.put_bytes("value", b"content")
    # Legacy read-only files can be valid even without a current writer-lock directory.
    import shutil

    shutil.rmtree(initial.locks_root)
    before = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    reader = ManagedFileStore(root, scope="checkpoint")
    assert reader.restore("value") == stored
    assert not reader.locks_root.exists()
    assert before == {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
