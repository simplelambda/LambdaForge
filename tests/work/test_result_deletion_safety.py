"""Execution deletion must coordinate with native writers and prune only owned empty parents."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.ResultStore import ResultStore


def owned(tmp_path: Path, *, status: str = "succeeded") -> tuple[ResultStore, Path]:
    store = ResultStore(tmp_path / "runs")
    directory = store.root / "work" / "execution-a1"
    directory.mkdir(parents=True)
    result = {
        "execution_result_version": 1,
        "execution_id": directory.name,
        "name": "work",
        "scientific_fingerprint": "sha256:test",
        "status": status,
        "runs": [],
        "summary": {},
    }
    (directory / "result.json").write_text(json.dumps(result))
    origin = {
        "execution_manifest_version": 1,
        "execution_id": directory.name,
        "name": "work",
        "scientific_fingerprint": "sha256:test",
        "ownership": {"execution_dir": "owned"},
    }
    (directory / "execution.json").write_text(json.dumps(origin))
    return store, directory


def test_preview_creates_nothing_and_apply_prunes_only_empty_work_parent(tmp_path: Path) -> None:
    store, directory = owned(tmp_path)
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
    preview = store.delete(directory.name)
    assert not preview["applied"] and "published products" in preview["preserved"]
    assert before == {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
    result = store.delete(directory.name, apply=True)
    assert result["removed_empty_work_directory"]
    assert store.root.is_dir() and not directory.parent.exists()
    assert store.delete(directory.name, apply=True)["already_deleted"]


def test_other_execution_and_foreign_parent_content_are_preserved(tmp_path: Path) -> None:
    store, directory = owned(tmp_path)
    foreign = directory.parent / "researcher-notes.txt"
    foreign.write_text("keep")
    sibling = directory.parent / "execution-b2"
    sibling.mkdir()
    result = store.delete(directory.name, apply=True)
    assert not result["removed_empty_work_directory"]
    assert foreign.read_text() == "keep" and sibling.is_dir()


@pytest.mark.parametrize("status", ["running", "preparing", "unknown"])
def test_active_or_unverifiable_execution_is_not_physically_deleted(
    tmp_path: Path, status: str
) -> None:
    store, directory = owned(tmp_path, status=status)
    with pytest.raises(ValueError, match="cancel/reconcile"):
        store.delete(directory.name, apply=True)
    assert directory.is_dir() and not (store.root / ".study-import.lock").exists()


def test_controller_lease_protects_terminal_result_during_recovery(tmp_path: Path) -> None:
    store, directory = owned(tmp_path, status="failed")
    with CrossProcessFileLock(
        directory / ".controller.lock",
        shared=False,
        timeout_seconds=1,
        poll_interval_seconds=0.01,
    ):
        with pytest.raises(TimeoutError, match="controller.lock"):
            store.delete(directory.name, apply=True)
    assert directory.is_dir() and not store._receipt_root.exists()


@pytest.mark.parametrize("problem", ["missing", "foreign", "identity", "symbolic"])
def test_missing_corrupt_or_unowned_origin_fails_closed(tmp_path: Path, problem: str) -> None:
    store, directory = owned(tmp_path)
    path = directory / "execution.json"
    if problem in {"missing", "symbolic"}:
        path.unlink()
        if problem == "symbolic":
            foreign = tmp_path / "origin.json"
            foreign.write_text("{}")
            path.symlink_to(foreign)
    else:
        value = json.loads(path.read_text())
        value["ownership"] = (
            {"execution_dir": "foreign"} if problem == "foreign" else value["ownership"]
        )
        if problem == "identity":
            value["scientific_fingerprint"] = "sha256:other"
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="ownership"):
        store.delete(directory.name, apply=True)
    assert directory.is_dir()


def test_symbolic_work_parent_never_redirects_deletion(tmp_path: Path) -> None:
    store, directory = owned(tmp_path)
    target = tmp_path / "foreign"
    directory.parent.rename(target)
    directory.parent.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        store.delete(directory.name, apply=True)
    assert (target / directory.name / "result.json").is_file()


def test_duplicate_concurrent_deletes_are_idempotent(tmp_path: Path) -> None:
    store, directory = owned(tmp_path)
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda _: store.delete(directory.name, apply=True), range(3)))
    assert all(item["applied"] for item in results)
    assert not directory.parent.exists() and store.root.is_dir()
