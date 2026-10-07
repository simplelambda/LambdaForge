"""Checkpoint retention keeps recovery and published evidence independent."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lambdaforge.data.DatasetOperations import DatasetOperations
from lambdaforge.work.checkpoint_retention import checkpoint_plan, compact_checkpoints
from lambdaforge.work.checkpoints import CheckpointCollection
from lambdaforge.work.outputs import OutputCollection
from lambdaforge.work.snapshot import copy_file, copy_tree


def test_snapshot_refuses_same_inode_and_symbolic_ancestors(tmp_path: Path) -> None:
    import shutil

    source = tmp_path / "source.bin"
    source.write_bytes(b"must remain intact")
    with pytest.raises(shutil.SameFileError):
        copy_file(source, source)
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        copy_file(source, alias / "copied.bin")
    assert source.read_bytes() == b"must remain intact"
    assert not (target / "copied.bin").exists()


def runtime(root: Path):
    execution = root / "execution-test"
    run = execution / "runs/run-test"
    attempt = run / "attempts/attempt-0001"
    attempt.mkdir(parents=True)
    checkpoints = CheckpointCollection(run / "checkpoints")
    return SimpleNamespace(
        run_dir=attempt,
        checkpoints=checkpoints,
        source_dir=root,
        config=SimpleNamespace(work_class="example.Example"),
        scientific_fingerprint="science",
        execution_id=execution.name,
        run_id=run.name,
        attempt_id=attempt.name,
    ), execution


@pytest.mark.parametrize("status", ["failed", "cancelled", "running", "completed_with_failures"])
def test_recoverable_execution_never_expires_checkpoints(tmp_path: Path, status: str) -> None:
    context, execution = runtime(tmp_path)
    context.checkpoints.save_json("state.json", {"epoch": 20})
    (execution / "result.json").write_text(json.dumps({"status": status}))
    assert checkpoint_plan(execution, grace_seconds=0) == []
    assert context.checkpoints.exists("state.json")


def test_grace_pins_and_terminal_cleanup_preserve_scientific_records(tmp_path: Path) -> None:
    context, execution = runtime(tmp_path)
    context.checkpoints.save_json("state.json", {"epoch": 20})
    context.checkpoints.save_json("best/model.json", {"weight": 3})
    context.checkpoints.pin("best", reason="published-paper-model")
    (execution / "result.json").write_text('{"status":"succeeded"}')
    (context.run_dir / "metrics.jsonl").write_text('{"value":0.9}\n')
    assert checkpoint_plan(execution, grace_seconds=86400) == []
    selected = checkpoint_plan(execution, grace_seconds=0)
    assert [Path(item["path"]).name for item in selected] == ["state.json"]
    assert compact_checkpoints(execution, selected) > 0
    assert not context.checkpoints.exists("state.json")
    assert context.checkpoints.exists("best/model.json")
    assert (context.run_dir / "metrics.jsonl").is_file()
    assert json.loads((execution / "checkpoint-retention.json").read_text())["removed"]
    assert compact_checkpoints(execution, selected) == 0
    context.checkpoints.unpin("best")
    assert [Path(item["path"]).name for item in checkpoint_plan(execution, grace_seconds=0)] == [
        "best"
    ]


@pytest.mark.parametrize(
    "policy",
    [
        '{"storage_policy_version":2}',
        '{"storage_policy_version":1,"pins":{"../bad":"keep"}}',
        "not json",
    ],
)
def test_invalid_policy_protects_collection(tmp_path: Path, policy: str) -> None:
    context, execution = runtime(tmp_path)
    context.checkpoints.save_json("state.json", {})
    (context.checkpoints._root / ".storage-policy.json").write_text(policy)
    (execution / "result.json").write_text('{"status":"succeeded"}')
    assert checkpoint_plan(execution, grace_seconds=0) == []


def test_checkpoint_artifact_snapshot_is_independent_and_can_release_intermediate(
    tmp_path: Path,
) -> None:
    context, execution = runtime(tmp_path)
    source = context.checkpoints.save_json("weights.json", {"weight": 3})
    outputs = OutputCollection(context)
    published = outputs.from_checkpoint("model", "weights.json", release=True)
    assert published.stat().st_ino != source.stat().st_ino
    source.write_text('{"weight":4}')
    assert json.loads(published.read_text()) == {"weight": 3}
    # Changed checkpoint bytes are no longer proven redundant before expiry.
    (execution / "result.json").write_text('{"status":"succeeded"}')
    assert checkpoint_plan(execution, grace_seconds=86400) == []
    source.write_text(json.dumps({"weight": 3}, sort_keys=True, indent=2) + "\n")
    # Re-seal the new exact bytes with a new logical output.
    outputs.from_checkpoint("model2", "weights.json", release=True)
    assert (
        checkpoint_plan(execution, grace_seconds=86400)[0]["reason"]
        == "verified-publication-released-checkpoint"
    )


def test_dataset_can_publish_directly_from_owned_checkpoint_directory(
    tmp_path: Path, monkeypatch
) -> None:
    context, execution = runtime(tmp_path)
    monkeypatch.setenv("LAMBDAFORGE_CLUSTER", "local")
    monkeypatch.setenv("LAMBDAFORGE_DATASET_ROOT", str(tmp_path / "published"))
    monkeypatch.setenv("LAMBDAFORGE_DATASET_REGISTRY", str(tmp_path / "datasets.json"))
    context.checkpoints.save_json("records/a.json", {"value": 1})
    outputs = OutputCollection(context)
    record = outputs.dataset(
        name="example",
        version="1",
        members=[{"id": "a", "assets": {"data": "a.json"}, "split": "train"}],
        source_checkpoint="records",
        release_checkpoints=["records"],
    )
    root = Path(record["placements"][0]["root"])
    assert DatasetOperations.verify(root, record["dataset_id"])["valid"]
    (execution / "result.json").write_text('{"status":"succeeded"}')
    candidates = checkpoint_plan(execution, grace_seconds=86400)
    assert len(candidates) == 1
    compact_checkpoints(execution, candidates)
    assert not context.checkpoints.exists("records")
    assert DatasetOperations.verify(root, record["dataset_id"])["valid"]
    assert not (context.run_dir / "records").exists()  # No intermediate Attempt copy.


def test_publication_rejects_root_and_missing_release_before_commit(
    tmp_path: Path, monkeypatch
) -> None:
    context, _execution = runtime(tmp_path)
    outputs = OutputCollection(context)
    monkeypatch.setenv("LAMBDAFORGE_CLUSTER", "local")
    monkeypatch.setenv("LAMBDAFORGE_DATASET_ROOT", str(tmp_path / "published"))
    with pytest.raises(ValueError, match="named checkpoint"):
        outputs.from_checkpoint("all", ".")
    with pytest.raises(FileNotFoundError):
        outputs.dataset(name="example", version="1", members=[], release_checkpoints=["missing"])
    assert not (tmp_path / "published").exists()


def test_snapshots_never_hardlink_and_reject_symbolic_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.bin").write_bytes(b"original")
    copy_tree(source, tmp_path / "snapshot")
    original, snapshot = source / "data.bin", tmp_path / "snapshot/data.bin"
    assert original.stat().st_ino != snapshot.stat().st_ino
    original.write_bytes(b"changed")
    assert snapshot.read_bytes() == b"original"
    copy_file(snapshot, tmp_path / "file.bin")
    (source / "unsafe").symlink_to(tmp_path / "file.bin")
    with pytest.raises(ValueError):
        copy_tree(source, tmp_path / "unsafe-snapshot")


def test_pin_added_after_preview_prevents_removal(tmp_path: Path) -> None:
    context, execution = runtime(tmp_path)
    context.checkpoints.save_json("state.json", {})
    (execution / "result.json").write_text('{"status":"succeeded"}')
    candidates = checkpoint_plan(execution, grace_seconds=0)
    context.checkpoints.pin("state.json")
    assert compact_checkpoints(execution, candidates) == 0
    assert context.checkpoints.exists("state.json")


def test_checkpoint_deletion_recovers_after_rename_crash(tmp_path: Path, monkeypatch) -> None:
    import importlib

    retention = importlib.import_module("lambdaforge.work.checkpoint_retention")
    context, execution = runtime(tmp_path)
    context.checkpoints.save_json("state.json", {})
    (execution / "result.json").write_text('{"status":"succeeded"}')
    candidates = checkpoint_plan(execution, grace_seconds=0)
    original_remove = retention._remove

    def crash(path):
        raise OSError("simulated interruption after rename")

    monkeypatch.setattr(retention, "_remove", crash)
    with pytest.raises(OSError, match="simulated"):
        compact_checkpoints(execution, candidates)
    receipt = execution / "checkpoint-retention.json"
    assert json.loads(receipt.read_text())["pending"]
    assert not context.checkpoints.exists("state.json")
    monkeypatch.setattr(retention, "_remove", original_remove)
    compact_checkpoints(execution, [])
    assert not json.loads(receipt.read_text())["pending"]
    assert len(json.loads(receipt.read_text())["removed"]) == 1
    assert not list(context.checkpoints._root.glob(".retired-*"))


def test_retention_journal_cannot_target_external_files(tmp_path: Path) -> None:
    _context, execution = runtime(tmp_path)
    protected = tmp_path / "protected.bin"
    protected.write_bytes(b"external")
    (execution / "checkpoint-retention.json").write_text(
        json.dumps(
            {
                "removed": [],
                "pending": [{"trash": str(protected)}],
            }
        )
    )
    with pytest.raises(ValueError):
        compact_checkpoints(execution, [])
    assert protected.read_bytes() == b"external"
