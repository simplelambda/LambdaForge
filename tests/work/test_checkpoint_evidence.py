"""Checkpoint cadence, integrity and retention do not imply scientific continuation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from lambdaforge.work.checkpoint_retention import checkpoint_plan, compact_checkpoints
from lambdaforge.work.checkpoints import CheckpointCollection, checkpoint_payloads
from lambdaforge.work.recovery import checkpoint_recovery_preview
from tests.work.test_checkpoint_retention import runtime


def test_checkpoint_policy_is_opt_in_and_never_creates_payload(tmp_path: Path) -> None:
    checkpoints = CheckpointCollection(tmp_path / "checkpoints")
    assert not checkpoints.should_save(step=0)
    assert not checkpoints._root.exists()
    checkpoints.configure(interval_steps=5, interval_seconds=1800, storage_durability="shared")
    assert checkpoints.should_save(step=0)
    assert list(checkpoint_payloads(checkpoints._root)) == []
    assert checkpoints.inventory()["total"] == 0
    checkpoints.save_json("state.json", {"epoch": 10}, step=10)
    last = checkpoints.inventory()["items"][0]
    at = datetime.fromisoformat(last["written_at_utc"]).timestamp()
    assert not checkpoints.should_save(step=14, now=at + 10)
    assert checkpoints.should_save(step=15, now=at + 10)
    assert checkpoints.should_save(step=11, now=at + 1800)
    assert not checkpoints.should_save(step=1, now=at - 1800)


def test_owner_metadata_and_explicit_integrity_are_preserved(tmp_path: Path) -> None:
    owner = {
        "run_id": "run-original",
        "attempt_id": "attempt-0001",
        "scientific_identity": "original",
    }
    checkpoints = CheckpointCollection(tmp_path / "checkpoints", owner=owner)
    path = checkpoints.save_json("state.json", {"epoch": 3}, step=3)
    before = {p: p.read_bytes() for p in checkpoints._root.rglob("*") if p.is_file()}
    assert checkpoints.inventory()["items"][0]["integrity"] == "not_checked"
    item = checkpoints.inventory(verify=True)["items"][0]
    assert item["owner"] == owner and item["step"] == 3
    assert item["integrity"] == "verified"
    assert item["compatibility"] == "application_validation_required"
    assert item["storage_durability"] == "unspecified"
    assert before == {p: p.read_bytes() for p in checkpoints._root.rglob("*") if p.is_file()}
    path.write_text('{"epoch": 99}')
    assert checkpoints.inventory(verify=True)["items"][0]["integrity"] == "invalid"
    with pytest.raises(ValueError, match="integrity"):
        checkpoints.load_json("state.json")


def test_legacy_checkpoint_is_readable_not_silently_registered(tmp_path: Path) -> None:
    checkpoints = CheckpointCollection(tmp_path)
    (tmp_path / "legacy.json").write_text('{"epoch":3}')
    assert checkpoints.load_json("legacy.json") == {"epoch": 3}
    assert checkpoints.inventory()["total"] == 0
    assert not (tmp_path / ".managed").exists()


@pytest.mark.parametrize("field", ["execution_id", "run_id"])
def test_registered_foreign_checkpoint_is_neither_loaded_nor_overwritten(
    tmp_path: Path,
    field: str,
) -> None:
    owner = {"execution_id": "execution-original", "run_id": "run-original"}
    writer = CheckpointCollection(tmp_path, owner=owner)
    writer.save_json("state.json", {"step": 9})
    managed = writer.file("state.bin", build=lambda path: path.write_bytes(b"state"))
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    foreign = CheckpointCollection(tmp_path, owner={**owner, field: "foreign"})
    for operation in (
        lambda: foreign.load_json("state.json"),
        lambda: foreign.save_json("state.json", {"step": 0}),
        lambda: foreign.file("state.bin", build=lambda target: target.write_bytes(b"new")),
        lambda: foreign.restore_reference(
            "state.bin", sha256=managed.sha256, size_bytes=managed.size_bytes
        ),
        lambda: foreign.inventory(verify=True),
    ):
        with pytest.raises(ValueError, match=f"another {field}"):
            operation()
        assert before == {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}


def test_registered_corrupt_metadata_never_silently_becomes_legacy_state(tmp_path: Path) -> None:
    checkpoints = CheckpointCollection(tmp_path)
    payload = checkpoints.save_json("state.json", {"step": 9})
    record = next(checkpoints._store.records_root.glob("*.json"))
    for content in ("[]", "null", '{"owner":'):
        record.write_text(content)
        before = payload.read_bytes()
        with pytest.raises(ValueError, match="Invalid managed checkpoint"):
            checkpoints.load_json("state.json")
        with pytest.raises(ValueError, match="Invalid managed checkpoint"):
            checkpoints.save_json("state.json", {"step": 0})
        assert payload.read_bytes() == before


def test_work_results_lists_payload_not_checkpoint_bookkeeping_and_never_reads_its_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lambdaforge.work import ResultStore
    from tests.work.test_work_results import execute

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nname="checkpoint-evidence"\nversion="1"\n')
    result = execute(tmp_path)
    payload = next(result.execution_dir.glob("runs/*/checkpoints/state.json"))
    read_bytes, read_text = Path.read_bytes, Path.read_text

    def bytes_without_payload(path: Path) -> bytes:
        assert path != payload, "Work output listing must not open checkpoint payload"
        return read_bytes(path)

    def text_without_payload(path: Path, *args: object, **kwargs: object) -> str:
        assert path != payload, "Work output listing must not open checkpoint payload"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", bytes_without_payload)
    monkeypatch.setattr(Path, "read_text", text_without_payload)
    checkpoints = ResultStore().view(result.execution_id, view="outputs")["checkpoints"]
    assert [item["name"] for item in checkpoints] == ["state.json"]
    assert checkpoints[0]["scope"].startswith("Current logical Run state")


def test_checkpoint_read_does_not_advance_commit_or_change_attempt_owner(tmp_path: Path) -> None:
    first = CheckpointCollection(tmp_path, owner={"attempt_id": "attempt-0001"})
    first.configure(interval_steps=2)
    first.file("state.bin", build=lambda path: path.write_bytes(b"checkpoint"), step=4)
    before = (tmp_path / ".storage-policy.json").read_bytes()
    second = CheckpointCollection(tmp_path, owner={"attempt_id": "attempt-0002"})
    restored = second.file("state.bin", step=99)
    assert restored.metadata["owner"]["attempt_id"] == "attempt-0001"
    assert (tmp_path / ".storage-policy.json").read_bytes() == before
    assert second.should_save(step=6)


def test_retention_change_between_preview_and_apply_protects_checkpoint(tmp_path: Path) -> None:
    context, execution = runtime(tmp_path)
    context.checkpoints.save_json("state.json", {})
    (execution / "result.json").write_text('{"status":"succeeded"}')
    plan = checkpoint_plan(execution, grace_seconds=0)
    assert len(plan) == 1
    context.checkpoints.configure(retention_seconds=86400)
    assert compact_checkpoints(execution, plan) == 0
    assert checkpoint_plan(execution, grace_seconds=0) == []
    assert context.checkpoints.exists("state.json")
    # Failed evidence remains protected even when terminal grace has elapsed.
    (execution / "result.json").write_text('{"status":"failed"}')
    assert compact_checkpoints(execution, plan) == 0


@pytest.mark.parametrize(
    "policy",
    [
        {"interval_steps": True},
        {"interval_steps": -1},
        {"interval_seconds": float("nan")},
        {"retention_seconds": 0},
        {"storage_durability": "automatically-durable"},
    ],
)
def test_invalid_policy_is_rejected_before_any_write(tmp_path: Path, policy: dict) -> None:
    checkpoints = CheckpointCollection(tmp_path / "checkpoints")
    with pytest.raises(ValueError):
        checkpoints.configure(**policy)
    assert not checkpoints._root.exists()


def test_linked_managed_metadata_and_path_traversal_fail_closed(tmp_path: Path) -> None:
    checkpoints = CheckpointCollection(tmp_path / "checkpoints")
    checkpoints.save_json("state.json", {})
    record = next(checkpoints._store.records_root.glob("*.json"))
    outside = tmp_path / "outside.json"
    outside.write_bytes(record.read_bytes())
    record.unlink()
    record.symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic|Unsafe|symbol"):
        checkpoints.inventory()
    with pytest.raises(ValueError):
        checkpoints.save_json("../outside.json", {})
    assert json.loads(outside.read_text())["scope"] == "checkpoint"


def test_recovery_preview_never_reads_payload_or_claims_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = tmp_path / "execution-original"
    run_id = "run-" + "a" * 20
    root = execution / "runs" / run_id / "checkpoints"
    owner = {
        "execution_id": execution.name,
        "run_id": run_id,
        "attempt_id": "attempt-0001",
        "scientific_identity": "exact-science",
    }
    checkpoints = CheckpointCollection(root, owner=owner)
    checkpoints.configure(storage_durability="ephemeral")
    payload = checkpoints.save_json("state.json", {"step": 9}, step=9)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path == payload:
            raise AssertionError("Preview must not read checkpoint payload bytes")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    preview = checkpoint_recovery_preview(execution, run_id)
    assert preview["registered"] == 1 and preview["verified"] == 0
    assert preview["last"]["integrity"] == "not_checked"
    assert preview["last"]["ownership"] == "recorded"
    assert preview["last"]["step"] == 9
    assert preview["storage_durability"] == "ephemeral"
    assert preview["continuation"] == "application_support_and_validation_required"
    monkeypatch.setattr(Path, "open", original_open)
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    payload.unlink()
    assert checkpoint_recovery_preview(execution, run_id)["last"]["integrity"] == "invalid"


def test_recovery_preview_reports_legacy_unregistered_state(tmp_path: Path) -> None:
    execution = tmp_path / "execution-original"
    run_id = "run-" + "b" * 20
    root = execution / "runs" / run_id / "checkpoints"
    root.mkdir(parents=True)
    (root / "weights.ckpt").write_bytes(b"legacy application state")
    preview = checkpoint_recovery_preview(execution, run_id)
    assert preview["unregistered_payloads_possible"]
    assert preview["last"] is None and preview["registered"] == 0
    assert preview["continuation"] == "application_support_and_validation_required"
