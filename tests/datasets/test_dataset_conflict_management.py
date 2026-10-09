"""Explicit conflict resolution operates on exact identities, never scientific equivalence."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from lambdaforge.cli.DatasetCommands import DatasetCommands
from lambdaforge.cli.parser import build_parser
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ClusterStoragePolicy import ClusterStoragePolicy
from lambdaforge.data.DatasetOperations import DatasetOperations
from lambdaforge.data.DatasetPublisher import DatasetPublisher
from lambdaforge.data.DatasetRegistry import DatasetRegistry
from lambdaforge.data.DatasetService import DatasetService
from lambdaforge.data.errors import DatasetRegistryCorruptionError


@pytest.fixture
def conflict(tmp_path, monkeypatch):
    profiles, registries, records = {}, {}, {}
    for name in ("local", "gpu12", "gpu16"):
        base = tmp_path / name
        storage = ClusterStoragePolicy(
            str(base / "state"),
            str(base / "cache"),
            str(base / "jobs"),
            str(base / "datasets"),
            safety_min_free_percent=0,
        )
        profiles[name] = ClusterProfile(
            name, workspace=str(base), python=sys.executable, storage=storage
        )
        registries[name] = DatasetRegistry(base / "state" / "datasets.json")
        source = base / "input"
        source.mkdir(parents=True)
        (source / "sample.txt").write_text(f"immutable content from {name}")
        records[name] = DatasetPublisher(registries[name]).publish_members(
            "corpus",
            "6",
            [{"id": "sample", "assets": {"data": "sample.txt"}}],
            source_root=source,
            publication_root=storage.dataset_root,
            cluster=name,
            build_provenance={"work": "preprocess"},
        )
    service = DatasetService(registries["local"], ClusterCatalog(profiles))
    monkeypatch.setattr(service, "_python", lambda *_: sys.executable)
    monkeypatch.setattr(service, "_active_consumers", lambda *_: ())
    return service, records, registries, tmp_path


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_explicit_selection_and_target_deletion_preserve_other_bytes(conflict):
    service, records, registries, tmp = conflict
    chosen, unwanted = records["gpu16"], records["gpu12"]
    original = snapshot(tmp)
    assert len(service.list(all_clusters=True)) == 3
    plan = service.adopt(chosen.key, cluster="gpu16", content_id=chosen.dataset_id)
    assert plan["safe"] and not plan["applied"]
    preview = service.delete(unwanted.key, cluster="gpu12", content_id=unwanted.dataset_id)
    assert preview.safe and not preview.applied
    assert snapshot(tmp) == original

    result = service.adopt(
        chosen.key,
        cluster="gpu16",
        content_id=chosen.dataset_id,
        apply=True,
        expected_controller_id=plan["previous_content_id"],
        expected_root=plan["root"],
    )
    assert service.registry.get(chosen.key).dataset_id == chosen.dataset_id
    history = json.loads(Path(result["history"]).read_text())
    assert history["previous"] == records["local"].to_dict()
    assert registries["gpu12"].get(chosen.key).dataset_id == unwanted.dataset_id
    assert Path(records["local"].placements[0].root).is_dir()
    assert snapshot(Path(chosen.placements[0].root))

    result = service.delete(
        unwanted.key,
        cluster="gpu12",
        content_id=unwanted.dataset_id,
        expected_root=preview.root,
        apply=True,
    )
    assert result.applied
    assert not Path(unwanted.placements[0].root).exists()
    assert registries["gpu12"].records() == ()  # empty old logical entry cannot reappear
    assert service.registry.get(chosen.key).dataset_id == chosen.dataset_id
    assert DatasetOperations.verify(chosen.placements[0].root, chosen.dataset_id)["valid"]
    assert len(service.list(all_clusters=True)) == 1
    remote_history = list(
        (registries["gpu12"].path.parent / "dataset-registry-history").glob("*.json")
    )
    assert json.loads(remote_history[0].read_text())["previous"] == unwanted.to_dict()


def test_registration_only_retirement_keeps_files_and_drops_empty_record(conflict):
    service, records, registries, _ = conflict
    record = records["gpu12"]
    result = service.retire(record.key, cluster="gpu12", content_id=record.dataset_id, apply=True)
    assert result["applied"]
    assert registries["gpu12"].records() == ()
    assert DatasetOperations.verify(record.placements[0].root, record.dataset_id)["valid"]
    assert service.registry.get(record.key).dataset_id == records["local"].dataset_id
    with pytest.raises(Exception, match="exact requested content ID"):
        service.retire(record.key, cluster="gpu12", content_id=record.dataset_id, apply=True)


def test_wrong_hash_and_unpinned_deletion_cannot_bypass_conflict(conflict):
    service, records, _, tmp = conflict
    original = snapshot(tmp)
    preview = service.delete("corpus@6", cluster="gpu12")
    assert not preview.safe
    with pytest.raises(Exception, match="exact requested content ID"):
        service.delete(
            "corpus@6", cluster="gpu12", content_id=records["gpu16"].dataset_id, apply=True
        )
    assert snapshot(tmp) == original


def test_missing_copy_deletion_retires_stale_target_and_not_the_chosen_reference(conflict):
    service, records, registries, _ = conflict
    unwanted = records["gpu12"]
    root = Path(unwanted.placements[0].root)
    # Move the test-owned immutable copy aside, simulating prior external removal.
    root.rename(root.with_name("retained-test-copy"))
    plan = service.delete(unwanted.key, cluster="gpu12", content_id=unwanted.dataset_id)
    assert plan.safe and plan.action == "CLEAN_STALE_REGISTRATION"
    assert service.delete(
        unwanted.key, cluster="gpu12", content_id=unwanted.dataset_id, apply=True
    ).applied
    assert registries["gpu12"].records() == ()
    assert service.registry.get(unwanted.key).dataset_id == records["local"].dataset_id


def test_legacy_external_copy_allows_index_retirement_but_not_physical_deletion(conflict):
    service, records, registries, tmp = conflict
    record = records["gpu12"]
    root = Path(record.placements[0].root)
    outside = tmp / "researcher-owned-data"
    root.rename(outside)
    moved = replace(record, placements=(replace(record.placements[0], root=str(outside)),))
    registries["gpu12"].choose(moved, expected_previous_id=record.dataset_id)
    plan = service.delete(record.key, cluster="gpu12", content_id=record.dataset_id)
    assert not plan.safe
    with pytest.raises(Exception, match="unsafe"):
        service.delete(record.key, cluster="gpu12", content_id=record.dataset_id, apply=True)
    service.retire(record.key, cluster="gpu12", content_id=record.dataset_id, apply=True)
    assert outside.is_dir() and registries["gpu12"].records() == ()


def test_unreachable_target_or_corrupt_registry_is_never_forgotten(conflict, monkeypatch):
    service, records, registries, _ = conflict
    record = records["gpu12"]
    original = registries["gpu12"].path.read_bytes()

    def offline(*_args, **_kwargs):
        raise OSError("target offline")

    with monkeypatch.context() as scoped:
        scoped.setattr(service, "_remote_records", offline)
        for operation in (service.retire, service.adopt):
            with pytest.raises(Exception, match="refused"):
                operation(record.key, cluster="gpu12", content_id=record.dataset_id, apply=True)
    assert registries["gpu12"].path.read_bytes() == original
    registries["gpu12"].path.write_text("invalid JSON")
    with pytest.raises(DatasetRegistryCorruptionError, match="corrupt"):
        service.retire(record.key, cluster="gpu12", content_id=record.dataset_id, apply=True)
    assert registries["gpu12"].path.read_text() == "invalid JSON"


def test_remote_success_controller_failure_reports_partial_retirement(conflict, monkeypatch):
    service, records, registries, _ = conflict
    chosen = records["gpu16"]
    service.adopt(chosen.key, cluster="gpu16", content_id=chosen.dataset_id, apply=True)
    monkeypatch.setattr(
        service.registry,
        "retire",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("controller disk unavailable")),
    )
    with pytest.raises(Exception, match="Target retirement completed"):
        service.retire(chosen.key, cluster="gpu16", content_id=chosen.dataset_id, apply=True)
    assert registries["gpu16"].records() == ()
    assert DatasetOperations.verify(chosen.placements[0].root, chosen.dataset_id)["valid"]


def test_corrupt_bytes_cannot_be_adopted_and_no_implicit_equivalence(conflict):
    service, records, _, _ = conflict
    chosen = records["gpu16"]
    root = Path(chosen.placements[0].root)
    next((root / "assets").rglob("data-*")).write_text("corruption")
    with pytest.raises(Exception, match="exact byte verification"):
        service.adopt(chosen.key, cluster="gpu16", content_id=chosen.dataset_id, apply=True)
    assert service.registry.get(chosen.key).dataset_id == records["local"].dataset_id


def test_absent_copy_can_be_retired_without_deleting_any_other_root(conflict):
    service, records, registries, _ = conflict
    record = records["gpu12"]
    # Simulate a missing placement using an index pointing at an absent owned path.
    missing = replace(
        record,
        placements=(replace(record.placements[0], root=record.placements[0].root + "-missing"),),
    )
    registries["gpu12"].choose(missing, expected_previous_id=record.dataset_id)
    plan = service.delete(record.key, cluster="gpu12", content_id=record.dataset_id)
    # Canonical bounded discovery finds original bytes, so index-only retirement is explicit.
    assert not plan.safe
    service.retire(record.key, cluster="gpu12", content_id=record.dataset_id, apply=True)
    assert Path(record.placements[0].root).exists()
    assert registries["gpu12"].records() == ()


def test_active_consumers_block_exact_retirement_and_deletion(conflict, monkeypatch):
    service, records, _, tmp = conflict
    record = records["gpu12"]
    monkeypatch.setattr(service, "_active_consumers", lambda *_: ("active-job",))
    original = snapshot(tmp)
    assert not service.retire(record.key, cluster="gpu12", content_id=record.dataset_id)["safe"]
    assert not service.delete(record.key, cluster="gpu12", content_id=record.dataset_id).safe
    with pytest.raises(Exception, match="active consumers"):
        service.retire(record.key, cluster="gpu12", content_id=record.dataset_id, apply=True)
    assert snapshot(tmp) == original


def test_changed_reference_or_target_root_requires_new_preview(conflict):
    service, records, registries, _ = conflict
    chosen = records["gpu16"]
    with pytest.raises(Exception, match="Reference changed"):
        service.adopt(
            chosen.key,
            cluster="gpu16",
            content_id=chosen.dataset_id,
            expected_controller_id="sha256:" + "0" * 64,
            apply=True,
        )
    with pytest.raises(Exception, match="root changed"):
        service.delete(
            chosen.key,
            cluster="gpu16",
            content_id=chosen.dataset_id,
            expected_root="/not-the-preview-root",
            apply=True,
        )
    assert registries["gpu16"].get(chosen.key) == chosen


def test_registry_compare_and_swap_serializes_concurrent_retirement(conflict):
    _, records, registries, _ = conflict
    record = records["gpu12"]

    def retire():
        try:
            registries["gpu12"].retire(record.to_dict(), cluster="gpu12")
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: retire(), range(2)))
    assert sorted(results) == [False, True]
    assert (
        len(list((registries["gpu12"].path.parent / "dataset-registry-history").glob("*.json")))
        == 1
    )


def test_audit_backup_precedes_write_and_symlinked_history_is_refused(conflict, monkeypatch):
    service, records, _, tmp = conflict
    chosen = records["gpu16"]
    previous = service.registry.path.read_bytes()
    monkeypatch.setattr(
        service.registry, "_write", lambda _: (_ for _ in ()).throw(OSError("disk error"))
    )
    with pytest.raises(OSError, match="disk error"):
        service.registry.choose(chosen, expected_previous_id=records["local"].dataset_id)
    assert service.registry.path.read_bytes() == previous
    assert list(service.registry.path.parent.glob("dataset-registry-history/*.json"))
    registry = DatasetRegistry(tmp / "new-state" / "datasets.json")
    registry.path.parent.mkdir()
    (registry.path.parent / "dataset-registry-history").symlink_to(tmp, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        registry.choose(chosen, expected_previous_id=None)
    assert not registry.path.exists()


@pytest.mark.parametrize("operation", ["adopt", "remove", "delete"])
def test_cli_exact_operations_default_to_read_only_preview(
    conflict, monkeypatch, capsys, operation
):
    service, records, _, tmp = conflict
    monkeypatch.setattr("lambdaforge.cli.DatasetCommands.DatasetService", lambda **_: service)
    args = build_parser().parse_args(
        [
            "datasets",
            operation,
            "corpus@6",
            "--on",
            "gpu16",
            "--content-id",
            records["gpu16"].dataset_id,
            "--json",
        ]
    )
    original = snapshot(tmp)
    assert DatasetCommands.run(args) == 0
    assert json.loads(capsys.readouterr().out)["applied"] is False
    assert snapshot(tmp) == original
