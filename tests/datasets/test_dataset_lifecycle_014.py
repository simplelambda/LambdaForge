"""Focused regressions for logical summaries and convergent DatasetVersion deletion."""

from __future__ import annotations

from pathlib import Path

import pytest

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ClusterStoragePolicy import ClusterStoragePolicy
from lambdaforge.data.DatasetPlacement import DatasetPlacement
from lambdaforge.data.DatasetPublisher import DatasetPublisher
from lambdaforge.data.DatasetRecord import DatasetRecord
from lambdaforge.data.DatasetRegistry import DatasetRegistry
from lambdaforge.data.DatasetResolution import DatasetPlacementResolution, DatasetPlacementState
from lambdaforge.data.DatasetService import DatasetService
from lambdaforge.data.errors import InvalidDatasetPublicationError
from lambdaforge.data.index import DatasetIndex, DatasetMember
from lambdaforge.diagnostics import LambdaForgeError


def test_logical_summary_crosses_binary_targets_with_splits(tmp_path: Path) -> None:
    index = DatasetIndex.write(
        tmp_path / "members.jsonl",
        (
            DatasetMember("a", {"split": "train"}, {"label": 1}),
            DatasetMember("b", {"split": "train"}, {"label": 0}),
            DatasetMember("c", {"split": "validation"}, {"label": 1}),
        ),
    )

    summary = index.summary()

    assert summary["partitions"]["split"] == {"train": 2, "validation": 1}
    assert summary["partition_targets"]["split"]["train"]["label"] == {"0": 1, "1": 1}
    assert summary["partition_targets"]["split"]["validation"]["label"] == {"1": 1}


def test_delete_version_forgets_a_registered_but_missing_local_dataset(
    tmp_path: Path, monkeypatch
) -> None:
    managed_root = tmp_path / "managed-datasets"
    missing = managed_root / "wisdom" / "4" / "sha256"
    registry = DatasetRegistry(tmp_path / "datasets.json")
    registry.register(
        DatasetRecord(
            "wisdom",
            "4",
            "sha256:" + "a" * 64,
            10,
            {"train": 8, "test": 2},
            "2026-09-02T00:00:00+00:00",
            (
                DatasetPlacement(
                    "local",
                    str(missing),
                    "2026-09-02T00:00:00+00:00",
                    1024,
                    3,
                    True,
                ),
            ),
        )
    )
    storage = ClusterStoragePolicy(
        str(tmp_path / "state"),
        str(tmp_path / "cache"),
        str(tmp_path / "jobs"),
        str(managed_root),
    )
    service = DatasetService(
        registry,
        ClusterCatalog(
            {"local": ClusterProfile("local", workspace=str(tmp_path), storage=storage)}
        ),
    )
    monkeypatch.setattr(service, "_active_consumers", lambda _record, _cluster: ())

    preview = service.delete_version("wisdom@4")
    assert preview["safe"] is True
    assert preview["placements"][0]["action"] == "CLEAN_STALE_REGISTRATION"

    applied = service.delete_version("wisdom@4", apply=True)
    assert applied["applied"] is True
    assert registry.records() == ()
    assert service.delete_version("wisdom@4", apply=True)["already_deleted"] is True


def test_delete_version_cleans_missing_remote_placement_and_both_indexes(
    tmp_path: Path, monkeypatch
) -> None:
    registry = DatasetRegistry(tmp_path / "datasets.json")
    placement = DatasetPlacement(
        "gpu4",
        "/managed/wisdom/4/sha256",
        "2026-09-02T00:00:00+00:00",
        2048,
        5,
        True,
    )
    record = registry.register(
        DatasetRecord(
            "wisdom",
            "4",
            "sha256:" + "b" * 64,
            20,
            {"train": 16, "test": 4},
            "2026-09-02T00:00:00+00:00",
            (placement,),
        )
    )
    storage = ClusterStoragePolicy(
        "/remote/state",
        "/remote/cache",
        "/remote/jobs",
        "/managed",
    )
    service = DatasetService(
        registry,
        ClusterCatalog(
            {
                "gpu4": ClusterProfile(
                    "gpu4",
                    workspace="/remote",
                    storage=storage,
                )
            }
        ),
    )
    resolution = DatasetPlacementResolution(
        record,
        "gpu4",
        DatasetPlacementState.REGISTERED_BUT_MISSING,
        placement,
        placement,
        placement,
        {"exists": False},
        "The indexed directory is already absent.",
        "remove_stale_registration",
    )
    remote_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(service, "resolve_placement", lambda _selector, _cluster: resolution)
    monkeypatch.setattr(service, "_active_consumers", lambda _record, _cluster: ())
    monkeypatch.setattr(
        service,
        "_remove_remote",
        lambda selector, cluster: remote_calls.append(("placement", f"{selector}:{cluster}")),
    )
    monkeypatch.setattr(
        service,
        "_forget_remote",
        lambda selector, cluster: remote_calls.append(("version", f"{selector}:{cluster}")),
    )

    applied = service.delete_version("wisdom@4", apply=True)

    assert applied["applied"] is True
    assert remote_calls == [
        ("placement", "wisdom@4:gpu4"),
        ("version", "wisdom@4:gpu4"),
    ]
    assert registry.records() == ()


def test_publication_conflict_does_not_leave_unregistered_dataset_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    asset = source / "payload.bin"
    asset.write_bytes(b"first identity")
    publication_root = tmp_path / "published"
    registry = DatasetRegistry(tmp_path / "datasets.json")
    publisher = DatasetPublisher(registry)
    members = ({"id": "sample", "assets": {"data": "payload.bin"}},)

    original = publisher.publish_members(
        "wisdom",
        "6",
        members,
        source_root=source,
        publication_root=publication_root,
        build_provenance={"work": "preprocess"},
    )
    asset.write_bytes(b"different identity")

    with pytest.raises(InvalidDatasetPublicationError, match="new dataset version"):
        publisher.publish_members(
            "wisdom",
            "6",
            members,
            source_root=source,
            publication_root=publication_root,
            build_provenance={"work": "preprocess"},
        )

    version_root = publication_root / "wisdom" / "6"
    assert {path.name for path in version_root.iterdir()} == {
        original.dataset_id.removeprefix("sha256:")[:16]
    }


def test_publication_rolls_back_bytes_when_final_registration_fails(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload.bin").write_bytes(b"complete")
    publication_root = tmp_path / "published"
    registry = DatasetRegistry(tmp_path / "datasets.json")

    def reject_registration(*_args, **_kwargs):
        raise RuntimeError("simulated registry race")

    monkeypatch.setattr(registry, "register_artifact", reject_registration)

    with pytest.raises(RuntimeError, match="registry race"):
        DatasetPublisher(registry).publish_members(
            "wisdom",
            "7",
            ({"id": "sample", "assets": {"data": "payload.bin"}},),
            source_root=source,
            publication_root=publication_root,
            build_provenance={},
        )

    assert list((publication_root / "wisdom" / "7").iterdir()) == []


def test_reconcile_removes_only_a_missing_conflicting_remote_registration(
    tmp_path: Path, monkeypatch
) -> None:
    controller = DatasetRecord(
        "wisdom",
        "6",
        "sha256:" + "a" * 64,
        10,
        {},
        "2026-09-10T00:00:00+00:00",
        (
            DatasetPlacement(
                "gpu12",
                "/datasets/wisdom/6/aaaaaaaaaaaaaaaa",
                "2026-09-10T00:00:00+00:00",
            ),
        ),
    )
    target = DatasetRecord(
        "wisdom",
        "6",
        "sha256:" + "b" * 64,
        10,
        {},
        "2026-09-10T00:00:00+00:00",
        (
            DatasetPlacement(
                "gpu16",
                "/datasets/wisdom/6/bbbbbbbbbbbbbbbb",
                "2026-09-10T00:00:00+00:00",
            ),
        ),
    )
    registry = DatasetRegistry(tmp_path / "datasets.json")
    registry.register(controller)
    storage = ClusterStoragePolicy(
        "/remote/state",
        "/remote/cache",
        "/remote/jobs",
        "/datasets",
    )
    service = DatasetService(
        registry,
        ClusterCatalog(
            {
                "gpu12": ClusterProfile("gpu12", workspace="/remote", storage=storage),
                "gpu16": ClusterProfile("gpu16", workspace="/remote", storage=storage),
            }
        ),
    )
    forgotten: list[tuple[str, str]] = []
    monkeypatch.setattr(service, "_remote_records", lambda _cluster: (target,))
    monkeypatch.setattr(
        service,
        "_operation",
        lambda _cluster, operation, root, *_args: {
            "exists": False,
            "operation": operation,
            "root": root,
        },
    )
    monkeypatch.setattr(
        service,
        "_forget_remote",
        lambda selector, cluster: forgotten.append((selector, cluster)),
    )

    preview = service.reconcile("wisdom@6", cluster="gpu16")
    assert preview["safe"] is True
    assert preview["action"] == "REMOVE_STALE_CONFLICTING_REGISTRATION"

    applied = service.reconcile("wisdom@6", cluster="gpu16", apply=True)
    assert applied["applied"] is True
    assert forgotten == [("wisdom@6", "gpu16")]
    assert registry.get("wisdom@6").dataset_id == controller.dataset_id


def test_remote_replication_refuses_an_implicit_controller_relay(tmp_path: Path) -> None:
    storage = ClusterStoragePolicy("/state", "/cache", "/jobs", "/datasets")
    service = DatasetService(
        DatasetRegistry(tmp_path / "datasets.json"),
        ClusterCatalog({"gpu16": ClusterProfile("gpu16", workspace="/remote", storage=storage)}),
    )
    placement = DatasetPlacement(
        "gpu12",
        "/datasets/wisdom/6/aaaaaaaaaaaaaaaa",
        "2026-09-10T00:00:00+00:00",
    )
    record = DatasetRecord(
        "wisdom",
        "6",
        "sha256:" + "a" * 64,
        10,
        {},
        "2026-09-10T00:00:00+00:00",
        (placement,),
    )

    with pytest.raises(LambdaForgeError, match="durable data-transfer provider"):
        service._replicate(record, placement, "gpu16")
