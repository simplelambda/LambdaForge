"""First-class dataset registry and conservative storage lifecycle tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lambdaforge.controlplane import (
    ClusterCatalog,
    ClusterProfile,
    CommandResult,
    ControlPlaneFactory,
)
from lambdaforge.controlplane.StorageOperations import StorageOperations
from lambdaforge.controlplane.Transport import Transport
from lambdaforge.data import (
    ClassificationDatasetProfiler,
    DatasetPlacement,
    DatasetRecord,
    DatasetRegistry,
    DatasetService,
)
from lambdaforge.preprocessing import DatasetArtifact
from lambdaforge.tasks import ArtifactDeclaration, TaskArtifact


class InventoryTransport(Transport):
    """Return one bounded synthetic remote registry inventory."""

    def __init__(self, payload: str) -> None:
        self.payload = payload

    def run(self, command, *, cwd=None, timeout=None) -> CommandResult:
        return CommandResult(0, self.payload, "")

    def put(self, source, destination) -> None:
        raise AssertionError("Inventory must be read-only.")

    def get(self, source, destination) -> None:
        raise AssertionError("Inventory must be read-only.")


class InventoryFactory(ControlPlaneFactory):
    """Route fake cluster names to deterministic inventory transports."""

    def __init__(self, payloads: dict[str, str]) -> None:
        super().__init__()
        self.payloads = payloads

    def transport(self, profile: ClusterProfile) -> Transport:
        return InventoryTransport(self.payloads[profile.name])


def dataset_manifest(root: Path) -> Path:
    output = root / "processed"
    output.mkdir(parents=True)
    (output / "a.json").write_text("{}", encoding="utf-8")
    task_artifact = TaskArtifact.materialize(ArtifactDeclaration("processed"), root)
    artifact = DatasetArtifact.create(
        name="corpus",
        version="v1",
        sample_count=1,
        splits={"train": 1},
        preprocessing_fingerprint="sha256:preprocess",
        source={"kind": "test"},
        artifacts=(task_artifact,),
    )
    manifest = root / "dataset-artifact.json"
    artifact.write_json(manifest)
    return manifest


def test_registry_registers_versions_and_remove_never_deletes_bytes(tmp_path: Path) -> None:
    root = tmp_path / "run"
    manifest = dataset_manifest(root)
    registry = DatasetRegistry(tmp_path / "datasets.json")
    record = registry.register_artifact(manifest, root=root)
    assert registry.get("corpus").dataset_id == record.dataset_id
    assert registry.get("corpus@v1").sample_count == 1
    registry.remove("corpus@v1")
    assert manifest.is_file()


def test_storage_gc_is_preview_first_reference_aware_and_excludes_datasets(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    incomplete = cache / "environments" / ".env-bad.tmp-1"
    protected = cache / "environments" / "env-live"
    dataset = tmp_path / "datasets" / "valuable"
    incomplete.mkdir(parents=True)
    protected.mkdir(parents=True)
    dataset.mkdir(parents=True)
    (incomplete / "partial").write_bytes(b"x")
    (protected / ".lambdaforge-environment.json").write_text("{}", encoding="utf-8")
    (dataset / "data").write_bytes(b"science")
    descriptor = {
        "state_root": str(tmp_path / "state"),
        "cache_root": str(cache),
        "run_root": str(tmp_path / "jobs"),
        "dataset_root": str(tmp_path / "datasets"),
    }
    preview = StorageOperations.gc(
        descriptor, {"environments": ["env-live"], "bundles": []}, apply=False
    )
    assert incomplete.exists()
    assert {item["name"] for item in preview["candidates"]} == {incomplete.name}
    StorageOperations.gc(descriptor, {"environments": ["env-live"], "bundles": []}, apply=True)
    assert not incomplete.exists()
    assert protected.exists()
    assert dataset.exists()


def test_classification_profile_requires_explicit_target_schema(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    manifest = dataset_manifest(root)
    record = DatasetRegistry(tmp_path / "registry.json").register_artifact(manifest, root=root)
    (root / "labels.csv").write_text("id,label\n1,cat\n2,dog\n3,cat\n4,\n", encoding="utf-8")
    profile = ClassificationDatasetProfiler().profile(
        root,
        record,
        {
            "task": "classification",
            "target": "label",
            "file": "labels.csv",
            "classes": ["cat", "dog"],
        },
    )
    assert profile["class_distribution"] == {"cat": 2, "dog": 1}
    assert profile["missing_targets"] == 1
    assert profile["imbalance_ratio"] == 2


def test_storage_quota_selects_oldest_unreferenced_complete_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    oldest = cache / "bundles" / "bundle-old"
    protected = cache / "bundles" / "bundle-live"
    for path in (oldest, protected):
        path.mkdir(parents=True)
        (path / "manifest.json").write_text("{}", encoding="utf-8")
        (path / "payload").write_bytes(b"x" * 64)
    descriptor = {
        "state_root": str(tmp_path / "state"),
        "cache_root": str(cache),
        "run_root": str(tmp_path / "jobs"),
        "cache_max_size": 1,
    }
    preview = StorageOperations.gc(
        descriptor, {"bundles": ["bundle-live"], "environments": []}, apply=False
    )
    assert {(item["name"], item["reason"]) for item in preview["candidates"]} == {
        ("bundle-old", "cache-quota")
    }
    assert oldest.exists() and protected.exists()


def test_storage_gc_fails_closed_during_environment_build(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / ".environment-build-active.lock").touch()
    payload = StorageOperations.gc(
        {
            "state_root": str(tmp_path / "state"),
            "cache_root": str(cache),
            "run_root": str(tmp_path / "jobs"),
        },
        {"bundles": [], "environments": []},
    )
    assert payload["candidates"] == []
    assert "environment build" in payload["blocked_reason"]


def test_storage_gc_keeps_referenced_python_runtimes_and_selects_stale_unused_one(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    state = tmp_path / "state"
    state.mkdir()
    active = cache / "runtimes" / "python-runtime-active"
    retained = cache / "runtimes" / "python-runtime-retained"
    active_job = cache / "runtimes" / "python-runtime-active-job"
    unused = cache / "runtimes" / "python-runtime-unused"
    for runtime in (active, retained, active_job, unused):
        runtime.mkdir(parents=True)
        (runtime / ".lambdaforge-python-runtime.json").write_text("{}", encoding="utf-8")
        os.utime(runtime, (1, 1))
    (state / "active-python-runtime.json").write_text(
        json.dumps({"runtime_id": active.name}), encoding="utf-8"
    )
    environment = cache / "environments" / "env-retained"
    environment.mkdir(parents=True)
    (environment / ".lambdaforge-environment.json").write_text(
        json.dumps({"environment_policy": {"python_runtime": {"runtime_id": retained.name}}}),
        encoding="utf-8",
    )

    preview = StorageOperations.gc(
        {
            "state_root": str(state),
            "cache_root": str(cache),
            "run_root": str(tmp_path / "jobs"),
            "cache_max_age": 1,
        },
        {"runtimes": [active_job.name], "environments": [], "bundles": []},
    )

    runtime_candidates = {
        item["name"] for item in preview["candidates"] if item["category"] == "runtimes"
    }
    assert runtime_candidates == {unused.name}


def test_build_materialization_plans_inputs_and_rejects_missing_lineage(tmp_path: Path) -> None:
    registry = DatasetRegistry(tmp_path / "registry.json")
    registry.register(
        DatasetRecord(
            "raw",
            "v1",
            "sha256:" + "1" * 64,
            1,
            {"train": 1},
            "2026-08-13T00:00:00+00:00",
            (DatasetPlacement("local", str(tmp_path / "raw"), "2026-08-13T00:00:00+00:00", 10, 1),),
        )
    )
    output = DatasetRecord(
        "processed",
        "v1",
        "sha256:" + "2" * 64,
        1,
        {"train": 1},
        "2026-08-13T00:00:00+00:00",
        producer={"config": "preprocess.yaml"},
        lineage=("raw@v1",),
    )
    registry.register(output)
    catalog = ClusterCatalog(
        {
            "local": ClusterProfile("local"),
            "atlas": ClusterProfile("atlas", storage={}, workspace=str(tmp_path / "atlas")),
        }
    )
    service = DatasetService(registry, catalog, InventoryFactory({"atlas": "[]"}))
    plan = service.materialize("processed@v1", cluster="atlas", strategy="build")
    assert plan.action == "BUILD"
    assert plan.prerequisites[0]["action"] == "REPLICATE"

    registry.register(
        DatasetRecord(
            "broken",
            "v1",
            "sha256:" + "3" * 64,
            0,
            {},
            "2026-08-13T00:00:00+00:00",
            producer={"config": "preprocess.yaml"},
            lineage=("absent@v1",),
        )
    )
    with pytest.raises(RuntimeError, match="not registered"):
        service.materialize("broken@v1", cluster="atlas", strategy="build")


def test_remote_inventory_merges_placements_from_multiple_clusters(tmp_path: Path) -> None:
    base = DatasetRecord(
        "corpus",
        "v1",
        "sha256:" + "4" * 64,
        2,
        {"train": 2},
        "2026-08-13T00:00:00+00:00",
    )
    payloads = {}
    profiles = {"local": ClusterProfile("local")}
    for cluster in ("atlas", "orion"):
        placement = DatasetPlacement(
            cluster, f"/data/{cluster}/corpus", "2026-08-13T00:00:00+00:00", 20, 2
        )
        record = DatasetRecord(
            base.name,
            base.version,
            base.dataset_id,
            base.sample_count,
            base.splits,
            base.created_at_utc,
            (placement,),
        )
        payloads[cluster] = json.dumps([record.to_dict()])
        profiles[cluster] = ClusterProfile(
            cluster,
            transport="ssh",
            host=cluster,
            workspace=f"/work/{cluster}",
            environment="existing",
        )
    service = DatasetService(
        DatasetRegistry(tmp_path / "registry.json"),
        ClusterCatalog(profiles),
        InventoryFactory(payloads),
    )
    records = service.list(all_clusters=True)
    assert len(records) == 1
    assert {value.cluster for value in records[0].placements} == {"atlas", "orion"}
