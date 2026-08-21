"""Local consistency and safe-convergence contracts for managed dataset placements."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from lambdaforge.controlplane import (
    ClusterCatalog,
    ClusterProfile,
    CommandResult,
    ControlPlaneFactory,
    JobService,
    JobStore,
    Transport,
)
from lambdaforge.data import (
    DatasetPlacement,
    DatasetPlacementState,
    DatasetRecord,
    DatasetRegistry,
    DatasetRegistryCorruptionError,
    DatasetResolutionError,
    DatasetService,
    OfflineClusterError,
    UnsafeDatasetOperationError,
)
from lambdaforge.data.DatasetOperations import DatasetOperations
from lambdaforge.execution import ResourceRequest
from lambdaforge.preprocessing import DatasetArtifact
from lambdaforge.tasks import ArtifactDeclaration, TaskArtifact


class LocalRemoteTransport(Transport):
    """Model one remote registry and managed root without network access."""

    def __init__(self) -> None:
        self.offline = False
        self.inventory_error = False
        self.fail_remove_once = False

    def run(self, command, *, cwd=None, timeout=None) -> CommandResult:
        del cwd, timeout
        values = tuple(str(value) for value in command)
        if self.offline:
            raise OSError("synthetic cluster outage")
        if "lambdaforge.data.DatasetRegistry" in values:
            if self.inventory_error:
                return CommandResult(1, "", "synthetic inventory execution failure")
            registry = values[-1]
            try:
                payload = [record.to_dict() for record in DatasetRegistry(registry).records()]
            except DatasetRegistryCorruptionError as error:
                return CommandResult(1, "", f"DatasetRegistryCorruptionError: {error}")
            return CommandResult(0, json.dumps(payload), "")
        if "lambdaforge.data.DatasetOperations" in values:
            offset = values.index("lambdaforge.data.DatasetOperations")
            operation, root, *arguments = values[offset + 1 :]
            if operation == "inspect":
                payload = DatasetOperations.inspect(root)
            elif operation == "stats":
                payload = DatasetOperations.stats(root)
            elif operation == "verify":
                payload = DatasetOperations.verify(root, arguments[0])
            elif operation == "delete":
                payload = DatasetOperations.delete(
                    root,
                    arguments[0],
                    managed_root=arguments[1],
                    apply="--apply" in arguments[2:],
                )
            else:
                raise AssertionError(f"Unexpected dataset operation: {operation}")
            return CommandResult(0, json.dumps(payload), "")
        if "-c" in values:
            code = values[values.index("-c") + 1]
            if ".register(" in code:
                staged, registry = values[-2:]
                record = DatasetRecord.from_mapping(
                    json.loads(Path(staged).read_text(encoding="utf-8"))
                )
                DatasetRegistry(registry).register(record)
                return CommandResult(0, "", "")
            if ".discard(" in code:
                if self.fail_remove_once:
                    self.fail_remove_once = False
                    return CommandResult(1, "", "synthetic registry update failure")
                registry, selector, cluster = values[-3:]
                DatasetRegistry(registry).discard(selector, cluster=cluster)
                return CommandResult(0, "", "")
        if values[:2] == ("mkdir", "-p"):
            Path(values[-1]).mkdir(parents=True, exist_ok=True)
            return CommandResult(0, "", "")
        if values[:2] == ("rm", "-f"):
            Path(values[-1]).unlink(missing_ok=True)
            return CommandResult(0, "", "")
        raise AssertionError(f"Unexpected synthetic remote command: {values}")

    def put(self, source, destination) -> None:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    def get(self, source, destination) -> None:
        raise AssertionError("Dataset reconciliation never downloads dataset bytes.")


class LocalRemoteFactory(ControlPlaneFactory):
    def __init__(self, transport: LocalRemoteTransport) -> None:
        super().__init__()
        self.instance = transport

    def transport(self, profile: ClusterProfile) -> Transport:
        del profile
        return self.instance


def _artifact(root: Path, *, name: str = "wisdom-dna", version: str = "1") -> DatasetRecord:
    root.mkdir(parents=True)
    processed = root / "processed"
    processed.mkdir()
    (processed / "sequence.json").write_text('{"sequence":"ACGT"}', encoding="utf-8")
    task_artifact = TaskArtifact.materialize(ArtifactDeclaration("processed"), root)
    artifact = DatasetArtifact.create(
        name=name,
        version=version,
        sample_count=1,
        splits={"all": 1},
        preprocessing_fingerprint="sha256:fixture",
        source={"kind": "test"},
        artifacts=(task_artifact,),
    )
    artifact.write_json(root / "dataset-artifact.json")
    return DatasetRegistry(root.parent / "scratch-registry.json").register_artifact(
        root / "dataset-artifact.json", cluster="cluster-a", root=root
    )


def _without_placements(record: DatasetRecord) -> DatasetRecord:
    return DatasetRecord(
        record.name,
        record.version,
        record.dataset_id,
        record.sample_count,
        record.splits,
        record.created_at_utc,
        producer=record.producer,
        lineage=record.lineage,
        metadata=record.metadata,
        build_id=record.build_id,
        index=record.index,
        partitions=record.partitions,
        target_schema=record.target_schema,
        global_assets=record.global_assets,
        lineage_graph=record.lineage_graph,
    )


def _service(
    tmp_path: Path,
) -> tuple[
    DatasetService,
    DatasetRegistry,
    DatasetRegistry,
    LocalRemoteTransport,
    Path,
]:
    managed_root = tmp_path / "remote-datasets"
    state_root = tmp_path / "remote-state"
    local = DatasetRegistry(tmp_path / "controller-registry.json")
    remote = DatasetRegistry(state_root / "datasets.json")
    transport = LocalRemoteTransport()
    catalog = ClusterCatalog(
        {
            "local": ClusterProfile("local", python=sys.executable),
            "cluster-a": ClusterProfile(
                "cluster-a",
                transport="ssh",
                host="synthetic",
                workspace=str(tmp_path / "remote-work"),
                python=sys.executable,
                environment="existing",
                storage={
                    "state_root": str(state_root),
                    "dataset_root": str(managed_root),
                },
            ),
        }
    )
    return (
        DatasetService(local, catalog, LocalRemoteFactory(transport)),
        local,
        remote,
        transport,
        managed_root,
    )


def test_stale_controller_valid_target_has_one_consistent_read_only_reality(
    tmp_path: Path,
) -> None:
    service, local, remote, _, managed_root = _service(tmp_path)
    record = _artifact(managed_root / "wisdom-dna" / "1" / "published")
    local.register(_without_placements(record))
    remote.register(record)
    local_before = local.path.read_bytes()
    remote_before = remote.path.read_bytes()

    resolution = service.resolve_placement("wisdom-dna@1", "cluster-a")
    shown = service.describe("wisdom-dna@1", cluster="cluster-a")
    verified = service.verify("wisdom-dna@1", cluster="cluster-a")
    materialized = service.materialize("wisdom-dna@1", cluster="cluster-a")
    deletion = service.delete("wisdom-dna@1", cluster="cluster-a")

    assert resolution.state is DatasetPlacementState.AVAILABLE
    assert shown["placement_consistency"]["state"] == "available"
    assert verified["valid"] and verified["placement_state"] == "available"
    assert materialized.action == "NOOP"
    assert deletion.safe and deletion.action == "DELETE_PLACEMENT"
    assert Path(record.placements[0].root).is_dir()
    assert local.path.read_bytes() == local_before
    assert remote.path.read_bytes() == remote_before

    service.remove(record.key, cluster="cluster-a")
    assert Path(record.placements[0].root).is_dir()
    assert not local.get(record.key).placements
    assert not remote.get(record.key).placements


def test_discovered_missing_conflict_and_offline_states_fail_safely(tmp_path: Path) -> None:
    service, local, remote, transport, managed_root = _service(tmp_path)
    staging = managed_root / "staging"
    record = _artifact(staging)
    expected = (
        managed_root / record.name / record.version / record.dataset_id.removeprefix("sha256:")[:16]
    )
    expected.parent.mkdir(parents=True)
    staging.rename(expected)
    exact = DatasetPlacement("cluster-a", expected.as_posix(), record.created_at_utc, verified=True)
    record = DatasetService._with_placement(record, exact)
    logical = _without_placements(record)
    local.register(logical)
    remote.register(logical)

    discovered = service.resolve_placement(record.key, "cluster-a")
    assert discovered.state is DatasetPlacementState.DISCOVERED_UNREGISTERED
    assert service.materialize(record.key, cluster="cluster-a").action == "RECONCILE"
    assert not service.delete(record.key, cluster="cluster-a").safe
    service.reconcile(record.key, cluster="cluster-a", apply=True)
    assert (
        service.resolve_placement(record.key, "cluster-a").state is DatasetPlacementState.AVAILABLE
    )

    shutil.rmtree(expected)
    missing = service.resolve_placement(record.key, "cluster-a")
    assert missing.state is DatasetPlacementState.REGISTERED_BUT_MISSING
    cleanup = service.delete(record.key, cluster="cluster-a")
    assert cleanup.safe and cleanup.action == "CLEAN_STALE_REGISTRATION"

    conflicting = _artifact(expected, name="different", version="1")
    del conflicting
    conflict = service.resolve_placement(record.key, "cluster-a")
    assert conflict.state is DatasetPlacementState.CONFLICT
    with pytest.raises(UnsafeDatasetOperationError):
        service.materialize(record.key, cluster="cluster-a")

    shutil.rmtree(expected)
    transport.offline = True
    offline = service.resolve_placement(record.key, "cluster-a")
    assert offline.state is DatasetPlacementState.UNREACHABLE
    assert not service.delete(record.key, cluster="cluster-a").safe
    with pytest.raises(OfflineClusterError):
        service.materialize(record.key, cluster="cluster-a")


def test_corrupt_registry_is_never_treated_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "datasets.json"
    path.write_text('{"datasets":', encoding="utf-8")

    with pytest.raises(DatasetRegistryCorruptionError, match="corrupt"):
        DatasetRegistry(path).records()

    assert path.read_text(encoding="utf-8") == '{"datasets":'

    path.write_text(
        json.dumps({"dataset_registry_version": 1, "datasets": {"broken@1": {}}}),
        encoding="utf-8",
    )
    with pytest.raises(DatasetRegistryCorruptionError, match="invalid"):
        DatasetRegistry(path).register(_without_placements(_artifact(tmp_path / "valid")))


def test_discovery_degrades_but_targeted_corruption_and_outage_propagate(
    tmp_path: Path,
) -> None:
    service, local, remote, transport, managed_root = _service(tmp_path)
    record = _artifact(managed_root / "wisdom-dna" / "1" / "published")
    local.register(record)
    remote.register(record)

    transport.offline = True
    assert service.list(all_clusters=True)[0].key == record.key
    assert service.discovery_warnings and "cluster-a" in service.discovery_warnings[0]
    with pytest.raises(OfflineClusterError):
        service.list(cluster="cluster-a")

    transport.offline = False
    transport.inventory_error = True
    with pytest.raises(DatasetResolutionError, match="without proving absence"):
        service.list(cluster="cluster-a")
    assert (
        service.resolve_placement(record.key, "cluster-a").state
        is DatasetPlacementState.UNREACHABLE
    )

    transport.inventory_error = False
    remote.path.write_text('{"datasets":', encoding="utf-8")
    with pytest.raises(DatasetRegistryCorruptionError):
        service.resolve_placement(record.key, "cluster-a")


def test_partial_delete_failures_converge_without_deleting_unrelated_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, local, remote, transport, managed_root = _service(tmp_path)
    root = managed_root / "wisdom-dna" / "1" / "published"
    record = _artifact(root)
    local.register(record)
    remote.register(record)
    unrelated = managed_root / "unrelated"
    unrelated.mkdir()
    (unrelated / "keep").write_text("safe", encoding="utf-8")

    transport.fail_remove_once = True
    with pytest.raises(RuntimeError, match="remote dataset registration"):
        service.delete(record.key, cluster="cluster-a", apply=True)
    assert not root.exists() and unrelated.is_dir()
    assert local.get(record.key).placements

    retry = service.delete(record.key, cluster="cluster-a")
    assert retry.action == "CLEAN_STALE_REGISTRATION" and retry.safe
    service.delete(record.key, cluster="cluster-a", apply=True)
    assert not local.get(record.key).placements
    assert not remote.get(record.key).placements

    second_root = managed_root / "wisdom-dna" / "2" / "published"
    second = _artifact(second_root, version="2")
    local.register(second)
    remote.register(second)
    original_discard = local.discard
    failed = False

    def fail_once(selector: str, *, cluster: str | None = None):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("synthetic controller registry failure")
        return original_discard(selector, cluster=cluster)

    monkeypatch.setattr(local, "discard", fail_once)
    with pytest.raises(OSError, match="controller registry"):
        service.delete(second.key, cluster="cluster-a", apply=True)
    assert not second_root.exists()
    service.delete(second.key, cluster="cluster-a", apply=True)
    assert not local.get(second.key).placements
    assert unrelated.is_dir()


def test_delete_blocks_exact_active_consumer_and_rejects_path_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "controller-state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    service, local, remote, _, managed_root = _service(tmp_path)
    root = managed_root / "wisdom-dna" / "1" / "published"
    record = _artifact(root)
    local.register(record)
    remote.register(record)
    JobService(service.clusters, JobStore()).reserve(
        cluster="cluster-a",
        resources=ResourceRequest(),
        config_path=tmp_path / "experiment.yaml",
        metadata={"datasets": [record.key], "dataset_ids": [record.dataset_id]},
        job_type="experiment",
    )

    blocked = service.delete(record.key, cluster="cluster-a")
    assert not blocked.safe and blocked.active_consumers

    outside = tmp_path / "outside"
    escaped = _artifact(outside, name="escaped")
    local.register(escaped)
    remote.register(escaped)
    unsafe = service.delete(escaped.key, cluster="cluster-a")
    assert not unsafe.safe
    assert any("outside" in reason for reason in unsafe.reasons)


def test_delete_rejects_symlink_root_even_when_manifest_identity_matches(
    tmp_path: Path,
) -> None:
    service, local, remote, _, managed_root = _service(tmp_path)
    physical = tmp_path / "physical"
    record = _artifact(physical, name="linked")
    linked = managed_root / "linked" / "1" / "published"
    linked.parent.mkdir(parents=True)
    linked.symlink_to(physical, target_is_directory=True)
    placement = DatasetPlacement(
        "cluster-a", linked.as_posix(), record.created_at_utc, verified=True
    )
    indexed = DatasetService._with_placement(_without_placements(record), placement)
    local.register(indexed)
    remote.register(indexed)

    resolution = service.resolve_placement(indexed.key, "cluster-a")
    preview = service.delete(indexed.key, cluster="cluster-a")

    assert resolution.state is DatasetPlacementState.CONFLICT
    assert not preview.safe
    assert physical.is_dir() and linked.is_symlink()
