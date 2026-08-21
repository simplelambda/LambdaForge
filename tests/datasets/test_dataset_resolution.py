"""Unified Registry/DataCatalog resolution and remote-bundle tests for 0.7."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lambdaforge.controlplane import (
    ClusterCatalog,
    ClusterProfile,
    CommandResult,
    ControlPlaneFactory,
    Doctor,
    ExecutionBundleBuilder,
    Transport,
)
from lambdaforge.data import (
    AmbiguousDatasetVersionError,
    DataCatalog,
    DatasetPlacement,
    DatasetRecord,
    DatasetRegistry,
    DatasetResolver,
    DatasetService,
    MissingDatasetPlacementError,
)


class ManagedProfilerTransport(Transport):
    """Model a project profiler running beside remote data without a data download."""

    def __init__(self, record: DatasetRecord) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.record = record

    def run(self, command, *, cwd=None, timeout=None) -> CommandResult:
        del cwd, timeout
        values = tuple(str(value) for value in command)
        self.commands.append(values)
        if values[:1] == ("cat",):
            return CommandResult(0, "/managed/bin/python\n", "")
        if "lambdaforge.data.DatasetOperations" in values:
            operation = values[values.index("lambdaforge.data.DatasetOperations") + 1]
            if operation == "inspect":
                return CommandResult(
                    0,
                    json.dumps(
                        {
                            "exists": True,
                            "manifest_valid": True,
                            "root": self.record.placements[0].root,
                            "name": self.record.name,
                            "version": self.record.version,
                            "dataset_id": self.record.dataset_id,
                            "content_id": self.record.dataset_id,
                        }
                    ),
                    "",
                )
            return CommandResult(
                0,
                json.dumps({"member_count": 2, "size_bytes": 128, "file_count": 3}),
                "",
            )
        if "lambdaforge.data.DatasetRegistry" in values:
            return CommandResult(0, json.dumps([self.record.to_dict()]), "")
        if "-c" in values:
            return CommandResult(0, json.dumps({"project_statistic": 17}), "")
        return CommandResult(1, "", "unexpected command")

    def put(self, source, destination) -> None:
        raise AssertionError("Remote profiling must not upload dataset bytes.")

    def get(self, source, destination) -> None:
        raise AssertionError("Remote profiling must not download dataset bytes.")


class ManagedProfilerFactory(ControlPlaneFactory):
    def __init__(self, transport: ManagedProfilerTransport) -> None:
        super().__init__()
        self.instance = transport

    def transport(self, profile: ClusterProfile) -> Transport:
        del profile
        return self.instance


def _record(name: str, version: str, cluster: str, root: str, digit: str) -> DatasetRecord:
    return DatasetRecord(
        name,
        version,
        "sha256:" + digit * 64,
        2,
        {"train": 2},
        f"2026-08-14T00:00:0{digit}+00:00",
        (DatasetPlacement(cluster, root, "2026-08-14T00:00:00+00:00"),),
    )


def test_resolver_exact_ambiguous_missing_and_external(tmp_path: Path) -> None:
    registry = DatasetRegistry(tmp_path / "registry.json")
    registry.register(_record("corpus", "1", "local", "/data/v1", "1"))
    registry.register(_record("corpus", "2", "atlas", "/shared/v2", "2"))
    exact = DatasetResolver(registry, environment="atlas").resolve("dataset:corpus@2/train")
    assert exact.location.uri == "/shared/v2/train"
    assert exact.identity["dataset_id"] == "sha256:" + "2" * 64
    with pytest.raises(AmbiguousDatasetVersionError):
        DatasetResolver(registry).resolve("dataset:corpus")
    with pytest.raises(MissingDatasetPlacementError, match="datasets materialize"):
        DatasetResolver(registry, environment="atlas").resolve("dataset:corpus@1")

    catalog = DataCatalog(
        {
            "external": {
                "identity": {
                    "strategy": "version",
                    "namespace": "institution/external",
                    "version": "2026-08",
                },
                "locations": {"atlas": "/institution/external"},
            }
        }
    )
    external = DatasetResolver(registry, catalog, environment="atlas", source_dir=tmp_path).resolve(
        "dataset:external"
    )
    assert not external.managed
    assert external.location.uri == "/institution/external"


def test_remote_bundle_consumes_registry_placement_without_catalog(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="consumer"\nversion="1"\n', encoding="utf-8"
    )
    registry = DatasetRegistry(tmp_path / ".lambdaforge" / "datasets.json")
    registry.register(_record("managed", "1", "atlas", "/datasets/managed/1", "3"))
    config = tmp_path / "task.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "kind": "task",
                "schema_version": "1.0",
                "name": "consume-managed",
                "inputs": [{"name": "data", "dataset": "dataset:managed@1/files"}],
                "task": {"target": "tests.fixtures.UserTask.UserTask", "params": {"message": "ok"}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    profile = ClusterProfile(
        "atlas",
        environment="existing",
        data_environment="institution-storage",
        workspace="/work",
        storage={"dataset_root": "/datasets"},
    )
    bundle = ExecutionBundleBuilder(tmp_path / "bundles").build(config, profile)
    materialized = yaml.safe_load(bundle.config_path.read_text(encoding="utf-8"))
    catalog_path = bundle.directory / materialized["extensions"]["authoring"]["data_catalog"]
    sliced = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    descriptor = sliced["datasets"]["managed@1"]
    assert descriptor["locations"]["institution-storage"]["uri"] == "/datasets/managed/1/files"
    assert descriptor["dataset_id"] == "sha256:" + "3" * 64
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    assert manifest["config"]["inputs"][0]["dataset"] == "dataset:managed@1/files"


def test_remote_project_profiler_uses_managed_environment_without_downloading(
    tmp_path: Path,
) -> None:
    registry = DatasetRegistry(tmp_path / "registry.json")
    registry.register(_record("managed", "1", "atlas", "/datasets/managed/1", "5"))
    catalog = ClusterCatalog(
        {
            "local": ClusterProfile("local"),
            "atlas": ClusterProfile(
                "atlas",
                transport="ssh",
                host="atlas",
                workspace="/work",
                environment="managed",
                storage={"dataset_root": "/datasets"},
            ),
        }
    )
    transport = ManagedProfilerTransport(registry.get("managed@1"))
    service = DatasetService(
        registry,
        catalog,
        ManagedProfilerFactory(transport),
    )
    payload = service.stats(
        "managed@1",
        cluster="atlas",
        schema={"profiler": {"target": "consumer.profilers.DatasetProfiler"}},
    )
    assert payload["member_count"] == 2
    assert payload["project_statistic"] == 17
    profiler_command = next(command for command in transport.commands if "-c" in command)
    assert profiler_command[0] == "/managed/bin/python"


def test_doctor_uses_registry_first_resolution_without_a_catalog(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="consumer"\nversion="1"\n', encoding="utf-8"
    )
    registry = DatasetRegistry(tmp_path / ".lambdaforge" / "datasets.json")
    registry.register(_record("managed", "1", "atlas", "/datasets/managed/1", "6"))
    config = tmp_path / "task.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "kind": "task",
                "schema_version": "1.0",
                "name": "consume-managed",
                "inputs": [{"name": "data", "dataset": "dataset:managed@1"}],
                "task": {"target": "tests.fixtures.UserTask.UserTask"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    catalog = ClusterCatalog(
        {
            "atlas": ClusterProfile(
                "atlas",
                data_environment="institution-storage",
                environment="existing",
            )
        }
    )
    report = Doctor(catalog)._data_checks("atlas", config)
    assert report[0].ok
    assert "dataset:managed@1" in report[0].message
