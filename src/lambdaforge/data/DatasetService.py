"""First-class dataset discovery, profiling and lifecycle service."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.data.ClassificationDatasetProfiler import ClassificationDatasetProfiler
from lambdaforge.data.DatasetDeletionPlan import DatasetDeletionPlan
from lambdaforge.data.DatasetMaterializationPlan import DatasetMaterializationPlan
from lambdaforge.data.DatasetOperations import DatasetOperations
from lambdaforge.data.DatasetPlacement import DatasetPlacement
from lambdaforge.data.DatasetRecord import DatasetRecord
from lambdaforge.data.DatasetRegistry import DatasetRegistry


class DatasetService:
    """Hide catalogs/manifests/paths behind a safe dataset entity API."""

    def __init__(
        self,
        registry: DatasetRegistry | None = None,
        clusters: ClusterCatalog | None = None,
        factory: ControlPlaneFactory | None = None,
        *,
        max_parallel: int = 4,
    ) -> None:
        self.registry = registry or DatasetRegistry()
        self.clusters = clusters or ClusterCatalog.load()
        self.factory = factory or ControlPlaneFactory()
        self.max_parallel = max(1, int(max_parallel))

    def list(
        self, *, cluster: str | None = None, all_clusters: bool = False
    ) -> tuple[DatasetRecord, ...]:
        """Merge local and requested remote inventories by immutable identity."""
        records = list(self.registry.records())
        names = (
            tuple(name for name in self.clusters.names() if name != "local")
            if all_clusters
            else (cluster,)
            if cluster and cluster != "local"
            else ()
        )
        remote: list[DatasetRecord] = []
        with ThreadPoolExecutor(max_workers=min(self.max_parallel, len(names) or 1)) as executor:
            futures = {executor.submit(self._remote_records, name): name for name in names}
            for future in as_completed(futures):
                try:
                    discovered = future.result()
                    remote.extend(discovered)
                    for record in discovered:
                        self.registry.register(record)
                except Exception:
                    continue
        by_identity: dict[str, DatasetRecord] = {}
        for record in (*records, *remote):
            key = record.dataset_id
            previous = by_identity.get(key)
            if previous is None:
                by_identity[key] = record
                continue
            placements = {item.cluster: item for item in previous.placements}
            placements.update({item.cluster: item for item in record.placements})
            by_identity[key] = DatasetRecord(
                previous.name,
                previous.version,
                previous.dataset_id,
                previous.sample_count,
                previous.splits,
                previous.created_at_utc,
                tuple(placements[name] for name in sorted(placements)),
                previous.producer or record.producer,
                previous.lineage or record.lineage,
                previous.metadata or record.metadata,
            )
        values = tuple(sorted(by_identity.values(), key=lambda value: (value.name, value.version)))
        if cluster is not None:
            values = tuple(
                value
                for value in values
                if any(item.cluster == cluster for item in value.placements)
            )
        return values

    def show(self, selector: str) -> DatasetRecord:
        """Resolve a local or reconciled remote logical dataset version."""
        try:
            return self.registry.get(selector)
        except KeyError:
            matches = tuple(
                record
                for record in self.list(all_clusters=True)
                if record.name == selector or record.key == selector
            )
            if not matches:
                raise
            return sorted(matches, key=lambda value: value.created_at_utc, reverse=True)[0]

    def add(
        self,
        manifest: str | Path,
        *,
        cluster: str = "local",
        root: str | Path | None = None,
    ) -> DatasetRecord:
        """Register an existing valid DatasetArtifact and publish its placement index."""
        record = self.registry.register_artifact(manifest, cluster=cluster, root=root)
        if cluster != "local":
            self._publish_remote(record, cluster)
        return record

    def locations(self, selector: str) -> tuple[DatasetPlacement, ...]:
        return self.show(selector).placements

    def stats(
        self,
        selector: str,
        *,
        cluster: str | None = None,
        schema: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = self.show(selector)
        placement = self._placement(record, cluster)
        payload = self._operation(placement.cluster, "stats", placement.root)
        if schema:
            if placement.cluster != "local":
                payload["profile"] = "Project profilers must run where their plugin is installed."
            elif schema.get("task") == "classification":
                payload.update(
                    ClassificationDatasetProfiler().profile(Path(placement.root), record, schema)
                )
            elif schema.get("profiler") is not None:
                from lambdaforge.experiments.ObjectFactory import ObjectFactory

                profiler = ObjectFactory.build(schema["profiler"])
                method = getattr(profiler, "profile", None)
                if not callable(method):
                    raise TypeError("Dataset profiler must expose profile(root, record, schema).")
                payload.update(method(Path(placement.root), record, schema))
        return payload

    def verify(self, selector: str, *, cluster: str | None = None) -> dict[str, Any]:
        record = self.show(selector)
        placement = self._placement(record, cluster)
        return self._operation(placement.cluster, "verify", placement.root, record.dataset_id)

    def remove(self, selector: str, *, cluster: str | None = None) -> DatasetRecord | None:
        """Remove local registration only; physical bytes are never touched."""
        return self.registry.remove(selector, cluster=cluster)

    def delete(self, selector: str, *, cluster: str, apply: bool = False) -> DatasetDeletionPlan:
        """Preview or delete one exact verified registered placement."""
        record = self.show(selector)
        placement = self._placement(record, cluster)
        reasons: list[str] = []
        if self._active_consumers(record, cluster):
            reasons.append("An active LambdaForge job declares this dataset.")
        verification = self.verify(selector, cluster=cluster)
        if not verification.get("valid"):
            reasons.extend(str(item) for item in verification.get("errors", ()))
        safe = not reasons
        applied = False
        if apply:
            if not safe:
                raise RuntimeError("Dataset deletion is unsafe: " + " ".join(reasons))
            self._operation(cluster, "delete", placement.root, record.dataset_id, "--apply")
            self.registry.remove(selector, cluster=cluster)
            if cluster != "local":
                self._remove_remote(selector, cluster)
            applied = True
        return DatasetDeletionPlan(
            record.key,
            cluster,
            placement.root,
            placement.size_bytes,
            placement.file_count,
            safe,
            tuple(reasons),
            applied,
        )

    def lineage(self, selector: str) -> dict[str, Any]:
        record = self.show(selector)
        return {
            "dataset": record.key,
            "inputs": list(record.lineage),
            "producer": dict(record.producer),
        }

    def materialize(
        self,
        selector: str,
        *,
        cluster: str,
        strategy: str = "auto",
        apply: bool = False,
    ) -> DatasetMaterializationPlan:
        """Deterministically plan NOOP/REPLICATE/BUILD; never transfer large bytes silently."""
        if strategy not in {"auto", "replicate", "build"}:
            raise ValueError("Dataset materialization strategy must be auto, replicate or build.")
        record = self.show(selector)
        if any(item.cluster == cluster for item in record.placements):
            return DatasetMaterializationPlan(
                record.key, cluster, "NOOP", reason="Already present."
            )
        source = next(iter(sorted(record.placements, key=lambda value: value.cluster)), None)
        producer = str(record.producer.get("config", "")) or None
        if strategy == "build" or (source is None and producer):
            if producer is None:
                raise RuntimeError("Dataset has no registered producer for BUILD.")
            prerequisites: list[dict[str, Any]] = []
            for input_selector in record.lineage:
                try:
                    input_record = self.show(input_selector)
                except KeyError as error:
                    raise RuntimeError(
                        f"Producer input {input_selector!r} is not registered."
                    ) from error
                target_placement = next(
                    (value for value in input_record.placements if value.cluster == cluster),
                    None,
                )
                if target_placement is not None:
                    prerequisites.append(
                        {"dataset": input_record.key, "action": "NOOP", "cluster": cluster}
                    )
                    continue
                input_source = next(
                    iter(sorted(input_record.placements, key=lambda value: value.cluster)),
                    None,
                )
                if input_source is None:
                    raise RuntimeError(
                        f"Producer input {input_selector!r} has no physical placement."
                    )
                prerequisites.append(
                    {
                        "dataset": input_record.key,
                        "action": "REPLICATE",
                        "source_cluster": input_source.cluster,
                        "target_cluster": cluster,
                        "estimated_bytes": input_source.size_bytes,
                    }
                )
            plan = DatasetMaterializationPlan(
                record.key,
                cluster,
                "BUILD",
                producer=producer,
                reason="Run the registered producer after materializing its declared inputs.",
                prerequisites=tuple(prerequisites),
            )
            if apply:
                raise RuntimeError(
                    "BUILD plans require 'lambdaforge tasks run PRODUCER --on CLUSTER'; "
                    "automatic mixed lineage execution remains intentionally explicit."
                )
            return plan
        if source is None:
            raise RuntimeError("Dataset has neither a placement nor a usable producer.")
        plan = DatasetMaterializationPlan(
            record.key,
            cluster,
            "REPLICATE",
            source.cluster,
            estimated_bytes=source.size_bytes,
            reason="Copy the registered immutable dataset bytes to the target dataset root.",
            requires_controller_online=True,
        )
        if apply:
            self._replicate(record, source, cluster)
        return plan

    def replicate(
        self, selector: str, *, source: str, destination: str, apply: bool = False
    ) -> DatasetMaterializationPlan:
        record = self.show(selector)
        placement = self._placement(record, source)
        plan = DatasetMaterializationPlan(
            record.key,
            destination,
            "REPLICATE",
            source,
            estimated_bytes=placement.size_bytes,
            reason="Explicit placement replication.",
            requires_controller_online=True,
        )
        if apply:
            self._replicate(record, placement, destination)
        return plan

    def _replicate(self, record: DatasetRecord, source: DatasetPlacement, destination: str) -> None:
        target = self.clusters.get(destination)
        assert target.storage is not None
        if target.storage.dataset_root is None:
            raise RuntimeError("Target cluster must configure storage.dataset_root.")
        target_root = str(PurePosixPath(target.storage.dataset_root) / record.name / record.version)
        if source.cluster != "local":
            raise RuntimeError(
                "The built-in safe transfer provider currently needs a local source. "
                "Use a shared filesystem/provider or stage an explicit durable transfer job."
            )
        if target.transport == "local":
            shutil.copytree(source.root, target_root, dirs_exist_ok=True)
        else:
            destination_value = (
                f"{target.user + '@' if target.user else ''}{target.host}:{target_root}"
            )
            completed = subprocess.run(
                (
                    "rsync",
                    "-a",
                    "--protect-args",
                    "--",
                    source.root.rstrip("/") + "/",
                    destination_value,
                ),
                check=False,
                capture_output=True,
                text=True,
                shell=False,
            )
            if completed.returncode:
                raise RuntimeError(f"Dataset replication failed: {completed.stderr}")
        placement = DatasetPlacement(
            destination,
            target_root,
            datetime.now(timezone.utc).isoformat(),
            source.size_bytes,
            source.file_count,
            True,
        )
        updated = DatasetRecord(
            record.name,
            record.version,
            record.dataset_id,
            record.sample_count,
            record.splits,
            record.created_at_utc,
            (*record.placements, placement),
            record.producer,
            record.lineage,
            record.metadata,
        )
        self.registry.register(updated)
        self._publish_remote(updated, destination)

    def _remote_records(self, cluster: str) -> tuple[DatasetRecord, ...]:
        profile = self.clusters.get(cluster)
        assert profile.storage is not None
        transport = self.factory.transport(profile)
        python = self._python(profile.name, transport)
        registry_path = str(PurePosixPath(profile.storage.state_root) / "datasets.json")
        result = transport.run(
            (
                *profile.command_prefix,
                python,
                "-m",
                "lambdaforge.data.DatasetRegistry",
                "inventory",
                registry_path,
            ),
            timeout=30.0,
        )
        if result.returncode:
            return ()
        value = json.loads(result.stdout or "[]")
        return tuple(
            DatasetRecord.from_mapping(item) for item in value if isinstance(item, Mapping)
        )

    def _publish_remote(self, record: DatasetRecord, cluster: str) -> None:
        profile = self.clusters.get(cluster)
        assert profile.storage is not None
        transport = self.factory.transport(profile)
        remote_path = PurePosixPath(profile.storage.state_root) / "datasets.json"
        python = self._python(cluster, transport)
        with tempfile.TemporaryDirectory(prefix="lambdaforge-dataset-record-") as temporary:
            record_path = Path(temporary) / "record.json"
            record_path.write_text(json.dumps(record.to_dict(), indent=2) + "\n", encoding="utf-8")
            staged = PurePosixPath(profile.storage.state_root) / f"dataset-{os.getpid()}.json"
            transport.run(("mkdir", "-p", profile.storage.state_root))
            transport.put(record_path, str(staged))
            code = (
                "import json,sys; from lambdaforge.data import DatasetRecord,DatasetRegistry; "
                "value=json.load(open(sys.argv[1],encoding='utf-8')); "
                "DatasetRegistry(sys.argv[2]).register(DatasetRecord.from_mapping(value))"
            )
            result = transport.run(
                (*profile.command_prefix, python, "-c", code, str(staged), str(remote_path))
            )
            transport.run(("rm", "-f", str(staged)))
            if result.returncode:
                raise RuntimeError(f"Could not publish remote dataset placement: {result.stderr}")

    def _remove_remote(self, selector: str, cluster: str) -> None:
        profile = self.clusters.get(cluster)
        assert profile.storage is not None
        transport = self.factory.transport(profile)
        registry = str(PurePosixPath(profile.storage.state_root) / "datasets.json")
        code = (
            "import sys; from lambdaforge.data import DatasetRegistry; "
            "DatasetRegistry(sys.argv[1]).remove(sys.argv[2],cluster=sys.argv[3])"
        )
        result = transport.run(
            (
                *profile.command_prefix,
                self._python(cluster, transport),
                "-c",
                code,
                registry,
                selector,
                cluster,
            ),
            timeout=30.0,
        )
        if result.returncode:
            raise RuntimeError(f"Could not remove remote dataset registration: {result.stderr}")

    def _operation(
        self, cluster: str, operation: str, root: str, *arguments: str
    ) -> dict[str, Any]:
        if cluster == "local":
            if operation == "stats":
                return DatasetOperations.stats(root)
            if operation == "verify":
                return DatasetOperations.verify(root, arguments[0])
            if operation == "delete":
                return DatasetOperations.delete(root, arguments[0], apply="--apply" in arguments)
        profile = self.clusters.get(cluster)
        assert profile.storage is not None
        transport = self.factory.transport(profile)
        result = transport.run(
            (
                *profile.command_prefix,
                self._python(cluster, transport),
                "-m",
                "lambdaforge.data.DatasetOperations",
                operation,
                root,
                *arguments,
            )
        )
        if result.returncode:
            raise RuntimeError(f"Remote dataset {operation} failed: {result.stderr}")
        return json.loads(result.stdout)

    def _active_consumers(self, record: DatasetRecord, cluster: str) -> tuple[str, ...]:
        jobs = JobService(self.clusters, factory=self.factory).list(cluster=cluster, refresh=False)
        return tuple(
            job.job_id
            for job in jobs
            if not job.state.terminal and record.name in tuple(job.metadata.get("datasets", ()))
        )

    @staticmethod
    def _placement(record: DatasetRecord, cluster: str | None) -> DatasetPlacement:
        values = record.placements
        if cluster is not None:
            values = tuple(value for value in values if value.cluster == cluster)
        if not values:
            raise KeyError(f"Dataset {record.key} has no placement on {cluster!r}.")
        if cluster is None and len(values) > 1:
            local = tuple(value for value in values if value.cluster == "local")
            if local:
                return local[0]
            raise ValueError("Dataset has multiple placements; select one with --on.")
        return values[0]

    def _python(self, cluster: str, transport: Any) -> str:
        profile = self.clusters.get(cluster)
        assert profile.storage is not None
        if profile.environment != "managed":
            return profile.python
        pointer = PurePosixPath(profile.storage.state_root) / "active-environment"
        result = transport.run(("cat", str(pointer)), timeout=15.0)
        if result.returncode or not result.stdout.strip():
            legacy = PurePosixPath(profile.workspace) / ".lambdaforge" / "active-environment"
            result = transport.run(("cat", str(legacy)), timeout=15.0)
        return result.stdout.strip() if result.returncode == 0 else profile.python
