"""First-class dataset discovery, profiling and lifecycle service."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.data.ClassificationDatasetProfiler import ClassificationDatasetProfiler
from lambdaforge.data.DatasetBuildService import DatasetBuildService
from lambdaforge.data.DatasetDeletionPlan import DatasetDeletionPlan
from lambdaforge.data.DatasetMaterializationPlan import DatasetMaterializationPlan
from lambdaforge.data.DatasetOperations import DatasetOperations
from lambdaforge.data.DatasetPlacement import DatasetPlacement
from lambdaforge.data.DatasetRecord import DatasetRecord
from lambdaforge.data.DatasetRegistry import DatasetRegistry
from lambdaforge.data.errors import (
    AmbiguousDatasetVersionError,
    MissingDatasetPlacementError,
    MissingDatasetRecipeError,
    MissingManagedEnvironmentError,
    OfflineClusterError,
    UnknownDatasetError,
    UnsafeDatasetOperationError,
)
from lambdaforge.data.recipe_config import DatasetRecipeConfig
from lambdaforge.diagnostics import ErrorCategory, LambdaForgeError, diagnostic


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
                previous.build_id or record.build_id,
                previous.index or record.index,
                previous.partitions or record.partitions,
                previous.target_schema or record.target_schema,
                previous.global_assets or record.global_assets,
                previous.lineage_graph or record.lineage_graph,
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
        local = tuple(
            record
            for record in self.registry.records()
            if record.key == selector or ("@" not in selector and record.name == selector)
        )
        if len(local) == 1:
            return local[0]
        if len(local) > 1:
            raise AmbiguousDatasetVersionError(
                selector, tuple(sorted(record.version for record in local))
            )
        matches = tuple(
            record
            for record in self.list(all_clusters=True)
            if record.key == selector or ("@" not in selector and record.name == selector)
        )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise AmbiguousDatasetVersionError(
                selector, tuple(sorted(record.version for record in matches))
            )
        raise UnknownDatasetError(selector, tuple(record.key for record in self.registry.records()))

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
            if placement.cluster != "local" and (
                schema.get("task") == "classification" or schema.get("profiler") is not None
            ):
                payload.update(self._remote_profile(placement, record, schema))
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

    def members(
        self,
        selector: str,
        *,
        cluster: str | None = None,
        partitions: Mapping[str, str] | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Inspect a bounded page of logical members on the selected placement."""
        placement = self._placement(self.show(selector), cluster)
        options = json.dumps(
            {"partitions": dict(partitions or {}), "offset": offset, "limit": limit},
            separators=(",", ":"),
        )
        return self._operation(placement.cluster, "members", placement.root, options)

    def member(
        self, selector: str, member_id: str, *, cluster: str | None = None
    ) -> dict[str, Any]:
        """Inspect one exact logical member."""
        placement = self._placement(self.show(selector), cluster)
        return self._operation(placement.cluster, "member", placement.root, member_id)

    def diff(self, left: str, right: str, *, cluster: str | None = None) -> dict[str, Any]:
        """Compare two immutable versions at one common placement."""
        left_record = self.show(left)
        right_record = self.show(right)
        if cluster is None:
            common = sorted(
                {value.cluster for value in left_record.placements}
                & {value.cluster for value in right_record.placements},
                key=lambda value: (value != "local", value),
            )
            if not common:
                raise MissingDatasetPlacementError(
                    f"{left_record.key} and {right_record.key}",
                    "a shared cluster",
                    tuple(sorted({value.cluster for value in left_record.placements})),
                )
            cluster = common[0]
        left_placement = self._placement(left_record, cluster)
        right_placement = self._placement(right_record, cluster)
        return self._operation(cluster, "diff", left_placement.root, right_placement.root)

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
                raise UnsafeDatasetOperationError(
                    "Dataset deletion is unsafe: " + " ".join(reasons)
                )
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
            "graph": dict(record.lineage_graph),
            "build_id": record.build_id,
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
        try:
            record = self.show(selector)
        except (KeyError, UnknownDatasetError):
            recipe = self._recipe(selector)
            if strategy == "replicate":
                raise RuntimeError("An unbuilt dataset cannot be replicated.") from None
            return self._build_materialization(recipe, cluster, apply=apply)
        if any(item.cluster == cluster for item in record.placements):
            return DatasetMaterializationPlan(
                record.key, cluster, "NOOP", reason="Already present."
            )
        source = next(iter(sorted(record.placements, key=lambda value: value.cluster)), None)
        producer = str(record.producer.get("config", "")) or None
        if strategy == "build" or (source is None and producer):
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
            try:
                recipe = self._recipe(producer or record.name)
            except MissingDatasetRecipeError:
                try:
                    recipe = self._recipe(record.name)
                except MissingDatasetRecipeError:
                    if apply:
                        raise
                    return DatasetMaterializationPlan(
                        record.key,
                        cluster,
                        "BUILD",
                        producer=producer,
                        reason=(
                            "Legacy task producer is recorded, but automatic BUILD apply "
                            "requires a kind: dataset recipe. Preview remains compatible."
                        ),
                        prerequisites=tuple(prerequisites),
                    )
            if apply:
                for item in prerequisites:
                    if item["action"] == "REPLICATE":
                        self.replicate(
                            str(item["dataset"]),
                            source=str(item["source_cluster"]),
                            destination=cluster,
                            apply=True,
                        )
            return self._build_materialization(
                recipe, cluster, apply=apply, prerequisites=tuple(prerequisites)
            )
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

    def _recipe(self, selector_or_path: str) -> DatasetRecipeConfig:
        """Resolve a recipe path/name without introducing a second project registry."""
        from lambdaforge.configuration.ProjectConfigService import ProjectConfigService

        raw = str(selector_or_path)
        path = Path(raw)
        if path.is_file():
            return DatasetRecipeConfig.from_yaml(path)
        name = raw.removeprefix("dataset:").split("@", 1)[0]
        try:
            resolved = ProjectConfigService().resolve(name, kind="dataset")
        except (KeyError, ValueError) as error:
            known = tuple(
                record.name
                for record in ProjectConfigService().list(kind="dataset")
                if record.valid
            )
            raise MissingDatasetRecipeError(name, known) from error
        recipe = DatasetRecipeConfig.from_yaml(resolved)
        requested_version = raw.partition("@")[2] or None
        if requested_version is not None and recipe.version != requested_version:
            raise MissingDatasetRecipeError(raw, (recipe.selector,))
        return recipe

    def _build_materialization(
        self,
        recipe: DatasetRecipeConfig,
        cluster: str,
        *,
        apply: bool,
        prerequisites: tuple[dict[str, Any], ...] = (),
    ) -> DatasetMaterializationPlan:
        service = DatasetBuildService(
            self.registry, JobService(self.clusters, factory=self.factory)
        )
        build_plan = service.plan(recipe, cluster=cluster)
        handle = service.submit(recipe, cluster=cluster) if apply else None
        return DatasetMaterializationPlan(
            recipe.selector,
            cluster,
            "BUILD",
            producer=str(recipe.source) if recipe.source is not None else None,
            reason=(
                "Dataset recipe submitted as one durable dataset-build job."
                if apply
                else "Missing placement can be produced by the registered dataset recipe."
            ),
            prerequisites=prerequisites,
            stages=tuple(stage.to_dict() for stage in build_plan.stages),
            job_id=handle.job_id if handle is not None else None,
        )

    def _remote_profile(
        self,
        placement: DatasetPlacement,
        record: DatasetRecord,
        schema: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Run an installed project profiler beside remote bytes in the managed environment."""
        profile = self.clusters.get(placement.cluster)
        transport = self.factory.transport(profile)
        python = self._python(placement.cluster, transport)
        code = (
            "import json,sys; "
            "from pathlib import Path; "
            "from lambdaforge.data import ClassificationDatasetProfiler,DatasetRecord; "
            "from lambdaforge.experiments.ObjectFactory import ObjectFactory; "
            "schema=json.loads(sys.argv[2]); "
            "record=DatasetRecord.from_mapping(json.loads(sys.argv[3])); "
            "profiler=(ClassificationDatasetProfiler() if schema.get('task')=='classification' "
            "else ObjectFactory.build(schema['profiler'])); "
            "print(json.dumps(profiler.profile(Path(sys.argv[1]),record,schema),default=str))"
        )
        result = transport.run(
            (
                *profile.command_prefix,
                python,
                "-c",
                code,
                placement.root,
                json.dumps(dict(schema), separators=(",", ":")),
                json.dumps(record.to_dict(), separators=(",", ":")),
            )
        )
        if result.returncode:
            raise RuntimeError(f"Remote dataset profiler failed: {result.stderr}")
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict):
            raise TypeError("Remote dataset profiler must return a JSON object.")
        return payload

    def _replicate(self, record: DatasetRecord, source: DatasetPlacement, destination: str) -> None:
        target = self.clusters.get(destination)
        assert target.storage is not None
        if target.storage.dataset_root is None:
            raise LambdaForgeError(
                diagnostic(
                    ErrorCategory.CONFIGURATION,
                    f"Cannot replicate {record.key!r} to {destination!r}.",
                    "The target cluster has no permanent dataset storage location.",
                    reason=(
                        "Dataset placements are immutable scientific data and cannot be "
                        "published into the reconstructible cache or transient job workspace."
                    ),
                    impact=("No dataset bytes were copied and no placement was registered.",),
                    fixes=(
                        "Configure storage.dataset_root as a persistent, writable cluster path.",
                    ),
                    commands=(
                        (
                            "Configure dataset storage",
                            "lf clusters set "
                            f"{shlex.quote(destination)} storage.dataset_root "
                            "/persistent/path/to/datasets",
                        ),
                        (
                            "Retry after configuring",
                            shlex.join(
                                (
                                    "lf",
                                    "datasets",
                                    "replicate",
                                    record.key,
                                    "--from",
                                    source.cluster,
                                    "--to",
                                    destination,
                                    "--apply",
                                )
                            ),
                        ),
                    ),
                    context={
                        "dataset": record.key,
                        "source_cluster": source.cluster,
                        "target_cluster": destination,
                        "storage.dataset_root": "not configured",
                    },
                    operation="dataset replication preflight",
                )
            )
        target_root = str(
            PurePosixPath(target.storage.dataset_root)
            / record.name
            / record.version
            / record.dataset_id.removeprefix("sha256:")[:16]
        )
        if source.cluster != "local":
            raise RuntimeError(
                "The built-in safe transfer provider currently needs a local source. "
                "Use a shared filesystem/provider or stage an explicit durable transfer job."
            )
        if target.transport == "local":
            destination_path = Path(target_root)
            if destination_path.exists():
                verification = DatasetOperations.verify(destination_path, record.dataset_id)
                if not verification["valid"]:
                    raise self._immutable_target_error(record, destination, target_root)
            else:
                staging = destination_path.with_name(f".{destination_path.name}.{uuid4().hex}.tmp")
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copytree(source.root, staging)
                    verification = DatasetOperations.verify(staging, record.dataset_id)
                    if not verification["valid"]:
                        raise RuntimeError("Replicated dataset failed validation before publish.")
                    os.replace(staging, destination_path)
                finally:
                    if staging.exists():
                        shutil.rmtree(staging)
        else:
            transport = self.factory.transport(target)
            exists = transport.run(("test", "-e", target_root))
            if exists.returncode == 0:
                verified = self._operation(destination, "verify", target_root, record.dataset_id)
                if not verified.get("valid"):
                    raise self._immutable_target_error(record, destination, target_root)
            else:
                remote_staging = f"{target_root}.tmp-{uuid4().hex}"
                created = transport.run(("mkdir", "-p", str(PurePosixPath(remote_staging).parent)))
                if created.returncode:
                    raise RuntimeError(f"Cannot create remote dataset staging: {created.stderr}")
                destination_value = (
                    f"{target.user + '@' if target.user else ''}{target.host}:{remote_staging}"
                )
                completed = subprocess.run(
                    (
                        "rsync",
                        "-a",
                        "--protect-args",
                        "--",
                        source.root.rstrip("/") + "/",
                        destination_value.rstrip("/") + "/",
                    ),
                    check=False,
                    capture_output=True,
                    text=True,
                    shell=False,
                )
                if completed.returncode:
                    transport.run(("rm", "-rf", remote_staging))
                    raise RuntimeError(f"Dataset replication failed: {completed.stderr}")
                verified = self._operation(destination, "verify", remote_staging, record.dataset_id)
                if not verified.get("valid"):
                    transport.run(("rm", "-rf", remote_staging))
                    raise RuntimeError("Replicated dataset failed validation before publish.")
                published = transport.run(("mv", remote_staging, target_root))
                if published.returncode:
                    raise RuntimeError(f"Atomic dataset publish failed: {published.stderr}")
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
            record.build_id,
            record.index,
            record.partitions,
            record.target_schema,
            record.global_assets,
            record.lineage_graph,
        )
        self.registry.register(updated)
        self._publish_remote(updated, destination)

    @staticmethod
    def _immutable_target_error(
        record: DatasetRecord, destination: str, target_root: str
    ) -> LambdaForgeError:
        return LambdaForgeError(
            diagnostic(
                ErrorCategory.OPERATION_REFUSED,
                f"Refusing to overwrite dataset {record.key!r}.",
                "The target path already contains different or invalid bytes.",
                reason=(
                    "A published dataset version is immutable; replacing its bytes would make "
                    "past experiments ambiguous and break its content identity."
                ),
                impact=("The existing target and the registered dataset were left unchanged.",),
                fixes=(
                    "Verify the existing placement and publish changed content as a new version.",
                ),
                commands=(
                    (
                        "Verify registered placement",
                        f"lf datasets verify {shlex.quote(record.key)} --on "
                        f"{shlex.quote(destination)}",
                    ),
                    ("Inspect version", f"lf datasets show {shlex.quote(record.key)}"),
                ),
                context={
                    "dataset": record.key,
                    "target_cluster": destination,
                    "target_path": target_root,
                    "expected_dataset_id": record.dataset_id,
                },
                operation="dataset replication publish",
            )
        )

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
            if operation == "members":
                options = json.loads(arguments[0]) if arguments else {}
                return DatasetOperations.members(
                    root,
                    partitions=options.get("partitions"),
                    offset=int(options.get("offset", 0)),
                    limit=int(options.get("limit", 100)),
                )
            if operation == "member":
                return DatasetOperations.member(root, arguments[0])
            if operation == "diff":
                return DatasetOperations.diff(root, arguments[0])
        profile = self.clusters.get(cluster)
        assert profile.storage is not None
        transport = self.factory.transport(profile)
        try:
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
        except (OSError, TimeoutError) as error:
            raise OfflineClusterError(cluster, str(error)) from error
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
            raise MissingDatasetPlacementError(
                record.key,
                cluster or "an explicitly selected cluster",
                tuple(sorted(value.cluster for value in record.placements)),
            )
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
        if result.returncode or not result.stdout.strip():
            raise MissingManagedEnvironmentError(cluster)
        return result.stdout.strip()
