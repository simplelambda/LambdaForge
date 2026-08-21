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
from lambdaforge.data.DatasetResolution import (
    DatasetPlacementResolution,
    DatasetPlacementState,
)
from lambdaforge.data.errors import (
    AmbiguousDatasetVersionError,
    DatasetRegistryCorruptionError,
    DatasetResolutionError,
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
        self.discovery_warnings: tuple[str, ...] = ()

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
        warnings: list[str] = []
        with ThreadPoolExecutor(max_workers=min(self.max_parallel, len(names) or 1)) as executor:
            futures = {executor.submit(self._remote_records, name): name for name in names}
            for future in as_completed(futures):
                try:
                    discovered = future.result()
                    remote.extend(discovered)
                    for record in discovered:
                        self.registry.register(record)
                except Exception as error:
                    if not all_clusters:
                        raise
                    warnings.append(f"{futures[future]}: {error.__class__.__name__}: {error}")
        self.discovery_warnings = tuple(sorted(warnings))
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

    def show(self, selector: str, *, cluster: str | None = None) -> DatasetRecord:
        """Resolve a local or reconciled remote logical dataset version."""
        if cluster is not None:
            return self.resolve_placement(selector, cluster).record
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
        if self.discovery_warnings:
            raise DatasetResolutionError(
                f"Dataset {selector!r} is not in the controller index, but remote discovery "
                "was incomplete. Its absence cannot be established.\n"
                "Retry with an explicit '--on CLUSTER' after restoring connectivity."
            )
        raise UnknownDatasetError(selector, tuple(record.key for record in self.registry.records()))

    def describe(self, selector: str, *, cluster: str | None = None) -> dict[str, Any]:
        """Describe cached logical state or one freshly reconciled target placement."""
        if cluster is None:
            record = self.show(selector)
            payload = record.to_dict()
            payload["placement_consistency"] = {
                "scope": "controller-cache",
                "states": {placement.cluster: "cached" for placement in record.placements},
            }
            return payload
        resolution = self.resolve_placement(selector, cluster)
        payload = resolution.record.to_dict()
        payload["placement_consistency"] = resolution.to_dict()
        return payload

    def resolve_placement(self, selector: str, cluster: str) -> DatasetPlacementResolution:
        """Reconcile controller, target index and bounded manifest identity in one path."""
        controller_record = self._select_record(self.registry.records(), selector)
        target_records: tuple[DatasetRecord, ...] = ()
        if cluster != "local":
            try:
                target_records = self._remote_records(cluster)
            except DatasetRegistryCorruptionError:
                raise
            except OfflineClusterError as error:
                if controller_record is None:
                    raise
                return DatasetPlacementResolution(
                    controller_record,
                    cluster,
                    DatasetPlacementState.UNREACHABLE,
                    controller_placement=self._placement_for(controller_record, cluster),
                    reason=f"Target state could not be observed: {error}",
                )
            except Exception as error:
                if controller_record is None:
                    raise
                return DatasetPlacementResolution(
                    controller_record,
                    cluster,
                    DatasetPlacementState.UNREACHABLE,
                    controller_placement=self._placement_for(controller_record, cluster),
                    reason=f"Target state could not be observed: {error}",
                )
        target_record = self._select_record(target_records, selector)
        if controller_record is None and target_record is None:
            raise UnknownDatasetError(
                selector, tuple(record.key for record in self.registry.records())
            )
        record = controller_record or target_record
        assert record is not None
        if target_record is not None and target_record.dataset_id != record.dataset_id:
            return DatasetPlacementResolution(
                record,
                cluster,
                DatasetPlacementState.CONFLICT,
                controller_placement=self._placement_for(controller_record, cluster),
                target_placement=self._placement_for(target_record, cluster),
                reason=(
                    "Controller and target registries assign different immutable identities "
                    f"to {record.key}: {record.dataset_id} != {target_record.dataset_id}."
                ),
            )
        controller_placement = self._placement_for(controller_record, cluster)
        target_placement = (
            controller_placement
            if cluster == "local"
            else self._placement_for(target_record, cluster)
        )
        roots = self._candidate_roots(
            record,
            cluster,
            controller_placement=controller_placement,
            target_placement=target_placement,
        )
        observations: dict[str, dict[str, Any]] = {}
        try:
            for root in roots:
                observations[root] = self._operation(cluster, "inspect", root)
        except Exception as error:
            return DatasetPlacementResolution(
                record,
                cluster,
                DatasetPlacementState.UNREACHABLE,
                controller_placement=controller_placement,
                target_placement=target_placement,
                reason=f"Physical target state could not be observed: {error}",
            )
        exact = tuple(
            root
            for root, observation in observations.items()
            if self._matches_artifact(record, observation)
        )
        conflicts = tuple(
            (root, observation)
            for root, observation in observations.items()
            if observation.get("exists") and root not in exact
        )
        registered = target_placement
        indexed_roots = {
            placement.root
            for placement in (controller_placement, target_placement)
            if placement is not None
        }
        indexed_conflict = next(
            ((root, observation) for root, observation in conflicts if root in indexed_roots),
            None,
        )
        if indexed_conflict is not None or (conflicts and not exact):
            selected_conflict = indexed_conflict or conflicts[0]
            assert selected_conflict is not None
            root, observation = selected_conflict
            return DatasetPlacementResolution(
                record,
                cluster,
                DatasetPlacementState.CONFLICT,
                controller_placement=controller_placement,
                target_placement=target_placement,
                physical=observation,
                reason=(
                    f"{root} exists but its DatasetArtifact does not match "
                    f"{record.key} ({record.dataset_id})."
                ),
            )
        registered_exact = registered is not None and registered.root in exact
        controller_exact = controller_placement is not None and controller_placement.root in exact
        if (
            controller_placement is not None
            and target_placement is not None
            and controller_placement.root != target_placement.root
            and controller_exact
            and registered_exact
        ):
            return DatasetPlacementResolution(
                record,
                cluster,
                DatasetPlacementState.CONFLICT,
                controller_placement=controller_placement,
                target_placement=target_placement,
                reason=(
                    "Controller and target registries point to two different valid roots on "
                    f"{cluster!r}; LambdaForge will not choose one silently."
                ),
            )
        if exact:
            selected_root = (
                registered.root
                if registered_exact and registered is not None
                else controller_placement.root
                if controller_exact and controller_placement is not None
                else exact[0]
            )
            observation = observations[selected_root]
            placement = self._observed_placement(
                record,
                cluster,
                selected_root,
                registered or controller_placement,
            )
            combined = self._with_placement(record, placement)
            if registered_exact:
                repair = (
                    "update_controller_index"
                    if controller_placement is None or controller_placement.root != selected_root
                    else None
                )
                return DatasetPlacementResolution(
                    combined,
                    cluster,
                    DatasetPlacementState.AVAILABLE,
                    placement,
                    controller_placement,
                    target_placement,
                    observation,
                    "The target registry and immutable DatasetArtifact agree.",
                    repair,
                )
            return DatasetPlacementResolution(
                combined,
                cluster,
                DatasetPlacementState.DISCOVERED_UNREGISTERED,
                placement,
                controller_placement,
                target_placement,
                observation,
                "An exact managed DatasetArtifact exists but the target index lacks its placement.",
                "register_exact_placement",
            )
        if registered is not None or controller_placement is not None:
            missing = registered or controller_placement
            assert missing is not None
            return DatasetPlacementResolution(
                record,
                cluster,
                DatasetPlacementState.REGISTERED_BUT_MISSING,
                missing,
                controller_placement,
                target_placement,
                observations.get(missing.root, {}),
                "A placement is indexed, but no exact DatasetArtifact exists at its root.",
                "remove_stale_registration",
            )
        return DatasetPlacementResolution(
            record,
            cluster,
            DatasetPlacementState.ABSENT,
            physical=next(iter(observations.values()), {}),
            reason="No registered or bounded managed placement was found on the target.",
        )

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
        record, placement = self._physical_placement(selector, cluster)
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
        _, placement = self._physical_placement(selector, cluster)
        options = json.dumps(
            {"partitions": dict(partitions or {}), "offset": offset, "limit": limit},
            separators=(",", ":"),
        )
        return self._operation(placement.cluster, "members", placement.root, options)

    def member(
        self, selector: str, member_id: str, *, cluster: str | None = None
    ) -> dict[str, Any]:
        """Inspect one exact logical member."""
        _, placement = self._physical_placement(selector, cluster)
        return self._operation(placement.cluster, "member", placement.root, member_id)

    def diff(self, left: str, right: str, *, cluster: str | None = None) -> dict[str, Any]:
        """Compare two immutable versions at one common placement."""
        if cluster is not None:
            _, left_placement = self._physical_placement(left, cluster)
            _, right_placement = self._physical_placement(right, cluster)
            return self._operation(cluster, "diff", left_placement.root, right_placement.root)
        left_record = self.show(left)
        right_record = self.show(right)
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
        if cluster is None:
            record = self.show(selector)
            placement = self._placement(record, None)
            return self._operation(placement.cluster, "verify", placement.root, record.dataset_id)
        resolution = self.resolve_placement(selector, cluster)
        if resolution.state is DatasetPlacementState.UNREACHABLE:
            raise OfflineClusterError(cluster, resolution.reason)
        if not resolution.physically_available or resolution.placement is None:
            return {
                "valid": False,
                "dataset": resolution.record.key,
                "dataset_id": resolution.record.dataset_id,
                "cluster": cluster,
                "placement_state": resolution.state.value,
                "errors": [resolution.reason],
                "reconcile": (
                    f"lf datasets reconcile {resolution.record.key} --on {cluster}"
                    if resolution.repair
                    else None
                ),
            }
        payload = self._operation(
            cluster,
            "verify",
            resolution.placement.root,
            resolution.record.dataset_id,
        )
        verified_state = (
            resolution.state.value if payload.get("valid") else DatasetPlacementState.CONFLICT.value
        )
        payload.update(
            {
                "dataset": resolution.record.key,
                "cluster": cluster,
                "placement_state": verified_state,
                "reconcile": (
                    f"lf datasets reconcile {resolution.record.key} --on {cluster}"
                    if resolution.repair and payload.get("valid")
                    else None
                ),
            }
        )
        return payload

    def remove(self, selector: str, *, cluster: str | None = None) -> DatasetRecord | None:
        """Remove index state only; physical dataset bytes are never touched."""
        if cluster is not None and cluster != "local":
            resolution = self.resolve_placement(selector, cluster)
            if resolution.state in {
                DatasetPlacementState.CONFLICT,
                DatasetPlacementState.UNREACHABLE,
            }:
                raise UnsafeDatasetOperationError(
                    f"Registration removal refused: {resolution.reason}"
                )
            selector = resolution.record.key
            self._remove_remote(selector, cluster)
        return self.registry.discard(selector, cluster=cluster)

    def delete(self, selector: str, *, cluster: str, apply: bool = False) -> DatasetDeletionPlan:
        """Preview or converge deletion of one exact manifest-backed managed placement."""
        resolution = self.resolve_placement(selector, cluster)
        record = resolution.record
        placement = resolution.placement
        reasons: list[str] = []
        consumers = self._active_consumers(record, cluster)
        if consumers:
            reasons.append(
                "Active LambdaForge consumers use this exact DatasetVersion: "
                + ", ".join(consumers)
            )
        profile = self.clusters.get(cluster)
        assert profile.storage is not None
        managed_root = profile.storage.dataset_root
        action = "DELETE_PLACEMENT"
        size_bytes = placement.size_bytes if placement is not None else None
        file_count = placement.file_count if placement is not None else None
        if resolution.state is DatasetPlacementState.UNREACHABLE:
            reasons.append("The target is unreachable; absence cannot be proven.")
        elif resolution.state is DatasetPlacementState.CONFLICT:
            reasons.append(resolution.reason)
        elif resolution.state is DatasetPlacementState.DISCOVERED_UNREGISTERED:
            reasons.append(
                "The exact physical copy is not registered on the target; reconcile it before "
                "requesting deletion."
            )
        elif resolution.state is DatasetPlacementState.ABSENT:
            reasons.append("No exact managed placement was found; nothing can be deleted.")
        elif resolution.state is DatasetPlacementState.REGISTERED_BUT_MISSING:
            action = "CLEAN_STALE_REGISTRATION"
        elif managed_root is None:
            reasons.append("The target has no configured storage.dataset_root safety boundary.")
        elif placement is None or not self._inside_managed_root(placement.root, managed_root):
            reasons.append("The placement root is outside configured managed dataset storage.")
        else:
            try:
                current_stats = self._operation(cluster, "stats", placement.root)
                size_bytes = current_stats.get("size_bytes")
                file_count = current_stats.get("file_count")
            except Exception as error:
                reasons.append(f"Current placement size could not be inspected: {error}")
        safe = not reasons
        applied = False
        if apply:
            if not safe:
                raise UnsafeDatasetOperationError(
                    "Dataset deletion is unsafe: " + " ".join(reasons)
                )
            if action == "DELETE_PLACEMENT":
                assert placement is not None and managed_root is not None
                self._operation(
                    cluster,
                    "delete",
                    placement.root,
                    record.dataset_id,
                    managed_root,
                    "--apply",
                )
            if cluster != "local":
                self._remove_remote(record.key, cluster)
            self.registry.discard(record.key, cluster=cluster)
            applied = True
        return DatasetDeletionPlan(
            record.key,
            cluster,
            placement.root if placement is not None else "",
            size_bytes,
            file_count,
            safe,
            tuple(reasons),
            applied,
            action,
            record.dataset_id,
            resolution.state.value,
            consumers,
        )

    def reconcile(self, selector: str, *, cluster: str, apply: bool = False) -> dict[str, Any]:
        """Preview or apply only identity-preserving placement-index repairs."""
        resolution = self.resolve_placement(selector, cluster)
        state = resolution.state
        action = "NOOP"
        safe = True
        if state is DatasetPlacementState.AVAILABLE and resolution.repair:
            action = "UPDATE_CONTROLLER_INDEX"
        elif state is DatasetPlacementState.DISCOVERED_UNREGISTERED:
            action = "REGISTER_EXACT_PLACEMENT"
        elif state is DatasetPlacementState.REGISTERED_BUT_MISSING:
            action = "REMOVE_STALE_REGISTRATION"
        elif state in {DatasetPlacementState.CONFLICT, DatasetPlacementState.UNREACHABLE}:
            action = "REFUSE"
            safe = False
        elif state is DatasetPlacementState.ABSENT:
            action = "NOOP"
        if apply and not safe:
            raise UnsafeDatasetOperationError(
                f"Dataset reconciliation is unsafe: {resolution.reason}"
            )
        applied = False
        if apply and action in {"UPDATE_CONTROLLER_INDEX", "REGISTER_EXACT_PLACEMENT"}:
            if resolution.placement is None:
                raise UnsafeDatasetOperationError(
                    "Reconciliation has no exact manifest-backed placement to register."
                )
            if action == "REGISTER_EXACT_PLACEMENT" and cluster != "local":
                self._publish_remote(resolution.record, cluster)
            self.registry.register(resolution.record)
            applied = True
        elif apply and action == "REMOVE_STALE_REGISTRATION":
            if cluster != "local":
                self._remove_remote(resolution.record.key, cluster)
            self.registry.discard(resolution.record.key, cluster=cluster)
            applied = True
        return {
            "dataset": resolution.record.key,
            "dataset_id": resolution.record.dataset_id,
            "cluster": cluster,
            "state": state.value,
            "safe": safe,
            "reason": resolution.reason,
            "action": action,
            "root": resolution.placement.root if resolution.placement is not None else None,
            "applied": applied,
            "why_safe": (
                "The controller, target and physical manifest use the same immutable dataset_id."
                if safe and action not in {"NOOP", "REMOVE_STALE_REGISTRATION"}
                else "Only stale index metadata is removed; dataset bytes are not touched."
                if safe and action == "REMOVE_STALE_REGISTRATION"
                else None
            ),
        }

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
            resolution = self.resolve_placement(selector, cluster)
        except (KeyError, UnknownDatasetError):
            recipe = self._recipe(selector)
            if strategy == "replicate":
                raise RuntimeError("An unbuilt dataset cannot be replicated.") from None
            return self._build_materialization(recipe, cluster, apply=apply)
        record = resolution.record
        if resolution.state is DatasetPlacementState.UNREACHABLE:
            raise OfflineClusterError(cluster, resolution.reason)
        if resolution.state is DatasetPlacementState.CONFLICT:
            raise UnsafeDatasetOperationError(
                "Dataset materialization refused because target state conflicts: "
                + resolution.reason
            )
        if resolution.state is DatasetPlacementState.AVAILABLE:
            return DatasetMaterializationPlan(
                record.key,
                cluster,
                "NOOP",
                reason="The target index and exact physical DatasetArtifact agree.",
            )
        if resolution.state is DatasetPlacementState.DISCOVERED_UNREGISTERED:
            plan = DatasetMaterializationPlan(
                record.key,
                cluster,
                "RECONCILE",
                reason=(
                    "The exact physical DatasetArtifact already exists; repair placement indexes "
                    "instead of copying or rebuilding bytes."
                ),
            )
            if apply:
                self.reconcile(record.key, cluster=cluster, apply=True)
            return plan
        source = next(
            iter(
                sorted(
                    (placement for placement in record.placements if placement.cluster != cluster),
                    key=lambda value: value.cluster,
                )
            ),
            None,
        )
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
        resolution = self.resolve_placement(selector, source)
        if resolution.state is DatasetPlacementState.UNREACHABLE:
            raise OfflineClusterError(source, resolution.reason)
        if not resolution.physically_available or resolution.placement is None:
            raise UnsafeDatasetOperationError(
                f"Dataset source {source!r} is {resolution.state.value}: {resolution.reason}"
            )
        record = resolution.record
        placement = resolution.placement
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
        selector = raw.removeprefix("dataset:").split("/", 1)[0]
        name = selector.split("@", 1)[0]
        try:
            resolved = ProjectConfigService().resolve(selector, kind="dataset")
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
        try:
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
        except (OSError, TimeoutError) as error:
            raise OfflineClusterError(cluster, str(error)) from error
        if result.returncode:
            if "DatasetRegistryCorruptionError" in result.stderr:
                raise DatasetRegistryCorruptionError(
                    registry_path, result.stderr.strip(), cluster=cluster
                )
            raise DatasetResolutionError(
                f"Dataset inventory on {cluster!r} failed without proving absence.\n"
                f"Reason: {result.stderr.strip()}"
            )
        try:
            value = json.loads(result.stdout or "[]")
        except json.JSONDecodeError as error:
            raise DatasetRegistryCorruptionError(
                registry_path, str(error), cluster=cluster
            ) from error
        if not isinstance(value, list):
            raise DatasetRegistryCorruptionError(
                registry_path, "inventory output is not a JSON list", cluster=cluster
            )
        records = []
        for item in value:
            if not isinstance(item, Mapping):
                raise DatasetRegistryCorruptionError(
                    registry_path, "inventory contains a non-object entry", cluster=cluster
                )
            try:
                record = DatasetRecord.from_mapping(item)
            except (KeyError, TypeError, ValueError) as error:
                raise DatasetRegistryCorruptionError(
                    registry_path, f"invalid inventory record: {error}", cluster=cluster
                ) from error
            placements = tuple(
                placement for placement in record.placements if placement.cluster == cluster
            )
            records.append(self._with_placements(record, placements))
        return tuple(records)

    def _publish_remote(self, record: DatasetRecord, cluster: str) -> None:
        profile = self.clusters.get(cluster)
        assert profile.storage is not None
        transport = self.factory.transport(profile)
        remote_path = PurePosixPath(profile.storage.state_root) / "datasets.json"
        python = self._python(cluster, transport)
        placement = self._placement_for(record, cluster)
        if placement is None:
            raise ValueError(f"Dataset {record.key} has no placement for {cluster} to publish.")
        target_record = self._with_placements(record, (placement,))
        with tempfile.TemporaryDirectory(prefix="lambdaforge-dataset-record-") as temporary:
            record_path = Path(temporary) / "record.json"
            record_path.write_text(
                json.dumps(target_record.to_dict(), indent=2) + "\n", encoding="utf-8"
            )
            staged = (
                PurePosixPath(profile.storage.state_root)
                / f"dataset-{os.getpid()}-{uuid4().hex}.json"
            )
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
            "DatasetRegistry(sys.argv[1]).discard(sys.argv[2],cluster=sys.argv[3])"
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
            if operation == "inspect":
                return DatasetOperations.inspect(root)
            if operation == "stats":
                return DatasetOperations.stats(root)
            if operation == "verify":
                return DatasetOperations.verify(root, arguments[0])
            if operation == "delete":
                return DatasetOperations.delete(
                    root,
                    arguments[0],
                    managed_root=arguments[1],
                    apply="--apply" in arguments[2:],
                )
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
        active = []
        for job in jobs:
            if job.state.terminal:
                continue
            references = tuple(str(value) for value in job.metadata.get("datasets", ()))
            identities = tuple(str(value) for value in job.metadata.get("dataset_ids", ()))
            exact = record.key in references or record.dataset_id in identities
            legacy = record.name in references and not any("@" in value for value in references)
            if exact or legacy:
                active.append(job.job_id)
        return tuple(active)

    def _physical_placement(
        self, selector: str, cluster: str | None
    ) -> tuple[DatasetRecord, DatasetPlacement]:
        if cluster is None:
            record = self.show(selector)
            return record, self._placement(record, None)
        resolution = self.resolve_placement(selector, cluster)
        if resolution.state is DatasetPlacementState.UNREACHABLE:
            raise OfflineClusterError(cluster, resolution.reason)
        if not resolution.physically_available or resolution.placement is None:
            raise UnsafeDatasetOperationError(
                f"Dataset {resolution.record.key} on {cluster!r} is "
                f"{resolution.state.value}: {resolution.reason}"
            )
        return resolution.record, resolution.placement

    @staticmethod
    def _inside_managed_root(root: str, managed_root: str) -> bool:
        candidate = PurePosixPath(root)
        allowed = PurePosixPath(managed_root)
        return candidate != allowed and candidate.is_relative_to(allowed)

    @staticmethod
    def _select_record(records: tuple[DatasetRecord, ...], selector: str) -> DatasetRecord | None:
        matches = tuple(
            record
            for record in records
            if record.key == selector or ("@" not in selector and record.name == selector)
        )
        if len(matches) > 1:
            raise AmbiguousDatasetVersionError(
                selector, tuple(sorted(record.version for record in matches))
            )
        return matches[0] if matches else None

    @staticmethod
    def _placement_for(record: DatasetRecord | None, cluster: str) -> DatasetPlacement | None:
        if record is None:
            return None
        return next(
            (placement for placement in record.placements if placement.cluster == cluster),
            None,
        )

    def _candidate_roots(
        self,
        record: DatasetRecord,
        cluster: str,
        *,
        controller_placement: DatasetPlacement | None,
        target_placement: DatasetPlacement | None,
    ) -> tuple[str, ...]:
        roots: list[str] = []
        for placement in (target_placement, controller_placement):
            if placement is not None and placement.root not in roots:
                roots.append(placement.root)
        profile = self.clusters.get(cluster)
        assert profile.storage is not None
        if profile.storage.dataset_root is not None:
            expected = str(
                PurePosixPath(profile.storage.dataset_root)
                / record.name
                / record.version
                / record.dataset_id.removeprefix("sha256:")[:16]
            )
            if expected not in roots:
                roots.append(expected)
        return tuple(roots)

    @staticmethod
    def _matches_artifact(record: DatasetRecord, observation: Mapping[str, Any]) -> bool:
        return bool(
            observation.get("manifest_valid")
            and observation.get("name") == record.name
            and str(observation.get("version")) == record.version
            and observation.get("dataset_id") == record.dataset_id
            and observation.get("content_id", record.dataset_id) == record.dataset_id
        )

    @staticmethod
    def _observed_placement(
        record: DatasetRecord,
        cluster: str,
        root: str,
        previous: DatasetPlacement | None,
    ) -> DatasetPlacement:
        del record
        return DatasetPlacement(
            cluster,
            root,
            previous.registered_at_utc
            if previous is not None
            else datetime.now(timezone.utc).isoformat(),
            previous.size_bytes if previous is not None else None,
            previous.file_count if previous is not None else None,
            True,
        )

    @classmethod
    def _with_placement(cls, record: DatasetRecord, placement: DatasetPlacement) -> DatasetRecord:
        values = {
            item.cluster: item for item in record.placements if item.cluster != placement.cluster
        }
        values[placement.cluster] = placement
        return cls._with_placements(record, tuple(values[key] for key in sorted(values)))

    @staticmethod
    def _with_placements(
        record: DatasetRecord, placements: tuple[DatasetPlacement, ...]
    ) -> DatasetRecord:
        return DatasetRecord(
            record.name,
            record.version,
            record.dataset_id,
            record.sample_count,
            record.splits,
            record.created_at_utc,
            placements,
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
