"""Application service for logical dataset operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.data.DataCatalog import DataCatalog
from lambdaforge.data.DataReplicationResult import DataReplicationResult
from lambdaforge.data.DatasetReference import DatasetReference
from lambdaforge.data.DataTransferProvider import DataTransferProvider
from lambdaforge.data.RsyncDataTransferProvider import RsyncDataTransferProvider
from lambdaforge.preprocessing.DatasetArtifact import DatasetArtifact


class DataService:
    """List placements and explicitly replicate catalogued datasets."""

    def __init__(
        self,
        catalog: DataCatalog,
        clusters: ClusterCatalog | None = None,
        transfer: DataTransferProvider | None = None,
    ) -> None:
        self.catalog = catalog
        self.clusters = clusters or ClusterCatalog.load()
        self.transfer = transfer or RsyncDataTransferProvider()

    def list(self) -> tuple[str, ...]:
        """Return registered logical dataset names."""
        return self.catalog.names()

    def locations(self, dataset: str) -> dict[str, object]:
        """Return declared placements without probing or moving bytes."""
        descriptor = self.catalog.descriptor(DatasetReference(dataset))
        locations = descriptor.get("locations", {})
        if not isinstance(locations, dict):
            raise TypeError("Dataset locations must be a mapping.")
        return locations

    def inspect(self, dataset: str) -> dict[str, Any]:
        """Inspect logical identity, placements and any reachable DatasetArtifact manifest."""
        reference = (
            DatasetReference.parse(dataset)
            if dataset.startswith("dataset:")
            else DatasetReference(dataset)
        )
        descriptor = self.catalog.descriptor(reference)
        locations = descriptor.get("locations", {})
        location_reports: dict[str, Any] = {}
        manifests: list[dict[str, Any]] = []
        if isinstance(locations, dict):
            for environment in sorted(locations):
                try:
                    location = self.catalog.resolve(reference, environment=environment)
                    candidate = location.local_path(
                        self.catalog.source.parent if self.catalog.source else Path.cwd()
                    )
                    manifest_path = (
                        candidate
                        if candidate.name == "dataset-artifact.json"
                        else candidate / "dataset-artifact.json"
                    )
                    size = (
                        candidate.stat().st_size
                        if candidate.is_file()
                        else sum(
                            item.stat().st_size for item in candidate.rglob("*") if item.is_file()
                        )
                        if candidate.is_dir()
                        else None
                    )
                    location_reports[environment] = {
                        **location.to_dict(),
                        "reachable": candidate.exists(),
                        "size_bytes": size,
                    }
                    if manifest_path.is_file():
                        manifests.append(DatasetArtifact.read_json(manifest_path).to_dict())
                except (OSError, TypeError, ValueError) as error:
                    location_reports[environment] = {"reachable": False, "error": str(error)}
        validation = {
            "valid": bool(descriptor.get("identity"))
            and all(
                not isinstance(value, dict) or "error" not in value
                for value in location_reports.values()
            ),
            "identity_declared": bool(descriptor.get("identity")),
        }
        return {
            "dataset": reference.name,
            "dataset_id": manifests[0].get("dataset_id") if manifests else None,
            "identity": descriptor.get("identity"),
            "producer": descriptor.get("producer"),
            "code_identity": descriptor.get("code_identity"),
            "preprocessing_config": descriptor.get("preprocessing_config"),
            "sample_count": manifests[0].get("sample_count") if manifests else None,
            "splits": manifests[0].get("splits") if manifests else descriptor.get("splits"),
            "artifacts": manifests[0].get("artifacts")
            if manifests
            else descriptor.get("artifacts", ()),
            "locations": location_reports,
            "validation": validation,
        }

    def replicate(
        self,
        dataset: str,
        *,
        source_environment: str,
        destination_environment: str,
        dry_run: bool = True,
    ) -> DataReplicationResult:
        """Preview or execute one transfer between already declared locations."""
        reference = DatasetReference(dataset)
        source = self.catalog.resolve(reference, environment=source_environment)
        destination = self.catalog.resolve(reference, environment=destination_environment)
        profile = self.clusters.for_data_environment(destination_environment)
        return self.transfer.replicate(reference, source, destination, profile, dry_run=dry_run)
