"""Application service for logical dataset operations."""

from __future__ import annotations

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.data.DataCatalog import DataCatalog
from lambdaforge.data.DataReplicationResult import DataReplicationResult
from lambdaforge.data.DatasetReference import DatasetReference
from lambdaforge.data.DataTransferProvider import DataTransferProvider
from lambdaforge.data.RsyncDataTransferProvider import RsyncDataTransferProvider


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
