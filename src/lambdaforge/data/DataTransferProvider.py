"""Explicit dataset transfer boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.data.DataReplicationResult import DataReplicationResult
from lambdaforge.data.DatasetLocation import DatasetLocation
from lambdaforge.data.DatasetReference import DatasetReference


class DataTransferProvider(ABC):
    """Replicate bytes only after a user explicitly requests it."""

    @abstractmethod
    def replicate(
        self,
        reference: DatasetReference,
        source: DatasetLocation,
        destination: DatasetLocation,
        destination_profile: ClusterProfile,
        *,
        dry_run: bool = True,
    ) -> DataReplicationResult:
        """Preview or apply one exact replication."""
