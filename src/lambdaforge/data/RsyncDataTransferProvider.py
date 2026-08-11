"""Explicit rsync dataset transfer provider."""

from __future__ import annotations

import subprocess

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.data.DataReplicationResult import DataReplicationResult
from lambdaforge.data.DatasetLocation import DatasetLocation
from lambdaforge.data.DatasetReference import DatasetReference
from lambdaforge.data.DataTransferProvider import DataTransferProvider


class RsyncDataTransferProvider(DataTransferProvider):
    """Copy a declared local source to a declared local or SSH destination."""

    def replicate(
        self,
        reference: DatasetReference,
        source: DatasetLocation,
        destination: DatasetLocation,
        destination_profile: ClusterProfile,
        *,
        dry_run: bool = True,
    ) -> DataReplicationResult:
        """Use rsync argument vectors; never infer or mutate catalogue locations."""
        if source.environment != "local":
            raise ValueError("The built-in rsync provider currently requires a local source.")
        source_value = source.uri.removeprefix("file://")
        destination_value = destination.uri.removeprefix("file://")
        if destination_profile.transport == "ssh":
            destination_value = f"{destination_profile.host}:{destination_value}"
        if dry_run:
            return DataReplicationResult(
                str(reference), source_value, destination_value, False, message="preview"
            )
        completed = subprocess.run(
            ("rsync", "-a", "--protect-args", "--", source_value, destination_value),
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        return DataReplicationResult(
            str(reference),
            source_value,
            destination_value,
            True,
            completed.returncode,
            completed.stdout or completed.stderr,
        )
