"""Unified managed-registry and external-catalog dataset resolution."""

from __future__ import annotations

import copy
from pathlib import Path, PurePosixPath

from lambdaforge.data.DataCatalog import DataCatalog
from lambdaforge.data.DatasetLocation import DatasetLocation
from lambdaforge.data.DatasetRecord import DatasetRecord
from lambdaforge.data.DatasetReference import DatasetReference
from lambdaforge.data.DatasetRegistry import DatasetRegistry
from lambdaforge.data.DatasetResolution import DatasetResolution
from lambdaforge.data.errors import (
    AmbiguousDatasetVersionError,
    MissingDatasetPlacementError,
    UnknownDatasetError,
)


class DatasetResolver:
    """Resolve every managed/external reference through one precedence and pinning policy."""

    def __init__(
        self,
        registry: DatasetRegistry | None = None,
        catalog: DataCatalog | None = None,
        *,
        environment: str = "local",
        managed_environment: str | None = None,
        source_dir: str | Path | None = None,
    ) -> None:
        self.source_dir = Path(source_dir or Path.cwd()).resolve()
        self.registry = registry or DatasetRegistry(DatasetRegistry.project_path(self.source_dir))
        self.catalog = catalog
        self.environment = str(environment)
        self.managed_environment = str(managed_environment or environment)

    def resolve(self, reference: DatasetReference | str) -> DatasetResolution:
        """Resolve an exact managed placement first, then an external DataCatalog entry."""
        parsed = DatasetReference.parse(reference) if isinstance(reference, str) else reference
        descriptor = self._descriptor(parsed)
        pinned = self._catalog_pin(parsed, descriptor)
        matches = tuple(
            record
            for record in self.registry.records()
            if record.name == pinned.name
            and (pinned.version is None or record.version == pinned.version)
            and (pinned.content_id is None or record.dataset_id == pinned.content_id)
        )
        if len(matches) > 1:
            raise AmbiguousDatasetVersionError(
                pinned.name, tuple(sorted(record.version for record in matches))
            )
        if matches:
            return self._managed(pinned, matches[0], descriptor)
        if descriptor is not None:
            location = self.catalog.resolve(pinned, environment=self.environment)  # type: ignore[union-attr]
            raw_identity = descriptor.get("identity", {})
            if not isinstance(raw_identity, dict):
                raise TypeError("DataCatalog identity must be a mapping.")
            identity = copy.deepcopy(raw_identity)
            exact = str(pinned)
            return DatasetResolution(
                pinned,
                exact,
                location,
                copy.deepcopy(identity),
                descriptor,
                managed=False,
            )
        known = tuple(record.key for record in self.registry.records())
        raise UnknownDatasetError(pinned.selector, known)

    def _managed(
        self,
        reference: DatasetReference,
        record: DatasetRecord,
        descriptor: dict[str, object] | None,
    ) -> DatasetResolution:
        placement = next(
            (value for value in record.placements if value.cluster == self.managed_environment),
            None,
        )
        if placement is None:
            raise MissingDatasetPlacementError(
                record.key,
                self.managed_environment,
                tuple(sorted(value.cluster for value in record.placements)),
            )
        root = placement.root
        if reference.subpath is not None:
            if "://" in root:
                root = f"{root.rstrip('/')}/{PurePosixPath(reference.subpath).as_posix()}"
            else:
                root = str(Path(root) / Path(*PurePosixPath(reference.subpath).parts))
        exact = f"dataset:{record.key}"
        if reference.subpath is not None:
            exact += f"/{reference.subpath}"
        identity = {
            "strategy": "dataset_id",
            "name": record.name,
            "version": record.version,
            "dataset_id": record.dataset_id,
            "content_id": record.dataset_id,
            "build_id": record.build_id,
        }
        effective_descriptor = dict(descriptor or {})
        for key in ("loader", "identity"):
            if key not in effective_descriptor and key in record.metadata:
                effective_descriptor[key] = copy.deepcopy(record.metadata[key])
        return DatasetResolution(
            reference,
            exact,
            DatasetLocation(self.managed_environment, root),
            identity,
            effective_descriptor,
            record,
            True,
        )

    def _descriptor(self, reference: DatasetReference) -> dict[str, object] | None:
        if self.catalog is None:
            return None
        try:
            return self.catalog.descriptor(reference)
        except KeyError:
            return None

    @staticmethod
    def _catalog_pin(
        reference: DatasetReference, descriptor: dict[str, object] | None
    ) -> DatasetReference:
        if descriptor is None:
            return reference
        alias = descriptor.get("selector", descriptor.get("alias"))
        if alias is not None:
            parsed = DatasetReference.parse(
                str(alias) if str(alias).startswith("dataset:") else f"dataset:{alias}"
            )
            return DatasetReference(
                parsed.name,
                reference.subpath or parsed.subpath,
                parsed.version,
                parsed.content_id,
            )
        version = reference.version or (
            str(descriptor["version"]) if descriptor.get("version") is not None else None
        )
        content_id = reference.content_id or (
            str(descriptor["dataset_id"]) if descriptor.get("dataset_id") is not None else None
        )
        return DatasetReference(reference.name, reference.subpath, version, content_id)
