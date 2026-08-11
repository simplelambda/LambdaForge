"""Configurable logical dataset catalogue."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from lambdaforge.data.DatasetLocation import DatasetLocation
from lambdaforge.data.DatasetReference import DatasetReference


class DataCatalog:
    """Map portable dataset names to identities and environment-specific locations."""

    def __init__(self, datasets: Mapping[str, Any], *, source: str | Path | None = None) -> None:
        self._datasets = copy.deepcopy(dict(datasets))
        self.source = Path(source).resolve() if source is not None else None

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataCatalog:
        """Load a duplicate-key-safe enough trusted local catalogue YAML."""
        source = Path(path).resolve()
        value = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(value, Mapping) or not isinstance(value.get("datasets"), Mapping):
            raise ValueError("A data catalog requires a top-level datasets mapping.")
        return cls(value["datasets"], source=source)

    def names(self) -> tuple[str, ...]:
        """Return stable registered dataset names."""
        return tuple(sorted(str(name) for name in self._datasets))

    def descriptor(self, reference: DatasetReference | str) -> dict[str, Any]:
        """Return a detached logical dataset descriptor."""
        parsed = DatasetReference.parse(reference) if isinstance(reference, str) else reference
        try:
            value = self._datasets[parsed.name]
        except KeyError as error:
            raise KeyError(f"Unknown dataset reference {parsed!s}.") from error
        if not isinstance(value, Mapping):
            raise TypeError(f"Dataset catalog entry {parsed.name!r} must be a mapping.")
        return copy.deepcopy(dict(value))

    def resolve(
        self, reference: DatasetReference | str, *, environment: str = "local"
    ) -> DatasetLocation:
        """Resolve one physical location without changing logical identity."""
        descriptor = self.descriptor(reference)
        locations = descriptor.get("locations", {})
        if not isinstance(locations, Mapping) or environment not in locations:
            raise KeyError(f"Dataset {reference!s} has no location for {environment!r}.")
        value = locations[environment]
        if isinstance(value, str):
            return DatasetLocation(environment, value)
        if not isinstance(value, Mapping) or "uri" not in value:
            raise TypeError("Dataset locations must be a URI string or mapping with uri.")
        return DatasetLocation(environment, str(value["uri"]), bool(value.get("shared", False)))
