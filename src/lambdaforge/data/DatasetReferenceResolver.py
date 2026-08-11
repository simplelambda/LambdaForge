"""Explicit dataset-reference resolution for experiment object specifications."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lambdaforge.data.DataCatalog import DataCatalog
from lambdaforge.data.DatasetReference import DatasetReference


class DatasetReferenceResolver:
    """Resolve only typed dataset markers, never strings that merely resemble paths."""

    def __init__(self, catalog: DataCatalog, *, environment: str, source_dir: Path) -> None:
        self.catalog = catalog
        self.environment = environment
        self.source_dir = source_dir
        self.bindings: list[dict[str, Any]] = []

    def resolve(self, value: Any, *, path: str = "") -> Any:
        """Resolve explicit ``{dataset, subpath}`` markers recursively."""
        if isinstance(value, Mapping):
            mapping = dict(value)
            if "dataset" in mapping and set(mapping).issubset({"dataset", "subpath"}):
                return self._physical(DatasetReference.from_mapping(mapping), path)
            return {
                str(key): self.resolve(item, path=f"{path}.{key}".strip("."))
                for key, item in mapping.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [
                self.resolve(item, path=f"{path}.{index}".strip("."))
                for index, item in enumerate(value)
            ]
        return copy.deepcopy(value)

    def resolve_split(self, value: Any, *, path: str) -> Any:
        """Resolve a direct ``dataset:NAME/subpath`` split through a catalog loader spec."""
        if not isinstance(value, str) or not value.startswith("dataset:"):
            return self.resolve(value, path=path)
        reference = DatasetReference.parse(value)
        descriptor = self.catalog.descriptor(reference)
        loader = descriptor.get("loader")
        if not isinstance(loader, Mapping):
            raise ValueError(
                f"Direct {value!r} at {path} requires a 'loader' object specification in "
                "the data catalog. Alternatively put an explicit {dataset, subpath} marker "
                "inside your dataset target params."
            )
        specification = copy.deepcopy(dict(loader))
        path_parameter = specification.pop("path_parameter", None)
        if not isinstance(path_parameter, str) or not path_parameter:
            raise ValueError(
                f"Dataset catalog loader for {reference.name!r} requires path_parameter."
            )
        params = specification.setdefault("params", {})
        if not isinstance(params, dict):
            raise TypeError("Dataset catalog loader params must be a mapping.")
        if path_parameter in params:
            raise ValueError(
                f"Dataset loader path_parameter {path_parameter!r} would overwrite params."
            )
        params[path_parameter] = self._physical(reference, path)
        return self.resolve(specification, path=path)

    def _physical(self, reference: DatasetReference, path: str) -> str:
        descriptor = self.catalog.descriptor(reference)
        location = self.catalog.resolve(reference, environment=self.environment)
        physical = str(location.local_path(self.source_dir))
        self.bindings.append(
            {
                "path": path,
                "reference": str(reference),
                "identity": copy.deepcopy(descriptor.get("identity", {})),
                "resolved_path": physical,
                "environment": self.environment,
            }
        )
        return physical
