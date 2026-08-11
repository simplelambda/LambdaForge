"""Entry-point registry for artifact inspectors, visualizers, schemas and exporters."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Iterable
from typing import Any


class ArtifactPluginRegistry:
    """Discover artifact extensions without importing unrelated provider modules."""

    GROUPS = {
        "inspector": "lambdaforge.artifact_inspectors",
        "visualizer": "lambdaforge.artifact_visualizers",
        "schema": "lambdaforge.artifact_schemas",
        "exporter": "lambdaforge.artifact_exporters",
        "validator": "lambdaforge.artifact_validators",
    }

    def names(self, kind: str | None = None) -> tuple[dict[str, str], ...]:
        """List metadata without loading extension modules."""
        kinds = (kind,) if kind is not None else tuple(self.GROUPS)
        output: list[dict[str, str]] = []
        for selected in kinds:
            if selected not in self.GROUPS:
                raise ValueError(f"Unknown artifact plugin kind {selected!r}.")
            for entry in self._entries(self.GROUPS[selected]):
                output.append({"kind": selected, "name": entry.name, "value": entry.value})
        return tuple(sorted(output, key=lambda value: (value["kind"], value["name"])))

    def load(self, kind: str, name: str) -> Any:
        """Load exactly one explicitly requested extension."""
        if kind not in self.GROUPS:
            raise ValueError(f"Unknown artifact plugin kind {kind!r}.")
        matches = tuple(entry for entry in self._entries(self.GROUPS[kind]) if entry.name == name)
        if len(matches) != 1:
            raise LookupError(
                f"Expected one artifact {kind} plugin {name!r}, found {len(matches)}."
            )
        return matches[0].load()

    @staticmethod
    def _entries(group: str) -> Iterable[importlib.metadata.EntryPoint]:
        entries = importlib.metadata.entry_points()
        return entries.select(group=group) if hasattr(entries, "select") else entries.get(group, ())
