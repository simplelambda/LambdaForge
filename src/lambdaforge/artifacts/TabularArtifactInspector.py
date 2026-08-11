"""Bounded safe text-table artifact inspector."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from lambdaforge.artifacts.ArtifactInspection import ArtifactInspection
from lambdaforge.artifacts.ArtifactInspector import ArtifactInspector


class TabularArtifactInspector(ArtifactInspector):
    """Inspect CSV/TSV/JSON/JSONL without executing embedded objects."""

    def supports(self, path: Path, *, media_type: str | None = None) -> bool:
        """Recognize explicit text-table extensions."""
        del media_type
        return path.suffix.lower() in {".csv", ".tsv", ".json", ".jsonl"}

    def inspect(
        self,
        path: Path,
        *,
        item: str | None = None,
        rows: int = 20,
        slice_expression: str | None = None,
    ) -> ArtifactInspection:
        """Return columns and a bounded preview."""
        del item, slice_expression
        if rows < 0 or rows > 1000:
            raise ValueError("rows must be between 0 and 1000.")
        source = path.resolve()
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"Artifact is missing or symbolic: {source}")
        suffix = source.suffix.lower()
        preview: list[Any]
        if suffix in {".csv", ".tsv"}:
            with source.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t" if suffix == ".tsv" else ",")
                preview = []
                for index, row in enumerate(reader):
                    if index >= rows:
                        break
                    preview.append(dict(row))
                columns = tuple(reader.fieldnames or ())
        elif suffix == ".jsonl":
            preview = []
            with source.open(encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    if index >= rows:
                        break
                    preview.append(json.loads(line))
            columns = self._columns(preview)
        else:
            value = json.loads(source.read_text(encoding="utf-8"))
            sequence = value if isinstance(value, list) else [value]
            preview = sequence[:rows]
            columns = self._columns(preview)
        return ArtifactInspection(
            "Tabular text",
            str(source),
            source.stat().st_size,
            ({"columns": list(columns), "preview": preview, "preview_rows": len(preview)},),
        )

    @staticmethod
    def _columns(values: list[Any]) -> tuple[str, ...]:
        return tuple(
            sorted({str(key) for value in values if isinstance(value, dict) for key in value})
        )
