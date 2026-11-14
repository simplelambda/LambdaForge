"""Ordinary Python preview and native Work output; no editor, interpreter or hidden network.

Use in a new installed/importable consumer project. `lf run` calls TableReport.run, not a
free function. Column declarations are explicit; model training/splits remain the consumer's
normal scientific code. This example never writes a source database or publishes a dataset.
"""

from pathlib import Path
from typing import Any

from lambdaforge import Work
from lambdaforge.etl import FilterTable, PreviewBudget, PreviewSession, ReadRecords, SortTable
from lambdaforge.etl.tables import Column, TableSchema


class TableReport(Work):
    """Read a bounded typed table and persist honest prefix evidence as a native report."""

    def run(self, source: Path, delimiter: str = ",", limit: int = 100) -> dict[str, Any]:  # type: ignore[override]
        schema = TableSchema(
            [Column("id"), Column("value", "number", nullable=True, null_values=("NA", ""))]
        )
        budget = PreviewBudget(
            max_rows=limit, max_bytes=1024**2, max_seconds=5, cancelled=lambda: self.stop_requested
        )
        session = PreviewSession([source.parent], budget)
        report = session.records(
            ReadRecords("csv", csv={"delimiter": delimiter}, schema=schema.to_dict()), source
        )
        rows = FilterTable({"column": "value", "op": "not_null"})(report["rows"])
        report["visible_sorted_rows"] = SortTable(["value"])(rows)
        self.outputs.file("preview", filename="preview.json", role="report").write_json(report)
        self.metrics.log("preview_rows", len(report["rows"]))
        return {
            "preview_rows": len(report["rows"]),
            "scope": "prefix",
            "complete": report["complete"],
        }
