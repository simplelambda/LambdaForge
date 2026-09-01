"""Completed execution and Study Analysis browser."""

from __future__ import annotations

import json
from typing import Any

from textual.widgets import DataTable

from lambdaforge.tui.screens.Base import DataScreen


class ResultsScreen(DataScreen):
    """Browse completed Executions and enter their persisted analysis."""

    TITLE = "Results · completed executions, comparison and Study Analysis"

    def populate(self, table: DataTable[Any], value: Any) -> None:
        table.clear(columns=True)
        table.add_columns("Name", "Execution", "Status", "Runs", "Objective / best")
        for item in value:
            summary = item.get("summary", {})
            objective = summary.get("objective", {}) if isinstance(summary, dict) else {}
            best = summary.get("best_candidate") if isinstance(summary, dict) else None
            table.add_row(
                str(item.get("name", "-")),
                str(item.get("execution_id", "-")),
                str(item.get("status", "unknown")),
                str(len(item.get("runs", []))),
                str(best or objective.get("metric", "-")),
                key=str(item.get("execution_id", item.get("name", "-"))),
            )

    def detail(self, row: int) -> str:
        if not isinstance(self.last_success, list) or not 0 <= row < len(self.last_success):
            return "Result selection is unavailable."
        item = self.last_success[row]
        return (
            f"Execution: {item.get('execution_id')}\n"
            f"Status: {item.get('status')}\n"
            f"Scientific fingerprint: {item.get('scientific_fingerprint')}\n"
            f"Summary:\n{json.dumps(item.get('summary', {}), indent=2)}\n\n"
            "Use Ctrl+P → Analyze result or Report result for the selected execution."
        )

    def selected_selector(self) -> str | None:
        """Return the exact selected Execution ID, never an ambiguous display name."""
        if not isinstance(self.last_success, list | tuple):
            return None
        row = self.query_one("#screen-table", DataTable).cursor_row
        if not 0 <= row < len(self.last_success):
            return None
        value = self.last_success[row]
        selector = value.get("execution_id") if isinstance(value, dict) else None
        return str(selector) if selector else None
