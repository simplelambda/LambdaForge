"""Work and low-level Job operations."""

from __future__ import annotations

from typing import Any

from textual.widgets import DataTable

from lambdaforge.tui.screens.Base import DataScreen


class WorkScreen(DataScreen):
    """Present semantic Work executions separately from low-level Jobs."""

    TITLE = "Work · executions, Attempts, Jobs and logs"

    def populate(self, table: DataTable[Any], value: Any) -> None:
        table.clear(columns=True)
        table.add_columns("Work", "Cluster", "State", "Attempt", "Progress")
        for item in value.get("work", {}).get("items", []):
            table.add_row(
                str(item.get("name", "-")),
                str(item.get("cluster", "local")),
                str(item.get("state", "unknown")),
                str(len(item.get("attempt_history", []))),
                str(item.get("progress", "-")),
                key=str(item.get("execution_id", item.get("name", "-"))),
            )

    def detail(self, row: int) -> str:
        if not isinstance(self.last_success, dict):
            return "Work data is unavailable."
        items = self.last_success.get("work", {}).get("items", [])
        if not 0 <= row < len(items):
            return "Work selection is unavailable."
        item = items[row]
        return (
            f"Work: {item.get('name')}\n"
            f"Execution: {item.get('execution_id')}\n"
            f"Cluster: {item.get('cluster')}\n"
            f"State: {item.get('state')}\n"
            f"Attempts: {len(item.get('attempt_history', []))}\n\n"
            "Enter opens details; Ctrl+P exposes logs, cancel, retry, delete and Advanced Jobs."
        )
