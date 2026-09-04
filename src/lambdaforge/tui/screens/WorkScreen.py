"""Work and low-level Job operations."""

from __future__ import annotations

from typing import Any

from textual.widgets import DataTable, Static

from lambdaforge.tui.screens.Base import DataScreen, EntitySelected
from lambdaforge.tui.viewmodels import entity_key


class WorkScreen(DataScreen):
    """Present semantic Work executions separately from low-level Jobs."""

    TITLE = "Work · executions, Attempts, Jobs and logs"

    def __init__(self, loader: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(loader, *args, **kwargs)
        self._selected_key: str | None = None
        self._rendering = False

    def on_mount(self) -> None:
        super().on_mount()
        self.set_interval(2.0, self._refresh_visible)

    def _refresh_visible(self) -> None:
        if self.display and self.app.screen is self.app.screen_stack[0]:
            self.reload()

    @staticmethod
    def _items(value: Any) -> list[dict[str, Any]]:
        return (
            [
                item
                for item in value.get("work", {}).get("items", [])
                if not item.get("study_expected")
            ]
            if isinstance(value, dict)
            else []
        )

    def populate(self, table: DataTable[Any], value: Any) -> None:
        self._rendering = True
        table.clear(columns=True)
        table.add_columns("Work", "Cluster", "State", "Attempt", "Progress")
        items = self._items(value)
        for index, item in enumerate(items):
            table.add_row(
                str(item.get("name", "-")),
                str(item.get("cluster", "local")),
                str(item.get("state", "unknown")),
                str(len(item.get("attempt_history", []))),
                self._progress(item.get("progress")),
                key=entity_key(item, kind="work", fallback_index=index),
            )
        row = next(
            (index for index, item in enumerate(items) if self._key(item) == self._selected_key),
            0,
        )
        if items:
            table.move_cursor(row=row)
            self._selected_key = self._key(items[row])
        self._rendering = False
        self.query_one("#screen-detail", Static).update(self.detail(row))

    def detail(self, row: int) -> str:
        if not isinstance(self.last_success, dict):
            return "Work data is unavailable."
        items = self._items(self.last_success)
        if not 0 <= row < len(items):
            return "Work selection is unavailable."
        item = items[row]
        return (
            f"Work: {item.get('name')}\n"
            f"Work ID: {item.get('work_id', item.get('execution_id', '-'))}\n"
            f"Cluster: {item.get('cluster')}\n"
            f"State: {item.get('state')}\n"
            f"Attempts: {len(item.get('attempt_history', []))}\n\n"
            "Enter opens details; Ctrl+P exposes logs, cancel, retry, delete and Advanced Jobs."
        )

    def open_row(self, row: int) -> None:
        if isinstance(self.last_success, dict):
            items = self._items(self.last_success)
            if 0 <= row < len(items):
                self.post_message(EntitySelected("work", items[row]))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "screen-table" or self._rendering:
            return
        items = self._items(self.last_success)
        if 0 <= event.cursor_row < len(items):
            self._selected_key = self._key(items[event.cursor_row])
        self.query_one("#screen-detail", Static).update(self.detail(event.cursor_row))

    @staticmethod
    def _key(item: dict[str, Any]) -> str:
        return entity_key(item, kind="work")

    @staticmethod
    def _progress(value: Any) -> str:
        if not isinstance(value, dict):
            return "not reported"
        completed, total = value.get("completed"), value.get("total")
        unit = value.get("unit") or "items"
        if isinstance(completed, int) and isinstance(total, int) and total:
            return f"{completed}/{total} {unit} · {100 * completed / total:.0f}%"
        if isinstance(completed, int):
            return f"{completed} {unit} completed"
        return "not reported"
