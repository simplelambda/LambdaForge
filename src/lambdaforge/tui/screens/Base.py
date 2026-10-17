"""Reusable asynchronous screen primitives."""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import Thread
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import DataTable, Input, Label, LoadingIndicator, Static


class EntitySelected(Message):
    """Request hierarchical navigation for one exact persisted entity."""

    def __init__(self, kind: str, value: Any) -> None:
        super().__init__()
        self.kind = kind
        self.value = value


class DataScreen(Vertical):
    """Load one table in a worker and retain stale data after transient failure."""

    TITLE = "LambdaForge"
    FILTERABLE = False
    BINDINGS = [("right", "open_selected", "Open")]

    def __init__(self, loader: Callable[[], Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.loader = loader
        self.last_success: Any = None
        self._load_generation = 0
        self._loading = False
        self._last_updated_at: float | None = None
        self._last_refresh_error: str | None = None
        self._filter_text = ""

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label(self.TITLE, classes="screen-title"),
            Static("Loading…", id="screen-status", classes="status-line"),
            LoadingIndicator(id="screen-loading"),
            Input(
                placeholder="Filter name, state, cluster or exact identity…",
                id="screen-filter",
                disabled=not self.FILTERABLE,
            ),
            DataTable(id="screen-table", cursor_type="row", zebra_stripes=True),
            Static(
                "Select a row for details. Press ? for contextual help.",
                id="screen-detail",
                classes="detail-panel",
                markup=False,
            ),
            Static(
                "Waiting for the first update",
                id="screen-freshness",
                classes="freshness-line",
            ),
        )

    def on_mount(self) -> None:
        # LambdaForgeApp mounts every root browser once, then reveals only one.  Loading
        # here would start duplicate overview/dataset/cluster probes for hidden screens
        # and could contend with an immediately requested bootstrap. ``show_screen`` is
        # the single activation boundary and calls ``reload`` for the visible browser.
        self.set_interval(1.0, self._update_freshness)
        for widget in self.query("#screen-filter"):
            widget.display = self.FILTERABLE

    def on_unmount(self) -> None:
        # Discard a late presentation result. This never cancels a Job or scientific worker.
        self._load_generation += 1
        self._loading = False

    def filter_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        query = self._filter_text.casefold().strip()
        if not query:
            return items
        return [
            item
            for item in items
            if query
            in " ".join(
                str(item.get(key, ""))
                for key in ("name", "state", "cluster", "work_id", "execution_id", "primary_job_id")
            ).casefold()
        ]

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "screen-filter" or not self.FILTERABLE:
            return
        self._filter_text = event.value
        if self.last_success is not None:
            self.populate(self.query_one("#screen-table", DataTable), self.last_success)

    def reload(self) -> None:
        if self._loading:
            return
        self._loading = True
        self._load_generation += 1
        if self.is_mounted:
            first_load = self.last_success is None
            self.query_one("#screen-loading", LoadingIndicator).display = first_load
            if first_load:
                self.query_one("#screen-status", Static).update("Loading…")
            self._update_freshness()
        Thread(
            target=self._load,
            args=(self._load_generation,),
            daemon=True,
            name=f"lambdaforge-tui-{self.id}",
        ).start()

    def _load(self, generation: int) -> None:
        try:
            value = self.loader()
        except Exception as error:
            self.post_message(self.Failed(error, generation))
        else:
            self.post_message(self.Loaded(value, generation))

    def on_data_screen_loaded(self, message: Loaded) -> None:
        if message.generation == self._load_generation:
            self._loading = False
            self._loaded(message.value)

    def on_data_screen_failed(self, message: Failed) -> None:
        if message.generation == self._load_generation:
            self._loading = False
            self._failed(message.error)

    def _loaded(self, value: Any) -> None:
        changed = self._last_updated_at is None or value != self.last_success
        table = self.query_one("#screen-table", DataTable)
        scroll = table.scroll_offset
        self._accept_snapshot(value)
        if changed:
            self.populate(table, value)
            table.scroll_to(x=scroll.x, y=scroll.y, animate=False, immediate=True)

    def _accept_snapshot(self, value: Any, *, status: str = "Up to date") -> None:
        """Record one successful snapshot without blanking an existing view."""
        self.last_success = value
        self._last_updated_at = time.monotonic()
        self._last_refresh_error = None
        self.query_one("#screen-loading", LoadingIndicator).display = False
        self.query_one("#screen-status", Static).update(status)
        self._update_freshness()

    def _failed(self, error: Exception) -> None:
        self.query_one("#screen-loading", LoadingIndicator).display = False
        self._last_refresh_error = f"{type(error).__name__}: {error}"
        suffix = " · showing the last successful snapshot" if self.last_success is not None else ""
        self.query_one("#screen-status", Static).update(
            f"Temporarily unavailable · {self._last_refresh_error}{suffix}"
        )
        self._update_freshness()

    def _update_freshness(self) -> None:
        if not self.is_mounted:
            return
        if self._last_updated_at is None:
            text = "Loading the first snapshot…" if self._loading else "No successful update yet"
        else:
            elapsed = max(0, int(time.monotonic() - self._last_updated_at))
            if elapsed < 5:
                age = "just now"
            elif elapsed < 60:
                age = f"{elapsed}s ago"
            else:
                minutes, seconds = divmod(elapsed, 60)
                age = f"{minutes}m {seconds:02d}s ago"
            text = f"Last successful update: {age}"
            if self._loading:
                text += " · refreshing in the background"
            elif self._last_refresh_error:
                text += " · refresh failed; retained data is shown"
        self.query_one("#screen-freshness", Static).update(text)

    def populate(self, table: DataTable[Any], value: Any) -> None:
        raise NotImplementedError

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "screen-table":
            return
        detail = self.detail(event.cursor_row)
        self.query_one("#screen-detail", Static).update(detail)
        self.open_row(event.cursor_row)

    def open_row(self, row: int) -> None:
        """Open the selected row; every concrete domain screen supplies real navigation."""
        raise NotImplementedError(f"{type(self).__name__} must implement hierarchical navigation.")

    def action_open_selected(self) -> None:
        self.open_row(self.query_one("#screen-table", DataTable).cursor_row)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Keep a compact preview in sync while arrows move between rows."""
        if event.data_table.id != "screen-table":
            return
        self.query_one("#screen-detail", Static).update(self.detail(event.cursor_row))

    def detail(self, row: int) -> str:
        return "No additional persisted details are available for this selection."

    class Loaded(Message):
        def __init__(self, value: Any, generation: int) -> None:
            super().__init__()
            self.value = value
            self.generation = generation

    class Failed(Message):
        def __init__(self, error: Exception, generation: int) -> None:
            super().__init__()
            self.error = error
            self.generation = generation
