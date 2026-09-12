"""Reusable asynchronous screen primitives."""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import Thread
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import DataTable, Label, LoadingIndicator, Static


class EntitySelected(Message):
    """Request hierarchical navigation for one exact persisted entity."""

    def __init__(self, kind: str, value: Any) -> None:
        super().__init__()
        self.kind = kind
        self.value = value


class DataScreen(Vertical):
    """Load one table in a worker and retain stale data after transient failure."""

    TITLE = "LambdaForge"
    BINDINGS = [("right", "open_selected", "Open")]

    def __init__(self, loader: Callable[[], Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.loader = loader
        self.last_success: Any = None
        self._load_generation = 0
        self._loading = False
        self._last_updated_at: float | None = None
        self._last_refresh_error: str | None = None

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label(self.TITLE, classes="screen-title"),
            Static("Loading…", id="screen-status", classes="status-line"),
            LoadingIndicator(id="screen-loading"),
            DataTable(id="screen-table", cursor_type="row", zebra_stripes=True),
            Static(
                "Select a row for details. Press ? for contextual help.",
                id="screen-detail",
                classes="detail-panel",
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
        self._accept_snapshot(value)
        self.populate(self.query_one("#screen-table", DataTable), value)

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
