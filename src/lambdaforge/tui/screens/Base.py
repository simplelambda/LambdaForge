"""Reusable asynchronous screen primitives."""

from __future__ import annotations

from collections.abc import Callable
from threading import Thread
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import DataTable, Label, LoadingIndicator, Static


class DataScreen(Vertical):
    """Load one table in a worker and retain stale data after transient failure."""

    TITLE = "LambdaForge"

    def __init__(self, loader: Callable[[], Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.loader = loader
        self.last_success: Any = None
        self._load_generation = 0

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
        )

    def on_mount(self) -> None:
        self.reload()

    def reload(self) -> None:
        self._load_generation += 1
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
            self._loaded(message.value)

    def on_data_screen_failed(self, message: Failed) -> None:
        if message.generation == self._load_generation:
            self._failed(message.error)

    def _loaded(self, value: Any) -> None:
        self.last_success = value
        self.query_one("#screen-loading", LoadingIndicator).display = False
        self.query_one("#screen-status", Static).update("Up to date")
        self.populate(self.query_one("#screen-table", DataTable), value)

    def _failed(self, error: Exception) -> None:
        self.query_one("#screen-loading", LoadingIndicator).display = False
        self.query_one("#screen-status", Static).update(
            f"Temporarily unavailable · {type(error).__name__}: {error} · "
            "showing last successful data"
        )

    def populate(self, table: DataTable[Any], value: Any) -> None:
        raise NotImplementedError

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
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
