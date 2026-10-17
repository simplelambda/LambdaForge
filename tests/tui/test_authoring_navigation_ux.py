"""Metadata-only local filters, preserved root view and discarded late presentation results."""

from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

pytest.importorskip("textual")
from textual.app import App, ComposeResult  # noqa: E402
from textual.widgets import DataTable, Input  # noqa: E402

from lambdaforge.tui.screens.Base import DataScreen, EntitySelected  # noqa: E402
from lambdaforge.tui.screens.WorkScreen import WorkScreen  # noqa: E402


def test_work_filter_is_local_opens_exact_row_and_refresh_keeps_cursor_scroll():
    async def exercise():
        snapshot = {
            "work": {
                "items": [
                    {
                        "work_id": f"work-{index}",
                        "name": f"work_name_{index}",
                        "cluster": "local",
                        "state": "succeeded",
                        "progress": {"completed": 1, "total": 1},
                        "attempt_history": [],
                    }
                    for index in range(40)
                ]
            }
        }
        calls = []
        selected = []

        def load():
            calls.append("load")
            return snapshot

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield WorkScreen(load, id="work")

            def on_entity_selected(self, event: EntitySelected):
                selected.append(event.value["work_id"])

        app = TestApp()
        async with app.run_test(size=(110, 30)) as pilot:
            screen = app.query_one(WorkScreen)
            screen.reload()
            await pilot.pause()
            table = screen.query_one("#screen-table", DataTable)
            assert table.row_count == 40
            table.move_cursor(row=20)
            table.scroll_to(y=10, animate=False, immediate=True)
            await pilot.pause()
            previous_scroll = table.scroll_offset
            screen._loaded(deepcopy(snapshot))
            assert table.cursor_row == 20 and table.scroll_offset == previous_scroll
            count = len(calls)
            screen.query_one("#screen-filter", Input).value = "work_name_19"
            await pilot.pause()
            assert table.row_count == 1 and len(calls) == count
            screen.action_open_selected()
            await pilot.pause()
            assert selected[-1] == "work-19"
            screen._loaded(deepcopy(snapshot))
            assert table.row_count == 1
            generation = screen._load_generation
            await screen.remove()
            screen.on_data_screen_loaded(DataScreen.Loaded({}, generation))
            assert screen.last_success == snapshot

    asyncio.run(exercise())
