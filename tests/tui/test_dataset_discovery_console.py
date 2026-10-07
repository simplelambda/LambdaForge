"""All dataset identities remain visible without ambiguous Console reads or writes."""

import asyncio

from textual.app import App
from textual.widgets import DataTable

from lambdaforge.tui.App import LambdaForgeApp
from lambdaforge.tui.screens.DatasetScreen import DatasetScreen
from lambdaforge.tui.screens.Workspace import DatasetWorkspace
from tests.tui.test_research_console_014 import FakeServices


def test_dataset_table_keeps_same_version_conflicting_rows():
    rows = [
        {
            "name": "corpus",
            "version": "6",
            "dataset_id": "sha256:" + digest * 64,
            "inventory_conflict": True,
            "placements": [{"cluster": cluster}],
        }
        for digest, cluster in (("a", "gpu12"), ("b", "gpu16"))
    ]

    class Harness(App):
        def compose(self):
            yield DatasetScreen(lambda: rows)

    async def exercise():
        app = Harness()
        async with app.run_test() as pilot:
            screen = app.query_one(DatasetScreen)
            screen.reload()
            await pilot.pause(0.1)
            table = screen.query_one(DataTable)
            assert table.row_count == 2
            assert "gpu12" in str(table.get_row_at(0))
            assert "gpu16" in str(table.get_row_at(1))
            assert "CONFLICT" in str(table.get_row_at(1))

    asyncio.run(exercise())


def test_conflict_workspace_never_reads_another_identity_by_logical_selector():
    services = FakeServices()
    dataset = {
        "name": "corpus",
        "version": "6",
        "inventory_conflict": True,
        "dataset_id": "sha256:" + "a" * 64,
        "placements": [{"cluster": "gpu16"}],
    }

    async def exercise():
        app = LambdaForgeApp(services)
        async with app.run_test(size=(120, 38)) as pilot:
            app.push_screen(DatasetWorkspace(dataset, services))
            await pilot.pause(0.1)
            assert app.screen.query_one("#dataset-delete").disabled
            app.screen.query_one("#dataset-tabs").active = "dataset-members"
            await pilot.pause(0.1)
            assert "conflicting content" in str(
                app.screen.query_one("#dataset-member-content").render()
            )
            assert not any(name.startswith("dataset_") for name, _ in services.calls)

    asyncio.run(exercise())
