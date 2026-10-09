"""Conflict actions remain visible, confirmed, pinned and guarded against double clicks."""

import asyncio
import threading

from textual.widgets import Select

from lambdaforge.tui.App import LambdaForgeApp
from lambdaforge.tui.screens.DatasetManagement import DatasetManagement
from lambdaforge.tui.screens.Workspace import DatasetWorkspace, ExactConfirmation
from tests.tui.test_research_console_014 import FakeServices


def test_conflict_preview_confirm_pins_identity_and_prevents_duplicates():
    release = threading.Event()
    record = {
        "name": "corpus",
        "version": "6",
        "dataset_id": "sha256:" + "a" * 64,
        "inventory_conflict": True,
        "placements": [{"cluster": "gpu12", "root": "/owned/copy"}],
    }

    class Services(FakeServices):
        def manage_dataset_copy(self, selector, **kwargs):
            self.calls.append((selector, kwargs))
            assert kwargs["content_id"] == record["dataset_id"]
            if kwargs.get("apply"):
                assert kwargs["expected_root"] == "/owned/copy"
                release.wait(timeout=4)
                raise ValueError("target changed since preview")
            return {
                "safe": True,
                "root": "/owned/copy",
                "content_id": kwargs["content_id"],
                "notice": "Delete only this exact managed placement.",
            }

    async def exercise():
        services = Services()
        app = LambdaForgeApp(services)
        async with app.run_test(size=(100, 40)) as pilot:
            workspace = DatasetWorkspace(record, services)
            app.push_screen(workspace)
            await pilot.pause(0.1)
            assert workspace.query_one("#dataset-delete").disabled
            assert not workspace.query_one("#dataset-manage").disabled
            await pilot.click("#dataset-manage")
            assert isinstance(app.screen, DatasetManagement)
            app.screen.query_one("#manage-operation", Select).value = "delete"
            await pilot.click("#manage-preview")
            await pilot.pause(0.2)
            assert isinstance(app.screen, ExactConfirmation)
            assert sum(name == "corpus@6" for name, _ in services.calls) == 1
            await pilot.click("#exact-apply")
            await pilot.pause(0.1)
            assert workspace.query_one("#dataset-manage").disabled
            assert "Applying" in str(workspace.query_one("#dataset-operation-status").render())
            workspace._choose_management()
            assert app.screen is workspace
            release.set()
            await pilot.pause(0.2)
            assert "target changed" in str(
                workspace.query_one("#dataset-operation-status").render()
            )
            assert not workspace.query_one("#dataset-manage").disabled
            assert workspace.query_one("#dataset-delete").disabled
            assert (
                sum(
                    bool(kwargs.get("apply"))
                    for name, kwargs in services.calls
                    if name == "corpus@6"
                )
                == 1
            )

    asyncio.run(exercise())


def test_adoption_confirmation_cancel_is_read_only_and_small_dialog_fits():
    class Services(FakeServices):
        def manage_dataset_copy(self, selector, **kwargs):
            assert not kwargs.get("apply")
            return {"safe": True, "previous_content_id": "sha256:old", "root": "/chosen"}

    async def exercise():
        services = Services()
        app = LambdaForgeApp(services)
        async with app.run_test(size=(80, 24)) as pilot:
            workspace = DatasetWorkspace(
                {
                    "name": "corpus",
                    "version": "6",
                    "dataset_id": "sha256:new",
                    "inventory_conflict": True,
                },
                services,
            )
            app.push_screen(workspace)
            await pilot.pause(0.1)
            workspace._choose_management()
            await pilot.pause(0.1)
            assert app.screen.query_one("#manage-preview").region.right <= 80
            await pilot.click("#manage-preview")
            await pilot.pause(0.2)
            assert isinstance(app.screen, ExactConfirmation)
            await pilot.click("#exact-cancel")
            await pilot.pause(0.1)
            assert app.screen is workspace
            assert not workspace.query_one("#dataset-manage").disabled
            assert "cancelled" in str(workspace.query_one("#dataset-operation-status").render())

    asyncio.run(exercise())
