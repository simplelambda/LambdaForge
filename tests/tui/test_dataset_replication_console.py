"""Native Dataset replication selection/confirmation/progress in the Research Console."""

import asyncio
import threading

from textual.widgets import Select

from lambdaforge.tui.App import LambdaForgeApp
from lambdaforge.tui.screens.DatasetReplication import DatasetReplication
from lambdaforge.tui.screens.Workspace import DatasetWorkspace, ExactConfirmation
from tests.tui.test_research_console_014 import FakeServices


def test_console_preview_confirmation_background_feedback_and_no_duplicate_apply():
    release = threading.Event()
    dataset = {
        "name": "example",
        "version": "7",
        "dataset_id": "sha256:" + "a" * 64,
        "placements": [{"cluster": "local", "root": "/owned/example"}],
    }

    class Services(FakeServices):
        def replicate_dataset(self, selector, **kwargs):
            self.calls.append(
                ("replicate", (selector, kwargs["apply"] if "apply" in kwargs else False))
            )
            assert kwargs["expected_content_id"] == dataset["dataset_id"]
            plan = {
                "dataset": selector,
                "target_cluster": kwargs["destination"],
                "source_cluster": kwargs["source"],
                "action": "REPLICATE",
                "transfer_route": "controller-stream",
                "compression": "tar-gzip-3",
            }
            if kwargs.get("apply"):
                kwargs["progress"](
                    {"message": "Streaming compressed bytes", "compressed_bytes": 500}
                )
                release.wait(timeout=4)
                plan["placements"] = [*dataset["placements"], {"cluster": "gpu12", "root": "/copy"}]
            return plan

    services = Services()

    async def exercise():
        app = LambdaForgeApp(services)
        async with app.run_test(size=(120, 42)) as pilot:
            workspace = DatasetWorkspace(dataset, services)
            app.push_screen(workspace)
            await pilot.pause(0.1)
            await pilot.click("#dataset-replicate")
            assert isinstance(app.screen, DatasetReplication)
            app.screen.query_one("#replicate-destination", Select).value = "gpu12"
            await pilot.click("#replicate-preview")
            await pilot.pause(0.2)
            assert isinstance(app.screen, ExactConfirmation)
            assert ("replicate", ("example@7", True)) not in services.calls
            assert "tar-gzip" in str(app.screen.query_one("#exact-preview").render())
            await pilot.click("#exact-apply")
            await pilot.pause(0.1)
            assert workspace.query_one("#dataset-replicate").disabled
            assert workspace.query_one("#dataset-delete").disabled
            assert "compressed" in str(workspace.query_one("#dataset-operation-status").render())
            workspace._choose_replication()
            assert app.screen is workspace
            release.set()
            await pilot.pause(0.2)
            assert not workspace.query_one("#dataset-replicate").disabled
            assert "Verified replica registered" in str(
                workspace.query_one("#dataset-operation-status").render()
            )
            assert "gpu12" in str(workspace.query_one("#dataset-locations Static").render())
            assert services.calls.count(("replicate", ("example@7", True))) == 1

    asyncio.run(exercise())


def test_preview_failure_is_visible_and_unlocks_controls():
    class Services(FakeServices):
        def replicate_dataset(self, *_args, **_kwargs):
            raise RuntimeError("destination content conflicts")

    async def exercise():
        services = Services()
        app = LambdaForgeApp(services)
        async with app.run_test() as pilot:
            workspace = DatasetWorkspace(
                {"name": "example", "version": "7", "placements": []}, services
            )
            app.push_screen(workspace)
            await pilot.pause(0.1)
            workspace._preview_replication(("local", "gpu12"))
            await pilot.pause(0.2)
            assert "destination content conflicts" in str(
                workspace.query_one("#dataset-operation-status").render()
            )
            assert not workspace.query_one("#dataset-replicate").disabled
            assert not workspace.query_one("#dataset-delete").disabled

    asyncio.run(exercise())


def test_leaving_workspace_does_not_abort_native_transfer():
    release, completed = threading.Event(), threading.Event()
    errors = []

    class Services(FakeServices):
        def replicate_dataset(self, selector, **kwargs):
            release.wait(timeout=4)
            try:
                kwargs["progress"]({"message": "Checking destination"})
            except Exception as error:
                errors.append(error)
                raise
            completed.set()
            return {"target_cluster": "gpu12", "applied": True}

    async def exercise():
        services = Services()
        app = LambdaForgeApp(services)
        async with app.run_test() as pilot:
            workspace = DatasetWorkspace({"name": "example", "version": "7"}, services)
            app.push_screen(workspace)
            await pilot.pause(0.1)
            workspace._apply_replication({"source": "local", "destination": "gpu12"}, True)
            app.pop_screen()
            await pilot.pause(0.1)
            release.set()
            await pilot.pause(0.2)
            assert completed.is_set() and not errors

    asyncio.run(exercise())


def test_endpoint_dialog_controls_fit_standard_terminal():
    async def exercise():
        app = LambdaForgeApp(FakeServices())
        async with app.run_test(size=(80, 24)) as pilot:
            app.push_screen(
                DatasetReplication("example@7", ("gpu12",), ("local", "gpu12", "gpu16"))
            )
            await pilot.pause(0.1)
            app.screen.query_one("#replicate-destination", Select).value = "gpu16"
            assert await pilot.click("#replicate-preview")

    asyncio.run(exercise())
