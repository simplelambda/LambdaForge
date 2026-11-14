"""Sequential submission is an explicit Console choice over the native service."""

import asyncio
from pathlib import Path

import pytest
from textual.widgets import Checkbox

from lambdaforge.tui.App import LambdaForgeApp, WorkLaunchDialog
from tests.tui.test_research_console_014 import FakeServices


def test_launch_dialog_passes_queue_choice_once(tmp_path: Path):
    class Services(FakeServices):
        def submit_work(self, config, cluster, **kwargs):
            self.calls.append(("queue_submit", (config, cluster, kwargs)))
            return {"job_id": "job-queue-fixture", "state": "preparing", "execution_queue": True}

    services = Services()
    config = tmp_path / "work.yaml"
    config.write_text("run: tests.work_cases.Producer\n")

    async def exercise():
        app = LambdaForgeApp(services)
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.click("#nav-run")
            assert isinstance(app.screen, WorkLaunchDialog)
            app.screen.query_one("#launch-config").value = str(config)
            app.screen.query_one("#launch-execution-queue", Checkbox).value = True
            await pilot.pause()
            await pilot.click("#launch-submit")
            await pilot.pause(0.3)
            assert ("validate_work", config) in services.calls
            assert ("queue_submit", (config, "local", {"execution_queue": True})) in services.calls
            assert len([row for row in services.calls if row[0] == "queue_submit"]) == 1
            assert app.screen is app.screen_stack[0]

    asyncio.run(exercise())


@pytest.mark.parametrize("size", [(100, 30), (140, 55)])
def test_queue_view_is_metadata_only_and_has_mouse_back_control(size):
    class Services(FakeServices):
        def execution_queue(self, cluster):
            self.calls.append(("execution_queue", cluster))
            return {
                "entries": [
                    {
                        "job_id": "job-a",
                        "name": "Training A",
                        "phase": "dispatched",
                        "last_known_state": "running",
                    }
                ]
            }

    services = Services()

    async def exercise():
        from textual.widgets import DataTable

        from lambdaforge.tui.App import ExecutionQueueDialog

        app = LambdaForgeApp(services)
        async with app.run_test(size=size) as pilot:
            await pilot.click("#nav-run")
            button = app.screen.query_one("#launch-view-execution-queue")
            button.scroll_visible(animate=False)
            await pilot.pause()
            await pilot.click("#launch-view-execution-queue")
            await pilot.pause(0.25)
            assert isinstance(app.screen, ExecutionQueueDialog)
            assert app.screen.query_one("#execution-queue-table", DataTable).row_count == 1
            assert ("execution_queue", "local") in services.calls
            assert not any(row[0] == "submit_work" for row in services.calls)
            await pilot.click("#execution-queue-close")
            await pilot.pause()
            assert isinstance(app.screen, WorkLaunchDialog)

    asyncio.run(exercise())
