"""Console consumes the real native chained plan/error, not a mocked successful preview."""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("textual")
from textual.widgets import Button, Checkbox, Static  # noqa: E402

from lambdaforge.tui.App import LambdaForgeApp  # noqa: E402
from lambdaforge.tui.screens.Workspace import StudyRetryConfirmation, WorkWorkspace  # noqa: E402
from tests.tui.test_research_console_014 import FakeServices  # noqa: E402
from tests.work.test_chained_staged_recovery import chain  # noqa: E402, F401


@pytest.mark.parametrize("corrupt", [False, True])
def test_console_native_chain_plan_or_meaningful_block(chain, corrupt):  # noqa: F811
    calls = []
    if corrupt:
        path = chain.configs[1].source.parent.parent / "request.json"
        value = json.loads(path.read_text())
        value["job_id"] = "job-forged"
        path.write_text(json.dumps(value))

    class Services(FakeServices):
        def work_logs(self, *_args):
            return {"text": "native failed Attempt log"}

        def retry_preview(self, job_id, **options):
            return chain.jobs.retry_preview(job_id, **options)

        def retry_job(self, job_id, **options):
            calls.append((job_id, options))
            return {"job_id": "job-next", "state": "preparing"}

    async def exercise():
        services = Services()
        app = LambdaForgeApp(services)
        async with app.run_test(size=(160, 55)) as pilot:
            workspace = WorkWorkspace(
                {
                    "name": "six-role",
                    "state": "failed",
                    "primary_job_id": "job-stage-2",
                    "cluster": "fixture",
                },
                services,
            )
            app.push_screen(workspace)
            await pilot.pause()
            await pilot.click("#work-retry")
            for _ in range(80):
                await pilot.pause(0.05)
                if isinstance(app.screen, StudyRetryConfirmation) or not workspace._recovery_busy:
                    break
            assert not calls
            if corrupt:
                assert app.screen is workspace
                text = str(workspace.query_one("#work-operational-health", Static).render())
                assert "Attempt attempt-0002" in text and "job-stage-2" in text
                assert "request" in text
                assert not workspace.query_one("#work-retry", Button).disabled
            else:
                assert isinstance(app.screen, StudyRetryConfirmation)
                dialog = app.screen
                assert dialog.preview["runs"][0]["next_attempt"] == 3
                dialog.query_one("#retry-compatible-code", Checkbox).value = True
                await pilot.pause()
                dialog.on_button_pressed(Button.Pressed(dialog.query_one("#retry-confirm", Button)))
                await pilot.pause()
                assert len(calls) == 1 and calls[0][0] == "job-stage-2"

    asyncio.run(exercise())
