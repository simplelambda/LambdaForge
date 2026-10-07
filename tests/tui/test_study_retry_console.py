"""Console Study recovery calls domain services without remote computation."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("textual")
from textual.widgets import Button, Checkbox  # noqa: E402

from lambdaforge.tui.App import LambdaForgeApp  # noqa: E402
from lambdaforge.tui.screens.Workspace import StudyRetryConfirmation, StudyWorkspace  # noqa: E402
from tests.tui.test_research_console_014 import FakeServices  # noqa: E402


@pytest.mark.parametrize("accept_code_change", [False, True])
@pytest.mark.parametrize("strategy", ["adaptive", "repeated", "sweep"])
def test_resume_dialog_submits_once_and_exposes_compatibility(
    accept_code_change: bool, strategy: str
) -> None:
    class Services(FakeServices):
        def retry_preview(self, job_id):
            self.calls.append(("retry_preview", job_id))
            return (
                {"resumable": True, "completed_attempts": 20, "pending_actions": 2}
                if strategy == "adaptive"
                else {
                    "resumable": True,
                    "strategy": strategy,
                    "reuse_runs": 9,
                    "retry_runs": 1,
                    "pending_runs": 0,
                    "runs": [{"trial": 1, "seed": 54, "action": "retry"}],
                }
            )

        def retry_job(self, job_id, *, accept_code_change=False):
            self.calls.append(("retry_job", (job_id, accept_code_change)))
            return {"job_id": "job-recovered", "state": "preparing"}

        def study(self, job_id):
            return {"candidates": [], "counts": {}, "job_state": "preparing"}

    services = Services()
    work = {
        "work_id": "work-study",
        "name": "Study",
        "state": "failed",
        "primary_job_id": "job-original",
        "study": {"candidates": [], "counts": {}},
    }

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test(size=(130, 38)) as pilot:
            app.push_screen(StudyWorkspace(work, services))
            await pilot.pause()
            assert not app.screen.query_one("#study-retry", Button).disabled
            await pilot.click("#study-retry")
            await pilot.pause()
            assert isinstance(app.screen, StudyRetryConfirmation)
            if strategy != "adaptive":
                assert "Reuse 9" in str(app.screen.query_one("#retry-plan-summary").render())
            app.screen.query_one("#retry-compatible-code", Checkbox).value = accept_code_change
            await pilot.click("#retry-confirm")
            await pilot.pause()
            assert isinstance(app.screen, StudyWorkspace)
            assert services.calls.count(("retry_job", ("job-original", accept_code_change))) == 1
            assert app.screen.work["primary_job_id"] == "job-recovered"
            assert app.screen.query_one("#study-retry", Button).disabled
            assert not app.screen.query_one("#study-cancel", Button).disabled

    asyncio.run(exercise())


def test_failed_recovery_preview_is_visible_and_can_be_retried() -> None:
    class Services(FakeServices):
        def retry_preview(self, job_id):
            raise ValueError("original HPO state is missing")

    services = Services()
    work = {
        "name": "Study",
        "state": "failed",
        "primary_job_id": "job-original",
        "study": {"candidates": [], "counts": {}},
    }

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test(size=(130, 38)) as pilot:
            app.push_screen(StudyWorkspace(work, services))
            await pilot.pause()
            await pilot.click("#study-retry")
            await pilot.pause()
            assert "missing" in str(app.screen.query_one("#study-action-status").render())
            assert not app.screen.query_one("#study-retry", Button).disabled

    asyncio.run(exercise())
