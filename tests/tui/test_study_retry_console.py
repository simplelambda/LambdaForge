"""Console Study recovery calls domain services without remote computation."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("textual")
from textual.widgets import Button, Checkbox, TabbedContent  # noqa: E402

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


def test_detected_code_change_requires_explicit_acknowledgement_before_submit() -> None:
    async def exercise() -> None:
        app = LambdaForgeApp(FakeServices())
        async with app.run_test(size=(130, 42)) as pilot:
            app.push_screen(StudyRetryConfirmation({"code_change_detected": True}))
            await pilot.pause()
            assert app.screen.query_one("#retry-confirm", Button).disabled
            assert app.screen.query_one("#retry-code-change-warning")
            app.screen.query_one("#retry-compatible-code", Checkbox).value = True
            await pilot.pause()
            assert not app.screen.query_one("#retry-confirm", Button).disabled
            app.screen.query_one("#retry-compatible-code", Checkbox).value = False
            await pilot.pause()
            assert app.screen.query_one("#retry-confirm", Button).disabled

    asyncio.run(exercise())


def test_recovery_logs_and_next_retry_target_latest_job_without_losing_telemetry() -> None:
    class Services(FakeServices):
        def study_workspace(self, job_id, *, latest_job_id):
            self.calls.append(("study_workspace", (job_id, latest_job_id)))
            return {
                "candidates": [],
                "counts": {},
                "job_state": "failed",
                "telemetry_job_id": "job-original",
            }

        def work_logs(self, job_id, *, tail=2_000):
            self.calls.append(("work_logs", job_id))
            return {
                "text": "Study recovery requires unchanged configuration, inputs and seed policy."
            }

        def retry_preview(self, job_id):
            self.calls.append(("retry_preview", job_id))
            return {
                "resumable": True,
                "strategy": "sweep",
                "reuse_runs": 43,
                "retry_runs": 0,
                "pending_runs": 33,
            }

    services = Services()
    work = {
        "name": "Study",
        "state": "running",
        "primary_job_id": "job-failed-recovery",
        "study_job_id": "job-original",
        "study": {"candidates": [], "counts": {}},
    }

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test(size=(130, 38)) as pilot:
            app.push_screen(StudyWorkspace(work, services))
            await pilot.pause()
            screen = app.screen
            screen._refresh_study(force=True)
            await pilot.pause()
            assert screen.work["state"] == "failed"
            assert screen.job_id == "job-original"
            assert screen.operation_job_id == "job-failed-recovery"
            screen.query_one("#study-tabs", TabbedContent).active = "study-logs"
            await pilot.pause()
            assert ("work_logs", "job-failed-recovery") in services.calls
            assert ("work_logs", "job-original") not in services.calls
            await pilot.click("#study-retry")
            await pilot.pause()
            assert ("retry_preview", "job-failed-recovery") in services.calls
            assert isinstance(app.screen, StudyRetryConfirmation)

    asyncio.run(exercise())
