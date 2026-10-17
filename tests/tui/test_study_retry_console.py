"""Console Study recovery calls domain services without remote computation."""

from __future__ import annotations

import asyncio
from threading import Event

import pytest

pytest.importorskip("textual")
from textual.widgets import Button, Checkbox, DataTable, Input, Static, TabbedContent  # noqa: E402

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
        from lambdaforge.tui.screens.RecoveryWorkspace import RecoveryFailureDialog

        app = LambdaForgeApp(services)
        async with app.run_test(size=(130, 38)) as pilot:
            workspace = StudyWorkspace(work, services)
            app.push_screen(workspace)
            await pilot.pause()
            await pilot.click("#study-retry")
            await pilot.pause()
            assert isinstance(app.screen, RecoveryFailureDialog)
            assert "missing" in str(app.screen.query_one("#recovery-failure-detail").render())
            await pilot.click("#recovery-failure-close")
            await pilot.pause()
            assert app.screen is workspace
            assert "missing" in str(workspace.query_one("#study-action-status").render())
            assert not workspace.query_one("#study-retry", Button).disabled

    asyncio.run(exercise())


def test_detected_code_change_requires_explicit_acknowledgement_before_submit() -> None:
    async def exercise() -> None:
        app = LambdaForgeApp(FakeServices())
        async with app.run_test(size=(130, 42)) as pilot:
            app.push_screen(StudyRetryConfirmation({"code_change_detected": True}))
            await pilot.pause()
            assert not app.screen.query_one("#retry-confirm", Button).disabled
            assert app.screen.query_one("#retry-code-change-warning")
            await pilot.click("#retry-confirm")
            await pilot.pause()
            assert isinstance(app.screen, StudyRetryConfirmation)
            assert "Not submitted" in str(app.screen.query_one("#retry-submit-status").render())
            app.screen.query_one("#retry-compatible-code", Checkbox).value = True
            await pilot.pause()
            assert not app.screen.query_one("#retry-confirm", Button).disabled
            app.screen.query_one("#retry-compatible-code", Checkbox).value = False
            await pilot.pause()
            assert not app.screen.query_one("#retry-confirm", Button).disabled
            await pilot.click("#retry-confirm")
            await pilot.pause()
            assert isinstance(app.screen, StudyRetryConfirmation)
            assert "Not submitted" in str(app.screen.query_one("#retry-submit-status").render())

    asyncio.run(exercise())


@pytest.mark.parametrize("size", [(120, 35), (80, 24)])
def test_long_realistic_plan_keeps_acknowledgement_and_blocking_reason_visible(size) -> None:
    class Services(FakeServices):
        def retry_preview(self, job_id):
            return {
                "resumable": True,
                "strategy": "sweep",
                "code_change_detected": True,
                "reuse_runs": 47,
                "retry_runs": 1,
                "pending_runs": 0,
                "configuration": {"selected_path": "v6b.yaml", "selected_exists": True},
                "runs": [
                    {"trial": i // 4 + 1, "seed": i % 4, "action": "reuse"} for i in range(48)
                ],
            }

        def retry_job(self, job_id, *, accept_code_change=False):
            self.calls.append(("retry_job", (job_id, accept_code_change)))
            return {"job_id": "job-recovered", "state": "preparing"}

        def study(self, job_id):
            return {"candidates": [], "counts": {}, "job_state": "preparing"}

    async def exercise() -> None:
        services = Services()
        app = LambdaForgeApp(services)
        async with app.run_test(size=size) as pilot:
            workspace = StudyWorkspace(
                {
                    "name": "v6b",
                    "state": "failed",
                    "primary_job_id": "job-original",
                    "study": {"candidates": [], "counts": {}},
                },
                services,
            )
            app.push_screen(workspace)
            await pilot.pause()
            workspace._confirm_retry(services.retry_preview("job-original"))
            await pilot.pause()
            dialog = app.screen
            for selector in ("#retry-compatible-code", "#retry-submit-status", "#retry-confirm"):
                widget = dialog.query_one(selector)
                assert widget.region.y >= 0 and widget.region.bottom <= size[1]
                assert widget.region.x >= 0 and widget.region.right <= size[0]
            assert "Action required" in str(dialog.query_one("#retry-submit-status").render())
            await pilot.click("#retry-confirm")
            await pilot.pause()
            assert app.screen is dialog
            assert not any(name == "retry_job" for name, _ in services.calls)
            assert "Not submitted" in str(dialog.query_one("#retry-submit-status").render())
            await pilot.click("#retry-compatible-code")
            await pilot.pause()
            assert dialog.query_one("#retry-compatible-code", Checkbox).value
            assert "Ready" in str(dialog.query_one("#retry-submit-status").render())
            await pilot.click("#retry-confirm")
            await pilot.pause()
            assert app.screen is workspace
            assert services.calls.count(("retry_job", ("job-original", True))) == 1
            assert workspace.operation_job_id == "job-recovered"

    asyncio.run(exercise())


def test_pending_configuration_validation_cannot_be_bypassed_by_acknowledgement() -> None:
    entered, release = Event(), Event()
    preview = {
        "configuration": {"selected_path": "old.yaml", "selected_exists": True},
        "code_change_detected": True,
    }

    class Services:
        def retry_preview(self, job_id, *, config):
            entered.set()
            assert release.wait(5)
            return {
                "configuration": {"selected_path": config, "sha256": "verified"},
                "code_change_detected": True,
            }

    async def exercise() -> None:
        app = LambdaForgeApp(FakeServices())
        async with app.run_test(size=(120, 35)) as pilot:
            dialog = StudyRetryConfirmation(preview, services=Services(), job_id="job-original")
            app.push_screen(dialog)
            await pilot.pause()
            dialog.query_one("#retry-config", Input).value = "moved.yaml"
            await pilot.pause()
            await pilot.click("#retry-confirm")
            await pilot.pause()
            assert "Not submitted" in str(dialog.query_one("#retry-submit-status").render())
            await pilot.click("#retry-config-validate")
            await pilot.pause()
            assert entered.is_set()
            assert "Validating" in str(dialog.query_one("#retry-submit-status").render())
            await pilot.click("#retry-compatible-code")
            await pilot.pause()
            assert dialog.query_one("#retry-confirm", Button).disabled
            assert app.screen is dialog
            release.set()
            for _ in range(60):
                await pilot.pause(0.02)
                if not dialog._validating:
                    break
            assert not dialog._validating
            assert not dialog.query_one("#retry-confirm", Button).disabled
            assert "Ready" in str(dialog.query_one("#retry-submit-status", Static).render())

    try:
        asyncio.run(exercise())
    finally:
        release.set()


def test_checkpoint_preview_is_visible_without_claiming_epoch_continuation() -> None:
    preview = {
        "reuse_runs": 9,
        "retry_runs": 1,
        "pending_runs": 0,
        "runs": [
            {
                "trial": 1,
                "seed": 54,
                "action": "retry",
                "run_id": "run-exact",
                "checkpoint": {
                    "last": {"step": 9, "written_at_utc": "2026-10-10", "integrity": "not_checked"}
                },
            }
        ],
    }

    async def exercise() -> None:
        app = LambdaForgeApp(FakeServices())
        async with app.run_test(size=(130, 42)) as pilot:
            app.push_screen(StudyRetryConfirmation(preview))
            await pilot.pause()
            table = app.screen.query_one("#retry-checkpoints", DataTable)
            assert table.row_count == 1
            assert table.get_row_at(0) == [
                "run-exact",
                "2026-10-10",
                "9",
                "not_checked",
                "Requires Work support",
            ]
            assert not app.screen.query_one("#retry-confirm", Button).disabled

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
