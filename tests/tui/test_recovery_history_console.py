"""Recovery errors/history remain actionable across UI closure and provider outage."""

from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Event
from typing import Any

import pytest

pytest.importorskip("textual")
from textual.app import App  # noqa: E402
from textual.widgets import Button, DataTable, Input  # noqa: E402

from lambdaforge.tui.App import LambdaForgeApp  # noqa: E402
from lambdaforge.tui.screens.RecoveryWorkspace import (  # noqa: E402
    RecoveryFailureDialog,
    RecoveryWorkspace,
)
from lambdaforge.tui.screens.Workspace import StudyRetryConfirmation, StudyWorkspace  # noqa: E402
from tests.controlplane.test_study_retry import recovery_jobs as recovery_fixture  # noqa: E402
from tests.tui.test_research_console_014 import FakeServices  # noqa: E402


def test_saved_audit_pages_work_without_provider_and_survive_console_reopen(tmp_path: Path) -> None:
    jobs, previous, _ = recovery_fixture.__wrapped__(tmp_path)
    for index in range(105):
        jobs.store.append_event(
            previous.job_id, state="failed", phase="RECOVERY_BLOCKED", message=f"Rejected {index}"
        )

    class Services:
        def recovery_history(self, job_id: str, **pagination: Any) -> Any:
            return jobs.recovery_history(job_id, **pagination)

    async def exercise() -> None:
        for reopening in (False, True):
            app = App()
            async with app.run_test(size=(140, 45)) as pilot:
                app.push_screen(RecoveryWorkspace(Services(), previous.job_id))
                for _ in range(60):
                    await pilot.pause(0.05)
                    if not app.screen._busy:
                        break
                screen = app.screen
                assert screen.query_one("#recovery-events", DataTable).row_count == 100
                assert "Rejected 104" in str(screen.query_one("#recovery-detail").render())
                if not reopening:
                    await pilot.click("#recovery-older")
                    await pilot.pause()
                    assert screen.query_one("#recovery-events", DataTable).row_count == 5
                    assert not screen.query_one("#recovery-newer", Button).disabled
                    assert screen.query_one("#recovery-older", Button).disabled

    asyncio.run(exercise())


def test_audit_loading_does_not_block_navigation_or_write_after_unmount() -> None:
    entered, released = Event(), Event()

    class Services:
        def recovery_history(self, *args: Any, **kwargs: Any) -> Any:
            entered.set()
            assert released.wait(10)
            raise TimeoutError("simulated slow read")

    async def exercise() -> None:
        app = App()
        async with app.run_test(size=(110, 35)) as pilot:
            original = app.screen
            app.push_screen(RecoveryWorkspace(Services(), "job-a"))
            for _ in range(50):
                await pilot.pause(0.02)
                if entered.is_set():
                    break
            assert entered.is_set()
            assert "Reading saved audit" in str(app.screen.query_one("#recovery-status").render())
            await pilot.press("escape")
            assert app.screen is original
            released.set()
            await pilot.pause()
        released.set()

    try:
        asyncio.run(exercise())
    finally:
        released.set()


def test_apply_failure_has_persistent_dialog_and_returns_to_same_plan() -> None:
    class Services(FakeServices):
        def retry_preview(self, job_id: str) -> Any:
            return {"resumable": True, "reuse_runs": 9, "retry_runs": 1, "pending_runs": 0}

        def retry_job(self, *args: Any, **kwargs: Any) -> Any:
            error = ValueError("Ownership became ambiguous after preview")
            error.recovery_request_id = "recovery-blocked"
            error.recovery_job_id = "job-original"
            raise error

    async def exercise() -> None:
        services = Services()
        app = LambdaForgeApp(services)
        async with app.run_test(size=(140, 45)) as pilot:
            workspace = StudyWorkspace(
                {
                    "name": "Study",
                    "state": "failed",
                    "primary_job_id": "job-original",
                    "study": {"candidates": [], "counts": {}},
                },
                services,
            )
            app.push_screen(workspace)
            await pilot.pause()
            await pilot.click("#study-retry")
            await pilot.pause()
            assert isinstance(app.screen, StudyRetryConfirmation)
            await pilot.click("#retry-confirm")
            await pilot.pause()
            assert isinstance(app.screen, RecoveryFailureDialog)
            detail = str(app.screen.query_one("#recovery-failure-detail").render())
            assert "recovery-blocked" in detail and "Phase: Apply" in detail
            assert "ambiguous" in detail
            await pilot.click("#recovery-failure-plan")
            await pilot.pause()
            assert isinstance(app.screen, StudyRetryConfirmation)
            assert app.screen.preview["reuse_runs"] == 9
            await pilot.click("#retry-cancel")
            await pilot.pause()
            assert app.screen is workspace
            assert not workspace.query_one("#study-retry", Button).disabled

    asyncio.run(exercise())


def test_configuration_rejection_keeps_selected_yaml_and_audited_failure_visible() -> None:
    class Services:
        def retry_preview(self, job_id: str, *, config: str) -> Any:
            error = ValueError("Changed seed policy; password=private-value")
            error.recovery_request_id = "recovery-preview-rejected"
            error.recovery_job_id = job_id
            raise error

    async def exercise() -> None:
        app = LambdaForgeApp(FakeServices())
        async with app.run_test(size=(140, 45)) as pilot:
            dialog = StudyRetryConfirmation(
                {"configuration": {"selected_path": "moved.yaml", "selected_exists": True}},
                services=Services(),
                job_id="job-original",
            )
            app.push_screen(dialog)
            await pilot.pause()
            await pilot.click("#retry-config-validate")
            for _ in range(60):
                await pilot.pause(0.02)
                if isinstance(app.screen, RecoveryFailureDialog):
                    break
            assert isinstance(app.screen, RecoveryFailureDialog)
            detail = str(app.screen.query_one("#recovery-failure-detail").render())
            assert "Configuration validation" in detail and "recovery-preview-rejected" in detail
            assert "private-value" not in detail
            await pilot.click("#recovery-failure-plan")
            await pilot.pause()
            assert app.screen is dialog
            assert dialog.query_one("#retry-config", Input).value == "moved.yaml"
            assert not dialog.query_one("#retry-confirm", Button).disabled
            await pilot.click("#retry-confirm")
            await pilot.pause()
            assert app.screen is dialog
            assert "Not submitted" in str(dialog.query_one("#retry-submit-status").render())
            assert not dialog.query_one("#retry-config-validate", Button).disabled

    asyncio.run(exercise())
