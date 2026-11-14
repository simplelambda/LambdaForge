"""Work recovery uses the same reviewed native plan, never an unconfirmed retry click."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("textual")
from textual.widgets import Button, Checkbox  # noqa: E402

from lambdaforge.tui.App import LambdaForgeApp  # noqa: E402
from lambdaforge.tui.screens.Workspace import StudyRetryConfirmation, WorkWorkspace  # noqa: E402
from tests.tui.test_research_console_014 import FakeServices  # noqa: E402


@pytest.mark.parametrize("code_change", [False, True])
@pytest.mark.parametrize("strategy", ["single", "sequential"])
def test_single_work_retry_requires_review_has_feedback_and_pins_yaml(code_change, strategy):
    calls = []

    class Services(FakeServices):
        def work_logs(self, *_args):
            return {"text": "persisted failure log"}

        def retry_preview(self, job_id, **options):
            return {
                "strategy": strategy,
                "resumable": True,
                "reuse_runs": 0,
                "retry_runs": 1,
                "pending_runs": 0,
                "runs": (
                    [{"step": "discover", "action": "retry", "run_id": "run-fixture"}]
                    if strategy == "sequential"
                    else []
                ),
                "configuration": {
                    "selected_path": "reviewed.yaml",
                    "selected_exists": True,
                    "sha256": "a" * 64,
                },
                "code_change_detected": code_change,
                "effective_resources": {"ram_bytes": 96 * 1024**3},
            }

        def retry_job(self, job_id, **options):
            calls.append((job_id, options))
            return {"job_id": "job-recovered", "state": "preparing"}

    async def exercise():
        services = Services()
        app = LambdaForgeApp(services)
        async with app.run_test(size=(150, 50)) as pilot:
            workspace = WorkWorkspace(
                {
                    "name": "preprocess",
                    "state": "failed",
                    "primary_job_id": "job-original",
                    "cluster": "fixture",
                },
                services,
            )
            app.push_screen(workspace)
            await pilot.pause()
            await pilot.click("#work-retry")
            for _ in range(40):
                await pilot.pause(0.05)
                if isinstance(app.screen, StudyRetryConfirmation):
                    break
            dialog = app.screen
            assert isinstance(dialog, StudyRetryConfirmation)
            assert not calls
            if code_change:
                dialog.on_button_pressed(Button.Pressed(dialog.query_one("#retry-confirm", Button)))
                assert not calls
                dialog.query_one("#retry-compatible-code", Checkbox).value = True
                await pilot.pause()
            dialog.on_button_pressed(Button.Pressed(dialog.query_one("#retry-confirm", Button)))
            await pilot.pause()
            assert calls == [
                (
                    "job-original",
                    {
                        "config": "reviewed.yaml",
                        "config_sha256": "a" * 64,
                        "accept_code_change": code_change,
                    },
                )
            ]
            assert workspace.query_one("#work-retry", Button).disabled
            assert app.screen is workspace

    asyncio.run(exercise())
