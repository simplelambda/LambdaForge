"""Review/apply feedback and duplicate prevention without launching scientific work."""

import asyncio
from threading import Event

import pytest

pytest.importorskip("textual")
from textual.app import App  # noqa: E402
from textual.widgets import Button, Input, Static  # noqa: E402

from lambdaforge.tui.screens.StudyExtension import StudyExtensionDialog  # noqa: E402


@pytest.mark.parametrize("revisioned", [False, True])
def test_extension_dialog_uses_original_yaml_invalidates_review_and_submits_once(
    tmp_path, revisioned
):
    async def exercise():
        submitted = []
        started, finish = Event(), Event()
        yaml = str(tmp_path / "original.yaml")

        class Services:
            def extension_context(self, work):
                return {"study": "execution-original", "config": yaml, "target": "local"}

            def extension_preview(self, context, path):
                return {
                    "plan_hash": "sha256:reviewed",
                    "apply_supported": True,
                    "code_compatibility": {
                        "kind": "revisioned-current" if revisioned else "exact",
                        "warning": "New code/runtime cohort" if revisioned else "",
                    },
                    "comparison": {
                        "counts": {"reused": 20, "new": 15, "total": 35},
                        "added_values": {"strength": [0.5, 0.7, 1.0]},
                        "added_seeds": [],
                    },
                }

            def extend_study(self, context, path, *, plan_hash):
                submitted.append((context, path, plan_hash))
                started.set()
                assert finish.wait(15)
                return {"state": "submitted", "job_id": "job-delta"}

        results = []

        class Console(App):
            def on_mount(self):
                self.push_screen(
                    StudyExtensionDialog({"name": "regularization"}, Services()), results.append
                )

        app = Console()
        try:
            async with app.run_test(size=(110, 40)) as pilot:

                async def wait_for(predicate):
                    for _ in range(100):
                        await pilot.pause(0.02)
                        if predicate():
                            return
                    raise AssertionError("Timed out waiting for asynchronous extension feedback")

                await wait_for(lambda: getattr(app.screen, "context", None) is not None)
                dialog = app.screen
                assert dialog.query_one("#extension-path", Input).value == yaml
                assert dialog.query_one("#extension-apply", Button).disabled
                await pilot.click("#extension-review")
                await wait_for(lambda: dialog.plan is not None)
                assert "KEEP 20" in str(dialog.query_one("#extension-plan", Static).render())
                assert not dialog.query_one("#extension-apply", Button).disabled
                if revisioned:
                    assert str(dialog.query_one("#extension-apply", Button).label) == (
                        "Extend · current code"
                    )
                    assert "New code/runtime cohort" in str(
                        dialog.query_one("#extension-plan", Static).render()
                    )
                dialog.query_one("#extension-path", Input).value = yaml + ".changed"
                await pilot.pause()
                assert dialog.query_one("#extension-apply", Button).disabled
                await pilot.click("#extension-review")
                await wait_for(lambda: dialog.plan is not None)
                await pilot.click("#extension-apply")
                await wait_for(started.is_set)
                assert started.is_set() and len(submitted) == 1
                assert dialog.query_one("#extension-path", Input).disabled
                assert dialog.query_one("#extension-apply", Button).disabled
                await pilot.press("escape")
                await pilot.pause()
                assert app.screen is dialog and "in progress" in str(
                    dialog.query_one("#extension-status", Static).render()
                )
                finish.set()
                await wait_for(lambda: bool(results))
                assert results == [{"state": "submitted", "job_id": "job-delta"}]
                assert len(submitted) == 1
        finally:
            finish.set()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "failure", [ValueError("Remote attestation unavailable"), SystemExit("consumer import stopped")]
)
def test_unsupported_placement_has_persistent_visible_feedback_and_back(failure):
    async def exercise():
        class Services:
            def extension_context(self, work):
                raise failure

        results = []

        class Console(App):
            def on_mount(self):
                self.push_screen(
                    StudyExtensionDialog({"cluster": "remote"}, Services()), results.append
                )

        app = Console()
        async with app.run_test(size=(100, 35)) as pilot:
            await pilot.pause()
            dialog = app.screen
            assert str(failure) in str(dialog.query_one("#extension-status", Static).render())
            assert not dialog.query_one("#extension-loading").display
            assert dialog.query_one("#extension-apply", Button).disabled
            await pilot.click("#extension-cancel")
            await pilot.pause()
            assert results == [None]

    asyncio.run(exercise())
