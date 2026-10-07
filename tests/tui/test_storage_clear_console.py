"""Clear storage is preview/confirm/apply and uses the shared domain service."""

import asyncio

import pytest

pytest.importorskip("textual")

from textual.widgets import Button  # noqa: E402

from lambdaforge.tui.App import ClusterEditor, LambdaForgeApp  # noqa: E402
from lambdaforge.tui.screens.Workspace import ClusterWorkspace, ExactConfirmation  # noqa: E402
from tests.tui.test_research_console_014 import FakeServices  # noqa: E402


def test_clear_requires_confirmation_and_reports_completion() -> None:
    class Services(FakeServices):
        def clean_storage(self, name, *, apply=False):
            self.calls.append(("clean_storage", (name, apply)))
            return {"reclaimable_bytes": 100, "applied": apply, "candidates": []}

    services = Services()

    async def exercise():
        app = LambdaForgeApp(services)
        async with app.run_test(size=(150, 45)) as pilot:
            app.push_screen(ClusterWorkspace("gpu12", services))
            await pilot.pause()
            await pilot.click("#cluster-storage-clear")
            await pilot.pause()
            assert isinstance(app.screen, ExactConfirmation)
            assert services.calls.count(("clean_storage", ("gpu12", False))) == 1
            assert ("clean_storage", ("gpu12", True)) not in services.calls
            # Confirmation ids belong to the common mutation dialog.
            confirm = app.screen.query_one("#exact-apply", Button)
            await pilot.click(confirm)
            await pilot.pause()
            assert services.calls.count(("clean_storage", ("gpu12", True))) == 1
            assert isinstance(app.screen, ClusterWorkspace)

    asyncio.run(exercise())


def test_editor_roundtrips_storage_safety_and_retention() -> None:
    profile = {
        "name": "gpu12",
        "host": "example.invalid",
        "workspace": "/owned/workspace",
        "storage": {
            "lease_root": "/owned/host-leases",
            "safety": {"min_free": 123, "min_free_percent": 12},
            "terminal_jobs": {"grace_period": 600},
        },
    }

    async def exercise():
        app = LambdaForgeApp(FakeServices())
        async with app.run_test(size=(140, 45)) as pilot:
            app.push_screen(ClusterEditor(app.services, profile))
            await pilot.pause()
            name, descriptor = app.screen._descriptor()
            from lambdaforge.controlplane.ClusterProfile import ClusterProfile

            storage = ClusterProfile.from_mapping(name, descriptor).storage
            assert storage.lease_root == "/owned/host-leases"
            assert storage.safety_min_free_bytes == 123
            assert storage.safety_min_free_percent == 12
            assert storage.terminal_grace_seconds == 600

    asyncio.run(exercise())
