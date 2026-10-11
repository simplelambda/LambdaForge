"""The independent observer's workbench calls native services and exposes every outcome."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("textual")
from textual.app import App  # noqa: E402
from textual.widgets import Button, Input, Static  # noqa: E402

from lambdaforge.tui.screens.OperationalWorkspace import OperationalWorkspace  # noqa: E402


class Services:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def operational_snapshot(self, *, refresh: bool = False) -> dict[str, Any]:
        self.calls.append(("snapshot", refresh))
        return {
            "monitor": {"health": "stopped", "process_lifetime": "dead"},
            "settings": {"interval_seconds": 60, "smtp": {"enabled": False}},
            "jobs": [],
            "incidents": [],
        }

    def monitor_action(self, operation: str, settings: Any = None) -> Any:
        self.calls.append((operation, settings))
        if operation == "test-alert":
            raise ValueError("SMTP is disabled; configure an explicit channel first.")
        return {}


def test_operational_console_is_read_only_until_action_and_shows_failed_test() -> None:
    async def exercise() -> None:
        services = Services()
        app = App()
        async with app.run_test(size=(130, 45)) as pilot:
            app.push_screen(OperationalWorkspace(services))
            await pilot.pause()
            for _ in range(40):
                if not app.screen._busy:
                    break
                await pilot.pause(0.05)
            assert services.calls == [("snapshot", False), ("snapshot", False)]
            await pilot.click("#operations-refresh")
            await pilot.pause()
            assert ("snapshot", True) in services.calls
            screen = app.screen
            # Invoke the same event as a visible button; avoids viewport-dependent tab coordinates.
            screen.on_button_pressed(Button.Pressed(screen.query_one("#operations-test", Button)))
            await pilot.pause()
            assert "SMTP is disabled" in str(
                screen.query_one("#operations-status", Static).render()
            )
            assert not screen.query_one("#operations-refresh", Button).disabled
            screen.query_one("#operations-unreachable", Input).value = "3"
            screen.query_one("#operations-smtp-cooldown_seconds", Input).value = "1200"
            screen.on_button_pressed(Button.Pressed(screen.query_one("#operations-save", Button)))
            await pilot.pause()
            saved = next(value for name, value in services.calls if name == "config")
            assert saved["observation"]["grace_seconds"] == 120
            assert saved["observation"]["progress_seconds"] is None
            assert saved["observation"]["unreachable_failures"] == 3
            assert saved["smtp"]["cooldown_seconds"] == 1200
            assert saved["smtp"]["enabled"] is False
            assert screen.query_one("#operations-smtp-credential", Input).value == ""

    asyncio.run(exercise())
