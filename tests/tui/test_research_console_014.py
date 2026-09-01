from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

textual = pytest.importorskip("textual")

from lambdaforge.cli.CommandLineInterface import CommandLineInterface  # noqa: E402
from lambdaforge.tui.App import ConfirmationDialog, LambdaForgeApp  # noqa: E402
from lambdaforge.tui.models import CLI_PARITY, CONSOLE_ACTIONS  # noqa: E402


class FakeServices:
    def __init__(self, snapshot=None, *, fail: bool = False):
        self.snapshot = snapshot or {"work": {"items": []}, "clusters": []}
        self.fail = fail

    def overview_snapshot(self):
        if self.fail:
            raise RuntimeError("provider temporarily unreachable")
        return self.snapshot

    def cluster_rows(self):
        return []

    def dataset_rows(self):
        return []

    def result_rows(self):
        return []

    def analyze(self, selector, *, recompute=False):
        return {"source": {"status": "final"}, "selector": selector}

    def report(self, selector, output: Path):
        return output


def test_console_starts_and_navigates() -> None:
    async def exercise() -> None:
        app = LambdaForgeApp(FakeServices())
        async with app.run_test(size=(100, 32)) as pilot:
            assert app.query_one("#overview")
            app.show_screen("studies")
            await pilot.pause(0.05)
            assert app.query_one("#studies").display is True
            assert app.query_one("#overview").display is False
            app.show_screen("results")
            await pilot.pause(0.05)
            assert app.query_one("#results").display is True
            app.exit()

    asyncio.run(exercise())


def test_cli_parity_inventory_covers_every_public_family() -> None:
    assert set(CLI_PARITY) == {
        "project/work",
        "system",
        "cluster",
        "job",
        "dataset",
        "result",
    }
    assert {"analyze", "report"} <= set(CLI_PARITY["result"])
    assert {"add", "bootstrap", "credentials delete"} <= set(CLI_PARITY["cluster"])


def test_pruned_trial_detail_marks_partial_evidence_as_censored() -> None:
    snapshot = {
        "work": {
            "items": [
                {
                    "name": "study",
                    "state": "running",
                    "study": {
                        "objective": {"metric": "score"},
                        "counts": {"candidates": 1, "pruned_runs": 1},
                        "candidates": [
                            {
                                "trial": 4,
                                "state": "pruned",
                                "best_objective": 0.42,
                                "partially_censored": True,
                            }
                        ],
                    },
                }
            ]
        },
        "clusters": [],
    }

    async def exercise() -> None:
        app = LambdaForgeApp(FakeServices(snapshot))
        async with app.run_test(size=(100, 32)) as pilot:
            app.show_screen("studies")
            await pilot.pause(0.1)
            screen = app.query_one("#studies")
            assert "partial/censored" in screen.detail(0)
            assert "0.42†" in screen.detail(0)

    asyncio.run(exercise())


def test_narrow_layout_and_worker_failure_keep_console_alive() -> None:
    async def exercise() -> None:
        app = LambdaForgeApp(FakeServices(fail=True))
        async with app.run_test(size=(70, 24)) as pilot:
            await pilot.pause(0.1)
            assert app.screen.has_class("narrow")
            assert "Temporarily unavailable" in str(
                app.query_one("#overview #screen-status").render()
            )
            app.show_screen("clusters")
            await pilot.pause(0.05)
            assert app.query_one("#clusters").display

    asyncio.run(exercise())


def test_destructive_palette_action_requires_confirmation() -> None:
    action = next(
        value
        for value in CONSOLE_ACTIONS
        if value.family == "result" and value.operation == "compare"
    )
    destructive = next(
        value
        for value in CONSOLE_ACTIONS
        if value.family == "dataset" and value.operation == "delete"
    )
    assert action.destructive is False

    async def exercise() -> None:
        app = LambdaForgeApp(FakeServices())
        async with app.run_test() as pilot:
            app.open_console_action(destructive)
            await pilot.pause(0.05)
            assert isinstance(app.screen, ConfirmationDialog)
            app.pop_screen()

    asyncio.run(exercise())


def test_no_command_dispatch_uses_console_only_for_a_tty(monkeypatch, capsys) -> None:
    calls: list[str] = []
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    stdout = SimpleNamespace(isatty=lambda: True, write=lambda value: None)
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("lambdaforge.tui.App.run_console", lambda: calls.append("console") or 0)
    assert CommandLineInterface._dispatch((), json_output=False) == 0
    assert calls == ["console"]

    monkeypatch.undo()
    assert CommandLineInterface._dispatch((), json_output=False) == 0
    assert "usage: lf" in capsys.readouterr().out
