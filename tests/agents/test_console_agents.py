"""Headless console configuration uses services and explicit local confirmation."""

import asyncio
from pathlib import Path

from textual.app import App
from textual.widgets import DataTable, Input, Static

from lambdaforge.tui.screens.AgentWorkspace import AgentWorkspace


def test_agent_workspace_displays_status_skills_and_preview_without_writing(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="console"\nversion="0.1"\n')

    class ExampleApp(App):
        def on_mount(self):
            self.push_screen(AgentWorkspace(tmp_path))

    async def scenario():
        app = ExampleApp()
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert screen.query_one("#agents-table", DataTable).row_count == 24
            assert "READ / PLAN" in str(screen.query_one("#agents-status", Static).render())
            screen._confirm(screen.integration.preview())
            await pilot.pause()
            assert not (tmp_path / "AGENTS.md").exists()
            app.pop_screen()
            await pilot.pause()

    asyncio.run(scenario())


def test_console_human_review_requires_exact_id_and_never_applies_itself(tmp_path: Path):
    from lambdaforge.agents.approvals import ApprovalService
    from lambdaforge.agents.knowledge import KnowledgeService
    from lambdaforge.agents.scope import ProjectScope
    from lambdaforge.tui.screens.AgentApprovals import AgentApprovals

    (tmp_path / "pyproject.toml").write_text('[project]\nname="review"\nversion="0.1"\n')
    scope = ProjectScope(tmp_path)
    knowledge = KnowledgeService(scope)
    knowledge.initialize()
    approvals = ApprovalService(scope)
    proposal = approvals.request(
        "knowledge.create",
        {
            "kind": "finding",
            "title": "Review",
            "body": "Exact body shown to the operator.",
        },
    )

    class ExampleApp(App):
        def on_mount(self):
            self.push_screen(AgentApprovals(approvals))

    async def scenario():
        app = ExampleApp()
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            screen = app.screen
            table = screen.query_one("#approval-table", DataTable)
            assert table.row_count == 1
            table.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert approvals.inspect(proposal["request_id"])["state"] == "pending"
            confirmation = app.screen
            confirmation.query_one("#approval-exact-id", Input).value = proposal["request_id"]
            await pilot.pause()
            # Scroll the reviewing modal down; the exact confirmation remains mouse reachable.
            from textual.containers import VerticalScroll

            confirmation.query_one(VerticalScroll).scroll_end(animate=False)
            await pilot.pause()
            await pilot.click("#approval-confirm")
            await pilot.pause()
            assert approvals.inspect(proposal["request_id"])["state"] == "approved"
            assert not knowledge.search()["items"]  # approval is not execution

    asyncio.run(scenario())
